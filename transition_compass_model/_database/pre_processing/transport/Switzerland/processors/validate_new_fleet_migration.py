################################################################################
# ONE-OFF VALIDATION: new-fleet source migration (STAT-TAB -> Swiss Stats SDMX)
#
# Commit 6264389 ("Replace stat tab with swiss stats for new veh") switched the
# new-fleet data source in passenger_fleet.py from the STAT-TAB pixel API to the
# Swiss Stats SDMX API. The two sources use different collection methodologies,
# so their numbers are close but not identical - exact parity is not the right
# bar. This script instead checks the two sources agree within a tolerance,
# replacing the by-eye trend/total comparison that was being done before.
################################################################################

import os
import pickle
import sys

import pandas as pd

from transition_compass_model._database.pre_processing.transport.Switzerland.get_data_functions import (
    passenger_fleet as get_data,
)


# --- Legacy pickle compatibility ---------------------------------------------
# tra_new_fleet.pickle (the old STAT-TAB snapshot) was pickled before the
# `model.` -> `transition_compass_model.model` import rename (commit 3438474),
# so its old module path needs aliasing before it can be unpickled.
def _register_legacy_module_aliases():
    import transition_compass_model.model as new_model
    import transition_compass_model.model.common as new_common
    import transition_compass_model.model.common.auxiliary_functions as new_aux
    import transition_compass_model.model.common.data_matrix_class as new_dmc

    sys.modules.setdefault("model", new_model)
    sys.modules.setdefault("model.common", new_common)
    sys.modules.setdefault("model.common.data_matrix_class", new_dmc)
    sys.modules.setdefault("model.common.auxiliary_functions", new_aux)


def _load_legacy_raw_fleet(file):
    _register_legacy_module_aliases()
    with open(file, "rb") as handle:
        return pickle.load(handle)


# --- Frozen copy of the pre-migration STAT-TAB extraction logic --------------
# This is extract_passenger_new_fleet_by_tech as it was before commit 6264389,
# kept here only so the old raw STAT-TAB pickle can be reprocessed into the same
# harmonized tech categories the current production function produces, for
# comparison. Not used anywhere in production - do not import this elsewhere.
def _extract_passenger_new_fleet_by_tech_stat_tab(dm_new_fleet):
    dm_new_fleet = dm_new_fleet.copy()
    dm_new_fleet.groupby(
        {"tra_passenger_new-vehicles": ".*"}, dim="Variables", regex=True, inplace=True
    )

    main_cat = [cat for cat in dm_new_fleet.col_labels["Categories1"] if ">" in cat]
    passenger_cat = [
        cat for cat in main_cat if "Passenger" in cat or "Motorcycles" in cat
    ]

    dm_pass_new_fleet = dm_new_fleet.filter(
        {"Categories1": passenger_cat}, inplace=False
    )
    dm_pass_new_fleet.groupby(
        {"LDV": ".*Passenger.*"}, dim="Categories1", regex=True, inplace=True
    )
    dm_pass_new_fleet.groupby(
        {"2W": ".*Motorcycles.*"}, dim="Categories1", regex=True, inplace=True
    )

    dict_tech = {
        "FCEV": ["Hydrogen"],
        "BEV": ["Electricity"],
        "ICE-diesel": ["Diesel", "Diesel-electricity: conventional hybrid"],
        "ICE-gasoline": ["Petrol", "Petrol-electricity: conventional hybrid"],
        "PHEV-diesel": ["Diesel-electricity: plug-in hybrid"],
        "PHEV-gasoline": ["Petrol-electricity: plug-in hybrid"],
        "ICE-gas": ["Gas (monovalent and bivalent)"],
    }
    dm_pass_new_fleet.groupby(dict_tech, dim="Categories2", regex=False, inplace=True)
    dm_pass_new_fleet.drop(col_label="Without motor", dim="Categories2")

    dm_tmp = dm_pass_new_fleet.normalise(dim="Categories2", inplace=False)
    dm_tmp.filter({"Categories2": ["Other"]}, inplace=True)
    if (dm_tmp.array > 0.01).any():
        raise ValueError(
            '"Other" category is greater than 1% of the fleet, it cannot be discarded'
        )

    dm_pass_new_fleet.drop(col_label="Other", dim="Categories2")

    return dm_pass_new_fleet


# --- Comparison ----------------------------------------------------------------
def compare_fleet_sources(dm_old, dm_new, tolerance):
    """
    Compare two harmonized new-fleet DataMatrices (same Categories1/2 scheme:
    LDV/2W x BEV/ICE-diesel/ICE-gasoline/FCEV/PHEV-diesel/PHEV-gasoline/ICE-gas)
    cell by cell, on their common Country/Years/Categories1/Categories2, and flag
    cells whose relative difference exceeds `tolerance`.
    """
    common_years = sorted(
        set(dm_old.col_labels["Years"]) & set(dm_new.col_labels["Years"])
    )
    common_countries = sorted(
        set(dm_old.col_labels["Country"]) & set(dm_new.col_labels["Country"])
    )
    common_cat1 = sorted(
        set(dm_old.col_labels["Categories1"]) & set(dm_new.col_labels["Categories1"])
    )
    common_cat2 = sorted(
        set(dm_old.col_labels["Categories2"]) & set(dm_new.col_labels["Categories2"])
    )

    var_old = dm_old.col_labels["Variables"][0]
    var_new = dm_new.col_labels["Variables"][0]
    idx_old = dm_old.idx
    idx_new = dm_new.idx

    rows = []
    for country in common_countries:
        for year in common_years:
            for cat1 in common_cat1:
                for cat2 in common_cat2:
                    old_val = dm_old.array[
                        idx_old[country],
                        idx_old[year],
                        idx_old[var_old],
                        idx_old[cat1],
                        idx_old[cat2],
                    ]
                    new_val = dm_new.array[
                        idx_new[country],
                        idx_new[year],
                        idx_new[var_new],
                        idx_new[cat1],
                        idx_new[cat2],
                    ]
                    # Guard against blow-up when the old value is 0/near-0
                    denom = max(abs(old_val), 1.0)
                    rel_diff = abs(new_val - old_val) / denom
                    rows.append(
                        {
                            "Country": country,
                            "Years": year,
                            "Categories1": cat1,
                            "Categories2": cat2,
                            "old_stat_tab": old_val,
                            "new_swiss_stats": new_val,
                            "abs_diff": new_val - old_val,
                            "rel_diff": rel_diff,
                            "exceeds_tolerance": rel_diff > tolerance,
                        }
                    )

    return pd.DataFrame(rows)


def run(tolerance=0.10):
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # Old STAT-TAB baseline snapshot, reprocessed with the pre-migration logic
    file_old = os.path.join(this_dir, "../data/tra_new_fleet.pickle")
    dm_old_raw = _load_legacy_raw_fleet(file_old)
    dm_old = _extract_passenger_new_fleet_by_tech_stat_tab(dm_old_raw)

    # Current production Swiss Stats source
    agency_new_veh = "CH1.MFZ_IVS"
    dataflow_new_veh = "DF_IVS_0_GENERAL"
    file_new = os.path.join(this_dir, "../data/tra_new_fleet_swiss_stats.pickle")
    dm_new_raw = get_data.get_new_fleet_by_tech_raw(
        agency_new_veh, dataflow_new_veh, file_new
    )
    dm_new, _ = get_data.extract_passenger_new_fleet_by_tech(dm_new_raw)

    df = compare_fleet_sources(dm_old, dm_new, tolerance=tolerance)

    n_total = len(df)
    n_flagged = int(df["exceeds_tolerance"].sum())
    print(f"Compared {n_total} cells, tolerance={tolerance:.0%}")
    print(f"{n_flagged} cells ({n_flagged / n_total:.1%}) exceed tolerance")

    if n_flagged:
        worst = df[df["exceeds_tolerance"]].sort_values("rel_diff", ascending=False)
        print(worst.head(30).to_string(index=False))

    out_file = os.path.join(this_dir, "../data/new_fleet_source_comparison.csv")
    df.to_csv(out_file, index=False)
    print(f"Full comparison written to {out_file}")

    return df


if __name__ == "__main__":
    run()
