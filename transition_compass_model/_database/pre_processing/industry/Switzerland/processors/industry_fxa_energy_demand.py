import os
import pickle

import numpy as np
from get_data_functions.data_fxa_energy_demand import get_sfoe_sector_data

from transition_compass_model.model.common.auxiliary_functions import create_years_list

# SFOE sectors 1-11 → model tech names
# Sector 12 (Construction) is excluded (not modelled in industry module)
TECH_TO_SECTOR = {
    "fbt-tech": 1,
    "textiles-tech": 2,
    "paper-tech": 3,
    "pulp-tech": 3,
    "wwp-tech": 3,
    "wwp-sec": 3,
    "chem-chem-tech": 4,
    "chem-sec": 4,
    "cement-dry-kiln": 5,
    "cement-wet-kiln": 5,
    "cement-geopolym": 5,
    "cement-sec": 5,
    "lime-lime": 6,  # SFOE s6 "Autres minéraux non ferreux" (not s5 "Ciment et béton")
    "glass-glass": 6,
    "glass-sec": 6,
    "steel-BF-BOF": 7,
    "steel-hisarna": 7,
    "steel-hydrog-DRI": 7,
    "steel-scrap-EAF": 7,
    "aluminium-prim": 8,
    "aluminium-sec": 8,
    "copper-tech": 8,
    "copper-sec": 8,
    "mae-tech": 9,
    "tra-equip-tech": 10,
    "ois-tech": 11,
    "ois-sec": 11,
}

# SFOE data covers 1999-2024; clamp model years to this range
SFOE_YEAR_MIN = 1999
SFOE_YEAR_MAX = 2023  # align with OTS end year; 2024 extrapolated is not used

# Swiss-specific intensity scale factors applied to EU27 base intensities.
# A scale factor is included only where Swiss production processes are physically
# different from the EU27 average (different product mix, different feedstock, or
# genuinely different process grade). Sectors where the model underestimates SFOE
# energy for other reasons are left at the EU27 default (1.0):
#   - S2 textiles, S4 chem, S9 mae, S10 tra-equip: IO-to-tonne conversion
#     underestimates physical production because Swiss products carry very high
#     CHF/kg values; the energy gap is a known model limitation of the IO methodology.
#   - S5 cement: SFOE sector 5 "Zement, Beton" covers clinker production AND
#     concrete manufacturing (precast, ready-mix); the model captures clinker only.
#     Inflating the clinker intensity to match SFOE would be physically incorrect —
#     Swiss kilns (Holcim, HeidelbergMaterials, modern dry-process) are not more
#     energy-intensive per tonne of clinker than the EU27 average.
#   - S6 glass+lime, S8 Al/Cu: ratio ≈ 1.0 after sector mapping corrections; S8
#     has a structural gap (Novelis imports scrap, processes for export — no domestic
#     demand signal) that cannot be fixed by adjusting intensity.
SWISS_INTENSITY_SCALE = {
    # Sector 1 - FBT (scale < 1, physically driven): Swiss food manufacturing is
    # dominated by specialty food and ingredients (Nestlé, Lindt, Givaudan) rather
    # than bulk processing (grain milling, sugar refining, dairy). Bulk processing is
    # more energy-intensive per tonne, so EU27 intensities — calibrated on a mix
    # weighted toward bulk — overstate Swiss energy use per tonne. This is a genuine
    # product-mix difference, not a production-quantity artefact.
    # Derivation: SFOE S1 2022 (4.70 TWh) / (Swiss FBT production ~7 792 kt ×
    # EU27 intensity 1.46 TWh/Mt) ≈ 0.41.
    "fbt-tech": 0.41,
    # Sector 3 - Paper/Pulp/WWP (scale < 1, physically driven): Switzerland has very
    # high paper recycling rates. Recycled-fibre production genuinely requires ~30–40%
    # less energy per tonne than virgin-pulp production because the energy-intensive
    # wood pulping stage is avoided. This is a real process difference.
    "paper-tech": 0.67,
    "pulp-tech": 0.67,
    "wwp-tech": 0.67,
    "wwp-sec": 0.67,
    # Sector 7 - Steel (scale > 1, physically driven): Swiss Steel Group produces
    # specialty and tool steels (alloyed, stainless, bearing steels) that require more
    # heat treatments, rolling passes and processing steps per tonne than commodity-grade
    # steel, which dominates EU27 production. This is a genuine process difference in
    # product grade, independent of any production-quantity uncertainty.
    "steel-BF-BOF": 1.9,
    "steel-hisarna": 1.9,
    "steel-hydrog-DRI": 1.9,
    "steel-scrap-EAF": 1.9,
    # Sector 11 - OIS (scale < 1, physically driven): Switzerland's other-industry
    # sector is genuinely weighted toward light manufacturing and service-adjacent
    # activities relative to the EU27 average, which includes heavier industry from
    # Central and Eastern Europe. EU27 OIS intensities structurally overestimate Swiss
    # energy use per tonne.
    "ois-tech": 0.7,
    "ois-sec": 0.7,
}


def _build_carrier_shares(sfoe_df, all_carriers):
    """Compute Swiss carrier shares per (sector_num, year).

    Returns dict {(sector_num, year, carrier): share}.
    Shares sum to 1.0 across all_carriers for each (sector, year).
    Carriers absent from SFOE data get share = 0.
    """
    df_pivot = sfoe_df.pivot_table(
        index=["year", "sector_num"],
        columns="carrier",
        values="value_TJ",
        aggfunc="sum",
    ).reset_index()
    df_pivot.columns.name = None

    for c in all_carriers:
        if c not in df_pivot.columns:
            df_pivot[c] = 0.0

    sfoe_carriers = [c for c in all_carriers if c in df_pivot.columns]
    df_pivot[sfoe_carriers] = df_pivot[sfoe_carriers].fillna(0.0)
    df_pivot["_total"] = df_pivot[sfoe_carriers].sum(axis=1)

    shares = {}
    for _, row in df_pivot.iterrows():
        s = int(row["sector_num"])
        y = int(row["year"])
        total = row["_total"]
        for c in sfoe_carriers:
            shares[(s, y, c)] = row[c] / total if total > 0 else 0.0

    return shares


def _apply_swiss_shares(dm_eu27, shares, sfoe_years_arr):
    """Return a Switzerland copy of dm_eu27 with Swiss carrier shares applied.

    Keeps EU27 total TWh/Mt per tech per year; redistributes across carriers
    using Swiss sector-level SFOE shares. Model years outside the SFOE range
    are clamped to SFOE_YEAR_MIN / SFOE_YEAR_MAX.
    """
    dm_ch = dm_eu27.copy()
    dm_ch.rename_col("EU27", "Switzerland", "Country")

    techs = dm_ch.col_labels["Categories1"]
    carriers = dm_ch.col_labels["Categories2"]
    years = dm_ch.col_labels["Years"]

    for y_idx, year in enumerate(years):
        sfoe_yr = int(np.clip(year, SFOE_YEAR_MIN, SFOE_YEAR_MAX))
        # snap to nearest available SFOE year if there is a gap
        if sfoe_yr not in sfoe_years_arr:
            sfoe_yr = int(sfoe_years_arr[np.argmin(np.abs(sfoe_years_arr - sfoe_yr))])

        for t_idx, tech in enumerate(techs):
            sector = TECH_TO_SECTOR.get(tech)
            if sector is None:
                continue  # keep EU27 value unchanged if tech has no sector mapping

            eu27_total = float(np.nansum(dm_eu27.array[0, y_idx, 0, t_idx, :]))

            scale = SWISS_INTENSITY_SCALE.get(tech, 1.0)
            for c_idx, carrier in enumerate(carriers):
                share = shares.get((sector, sfoe_yr, carrier), 0.0)
                dm_ch.array[0, y_idx, 0, t_idx, c_idx] = eu27_total * share * scale

    return dm_ch


def make_fxa_energy_demand(current_file_directory, years_ots, years_fts):
    # Load SFOE sector data
    sfoe_df = get_sfoe_sector_data(current_file_directory)
    sfoe_years_arr = np.array(sorted(sfoe_df["year"].unique().tolist()))

    # Load EU27 FXA from the main industry pickle
    industry_pickle = os.path.join(
        current_file_directory, "../../../../data/datamatrix/industry.pickle"
    )
    with open(industry_pickle, "rb") as f:
        DM_industry_eu = pickle.load(f)

    dm_excl_eu = DM_industry_eu["fxa"]["energy-demand-excl-feedstock"].filter(
        {"Country": ["EU27"]}
    )
    dm_feed_eu = DM_industry_eu["fxa"]["energy-demand-feedstock"].filter(
        {"Country": ["EU27"]}
    )

    # Build carrier shares from SFOE
    all_carriers = dm_excl_eu.col_labels["Categories2"]
    shares = _build_carrier_shares(sfoe_df, all_carriers)

    dm_excl_ch = _apply_swiss_shares(dm_excl_eu, shares, sfoe_years_arr)
    dm_feed_ch = _apply_swiss_shares(dm_feed_eu, shares, sfoe_years_arr)

    # Ammonia: no Swiss production, copy EU27 directly
    ammonia_pickle = os.path.join(
        current_file_directory, "../../../../data/datamatrix/ammonia.pickle"
    )
    with open(ammonia_pickle, "rb") as f:
        DM_ammonia_eu = pickle.load(f)

    dm_excl_amm = DM_ammonia_eu["fxa"]["energy-demand-excl-feedstock"].filter(
        {"Country": ["EU27"]}
    )
    dm_feed_amm = DM_ammonia_eu["fxa"]["energy-demand-feedstock"].filter(
        {"Country": ["EU27"]}
    )
    dm_excl_amm.rename_col("EU27", "Switzerland", "Country")
    dm_feed_amm.rename_col("EU27", "Switzerland", "Country")

    DM_result = {
        "industry": {
            "energy-demand-excl-feedstock": dm_excl_ch,
            "energy-demand-feedstock": dm_feed_ch,
        },
        "ammonia": {
            "energy-demand-excl-feedstock": dm_excl_amm,
            "energy-demand-feedstock": dm_feed_amm,
        },
    }

    lever_file = "fxa_energy-demand.pickle"
    f_path = os.path.join(current_file_directory, "../data/datamatrix/" + lever_file)
    with open(f_path, "wb") as handle:
        pickle.dump(DM_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return DM_result


def run(years_ots, years_fts):
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    DM_result = make_fxa_energy_demand(current_file_directory, years_ots, years_fts)
    return DM_result


if __name__ == "__main__":
    years_ots = create_years_list(1990, 2023, 1)
    years_fts = create_years_list(2025, 2050, 5)
    run(years_ots, years_fts)
