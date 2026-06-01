"""
Aviation FTS preprocessing for Switzerland (all 4 ambition levels).
Replaces the lev3/lev4 pickle dependency with direct fleet computation.
"""

import os
import pickle

import numpy as np
import pandas as pd

from transition_compass_model.model.common.auxiliary_functions import create_years_list
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def _compute_new_veh_max(dm_max, years_fts):
    years_fts_all = create_years_list(2025, 2050, 1)
    missing_years = list(set(years_fts_all) - set(dm_max.col_labels["Years"]))
    dm_max.add(np.nan, dim="Years", col_label=missing_years, dummy=True)
    dm_max.sort(dim="Years")
    dm_max.fill_nans("Years")
    dm_max.lag_variable("tra_vehicles-max", shift=1, subfix="_tm1")
    dm_max.operation(
        "tra_vehicles-max",
        "-",
        "tra_vehicles-max_tm1",
        out_col="tra_new-vehicles-max",
        unit="seat",
    )
    dm_max.add(
        0, dim="Variables", col_label="tra_vehicles-waste-max", dummy=True, unit="seat"
    )
    dm_max.drop(col_label="tra_vehicles-max_tm1", dim="Variables")
    dm_max.filter({"Years": years_fts_all}, inplace=True)
    return dm_max


def _compute_tech_new_from_fleet(dm_fleet_ofts, dm_max, years_fts):
    """Derive technology-share_new from total fleet trajectory and max-new-tech constraints."""
    dm_new_veh = dm_fleet_ofts.filter(
        {
            "Variables": [
                "tra_passenger_new-vehicles",
                "tra_passenger_vehicle-fleet",
                "tra_passenger_vehicle-waste",
            ]
        }
    )
    dm_new_veh.rename_col(
        [
            "tra_passenger_new-vehicles",
            "tra_passenger_vehicle-fleet",
            "tra_passenger_vehicle-waste",
        ],
        [
            "tra_new-vehicles-max_tot",
            "tra_vehicles-max_tot",
            "tra_vehicles-waste-max_tot",
        ],
        dim="Variables",
    )
    for var in dm_new_veh.col_labels["Variables"]:
        dm_new_veh.change_unit(var, factor=1, old_unit="number", new_unit="seat")

    dm_new_veh.deepen(based_on="Variables")

    years_fts_all = create_years_list(2025, 2050, 1)
    missing_years = list(set(years_fts_all) - set(dm_new_veh.col_labels["Years"]))
    dm_new_veh.add(np.nan, dim="Years", dummy=True, col_label=missing_years)
    dm_new_veh.sort("Years")
    dm_new_veh.fill_nans("Years")
    dm_new_veh.filter({"Years": years_fts_all}, inplace=True)

    dm_max.append(dm_new_veh, dim="Categories2")
    dm_max.add(0, col_label="kerosene", dim="Categories2", dummy=True)
    dm_max[0, :, :, "aviation", "kerosene"] = (
        dm_max[0, :, :, "aviation", "tot"]
        - dm_max[0, :, :, "aviation", "BEV"]
        - dm_max[0, :, :, "aviation", "H2"]
    )

    mask = np.any(dm_max.array < 0, axis=-1)
    if mask.any():
        dm_tot = dm_max.filter({"Categories2": ["tot"]})
        dm_max.drop("Categories2", "tot")
        idx = dm_max.idx
        dm_max.array[mask, idx["kerosene"]] = 0
        dm_max.normalise("Categories2")
        dm_max.array[...] = dm_max.array[...] * dm_tot.array[...]

    dm_new_tech = dm_max.filter(
        {
            "Variables": ["tra_new-vehicles-max"],
            "Categories2": ["kerosene", "BEV", "H2"],
        }
    )
    dm_new_tech.normalise(dim="Categories2", inplace=True, keep_original=False)
    dm_new_tech.rename_col(
        "tra_new-vehicles-max", "tra_passenger_technology-share_new", dim="Variables"
    )
    dm_new_tech.filter({"Years": years_fts}, inplace=True)
    dm_new_tech.fill_nans("Years")
    return dm_new_tech, dm_max


def _build_fleet_dm(
    dm_pkm_suisse_fts, dm_occ_fts, dm_util_fts, dm_pop_fts, retirement_rate, years_fts
):
    """
    Compute total fleet trajectory for one FTS level and return a
    DataMatrix with the structure expected by _compute_tech_new_from_fleet.
    """
    years_fts_all = create_years_list(2025, 2050, 1)

    # Fill all annual years by interpolating between quinquennial FTS years
    dm_pkm = dm_pkm_suisse_fts.copy()
    dm_occ = dm_occ_fts.copy()
    dm_util = dm_util_fts.copy()
    dm_pop = dm_pop_fts.copy()

    for dm_tmp in [dm_pkm, dm_occ, dm_util, dm_pop]:
        missing = list(set(years_fts_all) - set(dm_tmp.col_labels["Years"]))
        dm_tmp.add(np.nan, dim="Years", col_label=missing, dummy=True)
        dm_tmp.sort("Years")
        dm_tmp.fill_nans("Years")

    arr_seats = (
        dm_pkm.array / dm_occ.array / dm_util.array * dm_pop.array[..., np.newaxis]
    )

    # waste(t) = seats(t-1) * retirement_rate
    arr_waste = np.roll(arr_seats, shift=1, axis=1) * retirement_rate
    arr_waste[:, 0, ...] = 0  # no waste in first year

    # new(t) = max(0, seats(t) - seats(t-1) + waste(t))
    arr_delta = np.diff(arr_seats, prepend=arr_seats[:, :1, ...], axis=1)
    arr_new = np.maximum(0, arr_delta + arr_waste)

    # Build DataMatrix with the required structure
    dm_out = dm_pkm.copy()
    dm_out.rename_col(
        "tra_pkm-suisse-cap", "tra_passenger_new-vehicles", dim="Variables"
    )
    dm_out.change_unit(
        "tra_passenger_new-vehicles", factor=1, old_unit="pkm/cap", new_unit="number"
    )
    dm_out.array = arr_new

    dm_out.add(
        arr_seats,
        dim="Variables",
        col_label="tra_passenger_vehicle-fleet",
        unit="number",
    )
    dm_out.add(
        arr_waste,
        dim="Variables",
        col_label="tra_passenger_vehicle-waste",
        unit="number",
    )

    dm_out.filter({"Years": years_fts_all}, inplace=True)
    return dm_out


def run(ots_state, years_fts):
    """
    Compute Swiss aviation FTS data for all 4 ambition levels.

    Parameters
    ----------
    ots_state : dict
        '_state' sub-dict returned by aviation_ots_pipeline_CH.run().
    years_fts : list of int

    Returns
    -------
    dict with keys:
        'fts' : {key: {1: dm, 2: dm, 3: dm, 4: dm}} for each FTS aviation variable
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "../data")

    # Load population FTS from lifestyles pickle
    lifestyles_pickle = os.path.join(
        current_dir, "../../../../data/datamatrix/lifestyles.pickle"
    )
    with open(lifestyles_pickle, "rb") as f:
        DM_lifestyles = pickle.load(f)
    dm_pop_fts = DM_lifestyles["fts"]["pop"]["lfs_population_"][1].filter(
        {"Country": ["Switzerland"]}
    )

    # Unpack OTS state
    dm_pkmsuisse_ots = ots_state["pkm_suisse_ots"]
    dm_pkm_monde_ots = ots_state["pkm_monde_ots"]
    dm_occ_suisse_ots = ots_state["occ_suisse_ots"]
    dm_occ_monde_ots = ots_state["occ_monde_ots"]
    dm_util_ots = ots_state["util_rate_ots"]
    value_2024 = ots_state["value_2024"]
    retirement_rate_2023 = ots_state["retirement_rate_2023"]
    dm_vehicles_max = ots_state["dm_vehicles_max"]
    eff_2023_kerosene = ots_state["eff_2023_kerosene"]
    eff_2023_BEV = ots_state["eff_2023_BEV"]
    eff_2023_H2 = ots_state["eff_2023_H2"]

    # ----------------------------------------------------------------
    # WORLD PKM FTS — four growth rate scenarios
    # ----------------------------------------------------------------
    pkm_rates_monde = [0.04, 0.029, 0.02, -0.02]
    years_all = np.arange(2025, 2051)
    n_years = years_all - 2024
    DM_pkm_monde = {}
    for lev, rate in enumerate(pkm_rates_monde, start=1):
        values = value_2024 * (1 + rate) ** n_years
        df = pd.DataFrame(
            {
                "Country": "Switzerland",
                "Years": years_all,
                "tra_pkm-cap_aviation[pkm/cap]": values,
            }
        )
        DM_pkm_monde[lev] = DataMatrix.create_from_df(df, num_cat=1).filter(
            {"Years": years_fts}
        )

    # ----------------------------------------------------------------
    # SWISS PKM FTS — lev1–3 grow, lev4 declines
    # ----------------------------------------------------------------
    pkm_rates_suisse = [0.04, 0.029, 0.02, -0.02]
    value_2023_ch = float(dm_pkmsuisse_ots[0, 2023, 0, 0])
    years_ch = np.arange(2024, 2051)
    n_years_ch = years_ch - 2023
    DM_pkm_suisse = {}
    for lev, rate in enumerate(pkm_rates_suisse, start=1):
        values = value_2023_ch * (1 + rate) ** n_years_ch
        df = pd.DataFrame(
            {
                "Country": "Switzerland",
                "Years": years_ch,
                "tra_pkm-suisse-cap_aviation[pkm/cap]": values,
            }
        )
        DM_pkm_suisse[lev] = DataMatrix.create_from_df(df, num_cat=1).filter(
            {"Years": years_fts}
        )

    # ----------------------------------------------------------------
    # OCCUPANCY FTS — four target values in 2050
    # ----------------------------------------------------------------
    occ_scenarios_suisse = [0.75, 0.80, 0.85, 0.90]
    occ_scenarios_monde = [0.75, 0.80, 0.85, 0.90]
    value_2023_occ_suisse = float(dm_occ_suisse_ots[0, 2023, 0, 0])
    value_2023_occ_monde = float(dm_occ_monde_ots[0, 2023, 0, 0])

    def _make_occ_fts(value_2023, value_2050, years_fts_arg):
        all_years = [2023] + list(range(2024, 2051))
        all_vals = np.linspace(value_2023, value_2050, num=len(all_years))
        df = (
            pd.DataFrame(
                {
                    "Country": "Switzerland",
                    "Years": all_years,
                    "tra_passenger_occupancy[%]": all_vals,
                }
            )
            .query("Years >= 2024")
            .reset_index(drop=True)
        )
        dm = DataMatrix.create_from_df(df, num_cat=0)
        dm.rename_col(
            "tra_passenger_occupancy",
            "tra_passenger_occupancy_aviation",
            dim="Variables",
        )
        dm.change_unit(
            "tra_passenger_occupancy_aviation",
            factor=1,
            old_unit="%",
            new_unit="pkm/vkm",
        )
        dm.filter({"Years": years_fts_arg}, inplace=True)
        dm.deepen()
        return dm

    DM_occ_suisse_fts = {}
    DM_occ_monde_fts = {}
    for lev in range(1, 5):
        DM_occ_suisse_fts[lev] = _make_occ_fts(
            value_2023_occ_suisse, occ_scenarios_suisse[lev - 1], years_fts
        )
        DM_occ_monde_fts[lev] = _make_occ_fts(
            value_2023_occ_monde, occ_scenarios_monde[lev - 1], years_fts
        )

    # ----------------------------------------------------------------
    # EFFICIENCY FTS — annual reduction rates per level
    # ----------------------------------------------------------------
    reduction_rates = [0.01, 0.0125, 0.015, 0.02]
    years_eff = list(range(2024, 2051))
    DM_eff_fts = {}
    for lev, rate in enumerate(reduction_rates, start=1):
        values_ice = [
            eff_2023_kerosene * ((1 - rate) ** (yr - 2023)) for yr in years_eff
        ]
        df = pd.DataFrame(
            {
                "Country": "Switzerland",
                "Years": years_eff,
                "tra_passenger_veh-efficiency_new_aviation_kerosene[MJ/km]": values_ice,
            }
        )
        df["tra_passenger_veh-efficiency_new_aviation_BEV[MJ/km]"] = eff_2023_BEV
        df["tra_passenger_veh-efficiency_new_aviation_H2[MJ/km]"] = eff_2023_H2
        DM_eff_fts[lev] = DataMatrix.create_from_df(df, num_cat=2).filter(
            {"Years": years_fts}
        )

    # ----------------------------------------------------------------
    # UTILISATION RATE FTS — from Excel
    # ----------------------------------------------------------------
    file = os.path.join(data_dir, "aviation_utilisation-rate-FTS.xlsx")
    df = pd.read_excel(file)
    dm_util_fts_base = DataMatrix.create_from_df(df, num_cat=1)
    dm_util_fts_base.filter({"Years": years_fts}, inplace=True)
    DM_util_fts = {lev: dm_util_fts_base.copy() for lev in range(1, 5)}

    # ----------------------------------------------------------------
    # TECH SHARE NEW FTS — lev1: all kerosene; lev2: mix; lev3/4: computed
    # ----------------------------------------------------------------
    dm_max = _compute_new_veh_max(dm_vehicles_max.copy(), years_fts)

    def _build_fleet_for_lev(lev):
        dm_pkm_ch = DM_pkm_suisse[lev]
        dm_occ = DM_occ_suisse_fts[lev]
        dm_util = DM_util_fts[lev].filter({"Categories1": ["aviation"]})
        return _build_fleet_dm(
            dm_pkm_ch, dm_occ, dm_util, dm_pop_fts, retirement_rate_2023, years_fts
        )

    # lev3
    dm_fleet_3 = _build_fleet_for_lev(3)
    dm_new_tech_3, dm_max_lev3 = _compute_tech_new_from_fleet(
        dm_fleet_3, dm_max.copy(), years_fts
    )

    # lev4
    dm_fleet_4 = _build_fleet_for_lev(4)
    dm_new_tech_4, _ = _compute_tech_new_from_fleet(
        dm_fleet_4, dm_max.copy(), years_fts
    )

    # lev1: all kerosene
    dm_new_tech_1 = dm_new_tech_3.copy()
    dm_new_tech_1[...] = 0
    dm_new_tech_1[..., "kerosene"] = 1

    # lev2: halfway between lev1 and lev3
    dm_new_tech_2 = dm_new_tech_1.copy()
    dm_new_tech_2.array = 0.5 * dm_new_tech_1.array + 0.5 * dm_new_tech_3.array

    DM_tech_share_new = {
        1: dm_new_tech_1,
        2: dm_new_tech_2,
        3: dm_new_tech_3,
        4: dm_new_tech_4,
    }

    # ----------------------------------------------------------------
    # FUEL MIX FTS — SAF availability based on ATAG Waypoint 2050
    # ----------------------------------------------------------------
    # Abrantes et al. (2021): max SAF 2050 = 200 Mt; 43 MJ/kg; 0.6% allocated to Switzerland
    val_SAF_2050_MJ = 200 * 1e9 * 43 * 0.006

    dm_SAF_base = dm_util_fts_base.copy()
    dm_SAF_base.add(
        np.nan,
        dim="Variables",
        col_label="tra_passenger-max-SAF",
        unit="MJ",
        dummy=True,
    )
    dm_SAF_base["Switzerland", 2050, "tra_passenger-max-SAF", "aviation"] = (
        val_SAF_2050_MJ
    )
    dm_SAF_base["Switzerland", 2025, "tra_passenger-max-SAF", "aviation"] = 0
    dm_SAF_base["Switzerland", 2030, "tra_passenger-max-SAF", "aviation"] = 0
    dm_SAF_base.fill_nans("Years")

    # seats_max_SAF = SAF_energy / (eff_lev4_kerosene * util_rate)
    dm_eff_lev4_kero = DM_eff_fts[4].filter(
        {"Categories1": ["aviation"], "Categories2": ["kerosene"]}
    )
    dm_eff_lev4_kero.group_all("Categories2")

    dm_SAF_calc = dm_SAF_base.copy()
    dm_SAF_calc.append(dm_eff_lev4_kero, dim="Variables")
    dm_SAF_calc.operation(
        "tra_passenger-max-SAF",
        "/",
        "tra_passenger_veh-efficiency_new",
        out_col="tra_skm_SAF",
        unit="skm",
    )
    dm_SAF_calc.operation(
        "tra_skm_SAF",
        "/",
        "tra_passenger_utilisation-rate",
        out_col="tra_vehicles-max_SAF",
        unit="seat",
    )

    dm_max_seat_kero = dm_max_lev3.filter(
        {
            "Categories2": ["kerosene"],
            "Variables": ["tra_vehicles-max"],
            "Years": years_fts,
        }
    )
    dm_max_seat_kero.group_all("Categories2")
    dm_SAF_calc.append(dm_max_seat_kero, dim="Variables")
    dm_SAF_calc.operation(
        "tra_vehicles-max_SAF",
        "/",
        "tra_vehicles-max",
        out_col="tra_fuel-mix_biofuel",
        unit="%",
    )
    dm_fm = dm_SAF_calc.filter({"Variables": ["tra_fuel-mix_biofuel"]})
    dm_fm.array = np.minimum(
        dm_fm.array, 1.0
    )  # SAF can cover at most 100% of kerosene demand
    dm_fm.deepen(based_on="Variables")
    dm_fm.switch_categories_order()
    dm_fm.add(0, dim="Categories1", col_label="efuel", dummy=True)
    dm_fm.sort("Categories1")

    DM_fuel_mix = {}
    for lev in range(1, 5):
        DM_fuel_mix[lev] = dm_fm.copy()
    DM_fuel_mix[1].array[...] = 0
    DM_fuel_mix[2].array[...] = 0.5 * DM_fuel_mix[1].array + 0.5 * DM_fuel_mix[3].array
    # lev4 same as lev3 (same SAF budget)
    DM_fuel_mix[4].array[...] = DM_fuel_mix[3].array[...]

    return {
        "fts": {
            "passenger_aviation-pkm": DM_pkm_monde,
            "passenger_occupancy": DM_occ_monde_fts,
            "passenger_veh-efficiency_new": DM_eff_fts,
            "passenger_utilization-rate": DM_util_fts,
            "passenger_technology-share_new": DM_tech_share_new,
            "fuel-mix": DM_fuel_mix,
        },
    }
