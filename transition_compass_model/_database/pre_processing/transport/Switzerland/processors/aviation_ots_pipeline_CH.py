"""
Aviation OTS + FXA preprocessing for Switzerland.
Produces aviation sub-DataMatrices to be merged into the transport pickle.
"""

import os
import pickle

import numpy as np
import pandas as pd

from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    linear_fitting,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def run(dm_pop_ch_ots, years_ots, years_fts, dm_pkm_cap_aviation):
    """
    Compute Swiss aviation OTS and FXA preprocessing data.

    Parameters
    ----------
    dm_pop_ch_ots : DataMatrix
        Population OTS, Switzerland only.
    years_ots : list of int
    years_fts : list of int
    dm_pkm_cap_aviation : DataMatrix
        Swiss aviation pkm/cap from aviation_part1_pipeline_CH.run() —
        used as the monde reference curve, replaces the old pickle load.

    Returns
    -------
    dict with keys:
        'ots'    : aviation sub-DataMatrices for OTS fields (Categories1=["aviation"])
        'fxa'    : aviation sub-DataMatrices for all FXA fields
        '_state' : internal state needed by aviation_fts_CH.run() (called from
                   fts_bau_pickle_run)
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "../data")

    # ----------------------------------------------------------------
    # 1. WORLD PKM OTS — post-COVID interpolation
    # ----------------------------------------------------------------
    dm_pkm = dm_pkm_cap_aviation.filter({"Country": ["Switzerland"]}).copy()
    # IATA (2025): 2024 global demand = 2019 × 1.038
    value_2024 = float(dm_pkm[0, 2019, 0, 0]) * 1.038
    dm_pkm.add(value_2024, dim="Years", col_label=[2024], dummy=True)
    dm_pkm[0, 2023, 0, 0] = np.nan
    dm_pkm[0, 2022, 0, 0] = np.nan
    dm_pkm.fill_nans(dim_to_interp="Years")
    dm_pkm.drop(col_label=[2024], dim="Years")
    dm_pkm[0, 2020, 0, 0] = 3103.0  # apply US 2019/2020 COVID rate
    dm_pkm_monde_ots = dm_pkm

    # ----------------------------------------------------------------
    # 2. SWISS PKM OTS — from CSV, adjust world using CH/world ratio
    # ----------------------------------------------------------------
    file = os.path.join(data_dir, "aviation_pkm_suisse.csv")
    df = pd.read_csv(file, sep=";", decimal=",")
    df.drop(["Unnamed: 0"], axis=1, inplace=True)
    dm_pkmsuisse = DataMatrix.create_from_df(df, num_cat=1)
    dm_pkmsuisse.rename_col("tra_pkm-cap", "tra_pkm-suisse-cap", dim="Variables")

    dm_pkm_adj = dm_pkm_monde_ots.copy()
    idx = dm_pkm_adj.idx
    dm_pkm_adj.array[:, 0 : idx[2005], ...] = np.nan
    dm_pkm_adj.append(dm_pkmsuisse, dim="Variables")
    dm_pkm_adj.operation(
        "tra_pkm-cap", "/", "tra_pkm-suisse-cap", out_col="ratio", unit="%"
    )
    linear_fitting(dm_pkm_adj, years_ots, based_on=create_years_list(2006, 2017, 1))
    dm_pkm_adj[:, :, "ratio", ...] = np.maximum(1.05, dm_pkm_adj[:, :, "ratio", ...])
    dm_pkm_adj.operation(
        "ratio", "*", "tra_pkm-suisse-cap", out_col="tra_pkm-cap_adj", unit="pkm/cap"
    )
    dm_pkm_adj.filter({"Variables": ["tra_pkm-cap_adj"]}, inplace=True)
    dm_pkm_adj.rename_col("tra_pkm-cap_adj", "tra_pkm-cap", dim="Variables")
    dm_pkm_monde_ots = dm_pkm_adj

    # ----------------------------------------------------------------
    # 3. EFFICIENCY NEW OTS — from Excel
    # ----------------------------------------------------------------
    file = os.path.join(data_dir, "aviation_energy_intensity_fleet.xlsx")
    df = pd.read_excel(file, sheet_name="Feuil1")
    dm_efficiency_new = DataMatrix.create_from_df(df, num_cat=2)
    # Categories1=["aviation"], Categories2=["kerosene","BEV","H2"]

    # ----------------------------------------------------------------
    # 4. TECH SHARE NEW OTS — from Excel
    # ----------------------------------------------------------------
    file = os.path.join(data_dir, "aviation_passenger_technology_share_new_ots.xlsx")
    df = pd.read_excel(file)
    dm_tech_share_new = DataMatrix.create_from_df(df, num_cat=2)

    # ----------------------------------------------------------------
    # 5. OCCUPANCY — Swiss raw and world weighted average
    # ----------------------------------------------------------------
    file = os.path.join(data_dir, "aviation_occupancy_ots.xlsx")
    df = pd.read_excel(file)
    dm_occ_suisse = DataMatrix.create_from_df(df, num_cat=1)
    missing = set(years_ots) - set(dm_occ_suisse.col_labels["Years"])
    dm_occ_suisse.add(np.nan, "Years", list(missing), dummy=True)
    dm_occ_suisse.sort("Years")
    dm_occ_suisse.fill_nans("Years")
    dm_occ_suisse.change_unit(
        "tra_passenger_occupancy", factor=1, old_unit="%", new_unit="pkm/vkm"
    )

    file = os.path.join(data_dir, "aviation_occupancy_pondere_ots.xlsx")
    df = pd.read_excel(file)
    dm_occ_world_raw = DataMatrix.create_from_df(df, num_cat=1)
    missing = set(years_ots) - set(dm_occ_world_raw.col_labels["Years"])
    dm_occ_world_raw.add(np.nan, "Years", list(missing), dummy=True)
    dm_occ_world_raw.sort("Years")
    dm_occ_world_raw.fill_nans("Years")

    # Weighted occupancy: (pkm_CH×occ_CH + (pkm_tot−pkm_CH)×occ_world) / pkm_tot
    arr_pond = (
        dm_pkmsuisse[...] * dm_occ_suisse[...]
        + (dm_pkm_monde_ots[...] - dm_pkmsuisse[...]) * dm_occ_world_raw[...]
    ) / dm_pkm_monde_ots[...]
    dm_occ_world_raw.add(
        arr_pond,
        dim="Variables",
        col_label="tra_passenger_occupancy_ponderee",
        unit="%",
    )
    dm_occ_world_raw.drop(dim="Variables", col_label=["tra_passenger_occupancy"])
    dm_occ_world_raw.rename_col(
        "tra_passenger_occupancy_ponderee", "tra_passenger_occupancy", dim="Variables"
    )
    dm_occ_world_raw.fill_nans(dim_to_interp="Years")
    dm_occ_world_raw.change_unit(
        "tra_passenger_occupancy", old_unit="%", new_unit="pkm/vkm", factor=1
    )
    dm_occupancy_monde = dm_occ_world_raw

    # ----------------------------------------------------------------
    # 6. UTILISATION RATE OTS — from Excel + ratio adjustment
    # ----------------------------------------------------------------
    file = os.path.join(data_dir, "aviation_utilisation-rate-OTS.xlsx")
    df = pd.read_excel(file)
    dm_utilirate = DataMatrix.create_from_df(df, num_cat=1)
    dm_utilirate.append(dm_pkmsuisse.copy(), dim="Variables")
    dm_utilirate.operation(
        "tra_passenger_utilisation-rate",
        "/",
        "tra_pkm-suisse-cap",
        out_col="ratio",
        dim="Variables",
        unit="%",
    )
    for yr in [2020, 2021, 2022, 2023]:
        dm_utilirate[0, yr, "ratio", 0] = np.nan
    linear_fitting(
        dm_utilirate, years_ots=years_ots, based_on=create_years_list(2005, 2019, 1)
    )
    dm_utilirate.operation(
        "ratio",
        "*",
        "tra_pkm-suisse-cap",
        out_col="tra_passenger_utilisation-rate_new",
        dim="Variables",
        unit="vkm/veh",
    )
    dm_utilirate[0, 2022, "tra_passenger_utilisation-rate", 0] = dm_utilirate[
        0, 2022, "tra_passenger_utilisation-rate_new", 0
    ]
    dm_utilirate[0, 2023, "tra_passenger_utilisation-rate", 0] = dm_utilirate[
        0, 2023, "tra_passenger_utilisation-rate_new", 0
    ]
    dm_utilirate.filter({"Variables": ["tra_passenger_utilisation-rate"]}, inplace=True)

    # ----------------------------------------------------------------
    # 7. SEATS OTS — seats = pkm * pop / (occ * util_rate)
    # ----------------------------------------------------------------
    arr_seats = (
        dm_pkm_monde_ots.array
        / dm_occupancy_monde.array
        / dm_utilirate.array
        * dm_pop_ch_ots.array[..., np.newaxis]
    )
    dm_seats = dm_occupancy_monde.copy()
    dm_seats.rename_col(
        "tra_passenger_occupancy", "tra_passenger_seats", dim="Variables"
    )
    dm_seats.array = arr_seats

    # ----------------------------------------------------------------
    # 8. VEHICLE WASTE + NEW FLEET OTS — from retirement rate Excel
    # ----------------------------------------------------------------
    file = os.path.join(data_dir, "aviation_waste_suisse.xlsx")
    df_retir = pd.read_excel(file, sheet_name="Feuil2")
    dm_retirrate = DataMatrix.create_from_df(df_retir, num_cat=1)
    dm_retirrate.append(dm_seats, dim="Variables")
    dm_retirrate.lag_variable("tra_passenger_seats", shift=1, subfix="_tm1")
    dm_retirrate.operation(
        "tra_passenger_seats_tm1",
        "*",
        "tra_retirement-rate",
        out_col="tra_passenger_vehicle-waste",
        unit="number",
    )
    dm_retirrate.operation(
        "tra_passenger_seats",
        "-",
        "tra_passenger_seats_tm1",
        out_col="delta",
        unit="number",
    )
    dm_retirrate.operation(
        "delta",
        "+",
        "tra_passenger_vehicle-waste",
        out_col="tra_passenger_new-vehicles",
        unit="number",
    )
    dm_retirrate[:, :, "tra_passenger_new-vehicles", :] = np.maximum(
        0, dm_retirrate[:, :, "tra_passenger_new-vehicles", :]
    )
    dm_retirrate[:, 2022, "tra_passenger_new-vehicles", :] = np.nan
    dm_retirrate.fill_nans(dim_to_interp="Years")
    for t in create_years_list(2020, 2023, 1):
        dm_retirrate[0, t, "tra_passenger_seats", 0] = (
            dm_retirrate[0, t, "tra_passenger_new-vehicles", 0]
            - dm_retirrate[0, t, "tra_passenger_vehicle-waste", 0]
            + dm_retirrate[0, t - 1, "tra_passenger_seats", 0]
        )

    # Redo utilisation rate consistent with recomputed seats
    dm_seats_redo = dm_retirrate.filter({"Variables": ["tra_passenger_seats"]})
    arr_util_redo = (
        dm_pkm_monde_ots.array
        * dm_pop_ch_ots.array[..., np.newaxis]
        / (dm_occupancy_monde.array * dm_seats_redo.array)
    )
    dm_utilirate.array = arr_util_redo

    # waste with Categories2
    dm_waste = dm_retirrate.filter({"Variables": ["tra_passenger_vehicle-waste"]})
    dm_waste.add(np.nan, dim="Years", col_label=years_fts, dummy=True)
    dm_waste.rename_col(
        "tra_passenger_vehicle-waste",
        "tra_passenger_vehicle-waste_kerosene",
        dim="Variables",
    )
    dm_waste.deepen(based_on="Variables")
    dm_waste.add(0, dummy=True, col_label=["BEV", "H2"], dim="Categories2")

    # new-vehicles with Categories2
    dm_new = dm_retirrate.filter({"Variables": ["tra_passenger_new-vehicles"]})
    dm_new.add(np.nan, dim="Years", col_label=years_fts, dummy=True)
    dm_new.rename_col(
        "tra_passenger_new-vehicles",
        "tra_passenger_new-vehicles_kerosene",
        dim="Variables",
    )
    dm_new.deepen(based_on="Variables")
    dm_new.add(0, dummy=True, col_label=["BEV", "H2"], dim="Categories2")

    # ----------------------------------------------------------------
    # 9. LIFETIME FXA — from Excel
    # ----------------------------------------------------------------
    file = os.path.join(data_dir, "aviation_vehicles_lifetime.xlsx")
    df = pd.read_excel(file)
    dm_lifetime = DataMatrix.create_from_df(df, num_cat=2)
    dm_lifetime.add(25, dim="Years", dummy=True, col_label=years_fts)

    # ----------------------------------------------------------------
    # 10. VEHICLES MAX FXA — new-tech capacity constraints
    # ----------------------------------------------------------------
    file = os.path.join(data_dir, "aviation_number_seats_max.xlsx")
    df = pd.read_excel(file, sheet_name="1")
    dm_vehicles_max = DataMatrix.create_from_df(df, num_cat=2)
    dm_vehicles_max.rename_col(
        "tra_new-vehicules-max", "tra_vehicles-max", dim="Variables"
    )
    dm_vehicles_max.change_unit(
        "tra_vehicles-max", factor=0.1, old_unit="seat", new_unit="seat"
    )
    dm_vehicles_max.add(0, dim="Years", dummy=True, col_label=years_ots)
    dm_vehicles_max.sort("Years")

    # ----------------------------------------------------------------
    # 11. FXA PASSENGER TECH — fleet share + efficiency + waste + new
    # ----------------------------------------------------------------
    dm_fleet_eff = dm_efficiency_new.filter({"Categories1": ["aviation"]})
    dm_fleet_eff.rename_col_regex(
        "tra_passenger_veh-efficiency_new",
        "tra_passenger_veh-efficiency_fleet",
        dim="Variables",
    )
    dm_fleet_eff.add(np.nan, dim="Years", dummy=True, col_label=years_fts)

    dm_fleetshare = dm_tech_share_new.filter({"Categories1": ["aviation"]})
    dm_fleetshare.rename_col_regex(
        "tra_passenger_technology-share_new",
        "tra_passenger_technology-share_fleet",
        dim="Variables",
    )
    dm_fleetshare.add(np.nan, dim="Years", dummy=True, col_label=years_fts)

    tech_cats = dm_new.col_labels["Categories2"]
    dm_pass_tech = dm_new.copy()
    dm_pass_tech.append(dm_waste, dim="Variables")
    dm_pass_tech.append(
        dm_fleetshare.filter({"Categories2": tech_cats}), dim="Variables"
    )
    dm_pass_tech.append(
        dm_fleet_eff.filter({"Categories2": tech_cats}), dim="Variables"
    )

    # ----------------------------------------------------------------
    # 12. SKM CH and SKM monde (for share-local-emissions in FTS pipeline)
    # ----------------------------------------------------------------
    arr_skm_ch = (
        dm_pkmsuisse.array / dm_occ_suisse.array * dm_pop_ch_ots.array[..., np.newaxis]
    )
    dm_skm_ch = dm_occ_suisse.copy()
    dm_skm_ch.rename_col(
        "tra_passenger_occupancy", "tra_passenger-demand-skm_CH", dim="Variables"
    )
    dm_skm_ch.array = arr_skm_ch

    arr_skm_monde = (
        dm_pkm_monde_ots.array
        / dm_occupancy_monde.array
        * dm_pop_ch_ots.array[..., np.newaxis]
    )
    dm_skm_monde = dm_pkm_monde_ots.copy()
    dm_skm_monde.rename_col("tra_pkm-cap", "tra_passenger-demand-skm", dim="Variables")
    dm_skm_monde.array = arr_skm_monde

    # ----------------------------------------------------------------
    # 13. FXA SHARE LOCAL EMISSIONS — CH skm / world skm (OTS + FTS trend)
    # Uses reference growth rates: lev2 for CH (0.029), lev1 for world (0.04)
    # ----------------------------------------------------------------
    lifestyles_pickle = os.path.join(
        current_dir, "../../../../data/datamatrix/lifestyles.pickle"
    )
    with open(lifestyles_pickle, "rb") as _f:
        _DM_ls = pickle.load(_f)
    dm_pop_fts = _DM_ls["fts"]["pop"]["lfs_population_"][1].filter(
        {"Country": ["Switzerland"]}
    )

    _rate_ch = 0.029
    _value_2023_ch = float(dm_pkmsuisse[0, 2023, 0, 0])
    _years_ch = np.arange(2024, 2051)
    _df_pch = pd.DataFrame(
        {
            "Country": "Switzerland",
            "Years": _years_ch,
            "tra_pkm-suisse-cap_aviation[pkm/cap]": _value_2023_ch
            * (1 + _rate_ch) ** (_years_ch - 2023),
        }
    )
    dm_pkm_suisse_fts_ref = DataMatrix.create_from_df(_df_pch, num_cat=1).filter(
        {"Years": years_fts}
    )

    _rate_monde = 0.04
    _years_monde = np.arange(2025, 2051)
    _df_pm = pd.DataFrame(
        {
            "Country": "Switzerland",
            "Years": _years_monde,
            "tra_pkm-cap_aviation[pkm/cap]": value_2024
            * (1 + _rate_monde) ** (_years_monde - 2024),
        }
    )
    dm_pkm_monde_fts_ref = DataMatrix.create_from_df(_df_pm, num_cat=1).filter(
        {"Years": years_fts}
    )

    def _occ_fts(value_2023, value_2050):
        _ay = [2023] + list(range(2024, 2051))
        _av = np.linspace(value_2023, value_2050, num=len(_ay))
        _d = (
            pd.DataFrame(
                {
                    "Country": "Switzerland",
                    "Years": _ay,
                    "tra_passenger_occupancy[%]": _av,
                }
            )
            .query("Years >= 2024")
            .reset_index(drop=True)
        )
        _dm = DataMatrix.create_from_df(_d, num_cat=0)
        _dm.rename_col(
            "tra_passenger_occupancy",
            "tra_passenger_occupancy_aviation",
            dim="Variables",
        )
        _dm.change_unit(
            "tra_passenger_occupancy_aviation",
            factor=1,
            old_unit="%",
            new_unit="pkm/vkm",
        )
        _dm.filter({"Years": years_fts}, inplace=True)
        _dm.deepen()
        return _dm

    dm_occ_suisse_fts_ref = _occ_fts(float(dm_occ_suisse[0, 2023, 0, 0]), 0.80)
    dm_occ_monde_fts_ref = _occ_fts(float(dm_occupancy_monde[0, 2023, 0, 0]), 0.75)

    file = os.path.join(data_dir, "aviation_utilisation-rate-FTS.xlsx")
    df = pd.read_excel(file)
    dm_util_fts_base = DataMatrix.create_from_df(df, num_cat=1)
    dm_util_fts_base.filter({"Years": years_fts}, inplace=True)
    dm_util_ref = dm_util_fts_base.filter({"Categories1": ["aviation"]})

    years_fts_all = create_years_list(2025, 2050, 1)
    dm_p_ch = dm_pkm_suisse_fts_ref.copy()
    dm_o_ch = dm_occ_suisse_fts_ref.copy()
    dm_p_monde = dm_pkm_monde_fts_ref.copy()
    dm_o_monde = dm_occ_monde_fts_ref.copy()
    dm_u_ref = dm_util_ref.copy()
    dm_pop_fts_annual = dm_pop_fts.copy()
    for _dm in [dm_p_ch, dm_o_ch, dm_p_monde, dm_o_monde, dm_u_ref, dm_pop_fts_annual]:
        _miss = list(set(years_fts_all) - set(_dm.col_labels["Years"]))
        _dm.add(np.nan, "Years", _miss, dummy=True)
        _dm.sort("Years")
        _dm.fill_nans("Years")

    arr_skm_ch_fts = (
        dm_p_ch.array / dm_o_ch.array * dm_pop_fts_annual.array[..., np.newaxis]
    )
    dm_skm_ch_fts = dm_o_ch.copy()
    dm_skm_ch_fts.rename_col(
        "tra_passenger_occupancy", "tra_passenger-demand-skm_CH", dim="Variables"
    )
    dm_skm_ch_fts.array = arr_skm_ch_fts
    dm_skm_ch_fts.filter({"Years": years_fts}, inplace=True)

    arr_skm_monde_fts = (
        dm_p_monde.array / dm_o_monde.array * dm_pop_fts_annual.array[..., np.newaxis]
    )
    dm_skm_monde_fts = dm_p_monde.copy()
    dm_skm_monde_fts.rename_col(
        "tra_pkm-cap", "tra_passenger-demand-skm", dim="Variables"
    )
    dm_skm_monde_fts.array = arr_skm_monde_fts
    dm_skm_monde_fts.filter({"Years": years_fts}, inplace=True)

    dm_skm_ch_combined = dm_skm_ch.copy()
    dm_skm_ch_combined.append(dm_skm_ch_fts, dim="Years")
    dm_skm_monde_combined = dm_skm_monde.copy()
    dm_skm_monde_combined.append(dm_skm_monde_fts, dim="Years")
    dm_skm_ch_combined.append(dm_skm_monde_combined, dim="Variables")
    dm_skm_ch_combined.operation(
        "tra_passenger-demand-skm_CH",
        "/",
        "tra_passenger-demand-skm",
        out_col="tra_share-emissions-local",
        unit="%",
    )
    dm_share_local = dm_skm_ch_combined.filter(
        {"Variables": ["tra_share-emissions-local"]}
    )

    # ----------------------------------------------------------------
    # 14. FXA FUEL MIX AVAILABILITY — SAF energy budget (ATAG Waypoint 2050)
    # Abrantes et al. (2021): max SAF 2050 = 200 Mt; 43 MJ/kg; 0.6% to Switzerland
    # ----------------------------------------------------------------
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
    dm_max_SAF_fxa = dm_SAF_base.filter({"Variables": ["tra_passenger-max-SAF"]})
    dm_max_SAF_fxa.rename_col(
        "tra_passenger-max-SAF",
        "tra_passenger_available-fuel-mix_biofuel",
        dim="Variables",
    )
    dm_max_SAF_fxa.deepen(based_on="Variables")
    dm_max_SAF_fxa.add(0, dim="Categories2", col_label="efuel", dummy=True)
    dm_max_SAF_fxa.switch_categories_order()
    dm_max_SAF_fxa.add(0, dummy=True, dim="Years", col_label=years_ots)
    dm_max_SAF_fxa.sort("Years")

    return {
        "ots": {
            "passenger_aviation-pkm": dm_pkm_monde_ots,
            "passenger_veh-efficiency_new": dm_efficiency_new,
            "passenger_technology-share_new": dm_tech_share_new,
            "passenger_occupancy": dm_occupancy_monde,
            "passenger_utilization-rate": dm_utilirate,
        },
        "fxa": {
            "passenger_tech": dm_pass_tech,
            "passenger_vehicle-lifetime": dm_lifetime,
            "vehicles-max": dm_vehicles_max,
            "share-local-emissions": dm_share_local,
            "fuel-mix-availability": dm_max_SAF_fxa,
        },
        "_state": {
            "pkm_suisse_ots": dm_pkmsuisse,
            "pkm_monde_ots": dm_pkm_monde_ots,
            "occ_suisse_ots": dm_occ_suisse,
            "occ_monde_ots": dm_occupancy_monde,
            "util_rate_ots": dm_utilirate,
            "value_2024": value_2024,
            "retirement_rate_2023": float(
                dm_retirrate[0, 2023, "tra_retirement-rate", 0]
            ),
            "dm_vehicles_max": dm_vehicles_max,
            "eff_2023_kerosene": float(
                dm_efficiency_new[0, 2023, 0, "aviation", "kerosene"]
            ),
            "eff_2023_BEV": float(dm_efficiency_new[0, 2023, 0, "aviation", "BEV"]),
            "eff_2023_H2": float(dm_efficiency_new[0, 2023, 0, "aviation", "H2"]),
        },
    }
