"""
EU27 passenger fleet + technology FXA.

Produces four DataMatrix objects used as fixed assumptions in the transport module:
  dm_tech    : fleet tech share, vehicle efficiency, new vehicles, vehicle waste (OTS only)
  dm_fleet   : absolute fleet by mode×tech (OTS + FTS)
  dm_new     : new vehicles by mode×tech (OTS + FTS)
  dm_renrate : renewal rate by mode×tech (OTS only)

Accounting basis
----------------
All data sources are **territorial** (vehicles registered/operating in EU27), consistent
with the territorial accounting basis used throughout EU27 transport preprocessing:

  - Fleet stock (2W, LDV, bus, rail): JRC-IDEES TrRoad_act "Stock of vehicles" +
    Eurostat road_eqs_carmot (new registrations). Territorial (EU27-registered vehicles).
    For EU27 as a large bloc, territorial ≈ residency — cross-border transit is a minor
    fraction.

  - Aviation fleet: JRC-IDEES Aviation sheet "Stock of aircrafts - Passenger transport"
    × "Seats available per flight". Stored in **seats** (not aircraft), consistent with
    CH preprocessing. Divided by avg_seats_per_aircraft in transport_module.py for
    material decomposition only; does not affect energy or emissions.
    Territorial (EU27-based aircraft, covering domestic + intra-EU27 + extra-EU27 routes).

  - Vehicle efficiency: JRC-IDEES. Territorial.

  - New vehicles: JRC-IDEES + Eurostat. Territorial (new registrations).

This FXA shapes technology mix and renewal rates only; it does not affect the
accounting basis of energy or emissions outputs (those are driven by the pkm lever).

See CH preprocessing (passenger_fleet_pipeline.py, freight_tkm_modal_share.py) for the
same pattern applied to Switzerland.
"""

import os
import pickle

import numpy as np
import pandas as pd

from transition_compass_model._database.pre_processing.api_routine_Eurostat import (
    get_data_api_eurostat,
)
from transition_compass_model._database.pre_processing.routine_JRC import get_jrc_data
from transition_compass_model.model.common.auxiliary_functions import (
    eurostat_iso2_dict,
    jrc_iso2_dict,
    linear_fitting,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(_THIS_DIR, "../data/datamatrix/fxa_passenger_tech.pickle")

_CATEGORIES2_ALL = [
    "BEV",
    "CEV",
    "FCEV",
    "H2",
    "ICE-diesel",
    "ICE-gas",
    "ICE-gasoline",
    "PHEV-diesel",
    "PHEV-gasoline",
    "kerosene",
    "mt",
]


def run(years_ots: list, years_fts: list):
    """
    Build passenger fleet FXA for EU27.

    Returns
    -------
    dm_tech : DataMatrix
    dm_fleet : DataMatrix  (intermediate for occupancy/utilization processors)
    dm_new : DataMatrix    (intermediate for tech-share-new processor)
    dm_renrate : DataMatrix (intermediate for fxa_freight_mode_other)
    """
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    startyear = 1989
    years_ots_ext = list(range(startyear, years_ots[-1] + 1))

    ###########################################################################
    # FLEET
    ###########################################################################

    categories2_all = _CATEGORIES2_ALL.copy()

    dict_iso2 = eurostat_iso2_dict()
    dict_iso2.pop("CH")
    dict_iso2_jrc = jrc_iso2_dict()

    # 2W
    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_act",
        "variable": "Stock of vehicles - total (vehicles)",
        "sheet_last_row": "Powered two-wheelers",
        "sub_variables": ["Powered two-wheelers"],
        "calc_names": ["2W"],
    }
    dm_fleet_2w = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    # Anchor EU27 2W fleet to Eurostat tran_r_vehst MOTO (JRC is ~60% over Eurostat)
    dm_fleet_2w_jrc = dm_fleet_2w.copy()
    _filter_moto = {
        "geo\\TIME_PERIOD": [k for k in dict_iso2.keys() if k != "EU27_2020"],
        "vehicle": ["MOTO"],
        "unit": "NR",
    }
    dm_eurostat_moto = get_data_api_eurostat(
        "tran_r_vehst",
        _filter_moto,
        {"Country": "geo\\TIME_PERIOD", "Variables": "vehicle"},
        "num",
    )
    dm_eurostat_moto.drop("Country", "United Kingdom")
    _eu27_moto = np.nansum(dm_eurostat_moto.array, axis=0)  # (n_years, 1): EU27 sum
    _moto_years = dm_eurostat_moto.col_labels["Years"]
    _idx_2w = dm_fleet_2w.idx
    for _i, _y in enumerate(_moto_years):
        if _y in dm_fleet_2w.col_labels["Years"]:
            dm_fleet_2w.array[_idx_2w["EU27"], _idx_2w[_y], _idx_2w["2W"]] = float(
                _eu27_moto[_i, 0]
            )
    dm_fleet_2w.rename_col("2W", "2W_ICE-gasoline", "Variables")
    dm_fleet_2w.deepen()
    categories2_missing = categories2_all.copy()
    for cat in dm_fleet_2w.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_fleet_2w.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories1"
    )
    dm_fleet_2w.sort("Categories1")

    # LDV - total from Eurostat
    code = "road_eqs_carmot"
    filter_eurostat = {
        "geo\\TIME_PERIOD": list(dict_iso2.keys()),
        "engine": ["TOTAL"],
        "mot_nrg": ["TOTAL"],
        "unit": "NR",
    }
    mapping_dim = {"Country": "geo\\TIME_PERIOD", "Variables": "mot_nrg"}
    mapping_calc = {"LDV": "TOTAL"}
    dm_eurostat_fleet_total = get_data_api_eurostat(
        code, filter_eurostat, mapping_dim, "num"
    )
    dm_eurostat_fleet = dm_eurostat_fleet_total.filter(
        {"Years": list(range(2000, 2021 + 1, 1))}
    )
    dm_eurostat_fleet.drop("Country", "United Kingdom")

    # LDV - tech split from JRC
    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_act",
        "variable": "Stock of vehicles - total (vehicles)",
        "sheet_last_row": "Battery electric vehicles",
        "categories": "Passenger cars",
        "sub_variables": [
            "Gasoline engine",
            "Diesel oil engine",
            "LPG engine",
            "Natural gas engine",
            "Plug-in hybrid electric",
            "Battery electric vehicles",
        ],
        "calc_names": [
            "LDV_ICE-gasoline",
            "LDV_ICE-diesel",
            "LDV_gas-lpg",
            "LDV_gas-natural",
            "LDV_PHEV",
            "LDV_BEV",
        ],
    }
    dm_fleet_ldv = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_fleet_ldv.deepen()
    mapping_calc = {"ICE-gas": ["gas-lpg", "gas-natural"]}
    dm_fleet_ldv.groupby(
        mapping_calc, dim="Categories1", aggregation="sum", regex=False, inplace=True
    )
    dm_temp = dm_fleet_ldv.filter({"Categories1": ["PHEV"]})
    dm_temp.rename_col("PHEV", "PHEV-gasoline", "Categories1")
    dm_temp.add(dm_temp.array, col_label="PHEV-diesel", dim="Categories1")
    dm_temp.sort("Categories1")
    dm_temp1 = dm_fleet_ldv.filter({"Categories1": ["ICE-diesel", "ICE-gasoline"]})
    dm_temp1.normalise("Categories1")
    dm_temp.array = dm_temp.array * dm_temp1.array
    dm_fleet_ldv.drop("Categories1", "PHEV")
    dm_fleet_ldv.append(dm_temp, "Categories1")
    dm_fleet_ldv.sort("Categories1")
    dm_temp = dm_fleet_ldv.normalise("Categories1", inplace=False)
    dm_temp.array = dm_eurostat_fleet.array[..., np.newaxis] * dm_temp.array
    dm_temp.rename_col("LDV_share", "LDV", "Variables")
    dm_temp.units["LDV"] = "num"
    idx = dm_temp.idx
    for y in range(2000, 2009 + 1, 1):
        dm_temp.array[:, idx[y], :, :] = np.nan
    dm_fleet_ldv = dm_temp.copy()
    categories2_missing = categories2_all.copy()
    for cat in dm_fleet_ldv.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_fleet_ldv.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories1"
    )
    dm_fleet_ldv.sort("Categories1")

    # Buses
    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_act",
        "variable": "Stock of vehicles - total (vehicles)",
        "categories": "Motor coaches, buses and trolley buses",
        "sheet_last_row": "Battery electric vehicles",
        "sub_variables": [
            "Gasoline engine",
            "Diesel oil engine",
            "LPG engine",
            "Natural gas engine",
            "Battery electric vehicles",
        ],
        "calc_names": [
            "bus_ICE-gasoline",
            "bus_ICE-diesel",
            "bus_gas-lpg",
            "bus_gas-natural",
            "bus_BEV",
        ],
    }
    dm_fleet_bus = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_fleet_bus.deepen()
    mapping_calc = {"ICE-gas": ["gas-lpg", "gas-natural"]}
    dm_fleet_bus.groupby(
        mapping_calc, dim="Categories1", aggregation="sum", regex=False, inplace=True
    )
    categories2_missing = categories2_all.copy()
    for cat in dm_fleet_bus.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_fleet_bus.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories1"
    )
    dm_fleet_bus.sort("Categories1")

    # Rail
    dict_extract = {
        "database": "Transport",
        "sheet": "TrRail_act",
        "variable": "Stock of vehicles - total (representative train configuration)",
        "sheet_last_row": "High speed passenger trains",
        "sub_variables": [
            "Metro and tram, urban light rail",
            "Diesel oil",
            "Electric",
            "High speed passenger trains",
        ],
        "calc_names": [
            "metrotram_mt",
            "train-conv_ICE-diesel",
            "train-conv_CEV",
            "train-hs_CEV",
        ],
    }
    dm_fleet_rail = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    mapping_calc = {
        "rail_CEV": ["train-conv_CEV", "train-hs_CEV"],
        "rail_ICE-diesel": ["train-conv_ICE-diesel"],
    }
    dm_fleet_rail.groupby(
        mapping_calc, dim="Variables", aggregation="sum", regex=False, inplace=True
    )
    dm_fleet_rail.deepen()
    categories2_missing = categories2_all.copy()
    for cat in dm_fleet_rail.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_fleet_rail.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories1"
    )
    dm_fleet_rail.sort("Categories1")
    dm_fleet_rail.units["metrotram"] = "vehicles"
    dm_fleet_rail.units["rail"] = "vehicles"

    # Aviation
    def get_aviation_data(country_code, country_name, row_start, row_end, unit):
        filepath_jrc = os.path.join(
            _THIS_DIR,
            "../../../industry/eu/data/JRC-IDEES-2021/EU27/JRC-IDEES-2021_x1990_Aviation_EU.xlsx",
        )
        df_temp = pd.read_excel(filepath_jrc, sheet_name=country_code)
        df_temp = df_temp.iloc[row_start:row_end, :]
        indexes = df_temp.columns[0]
        df_temp = pd.melt(df_temp, id_vars=indexes, var_name="year")
        df_temp.columns = ["Country", "Years", f"aviation_kerosene[{unit}]"]
        df_temp["Country"] = country_name
        return df_temp

    country_codes = list(dict_iso2_jrc.keys())
    country_names = list(dict_iso2_jrc.values())
    df_fleet_avi = pd.concat(
        [
            get_aviation_data(code, name, 92, 93, "unit")
            for code, name in zip(country_codes, country_names)
        ],
        ignore_index=True,
    )
    dm_fleet_avi = DataMatrix.create_from_df(df_fleet_avi, 0)
    dm_fleet_avi.array = np.round(dm_fleet_avi.array, 0)
    df_avi_seats = pd.concat(
        [
            get_aviation_data(code, name, 273, 274, "unit")
            for code, name in zip(country_codes, country_names)
        ],
        ignore_index=True,
    )
    dm_avi_seats = DataMatrix.create_from_df(df_avi_seats, 0)
    dm_avi_seats.array = np.round(dm_avi_seats.array, 0)
    dm_fleet_avi.array = dm_fleet_avi.array * dm_avi_seats.array
    dm_fleet_avi.deepen()
    categories2_missing = categories2_all.copy()
    for cat in dm_fleet_avi.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_fleet_avi.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories1"
    )
    dm_fleet_avi.sort("Categories1")

    # Combine fleet
    dm_fleet = dm_fleet_2w.copy()
    dm_fleet.append(dm_fleet_ldv, "Variables")
    dm_fleet.append(dm_fleet_bus, "Variables")
    dm_fleet.append(dm_fleet_rail, "Variables")
    dm_fleet.sort("Variables")
    dm_fleet.sort("Country")
    dm_fleet.add(np.nan, "Years", list(range(1990, 2000)), dummy=True)
    dm_fleet.sort("Years")
    dm_fleet.append(dm_fleet_avi, "Variables")
    dm_fleet.sort("Variables")
    dm_fleet = dm_fleet.flatten()
    idx = dm_fleet.idx
    dm_fleet.array[:, idx[2010], idx["LDV_ICE-diesel"]] = np.nan
    dm_fleet.deepen()
    for y in [2020, 2021]:
        dm_fleet[:, y, "aviation", "kerosene"] = np.nan
    for v in dm_fleet.col_labels["Variables"]:
        dm_fleet.units[v] = "number"

    # Fix OTS
    dm_fleet = dm_fleet.flatten()

    def make_ots(dm, variable, periods_dicts, years_ots_arg=None):
        dm_temp = dm.filter({"Variables": [variable]})
        if periods_dicts["n_adj"] == 1:
            dm_temp = linear_fitting(dm_temp, years_ots_arg, min_t0=0.1, min_tb=0.1)
        if periods_dicts["n_adj"] == 2:
            dm_temp = linear_fitting(
                dm_temp,
                list(range(startyear, 1999 + 1)),
                based_on=list(range(2000, periods_dicts["year_end_first_adj"] + 1)),
                min_t0=0.1,
                min_tb=0.1,
            )
            dm_temp = linear_fitting(
                dm_temp,
                list(range(2022, 2023 + 1)),
                based_on=list(range(periods_dicts["year_start_second_adj"], 2021 + 1)),
                min_t0=0.1,
                min_tb=0.1,
            )
        return dm_temp

    dict_call = {
        "2W_ICE-gasoline": {"n_adj": 1},
        "LDV_BEV": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2021,
        },
        "LDV_ICE-diesel": {
            "n_adj": 2,
            "year_end_first_adj": 2019,
            "year_start_second_adj": 2020,
        },
        "LDV_ICE-gas": {
            "n_adj": 2,
            "year_end_first_adj": 2019,
            "year_start_second_adj": 2020,
        },
        "LDV_ICE-gasoline": {
            "n_adj": 2,
            "year_end_first_adj": 2014,
            "year_start_second_adj": 2014,
        },
        "LDV_PHEV-diesel": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2021,
        },
        "LDV_PHEV-gasoline": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2021,
        },
        "bus_BEV": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2021,
        },
        "bus_ICE-diesel": {"n_adj": 1},
        "bus_ICE-gas": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2020,
        },
        "bus_ICE-gasoline": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2020,
        },
        "metrotram_mt": {"n_adj": 1},
        "rail_CEV": {"n_adj": 1},
        "rail_ICE-diesel": {"n_adj": 1},
        "aviation_kerosene": {"n_adj": 1},
    }
    dict_new = {}
    for key in dict_call.keys():
        if len(dict_call[key]) > 1:
            dict_new[key] = make_ots(dm_fleet, key, dict_call[key])
        else:
            dict_new[key] = make_ots(dm_fleet, key, dict_call[key], years_ots_ext)
    dm_fleet = dict_new["2W_ICE-gasoline"].copy()
    mylist = list(dict_call.keys())
    mylist.remove("2W_ICE-gasoline")
    for v in mylist:
        dm_fleet.append(dict_new[v], "Variables")
    dm_fleet.sort("Variables")

    # Fix 2022-2023 bus_BEV using growth rate (no good Eurostat source for buses)
    idx = dm_fleet.idx
    dm_fleet.array[idx["EU27"], idx[2022], idx["bus_BEV"]] = np.nan
    dm_fleet.array[idx["EU27"], idx[2023], idx["bus_BEV"]] = np.nan
    _rate_bus_bev_fleet = (
        dm_fleet.array[idx["EU27"], idx[2021], idx["bus_BEV"]]
        - dm_fleet.array[idx["EU27"], idx[2020], idx["bus_BEV"]]
    ) / dm_fleet.array[idx["EU27"], idx[2020], idx["bus_BEV"]]
    dm_fleet.array[idx["EU27"], idx[2022], idx["bus_BEV"]] = dm_fleet.array[
        idx["EU27"], idx[2021], idx["bus_BEV"]
    ] * (1 + _rate_bus_bev_fleet)
    dm_fleet.array[idx["EU27"], idx[2023], idx["bus_BEV"]] = dm_fleet.array[
        idx["EU27"], idx[2022], idx["bus_BEV"]
    ] * (1 + _rate_bus_bev_fleet)

    # Fix 2022-2023 LDV fleet via per-tech stock tracking.
    # For each tech: stock_t = stock_tm1 + new_t - waste_t.
    # new_t: ICE-diesel/gasoline from Eurostat DIE/PET; BEV/PHEV/ICE-gas split from 2020→2021 delta.
    # waste_t: proportional to existing stock (total waste = total_new - delta_total_fleet).
    _filter_ldv_fleet_fix = {
        "geo\\TIME_PERIOD": ["EU27_2020"],
        "engine": ["TOTAL"],
        "mot_nrg": ["TOTAL", "DIE", "PET"],
        "unit": "NR",
    }
    _dm_new_for_fleet = get_data_api_eurostat(
        "road_eqr_carmot",
        _filter_ldv_fleet_fix,
        {"Country": "geo\\TIME_PERIOD", "Variables": "mot_nrg"},
        "num",
    )
    idx = dm_fleet.idx
    idx_nf = _dm_new_for_fleet.idx
    idx_etf = dm_eurostat_fleet_total.idx
    _other_vars_fleet = [
        "LDV_BEV",
        "LDV_PHEV-diesel",
        "LDV_PHEV-gasoline",
        "LDV_ICE-gas",
    ]
    _all_ldv_vars = ["LDV_ICE-diesel", "LDV_ICE-gasoline"] + _other_vars_fleet
    # Estimate new vehicle shares for "other" bucket from 2020→2021 fleet deltas (EV scrappage ≈ 0)
    _delta_other_2021 = np.array(
        [
            max(
                0.0,
                dm_fleet.array[idx["EU27"], idx[2021], idx[v]]
                - dm_fleet.array[idx["EU27"], idx[2020], idx[v]],
            )
            for v in _other_vars_fleet
        ]
    )
    _delta_other_sum = np.nansum(_delta_other_2021)
    _other_new_shares_fleet = (
        _delta_other_2021 / _delta_other_sum
        if _delta_other_sum > 0
        else np.full(len(_other_vars_fleet), 1.0 / len(_other_vars_fleet))
    )
    # Initialise stock vector from 2021 JRC values
    _stock_t = np.array(
        [dm_fleet.array[idx["EU27"], idx[2021], idx[v]] for v in _all_ldv_vars]
    )
    for _fix_year in [2022, 2023]:
        if _fix_year not in idx_nf or _fix_year not in idx_etf:
            continue
        _fleet_total_t = float(
            dm_eurostat_fleet_total.array[idx_etf["EU27"], idx_etf[_fix_year], 0]
        )
        _fleet_total_tm1 = float(
            dm_eurostat_fleet_total.array[idx_etf["EU27"], idx_etf[_fix_year - 1], 0]
        )
        _total_new = float(
            _dm_new_for_fleet.array[idx_nf["EU27"], idx_nf[_fix_year], idx_nf["TOTAL"]]
        )
        _die_new = float(
            _dm_new_for_fleet.array[idx_nf["EU27"], idx_nf[_fix_year], idx_nf["DIE"]]
        )
        _pet_new = float(
            _dm_new_for_fleet.array[idx_nf["EU27"], idx_nf[_fix_year], idx_nf["PET"]]
        )
        _other_new = max(0.0, _total_new - _die_new - _pet_new)
        # Total waste = new registrations - net fleet growth
        _total_waste = max(0.0, _total_new - (_fleet_total_t - _fleet_total_tm1))
        # Waste per tech proportional to existing stock
        _stock_sum = max(float(np.nansum(_stock_t)), 1.0)
        _waste_t = _total_waste * (_stock_t / _stock_sum)
        # New per tech: ICE-diesel/gasoline directly from Eurostat, others split by 2021 delta shares
        _new_t = np.array(
            [_die_new, _pet_new] + list(_other_new * _other_new_shares_fleet)
        )
        # Update and clamp stock
        _stock_t = np.maximum(0.0, _stock_t + _new_t - _waste_t)
        for _v, _val in zip(_all_ldv_vars, _stock_t):
            dm_fleet.array[idx["EU27"], idx[_fix_year], idx[_v]] = _val

    # Format as final DataMatrix
    dm_fleet.drop("Variables", "total")
    for v in dm_fleet.col_labels["Variables"]:
        dm_fleet.rename_col(v, "tra_passenger_technology-share_fleet_" + v, "Variables")
    dm_fleet.deepen_twice()
    dm_fleet_final = dm_fleet.copy()
    dm_fleet_final.add(np.nan, "Categories2", "H2", "number", True)
    dm_fleet_final.sort("Categories2")

    ###########################################################################
    # VEHICLE EFFICIENCY
    ###########################################################################

    categories2_all = _CATEGORIES2_ALL.copy()
    dict_iso2 = eurostat_iso2_dict()
    dict_iso2.pop("CH")
    dict_iso2_jrc = jrc_iso2_dict()

    # 2W
    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_ene",
        "variable": "Vehicle-efficiency - effective (kgoe/100 km)",
        "sheet_last_row": "Powered two-wheelers",
        "sub_variables": ["Powered two-wheelers"],
        "calc_names": ["2W"],
    }
    dm_eneff_2w = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_eneff_2w.rename_col("2W", "2W_ICE-gasoline", "Variables")
    dm_eneff_2w.deepen()
    categories2_missing = categories2_all.copy()
    for cat in dm_eneff_2w.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_eneff_2w.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories1"
    )
    dm_eneff_2w.sort("Categories1")

    # LDV
    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_ene",
        "variable": "Vehicle-efficiency - effective (kgoe/100 km)",
        "sheet_last_row": "Battery electric vehicles",
        "categories": "Passenger cars",
        "sub_variables": [
            "Gasoline engine",
            "Diesel oil engine",
            "LPG engine",
            "Natural gas engine",
            "Plug-in hybrid electric",
            "Battery electric vehicles",
        ],
        "calc_names": [
            "LDV_ICE-gasoline",
            "LDV_ICE-diesel",
            "LDV_gas-lpg",
            "LDV_gas-natural",
            "LDV_PHEV",
            "LDV_BEV",
        ],
    }
    dm_eneff_ldv = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_eneff_ldv.deepen()
    mapping_calc = {"ICE-gas": ["gas-lpg", "gas-natural"]}
    dm_eneff_ldv.groupby(
        mapping_calc, dim="Categories1", aggregation="mean", regex=False, inplace=True
    )
    dm_temp = dm_eneff_ldv.filter({"Categories1": ["PHEV"]})
    dm_temp1 = dm_temp.copy()
    dm_temp1.rename_col("PHEV", "PHEV-gasoline", "Categories1")
    dm_temp1.append(dm_temp, "Categories1")
    dm_temp1.rename_col("PHEV", "PHEV-diesel", "Categories1")
    dm_eneff_ldv.drop("Categories1", "PHEV")
    dm_eneff_ldv.append(dm_temp1, "Categories1")
    dm_eneff_ldv.sort("Categories1")
    categories2_missing = categories2_all.copy()
    for cat in dm_eneff_ldv.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_eneff_ldv.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories1"
    )
    dm_eneff_ldv.sort("Categories1")

    # Buses
    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_ene",
        "variable": "Vehicle-efficiency - effective (kgoe/100 km)",
        "sheet_last_row": "Battery electric vehicles",
        "sub_variables": [
            "Gasoline engine",
            "Diesel oil engine",
            "LPG engine",
            "Natural gas engine",
            "Battery electric vehicles",
        ],
        "calc_names": [
            "bus_ICE-gasoline",
            "bus_ICE-diesel",
            "bus_gas-lpg",
            "bus_gas-natural",
            "bus_BEV",
        ],
        "categories": "Motor coaches, buses and trolley buses",
    }
    dm_eneff_bus = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_eneff_bus.deepen()
    mapping_calc = {"ICE-gas": ["gas-lpg", "gas-natural"]}
    dm_eneff_bus.groupby(
        mapping_calc, dim="Categories1", aggregation="mean", regex=False, inplace=True
    )
    categories2_missing = categories2_all.copy()
    for cat in dm_eneff_bus.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_eneff_bus.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories1"
    )
    dm_eneff_bus.sort("Categories1")

    # Rail
    dict_extract = {
        "database": "Transport",
        "sheet": "TrRail_ene",
        "variable": "Vehicle-efficiency (kgoe/100 km)",
        "sheet_last_row": "High speed passenger trains",
        "sub_variables": [
            "Metro and tram, urban light rail",
            "Diesel oil",
            "Electric",
            "High speed passenger trains",
        ],
        "calc_names": [
            "metrotram_mt",
            "train-conv_ICE-diesel",
            "train-conv_CEV",
            "train-hs_CEV",
        ],
    }
    dm_eneff_rail = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    mapping_calc = {
        "rail_CEV": ["train-conv_CEV", "train-hs_CEV"],
        "rail_ICE-diesel": ["train-conv_ICE-diesel"],
    }
    dm_eneff_rail.groupby(
        mapping_calc, dim="Variables", aggregation="mean", regex=False, inplace=True
    )
    dm_eneff_rail.deepen()
    categories2_missing = categories2_all.copy()
    for cat in dm_eneff_rail.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_eneff_rail.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories1"
    )
    dm_eneff_rail.sort("Categories1")

    # Aviation
    country_codes = list(dict_iso2_jrc.keys())
    country_names = list(dict_iso2_jrc.values())
    df_temp = pd.concat(
        [
            get_aviation_data(code, name, 286, 287, "kgoe/seat-km")
            for code, name in zip(country_codes, country_names)
        ],
        ignore_index=True,
    )
    dm_eneff_avi = DataMatrix.create_from_df(df_temp, 0)
    dm_eneff_avi.change_unit("aviation_kerosene", 100, "kgoe/seat-km", "kgoe/100 km")
    dm_eneff_avi.deepen()
    categories2_missing = categories2_all.copy()
    for cat in dm_eneff_avi.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_eneff_avi.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories1"
    )
    dm_eneff_avi.sort("Categories1")

    # Combine efficiency
    dm_eneff = dm_eneff_2w.copy()
    dm_eneff.append(dm_eneff_ldv, "Variables")
    dm_eneff.append(dm_eneff_bus, "Variables")
    dm_eneff.append(dm_eneff_rail, "Variables")
    dm_eneff.sort("Variables")
    dm_eneff.sort("Country")
    dm_eneff.add(np.nan, "Years", list(range(1990, 2000)), dummy=True)
    dm_eneff.sort("Years")
    dm_eneff.append(dm_eneff_avi, "Variables")
    dm_eneff.sort("Variables")
    dm_eneff.array[dm_eneff.array == 0] = np.nan

    # Fix OTS
    dm_eneff_bus_icedie = dm_eneff.filter(
        {"Variables": ["bus"], "Categories1": ["ICE-diesel"]}
    )
    dm_eneff = linear_fitting(dm_eneff, years_ots_ext, min_t0=0, min_tb=0)
    dm_eneff_bus_icedie = linear_fitting(
        dm_eneff_bus_icedie,
        list(range(startyear, 1999 + 1)),
        based_on=list(range(2000, 2012 + 1)),
        min_t0=0.1,
        min_tb=0.1,
    )
    dm_eneff_bus_icedie = linear_fitting(
        dm_eneff_bus_icedie,
        list(range(2022, 2023 + 1)),
        based_on=list(range(2012, 2020 + 1)),
        min_t0=0.1,
        min_tb=0.1,
    )
    dm_eneff = dm_eneff.flatten()
    dm_eneff.drop("Variables", "bus_ICE-diesel")
    dm_eneff.append(dm_eneff_bus_icedie.flatten(), "Variables")
    dm_eneff.deepen()

    # Format as final DataMatrix
    for v in dm_eneff.col_labels["Variables"]:
        dm_eneff.rename_col(v, "tra_passenger_veh-efficiency_fleet_" + v, "Variables")
    dm_eneff.deepen(based_on="Variables")
    dm_eneff.switch_categories_order("Categories1", "Categories2")
    dm_eneff.change_unit(
        "tra_passenger_veh-efficiency_fleet", 41.868, "kgoe/100 km", "MJ/100 km"
    )
    dm_eneff.change_unit(
        "tra_passenger_veh-efficiency_fleet", 1e-2, "MJ/100 km", "MJ/km"
    )
    dm_eneff_final = dm_eneff.copy()
    dm_eneff_final.add(np.nan, "Categories2", "H2", "number", True)
    dm_eneff_final.sort("Categories2")

    ###########################################################################
    # NEW VEHICLES
    ###########################################################################

    categories2_all = _CATEGORIES2_ALL.copy()
    dict_iso2 = eurostat_iso2_dict()
    dict_iso2.pop("CH")
    dict_iso2_jrc = jrc_iso2_dict()

    # 2W
    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_act",
        "variable": "New vehicle-registrations",
        "sheet_last_row": "Powered two-wheelers",
        "sub_variables": ["Powered two-wheelers"],
        "calc_names": ["2W"],
    }
    dm_new_2w = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    # Scale EU27 2W new registrations by the same Eurostat/JRC fleet ratio
    _idx_new2w = dm_new_2w.idx
    _idx_jrc2w = dm_fleet_2w_jrc.idx
    for _i, _y in enumerate(_moto_years):
        if _y in dm_new_2w.col_labels["Years"]:
            _jrc_val = dm_fleet_2w_jrc.array[
                _idx_jrc2w["EU27"], _idx_jrc2w[_y], _idx_jrc2w["2W"]
            ]
            if not np.isnan(_jrc_val) and _jrc_val > 0:
                _ratio = float(_eu27_moto[_i, 0]) / _jrc_val
                dm_new_2w.array[
                    _idx_new2w["EU27"], _idx_new2w[_y], _idx_new2w["2W"]
                ] *= _ratio
    dm_new_2w.rename_col("2W", "2W_ICE-gasoline", "Variables")
    dm_new_2w.deepen()
    categories2_missing = categories2_all.copy()
    for cat in dm_new_2w.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_new_2w.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
    dm_new_2w.sort("Categories1")

    # LDV new - total from Eurostat
    code = "road_eqr_carmot"
    dm_eurostat_new_total = get_data_api_eurostat(
        code, filter_eurostat, mapping_dim, "num"
    )
    dm_eurostat_new = dm_eurostat_new_total.filter(
        {"Years": list(range(2000, 2021 + 1, 1))}
    )
    dm_eurostat_new.drop("Country", "United Kingdom")

    # LDV new - tech split from JRC
    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_act",
        "variable": "New vehicle-registrations",
        "sheet_last_row": "Battery electric vehicles",
        "categories": "Passenger cars",
        "sub_variables": [
            "Gasoline engine",
            "Diesel oil engine",
            "LPG engine",
            "Natural gas engine",
            "Plug-in hybrid electric",
            "Battery electric vehicles",
        ],
        "calc_names": [
            "LDV_ICE-gasoline",
            "LDV_ICE-diesel",
            "LDV_gas-lpg",
            "LDV_gas-natural",
            "LDV_PHEV",
            "LDV_BEV",
        ],
    }
    dm_new_ldv = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_new_ldv.deepen()
    mapping_calc = {"ICE-gas": ["gas-lpg", "gas-natural"]}
    dm_new_ldv.groupby(
        mapping_calc, dim="Categories1", aggregation="sum", regex=False, inplace=True
    )
    dm_temp = dm_new_ldv.filter({"Categories1": ["PHEV"]})
    dm_temp.rename_col("PHEV", "PHEV-gasoline", "Categories1")
    dm_temp.add(dm_temp.array, col_label="PHEV-diesel", dim="Categories1")
    dm_temp.sort("Categories1")
    dm_temp1 = dm_new_ldv.filter({"Categories1": ["ICE-diesel", "ICE-gasoline"]})
    dm_temp1.normalise("Categories1")
    dm_temp.array = dm_temp.array * dm_temp1.array
    dm_new_ldv.drop("Categories1", "PHEV")
    dm_new_ldv.append(dm_temp, "Categories1")
    dm_new_ldv.sort("Categories1")
    dm_temp = dm_new_ldv.normalise("Categories1", inplace=False)
    dm_temp.array = dm_eurostat_new.array[..., np.newaxis] * dm_temp.array
    dm_temp.rename_col("LDV_share", "LDV", "Variables")
    dm_temp.units["LDV"] = "num"
    idx = dm_temp.idx
    for y in range(2000, 2009 + 1, 1):
        dm_temp.array[:, idx[y], :, :] = np.nan
    dm_new_ldv = dm_temp.copy()
    categories2_missing = categories2_all.copy()
    for cat in dm_new_ldv.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_new_ldv.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
    dm_new_ldv.sort("Categories1")

    # Buses new
    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_act",
        "variable": "New vehicle-registrations",
        "categories": "Motor coaches, buses and trolley buses",
        "sheet_last_row": "Battery electric vehicles",
        "sub_variables": [
            "Gasoline engine",
            "Diesel oil engine",
            "LPG engine",
            "Natural gas engine",
            "Battery electric vehicles",
        ],
        "calc_names": [
            "bus_ICE-gasoline",
            "bus_ICE-diesel",
            "bus_gas-lpg",
            "bus_gas-natural",
            "bus_BEV",
        ],
    }
    dm_new_bus = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_new_bus.deepen()
    mapping_calc = {"ICE-gas": ["gas-lpg", "gas-natural"]}
    dm_new_bus.groupby(
        mapping_calc, dim="Categories1", aggregation="sum", regex=False, inplace=True
    )
    categories2_missing = categories2_all.copy()
    for cat in dm_new_bus.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_new_bus.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
    dm_new_bus.sort("Categories1")

    # Rail new
    dict_extract = {
        "database": "Transport",
        "sheet": "TrRail_act",
        "variable": "New vehicles - total (representative train configuration)",
        "sheet_last_row": "High speed passenger trains",
        "sub_variables": [
            "Metro and tram, urban light rail",
            "Diesel oil",
            "Electric",
            "High speed passenger trains",
        ],
        "calc_names": [
            "metrotram_mt",
            "train-conv_ICE-diesel",
            "train-conv_CEV",
            "train-hs_CEV",
        ],
    }
    dm_new_rail = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    mapping_calc = {
        "rail_CEV": ["train-conv_CEV", "train-hs_CEV"],
        "rail_ICE-diesel": ["train-conv_ICE-diesel"],
    }
    dm_new_rail.groupby(
        mapping_calc, dim="Variables", aggregation="sum", regex=False, inplace=True
    )
    dm_new_rail.deepen()
    categories2_missing = categories2_all.copy()
    for cat in dm_new_rail.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_new_rail.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories1"
    )
    dm_new_rail.sort("Categories1")
    dm_new_rail.units["metrotram"] = "vehicles"
    dm_new_rail.units["rail"] = "vehicles"

    # Aviation new
    country_codes = list(dict_iso2_jrc.keys())
    country_names = list(dict_iso2_jrc.values())
    df_temp = pd.concat(
        [
            get_aviation_data(code, name, 102, 103, "unit")
            for code, name in zip(country_codes, country_names)
        ],
        ignore_index=True,
    )
    dm_new_avi = DataMatrix.create_from_df(df_temp, 0)
    dm_new_avi.array = np.round(dm_new_avi.array, 0)
    dm_new_avi.array = dm_new_avi.array * dm_avi_seats.array
    dm_new_avi.rename_col("aviation", "aviation_kerosene", "Variables")
    dm_new_avi.deepen()
    categories2_missing = categories2_all.copy()
    for cat in dm_new_avi.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_new_avi.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
    dm_new_avi.sort("Categories1")

    # Combine new vehicles
    dm_new = dm_new_2w.copy()
    dm_new.append(dm_new_ldv, "Variables")
    dm_new.append(dm_new_bus, "Variables")
    dm_new.append(dm_new_rail, "Variables")
    dm_new.sort("Variables")
    dm_new.sort("Country")
    dm_new.add(np.nan, "Years", list(range(1990, 2000)), dummy=True)
    dm_new.sort("Years")
    dm_new.append(dm_new_avi, "Variables")
    dm_new.sort("Variables")
    for v in dm_new.col_labels["Variables"]:
        dm_new.units[v] = "number"
    dm_new[:, 1990, "aviation", "kerosene"] = np.nan

    # Fix OTS
    dm_new = dm_new.flatten()
    idx = dm_new.idx
    dm_new.array[idx["EU27"], idx[2000], idx["LDV_ICE-gasoline"]] = dm_new.array[
        idx["EU27"], idx[2019], idx["LDV_ICE-gasoline"]
    ]

    dict_call = {
        "2W_ICE-gasoline": {
            "n_adj": 2,
            "year_end_first_adj": 2007,
            "year_start_second_adj": 2021,
        },
        "LDV_BEV": {
            "n_adj": 2,
            "year_end_first_adj": 2013,
            "year_start_second_adj": 2021,
        },
        "LDV_ICE-diesel": {
            "n_adj": 2,
            "year_end_first_adj": 2016,
            "year_start_second_adj": 2017,
        },
        "LDV_ICE-gas": {
            "n_adj": 2,
            "year_end_first_adj": 2019,
            "year_start_second_adj": 2020,
        },
        "LDV_ICE-gasoline": {
            "n_adj": 2,
            "year_end_first_adj": 2013,
            "year_start_second_adj": 2020,
        },
        "LDV_PHEV-diesel": {
            "n_adj": 2,
            "year_end_first_adj": 2013,
            "year_start_second_adj": 2021,
        },
        "LDV_PHEV-gasoline": {
            "n_adj": 2,
            "year_end_first_adj": 2013,
            "year_start_second_adj": 2021,
        },
        "bus_BEV": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2021,
        },
        "bus_ICE-diesel": {
            "n_adj": 2,
            "year_end_first_adj": 2008,
            "year_start_second_adj": 2020,
        },
        "bus_ICE-gas": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2020,
        },
        "bus_ICE-gasoline": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2020,
        },
        "metrotram_mt": {"n_adj": 1},
        "rail_CEV": {
            "n_adj": 2,
            "year_end_first_adj": 2000,
            "year_start_second_adj": 2005,
        },
        "rail_ICE-diesel": {"n_adj": 1},
        "aviation_kerosene": {"n_adj": 1},
    }
    dict_new = {}
    for key in dict_call.keys():
        if len(dict_call[key]) > 1:
            dict_new[key] = make_ots(dm_new, key, dict_call[key])
        else:
            dict_new[key] = make_ots(dm_new, key, dict_call[key], years_ots_ext)
    dm_new = dict_new["2W_ICE-gasoline"].copy()
    mylist = list(dict_call.keys())
    mylist.remove("2W_ICE-gasoline")
    for v in mylist:
        dm_new.append(dict_new[v], "Variables")
    dm_new.sort("Variables")

    # Fix 2022-2023 bus_BEV using growth rate (no good Eurostat source for buses)
    idx = dm_new.idx
    dm_new.array[idx["EU27"], idx[2022], idx["bus_BEV"]] = np.nan
    dm_new.array[idx["EU27"], idx[2023], idx["bus_BEV"]] = np.nan
    _rate_bus_bev = (
        dm_new.array[idx["EU27"], idx[2021], idx["bus_BEV"]]
        - dm_new.array[idx["EU27"], idx[2020], idx["bus_BEV"]]
    ) / dm_new.array[idx["EU27"], idx[2020], idx["bus_BEV"]]
    dm_new.array[idx["EU27"], idx[2022], idx["bus_BEV"]] = dm_new.array[
        idx["EU27"], idx[2021], idx["bus_BEV"]
    ] * (1 + _rate_bus_bev)
    dm_new.array[idx["EU27"], idx[2023], idx["bus_BEV"]] = dm_new.array[
        idx["EU27"], idx[2022], idx["bus_BEV"]
    ] * (1 + _rate_bus_bev)

    # Fix 2022-2023 LDV new vehicles using Eurostat road_eqr_carmot
    # ICE-diesel = DIE, ICE-gasoline = PET directly from Eurostat.
    # Remainder (TOTAL - DIE - PET) = BEV + PHEV + ICE-gas, split using 2021 JRC shares.
    # This avoids the over-extrapolation of BEV/PHEV growth rates from the COVID-rebound year 2021.
    _filter_ldv_new_tech = {
        "geo\\TIME_PERIOD": ["EU27_2020"],
        "engine": ["TOTAL"],
        "mot_nrg": ["TOTAL", "DIE", "PET"],
        "unit": "NR",
    }
    _dm_ldv_new_tech = get_data_api_eurostat(
        "road_eqr_carmot",
        _filter_ldv_new_tech,
        {"Country": "geo\\TIME_PERIOD", "Variables": "mot_nrg"},
        "num",
    )
    idx = dm_new.idx
    idx_e = _dm_ldv_new_tech.idx
    _other_vars = ["LDV_BEV", "LDV_PHEV-diesel", "LDV_PHEV-gasoline", "LDV_ICE-gas"]
    _other_2021 = np.array(
        [dm_new.array[idx["EU27"], idx[2021], idx[v]] for v in _other_vars]
    )
    _other_2021_sum = np.nansum(_other_2021)
    _other_shares = (
        _other_2021 / _other_2021_sum
        if _other_2021_sum > 0
        else np.full(len(_other_vars), 1 / len(_other_vars))
    )
    for _fix_year in [2022, 2023]:
        if _fix_year not in idx_e:
            continue
        _total = float(
            _dm_ldv_new_tech.array[idx_e["EU27"], idx_e[_fix_year], idx_e["TOTAL"]]
        )
        _die = float(
            _dm_ldv_new_tech.array[idx_e["EU27"], idx_e[_fix_year], idx_e["DIE"]]
        )
        _pet = float(
            _dm_ldv_new_tech.array[idx_e["EU27"], idx_e[_fix_year], idx_e["PET"]]
        )
        _other = max(0.0, _total - _die - _pet)
        dm_new.array[idx["EU27"], idx[_fix_year], idx["LDV_ICE-diesel"]] = _die
        dm_new.array[idx["EU27"], idx[_fix_year], idx["LDV_ICE-gasoline"]] = _pet
        for _v, _sh in zip(_other_vars, _other_shares):
            dm_new.array[idx["EU27"], idx[_fix_year], idx[_v]] = max(0.0, _other * _sh)

    # Zero-out early BEV/PHEV years (pre-2013)
    variabs = ["LDV_BEV", "LDV_PHEV-diesel", "LDV_PHEV-gasoline"]
    idx = dm_new.idx
    for y in range(startyear, 2013 + 1, 1):
        for v in variabs:
            dm_new.array[:, idx[y], idx[v]] = 0

    # Format as final DataMatrix
    for v in dm_new.col_labels["Variables"]:
        dm_new.rename_col(v, "tra_passenger_new-vehicles_" + v, "Variables")
    dm_new.deepen_twice()
    dm_new_final = dm_new.copy()
    dm_new_final.add(np.nan, "Categories2", "H2", "number", True)
    dm_new_final.sort("Categories2")

    ###########################################################################
    # WASTE & RENEWAL RATE
    ###########################################################################

    dm_tech = dm_fleet_final.copy()
    dm_tech.append(dm_eneff_final, "Variables")
    dm_tech.append(dm_new_final, "Variables")

    dm_temp = dm_tech.filter({"Variables": ["tra_passenger_veh-efficiency_fleet"]})

    def compute_renewal_rate_and_adjust(dm, var_names, max_rr):
        s_col = var_names["stock"]
        new_col = var_names["new"]
        waste_col = var_names["waste"]
        rr_col = var_names["renewal-rate"]
        stock_unit = dm.units[s_col]

        dm.lag_variable(pattern=s_col, shift=1, subfix="_tm1")
        dm.operation(
            s_col + "_tm1", "-", s_col, out_col="tra_delta_stock", unit=stock_unit
        )
        dm.operation(
            "tra_delta_stock", "+", new_col, out_col=waste_col, unit=stock_unit
        )
        dm.operation(waste_col, "/", s_col + "_tm1", out_col="tmp", unit="%")
        dm.lag_variable(pattern="tmp", shift=-1, subfix="_rr")
        dm.rename_col("tmp_rr", rr_col, dim="Variables")
        dm.filter({"Variables": [s_col, s_col + "_tm1", rr_col]}, inplace=True)

        dm_rr = dm.filter({"Variables": [rr_col]}, inplace=False)
        mask = (dm_rr.array < 0) | (dm_rr.array > max_rr)
        dm_rr.array[mask] = np.nan
        dm_rr.fill_nans("Years")
        dm.drop(dim="Variables", col_label=rr_col)
        dm.append(dm_rr, dim="Variables")

        dm.lag_variable(pattern=rr_col, shift=1, subfix="_tm1")
        dm.operation(
            rr_col + "_tm1", "*", s_col + "_tm1", out_col=waste_col, unit=stock_unit
        )
        dm.operation(
            s_col, "-", s_col + "_tm1", out_col="tra_delta_stock", unit=stock_unit
        )
        dm.operation(
            "tra_delta_stock", "+", waste_col, out_col=new_col, unit=stock_unit
        )
        dm.filter({"Variables": [s_col, new_col, waste_col, rr_col]}, inplace=True)

        dm_new_var = dm.filter({"Variables": [new_col]}, inplace=False)
        mask = dm_new_var.array < 0
        dm_new_var.array[mask] = np.nan
        dm_new_var.fill_nans("Years")
        dm.drop(dim="Variables", col_label=new_col)
        dm.append(dm_new_var, dim="Variables")

        idx = dm.idx
        for t in dm.col_labels["Years"][1:]:
            s_tm1 = dm.array[:, idx[t - 1], idx[s_col], ...]
            new_t = dm.array[:, idx[t], idx[new_col], ...]
            waste_t = dm.array[:, idx[t], idx[waste_col], ...]
            dm.array[:, idx[t], idx[s_col], ...] = s_tm1 + new_t - waste_t

    var_names = {
        "stock": "tra_passenger_technology-share_fleet",
        "new": "tra_passenger_new-vehicles",
        "waste": "tra_passenger_vehicle-waste",
        "renewal-rate": "tra_passenger_renewal-rate",
    }
    compute_renewal_rate_and_adjust(dm_tech, var_names, max_rr=0.1)
    dm_renrate = dm_tech.filter({"Variables": ["tra_passenger_renewal-rate"]})
    dm_tech.drop("Variables", "tra_passenger_renewal-rate")
    dm_tech.append(dm_temp, "Variables")
    dm_tech.sort("Variables")

    ###########################################################################
    # FTS FOR FLEET
    ###########################################################################

    dm_fleet = dm_tech.filter({"Variables": ["tra_passenger_technology-share_fleet"]})
    dm_fleet = dm_fleet.flatten().flatten()
    dm_fleet.rename_col_regex("tra_passenger_technology-share_fleet_", "", "Variables")
    dm_fleet.add(np.nan, col_label=years_fts, dummy=True, dim="Years")

    def make_fts(
        dm,
        variable,
        year_start=2023,
        year_end=2050,
        country="EU27",
        dim="Categories1",
        min_t0=0,
        min_tb=0,
    ):
        dm = dm.copy()
        idx = dm.idx
        based_on_yars = list(range(year_start, year_end + 1, 1))
        dm_temp = linear_fitting(
            dm.filter({"Country": [country], dim: [variable]}),
            years_ots=years_fts,
            min_t0=min_t0,
            min_tb=min_tb,
            based_on=based_on_yars,
        )
        idx_temp = dm_temp.idx
        if dim == "Variables":
            dm.array[idx[country], :, idx[variable], ...] = dm_temp.array[
                idx_temp[country], :, idx_temp[variable], ...
            ]
        if dim == "Categories1":
            dm.array[idx[country], :, :, idx[variable]] = dm_temp.array[
                idx_temp[country], :, :, idx_temp[variable]
            ]
        if dim == "Categories2":
            dm.array[idx[country], :, :, :, idx[variable]] = dm_temp.array[
                idx_temp[country], :, :, :, idx_temp[variable]
            ]
        if dim == "Categories3":
            dm.array[idx[country], :, :, :, :, idx[variable]] = dm_temp.array[
                idx_temp[country], :, :, :, :, idx_temp[variable]
            ]
        return dm

    # LDV 2050 targets
    idx = dm_fleet.idx
    total_2050 = 300000000
    dm_fleet.array[idx["EU27"], idx[2050], idx["LDV_BEV"]] = total_2050 * 0.32
    phev_total_2023 = (
        dm_fleet.array[idx["EU27"], idx[2023], idx["LDV_PHEV-diesel"]]
        + dm_fleet.array[idx["EU27"], idx[2023], idx["LDV_PHEV-gasoline"]]
    )
    dm_fleet.array[idx["EU27"], idx[2050], idx["LDV_PHEV-diesel"]] = (
        total_2050
        * 0.18
        * dm_fleet.array[idx["EU27"], idx[2023], idx["LDV_PHEV-diesel"]]
        / phev_total_2023
    )
    dm_fleet.array[idx["EU27"], idx[2050], idx["LDV_PHEV-gasoline"]] = (
        total_2050
        * 0.18
        * dm_fleet.array[idx["EU27"], idx[2023], idx["LDV_PHEV-gasoline"]]
        / phev_total_2023
    )
    dm_fleet.array[idx["EU27"], idx[2050], idx["LDV_ICE-gasoline"]] = total_2050 * 0.16
    dm_fleet.array[idx["EU27"], idx[2050], idx["LDV_ICE-diesel"]] = total_2050 * 0.27
    dm_fleet.array[idx["EU27"], idx[2050], idx["LDV_ICE-gas"]] = total_2050 * 0.04
    dm_fleet.add(0, "Variables", "LDV_FCEV", dummy=True)
    idx = dm_fleet.idx
    for y in range(2030, 2055, 5):
        dm_fleet.array[idx["EU27"], idx[y], idx["LDV_FCEV"]] = np.nan
    dm_fleet.array[idx["EU27"], idx[2050], idx["LDV_FCEV"]] = total_2050 * 0.03
    dm_fleet = make_fts(dm_fleet, "LDV_BEV", dim="Variables")
    dm_fleet = make_fts(dm_fleet, "LDV_PHEV-diesel", dim="Variables")
    dm_fleet = make_fts(dm_fleet, "LDV_PHEV-gasoline", dim="Variables")
    dm_fleet = make_fts(dm_fleet, "LDV_ICE-gasoline", dim="Variables")
    dm_fleet = make_fts(dm_fleet, "LDV_ICE-diesel", dim="Variables")
    dm_fleet = make_fts(dm_fleet, "LDV_ICE-gas", dim="Variables")
    dm_fleet = make_fts(dm_fleet, "LDV_FCEV", dim="Variables")

    # Bus 2050 targets
    dm_fleet.append(
        dm_fleet.groupby(
            {"bus_total": "bus"},
            dim="Variables",
            aggregation="sum",
            regex=True,
            inplace=False,
        ),
        "Variables",
    )
    idx = dm_fleet.idx
    for y in range(2025, 2055, 5):
        dm_fleet.array[..., idx[y], idx["bus_total"]] = np.nan
    dm_fleet = make_fts(
        dm_fleet, "bus_total", year_start=2013, year_end=2023, dim="Variables"
    )
    idx = dm_fleet.idx
    bus_total_2050 = dm_fleet.array[idx["EU27"], idx[2050], idx["bus_total"]]
    dm_fleet.array[idx["EU27"], idx[2050], idx["bus_BEV"]] = 0.03 * bus_total_2050
    dm_fleet.array[idx["EU27"], idx[2050], idx["bus_ICE-gas"]] = 0.18 * bus_total_2050
    total_bus_ice_2023 = (
        dm_fleet.array[idx["EU27"], idx[2023], idx["bus_ICE-diesel"]]
        + dm_fleet.array[idx["EU27"], idx[2023], idx["bus_ICE-gasoline"]]
    )
    dm_fleet.array[idx["EU27"], idx[2050], idx["bus_ICE-diesel"]] = (
        0.78
        * bus_total_2050
        * dm_fleet.array[idx["EU27"], idx[2023], idx["bus_ICE-diesel"]]
        / total_bus_ice_2023
    )
    dm_fleet.array[idx["EU27"], idx[2050], idx["bus_ICE-gasoline"]] = (
        0.78
        * bus_total_2050
        * dm_fleet.array[idx["EU27"], idx[2023], idx["bus_ICE-gasoline"]]
        / total_bus_ice_2023
    )
    dm_fleet = make_fts(dm_fleet, "bus_BEV", dim="Variables")
    dm_fleet = make_fts(dm_fleet, "bus_ICE-gasoline", dim="Variables")
    dm_fleet = make_fts(dm_fleet, "bus_ICE-diesel", dim="Variables")
    dm_fleet = make_fts(dm_fleet, "bus_ICE-gas", dim="Variables")
    dm_fleet.drop("Variables", ["bus_total"])
    dm_fleet.sort("Variables")

    # Other modes
    baseyear_start = 2000
    baseyear_end = 2023
    dm_fleet = make_fts(
        dm_fleet, "2W_ICE-gasoline", baseyear_start, baseyear_end, dim="Variables"
    )
    dm_fleet = make_fts(
        dm_fleet, "metrotram_mt", baseyear_start, baseyear_end, dim="Variables"
    )
    dm_fleet = make_fts(
        dm_fleet, "rail_CEV", baseyear_start, baseyear_end, dim="Variables"
    )
    dm_fleet = make_fts(
        dm_fleet, "rail_ICE-diesel", baseyear_start, baseyear_end, dim="Variables"
    )
    dm_fleet = make_fts(
        dm_fleet, "aviation_kerosene", baseyear_start, baseyear_end, dim="Variables"
    )

    ###########################################################################
    # FTS FOR NEW VEHICLES
    ###########################################################################

    dm_new = dm_tech.filter({"Variables": ["tra_passenger_new-vehicles"]})
    dm_new = dm_new.flatten().flatten()
    dm_new.rename_col_regex("tra_passenger_new-vehicles_", "", "Variables")
    dm_new.add(np.nan, col_label=years_fts, dummy=True, dim="Years")
    dm_new.add(0, col_label="LDV_FCEV", dummy=True, dim="Variables")
    dm_new.sort("Variables")

    variables = dm_new.col_labels["Variables"]
    for v in variables:
        idx_new = dm_new.idx
        idx_fleet = dm_fleet.idx
        rate = (
            dm_fleet.array[idx_fleet["EU27"], idx_fleet[2050], idx_fleet[v]]
            - dm_fleet.array[idx_fleet["EU27"], idx_fleet[2023], idx_fleet[v]]
        ) / dm_fleet.array[idx_fleet["EU27"], idx_fleet[2023], idx_fleet[v]]
        dm_new.array[idx_new["EU27"], idx_new[2050], idx_new[v]] = dm_new.array[
            idx_new["EU27"], idx_new[2023], idx_new[v]
        ] * (1 + rate)
        dm_new = make_fts(dm_new, v, dim="Variables")
    dm_new.array[dm_new.array < 0] = 0

    # Fix FCEV
    idx = dm_new.idx
    idx_fleet = dm_fleet.idx
    dm_new.array[idx["EU27"], idx[2050], idx["LDV_FCEV"]] = dm_fleet.array[
        idx_fleet["EU27"], idx_fleet[2050], idx_fleet["LDV_FCEV"]
    ]
    for y in range(2030, 2050, 5):
        dm_new.array[idx["EU27"], idx[y], idx["LDV_FCEV"]] = np.nan
    dm_new = make_fts(dm_new, "LDV_FCEV", dim="Variables")

    ###########################################################################
    # DROP 1989 & FINALISE dm_tech
    ###########################################################################

    dm_tech.drop("Years", startyear)
    dm_renrate.drop("Years", startyear)
    dm_new.drop("Years", startyear)
    dm_fleet.drop("Years", startyear)

    # Add FCEV to dm_tech
    dm_tech.add(np.nan, col_label="FCEV", dummy=True, dim="Categories2")
    dm_tech.sort("Categories2")
    idx = dm_tech.idx
    dm_tech.array[:, :, idx["tra_passenger_new-vehicles"], idx["LDV"], idx["FCEV"]] = 0
    dm_tech.array[
        :, :, idx["tra_passenger_technology-share_fleet"], idx["LDV"], idx["FCEV"]
    ] = 0
    dm_tech.array[:, :, idx["tra_passenger_vehicle-waste"], idx["LDV"], idx["FCEV"]] = 0

    # Add FTS years as NaN
    dm_tech.add(np.nan, col_label=years_fts, dummy=True, dim="Years")

    # Get fleet shares
    dm_fleet_pc = dm_tech.filter(
        {"Variables": ["tra_passenger_technology-share_fleet"]}
    )
    dm_fleet_pc = dm_fleet_pc.normalise("Categories2", inplace=False)
    dm_fleet_pc.rename_col_regex("_share", "", "Variables")
    idx = dm_fleet_pc.idx
    for y in list(range(2025, 2050 + 5, 5)):
        dm_fleet_pc.array[:, idx[y], ...] = np.nan
    dm_tech.drop("Variables", "tra_passenger_technology-share_fleet")
    dm_tech.append(dm_fleet_pc, "Variables")
    dm_tech.sort("Variables")

    # Reorder variables
    dm_temp = dm_tech.filter({"Variables": ["tra_passenger_technology-share_fleet"]})
    dm_temp.append(
        dm_tech.filter({"Variables": ["tra_passenger_veh-efficiency_fleet"]}),
        "Variables",
    )
    dm_temp.append(
        dm_tech.filter({"Variables": ["tra_passenger_new-vehicles"]}), "Variables"
    )
    dm_temp.append(
        dm_tech.filter({"Variables": ["tra_passenger_vehicle-waste"]}), "Variables"
    )
    dm_tech = dm_temp.copy()

    result = (dm_tech, dm_fleet, dm_new, dm_renrate)
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(result, f)
    return result


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_fts, years_ots

    run(years_ots, years_fts)
