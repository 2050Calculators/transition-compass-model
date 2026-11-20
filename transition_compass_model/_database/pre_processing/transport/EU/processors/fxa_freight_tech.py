"""
EU27 freight technology FXA: fleet stock and vehicle efficiency (OTS only).

Two outputs:
  dm_final      : fleet tech share (%) + vehicle efficiency (MJ/km), OTS only.
  dm_fleet_final: absolute fleet by mode×tech (number of vehicles), OTS only.

Accounting basis
----------------
All data sources are territorial (vehicles operating in EU27), used as proxy
for residency — same approach as fxa_passenger_tech. For EU27 as a large bloc,
territorial ≈ residency for fleet composition and efficiency.

HDV fleet split methodology
----------------------------
HDVL (<3.5 t GVW): JRC "Light commercial vehicles" fleet, no split needed.
HDVM + HDVH: JRC "Heavy goods vehicles" fleet, split using Eurostat
  ROAD_GO_TA_MPLW shares (same source as freight_modal_share.py):
  - HDVM share ≈ 10% (MPLW ≤ 20 t)
  - HDVH share ≈ 90% (MPLW > 20 t)
  Tech distribution within HDVH and HDVM: HDVL tech ratios applied to total HGV.

HDV efficiency methodology
---------------------------
HDVL + HDVH: from JRC TrRoad_ene, using HDVL tech ratios for HDVH split.
HDVM: average of HDVL and HDVH efficiency per tech (proxy).

FTS: removed — generated in scenarios/transport_fts_EU.py.

Sources
-------
Road fleet/eff: JRC-IDEES-2021 Transport, sheets TrRoad_tech, TrRoad_ene
Rail fleet/eff: JRC-IDEES-2021 Transport, sheets TrRail_act, TrRail_ene
IWW fleet/eff:  JRC-IDEES-2021 Navigation_Domestic, rows 95-96 / 46-47
Aviation fleet/eff: JRC-IDEES-2021 Aviation_EU, rows 96-97 / 138-139
Maritime fleet/eff: JRC-IDEES-2021 Navigation_International_EU, rows 95-96 / 46-47
MPLW split:     Eurostat ROAD_GO_TA_MPLW (EU27_2020)
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd

from transition_compass_model._database.pre_processing.api_routine_Eurostat import (
    get_data_api_eurostat,
)
from transition_compass_model._database.pre_processing.routine_JRC import get_jrc_data
from transition_compass_model._database.pre_processing.transport.EU.processors.freight_modal_share import (
    _compute_mplw_shares,
)
from transition_compass_model.model.common.auxiliary_functions import (
    eurostat_iso2_dict,
    jrc_iso2_dict,
    linear_fitting,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(_THIS_DIR, "../data/datamatrix/fxa_freight_tech.pickle")

_CATEGORIES1_ALL = ["HDVH", "HDVL", "HDVM", "IWW", "aviation", "marine", "rail"]
_CATEGORIES2_ALL = [
    "BEV",
    "CEV",
    "FCEV",
    "H2",
    "ICE",
    "ICE-diesel",
    "ICE-gas",
    "ICE-gasoline",
    "PHEV-diesel",
    "PHEV-gasoline",
    "kerosene",
]

_JRC_EU27_DIR = os.path.join(_THIS_DIR, "../../../industry/eu/data/JRC-IDEES-2021/EU27")


def _compute_non_road_load_factors(years_ots_ext):
    """
    Compute load factors (tkm/vkm) for non-road freight modes from EU27 JRC data.

    Returns a dict {mode: np.ndarray of shape (len(years_ots_ext),)}.
    Dividing MJ/vkm by these factors gives MJ/tkm, which is the unit expected
    by workflows.py (non-road energy = eff × tkm).

    Sources
    -------
    IWW    : Navigation_Domestic EU27, rows 3 (tkm) and 11 (vkm)
    Marine : Navigation_International_EU EU27 rows 2/11 (intra tkm/vkm)
             + Navigation_Domestic EU27 rows 2/10 (coastal tkm/vkm)
    Rail   : Transport EU27 TrRail_act, rows 8 (tkm) and 19 (vkm), 2000-2021
    Aviation: Aviation_EU EU27, rows 6 (tkm) and 26 (vkm)
    """

    def read_eu27_row(database, sheet, row_idx):
        fp = os.path.join(_JRC_EU27_DIR, f"{database}.xlsx")
        df = pd.read_excel(fp, sheet_name=sheet)
        row = df.iloc[row_idx]
        return {
            int(c): float(row[c])
            for c in df.columns
            if isinstance(c, (int, float)) and c == int(c)
        }

    def to_array(year_dict, years):
        avail = sorted(year_dict.keys())
        return np.array(
            [
                year_dict[y]
                if y in year_dict
                else (year_dict[avail[0]] if y < avail[0] else year_dict[avail[-1]])
                for y in years
            ]
        )

    # IWW
    iww_tkm = read_eu27_row("JRC-IDEES-2021_x1990_Navigation_Domestic", "EU27", 3)
    iww_vkm = read_eu27_row("JRC-IDEES-2021_x1990_Navigation_Domestic", "EU27", 11)
    iww_lf = {y: iww_tkm[y] / iww_vkm[y] for y in iww_tkm if iww_vkm.get(y, 0) > 0}

    # Marine: intra-EU27 + domestic coastal
    intra_tkm = read_eu27_row(
        "JRC-IDEES-2021_x1990_Navigation_International_EU", "EU27", 2
    )
    intra_vkm = read_eu27_row(
        "JRC-IDEES-2021_x1990_Navigation_International_EU", "EU27", 11
    )
    coast_tkm = read_eu27_row("JRC-IDEES-2021_x1990_Navigation_Domestic", "EU27", 2)
    coast_vkm = read_eu27_row("JRC-IDEES-2021_x1990_Navigation_Domestic", "EU27", 10)
    marine_lf = {
        y: (intra_tkm[y] + coast_tkm.get(y, 0)) / (intra_vkm[y] + coast_vkm.get(y, 0))
        for y in intra_tkm
        if (intra_vkm.get(y, 0) + coast_vkm.get(y, 0)) > 0
    }

    # Rail: Transport EU27 aggregate (2000-2021 only; extrapolate outside range)
    rail_tkm = read_eu27_row("JRC-IDEES-2021_Transport_EU27", "TrRail_act", 8)
    rail_vkm = read_eu27_row("JRC-IDEES-2021_Transport_EU27", "TrRail_act", 19)
    rail_lf = {y: rail_tkm[y] / rail_vkm[y] for y in rail_tkm if rail_vkm.get(y, 0) > 0}

    # Aviation
    avi_tkm = read_eu27_row("JRC-IDEES-2021_x1990_Aviation_EU", "EU27", 6)
    avi_vkm = read_eu27_row("JRC-IDEES-2021_x1990_Aviation_EU", "EU27", 26)
    avi_lf = {y: avi_tkm[y] / avi_vkm[y] for y in avi_tkm if avi_vkm.get(y, 0) > 0}

    return {
        "IWW": to_array(iww_lf, years_ots_ext),
        "marine": to_array(marine_lf, years_ots_ext),
        "rail": to_array(rail_lf, years_ots_ext),
        "aviation": to_array(avi_lf, years_ots_ext),
    }


def run(years_ots: list, years_fts: list):
    """
    Build freight technology FXA for EU27 (OTS only).

    Parameters
    ----------
    years_ots, years_fts : list

    Returns
    -------
    dm_final : DataMatrix
        Variables=['tra_freight_technology-share_fleet', 'tra_freight_vehicle-efficiency_fleet'],
        Cat1=modes, Cat2=techs. OTS only.
    dm_fleet_final : DataMatrix
        Same structure as dm_final fleet variable, absolute fleet counts (number of vehicles).
        OTS only. Used by fxa_freight_mode_other and fxa_vehicles_max.
    """
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    years_ots_ext = list(range(1989, years_ots[-1] + 1))
    startyear = years_ots_ext[0]  # 1989

    dict_iso2_jrc = jrc_iso2_dict()
    country_codes = list(dict_iso2_jrc.keys())
    country_names = list(dict_iso2_jrc.values())

    # MPLW shares for HDVM/HDVH fleet split (EU27 aggregate, 2008-present; pre-2008 = 2008)
    mplw_shares = _compute_mplw_shares(years_ots_ext)

    ###############################################################################
    ##################################### FLEET ###################################
    ###############################################################################

    ################
    ##### HDVL #####
    ################

    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_tech",
        "variable": "Stock of vehicles - total (vehicles)",
        "categories": "Light commercial vehicles",
        "sheet_last_row": "Battery electric vehicles",
        "sub_variables": [
            "Gasoline engine",
            "Diesel oil engine",
            "LPG engine",
            "Natural gas engine",
            "Battery electric vehicles",
        ],
        "calc_names": [
            "HDVL_ICE-gasoline",
            "HDVL_ICE-diesel",
            "HDVL_ICE-gas-lpg",
            "HDVL_ICE-gas-natural",
            "HDVL_BEV",
        ],
    }
    dm_hdvl = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_hdvl.array[dm_hdvl.array == 0] = np.nan
    dm_hdvl.groupby(
        {"HDVL_ICE-gas": ["HDVL_ICE-gas-lpg", "HDVL_ICE-gas-natural"]},
        dim="Variables",
        aggregation="sum",
        regex=False,
        inplace=True,
    )
    dm_hdvl.deepen()
    categories2_missing = _CATEGORIES2_ALL.copy()
    for cat in dm_hdvl.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_hdvl.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
    dm_hdvl.sort("Categories1")

    ################
    ##### HDVH #####
    ################

    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_tech",
        "variable": "Stock of vehicles - total (vehicles)",
        "sheet_last_row": "Heavy goods vehicles",
        "sub_variables": ["Heavy goods vehicles"],
        "calc_names": ["HDVH"],
    }
    dm_hdvh_raw = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_hdvh_raw.array[dm_hdvh_raw.array == 0] = np.nan
    # Anchor EU27 HGV fleet to Eurostat tran_r_vehst: LOR - VG_LE3P5 (JRC is ~40% over Eurostat)
    _dict_iso2_eu = eurostat_iso2_dict()
    _dict_iso2_eu.pop("CH")
    _filter_hgv = {
        "geo\\TIME_PERIOD": [k for k in _dict_iso2_eu.keys() if k != "EU27_2020"],
        "unit": "NR",
    }
    _mapping_hgv = {"Country": "geo\\TIME_PERIOD", "Variables": "vehicle"}
    _dm_lor = get_data_api_eurostat(
        "tran_r_vehst", {**_filter_hgv, "vehicle": ["LOR"]}, _mapping_hgv, "num"
    )
    _dm_lor.drop("Country", "United Kingdom")
    _dm_vg = get_data_api_eurostat(
        "tran_r_vehst", {**_filter_hgv, "vehicle": ["VG_LE3P5"]}, _mapping_hgv, "num"
    )
    _dm_vg.drop("Country", "United Kingdom")
    # VG_LE3P5 reporting is incomplete before ~2023; only use years where VG/LOR > 85%
    _eu27_lor = np.nansum(_dm_lor.array, axis=0)[:, 0]  # (n_lor_years,)
    _eu27_vg = np.nansum(_dm_vg.array, axis=0)[:, 0]  # (n_vg_years,)
    _lor_years = _dm_lor.col_labels["Years"]
    _vg_years = _dm_vg.col_labels["Years"]
    _frac_heavy = None
    for _y_cal in _vg_years:
        if _y_cal in _lor_years:
            _il = _lor_years.index(_y_cal)
            _iv = _vg_years.index(_y_cal)
            if _eu27_lor[_il] > 0 and _eu27_vg[_iv] / _eu27_lor[_il] > 0.85:
                _frac_heavy = float((_eu27_lor[_il] - _eu27_vg[_iv]) / _eu27_lor[_il])
                break
    if _frac_heavy is None:
        _frac_heavy = 0.13  # fallback: ~13% of all lorries are heavy trucks >3.5t
    _idx_hgv = dm_hdvh_raw.idx
    for _i, _y in enumerate(_lor_years):
        if _y in dm_hdvh_raw.col_labels["Years"]:
            dm_hdvh_raw.array[_idx_hgv["EU27"], _idx_hgv[_y], _idx_hgv["HDVH"]] = float(
                _eu27_lor[_i] * _frac_heavy
            )

    # Apply HDVL tech ratios to total HGV fleet to get per-tech HDVH counts
    dm_temp = dm_hdvl.flatten()
    dm_temp1 = dm_temp.groupby(
        {"HDVL": ["HDVL_BEV", "HDVL_ICE-diesel", "HDVL_ICE-gas", "HDVL_ICE-gasoline"]},
        dim="Variables",
        aggregation="sum",
        regex=False,
        inplace=False,
    )
    dm_temp.append(dm_temp1, "Variables")
    idx = dm_temp.idx
    dm_temp.array = dm_temp.array / dm_temp.array[..., idx["HDVL"], np.newaxis]
    dm_temp.drop("Variables", ["HDVL"])
    dm_temp.array = dm_temp.array * dm_hdvh_raw.array
    dm_temp.rename_col_regex("HDVL", "HDVH", "Variables")
    dm_hdvh = dm_temp.copy()

    dm_hdvh.deepen()
    categories2_missing = _CATEGORIES2_ALL.copy()
    for cat in dm_hdvh.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_hdvh.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
    dm_hdvh.sort("Categories1")

    ################
    ##### HDVM #####
    ################

    # Split HGV fleet into HDVH and HDVM using Eurostat MPLW shares (consistent with
    # freight_modal_share.py). dm_hdvh currently holds the full HGV fleet with HDVL
    # tech ratios applied; scale by MPLW shares to get HDVH and HDVM separately.
    years_in_dm = dm_hdvh.col_labels["Years"]
    hdvm_sh_arr = np.array([mplw_shares[y][0] for y in years_in_dm])
    hdvh_sh_arr = np.array([mplw_shares[y][1] for y in years_in_dm])

    dm_hdvm = dm_hdvh.copy()
    dm_hdvm.rename_col("HDVH", "HDVM", "Variables")
    # array shape: (n_countries, n_years, 1, n_cat1) — broadcast over years
    dm_hdvh.array *= hdvh_sh_arr[np.newaxis, :, np.newaxis, np.newaxis]
    dm_hdvm.array *= hdvm_sh_arr[np.newaxis, :, np.newaxis, np.newaxis]

    ####################
    ##### AVIATION #####
    ####################

    def get_specific_jrc_data(
        country_code,
        country_name,
        row_start,
        row_end,
        unit,
        variable="aviation_kerosene",
        database="JRC-IDEES-2021_x1990_Aviation_EU",
    ):
        filepath_jrc = os.path.join(
            _THIS_DIR,
            f"../../../industry/eu/data/JRC-IDEES-2021/EU27/{database}.xlsx",
        )
        df_temp = pd.read_excel(filepath_jrc, sheet_name=country_code)
        df_temp = df_temp.iloc[row_start:row_end, :]
        indexes = df_temp.columns[0]
        df_temp = pd.melt(df_temp, id_vars=indexes, var_name="year")
        df_temp.columns = ["Country", "Years", f"{variable}[{unit}]"]
        df_temp["Country"] = country_name
        return df_temp

    df_fleet_avi = pd.concat(
        [
            get_specific_jrc_data(code, name, 96, 97, "unit")
            for code, name in zip(country_codes, country_names)
        ],
        ignore_index=True,
    )
    dm_avi = DataMatrix.create_from_df(df_fleet_avi, 0)
    dm_avi.array = np.round(dm_avi.array, 0)
    dm_avi.deepen()
    categories2_missing = _CATEGORIES2_ALL.copy()
    for cat in dm_avi.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_avi.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
    dm_avi.sort("Categories1")

    ################
    ##### RAIL #####
    ################

    dict_extract = {
        "database": "Transport",
        "sheet": "TrRail_act",
        "variable": "Stock of vehicles - total (representative train configuration)",
        "sheet_last_row": "Electric",
        "categories": "Freight transport",
        "sub_variables": ["Diesel oil", "Electric"],
        "calc_names": ["rail_ICE-diesel", "rail_CEV"],
    }
    dm_fleet_rail = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_fleet_rail.deepen()
    categories2_missing = _CATEGORIES2_ALL.copy()
    for cat in dm_fleet_rail.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_fleet_rail.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories1"
    )
    dm_fleet_rail.sort("Categories1")

    ###############
    ##### IWW #####
    ###############

    df_iww_fleet = pd.concat(
        [
            get_specific_jrc_data(
                code,
                name,
                95,
                96,
                "unit",
                "IWW_ICE",
                "JRC-IDEES-2021_x1990_Navigation_Domestic",
            )
            for code, name in zip(country_codes, country_names)
        ],
        ignore_index=True,
    )
    dm_iww_fleet = DataMatrix.create_from_df(df_iww_fleet, 0)
    dm_iww_fleet.array = np.round(dm_iww_fleet.array, 0)
    dm_iww_fleet.deepen()
    categories2_missing = _CATEGORIES2_ALL.copy()
    for cat in dm_iww_fleet.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_iww_fleet.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories1"
    )
    dm_iww_fleet.sort("Categories1")

    ##################
    ##### MARINE #####
    ##################

    df_mar_fleet = pd.concat(
        [
            get_specific_jrc_data(
                code,
                name,
                95,
                96,
                "unit",
                "marine_ICE",
                "JRC-IDEES-2021_x1990_Navigation_International_EU",
            )
            for code, name in zip(country_codes, country_names)
        ],
        ignore_index=True,
    )
    dm_mar_fleet = DataMatrix.create_from_df(df_mar_fleet, 0)
    dm_mar_fleet.array = np.round(dm_mar_fleet.array, 0)
    dm_mar_fleet.deepen()
    categories2_missing = _CATEGORIES2_ALL.copy()
    for cat in dm_mar_fleet.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_mar_fleet.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories1"
    )
    dm_mar_fleet.sort("Categories1")

    ########################
    ##### PUT TOGETHER #####
    ########################

    dm_fleet = dm_hdvl.copy()
    dm_fleet.append(dm_hdvh, "Variables")
    dm_fleet.append(dm_hdvm, "Variables")
    dm_fleet.sort("Variables")
    dm_fleet.add(np.nan, "Years", list(range(1990, 1999 + 1)), "number", True)
    dm_fleet.sort("Years")
    dm_avi.sort("Years")
    dm_fleet_rail.add(np.nan, "Years", list(range(1990, 1999 + 1)), "number", True)
    dm_fleet_rail.sort("Years")
    dm_fleet.append(dm_fleet_rail, "Variables")
    dm_iww_fleet.sort("Years")
    dm_fleet.append(dm_iww_fleet, "Variables")
    dm_mar_fleet.sort("Years")
    dm_fleet.append(dm_mar_fleet, "Variables")
    dm_fleet.append(dm_avi, "Variables")
    dm_fleet.sort("Variables")
    dm_fleet.sort("Country")
    dm_fleet.array[dm_fleet.array == 0] = np.nan

    ###################
    ##### FIX OTS #####
    ###################

    dm_fleet = dm_fleet.flatten()

    def make_ots_fleet(
        dm, variable, periods_dicts, yrs_ots_ext=None, min_t0=0.1, min_tb=0.1
    ):
        dm_temp = dm.filter({"Variables": [variable]})
        if periods_dicts["n_adj"] == 1:
            dm_temp = linear_fitting(dm_temp, yrs_ots_ext, min_t0=min_t0, min_tb=min_tb)
        if periods_dicts["n_adj"] == 2:
            dm_temp = linear_fitting(
                dm_temp,
                list(range(startyear, 1999 + 1)),
                based_on=list(range(2000, periods_dicts["year_end_first_adj"] + 1)),
                min_t0=min_t0,
                min_tb=min_tb,
            )
            dm_temp = linear_fitting(
                dm_temp,
                list(range(2022, 2023 + 1)),
                based_on=list(range(periods_dicts["year_start_second_adj"], 2021 + 1)),
                min_t0=min_t0,
                min_tb=min_tb,
            )
        return dm_temp

    dict_call_fleet = {
        "HDVH_BEV": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2021,
        },
        "HDVH_ICE-diesel": {"n_adj": 1},
        "HDVH_ICE-gas": {"n_adj": 1},
        "HDVH_ICE-gasoline": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2021,
            "min_t0": 0,
            "min_tb": 0,
        },
        "HDVL_BEV": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2021,
        },
        "HDVL_ICE-diesel": {"n_adj": 1},
        "HDVL_ICE-gas": {"n_adj": 1},
        "HDVL_ICE-gasoline": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2021,
            "min_t0": 0,
            "min_tb": 0,
        },
        "HDVM_BEV": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2021,
        },
        "HDVM_ICE-diesel": {"n_adj": 1},
        "HDVM_ICE-gas": {"n_adj": 1},
        "HDVM_ICE-gasoline": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2021,
            "min_t0": 0,
            "min_tb": 0,
        },
        "aviation_kerosene": {"n_adj": 1},
        "rail_CEV": {"n_adj": 1},
        "rail_ICE-diesel": {"n_adj": 1},
        "marine_ICE": {"n_adj": 1},
        "IWW_ICE": {"n_adj": 1},
    }

    dict_new_fleet = {}
    for key in dict_call_fleet.keys():
        min_t0 = dict_call_fleet[key].pop("min_t0", 0.1)
        min_tb = dict_call_fleet[key].pop("min_tb", 0.1)
        if len(dict_call_fleet[key]) > 1:
            dict_new_fleet[key] = make_ots_fleet(
                dm_fleet, key, dict_call_fleet[key], min_t0=min_t0, min_tb=min_tb
            )
        else:
            dict_new_fleet[key] = make_ots_fleet(
                dm_fleet,
                key,
                dict_call_fleet[key],
                years_ots_ext,
                min_t0=min_t0,
                min_tb=min_tb,
            )

    first_key = "HDVH_BEV"
    dm_fleet = dict_new_fleet[first_key].copy()
    for v in [k for k in dict_call_fleet.keys() if k != first_key]:
        dm_fleet.append(dict_new_fleet[v], "Variables")
    dm_fleet.sort("Variables")

    ################################################
    ##### FIX LAST YEARS FOR ELECTRIC VEHICLES #####
    ################################################

    idx = dm_fleet.idx
    for v in ["HDVH_BEV", "HDVL_BEV", "HDVM_BEV"]:
        dm_fleet.array[idx["EU27"], idx[2022], idx[v]] = np.nan
        dm_fleet.array[idx["EU27"], idx[2023], idx[v]] = np.nan

    rates = {}
    for v in ["HDVH_BEV", "HDVL_BEV", "HDVM_BEV"]:
        rates[v] = (
            dm_fleet.array[idx["EU27"], idx[2021], idx[v]]
            - dm_fleet.array[idx["EU27"], idx[2020], idx[v]]
        ) / dm_fleet.array[idx["EU27"], idx[2020], idx[v]]

    for v in ["HDVH_BEV", "HDVL_BEV", "HDVM_BEV"]:
        dm_fleet.array[idx["EU27"], idx[2022], idx[v]] = dm_fleet.array[
            idx["EU27"], idx[2021], idx[v]
        ] * (1 + rates[v])
        dm_fleet.array[idx["EU27"], idx[2023], idx[v]] = dm_fleet.array[
            idx["EU27"], idx[2022], idx[v]
        ] * (1 + rates[v])

    dm_fleet.deepen()

    ####################################
    ##### MAKE AS FINAL DATAMATRIX #####
    ####################################

    for v in dm_fleet.col_labels["Variables"]:
        dm_fleet.rename_col(v, "tra_freight_technology-share_fleet_" + v, "Variables")
    dm_fleet.deepen(based_on="Variables")
    dm_fleet.switch_categories_order("Categories1", "Categories2")
    dm_fleet_pc = dm_fleet.copy()
    dm_fleet_pc.normalise("Categories2")

    # Add missing categories to fleet_pc
    categories1_missing = [
        c for c in _CATEGORIES1_ALL if c not in dm_fleet_pc.col_labels["Categories1"]
    ]
    categories2_missing = [
        c for c in _CATEGORIES2_ALL if c not in dm_fleet_pc.col_labels["Categories2"]
    ]
    for c in categories1_missing:
        dm_fleet_pc.add(np.nan, col_label=c, dummy=True, dim="Categories1")
    for c in categories2_missing:
        dm_fleet_pc.add(np.nan, col_label=c, dummy=True, dim="Categories2")
    dm_fleet_pc.sort("Categories1")
    dm_fleet_pc.sort("Categories2")
    dm_fleet_pc_final = dm_fleet_pc.copy()

    # Add missing categories to absolute fleet
    categories1_missing = [
        c for c in _CATEGORIES1_ALL if c not in dm_fleet.col_labels["Categories1"]
    ]
    categories2_missing = [
        c for c in _CATEGORIES2_ALL if c not in dm_fleet.col_labels["Categories2"]
    ]
    for c in categories1_missing:
        dm_fleet.add(np.nan, col_label=c, dummy=True, dim="Categories1")
    for c in categories2_missing:
        dm_fleet.add(np.nan, col_label=c, dummy=True, dim="Categories2")
    dm_fleet.sort("Categories1")
    dm_fleet.sort("Categories2")
    dm_fleet.units["tra_freight_technology-share_fleet"] = "number"
    dm_fleet_final = dm_fleet.copy()

    ###############################################################################
    ############################### VEHICLE EFFICIENCY ############################
    ###############################################################################

    ################
    ##### HDVL #####
    ################

    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_ene",
        "variable": "Vehicle-efficiency - effective (kgoe/100 km)",
        "categories": "Light commercial vehicles",
        "sheet_last_row": "Battery electric vehicles",
        "sub_variables": [
            "Gasoline engine",
            "Diesel oil engine",
            "LPG engine",
            "Natural gas engine",
            "Battery electric vehicles",
        ],
        "calc_names": [
            "HDVL_ICE-gasoline",
            "HDVL_ICE-diesel",
            "HDVL_ICE-gas-lpg",
            "HDVL_ICE-gas-natural",
            "HDVL_BEV",
        ],
    }
    dm_hdvl = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_hdvl.array[dm_hdvl.array == 0] = np.nan
    dm_hdvl.groupby(
        {"HDVL_ICE-gas": ["HDVL_ICE-gas-lpg", "HDVL_ICE-gas-natural"]},
        dim="Variables",
        aggregation="mean",
        regex=False,
        inplace=True,
    )
    dm_hdvl.deepen()
    categories2_missing = _CATEGORIES2_ALL.copy()
    for cat in dm_hdvl.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_hdvl.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
    dm_hdvl.sort("Categories1")

    ################
    ##### HDVH #####
    ################

    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_ene",
        "variable": "Vehicle-efficiency - effective (kgoe/100 km)",
        "sheet_last_row": "Heavy goods vehicles",
        "sub_variables": ["Heavy goods vehicles"],
        "calc_names": ["HDVH"],
    }
    dm_hdvh_raw = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_hdvh_raw.array[dm_hdvh_raw.array == 0] = np.nan

    # Use HDVL tech ratios to split HGV efficiency across tech types
    dm_temp = dm_hdvl.flatten()
    dm_temp1 = dm_temp.groupby(
        {"HDVL": ["HDVL_BEV", "HDVL_ICE-diesel", "HDVL_ICE-gas", "HDVL_ICE-gasoline"]},
        dim="Variables",
        aggregation="mean",
        regex=False,
        inplace=False,
    )
    dm_temp.append(dm_temp1, "Variables")
    idx = dm_temp.idx
    dm_temp.array = dm_temp.array / dm_temp.array[..., idx["HDVL"], np.newaxis]
    dm_temp.drop("Variables", ["HDVL"])
    dm_temp.array = dm_temp.array * dm_hdvh_raw.array
    dm_temp.rename_col_regex("HDVL", "HDVH", "Variables")
    dm_hdvh = dm_temp.copy()

    dm_hdvh.deepen()
    categories2_missing = _CATEGORIES2_ALL.copy()
    for cat in dm_hdvh.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_hdvh.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
    dm_hdvh.sort("Categories1")

    ################
    ##### HDVM #####
    ################

    # HDVM efficiency = average of HDVL and HDVH efficiency per tech
    dm_eneff = dm_hdvl.copy()
    dm_eneff.append(dm_hdvh, "Variables")
    dm_hdvm = dm_eneff.flatten()
    dm_hdvm.deepen()
    dm_hdvm.groupby(
        {"HDVM": ["HDVL", "HDVH"]},
        dim="Variables",
        aggregation="mean",
        regex=False,
        inplace=True,
    )
    categories2_missing = _CATEGORIES2_ALL.copy()
    for cat in dm_hdvm.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_hdvm.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
    dm_hdvm.sort("Categories1")
    dm_eneff.append(dm_hdvm, "Variables")
    dm_eneff.sort("Variables")
    dm_eneff.sort("Country")
    dm_eneff.sort("Years")

    ###############
    ##### IWW #####
    ###############

    df_iww = pd.concat(
        [
            get_specific_jrc_data(
                code,
                name,
                46,
                47,
                "kgoe/100 km",
                "IWW_ICE",
                "JRC-IDEES-2021_x1990_Navigation_Domestic",
            )
            for code, name in zip(country_codes, country_names)
        ],
        ignore_index=True,
    )
    dm_iww = DataMatrix.create_from_df(df_iww, 0)
    dm_iww.array[dm_iww.array == 0] = np.nan
    dm_iww.deepen()
    categories2_missing = _CATEGORIES2_ALL.copy()
    for cat in dm_iww.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_iww.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
    dm_iww.sort("Categories1")

    ####################
    ##### AVIATION #####
    ####################

    df_avi = pd.concat(
        [
            get_specific_jrc_data(code, name, 138, 139, "kgoe/100 km")
            for code, name in zip(country_codes, country_names)
        ],
        ignore_index=True,
    )
    dm_avi = DataMatrix.create_from_df(df_avi, 0)
    dm_avi.deepen()
    categories2_missing = _CATEGORIES2_ALL.copy()
    for cat in dm_avi.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_avi.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
    dm_avi.sort("Categories1")

    ####################
    ##### MARITIME #####
    ####################

    df_mar = pd.concat(
        [
            get_specific_jrc_data(
                code,
                name,
                47,
                48,
                "kgoe/100 km",
                "marine_ICE",
                "JRC-IDEES-2021_x1990_Navigation_International_EU",
            )
            for code, name in zip(country_codes, country_names)
        ],
        ignore_index=True,
    )
    dm_mar = DataMatrix.create_from_df(df_mar, 0)
    dm_mar.array[dm_mar.array == 0] = np.nan
    dm_mar.deepen()
    categories2_missing = _CATEGORIES2_ALL.copy()
    for cat in dm_mar.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_mar.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
    dm_mar.sort("Categories1")

    ################
    ##### RAIL #####
    ################

    dict_extract = {
        "database": "Transport",
        "sheet": "TrRail_ene",
        "variable": "Vehicle-efficiency (kgoe/100 km)",
        "sheet_last_row": "Electric",
        "categories": "Freight transport",
        "sub_variables": ["Diesel oil", "Electric"],
        "calc_names": ["rail_ICE-diesel", "rail_CEV"],
    }
    dm_eneff_rail = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_eneff_rail.deepen()
    categories2_missing = _CATEGORIES2_ALL.copy()
    for cat in dm_eneff_rail.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_eneff_rail.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories1"
    )
    dm_eneff_rail.sort("Categories1")

    ########################
    ##### PUT TOGETHER #####
    ########################

    dm_eneff.add(np.nan, "Years", list(range(1990, 2000)), dummy=True)
    dm_eneff.sort("Years")
    dm_eneff.append(dm_iww, "Variables")
    dm_eneff.append(dm_mar, "Variables")
    dm_eneff_rail.add(np.nan, "Years", list(range(1990, 2000)), dummy=True)
    dm_eneff_rail.sort("Years")
    dm_eneff.append(dm_eneff_rail, "Variables")
    dm_eneff.sort("Variables")
    dm_eneff.sort("Country")
    dm_eneff.append(dm_avi, "Variables")
    dm_eneff.sort("Variables")
    dm_eneff.array[dm_eneff.array == 0] = np.nan

    ###################
    ##### FIX OTS #####
    ###################

    dm_eneff = dm_eneff.flatten()

    def make_ots_eneff(
        dm, variable, periods_dicts, yrs_ots_ext=None, min_t0=0.1, min_tb=0.1
    ):
        dm_temp = dm.filter({"Variables": [variable]})
        if periods_dicts["n_adj"] == 1:
            dm_temp = linear_fitting(dm_temp, yrs_ots_ext, min_t0=min_t0, min_tb=min_tb)
        if periods_dicts["n_adj"] == 2:
            dm_temp = linear_fitting(
                dm_temp,
                list(range(startyear, 1999 + 1)),
                based_on=list(range(2000, periods_dicts["year_end_first_adj"] + 1)),
                min_t0=min_t0,
                min_tb=min_tb,
            )
            dm_temp = linear_fitting(
                dm_temp,
                list(range(2022, 2023 + 1)),
                based_on=list(range(periods_dicts["year_start_second_adj"], 2021 + 1)),
                min_t0=min_t0,
                min_tb=min_tb,
            )
        return dm_temp

    dict_call_eneff = {
        "HDVH_BEV": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2020,
        },
        "HDVH_ICE-diesel": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2010,
        },
        "HDVH_ICE-gas": {
            "n_adj": 2,
            "year_end_first_adj": 2015,
            "year_start_second_adj": 2015,
        },
        "HDVH_ICE-gasoline": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2010,
            "min_t0": 0,
            "min_tb": 0,
        },
        "HDVL_BEV": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2020,
        },
        "HDVL_ICE-diesel": {
            "n_adj": 2,
            "year_end_first_adj": 2001,
            "year_start_second_adj": 2020,
        },
        "HDVL_ICE-gas": {
            "n_adj": 2,
            "year_end_first_adj": 2001,
            "year_start_second_adj": 2020,
        },
        "HDVL_ICE-gasoline": {
            "n_adj": 2,
            "year_end_first_adj": 2020,
            "year_start_second_adj": 2020,
            "min_t0": 0,
            "min_tb": 0,
        },
        "HDVM_BEV": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2020,
        },
        "HDVM_ICE-diesel": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2010,
        },
        "HDVM_ICE-gas": {
            "n_adj": 2,
            "year_end_first_adj": 2015,
            "year_start_second_adj": 2015,
        },
        "HDVM_ICE-gasoline": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2010,
            "min_t0": 0,
            "min_tb": 0,
        },
        "IWW_ICE": {"n_adj": 1},
        "aviation_kerosene": {"n_adj": 1},
        "marine_ICE": {"n_adj": 1},
        "rail_CEV": {"n_adj": 1},
        "rail_ICE-diesel": {"n_adj": 1},
    }

    dict_new_eneff = {}
    for key in dict_call_eneff.keys():
        min_t0 = dict_call_eneff[key].pop("min_t0", 0.1)
        min_tb = dict_call_eneff[key].pop("min_tb", 0.1)
        if len(dict_call_eneff[key]) > 1:
            dict_new_eneff[key] = make_ots_eneff(
                dm_eneff, key, dict_call_eneff[key], min_t0=min_t0, min_tb=min_tb
            )
        else:
            dict_new_eneff[key] = make_ots_eneff(
                dm_eneff,
                key,
                dict_call_eneff[key],
                years_ots_ext,
                min_t0=min_t0,
                min_tb=min_tb,
            )

    first_key = "HDVH_BEV"
    dm_eneff = dict_new_eneff[first_key].copy()
    for v in [k for k in dict_call_eneff.keys() if k != first_key]:
        dm_eneff.append(dict_new_eneff[v], "Variables")
    dm_eneff.sort("Variables")

    dm_eneff.deepen()

    ####################################
    ##### MAKE AS FINAL DATAMATRIX #####
    ####################################

    for v in dm_eneff.col_labels["Variables"]:
        dm_eneff.rename_col(v, "tra_freight_vehicle-efficiency_fleet_" + v, "Variables")
    dm_eneff.deepen(based_on="Variables")
    dm_eneff.switch_categories_order("Categories1", "Categories2")

    dm_eneff.change_unit(
        "tra_freight_vehicle-efficiency_fleet", 41.868, "kgoe/100 km", "MJ/100 km"
    )
    dm_eneff.change_unit(
        "tra_freight_vehicle-efficiency_fleet", 1e-2, "MJ/100 km", "MJ/km"
    )

    # Non-road modes: divide by load factor (tkm/vkm) to convert MJ/vkm → MJ/tkm.
    # workflows.py multiplies non-road efficiency directly by tkm, so MJ/tkm is needed.
    load_factors = _compute_non_road_load_factors(years_ots_ext)
    lf_year_idx = {y: i for i, y in enumerate(years_ots_ext)}
    for mode, lf_arr in load_factors.items():
        if mode not in dm_eneff.col_labels["Categories1"]:
            continue
        i_mode = dm_eneff.col_labels["Categories1"].index(mode)
        lf_for_dm = np.array(
            [lf_arr[lf_year_idx[y]] for y in dm_eneff.col_labels["Years"]]
        )
        # array shape: (n_country, n_years, n_vars, n_cat1, n_cat2)
        dm_eneff.array[:, :, :, i_mode, :] /= lf_for_dm.reshape(1, -1, 1, 1)

    categories1_missing = [
        c for c in _CATEGORIES1_ALL if c not in dm_eneff.col_labels["Categories1"]
    ]
    categories2_missing = [
        c for c in _CATEGORIES2_ALL if c not in dm_eneff.col_labels["Categories2"]
    ]
    dm_eneff.add(np.nan, col_label=categories1_missing, dummy=True, dim="Categories1")
    dm_eneff.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories2")
    dm_eneff.sort("Categories1")
    dm_eneff.sort("Categories2")
    dm_eneff_final = dm_eneff.copy()

    #############################################
    ################# DROP 1989 #################
    #############################################

    dm_fleet_pc_final.drop("Years", startyear)
    dm_eneff_final.drop("Years", startyear)
    dm_fleet_final.drop("Years", startyear)

    ##########################################
    ################## SAVE ##################
    ##########################################

    dm_final = dm_fleet_pc_final.copy()
    dm_final.append(dm_eneff_final, "Variables")

    dm_final = dm_final.filter({"Years": years_ots})
    dm_final.add(np.nan, col_label=years_fts, dummy=True, dim="Years")
    dm_final = linear_fitting(dm_final, years_fts, min_t0=0, min_tb=0)

    dm_fleet_final = dm_fleet_final.filter({"Years": years_ots})
    dm_fleet_final.add(np.nan, col_label=years_fts, dummy=True, dim="Years")
    dm_fleet_final = linear_fitting(dm_fleet_final, years_fts, min_t0=0, min_tb=0)

    result = (dm_final, dm_fleet_final)
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(result, f)
    return result


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_fts, years_ots

    run(years_ots, years_fts)
