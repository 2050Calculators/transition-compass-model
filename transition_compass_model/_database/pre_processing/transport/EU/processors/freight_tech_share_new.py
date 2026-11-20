"""
EU27 freight new-vehicle technology share OTS (1990-2023).

Faithful port of transport_lever_freight_technology-share_new.py. Two structural changes:
  1. HDVM split: old script split JRC LCV 40/60 into HDVL/HDVM — inconsistent with
     fxa_freight_tech and freight_modal_share which use Eurostat MPLW shares. Fixed
     here: HDVL = JRC LCV directly; HDVH/HDVM = JRC HGV × MPLW shares (same source
     as fxa_freight_tech and freight_modal_share).
  2. OTS only — FTS is generated in scenarios/transport_fts_EU.py (all 4 levels BAU).
     # TODO: differentiate FTS ambition levels for freight tech share.

Accounting basis
----------------
Technology share of new vehicle registrations by mode (HDVL, HDVH, HDVM, aviation, rail).
Source: EU27 vehicle registrations via JRC-IDEES-2021 (TrRoad_tech, TrRail_act, Aviation_EU).
Territorial proxy for residency — EU27-registered vehicles. No residency correction needed
(same reasoning as fxa_passenger_tech: fleet composition is independent of pkm accounting).

IWW and marine: set to ICE=100% (no per-tech disaggregation in JRC for new-vehicle tech).

Sources
-------
Road HDV:  JRC-IDEES-2021 Transport, TrRoad_tech, "New vehicle-registrations"
Rail:      JRC-IDEES-2021 Transport, TrRail_act, "New vehicles - total (representative...)"
Aviation:  JRC-IDEES-2021 Aviation_EU, row 106 (new aircraft deliveries, kerosene)
MPLW:      Eurostat ROAD_GO_TA_MPLW (EU27_2020) — shared with fxa_freight_tech
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd

from transition_compass_model._database.pre_processing.routine_JRC import get_jrc_data
from transition_compass_model._database.pre_processing.transport.EU.processors.freight_modal_share import (
    _compute_mplw_shares,
)
from transition_compass_model.model.common.auxiliary_functions import (
    jrc_iso2_dict,
    linear_fitting,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(
    _THIS_DIR, "../data/datamatrix/freight_tech_share_new.pickle"
)

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


def run(years_ots: list, years_fts: list):
    """
    Build freight new-vehicle technology share DataMatrix for EU27 (OTS only).

    Parameters
    ----------
    years_ots, years_fts : list

    Returns
    -------
    dm_new_ots : DataMatrix
        Variable='tra_freight_technology-share_new', Cat1=modes, Cat2=techs. OTS only.
        IWW and marine forced to ICE=1.
    """
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    years_ots_ext = list(range(1989, years_ots[-1] + 1))
    startyear = years_ots_ext[0]

    dict_iso2_jrc = jrc_iso2_dict()
    country_codes = list(dict_iso2_jrc.keys())
    country_names = list(dict_iso2_jrc.values())

    mplw_shares = _compute_mplw_shares(years_ots_ext)

    ###########################################################################
    # GET DATA
    ###########################################################################

    ################
    ##### HDVL #####
    ################

    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_tech",
        "variable": "New vehicle-registrations",
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
        "variable": "New vehicle-registrations",
        "sheet_last_row": "Heavy goods vehicles",
        "sub_variables": ["Heavy goods vehicles"],
        "calc_names": ["HDVH"],
    }
    dm_hdvh_raw = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_hdvh_raw.array[dm_hdvh_raw.array == 0] = np.nan

    # Apply HDVL tech ratios to total HGV to get per-tech HDVH counts (verbatim from old script)
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

    # MPLW-based HDVH/HDVM split (consistent with fxa_freight_tech and freight_modal_share).
    # Old script split JRC LCV 40/60 into HDVL/HDVM — that was inconsistent with other
    # freight processors that derive HDVM/HDVH from JRC HGV via Eurostat MPLW shares.
    years_in_dm = dm_hdvh.col_labels["Years"]
    hdvm_sh_arr = np.array([mplw_shares[y][0] for y in years_in_dm])
    hdvh_sh_arr = np.array([mplw_shares[y][1] for y in years_in_dm])

    dm_hdvm = dm_hdvh.copy()
    dm_hdvm.rename_col("HDVH", "HDVM", "Variables")
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

    df_avi = pd.concat(
        [
            get_specific_jrc_data(code, name, 106, 107, "unit")
            for code, name in zip(country_codes, country_names)
        ],
        ignore_index=True,
    )
    dm_avi = DataMatrix.create_from_df(df_avi, 0)
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
        "variable": "New vehicles - total (representative train configuration)",
        "sheet_last_row": "Electric",
        "categories": "Freight transport",
        "sub_variables": ["Diesel oil", "Electric"],
        "calc_names": ["rail_ICE-diesel", "rail_CEV"],
    }
    dm_new_rail = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_new_rail.deepen()
    categories2_missing = _CATEGORIES2_ALL.copy()
    for cat in dm_new_rail.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_new_rail.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories1"
    )
    dm_new_rail.sort("Categories1")

    ########################
    ##### PUT TOGETHER #####
    ########################

    dm_new = dm_hdvl.copy()
    dm_new.append(dm_hdvh, "Variables")
    dm_new.append(dm_hdvm, "Variables")
    dm_new.sort("Variables")
    dm_new.add(np.nan, "Years", list(range(1990, 1999 + 1)), "number", True)
    dm_new.sort("Years")
    dm_new.sort("Categories1")
    dm_new.append(dm_avi, "Variables")
    dm_new_rail.sort("Categories1")
    dm_new_rail.add(np.nan, "Years", list(range(1990, 1999 + 1)), "number", True)
    dm_new_rail.sort("Years")
    dm_new.append(dm_new_rail, "Variables")
    dm_new.sort("Variables")
    dm_new.sort("Country")

    dm_new.array[dm_new.array == 0] = np.nan

    ###########################################################################
    # FIX OTS
    ###########################################################################

    dm_new = dm_new.flatten()

    idx = dm_new.idx
    dm_new.array[:, idx[2000], ...] = np.nan

    dict_new = {}

    def make_ots(dm, variable, periods_dicts, years_ots=None, min_t0=0.1, min_tb=0.1):
        dm_temp = dm.filter({"Variables": [variable]})
        if periods_dicts["n_adj"] == 1:
            dm_temp = linear_fitting(dm_temp, years_ots, min_t0=min_t0, min_tb=min_tb)
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

    dict_call = {
        "HDVH_BEV": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2021,
        },
        "HDVH_ICE-diesel": {"n_adj": 1},
        "HDVH_ICE-gas": {"n_adj": 1},
        "HDVH_ICE-gasoline": {"n_adj": 1, "min_t0": 0, "min_tb": 0},
        "HDVL_BEV": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2021,
        },
        "HDVL_ICE-diesel": {"n_adj": 1},
        "HDVL_ICE-gas": {"n_adj": 1},
        "HDVL_ICE-gasoline": {"n_adj": 1, "min_t0": 0, "min_tb": 0},
        "HDVM_BEV": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2021,
        },
        "HDVM_ICE-diesel": {"n_adj": 1},
        "HDVM_ICE-gas": {"n_adj": 1},
        "HDVM_ICE-gasoline": {"n_adj": 1, "min_t0": 0, "min_tb": 0},
        "aviation_kerosene": {"n_adj": 1},
        "rail_CEV": {"n_adj": 1},
        "rail_ICE-diesel": {"n_adj": 1},
    }

    for key in dict_call.keys():
        min_t0 = dict_call[key].pop("min_t0", 0.1)
        min_tb = dict_call[key].pop("min_tb", 0.1)
        if len(dict_call[key]) > 1:
            dict_new[key] = make_ots(
                dm_new, key, dict_call[key], min_t0=min_t0, min_tb=min_tb
            )
        else:
            dict_new[key] = make_ots(
                dm_new, key, dict_call[key], years_ots_ext, min_t0=min_t0, min_tb=min_tb
            )

    dm_new = dict_new["HDVH_BEV"].copy()
    mylist = list(dict_call.keys())
    mylist.remove("HDVH_BEV")
    for v in mylist:
        dm_new.append(dict_new[v], "Variables")
    dm_new.sort("Variables")

    ###########################################################################
    # FIX LAST YEARS FOR ELECTRIC VEHICLES (EU27 2022-2023)
    ###########################################################################

    # Use BEV *share* growth rate (not absolute count rate) to avoid folding in the
    # HGV total market recovery from the COVID dip (HGV registrations +31% in 2021).
    # Absolute count rate for HDVH/HDVM was ~107%; share growth rate is ~58% (same as
    # HDVL vans, since all three use HDVL tech ratios applied to HGV total).
    _mode_ice_vars = {
        "HDVH_BEV": ["HDVH_ICE-diesel", "HDVH_ICE-gas", "HDVH_ICE-gasoline"],
        "HDVL_BEV": ["HDVL_ICE-diesel", "HDVL_ICE-gas", "HDVL_ICE-gasoline"],
        "HDVM_BEV": ["HDVM_ICE-diesel", "HDVM_ICE-gas", "HDVM_ICE-gasoline"],
    }

    idx = dm_new.idx
    for v in ["HDVH_BEV", "HDVL_BEV", "HDVM_BEV"]:
        dm_new.array[idx["EU27"], idx[2022], idx[v]] = np.nan
        dm_new.array[idx["EU27"], idx[2023], idx[v]] = np.nan

    share_rates = {}
    for v_bev, ice_vars in _mode_ice_vars.items():
        bev_2020 = dm_new.array[idx["EU27"], idx[2020], idx[v_bev]]
        ice_2020 = np.nansum(
            [dm_new.array[idx["EU27"], idx[2020], idx[iv]] for iv in ice_vars]
        )
        bev_2021 = dm_new.array[idx["EU27"], idx[2021], idx[v_bev]]
        ice_2021 = np.nansum(
            [dm_new.array[idx["EU27"], idx[2021], idx[iv]] for iv in ice_vars]
        )
        share_2020 = bev_2020 / (bev_2020 + ice_2020)
        share_2021 = bev_2021 / (bev_2021 + ice_2021)
        share_rates[v_bev] = (share_2021 - share_2020) / share_2020

    for v_bev, ice_vars in _mode_ice_vars.items():
        rate = share_rates[v_bev]
        bev_2021 = dm_new.array[idx["EU27"], idx[2021], idx[v_bev]]
        ice_2021 = np.nansum(
            [dm_new.array[idx["EU27"], idx[2021], idx[iv]] for iv in ice_vars]
        )
        share_2021 = bev_2021 / (bev_2021 + ice_2021)
        share_2022 = share_2021 * (1 + rate)
        ice_2022 = np.nansum(
            [dm_new.array[idx["EU27"], idx[2022], idx[iv]] for iv in ice_vars]
        )
        dm_new.array[idx["EU27"], idx[2022], idx[v_bev]] = (
            share_2022 * ice_2022 / (1 - share_2022)
        )
        share_2023 = share_2022 * (1 + rate)
        ice_2023 = np.nansum(
            [dm_new.array[idx["EU27"], idx[2023], idx[iv]] for iv in ice_vars]
        )
        dm_new.array[idx["EU27"], idx[2023], idx[v_bev]] = (
            share_2023 * ice_2023 / (1 - share_2023)
        )

    ###########################################################################
    # MAKE AS FINAL DATAMATRIX
    ###########################################################################

    dm_new.deepen()

    for v in dm_new.col_labels["Variables"]:
        dm_new.rename_col(v, "tra_freight_technology-share_new_" + v, "Variables")
    dm_new.deepen(based_on="Variables")
    dm_new.switch_categories_order("Categories1", "Categories2")
    dm_new_pc = dm_new.copy()
    dm_new_pc.normalise("Categories2")

    categories1_missing = [
        c for c in _CATEGORIES1_ALL if c not in dm_new_pc.col_labels["Categories1"]
    ]
    categories2_missing = [
        c for c in _CATEGORIES2_ALL if c not in dm_new_pc.col_labels["Categories2"]
    ]
    dm_new_pc.add(np.nan, col_label=categories1_missing, dummy=True, dim="Categories1")
    dm_new_pc.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories2")
    dm_new_pc.sort("Categories1")
    dm_new_pc.sort("Categories2")
    dm_new_pc_final = dm_new_pc.copy()

    dm_new_pc_final[:, :, :, "IWW", "ICE"] = 1
    dm_new_pc_final[:, :, :, "marine", "ICE"] = 1

    ###########################################################################
    # OTS ONLY
    ###########################################################################

    dm_new_ots = dm_new_pc_final.filter({"Years": years_ots})

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(dm_new_ots, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dm_new_ots


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_fts, years_ots

    run(years_ots, years_fts)
