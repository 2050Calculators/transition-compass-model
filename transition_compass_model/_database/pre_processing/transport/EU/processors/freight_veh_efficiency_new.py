"""
EU27 freight new-vehicle energy efficiency OTS (1990-2023).

Faithful port of transport_lever_freight_vehicle-efficiency_new.py. One structural change:
  OTS only — FTS is generated in scenarios/transport_fts_EU.py (all 4 levels BAU).
  # TODO: differentiate FTS ambition levels for freight vehicle efficiency.

No MPLW split needed here: HDVM efficiency = mean(HDVL, HDVH) regardless of fleet-count
shares. The MPLW correction applies to counts (tech_share_new, fleet), not to efficiency values.

Rail efficiency uses a new/stock ratio correction from HDVH trucks:
  rail_new_efficiency = rail_stock_efficiency × (mean_HDVH_new / mean_HDVH_stock)
Assumption: the new-vehicle efficiency improvement factor is the same for rail as for trucks.

Accounting basis
----------------
New-vehicle energy efficiency by mode and tech. JRC "Test cycle efficiency" is independent
of territorial vs residency accounting (it is a physical vehicle property). No residency
correction needed.

Sources
-------
Road HDV:  JRC-IDEES-2021 Transport, TrRoad_tech, "Test cycle efficiency of new vehicles"
Rail:      JRC-IDEES-2021 Transport, TrRail_ene, "Vehicle-efficiency (kgoe/100 km)"
           corrected by HDVH new/stock ratio from TrRoad_ene
IWW:       JRC-IDEES-2021 Navigation_Domestic, rows 46-47
Aviation:  JRC-IDEES-2021 Aviation_EU, rows 138-139
Maritime:  JRC-IDEES-2021 Navigation_International_EU, rows 46-47
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd

from transition_compass_model._database.pre_processing.routine_JRC import get_jrc_data
from transition_compass_model._database.pre_processing.transport.EU.processors.fxa_freight_tech import (
    _compute_non_road_load_factors,
)
from transition_compass_model.model.common.auxiliary_functions import (
    jrc_iso2_dict,
    linear_fitting,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(
    _THIS_DIR, "../data/datamatrix/freight_veh_efficiency_new.pickle"
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
    Build freight new-vehicle energy efficiency DataMatrix for EU27 (OTS only).

    Parameters
    ----------
    years_ots, years_fts : list

    Returns
    -------
    dm_eneff_new_ots : DataMatrix
        Variable='tra_freight_vehicle-efficiency_new', Cat1=modes, Cat2=techs,
        units=MJ/km. OTS only.
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

    ###########################################################################
    # GET DATA
    ###########################################################################

    ################
    ##### HDVL #####
    ################

    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_tech",
        "variable": "Test cycle efficiency of new vehicles (kgoe/100 km)",
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
        "sheet": "TrRoad_tech",
        "variable": "Test cycle efficiency of new vehicles (kgoe/100 km)",
        "sheet_last_row": "Heavy goods vehicles",
        "sub_variables": ["Heavy goods vehicles"],
        "calc_names": ["HDVH"],
    }
    dm_hdvh_raw = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_hdvh_raw.array[dm_hdvh_raw.array == 0] = np.nan

    # Apply HDVL tech efficiency ratios to total HGV efficiency to get per-tech HDVH values
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

    # HDVM efficiency = mean of HDVL and HDVH efficiency per tech (no MPLW split —
    # efficiency is a physical property of the vehicle, not a count)
    dm_eneff_new = dm_hdvl.copy()
    dm_eneff_new.append(dm_hdvh, "Variables")
    dm_hdvm = dm_eneff_new.flatten()
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

    dm_eneff_new.append(dm_hdvm, "Variables")
    dm_eneff_new.sort("Variables")
    dm_eneff_new.sort("Country")
    dm_eneff_new.sort("Years")

    ###########################################################################
    # IWW / AVIATION / MARITIME
    ###########################################################################

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

    # Correct rail stock efficiency to new-vehicle efficiency using HDVH new/stock ratio.
    # Assumption: new vehicles are more efficient than stock by the same factor as HDVH trucks.
    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_ene",
        "variable": "Vehicle-efficiency - effective (kgoe/100 km)",
        "sheet_last_row": "Heavy goods vehicles",
        "sub_variables": ["Heavy goods vehicles"],
        "calc_names": ["HDVH"],
    }
    dm_eneff_hdvh_stock = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_eneff_hdvh_stock.array[dm_eneff_hdvh_stock.array == 0] = np.nan

    dm_hdvh_mean = dm_hdvh.copy()
    dm_hdvh_mean.group_all(dim="Categories1", aggregation="mean")
    arr_ratio = dm_hdvh_mean.array / dm_eneff_hdvh_stock.array
    dm_eneff_rail.array = dm_eneff_rail.array * arr_ratio[..., np.newaxis]
    dm_eneff_new_rail = dm_eneff_rail.copy()

    categories2_missing = _CATEGORIES2_ALL.copy()
    for cat in dm_eneff_new_rail.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_eneff_new_rail.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories1"
    )
    dm_eneff_new_rail.sort("Categories1")

    ########################
    ##### PUT TOGETHER #####
    ########################

    dm_eneff_new.add(np.nan, "Years", list(range(1990, 1999 + 1)), dummy=True)
    dm_eneff_new.sort("Years")
    dm_eneff_new.append(dm_iww, "Variables")
    dm_eneff_new.append(dm_avi, "Variables")
    dm_eneff_new.append(dm_mar, "Variables")
    dm_eneff_new_rail.add(np.nan, "Years", list(range(1990, 1999 + 1)), dummy=True)
    dm_eneff_new_rail.sort("Years")
    dm_eneff_new.append(dm_eneff_new_rail, "Variables")
    dm_eneff_new.sort("Variables")
    dm_eneff_new.sort("Country")

    dm_eneff_new.array[dm_eneff_new.array == 0] = np.nan

    ###########################################################################
    # FIX OTS
    ###########################################################################

    dm_eneff_new = dm_eneff_new.flatten()

    dict_new = {}

    def make_ots(dm, variable, periods_dicts, years_ots=None):
        dm_temp = dm.filter({"Variables": [variable]})
        if periods_dicts["n_adj"] == 1:
            dm_temp = linear_fitting(dm_temp, years_ots, min_t0=0.1, min_tb=0.1)
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
        "HDVH_BEV": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2010,
        },
        "HDVH_ICE-diesel": {"n_adj": 1},
        "HDVH_ICE-gas": {"n_adj": 1},
        "HDVH_ICE-gasoline": {"n_adj": 1},
        "HDVL_BEV": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2020,
        },
        "HDVL_ICE-diesel": {"n_adj": 1},
        "HDVL_ICE-gas": {
            "n_adj": 2,
            "year_end_first_adj": 2015,
            "year_start_second_adj": 2020,
        },
        "HDVL_ICE-gasoline": {"n_adj": 1},
        "HDVM_BEV": {
            "n_adj": 2,
            "year_end_first_adj": 2010,
            "year_start_second_adj": 2015,
        },
        "HDVM_ICE-diesel": {"n_adj": 1},
        "HDVM_ICE-gas": {"n_adj": 1},
        "HDVM_ICE-gasoline": {"n_adj": 1},
        "IWW_ICE": {"n_adj": 1},
        "aviation_kerosene": {"n_adj": 1},
        "marine_ICE": {"n_adj": 1},
        "rail_CEV": {"n_adj": 1},
        "rail_ICE-diesel": {"n_adj": 1},
    }

    for key in dict_call.keys():
        if len(dict_call[key]) > 1:
            dict_new[key] = make_ots(dm_eneff_new, key, dict_call[key])
        else:
            dict_new[key] = make_ots(dm_eneff_new, key, dict_call[key], years_ots_ext)

    dm_eneff_new = dict_new["HDVH_BEV"].copy()
    mylist = list(dict_call.keys())
    mylist.remove("HDVH_BEV")
    for v in mylist:
        dm_eneff_new.append(dict_new[v], "Variables")
    dm_eneff_new.sort("Variables")

    ###########################################################################
    # MAKE AS FINAL DATAMATRIX
    ###########################################################################

    dm_eneff_new.deepen()

    for v in dm_eneff_new.col_labels["Variables"]:
        dm_eneff_new.rename_col(
            v, "tra_freight_vehicle-efficiency_new_" + v, "Variables"
        )
    dm_eneff_new.deepen(based_on="Variables")
    dm_eneff_new.switch_categories_order("Categories1", "Categories2")

    dm_eneff_new.change_unit(
        "tra_freight_vehicle-efficiency_new", 41.868, "kgoe/100 km", "MJ/100 km"
    )
    dm_eneff_new.change_unit(
        "tra_freight_vehicle-efficiency_new", 1e-2, "MJ/100 km", "MJ/km"
    )

    # Non-road modes: divide by load factor (tkm/vkm) to convert MJ/vkm → MJ/tkm.
    load_factors = _compute_non_road_load_factors(years_ots_ext)
    lf_year_idx = {y: i for i, y in enumerate(years_ots_ext)}
    for mode, lf_arr in load_factors.items():
        if mode not in dm_eneff_new.col_labels["Categories1"]:
            continue
        i_mode = dm_eneff_new.col_labels["Categories1"].index(mode)
        lf_for_dm = np.array(
            [lf_arr[lf_year_idx[y]] for y in dm_eneff_new.col_labels["Years"]]
        )
        dm_eneff_new.array[:, :, :, i_mode, :] /= lf_for_dm.reshape(1, -1, 1, 1)

    categories2_missing = [
        c for c in _CATEGORIES2_ALL if c not in dm_eneff_new.col_labels["Categories2"]
    ]
    dm_eneff_new.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories2"
    )
    dm_eneff_new.sort("Categories2")

    ###########################################################################
    # OTS ONLY
    ###########################################################################

    dm_eneff_new_ots = dm_eneff_new.filter({"Years": years_ots})

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(dm_eneff_new_ots, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dm_eneff_new_ots


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_fts, years_ots

    run(years_ots, years_fts)
