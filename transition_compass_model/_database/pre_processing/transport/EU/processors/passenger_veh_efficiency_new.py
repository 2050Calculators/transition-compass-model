import os
import pickle
import warnings

import numpy as np
import pandas as pd

from transition_compass_model._database.pre_processing.routine_JRC import get_jrc_data
from transition_compass_model.model.common.auxiliary_functions import (
    jrc_iso2_dict,
    linear_fitting,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(
    _THIS_DIR, "../data/datamatrix/passenger_veh_efficiency_new.pickle"
)

# all tech categories used across modes — FCEV and H2 are added separately at the end
_CATEGORIES2_ALL = [
    "BEV",
    "CEV",
    "ICE-diesel",
    "ICE-gas",
    "ICE-gasoline",
    "PHEV-diesel",
    "PHEV-gasoline",
    "kerosene",
    "mt",
]


def _get_aviation_jrc(
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


def run(years_ots: list):
    """
    Build passenger vehicle-efficiency (new vehicles) OTS for EU27.

    Sources
    -------
    - 2W/LDV/bus: JRC-IDEES TrRoad_tech "Test cycle efficiency of new vehicles (kgoe/100 km)"
    - Rail: JRC-IDEES TrRail_ene stock efficiency, adjusted by bus new/stock ratio
    - Aviation: JRC-IDEES EU27 aggregated Aviation file — energy efficiency (kgoe/100 km per
      flight) divided by seats per flight → efficiency per seat-km (kgoe/100 skm)

    Accounting basis
    ----------------
    Vehicle efficiency is a physical property of the vehicle; it is independent of residency
    vs territorial accounting. No residency correction needed. For EU27 as a large bloc,
    territorial ≈ residency regardless.

    Returns
    -------
    dm_eneff_new_ots : DataMatrix
        OTS efficiency (Variable='tra_passenger_veh-efficiency_new', Cat1=modes, Cat2=techs,
        unit=MJ/km), 1990-2023.
    """
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    dict_iso2_jrc = jrc_iso2_dict()

    ##############
    ##### 2W #####
    ##############

    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_tech",
        "variable": "Test cycle efficiency of new vehicles (kgoe/100 km)",
        "sheet_last_row": "Powered two-wheelers",
        "sub_variables": ["Powered two-wheelers"],
        "calc_names": ["2W"],
    }
    dm_eneff_new_2w = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)

    # 2W are mostly ICE-gasoline; rest assigned NaN
    dm_eneff_new_2w.rename_col("2W", "2W_ICE-gasoline", "Variables")
    dm_eneff_new_2w.deepen()
    categories2_missing = _CATEGORIES2_ALL.copy()
    for cat in dm_eneff_new_2w.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_eneff_new_2w.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories1"
    )
    dm_eneff_new_2w.sort("Categories1")

    ###############
    ##### LDV #####
    ###############

    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_tech",
        "variable": "Test cycle efficiency of new vehicles (kgoe/100 km)",
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
    dm_eneff_new_ldv = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)

    # deepen and mean-aggregate LPG + NG → ICE-gas
    dm_eneff_new_ldv.deepen()
    mapping_calc = {"ICE-gas": ["gas-lpg", "gas-natural"]}
    dm_eneff_new_ldv.groupby(
        mapping_calc, dim="Categories1", aggregation="mean", regex=False, inplace=True
    )

    # split PHEV into gasoline and diesel (same efficiency assumed)
    dm_temp = dm_eneff_new_ldv.filter({"Categories1": ["PHEV"]})
    dm_temp1 = dm_temp.copy()
    dm_temp1.rename_col("PHEV", "PHEV-gasoline", "Categories1")
    dm_temp1.append(dm_temp, "Categories1")
    dm_temp1.rename_col("PHEV", "PHEV-diesel", "Categories1")
    dm_eneff_new_ldv.drop("Categories1", "PHEV")
    dm_eneff_new_ldv.append(dm_temp1, "Categories1")
    dm_eneff_new_ldv.sort("Categories1")

    # FCEV and remaining techs: NaN for now
    categories2_missing = _CATEGORIES2_ALL.copy()
    for cat in dm_eneff_new_ldv.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_eneff_new_ldv.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories1"
    )
    dm_eneff_new_ldv.sort("Categories1")

    #################
    ##### BUSES #####
    #################

    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_tech",
        "variable": "Test cycle efficiency of new vehicles (kgoe/100 km)",
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
    dm_eneff_new_bus = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)

    # deepen and mean-aggregate gas
    dm_eneff_new_bus.deepen()
    mapping_calc = {"ICE-gas": ["gas-lpg", "gas-natural"]}
    dm_eneff_new_bus.groupby(
        mapping_calc, dim="Categories1", aggregation="mean", regex=False, inplace=True
    )

    categories2_missing = _CATEGORIES2_ALL.copy()
    for cat in dm_eneff_new_bus.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_eneff_new_bus.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories1"
    )
    dm_eneff_new_bus.sort("Categories1")

    ################
    ##### RAIL #####
    ################

    # use stock efficiency (no "new vehicles" series for rail in JRC TrRail_tech)
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

    # aggregate conventional + high-speed electric; rename diesel
    mapping_calc = {
        "rail_CEV": ["train-conv_CEV", "train-hs_CEV"],
        "rail_ICE-diesel": ["train-conv_ICE-diesel"],
    }
    dm_eneff_rail.groupby(
        mapping_calc, dim="Variables", aggregation="mean", regex=False, inplace=True
    )

    # adjust stock efficiency by bus new/stock ratio to approximate new-vehicle efficiency
    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_ene",
        "variable": "Vehicle-efficiency - effective (kgoe/100 km)",
        "sheet_last_row": "Motor coaches, buses and trolley buses",
        "sub_variables": ["Motor coaches, buses and trolley buses"],
        "calc_names": ["bus"],
    }
    dm_eneff_bus = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_eneff_bus.array[dm_eneff_bus.array == 0] = np.nan
    dm_new_bus_avg = dm_eneff_new_bus.copy()
    dm_new_bus_avg.group_all(dim="Categories1", aggregation="mean")
    arr_ratio = dm_new_bus_avg.array / dm_eneff_bus.array
    dm_eneff_rail.array = dm_eneff_rail.array * arr_ratio

    dm_eneff_new_rail = dm_eneff_rail.copy()

    # deepen and add missing techs as NaN
    dm_eneff_new_rail.deepen()
    categories2_missing = _CATEGORIES2_ALL.copy()
    for cat in dm_eneff_new_rail.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_eneff_new_rail.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories1"
    )
    dm_eneff_new_rail.sort("Categories1")

    ####################
    ##### AVIATION #####
    ####################

    # No "new vehicle" efficiency in JRC TrAvia — use fleet efficiency per seat-km:
    # efficiency (kgoe/100 km per flight) / seats per flight = kgoe/100 skm

    country_codes = list(dict_iso2_jrc.keys())
    country_names = list(dict_iso2_jrc.values())

    df_eneff_avi = pd.concat(
        [
            _get_aviation_jrc(code, name, 134, 135, "kgoe/100km", "efficiency")
            for code, name in zip(country_codes, country_names)
        ],
        ignore_index=True,
    )
    dm_eneff_avi = DataMatrix.create_from_df(df_eneff_avi, 0)

    df_seats = pd.concat(
        [
            _get_aviation_jrc(code, name, 273, 274, "number", "nseats")
            for code, name in zip(country_codes, country_names)
        ],
        ignore_index=True,
    )
    dm_seats_avi = DataMatrix.create_from_df(df_seats, 0)

    # efficiency per seat-km
    dm_eneff_avi.append(dm_seats_avi, "Variables")
    dm_eneff_avi.operation(
        "efficiency", "/", "nseats", out_col="aviation_kerosene", unit="kgoe/100 km"
    )
    dm_eneff_avi = dm_eneff_avi.filter({"Variables": ["aviation_kerosene"]})

    dm_eneff_avi.deepen()
    categories2_missing = _CATEGORIES2_ALL.copy()
    for cat in dm_eneff_avi.col_labels["Categories1"]:
        categories2_missing.remove(cat)
    dm_eneff_avi.add(
        np.nan, col_label=categories2_missing, dummy=True, dim="Categories1"
    )
    dm_eneff_avi.sort("Categories1")

    ########################
    ##### PUT TOGETHER #####
    ########################

    dm_eneff_new = dm_eneff_new_2w.copy()
    dm_eneff_new.append(dm_eneff_new_ldv, "Variables")
    dm_eneff_new.append(dm_eneff_new_bus, "Variables")
    dm_eneff_new.append(dm_eneff_new_rail, "Variables")
    dm_eneff_new.sort("Variables")
    dm_eneff_new.sort("Country")

    # road/rail modes have JRC data from 2000; add 1990-1999 as NaN before appending aviation
    dm_eneff_new.add(np.nan, "Years", list(range(1990, 2000)), dummy=True)
    dm_eneff_new.append(dm_eneff_avi, "Variables")
    dm_eneff_new.sort("Variables")

    # replace zeros with NaN
    dm_eneff_new.array[dm_eneff_new.array == 0] = np.nan

    ###################
    ##### FIX OTS #####
    ###################

    dm_eneff_new = dm_eneff_new.flatten()

    # NaN at 2000 for modes with unreliable first data point
    idx = dm_eneff_new.idx
    for v in ["bus_ICE-gas", "metrotram_mt", "rail_CEV", "rail_ICE-diesel"]:
        dm_eneff_new.array[:, idx[2000], idx[v]] = np.nan

    # linear fitting fills 1990-1999 (back-extrapolation) and 2022-2023 (forward)
    dm_eneff_new = linear_fitting(dm_eneff_new, years_ots, min_t0=0, min_tb=0)

    ####################################
    ##### MAKE AS FINAL DATAMATRIX #####
    ####################################

    for v in dm_eneff_new.col_labels["Variables"]:
        dm_eneff_new.rename_col(v, "tra_passenger_veh-efficiency_new_" + v, "Variables")
    dm_eneff_new.deepen()
    dm_eneff_new.deepen(based_on="Variables")
    dm_eneff_new.switch_categories_order("Categories1", "Categories2")

    dm_eneff_new.change_unit(
        "tra_passenger_veh-efficiency_new", 41.868, "kgoe/100 km", "MJ/100 km"
    )
    dm_eneff_new.change_unit(
        "tra_passenger_veh-efficiency_new", 1e-2, "MJ/100 km", "MJ/km"
    )

    dm_eneff_new.add(np.nan, col_label="FCEV", dummy=True, dim="Categories2")
    dm_eneff_new.sort("Categories2")

    dm_eneff_new.add(np.nan, "Categories2", "H2", "MJ/km", True)
    dm_eneff_new.sort("Categories2")

    ###############
    ##### OTS #####
    ###############

    dm_eneff_new_ots = dm_eneff_new.filter({"Years": years_ots})

    ################
    ##### SAVE #####
    ################

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(dm_eneff_new_ots, f)
    return dm_eneff_new_ots


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_ots

    run(years_ots)
