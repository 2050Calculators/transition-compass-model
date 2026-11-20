"""
EU27 fuel-mix (biofuel/efuel share) OTS lever (1990-2023).

Faithful port of transport_lever_fuel-mix.py. Structural changes only:
  1. No FTS — generated in scenarios/transport_fts_EU.py (all 4 levels BAU).
  2. Dependency on existing transport.pickle removed (was only used for inspection).
  3. Wrapped in run(); _THIS_DIR used in place of os.getcwd().

Sources
-------
Road:      JRC-IDEES-2021 Transport TrRoad_ene, biofuel share of total road energy
IWW:       JRC-IDEES-2021 Transport TrNavi_ene, biofuel (blended+biogas) share
Aviation:  External: 0.24 Mt SAF (EAFO 2021 EU27 estimate), scaled to EU total aviation energy.
           Trend: shaped by biofuel_rail trend normalised at 2021.
Maritime:  JRC-IDEES-2021 Transport MBunk_ene, biofuel share of intra-EEA maritime
Rail:      JRC-IDEES-2021 Transport TrRail_ene, biofuel share

OTS fix: IWW and marine backfilled 1989-1999 from 2000 value; 2022-2023 extrapolated
         from recent datapoints. Road/rail/aviation: linear_fitting over full OTS window.

Final structure
---------------
Variable : tra_fuel-mix
Cat1     : [IWW, aviation, marine, rail, road]  (mode)
Cat2     : [biofuel, efuel]                      (fuel type)
Unit     : %
"""

import os
import pickle
import warnings

import numpy as np

from transition_compass_model._database.pre_processing.routine_JRC import get_jrc_data
from transition_compass_model.model.common.auxiliary_functions import (
    eurostat_iso2_dict,
    jrc_iso2_dict,
    linear_fitting,
)

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(_THIS_DIR, "../data/datamatrix/fuel_mix.pickle")


def run(years_ots, years_fts):
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    years_ots_ext = list(range(1989, years_ots[-1] + 1))
    startyear = years_ots_ext[0]

    # get iso codes
    dict_iso2 = eurostat_iso2_dict()
    dict_iso2.pop("CH")  # Remove Switzerland
    dict_iso2_jrc = jrc_iso2_dict()

    ################
    ##### road #####
    ################

    # get data
    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_ene",
        "variable": "by fuel (EUROSTAT DATA)",
        "sheet_last_row": "Renewable energies and wastes",
        "sub_variables": ["by fuel (EUROSTAT DATA)", "Renewable energies and wastes"],
        "calc_names": ["total", "biofuel"],
    }
    dm_road = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)

    # make the share
    dm_road.operation(
        "biofuel", "/", "total", "Variables", out_col="biofuel-share", unit="%"
    )
    dm_road.drop("Variables", ["biofuel", "total"])
    dm_road.rename_col("biofuel-share", "biofuel_road", "Variables")

    ###############
    ##### IWW #####
    ###############

    # get data on energy efficiency
    dict_extract = {
        "database": "Transport",
        "sheet": "TrNavi_ene",
        "variable": "Energy consumption (ktoe)",
        "categories": "Inland waterways",
        "sheet_last_row": "Biogases",
        "sub_variables": ["Inland waterways", "Blended biofuels", "Biogases"],
        "calc_names": ["total", "biofuel-blended", "biofuel-biogas"],
    }
    dm_iww = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)

    # make the share
    dm_iww.groupby(
        {"biofuel": ["biofuel-biogas", "biofuel-blended"]},
        "Variables",
        "sum",
        inplace=True,
    )
    dm_iww.operation(
        "biofuel", "/", "total", "Variables", out_col="biofuel-share", unit="%"
    )
    dm_iww.drop("Variables", ["biofuel", "total"])
    dm_iww.rename_col("biofuel-share", "biofuel_IWW", "Variables")

    ####################
    ##### aviation #####
    ####################

    # get data on energy efficiency
    dict_extract = {
        "database": "Transport",
        "sheet": "TrAvia_ene",
        "variable": "Energy consumption (ktoe)",
        "sheet_last_row": "Energy consumption (ktoe)",
        "sub_variables": ["Energy consumption (ktoe)"],
        "calc_names": ["total"],
    }
    dm_avi_tot = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)

    # source: https://alternative-fuels-observatory.ec.europa.eu/transport-mode/aviation/general-information-and-context
    # 0.24 million tonnes
    # 1 tonne of SAF is approximately 1.03 toe.
    value = 0.24 * 1e6 * 1.03 * 1e-3
    dm_avi = dm_iww.filter({"Variables": ["biofuel_IWW"]})
    dm_avi.rename_col("biofuel_IWW", "biofuel", "Variables")
    idx = dm_avi.idx
    dm_avi.array[dm_avi.array > 0] = 0
    dm_avi.array[idx["EU27"], idx[2021], :] = value
    dm_avi.units["biofuel"] = "ktoe"

    # put together
    dm_avi.append(dm_avi_tot, "Variables")

    # make the share
    dm_avi.operation(
        "biofuel", "/", "total", "Variables", out_col="biofuel-share", unit="%"
    )
    dm_avi.drop("Variables", ["biofuel", "total"])
    dm_avi.rename_col("biofuel-share", "biofuel_aviation", "Variables")

    ####################
    ##### maritime #####
    ####################

    # get data on energy efficiency
    dict_extract = {
        "database": "Transport",
        "sheet": "MBunk_ene",
        "variable": "Total energy consumption (ktoe)",
        "categories": "Intra-EEA",
        "sheet_last_row": "Biogases",
        "sub_variables": ["Intra-EEA", "Blended biofuels", "Biogases"],
        "calc_names": ["total", "biofuel-blended", "biofuel-biogas"],
    }
    dm_mar = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)

    # make the share
    dm_mar.groupby(
        {"biofuel": ["biofuel-biogas", "biofuel-blended"]},
        "Variables",
        "sum",
        inplace=True,
    )
    dm_mar.operation(
        "biofuel", "/", "total", "Variables", out_col="biofuel-share", unit="%"
    )
    dm_mar.drop("Variables", ["biofuel", "total"])
    dm_mar.rename_col("biofuel-share", "biofuel_marine", "Variables")

    ################
    ##### RAIL #####
    ################

    # get data
    dict_extract = {
        "database": "Transport",
        "sheet": "TrRail_ene",
        "variable": "Energy consumption (ktoe)",
        "sheet_last_row": "blended liquid biofuels",
        "sub_variables": ["by fuel", "blended liquid biofuels"],
        "calc_names": ["total", "biofuel"],
    }
    dm_rail = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)

    # make the share
    dm_rail.operation(
        "biofuel", "/", "total", "Variables", out_col="biofuel-share", unit="%"
    )
    dm_rail.drop("Variables", ["biofuel", "total"])
    dm_rail.rename_col("biofuel-share", "biofuel_rail", "Variables")

    ########################
    ##### PUT TOGETHER #####
    ########################

    dm_biof = dm_road.copy()
    dm_biof.append(dm_iww, "Variables")
    dm_biof.append(dm_avi, "Variables")
    dm_biof.append(dm_mar, "Variables")
    dm_biof.append(dm_rail, "Variables")
    dm_biof.deepen()
    dm_biof.add(
        0, col_label="efuel", dummy=True, dim="Variables", unit="%"
    )  # assuming efuel at 0
    dm_biof.sort("Variables")
    dm_biof.sort("Country")

    # fix aviation with same trend of biofuel rail
    dm_temp = dm_biof.filter({"Variables": ["biofuel"], "Categories1": ["rail"]})
    dm_temp = dm_temp.flatten()
    idx = dm_temp.idx
    arr_temp = dm_temp.array[:, idx[2021], :]
    dm_temp.array = dm_temp.array / arr_temp[:, np.newaxis, :]
    dm_temp_avi = dm_biof.filter(
        {"Variables": ["biofuel"], "Categories1": ["aviation"]}
    ).flatten()
    idx = dm_temp_avi.idx
    arr_temp = dm_temp_avi.array[:, idx[2021], :]
    dm_temp_avi.array = arr_temp[:, np.newaxis, :] * dm_temp.array
    dm_temp_avi.array[np.isnan(dm_temp_avi.array)] = 0
    dm_biof = dm_biof.flatten()
    dm_biof.drop("Variables", "biofuel_aviation")
    dm_biof.append(dm_temp_avi, "Variables")
    dm_biof.deepen()

    ###################
    ##### FIX OTS #####
    ###################

    dm_iww_ots = dm_biof.filter({"Categories1": ["IWW"]})
    dm_mar_ots = dm_biof.filter({"Categories1": ["marine"]})
    dm_biof.drop("Categories1", ["IWW", "marine"])

    dm_biof = linear_fitting(dm_biof, years_ots_ext, min_t0=0, min_tb=0)
    dm_mar_ots = linear_fitting(
        dm_mar_ots,
        list(range(startyear, 1999 + 1)),
        min_t0=0,
        min_tb=0,
        based_on=[2000],
    )
    dm_mar_ots = linear_fitting(
        dm_mar_ots, list(range(2022, 2023 + 1)), min_t0=0, min_tb=0, based_on=[2020]
    )
    dm_iww_ots = linear_fitting(
        dm_iww_ots,
        list(range(startyear, 1999 + 1)),
        min_t0=0,
        min_tb=0,
        based_on=[2000],
    )
    dm_iww_ots = linear_fitting(
        dm_iww_ots, list(range(2022, 2023 + 1)), min_t0=0, min_tb=0, based_on=[2019]
    )
    dm_biof.append(dm_mar_ots, "Categories1")
    dm_biof.append(dm_iww_ots, "Categories1")
    dm_biof.sort("Categories1")

    ####################################
    ##### MAKE AS FINAL DATAMATRIX #####
    ####################################

    # rename and deepen
    for v in dm_biof.col_labels["Variables"]:
        dm_biof.rename_col(v, "tra_fuel-mix_" + v, "Variables")
    dm_biof.deepen(based_on="Variables")
    dm_biof.switch_categories_order("Categories1", "Categories2")

    dm_fuelmix_ots = dm_biof.filter({"Years": years_ots})

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(dm_fuelmix_ots, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dm_fuelmix_ots


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from transition_compass_model._database.pre_processing.transport.EU.params import (
        years_fts,
        years_ots,
    )

    run(years_ots, years_fts)
