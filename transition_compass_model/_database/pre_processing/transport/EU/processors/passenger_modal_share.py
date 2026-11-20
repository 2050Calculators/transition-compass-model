"""
EU27 passenger modal share lever.

Data source
-----------
JRC-IDEES-2021 Transport files (individual country Transport xlsx):
  - TrRoad_act: 2W, LDV, bus pkm (2000–2021)
  - TrRail_act: metrotram, rail pkm (2000–2021)

Accounting basis — TERRITORIAL (known limitation)
--------------------------------------------------
JRC Transport data counts pkm on EU27 infrastructure (roads, railways), not
pkm by EU27 residents. This is a territorial measure, not a residency measure.

For CH preprocessing, residency-based data comes from the MRMT household travel
survey (FSO), which asks residents about all their travel including abroad. No
equivalent harmonised EU-wide household travel survey exists in a single
standardised dataset:

  - tran_hv_psmod: modal split percentages only, mixed methodology (some
    countries use traffic counts, others use surveys), no 2W, no metrotram split
  - tour_dem_tttr: residency-based but tourism trips only (>=1 overnight stay),
    excludes commuting/day trips which dominate road pkm; no pkm unit; 2012+ only
  - National surveys (MiD, ENTD, OViN, etc.): not harmonised across EU27

For EU27 as a large bloc, territorial ≈ residency (non-EU residents' ground
travel in EU27 is a small fraction of total). For individual small countries
with high cross-border flows (LU, BE, AT, NL), territorial road pkm may
overstate residents' actual travel, inflating LDV/bus shares and understating
rail share. Since build_pickle filters to EU27 aggregate, this does not affect
model output under the current setup.
"""

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

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(_THIS_DIR, "../data/datamatrix/passenger_modal_share.pickle")


def run(years_ots: list, years_fts: list):
    """
    Build passenger modal share lever for EU27.

    Returns
    -------
    DM_mod : dict
        {"ots": {"passenger_modal-share": dm}, "fts": {"passenger_modal-share": {1..4: dm}}}
        Modal share percentages (normalised), 4 FTS ambition levels.
    DM_pkm : dict
        {"ots": {"passenger_pkm": dm}, "fts": {"passenger_pkm": {1..4: dm}}}
        Absolute pkm (BAU, same for all levels), intermediate for occupancy processor.
    """
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    startyear = 1989

    ################################################
    ################### GET DATA ###################
    ################################################

    dict_iso2_jrc = jrc_iso2_dict()

    ########################
    ##### LDV, 2W, BUS #####
    ########################

    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_act",
        "variable": "Passenger transport (mio pkm)",
        "sheet_last_row": "Motor coaches, buses and trolley buses",
        "sub_variables": [
            "Powered two-wheelers",
            "Passenger cars",
            "Motor coaches, buses and trolley buses",
        ],
        "calc_names": ["2W", "LDV", "bus"],
    }
    dm_pkm_ltb = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_pkm_ltb.sort("Variables")
    dm_pkm_ltb.sort("Country")

    ################
    ##### RAIL #####
    ################

    dict_extract = {
        "database": "Transport",
        "sheet": "TrRail_act",
        "variable": "Passenger transport (mio pkm)",
        "sheet_last_row": "High speed passenger trains",
        "sub_variables": [
            "Metro and tram, urban light rail",
            "Conventional passenger trains",
            "High speed passenger trains",
        ],
        "calc_names": ["metrotram", "train-conv", "train-hs"],
    }
    dm_pkm_r = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_pkm_r.sort("Country")
    mapping_calc = {"rail": ["train-conv", "train-hs"]}
    dm_pkm_r.groupby(
        mapping_calc, dim="Variables", aggregation="sum", regex=False, inplace=True
    )

    ########################
    ##### PUT TOGETHER #####
    ########################

    dm_pkm = dm_pkm_ltb.copy()
    dm_pkm.append(dm_pkm_r, "Variables")
    dm_pkm.sort("Variables")

    ###################
    ##### FIX OTS #####
    ###################

    # add years not in JRC Transport file (1989-1999 and 2022-2023)
    missing_years = list(range(startyear, 2000)) + [2022, 2023]
    dm_pkm.add(np.nan, col_label=missing_years, dummy=True, dim="Years")
    dm_pkm.sort("Years")

    years_fitting = list(range(startyear, 1999 + 1))

    # before 2000: 2W uses 2000-2014 trend (peaked before 2015), others use 2000-2018
    dm_2w = dm_pkm.filter({"Variables": ["2W"]})
    dm_pkm.drop("Variables", ["2W"])
    dm_pkm = linear_fitting(dm_pkm, years_fitting, based_on=list(range(2000, 2019)))
    dm_2w = linear_fitting(dm_2w, years_fitting, based_on=list(range(2000, 2015)))
    dm_pkm.append(dm_2w, "Variables")
    dm_pkm.sort("Variables")

    # 2022-2023: extrapolate from 2000-2019 pre-COVID trend
    dm_pkm = linear_fitting(dm_pkm, [2022, 2023], based_on=list(range(2000, 2019 + 1)))

    ####################
    ##### MAKE FTS #####
    ####################

    def make_fts(
        dm,
        variable,
        year_start,
        year_end,
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
            dm.array[idx[country], :, idx[variable], ...] = np.round(
                dm_temp.array[idx_temp[country], :, idx_temp[variable], ...], 0
            )
        if dim == "Categories1":
            dm.array[idx[country], :, :, idx[variable]] = np.round(
                dm_temp.array[idx_temp[country], :, :, idx_temp[variable]], 0
            )
        if dim == "Categories2":
            dm.array[idx[country], :, :, :, idx[variable]] = np.round(
                dm_temp.array[idx_temp[country], :, :, :, idx_temp[variable]], 0
            )
        if dim == "Categories3":
            dm.array[idx[country], :, :, :, :, idx[variable]] = np.round(
                dm_temp.array[idx_temp[country], :, :, :, :, idx_temp[variable]], 0
            )
        return dm

    # add fts years
    dm_pkm.add(np.nan, col_label=years_fts, dummy=True, dim="Years")

    baseyear_start = 2000
    baseyear_end = 2019

    # BAU: linear trend from 2000-2019 (2W flat at 2023 — declining EU trend clips to 0 otherwise)
    idx_pkm = dm_pkm.idx
    val_2w_2023 = dm_pkm.array[idx_pkm["EU27"], idx_pkm[2023], idx_pkm["2W"]]
    for y in years_fts:
        dm_pkm.array[idx_pkm["EU27"], idx_pkm[y], idx_pkm["2W"]] = val_2w_2023
    dm_pkm = make_fts(dm_pkm, "LDV", baseyear_start, baseyear_end, dim="Variables")
    dm_pkm = make_fts(dm_pkm, "bus", baseyear_start, baseyear_end, dim="Variables")
    dm_pkm = make_fts(
        dm_pkm, "metrotram", baseyear_start, baseyear_end, dim="Variables"
    )
    dm_pkm = make_fts(dm_pkm, "rail", baseyear_start, baseyear_end, dim="Variables")

    ####################################
    ##### MAKE AS FINAL DATAMATRIX #####
    ####################################

    # rename and deepen
    for v in dm_pkm.col_labels["Variables"]:
        dm_pkm.rename_col(v, "tra_passenger_modal-share_" + v, "Variables")
    dm_pkm.deepen()

    # get it in pkm
    dm_pkm.change_unit("tra_passenger_modal-share", 1e6, "mio pkm", "pkm")

    # normalise to modal share percentages
    dm_pkm_pc = dm_pkm.normalise("Categories1", inplace=False, keep_original=False)
    dm_pkm_pc.rename_col(
        "tra_passenger_modal-share_share", "tra_passenger_modal-share", "Variables"
    )

    # add modes not in JRC (set to 0 — very low compared to motorised modes)
    dm_pkm_pc.add(0, col_label=["bike", "walk"], dummy=True, dim="Categories1")
    dm_pkm_pc.sort("Categories1")

    # drop 1989 (only needed for OTS backfill)
    dm_pkm_pc.drop("Years", startyear)
    dm_pkm.drop("Years", startyear)

    ###############
    ##### OTS #####
    ###############

    dm_pkm_pc_ots = dm_pkm_pc.filter({"Years": years_ots})

    #######################
    ##### FTS LEVEL 1 #####
    #######################

    # level 1: continuing as is (BAU trend)
    dm_pkm_pc_fts_level1 = dm_pkm_pc.filter({"Years": years_fts})

    #######################
    ##### FTS LEVEL 4 #####
    #######################

    # source: page 95 https://www.itf-oecd.org/sites/default/files/docs/itf-transport-outlook-2023-launch.pdf
    dm_pkm_pc_level4 = dm_pkm_pc.copy()
    idx = dm_pkm_pc_level4.idx
    for y in range(2030, 2055, 5):
        dm_pkm_pc_level4.array[idx["EU27"], idx[y], :, :] = np.nan
    dm_pkm_pc_level4.array[idx["EU27"], idx[2050], :, idx["LDV"]] = 0.70
    dm_pkm_pc_level4.array[idx["EU27"], idx[2050], :, idx["bus"]] = 0.08
    dm_pkm_pc_level4.array[idx["EU27"], idx[2050], :, idx["metrotram"]] = 0.02
    dm_pkm_pc_level4.array[idx["EU27"], idx[2050], :, idx["rail"]] = 0.19
    dm_pkm_pc_level4.array[idx["EU27"], idx[2050], :, idx["2W"]] = 0.01
    dm_pkm_pc_level4.array[idx["EU27"], idx[2050], :, idx["bike"]] = 0
    dm_pkm_pc_level4.array[idx["EU27"], idx[2050], :, idx["walk"]] = 0
    dm_pkm_pc_level4 = linear_fitting(dm_pkm_pc_level4, years_fts)
    dm_pkm_pc_fts_level4 = dm_pkm_pc_level4.filter({"Years": years_fts})

    # interpolate levels 2 and 3 between level 1 and 4 at 2050
    variabs = dm_pkm_pc_fts_level1.col_labels["Categories1"]
    df_temp = pd.DataFrame(
        {
            "level": np.tile(range(1, 4 + 1), len(variabs)),
            "variable": np.repeat(variabs, 4),
        }
    )
    df_temp["value"] = np.nan
    df_temp = df_temp.pivot(
        index=["level"], columns=["variable"], values="value"
    ).reset_index()
    for v in variabs:
        idx = dm_pkm_pc_fts_level1.idx
        level1 = dm_pkm_pc_fts_level1.array[idx["EU27"], idx[2050], :, idx[v]][0]
        idx = dm_pkm_pc_fts_level4.idx
        level4 = dm_pkm_pc_fts_level4.array[idx["EU27"], idx[2050], :, idx[v]][0]
        arr = np.array([level1, np.nan, np.nan, level4])
        df_temp[v] = pd.Series(arr).interpolate().to_numpy()

    #######################
    ##### FTS LEVEL 2 #####
    #######################

    dm_pkm_pc_level2 = dm_pkm_pc.copy()
    idx = dm_pkm_pc_level2.idx
    for y in range(2030, 2055, 5):
        dm_pkm_pc_level2.array[idx["EU27"], idx[y], :, :] = np.nan
    for v in variabs:
        dm_pkm_pc_level2.array[idx["EU27"], idx[2050], :, idx[v]] = df_temp.loc[
            df_temp["level"] == 2, v
        ]
    dm_pkm_pc_level2 = linear_fitting(dm_pkm_pc_level2, years_fts)
    dm_pkm_pc_fts_level2 = dm_pkm_pc_level2.filter({"Years": years_fts})

    #######################
    ##### FTS LEVEL 3 #####
    #######################

    dm_pkm_pc_level3 = dm_pkm_pc.copy()
    idx = dm_pkm_pc_level3.idx
    for y in range(2030, 2055, 5):
        dm_pkm_pc_level3.array[idx["EU27"], idx[y], :, :] = np.nan
    for v in variabs:
        dm_pkm_pc_level3.array[idx["EU27"], idx[2050], :, idx[v]] = df_temp.loc[
            df_temp["level"] == 3, v
        ]
    dm_pkm_pc_level3 = linear_fitting(dm_pkm_pc_level3, years_fts)
    dm_pkm_pc_fts_level3 = dm_pkm_pc_level3.filter({"Years": years_fts})

    ################
    ##### SAVE #####
    ################

    DM_mod = {
        "ots": {"passenger_modal-share": []},
        "fts": {"passenger_modal-share": dict()},
    }
    DM_mod["ots"]["passenger_modal-share"] = dm_pkm_pc_ots.copy()
    DM_mod["fts"]["passenger_modal-share"][1] = dm_pkm_pc_fts_level1.copy()
    DM_mod["fts"]["passenger_modal-share"][2] = dm_pkm_pc_fts_level2.copy()
    DM_mod["fts"]["passenger_modal-share"][3] = dm_pkm_pc_fts_level3.copy()
    DM_mod["fts"]["passenger_modal-share"][4] = dm_pkm_pc_fts_level4.copy()

    # absolute pkm (BAU only — used by occupancy processor)
    dm_pkm.rename_col("tra_passenger_modal-share", "tra_passenger_pkm", "Variables")
    DM_pkm = {"ots": {"passenger_pkm": []}, "fts": {"passenger_pkm": dict()}}
    DM_pkm["ots"]["passenger_pkm"] = dm_pkm.filter({"Years": years_ots})
    for i in range(1, 4 + 1):
        DM_pkm["fts"]["passenger_pkm"][i] = dm_pkm.filter({"Years": years_fts})

    result = (DM_mod, DM_pkm)
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(result, f)
    return result


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_fts, years_ots

    run(years_ots, years_fts)
