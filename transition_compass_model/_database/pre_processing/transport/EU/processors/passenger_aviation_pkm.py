import os
import pickle
import warnings

import numpy as np

from transition_compass_model._database.pre_processing.routine_JRC import get_jrc_data
from transition_compass_model.model.common.auxiliary_functions import (
    jrc_iso2_dict,
    linear_fitting,
)

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(
    _THIS_DIR, "../data/datamatrix/passenger_aviation_pkm.pickle"
)


def run(years_ots: list, years_fts: list):
    """
    Build passenger aviation pkm lever for EU27 — territorial (scope 1).

    Accounting basis: territorial. JRC TrAvia_act captures pkm on flights *departing*
    from EU27 airports only — domestic, intra-EU, and extra-EU outbound legs. This is
    the territorial basis consistent with all other transport modes and with all other
    model modules (industry, buildings, agriculture, energy).

    No residency correction is applied. Extra-EU pkm is kept at ×1 (outbound only).

    To revert to residency basis: add the following line after loading dm_avi,
    before the groupby call:
        dm_avi.array[:, :, idx["international-extra"]] *= 2
    This doubles extra-EU pkm to capture return legs (depart non-EU airports),
    giving domestic + intra-EU + extra-EU×2 = full round-trip residency demand.
    The share-local-emissions FXA in the pickle then converts back to territorial
    for the emissions module at model runtime.

    Returns
    -------
    dm_avi : DataMatrix
        Territorial aviation pkm (variable=tra, cat1=aviation), OTS+FTS years combined.
        Intermediate passed to the occupancy processor.
    DM_avi : dict
        {"ots": {"passenger_aviation-pkm": dm}, "fts": {"passenger_aviation-pkm": {1..4: dm}}}
        Per-capita pkm/cap DataMatrix, for build_pickle.
    """
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    # load lifestyles pickle for population
    filepath = os.path.join(
        _THIS_DIR,
        "../../../lifestyles/Europe/data/lifestyles_allcountries.pickle",
    )
    with open(filepath, "rb") as handle:
        DM_lfe = pickle.load(handle)

    ####################
    ##### GET DATA #####
    ####################

    dict_iso2_jrc = jrc_iso2_dict()

    # load aviation pkm by sub-mode: domestic, intra-EU, extra-EU
    dict_extract = {
        "database": "Transport",
        "sheet": "TrAvia_act",
        "variable": "Passenger transport (mio pkm)",
        "sheet_last_row": "International - Extra-EEAwUK",
        "sub_variables": [
            "Domestic",
            "International - Intra-EEAwUK",
            "International - Extra-EEAwUK",
        ],
        "calc_names": ["domestic", "international-intra", "international-extra"],
    }
    dm_avi = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)

    # territorial basis: JRC gives outbound-only for extra-EU (departures from EU airports).
    # domestic and intra-EU are fully captured at ×1. extra-EU kept at ×1 (outbound only).
    # sum to total territorial pkm
    dm_avi.groupby(
        {"aviation": ["domestic", "international-intra", "international-extra"]},
        dim="Variables",
        aggregation="sum",
        regex=False,
        inplace=True,
    )
    dm_avi.change_unit("aviation", 1e6, "mio pkm", "pkm")

    ###################
    ##### FIX OTS #####
    ###################

    # add years not in JRC Transport file (1990-1999 and 2022-2023)
    dm_avi.add(
        np.nan,
        col_label=list(range(years_ots[0], 2000)) + [2022, 2023],
        dummy=True,
        dim="Years",
    )
    dm_avi.sort("Years")

    # fit 1990-1999 and 2022-2023 based on 2000-2019 trend
    years_fitting = list(range(years_ots[0], 2000)) + [2022, 2023]
    dm_avi = linear_fitting(dm_avi, years_fitting, based_on=list(range(2000, 2019 + 1)))

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

    # add fts years and extrapolate from 2000-2019 trend
    dm_avi.add(np.nan, col_label=years_fts, dummy=True, dim="Years")
    dm_avi = make_fts(dm_avi, "aviation", 2000, 2019, dim="Variables")

    ################
    ##### SAVE #####
    ################

    # sort
    dm_avi.sort("Country")
    dm_avi.sort("Years")
    dm_avi.sort("Variables")

    # make correct shape and name
    dm_avi.rename_col("aviation", "tra_aviation", "Variables")
    dm_avi.deepen()

    # per-capita normalisation
    dm_pop = DM_lfe["ots"]["pop"]["lfs_population_"].copy()
    dm_pop.append(DM_lfe["fts"]["pop"]["lfs_population_"][1], "Years")
    dm_pop.sort("Country")
    dm_pop.sort("Years")
    dm_pop = dm_pop.filter({"Country": dm_avi.col_labels["Country"]})
    dm_pop.sort("Country")
    dm_avi_cap = dm_avi.copy()
    dm_avi_cap.array = dm_avi_cap.array / dm_pop.array[..., np.newaxis]
    dm_avi_cap.units["tra"] = "pkm/cap"
    dm_avi_cap.rename_col("tra", "tra_pkm-cap", "Variables")

    # split between ots and fts (same trend for all 4 levels — BAU only)
    DM_avi = {
        "ots": {"passenger_aviation-pkm": []},
        "fts": {"passenger_aviation-pkm": dict()},
    }
    DM_avi["ots"]["passenger_aviation-pkm"] = dm_avi_cap.filter({"Years": years_ots})
    for i in range(1, 4 + 1):
        DM_avi["fts"]["passenger_aviation-pkm"][i] = dm_avi_cap.filter(
            {"Years": years_fts}
        )

    result = (dm_avi, DM_avi)
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(result, f)
    return result


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_fts, years_ots

    run(years_ots, years_fts)
