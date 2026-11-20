"""
EU27 buildings FTS scenarios.

Currently all 4 lever levels are identical (baseline linear extrapolation
from OTS). Lever-differentiated scenarios will be added in a future update.

Parameters
----------
DM_buildings_ots : dict
    Output of processors.build_pickle.run_ots() with keys "ots", "fxa", "constant".
years_fts : list[int]

Returns
-------
DM_fts : dict
    {lever_name: {1..4: DataMatrix (FTS 2025-2050)}}
"""

import os
import pickle
import warnings

import numpy as np

from transition_compass_model.model.common.auxiliary_functions import linear_fitting

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(_THIS_DIR, "../data/datamatrix/buildings_fts_EU.pickle")


#################################################################
########################## MAIN RUN ############################
#################################################################


def run(DM_buildings_ots, years_fts):
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    ots = DM_buildings_ots["ots"]

    DM_fts = {
        "floor-intensity": {},
        "heatcool-behaviour": {},
        "building-renovation-rate": {
            "bld_building-mix": {},
            "bld_renovation-rate": {},
            "bld_renovation-redistribution": {},
            "bld_demolition-rate": {},
        },
        "services-floor-area": {},
        "heating-technology-fuel": {
            "bld_heating-technology": {},
            "bld_hot-water-technology": {},
        },
        "heating-efficiency": {},
    }

    ###############################################################
    # floor-intensity
    ###############################################################
    dm = ots["floor-intensity"].copy()
    linear_fitting(dm, years_fts)
    dm_fts = dm.filter({"Years": years_fts})
    DM_fts["floor-intensity"] = {lev: dm_fts for lev in range(1, 5)}

    ###############################################################
    # heatcool-behaviour
    ###############################################################
    dm = ots["heatcool-behaviour"].copy()
    dm.add(np.nan, "Years", years_fts, dummy=True)
    dm.fill_nans("Years")
    dm_fts = dm.filter({"Years": years_fts})
    DM_fts["heatcool-behaviour"] = {lev: dm_fts for lev in range(1, 5)}

    ###############################################################
    # building-renovation-rate: bld_building-mix
    ###############################################################
    dm = ots["building-renovation-rate"]["bld_building-mix"].copy()
    dm.add(np.nan, "Years", years_fts, dummy=True)
    dm.fill_nans("Years")
    dm_fts = dm.filter({"Years": years_fts})
    DM_fts["building-renovation-rate"]["bld_building-mix"] = {
        lev: dm_fts for lev in range(1, 5)
    }

    ###############################################################
    # building-renovation-rate: bld_renovation-rate
    ###############################################################
    dm = ots["building-renovation-rate"]["bld_renovation-rate"].copy()
    dm.add(np.nan, "Years", years_fts, dummy=True)
    dm.fill_nans("Years")
    dm_fts = dm.filter({"Years": years_fts})
    DM_fts["building-renovation-rate"]["bld_renovation-rate"] = {
        lev: dm_fts for lev in range(1, 5)
    }

    ###############################################################
    # building-renovation-rate: bld_renovation-redistribution
    ###############################################################
    dm = ots["building-renovation-rate"]["bld_renovation-redistribution"].copy()
    dm.add(np.nan, "Years", years_fts, dummy=True)
    dm.fill_nans("Years")
    dm_fts = dm.filter({"Years": years_fts})
    DM_fts["building-renovation-rate"]["bld_renovation-redistribution"] = {
        lev: dm_fts for lev in range(1, 5)
    }

    ###############################################################
    # building-renovation-rate: bld_demolition-rate (mean of 2012-2023)
    ###############################################################
    dm = ots["building-renovation-rate"]["bld_demolition-rate"].copy()
    idx = dm.idx
    years_mean = [yr for yr in range(2012, 2024) if yr in idx]
    idx_yrs = [idx[yr] for yr in years_mean]
    val_mean = np.mean(dm.array[:, idx_yrs, ...], axis=1)
    dm.add(np.nan, "Years", years_fts, dummy=True)
    for yr in years_fts:
        dm.array[:, idx[yr], ...] = val_mean
    dm_fts = dm.filter({"Years": years_fts})
    DM_fts["building-renovation-rate"]["bld_demolition-rate"] = {
        lev: dm_fts for lev in range(1, 5)
    }

    ###############################################################
    # services-floor-area
    ###############################################################
    dm = ots["services-floor-area"].copy()
    linear_fitting(dm, years_fts, based_on=list(range(2012, 2024)))
    dm_fts = dm.filter({"Years": years_fts})
    DM_fts["services-floor-area"] = {lev: dm_fts for lev in range(1, 5)}

    ###############################################################
    # heating-technology-fuel: bld_heating-technology
    ###############################################################
    dm = ots["heating-technology-fuel"]["bld_heating-technology"].copy()
    dm.add(np.nan, "Years", years_fts, dummy=True)
    dm.fill_nans("Years")
    dm_fts = dm.filter({"Years": years_fts})
    DM_fts["heating-technology-fuel"]["bld_heating-technology"] = {
        lev: dm_fts for lev in range(1, 5)
    }

    ###############################################################
    # heating-technology-fuel: bld_hot-water-technology
    ###############################################################
    dm = ots["heating-technology-fuel"]["bld_hot-water-technology"].copy()
    dm.add(np.nan, "Years", years_fts, dummy=True)
    dm.fill_nans("Years")
    dm_fts = dm.filter({"Years": years_fts})
    DM_fts["heating-technology-fuel"]["bld_hot-water-technology"] = {
        lev: dm_fts.copy() for lev in range(1, 5)
    }

    ###############################################################
    # heating-efficiency (OTS: Cat1=energy_class, Cat2=tech)
    ###############################################################
    dm = ots["heating-efficiency"].copy()
    dm.switch_categories_order()  # Cat1=tech, Cat2=energy_class
    dm_hp = dm.filter({"Categories1": ["heat-pump"]})
    dm.drop("Categories1", "heat-pump")
    linear_fitting(dm, years_fts, based_on=list(range(2015, 2023)))
    dm.array = np.minimum(dm.array, 0.98)
    linear_fitting(dm_hp, years_fts, based_on=list(range(2015, 2023)))
    dm.append(dm_hp, "Categories1")
    dm_fts = dm.filter({"Years": years_fts})
    dm_fts.switch_categories_order()  # back to Cat1=energy_class, Cat2=tech
    dm_fts.sort("Categories2")
    DM_fts["heating-efficiency"] = {lev: dm_fts.copy() for lev in range(1, 5)}

    ###############################################################
    # SAVE
    ###############################################################
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(DM_fts, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved cache → {_CACHE_PATH}")
    return DM_fts


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    import pickle as _pkl

    from params import years_fts

    _ots_path = os.path.join(_THIS_DIR, "../data/datamatrix/build_ots_EU.pickle")
    with open(_ots_path, "rb") as f:
        DM_buildings_ots = _pkl.load(f)
    DM_fts = run(DM_buildings_ots, years_fts)
    for k, v in DM_fts.items():
        if isinstance(v, dict) and 1 in v:
            print(f"  {k}: {list(v[1].col_labels.keys())}")
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, dict) and 1 in v2:
                    print(f"  {k}/{k2}: {list(v2[1].col_labels.keys())}")
