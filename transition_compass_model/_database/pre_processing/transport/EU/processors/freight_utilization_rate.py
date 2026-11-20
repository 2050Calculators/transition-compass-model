"""
EU27 freight OTS: load factor and utilisation rate for road modes (HDVH, HDVL, HDVM).

Faithful port of transport_lever_freight_utilization-rate.py. Two changes:
  1. HDVM/HDVH VKM split: old script split JRC LCV 60/40 into HDVM/HDVL — inconsistent
     with fxa_freight_tech and freight_modal_share which derive HDVM/HDVH from JRC HGV
     via Eurostat MPLW shares. Fixed here to use the same MPLW split.
  2. OTS only — FTS is generated in scenarios/transport_fts_EU.py.

Accounting basis
----------------
Load factor (tkm/vkm): JRC territorial tkm (from DM_tkm parameter) / JRC territorial VKM.
  Both numerator and denominator are territorial (all trucks on EU27 roads). For EU27 as a
  large bloc, territorial ≈ residency (non-EU transit trucks a small share of total).

Utilisation rate (vkm/vehicle/year): JRC TrRoad_act "Vehicle-km driven per vehicle annum".
  JRC computes this as total VKM / registered fleet → approximately EU-registered basis.
  HDVM = mean of HDVL and HDVH (JRC does not disaggregate HGV by sub-type).

Sources
-------
VKM total:    JRC-IDEES-2021 Transport, TrRoad_act, "Vehicle-km driven (mio km)"
VKM/vehicle:  JRC-IDEES-2021 Transport, TrRoad_act, "Vehicle-km driven per vehicle annum"
MPLW split:   Eurostat ROAD_GO_TA_MPLW (same as freight_modal_share and fxa_freight_tech)
tkm:          DM_tkm parameter from freight_modal_share.run()
"""

import os
import pickle
import warnings

import numpy as np

from transition_compass_model._database.pre_processing.routine_JRC import get_jrc_data
from transition_compass_model._database.pre_processing.transport.EU.processors.freight_modal_share import (
    _compute_mplw_shares,
)
from transition_compass_model.model.common.auxiliary_functions import (
    jrc_iso2_dict,
    linear_fitting,
)

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(
    _THIS_DIR, "../data/datamatrix/freight_utilization_rate.pickle"
)


def run(years_ots: list, years_fts: list, DM_tkm: dict):
    """
    Build freight utilisation-rate and load-factor DataMatrix for road modes (OTS).

    Parameters
    ----------
    years_ots, years_fts : list
    DM_tkm : dict
        Output of freight_modal_share.run() — {"ots": {"freight_tkm": DataMatrix}}.
        Used for load factor computation. Variable='tra_freight_tkm', Cat1 includes
        HDVH, HDVL, HDVM (in tkm).

    Returns
    -------
    dm_uti : DataMatrix
        Variables=['tra_freight_load-factor', 'tra_freight_utilisation-rate'],
        Categories1=['HDVH', 'HDVL', 'HDVM'], Years=years_ots.
    """
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    years_ots_ext = list(range(1989, years_ots[-1] + 1))
    startyear = years_ots_ext[0]

    dict_iso2_jrc = jrc_iso2_dict()

    # MPLW shares for consistent HDVM/HDVH split
    mplw_shares = _compute_mplw_shares(years_ots_ext)

    ###############################################################################
    # VKM (total, mio vkm)
    ###############################################################################

    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_act",
        "variable": "Vehicle-km driven (mio km)",
        "sheet_last_row": "Heavy goods vehicles",
        "sub_variables": ["Light commercial vehicles", "Heavy goods vehicles"],
        "calc_names": ["HDVL", "HDVH"],
    }
    dm_vkm = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)

    # Split JRC HGV into HDVH and HDVM via MPLW shares (consistent with
    # freight_modal_share and fxa_freight_tech).
    # Old script split JRC LCV 60/40 into HDVM/HDVL — that was inconsistent.
    years_in_dm = dm_vkm.col_labels["Years"]
    hdvm_sh_arr = np.array([mplw_shares[y][0] for y in years_in_dm])
    hdvh_sh_arr = np.array([mplw_shares[y][1] for y in years_in_dm])
    idx = dm_vkm.idx
    arr_hgv = dm_vkm.array[:, :, idx["HDVH"]].copy()
    dm_vkm.array[:, :, idx["HDVH"]] = arr_hgv * hdvh_sh_arr[np.newaxis, :]
    dm_vkm.add(arr_hgv * hdvm_sh_arr[np.newaxis, :], "Variables", "HDVM", "vehicles")
    dm_vkm.sort("Variables")
    dm_vkm.sort("Country")
    for v in dm_vkm.col_labels["Variables"]:
        dm_vkm.units[v] = "mio vkm"

    dm_vkm = linear_fitting(dm_vkm, years_ots_ext)

    # rename and deepen: Variables → 'vkm', Categories1 → ['HDVH','HDVL','HDVM']
    for v in dm_vkm.col_labels["Variables"]:
        dm_vkm.rename_col(v, "vkm_" + v, "Variables")
    dm_vkm.deepen()
    dm_vkm.change_unit("vkm", 1e6, "mio vkm", "vkm")
    dm_vkm.drop("Years", startyear)

    ###############################################################################
    # LOAD FACTOR (tkm / vkm)
    ###############################################################################

    dm_tkm = DM_tkm["ots"]["freight_tkm"].filter(
        {"Categories1": ["HDVH", "HDVL", "HDVM"]}
    )

    dm_uti = dm_tkm.copy()
    dm_uti.array = dm_uti.array / dm_vkm.array
    dm_uti.units["tra_freight_tkm"] = "tkm/vkm"
    dm_uti.rename_col("tra_freight_tkm", "tra_freight_load-factor", "Variables")

    ###############################################################################
    # UTILISATION RATE (vkm / vehicle / year)
    ###############################################################################

    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_act",
        "variable": "Vehicle-km driven per vehicle annum (km/vehicle)",
        "sheet_last_row": "Heavy goods vehicles",
        "sub_variables": ["Light commercial vehicles", "Heavy goods vehicles"],
        "calc_names": ["HDVL", "HDVH"],
    }
    dm_hdv = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)

    # HDVM = mean of HDVL and HDVH (JRC does not disaggregate HGV by sub-type)
    dm_temp = dm_hdv.groupby(
        {"HDVM": ["HDVL", "HDVH"]},
        dim="Variables",
        aggregation="mean",
        regex=False,
        inplace=False,
    )
    dm_hdv.append(dm_temp, "Variables")
    dm_hdv.sort("Variables")

    dm_hdv = linear_fitting(dm_hdv, years_ots_ext)

    # rename and deepen
    for v in dm_hdv.col_labels["Variables"]:
        dm_hdv.rename_col(v, "tra_freight_utilisation-rate_" + v, "Variables")
    dm_hdv.deepen()
    dm_hdv.units["tra_freight_utilisation-rate"] = "vkm/year"
    dm_hdv.drop("Years", startyear)

    ###############################################################################
    # COMBINE
    ###############################################################################

    dm_uti.append(dm_hdv, "Variables")

    # OTS only
    dm_uti = dm_uti.filter({"Years": years_ots})

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(dm_uti, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dm_uti


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_fts, years_ots
    from processors.freight_modal_share import run as modal_run

    _, DM_tkm, _ = modal_run(years_ots, years_fts)
    run(years_ots, years_fts, DM_tkm)
