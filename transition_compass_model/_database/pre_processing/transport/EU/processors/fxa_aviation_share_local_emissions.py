"""
EU27 FXA: share of aviation emissions that are territorial (local).

STATUS: Kept in pickle for backward compatibility. No longer applied by the
transport module once aviation preprocessing switches to territorial basis
(passenger_aviation_pkm.py now uses territorial pkm directly, extra-EU at ×1).
Can be removed from build_pickle.py and this file deleted when the module-side
share_local machinery (compute_aviation_emission_variants, tra_emissions_interface
share_np scaling) is also removed.

--- Original purpose (residency-based pipeline) ---
Converts residency-based model emissions to territorial GHG inventory emissions.
The transport module applies: territorial_emissions = model_emissions × share_local.

Formula: share_local = (domestic + intra-EU + extra-EU) / (domestic + intra-EU + extra-EU×2)
  → territorial / residency seat-km ratio (~0.65 EU27 1990)
  → correct because residency demand = domestic + intra-EU + extra-EU×2 (round trips)
  → extra-EU return legs depart non-EU airports → not in EU inventory → ×0.5

Sources
-------
vkm:   JRC-IDEES-2021 Transport, TrAvia_act, "Vehicle-km (mio km)"
seats: JRC-IDEES-2021 Transport, TrAvia_png, "Seats available per flight"
"""

import os
import pickle
import warnings

from transition_compass_model._database.pre_processing.routine_JRC import get_jrc_data
from transition_compass_model.model.common.auxiliary_functions import (
    jrc_iso2_dict,
    linear_fitting,
)

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(
    _THIS_DIR, "../data/datamatrix/fxa_share-local-emissions.pickle"
)


def run(years_ots: list, years_fts: list):
    """
    Build share-local-emissions FXA for EU27 aviation.

    Parameters
    ----------
    years_ots, years_fts : list

    Returns
    -------
    dm_avi : DataMatrix
        Variable='tra_share-emissions-local', Cat1='aviation', unit='%'.
        OTS + FTS years. Values ≈ 0.65 EU27 (territorial / residency seat-km ratio).
    """
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    dict_iso2_jrc = jrc_iso2_dict()

    # get vkm by route type
    dict_extract = {
        "database": "Transport",
        "sheet": "TrAvia_act",
        "variable": "Vehicle-km (mio km)",
        "sheet_last_row": "International - Extra-EEAwUK",
        "sub_variables": [
            "Domestic",
            "International - Intra-EEAwUK",
            "International - Extra-EEAwUK",
        ],
        "calc_names": [
            "vkm_domestic",
            "vkm_international-int",
            "vkm_international-extra",
        ],
    }
    dm_vkm_avi = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_vkm_avi.deepen()

    # get seats per flight by route type
    dict_extract = {
        "database": "Transport",
        "sheet": "TrAvia_png",
        "variable": "Seats available per flight",
        "sheet_last_row": "International - Extra-EEAwUK",
        "sub_variables": [
            "Domestic",
            "International - Intra-EEAwUK",
            "International - Extra-EEAwUK",
        ],
        "calc_names": [
            "seats_domestic",
            "seats_international-int",
            "seats_international-extra",
        ],
    }
    dm_seats_avi = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_seats_avi.deepen()
    dm_seats_avi.units["seats"] = "number"

    # compute seat-km per route type
    dm_avi = dm_vkm_avi.copy()
    dm_avi.append(dm_seats_avi, "Variables")
    dm_avi.operation("vkm", "*", "seats", "Variables", "skm", "vkm")

    # --- residency correction ---
    # Old: share_local = (dom + intra) / (dom + intra + extra)   [territorial fraction]
    # New: share_local = (dom + intra + extra) / (dom + intra + extra×2) [territorial / residency]
    # Extra-EU return legs depart non-EU airports → not in EU inventory → doubled in residency demand
    dm_avi.filter({"Variables": ["skm"]}, inplace=True)

    # double extra-EU for residency denominator
    dm_extra = dm_avi.filter({"Categories1": ["international-extra"]}).copy()
    dm_extra.array *= 2
    dm_extra.rename_col("international-extra", "international-extra-x2", "Categories1")
    dm_avi.append(dm_extra, "Categories1")

    # numerator = territorial total = dom + intra + extra (×1)
    dm_avi.append(
        dm_avi.groupby(
            {"numerator": ["domestic", "international-int", "international-extra"]},
            "Categories1",
        ),
        "Categories1",
    )
    # denominator = residency total = dom + intra + extra×2
    dm_avi.append(
        dm_avi.groupby(
            {
                "denominator": [
                    "domestic",
                    "international-int",
                    "international-extra-x2",
                ]
            },
            "Categories1",
        ),
        "Categories1",
    )
    dm_avi.operation(
        "numerator", "/", "denominator", "Categories1", "tra_share-emissions-local", "%"
    )
    dm_avi = dm_avi.filter({"Categories1": ["tra_share-emissions-local"]})
    dm_avi = dm_avi.flatten()
    dm_avi.rename_col(
        "skm_tra_share-emissions-local",
        "tra_share-emissions-local_aviation",
        "Variables",
    )
    dm_avi.deepen()
    dm_avi.units["tra_share-emissions-local"] = "%"

    # OTS: fill missing years via linear fit
    years_fitting = list(range(1990, 2023 + 1))
    dm_avi = linear_fitting(dm_avi, years_fitting)

    # FTS: project forward via linear fit
    years_fitting = list(range(2025, 2050 + 5, 5))
    dm_avi = linear_fitting(dm_avi, years_fitting)

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(dm_avi, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dm_avi


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_fts, years_ots

    run(years_ots, years_fts)
