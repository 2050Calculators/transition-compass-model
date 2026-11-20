"""
EU27 FXA: maximum BEV and H2 aircraft seats deployable by 2050.

Territorial aviation seat-km (from occupancy OTS) is projected to 2050 via
BAU linear trend, then converted to max aircraft seats via technology-specific
utilisation rates.

Sources
-------
BEV max ~5% of EU seat-km in 2050:
  https://te-cdn.ams3.cdn.digitaloceanspaces.com/files/TE-aviation-decarbonisation-roadmap-FINAL.pdf
H2 max ~35-40% of EU seat-km in 2050:
  https://te-cdn.ams3.cdn.digitaloceanspaces.com/files/Study-Analysing-the-costs-of-hydrogen-aircraft.pdf

Note: territorial skm is appropriate here — this is a physical fleet capacity
constraint (aircraft deployable at EU airports), not a demand accounting choice.
"""

import os
import pickle
import warnings

import numpy as np

from transition_compass_model.model.common.auxiliary_functions import linear_fitting
from transition_compass_model.model.common.data_matrix_class import DataMatrix

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(_THIS_DIR, "../data/datamatrix/fxa_vehicles-max.pickle")


def run(years_ots: list, years_fts: list, dm_vkm_ots) -> DataMatrix:
    """
    Build vehicles-max FXA for EU27 aviation (BEV and H2 max seats).

    Parameters
    ----------
    years_ots, years_fts : list
    dm_vkm_ots : DataMatrix
        OTS vehicle-km from passenger_occupancy.run().
        Variable='tra_passenger_vkm', Cat1=modes (including 'aviation'), unit=vkm.

    Returns
    -------
    dm_max : DataMatrix
        Variable='tra_vehicles-max', Cat1=['aviation'], Cat2=['BEV', 'H2'].
        OTS years = 0; FTS 2025 = 0; 2030-2040 = 0; 2045 = interpolated; 2050 = computed.
    """
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    # load seat km (aviation, territorial)
    dm_skm = dm_vkm_ots.filter({"Country": ["EU27"], "Categories1": ["aviation"]})

    # project to 2050 via BAU linear trend (2000-2019 window, same as scenarios)
    dm_skm.add(np.nan, col_label=years_fts, dummy=True, dim="Years")
    dm_skm = linear_fitting(
        dm_skm,
        years_ots=years_fts,
        based_on=list(range(2000, 2020)),
    )

    # get max number of seats — create from scratch (zeros)
    dm_max = DataMatrix(
        col_labels={
            "Country": ["EU27"],
            "Years": years_ots + years_fts,
            "Variables": ["tra_vehicles-max"],
            "Categories1": ["aviation"],
            "Categories2": ["BEV", "H2"],
        },
        units={"tra_vehicles-max": "seat"},
    )
    dm_max.array[:] = 0.0

    # get 2050 levels
    skm_perseat_peryear_bev = 0.7 * 1e6
    skm_perseat_peryear_h2 = 2.5 * 1e6
    dm_max[:, 2050, :, :, "BEV"] = (
        dm_skm[:, 2050, :, :] * 0.05 / skm_perseat_peryear_bev
    )
    dm_max[:, 2050, :, :, "H2"] = dm_skm[:, 2050, :, :] * 0.40 / skm_perseat_peryear_h2

    # put zeroes in non 2050
    for y in list(range(2030, 2045 + 5, 5)):
        dm_max[:, y, ...] = 0

    # put nan in 2045
    dm_max[:, 2045, ...] = np.nan

    # linear fit
    dm_max = linear_fitting(dm_max, list(range(2030, 2050 + 5, 5)))

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(dm_max, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dm_max


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_fts, years_ots
    from processors.passenger_modal_share import run as modal_run
    from processors.passenger_occupancy import run as occ_run

    _, DM_pkm = modal_run(years_ots, years_fts)
    _, dm_vkm_ots, _ = occ_run(DM_pkm, years_ots)
    run(years_ots, years_fts, dm_vkm_ots)
