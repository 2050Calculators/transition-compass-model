import os
import pickle
import warnings

import numpy as np

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(
    _THIS_DIR, "../data/datamatrix/passenger_tech_share_new.pickle"
)


def run(dm_new, years_ots: list):
    """
    Build passenger technology-share_new OTS for EU27.

    Technology share = fraction of new vehicle registrations per mode by technology.
    OTS only (1990-2023). FTS scenarios (Green Deal targets) are computed separately
    in scenarios/transport_fts_EU.py.

    Accounting basis
    ----------------
    Technology share is a fleet composition metric — independent of residency vs
    territorial accounting. New vehicle registrations are EU27-registered vehicles
    in both cases. No residency correction needed.

    Parameters
    ----------
    dm_new : DataMatrix
        Flat tech-level new vehicles from fxa_passenger_tech.run() — variables like
        "2W_ICE-gasoline", "LDV_BEV", etc., OTS+FTS years.

    Returns
    -------
    dm_new_ots : DataMatrix
        OTS tech share of new vehicles (Variable="tra_passenger_technology-share_new",
        Cat1=modes, Cat2=techs), 1990-2023.
    """
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    dm_new = dm_new.copy()

    # filter to OTS years before normalising (FTS years in dm_new are not needed here)
    dm_new = dm_new.filter({"Years": years_ots})

    # normalise
    for v in dm_new.col_labels["Variables"]:
        dm_new.rename_col(v, "tra_passenger_technology-share_new_" + v, "Variables")
    dm_new.deepen_twice()
    dm_new.sort("Categories2")
    dm_new.normalise("Categories2")

    # add H2 and BEV as 0 for aviation
    dm_new.add(np.nan, "Categories2", "H2", "%", True)
    dm_new.sort("Categories2")
    dm_new[:, :, :, "aviation", "H2"] = 0
    dm_new[:, :, :, "aviation", "BEV"] = 0

    ###############
    ##### OTS #####
    ###############

    dm_new_ots = dm_new.copy()

    ################
    ##### SAVE #####
    ################

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(dm_new_ots, f)
    return dm_new_ots


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_fts, years_ots
    from processors.fxa_passenger_tech import run as tech_run

    _, dm_fleet, dm_new, dm_renrate = tech_run(years_ots, years_fts)
    run(dm_new, years_ots)
