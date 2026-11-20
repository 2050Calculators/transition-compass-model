"""
EU27 FXA: maximum SAF (sustainable aviation fuel) available to aviation by 2050.

Uses dm_share_local as a structural template (Country/Years shape) then fills
EU27 with the Abrantes et al. (2021) SAF supply ceiling, linearly interpolated
from 2030 (0) to 2050 (max). efuel availability is set to 0 (no global constraint
modelled for e-fuels at this stage).

Sources
-------
SAF max 2050: Abrantes et al. (2021), "Sustainable aviation fuels and imminent
  technologies - CO2 emissions evolution towards 2050", J. Clean. Prod.,
  https://doi.org/10.1016/j.jclepro.2021.127937
  Max 200 Mt SAF world-wide in 2050 @ 43 MJ/kg; EU27 share = 13% of global
  aviation CO2 (based on EU27+UK ≈ 15-17% global; UK ~2-4% removed).

No residency correction needed — this is a physical supply constraint on
EU-available SAF, not a demand accounting choice.
"""

import os
import pickle
import warnings

import numpy as np

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(
    _THIS_DIR, "../data/datamatrix/fxa_fuel-mix-availability.pickle"
)


def run(years_ots: list, years_fts: list, dm_share_local):
    """
    Build fuel-mix-availability FXA for EU27 aviation.

    Parameters
    ----------
    years_ots, years_fts : list
    dm_share_local : DataMatrix
        From fxa_aviation_share_local_emissions.run(). Used as a structural
        template (Country/Years); actual values are not used.

    Returns
    -------
    dm_tmp : DataMatrix
        Variable='tra_passenger_available-fuel-mix',
        Cat1=['biofuel','efuel'], Cat2=['aviation'].
        SAF availability ramps 0 → 1.118e12 MJ from 2030 to 2050.
        efuel = 0 throughout.
    """
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    dm_fxa = dm_share_local

    # According to ATAG, Waypoint 2050 (2021) (plot page 47)
    # https://aviationbenefits.org/media/167417/w2050_v2021_27sept_full.pdf
    # There will be 600-140 = 460 Mtoe of SAF available world-wide.
    # 1 toe = 41868 MJ
    # Of these, we allocate 0.6% to Switzerland. As EU27+UK accounted for roughly
    # 15–17% of global CO₂ emissions from aviation, for EU27 we put 13%
    val_SAF_2050_ATAG_MJ = 450 * 1e6 * 41868 * 0.13  # noqa: F841  (kept for reference)
    # Alternatively, the paper Abrantes et al. (2021)
    # "Sustainable aviation fuels and imminent technologies - CO2 emissions
    # evolution towards 2050", https://doi.org/10.1016/j.jclepro.2021.127937
    # Gives as max value for SAF in 2050 200 Mt. To convert to energy we use 43 MJ/kg
    val_SAF_2050_paper_MJ = 200 * 1e9 * 43 * 0.13

    dm_SAF = dm_fxa.flatten()
    dm_SAF.add(
        np.nan,
        dim="Variables",
        col_label="tra_passenger-max-SAF_aviation",
        unit="MJ",
        dummy=True,
    )
    dm_SAF = dm_SAF.filter({"Variables": ["tra_passenger-max-SAF_aviation"]})
    dm_SAF.deepen()
    dm_SAF["EU27", 2050, "tra_passenger-max-SAF", "aviation"] = val_SAF_2050_paper_MJ
    dm_SAF["EU27", 2025, "tra_passenger-max-SAF", "aviation"] = 0
    dm_SAF["EU27", 2030, "tra_passenger-max-SAF", "aviation"] = 0
    dm_SAF.fill_nans("Years")

    # Extract maximum fuel available
    dm_tmp = dm_SAF.filter({"Variables": ["tra_passenger-max-SAF"]})
    dm_tmp.rename_col(
        "tra_passenger-max-SAF",
        "tra_passenger_available-fuel-mix_biofuel",
        dim="Variables",
    )
    dm_tmp.deepen(based_on="Variables")
    dm_tmp.add(0, dim="Categories2", col_label="efuel", dummy=True)
    dm_tmp.switch_categories_order()

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(dm_tmp, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dm_tmp


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_fts, years_ots
    from processors.fxa_aviation_share_local_emissions import run as share_local_run

    dm_share_local = share_local_run(years_ots, years_fts)
    run(years_ots, years_fts, dm_share_local)
