"""
EU27 FXA: freight_mode_road (road modes: HDVH, HDVL, HDVM).

Faithful port of transport_fxa_freight_mode_road.py. The old script copied CH values
from the existing transport.pickle ("I do not know where these numbers are coming from").
This processor follows the CH freight_mode_road_fxa.py pattern instead:
  lifetime_vkm = lifetime_years × utilisation_rate(t)

Lifetime constants (years)
--------------------------
HDVH   16 yr  — heavy articulated trucks (HGV >3.5 t)
HDVM   16 yr  — medium rigid trucks (HGV >3.5 t)
HDVL   14 yr  — light commercial vans (LCV ≤3.5 t)

The TRACCS project and COPERT model report an operational lifetime of ~12 yr for HGVs,
which reflects the primary-market service life (time a truck is operated by its first owner
in a Western-EU fleet before being traded on). LCVs (HDVL) have a weaker secondary market:
low resale value and limited demand from second operators mean they are more often scrapped
after primary use, so their registration lifetime (~14 yr, as used in COPERT) is close to
their operational lifetime.

HGVs, by contrast, have an active intra-EU27 secondary market: trucks retired from Western-EU
primary fleets are sold to operators in Central/Eastern/Southern EU, where they continue to
operate — and remain registered in EU27 — for several additional years. The registration
(scrapping) lifetime therefore exceeds the primary-market operational lifetime. ACEA's
vehicles-in-use reports and IRU fleet analyses show average HGV fleet ages in EU27 consistent
with a registration lifetime of approximately 16 yr. We therefore use 16 yr for HDVH/HDVM to
represent the full registration lifetime, matching the methodological basis already used for
HDVL (14 yr = registration lifetime for LCVs).

Sources
-------
EU TRACCS project (operational lifetimes, EU-wide study):
  https://traccs.emisia.com/
COPERT fleet model (LCV/HGV default lifetimes):
  https://www.emisia.com/utilities/copert/
ACEA Report — Vehicles in use, Europe (fleet age and turnover data):
  https://www.acea.auto/vehicles-in-use/
IRU World Road Statistics / fleet analyses:
  https://www.iru.org/
Utilisation rate: freight_utilization_rate.run() output.
"""

import os
import pickle
import warnings

from transition_compass_model.model.common.data_matrix_class import DataMatrix

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(_THIS_DIR, "../data/datamatrix/fxa_freight_mode_road.pickle")

_LIFETIME_YEARS = {
    "HDVH": 16.0,
    "HDVM": 16.0,
    "HDVL": 14.0,
}

_MODES_ROAD = ["HDVH", "HDVL", "HDVM"]


def run(years_ots: list, years_fts: list, dm_utilization: DataMatrix):
    """
    Build freight_mode_road FXA DataMatrix.

    Parameters
    ----------
    years_ots, years_fts : list
    dm_utilization : DataMatrix
        Output of freight_utilization_rate.run(). Must contain
        'tra_freight_utilisation-rate' for HDVH, HDVL, HDVM.
        Country list is taken from dm_utilization.col_labels["Country"].

    Returns
    -------
    dm_fxa : DataMatrix
        Variable='tra_freight_lifetime', Categories1=['HDVH','HDVL','HDVM'],
        Years=years_ots+years_fts. EU27 + all individual EU countries.
    """
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    years_all = years_ots + years_fts
    last_ots = years_ots[-1]

    country_list = dm_utilization.col_labels["Country"]
    all_countries = ["EU27"] + [c for c in country_list if c != "EU27"]

    dm_fxa = DataMatrix(
        col_labels={
            "Country": all_countries,
            "Years": years_all,
            "Variables": ["tra_freight_lifetime"],
            "Categories1": _MODES_ROAD,
        },
        units={"tra_freight_lifetime": "vkm"},
    )
    idx = dm_fxa.idx
    idx_u = dm_utilization.idx
    ref = "EU27"

    # OTS: lifetime_vkm = lifetime_years × utilisation_rate(t)
    for mode in _MODES_ROAD:
        for yr in years_ots:
            ur = dm_utilization.array[
                idx_u[ref],
                idx_u[yr],
                idx_u["tra_freight_utilisation-rate"],
                idx_u[mode],
            ]
            dm_fxa.array[idx[ref], idx[yr], 0, idx[mode]] = _LIFETIME_YEARS[mode] * ur

    # FTS: hold last OTS value flat
    for mode in _MODES_ROAD:
        val = dm_fxa.array[idx[ref], idx[last_ots], 0, idx[mode]]
        for yr in years_fts:
            dm_fxa.array[idx[ref], idx[yr], 0, idx[mode]] = val

    # Copy EU27 to all individual countries
    for country in all_countries:
        if country != ref:
            dm_fxa.array[idx[country], :, :, :] = dm_fxa.array[idx[ref], :, :, :]

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(dm_fxa, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dm_fxa


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_fts, years_ots
    from processors.freight_modal_share import run as modal_run
    from processors.freight_utilization_rate import run as uti_run

    _, DM_tkm, _ = modal_run(years_ots, years_fts)
    dm_uti = uti_run(years_ots, years_fts, DM_tkm)
    run(years_ots, years_fts, dm_uti)
