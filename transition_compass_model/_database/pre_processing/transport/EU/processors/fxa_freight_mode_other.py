"""
EU27 freight mode-other FXA: tkm-per-vehicle and renewal rates for non-road modes.

Produces a single DataMatrix (dm_mode_oth) combining:
  - tra_freight_tkm-by-veh : tkm / fleet for IWW, aviation, marine, rail (OTS)
  - tra_freight_renewal-rate : renewal rates for IWW, aviation, marine, rail (OTS)

Accounting basis
----------------
All inputs are territorial (JRC Navigation / Aviation / Rail data), used as proxy
for residency — same approach as other FXA processors for EU27.

Renewal rates
-------------
Literature-based constants (1 / average operational fleet lifetime). Flat across
all years and countries. See _RENEWAL_RATE_EU27 below for sources.

Sources
-------
tkm:    freight_modal_share.run() -> DM_tkm["ots"]["freight_tkm"] (IWW, aviation, marine, rail)
fleet:  fxa_freight_tech.run() -> dm_fleet_final (tra_freight_technology-share_fleet)
renrate: _RENEWAL_RATE_EU27 constants (literature-derived, see below)
"""

import os
import pickle
import warnings

import numpy as np

from transition_compass_model.model.common.auxiliary_functions import linear_fitting
from transition_compass_model.model.common.data_matrix_class import DataMatrix

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(
    _THIS_DIR, "../data/datamatrix/fxa_freight_mode_other.pickle"
)

_MODES_OTHER = ["IWW", "aviation", "marine", "rail"]

# Fleet renewal rates for EU27 non-road freight modes (fraction/year = 1 / lifetime).
# Flat constants applied to all countries and years.
#
# Rail (freight wagons): average intermodal wagon lifetime ~35 years.
#   UIC 2024 Report on Combined Transport in Europe, p. 44 (wagon age profile).
#   https://uic.org/IMG/pdf/uic_uirr_report_2024-2.pdf
#   -> renewal rate = 1/35 ≈ 0.029
#
# IWW (barges): Rhine/Danube cargo barges typically 50-60 years operational life;
#   average Rhine cargo vessel age >40 years, dry cargo fleet >85% built pre-2000.
#   CCNR 2024 Annual Report on Inland Navigation in Europe, Chapter 6 (cargo fleets).
#   https://inland-navigation-market.org/wp-content/uploads/2024/10/CCNR_annual_report_EN_2024_WEB.pdf
#   -> renewal rate = 1/50 = 0.020
#
# Aviation (freighters): average cargo aircraft retirement age 32.5 years globally;
#   EU cargo fleet average age ~22 years in 2022.
#   IATA, "The older cargo fleet hampers emissions reductions" (2022).
#   https://www.iata.org/en/iata-repository/publications/economic-reports/the-older-cargo-fleet-hampers-emissions-reductions/
#   -> renewal rate = 1/30 ≈ 0.033 (slight rounding below retirement age; EU fleet newer)
#
# Marine (international ships): average scrapping age 25-30 years (varies with market);
#   planned design life typically 25-30 years.
#   UNCTAD Review of Maritime Transport 2023, Chapter 2 (fleet and port developments).
#   https://unctad.org/system/files/official-document/rmt2023_en.pdf
#   -> renewal rate = 1/25 = 0.040
_RENEWAL_RATE_EU27 = {
    "IWW": 0.020,
    "aviation": 0.033,
    "marine": 0.040,
    "rail": 0.029,
}


def run(years_ots: list, years_fts: list, DM_tkm: dict, dm_fleet_final):
    """
    Build freight mode-other FXA for EU27 (OTS only).

    Parameters
    ----------
    years_ots, years_fts : list
    DM_tkm : dict
        From freight_modal_share.run(): {"ots": {"freight_tkm": DataMatrix}}.
        Variable=tra_freight_tkm, Cat1=modes, unit=tkm.
    dm_fleet_final : DataMatrix
        From fxa_freight_tech.run(): absolute fleet by mode×tech.
        Variable=tra_freight_technology-share_fleet, Cat1=modes, Cat2=techs, unit=number.

    Returns
    -------
    dm_mode_oth : DataMatrix
        Variables=['tra_freight_tkm-by-veh', 'tra_freight_renewal-rate'],
        Cat1=['IWW', 'aviation', 'marine', 'rail']. OTS only.
    """
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    ###############
    ##### TKM #####
    ###############

    # get total tkm (OTS only — FTS appended in scenarios/transport_fts_EU.py)
    dm_tkm = DM_tkm["ots"]["freight_tkm"].copy()

    # get fleet (filter to OTS years so it matches dm_tkm for the operation)
    dm_fleet = dm_fleet_final.filter({"Years": years_ots})
    dm_fleet_agg = dm_fleet.group_all("Categories2", inplace=False)

    # get tkm by vehicle
    dm_tkm.append(dm_fleet_agg, "Variables")
    dm_tkm.operation(
        "tra_freight_tkm",
        "/",
        "tra_freight_technology-share_fleet",
        out_col="tra_freight_tkm-by-veh",
        unit="tkm",
    )
    dm_tkm.drop("Variables", ["tra_freight_tkm", "tra_freight_technology-share_fleet"])

    # filter
    dm_tkm.filter({"Categories1": ["IWW", "aviation", "marine", "rail"]}, inplace=True)

    # Extend tkm-by-veh to FTS years via linear fitting (BAU trend continuation)
    dm_tkm.add(np.nan, col_label=years_fts, dummy=True, dim="Years")
    dm_tkm = linear_fitting(dm_tkm, years_fts, min_t0=0, min_tb=0)

    # TODO: even though this is tkm/num, i leave it tkm as this is how the pickle is
    # currently built (to be changed at some point)

    ########################
    ##### RENEWAL RATE #####
    ########################

    # Build renewal rate DM directly from literature constants (flat for all countries/years).
    years_all = years_ots + years_fts
    countries = dm_tkm.col_labels["Country"]
    dm_renrate_freight = DataMatrix(
        col_labels={
            "Country": countries,
            "Years": years_all,
            "Variables": ["tra_freight_renewal-rate"],
            "Categories1": _MODES_OTHER,
        },
        units={"tra_freight_renewal-rate": "%"},
    )
    idx = dm_renrate_freight.idx
    for mode in _MODES_OTHER:
        dm_renrate_freight.array[:, :, idx["tra_freight_renewal-rate"], idx[mode]] = (
            _RENEWAL_RATE_EU27[mode]
        )

    ########################
    ##### PUT TOGETHER #####
    ########################

    dm_mode_oth = dm_tkm.copy()
    dm_mode_oth.append(dm_renrate_freight, "Variables")
    dm_mode_oth.sort("Variables")

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(dm_mode_oth, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dm_mode_oth


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_fts, years_ots
    from processors.freight_modal_share import run as modal_share_run
    from processors.fxa_freight_tech import run as freight_tech_run

    DM_mod, DM_tkm, DM_tkmtot = modal_share_run(years_ots, years_fts)
    dm_final, dm_fleet_final = freight_tech_run(years_ots, years_fts)
    run(years_ots, years_fts, DM_tkm, dm_fleet_final)
