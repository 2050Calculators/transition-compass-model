import os
import pickle
import warnings

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(_THIS_DIR, "../data/datamatrix/passenger_pkm.pickle")
_LIFESTYLES_PATH = os.path.join(
    _THIS_DIR, "../../../lifestyles/Europe/data/lifestyles_allcountries.pickle"
)


def run(DM_pkm, years_ots: list):
    """
    Build EU27 total ground passenger pkm per capita OTS.

    Sums all ground modes (2W, LDV, bus, metrotram, rail) from DM_pkm and divides
    by EU27 population from the lifestyles preprocessing pickle.

    Accounting basis
    ----------------
    DM_pkm is territorial (JRC TrRoad/TrRail). For EU27 as a large bloc, territorial ≈
    residency for ground transport — non-EU residents on EU roads/rails are a negligible
    share of total pkm. Aviation is excluded from this lever; it is handled separately
    by the passenger_aviation_pkm lever.

    Parameters
    ----------
    DM_pkm : dict
        Output of passenger_modal_share.run() — OTS + FTS ground pkm per mode.
    years_ots : list

    Returns
    -------
    dm_pkm_cap_ots : DataMatrix
        OTS total ground pkm per capita (Variable='tra_pkm-cap', unit=pkm/cap),
        EU27, 1990-2023.
    """
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    # load population
    with open(_LIFESTYLES_PATH, "rb") as f:
        DM_lfe = pickle.load(f)
    dm_pop = DM_lfe["ots"]["pop"]["lfs_population_"].copy()
    dm_pop = dm_pop.filter({"Country": ["EU27"]})
    dm_pop.sort("Years")

    # total ground pkm — sum all modes, EU27 only
    dm_pkmtot = DM_pkm["ots"]["passenger_pkm"].copy()
    dm_pkmtot = dm_pkmtot.filter({"Country": ["EU27"]})
    dm_pkmtot.sort("Years")
    dm_pkmtot.group_all("Categories1")

    # pkm / cap
    dm_pkmtot.array = dm_pkmtot.array / dm_pop.array
    dm_pkmtot.units["tra_passenger_pkm"] = "pkm/cap"
    dm_pkmtot.rename_col("tra_passenger_pkm", "tra_pkm-cap", "Variables")

    ###############
    ##### OTS #####
    ###############

    dm_pkm_cap_ots = dm_pkmtot.filter({"Years": years_ots})

    ################
    ##### SAVE #####
    ################

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(dm_pkm_cap_ots, f)
    return dm_pkm_cap_ots


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_fts, years_ots
    from processors.passenger_modal_share import run as modal_run

    _, DM_pkm = modal_run(years_ots, years_fts)
    run(DM_pkm, years_ots)
