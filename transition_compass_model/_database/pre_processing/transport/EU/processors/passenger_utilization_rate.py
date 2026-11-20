import os
import pickle
import warnings

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(
    _THIS_DIR, "../data/datamatrix/passenger_utilization_rate.pickle"
)


def run(dm_fleet, dm_vkm, years_ots: list):
    """
    Build passenger utilization-rate OTS for EU27.

    Utilization rate = vkm / fleet (vehicle-km per vehicle per year). OTS only
    (1990-2023). FTS scenarios are computed separately in scenarios/transport_fts_EU.py.

    Accounting basis
    ----------------
    Both vkm (from passenger_occupancy) and fleet (from fxa_passenger_tech) are
    territorial (JRC), so the ratio is internally consistent. No residency correction
    is needed — the rate is a pure vehicle property. For EU27 as a large bloc,
    territorial ≈ residency, same as the FXA proxy approach used throughout.

    Parameters
    ----------
    dm_fleet : DataMatrix
        Flat tech-level fleet from fxa_passenger_tech.run() — variables like
        "2W_ICE-gasoline", "LDV_BEV", etc., OTS+FTS years.
    dm_vkm : DataMatrix
        Vehicle-km OTS from passenger_occupancy.run() — Variable="tra_passenger_vkm",
        Cat1=modes, 1990-2023.

    Returns
    -------
    dm_uti_ots : DataMatrix
        OTS utilization rate (vkm/veh), 1990-2023.
    """
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    # get fleet data — deepen, collapse techs to mode totals, rename, deepen
    dm_fleet = dm_fleet.copy()
    dm_fleet.deepen()
    dm_fleet.group_all("Categories1")
    for v in dm_fleet.col_labels["Variables"]:
        dm_fleet.rename_col(v, "tra_fleet_" + v, "Variables")
    dm_fleet.deepen()

    # filter fleet to OTS years to match dm_vkm (which is OTS only)
    dm_fleet = dm_fleet.filter({"Years": dm_vkm.col_labels["Years"]})

    # obtain vkm/veh
    dm_uti = dm_vkm.copy()
    dm_uti.array = dm_uti.array / dm_fleet.array
    dm_uti.units["tra_passenger_vkm"] = "vkm/veh"
    dm_uti.rename_col(
        "tra_passenger_vkm", "tra_passenger_utilisation-rate", "Variables"
    )

    ###############
    ##### OTS #####
    ###############

    dm_uti_ots = dm_uti.filter({"Years": years_ots})

    ################
    ##### SAVE #####
    ################

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(dm_uti_ots, f)
    return dm_uti_ots


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_fts, years_ots
    from processors.fxa_passenger_tech import run as tech_run
    from processors.passenger_modal_share import run as modal_run
    from processors.passenger_occupancy import run as occ_run

    dm_tech, dm_fleet, dm_new, dm_renrate = tech_run(years_ots, years_fts)
    _, DM_pkm = modal_run(years_ots, years_fts)
    dm_occ_ots, dm_vkm_ots, dm_pkm_avi_terr_ots = occ_run(DM_pkm, years_ots)
    run(dm_fleet, dm_vkm_ots, years_ots)
