import os
import pickle
import warnings

from transition_compass_model.model.common.data_matrix_class import DataMatrix

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(
    _THIS_DIR, "../data/datamatrix/fxa_passenger_vehicle_lifetime.pickle"
)


def run(dm_tech, years_ots: list, years_fts: list):
    """
    Build passenger vehicle-lifetime FXA for EU27 and all individual EU countries.

    All countries receive identical lifetime values — the same constants used for
    Switzerland in the CH preprocessing. The old EU script (transport_fxa_passenger_vehicle-
    lifetime.py) copied these values from the CH transport pickle; here they are reproduced
    directly from the same source constants.

    Lifetime assumptions (from CH passenger_lifetime_pipeline):
    - LDV ICE / PHEV: 14 years (ICE-gasoline: 15)
    - LDV BEV / FCEV: 5.5 years at start of FTS, increases to 14 by 2035
    - 2W: 8 years
    - Rail: 30 years
    - Metrotram: 20 years
    - Bus: 10 years
    - Aviation: NaN (not modelled via fleet turnover)

    Parameters
    ----------
    dm_tech : DataMatrix
        From fxa_passenger_tech.run() — used to read mode/tech/country labels.
    years_ots, years_fts : list

    Returns
    -------
    dm_lifetime : DataMatrix
        Variable='tra_passenger_lifetime', Cat1=modes, Cat2=techs, unit=years.
        OTS values are NaN (only FTS is used by the model).
    """
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    mode_cat = dm_tech.col_labels["Categories1"]
    tech_cat = dm_tech.col_labels["Categories2"]
    country_list = dm_tech.col_labels["Country"]

    dm_lifetime = DataMatrix(
        col_labels={
            "Country": country_list,
            "Years": years_ots + years_fts,
            "Variables": ["tra_passenger_lifetime"],
            "Categories1": mode_cat,
            "Categories2": tech_cat,
        },
        units={"tra_passenger_lifetime": "years"},
    )
    dm_lifetime.sort("Categories1")
    dm_lifetime.sort("Categories2")

    # LDV: ICE and PHEV — 14 years; ICE-gasoline — 15 years
    for cat in dm_lifetime.col_labels["Categories2"]:
        if ("ICE" in cat) or ("PHEV" in cat):
            dm_lifetime[:, :, "tra_passenger_lifetime", "LDV", cat] = 14
    dm_lifetime[:, :, "tra_passenger_lifetime", "LDV", "ICE-gasoline"] = 15

    # LDV: BEV and FCEV — starts at 5.5, reaches 14 by 2035
    for cat in dm_lifetime.col_labels["Categories2"]:
        if ("BEV" in cat) or ("FCEV" in cat):
            dm_lifetime[:, years_fts[0], "tra_passenger_lifetime", "LDV", cat] = 5.5
            dm_lifetime[:, 2035, "tra_passenger_lifetime", "LDV", cat] = 14
            dm_lifetime[:, years_fts[-1], "tra_passenger_lifetime", "LDV", cat] = 14

    # 2W: 8 years for all techs
    dm_lifetime[:, :, "tra_passenger_lifetime", "2W", :] = 8

    # Rail: 30 years
    dm_lifetime[:, :, "tra_passenger_lifetime", "rail", "CEV"] = 30
    dm_lifetime[:, :, "tra_passenger_lifetime", "rail", "ICE-diesel"] = 30

    # Metrotram: 20 years
    dm_lifetime[:, :, "tra_passenger_lifetime", "metrotram", "mt"] = 20

    # Bus: 10 years
    dm_lifetime[:, :, "tra_passenger_lifetime", "bus", "CEV"] = 10
    dm_lifetime[:, :, "tra_passenger_lifetime", "bus", "ICE-diesel"] = 10
    dm_lifetime[:, :, "tra_passenger_lifetime", "bus", "ICE-gas"] = 10
    dm_lifetime[:, :, "tra_passenger_lifetime", "bus", "ICE-gasoline"] = 10
    dm_lifetime[:, :, "tra_passenger_lifetime", "bus", "BEV"] = 10

    dm_lifetime.fill_nans("Years")

    ################
    ##### SAVE #####
    ################

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(dm_lifetime, f)
    return dm_lifetime


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_fts, years_ots
    from processors.fxa_passenger_tech import run as tech_run

    dm_tech, _, _, _ = tech_run(years_ots, years_fts)
    run(dm_tech, years_ots, years_fts)
