import os
import pickle

import numpy as np

from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    my_pickle_dump,
    sort_pickle,
)


def heating_temp(DM_buildings):
    """Remove the rebound effect for lever 4, and make all enveloppe to 19°C. DLS confort temperature"""

    dm_heat_cool_fts = DM_buildings["fts"]["heatcool-behaviour"][4].copy()
    idx = dm_heat_cool_fts.idx
    dm_heat_cool_fts[:, idx[2030] :, idx["bld_Tint-heating"], :, :] = 19
    DM_buildings["fts"]["heatcool-behaviour"][4] = dm_heat_cool_fts
    return DM_buildings


def floor_area(DM_buildings):
    """decent floor area per capita is estimated at 30 m2 par habitation et 10
    m2 more par person.
    Average number of person per dwelling in switzerland is 2.2. 19, 1m2 /cap in 2024.
    Decent living area thus estimated to 19,1 m2 /cap"""

    # TODO : update with the services update.
    dm_fts_floor_intensity = DM_buildings["fts"]["floor-intensity"][1].copy()
    idx = dm_fts_floor_intensity.idx

    dm_fts_floor_intensity.array[
        idx["Vaud"], 1:, idx["lfs_floor-intensity_space-cap"]
    ] = np.nan
    dm_fts_floor_intensity.array[
        idx["Vaud"], idx[2050], idx["lfs_floor-intensity_space-cap"]
    ] = (
        dm_fts_floor_intensity.array[
            idx["Vaud"], idx[2025], idx["lfs_floor-intensity_space-cap"]
        ]
        - 23.9
    )

    dm_fts_floor_intensity.fill_nans("Years")
    DM_buildings["fts"]["floor-intensity"][4] = dm_fts_floor_intensity
    return DM_buildings


def run(DM_buildings, lev=4):
    ###### FLOOR AREA #####
    DM_buildings = floor_area(DM_buildings)

    ### Heating temperature ###
    DM_buildings = heating_temp(DM_buildings)

    this_dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(this_dir, "../../../../data/datamatrix/buildings.pickle")

    my_pickle_dump(DM_buildings, file)
    sort_pickle(file)

    return DM_buildings


if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    # !FIXME: use the actual values and not the calibration factor
    file = os.path.join(this_dir, "../../../../data/datamatrix/buildings.pickle")
    with open(file, "rb") as handle:
        DM_buildings = pickle.load(handle)

    years_ots = create_years_list(1990, 2023, 1)
    years_fts = create_years_list(2025, 2050, 5)

    run(DM_buildings, years_ots, years_fts)
