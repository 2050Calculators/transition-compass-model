import os
import pickle

import numpy as np

from transition_compass_model.model.common.auxiliary_functions import (
    my_pickle_dump,
    sort_pickle,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def heating_temp_fts_dls(DM_buildings, lev):
    """Remove the rebound effect for lever 4, and make all enveloppe to 19°C. DLS confort temperature"""

    dm_heat_cool_fts = DM_buildings["fts"]["heatcool-behaviour"][lev].copy()
    idx = dm_heat_cool_fts.idx
    dm_heat_cool_fts[:, idx[2030] :, idx["bld_Tint-heating"], :, :] = 19
    DM_buildings["fts"]["heatcool-behaviour"][lev] = dm_heat_cool_fts
    return DM_buildings


def floor_area_fts_dls(DM_buildings, lev):
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
    DM_buildings["fts"]["floor-intensity"][lev] = dm_fts_floor_intensity
    return DM_buildings


def services_area_fts_dls(DM_buildings: DataMatrix, lev: int) -> DataMatrix:
    """No new surfaces are constructed for services it stays constant."""

    dm_srv_floor = DM_buildings["ots"]["services-floor-area"].copy()
    years_fts = DM_buildings["fts"]["services-floor-area"][lev].col_labels["Years"]
    dm_srv_floor.add(np.nan, dim="Years", col_label=years_fts, dummy=True)
    dm_srv_floor.fill_nans("Years")

    DM_buildings["fts"]["services-floor-area"][lev] = dm_srv_floor.filter(
        {"Years": years_fts}
    )

    return DM_buildings


def run(DM_buildings, lev=4):
    #### FLOOR AREA ####
    DM_buildings = floor_area_fts_dls(DM_buildings, lev)

    #### SERVICES AREA ####
    DM_buildings = services_area_fts_dls(DM_buildings, lev)

    #### Heating temperature ####
    DM_buildings = heating_temp_fts_dls(DM_buildings, lev)

    this_dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(this_dir, "../../../../data/datamatrix/buildings.pickle")

    my_pickle_dump(DM_buildings, file)
    sort_pickle(file)

    return DM_buildings


if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))

    file = os.path.join(this_dir, "../../../../data/datamatrix/buildings.pickle")
    with open(file, "rb") as handle:
        DM_buildings = pickle.load(handle)

    run(DM_buildings, lev=4)
