import os
import pickle

from transition_compass_model.model.common.auxiliary_functions import (
    midpoint,
    my_pickle_dump,
    sort_pickle,
)


def dm_is_identic(list_dms):
    """Check if all elements in list_dms are equal."""
    return all(dm == list_dms[0] for dm in list_dms)


def run(DM_buildings):
    """Interpoalte missing"""

    #### Services area ####
    DM_fts = DM_buildings["fts"]
    list_services_dm = [DM_fts["services-floor-area"][i] for i in range(1, 3)]
    if dm_is_identic(list_services_dm):
        DM_buildings["fts"]["services-floor-area"][2] = midpoint(
            list_services_dm[0], DM_buildings["fts"]["services-floor-area"][4], 0.25
        )
        DM_buildings["fts"]["services-floor-area"][3] = midpoint(
            list_services_dm[0], DM_buildings["fts"]["services-floor-area"][4], 0.75
        )

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

    run(DM_buildings)
