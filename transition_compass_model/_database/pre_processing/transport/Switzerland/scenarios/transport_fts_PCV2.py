import os

import scenarios.helpers_PCV2 as wkf

from transition_compass_model.model.common.auxiliary_functions import (
    my_pickle_dump,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def run(DM_transport: DataMatrix, country_list, years_ots, years_fts):
    # MO- : favoriser les bus électriques
    dm_new_tech_share_3 = DM_transport["fts"]["passenger_technology-share_new"][2]
    dm_new_tech_share_3 = wkf.compute_tech_share_for_buses(dm_new_tech_share_3)
    # dm_new
    DM_transport["fts"]["passenger_technology-share_new"][
        2
    ].array = dm_new_tech_share_3.array
    #

    # EXPORTS FINAUX
    this_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(this_dir, "../../../../data/datamatrix/transport.pickle")
    # with open(pickle_file, "rb") as handle:
    #     DM_transport_old = pickle.load(handle)

    my_pickle_dump(DM_new=DM_transport, local_pickle_file=pickle_file)

    return DM_transport
