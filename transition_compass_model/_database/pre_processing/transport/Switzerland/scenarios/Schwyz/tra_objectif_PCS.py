import os

from transition_compass_model.model.common.auxiliary_functions import (
    my_pickle_dump,
    sort_pickle,
)


def electric_transport_schwyz(DM_transport):
    # DM_transport["fts"][lever_name][2]
    return DM_transport


def run(DM_transport: dict, lev=2):
    """Implment Schwyz objectif from Ivo table.

    Args:
        DM_transport (dict)
    Returns:
        dict: DM_transport updated
    """

    ## Implement the modal share for Vaud
    DM_transport = electric_transport_schwyz(DM_transport)

    ##### Save pickle #########
    this_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(
        this_dir, "../../../../../data/datamatrix/transport.pickle"
    )
    my_pickle_dump(DM_new=DM_transport, local_pickle_file=pickle_file)
    sort_pickle(pickle_file)

    return DM_transport
