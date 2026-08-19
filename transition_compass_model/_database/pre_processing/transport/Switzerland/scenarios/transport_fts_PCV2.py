import os

import numpy as np

from transition_compass_model.model.common.auxiliary_functions import (
    my_pickle_dump,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def run(DM_transport: DataMatrix, country_list, years_ots, years_fts):
    # TODO : see why only 2 and 3 levers
    # MO- : favoriser les bus électriques

    dm_freight_ots = DM_transport["ots"]["freight_modal-share"]
    idx_ots = dm_freight_ots.idx

    dm_freight_modal_share_3 = DM_transport["fts"]["freight_modal-share"][3]
    idx_freight = dm_freight_modal_share_3.idx

    share_without_aviation = (
        1
        - dm_freight_modal_share_3.array[
            idx_freight["Vaud"], idx_freight[2050], :, idx_freight["aviation"]
        ]
    )
    dm_freight_modal_share_3.array[idx_freight["Vaud"], 1:-1, :, :] = np.nan

    dm_freight_modal_share_3.array[
        idx_freight["Vaud"], idx_freight[2050], :, idx_freight["rail"]
    ] = (
        dm_freight_ots.array[idx_ots["Vaud"], idx_ots[2023], :, idx_ots["rail"]]
        * 1.45
        * share_without_aviation
    )
    dm_freight_modal_share_3.normalise_non_fixed_values(
        ["rail"],
        idx_freight[2050],
        variable_name="tra_freight_modal-share",
    )

    dm_freight_modal_share_3.fill_nans("Years")
    dm_freight_modal_share_3.normalise(dim="Categories1", inplace=True)
    DM_transport["fts"]["freight_modal-share"][4] = dm_freight_modal_share_3

    # Réduction de la dmeande de transport
    tkm_ots = DM_transport["ots"]["freight_tkm"].copy()
    dm_tkm_3 = DM_transport["fts"]["freight_tkm"][4].copy()
    idx_tkm = dm_tkm_3.idx
    idx_tkm_ots = tkm_ots.idx
    dm_tkm_3.array[idx_tkm["Vaud"], 1:-1, :] = np.nan
    dm_tkm_3.array[idx_tkm["Vaud"], -1, :] = (
        tkm_ots.array[idx_tkm_ots["Vaud"], idx_tkm_ots[2023], :] * 0.41 * (1 - 0.16)
        + tkm_ots.array[idx_tkm_ots["Vaud"], idx_tkm_ots[2023], :] * 0.6
    )

    dm_tkm_3.fill_nans("Years")
    DM_transport["fts"]["freight_tkm"][4] = dm_tkm_3
    ##### FREIGHT TRANSPORT #########
    this_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(this_dir, "../../../../data/datamatrix/transport.pickle")
    my_pickle_dump(DM_new=DM_transport, local_pickle_file=pickle_file)

    #### Demande de transport de stat vaud #####

    return DM_transport
