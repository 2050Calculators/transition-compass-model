import os

import numpy as np
from scenarios.transport_fts_PCV1 import (
    cat_dict,
)

from transition_compass_model.model.common.auxiliary_functions import (
    linear_fitting,
    my_pickle_dump,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def modal_share_DLS_fts(DM_transport, lev):
    """DLS sce"""

    # Values extracted from the SWICE working paper on DLS mobility
    DLS_2050_val_dict = {
        "TIM": 587 / 3241,
        "TP": 1143 / 3241,
        "MA": (3241 - 587 - 1143) / 3241,
    }

    dm_modal_share_ots = DM_transport["ots"]["passenger_modal-share"]
    idx_ots = dm_modal_share_ots.idx

    dm_modal_share_lever = DM_transport["fts"]["passenger_modal-share"][lev].copy()

    idx = dm_modal_share_lever.idx
    dm_modal_share_lever.array[:, idx[2025] + 1 :, :, :] = (
        np.nan
    )  # clear level except for 2025

    dm_modal_share_4 = None
    for key, cat in cat_dict.items():
        # Get the respective proportions for the modal share in 2050 for each category
        dm_modal_ots_cat = dm_modal_share_ots.filter({"Categories1": cat})
        dm_modal_ots_cat.normalise(dim="Categories1", inplace=True, keep_original=False)

        dm_modal_fts_cat_lever = dm_modal_share_lever.filter({"Categories1": cat})
        dm_modal_fts_cat_lever.array[:, idx[2050], ...] = (
            DLS_2050_val_dict[key] * dm_modal_ots_cat.array[:, idx_ots[2023], ...]
        )

        # If first iteration of loop create the new datamatrix, else append the new category to the existing datamatrix
        if dm_modal_share_4 is None:
            dm_modal_share_4 = dm_modal_fts_cat_lever.copy()
        else:
            dm_modal_share_4.append(dm_modal_fts_cat_lever, dim="Categories1")

    dm_modal_share_4.sort("Categories1")

    linear_fitting(dm_modal_share_4, dm_modal_share_4.col_labels["Years"])
    dm_modal_share_4.normalise(dim="Categories1", inplace=True)
    DM_transport["fts"]["passenger_modal-share"][lev] = dm_modal_share_4

    return DM_transport


def freight_tkm_dls(DM_transport):
    # Réduction de la dmeande de transport
    tkm_ots = DM_transport["ots"]["freight_tkm"].copy()
    dm_tkm_3 = DM_transport["fts"]["freight_tkm"][4].copy()
    idx_tkm = dm_tkm_3.idx
    idx_tkm_ots = tkm_ots.idx
    dm_tkm_3.array[idx_tkm["Vaud"], 1:-1, :] = np.nan

    # Réduction de 16% de la demande de bien et denrées (41% du transport de marchandise)
    dm_tkm_3.array[idx_tkm["Vaud"], -1, :] = (
        tkm_ots.array[idx_tkm_ots["Vaud"], idx_tkm_ots[2023], :] * 0.41 * (1 - 0.16)
        + tkm_ots.array[idx_tkm_ots["Vaud"], idx_tkm_ots[2023], :] * 0.6
    )

    dm_tkm_3.fill_nans("Years")
    DM_transport["fts"]["freight_tkm"][4] = dm_tkm_3
    return DM_transport


def vehicle_utilisation_DLS_fts(DM_transport):
    """Dummy increase of 10% utilization rate of all vehicles"""

    dm_utilization_fts = DM_transport["fts"]["passenger_utilization-rate"][4].copy()
    idx_util = dm_utilization_fts.idx

    dm_utilization_fts.array[:, idx_util[2050], ...] = (
        dm_utilization_fts.array[:, idx_util[2050], ...] * 1.10
    )
    dm_utilization_fts.array[:, 1 : idx_util[2050], ...] = np.nan
    dm_utilization_fts.fill_nans("Years")
    DM_transport["fts"]["passenger_utilization-rate"][4] = dm_utilization_fts
    return DM_transport


def run(DM_transport: DataMatrix, lev: int = 4) -> DataMatrix:
    #### Freight tkm ####
    DM_transport = freight_tkm_dls(DM_transport)

    #### Utilization rate ####
    DM_transport = vehicle_utilisation_DLS_fts(DM_transport)

    #### MODAL SHARE ####
    DM_transport = modal_share_DLS_fts(DM_transport, lev)

    ##### FREIGHT TRANSPORT #########
    this_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(this_dir, "../../../../data/datamatrix/transport.pickle")
    my_pickle_dump(DM_new=DM_transport, local_pickle_file=pickle_file)

    return DM_transport
