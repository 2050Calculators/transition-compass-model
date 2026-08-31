import os

import numpy as np

from transition_compass_model.model.common.auxiliary_functions import (
    my_pickle_dump,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix


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


def run(DM_transport: DataMatrix, country_list, years_ots, years_fts):
    #### Freight tkm ####
    DM_transport = freight_tkm_dls(DM_transport)

    #### Utilization rate ####
    DM_transport = vehicle_utilisation_DLS_fts(DM_transport)

    ##### FREIGHT TRANSPORT #########
    this_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(this_dir, "../../../../data/datamatrix/transport.pickle")
    my_pickle_dump(DM_new=DM_transport, local_pickle_file=pickle_file)

    return DM_transport
