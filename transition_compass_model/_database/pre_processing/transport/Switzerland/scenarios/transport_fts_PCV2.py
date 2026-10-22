import os

import numpy as np

from transition_compass_model.model.common.auxiliary_functions import (
    my_pickle_dump,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def compute_tkm_per_mode(dm_tkm, dm_mode):
    dm = dm_mode.copy()
    tmp = dm_tkm.array[:, :, 0, np.newaxis] * 1e9 * dm_mode.array[:, :, 0, :]
    dm.add(tmp, dim="Variables", col_label="tra_freight_transport-demand", unit="tkm")

    return dm


def increase_freight_rail_by_45_percent(DM_transport: DataMatrix) -> DataMatrix:
    """Update modal share fts to lead to increase in rail freight of 45%

    Args:
        DM_transport (DataMatrix)

    Returns:
        DataMatrix: DM_transport updated
    """

    dm_tkm_fts = DM_transport["fts"]["freight_tkm"][1].copy()
    idx_tkm_fts = dm_tkm_fts.idx
    dm_tkm_ots = DM_transport["ots"]["freight_tkm"].copy()
    idx_tkm_ots = dm_tkm_ots.idx

    # Project mention an increase of 45% of the use of train
    # (tkm_2050*mode2050)/(tkm_base*mode_base)=1.45
    # => (tkm_base/tkm_2050)*1.45 = mode2050/mode_base
    # Applied to every canton (Switzerland keeps its level-2 values).
    base_year = dm_tkm_ots.col_labels["Years"][-1]
    dm_freight_modal_ots = DM_transport["ots"]["freight_modal-share"].copy()
    idx_ots = dm_freight_modal_ots.idx
    dm_freight_modal_share_2 = DM_transport["fts"]["freight_modal-share"][2].copy()
    idx_freight = dm_freight_modal_share_2.idx
    cantons = [
        c for c in dm_freight_modal_share_2.col_labels["Country"] if c != "Switzerland"
    ]

    for canton in cantons:
        ratio = (
            dm_tkm_ots.array[idx_tkm_ots[canton], idx_tkm_ots[base_year], 0]
            / dm_tkm_fts.array[idx_tkm_fts[canton], idx_tkm_fts[2050], 0]
            * 1.45
        )
        dm_freight_modal_share_2.array[idx_freight[canton], 1:-1, :, :] = np.nan
        dm_freight_modal_share_2.array[
            idx_freight[canton], idx_freight[2050], :, idx_freight["rail"]
        ] = (
            dm_freight_modal_ots.array[
                idx_ots[canton], idx_ots[base_year], :, idx_ots["rail"]
            ]
            * ratio
        )
        dm_freight_modal_share_2.normalise_non_fixed_values(
            ["rail"],
            idx_freight[2050],
            variable_name="tra_freight_modal-share",
            country_name=canton,
        )

    dm_freight_modal_share_2.fill_nans("Years")
    dm_freight_modal_share_2.normalise(dim="Categories1", inplace=True)

    dm_mode_fts = compute_tkm_per_mode(dm_tkm_fts, dm_freight_modal_share_2)
    idx_mode_fts = dm_mode_fts.idx
    dm_mode_ots = compute_tkm_per_mode(dm_tkm_ots, dm_freight_modal_ots)
    idx_mode_ots = dm_mode_ots.idx

    var = "tra_freight_transport-demand"
    for canton in cantons:
        ratio_to_obtain = (
            dm_mode_fts.array[
                idx_mode_fts[canton],
                idx_mode_fts[2050],
                idx_mode_fts[var],
                idx_mode_fts["rail"],
            ]
            / dm_mode_ots.array[
                idx_mode_ots[canton],
                idx_mode_ots[base_year],
                idx_mode_ots[var],
                idx_mode_ots["rail"],
            ]
        )
        np.testing.assert_allclose(ratio_to_obtain, 1.45, rtol=1e-2, atol=0)
    DM_transport["fts"]["freight_modal-share"][2] = dm_freight_modal_share_2
    return DM_transport


def run(DM_transport: DataMatrix, country_list, years_ots, years_fts):
    #### Modal share ####
    DM_transport = increase_freight_rail_by_45_percent(DM_transport)

    this_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(this_dir, "../../../../data/datamatrix/transport.pickle")
    my_pickle_dump(DM_new=DM_transport, local_pickle_file=pickle_file)

    #### Demande de transport de stat vaud #####

    return DM_transport
