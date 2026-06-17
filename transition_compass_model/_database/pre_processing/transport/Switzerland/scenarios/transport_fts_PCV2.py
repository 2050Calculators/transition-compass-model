import os

import numpy as np
import scenarios.helpers_PCV2 as wkf

from transition_compass_model.model.common.auxiliary_functions import (
    my_pickle_dump,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def normalise_non_fixed_values(
    dm_modal_share_3, fixed_cat, years_start, variable_name="tra_passenger_modal-share"
):
    idx_fts = dm_modal_share_3.idx
    cat_labels = dm_modal_share_3.col_labels["Categories1"]
    other_cats = [c for c in cat_labels if c not in fixed_cat]
    other_idxs = [idx_fts[c] for c in other_cats]
    fixed_idx = [idx_fts[c] for c in fixed_cat]
    country_i = idx_fts["Vaud"]
    mode_i = idx_fts[variable_name]

    fixed_val = dm_modal_share_3.array[
        country_i, years_start, mode_i, fixed_idx
    ].astype(float)

    # compute sum of other categories (ignore NaNs)
    others = dm_modal_share_3.array[country_i, years_start, mode_i, other_idxs].astype(
        float
    )
    sum_others = np.nansum(others)
    remaining = 1.0 - fixed_val.sum()

    scale = remaining / sum_others
    dm_modal_share_3.array[country_i, years_start, mode_i, other_idxs] = others * scale
    return dm_modal_share_3


def run(DM_transport: DataMatrix, country_list, years_ots, years_fts):
    # TODO : see why only 2 and 3 levers
    # MO- : favoriser les bus électriques
    dm_new_tech_share_3 = DM_transport["fts"]["passenger_technology-share_new"][2]
    dm_new_tech_share_3 = wkf.compute_tech_share_for_buses(dm_new_tech_share_3)
    # dm_new

    DM_transport["fts"]["passenger_technology-share_new"][
        2
    ].array = dm_new_tech_share_3.array
    #

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
    ] = 0.451 * share_without_aviation
    dm_freight_modal_share_3 = normalise_non_fixed_values(
        dm_freight_modal_share_3,
        ["rail"],
        idx_freight[2050],
        variable_name="tra_freight_modal-share",
    )

    dm_freight_modal_share_3.fill_nans("Years")
    dm_freight_modal_share_3.normalise(dim="Categories1", inplace=True)
    DM_transport["fts"]["freight_modal-share"][3] = dm_freight_modal_share_3

    ##### FREIGHT TRANSPORT #########
    this_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(this_dir, "../../../../data/datamatrix/transport.pickle")
    my_pickle_dump(DM_new=DM_transport, local_pickle_file=pickle_file)

    #### Demande de transport de stat vaud #####

    return DM_transport
