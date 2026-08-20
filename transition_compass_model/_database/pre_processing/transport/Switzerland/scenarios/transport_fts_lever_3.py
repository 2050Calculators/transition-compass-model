import os

import numpy as np

from transition_compass_model.model.common.auxiliary_functions import (
    my_pickle_dump,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def comput_prop_in_cat(dm_ots, list_cat):
    share_road_TP = dm_ots.filter({"Categories1": list_cat})
    share_road_TP.normalise(dim="Categories1", inplace=True, keep_original=False)
    idx_road_TP = share_road_TP.idx
    return share_road_TP, idx_road_TP


def run(DM_transport: dict, country_list: list, years_ots: list, years_fts: list):
    def get_lev_data(
        DM_transport: DataMatrix, lev_name="passenger_modal-share"
    ) -> tuple[DataMatrix, list, DataMatrix, list]:
        dm_ots = DM_transport["ots"][lev_name]
        idx_ots = dm_ots.idx

        dm_fts_3 = DM_transport["fts"][lev_name][3]
        idx_fts = dm_fts_3.idx
        return dm_ots, idx_ots, dm_fts_3, idx_fts

    #### MODAL SHARE ####
    dm_ots_modal, idx_ots_modal, dm_fts_3_modal, idx_fts = get_lev_data(
        DM_transport, lev_name="passenger_modal-share"
    )

    dm_fts_3_modal

    # Get the proportions of metrotram and bus in 2023 to keep the same proportion in 2050
    share_road_TP, idx_road_TP = comput_prop_in_cat(dm_ots_modal, ["metrotram", "bus"])
    # Same for TIM
    share_road_TIM, idx_road_TIM = comput_prop_in_cat(dm_ots_modal, ["LDV", "2W"])

    # Replace all value by nan for overwriting with interpolated values later
    dm_fts_3_modal.array[idx_fts["Vaud"], 1:-1, :, :] = np.nan

    # INput values for 2050
    values_2050 = {
        "rail": 0.32,
        "metrotram": 0.09
        * share_road_TP.array[
            idx_road_TP["Vaud"], idx_road_TP[2023], :, idx_road_TP["metrotram"]
        ],
        "bus": 0.09
        * share_road_TP.array[
            idx_road_TP["Vaud"], idx_road_TP[2023], :, idx_road_TP["bus"]
        ],
        "walk": 0.11,
        "bike": 0.09,
    }

    TIM_ratio = 1 - sum(values_2050.values())

    values_2050["LDV"] = (
        TIM_ratio
        * share_road_TIM.array[
            idx_road_TIM["Vaud"], idx_road_TIM[2023], :, idx_road_TIM["LDV"]
        ]
    )
    values_2050["2W"] = (
        TIM_ratio
        * share_road_TIM.array[
            idx_road_TIM["Vaud"], idx_road_TIM[2023], :, idx_road_TIM["2W"]
        ]
    )

    for key, value in values_2050.items():
        dm_fts_3_modal.array[
            idx_fts["Vaud"],
            idx_fts[2050],
            idx_fts["tra_passenger_modal-share"],
            idx_fts[key],
        ] = value

    # dm_freight_modal_share_3.array[idx_freight["Vaud"], 1:-1, :, :] = np.nan

    # dm_freight_modal_share_3.array[
    #     idx_freight["Vaud"], idx_freight[2050], :, idx_freight["rail"]
    # ] = (
    #     dm_freight_ots.array[idx_ots["Vaud"], idx_ots[2023], :, idx_ots["rail"]]
    #     * 1.45
    #     * share_without_aviation
    # )
    #
    dm_fts_3_modal.fill_nans("Years")
    dm_fts_3_modal.normalise(dim="Categories1", inplace=True)
    DM_transport["fts"]["passenger_modal-share"][3] = dm_fts_3_modal

    ##### Save pickle #########
    this_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(this_dir, "../../../../data/datamatrix/transport.pickle")
    my_pickle_dump(DM_new=DM_transport, local_pickle_file=pickle_file)

    #### Demande de transport de stat vaud #####

    return DM_transport
