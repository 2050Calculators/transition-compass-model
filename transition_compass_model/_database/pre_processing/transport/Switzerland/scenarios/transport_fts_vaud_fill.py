import os

import numpy as np

from transition_compass_model.model.common.auxiliary_functions import (
    midpoint,
    my_pickle_dump,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def comput_prop_in_cat(dm_ots, list_cat):
    share_road_TP = dm_ots.filter({"Categories1": list_cat})
    share_road_TP.normalise(dim="Categories1", inplace=True, keep_original=False)
    idx_road_TP = share_road_TP.idx
    return share_road_TP, idx_road_TP


def get_lev_data(
    DM_transport, lev_name="passenger_modal-share", lev_number=3
) -> tuple[DataMatrix, list, DataMatrix, list]:
    dm_ots = DM_transport["ots"][lev_name].copy()
    idx_ots = dm_ots.idx

    dm_fts_3 = DM_transport["fts"][lev_name][lev_number].copy()
    idx_fts = dm_fts_3.idx
    return dm_ots, idx_ots, dm_fts_3, idx_fts


def implement_modal_share_fts(DM_transport):
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

    dm_fts_3_modal.fill_nans("Years")
    dm_fts_3_modal.normalise(dim="Categories1", inplace=True)
    DM_transport["fts"]["passenger_modal-share"][3] = dm_fts_3_modal
    return DM_transport


def tech_share_fts(DM_transport):
    dm_tech_ots, idx_ots_tech, dm_tech_fts, idx_fts_tech = get_lev_data(
        DM_transport, lev_name="passenger_technology-share_new"
    )

    # Objective of 0 diesel bus in 2050
    dm_tech_fts.array[idx_fts_tech["Vaud"], 1:-1, :, idx_fts_tech["bus"], :] = np.nan
    dm_tech_fts.array[
        idx_fts_tech["Vaud"],
        idx_fts_tech[2050],
        0,
        idx_fts_tech["bus"],
        [idx_fts_tech["CEV"], idx_fts_tech["ICE-diesel"]],
    ] = [1, 0]
    dm_tech_fts.fill_nans("Years")

    # Objectif initial de la comission européenne de 0 voiture diesel en 2035
    diesel_cat = ["ICE-diesel", "ICE-gasoline", "ICE-gas"]
    idx_diesel = []
    for i in diesel_cat:
        idx_diesel += [idx_fts_tech[i]]
    dm_tech_fts.array[idx_fts_tech["Vaud"], 1:, :, idx_fts_tech["LDV"], idx_diesel] = (
        np.nan
    )
    dm_tech_fts.array[
        idx_fts_tech["Vaud"], idx_fts_tech[2035] :, 0, idx_fts_tech["LDV"], idx_diesel
    ] = 0
    dm_tech_fts.fill_nans("Years")
    dm_tech_fts.normalise(dim="Categories2", inplace=True)

    DM_transport["fts"]["passenger_technology-share_new"][3] = dm_tech_fts
    return DM_transport


def occupancy_fts(DM_transport):
    dm_occ_ots, idx_ots_occ, dm_occ_fts, idx_fts_occ = get_lev_data(
        DM_transport, "passenger_occupancy", lev_number=1
    )
    occ_rate = round(
        dm_occ_fts.array[idx_fts_occ["Vaud"], idx_fts_occ[2050], 0, idx_fts_occ["LDV"]],
        1,
    )
    for i in [2, 3, 4]:
        occ_rate += 0.1
        dm_occ_fts.array[idx_fts_occ["Vaud"], 1:, :, idx_fts_occ["LDV"]] = np.nan
        dm_occ_fts.array[
            idx_fts_occ["Vaud"], idx_fts_occ[2050], 0, idx_fts_occ["LDV"]
        ] = occ_rate
        dm_occ_fts.fill_nans("Years")
        DM_transport["fts"]["passenger_occupancy"][i] = dm_occ_fts.copy()
    return DM_transport


def efficiency_fts(DM_transport):
    # Take the average of lever 2 and 4 to get lever 3
    dic_dm_eff_fts = {}
    for lev_number in [2, 4]:
        _, _, dic_dm_eff_fts[lev_number], _ = get_lev_data(
            DM_transport, "passenger_veh-efficiency_new", lev_number=lev_number
        )
    DM_transport["fts"]["passenger_veh-efficiency_new"][3] = midpoint(
        dic_dm_eff_fts[2], dic_dm_eff_fts[4], 0.5
    )
    return DM_transport


def freight_demand_fts(DM_transport):
    dic_dm_freight_fts = {}
    for lev_number in [1, 4]:
        _, _, dic_dm_freight_fts[lev_number], _ = get_lev_data(
            DM_transport, "freight_tkm", lev_number=lev_number
        )
    DM_transport["fts"]["freight_tkm"][2] = midpoint(
        dic_dm_freight_fts[1], dic_dm_freight_fts[4], 0.25
    )
    DM_transport["fts"]["freight_tkm"][3] = midpoint(
        dic_dm_freight_fts[1], dic_dm_freight_fts[4], 0.75
    )

    return DM_transport


def freight_modal_share_fts(DM_transport):
    dic_dm_freight_fts = {}
    dic_idx_fts = {}
    dic_dm_freight_ots = {}
    dic_idx_ots = {}
    for lev_number in [1, 4]:
        dm_ots, idx_ots, dic_dm_freight_fts[lev_number], dic_idx_fts[lev_number] = (
            get_lev_data(DM_transport, "freight_modal-share", lev_number=lev_number)
        )
    dm_ots, idx_ots, dm_freight_fts, idx_fts = get_lev_data(
        DM_transport, "freight_modal-share", lev_number=1
    )

    # The objectif are for terrestial transport only
    share_without_aviation_2050 = (
        1 - dm_freight_fts.array[idx_fts["Vaud"], idx_fts[2050], 0, idx_fts["aviation"]]
    )

    dm_copy = dm_freight_fts.copy()
    for i, share in enumerate([0.6, 0.7]):
        i += 3
        dm_freight_fts = dm_copy.copy()

        dm_freight_fts.array[idx_fts["Vaud"], 1:-1, :, :] = np.nan

        dm_freight_fts.array[idx_fts["Vaud"], idx_fts[2050], :, idx_fts["rail"]] = (
            share * share_without_aviation_2050
        )

        dm_freight_fts.normalise_non_fixed_values(
            ["rail"],
            idx_fts[2050],
            variable_name="tra_freight_modal-share",
        )

        dm_freight_fts.fill_nans("Years")
        dm_freight_fts.normalise(dim="Categories1", inplace=True)
        DM_transport["fts"]["freight_modal-share"][i] = dm_freight_fts.copy()

    return DM_transport


def run(DM_transport: dict, country_list: list, years_ots: list, years_fts: list):
    """Fill missing data with example from other countries, midpoint, or estimated values

    Args:
        DM_transport (dict)
        country_list (list)
        years_ots (list)
        years_fts (list)

    Returns:
        dict: DM_transport updated
    """

    ## Implement the modal share for Vaud
    DM_transport = implement_modal_share_fts(DM_transport)

    # TODO : do aviation
    #### TECHNOLOGY SHARE ####
    DM_transport = tech_share_fts(DM_transport)

    ### CAR OCCUPANCY ####
    DM_transport = occupancy_fts(DM_transport)

    #### VEHICLE EFFICIENCY ####
    DM_transport = efficiency_fts(DM_transport)

    ### Freight demand ###
    DM_transport = freight_demand_fts(DM_transport)

    ### Freight modal share ###
    DM_transport = freight_modal_share_fts(DM_transport)
    ##### Save pickle #########
    this_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(this_dir, "../../../../data/datamatrix/transport.pickle")
    my_pickle_dump(DM_new=DM_transport, local_pickle_file=pickle_file)

    #### Demande de transport de stat vaud #####

    return DM_transport
