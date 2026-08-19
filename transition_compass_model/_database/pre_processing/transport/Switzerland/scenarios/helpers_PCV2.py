# ======================  IMPORT PACKAGES & DATA  ========================================================

import numpy as np


def compute_prop_tech_EV_2023(dm_new_tech_share_ots):
    idx = dm_new_tech_share_ots.idx
    denominator_EV_PHEV = (
        dm_new_tech_share_ots.array[
            idx["Vaud"],
            idx[2023],
            idx["tra_passenger_technology-share_new"],
            idx["LDV"],
            idx["BEV"],
        ]
        + dm_new_tech_share_ots.array[
            idx["Vaud"],
            idx[2023],
            idx["tra_passenger_technology-share_new"],
            idx["LDV"],
            idx["PHEV-gasoline"],
        ]
        + dm_new_tech_share_ots.array[
            idx["Vaud"],
            idx[2023],
            idx["tra_passenger_technology-share_new"],
            idx["LDV"],
            idx["PHEV-diesel"],
        ]
    )

    prop_tech_EV_2023 = {}
    # on fait les calculs suivants pour garder les proportions entre les différentes motorisations
    for tech in ["BEV", "PHEV-diesel", "PHEV-gasoline"]:
        prop_tech_EV_2023[tech] = prop_BEV_EV_2023 = (
            dm_new_tech_share_ots.array[
                idx["Vaud"],
                idx[2023],
                idx["tra_passenger_technology-share_new"],
                idx["LDV"],
                idx[tech],
            ]
            / denominator_EV_PHEV
        )

    prop_gasoline_ICE_2023 = dm_new_tech_share_ots.array[
        idx["Vaud"],
        idx[2023],
        idx["tra_passenger_technology-share_new"],
        idx["LDV"],
        idx["ICE-gasoline"],
    ] / (
        dm_new_tech_share_ots.array[
            idx["Vaud"],
            idx[2023],
            idx["tra_passenger_technology-share_new"],
            idx["LDV"],
            idx["ICE-gasoline"],
        ]
        + dm_new_tech_share_ots.array[
            idx["Vaud"],
            idx[2023],
            idx["tra_passenger_technology-share_new"],
            idx["LDV"],
            idx["ICE-diesel"],
        ]
    )
    prop_diesel_ICE_2023 = 1 - prop_gasoline_ICE_2023
    return prop_tech_EV_2023, prop_diesel_ICE_2023, prop_gasoline_ICE_2023


def compute_tech_share_for_buses(dm_new_tech_share_3):
    idx = dm_new_tech_share_3.idx
    dm_new_tech_share_3.array[idx["Vaud"], 1:, :, idx["bus"], idx["CEV"]] = np.nan
    dm_new_tech_share_3.array[idx["Vaud"], 1:, :, idx["bus"], idx["ICE-diesel"]] = (
        np.nan
    )
    dm_new_tech_share_3.array[idx["Vaud"], idx[2050], :, idx["bus"], idx["CEV"]] = 0.8
    dm_new_tech_share_3.array[
        idx["Vaud"], idx[2050], :, idx["bus"], idx["ICE-diesel"]
    ] = 0.2

    dm_new_tech_share_3.fill_nans("Years")
    return dm_new_tech_share_3
