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
    """Vaud new buses in 2050: 80% electric (BEV + CEV), 20% diesel.

    The 80% is split between BEV (battery) and CEV (trolleybus) with their ratio in
    the first future year. Other technologies go to 0 in 2050. Years in between
    are interpolated linearly, so the shares still add up to 1.
    """
    idx = dm_new_tech_share_3.idx
    techs = dm_new_tech_share_3.col_labels["Categories2"]
    first = dm_new_tech_share_3.col_labels["Years"][0]
    bev = np.nan_to_num(
        dm_new_tech_share_3.array[idx["Vaud"], idx[first], :, idx["bus"], idx["BEV"]]
    )
    cev = np.nan_to_num(
        dm_new_tech_share_3.array[idx["Vaud"], idx[first], :, idx["bus"], idx["CEV"]]
    )
    elec = bev + cev
    share_bev = np.where(elec > 0, bev / np.where(elec > 0, elec, 1), 1.0)

    idx_techs = [idx[t] for t in techs]
    dm_new_tech_share_3.array[idx["Vaud"], 1:, :, idx["bus"], :] = np.nan
    target = {t: 0.0 for t in techs}
    for t in techs:
        # keep technologies that are NaN for buses (not used) at NaN
        if np.isnan(
            dm_new_tech_share_3.array[idx["Vaud"], idx[first], :, idx["bus"], idx[t]]
        ).all():
            target[t] = np.nan
    target["BEV"] = 0.8 * share_bev
    target["CEV"] = 0.8 * (1 - share_bev)
    target["ICE-diesel"] = 0.2
    for t, i in zip(techs, idx_techs, strict=True):
        dm_new_tech_share_3.array[idx["Vaud"], idx[2050], :, idx["bus"], i] = target[t]

    dm_new_tech_share_3.fill_nans("Years")
    return dm_new_tech_share_3
