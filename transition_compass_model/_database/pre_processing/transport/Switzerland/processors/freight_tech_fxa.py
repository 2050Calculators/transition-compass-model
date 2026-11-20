"""
Swiss freight FXA: freight_tech (fleet technology share + fleet efficiency).

Variables
---------
tra_freight_technology-share_fleet : %
    Share of each technology in the vehicle fleet, by mode x technology.
    Road: BFS fleet by fuel (2005+), EP2050+ before (processors/freight_road_ots.py).
    Non-road: same as the new-vehicle share (literature constants).

tra_freight_vehicle-efficiency_fleet : MJ/km (road), MJ/tkm (non-road)
    Road: back-calculated so that vkm x share x efficiency reproduces the energy
    targets (processors/freight_road_ots.py). Non-road: as new vehicles.

Notes
-----
The model (model/transport/utils.py, compute_fts_tech_split) keeps these values for
all historical years and only runs the fleet stock-flow after the base year, so they
set historical energy directly. FTS years are held flat here and overwritten by the
model's stock-flow.
"""

import numpy as np

from transition_compass_model.model.common.data_matrix_class import DataMatrix

_ROAD_MODES = ["HDVH", "HDVL", "HDVM"]


def run(
    dm_eff_new: DataMatrix,
    dm_share_new: DataMatrix,
    road: dict,
    years_ots: list,
    years_fts: list,
    country_list: list,
) -> DataMatrix:
    """Build freight_tech FXA DataMatrix.

    Parameters
    ----------
    dm_eff_new : DataMatrix
        freight_efficiency_tech_share.run()['freight_vehicle-efficiency_new'].
    dm_share_new : DataMatrix
        freight_efficiency_tech_share.run()['freight_technology-share_new'].
    road : dict
        Output of freight_road_ots.run() (uses "share_fleet").
    years_ots, years_fts : list of int
    country_list : list of str

    Returns
    -------
    DataMatrix with Variables=['tra_freight_technology-share_fleet',
    'tra_freight_vehicle-efficiency_fleet'], covering OTS + FTS years.
    """
    v_new = "tra_freight_technology-share_new"
    v_fleet = "tra_freight_technology-share_fleet"
    dm_share_fleet = dm_share_new.copy()
    dm_share_fleet.rename_col(v_new, v_fleet, dim="Variables")
    techs = dm_share_fleet.col_labels["Categories2"]
    for mode in _ROAD_MODES:
        share = road["share_fleet"][mode]
        for country in dm_share_fleet.col_labels["Country"]:
            dm_share_fleet[country, :, v_fleet, mode, :] = 0.0
            for tech in share.columns:
                if tech in techs:
                    dm_share_fleet[country, :, v_fleet, mode, tech] = share[tech].values

    # Road fleet efficiency is what freight_road_ots inverted; the new-vehicle
    # efficiency was set equal to it, so the copy holds for all modes.
    dm_eff_fleet = dm_eff_new.copy()
    dm_eff_fleet.rename_col(
        "tra_freight_vehicle-efficiency_new",
        "tra_freight_vehicle-efficiency_fleet",
        dim="Variables",
    )

    dm_tech = dm_share_fleet.copy()
    dm_tech.append(dm_eff_fleet, dim="Variables")

    # FTS: flat continuation (the model's stock-flow overwrites these years)
    dm_tech.add(np.nan, dim="Years", col_label=years_fts, dummy=True)
    dm_tech.fill_nans("Years")

    return dm_tech
