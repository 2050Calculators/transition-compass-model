"""
Swiss freight FXA: freight_tech (fleet technology share + fleet efficiency).

Variables
---------
tra_freight_technology-share_fleet : %
    Share of each technology in the total vehicle fleet, by mode × technology.
    OTS: initialised from new-vehicle tech shares (Step 5).  Since ICE-diesel
    dominated freight through 2023, fleet composition ≈ new-vehicle composition.
    FTS: extended flat from 2023 then overwritten at runtime by compute_fts_tech_split.

tra_freight_vehicle-efficiency_fleet : MJ/km
    Fleet-average energy intensity per technology, by mode × technology.
    OTS: initialised equal to new-vehicle efficiency (Step 5).  At fleet
    initialisation the fleet is homogeneous in each technology bucket, so fleet
    efficiency = new-vehicle efficiency.  The model evolves this via stock-flow.
    FTS: extended flat then overwritten by compute_fts_tech_split.

Notes
-----
compute_fts_tech_split (transport/utils.py) overwrites both variables for all
years ≥ 2010.  Only values for 1990–2009 are used as initial conditions for
the stock-flow, so the ICE-diesel = ~100% approximation is appropriate.
"""

import numpy as np

from transition_compass_model.model.common.data_matrix_class import DataMatrix


def run(
    dm_eff_new: DataMatrix,
    dm_share_new: DataMatrix,
    years_ots: list,
    years_fts: list,
    country_list: list,
) -> DataMatrix:
    """Build freight_tech FXA DataMatrix.

    Parameters
    ----------
    dm_eff_new : DataMatrix
        Output of freight_efficiency_tech_share.run()['freight_vehicle-efficiency_new'].
    dm_share_new : DataMatrix
        Output of freight_efficiency_tech_share.run()['freight_technology-share_new'].
    years_ots : list of int
    years_fts : list of int
    country_list : list of str

    Returns
    -------
    DataMatrix with Variables=['tra_freight_technology-share_fleet',
    'tra_freight_vehicle-efficiency_fleet'], covering OTS + FTS years.
    """
    # --- Fleet tech share: copy new-vehicle share and rename ---
    dm_share_fleet = dm_share_new.copy()
    dm_share_fleet.rename_col(
        "tra_freight_technology-share_new",
        "tra_freight_technology-share_fleet",
        dim="Variables",
    )

    # --- Fleet efficiency: copy new-vehicle efficiency and rename ---
    dm_eff_fleet = dm_eff_new.copy()
    dm_eff_fleet.rename_col(
        "tra_freight_vehicle-efficiency_new",
        "tra_freight_vehicle-efficiency_fleet",
        dim="Variables",
    )

    # --- Combine into one DataMatrix ---
    dm_tech = dm_share_fleet.copy()
    dm_tech.append(dm_eff_fleet, dim="Variables")

    # --- Extend to FTS with flat continuation (model overwrites 2010+ anyway) ---
    dm_tech.add(np.nan, dim="Years", col_label=years_fts, dummy=True)
    dm_tech.fill_nans("Years")

    return dm_tech
