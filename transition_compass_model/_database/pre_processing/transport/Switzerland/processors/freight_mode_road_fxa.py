"""
Swiss freight FXA: freight_mode_road (road modes: HDVH, HDVL, HDVM).

Variables
---------
tra_freight_lifetime : vkm
    Total km driven per vehicle over its operational life. The model computes
    renewal_rate = utilisation_rate / lifetime, so the lifetime is set to
    utilisation_rate / renewal_rate, with the renewal rate back-calculated from the
    BFS fleet and new registrations (processors/freight_road_ots.py).

    OTS: year by year.
    FTS: held at utilisation_rate(base year) / mean renewal rate over the last
    _RR_AVG_YEARS historical years, to avoid carrying one noisy year into the future.

Values are national and copied to every canton.
"""

import numpy as np

from transition_compass_model.model.common.data_matrix_class import DataMatrix

_MODES_ROAD = ["HDVH", "HDVL", "HDVM"]
_RR_AVG_YEARS = 10


def run(
    dm_utilization: DataMatrix,
    road: dict,
    years_ots: list,
    years_fts: list,
    country_list: list,
) -> DataMatrix:
    """Build freight_mode_road FXA DataMatrix.

    Parameters
    ----------
    dm_utilization : DataMatrix
        Output of freight_utilization_rate.run().
    road : dict
        Output of freight_road_ots.run() (uses "renewal_rate").
    years_ots, years_fts : list of int
    country_list : list of str

    Returns
    -------
    DataMatrix with Variables=['tra_freight_lifetime'],
    Categories1=_MODES_ROAD, covering OTS + FTS years.
    """
    years_all = years_ots + years_fts
    all_countries = ["Switzerland"] + [c for c in country_list if c != "Switzerland"]

    dm_fxa = DataMatrix(
        col_labels={
            "Country": all_countries,
            "Years": years_all,
            "Variables": ["tra_freight_lifetime"],
            "Categories1": _MODES_ROAD,
        },
        units={"tra_freight_lifetime": "vkm"},
    )
    ch = "Switzerland"
    for mode in _MODES_ROAD:
        ur = np.array(
            [
                dm_utilization[ch, yr, "tra_freight_utilisation-rate", mode]
                for yr in years_ots
            ]
        )
        rr = road["renewal_rate"][mode]
        lifetime = ur / rr
        rr_recent = np.mean(rr[-_RR_AVG_YEARS:])
        lifetime_fts = ur[-1] / rr_recent
        for country in all_countries:
            for j, yr in enumerate(years_ots):
                dm_fxa[country, yr, "tra_freight_lifetime", mode] = lifetime[j]
            for yr in years_fts:
                dm_fxa[country, yr, "tra_freight_lifetime", mode] = lifetime_fts
    return dm_fxa
