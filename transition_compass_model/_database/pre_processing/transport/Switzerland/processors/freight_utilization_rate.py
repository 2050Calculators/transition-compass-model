"""
Swiss freight OTS: load factor and utilisation rate for road modes (HDVH, HDVL, HDVM).

Variables
---------
tra_freight_load-factor : tkm/vkm
    Territorial tkm / territorial vkm. HDVL is fixed at 0.2 t/vkm by construction.
tra_freight_utilisation-rate : vkm/year
    Territorial vkm / Swiss-registered fleet, so that tkm / load factor /
    utilisation rate reproduces the BFS fleet.

Both are computed in processors/freight_road_ots.py (sources and assumptions are
documented there). Values are national and copied to every canton.
"""

from transition_compass_model.model.common.data_matrix_class import DataMatrix

_MODES_ROAD = ["HDVH", "HDVL", "HDVM"]


def run(years_ots: list, country_list: list, road: dict) -> DataMatrix:
    """Build the freight_utilization-rate DataMatrix (OTS, road modes).

    Parameters
    ----------
    years_ots : list of int
    country_list : list of str
    road : dict, output of freight_road_ots.run()

    Returns
    -------
    DataMatrix with Variables=['tra_freight_load-factor', 'tra_freight_utilisation-rate'],
    Categories1=['HDVH', 'HDVL', 'HDVM'], covering years_ots.
    """
    all_countries = ["Switzerland"] + [c for c in country_list if c != "Switzerland"]
    dm = DataMatrix(
        col_labels={
            "Country": all_countries,
            "Years": years_ots,
            "Variables": ["tra_freight_load-factor", "tra_freight_utilisation-rate"],
            "Categories1": _MODES_ROAD,
        },
        units={
            "tra_freight_load-factor": "tkm/vkm",
            "tra_freight_utilisation-rate": "vkm/year",
        },
    )
    for country in all_countries:
        for mode in _MODES_ROAD:
            dm[country, :, "tra_freight_load-factor", mode] = road["load_factor"][mode]
            dm[country, :, "tra_freight_utilisation-rate", mode] = road[
                "utilisation_rate"
            ][mode]
    return dm
