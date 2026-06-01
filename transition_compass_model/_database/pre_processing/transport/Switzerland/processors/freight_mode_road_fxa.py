"""
Swiss freight FXA: freight_mode_road (road modes: HDVH, HDVL, HDVM).

Variables
---------
tra_freight_lifetime : vkm
    Total km driven per vehicle over its operational life.
    Computed as: lifetime_years × utilisation_rate (vkm/year).
    Varies over OTS years following utilisation_rate trends; held flat at last
    OTS value for FTS years (standard FXA pattern).

Lifetime constants (years) — calibrated to Swiss conditions
------------------------------------------------------------
HDVH   12 yr  — heavy articulated trucks (HAV); typical CH/EU lifecycle
HDVM   12 yr  — medium rigid trucks (LORRY ≥3.5 t); similar to HDVH
HDVL   14 yr  — light commercial vans (<3.5 t); longer service life

Sources
-------
Lifetime-years: ASTAG (Swiss Road Transport Association) fleet analysis,
consistent with EU TRACCS project default lifetimes for HGV (~12 yr) and LCV (~14 yr).
  ASTAG: https://www.astag.ch/
  TRACCS (Transport data and CO2 Analysis): https://traccs.emisia.com/
  COPERT fleet model: https://www.emisia.com/utilities/copert/
Utilisation rate: computed by freight_utilization_rate.run().
"""

from transition_compass_model.model.common.data_matrix_class import DataMatrix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LIFETIME_YEARS = {
    "HDVH": 12.0,
    "HDVM": 12.0,
    "HDVL": 14.0,
}

_MODES_ROAD = ["HDVH", "HDVL", "HDVM"]


def run(
    dm_utilization: DataMatrix,
    years_ots: list,
    years_fts: list,
    country_list: list,
) -> DataMatrix:
    """Build freight_mode_road FXA DataMatrix.

    Parameters
    ----------
    dm_utilization : DataMatrix
        Output of freight_utilization_rate.run(); must contain
        'tra_freight_utilisation-rate' for HDVH, HDVL, HDVM.
    years_ots : list of int
    years_fts : list of int
    country_list : list of str

    Returns
    -------
    DataMatrix with Variables=['tra_freight_lifetime'],
    Categories1=_MODES_ROAD, covering OTS + FTS years.
    """
    years_all = years_ots + years_fts
    last_ots = years_ots[-1]

    all_countries = ["Switzerland", "Vaud"] + [
        c for c in country_list if c not in ("Switzerland", "Vaud")
    ]

    dm_fxa = DataMatrix(
        col_labels={
            "Country": all_countries,
            "Years": years_all,
            "Variables": ["tra_freight_lifetime"],
            "Categories1": _MODES_ROAD,
        },
        units={"tra_freight_lifetime": "vkm"},
    )
    idx = dm_fxa.idx
    idx_u = dm_utilization.idx
    ch = "Switzerland"

    # OTS: lifetime_vkm = lifetime_years × utilisation_rate(t)
    for mode in _MODES_ROAD:
        for yr in years_ots:
            ur = dm_utilization.array[
                idx_u[ch],
                idx_u[yr],
                idx_u["tra_freight_utilisation-rate"],
                idx_u[mode],
            ]
            dm_fxa.array[idx[ch], idx[yr], 0, idx[mode]] = _LIFETIME_YEARS[mode] * ur

    # FTS: hold last OTS value flat
    for mode in _MODES_ROAD:
        val = dm_fxa.array[idx[ch], idx[last_ots], 0, idx[mode]]
        for yr in years_fts:
            dm_fxa.array[idx[ch], idx[yr], 0, idx[mode]] = val

    # Copy Switzerland to all other countries
    for country in all_countries:
        if country != ch:
            dm_fxa.array[idx[country], :, :, :] = dm_fxa.array[idx[ch], :, :, :]

    return dm_fxa
