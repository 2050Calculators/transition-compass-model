"""
Swiss freight FXA: freight_mode_other (non-road modes: IWW, aviation, marine, rail).

Variables
---------
tra_freight_tkm-by-veh : tkm/vehicle-year
    Vehicle productivity = modal_tkm / fleet_size. Computed from Swiss modal tkm
    (from freight_tkm_modal_share) divided by fixed Swiss fleet size constants.
    Varies over OTS years with actual tkm; held flat at 2023 value for FTS.

tra_freight_renewal-rate : %
    Fleet renewal fraction per year. Fixed constants (flat across all years)
    derived from Swiss transport analysis used as reference in EU model.

Marine is landlocked (tkm = 0). tkm-by-veh is set to a positive placeholder so
that fleet = tkm / tkm-by-veh = 0 without division by zero.

Fleet size constants (vehicles) — Swiss-specific
-------------------------------------------------
Rail    7 000 wagons  — SBB Cargo + other rail operators (SBB Annual Report).
  https://company.sbb.ch/en/the-company/reports-and-key-figures/annual-report.html
IWW         4 barges  — Swiss-registered Rhine barges (Häfen Beider Basel annual report).
  https://www.hbbb.ch/en/
Aviation   12 aircraft — Full-freighter cargo aircraft at Swiss airports (ZRH, GVA, BSL).
  https://www.bazl.admin.ch/bazl/en/home/specialists/statistics-and-reports.html
Marine      1 ship    — Placeholder (Switzerland landlocked)

Renewal rate constants
----------------------
1 / renewal_rate gives approximate fleet operational life in years.
IWW 0.1559, aviation 0.1559 (≈ 6.4 yr) — from EU EUCALC transport module defaults.
Rail 0.4095 (≈ 2.4 yr) — from EU EUCALC transport module defaults.
Marine >1 placeholder — never used in model (marine tkm = 0 for Switzerland).
  EU Calculator project: https://www.european-calculator.eu/

tkm-by-veh is identical across countries (vehicle productivity is not canton-
specific). For Vaud, IWW tkm = 0 so the model computes fleet = 0/tkm-by-veh = 0.
"""

from transition_compass_model.model.common.data_matrix_class import DataMatrix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_FLEET_SIZE = {
    "rail": 7_000,
    "IWW": 4,
    "aviation": 12,
    "marine": 1,  # placeholder, Switzerland landlocked
}

_RENEWAL_RATE = {
    "IWW": 0.1559,
    "aviation": 0.1559,
    "marine": 1.1559,  # >1 placeholder: marine tkm = 0 for Switzerland
    "rail": 0.4095,
}

_MODES_OTHER = ["IWW", "aviation", "marine", "rail"]


def run(
    DM_freight_ots: dict, years_ots: list, years_fts: list, country_list: list
) -> DataMatrix:
    """Build freight_mode_other FXA DataMatrix.

    Parameters
    ----------
    DM_freight_ots : dict
        Output of freight_tkm_modal_share.run(), containing 'freight_tkm'
        and 'freight_modal-share' DataMatrices.
    years_ots : list of int
    years_fts : list of int
    country_list : list of str

    Returns
    -------
    DataMatrix with Variables=['tra_freight_renewal-rate', 'tra_freight_tkm-by-veh'],
    Categories1=_MODES_OTHER, covering OTS + FTS years.
    """
    years_all = years_ots + years_fts

    dm_tkm = DM_freight_ots["freight_tkm"]
    dm_modal = DM_freight_ots["freight_modal-share"]
    idx_t = dm_tkm.idx
    idx_m = dm_modal.idx

    all_countries = ["Switzerland", "Vaud"] + [
        c for c in country_list if c not in ("Switzerland", "Vaud")
    ]

    dm_fxa = DataMatrix(
        col_labels={
            "Country": all_countries,
            "Years": years_all,
            "Variables": ["tra_freight_renewal-rate", "tra_freight_tkm-by-veh"],
            "Categories1": _MODES_OTHER,
        },
        units={"tra_freight_renewal-rate": "%", "tra_freight_tkm-by-veh": "tkm"},
    )
    idx = dm_fxa.idx

    # --- Renewal rate: flat constant for all years and countries ----------
    for country in all_countries:
        for mode in _MODES_OTHER:
            dm_fxa.array[
                idx[country], :, idx["tra_freight_renewal-rate"], idx[mode]
            ] = _RENEWAL_RATE[mode]

    # --- tkm-by-veh: compute from Switzerland OTS, copy to others --------
    # (vehicle productivity is not canton-specific)
    ch = "Switzerland"
    for j, yr in enumerate(years_ots):
        total_bn_tkm = dm_tkm.array[idx_t[ch], idx_t[yr], 0]
        for mode in _MODES_OTHER:
            share = dm_modal.array[idx_m[ch], idx_m[yr], 0, idx_m[mode]]
            modal_tkm = total_bn_tkm * share * 1e9  # bn-tkm → tkm
            dm_fxa.array[idx[ch], idx[yr], idx["tra_freight_tkm-by-veh"], idx[mode]] = (
                modal_tkm / _FLEET_SIZE[mode]
            )

    # FTS years: hold 2023 value flat
    last_ots = years_ots[-1]
    for mode in _MODES_OTHER:
        val = dm_fxa.array[
            idx[ch], idx[last_ots], idx["tra_freight_tkm-by-veh"], idx[mode]
        ]
        for yr in years_fts:
            dm_fxa.array[idx[ch], idx[yr], idx["tra_freight_tkm-by-veh"], idx[mode]] = (
                val
            )

    # Copy Switzerland tkm-by-veh to all other countries
    for country in all_countries:
        if country != ch:
            dm_fxa.array[idx[country], :, idx["tra_freight_tkm-by-veh"], :] = (
                dm_fxa.array[idx[ch], :, idx["tra_freight_tkm-by-veh"], :]
            )

    return dm_fxa
