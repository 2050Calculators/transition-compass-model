"""
Swiss freight OTS: vehicle efficiency and technology share for new vehicles, plus the
EP2050+ readers and segment maps shared by the freight preprocessing.

Road classes are defined by body type, not weight: HDVH = articulated trucks
(EP2050 "TT/AT", BFS GTS "HAV", BFS fleet "Road tractor"), HDVM = rigid trucks
>= 3.5 t (EP2050 "RigidTruck", BFS "LORRY" / "Goods truck"), HDVL = vans < 3.5 t
(EP2050 "LCV", BFS "Delivery vehicle"). BFS tkm and the HBEFA consumption ratio only
exist by body type, so a split by weight would also need tkm and fuel factors by
weight class, not just new segment maps.

Variables
---------
tra_freight_vehicle-efficiency_new : MJ/km (road), MJ/tkm (non-road)
    Road: historical values equal the fleet efficiency back-calculated in
        processors/freight_road_ots.py (energy matched to FOEN CO2). They only drive
        the future stock-flow and the level-1 trend.
    Rail: EP2050+ freight rail energy (Abb. 112) / BFS rail tkm, CEV/diesel ratio kept.
    IWW, marine, aviation: literature constants.

tra_freight_technology-share_new : %
    Road: BFS first registrations by fuel (2005+), fleet split before
        (processors/freight_road_ots.py).
    Non-road: literature constants (see _NON_ROAD).

Sources
-------
EP2050 Verkehrssektor ZERO-Basis:
  Sheet 04 Energieverbrauch Strasse: HGV/LCV fuel energy (PJ), 1990-2060.
  Sheet 03 Fahrleistung: HGV/LCV total VKM (Mio. km), 1990-2060.
  Sheet 02 Flottenbestand: fleet counts by segment, 1990-2060.
  Sheet 05 Neuzulassungen Strasse: new registrations by segment (used for FTS level 4).
  # https://www.bfe.admin.ch/bfe/en/home/policy/energy-perspectives-2050-plus.html
  # file: EP2050+_Detailergebnisse 2020-2060_Verkehrssektor_alle Szenarien_2022-04-12.xlsx

Non-road efficiency constants:
  Rail CEV  0.117 MJ/tkm (ratio to diesel only; level from EP2050 energy) — SBB freight electric, consistent with IEA/UIC rail energy intensity.
  Rail ICE  0.316 MJ/tkm — Swiss diesel freight locomotive, EEA emission inventory reference.
  IWW       0.429 MJ/tkm — Rhine barge literature value (CCNR/IWT baseline).
  Marine    0.160 MJ/tkm — placeholder (Switzerland landlocked; tkm = 0 in model).
  Aviation 19.077 MJ/tkm — cargo aircraft, consistent with IATA fuel efficiency data.
"""

import os

import numpy as np
import pandas as pd

from transition_compass_model._database.pre_processing.transport.Switzerland.get_data_functions import (
    freight_data as fd,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# HBEFA 4.2: articulated trucks (TT/AT) consume ~38 % more fuel/km than rigid trucks
# at the same payload class due to higher gross vehicle mass and aerodynamic drag.
# https://www.hbefa.net/
_HDVH_HDVM_EFF_RATIO = 1.38  # HDVH ICE-diesel efficiency / HDVM ICE-diesel efficiency

# Efficiency ratios relative to ICE-diesel (MJ/km basis, road modes).
# ICE-gasoline +12 %: HBEFA 4.2 / COPERT petrol vs diesel heavy-duty vehicles.
#   https://www.hbefa.net/  https://www.emisia.com/utilities/copert/
# ICE-gas (CNG/LNG) ≈ same as diesel in MJ/km; IEA Future of Trucks (2017).
#   https://www.iea.org/reports/the-future-of-trucks
# BEV/CEV 30 % of diesel: electric drivetrain ~70 % more efficient; same IEA source.
# FCEV 33 %: ~10 % less efficient than BEV due to H2 compression/liquefaction losses.
# PHEV-diesel 55 %, PHEV-gasoline 62 %: 50 % electric driving fraction assumed.
#   ICCT HDV technology roadmap: https://theicct.org/publication/hdv-techno-roadmap-may21/
_ROAD_EFF_RATIOS = {
    "ICE-diesel": 1.00,
    "ICE-gasoline": 1.12,
    "ICE-gas": 0.99,
    "BEV": 0.30,
    "CEV": 0.30,
    "FCEV": 0.33,
    "PHEV-diesel": 0.55,
    "PHEV-gasoline": 0.62,
}

# Non-road efficiency (MJ/km) and baseline tech shares.
# Rail CEV 0.117 MJ/km: SBB annual energy report, electric freight traction.
#   https://company.sbb.ch/en/the-company/reports-and-key-figures/annual-report.html
# Rail ICE-diesel 0.316 MJ/km: EEA EMEP/EEA air pollutant emission guidebook 2023.
#   https://www.eea.europa.eu/publications/emep-eea-guidebook-2023
# IWW 0.429 MJ/km: CCNR inland navigation market observation report (Rhine barge).
#   https://www.ccr-zkr.org/
# Marine 0.160 MJ/km: placeholder (Switzerland landlocked, model sets tkm = 0).
# Aviation 19.077 MJ/km: IATA fuel efficiency data, cargo aircraft baseline.
#   https://www.iata.org/en/programs/ops-infra/fuel/
_NON_ROAD = {
    "rail": {
        "eff": {"CEV": 0.117, "ICE-diesel": 0.316},
        "share": {"CEV": 0.90, "ICE-diesel": 0.10},
    },
    "IWW": {
        "eff": {"ICE": 0.429, "BEV": 0.429, "FCEV": 0.429},
        "share": {"ICE": 1.0},
    },
    "marine": {
        "eff": {"ICE": 0.160, "BEV": 0.160, "FCEV": 0.160},
        "share": {"ICE": 1.0},
    },
    "aviation": {
        "eff": {"kerosene": 19.077, "BEV": 19.077},
        "share": {"kerosene": 1.0},
    },
}

# EP2050 Neuzulassungen segment → model technology, for HDVH (TT/AT)
_HDVH_SEG = {
    "TT/AT <=7,5t": "ICE-diesel",
    "TT/AT >7,5-14t": "ICE-diesel",
    "TT/AT >14-20t": "ICE-diesel",
    "TT/AT >20-28t": "ICE-diesel",
    "TT/AT >28-34t": "ICE-diesel",
    "TT/AT >34-40t": "ICE-diesel",
    "TT/AT >40-50t": "ICE-diesel",
    "TT/AT >50-60t": "ICE-diesel",
    "TT/AT >60t": "ICE-diesel",
    "TT/AT CNG": "ICE-gas",
    "TT/AT LNG": "ICE-gas",
    "TT/AT BEV": "BEV",
    "TT/AT PHEV": "PHEV-diesel",
    "TT/AT FCEV": "FCEV",
    "TT/AT <28t EE": "CEV",
    "TT/AT 28-34t EE": "CEV",
}

# EP2050 Neuzulassungen segment → model technology, for HDVM (RigidTruck)
_HDVM_SEG = {
    "RT petrol": "ICE-gasoline",
    "RigidTruck <7,5t": "ICE-diesel",
    "RigidTruck 7,5-12t": "ICE-diesel",
    "RigidTruck >12-14t": "ICE-diesel",
    "RigidTruck >14-20t": "ICE-diesel",
    "RigidTruck >20-26t": "ICE-diesel",
    "RigidTruck >26-28t": "ICE-diesel",
    "RigidTruck >28-32t": "ICE-diesel",
    "RigidTruck >32t": "ICE-diesel",
    "RigidTruck CNG ≤7,5t": "ICE-gas",
    "RigidTruck CNG >7,5-12t": "ICE-gas",
    "RigidTruck CNG >12t": "ICE-gas",
    "RigidTruck LNG <=7.5t": "ICE-gas",
    "RigidTruck LNG  >7.5-12t": "ICE-gas",
    "RigidTruck LNG >12t": "ICE-gas",
    "RigidTruck BEV ≤7.5t": "BEV",
    "RigidTruck BEV >7.5-12t": "BEV",
    "RigidTruck BEV >12t": "BEV",
    "RigidTruck PHEV <=7,5t": "PHEV-diesel",
    "RigidTruck PHEV >7,5-12t": "PHEV-diesel",
    "RigidTruck PHEV >12t": "PHEV-diesel",
    "RigidTruck FCEV <=7,5t": "FCEV",
    "RigidTruck FCEV >7,5-12t": "FCEV",
    "RigidTruck FCEV >12t": "FCEV",
    "RigidTruck <=7,5t EE": "CEV",
    "RigidTruck >7,5-12t EE": "CEV",
}

# Separation by weights

# EP2050 Neuzulassungen segment → model technology, for HDVL (LCV)
_HDVL_SEG = {
    "LCV petrol M+N1-I": "ICE-gasoline",
    "LCV petrol N1-II": "ICE-gasoline",
    "LCV petrol N1-III": "ICE-gasoline",
    "LCV diesel M+N1-I": "ICE-diesel",
    "LCV diesel N1-II": "ICE-diesel",
    "LCV diesel N1-III": "ICE-diesel",
    "LCV CNG/petrol M+N1-I": "ICE-gas",
    "LCV CNG/petrol N1-II": "ICE-gas",
    "LCV CNG/petrol N1-III": "ICE-gas",
    "LCV BEV M+N1-I": "BEV",
    "LCV BEV N1-II": "BEV",
    "LCV BEV N1-III": "BEV",
    "LCV PHEV diesel N1-III": "PHEV-diesel",
    "LCV FuelCell N1-III": "FCEV",
}

_ALL_TECHS = [
    "BEV",
    "CEV",
    "FCEV",
    "H2",
    "ICE",
    "ICE-diesel",
    "ICE-gas",
    "ICE-gasoline",
    "PHEV-diesel",
    "PHEV-gasoline",
    "kerosene",
]
_ROAD_MODES = ["HDVH", "HDVL", "HDVM"]
_ALL_MODES = ["HDVH", "HDVL", "HDVM", "IWW", "aviation", "marine", "rail"]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_THIS_DIR, "../data")
_EP2050_PATH = os.path.join(
    _DATA_DIR,
    "EP2050_sectors/EP2050+_Szenarienergebnisse_Details_Nachfragesektoren/"
    "EP2050+_Detailergebnisse 2020-2060_Verkehrssektor_alle Szenarien_2022-04-12.xlsx",
)


EP2050_tech_to_model = {
    "electricity": "BEV",
    "Electric": "CEV",
    "CNG": "ICE-gas",
    "LNG": "ICE-gas",
    "FFV": "ICE-gasoline",  # TODO check if acceptable
    "bifuel CNG/petrol": "ICE-gas",
    "CNG/petrol": "ICE-gas",
    "petrol 2S": "ICE-gasoline",
    "petrol 4S": "ICE-gasoline",
    "petrol (2S)": "ICE-gasoline",
    "petrol (4S)": "ICE-gasoline",
    "petrol": "ICE-gasoline",
    "bifuel LPG/petrol": "ICE-gasoline",
    "LPG/petrol": "ICE-gasoline",
    "LPG": "ICE-gas",
    "flex-fuel E85": "ICE-gasoline",
    "E85": "ICE-gasoline",
    "Plug-in Hybrid diesel/electric": "PHEV-diesel",
    "Plug-in Hybrid petrol/electric": "PHEV-gasoline",
    "FuelCell": "FCEV",
    "diesel": "ICE-diesel",
    "hydrogen": "H2",
    "Hybrid": "PHEV-diesel",
}


# ---------------------------------------------------------------------------
# EP2050 readers
# ---------------------------------------------------------------------------
def _read_ep2050_energy(scenario="ZERO-Basis") -> pd.DataFrame:
    """Read ZERO-Basis road-freight energy consumption from EP2050.

    The data covers light commercial vehicles (LCV) and heavy goods vehicles
    (HGV), disaggregated by fuel or propulsion technology. Energy values are
    expressed in petajoules (PJ) per year.

    Returns:
        pd.DataFrame : df containing:

        - ``Fahrzeugart``: vehicle category (``LCV`` or ``HGV``)
        - ``Treibstoff``: fuel or propulsion technology
        - year columns from 1990 to 2060: energy consumption in PJ

        Fuel names are normalized to the model terminology, including
        ``ICE-gasoline`` for petrol and ``BEV`` for electricity.
    """
    # Header row of each scenario sub-table (scenario label row + 4)
    header_scenar = {"ZERO-Basis": 19, "ZERO-B": 89, "WWB": 159}[scenario]
    df = pd.read_excel(
        _EP2050_PATH, sheet_name="04 Energieverbrauch Strasse", header=header_scenar
    )

    # Only keep truck data
    df_freight = df.iloc[7:17, 1:]
    df_freight.replace(EP2050_tech_to_model, inplace=True)

    return df_freight


def _read_ep2050_vkm(scenario="ZERO-Basis") -> pd.DataFrame:
    """Read ZERO-Basis road traffic activity from EP2050.

    The data covers light commercial vehicles (LCV) and heavy goods vehicles
    (HGV), disaggregated by propulsion technology. Traffic activity is given
    in million vehicle-kilometres (Mio Vehkm) per year for 1990–2060.

    Returns : pd.DataFrame
        DataFrame containing:

        - ``VehCat``: vehicle category (``LCV`` or ``HGV``)
        - ``Technology``: fuel or propulsion technology
        - ``Unit``: source unit (``Mio Vehkm``)
        - year columns from 1990 to 2060: vehicle-kilometres in million km

        Source technology names are normalized to the model terminology,
        including ``ICE-gasoline`` for petrol and ``BEV`` for electricity.
    """
    # Header row of each scenario sub-table (scenario label row + 4)
    header_scenar = {"ZERO-Basis": 19, "ZERO-B": 105, "WWB": 192}[scenario]

    df = pd.read_excel(_EP2050_PATH, sheet_name="03 Fahrleistung", header=header_scenar)
    # Get the vkm For the basis scenario  for LDV and HGV
    df_zero_basis_freight = df.iloc[9:22, 1:]

    df_zero_basis_freight.replace(EP2050_tech_to_model, inplace=True)

    df_zero_basis_freight = df_zero_basis_freight.groupby(
        ["VehCat", "Technology", "Unit"], as_index=False
    ).sum(numeric_only=True)

    return df_zero_basis_freight


def _read_ep2050_fleet_frac(years_list: list) -> dict:
    """Read the HDVH share of the HGV fleet by technology from EP2050.

    TT/AT (articulated truck) are classified  as HDVH and Rigid truck as HDVM.
    This separation is made because articulated trucks (TT/AT) consume ~38 % more fuel/km than rigid trucks.
    However the nomination should be changed.

    Args:
        years_list : list of int

    Returns:
        dict[str, np.ndarray] :  Mapping from model technology to an array aligned with ``years_list``.
        Each value is the HDVH fraction of the corresponding technology's HGV
        fleet. If no fleet data is available for a technology and year, a
        fallback fraction of ``0.25`` is used.
    """

    df = pd.read_excel(_EP2050_PATH, sheet_name="02 Flottenbestand", header=19).iloc[
        9:, 1:
    ]
    df["mode"] = np.select(
        [
            df["Segment"].isin(_HDVH_SEG),
            df["Segment"].isin(_HDVM_SEG),
        ],
        [
            "HDVH",
            "HDVM",
        ],
        default=None,  # or None if you want unmatched segments to be empty
    )

    df = df.loc[df["mode"].notnull()]
    df["tech"] = df["Segment"].map({**_HDVH_SEG, **_HDVM_SEG})

    frac_per_tech = {}
    for tech in df["tech"].unique():
        df_frac = pd.DataFrame(columns=years_list)

        df_frac.loc["HDVM", years_list] = df.loc[
            (df["mode"] == "HDVM") & (df["tech"] == tech), years_list
        ].sum()
        df_frac.loc["HDVH", years_list] = df.loc[
            (df["mode"] == "HDVH") & (df["tech"] == tech), years_list
        ].sum()

        df_frac.loc["total", :] = df_frac.sum()

        total = df_frac.loc["total", years_list]

        frac = (
            df_frac.loc["HDVH", years_list] / total.where(total > 0, np.nan)
        ).fillna(0.25)
        frac_per_tech[tech] = frac.values

    return frac_per_tech


def compute_efficiency(vkm, energy):
    # efficiency(MJ/km) = energy(PJ) × 1000 / VKM(Mkm)
    return np.where(vkm > 0, energy * 1000.0 / vkm, np.nan)


def compute_hdvm_hdvh_efficiency(hgv_vkm, hgv_energy, frac_hdvh):
    hgv_avg_eff = compute_efficiency(hgv_vkm, hgv_energy)

    # Split HDVH / HDVM using size ratio
    R = _HDVH_HDVM_EFF_RATIO
    denom = frac_hdvh * R + (1 - frac_hdvh)
    hdvm_eff = np.where(denom > 0, hgv_avg_eff / denom, np.nan)
    hdvh_eff = R * hdvm_eff
    return hdvm_eff, hdvh_eff


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run(years_ots: list, country_list: list, road: dict, DM_freight_ots: dict) -> dict:
    """Build freight_vehicle-efficiency_new and freight_technology-share_new DataMatrices.

    Parameters
    ----------
    years_ots : list of int
    country_list : list of str
    road : dict, output of freight_road_ots.run()
    DM_freight_ots : dict, output of freight_tkm_modal_share.run() (rail tkm)

    Returns
    -------
    dict with keys 'freight_vehicle-efficiency_new' and 'freight_technology-share_new',
    each a DataMatrix with Countries, Years=years_ots, Variables, Cat1=modes, Cat2=techs.
    """
    all_countries = ["Switzerland"] + [c for c in country_list if c != "Switzerland"]

    def _make_dm(varname, unit):
        return DataMatrix(
            col_labels={
                "Country": all_countries,
                "Years": years_ots,
                "Variables": [varname],
                "Categories1": _ALL_MODES,
                "Categories2": _ALL_TECHS,
            },
            units={varname: unit},
        )

    dm_eff = _make_dm("tra_freight_vehicle-efficiency_new", "MJ/km")
    dm_share = _make_dm("tra_freight_technology-share_new", "%")
    techs = dm_eff.col_labels["Categories2"]
    ch = "Switzerland"
    v_eff = "tra_freight_vehicle-efficiency_new"
    v_share = "tra_freight_technology-share_new"

    # --- Road: historical new-vehicle efficiency = fleet efficiency (as passenger);
    # it only drives the future stock-flow and the level-1 trend.
    for mode in _ROAD_MODES:
        for tech in road["eff"][mode].columns:
            if tech in techs:
                dm_eff[ch, :, v_eff, mode, tech] = road["eff"][mode][tech].values
        for tech in road["share_new"][mode].columns:
            if tech in techs:
                dm_share[ch, :, v_share, mode, tech] = road["share_new"][mode][
                    tech
                ].values

    # --- Non-road: literature constants, except rail energy from EP2050+
    for mode, data in _NON_ROAD.items():
        for tech, val in data["eff"].items():
            dm_eff[ch, :, v_eff, mode, tech] = val
        for tech, val in data["share"].items():
            dm_share[ch, :, v_share, mode, tech] = val
    _rail_efficiency_from_energy(dm_eff, DM_freight_ots, years_ots)

    # --- Copy Switzerland to all other countries ---
    for country in all_countries:
        if country != ch:
            dm_eff[country, :, :, :, :] = dm_eff[ch, :, :, :, :]
            dm_share[country, :, :, :, :] = dm_share[ch, :, :, :, :]

    return {
        "freight_vehicle-efficiency_new": dm_eff,
        "freight_technology-share_new": dm_share,
    }


def _rail_efficiency_from_energy(dm_eff, DM_freight_ots, years_ots):
    """Set rail efficiency [MJ/tkm] so that tkm x share x eff = EP2050 freight rail
    energy (Abb. 112). The CEV/diesel ratio and tech shares stay as in _NON_ROAD."""
    ch = "Switzerland"
    v = "tra_freight_vehicle-efficiency_new"
    dm_tkm = DM_freight_ots["freight_tkm"]
    dm_ms = DM_freight_ots["freight_modal-share"]
    tkm = np.array(
        [
            dm_tkm[ch, y, "tra_freight_tkm-total-demand"]
            * 1e9
            * dm_ms[ch, y, "tra_freight_modal-share", "rail"]
            for y in years_ots
        ]
    )
    energy_mj = fd.fill_years(fd.read_ep2050_rail_freight_energy(), years_ots) * 1e9
    rail = _NON_ROAD["rail"]
    ratio = rail["eff"]["ICE-diesel"] / rail["eff"]["CEV"]
    weight = rail["share"]["CEV"] + rail["share"]["ICE-diesel"] * ratio
    eff_cev = energy_mj / (tkm * weight)
    dm_eff[ch, :, v, "rail", "CEV"] = eff_cev
    dm_eff[ch, :, v, "rail", "ICE-diesel"] = eff_cev * ratio


if __name__ == "__main__":
    from processors.freight_road_ots import run as freight_road_run
    from processors.freight_tkm_modal_share import run as freight_tkm_run
    from processors.passenger_emission_factors import run as emission_factors_run

    from transition_compass_model._database.pre_processing.params import (
        country_list,
        years_ots,
    )

    road = freight_road_run(years_ots, emission_factors_run())
    DM_freight_eff_tech = run(
        years_ots, country_list, road, freight_tkm_run(years_ots, country_list)
    )
