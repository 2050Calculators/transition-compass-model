"""
Swiss road freight OTS (HDVH, HDVM, HDVL): back-calculate the model inputs from
official targets, so that the transport model reproduces them for the historical years.

Model chain (model/transport/workflows.py, freight_fleet_energy)
----------------------------------------------------------------
    vkm    = tkm / load_factor                       [vkm]
    fleet  = vkm / utilisation_rate                  [number]
    rr     = utilisation_rate / lifetime             [%/yr]
    energy = vkm * tech_share_fleet * eff_fleet      [MJ]
    CO2    = energy * (1 - biofuel share) * EF       [g]   (biofuel on ICE/PHEV only)

Each input is inverted from a target. Priority: BFS / FOEN > EP2050+ > literature.

    tkm        BFS GTS, territorial (freight_data.read_gts_road_tkm)
    vkm        EP2050+ sheet 03 (HGV, LCV). HGV is split between HDVH and HDVM
               with BFS fleet counts x _HDVH_HDVM_MILEAGE_RATIO.
    fleet      BFS stock by type and fuel 2005+ (stats.swiss); 1990-2004 from EP2050+
               sheet 02, scaled to BFS in 2005.
    new        BFS first registrations 2005+; before 2005 back-calculated from the
               fleet with compute_renewal_rate_and_adjust.
    energy     EP2050+ sheet 04 fuel split, rescaled per year so that
               energy * (1 - biofuel) * EF reproduces FOEN CO2 of heavy-duty trucks
               (HDVH + HDVM) and light-duty trucks (HDVL).

    load_factor      = tkm / vkm                          (HDVL: fixed 0.2 t/vkm)
    utilisation_rate = vkm / fleet                         (vkm/veh/yr)
    rr               = from the BFS fleet stock-flow balance
    tech_share_fleet = fleet_tech / fleet
    eff_fleet        = energy_tech / (vkm * tech_share_fleet)

Accounting basis: tkm, vkm, energy and CO2 are territorial (all traffic on Swiss roads,
any registration). The fleet is the Swiss-registered fleet (what the industry module
needs), so utilisation_rate = territorial vkm / Swiss fleet also carries foreign
trucks' km.

All values are national; cantons reuse them (only tkm is scaled per canton).
"""

import numpy as np
import pandas as pd
from processors.freight_efficiency_tech_share import (
    _HDVH_HDVM_EFF_RATIO,
    _HDVH_SEG,
    _HDVL_SEG,
    _HDVM_SEG,
    _ROAD_EFF_RATIOS,
)
from processors.freight_tkm_modal_share import _HDVL_PAYLOAD_T_PER_VKM, road_tkm
from processors.passenger_renewal_rate_and_waste_pipeline import (
    compute_renewal_rate_and_adjust,
)
from processors.transport_ots_pickle import _CH_BIOFUEL_BLEND

from transition_compass_model._database.pre_processing.transport.Switzerland.get_data_functions import (
    freight_data as fd,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

ROAD_MODES = fd.ROAD_MODES

# Articulated (HDVH) trucks drive more km per year than rigid (HDVM) trucks: they are
# mostly long-haul. Used to split EP2050 HGV vkm, which is not broken down by body type.
# TODO: placeholder, source it (ASTRA LSVA / HBEFA mileage by vehicle type).
_HDVH_HDVM_MILEAGE_RATIO = 2.0

# Van load (t/vkm): HDVL tkm is built as EP2050 LCV vkm x this value in
# freight_tkm_modal_share, so the load factor is the same constant by construction.
# TODO: no traceable source (see backlog).
_HDVL_LOAD_FACTOR = _HDVL_PAYLOAD_T_PER_VKM

# A technology needs at least this fleet share for its efficiency to be inverted from
# energy / (vkm * share); below it the ratio to diesel (_ROAD_EFF_RATIOS) is used.
_MIN_SHARE_FOR_INVERSION = 0.01

# Upper bound for the renewal rate when back-calculating it from the fleet.
_MAX_RR = 0.2

# Model technology -> CO2 emission factor key in cp_tra_emission-factor
_TECH_TO_EF = {
    "ICE-diesel": "ICE-diesel",
    "PHEV-diesel": "PHEV-diesel",
    "ICE-gasoline": "ICE-gasoline",
    "PHEV-gasoline": "PHEV-gasoline",
    "ICE-gas": "ICE-gas",
    "FCEV": "H2",
}
# Technologies the model blends with biofuel (utils.add_biofuel_efuel: PHEV.*|ICE.*)
_BIO_TECHS = ["ICE-diesel", "PHEV-diesel", "ICE-gasoline", "PHEV-gasoline", "ICE-gas"]

# EP2050 energy rows use "H2" for fuel-cell trucks; the model technology is FCEV.
_EP_ENERGY_RENAME = {"H2": "FCEV"}


def _co2_factors(cdm_emission_factors) -> dict:
    """Return {tech: g CO2 / MJ} for road technologies."""
    idx = cdm_emission_factors.idx
    arr = cdm_emission_factors.array
    return {
        tech: float(arr[0, idx["CO2"], idx[key]]) for tech, key in _TECH_TO_EF.items()
    }


def _biofuel_share(years: list) -> np.ndarray:
    """Historical road biofuel blend (same values as the fuel-mix OTS lever)."""
    return np.array([_CH_BIOFUEL_BLEND.get(y, 0.0) for y in years])


def _fleet_by_tech(years: list) -> dict:
    """Swiss-registered fleet {mode: DataFrame(year x tech)} for all OTS years.

    BFS 2005+. Before 2005: EP2050 fleet by technology, scaled per mode so that the
    total matches BFS in 2005.
    """
    bfs = fd.read_bfs_goods_fleet()
    ep = fd.read_ep2050_fleet_by_tech(
        {"HDVH": _HDVH_SEG, "HDVM": _HDVM_SEG, "HDVL": _HDVL_SEG}
    )
    out = {}
    for mode in ROAD_MODES:
        b = bfs[mode]
        first = b.index.min()
        techs = sorted(set(b.columns) | set(ep[mode].columns))
        b = b.reindex(columns=techs, fill_value=0.0)
        e = ep[mode].reindex(columns=techs, fill_value=0.0)
        scale = b.loc[first].sum() / e.loc[first].sum()
        early = e.loc[[y for y in years if y < first]] * scale
        tab = pd.concat([early, b.loc[[y for y in years if y >= first]]])
        out[mode] = tab.reindex(years).astype(float)
    return out


def _new_by_tech(years: list, fleet: dict) -> tuple:
    """Return (new {mode: DataFrame(year x tech)}, rr {mode: np.ndarray}).

    The renewal rate comes from the fleet stock-flow with BFS new registrations
    (2005+); before 2005 new registrations are back-calculated by
    compute_renewal_rate_and_adjust. The technology split of new vehicles is BFS
    (2005+) and the fleet split before.
    """
    bfs_new = fd.read_bfs_goods_new()
    first = min(bfs_new[m].index.min() for m in ROAD_MODES)

    stock = np.array([fleet[m].sum(axis=1).values for m in ROAD_MODES]).T
    new_tot = np.full_like(stock, np.nan)
    for j, m in enumerate(ROAD_MODES):
        s = bfs_new[m].sum(axis=1)
        for i, y in enumerate(years):
            if y in s.index:
                new_tot[i, j] = s[y]

    dm = DataMatrix(
        col_labels={
            "Country": ["Switzerland"],
            "Years": years,
            "Variables": ["tra_freight_vehicle-fleet", "tra_freight_new-vehicles"],
            "Categories1": ROAD_MODES,
        },
        units={
            "tra_freight_vehicle-fleet": "number",
            "tra_freight_new-vehicles": "number",
        },
    )
    dm.array[0, :, 0, :] = stock
    dm.array[0, :, 1, :] = new_tot
    compute_renewal_rate_and_adjust(
        dm,
        var_names={
            "stock": "tra_freight_vehicle-fleet",
            "new": "tra_freight_new-vehicles",
            "waste": "tra_freight_vehicle-waste",
            "renewal-rate": "tra_freight_renewal-rate",
        },
        max_rr=_MAX_RR,
    )
    idx = dm.idx
    rr = {
        m: dm.array[0, :, idx["tra_freight_renewal-rate"], idx[m]] for m in ROAD_MODES
    }
    new_total = {
        m: dm.array[0, :, idx["tra_freight_new-vehicles"], idx[m]] for m in ROAD_MODES
    }

    new = {}
    for m in ROAD_MODES:
        techs = fleet[m].columns
        share_fleet = fleet[m].div(fleet[m].sum(axis=1), axis=0)
        b = bfs_new[m].reindex(columns=techs, fill_value=0.0)
        share_bfs = b.div(b.sum(axis=1).where(b.sum(axis=1) > 0), axis=0)
        share = share_fleet.copy()
        for y in years:
            if y >= first and y in share_bfs.index:
                share.loc[y] = share_bfs.loc[y]
        new[m] = share.mul(new_total[m], axis=0)
    return new, rr


def _vkm_by_mode(years: list, fleet: dict) -> dict:
    """Territorial vkm {mode: np.ndarray} in million vkm from EP2050+ sheet 03."""
    ep = fd.read_ep2050_vkm_by_tech()
    hgv = fd.fill_years(ep["HGV"].sum(axis=1), years)
    lcv = fd.fill_years(ep["LCV"].sum(axis=1), years)
    w_h = fleet["HDVH"].sum(axis=1).values * _HDVH_HDVM_MILEAGE_RATIO
    w_m = fleet["HDVM"].sum(axis=1).values
    frac_h = w_h / (w_h + w_m)
    return {"HDVH": hgv * frac_h, "HDVM": hgv * (1 - frac_h), "HDVL": lcv}


def _energy_targets(years: list, vkm: dict, ef: dict, bio: np.ndarray) -> tuple:
    """Return (energy {mode: DataFrame(year x tech)} in MJ, FOEN scaling factors).

    EP2050 fuel split, HGV split into HDVH/HDVM by vkm x _HDVH_HDVM_EFF_RATIO, then
    combustion technologies rescaled so that CO2 matches FOEN for heavy (HDVH + HDVM)
    and light (HDVL) trucks.
    """
    ep = fd.read_ep2050_energy_by_tech()
    foen = fd.read_foen_truck_co2()

    def _ep(cat):
        tab = ep[cat].rename(columns=_EP_ENERGY_RENAME)
        return pd.DataFrame(
            {t: fd.fill_years(tab[t], years) for t in tab.columns}, index=years
        )

    hgv, lcv = _ep("HGV"), _ep("LCV")
    w_h = vkm["HDVH"] * _HDVH_HDVM_EFF_RATIO
    w_m = vkm["HDVM"]
    frac_h = w_h / (w_h + w_m)
    energy_pj = {
        "HDVH": hgv.mul(frac_h, axis=0),
        "HDVM": hgv.mul(1 - frac_h, axis=0),
        "HDVL": lcv,
    }

    def _co2_mt(e_pj):
        tot = np.zeros(len(years))
        for tech in e_pj.columns:
            if tech in ef:
                fossil = (1 - bio) if tech in _BIO_TECHS else 1.0
                tot += e_pj[tech].values * fossil * ef[tech] * 1e-3  # PJ*g/MJ -> Mt
        return tot

    factors = {}
    for group, modes in [("heavy", ["HDVH", "HDVM"]), ("light", ["HDVL"])]:
        target = fd.fill_years(foen[group], years)
        model = sum(_co2_mt(energy_pj[m]) for m in modes)
        f = target / model
        factors[group] = f
        for m in modes:
            for tech in energy_pj[m].columns:
                if tech in ef:
                    energy_pj[m][tech] = energy_pj[m][tech] * f

    energy_mj = {m: e * 1e9 for m, e in energy_pj.items()}
    return energy_mj, factors


def _efficiency(years, vkm, share, energy, ef, bio) -> dict:
    """Fleet efficiency {mode: DataFrame(year x tech)} in MJ/km.

    eff = energy_tech / (vkm * share_tech). Technologies with a small share, or no
    EP2050 energy, use ICE-diesel x _ROAD_EFF_RATIOS. ICE-diesel then closes the
    balance so that each mode's CO2 equals its target exactly.
    """
    out = {}
    for m in ROAD_MODES:
        km = vkm[m] * 1e6
        sh = share[m]
        e = energy[m].reindex(columns=sh.columns, fill_value=0.0)
        eff = pd.DataFrame(index=years, columns=sh.columns, dtype=float)
        # first pass: diesel from its own energy
        diesel = e["ICE-diesel"].values / (km * sh["ICE-diesel"].values)
        for tech in sh.columns:
            direct = (sh[tech].values >= _MIN_SHARE_FOR_INVERSION) & (
                e[tech].values > 0
            )
            ratio = diesel * _ROAD_EFF_RATIOS.get(tech, 1.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                inv = e[tech].values / (km * sh[tech].values)
            eff[tech] = np.where(direct, inv, ratio)

        # close CO2 per mode on ICE-diesel
        target = np.zeros(len(years))
        others = np.zeros(len(years))
        for tech in sh.columns:
            if tech not in ef:
                continue
            fossil = (1 - bio) if tech in _BIO_TECHS else 1.0
            target += e[tech].values * fossil * ef[tech]
            if tech != "ICE-diesel":
                others += km * sh[tech].values * eff[tech].values * fossil * ef[tech]
        eff["ICE-diesel"] = (target - others) / (
            km * sh["ICE-diesel"].values * (1 - bio) * ef["ICE-diesel"]
        )
        # technologies set from the diesel ratio follow the closed diesel value
        for tech in sh.columns:
            if tech == "ICE-diesel":
                continue
            direct = (sh[tech].values >= _MIN_SHARE_FOR_INVERSION) & (
                e[tech].values > 0
            )
            eff[tech] = np.where(
                direct,
                eff[tech].values,
                eff["ICE-diesel"].values * _ROAD_EFF_RATIOS.get(tech, 1.0),
            )
        out[m] = eff
    return out


def run(years_ots: list, cdm_emission_factors) -> dict:
    """Compute road freight OTS inputs for Switzerland.

    Parameters
    ----------
    years_ots : list of int
    cdm_emission_factors : ConstantDataMatrix cp_tra_emission-factor [g/MJ]

    Returns
    -------
    dict with, per mode (HDVH, HDVL, HDVM):
        "vkm" [million vkm], "load_factor" [tkm/vkm], "utilisation_rate" [vkm/yr],
        "renewal_rate" [%], "fleet" [number, year x tech], "new" [number, year x tech],
        "share_fleet" [%], "share_new" [%], "eff" [MJ/km, year x tech],
        "energy" [MJ, year x tech]; plus "foen_factor" {heavy, light}.
    """
    ef = _co2_factors(cdm_emission_factors)
    bio = _biofuel_share(years_ots)
    tkm_road = road_tkm(years_ots)  # million tkm, territorial

    fleet = _fleet_by_tech(years_ots)
    new, rr = _new_by_tech(years_ots, fleet)
    vkm = _vkm_by_mode(years_ots, fleet)

    load_factor = {
        "HDVH": tkm_road["HDVH"] / vkm["HDVH"],
        "HDVM": tkm_road["HDVM"] / vkm["HDVM"],
        "HDVL": np.full(len(years_ots), _HDVL_LOAD_FACTOR),
    }
    fleet_tot = {m: fleet[m].sum(axis=1).values for m in ROAD_MODES}
    utilisation = {m: vkm[m] * 1e6 / fleet_tot[m] for m in ROAD_MODES}
    share_fleet = {m: fleet[m].div(fleet_tot[m], axis=0) for m in ROAD_MODES}
    share_new = {m: new[m].div(new[m].sum(axis=1), axis=0) for m in ROAD_MODES}

    energy, factors = _energy_targets(years_ots, vkm, ef, bio)
    eff = _efficiency(years_ots, vkm, share_fleet, energy, ef, bio)

    return {
        "vkm": vkm,
        "load_factor": load_factor,
        "utilisation_rate": utilisation,
        "renewal_rate": rr,
        "fleet": fleet,
        "new": new,
        "share_fleet": share_fleet,
        "share_new": share_new,
        "eff": eff,
        "energy": energy,
        "foen_factor": factors,
    }
