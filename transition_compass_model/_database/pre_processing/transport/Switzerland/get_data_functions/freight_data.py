"""
Readers for the official datasets used as targets by the Swiss freight preprocessing.

Every reader returns plain pandas objects indexed by year, in the units of the source
(converted where stated). Processors combine them; this module only reads.

Priority when sources overlap: BFS / FOEN first, EP2050+ second.

Datasets
--------
BFS GTS (heavy-vehicle goods transport survey), ts-x-11.05-GTS-E26.csv
    Million tkm on Swiss roads by vehicle type (LORRY, HAV), registration and traffic
    type, 1993-2024. Read on a territorial basis (all registrations, all traffic).
BFS road vehicle stock and new registrations (stats.swiss SDMX CH1.MFZ_IVS)
    DF_MFZ_0_GENERAL (stock) and DF_IVS_0_GENERAL (first registrations of new
    vehicles), by vehicle type and fuel, 2005+. Cached as pickles in data/.
EP2050+ Verkehrssektor ZERO-Basis (sheets 02 Flottenbestand, 03 Fahrleistung,
    04 Energieverbrauch Strasse): fleet by segment, vkm and energy by technology,
    1990-2060.
FOEN GHG inventory, Evolution_GHG_since_1990_2025-04.xlsx
    CO2 of heavy and light duty trucks (Mt), 1990-2023.
EP2050+ Technical report Kap. 8, Abb. 112 (cached tra_EP2050_energy_demand_rail.pickle)
    Freight rail energy (TWh), 1990-2060.
"""

import os
import pickle

import numpy as np
import pandas as pd

from transition_compass_model._database.pre_processing.api_routines_swiss_stats import (
    get_data_api_swiss_stats,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_THIS_DIR, "../data")

ROAD_MODES = ["HDVH", "HDVL", "HDVM"]

# BFS vehicle type -> model road class (body type, see freight_efficiency_tech_share).
# HDVH = tractor units of articulated trucks: BFS "Road tractor" (Sattelschlepper,
# ~12 000) matches EP2050 TT/AT; BFS "Articulated vehicle" is a tiny legacy category
# (~100) added to it.
_BFS_VEH_TO_MODE = {
    "Road tractor": "HDVH",
    "Articulated vehicle": "HDVH",
    "Goods truck (lorry)": "HDVM",
    "Delivery vehicle": "HDVL",
}

# BFS fuel -> model technology. "Other" (< 1 % of goods vehicles) is spread
# proportionally over the known technologies; "No motor" is not a vehicle we count.
_BFS_FUEL_TO_TECH = {
    "Petrol: conventional": "ICE-gasoline",
    "Petrol: hybrid electric vehicle (HEV)": "ICE-gasoline",
    "Diesel: conventional": "ICE-diesel",
    "Diesel: hybrid electric vehicle (HEV)": "ICE-diesel",
    "Plug-in hybrid electric vehicle (PHEV): petrol": "PHEV-gasoline",
    "Plug-in hybrid electric vehicle (PHEV): diesel": "PHEV-diesel",
    "Battery electric vehicle (BEV)": "BEV",
    "Fuel cell electric vehicle (FCEV)": "FCEV",
    "Gas (mono- and bi-fuel)": "ICE-gas",
}


# ---------------------------------------------------------------------------
# tkm (BFS GTS)
# ---------------------------------------------------------------------------
def read_gts_road_tkm() -> dict:
    """Return {mode: pd.Series(year -> million tkm)} for HDVH (HAV) and HDVM (LORRY).

    Territorial basis: all registrations (CH and foreign), all traffic types
    (domestic, import, export, transit), i.e. all heavy-vehicle tkm on Swiss roads.
    """
    path = os.path.join(_DATA_DIR, "Freight/ts-x-11.05-GTS-E26.csv")
    df = pd.read_csv(path, sep=";")
    df = df[(df["IMMATRICULATION"] == "_T") & (df["TRAFFIC_TYPE"] == "_T")]
    df = df[df["VEH_TYPE"].isin(["LORRY", "HAV"])]
    piv = df.pivot(index="REF_YEAR", columns="VEH_TYPE", values="OBS_VALUE")
    piv.index = piv.index.astype(int)
    return {"HDVH": piv["HAV"].astype(float), "HDVM": piv["LORRY"].astype(float)}


# ---------------------------------------------------------------------------
# Fleet and new registrations (BFS, stats.swiss)
# ---------------------------------------------------------------------------
def _fetch_bfs_goods_vehicles(dataflow: str, cache_file: str) -> pd.DataFrame:
    """Download (or load from cache) goods-vehicle counts by type and fuel for CH.

    Returns a long DataFrame with columns [year, mode, tech, value].
    """
    cache = os.path.join(_DATA_DIR, cache_file)
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)

    agency = "CH1.MFZ_IVS"
    structure, _ = get_data_api_swiss_stats(agency, dataflow, mode="example")
    filtering = {
        "UV_HGDE_KT": ["Total"],
        "TIME_PERIOD": structure["TIME_PERIOD"],
        "UV_RV_VEHICLE_GROUP_AND_TYPE": list(_BFS_VEH_TO_MODE),
        "UV_RV_FUEL": [f for f in structure["UV_RV_FUEL"] if f != "No motor"],
        "UV_RV_OWNER_TYPE": ["Total"],
    }
    if "UV_RV_VEHICLE_AGE" in structure:
        filtering["UV_RV_VEHICLE_AGE"] = ["Total"]
        var_dim = "UV_RV_VEHICLE_AGE"
    else:
        filtering["UV_RV_REGISTRATION_TYPE"] = ["First registrations of new vehicles"]
        filtering["FREQ"] = ["Annual"]
        var_dim = "UV_RV_REGISTRATION_TYPE"
    dm = get_data_api_swiss_stats(
        agency,
        dataflow,
        mode="extract",
        filter=filtering,
        mapping_dims={
            "Country": "UV_HGDE_KT",
            "Years": "TIME_PERIOD",
            "Variables": var_dim,
            "Categories1": "UV_RV_VEHICLE_GROUP_AND_TYPE",
            "Categories2": "UV_RV_FUEL",
        },
        units=["number"],
    )
    if dm is None:
        raise ValueError(f"API returned None for {agency},{dataflow}")

    rows = []
    idx = dm.idx
    for veh in dm.col_labels["Categories1"]:
        for fuel in dm.col_labels["Categories2"]:
            for yr in dm.col_labels["Years"]:
                rows.append(
                    {
                        "year": int(yr),
                        "veh": veh,
                        "fuel": fuel,
                        "value": float(dm.array[0, idx[yr], 0, idx[veh], idx[fuel]]),
                    }
                )
    df = pd.DataFrame(rows)
    df["fuel"] = df["fuel"].replace({"Total_Categories2": "Total"})

    # Guard: the fuels must add up to BFS's own total for every vehicle type and year
    # (a first download once returned shuffled fuel values). BFS itself has small gaps
    # (road tractors 2023+: 1-4 %), so only large gaps stop the run.
    piv = df.pivot_table(index=["veh", "year"], columns="fuel", values="value")
    by_fuel = piv.drop(columns="Total").sum(axis=1)
    gap = (by_fuel - piv["Total"]).abs() / piv["Total"].where(piv["Total"] > 0)
    if (gap > 0.05).any():
        raise ValueError(
            f"BFS {dataflow}: fuels do not add up to the total\n{gap[gap > 0.05]}"
        )
    if (gap > 0.01).any():
        print(
            f"BFS {dataflow}: fuel sum differs from total by >1 %:\n{gap[gap > 0.01]}"
        )
    df = df[df["fuel"] != "Total"]

    with open(cache, "wb") as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    return df


def _bfs_to_mode_tech(df: pd.DataFrame) -> dict:
    """Map BFS vehicle types and fuels to {mode: DataFrame(year x tech)}.

    "Other" fuel is distributed proportionally over the mapped technologies.
    """
    df = df.copy()
    df["value"] = df["value"].fillna(0.0)
    df["mode"] = df["veh"].map(_BFS_VEH_TO_MODE)
    df = df[df["mode"].notnull()]
    out = {}
    for mode in ROAD_MODES:
        d = df[df["mode"] == mode]
        known = d[d["fuel"].isin(_BFS_FUEL_TO_TECH)].copy()
        known["tech"] = known["fuel"].map(_BFS_FUEL_TO_TECH)
        tab = known.pivot_table(
            index="year", columns="tech", values="value", aggfunc="sum"
        ).fillna(0.0)
        other = d[d["fuel"] == "Other"].groupby("year")["value"].sum()
        other = other.reindex(tab.index).fillna(0.0)
        share = tab.div(tab.sum(axis=1).where(tab.sum(axis=1) > 0), axis=0)
        out[mode] = tab + share.mul(other, axis=0).fillna(0.0)
    return out


def read_bfs_goods_fleet() -> dict:
    """Return {mode: DataFrame(year x tech)} of goods-vehicle stock (number), 2005+."""
    df = _fetch_bfs_goods_vehicles("DF_MFZ_0_GENERAL", "tra_freight_fleet_bfs.pickle")
    return _bfs_to_mode_tech(df)


def read_bfs_goods_new() -> dict:
    """Return {mode: DataFrame(year x tech)} of first registrations of new goods
    vehicles (number), 2005+."""
    df = _fetch_bfs_goods_vehicles("DF_IVS_0_GENERAL", "tra_freight_new_bfs.pickle")
    return _bfs_to_mode_tech(df)


# ---------------------------------------------------------------------------
# EP2050+ (fleet by segment, vkm and energy by technology)
# ---------------------------------------------------------------------------
def read_ep2050_fleet_by_tech(seg_maps: dict) -> dict:
    """Return {mode: DataFrame(year x tech)} of fleet counts from sheet 02, ZERO-Basis.

    seg_maps: {mode: {EP2050 segment: model tech}} (the _HDV*_SEG dicts).
    Segments are matched exactly, so every truck segment (including "RT petrol")
    is counted.
    """
    from processors.freight_efficiency_tech_share import _EP2050_PATH

    df = pd.read_excel(_EP2050_PATH, sheet_name="02 Flottenbestand", header=19)
    df = df.iloc[:, 1:]
    # ZERO-Basis is the first sub-table: stop at the next scenario label
    first_col = pd.read_excel(
        _EP2050_PATH, sheet_name="02 Flottenbestand", header=19, usecols=[0]
    ).iloc[:, 0]
    stop = first_col[first_col.astype(str).str.startswith("ZERO")].index
    if len(stop) > 0:
        df = df.loc[: stop[0] - 1]
    year_cols = [c for c in df.columns if isinstance(c, (int, float))]
    out = {}
    for mode, seg_map in seg_maps.items():
        d = df[df["Segment"].isin(seg_map)].copy()
        d["tech"] = d["Segment"].map(seg_map)
        tab = d.groupby("tech")[year_cols].sum().T
        tab.index = tab.index.astype(int)
        out[mode] = tab.astype(float)
    return out


def read_ep2050_vkm_by_tech() -> dict:
    """Return {"LCV"|"HGV": DataFrame(year x tech)} in million vkm (sheet 03)."""
    from processors.freight_efficiency_tech_share import _read_ep2050_vkm

    df = _read_ep2050_vkm()
    out = {}
    for cat in ["LCV", "HGV"]:
        d = df[df["VehCat"] == cat].set_index("Technology")
        year_cols = [c for c in d.columns if isinstance(c, (int, float))]
        tab = d[year_cols].T
        tab.index = tab.index.astype(int)
        out[cat] = tab.astype(float)
    return out


def read_ep2050_energy_by_tech() -> dict:
    """Return {"LCV"|"HGV": DataFrame(year x tech)} in PJ (sheet 04)."""
    from processors.freight_efficiency_tech_share import _read_ep2050_energy

    df = _read_ep2050_energy()
    out = {}
    for cat in ["LCV", "HGV"]:
        d = df[df["Fahrzeugart"] == cat]
        year_cols = [c for c in d.columns if isinstance(c, (int, float))]
        tab = d.groupby("Treibstoff")[year_cols].sum().T
        tab.index = tab.index.astype(int)
        out[cat] = tab.astype(float)
    return out


# ---------------------------------------------------------------------------
# FOEN GHG inventory
# ---------------------------------------------------------------------------
def read_foen_truck_co2() -> dict:
    """Return {"heavy"|"light": pd.Series(year -> Mt CO2)} from the FOEN inventory."""
    path = os.path.join(_DATA_DIR, "Evolution_GHG_since_1990_2025-04.xlsx")
    df = pd.read_excel(path, sheet_name="CO2", header=None)
    # Header row is the one labelled "Cat. (1)"
    header_idx = df.index[df.iloc[:, 0].astype(str).str.strip() == "Cat. (1)"][0]
    years_row = df.iloc[header_idx]
    year_cols = {
        j: int(v)
        for j, v in years_row.items()
        if isinstance(v, (int, float)) and not pd.isna(v) and 1900 < v < 2100
    }
    out = {}
    for key, label in [("heavy", "Heavy duty trucks"), ("light", "Light duty trucks")]:
        mask = df.apply(
            lambda r: any(isinstance(v, str) and v.strip() == label for v in r), axis=1
        )
        row = df[mask].iloc[0]
        out[key] = pd.Series({yr: float(row[j]) for j, yr in year_cols.items()})
    return out


# ---------------------------------------------------------------------------
# Rail freight energy (EP2050+ Abb. 112)
# ---------------------------------------------------------------------------
def read_ep2050_rail_freight_energy() -> pd.Series:
    """Return pd.Series(year -> PJ) of freight rail energy (EP2050+ Abb. 112).

    Uses the cached pickle written by get_data_ep2050 (passenger calibration step).
    """
    path = os.path.join(_DATA_DIR, "tra_EP2050_energy_demand_rail.pickle")
    with open(path, "rb") as f:
        dm = pickle.load(f)
    var = "tra_energy_demand_freight_rail"
    if dm.units[var] != "PJ":
        raise ValueError(f"Unexpected unit {dm.units[var]} for {var}")
    vals = dm.array[dm.idx["Switzerland"], :, dm.idx[var]]
    return pd.Series(dict(zip(dm.col_labels["Years"], vals))).astype(float)


def fill_years(raw: pd.Series, years: list) -> np.ndarray:
    """Reindex to years, interpolate interior gaps, hold ends flat."""
    s = raw.reindex(sorted(set(raw.index) | set(years))).astype(float)
    s = s.interpolate(method="index").bfill().ffill()
    return s.reindex(years).values
