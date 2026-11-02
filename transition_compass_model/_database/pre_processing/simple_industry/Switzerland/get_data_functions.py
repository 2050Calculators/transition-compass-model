"""Readers for the raw data used by simple_industry and simple_waste (Switzerland).

None of these functions call a stat-tab (pxweb.bfs.admin.ch) API: they read files on
disk (downloaded once from BFE / BAFU) and cache parsed results next to the raw file.
The stat-tab calls live in simple_industry_preprocessing_CH.py and are flagged there.
"""

import os
import re

import numpy as np
import pandas as pd
import requests

import transition_compass_model
from transition_compass_model._database.pre_processing.constants import CANTONS_NAME

ROOT_DIR = os.path.dirname(os.path.abspath(transition_compass_model.__file__))
PRE_DIR = os.path.join(ROOT_DIR, "_database/pre_processing")
DATA_DIR = os.path.join(PRE_DIR, "simple_industry/Switzerland/data")
POWER_DATA_DIR = os.path.join(PRE_DIR, "power/Switzerland/data")

# Raw files and where to get them again
GEST_FILE = os.path.join(POWER_DATA_DIR, "statistique_globale_suisse_energie.xlsx")
EP2050_INDUSTRY_FILE = os.path.join(
    POWER_DATA_DIR,
    "EP2050_sectors/EP2050+_Szenarienergebnisse_Details_Nachfragesektoren/"
    "EP2050+_Detailergebnisse 2020-2060_Industriessektor_alle Szenarien_2022-04-12.xlsx",
)
FOEN_GHG_FILE = os.path.join(
    PRE_DIR, "industry/Switzerland/data/emissions/Evolution_GHG_since_1990_2026-04.xlsx"
)
BFE_FILE = os.path.join(DATA_DIR, "bfe/bfe_industry_services_energy_2013-2025.xlsx")
BFE_URL = "https://pubdb.bfe.admin.ch/de/publication/download/8788"
PRTR_FILE = os.path.join(DATA_DIR, "prtr/swissprtr-daten-2007-2024.xlsx")
PRTR_URL = "https://www.bafu.admin.ch/dam/en/sd-web/CXgeZKcf8ez8/swissprtr-daten-2007-2024.xlsx"

# The 13 industry branches. The first 12 follow the BFE industry statistics
# (BranchenNr. 1-12, same split as EP2050+); water-waste is NACE 36-39 own energy use
# (waste incineration plants are in simple_waste).
BRANCHES = [
    "food",
    "textiles",
    "paper",
    "chemicals",
    "cement",
    "glass-ceramics",
    "iron-steel",
    "non-ferrous-metals",
    "metal-products",
    "machinery",
    "other-industries",
    "construction",
    "water-waste",
]

# EP2050+ "Branche" labels -> branch
EP2050_BRANCHES = {
    "Nahrung, Tabak": "food",
    "Textilien": "textiles",
    "Papier, Druck": "paper",
    "Chemie, Pharma": "chemicals",
    "Zement, Kalk, Ziegel": "cement",
    "Glas, Keramik, Beton, Steine": "glass-ceramics",
    "Eisen, Stahl": "iron-steel",
    "NE-Metalle": "non-ferrous-metals",
    "Metallerzeugnisse, Geräte": "metal-products",
    "Maschinenbau, Fahrzeugbau": "machinery",
    "Übrige Branchen": "other-industries",
    "Wasserversorgung, Abfallwirtschaft": "water-waste",
    "Bau": "construction",
}

# BFE fuel sheets -> fuel. "Fernwärme (Abgabe)" is heat delivered by firms, not used.
BFE_FUEL_SHEETS = {
    "electricity": "Elektrizität",
    "gas": "Erdgas",
    "heating-oil": "Heizöl extra-leicht",
    "other-oil": "Heizöl mittel und schwer",
    "waste": "Industrieabfälle",
    "coal": "Kohle",
    "district-heating": "Fernwärme (Bezug)",
    "wood": "Holz",
}

CANTON_CODES = {
    k: ("Neuchâtel" if v == "Neuchatel" else v)
    for k, v in CANTONS_NAME["accronym_to_name"].items()
    if k not in ("CH", "FB")
}


def _download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {url} -> {path}")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------------------
# GEST (Gesamtenergiestatistik, BFE) — file on disk
# --------------------------------------------------------------------------------------
def read_gest_industry():
    """Industry final energy by fuel, GEST table T17b, in TWh. Index = year."""
    df = pd.read_excel(GEST_FILE, sheet_name="T17b", header=None)
    cols = {
        "oil-total": 1,
        "heating-oil": 3,
        "electricity": 5,
        "gas": 7,
        "coal": 9,
        "wood": 11,
        "district-heating": 13,
        "waste": 15,
        "renewables": 17,
        "total": 19,
    }
    rows = {}
    for r in df.values:
        if str(r[0]).strip().isdigit():
            rows[int(r[0])] = {k: _num(r[i]) / 3600 for k, i in cols.items()}
    out = pd.DataFrame(rows).T.sort_index()
    out["other-oil"] = out["oil-total"] - out["heating-oil"]
    return out.drop(columns="oil-total")


def read_gest_kva():
    """Waste incineration plants (KVA), GEST table T27, in TWh. Index = year."""
    df = pd.read_excel(GEST_FILE, sheet_name="T27", header=None)
    cols = {
        "waste-input": 3,
        "fossil-input": 4,
        "heat-production": 5,
        "electricity-production": 6,
        "heat-own-use": 7,
        "electricity-own-use": 8,
        "heat-sold": 9,
        "electricity-sold": 10,
    }
    rows = {}
    for r in df.values:
        if str(r[0]).strip().isdigit():
            rows[int(r[0])] = {k: _num(r[i]) / 1000 for k, i in cols.items()}
    return pd.DataFrame(rows).T.sort_index()


# --------------------------------------------------------------------------------------
# FOEN GHG inventory summary — file on disk
# --------------------------------------------------------------------------------------
def read_foen_inventory(sheet="CO2"):
    """Return {row label: pd.Series(year -> Mt)} for the FOEN summary table."""
    df = pd.read_excel(FOEN_GHG_FILE, sheet_name=sheet, header=None)
    hdr = next(i for i in df.index if str(df.iloc[i, 0]).startswith("Cat."))
    years = {j: int(float(v)) for j, v in enumerate(df.iloc[hdr]) if _is_year(v)}
    out = {}
    for i in range(hdr + 1, len(df)):
        label = " ".join(
            str(v).strip() for v in df.iloc[i, : min(years)] if str(v) != "nan"
        )
        if not label:
            continue
        out[label] = pd.Series({y: _num(df.iloc[i, j]) for j, y in years.items()})
    return out


def _is_year(v):
    try:
        return 1980 <= float(v) <= 2100
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------------------
# EP2050+ industry workbook — file on disk
# --------------------------------------------------------------------------------------
def read_ep2050_table(sheet, table_no):
    """Read table 'Tabelle XX-<table_no>' of an EP2050+ industry sheet.

    Returns a DataFrame indexed by the last text label column (Verwendungszweck or
    Branche) with integer year columns.
    """
    df = pd.read_excel(EP2050_INDUSTRY_FILE, sheet_name=sheet, header=None)
    pattern = re.compile(rf"Tabelle \d\d-{table_no:02d}")
    title = next(
        i
        for i in df.index
        if any(pattern.match(str(v)) for v in df.iloc[i].values if isinstance(v, str))
    )
    hdr = next(
        i for i in range(title, title + 6) if sum(_is_year(v) for v in df.iloc[i]) > 10
    )
    year_cols = {j: int(float(v)) for j, v in enumerate(df.iloc[hdr]) if _is_year(v)}
    label_col = max(j for j in range(min(year_cols)) if str(df.iloc[hdr, j]) != "nan")
    rows = {}
    for i in range(hdr + 1, len(df)):
        label = df.iloc[i, label_col]
        if str(label) == "nan":
            break
        rows[str(label).strip()] = {
            y: _num(df.iloc[i, j]) for j, y in year_cols.items()
        }
    return pd.DataFrame(rows).T


def read_ep2050_branch_table(sheet, table_no):
    """Branch table (02-03 combustion GHG, 02-04 process GHG) as branch x year, Mt CO2eq."""
    df = read_ep2050_table(sheet, table_no)
    df = df.loc[[r for r in df.index if r in EP2050_BRANCHES]]
    df.index = [EP2050_BRANCHES[r] for r in df.index]
    return df.reindex(BRANCHES).fillna(0.0)


# --------------------------------------------------------------------------------------
# BFE "Energieverbrauch in der Industrie und im Dienstleistungssektor" — download
# --------------------------------------------------------------------------------------
def read_bfe_branch_fuel():
    """Industry energy by BFE branch (1-12) and fuel, TWh.

    Returns {fuel: DataFrame(branch x year)} for the 12 industry branches.
    """
    if not os.path.exists(BFE_FILE):
        _download(BFE_URL, BFE_FILE)
    out = {}
    for fuel, sheet in BFE_FUEL_SHEETS.items():
        df = pd.read_excel(BFE_FILE, sheet_name=sheet, header=3)
        df = df[pd.to_numeric(df["BranchenNr."], errors="coerce").between(1, 12)]
        df.index = [BRANCHES[int(n) - 1] for n in df["BranchenNr."]]
        years = [c for c in df.columns if _is_year(c)]
        df = df[years].apply(pd.to_numeric, errors="coerce").fillna(0.0) / 3600
        df.columns = [int(float(c)) for c in years]
        out[fuel] = df
    return out


# --------------------------------------------------------------------------------------
# SwissPRTR — download
# --------------------------------------------------------------------------------------
def prtr_branch(noga):
    """Map a 6-digit NOGA code of a PRTR facility to a branch (or kva / energy-sector)."""
    n2, n4 = noga // 10000, noga // 100
    if noga == 382100:
        return "kva"  # municipal waste incineration -> simple_waste
    if noga == 382200:
        # Hazardous-waste incinerators run inside chemical parks (CIMO Monthey,
        # GETEC Basel). Their fuel is the 'Industrieabfälle' of the chemical branch.
        return "chemicals"
    if n4 in (2332, 2351, 2352):
        return "cement"  # BFE 'Zement und Beton': 23.32, 23.51, 23.52
    if n2 == 23:
        return "glass-ceramics"
    if n4 in (2441, 2442, 2443, 2444, 2445, 2446, 2453, 2454):
        return "non-ferrous-metals"
    if n2 == 24:
        return "iron-steel"
    if n4 in (2540, 2751, 2752) or n2 == 28:
        return "machinery"
    if n2 in (25, 26, 27):
        return "metal-products"
    groups = {
        "food": (10, 11, 12),
        "textiles": (13, 14, 15),
        "paper": (17, 18),
        "chemicals": (20, 21),
        "other-industries": (5, 6, 7, 8, 9, 16, 22, 29, 30, 31, 32),
        "construction": (41, 42, 43),
        "water-waste": (36, 37, 38, 39),
    }
    for branch, codes in groups.items():
        if n2 in codes:
            return branch
    if n2 in (19, 35, 49):
        return "energy-sector"  # refinery, power/heat plants, pipelines
    return "other-sector"


def read_prtr_co2():
    """CO2 releases to air of PRTR point sources, one row per facility and year (Mt).

    PRTR CO2 is total CO2 (fossil + biogenic), process emissions included.
    """
    cache = os.path.splitext(PRTR_FILE)[0] + "_co2.csv"
    if os.path.exists(cache):
        return pd.read_csv(cache)
    if not os.path.exists(PRTR_FILE):
        _download(PRTR_URL, PRTR_FILE)
    df = pd.read_excel(PRTR_FILE, sheet_name=0, header=0)
    df = df[
        pd.to_numeric(df["Year"], errors="coerce").notna()
    ]  # drop FR/IT/DE header rows
    df = df[
        (df["Pollutant_name"] == "Kohlendioxid (CO2)")
        & (df["Source type"] == "Punktquelle")
        & (df["Medium"] == "Luft")
    ]
    out = pd.DataFrame(
        {
            "year": df["Year"].astype(int),
            "canton": df["Canton_facility"].map(CANTON_CODES),
            "facility": df["Facility"],
            "facility_id": df["Facility ID"],
            "noga": pd.to_numeric(df["NOGA code"], errors="coerce")
            .fillna(0)
            .astype(int),
            "activity": df["Installation_main activity"].astype(str),
            "co2_Mt": pd.to_numeric(df["Value"], errors="coerce") / 1e9,  # kg -> Mt
        }
    )
    if out["canton"].isna().any():
        raise ValueError(
            f"Unknown canton codes in PRTR: {df.loc[out['canton'].isna(), 'Canton_facility'].unique()}"
        )
    out["branch"] = out["noga"].apply(prtr_branch)
    out.to_csv(cache, index=False)
    return out


def prtr_by_canton_branch(df_prtr, cantons, years, branches):
    """PRTR CO2 as array [canton, year, branch] (Mt); NaN for years without PRTR data."""
    arr = np.full((len(cantons), len(years), len(branches)), np.nan)
    prtr_years = set(df_prtr["year"])
    g = df_prtr.groupby(["canton", "year", "branch"])["co2_Mt"].sum()
    for iy, y in enumerate(years):
        if y not in prtr_years:
            continue
        arr[:, iy, :] = 0.0
        for ic, c in enumerate(cantons):
            for ib, b in enumerate(branches):
                arr[ic, iy, ib] = g.get((c, y, b), 0.0)
    return arr
