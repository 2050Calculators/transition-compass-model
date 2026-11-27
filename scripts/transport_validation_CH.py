"""Compare the Swiss transport run with official energy and emission statistics.

Runs climate -> lifestyles -> transport for Switzerland and compares the historical
years (1990-2023) with official series:
  1. Energy by fuel: SFOE overall energy statistics (Gesamtenergiestatistik), table
     T17e, transport final energy by fuel [TWh/yr]. Passenger and freight together.
     The model is shown split into passenger and freight.
  2. Emissions by mode: FOEN greenhouse-gas inventory, CO2 + CH4 + N2O by mode
     [Mt CO2e/yr]. CH4 and N2O are converted with the model's factors (CH4 x 28,
     N2O x 265).
  3. Energy by mode: EP2050+ detailed results (road by vehicle type, rail, aviation)
     [TWh/yr], passenger and freight apart, 1990-2023. Note: for several modes the
     model efficiency was built as EP2050+ energy / vkm, so this mostly checks the
     model's vkm and technology split, not the energy on its own. Buses are an
     exception: their efficiency is FSO energy / FSO vkm, so this is an
     independent check for them.
  4. Passenger-km by mode: FSO "prestations de transport" (je-f-11.04.01.02)
     [Bpkm/yr]: cars, motorbikes, buses (with coaches), tram, rail, bike, walk.
  5. Vehicle-km by mode [Bvkm/yr]: FSO "prestations kilométriques"
     (je-f-11.04.01.01) for cars, motorbikes, buses, tram and rail (train-km), and
     EP2050+ road vkm (sheet "03 Fahrleistung", WWB) for cars, motorbikes, buses.
     Note: the model's Swiss pkm and vkm for 1990-2023 are built from these FSO
     tables, so checks 4-5 mostly confirm that the historical data passes through
     the model unchanged. EP2050+ vkm is an independent source.
  6. Energy by mode and fuel: same EP2050+ data as check 3, but split by fuel
     group (gasoline, diesel, gas, electric, hydrogen) [TWh/yr]. A right total with
     a wrong technology split shows up here.
  7. Electricity, road vs rail: SFOE T17e split of transport electricity
     [TWh/yr].

Official data:
  - SFOE energy, EP2050+ energy and FOEN emissions are built by this script and
    cached in scripts/data/ (git-ignored) as tra_energy_SFOE.pickle,
    tra_energy_EP2050.pickle and tra_emissions_FOEN.pickle. They are built on the
    first run, or again with --rebuild. Sources:
      SFOE   statistique_globale_suisse_energie.xlsx, downloaded to scripts/data/
             if missing (the buildings validation uses the same file);
      EP2050+ the cached extracts in _database/pre_processing/transport/
             Switzerland/data/ (tra_EP2050_energy_demand_private.pickle,
             tra_EP2050_energy_demand_rail.pickle), downloaded and extracted from
             the EP2050+ zips if missing;
      FOEN   Evolution_GHG_since_1990_2025-04.xlsx in the same data/ folder.
  - EP2050+ road vkm is read from calibration_vkm.pickle, which the transport
    pre-processing writes in .../Switzerland/data/datamatrix/ (it also uses it).
  - FSO pkm and vkm are read from the raw FSO Excel files in .../Switzerland/data/
    (tra_pkm_CH.xlsx, tra_vkm_CH.xlsx), with no gap filling: years FSO leaves
    empty stay empty.
The model values come from the transport TPE output, except checks 6-7, which
need energy by technology: run_model() takes it from the transport workflow
functions.

Prints one table per series (official, model, model/official) and writes an
interactive plotly page.

Usage (from the repo root):
    uv run python scripts/transport_validation_CH.py [--out DIR] [--rebuild]
        (default DIR: scripts/output/; --rebuild rebuilds the cached official data)
"""

import argparse
import math
import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from transition_compass_model._database.pre_processing.buildings.Switzerland.get_data_functions.energy_demand_for_calibration import (
    extract_energy_statistics_data,
)
from transition_compass_model._database.pre_processing.transport.Switzerland.get_data_functions.transport_energy import (
    extract_EP2050_transport_energy_demand,
    extract_EP2050_transport_energy_demand_rail,
)
from transition_compass_model.model.climate_module import climate
from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    filter_country_and_load_data_from_pickles,
)
from transition_compass_model.model.common.config_loader import load_lever_config
from transition_compass_model.model.common.data_matrix_class import DataMatrix
from transition_compass_model.model.common.interface_class import Interface
from transition_compass_model.model.lifestyles_module import lifestyles
from transition_compass_model.model.transport_module import transport
from transition_compass_model.model.transport_module import wkf as transport_wkf

COUNTRY = "Switzerland"
YEARS_SETTING = [1990, 2023, 2025, 2050, 5]
YEARS = create_years_list(YEARS_SETTING[0], YEARS_SETTING[1], 1)
KEY_YEARS = [1990, 2000, 2010, 2019, 2023]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")  # HTML plots (git-ignored)
CALIB_DIR = os.path.join(
    HERE,
    "../transition_compass_model/_database/pre_processing/transport/Switzerland",
    "data/datamatrix",
)
VKM_FILE = os.path.join(CALIB_DIR, "calibration_vkm.pickle")
VKM_KEY = "passenger-and-freight_road_EP-2050"
SOURCE_DIR = os.path.join(CALIB_DIR, "..")  # pre-processing raw data
FSO_PKM_FILE = os.path.join(SOURCE_DIR, "tra_pkm_CH.xlsx")
FSO_VKM_FILE = os.path.join(SOURCE_DIR, "tra_vkm_CH.xlsx")

# --- Official data built and cached by this script -----------------------------
DATA_DIR = os.path.join(HERE, "data")  # cached official files (git-ignored)
SFOE_FILE = os.path.join(DATA_DIR, "tra_energy_SFOE.pickle")
EP2050_FILE = os.path.join(DATA_DIR, "tra_energy_EP2050.pickle")
FOEN_FILE = os.path.join(DATA_DIR, "tra_emissions_FOEN.pickle")

# SFOE overall energy statistics, table T17e (transport). Source page:
# https://www.bfe.admin.ch/bfe/fr/home/versorgung/statistik-und-geodaten/energiestatistiken/gesamtenergiestatistik.html
SFOE_URL = "https://pubdb.bfe.admin.ch/fr/publication/download/7519"
SFOE_XLSX = os.path.join(DATA_DIR, "statistique_globale_suisse_energie.xlsx")
# Column header (regex) -> name. Sub-columns, not the overall "Total Treibstoffe".
# "Autres énergies renouvelables" = sun, wind, biogas, ambient heat (SFOE footnote).
SFOE_COLUMNS = {
    "gasoline": r".*Essence.*",
    "diesel": r".*Carburant diesel.*",
    "kerosene": r".*Carburants d'aviation.*",
    "electricity": r".*Total Electricité.*",
    "electricity-rail": r".*chemins de fer.*",
    "electricity-road": r".*routier.*",
    "electricity-non-road": r".*non routier.*",
    "gas-pipeline": r".*Gaz transport par conduites.*",
    "gas-other": r".*Gaz autres transport.*",
    "coal": r".*Charbon.*",
    "renewables-other": r".*Autres énergies renouvelables.*",
}
SFOE_BY_FUEL = [
    "diesel",
    "electricity",
    "gas",
    "gasoline",
    "kerosene",
    "renewables-other",
]
SFOE_ELEC = ["electricity-non-road", "electricity-rail", "electricity-road"]

# EP2050+ sources (cached extracts in the pre-processing data folder)
# Road: "EP2050+_Detailergebnisse 2020-2060_Verkehrssektor_alle Szenarien", WWB
EP2050_ROAD_URL = "https://www.bfe.admin.ch/bfe/de/home/politik/energieperspektiven-2050-plus.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvMTA0NDE=.html"
EP2050_ROAD_ZIP = os.path.join(SOURCE_DIR, "EP2050_sectors.zip")
EP2050_ROAD_PICKLE = os.path.join(SOURCE_DIR, "tra_EP2050_energy_demand_private.pickle")
# Rail and aviation: technical report Kap. 8, Abb. 112
EP2050_RAIL_URL = "https://www.bfe.admin.ch/bfe/de/home/politik/energieperspektiven-2050-plus.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvMTA5MDQ=.html"
EP2050_RAIL_ZIP = os.path.join(SOURCE_DIR, "EP2050_tables.zip")
EP2050_RAIL_PICKLE = os.path.join(SOURCE_DIR, "tra_EP2050_energy_demand_rail.pickle")

# FOEN greenhouse-gas inventory. CO2 is in Mt, CH4 and N2O in kt.
FOEN_XLSX = os.path.join(SOURCE_DIR, "Evolution_GHG_since_1990_2025-04.xlsx")
FOEN_TO_MT = {"CO2": 1, "CH4": 1e-3, "N2O": 1e-3}
FOEN_ROWS = list(range(12, 22)) + [57, 58]  # transport rows of each gas sheet
FOEN_MODES = {
    "Buses": "bus",
    "Cars": "LDV",
    "Heavy duty trucks": "trucks",
    "Light duty trucks": "trucks",
    "Motorcycles": "2W",
    "Domestic aviation (without military)": "aviation",
    "International aviation": "aviation",
    "Domestic navigation": "marine",
    "International navigation": "marine",
    "Railways": "rail",
}

# Global warming potentials, same as model/transport/workflows.py
# convert_to_cO2eq_emissions.
GWP = {"CO2": 1, "CH4": 28, "N2O": 265}

# --- Model fuel families -> SFOE fuel groups ----------------------------------
# SFOE "renewables-other" = biofuels, biogas, sun, wind, ambient heat. The model has
# no marine-fuel-oil use in Switzerland (no sea access), SFOE has no such group.
MODEL_FUEL_TO_GROUP = {
    "diesel": "diesel",
    "gasoline": "gasoline",
    "gas": "gas",
    "kerosene": "kerosene",
    "electricity": "electricity",
    "biofuels": "renewables-other",
    "efuels": "renewables-other",
    "hydrogen": "renewables-other",
    "marine-fuel-oil": "marine-fuel-oil",
}
FUEL_ORDER = [
    "diesel",
    "gasoline",
    "gas",
    "kerosene",
    "electricity",
    "renewables-other",
    "marine-fuel-oil",
    "total",
]

# --- Model modes -> FOEN modes -------------------------------------------------
# TPE variable tra_emissions-CO2e_<kind>_<mode>[Mt]
MODEL_MODE_TO_FOEN = {
    ("passenger", "LDV"): "LDV",
    ("passenger", "2W"): "2W",
    ("passenger", "bus"): "bus",
    ("freight", "HDV"): "trucks",
    ("passenger", "rail"): "rail",
    ("passenger", "metrotram"): "rail",
    ("freight", "rail"): "rail",
    ("passenger", "aviation"): "aviation",
    ("freight", "aviation"): "aviation",
    ("freight", "IWW"): "marine",
    ("freight", "marine"): "marine",
}
MODE_ORDER = ["LDV", "2W", "bus", "trucks", "rail", "aviation", "marine", "total"]
MODE_TITLES = {
    "LDV": "cars",
    "2W": "motorbikes",
    "bus": "buses",
    "trucks": "trucks and vans",
    "rail": "rail",
    "aviation": "aviation",
    "marine": "navigation",
    "total": "total",
}

# --- EP2050+ energy by mode ---------------------------------------------------
# (key in EP2050_FILE, EP2050+ mode) -> comparison column
EP2050_TO_COLUMN = {
    ("passenger", "LDV"): "passenger cars",
    ("passenger", "2W"): "passenger motorbikes",
    ("passenger", "bus"): "passenger buses",
    ("passenger", "rail"): "passenger rail",
    ("freight", "HDVH"): "freight trucks",
    ("freight", "HDVL"): "freight trucks",
    ("freight", "rail"): "freight rail",
    ("aviation", "aviation"): "aviation",
}
# TPE variable tra_<kind>_energy-demand-by-mode_<mode>[TWh]. Metro/tram is left
# out: EP2050+ rail is trains only. Model trucks include rigid trucks (HDVM), which
# EP2050+ does not have as a separate category.
MODEL_TO_EP2050_COLUMN = {
    ("passenger", "LDV"): "passenger cars",
    ("passenger", "2W"): "passenger motorbikes",
    ("passenger", "bus"): "passenger buses",
    ("passenger", "rail"): "passenger rail",
    ("freight", "HDV"): "freight trucks",
    ("freight", "rail"): "freight rail",
    ("passenger", "aviation"): "aviation",
    ("freight", "aviation"): "aviation",
}
EP2050_ORDER = list(dict.fromkeys(EP2050_TO_COLUMN.values())) + ["total"]
MJ_PER_TWH = 3.6e9

# --- EP2050+ energy by mode and technology ----------------------------------------
# (kind, mode) -> panel name, same for model and EP2050+. EP2050+ HDVH is all heavy
# goods vehicles; the model splits them into HDVH and HDVM (rigid trucks).
TECH_MODES = {
    ("passenger", "LDV"): "cars",
    ("passenger", "2W"): "motorbikes",
    ("passenger", "bus"): "buses",
    ("passenger", "rail"): "passenger rail",
    ("freight", "HDVH"): "heavy trucks",
    ("freight", "HDVM"): "heavy trucks",
    ("freight", "HDVL"): "vans",
    ("freight", "rail"): "freight rail",
}
# Technology -> fuel group. EP2050+ has no plug-in hybrids, so model PHEV are
# counted with their fuel. Model technologies not listed (aviation, ships, metro/
# tram) are not in these modes.
TECH_GROUP = {
    "ICE-gasoline": "gasoline",
    "PHEV-gasoline": "gasoline",
    "ICE-diesel": "diesel",
    "PHEV-diesel": "diesel",
    "ICE": "diesel",  # freight "ICE" (unspecified), treated as diesel
    "ICE-gas": "gas",
    "BEV": "electric",
    "CEV": "electric",
    "FCEV": "hydrogen",
    "H2": "hydrogen",
}
# Rail is compared as one total: EP2050+ (Abb. 112) gives rail energy for all fuels
# together, stored under "CEV" in the pickle; the model also has diesel trains.
TECH_GROUP_ORDER = ["gasoline", "diesel", "gas", "electric", "hydrogen", "all fuels"]
TECH_ORDER = [
    f"{m} | {g}" for m in dict.fromkeys(TECH_MODES.values()) for g in TECH_GROUP_ORDER
] + ["total"]

# --- Electricity, road vs rail ------------------------------------------------
# SFOE T17e splits transport electricity into rail, road and non-road. Model
# road = BEV of road vehicles + the 10 % of plug-in hybrid energy that the model
# counts as electricity (workflows.py, "PHEV-elec"). Model rail = trains + metro/tram.
ELEC_ROAD_MODES = ["2W", "LDV", "bus", "HDVH", "HDVM", "HDVL"]
ELEC_RAIL_MODES = ["rail", "metrotram"]
ELEC_ORDER = ["road", "rail", "total"]

# --- FSO pkm and vkm by mode ----------------------------------------------------
# FSO row label -> model mode, one map per sheet. Same mapping as the
# pre-processing (get_data_functions/demand_pkm_vkm.py). In the 1990-2004 sheets,
# "Transport par bus" is the bus subtotal (urban "Autobus" + regional "Bus"), so
# "Autobus" is not read there; from 2005 "Autobus" covers all buses.
_PKM_COMMON = {
    "Chemins de fer": "rail",
    "Chemins de fer à crémaillère": "rail",
    "Trams": "metrotram",
    "Trolleybus": "bus",
    "Voitures de tourisme": "LDV",
    "Cars": "bus",  # coaches, counted as bus as in the model
    "Motocycles": "2W",
}
FSO_PKM_ROWS = {
    "1990-2004": {
        **_PKM_COMMON,
        "Transport par bus": "bus",
        "Bicyclettes": "bike",
        "à pied": "walk",
    },
    "2005-2023": {
        **_PKM_COMMON,
        "Autobus": "bus",
        "Bicyclettes, y. c. vélos électriques lents": "bike",
        "À pied": "walk",
    },
}
_VKM_COMMON = {
    "en millions de trains-km": "rail",
    "Tram": "metrotram",
    "Trolleybus": "bus",
    "Voitures de tourisme": "LDV",
    "Cars privés": "bus",  # coaches
    "Motocycles": "2W",
}
FSO_VKM_ROWS = {
    "1990-2004": {**_VKM_COMMON, "Transport par bus": "bus"},
    "2005-2023": {**_VKM_COMMON, "Autobus": "bus"},
}
# Aviation is left out: FSO has no residents' flight pkm, and the model aviation
# "vkm" is seat-km.
PKM_ORDER = ["LDV", "2W", "bus", "metrotram", "rail", "bike", "walk", "total"]
VKM_ORDER = ["LDV", "2W", "bus", "metrotram", "rail", "total"]
# EP2050+ road modes compared with the model. EP2050+ trucks (HDVH, HDVL) are left
# out: the TPE has no freight vkm.
EP2050_VKM_MODES = ["LDV", "2W", "bus"]
PASSENGER_TITLES = {
    "LDV": "cars",
    "2W": "motorbikes",
    "bus": "buses",
    "metrotram": "tram",
    "rail": "rail",
    "bike": "bike",
    "walk": "walk",
    "total": "total",
}

COLOR_OFFICIAL = "#eb6834"
COLOR_PASSENGER = "#2a78d6"
COLOR_FREIGHT = "#8fbbea"
COLOR_LOCAL = "#1b4f8f"
COLOR_EP2050 = "#7a4fb5"


# --- Official data: build and cache ------------------------------------------
REBUILD = False  # set by --rebuild


def build_sfoe_energy():
    """SFOE T17e transport final energy [TWh]: {"by-fuel": dm (fuel),
    "electricity-split": dm (rail, road, non-road)}."""
    os.makedirs(DATA_DIR, exist_ok=True)
    var = "tra_energy-demand_by-fuel"
    parameters = {
        "mapping": dict(SFOE_COLUMNS),  # copied: the extractor edits it in place
        "var name": var,
        "headers indexes": (8, 9, 10),  # rows combined into the header
        "first row": 11,
        "unit": None,  # unit is read from the header (TJ)
        "cols to drop": r".*%.*|.*extra.*",  # keep TJ columns
    }
    dm = extract_energy_statistics_data(SFOE_URL, SFOE_XLSX, "T17e", parameters, YEARS)
    # Rebuild with this package's DataMatrix: the extractor may return the class
    # under its old module name (see transition_compass_model/__init__.py).
    dm = DataMatrix.create_from_df(dm.write_df(), num_cat=1)
    dm.drop("Categories1", ["coal", "gas-pipeline"])
    dm.rename_col("gas-other", "gas", "Categories1")
    dm.change_unit(var, 1 / 3600, "TJ", "TWh")
    out = {}
    for key, cats in (("by-fuel", SFOE_BY_FUEL), ("electricity-split", SFOE_ELEC)):
        out[key] = dm.filter({"Categories1": cats})
        out[key].sort("Categories1")
    return out


def build_ep2050_energy():
    """EP2050+ energy [MJ]: {"passenger": dm (2W, LDV, bus, rail x technology),
    "freight": dm (HDVH, HDVL, rail x technology), "aviation": dm (kerosene)}.
    Rail is a total for all fuels, stored under "CEV"."""
    dm_road = extract_EP2050_transport_energy_demand(
        EP2050_ROAD_URL, EP2050_ROAD_ZIP, EP2050_ROAD_PICKLE
    )
    # The extractor returns TWh.
    dm_road.change_unit("tra_energy_demand", MJ_PER_TWH, "TWh", "MJ", operator="*")
    dm_road.filter({"Years": YEARS}, inplace=True)
    dm_road.groupby(
        {"ICE-gasoline": ["biogasoline", "gasoline"]}, "Categories2", inplace=True
    )

    # Returns rail only, and writes the whole Abb. 112 table to EP2050_RAIL_PICKLE
    # if it is missing; aviation is read from that pickle below.
    dm_rail = extract_EP2050_transport_energy_demand_rail(
        EP2050_RAIL_URL, EP2050_RAIL_ZIP, EP2050_RAIL_PICKLE
    )
    dm_rail.filter({"Years": YEARS}, inplace=True)
    dm_rail.change_unit("tra_energy_demand", 3.6e9, "TWh", "MJ", operator="*")

    out = {}
    for kind, road_modes in (
        ("passenger", ["2W", "LDV", "bus"]),
        ("freight", ["HDVH", "HDVL"]),
    ):
        dm = dm_road.filter({"Categories1": road_modes}).flatten()
        dm_r = dm_rail.filter({"Categories2": [kind]})
        dm_r.rename_col(kind, "CEV", "Categories2")
        dm.append(dm_r.flatten(), "Categories1")
        dm.deepen(based_on="Categories1")
        out[kind] = dm

    aviation = [
        "tra_energy_demand_aviation (int.)",
        "tra_energy_demand_aviation (nat.)",
    ]
    with open(EP2050_RAIL_PICKLE, "rb") as f:
        dm = pickle.load(f).filter({"Variables": aviation})
    dm.groupby(
        {"tra-energy-demand_aviation_kerosene": aviation}, "Variables", inplace=True
    )
    dm.change_unit("tra-energy-demand_aviation_kerosene", 1e9, "PJ", "MJ", operator="*")
    dm.filter({"Years": YEARS}, inplace=True)
    dm.deepen_twice()
    out["aviation"] = dm
    return out


def build_foen_emissions():
    """FOEN transport emissions by mode and gas [Mt]: {"by-mode": dm}."""
    if not os.path.exists(FOEN_XLSX):
        raise FileNotFoundError(f"{FOEN_XLSX} is missing (FOEN GHG inventory).")
    frames = []
    for gas in GWP:
        df = pd.read_excel(FOEN_XLSX, sheet_name=gas)
        df.columns = df.iloc[3, :]
        df = df.iloc[FOEN_ROWS, :]
        df = pd.melt(df, id_vars=["Cat. (1)", np.nan], var_name="Years")
        df.columns = ["code", "row", "Years", "value"]
        df["row"] = df["row"].astype(str).str.strip()
        df = df.loc[df["row"].isin(FOEN_MODES)]
        df["mode"] = df["row"].map(FOEN_MODES)
        df["Years"] = df["Years"].astype(int)
        df["value"] = df["value"].astype(float) * FOEN_TO_MT[gas]
        df["gas"] = gas
        frames.append(df)
    df = pd.concat(frames).groupby(["Years", "mode", "gas"])["value"].sum(min_count=1)
    df = df.reset_index()
    df["col"] = "tra_emissions_" + df["mode"] + "_" + df["gas"] + "[Mt]"
    df = df.pivot(index="Years", columns="col", values="value").reset_index()
    df["Country"] = COUNTRY
    dm = DataMatrix.create_from_df(df, num_cat=2)
    missing = [y for y in YEARS if y not in dm.col_labels["Years"]]
    if missing:
        dm.add(np.nan, "Years", missing, dummy=True)
        dm.sort("Years")
    return {"by-mode": dm}


def _load_or_build(file, builder):
    """Cached official data: load `file`, or build it (always with --rebuild)."""
    if REBUILD or not os.path.exists(file):
        print(f"Building {os.path.relpath(file, HERE)}")
        data = builder()
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return data
    with open(file, "rb") as f:
        return pickle.load(f)


def _load_calib(file, key):
    """Pickle written by the transport pre-processing (EP2050+ vkm)."""
    if not os.path.exists(file):
        raise FileNotFoundError(
            f"{file} is missing. Run the transport pre-processing "
            "(transport_preprocessing_main_CH.py) to create it."
        )
    with open(file, "rb") as f:
        return pickle.load(f)[key]


def _dm_cat_df(dm, unit):
    """DataMatrix (one variable, one category level) -> Years x category."""
    dm = dm.filter({"Country": [COUNTRY]})
    var = dm.col_labels["Variables"][0]
    if dm.units[var] != unit:
        raise ValueError(f"Expected {var} in {unit}, got {dm.units[var]}")
    return pd.DataFrame(
        dm.array[0, :, 0, :],
        index=[int(y) for y in dm.col_labels["Years"]],
        columns=dm.col_labels["Categories1"],
    )


def load_official_energy():
    """SFOE transport final energy by fuel group [TWh]. Years x group."""
    df = _dm_cat_df(_load_or_build(SFOE_FILE, build_sfoe_energy)["by-fuel"], "TWh")
    return df.loc[df.index.isin(YEARS)]


def load_official_emissions():
    """FOEN transport emissions by mode [Mt CO2e]. Years x mode."""
    dm = _load_or_build(FOEN_FILE, build_foen_emissions)["by-mode"]
    dm = dm.filter({"Country": [COUNTRY]})
    arr = dm.array[0, :, 0, :, :]  # years x mode x gas
    gases = dm.col_labels["Categories2"]
    factor = np.array([GWP[g] for g in gases])
    df = pd.DataFrame(
        (arr * factor).sum(axis=-1),
        index=[int(y) for y in dm.col_labels["Years"]],
        columns=dm.col_labels["Categories1"],
    )
    return df.loc[df.index.isin(YEARS)].dropna(how="all")


def load_ep2050_energy_by_mode():
    """EP2050+ energy by mode [TWh], technologies summed. Years x column.
    Years with no EP2050+ data are dropped."""
    cols = {}
    for kind, dm in _load_or_build(EP2050_FILE, build_ep2050_energy).items():
        dm = dm.filter({"Country": [COUNTRY]})
        var = dm.col_labels["Variables"][0]
        if dm.units[var] != "MJ":
            raise ValueError(f"Expected EP2050+ {kind} energy in MJ")
        arr = np.nansum(dm.array[0, :, 0, :, :], axis=-1) / MJ_PER_TWH
        years = [int(y) for y in dm.col_labels["Years"]]
        for i, mode in enumerate(dm.col_labels["Categories1"]):
            col = EP2050_TO_COLUMN[(kind, mode)]
            s = pd.Series(arr[:, i], index=years)
            cols[col] = cols[col] + s if col in cols else s
    df = pd.DataFrame(cols)
    df = df.loc[df.index.isin(YEARS)]
    return df.loc[df.sum(axis=1) > 0]


def _by_mode_tech(arrays, years):
    """{(kind, mode, tech): array over years [TWh]} -> Years x "panel | fuel group".
    Pairs not in TECH_MODES are skipped. A technology not in TECH_GROUP is an error
    if it has energy, and skipped if it is empty."""
    cols = {}
    for (kind, mode, tech), arr in arrays.items():
        if (kind, mode) not in TECH_MODES:
            continue
        if tech not in TECH_GROUP:
            if np.nansum(np.abs(arr)) > 0:
                raise KeyError(f"Technology {tech} ({kind} {mode}) has no fuel group")
            continue  # empty technology slot
        group = "all fuels" if mode == "rail" else TECH_GROUP[tech]
        col = f"{TECH_MODES[(kind, mode)]} | {group}"
        s = pd.Series(np.nan_to_num(arr), index=years)
        cols[col] = cols[col] + s if col in cols else s
    return pd.DataFrame(cols)


def _dm_arrays(dm, kind):
    """DataMatrix (one variable, mode x technology, MJ) -> {(kind, mode, tech):
    array [TWh]}, plus the list of years."""
    dm = dm.filter({"Country": [COUNTRY]})
    var = dm.col_labels["Variables"][0]
    if dm.units[var] != "MJ":
        raise ValueError(f"Expected {var} in MJ")
    years = [int(y) for y in dm.col_labels["Years"]]
    arr = dm.array[0, :, 0, :, :] / MJ_PER_TWH  # years x mode x tech
    out = {
        (kind, mode, tech): arr[:, i, j]
        for i, mode in enumerate(dm.col_labels["Categories1"])
        for j, tech in enumerate(dm.col_labels["Categories2"])
    }
    return out, years


def load_ep2050_energy_by_tech():
    """EP2050+ energy by mode and fuel group [TWh]. Years x "panel | group"."""
    data = _load_or_build(EP2050_FILE, build_ep2050_energy)
    parts = []
    for kind in ("passenger", "freight"):
        arrays, years = _dm_arrays(data[kind], kind)
        parts.append(_by_mode_tech(arrays, years))
    df = pd.concat(parts, axis=1)
    df = df.loc[df.index.isin(YEARS)]
    return df.loc[df.sum(axis=1) > 0]


def load_official_electricity():
    """SFOE transport electricity, road and rail [TWh]. Years x {road, rail}."""
    data = _load_or_build(SFOE_FILE, build_sfoe_energy)
    df = _dm_cat_df(data["electricity-split"], "TWh")
    df = df.rename(columns={"electricity-road": "road", "electricity-rail": "rail"})
    return df.loc[df.index.isin(YEARS), ["road", "rail"]].dropna(how="all")


def _read_fso_excel(file, header_row, rows_by_sheet):
    """FSO table by mode [millions], one sheet per entry of `rows_by_sheet`
    ({sheet: {row label: mode}}). Empty cells and zeros are NaN. Years x mode."""
    if not os.path.exists(file):
        raise FileNotFoundError(
            f"{file} is missing. Run the transport pre-processing "
            "(transport_demand_pipeline.py) to download it."
        )
    parts = []
    for sheet, rows in rows_by_sheet.items():
        missing = set(rows)
        df = pd.read_excel(file, sheet_name=sheet, header=None)
        years = pd.to_numeric(df.iloc[header_row, 1:], errors="coerce")
        labels = (
            df.iloc[:, 0]
            .astype(str)
            .str.replace(r"\d+\)|\(\d+\)", "", regex=True)
            .str.strip()
        )
        missing -= set(labels)
        if missing:
            raise KeyError(f"{os.path.basename(file)} [{sheet}]: no row {missing}")
        body = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        body.columns = years
        body = body.loc[:, years.notna().values]
        body.columns = body.columns.astype(int)
        body = body.loc[labels.isin(rows).values]
        modes = labels[labels.isin(rows)].map(rows).values
        # NaN only if every row of the mode is empty
        parts.append(body.groupby(modes).sum(min_count=1).T)
    df = pd.concat(parts)
    df = df.loc[~df.index.duplicated(keep="last")].replace(0, np.nan)
    return df.loc[df.index.isin(YEARS)].dropna(how="all")


def load_fso_pkm():
    """FSO passenger-km by mode [Bpkm]. Years x mode."""
    return _read_fso_excel(FSO_PKM_FILE, 2, FSO_PKM_ROWS) / 1e3


def load_fso_vkm():
    """FSO vehicle-km by mode [Bvkm] (rail in train-km). Years x mode."""
    return _read_fso_excel(FSO_VKM_FILE, 1, FSO_VKM_ROWS) / 1e3


def load_ep2050_vkm():
    """EP2050+ road vkm by mode [Bvkm], technologies summed. Years x mode."""
    dm = _load_calib(VKM_FILE, VKM_KEY).filter({"Country": [COUNTRY]})
    if dm.units["tra_vkm_demand"] != "vkm":
        raise ValueError("Expected EP2050+ vkm in vkm")
    df = pd.DataFrame(
        np.nansum(dm.array[0, :, 0, :, :], axis=-1) / 1e9,
        index=[int(y) for y in dm.col_labels["Years"]],
        columns=dm.col_labels["Categories1"],
    )[EP2050_VKM_MODES]
    df = df.loc[df.index.isin(YEARS)]
    return df.loc[df.sum(axis=1) > 0]


# --- Model --------------------------------------------------------------------
def run_model():
    """Run climate -> lifestyles -> transport for Switzerland.

    Returns (tpe, tech):
      tpe   the TPE as a DataFrame, Years x "variable[unit]", historical years only.
      tech  energy demand by mode and technology [MJ], {"passenger": dm,
            "freight": dm}. The TPE has no such detail, so it is taken from the
            "tech" output of the two workflow functions, wrapped for this run.
    """
    lever_setting = {k: math.floor(v) for k, v in load_lever_config().items()}
    sectors = ["climate", "lifestyles", "transport"]
    DM_input = filter_country_and_load_data_from_pickles(
        country_list=[COUNTRY], modules_list=sectors
    )
    tech = {}
    originals = {}
    for kind in ("passenger", "freight"):
        name = f"{kind}_fleet_energy"
        originals[name] = getattr(transport_wkf, name)

        def wrapped(*args, _f=originals[name], _kind=kind, **kwargs):
            out = _f(*args, **kwargs)
            # Copy now: the module edits fleet counts in place later on.
            tech[_kind] = out["tech"].filter(
                {"Variables": [f"tra_{_kind}_energy-demand"]}
            )
            return out

        setattr(transport_wkf, name, wrapped)
    try:
        interface = Interface()
        climate(lever_setting, YEARS_SETTING, DM_input["climate"], interface)
        lifestyles(lever_setting, YEARS_SETTING, DM_input["lifestyles"], interface)
        dm_tpe, _ = transport(
            lever_setting, YEARS_SETTING, DM_input["transport"], interface
        )
    finally:
        for name, f in originals.items():
            setattr(transport_wkf, name, f)
    df = dm_tpe.filter({"Country": [COUNTRY]}).write_df()
    df = df.drop(columns="Country").set_index("Years")
    df.index = df.index.astype(int)
    return df.loc[df.index.isin(YEARS)], tech


def _tpe_block(tpe, prefix, unit):
    """TPE columns "<prefix>_<cat>[<unit>]" as a DataFrame Years x cat."""
    start, end = f"{prefix}_", f"[{unit}]"
    cols = [c for c in tpe.columns if c.startswith(start) and c.endswith(end)]
    if not cols:
        raise KeyError(f"No TPE variable {start}*{end}")
    return tpe[cols].rename(columns=lambda c: c[len(start) : -len(end)])


def model_energy(tpe):
    """Model energy by SFOE fuel group [TWh], {"passenger": df, "freight": df}."""
    out = {}
    for kind in ("passenger", "freight"):
        df = _tpe_block(tpe, f"tra_energy-demand-by-fuel-{kind}", "TWh")
        unmapped = sorted(set(df.columns) - set(MODEL_FUEL_TO_GROUP))
        if unmapped:
            raise KeyError(f"Model fuels with no SFOE group: {unmapped}")
        out[kind] = df.T.groupby(MODEL_FUEL_TO_GROUP).sum().T
    return out


def model_energy_by_mode(tpe):
    """Model energy by EP2050+ comparison column [TWh]. Years x column."""
    cols = {}
    for (kind, mode), col in MODEL_TO_EP2050_COLUMN.items():
        s = tpe[f"tra_{kind}_energy-demand-by-mode_{mode}[TWh]"]
        cols[col] = cols[col] + s if col in cols else s
    return pd.DataFrame(cols)


def model_energy_by_tech(tech):
    """Model energy by mode and fuel group [TWh]. Years x "panel | group"."""
    parts = []
    for kind, dm in tech.items():
        arrays, years = _dm_arrays(dm, kind)
        parts.append(_by_mode_tech(arrays, years))
    df = pd.concat(parts, axis=1)
    return df.T.groupby(level=0).sum().T  # heavy trucks = HDVH + HDVM


def model_electricity(tech):
    """Model electricity, road and rail [TWh]. Years x {road, rail}."""
    road = rail = 0
    for kind, dm in tech.items():
        arrays, years = _dm_arrays(dm, kind)
        for (_, mode, t), arr in arrays.items():
            arr = np.nan_to_num(arr)
            if mode in ELEC_ROAD_MODES:
                if t == "BEV":
                    road = road + arr
                elif t.startswith("PHEV"):
                    road = road + 0.1 * arr
            elif mode in ELEC_RAIL_MODES and t in ("CEV", "BEV", "mt"):
                rail = rail + arr
    return pd.DataFrame({"road": road, "rail": rail}, index=years)


def model_pkm(tpe):
    """Model passenger-km by mode [Bpkm], aviation left out. Years x mode."""
    df = _tpe_block(tpe, "tra_passenger_transport-demand-by-mode", "Bpkm")
    return df.drop(columns="aviation")


def model_vkm(tpe):
    """Model vehicle-km by mode [Bvkm], aviation (seat-km) left out."""
    df = _tpe_block(tpe, "tra_passenger_transport-demand-vkm", "Bvkm")
    return df.drop(columns="aviation")


def model_emissions(tpe):
    """Model emissions [Mt CO2e]: by FOEN mode (Years x mode), by kind
    (Years x passenger/freight) and the outbound-only passenger aviation."""
    cols = {}
    for (kind, mode), foen in MODEL_MODE_TO_FOEN.items():
        s = tpe[f"tra_emissions-CO2e_{kind}_{mode}[Mt]"]
        cols[foen] = cols[foen] + s if foen in cols else s
    by_mode = pd.DataFrame(cols)
    by_kind = pd.DataFrame(
        {
            kind: _tpe_block(tpe, f"tra_emissions-CO2e-by-fuel-{kind}", "Mt").sum(
                axis=1
            )
            for kind in ("passenger", "freight")
        }
    )
    # Aviation with passenger flights counted outbound only (about the flights
    # leaving Swiss airports, like the inventory) + freight aviation.
    aviation_local = (
        tpe["tra_passenger-emissions-local_aviation[Mt]"]
        + tpe["tra_emissions-CO2e_freight_aviation[Mt]"]
    )
    return by_mode, by_kind, aviation_local


# --- Comparison ---------------------------------------------------------------
def align(official, model_parts, order):
    """Same years and columns for the official data and each model part, plus a
    total column. A column one side does not have is 0. Gaps inside a column stay
    NaN, and so does the total of a year with a gap. Columns that are zero
    everywhere are dropped."""
    years = sorted(set(official.index).intersection(*[m.index for m in model_parts]))
    cols = [c for c in order if c in set(official).union(*model_parts)]
    out = []
    for df in (official, *model_parts):
        missing = [c for c in cols if c not in df.columns]
        df = df.reindex(index=years, columns=cols)
        df[missing] = 0.0
        out.append(df)
    keep = [c for c in cols if sum(df[c].abs().sum() for df in out) > 0]
    for df in out:
        df.drop(columns=[c for c in cols if c not in keep], inplace=True)
        df["total"] = df.sum(axis=1, min_count=len(keep)).where(df.notna().all(axis=1))
    return out


def summary_table(official, parts):
    """Rows per column: official, each model part, model total and model/official,
    for KEY_YEARS + the average over all years. `parts` = {name: df}."""
    years = [y for y in KEY_YEARS if y in official.index]
    label = f"avg {official.index[0]}-{official.index[-1]}"
    model = sum(parts.values())
    rows = {}
    for c in official.columns:
        valid = official[c].notna()  # average only over years with official data

        def vals(df):
            return df[c].loc[years].tolist() + [df[c][valid].mean()]

        o, m = vals(official), vals(model)
        rows[(c, "official")] = o
        if len(parts) > 1:
            for name, df in parts.items():
                rows[(c, f"model {name}")] = vals(df)
        rows[(c, "model")] = m
        rows[(c, "model/official")] = [
            mv / ov if abs(ov) > 1e-6 else np.nan for mv, ov in zip(m, o)
        ]
    df = pd.DataFrame(rows, index=years + [label]).T
    df.index.names = ["", ""]
    return df


def _grid(n, titles):
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=titles,
        shared_xaxes="all",
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )
    return fig, [(i // ncols + 1, i % ncols + 1) for i in range(n)]


def _official_trace(df, col, unit, show):
    return go.Scatter(
        x=df.index,
        y=df[col],
        mode="lines",
        name="official",
        legendgroup="official",
        showlegend=show,
        line={"width": 2, "dash": "dash", "color": COLOR_OFFICIAL},
        hovertemplate=f"official: %{{y:.2f}} {unit}<extra></extra>",
    )


def _stacked_traces(parts, col, unit, show):
    colors = {"passenger": COLOR_PASSENGER, "freight": COLOR_FREIGHT}
    return [
        go.Scatter(
            x=df.index,
            y=df[col],
            mode="lines",
            name=f"model {name}",
            legendgroup=name,
            showlegend=show,
            stackgroup="model",
            line={"width": 0.5, "color": colors[name]},
            fillcolor=colors[name],
            hovertemplate=f"model {name}: %{{y:.2f}} {unit}<extra></extra>",
        )
        for name, df in parts.items()
    ]


def energy_figure(official, parts):
    """One panel per fuel: official dashed, model passenger + freight stacked."""
    cols = list(official.columns)
    fig, pos = _grid(len(cols), cols)
    for i, (c, (r, k)) in enumerate(zip(cols, pos)):
        for tr in _stacked_traces(parts, c, "TWh", i == 0):
            fig.add_trace(tr, row=r, col=k)
        fig.add_trace(_official_trace(official, c, "TWh", i == 0), row=r, col=k)
        fig.update_yaxes(rangemode="tozero", row=r, col=k)
    fig.update_yaxes(title_text="TWh/yr", col=1)
    fig.update_layout(
        title=f"Transport final energy by fuel, {COUNTRY}: model vs SFOE (T17e)",
        template="plotly_white",
        hovermode="x unified",
        height=320 * len({r for r, _ in pos}) + 80,
    )
    return fig


def emissions_figure(official, model, by_kind, aviation_local):
    """One panel per mode (official dashed, model solid); the total panel shows
    model passenger + freight stacked."""
    cols = list(official.columns)
    fig, pos = _grid(len(cols), [MODE_TITLES[c] for c in cols])
    for i, (c, (r, k)) in enumerate(zip(cols, pos)):
        if c == "total":
            for tr in _stacked_traces(by_kind, "total", "Mt", True):
                fig.add_trace(tr, row=r, col=k)
        else:
            fig.add_trace(
                go.Scatter(
                    x=model.index,
                    y=model[c],
                    mode="lines",
                    name="model",
                    legendgroup="model",
                    showlegend=(i == 0),
                    line={"width": 2, "color": COLOR_PASSENGER},
                    hovertemplate="model: %{y:.2f} Mt<extra></extra>",
                ),
                row=r,
                col=k,
            )
        if c == "aviation":
            fig.add_trace(
                go.Scatter(
                    x=aviation_local.index,
                    y=aviation_local,
                    mode="lines",
                    name="model, passenger flights outbound only",
                    line={"width": 2, "dash": "dot", "color": COLOR_LOCAL},
                    hovertemplate="model outbound: %{y:.2f} Mt<extra></extra>",
                ),
                row=r,
                col=k,
            )
        fig.add_trace(_official_trace(official, c, "Mt", i == 0), row=r, col=k)
        fig.update_yaxes(rangemode="tozero", row=r, col=k)
    fig.update_yaxes(title_text="Mt CO2e/yr", col=1)
    fig.update_layout(
        title=f"Transport emissions by mode, {COUNTRY}: model vs FOEN inventory",
        template="plotly_white",
        hovermode="x unified",
        height=320 * len({r for r, _ in pos}) + 80,
    )
    return fig


def mode_figure(model, sources, unit, title, titles=None):
    """One panel per mode of `model`: model solid, each official source dashed.
    `sources` = {name: (df, color)}; a source without a mode is not drawn there."""
    cols = list(model.columns)
    fig, pos = _grid(len(cols), [(titles or {}).get(c, c) for c in cols])
    shown = set()
    for i, (c, (r, k)) in enumerate(zip(cols, pos)):
        fig.add_trace(
            go.Scatter(
                x=model.index,
                y=model[c],
                mode="lines",
                name="model",
                legendgroup="model",
                showlegend=(i == 0),
                line={"width": 2, "color": COLOR_PASSENGER},
                hovertemplate=f"model: %{{y:.2f}} {unit}<extra></extra>",
            ),
            row=r,
            col=k,
        )
        for name, (df, color) in sources.items():
            if c not in df.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[c],
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    showlegend=name not in shown,
                    line={"width": 2, "dash": "dash", "color": color},
                    hovertemplate=f"{name}: %{{y:.2f}} {unit}<extra></extra>",
                ),
                row=r,
                col=k,
            )
            shown.add(name)
        fig.update_yaxes(rangemode="tozero", row=r, col=k)
    fig.update_yaxes(title_text=f"{unit}/yr", col=1)
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        height=320 * len({r for r, _ in pos}) + 80,
    )
    return fig


def ep2050_figure(official, model):
    """One panel per mode: EP2050+ dashed, model solid."""
    return mode_figure(
        model,
        {"EP2050+": (official, COLOR_OFFICIAL)},
        "TWh",
        f"Transport final energy by mode, {COUNTRY}: model vs EP2050+",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", default=OUT_DIR, help="output directory for the HTML plot"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="rebuild the cached official data in scripts/data/",
    )
    args = parser.parse_args()
    global REBUILD
    REBUILD = args.rebuild

    print("Loading official data")
    off_energy = load_official_energy()
    off_emissions = load_official_emissions()
    off_ep2050 = load_ep2050_energy_by_mode()
    off_ep2050_tech = load_ep2050_energy_by_tech()
    off_elec = load_official_electricity()
    off_pkm = load_fso_pkm()
    off_vkm = load_fso_vkm()
    off_ep_vkm = load_ep2050_vkm()

    print("Running climate -> lifestyles -> transport")
    tpe, tech = run_model()

    # --- Energy by fuel -------------------------------------------------------
    m_energy = model_energy(tpe)
    official_e, pass_e, freight_e = align(
        off_energy, [m_energy["passenger"], m_energy["freight"]], FUEL_ORDER
    )
    energy_parts = {"passenger": pass_e, "freight": freight_e}

    # --- Energy by mode, EP2050+ ----------------------------------------------
    official_ep, model_ep = align(off_ep2050, [model_energy_by_mode(tpe)], EP2050_ORDER)
    official_tech, model_tech = align(
        off_ep2050_tech, [model_energy_by_tech(tech)], TECH_ORDER
    )

    # --- Electricity, road vs rail, SFOE --------------------------------------
    official_el, model_el = align(off_elec, [model_electricity(tech)], ELEC_ORDER)

    # --- pkm and vkm by mode ---------------------------------------------------
    official_pkm, model_pkm_ = align(off_pkm, [model_pkm(tpe)], PKM_ORDER)
    m_vkm = model_vkm(tpe)
    official_vkm, model_vkm_ = align(off_vkm, [m_vkm], VKM_ORDER)
    official_ep_vkm, model_ep_vkm = align(
        off_ep_vkm, [m_vkm[EP2050_VKM_MODES]], VKM_ORDER
    )

    # --- Emissions by mode ----------------------------------------------------
    m_by_mode, m_by_kind, aviation_local = model_emissions(tpe)
    official_m, model_m = align(off_emissions, [m_by_mode], MODE_ORDER)
    by_kind = {
        k: pd.DataFrame({"total": m_by_kind[k].loc[official_m.index]})
        for k in m_by_kind
    }
    aviation_local = aviation_local.loc[official_m.index]
    # Sanity check: emissions by fuel and by mode give the same total
    gap = (sum(df["total"] for df in by_kind.values()) - model_m["total"]).abs().max()
    if gap > 1e-6:
        print(f"WARNING: model emissions by fuel and by mode differ by {gap:.3f} Mt")

    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print("\n=== Energy by fuel, SFOE T17e [TWh/yr] ===")
        print(summary_table(official_e, energy_parts).round(2).to_string())
        print("\n=== Energy by mode, EP2050+ [TWh/yr] ===")
        print(summary_table(official_ep, {"": model_ep}).round(2).to_string())
        print("\n=== Energy by mode and fuel, EP2050+ [TWh/yr] ===")
        print(summary_table(official_tech, {"": model_tech}).round(2).to_string())
        print("\n=== Electricity, road vs rail, SFOE T17e [TWh/yr] ===")
        print(summary_table(official_el, {"": model_el}).round(2).to_string())
        print("\n=== Passenger-km by mode, FSO [Bpkm/yr] ===")
        print(summary_table(official_pkm, {"": model_pkm_}).round(2).to_string())
        print("\n=== Vehicle-km by mode, FSO [Bvkm/yr] (rail: train-km) ===")
        print(summary_table(official_vkm, {"": model_vkm_}).round(3).to_string())
        print("\n=== Road vehicle-km by mode, EP2050+ [Bvkm/yr] ===")
        print(summary_table(official_ep_vkm, {"": model_ep_vkm}).round(3).to_string())
        print("\n=== Emissions by mode, FOEN inventory [Mt CO2e/yr] ===")
        print(summary_table(official_m, {"": model_m}).round(2).to_string())
        y = official_m.index[-1]
        print(
            f"\nModel emissions {y}: passenger {by_kind['passenger'].loc[y, 'total']:.2f}"
            f" Mt, freight {by_kind['freight'].loc[y, 'total']:.2f} Mt. "
            f"Aviation with passenger flights outbound only: {aviation_local.loc[y]:.2f} Mt."
        )
    print(
        "\nNotes:\n"
        "- SFOE and FOEN do not split passenger and freight; only the model is split.\n"
        "- Aviation: SFOE kerosene and FOEN aviation count fuel sold at Swiss airports\n"
        "  (domestic + flights leaving Switzerland). The model counts all flights of\n"
        "  Swiss residents, both ways and with connections, so it is higher. The\n"
        "  'outbound only' line is closer to the official boundary.\n"
        "- Model renewables-other = biofuels + e-fuels + hydrogen. SFOE renewables-other\n"
        "  = biofuels, biogas and other renewables.\n"
        "- Electricity has no direct emissions, so it only appears in the energy check.\n"
        "- EP2050+: model trucks include rigid trucks (HDVM); model metro/tram is left\n"
        "  out (EP2050+ rail = trains). Model efficiency for several modes was built\n"
        "  from EP2050+ energy / vkm, so gaps there come from vkm or technology split.\n"
        "  Bus efficiency is FSO energy / FSO vkm (public operators), not EP2050+.\n"
        "- EP2050+ by mode and fuel: model plug-in hybrids are counted with their fuel\n"
        "  (EP2050+ has no PHEV); heavy trucks = model HDVH + HDVM; vans = HDVL.\n"
        "- Electricity road vs rail: model road = BEV + 10 % of PHEV energy; model rail\n"
        "  = trains + metro/tram. SFOE 'non-road' electricity is not compared.\n"
        "- pkm / vkm: the model's Swiss values for 1990-2023 are built from the same\n"
        "  FSO tables (with FSO gaps filled by trends), so a ratio far from 1 points to\n"
        "  a pre-processing or lever problem. Empty FSO years are left out of the\n"
        "  averages, and the total is only shown for years where every mode has data.\n"
        "- EP2050+ road vkm is territorial (vehicles driving in Switzerland). EP2050+\n"
        "  buses include coaches, as in the model and here."
    )

    os.makedirs(args.out, exist_ok=True)
    f = os.path.join(args.out, "transport_validation_CH.html")
    figs = [
        energy_figure(official_e, energy_parts),
        ep2050_figure(official_ep, model_ep),
        mode_figure(
            model_tech,
            {"EP2050+": (official_tech, COLOR_OFFICIAL)},
            "TWh",
            f"Transport final energy by mode and fuel, {COUNTRY}: model vs EP2050+",
        ),
        mode_figure(
            model_el,
            {"SFOE": (official_el, COLOR_OFFICIAL)},
            "TWh",
            f"Transport electricity, road vs rail, {COUNTRY}: model vs SFOE (T17e)",
        ),
        mode_figure(
            model_pkm_,
            {"FSO": (official_pkm, COLOR_OFFICIAL)},
            "Bpkm",
            f"Passenger-km by mode, {COUNTRY}: model vs FSO",
            PASSENGER_TITLES,
        ),
        mode_figure(
            model_vkm_,
            {
                "FSO": (official_vkm, COLOR_OFFICIAL),
                "EP2050+": (official_ep_vkm.drop(columns="total"), COLOR_EP2050),
            },
            "Bvkm",
            f"Vehicle-km by mode, {COUNTRY}: model vs FSO and EP2050+ (rail: train-km)",
            PASSENGER_TITLES,
        ),
        emissions_figure(official_m, model_m, by_kind, aviation_local),
    ]
    with open(f, "w") as fh:
        fh.write("<html><head><meta charset='utf-8'></head><body>\n")
        for i, fig in enumerate(figs):
            fh.write(fig.to_html(full_html=False, include_plotlyjs=(i == 0)))
        fh.write("</body></html>\n")
    print(f"\nPlot written to:\n  {os.path.abspath(f)}")


if __name__ == "__main__":
    main()
