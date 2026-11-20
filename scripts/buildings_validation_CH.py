"""Compare the Swiss buildings run with official energy statistics, by fuel.

Runs climate -> lifestyles -> buildings for Switzerland and compares the historical
years with three official series:
  1. BFE households, all uses   (Gesamtenergiestatistik, table T17a, 1990-2023)
  2. BFE services, all uses     (Gesamtenergiestatistik, table T17c, 1990-2023)
  3. FSO household heating      (STAT-TAB px-x-0204000000_106, "Ménages: chauffage",
                                 2000-2023). It includes hot water: FSO "Ménages:
                                 autres" has no heating oil or wood. So it is compared
                                 with model space heating + hot water.

"All uses" means space heating + hot water + appliances + lighting (+ cooling).
All values are final energy in TWh per year (1 TWh = 3,600 TJ).

It then checks services space heating on its own, with three more series:
  4. Useful heat per m² (EP2050 Dienstleistung, Tabelle 03-01, stock built before
     2020, kWh/m²). Useful heat is the heat the building loses, before boiler or
     heat-pump losses. The model computes it as U-value x envelope area x heating
     degree days, so this checks the services U-values. Same floor area as the
     model (EP2050).
  5. Space heating, final energy (Prognos 2021, Table 26, TWh, incl. agriculture).
  6. Heating mix (EP2050 Dienstleistung, Tabelle 02-02, share of heating systems in
     buildings built before 2010). Compared with the model's input heating mix,
     weighted by floor area: a share of systems, not of energy.

Prints one table per series (official, model, model/official) and writes an
interactive plotly page: one row per fuel group, one column per series, then the
services space-heating checks.

The official files are downloaded once and cached in scripts/data/ (git-ignored).
Delete them there to download again.

Usage (from the repo root):
    uv run python scripts/buildings_validation_CH.py [--out DIR]
        (default DIR: scripts/output/)
"""

import argparse
import math
import os
import pickle
import shutil
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from transition_compass_model._database.pre_processing.api_routines_CH import (
    get_data_api_CH,
)
from transition_compass_model._database.pre_processing.buildings.Switzerland.get_data_functions.energy_demand_for_calibration import (
    extract_energy_statistics_data,
)
from transition_compass_model._database.pre_processing.buildings.Switzerland.get_data_functions.services_CH import (
    PROGNOS_SERVICES_EUD_PJ,
)
from transition_compass_model.model.buildings_module import buildings
from transition_compass_model.model.climate_module import climate
from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    filter_country_and_load_data_from_pickles,
    save_url_to_file,
)
from transition_compass_model.model.common.config_loader import load_lever_config
from transition_compass_model.model.common.interface_class import Interface
from transition_compass_model.model.lifestyles_module import lifestyles

COUNTRY = "Switzerland"
YEARS_SETTING = [1990, 2023, 2025, 2050, 5]
YEARS = create_years_list(YEARS_SETTING[0], YEARS_SETTING[1], 1)
KEY_YEARS = [1990, 2000, 2010, 2020, 2023]
TJ_PER_TWH = 3600

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")  # cached official files (git-ignored)
OUT_DIR = os.path.join(HERE, "output")  # HTML plots (git-ignored)

# --- Official data sources ----------------------------------------------------
# BFE overall energy statistics. Source page:
# https://www.bfe.admin.ch/bfe/fr/home/versorgung/statistik-und-geodaten/energiestatistiken/gesamtenergiestatistik.html
BFE_URL = "https://pubdb.bfe.admin.ch/fr/publication/download/7519"
BFE_FILE = os.path.join(DATA_DIR, "statistique_globale_suisse_energie.xlsx")
BFE_SHEETS = {"households": "T17a", "services": "T17c"}
# Column header (regex) -> fuel group. The "% share" and "dont Huile extra-légère"
# columns are dropped, so "Produits pétroliers" is all oil products.
# "Autres énergies renouvelables" = sun, wind, biogas, ambient heat (BFE footnote).
BFE_GROUPS = {
    "heating-oil": ".*Produits pétroliers.*",
    "gas": ".*Gaz.*",
    "coal": ".*Charbon.*",
    "wood": ".*bois.*",
    "district-heating": ".*distance.*",
    "renewables": ".*renouvelables.*",
    "electricity": ".*Electricité.*",
}
# "Déchets industriels" (industrial waste) is ~300 TJ/yr for services since 2016. For
# households it is "-" in every year, and the extractor drops columns with no numbers,
# so it is only mapped for services.
BFE_EXTRA_GROUPS = {"services": {"other": ".*Déchets industriels.*"}}

# FSO energy accounts of households and the economy (STAT-TAB).
FSO_TABLE = "px-x-0204000000_106"
FSO_CATEGORY = "--- Ménages: chauffage"
FSO_FILE = os.path.join(DATA_DIR, "fso_household_heating_TJ.pickle")
FSO_GROUPS = {
    "heating-oil": ["1.1.2. Huile de chauffage extra-légère"],
    "coal": ["1.2. Charbon"],
    "gas": ["1.3. Gaz naturel"],
    "district-heating": [
        "2. Déchets (hors biomasse)",
        "3.1. Déchets (biomasse)",
        "4. Combustibles nucléaires",
        "6. Chaleur à distance",
    ],
    "wood": ["3.2. Bois et charbon de bois"],
    "renewables": ["3.4. Géothermie, chaleur ambiante et énergie solaire thermique"],
    "electricity": ["5. Electricité"],
    "other": ["3.3. Biogaz et biocarburants"],
}

# EP2050+ demand-sector details (BFE publication 10441, a zip of Excel files). The
# buildings pre-processing extracts it to its data folder; if that copy is missing,
# the zip is downloaded and only the services workbook is kept in scripts/data/.
EP2050_URL = "https://pubdb.bfe.admin.ch/de/publication/download/10441"
EP2050_SERVICES_NAME = (
    "EP2050+_Detailergebnisse 2020-2060_Dienstleistung_alle Szenarien_2022-10-20.xlsx"
)
EP2050_PREPROC_FILE = os.path.join(
    HERE,
    "../transition_compass_model/_database/pre_processing/buildings/Switzerland/data",
    "EP2050_sectors/EP2050+_Szenarienergebnisse_Details_Nachfragesektoren",
    EP2050_SERVICES_NAME,
)
EP2050_SERVICES_FILE = os.path.join(DATA_DIR, "EP2050_Dienstleistung.xlsx")
# Tabelle 03-01: useful space heat of services, ZERO scenarios, kWh/m².
EP2050_HEAT_TABLE = ("03 Nutzenergie RW", "Tabelle 03-01", "Baujahr < 2020")
# Tabelle 02-02: heating systems in buildings built before 2010, ZERO Basis
# (the historical years are the same in all scenarios). Row label -> model group.
EP2050_MIX_TABLE = ("02 Beheizungsstruktur", "Tabelle 02-02")
EP2050_MIX_GROUPS = {
    "Heizöl": "heating-oil",
    "Gas*": "gas",
    "Holz": "wood",
    "Strom": "electricity",
    "Wärmepumpen": "heat-pump",
    "Fernwärme/Nahwärme": "district-heating",
    "sonstige/keine Heizung": "other",
}
# Model heating technologies -> heating-mix groups ("other": coal, solar, other-tech)
MODEL_TO_MIX_GROUP = {
    "heating-oil": "heating-oil",
    "gas": "gas",
    "wood": "wood",
    "electricity": "electricity",
    "heat-pump": "heat-pump",
    "district-heating": "district-heating",
    "coal": "other",
    "solar": "other",
    "other-tech": "other",
}
MIX_ORDER = list(dict.fromkeys(EP2050_MIX_GROUPS.values()))
NONRES_TYPES = ["education", "health", "hotels", "offices", "other", "trade"]
TWH_PER_PJ = 1 / 3.6

# --- Model fuels -> official fuel groups --------------------------------------
# "heat-pump" is the electricity used by heat pumps; "ambient-heat" is the heat they
# take from air/ground. "other-tech" is 0 in the historical years.
# The model has no fuel for the official "other" group (industrial waste, biogas).
MODEL_TO_GROUP = {
    "heating-oil": "heating-oil",
    "gas": "gas",
    "coal": "coal",
    "wood": "wood",
    "district-heating": "district-heating",
    "electricity": "electricity",
    "heat-pump": "electricity",
    "solar": "renewables",
    "ambient-heat": "renewables",
    "other-tech": "renewables",
}
GROUP_ORDER = [
    "heating-oil",
    "gas",
    "coal",
    "wood",
    "district-heating",
    "renewables",
    "electricity",
    "other",
    "total",
]

COLOR_MODEL = "#2a78d6"
COLOR_OFFICIAL = "#eb6834"


# --- Official data ------------------------------------------------------------
def load_bfe_buildings_energy():
    """BFE final energy of households and services by fuel group [TWh].

    Returns {"households": df, "services": df}, each Years x fuel group.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    out = {}
    for name, sheet in BFE_SHEETS.items():
        var = f"bld_energy-demand_{name}"
        parameters = {
            # copied: the extractor edits the mapping in place
            "mapping": {**BFE_GROUPS, **BFE_EXTRA_GROUPS.get(name, {})},
            "var name": var,
            "headers indexes": (8, 9, 10),  # rows combined into the header
            "first row": 11,
            "unit": None,  # unit is read from the header (TJ)
            "cols to drop": ".*%.*|.*extra.*",
        }
        dm = extract_energy_statistics_data(BFE_URL, BFE_FILE, sheet, parameters, YEARS)
        out[name] = _dm_to_df(dm, var) / TJ_PER_TWH
    return out


def load_fso_household_heating():
    """FSO household heating (incl. hot water) by fuel group [TWh]. Years x group."""
    if os.path.exists(FSO_FILE):
        with open(FSO_FILE, "rb") as f:
            dm = pickle.load(f)
    else:
        structure, _ = get_data_api_CH(FSO_TABLE, mode="example", language="fr")
        dm = get_data_api_CH(
            FSO_TABLE,
            mode="extract",
            mapping_dims={
                "Country": "Unité de mesure",
                "Years": "Année",
                "Variables": "Économie et ménages",
                "Categories1": "Agent énergétique",
            },
            filter={
                "Économie et ménages": [FSO_CATEGORY],
                "Unité de mesure": ["Terajoules"],
                "Année": structure["Année"],
                "Agent énergétique": structure["Agent énergétique"],
            },
            units=["TJ"],
            language="fr",
        )
        dm.rename_col(FSO_CATEGORY, "bld_heating-demand", dim="Variables")
        dm.rename_col("Terajoules", COUNTRY, dim="Country")
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(FSO_FILE, "wb") as f:
            pickle.dump(dm, f, protocol=pickle.HIGHEST_PROTOCOL)

    dm = dm.groupby(FSO_GROUPS, dim="Categories1", inplace=False)
    df = _dm_to_df(dm, "bld_heating-demand") / TJ_PER_TWH
    return df.loc[df.index.isin(YEARS)]


def ep2050_services_workbook():
    """Path to the EP2050 services workbook (pre-processing copy, else download)."""
    if os.path.exists(EP2050_PREPROC_FILE):
        return EP2050_PREPROC_FILE
    if not os.path.exists(EP2050_SERVICES_FILE):
        os.makedirs(DATA_DIR, exist_ok=True)
        zip_path = os.path.join(DATA_DIR, "EP2050_sectors.zip")
        save_url_to_file(EP2050_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            member = next(n for n in z.namelist() if n.endswith(EP2050_SERVICES_NAME))
            with z.open(member) as src, open(EP2050_SERVICES_FILE, "wb") as dst:
                shutil.copyfileobj(src, dst)
        os.remove(zip_path)  # the other workbooks are not needed here
    return EP2050_SERVICES_FILE


def _ep2050_table(sheet, table, labels):
    """Rows `labels` of an EP2050 table as a DataFrame Years x label.

    The table starts at the row whose column 1 starts with `table` (e.g.
    "Tabelle 03-01"); the year header is the next row that contains 2000, and each
    label is the first row after it with that text in column 1.
    """
    df = pd.read_excel(ep2050_services_workbook(), sheet_name=sheet, header=None)
    col1 = df.iloc[:, 1].astype(str).str.strip()
    start = col1.index[col1.str.startswith(table)][0]
    head = next(i for i in df.index[df.index > start] if 2000 in df.loc[i].values)
    years = {
        int(v): c
        for c, v in df.loc[head].items()
        if isinstance(v, (int, float)) and not pd.isna(v) and int(v) in YEARS
    }
    out = {}
    for label in labels:
        row = col1.index[(col1 == label) & (col1.index > head)][0]
        out[label] = [float(df.loc[row, c]) for c in years.values()]
    return pd.DataFrame(out, index=list(years)).dropna(how="all")


def load_ep2050_services_useful_heat():
    """EP2050 useful space heat of services per m² [kWh/m²]. Years x ["services"]."""
    sheet, table, label = EP2050_HEAT_TABLE
    return _ep2050_table(sheet, table, [label]).set_axis(["services"], axis=1)


def load_ep2050_services_heating_mix():
    """EP2050 share of heating systems, services built before 2010. Years x group."""
    sheet, table = EP2050_MIX_TABLE
    df = _ep2050_table(sheet, table, list(EP2050_MIX_GROUPS))
    df = df.rename(columns=EP2050_MIX_GROUPS)
    df["other"] = df["other"].clip(lower=0)  # tiny negative values in the source
    return df.div(df.sum(axis=1), axis=0)


def load_prognos_services_space_heating():
    """Prognos services space heating, final energy [TWh] (incl. agriculture)."""
    pj = PROGNOS_SERVICES_EUD_PJ["space-heating"]
    return pd.DataFrame(
        {"services": [v * TWH_PER_PJ for v in pj.values()]}, index=list(pj)
    )


# --- Model --------------------------------------------------------------------
def run_model():
    """Run climate -> lifestyles -> buildings for Switzerland.

    Returns (TPE buildings DataMatrix, buildings->energy interface dict, historical
    heating-mix input DataMatrix).
    """
    lever_setting = {k: math.floor(v) for k, v in load_lever_config().items()}
    sectors = ["climate", "lifestyles", "buildings"]
    DM_input = filter_country_and_load_data_from_pickles(
        country_list=[COUNTRY], modules_list=sectors
    )
    # Copied before the run: buildings() appends the future years to the ots data
    dm_heating_mix = DM_input["buildings"]["ots"]["heating-technology-fuel"][
        "bld_heating-technology"
    ].copy()
    interface = Interface()
    climate(lever_setting, YEARS_SETTING, DM_input["climate"], interface)
    lifestyles(lever_setting, YEARS_SETTING, DM_input["lifestyles"], interface)
    dm_tpe, _ = buildings(
        lever_setting, YEARS_SETTING, DM_input["buildings"], interface
    )
    return dm_tpe, interface.get_link("buildings", "energy"), dm_heating_mix


def model_all_uses(dm_tpe, fuels, building_use):
    """Model final energy, all uses, by model fuel [TWh]. building_use is
    "residential" or "non-residential"."""
    var_of = {f: f"bld_energy-demand_{building_use}_{f}" for f in fuels}
    dm = dm_tpe.filter({"Country": [COUNTRY], "Variables": list(var_of.values())})
    idx = dm.idx
    df = pd.DataFrame(
        {f: dm.array[0, :, idx[v]] for f, v in var_of.items()},
        index=[int(y) for y in dm.col_labels["Years"]],
    )
    return df


def model_household_heating_hot_water(link):
    """Model household space heating + hot water by model fuel [TWh]."""
    df_heat = _dm_to_df(link["households_heating"], "bld_energy-demand_heating")
    df_hw = _dm_to_df(link["households_hot-water"], "bld_hot-water_energy-demand")
    return df_heat.add(df_hw, fill_value=0)


def _tpe_series(dm_tpe, var):
    """One TPE variable for COUNTRY as a Series over years."""
    dm = dm_tpe.filter({"Country": [COUNTRY], "Variables": [var]})
    return pd.Series(dm.array[0, :, 0], index=[int(y) for y in dm.col_labels["Years"]])


def model_services_useful_heat(dm_tpe):
    """Model useful space heat of services per occupied m² [kWh/m²].

    Returns (Years x ["services"], Years x building type). TWh/Mm² x 1000 = kWh/m².
    """
    heat = pd.DataFrame(
        {t: _tpe_series(dm_tpe, f"bld_useful-heat_{t}") for t in NONRES_TYPES}
    )
    area = pd.DataFrame(
        {t: _tpe_series(dm_tpe, f"bld_floor-area-occupied_{t}") for t in NONRES_TYPES}
    )
    total = pd.DataFrame({"services": heat.sum(axis=1) / area.sum(axis=1) * 1000})
    return total, heat / area * 1000


def model_services_space_heating(link):
    """Model services space heating, final energy [TWh], ambient heat excluded."""
    df = _dm_to_df(link["services_space-heating"], "bld_energy-demand_heating")
    return pd.DataFrame({"services": df.drop(columns="ambient-heat").sum(axis=1)})


def model_services_heating_mix(dm_mix, dm_tpe):
    """Model input heating mix of services, weighted by floor area by class.

    Services copy the multi-family mix, so the mix is the same for all services
    types; it is weighted by the services floor area of each energy class.
    Returns Years x heating-mix group (shares sum to 1).
    """
    dm = dm_mix.filter({"Country": [COUNTRY], "Categories1": NONRES_TYPES})
    arr = dm.array[0, :, 0]  # years x type x class x tech
    if not np.allclose(arr, arr[:, :1], equal_nan=True):
        raise ValueError("Services types have different heating mixes: weight by type")
    classes = dm.col_labels["Categories2"]
    years = [int(y) for y in dm.col_labels["Years"]]
    area = pd.DataFrame(
        {
            c: _tpe_series(dm_tpe, f"bld_floor-area_stock_non-residential_{c}")
            for c in classes
        }
    ).reindex(years)
    weights = area.div(area.sum(axis=1), axis=0).to_numpy()  # years x class
    mix = np.einsum("yc,yct->yt", weights, np.nan_to_num(arr[:, 0]))
    df = pd.DataFrame(mix, index=years, columns=dm.col_labels["Categories3"])
    df = df.T.groupby(MODEL_TO_MIX_GROUP).sum().T
    return df.reindex(columns=MIX_ORDER).loc[lambda d: d.index.isin(YEARS)]


def to_groups(df_model):
    """Sum model fuels into official fuel groups. Fails on an unmapped fuel."""
    unmapped = sorted(set(df_model.columns) - set(MODEL_TO_GROUP))
    if unmapped:
        raise KeyError(f"Model fuels with no official group: {unmapped}")
    return df_model.T.groupby(MODEL_TO_GROUP).sum().T


# --- Comparison ---------------------------------------------------------------
def _dm_to_df(dm, var):
    """Country=COUNTRY, one variable, Categories1 -> DataFrame Years x category."""
    dm = dm.filter({"Country": [COUNTRY], "Variables": [var]})
    return pd.DataFrame(
        dm.array[0, :, 0, :],
        index=[int(y) for y in dm.col_labels["Years"]],
        columns=dm.col_labels["Categories1"],
    )


def align(df_official, df_model):
    """Same years (both sources) and same groups (missing -> 0), plus a total."""
    years = sorted(set(df_official.index) & set(df_model.index))
    groups = [g for g in GROUP_ORDER if g in set(df_official) | set(df_model)]
    out = [
        df.reindex(index=years, columns=groups).fillna(0.0)
        for df in (df_official, df_model)
    ]
    # Drop groups that are zero in both sources (e.g. FSO biogas for households)
    keep = [g for g in groups if (out[0][g].abs() + out[1][g].abs()).sum() > 0]
    for df in out:
        df.drop(columns=[g for g in groups if g not in keep], inplace=True)
        df["total"] = df.sum(axis=1)
    return out


def summary_table(official, model, years=None):
    """Official, model and ratio by fuel group, for `years` (default KEY_YEARS) +
    the average."""
    years = [y for y in (years or KEY_YEARS) if y in official.index]
    label = f"avg {official.index[0]}-{official.index[-1]}"
    rows = {}
    for g in official.columns:
        o = official[g].loc[years].tolist() + [official[g].mean()]
        m = model[g].loc[years].tolist() + [model[g].mean()]
        for kind, vals in (("official", o), ("model", m)):
            rows[(g, kind)] = vals
        rows[(g, "model/official")] = [
            mv / ov if ov > 1e-6 else np.nan for mv, ov in zip(m, o)
        ]
    df = pd.DataFrame(rows, index=years + [label]).T
    df.index.names = ["fuel", ""]
    return df


def comparison_figure(comparisons):
    """Rows = fuel groups, columns = comparisons; official dashed, model solid."""
    groups = [
        g
        for g in GROUP_ORDER
        if any(g in off.columns for off, _ in comparisons.values())
    ]
    titles = list(comparisons)
    fig = make_subplots(
        rows=len(groups),
        cols=len(titles),
        shared_xaxes="all",  # same years in every column
        column_titles=titles,
        row_titles=groups,
        vertical_spacing=0.025,
        horizontal_spacing=0.06,
    )
    for c, (title, (official, model)) in enumerate(comparisons.items(), start=1):
        for r, g in enumerate(groups, start=1):
            if g not in official.columns:
                continue
            for name, df, dash, color in (
                ("official", official, "dash", COLOR_OFFICIAL),
                ("model", model, "solid", COLOR_MODEL),
            ):
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[g],
                        mode="lines",
                        name=name,
                        legendgroup=name,
                        showlegend=(r == 1 and c == 1),
                        line={"width": 2, "dash": dash, "color": color},
                        hovertemplate=f"{name}: %{{y:.2f}} TWh<extra>{g}</extra>",
                    ),
                    row=r,
                    col=c,
                )
            fig.update_yaxes(rangemode="tozero", row=r, col=c)
    fig.update_yaxes(title_text="TWh", col=1)
    fig.update_layout(
        title=f"Buildings final energy, {COUNTRY}: model vs official statistics",
        template="plotly_white",
        hovermode="x unified",
        height=230 * len(groups),
        # legend on its own line, between the main title and the column titles
        legend={"orientation": "h", "x": 0, "y": 1.012, "yanchor": "bottom"},
        margin={"t": 120},
    )
    return fig


def mix_table(official, model):
    """Heating-mix shares [%], official vs model, and the difference in points."""
    years = [y for y in KEY_YEARS if y in official.index and y in model.index]
    rows = {}
    for g in MIX_ORDER:
        o = official.loc[years, g] * 100
        m = model.loc[years, g] * 100
        rows[(g, "official %")] = o.tolist()
        rows[(g, "model %")] = m.tolist()
        rows[(g, "model-official")] = (m - o).tolist()
    df = pd.DataFrame(rows, index=years).T
    df.index.names = ["system", ""]
    return df


def services_heating_figure(heat, heat_by_type, space_heating, mix):
    """Services space-heating checks: useful heat per m², final energy, mix."""
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Useful heat per m² [kWh/m²]",
            "Space heating, final energy [TWh]",
            "Heating mix, share of systems [%]",
        ],
        horizontal_spacing=0.07,
    )
    for c, (official, model, name_off, unit) in enumerate(
        [
            (*heat, "EP2050", "kWh/m²"),
            (*space_heating, "Prognos (incl. agriculture)", "TWh"),
        ],
        start=1,
    ):
        for name, df, dash, color, mode in (
            (name_off, official, "dash", COLOR_OFFICIAL, "lines+markers"),
            ("model", model, "solid", COLOR_MODEL, "lines"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["services"],
                    mode=mode,
                    name=name,
                    line={"width": 2, "dash": dash, "color": color},
                    hovertemplate=f"{name}: %{{y:.1f}} {unit}<extra></extra>",
                ),
                row=1,
                col=c,
            )
        fig.update_yaxes(rangemode="tozero", row=1, col=c)
    # Model by building type, thin and hidden by default (click the legend)
    for t in heat_by_type.columns:
        fig.add_trace(
            go.Scatter(
                x=heat_by_type.index,
                y=heat_by_type[t],
                mode="lines",
                name=f"model {t}",
                visible="legendonly",
                line={"width": 1},
                hovertemplate=f"{t}: %{{y:.0f}} kWh/m²<extra></extra>",
            ),
            row=1,
            col=1,
        )
    palette = ["#6b4c3b", "#e0a030", "#8db255", "#7a5fb0", "#2fa59b", "#d6453d", "#999"]
    official, model = mix
    for g, color in zip(MIX_ORDER, palette):
        for name, df, dash in (("EP2050", official, "dash"), ("model", model, "solid")):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[g] * 100,
                    mode="lines",
                    name=f"{g} ({name})",
                    legendgroup=g,
                    line={"width": 2, "dash": dash, "color": color},
                    hovertemplate=f"{g} {name}: %{{y:.1f}}%<extra></extra>",
                ),
                row=1,
                col=3,
            )
    fig.update_layout(
        title=(
            f"Services space heating, {COUNTRY}: model (solid) vs official (dashed)"
        ),
        template="plotly_white",
        height=520,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", default=OUT_DIR, help="output directory for the HTML plot"
    )
    args = parser.parse_args()

    print("Loading official data")
    bfe = load_bfe_buildings_energy()
    fso = load_fso_household_heating()

    print("Running climate -> lifestyles -> buildings")
    dm_tpe, link, dm_heating_mix = run_model()
    fuels = link["households_heating"].col_labels["Categories1"]

    comparisons = {
        "BFE households, all uses": align(
            bfe["households"], to_groups(model_all_uses(dm_tpe, fuels, "residential"))
        ),
        "BFE services, all uses": align(
            bfe["services"],
            to_groups(model_all_uses(dm_tpe, fuels, "non-residential")),
        ),
        "FSO household heating + hot water": align(
            fso, to_groups(model_household_heating_hot_water(link))
        ),
    }

    # Sanity check: household heating + hot water is part of household all uses.
    hh_all = comparisons["BFE households, all uses"][1]["total"]
    hh_heat = comparisons["FSO household heating + hot water"][1]["total"]
    excess = (hh_heat - hh_all.reindex(hh_heat.index)).max()
    if excess > 1e-6:
        print(
            f"WARNING: model heating + hot water exceeds all uses by {excess:.3f} TWh"
        )

    with pd.option_context("display.width", 200, "display.max_rows", 200):
        for title, (official, model) in comparisons.items():
            print(f"\n=== {title} [TWh/yr] ===")
            print(summary_table(official, model).round(2).to_string())
    print(
        "\nNotes: 'other' is official only (BFE industrial waste; FSO biogas/"
        "biofuels).\nBFE 'renewables' includes ambient heat used by heat pumps for "
        "hot water; the model\ncounts no ambient heat for hot water."
    )

    # --- Services space heating ----------------------------------------------
    print("\nLoading EP2050 services workbook")
    model_heat, model_heat_by_type = model_services_useful_heat(dm_tpe)
    ep_heat = load_ep2050_services_useful_heat()
    heat = (ep_heat, model_heat.loc[ep_heat.index])
    prognos = load_prognos_services_space_heating()
    model_sh = model_services_space_heating(link)
    space_heating = (prognos, model_sh.loc[prognos.index])
    ep_mix = load_ep2050_services_heating_mix()
    model_mix = model_services_heating_mix(dm_heating_mix, dm_tpe)
    mix = (ep_mix, model_mix)

    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print("\n=== Services useful heat per m², EP2050 [kWh/m²/yr] ===")
        print(summary_table(*heat).round(1).to_string())
        y = YEARS[-1]
        print(f"\nModel useful heat per m² by building type, {y} [kWh/m²/yr]:")
        print(model_heat_by_type.loc[y].round(0).to_string())
        print("\n=== Services space heating, final energy, Prognos [TWh/yr] ===")
        print(
            summary_table(*space_heating, years=list(prognos.index))
            .round(2)
            .to_string()
        )
        print("\n=== Services heating mix, share of systems, EP2050 [%] ===")
        print(mix_table(*mix).round(1).to_string())
    print(
        "\nNotes: useful heat = heat lost by the building, before boiler/heat-pump "
        "losses;\nthe model computes it from U-value x envelope area x heating degree "
        "days.\nEP2050 kWh/m² is for the stock built before 2020; the model is not "
        "weather-corrected.\nPrognos space heating includes agriculture (a few %). "
        "Model final energy excludes\nambient heat. Model heating mix = input mix "
        "(the multi-family mix) weighted by\nservices floor area by class; EP2050 = "
        "systems in buildings built before 2010."
    )

    os.makedirs(args.out, exist_ok=True)
    f = os.path.join(args.out, "buildings_validation_CH.html")
    figs = [
        comparison_figure(comparisons),
        services_heating_figure(heat, model_heat_by_type, space_heating, mix),
    ]
    with open(f, "w") as fh:
        fh.write("<html><head><meta charset='utf-8'></head><body>\n")
        for i, fig in enumerate(figs):
            fh.write(fig.to_html(full_html=False, include_plotlyjs=(i == 0)))
        fh.write("</body></html>\n")
    print(f"\nPlot written to:\n  {os.path.abspath(f)}")


if __name__ == "__main__":
    main()
