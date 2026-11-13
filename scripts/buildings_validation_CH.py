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

Prints one table per series (official, model, model/official) and writes an
interactive plotly chart: one row per fuel group, one column per series.

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
from transition_compass_model.model.buildings_module import buildings
from transition_compass_model.model.climate_module import climate
from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    filter_country_and_load_data_from_pickles,
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


# --- Model --------------------------------------------------------------------
def run_model():
    """Run climate -> lifestyles -> buildings for Switzerland.

    Returns (TPE buildings DataMatrix, buildings->energy interface dict).
    """
    lever_setting = {k: math.floor(v) for k, v in load_lever_config().items()}
    sectors = ["climate", "lifestyles", "buildings"]
    DM_input = filter_country_and_load_data_from_pickles(
        country_list=[COUNTRY], modules_list=sectors
    )
    interface = Interface()
    climate(lever_setting, YEARS_SETTING, DM_input["climate"], interface)
    lifestyles(lever_setting, YEARS_SETTING, DM_input["lifestyles"], interface)
    dm_tpe, _ = buildings(
        lever_setting, YEARS_SETTING, DM_input["buildings"], interface
    )
    return dm_tpe, interface.get_link("buildings", "energy")


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


def summary_table(official, model):
    """Official, model and ratio by fuel group, for KEY_YEARS + the average."""
    years = [y for y in KEY_YEARS if y in official.index]
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
    dm_tpe, link = run_model()
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

    os.makedirs(args.out, exist_ok=True)
    f = os.path.join(args.out, "buildings_validation_CH.html")
    comparison_figure(comparisons).write_html(f)
    print(f"\nPlot written to:\n  {os.path.abspath(f)}")


if __name__ == "__main__":
    main()
