import math

import pandas as pd
import plotly.graph_objects as go
import Vaud.read_data as read_data
from plotly.subplots import make_subplots
from Vaud.plots import from_unit_to_other_unit_and_total, plot_comparaison

from transition_compass_model.model.climate_module import climate
from transition_compass_model.model.common.auxiliary_functions import (
    filter_country_and_load_data_from_pickles,
)
from transition_compass_model.model.common.config_loader import load_lever_config
from transition_compass_model.model.common.interface_class import Interface
from transition_compass_model.model.lifestyles_module import lifestyles
from transition_compass_model.model.transport_module import transport
from transition_compass_model.model.transport_module import wkf as transport_wkf

COLOR_MODEL = "#2a78d6"
COLOR_OFFICIAL = "#eb6834"

YEARS_SETTING = [1990, 2023, 2025, 2050, 5]
# --- Model --------------------------------------------------------------------
MODEL_MODE_TO_FOEN = {
    ("passenger", "LDV"): "LDV",
    ("passenger", "2W"): "2W",
    ("passenger", "bus"): "bus",
    ("freight", "HDV"): "trucks",
    ("passenger", "rail"): "rail",
    ("passenger", "metrotram"): "rail",
    ("freight", "rail"): "rail",
    ("freight", "IWW"): "marine",
    ("freight", "marine"): "marine",
}


def run_model(country):
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
        country_list=[country], modules_list=sectors
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
    df = dm_tpe.filter({"Country": [country]}).write_df()
    df = df.drop(columns="Country").set_index("Years")
    df.index = df.index.astype(int)
    return df, tech


def _tpe_block(tpe, prefix, unit):
    """TPE columns "<prefix>_<cat>[<unit>]" as a DataFrame Years x cat."""
    start, end = f"{prefix}_", f"[{unit}]"
    cols = [c for c in tpe.columns if c.startswith(start) and c.endswith(end)]
    if not cols:
        raise KeyError(f"No TPE variable {start}*{end}")
    return tpe[cols].rename(columns=lambda c: c[len(start) : -len(end)])


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

    cols_by_kind = {"passenger": {}, "freight": {}}
    for (kind, mode), foen in MODEL_MODE_TO_FOEN.items():
        s = tpe[f"tra_emissions-CO2e_{kind}_{mode}[Mt]"]
        cols_by_kind[kind][foen] = (
            cols_by_kind[kind][foen] + s if foen in cols_by_kind else s
        )

    by_mode_by_kind = {"passenger": {}, "freight": {}}
    for kind in cols_by_kind.keys():
        by_mode_by_kind[kind] = pd.DataFrame(cols_by_kind[kind])

    # Aviation with passenger flights counted outbound only (about the flights
    # leaving Swiss airports, like the inventory) + freight aviation.
    aviation_local = (
        tpe["tra_passenger-emissions-local_aviation[Mt]"]
        + tpe["tra_emissions-CO2e_freight_aviation[Mt]"]
    )
    return by_mode, by_kind, by_mode_by_kind, aviation_local


def fribourg_values():
    TOTAL_EMISSION = 5504411 * 10**-6  # MtCO2eq
    # Proportion from the bilan carbone
    emi_by_mode = {
        "LDV": 0.0738,
        "2W": 0.0043,
        "bus": 0.0027,
        "trucks": 0.0204,  # sum of heavy and low HDV
        "rail": 0.0002,
        "marine": 0,
        "military": 0.0001,
    }

    for mode, prop in emi_by_mode.items():
        emi_by_mode[mode] = prop * TOTAL_EMISSION
    total_sum = sum(emi_by_mode.values())
    emi_by_mode["total"] = {
        2023: 0.55,  # MTCO2eq
        2017: 0.63,
    }
    assert (total_sum - emi_by_mode["total"][2023]) < 1 * 10**-2

    return emi_by_mode


def plot_emissions_comparison(m_by_mode, emi_by_mode):
    # Transport modes to compare
    modes = ["LDV", "2W", "bus", "trucks", "rail", "marine"]

    # Model results: convert fractions to percentages
    model = m_by_mode.loc[[2023], modes] * 100

    # Reference emissions
    reference = {mode: emi_by_mode[mode] for mode in modes}

    # Create comparison figure
    fig = go.Figure()

    # 2023
    fig.add_trace(go.Bar(x=modes, y=model.loc[2023], name="Model 2023"))

    # Reference
    fig.add_trace(
        go.Bar(x=modes, y=[reference[mode] for mode in modes], name="Reference")
    )

    fig.update_layout(
        title="Transport emissions by mode",
        xaxis_title="Transport mode",
        yaxis_title="Emissions (%)",
        barmode="group",
        template="plotly_white",
        width=1000,
        height=550,
        legend_title="Dataset",
        hovermode="x unified",
    )

    fig.show()

    # --------------------------------
    # Percentage difference (2023)
    # --------------------------------

    difference = (
        (model.loc[2023] - __import__("pandas").Series(reference))
        / __import__("pandas").Series(reference)
        * 100
    )

    fig_ratio = go.Figure()

    fig_ratio.add_trace(
        go.Bar(
            x=modes,
            y=difference,
            text=[f"{v:.1f}%" if abs(v) < float("inf") else "N/A" for v in difference],
            textposition="outside",
            name="Difference 2023 vs Reference",
        )
    )

    fig_ratio.add_hline(y=0, line_color="black", line_width=1)

    fig_ratio.update_layout(
        title="Relative difference between model 2023 and reference",
        xaxis_title="Transport mode",
        yaxis_title="Difference (%)",
        template="plotly_white",
        width=1000,
        height=500,
    )

    fig_ratio.show()

    return fig, fig_ratio


def plot_tra_emissions_comparison(
    emissions_fribourg_model,
    EMISSION_FRIBOURG,
    categories=["services", "residential", "total"],
):
    n_cols = 3
    n_categories = len(categories)

    # Determine subplot layout
    n_rows = math.ceil(n_categories / n_cols)
    fig = make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=categories, shared_yaxes=True
    )

    colors = {
        "Model": COLOR_MODEL,
        "Fribourg": COLOR_OFFICIAL,
    }

    for i, category in enumerate(categories, start=1):
        row = (i - 1) // n_cols + 1
        col = (i - 1) % n_cols + 1
        # Model values
        years = [2017, 2023]
        model_values = [emissions_fribourg_model[category][year] for year in years]

        fig.add_trace(
            go.Bar(
                x=years,
                y=model_values,
                name="Transition Compass",
                marker_color=colors["Model"],
                showlegend=(i == 1),
                text=[f"{v:.3f}" for v in model_values],
                textposition="outside",
            ),
            row=row,
            col=col,
        )

        # Fribourg reference values
        if category == "total":
            reference_years = [2017, 2023]
            reference_values = [
                EMISSION_FRIBOURG["total"][year] for year in reference_years
            ]
        else:
            reference_years = [2023]
            reference_values = [EMISSION_FRIBOURG[category]]

        fig.add_trace(
            go.Bar(
                x=reference_years,
                y=reference_values,
                name="Fribourg",
                marker_color=colors["Fribourg"],
                showlegend=(i == 1),
                text=[f"{v:.3f}" for v in reference_values],
                textposition="outside",
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title="Comparaison des émissions par mode de transport",
        title_x=0.5,
        template="plotly_white",
        barmode="group",
        height=350 * n_rows,
        width=1100,
        legend_title="Modèle",
        font=dict(size=13),
    )

    # Determine global maximum across all categories
    all_values = []

    for category in categories:
        all_values.extend(
            emissions_fribourg_model[category][year] for year in [2017, 2023]
        )

        if category == "total":
            all_values.extend(EMISSION_FRIBOURG["total"].values())
        else:
            all_values.append(EMISSION_FRIBOURG[category])

    # Set common Y-axis range with 15% margin
    y_max = max(all_values) * 1.15

    fig.update_yaxes(
        range=[0, y_max],
        title_text="",
    )
    for row in range(1, n_rows + 1):
        fig.update_yaxes(
            title_text="Emissions [Mt CO2 eq]",
            row=row,
            col=1,
        )

    fig.update_xaxes(
        title_text="Year",
        tickvals=list(years),
    )

    fig.show()

    return fig


def fribourg_run():
    country = "Fribourg"
    tpe, tech = run_model(country)

    # --- Emissions by mode model ----------------------------------------------------
    years_list = [2017, 2023]
    m_by_mode, m_by_kind, m_by_kind_by_mode, aviation_local = model_emissions(tpe)
    m_by_mode = m_by_mode.loc[years_list, :]
    # emissions bilan carbone
    emi_by_mode_fribourg = fribourg_values()
    m_by_mode["total"] = m_by_mode.sum(axis=1)

    plot_tra_emissions_comparison(
        m_by_mode, emi_by_mode_fribourg, categories=m_by_mode.columns
    )

    print("finish")


def schwyz_tra_values():
    df_total = pd.read_excel("data/Schwyz/Eco_speed_sectors.xlsx")
    # get transport data
    df_bld_tot = df_total[
        (df_total["Bereiche"] == "Verkehr")
        | (df_total["Bereiche"] == "Kommunale Flotte")
    ]

    df_bld_tot.iloc[:, 1:] = df_bld_tot.iloc[:, 1:] * 10**-6
    df_tot = df_bld_tot.sum(axis=0)

    return df_tot.iloc[1:,].rename("total")


import plotly.express as px


def compare_series_plot(
    x,
    series1,
    series2,
    name1="Series 1",
    name2="Series 2",
    title="Comparison of two series",
):
    colors = {name1: COLOR_MODEL, name2: COLOR_OFFICIAL}

    df = pd.DataFrame(
        {
            "X": list(x) * 2,
            "Value": list(series1) + list(series2),
            "Series": [name1] * len(series1) + [name2] * len(series2),
        }
    )

    fig = px.line(
        df,
        x="X",
        y="Value",
        color="Series",
        color_discrete_map=colors,
        title=title,
        markers=True,
        template="plotly_white",
    )

    fig.update_layout(
        xaxis_title="Years",
        yaxis_title="Emissions [Mt CO2eq]",
        hovermode="x unified",
        legend_title="",
    )

    fig.show()

    return fig


import numpy as np


def compare_series_statistics(serie1, serie2):
    # Calculate the difference
    diff = serie1 - serie2

    # Calculate mean and maximum
    mean_diff = diff.mean(skipna=True)
    max_diff = diff.max()

    print("Mean difference:", mean_diff)
    print("Maximum difference:", max_diff)
    # Calculate proportional difference in %
    diff_pct = ((serie1 - serie2) / serie2) * 100

    # Calculate mean and maximum
    mean_diff_pct = diff_pct.mean()
    max_diff_pct = diff_pct.max()

    print("Mean proportional difference:", mean_diff_pct, "%")
    print("Maximum proportional difference:", max_diff_pct, "%")


def schwyz_run():
    country = "Schwyz"
    tpe, tech = run_model(country)
    # --- Emissions by mode model ----------------------------------------------------
    YEARS_SCHWYZ_BILAN = [y for y in range(2010, 2024)] + [2025]
    m_by_mode, m_by_kind, m_by_kind_by_mode, aviation_local = model_emissions(tpe)
    m_by_mode = m_by_mode.loc[YEARS_SCHWYZ_BILAN, :]
    m_by_mode["total"] = m_by_mode.sum(axis=1)
    serie_schwyz = schwyz_tra_values().astype(float)
    serie_schwyz.index = serie_schwyz.index.astype(int)

    m_by_mode.loc[2024, "total"] = np.nan
    m_by_mode.sort_index(inplace=True)
    compare_series_plot(
        x=serie_schwyz.index,
        series1=serie_schwyz,
        series2=m_by_mode["total"],
        name1="Schwyz",
        name2="Transition Compass",
        title="CO2 emissions of the transport sector",
    )

    compare_series_statistics(m_by_mode["total"], serie_schwyz)
    print("end schwyz")


def vaud_run():
    # Get bilan carbone values
    bilan_carbone_tot, _, groups, DIC_OFF_TO_MODEL = read_data.read_vaud_data()
    dummy_dic = {
        "moto/scooter": "2W",
        "voiture": "LDV",
        "bus ": "bus",
        "train": "rail",
        "total": "total",
    }
    groups_transports_emission = bilan_carbone_tot[
        bilan_carbone_tot["Catégorie"].isin(dummy_dic.keys())
    ].groupby(["Catégorie"])

    # Get model value
    country = "Vaud"
    tpe, tech = run_model(country)

    m_by_mode, m_by_kind, m_by_kind_by_mode, aviation_local = model_emissions(tpe)

    for kind in ["freight", "passenger"]:
        # Mt to t
        m_by_kind_by_mode[kind] = from_unit_to_other_unit_and_total(
            m_by_kind_by_mode[kind], 10**6
        )

    plot_comparaison(
        m_by_kind_by_mode["passenger"],
        col_to_plot_vaud="CO2",
        legend_label="Émissions de GES [tCo2eq]",
        groups=groups_transports_emission,
        dic_off_to_model=dummy_dic,
    )
    bilan_carbone_tot = pd.read_excel(
        "data/Vaud/bilan_carbone_territoire_2023_public2(8).xlsx",
        sheet_name="Transports - Emissions",
        header=8,
    ).iloc[:, 1:-3]
    bilan_carbone_tot = bilan_carbone_tot[
        bilan_carbone_tot["Année"].isin([2015, 2019, 2023])
    ]
    bilan_carbone_tot.rename(columns={"Année": "Years"}, inplace=True)
    bilan_carbone_tot.reset_index(inplace=True, drop=True)
    bilan_carbone_freight = bilan_carbone_tot[
        bilan_carbone_tot["Sous-secteur"] == "Transport de marchandises"
    ]
    bilan_carbone_freight["Modèle"] = "Bilan Carbone"
    bilan_carbone_freight["Sous-secteur"] = bilan_carbone_freight[
        "Sous-secteur"
    ].str.lower()

    plot_comparaison(
        m_by_kind_by_mode["freight"],
        col_to_plot_vaud="CO2",
        legend_label="Émissions de GES [tCo2eq]",
        groups=bilan_carbone_freight.groupby(["Sous-secteur"]),
        dic_off_to_model={"transport de marchandises": "total"},
    )


if __name__ == "__main__":
    # fribourg_run()
    # schwyz_run()
    vaud_run()
