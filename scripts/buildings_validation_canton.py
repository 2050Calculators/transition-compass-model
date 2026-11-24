import math
import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from transition_compass_model._database.pre_processing.params import years_ots
from transition_compass_model.model.buildings_module import buildings
from transition_compass_model.model.climate_module import climate
from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    filter_country_and_load_data_from_pickles,
)
from transition_compass_model.model.common.config_loader import load_lever_config
from transition_compass_model.model.common.interface_class import Interface
from transition_compass_model.model.lifestyles_module import lifestyles

COLOR_MODEL = "#2a78d6"
COLOR_OFFICIAL = "#eb6834"


YEARS_SETTING = [1990, 2023, 2025, 2050, 5]
YEARS = create_years_list(YEARS_SETTING[0], YEARS_SETTING[1], 1)


KEY_YEARS = [1990, 2000, 2010, 2020, 2023]
TJ_PER_TWH = 3600

NONRES_TYPES = ["education", "health", "hotels", "offices", "other", "trade"]
TWH_PER_PJ = 1 / 3.6

HERE = os.path.dirname(os.path.abspath(__file__))
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


# --- Model --------------------------------------------------------------------
def run_model(country):
    """Run climate -> lifestyles -> buildings for Switzerland.

    Returns (TPE buildings DataMatrix, buildings->energy interface dict, historical
    heating-mix input DataMatrix).
    """
    lever_setting = {k: math.floor(v) for k, v in load_lever_config().items()}
    sectors = ["climate", "lifestyles", "buildings"]
    DM_input = filter_country_and_load_data_from_pickles(
        country_list=[country], modules_list=sectors
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
    dm = dm_tpe.filter({"Variables": list(var_of.values())})
    idx = dm.idx
    df = pd.DataFrame(
        {f: dm.array[0, :, idx[v]] for f, v in var_of.items()},
        index=[int(y) for y in dm.col_labels["Years"]],
    )
    return df


##### FRIBOURG ###


def fribourg_values():
    TOTAL_EMISSION_FRIBOURG = 5504411 * 10**-6  # MtCO2eq

    # Le canton comptabilise aussi jardins et loisirs mais nous ne le comptabilisons pas
    EMISSION_FRIBOURG = {
        "services": TOTAL_EMISSION_FRIBOURG * 0.0138,
        "residential": TOTAL_EMISSION_FRIBOURG * 0.0579,
        "total": {
            2023: 0.39,  # MtCO2eq
            2017: 0.53,
        },
    }

    return EMISSION_FRIBOURG


def model_use_emissions(dm_tpe, fuels, building_use, years=years_ots):
    """Model final emisisons, all uses, by model fuel [MtCO2eq]. building_use is
    "residential" or "non-residential"."""
    # "bld_CO2-emissions_residential_heating[Mt]",
    #     "bld_CO2-emissions_residential_hot-water[Mt]",
    #     "bld_CO2-emissions_services_heating[Mt]",
    #     "bld_CO2-emissions_services_hot-water[Mt]"
    heat_tech = ["heating", "hot-water"]
    var_of = {f: f"bld_CO2-emissions_{building_use}_{f}" for f in heat_tech}
    dm = dm_tpe.filter({"Variables": list(var_of.values()), "Years": years})
    idx = dm.idx
    df = pd.DataFrame(
        {f: dm.array[0, :, idx[v]] for f, v in var_of.items()},
        index=[int(y) for y in dm.col_labels["Years"]],
    )
    return df


def plot_emissions_comparison(emissions_fribourg_model, EMISSION_FRIBOURG):
    categories = ["services", "residential", "total"]

    fig = make_subplots(rows=1, cols=3, subplot_titles=categories, shared_yaxes=True)

    colors = {
        "Model": COLOR_MODEL,
        "Fribourg": COLOR_OFFICIAL,
    }

    for i, category in enumerate(categories, start=1):
        # Model values
        years = [2017, 2023]
        model_values = [emissions_fribourg_model[category][year] for year in years]

        fig.add_trace(
            go.Bar(
                x=years,
                y=model_values,
                name="Model",
                marker_color=colors["Model"],
                showlegend=(i == 1),
                text=[f"{v:.3f}" for v in model_values],
                textposition="outside",
            ),
            row=1,
            col=i,
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
            row=1,
            col=i,
        )

    fig.update_layout(
        title="Comparison of CO₂ emissions in Fribourg",
        title_x=0.5,
        template="plotly_white",
        barmode="group",
        height=500,
        width=1100,
        legend_title="Dataset",
    )

    fig.update_yaxes(title_text="Emissions", row=1, col=1)

    fig.update_xaxes(title_text="Year", tickvals=[2017, 2023])

    fig.show()

    return fig


def fribourg_run():
    print("Running climate -> lifestyles -> buildings")
    country = "Fribourg"
    # Get model values
    dm_tpe, link, dm_heating_mix = run_model(country)
    fuels = link["households_heating"].col_labels["Categories1"]
    fuels.remove("other-tech")
    fuels.remove("ambient-heat")

    YEARS_FRIBOURG_BILAN = [2017, 2023]
    emissions_fribourg_model = {
        "services": model_use_emissions(
            dm_tpe, fuels, "services", YEARS_FRIBOURG_BILAN
        ).sum(axis=1),
        "residential": model_use_emissions(
            dm_tpe, fuels, "residential", YEARS_FRIBOURG_BILAN
        ).sum(axis=1),
    }
    emissions_fribourg_model["total"] = (
        emissions_fribourg_model["services"] + emissions_fribourg_model["residential"]
    )

    EMISSION_FRIBOURG = fribourg_values()
    fig = plot_emissions_comparison(emissions_fribourg_model, EMISSION_FRIBOURG)

    print("test")


def schwyz_values():
    df_raw = pd.read_excel("data/Schwyz/bld_emi_ecospeed.xlsx")

    years = [str(i) for i in range(2016, 2026)]

    # Keep single-family homes
    df = pd.DataFrame()
    # df = df_raw.loc[
    #     df_raw["Building Type"] == "Single-family homes"
    # ].copy()

    # df.replace("Single-family homes",'single-family-households')

    # Aggregate multifamily homes
    mask_res = df_raw["Building Type"].isin(
        [
            "Single-family homes",
            "Multifamily homes",
            "Residential buildings with secondary use",
            # "Buildings with partial residential use"
        ]
    )

    new_row = df_raw.loc[mask_res, years].sum()

    new_row["Building Type"] = "residential"

    df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

    # Aggregate services (all remaining building types)
    mask_services = ~mask_res

    new_row = df_raw.loc[mask_services, years].sum()
    new_row["Building Type"] = "services"

    df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

    df.loc["total", years] = df_raw.loc[:, years].sum()
    df.loc[:, years] = df.loc[:, years] * 10**-6  # Convert from tCO2eq to MTCO2eq
    df.loc["total", "Building Type"] = "Total"

    df_total = pd.read_excel("data/Schwyz/Eco_speed_sectors.xlsx")
    df_bld_tot = df_total[
        (df_total["Bereiche"] == "Haushalte") | (df_total["Bereiche"] == "Wirtschaft")
    ]

    df_bld_tot.iloc[:, 1:] = df_bld_tot.iloc[:, 1:] * 10**-6

    return df, df_bld_tot


def plot_emissions_comparison_schwyz(df_sector, df_detailed, emissions_model):
    years = list(range(2010, 2026))

    colors = {
        "Model": COLOR_MODEL,
        "Detailed data": COLOR_OFFICIAL,
        "Sector statistics": "#4C9F9A",  # Muted teal
    }
    # Extract data from df_sector
    households = df_sector.loc[df_sector["Bereiche"] == "Haushalte"].iloc[0]

    economy = df_sector.loc[df_sector["Bereiche"] == "Wirtschaft"].iloc[0]

    # Extract data from df_detailed
    residential = df_detailed.loc[df_detailed["Building Type"] == "residential"].iloc[0]

    services = df_detailed.loc[df_detailed["Building Type"] == "services"].iloc[0]

    # Create figure
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Residential emissions",
            "Services emissions",
            "Total emissions",
        ),
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.10,
    )

    # Helper function to add traces
    def add_trace(row, name, x, y, dash="solid"):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=name,
                legendgroup=name,
                showlegend=(row == 1),  # Show legend only in first subplot
                line=dict(color=colors[name], dash=dash, width=2),
                marker=dict(size=6),
            ),
            row=row,
            col=1,
        )

    # --------------------------------------------------
    # 1. RESIDENTIAL
    # --------------------------------------------------

    add_trace(
        1, "Sector statistics", years, [households.get(str(y), None) for y in years]
    )

    add_trace(
        1,
        "Detailed data",
        list(range(2016, 2026)),
        [residential.get(str(y), None) for y in range(2016, 2026)],
    )

    add_trace(
        1, "Model", years, [emissions_model["residential"].get(y, None) for y in years]
    )

    # --------------------------------------------------
    # 2. SERVICES
    # --------------------------------------------------

    add_trace(2, "Sector statistics", years, [economy.get(str(y), None) for y in years])

    add_trace(
        2,
        "Detailed data",
        list(range(2016, 2026)),
        [services.get(str(y), None) for y in range(2016, 2026)],
    )

    add_trace(
        2, "Model", years, [emissions_model["services"].get(y, None) for y in years]
    )

    # --------------------------------------------------
    # 3. TOTAL
    # --------------------------------------------------

    sector_total = [
        (households.get(str(y), 0) or 0) + (economy.get(str(y), 0) or 0) for y in years
    ]
    detailed_total = df_detailed.loc[df_detailed["Building Type"] == "Total"].iloc[0]

    add_trace(3, "Sector statistics", years, sector_total)

    add_trace(3, "Detailed data", years, detailed_total)

    add_trace(3, "Model", years, [emissions_model["total"].get(y, None) for y in years])

    # --------------------------------------------------
    # LAYOUT
    # --------------------------------------------------
    # Find common y-axis range across all subplots
    all_values = []

    for trace in fig.data:
        numeric_values = pd.to_numeric(pd.Series(trace.y), errors="coerce").dropna()

        all_values.extend(numeric_values.tolist())

    y_min = min(all_values)
    y_max = max(all_values)

    # Add 5% padding
    padding = (y_max - y_min) * 0.05

    y_range = [y_min - padding, y_max + padding]

    # Apply identical y-axis range to all subplots
    fig.update_yaxes(range=y_range, autorange=False)

    fig.update_layout(
        height=1100,
        width=1100,
        title_text="CO₂ Emissions Comparison – Canton Schwyz",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    fig.show()
    return fig


def schwyz_run():
    print("Running climate -> lifestyles -> buildings")
    country = "Schwyz"
    # Get model values
    dm_tpe, link, dm_heating_mix = run_model(country)
    fuels = link["households_heating"].col_labels["Categories1"]
    fuels.remove("other-tech")
    fuels.remove("ambient-heat")

    YEARS_SCHWYZ_BILAN = [y for y in range(2010, 2024)] + [2025]
    emissions_Shwyz_model = {
        "services": model_use_emissions(
            dm_tpe, fuels, "services", YEARS_SCHWYZ_BILAN
        ).sum(axis=1),
        "residential": model_use_emissions(
            dm_tpe, fuels, "residential", YEARS_SCHWYZ_BILAN
        ).sum(axis=1),
    }
    emissions_Shwyz_model["total"] = (
        emissions_Shwyz_model["services"] + emissions_Shwyz_model["residential"]
    )

    df_detailed, df_sector = schwyz_values()
    fig = plot_emissions_comparison_schwyz(
        df_sector, df_detailed, emissions_Shwyz_model
    )

    print("test")


if __name__ == "__main__":
    # fribourg_run()
    schwyz_run()
