"""Compare transport-sector GHG emissions per capita across all cantons.

Runs lifestyles -> transport for every canton (+ Switzerland) in one go
and writes two interactive plotly charts over the historical years:
  1. transport emissions per capita [t CO2e/cap]
  2. the same, normalised to 1 in 1990

Set SCOPE at the top of the file (or pass --scope) to choose which emissions to plot:
  "land"           road + rail, passenger and freight (default). Excludes aviation,
                   whose canton values are the Swiss total split by population share,
                   and inland waterways / marine.
  "passenger-land" passenger 2W, cars, bus, metro-tram, rail
  "cars"           passenger cars (LDV) only
  "all"            every mode, including aviation, inland waterways and marine

Usage (from the repo root):
    uv run python scripts/transport_emissions_per_capita_cantons.py \
        [--scope land|passenger-land|cars|all] [--out DIR]   (default DIR: scripts/output/)
"""

import argparse
import logging
import math
import os
import pickle

import pandas as pd
import plotly.graph_objects as go

from transition_compass_model.model.common.auxiliary_functions import (
    filter_country_and_load_data_from_pickles,
)
from transition_compass_model.model.common.config_loader import load_lever_config
from transition_compass_model.model.common.interface_class import Interface
from transition_compass_model.model.lifestyles_module import lifestyles
from transition_compass_model.model.transport_module import transport

YEARS_SETTING = [1990, 2023, 2025, 2050, 5]
SECTORS = ["lifestyles", "transport"]
# Which emissions to plot; the --scope command-line option overrides this.
SCOPE = "land"
_PASSENGER_LAND = [
    "tra_emissions-CO2e_passenger_2W",
    "tra_emissions-CO2e_passenger_LDV",
    "tra_emissions-CO2e_passenger_bus",
    "tra_emissions-CO2e_passenger_metrotram",
    "tra_emissions-CO2e_passenger_rail",
]
EMI_VARS = {
    "land": _PASSENGER_LAND
    + ["tra_emissions-CO2e_freight_HDV", "tra_emissions-CO2e_freight_rail"],
    "passenger-land": _PASSENGER_LAND,
    "cars": ["tra_emissions-CO2e_passenger_LDV"],
    "all": _PASSENGER_LAND
    + [
        "tra_emissions-CO2e_passenger_aviation",
        "tra_emissions-CO2e_freight_HDV",
        "tra_emissions-CO2e_freight_rail",
        "tra_emissions-CO2e_freight_IWW",
        "tra_emissions-CO2e_freight_aviation",
        "tra_emissions-CO2e_freight_marine",
    ],
}
LABELS = {
    "land": "Land transport (road + rail)",
    "passenger-land": "Passenger land transport",
    "cars": "Passenger cars",
    "all": "Transport (all modes)",
}
# Factor from the output emission unit to tonnes CO2e.
TO_TONNES = {"t": 1.0, "kt": 1e3, "Mt": 1e6}
POP_VAR = "lfs_population_total"
REFERENCE = "Switzerland"
# Default folder for the HTML plots (git-ignored), next to this script.
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def available_countries():
    """Swiss Country labels (Switzerland + cantons) in the transport input pickle."""
    import transition_compass_model

    path = os.path.join(
        os.path.dirname(transition_compass_model.__file__),
        "_database/data/datamatrix/transport.pickle",
    )
    with open(path, "rb") as f:
        data = pickle.load(f)

    def first_dm(x):
        if isinstance(x, dict):
            for v in x.values():
                dm = first_dm(v)
                if dm is not None:
                    return dm
            return None
        return x

    return [c for c in first_dm(data).col_labels["Country"] if c != "EU27"]


def to_long(dm, var):
    """Return (Years x Country table of `var`, unit) from a DataMatrix output."""
    if var not in dm.col_labels["Variables"]:
        raise KeyError(f"{var} not in {dm.col_labels['Variables']}")
    unit = dm.units[var]
    df = dm.filter({"Variables": [var]}).write_df()
    col = next(c for c in df.columns if c.split("[")[0] == var)
    df = df[["Country", "Years", col]].rename(columns={col: "value"})
    df["Years"] = df["Years"].astype(int)
    return df.pivot(index="Years", columns="Country", values="value"), unit


def emissions_tonnes(dm, variables):
    """Sum `variables` and convert to t CO2e (Years x Country)."""
    total = 0
    for var in variables:
        table, unit = to_long(dm, var)
        if unit not in TO_TONNES:
            raise ValueError(f"Unexpected unit {unit!r} for {var}")
        total = total + table * TO_TONNES[unit]
    return total


def run_model(countries):
    """Same order as interactions.runner (climate only feeds buildings, so skipped)."""
    lever_setting = {k: math.floor(v) for k, v in load_lever_config().items()}
    DM_input = filter_country_and_load_data_from_pickles(
        country_list=countries, modules_list=SECTORS
    )
    interface = Interface()
    TPE = {}
    TPE["lifestyles"] = lifestyles(
        lever_setting, YEARS_SETTING, DM_input["lifestyles"], interface
    )
    TPE["transport"], _ = transport(
        lever_setting, YEARS_SETTING, DM_input["transport"], interface
    )
    return TPE


def line_figure(table, title, yaxis_title, ref_line=None):
    fig = go.Figure()
    cantons = sorted(c for c in table.columns if c != REFERENCE)
    for c in cantons:
        fig.add_trace(
            go.Scatter(
                x=table.index,
                y=table[c],
                mode="lines",
                name=c,
                line={"width": 1.5},
                hovertemplate=f"{c}<br>%{{x}}: %{{y:.3f}}<extra></extra>",
            )
        )
    if REFERENCE in table.columns:
        fig.add_trace(
            go.Scatter(
                x=table.index,
                y=table[REFERENCE],
                mode="lines",
                name=REFERENCE,
                line={"width": 4, "color": "black"},
                hovertemplate=f"{REFERENCE}<br>%{{x}}: %{{y:.3f}}<extra></extra>",
            )
        )
    if ref_line is not None:
        fig.add_hline(y=ref_line, line_dash="dot", line_color="grey")
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title=yaxis_title,
        template="plotly_white",
        hovermode="closest",
        height=700,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", default=OUT_DIR, help="output directory for the HTML plots"
    )
    parser.add_argument(
        "--scope",
        choices=list(EMI_VARS),
        default=SCOPE,
        help="which transport emissions to plot (see the module docstring)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    countries = available_countries()
    logger.info(f"Running {SECTORS} for {len(countries)} regions")
    TPE = run_model(countries)

    hist = range(YEARS_SETTING[0], YEARS_SETTING[1] + 1)
    emi_t = emissions_tonnes(TPE["transport"], EMI_VARS[args.scope]).loc[hist]
    pop, _ = to_long(TPE["lifestyles"], POP_VAR)
    pop = pop.loc[hist]

    per_cap = emi_t / pop  # t CO2e per inhabitant
    index_1990 = per_cap / per_cap.loc[YEARS_SETTING[0]]

    os.makedirs(args.out, exist_ok=True)
    tag = f"_{args.scope}"
    label = LABELS[args.scope]
    f1 = os.path.join(args.out, f"transport_emissions_per_capita{tag}.html")
    f2 = os.path.join(args.out, f"transport_emissions_per_capita_index1990{tag}.html")
    line_figure(
        per_cap,
        f"{label} GHG emissions per capita (historical)",
        "t CO2e / capita",
    ).write_html(f1)
    line_figure(
        index_1990,
        f"{label} GHG emissions per capita, 1990 = 1",
        "index (1990 = 1)",
        ref_line=1,
    ).write_html(f2)

    y0, y1 = YEARS_SETTING[0], YEARS_SETTING[1]
    summary = pd.DataFrame(
        {
            f"{y0} [t/cap]": per_cap.loc[y0],
            f"{y1} [t/cap]": per_cap.loc[y1],
            "change [%]": (index_1990.loc[y1] - 1) * 100,
        }
    ).sort_values(f"{y1} [t/cap]", ascending=False)
    print(summary.round(2).to_string())
    cantons = [c for c in emi_t.columns if c != REFERENCE]
    if REFERENCE in emi_t.columns and cantons:
        share = emi_t.loc[y1, cantons].sum() / emi_t.loc[y1, REFERENCE]
        print(f"\nSum of cantons / {REFERENCE} emissions in {y1}: {share:.1%}")
    print(f"\nPlots written to:\n  {os.path.abspath(f1)}\n  {os.path.abspath(f2)}")


if __name__ == "__main__":
    main()
