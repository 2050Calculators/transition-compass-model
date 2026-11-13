"""Compare buildings-sector GHG emissions per capita across all cantons.

Runs climate -> lifestyles -> buildings for every canton (+ Switzerland) in one go
and writes two interactive plotly charts over the historical years:
  1. buildings emissions per capita [t CO2e/cap]
  2. the same, normalised to 1 in 1990

--climate typical replaces the weather inputs to buildings (HDD, CDD, days below 15 C,
days above 24 C) with one typical year -- the Swiss mean over TYPICAL_PERIOD -- for
every canton and every year, removing weather differences across cantons and years.

Usage (from the repo root):
    uv run python scripts/buildings_emissions_per_capita_cantons.py \
        [--climate actual|typical] [--out DIR]
"""

import argparse
import logging
import math
import os
import pickle

import pandas as pd
import plotly.graph_objects as go

from transition_compass_model.model.buildings_module import buildings
from transition_compass_model.model.climate_module import climate
from transition_compass_model.model.common.auxiliary_functions import (
    filter_country_and_load_data_from_pickles,
)
from transition_compass_model.model.common.config_loader import load_lever_config
from transition_compass_model.model.common.interface_class import Interface
from transition_compass_model.model.lifestyles_module import lifestyles

YEARS_SETTING = [1990, 2023, 2025, 2050, 5]
SECTORS = ["climate", "lifestyles", "buildings"]
EMI_VAR = "bld_emissions-CO2e"
POP_VAR = "lfs_population_total"
REFERENCE = "Switzerland"
# Weather-driven climate variables read by buildings (bld_energy_workflow).
# clm_AC-uptake is an adoption share, not weather, so it is left as is.
WEATHER_VARS = ["clm_HDD", "clm_CDD", "clm_days-below-15", "clm_days-above-24"]
TYPICAL_PERIOD = (1991, 2020)  # climate normal period


def available_countries():
    """All Country labels present in the buildings input pickle."""
    import transition_compass_model

    path = os.path.join(
        os.path.dirname(transition_compass_model.__file__),
        "_database/data/datamatrix/buildings.pickle",
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

    return list(first_dm(data).col_labels["Country"])


def to_long(out, var):
    """Return a Country x Years table of `var` from a DataMatrix or DataFrame output."""
    if isinstance(out, pd.DataFrame):
        col = next(c for c in out.columns if c.split("[")[0] == var)
        df = out[["Country", "Years", col]].rename(columns={col: "value"})
    else:  # DataMatrix
        dm = (
            out.filter({"Variables": [var]})
            if var in out.col_labels["Variables"]
            else None
        )
        if dm is None:
            raise KeyError(f"{var} not in {out.col_labels['Variables']}")
        df = dm.write_df()
        col = next(c for c in df.columns if c.split("[")[0] == var)
        df = df[["Country", "Years", col]].rename(columns={col: "value"})
    df["Years"] = df["Years"].astype(int)
    return df.pivot(index="Years", columns="Country", values="value")


def apply_typical_climate(dm_clm, logger):
    """Overwrite weather variables with the Swiss mean over TYPICAL_PERIOD (in place)."""
    years = [
        y
        for y in dm_clm.col_labels["Years"]
        if TYPICAL_PERIOD[0] <= y <= TYPICAL_PERIOD[1]
    ]
    idx = dm_clm.idx
    i_ref = idx[REFERENCE]
    i_years = [idx[y] for y in years]
    for var in WEATHER_VARS:
        typical = dm_clm.array[i_ref, i_years, idx[var]].mean()
        dm_clm.array[:, :, idx[var]] = typical
        logger.info(f"Typical climate: {var} = {typical:.1f} {dm_clm.units[var]}")


def run_model(countries, climate_mode, logger):
    """Same order as interactions.runner, with an optional climate override."""
    lever_setting = {k: math.floor(v) for k, v in load_lever_config().items()}
    DM_input = filter_country_and_load_data_from_pickles(
        country_list=countries, modules_list=SECTORS
    )
    interface = Interface()
    TPE = {}
    TPE["climate"] = climate(
        lever_setting, YEARS_SETTING, DM_input["climate"], interface
    )
    if climate_mode == "typical":
        apply_typical_climate(interface.get_link("climate", "buildings"), logger)
    TPE["lifestyles"] = lifestyles(
        lever_setting, YEARS_SETTING, DM_input["lifestyles"], interface
    )
    TPE["buildings"], _ = buildings(
        lever_setting, YEARS_SETTING, DM_input["buildings"], interface
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
        "--out", default=".", help="output directory for the HTML plots"
    )
    parser.add_argument(
        "--climate",
        choices=["actual", "typical"],
        default="actual",
        help="actual: canton/year weather; typical: one Swiss typical year everywhere",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    countries = available_countries()
    logger.info(f"Running {SECTORS} for {len(countries)} regions")
    TPE = run_model(countries, args.climate, logger)

    hist = range(YEARS_SETTING[0], YEARS_SETTING[1] + 1)
    emi_mt = to_long(TPE["buildings"], EMI_VAR).loc[hist]
    pop = to_long(TPE["lifestyles"], POP_VAR).loc[hist]

    per_cap = emi_mt * 1e6 / pop  # t CO2e per inhabitant
    index_1990 = per_cap / per_cap.loc[YEARS_SETTING[0]]

    os.makedirs(args.out, exist_ok=True)
    tag = "" if args.climate == "actual" else "_typical-climate"
    note = (
        ""
        if args.climate == "actual"
        else f" -- typical climate (CH mean {TYPICAL_PERIOD[0]}-{TYPICAL_PERIOD[1]})"
    )
    f1 = os.path.join(args.out, f"buildings_emissions_per_capita{tag}.html")
    f2 = os.path.join(args.out, f"buildings_emissions_per_capita_index1990{tag}.html")
    line_figure(
        per_cap,
        f"Buildings GHG emissions per capita (historical){note}",
        "t CO2e / capita",
    ).write_html(f1)
    line_figure(
        index_1990,
        f"Buildings GHG emissions per capita, 1990 = 1{note}",
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
    print(f"\nPlots written to:\n  {os.path.abspath(f1)}\n  {os.path.abspath(f2)}")


if __name__ == "__main__":
    main()
