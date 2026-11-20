"""Fit the services U-values (by envelope class) to EP2050 useful space heat.

The model computes useful space heat (heat lost by the building, before boiler or
heat-pump losses) as U-value x envelope area per m² x heating degree days x 24, for
each building type and envelope class. Heat is linear in the U-value, so each class
can be fitted with one factor on its current U-value, without re-running the model:

  - C (built 2001-2010): EP2050 "Neubau" (new buildings) heat per m², 2001-2010.
  - B (built from 2011): EP2050 "Neubau" heat per m², 2011-2015.
  - D, E, F: one common factor (their ratios are kept), so that the services stock
    matches EP2050 "Baujahr < 2020" over FIT_YEARS, as a ratio of sums
    (total heat / total floor area over the window).

Source: EP2050+ Dienstleistung, Tabelle 03-01 (kWh/m², same floor area as the model).
Prints the fitted U-values, yearly residuals vs EP2050 and the expected final space
heating vs Prognos. Writes nothing.

Usage (from the repo root):
    uv run python scripts/fit_services_u_values_CH.py
"""

import math

import numpy as np
import pandas as pd
from buildings_validation_CH import (
    COUNTRY,
    EP2050_HEAT_TABLE,
    NONRES_TYPES,
    YEARS,
    YEARS_SETTING,
    _ep2050_table,
    load_ep2050_services_useful_heat,
    load_prognos_services_space_heating,
)

import transition_compass_model.model.buildings.workflows as wkf
from transition_compass_model.model.buildings_module import read_data
from transition_compass_model.model.climate_module import climate
from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    filter_country_and_load_data_from_pickles,
)
from transition_compass_model.model.common.config_loader import load_lever_config
from transition_compass_model.model.common.interface_class import Interface
from transition_compass_model.model.lifestyles_module import lifestyles

FIT_YEARS = list(range(2010, 2016))  # stock, D-F
C_YEARS = list(range(2001, 2011))  # new buildings of class C
B_YEARS = list(range(2011, 2016))  # new buildings of class B
OLD_CLASSES = ["D", "E", "F"]


def run_buildings_workflow():
    """Useful heat [kWh], occupied floor [m²], final space heating [TWh] and U-values
    for the services types, by class. Returns DataFrames indexed by year, columns =
    class (summed over types), plus the current U-values by class."""
    lever_setting = {k: math.floor(v) for k, v in load_lever_config().items()}
    DM = filter_country_and_load_data_from_pickles(
        [COUNTRY], ["climate", "lifestyles", "buildings"]
    )
    it = Interface()
    climate(lever_setting, YEARS_SETTING, DM["climate"], it)
    lifestyles(lever_setting, YEARS_SETTING, DM["lifestyles"], it)
    DM_floor, _, DM_energy, _, _, _, cdm_const = read_data(
        DM["buildings"], lever_setting
    )
    years_ots = create_years_list(YEARS_SETTING[0], YEARS_SETTING[1], 1)
    years_fts = create_years_list(YEARS_SETTING[2], YEARS_SETTING[3], YEARS_SETTING[4])
    floor_out = wkf.bld_floor_area_workflow(
        DM_floor,
        it.get_link("lifestyles", "buildings")["pop"],
        cdm_const,
        years_ots,
        years_fts,
    )
    energy_out = wkf.bld_energy_workflow(
        DM_energy,
        it.get_link("climate", "buildings"),
        floor_out["wf-energy"],
        cdm_const,
    )

    dm = floor_out["wf-energy"].filter(
        {"Country": [COUNTRY], "Categories1": NONRES_TYPES}
    )
    years = [int(y) for y in dm.col_labels["Years"]]
    classes = dm.col_labels["Categories2"]
    idx = dm.idx

    def by_class(arr):  # years x type x class -> years x class
        return pd.DataFrame(arr.sum(axis=1), index=years, columns=classes)

    heat = by_class(dm.array[0, :, idx["bld_heating"]])
    floor = by_class(dm.array[0, :, idx["bld_floor-area_occupied"]])

    dm_fin = energy_out["energy_per_cat"].filter(
        {
            "Country": [COUNTRY],
            "Categories1": NONRES_TYPES,
            "Variables": ["bld_energy-demand_heating"],
        }
    )
    final = by_class(np.nansum(dm_fin.array[0, :, 0], axis=-1))

    dm_u = DM_energy["u-value"].filter({"Country": [COUNTRY]})
    u = dm_u.array[0, 0, 0][[dm_u.idx[t] for t in NONRES_TYPES]]
    if not np.allclose(u, u[:1]):
        raise ValueError("Services types have different U-values: fit by type")
    u_now = pd.Series(u[0], index=dm_u.col_labels["Categories2"])
    return heat, floor, final, u_now


def main():
    heat, floor, final, u_now = run_buildings_workflow()
    ep_stock = load_ep2050_services_useful_heat()["services"]
    sheet, table, _ = EP2050_HEAT_TABLE
    ep_new = _ep2050_table(
        sheet, table, ["Neubau (Wert gültig für das jeweilige Jahr)"]
    )
    ep_new = ep_new.iloc[:, 0]

    q = heat / floor  # kWh/m² per class (NaN where a class has no floor yet)

    factor = pd.Series(1.0, index=u_now.index)
    factor["C"] = ep_new.loc[C_YEARS].mean() / q.loc[C_YEARS, "C"].mean()
    factor["B"] = ep_new.loc[B_YEARS].mean() / q.loc[B_YEARS, "B"].mean()
    target = (ep_stock.loc[FIT_YEARS] * floor.loc[FIT_YEARS].sum(axis=1)).sum()
    new_part = (heat.loc[FIT_YEARS, ["B", "C"]] * factor[["B", "C"]]).sum().sum()
    factor[OLD_CLASSES] = (target - new_part) / heat.loc[
        FIT_YEARS, OLD_CLASSES
    ].sum().sum()
    u_new = u_now * factor

    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print("\n=== Services U-values by envelope class [W/m²K] ===")
        print(
            pd.DataFrame({"current": u_now, "factor": factor, "fitted": u_new})
            .round(2)
            .to_string()
        )
        print(
            "Check B against the SIA 380/1 new-build limits (whole envelope, about "
            "0.3-0.4 W/m²K)."
        )

        fitted_heat = heat * factor
        fitted_q = fitted_heat.sum(axis=1) / floor.sum(axis=1)
        years = [y for y in YEARS if y in ep_stock.index]
        res = pd.DataFrame(
            {
                "EP2050": ep_stock.loc[years],
                "model now": (heat.sum(axis=1) / floor.sum(axis=1)).loc[years],
                "model fitted": fitted_q.loc[years],
            }
        )
        res["fitted/EP2050"] = res["model fitted"] / res["EP2050"]
        print(
            f"\n=== Services useful heat per m² [kWh/m²/yr]; fit on "
            f"{FIT_YEARS[0]}-{FIT_YEARS[-1]} ==="
        )
        print(res.round(2).to_string())
        w = res.loc[FIT_YEARS]
        print(
            f"Fit window, ratio of sums: model now "
            f"{(w['model now'] * floor.sum(axis=1).loc[FIT_YEARS]).sum() / target:.2f}"
            f", fitted {(w['model fitted'] * floor.sum(axis=1).loc[FIT_YEARS]).sum() / target:.2f}"
        )

        new = pd.DataFrame(
            {
                "EP2050 Neubau": ep_new,
                "model C now": q["C"],
                "model C fitted": q["C"] * factor["C"],
                "model B now": q["B"],
                "model B fitted": q["B"] * factor["B"],
            }
        ).loc[[y for y in YEARS if y in ep_new.index]]
        print("\n=== New buildings, useful heat per m² [kWh/m²/yr] ===")
        print(new.round(1).to_string())

        prognos = load_prognos_services_space_heating()["services"]
        fin = pd.DataFrame(
            {
                "Prognos": prognos,
                "model now": final.sum(axis=1).loc[prognos.index],
                "model fitted": (final * factor).sum(axis=1).loc[prognos.index],
            }
        )
        fin["fitted/Prognos"] = fin["model fitted"] / fin["Prognos"]
        print(
            "\n=== Services space heating, final energy [TWh/yr] "
            "(Prognos incl. agriculture; model excl. ambient heat) ==="
        )
        print(fin.round(2).to_string())

    print(
        "\nnonres_u for others_pipeline_CH.py:\n  "
        + "{"
        + ", ".join(f'"{c}": {u_new[c]:.2f}' for c in ["F", "E", "D", "C", "B"])
        + "}"
    )


if __name__ == "__main__":
    main()
