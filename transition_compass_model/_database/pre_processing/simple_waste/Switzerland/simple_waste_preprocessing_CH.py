"""Build simple_waste.pickle for Switzerland: municipal waste incineration plants (KVA).

Waste burned, electricity and heat produced, and CO2 (fossil and biogenic separately),
by canton. See PLAN_simple_industry_waste.md in pre_processing/simple_industry/Switzerland.

- National history (1990-2023): GEST T27 (waste input, electricity/heat production).
- Fossil CO2: FOEN inventory line "thereof: waste incineration plant" (calibrated).
- Biogenic CO2: biogenic energy share of the waste x biomass emission factor.
- Cantons: SwissPRTR CO2 of each plant (21 of 29 plants, those above the 100 kt
  threshold); the remainder goes to cantons whose plant capacity (waste_power.xlsx)
  is not covered by PRTR. Hard check: canton CO2 >= PRTR.
- Forecast: not demand driven. Levers 1-4 on waste incinerated per capita.

KVA emissions are also counted by the energy module (electricity-generation, see
model/energy/workflows.py compute_electricity_generation_emissions). Removing them
there is a follow-up; until then do not add both.

No stat-tab API calls in this file.
"""

import os
import pickle

import numpy as np
import pandas as pd

from transition_compass_model._database.pre_processing.constants import (
    CANTONS_RENAME_TARGET,
)
from transition_compass_model._database.pre_processing.simple_industry.Switzerland import (
    get_data_functions as gd,
)
from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    load_pop,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

DATAMATRIX_DIR = os.path.join(gd.ROOT_DIR, "_database/data/datamatrix")
KVA_PLANTS_FILE = os.path.join(gd.POWER_DATA_DIR, "waste_power.xlsx")

YEARS_OTS = create_years_list(1990, 2023, 1)
YEARS_FTS = create_years_list(2025, 2050, 5)
BASE_YEAR = 2023
CANTONS = sorted(CANTONS_RENAME_TARGET)
GASES = ["CH4", "CO2", "N2O"]

# GEST counts 50% of the energy from waste incineration as renewable
BIOGENIC_ENERGY_SHARE = 0.5
EF_BIOGENIC_CO2 = 0.3774  # Mt/TWh, IPCC solid biomass (same as industry 'wood')
EF_CH4 = 1.08e-4  # Mt/TWh, IPCC solid waste
EF_N2O = 1.44e-5
# Level 4: waste incinerated per capita in 2050 relative to level 1 (circular economy).
# ASSUMPTION to be agreed with the model owner.
LEVEL4_PER_CAPITA_2050 = 0.8
AVG_YEARS = create_years_list(2019, 2023, 1)  # base for forecast ratios

ENERGY_VARS = {
    "wst_waste-incinerated": "waste-input",
    "wst_fossil-support-fuel": "fossil-input",
    "wst_electricity-production": "electricity-production",
    "wst_electricity-own-use": "electricity-own-use",
    "wst_heat-production": "heat-production",
    "wst_heat-own-use": "heat-own-use",
}
GERMAN_CANTONS = {
    "Basel-Stadt": "Basel Stadt",
    "Basel-Landschaft": "Basel Landschaft",
    "Freiburg": "Fribourg",
    "Genf": "Geneva",
    "Luzern": "Lucerne",
    "Neuenburg": "Neuchâtel",
    "Tessin": "Ticino",
    "Waadt": "Vaud",
    "Wallis": "Valais",
    "Zürich": "Zurich",
    "Graubünden": "Graubünden",
}


def plant_capacity_by_canton():
    """Daily waste throughput (t/day) of KVA plants by canton (Wikipedia list)."""
    df = pd.read_excel(KVA_PLANTS_FILE)
    canton = df["Kanton"].astype(str).str.replace("\xa0", " ").str.strip()
    canton = canton.map(lambda c: GERMAN_CANTONS.get(c, c))
    unknown = set(canton) - set(CANTONS)
    if unknown:
        raise ValueError(f"Unknown cantons in {KVA_PLANTS_FILE}: {unknown}")
    cap = df["Täglicher Abfalldurchsatz (Tonnen)"].groupby(canton).sum()
    return cap.reindex(CANTONS).fillna(0.0).values


def national_history():
    """National energy (TWh) and CO2 (Mt) per year, YEARS_OTS."""
    kva = gd.read_gest_kva().reindex(YEARS_OTS)
    fossil = gd.read_foen_inventory("CO2")["thereof: waste incineration plant"]
    nat = kva.copy()
    nat["co2-fossil"] = fossil.reindex(YEARS_OTS).values
    nat["co2-biogenic"] = nat["waste-input"] * BIOGENIC_ENERGY_SHARE * EF_BIOGENIC_CO2
    nat["ch4"] = nat["waste-input"] * EF_CH4
    nat["n2o"] = nat["waste-input"] * EF_N2O
    if nat.isna().any().any():
        raise ValueError(
            "Missing national KVA data:\n" + str(nat[nat.isna().any(axis=1)])
        )
    return nat


def canton_shares(nat, prtr, cap):
    """Canton shares of KVA activity [canton, year] and the list of national violations.

    Canton CO2 = PRTR + remainder spread over the capacity not covered by PRTR.
    """
    total = (nat["co2-fossil"] + nat["co2-biogenic"]).values
    shares = np.zeros((len(CANTONS), len(YEARS_OTS)))
    cap_share = cap / cap.sum()
    prtr_years = [iy for iy in range(len(YEARS_OTS)) if not np.isnan(prtr[0, iy])]
    violations = []
    for iy in range(len(YEARS_OTS)):
        if iy not in prtr_years:
            continue
        P = prtr[:, iy]
        rem = total[iy] - P.sum()
        if rem < 0:
            violations.append((YEARS_OTS[iy], total[iy], P.sum()))
        expected = cap_share * total[iy]  # what capacity alone would give
        uncovered = np.maximum(expected - P, 0.0)
        co2_c = P + max(rem, 0.0) * uncovered / uncovered.sum()
        shares[:, iy] = co2_c / co2_c.sum()
    first = prtr_years[0]
    shares[:, :first] = shares[:, [first]]  # before PRTR: first PRTR year
    return shares, violations


def project_national(nat):
    """National projection per lever level: dict lev -> DataFrame (YEARS_FTS)."""
    pop = load_pop(["Switzerland"], YEARS_OTS + YEARS_FTS)
    pop = pd.Series(pop.array[0, :, 0], index=pop.col_labels["Years"])
    avg = nat.loc[AVG_YEARS]
    per_cap = (avg["waste-input"] / pop[AVG_YEARS]).mean()
    per_waste = {
        k: (avg[k] / avg["waste-input"]).mean()
        for k in nat.columns
        if k != "waste-input"
    }
    out = {}
    for lev in (1, 2, 3, 4):
        factor = 1 + (LEVEL4_PER_CAPITA_2050 - 1) * (lev - 1) / 3
        rows = {}
        for y in YEARS_FTS:
            ramp = 1 + (factor - 1) * (y - BASE_YEAR) / (2050 - BASE_YEAR)
            waste = per_cap * ramp * pop[y]
            row = {"waste-input": waste}
            row.update({k: v * waste for k, v in per_waste.items()})
            rows[y] = row
        out[lev] = pd.DataFrame(rows).T[nat.columns]
    return out


def build_dms(nat, shares, years):
    """National table + canton shares [canton, year] -> energy dm, emissions dm."""
    countries = CANTONS + ["Switzerland"]
    sh = np.concatenate([shares, np.ones((1, shares.shape[1]))], axis=0)
    dm_e = DataMatrix(
        col_labels={
            "Country": countries,
            "Years": years,
            "Variables": list(ENERGY_VARS),
        },
        units={v: "TWh" for v in ENERGY_VARS},
    )
    dm_e.array = np.stack(
        [sh * nat[col].values[None, :] for col in ENERGY_VARS.values()], axis=2
    )
    em = np.zeros((len(countries), len(years), 2, len(GASES)))
    em[:, :, 0, GASES.index("CO2")] = sh * nat["co2-fossil"].values
    em[:, :, 0, GASES.index("CH4")] = sh * nat["ch4"].values
    em[:, :, 0, GASES.index("N2O")] = sh * nat["n2o"].values
    em[:, :, 1, GASES.index("CO2")] = sh * nat["co2-biogenic"].values
    dm_m = DataMatrix(
        col_labels={
            "Country": countries,
            "Years": years,
            "Variables": ["wst_emissions-fossil", "wst_emissions-biogenic"],
            "Categories1": GASES,
        },
        units={"wst_emissions-fossil": "Mt", "wst_emissions-biogenic": "Mt"},
    )
    dm_m.array = em
    dm_e.sort("Country")
    dm_m.sort("Country")
    return dm_e, dm_m


def check_and_report(nat, shares, prtr, violations, strict):
    ok = True
    total = (nat["co2-fossil"] + nat["co2-biogenic"]).values
    canton_co2 = shares * total[None, :]
    rows = []
    for iy, y in enumerate(YEARS_OTS):
        if np.isnan(prtr[0, iy]):
            continue
        for ic, c in enumerate(CANTONS):
            if prtr[ic, iy] > 0:
                rows.append((y, c, canton_co2[ic, iy], prtr[ic, iy]))
    df = pd.DataFrame(rows, columns=["year", "canton", "model_Mt", "prtr_Mt"])
    df["margin_Mt"] = df["model_Mt"] - df["prtr_Mt"]
    bad = df[df["margin_Mt"] < -1e-9]
    print(
        f"\n[check] canton KVA CO2 (fossil + biogenic) >= PRTR: {len(df) - len(bad)}/{len(df)} pass"
    )
    print(df[df.year == 2023].round(3).to_string(index=False))
    for v in violations:
        print(f"   NATIONAL {v[0]}: model {v[1]:.3f} < PRTR {v[2]:.3f} Mt")
    if len(bad) or violations:
        ok = False
        print(bad.round(4).to_string(index=False))
    print("\n[report] national KVA, selected years:")
    print(
        nat.loc[
            [1990, 2000, 2010, 2020, 2023],
            [
                "waste-input",
                "electricity-production",
                "heat-production",
                "co2-fossil",
                "co2-biogenic",
            ],
        ].round(2)
    )
    if not ok and strict:
        raise ValueError("simple_waste checks failed (see report above)")


def run(strict=True, write=True):
    nat = national_history()
    df_prtr = gd.read_prtr_co2()
    df_kva = df_prtr[df_prtr["branch"] == "kva"]
    prtr = gd.prtr_by_canton_branch(df_kva, CANTONS, YEARS_OTS, ["kva"])[:, :, 0]
    shares, violations = canton_shares(nat, prtr, plant_capacity_by_canton())
    check_and_report(nat, shares, prtr, violations, strict)

    dm_e, dm_m = build_dms(nat, shares, YEARS_OTS)
    proj = project_national(nat)
    sh_fts = np.repeat(shares[:, [-1]], len(YEARS_FTS), axis=1)  # frozen at 2023
    fts = {}
    for lev, df in proj.items():
        e, m = build_dms(df, sh_fts, YEARS_FTS)
        fts[lev] = {"energy": e, "emissions": m}
        print(
            f"[report] level {lev} 2050: waste {df.loc[2050, 'waste-input']:.2f} TWh, "
            f"fossil CO2 {df.loc[2050, 'co2-fossil']:.2f} Mt"
        )

    prtr_years = sorted(df_kva["year"].unique())
    prtr_all = gd.prtr_by_canton_branch(df_kva, CANTONS, prtr_years, ["kva"])
    dm_prtr = DataMatrix(
        col_labels={
            "Country": CANTONS + ["Switzerland"],
            "Years": prtr_years,
            "Variables": ["wst_prtr-co2"],
        },
        units={"wst_prtr-co2": "Mt"},
    )
    dm_prtr.array = np.concatenate([prtr_all, prtr_all.sum(axis=0, keepdims=True)])
    dm_prtr.sort("Country")

    DM = {
        "ots": {"simple-waste": {"energy": dm_e, "emissions": dm_m}},
        "fts": {"simple-waste": fts},
        "fxa": {"prtr-co2": dm_prtr},
    }
    if write:
        f_out = os.path.join(DATAMATRIX_DIR, "simple_waste.pickle")
        with open(f_out, "wb") as handle:
            pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\nSaved {f_out}")
    return DM


if __name__ == "__main__":
    run()
