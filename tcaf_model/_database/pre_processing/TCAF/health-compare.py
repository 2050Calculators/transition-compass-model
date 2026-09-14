"""
Equivalence check: R (Projection.R) vs Python (TCAF_health_diet_workflow),
stratified-adherence version.

Feeds R's own inputs (Mean_Intake + Swiss Target + adherence alpha) into the
Python calculation core and diffs against R's exported `par_total`. With the
stratified convention the two should now match at EVERY alpha (not just 0/1).

Usage
-----
1. Set DATA_DIR to the folder with PAF_grid.csv, Projected_DALYs.csv, Mean_Intake.csv.
2. In Projection.R:  readr::write_csv(par_total, "par_total_R.csv")
   (expects columns: serie, Year, attributable/evite/residuel; adjust names below).
3. python compare_R_python.py
"""

import os

import numpy as np
import pandas as pd

from tcaf_model.model.common.data_matrix_class import DataMatrix
from tcaf_model.model.tcaf_module import TCAF_health_diet_workflow

DATA_DIR = "data/health-diet-v2"
R_PAR_TOTAL = "par_total_R.csv"
YEARS = list(range(2025, 2051))
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
# R series label for each alpha (edit to match your R `serie` values)
R_SERIES = {
    0.0: "Référence",
    0.25: "FBDG 25%",
    0.5: "FBDG 50%",
    0.75: "FBDG 75%",
    1.0: "FBDG 100%",
}

DISEASE_MAP = {
    "Breast cancer": "BC",
    "Colon and rectum cancer": "CRC",
    "Diabetes mellitus type 2": "DT2",
    "Esophageal cancer": "EC",
    "Intracerebral hemorrhage": "ICH",
    "Ischemic heart disease": "IHD",
    "Ischemic stroke": "IS",
    "Subarachnoid hemorrhage": "SH",
    "Tracheal bronchus and lung cancer": "TBLC",
    "Stomach cancer": "SC",
}
RF2CAT = {
    "Fruits": "crop-fruit",
    "Vegetables": "crop-veg",
    "Whole_Grains": "crop-cereal-whole",
    "Nuts": "crop-oilcrop",
    "Legumes": "crop-pulse",
    "Milk": "pro-liv-abp-dairy-milk",
    "Red_Meat": "pro-liv-meat-bovine",
    "Processed_Meat": "pro-liv-meat-processed",
}
RF2PAFCAT = dict(RF2CAT)
RF2PAFCAT["Red_Meat"] = "pro-liv-meat-red"
TARGET = {
    "Fruits": 240,
    "Vegetables": 360,
    "Whole_Grains": 180,
    "Nuts": 30,
    "Milk": 300,
    "Legumes": 150,
    "Red_Meat": 16,
    "Processed_Meat": 16,
}
DIET_CATS = [
    "crop-fruit",
    "crop-veg",
    "crop-cereal-whole",
    "crop-oilcrop",
    "crop-pulse",
    "pro-liv-abp-dairy-milk",
    "pro-liv-meat-processed",
    "pro-liv-meat-bovine",
    "pro-liv-meat-pig",
    "pro-liv-meat-sheep",
    "pro-liv-meat-oth-animal",
]


def build_dalys():
    df = pd.read_csv(os.path.join(DATA_DIR, "Projected_DALYs.csv")).rename(
        columns={"Year": "Years", "Forecast_DALY": "value"}
    )
    df = df[df.Years.isin(YEARS)].copy()
    df["Country"] = "Switzerland"
    df["cause"] = df["Disease"].replace(DISEASE_MAP)
    df["variables"] = "tcaf_health-diet_dalys_" + df["cause"] + "[DALYs/y]"
    piv = (
        df[["Country", "Years", "variables", "value"]]
        .pivot_table(index=["Country", "Years"], columns="variables", values="value")
        .reset_index()
    )
    return DataMatrix.create_from_df(piv, num_cat=1)


def build_paf():
    df = pd.read_csv(os.path.join(DATA_DIR, "PAF_grid.csv"))
    df = df[df.Risk_Factor.isin(RF2PAFCAT)].copy()
    df["Risk_Factor"] = df["Risk_Factor"].replace(RF2PAFCAT)
    df["cause"] = df["Disease"].replace(DISEASE_MAP)
    df["Country"] = "Switzerland"
    df = df.rename(columns={"x": "Years", "PAF_mean": "value"})
    df["variables"] = "tcaf_health-diet_paf_" + df["cause"] + "[-]"
    var_total = [
        "tcaf_health-diet_paf_" + DISEASE_MAP[d]
        for d in sorted(DISEASE_MAP)
        if DISEASE_MAP[d] in df["cause"].unique()
    ]
    out = {}
    for rf in df.Risk_Factor.unique():
        sp = (
            df[df.Risk_Factor == rf]
            .pivot_table(
                index=["Country", "Years"], columns="variables", values="value"
            )
            .reset_index()
        )
        dm = DataMatrix.create_from_df(sp, num_cat=0)
        dm.dim_labels[1] = "Intake [g/day/cap]"
        for v in set(var_total) - set(dm.col_labels["Variables"]):
            dm.add(0.0, dummy=True, col_label=v, dim="Variables", unit="-")
        out[rf] = dm
    return out


def _diet_dm(value_fn, mean):
    a = np.zeros((1, len(YEARS), 1, len(DIET_CATS)))
    for rf, cat in RF2CAT.items():
        ci = DIET_CATS.index(cat)
        for yi, y in enumerate(YEARS):
            a[0, yi, 0, ci] = value_fn(rf, y, mean)
    base = DataMatrix(
        col_labels={
            "Country": ["Switzerland"],
            "Years": YEARS,
            "Variables": ["lfs_consumers-diet"],
            "Categories1": DIET_CATS,
        },
        units={"lfs_consumers-diet": "g/cap/day"},
    )
    d = base.copy()
    d.array = a
    return d


def make_diet(alpha, mean):
    # B = mean held at 2025 ; T = ramped linearly toward the Swiss target by 2050
    dm_B = _diet_dm(lambda rf, y, m: m[rf], mean)
    dm_T = _diet_dm(
        lambda rf, y, m: m[rf] + ((y - 2025) / 25.0) * (TARGET[rf] - m[rf]), mean
    )
    base = DataMatrix(
        col_labels={
            "Country": ["Switzerland"],
            "Years": YEARS,
            "Variables": ["share_diet_adherence"],
        },
        units={"share_diet_adherence": "-"},
    )
    dm_a = base.copy()
    dm_a.array = alpha * np.ones((1, len(YEARS), 1))
    return {
        "diet-consumed_bau": dm_B,
        "diet-consumed_target": dm_T,
        "diet-adherence": dm_a,
    }


def main():
    mean = (
        pd.read_csv(os.path.join(DATA_DIR, "Mean_Intake.csv"))
        .set_index("Diet")["mean"]
        .to_dict()
    )
    DM_TCAF = {"health-diet_paf": build_paf(), "health-diet_dalys": build_dalys()}
    r_tot = pd.read_csv(R_PAR_TOTAL) if os.path.exists(R_PAR_TOTAL) else None

    for alpha in ALPHAS:
        d, t = TCAF_health_diet_workflow(make_diet(alpha, mean), DM_TCAF, {})
        yrs = t.col_labels["Years"]
        py = pd.DataFrame(
            {
                "Year": yrs,
                "attributable": t[0, :, "tcaf_health-diet_dalys", "total"],
                "evite": t[0, :, "tcaf_health-diet_dalys-avoided", "total"],
                "residuel": t[0, :, "tcaf_health-diet_dalys-residual", "total"],
            }
        )
        print(f"\n=== alpha={alpha}  (R series: {R_SERIES.get(alpha)}) ===")
        print(
            py[py.Year.isin([2025, 2030, 2040, 2050])].to_string(
                index=False, float_format=lambda v: f"{v:,.1f}"
            )
        )
        if r_tot is not None and "serie" in r_tot:
            rr = r_tot[r_tot["serie"] == R_SERIES.get(alpha)]
            m = py.merge(rr, on="Year", suffixes=("_py", "_R"))
            for col in ["attributable", "evite", "residuel"]:
                if f"{col}_R" in m:
                    diff = (m[f"{col}_py"] - m[f"{col}_R"]).abs()
                    rel = diff / m[f"{col}_R"].replace(0, np.nan)
                    print(
                        f"  max |Δ| {col}: {diff.max():.3e}   max rel: {np.nanmax(rel.values):.3e}"
                    )


if __name__ == "__main__":
    main()
