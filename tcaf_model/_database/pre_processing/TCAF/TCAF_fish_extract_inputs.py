"""Extract the raw inputs of TCAF_fish_preprocessing.py into data/blue-food/.

Sources (none of them is in this repo):
  - the TCAF_fisheries_aquaculture project: physical ReCiPe impacts per kg of the
    LCA processes (outputs/lca_impacts_per_kg_wide.parquet), the ISSCAAP group ->
    division map (outputs/lca_process_map_by_group_method.csv) and the FishStatJ
    production file (data/FishstatJ/...);
  - the FAOSTAT bulk files FoodBalanceSheets_E_All_Data_(Normalized).csv (FBS,
    2010 on) and FoodBalanceSheetsHistoric_E_All_Data_(Normalized).csv (FBSH,
    to 2013) from https://bulks-faostat.fao.org/production/ (the FAOSTAT API now
    needs a token).

Usage:
  python TCAF_fish_extract_inputs.py <fisheries_repo> <FBS_normalized.csv> <FBSH_normalized.csv>

FBS is used from 2014 and FBSH before, because FBS is rounded to 1000 t from 2010
while FBSH keeps two decimals to 2013.
"""

import os
import sys

import numpy as np
import pandas as pd

DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "blue-food")

IMPACTS = [
    "fine-particulate-matter-formation",
    "fossil-resource-scarcity",
    "freshwater-ecotoxicity",
    "freshwater-eutrophication",
    "global-warming",
    "marine-ecotoxicity",
    "marine-eutrophication",
    "mineral-resource-scarcity",
    "ozone-formation,-human-health",
    "ozone-formation,-terrestrial-ecosystems",
    "stratospheric-ozone-depletion",
    "terrestrial-acidification",
    "terrestrial-ecotoxicity",
    "water-consumption",
]
YEARS = range(1990, 2024)
FBS_ELEMENTS = {
    "Production": "production_kt",
    "Import quantity": "import_kt",
    "Export quantity": "export_kt",
    "Stock Variation": "stock_variation_kt",
    "Food": "food_kt",
    "Feed": "feed_kt",
    "Processed": "processed_kt",
    "Other uses (non-food)": "other_uses_kt",
}


def lca_processes(repo):
    """Aquaculture processes plus the passive-gear (trammel net) capture process."""
    w = pd.read_parquet(os.path.join(repo, "outputs/lca_impacts_per_kg_wide.parquet"))
    long = pd.read_parquet(
        os.path.join(repo, "outputs/lca_impacts_per_kg_long.parquet")
    )
    unit = long.groupby("Process")["functional_unit"].agg(
        lambda s: ";".join(sorted(set(map(str, s))))
    )
    pm = pd.read_csv(os.path.join(repo, "outputs/lca_process_map_by_group_method.csv"))
    division = dict(zip(pm["isscaap_group"], pm["division"]))

    sel = w[
        (w["lca_production_env"] == "Aquaculture") | (w["lca_gear"] == "Trammel net")
    ].copy()
    sel["functional_unit"] = sel["Process"].map(unit)
    sel["division"] = sel["ISSCAAP group"].map(division)
    assert sel["division"].notna().all(), "ISSCAAP group without a division"
    assert sel["functional_unit"].str.contains("kg").all(), "process not per kg"
    out = sel[
        [
            "Process",
            "lca_production_env",
            "lca_gear",
            "ISSCAAP group",
            "division",
            "functional_unit",
        ]
        + IMPACTS
    ].rename(
        columns={
            "ISSCAAP group": "isscaap_group",
            "lca_production_env": "production_env",
        }
    )
    out.to_csv(os.path.join(DEST, "fish_lca_processes_per_kg_whole.csv"), index=False)
    return out


def fishstatj_switzerland(repo):
    """Swiss freshwater + diadromous production [t live weight] by group and source.

    These two FishStatJ divisions are what FAOSTAT's FBS calls Freshwater Fish
    (the model's `ffish`).
    """
    f = os.path.join(
        repo,
        "data/FishstatJ/fishstatj_global-fishery-and-aquaculture-production-statistics.csv",
    )
    df = pd.read_csv(f, low_memory=False)
    c = list(df.columns)
    ch = df[df.iloc[:, 1] == "CH"].copy()
    ch["division"] = ch[c[7]]
    ch["isscaap_group"] = ch[c[8]]
    ch["source"] = ch[c[17]].map(
        {"Capture production": "capture", "Aquaculture production": "aquaculture"}
    )
    ch = ch[ch["division"].isin(["Freshwater fishes", "Diadromous fishes"])]
    rows = []
    for y in YEARS:
        ch["t"] = pd.to_numeric(ch[f"[{y}]"], errors="coerce").fillna(0.0)
        g = ch.groupby(["isscaap_group", "division", "source"])["t"].sum().reset_index()
        g["year"] = y
        rows.append(g)
    out = (
        pd.concat(rows)[["year", "isscaap_group", "division", "source", "t"]]
        .rename(columns={"t": "tonnes_live_weight"})
        .sort_values(["year", "source", "isscaap_group"])
    )
    out.to_csv(os.path.join(DEST, "fish_ch_production_fishstatj.csv"), index=False)
    return out


def fbs_switzerland(fbs_csv, fbsh_csv):
    frames = []
    for tag, path in (("FBS", fbs_csv), ("FBSH", fbsh_csv)):
        d = pd.read_csv(path, encoding="latin-1", low_memory=False)
        d = d[(d["Area"] == "Switzerland") & (d["Item"] == "Freshwater Fish")].copy()
        d["dataset"] = tag
        frames.append(d)
    d = pd.concat(frames)
    d["Element"] = (
        d["Element"]
        .str.replace("Import Quantity", "Import quantity")
        .str.replace("Export Quantity", "Export quantity")
    )
    d = d[d["Element"].isin(FBS_ELEMENTS)]
    d["var"] = d["Element"].map(FBS_ELEMENTS)
    d = d[
        (d["dataset"] == np.where(d["Year"] <= 2013, "FBSH", "FBS"))
        & d["Year"].between(YEARS[0], YEARS[-1])
    ]
    p = (
        d.pivot_table(index=["Year", "dataset"], columns="var", values="Value")
        .reset_index()
        .rename(columns={"Year": "year", "dataset": "source"})
    )
    cols = ["year", "source"] + list(FBS_ELEMENTS.values())
    for col in cols:
        if col not in p:
            p[col] = np.nan
    p = p[cols]
    p.to_csv(os.path.join(DEST, "fish_ch_fbs_freshwater-fish.csv"), index=False)
    return p


def main(repo, fbs_csv, fbsh_csv):
    os.makedirs(DEST, exist_ok=True)
    print(len(lca_processes(repo)), "LCA processes")
    print(len(fishstatj_switzerland(repo)), "FishStatJ rows")
    print(len(fbs_switzerland(fbs_csv, fbsh_csv)), "FBS years")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
