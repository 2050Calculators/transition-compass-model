"""
Dressing Ratio Calculator
=========================
Computes kg boneless meat per kg live weight, by country and animal type.

Live weights sourced from:
  FAO Technical Conversion Factors (TCF) for Agricultural Commodities
  https://www.fao.org/fileadmin/templates/ess/documents/methodology/tcf.pdf

Production and slaughter data sourced from:
  FAOSTAT QCL dataset (Crops and Livestock Products)

For countries missing from TCF, species-level global averages are used as fallback.

Inputs:
    - data/faostat/QCL_csv/QCL_prod_meat.csv
    - data/faostat/QCL_csv/QCL_producing-livestock_meat.csv
    - data/fao_tcf/fao_technical-conversion-factors.pdf

Output:
    - dressing_ratio_results.csv

Requirements:
    pip install pandas pdfplumber

Usage:
    python dressing_ratio.py
"""

import re

import pandas as pd
import pdfplumber

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

FILES = {
    "production": "data/faostat/QCL_csv/QCL_prod_meat.csv",
    "slaughtered": "data/faostat/QCL_csv/QCL_producing-livestock_meat.csv",
    "tcf_pdf": "data/fao_tcf/fao_technical-conversion-factors.pdf",
}

YEAR_START = 1990
YEAR_END = 2023

OUTPUT_FILE = "dressing_ratio_results.csv"

# ─────────────────────────────────────────────────────────────────────────────
# SPECIES MAPPINGS
# ─────────────────────────────────────────────────────────────────────────────

# FAOSTAT item name → species label
FAOSTAT_ITEM_TO_SPECIES = {
    "Meat of cattle with the bone, fresh or chilled": "Cattle",
    "Meat of buffalo, fresh or chilled": "Buffalo",
    "Meat of sheep, fresh or chilled": "Sheep",
    "Meat of goat, fresh or chilled": "Goat",
    "Meat of pig with the bone, fresh or chilled": "Pig",
    "Meat of chickens, fresh or chilled": "Chicken",
    "Meat of turkeys, fresh or chilled": "Turkey",
    "Meat of ducks, fresh or chilled": "Duck",
    "Meat of geese, fresh or chilled": "Goose",
    "Horse meat, fresh or chilled": "Horse",
    "Meat of asses, fresh or chilled": "Ass",
    "Meat of mules, fresh or chilled": "Mule",
    "Meat of camels, fresh or chilled": "Camel",
    "Meat of rabbits and hares, fresh or chilled": "Rabbit",
}

# TCF species labels (across EN/FR/ES) → standard species label
TCF_SPECIES_MAP = {
    # English
    "cattle": "Cattle",
    "buffaloes": "Buffalo",
    "buffalo": "Buffalo",
    "sheep": "Sheep",
    "goats": "Goat",
    "pigs": "Pig",
    "chickens": "Chicken",
    "ducks": "Duck",
    "geese": "Goose",
    "turkeys": "Turkey",
    "horses": "Horse",
    "asses": "Ass",
    "mules": "Mule",
    "camels": "Camel",
    "rabbits": "Rabbit",
    # French
    "bovins": "Cattle",
    "buffles": "Buffalo",
    "ovins": "Sheep",
    "caprins": "Goat",
    "porcins": "Pig",
    "poules": "Chicken",
    "canards": "Duck",
    "oies": "Goose",
    "dindons": "Turkey",
    "chevaux": "Horse",
    "ânes": "Ass",
    "mulets": "Mule",
    "chameaux": "Camel",
    "lapins": "Rabbit",
    # Spanish
    "bovinos": "Cattle",
    "búfalos": "Buffalo",
    "ovinos": "Sheep",
    "caprinos": "Goat",
    "porcinos": "Pig",
    "gallinas": "Chicken",
    "patos": "Duck",
    "gansos": "Goose",
    "pavos": "Turkey",
    "caballos": "Horse",
    "asnos": "Ass",
    "mulos": "Mule",
    "camellos": "Camel",
    "conejos": "Rabbit",
}

# Species reported in grams/animal in TCF → convert to kg
GRAMS_SPECIES = {"Chicken", "Turkey", "Duck", "Goose", "Rabbit"}

# Items where FAOSTAT reports slaughtered animals in 1000 An
ITEMS_IN_1000_AN = {
    "Meat of chickens, fresh or chilled",
    "Meat of turkeys, fresh or chilled",
    "Meat of ducks, fresh or chilled",
    "Meat of geese, fresh or chilled",
    "Meat of rabbits and hares, fresh or chilled",
}

# TCF country name → FAOSTAT country name
TCF_COUNTRY_MAP = {
    "ALBANIE": "Albania",
    "ALGÉRIE": "Algeria",
    "BELARUS, REP. OF": "Belarus",
    "BELGIQUE-LUXEMBOURG": "Belgium",
    "BÉNIN": "Benin",
    "BOLIVIA": "Bolivia (Plurinational State of)",
    "BULGARIE": "Bulgaria",
    "CAMBODGE": "Cambodia",
    "CAMEROUN": "Cameroon",
    "CONGO, RÉPUBLIQUE DE": "Congo",
    "CZECHK.REP": "Czechia",
    "ESTONIA, REP. OF": "Estonia",
    "GRÈCE": "Greece",
    "GUINEA ECUATORIAL": "Equatorial Guinea",
    "GUINÉE": "Guinea",
    "GUINÉE-BISSAU": "Guinea-Bissau",
    "GUYANE FRANÇAISE": "French Guyana",
    "HAÏTI": "Haiti",
    "IRAN, ISLAMIC REP. OF": "Iran (Islamic Republic of)",
    "ITALIE": "Italy",
    "KOREA,DEM. PEOP. REP. OF": "Democratic People's Republic of Korea",
    "LAOS": "Lao People's Democratic Republic",
    "LATVIA, REP. OF": "Latvia",
    "LIBAN": "Lebanon",
    "LITHUANIA, REP. OF": "Lithuania",
    "MÉXICO": "Mexico",
    "MOLDOVA, REP. OF": "Republic of Moldova",
    "PANAMÁ": "Panama",
    "REPÚBLICA DOMINICANA": "Dominican Republic",
    "ROUMANIE": "Romania",
    "SAINT VINCENT/GRENADINES": "Saint Vincent and the Grenadines",
    "SAUDI ARABIA, KINGDOM OF": "Saudi Arabia",
    "SÉNÉGAL": "Senegal",
    "SWAZILAND": "Eswatini",
    "SYRIA": "Syrian Arab Republic",
    "TANZANIA": "United Republic of Tanzania",
    "TCHAD": "Chad",
    "TUNISIE": "Tunisia",
    "TURKEY": "Türkiye",
    "VENEZUELA": "Venezuela (Bolivarian Republic of)",
    "YUGOSLAVIA,FED.REP.": "Serbia",
    "NETHERLANDS": "Netherlands (Kingdom of the)",
}

# Parsing artifacts to ignore
TCF_ARTIFACTS = {
    "% %",
    "ANIMAUX D'ÉLEVAGE",
    "AVERAGE AVERAGE CARCASS WEIGHT AS",
    "BIRTH RATE TAKE-OFF RATE",
    "COEFFICIENT COEFFICIENT",
    "DE NATALIDAD DE MATANZAS",
    "DE NATALITÉ D'ABATTAGES",
    "FRUITS ET BAIES",
    "GANADERÍA",
    "KG O GR/AN KG O GR/AN %",
    "KG OR GR/AN KG OR GR/AN %",
    "KG OU GR/AN KG OU GR/AN %",
    "LIVE WEIGHT CARCASS WEIGHT % OF LIVE WEIGHT",
    "LIVESTOCK NUMBERS",
    "LIVESTOCK PRODUCTS",
    "LÉGUMES ET MELONS",
    "MEAT",
    "MOYEN DE LA CARCASSE EN % DU POIDS-VIF",
    "OIL-BEARING CROPS",
    "PRODUCTOS PECUARIOS",
    "PRODUITS DE L'ÉLEVAGE",
    "PULSES",
    "VIANDE",
}

# ─────────────────────────────────────────────────────────────────────────────
# BONELESS YIELD FROM CARCASS (fixed, FAO technical conversion factors)
# ─────────────────────────────────────────────────────────────────────────────

BONELESS_YIELD = {
    "Cattle": 0.70,
    "Buffalo": 0.70,
    "Sheep": 0.65,
    "Goat": 0.65,
    "Pig": 0.85,
    "Chicken": 0.80,
    "Turkey": 0.80,
    "Duck": 0.75,
    "Goose": 0.75,
    "Horse": 0.70,
    "Ass": 0.70,
    "Mule": 0.70,
    "Camel": 0.70,
    "Rabbit": 0.72,
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: PARSE FAO TCF PDF → live weight by country & species
# ─────────────────────────────────────────────────────────────────────────────


def parse_tcf_pdf(pdf_path):
    print(f"Parsing FAO TCF PDF: {pdf_path}")
    results = []
    current_country = None
    in_meat_table = False

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            lines = [l.strip() for l in text.split("\n") if l.strip()]

            for i, line in enumerate(lines):
                line_lower = line.lower()

                # Detect country name (line after "Technical Conversion Factors" header)
                if i > 0 and lines[i - 1].strip() == "Technical Conversion Factors":
                    candidate = line.strip()
                    if (
                        len(candidate) < 50
                        and not any(c.isdigit() for c in candidate)
                        and candidate.upper() not in TCF_ARTIFACTS
                        and not candidate.lower().startswith(
                            (
                                "crops",
                                "cultures",
                                "cultivos",
                                "cereals",
                                "céréales",
                                "cereales",
                                "à l",
                                "these",
                                "page",
                                "introduction",
                            )
                        )
                    ):
                        current_country = candidate.upper()
                        in_meat_table = False

                # Detect meat table start
                if any(
                    h in line_lower
                    for h in ["kg or gr/an", "kg ou gr/an", "kg o gr/an"]
                ):
                    in_meat_table = True
                    continue

                # Detect meat table end
                if in_meat_table and any(
                    h in line_lower
                    for h in [
                        "edible offal",
                        "abats comestibles",
                        "despojos comest",
                        "slaughter fat",
                        "graisses d'abattage",
                        "grasas de matadero",
                        "hides",
                        "cuirs",
                        "cueros",
                        "milk",
                        "lait",
                        "leche",
                    ]
                ):
                    in_meat_table = False

                # Parse meat rows: Species  live_wt  carcass_wt  pct
                if (
                    in_meat_table
                    and current_country
                    and current_country not in TCF_ARTIFACTS
                ):
                    m = re.match(
                        r"^([A-Za-zÀ-ÿ\s\+]+?)\s+"
                        r"(\d+(?:\.\d+)?)\s+"
                        r"(\d+(?:\.\d+)?)\s+"
                        r"(\d+(?:\.\d+)?)\s*$",
                        line,
                    )
                    if m:
                        species_raw = m.group(1).strip().lower()
                        live_wt = float(m.group(2))
                        carc_wt = float(m.group(3))
                        carc_pct = float(m.group(4))
                        species = TCF_SPECIES_MAP.get(species_raw)
                        if species:
                            if species in GRAMS_SPECIES:
                                live_wt /= 1000
                                carc_wt /= 1000
                            # Map TCF country name to FAOSTAT country name
                            country_faostat = TCF_COUNTRY_MAP.get(
                                current_country, current_country.title()
                            )
                            results.append(
                                {
                                    "country": country_faostat,
                                    "species": species,
                                    "live_weight_kg": live_wt,
                                }
                            )

    df_tcf = pd.DataFrame(results)
    # Keep one row per country × species (some countries appear multiple times)
    df_tcf = (
        df_tcf.groupby(["country", "species"])["live_weight_kg"].mean().reset_index()
    )
    print(
        f"  → {len(df_tcf)} records, {df_tcf['country'].nunique()} countries, "
        f"{df_tcf['species'].nunique()} species"
    )
    return df_tcf


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: LOAD & CLEAN FAOSTAT CSVs
# ─────────────────────────────────────────────────────────────────────────────


def load_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


def filter_and_prepare(df, items, year_start, year_end):
    df = df[(df["Year"] >= year_start) & (df["Year"] <= year_end)].copy()
    df = df[df["Item"].isin(items)].copy()
    df = df[df["Value"].notna() & (df["Value"] > 0)]
    return df


def aggregate_by_year(df, value_col):
    return (
        df.groupby(["Area Code (M49)", "Area", "Item", "Year"])["Value"]
        .mean()
        .reset_index()
        .rename(columns={"Value": value_col})
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # --- Parse TCF PDF ---
    df_tcf = parse_tcf_pdf(FILES["tcf_pdf"])

    # Remove zero/invalid live weights from TCF
    df_tcf = df_tcf[df_tcf["live_weight_kg"] > 0]

    # Compute global species average as fallback for missing countries
    species_global_avg = df_tcf.groupby("species")["live_weight_kg"].mean().to_dict()

    # Add IPCC defaults for species not in TCF (Ass, Mule)
    IPCC_FALLBACK = {"Ass": 130.0, "Mule": 130.0}
    for sp, wt in IPCC_FALLBACK.items():
        if sp not in species_global_avg:
            species_global_avg[sp] = wt

    # --- Load FAOSTAT ---
    print("\nLoading FAOSTAT CSV files...")
    df_prod = load_csv(FILES["production"])
    df_slau = load_csv(FILES["slaughtered"])
    print(f"  Production rows:  {len(df_prod):,}")
    print(f"  Slaughtered rows: {len(df_slau):,}")

    items = list(FAOSTAT_ITEM_TO_SPECIES.keys())
    df_prod = filter_and_prepare(df_prod, items, YEAR_START, YEAR_END)
    df_slau = filter_and_prepare(df_slau, items, YEAR_START, YEAR_END)

    # Fix 1000 An unit for poultry/rabbits
    mask = df_slau["Item"].isin(ITEMS_IN_1000_AN)
    df_slau.loc[mask, "Value"] = df_slau.loc[mask, "Value"] * 1000

    df_prod_yr = aggregate_by_year(df_prod, "production_tonnes")
    df_slau_yr = aggregate_by_year(df_slau, "animals_slaughtered")

    # Merge production + slaughtered
    df = pd.merge(
        df_prod_yr,
        df_slau_yr,
        on=["Area Code (M49)", "Area", "Item", "Year"],
        how="inner",
    )

    print(f"\n  Matched rows (country × item × year): {len(df):,}")

    # Add species label
    df["species"] = df["Item"].map(FAOSTAT_ITEM_TO_SPECIES)

    # --- Merge FAO TCF live weights ---
    df = pd.merge(
        df,
        df_tcf.rename(
            columns={"country": "Area", "live_weight_kg": "live_weight_kg_tcf"}
        ),
        on=["Area", "species"],
        how="left",
    )

    # Fill missing TCF values with global species average
    df["live_weight_source"] = "FAO TCF (country-specific)"
    missing_mask = df["live_weight_kg_tcf"].isna()
    df.loc[missing_mask, "live_weight_kg_tcf"] = df.loc[missing_mask, "species"].map(
        species_global_avg
    )
    df.loc[missing_mask, "live_weight_source"] = "FAO TCF (species average fallback)"

    n_tcf = (~missing_mask).sum()
    n_fallbk = missing_mask.sum()
    print(
        f"\n  Live weight source: {n_tcf:,} rows from TCF country values, "
        f"{n_fallbk:,} rows from species average fallback"
    )

    # --- Compute metrics ---

    # Carcass weight per animal from FAOSTAT (kg)
    df["carcass_weight_kg"] = (
        df["production_tonnes"] * 1000 / df["animals_slaughtered"]
    ).round(3)

    # Dressing % from FAOSTAT carcass + TCF live weight
    df["dressing_pct"] = (
        df["carcass_weight_kg"] / df["live_weight_kg_tcf"] * 100
    ).round(1)

    # Boneless yield factor (fixed per species)
    df["boneless_yield_factor"] = df["species"].map(BONELESS_YIELD)

    # Boneless meat per animal (kg)
    df["boneless_meat_kg"] = (
        df["carcass_weight_kg"] * df["boneless_yield_factor"]
    ).round(3)

    # KEY METRIC: kg boneless meat per kg live weight
    df["kg_boneless_per_kg_liveweight"] = (
        df["boneless_meat_kg"] / df["live_weight_kg_tcf"]
    ).round(3)

    # Inverse: kg live weight per kg boneless meat
    df["kg_liveweight_per_kg_boneless"] = (
        df["live_weight_kg_tcf"] / df["boneless_meat_kg"]
    ).round(3)

    # --- Final output ---
    df = (
        df[
            [
                "species",
                "Area Code (M49)",
                "Area",
                "Item",
                "Year",
                "production_tonnes",
                "animals_slaughtered",
                "carcass_weight_kg",
                "live_weight_kg_tcf",
                "live_weight_source",
                "dressing_pct",
                "boneless_yield_factor",
                "boneless_meat_kg",
                "kg_boneless_per_kg_liveweight",
                "kg_liveweight_per_kg_boneless",
            ]
        ]
        .rename(
            columns={
                "Area Code (M49)": "area_code",
                "Area": "country",
                "Item": "faostat_item",
                "Year": "year",
            }
        )
        .sort_values(["species", "country", "year"])
    )

    df.to_csv(OUTPUT_FILE, index=False)

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY: Average values by species (all countries, all years)")
    print("=" * 80)
    summary = (
        df.groupby("species")
        .agg(
            n_rows=("country", "count"),
            avg_live_weight_kg=("live_weight_kg_tcf", "mean"),
            avg_carcass_kg=("carcass_weight_kg", "mean"),
            avg_dressing_pct=("dressing_pct", "mean"),
            boneless_yield_factor=("boneless_yield_factor", "first"),
            avg_kg_boneless_per_kg_liveweight=("kg_boneless_per_kg_liveweight", "mean"),
        )
        .round(2)
    )
    print(summary.to_string())
    print(f"\n✓ Results saved to: {OUTPUT_FILE}")
    print(f"  Total rows:        {len(df):,}")
    print(f"  Species covered:   {df['species'].nunique()}")
    print(f"  Countries covered: {df['country'].nunique()}")

    return df


if __name__ == "__main__":
    main()
