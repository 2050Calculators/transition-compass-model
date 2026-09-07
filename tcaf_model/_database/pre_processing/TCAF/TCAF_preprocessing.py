import os
import pickle
import re
import unicodedata

import numpy as np
import pandas as pd
import pycountry

from tcaf_model.model.common.auxiliary_functions import (
    add_and_fill_missing_countries_dm,
    create_years_list,
    dm_match_countries,
    harmonize_countries,
    linear_fitting,
)
from tcaf_model.model.common.constant_data_matrix_class import (
    ConstantDataMatrix,
)
from tcaf_model.model.common.data_matrix_class import DataMatrix
from tcaf_model.model.common.io_database import (
    database_to_df_robust,
)

# from _database.pre_processing.api_routines_CH import get_data_api_CH

# CalculationLeaf other functions


def normalize(name):
    """Remove accents and normalize string."""
    name = name.strip()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return name


def name_to_iso3(name):
    """Convert country name to ISO3 code."""
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def match_countries_iso3(list_faostat, list_biodiversity):
    """
    Convert both country lists to ISO3 codes
    and return mapping biodiversity_name -> faostat_name
    """

    # --- Convert FAOSTAT countries to ISO3 ---
    faostat_iso = {}
    for country in list_faostat:
        iso = name_to_iso3(normalize(country))
        if iso:
            faostat_iso[iso] = country

    # --- Convert biodiversity countries to ISO3 and match ---
    mapping = {}
    unmatched = []

    for country in list_biodiversity:
        iso = name_to_iso3(normalize(country))

        if iso and iso in faostat_iso:
            mapping[country] = faostat_iso[iso]
        else:
            mapping[country] = None
            unmatched.append(country)

    return mapping, unmatched


# SimulateInteractions crop to TCAF
def simulate_crop_to_TCAF_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(
        current_file_directory, "../_database/data/interface/crop_to_TCAF.pickle"
    )
    with open(f, "rb") as handle:
        dm_production = pickle.load(handle)
    return dm_production


# SimulateInteractions land-use to TCAF
def simulate_landuse_to_TCAF_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    f = os.path.normpath(
        os.path.join(
            current_file_directory, "../../data/interface/land-use_to_TCAF.pickle"
        )
    )
    with open(f, "rb") as handle:
        DM_landuse_to_TCAF = pickle.load(handle)
    return DM_landuse_to_TCAF


# CalculationLeaf TCAF MONETIZATION FACTORS
def TCAF_MF_preprocessing():

    # Data -----------------------------------------------------------------------
    df_data = pd.read_excel(
        "data/monetization-factors/TCAF_monetization-factors.xlsx", sheet_name="MFs"
    )
    df_data = df_data[["name", "value"]]

    # Format as constant datamatrix
    CDM_MF = ConstantDataMatrix.create_from_constant(df_data, num_cat=0)
    return CDM_MF


# CalculationLeaf TCAF - Health diet
def TCAF_health_diet_preprocessing():

    # ----------------------------------------------------------------------------
    # Disease name -> short code (shared by DALYs and PAF so they align on merge)
    # ----------------------------------------------------------------------------
    disease_map = {
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

    # ----------------------------------------------------------------------------
    # DALYs
    # ----------------------------------------------------------------------------
    # The projected DALYs are no longer read from Projected_DALYs.csv. They are now
    # computed at runtime in TCAF_health_diet_workflow (frozen-rate demographic
    # projection): the 2023 GBD DALYs by age x sex x cause are projected with the
    # model's live demography. Here we only build the static 2023 GBD table; it is
    # constructed after the PAF block below so it can be aligned to the PAF disease
    # set (see end of function).

    # ----------------------------------------------------------------------------
    # PAF  (dose-response grid: PAF as a function of intake x [g/day/cap])
    # ----------------------------------------------------------------------------

    # Data -----------------------------------------------------------------------
    # cols: cause, Disease, Risk_Factor, x, PAF_mean, PAF_lower, PAF_upper, xmax, floor
    df_paf = pd.read_csv("data/health-diet-v2/PAF_grid.csv")

    # Preprocessing --------------------------------------------------------------
    # Keep only the dietary risk factors used by the model (matches Projection.R)
    risk_factor_map = {
        "Fruits": "crop-fruit",
        "Vegetables": "crop-veg",
        "Whole_Grains": "crop-cereal-whole",
        "Nuts": "crop-oilcrop",
        "Legumes": "crop-pulse",
        "Milk": "pro-liv-abp-dairy-milk",
        "Red_Meat": "pro-liv-meat-red",
        "Processed_Meat": "pro-liv-meat-processed",
    }
    df_paf = df_paf[df_paf["Risk_Factor"].isin(risk_factor_map.keys())].copy()

    # Rename terms
    df_paf["Risk_Factor"] = df_paf["Risk_Factor"].replace(risk_factor_map)
    df_paf["cause"] = df_paf["Disease"].replace(disease_map)
    df_paf["Country"] = "Switzerland"

    # The intake grid x is stored on the 'Years' axis and renamed afterwards.
    # Only the mean PAF is used by the workflow for now; PAF_lower / PAF_upper are
    # available in the source file if bounds are needed later.
    df_paf = df_paf.rename(columns={"x": "Years", "PAF_mean": "value"})

    # Create variables name
    df_paf["variables"] = "tcaf_health-diet_paf_" + df_paf["cause"] + "[-]"

    # Format as separate dm, according to the risk factor (or food categories)
    # Note: here, the intake is processed as the 'Years' dimension, and renamed
    # afterwards. Therefore, this DM has no timescale.
    DM_TCAF_health_diet_paf = {}

    # Full set of disease codes present across the kept risk factors: any disease
    # a given food does not affect is padded with PAF = 0 (contributes a factor
    # (1 - 0) = 1 to the multiplicative combination, i.e. no effect).
    var_total = [
        "tcaf_health-diet_paf_" + disease_map[d]
        for d in sorted(disease_map)
        if disease_map[d] in df_paf["cause"].unique()
    ]

    for rf in df_paf["Risk_Factor"].unique():
        sub_df = df_paf[df_paf["Risk_Factor"] == rf].copy()
        sub_df_pivot = sub_df.pivot_table(
            index=["Country", "Years"], columns="variables", values="value"
        ).reset_index()
        dm = DataMatrix.create_from_df(sub_df_pivot, num_cat=0)
        dm.dim_labels[1] = "Intake [g/day/cap]"
        # Add dummies for diseases not affected by this risk factor
        var_rf = dm.col_labels["Variables"]
        var_missing = set(var_total) - set(var_rf)
        for var in var_missing:
            dm.add(0.0, dummy=True, col_label=var, dim="Variables", unit="-")
        DM_TCAF_health_diet_paf[rf] = dm

    # ----------------------------------------------------------------------------
    # DALYs - static 2023 GBD table (numerator of the frozen-rate projection)
    # ----------------------------------------------------------------------------
    # Mirrors 03_projection_dalys.R. Auto-detects the stratification of the input:
    #   - if the file carries real age bands  -> table by disease x age x sex
    #   - otherwise ("All ages")              -> table by disease x sex
    # The workflow (_project_dalys) forms the 2025 per-capita rate and projects it
    # with the model's demography at the matching stratification. Age / sex labels
    # use the demography tokens (lfs_demography_<sex>-<age>) so they join directly.
    # Only diseases present in BOTH the DALYs file and the PAF grid are kept.
    paf_codes = [v.replace("tcaf_health-diet_paf_", "") for v in var_total]

    df_gbd = pd.read_csv("data/health-diet-v2/Disease_sex_DALYs_2023.csv")
    # harmonize disease naming (GBD uses a slightly different label)
    df_gbd["cause"] = df_gbd["cause"].replace(
        {"Tracheal, bronchus, and lung cancer": "Tracheal bronchus and lung cancer"}
    )
    # keep a single measure / metric / year (defensive, as in the R script)
    if "measure" in df_gbd.columns:
        df_gbd = df_gbd[df_gbd["measure"].astype(str).str.contains("DALY", na=False)]
    if "metric" in df_gbd.columns:
        df_gbd = df_gbd[df_gbd["metric"] == "Number"]
    if "year" in df_gbd.columns:
        df_gbd = df_gbd[df_gbd["year"] == 2023]
    # if 'location' in df_gbd.columns:
    #   df_gbd = df_gbd[df_gbd['location'] == 'Switzerland']

    df_gbd = df_gbd.rename(columns={"age": "Age", "sex": "Sex", "val": "DALYs"})
    df_gbd = df_gbd[df_gbd["Sex"].isin(["Male", "Female"])].copy()
    df_gbd["Sex"] = df_gbd["Sex"].map({"Male": "male", "Female": "female"})
    df_gbd["code"] = df_gbd["cause"].replace(disease_map)
    df_gbd = df_gbd[df_gbd["code"].isin(paf_codes)]

    # Diseases actually available (intersection with the PAF set), PAF ordering
    diseases = [c for c in paf_codes if c in set(df_gbd["code"].unique())]
    model_sexes = ["female", "male"]

    # Detect real age bands
    _non_age = {"All ages", "All Ages", "Age-standardized", "all ages", "nan"}
    if "Age" in df_gbd.columns:
        df_gbd["Age"] = df_gbd["Age"].astype(str).str.replace(" years", "", regex=False)
        real_ages = [a for a in df_gbd["Age"].unique() if a not in _non_age]
    else:
        real_ages = []

    if len(real_ages) > 0:
        # ---- age x sex table -----------------------------------------------------
        def _age_token(a):
            if a in ("<5", "5-9", "10-14", "15-19"):
                return "below19"
            if a in ("20-24", "25-29"):
                return "age20-29"
            if a in ("30-34", "35-39", "40-44", "45-49", "50-54"):
                return "age30-54"
            if a in ("55-59", "60-64"):
                return "age55-64"
            return "above65"

        df_gbd = df_gbd[~df_gbd["Age"].isin(_non_age)].copy()
        df_gbd["Age"] = df_gbd["Age"].map(_age_token)
        df_gbd = df_gbd.groupby(["Age", "Sex", "code"], as_index=False)["DALYs"].sum()

        model_ages = ["below19", "age20-29", "age30-54", "age55-64", "above65"]
        arr = np.zeros((1, 1, 1, len(diseases), len(model_ages), len(model_sexes)))
        di = {d: i for i, d in enumerate(diseases)}
        ai = {a: i for i, a in enumerate(model_ages)}
        si = {s: i for i, s in enumerate(model_sexes)}
        for row in df_gbd.itertuples(index=False):
            arr[0, 0, 0, di[row.code], ai[row.Age], si[row.Sex]] = row.DALYs
        dm_health_dalys = DataMatrix(
            col_labels={
                "Country": ["Switzerland"],
                "Years": [2023],
                "Variables": ["tcaf_health-diet_dalys"],
                "Categories1": diseases,
                "Categories2": model_ages,
                "Categories3": model_sexes,
            },
            units={"tcaf_health-diet_dalys": "DALYs/y"},
        )
        dm_health_dalys.array = arr
    else:
        # ---- sex-only table (no age resolution: projection scales by sex totals) --
        df_gbd = df_gbd.groupby(["Sex", "code"], as_index=False)["DALYs"].sum()
        arr = np.zeros((1, 1, 1, len(diseases), len(model_sexes)))
        di = {d: i for i, d in enumerate(diseases)}
        si = {s: i for i, s in enumerate(model_sexes)}
        for row in df_gbd.itertuples(index=False):
            arr[0, 0, 0, di[row.code], si[row.Sex]] = row.DALYs
        dm_health_dalys = DataMatrix(
            col_labels={
                "Country": ["Switzerland"],
                "Years": [2023],
                "Variables": ["tcaf_health-diet_dalys"],
                "Categories1": diseases,
                "Categories2": model_sexes,
            },
            units={"tcaf_health-diet_dalys": "DALYs/y"},
        )
        dm_health_dalys.array = arr

    return DM_TCAF_health_diet_paf, dm_health_dalys


# CalculationLeaf TCAF - Biodiversity


def TCAF_biodiversity_preprocessing():
    import sys

    import tcaf_model.model.common.data_matrix_class as dmc

    sys.modules["common.data_matrix_class"] = dmc

    # Read biodiversity_world.csv from TCAF Datapool
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "data/data_pool/biodiversity_world.csv")
    df_biodiversity_world = pd.read_csv(f)

    # Read biodiversity_switzerland.csv from TCAF Datapool
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(
        current_file_directory, "data/data_pool/biodiversity_switzerland.csv"
    )
    df_biodiversity_ch = pd.read_csv(f)

    # Format as Datamatrix (CH)
    lever = "dummy"
    df_biodiversity_ch["lever"] = lever
    df_ots, df_fts = database_to_df_robust(df_biodiversity_ch, lever, level="all")
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm_biodiversity_ch = DataMatrix.create_from_df(df_ots, num_cat=2)
    dm_biodiversity_ch.switch_categories_order(cat1="Categories2", cat2="Categories1")
    dm_biodiversity_ch.rename_col_regex("crop-", "", dim="Categories2")

    # Format as Datamatrix (world)
    lever = "dummy"
    df_biodiversity_ch["lever"] = lever
    df_ots, df_fts = database_to_df_robust(df_biodiversity_world, lever, level="all")
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm_biodiversity_world = DataMatrix.create_from_df(df_ots, num_cat=0)

    # Fixme change unit eco-cost EUR2024 to CHF

    # Create copies for to divide  "baltic states" in ["Estonia", "Latvia", "Lithuania"]
    for country_baltic in ["Estonia", "Latvia", "Lithuania"]:
        dm_biodiversity_world.add(
            0.0, dummy=True, col_label=country_baltic, dim="Country"
        )
        dm_biodiversity_world[country_baltic, :, :] = dm_biodiversity_world[
            "baltic states", :, :
        ]
    dm_biodiversity_world.drop(dim="Country", col_label="baltic states")

    # Read pickle from landuse_module to TCAF
    DM_landuse_to_TCAF = simulate_landuse_to_TCAF_input()
    dm_cropland = DM_landuse_to_TCAF["cropland-world"]

    # Format country names to match the ones in dm_production

    # Manual fixes for typos & alternative names
    # -----------------------------
    mapping_manual = {
        "hungaria": "Hungary",
        "sri lanca": "Sri Lanka",
        "mauretania": "Mauritania",
        "tunesia": "Tunisia",
        "ivory coast": "Côte d'Ivoire",
        "zaire": "Democratic Republic of the Congo",
        "south korea": "Republic of Korea",
        "north korea": "Democratic People's Republic of Korea",
        "russia": "Russian Federation",
        "bolivia": "Bolivia",
        "netherlands": "Netherlands (kingdom of the)",
        # depending on your ISO mapping, could be "Bolivia (Plurinational State of)"
        "iran": "Iran",
        "venezuela": "Venezuela (Bolivarian Republic of)",
        "turkey": "Türkiye",
        "us": "United States of America",
        "uk": "United Kingdom of Great Britain and Northern Ireland",
        "dominican rep": "Dominican Republic",
        "greenland": "Greenland",
        "new guinea": "Papua New Guinea",
        "surinam": "Suriname",
        "china": "China, mainland",
    }

    # Rename with manual fixes
    for key in mapping_manual.keys():
        dm_biodiversity_world.rename_col(key, mapping_manual[key], "Country")

    list_faostat = dm_cropland.col_labels["Country"]
    list_biodiversity = dm_biodiversity_world.col_labels["Country"]

    # Rename with ISO3 codes
    mapping, unmatched = harmonize_countries(list_biodiversity, list_faostat)

    # Group same country when necessary using mean values
    dm_biodiversity_world.groupby(
        {"united states of america": "united states.*"},
        dim="Country",
        aggregation="mean",
        regex=True,
        inplace=True,
    )
    dm_biodiversity_world.groupby(
        {"indonesia": "indonesia.*"},
        dim="Country",
        aggregation="mean",
        regex=True,
        inplace=True,
    )

    # Format country names to match the ones in dm_production
    for key in mapping.keys():
        if mapping[key] is not None:
            dm_biodiversity_world.rename_col(key, mapping[key], "Country")

    # Add missing countries with dummy values
    dm_match_countries(dm_biodiversity_world, dm_cropland, parameter="perfect match")

    # Format separately between Switzerland and other countries
    dm_biodiversity_world.drop(dim="Country", col_label="Switzerland")
    DM_TCAF_biodiversity = {
        "TCAF-biodiversity-CH": dm_biodiversity_ch,
        "TCAF-biodiversity-world": dm_biodiversity_world,
    }

    # Change unti from money/m2 to money/ha
    for key in DM_TCAF_biodiversity.keys():
        old_unit = DM_TCAF_biodiversity[key].units["eco-cost"]
        DM_TCAF_biodiversity[key].change_unit(
            "eco-cost", old_unit=old_unit, new_unit="CHF/ha", factor=10 ** (4)
        )

    return DM_TCAF_biodiversity


# CalculationLeaf TCAF - LCA MONETIZATION

# Crosswalk: ReCiPe 2016 midpoint category (as stored on the LCA impact axis)
# -> monetization-factor 'Variable' stem (text before '['). Only categories that
# can be monetized are listed here; any impact category NOT in this dict is
# dropped from the monetized matrix (it cannot be converted to money).
#
# EXCLUDED ON PURPOSE (do not add without a proper midpoint-level factor):
#   - human-carcinogenic-toxicity / human-non-carcinogenic-toxicity:
#       ReCiPe midpoint is kg 1,4-DCB, the only toxicity factor here is
#       'human-toxicity' per DALY (endpoint) -> dimensionally incompatible.
#   - land-use: ReCiPe m2a crop-eq vs 'land-occupation-*' per MSA*ha*yr,
#       biome-split -> not directly multipliable.
#   - ionizing-radiation: no monetization factor available.
# APPROXIMATE (accepted for now):
#   - mineral-resource-scarcity -> other-non-renewable-material-depletion
#   - water-consumption         -> scarce-blue-water-use (AWARE-weighted m3)
_LCA_MF_CROSSWALK = {
    "global-warming": "climate-change-ghg",
    "terrestrial-acidification": "acidification",
    "freshwater-eutrophication": "freshwater-eutrophication",
    "marine-eutrophication": "marine-eutrophication",
    "freshwater-ecotoxicity": "freshwater-ecotoxicity",
    "marine-ecotoxicity": "marine-ecotoxicity",
    "terrestrial-ecotoxicity": "terrestrial-ecotoxicity",
    "stratospheric-ozone-depletion": "ozone-depletion",
    "fine-particulate-matter-formation": "pm-formation",
    "ozone-formation,-human-health": "photochemical-oxidant-formation-human-health",
    "ozone-formation,-terrestrial-ecosystems": "photochemical-oxidant-formation-ecosystems",
    "fossil-resource-scarcity": "fossil-fuel-depletion",
    "mineral-resource-scarcity": "other-non-renewable-material-depletion",  # approx
    "water-consumption": "scarce-blue-water-use",  # approx
}

# FIXME: provisional EUR2022 -> CHF conversion. Verify the rate and the currency
# year (factors are eur2022; the rest of TCAF mixes EUR2024/CHF) before release.
_EUR_TO_CHF = 0.94


def _load_lca_monetization_factors_chf(current_file_directory):
    """Return {recipe_category: CHF per impact-unit} from monetization-factor.csv.

    Uses the EUR factor set, converts to CHF with _EUR_TO_CHF, and keys the result
    by ReCiPe category via _LCA_MF_CROSSWALK. Impact categories not in the
    crosswalk are intentionally absent (they get dropped during monetization).
    """
    f = os.path.join(
        current_file_directory, "data/monetization-factors/monetization-factor.csv"
    )
    df = pd.read_csv(f)
    df = df[df["Key"] == "EUR"].copy()
    df["stem"] = df["Variable"].str.split("[").str[0]
    eur_by_stem = dict(zip(df["stem"], df["Value"]))

    mf_by_recipe = {}
    for recipe_cat, stem in _LCA_MF_CROSSWALK.items():
        if stem in eur_by_stem:
            mf_by_recipe[recipe_cat] = eur_by_stem[stem] * _EUR_TO_CHF
        else:
            print(
                f"  ⚠️ Monetization factor stem '{stem}' not found for "
                f"ReCiPe category '{recipe_cat}' - it will be dropped."
            )
    return mf_by_recipe


def _monetize_lca_dm(dm, mf_by_recipe):
    """In place: multiply physical ReCiPe impacts by their CHF factor.

    The impact-category axis is NOT at a fixed position across the two LCA matrices
    (and differs from their finished form), so it is detected by content: the
    categorical dimension whose labels are ReCiPe impact categories (i.e. overlap
    the crosswalk keys). Non-monetizable categories are dropped, the axis is
    sorted, and the aligned factor vector is broadcast along that exact axis.
    After this, 'lca-impacts' is a monetized value in CHF per kg of product.
    """
    # Identify which Categories* dimension holds the impact categories.
    cat_dims = [d for d in dm.col_labels if d.startswith("Categories")]
    impact_dim = None
    for d in cat_dims:
        labels = set(dm.col_labels[d])
        if labels & set(mf_by_recipe):  # this axis carries ReCiPe impacts
            impact_dim = d
            break
    if impact_dim is None:
        raise ValueError(
            "_monetize_lca_dm: no axis matched the ReCiPe impact "
            f"crosswalk. Axes seen: "
            f"{ {d: dm.col_labels[d] for d in cat_dims} }"
        )

    keep = [c for c in dm.col_labels[impact_dim] if c in mf_by_recipe]
    dropped = [c for c in dm.col_labels[impact_dim] if c not in mf_by_recipe]
    if dropped:
        print(f"  ℹ️ Dropping non-monetizable impact categories: {sorted(dropped)}")
    dm.filter({impact_dim: keep}, inplace=True)
    dm.sort(impact_dim)

    # Build the factor vector aligned to the (now sorted) impact axis and multiply
    # along that axis explicitly, wherever it sits in the array.
    mf_vec = np.array([mf_by_recipe[c] for c in dm.col_labels[impact_dim]], dtype=float)
    axis = list(dm.col_labels).index(impact_dim)  # position in the array
    shape = [1] * dm.array.ndim
    shape[axis] = mf_vec.size
    dm.array = dm.array * mf_vec.reshape(shape)  # broadcast on that axis
    dm.units["lca-impacts"] = "CHF/kg"
    return dm


# CalculationLeaf TCAF - LCA


def TCAF_lca_preprocessing():
    # Read data from TCAF Datapool
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(
        current_file_directory, "data/data_pool/lcia_animal_production_recipe.csv"
    )
    df_lcia_animal_production_recipe = pd.read_csv(f)
    f = os.path.join(
        current_file_directory, "data/data_pool/lcia_plant_production_recipe.csv"
    )
    df_lcia_plant_production_recipe = pd.read_csv(f)

    # For animal-production:
    # drop the rows that contain 'co-product', 'by-product', 'edible part' (to keep only the entire animal)
    exclude_keywords = ["co-product", "by-product", "edible part"]
    mask = df_lcia_animal_production_recipe["Process"].str.contains(
        "|".join(exclude_keywords), case=False, na=False
    )
    df_lcia_animal_production_recipe = df_lcia_animal_production_recipe[~mask].copy()
    # If process contains egg, Category => avian-egg
    df_lcia_animal_production_recipe.loc[
        df_lcia_animal_production_recipe["Process"].str.contains(
            "egg", case=False, na=False
        ),
        "Category",
    ] = "avian-egg"

    # Milk: the AGRIBALYSE 'dairy' file tags every row Category='dairy', mixing
    # cow/goat/sheep milk AND a cull-goat (meat). Re-assign by reference product so
    # only cow milk feeds abp-dairy-milk; goat/sheep milk are excluded (no matching
    # production category) and the cull goat is routed to caprine meat.
    proc_lower = df_lcia_animal_production_recipe["Process"].str.lower()
    df_lcia_animal_production_recipe.loc[
        proc_lower.str.contains("cow milk", na=False), "Category"
    ] = "dairy-milk"
    df_lcia_animal_production_recipe.loc[
        proc_lower.str.contains("goat milk", na=False), "Category"
    ] = "milk-goat"
    df_lcia_animal_production_recipe.loc[
        proc_lower.str.contains("sheep milk", na=False), "Category"
    ] = "milk-sheep"
    df_lcia_animal_production_recipe.loc[
        proc_lower.str.contains("cull goat", na=False), "Category"
    ] = "caprine"

    # Convert values to numeric
    df_lcia_animal_production_recipe["Value"] = pd.to_numeric(
        df_lcia_animal_production_recipe["Value"], errors="coerce"
    )
    df_lcia_plant_production_recipe["Value"] = pd.to_numeric(
        df_lcia_plant_production_recipe["Value"], errors="coerce"
    )

    # Aggregate per product category (ex wheat + oat => cereals)
    df_lcia_animal_production_recipe_agg = df_lcia_animal_production_recipe.groupby(
        ["Impact category", "Category", "Country", "Production Method"], as_index=False
    )["Value"].mean()
    df_lcia_plant_production_recipe_agg = df_lcia_plant_production_recipe.groupby(
        ["Impact category", "Category", "Country", "Production Method"], as_index=False
    )["Value"].mean()

    # Create variable name
    def clean_process(process):
        process = process.lower()
        process = re.sub(
            r"[^a-z0-9\-]", "-", process
        )  # replace non alphanumeric/dash with -
        process = re.sub(r"-+", "-", process)  # collapse multiple dashes
        process = process.strip("-")  # remove leading/trailing dashes
        return process

    df_lcia_animal_production_recipe_agg["variables"] = (
        "lca-impacts_"
        + df_lcia_animal_production_recipe_agg["Category"].apply(clean_process)
        + "_"
        + df_lcia_animal_production_recipe_agg["Production Method"].apply(clean_process)
        + "_"
        + df_lcia_animal_production_recipe_agg["Impact category"]
    )

    df_lcia_plant_production_recipe_agg["variables"] = (
        "lca-impacts_"
        + df_lcia_plant_production_recipe_agg["Category"].apply(clean_process)
        + "_"
        + df_lcia_plant_production_recipe_agg["Production Method"].apply(clean_process)
        + "_"
        + df_lcia_plant_production_recipe_agg["Impact category"]
    )

    # Filter columns
    cols_to_filter = ["variables", "Country", "Value"]
    df_lcia_animal_production_recipe_agg = df_lcia_animal_production_recipe_agg[
        cols_to_filter
    ]
    df_lcia_plant_production_recipe_agg = df_lcia_plant_production_recipe_agg[
        cols_to_filter
    ]

    # Append dfs
    df_lcia_recipe_all = pd.concat(
        [df_lcia_plant_production_recipe_agg, df_lcia_animal_production_recipe_agg],
        ignore_index=True,
    )

    # Add Years
    df_lcia_recipe_all["Years"] = "2023"

    # Step Datamatrix Formatting
    # Format as DMs for Switzerland (with production methods)
    lever = "dummy"
    df_lcia_recipe_all["lever"] = lever
    df_ots, df_fts = database_to_df_robust(df_lcia_recipe_all, lever, level="all")
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm_lcia_recipe_all_ch = DataMatrix.create_from_df(df_ots, num_cat=3)

    # Group production method 'intensive', 'conventional' and 'not-specified' in the same 'intensive' category
    dm_lcia_recipe_all_ch.groupby(
        {"intensive": "conventional|intensive|not-specified"},
        dim="Categories2",
        aggregation="mean",
        regex=True,
        inplace=True,
    )

    # Format as DMs for world (without production methods)
    lever = "dummy"
    df_lcia_recipe_all["lever"] = lever
    df_ots, df_fts = database_to_df_robust(df_lcia_recipe_all, lever, level="all")
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm_lcia_recipe_all_world = DataMatrix.create_from_df(df_ots, num_cat=3)

    # Group all production methods by mean and delete col
    dm_lcia_recipe_all_world.group_all(
        dim="Categories2", inplace=True, aggregation="mean"
    )

    # Step Rename Categories with rest of TCAF-Calc
    # Mapping from dm categories to target categories
    mapping_lca = {
        "crop-cereal": ["cereals"],
        "crop-fruit": ["fruits"],
        "crop-oilcrop": ["oilcrops"],
        "crop-pulse": ["legumes"],
        "crop-starch": ["starchyroots", "starchycrops"],
        "crop-sugarcrop": ["sugarcrops"],
        "crop-veg": ["vegetables"],
        "abp-hens-egg": ["avian-egg"],
        "abp-dairy-milk": ["dairy-milk"],
        "meat-bovine": ["bovine"],
        "meat-poultry": ["avian"],
        "meat-pig": ["porcine"],
        "meat-sheep": ["ovine", "caprine"],
        "meat-oth-animal": ["others"],
        "to-exclude": [
            "roughage",
            "intercrops",
            "nuts",
            "seafood",
            "fish",
            "fish-market",
            "fish-transformation",
            "milk-goat",
            "milk-sheep",
        ],
    }

    dm_lcia_recipe_all_ch.groupby(
        mapping_lca, dim="Categories1", aggregation="mean", inplace=True
    )
    dm_lcia_recipe_all_ch.drop("Categories1", "to-exclude")
    dm_lcia_recipe_all_world.groupby(
        mapping_lca, dim="Categories1", aggregation="mean", inplace=True
    )
    dm_lcia_recipe_all_world.drop("Categories1", "to-exclude")

    # Step Monetization: convert physical ReCiPe impacts [impact-unit/kg] to money
    # [CHF/kg] using the EUR monetization factors (EUR->CHF via _EUR_TO_CHF).
    # Non-monetizable impact categories (see _LCA_MF_CROSSWALK) are dropped here,
    # so 'lca-impacts' is already in CHF/kg by the time the module multiplies it by
    # production [kg] to get costs [CHF]. Applied before linear_fitting: the factor
    # is a constant scaling, so fitting monetized values is equivalent to fitting
    # physical values then monetizing.
    mf_by_recipe = _load_lca_monetization_factors_chf(current_file_directory)
    _monetize_lca_dm(dm_lcia_recipe_all_ch, mf_by_recipe)
    _monetize_lca_dm(dm_lcia_recipe_all_world, mf_by_recipe)

    # Debug
    dm = dm_lcia_recipe_all_ch
    arr, cl = dm.array, dm.col_labels
    m_axis = list(cl).index("Categories2")
    f_axis = list(cl).index("Categories1")
    imp_axis = list(cl).index("Categories3")
    for mi, m in enumerate(cl["Categories2"]):
        sl = [slice(None)] * arr.ndim
        sl[m_axis] = mi
        print(
            f"{m:12s} populated fraction = {1 - float(np.isnan(arr[tuple(sl)]).mean()):.3f}"
        )
    # per food: which methods are populated?
    print()
    for fi, food in enumerate(cl["Categories1"]):
        present = []
        for mi, m in enumerate(cl["Categories2"]):
            sl = [slice(None)] * arr.ndim
            sl[f_axis] = fi
            sl[m_axis] = mi
            if not np.isnan(arr[tuple(sl)]).all():
                present.append(m)
        print(f"{food:20s} has data for: {present}")

    # Step Linear fitting for all years
    linear_fitting(dm_lcia_recipe_all_ch, years_all)
    linear_fitting(dm_lcia_recipe_all_world, years_all)

    # Step Proxies for existing countries

    faostat_country_names = {
        # Africa
        "DZ": "Algeria",
        "AO": "Angola",
        "BJ": "Benin",
        "BW": "Botswana",
        "BF": "Burkina Faso",
        "BI": "Burundi",
        "CV": "Cabo Verde",
        "CM": "Cameroon",
        "CF": "Central African Republic",
        "TD": "Chad",
        "KM": "Comoros",
        "CG": "Congo",
        "CD": "Democratic Republic of the Congo",
        "CI": "Côte d'Ivoire",
        "DJ": "Djibouti",
        "EG": "Egypt",
        "GQ": "Equatorial Guinea",
        "ER": "Eritrea",
        "SZ": "Eswatini",
        "ET": "Ethiopia",
        "GA": "Gabon",
        "GM": "Gambia",
        "GH": "Ghana",
        "GN": "Guinea",
        "GW": "Guinea-Bissau",
        "KE": "Kenya",
        "LS": "Lesotho",
        "LR": "Liberia",
        "LY": "Libya",
        "MG": "Madagascar",
        "MW": "Malawi",
        "ML": "Mali",
        "MR": "Mauritania",
        "MU": "Mauritius",
        "MA": "Morocco",
        "MZ": "Mozambique",
        "NA": "Namibia",
        "NE": "Niger",
        "NG": "Nigeria",
        "RE": "Réunion",
        "RW": "Rwanda",
        "ST": "Sao Tome and Principe",
        "SN": "Senegal",
        "SC": "Seychelles",
        "SL": "Sierra Leone",
        "SO": "Somalia",
        "ZA": "South Africa",
        "SS": "South Sudan",
        "SD": "Sudan",
        "TZ": "United Republic of Tanzania",
        "TG": "Togo",
        "TN": "Tunisia",
        "UG": "Uganda",
        "ZM": "Zambia",
        "ZW": "Zimbabwe",
        # Americas
        "AR": "Argentina",
        "BS": "Bahamas",
        "BB": "Barbados",
        "BZ": "Belize",
        "BO": "Bolivia (Plurinational State of)",
        "BR": "Brazil",
        "CA": "Canada",
        "CL": "Chile",
        "CO": "Colombia",
        "CR": "Costa Rica",
        "CU": "Cuba",
        "DM": "Dominica",
        "DO": "Dominican Republic",
        "EC": "Ecuador",
        "SV": "El Salvador",
        "GD": "Grenada",
        "GT": "Guatemala",
        "GY": "Guyana",
        "HT": "Haiti",
        "HN": "Honduras",
        "JM": "Jamaica",
        "MX": "Mexico",
        "NI": "Nicaragua",
        "PA": "Panama",
        "PY": "Paraguay",
        "PE": "Peru",
        "KN": "Saint Kitts and Nevis",
        "LC": "Saint Lucia",
        "VC": "Saint Vincent and the Grenadines",
        "SR": "Suriname",
        "TT": "Trinidad and Tobago",
        "US": "United States of America",
        "UY": "Uruguay",
        "VE": "Venezuela (Bolivarian Republic of)",
        # Asia
        "AF": "Afghanistan",
        "AM": "Armenia",
        "AZ": "Azerbaijan",
        "BH": "Bahrain",
        "BD": "Bangladesh",
        "BT": "Bhutan",
        "BN": "Brunei Darussalam",
        "KH": "Cambodia",
        "CN": "China",
        "CY": "Cyprus",
        "GE": "Georgia",
        "IN": "India",
        "ID": "Indonesia",
        "IR": "Iran (Islamic Republic of)",
        "IQ": "Iraq",
        "IL": "Israel",
        "JP": "Japan",
        "JO": "Jordan",
        "KZ": "Kazakhstan",
        "KW": "Kuwait",
        "KG": "Kyrgyzstan",
        "LA": "Lao People's Democratic Republic",
        "LB": "Lebanon",
        "MY": "Malaysia",
        "MV": "Maldives",
        "MN": "Mongolia",
        "MM": "Myanmar",
        "NP": "Nepal",
        "KP": "Democratic People's Republic of Korea",
        "OM": "Oman",
        "PK": "Pakistan",
        "PH": "Philippines",
        "QA": "Qatar",
        "KR": "Republic of Korea",
        "SA": "Saudi Arabia",
        "SG": "Singapore",
        "LK": "Sri Lanka",
        "SY": "Syrian Arab Republic",
        "TJ": "Tajikistan",
        "TH": "Thailand",
        "TL": "Timor-Leste",
        "TR": "Türkiye",
        "TM": "Turkmenistan",
        "AE": "United Arab Emirates",
        "UZ": "Uzbekistan",
        "VN": "Viet Nam",
        "YE": "Yemen",
        # Europe
        "AL": "Albania",
        "AD": "Andorra",
        "AT": "Austria",
        "BY": "Belarus",
        "BE": "Belgium",
        "BA": "Bosnia and Herzegovina",
        "BG": "Bulgaria",
        "HR": "Croatia",
        "CZ": "Czechia",
        "DK": "Denmark",
        "EE": "Estonia",
        "FI": "Finland",
        "FR": "France",
        "DE": "Germany",
        "GR": "Greece",
        "HU": "Hungary",
        "IS": "Iceland",
        "IE": "Ireland",
        "IT": "Italy",
        "LV": "Latvia",
        "LI": "Liechtenstein",
        "LT": "Lithuania",
        "LU": "Luxembourg",
        "MT": "Malta",
        "MD": "Republic of Moldova",
        "MC": "Monaco",
        "ME": "Montenegro",
        "NL": "Netherlands",
        "MK": "North Macedonia",
        "NO": "Norway",
        "PL": "Poland",
        "PT": "Portugal",
        "RO": "Romania",
        "RU": "Russian Federation",
        "SM": "San Marino",
        "RS": "Serbia",
        "SK": "Slovakia",
        "SI": "Slovenia",
        "ES": "Spain",
        "SE": "Sweden",
        "CH": "Switzerland",
        "UA": "Ukraine",
        "GB": "United Kingdom",
        # Oceania
        "AU": "Australia",
        "FJ": "Fiji",
        "KI": "Kiribati",
        "MH": "Marshall Islands",
        "FM": "Micronesia (Federated States of)",
        "NR": "Nauru",
        "NZ": "New Zealand",
        "PW": "Palau",
        "PG": "Papua New Guinea",
        "WS": "Samoa",
        "SB": "Solomon Islands",
        "TO": "Tonga",
        "TV": "Tuvalu",
        "VU": "Vanuatu",
        # Non-standard ecoinvent region codes
        "GLO": "World",
        "RoW": "Rest of World",
        "RER": "Europe",
        "RNA": "North America",
        "RLA": "Latin America",
        "WI": "West Indies",
        "EU": "European Union",
    }

    # Specific country proxies
    proxy_map = {
        "CH": "FR",
        "AT": "DE",
        "BE": "FR",
        "NL": "DE",
        # European countries → RER
        # Non-European → GLO or RoW
    }

    target_countries = list(faostat_country_names.keys())
    add_and_fill_missing_countries_dm(
        dm_lcia_recipe_all_ch, target_countries, proxy_map
    )
    add_and_fill_missing_countries_dm(
        dm_lcia_recipe_all_world, target_countries, proxy_map
    )

    # Step Match countries names with Faostat

    def convert_country_codes(dm):
        """Rename country codes to FAOSTAT country names in a DataMatrix."""

        col_in = []
        col_out = []

        for code in dm.col_labels["Country"]:
            if code in faostat_country_names:
                col_in.append(code)
                col_out.append(faostat_country_names[code])
            else:
                print(f"  ⚠️ No FAOSTAT name found for: {code}")

        dm.rename_col(col_in, col_out, dim="Country")
        print(f"  ✓ Converted {len(col_in)} country codes to FAOSTAT names")

    # Usage
    convert_country_codes(dm_lcia_recipe_all_ch)
    convert_country_codes(dm_lcia_recipe_all_world)

    # Drop non-standard ecoinvent region codes
    regions_to_drop = [
        "World",
        "Rest of World",
        "Europe",
        "North America",
        "Latin America",
        "West Indies",
        "European Union",
    ]

    existing_to_drop = [
        r for r in regions_to_drop if r in dm_lcia_recipe_all_ch.col_labels["Country"]
    ]
    dm_lcia_recipe_all_ch.drop(dim="Country", col_label=existing_to_drop)

    existing_to_drop = [
        r
        for r in regions_to_drop
        if r in dm_lcia_recipe_all_world.col_labels["Country"]
    ]
    dm_lcia_recipe_all_world.drop(dim="Country", col_label=existing_to_drop)

    # Format as big DM
    DM_TCAF_lca = {
        "lca-switzerland": dm_lcia_recipe_all_ch.filter({"Country": ["Switzerland"]}),
        "lca-world": dm_lcia_recipe_all_world,
    }

    return DM_TCAF_lca


# CalculationLeaf CONSTANTS


def constant():
    # KCAL TO T ----------------------------------------------------------------------------------------

    # Read excel
    df_kcal_t = pd.read_excel(
        "../dietary-habits/data/dietary-habits_constants.xlsx", sheet_name="cp_kcal_t"
    )

    # Filter columns
    df_kcal_t = df_kcal_t[["variables", "kcal per t"]].copy()

    # Turn the df in a dict
    dict_kcal_t = dict(zip(df_kcal_t["variables"], df_kcal_t["kcal per t"]))
    categories1 = df_kcal_t["variables"].tolist()

    # Format as a cdm
    cdm_kcal = ConstantDataMatrix(
        col_labels={"Variables": ["cp_kcal-per-t"], "Categories1": categories1}
    )
    arr = np.zeros(
        (len(cdm_kcal.col_labels["Variables"]), len(cdm_kcal.col_labels["Categories1"]))
    )
    cdm_kcal.array = arr
    idx = cdm_kcal.idx
    for cat, val in dict_kcal_t.items():
        cdm_kcal.array[idx["cp_kcal-per-t"], idx[cat]] = val
    cdm_kcal.units["cp_kcal-per-t"] = "kcal/t"

    return cdm_kcal


# CalculationLeaf CREATE PICKLE
def database_from_csv_to_datamatrix(years_ots, years_fts):

    # Make list with years from 2020 to 2050 (steps of 5 years)
    years_all = years_ots + years_fts

    # FixedAssumptionsToDatamatrix -----------------------------------------------

    # Initialise
    dict_fxa = {}

    # Add in fxa
    dict_fxa["health-diet_paf"] = DM_TCAF_health_diet
    dict_fxa["health-diet_dalys"] = dm_health_dalys
    dict_fxa["biodiversity"] = DM_TCAF_biodiversity
    dict_fxa["lca"] = DM_TCAF_lca

    # CalibrationDataToDatamatrix ------------------------------------------------

    # LeversToDatamatrix OTS -----------------------------------------------------
    dict_ots = {}

    # LeversToDatamatrix FTS -----------------------------------------------------
    dict_fts = {}

    # FTS linear fitting of ots
    """DM_ots = DM_agriculture_old['ots'].copy()
  DM_fts = DM_agriculture_old['fts'].copy()

  # To do once when adding a new lever
  # DM_fts['climate-smart-crop']['processing-net-import'] = {'processing-net-import': dict()}

  # Levers to be normalised
  list_norm = ['climate-smart-livestock_ration']

  for key in DM_ots.keys():
    if isinstance(DM_ots[key], dict):
      for subkey in DM_ots[key].keys():
        dm = DM_ots[key][subkey].copy()
        linear_fitting(dm, years_fts)

        for lev in range(1, 5):  # 1 to 4
          if subkey in list_norm:  # ✅ check subkey, not key
            dm_norm = dm.copy()
            # Replace negative values with 0
            array_temp = dm_norm.array[:, :, :, :]
            array_temp[array_temp < 0] = 0.0
            dm_norm.array[:, :, :, :] = array_temp
            # Normalise
            dm_norm.normalise(dim='Categories1', inplace=True)
            DM_fts[key][subkey][lev] = dm_norm.filter(
              {'Years': years_fts}, inplace=False
            )
          else:
            DM_fts[key][subkey][lev] = dm.filter(
              {'Years': years_fts}, inplace=False
            )
    else:
      dm = DM_ots[key].copy()
      linear_fitting(dm, years_fts)
      for lev in range(1, 5):
        DM_fts[key][lev] = dm.filter({'Years': years_fts}, inplace=False)

  # file
  __file__ = "agriculture_landuse_preprocessing_EU.py"

  # directories
  current_file_directory = os.path.dirname(os.path.abspath(__file__))"""

    # ConstantsToDatamatrix ------------------------------------------------------
    dict_const = {}
    dict_const = {"monetization-factors": CDM_MF, "cdm_kcal": cdm_kcal}

    # Group all datamatrix in a single structure ---------------------------------
    DM_TCAF = {
        "fxa": dict_fxa,
        "constant": dict_const,
        "fts": dict_fts,
        "ots": dict_ots,
    }

    # Write datamatrix to pickle -------------------------------------------------
    f = "../../data/datamatrix/TCAF.pickle"
    with open(f, "wb") as handle:
        pickle.dump(DM_TCAF, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return


# CalculationTree RUNNING PREPROCESSING ----------------------------------------
years_ots = create_years_list(1990, 2023, 1)  # make list with years from 1990 to 2015
years_fts = create_years_list(2025, 2050, 5)
years_all = years_ots + years_fts
DM_TCAF_health_diet, dm_health_dalys = TCAF_health_diet_preprocessing()
DM_TCAF_biodiversity = TCAF_biodiversity_preprocessing()
DM_TCAF_lca = TCAF_lca_preprocessing()
CDM_MF = TCAF_MF_preprocessing()
cdm_kcal = constant()

# CalculationTree RUNNING PICKLE CREATION --------------------------------------
database_from_csv_to_datamatrix(years_ots, years_fts)
