import os
import pickle

import numpy as np

from tcaf_model.model.common.auxiliary_functions import (
    create_years_list,
    dm_match_countries,
    filter_country_and_load_data_from_pickles,
    linear_fitting,
)
from tcaf_model.model.common.config_loader import load_lever_config
from tcaf_model.model.common.interface_class import Interface


def init_years_lever():
    # function that can be used when running the module as standalone to initialise years and levers
    years_setting = [1990, 2023, 2025, 2050, 5]
    lever_setting = load_lever_config()
    return years_setting, lever_setting


# CalculationLeaf READ PICKLE
def read_data(DM_TCAF, lever_setting, years_all):

    # Read fts based on lever_setting
    # DM_ots_fts = read_level_data(DM_TCAF, lever_setting)

    # Sub-matrix for TCAF health-diet
    dm_tcaf_paf = DM_TCAF["fxa"]["health-diet_paf"]
    dm_tcaf_dalys = DM_TCAF["fxa"]["health-diet_dalys"]

    # Aggregate Data Matrix - DIETARY HABITS
    DM_TCAF_health_diet = {
        "health-diet_paf": dm_tcaf_paf,
        "health-diet_dalys": dm_tcaf_dalys,
    }

    # Aggregate Data Matrix - LCA
    DM_TCAF_lca = {
        "lca-switzerland": DM_TCAF["fxa"]["lca"]["lca-switzerland"],
        "lca-world": DM_TCAF["fxa"]["lca"]["lca-world"],
    }

    # Aggregate Data Matrix - BIODIVERSITY
    DM_TCAF_biodiversity = {
        "biodiversity-ch": DM_TCAF["fxa"]["biodiversity"]["TCAF-biodiversity-CH"],
        "biodiversity-world": DM_TCAF["fxa"]["biodiversity"]["TCAF-biodiversity-world"],
    }
    for key in DM_TCAF_biodiversity.keys():
        linear_fitting(DM_TCAF_biodiversity[key], years_all)
        DM_TCAF_biodiversity[key].filter({"Years": years_all}, inplace=True)

    # Constants
    # Monetization factors
    CDM_MF = {}
    # For health-diet
    cdm_temp = DM_TCAF["constant"]["monetization-factors"].filter_w_regex(
        {"Variables": "tcaf_mf_health-diet.*"}
    )
    CDM_MF["health-diet"] = cdm_temp

    # Other constants
    CDM_const = {}
    CDM_const["cdm_kcal"] = DM_TCAF["constant"]["cdm_kcal"].copy()

    return DM_TCAF_lca, DM_TCAF_health_diet, DM_TCAF_biodiversity, CDM_MF, CDM_const


# SimulateInteractions


def simulate_diet_to_TCAF_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(
        current_file_directory,
        "../_database/data/interface/dietary-habits_to_TCAF.pickle",
    )
    with open(f, "rb") as handle:
        DM_diet = pickle.load(handle)
    return DM_diet


def simulate_landuse_to_TCAF_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    f = os.path.normpath(
        os.path.join(
            current_file_directory,
            "../_database/data/interface/land-use_to_TCAF.pickle",
        )
    )
    with open(f, "rb") as handle:
        dm_cropland = pickle.load(handle)
    return dm_cropland


def simulate_crop_to_TCAF_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(
        current_file_directory, "../_database/data/interface/crop_to_TCAF.pickle"
    )
    with open(f, "rb") as handle:
        DM_crop_to_TCAF = pickle.load(handle)
    return DM_crop_to_TCAF


def simulate_livestock_to_TCAF_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(
        current_file_directory, "../_database/data/interface/livestock_to_TCAF.pickle"
    )
    with open(f, "rb") as handle:
        DM_livestock_to_TCAF = pickle.load(handle)
    return DM_livestock_to_TCAF


# CalculationLeaf TCAF LCA
def TCAF_lca_workflow(
    DM_TCAF_lca, DM_crop_to_TCAF, DM_landuse_to_TCAF, DM_livestock_to_TCAF, CDM_const
):
    # Match countries FIXME better match countries
    DM_livestock_to_TCAF["meat-world"].drop(
        col_label=["Switzerland", "Tokelau"], dim="Country"
    )
    DM_livestock_to_TCAF["asf-world"].drop(
        col_label=["Switzerland", "Tokelau"], dim="Country"
    )
    DM_crop_to_TCAF.drop(col_label=["Switzerland"], dim="Country")
    dm_match_countries(
        DM_crop_to_TCAF, DM_TCAF_lca["lca-world"], parameter="perfect match"
    )
    dm_match_countries(
        DM_livestock_to_TCAF["asf-world"],
        DM_TCAF_lca["lca-world"],
        parameter="perfect match",
    )
    dm_match_countries(
        DM_livestock_to_TCAF["meat-world"],
        DM_TCAF_lca["lca-world"],
        parameter="perfect match",
    )

    # (Switzerland & World) Crop - Unit convertion: [kcal] to [kg]
    cdm_kcal = CDM_const["cdm_kcal"].copy()
    # cdm_kcal.rename_col_regex(str1="crop-", str2="", dim="Categories1")
    cat = DM_crop_to_TCAF.col_labels["Categories1"]
    cdm_kcal = cdm_kcal.filter({"Categories1": cat})
    # Sort
    DM_crop_to_TCAF.sort("Categories1")
    cdm_kcal.sort("Categories1")
    # Convert from [kcal] to [kg]
    array_temp = (
        10**3
        * DM_crop_to_TCAF[:, :, "agr_domestic-production_afw", :]
        / cdm_kcal[np.newaxis, np.newaxis, "cp_kcal-per-t", :]
    )
    DM_crop_to_TCAF.add(
        array_temp,
        dim="Variables",
        col_label="agr_domestic-production_afw_kg",
        unit="kg",
    )
    DM_landuse_to_TCAF["prod-ch"].add(
        0.0, dummy=True, col_label="rice", dim="Categories2", unit="kcal"
    )
    array_temp = (
        10**3
        * DM_landuse_to_TCAF["prod-ch"][:, :, "agr_domestic-production_afw", :, :]
        / cdm_kcal[np.newaxis, np.newaxis, "cp_kcal-per-t", np.newaxis, :]
    )
    DM_landuse_to_TCAF["prod-ch"].add(
        array_temp,
        dim="Variables",
        col_label="agr_domestic-production_afw_kg",
        unit="kg",
    )

    # FIXME cereals = cereals + rice
    DM_landuse_to_TCAF["prod-ch"].groupby(
        {"crop-cereal": "crop-cereal|crop-rice"},
        dim="Categories2",
        inplace=True,
        regex=True,
    )
    DM_crop_to_TCAF.groupby(
        {"crop-cereal": "crop-cereal|crop-rice"},
        dim="Categories1",
        inplace=True,
        regex=True,
    )

    # FIXME drop extensive for crops CH for now
    DM_landuse_to_TCAF["prod-ch"].drop(dim="Categories1", col_label="extensive")

    # Rename variables to agr_production-tcaf
    DM_landuse_to_TCAF["prod-ch"].rename_col(
        "agr_domestic-production_afw_kg", "agr_production-lca", dim="Variables"
    )
    DM_crop_to_TCAF.rename_col(
        "agr_domestic-production_afw_kg", "agr_production-lca", dim="Variables"
    )
    DM_livestock_to_TCAF["meat-ch"].rename_col(
        "agr_domestic_production_liv_afw_kg", "agr_production-lca", dim="Variables"
    )
    DM_livestock_to_TCAF["meat-world"].rename_col(
        "agr_domestic_production_liv_afw_kg", "agr_production-lca", dim="Variables"
    )
    DM_livestock_to_TCAF["asf-ch"].rename_col(
        "agr_domestic_production_liv_afw_kg", "agr_production-lca", dim="Variables"
    )
    DM_livestock_to_TCAF["asf-world"].rename_col(
        "agr_domestic_production_liv_afw_kg", "agr_production-lca", dim="Variables"
    )

    # Step World
    # Append together:
    # - kg crop produced
    # - kg meat liveweight produced
    # - kg asf produced
    dm_lca_world = DM_livestock_to_TCAF["meat-world"].filter(
        {"Variables": ["agr_production-lca"]}
    )
    dm_lca_world.append(
        DM_livestock_to_TCAF["asf-world"].filter({"Variables": ["agr_production-lca"]}),
        dim="Categories1",
    )
    dm_lca_world.append(
        DM_crop_to_TCAF.filter({"Variables": ["agr_production-lca"]}), dim="Categories1"
    )

    # Multiply with Monetized LCA impacts fixme sure already monetized?
    food_cat = dm_lca_world.col_labels["Categories1"]
    DM_TCAF_lca["lca-world"].filter({"Categories1": food_cat}, inplace=True)
    array_temp = (
        dm_lca_world[:, :, "agr_production-lca", :, np.newaxis]
        * DM_TCAF_lca["lca-world"][:, :, "lca-impacts", :, :]
    )
    DM_TCAF_lca["lca-world"].add(
        array_temp, dim="Variables", col_label="agr_production-tcaf", unit="CHF"
    )

    # Multiply with Monetization Factors (MF)

    # Step Switzerland
    # Append together:
    # - kg crop produced
    # - kg meat liveweight produced
    # - kg asf produced
    dm_lca_ch = DM_livestock_to_TCAF["meat-ch"].filter(
        {"Variables": ["agr_production-lca"]}
    )
    dm_lca_ch.append(
        DM_livestock_to_TCAF["asf-ch"].filter({"Variables": ["agr_production-lca"]}),
        dim="Categories1",
    )
    DM_landuse_to_TCAF["prod-ch"].switch_categories_order(
        cat1="Categories2", cat2="Categories1"
    )
    dm_lca_ch.append(
        DM_landuse_to_TCAF["prod-ch"].filter({"Variables": ["agr_production-lca"]}),
        dim="Categories1",
    )

    # Multiply with LCA impacts

    # Multiply with Monetization Factors (MF)

    return DM_TCAF_lca


# CalculationLeaf TCAF HEALTH DIET
def TCAF_health_diet_workflow(DM_diet, DM_TCAF_health_diet, CDM_MF):
    """
    Diet-attributable / avoidable DALYs, following the stratified-adherence logic
    of the R projection script (Projection.R).

    For every food risk-factor r and disease d:
      PAF_r(B) : PAF read off the dose-response curve at the BAU (reference) intake
      PAF_r(T) : PAF read off the curve at the full target intake (full adherent)
      PIF_r    : max((PAF_r(B) - PAF_r(T)) / (1 - PAF_r(T)), 0)      [full adoption]
    PAF is read by linear interpolation with flat extrapolation
    (numpy.interp == R's approx(..., rule = 2)).

    Combine the food risk-factors of a disease multiplicatively (GBD standard):
      AF_ref = 1 - prod_r (1 - PAF_r(B))
      PIF    = 1 - prod_r (1 - PIF_r)

    Population adherence alpha (share of the population adopting the target diet)
    enters as a LINEAR scaling of the disease-level impact fraction - the
    two-strata mixture used by R, PIF(alpha) = alpha * PIF:
      attributable = AF_ref        * DALYs
      avoided      = alpha * PIF    * DALYs
      residual     = attributable - avoided

    Inputs from the dietary-habits interface:
      B     = diet-consumed_bau     (unweighted full BAU diet,     g/cap/day)
      T     = diet-consumed_target  (unweighted full target diet,  g/cap/day)
      alpha = diet-adherence        (share_diet_adherence,         -)

    Monetization:
      MF    = CDM_MF['health-diet'] (value per DALY, CHF/DALY)  ->  cost = DALYs * MF

    FLAG - where alpha is applied: here it scales the COMBINED (disease-level) PIF,
    which is the standard "a fraction alpha of the population fully complies"
    interpretation and reproduces R's par_total at every alpha. If the R script
    instead scales each food's PIF *before* the multiplicative combination, use the
    line marked "ALT" below instead (the two differ because the combination is
    non-linear).
    """
    dm_B = DM_diet["diet-consumed_bau"].copy()  # reference (BAU) diet
    dm_T = DM_diet["diet-consumed_target"].copy()  # full target diet
    dm_alpha = DM_diet["diet-adherence"].copy()  # population adherence share

    # Pre-processing
    dm_data_paf = DM_TCAF_health_diet[
        "health-diet_paf"
    ]  # dict: food -> PAF dose-response curve
    dm_data_dalys = DM_TCAF_health_diet[
        "health-diet_dalys"
    ]  # DM: Country x Years x [dalys] x disease

    # Step 0 - Groupby categories relevant for health ----------------------------
    # Red meat = bovine + pig + sheep + other animal
    pattern = "pro-liv-meat-bovine|pro-liv-meat-pig|pro-liv-meat-sheep|pro-liv-meat-oth-animal"
    dm_B.groupby(
        {"pro-liv-meat-red": pattern}, dim="Categories1", inplace=True, regex=True
    )
    dm_T.groupby(
        {"pro-liv-meat-red": pattern}, dim="Categories1", inplace=True, regex=True
    )

    # Health categories (food risk factors) that we consider
    cat_health = [
        "crop-fruit",
        "crop-pulse",
        "pro-liv-abp-dairy-milk",
        "crop-oilcrop",
        "pro-liv-meat-processed",
        "pro-liv-meat-red",
        "crop-veg",
        "crop-cereal-whole",
    ]

    dm_B.filter({"Categories1": cat_health}, inplace=True)
    dm_T.filter({"Categories1": cat_health}, inplace=True)

    for cat in cat_health:
        if cat not in dm_B.col_labels["Categories1"]:
            print(f"Warning: {cat} not in diet-consumed_bau")
        if cat not in dm_data_paf:
            print(f"Warning: {cat} not in dm_data_paf")

    # Step 1 - Common years across intakes, adherence and (projected) DALYs ------
    years = sorted(
        set(dm_B.col_labels["Years"])
        & set(dm_data_dalys.col_labels["Years"])
        & set(dm_alpha.col_labels["Years"])
    )
    dm_B.filter({"Years": years}, inplace=True)
    dm_T.filter({"Years": years}, inplace=True)
    dm_alpha.filter({"Years": years}, inplace=True)

    country = dm_B.col_labels["Country"]
    dalys_disease = dm_data_dalys.col_labels["Categories1"]  # disease code order
    n_c, n_y, n_f, n_d = len(country), len(years), len(cat_health), len(dalys_disease)
    paf_var_prefix = "tcaf_health-diet_paf_"

    # Step 2 - Read the PAF off the dose-response curve at each intake -----------
    # numpy.interp does linear interpolation with flat extrapolation outside the
    # grid, reproducing R's approx(x, y, xout, rule = 2).
    def eval_paf(dm_intake):
        out = np.zeros((n_c, n_y, n_f, n_d))
        for fi, cat in enumerate(cat_health):
            dm_curve = dm_data_paf[cat]
            x_grid = np.array(
                dm_curve.col_labels["Years"], dtype=float
            )  # intake grid [g/day/cap]
            var_names = [paf_var_prefix + d for d in dalys_disease]
            y_grid = np.stack(
                [dm_curve[:, :, v][0, :] for v in var_names], axis=-1
            )  # (n_intake, n_disease)
            order = np.argsort(x_grid)
            x_s, y_s = x_grid[order], y_grid[order, :]
            xv = dm_intake[:, :, "lfs_consumers-diet", cat]  # (n_c, n_y)
            for di in range(n_d):
                out[:, :, fi, di] = np.interp(xv, x_s, y_s[:, di])
        return out

    arr_paf_B = eval_paf(dm_B)  # PAF at reference intake
    arr_paf_T = eval_paf(dm_T)  # PAF at full target intake

    # Step 3 - Full-adoption PIF per food x disease ------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        arr_pif = (arr_paf_B - arr_paf_T) / (1.0 - arr_paf_T)
    arr_pif = np.maximum(np.nan_to_num(arr_pif, nan=0.0), 0.0)

    # Step 4 - Combine the food risk-factors of a disease: 1 - prod(1 - p) -------
    af_ref_comb = 1.0 - np.prod(1.0 - arr_paf_B, axis=2)  # (n_c, n_y, n_d)
    pif_comb = 1.0 - np.prod(1.0 - arr_pif, axis=2)  # (n_c, n_y, n_d)  full adoption

    # Step 5 - Apply population adherence: alpha * PIF ---------------------------
    alpha = dm_alpha[:, :, "share_diet_adherence"]  # (n_c, n_y)
    pif_alpha = alpha[:, :, np.newaxis] * pif_comb  # alpha on the COMBINED PIF
    # ALT (scale each food's PIF before combining):
    # pif_alpha = 1.0 - np.prod(1.0 - alpha[:, :, np.newaxis, np.newaxis] * arr_pif, axis=2)

    # Step 6 - Attributable / avoided / residual DALYs ---------------------------
    dm_dalys = dm_data_dalys.copy()
    dm_dalys.filter({"Years": years}, inplace=True)
    arr_dalys = dm_dalys[:, :, "tcaf_health-diet_dalys", :]  # (n_c, n_y, n_d)

    arr_attr = af_ref_comb * arr_dalys
    arr_avoid = pif_alpha * arr_dalys
    arr_resid = arr_attr - arr_avoid

    # Detailed DM (per disease): attributable / avoided / residual
    dm_paf = dm_dalys.copy()
    dm_paf[:, :, "tcaf_health-diet_dalys", :] = arr_attr
    dm_paf.add(
        arr_avoid,
        dim="Variables",
        col_label="tcaf_health-diet_dalys-avoided",
        unit="DALYs/y",
    )
    dm_paf.add(
        arr_resid,
        dim="Variables",
        col_label="tcaf_health-diet_dalys-residual",
        unit="DALYs/y",
    )

    # Step 7 - Total across diseases = sum_d ------------------------------------
    dm_dalys_tot = dm_paf.copy()
    dm_dalys_tot.groupby({"total": ".*"}, dim="Categories1", inplace=True, regex=True)

    # Step 8 - Monetization: cost [CHF] = DALYs [DALYs/y] * MF [CHF/DALY] --------
    # CDM_MF['health-diet'] holds a single health-diet monetization factor
    # (monetary value per DALY). Each DALYs component (attributable / avoided /
    # residual) is monetized, both per disease and for the total.
    cdm_mf = CDM_MF["health-diet"]
    mf_var = cdm_mf.col_labels["Variables"][0]
    mf = cdm_mf[mf_var]  # scalar CHF/DALY
    cost_of = {
        "tcaf_health-diet_dalys": "tcaf_health-diet_cost-bau",
        "tcaf_health-diet_dalys-avoided": "tcaf_health-diet_cost-avoided",
        "tcaf_health-diet_dalys-residual": "tcaf_health-diet_cost-residual",
    }
    for dm in (dm_paf, dm_dalys_tot):
        for dalys_var, cost_var in cost_of.items():
            dm.add(
                dm[:, :, dalys_var, :] * mf,
                dim="Variables",
                col_label=cost_var,
                unit="CHF",
            )

    return dm_paf, dm_dalys_tot


# CalculationLeaf TCAF BIODIVERSITY


def TCAF_biodiversity_workflow(DM_TCAF_biodiversity, DM_landuse_to_TCAF):
    DM_TCAF_biodiversity = DM_TCAF_biodiversity.copy()

    # Step Biodiversity Switzerland
    # Drop treenut cat because not in cropland
    DM_TCAF_biodiversity["biodiversity-ch"].drop(dim="Categories2", col_label="treenut")
    # Add mean value for starch missing in biodiv
    dm_temp = DM_TCAF_biodiversity["biodiversity-ch"].groupby(
        {"starch": ".*"},
        dim="Categories2",
        aggregation="mean",
        regex=True,
        inplace=False,
    )
    DM_TCAF_biodiversity["biodiversity-ch"].append(dm_temp, dim="Categories2")
    # Append cropland to biodiversity for relevant geoscale
    DM_TCAF_biodiversity["biodiversity-ch"].append(
        DM_landuse_to_TCAF["cropland-ch"], dim="Variables"
    )

    # Biodiversity costs [CHF/ha] = cropland [ha] * eco-costs [CHF/ha]
    DM_TCAF_biodiversity["biodiversity-ch"].operation(
        "agr_cropland",
        "*",
        "eco-cost",
        dim="Variables",
        out_col="tcaf_biodiversity",
        unit="CHF",
    )

    # Step Biodiversity World
    # Drop Switzerland and differing countries if any
    DM_landuse_to_TCAF["cropland-world"].drop(dim="Country", col_label="Switzerland")
    set_countries = set(
        DM_TCAF_biodiversity["biodiversity-world"].col_labels["Country"]
    ) - set(DM_landuse_to_TCAF["cropland-world"].col_labels["Country"])
    DM_TCAF_biodiversity["biodiversity-world"].drop(
        dim="Country", col_label=list(set_countries)
    )
    # Sort countries
    DM_TCAF_biodiversity["biodiversity-world"].sort(dim="Country")
    DM_landuse_to_TCAF["cropland-world"].sort(dim="Country")

    # Sum total cropland
    DM_landuse_to_TCAF["cropland-world"].groupby(
        {"total": ".*"}, dim="Categories1", aggregation="sum", regex=True, inplace=True
    )
    DM_landuse_to_TCAF["cropland-world"] = DM_landuse_to_TCAF[
        "cropland-world"
    ].flatten()
    DM_landuse_to_TCAF["cropland-world"].rename_col_regex(
        str1="agr_cropland_total_total", str2="agr_cropland", dim="Variables"
    )

    # Append cropland to biodiversity for relevant geoscale
    DM_TCAF_biodiversity["biodiversity-world"].append(
        DM_landuse_to_TCAF["cropland-world"], dim="Variables"
    )

    # Biodiversity costs [EUR2024/ha] = cropland [ha] * eco-costs [EUR2024/ha]
    DM_TCAF_biodiversity["biodiversity-world"].operation(
        "agr_cropland",
        "*",
        "eco-cost",
        dim="Variables",
        out_col="tcaf_biodiversity",
        unit="EUR2024",
    )

    return DM_TCAF_biodiversity


# CalculationLeaf TPE INTERFACE
def TCAF_TPE_interface(dm_health_diet_detailed, dm_health_diet_tot):
    # attributable / avoided / residual DALYs, and their monetized costs [CHF]
    vars_out = [
        "tcaf_health-diet_dalys",
        "tcaf_health-diet_dalys-avoided",
        "tcaf_health-diet_dalys-residual",
        "tcaf_health-diet_cost",
        "tcaf_health-diet_cost-avoided",
        "tcaf_health-diet_cost-residual",
    ]

    # health-diet detailed (per disease)
    dm_health_diet_detailed.filter({"Variables": vars_out}, inplace=True)
    dm_tpe = dm_health_diet_detailed.flattest()

    # health-diet total (summed over diseases)
    dm_health_diet_tot.filter({"Variables": vars_out}, inplace=True)
    dm_tpe.append(dm_health_diet_tot.flattest(), dim="Variables")

    return dm_tpe


def TCAF(lever_setting, years_setting, DM_input, interface=Interface()):

    years_ots = create_years_list(
        years_setting[0], years_setting[1], 1
    )  # make list with years from 1990 to 2015
    years_fts = create_years_list(years_setting[2], years_setting[3], years_setting[4])
    years_all = years_ots + years_fts

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    DM_TCAF_lca, DM_TCAF_health_diet, DM_TCAF_biodiversity, CDM_MF, CDM_const = (
        read_data(DM_input, lever_setting, years_all)
    )
    country_list = ["Switzerland"]

    # INTERFACES IN ---------------------------------------------------------------------------------------------------

    # CalculationLeaf Link interface or Simulate data from other modules
    # dietary-habits
    if interface.has_link(from_sector="dietary-habits", to_sector="TCAF"):
        DM_diet = interface.get_link(from_sector="dietary-habits", to_sector="TCAF")
    else:
        if len(interface.list_link()) != 0:
            print("You are missing dietary-habits to TCAF interface")
        DM_diet = simulate_diet_to_TCAF_input()
        for key in DM_diet.keys():
            DM_diet[key].filter({"Country": country_list}, inplace=True)

    # land-use
    if interface.has_link(from_sector="land-use", to_sector="TCAF"):
        DM_landuse_to_TCAF = interface.get_link(
            from_sector="land-use", to_sector="TCAF"
        )
    else:
        if len(interface.list_link()) != 0:
            print("You are missing land-use to TCAF interface")
        DM_landuse_to_TCAF = simulate_landuse_to_TCAF_input()

    # crop
    if interface.has_link(from_sector="crop", to_sector="TCAF"):
        DM_crop_to_TCAF = interface.get_link(from_sector="crop", to_sector="TCAF")
    else:
        if len(interface.list_link()) != 0:
            print("You are missing crop to TCAF interface")
        DM_crop_to_TCAF = simulate_crop_to_TCAF_input()

    # livestock
    if interface.has_link(from_sector="livestock", to_sector="TCAF"):
        DM_livestock_to_TCAF = interface.get_link(
            from_sector="livestock", to_sector="TCAF"
        )
    else:
        if len(interface.list_link()) != 0:
            print("You are missing livestock to TCAF interface")
        DM_livestock_to_TCAF = simulate_livestock_to_TCAF_input()

    # CalculationTree ---------------------------------------------------------------------------------------------------
    # DM_TCAF_lca = TCAF_lca_workflow(DM_TCAF_lca, DM_crop_to_TCAF, DM_landuse_to_TCAF, DM_livestock_to_TCAF, CDM_const)
    dm_health_diet_detailed, dm_health_diet_tot = TCAF_health_diet_workflow(
        DM_diet, DM_TCAF_health_diet, CDM_MF
    )
    DM_TCAF_biodiversity = TCAF_biodiversity_workflow(
        DM_TCAF_biodiversity, DM_landuse_to_TCAF
    )
    # CalculationTree TPE OUTPUT -------------------------------------------------------------------------------------------------------
    results_run = TCAF_TPE_interface(dm_health_diet_detailed, dm_health_diet_tot)

    # INTERFACES OUT ---------------------------------------------------------------------------------------------------

    # interface to Land use
    # DM_lus = agriculture_landuse_interface(DM_bioenergy, dm_lgn, dm_land_use)
    # interface.add_link(from_sector='agriculture', to_sector='land-use',
    #                   dm=DM_lus)

    return results_run


def TCAF_module_local_run():
    country_list = ["Switzerland"]
    DM_input = filter_country_and_load_data_from_pickles(
        country_list=country_list, modules_list="TCAF", filter_country=False
    )
    years_setting, lever_setting = init_years_lever()
    TCAF(lever_setting, years_setting, DM_input["TCAF"])
    return


if __name__ == "__main__":
    TCAF_module_local_run()
