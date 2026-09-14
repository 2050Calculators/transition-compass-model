import os
import pickle

import numpy as np
import pandas as pd

from transition_compass_model.model.agriculture import workflows as wkf
from transition_compass_model.model.common.auxiliary_functions import (
    filter_country_and_load_data_from_pickles,
    read_level_data,
)
from transition_compass_model.model.common.config_loader import load_lever_config
from transition_compass_model.model.common.data_matrix_class import DataMatrix
from transition_compass_model.model.common.interface_class import Interface


def init_years_lever():
    # function that can be used when running the module as standalone to initialise years and levers
    years_setting = [1990, 2023, 2025, 2050, 5]
    lever_setting = load_lever_config()
    return years_setting, lever_setting


#######################################################################################################
######################################### LOAD AGRICULTURE DATA #########################################
#######################################################################################################


def simulate_industry_to_agriculture_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(
        current_file_directory,
        "../_database/data/interface/industry_to_agriculture.pickle",
    )
    with open(f, "rb") as handle:
        DM_ind = pickle.load(handle)
    return DM_ind


def simulate_transport_to_agriculture_input():
    # Read input from lifestyle : food waste & diet
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(
        current_file_directory,
        "../_database/data/interface/transport_to_agriculture.pickle",
    )
    with open(f, "rb") as handle:
        dm_tra = pickle.load(handle)
    return dm_tra


# CalculationLeaf READ PICKLE
def read_data(DM_agriculture, lever_setting):
    # Read fts based on lever_setting
    # FIXME error it adds ots and fts
    # DM_check = check_ots_fts_match(DM_agriculture, lever_setting)
    DM_ots_fts = read_level_data(DM_agriculture, lever_setting)

    # FXA data matrix
    dm_fxa_cal_diet = DM_agriculture["fxa"]["cal_agr_diet"]
    dm_fxa_cal_liv_prod = DM_agriculture["fxa"]["cal_agr_domestic-production-liv"]
    dm_fxa_cal_liv_pop = DM_agriculture["fxa"]["cal_agr_liv-population"]
    dm_fxa_cal_liv_CH4 = DM_agriculture["fxa"]["cal_agr_liv_CH4-emission"]
    dm_fxa_cal_liv_N2O = DM_agriculture["fxa"]["cal_agr_liv_N2O-emission"]
    dm_fxa_cal_demand_feed = DM_agriculture["fxa"]["cal_agr_demand_feed"]
    # dm_fxa_cal_land = DM_agriculture['fxa']['cal_agr_lus_land']
    dm_fxa_ef_liv_N2O = DM_agriculture["fxa"]["ef_liv_N2O-emission"]
    dm_fxa_ef_liv_CH4_treated = DM_agriculture["fxa"]["ef_liv_CH4-emission_treated"]
    dm_fxa_liv_nstock = DM_agriculture["fxa"]["liv_manure_n-stock"]

    # Extract sub-data-matrices according to the flow
    # Sub-matrix for LIFESTYLE
    # dm_demography = DM_ots_fts['pop']['lfs_demography_']
    dm_diet_requirement = DM_ots_fts["kcal-req"]
    dm_diet_split = DM_ots_fts["diet"]["lfs_consumers-diet"]
    dm_diet_share = DM_ots_fts["diet"]["share"]
    dm_diet_fwaste = DM_ots_fts["fwaste"]
    # dm_population = DM_ots_fts['pop']['lfs_population_']

    # Sub-matrix for the FOOD DEMAND
    dm_food_net_import_pro = DM_ots_fts["food-net-import"].filter_w_regex(
        {"Categories1": "pro-.*", "Variables": "agr_food-net-import"}
    )

    # Sub-matrix for LIVESTOCK
    dm_livestock_losses = DM_ots_fts["climate-smart-livestock"][
        "climate-smart-livestock_losses"
    ]
    dm_livestock_yield = DM_ots_fts["climate-smart-livestock"][
        "climate-smart-livestock_yield"
    ]
    dm_livestock_slaughtered = DM_ots_fts["climate-smart-livestock"][
        "climate-smart-livestock_slaughtered"
    ]
    dm_livestock_density = DM_ots_fts["climate-smart-livestock"][
        "climate-smart-livestock_density"
    ]
    dm_fxa_ratio_milk = DM_agriculture["fxa"]["ratio_milk"]

    # Sub-matrix for ALCOHOLIC BEVERAGES
    dm_alc_bev = DM_ots_fts["biomass-hierarchy"]["biomass-hierarchy-bev-ibp-use-oth"]

    # Sub-matrix for BIOENERGY
    dm_bioenergy_cap_load_factor = DM_ots_fts["bioenergy-capacity"][
        "bioenergy-capacity_load-factor"
    ]
    dm_bioenergy_cap_bgs_mix = DM_ots_fts["bioenergy-capacity"][
        "bioenergy-capacity_bgs-mix"
    ]
    dm_bioenergy_cap_efficiency = DM_ots_fts["bioenergy-capacity"][
        "bioenergy-capacity_efficiency"
    ]
    dm_bioenergy_cap_liq = DM_ots_fts["bioenergy-capacity"]["bioenergy-capacity_liq_b"]
    dm_bioenergy_cap_elec = DM_ots_fts["bioenergy-capacity"]["bioenergy-capacity_elec"]
    dm_bioenergy_mix_digestor = DM_ots_fts["biomass-hierarchy"][
        "biomass-hierarchy_biomass-mix_digestor"
    ]
    dm_bioenergy_mix_solid = DM_ots_fts["biomass-hierarchy"][
        "biomass-hierarchy_biomass-mix_solid"
    ]
    dm_bioenergy_mix_liquid = DM_ots_fts["biomass-hierarchy"][
        "biomass-hierarchy_biomass-mix_liquid"
    ]
    dm_bioenergy_liquid_biodiesel = DM_ots_fts["biomass-hierarchy"][
        "biomass-hierarchy_bioenergy_liquid_biodiesel"
    ]
    dm_bioenergy_liquid_biogasoline = DM_ots_fts["biomass-hierarchy"][
        "biomass-hierarchy_bioenergy_liquid_biogasoline"
    ]
    dm_bioenergy_liquid_biojetkerosene = DM_ots_fts["biomass-hierarchy"][
        "biomass-hierarchy_bioenergy_liquid_biojetkerosene"
    ]
    dm_bioenergy_cap_elec.append(dm_bioenergy_cap_load_factor, dim="Variables")
    dm_bioenergy_cap_elec.append(dm_bioenergy_cap_efficiency, dim="Variables")

    # Sub-matrix for LIVESTOCK MANURE MANGEMENT & GHG EMISSIONS
    dm_livestock_enteric_emissions = DM_ots_fts["climate-smart-livestock"][
        "climate-smart-livestock_enteric"
    ]
    dm_livestock_manure = DM_ots_fts["climate-smart-livestock"][
        "climate-smart-livestock_manure"
    ]

    # Sub-matrix for FEED
    dm_ration = DM_ots_fts["climate-smart-livestock"]["climate-smart-livestock_ration"]
    dm_alt_protein = DM_ots_fts["alt-protein"]
    dm_ruminant_feed = DM_ots_fts["ruminant-feed"]

    # Sub-matrix for CROP
    dm_food_net_import_crop = DM_ots_fts[
        "food-net-import"
    ].filter_w_regex(
        {"Categories1": "crop-.*", "Variables": "agr_food-net-import"}
    )  # filtered here on purpose and not in the pickle (other parts of the datamatrix are used)
    dm_food_net_import_crop.rename_col_regex(str1="crop-", str2="", dim="Categories1")
    dm_crop = DM_ots_fts["climate-smart-crop"]["climate-smart-crop_losses"]
    # dm_food_net_import_crop.drop(dim='Categories1', col_label=['stm'])
    dm_crop.append(dm_food_net_import_crop, dim="Variables")
    dm_residues_yield = DM_agriculture["fxa"]["residues_yield"]
    dm_hierarchy_residues_cereals = DM_ots_fts["biomass-hierarchy"][
        "biomass-hierarchy_crop_cereal"
    ]
    dm_cal_crop = DM_agriculture["fxa"]["cal_agr_domestic-production_food"]
    dm_cal_crop_bev = DM_agriculture["fxa"]["cal_agr_domestic-production_bev"]
    # dm_crop.append(dm_cal_crop, dim='Variables')
    dm_ef_residues = DM_agriculture["fxa"]["ef_burnt-residues"]
    dm_ssr_feed_crop = DM_ots_fts["climate-smart-crop"]["feed-net-import"]
    dm_processing_yield = DM_agriculture["fxa"]["processing-yield"]

    # Sub-matrix for LAND
    dm_cal_land = DM_agriculture["fxa"]["cal_agr_lus_land"]
    dm_yield = DM_ots_fts["climate-smart-crop"]["climate-smart-crop_yield"]
    dm_fibers = DM_agriculture["fxa"]["fibers"]
    dm_rice = DM_agriculture["fxa"]["rice"]
    dm_cal_cropland = DM_agriculture["fxa"]["cal_agr_lus_land_cropland"]

    # Sub-matrix for NITROGEN BALANCE
    dm_input = DM_ots_fts["climate-smart-crop"]["climate-smart-crop_input-use"]
    dm_fertilizer_emission = DM_agriculture["fxa"]["agr_emission_fertilizer"]
    dm_cal_n = DM_agriculture["fxa"]["cal_agr_crop_emission_N2O-emission_fertilizer"]
    # dm_fertilizer_emission.append(dm_cal_n, dim='Variables')

    # Sub-matrix for ENERGY & GHG EMISSIONS
    dm_cal_energy_demand = DM_agriculture["fxa"]["cal_agr_energy-demand"]
    dm_energy_demand = DM_ots_fts["climate-smart-crop"][
        "climate-smart-crop_energy-demand"
    ]
    dm_cal_GHG = DM_agriculture["fxa"]["cal_agr_emissions"]
    dm_cal_GHG.deepen()
    dm_cal_input = DM_agriculture["fxa"]["cal_agr_input-use_emissions-CO2"]

    # Aggregated Data Matrix - ENERGY & GHG EMISSIONS
    DM_energy_ghg = {
        "energy_demand": dm_energy_demand,
        "cal_energy_demand": dm_cal_energy_demand,
        "cal_input": dm_cal_input,
        "cal_GHG": dm_cal_GHG,
    }

    # Aggregate Data Matrix - LIFESTYLE
    DM_lifestyle = {
        "energy-requirement": dm_diet_requirement,
        "diet-split": dm_diet_split,
        "diet-share": dm_diet_share,
        "diet-fwaste": dm_diet_fwaste,
        # 'demography': dm_demography,
        # 'population': dm_population,
        "cal_diet": dm_fxa_cal_diet,
    }

    # Aggregated Data Matrix - FOOD DEMAND
    DM_food_demand = {"food-net-import-pro": dm_food_net_import_pro}

    # Aggregated Data Matrix - LIVESTOCK
    DM_livestock = {
        "losses": dm_livestock_losses,
        "yield": dm_livestock_yield,
        "liv_slaughtered_rate": dm_livestock_slaughtered,
        "cal_liv_prod": dm_fxa_cal_liv_prod,
        "cal_liv_population": dm_fxa_cal_liv_pop,
        "ruminant_density": dm_livestock_density,
        "ratio_milk": dm_fxa_ratio_milk,
    }

    # Aggregated Data Matrix - ALCOHOLIC BEVERAGES
    DM_alc_bev = {
        "biomass_hierarchy": dm_alc_bev,
        "processing-yields": dm_processing_yield,
    }

    # Aggregated Data Matrix - BIOENERGY
    DM_bioenergy = {
        "electricity_production": dm_bioenergy_cap_elec,
        "bgs-mix": dm_bioenergy_cap_bgs_mix,
        "liq": dm_bioenergy_cap_liq,
        "digestor-mix": dm_bioenergy_mix_digestor,
        "solid-mix": dm_bioenergy_mix_solid,
        "liquid-mix": dm_bioenergy_mix_liquid,
        "liquid-biodiesel": dm_bioenergy_liquid_biodiesel,
        "liquid-biogasoline": dm_bioenergy_liquid_biogasoline,
        "liquid-biojetkerosene": dm_bioenergy_liquid_biojetkerosene,
    }

    # Aggregated Data Matrix - LIVESTOCK MANURE MANAGEMENT & GHG EMISSIONS
    DM_manure = {
        "enteric_emission": dm_livestock_enteric_emissions,
        "manure": dm_livestock_manure,
        "cal_liv_CH4": dm_fxa_cal_liv_CH4,
        "cal_liv_N2O": dm_fxa_cal_liv_N2O,
        "ef_liv_N2O": dm_fxa_ef_liv_N2O,
        "ef_liv_CH4_treated": dm_fxa_ef_liv_CH4_treated,
        "liv_n-stock": dm_fxa_liv_nstock,
    }

    # Aggregated Data Matrix - FEED
    DM_feed = {
        "ration": dm_ration,
        "alt-protein": dm_alt_protein,
        "cal_agr_demand_feed": dm_fxa_cal_demand_feed,
        "ruminant-feed": dm_ruminant_feed,
    }

    # Aggregated Data Matrix - CROP
    DM_crop = {
        "crop": dm_crop,
        "cal_crop": dm_cal_crop,
        "cal_bev": dm_cal_crop_bev,
        "ef_residues": dm_ef_residues,
        "residues_yield": dm_residues_yield,
        "hierarchy_residues_cereals": dm_hierarchy_residues_cereals,
        "food-net-import-pro": dm_food_net_import_pro,
        "feed-net-import_crop": dm_ssr_feed_crop,
        "processing-yields": dm_processing_yield,
    }

    # Aggregated Data Matrix - LAND
    DM_land = {
        "cal_land": dm_cal_land,
        "cal_cropland": dm_cal_cropland,
        "yield": dm_yield,
        "fibers": dm_fibers,
        "rice": dm_rice,
    }

    # Aggregated Data Matrix - NITROGEN BALANCE
    DM_nitrogen = {
        "input": dm_input,
        "emissions": dm_fertilizer_emission,
        "cal_n": dm_cal_n,
    }

    CDM_const = DM_agriculture["constant"]

    return (
        DM_ots_fts,
        DM_lifestyle,
        DM_food_demand,
        DM_livestock,
        DM_alc_bev,
        DM_bioenergy,
        DM_manure,
        DM_feed,
        DM_crop,
        DM_land,
        DM_nitrogen,
        DM_energy_ghg,
        CDM_const,
    )


# SimulateInteractions
def simulate_lifestyles_to_agriculture_input_new():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(
        current_file_directory,
        "../_database/data/interface/lifestyles_to_agriculture.pickle",
    )
    with open(f, "rb") as handle:
        DM_lfs = pickle.load(handle)

    return DM_lfs


def simulate_lifestyles_to_agriculture_input():
    # Read input from lifestyle : food waste & diet
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(
        current_file_directory,
        "../_database/data/xls/All-Countries-interface_from-lifestyles-to-agriculture_EUCALC.xlsx",
    )
    df = pd.read_excel(f, sheet_name="default")
    df_population = df.copy()
    df = df.drop(columns=["lfs_population_total[inhabitants]"])
    dm_lfs = DataMatrix.create_from_df(df, num_cat=1)

    # Read input from lifestyle : population
    df_population = df_population[
        ["Years", "Country", "lfs_population_total[inhabitants]"]
    ]  # keep only population
    dm_population = DataMatrix.create_from_df(df_population, num_cat=0)

    # other way to do the step before but does not add it to the dm
    # idx = dm_lfs.idx
    # overall_food_demand = dm_lfs.array[:,:,idx['lfs_diet'],:] + dm_lfs.array[:,:,idx['lfs_food-wastes'],:]

    # Renaming to correct format to match iterators (simultaneously changes in lfs_diet, lfs_fwaste and agr_demand)
    # Adding meat prefix
    pro_liv_meat = ["bov", "sheep", "pigs", "poultry", "oth-animals"]
    for cat in pro_liv_meat:
        new_cat = "pro-liv-meat-" + cat
        # Dropping the -s at the end of pigs (for name matching reasons)
        if new_cat.endswith("pigs"):
            new_cat = new_cat[:-1]
        # Replacing bov by bovine (for name matching reasons)
        if new_cat == "pro-liv-meat-bov":
            new_cat = "pro-liv-meat-bovine"
        dm_lfs.rename_col(cat, new_cat, dim="Categories1")

    # Adding bev prefix
    pro_bev = ["beer", "bev-fer", "bev-alc", "wine"]
    for cat in pro_bev:
        new_cat = "pro-bev-" + cat
        dm_lfs.rename_col(cat, new_cat, dim="Categories1")

    # Adding milk prefix
    pro_milk = ["milk"]
    for cat in pro_milk:
        new_cat = "pro-liv-abp-dairy-" + cat
        dm_lfs.rename_col(cat, new_cat, dim="Categories1")

    # Adding egg prefix
    pro_egg = ["egg"]
    for cat in pro_egg:
        new_cat = "pro-liv-abp-hens-" + cat
        dm_lfs.rename_col(cat, new_cat, dim="Categories1")

    # Adding crop prefix
    crop = ["cereals", "oilcrops", "pulses", "starch", "fruits", "veg"]
    for cat in crop:
        new_cat = "crop-" + cat
        # Dropping the -s at the end of cereals, oilcrops, pulses, fruits (for name matching reasons)
        if new_cat.endswith("s"):
            new_cat = new_cat[:-1]
        dm_lfs.rename_col(cat, new_cat, dim="Categories1")

    # Adding abp-processed prefix
    processed = ["afat", "offal"]
    for cat in processed:
        new_cat = "pro-liv-abp-processed-" + cat
        # Dropping the -s at the end afats (for name matching reasons)
        if new_cat.endswith("s"):
            new_cat = new_cat[:-1]
        dm_lfs.rename_col(cat, new_cat, dim="Categories1")

    # Adding crop processed prefix
    processed = ["voil", "sweet", "sugar"]
    for cat in processed:
        new_cat = "pro-crop-processed-" + cat
        dm_lfs.rename_col(cat, new_cat, dim="Categories1")

    dm_lfs.sort("Categories1")

    return dm_population


def simulate_buildings_to_agriculture_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(
        current_file_directory,
        "../_database/data/interface/buildings_to_agriculture.pickle",
    )
    with open(f, "rb") as handle:
        dm_bld = pickle.load(handle)

    return dm_bld


def agriculture_landuse_interface(
    DM_bioenergy, dm_lgn, dm_land_use, write_xls=False, write_pickle=False
):
    dm_wood = DM_bioenergy["solid-mix"].filter(
        {
            "Variables": ["agr_bioenergy_biomass-demand_solid"],
            "Categories1": ["fuelwood-and-res"],
        }
    )
    dm_lgn = dm_lgn.filter(
        {
            "Variables": ["agr_bioenergy_biomass-demand_liquid_lgn"],
            "Categories1": ["lgn-btl-fuelwood-and-res"],
        }
    )
    dm_land_use = dm_land_use.filter({"Variables": ["agr_lus_land"]})

    DM_lus = {"wood": dm_wood, "lgn": dm_lgn, "landuse": dm_land_use}

    # if write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../_database/data/interface/agriculture_to_land-use.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(DM_lus, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # dm_dh
    if write_xls is True:
        dm_lus = (
            DM_bioenergy["solid-mix"]
            .filter(
                {
                    "Variables": ["agr_bioenergy_biomass-demand_solid"],
                    "Categories1": ["fuelwood-and-res"],
                }
            )
            .flatten()
        )
        dm_lus.append(
            dm_lgn.filter(
                {
                    "Variables": ["agr_bioenergy_biomass-demand_liquid_lgn"],
                    "Categories1": ["lgn-btl-fuelwood-and-res"],
                }
            ).flatten(),
            "Variables",
        )
        dm_lus.append(
            dm_land_use.filter({"Variables": ["agr_lus_land"]}).flatten(), "Variables"
        )

        """current_file_directory = os.path.dirname(os.path.abspath(__file__))
        df_lus = dm_lus.write_df()
        df_lus.to_excel(
            current_file_directory + "/../_database/data/xls/" + 'All-Countries_interface_from-agriculture-to-landuse.xlsx',
            index=False)"""

    return DM_lus


def agriculture_emissions_interface(
    DM_nitrogen,
    dm_CO2,
    DM_crop,
    DM_manure,
    DM_land,
    dm_input_use_CO2,
    dm_crop_residues,
    dm_CH4,
    dm_N2O_liv,
    dm_CH4_rice,
    dm_fertilizer_N2O,
    write_xls=False,
    write_pickle=False,
):
    def agg_zeroes_for_missing_gases(dm):
        gases = ["CH4", "CO2", "N2O"]
        gases_current = dm.col_labels["Categories1"]
        gases_missing = np.array(gases)[
            [gas not in gases_current for gas in gases]
        ].tolist()
        for gas in gases_missing:
            dm.add(0, "Categories1", gas, dummy=True)
        dm.sort("Categories1")

        return

    # input use
    dm_ems = dm_input_use_CO2.groupby(
        {
            "input-use_CO2": [
                "agr_input-use_emissions-CO2_fuel",
                "agr_input-use_emissions-CO2_liming",
                "agr_input-use_emissions-CO2_urea",
            ]
        },
        "Variables",
    )
    dm_ems.deepen()
    agg_zeroes_for_missing_gases(dm_ems)

    # liv
    dm_temp = (
        dm_CH4.filter({"Variables": ["agr_liv_CH4-emission"]})
        .group_all("Categories2", inplace=False)
        .group_all("Categories1", inplace=False)
    )
    dm_temp.rename_col("agr_liv_CH4-emission", "liv_CH4", "Variables")
    dm_temp2 = (
        dm_N2O_liv.filter({"Variables": ["agr_liv_N2O-emission"]})
        .group_all("Categories2", inplace=False)
        .group_all("Categories1", inplace=False)
    )
    dm_temp2.rename_col("agr_liv_N2O-emission", "liv_N2O", "Variables")
    dm_temp.append(dm_temp2, "Variables")
    dm_temp.deepen()
    agg_zeroes_for_missing_gases(dm_temp)
    dm_temp.change_unit("liv", factor=1e-6, old_unit="t", new_unit="Mt")
    dm_ems.append(dm_temp, "Variables")

    # crop residues
    dm_temp = dm_crop_residues.groupby(
        {
            "crop-resid_N2O": [
                "agr_emissions-N2O_crop_burnt-residues",
                "agr_emissions-N2O_crop_soil-residues",
            ]
        },
        "Variables",
    )
    dm_temp.rename_col(
        "agr_emissions-CH4_crop_burnt-residues", "crop-resid_CH4", "Variables"
    )
    dm_temp.deepen()
    agg_zeroes_for_missing_gases(dm_temp)
    dm_ems.append(dm_temp, "Variables")

    # ch4 rice
    dm_temp = dm_CH4_rice.copy()
    dm_temp.rename_col("agr_emissions-CH4_crop_rice", "rice_CH4", "Variables")
    dm_temp.deepen()
    agg_zeroes_for_missing_gases(dm_temp)
    dm_ems.append(dm_temp, "Variables")

    # fertlizer N2O
    dm_temp = dm_fertilizer_N2O.copy()
    dm_temp.rename_col(
        "agr_crop_emission_N2O-emission_fertilizer", "fertilizer_N2O", "Variables"
    )
    dm_temp.deepen()
    agg_zeroes_for_missing_gases(dm_temp)
    dm_ems.append(dm_temp, "Variables")

    # aggregate
    dm_ems.groupby(
        {"agriculture": ["input-use", "liv", "crop-resid", "rice", "fertilizer"]},
        "Variables",
        inplace=True,
    )

    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../_database/data/interface/agriculture_to_emissions.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(dm_ems, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # Append everything
    # dm_ems = dm_fertilizer_N2O.copy()
    # dm_ems.append(dm_input_use_CO2, dim="Variables")
    # dm_ems.append(dm_crop_residues, dim="Variables")
    # dm_ems.append(
    #     dm_CH4.filter({"Variables": ["agr_liv_CH4-emission"]}).flatten().flatten(),
    #     dim="Variables",
    # )
    # dm_ems.append(
    #     dm_N2O_liv.filter({"Variables": ["agr_liv_N2O-emission"]}).flatten().flatten(),
    #     dim="Variables",
    # )
    # dm_ems.append(dm_CH4_rice, dim="Variables")

    # import pprint
    # dm_ems.sort("Variables")
    # pprint.pprint(dm_ems.col_labels["Variables"])

    # write
    """if write_xls is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        dm_ems = dm_ems.write_df()
        dm_ems.to_excel(
            current_file_directory + "/../_database/data/xls/" + 'All-Countries_interface_from-agriculture-to-climate.xlsx',
            index=False)"""

    return dm_ems


def agriculture_ammonia_interface(dm_mineral_fertilizer, write_pickle=False):
    # Demand for Mineral fertilizers
    # TODO (pre-processing fix needed): FTS linear extrapolation drives per-ha application
    # rates below zero for some nutrients (potash from ~2030, phosphate from ~2040, urea
    # from ~2050 in the CH pickle). This makes agr_product-demand negative in those years.
    # Fix: clip the FTS climate-smart-crop_input-use values to >= 0 in the agriculture
    # pre-processing before building the pickle.
    dm_ammonia = dm_mineral_fertilizer.filter({"Variables": ["agr_input-use"]})
    dm_ammonia.rename_col("agr_input-use", "agr_product-demand", dim="Variables")
    dm_ammonia.rename_col("mineral", "fertilizer", dim="Categories1")

    if write_pickle == True:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(
            this_dir, "../_database/data/interface/agriculture_to_ammonia.pickle"
        )
        with open(file, "wb") as handle:
            pickle.dump(dm_ammonia, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm_ammonia


def agriculture_storage_interface(DM_energy_ghg, write_xls=False):
    # TODO: storage is not done for the moment, we'll add this when storage will be done
    # FIXME: Energy demand filter change to other unit ([TWh] instead of [ktoe])
    dm_storage = DM_energy_ghg["caf_energy_demand"].filter_w_regex(
        {"Variables": "agr_energy-demand", "Categories1": ".*ff.*"}
    )

    # Summing in the same category
    dm_storage.groupby(
        {"gas-ff-natural": "gas-ff-natural|liquid-ff-lpg"},
        dim="Categories1",
        regex=True,
        inplace=True,
    )

    # Renaming
    dm_storage.rename_col("liquid-ff-fuel-oil", "liquid-ff-oil", dim="Categories1")

    # Flatten
    dm_storage = dm_storage.flatten()

    # write
    """if write_xls is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        dm_storage = dm_storage.write_df()
        dm_storage.to_excel(
            current_file_directory + "/../_database/data/xls/" + 'All-Countries_interface_from-agriculture-to-storage.xlsx',
            index=False)"""

    return dm_storage


def agriculture_energy_interface(DM_energy_ghg, write_pickle=False):
    # NOTE: agr_climate-smart-crop_energy-demand electricity intensity is set to 0 in
    # preprocessing — farm electricity is not yet modeled. agr_energy-consumption will
    # therefore be 0 until the preprocessing is updated.
    dm_pow = DM_energy_ghg["energy_demand"].filter_w_regex(
        {"Variables": "^agr_energy-demand$", "Categories1": ".*electricity.*"}
    )
    # change_unit updates both array and units dict (avoids stale ktoe entry after rename)
    dm_pow.change_unit(
        "agr_energy-demand", old_unit="ktoe", new_unit="TWh", factor=0.0116222
    )
    dm_pow.rename_col("agr_energy-demand", "agr_energy-consumption", "Variables")
    elec_cats = dm_pow.col_labels.get("Categories1", [])
    if len(elec_cats) != 1 or elec_cats[0] != "electricity":
        dm_pow.group_all("Categories1", inplace=True)
        dm_pow.rename_col(
            dm_pow.col_labels["Categories1"][0], "electricity", "Categories1"
        )
    dm_pow.add(0, dim="Categories1", col_label="district-heating", dummy=True)
    DM_energy = {"power": dm_pow}

    # if write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../_database/data/interface/agriculture_to_energy.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(DM_energy, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return {"power": dm_pow}


def agriculture_minerals_interface(DM_nitrogen, DM_bioenergy, dm_lgn, write_xls=False):
    # Demand for phosphate & potash
    dm_minerals = DM_nitrogen["input"].filter(
        {"Variables": ["agr_input-use"], "Categories1": ["phosphate", "potash"]}
    )
    dm_minerals.change_unit("agr_input-use", 1e-6, old_unit="t", new_unit="Mt")
    dm_minerals.rename_col("agr_input-use", "agr_demand", "Variables")
    dm_minerals = dm_minerals.flatten()

    # Demand for fuelwood (solid)
    dm_solid = DM_bioenergy["solid-mix"].filter(
        {
            "Variables": ["agr_bioenergy_biomass-demand_solid"],
            "Categories1": ["fuelwood-and-res"],
        }
    )
    dm_solid.change_unit(
        "agr_bioenergy_biomass-demand_solid", 0.1264, old_unit="TWh", new_unit="Mt"
    )
    dm_solid = dm_solid.flatten()

    # Demand for fuelwood (liquid)
    dm_liquid = dm_lgn.filter(
        {
            "Variables": ["agr_bioenergy_biomass-demand_liquid_lgn"],
            "Categories1": ["lgn-btl-fuelwood-and-res"],
        }
    )
    dm_liquid.rename_col(
        "lgn-btl-fuelwood-and-res", "btl_fuelwood-and-res", dim="Categories1"
    )
    dm_liquid.rename_col(
        "agr_bioenergy_biomass-demand_liquid_lgn",
        "agr_bioenergy_biomass-demand_liquid",
        dim="Variables",
    )
    dm_liquid.change_unit(
        "agr_bioenergy_biomass-demand_liquid",
        factor=0.00000000000116222,
        old_unit="kcal",
        new_unit="TWh",
    )
    dm_liquid.change_unit(
        "agr_bioenergy_biomass-demand_liquid",
        factor=0.1264,
        old_unit="TWh",
        new_unit="Mt",
    )
    dm_liquid = dm_liquid.filter({"Variables": ["agr_bioenergy_biomass-demand_liquid"]})
    dm_liquid = dm_liquid.flatten()

    # Appending everything together
    dm_minerals.append(dm_solid, dim="Variables")
    dm_minerals.append(dm_liquid, dim="Variables")

    # writing dm minerals
    """if write_xls is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        dm_minerals = dm_minerals.write_df()
        dm_minerals.to_excel(
            current_file_directory + "/../_database/data/xls/" + 'All-Countries_interface_from-agriculture-to-minerals.xlsx',
            index=False)"""

    return dm_minerals


def agriculture_refinery_interface(DM_energy_ghg):
    dm_ref = DM_energy_ghg["energy_demand"].filter_w_regex(
        {"Variables": "agr_energy-demand", "Categories1": ".*ff.*"}
    )

    # Summing in the same category
    dm_ref.groupby(
        {"gas-ff-natural": "gas-ff-natural|liquid-ff-lpg"},
        dim="Categories1",
        regex=True,
        inplace=True,
    )

    # Renaming
    dm_ref.rename_col("liquid-ff-fuel-oil", "liquid-ff-oil", dim="Categories1")

    # order
    dm_ref.sort("Categories1")

    # change unit
    ktoe_to_twh = 0.0116222  # from KNIME factor
    dm_ref.change_unit(
        "agr_energy-demand", ktoe_to_twh, old_unit="ktoe", new_unit="TWh"
    )

    return dm_ref


def agriculture_TPE_interface(
    CDM_const,
    DM_livestock,
    DM_crop,
    dm_crop_other,
    DM_feed,
    dm_aps,
    dm_input_use_CO2,
    dm_crop_residues,
    dm_CH4,
    dm_liv_N2O,
    dm_CH4_rice,
    dm_fertilizer_N2O,
    DM_energy_ghg,
    DM_bioenergy,
    dm_lgn,
    dm_eth,
    dm_oil,
    dm_aps_ibp,
    DM_food_demand,
    dm_lfs_pro,
    dm_lfs,
    DM_land,
    dm_fiber,
    dm_aps_ibp_oil,
    dm_voil_tpe,
    DM_alc_bev,
    dm_biofuel_fdk,
    dm_liv_pop,
    DM_ssr,
    dm_fertilizer_co,
    DM_manure,
    dm_cropland,
):
    kcal_to_TWh = 1.163e-12

    # LIVESTOCK POPULATION, SLAUGTHERED & MANURE ------------------------------------------------------
    # Livestock population
    # Note : check if it includes the poultry for eggs
    dm_liv_meat = dm_liv_pop.filter_w_regex(
        {"Variables": "agr_liv_population", "Categories1": "meat.*"}, inplace=False
    )
    dm_tpe = dm_liv_meat.flattest()

    # Livestock slaughtered
    dm_slaughtered = DM_livestock["liv_slaughtered_rate"].filter(
        {"Variables": ["agr_liv_population_slau"]}
    )
    dm_tpe.append(dm_slaughtered.flattest(), dim="Variables")

    # Manure
    dm_manure = DM_manure["liv_n-stock"].filter({"Variables": ["agr_liv_n-stock"]})
    dm_tpe.append(dm_manure.flattest(), dim="Variables")

    # FOOD SUPPLY ------------------------------------------------------

    # Filter
    dm_supply = dm_lfs.filter({"Variables": ["agr_demand"]})
    cdm_kcal = CDM_const["cdm_kcal-per-t"].copy()
    cdm_kcal.drop(dim="Categories1", col_label="crop-sugarcrop")
    cdm_kcal.drop(dim="Categories1", col_label="stm")
    cdm_kcal.drop(dim="Categories1", col_label="pro-crop-processed-molasse")
    cdm_kcal.drop(dim="Categories1", col_label="pro-crop-processed-cake")
    cdm_kcal.drop(dim="Categories1", col_label="liv-meat-meal")

    # Sort
    dm_supply.sort("Categories1")
    cdm_kcal.sort("Categories1")

    # Convert from [kcal] to [t]
    array_temp = (
        dm_supply[:, :, "agr_demand", :]
        / cdm_kcal[np.newaxis, np.newaxis, "cp_kcal-per-t", :]
    )
    dm_supply.add(array_temp, dim="Variables", col_label="agr_demand_tpe", unit="t")
    dm_supply = dm_supply.filter({"Variables": ["agr_demand_tpe", "agr_demand"]})

    # Append for TPE
    dm_tpe.append(dm_supply.flattest(), dim="Variables")

    # FOOD WASTE------------------------------------------------------

    # Filter
    dm_foodwaste = dm_lfs.filter({"Variables": ["lfs_food-wastes"]})
    cdm_kcal = CDM_const["cdm_kcal-per-t"].copy()
    cdm_kcal.drop(dim="Categories1", col_label="crop-sugarcrop")
    cdm_kcal.drop(dim="Categories1", col_label="stm")
    cdm_kcal.drop(dim="Categories1", col_label="pro-crop-processed-molasse")
    cdm_kcal.drop(dim="Categories1", col_label="pro-crop-processed-cake")
    cdm_kcal.drop(dim="Categories1", col_label="liv-meat-meal")

    # Sort
    dm_foodwaste.sort("Categories1")
    cdm_kcal.sort("Categories1")

    # Convert from [kcal] to [t]
    array_temp = (
        dm_foodwaste[:, :, "lfs_food-wastes", :]
        / cdm_kcal[np.newaxis, np.newaxis, "cp_kcal-per-t", :]
    )
    dm_foodwaste.add(
        array_temp, dim="Variables", col_label="lfs_food-wastes_tpe", unit="t"
    )
    dm_foodwaste = dm_foodwaste.filter({"Variables": ["lfs_food-wastes_tpe"]})

    # Append for TPE
    dm_tpe.append(dm_foodwaste.flattest(), dim="Variables")

    # DOMESTIC PRODUCTION ------------------------------------------------------

    # Meat - Domestic production
    dm_meat = DM_livestock["yield"].filter(
        {"Variables": ["agr_domestic_production_liv_afw"]}
    )
    dm_meat.rename_col(
        "agr_domestic_production_liv_afw",
        "agr_domestic-production_afw",
        dim="Variables",
    )

    # Crop - Domestic production
    dm_crop_prod_food = DM_crop["crop"].filter(
        {"Variables": ["agr_domestic-production_afw"]}
    )

    # Append Meat & Crop together
    dm_crop_prod_food.append(dm_meat, dim="Categories1")

    # Filter constants and rename
    cdm_kcal = CDM_const["cdm_kcal-per-t"].copy()
    cdm_kcal = CDM_const["cdm_kcal-per-t"].filter(
        {
            "Categories1": [
                "pro-liv-meat-bovine",
                "pro-liv-meat-pig",
                "pro-liv-meat-poultry",
                "pro-liv-meat-sheep",
                "pro-liv-meat-oth-animals",
                "pro-liv-abp-dairy-milk",
                "pro-liv-abp-hens-egg",
                "crop-cereal",
                "crop-fruit",
                "crop-oilcrop",
                "crop-pulse",
                "crop-rice",
                "crop-starch",
                "crop-sugarcrop",
                "crop-veg",
            ]
        }
    )
    cdm_kcal.rename_col_regex(str1="crop-", str2="", dim="Categories1")
    cdm_kcal.rename_col_regex(str1="pro-liv-", str2="", dim="Categories1")

    # Sort
    dm_crop_prod_food.sort("Categories1")
    cdm_kcal.sort("Categories1")

    # Convert from [kcal] to [t]
    idx_dm = dm_crop_prod_food.idx
    idx_cdm = cdm_kcal.idx
    array_temp = (
        dm_crop_prod_food.array[:, :, idx_dm["agr_domestic-production_afw"], :]
        / cdm_kcal.array[idx_cdm["cp_kcal-per-t"], :]
    )
    dm_crop_prod_food.add(
        array_temp,
        dim="Variables",
        col_label="agr_domestic-production_afw_tpe",
        unit="t",
    )
    dm_crop_prod_food = dm_crop_prod_food.filter(
        {
            "Variables": [
                "agr_domestic-production_afw_tpe",
                "agr_domestic-production_afw",
            ]
        }
    )

    # Append for TPE
    dm_tpe.append(dm_crop_prod_food.flattest(), dim="Variables")

    # LIVESTOCK FEED ------------------------------------------------------

    # Livestock feed
    dm_feed = DM_feed["ration"].filter({"Variables": ["agr_demand_feed"]})
    dm_aps.rename_col("agr_feed_aps", "agr_demand_feed_aps", dim="Variables")
    dm_tpe.append(dm_feed.flattest(), dim="Variables")
    dm_tpe.append(dm_aps.flattest(), dim="Variables")

    # GHG EMISSIONS ------------------------------------------------------

    # CO2 emissions
    dm_tpe.append(dm_input_use_CO2.flattest(), dim="Variables")

    # CH4 emissions
    dm_tpe.append(dm_CH4.flattest(), dim="Variables")
    dm_tpe.append(dm_crop_residues.flattest(), dim="Variables")
    dm_tpe.append(dm_CH4_rice.flattest(), dim="Variables")

    # N2O emissions Note : residues already accounted for in df_residues in CH4 emissions
    dm_tpe.drop(col_label="cal_rate", dim="Variables")
    dm_liv_N2O.drop(dim="Variables", col_label="cal_rate")
    dm_tpe.append(dm_liv_N2O.flattest(), dim="Variables")
    dm_tpe.append(dm_fertilizer_N2O.flattest(), dim="Variables")

    # ENERGY ------------------------------------------------------

    # Energy use per type
    dm_energy_demand = DM_energy_ghg["energy_demand"].filter(
        {"Variables": ["agr_energy-demand"]}
    )
    # Unit conversion [ktoe] => [TWh]
    dm_energy_demand.change_unit(
        "agr_energy-demand", factor=0.0116222, old_unit="ktoe", new_unit="TWh"
    )
    dm_tpe.append(dm_energy_demand.flattest(), dim="Variables")

    # Bioenergy capacity
    dm_bio_cap_biogas = DM_bioenergy["bgs-mix"].filter(
        {"Variables": ["agr_bioenergy-capacity_bgs-tec"]}
    )
    dm_bio_cap_biodiesel = DM_bioenergy["liquid-biodiesel"].filter(
        {"Variables": ["agr_bioenergy-capacity_liq-bio-prod_biodiesel"]}
    )
    dm_bio_cap_biogasoline = DM_bioenergy["liquid-biogasoline"].filter(
        {"Variables": ["agr_bioenergy-capacity_liq-bio-prod_biogasoline"]}
    )
    dm_bio_cap_biojetkerosene = DM_bioenergy["liquid-biojetkerosene"].filter(
        {"Variables": ["agr_bioenergy-capacity_liq-bio-prod_biojetkerosene"]}
    )
    dm_tpe.append(dm_bio_cap_biogas.flattest(), dim="Variables")
    dm_tpe.append(dm_bio_cap_biodiesel.flattest(), dim="Variables")
    dm_tpe.append(dm_bio_cap_biojetkerosene.flattest(), dim="Variables")
    dm_tpe.append(dm_bio_cap_biogasoline.flattest(), dim="Variables")

    # Bioenergy feedstock mix (reunion of others fdk)

    # Liquid bioenergy-feedstock mix
    dm_fdk_oil = dm_oil.filter(
        {"Variables": ["agr_bioenergy_biomass-demand_liquid_oil"]}
    )
    dm_fdk_oil.rename_col(
        "agr_bioenergy_biomass-demand_liquid_oil",
        "agr_bioenergy_biomass-demand_liquid",
        dim="Variables",
    )
    dm_fdk_eth = dm_eth.filter(
        {"Variables": ["agr_bioenergy_biomass-demand_liquid_eth"]}
    )
    dm_fdk_eth.rename_col(
        "agr_bioenergy_biomass-demand_liquid_eth",
        "agr_bioenergy_biomass-demand_liquid",
        dim="Variables",
    )
    dm_fdk_lgn = dm_lgn.filter(
        {"Variables": ["agr_bioenergy_biomass-demand_liquid_lgn"]}
    )
    dm_fdk_lgn.rename_col(
        "agr_bioenergy_biomass-demand_liquid_lgn",
        "agr_bioenergy_biomass-demand_liquid",
        dim="Variables",
    )
    # Unit conversion [kcal] => [TWh]
    dm_fdk_oil.append(dm_fdk_eth, dim="Categories1")
    dm_fdk_oil.append(dm_fdk_lgn, dim="Categories1")
    dm_fdk_oil.change_unit(
        "agr_bioenergy_biomass-demand_liquid",
        kcal_to_TWh,
        old_unit="kcal",
        new_unit="TWh",
    )
    dm_fdk_liquid = dm_fdk_oil.copy()  # Rename

    # oil aps
    dm_oil_aps = dm_aps_ibp.filter(
        {"Variables": ["agr_aps"], "Categories2": ["fdk-voil"]}
    )
    dm_oil_aps.group_all("Categories2")
    dm_oil_aps.rename_col(
        "agr_aps", "agr_bioenergy_biomass-demand_liquid", dim="Variables"
    )
    dm_oil_aps.change_unit(
        "agr_bioenergy_biomass-demand_liquid",
        kcal_to_TWh,
        old_unit="kcal",
        new_unit="TWh",
    )
    dm_fdk_liquid.append(dm_oil_aps, dim="Categories1")

    # oil for oilcrops
    dm_voil_tpe.rename_col("oil-voil", "oil-oilcrop", dim="Categories1")
    dm_voil_tpe.change_unit(
        "agr_bioenergy_biomass-demand_liquid",
        factor=kcal_to_TWh,
        old_unit="kcal",
        new_unit="TWh",
    )
    dm_fdk_liquid.append(dm_voil_tpe, dim="Categories1")

    # lgn demand
    dm_liquid_lgn = dm_biofuel_fdk.filter(
        {"Variables": ["agr_bioenergy_biomass-demand_liquid_lgn"]}
    )
    dm_liquid_lgn.change_unit(
        "agr_bioenergy_biomass-demand_liquid_lgn",
        factor=kcal_to_TWh,
        old_unit="kcal",
        new_unit="TWh",
    )
    dm_liquid_lgn.deepen()
    dm_fdk_liquid.append(dm_liquid_lgn, dim="Categories1")

    dm_tpe.append(dm_fdk_liquid.flattest(), dim="Variables")

    # oil industry byproducts
    dm_aps_ibp_oil.change_unit(
        "agr_bioenergy_fdk-aby", factor=kcal_to_TWh, old_unit="kcal", new_unit="TWh"
    )
    dm_aps_ibp_oil = dm_aps_ibp_oil.flatten()

    # eth industry byproducts
    dm_eth_ind_bp = DM_alc_bev["biomass_hierarchy"].filter(
        {"Variables": ["agr_bev_ibp_use_oth"], "Categories1": ["biogasoline"]}
    )
    dm_eth_ind_bp.change_unit(
        "agr_bev_ibp_use_oth", factor=kcal_to_TWh, old_unit="kcal", new_unit="TWh"
    )
    dm_eth_ind_bp = dm_eth_ind_bp.flatten()
    dm_aps_ibp_oil.append(dm_eth_ind_bp, dim="Variables")
    dm_ind_bp = dm_aps_ibp_oil
    dm_tpe.append(dm_ind_bp.flattest(), dim="Variables")

    # Total bioenergy consumption (sum of liquid, biogas feedstock kcal) (solid not included in KNIME) FIXME check with crop_work if solid should be considered
    # Sum liquid & solid
    dm_bioenergy = dm_fdk_liquid.group_all("Categories1", inplace=False)
    dm_bioenergy.append(dm_ind_bp, dim="Variables")
    dm_bioenergy.groupby(
        {"agr_crop-cons_bioenergy": ".*"}, dim="Variables", inplace=True, regex=True
    )
    dm_bioenergy.change_unit(
        "agr_crop-cons_bioenergy", 1 / kcal_to_TWh, old_unit="TWh", new_unit="kcal"
    )
    dm_tpe.append(dm_bioenergy.flattest(), dim="Variables")

    # Notes : some oil categories seem to differ with KNIME (unit = kcal, tpe wants TWh)

    # Solid bioenergy - feedstock mix
    dm_fdk_solid = DM_bioenergy["solid-mix"].filter(
        {"Variables": ["agr_bioenergy_biomass-demand_solid"]}
    )
    dm_tpe.append(dm_fdk_solid.flattest(), dim="Variables")

    # Biogas feedstock mix
    dm_fdk_biogas = DM_bioenergy["digestor-mix"].filter(
        {"Variables": ["agr_bioenergy_biomass-demand_biogas"]}
    )
    dm_tpe.append(dm_fdk_biogas.flattest(), dim="Variables")

    # FOOD SUPPLY ------------------------------------------------------

    # Crop use
    # Total food from crop (does not include processed food)
    dm_crop_food = dm_lfs.filter_w_regex(
        {"Variables": "agr_demand", "Categories1": "crop.*"}
    )
    dm_crop_food.groupby(
        {"food_crop": ".*"}, dim="Categories1", regex=True, inplace=True
    )
    dm_tpe.append(dm_crop_food.flattest(), dim="Variables")
    # Total feed from crop
    dm_crop_feed = DM_feed["ration"].filter_w_regex(
        {"Variables": "agr_demand_feed", "Categories1": "crop.*"}
    )
    dm_crop_feed.groupby({"crop": ".*"}, dim="Categories1", regex=True, inplace=True)
    dm_tpe.append(dm_crop_feed.flattest(), dim="Variables")

    # Solid (same as bioenergy feedstock mix) Note : not included in KNIME

    # NON-FOOD : BEV & FIBER CROPS ------------------------------------------------------

    # Total non-food consumption (beverages and fiber crops) FIXME check with crop_work if okay to consider fiber crops
    # Beverages
    dm_crop_bev = dm_lfs_pro.filter_w_regex(
        {"Variables": "agr_domestic_production", "Categories1": "pro-bev.*"}
    )
    dm_crop_bev.groupby({"crop-bev": ".*"}, dim="Categories1", regex=True, inplace=True)
    dm_crop_bev = dm_crop_bev.flatten()
    # Fiber crops
    dm_crop_fiber = dm_fiber.filter_w_regex({"Variables": "agr_domestic-production.*"})
    # Unit conversion : [t] => [kcal]
    dm_crop_fiber.change_unit(
        "agr_domestic-production_fibres-plant-eq",
        4299300,
        old_unit="t",
        new_unit="kcal",
    )
    # Total non-food consumption [kcal] = bev + fibers
    dm_crop_bev.append(dm_crop_fiber, dim="Variables")
    dm_crop_bev.operation(
        "agr_domestic-production_fibres-plant-eq",
        "+",
        "agr_domestic_production_crop-bev",
        out_col="agr_crop-cons_non-food",
        unit="kcal",
    )
    dm_tpe.append(dm_crop_bev.flattest(), dim="Variables")
    # dm_tpe.append(dm_lfs.flattest(), dim='Variables')

    # SSR ------------------------------------------------------

    # Self-sufficiency ratio
    dm_ssr_food = DM_ssr["food"].flattest()
    dm_ssr_feed = DM_ssr["feed"].flattest()
    dm_ssr_bioenergy = DM_ssr["bioenergy"].flattest()
    dm_ssr_processed = DM_food_demand["food-net-import-pro"].flattest()
    dm_ssr = dm_ssr_food.copy()
    dm_ssr.append(dm_ssr_feed, dim="Variables")
    dm_ssr.append(dm_ssr_processed, dim="Variables")
    dm_ssr.append(dm_ssr_bioenergy, dim="Variables")
    # dm_tpe.append(dm_ssr, dim='Variables')

    # INPUTS ------------------------------------------------------

    # Input-use
    dm_input = dm_fertilizer_co.filter({"Variables": ["agr_input-use"]})
    dm_tpe.append(dm_input.flattest(), dim="Variables")

    # LAND USE ------------------------------------------------------

    # Land use
    dm_cropland = dm_cropland.filter({"Variables": ["agr_land_cropland"]})
    dm_tpe.append(dm_cropland.flattest(), dim="Variables")

    # Grassland
    dm_grassland = DM_livestock["ruminant_density"].filter(
        {
            "Variables": [
                "agr_lus_land_raw_grassland",
                "agr_climate-smart-livestock_density",
            ]
        }
    )
    dm_tpe.append(dm_grassland, dim="Variables")

    KPI = []
    yr = 2050
    # Land-Use (cropland)
    dm_cropland.group_all("Categories1", inplace=True)
    value = dm_cropland[0, yr, "agr_land_cropland"]
    KPI.append({"title": "Land use: Cropland", "value": value, "unit": "ha"})

    return dm_tpe, KPI


# ----------------------------------------------------------------------------------------------------------------------
# AGRICULTURE ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def agriculture(lever_setting, years_setting, DM_input, interface=Interface()):
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    (
        DM_ots_fts,
        DM_lifestyle,
        DM_food_demand,
        DM_livestock,
        DM_alc_bev,
        DM_bioenergy,
        DM_manure,
        DM_feed,
        DM_crop,
        DM_land,
        DM_nitrogen,
        DM_energy_ghg,
        CDM_const,
    ) = read_data(DM_input, lever_setting)

    cntr_list = DM_food_demand["food-net-import-pro"].col_labels["Country"]

    # Link interface or Simulate data from other modules
    if interface.has_link(from_sector="lifestyles", to_sector="agriculture"):
        DM_lfs = interface.get_link(from_sector="lifestyles", to_sector="agriculture")
        # FIXME ajouter lien pour la population dm_population
    else:
        if len(interface.list_link()) != 0:
            print("You are missing lifestyles to agriculture interface")
        DM_lfs = simulate_lifestyles_to_agriculture_input_new()
        for key in DM_lfs.keys():
            DM_lfs[key].filter({"Country": cntr_list}, inplace=True)

    if interface.has_link(from_sector="buildings", to_sector="agriculture"):
        dm_bld = interface.get_link(from_sector="buildings", to_sector="agriculture")
    else:
        if len(interface.list_link()) != 0:
            print("You are missing buildings to agriculture interface")
        dm_bld = simulate_buildings_to_agriculture_input()
        dm_bld.filter({"Country": cntr_list}, inplace=True)

    if interface.has_link(from_sector="industry", to_sector="agriculture"):
        DM_ind = interface.get_link(from_sector="industry", to_sector="agriculture")
    else:
        if len(interface.list_link()) != 0:
            print("You are missing industry to agriculture interface")
        DM_ind = simulate_industry_to_agriculture_input()
        for key in DM_ind.keys():
            DM_ind[key].filter({"Country": cntr_list}, inplace=True)

    if interface.has_link(from_sector="transport", to_sector="agriculture"):
        dm_tra = interface.get_link(from_sector="transport", to_sector="agriculture")
    else:
        if len(interface.list_link()) != 0:
            print("You are missing transport to agriculture interface")
        dm_tra = simulate_transport_to_agriculture_input()
        dm_tra.filter({"Country": cntr_list}, inplace=True)

    # CalculationTree AGRICULTURE

    dm_lfs, df_cal_rates_diet = wkf.lifestyle_workflow(
        DM_lifestyle, DM_lfs, CDM_const, years_setting
    )
    dm_lfs, dm_lfs_pro = wkf.food_demand_workflow(DM_food_demand, dm_lfs)
    (
        DM_livestock,
        dm_liv_ibp,
        dm_liv_ibp,
        dm_liv_prod,
        dm_liv_pop,
        df_cal_rates_liv_prod,
        df_cal_rates_liv_pop,
    ) = wkf.livestock_workflow(DM_livestock, CDM_const, dm_lfs_pro, years_setting)
    DM_alc_bev, dm_bev_ibp_cereal_feed, dm_bev_dom_prod = (
        wkf.alcoholic_beverages_workflow(DM_alc_bev, CDM_const, dm_lfs_pro)
    )
    DM_bioenergy, dm_oil, dm_lgn, dm_eth, dm_biofuel_fdk = wkf.bioenergy_workflow(
        DM_bioenergy, CDM_const, DM_ind, dm_bld, dm_tra
    )
    dm_liv_N2O, dm_CH4, df_cal_rates_liv_N2O, df_cal_rates_liv_CH4, DM_manure = (
        wkf.livestock_manure_workflow(
            DM_manure, DM_livestock, dm_liv_pop, CDM_const, years_setting
        )
    )
    DM_feed, dm_aps_ibp, dm_feed_req, dm_aps, dm_feed_demand = wkf.feed_workflow(
        DM_feed, dm_liv_prod, dm_bev_ibp_cereal_feed, CDM_const, years_setting
    )
    dm_voil, dm_aps_ibp_oil, dm_voil_tpe = wkf.biomass_allocation_workflow(
        dm_aps_ibp, dm_oil
    )
    (
        DM_crop,
        dm_crop,
        dm_crop_other,
        dm_feed_processed,
        dm_food_processed,
        df_cal_rates_crop,
        DM_ssr,
    ) = wkf.crop_workflow(
        DM_crop,
        DM_feed,
        DM_bioenergy,
        dm_voil,
        dm_lfs,
        dm_lfs_pro,
        dm_lgn,
        dm_aps_ibp,
        CDM_const,
        dm_oil,
        dm_bev_dom_prod,
        years_setting,
    )
    DM_land, dm_land, dm_land_use, dm_fiber, df_cal_rates_land, dm_cropland = (
        wkf.land_workflow(
            DM_land, DM_crop, DM_livestock, dm_crop_other, DM_ind, years_setting
        )
    )
    dm_n, dm_fertilizer_co, dm_mineral_fertilizer, df_cal_rates_n = (
        wkf.nitrogen_workflow(DM_nitrogen, dm_land, CDM_const, years_setting)
    )
    (
        DM_energy_ghg,
        dm_CO2,
        dm_input_use_CO2,
        dm_crop_residues,
        dm_CH4_liv_tpe,
        dm_N2O_liv_tpe,
        dm_CH4_rice,
        dm_fertilizer_N2O,
        df_cal_rates_ghg,
    ) = wkf.energy_ghg_workflow(
        DM_energy_ghg,
        DM_crop,
        DM_land,
        DM_manure,
        dm_land,
        dm_fertilizer_co,
        dm_liv_N2O,
        dm_CH4,
        CDM_const,
        dm_n,
        years_setting,
    )

    # INTERFACES OUT ---------------------------------------------------------------------------------------------------

    # interface to Land use
    DM_lus = agriculture_landuse_interface(
        DM_bioenergy, dm_lgn, dm_land_use, write_pickle=True
    )
    interface.add_link(from_sector="agriculture", to_sector="land-use", dm=DM_lus)

    # interface to Emissions
    dm_ems = agriculture_emissions_interface(
        DM_nitrogen,
        dm_CO2,
        DM_crop,
        DM_manure,
        DM_land,
        dm_input_use_CO2,
        dm_crop_residues,
        dm_CH4,
        dm_liv_N2O,
        dm_CH4_rice,
        dm_fertilizer_N2O,
        write_xls=False,
    )
    interface.add_link(from_sector="agriculture", to_sector="emissions", dm=dm_ems)

    # interface to Ammonia
    dm_ammonia = agriculture_ammonia_interface(dm_mineral_fertilizer)
    interface.add_link(from_sector="agriculture", to_sector="ammonia", dm=dm_ammonia)

    # interface to Oil Refinery
    dm_ref = agriculture_refinery_interface(DM_energy_ghg)
    interface.add_link(from_sector="agriculture", to_sector="oil-refinery", dm=dm_ref)

    # # interface to Storage
    # dm_storage = agriculture_storage_interface(DM_energy_ghg, write_xls=False)
    # interface.add_link(from_sector='agriculture', to_sector='power', dm=dm_storage)

    # interface to Energy
    DM_energy = agriculture_energy_interface(DM_energy_ghg, write_pickle=True)
    interface.add_link(from_sector="agriculture", to_sector="energy", dm=DM_energy)

    # interface to Minerals
    dm_minerals = agriculture_minerals_interface(DM_nitrogen, DM_bioenergy, dm_lgn)
    interface.add_link(from_sector="agriculture", to_sector="minerals", dm=dm_minerals)

    # TPE OUTPUT -------------------------------------------------------------------------------------------------------
    results_run, KPI = agriculture_TPE_interface(
        CDM_const,
        DM_livestock,
        DM_crop,
        dm_crop_other,
        DM_feed,
        dm_aps,
        dm_input_use_CO2,
        dm_crop_residues,
        dm_CH4,
        dm_liv_N2O,
        dm_CH4_rice,
        dm_fertilizer_N2O,
        DM_energy_ghg,
        DM_bioenergy,
        dm_lgn,
        dm_eth,
        dm_oil,
        dm_aps_ibp,
        DM_food_demand,
        dm_lfs_pro,
        dm_lfs,
        DM_land,
        dm_fiber,
        dm_aps_ibp_oil,
        dm_voil_tpe,
        DM_alc_bev,
        dm_biofuel_fdk,
        dm_liv_pop,
        DM_ssr,
        dm_fertilizer_co,
        DM_manure,
        dm_cropland,
    )

    return results_run, KPI


def agriculture_local_run():
    country_list = ["Switzerland", "Vaud"]
    DM_input = filter_country_and_load_data_from_pickles(
        country_list=country_list, modules_list="agriculture"
    )
    years_setting, lever_setting = init_years_lever()
    agriculture(lever_setting, years_setting, DM_input["agriculture"])
    return


if __name__ == "__main__":
    agriculture_local_run()
