import pandas as pd

from model.common.data_matrix_class import DataMatrix
from model.common.constant_data_matrix_class import ConstantDataMatrix
from model.common.io_database import dm_to_database
from model.common.interface_class import Interface
from model.common.auxiliary_functions import  calibration_rates, create_years_list
from model.common.auxiliary_functions import read_level_data, filter_country_and_load_data_from_pickles
import pickle
import json
import os
import numpy as np
import time


# __file__ = "/Users/crosnier/Documents/PathwayCalc/training/transport_module_notebook.py"


def init_years_lever():
    # function that can be used when running the module as standalone to initialise years and levers
    years_setting = [1990, 2023, 2025, 2050, 5]
    f = open('../config/lever_position.json')
    lever_setting = json.load(f)[0]
    return years_setting, lever_setting


#######################################################################################################
######################################### LOAD AGRICULTURE DATA #########################################
#######################################################################################################

# CalculationLeaf READ PICKLE
def read_data(DM_agriculture, lever_setting):

    # Read fts based on lever_setting
    # FIXME error it adds ots and fts
    # DM_check = check_ots_fts_match(DM_agriculture, lever_setting)
    DM_ots_fts = read_level_data(DM_agriculture, lever_setting)

    # FXA data matrix
    dm_fxa_cal_diet = DM_agriculture['fxa']['cal_agr_diet']
    dm_fxa_cal_liv_prod = DM_agriculture['fxa']['cal_agr_domestic-production-liv']
    dm_fxa_cal_liv_pop = DM_agriculture['fxa']['cal_agr_liv-population']
    dm_fxa_cal_liv_CH4 = DM_agriculture['fxa']['cal_agr_liv_CH4-emission']
    dm_fxa_cal_liv_N2O = DM_agriculture['fxa']['cal_agr_liv_N2O-emission']
    dm_fxa_cal_demand_feed = DM_agriculture['fxa']['cal_agr_demand_feed']
    # dm_fxa_cal_land = DM_agriculture['fxa']['cal_agr_lus_land']
    dm_fxa_ef_liv_N2O = DM_agriculture['fxa']['ef_liv_N2O-emission']
    dm_fxa_ef_liv_CH4_treated = DM_agriculture['fxa']['ef_liv_CH4-emission_treated']
    dm_fxa_liv_nstock = DM_agriculture['fxa']['liv_manure_n-stock']

    # Extract sub-data-matrices according to the flow
    # Sub-matrix for LIFESTYLE
    # dm_demography = DM_ots_fts['pop']['lfs_demography_']
    dm_diet_requirement = DM_ots_fts['kcal-req']
    dm_diet_split = DM_ots_fts['diet']['lfs_consumers-diet']
    dm_diet_share = DM_ots_fts['diet']['share']
    dm_diet_fwaste = DM_ots_fts['fwaste']
    # dm_population = DM_ots_fts['pop']['lfs_population_']

    # Sub-matrix for the FOOD DEMAND
    dm_food_net_import_pro = DM_ots_fts['food-net-import'].filter_w_regex(
        {'Categories1': 'pro-.*', 'Variables': 'agr_food-net-import'})

    # Sub-matrix for LIVESTOCK
    dm_livestock_losses = DM_ots_fts['climate-smart-livestock']['climate-smart-livestock_losses']
    dm_livestock_yield = DM_ots_fts['climate-smart-livestock']['climate-smart-livestock_yield']
    dm_livestock_slaughtered = DM_ots_fts['climate-smart-livestock']['climate-smart-livestock_slaughtered']
    dm_livestock_density = DM_ots_fts['climate-smart-livestock']['climate-smart-livestock_density']

    # Sub-matrix for ALCOHOLIC BEVERAGES
    dm_alc_bev = DM_ots_fts['biomass-hierarchy']['biomass-hierarchy-bev-ibp-use-oth']

    # Sub-matrix for BIOENERGY
    dm_bioenergy_cap_load_factor = DM_ots_fts['bioenergy-capacity']['bioenergy-capacity_load-factor']
    dm_bioenergy_cap_bgs_mix = DM_ots_fts['bioenergy-capacity']['bioenergy-capacity_bgs-mix']
    dm_bioenergy_cap_efficiency = DM_ots_fts['bioenergy-capacity']['bioenergy-capacity_efficiency']
    dm_bioenergy_cap_liq = DM_ots_fts['bioenergy-capacity']['bioenergy-capacity_liq_b']
    dm_bioenergy_cap_elec = DM_ots_fts['bioenergy-capacity']['bioenergy-capacity_elec']
    dm_bioenergy_mix_digestor = DM_ots_fts['biomass-hierarchy']['biomass-hierarchy_biomass-mix_digestor']
    dm_bioenergy_mix_solid = DM_ots_fts['biomass-hierarchy']['biomass-hierarchy_biomass-mix_solid']
    dm_bioenergy_mix_liquid = DM_ots_fts['biomass-hierarchy']['biomass-hierarchy_biomass-mix_liquid']
    dm_bioenergy_liquid_biodiesel = DM_ots_fts['biomass-hierarchy']['biomass-hierarchy_bioenergy_liquid_biodiesel']
    dm_bioenergy_liquid_biogasoline = DM_ots_fts['biomass-hierarchy']['biomass-hierarchy_bioenergy_liquid_biogasoline']
    dm_bioenergy_liquid_biojetkerosene = DM_ots_fts['biomass-hierarchy'][
        'biomass-hierarchy_bioenergy_liquid_biojetkerosene']
    dm_bioenergy_cap_elec.append(dm_bioenergy_cap_load_factor, dim='Variables')
    dm_bioenergy_cap_elec.append(dm_bioenergy_cap_efficiency, dim='Variables')

    # Sub-matrix for LIVESTOCK MANURE MANGEMENT & GHG EMISSIONS
    dm_livestock_enteric_emissions = DM_ots_fts['climate-smart-livestock']['climate-smart-livestock_enteric']
    dm_livestock_manure = DM_ots_fts['climate-smart-livestock']['climate-smart-livestock_manure']

    # Sub-matrix for FEED
    dm_ration = DM_ots_fts['climate-smart-livestock']['climate-smart-livestock_ration']
    dm_alt_protein = DM_ots_fts['alt-protein']

    # Sub-matrix for CROP
    dm_food_net_import_crop = DM_ots_fts['food-net-import'].filter_w_regex({'Categories1': 'crop-.*',
                                                                            'Variables': 'agr_food-net-import'})  # filtered here on purpose and not in the pickle (other parts of the datamatrix are used)
    dm_food_net_import_crop.rename_col_regex(str1="crop-", str2="", dim="Categories1")
    dm_crop = DM_ots_fts['climate-smart-crop']['climate-smart-crop_losses']
    dm_food_net_import_crop.drop(dim='Categories1', col_label=['stm'])
    dm_crop.append(dm_food_net_import_crop, dim='Variables')
    dm_residues_yield = DM_agriculture['fxa']['residues_yield']
    dm_hierarchy_residues_cereals = DM_ots_fts['biomass-hierarchy']['biomass-hierarchy_crop_cereal']
    dm_cal_crop = DM_agriculture['fxa']['cal_agr_domestic-production_food']
    # dm_crop.append(dm_cal_crop, dim='Variables')
    dm_ef_residues = DM_agriculture['fxa']['ef_burnt-residues']

    # Sub-matrix for LAND
    dm_cal_land = DM_agriculture['fxa']['cal_agr_lus_land']
    dm_yield = DM_ots_fts['climate-smart-crop']['climate-smart-crop_yield']
    dm_fibers = DM_agriculture['fxa']['fibers']
    dm_rice = DM_agriculture['fxa']['rice']
    dm_cal_cropland = DM_agriculture['fxa']['cal_agr_lus_land_cropland']

    # Sub-matrix for NITROGEN BALANCE
    dm_input = DM_ots_fts['climate-smart-crop']['climate-smart-crop_input-use']
    dm_fertilizer_emission = DM_agriculture['fxa']['agr_emission_fertilizer']
    dm_cal_n = DM_agriculture['fxa']['cal_agr_crop_emission_N2O-emission_fertilizer']
    # dm_fertilizer_emission.append(dm_cal_n, dim='Variables')

    # Sub-matrix for ENERGY & GHG EMISSIONS
    dm_cal_energy_demand = DM_agriculture['fxa']['cal_agr_energy-demand']
    dm_energy_demand = DM_ots_fts['climate-smart-crop']['climate-smart-crop_energy-demand']
    dm_cal_GHG = DM_agriculture['fxa']['cal_agr_emissions']
    dm_cal_GHG.deepen()
    dm_cal_input = DM_agriculture['fxa']['cal_agr_input-use_emissions-CO2']

    # Aggregated Data Matrix - ENERGY & GHG EMISSIONS
    DM_energy_ghg = {
        'energy_demand': dm_energy_demand,
        'cal_energy_demand': dm_cal_energy_demand,
        'cal_input': dm_cal_input,
        'cal_GHG': dm_cal_GHG
    }

    # Aggregate Data Matrix - LIFESTYLE
    DM_lifestyle = {
        'energy-requirement': dm_diet_requirement,
        'diet-split': dm_diet_split,
        'diet-share': dm_diet_share,
        'diet-fwaste': dm_diet_fwaste,
        # 'demography': dm_demography,
        # 'population': dm_population,
        'cal_diet': dm_fxa_cal_diet
    }

    # Aggregated Data Matrix - FOOD DEMAND
    DM_food_demand = {
        'food-net-import-pro': dm_food_net_import_pro
    }

    # Aggregated Data Matrix - LIVESTOCK
    DM_livestock = {
        'losses': dm_livestock_losses,
        'yield': dm_livestock_yield,
        'liv_slaughtered_rate': dm_livestock_slaughtered,
        'cal_liv_prod': dm_fxa_cal_liv_prod,
        'cal_liv_population': dm_fxa_cal_liv_pop,
        'ruminant_density': dm_livestock_density
    }

    # Aggregated Data Matrix - ALCOHOLIC BEVERAGES
    DM_alc_bev = {
        'biomass_hierarchy': dm_alc_bev
    }

    # Aggregated Data Matrix - BIOENERGY
    DM_bioenergy = {
        'electricity_production': dm_bioenergy_cap_elec,
        'bgs-mix': dm_bioenergy_cap_bgs_mix,
        'liq': dm_bioenergy_cap_liq,
        'digestor-mix': dm_bioenergy_mix_digestor,
        'solid-mix': dm_bioenergy_mix_solid,
        'liquid-mix': dm_bioenergy_mix_liquid,
        'liquid-biodiesel': dm_bioenergy_liquid_biodiesel,
        'liquid-biogasoline': dm_bioenergy_liquid_biogasoline,
        'liquid-biojetkerosene': dm_bioenergy_liquid_biojetkerosene
    }

    # Aggregated Data Matrix - LIVESTOCK MANURE MANAGEMENT & GHG EMISSIONS
    DM_manure = {
        'enteric_emission': dm_livestock_enteric_emissions,
        'manure': dm_livestock_manure,
        'cal_liv_CH4': dm_fxa_cal_liv_CH4,
        'cal_liv_N2O': dm_fxa_cal_liv_N2O,
        'ef_liv_N2O': dm_fxa_ef_liv_N2O,
        'ef_liv_CH4_treated': dm_fxa_ef_liv_CH4_treated,
        'liv_n-stock': dm_fxa_liv_nstock
    }

    # Aggregated Data Matrix - FEED
    DM_feed = {
        'ration': dm_ration,
        'alt-protein': dm_alt_protein,
        'cal_agr_demand_feed': dm_fxa_cal_demand_feed
    }

    # Aggregated Data Matrix - CROP
    DM_crop = {
        'crop': dm_crop,
        'cal_crop': dm_cal_crop,
        'ef_residues': dm_ef_residues,
        'residues_yield': dm_residues_yield,
        'hierarchy_residues_cereals': dm_hierarchy_residues_cereals,
        'food-net-import-pro': dm_food_net_import_pro
    }

    # Aggregated Data Matrix - LAND
    DM_land = {
        'cal_land': dm_cal_land,
        'cal_cropland': dm_cal_cropland,
        'yield': dm_yield,
        'fibers': dm_fibers,
        'rice': dm_rice
    }

    # Aggregated Data Matrix - NITROGEN BALANCE
    DM_nitrogen = {
        'input': dm_input,
        'emissions': dm_fertilizer_emission,
        'cal_n': dm_cal_n
    }

    CDM_const = DM_agriculture['constant']

    return DM_ots_fts, DM_lifestyle, DM_food_demand, DM_livestock, DM_alc_bev, DM_bioenergy, DM_manure, DM_feed, DM_crop, DM_land, DM_nitrogen, DM_energy_ghg, CDM_const


# SimulateInteractions
def simulate_lifestyles_to_agriculture_input_new():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../_database/data/interface/lifestyles_to_agriculture.pickle")
    with open(f, 'rb') as handle:
        DM_lfs = pickle.load(handle)

    return DM_lfs


def simulate_lifestyles_to_agriculture_input():
    # Read input from lifestyle : food waste & diet
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory,
                     "../_database/data/xls/All-Countries-interface_from-lifestyles-to-agriculture_EUCALC.xlsx")
    df = pd.read_excel(f, sheet_name="default")
    df_population = df.copy()
    df = df.drop(columns=['lfs_population_total[inhabitants]'])
    dm_lfs = DataMatrix.create_from_df(df, num_cat=1)

    # Read input from lifestyle : population
    df_population = df_population[['Years', 'Country', 'lfs_population_total[inhabitants]']]  # keep only population
    dm_population = DataMatrix.create_from_df(df_population, num_cat=0)

    # other way to do the step before but does not add it to the dm
    # idx = dm_lfs.idx
    # overall_food_demand = dm_lfs.array[:,:,idx['lfs_diet'],:] + dm_lfs.array[:,:,idx['lfs_food-wastes'],:]

    # Renaming to correct format to match iterators (simultaneously changes in lfs_diet, lfs_fwaste and agr_demand)
    # Adding meat prefix
    pro_liv_meat = ['bov', 'sheep', 'pigs', 'poultry', 'oth-animals']
    for cat in pro_liv_meat:
        new_cat = 'pro-liv-meat-' + cat
        # Dropping the -s at the end of pigs (for name matching reasons)
        if new_cat.endswith('pigs'):
            new_cat = new_cat[:-1]
        # Replacing bov by bovine (for name matching reasons)
        if new_cat == 'pro-liv-meat-bov':
            new_cat = 'pro-liv-meat-bovine'
        dm_lfs.rename_col(cat, new_cat, dim='Categories1')

    # Adding bev prefix
    pro_bev = ['beer', 'bev-fer', 'bev-alc', 'wine']
    for cat in pro_bev:
        new_cat = 'pro-bev-' + cat
        dm_lfs.rename_col(cat, new_cat, dim='Categories1')

    # Adding milk prefix
    pro_milk = ['milk']
    for cat in pro_milk:
        new_cat = 'pro-liv-abp-dairy-' + cat
        dm_lfs.rename_col(cat, new_cat, dim='Categories1')

    # Adding egg prefix
    pro_egg = ['egg']
    for cat in pro_egg:
        new_cat = 'pro-liv-abp-hens-' + cat
        dm_lfs.rename_col(cat, new_cat, dim='Categories1')

    # Adding crop prefix
    crop = ['cereals', 'oilcrops', 'pulses', 'starch', 'fruits', 'veg']
    for cat in crop:
        new_cat = 'crop-' + cat
        # Dropping the -s at the end of cereals, oilcrops, pulses, fruits (for name matching reasons)
        if new_cat.endswith('s'):
            new_cat = new_cat[:-1]
        dm_lfs.rename_col(cat, new_cat, dim='Categories1')

    # Adding abp-processed prefix
    processed = ['afats', 'offal']
    for cat in processed:
        new_cat = 'pro-liv-abp-processed-' + cat
        # Dropping the -s at the end afats (for name matching reasons)
        if new_cat.endswith('s'):
            new_cat = new_cat[:-1]
        dm_lfs.rename_col(cat, new_cat, dim='Categories1')

    # Adding crop processed prefix
    processed = ['voil', 'sweet', 'sugar']
    for cat in processed:
        new_cat = 'pro-crop-processed-' + cat
        dm_lfs.rename_col(cat, new_cat, dim='Categories1')

    dm_lfs.sort('Categories1')

    return dm_population


def simulate_buildings_to_agriculture_input():

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../_database/data/interface/buildings_to_agriculture.pickle")
    with open(f, 'rb') as handle:
        dm_bld = pickle.load(handle)

    return dm_bld


def simulate_industry_to_agriculture_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../_database/data/interface/industry_to_agriculture.pickle")
    with open(f, 'rb') as handle:
        DM_ind = pickle.load(handle)
    return DM_ind


def simulate_transport_to_agriculture_input():
    # Read input from lifestyle : food waste & diet
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../_database/data/interface/transport_to_agriculture.pickle")
    with open(f, 'rb') as handle:
        dm_tra = pickle.load(handle)
    return dm_tra


# CalculationLeaf LIFESTYLE TO DIET/FOOD DEMAND --------------------------------------------------------------
def lifestyle_workflow(DM_lifestyle, DM_lfs, CDM_const, years_setting):
    # Total kcal consumed
    dm_diet_split = DM_lifestyle['diet-split']
    ay_diet_intake = dm_diet_split.array[:, :, 0, :].sum(axis=-1)

    # [TUTORIAL] Compute tot kcal-req from kcal-req by age group
    dm_diet_requirement = DM_lifestyle['energy-requirement']
    dm_diet_requirement.append(DM_lfs['lfs_demography_'], dim='Variables')
    dm_diet_requirement.operation('lfs_demography', '*', 'agr_kcal-req', out_col='lfs_kcal-req', unit='kcal/day')
    dm_diet_requirement.group_all('Categories1')
    dm_diet_requirement.operation('lfs_kcal-req', '/', 'lfs_demography', out_col='lfs_kcal-req_req',
                                  unit='kcal/cap/day')
    dm_diet_requirement.filter({'Variables': ['lfs_kcal-req_req']}, inplace=True)

    # [TUTORIAL] Gap from healthy diet (Tree Parallel)
    dm_diet_requirement.add(ay_diet_intake, dim='Variables', col_label='lfs_energy-intake_total', unit='kcal/cap/day')
    dm_diet_requirement.operation('lfs_kcal-req_req', '-', 'lfs_energy-intake_total',
                                  dim="Variables", out_col='lfs_healthy-gap', unit='kcal/cap/day')

    # dm_population = DM_lifestyle['population']
    # [TUTORIAL] Consumer diet (operation with matrices with different structure/array specs)
    dm_diet_share = DM_lifestyle['diet-share']
    idx = dm_diet_requirement.idx
    ay_diet_consumers = dm_diet_share.array[:, :, 0, :] * dm_diet_requirement.array[:, :, idx['lfs_healthy-gap'],
                                                          np.newaxis]
    dm_diet_share.add(ay_diet_consumers, dim='Variables', col_label='lfs_consumers-diet', unit='kcal/cap/day')
    idx_d = dm_diet_share.idx
    # Calculate ay_total_diet
    dm_population = DM_lfs['lfs_population_']
    idx_p = dm_population.idx
    ay_total_diet = dm_diet_share.array[:, :, idx_d['lfs_consumers-diet'], :] * \
                    dm_population.array[:, :, idx_p['lfs_population_total'], np.newaxis] * 365
    start = time.time()
    dm_diet_tmp = DataMatrix.based_on(ay_total_diet[:, :, np.newaxis, :], dm_diet_share,
                                      change={'Variables': ['lfs_diet_raw']}, units={'lfs_diet_raw': 'kcal'})

    # Total Consumers food wastes
    dm_diet_fwaste = DM_lifestyle['diet-fwaste']
    cdm_lifestyle = CDM_const['cdm_lifestyle']
    idx = dm_population.idx
    idx_const = cdm_lifestyle.idx
    ay_total_fwaste = dm_diet_fwaste.array[:, :, 0, :] * dm_population.array[:, :, idx['lfs_population_total'],
                                                         np.newaxis] \
                      * cdm_lifestyle.array[idx_const['cp_time_days-per-year']]
    dm_diet_fwaste.add(ay_total_fwaste, dim='Variables', col_label='lfs_food-wastes',
                       unit='kcal')  # to bypass calibration data missing
    # dm_diet_fwaste.add(ay_total_fwaste, dim='Variables', col_label='lfs_food-wastes_raw', unit='kcal')

    # Total Consumers food supply (Total food intake)
    ay_total_food = dm_diet_split.array[:, :, 0, :] * dm_population.array[:, :, idx['lfs_population_total'], np.newaxis] \
                    * cdm_lifestyle.array[idx_const['cp_time_days-per-year']]
    dm_diet_food = DataMatrix.based_on(ay_total_food[:, :, np.newaxis, :], dm_diet_split,
                                       change={'Variables': ['lfs_diet_raw']}, units={'lfs_diet_raw': 'kcal'})

    # Total calorie demand = food intake + food waste
    dm_diet_food.append(dm_diet_tmp, dim='Categories1')  # Append all food categories
    dm_diet_food.append(dm_diet_fwaste, dim='Variables')  # Append with fwaste
    dm_diet_food.operation('lfs_diet_raw', '+', 'lfs_food-wastes', dim='Variables', out_col='agr_demand_raw',
                           unit='kcal')
    dm_diet_food.filter({'Variables': ['agr_demand_raw']}, inplace=True)

    # Calibration factors
    dm_cal_diet = DM_lifestyle['cal_diet']
    # Add dummy caf for afats and rice
    # dm_cal_diet.add(1, dummy=True, col_label=['afats', 'rice'], dim='Categories1')

    # Calibration - Food supply (accounting for food wastes)
    # dm_diet_food.append(dm_diet_tmp, dim='Categories1')
    dm_cal_rates_diet = calibration_rates(dm_diet_food, dm_cal_diet, calibration_start_year=1990,
                                          calibration_end_year=2023,
                                          years_setting=years_setting)
    dm_diet_food.append(dm_cal_rates_diet, dim='Variables')
    dm_diet_food.operation('agr_demand_raw', '*', 'cal_rate', dim='Variables', out_col='agr_demand', unit='kcal')
    df_cal_rates_diet = dm_to_database(dm_cal_rates_diet, 'none', 'agriculture',
                                       level=0)  # Exporting calibration rates to check at the end

    # Data to return to the TPE
    dm_diet_food.append(dm_diet_fwaste, dim='Variables')

    # Create copy
    dm_lfs = dm_diet_food.copy()

    # Format for same categories as rest Agriculture module
    cat_lfs = ['afats', 'beer', 'bev-alc', 'bev-fer', 'bov', 'cereals', 'coffee', 'dfish', 'egg', 'ffish', 'fruits', \
               'milk', 'offal', 'oilcrops', 'oth-animals', 'oth-aq-animals', 'pfish', 'pigs', 'poultry', 'pulses',
               'rice', 'seafood', 'sheep', 'starch', 'stm', 'sugar', 'sweet', 'veg', 'voil', 'wine']
    cat_agr = ['pro-liv-abp-processed-afat', 'pro-bev-beer', 'pro-bev-bev-alc', 'pro-bev-bev-fer',
               'pro-liv-meat-bovine',
               'crop-cereal', 'coffee', 'dfish', 'pro-liv-abp-hens-egg', 'ffish', 'crop-fruit',
               'pro-liv-abp-dairy-milk',
               'pro-liv-abp-processed-offal', 'crop-oilcrop', 'pro-liv-meat-oth-animals', 'oth-aq-animals', 'pfish',
               'pro-liv-meat-pig', 'pro-liv-meat-poultry', 'crop-pulse', 'rice', 'seafood', 'pro-liv-meat-sheep',
               'crop-starch', 'stm', 'pro-crop-processed-sugar', 'pro-crop-processed-sweet', 'crop-veg',
               'pro-crop-processed-voil', 'pro-bev-wine']

    dm_lfs.rename_col(cat_lfs, cat_agr, 'Categories1')
    dm_lfs.sort('Categories1')

    return dm_lfs, df_cal_rates_diet


# CalculationLeaf FOOD DEMAND TO DOMESTIC FOOD PRODUCTION --------------------------------------------------------------
def food_demand_workflow(DM_food_demand, dm_lfs):
    # Overall food demand [kcal] = food demand [kcal] + food waste [kcal] NOW IN lifestyle_workflow()
    # dm_lfs.operation('lfs_total-cal-demand', '+', 'lfs_food-wastes', out_col='agr_demand', unit='kcal')

    # Filtering dms to only keep pro
    dm_lfs_pro = dm_lfs.filter_w_regex({'Categories1': 'pro-.*', 'Variables': 'agr_demand'})
    food_net_import_pro = DM_food_demand['food-net-import-pro'].filter_w_regex(
        {'Categories1': 'pro-.*', 'Variables': 'agr_food-net-import'})
    # Dropping the unwanted columns
    food_net_import_pro.drop(dim='Categories1', col_label=['pro-crop-processed-cake', 'pro-crop-processed-molasse'])

    # Sorting the dms alphabetically
    food_net_import_pro.sort(dim='Categories1')
    dm_lfs_pro.sort(dim='Categories1')

    # Domestic production processed food [kcal] = agr_demand_pro_(.*) [kcal] * net-imports_pro_(.*) [%]
    idx_lfs = dm_lfs_pro.idx
    idx_import = food_net_import_pro.idx
    agr_domestic_production = dm_lfs_pro.array[:, :, idx_lfs['agr_demand'], :] \
                              * food_net_import_pro.array[:, :, idx_import['agr_food-net-import'], :]

    # Adding agr_domestic_production to dm_lfs_pro
    dm_lfs_pro.add(agr_domestic_production, dim='Variables', col_label='agr_domestic_production', unit='kcal')

    return dm_lfs, dm_lfs_pro


# CalculationLeaf ANIMAL SOURCED FOOD DEMAND TO LIVESTOCK POPULATION AND LIVESTOCK PRODUCTS ----------------------------
def livestock_workflow(DM_livestock, CDM_const, dm_lfs_pro, years_setting):
    # Filter dm_lfs_pro to only have livestock products
    dm_lfs_pro_liv = dm_lfs_pro.filter_w_regex({'Categories1': 'pro-liv.*', 'Variables': 'agr_domestic_production'})
    # Drop the pro- prefix of the categories
    dm_lfs_pro_liv.rename_col_regex(str1="pro-liv-", str2="", dim="Categories1")
    # Sort the dms
    dm_lfs_pro_liv.sort(dim='Categories1')
    DM_livestock['losses'].sort(dim='Categories1')
    DM_livestock['yield'].sort(dim='Categories1')

    # Append dm_lfs_pro_liv to DM_livestock['losses']
    DM_livestock['losses'].append(dm_lfs_pro_liv, dim='Variables')

    # Livestock domestic prod with losses [kcal] = livestock domestic prod [kcal] * Production losses livestock [%]
    DM_livestock['losses'].operation('agr_climate-smart-livestock_losses', '*', 'agr_domestic_production',
                                     out_col='agr_domestic_production_liv_afw_raw', unit='kcal')

    # Calibration - Livestock domestic production
    dm_cal_liv_prod = DM_livestock['cal_liv_prod']
    dm_liv_prod = DM_livestock['losses'].filter({'Variables': ['agr_domestic_production_liv_afw_raw']})
    dm_liv_prod.drop(dim='Categories1', col_label=['abp-processed-offal',
                                                   'abp-processed-afat'])  # Filter dm_liv_prod to drop offal & afats
    dm_cal_rates_liv_prod = calibration_rates(dm_liv_prod, dm_cal_liv_prod, calibration_start_year=1990,
                                              calibration_end_year=2023, years_setting=years_setting)
    dm_liv_prod.append(dm_cal_rates_liv_prod, dim='Variables')
    dm_liv_prod.operation('agr_domestic_production_liv_afw_raw', '*', 'cal_rate', dim='Variables',
                          out_col='agr_domestic_production_liv_afw', unit='kcal')
    df_cal_rates_liv_prod = dm_to_database(dm_cal_rates_liv_prod, 'none', 'agriculture', level=0)

    # DM_livestock['cal_liv_prod'].append(dm_cal_rates_liv_prod, dim='Variables')
    # DM_livestock['cal_liv_prod'].operation('caf_agr_domestic-production-liv', '*', 'agr_domestic_production_liv_afw',
    #                                       dim="Variables", out_col='cal_agr_domestic_production_liv_afw', unit='kcal')

    # Livestock slaughtered [lsu] = meat demand [kcal] / livestock meat content [kcal/lsu]
    dm_liv_slau = dm_liv_prod.filter({'Variables': ['agr_domestic_production_liv_afw']})
    DM_livestock['yield'].append(dm_liv_slau, dim='Variables')  # Append cal_agr_domestic_production_liv_afw in yield
    DM_livestock['yield'].operation('agr_domestic_production_liv_afw', '/', 'agr_climate-smart-livestock_yield',
                                    dim="Variables", out_col='agr_liv_population_slau', unit='lsu')

    # Livestock population (stock) [lsu] = Livestock slaughtered [lsu] / slaughter rate [%]
    dm_liv_slau_egg_dairy = DM_livestock['yield'].filter({'Variables': ['agr_liv_population_slau']})
    DM_livestock['liv_slaughtered_rate'].append(dm_liv_slau_egg_dairy, dim='Variables')
    # dm_liv_slau_meat = DM_livestock['yield'].filter({'Variables': ['agr_liv_population_raw'],
    #                                                 'Categories1': ['meat-bovine', 'meat-pig', 'meat-poultry',
    #                                                                 'meat-sheep', 'meat-oth-animals']})
    # DM_livestock['liv_slaughtered_rate'].append(dm_liv_slau_meat, dim='Variables')
    DM_livestock['liv_slaughtered_rate'].operation('agr_liv_population_slau', '/',
                                                   'agr_climate-smart-livestock_slaughtered',
                                                   dim="Variables", out_col='agr_liv_population_raw', unit='lsu')

    # Processing for calibration: Livestock population for meat, eggs and dairy ( meat pop & slaughtered livestock for eggs and dairy)
    # Filtering eggs, dairy and meat
    # dm_liv_slau_egg_dairy = DM_livestock['yield'].filter(
    #    {'Variables': ['agr_liv_population_raw'], 'Categories1': ['abp-dairy-milk', 'abp-hens-egg']})
    # dm_liv_slau_meat = DM_livestock['liv_slaughtered_rate'].filter({'Variables': ['agr_liv_population_meat']})
    # Rename dm_liv_slau_meat variable to match with dm_liv_slau_egg_dairy
    # dm_liv_slau_meat.rename_col('agr_liv_population_meat', 'agr_liv_population_raw', dim='Variables')
    # Appending between livestock population
    # dm_liv_slau_egg_dairy.append(dm_liv_slau_meat, dim='Categories1')

    # Calibration Livestock population
    dm_cal_liv_pop = DM_livestock['cal_liv_population']
    dm_liv_pop = DM_livestock['liv_slaughtered_rate'].filter({'Variables': ['agr_liv_population_raw']})
    dm_cal_rates_liv_pop = calibration_rates(dm_liv_pop, dm_cal_liv_pop, calibration_start_year=1990,
                                             calibration_end_year=2022, years_setting=years_setting)
    dm_liv_pop.append(dm_cal_rates_liv_pop, dim='Variables')
    dm_liv_pop.operation('agr_liv_population_raw', '*', 'cal_rate', dim='Variables', out_col='agr_liv_population',
                         unit='lsu')
    # dm_liv_slau_egg_dairy.operation('agr_liv_population_raw', '*', 'cal_rate', dim='Variables', out_col='agr_liv_population', unit='lsu')
    df_cal_rates_liv_pop = dm_to_database(dm_cal_rates_liv_pop, 'none', 'agriculture', level=0)

    # GRAZING LIVESTOCK
    # Filtering ruminants (bovine & sheep)
    dm_liv_ruminants = dm_liv_pop.filter(
        {'Variables': ['agr_liv_population'], 'Categories1': ['meat-bovine', 'meat-sheep', 'abp-dairy-milk']})
    # Ruminant livestock [lsu] = population bovine + population sheep + population dairy
    dm_liv_ruminants.groupby({'ruminant': '.*'}, dim='Categories1', regex=True, inplace=True)
    # Append to relevant dm
    dm_liv_ruminants = dm_liv_ruminants.filter({'Variables': ['agr_liv_population'], 'Categories1': ['ruminant']})
    dm_liv_ruminants = dm_liv_ruminants.flatten()  # change from category to variable
    DM_livestock['ruminant_density'].append(dm_liv_ruminants, dim='Variables')  # Append to caf
    # Agriculture grassland [ha] = ruminant livestock [lsu] / livestock density [lsu/ha]
    DM_livestock['ruminant_density'].operation('agr_liv_population_ruminant', '/',
                                               'agr_climate-smart-livestock_density',
                                               dim="Variables", out_col='agr_lus_land_raw_grassland', unit='ha')

    # LIVESTOCK BYPRODUCTS
    # Filter ibp constants for offal
    cdm_cp_ibp_offal = CDM_const['cdm_cp_ibp_offal']

    # Filter ibp constants for afat
    cdm_cp_ibp_afat = CDM_const['cdm_cp_ibp_afat']

    # Filter cal_agr_liv_population for meat
    cal_liv_population_meat = dm_liv_pop.filter_w_regex(
        {'Variables': 'agr_liv_population', 'Categories1': 'meat'})
    # DM_livestock['liv_slaughtered_rate'].append(cal_liv_population_meat,
    #                                            dim='Variables')  # Appending to the dm that has the same categories

    # Offal per livestock type [kcal] = livestock population meat [lsu] * yield offal [kcal/lsu]
    idx_liv_pop = cal_liv_population_meat.idx
    idx_cdm_offal = cdm_cp_ibp_offal.idx
    agr_ibp_offal = cal_liv_population_meat.array[:, :, idx_liv_pop['agr_liv_population'], :] \
                    * cdm_cp_ibp_offal.array[idx_cdm_offal['cp_ibp_liv']]
    cal_liv_population_meat.add(agr_ibp_offal, dim='Variables', col_label='agr_ibp_offal', unit='kcal')

    # Afat per livestock type [kcal] = livestock population meat [lsu] * yield afat [kcal/lsu]
    idx_liv_pop = cal_liv_population_meat.idx
    idx_cdm_afat = cdm_cp_ibp_afat.idx
    agr_ibp_afat = cal_liv_population_meat.array[:, :, idx_liv_pop['agr_liv_population'], :] \
                   * cdm_cp_ibp_afat.array[idx_cdm_afat['cp_ibp_liv']]
    cal_liv_population_meat.add(agr_ibp_afat, dim='Variables', col_label='agr_ibp_afat', unit='kcal')

    # Totals offal/afat [kcal] = sum (Offal/afat per livestock type [kcal])
    dm_offal = cal_liv_population_meat.filter({'Variables': ['agr_ibp_offal']})
    dm_liv_ibp = dm_offal.copy()
    dm_liv_ibp.groupby({'offal': '.*'}, dim='Categories1', regex=True, inplace=True)
    dm_afat = cal_liv_population_meat.filter({'Variables': ['agr_ibp_afat']})
    dm_total_afat = dm_afat.copy()
    dm_total_afat.groupby({'afat': '.*'}, dim='Categories1', regex=True, inplace=True)

    # Append Totals offal with total afat and rename variable
    dm_liv_ibp.rename_col('agr_ibp_offal', 'agr_ibp', "Variables")
    dm_total_afat.rename_col('agr_ibp_afat', 'agr_ibp', "Variables")
    dm_liv_ibp.append(dm_total_afat, dim='Categories1')
    dm_liv_ibp.rename_col('agr_ibp', 'agr_ibp_total', dim='Variables')

    # Filter Processed offal/afats afw (not calibrated), rename and append with dm_liv_ibp
    dm_processed_offal_afat = DM_livestock['losses'].filter({'Variables': ['agr_domestic_production_liv_afw_raw'],
                                                             'Categories1': ['abp-processed-offal',
                                                                             'abp-processed-afat']})
    dm_processed_offal_afat.rename_col_regex(str1="abp-processed-", str2="", dim="Categories1")
    dm_liv_ibp.append(dm_processed_offal_afat, dim='Variables')

    # Offal/afats for feedstock [kcal] = produced offal/afats [kcal] - processed offal/afat [kcal]
    dm_liv_ibp.operation('agr_ibp_total', '-', 'agr_domestic_production_liv_afw_raw', out_col='agr_ibp_liv_fdk',
                         unit='kcal')

    # Total offal and afats for feedstock [kcal] = Offal for feedstock [kcal] + Afats for feedstock [kcal]
    dm_ibp_fdk = dm_liv_ibp.filter({'Variables': ['agr_ibp_liv_fdk']})
    dm_liv_ibp.groupby({'total': '.*'}, dim='Categories1', regex=True, inplace=True)

    return DM_livestock, dm_liv_ibp, dm_liv_ibp, dm_liv_prod, dm_liv_pop, df_cal_rates_liv_prod, df_cal_rates_liv_pop


# CalculationLeaf ALCOHOLIC BEVERAGES INDUSTRY -------------------------------------------------------------------------
def alcoholic_beverages_workflow(DM_alc_bev, CDM_const, dm_lfs_pro):
    # From FOOD DEMAND filtering domestic production bev and renaming
    # Beer
    dm_bev_beer = dm_lfs_pro.filter_w_regex({'Categories1': 'pro-bev-beer.*', 'Variables': 'agr_domestic_production'})
    dm_bev_beer.rename_col_regex(str1="pro-bev-", str2="", dim="Categories1")
    dm_bev_beer = dm_bev_beer.flatten()
    # Bev-alc
    dm_bev_alc = dm_lfs_pro.filter_w_regex({'Categories1': 'pro-bev-bev-alc.*', 'Variables': 'agr_domestic_production'})
    dm_bev_alc.rename_col_regex(str1="pro-bev-", str2="", dim="Categories1")
    dm_bev_alc = dm_bev_alc.flatten()
    # Bev-fer
    dm_bev_fer = dm_lfs_pro.filter_w_regex({'Categories1': 'pro-bev-bev-fer.*', 'Variables': 'agr_domestic_production'})
    dm_bev_fer.rename_col_regex(str1="pro-bev-", str2="", dim="Categories1")
    dm_bev_fer = dm_bev_fer.flatten()
    # Wine
    dm_bev_wine = dm_lfs_pro.filter_w_regex({'Categories1': 'pro-bev-wine.*', 'Variables': 'agr_domestic_production'})
    dm_bev_wine.rename_col_regex(str1="pro-bev-", str2="", dim="Categories1")
    dm_bev_wine = dm_bev_wine.flatten()

    # Constants and sorting according to bev type (beer, wine, bev-alc, bev-fer)
    cdm_cp_ibp_bev_beer = CDM_const['cdm_cp_ibp_bev_beer']
    cdm_cp_ibp_bev_wine = CDM_const['cdm_cp_ibp_bev_wine']
    cdm_cp_ibp_bev_alc = CDM_const['cdm_cp_ibp_bev_alc']
    cdm_cp_ibp_bev_fer = CDM_const['cdm_cp_ibp_bev_fer']

    # FRUIT & CEREAL DEMAND FOR BEVERAGES ------------------------------------------------------------------------------

    # Beer - Crop Cereal
    idx_dm_bev_beer = dm_bev_beer.idx
    idx_cdm_ibp_beer = cdm_cp_ibp_bev_beer.idx
    agr_ibp_bev_beer_crop_cereal = dm_bev_beer.array[:, :, idx_dm_bev_beer['agr_domestic_production_beer']] \
                                   * cdm_cp_ibp_bev_beer.array[idx_cdm_ibp_beer['cp_ibp_bev_beer_brf_crop_cereal']]
    dm_bev_beer.add(agr_ibp_bev_beer_crop_cereal, dim='Variables', col_label='agr_ibp_bev_beer_crop_cereal',
                    unit='kcal')

    # Bev-fer - Crop cereal
    idx_dm_bev_fer = dm_bev_fer.idx
    idx_cdm_ibp_fer = cdm_cp_ibp_bev_fer.idx
    agr_ibp_bev_fer_crop_cereal = dm_bev_fer.array[:, :, idx_dm_bev_fer['agr_domestic_production_bev-fer']] \
                                  * cdm_cp_ibp_bev_fer.array[idx_cdm_ibp_fer['cp_ibp_bev_bev-fer_brf_crop_cereal']]
    dm_bev_fer.add(agr_ibp_bev_fer_crop_cereal, dim='Variables', col_label='agr_ibp_bev_bev-fer_crop_cereal',
                   unit='kcal')

    # Bev-alc - Crop fruit
    idx_dm_bev_alc = dm_bev_alc.idx
    idx_cdm_ibp_alc = cdm_cp_ibp_bev_alc.idx
    agr_ibp_bev_alc_crop_fruit = dm_bev_alc.array[:, :, idx_dm_bev_alc['agr_domestic_production_bev-alc']] \
                                 * cdm_cp_ibp_bev_alc.array[idx_cdm_ibp_alc['cp_ibp_bev_bev-alc_brf_crop_fruit']]
    dm_bev_alc.add(agr_ibp_bev_alc_crop_fruit, dim='Variables', col_label='agr_ibp_bev_bev-alc_crop_fruit',
                   unit='kcal')

    # Wine - Crop Grape (fruit)
    idx_dm_bev_wine = dm_bev_wine.idx
    idx_cdm_ibp_wine = cdm_cp_ibp_bev_wine.idx
    agr_ibp_bev_wine_crop_grape = dm_bev_wine.array[:, :, idx_dm_bev_wine['agr_domestic_production_wine']] \
                                  * cdm_cp_ibp_bev_wine.array[idx_cdm_ibp_wine['cp_ibp_bev_wine_brf_crop_grape']]
    dm_bev_wine.add(agr_ibp_bev_wine_crop_grape, dim='Variables', col_label='agr_ibp_bev_wine_crop_fruit', unit='kcal')

    # Append together
    dm_bev_dom_prod = dm_bev_beer.copy()
    dm_bev_dom_prod.append(dm_bev_alc, dim='Variables')
    dm_bev_dom_prod.append(dm_bev_fer, dim='Variables')
    dm_bev_dom_prod.append(dm_bev_wine, dim='Variables')

    # Cereals domestic production for beverages = cereals for beer + cereals for bev fer
    dm_bev_dom_prod.operation('agr_ibp_bev_beer_crop_cereal', '+',
                              'agr_ibp_bev_bev-fer_crop_cereal',
                              out_col='agr_domestic-production_bev_cereal', unit='kcal')

    # Fruit domestic production for beverages = fruits for bev-alc + fruits for wine
    dm_bev_dom_prod.operation('agr_ibp_bev_bev-alc_crop_fruit', '+',
                              'agr_ibp_bev_wine_crop_fruit',
                              out_col='agr_domestic-production_bev_fruit', unit='kcal')

    # Filter and deepen
    dm_bev_dom_prod = dm_bev_dom_prod.filter({'Variables': ['agr_domestic-production_bev_cereal',
                                                            'agr_domestic-production_bev_fruit']})
    dm_bev_dom_prod.deepen()

    # BYPRODUCT PRODUCTION OF BEVERAGES ------------------------------------------------------------------------------

    # Byproducts per bev type [kcal] = agr_domestic_production bev [kcal] * yields [%]
    # Beer - Feedstock Yeast
    idx_dm_bev_beer = dm_bev_beer.idx
    idx_cdm_ibp_beer = cdm_cp_ibp_bev_beer.idx
    agr_ibp_bev_beer_fdk_yeast = dm_bev_beer.array[:, :, idx_dm_bev_beer['agr_domestic_production_beer']] \
                                 * cdm_cp_ibp_bev_beer.array[idx_cdm_ibp_beer['cp_ibp_bev_beer_brf_fdk_yeast']]
    dm_bev_beer.add(agr_ibp_bev_beer_fdk_yeast, dim='Variables', col_label='agr_ibp_bev_beer_fdk_yeast', unit='kcal')

    # Beer - Feedstock Cereal
    idx_dm_bev_beer = dm_bev_beer.idx
    idx_cdm_ibp_beer = cdm_cp_ibp_bev_beer.idx
    agr_ibp_bev_beer_fdk_cereal = dm_bev_beer.array[:, :, idx_dm_bev_beer['agr_domestic_production_beer']] \
                                  * cdm_cp_ibp_bev_beer.array[idx_cdm_ibp_beer['cp_ibp_bev_beer_brf_fdk_cereal']]
    dm_bev_beer.add(agr_ibp_bev_beer_fdk_cereal, dim='Variables', col_label='agr_ibp_bev_beer_fdk_cereal', unit='kcal')

    # Wine - Feedstock Marc
    idx_dm_bev_wine = dm_bev_wine.idx
    idx_cdm_ibp_wine = cdm_cp_ibp_bev_wine.idx
    agr_ibp_bev_wine_fdk_marc = dm_bev_wine.array[:, :, idx_dm_bev_wine['agr_domestic_production_wine']] \
                                * cdm_cp_ibp_bev_wine.array[idx_cdm_ibp_wine['cp_ibp_bev_wine_brf_fdk_marc']]
    dm_bev_wine.add(agr_ibp_bev_wine_fdk_marc, dim='Variables', col_label='agr_ibp_bev_wine_fdk_marc', unit='kcal')

    # Wine - Feedstock Lees
    idx_dm_bev_wine = dm_bev_wine.idx
    idx_cdm_ibp_wine = cdm_cp_ibp_bev_wine.idx
    agr_ibp_bev_wine_fdk_lees = dm_bev_wine.array[:, :, idx_dm_bev_wine['agr_domestic_production_wine']] \
                                * cdm_cp_ibp_bev_wine.array[idx_cdm_ibp_wine['cp_ibp_bev_wine_brf_fdk_lees']]
    dm_bev_wine.add(agr_ibp_bev_wine_fdk_lees, dim='Variables', col_label='agr_ibp_bev_wine_fdk_lees', unit='kcal')

    # Byproducts for other uses [kcal] = sum (wine byproducts [kcal])
    dm_bev_wine.operation('agr_ibp_bev_wine_fdk_marc', '+',
                          'agr_ibp_bev_wine_fdk_lees',
                          out_col='agr_bev_ibp_use_oth', unit='kcal')
    dm_bev_ibp_use_oth = dm_bev_wine.filter({'Variables': ['agr_bev_ibp_use_oth']})

    # Byproducts biomass use per sector = byproducts for other uses * share of bev biomass per sector [%]
    idx_bev_ibp_use_oth = dm_bev_ibp_use_oth.idx
    idx_bev_biomass_hierarchy = DM_alc_bev['biomass_hierarchy'].idx
    agr_bev_ibp_use_oth = dm_bev_ibp_use_oth.array[:, :, idx_bev_ibp_use_oth['agr_bev_ibp_use_oth'], np.newaxis] * \
                          DM_alc_bev['biomass_hierarchy'].array[:, :,
                          idx_bev_biomass_hierarchy['agr_biomass-hierarchy-bev-ibp-use-oth'], :]
    DM_alc_bev['biomass_hierarchy'].add(agr_bev_ibp_use_oth, dim='Variables', col_label='agr_bev_ibp_use_oth',
                                        unit='kcal')

    # Cereal bev byproducts allocated to feed [kcal] = sum (beer byproducts for feedstock [kcal])
    dm_bev_beer.operation('agr_ibp_bev_beer_fdk_yeast', '+',
                          'agr_ibp_bev_beer_fdk_cereal',
                          out_col='agr_use_bev_ibp_cereal_feed', unit='kcal')
    dm_bev_ibp_cereal_feed = dm_bev_beer.filter({'Variables': ['agr_use_bev_ibp_cereal_feed']})

    # (Not used after) Fruits bev allocated to non-food [kcal] = dom prod bev alc + dom prod bev wine + bev byproducts for fertilizer

    # (Not used after) Cereals bev allocated to non-food [kcal] = dom prod bev beer + dom prod bev fer + bev byproducts for fertilizer
    # change the double count of bev byproducts for fertilizer in fruits/cereals bev allocated to non-food [kcal]

    # (Not used after) Fruits bev allocated to bioenergy [kcal] = bp bev for solid bioenergy (+ bp use for ethanol (not found in knime))
    return DM_alc_bev, dm_bev_ibp_cereal_feed, dm_bev_dom_prod


# CalculationLeaf BIOENERGY CAPACITY ----------------------------------------------------------------------------------
def bioenergy_workflow(DM_bioenergy, CDM_const, DM_ind, dm_bld, dm_tra):
    # Constant
    cdm_load = CDM_const['cdm_load']

    # Electricity production
    # Bioenergy capacity [TWh] = bioenergy capacity [GW] * load hours per year [h] (accounting for unit change)
    idx_bio_cap_elec = DM_bioenergy['electricity_production'].idx
    idx_const = cdm_load.idx
    dm_bio_cap = DM_bioenergy['electricity_production'].array[:, :, idx_bio_cap_elec['agr_bioenergy-capacity_elec'], :] \
                 * cdm_load.array[idx_const['cp_load_hours-per-year-twh']]
    DM_bioenergy['electricity_production'].add(dm_bio_cap, dim='Variables', col_label='agr_bioenergy-capacity_lfe',
                                               unit='TWh')

    # Electricity production [TWh] = bioenergy capacity [TWh] * load-factors per technology [%]
    DM_bioenergy['electricity_production'].operation('agr_bioenergy-capacity_lfe', '*',
                                                     'agr_bioenergy-capacity_load-factor',
                                                     out_col='agr_bioenergy-capacity_elec-prod', unit='TWh')

    # Feedstock requirements [TWh] = Electricity production [TWh] / Efficiency per technology [%]
    DM_bioenergy['electricity_production'].operation('agr_bioenergy-capacity_elec-prod', '/',
                                                     'agr_bioenergy-capacity_efficiency',
                                                     out_col='agr_bioenergy-capacity_fdk-req', unit='TWh')

    # Filtering input from other modules
    # Industry
    dm_ind_bioenergy = DM_ind["bioenergy"].copy()
    dm_ind_biomaterial = DM_ind["biomaterial"].copy()

    # BIOGAS -----------------------------------------------------------------------------------------------------------
    # Biogas feedstock requirements [TWh] =
    # (transport + bld + industry bioenergy + industry biomaterial) bio gas demand + biogases feedstock requirements
    idx_bld = dm_bld.idx
    idx_ind_bioenergy = dm_ind_bioenergy.idx
    idx_ind_biomaterial = dm_ind_biomaterial.idx
    idx_tra = dm_tra.idx
    idx_elec = DM_bioenergy['electricity_production'].idx

    dm_bio_gas_demand = dm_bld.array[:, :, idx_bld['bld_bioenergy'], idx_bld['gas-bio']] \
                        + dm_ind_bioenergy.array[:, :, idx_ind_bioenergy['ind_bioenergy'], idx_ind_bioenergy['gas-bio']] \
                        + dm_ind_biomaterial.array[:, :, idx_ind_biomaterial['ind_biomaterial'],
                          idx_ind_biomaterial['gas-bio']] \
                        + dm_tra.array[:, :, idx_tra['tra_bioenergy'], idx_tra['gas']] \
                        + DM_bioenergy['electricity_production'].array[:, :, idx_elec['agr_bioenergy-capacity_fdk-req'],
                          idx_elec['biogases']] \
                        + DM_bioenergy['electricity_production'].array[:, :, idx_elec['agr_bioenergy-capacity_fdk-req'],
                          idx_elec['biogases-hf']]

    dm_biogas = DM_ind["natfibers"].copy()  # FIXME backup I do not know how to create a blanck dm with Country & Years
    dm_biogas.add(dm_bio_gas_demand, dim='Variables', col_label='agr_bioenergy-capacity_biogas-req', unit='TWh')
    dm_biogas.drop(dim='Variables', col_label=['ind_dem_natfibers'])  # FIXME to empty when upper comment fixed

    # Biogas per type [TWh] = Biogas feedstock requirements [GWh] * biogas technology share [%]
    idx_biogas = dm_biogas.idx
    idx_mix = DM_bioenergy['bgs-mix'].idx
    dm_biogas_mix = dm_biogas.array[:, :, idx_biogas['agr_bioenergy-capacity_biogas-req'], np.newaxis] * \
                    DM_bioenergy['bgs-mix'].array[:, :, idx_mix['agr_bioenergy-capacity_bgs-mix'], :]
    DM_bioenergy['bgs-mix'].add(dm_biogas_mix, dim='Variables', col_label='agr_bioenergy-capacity_bgs-tec', unit='TWh')

    # Digestor feedstock per type [TWh] = biogas demand for digestor [TWh] * biomass share for digestor [%]
    dm_digestor_demand = DM_bioenergy['bgs-mix'].filter({'Variables': ['agr_bioenergy-capacity_bgs-tec'],
                                                         'Categories1': ['digestor']})
    dm_digestor_demand = dm_digestor_demand.flatten()
    idx_demand_digestor = dm_digestor_demand.idx
    idx_mix_digestor = DM_bioenergy['digestor-mix'].idx
    dm_biogas_demand_digestor = dm_digestor_demand.array[:, :,
                                idx_demand_digestor['agr_bioenergy-capacity_bgs-tec_digestor'], np.newaxis] * \
                                DM_bioenergy['digestor-mix'].array[:, :,
                                idx_mix_digestor['agr_biomass-hierarchy_biomass-mix_digestor'], :]
    DM_bioenergy['digestor-mix'].add(dm_biogas_demand_digestor, dim='Variables',
                                     col_label='agr_bioenergy_biomass-demand_biogas', unit='TWh')

    # SOLID BIOFUEL ----------------------------------------------------------------------------------------------------
    # Solid biomass feedstock requirements [TWh] =
    # (bld + industry) solid bioenergy demand + solid biofuel feedstock requirements (hf and not)
    idx_bld = dm_bld.idx
    idx_ind_bioenergy = dm_ind_bioenergy.idx
    idx_elec = DM_bioenergy['electricity_production'].idx

    dm_solid_demand = dm_bld.array[:, :, idx_bld['bld_bioenergy'], idx_bld['solid-bio']] \
                      + dm_ind_bioenergy.array[:, :, idx_ind_bioenergy['ind_bioenergy'], idx_ind_bioenergy['solid-bio']] \
                      + DM_bioenergy['electricity_production'].array[:, :, idx_elec['agr_bioenergy-capacity_fdk-req'],
                        idx_elec['solid-biofuel']] \
                      + DM_bioenergy['electricity_production'].array[:, :, idx_elec['agr_bioenergy-capacity_fdk-req'],
                        idx_elec['solid-biofuel-hf']]

    dm_solid = DM_ind["natfibers"].copy()  # FIXME backup I do not know how to create a blanck dm with Country & Years
    dm_solid.add(dm_solid_demand, dim='Variables', col_label='agr_bioenergy-capacity_solid-biofuel-req', unit='TWh')
    dm_solid.drop(dim='Variables', col_label=['ind_dem_natfibers'])  # FIXME to empty when upper comment fixed

    # Solid feedstock per type [TWh] = solid demand for  biofuel [TWh] * biomass share solid [%]
    idx_demand_solid = dm_solid.idx
    idx_mix_solid = DM_bioenergy['solid-mix'].idx
    dm_solid_fdk = dm_solid.array[:, :, idx_demand_solid['agr_bioenergy-capacity_solid-biofuel-req'], np.newaxis] * \
                   DM_bioenergy['solid-mix'].array[:, :, idx_mix_solid['agr_biomass-hierarchy_biomass-mix_solid'], :]
    DM_bioenergy['solid-mix'].add(dm_solid_fdk, dim='Variables', col_label='agr_bioenergy_biomass-demand_solid',
                                  unit='TWh')

    # LIQUID BIOFUEL ----------------------------------------------------------------------------------------------------

    # Liquid biofuel per type [TWh] = liquid biofuel capacity [TWh] * share per technology [%]
    # Biodiesel
    dm_biodiesel = DM_bioenergy['liq'].filter({'Categories1': ['biodiesel']})
    dm_biodiesel = dm_biodiesel.flatten()
    idx_biodiesel_cap = dm_biodiesel.idx
    idx_biodiesel_tec = DM_bioenergy['liquid-biodiesel'].idx
    dm_biodiesel_temp = dm_biodiesel.array[:, :, idx_biodiesel_cap['agr_bioenergy-capacity_liq_biodiesel'],
                        np.newaxis] * \
                        DM_bioenergy['liquid-biodiesel'].array[:, :,
                        idx_biodiesel_tec['agr_biomass-hierarchy_bioenergy_liquid_biodiesel'], :]
    DM_bioenergy['liquid-biodiesel'].add(dm_biodiesel_temp, dim='Variables',
                                         col_label='agr_bioenergy-capacity_liq-bio-prod_biodiesel', unit='TWh')

    # Biogasoline
    dm_biogasoline = DM_bioenergy['liq'].filter({'Categories1': ['biogasoline']})
    dm_biogasoline = dm_biogasoline.flatten()
    idx_biogasoline_cap = dm_biogasoline.idx
    idx_biogasoline_tec = DM_bioenergy['liquid-biogasoline'].idx
    dm_biogasoline_temp = dm_biogasoline.array[:, :, idx_biogasoline_cap['agr_bioenergy-capacity_liq_biogasoline'],
                          np.newaxis] * \
                          DM_bioenergy['liquid-biogasoline'].array[:, :,
                          idx_biogasoline_tec['agr_biomass-hierarchy_bioenergy_liquid_biogasoline'], :]
    DM_bioenergy['liquid-biogasoline'].add(dm_biogasoline_temp, dim='Variables',
                                           col_label='agr_bioenergy-capacity_liq-bio-prod_biogasoline', unit='TWh')

    # Biojetkerosene
    dm_biojetkerosene = DM_bioenergy['liq'].filter({'Categories1': ['biojetkerosene']})
    dm_biojetkerosene = dm_biojetkerosene.flatten()
    idx_biojetkerosene_cap = dm_biojetkerosene.idx
    idx_biojetkerosene_tec = DM_bioenergy['liquid-biojetkerosene'].idx
    dm_biojetkerosene_temp = dm_biojetkerosene.array[:, :,
                             idx_biojetkerosene_cap['agr_bioenergy-capacity_liq_biojetkerosene'],
                             np.newaxis] * \
                             DM_bioenergy['liquid-biojetkerosene'].array[:, :,
                             idx_biojetkerosene_tec['agr_biomass-hierarchy_bioenergy_liquid_biojetkerosene'], :]
    DM_bioenergy['liquid-biojetkerosene'].add(dm_biojetkerosene_temp, dim='Variables',
                                              col_label='agr_bioenergy-capacity_liq-bio-prod_biojetkerosene',
                                              unit='TWh')

    # Liquid biofuel feedstock requirements [kcal] = Liquid biofuel per type [TWh] * share per technology [kcal/TWh]

    # Constant pre processing
    cdm_biodiesel = CDM_const['cdm_biodiesel']
    cdm_biogasoline = CDM_const['cdm_biogasoline']
    cdm_biojetkerosene = CDM_const['cdm_biojetkerosene']

    # Biodiesel
    idx_cdm = cdm_biodiesel.idx
    idx_bio = DM_bioenergy['liquid-biodiesel'].idx
    dm_calc = DM_bioenergy['liquid-biodiesel'].array[:, :, idx_bio['agr_bioenergy-capacity_liq-bio-prod_biodiesel'], :] \
              * cdm_biodiesel.array[idx_cdm['cp_liquid_tec_biodiesel'], :]
    DM_bioenergy['liquid-biodiesel'].add(dm_calc, dim='Variables',
                                         col_label='agr_bioenergy-capacity_liq-bio-prod_fdk-req_biodiesel',
                                         unit='kcal')

    # Biogasoline
    idx_cdm = cdm_biogasoline.idx
    idx_bio = DM_bioenergy['liquid-biogasoline'].idx
    dm_calc = DM_bioenergy['liquid-biogasoline'].array[:, :, idx_bio['agr_bioenergy-capacity_liq-bio-prod_biogasoline'],
              :] \
              * cdm_biogasoline.array[idx_cdm['cp_liquid_tec_biogasoline'], :]
    DM_bioenergy['liquid-biogasoline'].add(dm_calc, dim='Variables',
                                           col_label='agr_bioenergy-capacity_liq-bio-prod_fdk-req_biogasoline',
                                           unit='kcal')

    # Biojetkerosene
    idx_cdm = cdm_biojetkerosene.idx
    idx_bio = DM_bioenergy['liquid-biojetkerosene'].idx
    dm_calc = DM_bioenergy['liquid-biojetkerosene'].array[:, :,
              idx_bio['agr_bioenergy-capacity_liq-bio-prod_biojetkerosene'], :] \
              * cdm_biojetkerosene.array[idx_cdm['cp_liquid_tec_biojetkerosene'], :]
    DM_bioenergy['liquid-biojetkerosene'].add(dm_calc, dim='Variables',
                                              col_label='agr_bioenergy-capacity_liq-bio-prod_fdk-req_biojetkerosene',
                                              unit='kcal')

    # Total liquid biofuel feedstock req per feedstock type [kcal]
    # = liquid biofuel fdk req per fdk type (biodiesel + biogasoline + biojetkerosene)
    # Feedstock types : lgn => lignin, eth => ethanol, oil

    # Filter the dms
    dm_biofuel_fdk = DM_bioenergy['liquid-biodiesel'].filter(
        {'Variables': ['agr_bioenergy-capacity_liq-bio-prod_fdk-req_biodiesel']})
    dm_biogasoline = DM_bioenergy['liquid-biogasoline'].filter(
        {'Variables': ['agr_bioenergy-capacity_liq-bio-prod_fdk-req_biogasoline']})
    dm_biojetkerosene = DM_bioenergy['liquid-biojetkerosene'].filter(
        {'Variables': ['agr_bioenergy-capacity_liq-bio-prod_fdk-req_biojetkerosene']})
    # Add dummy categories (to have Categories1 = btl, est, hvo, ezm, fer for all)
    dm_biofuel_fdk.add(0.0, dummy=True, col_label='ezm', dim='Categories1', unit='kcal')
    dm_biofuel_fdk.add(0.0, dummy=True, col_label='fer', dim='Categories1', unit='kcal')
    dm_biogasoline.add(0.0, dummy=True, col_label='btl', dim='Categories1', unit='kcal')
    dm_biogasoline.add(0.0, dummy=True, col_label='est', dim='Categories1', unit='kcal')
    dm_biogasoline.add(0.0, dummy=True, col_label='hvo', dim='Categories1', unit='kcal')
    dm_biojetkerosene.add(0.0, dummy=True, col_label='est', dim='Categories1', unit='kcal')
    dm_biojetkerosene.add(0.0, dummy=True, col_label='ezm', dim='Categories1', unit='kcal')
    dm_biojetkerosene.add(0.0, dummy=True, col_label='fer', dim='Categories1', unit='kcal')

    # Sort the dms
    dm_biofuel_fdk.sort(dim='Categories1')
    dm_biogasoline.sort(dim='Categories1')
    dm_biojetkerosene.sort(dim='Categories1')
    # Append the dms together
    dm_biofuel_fdk.append(dm_biogasoline, dim='Variables')
    dm_biofuel_fdk.append(dm_biojetkerosene, dim='Variables')
    # Create dms for each feedstock (eth, lgn & oil)
    # dm_eth = cdm_const.filter_w_regex(({'Variables': 'eth'}))

    # Total feedstock requirements = sum fdk for biogasoline + biodiesel + biojetkerosene
    dm_biofuel_fdk.operation('agr_bioenergy-capacity_liq-bio-prod_fdk-req_biodiesel', '+',
                             'agr_bioenergy-capacity_liq-bio-prod_fdk-req_biogasoline',
                             out_col='agr_bioenergy_biomass-demand_liquid_biodiesel_biogasoline', unit='kcal')
    dm_biofuel_fdk.operation('agr_bioenergy_biomass-demand_liquid_biodiesel_biogasoline', '+',
                             'agr_bioenergy-capacity_liq-bio-prod_fdk-req_biojetkerosene',
                             out_col='agr_bioenergy_biomass-demand_liquid', unit='kcal')
    dm_biofuel_fdk = dm_biofuel_fdk.filter({'Variables': ['agr_bioenergy_biomass-demand_liquid']})

    # Sum using group by for each feedstock (fer => eth, btl & ezm => lgn, hvo & est =>oil)
    dm_biofuel_fdk.groupby({'eth': '.*fer'}, dim='Categories1', regex=True,
                           inplace=True)
    dm_biofuel_fdk.groupby({'lgn': '.*btl|.*ezm'}, dim='Categories1', regex=True,
                           inplace=True)
    dm_biofuel_fdk.groupby({'oil': '.*hvo|.*est'}, dim='Categories1', regex=True,
                           inplace=True)
    dm_biofuel_fdk = dm_biofuel_fdk.flatten()

    # Liquid biofuel demand per type [kcal] = Total liquid biofuel feedstock req per fdk type [kcal]
    # * biomass hierarchy per type [%]

    # Filtering liquid-mix per fdk type
    dm_eth = DM_bioenergy['liquid-mix'].filter_w_regex(({'Categories1': 'eth'}))
    dm_lgn = DM_bioenergy['liquid-mix'].filter_w_regex(({'Categories1': 'lgn'}))
    dm_oil = DM_bioenergy['liquid-mix'].filter_w_regex(({'Categories1': 'oil'}))

    # computation for each fdk type using a tree split method
    # eth
    idx_eth_mix = dm_eth.idx
    idx_eth_demand = dm_biofuel_fdk.idx
    array_temp = dm_biofuel_fdk.array[:, :, idx_eth_demand['agr_bioenergy_biomass-demand_liquid_eth'], np.newaxis] * \
                 dm_eth.array[:, :, idx_eth_mix['agr_biomass-hierarchy_biomass-mix_liquid'], :]
    dm_eth.add(array_temp, dim='Variables', col_label='agr_bioenergy_biomass-demand_liquid_eth',
               unit='kcal')

    # lgn
    idx_lgn_mix = dm_lgn.idx
    idx_lgn_demand = dm_biofuel_fdk.idx
    array_temp = dm_biofuel_fdk.array[:, :, idx_lgn_demand['agr_bioenergy_biomass-demand_liquid_lgn'], np.newaxis] * \
                 dm_lgn.array[:, :, idx_lgn_mix['agr_biomass-hierarchy_biomass-mix_liquid'], :]
    dm_lgn.add(array_temp, dim='Variables', col_label='agr_bioenergy_biomass-demand_liquid_lgn',
               unit='kcal')

    # oil
    idx_oil_mix = dm_oil.idx
    idx_oil_demand = dm_biofuel_fdk.idx
    array_temp = dm_biofuel_fdk.array[:, :, idx_oil_demand['agr_bioenergy_biomass-demand_liquid_oil'], np.newaxis] * \
                 dm_oil.array[:, :, idx_oil_mix['agr_biomass-hierarchy_biomass-mix_liquid'], :]
    dm_oil.add(array_temp, dim='Variables', col_label='agr_bioenergy_biomass-demand_liquid_oil',
               unit='kcal')

    return DM_bioenergy, dm_oil, dm_lgn, dm_eth, dm_biofuel_fdk


# CalculationLeaf LIVESTOCK MANURE MANAGEMENT & GHG EMISSIONS ----------------------------------------------------------
def livestock_manure_workflow(DM_manure, DM_livestock, dm_liv_pop, cdm_const, years_setting):
    # Pre processing livestock population
    dm_liv_pop = dm_liv_pop.filter({'Variables': ['agr_liv_population']})
    DM_manure['liv_n-stock'].append(dm_liv_pop, dim='Variables')
    DM_manure['enteric_emission'].append(dm_liv_pop, dim='Variables')
    DM_manure['ef_liv_CH4_treated'].append(dm_liv_pop, dim='Variables')

    # N2O
    # Manure production [tN] = livestock population [lsu] * Manure yield [t/lsu]
    DM_manure['liv_n-stock'].operation('fxa_liv_manure_n-stock', '*', 'agr_liv_population',
                                       out_col='agr_liv_n-stock', unit='t')

    # Manure management practices [MtN] = Manure production [MtN] * Share of management practices [%]
    idx_nstock = DM_manure['liv_n-stock'].idx
    idx_split = DM_manure['manure'].idx
    dm_temp = DM_manure['liv_n-stock'].array[:, :, idx_nstock['agr_liv_n-stock'], :, np.newaxis] * \
              DM_manure['manure'].array[:, :, idx_split['agr_climate-smart-livestock_manure'], :, :]
    DM_manure['ef_liv_N2O'].add(dm_temp, dim='Variables', col_label='agr_liv_n-stock_split',
                                unit='t')

    # Manure emission [MtN2O] = Manure management practices [MtN] * emission factors per practices [MtN2O/Mt]
    DM_manure['ef_liv_N2O'].operation('agr_liv_n-stock_split', '*', 'fxa_ef_liv_N2O-emission_ef',
                                      out_col='agr_liv_N2O-emission_raw', unit='t')

    dm_temp = DM_manure['ef_liv_N2O'].copy()

    # Calibration N2O
    dm_liv_N2O = DM_manure['ef_liv_N2O'].filter({'Variables': ['agr_liv_N2O-emission_raw']})
    dm_cal_liv_N2O = DM_manure['cal_liv_N2O']
    dm_cal_liv_N2O.switch_categories_order(cat1='Categories2', cat2='Categories1')  # Switch categories
    dm_cal_liv_N2O.change_unit('cal_agr_liv_N2O-emission', factor=1e3, old_unit='kt', new_unit='t')
    dm_cal_rates_liv_N2O = calibration_rates(dm_liv_N2O, dm_cal_liv_N2O, calibration_start_year=1990,
                                             calibration_end_year=2023, years_setting=years_setting)
    dm_liv_N2O.append(dm_cal_rates_liv_N2O, dim='Variables')
    dm_liv_N2O.operation('agr_liv_N2O-emission_raw', '*', 'cal_rate', dim='Variables', out_col='agr_liv_N2O-emission',
                         unit='t')
    df_cal_rates_liv_N2O = dm_to_database(dm_cal_rates_liv_N2O, 'none', 'agriculture', level=0)

    # CH4
    # Enteric emission [tCH4] = livestock population [lsu] * enteric emission factor [tCH4/lsu]
    DM_manure['enteric_emission'].operation('agr_climate-smart-livestock_enteric', '*', 'agr_liv_population',
                                            dim="Variables", out_col='agr_liv_CH4-emission_raw', unit='t')

    # Manure emission [tCH4] = livestock population [lsu] * emission factors treated manure [tCH4/lsu]
    DM_manure['ef_liv_CH4_treated'].operation('fxa_ef_liv_CH4-emission_treated', '*', 'agr_liv_population',
                                              dim="Variables", out_col='agr_liv_CH4-emission_raw', unit='t')

    # Processing for calibration (putting enteric and treated CH4 emission in the same dm)
    # Treated
    dm_CH4 = DM_manure['ef_liv_CH4_treated'].filter({'Variables': ['agr_liv_CH4-emission_raw']})
    dm_CH4.rename_col_regex(str1="meat", str2="treated_meat", dim="Categories1")
    dm_CH4.rename_col_regex(str1="abp", str2="treated_abp", dim="Categories1")
    dm_CH4.deepen()
    dm_CH4.switch_categories_order(cat1='Categories2', cat2='Categories1')
    # Enteric
    dm_CH4_enteric = DM_manure['enteric_emission'].filter({'Variables': ['agr_liv_CH4-emission_raw']})
    dm_CH4_enteric.rename_col_regex(str1="meat", str2="enteric_meat", dim="Categories1")
    dm_CH4_enteric.rename_col_regex(str1="abp", str2="enteric_abp", dim="Categories1")
    dm_CH4_enteric.deepen()
    dm_CH4_enteric.switch_categories_order(cat1='Categories2', cat2='Categories1')
    # Appending
    dm_CH4.append(dm_CH4_enteric, dim='Categories2')

    # Calibration CH4
    dm_cal_liv_CH4 = DM_manure['cal_liv_CH4']
    dm_cal_liv_CH4.switch_categories_order(cat1='Categories2', cat2='Categories1')  # Switch categories
    dm_cal_liv_CH4.change_unit('cal_agr_liv_CH4-emission', factor=1e3, old_unit='kt', new_unit='t')
    dm_cal_rates_liv_CH4 = calibration_rates(dm_CH4, dm_cal_liv_CH4, calibration_start_year=1990,
                                             calibration_end_year=2023, years_setting=years_setting)
    dm_CH4.append(dm_cal_rates_liv_CH4, dim='Variables')
    dm_CH4.operation('agr_liv_CH4-emission_raw', '*', 'cal_rate', dim='Variables', out_col='agr_liv_CH4-emission',
                     unit='t')
    df_cal_rates_liv_CH4 = dm_to_database(dm_cal_rates_liv_CH4, 'none', 'agriculture', level=0)

    return dm_liv_N2O, dm_CH4, df_cal_rates_liv_N2O, df_cal_rates_liv_CH4, DM_manure


# CalculationLeaf FEED -------------------------------------------------------------------------------------------------
def feed_workflow(DM_feed, dm_liv_prod, dm_bev_ibp_cereal_feed, CDM_const, years_setting):
    # FEED REQUIREMENTS
    # Filter protein conversion efficiency constant
    cdm_cp_efficiency = CDM_const['cdm_cp_efficiency']

    # Pre processing domestic ASF prod accounting for waste [kcal]
    dm_feed_req = dm_liv_prod.filter({'Variables': ['agr_domestic_production_liv_afw']})

    # Feed req per livestock type [kcal] = domestic ASF prod accounting for waste [kcal] / protein conversion efficiency [%]
    idx_cdm = cdm_cp_efficiency.idx
    idx_feed = dm_feed_req.idx
    dm_temp = dm_feed_req.array[:, :, idx_feed['agr_domestic_production_liv_afw'], :] \
              / cdm_cp_efficiency.array[idx_cdm['cp_efficiency_liv'], :]
    dm_feed_req.add(dm_temp, dim='Variables', col_label='agr_feed-requierement', unit='kcal')

    # Total feed req [kcal] = sum(Feed req per livestock type [kcal])
    dm_feed_req_total = dm_feed_req.filter({'Variables': ['agr_feed-requierement']})
    dm_feed_req_total.groupby({'total': '.*'}, dim='Categories1', regex=True, inplace=True)
    dm_feed_req_total = dm_feed_req_total.flatten()

    # ALTERNATIVE PROTEIN SOURCE (APS) FOR LIVESTOCK FEED
    # APS [kcal] = Feed req per livestock type [kcal] * APS share per type [%]
    idx_aps = DM_feed['alt-protein'].idx
    idx_feed = dm_feed_req.idx
    dm_temp = dm_feed_req.array[:, :, idx_feed['agr_feed-requierement'], :, np.newaxis] \
              * DM_feed['alt-protein'].array[:, :, idx_aps['agr_alt-protein'], :, :]
    DM_feed['alt-protein'].add(dm_temp, dim='Variables', col_label='agr_feed_aps', unit='kcal')

    # Algae meals [kcal] = sum algae feed req
    dm_aps = DM_feed['alt-protein'].filter({'Variables': ['agr_feed_aps'], 'Categories2': ['algae']})
    dm_aps = dm_aps.flatten()
    dm_aps.groupby({'algae': '.*'}, dim='Categories1', regex=True, inplace=True)

    # Insect meals [kcal] = sum insect feed req
    dm_insect = DM_feed['alt-protein'].filter({'Variables': ['agr_feed_aps'], 'Categories2': ['insect']})
    dm_insect = dm_insect.flatten()
    dm_insect.groupby({'insect': '.*'}, dim='Categories1', regex=True, inplace=True)
    dm_aps.append(dm_insect, dim='Categories1')

    # Filter APS byproduct ration constant
    cdm_aps_ibp = CDM_const['cdm_aps_ibp']

    # APS byproducts [kcal] = APS production [kcal] * byproduct ratio [%]
    idx_cdm = cdm_aps_ibp.idx
    idx_aps = dm_aps.idx
    dm_temp = dm_aps.array[:, :, idx_aps['agr_feed_aps'], np.newaxis, :, np.newaxis] \
              * cdm_aps_ibp.array[idx_cdm['cp_ibp_aps'], np.newaxis, :, :]
    # dm_aps.add(dm_temp, dim='Variables', col_label='agr_aps', unit='kcal') FIXME find correct dm to add to or create one

    # Create datamatrix by depth
    col_labels = {
        'Country': dm_aps.col_labels['Country'].copy(),
        'Years': dm_aps.col_labels['Years'].copy(),
        'Variables': ['agr_aps'],
        'Categories1': cdm_aps_ibp.col_labels['Categories1'].copy(),
        'Categories2': cdm_aps_ibp.col_labels['Categories2'].copy()
    }
    dm_aps_ibp = DataMatrix(col_labels, units={'agr_aps': 'kcal'})
    dm_aps_ibp.array = dm_temp

    # FEED RATION
    # Alternative feed ration [kcal] = sum (cereals from bev for feed, APS, grass) FIXME check what is supposed to be considered
    idx_ibp_bev = dm_bev_ibp_cereal_feed.idx
    dm_alt_feed = dm_bev_ibp_cereal_feed.copy()
    # dm_alt_feed.rename_col('agr_use_bev_ibp_cereal_feed', 'agr_feed-diet-switch', dim='Variables') FIXME find the issue because this line does not work, probably because of previous groupby in ALC BEV
    dm_feed_req_total.append(dm_alt_feed, dim='Variables')

    # Total feed demand [kcal] = Total feed req [kcal] - Alternative feed ration [kcal] FIXME change 1st component name
    dm_feed_req_total.operation('agr_feed-requierement_total', '-', 'agr_use_bev_ibp_cereal_feed',
                                out_col='agr_feed-demand', unit='kcal')

    # Feed demand by type [kcal] = Crop based feed demand by type [kcal] * Share of feed per type [%]
    idx_feed = dm_feed_req_total.idx
    idx_ration = DM_feed['ration'].idx
    dm_temp = dm_feed_req_total.array[:, :, idx_feed['agr_feed-demand'], np.newaxis] \
              * DM_feed['ration'].array[:, :, idx_ration['agr_climate-smart-livestock_ration'], :]
    DM_feed['ration'].add(dm_temp, dim='Variables', col_label='agr_demand_feed_raw', unit='kcal')

    # Calibration Feed demand
    dm_feed_demand = DM_feed['ration'].filter({'Variables': ['agr_demand_feed_raw']})
    dm_cal_feed = DM_feed['cal_agr_demand_feed']
    dm_cal_rates_feed = calibration_rates(dm_feed_demand, dm_cal_feed, calibration_start_year=1990,
                                          calibration_end_year=2023,
                                          years_setting=years_setting)
    DM_feed['ration'].append(dm_cal_rates_feed, dim='Variables')
    DM_feed['ration'].operation('agr_demand_feed_raw', '*', 'cal_rate', dim='Variables', out_col='agr_demand_feed',
                                unit='kcal')
    df_cal_rates_feed = dm_to_database(dm_cal_rates_feed, 'none', 'agriculture',
                                       level=0)  # Exporting calibration rates to check at the end
    df_cal_feed = dm_to_database(dm_cal_feed, 'none', 'agriculture',
                                 level=0)  # Exporting calibration rates to check at the end
    df_feed_demand = dm_to_database(dm_feed_demand, 'none', 'agriculture',
                                    level=0)  # Exporting calibration rates to check at the end

    return DM_feed, dm_aps_ibp, dm_feed_req, dm_aps, dm_feed_demand, df_cal_rates_feed


# CalculationLeaf BIOMASS USE ALLOCATION ---------------------------------------------------------------------------
def biomass_allocation_workflow(dm_aps_ibp, dm_oil):
    # Sum oil substitutes [kcal] = algae fdk voil + insect fdk voil (uco+ afat not considered)
    dm_aps_ibp_oil = dm_aps_ibp.filter({'Categories2': ['fdk-voil']})
    dm_aps_ibp_oil = dm_aps_ibp_oil.flatten()
    dm_aps_ibp_oil.groupby({'oil-voil': '.*'}, dim='Categories1', regex=True,
                           inplace=True)
    dm_aps_ibp_oil.rename_col('agr_aps', 'agr_bioenergy_fdk-aby', dim='Variables')

    # Feedstock for biogasoline from bev [kcal] = sum (biomass from bev for biogasoline [kcal])

    # Oilcrop demand for biofuels [kcal] = Liquid biofuel demand oil voil [kcal] - Sum oil substitutes [kcal]
    dm_voil = dm_oil.filter({'Variables': ['agr_bioenergy_biomass-demand_liquid_oil'], 'Categories1': ['oil-voil']})
    dm_voil.append(dm_aps_ibp_oil, dim='Variables')
    dm_voil.operation('agr_bioenergy_biomass-demand_liquid_oil', '-', 'agr_bioenergy_fdk-aby',
                      out_col='agr_bioenergy_biomass-demand_liquid', unit='kcal')

    # For TPE
    dm_voil_tpe = dm_voil.filter({'Variables': ['agr_bioenergy_biomass-demand_liquid']})

    # Feedstock for biogasoline [kcal] =
    # Liquid ethanol demand from cereal [kcal] - Feedstock for biogasoline from bev [kcal]

    return dm_voil, dm_aps_ibp_oil, dm_voil_tpe


# CalculationLeaf CROP PRODUCTION ----------------------------------------------------------------------------------
def crop_workflow(DM_crop, DM_feed, DM_bioenergy, dm_voil, dm_lfs, dm_lfs_pro, dm_lgn, dm_aps_ibp, CDM_const, dm_oil,
                  dm_bev_dom_prod, years_setting):
    # DOMESTIC PRODUCTION ACCOUNTING FOR LOSSES ------------------------------------------------------------------------

    # ( Domestic production processed voil [kcal])
    # PROCESSED FEED ---------------------------------------------------------------------------------------------------

    # Constant pre-processing
    cdm_feed_yield = CDM_const['cdm_feed_yield']
    cdm_food_yield = CDM_const['cdm_food_yield']

    # Processed Feed pre-processing
    dm_feed_processed = DM_feed['ration'].filter(
        {'Variables': ['agr_demand_feed'],
         'Categories1': ['crop-processed-cake', 'crop-processed-molasse', 'crop-processed-sugar',
                         'crop-processed-voil']})
    dm_feed_processed.rename_col('crop-processed-cake', 'cake-to-oilcrop', dim='Categories1')
    dm_feed_processed.rename_col('crop-processed-molasse', 'molasse-to-sugarcrop', dim='Categories1')
    dm_feed_processed.rename_col('crop-processed-sugar', 'sugar-to-sugarcrop', dim='Categories1')
    dm_feed_processed.rename_col('crop-processed-voil', 'voil-to-oilcrop', dim='Categories1')

    # Processed Feed - Accounting for SSR
    # Domestic production [kcal] = Processed Feed-demand [kcal] * net import [%]
    dm_ssr_feed_pro = DM_crop['food-net-import-pro'].filter(
        {'Variables': ['agr_food-net-import'], 'Categories1': ['pro-crop-processed-cake', 'pro-crop-processed-molasse',
                                                               'pro-crop-processed-sugar',
                                                               'pro-crop-processed-voil']}).copy()
    dm_ssr_feed_pro.rename_col('pro-crop-processed-cake', 'cake-to-oilcrop', dim='Categories1')
    dm_ssr_feed_pro.rename_col('pro-crop-processed-molasse', 'molasse-to-sugarcrop', dim='Categories1')
    dm_ssr_feed_pro.rename_col('pro-crop-processed-sugar', 'sugar-to-sugarcrop', dim='Categories1')
    dm_ssr_feed_pro.rename_col('pro-crop-processed-voil', 'voil-to-oilcrop', dim='Categories1')
    dm_feed_processed.append(dm_ssr_feed_pro, dim='Variables')
    dm_feed_processed.operation('agr_demand_feed', '*', 'agr_food-net-import',
                                out_col='agr_domestic-production_feed_pro',
                                unit='kcal')

    # Processed Feed crop dom prod [kcal] = processed crops [kcal] / processing yield [%]
    idx_cdm = cdm_feed_yield.idx
    idx_feed = dm_feed_processed.idx
    dm_temp = dm_feed_processed.array[:, :, idx_feed['agr_domestic-production_feed_pro'], :] \
              / cdm_feed_yield.array[idx_cdm['cp_ibp_processed'], :]
    dm_feed_processed.add(dm_temp, dim='Variables', col_label='agr_demand_feed_processed', unit='kcal')
    dm_feed_processed.drop(dim='Variables', col_label=['agr_demand_feed'])
    # Summing by crop category (oilcrop and sugarcrop)
    dm_feed_processed.groupby({'crop-oilcrop': '.*-to-oilcrop', 'crop-sugarcrop': '.*-to-sugarcrop'}, dim='Categories1',
                              regex=True,
                              inplace=True)
    # Adding dummy columns filled with nan for total feed demand calculations
    dm_feed_processed.add(0.0, dummy=True, col_label='crop-cereal', dim='Categories1', unit='kcal')
    dm_feed_processed.add(0.0, dummy=True, col_label='crop-pulse', dim='Categories1', unit='kcal')
    dm_feed_processed.add(0.0, dummy=True, col_label='crop-fruit', dim='Categories1', unit='kcal')
    dm_feed_processed.add(0.0, dummy=True, col_label='crop-veg', dim='Categories1', unit='kcal')
    dm_feed_processed.add(0.0, dummy=True, col_label='crop-starch', dim='Categories1', unit='kcal')
    dm_feed_processed.add(0.0, dummy=True, col_label='crop-rice', dim='Categories1', unit='kcal')
    # Filling the dummy columns with zeros and sorting alphabetically
    dm_feed_processed.sort(dim='Categories1')
    # dm_feed_processed = np.nan_to_num(dm_feed_processed.array)

    # Pre processing total feed demand per category (with dummy categories when necessary)
    dm_crop_feed_demand = DM_feed['ration'].filter_w_regex(
        {'Variables': 'agr_demand_feed', 'Categories1': 'crop-'})
    # Dropping processed crops feed demand
    dm_crop_feed_demand.drop(dim='Categories1', col_label=['crop-processed-cake', 'crop-processed-molasse',
                                                           'crop-processed-sugar', 'crop-processed-voil'])
    # Fill NaN with 0.0
    array_temp = dm_crop_feed_demand.array[:, :, :, :]
    array_temp = np.nan_to_num(array_temp, nan=0)
    dm_crop_feed_demand.array[:, :, :, :] = array_temp

    # Accounting for processed feed demand : Adding the columns for sugarcrops and oilcrops from previous calculation
    # Appending with dm_feed_processed
    dm_crop_feed_demand.append(dm_feed_processed, dim='Variables')
    # Summing
    dm_crop_feed_demand.operation('agr_demand_feed_processed', '+', 'agr_demand_feed', out_col='agr_demand_feed_temp',
                                  unit='kcal')
    dm_crop_feed_demand = dm_crop_feed_demand.filter({'Variables': ['agr_demand_feed_temp']})

    # Adding dummy categories
    dm_crop_feed_demand.add(0.0, dummy=True, col_label='crop-lgn-energycrop', dim='Categories1', unit='kcal')
    dm_crop_feed_demand.add(0.0, dummy=True, col_label='crop-algae', dim='Categories1', unit='kcal')
    dm_crop_feed_demand.add(0.0, dummy=True, col_label='crop-insect', dim='Categories1', unit='kcal')

    # PROCESSED FOOD ---------------------------------------------------------------------------------------------------

    # Processed food - Accounting for SSR
    # Domestic production [kcal] = Processed Food-demand [kcal] * net import [%]
    dm_food_processed = dm_lfs_pro.filter(
        {'Variables': ['agr_demand'], 'Categories1': ['pro-crop-processed-sweet', 'pro-crop-processed-sugar']})
    dm_ssr_food_pro = DM_crop['food-net-import-pro'].filter(
        {'Variables': ['agr_food-net-import'],
         'Categories1': ['pro-crop-processed-sweet', 'pro-crop-processed-sugar']}).copy()
    dm_food_processed.append(dm_ssr_food_pro, dim='Variables')
    dm_food_processed.operation('agr_demand', '*', 'agr_food-net-import', out_col='agr_domestic-production_food_pro',
                                unit='kcal')

    # Processed Food crop demand [kcal] = processed crops [kcal] / processing yield [%] (only for sweets & processed sugar)
    # sum processed sugar in one variable : sweets : sweets + processed sugar
    dm_food_processed.groupby({'pro-crop-processed-sweet': '.*'}, dim='Categories1', regex=True, inplace=True)
    dm_food_processed.rename_col('pro-crop-processed-sweet', 'crop-sugarcrop', dim='Categories1')
    dm_food_processed.rename_col('agr_domestic-production_food_pro', 'agr_domestic-production_food_pro_temp',
                                 dim='Variables')
    idx_cdm = cdm_food_yield.idx
    idx_food = dm_food_processed.idx
    dm_temp = dm_food_processed.array[:, :, idx_food['agr_domestic-production_food_pro_temp'], :] \
              / cdm_food_yield.array[idx_cdm['cp_ibp_processed'], :]
    dm_food_processed.add(dm_temp, dim='Variables', col_label='agr_domestic-production_food', unit='kcal')

    # NON-PROCESSED FOOD ---------------------------------------------------------------------------------------------------

    # Pre processing total food demand per category (with dummy categories when necessary)
    # Categories x8 : cereals, oilcrop, pulse, fruit, veg, starch, sugarcrop, rice (+ maybe lgn, alage and insect)
    dm_crop_demand = dm_lfs.filter({'Variables': ['agr_demand']})
    dm_crop_demand = dm_crop_demand.filter_w_regex({'Variables': 'agr_demand', 'Categories1': 'crop-|rice'})
    # Renaming categories
    dm_crop_demand.rename_col_regex(str1="crop-", str2="", dim="Categories1")

    # Non-Processed food - Accounting for SSR
    # Domestic production [kcal] = Food-demand [kcal] * net import [%]
    dm_ssr_food = DM_crop['crop'].filter(
        {'Variables': ['agr_food-net-import'],
         'Categories1': ['cereal', 'fruit', 'oilcrop', 'pulse', 'starch', 'veg', 'rice']}).copy()
    dm_crop_demand.append(dm_ssr_food, dim='Variables')
    dm_crop_demand.operation('agr_demand', '*', 'agr_food-net-import', out_col='agr_domestic-production_food',
                             unit='kcal')

    # Accounting for processed food demand :Adding the column for sugarcrops (processed sweets) from previous calculation
    dm_sugarcrop = dm_food_processed.filter({'Variables': ['agr_domestic-production_food']})
    dm_crop_demand = dm_crop_demand.filter({'Variables': ['agr_domestic-production_food']})
    dm_crop_demand.append(dm_sugarcrop, dim='Categories1')
    # Sorting alphabetically and renaming col
    dm_crop_demand.sort(dim='Categories1')
    # dm_crop_demand.rename_col('agr_demand', 'agr_demand_food', dim='Variables')
    dm_crop_demand.rename_col('crop-sugarcrop', 'sugarcrop', dim='Categories1')
    # Adding dummy categories
    dm_crop_demand.add(0.0, dummy=True, col_label='lgn-energycrop', dim='Categories1', unit='kcal')
    dm_crop_demand.add(0.0, dummy=True, col_label='algae', dim='Categories1', unit='kcal')
    dm_crop_demand.add(0.0, dummy=True, col_label='insect', dim='Categories1', unit='kcal')

    # PROCESSED BEV ----------------------------------------------------------------------------------------------------

    # Here the SSR is already accounted for, but not the losses
    # Adding dummy categories
    dm_bev_dom_prod.add(0.0, dummy=True, col_label='oilcrop', dim='Categories1', unit='kcal')
    dm_bev_dom_prod.add(0.0, dummy=True, col_label='pulse', dim='Categories1', unit='kcal')
    dm_bev_dom_prod.add(0.0, dummy=True, col_label='veg', dim='Categories1', unit='kcal')
    dm_bev_dom_prod.add(0.0, dummy=True, col_label='starch', dim='Categories1', unit='kcal')
    dm_bev_dom_prod.add(0.0, dummy=True, col_label='sugarcrop', dim='Categories1', unit='kcal')
    dm_bev_dom_prod.add(0.0, dummy=True, col_label='rice', dim='Categories1', unit='kcal')
    dm_bev_dom_prod.add(0.0, dummy=True, col_label='algae', dim='Categories1', unit='kcal')
    dm_bev_dom_prod.add(0.0, dummy=True, col_label='insect', dim='Categories1', unit='kcal')
    dm_bev_dom_prod.add(0.0, dummy=True, col_label='lgn-energycrop', dim='Categories1', unit='kcal')
    dm_bev_dom_prod.sort(dim='Categories1')

    # PROCESSED BIOENERGY ----------------------------------------------------------------------------------------------

    # From BIOENERGY (oilcrop from voil + lgn from solid & liquid) (not accounted for in KNIME probably due to regex error)
    # Pre processing
    dm_oilcrop_voil = dm_oil.filter(
        {'Variables': ['agr_bioenergy_biomass-demand_liquid_oil'], 'Categories1': ['oil-voil']})
    # Accounting for SSR
    # Processed bioenergy - Accounting for SSR
    # Domestic production [kcal] = Processed Food-demand [kcal] * net import [%]
    dm_ssr_bioe_pro = DM_crop['food-net-import-pro'].filter(
        {'Variables': ['agr_food-net-import'], 'Categories1': ['pro-crop-processed-voil']}).copy()
    dm_ssr_bioe_pro.rename_col('pro-crop-processed-voil', 'oil-voil', dim='Categories1')
    dm_oilcrop_voil.append(dm_ssr_bioe_pro, dim='Variables')
    dm_oilcrop_voil.operation('agr_bioenergy_biomass-demand_liquid_oil', '*', 'agr_food-net-import',
                              out_col='agr_domestic-production_bioe_pro',
                              unit='kcal')

    # Accounting for processing yield
    idx_voil = dm_oilcrop_voil.idx
    idx_cdm = cdm_feed_yield.idx
    array_temp = dm_oilcrop_voil.array[:, :, idx_voil['agr_domestic-production_bioe_pro'], :] \
                 / cdm_feed_yield.array[idx_cdm['cp_ibp_processed'], idx_cdm['voil-to-oilcrop']]
    dm_oilcrop_voil.add(array_temp, dim='Variables', col_label='agr_domestic-production_bioe', unit='kcal')
    # Filtering and renaming for name matching
    dm_voil = dm_oilcrop_voil.filter({'Variables': ['agr_domestic-production_bioe']})
    dm_voil.rename_col('oil-voil', 'oilcrop', dim='Categories1')
    # Creating dummy categories
    dm_voil.add(0.0, dummy=True, col_label='cereal', dim='Categories1', unit='kcal')
    dm_voil.add(0.0, dummy=True, col_label='pulse', dim='Categories1', unit='kcal')
    dm_voil.add(0.0, dummy=True, col_label='fruit', dim='Categories1', unit='kcal')
    dm_voil.add(0.0, dummy=True, col_label='veg', dim='Categories1', unit='kcal')
    dm_voil.add(0.0, dummy=True, col_label='starch', dim='Categories1', unit='kcal')
    dm_voil.add(0.0, dummy=True, col_label='sugarcrop', dim='Categories1', unit='kcal')
    dm_voil.add(0.0, dummy=True, col_label='rice', dim='Categories1', unit='kcal')
    dm_voil.add(0.0, dummy=True, col_label='algae', dim='Categories1', unit='kcal')
    dm_voil.add(0.0, dummy=True, col_label='insect', dim='Categories1', unit='kcal')
    dm_voil.sort(dim='Categories1')

    # LGN
    # lgn from liquid biofuel FIXME SSR
    dm_lgn_energycrop = dm_lgn.filter(
        {'Variables': ['agr_bioenergy_biomass-demand_liquid_lgn'],
         'Categories1': ['lgn-btl-energycrop', 'lgn-ezm-energycrop']})
    dm_lgn_energycrop.groupby({'lgn-energycrop': '.*'}, dim='Categories1', regex=True, inplace=True)
    dm_lgn_energycrop.rename_col('agr_bioenergy_biomass-demand_liquid_lgn', 'agr_domestic-production_bioe',
                                 dim='Variables')
    # lgn from biogas FIXME not considered because not correct unit
    # dm_lgn_energycrop_biogas = DM_bioenergy['digestor-mix'].filter(
    #    {'Variables': ['agr_bioenergy_biomass-demand_biogas'],
    #     'Categories1': ['energycrop']})
    # summing total lgn
    # dm_lgn_energycrop.append(dm_lgn_energycrop_biogas, dim='Variables')

    # ALGAE & INSECT
    dm_aps = dm_aps_ibp.filter({'Variables': ['agr_aps'], 'Categories2': ['crop']})
    dm_aps = dm_aps.flatten()
    dm_aps.rename_col('algae_crop', 'algae', dim='Categories1')
    dm_aps.rename_col('insect_crop', 'insect', dim='Categories1')
    # Creating dummy categories
    dm_aps.add(0.0, dummy=True, col_label='cereal', dim='Categories1', unit='kcal')
    dm_aps.add(0.0, dummy=True, col_label='pulse', dim='Categories1', unit='kcal')
    dm_aps.add(0.0, dummy=True, col_label='fruit', dim='Categories1', unit='kcal')
    dm_aps.add(0.0, dummy=True, col_label='veg', dim='Categories1', unit='kcal')
    dm_aps.add(0.0, dummy=True, col_label='starch', dim='Categories1', unit='kcal')
    dm_aps.add(0.0, dummy=True, col_label='sugarcrop', dim='Categories1', unit='kcal')
    dm_aps.add(0.0, dummy=True, col_label='oilcrop', dim='Categories1', unit='kcal')
    dm_aps.add(0.0, dummy=True, col_label='rice', dim='Categories1', unit='kcal')
    dm_aps.add(0.0, dummy=True, col_label='lgn-energycrop', dim='Categories1', unit='kcal')
    dm_aps.sort(dim='Categories1')

    # FOOD + FEED + BEV + NON-FOOD ---------------------------------------------------------------------------------------------------

    # Appending the dms
    dm_voil.add(dm_lgn_energycrop.array, col_label='lgn-energycrop', dim='Categories1')
    dm_crop_feed_demand.rename_col_regex(str1="crop-", str2="", dim="Categories1")  # Renaming categories
    dm_crop_demand.append(dm_crop_feed_demand, dim='Variables')
    dm_crop_demand.append(dm_voil, dim='Variables')
    dm_crop_demand.append(dm_aps, dim='Variables')
    dm_crop_demand.append(dm_bev_dom_prod, dim='Variables')

    # Total crop demand by type [kcal] = Sum crop demand (feed + food + non-food)
    dm_crop_demand.operation('agr_demand_feed_temp', '+', 'agr_domestic-production_food',
                             out_col='agr_domestic-production_feed_food', unit='kcal')
    dm_crop_demand.operation('agr_domestic-production_feed_food', '+', 'agr_domestic-production_bioe',
                             out_col='agr_domestic-production_feed_food_bioe', unit='kcal')
    dm_crop_demand.operation('agr_domestic-production_feed_food_bioe', '+', 'agr_aps',
                             out_col='agr_domestic-production_feed_food_bioe_aps', unit='kcal')
    dm_crop_demand.operation('agr_domestic-production_feed_food_bioe_aps', '+', 'agr_domestic-production_bev',
                             out_col='agr_domestic-production', unit='kcal')
    dm_crop_demand = dm_crop_demand.filter({'Variables': ['agr_domestic-production']})

    # Pre processing to remove lgn, algae & insect
    list = ['lgn-energycrop', 'algae', 'insect']
    dm_crop_other = dm_crop_demand.filter({'Categories1': list})
    dm_crop_other.rename_col('agr_domestic-production', 'agr_domestic-production_afw', dim='Variables')
    # Appending for remaining categories
    dm_crop_demand.drop(dim='Categories1', col_label=list)
    DM_crop['crop'].append(dm_crop_demand, dim='Variables')

    # Domestic production with losses [kcal] = domestic prod * food losses [%]
    DM_crop['crop'].operation('agr_domestic-production', '*', 'agr_climate-smart-crop_losses',
                              out_col='agr_domestic-production_afw_raw', unit='kcal')

    # CALIBRATION CROP PRODUCTION --------------------------------------------------------------------------------------
    dm_cal_crop = DM_crop['cal_crop']
    dm_crop = DM_crop['crop'].filter({'Variables': ['agr_domestic-production_afw_raw']})
    dm_cal_rates_crop = calibration_rates(dm_crop, dm_cal_crop, calibration_start_year=1990,
                                          calibration_end_year=2023, years_setting=years_setting)
    DM_crop['crop'].append(dm_cal_rates_crop, dim='Variables')
    DM_crop['crop'].operation('agr_domestic-production_afw_raw', '*', 'cal_rate', dim='Variables',
                              out_col='agr_domestic-production_afw', unit='kcal')
    df_cal_rates_crop = dm_to_database(dm_cal_rates_crop, 'none', 'agriculture', level=0)
    df_cal_crop = dm_to_database(dm_cal_crop, 'none', 'agriculture', level=0)
    df_crop = dm_to_database(dm_crop.filter({'Variables': ['agr_domestic-production_afw_raw']}), 'none', 'agriculture',
                             level=0)

    # Fill NaN with 0.0 for rice (because no rice produced in Switzerland)
    array_temp = DM_crop['crop'].array[:, :, :, :]
    array_temp = np.nan_to_num(array_temp, nan=0)
    DM_crop['crop'].array[:, :, :, :] = array_temp

    # CROP RESIDUES ----------------------------------------------------------------------------------------------------

    # Crop residues per crop type (cereals, oilcrop, sugarcrop) = Domestic production with losses [kcal] * residue yield [kcal/kcal]
    dm_residues = DM_crop['crop'].filter(
        {'Variables': ['agr_domestic-production_afw'], 'Categories1': ['cereal', 'oilcrop', 'sugarcrop']})
    DM_crop['residues_yield'].append(dm_residues, dim='Variables')
    DM_crop['residues_yield'].operation('agr_domestic-production_afw', '*', 'fxa_residues_yield',
                                        out_col='agr_residues', unit='kcal')

    # Total crop residues = sum(Crop residues per crop type) (In KNIME but not used)

    # Residues per use (only for cereal residues) [Mt] = residues [kcal] * biomass hierarchy use [Mt/kcal] FIXME check with DM_SSR if KNIME error assumption is correct (to use residues instead of dom prod afw)
    dm_residues_cereal = DM_crop['residues_yield'].filter({'Variables': ['agr_residues'], 'Categories1': ['cereal']})
    dm_residues_cereal = dm_residues_cereal.flatten()
    idx_residues = dm_residues_cereal.idx
    idx_hierarchy = DM_crop['hierarchy_residues_cereals'].idx
    array_temp = dm_residues_cereal.array[:, :, idx_residues['agr_residues_cereal'], np.newaxis] \
                 * DM_crop['hierarchy_residues_cereals'].array[:, :, idx_hierarchy['agr_biomass-hierarchy_crop_cereal'],
                   :]
    DM_crop['hierarchy_residues_cereals'].add(array_temp, dim='Variables', col_label='agr_residues_emission', unit='Mt')

    # Residues emission [MtCH4, MtN2O] = crop residues [Mt] * emissions factors [MtCH4/Mt, MtN2O/Mt]
    idx_residues = DM_crop['hierarchy_residues_cereals'].idx
    idx_ef = DM_crop['ef_residues'].idx
    array_temp = DM_crop['hierarchy_residues_cereals'].array[:, :, idx_residues['agr_residues_emission'], :, np.newaxis] \
                 * DM_crop['ef_residues'].array[:, :, idx_ef['ef'], :, :]
    DM_crop['ef_residues'].add(array_temp, dim='Variables', col_label='agr_crop_emission', unit='Mt')

    # Gino: Adding SSR DM to send to the TPE
    DM_ssr  = {'food': dm_ssr_food,
               'feed': dm_ssr_feed_pro,
               'bioenergy': dm_ssr_bioe_pro,
               'processed': dm_ssr_food_pro}

    return DM_crop, dm_crop, dm_crop_other, dm_feed_processed, dm_food_processed, df_cal_rates_crop, DM_ssr


# CalculationLeaf AGRICULTURAL LAND DEMAND -----------------------------------------------------------------------------
def land_workflow(DM_land, DM_crop, DM_livestock, dm_crop_other, DM_ind, years_setting):
    # FIBERS -----------------------------------------------------------------------------------------------------------
    # Converting industry fibers from [kt] to [t]
    dm_ind_fiber = DM_ind["natfibers"]
    DM_land['fibers'].append(dm_ind_fiber, dim='Variables')

    DM_land['fibers'].change_unit('ind_dem_natfibers', factor=1000, old_unit='kt', new_unit='t')

    # Domestic supply fiber crop demand [t] = ind demand natural fibers [t] + domestic supply quantity fibers [t]
    DM_land['fibers'].operation('ind_dem_natfibers', '+', 'fxa_domestic-supply-quantity_fibres-plant-eq',
                                out_col='agr_domestic-supply-quantity_fibres-plant-eq', unit='t')

    # Domestic production fiber crop [t] = Domestic supply fiber crop demand [t] * Self sufficiency ratio [%]
    DM_land['fibers'].operation('agr_domestic-supply-quantity_fibres-plant-eq', '*',
                                'fxa_domestic-self-sufficiency_fibres-plant-eq',
                                out_col='agr_domestic-production_fibres-plant-eq', unit='t')

    # Fiber cropland demand [ha] = Domestic production fiber crop [t] / Fiber yield [t/ha]
    dm_fiber_yield = DM_land['yield'].filter({'Categories1': ['fibres-plant-eq']})
    dm_fiber_yield = dm_fiber_yield.flatten()
    DM_land['fibers'].append(dm_fiber_yield, dim='Variables')
    DM_land['fibers'].operation('agr_domestic-production_fibres-plant-eq', '/',
                                'agr_climate-smart-crop_yield_fibres-plant-eq',
                                out_col='agr_land_cropland_raw_fibres-plant-eq', unit='ha')

    # Fill NaN with 0.0
    array_temp = DM_land['fibers'].array[:, :, :]
    array_temp = np.nan_to_num(array_temp, nan=0)
    DM_land['fibers'].array[:, :, :] = array_temp

    # Copy for TPE
    dm_fiber = DM_land['fibers'].copy()

    # LAND DEMAND ------------------------------------------------------------------------------------------------------

    # Categories x11 : cereals, oilcrop, pulse, fruit, veg, starch, sugarcrop, rice , lgn, algae, insect FIXME gas energycrop in Knime but regex issue
    # Calibrated crop demand (8 categories)
    dm_crop_afw = DM_crop['crop'].filter({'Variables': ['agr_domestic-production_afw']})
    # dm_crop_afw.rename_col('cal_agr_domestic-production_food', 'agr_domestic-production_afw', dim='Variables')
    # Appending calibrated dom prod afw with lgn, algae, insect
    dm_crop_afw.append(dm_crop_other, dim='Categories1')
    # Dropping unused yield categories
    DM_land['yield'].drop(dim='Categories1', col_label=['gas-energycrop', 'fibres-plant-eq'])
    # Appending in DM_land
    DM_land['yield'].append(dm_crop_afw, dim='Variables')

    # Cropland by crop type [ha] = domestic prod afw & losses [kcal] / yields [kcal/ha]
    DM_land['yield'].operation('agr_domestic-production_afw', '/',
                               'agr_climate-smart-crop_yield',
                               out_col='agr_land_cropland_raw', unit='ha')

    # When yield = 0, change so that cropland = 0 (and not Nan because divided by 0)
    idx_land = DM_land['yield'].idx
    DM_land['yield'].array[:, :, idx_land['agr_land_cropland_raw'], :] = np.where(
        DM_land['yield'].array[:, :, idx_land['agr_climate-smart-crop_yield'], :] == 0,
        0,
        DM_land['yield'].array[:, :, idx_land['agr_land_cropland_raw'], :]
    )

    # Appending with fiber crop land
    DM_land['fibers'] = DM_land['fibers'].filter({'Variables': ['agr_land_cropland_raw_fibres-plant-eq']})
    DM_land['fibers'].deepen()
    DM_land['yield'].drop(dim='Variables', col_label=['agr_climate-smart-crop_yield', 'agr_domestic-production_afw'])
    DM_land['yield'].append(DM_land['fibers'], dim='Categories1')

    # Calibration cropland per type (without algae, insect and lgn-energycrop)
    dm_cal_cropland = DM_land['cal_cropland']
    dm_cropland = DM_land['yield'].copy()
    dm_cropland.drop(dim='Categories1', col_label=['algae', 'insect', 'lgn-energycrop'])
    dm_cal_rates_cropland = calibration_rates(dm_cropland, dm_cal_cropland, calibration_start_year=1990,
                                              calibration_end_year=2023, years_setting=years_setting)
    dm_cropland.append(dm_cal_rates_cropland, dim='Variables')
    dm_cropland.operation('agr_land_cropland_raw', '*', 'cal_rate', dim='Variables',
                          out_col='agr_land_cropland', unit='ha')

    # Append with cropland for lgn-energycrop, algae & insect
    dm_cropland_others = DM_land['yield'].filter({'Categories1': ['algae', 'insect', 'lgn-energycrop']})
    dm_cropland_others.rename_col('agr_land_cropland_raw', 'agr_land_cropland', dim='Variables')
    dm_cropland = dm_cropland.filter({'Variables': ['agr_land_cropland']})
    # dm_cropland.append(dm_cropland_others, dim='Categories1')

    # Overall cropland [ha] = sum of cropland by type [ha]
    dm_land = dm_cropland.copy()
    dm_land.groupby({'cropland': '.*'}, dim='Categories1', regex=True, inplace=True)
    dm_land.rename_col('agr_land_cropland', 'agr_lus_land_raw', dim='Variables')

    # Appending with grassland from feed
    dm_grassland = DM_livestock['ruminant_density'].filter({'Variables': ['agr_lus_land_raw_grassland']})
    dm_grassland.deepen()
    dm_land.append(dm_grassland, dim='Categories1')

    # Calibration total cropland & grassland
    dm_cal_land = DM_land['cal_land']
    dm_cal_rates_land = calibration_rates(dm_land, dm_cal_land, calibration_start_year=1990,
                                          calibration_end_year=2023, years_setting=years_setting)
    dm_land.append(dm_cal_rates_land, dim='Variables')
    dm_land.operation('agr_lus_land_raw', '*', 'cal_rate', dim='Variables',
                      out_col='agr_lus_land', unit='ha')
    df_cal_rates_land = dm_to_database(dm_cal_rates_land, 'none', 'agriculture', level=0)

    # Overall agricultural land [ha] = overall cropland + grasssland [ha]
    dm_land_use = dm_land.filter({'Variables': ['agr_lus_land']}).copy()  # copu for Land use module
    dm_land.groupby({'agriculture': '.*'}, dim='Categories1', regex=True, inplace=True)

    # RICE CH4 EMISSIONS -----------------------------------------------------------------------------------------------
    # Pre processing
    dm_rice = DM_land['yield'].filter({'Categories1': ['rice']})
    dm_rice = dm_rice.flatten()
    DM_land['rice'].append(dm_rice, dim='Variables')

    # Rice CH4 emissions [tCH4] = cropland for rice [ha] * emissions crop rice [tCH4/ha]
    DM_land['rice'].operation('fxa_emission_crop_rice', '*',
                              'agr_land_cropland_raw_rice',
                              out_col='agr_rice_crop_CH4-emission', unit='t')

    return DM_land, dm_land, dm_land_use, dm_fiber, df_cal_rates_land


# CalculationLeaf NITROGEN BALANCE -------------------------------------------------------------------------------------
def nitrogen_workflow(DM_nitrogen, dm_land, CDM_const, years_setting):
    # FOR GRAPHS -------------------------------------------------------------------------------------------------------

    # Fertilizer application [t] = agricultural land [ha] * input use per type [t] FIXME use calibrated agr_lus_land
    dm_agricultural_land = dm_land.filter({'Variables': ['agr_lus_land'], 'Categories1': ['agriculture']})
    dm_agricultural_land = dm_agricultural_land.flatten()
    idx_land = dm_agricultural_land.idx
    idx_fert = DM_nitrogen['input'].idx
    dm_temp = dm_agricultural_land.array[:, :, idx_land['agr_lus_land_agriculture'], np.newaxis] \
              * DM_nitrogen['input'].array[:, :, idx_fert['agr_climate-smart-crop_input-use'], :]
    DM_nitrogen['input'].add(dm_temp, dim='Variables', col_label='agr_input-use', unit='t')

    # Mineral fertilizers [t] = sum Fertilizer application [t] (nitrogen + phosphate + potash)
    dm_mineral_fertilizer = DM_nitrogen['input'].filter({'Variables': ['agr_input-use'],
                                                         'Categories1': ['nitrogen', 'phosphate', 'potash']})
    dm_mineral_fertilizer.groupby({'mineral': '.*'}, dim='Categories1', regex=True, inplace=True)

    # NO2 EMISSIONS ----------------------------------------------------------------------------------------------------
    # Mineral fertilizer emissions [tNO2] = input use nitrogen [tN] * fertilizer emission [N2O/N]
    dm_nitrogen = DM_nitrogen['input'].filter({'Variables': ['agr_input-use'], 'Categories1': ['nitrogen']})
    dm_nitrogen = dm_nitrogen.flatten()
    DM_nitrogen['emissions'].append(dm_nitrogen, dim='Variables')
    DM_nitrogen['emissions'].operation('agr_input-use_nitrogen', '*', 'fxa_agr_emission_fertilizer',
                                       out_col='agr_crop_emission_N2O-emission_fertilizer_raw', unit='t')

    # Calibration
    dm_n = DM_nitrogen['emissions'].filter({'Variables': ['agr_crop_emission_N2O-emission_fertilizer_raw']})
    dm_cal_n = DM_nitrogen['cal_n']
    dm_cal_n.change_unit('cal_agr_crop_emission_N2O-emission_fertilizer', 10 ** 6, old_unit='Mt',
                         new_unit='t')  # Unit conversion [Mt] => [t]
    dm_cal_rates_n = calibration_rates(dm_n, dm_cal_n, calibration_start_year=1990,
                                       calibration_end_year=2023, years_setting=years_setting)
    dm_n.append(dm_cal_rates_n, dim='Variables')
    dm_n.operation('agr_crop_emission_N2O-emission_fertilizer_raw', '*', 'cal_rate', dim='Variables',
                   out_col='agr_crop_emission_N2O-emission_fertilizer', unit='t')
    df_cal_rates_n = dm_to_database(dm_cal_rates_n, 'none', 'agriculture', level=0)

    # CO2 EMISSIONS ----------------------------------------------------------------------------------------------------
    # Pre processing
    dm_fertilizer_co = DM_nitrogen['input'].filter({'Variables': ['agr_input-use'], 'Categories1': ['liming', 'urea']})
    cdm_fertilizer_co = CDM_const['cdm_fertilizer_co']

    # For liming & urea: CO2 emissions [MtCO2] =  Fertilizer application[t] * emission factor [MtCO2/t]
    idx_cdm = cdm_fertilizer_co.idx
    idx_fert = dm_fertilizer_co.idx
    dm_temp = dm_fertilizer_co.array[:, :, idx_fert['agr_input-use'], :] \
              * cdm_fertilizer_co.array[idx_cdm['cp_ef'], :]
    dm_fertilizer_co.add(dm_temp, dim='Variables', col_label='agr_input-use_emissions-CO2', unit='t')

    return dm_n, dm_fertilizer_co, dm_mineral_fertilizer, df_cal_rates_n


# CalculationLeaf ENERGY & GHG -------------------------------------------------------------------------------------
def energy_ghg_workflow(DM_energy_ghg, DM_crop, DM_land, DM_manure, dm_land, dm_fertilizer_co, dm_liv_N2O, dm_CH4,
                        CDM_const, dm_n, years_setting):
    # ENERGY DEMAND ----------------------------------------------------------------------------------------------------
    # Energy demand from agriculture [ktoe] = energy demand [ktoe/ha] * Agricultural land [ha]
    dm_agricultural_land = dm_land.filter({'Variables': ['agr_lus_land']})
    dm_agricultural_land = dm_agricultural_land.flatten()
    idx_land = dm_agricultural_land.idx
    idx_energy = DM_energy_ghg['energy_demand'].idx
    array_temp = dm_agricultural_land.array[:, :, idx_land['agr_lus_land_agriculture'], np.newaxis] \
                 * DM_energy_ghg['energy_demand'].array[:, :, idx_energy['agr_climate-smart-crop_energy-demand'], :]
    DM_energy_ghg['energy_demand'].add(array_temp, dim='Variables', col_label='agr_energy-demand_raw', unit='ktoe')

    # Calibration - Energy demand
    dm_cal_energy_demand = DM_energy_ghg['cal_energy_demand']
    dm_energy_demand = DM_energy_ghg['energy_demand'].filter({'Variables': ['agr_energy-demand_raw']})
    dm_cal_rates_energy_demand = calibration_rates(dm_energy_demand, dm_cal_energy_demand, calibration_start_year=1990,
                                                   calibration_end_year=2023, years_setting=years_setting)
    # Fill NaN with 1.0 (when 0 & 0, issue with calibration)
    array_temp = dm_cal_rates_energy_demand.array[:, :, :, :]
    array_temp = np.nan_to_num(array_temp, nan=1.0)
    dm_cal_rates_energy_demand.array[:, :, :, :] = array_temp
    DM_energy_ghg['energy_demand'].append(dm_cal_rates_energy_demand, dim='Variables')
    DM_energy_ghg['energy_demand'].operation('agr_energy-demand_raw', '*', 'cal_rate', dim='Variables',
                                             out_col='agr_energy-demand', unit='ktoe')
    df_cal_rates_energy_demand = dm_to_database(dm_cal_rates_energy_demand, 'none', 'agriculture', level=0)

    # CO2 EMISSIONS ----------------------------------------------------------------------------------------------------
    # Pre processing : filtering and deepening constants
    cdm_CO2 = CDM_const['cdm_CO2']

    # Energy direct emission [MtCO2] = energy demand [ktoe] * emission factor [MtCO2/ktoe]
    dm_energy = DM_energy_ghg['energy_demand']
    idx_energy = dm_energy.idx
    idx_cdm = cdm_CO2.idx
    array_temp = dm_energy.array[:, :, idx_energy['agr_energy-demand'], :] \
                 * cdm_CO2.array[idx_cdm['cp_emission-factor_CO2'], :]
    DM_energy_ghg['energy_demand'].add(array_temp, dim='Variables', col_label='agr_input-use_emissions-CO2',
                                       unit='Mt')

    # Overall CO2 emission from fuel [Mt] = sum (Energy direct emission [MtCO2])
    dm_CO2 = DM_energy_ghg['energy_demand'].filter({'Variables': ['agr_input-use_emissions-CO2']})
    dm_CO2.groupby({'fuel': '.*'}, dim='Categories1', regex=True, inplace=True)
    dm_CO2 = dm_CO2.flatten()

    # Unit conversion : Overall CO2 emission from fuel [Mt] => [t]
    dm_CO2.change_unit('agr_input-use_emissions-CO2_fuel', 10 ** 6, old_unit='Mt',
                 new_unit='t')  # Unit conversion [kt] => [t]
    dm_CO2 = dm_CO2.filter({'Variables': ['agr_input-use_emissions-CO2_fuel']})
    dm_CO2.deepen()

    # Appending CO2 emissions: fuel, liming, urea from Nitrogen Balance workflow
    dm_CO2.append(dm_fertilizer_co.filter({'Variables': ['agr_input-use_emissions-CO2']}), dim='Categories1')

    # Rename to _raw for calibration
    dm_CO2.rename_col('agr_input-use_emissions-CO2', 'agr_input-use_emissions-CO2_raw', dim='Variables')

    # Calibration CO2 from fuel, liming, urea emissions FIXME check with crop_work if it makes sense to change the calibration order from KNIME to put it before summing
    dm_cal_CO2_input = DM_energy_ghg['cal_input']
    dm_cal_CO2_input.change_unit('cal_agr_input-use_emissions-CO2', 10 ** 3, old_unit='kt',
                                 new_unit='t')  # Unit conversion [kt] => [t]
    dm_cal_rates_CO2_input = calibration_rates(dm_CO2, dm_cal_CO2_input, calibration_start_year=1990,
                                               calibration_end_year=2023, years_setting=years_setting)
    dm_CO2.append(dm_cal_rates_CO2_input, dim='Variables')
    dm_CO2.operation('agr_input-use_emissions-CO2_raw', '*', 'cal_rate', dim='Variables',
                     out_col='agr_input-use_emissions-CO2', unit='t')
    df_cal_rates_CO2_input = dm_to_database(dm_cal_rates_CO2_input, 'none', 'agriculture', level=0)

    # Overall CO2 emission [t] = sum (fuel, liming, urea)
    dm_fuel_input = dm_CO2.filter({'Variables': ['agr_input-use_emissions-CO2']})
    dm_fuel_input.groupby({'CO2-emission': '.*'}, dim='Categories1', regex=True, inplace=True)

    # Adding dummy columns
    dm_fuel_input.add(0.0, dummy=True, col_label='N2O-emission', dim='Categories1', unit='t')
    dm_fuel_input.add(0.0, dummy=True, col_label='CH4-emission', dim='Categories1', unit='t')

    # CROP RESIDUE EMISSIONS -------------------------------------------------------------------------------------------
    # Unit conversion : N2O, CH4 from crop residues [Mt] => [t]
    dm_ghg = DM_crop['ef_residues'].filter(
        {'Variables': ['agr_crop_emission'], 'Categories2': ['N2O-emission', 'CH4-emission']})

    dm_ghg.change_unit('agr_crop_emission', factor=1e6, old_unit='Mt', new_unit='t')
    dm_ghg.rename_col('agr_crop_emission', 'agr_emission_residues', 'Variables')

    # Summing per residue emission type (soil & burnt)
    dm_ghg.group_all(dim='Categories1', inplace=True)

    # Pre processing (name matching, adding dummy columns)
    dm_ghg.add(0.0, dummy=True, col_label='CO2-emission', dim='Categories1', unit='t')

    # LIVESTOCK EMISSIONS -------------------------------------------------------------------------------------------
    # Manure N2O emissions = sum (manure emission per livestock type & manure type)
    dm_N2O_liv = dm_liv_N2O.filter({'Variables': ['agr_liv_N2O-emission']})
    dm_N2O_liv = dm_N2O_liv.flatten()
    dm_N2O_liv.groupby({'N2O-emission': '.*'}, dim='Categories1', regex=True, inplace=True)
    # Adding dummy columns
    dm_N2O_liv.add(0.0, dummy=True, col_label='CO2-emission', dim='Categories1', unit='t')
    dm_N2O_liv.add(0.0, dummy=True, col_label='CH4-emission', dim='Categories1', unit='t')

    # CH4 emissions = sum (manure & enteric emission per livestock type)
    dm_CH4_liv = dm_CH4.filter({'Variables': ['agr_liv_CH4-emission']})
    dm_CH4_liv = dm_CH4_liv.flatten()
    dm_CH4_liv.groupby({'CH4-emission': '.*'}, dim='Categories1', regex=True, inplace=True)  # Problem
    # Adding dummy columns
    dm_CH4_liv.add(0.0, dummy=True, col_label='CO2-emission', dim='Categories1', unit='t')
    dm_CH4_liv.add(0.0, dummy=True, col_label='N2O-emission', dim='Categories1', unit='t')

    # NO2 EMISSIONS FROM FERTILIZERS -----------------------------------------------------------------------------------
    # Filter and format
    dm_N2O_fert = dm_n.filter({'Variables': ['agr_crop_emission_N2O-emission_fertilizer']})
    dm_N2O_fert.rename_col('agr_crop_emission_N2O-emission_fertilizer',
                'agr_fertilizer_N2O-emission', 'Variables')
    dm_N2O_fert.deepen()
    # Adding dummy columns
    dm_N2O_fert.add(0.0, dummy=True, col_label='CO2-emission', dim='Categories1', unit='t')
    dm_N2O_fert.add(0.0, dummy=True, col_label='CH4-emission', dim='Categories1', unit='t')

    # RICE EMISSIONS ---------------------------------------------------------------------------------------------------
    # Adding rice emissions
    dm_CH4_rice = DM_land['rice'].filter({'Variables': ['agr_rice_crop_CH4-emission']})
    dm_CH4_rice.deepen()
    # Adding dummy columns
    dm_CH4_rice.add(0.0, dummy=True, col_label='CO2-emission', dim='Categories1', unit='t')
    dm_CH4_rice.add(0.0, dummy=True, col_label='N2O-emission', dim='Categories1', unit='t')

    # TOTAL GHG EMISSIONS ----------------------------------------------------------------------------------------------

    # Appending crop + fuel + livestock + rice emissions
    dm_ghg.append(dm_N2O_liv, dim='Variables')  # N2O, CH4 from crop residues with NO2 from livestock
    dm_ghg.append(dm_CH4_liv, dim='Variables')  # CH4 from livestock
    dm_ghg.append(dm_CH4_rice, dim='Variables')  # CH4 from rice
    dm_ghg.append(dm_fuel_input, dim='Variables')  # CO2 from fuel, liming, urea
    dm_ghg.append(dm_N2O_fert, dim='Variables')  # N2O from fertilizer

    # Agriculture GHG emissions per GHG [t] =  crop + fuel + livestock + rice + fertilizer emissions per GHG
    dm_ghg.operation('agr_emission_residues', '+', 'agr_liv_N2O-emission',
                     out_col='residues_and_N2O_liv', unit='t')
    dm_ghg.operation('residues_and_N2O_liv', '+', 'agr_liv_CH4-emission',
                     out_col='residues_and_N2O_liv_and_CH4_liv', unit='t')
    dm_ghg.operation('residues_and_N2O_liv_and_CH4_liv', '+', 'agr_rice_crop',
                     out_col='residues_and_N2O_liv_and_CH4_liv_and_rice', unit='t')
    dm_ghg.operation('residues_and_N2O_liv_and_CH4_liv_and_rice', '+', 'agr_input-use_emissions-CO2',
                     out_col='residues_and_N2O_liv_and_CH4_liv_and_rice_and_CO2', unit='t')
    dm_ghg.operation('residues_and_N2O_liv_and_CH4_liv_and_rice_and_CO2', '+',
                     'agr_fertilizer',
                     out_col='agr_emissions_raw', unit='t')
    # Dropping the intermediate values
    dm_ghg = dm_ghg.filter({'Variables': ['agr_emissions_raw']})

    # Calibration GHG emissions: overall CO2, CH4, NO2
    dm_cal_ghg = DM_energy_ghg['cal_GHG']
    dm_cal_rates_ghg = calibration_rates(dm_ghg, dm_cal_ghg, calibration_start_year=1990,
                                         calibration_end_year=2023, years_setting=years_setting)
    dm_ghg.append(dm_cal_rates_ghg, dim='Variables')
    dm_ghg.operation('agr_emissions_raw', '*', 'cal_rate', dim='Variables',
                     out_col='agr_emissions', unit='t')
    df_cal_rates_ghg = dm_to_database(dm_cal_rates_ghg, 'none', 'agriculture', level=0)

    # FORMATTING FOR TPE & INTERFACE -----------------------------------------------------------------------------------
    # CO2 emissions from fertilizer & energy
    dm_input_use_CO2 = dm_CO2.filter({'Variables': ['agr_input-use_emissions-CO2']})
    dm_input_use_CO2.change_unit('agr_input-use_emissions-CO2', 1e-6, old_unit='t', new_unit='Mt')
    dm_input_use_CO2 = dm_input_use_CO2.flatten()

    # Fertilizer emissions N2O
    dm_fertilizer_N2O = dm_n.filter({'Variables': ['agr_crop_emission_N2O-emission_fertilizer']})
    dm_fertilizer_N2O.change_unit('agr_crop_emission_N2O-emission_fertilizer', 1e-6, old_unit='t', new_unit='Mt')
    # dm_fertilizer_N2O.rename_col('agr_crop_emission_N2O-emission', 'agr_emissions-N2O_crop_fertilizer', 'Variables')

    # Crop residue emissions
    dm_crop_residues = DM_crop['ef_residues'].filter({'Variables': ['agr_crop_emission'],
                                                      'Categories1': ['burnt-residues', 'soil-residues'],
                                                      'Categories2': ['N2O-emission', 'CH4-emission']})
    dm_crop_residues.rename_col('agr_crop_emission', 'agr', dim='Variables')
    dm_crop_residues.rename_col_regex('emission', 'emissions', dim='Categories2')
    dm_crop_residues.rename_col('burnt-residues', 'crop_burnt-residues', dim='Categories1')
    dm_crop_residues.rename_col('soil-residues', 'crop_soil-residues', dim='Categories1')
    dm_crop_residues.switch_categories_order(cat1='Categories2', cat2='Categories1')
    dm_crop_residues.rename_col('CH4-emissions', 'emissions-CH4', "Categories1")
    dm_crop_residues.rename_col('N2O-emissions', 'emissions-N2O', "Categories1")
    dm_crop_residues = dm_crop_residues.flatten().flatten()
    dm_crop_residues.drop("Variables", ['agr_emissions-CH4_crop_soil-residues'])

    # Livestock emissions CH4 (manure & enteric)
    dm_CH4_liv_tpe = dm_CH4.filter({'Variables': ['agr_liv_CH4-emission']})
    dm_CH4_liv_tpe.change_unit('agr_liv_CH4-emission', 1e-6, old_unit='t', new_unit='Mt')
    dm_CH4_liv_tpe.switch_categories_order(cat1='Categories2', cat2='Categories1')
    dm_CH4_liv_tpe.rename_col("agr_liv_CH4-emission", "agr_emissions-CH4_liv", "Variables")
    dm_CH4_liv_tpe = dm_CH4_liv.flatten()
    dm_CH4_liv_tpe = dm_CH4_liv.flatten()

    # Livestock emissions N2O (manure)
    dm_N2O_liv_tpe = dm_liv_N2O.filter({'Variables': ['agr_liv_N2O-emission']})
    dm_N2O_liv_tpe.change_unit('agr_liv_N2O-emission', 1e-6, old_unit='t', new_unit='Mt')
    dm_N2O_liv_tpe.switch_categories_order(cat1='Categories2', cat2='Categories1')
    dm_N2O_liv_tpe.rename_col("agr_liv_N2O-emission", "agr_emissions-N2O_liv", "Variables")
    dm_N2O_liv_tpe = dm_N2O_liv.flatten()
    dm_N2O_liv_tpe = dm_N2O_liv.flatten()

    # Rice emissions
    dm_CH4_rice = DM_land['rice'].filter({'Variables': ['agr_rice_crop_CH4-emission']})
    dm_CH4_rice.change_unit('agr_rice_crop_CH4-emission', 1e-6, old_unit='t', new_unit='Mt')
    dm_CH4_rice.rename_col('agr_rice_crop_CH4-emission', 'agr_emissions-CH4_crop_rice', 'Variables')

    return DM_energy_ghg, dm_CO2, dm_input_use_CO2, dm_crop_residues, dm_CH4_liv_tpe, dm_N2O_liv_tpe, dm_CH4_rice, dm_fertilizer_N2O, df_cal_rates_ghg


def agriculture_landuse_interface(DM_bioenergy, dm_lgn, dm_land_use, write_xls=False):
    dm_wood = DM_bioenergy['solid-mix'].filter({"Variables": ["agr_bioenergy_biomass-demand_solid"],
                                                "Categories1": ['fuelwood-and-res']})
    dm_lgn = dm_lgn.filter({"Variables": ["agr_bioenergy_biomass-demand_liquid_lgn"],
                            "Categories1": ['lgn-btl-fuelwood-and-res']})
    dm_land_use = dm_land_use.filter({"Variables": ["agr_lus_land"]})

    DM_lus = {"wood": dm_wood,
              "lgn": dm_lgn,
              "landuse": dm_land_use}

    # dm_dh
    if write_xls is True:
        dm_lus = DM_bioenergy['solid-mix'].filter({"Variables": ["agr_bioenergy_biomass-demand_solid"],
                                                   "Categories1": ['fuelwood-and-res']}).flatten()
        dm_lus.append(dm_lgn.filter({"Variables": ["agr_bioenergy_biomass-demand_liquid_lgn"],
                                     "Categories1": ['lgn-btl-fuelwood-and-res']}).flatten(), "Variables")
        dm_lus.append(dm_land_use.filter({"Variables": ["agr_lus_land"]}).flatten(), "Variables")

        """current_file_directory = os.path.dirname(os.path.abspath(__file__))
        df_lus = dm_lus.write_df()
        df_lus.to_excel(
            current_file_directory + "/../_database/data/xls/" + 'All-Countries_interface_from-agriculture-to-landuse.xlsx',
            index=False)"""

    return DM_lus


def agriculture_emissions_interface(DM_nitrogen, dm_CO2, DM_crop, DM_manure, DM_land, dm_input_use_CO2,
                                    dm_crop_residues, dm_CH4, dm_N2O_liv, dm_CH4_rice, dm_fertilizer_N2O,
                                    write_xls=False):
    # Append everything
    dm_ems = dm_fertilizer_N2O.copy()
    dm_ems.append(dm_input_use_CO2, dim='Variables')
    dm_ems.append(dm_crop_residues, dim='Variables')
    dm_ems.append(dm_CH4.filter({'Variables': ['agr_liv_CH4-emission']}).flatten().flatten(), dim='Variables')
    dm_ems.append(dm_N2O_liv.filter({'Variables': ['agr_liv_N2O-emission']}).flatten().flatten(), dim='Variables')
    dm_ems.append(dm_CH4_rice, dim='Variables')

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


def agriculture_ammonia_interface(dm_mineral_fertilizer, write_xls=False):
    # Demand for Mineral fertilizers
    dm_ammonia = dm_mineral_fertilizer.filter({'Variables': ['agr_input-use']})
    dm_ammonia.rename_col('agr_input-use', 'agr_product-demand', dim='Variables')
    dm_ammonia.rename_col('mineral', 'fertilizer', dim='Categories1')

    # write
    """if write_xls is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        dm_ammonia = dm_ammonia.write_df()
        dm_ammonia.to_excel(
            current_file_directory + "/../_database/data/xls/" + 'All-Countries_interface_from-agriculture-to-ammonia.xlsx',
            index=False)"""

    return dm_ammonia


def agriculture_storage_interface(DM_energy_ghg, write_xls=False):
    # TODO: storage is not done for the moment, we'll add this when storage will be done
    # FIXME: Energy demand filter change to other unit ([TWh] instead of [ktoe])
    dm_storage = DM_energy_ghg['caf_energy_demand'].filter_w_regex(
        {'Variables': 'agr_energy-demand', 'Categories1': '.*ff.*'})

    # Summing in the same category
    dm_storage.groupby({'gas-ff-natural': 'gas-ff-natural|liquid-ff-lpg'}, dim='Categories1', regex=True, inplace=True)

    # Renaming
    dm_storage.rename_col('liquid-ff-fuel-oil', 'liquid-ff-oil', dim='Categories1')

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


def agriculture_power_interface(DM_energy_ghg, DM_bioenergy, write_xls=False):
    dm_pow = DM_energy_ghg['energy_demand'].filter_w_regex(
        {'Variables': 'agr_energy-demand', 'Categories1': '.*electricity.*'})
    dm_pow = dm_pow.flatten()
    ktoe_to_gwh = 0.0116222 * 1000  # from KNIME factor
    dm_pow.array = dm_pow.array * ktoe_to_gwh
    dm_pow.units["agr_energy-demand_electricity"] = "GWh"

    dm_wood = DM_bioenergy['solid-mix'].filter({"Variables": ["agr_bioenergy_biomass-demand_solid"],
                                                "Categories1": ['fuelwood-and-res']})

    DM_pow = {"wood": dm_wood,
              "pow": dm_pow}

    # write
    """if write_xls is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        dm_pow = dm_pow.write_df()
        dm_pow.to_excel(
            current_file_directory + "/../_database/data/xls/" + 'All-Countries_interface_from-agriculture-to-power.xlsx',
            index=False)"""

    return DM_pow


def agriculture_minerals_interface(DM_nitrogen, DM_bioenergy, dm_lgn, write_xls=False):
    # Demand for phosphate & potash
    dm_minerals = DM_nitrogen['input'].filter({'Variables': ['agr_input-use'], 'Categories1': ['phosphate', 'potash']})
    dm_minerals.change_unit('agr_input-use', 1e-6, old_unit='t', new_unit='Mt')
    dm_minerals.rename_col('agr_input-use', 'agr_demand', 'Variables')
    dm_minerals = dm_minerals.flatten()

    # Demand for fuelwood (solid)
    dm_solid = DM_bioenergy['solid-mix'].filter(
        {'Variables': ['agr_bioenergy_biomass-demand_solid'], 'Categories1': ['fuelwood-and-res']})
    dm_solid.change_unit('agr_bioenergy_biomass-demand_solid', 0.1264, old_unit='TWh', new_unit='Mt')
    dm_solid = dm_solid.flatten()

    # Demand for fuelwood (liquid)
    dm_liquid = dm_lgn.filter(
        {'Variables': ['agr_bioenergy_biomass-demand_liquid_lgn'], 'Categories1': ['lgn-btl-fuelwood-and-res']})
    dm_liquid.rename_col('lgn-btl-fuelwood-and-res', 'btl_fuelwood-and-res', dim='Categories1')
    dm_liquid.rename_col('agr_bioenergy_biomass-demand_liquid_lgn', 'agr_bioenergy_biomass-demand_liquid',
                         dim='Variables')
    dm_liquid.change_unit('agr_bioenergy_biomass-demand_liquid', factor=0.00000000000116222, old_unit='kcal',
                          new_unit='TWh')
    dm_liquid.change_unit('agr_bioenergy_biomass-demand_liquid', factor=0.1264, old_unit='TWh',
                          new_unit='Mt')
    dm_liquid = dm_liquid.filter({'Variables': ['agr_bioenergy_biomass-demand_liquid']})
    dm_liquid = dm_liquid.flatten()

    # Appending everything together
    dm_minerals.append(dm_solid, dim='Variables')
    dm_minerals.append(dm_liquid, dim='Variables')

    # writing dm minerals
    """if write_xls is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        dm_minerals = dm_minerals.write_df()
        dm_minerals.to_excel(
            current_file_directory + "/../_database/data/xls/" + 'All-Countries_interface_from-agriculture-to-minerals.xlsx',
            index=False)"""

    return dm_minerals


def agriculture_refinery_interface(DM_energy_ghg):
    dm_ref = DM_energy_ghg['energy_demand'].filter_w_regex({'Variables': 'agr_energy-demand', 'Categories1': '.*ff.*'})

    # Summing in the same category
    dm_ref.groupby({'gas-ff-natural': 'gas-ff-natural|liquid-ff-lpg'}, dim='Categories1', regex=True, inplace=True)

    # Renaming
    dm_ref.rename_col('liquid-ff-fuel-oil', 'liquid-ff-oil', dim='Categories1')

    # order
    dm_ref.sort("Categories1")

    # change unit
    ktoe_to_twh = 0.0116222  # from KNIME factor
    dm_ref.change_unit('agr_energy-demand', ktoe_to_twh, old_unit='ktoe', new_unit='TWh')

    return dm_ref


def agriculture_TPE_interface(DM_livestock, DM_crop, dm_crop_other, DM_feed, dm_aps, dm_input_use_CO2, dm_crop_residues,
                              dm_CH4, dm_liv_N2O, dm_CH4_rice, dm_fertilizer_N2O, DM_energy_ghg, DM_bioenergy, dm_lgn,
                              dm_eth, dm_oil, dm_aps_ibp, DM_food_demand, dm_lfs_pro, dm_lfs, DM_land, dm_fiber,
                              dm_aps_ibp_oil, dm_voil_tpe, DM_alc_bev, dm_biofuel_fdk, dm_liv_pop, DM_ssr, dm_fertilizer_co, DM_manure):
    kcal_to_TWh = 1.163e-12

    # Livestock population
    # Note : check if it includes the poultry for eggs
    dm_liv_meat = dm_liv_pop.filter_w_regex({'Variables': 'agr_liv_population', 'Categories1': 'meat.*'}, inplace=False)
    dm_tpe = dm_liv_meat.flattest()

    # Meat
    dm_meat = DM_livestock['losses'].filter({'Variables': ['agr_domestic_production']})
    dm_tpe.append(dm_meat.flattest(), dim='Variables')

    # Crop production
    dm_crop_prod_food = DM_crop['crop'].filter({'Variables': ['agr_domestic-production_afw']})
    dm_tpe.append(dm_crop_prod_food.flattest(), dim='Variables')
    dm_tpe.append(dm_crop_other.flattest(), dim='Variables')

    # Livestock feed
    dm_feed = DM_feed['ration'].filter({'Variables': ['agr_demand_feed']})
    dm_aps.rename_col('agr_feed_aps', 'agr_demand_feed_aps', dim='Variables')
    dm_tpe.append(dm_feed.flattest(), dim='Variables')
    dm_tpe.append(dm_aps.flattest(), dim='Variables')

    # CO2 emissions
    dm_tpe.append(dm_input_use_CO2.flattest(), dim='Variables')

    # CH4 emissions
    dm_tpe.append(dm_CH4.flattest(), dim='Variables')
    dm_tpe.append(dm_crop_residues.flattest(), dim='Variables')
    dm_tpe.append(dm_CH4_rice.flattest(), dim='Variables')

    # N2O emissions Note : residues already accounted for in df_residues in CH4 emissions
    dm_tpe.drop(col_label='cal_rate', dim='Variables')
    dm_liv_N2O.drop(dim='Variables', col_label='cal_rate')
    dm_tpe.append(dm_liv_N2O.flattest(), dim='Variables')
    dm_tpe.append(dm_fertilizer_N2O.flattest(), dim='Variables')

    # Energy use per type
    dm_energy_demand = DM_energy_ghg['energy_demand'].filter({'Variables': ['agr_energy-demand']})
    # Unit conversion [ktoe] => [TWh]
    dm_energy_demand.change_unit('agr_energy-demand', factor=0.0116222, old_unit='ktoe', new_unit='TWh')
    dm_tpe.append(dm_energy_demand.flattest(), dim='Variables')

    # Bioenergy capacity
    dm_bio_cap_biogas = DM_bioenergy['bgs-mix'].filter({'Variables': ['agr_bioenergy-capacity_bgs-tec']})
    dm_bio_cap_biodiesel = DM_bioenergy['liquid-biodiesel'].filter(
        {'Variables': ['agr_bioenergy-capacity_liq-bio-prod_biodiesel']})
    dm_bio_cap_biogasoline = DM_bioenergy['liquid-biogasoline'].filter(
        {'Variables': ['agr_bioenergy-capacity_liq-bio-prod_biogasoline']})
    dm_bio_cap_biojetkerosene = DM_bioenergy['liquid-biojetkerosene'].filter(
        {'Variables': ['agr_bioenergy-capacity_liq-bio-prod_biojetkerosene']})
    dm_tpe.append(dm_bio_cap_biogas.flattest(), dim='Variables')
    dm_tpe.append(dm_bio_cap_biodiesel.flattest(), dim='Variables')
    dm_tpe.append(dm_bio_cap_biojetkerosene.flattest(), dim='Variables')
    dm_tpe.append(dm_bio_cap_biogasoline.flattest(), dim='Variables')

    # Bioenergy feedstock mix (reunion of others fdk)

    # Liquid bioenergy-feedstock mix
    dm_fdk_oil = dm_oil.filter({'Variables': ['agr_bioenergy_biomass-demand_liquid_oil']})
    dm_fdk_oil.rename_col('agr_bioenergy_biomass-demand_liquid_oil', 'agr_bioenergy_biomass-demand_liquid',
                          dim='Variables')
    dm_fdk_eth = dm_eth.filter({'Variables': ['agr_bioenergy_biomass-demand_liquid_eth']})
    dm_fdk_eth.rename_col('agr_bioenergy_biomass-demand_liquid_eth', 'agr_bioenergy_biomass-demand_liquid',
                          dim='Variables')
    dm_fdk_lgn = dm_lgn.filter({'Variables': ['agr_bioenergy_biomass-demand_liquid_lgn']})
    dm_fdk_lgn.rename_col('agr_bioenergy_biomass-demand_liquid_lgn', 'agr_bioenergy_biomass-demand_liquid',
                          dim='Variables')
    # Unit conversion [kcal] => [TWh]
    dm_fdk_oil.append(dm_fdk_eth, dim='Categories1')
    dm_fdk_oil.append(dm_fdk_lgn, dim='Categories1')
    dm_fdk_oil.change_unit('agr_bioenergy_biomass-demand_liquid', kcal_to_TWh, old_unit='kcal', new_unit='TWh')
    dm_fdk_liquid = dm_fdk_oil.copy()  # Rename

    # oil aps
    dm_oil_aps = dm_aps_ibp.filter({'Variables': ['agr_aps'], 'Categories2': ['fdk-voil']})
    dm_oil_aps.group_all('Categories2')
    dm_oil_aps.rename_col('agr_aps', 'agr_bioenergy_biomass-demand_liquid', dim='Variables')
    dm_oil_aps.change_unit('agr_bioenergy_biomass-demand_liquid', kcal_to_TWh, old_unit='kcal', new_unit='TWh')
    dm_fdk_liquid.append(dm_oil_aps, dim='Categories1')

    # oil for oilcrops
    dm_voil_tpe.rename_col('oil-voil', 'oil-oilcrop', dim='Categories1')
    dm_voil_tpe.change_unit('agr_bioenergy_biomass-demand_liquid', factor=kcal_to_TWh, old_unit='kcal', new_unit='TWh')
    dm_fdk_liquid.append(dm_voil_tpe, dim='Categories1')

    # lgn demand
    dm_liquid_lgn = dm_biofuel_fdk.filter({'Variables': ['agr_bioenergy_biomass-demand_liquid_lgn']})
    dm_liquid_lgn.change_unit('agr_bioenergy_biomass-demand_liquid_lgn', factor=kcal_to_TWh, old_unit='kcal',
                              new_unit='TWh')
    dm_liquid_lgn.deepen()
    dm_fdk_liquid.append(dm_liquid_lgn, dim='Categories1')

    dm_tpe.append(dm_fdk_liquid.flattest(), dim='Variables')

    # oil industry byproducts
    dm_aps_ibp_oil.change_unit('agr_bioenergy_fdk-aby', factor=kcal_to_TWh, old_unit='kcal', new_unit='TWh')
    dm_aps_ibp_oil = dm_aps_ibp_oil.flatten()

    # eth industry byproducts
    dm_eth_ind_bp = DM_alc_bev['biomass_hierarchy'].filter(
        {'Variables': ['agr_bev_ibp_use_oth'], 'Categories1': ['biogasoline']})
    dm_eth_ind_bp.change_unit('agr_bev_ibp_use_oth', factor=kcal_to_TWh, old_unit='kcal',
                              new_unit='TWh')
    dm_eth_ind_bp = dm_eth_ind_bp.flatten()
    dm_aps_ibp_oil.append(dm_eth_ind_bp, dim='Variables')
    dm_ind_bp = dm_aps_ibp_oil
    dm_tpe.append(dm_ind_bp.flattest(), dim='Variables')

    # Total bioenergy consumption (sum of liquid, biogas feedstock kcal) (solid not included in KNIME) FIXME check with crop_work if solid should be considered
    # Sum liquid & solid
    dm_bioenergy = dm_fdk_liquid.group_all('Categories1', inplace=False)
    dm_bioenergy.append(dm_ind_bp, dim='Variables')
    dm_bioenergy.groupby({'agr_crop-cons_bioenergy': '.*'}, dim='Variables', inplace=True, regex=True)
    dm_bioenergy.change_unit('agr_crop-cons_bioenergy', 1 / kcal_to_TWh, old_unit='TWh', new_unit='kcal')
    dm_tpe.append(dm_bioenergy.flattest(), dim='Variables')

    # Notes : some oil categories seem to differ with KNIME (unit = kcal, tpe wants TWh)

    # Solid bioenergy - feedstock mix
    dm_fdk_solid = DM_bioenergy['solid-mix'].filter({'Variables': ['agr_bioenergy_biomass-demand_solid']})
    dm_tpe.append(dm_fdk_solid.flattest(), dim='Variables')

    # Biogas feedstock mix
    dm_fdk_biogas = DM_bioenergy['digestor-mix'].filter({'Variables': ['agr_bioenergy_biomass-demand_biogas']})
    dm_tpe.append(dm_fdk_biogas.flattest(), dim='Variables')

    # Crop use
    # Total food from crop (does not include processed food)
    dm_crop_food = dm_lfs.filter_w_regex({'Variables': 'agr_demand', 'Categories1': 'crop.*'})
    dm_crop_food.groupby({'food_crop': '.*'}, dim='Categories1', regex=True, inplace=True)
    dm_tpe.append(dm_crop_food.flattest(), dim='Variables')
    # Total feed from crop
    dm_crop_feed = DM_feed['ration'].filter_w_regex({'Variables': 'agr_demand_feed', 'Categories1': 'crop.*'})
    dm_crop_feed.groupby({'crop': '.*'}, dim='Categories1', regex=True, inplace=True)
    dm_tpe.append(dm_crop_feed.flattest(), dim='Variables')

    # Solid (same as bioenergy feedstock mix) Note : not included in KNIME

    # Total non-food consumption (beverages and fiber crops) FIXME check with crop_work if okay to consider fiber crops
    # Beverages
    dm_crop_bev = dm_lfs_pro.filter_w_regex({'Variables': 'agr_domestic_production', 'Categories1': 'pro-bev.*'})
    dm_crop_bev.groupby({'crop-bev': '.*'}, dim='Categories1', regex=True, inplace=True)
    dm_crop_bev = dm_crop_bev.flatten()
    # Fiber crops
    dm_crop_fiber = dm_fiber.filter_w_regex({'Variables': 'agr_domestic-production.*'})
    # Unit conversion : [t] => [kcal]
    dm_crop_fiber.change_unit('agr_domestic-production_fibres-plant-eq', 4299300, old_unit='t', new_unit='kcal')
    # Total non-food consumption [kcal] = bev + fibers
    dm_crop_bev.append(dm_crop_fiber, dim='Variables')
    dm_crop_bev.operation('agr_domestic-production_fibres-plant-eq', '+', 'agr_domestic_production_crop-bev',
                          out_col='agr_crop-cons_non-food', unit='kcal')
    dm_tpe.append(dm_crop_bev.flattest(), dim='Variables')
    dm_tpe.append(dm_lfs.flattest(), dim='Variables')

    # Self-sufficiency ratio
    dm_ssr_food = DM_ssr['food']
    dm_ssr_feed = DM_ssr['feed']
    dm_ssr_bioenergy = DM_ssr['bioenergy']
    dm_ssr_processed = DM_food_demand['food-net-import-pro']
    dm_ssr = dm_ssr_food.copy()
    dm_ssr.append(dm_ssr_feed, dim='Categories1')
    dm_ssr.append(dm_ssr_processed, dim='Categories1')
    dm_ssr.append(dm_ssr_bioenergy, dim='Categories1')
    dm_tpe.append(dm_ssr.flattest(), dim='Variables')

    #Input-use
    dm_input = dm_fertilizer_co.filter({'Variables': ['agr_input-use']})
    dm_tpe.append(dm_input.flattest(), dim='Variables')

    # Land use
    dm_cropland = DM_land['yield']
    dm_tpe.append(dm_cropland.flattest(), dim='Variables')

    # Livestock slaughtered
    dm_slaughtered = DM_livestock['liv_slaughtered_rate'].filter({'Variables': ['agr_liv_population_slau']})
    dm_tpe.append(dm_slaughtered.flattest(), dim='Variables')

    # Manure
    dm_manure = DM_manure['liv_n-stock'].filter({'Variables': ['agr_liv_n-stock']})
    dm_tpe.append(dm_manure.flattest(), dim='Variables')

    # Grassland
    dm_grassland = DM_livestock['ruminant_density'].filter({'Variables': ['agr_lus_land_raw_grassland','agr_climate-smart-livestock_density']})
    dm_tpe.append(dm_grassland, dim='Variables')

    return dm_tpe


# ----------------------------------------------------------------------------------------------------------------------
# AGRICULTURE ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def agriculture(lever_setting, years_setting, DM_input, interface=Interface()):
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    DM_ots_fts, DM_lifestyle, DM_food_demand, DM_livestock, DM_alc_bev, DM_bioenergy, DM_manure, DM_feed, DM_crop, DM_land, DM_nitrogen, DM_energy_ghg, CDM_const = read_data(
        DM_input, lever_setting)

    cntr_list = DM_food_demand['food-net-import-pro'].col_labels['Country']

    # Link interface or Simulate data from other modules
    if interface.has_link(from_sector='lifestyles', to_sector='agriculture'):
        DM_lfs = interface.get_link(from_sector='lifestyles', to_sector='agriculture')
        # FIXME ajouter lien pour la population dm_population
    else:
        if len(interface.list_link()) != 0:
            print('You are missing lifestyles to agriculture interface')
        DM_lfs = simulate_lifestyles_to_agriculture_input_new()
        for key in DM_lfs.keys():
            DM_lfs[key].filter({'Country': cntr_list}, inplace=True)

    if interface.has_link(from_sector='buildings', to_sector='agriculture'):
        dm_bld = interface.get_link(from_sector='buildings', to_sector='agriculture')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing buildings to agriculture interface')
        dm_bld = simulate_buildings_to_agriculture_input()
        dm_bld.filter({'Country': cntr_list}, inplace=True)

    if interface.has_link(from_sector='industry', to_sector='agriculture'):
        DM_ind = interface.get_link(from_sector='industry', to_sector='agriculture')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing industry to agriculture interface')
        DM_ind = simulate_industry_to_agriculture_input()
        for key in DM_ind.keys():
            DM_ind[key].filter({'Country': cntr_list}, inplace=True)

    if interface.has_link(from_sector='transport', to_sector='agriculture'):
        dm_tra = interface.get_link(from_sector='transport', to_sector='agriculture')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing transport to agriculture interface')
        dm_tra = simulate_transport_to_agriculture_input()
        dm_tra.filter({'Country': cntr_list}, inplace=True)

    # CalculationTree AGRICULTURE

    dm_lfs, df_cal_rates_diet = lifestyle_workflow(DM_lifestyle, DM_lfs, CDM_const, years_setting)
    dm_lfs, dm_lfs_pro = food_demand_workflow(DM_food_demand, dm_lfs)
    DM_livestock, dm_liv_ibp, dm_liv_ibp, dm_liv_prod, dm_liv_pop, df_cal_rates_liv_prod, df_cal_rates_liv_pop = livestock_workflow(
        DM_livestock, CDM_const, dm_lfs_pro, years_setting)
    DM_alc_bev, dm_bev_ibp_cereal_feed, dm_bev_dom_prod = alcoholic_beverages_workflow(DM_alc_bev, CDM_const,
                                                                                       dm_lfs_pro)
    DM_bioenergy, dm_oil, dm_lgn, dm_eth, dm_biofuel_fdk = bioenergy_workflow(DM_bioenergy, CDM_const, DM_ind, dm_bld,
                                                                              dm_tra)
    dm_liv_N2O, dm_CH4, df_cal_rates_liv_N2O, df_cal_rates_liv_CH4, DM_manure = livestock_manure_workflow(DM_manure, DM_livestock,
                                                                                               dm_liv_pop, CDM_const,
                                                                                               years_setting)
    DM_feed, dm_aps_ibp, dm_feed_req, dm_aps, dm_feed_demand, df_cal_rates_feed = feed_workflow(DM_feed, dm_liv_prod,
                                                                                                dm_bev_ibp_cereal_feed,
                                                                                                CDM_const,
                                                                                                years_setting)
    dm_voil, dm_aps_ibp_oil, dm_voil_tpe = biomass_allocation_workflow(dm_aps_ibp, dm_oil)
    DM_crop, dm_crop, dm_crop_other, dm_feed_processed, dm_food_processed, df_cal_rates_crop, DM_ssr = crop_workflow(DM_crop,
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
                                                                                                             years_setting)
    DM_land, dm_land, dm_land_use, dm_fiber, df_cal_rates_land = land_workflow(DM_land, DM_crop, DM_livestock,
                                                                               dm_crop_other, DM_ind, years_setting)
    dm_n, dm_fertilizer_co, dm_mineral_fertilizer, df_cal_rates_n = nitrogen_workflow(DM_nitrogen, dm_land, CDM_const,
                                                                                      years_setting)
    DM_energy_ghg, dm_CO2, dm_input_use_CO2, dm_crop_residues, dm_CH4_liv_tpe, dm_N2O_liv_tpe, dm_CH4_rice, dm_fertilizer_N2O, df_cal_rates_ghg = energy_ghg_workflow(
        DM_energy_ghg, DM_crop, DM_land, DM_manure, dm_land, dm_fertilizer_co, dm_liv_N2O, dm_CH4, CDM_const, dm_n,
        years_setting)

    # INTERFACES OUT ---------------------------------------------------------------------------------------------------

    # interface to Land use
    DM_lus = agriculture_landuse_interface(DM_bioenergy, dm_lgn, dm_land_use)
    interface.add_link(from_sector='agriculture', to_sector='land-use', dm=DM_lus)

    # interface to Emissions
    dm_ems = agriculture_emissions_interface(DM_nitrogen, dm_CO2, DM_crop, DM_manure, DM_land, dm_input_use_CO2,
                                             dm_crop_residues, dm_CH4, dm_liv_N2O, dm_CH4_rice, dm_fertilizer_N2O,
                                             write_xls=False)
    interface.add_link(from_sector='agriculture', to_sector='emissions', dm=dm_ems)

    # interface to Ammonia
    dm_ammonia = agriculture_ammonia_interface(dm_mineral_fertilizer)
    interface.add_link(from_sector='agriculture', to_sector='ammonia', dm=dm_ammonia)

    # interface to Oil Refinery
    dm_ref = agriculture_refinery_interface(DM_energy_ghg)
    interface.add_link(from_sector='agriculture', to_sector='oil-refinery', dm=dm_ref)

    # # interface to Storage
    # dm_storage = agriculture_storage_interface(DM_energy_ghg, write_xls=False)
    # interface.add_link(from_sector='agriculture', to_sector='power', dm=dm_storage)

    # interface to Power
    DM_pow = agriculture_power_interface(DM_energy_ghg, DM_bioenergy)
    interface.add_link(from_sector='agriculture', to_sector='power', dm=DM_pow)

    # interface to Minerals
    dm_minerals = agriculture_minerals_interface(DM_nitrogen, DM_bioenergy, dm_lgn)
    interface.add_link(from_sector='agriculture', to_sector='minerals', dm=dm_minerals)

    # TPE OUTPUT -------------------------------------------------------------------------------------------------------
    results_run = agriculture_TPE_interface(DM_livestock, DM_crop, dm_crop_other, DM_feed, dm_aps, dm_input_use_CO2,
                                            dm_crop_residues, dm_CH4, dm_liv_N2O, dm_CH4_rice, dm_fertilizer_N2O,
                                            DM_energy_ghg, DM_bioenergy, dm_lgn, dm_eth, dm_oil, dm_aps_ibp,
                                            DM_food_demand, dm_lfs_pro, dm_lfs, DM_land, dm_fiber, dm_aps_ibp_oil,
                                            dm_voil_tpe, DM_alc_bev, dm_biofuel_fdk, dm_liv_pop, DM_ssr, dm_fertilizer_co,DM_manure)

    return results_run


def agriculture_local_run():
    country_list = ['Switzerland', 'Vaud']
    DM_input = filter_country_and_load_data_from_pickles(country_list= country_list, modules_list = 'agriculture')
    years_setting, lever_setting = init_years_lever()
    agriculture(lever_setting, years_setting, DM_input['agriculture'])
    return


#agriculture_local_run()
