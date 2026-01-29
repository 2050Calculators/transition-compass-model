import pandas as pd

from model.common.data_matrix_class import DataMatrix
from model.common.constant_data_matrix_class import ConstantDataMatrix
from model.common.io_database import dm_to_database
from model.common.interface_class import Interface
from model.common.auxiliary_functions import  calibration_rates, create_years_list, linear_forecast_BAU
from model.common.auxiliary_functions import read_level_data, filter_country_and_load_data_from_pickles, my_pickle_dump
import pickle
import json
import os
import numpy as np
import time


def init_years_lever():
    # function that can be used when running the module as standalone to initialise years and levers
    years_setting = [1990, 2023, 2025, 2050, 5]
    f = open('../config/lever_position.json')
    lever_setting = json.load(f)[0]
    return years_setting, lever_setting


# CalculationLeaf READ PICKLE
def read_data(DM_diet_input, lever_setting, tpe_scenario):

    # Read fts based on lever_setting
    # FIXME error it adds ots and fts
    # DM_check = check_ots_fts_match(DM_agriculture, lever_setting)
    DM_ots_fts = read_level_data(DM_diet_input, lever_setting)

    # Sub-matrix for DIETARY HABITS
    dm_diet_requirement = DM_ots_fts['kcal-req']
    dm_diet_split_share = DM_ots_fts['diet-split-share']
    dm_diet_fwaste = DM_ots_fts['fwaste']
    dm_fxa_cal_diet = DM_diet_input['fxa']['cal_agr_diet']
    dm_diet_adherence = DM_ots_fts['diet-adherence']
    if tpe_scenario == 'diet-split-share':
      dm_share_pro_food = DM_ots_fts['share-processed-food_crop-cereal-whole']
      dm_share_pro_food_meat = DM_ots_fts[
        'share-processed-food_unprocessed-meat']
      dm_share_pro_food.append(dm_share_pro_food_meat, dim='Variables')
    elif tpe_scenario == 'diet-split-kcal':
      dm_share_pro_food = DM_ots_fts['share-kcal-processed-food_crop-cereal-whole']
      dm_share_pro_food_meat = DM_ots_fts[
        'share-kcal-processed-food_unprocessed-meat']
      dm_share_pro_food.append(dm_share_pro_food_meat, dim='Variables')

    # list of lever names
    levers = [
      "diet-split-kcal_crop-cereal",
      "diet-split-kcal_crop-rice",
      "diet-split-kcal_crop-fruit",
      "diet-split-kcal_crop-oilcrop",
      "diet-split-kcal_crop-pulse",
      "diet-split-kcal_crop-starch",
      "diet-split-kcal_crop-veg",
      "diet-split-kcal_pro-bev-beer",
      "diet-split-kcal_pro-bev-bev-alc",
      "diet-split-kcal_pro-bev-bev-fer",
      "diet-split-kcal_pro-bev-wine",
      "diet-split-kcal_pro-crop-processed-sugar",
      "diet-split-kcal_pro-crop-processed-sweet",
      "diet-split-kcal_pro-crop-processed-voil",
      "diet-split-kcal_pro-liv-abp-dairy-milk",
      "diet-split-kcal_pro-liv-abp-hens-egg",
      "diet-split-kcal_pro-liv-abp-processed-afat",
      "diet-split-kcal_pro-liv-abp-processed-offal",
      "diet-split-kcal_pro-liv-meat-bovine",
      "diet-split-kcal_pro-liv-meat-oth-animal",
      "diet-split-kcal_pro-liv-meat-pig",
      "diet-split-kcal_pro-liv-meat-poultry",
      "diet-split-kcal_pro-liv-meat-sheep",
      "diet-split-kcal_seafood-dfish",
      "diet-split-kcal_seafood-ffish",
      "diet-split-kcal_seafood-oth-aq-animal",
      "diet-split-kcal_seafood-pfish",
      "diet-split-kcal_seafood-seafood",
      "diet-split-kcal_stm-cocoa",
      "diet-split-kcal_stm-coffee",
      "diet-split-kcal_stm-tea"
    ]

    # 1: Create a dictionary of all DataMatrix objects
    dm_diet_split_kcal = {lever: DM_ots_fts[lever] for lever in levers}

    # 2: Merge them all into one DataMatrix along 'Variables'
    dm_diet_split_kcal_merged = None

    for lever_name, dm_split in dm_diet_split_kcal.items():
      # ensure each variable label is unique
      dm_split.col_labels['Variables'] = ['lfs_consumers-diet']
      #dm_split.rename_col([lever_name], 'lfs_consumers-diet', 'Variables')

      if dm_diet_split_kcal_merged is None:
        dm_diet_split_kcal_merged = dm_split
      else:
        dm_diet_split_kcal_merged.append(dm_split, dim='Categories1')


    # Sub-matrix for ALCOHOLIC BEVERAGES
    #dm_alc_bev = DM_ots_fts['biomass-hierarchy']['biomass-hierarchy-bev-ibp-use-oth']
    dm_processing_yield = DM_diet_input['fxa']['processing-yield']
    dm_bev_ssr = DM_ots_fts['ssr-bev']
    dm_cal_crop_bev = DM_diet_input['fxa']['cal_agr_domestic-production_bev']

    # Aggregate Data Matrix - DIETARY HABITS
    DM_diet = {
        'energy-requirement': dm_diet_requirement,
        'diet-split-share': dm_diet_split_share,
        'diet-split-kcal': dm_diet_split_kcal_merged,
        'diet-fwaste': dm_diet_fwaste,
        'cal_diet': dm_fxa_cal_diet,
        'diet-adherence': dm_diet_adherence,
        'share-processed-food': dm_share_pro_food
    }

    # Aggregated Data Matrix - ALCOHOLIC BEVERAGES
    DM_alc_bev = {
        #'biomass_hierarchy': dm_alc_bev,
        'processing-yields': dm_processing_yield,
        'ssr-bev': dm_bev_ssr,
        'cal_bev': dm_cal_crop_bev
    }


    CDM_const = DM_diet_input['constant']

    return DM_ots_fts, DM_diet, DM_alc_bev, CDM_const


# SimulateInteractions
def simulate_population_to_dietaryhabits_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../_database/data/interface/population_to_dietary-habits.pickle")
    with open(f, 'rb') as handle:
        DM_pop = pickle.load(handle)

    return DM_pop

# CalculationLeaf DIET WITH ADHERENCE SCENARIO --------------------------------------------------------------
def diet_adherence_scenarios(DM_diet, DM_pop, CDM_const, bau, tpe_scenario):

  # Processing for diet-adherence
  dm_adherence = DM_diet['diet-adherence'].copy()
  if bau == True :
    dm_adherence[:,:,'share_diet_adherence'] = 1 - dm_adherence[:,:,'share_diet_adherence']
    var_consumers_diet = 'lfs_consumers-diet_bau'
    var_kcal_req = 'agr_kcal-req_bau'
  else :
    dm_adherence[:, :, 'share_diet_adherence'] = dm_adherence[:, :,'share_diet_adherence']
    var_consumers_diet = 'lfs_consumers-diet'
    var_kcal_req = 'agr_kcal-req'

  dm_diet_food = {}
  if tpe_scenario == 'diet-split-share':
    # Average kcal-req [kcal/cap/day] = sum(demography [inhabitants] * kcal-req by demography [kcal/cap/day]) / sum(demography [inhabitants])
    dm_diet_requirement = DM_diet['energy-requirement'].copy()
    dm_diet_requirement.append(DM_pop['lfs_demography_'], dim='Variables')
    dm_diet_requirement.operation('lfs_demography', '*', var_kcal_req,
                                  out_col='lfs_kcal-req', unit='kcal/day')
    dm_diet_requirement.group_all('Categories1')
    dm_diet_requirement.operation('lfs_kcal-req', '/', 'lfs_demography',
                                  out_col='lfs_kcal-req_req',
                                  unit='kcal/cap/day')
    dm_diet_requirement.filter({'Variables': ['lfs_kcal-req_req']},
                               inplace=True)

    #  Intake of food categories i [kcal/cap/day] = kcal-req [kcal/cap/day] * share of i [%]
    dm_diet_split = DM_diet['diet-split-share'].copy()
    ay_total_diet = dm_diet_requirement[:, :, 'lfs_kcal-req_req', np.newaxis] * \
                    dm_diet_split[:, :, var_consumers_diet, :]
    dm_diet_food = DataMatrix.based_on(ay_total_diet[:, :, np.newaxis, :],
                                       dm_diet_split,
                                       change={'Variables': [var_consumers_diet]},
                                       units={var_consumers_diet: 'kcal/cap/day'})

  elif tpe_scenario =='diet-split-kcal':
    # Format
    dm_diet_food = DM_diet['diet-split-kcal'].filter({'Variables':[var_consumers_diet]}).copy()

  # Intake of food categories i [kcal/country/year] = Intake of food cat i [kcal/cap/cay] * diet adherence [%] * pop * days per year
  cdm_lifestyle = CDM_const['cdm_lifestyle'].copy()
  dm_population = DM_pop['lfs_population_'].copy()
  ay_total_diet = dm_population[:, :, 'lfs_population_total', np.newaxis] * \
                  dm_diet_food[:, :, var_consumers_diet, :] * cdm_lifestyle['cp_time_days-per-year'] \
                  * dm_adherence[:,:,'share_diet_adherence', np.newaxis]
  dm_diet_food.add(ay_total_diet, dim='Variables',
                       col_label='lfs_diet_raw',
                       unit='kcal')

  """# Intake of food categories i [kcal/country/year] = kcal-req [kcal/cap/day] * diet adherence [%] * share of i [%] * pop * days per year
  cdm_lifestyle = CDM_const['cdm_lifestyle'].copy()
  dm_diet_split = DM_diet['diet-split-share'].copy()
  dm_population = DM_pop['lfs_population_'].copy()
  ay_total_diet = dm_diet_requirement[:, :, 'lfs_kcal-req_req', np.newaxis] * \
                  dm_population[:, :, 'lfs_population_total', np.newaxis] * \
                  dm_diet_split[:, :, var_consumers_diet, :] * cdm_lifestyle[
                    'cp_time_days-per-year'] * dm_adherence[:,:,'share_diet_adherence', np.newaxis]
  dm_diet_food = DataMatrix.based_on(ay_total_diet[:, :, np.newaxis, :],
                                    dm_diet_split,
                                    change={'Variables': ['lfs_diet_raw']},
                                    units={'lfs_diet_raw': 'kcal'})"""

  # Total calorie demand [kcal/country/year] = food intake [kcal/country/year] / food waste [-]
  #dm_diet_food.append(dm_diet_tmp,
  #                    dim='Categories1')  # Append all food categories
  dm_diet_food.append(DM_diet['diet-fwaste'],
                      dim='Variables')  # Append with fwaste
  dm_diet_food.operation('lfs_diet_raw', '/', 'lfs_consumers-food-wastes',
                         dim='Variables', out_col='agr_demand_raw',
                         unit='kcal')

  # Food waste [kcal/country/year] = total calorie demand - food intake
  dm_diet_food.operation('agr_demand_raw', '-', 'lfs_diet_raw', dim='Variables',
                         out_col='lfs_food-wastes',
                         unit='kcal')

  # Format the diet for lever and health assessment --------------------------
  # create copy for what is actually consumed
  dm_diet_consumed = dm_diet_food.filter({'Variables': [
    'lfs_diet_raw']}).copy()
  # Unit conversion: [kcal/country/year] => [kcal/cap/day]
  dm_population = DM_pop['lfs_population_']
  array_temp = dm_diet_consumed[:, :, 'lfs_diet_raw', :] / \
               dm_population[:, :, 'lfs_population_total', np.newaxis] / 365.25
  dm_diet_consumed.add(array_temp, dim='Variables',
                       col_label='lfs_diet_raw_cap',
                       unit='kcal/cap/day')

  # Unit conversion: [kcal/cap/day] => [g/cap/day]
  dm_diet_consumed.sort('Categories1')
  cdm_kcal = CDM_const['cdm_kcal-per-t'].copy()
  cdm_kcal.drop(dim='Categories1', col_label=['stm']) # to drop only stm and not stm-coffee etc
  cdm_kcal.drop(dim='Categories1', col_label='crop-sugarcrop')
  cdm_kcal.drop(dim='Categories1', col_label='pro-crop-processed-molasse')
  cdm_kcal.drop(dim='Categories1', col_label='pro-crop-processed-cake')
  cdm_kcal.drop(dim='Categories1', col_label='liv-meat-meal')
  # Sort
  # Check that categories are the same
  # set(cdm_kcal.col_labels['Categories1']) - set(dm_diet_consumed.col_labels['Categories1'])
  dm_diet_consumed.sort('Categories1')
  cdm_kcal.sort('Categories1')
  # Convert from [kcal/cap/day] to [t/cap/day]
  array_temp = dm_diet_consumed[:, :, 'lfs_diet_raw_cap', :] \
               / cdm_kcal[np.newaxis, np.newaxis, 'cp_kcal-per-t', :]
  dm_diet_consumed.add(array_temp, dim='Variables',
                       col_label='lfs_consumers-diet',
                       unit='t/cap/day')
  dm_diet_consumed = dm_diet_consumed.filter(
    {'Variables': ['lfs_consumers-diet']})
  # Convert from [t/cap/day] to [g/cap/day]
  dm_diet_consumed.change_unit('lfs_consumers-diet', factor=1e6,
                               old_unit='t/cap/day',
                               new_unit='g/cap/day')

  # Compute share of processed meat
  dm_share_pro = DM_diet['share-processed-food']
  dm_meat_tot = dm_diet_consumed.groupby({'meat-total': '.*meat.*'}, regex=True, inplace=False, dim='Categories1')
  array_temp = dm_meat_tot[:,:,'lfs_consumers-diet','meat-total'] * \
               ( 1.0 - dm_share_pro[:,:,'lfs_share_unprocessed-meat'])
  dm_diet_consumed.add(array_temp[:,:,np.newaxis, np.newaxis], dummy=False, col_label='pro-liv-meat-processed', dim='Categories1', unit='g/cap/day')

  # Compute share of crop-cereal-whole
  array_temp = dm_diet_consumed[:,:,'lfs_consumers-diet','crop-cereal'] * \
               dm_share_pro[:,:,'lfs_share_crop-cereal-whole']
  dm_diet_consumed.add(array_temp[:,:,np.newaxis, np.newaxis], dummy=False, col_label='crop-cereal-whole', dim='Categories1', unit='g/cap/day')

  """ # Note: for diet preprocessing
  # Filter years ots
  years_ots = create_years_list(1990, 2023, 1)
  dm_diet_consumed = dm_diet_consumed.filter({'Years': years_ots})

  # Format as df with variables as rows and years as columns
  df_diet_pre = dm_to_database(dm_diet_consumed, 'none', 'agriculture',
                                     level=0)
  df_diet_pre = df_diet_pre[df_diet_pre['geoscale']=='Switzerland'].copy()
  df_diet_pre = df_diet_pre[['timescale', 'variables', 'value']].copy()
  df_pivot = df_diet_pre.pivot(index="variables", columns="timescale",
                               values="value")
  df_pivot = df_pivot.reset_index().rename_axis(None, axis=1)

  # Export as excel file
  df_pivot.to_excel("TCF-Calc_diet_ots.xlsx", index=True)"""

  return dm_diet_food, dm_diet_consumed

# CalculationLeaf LIFESTYLE (SHARE) TO DIET/FOOD DEMAND --------------------------------------------------------------
def lifestyle_share_workflow(DM_diet, DM_pop, CDM_const, years_setting, tpe_scenario):

    # DIFFERENTIATE BETWEEN BAU & SCENARIOS ------------------------------------

    # Create BAU variables - diet-split-share
    array_temp = DM_diet['diet-split-share'][:, :,'lfs_consumers-diet',:]
    DM_diet['diet-split-share'].add(array_temp, dummy=False, col_label='lfs_consumers-diet_bau', dim='Variables', unit='-')

    # Extrapolate BAU fts - diet-split-share
    years_ots = create_years_list(years_setting[0], years_setting[1], 1)
    years_fts = create_years_list(years_setting[2], years_setting[3], 5)
    dm_ots_temp = DM_diet['diet-split-share'].filter({'Years':years_ots})
    dm_bau_fts = linear_forecast_BAU(dm_ots_temp, years_setting[0], years_ots, years_fts, min_tb=None,
                        max_tb=None)
    for i in years_fts:
      DM_diet['diet-split-share'][:, i, 'lfs_consumers-diet_bau', :] = dm_bau_fts[:, i, 'lfs_consumers-diet_bau', :]

    # Create BAU variables - energy-requirement
    array_temp = DM_diet['energy-requirement'][:, :,'agr_kcal-req',:]
    DM_diet['energy-requirement'].add(array_temp, dummy=False, col_label='agr_kcal-req_bau', dim='Variables', unit='kcal/cap/day')

    # Extrapolate BAU fts - energy-requirement
    dm_ots_temp = DM_diet['energy-requirement'].filter({'Years':years_ots})
    dm_bau_fts = linear_forecast_BAU(dm_ots_temp, years_setting[0], years_ots, years_fts, min_tb=None,
                        max_tb=None)
    for i in years_fts:
      DM_diet['energy-requirement'][:, i, 'agr_kcal-req_bau', :] = dm_bau_fts[:, i, 'agr_kcal-req_bau', :]

    # Computing diets considering the adherence to the diet levers
    dm_diet_food_bau, dm_diet_consumed_bau = diet_adherence_scenarios(DM_diet, DM_pop, CDM_const, bau=True, tpe_scenario=tpe_scenario)
    dm_diet_food_scenario, dm_diet_consumed_scenario = diet_adherence_scenarios(DM_diet, DM_pop, CDM_const,
                                                bau=False, tpe_scenario=tpe_scenario)

    # Overall diet = diet bau [kcal/country share/year] + diet scenario [kcal/country share/year]
    dm_diet_food = dm_diet_food_bau.copy()
    dm_diet_food[:, :, 'lfs_diet_raw', :] = dm_diet_food_bau[:, :, 'lfs_diet_raw', :] + \
                                                dm_diet_food_scenario[:, :,
                                                'lfs_diet_raw', :]

    # Overall diet consumed = diet cons bau [kcal/country share/year] + diet cons scenario [kcal/country share/year]
    # (without food wastes)
    dm_diet_consumed = dm_diet_consumed_bau.copy()
    dm_diet_consumed[:, :, 'lfs_consumers-diet', :] = dm_diet_consumed_bau[:, :, 'lfs_consumers-diet', :] + \
                                                dm_diet_consumed_scenario[:, :,
                                                'lfs_consumers-diet', :]


    # Calibration - Food supply (accounting for food wastes)
    dm_cal_diet = DM_diet['cal_diet']
    dm_lfs = dm_diet_food.filter({'Variables':['agr_demand_raw']}, inplace=False)
    dm_cal_rates_diet = calibration_rates(dm_lfs, dm_cal_diet, calibration_start_year=1990,
                                          calibration_end_year=2023,
                                          years_setting=years_setting)
    dm_lfs.append(dm_cal_rates_diet, dim='Variables')
    dm_lfs.operation('agr_demand_raw', '*', 'cal_rate', dim='Variables', out_col='agr_demand', unit='kcal')

    # Format for same categories as rest of modules
    cat_lfs = ['afat', 'beer', 'bev-alc', 'bev-fer', 'bov', 'cereals', 'coffee', 'dfish', 'egg', 'ffish', 'fruits',
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

    #dm_lfs.rename_col(cat_lfs, cat_agr, 'Categories1')
    dm_lfs.sort('Categories1')

    return dm_lfs, dm_diet_consumed, dm_diet_consumed_bau, dm_diet_consumed_scenario, dm_diet_food



# CalculationLeaf LIFESTYLE (KCAL) TO DIET/FOOD DEMAND --------------------------------------------------------------
def lifestyle_kcal_workflow(DM_diet, DM_pop, CDM_const, years_setting, tpe_scenario):

    # DIFFERENTIATE BETWEEN BAU & SCENARIOS ------------------------------------

    # Create BAU variables - diet-split-kcal
    array_temp = DM_diet['diet-split-kcal'][:, :,'lfs_consumers-diet',:]
    DM_diet['diet-split-kcal'].add(array_temp, dummy=False, col_label='lfs_consumers-diet_bau', dim='Variables', unit='kcal/cap/day')

    # Extrapolate BAU fts - diet-split-share
    years_ots = create_years_list(years_setting[0], years_setting[1], 1)
    years_fts = create_years_list(years_setting[2], years_setting[3], 5)
    dm_ots_temp = DM_diet['diet-split-kcal'].filter({'Years':years_ots})
    dm_bau_fts = linear_forecast_BAU(dm_ots_temp, years_setting[0], years_ots, years_fts, min_tb=None,
                        max_tb=None)
    for i in years_fts:
      DM_diet['diet-split-kcal'][:, i, 'lfs_consumers-diet_bau', :] = dm_bau_fts[:, i, 'lfs_consumers-diet_bau', :]

    # Computing diets considering the adherence to the diet levers
    dm_diet_food_bau, dm_diet_consumed_bau = diet_adherence_scenarios(DM_diet, DM_pop, CDM_const, bau=True, tpe_scenario=tpe_scenario)
    dm_diet_food_scenario, dm_diet_consumed_scenario = diet_adherence_scenarios(DM_diet, DM_pop, CDM_const,
                                                bau=False, tpe_scenario=tpe_scenario)

    # Overall diet = diet bau [kcal/country share/year] + diet scenario [kcal/country share/year]
    dm_diet_food = dm_diet_food_bau.copy()
    dm_diet_food[:, :, 'lfs_diet_raw', :] = dm_diet_food_bau[:, :, 'lfs_diet_raw', :] + \
                                                dm_diet_food_scenario[:, :,
                                                'lfs_diet_raw', :]

    # Overall diet consumed = diet cons bau [kcal/country share/year] + diet cons scenario [kcal/country share/year]
    # (without food wastes)
    dm_diet_consumed = dm_diet_consumed_bau.copy()
    dm_diet_consumed[:, :, 'lfs_consumers-diet', :] = dm_diet_consumed_bau[:, :, 'lfs_consumers-diet', :] + \
                                                dm_diet_consumed_scenario[:, :,
                                                'lfs_consumers-diet', :]


    # Calibration - Food supply (accounting for food wastes)
    dm_cal_diet = DM_diet['cal_diet']
    dm_lfs = dm_diet_food.filter({'Variables':['agr_demand_raw']}, inplace=False)
    dm_cal_rates_diet = calibration_rates(dm_lfs, dm_cal_diet, calibration_start_year=1990,
                                          calibration_end_year=2023,
                                          years_setting=years_setting)
    dm_lfs.append(dm_cal_rates_diet, dim='Variables')
    dm_lfs.operation('agr_demand_raw', '*', 'cal_rate', dim='Variables', out_col='agr_demand', unit='kcal')

    # Format for same categories as rest of modules
    cat_lfs = ['afat', 'beer', 'bev-alc', 'bev-fer', 'bov', 'cereals', 'coffee', 'dfish', 'egg', 'ffish', 'fruits',
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

    #dm_lfs.rename_col(cat_lfs, cat_agr, 'Categories1')
    dm_lfs.sort('Categories1')

    return dm_lfs, dm_diet_consumed, dm_diet_consumed_bau, dm_diet_consumed_scenario, dm_diet_food

# CalculationLeaf ALCOHOLIC BEVERAGES INDUSTRY -------------------------------------------------------------------------
def alcoholic_beverages_workflow(DM_alc_bev, CDM_const, dm_lfs, years_setting):
    # Filtering dms to only keep pro
    dm_demand_bev = dm_lfs.filter_w_regex({'Categories1': 'pro-bev.*', 'Variables': 'agr_demand'})

    # Filtering dms to only keep pro
    food_net_import_pro = DM_alc_bev['ssr-bev'].filter_w_regex(
        {'Categories1': 'pro-bev.*', 'Variables': 'agr_ssr'})
    # Sorting the dms alphabetically
    food_net_import_pro.sort(dim='Categories1')
    dm_demand_bev.sort(dim='Categories1')

    # Domestic production processed food [kcal] = agr_demand_pro_(.*) [kcal] * net-imports_pro_(.*) [-]
    array_agr_domestic_production = dm_demand_bev[:, :, 'agr_demand', :] \
                              * food_net_import_pro[:, :, 'agr_ssr']

    # Adding agr_domestic_production to dm_lfs_pro
    dm_demand_bev.add(array_agr_domestic_production, dim='Variables', col_label='agr_domestic_production', unit='kcal')

    # Filter domestic production bev and rename
    # Beer
    dm_bev_beer = dm_demand_bev.filter_w_regex({'Categories1': 'pro-bev-beer.*', 'Variables': 'agr_domestic_production'})
    dm_bev_beer.rename_col_regex(str1="pro-bev-", str2="", dim="Categories1")
    dm_bev_beer = dm_bev_beer.flatten()
    # Bev-alc
    dm_bev_alc = dm_demand_bev.filter_w_regex({'Categories1': 'pro-bev-bev-alc.*', 'Variables': 'agr_domestic_production'})
    dm_bev_alc.rename_col_regex(str1="pro-bev-", str2="", dim="Categories1")
    dm_bev_alc = dm_bev_alc.flatten()
    # Bev-fer
    dm_bev_fer = dm_demand_bev.filter_w_regex({'Categories1': 'pro-bev-bev-fer.*', 'Variables': 'agr_domestic_production'})
    dm_bev_fer.rename_col_regex(str1="pro-bev-", str2="", dim="Categories1")
    dm_bev_fer = dm_bev_fer.flatten()
    # Wine
    dm_bev_wine = dm_demand_bev.filter_w_regex({'Categories1': 'pro-bev-wine.*', 'Variables': 'agr_domestic_production'})
    dm_bev_wine.rename_col_regex(str1="pro-bev-", str2="", dim="Categories1")
    dm_bev_wine = dm_bev_wine.flatten()

    # Constants and sorting according to bev type (beer, wine, bev-alc, bev-fer)
    cdm_cp_ibp_bev_beer = CDM_const['cdm_cp_ibp_bev_beer']
    cdm_cp_ibp_bev_wine = CDM_const['cdm_cp_ibp_bev_wine']
    cdm_cp_ibp_bev_alc = CDM_const['cdm_cp_ibp_bev_bev-alc']
    cdm_cp_ibp_bev_fer = CDM_const['cdm_cp_ibp_bev_bev-fer']

    # FRUIT & CEREAL DEMAND FOR BEVERAGES ------------------------------------------------------------------------------

    # Crop demand [kcal] = domestic production bev [kcal] * processing yield [input kcal/output kcal]

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
    array_temp = dm_bev_wine[:, :, 'agr_domestic_production_wine'] \
                                  * DM_alc_bev['processing-yields'][:,:, 'fxa_agr_processing-yield', 'wine-to-fruit']
    dm_bev_wine.add(array_temp, dim='Variables', col_label='agr_ibp_bev_wine_crop_fruit', unit='kcal')

    # Append together
    dm_bev_dom_prod = dm_bev_beer.copy()
    dm_bev_dom_prod.append(dm_bev_alc, dim='Variables')
    dm_bev_dom_prod.append(dm_bev_fer, dim='Variables')
    dm_bev_dom_prod.append(dm_bev_wine, dim='Variables')

    # Cereals domestic production for beverages = cereals for beer + cereals for bev fer
    dm_bev_dom_prod.operation('agr_ibp_bev_beer_crop_cereal', '+',
                              'agr_ibp_bev_bev-fer_crop_cereal',
                              out_col='agr_domestic-production_bev_raw_cereal', unit='kcal')

    # Fruit domestic production for beverages = fruits for bev-alc + fruits for wine
    dm_bev_dom_prod.operation('agr_ibp_bev_bev-alc_crop_fruit', '+',
                              'agr_ibp_bev_wine_crop_fruit',
                              out_col='agr_domestic-production_bev_raw_fruit', unit='kcal')

    # Filter and deepen
    dm_bev_dom_prod = dm_bev_dom_prod.filter({'Variables': ['agr_domestic-production_bev_raw_cereal',
                                                            'agr_domestic-production_bev_raw_fruit']})
    dm_bev_dom_prod.deepen()

    # (CH only) CALIBRATION CROP PRODUCTION BEVERAGES --------------------------------------------------------------------------------------
    dm_cal_rates_bev = calibration_rates(dm_bev_dom_prod, DM_alc_bev['cal_bev'], calibration_start_year=1990,
                                          calibration_end_year=2023, years_setting=years_setting)
    dm_bev_dom_prod.append(dm_cal_rates_bev, dim='Variables')
    dm_bev_dom_prod.operation('agr_domestic-production_bev_raw', '*', 'cal_rate', dim='Variables',
                          out_col='agr_domestic-production_bev', unit='kcal')
    dm_bev_dom_prod.filter({'Variables': ['agr_domestic-production_bev']}, inplace=True)

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
                                  * cdm_cp_ibp_bev_beer.array[idx_cdm_ibp_beer['cp_ibp_bev_beer_brf_fdk_crop_cereal']]
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

    """# Byproducts biomass use per sector = byproducts for other uses * share of bev biomass per sector [%]
    idx_bev_ibp_use_oth = dm_bev_ibp_use_oth.idx
    idx_bev_biomass_hierarchy = DM_alc_bev['biomass_hierarchy'].idx
    agr_bev_ibp_use_oth = dm_bev_ibp_use_oth.array[:, :, idx_bev_ibp_use_oth['agr_bev_ibp_use_oth'], np.newaxis] * \
                          DM_alc_bev['biomass_hierarchy'].array[:, :,
                          idx_bev_biomass_hierarchy['agr_biomass-hierarchy-bev-ibp-use-oth'], :]
    DM_alc_bev['biomass_hierarchy'].add(agr_bev_ibp_use_oth, dim='Variables', col_label='agr_bev_ibp_use_oth',
                                        unit='kcal')"""

    # Cereal bev byproducts allocated to feed [kcal] = sum (beer byproducts for feedstock [kcal])
    dm_bev_beer.operation('agr_ibp_bev_beer_fdk_yeast', '+',
                          'agr_ibp_bev_beer_fdk_cereal',
                          out_col='agr_use_bev_ibp_cereal_feed', unit='kcal')
    dm_bev_ibp_cereal_feed = dm_bev_beer.filter({'Variables': ['agr_use_bev_ibp_cereal_feed']})

    # Unit conversion: [kcal] to [t]
    # Filter
    cdm_kcal = CDM_const['cdm_kcal-per-t'].copy()
    cdm_kcal = cdm_kcal.filter({'Categories1': ['crop-cereal']})
    cdm_kcal = cdm_kcal.flatten()

    # Convert from [kcal] to [t]
    array_temp = dm_bev_ibp_cereal_feed[:, :, 'agr_use_bev_ibp_cereal_feed'] \
                 / cdm_kcal[np.newaxis, np.newaxis, 'cp_kcal-per-t_crop-cereal']
    dm_bev_ibp_cereal_feed.add(array_temp, dim='Variables',
                    col_label='agr_use_bev_ibp_cereal_feed_t',
                    unit='t')

    # (Not used after) Fruits bev allocated to non-food [kcal] = dom prod bev alc + dom prod bev wine + bev byproducts for fertilizer

    # (Not used after) Cereals bev allocated to non-food [kcal] = dom prod bev beer + dom prod bev fer + bev byproducts for fertilizer
    # change the double count of bev byproducts for fertilizer in fruits/cereals bev allocated to non-food [kcal]

    # (Not used after) Fruits bev allocated to bioenergy [kcal] = bp bev for solid bioenergy (+ bp use for ethanol (not found in knime))
    return DM_alc_bev, dm_bev_ibp_cereal_feed, dm_bev_dom_prod


# CalculationLeaf INTERFACE TO TPE  --------------------------------------------------------------
def dietaryhabits_TPE_interface(CDM_const, dm_lfs, dm_diet_consumed, dm_diet_food):

    # DIET (CONSUMED, WITHOUT FOOD WASTES) -------------------------------------

    # Flatten for TPE
    dm_tpe = dm_diet_consumed.flattest()

    # DIET (WITH FOOD WASTES) --------------------------------------------------

    # Filter
    dm_supply = dm_lfs.filter({'Variables': ['agr_demand']})
    cdm_kcal = CDM_const['cdm_kcal-per-t'].copy()
    cdm_kcal.drop(dim='Categories1', col_label='crop-sugarcrop')
    cdm_kcal.drop(dim='Categories1', col_label=['stm']) # to drop only stm and not stm-coffee etc
    cdm_kcal.drop(dim='Categories1', col_label='pro-crop-processed-molasse')
    cdm_kcal.drop(dim='Categories1', col_label='pro-crop-processed-cake')
    cdm_kcal.drop(dim='Categories1', col_label='liv-meat-meal')

    # Sort
    dm_supply.sort('Categories1')
    cdm_kcal.sort('Categories1')

    # Convert from [kcal] to [t]
    array_temp = dm_supply[:, :, 'agr_demand', :] \
                 / cdm_kcal[np.newaxis, np.newaxis, 'cp_kcal-per-t', :]
    dm_supply.add(array_temp, dim='Variables', col_label='agr_demand_tpe',
                                       unit='t')
    dm_supply = dm_supply.filter({'Variables': ['agr_demand_tpe', 'agr_demand']})

    # Append for TPE
    dm_tpe.append(dm_supply.flattest(), dim='Variables')

    # FOOD WASTE ---------------------------------------------------------------

    # Filter
    dm_foodwaste = dm_diet_food.filter({'Variables': ['lfs_food-wastes']})
    cdm_kcal = CDM_const['cdm_kcal-per-t'].copy()
    cdm_kcal.drop(dim='Categories1', col_label='crop-sugarcrop')
    cdm_kcal.drop(dim='Categories1', col_label=['stm']) # to drop only stm and not stm-coffee etc
    cdm_kcal.drop(dim='Categories1', col_label='pro-crop-processed-molasse')
    cdm_kcal.drop(dim='Categories1', col_label='pro-crop-processed-cake')
    cdm_kcal.drop(dim='Categories1', col_label='liv-meat-meal')

    # Sort
    dm_foodwaste.sort('Categories1')
    cdm_kcal.sort('Categories1')

    # Convert from [kcal] to [t]
    array_temp = dm_foodwaste[:, :, 'lfs_food-wastes', :] \
                 / cdm_kcal[np.newaxis, np.newaxis, 'cp_kcal-per-t', :]
    dm_foodwaste.add(array_temp, dim='Variables', col_label='lfs_food-wastes_tpe',
                                       unit='t')
    dm_foodwaste = dm_foodwaste.filter({'Variables': ['lfs_food-wastes_tpe']})

    # Append for TPE
    dm_tpe.append(dm_foodwaste.flattest(), dim='Variables')

    # LAND USE ------------------------------------------------------

    return dm_tpe


# CalculationLeaf INTERFACE OUT TCAF  --------------------------------------------------------------
def dietaryhabits_TCAF_interface(dm_diet_consumed_bau, dm_diet_consumed_scenario):

  # Filter
  dm_diet_consumed_bau.filter({'Variables':['lfs_consumers-diet']}, inplace=True)
  dm_diet_consumed_scenario.filter({'Variables': ['lfs_consumers-diet']},
                              inplace=True)

  # Aggregate in DM
  DM_TCAF_health_diet = {"diet-consumed_bau": dm_diet_consumed_bau,
                         "diet-consumed_scenario": dm_diet_consumed_scenario}

  return DM_TCAF_health_diet


def dietaryhabits(lever_setting, years_setting, DM_input, tpe_scenario, write_pickle, interface=Interface()):

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    DM_ots_fts, DM_diet, DM_alc_bev, CDM_const = read_data(DM_input, lever_setting, tpe_scenario)
    country_list = ['Switzerland']

    # INTERFACES IN ---------------------------------------------------------------------------------------------------

    # Link interface or Simulate data from other modules
    if interface.has_link(from_sector='lifestyles', to_sector='dietary-habits'):
        DM_pop = interface.get_link(from_sector='lifestyles', to_sector='dietary-habits')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing lifestyles to dietary-habits interface')
        DM_pop = simulate_population_to_dietaryhabits_input()
        for key in DM_pop.keys():
            DM_pop[key].filter({'Country': country_list}, inplace=True)

    # CalculationTree DIETARY HABITS

    dm_diet_consumed_bau = {}
    dm_diet_consumed_scenario = {}

    if tpe_scenario == 'diet-split-share':
      dm_lfs, dm_diet_consumed, dm_diet_consumed_bau, dm_diet_consumed_scenario, dm_diet_food = lifestyle_share_workflow(
        DM_diet, DM_pop, CDM_const, years_setting, tpe_scenario=tpe_scenario)
    elif tpe_scenario == 'diet-split-kcal':
      dm_lfs, dm_diet_consumed, dm_diet_consumed_bau, dm_diet_consumed_scenario, dm_diet_food = lifestyle_kcal_workflow(
        DM_diet, DM_pop, CDM_const, years_setting, tpe_scenario=tpe_scenario)

    DM_alc_bev, dm_bev_ibp_cereal_feed, dm_bev_dom_prod = alcoholic_beverages_workflow(DM_alc_bev, CDM_const, dm_lfs, years_setting,)

    # INTERFACES OUT ---------------------------------------------------------------------------------------------------

    # Dietary Habits to TCAF
    DM_TCAF_health_diet = dietaryhabits_TCAF_interface(dm_diet_consumed_bau, dm_diet_consumed_scenario)
    if write_pickle is True:
      current_file_directory = os.path.dirname(os.path.abspath(__file__))
      f = os.path.join(current_file_directory,
                       '../_database/data/interface/dietary-habits_to_TCAF.pickle')
      with open(f, 'wb') as handle:
        pickle.dump(DM_TCAF_health_diet, handle, protocol=pickle.HIGHEST_PROTOCOL)
    interface.add_link(from_sector='dietary-habits', to_sector='TCAF', dm=DM_TCAF_health_diet)
        # pour update un pickle qui existe déjà, par exemple pour gagner du temps au pre-processing,
        # Pour remplacer des valeurs dans la même structure. Accepete un pays différent
        #my_pickle_dump(DM_new=DM_TCAF_health_diet, local_pickle_file=f)


    # Dietary Habits to Livestock
    dm_demand = dm_lfs.filter({'Variables':['agr_demand']}, inplace=False)
    DM_diet_livestock = {'demand': dm_demand,
                         'bev_feed': dm_bev_ibp_cereal_feed}
    if write_pickle is True:
      current_file_directory = os.path.dirname(os.path.abspath(__file__))
      f = os.path.join(current_file_directory,
                       '../_database/data/interface/dietary-habits_to_livestock.pickle')
      with open(f, 'wb') as handle:
        pickle.dump(DM_diet_livestock, handle, protocol=pickle.HIGHEST_PROTOCOL)
    interface.add_link(from_sector='dietary-habits', to_sector='livestock', dm=dm_demand)

    # Dietary Habits to Crop
    dm_demand = dm_lfs.filter({'Variables':['agr_demand']}, inplace=False)
    DM_diet_crop = {'demand': dm_demand,
                    'crop_bev': dm_bev_dom_prod}
    if write_pickle is True:
      current_file_directory = os.path.dirname(os.path.abspath(__file__))
      f = os.path.join(current_file_directory,
                       '../_database/data/interface/dietary-habits_to_crop.pickle')
      with open(f, 'wb') as handle:
        pickle.dump(DM_diet_crop, handle, protocol=pickle.HIGHEST_PROTOCOL)
    interface.add_link(from_sector='dietary-habits', to_sector='crop', dm=dm_demand)

    # TPE OUTPUT -------------------------------------------------------------------------------------------------------
    results_run = dietaryhabits_TPE_interface(CDM_const, dm_lfs, dm_diet_consumed, dm_diet_food)

    return results_run


def dietaryhabits_local_run():
    country_list = ['Switzerland']
    DM_input = filter_country_and_load_data_from_pickles(country_list= country_list, modules_list = 'dietary-habits', filter_country=True)
    years_setting, lever_setting = init_years_lever()
    dietaryhabits(lever_setting, years_setting, DM_input['dietary-habits'], tpe_scenario='diet-split-kcal', write_pickle=True)


if __name__ == "__main__":
  dietaryhabits_local_run()
