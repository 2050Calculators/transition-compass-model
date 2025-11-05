import pandas as pd

from model.common.data_matrix_class import DataMatrix
from model.common.constant_data_matrix_class import ConstantDataMatrix
from model.common.io_database import dm_to_database
from model.common.interface_class import Interface
from model.common.auxiliary_functions import  calibration_rates, create_years_list, linear_forecast_BAU
from model.common.auxiliary_functions import read_level_data, filter_country_and_load_data_from_pickles
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
def read_data(DM_diet_input, lever_setting):

    # Read fts based on lever_setting
    # FIXME error it adds ots and fts
    # DM_check = check_ots_fts_match(DM_agriculture, lever_setting)
    DM_ots_fts = read_level_data(DM_diet_input, lever_setting)

    # Sub-matrix for DIETARY HABITS
    dm_diet_requirement = DM_ots_fts['kcal-req']
    dm_diet_split = DM_ots_fts['diet-split']
    dm_diet_fwaste = DM_ots_fts['fwaste']
    dm_fxa_cal_diet = DM_diet_input['fxa']['cal_agr_diet']
    dm_diet_adherence = DM_ots_fts['diet-adherence']

    # Aggregate Data Matrix - DIETARY HABITS
    DM_diet = {
        'energy-requirement': dm_diet_requirement,
        'diet-split': dm_diet_split,
        'diet-fwaste': dm_diet_fwaste,
        'cal_diet': dm_fxa_cal_diet,
        'diet-adherence': dm_diet_adherence
    }

    CDM_const = DM_diet_input['constant']

    return DM_ots_fts, DM_diet, CDM_const


# SimulateInteractions
def simulate_population_to_dietaryhabits_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../_database/data/interface/population_to_dietary-habits.pickle")
    with open(f, 'rb') as handle:
        DM_pop = pickle.load(handle)

    return DM_pop

# CalculationLeaf DIET WITH ADHERENCE SCENARIO --------------------------------------------------------------
def diet_adherence_scenarios(DM_diet, DM_pop, CDM_const, bau):

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


  # Average kcal-req [kcal/cap/day] = sum(demography [inhabitants] * kcal-req by demography [kcal/cap/day]) / sum(demography [inhabitants])
  dm_diet_requirement = DM_diet['energy-requirement'].copy()
  dm_diet_requirement.append(DM_pop['lfs_demography_'], dim='Variables')
  dm_diet_requirement.operation('lfs_demography', '*', var_kcal_req,
                                out_col='lfs_kcal-req', unit='kcal/day')
  dm_diet_requirement.group_all('Categories1')
  dm_diet_requirement.operation('lfs_kcal-req', '/', 'lfs_demography',
                                out_col='lfs_kcal-req_req',
                                unit='kcal/cap/day')
  dm_diet_requirement.filter({'Variables': ['lfs_kcal-req_req']}, inplace=True)

  # Intake of food categories i [kcal/country/year] = kcal-req [kcal/cap/day] * diet adherence [%] * share of i [%] * pop * days per year
  cdm_lifestyle = CDM_const['cdm_lifestyle'].copy()
  dm_diet_split = DM_diet['diet-split'].copy()
  dm_population = DM_pop['lfs_population_'].copy()
  ay_total_diet = dm_diet_requirement[:, :, 'lfs_kcal-req_req', np.newaxis] * \
                  dm_population[:, :, 'lfs_population_total', np.newaxis] * \
                  dm_diet_split[:, :, var_consumers_diet, :] * cdm_lifestyle[
                    'cp_time_days-per-year'] * dm_adherence[:,:,'share_diet_adherence', np.newaxis]
  dm_diet_food = DataMatrix.based_on(ay_total_diet[:, :, np.newaxis, :],
                                    dm_diet_split,
                                    change={'Variables': ['lfs_diet_raw']},
                                    units={'lfs_diet_raw': 'kcal'})

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


# CalculationLeaf LIFESTYLE TO DIET/FOOD DEMAND --------------------------------------------------------------
def lifestyle_workflow(DM_diet, DM_pop, CDM_const, years_setting):

    # DIFFERENTIATE BETWEEN BAU & SCENARIOS ------------------------------------

    # Create BAU variables - diet-split
    array_temp = DM_diet['diet-split'][:, :,'lfs_consumers-diet',:]
    DM_diet['diet-split'].add(array_temp, dummy=False, col_label='lfs_consumers-diet_bau', dim='Variables', unit='kcal/cap/day')

    # Extrapolate BAU fts - diet-split
    years_ots = create_years_list(years_setting[0], years_setting[1], 1)
    years_fts = create_years_list(years_setting[2], years_setting[3], 5)
    dm_ots_temp = DM_diet['diet-split'].filter({'Years':years_ots})
    dm_bau_fts = linear_forecast_BAU(dm_ots_temp, years_setting[0], years_ots, years_fts, min_tb=None,
                        max_tb=None)
    for i in years_fts:
      DM_diet['diet-split'][:, i, 'lfs_consumers-diet_bau', :] = dm_bau_fts[:, i, 'lfs_consumers-diet_bau', :]

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
    dm_diet_food_bau, dm_diet_consumed_bau = diet_adherence_scenarios(DM_diet, DM_pop, CDM_const, bau=True)
    dm_diet_food_scenario, dm_diet_consumed_scenario = diet_adherence_scenarios(DM_diet, DM_pop, CDM_const,
                                                bau=False)

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


# CalculationLeaf INTERFACE OUT  --------------------------------------------------------------
def dietaryhabits_TCAF_interface(dm_diet_consumed_bau, dm_diet_consumed_scenario):

  # Filter
  dm_diet_consumed_bau.filter({'Variables':['lfs_consumers-diet']}, inplace=True)
  dm_diet_consumed_scenario.filter({'Variables': ['lfs_consumers-diet']},
                              inplace=True)

  # Aggregate in DM
  DM_TCAF_health_diet = {"diet-consumed_bau": dm_diet_consumed_bau,
                         "diet-consumed_scenario": dm_diet_consumed_scenario}

  return DM_TCAF_health_diet


def dietaryhabits(lever_setting, years_setting, DM_input, interface=Interface()):

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    DM_ots_fts, DM_diet, CDM_const = read_data(DM_input, lever_setting)
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
    dm_lfs, dm_diet_consumed, dm_diet_consumed_bau, dm_diet_consumed_scenario, dm_diet_food = lifestyle_workflow(DM_diet, DM_pop, CDM_const, years_setting)

    # INTERFACES OUT ---------------------------------------------------------------------------------------------------

    # Dietary Habits to TCAF_health-diet
    DM_TCAF_health_diet = dietaryhabits_TCAF_interface(dm_diet_consumed_bau, dm_diet_consumed_scenario)
    interface.add_link(from_sector='dietary-habits', to_sector='TCAF_health-diet', dm=DM_TCAF_health_diet)

    # Dietary Habits to Production

    # TPE OUTPUT -------------------------------------------------------------------------------------------------------
    results_run = dietaryhabits_TPE_interface(CDM_const, dm_lfs, dm_diet_consumed, dm_diet_food)

    return results_run


def dietaryhabits_local_run():
    country_list = ['Switzerland', 'Vaud']
    DM_input = filter_country_and_load_data_from_pickles(country_list= country_list, modules_list = 'dietary-habits')
    years_setting, lever_setting = init_years_lever()
    dietaryhabits(lever_setting, years_setting, DM_input['dietary-habits'])


if __name__ == "__main__":
  dietaryhabits_local_run()
