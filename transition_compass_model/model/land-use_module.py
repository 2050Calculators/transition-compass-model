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
def read_data(DM_landuse_pickle, lever_setting):

    # Read fts based on lever_setting
    DM_ots_fts = read_level_data(DM_landuse_pickle, lever_setting)

    # Sub-matrix - CROPLAND

    # For levers : crop-share-.*
    # list of lever names
    levers = ['crop-share-organic',
              'crop-share-extensive',
              'crop-share-intensive']

    # 1: Create a dictionary of all DataMatrix objects
    dict_prod_share = {lever: DM_ots_fts[lever] for lever in levers}

    # 2: Merge them all into one DataMatrix along 'Variables'
    dm_prod_share_merged = None

    for lever_name, dm_temp in dict_prod_share.items():
      if dm_prod_share_merged is None:
        dm_prod_share_merged = dm_temp
      else:
        dm_prod_share_merged.append(dm_temp, dim='Variables')

    dm_cal_crop_share_area = DM_landuse_pickle['fxa']['cal_crop-share-area']
    dm_cal_cropland = DM_landuse_pickle['fxa']['cal_cropland_total']
    dm_fxa_yield_ch = DM_landuse_pickle['fxa']['yield-ch']
    dm_fxa_yield_imports = DM_landuse_pickle['fxa']['yield-imports']

    DM_cropland = {
      'yield-ch': dm_fxa_yield_ch,
      'yield-imports': dm_fxa_yield_imports,
      'crop-share': dm_prod_share_merged,
      'cal_crop-share-area': dm_cal_crop_share_area,
      'cal_cropland_total': dm_cal_cropland
    }

    # Sub matrix - GRASSLAND
    dm_livestock_density = DM_ots_fts['livestock-density']

    DM_grassland = {
      'ruminant_density': dm_livestock_density
    }

    CDM_const = DM_landuse_pickle['constant']

    return DM_ots_fts, DM_cropland, DM_grassland, CDM_const


# SimulateInteractions crop
def simulate_crop_to_landuse_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../_database/data/interface/crop_to_land-use.pickle")
    with open(f, 'rb') as handle:
        dm_demand = pickle.load(handle)
    return dm_demand

# SimulateInteractions livestock
def simulate_livestock_to_landuse_input():
  current_file_directory = os.path.dirname(os.path.abspath(__file__))
  f = os.path.join(current_file_directory,
                   "../_database/data/interface/livestock_to_land-use.pickle")
  with open(f, 'rb') as handle:
    DM_livestock_to_crop = pickle.load(handle)
  return DM_livestock_to_crop

# CalculationLeaf CROPLAND
def cropland_workflow(dm_crop_prod, DM_cropland, years_setting):

  # Step Switzerland - organic, extensive, intensive

  # Formatting
  DM_cropland['crop-share'].deepen(based_on='Variables')
  DM_cropland['crop-share'].switch_categories_order(cat1='Categories2', cat2='Categories1')
  dm_crop_ch = dm_crop_prod.filter({'Variables':['agr_domestic-production_afw'],
                                            'Country':['Switzerland'],
                                            'Categories1':DM_cropland['yield-ch'].col_labels['Categories1']})

  # (CH Only) Yield_T [kcal/ha] = yield_o * share_o + yield_e * share_e + yield_i * share_i
  # This will change values for fts only
  array_temp = DM_cropland['yield-ch']['Switzerland', :,'agr_crop_yield_organic', :] * \
               DM_cropland['crop-share']['Switzerland', :,'agr_share','organic', :] + \
               DM_cropland['yield-ch']['Switzerland', :,'agr_crop_yield_extensive', :] * \
               DM_cropland['crop-share']['Switzerland', :,'agr_share','extensive', :] + \
               DM_cropland['yield-ch']['Switzerland', :,'agr_crop_yield_intensive', :] * \
               DM_cropland['crop-share']['Switzerland', :,'agr_share','intensive', :]
  DM_cropland['yield-ch']['Switzerland', :, 'agr_crop_yield_total',:] = array_temp

  # (CH only) Total cropland per crop type [ha] = domestic prod [kcal] / yield_t [kcal/ha]
  dm_crop_ch.append(DM_cropland['yield-ch'], dim='Variables')
  dm_crop_ch.operation('agr_domestic-production_afw', '/', 'agr_crop_yield_total',
                                 dim='Variables',
                                 out_col='agr_cropland_total_raw',
                                 unit='ha')

  # (CH only) Calibration cropland per type (without algae, insect and lgn-energycrop)
  dm_cal_cropland = DM_cropland['cal_cropland_total']
  dm_cal_cropland.drop(dim='Categories1', col_label='fibres-plant-eq')
  dm_cal_rates_cropland = calibration_rates(dm_crop_ch.filter({'Variables':['agr_cropland_total_raw']}),
                                            dm_cal_cropland,
                                            calibration_start_year=1990,
                                            calibration_end_year=2023,
                                            years_setting=years_setting)
  dm_crop_ch.append(dm_cal_rates_cropland, dim='Variables')
  dm_crop_ch.operation('agr_cropland_total_raw', '*', 'cal_rate',
                        dim='Variables',
                        out_col='agr_cropland_total', unit='ha')

  # (CH only) Organic/ex/int cropland [ha] = Total cropland [ha] * share-organic/ext/int [-]
  array_temp = dm_crop_ch['Switzerland', :, 'agr_cropland_total', np.newaxis, :] * \
               DM_cropland['crop-share']['Switzerland', :,'agr_share',:, :]
  DM_cropland['crop-share'].add(array_temp[np.newaxis,:,np.newaxis,:,:],
                                dim='Variables',
                                col_label='agr_cropland_raw',
                                unit='ha')

  # (CH only) Calibration Cropland per crop type and production method
  dm_cal_crop_share_area = DM_cropland['cal_crop-share-area']
  dm_crop_share_area = DM_cropland['crop-share'].filter({'Variables': ['agr_cropland_raw']})
  dm_cal_rates_crop_share_area = calibration_rates(dm_crop_share_area,
    dm_cal_crop_share_area,
    calibration_start_year=1990,
    calibration_end_year=2022,
    years_setting=years_setting)
  DM_cropland['crop-share'].append(dm_cal_rates_crop_share_area, dim='Variables')
  DM_cropland['crop-share'].operation('agr_cropland_raw', '*',
                                         'cal_rate',
                                         dim='Variables',
                                         out_col='agr_cropland',
                                         unit='ha')

  # (CH only) Organic/ex/int dom prod [ha] = yield_o/e/i [kcal/ha] * Organic/ex/int cropland [ha]
  DM_cropland['crop-share'].add(0.0, dummy=True, col_label='agr_domestic-production_afw', dim='Variables', unit='kcal')
  array_org = DM_cropland['yield-ch']['Switzerland', :,'agr_crop_yield_organic', :] * \
               DM_cropland['crop-share']['Switzerland', :,'agr_cropland','organic', :]
  DM_cropland['crop-share'][:,:,'agr_domestic-production_afw','organic',:] = array_org

  array_org = DM_cropland['yield-ch']['Switzerland', :,'agr_crop_yield_extensive', :] * \
               DM_cropland['crop-share']['Switzerland', :,'agr_cropland','extensive', :]
  DM_cropland['crop-share'][:,:,'agr_domestic-production_afw','extensive',:] = array_org

  array_org = DM_cropland['yield-ch']['Switzerland', :,'agr_crop_yield_intensive', :] * \
               DM_cropland['crop-share']['Switzerland', :,'agr_cropland','intensive', :]
  DM_cropland['crop-share'][:,:,'agr_domestic-production_afw','intensive',:] = array_org

  # Step Imports

  # Formatting
  dm_crop_imports = dm_crop_prod.filter({'Variables':['agr_domestic-production_afw']})

  # Total cropland per crop type [ha] = domestic prod [kcal] / yield [kcal/ha]
  dm_crop_imports.append(DM_cropland['yield-imports'], dim='Variables')
  dm_crop_imports.operation('agr_domestic-production_afw', '/', 'agr_crop_yield',
                                 dim='Variables',
                                 out_col='agr_cropland_total',
                                 unit='ha')


  # Check Calibration FAO and Swiss data in line
  dm_cal_cropland_total_swiss = dm_cal_crop_share_area.groupby(
    {'total': '.*'}, dim='Categories1', regex=True, inplace=False)
  dm_cal_cropland_total_swiss = dm_cal_cropland_total_swiss.flatten()
  dm_cal_cropland_total_swiss.rename_col_regex('agr_cropland', 'agr_cropland_swiss', dim='Variables')
  dm_cal_cropland_total_swiss.rename_col_regex('total_', '', dim='Categories1')
  dm_cal_cropland_total_swiss.filter(
    {'Years': dm_cal_cropland.col_labels['Years']}, inplace=True)
  dm_cal_cropland_total_swiss.append(dm_cal_cropland, dim='Variables')

  return dm_crop_imports

# CalculationLeaf GRASSLAND

def grassland_workflow(DM_grassland, dm_liv_pop):
  # (CH only) GRAZING LIVESTOCK
  # Filtering ruminants (bovine & sheep)
  dm_liv_ruminants = dm_liv_pop.filter(
    {'Variables': ['agr_liv_population'],
     'Categories1': ['meat-bovine', 'meat-sheep', 'abp-dairy-milk'],
     'Country': ['Switzerland']})
  # Ruminant livestock [lsu] = population bovine + population sheep + population dairy
  dm_liv_ruminants.groupby({'ruminant': '.*'}, dim='Categories1', regex=True,
                           inplace=True)
  # Append to relevant dm
  dm_liv_ruminants = dm_liv_ruminants.filter(
    {'Variables': ['agr_liv_population'], 'Categories1': ['ruminant']})
  dm_liv_ruminants = dm_liv_ruminants.flatten()  # change from category to variable
  DM_grassland['ruminant_density'].append(dm_liv_ruminants,
                                         dim='Variables')  # Append to caf
  # Agriculture grassland [ha] = ruminant livestock [lsu] / livestock density [lsu/ha]
  DM_grassland['ruminant_density'].operation('agr_liv_population_ruminant', '/',
                                            'agr_livestock_density',
                                            dim="Variables",
                                            out_col='agr_lus_land_raw_grassland',
                                            unit='ha')

# CalculationLeaf INTERFACE TO TPE  --------------------------------------------------------------
def crop_TPE_interface():

    # DIET (CONSUMED, WITHOUT FOOD WASTES) -------------------------------------

    # Flatten for TPE
    dm_tpe = dm_diet_consumed.flattest()

    # DIET (WITH FOOD WASTES) --------------------------------------------------

    # Filter
    dm_supply = dm_demand.filter({'Variables': ['agr_demand']})
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
def crop_TCAF_interface(dm_diet_consumed_bau, dm_diet_consumed_scenario):

  # Filter
  dm_diet_consumed_bau.filter({'Variables':['lfs_consumers-diet']}, inplace=True)
  dm_diet_consumed_scenario.filter({'Variables': ['lfs_consumers-diet']},
                              inplace=True)

  # Aggregate in DM
  DM_TCAF_health_diet = {"diet-consumed_bau": dm_diet_consumed_bau,
                         "diet-consumed_scenario": dm_diet_consumed_scenario}

  return DM_TCAF_health_diet


def crop(lever_setting, years_setting, DM_input, write_pickle, interface=Interface()):

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    DM_ots_fts, DM_cropland, DM_grassland, CDM_const = read_data(DM_input, lever_setting)
    country_list = ['Switzerland']

    # INTERFACES IN ---------------------------------------------------------------------------------------------------

    # Link interface or Simulate data from other modules
    # crop
    if interface.has_link(from_sector='crop', to_sector='land-use'):
        dm_crop_prod = interface.get_link(from_sector='crop', to_sector='land-use')

    else:
        if len(interface.list_link()) != 0:
            print('You are missing crop to land-use interface')
        dm_crop_prod = simulate_crop_to_landuse_input()

    # livestock
    if interface.has_link(from_sector='livestock', to_sector='land-use'):
        dm_liv_pop = interface.get_link(from_sector='livestock', to_sector='land-use')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing livestock to land-use interface')
        dm_liv_pop = simulate_livestock_to_landuse_input()

    # CalculationTree LANDUSE MODULE

    dm_crop_imports = cropland_workflow(dm_crop_prod, DM_cropland, years_setting)
    grassland_workflow(DM_grassland, dm_liv_pop)

    # INTERFACES OUT ---------------------------------------------------------------------------------------------------

    # land-use to TCAF
    DM_landuse_to_TCAF = dm_crop_imports
    if write_pickle is True:
      current_file_directory = os.path.dirname(os.path.abspath(__file__))
      f = os.path.join(current_file_directory,
                       '../_database/data/interface/land-use_to_TCAF.pickle')
      with open(f, 'wb') as handle:
        pickle.dump(DM_landuse_to_TCAF, handle, protocol=pickle.HIGHEST_PROTOCOL)
    interface.add_link(from_sector='land-use', to_sector='TCAF', dm=DM_landuse_to_TCAF)

    # TPE OUTPUT -------------------------------------------------------------------------------------------------------
    #results_run = livestock_TPE_interface(CDM_const, dm_lfs, dm_diet_consumed, dm_diet_food)
    results_run = dm_crop_imports

    return results_run


def landuse_local_run():
    country_list = ['Switzerland']
    DM_input = filter_country_and_load_data_from_pickles(country_list= country_list, modules_list = 'land-use', filter_country=False)
    years_setting, lever_setting = init_years_lever()
    crop(lever_setting, years_setting, DM_input['land-use'], write_pickle=True)


if __name__ == "__main__":
  landuse_local_run()
