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
def read_data(DM_diet_input, lever_setting):

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

    # Step 1: Create a dictionary of all DataMatrix objects
    dm_diet_split_kcal = {lever: DM_ots_fts[lever] for lever in levers}

    # Step 2: Merge them all into one DataMatrix along 'Variables'
    dm_diet_split_kcal_merged = None

    for lever_name, dm_split in dm_diet_split_kcal.items():
      # ensure each variable label is unique
      dm_split.col_labels['Variables'] = ['lfs_consumers-diet']
      #dm_split.rename_col([lever_name], 'lfs_consumers-diet', 'Variables')

      if dm_diet_split_kcal_merged is None:
        dm_diet_split_kcal_merged = dm_split
      else:
        dm_diet_split_kcal_merged.append(dm_split, dim='Categories1')

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

    CDM_const = DM_diet_input['constant']

    return DM_ots_fts, DM_diet, CDM_const


# SimulateInteractions
def simulate_dietaryhabits_to_livestock_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../_database/data/interface/dietary-habits_to_trade.pickle")
    with open(f, 'rb') as handle:
        dm_demand = pickle.load(handle)

    return dm_demand

# CalculationLeaf LIVESTOCK FOOD DEMAND TO DOMESTIC FOOD PRODUCTION --------------------------------------------------------------
def trade_livestock_workflow(DM_food_demand, dm_demand):
    # Overall food demand [kcal] = food demand [kcal] + food waste [kcal] NOW IN lifestyle_workflow()
    # dm_lfs.operation('lfs_total-cal-demand', '+', 'lfs_food-wastes', out_col='agr_demand', unit='kcal')

    # Filtering dms to only keep pro
    dm_demand_pro = dm_demand.filter_w_regex({'Categories1': 'pro-.*', 'Variables': 'agr_demand'})
    food_net_import_pro = DM_food_demand['food-net-import-pro'].filter_w_regex(
        {'Categories1': 'pro-.*', 'Variables': 'agr_food-net-import'})
    # Dropping the unwanted columns
    food_net_import_pro.drop(dim='Categories1', col_label=['pro-crop-processed-cake', 'pro-crop-processed-molasse'])

    # Sorting the dms alphabetically
    food_net_import_pro.sort(dim='Categories1')
    dm_demand_pro.sort(dim='Categories1')

    # Domestic production processed food [kcal] = agr_demand_pro_(.*) [kcal] * net-imports_pro_(.*) [%]
    idx_lfs = dm_demand_pro.idx
    idx_import = food_net_import_pro.idx
    agr_domestic_production = dm_demand_pro.array[:, :, idx_lfs['agr_demand'], :] \
                              * food_net_import_pro.array[:, :, idx_import['agr_food-net-import'], :]

    # Adding agr_domestic_production to dm_demand_pro
    dm_demand_pro.add(agr_domestic_production, dim='Variables', col_label='agr_domestic_production', unit='kcal')

    return dm_demand, dm_demand_pro

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

    # Account for milk as Feed and Processed
    # Milk Food & Feed [kcal] = Milk Food [kcal] * fxa_milk_feed_food_ratio [%]
    array_temp = DM_livestock['losses'][:,:,'agr_domestic_production','abp-dairy-milk'] * \
                 DM_livestock['ratio_milk'][:,:,'fxa_agr_feed-processing-food-ratio_abp-dairy-milk']
    DM_livestock['losses'][:,:,'agr_domestic_production','abp-dairy-milk'] = array_temp

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

    return DM_livestock, dm_liv_ibp, dm_liv_ibp, dm_liv_prod, dm_liv_pop

# CalculationLeaf INTERFACE TO TPE  --------------------------------------------------------------
def livestock_TPE_interface(CDM_const, dm_lfs, dm_diet_consumed, dm_diet_food):

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
def livestock_TCAF_interface(dm_diet_consumed_bau, dm_diet_consumed_scenario):

  # Filter
  dm_diet_consumed_bau.filter({'Variables':['lfs_consumers-diet']}, inplace=True)
  dm_diet_consumed_scenario.filter({'Variables': ['lfs_consumers-diet']},
                              inplace=True)

  # Aggregate in DM
  DM_TCAF_health_diet = {"diet-consumed_bau": dm_diet_consumed_bau,
                         "diet-consumed_scenario": dm_diet_consumed_scenario}

  return DM_TCAF_health_diet


def livestock(lever_setting, years_setting, DM_input, write_pickle, interface=Interface()):

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    DM_ots_fts, DM_diet, CDM_const = read_data(DM_input, lever_setting)
    country_list = ['Switzerland']

    # INTERFACES IN ---------------------------------------------------------------------------------------------------

    # Link interface or Simulate data from other modules
    if interface.has_link(from_sector='dietary-habits', to_sector='livestock'):
        dm_demand = interface.get_link(from_sector='dietary-habits', to_sector='livestock')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing dietary-habits to livestock interface')
        dm_demand = simulate_dietaryhabits_to_livestock_input()
        for key in dm_demand.keys():
            dm_demand[key].filter({'Country': country_list}, inplace=True)

    # CalculationTree LIVESTOCK TRADE & PRODUCTION

    dm_diet_consumed_bau = {}
    dm_diet_consumed_scenario = {}
    dm_demand, dm_demand_pro = trade_livestock_workflow(DM_food_demand, dm_demand)
    DM_livestock, dm_liv_ibp, dm_liv_ibp, dm_liv_prod, dm_liv_pop = livestock_workflow(DM_livestock, CDM_const, dm_lfs_pro, years_setting)


    # INTERFACES OUT ---------------------------------------------------------------------------------------------------

    # Livestock to TCAF
    DM_TCAF_health_diet = livestock_TCAF_interface(dm_diet_consumed_bau, dm_diet_consumed_scenario,)
    if write_pickle is True:
      current_file_directory = os.path.dirname(os.path.abspath(__file__))
      f = os.path.join(current_file_directory,
                       '../_database/data/interface/livestock_to_TCAF.pickle')
      with open(f, 'wb') as handle:
        pickle.dump(DM_TCAF_health_diet, handle, protocol=pickle.HIGHEST_PROTOCOL)
    interface.add_link(from_sector='livestock', to_sector='TCAF',
                           dm=DM_TCAF_health_diet)
        # pour update un pickle qui existe déjà, par exemple pour gagner du temps au pre-processing,
        # Pour remplacer des valeurs dans la même structure. Accepete un pays différent
        #my_pickle_dump(DM_new=DM_TCAF_health_diet, local_pickle_file=f)


    # TPE OUTPUT -------------------------------------------------------------------------------------------------------
    results_run = livestock_TPE_interface(CDM_const, dm_lfs, dm_diet_consumed, dm_diet_food)

    return results_run


def livestock_local_run():
    country_list = ['Switzerland', 'Vaud']
    DM_input = filter_country_and_load_data_from_pickles(country_list= country_list, modules_list = 'livestock')
    years_setting, lever_setting = init_years_lever()
    livestock(lever_setting, years_setting, DM_input['livestock'], write_pickle=True)


if __name__ == "__main__":
  livestock_local_run()
