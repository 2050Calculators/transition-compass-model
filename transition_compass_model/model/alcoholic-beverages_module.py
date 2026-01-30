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
def read_data(DM_alc_bev_input, lever_setting):

    # Read fts based on lever_setting
    DM_ots_fts = read_level_data(DM_alc_bev_input, lever_setting)

    # Sub-matrix for ALCOHOLIC BEVERAGES
    #dm_alc_bev = DM_ots_fts['biomass-hierarchy']['biomass-hierarchy-bev-ibp-use-oth']
    dm_processing_yield = DM_alc_bev_input['fxa']['processing-yield']
    dm_bev_ssr = DM_ots_fts['ssr-bev']
    dm_cal_crop_bev = DM_alc_bev_input['fxa']['cal_agr_domestic-production_bev']

    # Aggregated Data Matrix - ALCOHOLIC BEVERAGES
    DM_alc_bev = {
        #'biomass_hierarchy': dm_alc_bev,
        'processing-yields': dm_processing_yield,
        'ssr-bev': dm_bev_ssr,
        'cal_bev': dm_cal_crop_bev
    }


    CDM_const = DM_alc_bev_input['constant']

    return DM_ots_fts, DM_alc_bev, CDM_const


# SimulateInteractions
def simulate_dietaryhabits_to_alcoholic_beverages_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../_database/data/interface/dietary-habits_to_alcoholic-beverages.pickle")
    with open(f, 'rb') as handle:
        DM_pop = pickle.load(handle)

    return DM_pop

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
    dm_demand_bev.add(array_agr_domestic_production, dim='Variables', col_label='agr_domestic_production', unit='kcal')

    # Imports processed bev [kcal] = demand bev processed bev[kcal] - domestic production processed bev[kcal]
    dm_demand_bev.operation('agr_demand', '-',
                              'agr_domestic_production',
                              out_col='agr_domestic_production_raw', unit='kcal')

    # Calibration imports

    # Splits imports per country

    # Raw crops [kcal] =  imported bev per country [kcal] * processing yield [input kcal/output kcal]


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


def alcoholic_beverages(lever_setting, years_setting, DM_input, write_pickle, interface=Interface()):

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    DM_ots_fts, DM_alc_bev, CDM_const = read_data(DM_input, lever_setting)
    country_list = ['Switzerland']

    # INTERFACES IN ---------------------------------------------------------------------------------------------------

    # Link interface or Simulate data from other modules
    # dietary-habits to alcoholic-beverages
    if interface.has_link(from_sector='dietary-habits', to_sector='alcoholic-beverages'):
        dm_lfs = interface.get_link(from_sector='dietary-habits', to_sector='alcoholic-beverages')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing dietary-habits to alcoholic-beverages interface')
        dm_lfs = simulate_dietaryhabits_to_alcoholic_beverages_input()

    # CalculationTree ALCOHOLIC BEVERAGES

    DM_alc_bev, dm_bev_ibp_cereal_feed, dm_bev_dom_prod = alcoholic_beverages_workflow(DM_alc_bev, CDM_const, dm_lfs, years_setting,)

    # INTERFACES OUT ---------------------------------------------------------------------------------------------------

    # alcoholic-beverages to livestock
    DM_alc_livestock = {'bev_feed': dm_bev_ibp_cereal_feed}
    if write_pickle is True:
      current_file_directory = os.path.dirname(os.path.abspath(__file__))
      f = os.path.join(current_file_directory,
                       '../_database/data/interface/alcoholic-beverages_to_livestock.pickle')
      with open(f, 'wb') as handle:
        pickle.dump(DM_alc_livestock, handle, protocol=pickle.HIGHEST_PROTOCOL)
    interface.add_link(from_sector='alcoholic-beverages', to_sector='livestock', dm=DM_alc_livestock)

    # alcoholic-beverages to crop
    DM_alc_to_crop = {'crop_bev': dm_bev_dom_prod}
    if write_pickle is True:
      current_file_directory = os.path.dirname(os.path.abspath(__file__))
      f = os.path.join(current_file_directory,
                       '../_database/data/interface/alcoholic-beverages_to_crop.pickle')
      with open(f, 'wb') as handle:
        pickle.dump(DM_alc_to_crop, handle, protocol=pickle.HIGHEST_PROTOCOL)
    interface.add_link(from_sector='alcoholic-beverages', to_sector='crop', dm=DM_alc_to_crop)

    # TPE OUTPUT -------------------------------------------------------------------------------------------------------
    #results_run = dietaryhabits_TPE_interface(CDM_const, dm_lfs, dm_diet_consumed, dm_diet_food)

    return


def alcoholic_beverages_local_run():
    country_list = ['Switzerland']
    DM_input = filter_country_and_load_data_from_pickles(country_list= country_list, modules_list = 'alcoholic-beverages', filter_country=True)
    years_setting, lever_setting = init_years_lever()
    alcoholic_beverages(lever_setting, years_setting, DM_input['alcoholic-beverages'], write_pickle=True)


if __name__ == "__main__":
  alcoholic_beverages_local_run()
