import pandas as pd

from model.common.data_matrix_class import DataMatrix
from model.common.constant_data_matrix_class import ConstantDataMatrix
from model.common.io_database import dm_to_database
from model.common.interface_class import Interface
from model.common.auxiliary_functions import  calibration_rates, create_years_list, linear_forecast_BAU, dm_match_countries
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
def read_data(DM_crop_pickle, lever_setting):

    # Read fts based on lever_setting
    DM_ots_fts = read_level_data(DM_crop_pickle, lever_setting)

    # Sub-matrix for CROP

    # For levers : ssr-crop-.*
    # list of lever names
    levers = ['ssr-crop-cereal',
              'ssr-crop-fruit',
              'ssr-crop-veg',
              'ssr-crop-rice',
              'ssr-crop-starch',
              'ssr-crop-oilcrop',
              'ssr-crop-pulse',
              'ssr-crop-sugarcrop']

    # 1: Create a dictionary of all DataMatrix objects
    dict_crop_ssr = {lever: DM_ots_fts[lever] for lever in levers}

    # 2: Merge them all into one DataMatrix along 'Variables'
    dm_crop_ssr_merged = None

    for lever_name, dm_temp in dict_crop_ssr.items():
      if dm_crop_ssr_merged is None:
        dm_crop_ssr_merged = dm_temp
      else:
        dm_crop_ssr_merged.append(dm_temp, dim='Categories1')

    # Rename Categories 1
    dm_crop_ssr_merged.rename_col_regex('crop-', '', dim='Categories1')

    # For levers : ssr-bev-.*
    # list of lever names
    levers = ['ssr-bev-beer',
              'ssr-bev-wine',
              'ssr-bev-bev-alc',
              'ssr-bev-bev-fer']

    # 1: Create a dictionary of all DataMatrix objects
    dict_bev_ssr = {lever: DM_ots_fts[lever] for lever in levers}

    # 2: Merge them all into one DataMatrix along 'Variables'
    dm_bev_ssr_merged = None

    for lever_name, dm_temp in dict_bev_ssr.items():
      if dm_bev_ssr_merged is None:
        dm_bev_ssr_merged = dm_temp
      else:
        dm_bev_ssr_merged.append(dm_temp, dim='Categories1')

    # For levers : ssr-pro-.*
    # list of lever names
    levers = ['ssr-pro-sugar',
              'ssr-pro-voil',
              'ssr-pro-sweet']

    # 1: Create a dictionary of all DataMatrix objects
    dict_pro_ssr = {lever: DM_ots_fts[lever] for lever in levers}

    # 2: Merge them all into one DataMatrix along 'Variables'
    dm_pro_ssr_merged = None

    for lever_name, dm_temp in dict_pro_ssr.items():
      if dm_pro_ssr_merged is None:
        dm_pro_ssr_merged = dm_temp
      else:
        dm_pro_ssr_merged.append(dm_temp, dim='Categories1')

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

    dm_crop = DM_ots_fts['crop-losses']
    dm_imports = DM_crop_pickle['fxa']['split-import-crop']
    dm_imports_pro = DM_crop_pickle['fxa']['split-import-crop-pro']
    #dm_food_net_import_crop.drop(dim='Categories1', col_label=['stm'])
    #dm_residues_yield = DM_crop_pickle['fxa']['residues_yield']
    #dm_hierarchy_residues_cereals = DM_ots_fts['biomass-hierarchy']['biomass-hierarchy_crop_cereal']
    dm_cal_crop = DM_crop_pickle['fxa']['cal_agr_domestic-production_food']
    dm_cal_crop_bev = DM_crop_pickle['fxa']['cal_agr_domestic-production_bev']
    dm_fxa_cal_crop_imports_countries = DM_crop_pickle['fxa']['cal_agr_imports-crop_countries']
    dm_fxa_cal_crop_imports_tot = DM_crop_pickle['fxa']['cal_agr_imports-crop_total']
    dm_fxa_cal_crop_pro_imports_tot = DM_crop_pickle['fxa'][
      'cal_agr_imports-crop-pro_total']
    dm_share_export = DM_crop_pickle['fxa']['share-export']
    dm_share_export.rename_col_regex('crop-', '', dim='Categories1')
    # dm_crop.append(dm_cal_crop, dim='Variables')
    #dm_ef_residues = DM_crop_pickle['fxa']['ef_burnt-residues']
    #dm_ssr_feed_crop = DM_ots_fts['feed-net-import']
    dm_processing_yield = DM_crop_pickle['fxa']['processing-yield']
    #dm_food_net_import_pro = DM_ots_fts['food-net-import'].filter_w_regex(
    #    {'Categories1': 'pro-.*', 'Variables': 'agr_food-net-import'})

    # Sub-matrix - CROPLAND
    dm_cal_crop_share_area = DM_crop_pickle['fxa']['cal_crop-share-area']
    dm_cal_cropland = DM_crop_pickle['fxa']['cal_cropland_total']
    dm_fxa_yield_ch = DM_crop_pickle['fxa']['yield-ch']
    dm_fxa_yield_imports = DM_crop_pickle['fxa']['yield-imports']

    # Aggregated Data Matrix - CROP
    DM_crop_prod = {
        'crop': dm_crop,
        'split-import-crop': dm_imports,
        'split-import-crop-pro': dm_imports_pro,
        'share-export': dm_share_export,
        'cal_crop': dm_cal_crop,
        'cal_bev': dm_cal_crop_bev,
        'cal_imports-crop_countries': dm_fxa_cal_crop_imports_countries,
        'cal_imports-crop_tot': dm_fxa_cal_crop_imports_tot,
        'cal_imports-crop-pro_tot': dm_fxa_cal_crop_pro_imports_tot,
        'ssr-crop': dm_crop_ssr_merged,
        'ssr-pro': dm_pro_ssr_merged,
        'processing-yields': dm_processing_yield
    }

    DM_cropland = {
      'yield-ch': dm_fxa_yield_ch,
      'yield-imports': dm_fxa_yield_imports,
      'crop-share': dm_prod_share_merged,
      'cal_crop-share-area': dm_cal_crop_share_area,
      'cal_cropland_total': dm_cal_cropland
    }

    CDM_const = DM_crop_pickle['constant']

    return DM_ots_fts, DM_crop_prod, DM_cropland, CDM_const


# SimulateInteractions dietary-habits
def simulate_dietaryhabits_to_crop_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../_database/data/interface/dietary-habits_to_crop.pickle")
    with open(f, 'rb') as handle:
        dm_demand = pickle.load(handle)
    return dm_demand

# SimulateInteractions livestock
def simulate_livestock_to_crop_input():
  current_file_directory = os.path.dirname(os.path.abspath(__file__))
  f = os.path.join(current_file_directory,
                   "../_database/data/interface/livestock_to_crop.pickle")
  with open(f, 'rb') as handle:
    DM_livestock_to_crop = pickle.load(handle)
  return DM_livestock_to_crop

# SimulateInteractions alcoholic-beverages
def simulate_alc_to_crop_input():
  current_file_directory = os.path.dirname(os.path.abspath(__file__))
  f = os.path.join(current_file_directory,
                   "../_database/data/interface/alcoholic-beverages_to_crop.pickle")
  with open(f, 'rb') as handle:
    DM_alc_to_crop = pickle.load(handle)
  return DM_alc_to_crop

# CalculationLeaf CROP PRODUCTION ----------------------------------------------------------------------------------
def crop_workflow(DM_crop_prod, CDM_const, dm_feed_processed, dm_feed_unprocessed, dm_demand, dm_bev_dom_prod, dm_feed_processed_imports,dm_bev_raw_imports, years_setting):

    # Step FEED ---------------------------------------------------------------------------------------------------

    # Constant pre-processing
    #cdm_feed_yield = CDM_const['cdm_feed_yield']
    #cdm_food_yield = CDM_const['cdm_food_yield']

    # (CH) Processed Feed crop dom prod [kcal] = dom prod processed crops for feed [kcal] * processing yield [%]
    dm_pro_yield = DM_crop_prod['processing-yields'].filter({'Categories1': [
      'cake-to-oilcrop',
      'molasse-to-sugarcrop']})
    dm_pro_yield.rename_col('cake-to-oilcrop','crop-processed-cake', dim='Categories1')
    dm_pro_yield.rename_col('molasse-to-sugarcrop', 'crop-processed-molasse', dim='Categories1')
    #dm_pro_yield.rename_col('voil-to-oilcrop', 'crop-processed-voil', dim='Categories1')
    #dm_pro_yield.rename_col('sugar-to-sugarcrop', 'crop-processed-sugar', dim='Categories1')
    dm_feed_processed.append(dm_pro_yield, dim='Variables')
    dm_feed_processed.operation('agr_domestic_production_feed_pro', '*', 'fxa_agr_processing-yield',
                                out_col='agr_domestic_production_feed_pro_raw',
                                unit='kcal')

    # Summing by crop category (oilcrop and sugarcrop)
    dm_feed_processed.groupby({'crop-oilcrop': '.*cake', 'crop-sugarcrop': '.*molasse'}, dim='Categories1',
                              regex=True,
                              inplace=True)

    # Creating copy
    dm_feed_processed_copy = dm_feed_processed.copy()

    # Adding dummy columns filled with 0.0 for total feed demand calculations
    dm_feed_processed.add(0.0, dummy=True, col_label='crop-cereal', dim='Categories1', unit='kcal')
    dm_feed_processed.add(0.0, dummy=True, col_label='crop-pulse', dim='Categories1', unit='kcal')
    dm_feed_processed.add(0.0, dummy=True, col_label='crop-fruit', dim='Categories1', unit='kcal')
    dm_feed_processed.add(0.0, dummy=True, col_label='crop-veg', dim='Categories1', unit='kcal')
    dm_feed_processed.add(0.0, dummy=True, col_label='crop-starch', dim='Categories1', unit='kcal')
    dm_feed_processed.add(0.0, dummy=True, col_label='crop-rice', dim='Categories1', unit='kcal')

    # Accounting for processed feed demand : Adding the columns for sugarcrops and oilcrops from previous calculation
    # Appending with dm_feed_processed
    dm_feed_unprocessed = dm_feed_unprocessed.filter({'Variables': ['agr_demand_feed']})
    dm_feed_processed = dm_feed_processed.filter({'Variables': ['agr_domestic_production_feed_pro_raw']})
    dm_feed_unprocessed.append(dm_feed_processed, dim='Variables')
    # Summing
    dm_feed_unprocessed.operation('agr_domestic_production_feed_pro_raw', '+', 'agr_demand_feed', out_col='agr_demand_feed_total',
                                  unit='kcal')
    dm_feed_unprocessed = dm_feed_unprocessed.filter({'Variables': ['agr_demand_feed_total']})

    # Adding dummy categories
    #dm_feed_unprocessed.add(0.0, dummy=True, col_label='crop-lgn-energycrop', dim='Categories1', unit='kcal')
    #dm_feed_unprocessed.add(0.0, dummy=True, col_label='crop-algae', dim='Categories1', unit='kcal')
    #dm_feed_unprocessed.add(0.0, dummy=True, col_label='crop-insect', dim='Categories1', unit='kcal')


    # Step PROCESSED FOOD ---------------------------------------------------------------------------------------------------

    # Processed food - Accounting for SSR

    # Domestic production [kcal] = Processed Food-demand [kcal] * net import [%]
    dm_food_processed = dm_demand.filter(
        {'Variables': ['agr_demand'], 'Categories1': ['pro-crop-processed-sweet', 'pro-crop-processed-sugar', 'pro-crop-processed-voil']})
    dm_ssr_food_pro = DM_crop_prod['ssr-pro'].filter(
        {'Variables': ['agr_ssr'],
         'Categories1': dm_food_processed.col_labels['Categories1']}).copy()
    dm_food_processed.append(dm_ssr_food_pro, dim='Variables')
    dm_food_processed.operation('agr_demand', '*', 'agr_ssr', out_col='agr_domestic-production_food_pro',
                                unit='kcal')

    # Create copy for imports
    dm_food_processed_imports = dm_food_processed.copy()

    # Processed Food crop demand [kcal] = processed crops [kcal] * processing yield [%] (only for sweets & processed sugar)
    # sum processed sugar in one variable : processed sugar : sweets + processed sugar
    dm_food_processed.groupby({'pro-crop-processed-sugar': '.*sugar|.*sweet'}, dim='Categories1', regex=True, inplace=True)
    dm_food_processed.rename_col('pro-crop-processed-sugar', 'crop-sugarcrop', dim='Categories1')
    dm_food_processed.rename_col('pro-crop-processed-voil', 'crop-oilcrop',
                                 dim='Categories1')
    dm_pro_yield = DM_crop_prod['processing-yields'].filter({'Categories1': ['sugar-to-sugarcrop', 'voil-to-oilcrop']})
    dm_pro_yield.rename_col('sugar-to-sugarcrop', 'crop-sugarcrop',
                                 dim='Categories1')
    dm_pro_yield.rename_col('voil-to-oilcrop', 'crop-oilcrop',
                            dim='Categories1')
    dm_food_processed.append(dm_pro_yield, dim='Variables')
    dm_food_processed.operation('agr_domestic-production_food_pro', '*', 'fxa_agr_processing-yield',
                                out_col='agr_demand_food',
                                unit='kcal')


    # Step Imports Processed Food -------------------------------------------------------------
    # Imports processed food [kcal] = demand processed food[kcal] - domestic production processed food[kcal]
    dm_food_processed_imports.operation('agr_demand', '-', 'agr_domestic-production_food_pro',
                                out_col='agr_imported-production_food_pro_raw',
                                unit='kcal')

    # Calibration imports (processed food)
    dm_cal_rates_food_pro = calibration_rates(dm_food_processed_imports.filter({'Variables': ['agr_imported-production_food_pro_raw']}), DM_crop_prod['cal_imports-crop-pro_tot'], calibration_start_year=1990,
                                          calibration_end_year=2023, years_setting=years_setting)
    dm_food_processed_imports.append(dm_cal_rates_food_pro, dim='Variables')
    dm_food_processed_imports.operation('agr_imported-production_food_pro_raw', '*', 'cal_rate', dim='Variables',
                          out_col='agr_imported-production_food_pro', unit='kcal')
    dm_food_processed_imports.filter({'Variables': ['agr_demand','agr_domestic-production_food_pro','agr_imported-production_food_pro']}, inplace=True)

    # Imported production per region [kcal] = Imported production total [kcal] * split per region [-]
    #DM_alc_bev['split-import-asf'].filter_w_regex({'Categories1': 'crop-'},inplace=True)
    dm_trade = DM_crop_prod['split-import-crop-pro'].copy()
    array_temp = dm_food_processed_imports[:, :, 'agr_imported-production_food_pro', :] * \
                 dm_trade[:, :, 'agr_split-import', :]
    DM_crop_prod['split-import-crop-pro'].add(array_temp, dim='Variables',
                                     col_label='agr_domestic_production',
                                     unit='kcal')

    # Raw crops [kcal] =  imported feed per country [kcal] * processing yield [input kcal/output kcal]
    cdm_food = CDM_const['cdm_ibp_food'].copy()

    # Create dummy variable to overwite
    DM_crop_prod['split-import-crop-pro'].add(0.0, dummy=True, col_label='agr_domestic_production_food-pro_raw',
                          dim='Variables', unit='kcal')

    # Sugar to sugarcrop
    array_temp = DM_crop_prod['split-import-crop-pro'][:, :, 'agr_domestic_production', 'pro-crop-processed-sugar'] \
                                   * cdm_food['cp_ibp_processed','crop-processed-sugar']
    DM_crop_prod['split-import-crop-pro'][:, :, 'agr_domestic_production_food-pro_raw', 'pro-crop-processed-sugar'] = array_temp

    # Sweet to sugarcrop
    array_temp = DM_crop_prod['split-import-crop-pro'][:, :, 'agr_domestic_production', 'pro-crop-processed-sweet'] \
                                   * cdm_food['cp_ibp_processed','crop-processed-sweet']
    DM_crop_prod['split-import-crop-pro'][:, :, 'agr_domestic_production_food-pro_raw', 'pro-crop-processed-sweet'] = array_temp


    # Voil to oilcrop
    array_temp = DM_crop_prod['split-import-crop-pro'][:, :, 'agr_domestic_production', 'pro-crop-processed-voil'] \
                                   * cdm_food['cp_ibp_processed','crop-processed-voil']
    DM_crop_prod['split-import-crop-pro'][:, :, 'agr_domestic_production_food-pro_raw', 'pro-crop-processed-voil'] = array_temp


    # Step NON-PROCESSED FOOD ---------------------------------------------------------------------------------------------------

    # Pre processing total food demand per category (with dummy categories when necessary)
    # Categories x8 : cereals, oilcrop, pulse, fruit, veg, starch, sugarcrop, rice (+ maybe lgn, alage and insect)
    dm_crop_demand = dm_demand.filter_w_regex({'Variables': 'agr_demand', 'Categories1': 'crop-|rice'})
    # Renaming categories
    dm_crop_demand.rename_col_regex(str1="agr_demand", str2="agr_demand_food", dim="Variables")
    dm_crop_demand.rename_col_regex(str1="crop-", str2="", dim="Categories1")

    # Accounting for processed food demand :Adding the column for sugarcrops & oilcrops from previous calculation
    dm_sugarcrop = dm_food_processed.filter({'Variables': ['agr_demand_food'], 'Categories1':['crop-sugarcrop']})
    dm_crop_demand = dm_crop_demand.filter({'Variables': ['agr_demand_food']})
    dm_crop_demand.append(dm_sugarcrop, dim='Categories1')
    # Sorting alphabetically and renaming col
    dm_crop_demand.sort(dim='Categories1')
    dm_crop_demand.rename_col('crop-sugarcrop', 'sugarcrop', dim='Categories1')
    dm_crop_demand[:,:,'agr_demand_food','oilcrop'] = dm_crop_demand[:,:,'agr_demand_food','oilcrop'] \
                                                    + dm_food_processed[:,:,'agr_demand_food','crop-oilcrop']


    # Adding dummy categories
    #dm_crop_demand.add(0.0, dummy=True, col_label='lgn-energycrop', dim='Categories1', unit='kcal')
    #dm_crop_demand.add(0.0, dummy=True, col_label='algae', dim='Categories1', unit='kcal')
    #dm_crop_demand.add(0.0, dummy=True, col_label='insect', dim='Categories1', unit='kcal')


    # Step PROCESSED BEV ----------------------------------------------------------------------------------------------------

    # Here the SSR is already accounted for, but not the losses
    # Adding dummy categories
    dm_bev_dom_prod.add(0.0, dummy=True, col_label='oilcrop', dim='Categories1', unit='kcal')
    dm_bev_dom_prod.add(0.0, dummy=True, col_label='pulse', dim='Categories1', unit='kcal')
    dm_bev_dom_prod.add(0.0, dummy=True, col_label='veg', dim='Categories1', unit='kcal')
    dm_bev_dom_prod.add(0.0, dummy=True, col_label='starch', dim='Categories1', unit='kcal')
    dm_bev_dom_prod.add(0.0, dummy=True, col_label='sugarcrop', dim='Categories1', unit='kcal')
    dm_bev_dom_prod.add(0.0, dummy=True, col_label='rice', dim='Categories1', unit='kcal')
    #dm_bev_dom_prod.add(0.0, dummy=True, col_label='algae', dim='Categories1', unit='kcal')
    #dm_bev_dom_prod.add(0.0, dummy=True, col_label='insect', dim='Categories1', unit='kcal')
    #dm_bev_dom_prod.add(0.0, dummy=True, col_label='lgn-energycrop', dim='Categories1', unit='kcal')
    dm_bev_dom_prod.sort(dim='Categories1')

    """# Step PROCESSED BIOENERGY ----------------------------------------------------------------------------------------------

    # From BIOENERGY (oilcrop from voil + lgn from solid & liquid) (not accounted for in KNIME probably due to regex error)
    # Pre processing
    dm_oilcrop_voil = dm_oil.filter(
        {'Variables': ['agr_bioenergy_biomass-demand_liquid_oil'], 'Categories1': ['oil-voil']})
    # Accounting for SSR
    # Processed bioenergy - Accounting for SSR
    # Domestic production [kcal] = Processed Food-demand [kcal] * net import [%]
    dm_ssr_bioe_pro = DM_crop_prod['food-net-import-pro'].filter(
        {'Variables': ['agr_food-net-import'], 'Categories1': ['pro-crop-processed-voil']}).copy()
    dm_ssr_bioe_pro.rename_col('pro-crop-processed-voil', 'oil-voil', dim='Categories1')
    dm_oilcrop_voil.append(dm_ssr_bioe_pro, dim='Variables')
    dm_oilcrop_voil.operation('agr_bioenergy_biomass-demand_liquid_oil', '*', 'agr_food-net-import',
                              out_col='agr_demand_bioe_pro',
                              unit='kcal')

    # Accounting for processing yield
    idx_voil = dm_oilcrop_voil.idx
    idx_cdm = cdm_feed_yield.idx
    array_temp = dm_oilcrop_voil.array[:, :, idx_voil['agr_demand_bioe_pro'], :] \
                 / cdm_feed_yield.array[idx_cdm['cp_ibp_processed'], idx_cdm['voil-to-oilcrop']]
    dm_oilcrop_voil.add(array_temp, dim='Variables', col_label='agr_demand_bioe', unit='kcal')
    # Filtering and renaming for name matching
    dm_voil = dm_oilcrop_voil.filter({'Variables': ['agr_demand_bioe']})
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
    dm_lgn_energycrop.rename_col('agr_bioenergy_biomass-demand_liquid_lgn', 'agr_demand_bioe',
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
    dm_aps.sort(dim='Categories1')"""

    # Step FOOD + FEED + BEV + NON-FOOD ---------------------------------------------------------------------------------------------------

    # Appending the dms
    dm_feed_unprocessed.rename_col_regex(str1="crop-", str2="",
                                         dim="Categories1")  # Renaming categories
    dm_crop_demand.append(dm_feed_unprocessed, dim='Variables')

    # (CH only) Total crop demand by type (without bev) [kcal] = Sum crop demand (feed + food)
    dm_crop_demand.operation('agr_demand_feed_total', '+', 'agr_demand_food',
                             out_col='agr_demand_total', unit='kcal')
    dm_crop_demand = dm_crop_demand.filter({'Variables': ['agr_demand_total']})


    """# Appending the dms
    dm_voil.add(dm_lgn_energycrop.array, col_label='lgn-energycrop', dim='Categories1')
    dm_feed_unprocessed.rename_col_regex(str1="crop-", str2="", dim="Categories1")  # Renaming categories
    dm_crop_demand.append(dm_feed_unprocessed, dim='Variables')
    dm_crop_demand.append(dm_voil, dim='Variables')
    dm_crop_demand.append(dm_aps, dim='Variables')

    # Total crop demand by type (without bev) [kcal] = Sum crop demand (feed + food + non-food)
    dm_crop_demand.operation('agr_demand_feed_total', '+', 'agr_demand_food',
                             out_col='agr_demand_feed_food', unit='kcal')
    dm_crop_demand.operation('agr_demand_feed_food', '+', 'agr_demand_bioe',
                             out_col='agr_demand_feed_food_bioe', unit='kcal')
    dm_crop_demand.operation('agr_demand_feed_food_bioe', '+', 'agr_aps',
                             out_col='agr_demand_total', unit='kcal')
    dm_crop_demand = dm_crop_demand.filter({'Variables': ['agr_demand_total']})

    # Pre processing to remove lgn, algae & insect
    list = ['lgn-energycrop', 'algae', 'insect']
    dm_crop_other = dm_crop_demand.filter({'Categories1': list})
    dm_crop_other.rename_col('agr_demand', 'agr_demand_afw', dim='Variables')
    # Appending for remaining categories
    dm_crop_demand.drop(dim='Categories1', col_label=list)
    DM_crop['crop'].append(dm_crop_demand, dim='Variables')"""

    # (CH only) Dom prod (without bev) [kcal] = demand * SSR [%]
    dm_crop_demand.append(DM_crop_prod['ssr-crop'], dim='Variables')
    dm_crop_demand.operation('agr_demand_total', '*',
                              'agr_ssr',
                              out_col='agr_domestic-production_without_bev',
                              unit='kcal')

    # (CH only) Dom prod [kcal] = Dom prod (without bev) + Dom prod bev (raw crop)
    dm_crop_demand.append(dm_bev_dom_prod, dim='Variables')
    dm_crop_demand.operation('agr_domestic-production_without_bev', '+', 'agr_domestic-production_bev',
                             out_col='agr_domestic_production_raw', unit='kcal')

    # (CH only) CALIBRATION CROP PRODUCTION TOTAL --------------------------------------------------------------------------------------
    dm_cal_crop = DM_crop_prod['cal_crop']
    dm_crop_ch = dm_crop_demand.filter({'Variables': ['agr_domestic_production_raw']})
    # Drop rice because not produced in Switzerland
    dm_crop_ch.drop(dim='Categories1', col_label='rice')
    dm_cal_rates_crop = calibration_rates(dm_crop_ch, dm_cal_crop, calibration_start_year=1990,
                                          calibration_end_year=2023, years_setting=years_setting)
    # Add dummy with 1.0 for rice (because no rice produced in Switzerland)
    dm_cal_rates_crop.add(1.0, dummy=True,col_label='rice',dim='Categories1', unit='%')
    dm_crop_demand.append(dm_cal_rates_crop, dim='Variables')
    dm_crop_demand.operation('agr_domestic_production_raw', '*', 'cal_rate', dim='Variables',
                              out_col='agr_domestic_production', unit='kcal')
    dm_crop_demand.filter({'Variables':['agr_domestic_production','agr_demand_total','agr_demand_total']}, inplace=True)

    # Step TOTAL IMPORTS (RAW PRODUCTS)

    # Dom prod for exports [kcal] = Domestic production [kcal] * share exports [exports/production]
    dm_crop_demand.append(DM_crop_prod['share-export'], dim='Variables')
    dm_crop_demand.operation('agr_share-export', '*', 'agr_domestic_production',
                                     out_col='agr_exported_production', unit='kcal')

    '''# Account for processed food and feed in the demand (raw products)
    # processed imports + raw products for processing imported ?
    # Sugarcrops
    array_temp = dm_food_processed[:,:,'agr_imports_food_pro_raw','crop-sugarcrop'] + \
                 dm_feed_processed_copy[:, :, 'agr_imports_feed_pro_raw', 'crop-sugarcrop'] + \
                 dm_crop_demand[:,:,'agr_demand_total','sugarcrop']
    dm_crop_demand[:, :, 'agr_demand_total', 'sugarcrop'] = array_temp
    # Oilcrops
    array_temp = dm_feed_processed_copy[:, :, 'agr_imports_feed_pro_raw', 'crop-oilcrop'] + \
                 dm_crop_demand[:,:,'agr_demand_total','oilcrop']
    dm_crop_demand[:, :, 'agr_demand_total', 'oilcrop'] = array_temp'''

    # Imported production total [kcal] = Demand [kcal] - (Domestic production [kcal] - Dom prod for exports [kcal])
    '''dm_crop_demand.operation('agr_domestic_production', '-', 'agr_exported_production',
                                     out_col='temp', unit='kcal')
    dm_crop_demand.operation('agr_demand_total', '-', 'temp',
                                     out_col='agr_imported_production_total_raw', unit='kcal')'''

    # Imported production total [kcal] = Demand [kcal] - Domestic production [kcal]
    dm_crop_demand.operation('agr_demand_total', '-', 'agr_domestic_production',
                             out_col='agr_imported_production_total_raw',
                             unit='kcal')

    # Calibration - Imports total
    dm_cal_imports_tot = DM_crop_prod['cal_imports-crop_tot'].filter_w_regex({'Categories1': 'crop-'})
    dm_cal_imports_tot.rename_col_regex('crop-', '', dim='Categories1')
    dm_imports_tot = dm_crop_demand.filter({'Variables':['agr_imported_production_total_raw']})
    dm_cal_rates_imports = calibration_rates(dm_imports_tot, dm_cal_imports_tot, calibration_start_year=1990,
                                              calibration_end_year=2023, years_setting=years_setting)
    dm_crop_demand.append(dm_cal_rates_imports, dim='Variables')
    dm_crop_demand.operation('agr_imported_production_total_raw', '*', 'cal_rate', dim='Variables',
                          out_col='agr_imported_production_total', unit='kcal')

    # Imported production per region [kcal] = Imported production total [kcal] * split per region [-]
    DM_crop_prod['split-import-crop'].filter_w_regex({'Categories1': 'crop-'}, inplace=True)
    dm_trade = DM_crop_prod['split-import-crop'].copy()
    array_temp = dm_crop_demand[:,:,'agr_imported_production_total',:] * \
                 dm_trade[:,:,'agr_split-import',:]
    DM_crop_prod['split-import-crop'].add(array_temp, dim='Variables', col_label='agr_domestic_production_raw', unit='kcal')

    # Filter to only have imported production per countries
    dm_production = DM_crop_prod['split-import-crop'].filter_w_regex({'Variables': 'agr_domestic_production_raw'})

    # Calibration - Imports per countries
    dm_cal_imports_countries = DM_crop_prod['cal_imports-crop_countries'].filter_w_regex({'Categories1': 'crop-'})
    dm_cal_rates_imports_countries = calibration_rates(dm_production, dm_cal_imports_countries, calibration_start_year=1990,
                                              calibration_end_year=2023, years_setting=years_setting)
    dm_production.append(dm_cal_rates_imports_countries, dim='Variables')
    dm_production.operation('agr_domestic_production_raw', '*', 'cal_rate', dim='Variables',
                          out_col='agr_domestic_production', unit='kcal')

    # Total raw imports [kcal] = raw imports + Imports of raw products for processed food, feed and bev
    # oilcrop (raw + pro-food + pro-feed)
    array_temp = \
              dm_production[:, :, 'agr_domestic_production', 'crop-oilcrop'] \
            + DM_crop_prod['split-import-crop-pro'][:, :, 'agr_domestic_production_food-pro_raw', 'pro-crop-processed-voil'] \
            + dm_feed_processed_imports[:, :, 'agr_domestic_production', 'pro-crop-processed-cake']
    dm_production[:,:,'agr_domestic_production','crop-oilcrop'] = array_temp


    # sugarcrop (raw + pro-food + pro-feed)
    array_temp = \
              dm_production[:, :, 'agr_domestic_production', 'crop-sugarcrop'] \
            + DM_crop_prod['split-import-crop-pro'][:, :, 'agr_domestic_production_food-pro_raw', 'pro-crop-processed-sugar'] \
            + DM_crop_prod['split-import-crop-pro'][:, :, 'agr_domestic_production_food-pro_raw','pro-crop-processed-sweet'] \
            + dm_feed_processed_imports[:, :, 'agr_domestic_production', 'pro-crop-processed-molasse']
    dm_production[:,:,'agr_domestic_production','crop-oilcrop'] = array_temp


    # fruit (raw + bev)
    array_temp = \
              dm_production[:, :, 'agr_domestic_production', 'crop-fruit'] \
            + dm_bev_raw_imports[:, :, 'agr_domestic_production_bev_raw', 'pro-bev-bev-alc'] \
            + dm_bev_raw_imports[:, :, 'agr_domestic_production_bev_raw', 'pro-bev-wine']
    dm_production[:,:,'agr_domestic_production','crop-fruit'] = array_temp


    # cereal (raw + bev)
    array_temp = \
              dm_production[:, :, 'agr_domestic_production', 'crop-cereal'] \
            + dm_bev_raw_imports[:, :, 'agr_domestic_production_bev_raw', 'pro-bev-bev-fer'] \
            + dm_bev_raw_imports[:, :, 'agr_domestic_production_bev_raw', 'pro-bev-beer']
    dm_production[:,:,'agr_domestic_production','crop-cereal'] = array_temp

    # Append domestic production Switzerland + other countries
    dm_production.rename_col_regex('crop-', '', dim='Categories1')
    array_temp = \
              dm_crop_demand['Switzerland', :, 'agr_domestic_production', :]
    dm_production['Switzerland',:,'agr_domestic_production',:] = array_temp

    # Domestic production with losses [kcal] = domestic prod * food losses [%]
    DM_crop_prod['crop'].append(dm_production, dim='Variables')
    DM_crop_prod['crop'].operation('agr_domestic_production', '*', 'agr_crop_losses',
                              out_col='agr_domestic-production_afw', unit='kcal')

    """# CROP RESIDUES ----------------------------------------------------------------------------------------------------

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
    DM_ssr  = {'food': DM_crop['crop'],
               'feed': DM_crop['feed-net-import_crop'],
               'bioenergy': dm_ssr_bioe_pro,
               'processed': DM_crop['food-net-import-pro']}"""

    return DM_crop_prod, DM_crop_prod, dm_feed_processed, dm_food_processed

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
    DM_ots_fts, DM_crop_prod, DM_cropland, CDM_const = read_data(DM_input, lever_setting)
    country_list = ['Switzerland']

    # CalculationLeaf INTERFACES IN ---------------------------------------------------------------------------------------------------

    # Link interface or Simulate data from other modules
    # dietary-habits
    if interface.has_link(from_sector='dietary-habits', to_sector='crop'):
        DM_diet_crop = interface.get_link(from_sector='dietary-habits', to_sector='crop')
        dm_demand = DM_diet_crop['demand']
    else:
        if len(interface.list_link()) != 0:
            print('You are missing dietary-habits to crop interface')
        DM_diet_crop = simulate_dietaryhabits_to_crop_input()
        for key in DM_diet_crop.keys():
            DM_diet_crop[key].filter({'Country': country_list}, inplace=True)
        dm_demand = DM_diet_crop['demand']

    # livestock
    if interface.has_link(from_sector='livestock', to_sector='crop'):
        DM_livestock_to_crop = interface.get_link(from_sector='livestock', to_sector='crop')
        dm_feed_processed = DM_livestock_to_crop['feed-processed']
        dm_feed_processed_imports = DM_livestock_to_crop['feed-processed-imports']
        dm_feed_unprocessed = DM_livestock_to_crop['feed-unprocessed']
    else:
        if len(interface.list_link()) != 0:
            print('You are missing livestock to crop interface')
        DM_livestock_to_crop = simulate_livestock_to_crop_input()
        dm_feed_processed = DM_livestock_to_crop['feed-processed']
        dm_feed_processed_imports = DM_livestock_to_crop[
          'feed-processed-imports']
        dm_feed_unprocessed = DM_livestock_to_crop['feed-unprocessed']

    # alcoholic-beverages
    if interface.has_link(from_sector='livestock', to_sector='crop'):
        DM_alc_to_crop = interface.get_link(from_sector='alcoholic-beverages', to_sector='crop')
        dm_bev_dom_prod = DM_alc_to_crop['crop_bev']
        dm_bev_raw_imports = DM_alc_to_crop['imports_bev_raw']
        dm_match_countries(dm_bev_raw_imports, DM_crop_prod['split-import-crop'], 'perfect match')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing livestock to crop interface')
        DM_alc_to_crop = simulate_alc_to_crop_input()
        #for key in DM_alc_to_crop.keys():
        #    DM_alc_to_crop[key].filter({'Country': country_list}, inplace=True)
        dm_bev_dom_prod = DM_alc_to_crop['crop_bev']
        dm_bev_raw_imports = DM_alc_to_crop['imports_bev_raw']
        dm_match_countries(dm_bev_raw_imports, DM_crop_prod['split-import-crop'],
                       'perfect match')

    # CalculationTree CROP MODULE

    DM_crop_prod, dm_crop_ch, dm_feed_processed, dm_food_processed = crop_workflow(DM_crop_prod, CDM_const, dm_feed_processed, dm_feed_unprocessed, dm_demand, dm_bev_dom_prod, dm_feed_processed_imports, dm_bev_raw_imports, years_setting)

    # INTERFACES OUT ---------------------------------------------------------------------------------------------------

    # crop to land-use
    DM_crop_landuse = DM_crop_prod['crop'].filter({'Variables':['agr_domestic-production_afw']})
    if write_pickle is True:
      current_file_directory = os.path.dirname(os.path.abspath(__file__))
      f = os.path.join(current_file_directory,
                       '../_database/data/interface/crop_to_land-use.pickle')
      with open(f, 'wb') as handle:
        pickle.dump(DM_crop_landuse, handle, protocol=pickle.HIGHEST_PROTOCOL)
    interface.add_link(from_sector='crop', to_sector='land-use',
                           dm=DM_crop_landuse)
        # pour update un pickle qui existe déjà, par exemple pour gagner du temps au pre-processing,
        # Pour remplacer des valeurs dans la même structure. Accepete un pays différent
        #my_pickle_dump(DM_new=DM_TCAF_health_diet, local_pickle_file=f)

    # crop to TCAF
    DM_crop_to_TCAF = DM_crop_prod['crop'].filter({'Variables':['agr_domestic-production_afw']})
    if write_pickle is True:
      current_file_directory = os.path.dirname(os.path.abspath(__file__))
      f = os.path.join(current_file_directory,
                       '../_database/data/interface/crop_to_TCAF.pickle')
      with open(f, 'wb') as handle:
        pickle.dump(DM_crop_to_TCAF, handle, protocol=pickle.HIGHEST_PROTOCOL)
    interface.add_link(from_sector='crop', to_sector='TCAF', dm=DM_crop_to_TCAF)


    # TPE OUTPUT -------------------------------------------------------------------------------------------------------
    #results_run = livestock_TPE_interface(CDM_const, dm_lfs, dm_diet_consumed, dm_diet_food)
    results_run = DM_crop_prod

    return results_run


def crop_local_run():
    country_list = ['Switzerland']
    DM_input = filter_country_and_load_data_from_pickles(country_list= country_list, modules_list = 'crop', filter_country=False)
    years_setting, lever_setting = init_years_lever()
    crop(lever_setting, years_setting, DM_input['crop'], write_pickle=True)


if __name__ == "__main__":
  crop_local_run()
