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
def read_data(DM_livestock, lever_setting):

    # Read fts based on lever_setting
    # FIXME error it adds ots and fts
    # DM_check = check_ots_fts_match(DM_agriculture, lever_setting)
    DM_ots_fts = read_level_data(DM_livestock, lever_setting)

    # Sub-matrix for LIVESTOCK
    dm_livestock_losses = DM_ots_fts['livestock-losses']
    dm_livestock_yield = DM_ots_fts['livestock-yield']
    dm_livestock_slaughtered = DM_ots_fts['slaughter-rates']
    dm_livestock_density = DM_ots_fts['livestock-density']
    dm_livestock_enteric_emissions = DM_ots_fts['livestock-enteric']
    dm_livestock_manure = DM_ots_fts['livestock-manure']
    dm_ration = DM_ots_fts['feed-ration']
    dm_alt_protein = DM_ots_fts['alt-protein']
    dm_ruminant_feed = DM_ots_fts['ruminant-feed']
    dm_fxa_ratio_milk = DM_livestock['fxa']['ratio_milk']
    dm_fxa_cal_liv_prod = DM_livestock['fxa']['cal_agr_domestic-production-liv']
    dm_fxa_cal_liv_pop = DM_livestock['fxa']['cal_agr_liv-population']
    dm_fxa_cal_liv_CH4 = DM_livestock['fxa']['cal_agr_liv_CH4-emission']
    dm_fxa_cal_liv_N2O = DM_livestock['fxa']['cal_agr_liv_N2O-emission']
    dm_fxa_cal_demand_feed = DM_livestock['fxa']['cal_agr_demand_feed']
    dm_fxa_ef_liv_N2O = DM_livestock['fxa']['ef_liv_N2O-emission']
    dm_fxa_ef_liv_CH4_treated = DM_livestock['fxa']['ef_liv_CH4-emission_treated']
    dm_fxa_liv_nstock = DM_livestock['fxa']['liv_manure_n-stock']
    dm_trade_origin = DM_livestock['fxa']['trade-origin']


    # Aggregate Data Matrix - LIVESTOCK PROD & POP
    DM_liv_prod = {
        'losses': dm_livestock_losses,
        'yield': dm_livestock_yield,
        'trade-origin': dm_trade_origin,
        'liv_slaughtered_rate': dm_livestock_slaughtered,
        'cal_liv_prod': dm_fxa_cal_liv_prod,
        'cal_liv_population': dm_fxa_cal_liv_pop,
        'ruminant_density': dm_livestock_density,
        'ratio_milk': dm_fxa_ratio_milk
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
        'cal_agr_demand_feed': dm_fxa_cal_demand_feed,
        'ruminant-feed': dm_ruminant_feed
    }

    CDM_const = DM_livestock['constant']

    return DM_ots_fts, DM_liv_prod, DM_manure, DM_feed, CDM_const


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
def livestock_production_workflow(DM_liv_prod, CDM_const, dm_lfs_pro, years_setting):
    # Filter dm_lfs_pro to only have livestock products
    dm_lfs_pro_liv = dm_lfs_pro.filter_w_regex({'Categories1': 'pro-liv.*', 'Variables': 'agr_domestic_production'})
    # Drop the pro- prefix of the categories
    dm_lfs_pro_liv.rename_col_regex(str1="pro-liv-", str2="", dim="Categories1")
    # Sort the dms
    dm_lfs_pro_liv.sort(dim='Categories1')
    DM_liv_prod['losses'].sort(dim='Categories1')
    DM_liv_prod['yield'].sort(dim='Categories1')

    # Append dm_lfs_pro_liv to DM_liv_prod['losses']
    DM_liv_prod['losses'].append(dm_lfs_pro_liv, dim='Variables')

    # Account for milk as Feed and Processed
    # Milk Food & Feed [kcal] = Milk Food [kcal] * fxa_milk_feed_food_ratio [%]
    array_temp = DM_liv_prod['losses'][:,:,'agr_domestic_production','abp-dairy-milk'] * \
                 DM_liv_prod['ratio_milk'][:,:,'fxa_agr_feed-processing-food-ratio_abp-dairy-milk']
    DM_liv_prod['losses'][:,:,'agr_domestic_production','abp-dairy-milk'] = array_temp

    # Livestock domestic prod with losses [kcal] = livestock domestic prod [kcal] * Production losses livestock [%]
    DM_liv_prod['losses'].operation('agr_climate-smart-livestock_losses', '*', 'agr_domestic_production',
                                     out_col='agr_domestic_production_liv_afw_raw', unit='kcal')

    # Calibration - Livestock domestic production
    dm_cal_liv_prod = DM_liv_prod['cal_liv_prod']
    dm_liv_prod = DM_liv_prod['losses'].filter({'Variables': ['agr_domestic_production_liv_afw_raw']})
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
    DM_liv_prod['yield'].append(dm_liv_slau, dim='Variables')  # Append cal_agr_domestic_production_liv_afw in yield
    DM_liv_prod['yield'].operation('agr_domestic_production_liv_afw', '/', 'agr_climate-smart-livestock_yield',
                                    dim="Variables", out_col='agr_liv_population_slau', unit='lsu')

    # Livestock population (stock) [lsu] = Livestock slaughtered [lsu] / slaughter rate [%]
    dm_liv_slau_egg_dairy = DM_liv_prod['yield'].filter({'Variables': ['agr_liv_population_slau']})
    DM_liv_prod['liv_slaughtered_rate'].append(dm_liv_slau_egg_dairy, dim='Variables')
    # dm_liv_slau_meat = DM_liv_prod['yield'].filter({'Variables': ['agr_liv_population_raw'],
    #                                                 'Categories1': ['meat-bovine', 'meat-pig', 'meat-poultry',
    #                                                                 'meat-sheep', 'meat-oth-animals']})
    # DM_liv_prod['liv_slaughtered_rate'].append(dm_liv_slau_meat, dim='Variables')
    DM_liv_prod['liv_slaughtered_rate'].operation('agr_liv_population_slau', '/',
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
    dm_cal_liv_pop = DM_liv_prod['cal_liv_population']
    dm_liv_pop = DM_liv_prod['liv_slaughtered_rate'].filter({'Variables': ['agr_liv_population_raw']})
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
    DM_liv_prod['ruminant_density'].append(dm_liv_ruminants, dim='Variables')  # Append to caf
    # Agriculture grassland [ha] = ruminant livestock [lsu] / livestock density [lsu/ha]
    DM_liv_prod['ruminant_density'].operation('agr_liv_population_ruminant', '/',
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
    dm_processed_offal_afat = DM_liv_prod['losses'].filter({'Variables': ['agr_domestic_production_liv_afw_raw'],
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

    return DM_liv_prod, dm_liv_ibp, dm_liv_ibp, dm_liv_prod, dm_liv_pop

# CalculationLeaf MANURE MANAGEMENT & GHG EMISSIONS ----------------------------------------------------------
def manure_workflow(DM_manure, dm_liv_pop, years_setting):
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

    # Unit conversion: [kcal] to [t]
    # Filter
    cdm_kcal = CDM_const['cdm_kcal-per-t'].copy()
    cdm_kcal.rename_col_regex(str1="pro-liv-", str2="", dim="Categories1")
    cdm_kcal = cdm_kcal.filter({'Categories1': ['abp-dairy-milk', 'abp-hens-egg', 'meat-bovine', 'meat-oth-animals', 'meat-pig', 'meat-poultry', 'meat-sheep']})
    # Sort
    dm_feed_req.sort('Categories1')
    cdm_kcal.sort('Categories1')
    # Convert from [kcal] to [t]
    array_temp = dm_feed_req[:, :, 'agr_domestic_production_liv_afw', :] \
                 / cdm_kcal[np.newaxis, np.newaxis, 'cp_kcal-per-t', :]
    dm_feed_req.add(array_temp, dim='Variables', col_label='agr_domestic_production_liv_afw_t',
                                       unit='t')

    # Sort
    dm_feed_req.sort('Categories1')
    cdm_cp_efficiency.sort('Categories1')

    # Feed req per livestock type [t] = domestic ASF prod accounting for waste [t] * feed conversion ratio [kg DM feed/kg EW] EW: edible weight
    dm_temp = dm_feed_req[:, :,'agr_domestic_production_liv_afw_t', :] \
              * cdm_cp_efficiency[np.newaxis, np.newaxis, 'cp_efficiency_liv', :]
    dm_feed_req.add(dm_temp, dim='Variables', col_label='agr_feed-requirement', unit='t')

    # For bovine & dairy cattle & sheep : Ruminant feed without grass [t] = ruminant feed [t] * (1-Share of grass in ruminant feed [%])
    list_ruminant =['abp-dairy-milk', 'meat-bovine', 'meat-sheep']
    dm_feed_ruminant = dm_feed_req.filter({'Variables': ['agr_feed-requirement'],'Categories1': list_ruminant})
    array_temp = dm_feed_ruminant[:, :, 'agr_feed-requirement', :] \
              * DM_feed['ruminant-feed']['ruminant-feed'][:, :, np.newaxis, 'agr_ruminant-feed_share-grass']
    dm_feed_ruminant.add(array_temp, dim='Variables', col_label='agr_feed-requirement_grass',
                    unit='t')
    dm_feed_ruminant.operation('agr_feed-requirement', '-',
                                'agr_feed-requirement_grass',
                                out_col='agr_feed-requirement_without-grass', unit='t')
    dm_feed_ruminant = dm_feed_ruminant.filter({'Variables': ['agr_feed-requirement_without-grass']})

    # Pre-processing for other feed and appending with ruminant feed without grass
    list_others = ['abp-hens-egg', 'meat-oth-animals', 'meat-pig', 'meat-poultry']
    dm_feed_without_grass = dm_feed_req.filter({'Variables': ['agr_feed-requirement'], 'Categories1': list_others})
    dm_feed_without_grass.rename_col('agr_feed-requirement',
                           'agr_feed-requirement_without-grass', dim='Variables')
    dm_feed_without_grass.append(dm_feed_ruminant, dim='Categories1')

    # Total feed req [t] = sum(Feed req per livestock type without grass [t])
    dm_feed_req_total = dm_feed_without_grass.filter({'Variables': ['agr_feed-requirement_without-grass']})
    dm_feed_req_total.groupby({'total': '.*'}, dim='Categories1', regex=True, inplace=True)
    dm_feed_req_total = dm_feed_req_total.flatten()

    # ALTERNATIVE PROTEIN SOURCE (APS) FOR LIVESTOCK FEED
    # APS [kcal] = Feed req per livestock type [kcal] * APS share per type [%]
    idx_aps = DM_feed['alt-protein'].idx
    idx_feed = dm_feed_without_grass.idx
    dm_temp = dm_feed_without_grass.array[:, :, idx_feed['agr_feed-requirement_without-grass'], :, np.newaxis] \
              * DM_feed['alt-protein'].array[:, :, idx_aps['agr_alt-protein'], :, :]
    DM_feed['alt-protein'].add(dm_temp, dim='Variables', col_label='agr_feed_aps', unit='t')

    # Insect meals [t] = sum algae feed req
    dm_aps = DM_feed['alt-protein'].filter({'Variables': ['agr_feed_aps'], 'Categories2': ['algae']})
    dm_aps = dm_aps.flatten()
    dm_aps.groupby({'algae': '.*'}, dim='Categories1', regex=True, inplace=True)

    # Insect meals [t] = sum insect feed req
    dm_insect = DM_feed['alt-protein'].filter({'Variables': ['agr_feed_aps'], 'Categories2': ['insect']})
    dm_insect = dm_insect.flatten()
    dm_insect.groupby({'insect': '.*'}, dim='Categories1', regex=True, inplace=True)
    dm_aps.append(dm_insect, dim='Categories1')

    # APS meals [t] = Insect meals [t] + Insect meals [t]
    dm_aps_feed = dm_aps.copy()
    dm_aps_feed.groupby({'total': '.*'}, dim='Categories1', regex=True, inplace=True)
    dm_aps_feed = dm_aps_feed.flatten()

    # Filter APS byproduct ration constant
    cdm_aps_ibp = CDM_const['cdm_aps_ibp']

    # APS byproducts [t] = APS production [t] * byproduct ratio [%]
    idx_cdm = cdm_aps_ibp.idx
    idx_aps = dm_aps.idx
    dm_temp = dm_aps.array[:, :, idx_aps['agr_feed_aps'], np.newaxis, :, np.newaxis] \
              * cdm_aps_ibp.array[idx_cdm['cp_ibp_aps'], np.newaxis, :, :]
    # dm_aps.add(dm_temp, dim='Variables', col_label='agr_aps', unit='t') FIXME find correct dm to add to or create one

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

    # Alternative feed ration [kcal] = sum (cereals from bev for feed, APS)
    dm_aps_feed.append(dm_bev_ibp_cereal_feed, dim='Variables')
    dm_aps_feed.operation('agr_feed_aps_total', '+', 'agr_use_bev_ibp_cereal_feed_t', dim='Variables',
               out_col='agr_alt-feed-ration',
               unit='t')

    # Crop based feed demand [kcal] = Total feed req without grass [kcal] - Alternative feed ration [kcal] FIXME change 1st component name
    dm_feed_req_total.append(dm_aps_feed, dim='Variables')
    dm_feed_req_total.operation('agr_feed-requirement_without-grass_total', '-', 'agr_alt-feed-ration',
                                out_col='agr_crop-feed-demand', unit='t')

    # Feed demand by type [kcal] = Crop based feed demand by type [kcal] * Share of feed per type [%]
    idx_feed = dm_feed_req_total.idx
    idx_ration = DM_feed['ration'].idx
    dm_temp = dm_feed_req_total.array[:, :, idx_feed['agr_feed-requirement_without-grass_total'], np.newaxis] \
              * DM_feed['ration'].array[:, :, idx_ration['agr_climate-smart-livestock_ration'], :]
    DM_feed['ration'].add(dm_temp, dim='Variables', col_label='agr_demand_feed_raw', unit='kcal')

    # Calibration Feed demand
    dm_cal_feed = DM_feed['cal_agr_demand_feed']
    dm_feed_demand = DM_feed['ration'].filter({'Variables': ['agr_demand_feed_raw']})
    dm_cal_rates_feed = calibration_rates(dm_feed_demand, dm_cal_feed, calibration_start_year=1990,
                                          calibration_end_year=2023,
                                          years_setting=years_setting)
    DM_feed['ration'].append(dm_cal_rates_feed, dim='Variables')
    DM_feed['ration'].operation('agr_demand_feed_raw', '*', 'cal_rate', dim='Variables', out_col='agr_demand_feed_t',
                                unit='t')
    # Calibration values fill na with 0
    dm_temp = DM_feed['ration'].filter({'Variables': ['agr_demand_feed_t']})
    array_temp = dm_temp.array[:, :, :, :]
    array_temp = np.nan_to_num(array_temp, nan=0)
    dm_temp.array[:, :, :, :] = array_temp
    DM_feed['ration'][:, :, 'agr_demand_feed_t', :] = dm_temp[:, :, 'agr_demand_feed_t', :]

    # Unit conversion : [t] => [kcal]
    cdm_kcal = CDM_const['cdm_kcal-per-t'].copy()
    cdm_kcal.rename_col_regex(str1="pro-", str2="", dim="Categories1")
    cdm_kcal.rename_col_regex(str1="seafood", str2="fish", dim="Categories1")
    categories_feed = ['crop-cereal', 'crop-fruit', 'crop-oilcrop',
                       'crop-processed-cake', 'crop-processed-molasse',
                       'crop-processed-sugar', 'crop-processed-voil',
                       'crop-pulse', 'crop-rice', 'crop-starch', 'crop-sugarcrop',
                       'crop-veg', 'fish', 'liv-meat-meal']
    cdm_kcal = cdm_kcal.filter({'Categories1': categories_feed})

    # Sort
    DM_feed['ration'].sort('Categories1')
    cdm_kcal.sort('Categories1')

    # Convert from [t] to [kcal]
    array_temp = DM_feed['ration'][:, :, 'agr_demand_feed_t', :] \
                 * cdm_kcal[np.newaxis, np.newaxis, 'cp_kcal-per-t', :]
    DM_feed['ration'].add(array_temp, dim='Variables', col_label='agr_demand_feed',
                                       unit='kcal')
    #dm_supply = dm_supply.filter({'Variables': ['agr_demand_tpe', 'agr_demand']})

    return DM_feed, dm_aps_ibp, dm_feed_req, dm_aps, dm_feed_demand

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
    DM_ots_fts, DM_liv_prod, DM_manure, DM_feed, CDM_const = read_data(DM_input, lever_setting)
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
    DM_livestock, dm_liv_ibp, dm_liv_ibp, dm_liv_prod, dm_liv_pop = livestock_production_workflow(DM_liv_prod, CDM_const, dm_lfs_pro, years_setting)


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
    country_list = ['Switzerland']
    DM_input = filter_country_and_load_data_from_pickles(country_list= country_list, modules_list = 'livestock', filter_country=False)
    years_setting, lever_setting = init_years_lever()
    livestock(lever_setting, years_setting, DM_input['livestock'], write_pickle=True)


if __name__ == "__main__":
  livestock_local_run()
