import numpy as np
from model.common.auxiliary_functions import interpolate_nans, add_missing_ots_years, linear_fitting_ots_db, linear_fitting, create_years_list
#from _database.pre_processing.api_routines_CH import get_data_api_CH
from scipy.stats import linregress
import pandas as pd
import faostat
import os
import re
from model.common.data_matrix_class import DataMatrix
from model.common.constant_data_matrix_class import ConstantDataMatrix
from model.common.io_database import read_database, read_database_fxa, edit_database, database_to_df, dm_to_database, database_to_dm
from model.common.io_database import read_database_to_ots_fts_dict, read_database_to_ots_fts_dict_w_groups, read_database_to_dm
from model.common.interface_class import Interface
from model.common.auxiliary_functions import compute_stock,  filter_geoscale, calibration_rates, filter_DM, add_dummy_country_to_DM, my_pickle_dump
from model.common.auxiliary_functions import read_level_data, simulate_input
from scipy.optimize import linprog
import pickle
import json
import os
import numpy as np
import time

# Ensure structure coherence
def ensure_structure(df):
    # Get unique values for geoscale, timescale, and variables
    df['timescale'] = df['timescale'].astype(int)
    df = df.drop_duplicates(subset=['geoscale', 'timescale', 'level', 'variables', 'lever', 'module'])
    lever_name = list(set(df['lever']))[0]
    countries = df['geoscale'].unique()
    years = df['timescale'].unique()
    variables = df['variables'].unique()
    level = df['level'].unique()
    lever = df['lever'].unique()
    module = df['module'].unique()
    # Create a complete multi-index from all combinations of unique values
    full_index = pd.MultiIndex.from_product(
         [countries, years, variables, level, lever, module],
            names=['geoscale', 'timescale', 'variables', 'level', 'lever', 'module']
        )
    # Reindex the DataFrame to include all combinations, filling missing values with NaN
    df = df.set_index(['geoscale', 'timescale', 'variables', 'level', 'lever', 'module'])
    df = df.reindex(full_index, fill_value=np.nan).reset_index()

    return df


# CalculationLeaf DIET ------------------------------------------------------------------------------------
def diet_processing(list_countries, file, cdm_kcal, dm_kcal_req):
    # ----------------------------------------------------------------------------------------------------------------------
    # FOOD SUPPLY Part 1 - including food waste
    # ----------------------------------------------------------------------------------------------------------------------

    # Read data ------------------------------------------------------------------------------------------------------------
    try:
        df_diet = pd.read_csv(file)
    except OSError:

        # FOOD BALANCE SHEETS (FBS) - -------------------------------------------------
        # List of elements
        list_elements = ['Food']

        list_items = ['Cereals - Excluding Beer + (Total)', 'Fruits - Excluding Wine + (Total)', 'Oilcrops + (Total)',
                      'Pulses + (Total)', 'Rice (Milled Equivalent)',
                      'Starchy Roots + (Total)', 'Stimulants > (List)', 'Sugar Crops + (Total)', 'Vegetables + (Total)',
                      'Demersal Fish', 'Freshwater Fish',
                      'Aquatic Animals, Others', 'Pelagic Fish', 'Beer', 'Beverages, Alcoholic', 'Beverages, Fermented',
                      'Wine', 'Sugar (Raw Equivalent)', 'Sweeteners, Other', 'Vegetable Oils + (Total)',
                      'Milk - Excluding Butter + (Total)', 'Eggs + (Total)', 'Animal fats + (Total)', 'Offals + (Total)',
                      'Bovine Meat', 'Meat, Other', 'Pigmeat',
                      'Poultry Meat', 'Mutton & Goat Meat', 'Fish, Seafood + (Total)', 'Coffee and products',
                      'Grand Total + (Total)']

        # 1990 - 2013
        ld = faostat.list_datasets()
        code = 'FBSH'
        pars = faostat.list_pars(code)
        my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries] # faostat.get_par(code, 'elements')
        my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
        my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
        list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                      '2002',
                      '2003', '2004', '2005', '2006', '2007', '2008', '2009']
        my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

        my_pars = {
            'area': my_countries,
            'element': my_elements,
            'item': my_items,
            'year': my_years
        }
        df_diet_1990_2013 = faostat.get_data_df(code, pars=my_pars, strval=False)

        # 2010-2022
        list_items = ['Cereals - Excluding Beer + (Total)', 'Fruits - Excluding Wine + (Total)', 'Oilcrops + (Total)',
                      'Pulses + (Total)', 'Rice and products',
                      'Starchy Roots + (Total)', 'Stimulants > (List)', 'Sugar Crops + (Total)', 'Vegetables + (Total)',
                      'Demersal Fish', 'Freshwater Fish',
                      'Aquatic Animals, Others', 'Pelagic Fish', 'Beer', 'Beverages, Alcoholic', 'Beverages, Fermented',
                      'Wine', 'Sugar (Raw Equivalent)', 'Sweeteners, Other', 'Vegetable Oils + (Total)',
                      'Milk - Excluding Butter + (Total)', 'Eggs + (Total)', 'Animal fats + (Total)', 'Offals + (Total)',
                      'Bovine Meat', 'Meat, Other', 'Pigmeat',
                      'Poultry Meat', 'Mutton & Goat Meat', 'Fish, Seafood + (Total)', 'Coffee and products'
                      ]
        code = 'FBS'
        my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
        my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
        my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
        list_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021',
                      '2022']
        my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

        my_pars = {
            'area': my_countries,
            'element': my_elements,
            'item': my_items,
            'year': my_years
        }
        df_diet_2010_2022 = faostat.get_data_df(code, pars=my_pars, strval=False)

        # Renaming the items for name matching
        df_diet_1990_2013.loc[
            df_diet_1990_2013['Item'].str.contains('Rice \\(Milled Equivalent\\)', case=False,
                                                   na=False), 'Item'] = 'Rice and products'

        # Concatenating all the years together
        df_diet = pd.concat([df_diet_1990_2013, df_diet_2010_2022])

        # Filtering to keep wanted columns
        columns_to_filter = ['Area', 'Element', 'Item', 'Year', 'Value']
        df_diet = df_diet[columns_to_filter]

        df_diet.to_csv(file, index=False)

    # Pivot the df
    pivot_df_consumers_diet = df_diet.pivot_table(index=['Area', 'Year', 'Item'], columns='Element',
                                                            values='Value').reset_index()

    # ----------------------------------------------------------------------------------------------------------------------
    # FOOD SUPPLY Part 2 - without food waste for diet actually consumed
    # ----------------------------------------------------------------------------------------------------------------------

    # Food item name matching with dictionary
    # Read excel file
    df_dict_waste = pd.read_excel('dictionaries/dictionnary_agriculture_landuse.xlsx', sheet_name='food-waste_lifestyle')

    # Merge based on 'Item'
    pivot_df_consumers_diet = pd.merge(df_dict_waste, pivot_df_consumers_diet, on='Item')

    # Diet [kcal/cap/day] = food supply [kcal/cap/day] * (1-food waste [%])
    pivot_df_consumers_diet['value'] = pivot_df_consumers_diet['Food'] * (1 - pivot_df_consumers_diet[
        'Proportion'])

    # Drop the unused columns
    pivot_df_diet = pivot_df_consumers_diet.drop(columns=['variables', 'Food', 'Proportion'])

    # Concatenating consumer diet & share
    #pivot_df_diet = pd.concat([pivot_df_consumers_diet, pivot_df_share])

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------
    # Food item name matching with dictionary
    # Read excel file
    df_dict_diet = pd.read_excel('dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='diet_lifestyle')

    # Merge based on 'Item'
    df_diet_pathwaycalc = pd.merge(df_dict_diet, pivot_df_diet, on='Item')

    # Drop the 'Item' column
    df_diet_pathwaycalc = df_diet_pathwaycalc.drop(columns=['Item'])

    # Renaming existing columns (geoscale, timescale, value)
    df_diet_pathwaycalc.rename(columns={'Area': 'geoscale', 'Year': 'timescale'}, inplace=True)

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_diet_pathwaycalc['module'] = 'agriculture'
    df_diet_pathwaycalc['level'] = 0
    cols = df_diet_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    df_diet_pathwaycalc = df_diet_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    df_diet_pathwaycalc['geoscale'] = df_diet_pathwaycalc['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    df_diet_pathwaycalc['geoscale'] = df_diet_pathwaycalc['geoscale'].replace('Netherlands (Kingdom of the)',
                                                                              'Netherlands')
    df_diet_pathwaycalc['geoscale'] = df_diet_pathwaycalc['geoscale'].replace('Czechia', 'Czech Republic')

    # Add lever for diet-split-share
    df_diet_share = df_diet_pathwaycalc.copy()
    df_diet_share['lever'] = 'diet-split-share'

    # Extrapolating
    df_diet_share = ensure_structure(df_diet_share)
    df_diet_share = linear_fitting_ots_db(df_diet_share, years_ots, countries='all')

    # Export as datamatrix for diet-split-share
    lever = 'diet-split-share'
    df_ots, df_fts = database_to_df(df_diet_share, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm = DataMatrix.create_from_df(df_ots, num_cat=0)
    dm_diet_share = dm.filter_w_regex({'Variables': 'lfs_consumers-diet.*'})
    dm_diet_share.deepen()
    # linear fitting
    linear_fitting(dm_diet_share, years_ots)

    # CalculationLeaf FOOD WASTE PROPORTION ----------------------------------------------------------------------------------------

    # Food item name matching with dictionary
    # Read excel file
    df_dict_waste = pd.read_excel('dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='food-waste_lifestyle')

    # Merge based on 'Item'
    df_waste_pathwaycalc = pd.merge(df_dict_waste, pivot_df_diet, on='Item')

    # Food waste [-]
    df_waste_pathwaycalc['value'] = 1.0 - df_waste_pathwaycalc['Proportion']

    # Filter
    df_waste_pathwaycalc = df_waste_pathwaycalc[['Area', 'Year', 'variables', 'value']]

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------
    # Renaming existing columns (geoscale, timsecale, value)
    df_waste_pathwaycalc.rename(columns={'Area': 'geoscale', 'Year': 'timescale'}, inplace=True)

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_waste_pathwaycalc['module'] = 'agriculture'
    df_waste_pathwaycalc['lever'] = 'fwaste'
    df_waste_pathwaycalc['level'] = 0

    # Extrapolating
    df_waste_pathwaycalc = ensure_structure(df_waste_pathwaycalc)
    df_waste_pathwaycalc = linear_fitting_ots_db(df_waste_pathwaycalc, years_ots,
                                                                 countries='all')

    # Format as datamatrix
    lever = 'fwaste'
    df_ots, df_fts = database_to_df(df_waste_pathwaycalc, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm_waste = DataMatrix.create_from_df(df_ots, num_cat=1)

    # The idea was to have energy requirements per demography (agr_kcal-req) based on the current consumption and not the
    # calculated based on the metabolism, and to update the food waste values (from % to kcal/cap/day)

    # Load pickle
    with open('../../data/datamatrix/population.pickle', 'rb') as handle:
      DM_population = pickle.load(handle)
    # Filter DM
    filter_DM(DM_population, {'Country': ['Switzerland']})

    # Load data
    dm_demography = DM_population['ots']['pop']['lfs_demography_'].copy()
    dm_population = DM_population['ots']['pop']['lfs_population_'].copy()
    dm_waste_temp = dm_waste.copy()

    # for dm_diet_share: Change unit: [kt] => [kcal/cap/day]
    cdm_kcal_copy = cdm_kcal.copy()
    # Filter constants depending on dm
    cdm_kcal_diet = cdm_kcal_copy.filter(
      {'Categories1': dm_diet_share.col_labels['Categories1']})
    cdm_kcal_fwaste = cdm_kcal.copy()
    cdm_kcal_fwaste = cdm_kcal_copy.filter(
      {'Categories1': dm_waste_temp.col_labels['Categories1']})
    # Check Category order
    dm_diet_share.sort('Categories1')
    cdm_kcal_diet.sort('Categories1')
    # Unit conversion: [kt] => [kcal]
    array_temp = 10 ** 3 * dm_diet_share[:, :, 'lfs_consumers-diet', :] \
                 * cdm_kcal_diet[np.newaxis, np.newaxis, 'cp_kcal-per-t', :]
    dm_diet_share[:, :, 'lfs_consumers-diet', :] = array_temp
    # Unit conversion: [kcal] => [kcal/cap/day]
    array_temp = dm_diet_share[:, :, 'lfs_consumers-diet', :] \
                 / dm_population[:, :, 'lfs_population_total',
                   np.newaxis] / 365.25
    dm_diet_share[:, :, 'lfs_consumers-diet', :] = array_temp

    # Create copy for updating kcal-req [-]
    dm_diet_share_temp = dm_diet_share.copy()

    # For diet-split-kcal_.* : Export as dms & convert in kcal/cap/day

    dm_diet_kcal = {}
    for lever in df_diet_pathwaycalc['lever'].unique():
      # Format as dm
      df_fts_filtered = df_diet_pathwaycalc[
        df_diet_pathwaycalc['lever'] == lever]
      df_ots, df_fts = database_to_df(df_fts_filtered.copy(), lever,
                                      level='all')
      df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
      dm = DataMatrix.create_from_df(df_ots, num_cat=1)
      # linear fitting
      linear_fitting(dm, years_ots)
      dm_diet_kcal[lever] = dm
      # Unit conversion :
      cdm_kcal_copy = cdm_kcal.copy()
      # Filter constants depending on dm
      food_cat = dm_diet_kcal[lever].col_labels['Categories1']
      cdm_kcal_diet = cdm_kcal_copy.filter(
        {'Categories1': food_cat})
      # Unit conversion: [kt] => [kcal]
      array_temp = 10 ** 3 * dm_diet_kcal[lever][:, :, 'lfs_consumers-diet', :] \
                   * cdm_kcal_diet[np.newaxis, np.newaxis, 'cp_kcal-per-t', :]
      dm_diet_kcal[lever][:, :, 'lfs_consumers-diet', :] = array_temp
      # Unit conversion: [kcal] => [kcal/cap/day]
      array_temp = dm_diet_kcal[lever][:, :, 'lfs_consumers-diet', :] \
                   / dm_population[:, :, 'lfs_population_total',
                     np.newaxis] / 365.25
      dm_diet_kcal[lever][:, :, 'lfs_consumers-diet', :] = array_temp
      dm_diet_kcal[lever].change_unit('lfs_consumers-diet', old_unit='-',
                   new_unit='kcal/cap/day', factor=1)

    # Total diet [kcal/cap/day] = sum(diet consumer per category [kcal/cap/day]
    dm_diet_share_temp.group_all(dim='Categories1', inplace=True)

    # Normalise to obtain a ratio sum = 1
    dm_diet_share.normalise('Categories1', inplace=True)
    dm_diet_share.change_unit('lfs_consumers-diet', old_unit='%',
                 new_unit='-', factor=1)

    # Diet demand [kcal/day] = Diet demand [kcal/cap/day] * Population [cap]
    dm_diet_share_temp.append(dm_population, dim='Variables')
    dm_diet_share_temp.operation('lfs_consumers-diet', '*', 'lfs_population_total',
                      out_col='lfs_consumers-diet_tot', unit='kcal/day')

    # Normalise dm_kcal_req to obtain the share of kcal by age & gender categorie
    dm_kcal_req.append(dm_demography, dim='Variables')
    dm_kcal_req.operation('agr_kcal-req', '*', 'lfs_demography',
                     out_col='agr_kcal-req_req', unit='kcal/day')
    dm_kcal_req.normalise('Categories1', keep_original=True)

    # Filter for same countries
    dm_diet_share_temp.filter({'Country': dm_kcal_req.col_labels['Country']}, inplace=True)

    # Check country order
    dm_diet_share_temp.sort('Country')
    dm_kcal_req.sort('Country')

    # Demand per age gender group [kcal/day]= share kcal per age gender group [%] * total food demand [kcal/day]
    arr = dm_diet_share_temp[:, :, 'lfs_consumers-diet_tot', np.newaxis] * dm_kcal_req[:, :,'agr_kcal-req_req_share',:]
    dm_kcal_req.add(arr, dim='Variables', col_label='demand_per_group',
               unit='kcal/day')

    # Demand per age gender group [kcal/cap/day] = Demand per age gender group [kcal/day] / Demography [cap]
    dm_kcal_req.operation('demand_per_group', '/', 'lfs_demography',
                     out_col='agr_kcal-req_temp', unit='kcal/cap/day')

    # Filter and rename to only keep the values we are interested in
    dm_kcal_req.filter({'Variables':['agr_kcal-req_temp']}, inplace=True)
    dm_kcal_req.rename_col('agr_kcal-req_temp', 'agr_kcal-req', dim='Variables')

    return dm_diet_share, dm_waste, dm_kcal_req, dm_diet_kcal

# CalculationLeaf ENERGY REQUIREMENTS (OVERCONSUMPTION) -----------------------------------------------------------------------------------
def energy_requirements_processing(country_list, years_ots):
    # Calorie requirements [kcal/cap/day] = BMR * PAL = ( C(age, sex) + S (age,sex) * BW(age,sex)) * PAL
    # BMR : Basal Metabolic Rate, PAL : Physical Activity Level (kept constant), BW : Body Weight
    # C constant, S Slope (depend on age and sex groups)

    # Computing average PAL of US adult non overweight
    # SOURCE : TABLE 5.10 https://openknowledge.fao.org/server/api/core/bitstreams/62ae7aeb-9536-4e43-b2d0-55120e662824/content
    men_mean_PAL = (1.75 + 1.78 + 1.84 + 1.60 + 1.61 + 1.62 + 1.17 + 1.38) / 8
    women_mean_PAL = (1.79 + 1.83 + 1.89 + 1.75 + 1.69 + 1.55 + 1.21 + 1.17) / 8
    mean_PAL = (men_mean_PAL + women_mean_PAL) / 2

    # Compute the calorie requirements per demography (age and gender)
    # PAL is constant
    # C and S come from https://pubs.acs.org/doi/10.1021/acs.est.5b05088 Table 1
    # Body Weight (constant through years) comes from https://pubs.acs.org/doi/10.1021/acs.est.5b05088 supplementary information

    # Read and format body weight
    df_body_weight = pd.read_excel('data/body_weight.xlsx',
        sheet_name='body-weight')
    df_body_weight_melted = pd.melt(
        df_body_weight,
        id_vars=['geoscale', 'sex'],  # Columns to keep
        value_vars=['age20-29', 'age30-59', 'age60-79', 'above80'],  # Columns to unpivot
        var_name='age',  # Name for the new 'age' column
        value_name='body weight'  # Name for the new 'body weight' column
    )
    df_body_weight_melted.sort_values(by=['geoscale', 'sex'], inplace=True)

    # Read and format C and S
    df_S_C = pd.read_excel('data/body_weight.xlsx',
        sheet_name='S_C')

    # Merge df based on columns age and sex
    df_kcal_req = pd.merge(
        df_body_weight_melted,
        df_S_C,
        on=['age', 'sex'],  # Columns to merge on
        how='inner'  # Merge method: 'inner' will keep only matching rows
    )
    df_kcal_req.sort_values(by=['geoscale', 'sex'], inplace=True)

    # Add the column with the constant PAL value
    df_kcal_req['PAL'] = mean_PAL

    # Compute the calorie requirements per demography (age and gender)
    df_kcal_req['Calorie requirement per demography [kcal/person/day]'] = \
        (df_kcal_req['C (kcal)'] + df_kcal_req['S (kcal/kg)'] * df_kcal_req['body weight']) * df_kcal_req['PAL']

    # Create a new column combining age and sex and merging it with the variable names
    df_kcal_req['sex_age'] = df_kcal_req['sex'] + '_' + df_kcal_req['age']
    df_kcal_req = df_kcal_req[['geoscale', 'sex_age', 'Calorie requirement per demography [kcal/person/day]']]
    df_dict_kcal = pd.read_excel('dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='energy-req_lifestyle')
    df_kcal_req = pd.merge(df_dict_kcal, df_kcal_req, on='sex_age')
    df_kcal_req = df_kcal_req.drop(columns=['sex_age'])
    df_kcal_req.rename(columns={'Calorie requirement per demography [kcal/person/day]': 'value'}, inplace=True)

    # Rename countries to Pathaywcalc name
    df_kcal_req['geoscale'] = df_kcal_req['geoscale'].replace('Czechia', 'Czech Republic')

    # Add missing cols
    df_kcal_req['timescale'] = 2020
    df_kcal_req['module'] = 'agriculture'
    df_kcal_req['lever'] = 'kcal-req'
    df_kcal_req['level'] = 0

    # Format as datamatrix
    lever = 'kcal-req'
    df_ots, df_fts = database_to_df(df_kcal_req, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm_kcal_req = DataMatrix.create_from_df(df_ots, num_cat=0)
    dm_kcal_req.filter({'Country': country_list}, inplace=True)

    # Add missing years
    missing_years = list(set(years_ots) - set(dm_kcal_req.col_labels['Years']))
    dm_kcal_req.add(np.nan, dim='Years', dummy=True, col_label=missing_years)
    dm_kcal_req.fill_nans('Years')
    dm_kcal_req.sort('Years')

    #Have age groups as categories and rename variable
    dm_kcal_req.deepen()
    dm_kcal_req.rename_col('lfs_demography', 'agr_kcal-req', dim='Variables')
    dm_kcal_req.change_unit('agr_kcal-req', old_unit='inhabitants', new_unit='kcal/cap/day', factor=1)

    return dm_kcal_req

# CalculationLeaf SHARE DIET ADHERENCE -----------------------------------------------------------------------------------
def diet_adherence_processing(list_countries, years_ots):

  # Create df with dummy year
  df_adherence = pd.DataFrame({
    'Years': [2020],
    'Country': ['Switzerland'],
    'share_diet_adherence[-]': [0.0]
  })

  # Format as dm
  dm_adherence = DataMatrix.create_from_df(df_adherence, num_cat=0)

  # Linear fitting (1 for all ots)
  linear_fitting(dm_adherence, years_ots)

  return dm_adherence



# CalculationLeaf CAL - DIETARY HABITS -----------------------------------------------------------------------------------
def dietaryhabits_calibration(list_countries, cdm_kcal):
    # ----------------------------------------------------------------------------------------------------------------------
    # FOOD SUPPLY (DIET) ---------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # Read data ------------------------------------------------------------------------------------------------------------

    # Common for all
    # List of countries

    # FOOD BALANCE SHEETS (FBS) - -------------------------------------------------
    # List of elements
    list_elements = ['Food']
    list_items = ['Cereals - Excluding Beer + (Total)', 'Fruits - Excluding Wine + (Total)', 'Oilcrops + (Total)',
                  'Pulses + (Total)', 'Rice (Milled Equivalent)',
                  'Starchy Roots + (Total)', 'Stimulants > (List)', 'Sugar Crops + (Total)', 'Vegetables + (Total)',
                  'Demersal Fish', 'Freshwater Fish',
                  'Aquatic Animals, Others', 'Pelagic Fish', 'Beer', 'Beverages, Alcoholic', 'Beverages, Fermented',
                  'Wine', 'Sugar (Raw Equivalent)', 'Sweeteners, Other', 'Vegetable Oils + (Total)',
                  'Milk - Excluding Butter + (Total)', 'Eggs + (Total)', 'Animal fats + (Total)', 'Offals + (Total)',
                  'Bovine Meat', 'Meat, Other', 'Pigmeat',
                  'Poultry Meat', 'Mutton & Goat Meat', 'Fish, Seafood + (Total)', 'Coffee and products']

    # 1990 - 2013 - Food supply
    ld = faostat.list_datasets()
    code = 'FBSH'
    pars = faostat.list_pars(code)
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                  '2002','2003', '2004', '2005', '2006', '2007', '2008', '2009']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items,
        'year': my_years
    }
    df_diet_1990_2013 = faostat.get_data_df(code, pars=my_pars, strval=False)

    # 2010-2022
    list_elements = ['Food']
    #list_elements = ['Food supply (kcal)']
    list_items = ['Cereals - Excluding Beer + (Total)', 'Fruits - Excluding Wine + (Total)', 'Oilcrops + (Total)',
                  'Pulses + (Total)', 'Rice and products',
                  'Starchy Roots + (Total)', 'Stimulants > (List)', 'Sugar Crops + (Total)', 'Vegetables + (Total)',
                  'Demersal Fish', 'Freshwater Fish',
                  'Aquatic Animals, Others', 'Pelagic Fish', 'Beer', 'Beverages, Alcoholic', 'Beverages, Fermented',
                  'Wine', 'Sugar (Raw Equivalent)', 'Sweeteners, Other', 'Vegetable Oils + (Total)',
                  'Milk - Excluding Butter + (Total)', 'Eggs + (Total)', 'Animal fats + (Total)', 'Offals + (Total)',
                  'Bovine Meat', 'Meat, Other', 'Pigmeat',
                  'Poultry Meat', 'Mutton & Goat Meat', 'Fish, Seafood + (Total)', 'Coffee and products']
    code = 'FBS'
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021',
                  '2022']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items,
        'year': my_years
    }
    df_diet_2010_2022 = faostat.get_data_df(code, pars=my_pars, strval=False)

    df_diet_1990_2013.loc[
        df_diet_1990_2013['Item'].str.contains('Rice \\(Milled Equivalent\\)', case=False,
                                               na=False), 'Item'] = 'Rice and products'

    # Filtering to keep wanted columns
    columns_to_filter = ['Area', 'Element', 'Item', 'Year', 'Value']
    df_diet_1990_2013 = df_diet_1990_2013[columns_to_filter]
    # df_population_1990_2013 = df_population_1990_2013[columns_to_filter]
    df_diet_2010_2022 = df_diet_2010_2022[columns_to_filter]

    # Pivot the df
    pivot_df_diet_1990_2013 = df_diet_1990_2013.pivot_table(index=['Area', 'Year', 'Item'], columns='Element',
                                          values='Value').reset_index()
    #pivot_df_population_1990_2013 = df_population_1990_2013.pivot_table(index=['Area', 'Year', 'Item'], columns='Element',
    #                                                        values='Value').reset_index()
    pivot_df_diet_2010_2022 = df_diet_2010_2022.pivot_table(index=['Area', 'Year', 'Item'], columns='Element',
                                                            values='Value').reset_index()

    # Concatenating all the years together
    pivot_df_diet = pd.concat([pivot_df_diet_1990_2013, pivot_df_diet_2010_2022])

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------
    # Food item name matching with dictionary
    # Read excel file
    df_dict_calibration = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='calibration')

    # Prepend "Diet" to each value in the 'Item' column
    pivot_df_diet['Item'] = pivot_df_diet['Item'].apply(lambda x: f"Diet {x}")

    # Merge based on 'Item'
    df_diet_calibration = pd.merge(df_dict_calibration, pivot_df_diet, on='Item')

    # Drop the 'Item' column
    df_diet_calibration = df_diet_calibration.drop(columns=['Item'])

    # Renaming existing columns (geoscale, timescale, value)
    df_diet_calibration.rename(columns={'Area': 'Country', 'Year': 'Years', 'Food': 'value'}, inplace=True)

    # Change data type of timescale to int
    df_diet_calibration["Years"] = pd.to_numeric(df_diet_calibration["Years"], errors="coerce")

    # Format as datamatrix
    df_pivot = df_diet_calibration.pivot_table(index=['Country', 'Years'], columns='variables', values='value').reset_index()
    dm_cal_diet = DataMatrix.create_from_df(df_pivot, num_cat=1)

    # Note : the goal here is to convert the diet calibration values from [kt] to [kcal/cap]
    cdm_kcal_cal = cdm_kcal.copy()
    # Filter constants DROP
    cdm_kcal_cal.drop(dim='Categories1', col_label=['pro-crop-processed-molasse',
                                                'pro-crop-processed-cake',
                                                'crop-sugarcrop',
                                                'liv-meat-meal',
                                                'stm'])

    # Check Category order
    dm_cal_diet.sort('Categories1')
    cdm_kcal_cal.sort('Categories1')

    # Unit conversion: [kt] => [kcal]
    array_temp = 10 ** 3 * dm_cal_diet[:, :,
                           'cal_agr_diet', :] \
                 * cdm_kcal_cal[np.newaxis, np.newaxis, 'cp_kcal-per-t', :]
    dm_cal_diet['Switzerland', :, 'cal_agr_diet',:] = array_temp

    # Extrapolate
    linear_fitting(dm_cal_diet, years_ots)

    return dm_cal_diet

# CalculationLeaf HEALTH (SHARE WHOLE GRAINS AND PROCESSED MEAT) ---------------
def health_processing():

  # Read csv files from Global Dietary Database
  # Filter for CH, all gender, age, education level and residential urban
  # Refined grains v07
  file = 'data/GDD_FinalEstimates_01102022/Country-level estimates/v07_cnty.csv'
  df_rg = pd.read_csv(file)
  df_rg = df_rg[(df_rg['iso3'] == 'CHE') & (df_rg['age'] == 999) & (df_rg['edu'] == 999) & (df_rg['urban'] == 999) & (df_rg['female'] == 999)]
  df_rg['variables'] = 'median_refined-grains[g/cap/day]'
  # Whole grains v08
  file = 'data/GDD_FinalEstimates_01102022/Country-level estimates/v08_cnty.csv'
  df_wg = pd.read_csv(file)
  df_wg = df_wg[(df_wg['iso3'] == 'CHE') & (df_wg['age'] == 999) & (df_wg['edu'] == 999) & (df_wg['urban'] == 999) & (df_wg['female'] == 999)]
  df_wg['variables'] = 'median_whole-grains[g/cap/day]'
  # Processed meat v09
  file = 'data/GDD_FinalEstimates_01102022/Country-level estimates/v09_cnty.csv'
  df_pm = pd.read_csv(file)
  df_pm = df_pm[(df_pm['iso3'] == 'CHE') & (df_pm['age'] == 999) & (df_pm['edu'] == 999) & (df_pm['urban'] == 999) & (df_pm['female'] == 999)]
  df_pm['variables'] = 'median_processed-meat[g/cap/day]'
  # Unprocessed red meat v10
  file = 'data/GDD_FinalEstimates_01102022/Country-level estimates/v10_cnty.csv'
  df_urm = pd.read_csv(file)
  df_urm = df_urm[(df_urm['iso3'] == 'CHE') & (df_urm['age'] == 999) & (df_urm['edu'] == 999) & (df_urm['urban'] == 999) & (df_urm['female'] == 999)]
  df_urm['variables'] = 'median_unprocessed-red-meat[g/cap/day]'

  # Filter
  # Note: here median is actually mean
  list_filter = ['variables', 'year', 'median']
  df_rg = df_rg[list_filter]
  df_wg = df_wg[list_filter]
  df_pm = df_pm[list_filter]
  df_urm = df_urm[list_filter]

  # Concat
  list_concat = [df_rg, df_wg, df_pm, df_urm]
  df_food_health = pd.concat(list_concat)

  # Format as datamatrix
  df_food_health['geoscale'] = 'Switzerland'
  df_food_health['level'] = 0.0
  df_food_health['lever'] = 'dummy'
  df_food_health.rename(columns={'median': 'value', 'year': 'timescale'}, inplace=True)
  lever = 'dummy'
  df_ots, df_fts = database_to_df(df_food_health , lever, level='all')
  df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
  dm_food_health = DataMatrix.create_from_df(df_ots, num_cat=0)

  # Compute share of wholegrains
  dm_food_health.operation('median_whole-grains', '+', 'median_refined-grains',out_col='median_total-grains', unit='g/cap/day')
  dm_food_health.operation('median_whole-grains', '/', 'median_total-grains',
                           out_col='lfs_share_crop-cereal-whole', unit='-')

  # Compute share of unprocessed meat
  dm_food_health.operation('median_processed-meat', '+', 'median_unprocessed-red-meat',out_col='median_total-meat', unit='g/cap/day')
  dm_food_health.operation('median_unprocessed-red-meat', '/', 'median_total-meat',
                           out_col='lfs_share_unprocessed-meat', unit='-')

  # Filter dm
  dm_food_health.filter({'Variables':['lfs_share_crop-cereal-whole', 'lfs_share_unprocessed-meat']}, inplace=True)

  # Linear fitting
  linear_fitting(dm_food_health, years_ots)

  return dm_food_health

# CalculationLeaf CALIBRATION FORMATTING
def calibration_formatting(df_diet_calibration):

    # Concatenate dfs
    df_calibration = df_diet_calibration

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_calibration['module'] = 'agriculture'
    df_calibration['lever'] = 'none'
    df_calibration['level'] = 0
    cols = df_calibration.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    df_calibration = df_calibration[cols]

    # Rename countries to Pathaywcalc name
    df_calibration['geoscale'] = df_calibration['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    df_calibration['geoscale'] = df_calibration['geoscale'].replace('Netherlands (Kingdom of the)',
                                                                                'Netherlands')
    df_calibration['geoscale'] = df_calibration['geoscale'].replace('Czechia', 'Czech Republic')

    # Change data type of timescale to int
    df_calibration["timescale"] = pd.to_numeric(df_calibration["timescale"], errors="coerce")

    # Extrapolation for missing data
    df_calibration_struct = ensure_structure(df_calibration)
    df_calibration_ext = linear_fitting_ots_db(df_calibration_struct, years_ots,
                                                countries='all')

    # Replace values <0 with 0 for energy-demand
    # Replace negative 'value' with 0 when 'variables' contains 'energy-demand' (case-insensitive)
    mask = df_calibration_ext['variables'].str.contains('energy-demand',case=False,na=False) & (df_calibration_ext['value'] < 0)
    df_calibration_ext.loc[mask, 'value'] = 0

    # Filter to keep only data from 1990
    df_calibration_ext_agr = df_calibration_ext[df_calibration_ext["timescale"] >= 1990]

    return df_calibration_ext_agr

# CalculationLeaf CONSTANTS  ------------------------------

def constant():

  # KCAL TO T ----------------------------------------------------------------------------------------

  # Read excel
  df_kcal_t = pd.read_excel('dictionaries/kcal_to_t.xlsx',
                            sheet_name='cp_kcal_t')

  # Filter columns
  df_kcal_t = df_kcal_t[['variables', 'kcal per t']].copy()

  # Turn the df in a dict
  dict_kcal_t = dict(zip(df_kcal_t['variables'], df_kcal_t['kcal per t']))
  categories1 = df_kcal_t['variables'].tolist()

  # Format as a cdm
  cdm_kcal = ConstantDataMatrix(col_labels={'Variables': ['cp_kcal-per-t'],
                                            'Categories1': categories1})
  arr = np.zeros((len(cdm_kcal.col_labels['Variables']),
                  len(cdm_kcal.col_labels['Categories1'])))
  cdm_kcal.array = arr
  idx = cdm_kcal.idx
  for cat, val in dict_kcal_t.items():
    cdm_kcal.array[idx['cp_kcal-per-t'], idx[cat]] = val
  cdm_kcal.units["cp_kcal-per-t"] = "kcal/t"

  # TIME PER YEAR ----------------------------------------------------------------------------------------

  # Format as a cdm
  cdm_lifestyle = ConstantDataMatrix(col_labels={'Variables': ['cp_time_days-per-year']})
  arr = np.zeros((len(cdm_kcal.col_labels['Variables'])))
  cdm_lifestyle.array = arr
  idx = cdm_lifestyle.idx
  cdm_lifestyle.array[idx['cp_time_days-per-year']] = 365.0
  cdm_lifestyle.units["cp_time_days-per-year"] = "days/year"

  return cdm_kcal, cdm_lifestyle

# CalculationLeaf FTS  ------------------------------
def fts_processing(list_countries, years_ots, years_fts, cdm_kcal):

  # fwaste, diet-adherence, kcal-req -------------------------------------------
  # Read Excel
  df_fts_data = pd.read_excel(
    'data/dietary-habits_fts.xlsx',
    sheet_name='fts')
  df_fts_data = df_fts_data[['variables', 'timescale', 'geoscale', 'level', 'value', 'lever']]

  # Format as dms for each lever
  dm_fts = {}
  for lever in df_fts_data['lever'].unique():
    dm = {}
    for level in df_fts_data['level'].unique():
      df_fts_filtered = df_fts_data[df_fts_data['level'] == level]
      df_fts_filtered = df_fts_filtered[df_fts_filtered['lever'] == lever]
      df_ots, df_fts = database_to_df(df_fts_filtered.copy(), lever, level='all')
      df_fts = df_fts.drop(columns=[lever])  # Drop column with lever name
      dm[level] = DataMatrix.create_from_df(df_fts, num_cat=0)
    dm_fts[lever] = dm

  # diet-split-kcal ---------------------------------------------------------------
  # Read Excel
  df_fts_diet_kcal = pd.read_excel(
      'data/dietary-habits_fts.xlsx',
    sheet_name='diet-split-kcal')
  df_fts_diet_kcal = df_fts_diet_kcal[
      ['variables', 'timescale', 'lever', 'level_1', 'level_2', 'level_3', 'level_4']]

  # Melt
  df_fts_diet_kcal = df_fts_diet_kcal.melt(
    id_vars=['variables', 'timescale', 'lever'],
    var_name='level_name',
    value_name='value'
  )

  # Associate a level for each diet
  # Mapping
  level_map = {
    'level_1': 1,
    'level_2': 2,
    'level_3': 3,
    'level_4': 4,

  }

  # Map lever strings to numbers
  df_fts_diet_kcal['level'] = df_fts_diet_kcal['level_name'].map(level_map)

  # Format as dm
  df_fts_diet_kcal['geoscale'] = 'Switzerland'

  for lever in df_fts_diet_kcal['lever'].unique():
    dm = {}
    for level in df_fts_diet_kcal['level'].unique():
      df_fts_filtered = df_fts_diet_kcal[df_fts_diet_kcal['level'] == level]
      df_fts_filtered = df_fts_filtered[df_fts_filtered['lever'] == lever]
      df_ots, df_fts = database_to_df(df_fts_filtered.copy(), lever, level='all')
      df_fts = df_fts.drop(columns=[lever])  # Drop column with lever name
      if (lever == 'share-kcal-processed-food_crop-cereal-whole') | (lever == 'share-kcal-processed-food_unprocessed-meat'):
        dm[level] = DataMatrix.create_from_df(df_fts, num_cat=0)
      else:
        dm[level] = DataMatrix.create_from_df(df_fts, num_cat=1)
    dm_fts[lever] = dm

  #dm = {}
  dm_fts_cereal = {}
  """for level in df_fts_diet_kcal['level'].unique():

    df_fts_filtered = df_fts_diet_kcal[df_fts_diet_kcal['level'] == level]
    df_ots, df_fts = database_to_df(df_fts_filtered.copy(), lever, level='all')
    df_fts = df_fts.drop(columns=[lever])  # Drop column with lever name
    dm[level] = DataMatrix.create_from_df(df_fts, num_cat=0)
    # Filter for diet-split-kcal
    dm_kcal = dm.copy()
    dm_kcal[level].filter_w_regex({'Variables': 'lfs_consumers-diet.*'}, inplace=True)
    # Filter & deepen for share wholegrains & processed meat

  dm_fts[lever] = dm_kcal"""

  # diet-split-share ---------------------------------------------------------------
  # Read Excel
  df_fts_diet = pd.read_excel(
      'data/dietary-habits_fts.xlsx',
    sheet_name='diet-split-share')
  df_fts_diet = df_fts_diet[
      ['variables', 'diet_eat-lancet-phd-2025', 'diet_eat-lancet-phd-2019', 'diet_sfp-2024']]

  # Fill nan with 0.0
  df_fts_diet.fillna(0.0, inplace=True)

  # Melt
  df_fts_diet = df_fts_diet.melt(
    id_vars='variables',
    var_name='level_name',
    value_name='value'
  )

  # Associate a level for each diet
  # Mapping
  level_map = {
    'diet_eat-lancet-phd-2025': 2,
    'diet_eat-lancet-phd-2019': 3,
    'diet_sfp-2024': 4
  }

  # Map lever strings to numbers
  df_fts_diet['level'] = df_fts_diet['level_name'].map(level_map)

  # Format as dm
  df_fts_diet['timescale'] = 2050
  df_fts_diet['geoscale'] = 'Switzerland'
  lever = 'diet-split-share'
  df_fts_diet['lever'] = lever
  dm = {}
  dm_fts_meat = {}
  dm_fts_cereal = {}

  for level in df_fts_diet['level'].unique():
    df_fts_filtered = df_fts_diet[df_fts_diet['level'] == level]
    df_ots, df_fts = database_to_df(df_fts_filtered.copy(), lever, level='all')
    df_fts = df_fts.drop(columns=[lever])  # Drop column with lever name
    dm[level] = DataMatrix.create_from_df(df_fts, num_cat=1)
    # Compute share of whole cereals
    dm_cereal = dm[level].filter({'Categories1':['crop-cereal-whole', 'crop-cereal-refined']}).copy()
    dm_cereal_tot = dm_cereal.copy()
    dm_cereal_tot.groupby({'crop-cereal': 'crop-cereal.*'}, regex=True, inplace=True, dim='Categories1')
    dm_cereal.append(dm_cereal_tot,dim='Categories1')
    dm_cereal = dm_cereal.flatten()
    dm_cereal.operation('lfs_consumers-diet_crop-cereal-whole', '/', 'lfs_consumers-diet_crop-cereal',out_col='lfs_share_crop-cereal-whole', unit='-')
    dm_cereal.filter({'Variables': ['lfs_share_crop-cereal-whole']}, inplace=True)
    dm_fts_cereal[level] = dm_cereal
    # Compute share of unprocessed meat
    dm_meat = dm[level].filter_w_regex({'Categories1': 'pro-liv-meat.*'})
    dm_meat_tot = dm_meat.copy()
    dm_meat_tot.groupby({'pro-liv-meat-total': 'pro-liv-meat.*'}, regex=True, inplace=True, dim='Categories1')
    dm_meat.append(dm_meat_tot,dim='Categories1')
    dm_meat = dm_meat.flatten()
    dm_meat.operation('lfs_consumers-diet_pro-liv-meat-unprocessed', '/', 'lfs_consumers-diet_pro-liv-meat-total',out_col='lfs_share_unprocessed-meat', unit='-')
    dm_meat.filter({'Variables':['lfs_share_unprocessed-meat']}, inplace=True)
    dm_fts_meat[level] = dm_meat
    # cereals = cereals-whole + cereals-refined
    dm[level].groupby({'crop-cereal': 'crop-cereal.*'}, regex=True, inplace=True, dim='Categories1')
    # oilcrops = oilcrops + treenuts
    dm[level].groupby({'crop-oilcrop': '.*oilcrop|.*treenut'}, regex=True, inplace=True, dim='Categories1')
    # Drop unprocessed-meat
    dm[level].drop(dim='Categories1', col_label=['pro-liv-meat-unprocessed'])

  dm_fts[lever] = dm
  dm_fts['share-processed-food_unprocessed-meat'] = dm_fts_meat
  dm_fts['share-processed-food_crop-cereal-whole'] = dm_fts_cereal

  # Convert in kcal
  # Filter constants
  cdm_kcal_copy = cdm_kcal.copy()
  cdm_kcal_copy.drop(dim='Categories1', col_label=['pro-crop-processed-molasse',
                                              'pro-crop-processed-cake',
                                              'crop-sugarcrop',
                                              'liv-meat-meal',
                                              'stm'])
  lever = 'diet-split-share'
  for level in df_fts_diet['level'].unique():
    dm_diet_share = dm_fts[lever][level].copy()
    # Check Category order
    dm_diet_share.sort('Categories1')
    cdm_kcal_copy.sort('Categories1')
    # Unit conversion: [g/cap/day] => [kcal/cap/day]
    array_temp = 10**(-6) * dm_diet_share[:, :,'lfs_consumers-diet', :] \
                 * cdm_kcal_copy[np.newaxis, np.newaxis, 'cp_kcal-per-t', :]
    dm_fts[lever][level]['Switzerland', :, 'lfs_consumers-diet', :] = array_temp
    # Normalise to compute share
    dm_fts[lever][level].normalise(dim='Categories1', inplace=True)
    # Change unit
    dm_fts[lever][level].change_unit('lfs_consumers-diet', old_unit='%',
                            new_unit='-', factor=1)

  return dm_fts

# CalculationLeaf PICKLE CREATION ------------------------------

def datamatrix_to_pickle(years_ots, years_fts, dm_waste, dm_kcal_req, dm_cal_diet, dm_diet_share, dm_diet_kcal, dm_adherence, dm_food_health, cdm_kcal, cdm_lifestyle, dm_fts):

  # Make list with all years
  years_all = years_ots + years_fts

  # FixedAssumptionsToDatamatrix -----------------------------------------------
  dict_fxa = {}

  # CalibrationDataToDatamatrix ------------------------------------------------

  # Diet
  dict_fxa['cal_agr_diet'] = dm_cal_diet

  # LeversToDatamatrix OTS -----------------------------------------------------
  dict_ots = {}

  # Diet-split-share
  dict_ots['diet-split-share'] = dm_diet_share

  # Share processed food (with diet-split-share)
  dict_ots['share-processed-food_crop-cereal-whole'] = dm_food_health.filter({'Variables':['lfs_share_crop-cereal-whole']})
  dict_ots['share-processed-food_unprocessed-meat'] = dm_food_health.filter({'Variables':['lfs_share_unprocessed-meat']})

  # Diet-split-kcal
  for lever in dm_diet_kcal.keys():
    dict_ots[lever] = dm_diet_kcal[lever]

  # Share processed food (with diet-split-kcal)
  dict_ots['share-kcal-processed-food_crop-cereal-whole'] = dm_food_health.filter({'Variables':['lfs_share_crop-cereal-whole']})
  dict_ots['share-kcal-processed-food_unprocessed-meat'] = dm_food_health.filter({'Variables':['lfs_share_unprocessed-meat']})

  # Food waste
  dict_ots['fwaste'] = dm_waste

  # Energy requirements
  dict_ots['kcal-req'] = dm_kcal_req

  # Share diet adherence
  dict_ots['diet-adherence'] = dm_adherence



  # LeversToDatamatrix FTS -----------------------------------------------------
  dict_fts = {}

  # FTS linear fitting of ots
  DM_ots = dict_ots.copy()

  # Adding a new lever with dummy values
  """dict_temp = {}
  dict_fts['diet-split-share'] = {'diet-split-share': dict()}
  dict_fts['diet-split-kcal'] = {'diet-split-kcal': dict()}
  dict_fts['fwaste'] = {'fwaste': dict()}
  dict_fts['kcal-req'] = {'kcal-req': dict()}
  dict_fts['diet-adherence'] = {'diet-adherence': dict()}"""

  # Levers to be normalised
  list_norm = ['climate-smart-livestock_ration']

  """for key in DM_ots.keys():
    if isinstance(DM_ots[key], dict):
      for subkey in DM_ots[key].keys():
        dm = DM_ots[key][subkey].copy()
        linear_fitting(dm, years_fts)

        for lev in range(1, 5):  # 1 to 4
          if subkey in list_norm:  # ✅ check subkey, not key
            dm_norm = dm.copy()
            # Replace negative values with 0
            array_temp = dm_norm.array[:, :, :, :]
            array_temp[array_temp < 0] = 0.0
            dm_norm.array[:, :, :, :] = array_temp
            # Normalise
            dm_norm.normalise(dim='Categories1', inplace=True)
            dict_fts[key][subkey][lev] = dm_norm.filter(
              {'Years': years_fts}, inplace=False
            )
          else:
            dict_fts[key][subkey][lev] = dm.filter(
              {'Years': years_fts}, inplace=False
            )
    else:
      dm = DM_ots[key].copy()
      linear_fitting(dm, years_fts)
      for lev in range(1, 5):
        dict_fts[key][lev] = dm.filter({'Years': years_fts}, inplace=False)"""

  # Linear fitting between ots and fts objective (2050) ------------------

  # Lever - diet-adherence
  lever = 'diet-adherence'
  for level in range(1,5):
    dm_fts[lever][level].append(dict_ots[lever], dim='Years')
    linear_fitting(dm_fts[lever][level], years_fts)
    dm_fts[lever][level].filter({'Years':years_fts}, inplace=True)
  dict_fts[lever] = dm_fts[lever]

  # Lever - fwaste
  lever = 'fwaste'
  for level in range(1,5):
    # Compute the reduction objective in 2050 compared to the last ots value,
    # for each food category
    dm_ots = dict_ots[lever].copy()
    array_temp =  1 - ( 1 - dm_ots[:,years_ots[-1],'lfs_consumers-food-wastes',:]) \
                  * dm_fts[lever][level][:,years_fts[-1],'lfs_consumers-food-wastes', np.newaxis]
    # Append with ots
    dm_ots.add(array_temp[:,np.newaxis,np.newaxis,:], dim='Years', dummy=True, col_label=years_fts[-1])
    # Linear fit
    linear_fitting(dm_ots, years_fts)
    dm_fts[lever][level] = dm_ots.filter({'Years':years_fts}, inplace=False)
  dict_fts[lever] = dm_fts[lever]

  # Lever - kcal-req
  lever = 'kcal-req'
  for level in range(1,5):
    # Compute the reduction objective in 2050 compared to the last ots value,
    # for each food category
    dm_ots = dict_ots[lever].copy()
    array_temp = dm_ots[:,years_ots[-1],'agr_kcal-req',:] \
                  * dm_fts[lever][level][:,years_fts[-1],'agr_kcal-req', np.newaxis]
    # Append with ots
    dm_ots.add(array_temp[:,np.newaxis,np.newaxis,:], dim='Years', dummy=True, col_label=years_fts[-1])
    # Linear fit
    linear_fitting(dm_ots, years_fts)
    dm_fts[lever][level] = dm_ots.filter({'Years':years_fts}, inplace=False)
  dict_fts[lever] = dm_fts[lever]

  # Lever - diet-split-share
  lever = 'diet-split-share'
  for level in range(2,5):
    dm_fts[lever][level].append(dict_ots[lever], dim='Years')
    linear_fitting(dm_fts[lever][level], years_fts)
    dm_fts[lever][level].filter({'Years':years_fts}, inplace=True)
  dict_fts[lever] = dm_fts[lever]
  # Compute BAU scenario level 1
  level = 1
  dm_fts[lever][level] = dict_ots[lever].copy()
  linear_fitting(dm_fts[lever][level], years_fts)
  dm_fts[lever][level].filter({'Years': years_fts}, inplace=True)
  dict_fts[lever][level] = dm_fts[lever][level]

  # Lever - diet-split-kcal_.*
  for lever in dm_diet_kcal.keys():
    for level in range(1,5):
      dm_fts[lever][level].append(dict_ots[lever], dim='Years')
      linear_fitting(dm_fts[lever][level], years_fts)
      dm_fts[lever][level].filter({'Years':years_fts}, inplace=True)
    dict_fts[lever] = dm_fts[lever]

  # Lever - share-processed-food_crop-cereal-whole
  lever = 'share-processed-food_crop-cereal-whole'
  for level in range(2,5):
    dm_fts[lever][level].append(dict_ots[lever], dim='Years')
    linear_fitting(dm_fts[lever][level], years_fts)
    dm_fts[lever][level].filter({'Years':years_fts}, inplace=True)
  dict_fts[lever] = dm_fts[lever]
  # Compute BAU scenario level 1
  level = 1
  dm_fts[lever][level] = dict_ots[lever].copy()
  linear_fitting(dm_fts[lever][level], years_fts)
  dm_fts[lever][level].filter({'Years': years_fts}, inplace=True)
  dict_fts[lever][level] = dm_fts[lever][level]

  # Lever - share-processed-food_unprocessed-meat
  lever = 'share-processed-food_unprocessed-meat'
  for level in range(2,5):
    dm_fts[lever][level].append(dict_ots[lever], dim='Years')
    linear_fitting(dm_fts[lever][level], years_fts)
    dm_fts[lever][level].filter({'Years':years_fts}, inplace=True)
  dict_fts[lever] = dm_fts[lever]
  # Compute BAU scenario level 1
  level = 1
  dm_fts[lever][level] = dict_ots[lever].copy()
  linear_fitting(dm_fts[lever][level], years_fts)
  dm_fts[lever][level].filter({'Years': years_fts}, inplace=True)
  dict_fts[lever][level] = dm_fts[lever][level]

  # Lever - share-kcal-processed-food_.*
  lever = 'share-kcal-processed-food_unprocessed-meat'
  for level in range(1,5):
    dm_fts[lever][level].append(dict_ots[lever], dim='Years')
    linear_fitting(dm_fts[lever][level], years_fts)
    dm_fts[lever][level].filter({'Years':years_fts}, inplace=True)
  dict_fts[lever] = dm_fts[lever]
  lever = 'share-kcal-processed-food_crop-cereal-whole'
  for level in range(1,5):
    dm_fts[lever][level].append(dict_ots[lever], dim='Years')
    linear_fitting(dm_fts[lever][level], years_fts)
    dm_fts[lever][level].filter({'Years':years_fts}, inplace=True)
  dict_fts[lever] = dm_fts[lever]

  # ConstantsToDatamatrix ------------------------------------------------------
  dict_const = {}
  dict_const['cdm_kcal-per-t'] = cdm_kcal
  dict_const['cdm_lifestyle'] = cdm_lifestyle

  # Group all datamatrix in a single structure ---------------------------------
  DM_diet = {
    'fxa': dict_fxa,
    'constant': dict_const,
    'fts': dict_fts,
    'ots': dict_ots
  }

  # Write datamatrix to pickle -------------------------------------------------
  f = '../../data/datamatrix/dietary-habits.pickle'
  with open(f, 'wb') as handle:
    pickle.dump(DM_diet, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return


# CalculationTree RUNNING PRE-PROCESSING -----------------------------------------------------------------------------------------------
years_ots = create_years_list(1990, 2023, 1)  # make list with years from 1990 to 2015
years_fts = create_years_list(2025, 2050, 5)
years_all = years_ots + years_fts

if not os.path.exists('data/faostat'):
    os.makedirs('data/faostat')

list_countries = ['Switzerland']

cdm_kcal, cdm_lifestyle = constant()
dm_cal_diet = dietaryhabits_calibration(list_countries, cdm_kcal)
dm_kcal_req_temp = energy_requirements_processing(list_countries, years_ots)
file = 'data/faostat/diet.csv' # Create file for storing data
dm_diet_share, dm_waste, dm_kcal_req, dm_diet_kcal = diet_processing(list_countries, file, cdm_kcal, dm_kcal_req_temp)
dm_adherence = diet_adherence_processing(list_countries, years_ots)
dm_fts = fts_processing(list_countries, years_ots, years_fts, cdm_kcal)
dm_food_health = health_processing()


# CalculationTree RUNNING PICKLE CREATION
datamatrix_to_pickle(years_ots, years_fts, dm_waste, dm_kcal_req, dm_cal_diet, dm_diet_share, dm_diet_kcal, dm_adherence, dm_food_health, cdm_kcal, cdm_lifestyle, dm_fts)
