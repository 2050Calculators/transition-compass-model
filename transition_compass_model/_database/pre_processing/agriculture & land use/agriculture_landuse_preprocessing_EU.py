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
from model.common.auxiliary_functions import compute_stock,  filter_geoscale, calibration_rates, filter_DM, add_dummy_country_to_DM
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


# CalculationLeaf DIET (LIFESTYLE) ------------------------------------------------------------------------------------
def diet_processing(list_countries, file):
    # ----------------------------------------------------------------------------------------------------------------------
    # CONSUMER DIET Part 1 - including food waste
    # ----------------------------------------------------------------------------------------------------------------------

    # Read data ------------------------------------------------------------------------------------------------------------
    try:
        df_diet = pd.read_csv(file)
    except OSError:

        # FOOD BALANCE SHEETS (FBS) - -------------------------------------------------
        # List of elements
        list_elements = ['Food supply (kcal/capita/day)']

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
                      'Poultry Meat', 'Mutton & Goat Meat', 'Fish, Seafood + (Total)', 'Coffee and products',
                      'Grand Total + (Total)']
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
            df_diet_1990_2013['Item'].str.contains('Rice \(Milled Equivalent\)', case=False,
                                                   na=False), 'Item'] = 'Rice and products'

        # Concatenating all the years together
        df_diet = pd.concat([df_diet_1990_2013, df_diet_2010_2022])

        # Filtering to keep wanted columns
        columns_to_filter = ['Area', 'Element', 'Item', 'Year', 'Value']
        df_diet = df_diet[columns_to_filter]

        df_diet.to_csv(file, index=False)

    # Filter to only have meat, fish, sugar, sweet, veg, fruits (the rest goes to share for further pre-processing)
    list_consumers_diet = ['Fruits - Excluding Wine', 'Vegetables',
                           'Demersal Fish', 'Freshwater Fish',
                           'Aquatic Animals, Others', 'Pelagic Fish',
                           'Sugar (Raw Equivalent)', 'Sweeteners, Other',
                           'Bovine Meat', 'Meat, Other', 'Pigmeat',
                           'Poultry Meat', 'Mutton & Goat Meat', 'Fish, Seafood']
    pattern_consumers_diet = '|'.join(re.escape(item) for item in list_consumers_diet)
    list_share = ['Cereals - Excluding Beer', 'Oilcrops',
                  'Pulses', 'Rice and products',
                  'Starchy Roots',
                  'Beer', 'Beverages, Alcoholic', 'Beverages, Fermented',
                  'Wine', 'Vegetable Oils',
                  'Milk - Excluding Butter', 'Eggs',
                  'Animal fats', 'Offals', 'Coffee and products', 'Cocoa Beans and products', 'Tea (including mate)', 'Grand Total']
    pattern_share = '|'.join(re.escape(item) for item in list_share)
    df_consumers_diet = df_diet[df_diet['Item'].str.contains(pattern_consumers_diet, case=False)]
    df_share = df_diet[df_diet['Item'].str.contains(pattern_share, case=False)]

    # Pivot the df
    pivot_df_consumers_diet = df_consumers_diet.pivot_table(index=['Area', 'Year', 'Item'], columns='Element',
                                                            values='Value').reset_index()
    # Rename columns
    #pivot_df_consumers_diet.rename(columns={'Food supply (kcal/capita/day)': 'value'}, inplace=True)

    # ----------------------------------------------------------------------------------------------------------------------
    # SHARE (FOR OTHER PRODUCTS)
    # ----------------------------------------------------------------------------------------------------------------------

    # Pivot the df
    pivot_df_share = df_share.pivot_table(index=['Area', 'Year', 'Item'], columns='Element',
                                          values='Value').reset_index()

    # Creating a column with the Grand Total kcal/cap/day
    # Step 1: Extract Grand Total for each Year and Area
    grand_totals = pivot_df_share[pivot_df_share['Item'] == 'Grand Total'][
        ['Area', 'Year', 'Food supply (kcal/capita/day)']]
    grand_totals.rename(columns={'Food supply (kcal/capita/day)': 'Grand Total'}, inplace=True)
    # Step 2: Merge the Grand Total back into the original DataFrame
    pivot_df_share = pivot_df_share.merge(grand_totals, on=['Area', 'Year'], how='left')
    # Step 3: Drop rows where Item is 'Grand Total'
    pivot_df_share = pivot_df_share[pivot_df_share['Item'] != 'Grand Total']

    # Divide the consumption by the total kcal to obtain the share
    #pivot_df_share['value'] = pivot_df_share['Food supply (kcal/capita/day)'] / pivot_df_share['Grand Total']
    pivot_df_share['value'] = pivot_df_share['Food supply (kcal/capita/day)'] # Test with Paola 28.05.2025

    # Drop the columns
    pivot_df_share = pivot_df_share.drop(columns=['Food supply (kcal/capita/day)', 'Grand Total'])

    # Normalize so that for each year and country, sum(share) = 1
    #pivot_df_share['value'] = pivot_df_share['value'] / pivot_df_share.groupby(['Area', 'Year'])['value'].transform(
    #    'sum')

    # ----------------------------------------------------------------------------------------------------------------------
    # CONSUMER DIET Part 2 - including food waste
    # ----------------------------------------------------------------------------------------------------------------------

    # Food item name matching with dictionary
    # Read excel file
    df_dict_waste = pd.read_excel('dictionaries/dictionnary_agriculture_landuse.xlsx', sheet_name='food-waste_lifestyle')

    # Merge based on 'Item'
    pivot_df_consumers_diet = pd.merge(df_dict_waste, pivot_df_consumers_diet, on='Item')

    # Diet [kcal/cap/day] = food supply [kcal/cap/day] * (1-food waste [%])
    pivot_df_consumers_diet['value'] = pivot_df_consumers_diet['Food supply (kcal/capita/day)'] * (1 - pivot_df_consumers_diet[
        'Proportion'])

    # Drop the unused columns
    pivot_df_consumers_diet = pivot_df_consumers_diet.drop(columns=['variables', 'Food supply (kcal/capita/day)', 'Proportion'])

    # Concatenating consumer diet & share
    pivot_df_diet = pd.concat([pivot_df_consumers_diet, pivot_df_share])

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------
    # Food item name matching with dictionary
    # Read excel file
    df_dict_diet = pd.read_excel('dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='diet_lifestyle')

    # Merge based on 'Item'
    df_diet_pathwaycalc = pd.merge(df_dict_diet, pivot_df_diet, on='Item')

    # Drop the 'Item' column
    df_diet_pathwaycalc = df_diet_pathwaycalc.drop(columns=['Item'])

    # Renaming existing columns (geoscale, timsecale, value)
    df_diet_pathwaycalc.rename(columns={'Area': 'geoscale', 'Year': 'timescale'}, inplace=True)

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_diet_pathwaycalc['module'] = 'agriculture'
    df_diet_pathwaycalc['lever'] = 'diet'
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

    # Extrapolating
    df_diet_pathwaycalc = ensure_structure(df_diet_pathwaycalc)
    df_diet_pathwaycalc = linear_fitting_ots_db(df_diet_pathwaycalc, years_ots,
                                                                 countries='all')
    return df_diet_pathwaycalc, df_diet


# CalculationLeaf FOOD WASTE (LIFESTYLE) -----------------------------------------------------------------------------------
def food_waste_processing(df_diet):
    # Pivot the df
    pivot_df_diet = df_diet.pivot_table(index=['Area', 'Year', 'Item'], columns='Element',
                                        values='Value').reset_index()

    # Food item name matching with dictionary
    # Read excel file
    df_dict_waste = pd.read_excel('dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='food-waste_lifestyle')

    # Merge based on 'Item'
    df_waste_pathwaycalc = pd.merge(df_dict_waste, pivot_df_diet, on='Item')

    # Food waste [kcal/cap/day] = food supply [kcal/cap/day] * food waste [%]
    df_waste_pathwaycalc['value'] = df_waste_pathwaycalc['Food supply (kcal/capita/day)'] * df_waste_pathwaycalc[
        'Proportion']

    # Drop the unused columns
    df_waste_pathwaycalc = df_waste_pathwaycalc.drop(columns=['Item', 'Food supply (kcal/capita/day)', 'Proportion'])

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------
    # Renaming existing columns (geoscale, timsecale, value)
    df_waste_pathwaycalc.rename(columns={'Area': 'geoscale', 'Year': 'timescale'}, inplace=True)

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_waste_pathwaycalc['module'] = 'agriculture'
    df_waste_pathwaycalc['lever'] = 'fwaste'
    df_waste_pathwaycalc['level'] = 0
    cols = df_waste_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    df_waste_pathwaycalc = df_waste_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    df_waste_pathwaycalc['geoscale'] = df_waste_pathwaycalc['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    df_waste_pathwaycalc['geoscale'] = df_waste_pathwaycalc['geoscale'].replace('Netherlands (Kingdom of the)',
                                                                                'Netherlands')
    df_waste_pathwaycalc['geoscale'] = df_waste_pathwaycalc['geoscale'].replace('Czechia', 'Czech Republic')

    # Extrapolating
    df_waste_pathwaycalc = ensure_structure(df_waste_pathwaycalc)
    df_waste_pathwaycalc = linear_fitting_ots_db(df_waste_pathwaycalc, years_ots,
                                                                 countries='all')

    return df_waste_pathwaycalc

# CalculationLeaf ENERGY REQUIREMENTS -----------------------------------------------------------------------------------
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


# CalculationLeaf SELF-SUFFICIENCY CROP & LIVESTOCK ------------------------------------------------------------------------------
def self_sufficiency_processing(years_ots, list_countries, file_dict):
    # Read data ------------------------------------------------------------------------------------------------------------
    try:
        df_ssr = pd.read_csv(file_dict['ssr'])
    except OSError:

        # FOOD BALANCE SHEETS (FBS) - For everything except molasses and cakes -------------------------------------------------
        # List of elements
        list_elements = ['Production Quantity', 'Import Quantity', 'Export Quantity', 'Feed']

        list_items = ['Cereals - Excluding Beer + (Total)', 'Fruits - Excluding Wine + (Total)', 'Oilcrops + (Total)',
                      'Pulses + (Total)', 'Rice (Milled Equivalent)',
                      'Starchy Roots + (Total)', 'Stimulants > (List)', 'Sugar Crops + (Total)', 'Vegetables + (Total)',
                      'Demersal Fish', 'Freshwater Fish',
                      'Aquatic Animals, Others', 'Pelagic Fish', 'Beer', 'Beverages, Alcoholic', 'Beverages, Fermented',
                      'Wine', 'Sugar (Raw Equivalent)', 'Sweeteners, Other', 'Vegetable Oils + (Total)',
                      'Milk - Excluding Butter + (Total)', 'Eggs + (Total)', 'Animal fats + (Total)', 'Offals + (Total)',
                      'Bovine Meat', 'Meat, Other', 'Pigmeat',
                      'Poultry Meat', 'Mutton & Goat Meat', 'Fish, Seafood + (Total)']

        # 1990 - 2013
        ld = faostat.list_datasets()
        code = 'FBSH'
        pars = faostat.list_pars(code)
        my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
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
        df_ssr_1990_2013 = faostat.get_data_df(code, pars=my_pars, strval=False)
        # Renaming the elements
        df_ssr_1990_2013.loc[df_ssr_1990_2013['Element'].str.contains('Production Quantity', case=False, na=False), 'Element'] = 'Production'
        df_ssr_1990_2013.loc[
            df_ssr_1990_2013['Element'].str.contains('Import Quantity', case=False, na=False), 'Element'] = 'Import'
        df_ssr_1990_2013.loc[
            df_ssr_1990_2013['Element'].str.contains('Export Quantity', case=False, na=False), 'Element'] = 'Export'

        # 2010 - 2022

        list_elements = ['Production Quantity', 'Import quantity', 'Export quantity', 'Feed']
        # Different list becuse different in item nomination such as rice
        list_items = ['Cereals - Excluding Beer + (Total)', 'Fruits - Excluding Wine + (Total)', 'Oilcrops + (Total)',
                      'Pulses + (Total)', 'Rice and products',
                      'Starchy Roots + (Total)', 'Stimulants > (List)', 'Sugar Crops + (Total)', 'Vegetables + (Total)',
                      'Demersal Fish', 'Freshwater Fish',
                      'Aquatic Animals, Others', 'Pelagic Fish', 'Beer', 'Beverages, Alcoholic', 'Beverages, Fermented',
                      'Wine', 'Sugar (Raw Equivalent)', 'Sweeteners, Other', 'Vegetable Oils + (Total)',
                      'Milk - Excluding Butter + (Total)', 'Eggs + (Total)', 'Animal fats + (Total)', 'Offals + (Total)',
                      'Bovine Meat', 'Meat, Other', 'Pigmeat',
                      'Poultry Meat', 'Mutton & Goat Meat', 'Fish, Seafood + (Total)']
        code = 'FBS'
        my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
        my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
        my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
        list_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
        my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

        my_pars = {
            'area': my_countries,
            'element': my_elements,
            'item': my_items,
            'year': my_years
        }
        df_ssr_2010_2021 = faostat.get_data_df(code, pars=my_pars, strval=False)

        # Renaming the elements
        df_ssr_2010_2021.loc[
            df_ssr_2010_2021['Element'].str.contains('Production Quantity', case=False, na=False), 'Element'] = 'Production'
        df_ssr_2010_2021.loc[
            df_ssr_2010_2021['Element'].str.contains('Import quantity', case=False, na=False), 'Element'] = 'Import'
        df_ssr_2010_2021.loc[
            df_ssr_2010_2021['Element'].str.contains('Export quantity', case=False, na=False), 'Element'] = 'Export'
        df_ssr = pd.concat([df_ssr_1990_2013, df_ssr_2010_2021])

        # Renaming the items for name matching
        df_ssr.loc[
            df_ssr['Item'].str.contains('Rice \(Milled Equivalent\)', case=False,
                                                   na=False), 'Item'] = 'Rice and products'

        df_ssr.to_csv(file_dict['ssr'], index=False)

    # COMMODITY BALANCES (NON-FOOD) (OLD METHODOLOGY) - For molasse and cakes ----------------------------------------------
    try:
        df_ssr_cake = pd.read_csv(file_dict['cake'])
        df_ssr_2010_2021_molasse_cake = pd.read_csv(file_dict['molasse'])
    except OSError:
        # 1990 - 2013
        list_elements = ['Production Quantity', 'Import quantity', 'Export quantity', 'Feed']
        list_items = ['Copra Cake', 'Cottonseed Cake', 'Groundnut Cake', 'Oilseed Cakes, Other', 'Palmkernel Cake',
                      'Rape and Mustard Cake', 'Sesameseed Cake', 'Soyabean Cake', 'Sunflowerseed Cake']
        code = 'CBH'
        my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
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
        df_ssr_1990_2013_cake = faostat.get_data_df(code, pars=my_pars, strval=False)
        # Renaming the elements
        df_ssr_1990_2013_cake.loc[
            df_ssr_1990_2013_cake['Element'].str.contains('Production Quantity', case=False, na=False), 'Element'] = 'Production'
        df_ssr_1990_2013_cake.loc[
            df_ssr_1990_2013_cake['Element'].str.contains('Import quantity', case=False, na=False), 'Element'] = 'Import'
        df_ssr_1990_2013_cake.loc[
            df_ssr_1990_2013_cake['Element'].str.contains('Export Quantity', case=False, na=False), 'Element'] = 'Export'


        # SUPPLY UTILIZATION ACCOUNTS (SCl) - For molasse and cakes ----------------------------------------------------------
        # 2010 - 2022
        list_elements = ['Production Quantity', 'Import quantity', 'Export quantity', 'Feed']
        list_items = ['Molasses', 'Cake of  linseed', 'Cake of  soya beans', 'Cake of copra', 'Cake of cottonseed',
                      'Cake of groundnuts', 'Cake of hempseed', 'Cake of kapok', 'Cake of maize', 'Cake of mustard seed',
                      'Cake of palm kernel', 'Cake of rapeseed', 'Cake of rice bran', 'Cake of safflowerseed',
                      'Cake of sesame seed', 'Cake of sunflower seed', 'Cake, oilseeds nes', 'Cake, poppy seed',
                      'Cocoa powder and cake']
        code = 'SCL'
        my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
        my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
        my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
        list_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
        my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

        my_pars = {
            'area': my_countries,
            'element': my_elements,
            'item': my_items,
            'year': my_years
        }
        df_ssr_2010_2021_molasse_cake = faostat.get_data_df(code, pars=my_pars, strval=False)

        # Renaming the elements
        df_ssr_2010_2021_molasse_cake.loc[
            df_ssr_2010_2021_molasse_cake['Element'].str.contains('Production Quantity', case=False, na=False), 'Element'] = 'Production'
        df_ssr_2010_2021_molasse_cake.loc[
            df_ssr_2010_2021_molasse_cake['Element'].str.contains('Import quantity', case=False, na=False), 'Element'] = 'Import'
        df_ssr_2010_2021_molasse_cake.loc[
            df_ssr_2010_2021_molasse_cake['Element'].str.contains('Export quantity', case=False, na=False), 'Element'] = 'Export'

        # Aggregating cakes
        df_ssr_cake = pd.concat([df_ssr_1990_2013_cake, df_ssr_2010_2021_molasse_cake])

        df_ssr_cake.to_csv(file_dict['cake'], index=False)
        df_ssr_2010_2021_molasse_cake.to_csv(file_dict['molasse'], index=False)

    # Filtering
    filtered_df = df_ssr_cake[df_ssr_cake['Item'].str.contains('cake', case=False)]
    # Groupby Area, Year and Element and sum the Value
    grouped_df = filtered_df.groupby(['Area', 'Element', 'Year'])['Value'].sum().reset_index()
    # Adding a column 'Item' containing 'Cakes' for all row, before the 'Value' column
    grouped_df['Item'] = 'Cakes'
    cols = grouped_df.columns.tolist()
    cols.insert(cols.index('Value'), cols.pop(cols.index('Item')))
    df_ssr_cake = grouped_df[cols]

    # Filtering for molasse
    df_ssr_molasses = df_ssr_2010_2021_molasse_cake[
        df_ssr_2010_2021_molasse_cake['Item'].str.contains('Molasses', case=False)]

    # Concatenating for feed
    #df_ssr = pd.concat([df_ssr, df_ssr_molasses])
    #df_ssr = pd.concat([df_ssr, df_ssr_cake])
    df_ssr_feed = pd.concat([df_ssr_molasses, df_ssr_cake])

    # Filtering to keep wanted columns
    columns_to_filter = ['Area', 'Element', 'Item', 'Year', 'Value']
    df_ssr = df_ssr[columns_to_filter]
    df_ssr_feed = df_ssr_feed[columns_to_filter]

    # Compute Self-Sufficiency Ratio (SSR) ---------------------------------------------------------------------------------
    # SSR [%] = (100*Production) / (Production + Imports - Exports)
    # Step 1: Pivot the DataFrame to get 'Production', 'Import Quantity', and 'Export Quantity' in separate columns
    pivot_df = df_ssr.pivot_table(index=['Area', 'Year', 'Item'], columns='Element', values='Value').reset_index()
    pivot_df_feed = df_ssr_feed.pivot_table(index=['Area', 'Year', 'Item'],
                                  columns='Element',
                                  values='Value').reset_index()

    # Fill na with 0
    pivot_df['Production'].fillna(0.0, inplace=True)
    pivot_df['Import'].fillna(0.0, inplace=True)
    pivot_df['Export'].fillna(0.0, inplace=True)
    pivot_df_feed['Production'].fillna(0.0, inplace=True)
    pivot_df_feed['Import'].fillna(0.0, inplace=True)
    pivot_df_feed['Export'].fillna(0.0, inplace=True)

    # Create a copy for feed pre-processing and drop irrelevant columns
    df_csl_feed = pd.concat([pivot_df, pivot_df_feed])
    df_csl_feed = df_csl_feed.drop(columns=['Production', 'Import', 'Export'])

    # Step 2: Compute the SSR [%]
    # Note : Update - the SSR is now computed afterwards for calibration reasons, in order to match it with the demand
    pivot_df['SSR[%]'] = (pivot_df['Production'])
    pivot_df_feed['SSR[%]'] = (pivot_df_feed['Production']) / (
                pivot_df_feed['Production'] + pivot_df_feed['Import'] - pivot_df_feed['Export'])

    # Concat dfs
    pivot_df = pd.concat([pivot_df, pivot_df_feed])

    # Drop the columns Production, Import Quantity and Export Quantity
    pivot_df = pivot_df.drop(columns=['Production', 'Import', 'Export', 'Feed'])

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------

    # Food item name matching with dictionary
    # Read excel file
    df_dict_ssr = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='self-sufficiency')

    # Merge based on 'Item'
    df_ssr_pathwaycalc = pd.merge(df_dict_ssr, pivot_df, on='Item')

    # Drop the 'Item' column
    df_ssr_pathwaycalc = df_ssr_pathwaycalc.drop(columns=['Item'])

    # Renaming existing columns (geoscale, timsecale, value)
    df_ssr_pathwaycalc.rename(columns={'Area': 'geoscale', 'Year': 'timescale', 'SSR[%]': 'value'}, inplace=True)

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_ssr_pathwaycalc['module'] = 'agriculture'
    df_ssr_pathwaycalc['lever'] = 'food-net-import'
    df_ssr_pathwaycalc['level'] = 0
    cols = df_ssr_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    df_ssr_pathwaycalc = df_ssr_pathwaycalc[cols]
    df_ssr_pathwaycalc = df_ssr_pathwaycalc.drop_duplicates()

    # Rename countries to Pathaywcalc name
    df_ssr_pathwaycalc['geoscale'] = df_ssr_pathwaycalc['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    df_ssr_pathwaycalc['geoscale'] = df_ssr_pathwaycalc['geoscale'].replace('Netherlands (Kingdom of the)',
                                                                            'Netherlands')
    df_ssr_pathwaycalc['geoscale'] = df_ssr_pathwaycalc['geoscale'].replace('Czechia', 'Czech Republic')

    # Extrapolation
    df_ssr_pathwaycalc = linear_fitting_ots_db(df_ssr_pathwaycalc, years_ots, countries='all')

    return df_ssr_pathwaycalc, df_csl_feed

# CalculationLeaf CLIMATE SMART CROP ---------------------------------------------------------------------------------------------
def climate_smart_crop_processing(list_countries, df_agri_land, file_dict):
    # ENERGY DEMAND --------------------------------------------------------------------------------------------------------

    # Importing UNFCCC excel files and reading them with a loop (only for Switzerland) Table1.A(a)s4 ---------------------------
    # Putting in a df in 3 dimensions (from, to, year)
    # Define the path where the Excel files are located
    folder_path = 'data/data_unfccc_2023'

    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter and sort files by the year (1990 to 2020)
    sorted_files = sorted([f for f in files if f.startswith('CHE_2023_') and int(f.split('_')[2]) in range(1990, 2021)],
                          key=lambda x: int(x.split('_')[2]))

    # Initialize a list to store DataFrames
    data_frames = []

    # Loop through sorted files, read the required rows, and append to the list
    for file in sorted_files:
        # Extract the year from the filename
        year = int(file.split('_')[2])

        # Full path to the file
        file_path = os.path.join(folder_path, file)

        # Read the specific rows and sheet from the Excel file
        df = pd.read_excel(file_path, sheet_name='Table1.A(a)s4', skiprows=53, nrows=15, header=None)

        # Add a column for the year to the DataFrame
        df['Year'] = year

        # Append to the list of DataFrames
        data_frames.append(df)

    # Combine all DataFrames into a single DataFrame with a multi-index
    combined_df = pd.concat(data_frames, axis=0).set_index(['Year'])

    # Replace NO with 0
    combined_df = combined_df.replace('NO', 0.0)

    # Rename columns
    combined_df.rename(columns={0: 'Item', 1:'Consumption [TJ]', 6:'CO2 emissions [kt]'}, inplace=True)
    combined_df = combined_df.reset_index().rename(columns={'Year': 'timescale'})
    my_items_list = ['i. Stationary',
                     'ii. Off-road vehicles and other machinery']
    combined_df = combined_df[~combined_df['Item'].isin(my_items_list)].copy() # Drop rows where Item is in my_items_list
    df_energy = combined_df[['timescale', 'Item', 'Consumption [TJ]']].copy()
    df_energy = df_energy.rename(columns={'Consumption [TJ]': 'value'})
    df_CO2_cal = combined_df[['timescale', 'Item','CO2 emissions [kt]']].copy()

    # Prep CO2 cal
    df_CO2_cal = df_CO2_cal[['timescale', 'CO2 emissions [kt]']].copy()
    df_CO2_cal = df_CO2_cal.groupby(['timescale'], as_index=False)[
      'CO2 emissions [kt]'].sum()
    df_CO2_cal['Item'] = 'CO2 emissions fuel'

    # Sum for the same item per year
    df_energy = df_energy.groupby(['timescale', 'Item'], as_index=False)[
      'value'].sum()

    # Keep only the correct rows
    my_items_list = ['Liquid fuels', 'Solid fuels', 'Gaseous fuels', 'Gasoline', 'Diesel oil',
                     'Liquefied petroleum gases (LPG)', 'Biomass(6)']
    df_energy = df_energy[df_energy['Item'].isin(my_items_list)]

    # Add dummy items
    # Define your dummy items
    dummy_items = ['Biogas (dummy)', 'Biodiesel (dummy)', 'Ethanol (dummy)',
                   'Liquid oth (dummy)', 'Heat (dummy)', 'Electricity (dummy)',
                   'Others (dummy)']
    # Step 1: Get unique timescales
    timescales = df_energy['timescale'].unique()
    # Step 2: Create a list of dicts for new rows
    new_rows = []
    for ts in timescales:
      for di in dummy_items:
        new_rows.append({
          'timescale': ts,
          'Item': di,
          'value': 0.0
        })
    # Step 3: Convert to DataFrame
    df_dummies = pd.DataFrame(new_rows)
    # Step 4: Concatenate
    df_energy_demand = pd.concat([df_energy, df_dummies], ignore_index=True)

    # convert from [TJ] to [ktoe]
    tj_to_ktoe = 0.02388458966275  # source https://www.unitjuggler.com/convertir-energy-de-TJ-en-kltoe.htm
    df_energy_demand.loc[:, df_energy_demand.columns == 'value'] *= tj_to_ktoe

    '''# ENERGY DEMAND --------------------------------------------------------------------------------------------------------
    # Read excel
    df_energy = pd.read_excel(
        'data/Energy_demand_agriculture_CH.xlsx',
        sheet_name='Di und indi Energie 2021',
        skiprows = 0,
        nrows = 8
    )
    df_energy = df_energy.drop(columns=['Unit'])
    df_energy.rename(columns={'Énergie directe': 'Item'}, inplace=True)

    # Unit conversion [GJ] => [ktoe]
    # convert from [TJ] to [ktoe]
    gj_to_ktoe = 0.00002388458966275  # source https://www.unitjuggler.com/convertir-energy-de-TJ-en-ktoe.html
    df_energy.loc[:, df_energy.columns != 'Item'] *= gj_to_ktoe

    # Add dummy rows
    # Identify year columns
    year_cols = [col for col in df_energy.columns if col != 'Item']
    # Define your dummy items
    dummy_items = ['Biogas (dummy)', 'Biodiesel (dummy)', 'Ethanol (dummy)',
                   'Liquid oth (dummy)', 'Heat (dummy)', 'LPG (dummy)',
                   'Others (dummy)', 'Coal (dummy)']
    # Create a list of dicts for each dummy
    dummy_rows = []
    for dummy in dummy_items:
      row = {'Item': dummy}
      for year in year_cols:
        row[year] = 0.0
      dummy_rows.append(row)
    # Convert to DataFrame
    df_dummies = pd.DataFrame(dummy_rows)
    # Append to original df
    df_energy = pd.concat([df_energy, df_dummies], ignore_index=True)

    # Melt
    df_energy_demand = df_energy.melt(
      id_vars='Item',  # Columns to keep fixed
      var_name='timescale',  # Name for the new 'item' column
      value_name='value'  # Name for the new 'value' column
    )'''


    '''# BIOENERGIES
    # Read excel
    df_bioenergy = pd.read_excel(
        'data/statistiques_energie_2023.xlsx',
        sheet_name='T34b',
        skiprows = 7,
        nrows = 27
    )
    df_bioenergy = df_bioenergy[['timescale', 'Biodiesel', 'Bioéthanol / Biométhanol', "Biocarburants d'aviation", 'Huiles vég. / anim.']]

    # convert from [GWh] to [ktoe]
    gwh_to_ktoe = 0.085984522785899  # source https://www.unitjuggler.com/convertir-energy-de-TJ-en-ktoe.html
    df_bioenergy.loc[:, df_bioenergy.columns != 'timescale'] *= gwh_to_ktoe

    # OTHER ENERGIES
    df_oth_energy = pd.read_excel(
        'data/statistiques_energie_2023.xlsx',
        sheet_name='T17d',
        skiprows=10,
        nrows=44
    )
    df_oth_energy = df_oth_energy[
        ['timescale', 'Energie du bois', 'Electricité', 'Gaz', 'Chaleur à distance', 'Charbon', 'Autres énergies renouvelables']]

    # Replace all occurrences of '-' with 0.0
    df_oth_energy = df_oth_energy.replace('-', 0.0)

    # Convert numeric columns to float (if necessary)
    df_oth_energy.iloc[:, 1:] = df_oth_energy.iloc[:, 1:].astype(float)

    # Keep only the years starting from 1990
    df_oth_energy = df_oth_energy[df_oth_energy["timescale"] >= 1990]

    # convert from [TJ] to [ktoe]
    tj_to_ktoe = 0.02388458966275  # source https://www.unitjuggler.com/convertir-energy-de-TJ-en-ktoe.html
    df_oth_energy.loc[:, df_oth_energy.columns != 'timescale'] *= tj_to_ktoe

    # PETROLEUM PRODUCTS
    df_petroleum = pd.read_excel(
        'data/statistiques_energie_2023.xlsx',
        sheet_name='T20',
        skiprows=6,
        nrows=51
    )
    # convert from [kt] to [ktoe]
    kt_to_ktoe = 1.05  # https://enerteam.org/conversion-to-toe.html
    df_petroleum.loc[:, df_petroleum.columns != 'timescale'] *= kt_to_ktoe

    # BIOGAS
    df_biogas = pd.read_excel(
        'data/statistiques_energie_2023.xlsx',
        sheet_name='T34a',
        skiprows=6,
        nrows=35
    )
    df_biogas = df_biogas[
        ['timescale', 'Biogas cons. Agr']]

    # convert from [GWh] to [ktoe]
    gwh_to_ktoe = 0.085984522785899 # source https://www.unitjuggler.com/convertir-energy-de-TJ-en-ktoe.html
    df_biogas.loc[:, df_biogas.columns != 'timescale'] *= gwh_to_ktoe

    # Merge (concat not possible due to different years)
    df_energy_demand = pd.merge(df_bioenergy, df_oth_energy, on='timescale', how='outer')
    df_energy_demand = pd.merge(df_energy_demand, df_petroleum, on='timescale', how='outer')
    df_energy_demand = pd.merge(df_energy_demand, df_biogas, on='timescale', how='outer')

    # Fill nan with 0.0
    df_energy_demand[:].fillna(0.0, inplace=True)

    # Biodisel = huiles végétales animales + biodiesel
    df_energy_demand['Biodiesel'] = df_energy_demand['Biodiesel'] + df_energy_demand['Huiles vég. / anim.']

    # Oth energies = other renouvelables energies
    df_energy_demand['Other energies'] = df_energy_demand['Autres énergies renouvelables']

    # Ajouter colonnes avec 0
    df_energy_demand['LPG'] = 0.0
    df_energy_demand['Other bioenergy liquids'] = 0.0

    # Pivot
    df_energy_demand = df_energy_demand.melt(
        id_vars='timescale',  # Columns to keep fixed
        var_name='Item',  # Name for the new 'item' column
        value_name='value'  # Name for the new 'value' column
    )'''

    # Create copy for calibration
    df_energy_demand_cal = df_energy_demand.copy()
    df_energy_demand_cal['geoscale'] = 'Switzerland'
    df_energy_demand_cal = df_energy_demand_cal.drop_duplicates()

    # convert from ktoe to ktoe/ha (divide by total agricultural area) -------------------------------------------------
    # Read FAO Values (for Switzerland)
    # List of countries
    list_countries_CH = ['Switzerland']

    # List of elements
    list_elements = ['Area']

    list_items = ['-- Cropland', '-- Permanent meadows and pastures']

    # 1990 - 2022
    try:
        df_land_use = pd.read_csv(file_dict['land'])
    except OSError:
        ld = faostat.list_datasets()
        code = 'RL'
        pars = faostat.list_pars(code)
        my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries_CH]
        my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
        my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
        list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                      '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                      '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
        my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

        my_pars = {
            'area': my_countries,
            'element': my_elements,
            'item': my_items,
            'year': my_years
        }
        df_land_use = faostat.get_data_df(code, pars=my_pars, strval=False)

        # Filtering to keep wanted columns
        columns_to_filter = ['Area', 'Item', 'Year', 'Value']
        df_land_use = df_land_use[columns_to_filter]
        df_land_use.to_csv(file_dict['land'], index=False)

    # Filer land for Switzerland and drop Area
    df_land_use = df_agri_land[df_agri_land['Area'].isin(['Switzerland'])]
    df_land_use = df_land_use.drop(columns=['Area'])
    df_land_use.rename(columns={'Year': 'timescale'}, inplace=True)

    # Merge and divide [kha]
    df_land_use['timescale'] = df_land_use['timescale'].astype(str)  # Convert to string
    df_energy_demand['timescale'] = df_energy_demand['timescale'].astype(str)  # Convert to string
    df_combined = pd.merge(
        df_energy_demand,
        df_land_use,
        on='timescale',
        how='inner'  # Use 'inner' to keep only matching rows
    )
    df_combined['value'] = df_combined['value'] / df_combined['Agricultural land [ha]']
    # Read excel file
    df_dict_csc = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='climate-smart-crops')

    # Merge based on 'Item'
    df_energy_pathwaycalc = pd.merge(df_dict_csc, df_combined, on='Item')

    # Drop the 'Item' column
    df_energy_pathwaycalc = df_energy_pathwaycalc.drop(columns=['Item', 'Agricultural land [ha]'])

    # Add a geoscale column
    df_energy_pathwaycalc['geoscale'] = 'Switzerland'

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_energy_pathwaycalc['module'] = 'agriculture'
    df_energy_pathwaycalc['lever'] = 'climate-smart-crop'
    df_energy_pathwaycalc['level'] = 0
    cols = df_energy_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    df_energy_pathwaycalc = df_energy_pathwaycalc[cols]

    # ----------------------------------------------------------------------------------------------------------------------
    # INPUT USE ------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # NITROGEN, PHOSPHATE, POTASH ------------------------------------------------------------------------------------------
    try:
        df_input_nitrogen_1990_2021 = pd.read_csv(file_dict['nitro'])
    except OSError:
        # List of elements
        list_elements = ['Agricultural Use']

        list_items = ['Nutrient nitrogen N (total)', 'Nutrient phosphate P2O5 (total)', 'Nutrient potash K2O (total)']

        # 1990 - 2021
        ld = faostat.list_datasets()
        code = 'RFN'
        pars = faostat.list_pars(code)
        my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
        my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
        my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
        list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                      '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                      '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
        my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

        my_pars = {
            'area': my_countries,
            'element': my_elements,
            'item': my_items,
            'year': my_years
        }
        df_input_nitrogen_1990_2021 = faostat.get_data_df(code, pars=my_pars, strval=False)
        df_input_nitrogen_1990_2021 = df_input_nitrogen_1990_2021.drop(
          columns=['Domain Code', 'Domain', 'Area Code', 'Element Code',
                   'Item Code', 'Year Code', 'Unit', 'Element'])

        df_input_nitrogen_1990_2021.to_csv(file_dict['nitro'], index=False)

    # PESTICIDES -----------------------------------------------------------------------------------------------------------
    try:
        df_input_pesticides_1990_2021 = pd.read_csv(file_dict['pesticide'])
    except OSError:
        # List of elements
        list_elements = ['Agricultural Use']

        list_items = ['Pesticides (total) + (Total)']

        # 1990 - 2021
        code = 'RP'
        my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
        my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
        my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
        list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                      '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                      '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
        my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

        my_pars = {
            'area': my_countries,
            'element': my_elements,
            'item': my_items,
            'year': my_years
        }
        df_input_pesticides_1990_2021 = faostat.get_data_df(code, pars=my_pars, strval=False)
        df_input_pesticides_1990_2021 = df_input_pesticides_1990_2021.drop(
          columns=['Domain Code', 'Domain', 'Area Code', 'Element Code',
                   'Item Code', 'Year Code', 'Unit', 'Element'])
        df_input_pesticides_1990_2021.to_csv(file_dict['pesticide'], index=False)

    # LIMING, UREA ---------------------------------------------------------------------------------------------------------
    try:
        df_input_urea_1990_2021 = pd.read_csv(file_dict['urea'])
        df_input_liming_1990_2021 = pd.read_csv(file_dict['liming'])
    except OSError:
        # List of elements
        list_elements = ['Agricultural Use']

        list_items = ['Urea', 'Calcium ammonium nitrate (CAN) and other mixtures with calcium carbonate']

        # Input Liming Urea 2002 - 2021
        code = 'RFB'
        my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
        my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
        my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
        list_years = ['2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                      '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
        my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

        my_pars = {
            'area': my_countries,
            'element': my_elements,
            'item': my_items,
            'year': my_years
        }
        df_input_liming_urea_1990_2021 = faostat.get_data_df(code, pars=my_pars, strval=False)

        '''# Area Harvested 2002 - 2021

        # List of elements
        list_elements = ['Area harvested']
        list_items = ['Cereals, primary + (Total)', 'Fibre Crops, Fibre Equivalent + (Total)', 'Fruit Primary + (Total)',
                      'Oilcrops, Oil Equivalent + (Total)', 'Pulses, Total + (Total)', 'Rice',
                      'Roots and Tubers, Total + (Total)',
                      'Sugar Crops Primary + (Total)', 'Vegetables Primary + (Total)']
        code = 'QCL'
        my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
        my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
        my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
        list_years = ['2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                      '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
        my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

        my_pars = {
            'area': my_countries,
            'element': my_elements,
            'item': my_items,
            'year': my_years
        }
        df_area_2022_2021 = faostat.get_data_df(code, pars=my_pars, strval=False)

        # Conversion from [t] in [t/ha]-----------------------------------------------------------------------------------------
        # Summming Area harvested per country and year (and element)
        df_area_total_2022_2021 = df_area_2022_2021.groupby(['Area', 'Element', 'Year'])['Value'].sum().reset_index()'''

        # UREA
        # Filtering and dropping columns
        df_input_urea_1990_2021 = df_input_liming_urea_1990_2021[df_input_liming_urea_1990_2021['Item'] == 'Urea']
        df_input_urea_1990_2021 = df_input_urea_1990_2021.drop(
            columns=['Domain Code', 'Domain', 'Area Code', 'Element Code',
                     'Item Code', 'Year Code', 'Unit', 'Element'])

        # LIMING
        # Filtering and dropping columns
        df_input_liming_1990_2021 = df_input_liming_urea_1990_2021[df_input_liming_urea_1990_2021[
                                                                       'Item'] == 'Calcium ammonium nitrate (CAN) and other mixtures with calcium carbonate']
        df_input_liming_1990_2021 = df_input_liming_1990_2021.drop(
            columns=['Domain Code', 'Domain', 'Area Code', 'Element Code',
                     'Item Code', 'Year Code', 'Unit', 'Element'])

        df_input_liming_1990_2021.to_csv(file_dict['liming'], index=False)
        df_input_urea_1990_2021.to_csv(file_dict['urea'], index=False)

    # Concatenate inputs
    df_input = pd.concat([df_input_urea_1990_2021, df_input_liming_1990_2021])
    df_input = pd.concat([df_input, df_input_pesticides_1990_2021])
    df_input = pd.concat([df_input, df_input_nitrogen_1990_2021])

    # Pivot
    pivot_df = df_input.pivot_table(index=['Area', 'Year'], columns='Item',
                                        values='Value').reset_index()

    # Fil na with zeros
    #pivot_df[:].fillna(0.0, inplace=True)

    # Merge inputs with agricultural land
    pivot_df['Year'] = pivot_df['Year'].astype(str)
    df_input_land = pd.merge(pivot_df, df_agri_land, on=['Area', 'Year'])

    # Compute the use per land [t/ha]
    # Identify the columns to divide (exclude Year, Area, Agricultural land)
    cols_to_divide = df_input_land.columns.difference(
      ['Year', 'Area', 'Agricultural land [ha]'])
    # Divide each of those columns by 'Agricultural land [ha]'
    df_input_land[cols_to_divide] = df_input_land[cols_to_divide].div(df_input_land['Agricultural land [ha]'],
                                                axis=0)

    # Melt the DataFrame
    df_input_land = df_input_land.melt(
      id_vars=['Year', 'Area'],  # columns to keep fixed
      var_name='Item',  # name of the new 'item' column
      value_name='value'  # name of the new 'value' column
    )

    # Food item name matching with dictionary
    # Read excel file
    df_dict_csc = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='climate-smart-crops')

    # Merge based on 'Item'
    df_input_pathwaycalc = pd.merge(df_dict_csc, df_input_land, on='Item')

    # Drop the 'Item' column
    df_input_pathwaycalc = df_input_pathwaycalc.drop(columns=['Item'])

    # Renaming existing columns (geoscale, timsecale, value)
    df_input_pathwaycalc.rename(columns={'Area': 'geoscale', 'Year': 'timescale'}, inplace=True)

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_input_pathwaycalc['module'] = 'agriculture'
    df_input_pathwaycalc['lever'] = 'climate-smart-crop'
    df_input_pathwaycalc['level'] = 0
    cols = df_input_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    df_input_pathwaycalc = df_input_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    df_input_pathwaycalc['geoscale'] = df_input_pathwaycalc['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    df_input_pathwaycalc['geoscale'] = df_input_pathwaycalc['geoscale'].replace('Netherlands (Kingdom of the)',
                                                                                'Netherlands')
    df_input_pathwaycalc['geoscale'] = df_input_pathwaycalc['geoscale'].replace('Czechia', 'Czech Republic')

    # ----------------------------------------------------------------------------------------------------------------------
    # EF AGROFORESTRY ------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    # Is equal to 0 for all ots for all countries

    # Use pivot_df_input as a structural basis
    agroforestry_crop = df_input_land.copy()

    # Drop the column Item
    agroforestry_crop = agroforestry_crop.drop(columns=['Item', 'value'])

    # Rename the column in geoscale and timescale
    agroforestry_crop.rename(columns={'Area': 'geoscale', 'Year': 'timescale'}, inplace=True)

    # Changing data type to numeric (except for the geoscale column)
    agroforestry_crop.loc[:, agroforestry_crop.columns != 'geoscale'] = agroforestry_crop.loc[:,
                                                                        agroforestry_crop.columns != 'geoscale'].apply(
        pd.to_numeric, errors='coerce')

    # Add rows to have 1990-2022
    # Generate a DataFrame with all combinations of geoscale and timescale
    geoscale_values = agroforestry_crop['geoscale'].unique()
    timescale_values = pd.Series(range(1990, 2023))

    # Create a DataFrame for the cartesian product
    cartesian_product = pd.MultiIndex.from_product([geoscale_values, timescale_values],
                                                   names=['geoscale', 'timescale']).to_frame(index=False)



    # Merge the original DataFrame with the cartesian product to include all combinations
    agroforestry_crop = pd.merge(cartesian_product, agroforestry_crop, on=['geoscale', 'timescale'], how='left')

    # Add the variables with a value of 0
    agroforestry_crop['agr_climate-smart-crop_ef_agroforestry_cover-crop[tC/ha]'] = 0
    agroforestry_crop['agr_climate-smart-crop_ef_agroforestry_cropland[tC/ha]'] = 0
    agroforestry_crop['agr_climate-smart-crop_ef_agroforestry_hedges[tC/ha]'] = 0
    agroforestry_crop['agr_climate-smart-crop_ef_agroforestry_no-till[tC/ha]'] = 0

    # Melt the df
    agroforestry_crop_pathwaycalc = pd.melt(agroforestry_crop, id_vars=['timescale', 'geoscale'],
                                           value_vars=['agr_climate-smart-crop_ef_agroforestry_cover-crop[tC/ha]',
                                                       'agr_climate-smart-crop_ef_agroforestry_cropland[tC/ha]',
                                                       'agr_climate-smart-crop_ef_agroforestry_hedges[tC/ha]',
                                                       'agr_climate-smart-crop_ef_agroforestry_no-till[tC/ha]'],
                                           var_name='variables', value_name='value')

    # PathwayCalc formatting
    agroforestry_crop_pathwaycalc['module'] = 'agriculture'
    agroforestry_crop_pathwaycalc['lever'] = 'climate-smart-crop'
    agroforestry_crop_pathwaycalc['level'] = 0
    cols = agroforestry_crop_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    cols.insert(cols.index('timescale'), cols.pop(cols.index('variables')))
    agroforestry_crop_pathwaycalc = agroforestry_crop_pathwaycalc[cols]


    # ----------------------------------------------------------------------------------------------------------------------
    # LOSSES ---------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # FOOD BALANCE SHEETS (FBS) - For everything  -------------------------------------------------
    try:
        df_losses = pd.read_csv(file_dict['losses'])
    except OSError:
        # List of elements
        list_elements = ['Losses', 'Production Quantity']

        list_items = ['Cereals - Excluding Beer + (Total)', 'Fruits - Excluding Wine + (Total)', 'Oilcrops + (Total)',
                      'Pulses + (Total)', 'Rice (Milled Equivalent)', 'Starchy Roots + (Total)', 'Sugar Crops + (Total)',
                      'Vegetables + (Total)', ]

        # 1990 - 2013
        code = 'FBSH'
        my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
        my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
        my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
        list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                      '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009']
        my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

        my_pars = {
            'area': my_countries,
            'element': my_elements,
            'item': my_items,
            'year': my_years
        }
        df_losses_1990_2013 = faostat.get_data_df(code, pars=my_pars, strval=False)

        # 2010 - 2022
        # Different list because different in item nomination such as rice
        list_items = ['Cereals - Excluding Beer + (Total)', 'Fruits - Excluding Wine + (Total)', 'Oilcrops + (Total)',
                      'Pulses + (Total)', 'Rice and products', 'Starchy Roots + (Total)', 'Sugar Crops + (Total)',
                      'Vegetables + (Total)', ]
        code = 'FBS'
        my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
        my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
        my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
        list_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
        my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

        my_pars = {
            'area': my_countries,
            'element': my_elements,
            'item': my_items,
            'year': my_years
        }
        df_losses_2010_2021 = faostat.get_data_df(code, pars=my_pars, strval=False)

        # Renanming rice to have same name with other df
        df_losses_1990_2013['Item'] = df_losses_1990_2013['Item'].replace('Rice (Milled Equivalent)', 'Rice and products')

        # Concatenating
        df_losses = pd.concat([df_losses_1990_2013, df_losses_2010_2021])
        df_losses.to_csv(file_dict['losses'], index=False)

    # Compute losses ([%] of production) -----------------------------------------------------------------------------------
    # Losses [%] = 1 / (1 - Losses [1000t] / Production [1000t]) (pre processing for multiplicating the workflow)

    # Step 1: Pivot the DataFrame
    pivot_df = df_losses.pivot_table(index=['Area', 'Year', 'Item'], columns='Element', values='Value').reset_index()

    # Step 2: Compute the Losses [%] (really it's unit less)
    pivot_df['Losses[%]'] = 1 / (1 - pivot_df['Losses'] / pivot_df['Production'])

    # Drop the columns Production, Import Quantity and Export Quantity
    pivot_df = pivot_df.drop(columns=['Production', 'Losses'])


    # Create a dummy for Rice as no products
    # Create a DataFrame for the new "Rice" rows
    new_rows = pivot_df[['Area', 'Year']].drop_duplicates().copy()
    new_rows['Item'] = 'Rice and products'
    new_rows['Losses[%]'] = 0

    # Append the new rows to the original DataFrame
    pivot_df = pd.concat([pivot_df, new_rows], ignore_index=True)

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------

    # Food item name matching with dictionary
    # Read excel file
    df_dict_csc = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='climate-smart-crops')

    # Merge based on 'Item'
    df_losses_pathwaycalc = pd.merge(df_dict_csc, pivot_df, on='Item')

    # Drop the 'Item' column
    df_losses_pathwaycalc = df_losses_pathwaycalc.drop(columns=['Item'])

    # Renaming existing columns (geoscale, timsecale, value)
    df_losses_pathwaycalc.rename(columns={'Area': 'geoscale', 'Year': 'timescale', 'Losses[%]': 'value'}, inplace=True)

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_losses_pathwaycalc['module'] = 'agriculture'
    df_losses_pathwaycalc['lever'] = 'climate-smart-crop'
    df_losses_pathwaycalc['level'] = 0
    cols = df_losses_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    df_losses_pathwaycalc = df_losses_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    df_losses_pathwaycalc['geoscale'] = df_losses_pathwaycalc['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    df_losses_pathwaycalc['geoscale'] = df_losses_pathwaycalc['geoscale'].replace('Netherlands (Kingdom of the)',
                                                                                  'Netherlands')
    df_losses_pathwaycalc['geoscale'] = df_losses_pathwaycalc['geoscale'].replace('Czechia', 'Czech Republic')

    # RESIDUE SHARE --------------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------------
    # YIELD ----------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # CROPS  (QCL) (for everything except lgn-energycrop, gas-energycrop, algae and insect)
    try:
        df_yield_1990_2022 = pd.read_csv(file_dict['yield'])
    except OSError:
        # List of elements
        list_elements = ['Yield']

        list_items = ['Cereals, primary + (Total)', 'Fibre Crops, Fibre Equivalent + (Total)', 'Fruit Primary + (Total)',
                      'Oilcrops, Oil Equivalent + (Total)', 'Pulses, Total + (Total)', 'Rice',
                      'Roots and Tubers, Total + (Total)',
                      'Sugar Crops Primary + (Total)', 'Vegetables Primary + (Total)']

        # 1990 - 2022
        code = 'QCL'
        my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
        my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
        my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
        list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                      '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                      '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
        my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

        my_pars = {
            'area': my_countries,
            'element': my_elements,
            'item': my_items,
            'year': my_years
        }
        df_yield_1990_2022 = faostat.get_data_df(code, pars=my_pars, strval=False)
        df_yield_1990_2022.loc[
            df_yield_1990_2022['Item'].str.contains('Rice', case=False,
                                                    na=False), 'Item'] = 'Rice and products'
        df_yield_1990_2022.to_csv(file_dict['yield'], index=False)

    # Unit conversion from [kg/ha] to [kcal/ha]  ----------------------------------------------------------------------------

    # Pivot the DataFrame
    pivot_df = df_yield_1990_2022.pivot_table(index=['Area', 'Year', 'Item'], columns='Element',
                                              values='Value').reset_index()

    # DataFrame with only 'Fibre Crops, Fibre Equivalent'
    df_fibre = pivot_df[pivot_df['Item'] == 'Fibre Crops, Fibre Equivalent']
    df_fibre = df_fibre.copy()
    df_fibre.rename(columns={'Value': 'Yield'}, inplace=True)

    # DataFrame with all other items
    df_other_items = pivot_df[pivot_df['Item'] != 'Fibre Crops, Fibre Equivalent']

    # Read excel
    df_kcal_t = pd.read_excel(
        'dictionaries/kcal_to_t.xlsx',
        sheet_name='kcal_per_100g')
    df_kcal_g = df_kcal_t[['Item crop yield', 'kcal per 100g']]
    # Merge
    merged_df = pd.merge(
        df_kcal_g,
        df_other_items.copy(),  # Only keep the needed columns
        left_on=['Item crop yield'],
        right_on=['Item']
    )
    # Operation
    merged_df['Yield'] = merged_df['Yield'] * merged_df['kcal per 100g'] / 0.1
    pivot_df_yield = merged_df[['Area', 'Year', 'Item', 'Yield']]
    pivot_df_yield = pivot_df_yield.copy()

    # Append with fibers crops (different unit as other yields)
    pivot_df_yield = pd.concat([pivot_df_yield, df_fibre.copy()], ignore_index=True)

    # Create a dummy for Rice as no products
    # Create a DataFrame for the new "Rice" rows
    new_rows = pivot_df_yield[['Area', 'Year']].drop_duplicates().copy()
    #new_rows['Item'] = 'Rice and products' # If rice is missing in Switzerland
    #new_rows['Losses[%]'] = 0

    # Append the new rows to the original DataFrame
    #pivot_df_yield = pd.concat([pivot_df_yield, new_rows], ignore_index=True)

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------

    # Food item name matching with dictionary
    # Read excel file
    df_dict_csc = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='climate-smart-crops')

    # Merge based on 'Item'
    df_yield_pathwaycalc = pd.merge(df_dict_csc, pivot_df_yield, on='Item')

    # Drop the 'Item' column
    df_yield_pathwaycalc = df_yield_pathwaycalc.drop(columns=['Item'])

    # Renaming existing columns (geoscale, timsecale, value)
    df_yield_pathwaycalc.rename(columns={'Area': 'geoscale', 'Year': 'timescale', 'Yield': 'value'}, inplace=True)

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_yield_pathwaycalc['module'] = 'agriculture'
    df_yield_pathwaycalc['lever'] = 'climate-smart-crop'
    df_yield_pathwaycalc['level'] = 0
    cols = df_yield_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    df_yield_pathwaycalc = df_yield_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    df_yield_pathwaycalc['geoscale'] = df_yield_pathwaycalc['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    df_yield_pathwaycalc['geoscale'] = df_yield_pathwaycalc['geoscale'].replace('Netherlands (Kingdom of the)',
                                                                                'Netherlands')
    df_yield_pathwaycalc['geoscale'] = df_yield_pathwaycalc['geoscale'].replace('Czechia', 'Czech Republic')

    # ------------------------------------------------------------------------------------------------------------------
    # YIELD ALGAE & INSECT ---------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # Use (agroforestry_crop) as a structural basis
    yield_aps = agroforestry_crop[['timescale', 'geoscale']].copy()

    # Add the variables with values based on EuCalc for those constant
    yield_aps['agr_climate-smart-crop_yield_algae[kcal/ha]'] = 119866666.666667
    yield_aps['agr_climate-smart-crop_yield_insect[kcal/ha]'] = 675000000.0
    yield_aps['agr_climate-smart-crop_yield_lgn-energycrop[kcal/ha]'] = 77387400.0

    # Melt the df
    yield_aps_pathwaycalc = pd.melt(yield_aps, id_vars=['timescale', 'geoscale'],
                                           value_vars=['agr_climate-smart-crop_yield_algae[kcal/ha]',
                                                       'agr_climate-smart-crop_yield_insect[kcal/ha]',
                                                       'agr_climate-smart-crop_yield_lgn-energycrop[kcal/ha]'],
                                           var_name='variables', value_name='value')


    # For other value : gas-energycrop
    # Load from previous EuCalc Data
    df_yield_data = pd.read_csv(
        'data/agriculture_climate-smart-crop_eucalc.csv',
        sep=';')

    # Filter columns
    df_filtered_columns = df_yield_data[['geoscale', 'timescale', 'eucalc-name', 'value']]

    # rename col 'eucalc-name' in 'variables'
    df_filtered_columns = df_filtered_columns.rename(columns={'eucalc-name': 'variables'})

    # Filter rows that contains biomass-mix
    df_filtered_rows = df_filtered_columns[
        df_filtered_columns['variables'].str.contains('ots_agr_climate-smart-crop_yield_gas-energycrop', case=False, na=False)
    ]

    # Rename from ots_agr to agr
    df_filtered_rows = df_filtered_rows.copy()
    df_filtered_rows['variables'] = df_filtered_rows['variables'].str.replace('ots_agr', 'agr', regex=False)


    # Concat
    yield_aps_pathwaycalc = pd.concat([yield_aps_pathwaycalc, df_filtered_rows])

    # PathwayCalc formatting --------------------------------------------------------------------------------------------
    yield_aps_pathwaycalc['module'] = 'agriculture'
    yield_aps_pathwaycalc['lever'] = 'climate-smart-crop'
    yield_aps_pathwaycalc['level'] = 0
    cols = yield_aps_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    cols.insert(cols.index('timescale'), cols.pop(cols.index('variables')))
    yield_aps_pathwaycalc = yield_aps_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    yield_aps_pathwaycalc['geoscale'] = yield_aps_pathwaycalc['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    yield_aps_pathwaycalc['geoscale'] = yield_aps_pathwaycalc['geoscale'].replace(
        'Netherlands (Kingdom of the)',
        'Netherlands')
    yield_aps_pathwaycalc['geoscale'] = yield_aps_pathwaycalc['geoscale'].replace('Czechia',
                                                                                                'Czech Republic')

    # FINAL RESULT ---------------------------------------------------------------------------------------------------------
    df_climate_smart_crop = pd.concat([df_input_pathwaycalc, df_losses_pathwaycalc])
    df_climate_smart_crop = pd.concat([df_climate_smart_crop, df_yield_pathwaycalc])
    df_climate_smart_crop = pd.concat([df_climate_smart_crop, agroforestry_crop_pathwaycalc])
    df_climate_smart_crop = pd.concat([df_climate_smart_crop, yield_aps_pathwaycalc])
    df_climate_smart_crop = pd.concat([df_climate_smart_crop, df_energy_pathwaycalc])
    df_climate_smart_crop = df_climate_smart_crop.drop_duplicates()

    # Rename countries to Pathaywcalc name
    df_climate_smart_crop['geoscale'] = df_climate_smart_crop['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    df_climate_smart_crop['geoscale'] = df_climate_smart_crop['geoscale'].replace(
       'Netherlands (Kingdom of the)', 'Netherlands')
    df_climate_smart_crop['geoscale'] = df_climate_smart_crop['geoscale'].replace('Czechia', 'Czech Republic')

    # Extrapolating
    df_climate_smart_crop= ensure_structure(df_climate_smart_crop)
    df_climate_smart_crop = df_climate_smart_crop.drop_duplicates()
    df_climate_smart_crop_pathwaycalc = linear_fitting_ots_db(df_climate_smart_crop, years_ots, countries='all')

    return df_climate_smart_crop_pathwaycalc, df_energy_demand_cal, df_CO2_cal

# CalculationLeaf CLIMATE SMART LIVESTOCK ------------------------------------------------------------------------------
def climate_smart_livestock_processing(df_feed_ration, df_liv_pop, df_cropland_density, list_countries):

    # ----------------------------------------------------------------------------------------------------------------------
    # LIVESTOCK DENSITY & GRAZING INTENSITY ---------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # Filter grazing ruminant livestock (cattle meat, sheep, goats) and sum per year
    df_ruminant = df_liv_pop[df_liv_pop['Item'].isin(
      ['Cattle, dairy','Cattle, non-dairy', 'Sheep and Goats'])]
    df_ruminant = df_ruminant.groupby(['Area', 'Year'], as_index=False)['Value'].sum()

    # Merge with cropland_density
    df_ruminant = pd.merge(df_ruminant, df_cropland_density, on=['Area', 'Year'])

    # Compute livestock density of ruminant per area of permanent meadows and pastures
    df_ruminant['Livestock density [lsu/ha]'] = df_ruminant['Value']/df_ruminant['Permanent meadows and pastures']

    # Filter and add column density
    df_ruminant = df_ruminant[['Year', 'Area', 'Livestock density [lsu/ha]']]

    # Adding an Item column for name
    df_ruminant['Item'] = 'Density'


    '''list_elements = ['Livestock units per agricultural land area', 'Share in total livestock']

    list_items = ['Major livestock types > (List)']

    # 1990 - 2021
    code = 'EK'
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                  '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                  '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items,
        'year': my_years
    }
    df_density_1990_2021 = faostat.get_data_df(code, pars=my_pars, strval=False)

    # Renaming item as the same animal (for meat and live/producing/slaugthered animals)
    # Commenting only to consider grazing animals (cattle, buffalo, sheep, goat, horse)
    # df_density_1990_2021.loc[df_density_1990_2021['Item'].str.contains('Pig', case=False, na=False), 'Item'] = 'Pig'
    df_density_1990_2021.loc[
        df_density_1990_2021['Item'].str.contains('Cattle', case=False, na=False), 'Item'] = 'Cattle'
    df_density_1990_2021.loc[
        df_density_1990_2021['Item'].str.contains('Buffalo', case=False, na=False), 'Item'] = 'Cattle'
    # df_density_1990_2021.loc[df_density_1990_2021['Item'].str.contains('Camel', case=False, na=False), 'Item'] = 'Other non-specified'
    # df_density_1990_2021.loc[df_density_1990_2021['Item'].str.contains('Rodent', case=False, na=False), 'Item'] = 'Other non-specified'
    # df_density_1990_2021.loc[df_density_1990_2021['Item'].str.contains('Chicken', case=False, na=False), 'Item'] = 'Chicken'
    # df_density_1990_2021.loc[df_density_1990_2021['Item'].str.contains('Duck', case=False, na=False), 'Item'] = 'Duck'
    # df_density_1990_2021.loc[df_density_1990_2021['Item'].str.contains('Geese', case=False, na=False), 'Item'] = 'Goose'
    # df_density_1990_2021.loc[df_density_1990_2021['Item'].str.contains('Pigeon', case=False, na=False), 'Item'] = 'Pigeon'
    df_density_1990_2021.loc[
        df_density_1990_2021['Item'].str.contains('Horses', case=False, na=False), 'Item'] = 'Horse'
    df_density_1990_2021.loc[df_density_1990_2021['Item'].str.contains('Sheep', case=False, na=False), 'Item'] = 'Sheep'
    df_density_1990_2021.loc[df_density_1990_2021['Item'].str.contains('Goat', case=False, na=False), 'Item'] = 'Goat'
    # df_density_1990_2021.loc[df_density_1990_2021['Item'].str.contains('Rabbits and hares', case=False, na=False), 'Item'] = 'Rabbit'

    # Filter only for Cattle, sheep and goats
    df_density_1990_2021 = df_density_1990_2021[df_density_1990_2021['Item'].isin(
      ['Cattle', 'Sheep', 'Goat'])]

    # Aggregating
    # Reading excel lsu equivalent (for aggregation)
    df_lsu = pd.read_excel(
        'dictionaries/lsu_equivalent.xlsx',
        sheet_name='lsu_equivalent')
    # Merging
    df_density_1990_2021 = pd.merge(df_density_1990_2021, df_lsu, on='Item')

    # Aggregating
    df_density_1990_2021 = \
    df_density_1990_2021.groupby(['Aggregation', 'Area', 'Year', 'Element', 'Unit'], as_index=False)['Value'].sum()

    # Pivot the df
    pivot_df = df_density_1990_2021.pivot_table(index=['Area', 'Year', 'Aggregation'], columns='Element',
                                                values='Value').reset_index()

    # Normalize the share of ruminants
    pivot_df['Total ruminant share [%]'] = pivot_df.groupby(['Area', 'Year'])['Share in total livestock'].transform(
        'sum')
    pivot_df['Normalized ruminant share [%]'] = pivot_df['Share in total livestock'] / pivot_df[
        'Total ruminant share [%]']

    # Multiply Livestock per ha per type [lsu/ha] with the normalized ratio
    pivot_df['Livestock area per type per share [lsu/ha]'] = pivot_df['Livestock units per agricultural land area'] * \
                                                             pivot_df['Normalized ruminant share [%]']

    # Sum
    # Livestock density [lsu/ha] = sum per year & country (Livestock area per type per share [lsu/ha])
    pivot_df['Livestock density [lsu/ha]'] = pivot_df.groupby(['Area', 'Year'])[
        'Livestock area per type per share [lsu/ha]'].transform('sum')

    # Grouping for one value per country & year
    grouped_df = pivot_df.groupby(['Year', 'Area', 'Livestock density [lsu/ha]']).size().reset_index(name='Count')
    # Drop other columns by selecting only the desired columns
    grouped_df = grouped_df[['Year', 'Area', 'Livestock density [lsu/ha]']]

    # Merge with df_cropland_density
    grouped_df = pd.merge(df_cropland_density, grouped_df, on=['Area', 'Year'])

    # Density per grassland instead of density per agricultural land
    # Calculate total livestock
    grouped_df['total_livestock'] = grouped_df['Livestock density [lsu/ha]'] * (grouped_df['Cropland'] + grouped_df['Permanent meadows and pastures'])
    # Calculate livestock density per grassland
    grouped_df['Livestock density [lsu/ha]'] = grouped_df['total_livestock'] / grouped_df['Permanent meadows and pastures']
    grouped_df = grouped_df[['Year', 'Area', 'Livestock density [lsu/ha]']]

    # Adding an Item column for name
    grouped_df['Item'] = 'Density' '''

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------

    # Renaming into 'Value'
    df_ruminant.rename(columns={'Area': 'geoscale', 'Year': 'timescale', 'Livestock density [lsu/ha]': 'value'},
                      inplace=True)

    # Read excel file
    df_dict_csl = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='climate-smart-livestock')

    # Merge based on 'Item'
    df_csl_density_pathwaycalc = pd.merge(df_dict_csl, df_ruminant, on='Item')

    # Drop the 'Item' column
    df_csl_density_pathwaycalc = df_csl_density_pathwaycalc.drop(columns=['Item'])

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_csl_density_pathwaycalc['module'] = 'agriculture'
    df_csl_density_pathwaycalc['lever'] = 'climate-smart-livestock'
    df_csl_density_pathwaycalc['level'] = 0
    cols = df_csl_density_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    df_csl_density_pathwaycalc = df_csl_density_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    df_csl_density_pathwaycalc['geoscale'] = df_csl_density_pathwaycalc['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    df_csl_density_pathwaycalc['geoscale'] = df_csl_density_pathwaycalc['geoscale'].replace(
        'Netherlands (Kingdom of the)',
        'Netherlands')
    df_csl_density_pathwaycalc['geoscale'] = df_csl_density_pathwaycalc['geoscale'].replace('Czechia', 'Czech Republic')

    # ----------------------------------------------------------------------------------------------------------------------
    # AGROFORESTRY (GRASSLAND & HEDGES) ------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    # Is equal to 0 for all ots for all countries

    # Use density (grouped_df) as a structural basis
    agroforestry_liv = df_ruminant.copy()

    # Drop the column Item
    agroforestry_liv = agroforestry_liv.drop(columns=['Item', 'value'])

    # Rename the column in geoscale and timescale
    agroforestry_liv.rename(columns={'Area': 'geoscale', 'Year': 'timescale'}, inplace=True)

    # Changing data type to numeric (except for the geoscale column)
    agroforestry_liv.loc[:, agroforestry_liv.columns != 'geoscale'] = agroforestry_liv.loc[:,
                                                                        agroforestry_liv.columns != 'geoscale'].apply(
        pd.to_numeric, errors='coerce')

    # Add rows to have 1990-2022
    # Generate a DataFrame with all combinations of geoscale and timescale
    geoscale_values = agroforestry_liv['geoscale'].unique()
    timescale_values = pd.Series(range(1990, 2023))

    # Create a DataFrame for the cartesian product
    cartesian_product = pd.MultiIndex.from_product([geoscale_values, timescale_values],
                                                   names=['geoscale', 'timescale']).to_frame(index=False)

    # Merge the original DataFrame with the cartesian product to include all combinations
    agroforestry_liv = pd.merge(cartesian_product, agroforestry_liv, on=['geoscale', 'timescale'], how='left')

    # Add the variables with a value of 0
    agroforestry_liv['agr_climate-smart-livestock_ef_agroforestry_grassland[tC/ha]'] = 0
    agroforestry_liv['agr_climate-smart-livestock_ef_agroforestry_hedges[tC/ha]'] = 0

    # Melt the df
    agroforestry_liv_pathwaycalc = pd.melt(agroforestry_liv, id_vars=['timescale', 'geoscale'],
                    value_vars=['agr_climate-smart-livestock_ef_agroforestry_grassland[tC/ha]', 'agr_climate-smart-livestock_ef_agroforestry_hedges[tC/ha]'],
                    var_name='variables', value_name='value')

    # PathwayCalc formatting --------------------------------------------------------------------------------------------
    agroforestry_liv_pathwaycalc['module'] = 'agriculture'
    agroforestry_liv_pathwaycalc['lever'] = 'climate-smart-livestock'
    agroforestry_liv_pathwaycalc['level'] = 0
    cols = agroforestry_liv_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    cols.insert(cols.index('timescale'), cols.pop(cols.index('variables')))
    agroforestry_liv_pathwaycalc = agroforestry_liv_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    agroforestry_liv_pathwaycalc['geoscale'] = agroforestry_liv_pathwaycalc['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    agroforestry_liv_pathwaycalc['geoscale'] = agroforestry_liv_pathwaycalc['geoscale'].replace('Netherlands (Kingdom of the)',
                                                                                  'Netherlands')
    agroforestry_liv_pathwaycalc['geoscale'] = agroforestry_liv_pathwaycalc['geoscale'].replace('Czechia', 'Czech Republic')

    # ----------------------------------------------------------------------------------------------------------------------
    # ENTERIC EMISSIONS ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    list_elements = ['Enteric fermentation (Emissions CH4)', 'Manure management (Emissions CH4)', 'Stocks']

    list_items = ['All Animals > (List)']
    list_sources = ['FAO TIER 1']

    # 1990 - 2021
    code = 'GLE'
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    my_sources = [faostat.get_par(code, 'sources')[i] for i in list_sources]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                  '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                  '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items,
        'year': my_years,
        'source': my_sources
    }
    df_enteric_1990_2021 = faostat.get_data_df(code, pars=my_pars, strval=False)

    # Renaming item as the same animal (for meat and live/producing/slaugthered animals)
    df_enteric_1990_2021.loc[
        df_enteric_1990_2021['Item'].str.contains('Cattle, dairy', case=False, na=False), 'Item'] = 'Dairy cows'
    df_enteric_1990_2021.loc[
        df_enteric_1990_2021['Item'].str.contains('Cattle, non-dairy', case=False, na=False), 'Item'] = 'Cattle'
    df_enteric_1990_2021.loc[df_enteric_1990_2021['Item'].str.contains('Goat', case=False, na=False), 'Item'] = 'Goat'
    df_enteric_1990_2021.loc[
        df_enteric_1990_2021['Item'].str.contains('Chickens, broilers', case=False, na=False), 'Item'] = 'Chicken'
    df_enteric_1990_2021.loc[df_enteric_1990_2021['Item'].str.contains('Chickens, layers', case=False,
                                                                       na=False), 'Item'] = 'Chicken laying hens'
    df_enteric_1990_2021.loc[df_enteric_1990_2021['Item'].str.contains('Duck', case=False, na=False), 'Item'] = 'Duck'
    df_enteric_1990_2021.loc[df_enteric_1990_2021['Item'].str.contains('Horse', case=False, na=False), 'Item'] = 'Horse'
    df_enteric_1990_2021.loc[df_enteric_1990_2021['Item'].str.contains('Sheep', case=False, na=False), 'Item'] = 'Sheep'
    df_enteric_1990_2021.loc[df_enteric_1990_2021['Item'].str.contains('Swine', case=False, na=False), 'Item'] = 'Pig'
    df_enteric_1990_2021.loc[
        df_enteric_1990_2021['Item'].str.contains('Turkey', case=False, na=False), 'Item'] = 'Turkey'
    df_enteric_1990_2021.loc[df_enteric_1990_2021['Item'].str.contains('Asse', case=False, na=False), 'Item'] = 'Asse'
    df_enteric_1990_2021.loc[
        df_enteric_1990_2021['Item'].str.contains('Buffalo', case=False, na=False), 'Item'] = 'Buffalo'
    df_enteric_1990_2021.loc[df_enteric_1990_2021['Item'].str.contains('Mule', case=False, na=False), 'Item'] = 'Mule'
    df_enteric_1990_2021.loc[
        df_enteric_1990_2021['Item'].str.contains('Camel', case=False, na=False), 'Item'] = 'Other non-specified'

    # Reading excel lsu equivalent
    df_lsu = pd.read_excel(
        'dictionaries/lsu_equivalent.xlsx',
        sheet_name='lsu_equivalent')
    # Merging
    df_enteric_1990_2021 = pd.merge(df_enteric_1990_2021, df_lsu, on='Item')

    # Converting Animals to lsu
    condition = df_enteric_1990_2021['Unit'] == 'An'
    df_enteric_1990_2021.loc[condition, 'Value'] *= df_enteric_1990_2021['lsu']

    # Aggregating
    df_enteric_1990_2021_grouped = \
    df_enteric_1990_2021.groupby(['Aggregation', 'Area', 'Year', 'Element', 'Unit'], as_index=False)['Value'].sum()

    # Pivot the df
    pivot_df = df_enteric_1990_2021_grouped.pivot_table(index=['Area', 'Year', 'Aggregation'], columns='Element',
                                                        values='Value').reset_index()

    # Enteric emissions CH4 [t/lsu] = 1000 * 'Enteric fermentation (Emissions CH4) [kt]'/ 'Stocks [lsu]'
    pivot_df['Enteric emissions CH4 [t/lsu]'] = 1000 * pivot_df['Enteric fermentation (Emissions CH4)'] / pivot_df[
        'Stocks']

    # Create duplicate for fxa
    df_manure_ch4_fxa = pivot_df.copy()
    df_manure_ch4_fxa['Manure emissions CH4 [t/lsu]'] = 1000 * df_manure_ch4_fxa[
      'Manure management (Emissions CH4)'] / df_manure_ch4_fxa[
                                                  'Stocks']
    df_manure_ch4_fxa = df_manure_ch4_fxa[['Area', 'Year', 'Aggregation', 'Manure emissions CH4 [t/lsu]']].copy()

    # Drop the columns 'Enteric fermentation (Emissions CH4)' 'Stocks'
    pivot_df = pivot_df.drop(columns=['Enteric fermentation (Emissions CH4)','Manure management (Emissions CH4)', 'Stocks'])

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------

    # Renaming into 'Value'
    pivot_df.rename(columns={'Area': 'geoscale', 'Year': 'timescale', 'Enteric emissions CH4 [t/lsu]': 'value'},
                    inplace=True)

    # Food item name matching with dictionary
    # Read excel file
    df_dict_csl_enteric = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='climate-smart-livestock_enteric')

    # Merge based on 'Item' & 'Aggregation'
    df_enteric_pathwaycalc = pd.merge(df_dict_csl_enteric, pivot_df, left_on='Item', right_on='Aggregation')

    # Drop the 'Item' column
    df_enteric_pathwaycalc = df_enteric_pathwaycalc.drop(columns=['Item', 'Aggregation'])

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_enteric_pathwaycalc['module'] = 'agriculture'
    df_enteric_pathwaycalc['lever'] = 'climate-smart-livestock'
    df_enteric_pathwaycalc['level'] = 0
    cols = df_enteric_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    df_enteric_pathwaycalc = df_enteric_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    df_enteric_pathwaycalc['geoscale'] = df_enteric_pathwaycalc['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    df_enteric_pathwaycalc['geoscale'] = df_enteric_pathwaycalc['geoscale'].replace('Netherlands (Kingdom of the)',
                                                                                    'Netherlands')
    df_enteric_pathwaycalc['geoscale'] = df_enteric_pathwaycalc['geoscale'].replace('Czechia', 'Czech Republic')

    # ----------------------------------------------------------------------------------------------------------------------
    # MANURE EMISSIONS (APPLIED, PASTURE & TREATED) ------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    list_elements = ['Amount excreted in manure (N content)', 'Manure left on pasture (N content)',
                     'Manure applied to soils (N content)', 'Losses from manure treated (N content)']

    list_items = ['All Animals > (List)']

    # 1990 - 2022
    code = 'EMN'
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                  '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                  '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items,
        'year': my_years
    }
    df_manure_1990_2021 = faostat.get_data_df(code, pars=my_pars, strval=False)

    # Renaming item as the same animal
    df_manure_1990_2021.loc[
        df_manure_1990_2021['Item'].str.contains('Cattle, dairy', case=False, na=False), 'Item'] = 'Dairy cows'
    df_manure_1990_2021.loc[
        df_manure_1990_2021['Item'].str.contains('Cattle, non-dairy', case=False, na=False), 'Item'] = 'Cattle'
    df_manure_1990_2021.loc[df_manure_1990_2021['Item'].str.contains('Goat', case=False, na=False), 'Item'] = 'Goat'
    df_manure_1990_2021.loc[
        df_manure_1990_2021['Item'].str.contains('Chickens, broilers', case=False, na=False), 'Item'] = 'Chicken'
    df_manure_1990_2021.loc[df_manure_1990_2021['Item'].str.contains('Chickens, layers', case=False,
                                                                     na=False), 'Item'] = 'Chicken laying hens'
    df_manure_1990_2021.loc[df_manure_1990_2021['Item'].str.contains('Duck', case=False, na=False), 'Item'] = 'Duck'
    df_manure_1990_2021.loc[df_manure_1990_2021['Item'].str.contains('Horse', case=False, na=False), 'Item'] = 'Horse'
    df_manure_1990_2021.loc[df_manure_1990_2021['Item'].str.contains('Sheep', case=False, na=False), 'Item'] = 'Sheep'
    df_manure_1990_2021.loc[df_manure_1990_2021['Item'].str.contains('Swine', case=False, na=False), 'Item'] = 'Pig'
    df_manure_1990_2021.loc[df_manure_1990_2021['Item'].str.contains('Turkey', case=False, na=False), 'Item'] = 'Turkey'
    df_manure_1990_2021.loc[df_manure_1990_2021['Item'].str.contains('Asse', case=False, na=False), 'Item'] = 'Asse'
    df_manure_1990_2021.loc[
        df_manure_1990_2021['Item'].str.contains('Buffalo', case=False, na=False), 'Item'] = 'Buffalo'
    df_manure_1990_2021.loc[df_manure_1990_2021['Item'].str.contains('Mule', case=False, na=False), 'Item'] = 'Mule'
    df_manure_1990_2021.loc[
        df_manure_1990_2021['Item'].str.contains('Camel', case=False, na=False), 'Item'] = 'Other non-specified'

    # Reading excel lsu equivalent (for aggregation)
    df_lsu = pd.read_excel(
        'dictionaries/lsu_equivalent.xlsx',
        sheet_name='lsu_equivalent')
    # Merging
    df_manure_1990_2021 = pd.merge(df_manure_1990_2021, df_lsu, on='Item')

    # Aggregating
    df_manure_1990_2021 = \
    df_manure_1990_2021.groupby(['Aggregation', 'Area', 'Year', 'Element', 'Unit'], as_index=False)['Value'].sum()

    # Pivot the df
    pivot_df = df_manure_1990_2021.pivot_table(index=['Area', 'Year', 'Aggregation'], columns='Element',
                                               values='Value').reset_index()

    # Create copy for manure_fxa()
    df_manure_n_fxa = pivot_df.copy()

    # Merge with df_liv_pop
    # Rename for merge (df_liv_pop => pivot_df_slau (meat) or df_slau_eggs_milk (eggs,dairy))
    terms = {
        'Cattle, dairy': 'Dairy-milk',
        'Cattle, non-dairy': 'Bovine',
        'Chickens, layers': 'Hens-egg',
        'Sheep and Goats': 'Sheep',
        'Swine': 'Pig',
        'Others Stocks': 'Other animal',
        'Poultry Stocks': 'Poultry'
    }

    # Apply the replacement
    df_liv_pop['Item'] = df_liv_pop['Item'].replace(terms)

    # Merge with stock from df_liv_pop
    pivot_df = pd.merge(pivot_df, df_liv_pop,
                         left_on=['Area', 'Year','Aggregation'],
                         right_on=['Area', 'Year','Item'],
                         how='inner')

    # Manure applied/treated/pasture [%] = Manure applied to soil/treated/left on pasture (N content) [kg] / Amount excreted (N content) [kg]

    pivot_df['Manure applied [%]'] = pivot_df['Manure applied to soils (N content)'] / pivot_df[
        'Amount excreted in manure (N content)']
    pivot_df['Manure treated [%]'] = pivot_df['Losses from manure treated (N content)'] / pivot_df[
        'Amount excreted in manure (N content)']
    pivot_df['Manure pasture [%]'] = pivot_df['Manure left on pasture (N content)'] / pivot_df[
        'Amount excreted in manure (N content)']

    # Compute manure yield fxa
    pivot_df['Manure yield [tN/lsu]'] = 10**-3 * pivot_df['Amount excreted in manure (N content)'] / pivot_df['Value']

    # Create copy for emission factor per practice

    # Drop the columns
    pivot_df = pivot_df.drop(columns=['Manure applied to soils (N content)', 'Losses from manure treated (N content)',
                                      'Manure left on pasture (N content)', 'Amount excreted in manure (N content)', 'Value', 'Item'])

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------

    # Melt the DataFrame
    df_melted = pd.melt(pivot_df, id_vars=['Area', 'Year', 'Aggregation'],
                        value_vars=['Manure applied [%]', 'Manure treated [%]', 'Manure pasture [%]', 'Manure yield [tN/lsu]'],
                        var_name='Item', value_name='value')

    # Concatenate the aggregation column with the manure column names
    df_melted['Item'] = df_melted['Aggregation'] + ' ' + df_melted['Item']

    # Drop the aggregation column as it's now part of the item column
    df_melted = df_melted.drop(columns=['Aggregation'])

    # Renaming
    df_melted.rename(columns={'Area': 'geoscale', 'Year': 'timescale'}, inplace=True)

    # Food item name matching with dictionary
    # Read excel file
    df_dict_csl = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='climate-smart-livestock')

    # Merge based on 'Item' & 'Aggregation'
    df_manure_pathwaycalc = pd.merge(df_dict_csl, df_melted, on='Item')

    # Drop the 'Item' column
    df_manure_pathwaycalc = df_manure_pathwaycalc.drop(columns=['Item'])

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_manure_pathwaycalc['module'] = 'agriculture'
    df_manure_pathwaycalc['lever'] = 'climate-smart-livestock'
    df_manure_pathwaycalc['level'] = 0
    cols = df_manure_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    df_manure_pathwaycalc = df_manure_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    df_manure_pathwaycalc['geoscale'] = df_manure_pathwaycalc['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    df_manure_pathwaycalc['geoscale'] = df_manure_pathwaycalc['geoscale'].replace('Netherlands (Kingdom of the)',
                                                                                  'Netherlands')
    df_manure_pathwaycalc['geoscale'] = df_manure_pathwaycalc['geoscale'].replace('Czechia', 'Czech Republic')

    # Filter for fxa
    df_csl_fxa = df_manure_pathwaycalc[
        df_manure_pathwaycalc['variables'].str.contains('fxa', case=False, na=False)]

   # Drop the rows from original df
    df_manure_pathwaycalc = df_manure_pathwaycalc[
      ~df_manure_pathwaycalc['variables'].str.contains('fxa', case=False,
                                                       na=False)]

    # ----------------------------------------------------------------------------------------------------------------------
    # LOSSES ---------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # FOOD BALANCE SHEETS (FBS) - For everything  -------------------------------------------------
    # List of elements
    list_elements = ['Losses', 'Production Quantity']

    list_items = ['Animal Products > (List)']

    # 1990 - 2013
    code = 'FBSH'
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                  '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items,
        'year': my_years
    }
    df_losses_csl_1990_2013 = faostat.get_data_df(code, pars=my_pars, strval=False)

    # Renaming Elements
    df_losses_csl_1990_2013.loc[df_losses_csl_1990_2013['Element'].str.contains('Production Quantity',
                                                                                case=False,
                                                                                na=False), 'Element'] = 'Production'

    # 2010 - 2022
    # Different list because different in item nomination such as rice
    list_elements = ['Losses', 'Production Quantity']
    code = 'FBS'
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items,
        'year': my_years
    }
    df_losses_csl_2010_2021 = faostat.get_data_df(code, pars=my_pars, strval=False)
    # Renaming Elements
    df_losses_csl_2010_2021.loc[df_losses_csl_2010_2021['Element'].str.contains('Production Quantity',
                                                                                case=False,
                                                                                na=False), 'Element'] = 'Production'

    # Concatenating
    df_losses_csl = pd.concat([df_losses_csl_1990_2013, df_losses_csl_2010_2021])

    # Compute losses ([%] of production) -----------------------------------------------------------------------------------
    # Losses [%] = 1 / (1 - Losses [1000t] / Production [1000t]) (pre processing for multiplicating the workflow)

    # Step 1: Pivot the DataFrame
    pivot_df = df_losses_csl.pivot_table(index=['Area', 'Year', 'Item'], columns='Element',
                                         values='Value').reset_index()

    # Replace NaN with 0
    pivot_df['Losses'].fillna(0.0, inplace=True)

    # Step 2: Compute the Losses [%] (really it's unit less)
    pivot_df['Losses[%]'] = 1 + (pivot_df['Losses'] / pivot_df['Production'])

    # Drop the columns Production, Import Quantity and Export Quantity
    pivot_df = pivot_df.drop(columns=['Production', 'Losses'])

    # Extrapolating for 2022 -----------------------------------------------------------------------------------------------

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------

    # Food item name matching with dictionary
    # Read excel file
    df_dict_csl_losses = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='climate-smart-livestock_losses')

    # Merge based on 'Item'
    df_losses_csl_pathwaycalc = pd.merge(df_dict_csl_losses, pivot_df, on='Item')

    # Drop the 'Item' column
    df_losses_csl_pathwaycalc = df_losses_csl_pathwaycalc.drop(columns=['Item'])

    # Renaming existing columns (geoscale, timsecale, value)
    df_losses_csl_pathwaycalc.rename(columns={'Area': 'geoscale', 'Year': 'timescale', 'Losses[%]': 'value'},
                                     inplace=True)

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_losses_csl_pathwaycalc['module'] = 'agriculture'
    df_losses_csl_pathwaycalc['lever'] = 'climate-smart-livestock'
    df_losses_csl_pathwaycalc['level'] = 0
    cols = df_losses_csl_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    df_losses_csl_pathwaycalc = df_losses_csl_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    df_losses_csl_pathwaycalc['geoscale'] = df_losses_csl_pathwaycalc['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    df_losses_csl_pathwaycalc['geoscale'] = df_losses_csl_pathwaycalc['geoscale'].replace(
        'Netherlands (Kingdom of the)',
        'Netherlands')
    df_losses_csl_pathwaycalc['geoscale'] = df_losses_csl_pathwaycalc['geoscale'].replace('Czechia', 'Czech Republic')

    # ----------------------------------------------------------------------------------------------------------------------
    # FEED RATION ----------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------

    # Fill nan with zeros
    df_feed_ration['Feed'].fillna(0, inplace=True)

    # Add a column with the total feed (per country and year)
    df_feed_ration['Total feed'] = df_feed_ration.groupby(['Area', 'Year'])['Feed'].transform('sum')

    # Feed ration [%] = Feed from item i / Total feed
    df_feed_ration['Feed ratio'] = df_feed_ration['Feed'] / df_feed_ration['Total feed']

    # Drop columns
    df_feed_ration = df_feed_ration.drop(columns=['Feed', 'Total feed'])

    # For Switzerland add Fruits = 0%
    # Duplicate rows only where Item = 'Pulses' and Area = 'Switzerland'
    duplicated_rows = df_feed_ration[
      (df_feed_ration['Item'] == 'Pulses') & (
          df_feed_ration['Area'] == 'Switzerland')
      ].copy()
    # Modify the duplicated rows
    duplicated_rows['Item'] = 'Fruits - Excluding Wine'
    duplicated_rows['Feed ratio'] = 0.0
    # Concatenate back to the main DataFrame
    df_feed_ration = pd.concat([df_feed_ration, duplicated_rows],
                               ignore_index=True)

    # For Switzerland add Vegetable oils = 0%
    # Duplicate rows only where Item = 'Pulses' and Area = 'Switzerland'
    duplicated_rows = df_feed_ration[
      (df_feed_ration['Item'] == 'Pulses') & (
          df_feed_ration['Area'] == 'Switzerland')
      ].copy()
    # Modify the duplicated rows
    duplicated_rows['Item'] = 'Vegetable Oils'
    duplicated_rows['Feed ratio'] = 0.0
    # Concatenate back to the main DataFrame
    df_feed_ration = pd.concat([df_feed_ration, duplicated_rows],
                               ignore_index=True)

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------

    # Renaming into 'Value'
    df_feed_ration.rename(columns={'Area': 'geoscale', 'Year': 'timescale', 'Feed ratio': 'value'}, inplace=True)

    # Read excel file
    df_dict_csl = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='climate-smart-livestock')

    # Merge based on 'Item'
    df_csl_feed_pathwaycalc = pd.merge(df_dict_csl, df_feed_ration, on='Item')

    # Drop the 'Item' column
    df_csl_feed_pathwaycalc = df_csl_feed_pathwaycalc.drop(columns=['Item'])

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_csl_feed_pathwaycalc['module'] = 'agriculture'
    df_csl_feed_pathwaycalc['lever'] = 'climate-smart-livestock'
    df_csl_feed_pathwaycalc['level'] = 0
    cols = df_csl_feed_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    df_csl_feed_pathwaycalc = df_csl_feed_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    df_csl_feed_pathwaycalc['geoscale'] = df_csl_feed_pathwaycalc['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    df_csl_feed_pathwaycalc['geoscale'] = df_csl_feed_pathwaycalc['geoscale'].replace('Netherlands (Kingdom of the)',
                                                                                      'Netherlands')
    df_csl_feed_pathwaycalc['geoscale'] = df_csl_feed_pathwaycalc['geoscale'].replace('Czechia', 'Czech Republic')

    # ----------------------------------------------------------------------------------------------------------------------
    # YIELD (DAIRY & EGGS) -------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    list_elements = ['Producing Animals/Slaughtered', 'Production Quantity']

    list_items = ['Milk, Total > (List)', 'Eggs Primary > (List)']

    # 1990 - 2022
    code = 'QCL'
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                  '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                  '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items,
        'year': my_years
    }
    df_producing_animals_1990_2022 = faostat.get_data_df(code, pars=my_pars, strval=False)

    # Keep the rows where Production is not in Nb of Eggs
    df_producing_animals_1990_2022 = df_producing_animals_1990_2022[df_producing_animals_1990_2022['Unit'] != '1000 No']

    # Renaming item as the same animal (for meat and live/producing/slaugthered animals)
    df_producing_animals_1990_2022.loc[
        df_producing_animals_1990_2022['Item'].str.contains('Cattle', case=False, na=False), 'Item'] = 'Dairy cows'
    df_producing_animals_1990_2022.loc[
        df_producing_animals_1990_2022['Item'].str.contains('Sheep', case=False, na=False), 'Item'] = 'Dairy sheep'
    df_producing_animals_1990_2022.loc[
        df_producing_animals_1990_2022['Item'].str.contains('Goat', case=False, na=False), 'Item'] = 'Dairy goat'
    df_producing_animals_1990_2022.loc[
        df_producing_animals_1990_2022['Item'].str.contains('Buffalo', case=False, na=False), 'Item'] = 'Dairy buffalo'
    df_producing_animals_1990_2022.loc[df_producing_animals_1990_2022['Item'].str.contains('Hen eggs', case=False,
                                                                                           na=False), 'Item'] = 'Chicken laying hens'
    df_producing_animals_1990_2022.loc[
        df_producing_animals_1990_2022['Item'].str.contains('Eggs from other birds', case=False,
                                                            na=False), 'Item'] = 'Other laying hens'

    # Unit conversion Poultry : [1000 An] => [An]
    df_producing_animals_1990_2022['Value'] = pd.to_numeric(df_producing_animals_1990_2022['Value'], errors='coerce')
    mask = df_producing_animals_1990_2022['Unit'].str.strip() == '1000 An'
    df_producing_animals_1990_2022.loc[mask, 'Value'] *= 1000
    df_producing_animals_1990_2022.loc[mask, 'Unit'] = 'An'
    df_producing_animals_1990_2022 = df_producing_animals_1990_2022.copy()

    # Reading excel lsu equivalent
    df_lsu = pd.read_excel(
        'dictionaries/lsu_equivalent.xlsx',
        sheet_name='lsu_equivalent')
    # Merging
    df_producing_animals_1990_2022 = pd.merge(df_producing_animals_1990_2022, df_lsu, on='Item')

    # Converting Animals to lsu
    condition = (df_producing_animals_1990_2022['Unit'] == 'An') | (df_producing_animals_1990_2022['Unit'] == '1000 An')
    df_producing_animals_1990_2022.loc[condition, 'Value'] *= df_producing_animals_1990_2022['lsu']

    # Aggregating
    grouped_df = \
    df_producing_animals_1990_2022.groupby(['Aggregation', 'Area', 'Year', 'Element', 'Unit'], as_index=False)[
        'Value'].sum()

    # Pivot the df
    pivot_df = grouped_df.pivot_table(index=['Area', 'Year', 'Aggregation'], columns='Element',
                                      values='Value').reset_index()

    # "Merging" the columns 'Laying' and 'Milk Animals' into 'Producing Animals'
    # Replace NaN with 0
    pivot_df['Laying'].fillna(0, inplace=True)
    pivot_df['Milk Animals'].fillna(0, inplace=True)

    # Sum the columns to create the 'Producing Animals' column
    pivot_df['Producing Animals'] = pivot_df['Laying'] + pivot_df['Milk Animals']

    # Yield [t/lsu] = Production quantity / Producing animals/Slaugthered NOW done after using cal values
    pivot_df['Yield [t/lsu]'] = pivot_df['Producing Animals']
    #pivot_df['Yield [t/lsu]'] = pivot_df['Production'] / pivot_df['Producing Animals']

    # Create a copy
    df_slau_eggs_milk = pivot_df.copy()
    df_slau_eggs_milk = df_slau_eggs_milk.drop(columns=['Laying', 'Milk Animals', 'Production', 'Yield [t/lsu]'])

    # Drop the columns to only have Yield and Slaughter rate
    pivot_df = pivot_df.drop(columns=['Laying', 'Milk Animals', 'Production', 'Producing Animals'])


    # ----------------------------------------------------------------------------------------------------------------------
    # YIELD (MEAT) --------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    list_elements = ['Producing Animals/Slaughtered', 'Production Quantity']

    list_items = ['Meat, Total > (List)']

    # 1990 - 2022 HERE
    code = 'QCL'
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                  '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                  '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items,
        'year': my_years
    }
    df_slaughtered_1990_2022 = faostat.get_data_df(code, pars=my_pars, strval=False)

    # Dropping 'Bees'
    df_slaughtered_1990_2022 = df_slaughtered_1990_2022[df_slaughtered_1990_2022['Item'] != 'Bees']

    # Renaming item as the same animal (for meat and live/producing/slaugthered animals)
    df_slaughtered_1990_2022.loc[
        df_slaughtered_1990_2022['Item'].str.contains('Pig', case=False, na=False), 'Item'] = 'Pig'
    df_slaughtered_1990_2022.loc[
        df_slaughtered_1990_2022['Item'].str.contains('Cattle', case=False, na=False), 'Item'] = 'Cattle'
    df_slaughtered_1990_2022.loc[
        df_slaughtered_1990_2022['Item'].str.contains('Buffalo', case=False, na=False), 'Item'] = 'Cattle'
    df_slaughtered_1990_2022.loc[
        df_slaughtered_1990_2022['Item'].str.contains('Chicken', case=False, na=False), 'Item'] = 'Chicken'
    df_slaughtered_1990_2022.loc[
        df_slaughtered_1990_2022['Item'].str.contains('Duck', case=False, na=False), 'Item'] = 'Duck'
    df_slaughtered_1990_2022.loc[
        df_slaughtered_1990_2022['Item'].str.contains('Turkeys', case=False, na=False), 'Item'] = 'Turkey'
    df_slaughtered_1990_2022.loc[
        df_slaughtered_1990_2022['Item'].str.contains('Geese', case=False, na=False), 'Item'] = 'Goose'
    df_slaughtered_1990_2022.loc[
        df_slaughtered_1990_2022['Item'].str.contains('Pigeon', case=False, na=False), 'Item'] = 'Pigeon'
    df_slaughtered_1990_2022.loc[
        df_slaughtered_1990_2022['Item'].str.contains('Horse', case=False, na=False), 'Item'] = 'Horse'
    df_slaughtered_1990_2022.loc[
        df_slaughtered_1990_2022['Item'].str.contains('Rabbits and hares', case=False, na=False), 'Item'] = 'Rabbit'
    df_slaughtered_1990_2022.loc[
        df_slaughtered_1990_2022['Item'].str.contains('Sheep', case=False, na=False), 'Item'] = 'Sheep'
    df_slaughtered_1990_2022.loc[
        df_slaughtered_1990_2022['Item'].str.contains('Goat', case=False, na=False), 'Item'] = 'Goat'
    df_slaughtered_1990_2022.loc[
        df_slaughtered_1990_2022['Item'].str.contains('Asse', case=False, na=False), 'Item'] = 'Asse'
    df_slaughtered_1990_2022.loc[
        df_slaughtered_1990_2022['Item'].str.contains('Camel', case=False, na=False), 'Item'] = 'Other non-specified'
    df_slaughtered_1990_2022.loc[
        df_slaughtered_1990_2022['Item'].str.contains('Rodent', case=False, na=False), 'Item'] = 'Other non-specified'
    df_slaughtered_1990_2022.loc[
        df_slaughtered_1990_2022['Item'].str.contains('Other', case=False, na=False), 'Item'] = 'Other non-specified'
    df_slaughtered_1990_2022.loc[
        df_slaughtered_1990_2022['Item'].str.contains('Game', case=False, na=False), 'Item'] = 'Game'
    df_slaughtered_1990_2022.loc[
        df_slaughtered_1990_2022['Item'].str.contains('Mule', case=False, na=False), 'Item'] = 'Mule'

    # HERE! Unit conversion Poultry : [1000 An] => [An]
    df_slaughtered_1990_2022['Value'] = pd.to_numeric(df_slaughtered_1990_2022['Value'], errors='coerce')
    mask = df_slaughtered_1990_2022['Unit'].str.strip() == '1000 An'
    df_slaughtered_1990_2022.loc[mask, 'Value'] *= 1000
    df_slaughtered_1990_2022.loc[mask, 'Unit'] = 'An'
    df_slaughtered_1990_2022 = df_slaughtered_1990_2022.copy()

    # Reading excel lsu equivalent
    df_lsu = pd.read_excel(
        'dictionaries/lsu_equivalent.xlsx',
        sheet_name='lsu_equivalent')
    # Merging
    df_slaughtered_1990_2022 = pd.merge(df_slaughtered_1990_2022, df_lsu, on='Item')

    # Converting Animals to lsu
    condition = (df_slaughtered_1990_2022['Unit'] == 'An') | (df_slaughtered_1990_2022['Unit'] == '1000 An')
    df_slaughtered_1990_2022.loc[condition, 'Value'] *= df_slaughtered_1990_2022['lsu']

    # Aggregating
    grouped_df = df_slaughtered_1990_2022.groupby(['Aggregation', 'Area', 'Year', 'Element', 'Unit'], as_index=False)[
        'Value'].sum()

    # Pivot the df
    pivot_df_slau = grouped_df.pivot_table(index=['Area', 'Year', 'Aggregation'], columns='Element',
                                           values='Value').reset_index()

    # Replace NaN with 0
    pivot_df_slau['Producing Animals/Slaughtered'].fillna(0, inplace=True)
    pivot_df_slau['Production'].fillna(0, inplace=True)

    # Create a copy for slau rate
    df_slau_meat = pivot_df_slau.copy()

    # Yield [t/lsu] = Production quantity / Producing animals/Slaugthered NOW DONE AFTER using cal values
    pivot_df_slau['Yield [t/lsu]'] = pivot_df_slau['Producing Animals/Slaughtered']
    #pivot_df_slau['Yield [t/lsu]'] = pivot_df_slau['Production'] / pivot_df_slau['Producing Animals/Slaughtered']

    # Drop the columns
    pivot_df_slau = pivot_df_slau.drop(columns=['Producing Animals/Slaughtered', 'Production'])

    # Replace NaN with 0
    pivot_df_slau['Yield [t/lsu]'].fillna(0, inplace=True)

    # ----------------------------------------------------------------------------------------------------------------------
    # SLAUGHTERED RATE (MEAT, EGGS & MILK) --------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # Concat df_slau_meat (meat) and df_slau_eggs_milk (eggs,dairy)
    df_slau_meat.rename(columns={'Producing Animals/Slaughtered': 'Producing Animals'}, inplace=True)
    df_slau_meat = df_slau_meat.drop(columns=['Production'])
    df_slau = pd.concat([df_slau_meat, df_slau_eggs_milk], ignore_index=True)

    # Rename for merge (df_liv_pop => pivot_df_slau (meat) or df_slau_eggs_milk (eggs,dairy))
    terms = {
        'Cattle, dairy': 'Dairy-milk',
        'Cattle, non-dairy': 'Bovine',
        'Chickens, layers': 'Hens-egg',
        'Sheep and Goats': 'Sheep',
        'Swine': 'Pig',
        'Others Stocks': 'Other animal',
        'Poultry Stocks': 'Poultry'
    }

    # Apply the replacement
    df_liv_pop['Item'] = df_liv_pop['Item'].replace(terms)

    # Merge with stock from df_liv_pop
    df_slau = pd.merge(df_slau, df_liv_pop,
                         left_on=['Area', 'Year','Aggregation'],
                         right_on=['Area', 'Year','Item'],
                         how='inner')

    # Slaughtered animals [%] = 'Producing Animals/Slaughtered' / 'Value' (value = stocks [lsu])
    df_slau['Slaughtered animals [%]'] = df_slau['Producing Animals']/df_slau['Value']
    df_slau['Slaughtered animals [%]'].fillna(0, inplace=True)

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------

    # Separating between slaugthered animals and yield (for meat)
    df_yield_meat = pivot_df_slau[['Area', 'Year', 'Aggregation', 'Yield [t/lsu]']]
    df_slau_meat = df_slau[['Area', 'Year', 'Aggregation', 'Slaughtered animals [%]']]

    # Creating copies
    df_yield_meat = df_yield_meat.copy()
    df_slau_meat = df_slau_meat.copy()

    # Renaming into 'Value'
    df_yield_meat.rename(columns={'Area': 'geoscale', 'Year': 'timescale', 'Yield [t/lsu]': 'value'}, inplace=True)
    pivot_df.rename(columns={'Area': 'geoscale', 'Year': 'timescale', 'Yield [t/lsu]': 'value'}, inplace=True)
    df_slau_meat.rename(columns={'Area': 'geoscale', 'Year': 'timescale', 'Slaughtered animals [%]': 'value'},
                        inplace=True)

    # Concatenating yield (meat, milk & eggs)
    df_yield_liv = pd.concat([df_yield_meat, pivot_df])

    # Read excel
    df_kcal_t = pd.read_excel(
        'dictionaries/kcal_to_t.xlsx',
        sheet_name='kcal_per_100g')
    df_kcal_g = df_kcal_t[['Item livestock yield', 'kcal per t']]
    # Merge
    merged_df = pd.merge(
        df_kcal_g,
        df_yield_liv,  # Only keep the needed columns
        left_on=['Item livestock yield'],
    right_on=['Aggregation']
    )
    # Operation Unit conversion t => kcal (not necessary since it's the producing animals now)
    #merged_df['value'] = merged_df['value'] * merged_df['kcal per t']
    df_yield_liv = merged_df[['geoscale', 'timescale', 'Aggregation', 'value']]
    df_yield_liv = df_yield_liv.copy()

    # Food item name matching with dictionary
    # Read excel file
    df_dict_csl_yield = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='climate-smart-livestock_yield')
    df_dict_csl_slau = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='climate-smart-livestock_slau')

    # Merge based on 'Item'
    df_yield_liv_pathwaycalc = pd.merge(df_dict_csl_yield, df_yield_liv, left_on='Item', right_on='Aggregation')
    df_slau_liv_pathwaycalc = pd.merge(df_dict_csl_slau, df_slau_meat, left_on='Item', right_on='Aggregation')

    # Drop the 'Item' column
    df_yield_liv_pathwaycalc = df_yield_liv_pathwaycalc.drop(columns=['Item', 'Aggregation'])
    df_slau_liv_pathwaycalc = df_slau_liv_pathwaycalc.drop(columns=['Item', 'Aggregation'])

    # Concatenating yield and slau
    df_yield_slau_liv_pathwaycalc = pd.concat([df_yield_liv_pathwaycalc, df_slau_liv_pathwaycalc])

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_yield_slau_liv_pathwaycalc['module'] = 'agriculture'
    df_yield_slau_liv_pathwaycalc['lever'] = 'climate-smart-livestock'
    df_yield_slau_liv_pathwaycalc['level'] = 0
    cols = df_yield_slau_liv_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    df_yield_pathwaycalc = df_yield_slau_liv_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    df_yield_slau_liv_pathwaycalc['geoscale'] = df_yield_slau_liv_pathwaycalc['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    df_yield_slau_liv_pathwaycalc['geoscale'] = df_yield_slau_liv_pathwaycalc['geoscale'].replace(
        'Netherlands (Kingdom of the)',
        'Netherlands')
    df_yield_slau_liv_pathwaycalc['geoscale'] = df_yield_slau_liv_pathwaycalc['geoscale'].replace('Czechia',
                                                                                                  'Czech Republic')

    # ----------------------------------------------------------------------------------------------------------------------
    # FINAL RESULTS --------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    df_csl = pd.concat([df_csl_density_pathwaycalc, df_enteric_pathwaycalc])
    df_csl = pd.concat([df_csl, df_manure_pathwaycalc])
    df_csl = pd.concat([df_csl, df_losses_csl_pathwaycalc])
    df_csl = pd.concat([df_csl, df_csl_feed_pathwaycalc])
    df_csl = pd.concat([df_csl, df_yield_slau_liv_pathwaycalc])
    df_csl = pd.concat([df_csl, agroforestry_liv_pathwaycalc])
    df_csl = df_csl.drop_duplicates()

    # Extrapolating
    df_climate_smart_livestock_pathwaycalc = ensure_structure(df_csl)
    df_climate_smart_livestock_pathwaycalc = linear_fitting_ots_db(df_climate_smart_livestock_pathwaycalc, years_ots, countries='all')


    return df_climate_smart_livestock_pathwaycalc, df_csl_fxa, df_manure_n_fxa, df_manure_ch4_fxa

# CalculationLeaf RUMINANT FEED ------------------------------------------------------------------------------

def ruminant_feed_processing(df_csl_feed):

  # Use df_csl_feed as a structural basis & Filter only one row
  df_ruminant_feed = df_csl_feed[df_csl_feed['Item']=='Beer'].copy()

  # Rename columns
  df_ruminant_feed.rename(columns={'Area': 'geoscale', 'Year': 'timescale',
                               'Feed': 'value'},
                      inplace=True)
  # Rename item
  df_ruminant_feed['Item'] = 'Share grass'

  # Set value
  df_ruminant_feed['value'] = 0.5

  # Pathwaycalc formatting
  # Food item name matching with dictionary
  # Read excel file
  df_dict_csl = pd.read_excel(
    'dictionaries/dictionnary_agriculture_landuse.xlsx',
    sheet_name='ruminant-feed')

  # Merge based on 'Item'
  df_ruminant_feed_pathwaycalc = pd.merge(df_dict_csl, df_ruminant_feed, on='Item')

  # Drop the 'Item' column
  df_ruminant_feed_pathwaycalc = df_ruminant_feed_pathwaycalc.drop(
    columns=['Item'])

  # Adding the columns module, lever, level and string-pivot at the correct places
  df_ruminant_feed_pathwaycalc['module'] = 'agriculture'
  df_ruminant_feed_pathwaycalc['lever'] = 'ruminant-feed'
  df_ruminant_feed_pathwaycalc['level'] = 0
  cols = df_ruminant_feed_pathwaycalc.columns.tolist()
  cols.insert(cols.index('value'), cols.pop(cols.index('module')))
  cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
  cols.insert(cols.index('value'), cols.pop(cols.index('level')))
  df_ruminant_feed_pathwaycalc = df_ruminant_feed_pathwaycalc[cols]

  # Rename countries to Pathaywcalc name
  df_ruminant_feed_pathwaycalc['geoscale'] = df_ruminant_feed_pathwaycalc[
    'geoscale'].replace(
    'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
  df_ruminant_feed_pathwaycalc['geoscale'] = df_ruminant_feed_pathwaycalc[
    'geoscale'].replace(
    'Netherlands (Kingdom of the)',
    'Netherlands')
  df_ruminant_feed_pathwaycalc['geoscale'] = df_ruminant_feed_pathwaycalc[
    'geoscale'].replace('Czechia',
                        'Czech Republic')

  # Extrapolating
  df_ruminant_feed_pathwaycalc = ensure_structure(df_ruminant_feed_pathwaycalc)
  df_ruminant_feed_pathwaycalc = linear_fitting_ots_db(df_ruminant_feed_pathwaycalc, years_ots, countries='all')

  return df_ruminant_feed_pathwaycalc

# CalculationLeaf FEED 2025 NEW VERSION ------------------------------------------------------------------------------
def feed_processing_lca():
  # Read excel sheets
  df_LCA_livestock = pd.read_excel('agriculture_feed_v2025.xlsx',
                                   sheet_name='data_LCA_livestock')
  df_LCA_feed = pd.read_excel('agriculture_feed_v2025.xlsx',
                                   sheet_name='data_LCA_feed')
  df_LCA_feed_yield = pd.read_excel('agriculture_feed_v2025.xlsx',
                                   sheet_name='data_LCA_feed_yield')

  # Divide all columns by the output to obtain values for 1 kg output
  # Identify the columns to divide (exclude Year, Area, Agricultural land)
  cols_to_divide_livestock = df_LCA_livestock.columns.difference(
    ['Item Livestock', 'Database', 'LCA item', 'Unit', 'Live weight per animal [kg]', 'LSU'])
  cols_to_divide_feed = df_LCA_feed.columns.difference(
    ['Item Feed', 'Database', 'LCA item', 'Unit'])
  cols_to_divide_feed_yield = df_LCA_feed_yield.columns.difference(
    ['Item Feed', 'Database', 'LCA item', 'Unit'])
  # Divide each of those columns by 'Agricultural land [ha]'
  df_LCA_livestock[cols_to_divide_livestock] = df_LCA_livestock[cols_to_divide_livestock].div(
    df_LCA_livestock['Output'],
    axis=0).copy()
  df_LCA_feed[cols_to_divide_feed] = df_LCA_feed[cols_to_divide_feed].div(
    df_LCA_feed['Output'],
    axis=0).copy()
  df_LCA_feed_yield[cols_to_divide_feed_yield] = df_LCA_feed_yield[cols_to_divide_feed_yield].div(
    df_LCA_feed_yield['Output'],
    axis=0).copy()

  # Fill Na with 0
  df_LCA_livestock.fillna(0.0, inplace=True)
  df_LCA_feed.fillna(0.0, inplace=True)
  df_LCA_feed_yield.fillna(0.0, inplace=True)

  # Melt dfs for feed and detailed feed
  df_long = df_LCA_livestock.melt(
    id_vars=['Item Livestock', 'Database', 'LCA item', 'Unit', 'Output', 'Live weight per animal [kg]', 'LSU'],
    # columns to keep
    var_name='Item Feed',  # new column for feed type names
    value_name='Feed'  # new column for feed values
  )
  df_long = df_long[['Item Livestock', 'Item Feed', 'Feed']].copy()
  df_feed_long = df_LCA_feed.melt(
    id_vars=['Item Feed', 'Database', 'LCA item', 'Unit', 'Output'],
    # columns to keep
    var_name='Feed item',  # new column for feed type names
    value_name='Input detailed'  # new column for feed values
  )
  df_feed_long = df_feed_long[['Item Feed', 'Feed item', 'Input detailed']].copy()

  # Separate between feedmix per animal
  df_long_feedmix = df_long[
        df_long['Item Feed'].str.contains('feed', case=False, na=False)
    ]
  df_long_nofeed = df_long[
    ~df_long['Item Feed'].str.contains('feed', case=False, na=False)
  ]

  # Feedmix : Merge
  df_merge = pd.merge(df_long_feedmix, df_feed_long, on='Item Feed', how='outer')
  df_merge.fillna(0.0, inplace=True)

  # Compute the feed inside the feedmix per animal
  df_merge['Feed'] = df_merge['Feed']* df_merge['Input detailed']

  # Concat between feed and feedmix
  df_merge = df_merge[['Item Livestock', 'Feed item', 'Feed']].copy()
  df_merge.rename(
    columns={'Feed item': 'Item Feed'}, inplace=True)
  df_feed = pd.concat([df_merge, df_long_nofeed])

  # Sum Feed per Item Livestock and Item Feed
  df_feed = df_feed.groupby(['Item Livestock', 'Item Feed'], as_index=False)[
    'Feed'].sum()

  # Merge with the processing yields
  df_LCA_feed_yield = df_LCA_feed_yield[['Item Feed', 'Output', 'Cereals', 'Oilcrops', 'Pulses', 'Sugarcrops']]
  df_feed = pd.merge(df_feed, df_LCA_feed_yield, on='Item Feed', how='inner')

  # Multiply with the processing yields
  cols_to_multiply = df_feed.columns.difference(
    ['Item Livestock','Item Feed', 'Output'])
  df_feed[cols_to_multiply] = df_feed[cols_to_multiply].mul(df_feed['Feed'],axis=0).copy()

  # Aggregated as overall feed category raw products (cereals, oilcrops, sugarcrops, pulses)
  feed_cols = ['Cereals', 'Pulses', 'Oilcrops',
               'Sugarcrops']  # adjust to your actual feed columns
  df_total_feed_per_livestock = df_feed.groupby('Item Livestock')[
    feed_cols].sum().reset_index()

  # Convert in feed per LSU
  df_lsu = df_LCA_livestock[['Item Livestock','Live weight per animal [kg]', 'LSU']]
  df_feed_lsu = pd.merge(df_lsu, df_total_feed_per_livestock, on='Item Livestock', how='inner')
  # Convert from feed per kg of live weight to feed per animal (multiplication)
  cols_to_multiply = df_feed_lsu.columns.difference(
    ['Item Livestock','Live weight per animal [kg]', 'LSU'])
  df_feed_lsu[cols_to_multiply] = df_feed_lsu[cols_to_multiply].mul(df_feed_lsu['Live weight per animal [kg]'],axis=0).copy()
  # Convert from feed per animal to feed per LSU (division)
  cols_to_multiply = df_feed_lsu.columns.difference(
    ['Item Livestock','Live weight per animal [kg]', 'LSU'])
  df_feed_lsu[cols_to_multiply] = df_feed_lsu[cols_to_multiply].div(df_feed_lsu['LSU'],axis=0).copy()

  # Format accordingly
  df_feed_lsu.rename(columns={'Cereals': 'crop-cereal',
                              'Pulses': 'crop-pulse',
                              'Oilcrops': 'crop-oilcrop',
                              'Sugarcrops': 'crop-sugarcrop',
                              'Item Livestock' : 'variables'}, inplace=True)
  df_feed_lsu = df_feed_lsu[['variables','crop-cereal','crop-pulse','crop-oilcrop','crop-sugarcrop']].copy()
  df_feed_lsu_melted = df_feed_lsu.melt(
    id_vars=['variables'],
    value_vars=['crop-cereal','crop-pulse','crop-oilcrop','crop-sugarcrop'],
    var_name='Item',
    value_name= 'value'
  )
  df_feed_lsu_melted['variables'] = 'fxa_agr_feed_' + df_feed_lsu_melted['variables'] + '_' + df_feed_lsu_melted['Item']+ '[kg/lsu]'
  df_feed_lsu_pathwaycalc = df_feed_lsu_melted[['variables','value']].copy()

  # Pathwaycalc formatting
  # Renaming existing columns (geoscale, timescale, value)
  df_feed_lsu_pathwaycalc.rename(columns={'Area': 'geoscale', 'Year': 'timescale'},
                             inplace=True)

  # Adding the columns module, lever, level and string-pivot at the correct places
  df_feed_lsu_pathwaycalc['geoscale'] = 'Switzerland'
  df_feed_lsu_pathwaycalc['timescale'] = '2020' #as an example but it is quite recent
  df_feed_lsu_pathwaycalc['module'] = 'agriculture'
  df_feed_lsu_pathwaycalc['lever'] = 'diet'
  df_feed_lsu_pathwaycalc['level'] = 0
  cols = df_feed_lsu_pathwaycalc.columns.tolist()
  cols.insert(cols.index('value'), cols.pop(cols.index('module')))
  cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
  cols.insert(cols.index('value'), cols.pop(cols.index('level')))
  df_feed_lsu_pathwaycalc = df_feed_lsu_pathwaycalc[cols]

  # Extrapolating
  df_feed_lsu_pathwaycalc = ensure_structure(df_feed_lsu_pathwaycalc)
  df_feed_lsu_pathwaycalc = linear_fitting_ots_db(df_feed_lsu_pathwaycalc, years_all,
                                              countries='all')

  return df_feed_lsu_pathwaycalc


# CalculationLeaf SELF SUFFICIENCY FEED ------------------------------------------------------------------------------

def feed_ssr_processing(years_ots):

  # Read excel (Link https://www.bfs.admin.ch/bfs/fr/home/statistiques/agriculture-sylviculture/agriculture.assetdetail.36135273.html)
  df_feed = pd.read_excel('data/OFS_bilan-fourrager.xlsx',
                                   sheet_name='T7.2.3.1.6')

  # List rows to filter
  rows_filter = "Année| Céréales| Tourteaux| Autres"
  filtered_df = df_feed[
    df_feed["Bilan fourrager"].str.contains(rows_filter,
                                            case=False, na=False)]

  # Format for computations
  # Set "Bilan fourrager" as index
  df_indexed = filtered_df.set_index("Bilan fourrager")
  # Transpose so that Unnamed columns become rows
  df_T = df_indexed.T
  # Promote "Année" row to the index
  df_T.index.name = None
  df_T = df_T.rename_axis("timescale").reset_index()
  # Replace "Year" values by the row under 'Année'
  df_T["timescale"] = df_T["Année"]
  # Drop the "Année" column since it's now the index
  df_T = df_T.drop(columns="Année")
  # Drop rows where 'Year' is NaN
  df_comp = df_T.dropna(subset=["timescale"])
  # Conver to numeric
  df_comp = df_comp.apply(pd.to_numeric, errors="coerce")

  # Compute SSR per category
  df_comp['SSR Cereals'] = df_comp['Production Céréales'] / (df_comp['Production Céréales'] + df_comp['Imports Céréales'])
  df_comp['SSR Cakes'] = df_comp['Production Tourteaux'] / (
      df_comp['Production Tourteaux'] + df_comp['Imports Tourteaux'])
  df_comp['SSR Other (veg)'] = df_comp['Production Autres (veg)'] / (
      df_comp['Production Autres (veg)'] + df_comp['Imports Autres (veg)'])
  df_comp['SSR Other (sugar, ibp)'] = df_comp['Production Autres (sucre, amidon, brasseries)'] / (
      df_comp['Production Autres (sucre, amidon, brasseries)'] + df_comp['Imports Autres (sucre, amidon, brasseries)'])

  # Filter SSR columns
  df_comp = df_comp[['timescale', 'SSR Cereals', 'SSR Cakes','SSR Other (veg)', 'SSR Other (sugar, ibp)']].copy()

  # Melt df
  df_melted = df_comp.melt(
    id_vars='timescale',
    var_name='Item',
    value_name='value'
  )

  # Pathwaycalc formatting
  # Read excel file
  df_dict_forestry = pd.read_excel(
    'dictionaries/dictionnary_agriculture_landuse.xlsx',
    sheet_name='self-sufficiency')

  # Merge based on 'Item'
  df_feed_ssr_pathwaycalc = pd.merge(df_dict_forestry, df_melted, on='Item')

  # Drop the 'Item' column
  df_feed_ssr_pathwaycalc = df_feed_ssr_pathwaycalc.drop(columns=['Item'])

  # Adding the columns module, lever, level and string-pivot at the correct places
  df_feed_ssr_pathwaycalc['geoscale'] = 'Switzerland'
  df_feed_ssr_pathwaycalc['module'] = 'agriculture'
  df_feed_ssr_pathwaycalc['lever'] = 'climate-smart-forestry'
  df_feed_ssr_pathwaycalc['level'] = 0
  cols = df_feed_ssr_pathwaycalc.columns.tolist()
  cols.insert(cols.index('value'), cols.pop(cols.index('module')))
  cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
  cols.insert(cols.index('value'), cols.pop(cols.index('level')))
  df_feed_ssr_pathwaycalc = df_feed_ssr_pathwaycalc[cols]

  # Extrapolating
  df_feed_ssr_pathwaycalc = ensure_structure(df_feed_ssr_pathwaycalc)
  df_feed_ssr_pathwaycalc = linear_fitting_ots_db(df_feed_ssr_pathwaycalc, years_ots,
                                              countries='Switzerland')

  return df_feed_ssr_pathwaycalc

# CalculationLeaf CLIMATE SMART FORESTRY -------------------------------------------------------------------------------
def climate_smart_forestry_processing():

    # ----------------------------------------------------------------------------------------------------------------------
    # INCREMENTAL GROWTH [m3/ha] -------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # Read csv
    df_g_inc = pd.read_excel(
        'data/data_forestry.xlsx',
        sheet_name='annual_ginc_per_area_m3ha')

    # Read and format forest area for later
    df_area = pd.read_csv(
        'data/fra-extentOfForest.csv')
    df_area.columns = df_area.iloc[0]
    df_area = df_area[1:]
    # Rename column name 'geoscale'
    df_area.rename(columns={df_area.columns[0]: 'geoscale'}, inplace=True)

    # Format correctly
    # Melting the dfs to have the relevant format (geoscale, year, value)
    df_g_inc = pd.melt(df_g_inc, id_vars=['geoscale'], var_name='timescale', value_name='value')
    df_area = pd.melt(df_area, id_vars=['geoscale'], var_name='timescale', value_name='forest area [ha]')
    # Changing data type to numeric (except for the geoscale column)
    df_g_inc.loc[:, df_g_inc.columns != 'geoscale'] = df_g_inc.loc[:, df_g_inc.columns != 'geoscale'].apply(
        pd.to_numeric, errors='coerce')
    # Merge the dfs (growing stock and area) to filter the relevant countries
    df_g_inc_area = pd.merge(df_g_inc, df_area, on=['geoscale', 'timescale'])
    # Only keep the columns geoscale, timescale and value
    df_g_inc_area = df_g_inc_area[['geoscale', 'timescale', 'value']]
    df_g_inc_area_pathwaycalc = df_g_inc_area.copy()

    # DEPRECIATED Compute the incremental difference -----------------------------------------------------------------------------------
    # Ensure the DataFrame is sorted by geoscale and timescale
    # df_g_inc.sort_values(by=['geoscale', 'timescale'], inplace=True)

    # Compute the incremental growing stock for each country : incremental growing stock [m3] = growing stock y(i) - growing stock y(i-1)
    # df_g_inc['incremental growing stock [m3/ha]'] = df_g_inc.groupby('geoscale')['growing stock [m3/ha]'].diff()

    # Calculate the number of years between each period
    # df_g_inc['years_diff'] = df_g_inc.groupby('geoscale')['timescale'].diff()

    # Calculate the annual increment by dividing the incremental growing stock by the number of years
    # df_g_inc['annual increment [m3/ha/yr]'] = df_g_inc['incremental growing stock [m3/ha]'] / df_g_inc['years_diff']

    # Drop the rows that are not countries (they both contain 2024)
    # df_g_inc_area = df_g_inc_area[~df_g_inc_area['geoscale'].str.contains('2024', na=False)]

    # Incremental growing stock [m3/ha] = Incremental growing stock [m3] / forest area [ha]
    # df_g_inc_area['Incremental growing stock [m3/ha]'] = df_g_inc_area['incremental growing stock [m3]'] / df_g_inc_area['forest area [ha]']

    # Incremental growing stock [m3/ha] = Incremental growing stock [m3] / forest area [ha]
    # df_g_inc_area['value'] = df_g_inc_area['annual increment [m3/yr]'] / df_g_inc_area['forest area [ha]']

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------
    # Adding the columns module, lever, level and string-pivot at the correct places
    df_g_inc_area_pathwaycalc['module'] = 'land-use'
    df_g_inc_area_pathwaycalc['lever'] = 'climate-smart-forestry'
    df_g_inc_area_pathwaycalc['level'] = 0
    df_g_inc_area_pathwaycalc['variables'] = 'agr_climate-smart-forestry_g-inc[m3/ha]'
    cols = df_g_inc_area_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    cols.insert(cols.index('geoscale'), cols.pop(cols.index('variables')))
    df_g_inc_area_pathwaycalc = df_g_inc_area_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    df_g_inc_area_pathwaycalc['geoscale'] = df_g_inc_area_pathwaycalc['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    df_g_inc_area_pathwaycalc['geoscale'] = df_g_inc_area_pathwaycalc['geoscale'].replace(
        'Netherlands (Kingdom of the)',
        'Netherlands')
    df_g_inc_area_pathwaycalc['geoscale'] = df_g_inc_area_pathwaycalc['geoscale'].replace('Czechia', 'Czech Republic')

    # ----------------------------------------------------------------------------------------------------------------------
    # CSF MANAGED ----------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # Is equal to 0 for all ots for all countries

    # Use df_g_inc_area_pathwaycalc as a structural basis
    csf_managed = df_g_inc_area_pathwaycalc.copy()

    # Add rows to have 1990-2022
    # Generate a DataFrame with all combinations of geoscale and timescale
    geoscale_values = csf_managed['geoscale'].unique()
    timescale_values = pd.Series(range(1990, 2023))

    # Create a DataFrame for the cartesian product
    cartesian_product = pd.MultiIndex.from_product([geoscale_values, timescale_values],
                                                   names=['geoscale', 'timescale']).to_frame(index=False)

    # Merge the original DataFrame with the cartesian product to include all combinations
    csf_managed = pd.merge(cartesian_product, csf_managed, on=['geoscale', 'timescale'], how='left')

    # Replace the variable with ots_agr_climate-smart-forestry_csf-man[m3/ha]
    csf_managed['variables'] = 'agr_climate-smart-forestry_csf-man[m3/ha]'

    # Replace the value with 0
    csf_managed['value'] = 0

    # PathwayCalc formatting
    csf_managed['module'] = 'land-use'
    csf_managed['lever'] = 'climate-smart-forestry'
    csf_managed['level'] = 0
    cols = csf_managed.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    csf_managed = csf_managed[cols]

    # ----------------------------------------------------------------------------------------------------------------------
    # FAWS SHARE & GSTOCK (FAWS & NON FAWS)  -------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # Read files (growing stock available fo wood supply and not)
    gstock_total = pd.read_excel(
        'data/data_forestry.xlsx',
        sheet_name='gstock_total_Mm3')
    gstock_faws = pd.read_excel(
        'data/data_forestry.xlsx',
        sheet_name='gstock_faws_Mm3')
    area_faws = pd.read_excel(
        'data/data_forestry.xlsx',
        sheet_name='forest_area_faws_1000ha')

    # Format correctly
    # Melting the dfs to have the relevant format (geoscale, year, value)
    gstock_faws = pd.melt(gstock_faws, id_vars=['geoscale'], var_name='timescale',
                          value_name='growing stock faws [Mm3]')
    area_faws = pd.melt(area_faws, id_vars=['geoscale'], var_name='timescale', value_name='area faws [1000ha]')
    gstock_total = pd.melt(gstock_total, id_vars=['geoscale'], var_name='timescale',
                           value_name='growing stock total [Mm3]')
    # Convert 'year' to integer type (optional, for better numerical handling)
    gstock_faws['timescale'] = gstock_faws['timescale'].astype(int)
    area_faws['timescale'] = area_faws['timescale'].astype(int)
    gstock_total['timescale'] = gstock_total['timescale'].astype(int)

    # Merge together and  with forest area (df_area) (also filters the relevant countries)
    gstock = pd.merge(gstock_faws, gstock_total, on=['geoscale', 'timescale'])
    gstock = pd.merge(gstock, area_faws, on=['geoscale', 'timescale'])
    gstock = pd.merge(gstock, df_area, on=['geoscale', 'timescale'])

    # Changing data type to numeric (except for the geoscale column)
    gstock.loc[:, gstock.columns != 'geoscale'] = gstock.loc[:, gstock.columns != 'geoscale'].apply(pd.to_numeric,
                                                                                                    errors='coerce')

    # Growing stock not faws [m3] = Growing stock total [m3] - Growing stock faws [m3]
    gstock['growing stock non faws [Mm3]'] = gstock['growing stock total [Mm3]'] - gstock['growing stock faws [Mm3]']

    # Forest area not for wood supply [ha] = total forest area [ha] - forest available for wood supply [ha]
    gstock['area non faws [ha]'] = gstock['forest area [ha]'] - 1000 * gstock['area faws [1000ha]']

    # Growing stock faws [m3/ha] = 10**6 * Growing stock faws [Mm3] / forest available for wood supply [ha]
    gstock['Growing stock faws [m3/ha]'] = (10 ** 6 * gstock['growing stock faws [Mm3]']) / (
                1000 * gstock['area faws [1000ha]'])

    # Growing stock non faws [m3/ha] = 10**6 * Growing stock non faws [Mm3] / forest non faws [ha]
    gstock['Growing stock non faws [m3/ha]'] = (10 ** 6 * gstock['growing stock non faws [Mm3]']) / gstock[
        'area non faws [ha]']

    # Share faws [%] = total forest area [ha] - forest available for wood supply [ha]
    gstock['Share faws [%]'] = 1000 * gstock['area faws [1000ha]'] / gstock['forest area [ha]']

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------
    # Melt the DataFrame
    gstock_pathwaycalc = pd.melt(gstock, id_vars=['geoscale', 'timescale'],
                                 value_vars=['Growing stock faws [m3/ha]', 'Growing stock non faws [m3/ha]',
                                             'Share faws [%]'],
                                 var_name='Item', value_name='value')

    # Read excel file
    df_dict_forestry = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='climate-smart-forestry')

    # Merge based on 'Item'
    gstock_pathwaycalc = pd.merge(df_dict_forestry, gstock_pathwaycalc, on='Item')

    # Drop the 'Item' column
    gstock_pathwaycalc = gstock_pathwaycalc.drop(columns=['Item'])

    # Adding the columns module, lever, level and string-pivot at the correct places
    gstock_pathwaycalc['module'] = 'land-use'
    gstock_pathwaycalc['lever'] = 'climate-smart-forestry'
    gstock_pathwaycalc['level'] = 0
    cols = gstock_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    gstock_pathwaycalc = gstock_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    gstock_pathwaycalc['geoscale'] = gstock_pathwaycalc['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    gstock_pathwaycalc['geoscale'] = gstock_pathwaycalc['geoscale'].replace('Netherlands (Kingdom of the)',
                                                                            'Netherlands')
    gstock_pathwaycalc['geoscale'] = gstock_pathwaycalc['geoscale'].replace('Czechia', 'Czech Republic')

    # ----------------------------------------------------------------------------------------------------------------------
    # HARVESTING RATE -------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    # Read files (growing stock available fo wood supply and not)
    h_rate = pd.read_excel(
        'data/data_forestry.xlsx',
        sheet_name='h-rate')

    # Replace - with Na
    # List of columns to modify
    columns_to_replace = [1990, 2000, 2010, 2015]
    # Replace '-' with NaN in the specified columns
    h_rate[columns_to_replace] = h_rate[columns_to_replace].replace('-', np.nan)

    # Format correctly
    # Melting the dfs to have the relevant format (geoscale, year, value)
    h_rate = pd.melt(h_rate, id_vars=['geoscale'], var_name='timescale', value_name='value')
    # Convert 'year' to integer type (optional, for better numerical handling)
    h_rate['timescale'] = h_rate['timescale'].astype(int)

    # Merge with forest area (df_area) (to filter the relevant countries) then filter out
    h_rate = pd.merge(h_rate, df_area, on=['geoscale', 'timescale'])
    h_rate = h_rate[['geoscale', 'timescale', 'value']]

    # Create copy
    h_rate_pathwaycalc = h_rate.copy()

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------
    # Adding the columns module, lever, level and string-pivot at the correct places
    h_rate_pathwaycalc['module'] = 'land-use'
    h_rate_pathwaycalc['lever'] = 'climate-smart-forestry'
    h_rate_pathwaycalc['level'] = 0
    h_rate_pathwaycalc['variables'] = 'agr_climate-smart-forestry_h-rate[%]'
    cols = h_rate_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    cols.insert(cols.index('geoscale'), cols.pop(cols.index('variables')))
    h_rate_pathwaycalc = h_rate_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    h_rate_pathwaycalc['geoscale'] = h_rate_pathwaycalc['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    h_rate_pathwaycalc['geoscale'] = h_rate_pathwaycalc['geoscale'].replace('Netherlands (Kingdom of the)',
                                                                            'Netherlands')
    h_rate_pathwaycalc['geoscale'] = h_rate_pathwaycalc['geoscale'].replace('Czechia', 'Czech Republic')

    # ----------------------------------------------------------------------------------------------------------------------
    # NATURAL LOSSES -------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # Read file
    nat_losses = pd.read_excel(
        'data/data_forestry.xlsx',
        sheet_name='nat-losses_1000ha')

    # Format correctly
    # Melt the DataFrame to long format
    df_melted = pd.melt(nat_losses, id_vars=['geoscale'], var_name='variable', value_name='Losses [1000ha]')

    # Extract 'item' and 'year' from the 'variable' column
    df_melted['Item'] = df_melted['variable'].str.extract(r'^(.*?)\s\d{4}$')[0]
    df_melted['timescale'] = df_melted['variable'].str.extract(r'(\d{4})$')[0]

    # Drop the original 'variable' column
    df_melted = df_melted.drop(columns=['variable'])

    # Rearrange the columns
    nat_losses = df_melted[['geoscale', 'timescale', 'Item', 'Losses [1000ha]']]

    # Change type to numeric for timescale to merge
    nat_losses['timescale'] = nat_losses['timescale'].apply(pd.to_numeric, errors='coerce')

    # Adding forest area and total growing stock
    nat_losses = pd.merge(nat_losses, df_area, on=['geoscale', 'timescale'])
    nat_losses = pd.merge(nat_losses, gstock_total, on=['geoscale', 'timescale'])

    # Change type to numeric
    numeric_cols = nat_losses.columns[3:]  # Get all columns except the first three
    nat_losses[numeric_cols] = nat_losses[numeric_cols].apply(pd.to_numeric,
                                                              errors='coerce')  # Convert to numeric, if not already

    # Ratio of losses area compared to total forest area
    nat_losses['Ratio of losses'] = 1000 * nat_losses['Losses [1000ha]'] / nat_losses['forest area [ha]']

    # Growing stock total [m3/ha] = Growing stock [Mm3] / forest area [ha]
    nat_losses['Growing stock total [m3/ha]'] = 10 ** 6 * nat_losses['growing stock total [Mm3]'] / nat_losses[
        'forest area [ha]']

    # Losses [m3/ha] = Ratio of losses [%] * Growing stock total [m3/ha]
    nat_losses['value'] = nat_losses['Ratio of losses'] * nat_losses['Growing stock total [m3/ha]']

    # Filtering
    nat_losses_pathwaycalc = nat_losses.copy()
    nat_losses_pathwaycalc = nat_losses_pathwaycalc[['Item', 'geoscale', 'timescale', 'value']]

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------
    # Read excel file
    df_dict_forestry = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='climate-smart-forestry')

    # Merge based on 'Item'
    nat_losses_pathwaycalc = pd.merge(df_dict_forestry, nat_losses_pathwaycalc, on='Item')

    # Drop the 'Item' column
    nat_losses_pathwaycalc = nat_losses_pathwaycalc.drop(columns=['Item'])

    # Adding the columns module, lever, level and string-pivot at the correct places
    nat_losses_pathwaycalc['module'] = 'land-use'
    nat_losses_pathwaycalc['lever'] = 'climate-smart-forestry'
    nat_losses_pathwaycalc['level'] = 0
    cols = nat_losses_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    nat_losses_pathwaycalc = nat_losses_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    nat_losses_pathwaycalc['geoscale'] = nat_losses_pathwaycalc['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    nat_losses_pathwaycalc['geoscale'] = nat_losses_pathwaycalc['geoscale'].replace('Netherlands (Kingdom of the)',
                                                                                    'Netherlands')
    nat_losses_pathwaycalc['geoscale'] = nat_losses_pathwaycalc['geoscale'].replace('Czechia', 'Czech Republic')

    # ----------------------------------------------------------------------------------------------------------------------
    # FINAL RESULT ---------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # Concat all dfs together
    df_csf = pd.concat([df_g_inc_area_pathwaycalc, csf_managed])
    df_csf = pd.concat([df_csf, gstock_pathwaycalc])
    df_csf = pd.concat([df_csf, h_rate_pathwaycalc])
    df_csf = pd.concat([df_csf, nat_losses_pathwaycalc])

    # Extrapolating
    df_climate_smart_forestry_pathwaycalc = ensure_structure(df_csf)
    df_climate_smart_forestry_pathwaycalc = linear_fitting_ots_db(df_climate_smart_forestry_pathwaycalc, years_ots,
                                                                   countries='all')

    return df_climate_smart_forestry_pathwaycalc, csf_managed

# CalculationLeaf LAND MANAGEMENT --------------------------------------------------------------------------------------
def land_management_processing(csf_managed):

    # ----------------------------------------------------------------------------------------------------------------------
    # LAND MATRIX & LAND MAN USE----------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # Importing UNFCCC excel files and reading them with a loop (only for Switzerland) Table 4.1 ---------------------------
    # Putting in a df in 3 dimensions (from, to, year)
    # Define the path where the Excel files are located
    folder_path = 'data/data_unfccc_2023'

    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter and sort files by the year (1990 to 2020)
    sorted_files = sorted([f for f in files if f.startswith('CHE_2023_') and int(f.split('_')[2]) in range(1990, 2021)],
                          key=lambda x: int(x.split('_')[2]))

    # Initialize a list to store DataFrames
    data_frames = []

    # Loop through sorted files, read the required rows, and append to the list
    for file in sorted_files:
        # Extract the year from the filename
        year = int(file.split('_')[2])

        # Full path to the file
        file_path = os.path.join(folder_path, file)

        # Read the specific rows and sheet from the Excel file
        df = pd.read_excel(file_path, sheet_name='Table4.1', skiprows=4, nrows=14, header=None)

        # Add a column for the year to the DataFrame
        df['Year'] = year

        # Append to the list of DataFrames
        data_frames.append(df)

    # Combine all DataFrames into a single DataFrame with a multi-index
    combined_df = pd.concat(data_frames, axis=0).set_index(['Year'])

    # Create a 3D array
    values_3d = np.array([df.values for df in data_frames])

    # Convert array in string
    data = values_3d.astype(str)

    # Create a row mask where the first column of each 14x13 slice doesn't contain 'unmanaged' -----------------------------
    row_mask = np.all(np.core.defchararray.find(data[:, :, 0], 'unmanaged') == -1, axis=0)

    # Create a column mask where the first row of each 14x13 slice doesn't contain 'unmanaged'
    col_mask = np.all(np.core.defchararray.find(data[:, 0, :], 'unmanaged') == -1, axis=0)

    # Apply the row mask to keep rows in each slice that do not contain 'unmanaged' in the first column
    filtered_data = data[:, row_mask, :]

    # Apply the column mask to keep columns in each slice that do not contain 'unmanaged' in the first row
    filtered_data = filtered_data[:, :, col_mask]

    # Creating a copy due to mask issue in the following steps
    filtered_data = filtered_data.copy()

    # Dropping the row that contain 'FROM' (index 1) ---------------------------------------------------------------------------------
    # Function to drop the second row (index 1) in a 14x13 slice
    def drop_second_row(slice_2d):
        # Create a mask for all rows except the one to drop (row index 1)
        row_mask = np.arange(slice_2d.shape[0]) != 1

        # Keep only the rows that are not the second row
        filtered_slice = slice_2d[row_mask, :]

        return filtered_slice

    # Apply the function to each 14x13 slice
    filtered_data_2 = np.array([drop_second_row(filtered_data[i]) for i in range(filtered_data.shape[0])])

    # Create a copy for potential issues due to mask
    filtered_data_2 = filtered_data_2.copy()

    # LAND MATRIX -------------------------------------------------------------------------------------------------------------

    # Create a copy
    array_land_matrix = filtered_data_2.copy()

    # Drop the unwanted row and column to only keep the land to and from (not final)
    # Function to drop the second row (index 1) in a 14x13 slice
    def drop_rows_and_columns(slice_2d, rows_to_drop=None, cols_to_drop=None):
        """
        Drop specific rows and columns from a 2D NumPy array.

        Parameters:
        slice_2d (np.ndarray): The 2D NumPy array from which rows and columns will be dropped.
        rows_to_drop (list or None): List of row indices to be dropped. If None, no rows are dropped.
        cols_to_drop (list or None): List of column indices to be dropped. If None, no columns are dropped.

        Returns:
        np.ndarray: The modified 2D NumPy array with specified rows and columns removed.
        """
        # If rows_to_drop is None or empty, don't drop any rows
        if rows_to_drop is None:
            rows_to_drop = []
        # If cols_to_drop is None or empty, don't drop any columns
        if cols_to_drop is None:
            cols_to_drop = []

        # Create a mask for rows to keep (not in rows_to_drop)
        if rows_to_drop:
            row_mask = np.ones(slice_2d.shape[0], dtype=bool)
            row_mask[rows_to_drop] = False
        else:
            row_mask = np.ones(slice_2d.shape[0], dtype=bool)

        # Create a mask for columns to keep (not in cols_to_drop)
        if cols_to_drop:
            col_mask = np.ones(slice_2d.shape[1], dtype=bool)
            col_mask[cols_to_drop] = False
        else:
            col_mask = np.ones(slice_2d.shape[1], dtype=bool)

        # Apply the masks to keep only the rows and columns that are not in the drop lists
        filtered_slice = slice_2d[row_mask, :][:, col_mask]

        return filtered_slice

    # Apply the function to each 14x13 slice
    array_land_matrix = np.array(
        [drop_rows_and_columns(array_land_matrix[i], rows_to_drop=[7, 8], cols_to_drop=None) for i in
         range(array_land_matrix.shape[0])])

    # Transform in a df
    # Reshape array
    array_2d = array_land_matrix.reshape(-1, array_land_matrix.shape[2])
    # Convert the 2D array to a DataFrame
    df_land_matrix = pd.DataFrame(array_2d)

    # Set the first row as index
    df_land_matrix.columns = df_land_matrix.iloc[0]  # Set the first row as the new column headers
    df_land_matrix = df_land_matrix[1:]  # Remove the first row from the DataFrame
    df_land_matrix = df_land_matrix.reset_index(drop=True)  # Reset the index after removing the first row

    # Drop the rows that contain TO:
    df_land_matrix = df_land_matrix[
        ~df_land_matrix.apply(lambda row: row.astype(str).str.contains('TO:').any(), axis=1)]

    # Rename cols 1990 into timescale
    df_land_matrix.rename(columns={'1990': 'timescale'}, inplace=True)

    # Change type to numeric
    numeric_cols = df_land_matrix.columns[1:]  # Get all columns except the first three
    df_land_matrix[numeric_cols] = df_land_matrix[numeric_cols].apply(pd.to_numeric,
                                                                      errors='coerce')  # Convert to numeric, if not already

    # Divide each column by the initial area to convert from [ha] to [%]
    df_land_matrix['Forest land (managed)'] = df_land_matrix['Forest land (managed)'] / df_land_matrix['Initial area']
    df_land_matrix['Cropland '] = df_land_matrix['Cropland '] / df_land_matrix['Initial area']
    df_land_matrix['Grassland (managed)'] = df_land_matrix['Grassland (managed)'] / df_land_matrix['Initial area']
    df_land_matrix['Wetlands (managed)'] = df_land_matrix['Wetlands (managed)'] / df_land_matrix['Initial area']
    df_land_matrix['Settlements'] = df_land_matrix['Settlements'] / df_land_matrix['Initial area']
    df_land_matrix['Other land'] = df_land_matrix['Other land'] / df_land_matrix['Initial area']

    # Drop the column 'Initial area'
    df_land_matrix = df_land_matrix.drop(columns=['Initial area'])

    # Melt to have year, values, land-to and land-from
    df_land_matrix = pd.melt(df_land_matrix, id_vars=['TO:', 'timescale'],
                             value_vars=['Forest land (managed)', 'Cropland ', 'Grassland (managed)',
                                         'Wetlands (managed)',
                                         'Settlements', 'Other land'],
                             var_name='FROM:', value_name='value')

    # Combine 'TO:' and 'FROM:' columns into a single 'item' column
    df_land_matrix['Item'] = df_land_matrix['FROM:'] + ' to ' + df_land_matrix['TO:']

    # Drop the original 'TO:' and 'FROM:' columns if no longer needed
    df_land_matrix = df_land_matrix.drop(columns=['TO:', 'FROM:'])

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------
    # Match with dictionary for correct names
    # Read excel file
    df_dict_land_man = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='land-management')

    # Merge based on 'Item'
    df_land_matrix_pathwaycalc = pd.merge(df_dict_land_man, df_land_matrix, on='Item')

    # Drop the 'Item' column
    df_land_matrix_pathwaycalc = df_land_matrix_pathwaycalc.drop(columns=['Item'])

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_land_matrix_pathwaycalc['module'] = 'land-use'
    df_land_matrix_pathwaycalc['lever'] = 'land-man'
    df_land_matrix_pathwaycalc['level'] = 0
    df_land_matrix_pathwaycalc['geoscale'] = 'Switzerland'
    cols = df_land_matrix_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    cols.insert(cols.index('timescale'), cols.pop(cols.index('geoscale')))
    df_land_matrix_pathwaycalc = df_land_matrix_pathwaycalc[cols]

    # LAND USE -------------------------------------------------------------------------------------------------------------
    # Use the row 'Final area' for 'land-man_use' forest, other, settlement and wetland ------------------------------------
    def keep_final_use_row(slice_2d):
        # Create a mask for all rows except the one to drop (row index 1)
        row_mask = np.arange(slice_2d.shape[0]) == 7

        # Keep only the rows that are not the second row
        filtered_slice = slice_2d[row_mask, :]

        return filtered_slice

    # Apply the function to each 14x13 slice
    filtered_data_land_use = np.array([keep_final_use_row(filtered_data_2[i]) for i in range(filtered_data_2.shape[0])])

    # Transform  array in df
    # Remove the extra dimension
    reshaped_array = filtered_data_land_use.reshape(31, 9)
    # Create a DataFrame from the reshaped array
    df_land_use = pd.DataFrame(reshaped_array)

    # Change the correct indices for columns
    new_column_names = ['element', 'agr_land-man_use_forest[ha]', 'agr_land-man_use_cropland[ha]',
                        'agr_land-man_use_grassland[ha]', 'agr_land-man_use_wetland[ha]',
                        'agr_land-man_use_settlement[ha]',
                        'agr_land-man_use_other[ha]', 'initial area', 'timescale']

    # Assign the new column names to the DataFrame
    df_land_use.columns = new_column_names

    # Dropping the columns 'element' and 'initial area'
    df_land_use = df_land_use.drop(columns=['element', 'initial area'])
    df_land_use_filtered = df_land_use.drop(columns=['agr_land-man_use_cropland[ha]', 'agr_land-man_use_grassland[ha]'])

    # Melting the dfs to have the relevant format (geoscale, year, value)
    df_land_use_pathwaycalc = pd.melt(df_land_use_filtered, id_vars=['timescale'], var_name='variables',
                                      value_name='value')

    # Convert the 'value' column from string to numeric
    df_land_use_pathwaycalc['value'] = pd.to_numeric(df_land_use_pathwaycalc['value'], errors='coerce')

    # Unit conversion [kha] => [ha]
    df_land_use_pathwaycalc['value'] = df_land_use_pathwaycalc['value'] * 1000

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------
    # Adding the columns module, lever, level and string-pivot at the correct places
    df_land_use_pathwaycalc['module'] = 'land-use'
    df_land_use_pathwaycalc['lever'] = 'land-man'
    df_land_use_pathwaycalc['level'] = 0
    df_land_use_pathwaycalc['geoscale'] = 'Switzerland'
    cols = df_land_use_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    cols.insert(cols.index('timescale'), cols.pop(cols.index('geoscale')))
    df_land_use_pathwaycalc = df_land_use_pathwaycalc[cols]

    # ----------------------------------------------------------------------------------------------------------------------
    # LAND DYN -------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # 1 for forest, 0 for grassland and unmanaged for all ots

    # Using csf_managed as a structural basis
    df_land_dyn_forest = csf_managed.copy()
    df_land_dyn_grass = csf_managed.copy()
    df_land_dyn_unmanaged = csf_managed.copy()

    # Changing values and variable name
    df_land_dyn_forest['variables'] = 'agr_land-man_dyn_forest[%]'
    df_land_dyn_forest['value'] = 1
    df_land_dyn_grass['variables'] = 'agr_land-man_dyn_grassland[%]'
    df_land_dyn_grass['value'] = 0
    df_land_dyn_unmanaged['variables'] = 'agr_land-man_dyn_unmanaged[%]'
    df_land_dyn_unmanaged['value'] = 0

    # Concatenating
    df_land_dyn = pd.concat([df_land_dyn_forest, df_land_dyn_grass])
    df_land_dyn = pd.concat([df_land_dyn, df_land_dyn_unmanaged])

    # PathwayCalc formatting
    df_land_dyn['lever'] = 'land-man'

    # ----------------------------------------------------------------------------------------------------------------------
    # LAND GAP -------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    # Difference in values between FAO and UNFCCC

    # Read FAO Values (for Switzerland) --------------------------------------------------------------------------------------------
    # List of countries
    list_countries_CH = ['Switzerland']

    # List of elements
    list_elements = ['Area']

    list_items = ['-- Cropland', '-- Permanent meadows and pastures', 'Forest land']

    # 1990 - 2022
    ld = faostat.list_datasets()
    code = 'RL'
    pars = faostat.list_pars(code)
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries_CH]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                  '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                  '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items,
        'year': my_years
    }
    df_land_use_fao = faostat.get_data_df(code, pars=my_pars, strval=False)

    # Drop columns
    df_land_use_fao = df_land_use_fao.drop(
        columns=['Domain Code', 'Domain', 'Area Code', 'Element Code',
                 'Item Code', 'Year Code', 'Unit', 'Element', 'Area'])

    # Reshape
    df = df_land_use_fao.copy()
    # Reshape the DataFrame using pivot
    reshaped_df = df.pivot(index='Year', columns='Item', values='Value')
    # Reset the index if you want a flat DataFrame with 'item' as a column
    reshaped_df = reshaped_df.reset_index()

    # Read UNFCCC values (for Switzerland)
    # done in previous steps, result is df_land_use

    # Merged based on timescale
    df_land_gap = pd.merge(reshaped_df, df_land_use, left_on='Year', right_on='timescale')

    # Change type to numeric
    numeric_cols = df_land_gap.columns[1:]  # Get all columns except the first three
    df_land_gap[numeric_cols] = df_land_gap[numeric_cols].apply(pd.to_numeric,
                                                                errors='coerce')

    # Computing the difference & Unit conversion [kha] => [ha]
    df_land_gap['agr_land-man_gap_cropland[ha]'] = 1000 * (df_land_gap['agr_land-man_use_cropland[ha]'] -
                                                          df_land_gap['Cropland'])
    df_land_gap['agr_land-man_gap_forest[ha]'] = 1000 * (df_land_gap['agr_land-man_use_forest[ha]'] -
                                                        df_land_gap['Forest land'])
    df_land_gap['agr_land-man_gap_grassland[ha]'] = 1000 * (df_land_gap['agr_land-man_use_grassland[ha]'] -
                                                           df_land_gap['Permanent meadows and pastures'])
    # df_land_gap['agr_land-man_gap_other[ha]'] = df_land_gap[''] - df_land_gap['']
    # df_land_gap['agr_land-man_gap_settlement[ha]'] = df_land_gap[''] - df_land_gap['']
    # df_land_gap['agr_land-man_gap_wetland[ha]'] = df_land_gap[''] - df_land_gap['']

    # Keep only the useful columns
    df_land_gap = df_land_gap[['timescale', 'agr_land-man_gap_cropland[ha]', 'agr_land-man_gap_forest[ha]',
                               'agr_land-man_gap_grassland[ha]']]

    # Melt the df
    df_land_gap = pd.melt(df_land_gap, id_vars=['timescale'],
                                          var_name='variables', value_name='value')

    # PathwayCalc formatting
    df_land_gap['module'] = 'land-use'
    df_land_gap['lever'] = 'land-man'
    df_land_gap['level'] = 0
    df_land_gap['geoscale'] = 'Switzerland' # Setting the geoscale for CH
    cols = df_land_gap.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    cols.insert(cols.index('timescale'), cols.pop(cols.index('variables')))
    df_land_gap = df_land_gap[cols]

    # Rename countries to Pathaywcalc name
    #df_land_gap['geoscale'] = df_land_gap['geoscale'].replace(
    #    'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    #df_land_gap['geoscale'] = df_land_gap['geoscale'].replace(
    #    'Netherlands (Kingdom of the)', 'Netherlands')
    #df_land_gap['geoscale'] = df_land_gap['geoscale'].replace('Czechia', 'Czech Republic')

    # ----------------------------------------------------------------------------------------------------------------------
    # FINAL RESULTS --------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    # Concatenating all dfs
    df_land_management_pathwaycalc = pd.concat([df_land_matrix_pathwaycalc, df_land_use_pathwaycalc])
    df_land_management_pathwaycalc = pd.concat([df_land_management_pathwaycalc, df_land_dyn])
    df_land_management_pathwaycalc = pd.concat([df_land_management_pathwaycalc, df_land_gap])

    # Extrapolating
    df_land_management_pathwaycalc = ensure_structure(df_land_management_pathwaycalc)
    df_land_management_pathwaycalc = linear_fitting_ots_db(df_land_management_pathwaycalc, years_ots, countries='all')

    return df_land_management_pathwaycalc

# CalculationLeaf BIOENERGY CAPACITY -----------------------------------------------------------------------------------

def bioernergy_capacity_processing(df_csl_feed):
    # Using and formatting df_csl_feed as a structural basis for constant ots values across all countries
    df_bioenergy_capacity_all = df_csl_feed.copy()
    df_bioenergy_capacity_all = df_bioenergy_capacity_all.drop(columns=['Item', 'Feed'])
    # Dropping duplicate rows
    df_bioenergy_capacity_all = df_bioenergy_capacity_all.drop_duplicates()

    # Using and formatting df_csl_feed as a structural basis for constant ots values in Switzerland
    df_bioenergy_capacity_CH = df_bioenergy_capacity_all.copy()
    # Keeping only the rows where geoscale = Switzerland
    df_bioenergy_capacity_CH = df_bioenergy_capacity_CH[df_bioenergy_capacity_CH['Area'] == 'Switzerland']

    # Adding the constant ots values
    df_bioenergy_capacity_CH['agr_bioenergy-capacity_elec_biogases-hf[GW]'] = 0.0
    df_bioenergy_capacity_CH['agr_bioenergy-capacity_elec_biogases[GW]'] = 0.0
    df_bioenergy_capacity_CH['agr_bioenergy-capacity_elec_solid-biofuel-hf[GW]'] = 0.0
    df_bioenergy_capacity_CH['agr_bioenergy-capacity_elec_solid-biofuel[GW]'] = 0.0
    df_bioenergy_capacity_CH['agr_bioenergy-capacity_efficiency_biogases-hf[%]'] = 1.0
    df_bioenergy_capacity_CH['agr_bioenergy-capacity_efficiency_biogases[%]'] = 1.0
    df_bioenergy_capacity_CH['agr_bioenergy-capacity_efficiency_solid-biofuel-hf[%]'] = 1.0
    df_bioenergy_capacity_CH['agr_bioenergy-capacity_efficiency_solid-biofuel[%]'] = 1.0
    df_bioenergy_capacity_CH['agr_bioenergy-capacity_liq_biodiesel[TWh]'] = 0.0
    df_bioenergy_capacity_CH['agr_bioenergy-capacity_liq_biogasoline[TWh]'] = 0.0
    df_bioenergy_capacity_CH['agr_bioenergy-capacity_liq_biojetkerosene[TWh]'] = 0.0
    df_bioenergy_capacity_CH['agr_bioenergy-capacity_liq_other-liquid-biofuel[TWh]'] = 0.0
    df_bioenergy_capacity_CH['agr_bioenergy-capacity_load-factor_biogases-hf[%]'] = 0.0
    df_bioenergy_capacity_CH['agr_bioenergy-capacity_load-factor_biogases[%]'] = 0.0
    df_bioenergy_capacity_CH['agr_bioenergy-capacity_load-factor_solid-biofuel-hf[%]'] = 0.0
    df_bioenergy_capacity_CH['agr_bioenergy-capacity_load-factor_solid-biofuel[%]'] = 0.0
    df_bioenergy_capacity_CH['agr_bioenergy-capacity_bgs-mix_digestor[%]'] = 1.0
    df_bioenergy_capacity_CH['agr_bioenergy-capacity_bgs-mix_landfill[%]'] = 0.3647531014351739
    df_bioenergy_capacity_CH['agr_bioenergy-capacity_bgs-mix_other-biogases[%]'] = 0.0
    df_bioenergy_capacity_CH['agr_bioenergy-capacity_bgs-mix_ren-mun-wastes[%]'] = 0.3567258574556069
    df_bioenergy_capacity_CH['agr_bioenergy-capacity_bgs-mix_sewage[%]'] = 0.6352468985648261
    df_bioenergy_capacity_CH['agr_bioenergy-capacity_bgs-mix_thermal-biogases[%]'] = 0.0

    # Drop columns 'Total feed' and 'Feed ratio'
    #df_bioenergy_capacity_CH = df_bioenergy_capacity_CH.drop(columns=['Total feed', 'Feed ratio'])

    # Melting
    df_bioenergy_capacity_CH_pathwaycalc= pd.melt(df_bioenergy_capacity_CH, id_vars=['Area', 'Year'],
                                          var_name='variables', value_name='value')

    # Renaming columns
    df_bioenergy_capacity_CH_pathwaycalc.rename(columns={'Area': 'geoscale', 'Year': 'timescale'}, inplace=True)

    # PathwayCalc formatting
    df_bioenergy_capacity_CH_pathwaycalc['module'] = 'agriculture'
    df_bioenergy_capacity_CH_pathwaycalc['lever'] = 'bioenergy-capacity'
    df_bioenergy_capacity_CH_pathwaycalc['level'] = 0
    cols = df_bioenergy_capacity_CH_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    cols.insert(cols.index('timescale'), cols.pop(cols.index('variables')))
    df_bioenergy_capacity_CH_pathwaycalc = df_bioenergy_capacity_CH_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    df_bioenergy_capacity_CH_pathwaycalc['geoscale'] = df_bioenergy_capacity_CH_pathwaycalc['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    df_bioenergy_capacity_CH_pathwaycalc['geoscale'] = df_bioenergy_capacity_CH_pathwaycalc['geoscale'].replace(
        'Netherlands (Kingdom of the)', 'Netherlands')
    df_bioenergy_capacity_CH_pathwaycalc['geoscale'] = df_bioenergy_capacity_CH_pathwaycalc['geoscale'].replace('Czechia', 'Czech Republic')

    # Extrapolating
    df_bioenergy_capacity_CH_pathwaycalc = ensure_structure(df_bioenergy_capacity_CH_pathwaycalc)
    df_bioenergy_capacity_CH_pathwaycalc = linear_fitting_ots_db(df_bioenergy_capacity_CH_pathwaycalc, years_ots, countries='all')

    return df_bioenergy_capacity_CH_pathwaycalc

# CalculationLeaf BIOMASS HIERARCHY ------------------------------------------------------------------------------------

def biomass_bioernergy_hierarchy_processing(df_csl_feed):
    # ------------------------------------------------------------------------------------------------------------------
    # BIOMASS MIX
    # ------------------------------------------------------------------------------------------------------------------
    # Load from previous EuCalc Data
    df_biomass_mix_data = pd.read_csv(
        'data/agriculture_biomass-use-hierarchy_eucalc.csv', sep=';')

    # Filter columns
    df_filtered_columns = df_biomass_mix_data[['geoscale', 'timescale', 'eucalc-name', 'value']]
    df_filtered_columns.copy()

    # rename col 'eucalc-name' in 'variables'
    df_filtered_columns = df_filtered_columns.rename(columns={'eucalc-name': 'variables'})

    # Filter rows that contains biomass-mix
    df_filtered_rows = df_filtered_columns[
        df_filtered_columns['variables'].str.contains('ots_agr_biomass-hierarchy_biomass-mix', case=False, na=False)
    ]

    # Drop rows where 'variables' contains '%_1'
    df_biomass_mix = df_filtered_rows[~df_filtered_rows['variables'].str.contains('%_1', na=False)]
    df_biomass_mix = df_biomass_mix.copy()

    # Rename from ots_agr to agr
    df_biomass_mix['variables'] = df_biomass_mix['variables'].str.replace('ots_agr', 'agr', regex=False)

    # Delete additional countries (Vaud, EU27, Paris)
    df_biomass_mix = df_biomass_mix[df_biomass_mix['geoscale'] != 'Vaud']
    df_biomass_mix = df_biomass_mix[df_biomass_mix['geoscale'] != 'EU27']
    df_biomass_mix = df_biomass_mix[df_biomass_mix['geoscale'] != 'Paris']

    # ------------------------------------------------------------------------------------------------------------------
    # BIOMASS RESIDUES CEREALS BURNT & SOIL
    # ------------------------------------------------------------------------------------------------------------------
    # Load from previous EuCalc Data
    df_biomass_residues_data = pd.read_csv(
        'data/agriculture_biomass-use-hierarchy_eucalc.csv', sep=';')

    # Filter columns
    df_filtered_columns = df_biomass_residues_data[['geoscale', 'timescale', 'eucalc-name', 'value']]

    # rename col 'eucalc-name' in 'variables'
    df_filtered_columns = df_filtered_columns.rename(columns={'eucalc-name': 'variables'})

    # Filter rows that contains biomass-mix
    df_filtered_rows = df_filtered_columns[
        df_filtered_columns['variables'].str.contains('ots_agr_biomass-hierarchy_crop_cereal', case=False, na=False)
    ]

    # Drop rows where 'variables' contains '%_1'
    df_biomass_residues = df_filtered_rows[~df_filtered_rows['variables'].str.contains('%_1', na=False)].copy()

    # Rename from ots_agr to agr
    df_biomass_residues['variables'] = df_biomass_residues['variables'].str.replace('ots_agr', 'agr', regex=False)

    # Delete additional countries (Vaud, EU27, Paris)
    df_biomass_residues = df_biomass_residues[df_biomass_residues['geoscale'] != 'Vaud']
    df_biomass_residues = df_biomass_residues[df_biomass_residues['geoscale'] != 'EU27']
    df_biomass_residues = df_biomass_residues[df_biomass_residues['geoscale'] != 'Paris']

    # ------------------------------------------------------------------------------------------------------------------
    # BIOMASS HIERARCHY
    # ------------------------------------------------------------------------------------------------------------------

    # Using and formatting df_csl_feed as a structural basis for constant ots values across all countries
    df_biomass_hierarchy_all = df_csl_feed.copy()
    df_biomass_hierarchy_all = df_biomass_hierarchy_all.drop(columns=['Item', 'Feed'])
    # Dropping duplicate rows
    df_biomass_hierarchy_all = df_biomass_hierarchy_all.drop_duplicates()

    # Using and formatting df_csl_feed as a structural basis for constant ots values in Switzerland
    df_biomass_hierarchy_CH = df_biomass_hierarchy_all.copy()
    # Keeping only the rows where geoscale = Switzerland
    df_biomass_hierarchy_CH = df_biomass_hierarchy_CH[df_biomass_hierarchy_CH['Area'] == 'Switzerland']

    # Renaming columns
    df_biomass_hierarchy_CH.rename(columns={'Area': 'geoscale', 'Year': 'timescale'}, inplace=True)
    df_biomass_hierarchy_all.rename(columns={'Area': 'geoscale', 'Year': 'timescale'}, inplace=True)

    # Adding ots values
    df_biomass_hierarchy_all['agr_biomass-hierarchy-bev-ibp-use-oth_fertilizer[%]'] = 0.05
    df_biomass_hierarchy_all['agr_biomass-hierarchy-bev-ibp-use-oth_solid-bioenergy[%]'] = 0.8
    df_biomass_hierarchy_all['agr_biomass-hierarchy-bev-ibp-use-oth_biogasoline[%]'] = 0.05
    df_biomass_hierarchy_all['agr_biomass-hierarchy_bioenergy_ibp_fdk[%]'] = 0.1
    df_biomass_hierarchy_all['agr_biomass-hierarchy_bioenergy_fdk-demand_eth_mix_cereal[%]'] = 0.5
    df_biomass_hierarchy_all['agr_biomass-hierarchy_bioenergy_fdk-demand_eth_mix_sugarcrop[%]'] = 0.5
    df_biomass_hierarchy_all['agr_biomass-hierarchy_bioenergy_fdk-demand_oil_mix_voil[%]'] = 1.0
    df_biomass_hierarchy_all['agr_biomass-hierarchy_bioenergy_liquid_biodiesel_btl[%]'] = 0.0
    df_biomass_hierarchy_all['agr_biomass-hierarchy_bioenergy_liquid_biodiesel_est[%]'] = 1.0
    df_biomass_hierarchy_all['agr_biomass-hierarchy_bioenergy_liquid_biodiesel_hvo[%]'] = 0.0
    df_biomass_hierarchy_all['agr_biomass-hierarchy_bioenergy_liquid_biogasoline_ezm[%]'] = 0.0
    df_biomass_hierarchy_all['agr_biomass-hierarchy_bioenergy_liquid_biogasoline_fer[%]'] = 1.0
    df_biomass_hierarchy_all['agr_biomass-hierarchy_bioenergy_liquid_biojetkerosene_btl[%]'] = 0.0
    df_biomass_hierarchy_all['agr_biomass-hierarchy_bioenergy_liquid_biojetkerosene_hvo[%]'] = 1.0

    # Drop columns 'Total feed' and 'Feed ratio'
    #df_biomass_hierarchy_all = df_biomass_hierarchy_all.drop(columns=['Total feed', 'Feed ratio'])

    # Melt df
    df_biomass_hierarchy_pathwaycalc = pd.melt(df_biomass_hierarchy_all, id_vars=['geoscale', 'timescale'],
                                               var_name='variables', value_name='value')

    # Rename countries to match with PathwayCalc
    df_biomass_hierarchy_pathwaycalc['geoscale'] = df_biomass_hierarchy_pathwaycalc['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    df_biomass_hierarchy_pathwaycalc['geoscale'] = df_biomass_hierarchy_pathwaycalc['geoscale'].replace('Netherlands (Kingdom of the)',
                                                                            'Netherlands')
    df_biomass_hierarchy_pathwaycalc['geoscale'] = df_biomass_hierarchy_pathwaycalc['geoscale'].replace('Czechia', 'Czech Republic')


    # Concat dfs
    df_biomass_hierarchy_pathwaycalc = pd.concat([df_biomass_hierarchy_pathwaycalc, df_biomass_mix])
    df_biomass_hierarchy_pathwaycalc = pd.concat([df_biomass_hierarchy_pathwaycalc, df_biomass_residues])

    # PathwayCalc formatting
    df_biomass_hierarchy_pathwaycalc['module'] = 'agriculture'
    df_biomass_hierarchy_pathwaycalc['lever'] = 'biomass-hierarchy'
    df_biomass_hierarchy_pathwaycalc['level'] = 0
    cols = df_biomass_hierarchy_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    df_biomass_hierarchy_pathwaycalc = df_biomass_hierarchy_pathwaycalc[cols]

    # Extrapolating
    df_biomass_hierarchy_pathwaycalc = ensure_structure(df_biomass_hierarchy_pathwaycalc)
    df_biomass_hierarchy_pathwaycalc = linear_fitting_ots_db(df_biomass_hierarchy_pathwaycalc, years_ots, countries='all')

    return df_biomass_hierarchy_pathwaycalc

# CalculationLeaf LIVESTOCK PROTEIN MEALS ------------------------------------------------------------------------------------
def livestock_protein_meals_processing(df_csl_feed):

    # Using and formatting df_csl_feed as a structural basis for constant ots values across all countries
    df_protein_meals_all = df_csl_feed.copy()
    df_protein_meals_all = df_protein_meals_all.drop(columns=['Item', 'Feed'])
    # Dropping duplicate rows
    df_protein_meals_all = df_protein_meals_all.drop_duplicates()

    # Adding ots values
    df_protein_meals_all['agr_alt-protein_abp-dairy-milk_algae[%]'] = 0.0
    df_protein_meals_all['agr_alt-protein_abp-dairy-milk_insect[%]'] = 0.0
    df_protein_meals_all['agr_alt-protein_abp-hens-egg_algae[%]'] = 0.0
    df_protein_meals_all['agr_alt-protein_abp-hens-egg_insect[%]'] = 0.0
    df_protein_meals_all['agr_alt-protein_meat-bovine_algae[%]'] = 0.0
    df_protein_meals_all['agr_alt-protein_meat-bovine_insect[%]'] = 0.0
    df_protein_meals_all['agr_alt-protein_meat-oth-animals_algae[%]'] = 0.0
    df_protein_meals_all['agr_alt-protein_meat-oth-animals_insect[%]'] = 0.0
    df_protein_meals_all['agr_alt-protein_meat-pig_algae[%]'] = 0.0
    df_protein_meals_all['agr_alt-protein_meat-pig_insect[%]'] = 0.0
    df_protein_meals_all['agr_alt-protein_meat-poultry_algae[%]'] = 0.0
    df_protein_meals_all['agr_alt-protein_meat-poultry_insect[%]'] = 0.0
    df_protein_meals_all['agr_alt-protein_meat-sheep_algae[%]'] = 0.0
    df_protein_meals_all['agr_alt-protein_meat-sheep_insect[%]'] = 0.0

    # Drop columns 'Total feed' and 'Feed ratio'
    #df_protein_meals_all = df_protein_meals_all.drop(columns=['Total feed', 'Feed ratio'])

    # Melt df
    df_protein_meals_pathwaycalc = pd.melt(df_protein_meals_all, id_vars=['Area', 'Year'],
                                           var_name='variables', value_name='value')

    # Renaming columns
    df_protein_meals_pathwaycalc.rename(columns={'Area': 'geoscale', 'Year': 'timescale'}, inplace=True)

    # PathwayCalc formatting
    df_protein_meals_pathwaycalc['module'] = 'agriculture'
    df_protein_meals_pathwaycalc['lever'] = 'alt-protein'
    df_protein_meals_pathwaycalc['level'] = 0
    cols = df_protein_meals_pathwaycalc.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    df_protein_meals_pathwaycalc = df_protein_meals_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    df_protein_meals_pathwaycalc['geoscale'] = df_protein_meals_pathwaycalc['geoscale'].replace(
        'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
    df_protein_meals_pathwaycalc['geoscale'] = df_protein_meals_pathwaycalc['geoscale'].replace(
        'Netherlands (Kingdom of the)', 'Netherlands')
    df_protein_meals_pathwaycalc['geoscale'] = df_protein_meals_pathwaycalc['geoscale'].replace(
        'Czechia', 'Czech Republic')

    # Extrapolating
    df_protein_meals_pathwaycalc = ensure_structure(df_protein_meals_pathwaycalc)
    df_protein_meals_pathwaycalc = linear_fitting_ots_db(df_protein_meals_pathwaycalc, years_ots,
                                                                 countries='all')

    return df_protein_meals_pathwaycalc




# CalculationLeaf CAL - LIFESTYLE -----------------------------------------------------------------------------------

def lifestyle_calibration(list_countries):
    # ----------------------------------------------------------------------------------------------------------------------
    # FOOD SUPPLY (DIET) ---------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # Read data ------------------------------------------------------------------------------------------------------------

    # Common for all
    # List of countries

    # FOOD BALANCE SHEETS (FBS) - -------------------------------------------------
    # List of elements
    list_elements = ['Food supply (kcal/capita/day)']
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

    # 1990 - 2013 - Population
    """list_elements = ['Total Population - Both sexes']
    list_items = ['Population']
    ld = faostat.list_datasets()
    code = 'FBSH'
    pars = faostat.list_pars(code)
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
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
    df_population_1990_2013 = faostat.get_data_df(code, pars=my_pars, strval=False)"""

    # 2010-2022
    list_elements = ['Food supply (kcal/capita/day)']
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
        df_diet_1990_2013['Item'].str.contains('Rice \(Milled Equivalent\)', case=False,
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
    # Merge the DataFrames on 'Area' and 'Year'
    #merged_df = pd.merge(
    #    pivot_df_diet_1990_2013,
    #    pivot_df_population_1990_2013[['Area', 'Year', 'Total Population - Both sexes']],  # Only keep the needed columns
    #    on=['Area', 'Year']
    #)

    # Multiplying population [capita] with food supply [kcal/capita/day] to have food supply [kcal] (per year implicitely)
    #merged_df['Food supply (kcal)'] = 365.25 * 1000 * merged_df['Total Population - Both sexes'] * merged_df['Food supply (kcal/capita/day)']
    #merged_df = merged_df[['Area', 'Year', 'Item', 'Food supply (kcal)']]

    # Concatenating all the years together
    pivot_df_diet = pd.concat([pivot_df_diet_1990_2013, pivot_df_diet_2010_2022])
    #pivot_df_diet = pd.concat([merged_df, pivot_df_diet_2010_2022])

    # Unit conversion [million kcal] => [kcal] (based on the definitions in FAOSTAT, even though it's written kcal)
    #pivot_df_diet_2010_2022['Food supply (kcal)'] = pivot_df_diet_2010_2022['Food supply (kcal)'] * 10 ** 6

    # Concatenating all the years together
    #pivot_df_diet = pd.concat([merged_df, pivot_df_diet_2010_2022])

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

    # Renaming existing columns (geoscale, timsecale, value)
    df_diet_calibration.rename(columns={'Area': 'geoscale', 'Year': 'timescale', 'Food supply (kcal/capita/day)': 'value'}, inplace=True)
    # df_diet_calibration.rename(columns={'Area': 'geoscale', 'Year': 'timescale', 'Food supply (kcal)': 'value'}, inplace=True)

    return df_diet_calibration


# CalculationLeaf CAL - LIVESTOCK & CROP -----------------------------------------------------------------------------------
def livestock_crop_calibration(df_energy_demand_cal, list_countries):
    # ----------------------------------------------------------------------------------------------------------------------
    # LIVESTOCK POPULATION -------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # Read data ------------------------------------------------------------------------------------------------------------

    # Common for all
    # List of countries

    # EMISSIONS FROM LIVESTOCK (GLE) - -------------------------------------------------
    # List of elements
    list_elements = ['Stocks']

    list_items = ['Swine + (Total)', 'Sheep and Goats + (Total)', 'Cattle, dairy', 'Cattle, non-dairy',
                  'Chickens, layers']

    list_items_poultry = ['Chickens, broilers', 'Ducks', 'Turkeys']

    list_items_others = ['Asses', 'Buffalo', 'Camels', 'Horses', 'Llamas', 'Mules and hinnies']
    list_sources = ['FAO TIER 1']

    # 1990 - 2022
    ld = faostat.list_datasets()
    code = 'GLE'
    pars = faostat.list_pars(code)
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    my_sources = [faostat.get_par(code, 'sources')[i] for i in list_sources]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                  '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                  '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items,
        'year': my_years,
        'source': my_sources
    }
    df_liv_population = faostat.get_data_df(code, pars=my_pars, strval=False)

    my_items_poultry = [faostat.get_par(code, 'item')[i] for i in list_items_poultry]
    my_pars_poultry = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items_poultry,
        'year': my_years,
        'source': my_sources
    }
    df_liv_population_poultry = faostat.get_data_df(code, pars=my_pars_poultry, strval=False)

    my_items_others = [faostat.get_par(code, 'item')[i] for i in list_items_others]
    my_pars_others = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items_others,
        'year': my_years,
        'source': my_sources
    }
    df_liv_population_others = faostat.get_data_df(code, pars=my_pars_others, strval=False)

    # Filtering to keep wanted columns
    columns_to_filter = ['Area', 'Element', 'Item', 'Year', 'Value']
    df_liv_population = df_liv_population[columns_to_filter]
    df_liv_population_poultry = df_liv_population_poultry[columns_to_filter]
    df_liv_population_others = df_liv_population_others[columns_to_filter]

    # Creating one column with Item and Element
    #df_liv_population['Item'] = df_liv_population['Item'] + ' ' + df_liv_population['Element']
    df_liv_population = df_liv_population.drop(columns=['Element'])

    # Reading excel lsu equivalent
    df_lsu = pd.read_excel(
        'dictionaries/lsu_equivalent.xlsx',
        sheet_name='lsu_equivalent_GLE')

    # Converting into lsu
    df_liv_population = pd.merge(df_liv_population, df_lsu, on='Item')
    df_liv_population['Value'] = df_liv_population['Value'] * df_liv_population['lsu']
    df_liv_population = df_liv_population.drop(columns=['lsu'])

    # Converting into lsu (other animals)
    df_liv_population_others = pd.merge(df_liv_population_others, df_lsu, on='Item')
    df_liv_population_others['Value'] = df_liv_population_others['Value'] * df_liv_population_others['lsu']
    df_liv_population_others = df_liv_population_others.drop(columns=['lsu'])

    # Aggregating for other animals
    df_liv_population_others = df_liv_population_others.groupby(['Area', 'Element', 'Year'], as_index=False)[
        'Value'].sum()
    # Prepend "Others" to each value in the 'Element' column
    df_liv_population_others['Element'] = df_liv_population_others['Element'].apply(lambda x: f"Others {x}")
    # Rename column
    df_liv_population_others.rename(
        columns={'Element': 'Item'}, inplace=True)

    # Converting into lsu (poultry)
    df_liv_population_poultry = pd.merge(df_liv_population_poultry, df_lsu, on='Item')
    df_liv_population_poultry['Value'] = df_liv_population_poultry['Value'] * df_liv_population_poultry['lsu']
    df_liv_population_poultry = df_liv_population_poultry.drop(columns=['lsu'])

    # Aggregating for poultry
    df_liv_population_poultry = df_liv_population_poultry.groupby(['Area', 'Element', 'Year'], as_index=False)[
        'Value'].sum()
    # Prepend "Poultry" to each value in the 'Element' column
    df_liv_population_poultry['Element'] = df_liv_population_poultry['Element'].apply(lambda x: f"Poultry {x}")
    # Rename column
    df_liv_population_poultry.rename(
        columns={'Element': 'Item'}, inplace=True)

    # Concatenating
    df_liv_population = pd.concat([df_liv_population, df_liv_population_others])
    df_liv_population = pd.concat([df_liv_population, df_liv_population_poultry])

    # Creating a copy for Livestock workflow
    df_liv_pop = df_liv_population.copy()

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------
    # Food item name matching with dictionary
    # Read excel file
    df_dict_calibration = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='calibration')

    # Merge based on 'Item'
    df_liv_population_calibration = pd.merge(df_dict_calibration, df_liv_population, on='Item')

    # Drop the 'Item' column
    df_liv_population_calibration = df_liv_population_calibration.drop(columns=['Item'])

    # Renaming existing columns (geoscale, timsecale, value)
    df_liv_population_calibration.rename(
        columns={'Area': 'geoscale', 'Year': 'timescale', 'Value': 'value'},
        inplace=True)

    # ----------------------------------------------------------------------------------------------------------------------
    # DOMESTIC PRODUCTION (CROP & LIVESTOCK PRODUCTS) ----------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    # Read data ------------------------------------------------------------------------------------------------------------

    # Common for all
    # List of countries

    # FOOD BALANCE SHEETS (FBS) - -------------------------------------------------
    # List of elements
    list_elements = ['Production Quantity', 'Losses']

    list_items = ['Cereals - Excluding Beer + (Total)', 'Fruits - Excluding Wine + (Total)', 'Oilcrops + (Total)',
                  'Pulses + (Total)', 'Rice (Milled Equivalent)',
                  'Starchy Roots + (Total)', 'Sugar Crops + (Total)', 'Vegetables + (Total)',
                  'Milk - Excluding Butter + (Total)', 'Eggs + (Total)',
                  'Bovine Meat', 'Meat, Other', 'Pigmeat',
                  'Poultry Meat', 'Mutton & Goat Meat']

    # 1990 - 2013
    ld = faostat.list_datasets()
    code = 'FBSH'
    pars = faostat.list_pars(code)
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
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
    df_domestic_supply_1990_2013 = faostat.get_data_df(code, pars=my_pars, strval=False)

    # 2010-2022
    list_items = ['Cereals - Excluding Beer + (Total)', 'Fruits - Excluding Wine + (Total)', 'Oilcrops + (Total)',
                  'Pulses + (Total)', 'Rice and products',
                  'Starchy Roots + (Total)', 'Sugar Crops + (Total)', 'Vegetables + (Total)',
                  'Milk - Excluding Butter + (Total)', 'Eggs + (Total)',
                  'Bovine Meat', 'Meat, Other', 'Pigmeat',
                  'Poultry Meat', 'Mutton & Goat Meat']
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
    df_domestic_supply_2010_2022 = faostat.get_data_df(code, pars=my_pars, strval=False)

    # Renaming the items for name matching
    df_domestic_supply_1990_2013.loc[
        df_domestic_supply_1990_2013['Item'].str.contains('Rice \(Milled Equivalent\)', case=False,
                                               na=False), 'Item'] = 'Rice and products'

    # Concatenating all the years together
    df_domestic_supply = pd.concat([df_domestic_supply_1990_2013, df_domestic_supply_2010_2022])

    # Filtering to keep wanted columns
    columns_to_filter = ['Area', 'Element', 'Item', 'Year', 'Value']
    df_domestic_supply = df_domestic_supply[columns_to_filter]

    # Pivot the df
    pivot_df_domestic_supply = df_domestic_supply.pivot_table(index=['Area', 'Year', 'Item'], columns='Element',
                                        values='Value').reset_index()

    # Unit conversion [kt] => [t]
    pivot_df_domestic_supply['Production [t]'] = 1000 * pivot_df_domestic_supply['Production']

    # Unit conversion [t] => [kcal]
    # Read excel
    df_kcal_t = pd.read_excel(
        'dictionaries/kcal_to_t.xlsx',
        sheet_name='kcal_per_100g')
    df_kcal_t = df_kcal_t[['Item', 'kcal per t']]
    # Merge
    merged_df = pd.merge(
        df_kcal_t,
        pivot_df_domestic_supply,  # Only keep the needed columns
        on=['Item']
    )
    # Operation
    merged_df['Production [kcal]'] = merged_df['Production [t]'] * merged_df['kcal per t']
    pivot_df_domestic_supply = merged_df[['Area', 'Year', 'Item', 'Production [kcal]']]
    pivot_df_domestic_supply = pivot_df_domestic_supply.copy()

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------
    # Food item name matching with dictionary
    # Read excel file
    df_dict_calibration = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='calibration')

    # Prepend "Diet" to each value in the 'Item' column
    pivot_df_domestic_supply['Item'] = pivot_df_domestic_supply['Item'].apply(lambda x: f"Production {x}")

    # Renaming existing columns (geoscale, timsecale, value)
    pivot_df_domestic_supply.rename(
        columns={'Area': 'geoscale', 'Year': 'timescale', 'Production [kcal]': 'value'},
        inplace=True)

    # Concat with energy demand
    df_supply_and_energy = pd.concat([pivot_df_domestic_supply, df_energy_demand_cal])

    # Merge based on 'Item'
    df_domestic_supply_calibration = pd.merge(df_dict_calibration, df_supply_and_energy, on='Item')

    # Drop the 'Item' column
    df_domestic_supply_calibration = df_domestic_supply_calibration.drop(columns=['Item'])

    return df_domestic_supply_calibration, df_liv_population_calibration, df_liv_pop



# CalculationLeaf CAL - LIVESTOCK MANURE -----------------------------------------------------------------------------------

def manure_calibration(list_countries):
    # ----------------------------------------------------------------------------------------------------------------------
    # MANURE EMISSIONS ---------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    # Read data ------------------------------------------------------------------------------------------------------------

    # Common for all

    # EMISSIONS FROM LIVESTOCK (GLE) - -------------------------------------------------
    # List of elements
    list_elements = ['Enteric fermentation (Emissions CH4)', 'Manure management (Emissions CH4)',
                     'Manure management (Emissions N2O)', 'Manure left on pasture (Emissions N2O)',
                     'Emissions (N2O) (Manure applied)']

    list_items = ['Swine + (Total)','Sheep and Goats + (Total)', 'Cattle, dairy', 'Cattle, non-dairy', 'Chickens, layers']

    list_items_poultry = ['Chickens, broilers', 'Ducks', 'Turkeys']

    list_items_others = ['Asses', 'Buffalo','Camels', 'Horses', 'Llamas', 'Mules and hinnies']
    list_sources = ['FAO TIER 1']

    # 1990 - 2022
    ld = faostat.list_datasets()
    code = 'GLE'
    pars = faostat.list_pars(code)
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    my_sources = [faostat.get_par(code, 'sources')[i] for i in list_sources]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                  '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                  '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items,
        'year': my_years,
        'source': my_sources
    }

    df_liv_emissions = faostat.get_data_df(code, pars=my_pars, strval=False)

    my_items_poultry = [faostat.get_par(code, 'item')[i] for i in list_items_poultry]
    my_pars_poultry = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items_poultry,
        'year': my_years,
        'source': my_sources
    }
    df_liv_emissions_poultry = faostat.get_data_df(code, pars=my_pars_poultry, strval=False)

    my_items_others = [faostat.get_par(code, 'item')[i] for i in list_items_others]
    my_pars_others = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items_others,
        'year': my_years,
        'source': my_sources
    }
    df_liv_emissions_others = faostat.get_data_df(code, pars=my_pars_others, strval=False)

    # Filtering to keep wanted columns
    columns_to_filter = ['Area', 'Element', 'Item', 'Year', 'Value']
    df_liv_emissions = df_liv_emissions[columns_to_filter]
    df_liv_emissions_poultry = df_liv_emissions_poultry[columns_to_filter]
    df_liv_emissions_others = df_liv_emissions_others[columns_to_filter]

    # Creating one column with Item and Element
    df_liv_emissions['Item'] = df_liv_emissions['Item'] + ' ' + df_liv_emissions['Element']
    df_liv_emissions = df_liv_emissions.drop(columns=['Element'])

    # Aggregating for other animals
    df_liv_emissions_others = df_liv_emissions_others.groupby(['Area', 'Element', 'Year'], as_index=False)['Value'].sum()
    # Prepend "Others" to each value in the 'Element' column
    df_liv_emissions_others['Element'] = df_liv_emissions_others['Element'].apply(lambda x: f"Others {x}")
    # Rename column
    df_liv_emissions_others.rename(
        columns={'Element': 'Item'}, inplace=True)

    # Aggregating for poultry
    df_liv_emissions_poultry = df_liv_emissions_poultry.groupby(['Area', 'Element', 'Year'], as_index=False)[
        'Value'].sum()
    # Prepend "Poultry" to each value in the 'Element' column
    df_liv_emissions_poultry['Element'] = df_liv_emissions_poultry['Element'].apply(lambda x: f"Poultry {x}")
    # Rename column
    df_liv_emissions_poultry.rename(
        columns={'Element': 'Item'}, inplace=True)

    # Concatenating
    df_liv_emissions = pd.concat([df_liv_emissions, df_liv_emissions_others])
    df_liv_emissions = pd.concat([df_liv_emissions, df_liv_emissions_poultry])

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------
    # Food item name matching with dictionary
    # Read excel file
    df_dict_calibration = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='calibration')

    # Merge based on 'Item'
    df_liv_emissions_calibration = pd.merge(df_dict_calibration, df_liv_emissions, on='Item')

    # Drop the 'Item' column
    df_liv_emissions_calibration = df_liv_emissions_calibration.drop(columns=['Item'])

    # Renaming existing columns (geoscale, timsecale, value)
    df_liv_emissions_calibration.rename(
        columns={'Area': 'geoscale', 'Year': 'timescale', 'Value': 'value'},
        inplace=True)

    # Add empty rows for enteric poultry and hens eggs = 0
    # Hens egg
    df_to_duplicate = df_liv_emissions_calibration[df_liv_emissions_calibration['variables'] == 'cal_agr_liv_CH4-emission_abp-hens-egg_treated[kt]'].copy()
    # Modify the duplicated rows
    df_to_duplicate['value'] = 0  # Set value to 0
    df_to_duplicate['variables'] = 'cal_agr_liv_CH4-emission_abp-hens-egg_enteric[kt]'  # Rename variable
    # Append the new rows to the original DataFrame
    df_liv_emissions_calibration = pd.concat([df_liv_emissions_calibration, df_to_duplicate], ignore_index=True)
    # Poultry meat
    df_to_duplicate['variables'] = 'cal_agr_liv_CH4-emission_meat-poultry_enteric[kt]'  # Rename variable
    df_liv_emissions_calibration = pd.concat([df_liv_emissions_calibration, df_to_duplicate], ignore_index=True)

    return df_liv_emissions_calibration, df_liv_emissions


# CalculationLeaf CAL - ENERGY & GHG -----------------------------------------------------------------------------------
def energy_ghg_calibration(list_countries, df_CO2_cal, df_liming_urea):
    # ----------------------------------------------------------------------------------------------------------------------
    # TOTAL GHG EMISSIONS ---------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    # Read data ------------------------------------------------------------------------------------------------------------

    # Common for all
    # List of countries

    # EMISSIONS TOTAL (GT) - -------------------------------------------------
    # List of elements
    list_elements = ['Emissions (CH4)', 'Emissions (N2O)', 'Emissions (CO2)']

    list_items = ['-- Emissions on agricultural land + (Total)', 'On-farm energy use', 'Drained organic soils']
    list_sources = ['FAO TIER 1']

    # 1990 - 2022
    ld = faostat.list_datasets()
    code = 'GT'
    pars = faostat.list_pars(code)
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    my_sources = [faostat.get_par(code, 'sources')[i] for i in list_sources]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                  '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                  '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items,
        'year': my_years,
        'source': my_sources
    }
    df_emissions = faostat.get_data_df(code, pars=my_pars, strval=False)

    # Create a column with the Element and Item
    df_emissions['Element'] = df_emissions['Element'] + ' ' + \
                                df_emissions['Item']

    # Filtering to keep wanted columns
    columns_to_filter = ['Area', 'Element', 'Year', 'Value']
    df_emissions = df_emissions[columns_to_filter].copy()

    # 1. Pivot to have Elements as columns for easy calculation
    df_pivot = df_emissions.pivot_table(index=['Area', 'Year'], columns='Element',
                              values='Value')

    # 2. Perform the subtraction
    df_pivot['Emissions (N2O) Emissions on agricultural land - Drained organic soils'] = df_pivot['Emissions (N2O) Emissions on agricultural land'] - \
                         df_pivot['Emissions (N2O) Drained organic soils (N2O)']

    # 1. Reset index to bring Area and Year back as columns
    df_reset = df_pivot.reset_index()

    # 2. Melt to long format
    df_emissions = df_reset.melt(id_vars=['Area', 'Year'], var_name='Element',
                              value_name='Value')

    # Unit conversion [kt] => [t]
    df_emissions['Value'] = df_emissions['Value'] * 10**(3)

    # Rename column
    df_emissions.rename(columns={'Element': 'Item'}, inplace=True)

    # ----------------------------------------------------------------------------------------------------------------------
    # ENERGY DEMAND (electricity, gas, coal, heat) ---------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # Read FAO Values (for Switzerland) --------------------------------------------------------------------------------------------


    # List of elements
    list_elements = ['Energy use in agriculture']

    list_items = ['Natural gas', 'Electricity', 'Coal', 'Heat']

    # 1990 - 2022
    ld = faostat.list_datasets()
    code = 'GN'
    pars = faostat.list_pars(code)
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                  '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                  '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items,
        'year': my_years
    }
    df_energy_fao = faostat.get_data_df(code, pars=my_pars, strval=False)

    # Filtering to keep wanted columns
    columns_to_filter = ['Area', 'Item', 'Year', 'Value']
    df_energy_fao = df_energy_fao[columns_to_filter].copy()

    # Pivot the df
    df_energy_fao = df_energy_fao.pivot_table(index=['Area', 'Year', 'Item'],
                                                  values='Value').reset_index()

    # ----------------------------------------------------------------------------------------------------------------------
    # CO2 EMISSIONS FROM ENERGY USE ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # Format value from UNFCCC
    df_CO2_cal.rename(columns={'CO2 emissions [kt]': 'Value', 'timescale':'Year'},
                               inplace=True)
    df_CO2_cal['Area'] = 'Switzerland'

    # ----------------------------------------------------------------------------------------------------------------------
    # CO2 EMISSIONS TOTAL (ENERGY USE + LIMING + UREA UNFCCC)  -------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    # Rename col
    df_liming_urea = df_liming_urea.rename(columns={'geoscale': 'Area'})

    # Concat
    df_CO2_cal_total = pd.concat([df_CO2_cal, df_liming_urea])

    # Sum by Year & Area
    df_CO2_cal_total = df_CO2_cal_total.groupby(['Area','Year'])['Value'].sum().reset_index()

    # Unit conversion [kt]=>[t]
    df_CO2_cal_total['Value'] = df_CO2_cal_total['Value'] * 10**3

    # Change the item name
    df_CO2_cal_total['Item'] = 'Emissions (CO2) Fuel, liming, urea'

    ''''# Read FAO Values (for Switzerland) --------------------------------------------------------------------------------------------
    # List of elements
    list_elements = ['Emissions (CO2)']

    list_items = ['Total Energy + (Total)']

    # 1990 - 2022
    ld = faostat.list_datasets()
    code = 'GN'
    pars = faostat.list_pars(code)
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                  '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                  '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items,
        'year': my_years
    }
    df_energy_use = faostat.get_data_df(code, pars=my_pars, strval=False)

    # Filtering to keep wanted columns
    columns_to_filter = ['Area', 'Item', 'Year', 'Value']
    df_energy_use = df_energy_use[columns_to_filter].copy()

    # Pivot the df
    df_energy_use = df_energy_use.pivot_table(index=['Area', 'Year', 'Item'],
                                              values='Value').reset_index()'''

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------
    # Food item name matching with dictionary
    # Read excel file
    df_dict_calibration = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='calibration')

    # Concat
    df_emissions = pd.concat([df_emissions, df_energy_fao])
    df_emissions = pd.concat([df_emissions, df_CO2_cal])
    df_emissions = pd.concat([df_emissions, df_CO2_cal_total])

    # Merge based on 'Item'
    df_emissions_calibration = pd.merge(df_dict_calibration, df_emissions, on='Item')

    # Drop the 'Item' column
    df_emissions_calibration = df_emissions_calibration.drop(columns=['Item'])

    # Renaming existing columns (geoscale, timsecale, value)
    df_emissions_calibration.rename(columns={'Area': 'geoscale', 'Year': 'timescale', 'Value': 'value'},
                               inplace=True)

    return df_emissions_calibration


# CalculationLeaf CAL - NITROGEN -----------------------------------------------------------------------------------
def nitrogen_calibration(list_countries):
    # ----------------------------------------------------------------------------------------------------------------------
    # TOTAL GHG EMISSIONS ---------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    # Read data ------------------------------------------------------------------------------------------------------------

    # Common for all

    # EMISSIONS TOTAL (GT) - -------------------------------------------------
    # List of elements
    list_elements = ['Emissions (N2O)']

    list_items = ['Synthetic Fertilizers']

    # 1990 - 2022
    ld = faostat.list_datasets()
    code = 'GT'
    pars = faostat.list_pars(code)
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                  '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                  '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items,
        'year': my_years
    }
    df_nitrogen = faostat.get_data_df(code, pars=my_pars, strval=False)

    # Filtering to keep wanted columns
    columns_to_filter = ['Area', 'Element', 'Year', 'Value']
    df_nitrogen = df_nitrogen[columns_to_filter]

    # Pivot the df
    df_nitrogen = df_nitrogen.pivot_table(index=['Area', 'Year', 'Element'],
                                          values='Value').reset_index()

    # Unit conversion [kt] => [Mt]
    df_nitrogen['Value'] = df_nitrogen['Value'] * 10**(-3)

    # Rename column
    df_nitrogen.rename(columns={'Element': 'Item'}, inplace=True)

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------
    # Food item name matching with dictionary
    # Read excel file
    df_dict_calibration = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='calibration')

    # Prepend "Fertilizers" to each value in the 'Item' column
    df_nitrogen['Item'] = df_nitrogen['Item'].apply(lambda x: f"Fertilizers {x}")

    # Merge based on 'Item'
    df_nitrogen_calibration = pd.merge(df_dict_calibration, df_nitrogen, on='Item')

    # Drop the 'Item' column
    df_nitrogen_calibration = df_nitrogen_calibration.drop(columns=['Item'])

    # Renaming existing columns (geoscale, timsecale, value)
    df_nitrogen_calibration.rename(columns={'Area': 'geoscale', 'Year': 'timescale', 'Value':'value'},
                               inplace=True)

    return df_nitrogen_calibration

# CalculationLeaf CAL - FEED DEMAND ----------------------------------------------------------------------------------

def feed_calibration(list_countries):
    # ----------------------------------------------------------------------------------------------------------------------
    # HERE! FEED DEMAND PART I --------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # Read data ------------------------------------------------------------------------------------------------------------

    # FOOD BALANCE SHEETS (FBS) - -------------------------------------------------
    # List of elements
    list_elements = ['Feed']

    list_items = ['Cereals - Excluding Beer + (Total)', 'Fruits - Excluding Wine + (Total)', 'Oilcrops + (Total)',
                  'Pulses + (Total)', 'Rice (Milled Equivalent)',
                  'Starchy Roots + (Total)', 'Sugar Crops + (Total)', 'Vegetables + (Total)',
                  'Fish, Seafood + (Total)', 'Animal Products + (Total)', 'Vegetable Oils + (Total)',
                  'Sugar & Sweeteners + (Total)']

    # 1990 - 2013
    ld = faostat.list_datasets()
    code = 'FBSH'
    pars = faostat.list_pars(code)
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
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
    df_feed_1990_2013 = faostat.get_data_df(code, pars=my_pars, strval=False)

    # 2010-2022
    list_items = ['Cereals - Excluding Beer + (Total)', 'Fruits - Excluding Wine + (Total)', 'Oilcrops + (Total)',
                  'Pulses + (Total)', 'Rice and products',
                  'Starchy Roots + (Total)', 'Sugar Crops + (Total)', 'Vegetables + (Total)',
                  'Fish, Seafood + (Total)', 'Animal Products + (Total)', 'Vegetable Oils + (Total)',
                  'Sugar & Sweeteners + (Total)']
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
    df_feed_2010_2022 = faostat.get_data_df(code, pars=my_pars, strval=False)

    # Renaming the items for name matching
    df_feed_1990_2013.loc[
        df_feed_1990_2013['Item'].str.contains('Rice \(Milled Equivalent\)', case=False,
                                                          na=False), 'Item'] = 'Rice and products'

    # Concatenating all the years together
    df_feed = pd.concat([df_feed_1990_2013, df_feed_2010_2022])



    # ----------------------------------------------------------------------------------------------------------------------
    # FEED DEMAND PART II (molasse & cake) --------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # COMMODITY BALANCES (NON-FOOD) (OLD METHODOLOGY) - For molasse and cakes ----------------------------------------------
    # 1990 - 2013
    list_elements = ['Feed']
    list_items = ['Copra Cake', 'Cottonseed Cake', 'Groundnut Cake', 'Oilseed Cakes, Other', 'Palmkernel Cake',
                  'Rape and Mustard Cake', 'Sesameseed Cake', 'Soyabean Cake', 'Sunflowerseed Cake']
    code = 'CBH'
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
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
    df_feed_1990_2013_cake = faostat.get_data_df(code, pars=my_pars, strval=False)

    # SUPPLY UTILIZATION ACCOUNTS (SCl) - For molasse and cakes ----------------------------------------------------------
    # 2010 - 2022
    list_elements = ['Feed']
    list_items = ['Molasses', 'Cake of  linseed', 'Cake of  soya beans', 'Cake of copra', 'Cake of cottonseed',
                  'Cake of groundnuts', 'Cake of hempseed', 'Cake of kapok', 'Cake of maize', 'Cake of mustard seed',
                  'Cake of palm kernel', 'Cake of rapeseed', 'Cake of rice bran', 'Cake of safflowerseed',
                  'Cake of sesame seed', 'Cake of sunflower seed', 'Cake, oilseeds nes', 'Cake, poppy seed',
                  'Cocoa powder and cake']
    code = 'SCL'
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items,
        'year': my_years
    }
    df_feed_2010_2021_molasse_cake = faostat.get_data_df(code, pars=my_pars, strval=False)

    # Aggregating cakes
    df_feed_cake = pd.concat([df_feed_1990_2013_cake, df_feed_2010_2021_molasse_cake])
    # Filtering
    filtered_df = df_feed_cake[df_feed_cake['Item'].str.contains('cake', case=False)]
    # Groupby Area, Year and Element and sum the Value
    grouped_df = filtered_df.groupby(['Area', 'Element', 'Year'])['Value'].sum().reset_index()
    # Unit conversion [t] => [kt]
    grouped_df['Value'] = grouped_df['Value'] / 1000
    # Adding a column 'Item' containing 'Cakes' for all row, before the 'Value' column
    grouped_df['Item'] = 'Cakes'
    cols = grouped_df.columns.tolist()
    cols.insert(cols.index('Value'), cols.pop(cols.index('Item')))
    df_feed_cake = grouped_df[cols]

    # Filtering for molasse
    df_feed_molasses = df_feed_2010_2021_molasse_cake[
        df_feed_2010_2021_molasse_cake['Item'].str.contains('Molasses', case=False)]
    df_feed_molasses = df_feed_molasses.copy()

    # Unit conversion [t] => [kt]
    df_feed_molasses['Value'] = df_feed_molasses['Value'] / 1000

    # Concatenating
    df_feed = pd.concat([df_feed, df_feed_molasses])
    df_feed = pd.concat([df_feed, df_feed_cake])

    # Filtering to keep wanted columns
    columns_to_filter = ['Area', 'Element', 'Item', 'Year', 'Value']
    df_feed = df_feed[columns_to_filter]

    # Pivot the df
    pivot_df_feed = df_feed.pivot_table(index=['Area', 'Year', 'Item'], columns='Element',
                                        values='Value').reset_index()

    # Univ conversion [kt] => [t]
    pivot_df_feed['Feed'] = 1000 * pivot_df_feed['Feed']

    # Univ conversion [kt] => [kcal]
    # Read excel
    """df_kcal_t = pd.read_excel(
        'dictionaries/kcal_to_t.xlsx',
        sheet_name='kcal_per_100g')
    df_kcal_t = df_kcal_t[['Item', 'kcal per t']]
    # Merge
    merged_df = pd.merge(
        df_kcal_t,
        pivot_df_feed,
    )
    # Operation
    merged_df['Feed [kcal]'] = 1000 * merged_df['Feed'] * merged_df['kcal per t']
    pivot_df_feed = merged_df[['Area', 'Year', 'Item', 'Feed [kcal]']]
    pivot_df_feed = pivot_df_feed.copy()"""

    # Adding meat products with 0 everywhere (no meat used as feed from FAOSTAT)
    duplicated_rows = pivot_df_feed[
        pivot_df_feed['Item'] == 'Pulses'].copy()  # Duplicate rows for random item
    duplicated_rows['Item'] = 'Animal Products'  # Change geoscale value to 'EU27' in duplicated rows
    duplicated_rows['Feed'] = 0 # Set the value to 0
    pivot_df_feed = pd.concat([pivot_df_feed, duplicated_rows],
                                   ignore_index=True)  # Append duplicated rows back to the original DataFrame


    # Create a copy for Lever : feed ration
    df_feed_ration = pivot_df_feed.copy()

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------
    # Food item name matching with dictionary
    # Read excel file
    df_dict_calibration = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='calibration')

    # Prepend "Diet" to each value in the 'Item' column
    pivot_df_feed['Item'] = pivot_df_feed['Item'].apply(lambda x: f"Feed {x}")

    # Merge based on 'Item'
    df_feed_calibration = pd.merge(df_dict_calibration, pivot_df_feed, on='Item')

    # Drop the 'Item' column
    df_feed_calibration = df_feed_calibration.drop(columns=['Item'])

    # Renaming existing columns (geoscale, timesecale, value)
    df_feed_calibration.rename(
        columns={'Area': 'geoscale', 'Year': 'timescale', 'Feed': 'value'},
        inplace=True)

    return df_feed_calibration, df_feed_ration

# CalculationLeaf CAL - LAND -----------------------------------------------------------------------------------

def land_calibration(list_countries):
    # Read FAO Values (for Switzerland) --------------------------------------------------------------------------------------------

    # List of elements
    list_elements = ['Area']

    list_items = ['-- Cropland','--- Temporary crops', '--- Temporary fallow', '-- Permanent meadows and pastures']

    # 1990 - 2022
    ld = faostat.list_datasets()
    code = 'RL'
    pars = faostat.list_pars(code)
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                  '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                  '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items,
        'year': my_years
    }
    df_land_use_fao = faostat.get_data_df(code, pars=my_pars, strval=False)

    # Filtering to keep wanted columns
    columns_to_filter = ['Area', 'Item', 'Year', 'Value']
    df_land_use_fao = df_land_use_fao[columns_to_filter]

    # Pivot the df
    df_land_use_fao = df_land_use_fao.pivot_table(index=['Area', 'Year', 'Item'],
                                          values='Value').reset_index()

    # Unit conversion [k ha] => [ha]
    df_land_use_fao['Value'] = df_land_use_fao['Value'] * 1000

    # Filter for Cropland for density lsu
    df_cropland_density = df_land_use_fao[df_land_use_fao['Item'].isin(['Cropland', 'Permanent meadows and pastures'])]
    df_cropland_density = df_cropland_density.pivot_table(
      index=['Area', 'Year'],
      columns='Item',
      values='Value'
    ).reset_index()

    # Create a copy with only agricultural land
    df_agri_land = df_land_use_fao.copy()
    df_agri_land = df_agri_land.pivot_table(
      index=['Area', 'Year'],
      columns='Item',
      values='Value'
    ).reset_index()
    df_agri_land['Agricultural land [ha]'] = df_agri_land['Cropland'] + df_agri_land['Permanent meadows and pastures']
    df_agri_land = df_agri_land[['Area', 'Year', 'Agricultural land [ha]']]

    # Cropland = temporary crops + temporary fallow
    # 1. Filter only the rows for the items of interest
    df_crop_fallow = df_land_use_fao[df_land_use_fao['Item'].isin(['Temporary crops', 'Temporary fallow'])]

    # 2. Group by Area and Year, then sum the values
    df_cropland = (
        df_crop_fallow
        .groupby(['Area', 'Year'], as_index=False)['Value']
        .sum()
    )
    # 3. Assign the new item name
    df_cropland['Item'] = 'Temporary crops and fallow'
    # 4. Optional: Remove original items and append the new one
    df_cleaned = df_land_use_fao[~df_land_use_fao['Item'].isin(['Temporary crops', 'Temporary fallow'])]
    df_land_use_fao= pd.concat([df_cleaned, df_cropland], ignore_index=True)

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------
    # Food item name matching with dictionary
    # Read excel file
    df_dict_calibration = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='calibration')

    # Prepend "Land" to each value in the 'Item' column
    df_land_use_fao['Item'] = df_land_use_fao['Item'].apply(lambda x: f"Land {x}")

    # Merge based on 'Item'
    df_land_use_fao_calibration = pd.merge(df_dict_calibration, df_land_use_fao, on='Item')

    # Drop the 'Item' column
    df_land_use_fao_calibration = df_land_use_fao_calibration.drop(columns=['Item'])

    # Renaming existing columns (geoscale, timsecale, value)
    df_land_use_fao_calibration.rename(columns={'Area': 'geoscale', 'Year': 'timescale', 'Value':'value'},
                                   inplace=True)

    return df_land_use_fao_calibration, df_cropland_density, df_agri_land

# CalculationLeaf CAL - CROPLAND -----------------------------------------------------------------------------------
def cropland_calibration(list_countries):
    # ----------------------------------------------------------------------------------------------------------------------
    # CROPLAND ----------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # CROPS  (QCL) (for everything except lgn-energycrop, gas-energycrop, algae and insect)
    try:
        df_cropland_1990_2022 = pd.read_csv(file_dict['cropland'])
    except OSError:
        # List of elements
        list_elements = ['Area harvested']

        list_items = ['Cereals, primary + (Total)', 'Fibre Crops, Fibre Equivalent + (Total)', 'Fruit Primary + (Total)',
                      'Citrus Fruit + (Total)',
                      'Oilcrops, Oil Equivalent + (Total)', 'Pulses, Total + (Total)', 'Rice',
                      'Roots and Tubers, Total + (Total)',
                      'Sugar Crops Primary + (Total)', 'Vegetables Primary + (Total)']

        # 1990 - 2022
        code = 'QCL'
        my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
        my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
        my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
        list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                      '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                      '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
        my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

        my_pars = {
            'area': my_countries,
            'element': my_elements,
            'item': my_items,
            'year': my_years
        }
        df_cropland_1990_2022 = faostat.get_data_df(code, pars=my_pars, strval=False)
        df_cropland_1990_2022.loc[
            df_cropland_1990_2022['Item'].str.contains('Rice', case=False,
                                                    na=False), 'Item'] = 'Rice and products'
        df_cropland_1990_2022.to_csv(file_dict['cropland'], index=False)

    # Filter columns
    list_filter = ['Area', 'Item', 'Year', 'Value']
    df_cropland_1990_2022 = df_cropland_1990_2022[list_filter]

    # Prepend "Cropland" to each value in the 'Item' column
    df_cropland_1990_2022['Item'] = df_cropland_1990_2022['Item'].apply(lambda x: f"Cropland {x}")

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------

    # Food item name matching with dictionary
    # Read excel file
    df_dict_csc = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='calibration')

    # Merge based on 'Item'
    df_cropland_pathwaycalc = pd.merge(df_dict_csc, df_cropland_1990_2022, on='Item')

    # Drop the 'Item' column
    df_cropland_pathwaycalc = df_cropland_pathwaycalc.drop(columns=['Item'])

    # Renaming existing columns (geoscale, timsecale, value)
    df_cropland_pathwaycalc.rename(columns={'Area': 'geoscale', 'Year': 'timescale', 'Value': 'value'}, inplace=True)

    return df_cropland_pathwaycalc

# CalculationLeaf CAL - LIMING & UREA CO2 EMISSIONS -----------------------------------------------------------------------------------

def CO2_emissions():
    # ----------------------------------------------------------------------------------------------------------------------
    # LIMING & URA CO2 EMISSIONS ----------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # Importing UNFCCC excel files and reading them with a loop (only for Switzerland) Table 4.1 ---------------------------
    # Putting in a df in 3 dimensions (from, to, year)
    # Define the path where the Excel files are located
    folder_path = 'data/data_unfccc_2023'

    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter and sort files by the year (1990 to 2020)
    sorted_files = sorted([f for f in files if f.startswith('CHE_2023_') and int(f.split('_')[2]) in range(1990, 2021)],
                          key=lambda x: int(x.split('_')[2]))

    # Initialize a list to store DataFrames
    data_frames = []

    # Loop through sorted files, read the required rows, and append to the list
    for file in sorted_files:
        # Extract the year from the filename
        year = int(file.split('_')[2])

        # Full path to the file
        file_path = os.path.join(folder_path, file)

        # Read the specific rows and sheet from the Excel file
        df = pd.read_excel(file_path, sheet_name='Table3s2', skiprows=10, nrows=2, header=None)

        # Add a column for the year to the DataFrame
        df['Year'] = year

        # Append to the list of DataFrames
        data_frames.append(df)

    # Combine all DataFrames into a single DataFrame with a multi-index
    combined_df = pd.concat(data_frames, axis=0).set_index(['Year'])

    # Filter the second and third columns (index 1 and 2)
    df_liming_urea = combined_df.iloc[:, [0, 1]]

    # Rename the columns to 'Year' and 'Item'
    df_liming_urea.columns = ['Item', 'Value']

    # Reset the index and rename it to 'Year'
    df_liming_urea = df_liming_urea.reset_index()
    df_liming_urea.rename(columns={'index': 'Year'}, inplace=True)



    # PathwayCalc formatting -----------------------------------------------------------------------------------------------
    # Food item name matching with dictionary
    # Read excel file
    df_dict_calibration = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='calibration')

    # Adding a column geoscale
    df_liming_urea['geoscale'] = 'Switzerland'

    # Prepend "Diet" to each value in the 'Item' column
    #df_liming_urea['Item'] = df_liming_urea['Item'].apply(lambda x: f"Diet {x}")

    # Merge based on 'Item'
    df_liming_urea_calibration = pd.merge(df_dict_calibration, df_liming_urea, on='Item')

    # Drop the 'Item' column
    df_liming_urea_calibration = df_liming_urea_calibration.drop(columns=['Item'])

    # Renaming existing columns (geoscale, timsecale, value)
    df_liming_urea_calibration.rename(
        columns={'Area': 'geoscale', 'Year': 'timescale', 'Value': 'value'}, inplace=True)

    return df_liming_urea_calibration, df_liming_urea


# CalculationLeaf CAL - WOOD ------------------------------

def wood_calibration(list_countries):
    # ----------------------------------------------------------------------------------------------------------------------
    # WOOD DEMAND ---------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # Read FAO Values (for Switzerland) --------------------------------------------------------------------------------------------

    # List of elements
    list_elements = ['Production Quantity']

    list_items = ['Wood fuel + (Total)', 'Industrial roundwood + (Total)',
                  'Pulpwood, round and split (production) + (Total)', 'Sawlogs and veneer logs + (Total)']

    # 1990 - 2022
    ld = faostat.list_datasets()
    code = 'FO'
    pars = faostat.list_pars(code)
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                  '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                  '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items,
        'year': my_years
    }
    df_wood = faostat.get_data_df(code, pars=my_pars, strval=False)

    # 1990 - 2022 ROUNDWOOD
    list_items = ['Roundwood + (Total)']
    ld = faostat.list_datasets()
    code = 'FO'
    pars = faostat.list_pars(code)
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                  '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                  '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items,
        'year': my_years
    }
    df_roundwood = faostat.get_data_df(code, pars=my_pars, strval=False)

    # Concatenating dfs
    df_wood = pd.concat([df_wood, df_roundwood], axis=0)

    # Filtering to keep wanted columns
    columns_to_filter = ['Area', 'Item', 'Year', 'Value']
    df_wood = df_wood[columns_to_filter]

    # Pivot the df
    df_wood = df_wood.pivot_table(index=['Area', 'Year', 'Item'],
                                              values='Value').reset_index()

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------
    # Food item name matching with dictionary
    # Read excel file
    df_dict_calibration = pd.read_excel(
        'dictionaries/dictionnary_agriculture_landuse.xlsx',
        sheet_name='calibration')

    # Merge based on 'Item'
    df_wood_calibration = pd.merge(df_dict_calibration, df_wood, on='Item')

    # Drop the 'Item' column
    df_wood_calibration = df_wood_calibration.drop(columns=['Item'])

    # Renaming existing columns (geoscale, timsecale, value)
    df_wood_calibration.rename(columns={'Area': 'geoscale', 'Year': 'timescale', 'Value': 'value'},
                                    inplace=True)

    return df_wood_calibration



# CalculationLeaf CALIBRATION FORMATTING
def calibration_formatting(df_diet_calibration, df_domestic_supply_calibration, df_liv_population_calibration,
                     df_nitrogen_calibration, df_liv_emissions_calibration, df_feed_calibration,
                     df_land_use_fao_calibration, df_liming_urea_calibration, df_wood_calibration,
                     df_emissions_calibration, df_cropland_fao_calibration):

    # AGRICULTURE MODULE -----------------------------------------------------------------------------------------------

    # Concatenate dfs
    df_calibration = pd.concat([df_diet_calibration, df_domestic_supply_calibration], axis=0)
    df_calibration = pd.concat([df_calibration, df_liv_population_calibration], axis=0)
    df_calibration = pd.concat([df_calibration, df_nitrogen_calibration], axis=0)
    df_calibration = pd.concat([df_calibration, df_liv_emissions_calibration], axis=0)
    df_calibration = pd.concat([df_calibration, df_feed_calibration], axis=0)
    df_calibration = pd.concat([df_calibration, df_land_use_fao_calibration], axis=0)
    df_calibration = pd.concat([df_calibration, df_liming_urea_calibration], axis=0)
    #df_calibration = pd.concat([df_calibration, df_wood_calibration], axis=0)
    df_calibration = pd.concat([df_calibration, df_emissions_calibration], axis=0)
    df_calibration = pd.concat([df_calibration, df_cropland_fao_calibration], axis=0)

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

    # Exporting to csv
    df_calibration_ext_agr.to_csv('agriculture_calibration.csv', index=False)

    # LANDUSE MODULE ---------------------------------------------------------------------------------------------------
    # Concatenate dfs
    df_calibration = df_wood_calibration

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_calibration['module'] = 'land-use'
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

    # Extrapolation for missing data
    df_calibration_struct = ensure_structure(df_calibration)
    df_calibration_ext = linear_fitting_ots_db(df_calibration_struct, years_ots,
                                                countries='all')
    df_calibration_ext_landuse = df_calibration_ext[df_calibration_ext["timescale"] >= 1990]

    # Exporting to csv
    df_calibration_ext_landuse.to_csv('land-use_calibration.csv', index=False)


    return df_calibration_ext_agr

# CalculationLeaf CNST  ------------------------------

def constant():
    # ----------------------------------------------------------------------------------------------------------------------
    # EMISSION FACTOR [CO2/ktoe] ---------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # Read FAO Values (for Switzerland) --------------------------------------------------------------------------------------------

    list_countries = ['Switzerland']
    list_elements = ['Energy use in agriculture', 'Emissions (CO2)']
    list_items = ['Total Energy > (List)']

    # 1990 - 2022
    ld = faostat.list_datasets()
    code = 'GN'
    pars = faostat.list_pars(code)
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['2022']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
        'area': my_countries,
        'element': my_elements,
        'item': my_items,
        'year': my_years
    }
    df_ef = faostat.get_data_df(code, pars=my_pars, strval=False)

    # Filter
    df_ef = df_ef[['Item','Element', 'Value']]

    # Pivot
    df_ef = df_ef.pivot_table(index=['Item'],
                              columns='Element',
                              values='Value').reset_index()

    # Add conversion factor from [TJ] to [ktoe]
    tj_to_ktoe = 0.02388458966275  # source https://www.unitjuggler.com/convertir-energy-de-TJ-en-ktoe.html
    df_ef['TJ to ktoe'] = tj_to_ktoe

    # Compute emission factor
    df_ef['EF [MtCO2/ktoe]'] = df_ef['Emissions (CO2)'] * 10**-3 / (df_ef['Energy use in agriculture'] * df_ef['TJ to ktoe'])

    # Filter and format
    df_ef = df_ef[['Item', 'EF [MtCO2/ktoe]']].copy()
    df_ef = df_ef.melt(id_vars=['Item'], var_name='Element',
                                 value_name='Value')
    df_ef['Item'] = df_ef['Item'] + ' ' + df_ef['Element']
    df_ef = df_ef[['Item', 'Value']].copy()

    # Merge with dict

    return df_ef



# CalculationLeaf FXA - MANURE ------------------------------

def manure_fxa(list_countries, df_liv_emissions, df_manure_n_fxa, df_manure_ch4_fxa):

   # N2O EMISSIONS -------------------------------------------------------------
   # Filter & Rename
   df_manure_n_fxa = df_manure_n_fxa[['Area', 'Year', 'Aggregation','Manure left on pasture (N content)',
                     'Manure applied to soils (N content)', 'Losses from manure treated (N content)']]
   df_manure_n_fxa.rename(columns={'Manure left on pasture (N content)':'N2O Pasture',
                                   'Manure applied to soils (N content)':'N2O Applied',
                                   'Losses from manure treated (N content)':'N2O Treated'},
                          inplace=True)

   # Melt df
   df_melted = pd.melt(df_manure_n_fxa, id_vars=['Area', 'Year', 'Aggregation'],
                       value_vars=['N2O Pasture', 'N2O Applied',
                                   'N2O Treated'],
                       var_name='Item', value_name='value N')

   # Concatenate the aggregation column with the manure column names
   df_melted['Item'] = df_melted['Aggregation'] + ' ' + df_melted['Item']

   # Rename cols
   # Rename for merge (df_liv_pop => pivot_df_slau (meat) or df_slau_eggs_milk (eggs,dairy))
   terms = {
     'Cattle, dairy': 'Dairy-milk',
     'Cattle, non-dairy': 'Bovine',
     'Chickens, layers': 'Hens-egg',
     'Sheep and Goats': 'Sheep',
     'Swine': 'Pig',
     'Others': 'Other animal',
     'Poultry Stocks': 'Poultry',
     'Manure management (Emissions N2O)': 'N2O Treated',
     'Manure left on pasture (Emissions N2O)': 'N2O Pasture',
     'Emissions (N2O) (Manure applied)': 'N2O Applied'
   }
   def replace_partial(text):
     for key, value in terms.items():
       if key in text:
         text = text.replace(key, value)
     return text
   df_liv_emissions['Item'] = df_liv_emissions['Item'].apply(replace_partial)

   # Merge with NO2 emission df_liv_emissions_calibration
   df_manure_fxa = df_melted.merge(df_liv_emissions, on=['Area', 'Year', 'Item'], how='inner')

   # Compute emission factor per practice : EF = Emissions NO2 [kt] / Manure applied-treated-pasture [kg N]
   df_manure_fxa['value'] = df_manure_fxa['Value'] * 10**6 / df_manure_fxa['value N']
   df_manure_fxa = df_manure_fxa[['Area', 'Year', 'Item', 'value']]

   # Fill na with 0
   df_manure_fxa['value'].fillna(0.0, inplace=True)

   # CH4 EMISSIONS -------------------------------------------------------------
   # Format
   df_manure_ch4_fxa.rename(
     columns={'Manure emissions CH4 [t/lsu]': 'value',
              'Aggregation': 'Item'},
     inplace=True)
   df_manure_ch4_fxa['Item'] = df_manure_ch4_fxa['Item'].apply(lambda x: f"CH4 Treated {x}")

   # Concat
   df_manure_fxa = pd.concat([df_manure_fxa, df_manure_ch4_fxa],
                              axis=0)

   # PathwayCalc formatting ------------------------------------------------------------------
   # Food item name matching with dictionary
   # Read excel file
   df_dict_csl = pd.read_excel(
     'dictionaries/dictionnary_agriculture_landuse.xlsx',
     sheet_name='climate-smart-livestock')

   # Merge based on 'Item'
   df_manure_fxa = pd.merge(df_dict_csl, df_manure_fxa, on='Item')

   # Drop the 'Item' column
   df_manure_fxa = df_manure_fxa.drop(columns=['Item'])

   # Renaming existing columns (geoscale, timsecale, value)
   df_manure_fxa.rename(columns={'Area': 'geoscale', 'Year': 'timescale'},
                              inplace=True)

   # Adding the columns module, lever, level and string-pivot at the correct places
   df_manure_fxa['module'] = 'agriculture'
   df_manure_fxa['lever'] = 'climate-smart-livestock'
   df_manure_fxa['level'] = 0
   cols = df_manure_fxa.columns.tolist()
   cols.insert(cols.index('value'), cols.pop(cols.index('module')))
   cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
   cols.insert(cols.index('value'), cols.pop(cols.index('level')))
   df_manure_fxa = df_manure_fxa[cols]

   # Rename countries to Pathaywcalc name
   df_manure_fxa['geoscale'] = df_manure_fxa['geoscale'].replace(
     'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
   df_manure_fxa['geoscale'] = df_manure_fxa['geoscale'].replace(
     'Netherlands (Kingdom of the)',
     'Netherlands')
   df_manure_fxa['geoscale'] = df_manure_fxa['geoscale'].replace(
     'Czechia', 'Czech Republic')

   # Extrapolating
   df_manure_fxa = ensure_structure(df_manure_fxa)
   df_manure_fxa = linear_fitting_ots_db(df_manure_fxa, years_ots,
                                               countries='all')

   return df_manure_fxa


# CalculationLeaf FXA FORMATTING
def fxa_preprocessing():

    # Load FXA data
    df_fxa = pd.read_csv(
        '../../data/csv/agriculture_fixed-assumptions.csv',
        sep=';')

    # Add fxa_ in front of lus_land_total-area[ha]
    df_fxa['eucalc-name'] = df_fxa['eucalc-name'].replace('lus_land_total-area[ha]', 'fxa_lus_land_total-area[ha]')

    # Extract fxa list we need for Agriculture & Land Use
    df_ref = pd.read_excel(
        '/Users/crosnier/Documents/PathwayCalc/_database/agriculture_land-use_references.xlsx',
        sheet_name='references_agriculture')
    df_ref_fxa = df_ref[df_ref['level'] =='fxa'].copy()

    # Reformat according to new format
    df_fxa = df_fxa.drop(columns=['string-pivot', 'type-prefix', 'element', 'item', 'unit', 'reference-id', 'interaction-file', 'module-prefix'])
    df_fxa.rename(columns={'eucalc-name': 'variables'}, inplace=True)

    # Filter those we need for Agriculture & Land Use
    df_fxa_pathwaycalc = df_fxa[df_fxa['variables'].isin(df_ref_fxa['variables'])]

    # Extrapolate
    df_fxa_pathwaycalc = ensure_structure(df_fxa_pathwaycalc)
    df_fxa_pathwaycalc = linear_fitting_ots_db(df_fxa_pathwaycalc, years_ots, countries='all')

    # Export as csv
    df_fxa_pathwaycalc.to_csv('../../data/csv/agriculture_fixed-assumptions_pathwaycalc.csv', sep=';', index=False)

    return


# CalculationLeaf Pickle creation
#  FIXME only Switzerland for now

def database_from_csv_to_datamatrix(years_ots, years_fts, dm_kcal_req_pathwaycalc, df_csl_fxa, df_manure_fxa, df_calibration, df_feed_lsu_pathwaycalc, df_diet_pathwaycalc, df_ruminant_feed_pathwaycalc):
    #############################################
    ##### database_from_csv_to_datamatrix() #####
    #############################################
     # make list with years from 2020 to 2050 (steps of 5 years)
    years_all = years_ots + years_fts

    # FixedAssumptionsToDatamatrix

    file = '../../data/datamatrix/agriculture.pickle'
    with open(file, 'rb') as handle:
        DM_agriculture_old = pickle.load(handle)

    for key in DM_agriculture_old['fxa'].keys():
        if 'cal_' in key:
            DM_agriculture_old['fxa'][key].filter({'Years': years_ots})
        else:
            dm = DM_agriculture_old['fxa'][key].copy()
            DM_agriculture_old['fxa'][key] = linear_fitting(dm, years_all)
            DM_agriculture_old['fxa'][key].filter({'Years': years_ots + years_all})

    dm = DM_agriculture_old['fxa']['lus_land_total-area']
    if 'lus_land_total-area' in dm.col_labels['Variables']:
        dm.rename_col('lus_land_total-area', 'fxa_lus_land_total-area', dim='Variables')
    DM_agriculture_old['fxa']['lus_land_total-area'] = dm

    # Include new fxa
    # Manure yield
    df_csl_fxa = ensure_structure(df_csl_fxa)
    df_csl_fxa = linear_fitting_ots_db(df_csl_fxa, years_all,countries='all')
    lever = 'climate-smart-livestock'
    df_ots, df_fts = database_to_df(df_csl_fxa, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm = DataMatrix.create_from_df(df_ots, num_cat=1)
    # Replace for Switzerland
    DM_agriculture_old['fxa']['liv_manure_n-stock']['Switzerland', :, 'fxa_liv_manure_n-stock', :] = dm['Switzerland', :, 'fxa_liv_manure_n-stock', :]

    # Emission factor N2O
    df_manure_fxa = ensure_structure(df_manure_fxa)
    df_manure_fxa = linear_fitting_ots_db(df_manure_fxa, years_all, countries='all')
    df_manure_N2O = df_manure_fxa[df_manure_fxa['variables'].str.contains('fxa_ef_liv_N2O-emission_ef_', case=False)]
    lever = 'climate-smart-livestock'
    df_ots, df_fts = database_to_df(df_manure_N2O, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm = DataMatrix.create_from_df(df_ots, num_cat=2)
    # Replace for Switzerland
    DM_agriculture_old['fxa']['ef_liv_N2O-emission']['Switzerland', :, 'fxa_ef_liv_N2O-emission_ef', :, :] = dm['Switzerland', :, 'fxa_ef_liv_N2O-emission_ef', :, :]

    # Emission factor CH4 treated manure
    df_manure_fxa = ensure_structure(df_manure_fxa)
    df_manure_fxa = linear_fitting_ots_db(df_manure_fxa, years_all, countries='all')
    df_manure_CH4 = df_manure_fxa[
      df_manure_fxa['variables'].str.contains('fxa_ef_liv_CH4-emission_treated', case=False)]
    lever = 'climate-smart-livestock'
    df_ots, df_fts = database_to_df(df_manure_CH4, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm = DataMatrix.create_from_df(df_ots, num_cat=1)
    # Replace for Switzerland
    DM_agriculture_old['fxa']['ef_liv_CH4-emission_treated']['Switzerland', :, 'fxa_ef_liv_CH4-emission_treated', :] = dm['Switzerland', :, 'fxa_ef_liv_CH4-emission_treated', :]

    # Feed
    lever = 'diet'
    df_ots, df_fts = database_to_df(df_feed_lsu_pathwaycalc, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm = DataMatrix.create_from_df(df_ots, num_cat=2)
    DM_agriculture_old['fxa']['feed'] = dm

    # LeversToDatamatrix FTS linear fitting of ots

    DM_ots = DM_agriculture_old['ots'].copy()
    DM_fts = DM_agriculture_old['fts'].copy()

    # To do once when adding a new lever
    #DM_fts['ruminant-feed'] = {'ruminant-feed': dict()}

    # Levers to be normalised
    list_norm = ['climate-smart-livestock_ration']

    for key in DM_ots.keys():
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
              DM_fts[key][subkey][lev] = dm_norm.filter(
                {'Years': years_fts}, inplace=False
              )
            else:
              DM_fts[key][subkey][lev] = dm.filter(
                {'Years': years_fts}, inplace=False
              )
      else:
        dm = DM_ots[key].copy()
        linear_fitting(dm, years_fts)
        for lev in range(1, 5):
          DM_fts[key][lev] = dm.filter({'Years': years_fts}, inplace=False)

    # To remove '_' at the ending of some keys as the ones for diet
    #for key in dm_fts.keys():
    #    if isinstance(dm_fts[key], dict):  # Check if the value is a dictionary
     #       new_dict = {}
     #       for sub_key, value in dm_fts[key].items():
                # Ensure the key is a string before modifying it
    #            new_key = sub_key.rstrip('_') if isinstance(sub_key, str) else sub_key
    #            new_dict[new_key] = value  # Assign value to the new key
    #        dm_fts[key] = new_dict  # Replace the original dictionary


    #file = 'agriculture_fixed-assumptions_pathwaycalc'
    #lever = 'none'
    #edit_database(file, lever, column='eucalc-name', mode='rename',pattern={'meat_': 'meat-', 'abp_': 'abp-'})
    #edit_database(file, lever, column='eucalc-name', mode='rename',pattern={'_rem_': '_', '_to_': '_', 'land-man_ef': 'fxa_land-man_ef'})
    #edit_database(file, lever, column='eucalc-name', mode='rename',pattern={'land-man_soil-type': 'fxa_land-man_soil-type'})
    #edit_database(file, lever, column='eucalc-name', mode='rename',pattern={'_def_': '_def-', '_gstock_': '_gstock-', '_nat-losses_': '_nat-losses-'})
    # AGRICULTURE ------------------------------------------------------------------------------------------------------
    # LIVESTOCK MANURE - N2O emissions
    #df = read_database_fxa(file, filter_dict={'variables': 'fxa_ef_liv_N2O-emission_ef.*'})
    #dm_ef_N2O = DataMatrix.create_from_df(df, num_cat=2)
    #dict_fxa['ef_liv_N2O-emission'] = dm_ef_N2O
    # LIVESTOCK MANURE - CH4 emissions
    """df = read_database_fxa(file, filter_dict={'variables': 'ef_liv_CH4-emission_treated.*'})
    dm_ef_CH4 = DataMatrix.create_from_df(df, num_cat=1)
    dict_fxa['ef_liv_CH4-emission_treated'] = dm_ef_CH4
    # LIVESTOCK MANURE - N stock
    df = read_database_fxa(file, filter_dict={'variables': 'liv_manure_n-stock.*'})
    dm_nstock = DataMatrix.create_from_df(df, num_cat=1)
    dict_fxa['liv_manure_n-stock'] = dm_nstock
    # CROP PRODUCTION - Burnt residues emission
    df = read_database_fxa(file, filter_dict={'variables': 'ef_burnt-residues.*'})
    dm_ef_burnt = DataMatrix.create_from_df(df, num_cat=1)
    dict_fxa['ef_burnt-residues'] = dm_ef_burnt
    # CROP PRODUCTION - Soil residues emission
    df = read_database_fxa(file, filter_dict={'variables': 'ef_soil-residues.*'})
    dm_ef_soil = DataMatrix.create_from_df(df, num_cat=1)
    dict_fxa['ef_soil-residues'] = dm_ef_soil
    # CROP PRODUCTION - Residue yield
    df = read_database_fxa(file, filter_dict={'variables': 'residues_yield.*'})
    dm_residues_yield = DataMatrix.create_from_df(df, num_cat=1)
    dict_fxa['residues_yield'] = dm_residues_yield
    # LAND - Fibers domestic-self-sufficiency
    df = read_database_fxa(file, filter_dict={'variables': 'domestic-self-sufficiency_fibres-plant-eq'})
    dm_fibers = DataMatrix.create_from_df(df, num_cat=0)
    dict_fxa['domestic-self-sufficiency_fibres-plant-eq'] = dm_fibers
    # LAND - Fibers domestic supply quantity
    df = read_database_fxa(file, filter_dict={'variables': 'domestic-supply-quantity_fibres-plant-eq'})
    dm_fibers_sup = DataMatrix.create_from_df(df, num_cat=0)
    dict_fxa['domestic-supply-quantity_fibres-plant-eq'] = dm_fibers_sup
    dm_fibers.append(dm_fibers_sup, dim='Variables')
    # LAND - Emission crop rice
    df = read_database_fxa(file, filter_dict={'variables': 'emission_crop_rice'})
    dm_rice = DataMatrix.create_from_df(df, num_cat=0)
    dict_fxa['emission_crop_rice'] = dm_rice
    # NITROGEN BALANCE - Emission fertilizer
    df = read_database_fxa(file, filter_dict={'variables': 'agr_emission_fertilizer'})
    dm_n_fertilizer = DataMatrix.create_from_df(df, num_cat=0)
    dict_fxa['agr_emission_fertilizer'] = dm_n_fertilizer


    # LAND USE --------------------------------------------------------------------------------------------------------
    # LAND ALLOCATION - Total area
    df = read_database_fxa(file, filter_dict={'variables': 'lus_land_total-area'})
    dm_land_total = DataMatrix.create_from_df(df, num_cat=0)
    dict_fxa['lus_land_total-area'] = dm_land_total
    # CARBON STOCK - c-stock biomass & soil
    df = read_database_fxa(file, filter_dict={'variables': 'land-man_ef'})
    dm_ef_biomass = DataMatrix.create_from_df(df, num_cat=0)
    dict_fxa['land-man_ef'] = dm_ef_biomass
    # CARBON STOCK - soil type
    df = read_database_fxa(file, filter_dict={'variables': 'land-man_soil-type'})
    dm_soil = DataMatrix.create_from_df(df, num_cat=0)
    dict_fxa['land-man_soil-type'] = dm_soil
    # AGROFORESTRY CROP - emission factors
    df = read_database_fxa(file, filter_dict={'variables': 'agr_climate-smart-crop_ef_agroforestry'})
    dm_crop_ef_agroforestry = DataMatrix.create_from_df(df, num_cat=1)
    dict_fxa['agr_climate-smart-crop_ef_agroforestry'] = dm_crop_ef_agroforestry
    # AGROFORESTRY livestock - emission factors
    df = read_database_fxa(file, filter_dict={'variables': 'agr_climate-smart-livestock_ef_agroforestry'})
    dm_livestock_ef_agroforestry = DataMatrix.create_from_df(df, num_cat=1)
    dict_fxa['agr_climate-smart-livestock_ef_agroforestry'] = dm_livestock_ef_agroforestry
    # AGROFORESTRY Forestry - natural losses & others
    df = read_database_fxa(file, filter_dict={'variables': 'agr_climate-smart-forestry'})
    dm_agroforestry = DataMatrix.create_from_df(df, num_cat=1)
    dict_fxa['agr_climate-smart-forestry'] = dm_agroforestry"""

    # file
    __file__ = "agriculture_landuse_preprocessing_EU.py"

    # directories
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # CalibrationDataToDatamatrix

    dict_fxa = {}
    # Data - Calibration
    #file = '../../data/csv/agriculture_calibration.csv'
    lever = 'none'
    #df_db = pd.read_csv(file)
    df_ots, df_fts = database_to_df(df_calibration, lever, level='all')
    df_ots = df_ots.drop(columns=['none']) # Drop column 'none'
    dm_cal = DataMatrix.create_from_df(df_ots, num_cat=0)

    # Data - Fixed assumptions - Calibration factors - Diet
    dm_cal_diet = dm_cal.filter_w_regex({'Variables': 'cal_agr_diet.*'})
    dm_cal_diet.deepen(based_on='Variables')
    DM_agriculture_old['fxa']['cal_agr_diet'] = dm_cal_diet

    # Data - Fixed assumptions - Calibration factors - Food waste
    #dm_cal_food_waste = dm_cal.filter_w_regex({'Variables': 'cal_agr_food-wastes.*'})
    #dm_cal_food_waste.deepen(based_on='Variables')
    #dict_fxa['cal_food_waste'] = dm_cal_food_waste

    # Data - Fixed assumptions - Calibration factors - Livestock domestic production
    dm_cal_liv_dom_prod = dm_cal.filter_w_regex({'Variables': 'cal_agr_domestic-production-liv.*'})
    dm_cal_liv_dom_prod.deepen(based_on='Variables')
    DM_agriculture_old['fxa']['cal_agr_domestic-production-liv'] = dm_cal_liv_dom_prod

    # Data - Fixed assumptions - Calibration factors - Livestock population
    dm_cal_liv_pop = dm_cal.filter_w_regex({'Variables': 'cal_agr_liv-population.*'})
    dm_cal_liv_pop.deepen(based_on='Variables')
    DM_agriculture_old['fxa']['cal_agr_liv-population'] = dm_cal_liv_pop

    # Data - Fixed assumptions - Calibration factors - Livestock CH4 emissions
    dm_cal_liv_CH4 = dm_cal.filter_w_regex({'Variables': 'cal_agr_liv_CH4-emission.*'})
    dm_cal_liv_CH4.deepen(based_on='Variables')
    dm_cal_liv_CH4.deepen(based_on='Variables')
    DM_agriculture_old['fxa']['cal_agr_liv_CH4-emission'] = dm_cal_liv_CH4

    # Data - Fixed assumptions - Calibration factors - Livestock N2O emissions
    dm_cal_liv_N2O = dm_cal.filter_w_regex({'Variables': 'cal_agr_liv_N2O-emission.*'})
    dm_cal_liv_N2O.deepen(based_on='Variables')
    dm_cal_liv_N2O.deepen(based_on='Variables')
    DM_agriculture_old['fxa']['cal_agr_liv_N2O-emission'] = dm_cal_liv_N2O

    # Data - Fixed assumptions - Calibration factors - Feed demand
    dm_cal_feed = dm_cal.filter_w_regex({'Variables': 'cal_agr_demand_feed.*'})
    dm_cal_feed.deepen(based_on='Variables')
    DM_agriculture_old['fxa']['cal_agr_demand_feed'] = dm_cal_feed

    # Data - Fixed assumptions - Calibration factors - Crop production
    dm_cal_crop = dm_cal.filter_w_regex({'Variables': 'cal_agr_domestic-production_food.*'})
    dm_cal_crop.deepen(based_on='Variables')
    DM_agriculture_old['fxa']['cal_agr_domestic-production_food'] = dm_cal_crop

    # Data - Fixed assumptions - Calibration factors - Cropland
    dm_cal_cropland = dm_cal.filter_w_regex({'Variables': 'cal_agr_lus_land_cropland.*'})
    dm_cal_cropland.drop("Variables", ['cal_agr_lus_land_cropland'])# Drop total cropland
    dm_cal_cropland.deepen(based_on='Variables')
    DM_agriculture_old['fxa']['cal_agr_lus_land_cropland'] = dm_cal_cropland

    # Data - Fixed assumptions - Calibration factors - Land
    dm_cal_land = dm_cal.filter({'Variables': ['cal_agr_lus_land_cropland', 'cal_agr_lus_land_grassland']})
    dm_cal_land.deepen(based_on='Variables')
    DM_agriculture_old['fxa']['cal_agr_lus_land'] = dm_cal_land

    # Data - Fixed assumptions - Calibration factors - Nitrogen balance
    dm_cal_n = dm_cal.filter_w_regex({'Variables': 'cal_agr_crop_emission_N2O-emission_fertilizer.*'})
    DM_agriculture_old['fxa']['cal_agr_crop_emission_N2O-emission_fertilizer'] = dm_cal_n

    # Data - Fixed assumptions - Calibration factors - Energy demand for agricultural land
    dm_cal_energy_demand = dm_cal.filter_w_regex({'Variables': 'cal_agr_energy-demand.*'})
    dm_cal_energy_demand.deepen(based_on='Variables')
    DM_agriculture_old['fxa']['cal_agr_energy-demand'] = dm_cal_energy_demand

    # Data - Fixed assumptions - Calibration factors - Agricultural emissions total (CH4, N2O, CO2)
    dm_cal_CH4 = dm_cal.filter_w_regex({'Variables': 'cal_agr_CH4-emission'})
    dm_cal_N2O = dm_cal.filter_w_regex({'Variables': 'cal_agr_N2O-emission'})
    dm_cal_CO2 = dm_cal.filter_w_regex({'Variables': 'cal_agr_CO2-emission'})
    dm_cal_CH4.append(dm_cal_N2O, dim='Variables')
    dm_cal_CH4.append(dm_cal_CO2, dim='Variables')
    DM_agriculture_old['fxa']['cal_agr_emissions'] = dm_cal_CH4

    # Data - Fixed assumptions - Calibration factors - CO2 emissions (fuel, liming, urea)
    dm_cal_input = dm_cal.filter_w_regex({'Variables': 'cal_agr_input-use_emissions-CO2.*'})
    dm_cal_input.deepen(based_on='Variables')
    DM_agriculture_old['fxa']['cal_agr_input-use_emissions-CO2'] = dm_cal_input

    # Create a dictionnary with all the fixed assumptions (only for calibration since the other comes from pickle)
    """dict_fxa = {
        'cal_agr_diet': dm_cal_diet,
        'cal_agr_domestic-production-liv': dm_cal_liv_dom_prod,
        'cal_agr_liv-population': dm_cal_liv_pop,
        'cal_agr_liv_CH4-emission': dm_cal_liv_CH4,
        'cal_agr_liv_N2O-emission': dm_cal_liv_N2O,
        'cal_agr_domestic-production_food': dm_cal_crop,
        'cal_agr_demand_feed': dm_cal_feed,
        'cal_agr_lus_land': dm_cal_land,
        'cal_agr_crop_emission_N2O-emission_fertilizer': dm_cal_n,
        'cal_agr_emission_CH4': dm_cal_CH4,
        'cal_agr_emission_N2O': dm_cal_N2O,
        'cal_agr_emission_CO2': dm_cal_CO2,
        'cal_agr_energy-demand': dm_cal_energy_demand,
        'cal_input': dm_cal_input,
        'ef_liv_N2O-emission': dm_ef_N2O,
        'ef_liv_CH4-emission_treated': dm_ef_CH4,
        'liv_manure_n-stock': dm_nstock,
        'ef_burnt-residues': dm_ef_burnt,
        'ef_soil-residues': dm_ef_soil,
        'residues_yield': dm_residues_yield,
        'fibers': dm_fibers,
        'rice': dm_rice,
        'agr_emission_fertilizer' : dm_n_fertilizer,
        'lus_land_total-area' : dm_land_total,
        'land-man_ef' : dm_ef_biomass,
        'land-man_soil-type' : dm_soil,
        'agr_climate-smart-crop_ef_agroforestry' : dm_crop_ef_agroforestry,
        'agr_climate-smart-livestock_ef_agroforestry': dm_livestock_ef_agroforestry,
        'agr_climate-smart-forestry' : dm_agroforestry
    }"""


    #####################
    ###### LEVERS #######
    #####################
    # LeversToDatamatrix
    dict_ots = {}
    dict_fts = {}

    # [TUTORIAL] Data - Lever - Population
    #file = 'lifestyles_population'  # File name to read
    #lever = 'pop'  # Lever name to match the JSON?

    # Creates the datamatrix for lifestyles population
    #dict_ots, dict_fts = read_database_to_ots_fts_dict_w_groups(file, lever, num_cat_list=[1, 0, 0], baseyear=baseyear,
    #                                                            years=years_all, dict_ots=dict_ots, dict_fts=dict_fts,
    #                                                            column='eucalc-name',
    #                                                            group_list=['lfs_demography_.*',
    #                                                                        'lfs_macro-scenarii_.*',
    #                                                                        'lfs_population_.*'])

    # Data - Lever - Diet
    lever = 'diet'
    df_ots, df_fts = database_to_df(df_diet_pathwaycalc, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm = DataMatrix.create_from_df(df_ots, num_cat=0)
    dict_temp = {}
    dm_temp = dm.filter_w_regex({'Variables': 'lfs_consumers-diet.*'})
    dm_temp.deepen()
    dict_temp['lfs_consumers-diet'] = dm_temp
    dm_temp = dm.filter_w_regex({'Variables': 'share.*'})
    dm_temp.deepen()
    dict_temp['share'] = dm_temp
    dict_ots[lever] = dict_temp

    # Data - Lever - Energy requirements
    lever = 'kcal-req'
    dict_ots[lever] = dm_kcal_req_pathwaycalc

    # Data - Lever - Food wastes
    lever = 'fwaste'
    df_ots, df_fts = database_to_df(df_waste_pathwaycalc, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm = DataMatrix.create_from_df(df_ots, num_cat=1)
    dict_ots[lever] = dm

    # Data - Lever - self-sufficiency
    lever = 'food-net-import'
    df_ots, df_fts = database_to_df(df_ssr_pathwaycalc, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm = DataMatrix.create_from_df(df_ots, num_cat=1)
    dict_ots[lever] = dm

    # Data - Lever - climate smart livestock
    lever = 'climate-smart-livestock'
    df_ots, df_fts = database_to_df(df_climate_smart_livestock_pathwaycalc, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm = DataMatrix.create_from_df(df_ots, num_cat=0)
    dict_temp = {}
    dm_losses = dm.filter_w_regex({'Variables': 'agr_climate-smart-livestock_losses.*'})
    dm_losses.deepen()
    dict_temp['climate-smart-livestock_losses'] = dm_losses
    dm_yield = dm.filter_w_regex({'Variables': 'agr_climate-smart-livestock_yield.*'})
    dm_yield.deepen()
    dict_temp['climate-smart-livestock_yield'] = dm_yield
    dm_slau = dm.filter_w_regex({'Variables': 'agr_climate-smart-livestock_slaughtered.*'})
    dm_slau.deepen()
    dict_temp['climate-smart-livestock_slaughtered'] = dm_slau
    dm_density = dm.filter_w_regex({'Variables': 'agr_climate-smart-livestock_density.*'})
    dict_temp['climate-smart-livestock_density'] = dm_density
    dm_enteric = dm.filter_w_regex({'Variables': 'agr_climate-smart-livestock_enteric.*'})
    dm_enteric.deepen()
    dict_temp['climate-smart-livestock_enteric'] = dm_enteric
    dm_manure = dm.filter_w_regex({'Variables': 'agr_climate-smart-livestock_manure.*'})
    dm_manure.deepen()
    dm_manure.deepen(based_on='Variables')
    dm_manure.switch_categories_order(cat1='Categories2', cat2='Categories1')
    dict_temp['climate-smart-livestock_manure'] = dm_manure
    dm_ration = dm.filter_w_regex({'Variables': 'agr_climate-smart-livestock_ration.*'})
    dm_ration.deepen()
    dm_ration.normalise(dim='Categories1') # normalise to keep sum ration = 1
    dict_temp['climate-smart-livestock_ration'] = dm_ration
    dm_ef = dm.filter_w_regex({'Variables': 'agr_climate-smart-livestock_ef_agroforestry.*'})
    dm_ef.deepen()
    dict_temp['agr_climate-smart-livestock_ef_agroforestry'] = dm_ef
    dict_ots[lever] = dict_temp
    #edit_database(file,lever,column='eucalc-name',pattern={'_CH4-emission':''},mode='rename')
    #edit_database(file,lever,column='eucalc-name',pattern={'ration_crop_':'ration_crop-', 'ration_liv_':'ration_liv-'},mode='rename')
    """dict_ots, dict_fts = read_database_to_ots_fts_dict_w_groups(file, lever, num_cat_list=[1, 1, 1, 0, 1, 2, 1, 1], baseyear=baseyear,
                                                                years=years_all, dict_ots=dict_ots, dict_fts=dict_fts,
                                                                column='eucalc-name',
                                                                group_list=['climate-smart-livestock_losses.*', 'climate-smart-livestock_yield.*',
                                                                            'climate-smart-livestock_slaughtered.*', 'climate-smart-livestock_density',
                                                                            'climate-smart-livestock_enteric.*', 'climate-smart-livestock_manure.*',
                                                                            'climate-smart-livestock_ration.*', 'agr_climate-smart-livestock_ef_agroforestry.*'])
    """

    # Data - Lever - ruminant-feed
    lever = 'ruminant-feed'
    df_ots, df_fts = database_to_df(df_ruminant_feed_pathwaycalc, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm = DataMatrix.create_from_df(df_ots, num_cat=0)
    dict_temp = {}
    dm_temp = dm.filter_w_regex({'Variables': 'agr_ruminant-feed_share-grass.*'})
    dict_temp['ruminant-feed'] = dm_temp
    dict_ots[lever] = dict_temp

    # Data - Lever - biomass hierarchy
    lever = 'biomass-hierarchy'
    df_ots, df_fts = database_to_df(df_biomass_hierarchy_pathwaycalc, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm = DataMatrix.create_from_df(df_ots, num_cat=0)
    dict_temp = {}
    dm_temp = dm.filter_w_regex({'Variables': '.*biomass-hierarchy-bev-ibp-use-oth.*'})
    dm_temp.deepen()
    dict_temp['biomass-hierarchy-bev-ibp-use-oth'] = dm_temp
    dm_temp = dm.filter_w_regex({'Variables': 'agr_biomass-hierarchy_biomass-mix_digestor.*'})
    dm_temp.deepen()
    dict_temp['biomass-hierarchy_biomass-mix_digestor'] = dm_temp
    dm_temp = dm.filter_w_regex({'Variables': 'agr_biomass-hierarchy_biomass-mix_solid.*'})
    dm_temp.deepen()
    dict_temp['biomass-hierarchy_biomass-mix_solid'] = dm_temp
    dm_temp = dm.filter_w_regex({'Variables': 'agr_biomass-hierarchy_biomass-mix_liquid.*'})
    dm_temp.deepen()
    dict_temp['biomass-hierarchy_biomass-mix_liquid'] = dm_temp
    dm_temp = dm.filter_w_regex({'Variables': 'agr_biomass-hierarchy_bioenergy_liquid_biodiesel.*'})
    dm_temp.deepen()
    dict_temp['biomass-hierarchy_bioenergy_liquid_biodiesel'] = dm_temp
    dm_temp = dm.filter_w_regex({'Variables': 'agr_biomass-hierarchy_bioenergy_liquid_biogasoline.*'})
    dm_temp.deepen()
    dict_temp['biomass-hierarchy_bioenergy_liquid_biogasoline'] = dm_temp
    dm_temp = dm.filter_w_regex({'Variables': 'agr_biomass-hierarchy_bioenergy_liquid_biojetkerosene.*'})
    dm_temp.deepen()
    dict_temp['biomass-hierarchy_bioenergy_liquid_biojetkerosene'] = dm_temp
    dm_temp = dm.filter_w_regex({'Variables': 'agr_biomass-hierarchy_crop_cereal.*'})
    dm_temp.deepen()
    # FIXME DOES NOT WORK
    dict_temp['biomass-hierarchy_crop_cereal'] = dm_temp
    dict_ots[lever] = dict_temp

    """    dict_ots, dict_fts = read_database_to_ots_fts_dict_w_groups(file, lever, num_cat_list=[1, 1, 1, 1, 1, 1, 1, 1], baseyear=baseyear,
                                                                years=years_all, dict_ots=dict_ots, dict_fts=dict_fts,
                                                                column='eucalc-name',
                                                                group_list=['.*biomass-hierarchy-bev-ibp-use-oth.*',
                                                                            'biomass-hierarchy_biomass-mix_digestor.*',
                                                                            'biomass-hierarchy_biomass-mix_solid.*',
                                                                            'biomass-hierarchy_biomass-mix_liquid.*',
                                                                            'biomass-hierarchy_bioenergy_liquid_biodiesel.*',
                                                                            'biomass-hierarchy_bioenergy_liquid_biogasoline.*',
                                                                            'biomass-hierarchy_bioenergy_liquid_biojetkerosene.*',
                                                                            'biomass-hierarchy_crop_cereal.*'])"""

    # Data - Lever - bioenergy capacity
    lever = 'bioenergy-capacity'
    df_ots, df_fts = database_to_df(df_bioenergy_capacity_CH_pathwaycalc, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm = DataMatrix.create_from_df(df_ots, num_cat=0)
    dict_temp = {}
    dm_temp = dm.filter_w_regex({'Variables': 'agr_bioenergy-capacity_load-factor.*'})
    dm_temp.deepen()
    dict_temp['bioenergy-capacity_load-factor'] = dm_temp
    dm_temp = dm.filter_w_regex({'Variables': 'agr_bioenergy-capacity_bgs-mix.*'})
    dm_temp.deepen()
    dict_temp['bioenergy-capacity_bgs-mix'] = dm_temp
    dm_temp = dm.filter_w_regex({'Variables': 'agr_bioenergy-capacity_efficiency.*'})
    dm_temp.deepen()
    dict_temp['bioenergy-capacity_efficiency'] = dm_temp
    dm_temp = dm.filter_w_regex({'Variables': 'agr_bioenergy-capacity_liq_b.*'})
    dm_temp.deepen()
    dict_temp['bioenergy-capacity_liq_b'] = dm_temp
    dm_temp = dm.filter_w_regex({'Variables': 'agr_bioenergy-capacity_elec.*'})
    dm_temp.deepen()
    dict_temp['bioenergy-capacity_elec'] = dm_temp
    dict_ots[lever] = dict_temp
    # Rename to correct format
    #edit_database(file,lever,column='eucalc-name',pattern={'capacity_solid-biofuel':'capacity_elec_solid-biofuel', 'capacity_biogases':'capacity_elec_biogases'},mode='rename')
    """    dict_ots, dict_fts = read_database_to_ots_fts_dict_w_groups(file, lever, num_cat_list=[1, 1, 1, 1, 1], baseyear=baseyear,
                                                                years=years_all, dict_ots=dict_ots, dict_fts=dict_fts,
                                                                column='eucalc-name',
                                                                group_list=['bioenergy-capacity_load-factor.*', 'bioenergy-capacity_bgs-mix.*',
                                                                            'bioenergy-capacity_efficiency.*', 'bioenergy-capacity_liq_b.*', 'bioenergy-capacity_elec.*'])
    """
    # Data - Lever - livestock protein meals
    lever = 'alt-protein'
    df_ots, df_fts = database_to_df(df_protein_meals_pathwaycalc, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm = DataMatrix.create_from_df(df_ots, num_cat=2)
    dict_ots[lever] = dm


    # Data - Lever - climate smart crop
    lever = 'climate-smart-crop'
    df_ots, df_fts = database_to_df(df_climate_smart_crop_pathwaycalc, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm = DataMatrix.create_from_df(df_ots, num_cat=0)
    dict_temp = {}
    dm_temp = dm.filter_w_regex({'Variables': 'agr_climate-smart-crop_losses.*'})
    dm_temp.deepen()
    dict_temp['climate-smart-crop_losses'] = dm_temp
    dm_temp = dm.filter_w_regex({'Variables': 'agr_climate-smart-crop_yield.*'})
    dm_temp.deepen()
    dict_temp['climate-smart-crop_yield'] = dm_temp
    dm_temp = dm.filter_w_regex({'Variables': 'agr_climate-smart-crop_input-use.*'})
    dm_temp.deepen()
    dict_temp['climate-smart-crop_input-use'] = dm_temp
    dm_temp = dm.filter_w_regex({'Variables': 'agr_climate-smart-crop_energy-demand.*'})
    dm_temp.deepen()
    dict_temp['climate-smart-crop_energy-demand'] = dm_temp
    dict_ots[lever] = dict_temp
    """dict_ots, dict_fts = read_database_to_ots_fts_dict_w_groups(file, lever, num_cat_list=[1, 1, 1, 1],
                                                                baseyear=baseyear,
                                                                years=years_all, dict_ots=dict_ots, dict_fts=dict_fts,
                                                                column='eucalc-name',
                                                                group_list=['climate-smart-crop_losses.*',
                                                                            'climate-smart-crop_yield.*',
                                                                            'agr_climate-smart-crop_input-use.*',
                                                                            'agr_climate-smart-crop_energy-demand.*'])"""

    #####################
    ###### CONSTANTS #######
    #####################
    # ConstantsToDatamatrix

    # FTS based on EuCalc fts
    dict_const = DM_agriculture_old['constant'].copy()


    # Data - Read Constants (use 'xx|xx|xx' to add)
    """cdm_const = ConstantDataMatrix.extract_constant('interactions_constants',
                                                    pattern='cp_time_days-per-year.*|cp_ibp_liv_.*_brf_fdk_afat|cp_ibp_liv_.*_brf_fdk_offal|cp_ibp_bev_.*|cp_liquid_tec.*|cp_load_hours|cp_ibp_aps_insect.*|cp_ibp_aps_algae.*|cp_efficiency_liv.*|cp_ibp_processed.*|cp_ef_urea.*|cp_ef_liming|cp_emission-factor_CO2.*',
                                                    num_cat=0)

    # Constant pre-processing ------------------------------------------------------------------------------------------
    # Creating a dictionnay with contants
    dict_const = {}

    # Time per year
    cdm_lifestyle = cdm_const.filter({'Variables': ['cp_time_days-per-year']})
    dict_const['cdm_lifestyle'] = cdm_lifestyle

    # Filter ibp constants for offal
    cdm_cp_ibp_offal = cdm_const.filter_w_regex({'Variables': 'cp_ibp_liv_.*_brf_fdk_offal'})
    cdm_cp_ibp_offal.rename_col_regex('_brf_fdk_offal', '', dim='Variables')
    cdm_cp_ibp_offal.rename_col_regex('liv_', 'liv_meat-', dim='Variables')
    cdm_cp_ibp_offal.deepen(based_on='Variables')  # Creating categories
    dict_const['cdm_cp_ibp_offal'] = cdm_cp_ibp_offal

    # Filter ibp constants for afat
    cdm_cp_ibp_afat = cdm_const.filter_w_regex({'Variables': 'cp_ibp_liv_.*_brf_fdk_afat'})
    cdm_cp_ibp_afat.rename_col_regex('_brf_fdk_afat', '', dim='Variables')
    cdm_cp_ibp_afat.rename_col_regex('liv_', 'liv_meat-', dim='Variables')
    cdm_cp_ibp_afat.deepen(based_on='Variables')  # Creating categories
    dict_const['cdm_cp_ibp_afat'] = cdm_cp_ibp_afat

    # Filtering relevant constants and sorting according to bev type (beer, wine, bev-alc, bev-fer)
    cdm_cp_ibp_bev_beer = cdm_const.filter_w_regex({'Variables': 'cp_ibp_bev_beer.*'})
    dict_const['cdm_cp_ibp_bev_beer'] = cdm_cp_ibp_bev_beer
    cdm_cp_ibp_bev_wine = cdm_const.filter_w_regex({'Variables': 'cp_ibp_bev_wine.*'})
    dict_const['cdm_cp_ibp_bev_wine'] = cdm_cp_ibp_bev_wine
    cdm_cp_ibp_bev_alc = cdm_const.filter_w_regex({'Variables': 'cp_ibp_bev_bev-alc.*'})
    dict_const['cdm_cp_ibp_bev_alc'] = cdm_cp_ibp_bev_alc
    cdm_cp_ibp_bev_fer = cdm_const.filter_w_regex({'Variables': 'cp_ibp_bev_bev-fer.*'})
    dict_const['cdm_cp_ibp_bev_fer'] = cdm_cp_ibp_bev_fer

    # Constants for biofuels
    cdm_biodiesel = cdm_const.filter_w_regex(({'Variables': 'cp_liquid_tec_biodiesel'}))
    cdm_biodiesel.rename_col_regex(str1="_fdk_oil", str2="", dim="Variables")
    cdm_biodiesel.rename_col_regex(str1="_fdk_lgn", str2="", dim="Variables")
    cdm_biodiesel.deepen()
    dict_const['cdm_biodiesel'] = cdm_biodiesel
    cdm_biogasoline = cdm_const.filter_w_regex(({'Variables': 'cp_liquid_tec_biogasoline'}))
    cdm_biogasoline.rename_col_regex(str1="_fdk_eth", str2="", dim="Variables")
    cdm_biogasoline.rename_col_regex(str1="_fdk_lgn", str2="", dim="Variables")
    cdm_biogasoline.deepen()
    dict_const['cdm_biogasoline'] = cdm_biogasoline
    cdm_biojetkerosene = cdm_const.filter_w_regex(({'Variables': 'cp_liquid_tec_biojetkerosene'}))
    cdm_biojetkerosene.rename_col_regex(str1="_fdk_oil", str2="", dim="Variables")
    cdm_biojetkerosene.rename_col_regex(str1="_fdk_lgn", str2="", dim="Variables")
    cdm_biojetkerosene.deepen()
    dict_const['cdm_biojetkerosene'] = cdm_biojetkerosene

    # Filter protein conversion efficiency constant
    cdm_cp_efficiency = cdm_const.filter_w_regex({'Variables': 'cp_efficiency_liv.*'})
    cdm_cp_efficiency.rename_col_regex('meat_', 'meat-', dim='Variables')
    cdm_cp_efficiency.rename_col_regex('abp_', 'abp-', dim='Variables')
    cdm_cp_efficiency.deepen(based_on='Variables')  # Creating categories
    dict_const['cdm_cp_efficiency'] = cdm_cp_efficiency

    # Constants for APS byproducts
    cdm_aps_ibp = cdm_const.filter_w_regex({'Variables': 'cp_ibp_aps.*'})
    cdm_aps_ibp.drop(dim='Variables', col_label=['cp_ibp_aps_insect_brf_fdk_manure'])
    cdm_aps_ibp.rename_col_regex('brf_', '', dim='Variables')
    cdm_aps_ibp.rename_col_regex('crop_algae', 'crop', dim='Variables')
    cdm_aps_ibp.rename_col_regex('crop_insect', 'crop', dim='Variables')
    cdm_aps_ibp.rename_col_regex('fdk_', 'fdk-', dim='Variables')
    cdm_aps_ibp.rename_col_regex('algae_', 'algae-', dim='Variables')  # Extra steps to have the correct cat order
    cdm_aps_ibp.rename_col_regex('insect_', 'insect-', dim='Variables')
    cdm_aps_ibp.deepen(based_on='Variables')  # Creating categories
    cdm_aps_ibp.rename_col_regex('algae-', 'algae_', dim='Categories1')  # Extra steps to have the correct cat order
    cdm_aps_ibp.rename_col_regex('insect-', 'insect_', dim='Categories1')
    cdm_aps_ibp.deepen(based_on='Categories1')
    dict_const['cdm_aps_ibp'] = cdm_aps_ibp

    # Food & Feed yield
    cdm_feed_yield = cdm_const.filter_w_regex({'Variables': 'cp_ibp_processed'})
    cdm_feed_yield.rename_col_regex(str1="_to_", str2="-to-", dim="Variables")
    cdm_feed_yield.deepen()
    cdm_food_yield = cdm_feed_yield.filter({'Categories1': ['sweet-to-sugarcrop']})
    cdm_feed_yield.drop(dim='Categories1', col_label=['sweet-to-sugarcrop'])
    dict_const['cdm_food_yield'] = cdm_food_yield
    dict_const['cdm_feed_yield'] = cdm_feed_yield

    # Fertilizer
    cdm_fertilizer_co = cdm_const.filter({'Variables': ['cp_ef_liming', 'cp_ef_urea']})
    cdm_fertilizer_co.deepen()
    dict_const['cdm_fertilizer_co'] = cdm_fertilizer_co

    # CO2 emissions factor bioenergy
    cdm_const.rename_col_regex(str1="liquid_", str2="liquid-", dim="Variables")
    cdm_const.rename_col_regex(str1="gas_", str2="gas-", dim="Variables")
    cdm_const.rename_col_regex(str1="solid_", str2="solid-", dim="Variables")
    cdm_CO2 = cdm_const.filter({'Variables': ['cp_emission-factor_CO2_bioenergy-gas-biogas',
                                              'cp_emission-factor_CO2_bioenergy-liquid-biodiesels',
                                              'cp_emission-factor_CO2_bioenergy-liquid-ethanol',
                                              'cp_emission-factor_CO2_bioenergy-liquid-oth',
                                              'cp_emission-factor_CO2_bioenergy-solid-wood',
                                              'cp_emission-factor_CO2_electricity',
                                              'cp_emission-factor_CO2_gas-ff-natural', 'cp_emission-factor_CO2_heat',
                                              'cp_emission-factor_CO2_liquid-ff-diesel',
                                              'cp_emission-factor_CO2_liquid-ff-fuel-oil',
                                              'cp_emission-factor_CO2_liquid-ff-gasoline',
                                              'cp_emission-factor_CO2_liquid-ff-lpg', 'cp_emission-factor_CO2_oth',
                                              'cp_emission-factor_CO2_solid-ff-coal'],
                                'units': ['MtCO2/ktoe']})
    cdm_CO2.deepen()
    dict_const['cdm_CO2'] = cdm_CO2

    # Electricity
    cdm_load = cdm_const.filter({'Variables': ['cp_load_hours-per-year-twh']})
    dict_const['cdm_load'] = cdm_load"""

    # Group all datamatrix in a single structure -----------------------------------------------------------------------
    DM_agriculture = {
        'fxa': DM_agriculture_old['fxa'],
        'constant': dict_const,
        'fts': DM_fts,
        'ots': dict_ots
    }

    # FXA pre-processing -----------------------------------------------------------------------------------------------

    # Emssion factors residues residues
    #DM_agriculture['fxa']['ef_soil-residues'].add(0.0, dummy=True, col_label='CH4-emission', dim='Categories1', unit='Mt')
    #DM_agriculture['fxa']['ef_soil-residues'].sort(dim='Categories1')
    #DM_agriculture['fxa']['ef_burnt-residues'].append(DM_agriculture['fxa']['ef_soil-residues'], dim='Variables')
    #DM_agriculture['fxa']['ef_burnt-residues'] = DM_agriculture['fxa']['ef_burnt-residues'].flatten()  # extra steps to have correct deepening
    #DM_agriculture['fxa']['ef_burnt-residues'].rename_col_regex(str1="residues_", str2="residues-", dim="Variables")
    #DM_agriculture['fxa']['ef_burnt-residues'].rename_col_regex(str1="fxa_", str2="", dim="Variables")
    #DM_agriculture['fxa']['ef_burnt-residues'].deepen()
    #DM_agriculture['fxa']['ef_burnt-residues'].rename_col_regex(str1="residues-", str2="residues_", dim="Categories1")
    #DM_agriculture['fxa']['ef_burnt-residues'].deepen()

    # caf GHG emissions
    #DM_agriculture['fxa']['cal_agr_emission_CH4'].append(DM_agriculture['fxa']['cal_agr_emission_N2O'], dim='Variables')
    #DM_agriculture['fxa']['cal_agr_emission_CH4'].append(DM_agriculture['fxa']['cal_agr_emission_CO2'], dim='Variables')
    #DM_agriculture['fxa']['cal_agr_emission_CH4'].rename_col_regex(str1='cal_agr_emissions-', str2='cal_agr_emissions_', dim='Variables')
    #DM_agriculture['fxa']['cal_agr_emission_CH4'].deepen()

    # write datamatrix to pickle
    f = '../../data/datamatrix/agriculture.pickle'
    with open(f, 'wb') as handle:
        pickle.dump(DM_agriculture, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


# CalculationTree RUNNING LEVERS PRE-PROCESSING -----------------------------------------------------------------------------------------------
years_ots = create_years_list(1990, 2023, 1)  # make list with years from 1990 to 2015
years_fts = create_years_list(2025, 2050, 5)
years_all = years_ots + years_fts

if not os.path.exists('data/faostat'):
    os.makedirs('data/faostat')

df_feed_lsu_pathwaycalc = feed_processing_lca()
list_countries = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark',
                  'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia',
                  'Lithuania', 'Luxembourg', 'Malta', 'Netherlands (Kingdom of the)', 'Poland', 'Portugal',
                  'Romania', 'Slovakia',
                  'Slovenia', 'Spain', 'Sweden', 'Switzerland',
                  'United Kingdom of Great Britain and Northern Ireland']

file = 'data/faostat/diet.csv'
df_diet_pathwaycalc, df_diet = diet_processing(list_countries, file)
df_waste_pathwaycalc = food_waste_processing(df_diet)
dm_kcal_req_pathwaycalc = energy_requirements_processing(country_list=df_waste_pathwaycalc['geoscale'].unique(), years_ots=years_ots)
file_dict = {'ssr': 'data/faostat/ssr.csv', 'cake': 'data/faostat/ssr_cake.csv',
             'molasse': 'data/faostat/ssr_2010_2021_molasse_cake.csv'}
df_ssr_pathwaycalc, df_csl_feed = self_sufficiency_processing(years_ots, list_countries, file_dict)
df_feed_ssr_pathwaycalc = feed_ssr_processing(years_ots)
df_land_use_fao_calibration, df_cropland_density, df_agri_land = land_calibration(list_countries)
file_dict = {'losses': 'data/faostat/losses.csv', 'yield': 'data/faostat/yield.csv','cropland': 'data/faostat/cropand.csv', 'urea': 'data/faostat/urea.csv',
             'land': 'data/faostat/land.csv', 'nitro': 'data/faostat/nitro.csv',
             'pesticide': 'data/faostat/pesticide.csv', 'liming': 'data/faostat/liming.csv'}
df_climate_smart_crop_pathwaycalc, df_energy_demand_cal, df_CO2_cal = climate_smart_crop_processing(list_countries, df_agri_land, file_dict)
# Exceptionnally running livestock calibration & feed calibration before to re-use
df_domestic_supply_calibration, df_liv_population_calibration, df_liv_pop = livestock_crop_calibration(df_energy_demand_cal, list_countries)
df_feed_calibration, df_feed_ration = feed_calibration(list_countries)
df_climate_smart_livestock_pathwaycalc, df_csl_fxa, df_manure_n_fxa, df_manure_ch4_fxa = climate_smart_livestock_processing(df_feed_ration, df_liv_pop, df_cropland_density, list_countries)
df_climate_smart_forestry_pathwaycalc, csf_managed = climate_smart_forestry_processing() #FutureWarning at last line
df_ruminant_feed_pathwaycalc = ruminant_feed_processing(df_csl_feed)
df_land_management_pathwaycalc = land_management_processing(csf_managed)
df_bioenergy_capacity_CH_pathwaycalc = bioernergy_capacity_processing(df_csl_feed)
df_biomass_hierarchy_pathwaycalc = biomass_bioernergy_hierarchy_processing(df_csl_feed)
df_protein_meals_pathwaycalc = livestock_protein_meals_processing(df_csl_feed)

# CalculationTree RUNNING CALIBRATION ----------------------------------------------------------------------------------
df_diet_calibration = lifestyle_calibration(list_countries)
df_nitrogen_calibration = nitrogen_calibration(list_countries)
df_liv_emissions_calibration, df_liv_emissions = manure_calibration(list_countries)
df_cropland_fao_calibration = cropland_calibration(list_countries)
df_liming_urea_calibration, df_liming_urea = CO2_emissions()
df_wood_calibration = wood_calibration(list_countries)
df_emissions_calibration = energy_ghg_calibration(list_countries, df_CO2_cal, df_liming_urea) # Fixme PerformanceWarning ?
df_calibration = calibration_formatting(df_diet_calibration, df_domestic_supply_calibration, df_liv_population_calibration,
                     df_nitrogen_calibration, df_liv_emissions_calibration, df_feed_calibration,
                     df_land_use_fao_calibration, df_liming_urea_calibration, df_wood_calibration,
                     df_emissions_calibration, df_cropland_fao_calibration) # Fixme PerformanceWarning ?

df_cnst = constant()
df_manure_fxa = manure_fxa(list_countries, df_liv_emissions, df_manure_n_fxa, df_manure_ch4_fxa)

# CalculationTree RUNNING FXA PRE-PROCESSING ---------------------------------------------------------------------------
#fxa_preprocessing()
# CalculationTree RUNNING PICKLE CREATION
database_from_csv_to_datamatrix(years_ots, years_fts, dm_kcal_req_pathwaycalc, df_csl_fxa, df_manure_fxa, df_calibration, df_feed_lsu_pathwaycalc, df_diet_pathwaycalc, df_ruminant_feed_pathwaycalc) #Fixme duplicates in constants

# CalculationTree ADDITIONAL PRE-PROCESSING ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

# Load pickles
with open('../../data/datamatrix/agriculture.pickle', 'rb') as handle:
    DM_agriculture = pickle.load(handle)

with open('../../data/datamatrix/lifestyles.pickle', 'rb') as handle:
    DM_lifestyles = pickle.load(handle)

# Filter DM
filter_DM(DM_agriculture, {'Country': ['Switzerland']})
filter_DM(DM_lifestyles, {'Country': ['Switzerland']})

# ---------------------------------------------------------------------------------------------------------
# ADDING CONSTANTS ----------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------

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
arr = np.zeros((len(cdm_kcal.col_labels['Variables']), len(cdm_kcal.col_labels['Categories1'])))
cdm_kcal.array = arr
idx = cdm_kcal.idx
for cat, val in dict_kcal_t.items():
    cdm_kcal.array[idx['cp_kcal-per-t'], idx[cat]] = val
cdm_kcal.units["cp_kcal-per-t"] = "kcal/t"

# Append to DM_agriculture['constant']
DM_agriculture['constant']['cdm_kcal-per-t'] = cdm_kcal

# CP EF FUEL ----------------------------------------------------------------------------------------

# convert from [TJ] to [ktoe]
tj_to_ktoe = 0.02388458966275  # source https://www.unitjuggler.com/convertir-energy-de-TJ-en-kltoe.htm

# Source : UNFCCC Table 1.1(a)s4
DM_agriculture['constant']['cdm_CO2']['cp_emission-factor_CO2','bioenergy-solid-wood'] = 10**-6 * 72.71 / tj_to_ktoe
DM_agriculture['constant']['cdm_CO2']['cp_emission-factor_CO2','gas-ff-natural'] = 10**-6 * 55.90 / tj_to_ktoe
DM_agriculture['constant']['cdm_CO2']['cp_emission-factor_CO2','liquid-ff-diesel'] = 10**-6 * 73.30  / tj_to_ktoe
DM_agriculture['constant']['cdm_CO2']['cp_emission-factor_CO2','liquid-ff-gasoline'] = 10**-6 * 73.80 / tj_to_ktoe
DM_agriculture['constant']['cdm_CO2']['cp_emission-factor_CO2','liquid-ff-lpg'] = 10**-6 * 0.0 / tj_to_ktoe
DM_agriculture['constant']['cdm_CO2']['cp_emission-factor_CO2','solid-ff-coal'] = 10**-6 * 0.0 / tj_to_ktoe

# CP - FEED - ENERGY CONVERSION EFFICIENCY  ----------------------------------------------------------------------------------------

# Source : Alexander, P., Brown, C., Arneth, A., Finnigan, J., Rounsevell, M.D.A., 2016.
# Human appropriation of land for food: The role of diet. Glob. Environ.
# Change 41, 88–98. https://doi.org/10.1016/j.gloenvcha.2016.09.005
# Feed conversion ratio (kg DM feed/kg EW) EW: edible weight
DM_agriculture['constant']['cdm_cp_efficiency']['cp_efficiency_liv','abp-dairy-milk'] = 0.7
DM_agriculture['constant']['cdm_cp_efficiency']['cp_efficiency_liv','abp-hens-egg'] = 2.3
DM_agriculture['constant']['cdm_cp_efficiency']['cp_efficiency_liv','meat-bovine'] = 25

# Source Agristat 2023 https://www.sbv-usp.ch/fr/ettiquettes/agristat
DM_agriculture['constant']['cdm_cp_efficiency']['cp_efficiency_liv','meat-oth-animals'] = 312619/3541
DM_agriculture['constant']['cdm_cp_efficiency']['cp_efficiency_liv','meat-pig'] = 723830/212594
DM_agriculture['constant']['cdm_cp_efficiency']['cp_efficiency_liv','meat-poultry'] = 390581/115808
DM_agriculture['constant']['cdm_cp_efficiency']['cp_efficiency_liv','meat-sheep'] = 212831/4932

# Change unit
DM_agriculture['constant']['cdm_cp_efficiency'].units = 'kg DM feed/kg EW'

# FXA EF NITROGEN FERTILIZER ----------------------------------------------------------------------------------------
# Load data
dm_emission_fert = DM_agriculture['fxa']['cal_agr_crop_emission_N2O-emission_fertilizer'].copy()
dm_input_fert = DM_agriculture['ots']['climate-smart-crop']['climate-smart-crop_input-use'].copy()
dm_land = DM_agriculture['fxa']['cal_agr_lus_land'].copy()

# Compute total land
dm_land.group_all(dim='Categories1', inplace=True)

# CHange unit from Mt => t
dm_emission_fert.change_unit('cal_agr_crop_emission_N2O-emission_fertilizer', old_unit='Mt', new_unit='t', factor=10**6)

# Filter and flatten
dm_input_fert = dm_input_fert.filter({'Categories1':['nitrogen']})
dm_input_fert = dm_input_fert.flatten()

# Append & compute
dm_input_fert.append(dm_emission_fert, dim='Variables')
dm_input_fert.append(dm_land, dim='Variables')
dm_input_fert.operation('agr_climate-smart-crop_input-use_nitrogen', '*', 'cal_agr_lus_land',
                                 out_col='temp', unit='tN')
dm_input_fert.operation('cal_agr_crop_emission_N2O-emission_fertilizer', '/', 'temp',
                                 out_col='fxa_agr_emission_fertilizer', unit='N2O/N')

# Extrapolate to fts
linear_fitting(dm_input_fert, years_all)

# Overwrite fxa_agr_emission_fertilizer in pickle
DM_agriculture['fxa']['agr_emission_fertilizer']['Switzerland',:,'fxa_agr_emission_fertilizer'] = dm_input_fert['Switzerland',:,'fxa_agr_emission_fertilizer']

# CALIBRATION DOMESTIC PROD WITH LOSSES ----------------------------------------------------------------------------------------

# Load data
dm_dom_prod_liv = DM_agriculture['fxa']['cal_agr_domestic-production-liv'].copy()
dm_losses_liv = DM_agriculture['ots']['climate-smart-livestock']['climate-smart-livestock_losses'].copy()
dm_dom_prod_crop = DM_agriculture['fxa']['cal_agr_domestic-production_food'].copy()
dm_losses_crop = DM_agriculture['ots']['climate-smart-crop']['climate-smart-crop_losses'].copy()


# Livestock domestic prod with losses [kcal] = livestock domestic prod [kcal] * Production losses livestock [%]
dm_losses_liv.drop(dim='Categories1', col_label=['abp-processed-afat', 'abp-processed-offal'])
dm_dom_prod_liv.rename_col('cal_agr_domestic-production-liv', 'cal_agr_domestic-production-liv_raw', dim='Variables')
dm_dom_prod_liv.append(dm_losses_liv, dim='Variables')
dm_dom_prod_liv.operation('agr_climate-smart-livestock_losses', '*', 'cal_agr_domestic-production-liv_raw',
                                 out_col='cal_agr_domestic-production-liv', unit='kcal')

# Crop domestic prod with losses [kcal] = crop domestic prod [kcal] * Production losses crop [%]
dm_dom_prod_crop.rename_col('cal_agr_domestic-production_food', 'cal_agr_domestic-production_food_raw', dim='Variables')
dm_dom_prod_crop.append(dm_losses_crop, dim='Variables')
dm_dom_prod_crop.operation('agr_climate-smart-crop_losses', '*', 'cal_agr_domestic-production_food_raw',
                                 out_col='cal_agr_domestic-production_food', unit='kcal')

# Overwrite
DM_agriculture['fxa']['cal_agr_domestic-production-liv']['Switzerland', :,'cal_agr_domestic-production-liv',:] \
    = dm_dom_prod_liv['Switzerland', :,'cal_agr_domestic-production-liv',:]
DM_agriculture['fxa']['cal_agr_domestic-production_food']['Switzerland', :,'cal_agr_domestic-production_food',:] \
    = dm_dom_prod_crop['Switzerland', :,'cal_agr_domestic-production_food',:]

# LIVESTOCK YIELD USING CALIBRATION DOMESTIC PROD WITH LOSSES ----------------------------------------------------------------------------------------

# Load data
dm_dom_prod_liv = DM_agriculture['fxa']['cal_agr_domestic-production-liv'].copy()
dm_yield = DM_agriculture['ots']['climate-smart-livestock']['climate-smart-livestock_yield'].copy()

# Yield [kcal/lsu] = Domestic prod with losses [kcal] / producing-slaugthered animals [lsu]
dm_yield.rename_col('agr_climate-smart-livestock_yield', 'agr_climate-smart-livestock_yield_raw', dim='Variables')
dm_dom_prod_liv.append(dm_yield, dim='Variables')
dm_dom_prod_liv.operation('cal_agr_domestic-production-liv', '/', 'agr_climate-smart-livestock_yield_raw',
                                 out_col='agr_climate-smart-livestock_yield', unit='kcal/lsu')

# Overwrite
DM_agriculture['ots']['climate-smart-livestock']['climate-smart-livestock_yield']['Switzerland', :,'agr_climate-smart-livestock_yield',:] \
    = dm_dom_prod_liv['Switzerland', :,'agr_climate-smart-livestock_yield',:]

# DIET ----------------------------------------------------------------------------------------
# The idea was to have energy requirements per demography (agr_kcal-req) based on the current consumption and not the
# calculated based on the metabolism.
# AND update the calibration values for cal_diet.

# Load data
dm_others = DM_agriculture['ots']['diet']['share'].copy()
dm_others.change_unit('share', old_unit='%', new_unit='kcal/cap/day', factor=1)
dm_diet = DM_agriculture['ots']['diet']['lfs_consumers-diet'].copy()
dm_waste = DM_agriculture['ots']['fwaste'].copy()
dm_waste.filter({'Categories1':dm_others.col_labels['Categories1']}, inplace=True)
dm_req = DM_agriculture['ots']['kcal-req'].copy()
dm_demography = DM_lifestyles['ots']['pop']['lfs_demography_'].copy()
dm_population = DM_lifestyles['ots']['pop']['lfs_population_'].copy()
dm_cal_diet = DM_agriculture['fxa']['cal_agr_diet'].copy() # Now it's actually in (kcal/capita/day)

# Diet demand [kcal/cap/day] = food supply [kcal/cap/day] - food waste [kcal/cap/day]
dm_others.append(dm_waste, dim='Variables')
dm_others.operation('share', '-', 'lfs_consumers-food-wastes', out_col='lfs_consumers-diet', unit='kcal/cap/day')

# In dm_diet, compute lfs_consumers-diet + lfs_consumers-food-wastes

# Append together
dm_diet.append(dm_others.filter({'Variables':['lfs_consumers-diet']}), dim='Categories1')

# Sum total food demand (based on actual consumption)
dm_diet.group_all(dim='Categories1', inplace=True)

# Divide share by the total food supply available
arr = dm_others[:,:,'lfs_consumers-diet',:] / dm_diet[:,:,'lfs_consumers-diet', np.newaxis]
dm_others.add(arr, dim='Variables', col_label='share_total', unit='%')

# Normalise to obtain a ratio sum = 1
dm_others.normalise('Categories1', inplace=True)

# Diet demand [kcal/day] = Diet demand [kcal/cap/day] * Population [cap]
dm_diet.append(dm_population, dim='Variables')
dm_diet.operation('lfs_consumers-diet', '*', 'lfs_population_total', out_col='lfs_consumers-diet_tot', unit='kcal/day')

# Normalise dm_req to obtain the share of kcal by age & gender categorie
dm_req.append(dm_demography, dim='Variables')
dm_req.operation('agr_kcal-req', '*', 'lfs_demography', out_col='agr_kcal-req_req', unit='kcal/day')
dm_req.normalise('Categories1', keep_original=True)

# Filter for same countries
dm_diet.filter({'Country':dm_req.col_labels['Country']}, inplace=True)

# Check country order
dm_diet.sort('Country')
dm_req.sort('Country')

# Demand per age gender group [kcal/day]= share kcal per age gender group [%] * total food demand [kcal/day]
arr = dm_diet[:,:,'lfs_consumers-diet_tot', np.newaxis] * dm_req[:,:,'agr_kcal-req_req_share',:]
dm_req.add(arr, dim='Variables', col_label='demand_per_group', unit='kcal/day')

# Demand per age gender group [kcal/cap/day] = Demand per age gender group [kcal/day] / Demography [cap]
dm_req.operation('demand_per_group', '/', 'lfs_demography', out_col='agr_kcal-req_temp', unit='kcal/cap/day')

# For calibration : cal_agr_diet [kcal/year] = cal_agr_diet [kcal/cap/day] * population [capita] * 365,25
arr = dm_cal_diet[:,:,'cal_agr_diet', :] * dm_population[:,:,'lfs_population_total',np.newaxis] * 365.25
dm_cal_diet.add(arr, dim='Variables', col_label='cal_agr_diet_new', unit='kcal')

# Save in DM_agriculture
DM_agriculture['ots']['kcal-req']['Switzerland', :,'agr_kcal-req',:] = dm_req['Switzerland',:,'agr_kcal-req_temp',:]
# Overwrite shares
DM_agriculture['ots']['diet']['share']['Switzerland', :,'share',:] = dm_others['Switzerland', :,'share',:]
# Overwrite cal_diet
DM_agriculture['fxa']['cal_agr_diet']['Switzerland', :,'cal_agr_diet',:] = dm_cal_diet['Switzerland', :,'cal_agr_diet_new',:]


# SSR ----------------------------------------------------------

# Load data
dm_dom_prod = DM_agriculture['ots']['food-net-import'].copy()
CDM_const = DM_agriculture['constant'].copy()
cdm_kcal = CDM_const['cdm_kcal-per-t'].copy()
#cdm_kcal.drop(dim='Categories1', col_label='crop-sugarcrop')
cdm_kcal.drop(dim='Categories1', col_label='stm')
cdm_kcal.drop(dim='Categories1', col_label='liv-meat-meal')
dm_cal_diet = dm_cal_diet.filter({'Variables': ['cal_agr_diet_new']}).copy()
cdm_food_yield = CDM_const['cdm_food_yield'].copy()

# Separate SSR of pro-crop-processed-cake, pro-crop-processed-molasse back in dm
dm_feed = dm_dom_prod.filter(
  {'Categories1': ['pro-crop-processed-cake','pro-crop-processed-molasse']})

# Rename categories
cat_diet = [
    'afat', 'beer', 'bev-alc', 'bev-fer', 'bov', 'cereals', 'cocoa', 'coffee',
    'dfish', 'egg', 'ffish', 'fruits', 'milk', 'offal', 'oilcrops', 'oth-animals',
    'oth-aq-animals', 'pfish', 'pigs', 'poultry', 'pulses', 'seafood',
    'sheep', 'starch', 'sugar', 'sweet', 'tea', 'veg', 'voil', 'wine'
]
cat_agr = [
    'pro-liv-abp-processed-afat', 'pro-bev-beer', 'pro-bev-bev-alc', 'pro-bev-bev-fer',
    'pro-liv-meat-bovine', 'crop-cereal', 'cocoa', 'coffee', 'dfish',
    'pro-liv-abp-hens-egg', 'ffish', 'crop-fruit', 'pro-liv-abp-dairy-milk',
    'pro-liv-abp-processed-offal', 'crop-oilcrop', 'pro-liv-meat-oth-animals',
    'oth-aq-animals', 'pfish', 'pro-liv-meat-pig', 'pro-liv-meat-poultry',
    'crop-pulse', 'seafood', 'pro-liv-meat-sheep', 'crop-starch',
    'pro-crop-processed-sugar', 'pro-crop-processed-sweet', 'tea', 'crop-veg',
    'pro-crop-processed-voil', 'pro-bev-wine'
]
dm_cal_diet.rename_col(cat_diet, cat_agr, 'Categories1')

# For sugarcrops : sugarcrops (processed) = processed sugar + processed sweet
dm_sugarcrop = dm_cal_diet.groupby({'crop-sugarcrop': '.*-sweet|.*-sugar'}, dim='Categories1',
                          regex=True, inplace=False)
# Account for processing yield
array_temp = dm_sugarcrop[:, :,
             'cal_agr_diet_new', :] \
             * cdm_food_yield[np.newaxis, np.newaxis, 'cp_ibp_processed', :]
dm_sugarcrop.add(array_temp, dim='Variables',
                      col_label='cal_agr_diet_temp', unit='kcal')
# add back in food demand
dm_sugarcrop = dm_sugarcrop.filter({'Variables': ['cal_agr_diet_temp']})
dm_sugarcrop.rename_col('cal_agr_diet_temp', 'cal_agr_diet_new', dim='Variables')
dm_cal_diet.append(dm_sugarcrop, dim='Categories1')

# Check Category order
dm_dom_prod.sort('Categories1')
cdm_kcal.sort('Categories1')

# Unit conversion: [kt] => [kcal]
# Convert from [kt] to [t]
dm_dom_prod.change_unit('agr_food-net-import', 10 ** 3, old_unit='%',
                         new_unit='t')
# Convert from [t] to [kcal]
array_temp = dm_dom_prod[:, :,
             'agr_food-net-import', :] \
             * cdm_kcal[np.newaxis, np.newaxis, 'cp_kcal-per-t', :]
dm_dom_prod.add(array_temp, dim='Variables',
                      col_label='agr_food-net-import_kcal',
                      unit='kcal')
dm_dom_prod = dm_dom_prod.filter(
  {'Variables': ['agr_food-net-import_kcal']})

# Drop columns that are not present in agr_demand (Diet calibration)
dm_dom_prod.drop(dim='Categories1', col_label=['pro-crop-processed-cake',
                                               'pro-crop-processed-molasse'])

# Compute SSR [%] : production / agr_demand
# Except for crop-sugarcrop, pro-crop-processed-cake, pro-crop-processed-molasse.
dm_dom_prod.append(dm_cal_diet, dim='Variables')
dm_dom_prod.operation('agr_food-net-import_kcal', '/', 'cal_agr_diet_new', dim='Variables',
                          out_col='agr_food-net-import', unit='%')
dm_dom_prod = dm_dom_prod.filter(
  {'Variables': ['agr_food-net-import']})

# Add SSR of pro-crop-processed-cake, pro-crop-processed-molasse back in dm
dm_dom_prod.append(dm_feed, dim='Categories1')

# Check Category order
dm_dom_prod.sort('Categories1')
DM_agriculture['ots']['food-net-import'].sort('Categories1')
for i in range(1, 5):
  DM_agriculture['fts']['food-net-import'][i].sort('Categories1')

# Overwrite
DM_agriculture['ots']['food-net-import']['Switzerland', :,'agr_food-net-import',:] = dm_dom_prod['Switzerland', :,'agr_food-net-import',:]

# FEED - SHARE GRASS OTS ----------------------------------------------------------------------------------------

# Load
dm_dom_prod_liv = DM_agriculture['fxa']['cal_agr_domestic-production-liv'].copy()
cdm_cp_efficiency = CDM_const['cdm_cp_efficiency']
cdm_kcal = CDM_const['cdm_kcal-per-t'].copy()
dm_feed_cal = DM_agriculture['fxa']['cal_agr_demand_feed'].copy()

# ASF domestic prod with losses => Unit conversion: [kcal] to [t]
cdm_kcal.rename_col_regex(str1="pro-liv-", str2="", dim="Categories1")
cdm_kcal = cdm_kcal.filter({'Categories1': ['abp-dairy-milk', 'abp-hens-egg',
                                            'meat-bovine', 'meat-oth-animals',
                                            'meat-pig', 'meat-poultry',
                                            'meat-sheep']})
dm_dom_prod_liv.sort('Categories1')
cdm_kcal.sort('Categories1')
array_temp = dm_dom_prod_liv[:, :, 'cal_agr_domestic-production-liv', :] \
             / cdm_kcal[np.newaxis, np.newaxis, 'cp_kcal-per-t', :]
dm_dom_prod_liv.add(array_temp, dim='Variables',
                col_label='agr_domestic_production_liv_afw_t',
                unit='t')

# Feed req with grass per type [t] =  ASF domestic prod with losses [kt] * FCR [%]
dm_dom_prod_liv.sort('Categories1')
cdm_cp_efficiency.sort('Categories1')
dm_temp = dm_dom_prod_liv[:, :, 'agr_domestic_production_liv_afw_t', :] \
          * cdm_cp_efficiency[np.newaxis, np.newaxis, 'cp_efficiency_liv', :]
dm_dom_prod_liv.add(dm_temp, dim='Variables', col_label='agr_feed-requirement',
                unit='t')

# Feed req total with grass [t] =  sum per type (Feed req with grass per type [t])
dm_dom_prod_liv = dm_dom_prod_liv.filter(
  {'Variables': ['agr_feed-requirement']})
dm_ruminant = dm_dom_prod_liv.filter(
  {'Categories1': ['abp-dairy-milk', 'meat-bovine', 'meat-sheep']}) # Create copy for ruminants
dm_dom_prod_liv.groupby({'total': '.*'}, dim='Categories1', regex=True,
                          inplace=True)
dm_dom_prod_liv = dm_dom_prod_liv.flatten()

# Feed req total without grass FAO [t] = sum (feed FBS + SQL)
dm_feed_cal = dm_feed_cal.filter(
  {'Variables': ['cal_agr_demand_feed']})
dm_feed_cal.groupby({'total': '.*'}, dim='Categories1', regex=True,
                          inplace=True)
dm_feed_cal = dm_feed_cal.flatten()

# Grass feed [t] = Feed req total with grass [t] - Feed req total without grass FAO [t]
dm_dom_prod_liv.append(dm_feed_cal, dim='Variables')
dm_dom_prod_liv.operation('agr_feed-requirement_total', '-', 'cal_agr_demand_feed_total',
                                 out_col='grass_feed', unit='t')

# Feed ruminant with grass [t] = sum (feed ruminant [t])
dm_ruminant.groupby({'ruminant': '.*'}, dim='Categories1', regex=True,
                          inplace=True)
dm_ruminant = dm_ruminant.flatten()

# Share grass feed ruminant [%] = Grass feed [t] / Feed ruminant with grass [t]
dm_dom_prod_liv.append(dm_ruminant, dim='Variables')
dm_dom_prod_liv.operation('grass_feed', '/', 'agr_feed-requirement_ruminant',
                                 out_col='agr_ruminant-feed_share-grass', unit='%')

# Overwrite in pickle
DM_agriculture['ots']['ruminant-feed']['ruminant-feed']['Switzerland',:,'agr_ruminant-feed_share-grass'] = dm_dom_prod_liv['Switzerland',:,'agr_ruminant-feed_share-grass']



# ADD DUMMY COUNTRIES ----------------------------------------------------------

# Add EU27 and Vaud as dummys
add_dummy_country_to_DM(DM_agriculture, 'Germany', 'Switzerland')
add_dummy_country_to_DM(DM_agriculture, 'EU27', 'Germany')
add_dummy_country_to_DM(DM_agriculture, 'Vaud', 'Switzerland')

# Overwrite in pickle
f = '../../data/datamatrix/agriculture.pickle'
with open(f, 'wb') as handle:
    pickle.dump(DM_agriculture, handle, protocol=pickle.HIGHEST_PROTOCOL)
