import numpy as np
from model.common.auxiliary_functions import interpolate_nans, add_missing_ots_years, linear_fitting_ots_db, linear_fitting, create_years_list
#from _database.pre_processing.api_routines_CH import get_data_api_CH
from scipy.stats import linregress
import pandas as pd
import faostat
from _database.pre_processing.api_routines_CH import get_data_api_CH
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

# CalculationLeaf SSR LIVESTOCK PROD & FEED
def self_sufficiency_processing(years_ots, list_countries_calc, file_dict):
    # Read data ------------------------------------------------------------------------------------------------------------
    try:
        df_ssr = pd.read_csv(file_dict['ssr'])
    except OSError:

        # FOOD BALANCE SHEETS (FBS) - For everything except molasses and cakes -------------------------------------------------
        # List of elements
        list_elements = ['Production Quantity', 'Import Quantity', 'Export Quantity', 'Feed', 'Processed', 'Stock Variation', 'Food', 'Other uses (non-food)', 'Residuals']

        list_items = ['Milk - Excluding Butter + (Total)', 'Eggs + (Total)',
                      'Bovine Meat', 'Meat, Other', 'Pigmeat',
                      'Poultry Meat', 'Mutton & Goat Meat']

        # 1990 - 2013
        ld = faostat.list_datasets()
        code = 'FBSH'
        pars = faostat.list_pars(code)
        my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries_calc]
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

        list_elements = ['Production Quantity', 'Import quantity', 'Export quantity', 'Feed', 'Processed', 'Stock Variation', 'Food', 'Other uses (non-food)', 'Residuals']
        # Different list becuse different in item nomination such as rice
        list_items = ['Milk - Excluding Butter + (Total)', 'Eggs + (Total)',
                      'Bovine Meat', 'Meat, Other', 'Pigmeat',
                      'Poultry Meat', 'Mutton & Goat Meat']
        code = 'FBS'
        my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries_calc]
        my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
        my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
        list_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
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
          df_ssr['Item'].str.contains('Rice (Milled Equivalent)', case=False,
                                      na=False, regex=False),'Item'] = 'Rice and products'

        df_ssr.to_csv(file_dict['ssr'], index=False)

    # COMMODITY BALANCES (NON-FOOD) (OLD METHODOLOGY) - For molasse and cakes ----------------------------------------------
    try:
        df_ssr_cake = pd.read_csv(file_dict['cake'])
        df_ssr_2010_2021_molasse_cake = pd.read_csv(file_dict['molasse'])
    except OSError:
        # 1990 - 2013
        list_elements = ['Production Quantity', 'Import quantity', 'Export quantity', 'Feed', 'Food']
        list_items = ['Copra Cake', 'Cottonseed Cake', 'Groundnut Cake', 'Oilseed Cakes, Other', 'Palmkernel Cake',
                      'Rape and Mustard Cake', 'Sesameseed Cake', 'Soyabean Cake', 'Sunflowerseed Cake']
        code = 'CBH'
        my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries_calc]
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
                      'Cake of sesame seed', 'Cake of sunflower seed', 'Cake, oilseeds nes', 'Cake, poppy seed']
        code = 'SCL'
        my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries_calc]
        my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
        my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
        list_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
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
        df_ssr_2010_2021_molasse_cake.loc[
          df_ssr_2010_2021_molasse_cake['Element'].str.contains(
            'Food supply quantity (tonnes)', case=False, na=False), 'Element'] = 'Food'
        df_ssr_1990_2013_cake.loc[
          df_ssr_1990_2013_cake['Element'].str.contains(
            'Food supply quantity (tonnes)', case=False,
            na=False), 'Element'] = 'Food'

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
    df_ssr_feed = pd.concat([df_ssr_molasses, df_ssr_cake])

    # Change unit from [t] => [kt]
    df_ssr_feed['Value'] = df_ssr_feed['Value'] * 10**(-3)

    # Filtering to keep wanted columns
    columns_to_filter = ['Area', 'Element', 'Item', 'Year', 'Value']
    df_ssr = df_ssr[columns_to_filter]
    df_ssr_feed = df_ssr_feed[columns_to_filter]

    # Concat and create copy for processing yield
    df_processing_yield_fxa = pd.concat([df_ssr, df_ssr_feed])

    # Compute Self-Sufficiency Ratio (SSR) ---------------------------------------------------------------------------------
    # 1: Pivot the DataFrame to get 'Production', 'Import Quantity', and 'Export Quantity' in separate columns
    pivot_df = df_ssr.pivot_table(index=['Area', 'Year', 'Item'], columns='Element', values='Value').reset_index()
    pivot_df_feed = df_ssr_feed.pivot_table(index=['Area', 'Year', 'Item'],
                                  columns='Element',
                                  values='Value').reset_index()

    # Fill na with 0
    cols = [
      'Production', 'Import', 'Export', 'Feed', 'Food',
      'Residuals', 'Processing', 'Other uses (non-food)', 'Stock Variation'
    ]

    for c in cols:
      pivot_df[c] = pivot_df[c].fillna(0.0)

    # For pivot_df_feed (only 3 columns)
    cols_feed = ['Production', 'Import', 'Export']

    for c in cols_feed:
      pivot_df_feed[c] = pivot_df_feed[c].fillna(0.0)

    # Create a copy for feed pre-processing and drop irrelevant columns
    df_csl_feed = pd.concat([pivot_df, pivot_df_feed])
    columns_to_filter = ['Area', 'Year', 'Item', 'Feed']
    df_csl_feed = df_csl_feed[columns_to_filter].copy()

    # Create a copy for milk feed food ratio (fxa)
    df_ffr_milk = pivot_df[pivot_df['Item'] =='Milk - Excluding Butter']
    df_ffr_milk = df_ffr_milk.copy()

    # 2: Compute the SSR [%]
    # (previously with special condition for milk because we
    # don't account for it as feed & processed. but now fixed with and fxa_ratio)
    pivot_df['SSR[%]'] = pivot_df['Production'] / (pivot_df['Food'] + pivot_df['Feed'] + pivot_df['Processing'])
    # Fill nan
    pivot_df_feed['SSR[%]'] = pivot_df_feed['Production']/(pivot_df_feed['Feed'])

    # Filter columns
    columns_to_filter = ['Area', 'Year', 'Item', 'SSR[%]']
    pivot_df = pivot_df[columns_to_filter]
    pivot_df_feed = pivot_df_feed[columns_to_filter]

    # Concat dfs
    #pivot_df = pd.concat([pivot_df, pivot_df_feed])

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------

    # Food item name matching with dictionary
    # Read excel file
    df_dict_ssr = pd.read_excel(
        'dictionaries/dictionnary_livestock.xlsx',
        sheet_name='self-sufficiency')

    # Prepend 'SSR'
    pivot_df['Item'] = pivot_df['Item'].apply(lambda x: f"SSR {x}")
    pivot_df_feed['Item'] = pivot_df_feed['Item'].apply(lambda x: f"SSR {x}")

    # Renaming existing columns (geoscale, timsecale, value)
    pivot_df.rename(columns={'Area': 'geoscale', 'Year': 'timescale', 'SSR[%]': 'value'}, inplace=True)
    pivot_df_feed.rename(
      columns={'Area': 'geoscale', 'Year': 'timescale', 'SSR[%]': 'value'},
      inplace=True)

    # Merge based on 'Item'
    df_ssr_liv = pd.merge(df_dict_ssr, pivot_df, on='Item')
    df_ssr_feed = pd.merge(df_dict_ssr, pivot_df_feed, on='Item')

    # Drop the 'Item' column
    df_ssr_liv = df_ssr_liv.drop(columns=['Item'])
    df_ssr_feed = df_ssr_feed.drop(columns=['Item'])


    # Adding the columns module, lever, level and string-pivot at the correct places
    lever = 'food-net-import'
    df_ssr_liv['module'] = 'agriculture'
    df_ssr_liv['lever'] = lever
    df_ssr_liv['level'] = 0
    cols = df_ssr_liv.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    df_ssr_liv = df_ssr_liv[cols]
    df_ssr_liv = df_ssr_liv.drop_duplicates()

    df_ssr_feed['module'] = 'agriculture'
    df_ssr_feed['lever'] = lever
    df_ssr_feed['level'] = 0
    cols = df_ssr_feed.columns.tolist()
    cols.insert(cols.index('value'), cols.pop(cols.index('module')))
    cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
    cols.insert(cols.index('value'), cols.pop(cols.index('level')))
    df_ssr_feed = df_ssr_feed[cols]
    df_ssr_feed = df_ssr_feed.drop_duplicates()

    # Extrapolation
    df_ssr_liv = linear_fitting_ots_db(df_ssr_liv, years_ots, countries='all')

    # Format as datamatrix - SSR liv
    df_ots, df_fts = database_to_df(df_ssr_liv, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm_ssr_liv = DataMatrix.create_from_df(df_ots, num_cat=1)
    linear_fitting(dm_ssr_liv, years_ots)

    # Format as datamatrix - SSR feed
    df_ots, df_fts = database_to_df(df_ssr_feed, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm_ssr_feed = DataMatrix.create_from_df(df_ots, num_cat=1)
    linear_fitting(dm_ssr_feed, years_ots)

    return dm_ssr_liv, dm_ssr_feed, df_csl_feed, df_ffr_milk


# CalculationLeaf FXA - SHARE EXPORTS
def exports_processing(list_countries_calc, file_dict):
    # Read data ------------------------------------------------------------------------------------------------------------
    try:
        df_exports = pd.read_csv(file_dict['exports'])
    except OSError:

        # FOOD BALANCE SHEETS (FBS) - For everything except molasses and cakes -------------------------------------------------
        # List of elements
        list_elements = ['Production Quantity', 'Export Quantity']

        list_items = ['Milk - Excluding Butter + (Total)', 'Eggs + (Total)',
                      'Bovine Meat', 'Meat, Other', 'Pigmeat',
                      'Poultry Meat', 'Mutton & Goat Meat']

        # 1990 - 2013
        ld = faostat.list_datasets()
        code = 'FBSH'
        pars = faostat.list_pars(code)
        my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries_calc]
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
            df_ssr_1990_2013['Element'].str.contains('Export Quantity', case=False, na=False), 'Element'] = 'Export'

        # 2010 - 2022

        list_elements = ['Production Quantity', 'Import quantity', 'Export quantity', 'Feed', 'Processed', 'Stock Variation', 'Food', 'Other uses (non-food)', 'Residuals']
        # Different list becuse different in item nomination such as rice
        list_items = ['Milk - Excluding Butter + (Total)', 'Eggs + (Total)',
                      'Bovine Meat', 'Meat, Other', 'Pigmeat',
                      'Poultry Meat', 'Mutton & Goat Meat']
        code = 'FBS'
        my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries_calc]
        my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
        my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
        list_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
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
          df_ssr['Item'].str.contains('Rice (Milled Equivalent)', case=False,
                                      na=False, regex=False),'Item'] = 'Rice and products'

        df_exports = df_ssr.copy()
        df_exports.to_csv(file_dict['exports'], index=False)

    # Filtering to keep wanted columns
    columns_to_filter = ['Area', 'Element', 'Item', 'Year', 'Value']
    df_exports = df_exports[columns_to_filter]

    # Compute the ratio of exports compared to imports ---------------------------------------------------------------------------------
    # 1: Pivot the DataFrame to get 'Production', 'Import Quantity', and 'Export Quantity' in separate columns
    pivot_df = df_exports.pivot_table(index=['Area', 'Year', 'Item'], columns='Element', values='Value').reset_index()

    # Fill na with 0
    cols = [
      'Production', 'Export'
    ]

    for c in cols:
      pivot_df[c] = pivot_df[c].fillna(0.0)


    # 2: Compute the ratio of exports compared to imports
    pivot_df['value'] = pivot_df['Export'] / pivot_df['Production']

    # Filter columns
    columns_to_filter = ['Area', 'Year', 'Item', 'value']
    pivot_df = pivot_df[columns_to_filter]

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------

    # Food item name matching with dictionary
    # Read excel file
    df_dict_exports = pd.read_excel(
        'dictionaries/dictionnary_livestock.xlsx',
        sheet_name='exports')

    # Renaming existing columns (geoscale, timsecale, value)
    pivot_df.rename(columns={'Area': 'geoscale', 'Year': 'timescale'}, inplace=True)

    # Merge based on 'Item'
    df_exports = pd.merge(df_dict_exports, pivot_df, on='Item')

    # Drop the 'Item' column
    df_exports = df_exports.drop(columns=['Item'])

    # Adding the columns module, lever, level and string-pivot at the correct places
    lever = 'food-net-import'
    df_exports['module'] = 'agriculture'
    df_exports['lever'] = lever
    df_exports['level'] = 0

    # Extrapolation
    df_exports = linear_fitting_ots_db(df_exports, years_all, countries='all')

    # Format as datamatrix
    df_ots, df_fts = database_to_df(df_exports, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm_fxa_exports = DataMatrix.create_from_df(df_ots, num_cat=1)


    return dm_fxa_exports


# CalculationLeaf TRADE ORIGIN
def trade_origin_processing(years_ots, list_countries_calc, file_dict):
  # Read data ------------------------------------------------------------------------------------------------------------
  list_partnerregions = ['-- Australia and New Zealand > (List)',
                         '-- Caribbean > (List)',
                         '-- Central America > (List)',
                         '-- Central Asia > (List)',
                         '-- Eastern Africa > (List)',
                         '-- Eastern Asia > (List)',
                         '-- Eastern Europe > (List)',
                         '-- Melanesia > (List)',
                         '-- Micronesia > (List)',
                         '-- Middle Africa > (List)',
                         '-- Northern Africa > (List)',
                         '-- Northern America > (List)',
                         '-- Northern Europe > (List)',
                         '-- Polynesia > (List)',
                         '-- South America > (List)',
                         '-- South-eastern Asia > (List)',
                         '-- Southern Africa > (List)',
                         '-- Southern Asia > (List)',
                         '-- Southern Europe > (List)',
                         '-- Western Africa > (List)',
                         '-- Western Asia > (List)',
                         '-- Western Europe > (List)']

  try:
    df_trade = pd.read_csv(file_dict['trade'])
  except OSError:

    # TRADE MATRIX (TM)
    # List of elements
    list_elements = ['Import quantity']

    list_items = ['Milk - Excluding Butter + (Total)', 'Eggs + (Total)',
                  'Bovine Meat', 'Meat, Other', 'Pigmeat',
                  'Poultry Meat', 'Mutton & Goat Meat']
    list_items = ['Raw milk of cattle',
                  'Meat of cattle boneless, fresh or chilled',
                  'Meat of cattle with the bone, fresh or chilled',
                  'Meat of asses, fresh or chilled',
                  'Meat of buffalo, fresh or chilled',
                  'Meat of camels, fresh or chilled',
                  'Meat of pig boneless, fresh or chilled',
                  'Meat of pig with the bone, fresh or chilled',
                  'Meat of chickens, fresh or chilled',
                  'Meat of turkeys, fresh or chilled',
                  'Meat of ducks, fresh or chilled',
                  'Meat of geese, fresh or chilled',
                  'Meat of rabbits and hares, fresh or chilled',
                  'Meat of goat, fresh or chilled',
                  'Meat of sheep, fresh or chilled',
                  'Hen eggs in shell, fresh']

    # 1990 - 2023
    ld = faostat.list_datasets()
    code = 'TM'
    pars = faostat.list_pars(code)
    my_reporter_countries = [faostat.get_par(code, 'reporterarea')[c] for c in list_countries_calc]
    my_partner_regions = [faostat.get_par(code, 'partnerregions')[p] for p in
                             list_partnerregions]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996',
                  '1997', '1998', '1999', '2000', '2001', '2002',
                  '2003', '2004', '2005', '2006', '2007', '2008', '2009',
                  '2010', '2011', '2012', '2013', '2014', '2015', '2016',
                  '2017', '2018', '2019', '2020', '2021', '2022', '2023']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
      'reporterarea': my_reporter_countries,
      'partnerregions': my_partner_regions,
      'element': my_elements,
      'item': my_items,
      'year': my_years
    }
    df_trade = faostat.get_data_df(code, pars=my_pars, strval=False)
    df_trade.to_csv(file_dict['trade'], index=False)

  # Filter
  df_trade = df_trade[
    ['Reporter Countries', 'Partner Countries', 'Item', 'Year', 'Value']]

  # Aggregate by item ----------------------------------------------------------
  mapping = {
    'Pig': 'Pig',
    'milk': 'Milk',
    'cattle': 'Cattle',
    'Buffalo': 'Cattle',
    'Chicken': 'Chicken',
    'Duck': 'Duck',
    'Turkeys': 'Turkey',
    'Geese': 'Goose',
    'Pigeon': 'Pigeon',
    'Horse': 'Horse',
    'Rabbits and hares': 'Rabbit',
    'Sheep': 'Sheep',
    'Goat': 'Goat',
    'Asse': 'Asse',
    'Camel': 'Other non-specified',
    'Rodent': 'Other non-specified',
    'Other': 'Other non-specified',
    'Game': 'Game',
    'Mule': 'Mule',
    'Hen eggs in shell, fresh': 'Eggs'
  }

  for key, value in mapping.items():
    mask = df_trade['Item'].str.contains(key, case=False,
                                                         na=False)
    df_trade.loc[mask, 'Item'] = value

    # Reading excel lsu equivalent
  df_lsu = pd.read_excel(
        'dictionaries/lsu_equivalent.xlsx',
        sheet_name='lsu_equivalent')
  # Merging
  df_trade_agg = pd.merge(df_trade, df_lsu, on='Item')

  # Aggregating
  df_trade_agg['Value'] = df_trade_agg['Value'].fillna(0.0)
  df_trade_agg = df_trade_agg.groupby(['variables', 'Partner Countries', 'Reporter Countries', 'Year'], as_index=False)['Value'].sum()

  # Aggregate by countries -----------------------------------------------------

  # Read csv
  df_countries = pd.read_csv('data/faostat/FAOSTAT_data_partner-countries-regions.csv')

  # Filter the regions
  clean_regions = [x.replace('-- ', '').replace(' > (List)', '') for x in list_partnerregions]
  mask = df_countries['Partner Country Group'].str.contains('|'.join(clean_regions),
                                                  case=False, na=False)
  df_countries = df_countries[mask].copy()
  df_countries = df_countries[['Partner Country Group', 'Partner Countries']]

  # Merge
  df_trade_agg = pd.merge(df_trade_agg, df_countries, on='Partner Countries')

  # Aggregating
  df_trade_agg = df_trade_agg.groupby(['variables', 'Partner Country Group', 'Year'], as_index=False)['Value'].sum()

  df_trade_agg.rename(columns={'Partner Country Group': 'geoscale',
                               'Year': 'timescale', 'Value':'value'}, inplace=True)

  # Extrapolation for missing data
  lever = 'dummy'
  df_trade_agg['lever'] = lever
  df_trade_agg['module'] = lever
  df_trade_agg['level'] = 0.0
  df_trade_agg = ensure_structure(df_trade_agg)
  df_trade_agg = linear_fitting_ots_db(df_trade_agg, years_all, countries='all')

  # Replace negative values by 0.0
  df_trade_agg['value'] = df_trade_agg['value'].clip(lower=0.0)

  # Format as datamatrix
  df_ots, df_fts = database_to_df(df_trade_agg, lever, level='all')
  df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
  dm_liv_trade_origin = DataMatrix.create_from_df(df_ots, num_cat=1)

  # Add Switzerland and Melanasia as dummy (because are in losses and other dms)
  dm_liv_trade_origin.add(0.0, dummy=True, col_label=['Switzerland'], dim='Country')
  dm_liv_trade_origin.add(0.0, dummy=True, col_label=['Melanesia'],
                          dim='Country')

  # Unit conversion: [t] => [kcal]
  cdm_kcal_temp = cdm_kcal.copy()
  cdm_kcal_temp.rename_col_regex(str1="pro-liv-", str2="", dim="Categories1")
  cdm_kcal_temp = cdm_kcal_temp.filter({'Categories1': ['abp-dairy-milk', 'abp-hens-egg',
                                              'meat-bovine', 'meat-oth-animal',
                                              'meat-pig', 'meat-poultry',
                                              'meat-sheep']})
  dm_liv_trade_origin.sort('Categories1')
  cdm_kcal_temp.sort('Categories1')
  array_temp = dm_liv_trade_origin[:, :, 'agr_split-import', :] \
               * cdm_kcal_temp[np.newaxis, np.newaxis, 'cp_kcal-per-t', :]
  dm_liv_trade_origin[:, :, 'agr_split-import', :] = array_temp

  # Step CALIBRATION IMPORTS
  dm_cal_imports = dm_liv_trade_origin.copy()
  dm_cal_imports.rename_col('agr_split-import', 'cal_agr_domestic-production','Variables')
  dm_cal_imports.change_unit('cal_agr_domestic-production', 1.0, '-', 'kcal', '*')
  dm_cal_imports.drop(dim='Country', col_label=['Switzerland'])

  # Normalise across countries for share of imports
  dm_liv_trade_origin.normalise(dim='Country', inplace=True)
  dm_liv_trade_origin.change_unit('agr_split-import', 1.0, '%', '-', '*')

  return dm_liv_trade_origin, dm_cal_imports

# CalculationLeaf SHARE PRODUCTION METHOD

def production_share(dm_cal_liv_pop):

  # Step CATTLE (DAIRY & MEAT)
  # Source: STAT-TAB Exploitations agricoles et animaux de rente selon le niveau de classification 3 par canton
  # https://www.pxweb.bfs.admin.ch/pxweb/fr/px-x-0702000000_108/-/px-x-0702000000_108.px/

  table_id = 'px-x-0702000000_108'
  file = 'data/stat-tab/livestock_cattle.pickle'
  try:
    with open(file, 'rb') as handle:
      dm_cattle = pickle.load(handle)
      print(
        f'The livestock units are read from file {file}. Delete it if you want to update data from api.')
  except OSError:
    structure, title = get_data_api_CH(table_id, mode='example', language='fr')
    i = 0
    filtering = {
      "Unité d'observation": ['Cheptel - Vaches laitières',
                              'Cheptel - Autres vaches',
                              'Cheptel - Veaux et autres bovins - de 1 an',
                              'Cheptel - Autres bovins'],
      'Canton': ['Suisse'],
      'Zone de production agricole': ['Zone de production agricole - total'],
      'Classe de taille': ['Classe de taille - total'],
      "Système d'exploitation": ['Système d\'exploitation - total', 'Exploitations biologiques'],
      "Forme d'exploitation": ["Forme d'exploitation - total"],
      'Année': structure['Année']}

    mapping_dim = {'Country': 'Canton',
                   'Years': 'Année',
                   'Variables': 'Zone de production agricole',
                   'Categories1': "Unité d'observation",
                   'Categories2': "Système d'exploitation"}

    # Extract data
    dm = get_data_api_CH(table_id, mode='extract', filter=filtering,
                         mapping_dims=mapping_dim,
                         units=['animals'], language='fr')
    # Format
    dm.rename_col('Suisse', 'Switzerland',
                  'Country')
    dm.rename_col('Zone de production agricole - total', 'agr_livestock',
                  'Variables')
    dm.rename_col('Système d\'exploitation - total', 'total',
                  'Categories2')
    dm.rename_col('Exploitations biologiques', 'organic',
                  'Categories2')
    dm.rename_col_regex('Cheptel - ', '', dim='Categories1')
    dict_cat = {'abp-dairy-milk': ['Vaches laitières'],
                'other-cattle': ['Autres vaches'],
                'young-cattle': ['Veaux et autres bovins - de 1 an'],
                'other-bovins': ['Autres bovins']}
    dm.groupby(dict_cat, dim='Categories1', inplace=True)
    dm.sort('Years')
    dm.filter({'Years': years_ots}, inplace=True)
    linear_fitting(dm, years_ots)
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, file)
    with open(f, 'wb') as handle:
      pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)
    dm_cattle = dm.copy()

  # For abp-dairy-milk & other-cattle: extrapolate for missing years (before 1999)
  years_temp = create_years_list(1999, 2023, 1)
  dm_temp = dm_cattle.filter({'Years': years_temp, 'Categories1': ['abp-dairy-milk', 'other-cattle']}, inplace=False)
  linear_fitting(dm_temp, years_ots)
  dm_cattle[:,:,:,'abp-dairy-milk',:] = dm_temp[:,:,:,'abp-dairy-milk',:]
  dm_cattle[:, :, :, 'other-cattle', :] = dm_temp[:, :, :, 'other-cattle',:]

  # Convert animals to lsu based on EUCALC doc
  # Note: we account for young cattle same as cattle to match FAOSTAT
  lsu_conversion = {'other-cattle': 0.6,
                    'young-cattle': 0.6,
                    'abp-dairy-milk': 0.7,
                    'other-bovins': 0.6}

  for cat in dm_cattle.col_labels['Categories1']:
    dm_cattle[:, :, 'agr_livestock', cat, :] = lsu_conversion[cat] \
                                            * dm_cattle[:, :,'agr_livestock', cat, :]
  dm_cattle.change_unit('agr_livestock', old_unit='animals', new_unit='lsu',
                     factor=1)

  # Sum cattle meat = young-cattle + other-cattle + other-bovins
  dm_cattle.groupby({'meat-bovine': ['young-cattle', 'other-cattle', 'other-bovins']}, dim='Categories1', inplace=True)


  # Step POULTRY (LAYERS & BROILERS)
  table_id = 'px-x-0702000000_108'
  file = 'data/stat-tab/livestock_poultry.pickle'
  try:
    with open(file, 'rb') as handle:
      dm_poultry = pickle.load(handle)
      print(
        f'The livestock units are read from file {file}. Delete it if you want to update data from api.')
  except OSError:
    structure, title = get_data_api_CH(table_id, mode='example', language='fr')
    i = 0
    filtering = {
      "Unité d'observation": ['Cheptel - Poules de ponte et d\'élevage',
                              'Cheptel - Poulets de chair',
                              'Cheptel - Dindes',
                              'Cheptel - Canards',
                              'Cheptel - Oies'],
      'Canton': ['Suisse'],
      'Zone de production agricole': ['Zone de production agricole - total'],
      'Classe de taille': ['Classe de taille - total'],
      "Système d'exploitation": ['Système d\'exploitation - total', 'Exploitations biologiques'],
      "Forme d'exploitation": ["Forme d'exploitation - total"],
      'Année': structure['Année']}

    mapping_dim = {'Country': 'Canton',
                   'Years': 'Année',
                   'Variables': 'Zone de production agricole',
                   'Categories1': "Unité d'observation",
                   'Categories2': "Système d'exploitation"}

    # Extract data
    dm = get_data_api_CH(table_id, mode='extract', filter=filtering,
                         mapping_dims=mapping_dim,
                         units=['animals'], language='fr')
    # Format
    dm.rename_col('Suisse', 'Switzerland',
                  'Country')
    dm.rename_col('Zone de production agricole - total', 'agr_livestock',
                  'Variables')
    dm.rename_col('Système d\'exploitation - total', 'total',
                  'Categories2')
    dm.rename_col('Exploitations biologiques', 'organic',
                  'Categories2')
    dm.rename_col_regex('Cheptel - ', '', dim='Categories1')
    dict_cat = {'abp-hens-egg': ['Poules de ponte et d\'élevage'],
                'meat-poultry': ['Poulets de chair'],
                'meat-other-poultry': ['Dindes',
                                       'Canards',
                                       'Oies']}
    dm.groupby(dict_cat, dim='Categories1', inplace=True)
    dm.sort('Years')
    dm.filter({'Years': years_ots}, inplace=True)
    linear_fitting(dm, years_ots)
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, file)
    with open(f, 'wb') as handle:
      pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)
    dm_poultry = dm.copy()

  # Convert animals to lsu
  # young-cattle from https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Glossary:Livestock_unit_(LSU),
  # else based on EUCALC doc
  lsu_conversion = {'abp-hens-egg': 0.014,
                    'meat-poultry': 0.007,
                    'meat-other-poultry': 0.03}

  for cat in dm_poultry.col_labels['Categories1']:
    dm_poultry[:, :, 'agr_livestock', cat, :] = lsu_conversion[cat] \
                                            * dm_poultry[:, :,'agr_livestock', cat, :]
  dm_poultry.change_unit('agr_livestock', old_unit='animals', new_unit='lsu',
                     factor=1)

  # Sum poultry meat = young-cattle + other-cattle + other-bovins
  dm_poultry.groupby({'meat-poultry': ['meat-poultry', 'meat-other-poultry']}, dim='Categories1', inplace=True)

  # Step OTHERS (PIG, SHEEP, OTHERS)
  # Source: Emplois, exploitations agricoles, surface agricole utile (SAU) et animaux de rente selon le niveau de classification 1 par canton
  # https://www.pxweb.bfs.admin.ch/pxweb/fr/px-x-0702000000_101/-/px-x-0702000000_101.px

  table_id = 'px-x-0702000000_101'
  file = 'data/stat-tab/livestock_others.pickle'
  try:
    with open(file, 'rb') as handle:
      dm_others = pickle.load(handle)
      print(
        f'The livestock units are read from file {file}. Delete it if you want to update data from api.')
  except OSError:
    structure, title = get_data_api_CH(table_id, mode='example', language='fr')
    i = 0
    filtering = {
      "Unité d'observation": ['Cheptel - Equidés',
                              'Cheptel - Moutons',
                              'Cheptel - Chèvres',
                              'Cheptel - Porcs',
                              'Cheptel - Autres animaux'],
      'Canton': ['Suisse'],
      'Zone de production agricole': ['Zone de production agricole - total'],
      'Classe de taille': ['Classe de taille - total'],
      "Système d'exploitation": ['Système d\'exploitation - total', 'Exploitations biologiques'],
      "Forme d'exploitation": ["Forme d'exploitation - total"],
      'Année': structure['Année']}

    mapping_dim = {'Country': 'Canton',
                   'Years': 'Année',
                   'Variables': 'Zone de production agricole',
                   'Categories1': "Unité d'observation",
                   'Categories2': "Système d'exploitation"}

    # Extract data
    dm = get_data_api_CH(table_id, mode='extract', filter=filtering,
                         mapping_dims=mapping_dim,
                         units=['animals'], language='fr')
    # Format
    dm.rename_col('Suisse', 'Switzerland',
                  'Country')
    dm.rename_col('Zone de production agricole - total', 'agr_livestock',
                  'Variables')
    dm.rename_col('Système d\'exploitation - total', 'total',
                  'Categories2')
    dm.rename_col('Exploitations biologiques', 'organic',
                  'Categories2')
    dm.rename_col_regex('Cheptel - ', '', dim='Categories1')
    dict_cat = {'meat-sheep': ['Moutons', 'Chèvres'],
                'meat-pig': ['Porcs'],
                'meat-horse': ['Equidés'],
                'meat-oth-animal': ['Autres animaux']}
    dm.groupby(dict_cat, dim='Categories1', inplace=True)
    dm.sort('Years')
    dm.filter({'Years': years_ots}, inplace=True)
    linear_fitting(dm, years_ots)
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, file)
    with open(f, 'wb') as handle:
      pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)
    dm_others = dm.copy()

  # Convert animals to lsu
  # young-cattle from https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Glossary:Livestock_unit_(LSU),
  # else based on EUCALC doc
  lsu_conversion = {'meat-sheep': 0.1,
                    'meat-pig': 0.22,
                    'meat-horse': 0.8,
                    'meat-oth-animal': 0.03}

  for cat in dm_others.col_labels['Categories1']:
    dm_others[:, :, 'agr_livestock', cat, :] = lsu_conversion[cat] \
                                            * dm_others[:, :,'agr_livestock', cat, :]
  dm_others.change_unit('agr_livestock', old_unit='animals', new_unit='lsu',
                     factor=1)

  # Group Horses and other animals
  dm_others.groupby({'meat-oth-animal': ['meat-horse', 'meat-oth-animal']}, dim='Categories1', inplace=True)

  # Append dms
  dm_cattle.append(dm_poultry, dim='Categories1')
  dm_cattle.append(dm_others, dim='Categories1')
  dm_prod_share = dm_cattle.copy()

  # Step CAL ORGANIC LIVESTOCK
  # Create copy for calibration
  dm_cal_liv_pop_org = dm_prod_share.filter({'Categories2': ['organic']})
  dm_cal_liv_pop_org.switch_categories_order(cat1='Categories2', cat2='Categories1')
  dm_cal_liv_pop_org = dm_cal_liv_pop_org.flatten()
  dm_cal_liv_pop_org.rename_col_regex('organic_', '', dim='Categories1')
  dm_cal_liv_pop_org.rename_col_regex('agr_livestock', 'cal_agr_liv-population_organic', dim='Variables')

  # Compute share of organic production compared to total
  dm_prod_share.switch_categories_order(cat1='Categories2', cat2='Categories1')
  dm_prod_share = dm_prod_share.flattest()
  dm_prod_share.deepen()
  dm_prod_share.operation('agr_livestock_organic', '/',
                            'agr_livestock_total',
                            out_col='livestock_share-organic', unit='-')

  # Filter
  dm_prod_share.filter({'Variables': ['livestock_share-organic']}, inplace=True)

  return dm_prod_share, dm_cal_liv_pop_org

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

# CalculationLeaf LIVESTOCK DENSITY & GRAZING INTENSITY ------------------------------------------------------------------------------

def livestock_density(df_liv_pop):
  # Read FAO Values (for Switzerland) --------------------------------------------------------------------------------------------

  # List of elements
  list_elements = ['Area']

  list_items = ['-- Cropland', '--- Temporary crops', '--- Temporary fallow',
                '-- Permanent meadows and pastures']

  # 1990 - 2022
  ld = faostat.list_datasets()
  code = 'RL'
  pars = faostat.list_pars(code)
  my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries_calc]
  my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
  my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
  list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997',
                '1998', '1999', '2000', '2001',
                '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',
                '2010', '2011', '2012', '2013',
                '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021',
                '2022']
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
  df_cropland_density = df_land_use_fao[df_land_use_fao['Item'].isin(
    ['Cropland', 'Permanent meadows and pastures'])]
  df_cropland_density = df_cropland_density.pivot_table(
    index=['Area', 'Year'],
    columns='Item',
    values='Value'
  ).reset_index()

  # Filter grazing ruminant livestock (cattle meat, sheep, goats) and sum per year
  df_ruminant = df_liv_pop[df_liv_pop['Item'].isin(
    ['Cattle, dairy', 'Cattle, non-dairy', 'Sheep and Goats'])]
  df_ruminant = df_ruminant.groupby(['Area', 'Year'], as_index=False)[
    'Value'].sum()

  # Merge with cropland_density
  df_ruminant = pd.merge(df_ruminant, df_cropland_density, on=['Area', 'Year'])

  # Compute livestock density of ruminant per area of permanent meadows and pastures
  df_ruminant['Livestock density [lsu/ha]'] = df_ruminant['Value'] / \
                                              df_ruminant[
                                                'Permanent meadows and pastures']

  # Filter and add column density
  df_ruminant = df_ruminant[['Year', 'Area', 'Livestock density [lsu/ha]']]

  # Adding an Item column for name
  df_ruminant['Item'] = 'Density'


  # PathwayCalc formatting -----------------------------------------------------------------------------------------------

  # Renaming into 'Value'
  df_ruminant.rename(columns={'Area': 'geoscale', 'Year': 'timescale',
                              'Livestock density [lsu/ha]': 'value'},
                     inplace=True)

  # Read excel file
  df_dict_csl = pd.read_excel(
    'dictionaries/dictionnary_livestock.xlsx',
    sheet_name='climate-smart-livestock')

  # Merge based on 'Item'
  df_csl_density_pathwaycalc = pd.merge(df_dict_csl, df_ruminant, on='Item')

  # Drop the 'Item' column
  df_csl_density_pathwaycalc = df_csl_density_pathwaycalc.drop(columns=['Item'])

  # Adding the columns module, lever, level and string-pivot at the correct places
  df_csl_density_pathwaycalc['module'] = 'agriculture'
  lever = 'dummy'
  df_csl_density_pathwaycalc['lever'] = lever
  df_csl_density_pathwaycalc['level'] = 0
  cols = df_csl_density_pathwaycalc.columns.tolist()
  cols.insert(cols.index('value'), cols.pop(cols.index('module')))
  cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
  cols.insert(cols.index('value'), cols.pop(cols.index('level')))
  df_csl_density_pathwaycalc = df_csl_density_pathwaycalc[cols]

  # Rename countries to Pathaywcalc name
  df_csl_density_pathwaycalc['geoscale'] = df_csl_density_pathwaycalc[
    'geoscale'].replace(
    'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
  df_csl_density_pathwaycalc['geoscale'] = df_csl_density_pathwaycalc[
    'geoscale'].replace(
    'Netherlands (Kingdom of the)',
    'Netherlands')
  df_csl_density_pathwaycalc['geoscale'] = df_csl_density_pathwaycalc[
    'geoscale'].replace('Czechia', 'Czech Republic')

  # Format as datamatrix
  df_ots, df_fts = database_to_df(df_csl_density_pathwaycalc, lever, level='all')
  df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
  dm_density = DataMatrix.create_from_df(df_ots, num_cat=0)

  return dm_density

# CalculationLeaf LIVESTOCK EMISSIONS ------------------------------------------------------------------------------
def livestock_emissions():
  # ----------------------------------------------------------------------------------------------------------------------
  # ENTERIC EMISSIONS ----------------------------------------------------------------------------------------------------
  # ----------------------------------------------------------------------------------------------------------------------
  list_elements = ['Enteric fermentation (Emissions CH4)',
                   'Manure management (Emissions CH4)', 'Stocks']

  list_items = ['All Animals > (List)']
  list_sources = ['FAO TIER 1']

  # 1990 - 2021
  code = 'GLE'
  my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries_calc]
  my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
  my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
  my_sources = [faostat.get_par(code, 'sources')[i] for i in list_sources]
  list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997',
                '1998', '1999', '2000', '2001',
                '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',
                '2010', '2011', '2012', '2013',
                '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021',
                '2022', '2023']
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
    df_enteric_1990_2021['Item'].str.contains('Cattle, dairy', case=False,
                                              na=False), 'Item'] = 'Dairy cows'
  df_enteric_1990_2021.loc[
    df_enteric_1990_2021['Item'].str.contains('Cattle, non-dairy', case=False,
                                              na=False), 'Item'] = 'Cattle'
  df_enteric_1990_2021.loc[
    df_enteric_1990_2021['Item'].str.contains('Goat', case=False,
                                              na=False), 'Item'] = 'Goat'
  df_enteric_1990_2021.loc[
    df_enteric_1990_2021['Item'].str.contains('Chickens, broilers', case=False,
                                              na=False), 'Item'] = 'Chicken'
  df_enteric_1990_2021.loc[
    df_enteric_1990_2021['Item'].str.contains('Chickens, layers', case=False,
                                              na=False), 'Item'] = 'Chicken laying hens'
  df_enteric_1990_2021.loc[
    df_enteric_1990_2021['Item'].str.contains('Duck', case=False,
                                              na=False), 'Item'] = 'Duck'
  df_enteric_1990_2021.loc[
    df_enteric_1990_2021['Item'].str.contains('Horse', case=False,
                                              na=False), 'Item'] = 'Horse'
  df_enteric_1990_2021.loc[
    df_enteric_1990_2021['Item'].str.contains('Sheep', case=False,
                                              na=False), 'Item'] = 'Sheep'
  df_enteric_1990_2021.loc[
    df_enteric_1990_2021['Item'].str.contains('Swine', case=False,
                                              na=False), 'Item'] = 'Pig'
  df_enteric_1990_2021.loc[
    df_enteric_1990_2021['Item'].str.contains('Turkey', case=False,
                                              na=False), 'Item'] = 'Turkey'
  df_enteric_1990_2021.loc[
    df_enteric_1990_2021['Item'].str.contains('Asse', case=False,
                                              na=False), 'Item'] = 'Asse'
  df_enteric_1990_2021.loc[
    df_enteric_1990_2021['Item'].str.contains('Buffalo', case=False,
                                              na=False), 'Item'] = 'Buffalo'
  df_enteric_1990_2021.loc[
    df_enteric_1990_2021['Item'].str.contains('Mule', case=False,
                                              na=False), 'Item'] = 'Mule'
  df_enteric_1990_2021.loc[
    df_enteric_1990_2021['Item'].str.contains('Camel', case=False,
                                              na=False), 'Item'] = 'Other non-specified'

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
    df_enteric_1990_2021.groupby(
      ['Aggregation', 'Area', 'Year', 'Element', 'Unit'], as_index=False)[
      'Value'].sum()

  # Pivot the df
  pivot_df = df_enteric_1990_2021_grouped.pivot_table(
    index=['Area', 'Year', 'Aggregation'], columns='Element',
    values='Value').reset_index()

  # Enteric emissions CH4 [t/lsu] = 1000 * 'Enteric fermentation (Emissions CH4) [kt]'/ 'Stocks [lsu]'
  pivot_df['Enteric emissions CH4 [t/lsu]'] = 1000 * pivot_df[
    'Enteric fermentation (Emissions CH4)'] / pivot_df[
                                                'Stocks']

  # Create duplicate for fxa
  df_manure_ch4_fxa = pivot_df.copy()
  df_manure_ch4_fxa['Manure emissions CH4 [t/lsu]'] = 1000 * df_manure_ch4_fxa[
    'Manure management (Emissions CH4)'] / df_manure_ch4_fxa[
                                                        'Stocks']
  df_manure_ch4_fxa = df_manure_ch4_fxa[
    ['Area', 'Year', 'Aggregation', 'Manure emissions CH4 [t/lsu]']].copy()

  # Drop the columns 'Enteric fermentation (Emissions CH4)' 'Stocks'
  pivot_df = pivot_df.drop(columns=['Enteric fermentation (Emissions CH4)',
                                    'Manure management (Emissions CH4)',
                                    'Stocks'])

  # PathwayCalc formatting -----------------------------------------------------------------------------------------------

  # Renaming into 'Value'
  pivot_df.rename(columns={'Area': 'geoscale', 'Year': 'timescale',
                           'Enteric emissions CH4 [t/lsu]': 'value'},
                  inplace=True)

  # Food item name matching with dictionary
  # Read excel file
  df_dict_csl_enteric = pd.read_excel(
    'dictionaries/dictionnary_livestock.xlsx',
    sheet_name='climate-smart-livestock_enteric')

  # Merge based on 'Item' & 'Aggregation'
  df_enteric_pathwaycalc = pd.merge(df_dict_csl_enteric, pivot_df,
                                    left_on='Item', right_on='Aggregation')

  # Drop the 'Item' column
  df_enteric_pathwaycalc = df_enteric_pathwaycalc.drop(
    columns=['Item', 'Aggregation'])

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
  df_enteric_pathwaycalc['geoscale'] = df_enteric_pathwaycalc[
    'geoscale'].replace(
    'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
  df_enteric_pathwaycalc['geoscale'] = df_enteric_pathwaycalc[
    'geoscale'].replace('Netherlands (Kingdom of the)',
                        'Netherlands')
  df_enteric_pathwaycalc['geoscale'] = df_enteric_pathwaycalc[
    'geoscale'].replace('Czechia', 'Czech Republic')

  # Format as datamatrix
  lever = 'dummy'
  df_enteric_pathwaycalc['lever'] = lever
  df_enteric_pathwaycalc['level'] = 0.0
  df_ots, df_fts = database_to_df(df_enteric_pathwaycalc, lever,
                                  level='all')
  df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
  dm_enteric = DataMatrix.create_from_df(df_ots, num_cat=2)

  # ----------------------------------------------------------------------------------------------------------------------
  # MANURE EMISSIONS (APPLIED, PASTURE & TREATED) ------------------------------------------------------------------------
  # ----------------------------------------------------------------------------------------------------------------------
  list_elements = ['Amount excreted in manure (N content)',
                   'Manure left on pasture (N content)',
                   'Manure applied to soils (N content)',
                   'Losses from manure treated (N content)']

  list_items = ['All Animals > (List)']

  # 1990 - 2022
  code = 'EMN'
  my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries_calc]
  my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
  my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
  list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997',
                '1998', '1999', '2000', '2001',
                '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',
                '2010', '2011', '2012', '2013',
                '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021',
                '2022']
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
    df_manure_1990_2021['Item'].str.contains('Cattle, dairy', case=False,
                                             na=False), 'Item'] = 'Dairy cows'
  df_manure_1990_2021.loc[
    df_manure_1990_2021['Item'].str.contains('Cattle, non-dairy', case=False,
                                             na=False), 'Item'] = 'Cattle'
  df_manure_1990_2021.loc[
    df_manure_1990_2021['Item'].str.contains('Goat', case=False,
                                             na=False), 'Item'] = 'Goat'
  df_manure_1990_2021.loc[
    df_manure_1990_2021['Item'].str.contains('Chickens, broilers', case=False,
                                             na=False), 'Item'] = 'Chicken'
  df_manure_1990_2021.loc[
    df_manure_1990_2021['Item'].str.contains('Chickens, layers', case=False,
                                             na=False), 'Item'] = 'Chicken laying hens'
  df_manure_1990_2021.loc[
    df_manure_1990_2021['Item'].str.contains('Duck', case=False,
                                             na=False), 'Item'] = 'Duck'
  df_manure_1990_2021.loc[
    df_manure_1990_2021['Item'].str.contains('Horse', case=False,
                                             na=False), 'Item'] = 'Horse'
  df_manure_1990_2021.loc[
    df_manure_1990_2021['Item'].str.contains('Sheep', case=False,
                                             na=False), 'Item'] = 'Sheep'
  df_manure_1990_2021.loc[
    df_manure_1990_2021['Item'].str.contains('Swine', case=False,
                                             na=False), 'Item'] = 'Pig'
  df_manure_1990_2021.loc[
    df_manure_1990_2021['Item'].str.contains('Turkey', case=False,
                                             na=False), 'Item'] = 'Turkey'
  df_manure_1990_2021.loc[
    df_manure_1990_2021['Item'].str.contains('Asse', case=False,
                                             na=False), 'Item'] = 'Asse'
  df_manure_1990_2021.loc[
    df_manure_1990_2021['Item'].str.contains('Buffalo', case=False,
                                             na=False), 'Item'] = 'Buffalo'
  df_manure_1990_2021.loc[
    df_manure_1990_2021['Item'].str.contains('Mule', case=False,
                                             na=False), 'Item'] = 'Mule'
  df_manure_1990_2021.loc[
    df_manure_1990_2021['Item'].str.contains('Camel', case=False,
                                             na=False), 'Item'] = 'Other non-specified'

  # Reading excel lsu equivalent (for aggregation)
  df_lsu = pd.read_excel(
    'dictionaries/lsu_equivalent.xlsx',
    sheet_name='lsu_equivalent')
  # Merging
  df_manure_1990_2021 = pd.merge(df_manure_1990_2021, df_lsu, on='Item')

  # Aggregating
  df_manure_1990_2021 = \
    df_manure_1990_2021.groupby(
      ['Aggregation', 'Area', 'Year', 'Element', 'Unit'], as_index=False)[
      'Value'].sum()

  # Pivot the df
  pivot_df = df_manure_1990_2021.pivot_table(
    index=['Area', 'Year', 'Aggregation'], columns='Element',
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
                      left_on=['Area', 'Year', 'Aggregation'],
                      right_on=['Area', 'Year', 'Item'],
                      how='inner')

  # Manure applied/treated/pasture [%] = Manure applied to soil/treated/left on pasture (N content) [kg] / Amount excreted (N content) [kg]

  pivot_df['Manure applied [%]'] = pivot_df[
                                     'Manure applied to soils (N content)'] / \
                                   pivot_df[
                                     'Amount excreted in manure (N content)']
  pivot_df['Manure treated [%]'] = pivot_df[
                                     'Losses from manure treated (N content)'] / \
                                   pivot_df[
                                     'Amount excreted in manure (N content)']
  pivot_df['Manure pasture [%]'] = pivot_df[
                                     'Manure left on pasture (N content)'] / \
                                   pivot_df[
                                     'Amount excreted in manure (N content)']

  # Compute manure yield fxa
  pivot_df['Manure yield [tN/lsu]'] = 10 ** -3 * pivot_df[
    'Amount excreted in manure (N content)'] / pivot_df['Value']

  # Create copy for emission factor per practice

  # Drop the columns
  pivot_df = pivot_df.drop(columns=['Manure applied to soils (N content)',
                                    'Losses from manure treated (N content)',
                                    'Manure left on pasture (N content)',
                                    'Amount excreted in manure (N content)',
                                    'Value', 'Item'])

  # PathwayCalc formatting -----------------------------------------------------------------------------------------------

  # Melt the DataFrame
  df_melted = pd.melt(pivot_df, id_vars=['Area', 'Year', 'Aggregation'],
                      value_vars=['Manure applied [%]', 'Manure treated [%]',
                                  'Manure pasture [%]',
                                  'Manure yield [tN/lsu]'],
                      var_name='Item', value_name='value')

  # Concatenate the aggregation column with the manure column names
  df_melted['Item'] = df_melted['Aggregation'] + ' ' + df_melted['Item']

  # Drop the aggregation column as it's now part of the item column
  df_melted = df_melted.drop(columns=['Aggregation'])

  # Renaming
  df_melted.rename(columns={'Area': 'geoscale', 'Year': 'timescale'},
                   inplace=True)

  # Food item name matching with dictionary
  # Read excel file
  df_dict_csl = pd.read_excel(
    'dictionaries/dictionnary_livestock.xlsx',
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
  df_manure_pathwaycalc['geoscale'] = df_manure_pathwaycalc['geoscale'].replace(
    'Netherlands (Kingdom of the)',
    'Netherlands')
  df_manure_pathwaycalc['geoscale'] = df_manure_pathwaycalc['geoscale'].replace(
    'Czechia', 'Czech Republic')

  # Filter for fxa
  df_fxa_manure_yield = df_manure_pathwaycalc[
    df_manure_pathwaycalc['variables'].str.contains('fxa', case=False,
                                                    na=False)]

  # Drop the rows from original df
  df_manure_pathwaycalc = df_manure_pathwaycalc[
    ~df_manure_pathwaycalc['variables'].str.contains('fxa', case=False,
                                                     na=False)]

  # Format as datamatrix
  lever = 'dummy'
  df_manure_pathwaycalc['lever'] = lever
  df_manure_pathwaycalc['level'] = 0.0
  df_ots, df_fts = database_to_df(df_manure_pathwaycalc, lever,
                                  level='all')
  df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
  dm_manure = DataMatrix.create_from_df(df_ots, num_cat=2)

  # Format as datamatrix
  lever = 'dummy'
  df_fxa_manure_yield['lever'] = lever
  df_fxa_manure_yield['level'] = 0.0
  df_ots, df_fts = database_to_df(df_fxa_manure_yield, lever,
                                  level='all')
  df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
  dm_fxa_manure_yield = DataMatrix.create_from_df(df_ots, num_cat=1)

  return dm_manure, dm_enteric, dm_fxa_manure_yield, df_manure_ch4_fxa, df_manure_n_fxa

# CalculationLeaf LOSSES ------------------------------------------------------------------------------

def livestock_losses():
  # ----------------------------------------------------------------------------------------------------------------------
  # LOSSES ---------------------------------------------------------------------------------------------------------------
  # ----------------------------------------------------------------------------------------------------------------------

  # FOOD BALANCE SHEETS (FBS) - For everything  -------------------------------------------------
  # List of elements
  list_elements = ['Losses', 'Production Quantity']

  list_items = ['Animal Products > (List)']

  # 1990 - 2013
  code = 'FBSH'
  my_countries = [faostat.get_par(code, 'area')[c] for c in list_partnerregions_trade]
  my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
  my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
  list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997',
                '1998', '1999', '2000', '2001',
                '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009']
  my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

  my_pars = {
    'area': my_countries,
    'element': my_elements,
    'item': my_items,
    'year': my_years
  }
  df_losses_csl_1990_2013 = faostat.get_data_df(code, pars=my_pars,
                                                strval=False)

  # Renaming Elements
  df_losses_csl_1990_2013.loc[
    df_losses_csl_1990_2013['Element'].str.contains('Production Quantity',
                                                    case=False,
                                                    na=False), 'Element'] = 'Production'

  # 2010 - 2022
  # Different list because different in item nomination such as rice
  list_elements = ['Losses', 'Production Quantity']
  code = 'FBS'
  my_countries = [faostat.get_par(code, 'area')[c] for c in list_partnerregions_trade]
  my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
  my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
  list_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017',
                '2018', '2019', '2020', '2021', '2022', '2023']
  my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

  my_pars = {
    'area': my_countries,
    'element': my_elements,
    'item': my_items,
    'year': my_years
  }
  df_losses_csl_2010_2021 = faostat.get_data_df(code, pars=my_pars,
                                                strval=False)
  # Renaming Elements
  df_losses_csl_2010_2021.loc[
    df_losses_csl_2010_2021['Element'].str.contains('Production Quantity',
                                                    case=False,
                                                    na=False), 'Element'] = 'Production'

  # Concatenating
  df_losses_csl = pd.concat([df_losses_csl_1990_2013, df_losses_csl_2010_2021])

  # Compute losses ([%] of production) -----------------------------------------------------------------------------------
  # Losses [%] = 1 / (1 - Losses [1000t] / Production [1000t]) (pre processing for multiplicating the workflow)

  # 1: Pivot the DataFrame
  pivot_df = df_losses_csl.pivot_table(index=['Area', 'Year', 'Item'],
                                       columns='Element',
                                       values='Value').reset_index()

  # Replace NaN with 0
  pivot_df['Losses'] = pivot_df['Losses'].fillna(0.0)

  # 2: Compute the Losses [%] (really it's unit less)
  pivot_df['Losses[%]'] = 1 + (pivot_df['Losses'] / pivot_df['Production'])

  # Drop the columns Production, Import Quantity and Export Quantity
  pivot_df = pivot_df.drop(columns=['Production', 'Losses'])

  # Extrapolating for 2022 -----------------------------------------------------------------------------------------------

  # PathwayCalc formatting -----------------------------------------------------------------------------------------------

  # Food item name matching with dictionary
  # Read excel file
  df_dict_csl_losses = pd.read_excel(
    'dictionaries/dictionnary_livestock.xlsx',
    sheet_name='climate-smart-livestock_losses')

  # Merge based on 'Item'
  df_losses_csl_pathwaycalc = pd.merge(df_dict_csl_losses, pivot_df, on='Item')

  # Drop the 'Item' column
  df_losses_csl_pathwaycalc = df_losses_csl_pathwaycalc.drop(columns=['Item'])

  # Renaming existing columns (geoscale, timsecale, value)
  df_losses_csl_pathwaycalc.rename(
    columns={'Area': 'geoscale', 'Year': 'timescale', 'Losses[%]': 'value'},
    inplace=True)

  # Adding the columns module, lever, level and string-pivot at the correct places
  df_losses_csl_pathwaycalc['module'] = 'agriculture'

  # Rename countries to Pathaywcalc name


  # Format as datamatrix
  lever = 'dummy'
  df_losses_csl_pathwaycalc['lever'] = lever
  df_losses_csl_pathwaycalc['module'] = lever
  df_losses_csl_pathwaycalc['level'] = 0.0
  df_losses_csl_pathwaycalc = ensure_structure(df_losses_csl_pathwaycalc)
  df_ots, df_fts = database_to_df(df_losses_csl_pathwaycalc, lever,
                                  level='all')
  df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
  dm_losses = DataMatrix.create_from_df(df_ots, num_cat=1)


  return dm_losses

# CalculationLeaf FEED RATION ------------------------------------------------------------------------------

def feed_ration(df_feed_ration, cdm_efficiency, cdm_kcal):
  # ----------------------------------------------------------------------------------------------------------------------
  # step FEED RATION ----------------------------------------------------------------------------------------------------------
  # ---------------------------------------------------------------------------------------------------------------------

  # Fill nan with zeros
  df_feed_ration['Feed'] = df_feed_ration['Feed'].fillna(0.0)

  # Add a column with the total feed (per country and year)
  df_feed_ration['Total feed'] = df_feed_ration.groupby(['Area', 'Year'])[
    'Feed'].transform('sum')

  # Feed ration [%] = Feed from item i / Total feed
  df_feed_ration['Feed ratio'] = df_feed_ration['Feed'] / df_feed_ration[
    'Total feed']

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
  df_feed_ration.rename(
    columns={'Area': 'geoscale', 'Year': 'timescale', 'Feed ratio': 'value'},
    inplace=True)

  # Read excel file
  df_dict_csl = pd.read_excel(
    'dictionaries/dictionnary_livestock.xlsx',
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
  df_csl_feed_pathwaycalc['geoscale'] = df_csl_feed_pathwaycalc[
    'geoscale'].replace(
    'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
  df_csl_feed_pathwaycalc['geoscale'] = df_csl_feed_pathwaycalc[
    'geoscale'].replace('Netherlands (Kingdom of the)',
                        'Netherlands')
  df_csl_feed_pathwaycalc['geoscale'] = df_csl_feed_pathwaycalc[
    'geoscale'].replace('Czechia', 'Czech Republic')

  # Format as datamatrix
  lever = 'dummy'
  df_csl_feed_pathwaycalc['lever'] = lever
  df_csl_feed_pathwaycalc['level'] = 0.0
  df_ots, df_fts = database_to_df(df_csl_feed_pathwaycalc, lever,
                                  level='all')
  df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
  dm_feed_ration = DataMatrix.create_from_df(df_ots, num_cat=1)

  # Step SHARE GRASS -----------------------------------------------------------

  # Load
  dm_dom_prod_liv = dm_cal_dom_prod.filter({'Country':list_countries_calc}, inplace=False).copy()
  cdm_cp_efficiency = cdm_efficiency.copy()
  cdm_kcal_temp = cdm_kcal.copy()
  dm_feed_cal = dm_cal_feed.copy()

  # ASF domestic prod with losses => Unit conversion: [kcal] to [t]
  cdm_kcal_temp.rename_col_regex(str1="pro-liv-", str2="", dim="Categories1")
  cdm_kcal_temp = cdm_kcal_temp.filter({'Categories1': ['abp-dairy-milk', 'abp-hens-egg',
                                              'meat-bovine', 'meat-oth-animal',
                                              'meat-pig', 'meat-poultry',
                                              'meat-sheep']})
  dm_dom_prod_liv.sort('Categories1')
  cdm_kcal_temp.sort('Categories1')
  array_temp = dm_dom_prod_liv[:, :, 'cal_agr_domestic-production-liv', :] \
               / cdm_kcal_temp[np.newaxis, np.newaxis, 'cp_kcal-per-t', :]
  dm_dom_prod_liv.add(array_temp, dim='Variables',
                      col_label='agr_domestic_production_liv_afw_t',
                      unit='t')

  # Feed req with grass per type [t] =  ASF domestic prod with losses [kt] * FCR [%]
  dm_dom_prod_liv.sort('Categories1')
  cdm_cp_efficiency.sort('Categories1')
  dm_temp = dm_dom_prod_liv[:, :, 'agr_domestic_production_liv_afw_t', :] \
            * cdm_cp_efficiency[np.newaxis, np.newaxis, 'cp_efficiency_liv', :]
  dm_dom_prod_liv.add(dm_temp, dim='Variables',
                      col_label='agr_feed-requirement',
                      unit='t')

  # Feed req total with grass [t] =  sum per type (Feed req with grass per type [t])
  dm_dom_prod_liv = dm_dom_prod_liv.filter(
    {'Variables': ['agr_feed-requirement']})
  dm_ruminant = dm_dom_prod_liv.filter(
    {'Categories1': ['abp-dairy-milk', 'meat-bovine',
                     'meat-sheep']})  # Create copy for ruminants
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
  dm_dom_prod_liv.operation('agr_feed-requirement_total', '-',
                            'cal_agr_demand_feed_total',
                            out_col='grass_feed', unit='t')

  # Feed ruminant with grass [t] = sum (feed ruminant [t])
  dm_ruminant.groupby({'ruminant': '.*'}, dim='Categories1', regex=True,
                      inplace=True)
  dm_ruminant = dm_ruminant.flatten()

  # Share grass feed ruminant [%] = Grass feed [t] / Feed ruminant with grass [t]
  dm_dom_prod_liv.append(dm_ruminant, dim='Variables')
  dm_dom_prod_liv.operation('grass_feed', '/', 'agr_feed-requirement_ruminant',
                            out_col='agr_ruminant-feed_share-grass', unit='-')
  dm_grass = dm_dom_prod_liv.filter({'Variables':['agr_ruminant-feed_share-grass']}, inplace=False)

  return dm_feed_ration, dm_grass

# CalculationLeaf YIELD & SLAUGHTER RATE ------------------------------------------------------------------------------
def yield_slaughter_rate(df_liv_pop, dm_prod_share):

    # ----------------------------------------------------------------------------------------------------------------------
    # YIELD (DAIRY & EGGS) -------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    list_elements = ['Producing Animals/Slaughtered', 'Production Quantity']

    list_items = ['Milk, Total > (List)', 'Eggs Primary > (List)']

    # 1990 - 2022
    code = 'QCL'
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_partnerregions_trade]
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
    pivot_df['Laying'] = pivot_df['Laying'].fillna(0.0)
    pivot_df['Milk Animals'] = pivot_df['Milk Animals'].fillna(0.0)

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

    # 1990 - 2022
    code = 'QCL'
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_partnerregions_trade]
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
    pivot_df_slau['Producing Animals/Slaughtered'] = pivot_df_slau['Producing Animals/Slaughtered'].fillna(0.0)
    pivot_df_slau['Production'] = pivot_df_slau['Production'].fillna(0.0)

    # Create a copy for slau rate
    df_slau_meat = pivot_df_slau.copy()

    # Yield [t/lsu] = Production quantity / Producing animals/Slaugthered NOW DONE AFTER using cal values
    pivot_df_slau['Yield [t/lsu]'] = pivot_df_slau['Producing Animals/Slaughtered']
    #pivot_df_slau['Yield [t/lsu]'] = pivot_df_slau['Production'] / pivot_df_slau['Producing Animals/Slaughtered']

    # Drop the columns
    pivot_df_slau = pivot_df_slau.drop(columns=['Producing Animals/Slaughtered', 'Production'])

    # Replace NaN with 0
    pivot_df_slau['Yield [t/lsu]'] = pivot_df_slau['Yield [t/lsu]'].fillna(0.0)

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
    df_slau['Slaughtered animals [%]'] = df_slau['Slaughtered animals [%]'].fillna(0.0)

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
        'dictionaries/dictionnary_livestock.xlsx',
        sheet_name='climate-smart-livestock_yield')
    df_dict_csl_slau = pd.read_excel(
        'dictionaries/dictionnary_livestock.xlsx',
        sheet_name='climate-smart-livestock_slau')

    # Merge based on 'Item'
    df_yield_liv_pathwaycalc = pd.merge(df_dict_csl_yield, df_yield_liv, left_on='Item', right_on='Aggregation')
    df_slau_liv_pathwaycalc = pd.merge(df_dict_csl_slau, df_slau_meat, left_on='Item', right_on='Aggregation')

    # Drop the 'Item' column
    df_yield_liv_pathwaycalc = df_yield_liv_pathwaycalc.drop(columns=['Item', 'Aggregation'])
    df_slau_liv_pathwaycalc = df_slau_liv_pathwaycalc.drop(columns=['Item', 'Aggregation'])

    # ----------------------------------------------------------------------------------------------------------------------
    # FINAL RESULTS --------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # Format as datamatrix - Yields
    lever = 'dummy'
    df_yield_liv_pathwaycalc['lever'] = lever
    df_yield_liv_pathwaycalc['module'] = lever
    df_yield_liv_pathwaycalc['level'] = 0.0
    df_yield_liv_pathwaycalc = ensure_structure(df_yield_liv_pathwaycalc)
    df_ots, df_fts = database_to_df(df_yield_liv_pathwaycalc, lever,
                                    level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm_liv_yield = DataMatrix.create_from_df(df_ots, num_cat=1)

    # Yield total [kcal/lsu] = Domestic prod with losses [kcal] / producing-slaugthered animals [lsu]
    dm_liv_yield.rename_col('agr_livestock_yield',
                        'agr_livestock_producing',
                        dim='Variables')
    dm_liv_yield.append(dm_cal_dom_prod, dim='Variables')
    dm_liv_yield.operation('cal_agr_domestic-production-liv', '/',
                              'agr_livestock_producing',
                              out_col='agr_livestock_yield_total',
                              unit='kcal/lsu')


    # Yield evolution_o/i (organic with respect to intensive) [-] HERE
    # Source: animal welfare working paper
    yield_evolution = {'meat-bovine': 0.8,
                      'meat-poultry': 0.8,
                      'meat-sheep': 0.8,
                      'meat-pig': 0.8,
                      'meat-oth-animal': 0.8,
                      'abp-dairy-milk': 0.8,
                      'abp-hens-egg': 0.8}

    # Intensive yield_i [kcal/lsu] = dom prod with losses (total) [kcal] / [ lsu_T * (share_o * (yield evolution_o/i - 1) +1)]
    dm_liv_yield.rename_col('agr_livestock_yield',
                            'agr_livestock_producing',
                            dim='Variables')
    dm_liv_yield.add(0.0, dim='Variables', dummy=True, col_label='agr_livestock_yield_intensive')
    for cat in dm_liv_yield.col_labels['Categories1']:
      dm_liv_yield[:, :, 'agr_livestock_yield_intensive', cat] =\
        dm_cal_dom_prod[:, :, 'cal_agr_domestic-production-liv', cat] \
         / (dm_liv_yield[:, :,'agr_livestock_producing', cat] *
            ( dm_prod_share[:, :,'livestock_share-organic', cat] * (yield_evolution[cat] - 1.0 ) +1))

    # Organic yield_c [kcal/lsu] =  yield_i [kcal/lsu] * yield evolution_o/i [-]
    dm_liv_yield.add(0.0, dim='Variables', dummy=True, col_label='agr_livestock_yield_organic')
    for cat in dm_liv_yield.col_labels['Categories1']:
      dm_liv_yield[:, :, 'agr_livestock_yield_organic', cat] =\
        dm_liv_yield[:, :, 'agr_livestock_yield_intensive', cat] \
         * yield_evolution[cat]

    # Format yield
    dm_liv_yield.filter({'Variables':['agr_livestock_yield_organic', 'agr_livestock_yield_intensive', 'agr_livestock_yield_total']}, inplace=True)
    linear_fitting(dm_liv_yield, years_all)

    # Format as datamatrix - Slaughter rates
    lever = 'dummy'
    df_slau_liv_pathwaycalc['lever'] = lever
    df_slau_liv_pathwaycalc['module'] = lever
    df_slau_liv_pathwaycalc['level'] = 0.0
    df_slau_liv_pathwaycalc = ensure_structure(df_slau_liv_pathwaycalc)
    df_ots, df_fts = database_to_df(df_slau_liv_pathwaycalc, lever,
                                    level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm_slaughter_rates = DataMatrix.create_from_df(df_ots, num_cat=1)

    return dm_liv_yield, dm_slaughter_rates

# CalculationLeaf LIVESTOCK ALT PROTEIN MEALS ------------------------------------------------------------------------------------
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
    df_protein_meals_all['agr_alt-protein_meat-oth-animal_algae[%]'] = 0.0
    df_protein_meals_all['agr_alt-protein_meat-oth-animal_insect[%]'] = 0.0
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

    # Format as datamatrix
    lever = 'dummy'
    df_protein_meals_pathwaycalc['lever'] = lever
    df_protein_meals_pathwaycalc['level'] = 0.0
    df_ots, df_fts = database_to_df(df_protein_meals_pathwaycalc, lever,
                                    level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm_feed_alt_protein = DataMatrix.create_from_df(df_ots, num_cat=2)

    return dm_feed_alt_protein

# CalculationLeaf CAL - POP & DOM PROD -----------------------------------------------------------------------------------
def livestock_calibration(list_countries_calc, dm_losses):
    # ----------------------------------------------------------------------------------------------------------------------
    # Step POPULATION ----------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

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
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_partnerregions_trade]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    my_sources = [faostat.get_par(code, 'sources')[i] for i in list_sources]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                  '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                  '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
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
        'dictionaries/dictionnary_livestock.xlsx',
        sheet_name='calibration')

    # Merge based on 'Item'
    df_liv_population_calibration = pd.merge(df_dict_calibration, df_liv_population, on='Item')

    # Drop the 'Item' column
    df_liv_population_calibration = df_liv_population_calibration.drop(columns=['Item'])

    # Renaming existing columns (geoscale, timsecale, value)
    df_liv_population_calibration.rename(
        columns={'Area': 'geoscale', 'Year': 'timescale', 'Value': 'value'},
        inplace=True)

    # Format as datamatrix
    lever = 'dummy'
    df_liv_population_calibration['lever'] = lever
    df_liv_population_calibration['module'] = lever
    df_liv_population_calibration['level'] = 0.0
    df_liv_population_calibration = ensure_structure(df_liv_population_calibration)
    df_ots, df_fts = database_to_df(df_liv_population_calibration, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm_cal_liv_pop = DataMatrix.create_from_df(df_ots, num_cat=1)

    # ----------------------------------------------------------------------------------------------------------------------
    # Step DOMESTIC PRODUCTION (LIVESTOCK PRODUCTS) ----------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    # Read data ------------------------------------------------------------------------------------------------------------

    # Common for all
    # List of countries

    # FOOD BALANCE SHEETS (FBS) - -------------------------------------------------
    # List of elements
    list_elements = ['Production Quantity', 'Losses']

    list_items = ['Milk - Excluding Butter + (Total)', 'Eggs + (Total)',
                  'Bovine Meat', 'Meat, Other', 'Pigmeat',
                  'Poultry Meat', 'Mutton & Goat Meat']

    # 1990 - 2013
    ld = faostat.list_datasets()
    code = 'FBSH'
    pars = faostat.list_pars(code)
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_partnerregions_trade]
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
    df_domestic_supply_1990_2013 = faostat.get_data_df(code, pars=my_pars, strval=False)

    # 2010-2022
    list_items = ['Milk - Excluding Butter + (Total)', 'Eggs + (Total)',
                  'Bovine Meat', 'Meat, Other', 'Pigmeat',
                  'Poultry Meat', 'Mutton & Goat Meat']
    code = 'FBS'
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_partnerregions_trade]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021',
                  '2022', '2023']
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
      df_domestic_supply_1990_2013['Item'].str.contains(
        'Rice (Milled Equivalent)', case=False, regex=False
      ), 'Item'] = 'Rice and products'

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
        'dictionaries/dictionnary_livestock.xlsx',
        sheet_name='calibration')

    # Prepend "Diet" to each value in the 'Item' column
    pivot_df_domestic_supply['Item'] = pivot_df_domestic_supply['Item'].apply(lambda x: f"Production {x}")

    # Renaming existing columns (geoscale, timsecale, value)
    pivot_df_domestic_supply.rename(
        columns={'Area': 'geoscale', 'Year': 'timescale', 'Production [kcal]': 'value'},
        inplace=True)

    # Merge based on 'Item'
    df_domestic_supply_calibration = pd.merge(df_dict_calibration, pivot_df_domestic_supply, on='Item')

    # Drop the 'Item' column
    df_domestic_supply_calibration = df_domestic_supply_calibration.drop(columns=['Item'])

    # Format as datamatrix
    lever = 'dummy'
    df_domestic_supply_calibration['lever'] = lever
    df_domestic_supply_calibration['module'] = lever
    df_domestic_supply_calibration['level'] = 0.0
    df_domestic_supply_calibration = ensure_structure(df_domestic_supply_calibration)
    df_ots, df_fts = database_to_df(df_domestic_supply_calibration, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm_cal_dom_prod = DataMatrix.create_from_df(df_ots, num_cat=1)

    # Livestock domestic prod with losses [kcal] = livestock domestic prod [kcal] * Production losses livestock [%]
    dm_losses_liv = dm_losses.copy()
    dm_losses_liv.drop(dim='Categories1',
                       col_label=['abp-processed-afat', 'abp-processed-offal'])
    dm_cal_dom_prod.rename_col('cal_agr_domestic-production-liv',
                               'cal_agr_domestic-production-liv_raw',
                               dim='Variables')
    dm_cal_dom_prod.append(dm_losses_liv, dim='Variables')
    dm_cal_dom_prod.operation('agr_livestock_losses', '*',
                              'cal_agr_domestic-production-liv_raw',
                              out_col='cal_agr_domestic-production-liv',
                              unit='kcal')
    dm_cal_dom_prod.filter({'Variables':['cal_agr_domestic-production-liv']}, inplace=True)

    return dm_cal_dom_prod, dm_cal_liv_pop, df_liv_pop



# CalculationLeaf CAL - LIVESTOCK MANURE -----------------------------------------------------------------------------------

def manure_calibration(list_countries_calc):
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
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries_calc]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    my_sources = [faostat.get_par(code, 'sources')[i] for i in list_sources]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
                  '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                  '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
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
        'dictionaries/dictionnary_livestock.xlsx',
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

    # Ensure structure & linear fit
    lever = 'dummy'
    df_liv_emissions_calibration['lever'] = lever
    df_liv_emissions_calibration['module'] = 'agriculture'
    df_liv_emissions_calibration['level'] = 0.0
    df_liv_emissions_calibration = ensure_structure(df_liv_emissions_calibration)
    df_liv_emissions_calibration = linear_fitting_ots_db(df_liv_emissions_calibration, years_ots,
                                                countries='all')

    # Format as datamatrix
    df_ots, df_fts = database_to_df(df_liv_emissions_calibration, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm_cal_liv_emissions = DataMatrix.create_from_df(df_ots, num_cat=2)

    return dm_cal_liv_emissions, df_liv_emissions

# CalculationLeaf FXA - MILK FEED FOOD RATIO---------------------------------------------------------------------------------------------
def fxa_ffr_milk(df_ffr_milk):

  # ffr ratio [-] = (Feed + Food + Processing) / Food
  df_ffr_milk['value'] = (df_ffr_milk['Feed'] + df_ffr_milk['Food'] + df_ffr_milk['Processing']) / df_ffr_milk['Food']
  df_ffr_milk = df_ffr_milk[['Area', 'Year', 'Item', 'value']]

  # Calc Formatting ------------------------------------------------------------

  # Food item name matching with dictionary
  # Read excel file
  df_dict = pd.read_excel(
    'dictionaries/dictionnary_livestock.xlsx',
    sheet_name='fxa')

  # Renaming existing columns (geoscale, timsecale, value)
  df_ffr_milk = df_ffr_milk.rename(columns={'Area': 'geoscale', 'Year': 'timescale'})

  # Merge based on 'Item'
  df_ffr_milk = pd.merge(df_dict, df_ffr_milk, on='Item')

  # Drop the 'Item' column
  df_ffr_milk = df_ffr_milk.drop(columns=['Item'])

  # Adding the columns module, lever, level and string-pivot at the correct places
  lever = 'dummy'
  df_ffr_milk['module'] = lever
  df_ffr_milk['lever'] = lever
  df_ffr_milk['level'] = 0

  # Rename countries to Pathaywcalc name
  df_ffr_milk['geoscale'] = df_ffr_milk['geoscale'].replace(
    'United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
  df_ffr_milk['geoscale'] = df_ffr_milk['geoscale'].replace(
    'Netherlands (Kingdom of the)',
    'Netherlands')
  df_ffr_milk['geoscale'] = df_ffr_milk['geoscale'].replace(
    'Czechia', 'Czech Republic')

  # Extrapolation
  df_ffr_milk = linear_fitting_ots_db(df_ffr_milk, years_all,
                                             countries='all')

  # Format as dm
  df_ots, df_fts = database_to_df(df_ffr_milk, lever, level='all')
  df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
  dm_fxa_ffr_milk = DataMatrix.create_from_df(df_ots, num_cat=1)

  return dm_fxa_ffr_milk

# CalculationLeaf FXA - MANURE EMISSION FACTORS ------------------------------

def manure_fxa(list_countries_calc, df_liv_emissions, df_manure_n_fxa, df_manure_ch4_fxa):

   # N2O EMISSIONS -------------------------------------------------------------
   # Filter & Rename
   df_manure_n_fxa = df_manure_n_fxa[['Area', 'Year', 'Aggregation','Manure left on pasture (N content)',
                     'Manure applied to soils (N content)', 'Losses from manure treated (N content)']]
   df_manure_n_fxa = df_manure_n_fxa.rename(columns={'Manure left on pasture (N content)':'N2O Pasture',
                                   'Manure applied to soils (N content)':'N2O Applied',
                                   'Losses from manure treated (N content)':'N2O Treated'})

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
   df_manure_fxa['value'] = df_manure_fxa['value'].fillna(0.0)

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
     'dictionaries/dictionnary_livestock.xlsx',
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
   lever = 'dummy'
   df_manure_fxa['lever'] = lever
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
   df_manure_fxa = linear_fitting_ots_db(df_manure_fxa, years_all,
                                               countries='all')

   # Format as datamatrix
   df_ots, df_fts = database_to_df(df_manure_fxa, lever,
                                   level='all')
   df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
   dm_manure_fxa = DataMatrix.create_from_df(df_ots, num_cat=0)

  # Create separate dm
   dm_fxa_N2O = dm_manure_fxa.filter_w_regex(
     {'Variables': 'fxa_ef_liv_N2O-emission.*'})
   dm_fxa_N2O.deepen_twice()
   dm_fxa_CH4 = dm_manure_fxa.filter_w_regex(
     {'Variables': 'fxa_ef_liv_CH4-emission.*'})
   dm_fxa_CH4.deepen_twice()

   return dm_fxa_CH4, dm_fxa_N2O

# CalculationLeaf CAL - FEED DEMAND ----------------------------------------------------------------------------------

def feed_calibration(list_countries_calc):
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
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries_calc]
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
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries_calc]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021',
                  '2022', '2023']
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
      df_feed_1990_2013['Item'].str.contains(
        'Rice (Milled Equivalent)', case=False, regex=False
      ), 'Item'] = 'Rice and products'

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
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries_calc]
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
                  'Cake of sesame seed', 'Cake of sunflower seed', 'Cake, oilseeds nes', 'Cake, poppy seed']
    code = 'SCL'
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries_calc]
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
        'dictionaries/dictionnary_livestock.xlsx',
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

    # Format as datamatrix
    lever = 'dummy'
    df_feed_calibration['lever'] = lever
    df_feed_calibration['level'] = 0.0
    df_ots, df_fts = database_to_df(df_feed_calibration, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm_cal_feed = DataMatrix.create_from_df(df_ots, num_cat=1)

    return dm_cal_feed, df_feed_ration

# CalculationLeaf CONSTANTS  ------------------------------

def constant():
  # FEED - ENERGY CONVERSION EFFICIENCY  ----------------------------------------------------------------------------------------

  # Read excel
  df_feed_conv = pd.read_excel('dictionaries/constants_livestock.xlsx',
                            sheet_name='cp_feed_efficiency')

  # Filter columns
  df_feed_conv = df_feed_conv[['variables', 'value']].copy()

  # Turn the df in a dict
  dict_feed = dict(zip(df_feed_conv['variables'], df_feed_conv['value']))
  categories1 = df_feed_conv['variables'].tolist()

  # Format as a cdm
  cdm_efficiency = ConstantDataMatrix(col_labels={'Variables': ['cp_efficiency_liv'],
                                            'Categories1': categories1})
  arr = np.zeros((len(cdm_efficiency.col_labels['Variables']),
                  len(cdm_efficiency.col_labels['Categories1'])))
  cdm_efficiency.array = arr
  idx = cdm_efficiency.idx
  for cat, val in dict_feed.items():
    cdm_efficiency.array[idx['cp_efficiency_liv'], idx[cat]] = val
  cdm_efficiency.units["cp_efficiency_liv"] = "kg DM feed/kg EW"

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

  return cdm_efficiency, cdm_kcal

# CalculationLeaf FTS  ------------------------------
def fts_processing():

  # ssr-feed, ssr-liv, livestock-losses, share-organic, ruminand-feed ----------
  # Read Excel
  df_fts_data = pd.read_excel(
    'data/livestock_fts.xlsx',
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

  return dm_fts

# CalculationLeaf PICKLE CREATION ------------------------------

def datamatrix_to_pickle(dm_fts):

  # Make list with all years
  years_all = years_ots + years_fts


  # FixedAssumptionsToDatamatrix -----------------------------------------------
  dict_fxa = {}

  dict_fxa['split-import'] = dm_liv_trade_origin
  dict_fxa['share-export'] = dm_fxa_exports
  dict_fxa['livestock-yield'] = dm_liv_yield
  dict_fxa['ef_liv_N2O-emission'] = dm_fxa_N2O
  dict_fxa['ef_liv_CH4-emission_treated'] = dm_fxa_CH4
  dict_fxa['liv_manure_n-stock'] = dm_fxa_manure_yield
  dict_fxa['ratio_milk'] = dm_fxa_ffr_milk


  # CalibrationDataToDatamatrix ------------------------------------------------

  dict_fxa['cal_agr_liv-population'] = dm_cal_liv_pop.filter({'Country':['Switzerland']}, inplace=False)
  dict_fxa['cal_agr_liv-population_organic'] = dm_cal_liv_pop_org
  dict_fxa['cal_agr_domestic-production-liv'] = dm_cal_dom_prod
  dict_fxa['cal_agr_imports-liv'] = dm_cal_imports
  dict_fxa['cal_agr_liv_CH4-emission'] = dm_cal_liv_emissions.filter({'Variables':['cal_agr_liv_CH4-emission']}, inplace=False)
  dict_fxa['cal_agr_liv_N2O-emission'] = dm_cal_liv_emissions.filter({'Variables':['cal_agr_liv_N2O-emission']}, inplace=False)
  dict_fxa['cal_agr_demand_feed'] = dm_cal_feed

  # LeversToDatamatrix OTS -----------------------------------------------------
  dict_ots = {}

  # ssr-liv
  dict_ots['ssr-liv'] = dm_ssr_liv
  # ssr-feed
  dict_ots['ssr-feed'] = dm_ssr_feed
  # livestock-losses
  dict_ots['livestock-losses'] = dm_losses
  # 'slaughter-rates'
  dict_ots['slaughter-rates'] = dm_slaughter_rates
  # livestock-density
  dict_ots['livestock-density'] = dm_density
  # livestock-enteric
  dict_ots['livestock-enteric'] = dm_enteric
  # livestock-manure
  dict_ots['livestock-manure'] = dm_manure
  # feed-ration
  dict_ots['feed-ration'] = dm_feed_ration
  # alt-protein
  dict_ots['alt-protein'] = dm_feed_alt_protein
  # ruminant-feed
  dict_ots['ruminant-feed'] = dm_grass
  # share-organic
  dict_ots['share-organic'] = dm_prod_share


  # LeversToDatamatrix FTS -----------------------------------------------------
  dict_fts = {}

  # FTS linear fitting of ots
  DM_ots = dict_ots.copy()

  # Adding a new lever with dummy values
  dict_fts['slaughter-rates'] = {'slaughter-rates': dict()}
  dict_fts['livestock-density'] = {'livestock-density': dict()}
  dict_fts['livestock-enteric'] = {'livestock-enteric': dict()}
  dict_fts['livestock-manure'] = {'livestock-manure': dict()}
  dict_fts['feed-ration'] = {'feed-ration': dict()}
  dict_fts['alt-protein'] = {'alt-protein': dict()}


  # Levers to be normalised
  list_norm = ['climate-smart-livestock_ration']

  for key in dict_fts.keys():
    if isinstance(DM_ots[key], dict):
      for subkey in dict_fts[key].keys():
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
        dict_fts[key][lev] = dm.filter({'Years': years_fts}, inplace=False)

  # Linear fitting between ots and fts objective (2050) ------------------

  # Lever - ssr-liv
  lever = 'ssr-liv'
  for level in range(1,5):
    # Propagate the overall lever value across all livestock categories
    dm_ots = dict_ots[lever].copy()
    dm_fts_temp = dm_fts[lever][level]

    array_temp =  dm_fts[lever][level][:,years_fts[-1],'agr_ssr', np.newaxis] + \
                  dm_ots[:,years_ots[-1],'agr_ssr',:] - dm_ots[:,years_ots[-1],'agr_ssr',:] # +x-x To get the correct structure
    # Append with ots
    dm_ots.add(array_temp[:,np.newaxis,np.newaxis,:], dim='Years', dummy=True, col_label=years_fts[-1])
    # Linear fit
    linear_fitting(dm_ots, years_fts)
    dm_fts[lever][level] = dm_ots.filter({'Years':years_fts}, inplace=False)
  dict_fts[lever] = dm_fts[lever]

  # Lever - ssr-feed
  lever = 'ssr-feed'
  for level in range(1,5):
    # Propagate the overall lever value across all feed categories
    dm_ots = dict_ots[lever].copy()
    dm_fts_temp = dm_fts[lever][level]
    array_temp =  dm_fts[lever][level][:,years_fts[-1],'agr_ssr', np.newaxis] + \
                  dm_ots[:,years_ots[-1],'agr_ssr',:] - dm_ots[:,years_ots[-1],'agr_ssr',:] # +x-x To get the correct structure
    # Append with ots
    dm_ots.add(array_temp[:,np.newaxis,np.newaxis,:], dim='Years', dummy=True, col_label=years_fts[-1])
    # Linear fit
    linear_fitting(dm_ots, years_fts)
    dm_fts[lever][level] = dm_ots.filter({'Years':years_fts}, inplace=False)
  dict_fts[lever] = dm_fts[lever]

  # Lever - share-organic
  lever = 'share-organic'
  for level in range(1,5):
    # Propagate the overall lever value across all feed categories
    dm_ots = dict_ots[lever].copy()
    dm_fts_temp = dm_fts[lever][level]
    array_temp =  dm_fts[lever][level][:,years_fts[-1],'livestock_share-organic', np.newaxis] + \
                  dm_ots[:,years_ots[-1],'livestock_share-organic',:] - \
                  dm_ots[:,years_ots[-1],'livestock_share-organic',:] # +x-x To get the correct structure
    # Append with ots
    dm_ots.add(array_temp[:,np.newaxis,np.newaxis,:], dim='Years', dummy=True, col_label=years_fts[-1])
    # Linear fit
    linear_fitting(dm_ots, years_fts)
    dm_fts[lever][level] = dm_ots.filter({'Years':years_fts}, inplace=False)
  dict_fts[lever] = dm_fts[lever]

  # Lever - ruminant-feed
  lever = 'ruminant-feed'
  for level in range(1,5):
    # Propagate the overall lever value across all feed categories
    dm_fts[lever][level].append(dict_ots[lever], dim='Years')
    linear_fitting(dm_fts[lever][level], years_fts)
    dm_fts[lever][level].filter({'Years':years_fts}, inplace=True)
  dict_fts[lever] = dm_fts[lever]

  # Lever - livestock-losses
  lever = 'livestock-losses'
  for level in range(1,5):
    # Compute the reduction objective in 2050 compared to the last ots value,
    # for each food category
    dm_ots = dict_ots[lever].copy()
    array_temp =  1 - ( 1 - dm_ots[:,years_ots[-1],'agr_livestock_losses',:]) \
                  * dm_fts[lever][level][:,years_fts[-1],'agr_livestock_losses', np.newaxis]
    # Append with ots
    dm_ots.add(array_temp[:,np.newaxis,np.newaxis,:], dim='Years', dummy=True, col_label=years_fts[-1])
    # Linear fit
    linear_fitting(dm_ots, years_fts)
    dm_fts[lever][level] = dm_ots.filter({'Years':years_fts}, inplace=False)
  dict_fts[lever] = dm_fts[lever]


  # ConstantsToDatamatrix ------------------------------------------------------
  dict_const = {}
  dict_const['cdm_kcal-per-t'] = cdm_kcal
  #dict_const['cdm_lifestyle'] = cdm_lifestyle

  # Group all datamatrix in a single structure ---------------------------------
  DM_livestock = {
    'fxa': dict_fxa,
    'constant': dict_const,
    'fts': dict_fts,
    'ots': dict_ots
  }

  # Write datamatrix to pickle -------------------------------------------------
  f = '../../data/datamatrix/livestock.pickle'
  with open(f, 'wb') as handle:
    pickle.dump(DM_livestock, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return


# CalculationTree RUNNING PRE-PROCESSING -----------------------------------------------------------------------------------------------
years_ots = create_years_list(1990, 2023, 1)  # make list with years from 1990 to 2015
years_fts = create_years_list(2025, 2050, 5)
years_all = years_ots + years_fts

if not os.path.exists('data/faostat'):
    os.makedirs('data/faostat')

list_countries_calc = ['Switzerland']
list_partnerregions_trade = ['Switzerland',
                         '-- Australia and New Zealand + (Total)',
                         '-- Caribbean + (Total)',
                         '-- Central America + (Total)',
                         '-- Central Asia + (Total)',
                         '-- Eastern Africa + (Total)',
                         '-- Eastern Asia + (Total)',
                         '-- Eastern Europe + (Total)',
                         '-- Melanesia + (Total)',
                         '-- Micronesia + (Total)',
                         '-- Middle Africa + (Total)',
                         '-- Northern Africa + (Total)',
                         '-- Northern America + (Total)',
                         '-- Northern Europe + (Total)',
                         '-- Polynesia + (Total)',
                         '-- South America + (Total)',
                         '-- South-eastern Asia + (Total)',
                         '-- Southern Africa + (Total)',
                         '-- Southern Asia + (Total)',
                         '-- Southern Europe + (Total)',
                         '-- Western Africa + (Total)',
                         '-- Western Asia + (Total)',
                         '-- Western Europe + (Total)']

file_dict = {'ssr': 'data/faostat/ssr.csv',
             'cake': 'data/faostat/ssr_cake.csv',
             'molasse': 'data/faostat/ssr_2010_2021_molasse_cake.csv',
             'trade': 'data/faostat/trade.csv',
             'exports': 'data/faostat/exports.csv'}

cdm_efficiency, cdm_kcal = constant()
dm_ssr_liv, dm_ssr_feed, df_csl_feed, df_ffr_milk = self_sufficiency_processing(years_ots, list_countries_calc, file_dict)
dm_fxa_ffr_milk = fxa_ffr_milk(df_ffr_milk)
dm_liv_trade_origin, dm_cal_imports = trade_origin_processing(years_ots, list_countries_calc, file_dict)
dm_losses = livestock_losses()
dm_cal_dom_prod, dm_cal_liv_pop, df_liv_pop = livestock_calibration(list_countries_calc, dm_losses)
dm_prod_share, dm_cal_liv_pop_org = production_share(dm_cal_liv_pop)
dm_density = livestock_density(df_liv_pop)
dm_manure, dm_enteric, dm_fxa_manure_yield, df_manure_ch4_fxa, df_manure_n_fxa = livestock_emissions()
dm_cal_feed, df_feed_ration = feed_calibration(list_countries_calc)
dm_feed_ration, dm_grass = feed_ration(df_feed_ration, cdm_efficiency, cdm_kcal)
dm_liv_yield, dm_slaughter_rates = yield_slaughter_rate(df_liv_pop, dm_prod_share)
dm_cal_liv_emissions, df_liv_emissions = manure_calibration(list_countries_calc)
dm_fxa_CH4, dm_fxa_N2O = manure_fxa(list_countries_calc, df_liv_emissions, df_manure_n_fxa, df_manure_ch4_fxa)
dm_fxa_exports = exports_processing(list_countries_calc,file_dict)
dm_feed_alt_protein = livestock_protein_meals_processing(df_csl_feed)
dm_fts = fts_processing()

# CalculationTree RUNNING PICKLE CREATION
datamatrix_to_pickle(dm_fts)


