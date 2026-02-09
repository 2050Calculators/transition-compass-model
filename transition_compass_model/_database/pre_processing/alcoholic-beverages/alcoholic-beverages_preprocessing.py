import numpy as np
from model.common.auxiliary_functions import interpolate_nans, add_missing_ots_years, linear_fitting_ots_db, linear_fitting, create_years_list, dm_match_countries
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



# CalculationLeaf CAL - DOM PROD & BEV
def bev_calibration(list_countries_calc, dm_fxa_pro_yield, cdm_bev):

    # ----------------------------------------------------------------------------------------------------------------------
    # DOMESTIC PRODUCTION (CROP PRODUCTS) ----------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    try:
      df_domestic_supply = pd.read_csv(file_dict['dom-prod-bev'])
    except OSError:

      # FOOD BALANCE SHEETS (FBS) - -------------------------------------------------
      # List of elements
      list_elements = ['Production Quantity', 'Losses']

      list_items = ['Cereals - Excluding Beer + (Total)', 'Fruits - Excluding Wine + (Total)', 'Oilcrops + (Total)',
                  'Pulses + (Total)', 'Rice (Milled Equivalent)',
                  'Starchy Roots + (Total)', 'Sugar Crops + (Total)', 'Vegetables + (Total)',
                  'Beverages, Fermented', 'Beverages, Alcoholic', 'Beer', 'Wine']

      # 1990 - 2013
      ld = faostat.list_datasets()
      code = 'FBSH'
      pars = faostat.list_pars(code)
      my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries_calc ]
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
                  'Beverages, Fermented', 'Beverages, Alcoholic', 'Beer', 'Wine']
      code = 'FBS'
      my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries_calc ]
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
          'Rice (Milled Equivalent)', case=False, na=False, regex=False
        ),
        'Item'
      ] = 'Rice and products'

      # Concatenating all the years together
      df_domestic_supply = pd.concat([df_domestic_supply_1990_2013, df_domestic_supply_2010_2022])

      # Save to csv
      df_domestic_supply.to_csv(file_dict['dom-prod-bev'], index=False)

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
        'dictionaries/dictionnary_alcoholic-beverages.xlsx',
        sheet_name='calibration')

    # Prepend "Diet" to each value in the 'Item' column
    pivot_df_domestic_supply['Item'] = pivot_df_domestic_supply['Item'].apply(lambda x: f"Production {x}")

    # Renaming existing columns (geoscale, timsecale, value)
    pivot_df_domestic_supply.rename(
        columns={'Area': 'geoscale', 'Year': 'timescale', 'Production [kcal]': 'value'},
        inplace=True)

    # Merge based on 'Item'
    df_cal_dom_prod = pd.merge(df_dict_calibration, pivot_df_domestic_supply, on='Item')

    # Drop the 'Item' column
    df_cal_dom_prod = df_cal_dom_prod.drop(columns=['Item'])

    # Adding the columns module, lever, level and string-pivot at the correct places
    lever = 'food-net-import'
    df_cal_dom_prod['module'] = 'agriculture'
    df_cal_dom_prod['lever'] = lever
    df_cal_dom_prod['level'] = 0

    # Extrapolation
    df_cal_dom_prod = linear_fitting_ots_db(df_cal_dom_prod, years_ots, countries='all')

    # Format as datamatrix - Cal dom prod bev
    df_cal_dom_prod_bev = df_cal_dom_prod[
        df_cal_dom_prod['variables'].str.contains('cal_agr_domestic-production_bev', case=False, na=False)
    ].copy()
    df_ots, df_fts = database_to_df(df_cal_dom_prod_bev, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm_cal_dom_prod_bev = DataMatrix.create_from_df(df_ots, num_cat=1)

    # Here we want to convert the domestic production of beverages in raw materials
    # (e.g. in fruits and not wine) for wine & bev-alc

    # Filter processing yields
    dm_fxa_pro_yield_temp = dm_fxa_pro_yield.filter({'Years':years_ots})

    # Wine : Raw materials [kcal] = product [kcal] * processing yield [%]
    array_temp = dm_cal_dom_prod_bev['Switzerland', :, 'cal_agr_domestic-production_bev',
                 'wine'] \
                 * dm_fxa_pro_yield_temp['Switzerland',:, 'fxa_agr_processing-yield', 'wine-to-fruit']
    # Overwrite
    dm_cal_dom_prod_bev[:, :,'cal_agr_domestic-production_bev', 'wine'] = array_temp

    # Bev-alc : Raw materials [kcal] = product [kcal] * processing yield [%]
    array_temp = dm_cal_dom_prod_bev['Switzerland', :, 'cal_agr_domestic-production_bev',
                 'bev-alc'] \
                 * cdm_bev[
                   np.newaxis, np.newaxis, 'cp_ibp_bev_bev-alc_brf_crop_fruit', np.newaxis]
    # Overwrite
    dm_cal_dom_prod_bev['Switzerland', :,'cal_agr_domestic-production_bev', 'bev-alc'] = array_temp

    # Bev-fer : Raw materials [kcal] = product [kcal] * processing yield [%]
    array_temp = dm_cal_dom_prod_bev['Switzerland', :, 'cal_agr_domestic-production_bev',
                 'bev-fer'] \
                 * cdm_bev[
                   np.newaxis, np.newaxis, 'cp_ibp_bev_bev-fer_brf_crop_cereal', np.newaxis]
    # Overwrite
    dm_cal_dom_prod_bev['Switzerland', :,'cal_agr_domestic-production_bev', 'bev-fer'] = array_temp

    # Beer : Raw materials [kcal] = product [kcal] * processing yield [%]
    array_temp = dm_cal_dom_prod_bev['Switzerland', :, 'cal_agr_domestic-production_bev',
                 'bev-beer'] \
                 * cdm_bev[
                   np.newaxis, np.newaxis, 'cp_ibp_bev_beer_brf_crop_cereal', np.newaxis]
    # Overwrite
    dm_cal_dom_prod_bev['Switzerland', :, 'cal_agr_domestic-production_bev', 'bev-beer'] = array_temp

    # Sum crops for beverages with crops for food/feed
    # Groupby fruits or cereals
    dm_cal_dom_prod_bev.groupby({'cereal': 'bev-fer|bev-beer'}, dim='Categories1', regex=True,
                             inplace=True)
    dm_cal_dom_prod_bev.groupby({'fruit': 'bev-alc|wine'}, dim='Categories1', regex=True,
                              inplace=True)

    return dm_cal_dom_prod_bev

# CalculationLeaf SSR ALCOHOLIC BEVERAGES---------------------------------------------------------------------------------------------
def ssr_beverages_processing():
  # Read data ------------------------------------------------------------------------------------------------------------
  try:
    df_ssr = pd.read_csv(file_dict['ssr_bev'])
  except OSError:

    # FOOD BALANCE SHEETS (FBS) - For everything except molasses and cakes -------------------------------------------------
    # List of elements
    list_elements = ['Production Quantity', 'Import Quantity',
                     'Export Quantity', 'Feed', 'Processed', 'Stock Variation',
                     'Food', 'Other uses (non-food)', 'Residuals']

    list_items = ['Beer',
                  'Beverages, Alcoholic',
                  'Beverages, Fermented',
                  'Wine']

    # 1990 - 2013
    ld = faostat.list_datasets()
    code = 'FBSH'
    pars = faostat.list_pars(code)
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries_calc]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996',
                  '1997', '1998', '1999', '2000', '2001',
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
    df_ssr_1990_2013.loc[
      df_ssr_1990_2013['Element'].str.contains('Production Quantity',
                                               case=False,
                                               na=False), 'Element'] = 'Production'
    df_ssr_1990_2013.loc[
      df_ssr_1990_2013['Element'].str.contains('Import Quantity', case=False,
                                               na=False), 'Element'] = 'Import'
    df_ssr_1990_2013.loc[
      df_ssr_1990_2013['Element'].str.contains('Export Quantity', case=False,
                                               na=False), 'Element'] = 'Export'

    # 2010 - 2022

    list_elements = ['Production Quantity', 'Import quantity',
                     'Export quantity', 'Feed', 'Processed', 'Stock Variation',
                     'Food', 'Other uses (non-food)', 'Residuals']
    code = 'FBS'
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries_calc]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016',
                  '2017', '2018', '2019', '2020', '2021', '2022', '2023']
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
      df_ssr_2010_2021['Element'].str.contains('Production Quantity',
                                               case=False,
                                               na=False), 'Element'] = 'Production'
    df_ssr_2010_2021.loc[
      df_ssr_2010_2021['Element'].str.contains('Import quantity', case=False,
                                               na=False), 'Element'] = 'Import'
    df_ssr_2010_2021.loc[
      df_ssr_2010_2021['Element'].str.contains('Export quantity', case=False,
                                               na=False), 'Element'] = 'Export'
    df_ssr = pd.concat([df_ssr_1990_2013, df_ssr_2010_2021])

    # Renaming the items for name matching
    df_ssr.loc[
      df_ssr['Item'].str.contains('Rice (Milled Equivalent)', case=False,
                                      na=False, regex=False), 'Item'] = 'Rice and products'

    df_ssr.to_csv(file_dict['ssr_bev'], index=False)

  # Compute Self-Sufficiency Ratio (SSR) ---------------------------------------------------------------------------------
  # SSR [%] = (100*Production) / (Production + Imports - Exports)
  # 1: Pivot the DataFrame to get 'Production', 'Import Quantity', and 'Export Quantity' in separate columns
  df_ssr_bev = df_ssr.pivot_table(index=['Area', 'Year', 'Item'],
                                columns='Element', values='Value').reset_index()

  # Fill na with 0
  cols = [
    'Production', 'Import', 'Export', 'Food',
    'Residuals', 'Other uses (non-food)', 'Stock Variation'
  ]

  for c in cols:
    df_ssr_bev[c] = df_ssr_bev[c].fillna(0.0)

  # 2: Compute the SSR [%]
  df_ssr_bev['value'] = df_ssr_bev['Production'] / df_ssr_bev['Food']

  # Filter columns
  columns_to_filter = ['Area', 'Year', 'Item', 'value']
  df_ssr_bev = df_ssr_bev[columns_to_filter]

  # Calc Formatting ------------------------------------------------------------

  # Food item name matching with dictionary
  # Read excel file
  df_dict = pd.read_excel(
    'dictionaries/dictionnary_alcoholic-beverages.xlsx',
    sheet_name='self-sufficiency')

  # Prepend 'SSR'
  df_ssr_bev['Item'] = df_ssr_bev['Item'].apply(lambda x: f"SSR {x}")

  # Renaming existing columns (geoscale, timsecale, value)
  df_ssr_bev.rename(
    columns={'Area': 'geoscale', 'Year': 'timescale'},
    inplace=True)

  # Merge based on 'Item'
  df_ssr_bev = pd.merge(df_dict, df_ssr_bev, on='Item')

  # Drop the 'Item' column
  df_ssr_bev = df_ssr_bev.drop(columns=['Item'])

  # Adding the columns module, lever, level and string-pivot at the correct places
  lever = 'dummy'
  df_ssr_bev['module'] = lever
  df_ssr_bev['lever'] = lever
  df_ssr_bev['level'] = 0

  # Extrapolation
  df_ssr_bev = ensure_structure(df_ssr_bev)
  df_ssr_bev = linear_fitting_ots_db(df_ssr_bev, years_ots,
                                             countries='all')

  # Format as datamatrix
  df_ots, df_fts = database_to_df(df_ssr_bev, lever, level='all')
  df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
  dm_ssr_bev = DataMatrix.create_from_df(df_ots, num_cat=1)

  return dm_ssr_bev

# CalculationLeaf TRADE ORIGIN BEV
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
    df_trade_agg = pd.read_csv(file_dict['trade-bev'])
  except OSError:

    # TRADE MATRIX (TM)
    # List of elements
    list_elements = ['Import quantity']

    # List items
    # Total items FAOSTAT
    code = 'TM'
    dict_items_faostat = faostat.get_par(code, 'item')
    list_items_faostat = list(dict_items_faostat.keys())

    # Create item list for Beer
    keywords = ["beer"]
    exclude = ["none"]
    list_items_beer = [
      k for k in dict_items_faostat.keys()
      if any(word in k.lower() for word in keywords)
         and not any(bad in k.lower() for bad in exclude)
    ]

    # Create item list for Beverages, alcoholic
    keywords = ["alcohol"]
    exclude = ["non"]
    list_items_bev_alc = [
      k for k in dict_items_faostat.keys()
      if any(word in k.lower() for word in keywords)
         and not any(bad in k.lower() for bad in exclude)
    ]

    # Create item list for Beverages, fermented
    keywords = ["fermented beverages"]
    exclude = ["none"]
    list_items_bev_fer = [
      k for k in dict_items_faostat.keys()
      if any(word in k.lower() for word in keywords)
         and not any(bad in k.lower() for bad in exclude)
    ]

    # Create item list for Wine
    keywords = ["wine"]
    exclude = ["swine"]
    list_items_wine = [
      k for k in dict_items_faostat.keys()
      if any(word in k.lower() for word in keywords)
         and not any(bad in k.lower() for bad in exclude)
    ]

    dict_item_groups = {
      # Beverages
      "pro-bev-beer": list_items_beer,
      "pro-bev-bev-alc": list_items_bev_alc,
      "pro-bev-bev-fer": list_items_bev_fer,
      "pro-bev-wine": list_items_wine,
    }


    # 1990 - 2023
    ld = faostat.list_datasets()
    code = 'TM'
    pars = faostat.list_pars(code)
    my_reporter_countries = [faostat.get_par(code, 'reporterarea')[c] for c in list_countries_calc]
    my_partner_regions = [faostat.get_par(code, 'partnerregions')[p] for p in
                             list_partnerregions]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996',
                  '1997', '1998', '1999', '2000', '2001', '2002',
                  '2003', '2004', '2005', '2006', '2007', '2008', '2009',
                  '2010', '2011', '2012', '2013', '2014', '2015', '2016',
                  '2017', '2018', '2019', '2020', '2021', '2022', '2023']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]


    # Loop to download data from FAOSTAT

    dict_dfs_trade = {}

    for group_name, item_list in dict_item_groups.items():
      my_items = [
        faostat.get_par(code, 'item')[i]
        for i in item_list
      ]

      my_pars = {
        'reporterarea': my_reporter_countries,
        'partnerregions': my_partner_regions,
        'element': my_elements,
        'item': my_items,
        'year': my_years
      }

      # Download FAOSTAT data
      data = faostat.get_data(code, pars=my_pars)

      # Convert safely to DataFrame
      df = pd.DataFrame(data)

      # Change 1st row as column name
      df.columns = df.iloc[0]
      df = df.iloc[1:].reset_index(drop=True)

      # Ensure Value is numeric
      df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

      # Store dataframe
      dict_dfs_trade[group_name] = df

    # Filter & sum items per category
    col_filter = ['Reporter Countries', 'Partner Countries', 'Item', 'Year',
                  'Value']
    df_trade_agg_temp = []

    for item_name, df in dict_dfs_trade.items():
      df_tmp = (
        df[col_filter]
        .assign(Item=item_name)  # replace Item safely
        .groupby(
          ['Reporter Countries', 'Partner Countries', 'Item', 'Year'],
          as_index=False
        )['Value']
        .sum()
      )

      df_trade_agg_temp.append(df_tmp)

    # Final combined dataframe
    df_trade_agg = pd.concat(df_trade_agg_temp, ignore_index=True)
    df_trade_agg.to_csv(file_dict['trade-bev'], index=False)

  # Rename Item as variables
  df_trade_agg.rename(columns={'Item': 'variables'},inplace=True)

  # Prepend var name and unit
  df_trade_agg['variables'] = df_trade_agg['variables'].apply(lambda x: f"agr_split-import_{x}[-]")

  # Aggregate countries by region -----------------------------------------------------

  '''# Read csv
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
                               'Year': 'timescale', 'Value':'value'}, inplace=True)'''

  # Rename and format correctly
  df_trade_agg = df_trade_agg[['Partner Countries','variables','Year','Value']]
  df_trade_agg.rename(columns={'Partner Countries': 'geoscale',
                               'Year': 'timescale', 'Value': 'value'},
                      inplace=True)


  # Extrapolation for missing data
  lever = 'dummy'
  df_trade_agg['lever'] = lever
  df_trade_agg['module'] = lever
  df_trade_agg['level'] = 0.0
  df_trade_agg = ensure_structure(df_trade_agg)
  df_trade_agg = linear_fitting_ots_db(df_trade_agg, years_all, countries='all')
  df_trade_agg['value'] = df_trade_agg['value'].fillna(0.0)

  # Replace negative values by 0.0
  df_trade_agg['value'] = df_trade_agg['value'].clip(lower=0.0)

  # Format as datamatrix
  df_ots, df_fts = database_to_df(df_trade_agg, lever, level='all')
  df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
  dm_bev_trade_origin = DataMatrix.create_from_df(df_ots, num_cat=1)

  # Add Switzerland as dummy (because are in losses and other dms)
  dm_bev_trade_origin.add(0.0, dummy=True, col_label=['Switzerland'], dim='Country')

  # Unit conversion: [t] => [kcal]
  cdm_kcal_temp = cdm_kcal.copy()
  cdm_kcal_temp.rename_col_regex(str1="pro-liv-", str2="", dim="Categories1")
  cdm_kcal_temp = cdm_kcal_temp.filter({'Categories1': dm_bev_trade_origin.col_labels['Categories1']})
  dm_bev_trade_origin.sort('Categories1')
  cdm_kcal_temp.sort('Categories1')
  array_temp = dm_bev_trade_origin[:, :, 'agr_split-import', :] \
               * cdm_kcal_temp[np.newaxis, np.newaxis, 'cp_kcal-per-t', :]
  dm_bev_trade_origin[:, :, 'agr_split-import', :] = array_temp

  # Step CALIBRATION IMPORTS PER COUNTRY
  dm_cal_imports_countries = dm_bev_trade_origin.copy()
  dm_cal_imports_countries.rename_col('agr_split-import', 'cal_agr_domestic-production','Variables')
  dm_cal_imports_countries.change_unit('cal_agr_domestic-production', 1.0, '-', 'kcal', '*')
  dm_cal_imports_countries.drop(dim='Country', col_label=['Switzerland'])

  # Step CALIBRATION IMPORTS TOTAL
  dm_cal_imports_tot = dm_bev_trade_origin.copy()
  dm_cal_imports_tot.rename_col('agr_split-import', 'cal_agr_imported_production_total','Variables')
  dm_cal_imports_tot.change_unit('cal_agr_imported_production_total', 1.0, '-', 'kcal', '*')
  dm_cal_imports_tot.groupby({'Switzerland': '.*'}, dim='Country', regex=True, inplace=True)

  # Normalise across countries for share of imports
  dm_bev_trade_origin.drop(dim='Country', col_label=['Switzerland'])
  dm_bev_trade_origin.normalise(dim='Country', inplace=True)
  dm_bev_trade_origin.change_unit('agr_split-import', 1.0, '%', '-', '*')

  return dm_bev_trade_origin, dm_cal_imports_countries, dm_cal_imports_tot

# CalculationLeaf FXA PROCESSING YIELD---------------------------------------------------------------------------------------------
def fxa_processing_yield(cdm_kcal):
  # Read data ------------------------------------------------------------------------------------------------------------
  try:
    df_ssr = pd.read_csv(file_dict['ssr'])
  except OSError:

    # FOOD BALANCE SHEETS (FBS) - For everything except molasses and cakes -------------------------------------------------
    # List of elements
    list_elements = ['Production Quantity', 'Import Quantity',
                     'Export Quantity', 'Feed', 'Processed', 'Stock Variation',
                     'Food', 'Other uses (non-food)', 'Residuals']

    list_items = ['Cereals - Excluding Beer + (Total)',
                  'Fruits - Excluding Wine + (Total)', 'Oilcrops + (Total)',
                  'Pulses + (Total)', 'Rice (Milled Equivalent)',
                  'Starchy Roots + (Total)', 'Stimulants > (List)',
                  'Sugar Crops + (Total)', 'Vegetables + (Total)',
                  'Demersal Fish', 'Freshwater Fish',
                  'Aquatic Animals, Others', 'Pelagic Fish', 'Beer',
                  'Beverages, Alcoholic', 'Beverages, Fermented',
                  'Wine', 'Sugar (Raw Equivalent)', 'Sweeteners, Other',
                  'Vegetable Oils + (Total)',
                  'Milk - Excluding Butter + (Total)', 'Eggs + (Total)',
                  'Animal fats + (Total)', 'Offals + (Total)',
                  'Bovine Meat', 'Meat, Other', 'Pigmeat',
                  'Poultry Meat', 'Mutton & Goat Meat',
                  'Fish, Seafood + (Total)', 'Sugar & Sweeteners + (Total)',
                  'Grapes and products (excl wine)']

    # 1990 - 2013
    ld = faostat.list_datasets()
    code = 'FBSH'
    pars = faostat.list_pars(code)
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_partnerregions_trade]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996',
                  '1997', '1998', '1999', '2000', '2001',
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
    df_ssr_1990_2013.loc[
      df_ssr_1990_2013['Element'].str.contains('Production Quantity',
                                               case=False,
                                               na=False), 'Element'] = 'Production'
    df_ssr_1990_2013.loc[
      df_ssr_1990_2013['Element'].str.contains('Import Quantity', case=False,
                                               na=False), 'Element'] = 'Import'
    df_ssr_1990_2013.loc[
      df_ssr_1990_2013['Element'].str.contains('Export Quantity', case=False,
                                               na=False), 'Element'] = 'Export'

    # 2010 - 2022

    list_elements = ['Production Quantity', 'Import quantity',
                     'Export quantity', 'Feed', 'Processed', 'Stock Variation',
                     'Food', 'Other uses (non-food)', 'Residuals']
    # Different list becuse different in item nomination such as rice
    list_items = ['Cereals - Excluding Beer + (Total)',
                  'Fruits - Excluding Wine + (Total)', 'Oilcrops + (Total)',
                  'Pulses + (Total)', 'Rice and products',
                  'Starchy Roots + (Total)', 'Stimulants > (List)',
                  'Sugar Crops + (Total)', 'Vegetables + (Total)',
                  'Demersal Fish', 'Freshwater Fish',
                  'Aquatic Animals, Others', 'Pelagic Fish', 'Beer',
                  'Beverages, Alcoholic', 'Beverages, Fermented',
                  'Wine', 'Sugar (Raw Equivalent)', 'Sweeteners, Other',
                  'Vegetable Oils + (Total)',
                  'Milk - Excluding Butter + (Total)', 'Eggs + (Total)',
                  'Animal fats + (Total)', 'Offals + (Total)',
                  'Bovine Meat', 'Meat, Other', 'Pigmeat',
                  'Poultry Meat', 'Mutton & Goat Meat',
                  'Fish, Seafood + (Total)', 'Sugar & Sweeteners + (Total)',
                  'Grapes and products (excl wine)']
    code = 'FBS'
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries_calc]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016',
                  '2017', '2018', '2019', '2020', '2021']
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
      df_ssr_2010_2021['Element'].str.contains('Production Quantity',
                                               case=False,
                                               na=False), 'Element'] = 'Production'
    df_ssr_2010_2021.loc[
      df_ssr_2010_2021['Element'].str.contains('Import quantity', case=False,
                                               na=False), 'Element'] = 'Import'
    df_ssr_2010_2021.loc[
      df_ssr_2010_2021['Element'].str.contains('Export quantity', case=False,
                                               na=False), 'Element'] = 'Export'
    df_ssr = pd.concat([df_ssr_1990_2013, df_ssr_2010_2021])

    # Renaming the items for name matching
    df_ssr.loc[
      df_ssr['Item'].str.contains('Rice (Milled Equivalent)', case=False,
                                      na=False, regex=False), 'Item'] = 'Rice and products'

    df_ssr.to_csv(file_dict['ssr'], index=False)

  # COMMODITY BALANCES (NON-FOOD) (OLD METHODOLOGY) - For molasse and cakes ----------------------------------------------
  try:
    df_ssr_cake = pd.read_csv(file_dict['cake'])
    df_ssr_2010_2021_molasse_cake = pd.read_csv(file_dict['molasse'])
  except OSError:
    # 1990 - 2013
    list_elements = ['Production Quantity', 'Import quantity',
                     'Export quantity', 'Feed', 'Food']
    list_items = ['Copra Cake', 'Cottonseed Cake', 'Groundnut Cake',
                  'Oilseed Cakes, Other', 'Palmkernel Cake',
                  'Rape and Mustard Cake', 'Sesameseed Cake', 'Soyabean Cake',
                  'Sunflowerseed Cake']
    code = 'CBH'
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_partnerregions_trade]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996',
                  '1997', '1998', '1999', '2000', '2001',
                  '2002',
                  '2003', '2004', '2005', '2006', '2007', '2008', '2009']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
      'area': my_countries,
      'element': my_elements,
      'item': my_items,
      'year': my_years
    }
    df_ssr_1990_2013_cake = faostat.get_data_df(code, pars=my_pars,
                                                strval=False)
    # Renaming the elements
    df_ssr_1990_2013_cake.loc[
      df_ssr_1990_2013_cake['Element'].str.contains('Production Quantity',
                                                    case=False,
                                                    na=False), 'Element'] = 'Production'
    df_ssr_1990_2013_cake.loc[
      df_ssr_1990_2013_cake['Element'].str.contains('Import quantity',
                                                    case=False,
                                                    na=False), 'Element'] = 'Import'
    df_ssr_1990_2013_cake.loc[
      df_ssr_1990_2013_cake['Element'].str.contains('Export Quantity',
                                                    case=False,
                                                    na=False), 'Element'] = 'Export'

    # SUPPLY UTILIZATION ACCOUNTS (SCl) - For molasse and cakes ----------------------------------------------------------
    # 2010 - 2022
    list_elements = ['Production Quantity', 'Import quantity',
                     'Export quantity', 'Feed']
    list_items = ['Molasses', 'Cake of  linseed', 'Cake of  soya beans',
                  'Cake of copra', 'Cake of cottonseed',
                  'Cake of groundnuts', 'Cake of hempseed', 'Cake of kapok',
                  'Cake of maize', 'Cake of mustard seed',
                  'Cake of palm kernel', 'Cake of rapeseed',
                  'Cake of rice bran', 'Cake of safflowerseed',
                  'Cake of sesame seed', 'Cake of sunflower seed',
                  'Cake, oilseeds nes', 'Cake, poppy seed']
    code = 'SCL'
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_partnerregions_trade]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016',
                  '2017', '2018', '2019', '2020', '2021']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
      'area': my_countries,
      'element': my_elements,
      'item': my_items,
      'year': my_years
    }
    df_ssr_2010_2021_molasse_cake = faostat.get_data_df(code, pars=my_pars,
                                                        strval=False)

    # Renaming the elements
    df_ssr_2010_2021_molasse_cake.loc[
      df_ssr_2010_2021_molasse_cake['Element'].str.contains(
        'Production Quantity', case=False, na=False, regex=False), 'Element'] = 'Production'
    df_ssr_2010_2021_molasse_cake.loc[
      df_ssr_2010_2021_molasse_cake['Element'].str.contains('Import quantity',
                                                            case=False,
                                                            na=False, regex=False), 'Element'] = 'Import'
    df_ssr_2010_2021_molasse_cake.loc[
      df_ssr_2010_2021_molasse_cake['Element'].str.contains('Export quantity',
                                                            case=False,
                                                            na=False, regex=False), 'Element'] = 'Export'
    df_ssr_2010_2021_molasse_cake.loc[
      df_ssr_2010_2021_molasse_cake['Element'].str.contains(
        'Food supply quantity (tonnes)', case=False, na=False, regex=False
      ),
      'Element'
    ] = 'Food'

    df_ssr_1990_2013_cake.loc[
      df_ssr_1990_2013_cake['Element'].str.contains(
        'Food supply quantity (tonnes)', case=False, na=False, regex=False
      ),
      'Element'
    ] = 'Food'

    # Aggregating cakes
    df_ssr_cake = pd.concat(
      [df_ssr_1990_2013_cake, df_ssr_2010_2021_molasse_cake])

    df_ssr_cake.to_csv(file_dict['cake'], index=False)
    df_ssr_2010_2021_molasse_cake.to_csv(file_dict['molasse'], index=False)

  # Filtering
  filtered_df = df_ssr_cake[
    df_ssr_cake['Item'].str.contains('cake', case=False)]
  # Groupby Area, Year and Element and sum the Value
  grouped_df = filtered_df.groupby(['Area', 'Element', 'Year'])[
    'Value'].sum().reset_index()
  # Adding a column 'Item' containing 'Cakes' for all row, before the 'Value' column
  grouped_df['Item'] = 'Cakes'
  cols = grouped_df.columns.tolist()
  cols.insert(cols.index('Value'), cols.pop(cols.index('Item')))
  df_ssr_cake = grouped_df[cols]

  # Filtering for molasse
  df_ssr_molasses = df_ssr_2010_2021_molasse_cake[
    df_ssr_2010_2021_molasse_cake['Item'].str.contains('Molasses', case=False)]

  # Concatenating for feed
  # df_ssr = pd.concat([df_ssr, df_ssr_molasses])
  # df_ssr = pd.concat([df_ssr, df_ssr_cake])
  df_ssr_feed = pd.concat([df_ssr_molasses, df_ssr_cake])

  # Change unit from [t] => [kt]
  df_ssr_feed['Value'] = df_ssr_feed['Value'] * 10 ** (-3)

  # Filtering to keep wanted columns
  columns_to_filter = ['Area', 'Element', 'Item', 'Year', 'Value']
  df_ssr = df_ssr[columns_to_filter]
  df_ssr_feed = df_ssr_feed[columns_to_filter]

  # Concat and create copy for processing yield
  df_processing_yield_fxa = pd.concat([df_ssr, df_ssr_feed])

  # PROCESSING YIELD
  # Pivot df
  pivot_df = df_processing_yield_fxa.pivot_table(index=['Area', 'Year', 'Item'],
                                columns='Element', values='Value').reset_index()

  # Filter columns
  list_cols = ['Area', 'Year', 'Item', 'Production', 'Processing']
  pivot_df = pivot_df[list_cols]

  # Filter rows where 'Item' contains any of these terms (case-insensitive)
  list_items = ['Beer', 'Bev', 'Cereal', 'Fruit', 'Wine', 'Cake', 'Oil',
                'Molasse', 'Sugar']
  pattern = '|'.join(list_items)
  pivot_df = pivot_df[pivot_df['Item'].str.contains(pattern, case=False, na=False)]

  # Wine--------------------------------------------------------------------
  list_items = ['wine', 'grape']
  pattern = '|'.join(list_items)
  df_wine = pivot_df[pivot_df['Item'].str.contains(pattern, case=False, na=False)]

  # Extract the processing value for Oilcrops per Area & Year
  wine_proc = (
    df_wine[df_wine["Item"] == "Grapes and products (excl wine)"]
    .loc[:, ["Area", "Year", "Processing"]]
    .rename(columns={"Processing": "grapes_processing"})
  )

  # Merge it back into the original dataframe
  df_wine = df_wine.merge(wine_proc, on=["Area", "Year"], how="left")

  # Replace Processing for Wine with Oilcrops_processing
  df_wine.loc[
    df_wine["Item"].isin(["Wine"]),
    "Processing"
  ] = df_wine["grapes_processing"]

  # Drop the helper column
  df_wine = df_wine.drop(columns=["grapes_processing"])

  # Processing yields [input/output] = Processing / Production
  df_wine['value'] = df_wine['Processing'] / df_wine['Production']

  # Filter
  df_wine = df_wine[['Area', 'Year', 'Item', 'value']]


  # Sugar --------------------------------------------------------------------
  list_items = ['Sugar & Sweeteners', 'Molasse', 'Sugar Crops']
  pattern = '|'.join(list_items)
  df_sugar = pivot_df[pivot_df['Item'].str.contains(pattern, case=False, na=False)]

  # Extract the processing value for Oilcrops per Area & Year
  sugar_proc = (
    df_sugar[df_sugar["Item"] == "Sugar Crops"]
    .loc[:, ["Area", "Year", "Processing"]]
    .rename(columns={"Processing": "Sugarcrops_processing"})
  )

  # Merge it back into the original dataframe
  df_sugar = df_sugar.merge(sugar_proc, on=["Area", "Year"], how="left")

  # Replace Processing for Molasses & Sugar (Raw Equivalent) with Oilcrops_processing
  df_sugar.loc[
    df_sugar["Item"].isin(["Molasses", "Sugar & Sweeteners"]),
    "Processing"
  ] = df_sugar["Sugarcrops_processing"]

  # Drop the helper column
  df_sugar = df_sugar.drop(columns=["Sugarcrops_processing"])

  # Processing yields [input/output] = Processing / Production
  df_sugar['value'] = df_sugar['Processing'] / df_sugar['Production']

  # Filter
  df_sugar = df_sugar[['Area', 'Year', 'Item', 'value']]

  # Oilcrops --------------------------------------------------------------------
  list_items = ['Cake', 'Oil']
  pattern = '|'.join(list_items)
  df_oil = pivot_df[pivot_df['Item'].str.contains(pattern, case=False, na=False)]

  # Extract the processing value for Oilcrops per Area & Year
  oilcrops_proc = (
    df_oil[df_oil["Item"] == "Oilcrops"]
    .loc[:, ["Area", "Year", "Processing"]]
    .rename(columns={"Processing": "Oilcrops_processing"})
  )

  # Merge it back into the original dataframe
  df_oil = df_oil.merge(oilcrops_proc, on=["Area", "Year"], how="left")

  # Replace Processing for Vegetable Oils & Cakes with Oilcrops_processing
  df_oil.loc[
    df_oil["Item"].isin(["Vegetable Oils", "Cakes"]),
    "Processing"
  ] = df_oil["Oilcrops_processing"]

  # Drop the helper column
  df_oil = df_oil.drop(columns=["Oilcrops_processing"])

  # Processing yields [t input/ t output] = Processing / Production
  df_oil['value'] = df_oil['Processing'] / df_oil['Production']

  # Filter
  df_oil = df_oil[['Area', 'Year', 'Item', 'value']]

  # Calc Formatting ------------------------------------------------------------

  # Concat dfs
  df_calc_processing_yield = pd.concat([df_oil, df_sugar])
  df_calc_processing_yield = pd.concat(
    [df_calc_processing_yield, df_wine])

  # Food item name matching with dictionary
  # Read excel file
  df_dict = pd.read_excel(
    'dictionaries/dictionnary_alcoholic-beverages.xlsx',
    sheet_name='fxa')

  # Renaming existing columns (geoscale, timsecale, value)
  df_calc_processing_yield.rename(
    columns={'Area': 'geoscale', 'Year': 'timescale'},
    inplace=True)

  # Merge based on 'Item'
  df_calc_processing_yield = pd.merge(df_dict, df_calc_processing_yield, on='Item')

  # Drop the 'Item' column
  df_calc_processing_yield = df_calc_processing_yield.drop(columns=['Item'])

  # Adding the columns module, lever, level and string-pivot at the correct places
  lever = 'dummy'
  df_calc_processing_yield['module'] = lever
  df_calc_processing_yield['lever'] = lever
  df_calc_processing_yield['level'] = 0
  cols = df_calc_processing_yield.columns.tolist()
  cols.insert(cols.index('value'), cols.pop(cols.index('module')))
  cols.insert(cols.index('value'), cols.pop(cols.index('lever')))
  cols.insert(cols.index('value'), cols.pop(cols.index('level')))
  df_calc_processing_yield = df_calc_processing_yield[cols]
  df_calc_processing_yield = df_calc_processing_yield.drop_duplicates()

  # Extrapolation
  df_calc_processing_yield = ensure_structure(df_calc_processing_yield)
  df_calc_processing_yield = linear_fitting_ots_db(df_calc_processing_yield, years_all,
                                             countries='all')

  # Format as datamatrix
  df_ots, df_fts = database_to_df(df_calc_processing_yield , lever, level='all')
  df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
  dm_fxa_pro_yield = DataMatrix.create_from_df(df_ots, num_cat=1)

  # The idea is to change unit from t input / t output to kcal input / kcal output
  # because that is what is used in the Calculator
  # Yield [kcal input / kcal output] = Yield [t input / t output] * (kcal per t input) / (kcal per t output)

  # Voil
  array_temp = dm_fxa_pro_yield[:, :, 'fxa_agr_processing-yield', 'voil-to-oilcrop'] \
               * cdm_kcal[
                 np.newaxis, np.newaxis, 'cp_kcal-per-t', 'crop-oilcrop'] \
               / cdm_kcal[
                 np.newaxis, np.newaxis, 'cp_kcal-per-t', 'pro-crop-processed-voil']
  dm_fxa_pro_yield[:, :, 'fxa_agr_processing-yield','voil-to-oilcrop'] = array_temp

  # Cake
  array_temp = dm_fxa_pro_yield[:, :, 'fxa_agr_processing-yield', 'cake-to-oilcrop'] \
               * cdm_kcal[
                 np.newaxis, np.newaxis, 'cp_kcal-per-t', 'crop-oilcrop'] \
               / cdm_kcal[
                 np.newaxis, np.newaxis, 'cp_kcal-per-t', 'pro-crop-processed-cake']
  dm_fxa_pro_yield[:, :, 'fxa_agr_processing-yield','cake-to-oilcrop'] = array_temp

  # Molasse
  array_temp = dm_fxa_pro_yield[:, :, 'fxa_agr_processing-yield',
               'molasse-to-sugarcrop'] \
               * cdm_kcal[
                 np.newaxis, np.newaxis, 'cp_kcal-per-t', 'crop-sugarcrop'] \
               / cdm_kcal[
                 np.newaxis, np.newaxis, 'cp_kcal-per-t', 'pro-crop-processed-molasse']
  dm_fxa_pro_yield[:, :, 'fxa_agr_processing-yield','molasse-to-sugarcrop'] = array_temp

  # Sugar
  array_temp = dm_fxa_pro_yield[:, :, 'fxa_agr_processing-yield', 'sugar-to-sugarcrop'] \
               * cdm_kcal[
                 np.newaxis, np.newaxis, 'cp_kcal-per-t', 'crop-sugarcrop'] \
               / cdm_kcal[
                 np.newaxis, np.newaxis, 'cp_kcal-per-t', 'pro-crop-processed-sugar']
  dm_fxa_pro_yield[:, :, 'fxa_agr_processing-yield','sugar-to-sugarcrop'] = array_temp

  # Wine
  array_temp = dm_fxa_pro_yield[:, :, 'fxa_agr_processing-yield', 'wine-to-fruit'] \
               * cdm_kcal[np.newaxis, np.newaxis, 'cp_kcal-per-t', 'crop-fruit'] \
               / cdm_kcal[
                 np.newaxis, np.newaxis, 'cp_kcal-per-t', 'pro-bev-wine']
  dm_fxa_pro_yield[:, :, 'fxa_agr_processing-yield','wine-to-fruit'] = array_temp

  return dm_fxa_pro_yield

# CalculationLeaf CONSTANTS  ------------------------------

def constant():

  # KCAL TO T ----------------------------------------------------------------------------------------

  # Read excel
  df_kcal_t = pd.read_excel('data/alcoholic-beverages_constants.xlsx',
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

  # Beverages processing yield and byproducts ----------------------------------

  # Read excel
  df_cp_bev = pd.read_excel('data/alcoholic-beverages_constants.xlsx',
                            sheet_name='cp_ibp_bev')

  # Filter columns
  df_cp_bev = df_cp_bev[['variables', 'value']].copy()

  # Turn the df in a dict
  dict_cp_bev = dict(zip(df_cp_bev['variables'], df_cp_bev['value']))
  variables = df_cp_bev['variables'].tolist()

  # Format as a cdm
  cdm_bev = ConstantDataMatrix(col_labels={'Variables': variables})
  arr = np.zeros((len(cdm_bev.col_labels['Variables'])))
  cdm_bev.array = arr
  idx = cdm_bev.idx
  for var, val in dict_cp_bev.items():
    cdm_bev.array[idx[var]] = val
    cdm_bev.units[var] = "-"


  # TIME PER YEAR ----------------------------------------------------------------------------------------

  # Format as a cdm
  cdm_lifestyle = ConstantDataMatrix(col_labels={'Variables': ['cp_time_days-per-year']})
  arr = np.zeros((len(cdm_kcal.col_labels['Variables'])))
  cdm_lifestyle.array = arr
  idx = cdm_lifestyle.idx
  cdm_lifestyle.array[idx['cp_time_days-per-year']] = 365.0
  cdm_lifestyle.units["cp_time_days-per-year"] = "days/year"

  return cdm_kcal, cdm_lifestyle, cdm_bev

# CalculationLeaf FTS  ------------------------------
def fts_processing(list_countries_calc, years_ots, years_fts, cdm_kcal):

  # fwaste, diet-adherence, kcal-req, ssr-bev -------------------------------------------
  # Read Excel
  df_fts_data = pd.read_excel(
    'data/alcoholic-beverages_fts.xlsx',
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

def datamatrix_to_pickle(dm_fts, cdm_bev):

  # Make list with all years
  years_all = years_ots + years_fts

  # FixedAssumptionsToDatamatrix -----------------------------------------------
  dict_fxa = {}

  dict_fxa['processing-yield'] = dm_fxa_pro_yield
  dict_fxa['split-import'] = dm_bev_trade_origin

  # CalibrationDataToDatamatrix ------------------------------------------------

  dict_fxa['cal_agr_domestic-production_bev'] = dm_cal_dom_prod_bev
  dict_fxa['cal_agr_imports-bev_total'] = dm_cal_imports_tot


  # LeversToDatamatrix OTS -----------------------------------------------------
  dict_ots = {}

  # ssr (for alcoholic beverages)
  dict_ots['ssr-bev'] = dm_ssr_bev

  # LeversToDatamatrix FTS -----------------------------------------------------
  dict_fts = {}

  # Linear fitting between ots and fts objective (2050) ------------------

  # Lever - ssr-bev
  lever = 'ssr-bev'
  for level in range(1,5):
    # Compute the reduction objective in 2050 compared to the last ots value,
    # for each food category
    dm_ots = dict_ots[lever].copy()
    array_temp =  1 - ( 1 - dm_ots[:,years_ots[-1],'agr_ssr',:]) \
                  * dm_fts[lever][level][:,years_fts[-1],'ssr-bev', np.newaxis]
    # Append with ots
    dm_ots.add(array_temp[:,np.newaxis,np.newaxis,:], dim='Years', dummy=True, col_label=years_fts[-1])
    # Linear fit
    linear_fitting(dm_ots, years_fts)
    dm_fts[lever][level] = dm_ots.filter({'Years':years_fts}, inplace=False)
  dict_fts[lever] = dm_fts[lever]

  # ConstantsToDatamatrix ------------------------------------------------------
  dict_const = {}
  dict_const['cdm_kcal-per-t'] = cdm_kcal

  # Alcoholic beverages byproduct
  dict_const['cdm_cp_ibp_bev_beer'] = cdm_bev.filter_w_regex({'Variables': '.*beer.*'})
  dict_const['cdm_cp_ibp_bev_wine'] = cdm_bev.filter_w_regex(
    {'Variables': '.*wine.*'})
  dict_const['cdm_cp_ibp_bev_bev-fer'] = cdm_bev.filter_w_regex(
    {'Variables': '.*bev-fer.*'})
  dict_const['cdm_cp_ibp_bev_bev-alc'] = cdm_bev.filter_w_regex(
    {'Variables': '.*bev-alc.*'})

  # Group all datamatrix in a single structure ---------------------------------
  DM_diet = {
    'fxa': dict_fxa,
    'constant': dict_const,
    'fts': dict_fts,
    'ots': dict_ots
  }

  # Write datamatrix to pickle -------------------------------------------------
  f = '../../data/datamatrix/alcoholic-beverages.pickle'
  with open(f, 'wb') as handle:
    pickle.dump(DM_diet, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return


# CalculationTree RUNNING PRE-PROCESSING -----------------------------------------------------------------------------------------------
years_ots = create_years_list(1990, 2023, 1)  # make list with years from 1990 to 2015
years_fts = create_years_list(2025, 2050, 5)
years_all = years_ots + years_fts

if not os.path.exists('data/faostat'):
    os.makedirs('data/faostat')

list_countries_calc = ['Switzerland']
list_partnerregions_trade = ['Switzerland',
                         '-- Australia and New Zealand > (List)',
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

# Create files for storing data
file_dict = {'ssr': 'data/faostat/ssr.csv',
             'ssr_bev': 'data/faostat/ssr_bev.csv',
             'dom-prod-bev': 'data/faostat/dom-prod-bev.csv',
             'cake': 'data/faostat/ssr_cake.csv',
             'molasse': 'data/faostat/ssr_2010_2021_molasse_cake.csv',
             'diet': 'data/faostat/diet.csv',
             'trade-bev': 'data/faostat/trade-bev.csv'}

cdm_kcal, cdm_lifestyle, cdm_bev = constant()
dm_fts = fts_processing(list_countries_calc, years_ots, years_fts, cdm_kcal)
dm_fxa_pro_yield = fxa_processing_yield(cdm_kcal)
dm_ssr_bev = ssr_beverages_processing()
dm_cal_dom_prod_bev = bev_calibration(list_countries_calc, dm_fxa_pro_yield, cdm_bev)
dm_bev_trade_origin, dm_cal_imports_countries, dm_cal_imports_tot = trade_origin_processing(years_ots, list_countries_calc, file_dict)

# Match countries for imports
dm_match_countries(dm_bev_trade_origin, dm_fxa_pro_yield, parameter='perfect match')
dm_match_countries(dm_cal_imports_countries, dm_fxa_pro_yield, parameter='perfect match')

'''# Filter countries to match Food Balance Sheet
dm_bev_trade_origin.filter({'Country': dm_fxa_pro_yield.col_labels['Country']}, inplace=True)
countries_filter = (['Switzerland'] + dm_fxa_pro_yield.col_labels['Country'])
dm_fxa_pro_yield.filter({'Country': countries_filter}, inplace=True)'''


# CalculationTree RUNNING PICKLE CREATION
datamatrix_to_pickle(dm_fts, cdm_bev)
