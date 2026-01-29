import numpy as np
from model.common.auxiliary_functions import interpolate_nans, add_missing_ots_years, linear_fitting_ots_db, linear_fitting, create_years_list
#from _database.pre_processing.api_routines_CH import get_data_api_CH
from scipy.stats import linregress
import pandas as pd
import faostat
import copy
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

# CalculationLeaf CROP LOSSES

def crop_losses():
  # FOOD BALANCE SHEETS (FBS) - For everything  -------------------------------------------------
  try:
    df_losses = pd.read_csv(file_dict['losses'])
  except OSError:
    # List of elements
    list_elements = ['Losses', 'Production Quantity']

    list_items = ['Cereals - Excluding Beer + (Total)',
                  'Fruits - Excluding Wine + (Total)', 'Oilcrops + (Total)',
                  'Pulses + (Total)', 'Rice (Milled Equivalent)',
                  'Starchy Roots + (Total)', 'Sugar Crops + (Total)',
                  'Vegetables + (Total)' ]

    # 1990 - 2013
    code = 'FBSH'
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_partnerregions_trade]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996',
                  '1997', '1998', '1999', '2000', '2001',
                  '2002', '2003', '2004', '2005', '2006', '2007', '2008',
                  '2009']
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
    list_items = ['Cereals - Excluding Beer + (Total)',
                  'Fruits - Excluding Wine + (Total)', 'Oilcrops + (Total)',
                  'Pulses + (Total)', 'Rice and products',
                  'Starchy Roots + (Total)', 'Sugar Crops + (Total)',
                  'Vegetables + (Total)', ]
    code = 'FBS'
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_partnerregions_trade]
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
    df_losses_2010_2021 = faostat.get_data_df(code, pars=my_pars, strval=False)

    # Renanming rice to have same name with other df
    df_losses_1990_2013 = df_losses_1990_2013.copy()
    df_losses_1990_2013['Item'] = df_losses_1990_2013['Item'].replace(
      'Rice (Milled Equivalent)', 'Rice and products'
    )

    # Concatenating
    df_losses = pd.concat([df_losses_1990_2013, df_losses_2010_2021])
    df_losses.to_csv(file_dict['losses'], index=False)

  # Compute losses ([%] of production) -----------------------------------------------------------------------------------
  # Losses [%] = 1 / (1 - Losses [1000t] / Production [1000t]) (pre processing for multiplicating the workflow)

  # 1: Pivot the DataFrame
  pivot_df = df_losses.pivot_table(index=['Area', 'Year', 'Item'],
                                   columns='Element',
                                   values='Value').reset_index()

  # 2: Compute the Losses [%] (really it's unit less)
  pivot_df['Losses[%]'] = 1 / (1 - pivot_df['Losses'] / pivot_df['Production'])

  # Drop the columns Production, Import Quantity and Export Quantity
  pivot_df = pivot_df.drop(columns=['Production', 'Losses'])

  # PathwayCalc formatting -----------------------------------------------------------------------------------------------

  # Food item name matching with dictionary
  # Read excel file
  df_dict_csc = pd.read_excel(
    'dictionaries/dictionary_crop.xlsx',
    sheet_name='climate-smart-crops')

  # Prepend 'Losses'
  pivot_df['Item'] = pivot_df['Item'].apply(lambda x: f"Losses {x}")

  # Merge based on 'Item'
  df_losses_pathwaycalc = pd.merge(df_dict_csc, pivot_df, on='Item')

  # Drop the 'Item' column
  df_losses_pathwaycalc = df_losses_pathwaycalc.drop(columns=['Item'])

  # Renaming existing columns (geoscale, timsecale, value)
  df_losses_pathwaycalc.rename(
    columns={'Area': 'geoscale', 'Year': 'timescale', 'Losses[%]': 'value'},
    inplace=True)

  # Convert to datamatrix
  lever = 'dummy'
  df_losses_pathwaycalc['lever'] = lever
  df_losses_pathwaycalc['module'] = lever
  df_losses_pathwaycalc['level'] = 0.0
  df_losses_pathwaycalc = ensure_structure(df_losses_pathwaycalc)
  df_ots, df_fts = database_to_df(df_losses_pathwaycalc, lever,
                                  level='all')
  df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
  dm_losses= DataMatrix.create_from_df(df_ots, num_cat=1)

  linear_fitting(dm_losses, years_ots)

  return dm_losses

# CalculationLeaf CROP YIELD

def crop_yield(dm_prod_share):

  # CROPS  (QCL) (for everything except lgn-energycrop, gas-energycrop, algae and insect)
  try:
    df_yield = pd.read_csv(file_dict['yield'])
  except OSError:
    # List of elements
    list_elements = ['Yield']

    list_items = ['Cereals, primary + (Total)',
                  'Fruit Primary + (Total)',
                  'Rice',
                  'Oilcrops, Oil Equivalent + (Total)',
                  'Pulses, Total + (Total)', 'Rice',
                  'Roots and Tubers, Total + (Total)',
                  'Sugar Crops Primary + (Total)',
                  'Vegetables Primary + (Total)']

    """list_items = ['Cereals, primary + (Total)',
                  'Fibre Crops, Fibre Equivalent + (Total)',
                  'Fruit Primary + (Total)',
                  'Oilcrops, Oil Equivalent + (Total)',
                  'Pulses, Total + (Total)', 'Rice',
                  'Roots and Tubers, Total + (Total)',
                  'Sugar Crops Primary + (Total)',
                  'Vegetables Primary + (Total)']"""

    # 1990 - 2022
    code = 'QCL'
    my_countries = [faostat.get_par(code, 'area')[c] for c in list_partnerregions_trade]
    my_elements = [faostat.get_par(code, 'elements')[e] for e in list_elements]
    my_items = [faostat.get_par(code, 'item')[i] for i in list_items]
    list_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996',
                  '1997', '1998', '1999', '2000', '2001',
                  '2002', '2003', '2004', '2005', '2006', '2007', '2008',
                  '2009', '2010', '2011', '2012', '2013',
                  '2014', '2015', '2016', '2017', '2018', '2019', '2020',
                  '2021', '2022', '2023']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
      'area': my_countries,
      'element': my_elements,
      'item': my_items,
      'year': my_years
    }
    df_yield = faostat.get_data_df(code, pars=my_pars, strval=False)
    df_yield.to_csv(file_dict['yield'], index=False)

  # Unit conversion from [kg/ha] to [kcal/ha]  ----------------------------------------------------------------------------

  # Pivot the DataFrame
  pivot_df = df_yield.pivot_table(index=['Area', 'Year', 'Item'],
                                            columns='Element',
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
  pivot_df_yield = pd.concat([pivot_df_yield, df_fibre.copy()],
                             ignore_index=True)

  # Create a dummy for Rice as no products
  # Create a DataFrame for the new "Rice" rows
  new_rows = pivot_df_yield[['Area', 'Year']].drop_duplicates().copy()
  # new_rows['Item'] = 'Rice and products' # If rice is missing in Switzerland
  # new_rows['Losses[%]'] = 0

  # Append the new rows to the original DataFrame
  # pivot_df_yield = pd.concat([pivot_df_yield, new_rows], ignore_index=True)

  # PathwayCalc formatting -----------------------------------------------------------------------------------------------

  # Food item name matching with dictionary
  # Read excel file
  df_dict_csc = pd.read_excel(
    'dictionaries/dictionary_crop.xlsx',
    sheet_name='climate-smart-crops')

  # Prepend 'Yield'
  pivot_df_yield['Item'] = pivot_df_yield['Item'].apply(lambda x: f"Yield {x}")

  # Merge based on 'Item'
  df_yield_pathwaycalc = pd.merge(df_dict_csc, pivot_df_yield, on='Item')

  # Drop the 'Item' column
  df_yield_pathwaycalc = df_yield_pathwaycalc.drop(columns=['Item'])

  # Renaming existing columns (geoscale, timsecale, value)
  df_yield_pathwaycalc.rename(
    columns={'Area': 'geoscale', 'Year': 'timescale', 'Yield': 'value'},
    inplace=True)

  # Convert to datamatrix
  lever = 'dummy'
  df_yield_pathwaycalc['lever'] = lever
  df_yield_pathwaycalc['module'] = lever
  df_yield_pathwaycalc['level'] = 0.0
  df_yield_pathwaycalc = ensure_structure(df_yield_pathwaycalc)
  df_ots, df_fts = database_to_df(df_yield_pathwaycalc, lever,
                                  level='all')
  df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
  dm_yield = DataMatrix.create_from_df(df_ots, num_cat=1)

  # Yields for all countries without Switzerland
  dm_yield_world = dm_yield.copy()
  dm_yield_world.drop(dim='Country', col_label='Switzerland')
  linear_fitting(dm_yield_world, years_all)

  # Step CH: Yield evolution_o/i & _e/i (organic/extensive with respect to intensive) [-]
  # fixme Source: find correct source
  yield_evolution_o = {'cereal': 1.0,
                      'sugarcrop': 1.0,
                      'oilcrop': 1.0,
                      'veg': 1.0,
                      'fruit': 1.0,
                      'starch': 1.0,
                      'pulse': 1.0}
  yield_evolution_e = {'cereal': 1.0,
                      'sugarcrop': 1.0,
                      'oilcrop': 1.0,
                      'veg': 1.0,
                      'fruit': 1.0,
                      'starch': 1.0,
                      'pulse': 1.0}

  # Format
  dm_yield_ch = dm_yield.filter({'Country':['Switzerland']})
  dm_yield_ch.drop(dim='Categories1', col_label='rice')

  # Intensive yield_i [kcal/ha] = yield_T / [share_i + evol_o*share_o + evol_e*share_e]
  dm_yield_ch.rename_col('agr_crop_yield',
                          'agr_crop_yield_total',
                          dim='Variables')
  dm_yield_ch.add(0.0, dim='Variables', dummy=True,
                   col_label='agr_crop_yield_intensive')
  for cat in dm_yield_ch.col_labels['Categories1']:
    dm_yield_ch['Switzerland', :, 'agr_crop_yield_intensive', cat] = \
      dm_yield_ch['Switzerland', :, 'agr_crop_yield_total', cat] \
      / (dm_prod_share['Switzerland', :, 'agr_share_intensive', cat] +
         dm_prod_share['Switzerland', :, 'agr_share_organic', cat] * yield_evolution_o[cat] +
         dm_prod_share['Switzerland', :, 'agr_share_extensive', cat] * yield_evolution_e[cat])

  # Organic yield_c [kcal/lsu] =  yield_i [kcal/lsu] * yield evolution_o/i [-]
  dm_yield_ch.add(0.0, dim='Variables', dummy=True,
                   col_label='agr_crop_yield_organic')
  for cat in dm_yield_ch.col_labels['Categories1']:
    dm_yield_ch[:, :, 'agr_crop_yield_organic', cat] = \
      dm_yield_ch[:, :, 'agr_crop_yield_intensive', cat] \
      * yield_evolution_o[cat]

  # Extensive yield_c [kcal/lsu] =  yield_i [kcal/lsu] * yield evolution_o/i [-]
  dm_yield_ch.add(0.0, dim='Variables', dummy=True,
                   col_label='agr_crop_yield_extensive')
  for cat in dm_yield_ch.col_labels['Categories1']:
    dm_yield_ch[:, :, 'agr_crop_yield_extensive', cat] = \
      dm_yield_ch[:, :, 'agr_crop_yield_intensive', cat] \
      * yield_evolution_e[cat]

  # Format yield
  dm_yield_ch.filter({'Variables': ['agr_crop_yield_organic',
                                 'agr_crop_yield_extensive',
                                 'agr_crop_yield_intensive',
                                 'agr_crop_yield_total']}, inplace=True)
  linear_fitting(dm_yield_ch, years_all)

  return dm_yield_world, dm_yield_ch

# CalculationLeaf CAL - DOM PROD CROP & BEV
def crop_calibration(list_countries_calc, dm_losses, dm_fxa_pro_yield, cdm_bev):

    # ----------------------------------------------------------------------------------------------------------------------
    # DOMESTIC PRODUCTION (CROP PRODUCTS) ----------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    try:
      df_domestic_supply = pd.read_csv(file_dict['dom-prod-crop'])
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
      df_domestic_supply.to_csv(file_dict['dom-prod-crop'], index=False)

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
        'dictionaries/dictionary_crop.xlsx',
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

    # Format as datamatrix - Cal dom prod crop
    df_cal_dom_prod_crop = df_cal_dom_prod[
        ~df_cal_dom_prod['variables'].str.contains('cal_agr_domestic-production_bev', case=False, na=False)
    ].copy() # filter what does not contain _bev
    df_ots, df_fts = database_to_df(df_cal_dom_prod_crop, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm_cal_dom_prod_crop = DataMatrix.create_from_df(df_ots, num_cat=1)

    '''# Crop domestic prod with losses [kcal] = crop domestic prod [kcal] * Production losses crop [%]
    dm_cal_dom_prod_crop.rename_col('cal_agr_domestic-production_food',
                                'cal_agr_domestic-production_withoutafw',
                                dim='Variables')
    list_cat_crop = dm_cal_dom_prod_crop.col_labels['Categories1']
    dm_cal_dom_prod_crop.append(dm_losses.filter({'Country':['Switzerland'], 'Categories1': list_cat_crop}), dim='Variables')
    dm_cal_dom_prod_crop.operation('agr_crop_losses', '*',
                               'cal_agr_domestic-production_withoutafw',
                               out_col='cal_agr_domestic-production',
                               unit='kcal')
    dm_cal_dom_prod_crop.filter({'Variables':['cal_agr_domestic-production']}, inplace=True)'''

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
    array_temp = dm_cal_dom_prod_bev[:, :, 'cal_agr_domestic-production_bev',
                 'wine'] \
                 * dm_fxa_pro_yield_temp[:,:, 'fxa_agr_processing-yield', 'wine-to-fruit']
    # Overwrite
    dm_cal_dom_prod_bev['Switzerland', :,'cal_agr_domestic-production_bev', 'wine'] = array_temp

    # Bev-alc : Raw materials [kcal] = product [kcal] * processing yield [%]
    array_temp = dm_cal_dom_prod_bev[:, :, 'cal_agr_domestic-production_bev',
                 'bev-alc'] \
                 * cdm_bev[
                   np.newaxis, np.newaxis, 'cp_ibp_bev_bev-alc_brf_crop_fruit', np.newaxis]
    # Overwrite
    dm_cal_dom_prod_bev['Switzerland', :,'cal_agr_domestic-production_bev', 'bev-alc'] = array_temp

    # Bev-fer : Raw materials [kcal] = product [kcal] * processing yield [%]
    array_temp = dm_cal_dom_prod_bev[:, :, 'cal_agr_domestic-production_bev',
                 'bev-fer'] \
                 * cdm_bev[
                   np.newaxis, np.newaxis, 'cp_ibp_bev_bev-fer_brf_crop_cereal', np.newaxis]
    # Overwrite
    dm_cal_dom_prod_bev['Switzerland', :,'cal_agr_domestic-production_bev', 'bev-fer'] = array_temp

    # Beer : Raw materials [kcal] = product [kcal] * processing yield [%]
    array_temp = dm_cal_dom_prod_bev[:, :, 'cal_agr_domestic-production_bev',
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
    # cal_crop total = cal_crop_food (actually also includes feed) + cal_crop_bev
    array_temp_cereal = dm_cal_dom_prod_bev[:, :,
                        'cal_agr_domestic-production_bev', 'cereal'] \
                        + dm_cal_dom_prod_crop[:, :, 'cal_agr_domestic-production',
                          'cereal']
    dm_cal_dom_prod_crop[:, :, 'cal_agr_domestic-production','cereal'] = array_temp_cereal
    array_temp_fruit = dm_cal_dom_prod_bev[:, :, 'cal_agr_domestic-production_bev',
                       'fruit'] \
                       + dm_cal_dom_prod_crop[:, :, 'cal_agr_domestic-production',
                         'fruit']
    dm_cal_dom_prod_crop[:, :, 'cal_agr_domestic-production', 'fruit'] = array_temp_fruit

    return dm_cal_dom_prod_crop, dm_cal_dom_prod_bev

# CalculationLeaf CAL - CROPLAND
def cropland_calibration(list_countries):
    # ----------------------------------------------------------------------------------------------------------------------
    # CROPLAND ----------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # CROPS  (QCL) (for everything except lgn-energycrop, gas-energycrop, algae and insect)
    try:
        df_cropland = pd.read_csv(file_dict['cropland'])
    except OSError:
        # List of elements
        list_elements = ['Area harvested']

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
        df_cropland = faostat.get_data_df(code, pars=my_pars, strval=False)
        df_cropland.loc[
            df_cropland['Item'].str.contains('Rice', case=False,
                                                    na=False), 'Item'] = 'Rice and products'
        df_cropland.to_csv(file_dict['cropland'], index=False)

    # Filter columns
    list_filter = ['Area', 'Item', 'Year', 'Value']
    df_cropland = df_cropland[list_filter]

    # Prepend "Cropland" to each value in the 'Item' column
    df_cropland['Item'] = df_cropland['Item'].apply(lambda x: f"Cropland {x}")

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------

    # Food item name matching with dictionary
    # Read excel file
    df_dict_csc = pd.read_excel(
        'dictionaries/dictionary_crop.xlsx',
        sheet_name='calibration')

    # Merge based on 'Item'
    df_cropland_pathwaycalc = pd.merge(df_dict_csc, df_cropland, on='Item')

    # Drop the 'Item' column
    df_cropland_pathwaycalc = df_cropland_pathwaycalc.drop(columns=['Item'])

    # Renaming existing columns (geoscale, timsecale, value)
    df_cropland_pathwaycalc.rename(columns={'Area': 'geoscale', 'Year': 'timescale', 'Value': 'value'}, inplace=True)

    # Adding the columns module, lever, level and string-pivot at the correct places
    lever = 'dummy'
    df_cropland_pathwaycalc['module'] = 'agriculture'
    df_cropland_pathwaycalc['lever'] = lever
    df_cropland_pathwaycalc['level'] = 0

    # Extrapolation
    df_cropland_pathwaycalc = linear_fitting_ots_db(df_cropland_pathwaycalc,
                                                     years_ots,
                                                     countries='all')

    # Format as datamatrix
    df_ots, df_fts = database_to_df(df_cropland_pathwaycalc, lever,
                                    level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm_cal_cropland = DataMatrix.create_from_df(df_ots, num_cat=1)

    return dm_cal_cropland

# CalculationLeaf FXA - PROCESSING YIELD CROP & BEV
def fxa_processing_yield(df_processing_yield_fxa, cdm_kcal):

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

  # Replace Processing for Wine with grapes_processing
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
    'dictionaries/dictionary_crop.xlsx',
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
  df_calc_processing_yield['module'] = 'agriculture'
  df_calc_processing_yield['lever'] = lever
  df_calc_processing_yield['level'] = 0

  # Extrapolation
  df_calc_processing_yield = linear_fitting_ots_db(df_calc_processing_yield, years_ots,
                                             countries='all')

  # Format as datamatrix
  df_ots, df_fts = database_to_df(df_calc_processing_yield, lever, level='all')
  df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
  dm_fxa_pro_yield = DataMatrix.create_from_df(df_ots, num_cat=1)
  linear_fitting(dm_fxa_pro_yield, years_all)

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

# CalculationLeaf SSR CROP PROD & BEV
def self_sufficiency_processing(years_ots, list_countries_calc, file_dict):
    # Read data ------------------------------------------------------------------------------------------------------------
    try:
        df_ssr = pd.read_csv(file_dict['ssr-crop'])
    except OSError:

        # FOOD BALANCE SHEETS (FBS) - For everything except molasses and cakes -------------------------------------------------
        # List of elements
        list_elements = ['Production Quantity', 'Import Quantity', 'Export Quantity', 'Feed', 'Processed', 'Stock Variation', 'Food', 'Other uses (non-food)', 'Residuals']

        list_items = ['Cereals - Excluding Beer + (Total)', 'Fruits - Excluding Wine + (Total)', 'Oilcrops + (Total)',
                      'Pulses + (Total)', 'Rice (Milled Equivalent)',
                      'Starchy Roots + (Total)', 'Stimulants > (List)', 'Sugar Crops + (Total)', 'Vegetables + (Total)',
                      'Beer', 'Beverages, Alcoholic', 'Beverages, Fermented',
                      'Wine', 'Sugar (Raw Equivalent)', 'Sweeteners, Other', 'Vegetable Oils + (Total)',
                      'Sugar & Sweeteners + (Total)', 'Grapes and products (excl wine)']

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
        list_items = ['Cereals - Excluding Beer + (Total)',
                      'Fruits - Excluding Wine + (Total)', 'Oilcrops + (Total)',
                      'Pulses + (Total)', 'Rice and products',
                      'Starchy Roots + (Total)', 'Stimulants > (List)',
                      'Sugar Crops + (Total)', 'Vegetables + (Total)',
                      'Beer', 'Beverages, Alcoholic', 'Beverages, Fermented',
                      'Wine', 'Sugar (Raw Equivalent)', 'Sweeteners, Other',
                      'Vegetable Oils + (Total)',
                      'Sugar & Sweeteners + (Total)',
                      'Grapes and products (excl wine)']
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

        df_ssr.to_csv(file_dict['ssr-crop'], index=False)

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

    # Create a copy to check imports equivalence between FBS & TM
    df_imports = pivot_df[['Area', 'Year', 'Item','Import']].copy()

    # 2: Compute the SSR [%]
    # (previously with special condition for milk because we
    # don't account for it as feed & processed. but now fixed with and fxa_ratio)
    pivot_df['SSR[%]'] = pivot_df['Production'] / (pivot_df['Food'] + pivot_df['Feed'] + pivot_df['Processing'])
    df_imports['SSR[%]'] = df_imports['Import']

    # Filter columns
    columns_to_filter = ['Area', 'Year', 'Item', 'SSR[%]']
    pivot_df = pivot_df[columns_to_filter]
    df_imports = df_imports[columns_to_filter]

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------

    # Food item name matching with dictionary
    # Read excel file
    df_dict_ssr = pd.read_excel(
        'dictionaries/dictionary_crop.xlsx',
        sheet_name='self-sufficiency')

    # Prepend 'SSR'
    pivot_df['Item'] = pivot_df['Item'].apply(lambda x: f"SSR {x}")
    df_imports['Item'] = df_imports['Item'].apply(lambda x: f"SSR {x}")

    # Renaming existing columns (geoscale, timsecale, value)
    pivot_df.rename(columns={'Area': 'geoscale', 'Year': 'timescale', 'SSR[%]': 'value'}, inplace=True)
    df_imports.rename(
      columns={'Area': 'geoscale', 'Year': 'timescale', 'SSR[%]': 'value'},
      inplace=True)

    # Merge based on 'Item'
    df_ssr_crop = pd.merge(df_dict_ssr, pivot_df, on='Item')
    df_imports = pd.merge(df_dict_ssr, df_imports, on='Item')

    # Drop the 'Item' column
    df_ssr_crop = df_ssr_crop.drop(columns=['Item'])
    df_imports = df_imports.drop(columns=['Item'])

    # Adding the columns module, lever, level and string-pivot at the correct places
    lever = 'dummy'
    df_ssr_crop['module'] = 'agriculture'
    df_ssr_crop['lever'] = lever
    df_ssr_crop['level'] = 0

    df_imports['module'] = 'agriculture'
    df_imports['lever'] = lever
    df_imports['level'] = 0

    # Extrapolation
    df_ssr_crop = linear_fitting_ots_db(df_ssr_crop, years_ots, countries='all')

    # Format as datamatrix - SSR crop
    df_ots, df_fts = database_to_df(df_ssr_crop, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm_ssr_crop = DataMatrix.create_from_df(df_ots, num_cat=1)
    linear_fitting(dm_ssr_crop, years_ots)

    # Format as datamatrix - Imports
    df_ots, df_fts = database_to_df(df_imports, lever, level='all')
    df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
    dm_imports_fbs = DataMatrix.create_from_df(df_ots, num_cat=1)
    linear_fitting(dm_imports_fbs, years_ots)

    # Unit conversion: [kt] => [kcal]
    cdm_kcal_temp = cdm_kcal.copy()
    #cdm_kcal_temp.rename_col_regex(str1="pro-liv-", str2="", dim="Categories1")
    list_cat_imports = dm_imports_fbs.col_labels['Categories1']
    cdm_kcal_temp = cdm_kcal_temp.filter(
      {'Categories1': list_cat_imports})
    dm_imports_fbs.sort('Categories1')
    cdm_kcal_temp.sort('Categories1')
    array_temp = 1000 * dm_imports_fbs[:, :, 'agr_ssr', :] \
                 * cdm_kcal_temp[np.newaxis, np.newaxis, 'cp_kcal-per-t', :]
    dm_imports_fbs[:, :, 'agr_ssr', :] = array_temp

    return dm_ssr_crop, df_processing_yield_fxa, dm_imports_fbs


# CalculationLeaf FXA - SHARE EXPORTS
def exports_processing(list_countries_calc, file_dict):
    # Read data ------------------------------------------------------------------------------------------------------------
    try:
        df_exports = pd.read_csv(file_dict['exports'])
    except OSError:

        # FOOD BALANCE SHEETS (FBS) - For everything except molasses and cakes -------------------------------------------------
        # List of elements
        list_elements = ['Production Quantity', 'Export Quantity']

        list_items = ['Cereals - Excluding Beer + (Total)',
                  'Fruits - Excluding Wine + (Total)', 'Oilcrops + (Total)',
                  'Pulses + (Total)', 'Rice (Milled Equivalent)',
                  'Starchy Roots + (Total)', 'Sugar Crops + (Total)',
                  'Vegetables + (Total)']

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
        list_items = ['Cereals - Excluding Beer + (Total)',
                  'Fruits - Excluding Wine + (Total)', 'Oilcrops + (Total)',
                  'Pulses + (Total)', 'Rice and products',
                  'Starchy Roots + (Total)', 'Sugar Crops + (Total)',
                  'Vegetables + (Total)']
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
        'dictionaries/dictionary_crop.xlsx',
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
    df_trade_agg = pd.read_csv(file_dict['trade-crop'])
  except OSError:

    # TRADE MATRIX (TM)
    # List of elements
    list_elements = ['Import quantity']

    # List items
    # Total items FAOSTAT
    code = 'TM'
    dict_items_faostat = faostat.get_par(code, 'item')
    list_items_faostat = list(dict_items_faostat.keys())

    # Create item list for fruits
    keywords = ["fruit"]
    exclude = ["feed"]
    list_items_fruit = [
      k for k in dict_items_faostat.keys()
      if any(word in k.lower() for word in keywords)
         and not any(bad in k.lower() for bad in exclude)
    ]

    # Create item list for Cereals - excluding beer
    keywords = ["cereal", "wheat", "barley", "oat", "grain", "fonio", "millet",
                "maize", "rye", "sorghum"]
    exclude = ["straw", "beer"]
    list_items_cereal = [
      k for k in dict_items_faostat.keys()
      if any(word in k.lower() for word in keywords)
         and not any(bad in k.lower() for bad in exclude)
    ]

    # Create item list for Oilcrops
    keywords = ["coconut", "olive", "soya bean",
                "canola", "seed", "groundnut", "sesame"]
    exclude = ["oil", "cake"]
    list_items_oilcrop = [
      k for k in dict_items_faostat.keys()
      if any(word in k.lower() for word in keywords)
         and not any(bad in k.lower() for bad in exclude)
    ]

    # Create item list for Processed voil
    keywords = ["oil of", "coconut oil", "palm oil", "olive oil", "soya bean oil",
                "canola oil", "seed oil", "groundnut oil"]
    exclude = ["rice"]
    list_items_pro_voil = [
      k for k in dict_items_faostat.keys()
      if any(word in k.lower() for word in keywords)
         and not any(bad in k.lower() for bad in exclude)
    ]

    # Create item list for Pulses
    keywords = ["pulse", "beans", "peas"]
    exclude = ["vegetables", "oil", "cake", "cocoa"]
    list_items_pulse = [
      k for k in dict_items_faostat.keys()
      if any(word in k.lower() for word in keywords)
         and not any(bad in k.lower() for bad in exclude)
    ]

    # Create item list for Rice
    keywords = ["rice"]
    exclude = ["beverages", "cake", "oil", "paper"]
    list_items_rice = [
      k for k in dict_items_faostat.keys()
      if any(word in k.lower() for word in keywords)
         and not any(bad in k.lower() for bad in exclude)
    ]

    # Create item list for Starchy Roots
    keywords = ["root", "potatoe", "cassava", "yam"]
    exclude = ["cigars", "vegetables", "string"]
    list_items_starch = [
      k for k in dict_items_faostat.keys()
      if any(word in k.lower() for word in keywords)
         and not any(bad in k.lower() for bad in exclude)
    ]

    # Create item list for Sugarcrops
    keywords = ["sugar cane", "sugar beet"]
    exclude = ["none"]
    list_items_sugarcrop = [
      k for k in dict_items_faostat.keys()
      if any(word in k.lower() for word in keywords)
         and not any(bad in k.lower() for bad in exclude)
    ]

    # Create item list for Vegetables
    keywords = ['Artichokes',
    'Asparagus',
    'Cabbages',
    'Carrots and turnips',
    'Cauliflowers and broccoli',
    'Cucumbers and gherkins',
    'Eggplants (aubergines)',
    'Green garlic',
    'Leeks',
    'Lettuce',
    'Mushrooms',
    'Okra',
    'Onions',
    'Spinach',
    'String beans',
    'Tomatoes',
    'Pumpkins, squash and gourds',]
    exclude = ["sugar", "oil"]
    list_items_vegetables = [
      k for k in dict_items_faostat.keys()
      if any(word in k.lower() for word in keywords)
         and not any(bad in k.lower() for bad in exclude)
    ]

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

    # Create item list for Cocoa and products
    keywords = ["cocoa"]
    exclude = ["cake"]
    list_items_cocoa = [
      k for k in dict_items_faostat.keys()
      if any(word in k.lower() for word in keywords)
         and not any(bad in k.lower() for bad in exclude)
    ]

    # Create item list for Coffee
    keywords = ["coffee"]
    exclude = ["subsitute", "extract"]
    list_items_coffee = [
      k for k in dict_items_faostat.keys()
      if any(word in k.lower() for word in keywords)
         and not any(bad in k.lower() for bad in exclude)
    ]

    # Create item list for Tea
    keywords = ["tea"]
    exclude = ["stearine"]
    list_items_tea = [
      k for k in dict_items_faostat.keys()
      if any(word in k.lower() for word in keywords)
         and not any(bad in k.lower() for bad in exclude)
    ]

    # Create item list for Processed sugar
    keywords = ["refined sugar", "refined cane or beet sugar",
                "raw cane or beet sugar"]
    exclude = ["syrup"]
    list_items_pro_sugar = [
      k for k in dict_items_faostat.keys()
      if any(word in k.lower() for word in keywords)
         and not any(bad in k.lower() for bad in exclude)
    ]

    # Create item list for Processed sweeteners
    keywords = ["honey", "syrup"]
    exclude = ["none"]
    list_items_pro_sweet = [
      k for k in dict_items_faostat.keys()
      if any(word in k.lower() for word in keywords)
         and not any(bad in k.lower() for bad in exclude)
    ]

    dict_item_groups = {
      # Crops
      "crop-fruit": list_items_fruit,
      "crop-cereal": list_items_cereal,
      "crop-oilcrop": list_items_oilcrop,
      "crop-pulse": list_items_pulse,
      "crop-rice": list_items_rice,
      "crop-starch": list_items_starch,
      "crop-sugarcrop": list_items_sugarcrop,
      "crop-veg": list_items_vegetables,

      # Processed crop products
      "pro-crop-processed-voil": list_items_pro_voil,
      "pro-crop-processed-sugar": list_items_pro_sugar,
      "pro-crop-processed-sweet": list_items_pro_sweet,

      # Beverages
      "pro-bev-beer": list_items_beer,
      "pro-bev-bev-alc": list_items_bev_alc,
      "pro-bev-bev-fer": list_items_bev_fer,
      "pro-bev-wine": list_items_wine,

      # Stimulants
      "stm-cocoa": list_items_cocoa,
      "stm-coffee": list_items_coffee,
      "stm-tea": list_items_tea,
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
    df_trade_agg.to_csv(file_dict['trade-crop'], index=False)

  # Rename Item as variables
  df_trade_agg.rename(columns={'Item': 'variables'},inplace=True)

  # Prepend var name and unit
  df_trade_agg['variables'] = df_trade_agg['variables'].apply(lambda x: f"agr_split-import_{x}[-]")

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
  dm_crop_trade_origin = DataMatrix.create_from_df(df_ots, num_cat=1)

  # Add Switzerland as dummy (because are in losses and other dms)
  dm_crop_trade_origin.add(0.0, dummy=True, col_label=['Switzerland'], dim='Country')

  # Unit conversion: [t] => [kcal]
  cdm_kcal_temp = cdm_kcal.copy()
  cdm_kcal_temp.rename_col_regex(str1="pro-liv-", str2="", dim="Categories1")
  cdm_kcal_temp = cdm_kcal_temp.filter({'Categories1': dm_crop_trade_origin.col_labels['Categories1']})
  dm_crop_trade_origin.sort('Categories1')
  cdm_kcal_temp.sort('Categories1')
  array_temp = dm_crop_trade_origin[:, :, 'agr_split-import', :] \
               * cdm_kcal_temp[np.newaxis, np.newaxis, 'cp_kcal-per-t', :]
  dm_crop_trade_origin[:, :, 'agr_split-import', :] = array_temp

  # Step CALIBRATION IMPORTS PER COUNTRY
  dm_cal_imports_countries = dm_crop_trade_origin.copy()
  dm_cal_imports_countries.rename_col('agr_split-import', 'cal_agr_domestic-production','Variables')
  dm_cal_imports_countries.change_unit('cal_agr_domestic-production', 1.0, '-', 'kcal', '*')
  dm_cal_imports_countries.drop(dim='Country', col_label=['Switzerland'])

  # Step CALIBRATION IMPORTS TOTAL
  dm_cal_imports_tot = dm_crop_trade_origin.copy()
  dm_cal_imports_tot.rename_col('agr_split-import', 'cal_agr_imported_production_total','Variables')
  dm_cal_imports_tot.change_unit('cal_agr_imported_production_total', 1.0, '-', 'kcal', '*')
  dm_cal_imports_tot.groupby({'Switzerland': '.*'}, dim='Country', regex=True, inplace=True)

  # Normalise across countries for share of imports
  dm_crop_trade_origin.drop(dim='Country', col_label=['Switzerland'])
  dm_crop_trade_origin.normalise(dim='Country', inplace=True)
  dm_crop_trade_origin.change_unit('agr_split-import', 1.0, '%', '-', '*')

  return dm_crop_trade_origin, dm_cal_imports_countries, dm_cal_imports_tot

# CalculationLeaf AREA & SHARE PRODUCTION METHOD

def production_share():

  # Step DATA CROPS
  # Source: Exploitations agricoles et surface agricole utile (SAU) selon le niveau de classification 3 par canton
  # https://www.pxweb.bfs.admin.ch/pxweb/fr/px-x-0702000000_106/px-x-0702000000_106/px-x-0702000000_106.px

  table_id = 'px-x-0702000000_106'
  file = 'data/stat-tab/crop_area.pickle'

  try:
    with open(file, 'rb') as handle:
      dm_crop_area = pickle.load(handle)
      print(
        f'The crops are read from file {file}. Delete it if you want to update data from api.')
  except OSError:
    structure, title = get_data_api_CH(table_id, mode='example', language='fr')
    # The table is too big to be downloaded at once
    filtering = {"Unité d'observation": structure["Unité d'observation"],
                 'Canton': ['Suisse'],
                 'Zone de production agricole': [
                   'Zone de production agricole - total'],
                 "Système d'exploitation": ['Système d\'exploitation - total', 'Exploitations biologiques',
                                            'Exploitations conventionnelles', 'Indéterminé'],
                 "Forme d'exploitation": ["Forme d'exploitation - total"],
                 'Année': structure['Année']}
    mapping_dim = {'Country': 'Canton',
                   'Years': 'Année',
                   'Variables': 'Zone de production agricole',
                   'Categories1': "Unité d'observation",
                   'Categories2': "Système d'exploitation"}
    dm_crop_area = get_data_api_CH(table_id, mode='extract', filter=filtering,
                         mapping_dims=mapping_dim,
                         units=['ha'], language='fr')
    dm_crop_area.drop(dim='Categories1',
            col_label=['Exploitations', 'SAU - Total (en ha)'])
    dm_crop_area.rename_col_regex('SAU - ', '', dim='Categories1')
    dm_crop_area.rename_col_regex(' (en ha)', '', dim='Categories1')
    dm_crop_area.rename_col('Suisse', 'Switzerland',
                  'Country')
    dm_crop_area.rename_col('Zone de production agricole - total','agr_land-use',
                            dim='Variables')
    dm_crop_area.rename_col('Système d\'exploitation - total', 'total',
                  'Categories2')
    dm_crop_area.rename_col('Exploitations biologiques', 'organic',
                  'Categories2')
    dm_crop_area.rename_col('Exploitations conventionnelles', 'intensive',
                            'Categories2')
    dm_crop_area.rename_col('Indéterminé', 'extensive',
                            'Categories2')

    cat_map = {
      "cereal": ["Blé", "Orge", "Avoine", "Seigle", "Triticale",
                      "Epeautre", 'Céréales en général',
                      "Méteil et autres céréales panifiables", "Maïs grain",
                      'Autres céréales', "Maïs d'ensilage et maïs vert", "Houblon"],
      "fruit": ["Baies annuelles", "Cultures de baies sous abri",
                     'Cultures fruitières en général', 'Pommes',
                     'Poires', 'Fruits à noyaux', 'Baies pluriannuelles', 'Vigne'],
      "oilcrop": ["Colza pour matière première renouvelable",
                       "Tournesol pour matière première renouvelable",
                       "Lin", "Chanvre", 'Colza pour huile comestible',
                       'Tournesol pour huile comestible',
                       'Courge à huile'],
      "pulse": ['Pois protéagineux', 'Féveroles',
                     'Légumineuses en général', 'Lupin fourrager', "Soja"],
      "starch": ["Pommes de terre"],
      "sugarcrop": ["Betteraves sucrières","Betteraves fourragères"],
      "veg": ["Cultures maraîchères de plein champ",
                   "Cultures maraîchères sous abri", "Asperges",
                   "Rhubarbe"],
      "remove": ["Plantes aromatiques et médicinales annuelles",
                 "Plantes aromatiques et médicinales pluriannuelles",
                 "Arbrisseaux ornementaux", "Sapins de Noël",
                 "Pépinières forestières hors forêt sur SAU",
                 "Autres pépinières", "Prairies artificielles", "Pâturages",
                 "Prairies extensives", "Prairies peu intensives",
                 "Prairies dans la région d'estivage",
                 "Autres prairies permanentes",
                 "Surfaces à litières",
                 "Haies, bosquets champêtres et berges boisées",
                 "Matières premières renouvelables annuelles",
                 "Matières premières renouvelables pluriannuelles",
                 "Autres SAU", "Tabac", "Jachère", "Autres terres ouvertes",
                 'Méteil et autres céréales fourragères'],
      "other": ["Cultures horticoles de plein champ annuelles",
                "Cultures horticoles sous abri", "Autres cultures pérennes",
                "Autres cultures sous abri", "Cultures sous abri en général"]
    }

    dm_crop_area.groupby(cat_map, dim='Categories1', inplace=True)
    dm_crop_area.drop(dim='Categories1', col_label=['remove'])
    dm_crop_area.drop(dim='Categories1', col_label=['other'])

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, file)
    with open(f, 'wb') as handle:
      pickle.dump(dm_crop_area, handle, protocol=pickle.HIGHEST_PROTOCOL)


  # Linear fitting
  linear_fitting(dm_crop_area, years_ots)
  dm_crop_area.filter({'Years': years_ots}, inplace=True)

  # Compute share of organic, intensive and extensive (we assume undertermined)
  # crop area

  # Step CAL ORGANIC, INTENSIVE, EXTENSIVE CROPS
  # Note: for later in landuse module?
  # Create copy for calibration
  dm_cal_crop_area = dm_crop_area.filter({'Categories2': ['intensive','extensive','organic']})
  dm_cal_crop_area.rename_col('agr_land-use', 'agr_cropland', dim='Variables')
  dm_cal_crop_area.switch_categories_order(cat1='Categories2', cat2='Categories1')
  #dm_cal_crop_area = dm_cal_crop_area.flatten()
  #dm_cal_crop_area.rename_col_regex('organic_', '', dim='Categories1')
  #dm_cal_crop_area.rename_col_regex('agr_livestock', 'cal_agr_liv-population_organic', dim='Variables')

  # Step SHARE ORGANIC, INTENSIVE, EXTENSIVE CROPS
  dm_crop_area.switch_categories_order(cat1='Categories2', cat2='Categories1')
  dm_prod_share = dm_crop_area.flattest()
  dm_prod_share.deepen()
  dm_prod_share.operation('agr_land-use_organic', '/',
                            'agr_land-use_total',
                            out_col='agr_share_organic', unit='-')
  dm_prod_share.operation('agr_land-use_extensive', '/',
                          'agr_land-use_total',
                          out_col='agr_share_extensive', unit='-')
  dm_prod_share.operation('agr_land-use_intensive', '/',
                          'agr_land-use_total',
                          out_col='agr_share_intensive', unit='-')


  # Filter
  dm_prod_share.filter({'Variables': ['agr_share_organic',
                                      'agr_share_extensive',
                                      'agr_share_intensive']}, inplace=True)

  return dm_cal_crop_area, dm_prod_share





# CalculationLeaf CONSTANTS  ------------------------------

def constant():
  # Beverages processing yield and byproducts ----------------------------------

  # Read excel
  df_cp_bev = pd.read_excel('data/crop_constants.xlsx',
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

  # KCAL TO T ------------------------------------------------------------------

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

  return cdm_kcal, cdm_bev

# CalculationLeaf FTS
def fts_processing():

  # Read Excel
  df_fts_data = pd.read_excel(
    'data/crop_fts.xlsx',
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

# CalculationLeaf PICKLE CREATION

def datamatrix_to_pickle(dm_fts):

  # Make list with all years
  years_all = years_ots + years_fts

  # FixedAssumptionsToDatamatrix -----------------------------------------------
  dict_fxa = {}

  dict_fxa['processing-yield'] = dm_fxa_pro_yield
  dict_fxa['split-import'] = dm_crop_trade_origin
  dict_fxa['share-export'] = dm_fxa_exports
  dict_fxa['yield-ch'] = dm_yield_ch
  dict_fxa['yield-imports'] = dm_yield_world

  # CalibrationDataToDatamatrix ------------------------------------------------

  dict_fxa['cal_agr_domestic-production_food'] = dm_cal_dom_prod_crop
  dict_fxa['cal_agr_domestic-production_bev'] = dm_cal_dom_prod_bev
  dict_fxa['cal_agr_imports-crop_total'] = dm_cal_imports_tot
  dict_fxa['cal_agr_imports-crop_countries'] = dm_cal_imports_countries
  dict_fxa['cal_crop-share-area'] = dm_cal_crop_area
  dict_fxa['cal_cropland_total'] = dm_cal_cropland

  # LeversToDatamatrix OTS -----------------------------------------------------
  dict_ots = {}

  # ssr-crop_.*
  #dict_ots['ssr-crop-cereal'] =
  dict_ots['ssr-crop-cereal'] = dm_ssr_crop.filter({'Categories1': ['crop-cereal']})
  dict_ots['ssr-crop-fruit'] = dm_ssr_crop.filter({'Categories1': ['crop-fruit']})
  dict_ots['ssr-crop-veg'] = dm_ssr_crop.filter({'Categories1': ['crop-veg']})
  dict_ots['ssr-crop-pulse'] = dm_ssr_crop.filter({'Categories1': ['crop-pulse']})
  dict_ots['ssr-crop-oilcrop'] = dm_ssr_crop.filter({'Categories1': ['crop-oilcrop']})
  dict_ots['ssr-crop-sugarcrop'] = dm_ssr_crop.filter({'Categories1': ['crop-sugarcrop']})
  dict_ots['ssr-crop-starch'] = dm_ssr_crop.filter({'Categories1': ['crop-starch']})
  dict_ots['ssr-crop-rice'] = dm_ssr_crop.filter({'Categories1': ['crop-rice']})
  #dict_ots['ssr-crop-stm'] = dm_ssr_crop.filter({'Categories1': ['crop-stm']})

  # ssr-bev-.*
  dict_ots['ssr-bev-beer'] = dm_ssr_crop.filter({'Categories1': ['pro-bev-beer']})
  dict_ots['ssr-bev-bev-alc'] = dm_ssr_crop.filter({'Categories1': ['pro-bev-bev-alc']})
  dict_ots['ssr-bev-bev-fer'] = dm_ssr_crop.filter({'Categories1': ['pro-bev-bev-fer']})
  dict_ots['ssr-bev-wine'] = dm_ssr_crop.filter({'Categories1': ['pro-bev-wine']})

  # ssr-pro-.*
  dict_ots['ssr-pro-sugar'] = dm_ssr_crop.filter(
    {'Categories1': ['pro-crop-processed-sugar']})
  dict_ots['ssr-pro-sweet'] = dm_ssr_crop.filter(
    {'Categories1': ['pro-crop-processed-sweet']})
  dict_ots['ssr-pro-voil'] = dm_ssr_crop.filter(
    {'Categories1': ['pro-crop-processed-voil']})

  # crop-losses
  dict_ots['crop-losses'] = dm_losses

  # crop-share-.*
  dict_ots['crop-share-organic'] = dm_prod_share.filter({'Variables': ['agr_share_organic']})
  dict_ots['crop-share-extensive'] = dm_prod_share.filter({'Variables': ['agr_share_extensive']})
  dict_ots['crop-share-intensive'] = dm_prod_share.filter(
    {'Variables': ['agr_share_intensive']})


  # LeversToDatamatrix FTS -----------------------------------------------------
  dict_fts = {}

  # FTS linear fitting of ots
  DM_ots = dict_ots.copy()

  # Adding a new lever with dummy values
  for lever_temp in dict_ots.keys():
    dict_fts[lever_temp] = {lever_temp: dict()}

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

  # Lever - ssr-crop-.*
  dict_lever_ssr_crop = {
    lever: value
    for lever, value in dict_ots.items()
    if 'ssr-crop' in lever
  }
  for lever in dict_lever_ssr_crop:
    # Create a copy across all dimensions to not have issues
    dm_fts[lever] = copy.deepcopy(dm_fts['ssr-crop'])
    # Create new variable name
    var = "agr_ssr_" + lever.replace("ssr-", "", 1)
    for level in range(1,5):
      dm_fts[lever][level].rename_col('agr_ssr', var,'Variables')
      dm_fts[lever][level].deepen()
      dm_fts[lever][level].append(dict_ots[lever], dim='Years')
      linear_fitting(dm_fts[lever][level], years_fts)
      dm_fts[lever][level].filter({'Years':years_fts}, inplace=True)
    dict_fts[lever] = dm_fts[lever]

  # Lever - ssr-bev-.*
  dict_lever_ssr_bev = {
    lever: value
    for lever, value in dict_ots.items()
    if 'ssr-bev' in lever
  }
  for lever in dict_lever_ssr_bev:
    # Create a copy across all dimensions to not have issues
    dm_fts[lever] = copy.deepcopy(dm_fts['ssr-bev'])
    # Create new variable name
    var = "agr_ssr_pro-" + lever.replace("ssr-", "", 1)
    for level in range(1,5):
      dm_fts[lever][level].rename_col('agr_ssr', var,'Variables')
      dm_fts[lever][level].deepen()
      dm_fts[lever][level].append(dict_ots[lever], dim='Years')
      linear_fitting(dm_fts[lever][level], years_fts)
      dm_fts[lever][level].filter({'Years':years_fts}, inplace=True)
    dict_fts[lever] = dm_fts[lever]

  # Lever - ssr-pro-.*
  dict_lever_ssr_pro = {
    lever: value
    for lever, value in dict_ots.items()
    if 'ssr-pro' in lever
  }
  for lever in dict_lever_ssr_pro:
    # Create a copy across all dimensions to not have issues
    dm_fts[lever] = copy.deepcopy(dm_fts['ssr-pro'])
    # Create new variable name
    var = "agr_ssr_pro-crop-processed-" + lever.replace("ssr-pro-", "", 1)
    for level in range(1,5):
      dm_fts[lever][level].rename_col('agr_ssr', var,'Variables')
      dm_fts[lever][level].deepen()
      dm_fts[lever][level].append(dict_ots[lever], dim='Years')
      linear_fitting(dm_fts[lever][level], years_fts)
      dm_fts[lever][level].filter({'Years':years_fts}, inplace=True)
    dict_fts[lever] = dm_fts[lever]

  # Lever - crop-share-intensive
  lever = 'crop-share-intensive'
  for level in range(1,5):
    # Propagate the overall lever value across all categories
    dm_ots = dict_ots[lever].copy()
    dm_fts_temp = dm_fts[lever][level]
    array_temp =  dm_fts[lever][level][:,years_fts[-1],'agr_share_intensive', np.newaxis] + \
                  dm_ots[:,years_ots[-1],'agr_share_intensive',:] - \
                  dm_ots[:,years_ots[-1],'agr_share_intensive',:] # +x-x To get the correct structure
    # Append with ots
    dm_ots.add(array_temp[:,np.newaxis,np.newaxis,:], dim='Years', dummy=True, col_label=years_fts[-1])
    # Linear fit
    linear_fitting(dm_ots, years_fts)
    dm_fts[lever][level] = dm_ots.filter({'Years':years_fts}, inplace=False)
  dict_fts[lever] = dm_fts[lever]

  # Lever - crop-share-extensive
  lever = 'crop-share-extensive'
  for level in range(1,5):
    # Propagate the overall lever value across all categories
    dm_ots = dict_ots[lever].copy()
    dm_fts_temp = dm_fts[lever][level]
    array_temp =  dm_fts[lever][level][:,years_fts[-1],'agr_share_extensive', np.newaxis] + \
                  dm_ots[:,years_ots[-1],'agr_share_extensive',:] - \
                  dm_ots[:,years_ots[-1],'agr_share_extensive',:] # +x-x To get the correct structure
    # Append with ots
    dm_ots.add(array_temp[:,np.newaxis,np.newaxis,:], dim='Years', dummy=True, col_label=years_fts[-1])
    # Linear fit
    linear_fitting(dm_ots, years_fts)
    dm_fts[lever][level] = dm_ots.filter({'Years':years_fts}, inplace=False)
  dict_fts[lever] = dm_fts[lever]

  # Lever - crop-share-organic
  lever = 'crop-share-organic'
  for level in range(1,5):
    # Propagate the overall lever value across all categories
    dm_ots = dict_ots[lever].copy()
    dm_fts_temp = dm_fts[lever][level]
    array_temp =  dm_fts[lever][level][:,years_fts[-1],'agr_share_organic', np.newaxis] + \
                  dm_ots[:,years_ots[-1],'agr_share_organic',:] - \
                  dm_ots[:,years_ots[-1],'agr_share_organic',:] # +x-x To get the correct structure
    # Append with ots
    dm_ots.add(array_temp[:,np.newaxis,np.newaxis,:], dim='Years', dummy=True, col_label=years_fts[-1])
    # Linear fit
    linear_fitting(dm_ots, years_fts)
    dm_fts[lever][level] = dm_ots.filter({'Years':years_fts}, inplace=False)
  dict_fts[lever] = dm_fts[lever]

  # Lever - crop-losses
  lever = 'crop-losses'
  for level in range(1,5):
    # Compute the reduction objective in 2050 compared to the last ots value,
    # for each food category
    dm_ots = dict_ots[lever].copy()
    array_temp =  1 - ( 1 - dm_ots[:,years_ots[-1],'agr_crop_losses',:]) \
                  * dm_fts[lever][level][:,years_fts[-1],'agr_crop_losses', np.newaxis]
    # Append with ots
    dm_ots.add(array_temp[:,np.newaxis,np.newaxis,:], dim='Years', dummy=True, col_label=years_fts[-1])
    # Linear fit
    linear_fitting(dm_ots, years_fts)
    dm_fts[lever][level] = dm_ots.filter({'Years':years_fts}, inplace=False)
  dict_fts[lever] = dm_fts[lever]

  """# Lever - ssr-liv
  dict_lever_ssr_liv = ['ssr-liv-abp-dairy-milk',
                        'ssr-liv-abp-hens-egg',
                        'ssr-liv-meat-poultry',
                        'ssr-liv-meat-pig',
                        'ssr-liv-meat-bovine',
                        'ssr-liv-meat-sheep',
                        'ssr-liv-meat-oth-animal']
  for lever in dict_lever_ssr_liv:
    for level in range(1,5):
      dm_fts[lever][level].deepen()
      dm_fts[lever][level].append(dict_ots[lever], dim='Years')
      linear_fitting(dm_fts[lever][level], years_fts)
      dm_fts[lever][level].filter({'Years':years_fts}, inplace=True)
    dict_fts[lever] = dm_fts[lever]"""

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
  DM_crop_pickle = {
    'fxa': dict_fxa,
    'constant': dict_const,
    'fts': dict_fts,
    'ots': dict_ots
  }

  # Write datamatrix to pickle -------------------------------------------------
  f = '../../data/datamatrix/crop.pickle'
  with open(f, 'wb') as handle:
    pickle.dump(DM_crop_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

file_dict = {'losses': 'data/faostat/losses.csv',
             'cake': 'data/faostat/ssr_cake.csv',
             'feed-pro': 'data/faostat/ssr_feed_pro.csv',
             'molasse': 'data/faostat/ssr_2010_2021_molasse_cake.csv',
             'yield': 'data/faostat/yield.csv',
             'ssr-crop': 'data/faostat/ssr-crop.csv',
             'dom-prod-crop': 'data/faostat/dom-prod-crop.csv',
             'trade-crop': 'data/faostat/trade-crop.csv',
             'exports': 'data/faostat/exports.csv',
             'cropland': 'data/faostat/cropland.csv',
             'urea': 'data/faostat/urea.csv',
             'land': 'data/faostat/land.csv',
             'nitro': 'data/faostat/nitro.csv',
             'pesticide': 'data/faostat/pesticide.csv',
             'liming': 'data/faostat/liming.csv'}

cdm_kcal, cdm_bev = constant()
dm_losses = crop_losses()
dm_ssr_crop, df_processing_yield_fxa, dm_imports_fbs = self_sufficiency_processing(years_ots, list_countries_calc, file_dict)
dm_fxa_pro_yield = fxa_processing_yield(df_processing_yield_fxa, cdm_kcal)
dm_cal_dom_prod_crop, dm_cal_dom_prod_bev = crop_calibration(list_countries_calc, dm_losses, dm_fxa_pro_yield, cdm_bev)
dm_crop_trade_origin, dm_cal_imports_countries, dm_cal_imports_tot = trade_origin_processing(years_ots, list_countries_calc, file_dict)
dm_cal_crop_area, dm_prod_share = production_share()
dm_cal_cropland = cropland_calibration(list_countries_calc)
dm_yield_world, dm_yield_ch = crop_yield(dm_prod_share)
dm_fxa_exports = exports_processing(list_countries_calc,file_dict)
dm_fts = fts_processing()

# CalculationTree RUNNING PICKLE CREATION
datamatrix_to_pickle(dm_fts)


