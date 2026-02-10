import numpy as np
from model.common.auxiliary_functions import interpolate_nans, add_missing_ots_years, linear_fitting_ots_db, linear_fitting, create_years_list, dm_match_countries
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

# CalculationLeaf CROP YIELD

def crop_yield(dm_prod_share):

  # CROPS  (QCL) (for everything except lgn-energycrop, gas-energycrop, algae and insect)
  try:
    df_yield = pd.read_csv(file_dict['QCL_yield'])
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
    'dictionaries/dictionary_land-use.xlsx',
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
  dm_yield.filter({'Years': years_ots}, inplace=True)

  # Yields for all countries
  dm_yield_world = dm_yield.copy()
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
        'dictionaries/dictionary_land-use.xlsx',
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
                      "Epeautre", "Méteil et autres céréales panifiables", "Maïs grain",
                      'Autres céréales', "Maïs d'ensilage et maïs vert", "Houblon", "Céréales en général"],
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


# CalculationLeaf LIVESTOCK DENSITY & GRAZING INTENSITY
def livestock_density(df_liv_pop):

  try:
    df_land_use_fao = pd.read_csv(file_dict['RL_land-use'])
  except OSError:
    # Read FAO Values (for Switzerland) --------------------------------------------------------------------------------------------

    # List of elements
    list_elements = ['Area']

    list_items = ['-- Cropland', '---- Temporary crops', '---- Temporary fallow',
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
                  '2022', '2023']
    my_years = [faostat.get_par(code, 'year')[y] for y in list_years]

    my_pars = {
      'area': my_countries,
      'element': my_elements,
      'item': my_items,
      'year': my_years
    }
    df_land_use_fao = faostat.get_data_df(code, pars=my_pars, strval=False)
    df_land_use_fao.to_csv(file_dict['RL_land-use'], index=False)

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

# CalculationLeaf LOSSES ------------------------------------------------------------------------------

def livestock_losses():
  # ----------------------------------------------------------------------------------------------------------------------
  # LOSSES ---------------------------------------------------------------------------------------------------------------
  # ----------------------------------------------------------------------------------------------------------------------

  try:
    df_losses_FBS_a = pd.read_csv(file_dict['losses-FBS-a'])
    df_losses_FBS_b = pd.read_csv(file_dict['losses-FBS-b'])
    df_losses_FBSH_a = pd.read_csv(file_dict['losses-FBSH-a'])
    df_losses_FBSH_b = pd.read_csv(file_dict['losses-FBSH-b'])
    # Concatenating
    df_losses_csl = pd.concat(
      [df_losses_FBS_a, df_losses_FBS_b, df_losses_FBSH_a, df_losses_FBSH_b])
  except OSError:

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

# CalculationLeaf CAL - POP & DOM PROD -----------------------------------------------------------------------------------
# CalculationLeaf CAL - POP & DOM PROD
def livestock_calibration(list_countries_calc, dm_losses):
    # ----------------------------------------------------------------------------------------------------------------------
    # Step POPULATION ----------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    try:
      df_liv_population = pd.read_csv(file_dict['GLE_liv-pop'])
      df_liv_population_poultry = pd.read_csv(file_dict['GLE_liv-pop_poultry'])
      df_liv_population_others = pd.read_csv(file_dict['GLE_liv-pop_others'])

    except OSError:

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

    try:
      df_domestic_supply_FBS = pd.read_csv(file_dict['FBS_asf_dom-prod'])
      df_domestic_supply_FBSH = pd.read_csv(file_dict['FBSH_asf_dom-prod'])
      # Concatenating all the years together
      df_domestic_supply = pd.concat(
        [df_domestic_supply_FBS, df_domestic_supply_FBSH])

    except OSError:
      # Read data ------------------------------------------------------------------------------------------------------------

      # Common for all
      # List of countries

      # FOOD BALANCE SHEETS (FBS) - -------------------------------------------------
      # List of elements
      list_elements = ['Production Quantity']

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
    dm_losses_liv.filter({'Country':dm_cal_dom_prod.col_labels['Country']}, inplace=True)
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


# CalculationLeaf CONSTANTS  ------------------------------

def constant():
  # Beverages processing yield and byproducts ----------------------------------

  # Read excel
  df_cp_bev = pd.read_excel('data/land-use_constants.xlsx',
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
    'data/land-use_fts.xlsx',
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

  dict_fxa['yield-ch'] = dm_yield_ch
  dict_fxa['yield-imports'] = dm_yield_world

  # CalibrationDataToDatamatrix ------------------------------------------------

  dict_fxa['cal_crop-share-area'] = dm_cal_crop_area
  dict_fxa['cal_cropland_total'] = dm_cal_cropland

  # LeversToDatamatrix OTS -----------------------------------------------------
  dict_ots = {}

  # crop-share-.*
  dict_ots['crop-share-organic'] = dm_prod_share.filter({'Variables': ['agr_share_organic']})
  dict_ots['crop-share-extensive'] = dm_prod_share.filter({'Variables': ['agr_share_extensive']})
  dict_ots['crop-share-intensive'] = dm_prod_share.filter(
    {'Variables': ['agr_share_intensive']})

  # livestock-density
  dict_ots['livestock-density'] = dm_density


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

  # ConstantsToDatamatrix ------------------------------------------------------
  dict_const = {}

  dict_const['cdm_kcal-per-t'] = cdm_kcal

  # Group all datamatrix in a single structure ---------------------------------
  DM_landuse_pickle = {
    'fxa': dict_fxa,
    'constant': dict_const,
    'fts': dict_fts,
    'ots': dict_ots
  }

  # Write datamatrix to pickle -------------------------------------------------
  f = '../../data/datamatrix/land-use.pickle'
  with open(f, 'wb') as handle:
    pickle.dump(DM_landuse_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

file_dict = {'yield': 'data/faostat/yield.csv',
             'QCL_yield':'data/faostat/QCL_csv/QCL_yield.csv',
             'cropland': 'data/faostat/cropland.csv',
             'land': 'data/faostat/land.csv',
             'RL_land-use':'data/faostat/RL_csv/RL_land.csv',
             'losses-FBS-a': 'data/faostat/FBS-H_csv/FBS_losses_Africa-America-Asia.csv',
             'losses-FBS-b': 'data/faostat/FBS-H_csv/FBS_losses_Europe-Oceania.csv',
             'losses-FBSH-a': 'data/faostat/FBS-H_csv/FBSH_losses_Africa-America.csv',
             'losses-FBSH-b': 'data/faostat/FBS-H_csv/FBSH_losses_Asia-Europe-Oceania.csv',
             'FBS_asf_dom-prod': 'data/faostat/FBS-H_csv/FBS_asf_dom-prod.csv',
             'FBSH_asf_dom-prod': 'data/faostat/FBS-H_csv/FBSH_asf_dom-prod.csv',
             'GLE_liv-pop': 'data/faostat/GLE_csv/GLE_liv-pop.csv',
             'GLE_liv-pop_poultry': 'data/faostat/GLE_csv/GLE_liv-pop_poultry.csv',
             'GLE_liv-pop_others': 'data/faostat/GLE_csv/GLE_liv-pop_others.csv'
             }

cdm_kcal, cdm_bev = constant()
dm_cal_crop_area, dm_prod_share = production_share()
dm_cal_cropland = cropland_calibration(list_countries_calc)
dm_yield_world, dm_yield_ch = crop_yield(dm_prod_share)
dm_fts = fts_processing()
dm_losses = livestock_losses()
dm_cal_dom_prod, dm_cal_liv_pop, df_liv_pop = livestock_calibration(list_countries_calc, dm_losses)
dm_density = livestock_density(df_liv_pop)

# Match countries for imports
dm_match_countries(dm_yield_world, dm_losses, parameter='perfect match')
dm_match_countries(dm_cal_dom_prod, dm_losses, parameter='perfect match')
dm_match_countries(dm_cal_liv_pop, dm_losses, parameter='perfect match')

# CalculationTree RUNNING PICKLE CREATION
datamatrix_to_pickle(dm_fts)


