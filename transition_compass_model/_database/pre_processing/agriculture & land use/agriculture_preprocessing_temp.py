import pickle

import numpy as np
from model.common.auxiliary_functions import filter_DM, linear_fitting, create_years_list, linear_fitting_ots_db
from model.common.data_matrix_class import DataMatrix
import faostat
import pandas as pd
from model.common.constant_data_matrix_class import ConstantDataMatrix

years_ots = create_years_list(1990, 2023, 1)  # make list with years from 1990 to 2015
years_fts = create_years_list(2025, 2050, 5)
years_all = years_ots + years_fts


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

def feed_workflow_new():
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
  df_feed_lsu_melted['variables'] = 'cp_agr_feed_' + df_feed_lsu_melted['variables'] + '_' + df_feed_lsu_melted['Item']+ '[kg/lsu]'
  df_feed_lsu_pathwaycalc = df_feed_lsu_melted[['variables','value']].copy()

  # Pathwaycalc formatting
  # Renaming existing columns (geoscale, timsecale, value)
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
  df_feed_lsu_pathwaycalc = linear_fitting_ots_db(df_feed_lsu_pathwaycalc, years_ots,
                                              countries='all')

  return df_feed_lsu_pathwaycalc

df_feed_lsu_pathwaycalc = feed_workflow_new()


# Load pickles
with open('../../data/datamatrix/agriculture.pickle', 'rb') as handle:
    DM_agriculture = pickle.load(handle)

with open('../../data/datamatrix/lifestyles.pickle', 'rb') as handle:
    DM_lifestyles = pickle.load(handle)



# ADDING CONSTANTS ----------------------------------------------------------------------------------------

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

# Overwrite in pickle
f = '../../data/datamatrix/agriculture.pickle'
with open(f, 'wb') as handle:
    pickle.dump(DM_agriculture, handle, protocol=pickle.HIGHEST_PROTOCOL)


print('hello')
