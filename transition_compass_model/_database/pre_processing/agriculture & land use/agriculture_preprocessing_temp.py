import pickle

import numpy as np
from model.common.auxiliary_functions import filter_DM, linear_fitting, create_years_list, linear_fitting_ots_db
from model.common.data_matrix_class import DataMatrix
import faostat
import pandas as pd
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


years_ots = create_years_list(1990, 2023, 1)  # make list with years from 1990 to 2015
years_fts = create_years_list(2025, 2050, 5)
years_all = years_ots + years_fts


# ADDING CONSTANTS ----------------------------------------------------------------------------------------
# FXA EF NITROGEN FERTILIZER ----------------------------------------------------------------------------------------
# Load data
dm_emission_fert = DM_agriculture['fxa']['cal_agr_crop_emission_N2O-emission_fertilizer']
dm_input_fert = DM_agriculture['ots']['climate-smart-crop']['climate-smart-crop_input-use']
dm_land = DM_agriculture['fxa']['cal_agr_lus_land']

# COmpute total land
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
DM_agriculture['fxa']['agr_emission_fertilizer'][:,:,'fxa_agr_emission_fertilizer'] = dm_input_fert[:,:,'fxa_agr_emission_fertilizer']

# CALIBRATION DOMESTIC PROD WITH LOSSES ----------------------------------------------------------------------------------------

# Load data
dm_dom_prod_liv = DM_agriculture['fxa']['cal_agr_domestic-production-liv'].copy()
dm_losses_liv = DM_agriculture['ots']['climate-smart-livestock']['climate-smart-livestock_losses'].copy()

# Livestock domestic prod with losses [kcal] = livestock domestic prod [kcal] * Production losses livestock [%]
dm_losses_liv.drop(dim='Categories1', col_label=['abp-processed-afat', 'abp-processed-offal'])
dm_dom_prod_liv.rename_col('cal_agr_domestic-production-liv', 'cal_agr_domestic-production-liv_raw', dim='Variables')
dm_dom_prod_liv.append(dm_losses_liv, dim='Variables')
dm_dom_prod_liv.operation('agr_climate-smart-livestock_losses', '*', 'cal_agr_domestic-production-liv_raw',
                                 out_col='cal_agr_domestic-production-liv', unit='kcal')

# Overwrite
DM_agriculture['fxa']['cal_agr_domestic-production-liv'][:, :,'cal_agr_domestic-production-liv',:] \
    = dm_dom_prod_liv[:, :,'cal_agr_domestic-production-liv',:]

# YIELD USING CALIBRATION DOMESTIC PROD WITH LOSSES ----------------------------------------------------------------------------------------

# Load data
dm_dom_prod_liv = DM_agriculture['fxa']['cal_agr_domestic-production-liv'].copy()
dm_yield = DM_agriculture['ots']['climate-smart-livestock']['climate-smart-livestock_yield'].copy()

# Yield [kcal/lsu] = Domestic prod with losses [kcal] / producing-slaugthered animals [lsu]
dm_yield.rename_col('agr_climate-smart-livestock_yield', 'agr_climate-smart-livestock_yield_raw', dim='Variables')
dm_dom_prod_liv.append(dm_yield, dim='Variables')
dm_dom_prod_liv.operation('cal_agr_domestic-production-liv', '/', 'agr_climate-smart-livestock_yield_raw',
                                 out_col='agr_climate-smart-livestock_yield', unit='kcal/lsu')

# Overwrite
DM_agriculture['ots']['climate-smart-livestock']['climate-smart-livestock_yield'][:, :,'agr_climate-smart-livestock_yield',:] \
    = dm_dom_prod_liv[:, :,'agr_climate-smart-livestock_yield',:]



# DIET ----------------------------------------------------------------------------------------

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
dm_cal_diet.add(arr, dim='Variables', col_label='cal_agr_diet_new', unit='kcal/year')

# Save in DM_agriculture
DM_agriculture['ots']['kcal-req']['Switzerland', :,'agr_kcal-req',:] = dm_req['Switzerland',:,'agr_kcal-req_temp',:]
DM_agriculture['ots']['kcal-req']['Vaud', :,'agr_kcal-req',:] = dm_req['Vaud',:,'agr_kcal-req_temp',:]
DM_agriculture['ots']['kcal-req']['EU27', :,'agr_kcal-req',:] = dm_req['EU27',:,'agr_kcal-req_temp',:]
# Overwrite shares
DM_agriculture['ots']['diet']['share']['Switzerland', :,'share',:] = dm_others['Switzerland', :,'share',:]
DM_agriculture['ots']['diet']['share']['EU27', :,'share',:] = dm_others['EU27', :,'share',:]
DM_agriculture['ots']['diet']['share']['Vaud', :,'share',:] = dm_others['Vaud', :,'share',:]
# Overwrite cal_diet
DM_agriculture['fxa']['cal_agr_diet']['Switzerland', :,'cal_agr_diet',:] = dm_cal_diet['Switzerland', :,'cal_agr_diet_new',:]
DM_agriculture['fxa']['cal_agr_diet']['EU27', :,'cal_agr_diet',:] = dm_cal_diet['EU27', :,'cal_agr_diet_new',:]
DM_agriculture['fxa']['cal_agr_diet']['Vaud', :,'cal_agr_diet',:] = dm_cal_diet['Vaud', :,'cal_agr_diet_new',:]

# Overwrite in pickle
f = '../../data/datamatrix/agriculture.pickle'
with open(f, 'wb') as handle:
    pickle.dump(DM_agriculture, handle, protocol=pickle.HIGHEST_PROTOCOL)


print('hello')
