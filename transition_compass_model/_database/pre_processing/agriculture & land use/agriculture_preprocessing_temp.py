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

# FEED - ENERGY CONVERSION EFFICIENCY  ----------------------------------------------------------------------------------------

# Note : the unit % really means unitless

# Source : Alexander, P., Brown, C., Arneth, A., Finnigan, J., Rounsevell, M.D.A., 2016.
# Human appropriation of land for food: The role of diet. Glob. Environ.
# Change 41, 88–98. https://doi.org/10.1016/j.gloenvcha.2016.09.005
DM_agriculture['constant']['cdm_cp_efficiency']['cp_efficiency_liv','abp-dairy-milk'] = 0.24
DM_agriculture['constant']['cdm_cp_efficiency']['cp_efficiency_liv','abp-hens-egg'] = 0.19
DM_agriculture['constant']['cdm_cp_efficiency']['cp_efficiency_liv','meat-bovine'] = 0.019
DM_agriculture['constant']['cdm_cp_efficiency']['cp_efficiency_liv','meat-oth-animals'] = 0.044
DM_agriculture['constant']['cdm_cp_efficiency']['cp_efficiency_liv','meat-pig'] = 0.086
DM_agriculture['constant']['cdm_cp_efficiency']['cp_efficiency_liv','meat-poultry'] = 0.13
DM_agriculture['constant']['cdm_cp_efficiency']['cp_efficiency_liv','meat-sheep'] = 0.044

# FXA EF NITROGEN FERTILIZER ----------------------------------------------------------------------------------------
# Load data
dm_emission_fert = DM_agriculture['fxa']['cal_agr_crop_emission_N2O-emission_fertilizer'].copy()
dm_input_fert = DM_agriculture['ots']['climate-smart-crop']['climate-smart-crop_input-use'].copy()
dm_land = DM_agriculture['fxa']['cal_agr_lus_land'].copy()

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
dm_cal_diet = DM_agriculture['fxa']['cal_agr_diet'].copy()

# Separate SSR of pro-crop-processed-cake, pro-crop-processed-molasse back in dm
dm_feed = dm_dom_prod.filter(
  {'Categories1': ['pro-crop-processed-cake','pro-crop-processed-molasse']})

# Rename categories
cat_diet = [
    'afats', 'beer', 'bev-alc', 'bev-fer', 'bov', 'cereals', 'cocoa', 'coffee',
    'dfish', 'egg', 'ffish', 'fruits', 'milk', 'offal', 'oilcrops', 'oth-animals',
    'oth-aq-animals', 'pfish', 'pigs', 'poultry', 'pulses', 'rice', 'seafood',
    'sheep', 'starch', 'sugar', 'sweet', 'tea', 'veg', 'voil', 'wine'
]
cat_agr = [
    'pro-liv-abp-processed-afat', 'pro-bev-beer', 'pro-bev-bev-alc', 'pro-bev-bev-fer',
    'pro-liv-meat-bovine', 'crop-cereal', 'cocoa', 'coffee', 'dfish',
    'pro-liv-abp-hens-egg', 'ffish', 'crop-fruit', 'pro-liv-abp-dairy-milk',
    'pro-liv-abp-processed-offal', 'crop-oilcrop', 'pro-liv-meat-oth-animals',
    'oth-aq-animals', 'pfish', 'pro-liv-meat-pig', 'pro-liv-meat-poultry',
    'crop-pulse', 'crop-rice', 'seafood', 'pro-liv-meat-sheep', 'crop-starch',
    'pro-crop-processed-sugar', 'pro-crop-processed-sweet', 'tea', 'crop-veg',
    'pro-crop-processed-voil', 'pro-bev-wine'
]
dm_cal_diet.rename_col(cat_diet, cat_agr, 'Categories1')

# Unit conversion: [kt] => [kcal]
# Convert from [kcal] to [t]
idx_dm = dm_dom_prod.idx
idx_cdm = cdm_kcal.idx
array_temp = dm_dom_prod.array[:, :,
             idx_dm['agr_food-net-import'], :] \
             / cdm_kcal.array[idx_cdm['cp_kcal-per-t'], :]
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
dm_dom_prod.operation('agr_food-net-import_kcal', '/', 'cal_agr_diet', dim='Variables',
                          out_col='agr_food-net-import', unit='%')
dm_dom_prod = dm_dom_prod.filter(
  {'Variables': ['agr_food-net-import']})

# Add SSR of pro-crop-processed-cake, pro-crop-processed-molasse back in dm
dm_dom_prod.append(dm_feed, dim='Categories1')

# Overwrite
DM_agriculture['ots']['food-net-import']['Switzerland', :,'agr_food-net-import',:] = dm_dom_prod['Switzerland', :,'agr_food-net-import',:]

print('hello')
