import pickle

import numpy as np
from model.common.auxiliary_functions import filter_DM, linear_fitting, create_years_list
from model.common.data_matrix_class import DataMatrix
import faostat
import pandas as pd


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
