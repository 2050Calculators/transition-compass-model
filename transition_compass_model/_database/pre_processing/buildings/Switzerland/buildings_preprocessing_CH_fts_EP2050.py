# -------------------------------------------------------------------------
#    IMPLEMENTING EP2050 SCENARIO
# -------------------------------------------------------------------------
import copy
import pickle
import numpy as np
import pandas as pd
from model.common.auxiliary_functions import linear_fitting, my_pickle_dump, filter_DM, create_years_list, sort_pickle
from model.common.data_matrix_class import DataMatrix
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go



# --------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------

#Loading the data
data_file = '../../../data/datamatrix/buildings.pickle'
with open(data_file, 'rb') as handle:
    DM_buildings = pickle.load(handle)

data_file_2 = '../../../data/datamatrix/lifestyles.pickle'
with open(data_file_2, 'rb') as handle:
    DM_lifestyles = pickle.load(handle)


#Creation of DataMatrix for the 4 scenarios
#DM_fts = {'fts': dict()}

#Path for the csv files
csv_path = 'data/EP2050/'


# --------------------------------------------------------------------------------
# CONSTANTS & PARAMETERS
# --------------------------------------------------------------------------------
years_fts = [2025, 2030, 2035, 2040, 2045, 2050]

#variables nomenclature
name_mapping = {
    "efficiency of space heating systems": "effshs",
    "specific space heating": "ssh",
    "heating structure": "heatstruct"
}

fuel_mapping = {
    "Central Heating oil": "oil",
    "Gas": "gas",
    "Wood": "wood",
    "Electricity": "elec",
    "HP": "hp",
    "DH": "dh",
    "other": "other"
}

scenarios_mapping = {
    "ZERO_basis": "0basis",
    "ZERO_A": "a",
    "ZERO_B": "b",
    "ZERO_C": "c"
}

bld_type_mapping = {
    "SFH": "single-family-households",
    "MFH": "multi-family-households"
}
# --------------------------------------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------------------------------------

#Plot function
#another file for python

# --------------------------------------------------------------------------------
# SIDE NOTES
# --------------------------------------------------------------------------------
#DataMatrix.create_from_df(df, num_cat=1)
# si num_cat = 1, tout ce qui est avant le _ devient une variable, et le reste devient une catégorie

# --------------------------------------------------------------------------------
# BUILDING SECTOR - MAIN
# --------------------------------------------------------------------------------

#Filtering the former datamatrix in order only to consider Switzerland
filter_DM(DM_buildings,{'Country':['Switzerland']})
filter_DM(DM_lifestyles,{'Country':['Switzerland']})
DM_buildings_old = copy.deepcopy(DM_buildings)
#Swiss population since 2000
#Mettre toute les modif dans le code au lieu que dans l'Excel
dm_pop = DM_lifestyles['ots']['pop']['lfs_population_']['lfs_population_total']


# --------------------------------------------------------------------------------
# EP2050+ general data
# --------------------------------------------------------------------------------

#Mettre script pour dm_ep2050 ?
#Script pour population ?


# --------------------------------------------------------------------------------
# 1-FLOOR INTENSITY
# --------------------------------------------------------------------------------

## 1.1 'lfs_floor-intensity_space-cap'
df_floor_intensity = pd.read_csv(csv_path+'floor_intensity.csv')
df_floor_intensity = df_floor_intensity[['Country', 'Years', 'lfs_floor-intensity_space-cap[m2/cap]']]
dm_floor_intensity = DataMatrix.create_from_df(df_floor_intensity, num_cat=0)
dm_floor_intensity.filter({'Years': years_fts}, inplace=True)

dm_ots = DM_buildings['ots']['floor-intensity'].filter({'Variables': ['lfs_floor-intensity_space-cap']})

dm_ots.append(dm_floor_intensity, dim='Years')
idx = dm_ots.idx
dm_ots.array[0,idx[2025]:idx[2050],...]=np.nan
dm_ots.fill_nans('Years')
dm_ots.filter({'Years': years_fts}, inplace=True)
dm_ots.append(DM_buildings['fts']['floor-intensity'][3].filter({'Variables': ['lfs_household-size']}), dim='Variables')

#Considering EP2050+ values as scenario 1 (BAU)
#DM_buildings['fts']['floor-intensity'][1] = dm_ots

#Trasnferring former scenario 1 to scenario 2 as values are lower than EP ones
DM_buildings['fts']['floor-intensity'][2] = DM_buildings['fts']['floor-intensity'][3]

dm_sre_dls = DM_buildings['fts']['floor-intensity'][3].filter({'Variables': ['lfs_floor-intensity_space-cap']})
df_sre_dls = dm_sre_dls.write_df()
df_sre_dls['lfs_floor-intensity_space-cap[m2/cap]']  = df_sre_dls.apply(lambda row: 35.0 if row['Years'] == 2050 else np.nan, axis=1)
dm_sre_dls = DataMatrix.create_from_df(df_sre_dls, num_cat = 0)

dm_ots_dls = DM_buildings['ots']['floor-intensity'].filter({'Variables': ['lfs_floor-intensity_space-cap']})
dm_ots_dls.append(dm_sre_dls, dim='Years')
idx = dm_ots_dls.idx
dm_ots_dls.array[0,idx[2025]:idx[2050],...]=np.nan
dm_ots_dls.fill_nans('Years')
dm_ots_dls.filter({'Years': years_fts}, inplace=True)
dm_ots_dls.append(DM_buildings['fts']['floor-intensity'][3].filter({'Variables': ['lfs_household-size']}), dim='Variables')
DM_buildings['fts']['floor-intensity'][4] = dm_ots_dls


#df_floor_area = pd.read_csv(csv_path+'floor_intensity.csv')
#df_floor_area.rename({'Year': 'Years'}, axis=1, inplace=True)
#df_floor_area.drop(columns=['comment'],inplace=True)
#dm_area = DataMatrix.create_from_df(df_floor_area, num_cat=0)
#dm_area.change_unit('era_SFH', factor=1e6, old_unit='Mm2', new_unit='m2')
#dm_area.change_unit('era_MFH', factor=1e6, old_unit='Mm2', new_unit='m2')

#dm_area.rename_col_regex('era', 'bld_floor-area_stock', 'Variables')
#dm_area.rename_col_regex('bld_building_mix_new', 'bld_floor-area_new', 'Variables')
#dm_floor_area = dm_area.filter_w_regex({'Variables': 'bld_floor-area.*'})
#dm_floor_area.rename_col_regex('MFH', 'multi-family-households', dim='Variables')
#dm_floor_area.rename_col_regex('SFH', 'single-family-households', dim='Variables')
#dm_floor_area.deepen()

# DM_fts['fts']['floor-intensity'][1]['lfs_floor-intensity_space-cap']
# = DM_buildings['fts']['floor-intensity'][1].filter({'Variables': ['lfs_floor-intensity_space-cap']})

# --------------------------------------------------------------------------------
# 2 - HEATING/COOLING BEHAVIOUR
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# 3 - BUILDING RENOVATION RATE
# --------------------------------------------------------------------------------

### 3.1 bld_building_mix ###
# State : Done

df_bld_building_mix = pd.read_csv( csv_path + 'bld_building-mix_new_2_full.csv')
#df = df_bld_building_mix[['Country', 'Years', 'bld_building_mix_new_SFH[%]', 'bld_building_mix_new_MFH[%]']].copy()
#df.rename(columns={
#    f"bld_building_mix_new_{k}[%]": f"bld_building_mix_new_{v}_B[%]"
#    for k, v in bld_type_mapping.items()
#}, inplace=True)
#df["Years"] = pd.to_numeric(df["Years"], errors="coerce").astype("Int64")

#df = df.dropna(subset=['Country'])
#df = df.replace(",", ".", regex=True)
# In EP2050
## 2024 :
# - sfh : 0,213 / mfh: 0,787
# 2050:
# - sfh : 0,185 / mfh: 0,815
# Increase of ~ 0.03
dm_bld_building_mix_new = DataMatrix.create_from_df(df_bld_building_mix, num_cat=2)
dm_bld_building_mix_new.filter({'Years': years_fts}, inplace=True)
dm_bld_building_mix_new_adj = copy.deepcopy(DM_buildings['fts']['building-renovation-rate']['bld_building-mix'][1])
dm_bld_building_mix_new_adj[0, 1:-1, 'bld_building-mix_new', :, 'B'] = np.nan
dm_bld_building_mix_new_adj[0, -1, 'bld_building-mix_new', 'multi-family-households', 'B'] = \
    dm_bld_building_mix_new_adj[0, 0, 'bld_building-mix_new', 'multi-family-households', 'B'] + 0.01
dm_bld_building_mix_new_adj[0, -1, 'bld_building-mix_new', 'single-family-households', 'B'] = \
    dm_bld_building_mix_new_adj[0, 0, 'bld_building-mix_new', 'single-family-households', 'B'] - 0.01
dm_bld_building_mix_new_adj.fill_nans(dim_to_interp='Years')
DM_buildings['fts']['building-renovation-rate']['bld_building-mix'][3] = dm_bld_building_mix_new_adj



# --------------------------------------------------------------------------------

### 3.2 bld_renovation_rate ###
# State : DONE

df_bld_renovation_rate = pd.read_csv(csv_path+'bld_renovation-rate.csv')

dm_bld_renovation_rate = DataMatrix.create_from_df(df_bld_renovation_rate, num_cat=1)
dm_bld_renovation_rate.filter({'Years': years_fts}, inplace=True)

DM_buildings['fts']['building-renovation-rate']['bld_renovation-rate'][3] = dm_bld_renovation_rate.filter({'Variables': ['bld_renovation-rate']})

#DLS scenario
dm_ots = DM_buildings['ots']['floor-intensity'].filter({'Variables': ['lfs_floor-intensity_space-cap']})

dm_ots.append(dm_floor_intensity, dim='Years')
idx = dm_ots.idx
dm_ots.array[0,idx[2025]:idx[2050],...]=np.nan
dm_ots.fill_nans('Years')
dm_ots.filter({'Years': years_fts}, inplace=True)
dm_ots.append(DM_buildings['fts']['floor-intensity'][3].filter({'Variables': ['lfs_household-size']}), dim='Variables')

dm_renovation = DM_buildings['ots']['building-renovation-rate']['bld_renovation-rate'].copy()
df_renovation_dls = dm_bld_renovation_rate.write_df()
#df_renovation_dls['bld_renovation-rate_multi-family-households[%]'] = 0.03 # value from EP2050+ technical report (Kemmler et. al 2021) ; value from Nick2024 : 1.44%
#df_renovation_dls['bld_renovation-rate_single-family-households[%]'] = 0.03 #value from Nick2024 : 1.44%

df_renovation_dls['bld_renovation-rate_single-family-households[%]'] = df_renovation_dls.apply(lambda row: 0.038 if row['Years'] == 2030 else (0.05 if row['Years'] == 2050 else np.nan), axis=1)
df_renovation_dls['bld_renovation-rate_multi-family-households[%]'] = df_renovation_dls.apply(lambda row: 0.038 if row['Years'] == 2030 else (0.05 if row['Years'] == 2050 else np.nan), axis=1)

dm_renovation_dls = DataMatrix.create_from_df(df_renovation_dls, num_cat=1)
dm_renovation.append(dm_renovation_dls, dim='Years')
idx = dm_renovation.idx
#dm_renovation.array[0,idx[2025]:idx[2050],...]=np.nan
dm_renovation.array[0, idx[2025]:idx[2050], ...] = np.nan
dm_renovation.array[0, idx[2030], ...] = 0.038
dm_renovation.fill_nans('Years')
dm_renovation.filter({'Years': years_fts}, inplace=True)
DM_buildings['fts']['building-renovation-rate']['bld_renovation-rate'][4]=dm_renovation
# --------------------------------------------------------------------------------

### 3.3 bld_renovation_redistribution ###
# State : in progress

# --------------------------------------------------------------------------------

### 3.4 bld_demolition_rate ###
# State : in progress
df_dem_rate = pd.read_csv(csv_path+'bld_demolition-rate.csv')

dm_dem_rate = DataMatrix.create_from_df(df_dem_rate, num_cat=1)
#dm_dem_rate.filter({'Years': years_fts}, inplace=True)
#dm_dem_rate.rename_col('mfh', 'multi-family-households', dim='Categories1')
#dm_dem_rate.rename_col('sfh', 'single-family-households', dim='Categories1')
#dm_dem_rate.rename_col('demolition_rate', 'bld_demolition-rate', dim='Variables')

# Demolition rate set to the minimal value of 0.2%
dm_dem_rate = DM_buildings['fts']['building-renovation-rate']['bld_demolition-rate'][3]
dm_dem_rate['Switzerland', :, 'bld_demolition-rate', :] = 0.002
dm_dem_rate_dls = dm_dem_rate.copy()
dm_dem_rate_dls['Switzerland', :, 'bld_demolition-rate', :] = 0.00 #old value = 0
DM_buildings['fts']['building-renovation-rate']['bld_demolition-rate'][3] = dm_dem_rate
DM_buildings['fts']['building-renovation-rate']['bld_demolition-rate'][4] = dm_dem_rate_dls


# --------------------------------------------------------------------------------
# 4 - HEATING TECHNOLOGY FUEL
# --------------------------------------------------------------------------------
# Heating mix of fuel by envelope category and building type for stock - Historical data
dm_heatmix_ots = DM_buildings['ots']['heating-technology-fuel']['bld_heating-technology'].copy()
# Heating mix of fuel by building type for stock and new - Future data
data_file = 'data/EP2050/heating_mix.pickle'
with open(data_file, 'rb') as handle:
    dm_heattech_fts = pickle.load(handle)

map_cat = {'district-heating': 'dh', 'electricity': 'elec', 'gas': 'gas',
           'heat-pump': 'hp', 'heating-oil': 'oil', 'other': 'other', 'wood':  'wood'}
for cat_new, cat_old in map_cat.items():
    dm_heattech_fts.rename_col(cat_old, cat_new, dim='Categories2')
dm_heattech_fts.add(0, dim='Categories2', dummy=True, col_label=['solar', 'coal'])
dm_heattech_fts.sort('Categories2')
dm_heattech_fts.sort('Categories1')
years_fts_full = create_years_list(2025, 2050, 1)
dm_heattech_fts.filter({'Years': years_fts_full}, inplace=True)


# New area by building type in m2
# Tot stock by building type in m2
#dm_new_fts
# dm_floor_area.sort('Categories1')
dm_heatmix = dm_heatmix_ots.copy()
dm_heatmix.add(np.nan, dummy=True, dim='Years', col_label=years_fts_full)
cntr = 'Switzerland'
# Apply the yearly gr by fuel type to all categories to determine the bld_heating-mix, then normalise
dm_heattech_fts.lag_variable('bld_heating-technology_stock', shift=1, subfix='_tm1')
dm_heattech_fts.operation('bld_heating-technology_stock', '-', 'bld_heating-technology_stock_tm1', out_col='delta_stock', unit='%')
dm_heattech_fts.operation('delta_stock', '/', 'bld_heating-technology_stock_tm1', out_col='growth_rate', unit='%')
dm_heattech_fts.rename_col('other', 'other-tech', dim='Categories2')
tm1 = 2023
for t in years_fts_full:
    for cat in dm_heatmix.col_labels['Categories2']:
        for fuel in dm_heatmix.col_labels['Categories3']:
            if not (cat == 'F' and fuel == 'heat-pump'): # no new hp in F buildings -> more DH
                growthrate_t = dm_heattech_fts[cntr, t, 'growth_rate', :, fuel]
                dm_heatmix[cntr, t, 'bld_heating-mix', :, cat, fuel] = \
                    np.maximum(dm_heatmix[cntr, tm1, 'bld_heating-mix', :, cat, fuel] * (1 + growthrate_t), 0)
    tm1 = t

dm_heatmix.normalise('Categories3')

dm_heatmix.fill_nans('Years')

#mask = np.isnan(dm_heatmix.array)
#dm_heatmix.array[mask] = 0
years_fts = create_years_list(2025, 2050, 5)
years_ots = create_years_list(1990, 2023, 1)
dm_heatmix = dm_heatmix.filter({'Years': years_ots + years_fts})
dm_heatmix[:, 2025, ...] = np.nan
dm_heatmix.fill_nans('Years')
dm = dm_heatmix.filter({'Years': years_fts})
df_heat_mix = dm.write_df()
DM_buildings['fts']['heating-technology-fuel']['bld_heating-technology'][3] = dm.copy()

#DLS scenario
DM_buildings['fts']['heating-technology-fuel']['bld_heating-technology'][4] = DM_buildings['fts']['heating-technology-fuel']['bld_heating-technology'][3].copy()

df_heat_dls = DM_buildings['fts']['heating-technology-fuel']['bld_heating-technology'][4].write_df()


# --------------------------------------------------------------------------------
# 5 - HEATING EFFICIENCY
# --------------------------------------------------------------------------------

#dm_heat_mix_new_ots = DM_buildings['ots']['building-renovation-rate']['bld_building-mix']
#df_heat_mix_new_ots = dm_heat_mix_new_ots.write_df()
#df_heat_mix_new_ots.to_csv('data/heating-mix_new_ots.csv', index=False)

df_heat_efficiency = pd.read_csv(csv_path+'efficiency_result.csv')
dm_heat_eff_fts = DataMatrix.create_from_df(df_heat_efficiency, num_cat = 2)
dm_heat_eff_fts.rename_col('other', 'other-tech', dim='Categories2')
dm_ots = DM_buildings['ots']['heating-efficiency'].filter({'Variables': ['bld_heating-efficiency']}).copy()
df_ots = dm_ots.write_df()

dm_ots.append(dm_heat_eff_fts, dim='Years')
idx = dm_ots.idx
dm_ots.array[0,idx[2025]:idx[2050],...]=np.nan
dm_ots.fill_nans('Years')

# !FIXME : write to csv and recompute efficiencies!
dm_orig = DM_buildings['fts']['heating-efficiency'][3].copy()
dm_orig.drop(col_label=['bld_heating-efficiency'], dim='Variables')
dm_ots.filter({'Years': years_fts}, inplace=True)
dm_orig.append(dm_ots, dim='Variables')

df_fuel_eff = dm_orig.write_df()
DM_buildings['fts']['heating-efficiency'][3] = dm_orig.copy()
DM_buildings['fts']['heating-efficiency'][4] = DM_buildings['fts']['heating-efficiency'][3]



# --------------------------------------------------------------------------------
# RENOVATION-REDISTRIBUTION
# --------------------------------------------------------------------------------
df_area = pd.read_csv(csv_path + 'bld_floor-area_stock.csv')
dm_stock_cat = DataMatrix.create_from_df(df_area, num_cat=2)
dm_stock_cat.filter({'Years': years_fts}, inplace=True)
# Units are actually Mm2
dm_stock_cat.change_unit('bld_floor_area_stock', 1, 'm2', 'Mm2')

arr_waste = dm_stock_cat.array*0.002
dm_stock_cat.add(arr_waste, dim='Variables', col_label='bld_floor-area_waste', unit='Mm2')
dm_stock_cat.lag_variable('bld_floor_area_stock', shift=1, subfix='_tm1')

# s(t) = s(t-1) + n(t) - w(t)
dm_stock_all = dm_stock_cat.group_all('Categories2', inplace=False)
arr_new = dm_stock_all[0, :, 'bld_floor_area_stock', :] - dm_stock_all[0, :, 'bld_floor_area_stock_tm1', :] \
          + dm_stock_all[0, :, 'bld_floor-area_waste', :]
dm_stock_all.add(arr_new[np.newaxis, ...], dim='Variables', unit='Mm2', col_label='bld_floor-area_new')

dm_stock_cat.add(0, dim='Variables', col_label='bld_floor-area_new', unit='Mm2', dummy=True)
dm_stock_cat[0, :, 'bld_floor-area_new', :, 'B'] = dm_stock_all[0, :, 'bld_floor-area_new', :]
arr_ren = dm_stock_cat[0, :, 'bld_floor_area_stock', ...] - dm_stock_cat[0, :, 'bld_floor_area_stock_tm1', ...] - \
          dm_stock_cat[0, :, 'bld_floor-area_new', ...] + dm_stock_cat[0, :, 'bld_floor-area_waste', ...]
dm_stock_cat.add(arr_ren[np.newaxis, ...], dim='Variables', col_label='bld_floor-area_renovated', unit='m2')

dm_renov_redistr = dm_stock_cat.filter({'Variables': ['bld_floor-area_renovated']})
dm_renov_redistr.group_all('Categories1')
dm_renov_redistr[0, 0, ...] = np.nan
dm_renov_redistr.fill_nans(dim_to_interp='Years')
mask = dm_renov_redistr.array > 0
dm_renov_redistr.array[mask] = 0
dm_renov_redistr.array = - dm_renov_redistr.array
dm_renov_redistr.normalise('Categories1', inplace=True)
dm_renov_redistr.rename_col('bld_floor-area_renovated', 'bld_floor-area_ren-normalised', dim='Variables')

dm_renov_redistr.add(0, col_label='bld_renovation-redistribution-in', dim='Variables', unit='%', dummy=True)
dm_renov_redistr.add(0, col_label='bld_renovation-redistribution-out', dim='Variables', unit='%', dummy=True)
#
dm_renov_redistr[0, :, 'bld_renovation-redistribution-in', 'B'] = 1
dm_renov_redistr[0, :, 'bld_renovation-redistribution-out', :] \
    = dm_renov_redistr[0, :, 'bld_floor-area_ren-normalised', :]

dm_renov_redistr.drop(col_label='bld_floor-area_ren-normalised', dim='Variables')
#finalement on ne touche pas à ce levier, on fait l'assumption que les renovations redistrob sont les memes (manque d'info)
DM_buildings['fts']['building-renovation-rate']['bld_renovation-redistribution'][3] = copy.deepcopy(DM_buildings['fts']['building-renovation-rate']['bld_renovation-redistribution'][1])
DM_buildings['fts']['building-renovation-rate']['bld_renovation-redistribution'][4] = DM_buildings['fts']['building-renovation-rate']['bld_renovation-redistribution'][3]



my_pickle_dump(DM_buildings, '../../../data/datamatrix/buildings.pickle')

sort_pickle('../../../data/datamatrix/buildings.pickle')
