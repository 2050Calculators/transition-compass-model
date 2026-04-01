
# packages
import pickle
import os
import numpy as np
import pandas as pd
import warnings
import eurostat
warnings.simplefilter("ignore")
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat
from _database.pre_processing.routine_JRC import get_jrc_data, get_jrc_balance_data_for_transport, \
    get_names_map_for_transport, get_mapping_fuels_for_transport
from transition_compass_model.model.common.auxiliary_functions import eurostat_iso2_dict, jrc_iso2_dict, linear_fitting
from transition_compass_model.model.common.data_matrix_class import DataMatrix

#######################################################
######################### JRC #########################
#######################################################

# directories
current_file_directory = os.getcwd()

# get iso codes
dict_iso2 = eurostat_iso2_dict()
dict_iso2.pop('CH')  # Remove Switzerland
dict_iso2_jrc = jrc_iso2_dict()

# set function inputs
database = "JRC-IDEES-2021_EnergyBalance"
variable = "energy-demand"
unit = "ktoe"
names_map = get_names_map_for_transport()
jrc_to_model_fuels = get_mapping_fuels_for_transport()

DM_temp = {}
DM_temp_country = {}

for country_code in dict_iso2_jrc.keys():
    country_name = dict_iso2_jrc[country_code]
    print(f"Doing {country_name} ... ")
    
    for sheet_name in names_map.keys():
        DM_temp[names_map[sheet_name]] = get_jrc_balance_data_for_transport(
            current_file_directory, country_code, country_name, unit, variable, database, 
            sheet_name, names_map, jrc_to_model_fuels)
        
    # put data together
    dm_temp = DM_temp[list(DM_temp)[0]].copy()
    labels = list(DM_temp)[1:]
    for l in labels: dm_temp.append(DM_temp[l], "Variables")
    
    # aggregate
    dm_temp.groupby({"aviation-passenger" : ['aviation-passenger-domestic', 'aviation-passenger-extra', 'aviation-passenger-intra'], 
                     "aviation-freight" : ['aviation-freight-domestic', 'aviation-freight-extra', 'aviation-freight-intra'],
                     "marine" : ['marine-domestic', 'marine-international'],
                     'rail-passenger': ['rail-passenger-normal', 'rail-passenger-high'],
                     'trucks' : ['trucks-lm', 'trucks-h']}, 
                    "Variables", inplace=True)
    
    # save
    DM_temp_country[country_name] = dm_temp.copy()

# append
dm_temp = DM_temp_country["Austria"].copy()
labels = list(DM_temp_country)[1:]
for l in labels: dm_temp.append(DM_temp_country[l], "Country")

# change unit
for v in dm_temp.col_labels["Variables"]:
    dm_temp.change_unit(v, 0.01163, "ktoe", "TWh")

# deepen
for v in dm_temp.col_labels["Variables"]:
    dm_temp.rename_col(v, "calib-energy-demand_" + v, "Variables")
dm_temp.deepen(based_on="Variables")
dm_temp.switch_categories_order("Categories1","Categories2")

# add missing years as nan
years = list(range(1990,2023+1)) + list(range(2025,2050+5,5))
missing = np.array(years)[[y not in dm_temp.col_labels["Years"] for y in years]].tolist()
dm_temp.add(np.nan, "Years", missing, dummy=True)
dm_temp.sort("Years")

# make final DF
DM_final = {}
dm_pas = dm_temp.filter({"Categories1" : ['2W', 'LDV', 
                                          'aviation-passenger', 
                                          'bus', 'metrotram', 'rail-passenger']})
dm_pas.rename_col(['aviation-passenger','rail-passenger'], ['aviation','rail'], "Categories1")
dm_pas.sort("Categories1")
DM_final["passenger_jrc"] = dm_pas.copy()

dm_fre = dm_temp.filter({"Categories1" : ['IWW', 
                                          'aviation-freight', 'marine',
                                          'rail-freight', 'trucks']})
dm_fre.rename_col(['aviation-freight','rail-freight'], ['aviation','rail'], "Categories1")
dm_fre.sort("Categories1")
DM_final["freight_jrc"] = dm_fre.copy()


#######################################################
####################### EUROSTAT ######################
#######################################################

names_map = {
    "INTMARB" : "marine-international",
    "INTAVI" : "aviation-international",
    "FC_TRA_RAIL_E" : "rail", # this has passenger and freight (including metrotram)
    "FC_TRA_ROAD_E" : "road", # this has passenger and freight
    "FC_TRA_DAVI_E" : "aviation-domestic", # this has passenger and freight
    "FC_TRA_DNAVI_E" : "marine-domestic", # this also has IWW
    }


# get data
code = "nrg_bal_c"
eurostat.get_pars(code)
filter = {'geo\\TIME_PERIOD': list(dict_iso2.keys()),
          'nrg_bal': list(names_map.keys()),
          'siec' : [code for codes in jrc_to_model_fuels.values() for code in codes],
          'unit' : ['KTOE']}
mapping_dim = {'Country': 'geo\\TIME_PERIOD',
                'Variables': 'nrg_bal',
                'Categories1' : 'siec'}
dm_temp = get_data_api_eurostat(code, filter, mapping_dim, "ktoe")
# dm_save = dm_temp.copy()
# dm_temp = dm_save.copy()

# rename
for key in names_map.keys():
    dm_temp.rename_col(key, names_map[key], "Variables")

# aggregate
jrc_to_model_fuels_temp = jrc_to_model_fuels.copy()
categs = dm_temp.col_labels["Categories1"]
for key in jrc_to_model_fuels_temp.keys():
    list_temp = jrc_to_model_fuels_temp[key].copy()
    list_temp = np.array(list_temp)[[l in categs for l in list_temp]].tolist()
    jrc_to_model_fuels_temp[key] = list_temp
jrc_to_model_fuels_temp["other"] = jrc_to_model_fuels_temp["other"] + ['O4100TOT', 'R5110-5150W6000RI', 'W61006220']

dm_temp.groupby(jrc_to_model_fuels_temp, "Categories1", inplace=True)
dm_temp.add(0, "Categories1", ["efuel","hydrogen"], "ktoe", dummy=True)
dm_temp.sort("Categories1")

# change unit
for v in dm_temp.col_labels["Variables"]:
    dm_temp.change_unit(v, 0.01163, "ktoe", "TWh")

# deepen
for v in dm_temp.col_labels["Variables"]:
    dm_temp.rename_col(v, "calib-energy-demand_" + v, "Variables")
dm_temp.deepen(based_on="Variables")
dm_temp.switch_categories_order("Categories1","Categories2")

# aggregate
dm_temp.groupby({"aviation" : ['aviation-domestic', 'aviation-international'],
                 "marine" : ['marine-domestic', 'marine-international']}, 
                "Categories1", inplace=True)
dm_temp.sort("Categories1")

# add missing years as nan
years = list(range(1990,2023+1)) + list(range(2025,2050+5,5))
missing = np.array(years)[[y not in dm_temp.col_labels["Years"] for y in years]].tolist()
dm_temp.add(np.nan, "Years", missing, dummy=True)
dm_temp.sort("Years")
dm_temp.drop("Years", 2024)

# drop uk
dm_temp.drop("Country", "United Kingdom")

# put in final
DM_final["all_eurostat"] = dm_temp.copy()

# # check
# dm_check = DM_final["passenger_jrc"].filter({"Country" : ["EU27"], "Categories1" : ["LDV"]})
# dm_check.group_all("Categories2")
# dm_check.change_unit("calib-energy-demand", 1/0.01163, "TWh", "ktoe")
# df_check = dm_check.write_df()
# dm_check = DM_final["passenger_jrc"].filter({"Country" : ["EU27"]})
# df_check = dm_check.write_df()

#######################################################
######################### SAVE ########################
#######################################################

# save
f = os.path.join(current_file_directory, '../data/datamatrix/calibration_energy-demand.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM_final, handle, protocol=pickle.HIGHEST_PROTOCOL)





