
# packages
from model.common.data_matrix_class import DataMatrix
from model.common.auxiliary_functions import linear_fitting
import pandas as pd
import pickle
import os
import numpy as np
import warnings

# from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat
warnings.simplefilter("ignore")
import plotly.express as px
import plotly.io as pio
import re
pio.renderers.default='browser'

from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat
from _database.pre_processing.routine_JRC import get_jrc_data
from model.common.auxiliary_functions import eurostat_iso2_dict, jrc_iso2_dict

# file
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/transport/EU/python/transport_fxa_passenger_new-vehicles.py"

# directories
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# load current transport pickle
filepath = os.path.join(current_file_directory, '../../../../data/datamatrix/transport.pickle')
with open(filepath, 'rb') as handle:
    DM_tra = pickle.load(handle)

# Set years range
years_setting = [1990, 2023, 2050, 5]
startyear = years_setting[0]
baseyear = years_setting[1]
lastyear = years_setting[2]
step_fts = years_setting[3]
years_ots = list(range(startyear, baseyear+1, 1))
years_fts = list(range(baseyear+2, lastyear+1, step_fts))
years_all = years_ots + years_fts

################################################################################
############################ TECHNOLOGY SHARE new ############################
################################################################################

################################################
################### GET DATA ###################
################################################

DM_tra["fxa"]["passenger_tech"].units
categories2_all = DM_tra["fxa"]["passenger_tech"].col_labels["Categories2"]

# get iso codes
dict_iso2 = eurostat_iso2_dict()
dict_iso2.pop('CH')  # Remove Switzerland
dict_iso2_jrc = jrc_iso2_dict()

##############
##### 2W #####
##############

# # get data
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrRoad_act",
#                 "variable" : "New vehicle-registrations",
#                 "sheet_last_row" : "Powered two-wheelers",
#                 "sub_variables" : ["Powered two-wheelers"],
#                 "calc_names" : ["2W"]}
# dm_new_2w = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/new_2w.pickle')
# with open(f, 'wb') as handle: pickle.dump(dm_new_2w, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/new_2w.pickle')
with open(f, 'rb') as handle: dm_new_2w = pickle.load(handle)

# make other variables
# assumption: 2w are mostly ICE-gasoline, so the energy efficiency we have will be assigned
# to 2W_ICE-gasoline, and the rest will be missing values for now
dm_new_2w.rename_col("2W","2W_ICE-gasoline","Variables")
dm_new_2w.deepen()
categories2_missing = categories2_all.copy()
for cat in dm_new_2w.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_new_2w.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_new_2w.sort("Categories1")


###############
##### LDV #####
###############

# # get data
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrRoad_act",
#                 "variable" : "New vehicle-registrations",
#                 "sheet_last_row" : "Battery electric vehicles",
#                 "categories" : "Passenger cars",
#                 "sub_variables" : ["Gasoline engine",
#                                     "Diesel oil engine",
#                                     "LPG engine", "Natural gas engine", 
#                                     "Plug-in hybrid electric",
#                                     "Battery electric vehicles"],
#                 "calc_names" : ["LDV_ICE-gasoline","LDV_ICE-diesel","LDV_gas-lpg",
#                                 "LDV_gas-natural","LDV_PHEV","LDV_BEV"]}
# dm_new_ldv = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/new_ldv.pickle')
# with open(f, 'wb') as handle: pickle.dump(dm_new_ldv, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/new_ldv.pickle')
with open(f, 'rb') as handle: dm_new_ldv = pickle.load(handle)

# deepen and sum gas
dm_new_ldv.deepen()
mapping_calc = {'ICE-gas': ['gas-lpg', 'gas-natural']}
dm_new_ldv.groupby(mapping_calc, dim='Categories1', aggregation = "sum", regex=False, inplace=True)

# make PHEV diesel and PHEV gasoline
# assumption: they are the same proportion of ldv diesel and ldv gasoline
dm_temp = dm_new_ldv.filter({"Categories1" : ["PHEV"]})
dm_temp.rename_col("PHEV","PHEV-gasoline","Categories1")
dm_temp.add(dm_temp.array, col_label="PHEV-diesel", dim="Categories1")
dm_temp.sort("Categories1")
dm_temp1 = dm_new_ldv.filter({"Categories1" : ['ICE-diesel', 'ICE-gasoline']})
dm_temp1.normalise("Categories1")
dm_temp.array = dm_temp.array * dm_temp1.array
dm_new_ldv.drop("Categories1","PHEV")
dm_new_ldv.append(dm_temp,"Categories1")
dm_new_ldv.sort("Categories1")

# make other variables
# assumption: for now, rest is assumed to be missing
categories2_missing = categories2_all.copy()
for cat in dm_new_ldv.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_new_ldv.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_new_ldv.sort("Categories1")

#################
##### BUSES #####
#################

# # get data
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrRoad_act",
#                 "variable" : "New vehicle-registrations",
#                 "categories" : "Motor coaches, buses and trolley buses",
#                 "sheet_last_row" : "Battery electric vehicles",
#                 "sub_variables" : ["Gasoline engine",
#                                     "Diesel oil engine",
#                                     "LPG engine", "Natural gas engine", 
#                                     "Battery electric vehicles"],
#                 "calc_names" : ["bus_ICE-gasoline","bus_ICE-diesel","bus_gas-lpg",
#                                 "bus_gas-natural","bus_BEV"]}
# dm_new_bus = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/new_bus.pickle')
# with open(f, 'wb') as handle: pickle.dump(dm_new_bus, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/new_bus.pickle')
with open(f, 'rb') as handle: dm_new_bus = pickle.load(handle)

# deepen and sum gas
dm_new_bus.deepen()
mapping_calc = {'ICE-gas': ['gas-lpg', 'gas-natural']}
dm_new_bus.groupby(mapping_calc, dim='Categories1', aggregation = "mean", regex=False, inplace=True)

# make rest of the variables (assuming they are all missing for now)
categories2_missing = categories2_all.copy()
for cat in dm_new_bus.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_new_bus.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_new_bus.sort("Categories1")

################
##### RAIL #####
################

# # get data
# # note: I assume that all high speed passenger trains are electric, and I'll take an average
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrRail_act",
#                 "variable" : "New vehicles - total (representative train configuration)",
#                 "sheet_last_row" : "High speed passenger trains",                
#                 "sub_variables" : ["Metro and tram, urban light rail",
#                                     "Diesel oil",
#                                     "Electric",
#                                     "High speed passenger trains"],
#                 "calc_names" : ["metrotram_mt","train-conv_ICE-diesel","train-conv_CEV","train-hs_CEV"]}
# dm_new_rail = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/new_rail.pickle')
# with open(f, 'wb') as handle: pickle.dump(dm_new_rail, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/new_rail.pickle')
with open(f, 'rb') as handle: dm_new_rail = pickle.load(handle)

# aggregate trains and deepen
mapping_calc = {'rail_CEV': ['train-conv_CEV', 'train-hs_CEV'],
                'rail_ICE-diesel' : ["train-conv_ICE-diesel"]}
dm_new_rail.groupby(mapping_calc, dim='Variables', aggregation = "mean", regex=False, inplace=True)
dm_new_rail.deepen()

# make rest of the variables (assuming they are all missing for now)
categories2_missing = categories2_all.copy()
for cat in dm_new_rail.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_new_rail.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_new_rail.sort("Categories1")

# fix unit
dm_new_rail.units["metrotram"] = "vehicles"
dm_new_rail.units["rail"] = "vehicles"

########################
##### PUT TOGETHER #####
########################

dm_new = dm_new_2w.copy()
dm_new.append(dm_new_ldv,"Variables")
dm_new.append(dm_new_bus,"Variables")
dm_new.append(dm_new_rail,"Variables")
dm_new.sort("Variables")
dm_new.sort("Country")

# check
# dm_new.flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()

# there is a problem with new LDV ice gasoline before 2014. Seeing
# that also in Eurostat it's missing, I assume there is a problem with JRC data
# and I put it missing
dm_new = dm_new.flatten()
idx = dm_new.idx
years_fix = list(range(2000,2013+1))
for y in years_fix:
    dm_new.array[:,idx[y],idx["LDV_ICE-gasoline"]] = np.nan
dm_new.deepen()

# check
# dm_new.flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()


###################
##### FIX OTS #####
###################

# flatten
dm_new = dm_new.flatten()

# new variabs list
dict_new = {}

def make_ots(dm, variable, periods_dicts, years_ots = None):
    
    dm_temp = dm.filter({"Variables" : [variable]})
    if periods_dicts["n_adj"] == 1:
        dm_temp = linear_fitting(dm_temp, years_ots, min_t0=0.1,min_tb=0.1)
    if periods_dicts["n_adj"] == 2:
        dm_temp = linear_fitting(dm_temp, list(range(1990,1999+1)), 
                                 based_on=list(range(2000,periods_dicts["year_end_first_adj"]+1)), 
                                 min_t0=0.1,min_tb=0.1)
        dm_temp = linear_fitting(dm_temp, list(range(2022,2023+1)), 
                                 based_on=list(range(periods_dicts["year_start_second_adj"],2021+1)), 
                                 min_t0=0.1,min_tb=0.1)
    return dm_temp

dict_call = {"2W_ICE-gasoline" : {"n_adj" : 2, "year_end_first_adj" : 2007, "year_start_second_adj" : 2021},
             "LDV_BEV" : {"n_adj" : 2, "year_end_first_adj" : 2010, "year_start_second_adj" : 2021},
             "LDV_ICE-diesel" : {"n_adj" : 2, "year_end_first_adj" : 2006, "year_start_second_adj" : 2017},
             "LDV_ICE-gas" : {"n_adj" : 1},
             "LDV_ICE-gasoline" : {"n_adj" : 1},
             "LDV_PHEV-diesel" : {"n_adj" : 2, "year_end_first_adj" : 2010, "year_start_second_adj" : 2021},
             "LDV_PHEV-gasoline" : {"n_adj" : 2, "year_end_first_adj" : 2010, "year_start_second_adj" : 2021},
             "bus_BEV" : {"n_adj" : 2, "year_end_first_adj" : 2010, "year_start_second_adj" : 2021},
             "bus_ICE-diesel" : {"n_adj" : 1},
             "bus_ICE-gas" : {"n_adj" : 2, "year_end_first_adj" : 2010, "year_start_second_adj" : 2020},
             "bus_ICE-gasoline" : {"n_adj" : 2, "year_end_first_adj" : 2010, "year_start_second_adj" : 2020},
             "metrotram_mt" :  {"n_adj" : 1},
             "rail_CEV" :  {"n_adj" : 1},
             "rail_ICE-diesel" :  {"n_adj" : 1}}

for key in dict_call.keys():
    if len(dict_call[key]) > 1:
        dict_new[key] = make_ots(dm_new, key, dict_call[key])
    else:
        dict_new[key] = make_ots(dm_new, key, dict_call[key], years_ots)

# append
dm_new = dict_new["2W_ICE-gasoline"].copy()
mylist = list(dict_call.keys())
mylist.remove("2W_ICE-gasoline")
for v in mylist:
    dm_new.append(dict_new[v],"Variables")
dm_new.sort("Variables")

# check
# dm_new.filter({"Country" : ["EU27"]}).datamatrix_plot()


####################
##### MAKE FTS #####
####################

# make function to fill in missing years fts for EU27 with linear fitting
def make_fts(dm, variable, year_start, year_end, country = "EU27", dim = "Categories1", 
             min_t0=0.1, min_tb=0.1, years_fts = years_fts): # I put minimum to 1 so it does not go to zero
    dm = dm.copy()
    idx = dm.idx
    based_on_yars = list(range(year_start, year_end + 1, 1))
    dm_temp = linear_fitting(dm.filter({"Country" : [country], dim : [variable]}), 
                             years_ots = years_fts, min_t0=min_t0, min_tb=min_tb, based_on = based_on_yars)
    idx_temp = dm_temp.idx
    if dim == "Variables":
        dm.array[idx[country],:,idx[variable],...] = \
            dm_temp.array[idx_temp[country],:,idx_temp[variable],...]
    if dim == "Categories1":
        dm.array[idx[country],:,:,idx[variable]] = \
            dm_temp.array[idx_temp[country],:,:,idx_temp[variable]]
    if dim == "Categories2":
        dm.array[idx[country],:,:,:,idx[variable]] = \
            dm_temp.array[idx_temp[country],:,:,:,idx_temp[variable]]
    if dim == "Categories3":
        dm.array[idx[country],:,:,:,:,idx[variable]] = \
            dm_temp.array[idx_temp[country],:,:,:,:,idx_temp[variable]]
    
    return dm

# # make a total
# dm_total = dm_new.groupby({"total" : dm_new.col_labels["Variables"]}, dim='Variables', 
#                             aggregation = "sum", regex=False, inplace=False)
# dm_new.append(dm_total,"Variables")

# get fleet
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/passenger_fleet.pickle')
with open(f, 'rb') as handle: dm_fleet = pickle.load(handle)
dm_fleet = dm_fleet.flatten().flatten()
dm_fleet.rename_col_regex("tra_passenger_technology-share_fleet_","","Variables")

# check
# dm_fleet.filter({"Country" : ["EU27"]}).datamatrix_plot()

# add missing years fts
dm_new.add(np.nan, col_label=years_fts, dummy=True, dim='Years')

# drop 2022-2023 for the electric vehicles (to avoid a flat line)
idx = dm_new.idx
for v in ["LDV_BEV","LDV_PHEV-diesel","LDV_PHEV-gasoline","bus_BEV"]:
    dm_new.array[idx["EU27"],idx[2022],idx[v]] = np.nan
    dm_new.array[idx["EU27"],idx[2023],idx[v]] = np.nan

# set default time window for linear trend
baseyear_start = 2000
baseyear_end = 2023

# 2W
dm_new = make_fts(dm_new, "2W_ICE-gasoline", 2012, baseyear_end, dim = "Variables")

# for electric vehicles, get % change of fleet over 2023-2050 and apply it to new vehicles

# bev
idx = dm_fleet.idx
product = "LDV_BEV"
rate_increase = (dm_fleet.array[idx["EU27"],idx[2050],idx[product]] - 
                 dm_fleet.array[idx["EU27"],idx[2021],idx[product]])/\
    dm_fleet.array[idx["EU27"],idx[2021],idx[product]]
idx = dm_new.idx
value_2050 = round(dm_new.array[idx["EU27"],idx[2021],idx[product]] * rate_increase,0)
dm_new.array[idx["EU27"],idx[2050],idx[product]] = value_2050
dm_temp = linear_fitting(dm_new.filter({"Country" : ["EU27"], "Variables" : [product]}), years_ots + years_fts)
idx_temp = dm_temp.idx
dm_new.array[idx["EU27"],:,idx[product]] = dm_temp.array[idx_temp["EU27"],:,idx_temp[product]]

# ice
dm_new = make_fts(dm_new, "LDV_ICE-diesel", 2017, baseyear_end, dim = "Variables")
dm_new = make_fts(dm_new, "LDV_ICE-gas", baseyear_start, baseyear_end, dim = "Variables")
dm_new = make_fts(dm_new, "LDV_ICE-gasoline", baseyear_start, baseyear_end, dim = "Variables")

# LDV_PHEV-diesel
idx = dm_fleet.idx
product = "LDV_PHEV-diesel"
rate_increase = (dm_fleet.array[idx["EU27"],idx[2050],idx[product]] - 
                 dm_fleet.array[idx["EU27"],idx[2021],idx[product]])/\
    dm_fleet.array[idx["EU27"],idx[2021],idx[product]]
idx = dm_new.idx
value_2050 = round(dm_new.array[idx["EU27"],idx[2021],idx[product]] * rate_increase,0)
dm_new.array[idx["EU27"],idx[2050],idx[product]] = value_2050
dm_temp = linear_fitting(dm_new.filter({"Country" : ["EU27"], "Variables" : [product]}), years_ots + years_fts)
idx_temp = dm_temp.idx
dm_new.array[idx["EU27"],:,idx[product]] = dm_temp.array[idx_temp["EU27"],:,idx_temp[product]]

# LDV_PHEV-gasoline
idx = dm_fleet.idx
product = "LDV_PHEV-gasoline"
rate_increase = (dm_fleet.array[idx["EU27"],idx[2050],idx[product]] - 
                 dm_fleet.array[idx["EU27"],idx[2021],idx[product]])/\
    dm_fleet.array[idx["EU27"],idx[2021],idx[product]]
idx = dm_new.idx
value_2050 = round(dm_new.array[idx["EU27"],idx[2021],idx[product]] * rate_increase,0)
dm_new.array[idx["EU27"],idx[2050],idx[product]] = value_2050
dm_temp = linear_fitting(dm_new.filter({"Country" : ["EU27"], "Variables" : [product]}), years_ots + years_fts)
idx_temp = dm_temp.idx
dm_new.array[idx["EU27"],:,idx[product]] = dm_temp.array[idx_temp["EU27"],:,idx_temp[product]]

# bus_BEV
idx = dm_fleet.idx
product = "bus_BEV"
rate_increase = (dm_fleet.array[idx["EU27"],idx[2050],idx[product]] - 
                 dm_fleet.array[idx["EU27"],idx[2021],idx[product]])/\
    dm_fleet.array[idx["EU27"],idx[2021],idx[product]]
idx = dm_new.idx
value_2050 = round(dm_new.array[idx["EU27"],idx[2021],idx[product]] * rate_increase,0)
dm_new.array[idx["EU27"],idx[2050],idx[product]] = value_2050
dm_temp = linear_fitting(dm_new.filter({"Country" : ["EU27"], "Variables" : [product]}), years_ots + years_fts)
idx_temp = dm_temp.idx
dm_new.array[idx["EU27"],:,idx[product]] = dm_temp.array[idx_temp["EU27"],:,idx_temp[product]]

# rest
dm_new = make_fts(dm_new, "bus_ICE-diesel", baseyear_start, baseyear_end, dim = "Variables")
dm_new = make_fts(dm_new, "bus_ICE-gas", baseyear_start, baseyear_end, dim = "Variables")
dm_new = make_fts(dm_new, "bus_ICE-gasoline", baseyear_start, baseyear_end, dim = "Variables")
dm_new = make_fts(dm_new, "metrotram_mt", baseyear_start, baseyear_end, dim = "Variables")
dm_new = make_fts(dm_new, "rail_CEV", baseyear_start, baseyear_end, dim = "Variables")
dm_new = make_fts(dm_new, "rail_ICE-diesel", baseyear_start, baseyear_end, dim = "Variables")

# check
# dm_new.filter({"Country" : ["EU27"]}).datamatrix_plot()

####################################
##### MAKE AS FINAL DATAMATRIX #####
####################################

DM_tra["fxa"]["passenger_tech"].units

# rename and deepen
for v in dm_new.col_labels["Variables"]:
    dm_new.rename_col(v,"tra_passenger_new-vehicles_" + v, "Variables")
dm_new.deepen_twice()

# check
# dm_new.flatten().flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()

################
##### SAVE #####
################

list(DM_tra["fxa"])
DM_tra["fxa"]["passenger_tech"]

# save
f = os.path.join(current_file_directory, '../data/datamatrix/fxa_passenger_new-vehicles.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_new, handle, protocol=pickle.HIGHEST_PROTOCOL)




