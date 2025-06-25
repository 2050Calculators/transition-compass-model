
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
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/transport/EU/python/transport_fxa_passenger_tech.py"

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
############################ TECHNOLOGY SHARE FLEET ############################
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
#                 "variable" : "Stock of vehicles - total (vehicles)",
#                 "sheet_last_row" : "Powered two-wheelers",
#                 "sub_variables" : ["Powered two-wheelers"],
#                 "calc_names" : ["2W"]}
# dm_fleet_2w = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/fleet_2w.pickle')
# with open(f, 'wb') as handle: pickle.dump(dm_fleet_2w, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/fleet_2w.pickle')
with open(f, 'rb') as handle: dm_fleet_2w = pickle.load(handle)

# make other variables
# assumption: 2w are mostly ICE-gasoline, so the energy efficiency we have will be assigned
# to 2W_ICE-gasoline, and the rest will be missing values for now
dm_fleet_2w.rename_col("2W","2W_ICE-gasoline","Variables")
dm_fleet_2w.deepen()
categories2_missing = categories2_all.copy()
for cat in dm_fleet_2w.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_fleet_2w.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_fleet_2w.sort("Categories1")


###############
##### LDV #####
###############

# # get data
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrRoad_act",
#                 "variable" : "Stock of vehicles - total (vehicles)",
#                 "sheet_last_row" : "Battery electric vehicles",
#                 "categories" : "Passenger cars",
#                 "sub_variables" : ["Gasoline engine",
#                                     "Diesel oil engine",
#                                     "LPG engine", "Natural gas engine", 
#                                     "Plug-in hybrid electric",
#                                     "Battery electric vehicles"],
#                 "calc_names" : ["LDV_ICE-gasoline","LDV_ICE-diesel","LDV_gas-lpg",
#                                 "LDV_gas-natural","LDV_PHEV","LDV_BEV"]}
# dm_fleet_ldv = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/fleet_ldv.pickle')
# with open(f, 'wb') as handle: pickle.dump(dm_fleet_ldv, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/fleet_ldv.pickle')
with open(f, 'rb') as handle: dm_fleet_ldv = pickle.load(handle)

# deepen and sum gas
dm_fleet_ldv.deepen()
mapping_calc = {'ICE-gas': ['gas-lpg', 'gas-natural']}
dm_fleet_ldv.groupby(mapping_calc, dim='Categories1', aggregation = "sum", regex=False, inplace=True)

# make PHEV diesel and PHEV gasoline
# assumption: they are the same proportion of ldv diesel and ldv gasoline
dm_temp = dm_fleet_ldv.filter({"Categories1" : ["PHEV"]})
dm_temp.rename_col("PHEV","PHEV-gasoline","Categories1")
dm_temp.add(dm_temp.array, col_label="PHEV-diesel", dim="Categories1")
dm_temp.sort("Categories1")
dm_temp1 = dm_fleet_ldv.filter({"Categories1" : ['ICE-diesel', 'ICE-gasoline']})
dm_temp1.normalise("Categories1")
dm_temp.array = dm_temp.array * dm_temp1.array
dm_fleet_ldv.drop("Categories1","PHEV")
dm_fleet_ldv.append(dm_temp,"Categories1")
dm_fleet_ldv.sort("Categories1")

# make other variables
# assumption: for now, rest is assumed to be missing
categories2_missing = categories2_all.copy()
for cat in dm_fleet_ldv.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_fleet_ldv.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_fleet_ldv.sort("Categories1")

#################
##### BUSES #####
#################

# # get data
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrRoad_act",
#                 "variable" : "Stock of vehicles - total (vehicles)",
#                 "categories" : "Motor coaches, buses and trolley buses",
#                 "sheet_last_row" : "Battery electric vehicles",
#                 "sub_variables" : ["Gasoline engine",
#                                     "Diesel oil engine",
#                                     "LPG engine", "Natural gas engine", 
#                                     "Battery electric vehicles"],
#                 "calc_names" : ["bus_ICE-gasoline","bus_ICE-diesel","bus_gas-lpg",
#                                 "bus_gas-natural","bus_BEV"]}
# dm_fleet_bus = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/fleet_bus.pickle')
# with open(f, 'wb') as handle: pickle.dump(dm_fleet_bus, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/fleet_bus.pickle')
with open(f, 'rb') as handle: dm_fleet_bus = pickle.load(handle)

# deepen and sum gas
dm_fleet_bus.deepen()
mapping_calc = {'ICE-gas': ['gas-lpg', 'gas-natural']}
dm_fleet_bus.groupby(mapping_calc, dim='Categories1', aggregation = "mean", regex=False, inplace=True)

# make rest of the variables (assuming they are all missing for now)
categories2_missing = categories2_all.copy()
for cat in dm_fleet_bus.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_fleet_bus.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_fleet_bus.sort("Categories1")

################
##### RAIL #####
################

# # get data
# # note: I assume that all high speed passenger trains are electric, and I'll take an average
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrRail_act",
#                 "variable" : "Stock of vehicles - total (representative train configuration)",
#                 "sheet_last_row" : "High speed passenger trains",                
#                 "sub_variables" : ["Metro and tram, urban light rail",
#                                     "Diesel oil",
#                                     "Electric",
#                                     "High speed passenger trains"],
#                 "calc_names" : ["metrotram_mt","train-conv_ICE-diesel","train-conv_CEV","train-hs_CEV"]}
# dm_fleet_rail = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/fleet_rail.pickle')
# with open(f, 'wb') as handle: pickle.dump(dm_fleet_rail, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/fleet_rail.pickle')
with open(f, 'rb') as handle: dm_fleet_rail = pickle.load(handle)

# aggregate trains and deepen
mapping_calc = {'rail_CEV': ['train-conv_CEV', 'train-hs_CEV'],
                'rail_ICE-diesel' : ["train-conv_ICE-diesel"]}
dm_fleet_rail.groupby(mapping_calc, dim='Variables', aggregation = "mean", regex=False, inplace=True)
dm_fleet_rail.deepen()

# make rest of the variables (assuming they are all missing for now)
categories2_missing = categories2_all.copy()
for cat in dm_fleet_rail.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_fleet_rail.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_fleet_rail.sort("Categories1")

# fix unit
dm_fleet_rail.units["metrotram"] = "vehicles"
dm_fleet_rail.units["rail"] = "vehicles"

########################
##### PUT TOGETHER #####
########################

dm_fleet = dm_fleet_2w.copy()
dm_fleet.append(dm_fleet_ldv,"Variables")
dm_fleet.append(dm_fleet_bus,"Variables")
dm_fleet.append(dm_fleet_rail,"Variables")
dm_fleet.sort("Variables")
dm_fleet.sort("Country")

# check
# dm_fleet.flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()

# there is a problem with fleet LDV ice gasoline before 2014. Seeing
# that also in Eurostat it's missing, I assume there is a problem with JRC data
# and I put it missing
dm_fleet = dm_fleet.flatten()
idx = dm_fleet.idx
years_fix = list(range(2000,2013+1))
for y in years_fix:
    dm_fleet.array[:,idx[y],idx["LDV_ICE-gasoline"]] = np.nan
dm_fleet.deepen()

# check
# dm_fleet.flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()


###################
##### FIX OTS #####
###################

# flatten
dm_fleet = dm_fleet.flatten()

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

dict_call = {"2W_ICE-gasoline" : {"n_adj" : 1},
             "LDV_BEV" : {"n_adj" : 2, "year_end_first_adj" : 2010, "year_start_second_adj" : 2021},
             "LDV_ICE-diesel" : {"n_adj" : 2, "year_end_first_adj" : 2010, "year_start_second_adj" : 2020},
             "LDV_ICE-gas" : {"n_adj" : 2, "year_end_first_adj" : 2010, "year_start_second_adj" : 2020},
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
        dict_new[key] = make_ots(dm_fleet, key, dict_call[key])
    else:
        dict_new[key] = make_ots(dm_fleet, key, dict_call[key], years_ots)

# append
dm_fleet = dict_new["2W_ICE-gasoline"].copy()
mylist = list(dict_call.keys())
mylist.remove("2W_ICE-gasoline")
for v in mylist:
    dm_fleet.append(dict_new[v],"Variables")
dm_fleet.sort("Variables")

# check
# dm_fleet.filter({"Country" : ["EU27"]}).datamatrix_plot()


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

# make a total
dm_total = dm_fleet.groupby({"total" : dm_fleet.col_labels["Variables"]}, dim='Variables', 
                            aggregation = "sum", regex=False, inplace=False)
dm_fleet.append(dm_total,"Variables")

# add missing years fts
dm_fleet.add(np.nan, col_label=years_fts, dummy=True, dim='Years')

# set default time window for linear trend
baseyear_start = 2000
baseyear_end = 2023

# check
# dm_fleet.filter({"Country" : ["EU27"]}).datamatrix_plot()

# # try fts
# product = "LDV_PHEV-diesel"
# (make_fts(dm_fleet, product, baseyear_start, baseyear_end, dim = "Variables").
#   datamatrix_plot(selected_cols={"Country" : ["EU27"], "Variables" : [product]}))

# make fts
dm_fleet = make_fts(dm_fleet, "total", baseyear_start, baseyear_end, dim = "Variables")
dm_fleet = make_fts(dm_fleet, "2W_ICE-gasoline", baseyear_start, baseyear_end, dim = "Variables")

# get 2050 values for bev and phev
# note: assuming 8% of total fleet being electric in 2050
# source: https://www.eea.europa.eu/publications/electric-vehicles-and-the-energy/download
idx = dm_fleet.idx
electric_2050 = dm_fleet.array[idx["EU27"],idx[2050],idx["total"]] * 0.08
dm_share = dm_fleet.filter({"Country" : ["EU27"], "Years" : [2021], "Variables" : ["LDV_BEV","LDV_PHEV-diesel","LDV_PHEV-gasoline","bus_BEV"]})
dm_share.normalise("Variables")
idx_share = dm_share.idx
BEV_2050 = dm_share.array[:,:,idx_share["LDV_BEV"]] * electric_2050
PHEV_diesel_2050 = dm_share.array[:,:,idx_share["LDV_PHEV-diesel"]] * electric_2050
PHEV_gasoline_2050 = dm_share.array[:,:,idx_share["LDV_PHEV-gasoline"]] * electric_2050
bus_BEV_2050 = dm_share.array[:,:,idx_share["bus_BEV"]] * electric_2050

# drop 2022-2023 for the electric vehicles (to avoid a flat line)
idx = dm_fleet.idx
for v in ["LDV_BEV","LDV_PHEV-diesel","LDV_PHEV-gasoline","bus_BEV"]:
    dm_fleet.array[idx["EU27"],idx[2022],idx[v]] = np.nan
    dm_fleet.array[idx["EU27"],idx[2023],idx[v]] = np.nan

# bev
dm_fleet.array[idx["EU27"],idx[2050],idx["LDV_BEV"]] = BEV_2050
dm_temp = linear_fitting(dm_fleet.filter({"Country" : ["EU27"], "Variables" : ["LDV_BEV"]}), years_ots + years_fts)
idx_temp = dm_temp.idx
dm_fleet.array[idx["EU27"],:,idx["LDV_BEV"]] = dm_temp.array[idx_temp["EU27"],:,idx_temp["LDV_BEV"]]

# ice
dm_fleet = make_fts(dm_fleet, "LDV_ICE-diesel", 2020, baseyear_end, dim = "Variables")
dm_fleet = make_fts(dm_fleet, "LDV_ICE-gas", baseyear_start, baseyear_end, dim = "Variables")
dm_fleet = make_fts(dm_fleet, "LDV_ICE-gasoline", baseyear_start, baseyear_end, dim = "Variables")

# phev
dm_fleet.array[idx["EU27"],idx[2050],idx["LDV_PHEV-diesel"]] = PHEV_diesel_2050
dm_temp = linear_fitting(dm_fleet.filter({"Country" : ["EU27"], "Variables" : ["LDV_PHEV-diesel"]}), years_ots + years_fts)
idx_temp = dm_temp.idx
dm_fleet.array[idx["EU27"],:,idx["LDV_PHEV-diesel"]] = dm_temp.array[idx_temp["EU27"],:,idx_temp["LDV_PHEV-diesel"]]
dm_fleet.array[idx["EU27"],idx[2050],idx["LDV_PHEV-gasoline"]] = PHEV_gasoline_2050
dm_temp = linear_fitting(dm_fleet.filter({"Country" : ["EU27"], "Variables" : ["LDV_PHEV-gasoline"]}), years_ots + years_fts)
idx_temp = dm_temp.idx
dm_fleet.array[idx["EU27"],:,idx["LDV_PHEV-gasoline"]] = dm_temp.array[idx_temp["EU27"],:,idx_temp["LDV_PHEV-gasoline"]]

# bus bev
dm_fleet.array[idx["EU27"],idx[2050],idx["bus_BEV"]] = bus_BEV_2050
dm_temp = linear_fitting(dm_fleet.filter({"Country" : ["EU27"], "Variables" : ["bus_BEV"]}), years_ots + years_fts)
idx_temp = dm_temp.idx
dm_fleet.array[idx["EU27"],:,idx["bus_BEV"]] = dm_temp.array[idx_temp["EU27"],:,idx_temp["bus_BEV"]]

# rest
dm_fleet = make_fts(dm_fleet, "bus_ICE-diesel", baseyear_start, baseyear_end, dim = "Variables")
dm_fleet = make_fts(dm_fleet, "bus_ICE-gas", baseyear_start, baseyear_end, dim = "Variables")
dm_fleet = make_fts(dm_fleet, "bus_ICE-gasoline", baseyear_start, baseyear_end, dim = "Variables")
dm_fleet = make_fts(dm_fleet, "metrotram_mt", baseyear_start, baseyear_end, dim = "Variables")
dm_fleet = make_fts(dm_fleet, "rail_CEV", baseyear_start, baseyear_end, dim = "Variables")
dm_fleet = make_fts(dm_fleet, "rail_ICE-diesel", baseyear_start, baseyear_end, dim = "Variables")

# check
# dm_fleet.filter({"Country" : ["EU27"]}).datamatrix_plot()

####################################
##### MAKE AS FINAL DATAMATRIX #####
####################################

DM_tra["fxa"]["passenger_tech"].units

# get it in shares
dm_fleet.drop("Variables","total")
dm_fleet_pc = dm_fleet.normalise("Variables",inplace=False)
dm_fleet_pc.rename_col_regex("_share","","Variables")

# rename and deepen
for v in dm_fleet.col_labels["Variables"]:
    dm_fleet.rename_col(v,"tra_passenger_technology-share_fleet_" + v, "Variables")
    dm_fleet_pc.rename_col(v,"tra_passenger_technology-share_fleet_" + v, "Variables")
dm_fleet.deepen_twice()
dm_fleet_pc.deepen_twice()

# check
# dm_fleet_pc.flatten().flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()

################
##### SAVE #####
################

list(DM_tra["fxa"])
DM_tra["fxa"]["passenger_tech"]

# save
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/passenger_fleet.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_fleet, handle, protocol=pickle.HIGHEST_PROTOCOL)
f = os.path.join(current_file_directory, '../data/datamatrix/fxa_passenger_technology-share_fleet.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_fleet_pc, handle, protocol=pickle.HIGHEST_PROTOCOL)



