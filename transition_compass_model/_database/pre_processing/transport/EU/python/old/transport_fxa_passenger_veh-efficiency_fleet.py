
# packages
from model.common.data_matrix_class import DataMatrix
from model.common.auxiliary_functions import linear_fitting
import pandas as pd
import pickle
import os
import numpy as np
import warnings
import eurostat
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
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/transport/EU/python/transport_fxa_passenger_veh-efficiency_fleet.py"

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


################################################
################### GET DATA ###################
################################################

DM_tra["ots"]["passenger_veh-efficiency_new"].units
DM_tra["ots"]["passenger_veh-efficiency_new"].write_df().columns
categories2_all = DM_tra["ots"]["passenger_veh-efficiency_new"].col_labels["Categories2"]

# get iso codes
dict_iso2 = eurostat_iso2_dict()
dict_iso2.pop('CH')  # Remove Switzerland
dict_iso2_jrc = jrc_iso2_dict()

# note on data in JRC
# On efficiency of total stock:
# Vehicle-efficiency - effective (kgoe/100 km) = Test cycle efficiency of total stock (kgoe/100 km) * Discrepancy between effective and test cycle efficiencies (ratio)
# The discrepancy factors have been created to match the data of the energy balances
# On efficiency of new:
# Test cycle efficiency of new vehicles (kgoe/100 km)

##############
##### 2W #####
##############

# get data
dict_extract = {"database" : "Transport",
                "sheet" : "TrRoad_ene",
                "variable" : "Vehicle-efficiency - effective (kgoe/100 km)",
                "sheet_last_row" : "Powered two-wheelers",
                "sub_variables" : ["Powered two-wheelers"],
                "calc_names" : ["2W"]}
dm_eneff_2w = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/eneff_2w.pickle')
with open(f, 'wb') as handle: pickle.dump(dm_eneff_2w, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/eneff_2w.pickle')
with open(f, 'rb') as handle: dm_eneff_2w = pickle.load(handle)

# make other variables
# assumption: 2w are mostly ICE-gasoline, so the energy efficiency we have will be assigned
# to 2W_ICE-gasoline, and the rest will be missing values for now
dm_eneff_2w.rename_col("2W","2W_ICE-gasoline","Variables")
dm_eneff_2w.deepen()
categories2_missing = categories2_all.copy()
for cat in dm_eneff_2w.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_eneff_2w.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_eneff_2w.sort("Categories1")


###############
##### LDV #####
###############


# # get data
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrRoad_ene",
#                 "variable" : "Vehicle-efficiency - effective (kgoe/100 km)",
#                 "sheet_last_row" : "Battery electric vehicles",
#                 "categories" : "Passenger cars",
#                 "sub_variables" : ["Gasoline engine",
#                                     "Diesel oil engine",
#                                     "LPG engine", "Natural gas engine", 
#                                     "Plug-in hybrid electric",
#                                     "Battery electric vehicles"],
#                 "calc_names" : ["LDV_ICE-gasoline","LDV_ICE-diesel","LDV_gas-lpg",
#                                 "LDV_gas-natural","LDV_PHEV","LDV_BEV"]}
# dm_eneff_ldv = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/eneff_ldv.pickle')
# with open(f, 'wb') as handle: pickle.dump(dm_eneff_ldv, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/eneff_ldv.pickle')
with open(f, 'rb') as handle: dm_eneff_ldv = pickle.load(handle)

# deepen and sum gas
dm_eneff_ldv.deepen()
mapping_calc = {'ICE-gas': ['gas-lpg', 'gas-natural']}
dm_eneff_ldv.groupby(mapping_calc, dim='Categories1', aggregation = "mean", regex=False, inplace=True)

# make PHEV diesel and PHEV gasoline
# assumption: they have the same efficiency
dm_temp = dm_eneff_ldv.filter({"Categories1" : ["PHEV"]})
dm_temp1 = dm_temp.copy()
dm_temp1.rename_col("PHEV","PHEV-gasoline","Categories1")
dm_temp1.append(dm_temp,"Categories1")
dm_temp1.rename_col("PHEV","PHEV-diesel","Categories1")
dm_eneff_ldv.drop("Categories1","PHEV")
dm_eneff_ldv.append(dm_temp1,"Categories1")
dm_eneff_ldv.sort("Categories1")

# make other variables
# TODO: FCEV energy efficiency can be same of Switzerland, to be added at the end, for now missing
# assumption: for now, rest is assumed to be missing
categories2_missing = categories2_all.copy()
for cat in dm_eneff_ldv.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_eneff_ldv.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_eneff_ldv.sort("Categories1")


#################
##### BUSES #####
#################

# # get data
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrRoad_ene",
#                 "variable" : "Vehicle-efficiency - effective (kgoe/100 km)",
#                 "sheet_last_row" : "Battery electric vehicles",
#                 "sub_variables" : ["Gasoline engine",
#                                     "Diesel oil engine",
#                                     "LPG engine", "Natural gas engine", 
#                                     "Battery electric vehicles"],
#                 "calc_names" : ["bus_ICE-gasoline","bus_ICE-diesel","bus_gas-lpg",
#                                 "bus_gas-natural","bus_BEV"],
#                 "categories" : "Motor coaches, buses and trolley buses"}
# dm_eneff_bus = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/eneff_bus.pickle')
# with open(f, 'wb') as handle: pickle.dump(dm_eneff_bus, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/eneff_bus.pickle')
with open(f, 'rb') as handle: dm_eneff_bus = pickle.load(handle)

# deepen and sum gas
dm_eneff_bus.deepen()
mapping_calc = {'ICE-gas': ['gas-lpg', 'gas-natural']}
dm_eneff_bus.groupby(mapping_calc, dim='Categories1', aggregation = "mean", regex=False, inplace=True)

# make rest of the variables (assuming they are all missing for now)
categories2_missing = categories2_all.copy()
for cat in dm_eneff_bus.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_eneff_bus.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_eneff_bus.sort("Categories1")

################
##### RAIL #####
################

# # get data
# # note: I assume that all high speed passenger trains are electric, and I'll take an average
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrRail_ene",
#                 "variable" : "Vehicle-efficiency (kgoe/100 km)",
#                 "sheet_last_row" : "High speed passenger trains",                
#                 "sub_variables" : ["Metro and tram, urban light rail",
#                                     "Diesel oil",
#                                     "Electric",
#                                     "High speed passenger trains"],
#                 "calc_names" : ["metrotram_mt","train-conv_ICE-diesel","train-conv_CEV","train-hs_CEV"]}
# dm_eneff_rail = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/eneff_rail.pickle')
# with open(f, 'wb') as handle: pickle.dump(dm_eneff_rail, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/eneff_rail.pickle')
with open(f, 'rb') as handle: dm_eneff_rail = pickle.load(handle)

# aggregate trains and deepen
mapping_calc = {'rail_CEV': ['train-conv_CEV', 'train-hs_CEV'],
                'rail_ICE-diesel' : ["train-conv_ICE-diesel"]}
dm_eneff_rail.groupby(mapping_calc, dim='Variables', aggregation = "mean", regex=False, inplace=True)
dm_eneff_rail.deepen()

# make rest of the variables (assuming they are all missing for now)
categories2_missing = categories2_all.copy()
for cat in dm_eneff_rail.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_eneff_rail.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_eneff_rail.sort("Categories1")

########################
##### PUT TOGETHER #####
########################

dm_eneff = dm_eneff_2w.copy()
dm_eneff.append(dm_eneff_ldv,"Variables")
dm_eneff.append(dm_eneff_bus,"Variables")
dm_eneff.append(dm_eneff_rail,"Variables")
dm_eneff.sort("Variables")
dm_eneff.sort("Country")

# substitute zero values with missing
dm_eneff.array[dm_eneff.array==0] = np.nan

###################
##### FIX OTS #####
###################

# do linear fitting of each variable with wathever is available
# note: bus_ICE-diesel: until before 2000 until 2012, after 2021 after 2012
dm_eneff_bus_icedie = dm_eneff.filter({"Variables" : ["bus"],"Categories1" : ["ICE-diesel"]})
dm_eneff = linear_fitting(dm_eneff, years_ots, min_t0=0,min_tb=0)
dm_eneff_bus_icedie = linear_fitting(dm_eneff_bus_icedie, list(range(1990,1999+1)),based_on=list(range(2000,2012+1)), min_t0=0.1,min_tb=0.1)
dm_eneff_bus_icedie = linear_fitting(dm_eneff_bus_icedie, list(range(2022,2023+1)),based_on=list(range(2012,2020+1)), min_t0=0.1,min_tb=0.1)
dm_eneff = dm_eneff.flatten()
dm_eneff.drop("Variables","bus_ICE-diesel")
dm_eneff.append(dm_eneff_bus_icedie.flatten(),"Variables")
dm_eneff.deepen()

# check
# dm_eneff.filter({"Country" : ["EU27"]}).datamatrix_plot()
# df = dm_eneff.write_df()
# df.columns


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

# add missing years fts
dm_eneff.add(np.nan, col_label=years_fts, dummy=True, dim='Years')

# set default time window for linear trend
baseyear_start = 2000
baseyear_end = 2019

# flatten
dm_eneff = dm_eneff.flatten()

# check
# dm_eneff.filter({"Country" : ["EU27"]}).datamatrix_plot()

# # try fts
# product = "LDV_BEV"
# (make_fts(dm_eneff, product, baseyear_start, baseyear_end, dim = "Variables").
#  datamatrix_plot(selected_cols={"Country" : ["EU27"], "Variables" : [product]}))

# make fts
dm_eneff = make_fts(dm_eneff, "2W_ICE-gasoline", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff = make_fts(dm_eneff, "LDV_BEV", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff = make_fts(dm_eneff, "LDV_ICE-diesel", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff = make_fts(dm_eneff, "LDV_ICE-gas", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff = make_fts(dm_eneff, "LDV_ICE-gasoline", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff = make_fts(dm_eneff, "LDV_PHEV-diesel", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff = make_fts(dm_eneff, "LDV_PHEV-gasoline", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff = make_fts(dm_eneff, "bus_BEV", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff = make_fts(dm_eneff, "bus_ICE-diesel", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff = make_fts(dm_eneff, "bus_ICE-gas", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff = make_fts(dm_eneff, "bus_ICE-gasoline", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff = make_fts(dm_eneff, "metrotram_mt", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff = make_fts(dm_eneff, "rail_CEV", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff = make_fts(dm_eneff, "rail_ICE-diesel", baseyear_start, baseyear_end, dim = "Variables")

# check
# dm_eneff.filter({"Country" : ["EU27"]}).datamatrix_plot()

####################################
##### MAKE AS FINAL DATAMATRIX #####
####################################

DM_tra["ots"]["passenger_veh-efficiency_new"].units

# rename and deepen
for v in dm_eneff.col_labels["Variables"]:
    dm_eneff.rename_col(v,"tra_passenger_veh-efficiency_new_" + v, "Variables")
dm_eneff.deepen()
dm_eneff.deepen(based_on="Variables")
dm_eneff.switch_categories_order("Categories1","Categories2")

# get it in MJ over km
dm_eneff.change_unit("tra_passenger_veh-efficiency_new", 41.868, "kgoe/100 km", "MJ/100 km")
dm_eneff.change_unit("tra_passenger_veh-efficiency_new", 1e-2, "MJ/100 km", "MJ/km")

# check
# dm_eneff.flatten().flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()

################
##### SAVE #####
################

# save
f = os.path.join(current_file_directory, '../data/datamatrix/fxa_passenger_veh-efficiency_fleet.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_eneff, handle, protocol=pickle.HIGHEST_PROTOCOL)






