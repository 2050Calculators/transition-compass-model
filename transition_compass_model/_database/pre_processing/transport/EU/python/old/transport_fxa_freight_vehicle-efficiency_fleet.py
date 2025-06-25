
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
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/transport/EU/python/transport_lever_freight_vehicle-efficiency_fleet.py"

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

DM_tra["ots"]["freight_vehicle-efficiency_new"].units
DM_tra["ots"]["freight_vehicle-efficiency_new"].write_df().columns
categories2_all = DM_tra["ots"]["freight_vehicle-efficiency_new"].col_labels["Categories2"]

# get iso codes
dict_iso2 = eurostat_iso2_dict()
dict_iso2.pop('CH')  # Remove Switzerland
dict_iso2_jrc = jrc_iso2_dict()

################
##### HDVL #####
################

# # get data
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrRoad_ene",
#                 "variable" : "Vehicle-efficiency - effective (kgoe/100 km)",
#                 "categories" : "Light commercial vehicles",
#                 "sheet_last_row" : "Battery electric vehicles",
#                 "sub_variables" : ["Gasoline engine","Diesel oil engine","LPG engine",
#                                     "Natural gas engine","Battery electric vehicles"],
#                 "calc_names" : ["HDVL_ICE-gasoline","HDVL_ICE-diesel","HDVL_ICE-gas-lpg",
#                                 "HDVL_ICE-gas-natural","HDVL_BEV"]}
# dm_hdvl = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/eneff_hdvl.pickle')
# with open(f, 'wb') as handle: pickle.dump(dm_hdvl, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/eneff_hdvl.pickle')
with open(f, 'rb') as handle: dm_hdvl = pickle.load(handle)

# substitute 0 with nans (to avoid that zeroes get in the averages)
dm_hdvl.array[dm_hdvl.array==0] = np.nan

# aggregate gas
dm_hdvl.groupby({"HDVL_ICE-gas" : ["HDVL_ICE-gas-lpg","HDVL_ICE-gas-natural"]}, 
                dim='Variables', aggregation = "mean", regex=False, inplace=True)

# make other variables
dm_hdvl.deepen()
categories2_missing = categories2_all.copy()
for cat in dm_hdvl.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_hdvl.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_hdvl.sort("Categories1")

################
##### HDVH #####
################

# # get data
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrRoad_ene",
#                 "variable" : "Vehicle-efficiency - effective (kgoe/100 km)",
#                 "sheet_last_row" : "Heavy goods vehicles",
#                 "sub_variables" : ["Heavy goods vehicles"],
#                 "calc_names" : ["HDVH"]}
# dm_hdvh = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/eneff_hdvh.pickle')
# with open(f, 'wb') as handle: pickle.dump(dm_hdvh, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/eneff_hdvh.pickle')
with open(f, 'rb') as handle: dm_hdvh = pickle.load(handle)

# substitute 0 with nans (to avoid that zeroes get in the averages)
dm_hdvh.array[dm_hdvh.array==0] = np.nan

# use same ratios of HDVL to make the split between types of engines
dm_temp = dm_hdvl.flatten()
dm_temp1 = dm_temp.groupby({"HDVL" : ['HDVL_BEV', 'HDVL_ICE-diesel', 'HDVL_ICE-gas', 'HDVL_ICE-gasoline']}, 
                           dim='Variables', aggregation = "mean", regex=False, inplace=False)
dm_temp.append(dm_temp1,"Variables")
idx = dm_temp.idx
dm_temp.array = dm_temp.array/dm_temp.array[...,idx["HDVL"],np.newaxis]
dm_temp.drop("Variables",["HDVL"])
dm_temp.array = dm_temp.array * dm_hdvh.array
dm_temp.rename_col_regex("HDVL","HDVH","Variables")
dm_hdvh = dm_temp.copy()

# make other variables
dm_hdvh.deepen()
categories2_missing = categories2_all.copy()
for cat in dm_hdvh.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_hdvh.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_hdvh.sort("Categories1")

################
##### HDVM #####
################

# put together
dm_eneff = dm_hdvl.copy()
dm_eneff.append(dm_hdvh,"Variables")

# make HDVM as average between HDVL and HDVM
dm_hdvm = dm_eneff.flatten()
dm_hdvm.deepen()
dm_hdvm.groupby({"HDVM" : ["HDVL","HDVH"]}, 
                dim='Variables', aggregation = "mean", regex=False, inplace=True)

# make other variables
categories2_missing = categories2_all.copy()
for cat in dm_hdvm.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_hdvm.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_hdvm.sort("Categories1")

# put together
dm_eneff.append(dm_hdvm,"Variables")
dm_eneff.sort("Variables")
dm_eneff.sort("Country")
dm_eneff.sort("Years")

###############
##### IWW #####
###############

# # get data on total energy consumption
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrNavi_ene",
#                 "variable" : "Energy consumption (ktoe)",
#                 "categories" : "Inland waterways",
#                 "sheet_last_row" : "Biogases",
#                 "sub_variables" : ["Gas oil and diesel oil (excluding biofuel portion)",
#                                    "Fuel oil and Other oil products",
#                                    "Blended biofuels",
#                                    "Natural gas","Biogases"],
#                 "calc_names" : ["IWW_ICE-diesel-fuel",
#                                 "IWW_ICE-gasoline",
#                                 "IWW_ICE-diesel-biofuel",
#                                 "IWW_ICE-gas-natural","IWW_ICE-gas-biogas"]}
# dm_iww_tot = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/ene_iww.pickle')
# with open(f, 'wb') as handle: pickle.dump(dm_iww_tot, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # load
# f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/ene_iww.pickle')
# with open(f, 'rb') as handle: dm_iww_tot = pickle.load(handle)

# # get diesel and gas
# dm_iww_tot.groupby({"IWW_ICE-diesel" : ["IWW_ICE-diesel-fuel","IWW_ICE-diesel-biofuel"],
#                     "IWW_ICE-gas" : ["IWW_ICE-gas-natural","IWW_ICE-gas-biogas"]}, 
#                    dim='Variables', aggregation = "sum", regex=False, inplace=True)

# # substitute 0 with nans (to avoid that zeroes get in the averages)
# dm_iww_tot.array[dm_iww_tot.array==0] = np.nan

# # # get data on energy efficiency
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrNavi_ene",
#                 "variable" : "Vehicle-efficiency (kgoe/100 km)",
#                 "sheet_last_row" : "Inland waterways",
#                 "sub_variables" : ["Inland waterways"],
#                 "calc_names" : ["IWW"]}
# dm_iww = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/eneff_iww.pickle')
# with open(f, 'wb') as handle: pickle.dump(dm_iww, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/eneff_iww.pickle')
with open(f, 'rb') as handle: dm_iww = pickle.load(handle)

# substitute 0 with nans (to avoid that zeroes get in the averages)
dm_iww.array[dm_iww.array==0] = np.nan

# # use ratios of total energy consumed to get the energy efficiency values by engine type
# dm_temp = dm_iww_tot.copy()
# dm_temp1 = dm_temp.groupby({"IWW" : ['IWW_ICE-diesel', 'IWW_ICE-gas', 'IWW_ICE-gasoline']}, 
#                            dim='Variables', aggregation = "mean", regex=False, inplace=False)
# dm_temp.append(dm_temp1,"Variables")
# idx = dm_temp.idx
# dm_temp.array = dm_temp.array/dm_temp.array[...,idx["IWW"],np.newaxis]
# dm_temp.drop("Variables",["IWW"])
# dm_temp.array = dm_temp.array * dm_iww.array
# for v in dm_temp.col_labels["Variables"]: dm_temp.units[v] = "kgoe/100 km"
# dm_iww = dm_temp.copy()

# assuming most of it is diesel
dm_iww.rename_col("IWW","IWW_ICE-diesel","Variables")

# make other variables as missing
dm_iww.deepen()
categories2_missing = categories2_all.copy()
for cat in dm_iww.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_iww.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_iww.sort("Categories1")

####################
##### aviation #####
####################

# # get data
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrAvia_ene",
#                 "variable" : "Vehicle-efficiency - effective (kgoe/100 km)",
#                 "sheet_last_row" : "Freight transport",
#                 "sub_variables" : ["Freight transport"],
#                 "calc_names" : ["aviation"]}
# dm_avi = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/eneff_avi.pickle')
# with open(f, 'wb') as handle: pickle.dump(dm_avi, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/eneff_avi.pickle')
with open(f, 'rb') as handle: dm_avi = pickle.load(handle)

# assuming most planes are gasoline
dm_avi.rename_col("aviation","aviation_ICE-gasoline","Variables")
dm_avi.deepen()

# assuming diesel planes are as efficient as gasoline planes: https://www.aviationconsumer.com/industry-news/flight-fuel-efficiency-is-diesel-really-better/?utm_source=chatgpt.com
dm_temp = dm_avi.copy()
dm_temp.rename_col("ICE-gasoline","ICE-diesel","Categories1")
dm_avi.append(dm_temp,"Categories1")

# assuming that all else is nan (there should be kerosene but we do not have it in the model)
categories2_missing = categories2_all.copy()
for cat in dm_avi.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_avi.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_avi.sort("Categories1")

####################
##### maritime #####
####################

# # get data on total energy consumption for eea
# dict_extract = {"database" : "Transport",
#                 "sheet" : "MBunk_ene",
#                 "variable" : "Total energy consumption (ktoe)",
#                 "categories" : "Intra-EEA",
#                 "sheet_last_row" : "Biogases",
#                 "sub_variables" : ["Gas oil and diesel oil (excluding biofuel portion)",
#                                     "Fuel oil and Other oil products",
#                                     "Blended biofuels",
#                                     "Natural gas","Biogases"],
#                 "calc_names" : ["marine_ICE-diesel-fuel",
#                                 "marine_ICE-gasoline",
#                                 "marine_ICE-diesel-biofuel",
#                                 "marine_ICE-gas-natural","marine_ICE-gas-biogas"]}
# dm_mar_eea_tot = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/ene_marine_eea.pickle')
# with open(f, 'wb') as handle: pickle.dump(dm_mar_eea_tot, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # load
# f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/ene_marine_eea.pickle')
# with open(f, 'rb') as handle: dm_mar_eea_tot = pickle.load(handle)

# # get diesel and gas
# dm_mar_eea_tot.groupby({"marine_ICE-diesel" : ["marine_ICE-diesel-fuel","marine_ICE-diesel-biofuel"],
#                         "marine_ICE-gas" : ["marine_ICE-gas-natural","marine_ICE-gas-biogas"]}, 
#                        dim='Variables', aggregation = "sum", regex=False, inplace=True)

# # substitute 0 with nans (to avoid that zeroes get in the averages)
# dm_mar_eea_tot.array[dm_mar_eea_tot.array==0] = np.nan

# # get data on energy efficiency
# dict_extract = {"database" : "Transport",
#                 "sheet" : "MBunk_ene",
#                 "variable" : "Vehicle-efficiency (kgoe/100 km)",
#                 "sheet_last_row" : "Intra-EEA",
#                 "sub_variables" : ["Intra-EEA"],
#                 "calc_names" : ["marine"]}
# dm_mar = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/eneff_marine.pickle')
# with open(f, 'wb') as handle: pickle.dump(dm_mar, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/eneff_marine.pickle')
with open(f, 'rb') as handle: dm_mar = pickle.load(handle)

# substitute 0 with nans (to avoid that zeroes get in the averages)
dm_mar.array[dm_mar.array==0] = np.nan

# assuming they are all diesel
dm_mar.rename_col("marine","marine_ICE-diesel","Variables")

# make other variables
dm_mar.deepen()
categories2_missing = categories2_all.copy()
for cat in dm_mar.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_mar.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_mar.sort("Categories1")


################
##### RAIL #####
################

# # get data
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrRail_ene",
#                 "variable" : "Vehicle-efficiency (kgoe/100 km)",
#                 "sheet_last_row" : "Electric",
#                 "categories": "Freight transport",
#                 "sub_variables" : ["Diesel oil",
#                                    "Electric"],
#                 "calc_names" : ["rail_ICE-diesel","rail_CEV"]}
# dm_eneff_rail = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/eneff_freight_rail.pickle')
# with open(f, 'wb') as handle: pickle.dump(dm_eneff_rail, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/eneff_freight_rail.pickle')
with open(f, 'rb') as handle: dm_eneff_rail = pickle.load(handle)

# make rest of the variables (assuming they are all missing for now)
dm_eneff_rail.deepen()
categories2_missing = categories2_all.copy()
for cat in dm_eneff_rail.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_eneff_rail.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_eneff_rail.sort("Categories1")

########################
##### PUT TOGETHER #####
########################

dm_eneff.append(dm_iww,"Variables")
dm_eneff.append(dm_avi,"Variables")
dm_eneff.append(dm_mar,"Variables")
dm_eneff.append(dm_eneff_rail,"Variables")
dm_eneff.sort("Variables")
dm_eneff.sort("Country")

# substitute zero values with missing
dm_eneff.array[dm_eneff.array==0] = np.nan

###################
##### FIX OTS #####
###################

# flatten
dm_eneff = dm_eneff.flatten()

# check
# dm_eneff.filter({"Country" : ["EU27"]}).datamatrix_plot()

# # add missing years
# dm_eneff.add(np.nan,col_label=list(range(1990,1999+1)), dummy=True, dim='Years')
# dm_eneff.add(np.nan,col_label=list(range(2022,2023+1)), dummy=True, dim='Years')
# dm_eneff.sort("Years")

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

dict_call = {"HDVH_BEV" : {"n_adj" : 2, "year_end_first_adj" : 2010, "year_start_second_adj" : 2020},
             "HDVH_ICE-diesel" : {"n_adj" : 2, "year_end_first_adj" : 2010, "year_start_second_adj" : 2010},
             "HDVH_ICE-gas" : {"n_adj" : 2, "year_end_first_adj" : 2015, "year_start_second_adj" : 2015},
             "HDVH_ICE-gasoline": {"n_adj" : 2, "year_end_first_adj" : 2010, "year_start_second_adj" : 2010},
             "HDVL_BEV" : {"n_adj" : 2, "year_end_first_adj" : 2010, "year_start_second_adj" : 2020},
             "HDVL_ICE-diesel" : {"n_adj" : 2, "year_end_first_adj" : 2001, "year_start_second_adj" : 2020},
             "HDVL_ICE-gas" : {"n_adj" : 2, "year_end_first_adj" : 2001, "year_start_second_adj" : 2020},
             "HDVL_ICE-gasoline" : {"n_adj" : 2, "year_end_first_adj" : 2020, "year_start_second_adj" : 2020},
             "HDVM_BEV" : {"n_adj" : 2, "year_end_first_adj" : 2010, "year_start_second_adj" : 2020},
             "HDVM_ICE-diesel" :  {"n_adj" : 2, "year_end_first_adj" : 2010, "year_start_second_adj" : 2010},
             "HDVM_ICE-gas" : {"n_adj" : 2, "year_end_first_adj" : 2015, "year_start_second_adj" : 2015},
             "HDVM_ICE-gasoline" : {"n_adj" : 2, "year_end_first_adj" : 2010, "year_start_second_adj" : 2010},
             "IWW_ICE-diesel" : {"n_adj" : 1},
             "aviation_ICE-diesel" : {"n_adj" : 1},
             "aviation_ICE-gasoline" : {"n_adj" : 1},
             "marine_ICE-diesel" : {"n_adj" : 1},
             "rail_CEV" : {"n_adj" : 1},
             "rail_ICE-diesel" : {"n_adj" : 1}}

for key in dict_call.keys():
    if len(dict_call[key]) > 1:
        dict_new[key] = make_ots(dm_eneff, key, dict_call[key])
    else:
        dict_new[key] = make_ots(dm_eneff, key, dict_call[key], years_ots)

# append
dm_eneff = dict_new["HDVH_BEV"].copy()
mylist = list(dict_call.keys())
mylist.remove("HDVH_BEV")
for v in mylist:
    dm_eneff.append(dict_new[v],"Variables")
dm_eneff.sort("Variables")

# check
# dm_eneff.filter({"Country" : ["EU27"]}).datamatrix_plot()

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
baseyear_end = 2021

# check
# dm_eneff.filter({"Country" : ["EU27"]}).datamatrix_plot()

# # try fts
# product = 'HDVH_ICE-gasoline'
# (make_fts(dm_eneff, product, dict_call[product]["year_start_second_adj"], baseyear_end, dim = "Variables").
#   datamatrix_plot(selected_cols={"Country" : ["EU27"], "Variables" : [product]}))

# make fts
for key in dict_call.keys():
    if len(dict_call[key]) > 1:
        dm_eneff = make_fts(dm_eneff, key, dict_call[key]["year_start_second_adj"], baseyear_end, dim = "Variables")
    else:
        dm_eneff = make_fts(dm_eneff, key, baseyear_start, baseyear_end, dim = "Variables")

# check
# dm_eneff.filter({"Country" : ["EU27"]}).datamatrix_plot()

# deepen
dm_eneff.deepen()

####################################
##### MAKE AS FINAL DATAMATRIX #####
####################################

DM_tra["ots"]["freight_vehicle-efficiency_new"].units

# rename and deepen
for v in dm_eneff.col_labels["Variables"]:
    dm_eneff.rename_col(v,"tra_freight_vehicle-efficiency_new_" + v, "Variables")
dm_eneff.deepen(based_on="Variables")
dm_eneff.switch_categories_order("Categories1","Categories2")

# get it in MJ over km
dm_eneff.change_unit("tra_freight_vehicle-efficiency_new", 41.868, "kgoe/100 km", "MJ/100 km")
dm_eneff.change_unit("tra_freight_vehicle-efficiency_new", 1e-2, "MJ/100 km", "MJ/km")

# check
# dm_eneff.flatten().flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()

################
##### SAVE #####
################

# save
f = os.path.join(current_file_directory, '../data/datamatrix/fxa_freight_vehicle-efficiency_new.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_eneff, handle, protocol=pickle.HIGHEST_PROTOCOL)
