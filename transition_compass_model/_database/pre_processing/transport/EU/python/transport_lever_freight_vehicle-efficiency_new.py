
# packages
from model.common.auxiliary_functions import linear_fitting
from _database.pre_processing.routine_JRC import get_jrc_data
from model.common.auxiliary_functions import eurostat_iso2_dict, jrc_iso2_dict, DataMatrix
import pickle
import os
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter("ignore")

# directories
current_file_directory = os.getcwd()

# load current transport pickle
filepath = os.path.join(current_file_directory, '../../../../data/datamatrix/transport.pickle')
with open(filepath, 'rb') as handle:
    DM_tra = pickle.load(handle)

# Set years range
years_setting = [1989, 2023, 2050, 5]
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

DM_tra["ots"]["freight_vehicle-efficiency_new"].write_df().columns
categories2_all = DM_tra["ots"]["freight_vehicle-efficiency_new"].col_labels["Categories2"]

# get iso codes
dict_iso2 = eurostat_iso2_dict()
dict_iso2.pop('CH')  # Remove Switzerland
dict_iso2_jrc = jrc_iso2_dict()

################
##### HDVL #####
################

# get data
dict_extract = {"database" : "Transport",
                "sheet" : "TrRoad_tech",
                "variable" : "Test cycle efficiency of new vehicles (kgoe/100 km)",
                "categories" : "Light commercial vehicles",
                "sheet_last_row" : "Battery electric vehicles",
                "sub_variables" : ["Gasoline engine","Diesel oil engine","LPG engine",
                                    "Natural gas engine","Battery electric vehicles"],
                "calc_names" : ["HDVL_ICE-gasoline","HDVL_ICE-diesel","HDVL_ICE-gas-lpg",
                                "HDVL_ICE-gas-natural","HDVL_BEV"]}
dm_hdvl = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

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

# get data
dict_extract = {"database" : "Transport",
                "sheet" : "TrRoad_tech",
                "variable" : "Test cycle efficiency of new vehicles (kgoe/100 km)",
                "sheet_last_row" : "Heavy goods vehicles",
                "sub_variables" : ["Heavy goods vehicles"],
                "calc_names" : ["HDVH"]}
dm_hdvh = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

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
dm_eneff_new = dm_hdvl.copy()
dm_eneff_new.append(dm_hdvh,"Variables")

# make HDVM as average between HDVL and HDVM
dm_hdvm = dm_eneff_new.flatten()
dm_hdvm.deepen()
dm_hdvm.groupby({"HDVM" : ["HDVL","HDVH"]}, 
                dim='Variables', aggregation = "mean", regex=False, inplace=True)

# make other variables
categories2_missing = categories2_all.copy()
for cat in dm_hdvm.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_hdvm.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_hdvm.sort("Categories1")

# put together
dm_eneff_new.append(dm_hdvm,"Variables")
dm_eneff_new.sort("Variables")
dm_eneff_new.sort("Country")
dm_eneff_new.sort("Years")

###############
##### IWW #####
###############

def get_specific_jrc_data(country_code, country_name, row_start, row_end, unit, variable = "aviation_kerosene", 
                          database = "JRC-IDEES-2021_x1990_Aviation_EU"):
    
    filepath_jrc = os.path.join(current_file_directory, f"../../../industry/eu/data/JRC-IDEES-2021/EU27/{database}.xlsx")
    df_temp = pd.read_excel(filepath_jrc, sheet_name=country_code)
    df_temp = df_temp.iloc[row_start:row_end,:]
    indexes = df_temp.columns[0]
    df_temp = pd.melt(df_temp, id_vars = indexes, var_name='year')
    df_temp.columns = ["Country","Years",f"{variable}[{unit}]"]
    df_temp["Country"] = country_name
    
    return df_temp

country_codes = list(dict_iso2_jrc.keys())
country_names = list(dict_iso2_jrc.values())

# get data from jrc
df_iww = pd.concat(
    [get_specific_jrc_data(code, name, 46, 47, "kgoe/100 km", "IWW_ICE", "JRC-IDEES-2021_x1990_Navigation_Domestic") 
     for code,name in zip(country_codes, country_names)],
    ignore_index=True)
dm_iww = DataMatrix.create_from_df(df_iww, 0)

# # assumption: energy efficiency of new is the same of energy efficiency of stock

# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrNavi_ene",
#                 "variable" : "Vehicle-efficiency (kgoe/100 km)",
#                 "sheet_last_row" : "Inland waterways",
#                 "sub_variables" : ["Inland waterways"],
#                 "calc_names" : ["IWW"]}
# dm_iww = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# substitute 0 with nans (to avoid that zeroes get in the averages)
dm_iww.array[dm_iww.array==0] = np.nan

# # assuming most of it is ICE
# dm_iww.rename_col("IWW","IWW_ICE","Variables")

# make other variables as missing
dm_iww.deepen()
categories2_missing = categories2_all.copy()
for cat in dm_iww.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_iww.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_iww.sort("Categories1")

####################
##### aviation #####
####################

df_avi = pd.concat([get_specific_jrc_data(code, name, 138, 139, "kgoe/100 km") for code,name in zip(country_codes, country_names)],ignore_index=True)
dm_avi = DataMatrix.create_from_df(df_avi, 0)

# # assumption: energy efficiency of new is theoretical energy efficiency of stock (which is without the adjustment to match energy balances)
# # in theory it's not, but this one is lower than the effective, and the assumption is that this difference
# # is the same there would be between stock efficiency and new efficiency

# # get data
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrAvia_ene",
#                 "variable" : "Vehicle-efficiency - theoretical (kgoe/100 km)*",
#                 "sheet_last_row" : "Freight transport",
#                 "sub_variables" : ["Freight transport"],
#                 "calc_names" : ["aviation"]}
# dm_avi = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# # assuming most planes are gasoline
# dm_avi.rename_col("aviation","aviation_ICE","Variables")
dm_avi.deepen()

# assuming that all else is nan
categories2_missing = categories2_all.copy()
for cat in dm_avi.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_avi.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_avi.sort("Categories1")

####################
##### maritime #####
####################

# get data from jrc
# note: I assume that inside international EU file, the voice "intra-EU27" includes also the 
# voice "coastal shipping" mentioned in the domestic EU file
df_mar = pd.concat(
    [get_specific_jrc_data(code, name, 46, 47, "kgoe/100 km", "marine_ICE", "JRC-IDEES-2021_x1990_Navigation_International_EU") 
     for code,name in zip(country_codes, country_names)],
    ignore_index=True)
dm_mar = DataMatrix.create_from_df(df_mar, 0)

# # assumption: energy efficiency of new is the same of energy efficiency of stock

# # get data on energy efficiency
# dict_extract = {"database" : "Transport",
#                 "sheet" : "MBunk_ene",
#                 "variable" : "Vehicle-efficiency (kgoe/100 km)",
#                 "sheet_last_row" : "Intra-EEA",
#                 "sub_variables" : ["Intra-EEA"],
#                 "calc_names" : ["marine"]}
# dm_mar = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# substitute 0 with nans (to avoid that zeroes get in the averages)
dm_mar.array[dm_mar.array==0] = np.nan

# # assuming they are all diesel
# dm_mar.rename_col("marine","marine_ICE","Variables")

# make other variables
dm_mar.deepen()
categories2_missing = categories2_all.copy()
for cat in dm_mar.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_mar.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_mar.sort("Categories1")


################
##### RAIL #####
################

# assumption: difference between stock and new is same of hdv

# get rail
dict_extract = {"database" : "Transport",
                "sheet" : "TrRail_ene",
                "variable" : "Vehicle-efficiency (kgoe/100 km)",
                "sheet_last_row" : "Electric",
                "categories": "Freight transport",
                "sub_variables" : ["Diesel oil",
                                    "Electric"],
                "calc_names" : ["rail_ICE-diesel","rail_CEV"]}
dm_eneff_rail = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# aggregate trains and deepen
dm_eneff_rail.deepen()

# load data of energy efficiency for stock of hdvh and get ratios, and apply them to trains
dict_extract = {"database" : "Transport",
                "sheet" : "TrRoad_ene",
                "variable" : "Vehicle-efficiency - effective (kgoe/100 km)",
                "sheet_last_row" : "Heavy goods vehicles",
                "sub_variables" : ["Heavy goods vehicles"],
                "calc_names" : ["HDVH"]}
dm_eneff_hdvh = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
dm_eneff_hdvh.array[dm_eneff_hdvh.array==0] = np.nan
dm_new = dm_hdvh.copy()
dm_new.group_all(dim='Categories1', aggregation = "mean")
arr_temp = dm_new.array / dm_eneff_hdvh.array
dm_eneff_rail.array = dm_eneff_rail.array * arr_temp[...,np.newaxis]
dm_eneff_new_rail = dm_eneff_rail.copy()

# make rest of the variables (assuming they are all missing for now)
categories2_missing = categories2_all.copy()
for cat in dm_eneff_new_rail.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_eneff_new_rail.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_eneff_new_rail.sort("Categories1")

########################
##### PUT TOGETHER #####
########################

dm_eneff_new.add(np.nan, "Years", list(range(1990,1999+1)), dummy=True)
dm_eneff_new.sort("Years")
dm_eneff_new.append(dm_iww,"Variables")
dm_eneff_new.append(dm_avi,"Variables")
dm_eneff_new.append(dm_mar,"Variables")
dm_eneff_new_rail.add(np.nan, "Years", list(range(1990,1999+1)), dummy=True)
dm_eneff_new_rail.sort("Years")
dm_eneff_new.append(dm_eneff_new_rail,"Variables")
dm_eneff_new.sort("Variables")
dm_eneff_new.sort("Country")

# substitute zero values with missing
dm_eneff_new.array[dm_eneff_new.array==0] = np.nan

###################
##### FIX OTS #####
###################

# flatten
dm_eneff_new = dm_eneff_new.flatten()

# check
# dm_eneff_new.filter({"Country" : ["EU27"]}).datamatrix_plot()

# # add missing years
# dm_eneff_new.add(np.nan,col_label=list(range(1990,1999+1)), dummy=True, dim='Years')
# dm_eneff_new.add(np.nan,col_label=list(range(2022,2023+1)), dummy=True, dim='Years')
# dm_eneff_new.sort("Years")

# new variabs list
dict_new = {}

def make_ots(dm, variable, periods_dicts, years_ots = None):
    
    dm_temp = dm.filter({"Variables" : [variable]})
    if periods_dicts["n_adj"] == 1:
        dm_temp = linear_fitting(dm_temp, years_ots, min_t0=0.1,min_tb=0.1)
    if periods_dicts["n_adj"] == 2:
        dm_temp = linear_fitting(dm_temp, list(range(startyear,1999+1)), 
                                 based_on=list(range(2000,periods_dicts["year_end_first_adj"]+1)), 
                                 min_t0=0.1,min_tb=0.1)
        dm_temp = linear_fitting(dm_temp, list(range(2022,2023+1)), 
                                 based_on=list(range(periods_dicts["year_start_second_adj"],2021+1)), 
                                 min_t0=0.1,min_tb=0.1)
    return dm_temp

dict_call = {"HDVH_BEV" : {"n_adj" : 2, "year_end_first_adj" : 2010, "year_start_second_adj" : 2010},
             "HDVH_ICE-diesel" : {"n_adj" : 1},
             "HDVH_ICE-gas" : {"n_adj" : 1},
             "HDVH_ICE-gasoline": {"n_adj" : 1},
             "HDVL_BEV" : {"n_adj" : 2, "year_end_first_adj" : 2010, "year_start_second_adj" : 2020},
             "HDVL_ICE-diesel" : {"n_adj" : 1},
             "HDVL_ICE-gas" : {"n_adj" : 2, "year_end_first_adj" : 2015, "year_start_second_adj" : 2020},
             "HDVL_ICE-gasoline" : {"n_adj" : 1},
             "HDVM_BEV" : {"n_adj" : 2, "year_end_first_adj" : 2010, "year_start_second_adj" : 2015},
             "HDVM_ICE-diesel" :  {"n_adj" : 1},
             "HDVM_ICE-gas" : {"n_adj" : 1},
             "HDVM_ICE-gasoline" : {"n_adj" : 1},
             "IWW_ICE" : {"n_adj" : 1},
             "aviation_kerosene" : {"n_adj" : 1},
             "marine_ICE" : {"n_adj" : 1},
             "rail_CEV" : {"n_adj" : 1},
             "rail_ICE-diesel" : {"n_adj" : 1}}

for key in dict_call.keys():
    if len(dict_call[key]) > 1:
        dict_new[key] = make_ots(dm_eneff_new, key, dict_call[key])
    else:
        dict_new[key] = make_ots(dm_eneff_new, key, dict_call[key], years_ots)

# append
dm_eneff_new = dict_new["HDVH_BEV"].copy()
mylist = list(dict_call.keys())
mylist.remove("HDVH_BEV")
for v in mylist:
    dm_eneff_new.append(dict_new[v],"Variables")
dm_eneff_new.sort("Variables")

# check
# dm_eneff_new.filter({"Country" : ["EU27"]}).datamatrix_plot()

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
dm_eneff_new.add(np.nan, col_label=years_fts, dummy=True, dim='Years')

# set default time window for linear trend
baseyear_start = 2000
baseyear_end = 2021

# check
# dm_eneff_new.filter({"Country" : ["EU27"]}).datamatrix_plot()

# # try fts
# product = 'HDVH_ICE-gasoline'
# (make_fts(dm_eneff_new, product, dict_call[product]["year_start_second_adj"], baseyear_end, dim = "Variables").
#   datamatrix_plot(selected_cols={"Country" : ["EU27"], "Variables" : [product]}))

# make fts
for key in dict_call.keys():
    if len(dict_call[key]) > 1:
        dm_eneff_new = make_fts(dm_eneff_new, key, dict_call[key]["year_start_second_adj"], baseyear_end, dim = "Variables")
    else:
        dm_eneff_new = make_fts(dm_eneff_new, key, baseyear_start, baseyear_end, dim = "Variables")

# check
# dm_eneff_new.filter({"Country" : ["EU27"]}).datamatrix_plot()

# deepen
dm_eneff_new.deepen()

####################################
##### MAKE AS FINAL DATAMATRIX #####
####################################

DM_tra["ots"]["freight_vehicle-efficiency_new"].units

# rename and deepen
for v in dm_eneff_new.col_labels["Variables"]:
    dm_eneff_new.rename_col(v,"tra_freight_vehicle-efficiency_new_" + v, "Variables")
dm_eneff_new.deepen(based_on="Variables")
dm_eneff_new.switch_categories_order("Categories1","Categories2")

# get it in MJ over km
dm_eneff_new.change_unit("tra_freight_vehicle-efficiency_new", 41.868, "kgoe/100 km", "MJ/100 km")
dm_eneff_new.change_unit("tra_freight_vehicle-efficiency_new", 1e-2, "MJ/100 km", "MJ/km")

# insert missing categories
categories2_missing = categories2_all.copy()
for cat in dm_eneff_new.col_labels["Categories2"]: 
    if cat in categories2_missing: categories2_missing.remove(cat)
dm_eneff_new.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories2")
dm_eneff_new.sort("Categories2")

# check
# dm_eneff_new.flatten().flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()

# # for now freight does not have the category kerosene, so move it back to ICE
# dm_eneff_new[...,"aviation","ICE"] = dm_eneff_new[...,"aviation","kerosene"]
# dm_eneff_new.drop("Categories2","kerosene")

# drop start year
dm_eneff_new.drop("Years",startyear)

################
##### SAVE #####
################

# split between ots and fts
DM_ene = {"ots": {"freight_vehicle-efficiency_new" : []}, "fts": {"freight_vehicle-efficiency_new" : dict()}}
DM_ene["ots"]["freight_vehicle-efficiency_new"] = dm_eneff_new.filter({"Years" : list(range(1990,2023+1))})
for i in range(1,4+1):
    DM_ene["fts"]["freight_vehicle-efficiency_new"][i] = dm_eneff_new.filter({"Years" : years_fts})

# save
f = os.path.join(current_file_directory, '../data/datamatrix/lever_freight_vehicle-efficiency_new.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM_ene, handle, protocol=pickle.HIGHEST_PROTOCOL)




