
# packages
from model.common.auxiliary_functions import linear_fitting, eurostat_iso2_dict, jrc_iso2_dict
from _database.pre_processing.routine_JRC import get_jrc_data
from model.common.data_matrix_class import DataMatrix
import pickle
import os
import numpy as np
import warnings
import pandas as pd
# from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat
warnings.simplefilter("ignore")
import plotly.io as pio
pio.renderers.default='browser'

# directories
current_file_directory = os.getcwd()

# load current transport pickle
filepath = os.path.join(current_file_directory, '../../../../data/datamatrix/transport.pickle')
with open(filepath, 'rb') as handle:
    DM_tra = pickle.load(handle)

# load pkm
filepath = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/passenger_pkm.pickle')
with open(filepath, 'rb') as handle:
    DM_pkm = pickle.load(handle)
    
# load pkm for planes
filepath = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/aviation_pkm.pickle')
with open(filepath, 'rb') as handle:
    dm_pkm_avi = pickle.load(handle)
    
# load seats for planes
filepath = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/passenger_fleet.pickle')
with open(filepath, 'rb') as handle:
    dm_seats_avi = pickle.load(handle)
    dm_seats_avi = dm_seats_avi.filter({"Variables" : ["aviation_kerosene"]})

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

DM_tra["ots"]["passenger_occupancy"].units

# get iso codes
dict_iso2 = eurostat_iso2_dict()
dict_iso2.pop('CH')  # Remove Switzerland
dict_iso2_jrc = jrc_iso2_dict()

########################
##### LDV, 2W, BUS #####
########################

# get data
dict_extract = {"database" : "Transport",
                "sheet" : "TrRoad_act",
                "variable" : "Vehicle-km driven (mio km)",
                "sheet_last_row" : "Freight transport",
                "sub_variables" : ["Powered two-wheelers",
                                    "Passenger cars",
                                    "Motor coaches, buses and trolley buses"],
                "calc_names" : ["2W","LDV","bus"]}
dm_vkm_ltb = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# sort
dm_vkm_ltb.sort("Variables")

# check
# dm_vkm_ltb.filter({"Country" : ["EU27"]}).datamatrix_plot()

################
##### RAIL #####
################

# note: also taking this one directly from JRC

# get data
dict_extract = {"database" : "Transport",
                "sheet" : "TrRail_act",
                "variable" : "Vehicle-km (mio km)",
                "sheet_last_row" : "High speed passenger trains",
                "sub_variables" : ["Metro and tram, urban light rail",
                                    "Conventional passenger trains",
                                    "High speed passenger trains"],
                "calc_names" : ["metrotram","train-conv","train-hs"]}
dict_iso2_jrc = jrc_iso2_dict()
dm_vkm_r = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# groupby trains
mapping_calc = {'rail': ['train-conv', 'train-hs']}
dm_vkm_r.groupby(mapping_calc, dim='Variables', aggregation = "sum", regex=False, inplace=True)

####################
##### AVIATION #####
####################

# note: for planes, occupancy will be pkm/skm, and skm = seats * vkm
# the reported unit will remain pkm/vkm also for planes even though it's pkm/skm
# for a question on how dms are structured for the model
# update: in the method above, we would have occupancy = (pkm/vkm)/seats
# in JRC, their occupancy ratio is (passenger/flights)/seats.
# so my occupancy goes also above 1, while their ratio stays below 1
# I will get theirs, which is more realistic
# TODO: discuss this with Paola.

# function
def get_specific_jrc_data(country_code, country_name, row_start, row_end, unit, variable = "aviation", 
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

# get data vkm
df_vkm_avi = pd.concat([get_specific_jrc_data(code, name, 22, 23, "mio vkm","vkm") for code,name in zip(country_codes, country_names)],ignore_index=True)
dm_vkm_avi = DataMatrix.create_from_df(df_vkm_avi, 0)
# dm_vkm_avi.filter({"Country": ["EU27"], "Variables" : ["vkm"]}).write_df()

# # get data vkm
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrAvia_act",
#                 "variable" : "Vehicle-km (mio km)",
#                 "sheet_last_row" : "Passenger transport",
#                 "sub_variables" : ["Passenger transport"],
#                 "calc_names" : ["vkm"]}
# dm_vkm_avi = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# # dm_vkm_avi.filter({"Country": ["EU27"]}).datamatrix_plot()

# get average number of seats per flight (so per plane)
df_seats_avi = pd.concat([get_specific_jrc_data(code, name, 273, 274, "number", "seats per plane") for code,name in zip(country_codes, country_names)],ignore_index=True)
dm_seats_avi = DataMatrix.create_from_df(df_seats_avi, 0)
# dm_seats_avi.filter({"Country": ["EU27"], "Variables" : ["seats per plane"]}).write_df()

# # get data seats
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrAvia_png",
#                 "variable" : "Seats available per flight",
#                 "sheet_last_row" : "Passenger transport",
#                 "sub_variables" : ["Passenger transport"],
#                 "calc_names" : ["seats"]}
# dm_seats_avi = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# dm_seats_avi.units["seats"] = "number"
# # dm_seats_avi.filter({"Country": ["EU27"]}).datamatrix_plot()
# # (dm_pkm_avi.flatten()["EU27",2021,:]*1e-6/dm_vkm_avi["EU27",2021,:]) / dm_seats_avi["EU27",2021,:]
# # dm_pkm_avi.flatten()["EU27",2021,:]*1e-6 / ( dm_vkm_avi["EU27",2021,:] * dm_seats_avi["EU27",2021,:])

# make million skm
dm_skm_avi = dm_vkm_avi.copy()
dm_seats_avi = dm_seats_avi.filter({"Years" : dm_skm_avi.col_labels["Years"]})
dm_skm_avi.append(dm_seats_avi,"Variables")
dm_skm_avi.operation("seats per plane", "*", "vkm","Variables","aviation","mio skm")
dm_skm_avi.filter({"Variables" : ["aviation"]},inplace=True)

# check
# df = dm_skm_avi.write_df()
# dm_skm_avi.filter({"Country": ["EU27"]}).datamatrix_plot()
# dm_skm_avi["EU27",2021,:]

########################
##### PUT TOGETHER #####
########################

dm_vkm = dm_vkm_ltb.copy()
dm_vkm.append(dm_vkm_r,"Variables")
dm_vkm.add(np.nan, "Years", list(range(1990,2000)), dummy=True)
dm_vkm.sort("Years")
dm_vkm.append(dm_skm_avi,"Variables")
dm_vkm.sort("Variables")
dm_vkm.sort("Country")

###################
##### FIX OTS #####
###################

# before 2000: do trend on 2000-2019, and for 2w do 2000-2016 (before the drop)
dm_2w = dm_vkm.filter({"Variables" : ["2W"]})
dm_vkm.drop("Variables",["2W"])
years_fitting = list(range(startyear,1999+1))
dm_vkm = linear_fitting(dm_vkm, years_fitting, based_on=list(range(2000,2019)))
dm_2w = linear_fitting(dm_2w, years_fitting, based_on=list(range(2000,2015)))
dm_vkm.append(dm_2w,"Variables")
dm_vkm.sort("Variables")

# check
# dm_vkm.filter({"Country" : ["EU27"]}).datamatrix_plot()

# for 2022-2023: do trend on 2000-2019 (when we have data pre covid)
dm_vkm = linear_fitting(dm_vkm , [2022,2023], based_on=list(range(2000,2019+1)))

# check
# dm_vkm.filter({"Country" : ["EU27"]}).datamatrix_plot()

####################
##### MAKE FTS #####
####################

# make function to fill in missing years fts for EU27 with linear fitting
def make_fts(dm, variable, year_start, year_end, country = "EU27", dim = "Categories1", 
             min_t0=0, min_tb=0, years_fts = years_fts): # I put minimum to 1 so it does not go to zero
    dm = dm.copy()
    idx = dm.idx
    based_on_yars = list(range(year_start, year_end + 1, 1))
    dm_temp = linear_fitting(dm.filter({"Country" : [country], dim : [variable]}), 
                             years_ots = years_fts, min_t0=min_t0, min_tb=min_tb, based_on = based_on_yars)
    idx_temp = dm_temp.idx
    if dim == "Variables":
        dm.array[idx[country],:,idx[variable],...] = \
            np.round(dm_temp.array[idx_temp[country],:,idx_temp[variable],...],0)
    if dim == "Categories1":
        dm.array[idx[country],:,:,idx[variable]] = \
            np.round(dm_temp.array[idx_temp[country],:,:,idx_temp[variable]], 0)
    if dim == "Categories2":
        dm.array[idx[country],:,:,:,idx[variable]] = \
            np.round(dm_temp.array[idx_temp[country],:,:,:,idx_temp[variable]], 0)
    if dim == "Categories3":
        dm.array[idx[country],:,:,:,:,idx[variable]] = \
            np.round(dm_temp.array[idx_temp[country],:,:,:,:,idx_temp[variable]], 0)
    
    return dm

# add missing years fts
dm_vkm.add(np.nan, col_label=years_fts, dummy=True, dim='Years')

# set default time window for linear trend
baseyear_start = 2000
baseyear_end = 2019

# # try fts
# product = "rail"
# (make_fts(dm_vkm, product, baseyear_start, baseyear_end, dim = "Variables").
#   datamatrix_plot(selected_cols={"Country" : ["EU27"], "Variables" : [product]}))

# make fts
dm_vkm = make_fts(dm_vkm, "2W", baseyear_start, baseyear_end, dim = "Variables")
dm_vkm = make_fts(dm_vkm, "LDV", baseyear_start, baseyear_end, dim = "Variables")
dm_vkm = make_fts(dm_vkm, "bus", baseyear_start, baseyear_end, dim = "Variables")
dm_vkm = make_fts(dm_vkm, "metrotram", baseyear_start, baseyear_end, dim = "Variables")
dm_vkm = make_fts(dm_vkm, "rail", baseyear_start, baseyear_end, dim = "Variables")
dm_vkm = make_fts(dm_vkm, "aviation", baseyear_start, baseyear_end, dim = "Variables")

# check
# dm_vkm.filter({"Country" : ["EU27"]}).datamatrix_plot()

####################################
##### MAKE AS FINAL DATAMATRIX #####
####################################

DM_tra["ots"]["passenger_occupancy"].units

# rename and deepen
for v in dm_vkm.col_labels["Variables"]:
    dm_vkm.rename_col(v,"tra_passenger_vkm_" + v, "Variables")
dm_vkm.deepen()

# get it in vkm
dm_vkm.change_unit("tra_passenger_vkm", 1e6, "mio km", "vkm")

# do pkm/vkm
dm_pkm = DM_pkm["ots"]["passenger_pkm"].copy()
dm_pkm.append(DM_pkm["fts"]["passenger_pkm"][1],"Years")
dm_pkm.sort("Years")
dm_pkm_avi.rename_col("tra","tra_passenger_pkm","Variables")
dm_pkm.append(dm_pkm_avi,"Categories1")
dm_pkm.sort("Categories1")
# dm_pkm["EU27",2021,:,"aviation"]*1e-6
# dm_vkm["EU27",2021,:,"aviation"]*1e-6
# dm_pkm.filter({"Country": ["EU27"], "Categories1": ["aviation"]}).write_df()
# dm_vkm.filter({"Country": ["EU27"], "Categories1": ["aviation"]}).write_df()
dm_occ = dm_pkm.copy()
dm_vkm.drop("Years",startyear)
dm_occ.array = dm_occ.array/dm_vkm.array
dm_occ.rename_col("tra_passenger_pkm","tra_passenger_occupancy","Variables")
dm_occ.units["tra_passenger_occupancy"] = "pkm/vkm"

# check
# dm_occ.filter({"Country" : ["EU27"]}).datamatrix_plot()
# dm_occ.filter({"Country" : ["EU27"]}).write_df().to_csv("/Users/echiarot/Desktop/occupancy.csv")
# dm_occ["EU27",2021,:,"aviation"]

# scale down aviation to get max to 1
for c in dm_occ.col_labels["Country"]:
    scale_factor = 1 - max(dm_occ[c,:,:,"aviation"])
    dm_occ[c,:,:,"aviation"] = dm_occ[c,:,:,"aviation"] + scale_factor

years_ots = list(range(1990,2023+1))
years_fts = list(range(2025,2055,5))

###############
##### OTS #####
###############

dm_occ_ots = dm_occ.filter({"Years" : years_ots})

#######################
##### FTS LEVEL 1 #####
#######################

# level 1: continuing as is
dm_occ_fts_level1 = dm_occ.filter({"Years" : years_fts})
# dm_occ.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

#######################
##### FTS LEVEL 4 #####
#######################

# for ldv source: page 56-57 https://www.itf-oecd.org/sites/default/files/docs/itf-transport-outlook-2023-launch.pdf
# rest is eucalc

dm_occ_level4 = dm_occ.copy()
idx = dm_occ_level4.idx
for y in range(2030,2055,5):
    dm_occ_level4.array[idx["EU27"],idx[y],:,:] = np.nan
dm_occ_level4.array[idx["EU27"],idx[2050],:,idx["LDV"]] = dm_occ_level4.array[idx["EU27"],idx[2023],:,idx["LDV"]]*(1+0.0815)
dm_occ_level4.array[idx["EU27"],idx[2050],:,idx["2W"]] = 1.3
dm_occ_level4.array[idx["EU27"],idx[2050],:,idx["bus"]] = 27.2
bus_2023 = dm_occ_level4.array[idx["EU27"],idx[2023],:,idx["bus"]]
bus_2050 = dm_occ_level4.array[idx["EU27"],idx[2050],:,idx["bus"]]
rate_bus = (bus_2050-bus_2023)/bus_2023
dm_occ_level4.array[idx["EU27"],idx[2050],:,idx["metrotram"]] = \
    dm_occ_level4.array[idx["EU27"],idx[2023],:,idx["metrotram"]] * (1+rate_bus)
dm_occ_level4.array[idx["EU27"],idx[2050],:,idx["rail"]] = \
    dm_occ_level4.array[idx["EU27"],idx[2023],:,idx["rail"]] * (1+rate_bus)
dm_occ_level4.array[idx["EU27"],idx[2050],:,idx["aviation"]] = 1 # max occupation in aviation is one
dm_occ_level4 = linear_fitting(dm_occ_level4, years_fts)
# dm_occ_level4.filter({"Country" : ["EU27"], "Categories1":["LDV"]}).flatten().datamatrix_plot()
dm_occ_fts_level4 = dm_occ_level4.filter({"Years" : years_fts})
# dm_occ_level4.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

# get levels for 2 and 3
variabs = dm_occ_fts_level1.col_labels["Categories1"]
df_temp = pd.DataFrame({"level" : np.tile(range(1,4+1),len(variabs)), 
                        "variable": np.repeat(variabs, 4)})
df_temp["value"] = np.nan
df_temp = df_temp.pivot(index=['level'], 
                        columns=['variable'], values="value").reset_index()
for v in variabs:
    idx = dm_occ_fts_level1.idx
    level1 = dm_occ_fts_level1.array[idx["EU27"],idx[2050],:,idx[v]][0]
    idx = dm_occ_fts_level4.idx
    level4 = dm_occ_fts_level4.array[idx["EU27"],idx[2050],:,idx[v]][0]
    arr = np.array([level1,np.nan,np.nan,level4])
    df_temp[v] = pd.Series(arr).interpolate().to_numpy()

#######################
##### FTS LEVEL 2 #####
#######################

dm_occ_level2 = dm_occ.copy()
idx = dm_occ_level2.idx
for y in range(2030,2055,5):
    dm_occ_level2.array[idx["EU27"],idx[y],:,:] = np.nan
for v in variabs:
    dm_occ_level2.array[idx["EU27"],idx[2050],:,idx[v]] = df_temp.loc[df_temp["level"] == 2,v]
dm_occ_level2 = linear_fitting(dm_occ_level2, years_fts)
# dm_occ_level2.filter({"Country" : ["EU27"], "Categories1":["LDV"]}).flatten().datamatrix_plot()
dm_occ_fts_level2 = dm_occ_level2.filter({"Years" : years_fts})
# dm_occ_fts_level2.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

#######################
##### FTS LEVEL 3 #####
#######################

dm_occ_level3 = dm_occ.copy()
idx = dm_occ_level3.idx
for y in range(2030,2055,5):
    dm_occ_level3.array[idx["EU27"],idx[y],:,:] = np.nan
for v in variabs:
    dm_occ_level3.array[idx["EU27"],idx[2050],:,idx[v]] = df_temp.loc[df_temp["level"] == 3,v]
dm_occ_level3 = linear_fitting(dm_occ_level3, years_fts)
# dm_occ_level3.filter({"Country" : ["EU27"], "Categories1":["LDV"]}).flatten().datamatrix_plot()
dm_occ_fts_level3 = dm_occ_level3.filter({"Years" : years_fts})
# dm_occ_fts_level3.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

################
##### SAVE #####
################

# split between ots and fts
DM_occ = {"ots": {"passenger_occupancy" : []}, "fts": {"passenger_occupancy" : dict()}}
DM_occ["ots"]["passenger_occupancy"] = dm_occ_ots.copy()
DM_occ["fts"]["passenger_occupancy"][1] = dm_occ_fts_level1.copy()
DM_occ["fts"]["passenger_occupancy"][2] = dm_occ_fts_level2.copy()
DM_occ["fts"]["passenger_occupancy"][3] = dm_occ_fts_level3.copy()
DM_occ["fts"]["passenger_occupancy"][4] = dm_occ_fts_level4.copy()

# save
f = os.path.join(current_file_directory, '../data/datamatrix/lever_passenger_occupancy.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM_occ, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # split between ots and fts
# DM_vkm = {"ots": {"passenger_vkm" : []}, "fts": {"passenger_vkm" : dict()}}
# DM_vkm ["ots"]["passenger_vkm"] = dm_vkm.filter({"Years" : list(range(1990,baseyear+1))})
# for i in range(1,4+1):
#     DM_vkm["fts"]["passenger_vkm"][i] = dm_vkm.filter({"Years" : years_fts})
    
# save
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/passenger_vkm.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_vkm, handle, protocol=pickle.HIGHEST_PROTOCOL)

















