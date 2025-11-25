
# packages

from model.common.auxiliary_functions import linear_fitting
from _database.pre_processing.routine_JRC import get_jrc_data
from model.common.auxiliary_functions import eurostat_iso2_dict, jrc_iso2_dict
from model.common.data_matrix_class import DataMatrix

import pickle
import os
import numpy as np
import warnings
import pandas as pd
warnings.simplefilter("ignore")
import plotly.io as pio
pio.renderers.default='browser'

# directories
current_file_directory = os.getcwd()

# load current transport pickle
filepath = os.path.join(current_file_directory, '../../../../data/datamatrix/transport.pickle')
with open(filepath, 'rb') as handle:
    DM_tra = pickle.load(handle)
    
# load vkm (which is seat km for aviation)
filepath = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/passenger_vkm.pickle')
with open(filepath, 'rb') as handle:
    dm_avi_seatkm = pickle.load(handle)
    dm_avi_seatkm = dm_avi_seatkm.filter({"Categories1" : ["aviation"]})

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
                "sheet" : "TrRoad_tech",
                "variable" : "Test cycle efficiency of new vehicles (kgoe/100 km)",
                "sheet_last_row" : "Powered two-wheelers",
                "sub_variables" : ["Powered two-wheelers"],
                "calc_names" : ["2W"]}
dm_eneff_new_2w = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# make other variables
# assumption: 2w are mostly ICE-gasoline, so the energy efficiency we have will be assigned
# to 2W_ICE-gasoline, and the rest will be missing values for now
dm_eneff_new_2w.rename_col("2W","2W_ICE-gasoline","Variables")
dm_eneff_new_2w.deepen()
categories2_missing = categories2_all.copy()
for cat in dm_eneff_new_2w.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_eneff_new_2w.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_eneff_new_2w.sort("Categories1")


###############
##### LDV #####
###############


# get data
dict_extract = {"database" : "Transport",
                "sheet" : "TrRoad_tech",
                "variable" : "Test cycle efficiency of new vehicles (kgoe/100 km)",
                "sheet_last_row" : "Battery electric vehicles",
                "categories" : "Passenger cars",
                "sub_variables" : ["Gasoline engine",
                                    "Diesel oil engine",
                                    "LPG engine", "Natural gas engine", 
                                    "Plug-in hybrid electric",
                                    "Battery electric vehicles"],
                "calc_names" : ["LDV_ICE-gasoline","LDV_ICE-diesel","LDV_gas-lpg",
                                "LDV_gas-natural","LDV_PHEV","LDV_BEV"]}
dm_eneff_new_ldv = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# deepen and sum gas
dm_eneff_new_ldv.deepen()
mapping_calc = {'ICE-gas': ['gas-lpg', 'gas-natural']}
dm_eneff_new_ldv.groupby(mapping_calc, dim='Categories1', aggregation = "mean", regex=False, inplace=True)

# make PHEV diesel and PHEV gasoline
# assumption: they have the same efficiency
dm_temp = dm_eneff_new_ldv.filter({"Categories1" : ["PHEV"]})
dm_temp1 = dm_temp.copy()
dm_temp1.rename_col("PHEV","PHEV-gasoline","Categories1")
dm_temp1.append(dm_temp,"Categories1")
dm_temp1.rename_col("PHEV","PHEV-diesel","Categories1")
dm_eneff_new_ldv.drop("Categories1","PHEV")
dm_eneff_new_ldv.append(dm_temp1,"Categories1")
dm_eneff_new_ldv.sort("Categories1")

# make other variables
# TODO: FCEV energy efficiency can be same of Switzerland, to be added at the end, for now missing
# assumption: for now, rest is assumed to be missing
categories2_missing = categories2_all.copy()
for cat in dm_eneff_new_ldv.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_eneff_new_ldv.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_eneff_new_ldv.sort("Categories1")


#################
##### BUSES #####
#################

# get data
dict_extract = {"database" : "Transport",
                "sheet" : "TrRoad_tech",
                "variable" : "Test cycle efficiency of new vehicles (kgoe/100 km)",
                "sheet_last_row" : "Battery electric vehicles",
                "sub_variables" : ["Gasoline engine",
                                    "Diesel oil engine",
                                    "LPG engine", "Natural gas engine", 
                                    "Battery electric vehicles"],
                "calc_names" : ["bus_ICE-gasoline","bus_ICE-diesel","bus_gas-lpg",
                                "bus_gas-natural","bus_BEV"],
                "categories" : "Motor coaches, buses and trolley buses"}
dm_eneff_new_bus = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# deepen and sum gas
dm_eneff_new_bus.deepen()
mapping_calc = {'ICE-gas': ['gas-lpg', 'gas-natural']}
dm_eneff_new_bus.groupby(mapping_calc, dim='Categories1', aggregation = "mean", regex=False, inplace=True)

# make rest of the variables (assuming they are all missing for now)
categories2_missing = categories2_all.copy()
for cat in dm_eneff_new_bus.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_eneff_new_bus.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_eneff_new_bus.sort("Categories1")

################
##### RAIL #####
################

# get data on efficiency of the stock of trains
# note: I assume that all high speed passenger trains are electric, and I'll take an average

# get data
dict_extract = {"database" : "Transport",
                "sheet" : "TrRail_ene",
                "variable" : "Vehicle-efficiency (kgoe/100 km)",
                "sheet_last_row" : "High speed passenger trains",
                "sub_variables" : ["Metro and tram, urban light rail",
                                   "Diesel oil","Electric",
                                    "High speed passenger trains"],
                "calc_names" : ["metrotram_mt","train-conv_ICE-diesel","train-conv_CEV","train-hs_CEV"]}
dict_iso2_jrc = jrc_iso2_dict()
dm_eneff_rail = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# aggregate trains and deepen
mapping_calc = {'rail_CEV': ['train-conv_CEV', 'train-hs_CEV'],
                'rail_ICE-diesel' : ["train-conv_ICE-diesel"]}
dm_eneff_rail.groupby(mapping_calc, dim='Variables', aggregation = "mean", regex=False, inplace=True)

# load data of energy efficiency for stock of buses and get ratios, and apply them to trains
dict_extract = {"database" : "Transport",
                "sheet" : "TrRoad_ene",
                "variable" : "Vehicle-efficiency - effective (kgoe/100 km)",
                "sheet_last_row" : "Motor coaches, buses and trolley buses",
                "sub_variables" : ["Motor coaches, buses and trolley buses"],
                "calc_names" : ["bus"]}
dict_iso2_jrc = jrc_iso2_dict()
dm_eneff_bus = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
dm_eneff_bus.array[dm_eneff_bus.array==0] = np.nan
dm_new = dm_eneff_new_bus.copy()
dm_new.group_all(dim='Categories1', aggregation = "mean")
arr_temp = dm_new.array / dm_eneff_bus.array
dm_eneff_rail.array = dm_eneff_rail.array * arr_temp
dm_eneff_new_rail = dm_eneff_rail.copy()

# make rest of the variables (assuming they are all missing for now)
dm_eneff_new_rail.deepen()
categories2_missing = categories2_all.copy()
for cat in dm_eneff_new_rail.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_eneff_new_rail.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_eneff_new_rail.sort("Categories1")

####################
##### AVIATION #####
####################

# NOTE: here I do not have the energy efficiency of new planes, but rather the energy efficiency
# of the fleet (effective). In addition, I need to calculate efficiency, as I need to do it wrt
# seat km. So basically I will assume that this "computed efficiency of the seats"
# is the same of the one of the seats in new planes.

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

# get efficiency
df_eneff_new_avi = pd.concat([get_specific_jrc_data(code, name, 134, 135, "kgoe/100km", "efficiency") for code,name in zip(country_codes, country_names)],ignore_index=True)
dm_eneff_new_avi  = DataMatrix.create_from_df(df_eneff_new_avi , 0)
# dm_eneff_new_avi.change_unit("efficiency", 1e-2, "kgoe/100km", "kgoe/vkm")

# get number of seat per flight
df_seatsperflight = pd.concat([get_specific_jrc_data(code, name, 273, 274, "number", "nseats") for code,name in zip(country_codes, country_names)],ignore_index=True)
dm_seatsperflight_avi  = DataMatrix.create_from_df(df_seatsperflight , 0)

# get efficiency per seat km
dm_eneff_new_avi.append(dm_seatsperflight_avi, "Variables")
dm_eneff_new_avi.operation("efficiency", "/", "nseats", out_col="aviation_kerosene", unit="kgoe/100 km")
dm_eneff_new_avi = dm_eneff_new_avi.filter({"Variables" : ["aviation_kerosene"]})

# df_totener_new_avi = pd.concat([get_specific_jrc_data(code, name, 133, 134, "kgoe") for code,name in zip(country_codes, country_names)],ignore_index=True)
# dm_totener_new_avi  = DataMatrix.create_from_df(df_totener_new_avi , 0)

# # get energy efficiency = total energy consumption / 100 skm
# dm_avi_seatkm = dm_avi_seatkm.flatten()
# dm_avi_seatkm.change_unit("tra_passenger_vkm_aviation", 1e2, "vkm", "100 km")
# dm_avi_seatkm = dm_avi_seatkm.filter({"Years" : dm_totener_new_avi.col_labels["Years"]})
# dm_totener_new_avi.append(dm_avi_seatkm,"Variables")
# dm_totener_new_avi.rename_col("aviation_kerosene", "energy", "Variables")
# dm_totener_new_avi.operation("energy", "/", "tra_passenger_vkm_aviation", "Variables", "aviation_kerosene", "kgoe/100 km")
# dm_totener_new_avi = dm_totener_new_avi.filter({"Variables" : ["aviation_kerosene"]})
# dm_eneff_new_avi = dm_totener_new_avi.copy()

# # note: here we do not have test efficiency of new vehicles, but we have the theoretical one,
# # which is lower than the effective (as the test efficiency is lower than effective)
# # so I will take the theoretical, and assume that it proxies the efficiency of new planes

# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrAvia_ene",
#                 "variable" : "Vehicle-efficiency - theoretical (kgoe/100 km)*",
#                 "sheet_last_row" : "Passenger transport",                
#                 "sub_variables" : ["Passenger transport"],
#                 "calc_names" : ["aviation"]}
# dm_eneff_new_avi = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# # add techs (all kerosene)
# dm_eneff_new_avi.rename_col("aviation", "aviation_kerosene", "Variables")
dm_eneff_new_avi.deepen()
categories2_missing = categories2_all.copy()
for cat in dm_eneff_new_avi.col_labels["Categories1"]: categories2_missing.remove(cat)
dm_eneff_new_avi.add(np.nan, col_label=categories2_missing, dummy=True, dim="Categories1")
dm_eneff_new_avi.sort("Categories1")


########################
##### PUT TOGETHER #####
########################

dm_eneff_new = dm_eneff_new_2w.copy()
dm_eneff_new.append(dm_eneff_new_ldv,"Variables")
dm_eneff_new.append(dm_eneff_new_bus,"Variables")
dm_eneff_new.append(dm_eneff_new_rail,"Variables")
dm_eneff_new.sort("Variables")
dm_eneff_new.sort("Country")

# add aviation
dm_eneff_new.add(np.nan, "Years", list(range(1990,2000)), dummy=True)
dm_eneff_new.append(dm_eneff_new_avi,"Variables")
dm_eneff_new.sort("Variables")

# substitute zero values with missing
dm_eneff_new.array[dm_eneff_new.array==0] = np.nan

# check
# dm_eneff_new.flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()

###################
##### FIX OTS #####
###################

dm_eneff_new = dm_eneff_new.flatten()

# put nan for 2000 for bus_ICE-gas, metrotram_mt, rail_CEV, rail_ICE-diesel
idx = dm_eneff_new.idx
for v in ["bus_ICE-gas", "metrotram_mt", "rail_CEV", "rail_ICE-diesel"]:
    dm_eneff_new.array[:,idx[2000],idx[v]] = np.nan

# do linear fitting of each variable with wathever is available
dm_eneff_new = linear_fitting(dm_eneff_new, years_ots, min_t0=0,min_tb=0)

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
baseyear_end = 2019

# check
# dm_eneff_new.filter({"Country" : ["EU27"]}).datamatrix_plot()

# # try fts
# product = "LDV_BEV"
# (make_fts(dm_eneff_new, product, baseyear_start, baseyear_end, dim = "Variables").
#  datamatrix_plot(selected_cols={"Country" : ["EU27"], "Variables" : [product]}))

# make fts
dm_eneff_new = make_fts(dm_eneff_new, "2W_ICE-gasoline", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff_new = make_fts(dm_eneff_new, "LDV_BEV", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff_new = make_fts(dm_eneff_new, "LDV_ICE-diesel", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff_new = make_fts(dm_eneff_new, "LDV_ICE-gas", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff_new = make_fts(dm_eneff_new, "LDV_ICE-gasoline", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff_new = make_fts(dm_eneff_new, "LDV_PHEV-diesel", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff_new = make_fts(dm_eneff_new, "LDV_PHEV-gasoline", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff_new = make_fts(dm_eneff_new, "bus_BEV", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff_new = make_fts(dm_eneff_new, "bus_ICE-diesel", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff_new = make_fts(dm_eneff_new, "bus_ICE-gas", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff_new = make_fts(dm_eneff_new, "bus_ICE-gasoline", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff_new = make_fts(dm_eneff_new, "metrotram_mt", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff_new = make_fts(dm_eneff_new, "rail_CEV", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff_new = make_fts(dm_eneff_new, "rail_ICE-diesel", baseyear_start, baseyear_end, dim = "Variables")
dm_eneff_new = make_fts(dm_eneff_new, "aviation_kerosene", baseyear_start, baseyear_end, dim = "Variables",
                        min_t0=0, min_tb=0)

# check
# dm_eneff_new.filter({"Country" : ["EU27"]}).datamatrix_plot()

####################################
##### MAKE AS FINAL DATAMATRIX #####
####################################

DM_tra["ots"]["passenger_veh-efficiency_new"].units

# rename and deepen
for v in dm_eneff_new.col_labels["Variables"]:
    dm_eneff_new.rename_col(v,"tra_passenger_veh-efficiency_new_" + v, "Variables")
dm_eneff_new.deepen()
dm_eneff_new.deepen(based_on="Variables")
dm_eneff_new.switch_categories_order("Categories1","Categories2")

# get it in MJ over km
dm_eneff_new.change_unit("tra_passenger_veh-efficiency_new", 41.868, "kgoe/100 km", "MJ/100 km")
dm_eneff_new.change_unit("tra_passenger_veh-efficiency_new", 1e-2, "MJ/100 km", "MJ/km")

# add fcev
dm_eneff_new.add(np.nan, col_label="FCEV", dummy=True, dim="Categories2")
dm_eneff_new.sort("Categories2")

# check
# dm_eneff_new.flatten().flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()
# dm_eneff_new.filter({"Country" : ["EU27"], "Categories1" : ["aviation"], "Categories2" : ["kerosene"]}).write_df()
# these values are too small, probably same thing of before, you need to get average number of seat per flight etc

# add back h2
dm_eneff_new.add(np.nan, "Categories2", "H2", 'MJ/km', True)
dm_eneff_new.sort("Categories2")

################
##### SAVE #####
################

# split between ots and fts
DM_ene = {"ots": {"passenger_veh-efficiency_new" : []}, "fts": {"passenger_veh-efficiency_new" : dict()}}
DM_ene["ots"]["passenger_veh-efficiency_new"] = dm_eneff_new.filter({"Years" : years_ots})
DM_ene["ots"]["passenger_veh-efficiency_new"].drop("Years",1989)
for i in range(1,4+1):
    DM_ene["fts"]["passenger_veh-efficiency_new"][i] = dm_eneff_new.filter({"Years" : years_fts})

# save
f = os.path.join(current_file_directory, '../data/datamatrix/lever_passenger_veh-efficiency_new.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM_ene, handle, protocol=pickle.HIGHEST_PROTOCOL)




