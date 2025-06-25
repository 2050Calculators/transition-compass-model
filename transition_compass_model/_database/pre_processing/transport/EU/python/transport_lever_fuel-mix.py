
# packages
from model.common.data_matrix_class import DataMatrix
from model.common.auxiliary_functions import linear_fitting
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
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/transport/EU/python/transport_fuel-mix.py"

# directories
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# load current transport pickle
filepath = os.path.join(current_file_directory, '../../../../data/datamatrix/transport.pickle')
with open(filepath, 'rb') as handle:
    DM = pickle.load(handle)

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

DM["ots"]["fuel-mix"].units
df = DM["ots"]["fuel-mix"].group_all("Categories1",inplace=False).write_df()

# get iso codes
dict_iso2 = eurostat_iso2_dict()
dict_iso2.pop('CH')  # Remove Switzerland
dict_iso2_jrc = jrc_iso2_dict()

################
##### road #####
################

# get data
dict_extract = {"database" : "Transport",
                "sheet" : "TrRoad_ene",
                "variable" : "by fuel (EUROSTAT DATA)",
                "sheet_last_row" : "Renewable energies and wastes",
                "sub_variables" : ["by fuel (EUROSTAT DATA)",
                                    "Renewable energies and wastes"],
                "calc_names" : ["total","biofuel"]}
dm_road = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# make the share
dm_road.operation("biofuel","/","total","Variables", out_col="biofuel-share",unit="%")
dm_road.drop("Variables",['biofuel', 'total'])
dm_road.rename_col("biofuel-share","biofuel_road","Variables")

###############
##### IWW #####
###############

# get data on energy efficiency
dict_extract = {"database" : "Transport",
                "sheet" : "TrNavi_ene",
                "variable" : "Energy consumption (ktoe)",
                "categories" : "Inland waterways",
                "sheet_last_row" : "Biogases",
                "sub_variables" : ["Inland waterways",
                                    "Blended biofuels",
                                    "Biogases"],
                "calc_names" : ["total","biofuel-blended","biofuel-biogas"]}
dm_iww = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# make the share
dm_iww.groupby({"biofuel" : ['biofuel-biogas', 'biofuel-blended']}, "Variables", "sum",inplace=True)
dm_iww.operation("biofuel","/","total","Variables", out_col="biofuel-share",unit="%")
dm_iww.drop("Variables",['biofuel', 'total'])
dm_iww.rename_col("biofuel-share","biofuel_IWW","Variables")

####################
##### aviation #####
####################

# get data on energy efficiency
dict_extract = {"database" : "Transport",
                "sheet" : "TrAvia_ene",
                "variable" : "Energy consumption (ktoe)",
                "sheet_last_row" : "Energy consumption (ktoe)",
                "sub_variables" : ["Energy consumption (ktoe)"],
                "calc_names" : ["total"]}
dm_avi_tot = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# source: https://alternative-fuels-observatory.ec.europa.eu/transport-mode/aviation/general-information-and-context
# 0.24 million tonnes
# 1 tonne of SAF is approximately 1.03 toe.
value = 0.24*1e6*1.03*1e-3
dm_avi = dm_iww.filter({"Variables" : ["biofuel_IWW"]})
dm_avi.rename_col("biofuel_IWW","biofuel","Variables")
idx = dm_avi.idx
dm_avi.array[dm_avi.array>0]=0
dm_avi.array[idx["EU27"],idx[2021],:] = value
dm_avi.units["biofuel"] = "ktoe"

# put together
dm_avi.append(dm_avi_tot,"Variables")

# make the share
dm_avi.operation("biofuel","/","total","Variables", out_col="biofuel-share",unit="%")
dm_avi.drop("Variables",['biofuel', 'total'])
dm_avi.rename_col("biofuel-share","biofuel_aviation","Variables")

####################
##### maritime #####
####################

# get data on energy efficiency
dict_extract = {"database" : "Transport",
                "sheet" : "MBunk_ene",
                "variable" : "Total energy consumption (ktoe)",
                "categories" : "Intra-EEA",
                "sheet_last_row" : "Biogases",
                "sub_variables" : ["Intra-EEA",
                                    "Blended biofuels",
                                    "Biogases"],
                "calc_names" : ["total","biofuel-blended","biofuel-biogas"]}
dm_mar = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# make the share
dm_mar.groupby({"biofuel" : ['biofuel-biogas', 'biofuel-blended']}, "Variables", "sum",inplace=True)
dm_mar.operation("biofuel","/","total","Variables", out_col="biofuel-share",unit="%")
dm_mar.drop("Variables",['biofuel', 'total'])
dm_mar.rename_col("biofuel-share","biofuel_marine","Variables")

################
##### RAIL #####
################

# get data
dict_extract = {"database" : "Transport",
                "sheet" : "TrRail_ene",
                "variable" : "Energy consumption (ktoe)",
                "sheet_last_row" : "blended liquid biofuels",
                "sub_variables" : ["by fuel",
                                    "blended liquid biofuels"],
                "calc_names" : ["total","biofuel"]}
dm_rail = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# make the share
dm_rail.operation("biofuel","/","total","Variables", out_col="biofuel-share",unit="%")
dm_rail.drop("Variables",['biofuel', 'total'])
dm_rail.rename_col("biofuel-share","biofuel_rail","Variables")

########################
##### PUT TOGETHER #####
########################

dm_biof = dm_road.copy()
dm_biof.append(dm_iww,"Variables")
dm_biof.append(dm_avi,"Variables")
dm_biof.append(dm_mar,"Variables")
dm_biof.append(dm_rail,"Variables")
dm_biof.deepen()
dm_biof.add(0, col_label="efuel", dummy=True, dim='Variables',unit="%") # assuming efuel at 0
dm_biof.sort("Variables")
dm_biof.sort("Country")

# check
# dm_biof.filter({"Country" : ["EU27"]}).datamatrix_plot()

# fix aviation with same trend of biofuel rail
dm_temp = dm_biof.filter({"Variables" : ["biofuel"],"Categories1" : ["rail"]})
dm_temp = dm_temp.flatten()
idx = dm_temp.idx
arr_temp = dm_temp.array[:,idx[2021],:]
dm_temp.array = dm_temp.array / arr_temp[:,np.newaxis,:]
dm_temp_avi = dm_biof.filter({"Variables" : ["biofuel"],"Categories1" : ["aviation"]}).flatten()
idx = dm_temp_avi.idx
arr_temp = dm_temp_avi.array[:,idx[2021],:]
dm_temp_avi.array = arr_temp[:,np.newaxis,:] * dm_temp.array
dm_temp_avi.array[np.isnan(dm_temp_avi.array)]=0
dm_biof = dm_biof.flatten()
dm_biof.drop("Variables","biofuel_aviation")
dm_biof.append(dm_temp_avi,"Variables")
dm_biof.deepen()

# check
# dm_biof.filter({"Country" : ["EU27"]}).datamatrix_plot()

###################
##### FIX OTS #####
###################

dm_iww = dm_biof.filter({"Categories1" : ["IWW"]})
dm_mar = dm_biof.filter({"Categories1" : ["marine"]})
dm_biof.drop("Categories1",["IWW","marine"])

dm_biof = linear_fitting(dm_biof, years_ots, min_t0=0,min_tb=0)
dm_mar = linear_fitting(dm_mar, list(range(startyear,1999+1)), min_t0=0,min_tb=0, based_on=[2000])
dm_mar = linear_fitting(dm_mar, list(range(2022,2023+1)), min_t0=0,min_tb=0, based_on=[2020])
dm_iww = linear_fitting(dm_iww, list(range(startyear,1999+1)), min_t0=0,min_tb=0, based_on=[2000])
dm_iww = linear_fitting(dm_iww, list(range(2022,2023+1)), min_t0=0,min_tb=0, based_on=[2019])
dm_biof.append(dm_mar,"Categories1")
dm_biof.append(dm_iww,"Categories1")
dm_biof.sort("Categories1")

# check
# dm_biof.filter({"Country" : ["EU27"]}).datamatrix_plot()

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
dm_biof.add(np.nan, col_label=years_fts, dummy=True, dim='Years')

# set default time window for linear trend
baseyear_start = 2000
baseyear_end = 2019

# # try fts
# product = "aviation"
# (make_fts(dm_biof, product, baseyear_start, baseyear_end, dim = "Categories1").
#   datamatrix_plot(selected_cols={"Country" : ["EU27"], "Categories1" : [product]}))

# make fts
dm_biof = make_fts(dm_biof, "IWW", 2021, 2022, dim = "Categories1")
dm_biof = make_fts(dm_biof, "aviation", baseyear_start, baseyear_end, dim = "Categories1")
dm_biof = make_fts(dm_biof, "marine", 2021, 2022, dim = "Categories1")
dm_biof = make_fts(dm_biof, "rail", baseyear_start, baseyear_end, dim = "Categories1")
dm_biof = make_fts(dm_biof, "road", baseyear_start, baseyear_end, dim = "Categories1")

# check
# dm_biof.filter({"Country" : ["EU27"]}).datamatrix_plot()

####################################
##### MAKE AS FINAL DATAMATRIX #####
####################################

DM["ots"]["fuel-mix"]

# rename and deepen
for v in dm_biof.col_labels["Variables"]:
    dm_biof.rename_col(v,"tra_fuel-mix_" + v, "Variables")
dm_biof.deepen(based_on="Variables")
dm_biof.switch_categories_order("Categories1","Categories2")

################
##### SAVE #####
################

# split between ots and fts
DM = {"ots": {"fuel-mix" : []}, "fts": {"fuel-mix" : dict()}}
DM["ots"]["fuel-mix"] = dm_biof.filter({"Years" : years_ots})
DM["ots"]["fuel-mix"].drop("Years",startyear)
for i in range(1,4+1):
    DM["fts"]["fuel-mix"][i] = dm_biof.filter({"Years" : years_fts})

# save
f = os.path.join(current_file_directory, '../data/datamatrix/lever_fuel-mix.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)

