
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
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/transport/EU/python/transport_lever_freight_utilization-rate.py"

# directories
current_file_directory = os.path.dirname(os.path.abspath(__file__))

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

###############################################################################
##################################### VKM #####################################
###############################################################################

################################################
################### GET DATA ###################
################################################

# check
list(DM_tra["ots"])
DM_tra["ots"]["freight_utilization-rate"].units

# get iso codes
dict_iso2 = eurostat_iso2_dict()
dict_iso2.pop('CH')  # Remove Switzerland
dict_iso2_jrc = jrc_iso2_dict()

###############
##### HDV #####
###############

# get data
dict_extract = {"database" : "Transport",
                "sheet" : "TrRoad_act",
                "variable" : "Vehicle-km driven (mio km)",
                "sheet_last_row" : "Heavy goods vehicles",
                "sub_variables" : ["Light commercial vehicles",
                                    "Heavy goods vehicles"],
                "calc_names" : ["HDVL","HDVH"]}
dm_hdvl = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# make hdvm as average between hdvh and hdvl (as it's vkm, the medium ones should carry around the average)
dm_temp = dm_hdvl.groupby({"HDVM" : ["HDVL","HDVH"]}, 
                          dim='Variables', aggregation = "mean", regex=False, inplace=False)
dm_hdvl.append(dm_temp, "Variables")
dm_hdvl.sort("Variables")

# ###############
# ##### IWW #####
# ###############

# # get data on energy efficiency
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrNavi_act",
#                 "variable" : "Vehicle-km (Mkm)",
#                 "sheet_last_row" : "Inland waterways",
#                 "sub_variables" : ["Inland waterways"],
#                 "calc_names" : ["IWW"]}
# dm_iww = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# ####################
# ##### aviation #####
# ####################

# # get data
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrAvia_act",
#                 "variable" : "Vehicle-km (mio km)",
#                 "sheet_last_row" : "Freight transport",
#                 "sub_variables" : ["Freight transport"],
#                 "calc_names" : ["aviation"]}
# dm_avi = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# ####################
# ##### maritime #####
# ####################

# # get data on energy efficiency
# dict_extract = {"database" : "Transport",
#                 "sheet" : "MBunk_act",
#                 "variable" : "Vehicle-km (mio km)",
#                 "sheet_last_row" : "Intra-EEA",
#                 "sub_variables" : ["Intra-EEA"],
#                 "calc_names" : ["marine"]}
# dm_mar = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)


# ################
# ##### RAIL #####
# ################

# # get data
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrRail_act",
#                 "variable" : "Vehicle-km (mio km)",
#                 "sheet_last_row" : "Freight transport",
#                 "sub_variables" : ["Freight transport"],
#                 "calc_names" : ["rail"]}
# dm_vkm_rail = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

########################
##### PUT TOGETHER #####
########################

dm_vkm = dm_hdvl.copy()
dm_vkm.sort("Variables")
dm_vkm.sort("Country")
for v in dm_vkm.col_labels["Variables"]:
    dm_vkm.units[v] = "mio vkm"

# check
# dm_vkm.filter({"Country" : ["EU27"]}).datamatrix_plot()

###################
##### FIX OTS #####
###################

dm_vkm = linear_fitting(dm_vkm, years_ots)

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
baseyear_end = 2023

# # try fts
# product = "rail"
# (make_fts(dm_vkm, product, baseyear_start, baseyear_end, dim = "Variables").
#   datamatrix_plot(selected_cols={"Country" : ["EU27"], "Variables" : [product]}))

# make fts
dm_vkm = make_fts(dm_vkm, "HDVH", baseyear_start, baseyear_end, dim = "Variables")
dm_vkm = make_fts(dm_vkm, "HDVL", baseyear_start, baseyear_end, dim = "Variables")
dm_vkm = make_fts(dm_vkm, "HDVM", baseyear_start, baseyear_end, dim = "Variables")

# rename and deepen
for v in dm_vkm.col_labels["Variables"]:
    dm_vkm.rename_col(v,"vkm_" + v, "Variables")
dm_vkm.deepen()

# get it in vkm
dm_vkm.change_unit("vkm", 1e6, "mio vkm", "vkm")

# check
# dm_vkm.filter({"Country" : ["EU27"]}).datamatrix_plot()

###############################################################################
################################## LOAD FACTOR ################################
###############################################################################

DM_tra["ots"]["freight_utilization-rate"].units

# load tkm
filepath = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/freight_tkm.pickle')
with open(filepath, 'rb') as handle:
    DM_tkm = pickle.load(handle)
dm_tkm = DM_tkm["ots"]["freight_tkm"].filter({"Categories1" : ['HDVH', 'HDVL', 'HDVM']})
dm_tkm.append(DM_tkm["fts"]["freight_tkm"][1].filter({"Categories1" : ['HDVH', 'HDVL', 'HDVM']}),"Years")
dm_tkm.sort("Years")
dm_vkm.drop("Years",startyear)

# make tkm/vkm
dm_uti = dm_tkm.copy()
dm_uti.array = dm_uti.array / dm_vkm.array
dm_uti.units["tra_freight_tkm"] = "tkm/vkm"
dm_uti.rename_col("tra_freight_tkm","tra_freight_load-factor","Variables")

###############################################################################
############################### UTILIZATION RATE ##############################
###############################################################################

DM_tra["ots"]["freight_utilization-rate"].units

###############
##### HDV #####
###############

# get data
dict_extract = {"database" : "Transport",
                "sheet" : "TrRoad_act",
                "variable" : "Vehicle-km driven per vehicle annum (km/vehicle)",
                "sheet_last_row" : "Heavy goods vehicles",
                "sub_variables" : ["Light commercial vehicles",
                                    "Heavy goods vehicles"],
                "calc_names" : ["HDVL","HDVH"]}
dm_hdv = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# make hdvm as average between hdvh and hdvl (as it's vkm, the medium ones should carry around the average)
dm_temp = dm_hdv.groupby({"HDVM" : ["HDVL","HDVH"]}, 
                          dim='Variables', aggregation = "mean", regex=False, inplace=False)
dm_hdv.append(dm_temp, "Variables")
dm_hdv.sort("Variables")

###################
##### FIX OTS #####
###################

dm_hdv = linear_fitting(dm_hdv, years_ots)

# check
# dm_hdv.filter({"Country" : ["EU27"]}).datamatrix_plot()

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
dm_hdv.add(np.nan, col_label=years_fts, dummy=True, dim='Years')

# set default time window for linear trend
baseyear_start = 2000
baseyear_end = 2023

# make fts
dm_hdv = make_fts(dm_hdv, "HDVH", baseyear_start, baseyear_end, dim = "Variables")
dm_hdv = make_fts(dm_hdv, "HDVL", baseyear_start, baseyear_end, dim = "Variables")
dm_hdv = make_fts(dm_hdv, "HDVM", baseyear_start, baseyear_end, dim = "Variables")

########################
##### PUT TOGETHER #####
########################

# rename and deepen
for v in dm_hdv.col_labels["Variables"]:
    dm_hdv.rename_col(v,"tra_freight_utilisation-rate_" + v, "Variables")
dm_hdv.deepen()
dm_hdv.units["tra_freight_utilisation-rate"] = "vkm/year"
dm_hdv.drop("Years",startyear)

# check
# dm_hdv.filter({"Country" : ["EU27"]}).datamatrix_plot()

# put in uti
dm_uti.append(dm_hdv, "Variables")

# split ots and fts
DM_uti = {"ots": {"freight_utilization-rate" : []}, "fts": {"freight_utilization-rate" : dict()}}
DM_uti["ots"]["freight_utilization-rate"] = dm_uti.filter({"Years" : list(range(1990,baseyear+1))})
for i in range(1,4+1):
    DM_uti["fts"]["freight_utilization-rate"][i] = dm_uti.filter({"Years" : years_fts})

# save
f = os.path.join(current_file_directory, '../data/datamatrix/lever_freight_utilization-rate.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM_uti, handle, protocol=pickle.HIGHEST_PROTOCOL)


































