
# packages
from model.common.data_matrix_class import DataMatrix

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
import subprocess

# file
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/minerals/eu/python/minerals_build-pickle.py"

# directories
current_file_directory = os.path.dirname(os.path.abspath(__file__))

###############################################################################
############################### EXECUTE SCRIPTS ###############################
###############################################################################

# subprocess.run(['python', os.path.join(current_file_directory, 'minerals_lever_material-recovery.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'minerals_const_material-decomposition.py')])

###############################################################################
################################ BUILD PICKLE #################################
###############################################################################

# files
files_directory = os.path.join(current_file_directory, '../data/datamatrix')
files = os.listdir(files_directory)

# create DM_minerals
DM_ots = {}
DM_fts = {}
# DM_fxa = {}
# DM_cal = {}
CDM_const = {}
DM_minerals = {}

##################
##### LEVERS #####
##################

# list(np.array(files)[[bool(re.search("lever", i)) for i in files]])
lever_files = ['lever_material-recovery.pickle']
lever_names = ['eol-material-recovery']

# load dms
for i in range(0, len(lever_files)):
    filepath = os.path.join(current_file_directory, '../data/datamatrix/' + lever_files[i])
    with open(filepath, 'rb') as handle:
        DM = pickle.load(handle)
    DM_ots[lever_names[i]] = DM["ots"]
    DM_fts[lever_names[i]] = DM["fts"]

# drop ammonia
lever_names = ['eol-material-recovery']
for n in lever_names:
    DM_ots[n].drop("Categories1","ammonia")
    for i in range(1,4+1):
        DM_fts[n][i].drop("Categories1","ammonia")

#####################
##### CONSTANTS #####
#####################

# material decomposition
filepath = os.path.join(current_file_directory, '../data/datamatrix/' + 'const_material-decomposition.pickle')
with open(filepath, 'rb') as handle:
    CDM = pickle.load(handle)

# CDM_const["material-decomposition_pipe"] = CDM["bld_pipe"] # do not load pipes for dh for now as this needs to be implemented in buildings
CDM_const["material-decomposition_floor"] = CDM["bld_floor"].filter({"Categories1" : ['floor-area-new-residential', 'floor-area-reno-residential']}) # keep only residential for now as non residential need to be implemented in buildings
# CDM_const["material-decomposition_domapp"] = CDM["bld_domapp"] # do not load domestic appliances for now as this needs to be implemented in buildings
CDM_const["material-decomposition_infra"] = CDM["tra_infra"]
CDM_const["material-decomposition_veh"] = CDM["tra_veh"]
CDM_const["material-decomposition_bat"] = CDM["tra_bat"]
CDM_const["material-decomposition_pack"] = CDM["pack"]

# drop ammonia
lever_names = ['material-decomposition_floor', 'material-decomposition_infra',
               'material-decomposition_pack']
for n in lever_names:
    CDM_const[n].drop("Categories2","ammonia")
CDM_const['material-decomposition_veh'].drop("Categories3","ammonia")
CDM_const['material-decomposition_bat'].drop("Categories3","ammonia")

########################
##### PUT TOGETHER #####
########################

DM_minerals = {
    'fts': DM_fts,
    'ots': DM_ots,
    "constant" : CDM_const
}

##########################
##### KEEP ONLY EU27 #####
##########################

for key in DM_minerals["ots"].keys():
    DM_minerals["ots"][key].filter({"Country" : ["EU27"]},inplace=True)
for key in DM_minerals["fts"].keys():
    for level in list(range(1,4+1)):
        DM_minerals["fts"][key][level].filter({"Country" : ["EU27"]},inplace=True)


#######################################
###### GENERATE FAKE SWITZERLAND ######
#######################################

for key in ['ots']:
    dm_names = list(DM_minerals[key])
    for name in dm_names:
        dm_temp = DM_minerals[key][name]
        if "Switzerland" not in dm_temp.col_labels["Country"]:
            idx = dm_temp.idx
            arr_temp = dm_temp.array[idx["EU27"],...]
            dm_temp.add(arr_temp[np.newaxis,...], "Country", "Switzerland")
            dm_temp.sort("Country")


dm_names = list(DM_minerals["fts"])
for name in dm_names:
    for i in range(1,4+1):
        dm_temp = DM_minerals["fts"][name][i]
        if "Switzerland" not in dm_temp.col_labels["Country"]:
            idx = dm_temp.idx
            arr_temp = dm_temp.array[idx["EU27"],...]
            dm_temp.add(arr_temp[np.newaxis,...], "Country", "Switzerland")
            dm_temp.sort("Country")
            
################################
###### GENERATE FAKE VAUD ######
################################

for key in ['ots']:
    dm_names = list(DM_minerals[key])
    for name in dm_names:
        dm_temp = DM_minerals[key][name]
        if "Vaud" not in dm_temp.col_labels["Country"]:
            idx = dm_temp.idx
            arr_temp = dm_temp.array[idx["EU27"],...]
            dm_temp.add(arr_temp[np.newaxis,...], "Country", "Vaud")
            dm_temp.sort("Country")


dm_names = list(DM_minerals["fts"])
for name in dm_names:
    for i in range(1,4+1):
        dm_temp = DM_minerals["fts"][name][i]
        if "Vaud" not in dm_temp.col_labels["Country"]:
            idx = dm_temp.idx
            arr_temp = dm_temp.array[idx["EU27"],...]
            dm_temp.add(arr_temp[np.newaxis,...], "Country", "Vaud")
            dm_temp.sort("Country")

################
##### SAVE #####
################

# save
f = os.path.join(current_file_directory, '../../../../data/datamatrix/minerals_new.pickle')
my_pickle_dump(DM_minerals, f)
