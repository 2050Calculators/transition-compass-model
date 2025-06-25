
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
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/transport/EU/python/transport_lever_passenger_technology-share_new.py"

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

# check
list(DM_tra["ots"])
DM_tra["ots"]["passenger_technology-share_new"]

# get new registration data
filepath = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/passenger_new-vehicles.pickle')
with open(filepath, 'rb') as handle:
    dm_new = pickle.load(handle)

# normalise
for v in dm_new.col_labels["Variables"]:
    dm_new.rename_col(v,"tra_passenger_technology-share_new_" + v,"Variables")
dm_new.deepen_twice()
dm_new.sort("Categories2")
dm_new.normalise("Categories2")

# check
# dm_new.filter({"Country" : ["EU27"]}).flatten().flatten().datamatrix_plot()

###############
##### OTS #####
###############

dm_new_ots = dm_new.filter({"Years" : years_ots})

#######################
##### FTS LEVEL 1 #####
#######################

# level 1: continuing as is
dm_new_fts_level1 = dm_new.filter({"Years" : years_fts}).flatten()

#######################
##### FTS LEVEL 4 #####
#######################

# make level as per green deal, i.e. no more emission vehicles after 2035
dm_new_level4 = dm_new.copy()
dm_new_level4  = dm_new_level4.flatten()
idx = dm_new_level4.idx
for y in list(range(2030,2055,5)):
    dm_new_level4.array[idx["EU27"],idx[y],:,:] = np.nan
for y in range(2035,2055,5):
    dm_new_level4.array[idx["EU27"],idx[y],:,idx["LDV_ICE-gasoline"]] = 0
    dm_new_level4.array[idx["EU27"],idx[y],:,idx["LDV_ICE-gas"]] = 0
    dm_new_level4.array[idx["EU27"],idx[y],:,idx["LDV_ICE-diesel"]] = 0
    dm_new_level4.array[idx["EU27"],idx[y],:,idx["LDV_PHEV-gasoline"]] = 0
    dm_new_level4.array[idx["EU27"],idx[y],:,idx["LDV_PHEV-diesel"]] = 0
    dm_new_level4.array[idx["EU27"],idx[y],:,idx["LDV_FCEV"]] = 0.1
    dm_new_level4.array[idx["EU27"],idx[y],:,idx["LDV_BEV"]] = 0.9
    dm_new_level4.array[idx["EU27"],idx[y],:,idx["2W_ICE-gasoline"]] = 1
    dm_new_level4.array[idx["EU27"],idx[y],:,idx["bus_ICE-gasoline"]] = 0
    dm_new_level4.array[idx["EU27"],idx[y],:,idx["bus_ICE-gas"]] = 0
    dm_new_level4.array[idx["EU27"],idx[y],:,idx["bus_ICE-diesel"]] = 0
    dm_new_level4.array[idx["EU27"],idx[y],:,idx["bus_BEV"]] = 1
    dm_new_level4.array[idx["EU27"],idx[y],:,idx["metrotram_mt"]] = 1
    dm_new_level4.array[idx["EU27"],idx[y],:,idx["rail_ICE-diesel"]] = 0
    dm_new_level4.array[idx["EU27"],idx[y],:,idx["rail_CEV"]] = 1
dm_new_level4 = linear_fitting(dm_new_level4, years_fts)
# dm_new_level4.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()
dm_new_fts_level4 = dm_new_level4.filter({"Years" : years_fts})
# dm_new_fts_level4.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot(stacked=True)

# get levels for 2 and 3
variabs = dm_new_fts_level1.col_labels["Categories1"]
df_temp = pd.DataFrame({"level" : np.tile(range(1,4+1),len(variabs)), 
                        "variable": np.repeat(variabs, 4)})
df_temp["value"] = np.nan
df_temp = df_temp.pivot(index=['level'], 
                        columns=['variable'], values="value").reset_index()
for v in variabs:
    idx = dm_new_fts_level1.idx
    level1 = dm_new_fts_level1.array[idx["EU27"],idx[2050],:,idx[v]][0]
    idx = dm_new_fts_level4.idx
    level4 = dm_new_fts_level4.array[idx["EU27"],idx[2050],:,idx[v]][0]
    arr = np.array([level1,np.nan,np.nan,level4])
    df_temp[v] = pd.Series(arr).interpolate().to_numpy()

#######################
##### FTS LEVEL 2 #####
#######################

dm_new_level2 = dm_new.flatten()
idx = dm_new_level2.idx
for y in range(2030,2055,5):
    dm_new_level2.array[idx["EU27"],idx[y],:,:] = np.nan
for v in variabs:
    dm_new_level2.array[idx["EU27"],idx[2050],:,idx[v]] = df_temp.loc[df_temp["level"] == 2,v]
dm_new_level2 = linear_fitting(dm_new_level2, years_fts)
# dm_new_level2.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()
dm_new_fts_level2 = dm_new_level2.filter({"Years" : years_fts})
# dm_new_fts_level2.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

#######################
##### FTS LEVEL 3 #####
#######################

dm_new_level3 = dm_new.flatten()
idx = dm_new_level3.idx
for y in range(2030,2055,5):
    dm_new_level3.array[idx["EU27"],idx[y],:,:] = np.nan
for v in variabs:
    dm_new_level3.array[idx["EU27"],idx[2050],:,idx[v]] = df_temp.loc[df_temp["level"] == 3,v]
dm_new_level3 = linear_fitting(dm_new_level3, years_fts)
# dm_new_level3.filter({"Country" : ["EU27"], "Categories1":["LDV"]}).flatten().datamatrix_plot()
dm_new_fts_level3 = dm_new_level3.filter({"Years" : years_fts})
# dm_new_fts_level3.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

################
##### SAVE #####
################

# deepen
dm_new_fts_level1.deepen()
# dm_new_fts_level1.filter({"Country" : ["EU27"], "Categories1":["LDV"]}).flatten().datamatrix_plot(stacked=True)
dm_new_fts_level2.deepen()
# dm_new_fts_level2.filter({"Country" : ["EU27"], "Categories1":["LDV"]}).flatten().datamatrix_plot(stacked=True)
dm_new_fts_level3.deepen()
# dm_new_fts_level3.filter({"Country" : ["EU27"], "Categories1":["LDV"]}).flatten().datamatrix_plot(stacked=True)
dm_new_fts_level4.deepen()
# dm_new_fts_level4.filter({"Country" : ["EU27"], "Categories1":["LDV"]}).flatten().datamatrix_plot(stacked=True)

# split between ots and fts
DM_new = {"ots": {"passenger_technology-share_new" : []}, "fts": {"passenger_technology-share_new" : dict()}}
DM_new["ots"]["passenger_technology-share_new"] = dm_new_ots.copy()
DM_new["fts"]["passenger_technology-share_new"][1] = dm_new_fts_level1.copy()
DM_new["fts"]["passenger_technology-share_new"][2] = dm_new_fts_level2.copy()
DM_new["fts"]["passenger_technology-share_new"][3] = dm_new_fts_level3.copy()
DM_new["fts"]["passenger_technology-share_new"][4] = dm_new_fts_level4.copy()

# save
f = os.path.join(current_file_directory, '../data/datamatrix/lever_passenger_technology-share_new.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM_new, handle, protocol=pickle.HIGHEST_PROTOCOL)









