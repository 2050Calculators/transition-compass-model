
# packages
import pickle
import os
import numpy as np
import warnings
warnings.simplefilter("ignore")
import pandas as pd
from model.common.auxiliary_functions import linear_fitting

# directories
current_file_directory = os.getcwd()

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
DM_tra["ots"]["passenger_utilization-rate"].units

# get fleet data
filepath = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/passenger_fleet.pickle')
with open(filepath, 'rb') as handle:
    dm_fleet = pickle.load(handle)
dm_fleet.deepen()
dm_fleet.group_all("Categories1")
for v in dm_fleet.col_labels["Variables"]:
    dm_fleet.rename_col(v,"tra_fleet_" + v,"Variables")
dm_fleet.deepen()

# get vkm data
filepath = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/passenger_vkm.pickle')
with open(filepath, 'rb') as handle:
    dm_vkm = pickle.load(handle)
# dm_vkm = DM_vkm["ots"]["passenger_vkm"].copy()
# dm_vkm.append(DM_vkm["fts"]["passenger_vkm"][1].copy(),"Years")
# dm_vkm.sort("Years")

# obtain vkm/veh
dm_uti = dm_vkm.copy()
dm_uti.array = dm_uti.array / dm_fleet.array
dm_uti.units["tra_passenger_vkm"] = "vkm/veh"
dm_uti.rename_col("tra_passenger_vkm","tra_passenger_utilisation-rate","Variables")

# check
# dm_uti.filter({"Country" : ["EU27"]}).datamatrix_plot()
# dm_uti.filter({"Country" : ["EU27"]}).write_df().to_csv("/Users/echiarot/Desktop/check.csv")

###############
##### OTS #####
###############

dm_uti_ots = dm_uti.filter({"Years" : years_ots})

#######################
##### FTS LEVEL 1 #####
#######################

# level 1: continuing as is
dm_uti_fts_level1 = dm_uti.filter({"Years" : years_fts})

#######################
##### FTS LEVEL 4 #####
#######################

# make level 4 with levels in eucalc
dm_uti_level4 = dm_uti.copy()
years_fts = list(range(2025,2055,5))
idx = dm_uti_level4.idx
for y in years_fts:
    dm_uti_level4.array[idx["EU27"],idx[y],:,:] = np.nan
dm_uti_level4.array[idx["EU27"],idx[2050],:,idx["LDV"]] = dm_uti_level4.array[idx["EU27"],idx[2023],:,idx["LDV"]]*(1+0.30) # I do not put 900% here as it would blow all the results
dm_uti_level4.array[idx["EU27"],idx[2050],:,idx["2W"]] = dm_uti_level4.array[idx["EU27"],idx[2023],:,idx["2W"]]*(1+0.15)
dm_uti_level4.array[idx["EU27"],idx[2050],:,idx["bus"]] = dm_uti_level4.array[idx["EU27"],idx[2023],:,idx["bus"]]*(1+0.45)
dm_uti_level4.array[idx["EU27"],idx[2050],:,idx["aviation"]] = dm_uti_level4.array[idx["EU27"],idx[2023],:,idx["aviation"]]*(1+0.45) # TODO: temporary as bus, re do this later
dm_uti_level4.array[idx["EU27"],idx[2050],:,idx["rail"]] = dm_uti_level4.array[idx["EU27"],idx[2023],:,idx["rail"]]*(1+0.45)
dm_uti_level4.array[idx["EU27"],idx[2050],:,idx["metrotram"]] = dm_uti_level4.array[idx["EU27"],idx[2023],:,idx["metrotram"]]*(1+0.45)
dm_uti_level4 = linear_fitting(dm_uti_level4, years_fts)
# dm_uti_level4.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()
dm_uti_fts_level4 = dm_uti_level4.filter({"Years" : years_fts})
# dm_uti_fts_level4.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot(stacked=True)

# get levels for 2 and 3
variabs = dm_uti_fts_level1.col_labels["Categories1"]
df_temp = pd.DataFrame({"level" : np.tile(range(1,4+1),len(variabs)), 
                        "variable": np.repeat(variabs, 4)})
df_temp["value"] = np.nan
df_temp = df_temp.pivot(index=['level'], 
                        columns=['variable'], values="value").reset_index()
for v in variabs:
    idx = dm_uti_fts_level1.idx
    level1 = dm_uti_fts_level1.array[idx["EU27"],idx[2050],:,idx[v]][0]
    idx = dm_uti_fts_level4.idx
    level4 = dm_uti_fts_level4.array[idx["EU27"],idx[2050],:,idx[v]][0]
    arr = np.array([level1,np.nan,np.nan,level4])
    df_temp[v] = pd.Series(arr).interpolate().to_numpy()

#######################
##### FTS LEVEL 2 #####
#######################

dm_uti_level2 = dm_uti.copy()
idx = dm_uti_level2.idx
for y in range(2030,2055,5):
    dm_uti_level2.array[idx["EU27"],idx[y],:,:] = np.nan
for v in variabs:
    dm_uti_level2.array[idx["EU27"],idx[2050],:,idx[v]] = df_temp.loc[df_temp["level"] == 2,v]
dm_uti_level2 = linear_fitting(dm_uti_level2, years_fts)
# dm_uti_level2.filter({"Country" : ["EU27"], "Categories1":["LDV"]}).flatten().datamatrix_plot()
dm_uti_fts_level2 = dm_uti_level2.filter({"Years" : years_fts})
# dm_uti_fts_level2.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

#######################
##### FTS LEVEL 3 #####
#######################

dm_uti_level3 = dm_uti.copy()
idx = dm_uti_level3.idx
for y in range(2030,2055,5):
    dm_uti_level3.array[idx["EU27"],idx[y],:,:] = np.nan
for v in variabs:
    dm_uti_level3.array[idx["EU27"],idx[2050],:,idx[v]] = df_temp.loc[df_temp["level"] == 3,v]
dm_uti_level3 = linear_fitting(dm_uti_level3, years_fts)
# dm_uti_level3.filter({"Country" : ["EU27"], "Categories1":["LDV"]}).flatten().datamatrix_plot()
dm_uti_fts_level3 = dm_uti_level3.filter({"Years" : years_fts})
# dm_uti_fts_level3.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

################
##### SAVE #####
################

# split between ots and fts
DM_uti = {"ots": {"passenger_utilization-rate" : []}, "fts": {"passenger_utilization-rate" : dict()}}
DM_uti["ots"]["passenger_utilization-rate"] = dm_uti_ots.copy()
DM_uti["fts"]["passenger_utilization-rate"][1] = dm_uti_fts_level1.copy()
DM_uti["fts"]["passenger_utilization-rate"][2] = dm_uti_fts_level2.copy()
DM_uti["fts"]["passenger_utilization-rate"][3] = dm_uti_fts_level3.copy()
DM_uti["fts"]["passenger_utilization-rate"][4] = dm_uti_fts_level4.copy()

# save
f = os.path.join(current_file_directory, '../data/datamatrix/lever_passenger_utilization-rate.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM_uti, handle, protocol=pickle.HIGHEST_PROTOCOL)






