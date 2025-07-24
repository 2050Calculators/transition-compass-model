
# packages
import pickle
import os
import numpy as np
import warnings
warnings.simplefilter("ignore")
from model.common.auxiliary_functions import linear_fitting
import pandas as pd

# directories
current_file_directory = os.getcwd()

# load current transport pickle
filepath = os.path.join(current_file_directory, '../../../../data/datamatrix/transport.pickle')
with open(filepath, 'rb') as handle:
    DM_tra = pickle.load(handle)
    
# load pkm pickle
filepath = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/passenger_pkm.pickle')
with open(filepath, 'rb') as handle:
    DM_pkm = pickle.load(handle)

# load population pickle
filepath = os.path.join(current_file_directory, '../../../../data/datamatrix/lifestyles.pickle')
with open(filepath, 'rb') as handle:
    DM_lfs = pickle.load(handle)

# Set years range
years_setting = [1990, 2023, 2050, 5]
startyear = years_setting[0]
baseyear = years_setting[1]
lastyear = years_setting[2]
step_fts = years_setting[3]
years_ots = list(range(startyear, baseyear+1, 1))
years_fts = list(range(baseyear+2, lastyear+1, step_fts))
years_all = years_ots + years_fts

# get total pkm
dm_pkmtot = DM_pkm["ots"]["passenger_pkm"].copy()
dm_pkmtot.append(DM_pkm["fts"]["passenger_pkm"][1],"Years")
# dm_pkmtot.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot(stacked=True)
dm_pkmtot.group_all("Categories1")
dm_pkmtot = dm_pkmtot.filter({"Country" : ["EU27"]})
# dm_pkmtot.filter({"Country" : ["EU27"]}).datamatrix_plot(stacked=True)

# get pkm/cap
dm_pkmtot.sort("Country")
dm_pkmtot.sort("Years")
dm_pop = DM_lfs["ots"]["pop"]["lfs_population_"].copy()
dm_pop.append(DM_lfs["fts"]["pop"]["lfs_population_"][1],"Years")
dm_pop = dm_pop.filter({"Country" : ["EU27"]})
dm_pop.sort("Years")
dm_pkmtot.array = dm_pkmtot.array / dm_pop.array
dm_pkmtot.units["tra_passenger_pkm"] = "pkm/cap"
dm_pkmtot.rename_col("tra_passenger_pkm","tra_pkm-cap","Variables")

# check
# dm_pkmtot.filter({"Country" : ["EU27"]}).datamatrix_plot()

years_ots = list(range(1990,2023+1))
years_fts = list(range(2025,2055,5))

###############
##### OTS #####
###############

dm_pkmtot_ots = dm_pkmtot.filter({"Years" : years_ots})

#######################
##### FTS LEVEL 1 #####
#######################

# level 1: continuing as is
dm_pkmtot_fts_level1 = dm_pkmtot.filter({"Years" : years_fts})
# dm_pkmtot.filter({"Country" : ["EU27"]}).datamatrix_plot()

#######################
##### FTS LEVEL 4 #####
#######################

# for ldv source: 

dm_pkmtot_level4 = dm_pkmtot.copy()
idx = dm_pkmtot_level4.idx
for y in range(2030,2055,5):
    dm_pkmtot_level4.array[idx["EU27"],idx[y],:] = np.nan
dm_pkmtot_level4.array[idx["EU27"],idx[2050],:] = dm_pkmtot_level4.array[idx["EU27"],idx[2023],:]*(1-0.14)
dm_pkmtot_level4 = linear_fitting(dm_pkmtot_level4, years_fts)
# dm_pkmtot_level4.filter({"Country" : ["EU27"]}).datamatrix_plot()
dm_pkmtot_fts_level4 = dm_pkmtot_level4.filter({"Years" : years_fts})
# dm_pkmtot_level4.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

# get levels for 2 and 3
variabs = dm_pkmtot_fts_level1.col_labels["Variables"]
df_temp = pd.DataFrame({"level" : np.tile(range(1,4+1),len(variabs)), 
                        "variable": np.repeat(variabs, 4)})
df_temp["value"] = np.nan
df_temp = df_temp.pivot(index=['level'], 
                        columns=['variable'], values="value").reset_index()
for v in variabs:
    idx = dm_pkmtot_fts_level1.idx
    level1 = dm_pkmtot_fts_level1.array[idx["EU27"],idx[2050],idx[v]]
    idx = dm_pkmtot_fts_level4.idx
    level4 = dm_pkmtot_fts_level4.array[idx["EU27"],idx[2050],idx[v]]
    arr = np.array([level1,np.nan,np.nan,level4])
    df_temp[v] = pd.Series(arr).interpolate().to_numpy()

#######################
##### FTS LEVEL 2 #####
#######################

dm_pkmtot_level2 = dm_pkmtot.copy()
idx = dm_pkmtot_level2.idx
for y in range(2030,2055,5):
    dm_pkmtot_level2.array[idx["EU27"],idx[y],:] = np.nan
for v in variabs:
    dm_pkmtot_level2.array[idx["EU27"],idx[2050],idx[v]] = df_temp.loc[df_temp["level"] == 2,v]
dm_pkmtot_level2 = linear_fitting(dm_pkmtot_level2, years_fts)
# dm_pkmtot_level2.filter({"Country" : ["EU27"]}).datamatrix_plot()
dm_pkmtot_fts_level2 = dm_pkmtot_level2.filter({"Years" : years_fts})
# dm_pkmtot_fts_level2.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

#######################
##### FTS LEVEL 3 #####
#######################

dm_pkmtot_level3 = dm_pkmtot.copy()
idx = dm_pkmtot_level3.idx
for y in range(2030,2055,5):
    dm_pkmtot_level3.array[idx["EU27"],idx[y],:] = np.nan
for v in variabs:
    dm_pkmtot_level3.array[idx["EU27"],idx[2050],idx[v]] = df_temp.loc[df_temp["level"] == 3,v]
dm_pkmtot_level3 = linear_fitting(dm_pkmtot_level3, years_fts)
# dm_pkmtot_level3.filter({"Country" : ["EU27"]}).datamatrix_plot()
dm_pkmtot_fts_level3 = dm_pkmtot_level3.filter({"Years" : years_fts})
# dm_pkmtot_fts_level3.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

################
##### SAVE #####
################

# split between ots and fts
DM_pkmtot = {"ots": {"pkm" : []}, "fts": {"pkm" : dict()}}
DM_pkmtot["ots"]["pkm"] = dm_pkmtot_ots.copy()
DM_pkmtot["fts"]["pkm"][1] = dm_pkmtot_fts_level1.copy()
DM_pkmtot["fts"]["pkm"][2] = dm_pkmtot_fts_level2.copy()
DM_pkmtot["fts"]["pkm"][3] = dm_pkmtot_fts_level3.copy()
DM_pkmtot["fts"]["pkm"][4] = dm_pkmtot_fts_level4.copy()

# save
f = os.path.join(current_file_directory, '../data/datamatrix/lever_pkm.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM_pkmtot, handle, protocol=pickle.HIGHEST_PROTOCOL)








