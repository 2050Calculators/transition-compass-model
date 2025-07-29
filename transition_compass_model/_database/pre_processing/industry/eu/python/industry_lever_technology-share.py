
# packages
from model.common.data_matrix_class import DataMatrix
from model.common.auxiliary_functions import linear_fitting
import pandas as pd
import pickle
import os
import numpy as np
import warnings
warnings.simplefilter("ignore")

# directories
current_file_directory = os.getcwd()

# create dm
countries = ['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czech Republic','Denmark',
             'EU27','Estonia','Finland','France','Germany','Greece','Hungary','Ireland','Italy',
             'Latvia','Lithuania','Luxembourg','Malta','Netherlands','Poland','Portugal',
             'Romania','Slovakia','Slovenia','Spain','Sweden','United Kingdom']
years = list(range(1990,2023+1,1))
years = years + list(range(2025, 2050+1, 5))
variabs = ['steel-BF-BOF','steel-hisarna', 'steel-hydrog-DRI','cement-dry-kiln','cement-geopolym','cement-wet-kiln']
units = list(np.repeat("%", len(variabs)))
units_dict = dict()
for i in range(0, len(variabs)):
    units_dict[variabs[i]] = units[i]
index_dict = dict()
for i in range(0, len(countries)):
    index_dict[countries[i]] = i
for i in range(0, len(years)):
    index_dict[years[i]] = i
for i in range(0, len(variabs)):
    index_dict[variabs[i]] = i
dm = DataMatrix(empty=True)
dm.col_labels = {"Country" : countries, "Years" : years, "Variables" : variabs}
dm.units = units_dict
dm.idx = index_dict
dm.array = np.zeros((len(countries), len(years), len(variabs)))
dm[...] = np.nan

# steel: in bau it's 100% steel-BF-BOF and 0% steel-hisarna and steel-hydrog-DRI
dm[:,:,'steel-BF-BOF'] = 1
dm[:,:,'steel-hisarna'] = 0
dm[:,:,'steel-hydrog-DRI'] = 0

# cement: it's mostly dry (91% dry vs 9% wet), and geopolymer is zero in bau
dm[:,:,'cement-dry-kiln'] = 0.91
dm[:,:,'cement-wet-kiln'] = 0.09
dm[:,:,'cement-geopolym'] = 0

# make nan for other than EU27 for fts
countries_oth = np.array(countries)[[i not in "EU27" for i in countries]].tolist()
idx = dm.idx
years = list(range(2025, 2050+1, 5))
for c in countries_oth:
    for y in years:
        for v in variabs:
            dm.array[idx[c],idx[y],idx[v]] = np.nan
df = dm.write_df()

# rename
for i in variabs:
    dm.rename_col(i, "technology-share_" + i, "Variables")
dm.deepen()

# dm.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot(stacked=True)

# set years
years_ots = list(range(1990,2023+1))
years_fts = list(range(2025,2055,5))

###############
##### OTS #####
###############

dm_ots = dm.filter({"Years" : years_ots})

#######################
##### FTS LEVEL 1 #####
#######################

# level 1: continuing as is
dm_fts_level1 = dm.filter({"Years" : years_fts})

#######################
##### FTS LEVEL 4 #####
#######################

# 2050: steel-hisarna and steel-hydrog-DRI to 10% each, cement-geopolymer 20%

# create a dm level4
dm_level4 = dm.copy()
years = list(range(2030,2055,5))
for y in years:
    dm_level4["EU27",y,:,:] = np.nan
dm_level4["EU27",2050,:,"steel-hisarna"] = 0.1
dm_level4["EU27",2050,:,"steel-hydrog-DRI"] = 0.1
dm_level4["EU27",2050,:,"steel-BF-BOF"] = 0.8
dm_level4["EU27",2050,:,'cement-dry-kiln'] = 0.8
dm_level4["EU27",2050,:,'cement-wet-kiln'] = 0
dm_level4["EU27",2050,:,'cement-geopolym'] = 0.2
dm_level4 = linear_fitting(dm_level4, years_fts)
# dm_level4.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot(stacked=True)
dm_fts_level4 = dm_level4.filter({"Years" : years_fts})
# dm_fts_level4.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

# get levels for 2 and 3
variabs = dm_fts_level1.col_labels["Categories1"]
df_temp = pd.DataFrame({"level" : np.tile(range(1,4+1),len(variabs)), 
                        "variable": np.repeat(variabs, 4)})
df_temp["value"] = np.nan
df_temp = df_temp.pivot(index=['level'], 
                        columns=['variable'], values="value").reset_index()
for v in variabs:
    level1 = dm_fts_level1["EU27",2050,:,v][0]
    level4 = dm_fts_level4["EU27",2050,:,v][0]
    arr = np.array([level1,np.nan,np.nan,level4])
    df_temp[v] = pd.Series(arr).interpolate().to_numpy()

#######################
##### FTS LEVEL 2 #####
#######################

dm_level2 = dm.copy()
for y in range(2030,2055,5): dm_level2["EU27",y,:,:] = np.nan
for v in variabs:
    dm_level2["EU27",2050,:,v] = df_temp.loc[df_temp["level"] == 2,v]
dm_level2 = linear_fitting(dm_level2, years_fts)
# dm_level2.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot(stacked=True)
dm_fts_level2 = dm_level2.filter({"Years" : years_fts})
# dm_fts_level2.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot(stacked=True)

#######################
##### FTS LEVEL 3 #####
#######################

dm_level3 = dm.copy()
for y in range(2030,2055,5): dm_level3["EU27",y,:,:] = np.nan
for v in variabs:
    dm_level3["EU27",2050,:,v] = df_temp.loc[df_temp["level"] == 3,v]
dm_level3 = linear_fitting(dm_level3, years_fts)
# dm_level3.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot(stacked=True)
dm_fts_level3 = dm_level3.filter({"Years" : years_fts})
# dm_fts_level3.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot(stacked=True)

################
##### SAVE #####
################

DM_fts = {1: dm_fts_level1.copy(), 2: dm_fts_level2.copy(), 3: dm_fts_level3.copy(), 4: dm_fts_level4.copy()}
DM = {"ots" : dm_ots,
      "fts" : DM_fts}
f = os.path.join(current_file_directory, '../data/datamatrix/lever_technology-share.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)

# df = dm.write_df()
# df_temp = pd.melt(df, id_vars = ['Country', 'Years'], var_name='variable')
# df_temp = df_temp.loc[df_temp["Country"].isin(["Austria","France"]),:]
# df_temp = df_temp.loc[df_temp["Years"]==1990,:]
# name = "temp.xlsx"
# df_temp.to_excel("~/Desktop/" + name)




