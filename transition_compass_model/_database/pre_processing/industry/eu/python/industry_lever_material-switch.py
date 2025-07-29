
# packages
from model.common.data_matrix_class import DataMatrix
from model.common.auxiliary_functions import linear_fitting
import pickle
import os
import numpy as np
import pandas as pd
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
variabs = ['build-cement-to-timber', 'build-steel-to-timber', 'cars-steel-to-aluminium', 'cars-steel-to-chem', 'reno-chem-to-natfibers', 'reno-chem-to-paper', 'trucks-steel-to-aluminium', 'trucks-steel-to-chem']
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
# note that this is all zeroes as from page 27 Table 4 of https://www.european-calculator.eu/wp-content/uploads/2020/04/D3.1-Raw-materials-module-and-manufacturing.pdf

# make nan for other than EU27
countries_oth = np.array(countries)[[i not in "EU27" for i in countries]].tolist()
idx = dm.idx
years = list(range(2025, 2050+1, 5))
for c in countries_oth:
    for y in years:
        for v in variabs:
            dm.array[idx[c],idx[y],idx[v]] = np.nan
df = dm.write_df()

# rename
variabs = dm.col_labels["Variables"]
for v in variabs:
    dm.rename_col(v, "material-switch_" + v, "Variables")
dm.deepen()

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
# dm_fts_level1.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

#######################
##### FTS LEVEL 4 #####
#######################

# we take levels for 2050 from eucalc, and do a linear trend for 2025-2050
dm_level4 = dm.copy()
idx = dm_level4.idx
for y in range(2030,2055,5):
    dm_level4.array[idx["EU27"],idx[y],:,:] = np.nan
dm_level4.array[idx["EU27"],idx[2050],:,idx["cars-steel-to-chem"]] = 0.20
dm_level4.array[idx["EU27"],idx[2050],:,idx["trucks-steel-to-chem"]] = 0.15
dm_level4.array[idx["EU27"],idx[2050],:,idx["cars-steel-to-aluminium"]] = 0.50
dm_level4.array[idx["EU27"],idx[2050],:,idx["trucks-steel-to-aluminium"]] = 0.45
dm_level4.array[idx["EU27"],idx[2050],:,idx["build-steel-to-timber"]] = 0.20
dm_level4.array[idx["EU27"],idx[2050],:,idx["build-cement-to-timber"]] = 0.60
dm_level4.array[idx["EU27"],idx[2050],:,idx["reno-chem-to-paper"]] = 0.10
dm_level4.array[idx["EU27"],idx[2050],:,idx["reno-chem-to-natfibers"]] = 0.20
dm_level4 = linear_fitting(dm_level4, years_fts)
# dm_level4.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()
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
    idx = dm_fts_level1.idx
    level1 = dm_fts_level1.array[idx["EU27"],idx[2050],:,idx[v]][0]
    idx = dm_fts_level4.idx
    level4 = dm_fts_level4.array[idx["EU27"],idx[2050],:,idx[v]][0]
    arr = np.array([level1,np.nan,np.nan,level4])
    df_temp[v] = pd.Series(arr).interpolate().to_numpy()

#######################
##### FTS LEVEL 2 #####
#######################

dm_level2 = dm.copy()
idx = dm_level2.idx
for y in range(2030,2055,5):
    dm_level2.array[idx["EU27"],idx[y],:,:] = np.nan
for v in variabs:
    dm_level2.array[idx["EU27"],idx[2050],:,idx[v]] = df_temp.loc[df_temp["level"] == 2,v]
dm_level2 = linear_fitting(dm_level2, years_fts)
# dm_level2.filter({"Country" : ["EU27"], "Categories1":["LDV"]}).flatten().datamatrix_plot()
dm_fts_level2 = dm_level2.filter({"Years" : years_fts})
# dm_fts_level2.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

#######################
##### FTS LEVEL 3 #####
#######################

dm_level3 = dm.copy()
idx = dm_level3.idx
for y in range(2030,2055,5):
    dm_level3.array[idx["EU27"],idx[y],:,:] = np.nan
for v in variabs:
    dm_level3.array[idx["EU27"],idx[2050],:,idx[v]] = df_temp.loc[df_temp["level"] == 3,v]
dm_level3 = linear_fitting(dm_level3, years_fts)
# dm_level3.filter({"Country" : ["EU27"], "Categories1":["LDV"]}).flatten().datamatrix_plot()
dm_fts_level3 = dm_level3.filter({"Years" : years_fts})
# dm_fts_level3.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

################
##### SAVE #####
################

DM_fts = {1: dm_fts_level1.copy(), 2: dm_fts_level2.copy(), 3: dm_fts_level3.copy(), 4: dm_fts_level4.copy()}
DM = {"ots" : dm_ots,
      "fts" : DM_fts}
f = os.path.join(current_file_directory, '../data/datamatrix/lever_material-switch.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)

# df = dm.write_df()
# import pandas as pd
# df_temp = pd.melt(df, id_vars = ['Country', 'Years'], var_name='variable')
# df_temp = df_temp.loc[df_temp["Country"].isin(["Austria","France"]),:]
# df_temp = df_temp.loc[df_temp["Years"]==1990,:]
# name = "_temp.xlsx"
# df_temp.to_excel("~/Desktop/" + name)
