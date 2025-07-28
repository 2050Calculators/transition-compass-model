
# packages
from model.common.data_matrix_class import DataMatrix
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
dm[:,:,'steel-BF-BOF'] = 0.91
dm[:,:,'steel-hisarna'] = 0.9
dm[:,:,'steel-hydrog-DRI'] = 0

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
years_fts = list(range(2025,2055,5))
for y in years_fts:
    dm_level4.array[idx["EU27"],idx[y],:,:] = np.nan
for tech in dm_level4.col_labels["Categories1"]:
    dm_level4.array[idx["EU27"],idx[2050],:,idx[tech]] = df.loc[df["material-tech"] == tech,"value"]
dm_level4 = linear_fitting(dm_level4, years_fts)
# dm_level4.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot(stacked=True)
dm_fts_level4 = dm_level4.filter({"Years" : years_fts})
# dm_fts_level4.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

#######################
##### FTS LEVEL 2 #####
#######################

# TODO: level 2 to do, for the moment we set it continuing as is
dm_fts_level2 = dm.filter({"Years" : years_fts})

#######################
##### FTS LEVEL 3 #####
#######################

# TODO: level 3 to do, for the moment we set it continuing as is
dm_fts_level3 = dm.filter({"Years" : years_fts})


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




