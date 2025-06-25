
# packages
from model.common.data_matrix_class import DataMatrix
import pandas as pd
import pickle
import os
import numpy as np
import warnings
import eurostat
import re
# from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat
warnings.simplefilter("ignore")
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

# NOTE: for the business as usual, we will put no improvement on the energy efficiency.
# TODO: use documentation EUCalc to do level 1 etc.

# file
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/industry/eu/python/industry_lever_technology-development.py"

# directories
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# create dm
countries = ['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czech Republic','Denmark',
             'EU27','Estonia','Finland','France','Germany','Greece','Hungary','Ireland','Italy',
             'Latvia','Lithuania','Luxembourg','Malta','Netherlands','Poland','Portugal',
             'Romania','Slovakia','Slovenia','Spain','Sweden','United Kingdom']
years = list(range(1990,2023+1,1))
years = years + list(range(2025, 2050+1, 5))
variabs = ['aluminium-prim', 'aluminium-sec', 'aluminium-sec-post-consumer', 
           'cement-dry-kiln', 'cement-geopolym', 'cement-sec-post-consumer', 
           'cement-wet-kiln', 'chem-chem-tech', 'chem-sec-post-consumer', 
           'copper-sec-post-consumer', 'copper-tech', 'fbt-tech', 'glass-glass', 
           'glass-sec-post-consumer', 'lime-lime',
           'mae-tech', 'ois-tech', 'paper-sec-post-consumer', 'paper-tech', 
           'pulp-tech', 'steel-BF-BOF', 'steel-hisarna', 'steel-hydrog-DRI', 
           'steel-scrap-EAF', 'steel-sec-post-consumer', 'textiles-tech', 
           'tra-equip-tech', 'wwp-tech']
variabs = ["technology-development_" + i for i in variabs]
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
dm = DataMatrix()
dm.col_labels = {"Country" : countries, "Years" : years, "Variables" : variabs}
dm.units = units_dict
dm.idx = index_dict
dm.array = np.zeros((len(countries), len(years), len(variabs)))

# make nan for other than EU27, as for EU for the moment we keep BAU which is 0
countries_oth = np.array(countries)[[i not in "EU27" for i in countries]].tolist()
idx = dm.idx
years = list(range(2025, 2050+1, 5))
for c in countries_oth:
    for y in years:
        for v in variabs:
            dm.array[idx[c],idx[y],idx[v]] = np.nan
df = dm.write_df()

# deepen
dm.deepen()

# split between ots and fts
years_ots = list(range(1990,2023+1))
years_fts = list(range(2025,2055,5))
dm_ots = dm.filter({"Years" : years_ots})
dm_fts = dm.filter({"Years" : years_fts})
DM_fts = {1: dm_fts.copy(), 2: dm_fts.copy(), 3: dm_fts.copy(), 4: dm_fts.copy()} # for now we set all levels to be the same
DM = {"ots" : dm_ots,
      "fts" : DM_fts}

# save
f = os.path.join(current_file_directory, '../data/datamatrix/lever_technology-development.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)

# df = dm.write_df()
# df = df.loc[df["Country"] == "EU27",:]
# df = df.loc[df["Years"].isin([2022,2023]),:]
# df_temp = pd.melt(df, id_vars = ['Country','Years'], var_name='variable')
# name = "temp.xlsx"
# df_temp.to_excel("~/Desktop/" + name)




