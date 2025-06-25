
# packages
from model.common.data_matrix_class import DataMatrix
from model.common.auxiliary_functions import my_pickle_dump
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
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/industry/Switzerland/industry_buildpickle.py"

# directories
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# load current industry pickle
filepath = os.path.join(current_file_directory, '../../../data/datamatrix/industry.pickle')
with open(filepath, 'rb') as handle:
    DM_industry = pickle.load(handle)

#######################################
###### GENERATE FAKE SWITZERLAND ######
#######################################

for key in ['fxa', 'ots', 'calibration']:
    dm_names = list(DM_industry[key])
    for name in dm_names:
        dm_temp = DM_industry[key][name]
        if "Switzerland" not in dm_temp.col_labels["Country"]:
            idx = dm_temp.idx
            arr_temp = dm_temp.array[idx["EU27"],...]
            dm_temp.add(arr_temp[np.newaxis,...], "Country", "Switzerland")
            dm_temp.sort("Country")


dm_names = list(DM_industry["fts"])
for name in dm_names:
    for i in range(1,4+1):
        dm_temp = DM_industry["fts"][name][i]
        if "Switzerland" not in dm_temp.col_labels["Country"]:
            idx = dm_temp.idx
            arr_temp = dm_temp.array[idx["EU27"],...]
            dm_temp.add(arr_temp[np.newaxis,...], "Country", "Switzerland")
            dm_temp.sort("Country")
            
################################
###### GENERATE FAKE VAUD ######
################################

for key in ['fxa', 'ots', 'calibration']:
    dm_names = list(DM_industry[key])
    for name in dm_names:
        dm_temp = DM_industry[key][name]
        if "Vaud" not in dm_temp.col_labels["Country"]:
            idx = dm_temp.idx
            arr_temp = dm_temp.array[idx["EU27"],...]
            dm_temp.add(arr_temp[np.newaxis,...], "Country", "Vaud")
            dm_temp.sort("Country")


dm_names = list(DM_industry["fts"])
for name in dm_names:
    for i in range(1,4+1):
        dm_temp = DM_industry["fts"][name][i]
        if "Vaud" not in dm_temp.col_labels["Country"]:
            idx = dm_temp.idx
            arr_temp = dm_temp.array[idx["EU27"],...]
            dm_temp.add(arr_temp[np.newaxis,...], "Country", "Vaud")
            dm_temp.sort("Country")

################
##### SAVE #####
################

# save
f = os.path.join(current_file_directory, '../../../data/datamatrix/industry.pickle')
my_pickle_dump(DM_industry, f)






