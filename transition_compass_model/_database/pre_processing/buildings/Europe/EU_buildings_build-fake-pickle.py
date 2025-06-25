
# packages
from model.common.data_matrix_class import DataMatrix
from model.common.auxiliary_functions import my_pickle_dump
import pandas as pd
import pickle
import os
import numpy as np
import warnings

# from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat
warnings.simplefilter("ignore")
import plotly.express as px
import plotly.io as pio
import re
pio.renderers.default='browser'
import subprocess

# file
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/buildings/Europe/buildings_build-pickle.py"

# current file directory
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# load current building pickle
filepath = os.path.join(current_file_directory, '../../../data/datamatrix/buildings.pickle')
with open(filepath, 'rb') as handle:
    DM_bui = pickle.load(handle)

# create fake EU27 equals to CH
for key in DM_bui["ots"].keys():
    if type(DM_bui["ots"][key]) is dict:
        for sub_key in DM_bui["ots"][key].keys():
            dm_temp = DM_bui["ots"][key][sub_key].filter({"Country" : ["Switzerland"]})
            dm_temp.rename_col("Switzerland","EU27","Country")
            DM_bui["ots"][key][sub_key].append(dm_temp, "Country")
    else:
        dm_temp = DM_bui["ots"][key].filter({"Country" : ["Switzerland"]})
        dm_temp.rename_col("Switzerland","EU27","Country")
        DM_bui["ots"][key].append(dm_temp, "Country")

for key in DM_bui["fts"].keys():
    if list(DM_bui["fts"][key].keys()) != list(range(1,5)):
        for sub_key in DM_bui["fts"][key].keys():
            for i in range(1,4+1):
                if "EU27" in DM_bui["fts"][key][sub_key][i].col_labels["Country"]:
                    continue
                else:
                    dm_temp = DM_bui["fts"][key][sub_key][i].filter({"Country" : ["Switzerland"]})
                    dm_temp.rename_col("Switzerland","EU27","Country")
                    DM_bui["fts"][key][sub_key][i].append(dm_temp, "Country")
    else:
        for i in range(1,4+1):
            if "EU27" in DM_bui["fts"][key][i].col_labels["Country"]:
                continue
            else:
                dm_temp = DM_bui["fts"][key][i].filter({"Country" : ["Switzerland"]})
                dm_temp.rename_col("Switzerland","EU27","Country")
                DM_bui["fts"][key][i].append(dm_temp, "Country")

for key in DM_bui["fxa"].keys():
    if type(DM_bui["fxa"][key]) is dict:
        dm_temp = DM_bui["fxa"][key][sub_key].filter({"Country" : ["Switzerland"]})
        dm_temp.rename_col("Switzerland","EU27","Country")
        DM_bui["fxa"][key][sub_key].append(dm_temp, "Country")
    else:
        dm_temp = DM_bui["fxa"][key].filter({"Country" : ["Switzerland"]})
        dm_temp.rename_col("Switzerland","EU27","Country")
        DM_bui["fxa"][key].append(dm_temp, "Country")

###############
#### SAVE #####
###############

# save
f = os.path.join(current_file_directory, '../../../data/datamatrix/buildings.pickle')
my_pickle_dump(DM_bui, f)
