
# packages
from model.common.data_matrix_class import DataMatrix
from model.common.auxiliary_functions import linear_fitting
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

from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat
from _database.pre_processing.routine_JRC import get_jrc_data
from model.common.auxiliary_functions import eurostat_iso2_dict, jrc_iso2_dict

# file
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/transport/EU/python/transport_fxa_freight_mode_other.py"

# directories
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# load current transport pickle
filepath = os.path.join(current_file_directory, '../../../../data/datamatrix/transport.pickle')
with open(filepath, 'rb') as handle:
    DM_tra = pickle.load(handle)

# load tkm pickle
filepath = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/freight_tkm.pickle')
with open(filepath, 'rb') as handle:
    DM_tkm = pickle.load(handle)
    
# load renewal rate pickle
filepath = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/passenger_renewal-rate.pickle')
with open(filepath, 'rb') as handle:
    dm_renrate = pickle.load(handle)

###############
##### TKM #####
###############

DM_tra["fxa"]["freight_mode_other"].units

# get total tkm
dm_tkm = DM_tkm["ots"]["freight_tkm"].copy()
dm_tkm.append(DM_tkm["fts"]["freight_tkm"][1],"Years")

# check
# dm_tkm.filter({"Country" : ["EU27"]}).datamatrix_plot()

# rename
dm_tkm.rename_col("tra_freight_tkm","tra_freight_tkm-by-veh","Variables")

# filter
dm_tkm.filter({"Categories1" : ['IWW', 'aviation', 'marine', 'rail']}, inplace=True)

########################
##### RENEWAL RATE #####
########################

# TODO: for the moment I take some values from renewal rate of passenger vehicles
# to check with Paola what this is

df = DM_tra["fxa"]["freight_mode_other"].filter({"Variables" : ["tra_freight_renewal-rate"]}).write_df()
df1 = dm_renrate.write_df()

# take renewal rate rail CEV as proxy for renewal rate of rail and aviation, and for IWW and marine put missing
dm_renrate_freight = dm_renrate.copy()
dm_renrate_freight = dm_renrate_freight.flatten()
dm_renrate_freight.filter({"Categories1" : ['rail_CEV']}, inplace=True)
dm_renrate_freight.rename_col("rail_CEV","rail","Categories1")
arr_temp = dm_renrate_freight.array
dm_renrate_freight.add(arr_temp, col_label="aviation", dim="Categories1")
dm_renrate_freight.add(np.nan, col_label="IWW", dummy=True, dim="Categories1")
dm_renrate_freight.add(np.nan, col_label="marine", dummy=True, dim="Categories1")
dm_renrate_freight.sort("Categories1")
dm_renrate_freight.rename_col("tra_passenger_renewal-rate","tra_freight_renewal-rate","Variables")

# make fts
dm_renrate_freight = linear_fitting(dm_renrate_freight, list(range(2025,2050+5,5)))

# # check
# dm_renrate_freight.filter({"Country" : ["EU27"]}).datamatrix_plot()

########################
##### PUT TOGETHER #####
########################

dm_mode_oth = dm_tkm.copy()
dm_mode_oth.append(dm_renrate_freight,"Variables")
dm_mode_oth.sort("Variables")

# save
f = os.path.join(current_file_directory, '../data/datamatrix/fxa_freight_mode_other.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_mode_oth, handle, protocol=pickle.HIGHEST_PROTOCOL)

