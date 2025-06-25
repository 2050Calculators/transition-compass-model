
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
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/transport/EU/python/transport_lever_freight_tkm.py"

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

# Set years range
years_setting = [1990, 2023, 2050, 5]
startyear = years_setting[0]
baseyear = years_setting[1]
lastyear = years_setting[2]
step_fts = years_setting[3]
years_ots = list(range(startyear, baseyear+1, 1))
years_fts = list(range(baseyear+2, lastyear+1, step_fts))
years_all = years_ots + years_fts

# get total tkm
dm_tkmtot = DM_tkm["ots"]["freight_tkm"].copy()
dm_tkmtot.append(DM_tkm["fts"]["freight_tkm"][1],"Years")
dm_tkmtot.group_all("Categories1")

# check
# dm_tkmtot.filter({"Country" : ["EU27"]}).datamatrix_plot()

# adjust units
dm_tkmtot.change_unit("tra_freight_tkm", 1e-9, "tkm", "bn-tkm")

# rename
dm_tkmtot.rename_col("tra_freight_tkm","tra_freight_tkm-total-demand","Variables")

# check
list(DM_tra["ots"])
DM_tra["ots"]["freight_tkm"]
DM_tra["ots"]["freight_tkm"].units
dm_tkmtot.units

# split between ots and fts
DM_tkmtot = {"ots": {"freight_tkm" : []}, "fts": {"freight_tkm" : dict()}}
DM_tkmtot ["ots"]["freight_tkm"] = dm_tkmtot.filter({"Years" : years_ots})
for i in range(1,4+1):
    DM_tkmtot["fts"]["freight_tkm"][i] = dm_tkmtot.filter({"Years" : years_fts})

# save
f = os.path.join(current_file_directory, '../data/datamatrix/lever_freight_tkm.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM_tkmtot, handle, protocol=pickle.HIGHEST_PROTOCOL)








