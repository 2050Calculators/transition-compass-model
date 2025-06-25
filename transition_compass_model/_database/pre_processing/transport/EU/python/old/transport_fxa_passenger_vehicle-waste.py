
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
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/transport/EU/python/transport_fxa_passenger_new-vehicles.py"

# directories
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# load current transport pickle
filepath = os.path.join(current_file_directory, '../../../../data/datamatrix/transport.pickle')
with open(filepath, 'rb') as handle:
    DM_tra = pickle.load(handle)
    
# load fleet
filepath = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/passenger_fleet.pickle')
with open(filepath, 'rb') as handle:
    dm_fleet = pickle.load(handle)
    
# load new vehicles
filepath = os.path.join(current_file_directory, '../data/datamatrix/fxa_passenger_new-vehicles.pickle')
with open(filepath, 'rb') as handle:
    dm_new = pickle.load(handle)

# fleet (t) = fleet (t-1) - waste (t) + new (t)
# waste (t) = fleet (t-1) - fleet (t) + new (t)
# ren rate (t-1) = waste (t) / fleet (t-1)
# when waste is negative, it should be assigned a zero, and new registrations should
# be adjusted to be equal the difference between stock t and stock t-1
# probably all fts should be done after this adjustment is done

# Paola does this in compute_renewal_rate_and_adjust(). The logic is obtain waste from
# equation above, and then do readjustments on stocks. On Monday:
# - re-organize all files to have only OTS (re-save all data matrices etc in riorganised folders)
# - get this function going with your data
# - write a file for FTS













