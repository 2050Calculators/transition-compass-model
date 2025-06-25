
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
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/transport/EU/python/transport_fxa_passenger_vehicle-lifetime.py"

# directories
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# load current transport pickle
filepath = os.path.join(current_file_directory, '../../../../data/datamatrix/transport.pickle')
with open(filepath, 'rb') as handle:
    DM_tra = pickle.load(handle)

# lifetime is a set of constants with the shape of country and time (so fxa), with all fts
# LDV: ICE and PHEV vehicles have lifetime of 13.5 years
# LDV: New technology like BEV and PHEV have lifetimes initially of 5.5, and the 13.5
# 2W: 8 years
# We assume rail lifetime is 30 years, metrotram lifetime is 20 years and bus lifetime is 10 years

# I assume that in all countries it's the same than it is in CH
dm_lifetime_ch = DM_tra["fxa"]["passenger_vehicle-lifetime"].filter({"Country" : ["Switzerland"]})
countries = ['Belgium','Bulgaria','Croatia','Cyprus','Czech Republic','Denmark',
             'EU27','Estonia','Finland','France','Germany','Greece','Hungary','Ireland','Italy',
             'Latvia','Lithuania','Luxembourg','Malta','Netherlands','Poland','Portugal',
             'Romania','Slovakia','Slovenia','Spain','Sweden','United Kingdom']
dm_temp = dm_lifetime_ch.copy()
dm_temp.rename_col("Switzerland","Austria","Country")
dm_lifetime = dm_temp.copy()
for c in countries:
    dm_temp = dm_lifetime_ch.copy()
    dm_temp.rename_col("Switzerland",c,"Country")
    dm_lifetime.append(dm_temp,"Country")
# dm_lifetime.flatten().flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()

# save
f = os.path.join(current_file_directory, '../data/datamatrix/fxa_passenger_vehicle-lifetime.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_lifetime, handle, protocol=pickle.HIGHEST_PROTOCOL)





