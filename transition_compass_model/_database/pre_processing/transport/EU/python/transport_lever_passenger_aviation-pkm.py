
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
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/transport/EU/python/transport_passenger_aviation-pkm.py"

# directories
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# load current transport pickle
filepath = os.path.join(current_file_directory, '../../../../data/datamatrix/transport.pickle')
with open(filepath, 'rb') as handle:
    DM_tra = pickle.load(handle)
    
# load current lifestyles pickle
filepath = os.path.join(current_file_directory, '../../../../data/datamatrix/lifestyles.pickle')
with open(filepath, 'rb') as handle:
    DM_lfe = pickle.load(handle)

####################
##### GET DATA #####
####################

# get iso codes
dict_iso2 = eurostat_iso2_dict()
dict_iso2.pop('CH')  # Remove Switzerland

# downloand and save
code = "avia_tppa"
eurostat.get_pars(code)
filter = {'geo\\TIME_PERIOD': list(dict_iso2.keys()),
          'tra_cov': 'TOTAL',
          'unit' : 'MIO_PKM'}
mapping_dim = {'Country': 'geo\\TIME_PERIOD',
                'Variables': 'tra_cov'}
dm_avi = get_data_api_eurostat(code, filter, mapping_dim, 'mi-pkm')

# check
# dm_avi.filter({"Country" : ["EU27"]}).datamatrix_plot()

# rename to aviation
dm_avi.rename_col("TOTAL","aviation","Variables")

###################
##### FIX OTS #####
###################

# Set years range
years_setting = [1990, 2023, 2050, 5]
startyear = years_setting[0]
baseyear = years_setting[1]
lastyear = years_setting[2]
step_fts = years_setting[3]
years_ots = list(range(startyear, baseyear+1, 1))
years_fts = list(range(baseyear+2, lastyear+1, step_fts))
years_all = years_ots + years_fts

# before 2008: do trend on 2008-2012
# note: I do not do trend on 2008-2019 as with that it would make pkm equal zero in 1990
# years_present = dm_avi .col_labels["Years"]
# years_fitting = years_ots.copy()
# for y in years_present: years_fitting.remove(y)
years_fitting = list(range(startyear,2007+1))
dm_avi = linear_fitting(dm_avi , years_fitting, based_on=list(range(2008,2012)))

# check
# dm_avi.filter({"Country" : ["EU27"]}).datamatrix_plot()

# for 2023: do trend on 2009-2019 (when we have data pre covid)
dm_avi = linear_fitting(dm_avi , [2023], based_on=list(range(2009,2019+1)))

# check
# dm_avi.filter({"Country" : ["EU27"]}).datamatrix_plot()

####################
##### MAKE FTS #####
####################

# make function to fill in missing years fts for EU27 with linear fitting
def make_fts(dm, variable, year_start, year_end, country = "EU27", dim = "Categories1", 
             min_t0=0, min_tb=0, years_fts = years_fts): # I put minimum to 1 so it does not go to zero
    dm = dm.copy()
    idx = dm.idx
    based_on_yars = list(range(year_start, year_end + 1, 1))
    dm_temp = linear_fitting(dm.filter({"Country" : [country], dim : [variable]}), 
                             years_ots = years_fts, min_t0=min_t0, min_tb=min_tb, based_on = based_on_yars)
    idx_temp = dm_temp.idx
    if dim == "Variables":
        dm.array[idx[country],:,idx[variable],...] = \
            np.round(dm_temp.array[idx_temp[country],:,idx_temp[variable],...],0)
    if dim == "Categories1":
        dm.array[idx[country],:,:,idx[variable]] = \
            np.round(dm_temp.array[idx_temp[country],:,:,idx_temp[variable]], 0)
    if dim == "Categories2":
        dm.array[idx[country],:,:,:,idx[variable]] = \
            np.round(dm_temp.array[idx_temp[country],:,:,:,idx_temp[variable]], 0)
    if dim == "Categories3":
        dm.array[idx[country],:,:,:,:,idx[variable]] = \
            np.round(dm_temp.array[idx_temp[country],:,:,:,:,idx_temp[variable]], 0)
    
    return dm

# add missing years fts
dm_avi.add(np.nan, col_label=years_fts, dummy=True, dim='Years')

# set default time window for linear trend
baseyear_start = 2008
baseyear_end = 2019

# try fts
# product = "aviation"
# (make_fts(dm_avi, product, baseyear_start, baseyear_end, dim = "Variables").
#   datamatrix_plot(selected_cols={"Country" : ["EU27"]}))

# make fts
dm_avi = make_fts(dm_avi, "aviation", baseyear_start, baseyear_end, dim = "Variables")

################
##### SAVE #####
################

DM_tra["ots"]["passenger_aviation-pkm"]

# sort
dm_avi.sort("Country")
dm_avi.sort("Years")
dm_avi.sort("Variables")

# make correct shape and name
dm_avi.rename_col("aviation","tra_aviation","Variables")
dm_avi.deepen()
dm_avi.rename_col("tra","tra_pkm-cap","Variables")

# change unit
dm_avi.change_unit("tra_pkm-cap", 1e6, "mi-pkm", "pkm")
dm_pop = DM_lfe["ots"]['pop']['lfs_population_'].copy()
dm_pop.append(DM_lfe["fts"]['pop']['lfs_population_'][1],"Years")
dm_pop.sort("Country")
dm_pop.sort("Years")
dm_pop = dm_pop.filter({"Country" : dm_avi.col_labels["Country"]})
dm_pop.sort("Country")
dm_avi.array = dm_avi.array / dm_pop.array[...,np.newaxis]
dm_avi.units["tra_pkm-cap"] = "pkm/cap"

# check
# dm_avi.filter({"Country" : ["EU27"]}).datamatrix_plot()

# split between ots and fts
DM_avi = {"ots": {"passenger_aviation-pkm" : []}, "fts": {"passenger_aviation-pkm" : dict()}}
DM_avi["ots"]["passenger_aviation-pkm"] = dm_avi.filter({"Years" : years_ots})
for i in range(1,4+1):
    DM_avi["fts"]["passenger_aviation-pkm"][i] = dm_avi.filter({"Years" : years_fts})

# save
f = os.path.join(current_file_directory, '../data/datamatrix/lever_passenger_aviation-pkm.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM_avi, handle, protocol=pickle.HIGHEST_PROTOCOL)









