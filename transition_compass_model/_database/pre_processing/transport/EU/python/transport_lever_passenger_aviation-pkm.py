
# packages
from model.common.auxiliary_functions import linear_fitting
import pickle
import os
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter("ignore")
import plotly.io as pio
pio.renderers.default='browser'
from _database.pre_processing.routine_JRC import get_jrc_data
from model.common.auxiliary_functions import jrc_iso2_dict
from model.common.data_matrix_class import DataMatrix

# directories
current_file_directory = os.getcwd()

# load current transport pickle
filepath = os.path.join(current_file_directory, '../../../../data/datamatrix/transport.pickle')
with open(filepath, 'rb') as handle:
    DM_tra = pickle.load(handle)
    
# load current lifestyles pickle
filepath = os.path.join(current_file_directory, '../../../lifestyles/Europe/data/lifestyles_allcountries.pickle')
with open(filepath, 'rb') as handle:
    DM_lfe = pickle.load(handle)

####################
##### GET DATA #####
####################

dict_iso2_jrc = jrc_iso2_dict()

# get data
def get_specific_jrc_data(country_code, country_name, row_start, row_end, unit, variable = "aviation", 
                          database = "JRC-IDEES-2021_x1990_Aviation_EU"):
    
    filepath_jrc = os.path.join(current_file_directory, f"../../../industry/eu/data/JRC-IDEES-2021/EU27/{database}.xlsx")
    df_temp = pd.read_excel(filepath_jrc, sheet_name=country_code)
    df_temp = df_temp.iloc[row_start:row_end,:]
    indexes = df_temp.columns[0]
    df_temp = pd.melt(df_temp, id_vars = indexes, var_name='year')
    df_temp.columns = ["Country","Years",f"{variable}[{unit}]"]
    df_temp["Country"] = country_name
    
    return df_temp

country_codes = list(dict_iso2_jrc.keys())
country_names = list(dict_iso2_jrc.values())
df_avi = pd.concat([get_specific_jrc_data(code, name, 2, 3, "mio pkm") for code,name in zip(country_codes, country_names)],ignore_index=True)
dm_avi = DataMatrix.create_from_df(df_avi, 0)


# # get data
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrAvia_act",
#                 "variable" : "Passenger transport (mio pkm)",
#                 "sheet_last_row" : "International - Extra-EEAwUK",
#                 "sub_variables" : ["Domestic",
#                                     "International - Intra-EEAwUK",
#                                     "International - Extra-EEAwUK"],
#                 "calc_names" : ["domestic","international-int","international-extra"]}
# dm_avi = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# for v in dm_avi.col_labels["Variables"]:
#     dm_avi.rename_col(v, "aviation_" + v, "Variables")
# dm_avi.deepen()
# dm_avi.group_all("Categories1")
dm_avi.change_unit("aviation", 1e6, "mio pkm", "pkm")

# check
# df = dm_avi.write_df()
# dm_avi.filter({"Country": ["EU27"]}).datamatrix_plot()

# # get iso codes
# dict_iso2 = eurostat_iso2_dict()
# dict_iso2.pop('CH')  # Remove Switzerland

# # downloand and save
# code = "avia_tppa"
# eurostat.get_pars(code)
# filter = {'geo\\TIME_PERIOD': list(dict_iso2.keys()),
#           'tra_cov': 'TOTAL',
#           'unit' : 'MIO_PKM'}
# mapping_dim = {'Country': 'geo\\TIME_PERIOD',
#                 'Variables': 'tra_cov'}
# dm_avi = get_data_api_eurostat(code, filter, mapping_dim, 'mi-pkm')

# # check
# # dm_avi.filter({"Country" : ["EU27"]}).datamatrix_plot()

# # rename to aviation
# dm_avi.rename_col("TOTAL","aviation","Variables")

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

# fit until 2019
years_fitting = list(range(startyear,2000)) + [2022,2023]
dm_avi = linear_fitting(dm_avi , years_fitting, based_on=list(range(2000,2019+1)))

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
baseyear_start = 1990
baseyear_end = 2019

# make fts
dm_avi = make_fts(dm_avi, "aviation", baseyear_start, baseyear_end, dim = "Variables")

# check
# dm_avi.filter({"Country" : ["EU27"]}).datamatrix_plot()

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

# change unit
dm_pop = DM_lfe["ots"]['pop']['lfs_population_'].copy()
dm_pop.append(DM_lfe["fts"]['pop']['lfs_population_'][1],"Years")
dm_pop.sort("Country")
dm_pop.sort("Years")
dm_pop = dm_pop.filter({"Country" : dm_avi.col_labels["Country"]})
dm_pop.sort("Country")
dm_avi_cap = dm_avi.copy()
dm_avi_cap.array = dm_avi_cap.array / dm_pop.array[...,np.newaxis]
dm_avi_cap.units["tra"] = "pkm/cap"
dm_avi_cap.rename_col("tra","tra_pkm-cap","Variables")

# check
# dm_avi_cap.filter({"Country" : ["EU27"]}).datamatrix_plot()

# split between ots and fts
DM_avi = {"ots": {"passenger_aviation-pkm" : []}, "fts": {"passenger_aviation-pkm" : dict()}}
DM_avi["ots"]["passenger_aviation-pkm"] = dm_avi_cap.filter({"Years" : years_ots})
for i in range(1,4+1):
    DM_avi["fts"]["passenger_aviation-pkm"][i] = dm_avi_cap.filter({"Years" : years_fts})

# save
f = os.path.join(current_file_directory, '../data/datamatrix/lever_passenger_aviation-pkm.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM_avi, handle, protocol=pickle.HIGHEST_PROTOCOL)

# save pkm as intermediate file
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/aviation_pkm.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_avi, handle, protocol=pickle.HIGHEST_PROTOCOL)









