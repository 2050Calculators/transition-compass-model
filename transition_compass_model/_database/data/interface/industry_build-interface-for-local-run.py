
# packages
from model.common.data_matrix_class import DataMatrix
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

# file
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/data/interface/industry_build-interface-for-local-run.py"

# directories
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# def simulate input
def simulate_input(from_sector, to_sector, num_cat = 0):
    
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    
    # get file
    xls_directory = os.path.join(current_file_directory, "../xls")
    files = np.array(os.listdir(xls_directory))
    file = files[[bool(re.search(from_sector + "-to-" + to_sector + ".xlsx", str(i))) for i in files]].tolist()[0]
    xls_file_directory = xls_directory +  "/" + file
    df = pd.read_excel(xls_file_directory)
    
    # get dm
    dm = DataMatrix.create_from_df(df, num_cat=num_cat)
    return dm

#####################
##### TRANSPORT #####
#####################

# transport
dm_transport = simulate_input(from_sector='transport', to_sector='industry')

# rename
# dm_transport.rename_col_regex(str1 = "tra_", str2 = "tra_product-demand_", dim = "Variables")
dm_transport.deepen()

# infra
dm_demand_tra_infra = dm_transport.filter({"Variables" : ['tra_product-demand'],
                                           "Categories1" : ['rail','road','trolley-cables']})
dm_demand_tra_infra.units["tra_product-demand"] = "km"

# vehicules
dm_demand_tra_veh = dm_transport.filter({"Variables" : ['tra_product-demand'],
                                         "Categories1" : ['cars-EV', 'cars-FCV', 'cars-ICE',
                                                          'planes', 'ships', 'trains',
                                                          'trucks-EV', 'trucks-FCV', 'trucks-ICE']})
dm_demand_tra_veh.units["tra_product-demand"] = "num"

# waste
dm_waste_veh =  dm_transport.filter({"Variables" : ['tra_product-waste'],
                                     "Categories1" : ['cars-EV', 'cars-FCV', 'cars-ICE',
                                                      'planes', 'ships', 'trains',
                                                      'trucks-EV', 'trucks-FCV', 'trucks-ICE']})
dm_waste_veh.units["tra_product-waste"] = "num"

# stock
dm_stock_veh =  dm_transport.filter({"Variables" : ['tra_product-stock'],
                                     "Categories1" : ['cars-EV', 'cars-FCV', 'cars-ICE',
                                                      'planes', 'ships', 'trains',
                                                      'trucks-EV', 'trucks-FCV', 'trucks-ICE']})
dm_stock_veh.units["tra_product-stock"] = "num"


# put together
DM_transport = {
    "tra-infra-demand": dm_demand_tra_infra,
    "tra-veh-demand": dm_demand_tra_veh,
    "tra-veh-waste" : dm_waste_veh,
    "tra-veh-stock" : dm_stock_veh
}

# add missing years
years = list(range(1990, 2023+1))
years_current = dm_demand_tra_infra.col_labels["Years"]
years_missing = list(np.array(years)[[i not in years_current for i in years]])
for key in DM_transport.keys():
    for y in years_missing:
        DM_transport[key].add(np.nan,"Years",[y],dummy=True)
    DM_transport[key].fill_nans(dim_to_interp='Years')

# drop paris and canton vaud
for key in DM_transport.keys():
    DM_transport[key].drop("Country",["Paris"])

# save
f = os.path.join(current_file_directory, 'transport_to_industry.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM_transport, handle, protocol=pickle.HIGHEST_PROTOCOL)


######################
##### LIFESTYLES #####
######################

dm_lifestyles = simulate_input(from_sector='lifestyles', to_sector='industry')
dm_lifestyles.rename_col_regex(str1="lfs_", str2="lfs_product-demand_", dim="Variables")
dm_lifestyles.deepen()

# add missing years
years_current = dm_lifestyles.col_labels["Years"]
years_missing = list(np.array(years)[[i not in years_current for i in years]])
for y in years_missing:
    dm_lifestyles.add(np.nan,"Years",[y],dummy=True)
dm_lifestyles.fill_nans(dim_to_interp='Years')

# drop paris and canton vaud
dm_lifestyles.drop("Country",["Paris"])

# save
f = os.path.join(current_file_directory, 'lifestyles_to_industry.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_lifestyles, handle, protocol=pickle.HIGHEST_PROTOCOL)


#####################
##### BUILDINGS #####
#####################

dm_buildings = simulate_input(from_sector='buildings', to_sector='industry')

# rename
dm_buildings.rename_col_regex(str1 = "bld_", str2 = "bld_product-demand_", dim = "Variables")
dm_buildings.rename_col_regex(str1 = "_new_",str2 = "-new-",dim = "Variables")
dm_buildings.rename_col_regex(str1 = "_reno_",str2 = "-reno-",dim = "Variables")
dm_buildings.rename_col_regex(str1 = "-new-dhg_pipe",str2 = "_new-dhg-pipe",dim = "Variables")

# deepen
dm_buildings.deepen()

# pipes
dm_demand_bld_pipe = dm_buildings.filter_w_regex({"Categories1" : ".*pipe"})
dm_demand_bld_pipe.units["bld_product-demand"] = "km"

# floor
dm_demand_bld_floor = dm_buildings.filter_w_regex({"Categories1" : ".*floor"})
dm_demand_bld_floor.units["bld_product-demand"] = "m2"

# domestic appliances
dm_demand_bld_domapp = dm_buildings.filter({"Categories1" : ['computer', 'dishwasher', 'dryer',
                                                              'freezer', 'fridge', 'phone',
                                                              'tv', 'wmachine']})
dm_demand_bld_domapp.units["bld_product-demand"] = "num"

DM_buildings = {
    "bld-pipe-demand" : dm_demand_bld_pipe,
    "bld-floor-demand" : dm_demand_bld_floor,
    "bld-domapp-demand" : dm_demand_bld_domapp
}

# add missing years
years_current = dm_demand_bld_pipe.col_labels["Years"]
years_missing = list(np.array(years)[[i not in years_current for i in years]])
for key in DM_buildings.keys():
    for y in years_missing:
        DM_buildings[key].add(np.nan,"Years",[y],dummy=True)
    DM_buildings[key].fill_nans(dim_to_interp='Years')

# drop paris and canton vaud
for key in DM_buildings.keys():
    DM_buildings[key].drop("Country",["Paris"])

# save
f = os.path.join(current_file_directory, 'buildings_to_industry.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM_buildings, handle, protocol=pickle.HIGHEST_PROTOCOL)














