# Import the Packages

import pandas as pd
import numpy as np
import pickle  # read/write the data in pickle
import json  # read the lever setting
import os  # operating system (e.g., look for workspace)

# Import Class
from model.common.data_matrix_class import DataMatrix  # Class for the model inputs
from model.common.constant_data_matrix_class import ConstantDataMatrix  # Class for the constant inputs
from model.common.interface_class import Interface

# ImportFunctions
from model.common.auxiliary_functions import read_level_data, filter_geoscale, my_pickle_dump
from model.common.auxiliary_functions import filter_country_and_load_data_from_pickles


# init years and lever
def init_years_lever():
    # function that can be used when running the module as standalone to initialise years and levers
    years_setting = [1990, 2015, 2050, 5]
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(current_file_directory, '../config/lever_position.json'))
    lever_setting = json.load(f)[0]
    return years_setting, lever_setting


#  Reading the Pickle
def read_data(DM_lfs, lever_setting):

    # Get ots fts based on lever_setting
    DM_ots_fts = read_level_data(DM_lfs, lever_setting)
    
    # return
    return DM_ots_fts


# CORE module
def lifestyles(lever_setting, years_setting, DM_input, interface=Interface(), write_pickle = False):

    # get population data
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    DM_pop = read_data(DM_input, lever_setting)
    dm_pop = DM_pop['pop']["lfs_population_"]

    # send population to agriculture
    if write_pickle is True:
        f = os.path.join(current_file_directory, '../_database/data/interface/lifestyles_to_agriculture.pickle')
        my_pickle_dump(DM_new=DM_pop['pop'], local_pickle_file=f)
    interface.add_link(from_sector='lifestyles', to_sector='agriculture', dm=DM_pop['pop'])
    
    # send population to transport
    if write_pickle is True:
        f = os.path.join(current_file_directory, '../_database/data/interface/lifestyles_to_transport.pickle')
        my_pickle_dump(DM_new={'pop': dm_pop}, local_pickle_file=f)
    interface.add_link(from_sector='lifestyles', to_sector='transport', dm={'pop': dm_pop})
    
    # send population to buildings
    if write_pickle is True:
        f = os.path.join(current_file_directory, '../_database/data/interface/lifestyles_to_buildings.pickle')
        my_pickle_dump(DM_new={'pop': dm_pop}, local_pickle_file=f)
    interface.add_link(from_sector='lifestyles', to_sector='buildings', dm={'pop': dm_pop})
    
    # send population to industry
    if write_pickle is True:
        f = os.path.join(current_file_directory, '../_database/data/interface/lifestyles_to_industry.pickle')
        my_pickle_dump(DM_new={'pop': dm_pop}, local_pickle_file=f)
    interface.add_link(from_sector='lifestyles', to_sector='industry', dm={'pop': dm_pop})

    # dm_minerals = DM_industry['macro']
    # dm_minerals.append(DM_industry['population'], dim='Variables')
    # interface.add_link(from_sector='lifestyles', to_sector='minerals', dm=dm_minerals)

    return dm_pop


# Local run of lifestyles
def local_lifestyles_run(write_pickle=False):
    # Initiate the year & lever setting
    years_setting, lever_setting = init_years_lever()

    country_list = ['EU27', 'Switzerland', 'Vaud']
    DM_input = filter_country_and_load_data_from_pickles(country_list= country_list, modules_list = 'lifestyles')

    lifestyles(lever_setting, years_setting, DM_input['lifestyles'], write_pickle=write_pickle)
    return

# Update/Create the Pickle
# __file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/model/lifestyles_module.py"
#local_lifestyles_run()  # to un-comment to run in local

