# Import the Packages

import pandas as pd
import numpy as np
import pickle  # read/write the data in pickle
import json  # read the lever setting
import os  # operating system (e.g., look for workspace)

# Import Class
from model.common.data_matrix_class import DataMatrix  # Class for the model inputs
from model.common.constant_data_matrix_class import ConstantDataMatrix  # Class for the constant inputs
from model.common.auxiliary_functions import read_level_data
from model.common.interface_class import Interface

# ImportFunctions
from model.common.io_database import read_database_to_ots_fts_dict_w_groups  # read functions for levers & fixed assumptions
from model.common.auxiliary_functions import filter_geoscale


# filtering the constants & read csv and prepares it for the pickle format

# Lever setting for local purpose


def init_years_lever():
    # function that can be used when running the module as standalone to initialise years and levers
    years_setting = [1990, 2023, 2050, 5]
    f = open('../config/lever_position.json')
    lever_setting = json.load(f)[0]
    return years_setting, lever_setting


# Setting up the database in the module
def database_from_csv_to_datamatrix():
    
    # Set years range
    years_setting = [1990, 2015, 2100, 1]
    startyear = years_setting[0]
    baseyear = years_setting[1]
    lastyear = years_setting[2]
    step_fts = years_setting[3]
    years_ots = list(np.linspace(start=startyear, stop=baseyear, num=(baseyear-startyear)+1).astype(int)) 
    years_fts = list(np.linspace(start=baseyear+step_fts, stop=lastyear, num=int((lastyear-baseyear)/step_fts)).astype(int))
    years_all = years_ots + years_fts

    # Initiate the dictionary for ots & fts
    dict_ots = {}
    dict_fts = {}

    # Data - Lever - Climate temperature
    file = 'climate_temperature'
    lever = "temp"

    # Creates the datamatrix for lifestyles population
    dict_ots, dict_fts = read_database_to_ots_fts_dict_w_groups(file, lever, num_cat_list=[1, 0, 1, 0], baseyear=baseyear,
                                                                years=years_all, dict_ots=dict_ots, dict_fts=dict_fts,
                                                                column='eucalc-name', group_list=[
                                                                            'bld_climate-impact-space',
                                                                            'bld_climate-impact_average',
                                                                            'clm_capacity-factor',
                                                                            'clm_temp_global'])

    #  Create the data matrix for lifestyles
    DM_climate = {
        'fts': dict_fts,
        'ots': dict_ots
    }

    # Pickle
    current_file_directory = os.path.dirname(os.path.abspath(__file__))  # creates local path variable
    f = os.path.join(current_file_directory, '../_database/data/datamatrix/climate.pickle')
    # creates path variable for the pickle
    with open(f, 'wb') as handle:  # 'wb': writing binary / standard protocol for pickle
        pickle.dump(DM_climate, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return DM_climate

def read_data(data_file, lever_setting):
    
    # load dm
    with open(data_file, 'rb') as handle:
        DM_climate = pickle.load(handle)

    # get lever
    DM_ots_fts = read_level_data(DM_climate, lever_setting)

    return DM_ots_fts

def climate_buildings_interface(DM_ots_fts, write_pickle = False):
    
    # append
    dm_bld = DM_ots_fts["temp"]["bld_climate-impact-space"]
    
    # if write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, '../_database/data/interface/climate_to_buildings.pickle')
        with open(f, 'wb') as handle:
            pickle.dump(dm_bld, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # return
    return dm_bld

def variables_to_tpe(DM_ots_fts):
    
    dm_tpe = DM_ots_fts["temp"]["clm_capacity-factor"].flatten()
    dm_tpe.append(DM_ots_fts["temp"]["clm_temp_global"], "Variables")
    dm_tpe.flattest()
    
    return dm_tpe

def climate_power_interface(DM_ots_fts, write_pickle = False):
    
    dm = DM_ots_fts["temp"]["clm_capacity-factor"].copy()
    
    # if write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, '../_database/data/interface/climate_to_power.pickle')
        with open(f, 'wb') as handle:
            pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return dm

# CORE module
def climate(lever_setting, years_setting, interface = Interface(), calibration = False):
    
    # climate data file
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    climate_data_file = os.path.join(current_file_directory,'../_database/data/datamatrix/geoscale/climate.pickle')
    DM_ots_fts = read_data(climate_data_file, lever_setting)
    
    # tpe
    results_run = variables_to_tpe(DM_ots_fts)
    
    # interface buildings
    dm_bld = climate_buildings_interface(DM_ots_fts)
    interface.add_link(from_sector='climate', to_sector='buildings', dm=dm_bld)
    
    # interface power
    dm_pow = climate_power_interface(DM_ots_fts)
    interface.add_link(from_sector='climate', to_sector='power', dm=dm_pow)

    # TODO: interface water when water is ready

    return results_run


# Local run of lifestyles
def local_climate_run():
    
    # Initiate the year & lever setting
    years_setting, lever_setting = init_years_lever()

    # get geoscale
    global_vars = {'geoscale': 'EU27|Switzerland|Vaud'}
    filter_geoscale(global_vars['geoscale'])

    # run
    results_run = climate(lever_setting, years_setting)

    return results_run

# # local
# __file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/model/climate_module.py"
#results_run = local_climate_run()



