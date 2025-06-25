
from model.common.data_matrix_class import DataMatrix
from model.common.interface_class import Interface
from model.common.auxiliary_functions import filter_geoscale, cdm_to_dm, read_level_data
from model.common.auxiliary_functions import calibration_rates, cost
from model.common.auxiliary_functions import material_switch, energy_switch
import pickle
import json
import os
import numpy as np
import re
import warnings
import time
warnings.simplefilter("ignore")

def read_data(data_file, lever_setting):
    
    # load dm
    with open(data_file, 'rb') as handle:
        DM_minerals = pickle.load(handle)

    # # get fxa
    # DM_fxa = DM_minerals['fxa']

    # Get ots fts based on lever_setting
    DM_ots_fts = read_level_data(DM_minerals, lever_setting)

    # # get calibration
    # dm_cal = DM_minerals['calibration']

    # get constants
    CMD_const = DM_minerals['constant']

    # clean
    del handle, DM_minerals, data_file, lever_setting
    
    # return
    return DM_ots_fts, CMD_const

def get_interface(current_file_directory, interface, from_sector, to_sector, country_list):
    
    if interface.has_link(from_sector=from_sector, to_sector=to_sector):
        DM = interface.get_link(from_sector=from_sector, to_sector=to_sector)
    else:
        if len(interface.list_link()) != 0:
            print("You are missing " + from_sector + " to " + to_sector + " interface")
        filepath = os.path.join(current_file_directory, '../_database/data/interface/' + from_sector + "_to_" + to_sector + '.pickle')
        with open(filepath, 'rb') as handle:
            DM = pickle.load(handle)
        if type(DM) is dict:
            for key in DM.keys():
                if type(DM[key]) is dict:
                    for key2 in DM[key].keys():
                        DM[key][key2].filter({'Country': country_list}, inplace=True)
                else:        
                    DM[key].filter({'Country': country_list}, inplace=True)
        else:
            DM.filter({'Country': country_list}, inplace=True)
    return DM

def eol_battery(dm_bev_eol, cdm_bev_matdec):
    
    # create batteries in eol assuming 1 battery per car
    dm_lib_eol = dm_bev_eol.copy()
    dm_lib_eol.array = dm_bev_eol.array * 1
    dm_lib_eol = dm_lib_eol.flatten()
    dm_lib_eol.rename_col("LDV_BEV","battery-lion-LDV-BEV","Categories1")

    # create datamatrix for equation 2
    cdm_temp = cdm_bev_matdec.flatten().flatten()
    cdm_temp.deepen()
    arr_temp = dm_lib_eol.array[...,np.newaxis] * cdm_temp.array[np.newaxis,np.newaxis,...]
    dm_lib_matrec = DataMatrix.based_on(arr_temp, dm_lib_eol,
                                        {"Categories2": cdm_temp.col_labels["Categories2"]},
                                        units = "t")
    
    
    
    return

def minerals(lever_setting, years_setting, interface = Interface(), calibration = False):
    
    # minerals data file
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    minerals_data_file = os.path.join(current_file_directory, '../_database/data/datamatrix/geoscale/minerals_new.pickle')
    DM_ots_fts, CDM_const = read_data(minerals_data_file, lever_setting)
    
    # get interfaces
    cntr_list = DM_ots_fts["eol-material-recovery"].col_labels['Country']
    DM_industry = get_interface(current_file_directory, interface, "industry", "minerals", cntr_list)
    
    # batteries eol
    eol_battery(DM_industry["veh-to-recycling"].filter({"Categories1": ["LDV"], "Categories2": ["BEV"]}),
                CDM_const['material-decomposition_bat'].filter({"Categories1" : ["battery-lion-LDV"],
                                                                "Categories2" : ["BEV"]}))
    
    # DM_ots_fts["eol-material-recovery"].filter({"Categories1":["battery-lion"]})
    
    
    # to be done:
    
    # cradle-to-gate material decomp of vehicles for EU27 (HDV, LDV, bus)
    
    return

def local_minerals_run():
    
    # get years and lever setting
    years_setting = [1990, 2023, 2025, 2050, 5]
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(current_file_directory, '../config/lever_position.json'))
    lever_setting = json.load(f)[0]
    # lever_setting["lever_energy-carrier-mix"] = 3
    # lever_setting["lever_cc"] = 3
    # lever_setting["lever_material-switch"] = 3
    # lever_setting["lever_technology-share"] = 4
    
    # get geoscale
    global_vars = {'geoscale': 'EU27|Switzerland|Vaud'}
    filter_geoscale(global_vars)

    # run
    results_run = minerals(lever_setting, years_setting)
    
    # return
    return results_run

# # run local
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/model/minerals_module_new.py"
# start = time.time()
results_run = local_minerals_run()
# end = time.time()
# print(end-start)


