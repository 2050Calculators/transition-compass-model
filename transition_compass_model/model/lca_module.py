
from model.common.data_matrix_class import DataMatrix
from model.common.interface_class import Interface
from model.common.auxiliary_functions import filter_geoscale, cdm_to_dm, read_level_data
from model.common.auxiliary_functions import calibration_rates, cost
from model.common.auxiliary_functions import filter_country_and_load_data_from_pickles
import os
import json
import pickle
import numpy as np
import warnings
warnings.simplefilter("ignore")
import plotly.io as pio
pio.renderers.default='browser'

def read_data(DM_lca, lever_setting):

    # # get fxa
    # DM_fxa = DM_industry['fxa']

    # Get ots fts based on lever_setting
    DM_ots_fts = read_level_data(DM_lca, lever_setting)

    # # get calibration
    # dm_cal = DM_industry['calibration']

    # # get constants
    # CMD_const = DM_industry['constant']
    
    # return
    return DM_ots_fts

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
                if "Country" in list(DM[key].col_labels.keys()):
                    DM[key].filter({'Country': country_list}, inplace=True)
        else:
            if "Country" in list(DM[key].col_labels.keys()):
                DM.filter({'Country': country_list}, inplace=True)
    return DM

def get_footprint_by_group(DM_footprint):
    
    
    def reshape_and_store(DM_footprint, keyword):
    
        DM = {}
        dm = DM_footprint[keyword].copy()
        
        dm_veh = dm.filter_w_regex(({"Variables" : 'HDV_.*|LDV_.*|bus_.*|planes_.*|ships_.*|trains_.*'}))
        for v in dm_veh.col_labels["Variables"]:
            dm_veh.rename_col(v, f"{keyword}_" + v, "Variables")
        dm_veh.deepen(based_on="Variables")
        dm_veh.deepen(based_on="Variables")
        dm_veh.switch_categories_order("Categories1","Categories3")
        lastcat = list(dm_veh.col_labels.keys())[-1]
        if len(dm_veh.col_labels[lastcat]) == 1:
            dm_veh.group_all(lastcat)
        DM["vehicles"] = dm_veh.copy()
        
        def deepen_else(dm, keyword = keyword):
            
            lastcat = dm.dim_labels[-1]
            if len(dm.col_labels[lastcat]) == 1:
                dm.group_all(lastcat)
            
            for v in dm.col_labels["Variables"]:
                dm.rename_col(v, f"{keyword}_" + v, "Variables")
            dm.deepen(based_on="Variables")
            
            if len(dm.dim_labels) == 5:
                dm.switch_categories_order("Categories1","Categories2")
                
        # transport infrastructure
        dm_temp = dm.filter({"Variables" : ['rail','road','trolley-cables']})
        deepen_else(dm_temp)
        DM["infra-tra"] = dm_temp.copy()
        
        # domapp
        dm_temp = dm.filter(({"Variables" : ['dishwasher','fridge','wmachine']}))
        deepen_else(dm_temp)
        DM["domapp"] = dm_temp.copy()
        
        # electronics
        dm_temp = dm.filter(({"Variables" : ['computer','phone','tv']}))
        deepen_else(dm_temp)
        DM["electronics"] = dm_temp.copy()
        
        return DM
        
    DM_footprint_split = {}
    for keyword in DM_footprint.keys():
        DM_footprint_split[keyword] = reshape_and_store(DM_footprint, keyword)
    
    return DM_footprint_split

def get_footprint(footprint, DM_demand, DM_footprint):
    
    # vehicles
    dm_veh = DM_footprint["vehicles"].copy()
    dm_veh.units[footprint] = dm_veh.units[footprint].split("/")[0]
    if len(dm_veh.dim_labels) == 5:
        dm_veh.array = DM_demand["vehicles"].array * dm_veh.array
        dm_all = dm_veh.flatten()
    elif len(dm_veh.dim_labels) == 6:
        dm_veh.array = DM_demand["vehicles"].array[...,np.newaxis] * dm_veh.array
        dm_all = dm_veh.flatten().flatten()
        dm_all.deepen()
        
    def make_multiplication(dm_footprint, dm_demand):
        
        dm_temp = dm_footprint.copy()
        if len(dm_temp.dim_labels) == 4:
            dm_temp.array = dm_temp.array * dm_demand.array
        elif len(dm_temp.dim_labels) == 5:
            dm_temp.array = dm_temp.array * dm_demand.array[...,np.newaxis]
        dm_temp.units[footprint] = dm_temp.units[footprint].split("/")[0]
        
        return dm_temp
        
    # transport infra
    dm_temp = make_multiplication(DM_footprint["infra-tra"], DM_demand["tra-infra"])
    dm_all.append(dm_temp, "Categories1")
    
    # domapp
    dm_temp = make_multiplication(DM_footprint["domapp"], DM_demand["domapp"])
    dm_all.append(dm_temp, "Categories1")
    
    # electronics
    dm_temp = make_multiplication(DM_footprint["electronics"], DM_demand["electronics"])
    dm_all.append(dm_temp, "Categories1")
    
    # aggregate
    dm_all_agg = dm_all.copy() 
    dm_all_agg.group_all("Categories1")
    
    return dm_all_agg

def variables_for_tpe(DM_footprint_agg):
    
    dm_tpe = DM_footprint_agg["materials"].flatten()
    dm_tpe.append(DM_footprint_agg['ecological'], "Variables")
    dm_tpe.append(DM_footprint_agg['global-warming-potential'], "Variables")
    dm_tpe.append(DM_footprint_agg['water'], "Variables")
    dm_tpe.append(DM_footprint_agg['air-pollution'].flatten(), "Variables")
    dm_tpe.append(DM_footprint_agg['heavy-metals-in-soil'].flatten(), "Variables")
    dm_tpe.append(DM_footprint_agg['energy-demand'], "Variables")
    
    # # checks
    # DM_footprint_agg["materials"].datamatrix_plot(stacked=True)
    # DM_footprint_agg['ecological'].datamatrix_plot()
    # DM_footprint_agg['gwp'].datamatrix_plot()
    # DM_footprint_agg['water'].datamatrix_plot()
    # DM_footprint_agg['air-pollutant'].datamatrix_plot(stacked=True)
    # DM_footprint_agg['heavy-metals-to-soil'].datamatrix_plot(stacked=True)
    # DM_footprint_agg['energy-demand'].filter({"Variables" : ["energy-demand-elec"]}).datamatrix_plot()
    # DM_footprint_agg['energy-demand'].filter({"Variables" : ["energy-demand-ff"]}).datamatrix_plot()
    
    return dm_tpe

def lca(lever_setting, years_setting, DM_input, interface = Interface(), calibration = True):

    # industry data file
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    DM_ots_fts = read_data(DM_input, lever_setting)

    # get interfaces
    cntr_list = DM_ots_fts["footprint"]["materials"].col_labels['Country']
    DM_transport = get_interface(current_file_directory, interface, "transport", "lca", cntr_list)
    DM_buildings = get_interface(current_file_directory, interface, "buildings", "lca", cntr_list)
    DM_industry = get_interface(current_file_directory, interface, "industry", "lca", cntr_list)
    
    # split footrpint by product group
    DM_footprint = get_footprint_by_group(DM_ots_fts["footprint"])
    
    # split product demand
    dm_tra_infra_new = DM_transport['tra-infra'].filter({"Variables" : ["tra_product-demand"]})
    dm_bld_new = DM_buildings["floor-area"].filter({"Variables" : ["bld_floor-area_new"]})
    dm_domapp_new = DM_buildings["domapp"].filter({"Variables" : ["bld_domapp_new"], "Categories1" : ['dishwasher', 'fridge', 'wmachine']}) # TODO: add other ones when data will be available
    dm_elec_new = DM_buildings["electronics"].filter({"Variables" : ["bld_electronics_new"]})
    DM_demand = {"vehicles" : DM_transport['tra-veh'], 
                 "tra-infra": dm_tra_infra_new, 
                 "domapp" : dm_domapp_new, 
                 "electronics" : dm_elec_new}
    
    # get footprint
    DM_footprint_agg = {}
    for key in DM_footprint.keys():
        DM_footprint_agg[key] = get_footprint(key, DM_demand, DM_footprint[key])
    DM_footprint_agg["energy-demand"] = DM_footprint_agg['ene-demand-elec'].copy()
    DM_footprint_agg["energy-demand"].append(DM_footprint_agg["ene-demand-ff"], "Variables")
    del DM_footprint_agg['ene-demand-elec']
    del DM_footprint_agg['ene-demand-ff']
    
    # pass to TPE
    results_run = variables_for_tpe(DM_footprint_agg)
    
    # return
    return results_run

def local_lca_run():
    
    # Configures initial input for model run
    f = open('../config/lever_position.json')
    lever_setting = json.load(f)[0]
    years_setting = [1990, 2023, 2025, 2050, 5]

    country_list = ["EU27"]

    sectors = ['lca']
    
    # Filter geoscale
    # from database/data/datamatrix/.* reads the pickles, filters the geoscale, and loads them
    DM_input = filter_country_and_load_data_from_pickles(country_list= country_list, modules_list = sectors)

    # run
    results_run = lca(lever_setting, years_setting, DM_input['lca'])
    
    # return
    return results_run


# run local
if __name__ == "__main__": results_run = local_lca_run()
