
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
from model.common.auxiliary_functions import filter_country_and_load_data_from_pickles
import time
warnings.simplefilter("ignore")
import plotly.io as pio
pio.renderers.default='browser'

def read_data(DM_ammonia, lever_setting):

    # get fxa
    DM_fxa = DM_ammonia['fxa']

    # Get ots fts based on lever_setting
    DM_ots_fts = read_level_data(DM_ammonia, lever_setting)

    # get calibration
    dm_cal = DM_ammonia['calibration']

    # get constants
    CMD_const = DM_ammonia['constant']
    
    # return
    return DM_fxa, DM_ots_fts, dm_cal, CMD_const

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
                DM[key].filter({'Country': country_list}, inplace=True)
        else:
            DM.filter({'Country': country_list}, inplace=True)
    return DM

def product_production(dm_agriculture, dm_import):
    
    # net import [%] is net import [unit] / demand [unit]
    # production [unit] = demand [unit] - net import [unit]
    
    # buildings
    dm_netimport_fert = dm_import.copy()
    dm_netimport_fert.array = dm_netimport_fert.array * dm_agriculture.array
    dm_netimport_fert.units["product-net-import"] = dm_agriculture.units["agr_product-demand"]
    dm_prod_fert = dm_agriculture.copy()
    dm_prod_fert.array = dm_prod_fert.array - dm_netimport_fert.array
    dm_prod_fert.rename_col("agr_product-demand","product-production","Variables")
        
    # return
    return dm_prod_fert

def apply_material_decomposition(dm_production_fert, cdm_matdec_fert):
    
    countries = dm_production_fert.col_labels["Country"]
    years = dm_production_fert.col_labels["Years"]
    
    # material demand [t] = product production [unit] * material decomposition coefficient [t/unit]

    dm_bld_fert_matdec = cdm_to_dm(cdm_matdec_fert, countries, years)
    dm_bld_fert_matdec.array = dm_production_fert.array[...,np.newaxis] * dm_bld_fert_matdec.array
    dm_bld_fert_matdec.units['material-decomp'] = "t"
    dm_bld_fert_matdec = dm_bld_fert_matdec.filter({"Categories2" : ["ammonia"]}) # for now it's 100% ammonia, we'll see later if we'll add other types of fertilizers

    # return
    return dm_bld_fert_matdec

def material_production(dm_material_efficiency, dm_material_net_import, dm_material_demand):
    
    ######################
    ##### EFFICIENCY #####
    ######################
    
    dm_temp = dm_material_efficiency.copy()    
    dm_material_demand.array = dm_material_demand.array * (1 - dm_temp.array[...,np.newaxis])
    
    ############################
    ##### AGGREGATE DEMAND #####
    ############################

    # get aggregate demand
    dm_matdec_agg = dm_material_demand.group_all(dim='Categories1', inplace=False)
    dm_matdec_agg.change_unit('material-decomp', factor=1e-3, old_unit='t', new_unit='kt')

    ######################
    ##### PRODUCTION #####
    ######################

    # material production [kt] = material demand [kt] * (1 - material net import [%])
    # TODO: add this quantity to the material stock

    # get net import % and make production %
    dm_temp = dm_material_net_import.copy()
    dm_temp.array = 1 - dm_temp.array

    # get material production in units
    dm_material_production_bymat = dm_matdec_agg.copy()
    dm_material_production_bymat.array = dm_matdec_agg.array * dm_temp.array
    dm_material_production_bymat.rename_col(col_in = 'material-decomp', col_out = 'material-production', dim = "Variables")
    
    # get material net import in kilo tonnes
    dm_material_net_import_kt = dm_matdec_agg.copy()
    dm_material_net_import_kt.array = dm_material_net_import_kt.array * dm_material_net_import.array
 
    # put together
    DM_material_production = {"bymat" : dm_material_production_bymat, 
                              "material-net-import" : dm_material_net_import_kt}
    
    # return
    return DM_material_production

def calibration_material_production(DM_cal, dm_material_production_bymat, DM_material_production, 
                                    years_setting):
    
    # get calibration series
    dm_cal_sub = DM_cal["material-production"].copy()
    materials = dm_material_production_bymat.col_labels["Categories1"]
    dm_cal_sub.filter({"Categories1" : materials}, inplace = True)

    # get calibration rates
    DM_material_production["calib_rates_bymat"] = calibration_rates(dm = dm_material_production_bymat, dm_cal = dm_cal_sub,
                                                                    calibration_start_year = 1990, calibration_end_year = 2023,
                                                                    years_setting=years_setting)

    # do calibration
    dm_material_production_bymat.array = dm_material_production_bymat.array * DM_material_production["calib_rates_bymat"].array

    # clean
    del dm_cal_sub, materials
    
    return

def energy_demand(dm_material_production_bytech, CDM_const):
    
    # this is by material-technology and carrier

    # get energy demand for material production by technology both without and with feedstock
    feedstock = ["excl-feedstock", "feedstock"]
    DM_energy_demand = {}

    for f in feedstock:

        # get constants for energy demand for material production by technology
        cdm_temp = CDM_const["energy_" + f]
        
        # create dm for energy demand for material production by technology
        dm_energy_demand = cdm_to_dm(cdm_temp, 
                                     dm_material_production_bytech.col_labels["Country"], 
                                     dm_material_production_bytech.col_labels["Years"])
        
        # get energy demand for material production by technology
        dm_temp = dm_material_production_bytech.copy()
        dm_temp.change_unit('material-production', factor=1e-3, old_unit='kt', new_unit='Mt')
        dm_energy_demand.array = dm_energy_demand.array * dm_temp.array[...,np.newaxis]
        dm_energy_demand.units["energy-demand-" + f] = "TWh"
        DM_energy_demand[f] = dm_energy_demand

    # get overall energy demand
    dm_energy_demand_temp = DM_energy_demand["excl-feedstock"].copy()
    dm_energy_demand_temp.append(DM_energy_demand["feedstock"], dim = "Variables")
    dm_energy_demand_bytechcarr = DM_energy_demand["excl-feedstock"].copy()
    dm_energy_demand_bytechcarr.array = np.nansum(dm_energy_demand_temp.array, axis = -3, keepdims= True)
    dm_energy_demand_bytechcarr.rename_col(col_in = 'energy-demand-excl-feedstock', col_out = "energy-demand", dim = "Variables")
    dm_energy_demand_feedstock_bytechcarr = DM_energy_demand["feedstock"]

    DM_energy_demand = {"bytechcarr" : dm_energy_demand_bytechcarr, 
                        "feedstock_bytechcarr" : dm_energy_demand_feedstock_bytechcarr}
    
    # aggregate energy demand by energy carrier
    DM_energy_demand["bycarr"] = DM_energy_demand["bytechcarr"].group_all(dim='Categories1', inplace=False)

    # return
    return DM_energy_demand

def calibration_energy_demand(DM_cal, dm_energy_demand_bycarr, dm_energy_demand_bytechcarr, 
                              DM_energy_demand, years_setting):
    
    # this is by material-technology and carrier

    # get calibration rates
    dm_energy_demand_calib_rates_bycarr = calibration_rates(dm = dm_energy_demand_bycarr.copy(), 
                                                            dm_cal = DM_cal["energy-demand"].copy(), 
                                                            calibration_start_year = 2000, calibration_end_year = 2021,
                                                            years_setting=years_setting)

    # FIXME!: before 2000, instead of 1 put the calib rate of 2000 (it's done like this in the KNIME for industry, tbc what to do)
    idx = dm_energy_demand_calib_rates_bycarr.idx
    years_bef2000 = np.array(range(1990, 2000, 1)).tolist()
    for i in years_bef2000:
        dm_energy_demand_calib_rates_bycarr.array[:,idx[i],...] = dm_energy_demand_calib_rates_bycarr.array[:,idx[2000],...]

    # store dm_energy_demand_calib_rates_bycarr
    DM_energy_demand["calib_rates_bycarr"] = dm_energy_demand_calib_rates_bycarr

    # do calibration
    dm_energy_demand_bycarr.array = dm_energy_demand_bycarr.array * dm_energy_demand_calib_rates_bycarr.array

    # do calibration for each technology (by applying aggregate calibration rates)
    dm_energy_demand_bytechcarr.array = dm_energy_demand_bytechcarr.array * dm_energy_demand_calib_rates_bycarr.array[:,:,:,np.newaxis,:]

    # clean
    del idx, years_bef2000, dm_energy_demand_calib_rates_bycarr
        
    # return
    return

def technology_development(dm_technology_development, dm_energy_demand_bytechcarr):
    
    dm_temp = dm_energy_demand_bytechcarr.copy()

    # get energy demand after technology development (tech dev improves energy efficiency)
    dm_temp.array = dm_temp.array * (1 - dm_technology_development.array[...,np.newaxis])

    # return
    return dm_temp

def apply_energy_switch(dm_energy_carrier_mix, dm_energy_demand_bytechcarr):
    
    # this is by material-technology and carrier

    # energy demand for electricity [TWh] = (energy demand [TWh] * electricity share) + energy demand coming from switch to electricity [TWh]

    # get energy mix
    dm_temp = dm_energy_carrier_mix.copy()

    #######################
    ##### ELECTRICITY #####
    #######################

    carrier_in = dm_energy_demand_bytechcarr.col_labels["Categories2"].copy()
    carrier_in.remove("electricity")
    carrier_in.remove("hydrogen")
    energy_switch(dm_energy_demand = dm_energy_demand_bytechcarr, dm_energy_carrier_mix = dm_temp, 
                  carrier_in = carrier_in, carrier_out = "electricity", 
                  dm_energy_carrier_mix_prefix = "to-electricity")

    ####################
    ##### HYDROGEN #####
    ####################

    carrier_in = dm_energy_demand_bytechcarr.col_labels["Categories2"].copy()
    carrier_in.remove("electricity")
    carrier_in.remove("hydrogen")
    energy_switch(dm_energy_demand = dm_energy_demand_bytechcarr, dm_energy_carrier_mix = dm_temp, 
                  carrier_in = carrier_in, carrier_out = "hydrogen", 
                  dm_energy_carrier_mix_prefix = "to-hydrogen")

    ###############
    ##### GAS #####
    ###############

    energy_switch(dm_energy_demand = dm_energy_demand_bytechcarr, dm_energy_carrier_mix = dm_temp, 
                        carrier_in = ["solid-ff-coal"], carrier_out = "gas-ff-natural", 
                        dm_energy_carrier_mix_prefix = "solid-to-gas")

    energy_switch(dm_energy_demand = dm_energy_demand_bytechcarr, dm_energy_carrier_mix = dm_temp, 
                        carrier_in = ["liquid-ff-oil"], carrier_out = "gas-ff-natural", 
                        dm_energy_carrier_mix_prefix = "liquid-to-gas")


    ###########################
    ##### SYNTHETIC FUELS #####
    ###########################

    # TODO: TO BE DONE

    #####################
    ##### BIO FUELS #####
    #####################

    energy_switch(dm_energy_demand = dm_energy_demand_bytechcarr, dm_energy_carrier_mix = dm_temp, 
                        carrier_in = ["solid-ff-coal"], carrier_out = "solid-bio", 
                        dm_energy_carrier_mix_prefix = "to-biomass")

    energy_switch(dm_energy_demand = dm_energy_demand_bytechcarr, dm_energy_carrier_mix = dm_temp, 
                        carrier_in = ["liquid-ff-oil"], carrier_out = "liquid-bio", 
                        dm_energy_carrier_mix_prefix = "to-biomass")

    energy_switch(dm_energy_demand = dm_energy_demand_bytechcarr, dm_energy_carrier_mix = dm_temp, 
                        carrier_in = ["gas-ff-natural"], carrier_out = "gas-bio", 
                        dm_energy_carrier_mix_prefix = "to-biomass")

    # clean
    del dm_temp, carrier_in

    # return
    return

def add_specific_energy_demands(dm_energy_demand_bytechcarr, 
                                dm_energy_demand_feedstock_bytechcarr, DM_energy_demand, dict_groupby):

    # get demand for biomaterial from feedstock
    dm_energy_demand_feedstock_bycarr = dm_energy_demand_feedstock_bytechcarr.group_all("Categories1", inplace = False)
    dm_energy_demand_feedstock_bybiomat = \
        dm_energy_demand_feedstock_bycarr.filter({"Categories1" : ["solid-bio", 'gas-bio', 'liquid-bio']})

    # get demand for industrial waste
    dm_energy_demand_bycarr = dm_energy_demand_bytechcarr.group_all("Categories1", inplace = False)
    dm_energy_demand_indwaste = dm_energy_demand_bycarr.filter({"Categories1" : ['solid-waste']})

    # get demand for bioenergy solid, bioenergy gas, bioenergy liquid
    dm_energy_demand_bioener_bybiomat = dm_energy_demand_bycarr.filter({"Categories1" : ['solid-bio', 'gas-bio', 'liquid-bio']})
    dm_energy_demand_bioener_bybiomat.rename_col("energy-demand","energy-demand_bioenergy","Variables")
    dm_energy_demand_bioener = dm_energy_demand_bioener_bybiomat.group_all("Categories1", inplace = False)

    # get demand by material
    dm_energy_demand_bymatcarr = \
        dm_energy_demand_bytechcarr.groupby(dict_groupby, 
                                            dim='Categories1', aggregation = "sum", regex=True, inplace=False)
    dm_energy_demand_bymat = dm_energy_demand_bymatcarr.group_all("Categories2", inplace = False)

    # get demand by carrier
    dm_energy_demand_bycarr = dm_energy_demand_bymatcarr.group_all("Categories1", inplace = False)
    
    # get energy demand by tech
    dm_energy_demand_bytech = dm_energy_demand_bytechcarr.group_all("Categories2", inplace = False)

    # put in DM
    DM_energy_demand["bymatcarr"] = dm_energy_demand_bymatcarr
    DM_energy_demand["feedstock_bybiomat"] = dm_energy_demand_feedstock_bybiomat
    DM_energy_demand["indwaste"] = dm_energy_demand_indwaste
    DM_energy_demand["bioener_bybiomat"] = dm_energy_demand_bioener_bybiomat
    DM_energy_demand["bioener"] = dm_energy_demand_bioener

    DM_energy_demand["bymat"] = dm_energy_demand_bymat
    DM_energy_demand["bycarr"] = dm_energy_demand_bycarr
    DM_energy_demand["bytech"] = dm_energy_demand_bytech

    # clean
    del dm_energy_demand_bymatcarr, dm_energy_demand_feedstock_bybiomat, dm_energy_demand_indwaste, \
        dm_energy_demand_bioener, dm_energy_demand_bymat, dm_energy_demand_bioener_bybiomat, \
        dm_energy_demand_bycarr, dm_energy_demand_feedstock_bycarr

    # return
    return

def emissions(cdm_const_emission_factor_process, cdm_const_emission_factor, 
              dm_energy_demand_bytechcarr, dm_material_production_bytech):
    
    # get emission factors
    cdm_temp1 = cdm_const_emission_factor_process
    cdm_temp2 = cdm_const_emission_factor

    # emissions = energy demand * emission factor

    # combustion
    dm_emissions_combustion = dm_energy_demand_bytechcarr.copy()
    dm_emissions_combustion.rename_col('energy-demand', "emissions", "Variables")
    dm_emissions_combustion.units["emissions"] = "Mt"
    dm_emissions_combustion.rename_col("emissions", "emissions_CH4", "Variables")
    dm_emissions_combustion.deepen("_", based_on = "Variables")
    arr_temp = dm_emissions_combustion.array
    dm_emissions_combustion.add(arr_temp, "Categories3", "CO2")
    dm_emissions_combustion.add(arr_temp, "Categories3", "N2O")
    dm_emissions_combustion.array = dm_emissions_combustion.array * \
        cdm_temp2.array[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:]

    # biogenic total
    bio = ['gas-bio','liquid-bio','solid-bio']
    dm_emissions_combustion_bio = dm_emissions_combustion.filter({"Categories2" : bio}, inplace = False)
    dm_emissions_combustion_bio.group_all("Categories2")
    dm_emissions_combustion_bio.switch_categories_order("Categories2","Categories1")
    dm_emissions_combustion_bio.rename_col("emissions", "emissions-biogenic", dim = "Variables")

    # process
    dm_emissions_process = dm_material_production_bytech.copy()
    dm_emissions_process.change_unit('material-production', factor=1e-3, old_unit='kt', new_unit='Mt')
    dm_emissions_process.rename_col('material-production', "emissions-process_CH4", "Variables")
    dm_emissions_process.deepen("_", based_on = "Variables")
    arr_temp = dm_emissions_process.array
    dm_emissions_process.add(arr_temp, "Categories2", "CO2")
    dm_emissions_process.add(arr_temp, "Categories2", "N2O")
    dm_emissions_process.array = dm_emissions_process.array * cdm_temp1.array[np.newaxis,np.newaxis,...]

    # total emissions per technology
    dm_emissions_bygastech = dm_emissions_combustion.group_all("Categories2", inplace = False)
    dm_emissions_bygastech.append(dm_emissions_process, dim = "Variables")
    dm_emissions_bygastech.add(np.nansum(dm_emissions_bygastech.array, -3, keepdims=True), 
                               dim = "Variables", col_label = "emissions-total", unit = "Mt")
    dm_emissions_bygastech.drop("Variables", ['emissions', 'emissions-process'])
    dm_emissions_bygastech.rename_col("emissions-total","emissions", "Variables")
    dm_emissions_bygastech.switch_categories_order("Categories1","Categories2")

    # put in dict
    DM_emissions = {"bygastech" : dm_emissions_bygastech,
                    "combustion_bio" : dm_emissions_combustion_bio,
                    "bygastech_beforecc" : dm_emissions_bygastech}
    
    # return
    return DM_emissions

def carbon_capture(dm_ots_fts_cc, dm_emissions_bygastech, dm_emissions_combustion_bio, DM_emissions, 
                   dict_groupby):
    
    # get carbon capture
    dm_temp = dm_ots_fts_cc.copy()

    # subtract carbon captured to total CO2 emissions per technology
    idx = dm_emissions_bygastech.idx
    arr_temp = dm_emissions_bygastech.array[:,:,:,idx["CO2"],:] * (1 - dm_temp.array)
    dm_emissions_bygastech.add(arr_temp[:,:,:,np.newaxis,:], dim = "Categories1", col_label = "after-cc")

    # get emissions captured with carbon capture
    idx = dm_emissions_bygastech.idx
    arr_temp = dm_emissions_bygastech.array[:,:,:,idx["CO2"],:] - dm_emissions_bygastech.array[:,:,:,idx["after-cc"],:]
    dm_emissions_bygastech.add(arr_temp[:,:,:,np.newaxis,:], dim = "Categories1", col_label = "CO2-capt-w-cc")
    dm_emissions_capt_w_cc_bytech = dm_emissions_bygastech.filter({"Categories1" : ['CO2-capt-w-cc']})
    dm_emissions_capt_w_cc_bytech = dm_emissions_capt_w_cc_bytech.flatten()
    dm_emissions_capt_w_cc_bytech.rename_col_regex("CO2-capt-w-cc_", "", dim = "Categories1")
    dm_emissions_capt_w_cc_bytech.rename_col('emissions', "CO2-capt-w-cc", "Variables")
    dm_emissions_bygastech.drop("Categories1", "CO2")
    dm_emissions_bygastech.rename_col(col_in = 'after-cc', col_out = "CO2", dim = "Categories1")
    dm_emissions_bygastech.sort("Categories1")

    # get captured biogenic emissions
    dm_emissions_combustion_bio_capt_w_cc = dm_emissions_combustion_bio.copy()
    idx = dm_emissions_combustion_bio_capt_w_cc.idx
    arr_temp = dm_emissions_combustion_bio_capt_w_cc.array[:,:,:,idx["CO2"],:] * (1 - dm_temp.array)
    dm_emissions_combustion_bio_capt_w_cc.add(arr_temp[:,:,:,np.newaxis,:], dim = "Categories1", col_label = "after-cc")
    idx = dm_emissions_combustion_bio_capt_w_cc.idx
    arr_temp = dm_emissions_combustion_bio_capt_w_cc.array[:,:,:,idx["CO2"],:] - \
        dm_emissions_combustion_bio_capt_w_cc.array[:,:,:,idx["after-cc"],:]
    dm_emissions_combustion_bio_capt_w_cc.add(arr_temp[:,:,:,np.newaxis,:], dim = "Categories1", col_label = "capt-w-cc")
    dm_emissions_combustion_bio_capt_w_cc = dm_emissions_combustion_bio_capt_w_cc.filter({"Categories1" : ["capt-w-cc"]})
    dm_emissions_combustion_bio_capt_w_cc = dm_emissions_combustion_bio_capt_w_cc.flatten().flatten()
    dm_emissions_combustion_bio_capt_w_cc.deepen()
    dm_emissions_combustion_bio_capt_w_cc.rename_col(col_in = 'emissions-biogenic_capt-w-cc', 
                                                     col_out = "emissions-biogenic_CO2-capt-w-cc", dim = "Variables")

    # get these captured biogenic emissions by material
    dm_emissions_combustion_bio_capt_w_cc = \
        dm_emissions_combustion_bio_capt_w_cc.groupby(dict_groupby, 
                                                      dim='Categories1', 
                                                      aggregation = "sum", 
                                                      regex=True, inplace=False)

    # make negative captured biogenic emissions to supply to the climate module
    dm_emissions_combustion_bio_capt_w_cc_neg_bymat = dm_emissions_combustion_bio_capt_w_cc.copy()
    dm_emissions_combustion_bio_capt_w_cc_neg_bymat.array = dm_emissions_combustion_bio_capt_w_cc_neg_bymat.array * -1
    dm_emissions_combustion_bio_capt_w_cc_neg_bymat.rename_col("emissions-biogenic_CO2-capt-w-cc", 
                                                               "emissions-biogenic_CO2-capt-w-cc-negative", 
                                                               "Variables")

    # store
    DM_emissions["combustion_bio_capt_w_cc_neg_bymat"] = dm_emissions_combustion_bio_capt_w_cc_neg_bymat
    DM_emissions["capt_w_cc_bytech"] = dm_emissions_capt_w_cc_bytech
    
    # store also bygas (which is used in calibration if it's done)
    DM_emissions["bygas"] = dm_emissions_bygastech.group_all("Categories2", inplace = False)
        
    # return
    return

def calibration_emissions(DM_cal, dm_emissions_bygas, dm_emissions_bygastech, DM_emissions, years_setting):

    # get calibration rates
    DM_emissions["calib_rates_bygas"] = calibration_rates(dm = dm_emissions_bygas, 
                                                          dm_cal = DM_cal["emissions"],
                                                          calibration_start_year = 2008, calibration_end_year = 2023,
                                                          years_setting=years_setting)

    # do calibration
    dm_emissions_bygas.array = dm_emissions_bygas.array * DM_emissions["calib_rates_bygas"].array

    # do calibration for each technology (by applying aggregate calibration rates)
    dm_emissions_bygastech.array = dm_emissions_bygastech.array * DM_emissions["calib_rates_bygas"].array[:,:,:,:,np.newaxis]
    
    # return
    return

def compute_costs(dm_fxa_cost_matprod, dm_fxa_cost_cc, dm_material_production_bytech,
                  dm_emissions_capt_w_cc_bytech):

    ###############################
    ##### MATERIAL PRODUCTION #####
    ###############################

    # subset costs
    dm_cost_sub = dm_fxa_cost_matprod.copy()

    # get material production by technology
    keep = dm_fxa_cost_matprod.col_labels["Categories1"]
    dm_material_techshare_sub = dm_material_production_bytech.filter({"Categories1" : keep})
    dm_cost_sub.change_unit("capex-baseyear", factor=1e3, old_unit='EUR/t', new_unit='EUR/kt')
    dm_cost_sub.change_unit("capex-d-factor", factor=1e3, old_unit='num', new_unit='num')

    # get costs
    dm_material_techshare_sub_capex = cost(dm_activity = dm_material_techshare_sub, dm_cost = dm_cost_sub, cost_type = "capex")
    
    ######################################
    ##### EMISSIONS CAPTURED WITH CC #####
    ######################################

    # subset cdm
    dm_cost_sub = dm_fxa_cost_cc.copy()

    # get emissions captured with carbon capture
    keep = dm_fxa_cost_cc.col_labels["Categories1"]
    dm_emissions_capt_w_cc_sub = dm_emissions_capt_w_cc_bytech.filter({"Categories1" : keep})
    dm_emissions_capt_w_cc_sub.change_unit("CO2-capt-w-cc", factor=1e6, old_unit='Mt', new_unit='t')

    # get costs
    dm_emissions_capt_w_cc_sub_capex = cost(dm_activity = dm_emissions_capt_w_cc_sub, dm_cost = dm_cost_sub, cost_type = "capex")

    ########################
    ##### PUT TOGETHER #####
    ########################

    DM_cost = {"material-production_capex" : dm_material_techshare_sub_capex,
               "CO2-capt-w-cc_capex" : dm_emissions_capt_w_cc_sub_capex}
    
    # fix names
    for key in DM_cost.keys(): 
        cost_type = re.split("_", key)[1]
        activity_type = re.split("_", key)[0]
        DM_cost[key].filter({"Variables" : ["unit-cost",cost_type]}, inplace = True)
        DM_cost[key].rename_col("unit-cost", activity_type + "_" + cost_type + "-unit","Variables")
        DM_cost[key].rename_col(cost_type, activity_type + "_" + cost_type,"Variables")
    
    # make datamatrixes by material
    keys = list(DM_cost)
    for key in keys:
        materials = [i.split("-")[0] for i in DM_cost[key].col_labels["Categories1"]]
        materials = list(dict.fromkeys(materials))
        dict_groupby = {}
        for m in materials: dict_groupby[m] = ".*" + m + ".*"
        DM_cost[key + "_bymat"] = \
            DM_cost[key].groupby(dict_groupby, dim='Categories1', aggregation = "sum", regex=True, inplace=False)

    # return
    return DM_cost

def variables_for_tpe(dm_material_production_bymat, dm_ind_material_production, dm_energy_demand_bymat,
                      dm_ind_energy_demand, dm_energy_demand_bymatcarr):
    
    # production of chemicals (chem in ind + chem in ammonia)
    dm_tpe = dm_material_production_bymat.copy()
    dm_tpe.change_unit('material-production', factor=1e-3, old_unit='kt', new_unit='Mt')
    dm_tpe.append(dm_ind_material_production.copy(), "Categories1")
    dm_tpe.group_all("Categories1")
    dm_tpe.rename_col("material-production", "ind_material-production_chemicals", "Variables")
    
    # energy demand chemicals
    dm_temp = dm_energy_demand_bymat.copy()
    dm_temp.append(dm_ind_energy_demand.group_all("Categories2", inplace=False), "Categories1")
    dm_temp.group_all("Categories1")
    dm_temp.rename_col("energy-demand", "ind_energy-demand_chemicals", "Variables")
    dm_tpe.append(dm_temp, "Variables")
    
    # energy demand chemicals by energy carriers
    dm_temp = dm_energy_demand_bymatcarr.copy()
    dm_temp.append(dm_ind_energy_demand, "Categories1")
    dm_temp.group_all("Categories1")
    dm_temp.rename_col("energy-demand", "ind_energy-demand_chemicals", "Variables")
    dm_tpe.append(dm_temp.flatten(), "Variables")
    
    # # NOTE: FOR THE MOMENT THE CODE BELOW IS COMMENTED OUT, TO KEEP UNTIL WE FINALIZE THE TPE
    # # adjust variables' names
    # DM_cost["material-production_capex"].rename_col_regex("capex", "investment", "Variables")
    # DM_cost["material-production_capex"].rename_col('ammonia-amm-tech','amm-tech',"Categories1")
    # DM_cost["CO2-capt-w-cc_capex"].rename_col_regex("capex", "investment_CC", "Variables")
    # DM_cost["CO2-capt-w-cc_capex"].rename_col_regex("ammonia-amm-tech", "amm-tech", "Categories1")
    # DM_cost["material-production_opex"].rename_col_regex("opex", "operating-costs", "Variables")
    # DM_cost["material-production_opex"].rename_col('ammonia-amm-tech','amm-tech',"Categories1")
    # DM_cost["CO2-capt-w-cc_opex"].rename_col_regex("opex", "operating-costs_CC", "Variables")
    # DM_cost["CO2-capt-w-cc_opex"].rename_col_regex("ammonia-amm-tech", "amm-tech", "Categories1")
    # DM_emissions["bygas"] = DM_emissions["bygas"].flatten()
    # DM_emissions["bygas"].rename_col_regex("_","-","Variables")
    # variables = DM_material_production["bytech"].col_labels["Categories1"]
    # variables_new = [rename_tech_fordeepen(i) for i in variables]
    # for i in range(len(variables)):
    #     DM_material_production["bytech"].rename_col(variables[i], variables_new[i], dim = "Categories1")
    # DM_material_production["bymat"].array = DM_material_production["bymat"].array / 1000
    # DM_material_production["bymat"].units["material-production"] = "Mt"

    # # dm_tpe
    # dm_tpe = DM_emissions["bygas"].copy()
    # dm_tpe.append(DM_energy_demand["bymat"].flatten(), "Variables")
    # dm_tpe.append(DM_energy_demand["bycarr"].flatten(), "Variables")
    # dm_tpe.append(DM_cost["CO2-capt-w-cc_capex"].filter({"Variables" : ["investment_CC"]}).flatten(), "Variables")
    # dm_tpe.append(DM_cost["material-production_capex"].filter({"Variables" : ["investment"]}).flatten(), "Variables")
    # dm_tpe.append(DM_cost["CO2-capt-w-cc_opex"].filter({"Variables" : ["operating-costs_CC"]}).flatten(), "Variables")
    # dm_tpe.append(DM_cost["material-production_opex"].filter({"Variables" : ["operating-costs"]}).flatten(), "Variables")
    # dm_tpe.append(DM_material_production["bymat"].flatten(), "Variables")
    # variables = dm_tpe.col_labels["Variables"]
    # for i in variables:
    #     dm_tpe.rename_col(i, "amm_" + i, "Variables")
    # variables = ['amm_investment_CC_amm-tech', 'amm_investment_amm-tech', 
    #              'amm_operating-costs_CC_amm-tech', 'amm_operating-costs_amm-tech']
    # variables_new = ['ind_investment_CC_amm-tech', 'ind_investment_amm-tech', 
    #                  'ind_operating-costs_CC_amm-tech', 'ind_operating-costs_amm-tech']
    # for i in range(len(variables)):
    #     dm_tpe.rename_col(variables[i], variables_new[i], "Variables")
    # dm_tpe.sort("Variables")
    
    return dm_tpe

def ammonia_energy_interface(dm_energy_demand_by_carr, cdm_split, cdm_eneff, write_pickle = False):
    
    # split between electricity and lighting
    dm_temp = dm_energy_demand_by_carr.filter({"Categories1" : ["electricity"]})
    dm_temp.rename_col("electricity","lighting","Categories1")
    dm_energy_demand_by_carr.append(dm_temp,"Categories1")
    
    # reshape
    dm_temp = dm_energy_demand_by_carr.copy()
    dm_temp.drop("Categories1","lighting")
    for c in dm_temp.col_labels["Variables"]:
        dm_temp.rename_col(c, c + "_process-heat", "Variables")
    dm_temp.deepen("_","Variables")
    dm_temp.switch_categories_order("Categories1","Categories2")
    dm_temp[:,:,:,"electricity"] = 0
    dm_temp1 = dm_energy_demand_by_carr.filter({"Categories1" : ["lighting","electricity"]})
    dm_temp1.rename_col(["lighting","electricity"], ["lighting_electricity","elec_electricity"], "Categories1")
    dm_temp1.deepen()
    missing = dm_temp.col_labels["Categories2"].copy()
    missing.remove("electricity")
    for m in missing:
        dm_temp1.add(0, "Categories2", m, dummy=True)
    dm_temp1.sort("Categories2")
    dm_temp.append(dm_temp1, "Categories1")
    dm_temp.sort("Categories1")
    dm_energy_demand_by_carr_reshaped = dm_temp.copy()
    
    # get useful enery demand
    dm_useful_energy_demand_by_carr = dm_energy_demand_by_carr_reshaped.copy()
    dm_useful_energy_demand_by_carr.array = dm_useful_energy_demand_by_carr.array * cdm_eneff[np.newaxis,np.newaxis,...]
    
    # rename energy carriers to match energy ones
    dm_useful_energy_demand_by_carr.rename_col(
        ['electricity', 'gas-bio', 'gas-ff-natural', 'hydrogen', 'liquid-bio', 'solid-bio', 'solid-ff-coal', 'solid-waste'], 
        ['electricity', 'biogas',  'gas',            'other',    'renewables', 'biomass',   'coal',          'waste'], 
        "Categories2")
    dm_useful_energy_demand_by_carr.groupby({"heating-oil" : ['liquid-ff-diesel', 'liquid-ff-oil']},"Categories2",inplace=True)
    
    # for the moment add zero for the missing carrier
    # TODO: in pre-processing, you could get constants to get 'district-heating', 'heat-pump' and 'solar'
    # out of electricity (for the moment they have been aggregated to electricity). For 'wood', probably
    # a constant that separates it from biomass, but that's not in JRC. And 'nuclear-fuel' also not in JRC,
    # not sure how we could do itn (probably with some nuclear mix of how electricity is produced by country).
    missing = ['district-heating', 'heat-pump', 'nuclear-fuel', 'solar', 'wood']
    for m in missing:
        dm_useful_energy_demand_by_carr.add(0, "Categories2", m, dummy=True)
    dm_useful_energy_demand_by_carr.sort("Categories2")
    
    # add zero for 'hot-water' and 'space-heating'
    # TODO: they will either come from bld or we'll do something by tonne in industry
    dm_useful_energy_demand_by_carr.add(0, "Categories1", "hot-water", dummy=True)
    dm_useful_energy_demand_by_carr.add(0, "Categories1", "space-heating", dummy=True)
    dm_useful_energy_demand_by_carr.sort("Categories1")
    dm_useful_energy_demand_by_carr.rename_col('energy-demand', 'ind_energy-end-use', "Variables")
    
    # pass energy efficiency ratios
    cmd_temp = cdm_eneff.copy()
    missing = ['district-heating', 'heat-pump', 'nuclear-fuel', 'solar'] # for these ones assumed to be same of electricity
    for m in missing:
        cmd_temp.add(cmd_temp[...,"electricity"], "Categories2", m)
    cmd_temp.add(cmd_temp[...,"solid-bio"], "Categories2", "wood") # for wood assumed to be the same of biomass
    cmd_temp.rename_col(
        ['electricity', 'gas-bio', 'gas-ff-natural', 'hydrogen', 'liquid-bio', 'solid-bio', 'solid-ff-coal', 'solid-waste'], 
        ['electricity', 'biogas',  'gas',            'other',    'renewables', 'biomass',   'coal',          'waste'], 
        "Categories2")
    cmd_temp.groupby({"heating-oil" : ['liquid-ff-diesel', 'liquid-ff-oil']},"Categories2",inplace=True, aggregation="mean")
    cmd_temp.sort("Categories2")
    cmd_temp.add(cmd_temp[:,"process-heat",:], "Categories1", "hot-water") # put same of process heat for now
    cmd_temp.add(cmd_temp[:,"process-heat",:], "Categories1", "space-heating") # put same of process heat for now
    cmd_temp.sort("Categories1")
    
    DM_ene = {'ind-energy-demand' : dm_useful_energy_demand_by_carr,
              "ind-energy-efficiency-const" : cmd_temp}

    # of write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, '../_database/data/interface/ammonia_to_energy.pickle')
        with open(f, 'wb') as handle:
            pickle.dump(DM_ene, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # return
    return DM_ene

def ammonia_emissions_interface(DM_emissions, write_pickle = False):
    
    # adjust variables' names
    dm_temp = DM_emissions["bygasmat"].flatten().flatten()
    dm_temp.deepen()
    dm_temp.rename_col_regex("_","-","Variables")

    # dm_cli
    dm_ems = dm_temp.flatten()
    variables = dm_ems.col_labels["Variables"]
    for i in variables:
        dm_ems.rename_col(i, "amm_" + i, "Variables")
    dm_ems.sort("Variables")

    # write
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, '../_database/data/interface/ammonia_to_emissions.pickle')
        with open(f, 'wb') as handle:
            pickle.dump(dm_ems, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # return
    return dm_ems

def ammonia(lever_setting, years_setting, DM_input, interface = Interface(), calibration = False):

    # ammonia data file
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    DM_fxa, DM_ots_fts, DM_cal, CDM_const = read_data(DM_input, lever_setting)

    # get interfaces
    cntr_list = DM_ots_fts["product-net-import"].col_labels['Country']
    dm_agriculture = get_interface(current_file_directory, interface, "agriculture", "ammonia", cntr_list)
    DM_industry = get_interface(current_file_directory, interface, "industry", "ammonia", cntr_list)

    # get product import
    dm_import = DM_ots_fts["product-net-import"]
    
    # get product production
    dm_production_fert = product_production(dm_agriculture, dm_import)
    
    # get material demand
    dm_material_demand = apply_material_decomposition(dm_production_fert,
                                                      CDM_const["material-decomposition_fertilizer"])
    
    # get material production
    DM_material_production = material_production(DM_ots_fts['material-efficiency'], DM_ots_fts['material-net-import'], 
                                                 dm_material_demand)

    # calibrate material production (writes in DM_material_production)
    if calibration is True:
        calibration_material_production(DM_cal, DM_material_production["bymat"], DM_material_production,
                                        years_setting)
        
    # end of life
    # note: for the moment we do not do recycled ammonia
        
    # get material production by technology (writes in DM_material_production)
    dm_temp = DM_material_production["bymat"].copy()
    dm_temp.rename_col("ammonia", "ammonia-tech", "Categories1")
    DM_material_production["bytech"] = dm_temp.copy()
    
    # get energy demand for material production
    DM_energy_demand = energy_demand(DM_material_production["bytech"], CDM_const)
    
    # calibrate energy demand for material production (writes in DM_energy_demand)
    if calibration is True:
        calibration_energy_demand(DM_cal, DM_energy_demand["bycarr"], DM_energy_demand["bytechcarr"], 
                                  DM_energy_demand, years_setting)
        
    # compute energy demand for material production after taking into account technology development (writes in DM_energy_demand)
    technology_development(DM_ots_fts['technology-development'], DM_energy_demand["bytechcarr"])
    
    # do energy switch (writes in DM_energy_demand["bytechcarr"])
    apply_energy_switch(DM_ots_fts['energy-carrier-mix'], DM_energy_demand["bytechcarr"])
    
    # do dictionary to sum across technologies by materials
    materials = [i.split("-")[0] for i in DM_energy_demand["bytechcarr"].col_labels["Categories1"]]
    materials = list(dict.fromkeys(materials))
    dict_groupby = {}
    for m in materials: dict_groupby[m] = ".*" + m + ".*"
    
    # compute specific energy demands that will be used for tpe (writes in DM_energy_demand)
    add_specific_energy_demands(DM_energy_demand["bytechcarr"], 
                                DM_energy_demand["feedstock_bytechcarr"], DM_energy_demand, dict_groupby)
    
    # get emissions
    DM_emissions = emissions(CDM_const["emission-factor-process"], CDM_const["emission-factor"], 
                             DM_energy_demand["bytechcarr"], DM_material_production["bytech"])
    
    # compute captured carbon (writes in DM_emissions)
    carbon_capture(DM_ots_fts['cc'], DM_emissions["bygastech"], DM_emissions["combustion_bio"], 
                   DM_emissions, dict_groupby)
    
    # calibrate emissions (writes in DM_emissions)
    if calibration is True:
        calibration_emissions(DM_cal, DM_emissions["bygas"], DM_emissions["bygastech"], 
                              DM_emissions, years_setting)
    
    # comute specific groups of emissions that will be used for tpe (writes in DM_emissions)
    # emissions with different techs
    DM_emissions["bygasmat"] = DM_emissions["bygastech"].groupby(dict_groupby, dim='Categories2', 
                                                                 aggregation = "sum", regex=True, inplace=False)

    # get costs (capex and opex) for material production and carbon catpure
    DM_cost = compute_costs(DM_fxa["cost-matprod"], DM_fxa["cost-CC"], 
                            DM_material_production["bytech"], DM_emissions["capt_w_cc_bytech"])
    
    # get variables for tpe (also writes in DM_cost, dm_bld_matswitch_savings_bymat, DM_emissions and DM_material_production for renaming)
    results_run = variables_for_tpe(DM_material_production["bymat"], DM_industry["material-production"],  
                                    DM_energy_demand["bymat"],
                                    DM_industry["energy-demand"], DM_energy_demand["bymatcarr"])
    
    # interface energy
    DM_ene = ammonia_energy_interface(DM_energy_demand["bycarr"], 
                                      CDM_const['energy_excl-feedstock_eleclight-split'],
                                      CDM_const['energy_efficiency'])
    interface.add_link(from_sector='ammonia', to_sector='energy', dm=DM_ene)
    
    # interface emissions
    dm_ems = ammonia_emissions_interface(DM_emissions)
    interface.add_link(from_sector='ammonia', to_sector='emissions', dm=dm_ems)
    
    # # interface refinery
    # dm_refinery = ammonia_refinery_interface(DM_energy_demand)
    # interface.add_link(from_sector='ammonia', to_sector='oil-refinery', dm=dm_refinery)
    
    # # interface water
    # dm_water = ammonia_water_inferface(DM_energy_demand, DM_material_production)
    # interface.add_link(from_sector='ammonia', to_sector='water', dm=dm_water)
    
    # # interface ccus
    # dm_ccus = ammonia_ccus_interface(DM_emissions)
    # interface.add_link(from_sector='ammonia', to_sector='ccus', dm=dm_ccus)
    
    # # interface air pollution
    # dm_airpoll = ammonia_airpollution_interface(DM_material_production, DM_energy_demand)
    # interface.add_link(from_sector='ammonia', to_sector='air-pollution', dm=dm_airpoll)
    
    # return
    return results_run
    
def local_ammonia_run():
    
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
    country_list = ['EU27', 'Switzerland', 'Vaud']
    DM_input = filter_country_and_load_data_from_pickles(country_list= country_list, modules_list = 'ammonia')

    # run
    results_run = ammonia(lever_setting, years_setting, DM_input['ammonia'])
    
    # return
    return results_run

# run local
if __name__ == "__main__": results_run = local_ammonia_run()


