
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
        DM_industry = pickle.load(handle)

    # get fxa
    DM_fxa = DM_industry['fxa']

    # Get ots fts based on lever_setting
    DM_ots_fts = read_level_data(DM_industry, lever_setting)

    # get calibration
    dm_cal = DM_industry['calibration']

    # get constants
    CMD_const = DM_industry['constant']

    # clean
    del handle, DM_industry, data_file, lever_setting
    
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

def product_production(dm_demand_bld_floor, dm_demand_tra_infra, dm_demand_tra_veh, 
                       dm_pop, dm_demand_packaging_percapita, 
                       dm_import):
    
    # net import [%] is net import [unit] / demand [unit]
    # production [unit] = demand [unit] - net import [unit]
    
    # buildings
    variabs = dm_demand_bld_floor.col_labels["Variables"]
    for v in variabs:
        dm_demand_bld_floor.rename_col(v, "product-demand_" + v, "Variables")
    dm_demand_bld_floor.deepen()
    dm_netimport_bld_floor = dm_import.filter_w_regex({"Categories1": "floor-area"})
    dm_netimport_bld_floor.array = dm_netimport_bld_floor.array * dm_demand_bld_floor.array
    dm_netimport_bld_floor.units["product-net-import"] = dm_demand_bld_floor.units["product-demand"]
    dm_prod_bld_floor = dm_demand_bld_floor.copy()
    dm_prod_bld_floor.array = dm_demand_bld_floor.array - dm_netimport_bld_floor.array
    dm_prod_bld_floor.rename_col("product-demand","product-production","Variables")
    
    # transport infra
    dm_netimport_tra_infra = dm_import.filter({"Categories1": ['rail', 'road', 'trolley-cables']})
    dm_netimport_tra_infra.array = dm_netimport_tra_infra.array * dm_demand_tra_infra.array
    dm_netimport_tra_infra.units["product-net-import"] = dm_demand_tra_infra.units["tra_product-demand"]
    dm_prod_tra_infra = dm_demand_tra_infra.copy()
    dm_prod_tra_infra.array = dm_demand_tra_infra.array - dm_netimport_tra_infra.array
    dm_prod_tra_infra.rename_col("tra_product-demand","product-production","Variables")
    
    # transport vehicles
    dm_netimport_tra_veh = dm_import.filter_w_regex({"Categories1": "HDV|LDV|bus|planes|ships|trains"})
    dm_netimport_tra_veh.deepen()
    dm_netimport_tra_veh.array = dm_netimport_tra_veh.array * dm_demand_tra_veh.array
    dm_netimport_tra_veh.units["product-net-import"] = dm_demand_tra_veh.units["tra_product-demand"]
    dm_prod_tra_veh = dm_demand_tra_veh.copy()
    dm_prod_tra_veh.array = dm_demand_tra_veh.array - dm_netimport_tra_veh.array
    dm_prod_tra_veh.rename_col("tra_product-demand","product-production","Variables")
    
    # packaging
    dm_demand_pack = dm_demand_packaging_percapita.copy()
    dm_demand_pack.array = dm_demand_pack.array * dm_pop.array[...,np.newaxis]
    dm_demand_pack.units["product-demand"] = "t"
    dm_netimport_pack = dm_import.filter({"Categories1": ['aluminium-pack', 'glass-pack', 'paper-pack',
                                                          'paper-print', 'paper-san', 'plastic-pack']})
    dm_netimport_pack.array = dm_netimport_pack.array * dm_demand_pack.array
    dm_netimport_pack.units["product-net-import"] = dm_demand_pack.units["product-demand"]
    dm_prod_pack = dm_demand_pack.copy()
    dm_prod_pack.array = dm_demand_pack.array - dm_netimport_pack.array
    dm_prod_pack.rename_col("product-demand","product-production","Variables")
    

    ########################
    ##### PUT TOGETHER #####
    ########################

    DM_production = {"bld-floor": dm_prod_bld_floor,
                     "tra-infra": dm_prod_tra_infra,
                     "tra-veh": dm_prod_tra_veh,
                     "pack": dm_prod_pack,
                     "bld-floor-net-import" : dm_netimport_bld_floor,
                     "tra-infra-net-import" : dm_netimport_tra_infra,
                     "tra-veh-net-import" : dm_netimport_tra_veh,
                     "pack-net-import" : dm_netimport_pack}
        
    # return
    return DM_production

def apply_material_decomposition(dm_production_bld_floor,
                                 dm_production_tra_infra, dm_production_tra_veh, dm_production_pack,
                                 cdm_matdec_floor,
                                 cdm_matdec_tra_infra, cdm_matdec_tra_veh, cdm_matdec_pack, cdm_matdec_tra_bat):
    
    countries = dm_production_bld_floor.col_labels["Country"]
    years = dm_production_bld_floor.col_labels["Years"]
    
    # material demand [t] = product production [unit] * material decomposition coefficient [t/unit]

    #####################
    ##### BUILDINGS #####
    #####################

    # floor
    dm_bld_floor_matdec = cdm_to_dm(cdm_matdec_floor, countries, years)
    dm_bld_floor_matdec.array = dm_production_bld_floor.array[...,np.newaxis] * dm_bld_floor_matdec.array
    dm_bld_floor_matdec.units['material-decomp'] = "t"
    
    #####################
    ##### TRANSPORT #####
    #####################

    # infra
    dm_tra_infra_matdec = cdm_to_dm(cdm_matdec_tra_infra, countries, years)
    dm_tra_infra_matdec.array = dm_production_tra_infra.array[...,np.newaxis] * dm_tra_infra_matdec.array
    dm_tra_infra_matdec.units['material-decomp'] = "t"

    # veh
    
    # add battery weight
    dm_tra_veh_matdec = cdm_to_dm(cdm_matdec_tra_veh, countries, years)
    dm_tra_bat_matdec = cdm_to_dm(cdm_matdec_tra_bat, countries, years)
    dm_tra_veh_matdec.append(dm_tra_bat_matdec,"Categories1")
    dm_tra_veh_matdec.groupby({"HDV" : ["HDV","battery-lion-HDV"], "LDV" : ["LDV","battery-lion-LDV"]}, 
                              "Categories1", inplace=True)
    
    # do material decomp
    dm_tra_veh_matdec.array = dm_production_tra_veh.array[...,np.newaxis] * dm_tra_veh_matdec.array
    dm_tra_veh_matdec.units['material-decomp'] = "t"

    #####################
    ##### PACKAGING #####
    #####################

    # packaging
    dm_pack_matdec = cdm_to_dm(cdm_matdec_pack, countries, years)
    dm_pack_matdec.array = dm_production_pack.array[...,np.newaxis] * dm_pack_matdec.array
    dm_pack_matdec.units['material-decomp'] = "t"

    ########################
    ##### PUT TOGETHER #####
    ########################

    dm_matdec = dm_bld_floor_matdec.copy()
    dm_matdec.append(dm_tra_infra_matdec, dim="Categories1")
    dm_temp = dm_tra_veh_matdec.flatten().flatten()
    dm_temp.deepen(based_on="Categories1")
    dm_matdec.append(dm_temp, dim="Categories1")
    dm_matdec.append(dm_pack_matdec, dim="Categories1")

    # note: we are calling this material demand as this is the demand of materials 
    # that comes from the production sector (e.g. how much material is needed to
    # produce a car)
    DM_material_demand = {"material-demand": dm_matdec}

    # clean
    del dm_bld_floor_matdec, \
        dm_tra_infra_matdec, dm_tra_veh_matdec, dm_pack_matdec

    # return
    return DM_material_demand

def apply_material_switch(dm_material_demand, dm_material_switch, cdm_material_switch, DM_input_matswitchimpact):
    
    # material in-to-out [t] = material in [t] * in-to-out [%]
    # material in [t] = material in [t] - material in-to-out [t]
    # material out [t] = material out [t] + material in-to-out [t] * switch ratio [t/t]
    
    products = dm_material_demand.col_labels["Categories1"]
    
    #####################
    ##### TRANSPORT #####
    #####################
    
    cars = list(np.array(products)[[bool(re.search("LDV", p)) for p in products]])
    for car in cars:
        material_switch(dm = dm_material_demand, dm_ots_fts=dm_material_switch,
                        cdm_const=cdm_material_switch, material_in="steel", material_out=["chem", "aluminium"],
                        product=car, switch_percentage_prefix="cars-",
                        switch_ratio_prefix="material-switch-ratios_")
        
    trucks = list(np.array(products)[[bool(re.search("HDV", p)) for p in products]])
    for truck in trucks:
        material_switch(dm=dm_material_demand, dm_ots_fts=dm_material_switch,
                        cdm_const=cdm_material_switch, material_in="steel", material_out=["chem", "aluminium"],
                        product=truck, switch_percentage_prefix="trucks-",
                        switch_ratio_prefix="material-switch-ratios_")

    #####################
    ##### BUILDINGS #####
    #####################

    # new buildings: switch to renewable materials (steel and cement to timber in new residential and non-residential)

    material_switch(dm = dm_material_demand, dm_ots_fts = dm_material_switch, 
                    cdm_const = cdm_material_switch, material_in = "steel", material_out = ["timber"], 
                    product = 'floor-area-new-residential', switch_percentage_prefix = "build-", 
                    switch_ratio_prefix = "material-switch-ratios_", dict_for_output = DM_input_matswitchimpact)

    material_switch(dm = dm_material_demand, dm_ots_fts = dm_material_switch, 
                    cdm_const = cdm_material_switch, material_in = "cement", material_out = ["timber"], 
                    product = 'floor-area-new-residential', switch_percentage_prefix = "build-", 
                    switch_ratio_prefix = "material-switch-ratios_", dict_for_output = DM_input_matswitchimpact)

    # material_switch(dm = dm_material_demand, dm_ots_fts = dm_material_switch, 
    #                 cdm_const = cdm_material_switch, material_in = "steel", material_out = ["timber"], 
    #                 product = 'floor-area-new-non-residential', switch_percentage_prefix = "build-", 
    #                 switch_ratio_prefix = "material-switch-ratios_", dict_for_output = DM_input_matswitchimpact)

    # material_switch(dm = dm_material_demand, dm_ots_fts = dm_material_switch, 
    #                 cdm_const = cdm_material_switch, material_in = "cement", material_out = ["timber"], 
    #                 product = 'floor-area-new-non-residential', switch_percentage_prefix = "build-", 
    #                 switch_ratio_prefix = "material-switch-ratios_", dict_for_output = DM_input_matswitchimpact)

    # renovated buildings: switch to insulated surfaces (chemicals to paper and natural fibers in renovated residential and non-residential)

    material_switch(dm = dm_material_demand, dm_ots_fts = dm_material_switch, 
                    cdm_const = cdm_material_switch, material_in = "chem", material_out = ["paper","natfibers"], 
                    product = "floor-area-reno-residential", switch_percentage_prefix = "reno-", 
                    switch_ratio_prefix = "material-switch-ratios_")

    # material_switch(dm = dm_material_demand, dm_ots_fts = dm_material_switch, 
    #                 cdm_const = cdm_material_switch, material_in = "chem", material_out = ["paper","natfibers"], 
    #                 product = "floor-area-reno-non-residential", switch_percentage_prefix = "reno-", 
    #                 switch_ratio_prefix = "material-switch-ratios_")
    
    return

def material_production(dm_material_efficiency, dm_material_net_import, 
                        dm_material_demand, dm_matprod_other_industries):
    
    ######################
    ##### EFFICIENCY #####
    ######################
    
    # get efficiency coefficients
    dm_temp = dm_material_efficiency.copy()
    dm_temp.filter({"Categories1" : dm_material_demand.col_labels["Categories2"]}, inplace=True)
    dm_temp.add(0, "Categories1", "natfiber", unit="%", dummy=True)
    dm_temp.sort("Categories1")
    
    # apply formula to material demand (and overwrite)
    dm_material_demand.array = dm_material_demand.array * (1 - dm_temp.array[:,:,:,np.newaxis,:])
    
    ############################
    ##### AGGREGATE DEMAND #####
    ############################

    # get aggregate demand
    dm_matdec_agg = dm_material_demand.group_all(dim='Categories1', inplace=False)
    dm_matdec_agg.change_unit('material-decomp', factor=1e-3, old_unit='t', new_unit='kt')

    # subset aggregate demand for the materials we keep
    # materials = ['aluminium', 'cement', 'chem', 'copper', 'glass', 'lime', 'paper', 'steel', 'timber']
    dm_material_production_natfiber = dm_matdec_agg.filter({"Categories1": ["natfibers"]}) # this will be used for interface agriculture
    dm_matdec_agg.drop("Categories1","natfibers")

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

    # include other industries from fxa
    dm_material_production_bymat.append(data2 = dm_matprod_other_industries, dim = "Categories1")
    dm_material_production_bymat.sort("Categories1")
    
    # put together
    DM_material_production = {"bymat" : dm_material_production_bymat, 
                              "material-net-import" : dm_material_net_import_kt,
                              "natfiber" : dm_material_production_natfiber}
    
    # clean
    del dm_matdec_agg, dm_temp, dm_material_production_bymat, dm_material_production_natfiber
    
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

def end_of_life(dm_transport_waste, dm_waste_management, 
                dm_matrec_veh, cdm_matdec_veh, 
                dm_matrec_bat, cdm_matdec_bat,
                dm_material_production_bymat):
    
    # in general:
    # littered + export + uncolleted + collected = 1
    # recycling + energy recovery + reuse + landfill + incineration = 1
    # note on incineration: transport will not have incineration, while electric waste yes
    
    # TODO: for the moment I am doing waste management only for cars and trucks. I am
    # also adding the recycling of batteries of the cars that are recycled.
    # this will have to be done
    # for 'planes', 'ships', 'trains', batteries (to understand if to treat them as a separate product), 
    # appliances (for which we'll have to update buildings) and
    # buildings (which is already present in buildings, though we need to develop the data on their eol).
    # So far, the split of waste categories should be the same for transport and appliances, with the
    # only difference that incineration will be positive values for appliances. That should be the only
    # difference, so same formulas should apply.
    
    # make waste-collected, waste-uncollected, export, littered
    layer1 = ["export","littered","waste-collected","waste-uncollected"]
    dm_waste_layer1 = dm_waste_management.filter({"Categories1" : layer1})
    arr_temp = dm_transport_waste.array[...,np.newaxis] * dm_waste_layer1.array[:,:,np.newaxis,:,np.newaxis,:]
    dm_transport_waste_bywsm_layer1 = DataMatrix.based_on(arr_temp, dm_transport_waste, 
                                                          {'Categories3': layer1}, 
                                                          units = dm_transport_waste.units)
    
    # do material decomp of layer 1 (to be used in stock and flow analysis)
    dm_temp = dm_transport_waste_bywsm_layer1.copy()
    arr_temp = dm_temp.array[...,np.newaxis] * cdm_matdec_veh.array[np.newaxis,np.newaxis,:,:,:,np.newaxis,:] / 1000
    dm_transport_waste_bymat = DataMatrix.based_on(arr_temp, dm_temp, 
                                                   {'Categories4': cdm_matdec_veh.col_labels["Categories3"]}, 
                                                   units = "kt")
    dm_transport_waste_bymat.units["tra_product-waste"] = "kt"
    dm_transport_waste_bymat.group_all("Categories2")
    dm_transport_waste_bymat.group_all("Categories1")
    
    # make recycling, energy recovery, reuse, landfill, incineration
    layer2 = ["energy-recovery","incineration","landfill","recycling","reuse"]
    dm_waste_layer2 = dm_waste_management.filter({"Categories1" : layer2})
    dm_waste_collected = dm_transport_waste_bywsm_layer1.filter({"Categories3" : ["waste-collected"]})
    arr_temp = dm_waste_collected.array * dm_waste_layer2.array[:,:,np.newaxis,:,np.newaxis,:]
    dm_transport_waste_bywsm_layer2 = DataMatrix.based_on(arr_temp, dm_waste_collected, 
                                                          {'Categories3': layer2}, 
                                                          units = dm_waste_collected.units)

    # do material decomposition of layer 2 (to be used in material flow analysis, and recycling to be used below)
    dm_temp = dm_transport_waste_bywsm_layer2.copy()
    dm_veh_eol_to_recycling = dm_temp.copy()
    arr_temp = dm_temp.array[...,np.newaxis] * cdm_matdec_veh.array[np.newaxis,np.newaxis,:,:,:,np.newaxis,:]
    dm_transport_waste_collect_bymat = DataMatrix.based_on(arr_temp, dm_temp, 
                                                   {'Categories4': cdm_matdec_veh.col_labels["Categories3"]}, 
                                                   units = "t")
    dm_transport_waste_collect_bymat.units["tra_product-waste"] = "t"
    dm_transport_waste_collect_bymat.group_all("Categories2")
    dm_transport_waste_collect_bymat.group_all("Categories1")
    
    # get material recovered from recycling
    dm_transport_matrecovered_veh = dm_transport_waste_collect_bymat.filter({"Categories1" : ["recycling"]})
    dm_transport_matrecovered_veh.array = dm_transport_matrecovered_veh.array * \
        dm_matrec_veh.array
        
    # add material recovered from battery recycling
    # note: assuming 1 battery pack per vehicle
    dm_tra_batteries_torecy = dm_transport_waste_bywsm_layer2.filter(
        {"Categories1" : ['HDV', 'LDV'], 
         "Categories3" : ['recycling']})
    arr_temp = dm_tra_batteries_torecy.array * cdm_matdec_bat.array[np.newaxis,np.newaxis,...]
    dm_batteries_matrecovered = DataMatrix.based_on(arr_temp, dm_tra_batteries_torecy, 
                                                         {'Categories3': cdm_matdec_bat.col_labels["Categories3"]}, 
                                                         units = "t")
    dm_batteries_matrecovered.units["tra_product-waste"] = "t"
    dm_batteries_matrecovered.array = dm_batteries_matrecovered.array * dm_matrec_bat.array[:,:,:,:,np.newaxis,:]
    
    # sum across products
    dm_transport_matrecovered_veh.group_all("Categories1")
    dm_transport_matrecovered_veh.rename_col('tra_product-waste','vehicles',"Variables")
    # dm_transport_matrecovered_veh = dm_transport_matrecovered_veh.flatten()
    # dm_transport_matrecovered_veh.rename_col_regex("vehicles_","","Variables")
    dm_batteries_matrecovered.group_all("Categories2")
    dm_batteries_matrecovered.group_all("Categories1")
    dm_batteries_matrecovered.rename_col('tra_product-waste','batteries',"Variables")
    
    # get material recovered across sectors
    # TODO!: later on add here the other sectors, like buildings
    dm_matrecovered = dm_transport_matrecovered_veh.copy()
    dm_matrecovered.append(dm_batteries_matrecovered,"Variables")
    dm_matrecovered.groupby({"material-recovered" : ["vehicles","batteries"]}, "Variables", inplace=True)
    dm_matrecovered.change_unit('material-recovered', factor=1e-3, old_unit='t', new_unit='kt')
    
    # if material recovered is larger than material produced, impose material recovered to be equal to material produced
    materials = dm_matrecovered.col_labels["Categories1"]
    dm_temp = dm_matrecovered.copy()
    dm_temp1 = dm_material_production_bymat.filter({"Categories1" : materials})
    dm_temp.array = dm_matrecovered.array - dm_temp1.array # create a dm in which you do the difference
    dm_temp.array[dm_temp.array > 0] = 0 # where the difference is > 0, put difference = 0
    dm_temp.array = dm_temp.array + dm_temp1.array # sum back the material production, so where the difference > 0, the value of material recovered now equals the value of material production
    dm_matrecovered_corrected = dm_temp
    
    # reshape dm_veh_eol_to_recycling
    dm_veh_eol_to_recycling.group_all("Categories3")
    
    # save
    DM_eol = {
        "material-towaste": dm_transport_waste_bymat,
        "material-towaste-collected" : dm_transport_waste_collect_bymat,
        "material-recovered" : dm_matrecovered_corrected,
        "veh_eol_to_wst_mgt" : dm_transport_waste_bywsm_layer1,
        "veh_el_to_collection" : dm_transport_waste_bywsm_layer2,
        "veh_eol_to_recycling" : dm_veh_eol_to_recycling
        }
    
    return DM_eol

def material_production_by_technology(dm_technology_share, dm_material_production_bymat, 
                                      dm_material_recovered):
    
    # Note: now that we have the material recovered via the end of life module (which
    # is post-consumer recovery), the technology shares will be applied to whatever
    # produced material that is left. It follows that the percentage share of aluminium
    # secondary and steel made from scrap steel, represent the pre-consumer material
    # recovery.
    
    # subtract material recovered from material production
    # note: in eol we have already adjusted for the cases when recovery > production (we have set recovery = production)
    dm_material_production_bymat_adj = dm_material_production_bymat.copy()
    materials_recovered = dm_material_recovered.col_labels["Categories1"]
    idx_matprod_bymat = dm_material_production_bymat_adj.idx
    idx_recovered = dm_material_recovered.idx
    for m in materials_recovered:
        dm_material_production_bymat_adj.array[...,idx_matprod_bymat[m]] = \
        dm_material_production_bymat_adj.array[...,idx_matprod_bymat[m]] - \
        dm_material_recovered.array[...,idx_recovered[m]]
        
    # add timber to wood and wooden products (wwp), and other ot other industrial sector (ois)
    dm_material_production_bymat.groupby({'wwp': 'timber|wwp', 'ois' : 'other|ois'}, 
                                          dim='Categories1', aggregation = "sum", regex=True, inplace=True)
    dm_material_production_bymat_adj.groupby({'wwp': 'timber|wwp', 'ois' : 'other|ois'}, 
                                              dim='Categories1', aggregation = "sum", regex=True, inplace=True)
    
    # create dm_material_production_bytech with production data and shape of tech share
    dm_material_production_bytech = dm_technology_share.copy()
    idx_matprod_bytech = dm_material_production_bytech.idx
    idx_matprod_bymat = dm_material_production_bymat_adj.idx
    techs = dm_material_production_bytech.col_labels["Categories1"].copy()
    techs.remove("pulp-tech") # doing pulp-tech later as the code below would not work for that 
    for tech in techs:
        material = tech.split("-")[0]
        if material == "tra":
            material = "tra-equip"
        dm_material_production_bytech.array[...,idx_matprod_bytech[tech]] =\
            dm_material_production_bymat_adj.array[...,idx_matprod_bymat[material]]
    dm_material_production_bytech.array[...,idx_matprod_bytech["pulp-tech"]] = \
        dm_material_production_bymat_adj.array[...,idx_matprod_bymat["paper"]]
    dm_material_production_bytech.rename_col("technology-share","material-production","Variables")
    dm_material_production_bytech.units["material-production"] = dm_material_production_bymat.units["material-production"]    
    
    # get material production by technology (as tech share * (material production net of recovered))
    dm_material_production_bytech.array = dm_material_production_bytech.array * dm_technology_share.array
    
    # put in dm_material_production_bytech the material recovered as post consumer tech
    dm_temp = dm_material_recovered.copy()
    dm_temp.rename_col("timber","wwp","Categories1") # rename timber as wwp as it will be called wwp-tech-sec-post-consumer
    dm_temp.rename_col("other","ois","Categories1") # rename other as ois as it will be called ois-tech-sec-post-consumer
    for material in dm_temp.col_labels["Categories1"]:
        dm_temp.rename_col(material, material + "-sec-post-consumer","Categories1")
    dm_temp.rename_col('material-recovered','material-production','Variables')
    dm_material_production_bytech.append(dm_temp,"Categories1")
    dm_material_production_bytech.sort("Categories1")
    
    # # checks
    # # production = primary + secondary pre consumer + secondary post consumer
    # dm_material_production_bytech.array[dm_material_production_bytech.array < 0]
    # dict_aggregation = {"aluminium" : ['aluminium-prim', 'aluminium-sec', 'aluminium-sec-post-consumer'],
    #                     "cement" : ['cement-dry-kiln', 'cement-geopolym', 'cement-sec-post-consumer', 'cement-wet-kiln'],
    #                     "chem" : ['chem-chem-tech', 'chem-sec-post-consumer'],
    #                     "copper" : ['copper-sec-post-consumer', 'copper-tech'],
    #                     "glass" : ['glass-glass', 'glass-sec-post-consumer'],
    #                     "lime" : ['lime-lime', 'lime-sec-post-consumer'],
    #                     "paper" : ['paper-sec-post-consumer', 'paper-tech', 'pulp-tech'],
    #                     "steel" : ['steel-BF-BOF', 'steel-hisarna', 'steel-hydrog-DRI', 'steel-scrap-EAF', 'steel-sec-post-consumer'],
    #                     "wwp" : ['wwp-sec-post-consumer', 'wwp-tech']}
    # checks = {}
    # for key in dict_aggregation.keys():
    #     dm_temp1 = dm_material_production_bymat.filter({"Country" : ["EU27"], "Categories1" : [key]})
    #     dm_temp2 = dm_material_production_bytech.filter({"Country" : ["EU27"], "Categories1" : dict_aggregation[key]})
    #     dm_temp1 = dm_temp1.flatten()
    #     dm_temp2.group_all("Categories1")
    #     checks[key] = np.allclose(dm_temp1.array, dm_temp2.array) # this is to avoid rounding issues
    # checks
    # # ok
    
    # return
    return dm_material_production_bytech

def energy_demand(dm_material_production_bytech, CDM_const):
    
    # this is by material-technology and carrier
    
    # drop post consumer techs for lime and wood, as we assume they are not
    # widely used at the moment. Also drop ois-sec-post-consumer, as for the
    # moment we assume that it negligible for energy demand and emissions.
    # TODO: do the energy demand and emissions for these and their respective pre consumer techs
    dm_material_production_bytech.drop("Categories1",['lime-sec-post-consumer',
                                                      'wwp-sec-post-consumer',
                                                      'ois-sec-post-consumer'])

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

    # clean
    del cdm_temp1, cdm_temp2, arr_temp, bio, \
        dm_emissions_bygastech, dm_emissions_combustion_bio, dm_emissions_process, \
        dm_emissions_combustion
    
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

    # clean
    del arr_temp, dm_emissions_combustion_bio_capt_w_cc, \
        dm_temp, idx, dm_emissions_combustion_bio_capt_w_cc_neg_bymat, \
        dm_emissions_capt_w_cc_bytech
        
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

# TODO: bring the function material_flows() into minerals (or somewhere external, probably together with the
# cost curves ... i.e. it could be in the tpe) and finalize it
def material_flows(dm_transport_stock, dm_material_towaste, dm_material_recovered, cdm_matdec_veh,
                   dm_emissions_bygasmat):
    
    # do material decomposition of stock
    arr_temp = dm_transport_stock.array[...,np.newaxis] * cdm_matdec_veh.array[np.newaxis,np.newaxis,:,:,:]
    dm_transport_stock_bymat = DataMatrix.based_on(arr_temp, dm_transport_stock, {'Categories2': cdm_matdec_veh.col_labels["Categories2"]}, 
                                                   units = dm_transport_stock.units)
    dm_transport_stock_bymat.units["tra_product-stock"] = "t"
    
    # sum across sectors
    dm_stock_bymat = dm_transport_stock_bymat.group_all("Categories1", inplace = False)
    dm_stock_bymat.rename_col('tra_product-stock','product-stock',"Variables")
    
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

def variables_for_tpe(dm_cost_material_production_capex, dm_cost_CO2_capt_w_cc_capex,
                      dm_emissions_bygas, 
                      dm_material_production_bytech, dm_material_production_bymat,
                      dm_energy_demand_bymat, dm_energy_demand_bymatcarr, 
                      dm_energy_demand_bioener):
    
    # TODO: recheck variables' names compared to TPE (while running TPE)
    
    # adjust variables' names
    dm_cost_material_production_capex.rename_col_regex("material-production_capex", "investment", "Variables")
    dm_cost_CO2_capt_w_cc_capex.rename_col_regex("CO2-capt-w-cc_capex", "investment_CC", "Variables")
    dm_emissions_bygas = dm_emissions_bygas.flatten()
    dm_emissions_bygas.rename_col_regex("_","-","Variables")
    # variables = dm_material_production_bytech.col_labels["Categories1"]
    # variables_new = [rename_tech_fordeepen(i) for i in variables]
    # for i in range(len(variables)):
    #     dm_material_production_bytech.rename_col(variables[i], variables_new[i], dim = "Categories1")
        
    # convert kt to mt
    dm_material_production_bytech.change_unit('material-production', factor=1e-3, old_unit='kt', new_unit='Mt')
    dm_material_production_bymat.change_unit('material-production', factor=1e-3, old_unit='kt', new_unit='Mt')

    # material production total (chemicals done in ammonia)
    dm_mat_prod = dm_material_production_bymat.filter({"Categories1" : ["aluminium","cement","copper",
                                                                      "glass","lime","paper","steel"]})
    dm_mat_prod.rename_col('material-production', 'ind_material-production', 'Variables')
    
    # energy demand by material
    dm_energy_by_mat = dm_energy_demand_bymat.copy()
    dm_energy_by_mat.rename_col('energy-demand', 'ind_energy-demand', 'Variables')
    
    # emissions (done in emissions)
    
    # production technologies (aluminium, cement, paper, steel)
    dm_prod_tech = dm_material_production_bytech.copy()
    # dm_temp.groupby({'aluminium_sec': 'aluminium_sec.*',
    #                  'steel_scrap-EAF': 'steel_scrap-EAF-precons.*|teel_sec-postcons'}, 
    #                 dim='Categories1', regex=True, inplace=True)
    # dm_prod_tech = dm_temp.filter({"Categories1" : ['aluminium_prim', 'aluminium_sec',
    #                                                 'cement_dry-kiln', 'cement_geopolym',
    #                                                 'cement_wet-kiln','paper_recycled',
    #                                                 'paper_woodpulp', 'steel_BF-BOF',
    #                                                 'steel_hisarna', 'steel_hydrog-DRI',
    #                                                 'steel_scrap-EAF']})
    dm_prod_tech.rename_col('material-production', 'ind_material-production', 'Variables')
    
    # energy demand for material production by ener carrier (aluminium, cement, chem, glass, lime, paper, steel)
    dm_energy_by_carrier = dm_energy_demand_bymatcarr.filter({"Categories1": ['aluminium', 'cement', 'glass',
                                                                              'lime', 'paper', 'steel']})
    dm_energy_by_carrier.rename_col('energy-demand', 'ind_energy-demand', 'Variables')
    

    # dm_tpe
    dm_tpe = dm_emissions_bygas.copy()
    dm_tpe.append(dm_energy_by_mat.flatten(), "Variables")
    dm_tpe.append(dm_energy_by_carrier.flatten().flatten(), "Variables")
    dm_tpe.append(dm_energy_demand_bioener, "Variables")
    dm_tpe.append(dm_cost_CO2_capt_w_cc_capex.flatten(), "Variables")
    dm_tpe.append(dm_cost_material_production_capex.flatten(), "Variables")
    dm_tpe.append(dm_mat_prod.flatten(), "Variables")
    dm_tpe.append(dm_prod_tech.flatten(), "Variables")

    # return
    return dm_tpe

def industry_agriculture_interface(DM_material_production, DM_energy_demand, write_pickle = False):
    
    DM_agr = {}
    
    # natfibers
    dm_temp = DM_material_production["natfiber"].copy()
    dm_temp.rename_col('material-decomp', "ind_dem", "Variables")
    DM_agr["natfibers"] = dm_temp.flatten()
    
    # bioenergy
    dm_temp = DM_energy_demand["bioener_bybiomat"].copy()
    dm_temp.rename_col("energy-demand_bioenergy", "ind_bioenergy", "Variables")
    dm_temp = dm_temp.filter({"Categories1" : ['gas-bio', 'solid-bio']})
    DM_agr["bioenergy"] = dm_temp
    
    # biomaterial
    dm_temp = DM_energy_demand["feedstock_bybiomat"].copy()
    dm_temp.rename_col("energy-demand-feedstock", "ind_biomaterial", "Variables")
    dm_temp = dm_temp.filter({"Categories1" : ['gas-bio']})
    DM_agr["biomaterial"] = dm_temp
    
    # if write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, '../_database/data/interface/industry_to_agriculture.pickle')
        with open(f, 'wb') as handle:
            pickle.dump(DM_agr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return DM_agr

def industry_ammonia_interface(DM_material_production, DM_energy_demand, write_pickle = False):
    
    dm_amm_matprod = DM_material_production["bymat"].filter({"Categories1" : ["chem"]})
    dm_amm_endem = DM_energy_demand["bymatcarr"].filter({"Categories1" : ['chem']})
    DM_amm = {"material-production" : dm_amm_matprod,
              "energy-demand" : dm_amm_endem}
    
    # of write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, '../_database/data/interface/industry_to_ammonia.pickle')
        with open(f, 'wb') as handle:
            pickle.dump(DM_amm, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return DM_amm

def industry_landuse_interface(DM_material_production, DM_energy_demand, write_pickle = False):
    
    DM_lus = {}
    
    # timber
    dm_timber = DM_material_production["bymat"].filter({"Categories1" : ["timber"]})
    dm_timber.rename_col("material-production", "ind_material-production", "Variables")
    dm_timber = dm_timber.flatten()
    DM_lus["timber"] = dm_timber
    
    # woodpuplp
    dm_woodpulp = DM_material_production["bytech"].filter({"Categories1" : ['pulp-tech']})
    dm_woodpulp.rename_col("material-production", "ind_material-production", "Variables")
    DM_lus["woodpulp"] = dm_woodpulp.flatten()
    
    # biomaterial solid bio
    dm_temp = DM_energy_demand["feedstock_bybiomat"].copy()
    dm_temp.rename_col("energy-demand-feedstock", "ind_biomaterial", "Variables")
    dm_temp = dm_temp.filter({"Categories1" : ['solid-bio']})
    DM_lus["biomaterial"] = dm_temp.flatten()
        
    # of write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, '../_database/data/interface/industry_to_land-use.pickle')
        with open(f, 'wb') as handle:
            pickle.dump(DM_lus, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # return
    return DM_lus

def industry_power_interface(DM_energy_demand, write_pickle = False):
    
    # dm_elc
    dm_elc = DM_energy_demand["bycarr"].filter(
        {"Categories1" : ['electricity','hydrogen']})
    dm_elc.rename_col("energy-demand", "ind_energy-demand", "Variables")
    dm_elc = dm_elc.flatten()

    dm_elc.change_unit('ind_energy-demand_electricity', factor=1e3, old_unit='TWh', new_unit='GWh')
    dm_elc.change_unit('ind_energy-demand_hydrogen', factor=1e3, old_unit='TWh', new_unit='GWh')

    DM_pow = {
        'electricity': dm_elc.filter({'Variables': ['ind_energy-demand_electricity']}),
        'hydrogen': dm_elc.filter({'Variables': ['ind_energy-demand_hydrogen']})
    }

    # of write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, '../_database/data/interface/industry_to_power.pickle')
        with open(f, 'wb') as handle:
            pickle.dump(DM_pow, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # return
    return DM_pow

def industry_refinery_interface(DM_energy_demand, write_pickle = False):
    
    # dm_elc
    dm_ref = DM_energy_demand["bycarr"].filter(
        {"Categories1": ['liquid-ff-diesel', 'liquid-ff-oil',
                          'gas-ff-natural', 'solid-ff-coal']})
    dm_ref.rename_col("energy-demand", "ind_energy-demand", "Variables")
    dm_ref.rename_col_regex('liquid-ff-oil_', '', dim='Categories1')
    dm_ref.sort("Categories1")

    # of write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, '../_database/data/interface/industry_to_refinery.pickle')
        with open(f, 'wb') as handle:
            pickle.dump(dm_ref, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # return
    return dm_ref

def industry_district_heating_interface(DM_energy_demand, write_pickle = False):
    
    # FIXME!: fix dummy values for dh with real values
    dm_dh = DM_energy_demand["bycarr"].filter({"Categories1" : ["electricity"]})
    dm_dh = dm_dh.flatten()
    dm_dh.add(0, dim = "Variables", col_label = "dhg_energy-demand_contribution_heat-waste", unit = "TWh/year", dummy = True)
    dm_dh.drop("Variables", 'energy-demand_electricity')
    
    # of write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, '../_database/data/interface/industry_to_district-heating.pickle')
        with open(f, 'wb') as handle:
            pickle.dump(dm_dh, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # return
    return dm_dh

def industry_emissions_interface(DM_emissions, write_pickle = False):
    
    # adjust variables' names
    dm_temp = DM_emissions["bygasmat"].flatten().flatten()
    dm_temp.deepen()
    dm_temp.rename_col_regex("_","-","Variables")
    DM_emissions["combustion_bio_capt_w_cc_neg_bymat"].rename_col("emissions-biogenic_CO2-capt-w-cc-negative","emissions-CO2_biogenic","Variables")
    dm_temp1 = DM_emissions["combustion_bio_capt_w_cc_neg_bymat"].flatten()
    # TODO: re-check what it's doing here above. In principle, it would make sense
    # to gather negative emissions, to then be put together with other emissions below ...
    # but why only gater biogenic negative emissions, and not all negative emissions?
    # to be re-checked

    # dm_ems
    dm_ems = dm_temp.flatten()
    dm_ems.append(dm_temp1, "Variables")
    variables = dm_ems.col_labels["Variables"]
    for i in variables:
        dm_ems.rename_col(i, "ind_" + i, "Variables")
    dm_ems.sort("Variables")

    # of write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, '../_database/data/interface/industry_to_emissions.pickle')
        with open(f, 'wb') as handle:
            pickle.dump(dm_ems, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # return
    return dm_ems

def industry_water_inferface(DM_energy_demand, DM_material_production, write_pickle = False):
    
    # dm_water
    dm_water = DM_energy_demand["bycarr"].filter(
        {"Categories1" : ['electricity', 'gas-ff-natural', 'hydrogen', 'liquid-ff-oil', 
                          'solid-ff-coal']}).flatten()
    dm_water.append(DM_material_production["bytech"].filter(
        {"Categories1" : ['aluminium-prim', 'aluminium-sec', 'cement-dry-kiln', 'cement-geopolym', 
                          'cement-wet-kiln', 'chem-chem-tech', 'copper-tech', 'glass-glass', 'lime-lime',
                          'paper-recycled', 'paper-woodpulp', 'steel-BF-BOF', 'steel-hisarna', 
                          'steel-hydrog-DRI', 'steel-scrap-EAF']}).flatten(), "Variables")
    variables = dm_water.col_labels["Variables"]
    for i in variables:
        dm_water.rename_col(i, "ind_" + i, "Variables")
    dm_water.sort("Variables")

    # of write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, '../_database/data/interface/industry_to_water.pickle')
        with open(f, 'wb') as handle:
            pickle.dump(dm_water, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # return
    return dm_water

def industry_ccus_interface(DM_emissions, write_xls = False):
    
    # adjust variables' names
    DM_emissions["capt_w_cc_bytech"].rename_col('CO2-capt-w-cc','ind_CO2-emissions-CC',"Variables")

    # dm_ccus
    dm_ccus = DM_emissions["capt_w_cc_bytech"].filter(
        {"Categories1" : ['aluminium_prim', 'aluminium_sec', 'cement_dry-kiln', 'cement_geopolym', 
                          'cement_wet-kiln', 'chem_chem-tech', 'lime_lime',
                          'paper_recycled', 'paper_woodpulp', 'steel_BF-BOF', 'steel_hisarna', 
                          'steel_hydrog-DRI', 'steel_scrap-EAF']}).flatten()
    dm_ccus.sort("Variables")

    # df_ccus
    if write_xls is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        df_ccus = dm_ccus.write_df()
        df_ccus.to_excel(current_file_directory + "/../_database/data/xls/" + 'industry-to-ccus.xlsx', index=False)
        
    # return
    return dm_ccus

# def industry_minerals_interface(DM_material_production, DM_production, DM_ots_fts, write_xls = False):
    
#     DM_ind = {}
    
#     # aluminium pack
#     dm_alupack = DM_production["pack"].filter({"Categories1" : ["aluminium-pack"]})
#     DM_ind["aluminium-pack"] = dm_alupack.flatten()
    
#     # material production
#     dm_matprod = DM_material_production["bymat"].filter({"Categories1": ["timber", 'glass', 'cement']})
#     dm_paper_woodpulp = DM_material_production["bytech"].filter({"Categories1": ['paper_woodpulp']})
#     dm_matprod.append(dm_paper_woodpulp, "Categories1")
#     dm_matprod.rename_col("material-production", "ind_material-production", "Variables")
#     DM_ind["material-production"] = dm_matprod.flatten()
    
#     # technology development
#     dm_techdev = DM_ots_fts['technology-development'].filter(
#         {"Categories1" : ['aluminium-prim', 'aluminium-sec','copper-tech',
#                           'steel-BF-BOF', 'steel-hisarna', 'steel-hydrog-DRI', 
#                           'steel-scrap-EAF']})
#     variables = dm_techdev.col_labels["Categories1"]
#     variables_new = ['aluminium_primary', 'aluminium_secondary','copper_secondary',
#                       'steel_BF-BOF', 'steel_hisarna', 'steel_hydrog-DRI', 
#                       'steel_scrap-EAF']
#     for i in range(len(variables)):
#         dm_techdev.rename_col(variables[i], variables_new[i], dim = "Categories1")
#     dm_techdev.rename_col("ind_technology-development","ind_proportion","Variables")
#     DM_ind["technology-development"] = dm_techdev.flatten()
    
#     # material efficiency
#     DM_ind["material-efficiency"] = DM_ots_fts['material-efficiency'].filter(
#         {"Variables" : ['ind_material-efficiency'],
#          "Categories1" : ['aluminium','copper','steel']})
    
#     # material switch
#     dm_temp = DM_ots_fts['material-switch'].filter(
#         {"Categories1" : ['build-steel-to-timber', 'cars-steel-to-chem', 
#                           'trucks-steel-to-aluminium', 'trucks-steel-to-chem']}).flatten()
#     dm_temp.rename_col_regex("material-switch_","material-switch-","Variables")
#     DM_ind["material-switch"] = dm_temp
    
#     # product net import
#     dm_temp = DM_ots_fts["product-net-import"].filter(
#         {"Variables" : ["ind_product-net-import"],
#          "Categories1" : ['cars-EV', 'cars-FCV', 'cars-ICE', 'computer', 'dishwasher', 'dryer',
#                           'freezer', 'fridge','phone','planes','rail','road', 'ships', 'trains',
#                           'trolley-cables', 'trucks-EV', 'trucks-FCV', 'trucks-ICE', 'tv', 
#                           'wmachine','new-dhg-pipe']})
#     dm_temp.rename_col_regex("cars","LDV","Categories1")
#     dm_temp.rename_col_regex("trucks","HDVL","Categories1")
#     dm_temp.rename_col("computer","electronics-computer","Categories1")
#     dm_temp.rename_col("phone","electronics-phone","Categories1")
#     dm_temp.rename_col("tv","electronics-tv","Categories1")
#     dm_temp.rename_col("dishwasher","dom-appliance-dishwasher","Categories1")
#     dm_temp.rename_col("dryer","dom-appliance-dryer","Categories1")
#     dm_temp.rename_col("freezer","dom-appliance-freezer","Categories1")
#     dm_temp.rename_col("fridge","dom-appliance-fridge","Categories1")
#     dm_temp.rename_col("wmachine","dom-appliance-wmachine","Categories1")
#     dm_temp.rename_col("new-dhg-pipe","infra-pipe","Categories1")
#     dm_temp.rename_col("rail","infra-rail","Categories1")
#     dm_temp.rename_col("road","infra-road","Categories1")
#     dm_temp.rename_col("trolley-cables","infra-trolley-cables","Categories1")
#     dm_temp.rename_col("planes","other-planes","Categories1")
#     dm_temp.rename_col("ships","other-ships","Categories1")
#     dm_temp.rename_col("trains","other-trains","Categories1")
#     dm_temp.rename_col_regex('FCV', 'FCEV', 'Categories1')
#     dm_temp.sort("Categories1")
#     DM_ind["product-net-import"] = dm_temp.flatten()

#     # df_min
#     if write_xls is True:
        
#         current_file_directory = os.path.dirname(os.path.abspath(__file__))
        
#         dm_min = DM_ind['aluminium-pack']
#         dm_min.append(DM_ind['material-production'], "Variables")
#         dm_min.append(DM_ind['technology-development'], "Variables")
#         dm_min.append(DM_ind['material-efficiency'].flatten(), "Variables")
#         dm_min.append(DM_ind['material-switch'], "Variables")
#         dm_min.append(DM_ind['product-net-import'], "Variables")
#         dm_min.sort("Variables")
        
#         df_min = dm_min.write_df()
#         df_min.to_excel(current_file_directory + "/../_database/data/xls/" + 'industry-to-minerals.xlsx', index=False)
        
#     # return
#     return DM_ind

def industry_minerals_interface(DM_production, veh_eol_to_recycling, write_pickle = False):
    
    DM_ind = {"production" : DM_production,
              "veh-to-recycling" : veh_eol_to_recycling}
    
    # of write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, '../_database/data/interface/industry_to_minerals.pickle')
        with open(f, 'wb') as handle:
            pickle.dump(DM_ind, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # return
    return DM_ind

def industry_employment_interface(DM_material_demand, DM_energy_demand, DM_material_production, DM_cost, DM_ots_fts, write_xls = False):
    
    # get material demand for appliances
    DM_material_demand["appliances"] = \
        DM_material_demand["material-demand"].filter(
            {"Categories1" : ["computer", "dishwasher", "dryer",
                              "freezer", "fridge", "tv"]}).group_all("Categories1",
                                                                   inplace=False)
    DM_material_demand["appliances"].rename_col("material-decomposition", "material-demand_appliances", "Variables")
    
    # get material demand for transport
    DM_material_demand["transport"] = \
        DM_material_demand["material-demand"].filter(
            {"Categories1": ['cars-EV', 'cars-FCV', 'cars-ICE',
                              'trucks-EV', 'trucks-FCV', 'trucks-ICE',
                              'planes', 'ships', 'trains']}).group_all("Categories1", inplace=False)
    DM_material_demand["transport"].rename_col("material-decomposition", "material-demand_transport", "Variables")
    
    # get material demand for construction
    DM_material_demand["construction"] = \
        DM_material_demand["material-demand"].filter(
            {"Categories1": ['floor-area-new-non-residential', 'floor-area-new-residential',
                              'floor-area-reno-non-residential', 'floor-area-reno-residential',
                              'rail', 'road', 'trolley-cables']}).group_all("Categories1", inplace = False)
    DM_material_demand["construction"].rename_col("material-decomposition", "material-demand_construction", "Variables")
    
    # dm_emp
    dm_emp = DM_material_demand["appliances"].flatten()
    dm_emp.append(DM_material_demand["transport"].flatten(), "Variables")
    dm_emp.append(DM_material_demand["construction"].flatten(), "Variables")
    dm_emp.append(DM_energy_demand["bymat"].flatten(), "Variables")
    dm_emp.append(DM_energy_demand["bymatcarr"].flatten().flatten(), "Variables")
    dm_emp.append(DM_material_production["bymat"].filter(
        {"Categories1" : DM_energy_demand["bymat"].col_labels["Categories1"]}).flatten(), "Variables")
    dm_emp.append(DM_cost["material-production_capex"].flatten(), "Variables")
    dm_emp.append(DM_cost["material-production_opex"].flatten(), "Variables")
    dm_emp.append(DM_cost["CO2-capt-w-cc_capex"].flatten(), "Variables")
    dm_emp.append(DM_cost["CO2-capt-w-cc_opex"].flatten(), "Variables")
    variables = dm_emp.col_labels["Variables"]
    for i in variables:
        dm_emp.rename_col(i, "ind_" + i, "Variables")
    dm_emp.append(DM_ots_fts["material-net-import"].filter(
        {"Categories1" : ['aluminium', 'cement', 'chem', 'copper', 'glass', 'lime', 
                          'paper', 'steel', 'timber'],}).flatten(), "Variables")
    dm_emp.sort("Variables")

    # df_emp
    if write_xls is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        df_emp = dm_emp.write_df()
        df_emp.to_excel(current_file_directory + "/../_database/data/xls/" + 'industry-to-employment.xlsx', index=False)

    # return
    return dm_emp

def industry_airpollution_interface(DM_material_production, DM_energy_demand, write_xls = False):
    
    # dm_airpoll
    dm_airpoll = DM_material_production["bytech"].flatten()
    dm_airpoll.append(DM_energy_demand["bymatcarr"].flatten().flatten(), "Variables")
    variables = dm_airpoll.col_labels["Variables"]
    for i in variables:
        dm_airpoll.rename_col(i, "ind_" + i, "Variables")
    dm_airpoll.sort("Variables")

    # write
    if write_xls is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        df_airpoll = dm_airpoll.write_df()
        df_airpoll.to_excel(current_file_directory + "/../_database/data/xls/" + 'industry-to-air_pollution.xlsx', index=False)
        
    # return
    return dm_airpoll

def industry(lever_setting, years_setting, interface = Interface(), calibration = False):

    # industry data file
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    industry_data_file = os.path.join(current_file_directory, '../_database/data/datamatrix/geoscale/industry.pickle')
    DM_fxa, DM_ots_fts, DM_cal, CDM_const = read_data(industry_data_file, lever_setting)

    # get interfaces
    cntr_list = DM_ots_fts["product-net-import"].col_labels['Country']
    DM_transport = get_interface(current_file_directory, interface, "transport", "industry", cntr_list)
    DM_lfs = get_interface(current_file_directory, interface, "lifestyles", "industry", cntr_list)
    DM_buildings = get_interface(current_file_directory, interface, "buildings", "industry", cntr_list)

    # get product import
    dm_import = DM_ots_fts["product-net-import"]
    
    # get product production
    DM_production = product_production(DM_buildings["floor-demand"], DM_transport["tra-infra"], DM_transport["tra-veh"], 
                                       DM_lfs["pop"], DM_ots_fts["paperpack"], 
                                       dm_import)
    
    # get material demand
    DM_material_demand = apply_material_decomposition(DM_production["bld-floor"],
                                                      DM_production["tra-infra"], 
                                                      DM_production["tra-veh"], 
                                                      DM_production["pack"].copy(),
                                                      CDM_const["material-decomposition_floor"],
                                                      CDM_const["material-decomposition_infra"], 
                                                      CDM_const["material-decomposition_veh"], 
                                                      CDM_const["material-decomposition_pack"],
                                                      CDM_const["material-decomposition_bat"])
    
    # do material switch (writes in DM_material_demand and DM_input_matswitchimpact)
    DM_input_matswitchimpact = {} # create dict to save material switch that will be used later for environmental impact
    apply_material_switch(DM_material_demand["material-demand"], DM_ots_fts["material-switch"], 
                          CDM_const["material-switch"], DM_input_matswitchimpact)
    
    # get material production
    DM_material_production = material_production(DM_ots_fts['material-efficiency'], DM_ots_fts['material-net-import'], 
                                                 DM_material_demand["material-demand"], DM_fxa["prod"])

    # calibrate material production (writes in DM_material_production)
    if calibration is True:
        calibration_material_production(DM_cal, DM_material_production["bymat"], DM_material_production,
                                        years_setting)
        
    # get end of life
    DM_eol = end_of_life(DM_transport["tra-waste"].filter({"Categories1" : ["HDV","LDV","bus"]}), 
                         DM_ots_fts['eol-waste-management'].filter({"Variables" : ['vehicles']}),
                         DM_ots_fts['eol-material-recovery'].filter({"Categories1" : ['vehicles']}),
                         CDM_const["material-decomposition_veh"].filter({"Categories1" : ["HDV","LDV","bus"]}),
                         DM_ots_fts['eol-material-recovery'].filter({"Categories1" : ['battery-lion']}),
                         CDM_const["material-decomposition_bat"].filter({"Categories1" : ['battery-lion-HDV', 'battery-lion-LDV']}),
                         DM_material_production["bymat"])
        
    # get material production by technology (writes in DM_material_production)
    DM_material_production["bytech"] = material_production_by_technology(DM_ots_fts['technology-share'], 
                                                                         DM_material_production["bymat"].copy(), # note that here is copy as I cannot overwrite the aggregation between timber and wwp, as I still need timber to be sent to landuse at the end
                                                                         DM_eol["material-recovered"])
    
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
        
    # TODO: the code below on stock and flows will be moved to minerals or tpe
    # # do stock and flows of materials
    # material_flows(DM_transport["tra-stock-veh"].filter({"Categories1" : ['cars-EV', 'cars-FCV', 'cars-ICE', 'trucks-EV', 'trucks-FCV', 'trucks-ICE']}), 
    #                DM_eol["material-towaste"], DM_eol["material-recovered"],
    #                CDM_const["material-decomposition_veh"].filter({"Categories1" : ['cars-EV', 'cars-FCV', 'cars-ICE', 'trucks-EV', 'trucks-FCV', 'trucks-ICE']}),
    #                DM_emissions["bygasmat"])

    # get costs (capex and opex) for material production and carbon catpure
    DM_cost = compute_costs(DM_fxa["cost-matprod"], DM_fxa["cost-CC"], 
                            DM_material_production["bytech"], DM_emissions["capt_w_cc_bytech"])
    
    # get variables for tpe (also writes in DM_cost, dm_bld_matswitch_savings_bymat, DM_emissions and DM_material_production for renaming)
    results_run = variables_for_tpe(DM_cost["material-production_capex"], DM_cost["CO2-capt-w-cc_capex"],
                           DM_emissions["bygas"], DM_material_production["bytech"], 
                           DM_material_production["bymat"], DM_energy_demand["bymat"],
                           DM_energy_demand["bymatcarr"], DM_energy_demand["bioener"])
    
    # interface agriculture
    DM_agr = industry_agriculture_interface(DM_material_production, DM_energy_demand)
    interface.add_link(from_sector='industry', to_sector='agriculture', dm=DM_agr)
    
    # interface ammonia
    DM_amm = industry_ammonia_interface(DM_material_production, DM_energy_demand)
    interface.add_link(from_sector='industry', to_sector='ammonia', dm=DM_amm)
    
    # interface landuse
    DM_lus = industry_landuse_interface(DM_material_production, DM_energy_demand)
    interface.add_link(from_sector='industry', to_sector='land-use', dm=DM_lus)
    
    # interface power
    DM_pow = industry_power_interface(DM_energy_demand)
    interface.add_link(from_sector='industry', to_sector='power', dm=DM_pow)
    
    # interface refinery
    dm_refinery = industry_refinery_interface(DM_energy_demand)
    interface.add_link(from_sector='industry', to_sector='oil-refinery', dm=dm_refinery)
    
    # interface district heating
    dm_dh = industry_district_heating_interface(DM_energy_demand)
    interface.add_link(from_sector='industry', to_sector='district-heating', dm=dm_dh)
    
    # interface emissions
    dm_ems = industry_emissions_interface(DM_emissions)
    interface.add_link(from_sector='industry', to_sector='emissions', dm=dm_ems)
    
    # # interface water
    # dm_water = industry_water_inferface(DM_energy_demand, DM_material_production)
    # interface.add_link(from_sector='industry', to_sector='water', dm=dm_water)
    
    # # interface ccus
    # dm_ccus = industry_ccus_interface(DM_emissions)
    # interface.add_link(from_sector='industry', to_sector='ccus', dm=dm_ccus)
    
    # # interface gtap
    # dm_gtap = industry_gtap_interface(DM_energy_demand, DM_material_production)
    # interface.add_link(from_sector='industry', to_sector='gtap', dm=dm_gtap)
    
    # interface minerals
    DM_ind = industry_minerals_interface(DM_production, DM_eol["veh_eol_to_recycling"])
    interface.add_link(from_sector='industry', to_sector='minerals', dm=DM_ind)
    
    # # interface employment
    # dm_emp = industry_employment_interface(DM_material_demand, DM_energy_demand, DM_material_production, DM_cost, DM_ots_fts)
    # interface.add_link(from_sector='industry', to_sector='employment', dm=dm_emp)
    
    # # interface air pollution
    # dm_airpoll = industry_airpollution_interface(DM_material_production, DM_energy_demand)
    # interface.add_link(from_sector='industry', to_sector='air-pollution', dm=dm_airpoll)
    
    # return
    return results_run
    
def local_industry_run():
    
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
    filter_geoscale(global_vars['geoscale'])

    # run
    results_run = industry(lever_setting, years_setting)
    
    # return
    return results_run

# # run local
# __file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/model/industry_module.py"
# start = time.time()
# results_run = local_industry_run()
# end = time.time()
# print(end-start)

