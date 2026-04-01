from transition_compass_model.model.common.data_matrix_class import DataMatrix
from transition_compass_model.model.common.auxiliary_functions import cdm_to_dm
from transition_compass_model.model.common.auxiliary_functions import calibration_rates, cost
from transition_compass_model.model.common.auxiliary_functions import material_switch, energy_switch
import numpy as np
import re
import warnings

warnings.simplefilter("ignore")
import plotly.io as pio

pio.renderers.default = "browser"


def product_production(
    dm_demand_bld_floor,
    dm_demand_bld_domapp,
    dm_demand_bld_elec,
    dm_demand_tra_infra,
    dm_demand_tra_veh,
    dm_pop,
    dm_demand_packaging_percapita,
    dm_import,
):

    # net import [%] is net import [unit] / demand [unit]
    # production [unit] = demand [unit] - net import [unit]

    # buildings
    dm_netimport_bld_floor = dm_import.filter(
        {"Categories1": ["floor-area-new-residential"]}
    )
    dm_netimport_bld_floor.array = (
        dm_netimport_bld_floor.array * dm_demand_bld_floor.array
    )
    dm_netimport_bld_floor.units["product-net-import"] = dm_demand_bld_floor.units[
        "bld_floor-area_new"
    ]
    dm_prod_bld_floor = dm_demand_bld_floor.copy()
    dm_prod_bld_floor.array = dm_demand_bld_floor.array - dm_netimport_bld_floor.array
    dm_prod_bld_floor.rename_col(
        "bld_floor-area_new", "product-production", "Variables"
    )

    # domapp
    dm_netimport_bld_domapp = dm_import.filter(
        {"Categories1": ["dishwasher", "dryer", "freezer", "fridge", "wmachine"]}
    )
    dm_netimport_bld_domapp.array = (
        dm_netimport_bld_domapp.array * dm_demand_bld_domapp.array
    )
    dm_netimport_bld_domapp.units["product-net-import"] = dm_demand_bld_domapp.units[
        "bld_domapp_new"
    ]
    dm_prod_bld_domapp = dm_demand_bld_domapp.copy()
    dm_prod_bld_domapp.array = (
        dm_demand_bld_domapp.array - dm_netimport_bld_domapp.array
    )
    dm_prod_bld_domapp.rename_col("bld_domapp_new", "product-production", "Variables")

    # electronics
    dm_netimport_bld_elec = dm_import.filter(
        {"Categories1": ["computer", "phone", "tv"]}
    )
    dm_netimport_bld_elec.array = dm_netimport_bld_elec.array * dm_demand_bld_elec.array
    dm_netimport_bld_elec.units["product-net-import"] = dm_demand_bld_elec.units[
        "bld_electronics_new"
    ]
    dm_prod_bld_elec = dm_demand_bld_elec.copy()
    dm_prod_bld_elec.array = dm_demand_bld_elec.array - dm_netimport_bld_elec.array
    dm_prod_bld_elec.rename_col(
        "bld_electronics_new", "product-production", "Variables"
    )

    # transport infra
    dm_netimport_tra_infra = dm_import.filter(
        {"Categories1": ["rail", "road", "trolley-cables"]}
    )
    dm_netimport_tra_infra.array = (
        dm_netimport_tra_infra.array * dm_demand_tra_infra.array
    )
    dm_netimport_tra_infra.units["product-net-import"] = dm_demand_tra_infra.units[
        "tra_product-demand"
    ]
    dm_prod_tra_infra = dm_demand_tra_infra.copy()
    dm_prod_tra_infra.array = dm_demand_tra_infra.array - dm_netimport_tra_infra.array
    dm_prod_tra_infra.rename_col(
        "tra_product-demand", "product-production", "Variables"
    )

    # transport vehicles
    dm_netimport_tra_veh = dm_import.filter_w_regex(
        {"Categories1": "HDV|LDV|bus|planes|ships|trains"}
    )
    dm_netimport_tra_veh.deepen()

    # dm_demand_tra_veh.array = dm_demand_tra_veh.array * (1 - dm_netimport_tra_veh.array)
    # dm_prod_tra_veh = dm_demand_tra_veh.copy()
    # dm_prod_tra_veh.rename_col("tra_product-demand","product-production","Variables")
    # df_temp = dm_prod_tra_veh.write_df()
    # dm_prod_tra_veh.flatten().datamatrix_plot()

    dm_netimport_tra_veh.array = dm_netimport_tra_veh.array * dm_demand_tra_veh.array
    dm_netimport_tra_veh.units["product-net-import"] = dm_demand_tra_veh.units[
        "tra_product-demand"
    ]
    dm_prod_tra_veh = dm_demand_tra_veh.copy()
    dm_prod_tra_veh.array = dm_demand_tra_veh.array - dm_netimport_tra_veh.array
    dm_prod_tra_veh.rename_col("tra_product-demand", "product-production", "Variables")

    # packaging
    dm_demand_pack = dm_demand_packaging_percapita.copy()
    dm_demand_pack.array = dm_demand_pack.array * dm_pop.array[..., np.newaxis]
    dm_demand_pack.units["product-demand"] = "t"
    dm_netimport_pack = dm_import.filter(
        {
            "Categories1": [
                "aluminium-pack",
                "glass-pack",
                "paper-pack",
                "paper-print",
                "paper-san",
                "plastic-pack",
            ]
        }
    )
    dm_netimport_pack.array = dm_netimport_pack.array * dm_demand_pack.array
    dm_netimport_pack.units["product-net-import"] = dm_demand_pack.units[
        "product-demand"
    ]
    dm_prod_pack = dm_demand_pack.copy()
    dm_prod_pack.array = dm_demand_pack.array - dm_netimport_pack.array
    dm_prod_pack.rename_col("product-demand", "product-production", "Variables")

    ########################
    ##### PUT TOGETHER #####
    ########################

    DM_production = {
        "bld-floor": dm_prod_bld_floor,
        "bld-domapp": dm_prod_bld_domapp,
        "bld-electronics": dm_prod_bld_elec,
        "tra-infra": dm_prod_tra_infra,
        "tra-veh": dm_prod_tra_veh,
        "pack": dm_prod_pack,
        "bld-floor-net-import": dm_netimport_bld_floor,
        "bld-domapp-net-import": dm_netimport_bld_domapp,
        "bld-electronics-net-import": dm_netimport_bld_elec,
        "tra-infra-net-import": dm_netimport_tra_infra,
        "tra-veh-net-import": dm_netimport_tra_veh,
        "pack-net-import": dm_netimport_pack,
    }

    # return
    return DM_production


def apply_material_decomposition(
    dm_production_bld_floor,
    dm_production_bld_domapp,
    dm_production_bld_elec,
    dm_production_tra_infra,
    dm_production_tra_veh,
    dm_production_pack,
    cdm_matdec_floor,
    cdm_matdec_domapp,
    cdm_matdec_elec,
    cdm_matdec_tra_infra,
    cdm_matdec_tra_veh,
    cdm_matdec_pack,
    cdm_matdec_tra_bat,
):

    countries = dm_production_bld_floor.col_labels["Country"]
    years = dm_production_bld_floor.col_labels["Years"]

    # material demand [t] = product production [unit] * material decomposition coefficient [t/unit]

    # floor
    dm_bld_floor_matdec = cdm_to_dm(cdm_matdec_floor, countries, years)
    dm_bld_floor_matdec.array = (
        dm_production_bld_floor.array[..., np.newaxis] * dm_bld_floor_matdec.array
    )
    dm_bld_floor_matdec.units["material-decomp"] = "t"

    # domapp
    dm_bld_domapp_matdec = cdm_to_dm(cdm_matdec_domapp, countries, years)
    dm_bld_domapp_matdec.array = (
        dm_production_bld_domapp.array[..., np.newaxis] * dm_bld_domapp_matdec.array
    )
    dm_bld_domapp_matdec.units["material-decomp"] = "t"

    # electronics
    dm_bld_elec_matdec = cdm_to_dm(cdm_matdec_elec, countries, years)
    dm_bld_elec_matdec.array = (
        dm_production_bld_elec.array[..., np.newaxis] * dm_bld_elec_matdec.array
    )
    dm_bld_elec_matdec.units["material-decomp"] = "t"

    # infra
    dm_tra_infra_matdec = cdm_to_dm(cdm_matdec_tra_infra, countries, years)
    dm_tra_infra_matdec.array = (
        dm_production_tra_infra.array[..., np.newaxis] * dm_tra_infra_matdec.array
    )
    dm_tra_infra_matdec.units["material-decomp"] = "t"

    # veh

    # add battery weight
    dm_tra_veh_matdec = cdm_to_dm(cdm_matdec_tra_veh, countries, years)
    dm_tra_bat_matdec = cdm_to_dm(cdm_matdec_tra_bat, countries, years)
    dm_tra_veh_matdec.append(dm_tra_bat_matdec, "Categories1")
    dm_tra_veh_matdec.groupby(
        {
            "HDV": ["HDV", "battery-lion-HDV"],
            "LDV": ["LDV", "battery-lion-LDV"],
            "bus": ["bus", "battery-lion-bus"],
        },
        "Categories1",
        inplace=True,
    )

    # do material decomp
    dm_tra_veh_matdec.array = (
        dm_production_tra_veh.array[..., np.newaxis] * dm_tra_veh_matdec.array
    )
    dm_tra_veh_matdec.units["material-decomp"] = "t"

    # packaging
    dm_pack_matdec = cdm_to_dm(cdm_matdec_pack, countries, years)
    dm_pack_matdec.array = (
        dm_production_pack.array[..., np.newaxis] * dm_pack_matdec.array
    )
    dm_pack_matdec.units["material-decomp"] = "t"

    # put together
    dm_matdec = dm_bld_floor_matdec.copy()
    dm_matdec.append(dm_tra_infra_matdec, dim="Categories1")
    dm_matdec.append(dm_bld_domapp_matdec, dim="Categories1")
    dm_matdec.append(dm_bld_elec_matdec, dim="Categories1")
    dm_temp = dm_tra_veh_matdec.flatten().flatten()
    dm_temp.deepen(based_on="Categories1")
    dm_matdec.append(dm_temp, dim="Categories1")
    dm_matdec.append(dm_pack_matdec, dim="Categories1")
    dm_matdec.sort("Categories1")

    # note: we are calling this material demand as this is the demand of materials
    # that comes from the production sector (e.g. how much material is needed to
    # produce a car)
    DM_material_demand = {"material-demand": dm_matdec}

    # clean
    del dm_bld_floor_matdec, dm_tra_infra_matdec, dm_tra_veh_matdec, dm_pack_matdec

    # return
    return DM_material_demand


def apply_material_switch(
    dm_material_demand,
    dm_material_switch,
    cdm_material_switch,
    DM_input_matswitchimpact,
):

    # material in-to-out [t] = material in [t] * in-to-out [%]
    # material in [t] = material in [t] - material in-to-out [t]
    # material out [t] = material out [t] + material in-to-out [t] * switch ratio [t/t]

    products = dm_material_demand.col_labels["Categories1"]

    #####################
    ##### TRANSPORT #####
    #####################

    cars = list(np.array(products)[[bool(re.search("LDV", p)) for p in products]])
    for car in cars:
        material_switch(
            dm=dm_material_demand,
            dm_ots_fts=dm_material_switch,
            cdm_const=cdm_material_switch,
            material_in="steel",
            material_out=["chem", "aluminium"],
            product=car,
            switch_percentage_prefix="cars-",
            switch_ratio_prefix="material-switch-ratios_",
        )

    trucks = list(np.array(products)[[bool(re.search("HDV", p)) for p in products]])
    for truck in trucks:
        material_switch(
            dm=dm_material_demand,
            dm_ots_fts=dm_material_switch,
            cdm_const=cdm_material_switch,
            material_in="steel",
            material_out=["chem", "aluminium"],
            product=truck,
            switch_percentage_prefix="trucks-",
            switch_ratio_prefix="material-switch-ratios_",
        )

    #####################
    ##### BUILDINGS #####
    #####################

    # new buildings: switch to renewable materials (steel and cement to timber in new residential and non-residential)

    material_switch(
        dm=dm_material_demand,
        dm_ots_fts=dm_material_switch,
        cdm_const=cdm_material_switch,
        material_in="steel",
        material_out=["timber"],
        product="floor-area-new-residential",
        switch_percentage_prefix="build-",
        switch_ratio_prefix="material-switch-ratios_",
        dict_for_output=DM_input_matswitchimpact,
    )

    material_switch(
        dm=dm_material_demand,
        dm_ots_fts=dm_material_switch,
        cdm_const=cdm_material_switch,
        material_in="cement",
        material_out=["timber"],
        product="floor-area-new-residential",
        switch_percentage_prefix="build-",
        switch_ratio_prefix="material-switch-ratios_",
        dict_for_output=DM_input_matswitchimpact,
    )

    # material_switch(dm = dm_material_demand, dm_ots_fts = dm_material_switch,
    #                 cdm_const = cdm_material_switch, material_in = "steel", material_out = ["timber"],
    #                 product = 'floor-area-new-non-residential', switch_percentage_prefix = "build-",
    #                 switch_ratio_prefix = "material-switch-ratios_", dict_for_output = DM_input_matswitchimpact)

    # material_switch(dm = dm_material_demand, dm_ots_fts = dm_material_switch,
    #                 cdm_const = cdm_material_switch, material_in = "cement", material_out = ["timber"],
    #                 product = 'floor-area-new-non-residential', switch_percentage_prefix = "build-",
    #                 switch_ratio_prefix = "material-switch-ratios_", dict_for_output = DM_input_matswitchimpact)

    # renovated buildings: switch to insulated surfaces (chemicals to paper and natural fibers in renovated residential and non-residential)

    material_switch(
        dm=dm_material_demand,
        dm_ots_fts=dm_material_switch,
        cdm_const=cdm_material_switch,
        material_in="chem",
        material_out=["paper", "natfibers"],
        product="floor-area-reno-residential",
        switch_percentage_prefix="reno-",
        switch_ratio_prefix="material-switch-ratios_",
    )

    # material_switch(dm = dm_material_demand, dm_ots_fts = dm_material_switch,
    #                 cdm_const = cdm_material_switch, material_in = "chem", material_out = ["paper","natfibers"],
    #                 product = "floor-area-reno-non-residential", switch_percentage_prefix = "reno-",
    #                 switch_ratio_prefix = "material-switch-ratios_")

    return


def material_production(
    dm_material_efficiency,
    dm_material_net_import,
    dm_material_demand,
    dm_matprod_other_industries,
):

    ######################
    ##### EFFICIENCY #####
    ######################

    # get efficiency coefficients
    dm_temp = dm_material_efficiency.copy()
    dm_temp.filter(
        {"Categories1": dm_material_demand.col_labels["Categories2"]}, inplace=True
    )
    dm_temp.add(0, "Categories1", "natfiber", unit="%", dummy=True)
    dm_temp.sort("Categories1")

    # apply formula to material demand (and overwrite)
    dm_material_demand.array = dm_material_demand.array * (
        1 - dm_temp.array[:, :, :, np.newaxis, :]
    )

    ############################
    ##### AGGREGATE DEMAND #####
    ############################

    # get aggregate demand
    dm_matdec_agg = dm_material_demand.group_all(dim="Categories1", inplace=False)
    dm_matdec_agg.change_unit(
        "material-decomp", factor=1e-3, old_unit="t", new_unit="kt"
    )

    # subset aggregate demand for the materials we keep
    # materials = ['aluminium', 'cement', 'chem', 'copper', 'glass', 'lime', 'paper', 'steel', 'timber']
    dm_material_production_natfiber = dm_matdec_agg.filter(
        {"Categories1": ["natfibers"]}
    )  # this will be used for interface agriculture
    dm_matdec_agg.drop("Categories1", "natfibers")

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
    dm_material_production_bymat.rename_col(
        col_in="material-decomp", col_out="material-production", dim="Variables"
    )

    # get material net import in kilo tonnes
    dm_material_net_import_kt = dm_matdec_agg.copy()
    dm_material_net_import_kt.array = (
        dm_material_net_import_kt.array * dm_material_net_import.array
    )

    # include other industries from fxa
    dm_material_production_bymat.append(
        data2=dm_matprod_other_industries, dim="Categories1"
    )
    dm_material_production_bymat.sort("Categories1")

    # put together
    DM_material_production = {
        "bymat": dm_material_production_bymat,
        "material-net-import": dm_material_net_import_kt,
        "natfiber": dm_material_production_natfiber,
    }

    # clean
    del (
        dm_matdec_agg,
        dm_temp,
        dm_material_production_bymat,
        dm_material_production_natfiber,
    )

    # return
    return DM_material_production


def calibration_material_production(
    DM_cal, dm_material_production_bymat, DM_material_production, years_setting
):

    # get calibration series
    dm_cal_sub = DM_cal["material-production"].copy()
    materials = dm_material_production_bymat.col_labels["Categories1"]
    dm_cal_sub.filter({"Categories1": materials}, inplace=True)

    # get calibration rates
    DM_material_production["calib_rates_bymat"] = calibration_rates(
        dm=dm_material_production_bymat,
        dm_cal=dm_cal_sub,
        calibration_start_year=1990,
        calibration_end_year=2023,
        years_setting=years_setting,
    )

    # do calibration
    dm_material_production_bymat.array = (
        dm_material_production_bymat.array
        * DM_material_production["calib_rates_bymat"].array
    )

    # clean
    del dm_cal_sub, materials

    return


def make_pack_waste(dm_paperpack, dm_pop):
    dm_pack_waste = dm_paperpack.copy()
    dm_pack_waste.rename_col("product-demand", "product-waste", "Variables")
    dm_pack_waste.array = dm_pack_waste.array * dm_pop[..., np.newaxis]
    dm_pack_waste.units["product-waste"] = "t"
    return dm_pack_waste


def end_of_life(
    dm_tra_waste,
    dm_tra_infra_waste,
    dm_bld_waste,
    dm_pack_waste,
    dm_domapp_waste,
    dm_elec_waste,
    dm_waste_management,
    dm_matrec,
    cdm_matdec_floor,
    cdm_matdec_tra_infra,
    cdm_matdec_tra_veh,
    cdm_matdec_pack,
    cdm_matdec_tra_bat,
    cdm_matdec_domapp,
    cdm_matdec_elec,
    dm_material_production_bymat,
):

    # in general:
    # littered + export + uncolleted + collected = 1 (layer 1)
    # recycling + energy recovery + reuse + landfill + incineration = 1 (layer 2)
    # note on incineration: transport will not have incineration, while electric waste yes

    # TODO: do batteries for electronics, and appliances
    # assumption: for the moment we assume 1 battery pack per eol bev vehicle

    # note: we'll do both units and materials (usually it's in materials or weight,
    # but we'll keep units to be used in case later)

    layer1 = ["export", "littered", "waste-collected", "waste-uncollected"]
    layer2 = ["energy-recovery", "incineration", "landfill", "recycling", "reuse"]
    materials = cdm_matdec_tra_veh.col_labels["Categories3"]

    ####################
    ##### VEHICLES #####
    ####################

    # get layers in %
    dm_tra_waste_layer1 = dm_waste_management.filter(
        {"Variables": ["vehicles"], "Categories1": layer1}
    )
    dm_tra_waste_layer2 = dm_waste_management.filter(
        {"Variables": ["vehicles"], "Categories1": layer2}
    )

    # layer 1 units
    arr_temp = (
        dm_tra_waste[..., np.newaxis]
        * dm_tra_waste_layer1[:, :, :, np.newaxis, np.newaxis, :]
    )
    dm_tra_waste_bywsm_layer1 = DataMatrix.based_on(
        arr_temp, dm_tra_waste, {"Categories3": layer1}, units=dm_tra_waste.units
    )

    # layer 1 materials
    # df_temp = cdm_matdec_tra_veh.write_df()
    # df_temp = df_temp.melt()
    arr_temp = (
        dm_tra_waste_bywsm_layer1[..., np.newaxis]
        * cdm_matdec_tra_veh[np.newaxis, np.newaxis, :, :, :, np.newaxis, :]
    )
    dm_tra_waste_bywsm_layer1_bymat = DataMatrix.based_on(
        arr_temp, dm_tra_waste_bywsm_layer1, {"Categories4": materials}, units="t"
    )
    dm_tra_waste_bywsm_layer1_bymat.units["tra_product-waste"] = "t"
    dm_tra_waste_bywsm_layer1_bymat.switch_categories_order(
        "Categories3", "Categories4"
    )

    # layer 2 units
    dm_collected = dm_tra_waste_bywsm_layer1.filter(
        {"Categories3": ["waste-collected"]}
    ).group_all("Categories3", inplace=False)
    arr_temp = (
        dm_collected[..., np.newaxis]
        * dm_tra_waste_layer2[:, :, :, np.newaxis, np.newaxis, :]
    )
    dm_tra_waste_bywsm_layer2 = DataMatrix.based_on(
        arr_temp, dm_collected, {"Categories3": layer2}, units=dm_collected.units
    )

    # layer 2 materials
    dm_collected_bymat = dm_tra_waste_bywsm_layer1_bymat.filter(
        {"Categories4": ["waste-collected"]}
    ).group_all("Categories4", inplace=False)
    arr_temp = (
        dm_collected_bymat[..., np.newaxis]
        * dm_tra_waste_layer2[:, :, :, np.newaxis, np.newaxis, np.newaxis, :]
    )
    dm_tra_waste_bywsm_layer2_bymat = DataMatrix.based_on(
        arr_temp,
        dm_collected_bymat,
        {"Categories4": layer2},
        units=dm_collected_bymat.units,
    )

    ##############################
    ##### BATTERIES VEHICLES #####
    ##############################

    # layer 1 units
    dm_tra_batt_waste_bywsm_layer1 = dm_tra_waste_bywsm_layer1.filter(
        {"Categories1": ["HDV", "LDV", "bus"]}
    )

    # layer 1 materials
    arr_temp = (
        dm_tra_batt_waste_bywsm_layer1[..., np.newaxis]
        * cdm_matdec_tra_bat[np.newaxis, np.newaxis, :, :, :, np.newaxis, :]
    )
    dm_tra_batt_waste_bywsm_layer1_bymat = DataMatrix.based_on(
        arr_temp, dm_tra_batt_waste_bywsm_layer1, {"Categories4": materials}, units="t"
    )
    dm_tra_batt_waste_bywsm_layer1_bymat.units["tra_product-waste"] = "t"
    dm_tra_batt_waste_bywsm_layer1_bymat.switch_categories_order(
        "Categories3", "Categories4"
    )

    # layer 2 units
    dm_tra_batt_waste_bywsm_layer2 = dm_tra_waste_bywsm_layer2.filter(
        {"Categories1": ["HDV", "LDV", "bus"]}
    )

    # layer 2 materials
    dm_collected_bymat = dm_tra_batt_waste_bywsm_layer1_bymat.filter(
        {"Categories4": ["waste-collected"]}
    ).group_all("Categories4", inplace=False)
    arr_temp = (
        dm_collected_bymat[..., np.newaxis]
        * dm_tra_waste_layer2[:, :, :, np.newaxis, np.newaxis, np.newaxis, :]
    )
    dm_tra_batt_waste_bywsm_layer2_bymat = DataMatrix.based_on(
        arr_temp,
        dm_collected_bymat,
        {"Categories4": layer2},
        units=dm_collected_bymat.units,
    )

    ####################################
    ##### TRANSPORT INFRASTRUCTURE #####
    ####################################

    # get layers in %
    dm_tra_infra_waste_layer1 = dm_waste_management.filter(
        {"Variables": ["rail", "road", "trolley-cables"], "Categories1": layer1}
    )
    dm_tra_infra_waste_layer2 = dm_waste_management.filter(
        {"Variables": ["rail", "road", "trolley-cables"], "Categories1": layer2}
    )

    # layer 1 units
    arr_temp = (
        dm_tra_infra_waste[..., np.newaxis]
        * dm_tra_infra_waste_layer1[:, :, np.newaxis, :, :]
    )
    dm_tra_infra_waste_bywsm_layer1 = DataMatrix.based_on(
        arr_temp,
        dm_tra_infra_waste,
        {"Categories2": layer1},
        units=dm_tra_infra_waste.units,
    )

    # layer 1 materials
    arr_temp = (
        dm_tra_infra_waste_bywsm_layer1[..., np.newaxis]
        * cdm_matdec_tra_infra[np.newaxis, np.newaxis, :, :, np.newaxis, :]
    )
    dm_tra_infra_waste_bywsm_layer1_bymat = DataMatrix.based_on(
        arr_temp, dm_tra_infra_waste_bywsm_layer1, {"Categories3": materials}, units="t"
    )
    dm_tra_infra_waste_bywsm_layer1_bymat.units["tra_product-waste"] = "t"
    dm_tra_infra_waste_bywsm_layer1_bymat.switch_categories_order(
        "Categories2", "Categories3"
    )

    # layer 2 units
    dm_collected = dm_tra_infra_waste_bywsm_layer1.filter(
        {"Categories2": ["waste-collected"]}
    ).group_all("Categories2", inplace=False)
    arr_temp = (
        dm_collected[..., np.newaxis]
        * dm_tra_infra_waste_layer2[:, :, np.newaxis, :, :]
    )
    dm_tra_infra_waste_bywsm_layer2 = DataMatrix.based_on(
        arr_temp, dm_collected, {"Categories2": layer2}, units=dm_collected.units
    )

    # layer 2 materials
    dm_collected_bymat = dm_tra_infra_waste_bywsm_layer1_bymat.filter(
        {"Categories3": ["waste-collected"]}
    ).group_all("Categories3", inplace=False)
    arr_temp = (
        dm_collected_bymat[..., np.newaxis]
        * dm_tra_infra_waste_layer2[:, :, np.newaxis, :, np.newaxis, :]
    )
    dm_tra_infra_waste_bywsm_layer2_bymat = DataMatrix.based_on(
        arr_temp,
        dm_collected_bymat,
        {"Categories3": layer2},
        units=dm_collected_bymat.units,
    )

    #####################
    ##### BUILDINGS #####
    #####################

    # get layers in %
    dm_bld_waste_layer1 = dm_waste_management.filter(
        {"Variables": ["floor-area-new-residential"], "Categories1": layer1}
    )
    dm_bld_waste_layer2 = dm_waste_management.filter(
        {"Variables": ["floor-area-new-residential"], "Categories1": layer2}
    )

    # layer 1 units
    arr_temp = (
        dm_bld_waste[..., np.newaxis] * dm_bld_waste_layer1[:, :, :, np.newaxis, :]
    )
    dm_bld_waste_bywsm_layer1 = DataMatrix.based_on(
        arr_temp, dm_bld_waste, {"Categories2": layer1}, units=dm_bld_waste.units
    )

    # layer 1 materials
    cdm_temp = cdm_matdec_floor.filter({"Categories1": ["floor-area-new-residential"]})
    arr_temp = (
        dm_bld_waste_bywsm_layer1[..., np.newaxis]
        * cdm_temp[np.newaxis, np.newaxis, np.newaxis, ...]
    )
    dm_bld_waste_bywsm_layer1_bymat = DataMatrix.based_on(
        arr_temp, dm_bld_waste_bywsm_layer1, {"Categories3": materials}, units="t"
    )
    dm_bld_waste_bywsm_layer1_bymat.units["bld_floor-area_waste"] = "t"
    dm_bld_waste_bywsm_layer1_bymat.switch_categories_order(
        "Categories2", "Categories3"
    )

    # layer 2 units
    dm_collected = dm_bld_waste_bywsm_layer1.filter(
        {"Categories2": ["waste-collected"]}
    ).group_all("Categories2", inplace=False)
    arr_temp = (
        dm_collected[..., np.newaxis] * dm_bld_waste_layer2[:, :, :, np.newaxis, :]
    )
    dm_bld_waste_bywsm_layer2 = DataMatrix.based_on(
        arr_temp, dm_collected, {"Categories2": layer2}, units=dm_collected.units
    )

    # layer 2 materials
    dm_collected_bymat = dm_bld_waste_bywsm_layer1_bymat.filter(
        {"Categories3": ["waste-collected"]}
    ).group_all("Categories3", inplace=False)
    arr_temp = (
        dm_collected_bymat[..., np.newaxis]
        * dm_bld_waste_layer2[:, :, np.newaxis, :, np.newaxis, :]
    )
    dm_bld_waste_bywsm_layer2_bymat = DataMatrix.based_on(
        arr_temp,
        dm_collected_bymat,
        {"Categories3": layer2},
        units=dm_collected_bymat.units,
    )

    ####################
    ##### PACKAGES #####
    ####################

    # get layers in %
    dm_pack_waste_layer1 = dm_waste_management.filter(
        {"Variables": dm_pack_waste.col_labels["Categories1"], "Categories1": layer1}
    )
    dm_pack_waste_layer2 = dm_waste_management.filter(
        {"Variables": dm_pack_waste.col_labels["Categories1"], "Categories1": layer2}
    )

    # layer 1 units
    arr_temp = (
        dm_pack_waste[..., np.newaxis] * dm_pack_waste_layer1[:, :, np.newaxis, :, :]
    )
    dm_pack_waste_bywsm_layer1 = DataMatrix.based_on(
        arr_temp, dm_pack_waste, {"Categories2": layer1}, units=dm_pack_waste.units
    )

    # layer 1 materials
    arr_temp = (
        dm_pack_waste_bywsm_layer1[..., np.newaxis]
        * cdm_matdec_pack[np.newaxis, np.newaxis, :, :, np.newaxis, :]
    )
    dm_pack_waste_bywsm_layer1_bymat = DataMatrix.based_on(
        arr_temp, dm_pack_waste_bywsm_layer1, {"Categories3": materials}, units="t"
    )
    dm_pack_waste_bywsm_layer1_bymat.switch_categories_order(
        "Categories2", "Categories3"
    )

    # layer 2 units
    dm_collected = dm_pack_waste_bywsm_layer1.filter(
        {"Categories2": ["waste-collected"]}
    ).group_all("Categories2", inplace=False)
    arr_temp = (
        dm_collected[..., np.newaxis] * dm_pack_waste_layer2[:, :, np.newaxis, :, :]
    )
    dm_pack_waste_bywsm_layer2 = DataMatrix.based_on(
        arr_temp, dm_collected, {"Categories2": layer2}, units=dm_collected.units
    )

    # layer 2 materials
    dm_collected_bymat = dm_pack_waste_bywsm_layer1_bymat.filter(
        {"Categories3": ["waste-collected"]}
    ).group_all("Categories3", inplace=False)
    arr_temp = (
        dm_collected_bymat[..., np.newaxis]
        * dm_pack_waste_layer2[:, :, np.newaxis, :, np.newaxis, :]
    )
    dm_pack_waste_bywsm_layer2_bymat = DataMatrix.based_on(
        arr_temp,
        dm_collected_bymat,
        {"Categories3": layer2},
        units=dm_collected_bymat.units,
    )

    ###############################
    ##### DOMESTIC APPLIANCES #####
    ###############################

    # get layers in %
    dm_domapp_waste_layer1 = dm_waste_management.filter(
        {"Variables": ["domapp"], "Categories1": layer1}
    )
    dm_domapp_waste_layer2 = dm_waste_management.filter(
        {"Variables": ["domapp"], "Categories1": layer2}
    )

    # layer 1 units
    arr_temp = (
        dm_domapp_waste[..., np.newaxis]
        * dm_domapp_waste_layer1[:, :, :, np.newaxis, :]
    )
    dm_domapp_waste_bywsm_layer1 = DataMatrix.based_on(
        arr_temp, dm_domapp_waste, {"Categories2": layer1}, units=dm_domapp_waste.units
    )

    # layer 1 materials
    arr_temp = (
        dm_domapp_waste_bywsm_layer1[..., np.newaxis]
        * cdm_matdec_domapp[np.newaxis, np.newaxis, :, :, np.newaxis, :]
    )
    dm_domapp_waste_bywsm_layer1_bymat = DataMatrix.based_on(
        arr_temp, dm_domapp_waste_bywsm_layer1, {"Categories3": materials}, units="t"
    )
    dm_domapp_waste_bywsm_layer1_bymat.units["bld_domapp_waste"] = "t"
    dm_domapp_waste_bywsm_layer1_bymat.switch_categories_order(
        "Categories2", "Categories3"
    )

    # layer 2 units
    dm_collected = dm_domapp_waste_bywsm_layer1.filter(
        {"Categories2": ["waste-collected"]}
    ).group_all("Categories2", inplace=False)
    arr_temp = (
        dm_collected[..., np.newaxis] * dm_domapp_waste_layer2[:, :, np.newaxis, :, :]
    )
    dm_domapp_waste_bywsm_layer2 = DataMatrix.based_on(
        arr_temp, dm_collected, {"Categories2": layer2}, units=dm_collected.units
    )

    # layer 2 materials
    dm_collected_bymat = dm_domapp_waste_bywsm_layer1_bymat.filter(
        {"Categories3": ["waste-collected"]}
    ).group_all("Categories3", inplace=False)
    arr_temp = (
        dm_collected_bymat[..., np.newaxis]
        * dm_domapp_waste_layer2[:, :, :, np.newaxis, np.newaxis, :]
    )
    dm_domapp_waste_bywsm_layer2_bymat = DataMatrix.based_on(
        arr_temp,
        dm_collected_bymat,
        {"Categories3": layer2},
        units=dm_collected_bymat.units,
    )

    #######################
    ##### ELECTRONICS #####
    #######################

    # get layers in %
    dm_elec_waste_layer1 = dm_waste_management.filter(
        {"Variables": ["electronics"], "Categories1": layer1}
    )
    dm_elec_waste_layer2 = dm_waste_management.filter(
        {"Variables": ["electronics"], "Categories1": layer2}
    )

    # layer 1 units
    arr_temp = (
        dm_elec_waste[..., np.newaxis] * dm_elec_waste_layer1[:, :, :, np.newaxis, :]
    )
    dm_elec_waste_bywsm_layer1 = DataMatrix.based_on(
        arr_temp, dm_elec_waste, {"Categories2": layer1}, units=dm_elec_waste.units
    )

    # layer 1 materials
    arr_temp = (
        dm_elec_waste_bywsm_layer1[..., np.newaxis]
        * cdm_matdec_elec[np.newaxis, np.newaxis, :, :, np.newaxis, :]
    )
    dm_elec_waste_bywsm_layer1_bymat = DataMatrix.based_on(
        arr_temp, dm_elec_waste_bywsm_layer1, {"Categories3": materials}, units="t"
    )
    dm_elec_waste_bywsm_layer1_bymat.units["bld_electronics_waste"] = "t"
    dm_elec_waste_bywsm_layer1_bymat.switch_categories_order(
        "Categories2", "Categories3"
    )

    # layer 2 units
    dm_collected = dm_elec_waste_bywsm_layer1.filter(
        {"Categories2": ["waste-collected"]}
    ).group_all("Categories2", inplace=False)
    arr_temp = (
        dm_collected[..., np.newaxis] * dm_elec_waste_layer2[:, :, :, np.newaxis, :]
    )
    dm_elec_waste_bywsm_layer2 = DataMatrix.based_on(
        arr_temp, dm_collected, {"Categories2": layer2}, units=dm_collected.units
    )

    # layer 2 materials
    dm_collected_bymat = dm_elec_waste_bywsm_layer1_bymat.filter(
        {"Categories3": ["waste-collected"]}
    ).group_all("Categories3", inplace=False)
    arr_temp = (
        dm_collected_bymat[..., np.newaxis]
        * dm_elec_waste_layer2[:, :, :, np.newaxis, np.newaxis, :]
    )
    dm_elec_waste_bywsm_layer2_bymat = DataMatrix.based_on(
        arr_temp,
        dm_collected_bymat,
        {"Categories3": layer2},
        units=dm_collected_bymat.units,
    )

    #############################
    ##### MATERIAL RECOVERY #####
    #############################

    # infra
    dm_rec = dm_tra_infra_waste_bywsm_layer2_bymat.filter(
        {"Categories3": ["recycling"]}
    ).group_all("Categories3", inplace=False)
    dm_matrec_inf = dm_matrec.filter(
        {"Categories1": ["rail", "road", "trolley-cables"]}
    )
    dm_rec.array = dm_rec.array * dm_matrec_inf.array
    dm_rec.rename_col(
        dm_rec.col_labels["Variables"][0], "material-recovered", "Variables"
    )

    # buildings
    dm_temp = dm_bld_waste_bywsm_layer2_bymat.filter(
        {"Categories3": ["recycling"]}
    ).group_all("Categories3", inplace=False)
    dm_matrec_bld = dm_matrec.filter({"Categories1": ["floor-area"]})
    dm_temp.array = dm_temp.array * dm_matrec_bld.array
    dm_temp.rename_col(
        dm_temp.col_labels["Variables"][0], "material-recovered", "Variables"
    )
    dm_temp.rename_col("residential", "floor-area", "Categories1")
    dm_rec.append(dm_temp, "Categories1")

    # packaging
    dm_temp = dm_pack_waste_bywsm_layer2_bymat.filter(
        {"Categories3": ["recycling"]}
    ).group_all("Categories3", inplace=False)
    dm_matrec_pack = dm_matrec.filter(
        {
            "Categories1": [
                "aluminium-pack",
                "glass-pack",
                "paper-pack",
                "paper-print",
                "paper-san",
                "plastic-pack",
            ]
        }
    )
    dm_temp.array = dm_temp.array * dm_matrec_pack.array
    dm_temp.rename_col(
        dm_temp.col_labels["Variables"][0], "material-recovered", "Variables"
    )
    dm_rec.append(dm_temp, "Categories1")

    # domestic appliances
    dm_temp = dm_domapp_waste_bywsm_layer2_bymat.filter(
        {"Categories3": ["recycling"]}
    ).group_all("Categories3", inplace=False)
    dm_matrec_domapp = dm_matrec.filter(
        {"Categories1": ["dishwasher", "dryer", "freezer", "fridge", "wmachine"]}
    )
    dm_temp.array = dm_temp.array * dm_matrec_domapp.array
    dm_temp.rename_col(
        dm_temp.col_labels["Variables"][0], "material-recovered", "Variables"
    )
    dm_rec.append(dm_temp, "Categories1")

    # electronics
    dm_temp = dm_elec_waste_bywsm_layer2_bymat.filter(
        {"Categories3": ["recycling"]}
    ).group_all("Categories3", inplace=False)
    dm_matrec_elec = dm_matrec.filter({"Categories1": ["electronics"]})
    dm_temp.array = dm_temp.array * dm_matrec_elec.array
    dm_temp.rename_col(
        dm_temp.col_labels["Variables"][0], "material-recovered", "Variables"
    )
    dm_rec.append(dm_temp, "Categories1")

    # vehicles
    dm_rec_veh = dm_tra_waste_bywsm_layer2_bymat.filter(
        {"Categories1": ["HDV", "LDV", "bus"], "Categories4": ["recycling"]}
    ).group_all("Categories4", inplace=False)
    dm_matrec_veh = dm_matrec.filter({"Categories1": ["vehicles"]})
    dm_rec_veh.array = dm_rec_veh.array * dm_matrec_veh.array[:, :, :, :, np.newaxis, :]
    dm_temp = dm_tra_waste_bywsm_layer2_bymat.filter(
        {"Categories1": ["planes", "ships", "trains"], "Categories4": ["recycling"]}
    ).group_all("Categories4", inplace=False)
    dm_matrec_veh2 = dm_matrec.filter({"Categories1": ["planes", "ships", "trains"]})
    dm_temp.array = dm_temp.array * dm_matrec_veh2.array[:, :, :, :, np.newaxis, :]
    dm_rec_veh.append(dm_temp, "Categories1")

    # batteries from vehicles
    dm_temp = dm_tra_batt_waste_bywsm_layer2_bymat.filter(
        {"Categories4": ["recycling"]}
    ).group_all("Categories4", inplace=False)
    dm_matrec_bat = dm_matrec.filter({"Categories1": ["battery-lion"]})
    dm_temp.array = dm_temp.array * dm_matrec_bat.array[:, :, :, :, np.newaxis, :]
    dm_temp.rename_col(
        ["HDV", "LDV", "bus"],
        ["battery-HDV", "battery-LDV", "battery-bus"],
        "Categories1",
    )
    dm_rec_veh.append(dm_temp, "Categories1")
    dm_rec_veh.rename_col("tra_product-waste", "material-recovered", "Variables")

    # put together
    dm_rec_veh.array[dm_rec_veh.array == 0] = np.nan
    dm_rec_veh = dm_rec_veh.flatten().flatten()
    dm_rec_veh.deepen()
    missing = np.array(materials)[
        [m not in dm_rec_veh.col_labels["Categories2"] for m in materials]
    ]
    dm_rec_veh.add(0, "Categories2", missing.tolist(), dummy=True)
    dm_rec_veh.sort("Categories2")
    dm_rec.append(dm_rec_veh, "Categories1")
    dm_rec.sort("Categories1")

    # aggregate
    dm_rec_agg = dm_rec.group_all("Categories1", inplace=False)
    dm_rec_agg.change_unit(
        "material-recovered", factor=1e-3, old_unit="t", new_unit="kt"
    )

    # allocate "recycled cement" to other materials
    # note:
    # Cement production: limestone + clay → kiln (1450 °C) → clinker → ground → cement
    # Cement → not directly recyclable; only partially recoverable via thermal/mechanical or SCM routes, mostly still emerging
    # Concrete is done by mixing cement, sand, water and aggregates, and it is the main construction material used in buildings
    # Concrete can be recycled from eol buildings, and usually it's downcycled to make roads, base, etc
    # what we call here "recycled cement" is recovered concrete from eol buildings, so for now I allocate it to other recovered materials
    dm_rec_agg[..., "other"] = dm_rec_agg[..., "other"] + dm_rec_agg[..., "cement"]
    dm_rec_agg[..., "cement"] = 0

    # # TODO: tbc but Switzerland in theroy does not have facilities to recycle non-ferrous metals like
    # # aluminium and copper, so those should be set to zero
    # if "Switzerland" in dm_rec_agg.col_labels["Country"]:
    #     dm_rec_agg["Switzerland",:,:,"aluminium"] = 0
    #     dm_rec_agg["Switzerland",:,:,"copper"] = 0

    # if material recovered is larger than material produced, impose material recovered to be equal to material produced
    dm_temp = dm_rec_agg.copy()
    dm_temp1 = dm_material_production_bymat.filter({"Categories1": materials})
    dm_temp.array = (
        dm_rec_agg.array - dm_temp1.array
    )  # create a dm in which you do the difference
    dm_temp.array[dm_temp.array > 0] = (
        0  # where the difference is > 0, put difference = 0
    )
    dm_temp.array = (
        dm_temp.array + dm_temp1.array
    )  # sum back the material production, so where the difference > 0, the value of material recovered now equals the value of material production
    dm_rec_agg_corrected = dm_temp
    dm_rec_agg_corrected.array[np.isnan(dm_rec_agg_corrected.array)] = 0

    # checks
    # dm_rec_agg.flatten().filter({"Country":["EU27"]}).datamatrix_plot()
    # dm_rec_agg_corrected.flatten().filter({"Country":["EU27"]}).datamatrix_plot()
    # dm_material_production_bymat.flatten().filter({"Country":["EU27"]}).datamatrix_plot()

    #################################################
    ##### ADJUST WASTE WITH WASTE FROM RECOVERY #####
    #################################################

    # turn dm_rec_agg_corrected back to product level
    dm_rec_corrected = dm_rec.copy()
    dm_rec_corrected.switch_categories_order("Categories1", "Categories2")
    dm_rec_corrected.normalise("Categories2")
    dm_temp = dm_rec_agg_corrected.copy()
    dm_rec_corrected.array = dm_rec_corrected.array * dm_temp.array[..., np.newaxis]
    # print(np.allclose(dm_rec_corrected.group_all("Categories2", inplace=False).array, dm_rec_agg_corrected.array))
    dm_rec_corrected.switch_categories_order("Categories1", "Categories2")

    # before aggregating, create missing products as zeroes (otherwise aggregating function will return an error)
    # note: this is due to the flatten above (and I prefer introducing nas above as otherwise we have
    # big matrixes for all vehicles' types).
    dict_agg = {
        "vehicles": "bus|HDV|LDV",
        "trains": "trains",
        "planes": "planes",
        "ships": "ships",
        "battery-lion": "battery",
        "electronics": "computer|phone|tv",
    }
    mysearch = np.concatenate(
        [dict_agg[key].split("|") for key in list(dict_agg.keys())]
    ).tolist()
    products = dm_rec_corrected.col_labels["Categories1"]
    prod_idx = [
        not any([bool(re.search(my, prod)) for prod in products]) for my in mysearch
    ]
    prod_missing = np.array(mysearch)[prod_idx].tolist()
    dm_rec_corrected.add(0, "Categories1", prod_missing, dummy=True)

    dm_rec_corrected.groupby(dict_agg, "Categories1", regex=True, inplace=True)
    dm_rec_corrected.sort("Categories1")
    dm_rec_corrected.units = dm_rec.units

    # put in waste material that is not recovered
    # formulas:
    # recovered = quantity * param -> quantity = recovered / param
    # extra waste = quantity * (1 - param)
    # -> extra waste = recovered / param * (1 - param)
    dm_param = dm_matrec.copy()
    dm_extra_wst = dm_rec_corrected.copy()
    dm_extra_wst.array = dm_rec_corrected.array / dm_param.array * (1 - dm_param.array)

    # create waste by material
    dm_waste_bywsm_layer2_bymat = dm_elec_waste_bywsm_layer2_bymat.copy()
    dm_waste_bywsm_layer2_bymat.rename_col(
        dm_waste_bywsm_layer2_bymat.col_labels["Variables"][0], "waste", "Variables"
    )
    dm_list = [
        dm_bld_waste_bywsm_layer2_bymat,
        dm_domapp_waste_bywsm_layer2_bymat,
        dm_pack_waste_bywsm_layer2_bymat,
        dm_tra_infra_waste_bywsm_layer2_bymat,
    ]
    for dm in dm_list:
        dm.rename_col(dm.col_labels["Variables"][0], "waste", "Variables")
        dm_waste_bywsm_layer2_bymat.append(dm, "Categories1")
    dm_waste_bywsm_layer2_bymat.group_all("Categories1")

    dm_temp = dm_tra_batt_waste_bywsm_layer2_bymat.group_all(
        "Categories2", inplace=False
    ).group_all("Categories1", inplace=False)
    dm_temp.rename_col("tra_product-waste", "tra_batt-waste", "Variables")
    dm_waste_bywsm_layer2_bymat.append(dm_temp, "Variables")
    dm_temp = dm_tra_waste_bywsm_layer2_bymat.group_all(
        "Categories2", inplace=False
    ).group_all("Categories1", inplace=False)
    dm_waste_bywsm_layer2_bymat.append(dm_temp, "Variables")
    dm_waste_bywsm_layer2_bymat.groupby(
        {"material-to-waste": ["waste", "tra_batt-waste", "tra_product-waste"]},
        "Variables",
        inplace=True,
    )
    dm_waste_bywsm_layer2_bymat.change_unit("material-to-waste", 1e-3, "t", "kt")

    # adjust waste by material with extra waste from recovery
    dm_temp_share = dm_waste_bywsm_layer2_bymat.filter(
        {"Categories2": ["energy-recovery", "incineration", "landfill"]}
    )
    dm_extra_wst_split = dm_temp_share.copy()
    dm_temp_share.normalise("Categories2")
    dm_extra_wst.group_all("Categories1")
    dm_extra_wst_split.array = dm_extra_wst.array[..., np.newaxis] * dm_temp_share.array
    dm_extra_wst_split.rename_col(
        "material-to-waste", "material-to-waste-extra", "Variables"
    )
    dm_extra_wst_split.add(0, "Categories2", ["recycling", "reuse"], dummy=True)
    dm_extra_wst_split.sort("Categories2")
    dm_waste_bywsm_layer2_bymat.append(dm_extra_wst_split, "Variables")
    dm_waste_bywsm_layer2_bymat.groupby(
        {"material-to-waste": ["material-to-waste", "material-to-waste-extra"]},
        "Variables",
        inplace=True,
    )

    # save
    DM_eol = {
        # "material-towaste": dm_transport_waste_bymat,
        # "material-towaste-collected" : dm_transport_waste_collect_bymat,
        "material-recovered": dm_rec_agg_corrected,
        # "veh_eol_to_wst_mgt" : dm_transport_waste_bywsm_layer1,
        # "veh_el_to_collection" : dm_transport_waste_bywsm_layer2,
        "veh_eol_to_recycling": dm_tra_waste_bywsm_layer2.filter(
            {"Categories3": ["recycling"]}
        ).group_all("Categories3", inplace=False),
    }

    return DM_eol


def material_production_by_technology(
    dm_technology_share, dm_material_production_bymat, dm_material_recovered
):

    # Note: for paper, pulp is the raw material for making paper. When we recycle
    # paper, the EOL paper is turned into pulp, which then is turned into paper.

    # subtract material recovered from material production
    # note: in eol we have already adjusted for the cases when recovery > production (we have set recovery = production)
    dm_material_production_bymat_primary = dm_material_production_bymat.copy()
    materials_recovered = dm_material_recovered.col_labels["Categories1"]
    for m in materials_recovered:
        dm_material_production_bymat_primary[..., m] = (
            dm_material_production_bymat_primary[..., m] - dm_material_recovered[..., m]
        )

    # add timber to wood and wooden products (wwp), and other ot other industrial sector (ois)
    dm_material_production_bymat.groupby(
        {"wwp": "timber|wwp", "ois": "other|ois"},
        dim="Categories1",
        aggregation="sum",
        regex=True,
        inplace=True,
    )
    dm_material_production_bymat_primary.groupby(
        {"wwp": "timber|wwp", "ois": "other|ois"},
        dim="Categories1",
        aggregation="sum",
        regex=True,
        inplace=True,
    )

    # make material production by tech
    dm_material_production_bytech = dm_material_production_bymat_primary.copy()
    dm_material_production_bytech.rename_col(
        [
            "aluminium",
            "cement",
            "chem",
            "copper",
            "fbt",
            "glass",
            "lime",
            "mae",
            "ois",
            "paper",
            "steel",
            "textiles",
            "tra-equip",
            "wwp",
        ],
        [
            "aluminium-prim",
            "cement-dry-kiln",
            "chem-chem-tech",
            "copper-tech",
            "fbt-tech",
            "glass-glass",
            "lime-lime",
            "mae-tech",
            "ois-tech",
            "paper-tech",
            "steel-BF-BOF",
            "textiles-tech",
            "tra-equip-tech",
            "wwp-tech",
        ],
        "Categories1",
    )

    # add primary techs for cement and steel
    dict_missing = {
        "cement-dry-kiln": ["cement-geopolym", "cement-wet-kiln"],
        "steel-BF-BOF": ["steel-hisarna", "steel-hydrog-DRI"],
    }
    for key in dict_missing:
        missing = dict_missing[key]
        for m in missing:
            dm_temp = dm_material_production_bytech.filter({"Categories1": [key]})
            dm_temp.rename_col(key, m, "Categories1")
            dm_material_production_bytech.append(dm_temp, "Categories1")
    dm_material_production_bytech.sort("Categories1")
    primary_techs = [
        "cement-dry-kiln",
        "cement-geopolym",
        "cement-wet-kiln",
        "steel-BF-BOF",
        "steel-hisarna",
        "steel-hydrog-DRI",
    ]
    dm_temp = dm_material_production_bytech.filter({"Categories1": primary_techs})
    dm_temp.sort("Categories1")
    dm_technology_share.sort("Categories1")
    dm_temp.array = dm_temp.array * dm_technology_share.array
    dm_material_production_bytech.drop("Categories1", primary_techs)
    dm_material_production_bytech.append(dm_temp, "Categories1")
    dm_material_production_bytech.sort("Categories1")

    # add secondary techs
    dm_material_recovered.rename_col(["other", "timber"], ["ois", "wwp"], "Categories1")
    secondary_techs = dm_material_recovered.col_labels["Categories1"]
    for s in secondary_techs:
        dm_material_recovered.rename_col(s, s + "-sec", "Categories1")
    dm_material_recovered.rename_col(
        "paper-sec", "pulp-tech", "Categories1"
    )  # recycled paper is recycled pulp, which will then turned back to paper
    dm_material_recovered.rename_col(
        "steel-sec", "steel-scrap-EAF", "Categories1"
    )  # secondary steel is made is electric arc furnace using scrap steel
    dm_material_recovered.rename_col(
        "material-recovered", "material-production", "Variables"
    )
    dm_material_production_bytech.append(dm_material_recovered, "Categories1")
    dm_material_production_bytech.sort("Categories1")

    # TODO: you'll have to update names and review some techs (for example wwp-sec would be timber-sec,
    # and you would need to get the properties of that technology, i.e. energy consumed, emissions, etc)

    # # checks
    # # production = primary + secondary
    # dm_material_production_bytech.array[dm_material_production_bytech.array < 0]
    # dict_aggregation = {"aluminium" : ['aluminium-prim', 'aluminium-sec'],
    #                     "cement" : ['cement-dry-kiln', 'cement-geopolym', 'cement-sec', 'cement-wet-kiln'],
    #                     "chem" : ['chem-chem-tech', 'chem-sec'],
    #                     "copper" : ['copper-sec', 'copper-tech'],
    #                     "glass" : ['glass-glass', 'glass-sec'],
    #                     "lime" : ['lime-lime', 'lime-sec'],
    #                     "paper" : ['paper-tech', 'pulp-tech'],
    #                     "steel" : ['steel-BF-BOF', 'steel-hisarna', 'steel-hydrog-DRI', 'steel-scrap-EAF'],
    #                     "wwp" : ['wwp-sec', 'wwp-tech']}
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

    # drop post consumer techs for lime as recycling lime does not seem feasible with current techs
    dm_material_production_bytech.drop("Categories1", ["lime-sec"])

    # get energy demand for material production by technology both without and with feedstock
    feedstock = ["excl-feedstock", "feedstock"]
    DM_energy_demand = {}

    for f in feedstock:

        # get constants for energy demand for material production by technology
        cdm_temp = CDM_const["energy_" + f]

        # create dm for energy demand for material production by technology
        dm_energy_demand = cdm_to_dm(
            cdm_temp,
            dm_material_production_bytech.col_labels["Country"],
            dm_material_production_bytech.col_labels["Years"],
        )

        # get energy demand for material production by technology
        dm_temp = dm_material_production_bytech.copy()
        dm_temp.change_unit(
            "material-production", factor=1e-3, old_unit="kt", new_unit="Mt"
        )
        dm_energy_demand.array = dm_energy_demand.array * dm_temp.array[..., np.newaxis]
        dm_energy_demand.units["energy-demand-" + f] = "TWh"
        DM_energy_demand[f + "_bytechcarr"] = dm_energy_demand

    # # get overall energy demand
    # dm_energy_demand_temp = DM_energy_demand["excl-feedstock_bytechcarr"].copy()
    # dm_energy_demand_temp.append(DM_energy_demand["feedstock_bytechcarr"], dim = "Variables")
    # dm_energy_demand_bytechcarr = DM_energy_demand["excl-feedstock_bytechcarr"].copy()
    # dm_energy_demand_bytechcarr.array = np.nansum(dm_energy_demand_temp.array, axis = -3, keepdims= True) # here we are summing feedstock and excluding feedstock together
    # dm_energy_demand_bytechcarr.rename_col(col_in = 'energy-demand-excl-feedstock', col_out = "energy-demand", dim = "Variables")
    # DM_energy_demand["total_bytechcarr"] = dm_energy_demand_bytechcarr.copy()
    # DM_energy_demand["total_bycarr"] = DM_energy_demand["total_bytechcarr"].group_all(dim='Categories1', inplace=False)

    # aggregate energy demand by energy carrier
    DM_energy_demand["excl-feedstock_bycarr"] = DM_energy_demand[
        "excl-feedstock_bytechcarr"
    ].group_all(dim="Categories1", inplace=False)

    # return
    return DM_energy_demand


def calibration_energy_demand(
    DM_cal,
    dm_energy_demand_exclfeedstock_bycarr,
    dm_energy_demand_exclfeedstock_bytechcarr,
    DM_energy_demand,
    years_setting,
):

    # this is by material-technology and carrier

    # get calibration rates
    dm_energy_demand_calib_rates_bycarr = calibration_rates(
        dm=dm_energy_demand_exclfeedstock_bycarr.copy(),
        dm_cal=DM_cal["energy-demand"].copy(),
        calibration_start_year=2000,
        calibration_end_year=2021,
        years_setting=years_setting,
    )

    # # FIXME!: before 2000, instead of 1 put the calib rate of 2000 (it's done like this in the KNIME for industry, tbc what to do)
    # idx = dm_energy_demand_calib_rates_bycarr.idx
    # years_bef2000 = np.array(range(1990, 2000, 1)).tolist()
    # for i in years_bef2000:
    #     dm_energy_demand_calib_rates_bycarr.array[:,idx[i],...] = dm_energy_demand_calib_rates_bycarr.array[:,idx[2000],...]

    # store dm_energy_demand_calib_rates_bycarr
    DM_energy_demand["calib_rates_bycarr"] = dm_energy_demand_calib_rates_bycarr.copy()

    # do calibration
    dm_energy_demand_exclfeedstock_bycarr.array = (
        dm_energy_demand_exclfeedstock_bycarr.array
        * dm_energy_demand_calib_rates_bycarr.array
    )

    # do calibration for each technology (by applying aggregate calibration rates)
    dm_energy_demand_exclfeedstock_bytechcarr.array = (
        dm_energy_demand_exclfeedstock_bytechcarr.array
        * dm_energy_demand_calib_rates_bycarr.array[:, :, :, np.newaxis, :]
    )

    # return
    return


def technology_development(dm_technology_development, dm_energy_demand_bytechcarr):

    # get energy demand after technology development (tech dev improves energy efficiency)
    dm_energy_demand_bytechcarr.array = dm_energy_demand_bytechcarr.array * (
        1 - dm_technology_development.array[..., np.newaxis]
    )

    # return
    return


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
    energy_switch(
        dm_energy_demand=dm_energy_demand_bytechcarr,
        dm_energy_carrier_mix=dm_temp,
        carrier_in=carrier_in,
        carrier_out="electricity",
        dm_energy_carrier_mix_prefix="to-electricity",
    )

    ####################
    ##### HYDROGEN #####
    ####################

    carrier_in = dm_energy_demand_bytechcarr.col_labels["Categories2"].copy()
    carrier_in.remove("electricity")
    carrier_in.remove("hydrogen")
    energy_switch(
        dm_energy_demand=dm_energy_demand_bytechcarr,
        dm_energy_carrier_mix=dm_temp,
        carrier_in=carrier_in,
        carrier_out="hydrogen",
        dm_energy_carrier_mix_prefix="to-hydrogen",
    )

    ###############
    ##### GAS #####
    ###############

    energy_switch(
        dm_energy_demand=dm_energy_demand_bytechcarr,
        dm_energy_carrier_mix=dm_temp,
        carrier_in=["solid-ff-coal"],
        carrier_out="gas-ff-natural",
        dm_energy_carrier_mix_prefix="solid-to-gas",
    )

    energy_switch(
        dm_energy_demand=dm_energy_demand_bytechcarr,
        dm_energy_carrier_mix=dm_temp,
        carrier_in=["liquid-ff-oil"],
        carrier_out="gas-ff-natural",
        dm_energy_carrier_mix_prefix="liquid-to-gas",
    )

    ###########################
    ##### SYNTHETIC FUELS #####
    ###########################

    # TODO: TO BE DONE

    #####################
    ##### BIO FUELS #####
    #####################

    energy_switch(
        dm_energy_demand=dm_energy_demand_bytechcarr,
        dm_energy_carrier_mix=dm_temp,
        carrier_in=["solid-ff-coal"],
        carrier_out="solid-bio",
        dm_energy_carrier_mix_prefix="to-biomass",
    )

    energy_switch(
        dm_energy_demand=dm_energy_demand_bytechcarr,
        dm_energy_carrier_mix=dm_temp,
        carrier_in=["liquid-ff-oil"],
        carrier_out="liquid-bio",
        dm_energy_carrier_mix_prefix="to-biomass",
    )

    energy_switch(
        dm_energy_demand=dm_energy_demand_bytechcarr,
        dm_energy_carrier_mix=dm_temp,
        carrier_in=["gas-ff-natural"],
        carrier_out="gas-bio",
        dm_energy_carrier_mix_prefix="to-biomass",
    )

    # clean
    del dm_temp, carrier_in

    # return
    return


def add_specific_energy_demands(
    dm_energy_demand_exclfeedstock_bytechcarr,
    dm_energy_demand_feedstock_bytechcarr,
    DM_energy_demand,
    dict_groupby,
):

    # get demand for biomaterial from feedstock
    dm_energy_demand_feedstock_bycarr = dm_energy_demand_feedstock_bytechcarr.group_all(
        "Categories1", inplace=False
    )
    dm_energy_demand_feedstock_bybiomat = dm_energy_demand_feedstock_bycarr.filter(
        {"Categories1": ["solid-bio", "gas-bio", "liquid-bio"]}
    )

    # get total energy demand
    dm_energy_demand_bytechcarr = dm_energy_demand_exclfeedstock_bytechcarr.copy()
    dm_energy_demand_bytechcarr.append(
        dm_energy_demand_feedstock_bytechcarr, "Variables"
    )
    dm_energy_demand_bytechcarr.groupby(
        {"energy-demand": ["energy-demand-excl-feedstock", "energy-demand-feedstock"]},
        "Variables",
        inplace=True,
    )

    # get demand for industrial waste
    dm_energy_demand_bycarr = dm_energy_demand_bytechcarr.group_all(
        "Categories1", inplace=False
    )
    dm_energy_demand_indwaste = dm_energy_demand_bycarr.filter(
        {"Categories1": ["solid-waste"]}
    )

    # get demand for bioenergy solid, bioenergy gas, bioenergy liquid
    dm_energy_demand_bioener_bybiomat = dm_energy_demand_bycarr.filter(
        {"Categories1": ["solid-bio", "gas-bio", "liquid-bio"]}
    )
    dm_energy_demand_bioener_bybiomat.rename_col(
        "energy-demand", "energy-demand_bioenergy", "Variables"
    )
    dm_energy_demand_bioener = dm_energy_demand_bioener_bybiomat.group_all(
        "Categories1", inplace=False
    )

    # get demand by material
    dm_energy_demand_bymatcarr = dm_energy_demand_bytechcarr.groupby(
        dict_groupby, dim="Categories1", aggregation="sum", regex=True, inplace=False
    )
    dm_energy_demand_bymat = dm_energy_demand_bymatcarr.group_all(
        "Categories2", inplace=False
    )

    # get demand by carrier
    dm_energy_demand_bycarr = dm_energy_demand_bymatcarr.group_all(
        "Categories1", inplace=False
    )

    # get energy demand by tech
    dm_energy_demand_bytech = dm_energy_demand_bytechcarr.group_all(
        "Categories2", inplace=False
    )

    # put in DM
    DM_energy_demand["bymatcarr"] = dm_energy_demand_bymatcarr
    DM_energy_demand["feedstock_bybiomat"] = dm_energy_demand_feedstock_bybiomat
    DM_energy_demand["indwaste"] = dm_energy_demand_indwaste
    DM_energy_demand["bioener_bybiomat"] = dm_energy_demand_bioener_bybiomat
    DM_energy_demand["bioener"] = dm_energy_demand_bioener

    DM_energy_demand["bymat"] = dm_energy_demand_bymat
    DM_energy_demand["bycarr"] = dm_energy_demand_bycarr
    DM_energy_demand["bytech"] = dm_energy_demand_bytech

    # return
    return


def emissions(
    cdm_const_emission_factor_process,
    cdm_const_emission_factor,
    dm_energy_demand_exclfeedstock_bytechcarr,
    dm_material_production_bytech,
):

    # get emission factors
    cdm_temp1 = cdm_const_emission_factor_process
    cdm_temp2 = cdm_const_emission_factor

    # emissions = energy demand * emission factor

    # combustion
    dm_emissions_combustion = dm_energy_demand_exclfeedstock_bytechcarr.copy()
    dm_emissions_combustion.rename_col(
        "energy-demand-excl-feedstock", "emissions", "Variables"
    )
    dm_emissions_combustion.units["emissions"] = "Mt"
    dm_emissions_combustion.rename_col("emissions", "emissions_CH4", "Variables")
    dm_emissions_combustion.deepen("_", based_on="Variables")
    arr_temp = dm_emissions_combustion.array
    dm_emissions_combustion.add(arr_temp, "Categories3", "CO2")
    dm_emissions_combustion.add(arr_temp, "Categories3", "N2O")
    dm_emissions_combustion.array = (
        dm_emissions_combustion.array
        * cdm_temp2.array[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    )

    # biogenic total
    bio = ["gas-bio", "liquid-bio", "solid-bio"]
    dm_emissions_combustion_bio = dm_emissions_combustion.filter(
        {"Categories2": bio}, inplace=False
    )
    dm_emissions_combustion_bio.group_all("Categories2")
    dm_emissions_combustion_bio.switch_categories_order("Categories2", "Categories1")
    dm_emissions_combustion_bio.rename_col(
        "emissions", "emissions-biogenic", dim="Variables"
    )

    # process
    dm_emissions_process = dm_material_production_bytech.copy()
    dm_emissions_process.change_unit(
        "material-production", factor=1e-3, old_unit="kt", new_unit="Mt"
    )
    dm_emissions_process.rename_col(
        "material-production", "emissions-process_CH4", "Variables"
    )
    dm_emissions_process.deepen("_", based_on="Variables")
    arr_temp = dm_emissions_process.array
    dm_emissions_process.add(arr_temp, "Categories2", "CO2")
    dm_emissions_process.add(arr_temp, "Categories2", "N2O")
    dm_emissions_process.array = (
        dm_emissions_process.array * cdm_temp1.array[np.newaxis, np.newaxis, ...]
    )

    # total emissions per technology
    dm_emissions_bygastech = dm_emissions_combustion.group_all(
        "Categories2", inplace=False
    )
    dm_emissions_bygastech.append(dm_emissions_process, dim="Variables")
    dm_emissions_bygastech.add(
        np.nansum(dm_emissions_bygastech.array, -3, keepdims=True),
        dim="Variables",
        col_label="emissions-total",
        unit="Mt",
    )
    dm_emissions_bygastech.drop("Variables", ["emissions", "emissions-process"])
    dm_emissions_bygastech.rename_col("emissions-total", "emissions", "Variables")
    dm_emissions_bygastech.switch_categories_order("Categories1", "Categories2")

    # put in dict
    DM_emissions = {
        "combustion": dm_emissions_combustion,
        "process": dm_emissions_process,
        "bygastech": dm_emissions_bygastech,
        "combustion_bio": dm_emissions_combustion_bio,
        "bygastech_beforecc": dm_emissions_bygastech,
    }

    # return
    return DM_emissions


def carbon_capture(
    dm_ots_fts_cc,
    dm_emissions_bygastech,
    dm_emissions_combustion_bio,
    DM_emissions,
    dict_groupby,
):

    # get carbon capture
    dm_temp = dm_ots_fts_cc.copy()

    # subtract carbon captured to total CO2 emissions per technology
    idx = dm_emissions_bygastech.idx
    arr_temp = dm_emissions_bygastech.array[:, :, :, idx["CO2"], :] * (
        1 - dm_temp.array
    )
    dm_emissions_bygastech.add(
        arr_temp[:, :, :, np.newaxis, :], dim="Categories1", col_label="after-cc"
    )

    # get emissions captured with carbon capture
    idx = dm_emissions_bygastech.idx
    arr_temp = (
        dm_emissions_bygastech.array[:, :, :, idx["CO2"], :]
        - dm_emissions_bygastech.array[:, :, :, idx["after-cc"], :]
    )
    dm_emissions_bygastech.add(
        arr_temp[:, :, :, np.newaxis, :], dim="Categories1", col_label="CO2-capt-w-cc"
    )
    dm_emissions_capt_w_cc_bytech = dm_emissions_bygastech.filter(
        {"Categories1": ["CO2-capt-w-cc"]}
    )
    dm_emissions_capt_w_cc_bytech = dm_emissions_capt_w_cc_bytech.flatten()
    dm_emissions_capt_w_cc_bytech.rename_col_regex(
        "CO2-capt-w-cc_", "", dim="Categories1"
    )
    dm_emissions_capt_w_cc_bytech.rename_col("emissions", "CO2-capt-w-cc", "Variables")
    dm_emissions_bygastech.drop("Categories1", "CO2")
    dm_emissions_bygastech.rename_col(
        col_in="after-cc", col_out="CO2", dim="Categories1"
    )
    dm_emissions_bygastech.sort("Categories1")

    # get captured biogenic emissions
    dm_emissions_combustion_bio_capt_w_cc = dm_emissions_combustion_bio.copy()
    idx = dm_emissions_combustion_bio_capt_w_cc.idx
    arr_temp = dm_emissions_combustion_bio_capt_w_cc.array[:, :, :, idx["CO2"], :] * (
        1 - dm_temp.array
    )
    dm_emissions_combustion_bio_capt_w_cc.add(
        arr_temp[:, :, :, np.newaxis, :], dim="Categories1", col_label="after-cc"
    )
    idx = dm_emissions_combustion_bio_capt_w_cc.idx
    arr_temp = (
        dm_emissions_combustion_bio_capt_w_cc.array[:, :, :, idx["CO2"], :]
        - dm_emissions_combustion_bio_capt_w_cc.array[:, :, :, idx["after-cc"], :]
    )
    dm_emissions_combustion_bio_capt_w_cc.add(
        arr_temp[:, :, :, np.newaxis, :], dim="Categories1", col_label="capt-w-cc"
    )
    dm_emissions_combustion_bio_capt_w_cc = (
        dm_emissions_combustion_bio_capt_w_cc.filter({"Categories1": ["capt-w-cc"]})
    )
    dm_emissions_combustion_bio_capt_w_cc = (
        dm_emissions_combustion_bio_capt_w_cc.flatten().flatten()
    )
    dm_emissions_combustion_bio_capt_w_cc.deepen()
    dm_emissions_combustion_bio_capt_w_cc.rename_col(
        col_in="emissions-biogenic_capt-w-cc",
        col_out="emissions-biogenic_CO2-capt-w-cc",
        dim="Variables",
    )

    # get these captured biogenic emissions by material
    dm_emissions_combustion_bio_capt_w_cc = (
        dm_emissions_combustion_bio_capt_w_cc.groupby(
            dict_groupby,
            dim="Categories1",
            aggregation="sum",
            regex=True,
            inplace=False,
        )
    )

    # make negative captured biogenic emissions to supply to the climate module
    dm_emissions_combustion_bio_capt_w_cc_neg_bymat = (
        dm_emissions_combustion_bio_capt_w_cc.copy()
    )
    dm_emissions_combustion_bio_capt_w_cc_neg_bymat.array = (
        dm_emissions_combustion_bio_capt_w_cc_neg_bymat.array * -1
    )
    dm_emissions_combustion_bio_capt_w_cc_neg_bymat.rename_col(
        "emissions-biogenic_CO2-capt-w-cc",
        "emissions-biogenic_CO2-capt-w-cc-negative",
        "Variables",
    )

    # store
    DM_emissions["combustion_bio_capt_w_cc_neg_bymat"] = (
        dm_emissions_combustion_bio_capt_w_cc_neg_bymat
    )
    DM_emissions["capt_w_cc_bytech"] = dm_emissions_capt_w_cc_bytech

    # store also bygas (which is used in calibration if it's done)
    DM_emissions["bygas"] = dm_emissions_bygastech.group_all(
        "Categories2", inplace=False
    )

    # return
    return


def calibration_emissions(
    DM_cal, dm_emissions_bygas, dm_emissions_bygastech, DM_emissions, years_setting
):

    # get calibration rates
    DM_emissions["calib_rates_bygas"] = calibration_rates(
        dm=dm_emissions_bygas,
        dm_cal=DM_cal["emissions"],
        calibration_start_year=1990,
        calibration_end_year=2023,
        years_setting=years_setting,
    )

    # do calibration
    dm_emissions_bygas.array = (
        dm_emissions_bygas.array * DM_emissions["calib_rates_bygas"].array
    )

    # do calibration for each technology (by applying aggregate calibration rates)
    dm_emissions_bygastech.array = (
        dm_emissions_bygastech.array
        * DM_emissions["calib_rates_bygas"].array[:, :, :, :, np.newaxis]
    )

    # return
    return


# TODO: bring the function material_flows() into minerals (or somewhere external, probably together with the
# cost curves ... i.e. it could be in the tpe) and finalize it
def material_flows(
    dm_transport_stock,
    dm_material_towaste,
    dm_material_recovered,
    cdm_matdec_veh,
    dm_emissions_bygasmat,
):

    # do material decomposition of stock
    arr_temp = (
        dm_transport_stock.array[..., np.newaxis]
        * cdm_matdec_veh.array[np.newaxis, np.newaxis, :, :, :]
    )
    dm_transport_stock_bymat = DataMatrix.based_on(
        arr_temp,
        dm_transport_stock,
        {"Categories2": cdm_matdec_veh.col_labels["Categories2"]},
        units=dm_transport_stock.units,
    )
    dm_transport_stock_bymat.units["tra_product-stock"] = "t"

    # sum across sectors
    dm_stock_bymat = dm_transport_stock_bymat.group_all("Categories1", inplace=False)
    dm_stock_bymat.rename_col("tra_product-stock", "product-stock", "Variables")

    return


def compute_costs(
    dm_fxa_cost_matprod,
    dm_fxa_cost_cc,
    dm_material_production_bytech,
    dm_emissions_capt_w_cc_bytech,
):

    ###############################
    ##### MATERIAL PRODUCTION #####
    ###############################

    # subset costs
    dm_cost_sub = dm_fxa_cost_matprod.copy()

    # get material production by technology
    keep = dm_fxa_cost_matprod.col_labels["Categories1"]
    dm_material_techshare_sub = dm_material_production_bytech.filter(
        {"Categories1": keep}
    )
    dm_cost_sub.change_unit(
        "capex-baseyear", factor=1e3, old_unit="EUR/t", new_unit="EUR/kt"
    )
    dm_cost_sub.change_unit(
        "capex-d-factor", factor=1e3, old_unit="num", new_unit="num"
    )

    # get costs
    dm_material_techshare_sub_capex = cost(
        dm_activity=dm_material_techshare_sub, dm_cost=dm_cost_sub, cost_type="capex"
    )

    ######################################
    ##### EMISSIONS CAPTURED WITH CC #####
    ######################################

    # subset cdm
    dm_cost_sub = dm_fxa_cost_cc.copy()

    # get emissions captured with carbon capture
    keep = dm_fxa_cost_cc.col_labels["Categories1"]
    dm_emissions_capt_w_cc_sub = dm_emissions_capt_w_cc_bytech.filter(
        {"Categories1": keep}
    )
    dm_emissions_capt_w_cc_sub.change_unit(
        "CO2-capt-w-cc", factor=1e6, old_unit="Mt", new_unit="t"
    )

    # get costs
    dm_emissions_capt_w_cc_sub_capex = cost(
        dm_activity=dm_emissions_capt_w_cc_sub, dm_cost=dm_cost_sub, cost_type="capex"
    )

    ########################
    ##### PUT TOGETHER #####
    ########################

    DM_cost = {
        "material-production_capex": dm_material_techshare_sub_capex,
        "CO2-capt-w-cc_capex": dm_emissions_capt_w_cc_sub_capex,
    }

    # fix names
    for key in DM_cost.keys():
        cost_type = re.split("_", key)[1]
        activity_type = re.split("_", key)[0]
        DM_cost[key].filter({"Variables": ["unit-cost", cost_type]}, inplace=True)
        DM_cost[key].rename_col(
            "unit-cost", activity_type + "_" + cost_type + "-unit", "Variables"
        )
        DM_cost[key].rename_col(cost_type, activity_type + "_" + cost_type, "Variables")

    # make datamatrixes by material
    keys = list(DM_cost)
    for key in keys:
        materials = [i.split("-")[0] for i in DM_cost[key].col_labels["Categories1"]]
        materials = list(dict.fromkeys(materials))
        dict_groupby = {}
        for m in materials:
            dict_groupby[m] = ".*" + m + ".*"
        DM_cost[key + "_bymat"] = DM_cost[key].groupby(
            dict_groupby,
            dim="Categories1",
            aggregation="sum",
            regex=True,
            inplace=False,
        )

    # return
    return DM_cost
