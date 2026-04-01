from transition_compass_model.model.common.interface_class import Interface
from transition_compass_model.model.common.auxiliary_functions import (
    read_level_data,
    filter_country_and_load_data_from_pickles,
)

import transition_compass_model.model.industry.interfaces as inter
import transition_compass_model.model.industry.workflows as wkf

import os
import json


def read_data(DM_industry, lever_setting):

    # get fxa
    DM_fxa = DM_industry["fxa"]

    # Get ots fts based on lever_setting
    DM_ots_fts = read_level_data(DM_industry, lever_setting)

    # get calibration
    dm_cal = DM_industry["calibration"]

    # get constants
    CMD_const = DM_industry["constant"]

    # return
    return DM_fxa, DM_ots_fts, dm_cal, CMD_const


def industry(
    lever_setting, years_setting, DM_input, interface=Interface(), calibration=False
):

    # TODO: understand what to do with calibration and set calibration True

    # industry data file
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    DM_fxa, DM_ots_fts, DM_cal, CDM_const = read_data(DM_input, lever_setting)

    # get interfaces
    cntr_list = DM_ots_fts["product-net-import"].col_labels["Country"]
    DM_transport = inter.get_interface(
        current_file_directory, interface, "transport", "industry", cntr_list
    )
    DM_lfs = inter.get_interface(
        current_file_directory, interface, "lifestyles", "industry", cntr_list
    )
    DM_buildings = inter.get_interface(
        current_file_directory, interface, "buildings", "industry", cntr_list
    )

    # get product import
    dm_import = DM_ots_fts["product-net-import"]

    # get product production
    DM_production = wkf.product_production(
        DM_buildings["floor-area"].filter({"Variables": ["bld_floor-area_new"]}),
        DM_buildings["domapp"].filter({"Variables": ["bld_domapp_new"]}),
        DM_buildings["electronics"].filter({"Variables": ["bld_electronics_new"]}),
        DM_transport["tra-infra"].filter({"Variables": ["tra_product-demand"]}),
        DM_transport["tra-veh"],
        DM_lfs["pop"],
        DM_ots_fts["paperpack"],
        dm_import,
    )

    # get material demand
    DM_material_demand = wkf.apply_material_decomposition(
        DM_production["bld-floor"],
        DM_production["bld-domapp"],
        DM_production["bld-electronics"],
        DM_production["tra-infra"],
        DM_production["tra-veh"],
        DM_production["pack"].copy(),
        CDM_const["material-decomposition_floor"],
        CDM_const["material-decomposition_domapp"],
        CDM_const["material-decomposition_electronics"],
        CDM_const["material-decomposition_infra"],
        CDM_const["material-decomposition_veh"],
        CDM_const["material-decomposition_pack"],
        CDM_const["material-decomposition_bat"],
    )

    # do material switch (writes in DM_material_demand and DM_input_matswitchimpact)
    DM_input_matswitchimpact = (
        {}
    )  # create dict to save material switch that will be used later for environmental impact
    wkf.apply_material_switch(
        DM_material_demand["material-demand"],
        DM_ots_fts["material-switch"],
        CDM_const["material-switch"],
        DM_input_matswitchimpact,
    )

    # get material production
    DM_material_production = wkf.material_production(
        DM_ots_fts["material-efficiency"],
        DM_ots_fts["material-net-import"],
        DM_material_demand["material-demand"],
        DM_fxa["prod"],
    )

    # calibrate material production (writes in DM_material_production)
    if calibration is True:
        wkf.calibration_material_production(
            DM_cal,
            DM_material_production["bymat"],
            DM_material_production,
            years_setting,
        )

    # make packaging waste (assuming 100% of packaged used in one year, i.e. pack demand, goes to waste)
    dm_pack_waste = wkf.make_pack_waste(DM_ots_fts["paperpack"], DM_lfs["pop"])

    # get end of life
    DM_eol = wkf.end_of_life(
        DM_transport["tra-waste"],
        DM_transport["tra-infra"].filter({"Variables": ["tra_product-waste"]}),
        DM_buildings["floor-area"].filter({"Variables": ["bld_floor-area_waste"]}),
        dm_pack_waste,
        DM_buildings["domapp"].filter({"Variables": ["bld_domapp_waste"]}),
        DM_buildings["electronics"].filter({"Variables": ["bld_electronics_waste"]}),
        DM_ots_fts["eol-waste-management"],
        DM_ots_fts["eol-material-recovery"],
        CDM_const["material-decomposition_floor"],
        CDM_const["material-decomposition_infra"],
        CDM_const["material-decomposition_veh"],
        CDM_const["material-decomposition_pack"],
        CDM_const["material-decomposition_bat"],
        CDM_const["material-decomposition_domapp"],
        CDM_const["material-decomposition_electronics"],
        DM_material_production["bymat"],
    )

    # tomorrow: there are issues with recycled cement (cement cannot be recycled it seems)
    # and paper / pulp. Review these two inside end_of_life().

    # get material production by technology (writes in DM_material_production)
    DM_material_production["bytech"] = wkf.material_production_by_technology(
        DM_ots_fts["technology-share"],
        DM_material_production[
            "bymat"
        ].copy(),  # note that here is copy as I cannot overwrite the aggregation between timber and wwp, as I still need timber to be sent to landuse at the end
        DM_eol["material-recovered"],
    )

    # get energy demand for material production
    DM_energy_demand = wkf.energy_demand(DM_material_production["bytech"], CDM_const)

    # calibrate energy demand for material production (writes in DM_energy_demand)
    if calibration is True:
        wkf.calibration_energy_demand(
            DM_cal,
            DM_energy_demand["excl-feedstock_bycarr"],
            DM_energy_demand["excl-feedstock_bytechcarr"],
            DM_energy_demand,
            years_setting,
        )

    # compute energy demand for material production after taking into account technology development (writes in DM_energy_demand)
    wkf.technology_development(
        DM_ots_fts["technology-development"],
        DM_energy_demand["excl-feedstock_bytechcarr"],
    )

    # do energy switch (writes in DM_energy_demand["bytechcarr"])
    wkf.apply_energy_switch(
        DM_ots_fts["energy-carrier-mix"], DM_energy_demand["excl-feedstock_bytechcarr"]
    )

    # do dictionary to sum across technologies by materials
    materials = [
        i.split("-")[0]
        for i in DM_energy_demand["excl-feedstock_bytechcarr"].col_labels["Categories1"]
    ]
    materials = list(dict.fromkeys(materials))
    dict_groupby = {}
    for m in materials:
        dict_groupby[m] = ".*" + m + ".*"

    # compute specific energy demands (writes in DM_energy_demand)
    wkf.add_specific_energy_demands(
        DM_energy_demand["excl-feedstock_bytechcarr"],
        DM_energy_demand["feedstock_bytechcarr"],
        DM_energy_demand,
        dict_groupby,
    )

    # get emissions
    DM_emissions = wkf.emissions(
        CDM_const["emission-factor-process"],
        CDM_const["emission-factor"],
        DM_energy_demand["excl-feedstock_bytechcarr"],
        DM_material_production["bytech"],
    )

    # compute captured carbon (writes in DM_emissions)
    wkf.carbon_capture(
        DM_ots_fts["cc"],
        DM_emissions["bygastech"],
        DM_emissions["combustion_bio"],
        DM_emissions,
        dict_groupby,
    )

    # calibrate emissions (writes in DM_emissions)
    if calibration is True:
        wkf.calibration_emissions(
            DM_cal,
            DM_emissions["bygas"],
            DM_emissions["bygastech"],
            DM_emissions,
            years_setting,
        )

    # comute specific groups of emissions that will be used for tpe (writes in DM_emissions)
    # emissions with different techs
    DM_emissions["bygasmat"] = DM_emissions["bygastech"].groupby(
        dict_groupby, dim="Categories2", aggregation="sum", regex=True, inplace=False
    )

    # TODO: the code below on stock and flows will be moved to minerals or tpe
    # # do stock and flows of materials
    # material_flows(DM_transport["tra-stock-veh"].filter({"Categories1" : ['cars-EV', 'cars-FCV', 'cars-ICE', 'trucks-EV', 'trucks-FCV', 'trucks-ICE']}),
    #                DM_eol["material-towaste"], DM_eol["material-recovered"],
    #                CDM_const["material-decomposition_veh"].filter({"Categories1" : ['cars-EV', 'cars-FCV', 'cars-ICE', 'trucks-EV', 'trucks-FCV', 'trucks-ICE']}),
    #                DM_emissions["bygasmat"])

    # get costs (capex and opex) for material production and carbon catpure
    DM_cost = wkf.compute_costs(
        DM_fxa["cost-matprod"],
        DM_fxa["cost-CC"],
        DM_material_production["bytech"],
        DM_emissions["capt_w_cc_bytech"],
    )

    # get variables for tpe (also writes in DM_cost, dm_bld_matswitch_savings_bymat, DM_emissions and DM_material_production for renaming)
    results_run = inter.variables_for_tpe(
        DM_material_production["bymat"], DM_emissions["bygas"]
    )

    # interface agriculture
    DM_agr = inter.industry_agriculture_interface(
        DM_material_production, DM_energy_demand
    )
    interface.add_link(from_sector="industry", to_sector="agriculture", dm=DM_agr)

    # interface ammonia
    DM_amm = inter.industry_ammonia_interface(DM_material_production, DM_energy_demand)
    interface.add_link(from_sector="industry", to_sector="ammonia", dm=DM_amm)

    # interface landuse
    DM_lus = inter.industry_landuse_interface(DM_material_production, DM_energy_demand)
    interface.add_link(from_sector="industry", to_sector="land-use", dm=DM_lus)

    # interface energy
    DM_ene = inter.industry_energy_interface(
        DM_energy_demand["bycarr"],
        CDM_const["energy_excl-feedstock_eleclight-split"],
        CDM_const["energy_efficiency"],
    )
    interface.add_link(from_sector="industry", to_sector="energy", dm=DM_ene)

    # interface forestry
    dm_for = inter.industry_forestry_interface(
        DM_material_demand["material-demand"], DM_fxa["demand"]
    )
    interface.add_link(from_sector="industry", to_sector="forestry", dm=dm_for)

    # # interface lca
    # DM_lca = inter.industry_lca_interface(CDM_const["material-decomposition_veh"], DM_eol["veh_eol_to_recycling"])
    # interface.add_link(from_sector='industry', to_sector='lca', dm=DM_lca)

    # # interface refinery
    # dm_refinery = industry_refinery_interface(DM_energy_demand)
    # interface.add_link(from_sector='industry', to_sector='oil-refinery', dm=dm_refinery)

    # # interface district heating
    # dm_dh = industry_district_heating_interface(DM_energy_demand)
    # interface.add_link(from_sector='industry', to_sector='district-heating', dm=dm_dh)

    # interface emissions
    dm_ems = inter.industry_emissions_interface(DM_emissions)
    interface.add_link(from_sector="industry", to_sector="emissions", dm=dm_ems)

    # # interface water
    # dm_water = industry_water_inferface(DM_energy_demand, DM_material_production)
    # interface.add_link(from_sector='industry', to_sector='water', dm=dm_water)

    # # interface ccus
    # dm_ccus = industry_ccus_interface(DM_emissions)
    # interface.add_link(from_sector='industry', to_sector='ccus', dm=dm_ccus)

    # # interface gtap
    # dm_gtap = industry_gtap_interface(DM_energy_demand, DM_material_production)
    # interface.add_link(from_sector='industry', to_sector='gtap', dm=dm_gtap)

    # # interface minerals
    # DM_ind = industry_minerals_interface(DM_production, DM_eol["veh_eol_to_recycling"])
    # interface.add_link(from_sector='industry', to_sector='minerals', dm=DM_ind)

    # # interface employment
    # dm_emp = industry_employment_interface(DM_material_demand, DM_energy_demand, DM_material_production, DM_cost, DM_ots_fts)
    # interface.add_link(from_sector='industry', to_sector='employment', dm=dm_emp)

    # # interface air pollution
    # dm_airpoll = industry_airpollution_interface(DM_material_production, DM_energy_demand)
    # interface.add_link(from_sector='industry', to_sector='air-pollution', dm=dm_airpoll)

    # return
    return results_run


def local_industry_run():

    # Configures initial input for model run
    f = open("../config/lever_position.json")
    lever_setting = json.load(f)[0]
    years_setting = [1990, 2023, 2025, 2050, 5]

    country_list = ["Switzerland", "EU27", "Vaud"]

    sectors = ["industry"]
    # Filter geoscale
    # from database/data/datamatrix/.* reads the pickles, filters the geoscale, and loads them
    DM_input = filter_country_and_load_data_from_pickles(
        country_list=country_list, modules_list=sectors
    )

    # run
    results_run = industry(lever_setting, years_setting, DM_input["industry"])

    # return
    return results_run


# run local
if __name__ == "__main__":
    results_run = local_industry_run()
