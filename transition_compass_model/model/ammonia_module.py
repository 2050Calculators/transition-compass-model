import os
import json

from transition_compass_model.model.common.auxiliary_functions import (
    read_level_data,
    filter_country_and_load_data_from_pickles,
)
from transition_compass_model.model.common.interface_class import Interface

import transition_compass_model.model.ammonia.interfaces as inter
import transition_compass_model.model.ammonia.workflows as wkf


def read_data(DM_ammonia, lever_setting):

    # get fxa
    DM_fxa = DM_ammonia["fxa"]

    # Get ots fts based on lever_setting
    DM_ots_fts = read_level_data(DM_ammonia, lever_setting)

    # get calibration
    dm_cal = DM_ammonia["calibration"]

    # get constants
    CMD_const = DM_ammonia["constant"]

    # return
    return DM_fxa, DM_ots_fts, dm_cal, CMD_const


def ammonia(
    lever_setting, years_setting, DM_input, interface=Interface(), calibration=True
):

    # ammonia data file
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    DM_fxa, DM_ots_fts, DM_cal, CDM_const = read_data(DM_input, lever_setting)

    # get interfaces
    cntr_list = DM_ots_fts["product-net-import"].col_labels["Country"]
    dm_agriculture = inter.get_interface(
        current_file_directory, interface, "agriculture", "ammonia", cntr_list
    )
    DM_industry = inter.get_interface(
        current_file_directory, interface, "industry", "ammonia", cntr_list
    )

    # get product import
    dm_import = DM_ots_fts["product-net-import"]

    # get product production
    dm_production_fert = wkf.product_production(dm_agriculture, dm_import)

    # get material demand
    dm_material_demand = wkf.apply_material_decomposition(
        dm_production_fert, CDM_const["material-decomposition_fertilizer"]
    )

    # note: this is only the demand coming from material content in fertilizer, but demand for ammonia
    # is usually higher, so you should have a fixed assumption or constant to add the ammonia from other
    # sectors, so that then ammonia prodiction below can be computed correctly with the current
    # material net import share which is around zero.

    # get material production
    DM_material_production = wkf.material_production(
        DM_ots_fts["material-efficiency"],
        DM_ots_fts["material-net-import"],
        dm_material_demand,
    )

    # calibrate material production (writes in DM_material_production)
    # note: so far the module computes the production of ammonia that is used to make fertilizers
    # there are other uses of ammonia, which make the production usually much bigger
    # at the moment this adjustment is done via calibration
    if calibration is True:
        wkf.calibration_material_production(
            DM_cal,
            DM_material_production["bymat"],
            DM_material_production,
            years_setting,
        )

    # end of life
    # note: for the moment we do not do recycled ammonia

    # get material production by technology (writes in DM_material_production)
    dm_temp = DM_material_production["bymat"].copy()
    dm_temp.rename_col("ammonia", "ammonia-tech", "Categories1")
    DM_material_production["bytech"] = dm_temp.copy()

    # get energy demand for material production
    DM_energy_demand = wkf.energy_demand(DM_material_production["bytech"], CDM_const)

    # # calibrate energy demand for material production (writes in DM_energy_demand)
    # # note: difficult to find data on energy demand for ammonia manufacturing, to do later on in case
    # if calibration is True:
    #     wkf.calibration_energy_demand(DM_cal, DM_energy_demand["bycarr"], DM_energy_demand["bytechcarr"],
    #                               DM_energy_demand, years_setting)

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

    # compute specific energy demands that will be used for tpe (writes in DM_energy_demand)
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
    # note: FAO has data on emissions from syntetic fertilizer, but it's mixed with use (not only manufacturing)
    # for only production, possibly some data are here: https://www.nature.com/articles/s41598-022-18773-w#data-availability
    # for the moment data in pre processing for this is all missing (so I leave the calibration step, which will not calibrate),
    # in case for later to update data in calibration with literature
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

    # interface energy
    DM_ene = inter.ammonia_energy_interface(
        DM_energy_demand["bycarr"],
        CDM_const["energy_excl-feedstock_eleclight-split"],
        CDM_const["energy_efficiency"],
    )
    interface.add_link(from_sector="ammonia", to_sector="energy", dm=DM_ene)

    # interface emissions
    dm_ems = inter.ammonia_emissions_interface(DM_emissions)
    interface.add_link(from_sector="ammonia", to_sector="emissions", dm=dm_ems)

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

    # Configures initial input for model run
    f = open("../config/lever_position.json")
    lever_setting = json.load(f)[0]
    years_setting = [1990, 2023, 2025, 2050, 5]

    country_list = ["Switzerland", "EU27", "Vaud"]

    sectors = ["ammonia"]
    # Filter geoscale
    # from database/data/datamatrix/.* reads the pickles, filters the geoscale, and loads them
    DM_input = filter_country_and_load_data_from_pickles(
        country_list=country_list, modules_list=sectors
    )

    # run
    results_run = ammonia(lever_setting, years_setting, DM_input["ammonia"])

    # return
    return results_run


# run local
if __name__ == "__main__":
    results_run = local_ammonia_run()
