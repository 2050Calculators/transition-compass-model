from transition_compass_model.model.common.interface_class import Interface
from transition_compass_model.model.common.auxiliary_functions import (
    read_level_data,
    filter_country_and_load_data_from_pickles,
)

import transition_compass_model.model.emissions.interfaces as inter
import transition_compass_model.model.emissions.workflows as wkf

import os
import json


def emissions(years_setting, interface=Interface(), cntr_list=None, calibration=False):

    # get interfaces
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    dm_transport = inter.get_interface(
        current_file_directory, interface, "transport", "emissions", cntr_list
    )
    dm_buildings = inter.get_interface(
        current_file_directory, interface, "buildings", "emissions", cntr_list
    )
    dm_industry = inter.get_interface(
        current_file_directory, interface, "industry", "emissions", cntr_list
    )
    dm_agriculture = inter.get_interface(
        current_file_directory, interface, "agriculture", "emissions", cntr_list
    )
    dm_ammonia = inter.get_interface(
        current_file_directory, interface, "ammonia", "emissions", cntr_list
    )
    DM_emi = {
        "transport": dm_transport,
        "buildings": dm_buildings,
        "industry": dm_industry,
        "agriculture": dm_agriculture,
        "ammonia": dm_ammonia,
    }

    # put together
    dm_emi = wkf.put_together_emissions(DM_emi)

    # make co2 equivalent
    dm_emi_co2eq = wkf.make_co2_equivalent(dm_emi)
    # dm_emi_co2eq.datamatrix_plot(stacked=True)

    # send to TPE
    results_run = inter.variables_for_tpe(dm_emi, dm_emi_co2eq)

    return results_run


def local_industry_run():

    # Configures initial input for model run
    # f = open('../config/lever_position.json')
    # lever_setting = json.load(f)[0]
    years_setting = [1990, 2023, 2025, 2050, 5]

    country_list = ["Switzerland"]

    # sectors = ['emissions']
    # Filter geoscale
    # from database/data/datamatrix/.* reads the pickles, filters the geoscale, and loads them
    # DM_input = filter_country_and_load_data_from_pickles(country_list= country_list, modules_list = sectors)

    # run
    results_run = emissions(years_setting, cntr_list=country_list)

    # return
    return results_run


# run local
if __name__ == "__main__":
    results_run = local_industry_run()
