# Import the Packages

import json
import os  # operating system (e.g., look for workspace)
import pickle  # read/write the data in pickle

from transition_compass_model.model.common.auxiliary_functions import (
    filter_country_and_load_data_from_pickles,
    read_level_data,
)

# Import Class
from transition_compass_model.model.common.interface_class import Interface


def init_years_lever():
    # function that can be used when running the module as standalone to initialise years and levers
    years_setting = [1990, 2023, 2050, 5]
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(current_file_directory, "../config/lever_position.json"))
    lever_setting = json.load(f)[0]
    return years_setting, lever_setting


def read_data(DM_climate, lever_setting):
    # get lever
    DM_ots_fts = read_level_data(DM_climate, lever_setting)

    return DM_ots_fts


def climate_buildings_interface(DM_ots_fts, write_pickle=False):
    # append
    dm_bld = DM_ots_fts["temp"]

    # if write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../_database/data/interface/climate_to_buildings.pickle",
        )
        DM_bld = {}
        DM_bld["cdd-hdd"] = dm_bld.copy()
        with open(f, "wb") as handle:
            pickle.dump(DM_bld, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # return
    return dm_bld


def variables_to_tpe(DM_ots_fts):
    dm_tpe = DM_ots_fts["temp"]["clm_capacity-factor"].flatten()
    dm_tpe.append(DM_ots_fts["temp"]["clm_temp_global"], "Variables")
    dm_tpe.flattest()

    return dm_tpe


# CORE module
def climate(
    lever_setting, years_setting, DM_input, interface=Interface(), calibration=False
):
    # climate data file
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    DM_ots_fts = read_data(DM_input, lever_setting)

    # tpe
    results_run = DM_ots_fts["temp"]

    # interface buildings
    dm_bld = climate_buildings_interface(DM_ots_fts, write_pickle=False)
    interface.add_link(from_sector="climate", to_sector="buildings", dm=dm_bld)

    # TODO: interface water when water is ready

    return results_run


# Local run of lifestyles
def local_climate_run():
    # Initiate the year & lever setting
    years_setting, lever_setting = init_years_lever()

    # get geoscale
    country_list = ["EU27", "Switzerland", "Vaud"]
    DM_input = filter_country_and_load_data_from_pickles(
        country_list=country_list, modules_list="climate"
    )

    # run
    results_run = climate(lever_setting, years_setting, DM_input["climate"])

    return results_run


# # local
if __name__ == "__main__":
    results_run = local_climate_run()
