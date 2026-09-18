import json
import os
import pickle

import transition_compass_model.model.energy.workflows as wkf
from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    filter_DM,
)
from transition_compass_model.model.common.interface_class import Interface


def energy(lever_setting, years_setting, country_list, interface=Interface()):
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    years_fts = create_years_list(years_setting[2], years_setting[3], years_setting[4])
    years_ots = create_years_list(years_setting[0], years_setting[1], 1)
    # Read transport input
    if interface.has_link(from_sector="transport", to_sector="energy"):
        DM_transport = interface.get_link(from_sector="transport", to_sector="energy")
    else:
        if len(interface.list_link()) != 0:
            print("You are missing " + "transport" + " to " + "energy" + " interface")
        tra_interface_data_file = os.path.join(
            current_file_directory,
            "../_database/data/interface/transport_to_energy.pickle",
        )
        with open(tra_interface_data_file, "rb") as handle:
            DM_transport = pickle.load(handle)
        for key in DM_transport.keys():
            DM_transport[key].filter({"Country": country_list}, inplace=True)
    # !FIXME: I'm dropping aviation
    DM_transport["passenger"].drop("Categories1", "aviation")

    # Check country selection for energy module run
    if "EU27" in country_list:
        if country_list != ["EU27"]:
            raise RuntimeError(
                "If you want to solve the energy module for EU27, set geoscale=EU27"
            )
    else:
        list_wo_CH = set(country_list) - {"Switzerland"}
        if len(list_wo_CH) > 1:
            raise RuntimeError(
                "You are trying to solve the energy module for 2 cantons at the same time, "
                "pick only one canton and eventually Switzerland (see geoscale variable)"
            )

    if interface.has_link(from_sector="buildings", to_sector="energy"):
        DM_buildings = interface.get_link(from_sector="buildings", to_sector="energy")
    else:
        if len(interface.list_link()) != 0:
            print("You are missing " + "buildings" + " to " + "energy" + " interface")
        bld_file = os.path.join(
            current_file_directory,
            "../_database/data/interface/buildings_to_energy.pickle",
        )
        with open(bld_file, "rb") as handle:
            DM_buildings = pickle.load(handle)
        filter_DM(DM_buildings, {"Country": country_list})

    if interface.has_link(from_sector="industry", to_sector="energy"):
        DM_industry = interface.get_link(from_sector="industry", to_sector="energy")
    else:
        if len(interface.list_link()) != 0:
            print("You are missing " + "industry" + " to " + "energy" + " interface")
        bld_file = os.path.join(
            current_file_directory,
            "../_database/data/interface/industry_to_energy.pickle",
        )
        with open(bld_file, "rb") as handle:
            DM_industry = pickle.load(handle)
        filter_DM(DM_industry, {"Country": country_list})

    if interface.has_link(from_sector="agriculture", to_sector="energy"):
        DM_agriculture = interface.get_link(
            from_sector="agriculture", to_sector="energy"
        )
    else:
        if len(interface.list_link()) != 0:
            print("You are missing " + "agriculture" + " to " + "energy" + " interface")
        agr_file = os.path.join(
            current_file_directory,
            "../_database/data/interface/agriculture_to_energy.pickle",
        )
        with open(agr_file, "rb") as handle:
            DM_agriculture = pickle.load(handle)
        if ("Vaud" in country_list) and (
            "Vaud" not in DM_agriculture["power"].col_labels["Country"]
        ):
            DM_agriculture["power"].add(0, dim="Country", dummy=True, col_label="Vaud")
        filter_DM(DM_agriculture, {"Country": country_list})

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    data_filepath = os.path.join(
        current_file_directory, "../_database/data/datamatrix/energy.pickle"
    )
    results_run, dm_energy_emi = wkf.energyscope_pyomo(
        data_filepath,
        DM_transport,
        DM_buildings,
        DM_industry,
        DM_agriculture,
        years_ots,
        years_fts,
        country_list,
        lever_setting,
    )

    interface.add_link(from_sector="energy", to_sector="emissions", dm=dm_energy_emi)

    return results_run


def local_energy_run():
    # Function to run module as stand alone without other modules/converter or TPE
    years_setting = [1990, 2023, 2025, 2050, 5]
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(current_file_directory, "../config/lever_position.json"))
    lever_setting = json.load(f)[0]
    # Function to run only transport module without converter and tpe

    # get geoscale
    country_list = ["Vaud"]

    results_run = energy(lever_setting, years_setting, country_list)

    return results_run


# database_from_csv_to_datamatrix()
# print('In transport, the share of waste by fuel/tech type does not seem right. Fix it.')
# print('Apply technology shares before computing the stock')
# print('For the efficiency, use the new methodology developed for Building (see overleaf on U-value)')
if __name__ == "__main__":
    results_run = local_energy_run()
# local_energy_run()

#    DM_transport = pickle.load(handle)


#    DM_transport = pickle.load(handle)
# print('Hello')
