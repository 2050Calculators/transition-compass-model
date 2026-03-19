import os
import pickle


def get_interface(
    current_file_directory, interface, from_sector, to_sector, country_list
):

    if interface.has_link(from_sector=from_sector, to_sector=to_sector):
        DM = interface.get_link(from_sector=from_sector, to_sector=to_sector)
    else:
        if len(interface.list_link()) != 0:
            print("You are missing " + from_sector + " to " + to_sector + " interface")
        filepath = os.path.join(
            current_file_directory,
            "../_database/data/interface/"
            + from_sector
            + "_to_"
            + to_sector
            + ".pickle",
        )
        with open(filepath, "rb") as handle:
            DM = pickle.load(handle)
        if type(DM) is dict:
            for key in DM.keys():
                DM[key].filter({"Country": country_list}, inplace=True)
        else:
            DM.filter({"Country": country_list}, inplace=True)
    return DM


def variables_for_tpe(dm_emi, dm_emi_co2eq):

    # overall emissions co2eq
    dm_out = dm_emi_co2eq.group_all("Categories1", inplace=False)
    dm_out.rename_col("emissions", "emissions_total", "Variables")

    # emissions co2eq by sector
    dm_out.append(dm_emi_co2eq.flatten(), "Variables")

    # emissions by gas
    dm_out.append(dm_emi.group_all("Categories1", inplace=False).flatten(), "Variables")

    return dm_out
