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
                if "Country" in list(DM[key].col_labels.keys()):
                    DM[key].filter({"Country": country_list}, inplace=True)
        else:
            if "Country" in list(DM[key].col_labels.keys()):
                DM.filter({"Country": country_list}, inplace=True)
    return DM
