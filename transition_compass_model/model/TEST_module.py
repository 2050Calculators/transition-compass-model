import os
import pickle

import transition_compass_model.model.TEST.interfaces as inter
from transition_compass_model.model.common.auxiliary_functions import (
    filter_country_and_load_data_from_pickles,
    init_years_lever,
    my_pickle_dump,
    read_level_data,
)
from transition_compass_model.model.common.interface_class import Interface


def read_data(DM_lca, lever_setting):
    # # get fxa
    # DM_fxa = DM_industry['fxa']

    # Get ots fts based on lever_setting
    DM_ots_fts = read_level_data(DM_lca, lever_setting)

    # # get calibration
    # dm_cal = DM_industry['calibration']

    # # get constants
    # CMD_const = DM_industry['constant']

    # return
    return DM_ots_fts


def test(
    lever_setting, years_setting, DM_input, interface=Interface(), calibration=True
):
    # industry data file
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    DM_ots_fts = read_data(DM_input, lever_setting)

    # ------------------------------------------------------------------
    # Exercise Anna (open TEST.pickle)
    # ------------------------------------------------------------------
    # Load/open TEST.pickle
    data_file = os.path.join(
        current_file_directory,
        "../_database/data/datamatrix/TEST.pickle",
    )

    with open(data_file, "rb") as handle:
        DM_TEST = pickle.load(handle)

    # test
    print(DM_TEST)
    print(DM_TEST.keys())
    print("TEST.pickle loaded!")
    # ------------------------------------------------------------------

    # get interfaces
    cntr_list = ["Switzerland", "Vaud"]
    # DM_transport = inter.get_interface(
    #     current_file_directory, interface, "transport", "lca", cntr_list
    # )

    # ----------------------------------------------------------------------------
    # Letting TEST.pickle and Building.pickle interact
    # ----------------------------------------------------------------------------

    # have to load the buildings.pickle
    data_file = os.path.join(
        current_file_directory,
        "../_database/data/datamatrix/buildings.pickle",
    )

    with open(data_file, "rb") as handle:
        DM_buildings = pickle.load(handle)

    # test
    print(DM_buildings)
    print(DM_buildings.keys())
    print("buildings.pickle loaded!")

    # only want floor intenstiy from buildings
    dm_floor_intensity_ots = DM_buildings["ots"]["floor-intensity"]
    dm_floor_intensity_fts = DM_buildings["fts"]["floor-intensity"]

    # interaction (adding buildings and TEST into one dictionary)
    DM_input_test = {
        "buildings": {
            "ots": DM_buildings["ots"]["floor-intensity"],
            "fts": DM_buildings["fts"]["floor-intensity"],
        },
        "TEST": {"ots": DM_TEST["ots"], "fts": DM_TEST["fts"]},
    }
    # test
    # print(type(DM_input_test["buildings"]["ots"]))
    # print(type(DM_input_test["buildings"]["fts"]))
    # print(DM_input_test["buildings"]["fts"].keys())

    # --------------------------------------------------------------------
    # ADDING THE MATRIXES TOGETHER
    # --------------------------------------------------------------------
    # .append() --> Attaches data from one DataMatrix to another DataMatrix
    # dm.append
    # dm.rename_col_regex()
    # printing indexes----------------------------------------------------
    # print indexes of TEST, so i know what to rename the column
    DM_input_test["TEST"]["ots"]["floor-intensity"].col_labels["Variables"]
    print(DM_input_test["TEST"]["ots"]["floor-intensity"].col_labels["Variables"])
    # print(DM_input_test["TEST"]["ots"]["floor-intensity"].idx)

    # renaming the TEST variables so that we can distinguish
    # it from the original Buildings variable
    DM_input_test["TEST"]["ots"]["floor-intensity"].rename_col_regex(
        "bld_floor-intensity_nonres-cap", "TEST_floor-intensity_nonres-cap", "Variables"
    )

    DM_input_test["TEST"]["ots"]["floor-intensity"].rename_col_regex(
        "lfs_floor-intensity_space-cap",  # old name
        "TEST_floor-intensity_space-cap",  # new name
        "Variables",
    )

    DM_input_test["TEST"]["ots"]["floor-intensity"].rename_col_regex(
        "lfs_household-size", "TEST_household-size", "Variables"
    )

    print(DM_input_test["TEST"]["ots"]["floor-intensity"].col_labels["Variables"])

    # -----------------------------------------------------------------
    # append TEST ots to buildings ots

    DM_input_test["buildings"]["ots"].append(
        DM_input_test["TEST"]["ots"]["floor-intensity"], dim="Variables"
    )

    # Check whether the append worked
    print(DM_input_test["buildings"]["ots"].col_labels["Variables"])
    # print(DM_input_test["buildings"]["ots"].dim_labels)

    # ----------------------------------------------------------------
    # now actually adding the columns together

    dm_buildings = DM_input_test["buildings"]["ots"].filter(
        {"Variables": ["bld_floor-intensity_nonres-cap"]}
    )

    dm_test = DM_input_test["buildings"]["ots"].filter(
        {"Variables": ["TEST_floor-intensity_nonres-cap"]}
    )
    # combined DataMatrix --> need to make a copy of dm_buildings (there i added the TEST Data)
    dm_buildings_test_combined = dm_buildings.copy()
    index = dm_buildings.idx

    # adding numerical values
    # only for Vaud, the ":,:" says from [2,34,1] --> [34,1]
    dm_buildings_test_combined.array[index["Vaud"], :, :] = (
        dm_buildings.array[index["Vaud"], :, :] + dm_test.array[index["Vaud"], :, :]
    )

    # renaming the resulting variable
    dm_buildings_test_combined.rename_col_regex(
        "bld_floor-intensity_nonres-cap",
        "SUM_bld_floor-intensity_nonres-cap",
        "Variables",
    )

    def cretqte_fts(lever):
        # print indexes of TEST, so i know what to rename the column
        DM_input_test["TEST"]["fts"]["floor-intensity"][lever].col_labels["Variables"]
        print(
            DM_input_test["TEST"]["fts"]["floor-intensity"][lever].col_labels[
                "Variables"
            ]
        )
        # print(DM_input_test["TEST"]["fts"]["floor-intensity"][lever].idx)

        # renaming the TEST variables so that we can distinguish
        # it from the original Buildings variable
        DM_input_test["TEST"]["fts"]["floor-intensity"][lever].rename_col_regex(
            "bld_floor-intensity_nonres-cap",
            "TEST_floor-intensity_nonres-cap",
            "Variables",
        )

        DM_input_test["TEST"]["fts"]["floor-intensity"][lever].rename_col_regex(
            "lfs_floor-intensity_space-cap",  # old name
            "TEST_floor-intensity_space-cap",  # new name
            "Variables",
        )

        DM_input_test["TEST"]["fts"]["floor-intensity"][lever].rename_col_regex(
            "lfs_household-size", "TEST_household-size", "Variables"
        )

        print(
            DM_input_test["TEST"]["fts"]["floor-intensity"][lever].col_labels[
                "Variables"
            ]
        )

        # -----------------------------------------------------------------
        # append TEST ots to buildings ots

        DM_input_test["buildings"]["fts"][lever].append(
            DM_input_test["TEST"]["fts"]["floor-intensity"][lever], dim="Variables"
        )

        # Check whether the append worked
        print(DM_input_test["buildings"]["fts"][lever].col_labels["Variables"])
        # print(DM_input_test["buildings"]["fts"][lever].dim_labels)

        # ----------------------------------------------------------------
        # now actually adding the columns together

        dm_buildings = DM_input_test["buildings"]["fts"][lever].filter(
            {"Variables": ["bld_floor-intensity_nonres-cap"]}
        )

        dm_test = DM_input_test["buildings"]["fts"][lever].filter(
            {"Variables": ["TEST_floor-intensity_nonres-cap"]}
        )

        # combined DataMatrix --> need to make a copy of dm_buildings (there i added the TEST Data)
        dm_buildings_test_combined = dm_buildings.copy()
        index = dm_buildings.idx

        # adding numerical values
        # only for Vaud, the ":,:" says from [2,34,1] --> [34,1]
        dm_buildings_test_combined.array[index["Vaud"], :, :] = (
            dm_buildings.array[index["Vaud"], :, :] + dm_test.array[index["Vaud"], :, :]
        ) * i

        # renaming the resulting variable
        dm_buildings_test_combined.rename_col_regex(
            "bld_floor-intensity_nonres-cap",
            "SUM_bld_floor-intensity_nonres-cap",
            "Variables",
        )
        return dm_buildings_test_combined

    # had to create a dictionary in order for the pickle_dump to work (was a dm_combined is a datamatrix but for the pickle i need a dictionary)
    DM_test_to_energy = {
        "ots": {"floor-intensity": dm_buildings_test_combined},
        "fts": {"floor-intensity": {}},
    }

    for i in range(1, 5):
        DM_test_to_energy["fts"]["floor-intensity"][i] = cretqte_fts(i)

    print(dm_buildings_test_combined.col_labels["Variables"])
    # print(dm_buildings.array)
    print(dm_buildings_test_combined.array)

    # ------------------------------------------------------------------
    # Exercise Anna
    # ------------------------------------------------------------------
    # create a new interface between buildings and TEST:
    if interface.has_link(from_sector="buildings", to_sector="TEST"):
        dm_test = interface.get_link(from_sector="buildings", to_sector="TEST")
        # get_link() retrieves data that building sector has made available to the TEST sector

    else:
        if len(interface.list_link()) != 0:
            print("You are missing buildings to TEST interface")
        data_file = os.path.join(
            current_file_directory,
            "../_database/data/interface/buildings_to_TEST.pickle",
        )
        with open(data_file, "rb") as handle:
            DM_test = pickle.load(handle)
        dm_test = DM_test["floor-area"]

        # dm_test.filter({"Country": cntr_list}, inplace=True)
    # TESTING
    # print("Buildings → TEST interface exists!")
    # print(type(dm_test))
    # print(dm_test.keys())

    # ------------------------------------------------------------------

    # DM_buildings = inter.get_interface(
    #     current_file_directory, interface, "buildings", "lca", cntr_list
    # )
    DM_industry = inter.get_interface(
        current_file_directory, interface, "industry", "lca", cntr_list
    )

    # split footrpint by product group
    # DM_footprint = wkf.get_footprint_by_group(DM_ots_fts["footprint"])

    # dm_bld_new = DM_buildings["floor-area"].filter(
    #     {"Variables": ["bld_floor-area_new"]}
    # )

    # # get footprint
    # DM_footprint_agg = {}
    # for key in DM_footprint.keys():
    #     DM_footprint_agg[key] = wkf.get_footprint(key, DM_footprint[key])
    # DM_footprint_agg["energy-demand"] = DM_footprint_agg["energy-demand-elec"].copy()
    # DM_footprint_agg["energy-demand"].append(
    #     DM_footprint_agg["energy-demand-ff"], "Variables"
    # )
    # del DM_footprint_agg["energy-demand-elec"]
    # del DM_footprint_agg["energy-demand-ff"]

    # # pass to TPE
    # results_run = wkf.variables_for_tpe(
    #     DM_footprint_agg, DM_buildings, DM_industry
    # )
    # --------------------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Exercise Anna
    # ------------------------------------------------------------------
    # creating the interface TEST_to_energy
    # The data received from buildings

    # dm_combined is the DataMatrix that we want to send to Energy.
    # Use the combined DataMatrix as the output of TEST
    dm_combined = dm_buildings_test_combined
    # Check that it really is a DataMatrix
    print("TEST → ENERGY")
    # print("Type:", type(dm_combined))
    # print("Variables:", dm_combined.col_labels["Variables"])
    print("Shape:", dm_combined.array.shape)

    write_pickle = True
    if write_pickle:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../_database/data/interface/TEST_to_energy.pickle",
        )

        # had to create a dictionary in order for the pickle_dump to work (was a dm_combined is a datamatrix but for the pickle i need a dictionary)
        # DM_test_to_energy = {
        # "floor-intensity": dm_combined
        #  }

        my_pickle_dump(DM_test_to_energy, f)

        # my_pickle_dump(dm_combined, f)

    KPI = [{"title": "A-C class", "value": 5, "unit": "%"}]
    dm_combined.append(DM_test_to_energy["fts"]["floor-intensity"][1], dim="Years")
    results_run = dm_combined.flattest()

    interface.add_link(from_sector="TEST", to_sector="energy", dm=dm_combined)

    return results_run, KPI


def local_lca_run():
    # Configures initial input for model run
    years_setting, lever_setting = init_years_lever()
    country_list = ["Vaud"]

    sectors = ["TEST"]

    # Filter geoscale
    # from database/data/datamatrix/.* reads the pickles, filters the geoscale, and loads them
    DM_input = filter_country_and_load_data_from_pickles(
        country_list=country_list, modules_list=sectors
    )

    # run
    results_run = test(lever_setting, years_setting, DM_input["TEST"])

    # return
    return results_run


# run local
if __name__ == "__main__":
    results_run = local_lca_run()
