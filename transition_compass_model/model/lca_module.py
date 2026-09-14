import os

import transition_compass_model.model.lca.interfaces as inter
import transition_compass_model.model.lca.workflows as wkf
from transition_compass_model.model.common.auxiliary_functions import (
    filter_country_and_load_data_from_pickles,
    init_years_lever,
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


def lca(
    lever_setting, years_setting, DM_input, interface=Interface(), calibration=True
):
    # industry data file
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    DM_ots_fts = read_data(DM_input, lever_setting)

    # get interfaces
    cntr_list = DM_ots_fts["footprint"]["materials"].col_labels["Country"]
    DM_transport = inter.get_interface(
        current_file_directory, interface, "transport", "lca", cntr_list
    )
    DM_buildings = inter.get_interface(
        current_file_directory, interface, "buildings", "lca", cntr_list
    )
    DM_industry = inter.get_interface(
        current_file_directory, interface, "industry", "lca", cntr_list
    )

    # split footrpint by product group
    DM_footprint = wkf.get_footprint_by_group(DM_ots_fts["footprint"])

    # split product demand
    dm_tra_infra_new = DM_transport["tra-infra"].filter(
        {"Variables": ["tra_product-demand"]}
    )
    dm_bld_new = DM_buildings["floor-area"].filter(
        {"Variables": ["bld_floor-area_new"]}
    )
    dm_domapp_new = DM_buildings["domapp"].filter(
        {
            "Variables": ["bld_domapp_new"],
            "Categories1": ["dishwasher", "dryer", "freezer", "fridge", "wmachine"],
        }
    )
    # dryer and freezer have no dedicated LCA coefficients; use wmachine and fridge as proxies.
    dm_domapp_new.groupby(
        {"wmachine": ["wmachine", "dryer"], "fridge": ["fridge", "freezer"]},
        "Categories1",
        inplace=True,
    )
    dm_elec_new = DM_buildings["electronics"].filter(
        {"Variables": ["bld_electronics_new"]}
    )
    DM_demand = {
        "vehicles": DM_transport["tra-veh"].filter(
            {"Categories1": ["HDV", "LDV", "bus", "planes", "ships", "trains"]}
        ),
        "tra-infra": dm_tra_infra_new,
        "domapp": dm_domapp_new,
        "electronics": dm_elec_new,
    }

    # get footprint
    DM_footprint_agg = {}
    for key in DM_footprint.keys():
        DM_footprint_agg[key] = wkf.get_footprint(key, DM_demand, DM_footprint[key])
    DM_footprint_agg["energy-demand"] = DM_footprint_agg["energy-demand-elec"].copy()
    DM_footprint_agg["energy-demand"].append(
        DM_footprint_agg["energy-demand-ff"], "Variables"
    )
    del DM_footprint_agg["energy-demand-elec"]
    del DM_footprint_agg["energy-demand-ff"]

    # pass to TPE
    results_run = wkf.variables_for_tpe(
        DM_footprint_agg, DM_transport, DM_buildings, DM_industry
    )

    # return
    # TODO : add other results run outputs to website.
    return results_run


def local_lca_run():
    # Configures initial input for model run
    years_setting, lever_setting = init_years_lever()
    country_list = ["Vaud"]

    sectors = ["lca"]

    # Filter geoscale
    # from database/data/datamatrix/.* reads the pickles, filters the geoscale, and loads them
    DM_input = filter_country_and_load_data_from_pickles(
        country_list=country_list, modules_list=sectors
    )

    # run
    results_run = lca(lever_setting, years_setting, DM_input["lca"])

    # return
    return results_run


# run local
if __name__ == "__main__":
    results_run = local_lca_run()
