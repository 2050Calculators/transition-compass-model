
from model.common.interface_class import Interface

from model.common.auxiliary_functions import (
    read_level_data,
  filter_country_and_load_data_from_pickles,
)
import pickle
import json
import os

import model.transport.interfaces as inter
import model.transport.workflows as wkf


def read_data(DM_transport, lever_setting):

    dict_fxa = DM_transport["fxa"]
    dm_freight_tech = dict_fxa["freight_tech"]
    dm_passenger_tech = dict_fxa["passenger_tech"]
    dm_passenger_tech.append(dict_fxa["passenger_vehicle-lifetime"], dim="Variables")
    dm_freight_mode_other = dict_fxa["freight_mode_other"]
    dm_freight_mode_road = dict_fxa["freight_mode_road"]

    # Read fts based on lever_setting
    DM_ots_fts = read_level_data(DM_transport, lever_setting)

    # PASSENGER
    dm_passenger_aviation = DM_ots_fts["passenger_aviation-pkm"]
    dm_passenger_tech.append(
        DM_ots_fts["passenger_veh-efficiency_new"], dim="Variables"
    )
    dm_passenger_tech.append(
        DM_ots_fts["passenger_technology-share_new"], dim="Variables"
    )
    dm_passenger_modal = DM_ots_fts["passenger_modal-share"]
    dm_passenger = DM_ots_fts["passenger_occupancy"]
    dm_passenger.append(DM_ots_fts["passenger_utilization-rate"], dim="Variables")

    # FREIGHT
    dm_freight_tech.append(
        DM_ots_fts["freight_vehicle-efficiency_new"], dim="Variables"
    )
    dm_freight_tech.append(DM_ots_fts["freight_technology-share_new"], dim="Variables")
    dm_freight_mode_road.append(DM_ots_fts["freight_utilization-rate"], dim="Variables")
    dm_freight_modal_share = DM_ots_fts["freight_modal-share"]
    dm_freight_demand = DM_ots_fts["freight_tkm"]
    # OTHER
    dm_fuels = DM_ots_fts["fuel-mix"]

    DM_passenger = {
        "passenger_tech": dm_passenger_tech,
        "passenger_aviation": dm_passenger_aviation,
        "passenger_modal_split": dm_passenger_modal,
        "passenger_all": dm_passenger,
        "passenger_pkm_demand": DM_ots_fts["pkm"],
        "passenger_aviation-share-local": dict_fxa["share-local-emissions"],  # pkm/cap
    }

    DM_freight = {
        "freight_tech": dm_freight_tech,
        "freight_mode_other": dm_freight_mode_other,
        "freight_mode_road": dm_freight_mode_road,
        "freight_demand": dm_freight_demand,
        "freight_modal_split": dm_freight_modal_share,
    }

    DM_other = {
        "fuels": dm_fuels,
        "fuel-availability": dict_fxa["fuel-mix-availability"],
        "electricity-emissions": DM_transport["fxa"]["emission-factor-electricity"],
    }

    cdm_const = DM_transport["constant"]

    return DM_passenger, DM_freight, DM_other, cdm_const


def transport(lever_setting, years_setting, DM_input, interface=Interface()):

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    DM_passenger, DM_freight, DM_other, cdm_const = read_data(DM_input, lever_setting)

    cntr_list = DM_passenger["passenger_modal_split"].col_labels["Country"]

    # If the input from lifestyles are available in the interface, read them, else read from xls
    if interface.has_link(from_sector="lifestyles", to_sector="transport"):
        DM_lfs = interface.get_link(from_sector="lifestyles", to_sector="transport")
        dm_lfs = DM_lfs["pop"]
    else:
        if len(interface.list_link()) != 0:
            print(
                "You are missing " + "lifestyles" + " to " + "transport" + " interface"
            )
        lfs_interface_data_file = os.path.join(
            current_file_directory,
            "../_database/data/interface/lifestyles_to_transport.pickle",
        )
        with open(lfs_interface_data_file, "rb") as handle:
            DM_lfs = pickle.load(handle)
        dm_lfs = DM_lfs["pop"]
        dm_lfs.filter({"Country": cntr_list}, inplace=True)

    # PASSENGER
    cdm_const_passenger = cdm_const.copy()
    DM_passenger_out = wkf.passenger_fleet_energy(
        DM_passenger, dm_lfs, DM_other, cdm_const_passenger, years_setting
    )
    DM_passenger_out["aviation-share-local"] = DM_passenger[
        "passenger_aviation-share-local"
    ]
    # FREIGHT
    cdm_const_freight = cdm_const.copy()
    DM_freight_out = wkf.freight_fleet_energy(
        DM_freight, DM_other, cdm_const_freight, years_setting
    )

    DM_power = inter.tra_energy_interface(DM_passenger_out['power'], DM_freight_out['power'])
    interface.add_link(from_sector='transport', to_sector='energy', dm=DM_power)
    # df = dm_power.write_df()
    # df.to_excel('transport-to-power.xlsx', index=False)

    # Storage-module
    dm_oil_refinery = inter.tra_oilrefinery_interface(
        DM_passenger_out["oil-refinery"], DM_freight_out["oil-refinery"]
    )
    interface.add_link(
        from_sector="transport", to_sector="oil-refinery", dm=dm_oil_refinery
    )

    # Agriculture-module
    dm_agriculture = inter.tra_agriculture_interface(
        DM_freight_out["agriculture"], DM_passenger_out["agriculture"]
    )
    interface.add_link(
        from_sector="transport", to_sector="agriculture", dm=dm_agriculture
    )

    # Minerals and Industry
    dm_freight_veh = DM_freight_out["tech"].filter(
        {
            "Variables": [
                "tra_freight_new-vehicles",
                "tra_freight_vehicle-waste",
                "tra_freight_vehicle-fleet",
            ]
        }
    )
    dm_passenger_veh = DM_passenger_out["tech"].filter(
        {
            "Variables": [
                "tra_passenger_new-vehicles",
                "tra_passenger_vehicle-waste",
                "tra_passenger_vehicle-fleet",
            ]
        }
    )
    dm_infrastructure = wkf.dummy_tra_infrastructure_workflow(dm_lfs)
    DM_industry = inter.tra_industry_interface(dm_freight_veh.copy(), dm_passenger_veh.copy(), dm_infrastructure)
    # DM_minerals = tra_minerals_interface(dm_freight_veh, dm_passenger_veh, DM_industry, dm_infrastructure, write_xls=False)
    # !FIXME: add km infrastructure data, using compute_stock with tot_km and renovation rate as input.
    #  data for ch ok, data for eu, backcalculation? dummy based on swiss pop?
    interface.add_link(from_sector="transport", to_sector="industry", dm=DM_industry)
    # interface.add_link(from_sector='transport', to_sector='minerals', dm=DM_minerals)

    # Emissions
    dm_emissions = inter.tra_emissions_interface(DM_passenger_out["emissions"], DM_freight_out["emissions"])
    interface.add_link(from_sector="transport", to_sector="emissions", dm=dm_emissions.copy())

    # Local transport emissions
    N2O_to_CO2 = 265
    CH4_to_CO2 = 28

    dm_emissions = DM_passenger_out["emissions"]
    idx = dm_emissions.idx
    dm_emissions.array[:, :, :, :, idx["CH4"]] = (
        dm_emissions.array[:, :, :, :, idx["CH4"]] * CH4_to_CO2
    )
    dm_emissions.array[:, :, :, :, idx["N2O"]] = (
        dm_emissions.array[:, :, :, :, idx["N2O"]] * N2O_to_CO2
    )
    dm_emissions.rename_col(
        "tra_emissions_passenger", "tra_emissions-CO2e_passenger", dim="Variables"
    )
    dm_emissions.group_all("Categories2")

    results_run, KPI = inter.prepare_TPE_output(DM_passenger_out, DM_freight_out)
    return results_run, KPI


def local_transport_run():
    # Function to run only transport module without converter and tpe
    years_setting = [1990, 2023, 2025, 2050, 5]
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(current_file_directory, "../config/lever_position.json"))
    lever_setting = json.load(f)[0]

    # get geoscale
    country_list = ['EU27', 'Switzerland', 'Vaud']
    DM_input = filter_country_and_load_data_from_pickles(country_list= country_list, modules_list = 'transport')

    results_run = transport(lever_setting, years_setting, DM_input['transport'])

    return results_run


# database_from_csv_to_datamatrix()
# print('In transport, the share of waste by fuel/tech type does not seem right. Fix it.')
# print('Apply technology shares before computing the stock')
# print('For the efficiency, use the new methodology developped for Building (see overleaf on U-value)')
if __name__ == "__main__":
  results_run = local_transport_run()
