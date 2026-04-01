import pickle
import json
import os
import warnings

warnings.simplefilter("ignore")
import plotly.io as pio

pio.renderers.default = "browser"

from transition_compass_model.model.common.auxiliary_functions import (
    difference_with_data,
    difference_with_data_graph,
)


def check_transport_EU(
    current_file_directory,
    DM_passenger_out,
    DM_freight_out,
    year_start=2000,
    year_end=2021,
    years_calibration=[2019],
):

    DM = {}

    #################
    ##### FLEET #####
    #################

    # get data
    filepath = os.path.join(
        current_file_directory,
        "../_database/pre_processing/transport/EU/data/datamatrix/calibration_fleet.pickle",
    )
    with open(filepath, "rb") as handle:
        DM_fleet_check = pickle.load(handle)

    # passenger stock and new by engine jrc
    model_variables = ["tra_passenger_vehicle-fleet", "tra_passenger_new-vehicles"]
    dm_model = DM_passenger_out["tech"].filter({"Variables": model_variables})
    dm_model.rename_col(
        model_variables, ["stock-vehicles", "new-vehicles"], "Variables"
    )
    dm_data = DM_fleet_check["jrc_passenger"].filter({"Country": ["EU27"]})
    dm_data.sort("Variables")
    DM["stock-vehicles"] = {}
    DM["stock-vehicles"]["passenger"] = {}
    DM["stock-vehicles"]["passenger"]["byengine"] = {}
    DM["stock-vehicles"]["passenger"]["byengine"]["JRC_Transport"] = (
        difference_with_data(
            dm_model.filter({"Variables": ["stock-vehicles"]}),
            dm_data.filter({"Variables": ["stock-vehicles"]}),
            year_start,
            year_end,
            years_calibration,
        )
    )
    DM["new-vehicles"] = {}
    DM["new-vehicles"]["passenger"] = {}
    DM["new-vehicles"]["passenger"]["byengine"] = {}
    DM["new-vehicles"]["passenger"]["byengine"]["JRC_Transport"] = difference_with_data(
        dm_model.filter({"Variables": ["new-vehicles"]}),
        dm_data.filter({"Variables": ["new-vehicles"]}),
        year_start,
        year_end,
        years_calibration,
    )

    # freight stock and new by engine jrc
    model_variables = ["tra_freight_vehicle-fleet", "tra_freight_new-vehicles"]
    dm_model = DM_freight_out["tech"].filter({"Variables": model_variables})
    dm_model.rename_col(
        model_variables, ["stock-vehicles", "new-vehicles"], "Variables"
    )
    dm_model.groupby({"trucks": ["HDVH", "HDVL", "HDVM"]}, "Categories1", inplace=True)
    dm_data = DM_fleet_check["jrc_freight"].filter({"Country": ["EU27"]})
    DM["stock-vehicles"]["freight"] = {}
    DM["stock-vehicles"]["freight"]["byengine"] = {}
    DM["stock-vehicles"]["freight"]["byengine"]["JRC_Transport"] = difference_with_data(
        dm_model.filter({"Variables": ["stock-vehicles"]}),
        dm_data.filter({"Variables": ["stock-vehicles"]}),
        year_start,
        year_end,
        years_calibration,
    )
    DM["new-vehicles"]["freight"] = {}
    DM["new-vehicles"]["freight"]["byengine"] = {}
    DM["new-vehicles"]["freight"]["byengine"]["JRC_Transport"] = difference_with_data(
        dm_model.filter({"Variables": ["new-vehicles"]}),
        dm_data.filter({"Variables": ["new-vehicles"]}),
        year_start,
        year_end,
        years_calibration,
    )

    # passenger stock and new eurostat
    model_variables = ["tra_passenger_vehicle-fleet", "tra_passenger_new-vehicles"]
    dm_model = DM_passenger_out["tech"].filter({"Variables": model_variables})
    dm_model.rename_col(
        model_variables, ["stock-vehicles", "new-vehicles"], "Variables"
    )
    # dm_model_planes = dm_model.filter({"Categories1" : ['aviation'], "Categories2" : ["kerosene"]})
    dm_model = dm_model.filter({"Categories1": ["2W", "LDV", "bus", "metrotram"]})
    dm_data = DM_fleet_check["eurostat_passenger"].filter({"Country": ["EU27"]})
    dm_data_total = dm_data.filter({"Categories2": ["total"]})
    # dm_data_planes =  DM_fleet_check["eurostat_passenger_aviation_number_of_planes"].filter({"Country" : ["EU27"], "Categories2" : ["kerosene"]})
    dm_data_total.group_all("Categories2")
    dm_data.drop("Categories2", "total")

    DM["new-vehicles"]["passenger"]["byengine"]["EUROSTAT_road_eqr"] = (
        difference_with_data(
            dm_model.filter({"Variables": ["new-vehicles"]}),
            dm_data.filter({"Variables": ["new-vehicles"]}),
            year_start,
            year_end,
            years_calibration,
        )
    )
    DM["stock-vehicles"]["passenger"]["byengine"]["EUROSTAT_road_eqs"] = (
        difference_with_data(
            dm_model.filter({"Variables": ["stock-vehicles"]}),
            dm_data.filter({"Variables": ["stock-vehicles"]}),
            year_start,
            year_end,
            years_calibration,
        )
    )

    dm_model_total = dm_model.group_all("Categories2", inplace=False)
    DM["stock-vehicles"]["passenger"]["bymode"] = {}
    DM["stock-vehicles"]["passenger"]["bymode"]["EUROSTAT_road_eqs"] = (
        difference_with_data(
            dm_model_total.filter({"Variables": ["stock-vehicles"]}),
            dm_data_total.filter({"Variables": ["stock-vehicles"]}),
            year_start,
            year_end,
            years_calibration,
        )
    )

    DM["new-vehicles"]["passenger"]["bymode"] = {}
    DM["new-vehicles"]["passenger"]["bymode"]["EUROSTAT_road_eqr"] = (
        difference_with_data(
            dm_model_total.filter({"Variables": ["new-vehicles"]}),
            dm_data_total.filter({"Variables": ["new-vehicles"]}),
            year_start,
            year_end,
            years_calibration,
        )
    )

    # TODO: for the moment I can't do the following without transformation as model is in seats and data is in planes
    # DM["stock-vehicles"]["passenger"]["aviation"] = {}
    # DM["stock-vehicles"]["passenger"]["aviation"]["EUROSTAT_avia_eq_arc_typ"] = \
    #     difference_with_data(dm_model_planes.filter({"Variables" : ["stock-vehicles"]}),
    #                          dm_data_planes.filter({"Variables" : ["stock-vehicles"]}),
    #                          year_start, year_end, years_calibration)
    # DM["new-vehicles"]["passenger"]["aviation"] = {}
    # DM["new-vehicles"]["passenger"]["aviation"]["EUROSTAT_avia_eq_arc_typ"] = \
    #     difference_with_data(dm_model_planes.filter({"Variables" : ["new-vehicles"]}),
    #                          dm_data_planes.filter({"Variables" : ["new-vehicles"]}),
    #                          year_start, year_end, years_calibration)

    # freight stock and new eurostat
    model_variables = ["tra_freight_vehicle-fleet", "tra_freight_new-vehicles"]
    dm_model = DM_freight_out["tech"].filter({"Variables": model_variables})
    dm_model.rename_col(
        model_variables, ["stock-vehicles", "new-vehicles"], "Variables"
    )
    dm_model.groupby({"trucks": ["HDVH", "HDVL", "HDVM"]}, "Categories1", inplace=True)
    dm_model = dm_model.filter({"Categories1": ["IWW", "trucks"]})
    dm_model.group_all("Categories2")
    dm_data = DM_fleet_check["eurostat_freight"].filter({"Country": ["EU27"]})
    dm_data = dm_data.filter({"Categories2": ["total"]})
    dm_data.group_all("Categories2")

    DM["new-vehicles"]["freight"]["bymode"] = {}
    DM["new-vehicles"]["freight"]["bymode"]["EUROSTAT_road_eqr_lormot"] = (
        difference_with_data(
            dm_model.filter({"Variables": ["new-vehicles"]}),
            dm_data.filter({"Variables": ["new-vehicles"]}),
            year_start,
            year_end,
            years_calibration,
        )
    )

    DM["stock-vehicles"]["freight"]["bymode"] = {}
    DM["stock-vehicles"]["freight"]["bymode"][
        "EUROSTAT_road_eqs_lormot_iww_eq_loadcap"
    ] = difference_with_data(
        dm_model.filter({"Variables": ["stock-vehicles"]}),
        dm_data.filter({"Variables": ["stock-vehicles"]}),
        year_start,
        year_end,
        years_calibration,
    )

    # train passenger and freight
    model_variables = ["tra_passenger_vehicle-fleet", "tra_passenger_new-vehicles"]
    dm_model = DM_passenger_out["tech"].filter(
        {
            "Variables": model_variables,
            "Categories1": ["rail"],
            "Categories2": ["CEV", "ICE-diesel"],
        }
    )
    dm_model.rename_col(
        model_variables, ["stock-vehicles", "new-vehicles"], "Variables"
    )
    dm_model.rename_col("rail", "rail-passenger", "Categories1")
    model_variables = ["tra_freight_vehicle-fleet", "tra_freight_new-vehicles"]
    dm_temp = DM_freight_out["tech"].filter(
        {
            "Variables": model_variables,
            "Categories1": ["rail"],
            "Categories2": ["CEV", "ICE-diesel"],
        }
    )
    dm_temp.rename_col(model_variables, ["stock-vehicles", "new-vehicles"], "Variables")
    dm_temp.rename_col("rail", "rail-freight", "Categories1")
    dm_model.append(dm_temp, "Categories1")
    dm_model.groupby(
        {"rail": ["rail-passenger", "rail-freight"]}, "Categories1", inplace=True
    )
    dm_data = DM_fleet_check["eurostat_total_rail"].filter(
        {"Country": ["EU27"], "Categories2": ["CEV", "ICE-diesel"]}
    )

    DM["stock-vehicles"]["passenger_and_freight"] = {}
    DM["stock-vehicles"]["passenger_and_freight"]["rail"] = {}
    DM["stock-vehicles"]["passenger_and_freight"]["rail"]["EUROSTAT_rail_eq_locon"] = (
        difference_with_data(
            dm_model.filter({"Variables": ["stock-vehicles"]}),
            dm_data.filter({"Variables": ["stock-vehicles"]}),
            year_start,
            year_end,
            years_calibration,
        )
    )

    ###################
    ##### PKM-TKM #####
    ###################

    # get data
    filepath = os.path.join(
        current_file_directory,
        "../_database/pre_processing/transport/EU/data/datamatrix/calibration_pkm_tkm.pickle",
    )
    with open(filepath, "rb") as handle:
        DM_pkm_check = pickle.load(handle)

    # passenger
    dm_model = DM_passenger_out["power"].filter(
        {"Variables": ["tra_passenger_transport-demand"]}
    )
    dm_model.group_all("Categories2")
    dm_model.groupby({"train": ["metrotram", "rail"]}, "Categories1", inplace=True)
    dm_data = DM_pkm_check["eurostat_passenger_pkm"].filter({"Country": ["EU27"]})
    DM["pkm-tkm"] = {}
    DM["pkm-tkm"]["passenger"] = {}
    DM["pkm-tkm"]["passenger"]["EUROSTAT_various_pa"] = difference_with_data(
        dm_model, dm_data, year_start, year_end, years_calibration
    )

    # freight
    dm_model = DM_freight_out["power"].filter(
        {"Variables": ["tra_freight_transport-demand-tkm"]}
    )
    dm_model.group_all("Categories2")
    dm_model.groupby({"trucks": ["HDVH", "HDVL", "HDVM"]}, "Categories1", inplace=True)
    dm_data = DM_pkm_check["eurostat_freight_tkm"].filter({"Country": ["EU27"]})
    DM["pkm-tkm"]["freight"] = {}
    DM["pkm-tkm"]["freight"]["EUROSTAT_various_go"] = difference_with_data(
        dm_model, dm_data, year_start, year_end, years_calibration
    )

    ###############
    ##### VKM #####
    ###############

    # TODO: there is no eurostat data at EU27 level, just on single countries, to be decided what to do

    # # get data
    # filepath = os.path.join(current_file_directory, '../_database/pre_processing/transport/EU/data/datamatrix/calibration_vkm.pickle')
    # with open(filepath, 'rb') as handle: DM_vkm_check = pickle.load(handle)

    # # passenger
    # dm_model = DM_passenger_out["tech"].filter({"Variables" : ["tra_passenger_transport-demand-vkm"]})
    # dm_model.filter({"Categories1" : ["LDV","rail"]}, inplace=True)
    # dm_model.change_unit("tra_passenger_transport-demand-vkm",1,"km","vkm")
    # dm_model_total = dm_model.group_all("Categories2", inplace=False)
    # dm_data = DM_vkm_check["eurostat_passenger_vkm"].filter({"Country" : ["EU27"]})
    # dm_data.rename_col("train","rail","Categories1")
    # dm_data_total = dm_data.filter({"Categories2" : ["total"]})
    # dm_data_total.group_all("Categories2")
    # dm_data.drop("Categories2","total")
    # DM["vkm"] = {}
    # DM["vkm"]["passenger"] = {}
    # DM["vkm"]["passenger"]["byengine"] = {}
    # DM["vkm"]["passenger"]["byengine"]["EUROSTAT_road_tf_veh"] = \
    #     difference_with_data(dm_model, dm_data, year_start, year_end, years_calibration)

    # DM["vkm"]["passenger"]["bymode"] = {}
    # DM["vkm"]["passenger"]["byengine"]["EUROSTAT_road_tf_veh"] = \
    #     difference_with_data(dm_model_total, dm_data_total, year_start, year_end, years_calibration)

    ##################
    ##### ENERGY #####
    ##################

    # get data
    filepath = os.path.join(
        current_file_directory,
        "../_database/pre_processing/transport/EU/data/datamatrix/calibration_energy-demand.pickle",
    )
    with open(filepath, "rb") as handle:
        DM_ene_check = pickle.load(handle)

    # passenger energy demand against energy balance jrc (by mode)
    dm_model = DM_passenger_out["mode"].filter(
        {"Variables": ["tra_passenger_energy-demand-by-mode"]}
    )
    # dm_data = DM_ene_check["passenger_jrc"].group_all("Categories2", inplace=False).filter({"Country" : ["EU27"]})
    dm_data = DM_ene_check["passenger_jrc"].filter({"Country": ["EU27"]})
    dm_data.drop("Categories2", "other")
    dm_data.group_all("Categories2")
    DM["energy"] = {}
    DM["energy"]["passenger"] = {}
    DM["energy"]["passenger"]["bymode"] = {}
    DM["energy"]["passenger"]["bymode"]["JRC_EnergyBalance"] = difference_with_data(
        dm_model, dm_data, year_start, year_end, years_calibration
    )

    # passenger energy demand against energy balance jrc (by carrier)
    dm_model = DM_passenger_out["energy"].copy()
    dm_data = (
        DM_ene_check["passenger_jrc"]
        .group_all("Categories1", inplace=False)
        .filter({"Country": ["EU27"]})
    )
    dm_data.drop("Categories1", "other")
    DM["energy"]["passenger"]["bycarrier"] = {}
    DM["energy"]["passenger"]["bycarrier"]["JRC_EnergyBalance"] = difference_with_data(
        dm_model, dm_data, year_start, year_end, years_calibration
    )

    # freight energy demand against energy balance jrc (by mode)
    dm_model = DM_freight_out["mode"].filter(
        {"Variables": ["tra_freight_energy-demand"]}
    )
    dm_model.groupby({"trucks": ["HDVH", "HDVL", "HDVM"]}, "Categories1", inplace=True)
    dm_model.sort("Categories1")
    dm_data = DM_ene_check["freight_jrc"].copy()
    dm_data.drop("Categories2", "other")
    dm_data = dm_data.group_all("Categories2", inplace=False).filter(
        {"Country": ["EU27"]}
    )
    DM["energy"]["freight"] = {}
    DM["energy"]["freight"]["bymode"] = {}
    DM["energy"]["freight"]["bymode"]["JRC_EnergyBalance"] = difference_with_data(
        dm_model, dm_data, year_start, year_end, years_calibration
    )

    # freight energy demand against energy balance jrc (by carrier)
    dm_model = DM_freight_out["energy"].copy()
    dm_data = (
        DM_ene_check["freight_jrc"]
        .group_all("Categories1", inplace=False)
        .filter({"Country": ["EU27"]})
    )
    carriers_model = dm_model.col_labels["Categories1"]
    carriers_data = dm_data.col_labels["Categories1"]
    import numpy as np

    other = np.array(carriers_model)[
        [c not in carriers_data for c in carriers_model]
    ].tolist()
    dm_model.groupby({"other": other}, "Categories1", inplace=True)
    DM["energy"]["freight"]["bycarrier"] = {}
    DM["energy"]["freight"]["bycarrier"]["JRC_EnergyBalance"] = difference_with_data(
        dm_model, dm_data, year_start, year_end, years_calibration
    )

    # energy demand against energy balance eurostat (by mode)
    dm_model = DM_passenger_out["mode"].filter(
        {"Variables": ["tra_passenger_energy-demand-by-mode"]}
    )
    dm_model_fre = DM_freight_out["mode"].filter(
        {"Variables": ["tra_freight_energy-demand"]}
    )
    dm_model_fre.groupby(
        {"trucks": ["HDVH", "HDVL", "HDVM"]}, "Categories1", inplace=True
    )
    for c in dm_model_fre.col_labels["Categories1"]:
        dm_model_fre.rename_col(c, "freight_" + c, "Categories1")
    dm_model_fre.rename_col(
        "tra_freight_energy-demand", "tra_passenger_energy-demand-by-mode", "Variables"
    )
    dm_model.append(dm_model_fre, "Categories1")
    dm_model.groupby(
        {
            "aviation": ["aviation", "freight_aviation"],
            "rail": ["rail", "freight_rail", "metrotram"],
            "road": ["2W", "LDV", "bus", "freight_trucks"],
            "marine": ["freight_IWW", "freight_marine"],
        },
        "Categories1",
        inplace=True,
    )
    dm_model.sort("Categories1")
    dm_data = DM_ene_check["all_eurostat"].copy()
    dm_data.drop("Categories2", "other")
    dm_data = dm_data.group_all("Categories2", inplace=False).filter(
        {"Country": ["EU27"]}
    )
    DM["energy"]["passenger_and_freight"] = {}
    DM["energy"]["passenger_and_freight"]["bymode"] = {}
    DM["energy"]["passenger_and_freight"]["bymode"]["EUROSTAT_nrg_bal_c"] = (
        difference_with_data(dm_model, dm_data, year_start, year_end, years_calibration)
    )

    # energy demand against energy balance eurostat (by carrier)
    dm_model = DM_passenger_out["energy"].copy()
    dm_model_fre = DM_freight_out["energy"].copy()
    dm_data = (
        DM_ene_check["all_eurostat"]
        .group_all("Categories1", inplace=False)
        .filter({"Country": ["EU27"]})
    )
    carriers_model = dm_model_fre.col_labels["Categories1"]
    carriers_data = dm_data.col_labels["Categories1"]
    other = np.array(carriers_model)[
        [c not in carriers_data for c in carriers_model]
    ].tolist()
    dm_model_fre.groupby({"other": other}, "Categories1", inplace=True)
    dm_model.add(np.nan, "Categories1", "other", dummy=True)
    dm_model.append(dm_model_fre, "Variables")
    dm_model.groupby(
        {
            "energy-demand-total": [
                "tra_passenger_energy-demand-by-fuel",
                "tra_freight_total-energy",
            ]
        },
        "Variables",
        inplace=True,
    )
    DM["energy"]["passenger_and_freight"]["bycarrier"] = {}
    DM["energy"]["passenger_and_freight"]["bycarrier"]["EUROSTAT_nrg_bal_c"] = (
        difference_with_data(dm_model, dm_data, year_start, year_end, years_calibration)
    )

    #####################
    ##### EMISSIONS #####
    #####################

    # get data
    filepath = os.path.join(
        current_file_directory,
        "../_database/pre_processing/transport/EU/data/datamatrix/calibration_emissions.pickle",
    )
    with open(filepath, "rb") as handle:
        DM_emi_check = pickle.load(handle)

    # passenger emissions against emissions balance jrc by mode
    dm_model = DM_passenger_out["mode"].filter(
        {"Variables": ["tra_passenger_emissions-by-mode_CO2"]}
    )
    dm_model.deepen(based_on="Variables")
    dm_model.switch_categories_order("Categories1", "Categories2")
    dm_data = DM_emi_check["passenger_jrc"].copy()
    dm_data.drop("Categories3", "other")
    dm_data = dm_data.group_all("Categories3", inplace=False).filter(
        {"Country": ["EU27"]}
    )
    dm_data.change_unit("calib-emissions", factor=1e-3, old_unit="kt", new_unit="Mt")
    DM["emissions"] = {}
    DM["emissions"]["passenger"] = {}
    DM["emissions"]["passenger"]["bymode"] = {}
    DM["emissions"]["passenger"]["bymode"]["JRC_EmissionBalance"] = (
        difference_with_data(dm_model, dm_data, year_start, year_end, years_calibration)
    )

    # freight emissions against emissions balance jrc by mode
    dm_model = DM_freight_out["emissions"].filter({"Categories2": ["CO2"]})
    dm_model.groupby({"trucks": ["HDVH", "HDVL", "HDVM"]}, "Categories1", inplace=True)
    dm_data = DM_emi_check["freight_jrc"].copy()
    dm_data.drop("Categories3", "other")
    dm_data.group_all("Categories3")
    dm_data.switch_categories_order("Categories1", "Categories2")
    dm_data.filter({"Country": ["EU27"]}, inplace=True)
    dm_data.change_unit("calib-emissions", 1e-3, "kt", "Mt")
    DM["emissions"]["freight"] = {}
    DM["emissions"]["freight"]["bymode"] = {}
    DM["emissions"]["freight"]["bymode"]["JRC_EmissionBalance"] = difference_with_data(
        dm_model, dm_data, year_start, year_end, years_calibration
    )

    # passenger and freight emissions against emissions balance eurostat by mode
    dm_model = DM_passenger_out["emissions"].filter({"Categories2": ["CO2"]})
    dm_temp = DM_freight_out["emissions"].filter({"Categories2": ["CO2"]})
    dm_temp.groupby({"trucks": ["HDVH", "HDVL", "HDVM"]}, "Categories1", inplace=True)
    dm_temp.rename_col(
        ["IWW", "aviation", "marine", "rail", "trucks"],
        ["IWW", "aviation-freight", "marine", "rail-freight", "trucks"],
        "Categories1",
    )
    dm_model.rename_col("tra_passenger_emissions", "emissions", "Variables")
    dm_temp.rename_col("tra_freight_emissions", "emissions", "Variables")
    dm_model.append(dm_temp, "Categories1")
    dm_model.groupby(
        {
            "aviation": ["aviation", "aviation-freight"],
            "rail": ["rail", "rail-freight", "metrotram"],
            "trucks-bus": ["trucks", "bus"],
            "marine": ["IWW", "marine"],
        },
        "Categories1",
        inplace=True,
    )
    dm_model.sort("Categories1")
    dm_data = DM_emi_check["all_eurostat"].filter(
        {"Country": ["EU27"], "Categories1": ["CO2"]}
    )
    dm_data.drop("Categories2", ["other-road", "other-transport"])
    dm_data.switch_categories_order("Categories1", "Categories2")
    dm_data.sort("Categories1")
    dm_data.change_unit("calib-emissions", 1e-3, "kt", "Mt")
    DM["emissions"]["passenger_and_freight"] = {}
    DM["emissions"]["passenger_and_freight"]["bymode"] = {}
    DM["emissions"]["passenger_and_freight"]["bymode"]["EUROSTAT_env_air_gge"] = (
        difference_with_data(dm_model, dm_data, year_start, year_end, years_calibration)
    )

    #################
    ##### CHECK #####
    #################

    # stock and new
    DM["stock-vehicles"]["passenger"]["byengine"]["JRC_Transport"]
    DM["new-vehicles"]["passenger"]["byengine"]["JRC_Transport"]
    # fine
    DM["stock-vehicles"]["passenger"]["byengine"]["EUROSTAT_road_eqs"]
    DM["new-vehicles"]["passenger"]["byengine"]["EUROSTAT_road_eqr"]
    # LDV_PHEV-diesel and bus_ICE-gas quite off, rest kinda fine
    DM["stock-vehicles"]["passenger"]["bymode"][
        "EUROSTAT_road_eqs"
    ]  # this is with Eurostat total
    DM["new-vehicles"]["passenger"]["bymode"][
        "EUROSTAT_road_eqr"
    ]  # this is with Eurostat total
    # 2W off
    DM["stock-vehicles"]["freight"]["byengine"]["JRC_Transport"]
    DM["new-vehicles"]["freight"]["byengine"]["JRC_Transport"]
    # stock generally fine but off in new vehicles for IWW, aviation, marine, rail (especially rail)
    DM["stock-vehicles"]["freight"]["bymode"]["EUROSTAT_road_eqs_lormot_iww_eq_loadcap"]
    # IWW high but generally fine

    # pkm-tkm
    DM["pkm-tkm"]["passenger"]["EUROSTAT_various_pa"]
    # kind of ok
    DM["pkm-tkm"]["freight"]["EUROSTAT_various_go"]
    # aviation high but rest it's kind of ok (bizarre)

    # energy
    # TODO: when checking by mode, I have exlcuded other carrier from data before aggregating data, see if this is fine
    DM["energy"]["passenger"]["bymode"]["JRC_EnergyBalance"]
    # 2W, LDV, bus, metro good, aviation 60% higher in model, rail 20% higher in model.
    DM["energy"]["freight"]["bymode"]["JRC_EnergyBalance"]
    # Other than trucks, rest is completely off (aviation less than other), especially marine
    DM["energy"]["passenger_and_freight"]["bymode"]["EUROSTAT_nrg_bal_c"]
    # similar results, road is ok, and rest is off (aviation less off than marine and rail, marine super off)
    DM["energy"]["passenger"]["bycarrier"]["JRC_EnergyBalance"]
    # nothing too bad but biogas. Diesel, gasoline and electricity ok.
    DM["energy"]["freight"]["bycarrier"]["JRC_EnergyBalance"]
    # electricity off (probably due to rail, off by the similar scale), and other completely off
    DM["energy"]["passenger_and_freight"]["bycarrier"]["EUROSTAT_nrg_bal_c"]
    # electricity less of than jrc energy balance as probably freight gets a bit absorbed by passenger (they are together here), other completely off

    # emissions by mode
    # TODO: when checking by mode, I have exlcuded other carrier from data before aggregating data, see if this is fine
    DM["emissions"]["passenger"]["bymode"]["JRC_EmissionBalance"]
    # with aviation and rail there is probably an issue with emission factors of kerosene and electricity (or rail it can
    # also be correct emission factors wrong mix of electricity and diesel)
    DM["emissions"]["freight"]["bymode"]["JRC_EmissionBalance"]
    # everything much more off than energy demand. For example for trucks we have 37% energy and 35666% emissions.
    # Probably there are issues of fuel mix.
    DM["emissions"]["passenger_and_freight"]["bymode"]["EUROSTAT_env_air_gge"]
    # same

    # Conclusions 16.01.26:
    # passengers are kinda fine, freight are problematic
    # For trucks: stock, new, tkm, energy are fine but emissions is off
    # for rail freight: stock is fine, new is off, and energy electricity for freight is off
    # aviation and marine: they are quite off, to check system boundaries

    return DM


def check_transport_CH(
    current_file_directory,
    DM_passenger_out,
    DM_freight_out,
    year_start=2000,
    year_end=2021,
    years_calibration=[2019],
):

    DM = {}

    ###############
    ##### VKM #####
    ###############

    # get data
    filepath = os.path.join(
        current_file_directory,
        "../_database/pre_processing/transport/Switzerland/data/datamatrix/calibration_vkm.pickle",
    )
    with open(filepath, "rb") as handle:
        DM_vkm_check = pickle.load(handle)

    modes = ["2W", "LDV", "bus"]
    dm_model = DM_passenger_out["tech"].filter(
        {"Variables": ["tra_passenger_transport-demand-vkm"], "Categories1": modes}
    )
    dm_data = DM_vkm_check["passenger-and-freight_road_EP-2050"].filter(
        {"Categories1": modes}
    )
    dm_model = dm_model.filter(
        {
            "Years": dm_data.col_labels["Years"],
            "Categories2": dm_data.col_labels["Categories2"],
        }
    )
    # difference_with_data_graph(dm_model, dm_data, "vehicle km [vkm]")

    ##################
    ##### ENERGY #####
    ##################

    # get data
    filepath = os.path.join(
        current_file_directory,
        "../_database/pre_processing/transport/Switzerland/data/datamatrix/calibration_energy-demand.pickle",
    )
    with open(filepath, "rb") as handle:
        DM_ene_check = pickle.load(handle)

    # passenger energy demand against EP-2050 (by engine)
    dm_model = DM_passenger_out["tech"].filter(
        {"Variables": ["tra_passenger_energy-demand"]}
    )
    dm_data = DM_ene_check["passenger_EP-2050"].copy()
    dm_model = dm_model.filter(
        {
            "Categories1": dm_data.col_labels["Categories1"],
            "Categories2": dm_data.col_labels["Categories2"],
        }
    )
    DM["energy"] = {}
    DM["energy"]["passenger"] = {}
    DM["energy"]["passenger"]["byengine"] = {}
    DM["energy"]["passenger"]["byengine"]["EP-2050"] = difference_with_data(
        dm_model, dm_data, year_start, year_end, years_calibration
    )
    # difference_with_data_graph(dm_model, dm_data, "energy demand [MJ]")

    # freight energy demand against EP-2050 (by engine)
    dm_model = DM_freight_out["tech"].filter(
        {"Variables": ["tra_freight_energy-demand"]}
    )
    dm_data = DM_ene_check["freight_EP-2050"].copy()
    dm_model = dm_model.filter(
        {
            "Categories1": dm_data.col_labels["Categories1"],
            "Categories2": dm_data.col_labels["Categories2"],
        }
    )
    DM["energy"]["freight"] = {}
    DM["energy"]["freight"]["byengine"] = {}
    DM["energy"]["freight"]["byengine"]["EP-2050"] = difference_with_data(
        dm_model, dm_data, year_start, year_end, years_calibration
    )

    # passenger and freight aviation against EP-2050
    dm_model = DM_freight_out["tech"].filter(
        {
            "Variables": ["tra_freight_energy-demand"],
            "Categories1": ["aviation"],
            "Categories2": ["kerosene"],
        }
    )
    dm_model.append(
        DM_passenger_out["tech"].filter(
            {
                "Variables": ["tra_passenger_energy-demand"],
                "Categories1": ["aviation"],
                "Categories2": ["kerosene"],
            }
        ),
        "Variables",
    )
    dm_model.groupby(
        {
            "tra_energy-demand": [
                "tra_freight_energy-demand",
                "tra_passenger_energy-demand",
            ]
        },
        "Variables",
        inplace=True,
    )
    dm_data = DM_ene_check["freight-and-passenger_aviation_EP-2050"].copy()
    DM["energy"]["passenger_and_freight"] = {}
    DM["energy"]["passenger_and_freight"]["aviation"] = {}
    DM["energy"]["passenger_and_freight"]["aviation"]["EP-2050"] = difference_with_data(
        dm_model, dm_data, year_start, year_end, years_calibration
    )

    # passenger and freight against energy balance (by carrier)
    main_carriers = ["diesel", "electricity", "gas", "gasoline", "kerosene"]
    other_renewables = [
        "ICEbio",
        "biodiesel",
        "biogas",
        "biogasoline",
        "efuel",
        "hydrogen",
        "kerosenebio",
        "biojetfuel",
        "ejetfuel",
    ]
    dm_model = DM_freight_out["energy"].filter(
        {"Categories1": main_carriers + other_renewables}
    )
    dm_model.groupby(
        {"renewables-other": other_renewables}, "Categories1", inplace=True
    )
    dm_temp = DM_passenger_out["energy"].copy()
    other_renewables = [
        "biodiesel",
        "biogas",
        "biogasoline",
        "efuel",
        "hydrogen",
        "kerosenebio",
    ]
    dm_temp.groupby({"renewables-other": other_renewables}, "Categories1", inplace=True)
    dm_model.append(dm_temp, "Variables")
    dm_model.groupby(
        {
            "tra_energy-demand": [
                "tra_freight_total-energy",
                "tra_passenger_energy-demand-by-fuel",
            ]
        },
        "Variables",
        inplace=True,
    )
    dm_data = DM_ene_check["freight-and-passenger_energy-balance"].copy()
    DM["energy"]["passenger_and_freight"]["bycarrier"] = {}
    DM["energy"]["passenger_and_freight"]["bycarrier"]["energy-balance"] = (
        difference_with_data(dm_model, dm_data, year_start, year_end, years_calibration)
    )

    # passenger and freight electricity against energy balance
    dm_model = DM_freight_out["tech"].filter(
        {"Variables": ["tra_freight_energy-demand"], "Categories2": ["BEV", "CEV"]}
    )
    dm_model.group_all("Categories1")
    dm_model.rename_col(
        ["CEV", "BEV"], ["electricity-rail", "electricity-road"], "Categories1"
    )
    dm_temp = DM_passenger_out["tech"].filter(
        {
            "Variables": ["tra_passenger_energy-demand"],
            "Categories1": ["2W", "LDV", "bus", "metrotram", "rail"],
            "Categories2": ["BEV", "CEV"],
        }
    )
    dm_temp.groupby(
        {"road": ["2W", "LDV", "bus"], "rail": ["metrotram", "rail"]},
        "Categories1",
        inplace=True,
    )
    dm_temp = dm_temp.flatten()
    dm_temp.drop("Categories1", ["rail_BEV", "road_CEV"])
    dm_temp.rename_col(
        ["rail_CEV", "road_BEV"],
        ["electricity-rail", "electricity-road"],
        "Categories1",
    )
    dm_model.append(dm_temp, "Variables")
    dm_model.groupby(
        {"energy-demand": ["tra_freight_energy-demand", "tra_passenger_energy-demand"]},
        "Variables",
        inplace=True,
    )
    dm_data = DM_ene_check[
        "freight-and-passenger_electricity-split_energy-balance"
    ].filter({"Categories1": ["electricity-rail", "electricity-road"]})
    DM["energy"]["passenger_and_freight"]["electricity"] = difference_with_data(
        dm_model, dm_data, year_start, year_end, years_calibration
    )

    #####################
    ##### EMISSIONS #####
    #####################

    # get data
    filepath = os.path.join(
        current_file_directory,
        "../_database/pre_processing/transport/Switzerland/data/datamatrix/calibration_emissions.pickle",
    )
    with open(filepath, "rb") as handle:
        DM_emi_check = pickle.load(handle)

    dm_model = DM_passenger_out["emissions"].copy()
    dm_model.groupby({"rail": ["metrotram", "rail"]}, "Categories1", inplace=True)
    dm_model.rename_col("aviation", "aviation-passenger", "Categories1")
    dm_model.rename_col("rail", "rail-passenger", "Categories1")
    dm_model.rename_col("tra_passenger_emissions", "emissions", "Variables")
    dm_temp = DM_freight_out["emissions"].copy()
    dm_temp.groupby(
        {"trucks": ["HDVH", "HDVL", "HDVM"], "marine": ["marine", "IWW"]},
        "Categories1",
        inplace=True,
    )
    dm_temp.rename_col("aviation", "aviation-freight", "Categories1")
    dm_temp.rename_col("rail", "rail-freight", "Categories1")
    dm_temp.rename_col("tra_freight_emissions", "emissions", "Variables")
    dm_model.append(dm_temp, "Categories1")
    dm_model.groupby(
        {
            "aviation": ["aviation-passenger", "aviation-freight"],
            "rail": ["rail-passenger", "rail-freight"],
        },
        "Categories1",
        inplace=True,
    )
    dm_data = DM_emi_check["freight-and-passenger_emission-balance"].copy()
    DM["emissions"] = {}
    DM["emissions"]["passenger_and_freight"] = {}
    DM["emissions"]["passenger_and_freight"]["bymode"] = {}
    DM["emissions"]["passenger_and_freight"]["bymode"]["emission-balance"] = (
        difference_with_data(dm_model, dm_data, year_start, year_end, years_calibration)
    )
    # difference_with_data_graph(dm_model, dm_data, "Emissions [Mt]")

    return DM
