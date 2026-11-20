###################################
#####  VEHICLE EFFICIENCY  ########
###################################
# Passenger vehicle efficiency (MJ/km), fleet and new vehicles, ots.
#
# How each (mode, technology) is obtained:
#   - LDV ICE / PHEV: Swiss statistics on CO2 emissions (gCO2/km), converted to
#     MJ/km with the CO2 emission factors. Available for every region.
#   - Other modes and LDV BEV / FCEV: Swiss energy demand / vkm by technology.
#     Only available for Switzerland: cantons get the Swiss value, and new
#     vehicles get the fleet value.
#   - EP2050 calibration of the main ICE (mode, technology) pairs: the Swiss
#     fleet efficiency is set to EP2050 energy / EP2050 vkm. For LDV the same
#     correction factor is applied to every canton and to new vehicles.

import os

import numpy as np

import transition_compass_model._database.pre_processing.transport.Switzerland.get_data_functions.efficiency as get_data
from transition_compass_model._database.pre_processing.params import country_list
from transition_compass_model.model.common.auxiliary_functions import (
    dm_add_missing_variables,
    linear_fitting,
)

VAR_FLEET = "tra_passenger_veh-efficiency_fleet"
VAR_NEW = "tra_passenger_veh-efficiency_new"

# (mode, technology) pairs whose Swiss fleet efficiency is calibrated on EP2050
EP2050_CALIBRATED = {
    "2W": ["ICE-gasoline"],
    "LDV": ["ICE-diesel", "ICE-gas", "ICE-gasoline"],
    "bus": ["ICE-diesel"],
}
# EP2050 LDV ICE-gas efficiency is not usable before 2007: in those years the
# 2007 correction factor is used instead.
LDV_GAS_FIRST_GOOD_YEAR = 2007


def convert_eff_from_gCO2_km_to_MJ_km(
    dm_veh_eff_LDV, cdm_emissions_factors, new_var_name
):
    if "BEV" in dm_veh_eff_LDV.col_labels["Categories2"]:
        dm_veh_eff_LDV.drop("Categories2", ["BEV", "FCEV"])
    var_name = dm_veh_eff_LDV.col_labels["Variables"][0]
    dm_veh_eff_LDV.rename_col(var_name, "tmp_name", dim="Variables")
    cdm_emissions_CO2_LDV = cdm_emissions_factors.filter(
        {
            "Categories1": ["CO2"],
            "Categories2": dm_veh_eff_LDV.col_labels["Categories2"],
        }
    )
    cdm_emissions_CO2_LDV.sort("Categories2")
    dm_veh_eff_LDV.sort("Categories2")
    # I want to have an efficiency in MJ/km -> then I have to do  veh-eff(g/km) / emission-fact(g/MJ) = eff (MJ/km)
    arr_eff_MJ_km = (
        dm_veh_eff_LDV.array / cdm_emissions_CO2_LDV.array[np.newaxis, :, :, :]
    )
    dm_veh_eff_LDV.add(
        arr_eff_MJ_km, dim="Variables", col_label=new_var_name, unit="MJ/km"
    )
    dm_veh_eff_LDV.filter({"Variables": [new_var_name]}, inplace=True)
    return dm_veh_eff_LDV


def compute_tech_share(dm_private_fleet, dm_public_fleet):
    # All technologies
    dm_private = dm_private_fleet.copy()
    dm_public = dm_public_fleet.copy()
    cat_tech = set(dm_private_fleet.col_labels["Categories2"]).union(
        set(dm_public_fleet.col_labels["Categories2"])
    )
    # Join private and public
    dm_add_missing_variables(dm_public, {"Categories2": cat_tech})
    dm_add_missing_variables(dm_private, {"Categories2": cat_tech})
    dm_private.append(dm_public, dim="Categories1")
    # Technology share
    dm_tech_share = dm_private.normalise("Categories2", inplace=False)

    return dm_tech_share


def get_energy_demand_MJ(dm_energy):
    # Energy demand by mode and technology in MJ, with bio-gasoline counted as gasoline
    dm_energy_MJ = dm_energy.copy()
    dm_energy_MJ.groupby(
        {"ICE-gasoline": ["biogasoline", "gasoline"]}, dim="Categories2", inplace=True
    )
    dm_energy_MJ.change_unit(
        "tra_energy_demand", old_unit="TWh", new_unit="MJ", factor=3.6e9
    )
    return dm_energy_MJ


def get_LDV_efficiency_from_co2(this_dir, cdm_emissions_factors, years_ots):
    # LDV ICE and PHEV efficiency (MJ/km), fleet and new vehicles, every region.
    # Swiss statistics give gCO2/km; BEV and FCEV values are unreliable and dropped.
    # Fleet: https://stats.swiss/vis?lc=fr&df[ds]=disseminate&df[id]=DF_MFZ_1_EMISSION&df[ag]=CH1.MFZ_IVS
    dm_fleet_co2 = get_data.get_vehicle_efficiency_co2(
        os.path.join(this_dir, "../data/tra_veh_efficiency_swiss_stat_co2.pickle"),
        "CH1.MFZ_IVS",
        "DF_MFZ_1_EMISSION",
        var_name=VAR_FLEET,
    )
    dm_fleet_co2.filter({"Country": country_list, "Years": years_ots}, inplace=True)

    # New vehicles: https://stats.swiss/vis?lc=fr&df%5bds%5d=disseminate&df%5bid%5d=DF_IVS_1_EMISSION&df%5bag%5d=CH1.MFZ_IVS
    dm_new_co2, _ = get_data.get_new_vehicle_efficiency_co2(
        os.path.join(this_dir, "../data/tra_new-veh_efficiency_co2.pickle"),
        "CH1.MFZ_IVS",
        "DF_IVS_1_EMISSION",
        var_name=VAR_NEW,
    )
    dm_new_co2.filter({"Country": country_list}, inplace=True)

    # gCO2/km -> MJ/km
    dm_LDV = convert_eff_from_gCO2_km_to_MJ_km(
        dm_fleet_co2, cdm_emissions_factors, new_var_name=VAR_FLEET
    )
    dm_LDV_new = convert_eff_from_gCO2_km_to_MJ_km(
        dm_new_co2, cdm_emissions_factors, new_var_name=VAR_NEW
    )
    dm_LDV.append(dm_LDV_new, dim="Variables")
    dm_LDV.filter({"Years": years_ots}, inplace=True)
    return dm_LDV


def compute_vehicle_efficiency_from_energy_demand(
    dm_energy_MJ, dm_vkm, dm_private_fleet, dm_public_fleet
):
    # Efficiency (MJ/km) = Energy (MJ) / Demand by technology (vkm),
    # with Demand by technology = Demand (vkm) * fleet technology share (%)
    dm_tech_share = compute_tech_share(
        dm_private_fleet=dm_private_fleet.filter(
            {"Variables": ["tra_passenger_vehicle-fleet"]}
        ),
        dm_public_fleet=dm_public_fleet,
    )
    arr_demand_tech = dm_vkm[..., np.newaxis] * dm_tech_share[...]
    dm_tech_share.add(
        arr_demand_tech,
        dim="Variables",
        col_label="tra_passenger_demand-vkm",
        unit="vkm",
    )

    # Energy is only available for Switzerland
    dm_veh_eff = dm_tech_share.filter(
        {
            "Variables": ["tra_passenger_demand-vkm"],
            "Categories1": dm_energy_MJ.col_labels["Categories1"],
            "Categories2": dm_energy_MJ.col_labels["Categories2"],
            "Country": dm_energy_MJ.col_labels["Country"],
        }
    )
    dm_veh_eff.append(dm_energy_MJ, dim="Variables")
    dm_veh_eff.operation(
        "tra_energy_demand",
        "/",
        "tra_passenger_demand-vkm",
        out_col=VAR_FLEET,
        unit="MJ/km",
    )
    dm_veh_eff.filter({"Variables": [VAR_FLEET]}, inplace=True)

    return dm_veh_eff


def clean_electric_efficiency(dm_veh_eff):
    # No energy or no vkm -> efficiency is not known
    dm_veh_eff.array[dm_veh_eff.array == 0] = np.nan
    # LDV BEV 2011-2012 are outliers
    dm_veh_eff[:, 2011, VAR_FLEET, "LDV", "BEV"] = np.nan
    dm_veh_eff[:, 2012, VAR_FLEET, "LDV", "BEV"] = np.nan
    # Electric and hydrogen efficiencies are noisy: use their average over all years
    for mode, tech in [("LDV", "BEV"), ("LDV", "FCEV"), ("2W", "BEV")]:
        dm_veh_eff[:, :, VAR_FLEET, mode, tech] = np.nanmean(
            dm_veh_eff[:, :, VAR_FLEET, mode, tech], axis=1, keepdims=True
        )
    return dm_veh_eff


def add_metrotram_efficiency_from_JRC(dm_veh_eff):
    # JRC-IDEES-2021, TrRail_ene sheet (Vehicle-efficiency kgoe/100 km)
    # metrotram values go from 45 kgoe/100km in 2000 to 32.2 kgoe/100km  in 2021, roughly linearly
    var_name = VAR_FLEET
    dm_veh_eff.add(np.nan, col_label="metrotram", dim="Categories1", dummy=True)
    dm_veh_eff.add(np.nan, col_label="mt", dim="Categories2", dummy=True)
    dm_veh_eff["Switzerland", 2000, var_name, "metrotram", "mt"] = 45 * 41.868 / 100
    dm_veh_eff["Switzerland", 2021, var_name, "metrotram", "mt"] = 32.2 * 41.868 / 100
    dm_metrotram = dm_veh_eff.filter(
        {"Categories1": ["metrotram"], "Categories2": ["mt"]}
    )
    years_ots = dm_metrotram.col_labels["Years"]
    linear_fitting(dm_metrotram, years_ots=years_ots)
    dm_veh_eff["Switzerland", :, var_name, "metrotram", "mt"] = dm_metrotram[
        "Switzerland", :, var_name, "metrotram", "mt"
    ]
    return dm_veh_eff


def compute_ep2050_efficiency(dm_energy_MJ, dm_vkm_ep2050, mode, tech):
    # Swiss fleet efficiency (MJ/km) = EP2050 energy (MJ) / EP2050 vkm, by year
    energy = dm_energy_MJ["Switzerland", :, "tra_energy_demand", mode, tech]
    vkm = dm_vkm_ep2050["Switzerland", :, "tra_vkm_demand", mode, tech]
    return energy / vkm


def calibrate_on_ep2050(dm_veh_eff_CH, dm_LDV, dm_energy_MJ, dm_vkm_ep2050):
    years = dm_LDV.col_labels["Years"]
    for mode, techs in EP2050_CALIBRATED.items():
        for tech in techs:
            eff_ep2050 = compute_ep2050_efficiency(
                dm_energy_MJ, dm_vkm_ep2050, mode, tech
            )
            if mode != "LDV":
                # Switzerland only: replace with the EP2050 efficiency
                dm_veh_eff_CH["Switzerland", :, VAR_FLEET, mode, tech] = eff_ep2050
                continue
            # LDV: factor = EP2050 / Swiss CO2-based fleet efficiency, applied to
            # every region and to both fleet and new vehicles
            factor = eff_ep2050 / dm_LDV["Switzerland", :, VAR_FLEET, mode, tech]
            if tech == "ICE-gas":
                i_good = years.index(LDV_GAS_FIRST_GOOD_YEAR)
                factor[:i_good] = factor[i_good]
            dm_LDV[:, :, :, mode, tech] = (
                dm_LDV[:, :, :, mode, tech] * factor[np.newaxis, :, np.newaxis]
            )
    return dm_veh_eff_CH, dm_LDV


def expand_to_regions_and_new_vehicles(dm_veh_eff_CH):
    # New vehicles efficiency = fleet efficiency
    dm_veh_eff = dm_veh_eff_CH.copy()
    dm_veh_eff.add(
        dm_veh_eff.array.copy(), dim="Variables", col_label=VAR_NEW, unit="MJ/km"
    )
    # Every canton = Switzerland
    dm_CH = dm_veh_eff.copy()
    for canton in country_list:
        if canton == "Switzerland":
            continue
        dm_canton = dm_CH.copy()
        dm_canton.rename_col("Switzerland", canton, dim="Country")
        dm_veh_eff.append(dm_canton, dim="Country")
    return dm_veh_eff


def run(
    dm_energy,
    dm_vkm,
    dm_private_fleet,
    dm_public_fleet,
    cdm_emissions_factors,
    years_ots,
    dm_data_check_vkm,
):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    dm_energy_MJ = get_energy_demand_MJ(dm_energy)

    # LDV ICE / PHEV, every region, fleet + new: from CO2 statistics
    dm_LDV = get_LDV_efficiency_from_co2(this_dir, cdm_emissions_factors, years_ots)

    # Other modes + LDV BEV / FCEV, Switzerland, fleet: from energy / vkm
    dm_veh_eff_CH = compute_vehicle_efficiency_from_energy_demand(
        dm_energy_MJ, dm_vkm, dm_private_fleet, dm_public_fleet
    )
    dm_veh_eff_CH = clean_electric_efficiency(dm_veh_eff_CH)
    dm_veh_eff_CH = add_metrotram_efficiency_from_JRC(dm_veh_eff_CH)
    dm_veh_eff_CH.fill_nans("Years")

    # Calibrate main ICE technologies on EP2050
    dm_veh_eff_CH, dm_LDV = calibrate_on_ep2050(
        dm_veh_eff_CH,
        dm_LDV,
        dm_energy_MJ,
        dm_data_check_vkm["passenger-and-freight_road_EP-2050"],
    )

    # Cantons = Switzerland, new vehicles = fleet
    dm_veh_eff = expand_to_regions_and_new_vehicles(dm_veh_eff_CH)

    # Join: LDV = CO2-based ICE / PHEV + energy-based BEV / FCEV
    dm_LDV.append(
        dm_veh_eff.filter({"Categories1": ["LDV"], "Categories2": ["BEV", "FCEV"]}),
        dim="Categories2",
    )
    dm_veh_eff.drop("Categories1", "LDV")
    all_tech = set(dm_LDV.col_labels["Categories2"]).union(
        dm_veh_eff.col_labels["Categories2"]
    )
    dm_add_missing_variables(dm_LDV, {"Categories2": all_tech})
    dm_add_missing_variables(dm_veh_eff, {"Categories2": all_tech})
    dm_veh_eff.append(dm_LDV, dim="Categories1")

    return dm_veh_eff
