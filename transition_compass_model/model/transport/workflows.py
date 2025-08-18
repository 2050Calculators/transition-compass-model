from model.common.data_matrix_class import DataMatrix
import numpy as np
import model.transport.utils as utils
from model.common.auxiliary_functions import compute_stock

def passenger_fleet_energy(DM_passenger, dm_lfs, DM_other, cdm_const, years_setting):
    # SECTION Passenger - Demand-pkm by mode
    # dm_demand_by_mode [pkm] = modal_shares(urban) * demand_pkm(urban) + modal_shares(non-urban) * demand_pkm(non-urban)
    dm_modal_split = DM_passenger["passenger_modal_split"]
    dm_pkm_demand = DM_passenger["passenger_pkm_demand"]  # pkm/cap
    arr = (
        dm_lfs.array[..., np.newaxis]
        * dm_pkm_demand.array[..., np.newaxis]
        * dm_modal_split.array
    )
    dm_demand_by_mode = DataMatrix.based_on(
        arr,
        dm_modal_split,
        change={"Variables": ["tra_passenger_transport-demand"]},
        units={"tra_passenger_transport-demand": "pkm"},
    )
    # dm_demand_by_mode = compute_pkm_demand(dm_modal_split, dm_lfs_demand)
    del dm_modal_split, dm_pkm_demand
    # Remove walking and biking
    dm_demand_soft = dm_demand_by_mode.filter({"Categories1": ["walk", "bike"]})
    dm_demand_soft.rename_col(
        "tra_passenger_transport-demand",
        "tra_passenger_transport-demand-by-mode",
        dim="Variables",
    )
    dm_demand_by_mode.drop(dim="Categories1", col_label="walk|bike")

    # SECTION Passenger - Aviation Demand-pkm
    # demand_aviation [pkm] = demand aviation [pkm/cap] * pop
    dm_aviation_pkm = DM_passenger["passenger_aviation"]
    dm_pop = dm_lfs
    arr_av_pkm = dm_aviation_pkm.array * dm_pop.array[..., np.newaxis]
    dm_aviation_pkm.add(
        arr_av_pkm,
        dim="Variables",
        unit="pkm",
        col_label="tra_passenger_transport-demand",
    )
    dm_aviation_pkm.drop(col_label="tra_pkm-cap", dim="Variables")
    # tmp_aviation = dm_aviation_pkm.array[..., 0] * dm_pop.array[...]
    # dm_demand_by_mode.add(tmp_aviation, dim='Categories1', col_label='aviation')
    del arr_av_pkm

    dm_demand = dm_demand_by_mode.filter(
        {"Categories1": ["LDV", "2W", "bus", "metrotram", "rail"]}
    )
    # Add aviation to the demand
    dm_demand.append(dm_aviation_pkm, dim="Categories1")
    dm_mode = DM_passenger["passenger_all"]
    dm_mode.append(dm_demand, dim="Variables")
    del dm_demand

    # SECTION Passenger - Demand-vkm by mode
    # demand [vkm] = demand [pkm] / occupancy [pkm/vkm]
    dm_mode.operation(
        "tra_passenger_transport-demand",
        "/",
        "tra_passenger_occupancy",
        dim="Variables",
        out_col="tra_passenger_transport-demand-vkm",
        unit="vkm",
        div0="error",
    )
    # SECTION Passenger - Vehicle-fleet by mode
    # vehicle-fleet [number] = demand [vkm] / utilisation-rate [vkm/veh/year]
    dm_mode.operation(
        "tra_passenger_transport-demand-vkm",
        "/",
        "tra_passenger_utilisation-rate",
        dim="Variables",
        out_col="tra_passenger_vehicle-fleet",
        unit="number",
        div0="error",
        type=int,
    )

    # SECTION Passenger - Vehicle fleet, new-vehicle, vehicle-waste, efficiency by technology
    dm_tech = DM_passenger["passenger_tech"]
    #
    cols = {
        "lifetime": "tra_passenger_lifetime",
        "stock": "tra_passenger_vehicle-fleet",
        "waste": "tra_passenger_vehicle-waste",
        "new": "tra_passenger_new-vehicles",
        "tech-new": "tra_passenger_technology-share_new",
        "tech-stock": "tra_passenger_technology-share_fleet",
        "eff-stock": "tra_passenger_veh-efficiency_fleet",
        "eff-new": "tra_passenger_veh-efficiency_new",
    }
    dm_tech = utils.compute_stock_from_lifetime(
        dm_mode, dm_tech, var_names=cols, years_setting=years_setting
    )
    # dm_out_4 = dm_tech.group_all('Categories2', inplace=False)
    # dm_out_4.filter({'Categories1': ['aviation']}, inplace=True)
    # file = '../_database/pre_processing/transport/Switzerland/data/tra_aviation_fleet_lev4.pickle'
    # with open(file, 'wb') as handle:
    #    pickle.dump(dm_out_4, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # SECTION Passenger - New-vehicles by mode
    dm_new_veh = dm_tech.filter({"Variables": [cols["new"]]})
    dm_mode.append(dm_new_veh.group_all("Categories2", inplace=False), dim="Variables")

    # SECTION Passenger - Energy demand
    # Energy demand [MJ] = Transport demand [vkm] x efficiency [MJ/vkm]
    # Extract passenger transport demand vkm for road and pkm for others,
    # join and compute transport demand by technology
    dm_demand_vkm = dm_mode.filter(
        selected_cols={"Variables": ["tra_passenger_transport-demand-vkm"]}
    )
    dm_demand_vkm.sort(dim="Categories1")
    dm_tech.sort(dim="Categories1")
    idx_t = dm_tech.idx
    tmp = (
        dm_demand_vkm.array[:, :, 0, :, np.newaxis]
        * dm_tech.array[:, :, idx_t["tra_passenger_technology-share_fleet"], ...]
    )
    dm_tech.add(
        tmp, dim="Variables", col_label="tra_passenger_transport-demand-vkm", unit="km"
    )
    del tmp
    # Compute energy consumption
    dm_tech.operation(
        "tra_passenger_veh-efficiency_fleet",
        "*",
        "tra_passenger_transport-demand-vkm",
        out_col="tra_passenger_energy-demand",
        unit="MJ",
    )

    # SECTION Passenger - e-fuel and bio-fuel
    # Add e-fuel and bio-fuel to energy consumption
    dm_fuel = DM_other["fuels"].copy()
    # dm_fuel.drop(col_label='aviation', dim='Categories2')
    mapping_cat = {
        "road": ["LDV", "2W", "rail", "metrotram", "bus"],
        "aviation": ["aviation"],
    }
    dm_energy = dm_tech.filter({"Variables": ["tra_passenger_energy-demand"]})
    utils.add_biofuel_efuel(dm_energy, dm_fuel, mapping_cat)
    # Adjust biofuel demand based on availability
    dm_avail_fuel = DM_other["fuel-availability"]
    overcapacity_biofuel_aviation = np.maximum(
        0,
        dm_energy[:, :, "tra_passenger_energy-demand", "aviation", "kerosenebio"]
        - dm_avail_fuel[
            :, :, "tra_passenger_available-fuel-mix", "biofuel", "aviation"
        ],
    )

    dm_energy[
        :, :, "tra_passenger_energy-demand", "aviation", "kerosenebio"
    ] -= overcapacity_biofuel_aviation
    overcapacity_efuel_aviation = np.maximum(
        0,
        dm_energy[:, :, "tra_passenger_energy-demand", "aviation", "keroseneefuel"]
        - dm_avail_fuel[:, :, "tra_passenger_available-fuel-mix", "efuel", "aviation"],
    )

    dm_energy[
        :, :, "tra_passenger_energy-demand", "aviation", "keroseneefuel"
    ] -= overcapacity_efuel_aviation
    dm_energy[:, :, "tra_passenger_energy-demand", "aviation", "kerosene"] += (
        overcapacity_efuel_aviation + overcapacity_biofuel_aviation
    )

    # SECTION Passenger - GHG Emissions fossil
    # Compute emissions by fuel for fossil fuels, mode, GHG
    cdm_const.drop(col_label=["marinefueloil"], dim="Categories2")
    dm_energy_fossil = dm_energy.copy()
    dm_energy_fossil.groupby(
        {"SAF": ["kerosenebio", "keroseneefuel"]}, dim="Categories2", inplace=True
    )
    dm_energy_fossil.filter(
        {"Categories2": cdm_const.col_labels["Categories2"]}, inplace=True
    )
    dm_energy_fossil.sort("Categories2")
    cdm_const.sort("Categories2")
    arr_emis_mode_GHG_fuel = (
        dm_energy_fossil.array[:, :, :, :, np.newaxis, :]
        * cdm_const.array[np.newaxis, np.newaxis, :, np.newaxis, :, :]
    )
    arr_emis_mode_GHG_fuel = np.moveaxis(
        arr_emis_mode_GHG_fuel, -2, -1
    )  # Move GHG to end
    dm_emissions = DataMatrix.based_on(
        arr_emis_mode_GHG_fuel,
        dm_energy_fossil,
        change={
            "Variables": ["tra_passenger_emissions"],
            "Categories3": cdm_const.col_labels["Categories1"],
        },
        units={"tra_passenger_emissions": "g"},
    )

    # SECTION Passenger - GHG Emissions EV
    # Compute emissions from electricity
    dm_energy_EV = dm_energy.filter({"Categories2": ["BEV", "CEV"]})
    dm_fact = DM_other["electricity-emissions"]
    idx_f = dm_fact.idx
    idx_e = dm_energy_EV.idx
    arr = (
        dm_energy_EV.array[:, :, idx_e["tra_passenger_energy-demand"], :, :, np.newaxis]
        * dm_fact.array[
            :,
            :,
            idx_f["tra_emission-factor"],
            np.newaxis,
            np.newaxis,
            :,
            idx_f["electricity"],
        ]
    )
    dm_emissions_elec = DataMatrix.based_on(
        arr[:, :, np.newaxis, ...],
        dm_energy_EV,
        change={
            "Variables": ["tra_passenger_emissions"],
            "Categories3": dm_fact.col_labels["Categories1"],
        },
        units={"tra_passenger_emissions": "g"},
    )
    dm_emissions.append(dm_emissions_elec, dim="Categories2")
    dm_emissions.change_unit("tra_passenger_emissions", 1e-12, "g", "Mt")

    # SECTION Prepare output
    dm_emissions_by_mode = dm_emissions.group_all("Categories2", inplace=False)
    dm_emissions_by_fuel = dm_emissions.group_all("Categories1", inplace=False)
    dm_emissions_by_fuel.groupby(
        {
            "diesel": ".*diesel",
            "gasoline": ".*gasoline",
            "gas": ".*gas",
            "electricity": ".*EV",
        },
        dim="Categories1",
        regex=True,
        inplace=True,
    )
    dm_emissions_by_GHG = dm_emissions.group_all("Categories2", inplace=False)
    dm_emissions_by_GHG.group_all("Categories1")

    dm_energy.change_unit(
        "tra_passenger_energy-demand", 2.77778e-10, old_unit="MJ", new_unit="TWh"
    )

    # Deal with PHEV and electricity. For each mode of transport,
    # sum PHEV energy demand and multiply it by 0.1 to obtain a new category, the PHEV_elec
    dm_energy_phev = dm_energy.filter_w_regex(
        {"Variables": "tra_passenger_energy-demand", "Categories2": "PHEV.*"}
    )
    PHEV_elec = 0.1 * np.nansum(dm_energy_phev.array, axis=-1)
    dm_energy.add(PHEV_elec, dim="Categories2", col_label="PHEV-elec")

    # Rename and group fuel types
    dm_energy.groupby(
        {
            "biodiesel": ".*dieselbio",
            "biogas": ".*gasbio",
            "biogasoline": ".*gasolinebio",
            "efuel": ".*efuel",
        },
        dim="Categories2",
        regex=True,
        inplace=True,
    )
    dm_energy.groupby(
        {
            "diesel": ".*-diesel",
            "gasoline": ".*-gasoline",
            "hydrogen": "FCEV|H2",
            "gas": ".*-gas",
            "electricity": "BEV|CEV|PHEV-elec|mt",
        },
        dim="Categories2",
        regex=True,
        inplace=True,
    )

    # Compute energy demand by mode
    idx = dm_energy.idx
    tmp = np.nansum(
        dm_energy.array[:, :, idx["tra_passenger_energy-demand"], :, :], axis=-1
    )
    dm_mode.add(
        tmp,
        dim="Variables",
        col_label="tra_passenger_energy-demand-by-mode",
        unit="TWh",
    )

    # Prepare output for energy
    dm_electricity = dm_energy.filter({"Categories2": ["electricity"]})
    dm_electricity.group_all(dim="Categories2")
    dm_electricity.groupby(
        {"road": ["2W", "bus", "LDV"], "rail": ["metrotram", "rail"]},
        dim="Categories1",
        inplace=True,
    )
    dm_electricity.rename_col(
        "tra_passenger_energy-demand", "tra_power-demand", dim="Variables"
    )

    # Extract aviation
    dm_energy_aviation = dm_energy.filter({"Categories1": ["aviation"]})
    dm_emissions_aviation = dm_emissions.filter(
        {
            "Categories1": ["aviation"],
            "Categories3": ["CO2"],
            "Categories2": ["SAF", "BEV", "H2", "kerosene"],
        }
    )

    # Drop mode split in energy
    dm_energy.drop(col_label="aviation", dim="Categories1")
    dm_energy.group_all("Categories1")

    # Prepare efuel output
    dm_efuel = dm_energy.filter({"Categories1": ["efuel"]})
    dm_efuel.rename_col(
        "tra_passenger_energy-demand", "tra_power-demand", dim="Variables"
    )

    dm_electricity.append(dm_efuel, dim="Categories1")
    dm_electricity.change_unit("tra_power-demand", 1e3, old_unit="TWh", new_unit="GWh")

    # Energy output for EnergyScope
    arr_pkm = dm_mode[:, :, 'tra_passenger_occupancy', :, np.newaxis] \
              * dm_tech[:, :, 'tra_passenger_transport-demand-vkm', :, :]
    dm_tech.add(arr_pkm, dim='Variables', col_label='tra_passenger_transport-demand', unit='pkm')
    dm_tech.operation('tra_passenger_energy-demand', '/', 'tra_passenger_transport-demand',
                      out_col='tra_passenger_energy-intensity', unit='MJ/pkm')
    dm_tech.change_unit('tra_passenger_energy-intensity', factor=2.77778e-1, old_unit='MJ/pkm', new_unit='kWh/pkm')
    dm_energyscope = dm_tech.filter({'Variables':  ['tra_passenger_transport-demand','tra_passenger_energy-intensity']})

    DM_passenger_out = {
        'power': dm_energyscope,
    }


    # Power output (tra_power_demand _ hydrogen)
    dm_pow_hydrogen = dm_energy.filter({"Categories1": ["hydrogen"]})
    dm_pow_hydrogen.rename_col(
        "tra_passenger_energy-demand", "tra_power-demand", dim="Variables"
    )
    dm_pow_hydrogen.change_unit(
        "tra_power-demand", factor=1e3, old_unit="TWh", new_unit="GWh"
    )

    DM_passenger_out["oil-refinery"] = dm_energy.filter(
        {"Categories1": ["gasoline", "diesel", "gas"]}
    )

    # (tra_power_demand _ hydrogen)
    dm_biogas = dm_energy.filter({"Categories1": ["biogas"]})

    dm_tech.rename_col(
        "tra_passenger_technology-share_fleet",
        "tra_passenger_technology-share-fleet",
        dim="Variables",
    )

    # Compute passenger demand by mode
    # dm_mode.append(dm_demand_by_mode, dim='Variables')
    dm_mode.rename_col(
        ["tra_passenger_transport-demand"],
        ["tra_passenger_transport-demand-by-mode"],
        dim="Variables",
    )

    # Add passenger demand vkm to output
    # dm_mode.append(dm_all.filter({'Variables': ['tra_passenger_transport-demand-vkm']}), dim='Variables')

    # Compute CO2 emissions by mode
    idx = dm_emissions_by_mode.idx
    tmp = dm_emissions_by_mode.array[
        :, :, idx["tra_passenger_emissions"], :, idx["CO2"]
    ]
    dm_mode.add(
        tmp, dim="Variables", col_label="tra_passenger_emissions-by-mode_CO2", unit="Mt"
    )

    dm_energy.rename_col(
        col_in="tra_passenger_energy-demand",
        col_out="tra_passenger_energy-demand-by-fuel",
        dim="Variables",
    )

    DM_passenger_out["mode"] = dm_mode
    DM_passenger_out["tech"] = dm_tech
    DM_passenger_out["fuel"] = dm_fuel
    DM_passenger_out["agriculture"] = dm_biogas
    DM_passenger_out["emissions"] = dm_emissions_by_mode
    DM_passenger_out["energy"] = dm_energy
    DM_passenger_out["soft-mobility"] = dm_demand_soft
    DM_passenger_out["aviation"] = {
        "energy": dm_energy_aviation,
        "emissions": dm_emissions_aviation,
    }

    return DM_passenger_out


def freight_fleet_energy(DM_freight, DM_other, cdm_const, years_setting):
    # FREIGHT
    dm_tkm = DM_freight["freight_demand"]
    dm_mode = DM_freight["freight_modal_split"]
    # From bn tkm to tkm by mode of transport
    tmp = dm_tkm.array[:, :, 0, np.newaxis] * 1e9 * dm_mode.array[:, :, 0, :]
    dm_mode.add(
        tmp, dim="Variables", col_label="tra_freight_transport-demand", unit="tkm"
    )

    dm_mode_road = DM_freight["freight_mode_road"]
    dm_mode_road.append(
        dm_mode.filter_w_regex(dict_dim_pattern={"Categories1": "HDV.*"}),
        dim="Variables",
    )
    dm_mode_road.operation(
        "tra_freight_transport-demand",
        "/",
        "tra_freight_load-factor",
        out_col="tra_freight_transport-demand-vkm",
        unit="vkm",
    )
    dm_mode_road.operation(
        "tra_freight_transport-demand-vkm",
        "/",
        "tra_freight_utilisation-rate",
        out_col="tra_freight_vehicle-fleet",
        unit="number",
    )
    dm_mode_road.operation(
        "tra_freight_utilisation-rate",
        "/",
        "tra_freight_lifetime",
        out_col="tra_freight_renewal-rate",
        unit="%",
    )

    dm_mode_other = DM_freight["freight_mode_other"]
    dm_mode_other.append(
        dm_mode.filter({"Categories1": ["IWW", "marine", "aviation", "rail"]}),
        dim="Variables",
    )
    dm_mode_other.operation(
        "tra_freight_transport-demand",
        "/",
        "tra_freight_tkm-by-veh",
        out_col="tra_freight_vehicle-fleet",
        unit="number",
    )

    # Compute vehicle waste and new vehicles for both road and other
    dm_other_tmp = dm_mode_other.filter_w_regex(
        dict_dim_pattern={"Variables": ".*vehicle-fleet|.*renewal-rate"}
    )
    dm_road_tmp = dm_mode_road.filter_w_regex(
        dict_dim_pattern={"Variables": ".*vehicle-fleet|.*renewal-rate"}
    )
    dm_mode_other.drop(dim="Variables", col_label=".*vehicle-fleet|.*renewal-rate")
    dm_mode_road.drop(dim="Variables", col_label=".*vehicle-fleet|.*renewal-rate")
    dm_road_tmp.append(dm_other_tmp, dim="Categories1")
    dm_mode.append(dm_road_tmp, dim="Variables")
    del dm_road_tmp, dm_other_tmp

    compute_stock(
        dm_mode,
        rr_regex="tra_freight_renewal-rate",
        tot_regex="tra_freight_vehicle-fleet",
        waste_col="tra_freight_vehicle-waste",
        new_col="tra_freight_new-vehicles",
    )

    # Compute fleet by technology type
    dm_tech = DM_freight["freight_tech"]
    idx_t = dm_tech.idx
    idx_m = dm_mode.idx
    tmp_1 = (
        dm_tech.array[:, :, idx_t["tra_freight_technology-share_fleet"], :, :]
        * dm_mode.array[:, :, idx_m["tra_freight_vehicle-fleet"], :, np.newaxis]
    )
    tmp_2 = (
        dm_tech.array[:, :, idx_t["tra_freight_technology-share_fleet"], :, :]
        * dm_mode.array[:, :, idx_m["tra_freight_vehicle-waste"], :, np.newaxis]
    )
    tmp_3 = (
        dm_tech.array[:, :, idx_t["tra_freight_technology-share_new"], :, :]
        * dm_mode.array[:, :, idx_m["tra_freight_new-vehicles"], :, np.newaxis]
    )
    dm_tech.add(
        tmp_1, col_label="tra_freight_vehicle-fleet", dim="Variables", unit="number"
    )
    dm_tech.add(
        tmp_2, col_label="tra_freight_vehicle-waste", dim="Variables", unit="number"
    )
    dm_tech.add(
        tmp_3, col_label="tra_freight_new-vehicles", dim="Variables", unit="number"
    )
    del tmp_1, tmp_2, tmp_3
    #
    cols = {
        "renewal-rate": "tra_freight_renewal-rate",
        "tot": "tra_freight_vehicle-fleet",
        "waste": "tra_freight_vehicle-waste",
        "new": "tra_freight_new-vehicles",
        "tech_tot": "tra_freight_technology-share_fleet",
        "eff_tot": "tra_freight_vehicle-efficiency_fleet",
        "eff_new": "tra_freight_vehicle-efficiency_new",
    }
    utils.compute_fts_tech_split(dm_mode, dm_tech, cols)

    # Extract freight transport demand vkm for road and tkm for others, join and compute transport demand by technology
    dm_demand_km = dm_mode_road.filter(
        selected_cols={"Variables": ["tra_freight_transport-demand-vkm"]}
    )
    dm_demand_km_other = dm_mode_other.filter(
        selected_cols={"Variables": ["tra_freight_transport-demand"]}
    )
    dm_demand_km_other.units["tra_freight_transport-demand"] = "km"
    dm_demand_km.units["tra_freight_transport-demand-vkm"] = "km"
    dm_demand_km.rename_col(
        "tra_freight_transport-demand-vkm",
        "tra_freight_transport-demand",
        dim="Variables",
    )
    dm_demand_km.append(dm_demand_km_other, dim="Categories1")
    dm_demand_km.sort(dim="Categories1")
    idx_t = dm_tech.idx
    tmp = (
        dm_demand_km.array[:, :, 0, :, np.newaxis]
        * dm_tech.array[:, :, idx_t["tra_freight_technology-share_fleet"], ...]
    )
    dm_tech.add(
        tmp, dim="Variables", col_label="tra_freight_transport-demand", unit="km"
    )
    del tmp, dm_demand_km, dm_demand_km_other
    # Compute energy consumption
    dm_tech.operation(
        "tra_freight_vehicle-efficiency_fleet",
        "*",
        "tra_freight_transport-demand",
        out_col="tra_freight_energy-demand",
        unit="MJ",
    )

    # Compute biofuel and efuel and extract energy as standalone dm
    dm_fuel = DM_other["fuels"]
    mapping_cat = {
        "road": ["HDVH", "HDVM", "HDVL"],
        "aviation": ["aviation"],
        "rail": ["rail"],
        "marine": ["IWW", "marine"],
    }
    dm_energy = dm_tech.filter({"Variables": ["tra_freight_energy-demand"]})
    utils.add_biofuel_efuel(dm_energy, dm_fuel, mapping_cat)

    # Deal with PHEV and electricity. For each mode of transport,
    # sum PHEV energy demand and multiply it by 0.1 to obtain a new category, the PHEV_elec
    dm_energy_phev = dm_energy.filter_w_regex(
        {"Variables": "tra_freight_energy-demand", "Categories2": "PHEV.*"}
    )
    PHEV_elec = 0.1 * np.nansum(dm_energy_phev.array, axis=-1)
    dm_energy.add(PHEV_elec, dim="Categories2", col_label="PHEV-elec")

    dm_energy.array = dm_energy.array * 2.77778e-10
    dm_energy.units["tra_freight_energy-demand"] = "TWh"

    dm_energy_by_mode = dm_energy.group_all(dim="Categories2", inplace=False)
    dm_mode.append(dm_energy_by_mode, dim="Variables")

    # Prepare output for energy
    dm_electricity = dm_energy.groupby(
        {"power-demand": ["BEV", "CEV", "PHEV-elec"]}, dim="Categories2"
    )
    dm_electricity.groupby(
        {
            "road": ["HDVH", "HDVL", "HDVM"],
            "rail": ["rail"],
            "other": ["aviation", "marine", "IWW"],
        },
        dim="Categories1",
        inplace=True,
    )
    dm_electricity.switch_categories_order()
    dm_electricity.rename_col("tra_freight_energy-demand", "tra", dim="Variables")
    dm_electricity = dm_electricity.flatten()
    dm_electricity = dm_electricity.flatten()
    dm_electricity.deepen()

    dm_efuel = dm_energy.groupby({"efuel": ".*efuel"}, dim="Categories2", regex=True)
    dm_efuel.groupby(
        {"power-demand": ".*"}, dim="Categories1", inplace=True, regex=True
    )
    dm_efuel.rename_col("tra_freight_energy-demand", "tra", dim="Variables")
    dm_efuel = dm_efuel.flatten()
    dm_efuel = dm_efuel.flatten()
    dm_efuel.deepen()

    dm_electricity.append(dm_efuel, dim="Categories1")
    dm_electricity.change_unit("tra_power-demand", 1e3, old_unit="TWh", new_unit="GWh")

    # Energy output for EnergyScope
    dm_demand_tkm = dm_mode_other.filter({'Variables': ['tra_freight_transport-demand']})
    dm_demand_tkm.append(dm_mode_road.filter({'Variables': ['tra_freight_transport-demand']}), dim='Categories1')
    dm_demand_tkm.sort('Categories1')
        #.add(dm_mode_road, dim='Variables')
    arr_pkm = dm_demand_tkm[:, :, 'tra_freight_transport-demand', :, np.newaxis] \
              * dm_tech[:, :, 'tra_freight_technology-share_fleet', :, :]
    dm_tech.add(arr_pkm, dim='Variables', col_label='tra_freight_transport-demand-tkm', unit='tkm')
    dm_tech.operation('tra_freight_energy-demand', '/', 'tra_freight_transport-demand-tkm',
                      out_col='tra_freight_energy-intensity', unit='MJ/tkm')
    dm_tech.change_unit('tra_freight_energy-intensity', factor=2.77778e-1, old_unit='MJ/tkm', new_unit='kWh/tkm')
    dm_energyscope = dm_tech.filter({'Variables': ['tra_freight_transport-demand-tkm', 'tra_freight_energy-intensity']})

    DM_freight_out = {
        'power': dm_energyscope,
    }

    ## end
    all_modes = dm_energy.col_labels["Categories1"].copy()
    dm_energy_aviation = dm_energy.filter({"Categories1": ["aviation"]})
    dm_energy_marine = dm_energy.filter({"Categories1": ["marine", "IWW"]})
    dm_energy.drop(dim="Categories1", col_label=["marine", "IWW", "aviation"])

    # Rename and group fuel types
    dm_energy.groupby(
        {
            "biodiesel": ".*dieselbio",
            "biogas": ".*gasbio",
            "biogasoline": ".*gasolinebio",
            "efuel": ".*efuel",
        },
        dim="Categories2",
        regex=True,
        inplace=True,
    )
    dm_energy.groupby(
        {
            "diesel": ".*-diesel",
            "gasoline": ".*-gasoline",
            "hydrogen": "FCEV|H2",
            "gas": ".*-gas",
            "electricity": "BEV|CEV|PHEV-elec|mt",
        },
        dim="Categories2",
        regex=True,
        inplace=True,
    )

    dm_energy_aviation = dm_energy_aviation.groupby(
        {"ejetfuel": ["ICEefuel"], "biojetfuel": ["ICEbio"], "kerosene": ["ICE"]},
        inplace=False,
        dim="Categories2",
    )

    dm_energy_marine.rename_col(
        ["ICEbio", "ICEefuel", "ICE"],
        ["biomarinefueloil", "emarinefueloil", "marinefueloil"],
        dim="Categories2",
    )
    dm_energy_marine.filter(
        {"Categories2": ["biomarinefueloil", "emarinefueloil", "marinefueloil"]},
        inplace=True,
    )

    # Merge
    missing_cat_aviation = list(
        set(all_modes) - set(dm_energy_aviation.col_labels["Categories1"])
    )
    dm_energy_aviation.add(
        np.nan, dummy=True, dim="Categories1", col_label=missing_cat_aviation
    )
    missing_cat_marine = list(
        set(all_modes) - set(dm_energy_marine.col_labels["Categories1"])
    )
    dm_energy_marine.add(
        np.nan, dummy=True, dim="Categories1", col_label=missing_cat_marine
    )
    missing_cat_road = list(set(all_modes) - set(dm_energy.col_labels["Categories1"]))
    dm_energy.add(np.nan, dummy=True, dim="Categories1", col_label=missing_cat_road)

    dm_energy.append(dm_energy_marine, dim='Categories2')
    dm_energy.append(dm_energy_aviation, dim='Categories2')

    # Group together by fuel type, drop mode
    dm_total_energy = dm_energy.group_all('Categories1', inplace=False)
    dm_total_energy.rename_col('tra_freight_energy-demand', 'tra_freight_total-energy', dim='Variables')

    # Output to power:
    dm_pow_hydrogen = dm_total_energy.filter({"Categories1": ["hydrogen"]})
    dm_pow_hydrogen.rename_col(
        "tra_freight_total-energy", "tra_power-demand", dim="Variables"
    )
    dm_pow_hydrogen.change_unit("tra_power-demand", 1e3, old_unit="TWh", new_unit="GWh")

    # Prepare output to refinery:
    DM_freight_out["oil-refinery"] = dm_total_energy.filter(
        {"Categories1": ["gasoline", "diesel", "marinefueloil", "gas", "kerosene"]}
    )

    dm_biogas = dm_total_energy.filter({"Categories1": ["biogas"]})

    # Compute emission by fuel
    # Filter fuels for which we have emissions
    cdm_const.drop("Categories2", ["PHEV-diesel", "PHEV-gasoline"])
    cdm_const.rename_col("ICE-diesel", "diesel", dim="Categories2")
    cdm_const.rename_col("ICE-gasoline", "gasoline", dim="Categories2")
    cdm_const.rename_col("ICE-gas", "gas", dim="Categories2")
    cdm_const.rename_col("H2", "hydrogen", dim="Categories2")
    cdm_const.drop(col_label="SAF", dim="Categories2")

    dm_energy_em = dm_total_energy.filter(
        {"Categories1": cdm_const.col_labels["Categories2"]}
    )
    # Sort categories to make sure they match
    dm_energy_em.sort(dim="Categories1")
    cdm_const.sort(dim="Categories2")
    idx_e = dm_energy_em.idx
    idx_c = cdm_const.idx
    # emissions = energy * emission-factor
    tmp = (
        dm_energy_em.array[:, :, idx_e["tra_freight_total-energy"], np.newaxis, :]
        * cdm_const.array[np.newaxis, np.newaxis, idx_c["cp_tra_emission-factor"], :, :]
    )
    tmp = np.moveaxis(tmp, -2, -1)
    # Save emissions by fuel in a datamatrix
    col_labels = dm_energy_em.col_labels.copy()
    col_labels["Variables"] = ["tra_freight_emissions"]
    col_labels["Categories2"] = cdm_const.col_labels[
        "Categories1"
    ].copy()  # GHG category
    unit = {"tra_freight_emissions": "Mt"}
    dm_emissions_by_fuel = DataMatrix(col_labels=col_labels, units=unit)
    dm_emissions_by_fuel.array = tmp[
        :, :, np.newaxis, :, :
    ]  # The variable dimension was lost when doing nansum
    del dm_energy_em, tmp, col_labels, unit

    # Compute emissions by mode
    dm_energy_em = dm_energy.filter({'Categories2': cdm_const.col_labels['Categories2']})
    dm_energy_em.sort(dim='Categories2')
    cdm_const.sort(dim='Categories2')
    idx_e = dm_energy_em.idx
    idx_c = cdm_const.idx
    tmp = dm_energy_em.array[:, :, idx_e['tra_freight_energy-demand'], :, np.newaxis, :] \
          * cdm_const.array[np.newaxis, np.newaxis, idx_c['cp_tra_emission-factor'], np.newaxis, :, :]
    tmp = np.nansum(tmp, axis=-1)  # Remove split by fuel
    tmp = tmp[:, :, np.newaxis, :, :]
    dm_emissions_by_mode = DataMatrix.based_on(tmp, format=dm_energy_em,
                                               change={'Variables': ['tra_freight_emissions'],
                                                       'Categories2': cdm_const.col_labels['Categories1']},
                                               units={'tra_freight_emissions': 'Mt'})

    del tmp, idx_e, idx_c, dm_energy_em

    tmp = np.nansum(dm_emissions_by_mode.array, axis=-2)
    col_labels = dm_emissions_by_mode.col_labels.copy()
    col_labels["Categories1"] = col_labels["Categories2"].copy()
    col_labels.pop("Categories2")
    unit = dm_emissions_by_mode.units
    dm_emissions_by_GHG = DataMatrix(col_labels=col_labels, units=unit)
    dm_emissions_by_GHG.array = tmp[:, :, np.newaxis, :]
    del tmp, unit, col_labels

    dm_tech.rename_col('tra_freight_technology-share_fleet', 'tra_freight_techology-share-fleet', dim='Variables')

    DM_freight_out['mode'] = dm_mode
    DM_freight_out['tech'] = dm_tech
    DM_freight_out['energy'] = dm_total_energy
    DM_freight_out['agriculture'] = dm_biogas
    DM_freight_out['emissions'] = dm_emissions_by_mode

    return DM_freight_out


# !FIXME: infrastructure dummy not OK, find real tot infrastructure data and real renewal-rates or new-infrastructure
def dummy_tra_infrastructure_workflow(dm_pop):

    # Industry and Minerals need the new infrastructure in km for rails, roads, and trolley-cables
    # In order to compute the new infrastructure we need the tot infrastructure and a renewal-rate
    # tot_infrastructure = Looking at Swiss data it looks like there are around 10 m of road per capita
    # (Longueurs des routes nationales, cantonales et des autres routes ouvertes aux véhicules à moteur selon le canton)
    # and 0.6 m of rail per capita and 0.0017 of trolley-bus, I'm using this approximation for all countries
    # for the renewal rate eucalc was using 5%, which correspond to a resurfacing every 20 years. I use this for road
    # for rails I use 2.5% (40 years lifetime). For the wires I have no idea,
    # I'm going with 25 that seem to be the rewiring span of electrical cables (rr = 4%)
    # I'm using the stock function to compute the new km and the 'waste' km

    ay_infra_road = dm_pop.array * 10 / 1000  # road infrastructure in km
    ay_infra_rail = dm_pop.array * 0.6 / 1000  # rail infrastructure in km
    ay_infra_trolleybus = dm_pop.array * 0.0017 / 1000  # rail infrastructure in km

    ay_tot = np.concatenate(
        (ay_infra_rail, ay_infra_road, ay_infra_trolleybus), axis=-1
    )

    dm_infra = DataMatrix.based_on(
        ay_tot[:, :, np.newaxis, :],
        format=dm_pop,
        change={
            "Variables": ["tra_tot-infrastructure"],
            "Categories1": ["infra-rail", "infra-road", "infra-trolley-cables"],
        },
        units={"tra_tot-infrastructure": "km"},
    )
    # Add dummy renewal rates
    dm_infra.add(0, dummy=True, dim="Variables", col_label="tra_renewal-rate", unit="%")
    idx = dm_infra.idx
    dm_infra.array[:, :, idx["tra_renewal-rate"], idx["infra-road"]] = 0.05
    dm_infra.array[:, :, idx["tra_renewal-rate"], idx["infra-rail"]] = 0.025
    dm_infra.array[:, :, idx["tra_renewal-rate"], idx["infra-trolley-cables"]] = 0.04

    compute_stock(
        dm_infra,
        "tra_renewal-rate",
        "tra_tot-infrastructure",
        waste_col="tra_infrastructure_waste",
        new_col="tra_new_infrastructure",
    )

    return dm_infra.filter({"Variables": ["tra_new_infrastructure","tra_infrastructure_waste","tra_tot-infrastructure"]})
