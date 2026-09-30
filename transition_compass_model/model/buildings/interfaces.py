import os
import pickle

import numpy as np


def bld_power_interface(dm_appliances, dm_energy, dm_fuel, dm_light_heat):
    dm_light_heat.append(dm_appliances, dim="Variables")  # append appliances
    dm_light_heat.append(dm_fuel, dim="Variables")  # append hot-water
    dm_light_heat.deepen_twice()

    # space-cooling to separate dm
    dm_cooling = dm_light_heat.filter({"Categories2": ["space-cooling"]})
    dm_light_heat.drop(col_label="space-cooling", dim="Categories2")

    # split space-heating and heatpumps
    dm_energy.deepen_twice()
    dm_heating = dm_energy.filter({"Categories2": ["space-heating"]})
    dm_heatpumps = dm_energy.filter({"Categories2": ["heatpumps"]})

    DM_pow = {
        "appliance": dm_light_heat,
        "space-heating": dm_heating,
        "heatpump": dm_heatpumps,
        "cooling": dm_cooling,
    }
    return DM_pow


def bld_emissions_interface(
    dm_emissions_heating,
    dm_hotwater=None,
    dm_services_hotwater=None,
    write_pickle=False,
):
    # TODO: we are missing appliances emissions

    dm_out = dm_emissions_heating.groupby(
        {"CO2": dm_emissions_heating.col_labels["Categories1"]}, "Categories1"
    )
    dm_out.rename_col("bld_CO2-emissions_heating", "buildings-heating", "Variables")

    if dm_hotwater is not None:
        dm_hw = dm_hotwater.copy()
        dm_hw.group_all("Categories1", inplace=True)
        dm_out.array += dm_hw.array[..., np.newaxis]

    if dm_services_hotwater is not None:
        dm_srv = dm_services_hotwater.copy()
        dm_srv.group_all("Categories1", inplace=True)
        dm_out.array += dm_srv.array[..., np.newaxis]

    dm_out.add(0, "Categories1", ["CH4"], dummy=True)
    dm_out.add(0, "Categories1", ["N2O"], dummy=True)
    dm_out.sort("Categories1")

    # if write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../../_database/data/interface/buildings_to_emissions.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(dm_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # dm_emissions_fuel = DM_energy['heat-emissions-by-fuel'].filter({"Categories1": ["gas-ff-natural", "heat-ambient",
    #                                                                                 "heat-geothermal", "heat-solar",
    #                                                                                 "liquid-ff-heatingoil", "solid-bio",
    #                                                                                 "solid-ff-coal"]})
    # dm_emissions_fuel.rename_col('bld_CO2-emissions', 'bld_emissions-CO2', dim='Variables')

    # dm_appliances = dm_appliances.filter({"Categories1": ["non-residential"]})
    # dm_appliances.rename_col('bld_CO2-emissions_appliances', 'bld_emissions-CO2_appliances', dim='Variables')
    # # dm_appliances.rename_col('bld_CO2-emissions_appliances', 'bld_residential-emissions-CO2', dim='Variables')
    # # dm_appliances.rename_col('non-residential', 'non_appliances', dim='Categories1')
    # # dm_appliances.rename_col('residential', 'appliances', dim='Categories1')

    # dm_emissions_fuel = dm_emissions_fuel.flatten()
    # dm_appliances = dm_appliances.flatten()

    # dm_emissions_fuel.append(dm_appliances, dim='Variables')

    return dm_out


# def bld_industry_interface(DM_floor, dm_appliances, dm_pipes):
#     # Renovated wall + new floor area constructed
#     groupby_dict = {'floor-area-reno-residential': ['single-family-households', 'multi-family-households'],
#                     'floor-area-reno-non-residential': ['education', 'health', 'hotels', 'offices', 'other', 'trade']}
#     dm_reno = DM_floor['renovated-wall'].group_all(dim='Categories2', inplace=False)
#     dm_reno.groupby(groupby_dict, dim='Categories1', inplace=True, regex=False)
#     dm_reno.rename_col('bld_renovated-surface-area', 'bld_product-demand', dim='Variables')

#     groupby_dict = {'floor-area-new-residential': ['single-family-households', 'multi-family-households'],
#                     'floor-area-new-non-residential': ['education', 'health', 'hotels', 'offices', 'other', 'trade']}
#     dm_constructed = DM_floor['constructed-area']
#     dm_constructed.groupby(groupby_dict, dim='Categories1', inplace=True, regex=False)
#     dm_constructed.rename_col('bld_floor-area-constructed', 'bld_product-demand', dim='Variables')

#     dm_constructed.append(dm_reno, dim='Categories1')

#     # Pipes
#     dm_pipes.rename_col('bld_district-heating_new-pipe-need', 'bld_product-demand_new-dhg-pipe', dim='Variables')
#     dm_pipes.deepen()

#     # Appliances
#     dm_appliances.rename_col('bld_appliance-new', 'bld_product-demand', dim='Variables')
#     dm_appliances.rename_col('comp', 'computer', dim='Categories1')

#     DM_industry = {
#         'bld-pipe': dm_pipes,
#         'bld-floor': dm_constructed,
#         'bld-domapp': dm_appliances
#     }

#     return DM_industry


def bld_industry_interface(DM_floor, dm_appliances, write_pickle=False):
    dm_domapp = dm_appliances.filter(
        {
            "Categories1": [
                "dishwasher",
                "tumble-dryer",
                "freezer",
                "refrigerator",
                "washing-machine",
            ]
        }
    )
    dm_domapp.rename_col_regex("appliances", "domapp", "Variables")
    dm_domapp.rename_col("tumble-dryer", "dryer", "Categories1")
    dm_domapp.rename_col("refrigerator", "fridge", "Categories1")
    dm_domapp.rename_col("washing-machine", "wmachine", "Categories1")
    dm_domapp.sort("Categories1")

    dm_ele = dm_appliances.filter({"Categories1": ["PC", "laptop", "TV"]})
    dm_ele.groupby({"PC": ["PC", "laptop"]}, "Categories1", inplace=True)
    dm_ele.rename_col(["PC", "TV"], ["computer", "tv"], "Categories1")
    dm_ele.add(np.nan, "Categories1", "phone", dummy=True)
    dm_ele.rename_col_regex("appliances", "electronics", "Variables")

    dm_fa = DM_floor["floor-area"].copy()
    if "floor-area-nonres" in DM_floor:
        dm_nonres = DM_floor["floor-area-nonres"].copy()
        dm_nonres.groupby(
            {"non-residential": ".*"}, dim="Categories1", regex=True, inplace=True
        )
        dm_fa.append(dm_nonres, dim="Categories1")
    dm_fa.sort("Categories1")

    DM_industry = {
        "floor-area": dm_fa,
        "domapp": dm_domapp,
        "electronics": dm_ele,
    }

    # if write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../../_database/data/interface/buildings_to_industry.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(DM_industry, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return DM_industry


def bld_minerals_interface(DM_industry, write_xls):
    # Pipe
    dm_pipe = DM_industry["bld-pipe"].copy()
    dm_pipe.rename_col("bld_product-demand", "product-demand", dim="Variables")
    dm_pipe.rename_col("new-dhg-pipe", "infra-pipe", dim="Categories1")

    # Appliances
    dm_appliances = DM_industry["bld-domapp"].copy()
    dm_appliances.rename_col("bld_product-demand", "product-demand", dim="Variables")
    cols_in = [
        "dishwasher",
        "dryer",
        "freezer",
        "fridge",
        "wmachine",
        "computer",
        "phone",
        "tv",
    ]
    cols_out = [
        "dom-appliance-dishwasher",
        "dom-appliance-dryer",
        "dom-appliance-freezer",
        "dom-appliance-fridge",
        "dom-appliance-wmachine",
        "electronics-computer",
        "electronics-phone",
        "electronics-tv",
    ]
    dm_appliances.rename_col(cols_in, cols_out, dim="Categories1")
    dm_electronics = dm_appliances.filter_w_regex(
        {"Categories1": "electronics.*"}, inplace=False
    )
    dm_appliances.filter_w_regex({"Categories1": "dom-appliance.*"}, inplace=True)

    # Floor
    dm_floor = DM_industry["bld-floor"].copy()
    dm_floor.rename_col("bld_product-demand", "product-demand", dim="Variables")

    DM_minerals = {
        "bld-pipe": dm_pipe,
        "bld-floor": dm_floor,
        "bld-appliance": dm_appliances,
        "bld-electr": dm_electronics,
    }

    return DM_minerals


def bld_agriculture_interface(dm_agriculture, write_pickle=False):
    # Input: Variables=['bld_energy-demand_heating'], Categories1=['wood'], unit=TWh
    # Output: Variables=['bld_bioenergy'], Categories1=['gas-bio', 'solid-bio'], unit=TWh
    # gas-bio is zero because the heating workflow tracks 'gas' (FF+bio mix) not gas-bio separately.
    dm_agriculture = dm_agriculture.copy()
    dm_agriculture.rename_col("bld_energy-demand_heating", "bld_bioenergy", "Variables")
    dm_agriculture.rename_col("wood", "solid-bio", "Categories1")
    dm_agriculture.add(0.0, dummy=True, dim="Categories1", col_label="gas-bio")
    dm_agriculture.sort("Categories1")

    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../../_database/data/interface/buildings_to_agriculture.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(dm_agriculture, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm_agriculture


def bld_energy_demand_by_use(
    DM_heating_by_use, DM_services, DM_appliances, DM_light, DM_hotwater
):
    """Buildings energy demand split by building category and end-use.

    Returns bld_energy-demand [TWh] with
      Categories1 = residential / non-residential
      Categories2 = space-heating / hot-water / appliances / lighting

    Space heating is the one end-use that is not split anywhere downstream:
    DM_energy['energy-demand-heating'] covers residential *and* services together,
    and DM_services['services_energy-consumption'] has no space-heating category.
    The split therefore comes from the energy workflow's 'power_households' and
    'power_services', which sum back exactly to bld_energy-demand_heating.
    """
    parts = []

    # -- residential --------------------------------------------------------
    dm = DM_heating_by_use["residential"].filter(
        {"Variables": ["bld_energy-demand_heating"]}
    )
    dm.group_all("Categories1", inplace=True)
    dm.rename_col(
        "bld_energy-demand_heating",
        "bld_energy-demand_residential_space-heating",
        "Variables",
    )
    parts.append(dm)

    dm = DM_hotwater["power"].filter({"Variables": ["bld_hot-water_energy-demand"]})
    dm.group_all("Categories1", inplace=True)
    dm.rename_col(
        "bld_hot-water_energy-demand",
        "bld_energy-demand_residential_hot-water",
        "Variables",
    )
    parts.append(dm)

    dm = DM_appliances.filter({"Variables": ["bld_appliances_tot-elec-demand"]})
    dm.rename_col(
        "bld_appliances_tot-elec-demand",
        "bld_energy-demand_residential_appliances",
        "Variables",
    )
    parts.append(dm)

    dm = DM_light.filter({"Variables": ["bld_residential-lighting"]})
    dm.rename_col(
        "bld_residential-lighting",
        "bld_energy-demand_residential_lighting",
        "Variables",
    )
    parts.append(dm)

    # -- non-residential ----------------------------------------------------
    dm = DM_heating_by_use["non-residential"].filter(
        {"Variables": ["bld_energy-demand_heating"]}
    )
    dm.group_all("Categories1", inplace=True)
    dm.rename_col(
        "bld_energy-demand_heating",
        "bld_energy-demand_non-residential_space-heating",
        "Variables",
    )
    parts.append(dm)

    # 'elec' is the services catch-all for non-lighting electricity, i.e. the
    # non-residential counterpart of residential appliances.
    dm_services = DM_services["services_energy-consumption"].group_all(
        "Categories2", inplace=False
    )
    for category, end_use in [
        ("hot-water", "hot-water"),
        ("elec", "appliances"),
        ("lighting", "lighting"),
    ]:
        dm = dm_services.filter({"Categories1": [category]}).flatten()
        dm.rename_col(
            "bld_services_energy-consumption_" + category,
            "bld_energy-demand_non-residential_" + end_use,
            "Variables",
        )
        parts.append(dm)

    dm_out = parts[0]
    for dm in parts[1:]:
        dm_out.append(dm, dim="Variables")
    dm_out.sort("Variables")
    dm_out.deepen_twice()

    return dm_out


def bld_TPE_interface(
    DM_energy,
    DM_area,
    DM_services,
    DM_appliances,
    DM_light,
    DM_hotwater,
    DM_heating_by_use,
):
    dm_tpe = DM_energy["energy-emissions-by-class"].flattest()
    dm_tpe.append(
        DM_energy["energy-demand-heating"].flattest(), dim="Variables"
    )  # both services and residential
    dm_tpe.append(DM_energy["energy-demand-cooling"].flattest(), dim="Variables")
    dm_tpe.append(DM_energy["emissions"].flattest(), dim="Variables")
    dm_tpe.append(DM_area["floor-area-cumulated"].flattest(), dim="Variables")
    dm_tpe.append(DM_area["floor-area-cat"].flattest(), dim="Variables")
    dm_tpe.append(DM_area["floor-area-bld-type"].flattest(), dim="Variables")

    # Hot water residential
    dm_hw = DM_hotwater["power"].filter({"Variables": ["bld_hot-water_energy-demand"]})
    dm_tpe.append(dm_hw.flattest(), dim="Variables")
    dm_tpe.append(DM_hotwater["hotwater_emissions"].flattest(), dim="Variables")

    # Lighting residential
    dm_tpe.append(DM_light.flattest(), dim="Variables")

    # Appliances residential
    dm_tpe.append(DM_appliances, dim="Variables")

    # Non-residential
    dm_nonres_fuels = DM_services["services_energy-consumption"].group_all(
        "Categories1", inplace=False
    )
    dm_nonres_type = DM_services["services_energy-consumption"].group_all(
        "Categories2", inplace=False
    )
    dm_tpe.append(dm_nonres_type.flattest(), dim="Variables")
    dm_tpe.append(dm_nonres_fuels.flattest(), dim="Variables")
    dm_tpe.append(DM_services["services_emissions"].flattest(), dim="Variables")
    if "services_floor-area" in DM_services:
        dm_srv_floor_tpe = DM_services["services_floor-area"].copy()
        dm_srv_floor_tpe.change_unit(
            "bld_floor-area_services", old_unit="m2", new_unit="Mm2", factor=1e-6
        )
        dm_tpe.append(dm_srv_floor_tpe.flattest(), dim="Variables")

    KPI = []
    yr = 2050

    # Emissions global
    dm_emission_global = DM_services["services_emissions"].copy()
    dm_emission_global.append(DM_hotwater["hotwater_emissions"], dim="Variables")
    dm_energy_emissions_scope1 = DM_energy["emissions"].filter(
        {
            "Categories1": [
                "coal",
                "district-heating",
                "gas",
                "heating-oil",
                "solar",
                "wood",
            ]
        }
    )  # only keep scope 1 emissions emetter
    # Add zero entries for electricity/heat-pump so dimensions match hotwater emissions
    for _fuel in ["electricity", "heat-pump"]:
        if _fuel not in dm_energy_emissions_scope1.col_labels["Categories1"]:
            dm_energy_emissions_scope1.add(
                0, dummy=True, dim="Categories1", col_label=_fuel
            )
    dm_energy_emissions_scope1.sort("Categories1")
    dm_emission_global.append(dm_energy_emissions_scope1, dim="Variables")
    dm_emission_global.groupby(
        {
            "Variables": [
                "services_CO2-emissions_heating",
                "bld_hotwater_CO2-emissions",
                "bld_CO2-emissions_heating",
            ]
        },
        dim="Variables",
        inplace=True,
    )
    dm_emission_global.rename_col("Variables", "bld_CO2-emissions", "Variables")
    dm_tpe.append(dm_emission_global.flattest(), dim="Variables")
    # dm_emission_global.change_unit("bld_CO2-emissions", factor=1e6, old_unit="Mt", new_unit="t" )

    # Buildings-wide total, published for the cross-sector (Overall) charts and read
    # back for the KPI so card and chart share one source. Named CO2e for consistency
    # with the tra_emissions-CO2e_* series; buildings model no CH4/N2O, so CO2e = CO2.
    dm_emission_global.group_all("Categories1", inplace=True)
    dm_emission_global.rename_col(
        "bld_CO2-emissions", "bld_emissions-CO2e", "Variables"
    )
    dm_tpe.append(dm_emission_global.flattest(), dim="Variables")

    value = dm_emission_global[0, yr, "bld_emissions-CO2e"]

    KPI.append({"title": "Total emissions", "value": value, "unit": "Mt"})

    # Energy demand total heating  (hotwater + space heating residential + services)
    # TODO : add a plot for the total energy demand for heating (hot water + space heating for residential + services) and the share of ambient heat and other tech in it.
    dm_energy_global = (
        DM_energy["energy-demand-heating"]
        .filter({"Variables": ["bld_energy-demand_heating"]})
        .copy()
    )
    dm_hw_copy = dm_hw.copy()
    for _cat in ["other-tech", "ambient-heat"]:
        if _cat not in dm_hw_copy.col_labels["Categories1"]:
            dm_hw_copy.add(0, "Categories1", [_cat], dummy=True)
    dm_energy_global.append(dm_hw_copy, dim="Variables")
    dmservices_flat = (
        DM_services["services_energy-consumption"]
        .filter({"Categories1": ["hot-water"]})
        .flatten()
        .flatten()
    )
    dmservices_flat.deepen()
    for _cat in ["other-tech", "ambient-heat"]:
        if _cat not in dmservices_flat.col_labels["Categories1"]:
            dmservices_flat.add(0, "Categories1", [_cat], dummy=True)
    dm_energy_global.append(dmservices_flat, dim="Variables")

    # Create Categories1 dimension for appliances with heating technologies
    dm_appliances_energy = DM_appliances.copy()
    dm_appliances_energy.deepen()
    dm_appliances_energy.rename_col("tot-elec-demand", "electricity", dim="Categories1")
    other_cats = [
        cat
        for cat in dm_energy_global.col_labels["Categories1"]
        if cat != "electricity"
    ]
    dm_appliances_energy.add(0, "Categories1", other_cats, dummy=True)
    dm_energy_global.append(dm_appliances_energy, dim="Variables")

    dm_energy_comsumption_tot = dm_energy_global.groupby(
        {
            "energy_consumption": [
                "bld_services_energy-consumption_hot-water",
                "bld_energy-demand_heating",
                "bld_hot-water_energy-demand",
                "bld_appliances",
            ]
        },
        dim="Variables",
    )

    dm_tpe.append(dm_energy_comsumption_tot.flattest(), dim="Variables")
    # dm_energy_heating = dm_energy_global.groupby(
    #         {
    #             "energy_consumption": [
    #                 "bld_services_energy-consumption_hot-water",
    #                 "bld_energy-demand_heating",
    #                 "bld_hot-water_energy-demand",
    #             ]
    #         },
    #         dim="Variables",
    #     )
    # dm_energy_heating.change_unit("energy_consumption", factor=1e6, old_unit="TWh", new_unit="MWh")

    # Energy demand by building category and end-use (residential / non-residential
    # x space-heating / hot-water / appliances / lighting), plus the total across all
    # of them. The KPI reads off the same series so chart and card cannot drift.
    dm_energy_by_use = bld_energy_demand_by_use(
        DM_heating_by_use, DM_services, DM_appliances, DM_light, DM_hotwater
    )
    dm_tpe.append(dm_energy_by_use.flattest(), dim="Variables")

    dm_energy_total = dm_energy_by_use.group_all("Categories2", inplace=False)
    dm_energy_total.group_all("Categories1", inplace=True)
    dm_energy_total.rename_col(
        "bld_energy-demand", "bld_energy-demand_total", "Variables"
    )
    dm_tpe.append(dm_energy_total.flattest(), dim="Variables")

    value = dm_energy_total[0, yr, "bld_energy-demand_total"]
    KPI.append({"title": "Total energy demand", "value": value, "unit": "TWh"})

    # Floor area stock (residential + non-residential)
    dm_tot_area = DM_area["floor-area-cumulated"].groupby(
        {"bld_tot-area": ".*"}, dim="Variables", regex=True, inplace=False
    )
    value_stock_res = dm_tot_area[0, yr, "bld_tot-area"]

    value_stock_nonres = 0.0
    if "services_floor-area" in DM_services:
        dm_srv_floor_stock = dm_srv_floor_tpe.group_all("Categories2", inplace=False)
        dm_srv_floor_stock.group_all("Categories1", inplace=True)
        value_stock_nonres = dm_srv_floor_stock[0, yr, "bld_floor-area_services"]

    value = value_stock_res + value_stock_nonres
    KPI.append({"title": "Floor Area Stock", "value": value, "unit": "Mm2"})

    return dm_tpe, KPI
