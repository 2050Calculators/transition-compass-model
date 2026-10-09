import os
import pickle

import numpy as np


def bld_emissions_interface(
    dm_emissions_heating,
    dm_hotwater=None,
    dm_services_hotwater=None,
    write_pickle=False,
):
    # TODO: we are missing appliances emissions

    # Heating emissions are split by building use (Categories1) and fuel (Categories2);
    # the emissions module only needs the per-gas total.
    dm_by_fuel = dm_emissions_heating.group_all("Categories1", inplace=False)
    dm_out = dm_by_fuel.groupby(
        {"CO2": dm_by_fuel.col_labels["Categories1"]}, "Categories1"
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
    """Buildings energy demand split by building category, end-use and fuel.

    Returns bld_energy-demand [TWh] with
      Categories1 = residential / non-residential
      Categories2 = space-heating / hot-water / appliances / lighting
      Categories3 = fuel (the space-heating carriers, incl. ambient-heat)

    Space heating is the one end-use that is not split anywhere downstream:
    DM_energy['energy-demand-heating'] covers residential *and* services together,
    and DM_services['services_energy-consumption'] has no space-heating category.
    The split therefore comes from the energy workflow's 'power_households' and
    'power_services', which sum back exactly to bld_energy-demand_heating.
    Residential appliances and lighting have no fuel split: they are all electricity.
    """
    fuels = DM_heating_by_use["residential"].col_labels["Categories1"]

    def _align_fuels(dm):
        # Not every end-use uses every carrier (e.g. ambient-heat is space-heating
        # only); pad with zeros so all parts share the same fuel dimension.
        for fuel in fuels:
            if fuel not in dm.col_labels["Categories1"]:
                dm.add(0, dim="Categories1", col_label=fuel, dummy=True)
        dm.sort("Categories1")
        return dm

    parts = []

    # -- residential --------------------------------------------------------
    dm = DM_heating_by_use["residential"].filter(
        {"Variables": ["bld_energy-demand_heating"]}
    )
    dm.rename_col(
        "bld_energy-demand_heating",
        "bld_energy-demand_residential_space-heating",
        "Variables",
    )
    parts.append(dm)

    dm = DM_hotwater["power"].filter({"Variables": ["bld_hot-water_energy-demand"]})
    dm.rename_col(
        "bld_hot-water_energy-demand",
        "bld_energy-demand_residential_hot-water",
        "Variables",
    )
    parts.append(dm)

    dm = DM_appliances.filter({"Variables": ["bld_appliances_tot-elec-demand"]})
    dm.rename_col(
        "bld_appliances_tot-elec-demand",
        "bld_energy-demand_residential_appliances_electricity",
        "Variables",
    )
    dm.deepen()
    parts.append(dm)

    dm = DM_light.filter({"Variables": ["bld_residential-lighting"]})
    dm.rename_col(
        "bld_residential-lighting",
        "bld_energy-demand_residential_lighting_electricity",
        "Variables",
    )
    dm.deepen()
    parts.append(dm)

    # -- non-residential ----------------------------------------------------
    dm = DM_heating_by_use["non-residential"].filter(
        {"Variables": ["bld_energy-demand_heating"]}
    )
    dm.rename_col(
        "bld_energy-demand_heating",
        "bld_energy-demand_non-residential_space-heating",
        "Variables",
    )
    parts.append(dm)

    # 'elec' is the services catch-all for non-lighting electricity, i.e. the
    # non-residential counterpart of residential appliances.
    dm_services = DM_services["services_energy-consumption"]
    for category, end_use in [
        ("hot-water", "hot-water"),
        ("elec", "appliances"),
        ("lighting", "lighting"),
    ]:
        dm = dm_services.filter({"Categories1": [category]})
        dm.group_all("Categories1", inplace=True)  # fuel moves to Categories1
        dm.rename_col(
            "bld_services_energy-consumption",
            "bld_energy-demand_non-residential_" + end_use,
            "Variables",
        )
        parts.append(dm)

    dm_out = _align_fuels(parts[0])
    for dm in parts[1:]:
        dm_out.append(_align_fuels(dm), dim="Variables")
    dm_out.sort("Variables")
    # Variables -> end-use (Categories2) -> building use (Categories3), then put
    # building use first: (use, end-use, fuel)
    dm_out.deepen(based_on="Variables")
    dm_out.deepen(based_on="Variables")
    dm_out.switch_categories_order("Categories1", "Categories3")

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
    dm_tpe.append(DM_area["floor-area-use-cat"].flattest(), dim="Variables")
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
            "Categories2": [
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
        if _fuel not in dm_energy_emissions_scope1.col_labels["Categories2"]:
            dm_energy_emissions_scope1.add(
                0, dummy=True, dim="Categories2", col_label=_fuel
            )
    dm_energy_emissions_scope1.sort("Categories2")

    # Restructure hot-water emissions dm
    dm_emission_global.rename_col(
        ["services_CO2-emissions_hot-water", "bld_hotwater_CO2-emissions"],
        [
            "bld_CO2-emissions_hot-water_services",
            "bld_CO2-emissions_hot-water_residential",
        ],
        dim="Variables",
    )
    dm_emission_global.deepen(based_on="Variables")
    dm_emission_global.switch_categories_order("Categories1", "Categories2")

    dm_emission_global.append(dm_energy_emissions_scope1, dim="Variables")
    dm_emission_global.deepen(based_on="Variables")

    dm_emission_enduse = dm_emission_global.group_all("Categories2", inplace=False)
    dm_emission_global_fuel = dm_emission_global.group_all("Categories3", inplace=False)
    dm_emission_global_fuel.group_all("Categories1", inplace=True)
    dm_tpe.append(dm_emission_global_fuel.flattest(), dim="Variables")
    dm_tpe.append(dm_emission_enduse.flattest(), dim="Variables")

    # Buildings-wide total, published for the cross-sector (Overall) charts and read
    # back for the KPI so card and chart share one source. Named CO2e for consistency
    # with the tra_emissions-CO2e_* series; buildings model no CH4/N2O, so CO2e = CO2.
    # Collapse building use, fuel and end-use down to a single total
    for _ in range(3):
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

    # Energy demand by building category, end-use (space-heating / hot-water /
    # appliances / lighting) and fuel, plus the total across all of them. Same
    # structure as the emissions block above: one (use, end-use, fuel) matrix,
    # collapsed by fuel for the end-use view and by end-use for the fuel views.
    # The KPI reads off the same series so chart and card cannot drift.
    dm_energy_by_use = bld_energy_demand_by_use(
        DM_heating_by_use, DM_services, DM_appliances, DM_light, DM_hotwater
    )
    dm_energy_enduse = dm_energy_by_use.group_all("Categories3", inplace=False)
    dm_tpe.append(dm_energy_enduse.flattest(), dim="Variables")
    dm_energy_use_fuel = dm_energy_by_use.group_all("Categories2", inplace=False)
    dm_tpe.append(dm_energy_use_fuel.flattest(), dim="Variables")
    dm_energy_fuel = dm_energy_use_fuel.group_all("Categories1", inplace=False)
    dm_tpe.append(dm_energy_fuel.flattest(), dim="Variables")

    dm_energy_total = dm_energy_fuel.group_all("Categories1", inplace=False)
    dm_energy_total.rename_col(
        "bld_energy-demand", "bld_energy-demand_total", "Variables"
    )
    dm_tpe.append(dm_energy_total.flattest(), dim="Variables")

    value = dm_energy_total[0, yr, "bld_energy-demand_total"]
    KPI.append({"title": "Total energy demand", "value": value, "unit": "TWh"})

    # Floor area stock (residential + non-residential), from the same series as the
    # stock charts. The building stock already covers the services building types,
    # so services_floor-area must not be added on top (it would count them twice).
    dm_tot_area = DM_area["floor-area-use-cat"].group_all("Categories2", inplace=False)
    dm_tot_area.group_all("Categories1", inplace=True)
    value = dm_tot_area[0, yr, "bld_floor-area_stock"]
    KPI.append({"title": "Floor Area Stock", "value": value, "unit": "Mm2"})

    return dm_tpe, KPI
