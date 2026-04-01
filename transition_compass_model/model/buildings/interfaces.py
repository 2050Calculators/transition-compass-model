import os
import numpy as np
import pickle
from transition_compass_model.model.common.data_matrix_class import DataMatrix


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


def bld_emissions_interface(dm_emissions_heating, write_pickle=False):

    # TODO: we are missing appliances emissions

    dm_out = dm_emissions_heating.groupby(
        {"CO2": dm_emissions_heating.col_labels["Categories1"]}, "Categories1"
    )
    dm_out.rename_col("bld_CO2-emissions_heating", "buildings-heating", "Variables")
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


def bld_industry_interface(DM_floor, dm_appliances):

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

    DM_industry = {
        "floor-area": DM_floor["floor-area"].copy(),
        "domapp": dm_domapp,
        "electronics": dm_ele,
    }

    # {'floor-area': DataMatrix with shape (1, 40, 4, 1), variables ['bld_floor-area_stock', 'bld_floor-area_waste', 'bld_floor-area_renovated', 'bld_floor-area_new'] and categories1 ['residential'], 'domapp': DataMatrix with shape (1, 40, 3, 5), variables ['bld_domapp_stock', 'bld_domapp_waste', 'bld_domapp_new'] and categories1 ['dishwasher', 'dryer', 'freezer', 'fridge', 'wmachine'], 'electronics': DataMatrix with shape (1, 40, 3, 3), variables ['bld_electronics_stock', 'bld_electronics_waste', 'bld_electronics_new'] and categories1 ['computer', 'phone', 'tv']}

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


def bld_agriculture_interface(dm_agriculture):
    dm_agriculture.filter({"Categories2": ["gas-bio", "solid-bio"]}, inplace=True)
    dm_agriculture.group_all("Categories1")
    dm_agriculture.rename_col(
        "bld_space-heating-energy-demand", "bld_bioenergy", "Variables"
    )
    dm_agriculture.change_unit(
        "bld_bioenergy", factor=1e-3, old_unit="GWh", new_unit="TWh"
    )

    return dm_agriculture


def bld_TPE_interface(
    DM_energy, DM_area, DM_services, DM_appliances, DM_light, DM_hotwater
):

    dm_tpe = DM_energy["energy-emissions-by-class"].flattest()
    dm_tpe.append(DM_energy["energy-demand-heating"].flattest(), dim="Variables")
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

    dm_emission_global.group_all("Categories1", inplace=True)
    value = dm_emission_global[0, yr, "bld_CO2-emissions"]

    KPI.append({"title": "CO2 emissions", "value": value, "unit": "Mt"})

    # Energy demand in TWh
    dm_tot_enr = DM_energy["energy-demand-heating"].filter(
        {"Variables": ["bld_energy-demand_heating"]}
    )
    dm_tot_enr.drop("Categories1", ["solar", "ambient-heat"])
    dm_tot_enr.group_all("Categories1", inplace=True)
    value = dm_tot_enr[0, yr, "bld_energy-demand_heating"]
    KPI.append(
        {"title": "Energy Demand for Space Heating", "value": value, "unit": "TWh"}
    )

    # A-C buildings buildings %
    dm_area = DM_area["floor-area-cat"].normalise("Categories1", inplace=False)
    value = (
        dm_area[0, yr, "bld_floor-area_stock_share", "B"]
        + dm_area[0, yr, "bld_floor-area_stock_share", "C"]
    ) * 100
    KPI.append({"title": "A-C class", "value": value, "unit": "%"})

    # Unrenovated buildings
    dm_tot_area = DM_area["floor-area-cumulated"].groupby(
        {"bld_tot-area": ".*"}, dim="Variables", regex=True, inplace=False
    )
    value = (
        DM_area["floor-area-cumulated"][0, yr, "bld_floor-area_unrenovated-cumulated"]
        / dm_tot_area[0, yr, "bld_tot-area"]
        * 100
    )
    KPI.append({"title": "Unrenovated Envelope Share", "value": value, "unit": "%"})

    return dm_tpe, KPI
