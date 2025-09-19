import os
from model.common.data_matrix_class import DataMatrix

def bld_power_interface(dm_appliances, dm_energy, dm_fuel, dm_light_heat):
    dm_light_heat.append(dm_appliances, dim='Variables')  # append appliances
    dm_light_heat.append(dm_fuel, dim='Variables')  # append hot-water
    dm_light_heat.deepen_twice()

    # space-cooling to separate dm
    dm_cooling = dm_light_heat.filter({'Categories2': ['space-cooling']})
    dm_light_heat.drop(col_label='space-cooling', dim='Categories2')

    # split space-heating and heatpumps
    dm_energy.deepen_twice()
    dm_heating = dm_energy.filter({'Categories2': ['space-heating']})
    dm_heatpumps = dm_energy.filter({'Categories2': ['heatpumps']})

    DM_pow = {
        'appliance': dm_light_heat,
        'space-heating': dm_heating,
        'heatpump': dm_heatpumps,
        'cooling': dm_cooling
    }
    return DM_pow


def bld_emissions_interface(dm_appliances, DM_energy):
    dm_emissions_fuel = DM_energy['heat-emissions-by-fuel'].filter({"Categories1": ["gas-ff-natural", "heat-ambient",
                                                                                    "heat-geothermal", "heat-solar",
                                                                                    "liquid-ff-heatingoil", "solid-bio",
                                                                                    "solid-ff-coal"]})
    dm_emissions_fuel.rename_col('bld_CO2-emissions', 'bld_emissions-CO2', dim='Variables')

    dm_appliances = dm_appliances.filter({"Categories1": ["non-residential"]})
    dm_appliances.rename_col('bld_CO2-emissions_appliances', 'bld_emissions-CO2_appliances', dim='Variables')
    # dm_appliances.rename_col('bld_CO2-emissions_appliances', 'bld_residential-emissions-CO2', dim='Variables')
    # dm_appliances.rename_col('non-residential', 'non_appliances', dim='Categories1')
    # dm_appliances.rename_col('residential', 'appliances', dim='Categories1')

    dm_emissions_fuel = dm_emissions_fuel.flatten()
    dm_appliances = dm_appliances.flatten()

    dm_emissions_fuel.append(dm_appliances, dim='Variables')

    return dm_emissions_fuel


def bld_industry_interface(DM_floor, dm_appliances, dm_pipes):
    # Renovated wall + new floor area constructed
    groupby_dict = {'floor-area-reno-residential': ['single-family-households', 'multi-family-households'],
                    'floor-area-reno-non-residential': ['education', 'health', 'hotels', 'offices', 'other', 'trade']}
    dm_reno = DM_floor['renovated-wall'].group_all(dim='Categories2', inplace=False)
    dm_reno.groupby(groupby_dict, dim='Categories1', inplace=True, regex=False)
    dm_reno.rename_col('bld_renovated-surface-area', 'bld_product-demand', dim='Variables')

    groupby_dict = {'floor-area-new-residential': ['single-family-households', 'multi-family-households'],
                    'floor-area-new-non-residential': ['education', 'health', 'hotels', 'offices', 'other', 'trade']}
    dm_constructed = DM_floor['constructed-area']
    dm_constructed.groupby(groupby_dict, dim='Categories1', inplace=True, regex=False)
    dm_constructed.rename_col('bld_floor-area-constructed', 'bld_product-demand', dim='Variables')

    dm_constructed.append(dm_reno, dim='Categories1')

    # Pipes
    dm_pipes.rename_col('bld_district-heating_new-pipe-need', 'bld_product-demand_new-dhg-pipe', dim='Variables')
    dm_pipes.deepen()

    # Appliances
    dm_appliances.rename_col('bld_appliance-new', 'bld_product-demand', dim='Variables')
    dm_appliances.rename_col('comp', 'computer', dim='Categories1')

    DM_industry = {
        'bld-pipe': dm_pipes,
        'bld-floor': dm_constructed,
        'bld-domapp': dm_appliances
    }

    return DM_industry


def bld_minerals_interface(DM_industry, write_xls):
    # Pipe
    dm_pipe = DM_industry['bld-pipe'].copy()
    dm_pipe.rename_col('bld_product-demand', 'product-demand', dim='Variables')
    dm_pipe.rename_col('new-dhg-pipe', 'infra-pipe', dim='Categories1')

    # Appliances
    dm_appliances = DM_industry['bld-domapp'].copy()
    dm_appliances.rename_col('bld_product-demand', 'product-demand', dim='Variables')
    cols_in = ['dishwasher', 'dryer', 'freezer', 'fridge', 'wmachine', 'computer', 'phone', 'tv']
    cols_out = ['dom-appliance-dishwasher', 'dom-appliance-dryer', 'dom-appliance-freezer', 'dom-appliance-fridge',
                'dom-appliance-wmachine', 'electronics-computer', 'electronics-phone', 'electronics-tv']
    dm_appliances.rename_col(cols_in, cols_out, dim='Categories1')
    dm_electronics = dm_appliances.filter_w_regex({'Categories1': 'electronics.*'}, inplace=False)
    dm_appliances.filter_w_regex({'Categories1': 'dom-appliance.*'}, inplace=True)

    # Floor
    dm_floor = DM_industry['bld-floor'].copy()
    dm_floor.rename_col('bld_product-demand', 'product-demand', dim='Variables')

    DM_minerals = {
        'bld-pipe': dm_pipe,
        'bld-floor': dm_floor,
        'bld-appliance': dm_appliances,
        'bld-electr': dm_electronics
    }

    return DM_minerals


def bld_agriculture_interface(dm_agriculture):
    dm_agriculture.filter({'Categories2': ['gas-bio', 'solid-bio']}, inplace=True)
    dm_agriculture.group_all('Categories1')
    dm_agriculture.rename_col('bld_space-heating-energy-demand', 'bld_bioenergy', 'Variables')
    dm_agriculture.change_unit('bld_bioenergy', factor=1e-3, old_unit='GWh', new_unit='TWh')

    return dm_agriculture


def bld_TPE_interface(DM_energy, DM_area):

    dm_tpe = DM_energy['energy-emissions-by-class'].flattest()
    dm_tpe.append(DM_energy['energy-demand-heating'].flattest(), dim='Variables')
    dm_tpe.append(DM_energy['energy-demand-cooling'].flattest(), dim='Variables')
    dm_tpe.append(DM_energy['emissions'].flattest(), dim='Variables')
    dm_tpe.append(DM_area['floor-area-cumulated'].flattest(), dim='Variables')
    dm_tpe.append(DM_area['floor-area-cat'].flattest(), dim='Variables')
    dm_tpe.append(DM_area['floor-area-bld-type'].flattest(), dim='Variables')

    KPI = []
    yr = 2050
    # Emissions
    dm_tot_emi = DM_energy['energy-emissions-by-class'].filter({'Variables': ['bld_CO2-emissions_heating']})
    dm_tot_emi.group_all('Categories1', inplace=True)
    value = dm_tot_emi[0, yr, 'bld_CO2-emissions_heating']
    KPI.append({'title': 'CO2 emissions', 'value': value, 'unit': 'Mt'})

    # Energy demand in TWh
    dm_tot_enr = DM_energy['energy-demand-heating'].filter({'Variables': ['bld_energy-demand_heating']})
    dm_tot_enr.drop('Categories1', ['solar', 'ambient-heat'])
    dm_tot_enr.group_all('Categories1', inplace=True)
    value = dm_tot_enr[0, yr, 'bld_energy-demand_heating']
    KPI.append({'title': 'Energy Demand for Space Heating', 'value': value, 'unit': 'TWh'})

    # A-C buildings buildings %
    dm_area = DM_area['floor-area-cat'].normalise('Categories1', inplace=False)
    value = (dm_area[0, yr, 'bld_floor-area_stock_share', 'B']
             + dm_area[0, yr, 'bld_floor-area_stock_share', 'C'] ) * 100
    KPI.append({'title': 'A-C class', 'value': value, 'unit': '%'})

    # Unrenovated buildings
    dm_tot_area = DM_area['floor-area-cumulated'].groupby({'bld_tot-area': '.*'}, dim='Variables', regex=True, inplace=False)
    value = DM_area['floor-area-cumulated'][0, yr, 'bld_floor-area_unrenovated-cumulated'] / dm_tot_area[0, yr, 'bld_tot-area']*100
    KPI.append({'title': 'Unrenovated Envelope Share', 'value': value, 'unit': '%'})

    return dm_tpe, KPI




