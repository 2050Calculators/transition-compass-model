#######################################################################################################################
# SECTION: Import Packages, Classes & Functions
#######################################################################################################################
import numpy as np
import pickle  # read/write the data in pickle
import json  # read the lever setting
import os  # operating system (e.g., look for workspace)
import pandas as pd
import warnings

# Import Class
from model.common.data_matrix_class import DataMatrix  # Class for the model inputs
from model.common.constant_data_matrix_class import ConstantDataMatrix  # Class for the constant inputs
from model.common.interface_class import Interface
# ImportFunctions
from model.common.io_database import read_database, read_database_fxa  # read functions for levers & fixed assumptions
from model.common.auxiliary_functions import read_level_data, filter_geoscale, simulate_input

warnings.simplefilter("ignore")
#######################################################################################################################
# ModelSetting - Oil Refinery
#######################################################################################################################

def database_from_csv_to_datamatrix():
    years_setting = [1990, 2015, 2050, 5]  # Set the timestep for historical years & scenarios
    startyear: int = years_setting[0]  # Start year is argument [0], i.e., 1990
    baseyear: int = years_setting[1]  # Base/Reference year is argument [1], i.e., 2015
    lastyear: int = years_setting[2]  # End/Last year is argument [2], i.e., 2050
    step_fts = years_setting[3]  # Timestep for scenario is argument [3], i.e., 5 years
    years_ots = list(
        np.linspace(start=startyear, stop=baseyear, num=(baseyear - startyear) + 1).astype(int))
    # Defines the part of dataset that is historical years
    years_fts = list(
        np.linspace(start=baseyear + step_fts, stop=lastyear, num=int((lastyear - baseyear) / step_fts)).astype(int))
    # Defines the part of dataset that is scenario
    years_all = years_ots + years_fts  # Defines all years

#######################################################################################################################
# DataFixedAssumptions - Oil refinery
#######################################################################################################################

    # Read fixed assumptions to datamatrix
    df = read_database_fxa('oil-refinery_fixed-assumptions')
    dm = DataMatrix.create_from_df(df, num_cat=0)

    # Keep only ots and fts years
    dm = dm.filter(selected_cols={'Years': years_all})

    # Dictionary
    dm_refinery_ratio = dm.filter({'Variables': ['ory_refinery_country-ratio']})

    # ToDo: check the values as 12% for France is very low, so meaning of this ratio?
    # ToDo: add calibration factors

    dict_fxa = {
        'refinery-ratio': dm_refinery_ratio
    }

#######################################################################################################################
# DataConstants - Oil refinery
#######################################################################################################################

    cdm_const = ConstantDataMatrix.extract_constant('interactions_constants', pattern='cp_refinery.*', num_cat=0)


#######################################################################################################################
# DataMatrices - Oil refinery Data Matrix
#######################################################################################################################

    DM_refinery = {
        'fxa': dict_fxa,
        'constant': cdm_const
    }

#######################################################################################################################
# DataPickle - Oil refinery
#######################################################################################################################

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, '../_database/data/datamatrix/oil-refinery.pickle')
    with open(f, 'wb') as handle:
        pickle.dump(DM_refinery, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return DM_refinery

# update_interaction_constant_from_file('interactions_constants_local') # uncomment to update constant
# database_from_csv_to_datamatrix()  # un-comment to update pickle

#######################################################################################################################
# DataSubMatrices - Oil refinery
#######################################################################################################################

#######################################################################################################################
# LocalInterfaces - Power
#######################################################################################################################
def simulate_power_to_refinery_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../_database/data/xls/"
                                             "All-Countries-interface_from-power-to-oil-refinery.xlsx")
    df = pd.read_excel(f)
    dm_power = DataMatrix.create_from_df(df, num_cat=1)
    dm_power.sort(dim='Categories1')

    return dm_power

#######################################################################################################################
# LocalInterfaces - Buildings
#######################################################################################################################
def simulate_buildings_to_refinery_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../_database/data/xls/"
                                             "All-Countries-interface_from-buildings-to-oil-refinery.xlsx")
    df = pd.read_excel(f, sheet_name="default")
    dm_buildings = DataMatrix.create_from_df(df, num_cat=1)
    dm_buildings.change_unit('bld_energy-demand', 1e-3, old_unit='GWh', new_unit='TWh')
    return dm_buildings

#######################################################################################################################
# LocalInterfaces - Transport
#######################################################################################################################
def simulate_transport_to_refinery_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../_database/data/xls/"
                                             "All-Countries-interface_from-transport-to-oil-refinery.xlsx")
    df = pd.read_excel(f, sheet_name="default")
    dm_transport = DataMatrix.create_from_df(df, num_cat=1)

    return dm_transport

#######################################################################################################################
# LocalInterfaces - Industry
#######################################################################################################################
def simulate_industry_to_refinery_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../_database/data/xls/"
                                             "All-Countries-interface_from-industry-to-oil-refinery.xlsx")
    df = pd.read_excel(f)
    dm_industry = DataMatrix.create_from_df(df, num_cat=1)

    return dm_industry

#######################################################################################################################
# LocalInterfaces - Ammonia
#######################################################################################################################
def simulate_ammonia_to_refinery_input():
    
    dm_ammonia = simulate_input("ammonia","oil-refinery",num_cat = 1)
    
    return dm_ammonia

#######################################################################################################################
# LocalInterfaces - Agriculture
#######################################################################################################################
def simulate_agriculture_to_refinery_input():
    
    dm_agriculture = simulate_input("agriculture","oil-refinery",num_cat = 1)

    return dm_agriculture

#######################################################################################################################
# CalculationTree - Module - sub flow
#######################################################################################################################
def fuel_production_workflow(DM_refinery, DM_fuel_demand):

    def check_unit(dm, unit):
        for var in dm.col_labels['Variables']:
            if dm.units[var] != unit:
                raise ValueError(f'variable {var} does not have unit {unit}')

    ######################################
    # CalculationLeafs - Energy demand per fuel [TWh]
    ######################################

    # Uniforming the matrices (energy carriers)
    dm_power = DM_fuel_demand['power']
    dm_power.add(0, dummy=True, col_label=['diesel', 'gasoline', 'kerosene'],
                 dim='Categories1', unit=['TWh', 'TWh', 'TWh'])
    check_unit(dm_power, 'TWh')
    dm_buildings = DM_fuel_demand['buildings']
    dm_buildings.add(0, dummy=True, col_label=['nuclear','diesel', 'gasoline', 'kerosene'],
                 dim='Categories1', unit=['TWh', 'TWh', 'TWh', 'TWh'])
    check_unit(dm_buildings, 'TWh')
    dm_transport = DM_fuel_demand['transport']
    dm_transport.add(0, dummy=True, col_label=['coal', 'nuclear'],
                 dim='Categories1', unit=['TWh', 'TWh'])
    check_unit(dm_transport, 'TWh')
    dm_industry = DM_fuel_demand['industry']
    dm_industry.add(0, dummy=True, col_label=['nuclear', 'gasoline', 'kerosene'],
                 dim='Categories1', unit=['TWh', 'TWh', 'TWh'])
    check_unit(dm_industry, 'TWh')
    dm_ammonia = DM_fuel_demand['ammonia']
    dm_ammonia.add(0, dummy=True, col_label=['nuclear', 'gasoline', 'kerosene'],
                    dim='Categories1', unit=['TWh', 'TWh', 'TWh'])
    check_unit(dm_ammonia, 'TWh')
    dm_agriculture = DM_fuel_demand['agriculture']
    dm_agriculture.add(0, dummy=True, col_label=['nuclear','kerosene'],
                    dim='Categories1', unit=['TWh', 'TWh'])
    check_unit(dm_agriculture, 'TWh')
    # ToDo: to remove when energy carriers will be uniform
    dm_hydrogen = dm_power.filter({'Categories1': ['hydrogen']})
    check_unit(dm_hydrogen, 'TWh')
    dm_power.rename_col('oil', 'fuel-oil', dim='Categories1')
    dm_power = dm_power.filter({'Categories1': ['coal', 'gas', 'nuclear', 'fuel-oil', 'diesel', 'gasoline', 'kerosene']})


    dm_buildings.rename_col('gas-ff-natural', 'gas', dim='Categories1')
    dm_buildings.rename_col('liquid-ff-heatingoil', 'fuel-oil', dim='Categories1')
    dm_buildings.rename_col('solid-ff-coal', 'coal', dim='Categories1')

    dm_transport.rename_col('gas-ff-natural', 'gas', dim='Categories1')
    dm_transport.rename_col('liquid-ff-fuel-oil', 'fuel-oil', dim='Categories1')
    dm_transport.rename_col('liquid-ff-kerosene', 'kerosene', dim='Categories1')
    dm_transport.rename_col('liquid-ff-diesel', 'diesel', dim='Categories1')
    dm_transport.rename_col('liquid-ff-gasoline', 'gasoline', dim='Categories1')

    dm_industry.rename_col('gas-ff-natural', 'gas', dim='Categories1')
    dm_industry.rename_col('solid-ff-coal', 'coal', dim='Categories1')
    dm_ammonia.rename_col('gas-ff-natural', 'gas', dim='Categories1')
    dm_ammonia.rename_col('solid-ff-coal', 'coal', dim='Categories1')

    dm_agriculture.rename_col('gas-ff-natural', 'gas', dim='Categories1')
    dm_agriculture.rename_col('liquid-ff-oil', 'fuel-oil', dim='Categories1')
    dm_agriculture.rename_col('solid-ff-coal', 'coal', dim='Categories1')
    dm_agriculture.rename_col('liquid-ff-diesel', 'diesel', dim='Categories1')
    dm_agriculture.rename_col('liquid-ff-gasoline', 'gasoline', dim='Categories1')

    # Build the matrix for operation
    dm_fuel_demand = dm_power.copy()
    dm_fuel_demand.append(dm_buildings, dim='Variables')
    dm_fuel_demand.append(dm_transport, dim='Variables')
    dm_fuel_demand.append(dm_industry, dim='Variables')
    dm_fuel_demand.append(dm_ammonia, dim='Variables')
    dm_fuel_demand.append(dm_agriculture, dim='Variables')

    ######################################
    # CalculationLeafs - Gas demand for Hydrogen production [GWh]
    ######################################

    dm_cp = DM_refinery['constant']
    idx_cst = dm_cp.idx
    ay_hydrogen_to_gas = \
        dm_hydrogen.array[...] \
        * dm_cp.array[idx_cst['cp_refinery-yield_hydrogen-to-gas']]

    dm_demand_gas = dm_fuel_demand.filter({'Categories1': ['gas']})
    #dm_demand_gas.add(ay_hydrogen_to_gas, dim='Variables', col_label='hyd_energy-demand')

    ######################################
    # CalculationLeafs - Oil demand for oil-based fuel production [GWh]
    ######################################

    # Oil equivalent [TWh]
    dm_oil = dm_fuel_demand.filter({'Categories1': ['diesel', 'fuel-oil', 'gasoline', 'kerosene']})
    dm_cp = DM_refinery['constant']
    dm_cp = dm_cp.filter({'Variables': ['cp_refinery-yield_diesel', 'cp_refinery-yield_fuel-oil',
                                        'cp_refinery-yield_gasoline', 'cp_refinery-yield_kerosene']})
    dm_cp.deepen(based_on='Variables')
    ay_oil_equivalent = dm_oil.array[...] * dm_cp.array[np.newaxis, np.newaxis, :, :]
    dm_oil_equivalent = DataMatrix.based_on(ay_oil_equivalent, dm_oil)

    # Refinery self-consumption [TWh]
    dm_cp = DM_refinery['constant']
    dm_self_consumption = dm_cp.filter({'Variables': ['cp_refinery-efficiency_energy-use']})
    ay_oil_equivalent = dm_oil_equivalent.array[...] \
                        / (1 - (dm_self_consumption.array[np.newaxis, np.newaxis, np.newaxis, :]))
    dm_oil_equivalent_gross = DataMatrix.based_on(ay_oil_equivalent, dm_oil)

    # Refinery losses [GWh]
    dm_cp = DM_refinery['constant']
    dm_loss = dm_cp.filter({'Variables': ['cp_refinery-efficiency_loss']})
    ay_oil_equivalent_net = dm_oil_equivalent_gross.array[...] \
                        /(1 - dm_loss.array[np.newaxis, np.newaxis, np.newaxis, :])
    dm_demand_oil = DataMatrix.based_on(ay_oil_equivalent_net, dm_oil)

    ######################################
    # CalculationLeafs - Fossil fuels emissions [TWh]
    ######################################

    # Energy demand
    dm_demand_coal = dm_fuel_demand.filter({'Categories1': ['coal']})
    dm_fossil_demand = dm_demand_coal.copy()
    dm_fossil_demand.append(dm_demand_oil, dim='Categories1')
    dm_fossil_demand.append(dm_demand_gas, dim='Categories1')

    # Emission factors
    dm_factors = DM_refinery['constant'].filter_w_regex({'Variables': 'cp_refinery-emission-factor.*'})
    dm_factors.deepen(based_on='Variables')

    # Emissions
    ay_fossil_emissions = dm_fossil_demand.array[...] \
                            / dm_factors.array[np.newaxis, np.newaxis, :, :]
    dm_fossil_emissions = DataMatrix.based_on(ay_fossil_emissions, dm_fossil_demand)
    for v in dm_fossil_emissions.col_labels["Variables"]:
        dm_fossil_emissions.units[v] = "Mt"

    DM_refinery_out = {
        'fossil-demand': dm_fossil_demand,
        'fossil-emissions': dm_fossil_emissions
    }
    return DM_refinery_out

    # TODO: (1) Hydrogen; (2) Balance; (3) Costs; (4) CCUS
    
def primary_demand(DM_refinery_out):
    
    dm_temp = DM_refinery_out["fossil-demand"].copy()
    dm_fos = dm_temp.groupby({'fos_primary-demand': '.*'}, dim='Variables', regex=True, inplace=False)
    dm_fos.groupby({'oil': ['diesel', 'fuel-oil', 'gasoline', 'kerosene']}, dim='Categories1', regex=False, inplace=True)
    
    return dm_fos

def variables_to_tpe(DM_refinery_out, dm_fos):
    
    dm_temp = DM_refinery_out["fossil-demand"].copy()
    dm_temp.rename_col("pow_energy-demand", "elc_energy-demand","Variables")
    dm_temp.rename_col("coal", "solid-ff-coal","Categories1")
    dm_temp.rename_col("diesel", "liquid-ff-diesel","Categories1")
    dm_temp.rename_col("fuel-oil", "liquid-ff-fuel-oil","Categories1")
    dm_temp.rename_col("gasoline", "liquid-ff-gasoline","Categories1")
    dm_temp.rename_col("kerosene", "liquid-ff-kerosene","Categories1")
    dm_temp.rename_col("gas", "gas-ff-natural","Categories1")
    dm_tpe = dm_temp.flatten()
    
    dm_tpe.append(dm_fos.flatten(), "Variables")
    
    return dm_tpe

def oilrefinery_emissions_interface(DM_refinery_out):
    
    dm_temp = DM_refinery_out["fossil-emissions"].copy()
    dm_temp.group_all("Categories1")
    dm_temp.groupby({'fos_emissions-CO2': '.*'}, "Variables", regex=True, inplace=True)
    
    return dm_temp

#######################################################################################################################
# CoreModule - Refinery
#######################################################################################################################

def refinery(lever_setting, years_setting, interface=Interface()):
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    refinery_data_file = os.path.join(current_file_directory,
                                        '../_database/data/datamatrix/geoscale/oil-refinery.pickle')
    with open(refinery_data_file, 'rb') as handle:  # read binary (rb)
        DM_refinery = pickle.load(handle)

    # Country filter setting (based on fxa, because their is no read data function / no levers)

    dm_fxa = DM_refinery['fxa']['refinery-ratio']
    cntr_list = dm_fxa.col_labels['Country']

    # Data input (other modules) & filter the country

    if interface.has_link(from_sector='power', to_sector='oil-refinery'):
        dm_power = interface.get_link(from_sector='power', to_sector='oil-refinery')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing power to oil-refinery interface')
        dm_power = simulate_power_to_refinery_input()
        dm_power = dm_power.filter({'Country': cntr_list})

    if interface.has_link(from_sector='buildings', to_sector='oil-refinery'):
        dm_buildings = interface.get_link(from_sector='buildings', to_sector='oil-refinery')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing buildings to oil-refinery interface')
        dm_buildings = simulate_buildings_to_refinery_input()
        dm_buildings = dm_buildings.filter({'Country': cntr_list})

    if interface.has_link(from_sector='transport', to_sector='oil-refinery'):
        dm_transport = interface.get_link(from_sector='transport', to_sector='oil-refinery')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing transport to oil-refinery interface')
        dm_transport = simulate_transport_to_refinery_input()
        dm_transport = dm_transport.filter({'Country': cntr_list})

    if interface.has_link(from_sector='industry', to_sector='oil-refinery'):
        dm_industry = interface.get_link(from_sector='industry', to_sector='oil-refinery')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing industry to oil-refinery interface')
        dm_industry = simulate_industry_to_refinery_input()
        dm_industry = dm_industry.filter({'Country': cntr_list})

    if interface.has_link(from_sector='ammonia', to_sector='oil-refinery'):
        dm_ammonia = interface.get_link(from_sector='ammonia', to_sector='oil-refinery')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing ammonia to oil-refinery interface')
        dm_ammonia = simulate_ammonia_to_refinery_input()
        dm_ammonia = dm_ammonia.filter({'Country': cntr_list})

    if interface.has_link(from_sector='agriculture', to_sector='oil-refinery'):
        dm_agriculture = interface.get_link(from_sector='agriculture', to_sector='oil-refinery')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing agriculture to oil-refinery interface')
        dm_agriculture = simulate_agriculture_to_refinery_input()
        dm_agriculture = dm_agriculture.filter({'Country': cntr_list})

    DM_fuel_demand = {
        'power': dm_power,
        'buildings': dm_buildings,
        'transport': dm_transport,
        'industry': dm_industry,
        'ammonia': dm_ammonia,
        'agriculture': dm_agriculture
    }

    # Fuel production function
    DM_refinery_out = fuel_production_workflow(DM_refinery, DM_fuel_demand)
    
    # primary demand
    dm_fos = primary_demand(DM_refinery_out)
    
    # tpe
    dm_tpe = variables_to_tpe(DM_refinery_out, dm_fos)
    df_tpe = dm_tpe.write_df()
    
    # interface emissions
    dm_ems = oilrefinery_emissions_interface(DM_refinery_out)
    interface.add_link(from_sector='oil-refinery', to_sector='emissions', dm=dm_ems)
    
    # interface minerals
    interface.add_link(from_sector='oil-refinery', to_sector='minerals', dm=dm_fos.flatten())

    return df_tpe

#######################################################################################################################
# LocalRun - Refinery
#######################################################################################################################

def local_refinery_run():
    # Function to run only transport module without converter and tpe
    years_setting = [1990, 2015, 2050, 5]
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(current_file_directory, '../config/lever_position.json'))
    lever_setting = json.load(f)[0]

    global_vars = {'geoscale': '.*'}
    filter_geoscale(global_vars)

    results_run = refinery(lever_setting, years_setting)

    return results_run

# __file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/model/emissions_module.py"
# results_run = local_refinery_run()


