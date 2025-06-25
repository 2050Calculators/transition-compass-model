#######################################################################################################################
# SECTION: Import Packages, Classes & Functions
#######################################################################################################################

import numpy as np
import pickle  # read/write the data in pickle
import json  # read the lever setting
import os  # operating system (e.g., look for workspace)
import pandas as pd
import time

# Import Class
from model.common.data_matrix_class import DataMatrix  # Class for the model inputs
from model.common.constant_data_matrix_class import ConstantDataMatrix  # Class for the constant inputs
from model.common.auxiliary_functions import read_level_data, filter_geoscale, simulate_input

# ImportFunctions
from model.common.io_database import read_database_to_ots_fts_dict
from model.common.hourly_data_functions import hourly_data_reader
from model.common.interface_class import Interface

#######################################################################################################################
# ModelSetting - Power
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

    # Initiate the dictionary for ots & fts
    dict_ots = {}
    dict_fts = {}

#######################################################################################################################
# DataLever - Power
#######################################################################################################################

    # Database - Power - Lever: Solar PV capacity
    file = 'power_pv-capacity'
    lever = 'pv-capacity'
    dict_ots, dict_fts = read_database_to_ots_fts_dict(file, lever, num_cat=1,baseyear=baseyear, years=years_all,
                                                       dict_ots=dict_ots,dict_fts=dict_fts)

    # Database - Power - Lever: Solar CSP capacity
    file = 'power_csp-capacity'
    lever = 'csp-capacity'
    dict_ots, dict_fts = read_database_to_ots_fts_dict(file, lever, num_cat=1,baseyear=baseyear, years=years_all,
                                                       dict_ots=dict_ots,dict_fts=dict_fts)

    # Database - Power - Lever: Offshore Wind Power capacity
    file = 'power_offshore-wind-capacity'
    lever = 'offshore-wind-capacity'
    dict_ots, dict_fts = read_database_to_ots_fts_dict(file, lever, num_cat=1,baseyear=baseyear, years=years_all,
                                                       dict_ots=dict_ots,dict_fts=dict_fts)

    # Database - Power - Lever: Onshore Wind Power capacity
    file = 'power_onshore-wind-capacity'
    lever = 'onshore-wind-capacity'
    dict_ots, dict_fts = read_database_to_ots_fts_dict(file, lever, num_cat=1,baseyear=baseyear, years=years_all,
                                                       dict_ots=dict_ots,dict_fts=dict_fts)

    # Database - Power - Lever: Biogas capacity
    file = 'power_biogas-capacity'
    lever = 'biogas-capacity'
    dict_ots, dict_fts = read_database_to_ots_fts_dict(file, lever, num_cat=1,baseyear=baseyear, years=years_all,
                                                       dict_ots=dict_ots,dict_fts=dict_fts)

    # Database - Power - Lever: Biomass capacity
    file = 'power_biomass-capacity'
    lever = 'biomass-capacity'
    dict_ots, dict_fts = read_database_to_ots_fts_dict(file, lever, num_cat=1,baseyear=baseyear, years=years_all,
                                                       dict_ots=dict_ots,dict_fts=dict_fts)

    # Database - Power - Lever: Hydroelectric capacity
    file = 'power_hydroelectric-capacity'
    lever = 'hydroelectric-capacity'
    dict_ots, dict_fts = read_database_to_ots_fts_dict(file, lever, num_cat=1,baseyear=baseyear, years=years_all,
                                                       dict_ots=dict_ots,dict_fts=dict_fts)

    # Database - Power - Lever: Geothermal capacity
    file = 'power_geothermal-capacity'
    lever = 'geothermal-capacity'
    dict_ots, dict_fts = read_database_to_ots_fts_dict(file, lever, num_cat=1, baseyear=baseyear, years=years_all,
                                                       dict_ots=dict_ots, dict_fts=dict_fts)

    # Database - Power - Lever: Marine energy capacity
    file = 'power_marine-capacity'
    lever = 'marine-capacity'
    dict_ots, dict_fts = read_database_to_ots_fts_dict(file, lever, num_cat=1, baseyear=baseyear, years=years_all,
                                                       dict_ots=dict_ots, dict_fts=dict_fts)

    # Database - Power - Lever: Gas capacity
    file = 'power_gas-capacity'
    lever = 'gas-capacity'
    dict_ots, dict_fts = read_database_to_ots_fts_dict(file, lever, num_cat=1,baseyear=baseyear, years=years_all,
                                                       dict_ots=dict_ots, dict_fts=dict_fts)

    # Database - Power - Lever: Oil capacity
    file = 'power_oil-capacity'
    lever = 'oil-capacity'
    dict_ots, dict_fts = read_database_to_ots_fts_dict(file, lever, num_cat=1,baseyear=baseyear, years=years_all,
                                                       dict_ots=dict_ots, dict_fts=dict_fts)

    # Database - Power - Lever: Coal capacity
    file = 'power_coal-capacity'
    lever = 'coal-capacity'
    dict_ots, dict_fts = read_database_to_ots_fts_dict(file, lever, num_cat=1, baseyear=baseyear, years=years_all,
                                                       dict_ots=dict_ots, dict_fts=dict_fts)

    # Database - Power - Lever: Nuclear capacity
    file = 'power_nuclear-capacity'
    lever = 'nuclear-capacity'
    dict_ots, dict_fts = read_database_to_ots_fts_dict(file, lever, num_cat=1, baseyear=baseyear, years=years_all,
                                                       dict_ots=dict_ots, dict_fts=dict_fts)

    # Database - Power - Lever: CCUS capacity
    file = 'power_carbon-storage'
    lever = 'carbon-storage-capacity'
    dict_ots, dict_fts = read_database_to_ots_fts_dict(file, lever, num_cat=1, baseyear=baseyear, years=years_all,
                                                       dict_ots=dict_ots, dict_fts=dict_fts)

    # Database - Power - Lever: Vehicle charging profile
    file = 'power_ev-charging-profile'
    lever = 'ev-charging-profile'
    dict_ots, dict_fts = hourly_data_reader(file, years_setting, lever, dict_ots, dict_fts)

    # Database - Power - Lever: Non-residential heating
    file = 'power_non-residential-heat-profile'
    lever = 'non-residential-heat-profile'
    dict_ots, dict_fts = hourly_data_reader(file, years_setting, lever, dict_ots, dict_fts)

    # Database - Power - Lever: Residential heating
    file = 'power_residential-heat-profile'
    lever = 'residential-heat-profile'
    dict_ots, dict_fts = hourly_data_reader(file, years_setting, lever, dict_ots, dict_fts)

    # Database - Power - Lever: Non-residential cooling
    file = 'power_non-residential-cooling-profile'
    lever = 'non-residential-cooling-profile'
    dict_ots, dict_fts = hourly_data_reader(file, years_setting, lever, dict_ots, dict_fts)

    # Database - Power - Lever: Residential cooling
    file = 'power_residential-cooling-profile'
    lever = 'residential-cooling-profile'
    dict_ots, dict_fts = hourly_data_reader(file, years_setting, lever, dict_ots, dict_fts)

#######################################################################################################################
# DataFixedAssumptions - Power
#######################################################################################################################

    # Database - PV profile
    file = 'power_pv-profile'
    dm_profile_pv = hourly_data_reader(file, years_setting)

    # Database - Onshore wind profile
    file = 'power_wind-onshore-profile'
    dm_profile_onshore = hourly_data_reader(file, years_setting)

    # Database - Onshore wind profile
    file = 'power_wind-offshore-profile'
    dm_profile_offshore = hourly_data_reader(file, years_setting)

    # Database - Train profile
    file = 'power_train-profile'
    dm_pow_train = hourly_data_reader(file, years_setting)

    # Database - Non-residential appliance profile
    file = 'power_non-residential-appliances-profile'
    dm_appliances_non_residential = hourly_data_reader(file, years_setting)

    # Database - Residential appliance profile
    file = 'power_residential-appliances-profile'
    dm_appliances_residential = hourly_data_reader(file, years_setting)

    # Database - Non-residential hotwater profile
    file = 'power_non-residential-hotwater-profile'
    dm_hotwater_non_residential = hourly_data_reader(file, years_setting)

    # Database - Residential hotwater profile
    file = 'power_residential-hotwater-profile'
    dm_hotwater_residential = hourly_data_reader(file, years_setting)

    dict_fxa = {
        'train-profile': dm_pow_train,
        'non-residential-appliances-profile': dm_appliances_non_residential,
        'residential-appliances-profile': dm_appliances_residential,
        'non-residential-hotwater-profile': dm_hotwater_non_residential,
        'residential-hotwater-profile': dm_hotwater_residential,
        'pv-profile': dm_profile_pv,
        'onshore-wind-profile': dm_profile_onshore,
        'offshore-wind-profile': dm_profile_offshore
    }


#######################################################################################################################
# DataConstants - Power
#######################################################################################################################



    cdm_const_cat0 = ConstantDataMatrix.extract_constant('interactions_constants',
                                                        pattern='cp_timestep_hours-a-year|'
                                                                'cp_carbon-capture_power-self-consumption', num_cat=0)

    cdm_const_cat1 = ConstantDataMatrix.extract_constant('interactions_constants',
                                                        pattern='cp_power-unit-self-consumption|'
                                                                'cp_fuel-based-power-efficiency', num_cat=1)

    cdm_emission_factor = ConstantDataMatrix.extract_constant('interactions_constants',
                                                              pattern='cp_tec_emission-factor_.*', num_cat=2)

    cdm_emission_factor.rename_col(['gas-bio', 'gas-ff-natural', 'liquid-ff-oil', 'solid-bio', 'solid-ff-coal'],
                                   ['biogas', 'gas', 'oil', 'biomass', 'coal'], dim='Categories2')
    cdm_emission_factor = cdm_emission_factor.filter({'Categories2': ['biogas', 'gas', 'oil', 'biomass', 'coal']})

    dict_const = {
        'constant_0': cdm_const_cat0,
        'constant_1': cdm_const_cat1,
        'emission-factors': cdm_emission_factor
    }

#######################################################################################################################
# DataMatrices - Power Data Matrix
#######################################################################################################################

    DM_power = {
        'fts': dict_fts,
        'ots': dict_ots,
        'fxa': dict_fxa,
        'constant': dict_const
    }
#######################################################################################################################
# DataPickle - Power
#######################################################################################################################

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, '../_database/data/datamatrix/power.pickle')
    with open(f, 'wb') as handle:
        pickle.dump(DM_power, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return

# update_interaction_constant_from_file('interactions_constants_local') # uncomment to update constant
# database_from_csv_to_datamatrix()  # un-comment to update

#######################################################################################################################
# DataSubMatrices - Power
#######################################################################################################################

def read_data(data_file, lever_setting):
    with open(data_file, 'rb') as handle:
        DM_power = pickle.load(handle)

    # FXA data matrix
    DM_ots_fts = read_level_data(DM_power, lever_setting)

    # Capacity per technology (fuel-based)
    dm_coal = DM_ots_fts['coal-capacity']
    dm_capacity = dm_coal.copy()

    dm_oil = DM_ots_fts['oil-capacity']
    dm_capacity.append(dm_oil, dim='Categories1')

    dm_gas = DM_ots_fts['gas-capacity']
    dm_capacity.append(dm_gas, dim='Categories1')

    dm_nuclear = DM_ots_fts['nuclear-capacity']
    dm_capacity.append(dm_nuclear, dim='Categories1')

    dm_biogas = DM_ots_fts['biogas-capacity']
    dm_capacity.append(dm_biogas, dim='Categories1')

    dm_biomass = DM_ots_fts['biomass-capacity']
    dm_capacity.append(dm_biomass, dim='Categories1')

    # Capacity per technology (non-fuel based)

    dm_pv = DM_ots_fts['pv-capacity']
    dm_capacity.append(dm_pv, dim='Categories1')

    dm_csp = DM_ots_fts['csp-capacity']
    dm_capacity.append(dm_csp, dim='Categories1')

    dm_offshore_wind = DM_ots_fts['offshore-wind-capacity']
    dm_capacity.append(dm_offshore_wind, dim='Categories1')

    dm_onshore_wind = DM_ots_fts['onshore-wind-capacity']
    dm_capacity.append(dm_onshore_wind, dim='Categories1')

    dm_hydroelectric = DM_ots_fts['hydroelectric-capacity']
    dm_capacity.append(dm_hydroelectric, dim='Categories1')

    dm_geothermal = DM_ots_fts['geothermal-capacity']
    dm_capacity.append(dm_geothermal, dim='Categories1')

    dm_marine = DM_ots_fts['marine-capacity']
    dm_capacity.append(dm_marine, dim='Categories1')

    dm_ccus = DM_ots_fts['carbon-storage-capacity']

    # Hourly data (lever)

    dm_profile_vehicle = DM_ots_fts['ev-charging-profile']
    dm_profile_heating_nr = DM_ots_fts['non-residential-heat-profile']
    dm_profile_heating_r = DM_ots_fts['residential-heat-profile']
    dm_profile_cooling_nr = DM_ots_fts['non-residential-cooling-profile']
    dm_profile_cooling_r = DM_ots_fts['residential-cooling-profile']

    # Hourly data (fxa)
    dm_profile_pv = DM_power['fxa']['pv-profile']
    dm_profile_onshore = DM_power['fxa']['onshore-wind-profile']
    dm_profile_offshore = DM_power['fxa']['offshore-wind-profile']

    dm_profile_appliances_nr = DM_power['fxa']['non-residential-appliances-profile']
    dm_profile_appliances_r = DM_power['fxa']['residential-appliances-profile']
    dm_profile_hotwater_nr = DM_power['fxa']['non-residential-hotwater-profile']
    dm_profile_hotwater_r = DM_power['fxa']['residential-hotwater-profile']
    dm_profile_train = DM_power['fxa']['train-profile']

    DM_production_profiles = {
        'pv-profile':dm_profile_pv,
        'onshore-wind-profile': dm_profile_onshore,
        'offshore-wind-profile': dm_profile_offshore,
    }

    DM_demand_profiles = {
        'non-residential-appliances': dm_profile_appliances_nr,
        'residential-appliances': dm_profile_appliances_r,
        'train-profile': dm_profile_train,
        'ev-charging-profile':  dm_profile_vehicle,
        'non-residential-hotwater': dm_profile_hotwater_nr,
        'residential-hotwater': dm_profile_hotwater_r,
        'non-residential-heat-profile': dm_profile_heating_nr,
        'residential-heat-profile': dm_profile_heating_r,
        'non-residential-cooling-profile': dm_profile_cooling_nr,
        'residential-cooling-profile': dm_profile_cooling_r
    }

    # Constants
    cdm_const = DM_power['constant']

    return dm_capacity, dm_ccus, cdm_const, DM_production_profiles, DM_demand_profiles


#######################################################################################################################
# LocalInterfaces - Climate
#######################################################################################################################
def simulate_climate_to_power_input():

    dm_climate = simulate_input(from_sector="climate", to_sector="power", num_cat=1)

    return dm_climate

#######################################################################################################################
# LocalInterfaces - Buildings
#######################################################################################################################

def simulate_buildings_to_power_input():
    # Tuto: Local module interface
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../_database/data/xls/"
                                             "All-Countries-interface_from-buildings-to-power.xlsx")
    df = pd.read_excel(f)
    dm = DataMatrix.create_from_df(df, num_cat=2)

    # Space heating flow:
    dm_bld_heating = dm.filter_w_regex({'Categories2': 'space-heating'})
    dm_bld_cooling = dm.filter_w_regex({'Categories2': 'space-cooling'})
    dm_bld_appliances = dm.filter_w_regex({'Categories2': 'cooking|hot-water|lighting|appliances'})
    dm_bld_heatpump = dm.filter_w_regex({'Categories2': 'heatpumps'})

    DM_bld= {
        'appliance': dm_bld_appliances,
        'space-heating': dm_bld_heating,
        'cooling': dm_bld_cooling,
        'heatpump': dm_bld_heatpump
    }

    return DM_bld
#######################################################################################################################
# LocalInterfaces - Transport
#######################################################################################################################

def simulate_transport_to_power_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../_database/data/xls/"
                                             "All-Countries-interface_from-transport-to-power.xlsx")
    df = pd.read_excel(f)
    dm_tra = DataMatrix.create_from_df(df, num_cat=0)

    dm_hydro = dm_tra.filter_w_regex({'Variables': '.*hydrogen'}, inplace=False)
    dm_tra.drop(dim='Variables', col_label='.*hydrogen')

    DM_tra = {
        'hydrogen': dm_hydro,
        'electricity': dm_tra
    }

    return DM_tra

#######################################################################################################################
# LocalInterfaces - Industry
#######################################################################################################################

def simulate_industry_to_power_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../_database/data/xls/"
                                             "All-Countries-interface_from-industry-to-power.xlsx")
    df = pd.read_excel(f, sheet_name="default")
    dm = DataMatrix.create_from_df(df, num_cat=0)

    # Units from TWh to GWh
    dm.change_unit('ind_energy-demand_electricity', factor=1e3, old_unit='TWh', new_unit='GWh')
    dm.change_unit('ind_energy-demand_hydrogen', factor=1e3, old_unit='TWh', new_unit='GWh')

    # Space heating flow:
    dm_ind_electricity = dm.filter_w_regex({'Variables': 'ind_energy-demand_electricity'})
    dm_ind_hydrogen = dm.filter_w_regex({'Variables': 'ind_energy-demand_hydrogen'})

    DM_industry = {
        'electricity': dm_ind_electricity,
        'hydrogen': dm_ind_hydrogen
    }

    return DM_industry

#######################################################################################################################
# LocalInterfaces - Ammonia
#######################################################################################################################

def simulate_ammonia_to_power_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../_database/data/xls/"
                                             "All-Countries-interface_from-ammonia-to-power.xlsx")
    df = pd.read_excel(f, sheet_name="default")
    dm = DataMatrix.create_from_df(df, num_cat=0)

    # From TWh to GWh
    dm.change_unit('amm_energy-demand_electricity', factor=1e3, old_unit='TWh', new_unit='GWh')
    dm.change_unit('amm_energy-demand_hydrogen', factor=1e3, old_unit='TWh', new_unit='GWh')

    # Space heating flow:
    dm_amm_electricity = dm.filter_w_regex({'Variables': 'amm_energy-demand_electricity'})
    dm_amm_hydrogen = dm.filter_w_regex({'Variables': 'amm_energy-demand_hydrogen'})

    DM_ammonia = {
        'electricity': dm_amm_electricity,
        'hydrogen': dm_amm_hydrogen
    }
    return DM_ammonia

#######################################################################################################################
# LocalInterfaces - Agriculture
#######################################################################################################################

def simulate_agriculture_to_power_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../_database/data/xls/"
                                             "All-Countries-interface_from-agriculture-to-power.xlsx")
    df = pd.read_excel(f, sheet_name="default")
    dm = DataMatrix.create_from_df(df, num_cat=0)

    # Space heating flow:
    dm_agr_electricity = dm.filter_w_regex({'Variables': 'agr_energy-demand_electricity'})
    dm_agr_electricity.change_unit('agr_energy-demand_electricity', factor=1e3, old_unit='TWh', new_unit='GWh')

    return dm_agr_electricity

#######################################################################################################################
# CalculationTree - Power - Yearly Production
#######################################################################################################################
def yearly_production_workflow(dm_climate, dm_capacity, dm_ccus, cdm_const):

    ######################################
    # CalculationLeafs - Gross electricity production [GWh]
    ######################################
    # Tuto: Tree parallel (array)
    idx_cap = dm_capacity.idx
    idx_clm = dm_climate.idx
    ay_gross_yearly_production = dm_capacity.array[:, :, idx_cap['pow_existing-capacity'], :] \
                                 * dm_climate.array[:, :, idx_clm['clm_capacity-factor'], :]*8760
    dm_capacity.add(ay_gross_yearly_production, dim='Variables', col_label='pow_gross-yearly-production', unit='GWh')

    #######################################################
    # CalculationLeafs - Net production (fuel based, self-consumption) [GWh]
    #######################################################
    # Tuto: Tree Split & constants
    cdm_self_consumption = cdm_const['constant_1']
    idx_const = cdm_self_consumption.idx
    dm_fb_capacity = dm_capacity.filter_w_regex({'Categories1': 'biogas|biomass|coal|gas|oil|nuclear'})
    ay_self_consumption = dm_fb_capacity.array[:, :, idx_cap['pow_gross-yearly-production'], :] \
                          * cdm_self_consumption.array[np.newaxis, np.newaxis,
                            idx_const['cp_power-unit-self-consumption'],:]
    dm_fb_capacity.add(ay_self_consumption, dim='Variables', col_label='pow_net-yearly-production', unit='GWh')

    #########################################
    # CalculationLeafs - Self consumption of power units [GWh]
    #########################################
    # Tuto: Tree parallel (operation)
    dm_fb_capacity.operation('pow_gross-yearly-production', '-', 'pow_net-yearly-production',
                            dim="Variables", out_col='pow_power-loss-self-consumption', unit='GWh')

    #########################################
    # CalculationLeafs - Self consumption of power units [GWh]
    #########################################

    idx_cap = dm_fb_capacity.idx
    cdm_fuel_efficiency = cdm_const['constant_1']
    idx_const = cdm_fuel_efficiency.idx
    ay_fuel_consumption = dm_fb_capacity.array[:, :, idx_cap['pow_gross-yearly-production'], :] \
                          / cdm_fuel_efficiency.array[np.newaxis, np.newaxis,
                            idx_const['cp_fuel-based-power-efficiency'], :]
    dm_fb_capacity.add(ay_fuel_consumption, dim='Variables', col_label='pow_fuel-demand-for-power', unit='GWh')

    #########################################
    # CalculationLeafs - Production share under CCUS [GWh]
    #########################################

    idx_cap = dm_fb_capacity.idx
    idx_ccus = dm_ccus.idx
    ay_ccus_production = dm_fb_capacity.array[:, :, idx_cap['pow_net-yearly-production'], :] \
                          * dm_ccus.array[:, :, idx_ccus['pow_carbon-capture-storage'], idx_ccus['ratio'],np.newaxis]
    dm_fb_capacity.add(ay_ccus_production,
                       dim='Variables', col_label='pow_gross-yearly-production-with-ccs', unit='GWh')

    #########################################
    # CalculationLeafs - Production share without CCUS [GWh]
    #########################################

    idx_cap = dm_fb_capacity.idx
    idx_ccus = dm_ccus.idx
    ay_ccus_production = dm_fb_capacity.array[:, :, idx_cap['pow_net-yearly-production'], :] \
                         * dm_ccus.array[:, :, idx_ccus['pow_carbon-capture-storage'], idx_ccus['reverse-ratio'], np.newaxis]
    dm_fb_capacity.add(ay_ccus_production,
                       dim='Variables', col_label='pow_net-yearly-production-without-ccs', unit='GWh')

    ###########################################################################
    # CalculationLeafs - Net fuel-based production accounting for CCUS process consumption [GWh]
    ###########################################################################

    idx_cap = dm_fb_capacity.idx
    cdm_ccus_efficiency = cdm_const['constant_0']
    idx_const = cdm_ccus_efficiency.idx
    ay_net_production_ccus = dm_fb_capacity.array[:, :, idx_cap['pow_gross-yearly-production-with-ccs'], :] \
                          * cdm_fuel_efficiency.array[np.newaxis, np.newaxis,
                            idx_const['cp_carbon-capture_power-self-consumption'], :]
    dm_fb_capacity.add(ay_net_production_ccus, dim='Variables',
                       col_label='pow_net-yearly-production-with-ccs', unit='GWh')

    ###########################################################################
    # CalculationLeafs - CCUS process consumption [GWh]
    ###########################################################################

    dm_fb_capacity.operation('pow_gross-yearly-production-with-ccs', '-', 'pow_net-yearly-production-with-ccs',
                             dim="Variables", out_col='pow_power-consumption-ccus', unit='GWh')

    #####################################################################################
    # CalculationLeafs - Net power production [GWh] - Fuel based - Accounting for self & CCUS consumption
    #####################################################################################

    dm_fb_capacity.drop(col_label=['pow_net-yearly-production'], dim='Variables')
    dm_fb_capacity.operation('pow_net-yearly-production-with-ccs', '+', 'pow_net-yearly-production-without-ccs',
                             dim="Variables", out_col='pow_net-yearly-production', unit='GWh')

    #####################################################################################
    # CalculationLeafs - Fuel demand for fuel-based power production
    #####################################################################################

    dm_fuel_demand = dm_fb_capacity.filter({'Variables': ['pow_gross-yearly-production']})
    dm_fuel_efficiency = cdm_fuel_efficiency.filter({'Variables': ['cp_fuel-based-power-efficiency']})
    ay_fuel_demand = dm_fuel_demand.array[...] \
                                 * dm_fuel_efficiency.array[...]
    dm_fuel_demand.add(ay_fuel_demand, dim='Variables', col_label='pow_fuel-demand-for-power', unit='GWh')


    #####################################################################################
    # CalculationLeafs - Aggregated net power production with no hourly profiles
    #####################################################################################

    # Filter - Energy production with no hourly profile
    dm_fb_np = dm_fb_capacity.filter({'Variables': ['pow_net-yearly-production']})


    # Filter - Renewable energy production
    dm_nfb = dm_capacity.filter({'Variables': ['pow_gross-yearly-production']})
    dm_nfb.rename_col(col_in='pow_gross-yearly-production', col_out='pow_net-yearly-production', dim='Variables')
    dm_nfb = dm_nfb.filter({'Categories1':['hydroelectric','solar-csp','geothermal','marine','solar-pv',
                                           'wind-onshore','wind-offshore']})

    # Data Matrix - Yearly production [GWh]
    dm_yearly_production = dm_fb_np.copy()
    dm_yearly_production.append(dm_nfb, dim='Categories1')

    # Sub Data Matrix - Yearly production [GWh] (no hourly profiles)

    dm_production_np = dm_yearly_production.filter_w_regex({'Categories1': 'hydroelectric|solar-csp|geothermal|marine|'
                                                                    'oil|gas|coal|biomass|biogas|nuclear'})

    # Sum - Total (Categories1) energy production with no hourly profile
    ay_total_np = np.nansum(dm_production_np.array[...], axis=-1)
    dm_production_np.add(ay_total_np, dim='Categories1', col_label='total')

    #####################################################################################
    # CalculationLeafs - Net power production with hourly profiles
    #####################################################################################

    # Sub Data Matrix - Yearly production [GWh] (hourly profiles)

    dm_production_p = dm_yearly_production.filter_w_regex({'Categories1': 'solar-pv|wind-onshore|wind-offshore'})

    return dm_capacity, dm_fb_capacity, dm_production_np, dm_production_p

#######################################################################################################################
# CalculationTree - Power - Hourly production
#######################################################################################################################

    # Tuto: Hourly data computation
def hourly_production_workflow(dm_production_np, dm_production_p, DM_production_profiles, baseyear):

    ######################################
    # CalculationLeafs - Hourly production per technology (wind & solar)[GWh]
    ######################################

    # Extract hourly data (fake pv hourly profile)
    dm_profile_hourly = DM_production_profiles['pv-profile']
    dm_profile_hourly.append(DM_production_profiles['offshore-wind-profile'], dim='Variables')
    dm_profile_hourly.append(DM_production_profiles['onshore-wind-profile'], dim='Variables')

    # Indexes for computation
    idx_cap = dm_production_p.idx

    # Hourly profile
    # Warning (1): you need to only multiply 2015 and FTS ('idx_cap[base-year]:')
    # Warning (2): you need to add 3 new axis to match the dimensions
    ay_hourly_profile = dm_production_p.array[:, idx_cap[baseyear]:, idx_cap['pow_net-yearly-production'], :,
                   np.newaxis, np.newaxis, np.newaxis] \
                     * dm_profile_hourly.array[...]

    # Reshape of the output
    dm_profile_hourly.array = ay_hourly_profile
    for key in dm_profile_hourly.units.keys():
        dm_profile_hourly.units[key] = 'GWh'

    ######################################
    # CalculationLeafs - Hourly production (total)[GWh]
    ######################################

    dm_yearly_production_other = dm_production_np.filter({'Categories1':['total']})
    ay_hourly_profile_total = np.sum(dm_profile_hourly.array[...],axis=2)
    dm_profile_hourly.add(ay_hourly_profile_total,dim='Variables', col_label='pow_total-profile',unit='GWh')

    idx_year = dm_yearly_production_other.idx
    idx_hourly = dm_profile_hourly.idx

    dm_yearly_production_other.array = dm_yearly_production_other.array/8760
    ay_total_hourly = dm_yearly_production_other.array[:,idx_year[baseyear]:,idx_year['pow_net-yearly-production'],idx_year['total'],np.newaxis,np.newaxis,np.newaxis]\
                      +dm_profile_hourly.array[:,:,idx_hourly['pow_total-profile'],...]

    dm_profile_hourly.add(ay_total_hourly,dim='Variables', col_label='pow_total-hourly-production',unit='GWh')
    dm_hourly_production = dm_profile_hourly.filter({'Variables':['pow_total-hourly-production']})

    return dm_hourly_production

#######################################################################################################################
# CalculationTree - Power - Yearly demand
#######################################################################################################################

def yearly_demand_workflow(DM_bld, dm_ind_electricity, dm_amm_electricity, dm_agr_electricity, DM_tra,
                           dm_ind_hydrogen, dm_amm_hydrogen):

    def check_unit(dm, unit):
        for var in dm.col_labels['Variables']:
            if dm.units[var] != unit:
                raise ValueError(f'variable {var} does not have unit {unit}')


    #########################################################################
    # CalculationLeafs - Electricity demand - Appliances [GWh]
    #########################################################################

    #Tuto: Group Tree Merge appender & overwrite

    dm_bld_appliances = DM_bld['appliance']
    ay_x = np.nansum(dm_bld_appliances.array[...],axis=-1)
    dm_bld_appliances.add(ay_x, dim='Categories2', col_label='total')

    #########################################################################
    # CalculationLeafs - Electricity demand - Heating & cooling [GWh]
    #########################################################################
    dm_bld_cooling = DM_bld['cooling']
    dm_bld_heating = DM_bld['space-heating']
    # dm_bld_heatpump = DM_bld['heatpump']
    # TODO: To uncomment when profile is available (Speed-2-Zero)

    #############################################################################
    # CalculationLeafs - Electricity demand - Transport sectors [GWh] (no-profiles)
    #############################################################################
    dm_tra_elec = DM_tra['electricity']
    dm_demand_train = dm_tra_elec.filter_w_regex({'Variables': '.*rail'})
    dm_demand_road = dm_tra_elec.filter_w_regex({'Variables': '.*road'})

    #############################################################################
    # CalculationLeafs - Electricity demand - Other sectors [GWh] (no-profiles)
    #############################################################################
    dm_demand_other = dm_tra_elec.filter_w_regex({'Variables': '.*other'})

    # Tuto: Append matrices
    dm_demand_other.append(dm_agr_electricity, dim='Variables')
    dm_demand_other.append(dm_ind_electricity, dim='Variables')
    dm_demand_other.append(dm_amm_electricity, dim='Variables')
    dm_demand_other.groupby({'total-other': '.*'}, dim='Variables', regex=True, inplace=True)

    #############################################################################
    # CalculationLeafs - Electricity demand - Hydrogen [GWh]
    #############################################################################

    dm_demand_hydrogen = DM_tra['hydrogen'].copy()
    dm_demand_hydrogen.append(dm_amm_hydrogen, dim='Variables')
    dm_demand_hydrogen.append(dm_ind_hydrogen, dim='Variables')

    dm_demand_hydrogen.groupby({'total-hydrogen': '.*'}, dim='Variables', regex=True, inplace=True)

    #############################################################################
    # CalculationLeafs - Output Matrix Yearly Demand
    #############################################################################

    DM_yearly_demand = {
        'appliances-profile': dm_bld_appliances,
        'other-profile': dm_demand_other,
        'hydrogen-profile': dm_demand_hydrogen,
        'train-profile': dm_demand_train,
        'road-profile': dm_demand_road,
        'space-cooling': dm_bld_cooling,
        'space-heating': dm_bld_heating
    }

    return DM_yearly_demand


#######################################################################################################################
# CalculationTree - Power - Hourly demand
#######################################################################################################################
def hourly_demand_workflow(DM_yearly_demand, DM_demand_profiles, baseyear):

    ######################################
    # CalculationLeafs - Hourly profiles per sector [GWh]
    ######################################

    # Extract hourly data (demand profiles)
    dm_profile_hourly = DM_demand_profiles['non-residential-appliances']
    dm_profile_hourly.append(DM_demand_profiles['residential-appliances'],dim='Variables')
    dm_profile_hourly.append(DM_demand_profiles['train-profile'],dim='Variables')
    dm_profile_hourly.append(DM_demand_profiles['ev-charging-profile'], dim='Variables')
    dm_profile_hourly.append(DM_demand_profiles['non-residential-hotwater'], dim='Variables')
    dm_profile_hourly.append(DM_demand_profiles['residential-hotwater'], dim='Variables')
    dm_profile_hourly.append(DM_demand_profiles['non-residential-heat-profile'], dim='Variables')
    dm_profile_hourly.append(DM_demand_profiles['residential-heat-profile'], dim='Variables')
    dm_profile_hourly.append(DM_demand_profiles['non-residential-cooling-profile'], dim='Variables')
    dm_profile_hourly.append(DM_demand_profiles['residential-cooling-profile'], dim='Variables')

    ######################################
    # CalculationLeafs - Yearly demand per sector [GWh]
    ######################################

    # Buildings - Hotwater:
    dm_yearly_appliances = DM_yearly_demand['appliances-profile']
    dm_yearly_hotwater = dm_yearly_appliances.filter({'Categories2':['hot-water']})
    dm_yearly_hotwater.rename_col('hot-water','hotwater',dim='Categories2')
    dm_yearly_hotwater = dm_yearly_hotwater.flatten()
    dm_yearly_hotwater = dm_yearly_hotwater.flatten()

    # Buildings - Appliances
    dm_yearly_appliances.filter({'Categories2':['total']},inplace=True)
    dm_yearly_appliances.rename_col('total','appliances',dim='Categories2')
    dm_profile_yearly = dm_yearly_appliances.flatten()
    dm_profile_yearly = dm_profile_yearly.flatten()
    dm_profile_yearly.append(dm_yearly_hotwater, dim='Variables')

    # ToDo: Here I am: (2) rename & sort; (3) compute hourly demand (to check)
    # FixMe: total appliances may double count hot-water

    # Buildings - Cooling
    dm_yearly_cooling = DM_yearly_demand['space-cooling']
    dm_yearly_cooling = dm_yearly_cooling.flatten()
    dm_yearly_cooling = dm_yearly_cooling.flatten()
    dm_profile_yearly.append(dm_yearly_cooling, dim='Variables')

    # Buildings - Heating
    dm_yearly_heating = DM_yearly_demand['space-heating']
    dm_yearly_heating = dm_yearly_heating.flatten()
    dm_yearly_heating = dm_yearly_heating.flatten()
    dm_profile_yearly.append(dm_yearly_heating, dim='Variables')

    # Transport - Rail
    dm_yearly_rail = DM_yearly_demand['train-profile']
    dm_profile_yearly.append(dm_yearly_rail, dim='Variables')

    # Transport - Electric vehicle
    dm_yearly_road = DM_yearly_demand['road-profile']
    dm_profile_yearly.append(dm_yearly_road, dim='Variables')

    ######################################
    # CalculationLeafs - Hourly demand per sector [GWh]
    ######################################

    # Sorting
    dm_profile_yearly.rename_col_regex('bld_power-demand_', 'pow_', dim='Variables')
    dm_profile_yearly.rename_col_regex('tra_', 'pow_', dim='Variables')
    dm_profile_hourly.rename_col(['pow_rail', 'pow_road'], ['pow_power-demand_rail', 'pow_power-demand_road'], 'Variables')
    dm_profile_hourly.sort(dim='Variables')
    dm_profile_yearly.sort(dim='Variables')

    # Hourly demand
    idx_dd = dm_profile_yearly.idx
    ay_hourly_profile = dm_profile_yearly.array[:, idx_dd[baseyear]:, :, np.newaxis, np.newaxis, np.newaxis] \
                        * dm_profile_hourly.array[...]

    # Reshape of the output
    dm_profile_hourly.array = ay_hourly_profile
    for key in dm_profile_hourly.units.keys():
        dm_profile_hourly.units[key] = 'GWh'

    ######################################
    # CalculationLeafs - Hourly demand (total)[GWh]
    ######################################

    # Yearly demand "other" with no profiles
    dm_yearly_demand_other = DM_yearly_demand['other-profile'].copy()
    dm_yearly_demand_hydrogen = DM_yearly_demand['hydrogen-profile']
    dm_yearly_demand_other.append(dm_yearly_demand_hydrogen, dim='Variables')
    dm_yearly_demand_other = dm_yearly_demand_other.filter({'Variables':['total-hydrogen','total-other']})
    ay_yearly_demand_other = dm_yearly_demand_other.array[...].sum(axis=-1)
    dm_yearly_demand_other.add(ay_yearly_demand_other, dim='Variables', col_label='grand-total-other', unit='GWh')

    # Hourly profiles aggregation
    ay_hourly_profile_total = np.sum(dm_profile_hourly.array[...],axis=2)
    dm_profile_hourly.add(ay_hourly_profile_total,dim='Variables', col_label='pow_total-profile',unit='GWh')

    idx_year = dm_yearly_demand_other.idx
    idx_hourly = dm_profile_hourly.idx

    dm_yearly_demand_other.array = dm_yearly_demand_other.array/8760
    ay_total_hourly = dm_yearly_demand_other.array[:,idx_year[baseyear]:,idx_year['grand-total-other'],np.newaxis,np.newaxis,np.newaxis]\
                      +dm_profile_hourly.array[:,:,idx_hourly['pow_total-profile'],...]

    dm_profile_hourly.add(ay_total_hourly,dim='Variables', col_label='pow_total-hourly-demand',unit='GWh')
    dm_hourly_demand = dm_profile_hourly.filter({'Variables':['pow_total-hourly-demand']})

    return dm_hourly_demand


#######################################################################################################################
# CalculationTree - Power - Storage
#######################################################################################################################
def storage_workflow(dm_hourly_demand, dm_hourly_production):

    ######################################
    # CalculationLeafs - Hourly equilibrium [GWh]
    ######################################

    dm_hourly_equilibrium = dm_hourly_demand.copy()
    dm_hourly_equilibrium.append(dm_hourly_production, dim='Variables')
    dm_hourly_equilibrium.operation('pow_total-hourly-production', '-', 'pow_total-hourly-demand',dim="Variables",
                                    out_col='sto_hourly-equilibrium', unit='GWh')
    dm_hourly_equilibrium = dm_hourly_equilibrium.filter({'Variables': ['sto_hourly-equilibrium']})

    ######################################
    # CalculationLeafs - Hourly residual demand [GWh]
    ######################################

    ay_residual_demand = -np.where(np.isnan(dm_hourly_equilibrium.array), dm_hourly_equilibrium.array,
             np.fmin(dm_hourly_equilibrium.array, 0))

    ######################################
    # CalculationLeafs - Hourly residual supply [GWh]
    ######################################
    idx_he = dm_hourly_equilibrium.idx

    ay_residual_supply = np.where(np.isnan(dm_hourly_equilibrium.array), dm_hourly_equilibrium.array,
                                 np.fmax(dm_hourly_equilibrium.array, 0))
    dm_hourly_equilibrium.array[:, :, idx_he['sto_hourly-equilibrium'], np.newaxis, ...] = ay_residual_supply
    dm_hourly_equilibrium.rename_col('sto_hourly-equilibrium', 'sto_residual-supply', dim='Variables')
    dm_hourly_equilibrium.add(ay_residual_demand, dim='Variables', col_label='sto_residual-demand', unit='GWh')

    return dm_hourly_equilibrium


def emissions_workflow(dm_gross_production, cdm_emissions_fact):

    cdm_emissions_fact.sort('Categories2')
    dm_gross_production.filter({'Categories1': cdm_emissions_fact.col_labels['Categories2']}, inplace=True)
    dm_gross_production.sort('Categories1')

    ay_emissions = cdm_emissions_fact.array[np.newaxis, np.newaxis, :, :, :] \
                   * dm_gross_production.array[:, :, :, np.newaxis, :]/1000

    ay_emissions = np.moveaxis(ay_emissions, -2, -1)

    GHG_gas_list = cdm_emissions_fact.col_labels['Categories1']
    dm_emissions = DataMatrix.based_on(ay_emissions, format=dm_gross_production,
                                       change={'Variables': ['pow_emissions'], 'Categories2': GHG_gas_list},
                                       units={'pow_emissions': 'Mt'})

    return dm_emissions


def pow_refinery_interface(dm_gross_production):

    # !FIXME once we figure out the trade, we should include the electricty produced from fossil abroad
    dm_fossil_production = dm_gross_production.filter({'Categories1': ['coal', 'oil', 'gas', 'nuclear']})
    dm_fossil_production.add(0, dim='Categories1', col_label=['hydrogen'], dummy=True)
    dm_fossil_production.change_unit('pow_gross-yearly-production', 1e-3, old_unit='GWh', new_unit='TWh')
    dm_fossil_production.sort('Categories1')
    dm_fossil_production.rename_col('pow_gross-yearly-production', 'pow_energy-demand', dim='Variables')

    return dm_fossil_production


def pow_minerals_interface(dm_new_capacity, DM_yearly_demand):
    ##############################
    # Sum all electricity demand #
    ##############################

    # rename space-cooling and space-heating so that you can append them
    DM_yearly_demand['space-cooling'].rename_col('bld_power-demand', 'bld_power-demand_cooling', 'Variables')
    DM_yearly_demand['space-heating'].rename_col('bld_power-demand', 'bld_power-demand_heating', 'Variables')
    i = 0
    for key in DM_yearly_demand.keys():
        dm = DM_yearly_demand[key]
        # Sum over categories if any
        dim_cat_list = [dim for dim in dm.dim_labels if 'Categories' in dim]
        for dim_cat in reversed(dim_cat_list):
            dm.group_all(dim_cat, inplace=True)
        # Group all dm together
        if i == 0:
            dm_tot_elec = dm.copy()
        else:
            dm_tot_elec.append(dm, dim='Variables')
        i = i+1
    dm_tot_elec.groupby({'elc_electricity-demand_total': '.*'}, dim='Variables', regex=True, inplace=True)

    rename_dict = {
        'coal': 'energy-coal',
        'gas': 'energy-gas',
        'oil': 'energy-oil',
        'nuclear': 'energy-nuclear',
        'solar-pv': 'energy-pv',
        'solar-csp': 'energy-csp',
        'wind-offshore': 'energy-off-wind',
        'wind-onshore': 'energy-on-wind',
        'hydroelectric': 'energy-hydro',
        'geothermal': 'energy-geo',
        'marine': 'energy-marine'
    }
    name_in = []
    name_out = []
    for k, v in rename_dict.items():
        name_in.append(k)
        name_out.append(v)
    # !FIXME biogas and biomass demand not considered in minerals infrastructure
    dm_new_capacity.drop(dim='Categories1', col_label=['biogas', 'biomass'])
    dm_new_capacity.rename_col(name_in, name_out, 'Categories1')
    dm_new_capacity.sort('Categories1')
    dm_new_capacity.rename_col('pow_new-capacity', 'product-demand', 'Variables')

    # Dummy battery
    # !FIXME this is a dummy battery demand, you need to do storage
    ay_battery = dm_tot_elec.array*0
    dm_battery = DataMatrix.based_on(ay_battery, format=dm_tot_elec, change={'Variables': ['str_energy-battery']},
                                     units={'str_energy-battery': 'GW'})

    DM_minerals = {
        'battery': dm_battery,
        'electricity-demand': dm_tot_elec,
        'energy': dm_new_capacity,
    }

    return DM_minerals


def pow_TPE_interface(dm_production, dm_decommission):
    # From GWh to TWh
    dm_production.change_unit('pow_gross-yearly-production', factor=1e-3, old_unit='GWh', new_unit='TWh')

    df = dm_production.write_df()
    df2 = dm_decommission.write_df()
    df = pd.concat([df, df2.drop(columns=['Country', 'Years'])], axis=1)

    return df
#######################################################################################################################
# CoreModule - Power
#######################################################################################################################

def power(lever_setting, years_setting, interface=Interface()):
    baseyear = years_setting[1]
    current_file_directory = os.path.dirname(os.path.abspath(__file__))


    power_data_file = os.path.join(current_file_directory,
                                        '../_database/data/datamatrix/geoscale/power.pickle')
    dm_capacity, dm_ccus, cdm_const, DM_production_profiles, DM_demand_profiles = read_data(power_data_file, lever_setting)

    cntr_list = dm_capacity.col_labels['Country']

    if interface.has_link(from_sector='climate', to_sector='power'):
        dm_climate = interface.get_link(from_sector='climate', to_sector='power')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing climate to power interface')
        dm_climate = simulate_climate_to_power_input()
        dm_climate.filter({'Country': cntr_list}, inplace=True)

    if interface.has_link(from_sector='agriculture', to_sector='power'):
        dm_agr_electricity = interface.get_link(from_sector='agriculture', to_sector='power')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing agriculture to power interface')
        dm_agr_electricity = simulate_agriculture_to_power_input()
        dm_agr_electricity.filter({'Country': cntr_list}, inplace=True)

    if interface.has_link(from_sector='buildings', to_sector='power'):
        DM_bld = interface.get_link(from_sector='buildings', to_sector='power')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing buildings to power interface')
        DM_bld = simulate_buildings_to_power_input()
        for key in DM_bld.keys():
            DM_bld[key].filter({'Country': cntr_list}, inplace=True)

    if interface.has_link(from_sector='industry', to_sector='power'):
        DM_industry = interface.get_link(from_sector='industry', to_sector='power')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing industry to power interface')
        DM_industry = simulate_industry_to_power_input()
        for key in DM_industry.keys():
            DM_industry[key].filter({'Country': cntr_list}, inplace=True)

    if interface.has_link(from_sector='ammonia', to_sector='power'):
        DM_ammonia = interface.get_link(from_sector='ammonia', to_sector='power')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing ammonia to power interface')
        DM_ammonia = simulate_ammonia_to_power_input()
        for key in DM_ammonia.keys():
            DM_ammonia[key].filter({'Country': cntr_list}, inplace=True)

    if interface.has_link(from_sector='transport', to_sector='power'):
        DM_tra = interface.get_link(from_sector='transport', to_sector='power')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing transport to power interface')
        DM_tra = simulate_transport_to_power_input()
        for key in DM_tra.keys():
            DM_tra[key].filter({'Country': cntr_list}, inplace=True)

    # To send to TPE (result run)
    dm_capacity, dm_fb_capacity, dm_production_np, dm_production_p =\
        yearly_production_workflow(dm_climate, dm_capacity, dm_ccus, cdm_const)
    dm_hourly_production = hourly_production_workflow(dm_production_np, dm_production_p, DM_production_profiles, baseyear)

    # TUTO give dm_ev_hourly as input to yearly_demand_workflow
    DM_yearly_demand = yearly_demand_workflow(DM_bld, DM_industry['electricity'], DM_ammonia['electricity'],
                                              dm_agr_electricity, DM_tra, DM_industry['hydrogen'], DM_ammonia['hydrogen'])

    dm_gross_production = dm_capacity.filter({'Variables': ['pow_gross-yearly-production']})
    dm_emissions = emissions_workflow(dm_gross_production.copy(), cdm_const['emission-factors'])

    dm_hourly_demand = hourly_demand_workflow(DM_yearly_demand.copy(), DM_demand_profiles, baseyear)

    dm_hourly_equilibrium = storage_workflow(dm_hourly_demand, dm_hourly_production)

    interface.add_link(from_sector='power', to_sector='emissions', dm=dm_emissions.flatten().flatten())

    dm_refinery = pow_refinery_interface(dm_gross_production)
    interface.add_link(from_sector='power', to_sector='oil-refinery', dm=dm_refinery)
    # same number of arg than the return function

    dm_new_capacity = dm_capacity.filter({'Variables': ['pow_new-capacity']})
    DM_minerals = pow_minerals_interface(dm_new_capacity, DM_yearly_demand)
    interface.add_link(from_sector='power', to_sector='minerals', dm=DM_minerals)
    # concatenate all results to df
    #results_run = dm_capacity
    # FIXME: I think this is wrong
    dm_decommission = dm_capacity.filter({'Variables': ['pow_existing-capacity']})
    results_run = pow_TPE_interface(dm_gross_production, dm_decommission)

    return results_run


#######################################################################################################################
# LocalRun - Power
#######################################################################################################################

def local_power_run():
    # Function to run only transport module without converter and tpe
    years_setting = [1990, 2015, 2050, 5]
    f = open('../config/lever_position.json')
    lever_setting = json.load(f)[0]

    global_vars = {'geoscale': 'Switzerland'}
    filter_geoscale(global_vars)

    results_run = power(lever_setting, years_setting)


    return results_run

# __file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/model/minerals_module.py"
# # database_from_csv_to_datamatrix()
#results_run = local_power_run()



