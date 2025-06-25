# Import Python packages
import pandas as pd
import pickle
import json
import os
import numpy as np
import warnings
import time

# Import classes
from model.common.data_matrix_class import DataMatrix
from model.common.interface_class import Interface
from model.common.constant_data_matrix_class import ConstantDataMatrix

# Import functions
from model.common.io_database import read_database, read_database_fxa, read_database_w_filter, update_database_from_db_old
from model.common.io_database import read_database_to_ots_fts_dict, read_database_to_ots_fts_dict_w_groups
from model.common.auxiliary_functions import read_level_data, cost

warnings.simplefilter("ignore")


def init_years_lever():
    # function that can be used when running the module as standalone to initialise years and levers
    years_setting = [1990, 2015, 2050, 5]
    f = open('../config/lever_position.json')
    lever_setting = json.load(f)[0]
    return years_setting, lever_setting

def dummy_countries_fxa():
    file = 'district-heating_fixed-assumptions'
    df_db = read_database_fxa(file, db_format=True)
    df_db_vd = df_db.loc[df_db['geoscale'] == 'Switzerland']
    df_db_vd['geoscale'] = 'Vaud'
    df_db_eu = df_db.loc[df_db['geoscale'] == 'Germany']
    df_db_eu['geoscale'] = 'EU27'
    update_database_from_db_old(file, df_db_vd)
    update_database_from_db_old(file, df_db_eu)
    return

def dummy_countries_power():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    file = 'EUCalc-interface_from-electricity_supply-to-district-heating.xlsx'
    file_path = os.path.join(current_file_directory, '../_database/data/xls/', file)
    df = pd.read_excel(file_path, sheet_name="default")
    vaud = 1
    eu27 = 1
    paris = 1
    if vaud:
        df_vd = df.loc[df['Country'] == 'Switzerland']
        df_vd['Country'] = 'Vaud'
        df_vd['elc_heat-supply-CHP_bio[TWh]'] = df_vd['elc_heat-supply-CHP_bio[TWh]']*0.1
        df_vd['elc_heat-supply-CHP_fossil[TWh]'] = df_vd['elc_heat-supply-CHP_bio[TWh]']*0.1
        df = pd.concat([df, df_vd], axis=0)
    if eu27:
        df_eu = df.loc[df['Country'] == 'Germany']
        df_eu['Country'] = 'EU27'
        df = pd.concat([df, df_eu], axis=0)
    if paris:
        df_p = df.loc[df['Country'] == 'France']
        df_p['Country'] = 'Paris'
        df_p['elc_heat-supply-CHP_bio[TWh]'] = df_p['elc_heat-supply-CHP_bio[TWh]']*0.19
        df_p['elc_heat-supply-CHP_fossil[TWh]'] = df_p['elc_heat-supply-CHP_bio[TWh]']*0.19
        df = pd.concat([df, df_p], axis=0)
    file = 'All-Countries-interface_from-power-to-district-heating.xlsx'
    file_path = os.path.join(current_file_directory, '../_database/data/xls/', file)
    df.to_excel(file_path, sheet_name="default", index=False)
    return

def dummy_countries_industry():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    file = 'EUCalc-interface_from-industry-to-district-heating.xlsx'
    file_path = os.path.join(current_file_directory, '../_database/data/xls/', file)
    df = pd.read_excel(file_path, sheet_name="default")
    vaud = 1
    eu27 = 1
    paris = 1
    if vaud:
        df_vd = df.loc[df['Country'] == 'Switzerland']
        df_vd['Country'] = 'Vaud'
        df = pd.concat([df, df_vd], axis=0)
    if eu27:
        df_eu = df.loc[df['Country'] == 'Germany']
        df_eu['Country'] = 'EU27'
        df = pd.concat([df, df_eu], axis=0)
    if paris:
        df_p = df.loc[df['Country'] == 'France']
        df_p['Country'] = 'Paris'
        df = pd.concat([df, df_p], axis=0)
    file = 'All-Countries-interface_from-industry-to-district-heating.xlsx'
    file_path = os.path.join(current_file_directory, '../_database/data/xls/', file)
    df.to_excel(file_path, sheet_name="default", index=False)
    return


def database_from_csv_to_datamatrix():

    # Read database
    # Set years range
    years_setting, lever_setting = init_years_lever()
    startyear = years_setting[0]
    baseyear = years_setting[1]
    lastyear = years_setting[2]
    step_fts = years_setting[3]
    years_ots = list(np.linspace(start=startyear, stop=baseyear, num=(baseyear-startyear)+1).astype(int))
    years_fts = list(np.linspace(start=baseyear+step_fts, stop=lastyear, num=int((lastyear-baseyear)/step_fts)).astype(int))
    years_all = years_ots + years_fts

    dict_ots = {}
    dict_fts = {}

    # Read heatcool-efficiency share
    file = 'buildings_heatcool-efficiency'
    lever = 'heatcool-efficiency'
    dict_ots, dict_fts = read_database_to_ots_fts_dict(file, lever, num_cat=1, baseyear=baseyear, years=years_all,
                                                       dict_ots=dict_ots, dict_fts=dict_fts)

    file = 'buildings_heatcool-technology-fuel'
    lever = 'heatcool-technology-fuel'
    dict_ots, dict_fts = read_database_to_ots_fts_dict_w_groups(file, lever, num_cat_list=[1], baseyear=baseyear,
                                                                years=years_all, dict_ots=dict_ots, dict_fts=dict_fts,
                                                                column='eucalc-name',
                                                                group_list=['bld_heat-district-technology'])

    # Read fixed assumptions & create dict_fxa
    file = 'district-heating_fixed-assumptions'
    dict_fxa = {}
    # this is just a dataframe of zeros
    # df = read_database_fxa(file, filter_dict={'eucalc-name': 'bld_CO2-factors-GHG'})
    df = read_database_fxa(file, filter_dict={'eucalc-name': 'bld_district-capacity_'})
    dm_dhg_capacity = DataMatrix.create_from_df(df, num_cat=1)
    dict_fxa['dhg-capacity'] = dm_dhg_capacity
    df = read_database_fxa(file, filter_dict={'eucalc-name': 'bld_district-fixed-assumptions_'})
    dm_dhg_replacement = DataMatrix.create_from_df(df, num_cat=0)
    dict_fxa['dhg-replacement-rate'] = dm_dhg_replacement

    df = read_database_fxa('costs_fixed-assumptions')
    dm_cost = DataMatrix.create_from_df(df, num_cat=0)
    dict_fxa['cost'] = dm_cost

    cdm_const = ConstantDataMatrix.extract_constant('interactions_constants', pattern='cp_emission-factor_.*',
                                                    num_cat=2)

    cdm_cost = ConstantDataMatrix.extract_constant('interactions_constants', pattern='cp_cost-bld_', num_cat=0)
    cdm_cost = cdm_cost.filter_w_regex({'Variables': '.*_dh_.*'})
    cdm_cost.rename_col_regex('_shared-infrastructures', '', dim='Variables')
    cdm_cost.rename_col_regex('_dh', '', dim='Variables')
    cdm_cost.rename_col_regex('cp_cost-bld_', '', dim='Variables')
    cdm_cost.rename_col_regex('-woodlogs', '', dim='Variables')

    dict_const = {
        'emission-factor': cdm_const,
        'cost': cdm_cost
    }

    # group all datamatrix in a single structure
    DM_district_heating = {
        'fxa': dict_fxa,
        'fts': dict_fts,
        'ots': dict_ots,
        'constant': dict_const
    }

    # write datamatrix to pickle
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, '../_database/data/datamatrix/district-heating.pickle')
    with open(f, 'wb') as handle:
        pickle.dump(DM_district_heating, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return


def read_data(data_file, lever_setting):

    with open(data_file, 'rb') as handle:
        DM_district_heating = pickle.load(handle)

    dm_rr = DM_district_heating['fxa']['dhg-replacement-rate']
    dm_capacity = DM_district_heating['fxa']['dhg-capacity']
    dm_price = DM_district_heating['fxa']['cost']

    # Read fts based on lever_setting
    DM_ots_fts = read_level_data(DM_district_heating, lever_setting)

    dm_dhg = DM_ots_fts['heatcool-efficiency']
    dm_dhg.append(DM_ots_fts['heatcool-technology-fuel']['bld_heat-district-technology'], dim='Variables')

    cdm_emission = DM_district_heating['constant']['emission-factor']
    cdm_cost = DM_district_heating['constant']['cost']

    return dm_dhg, dm_rr, dm_capacity, dm_price, cdm_emission, cdm_cost


def simulate_power_to_district_heating_input():

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    file = 'All-Countries-interface_from-power-to-district-heating.xlsx'
    file_path = os.path.join(current_file_directory, '../_database/data/xls/', file)
    df = pd.read_excel(file_path, sheet_name="default")
    dm_pow = DataMatrix.create_from_df(df, num_cat=0)
    # Filter heat supply from CHP, sum bio + fossil, rename
    dm_pow.operation('elc_heat-supply-CHP_bio', '+', 'elc_heat-supply-CHP_fossil',
                     out_col='dhg_energy-demand_contribution_CHP', unit='TWh', nansum=True)
    dm_emissions = dm_pow.filter({'Variables': ['elc_CO2-heat_specific']}, inplace=False)
    dm_pow.filter({'Variables': ['dhg_energy-demand_contribution_CHP']}, inplace=True)
    DM_pow = {
        'wf_emissions': dm_emissions,
        'wf_energy': dm_pow
    }
    return DM_pow


def simulate_industry_to_district_heating_input():

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    file = 'All-Countries-interface_from-industry-to-district-heating.xlsx'
    file_path = os.path.join(current_file_directory, '../_database/data/xls/', file)
    df = pd.read_excel(file_path, sheet_name="default")
    dm_ind = DataMatrix.create_from_df(df, num_cat=0)
    dm_ind.rename_col('ind_supply_heat-waste', 'dhg_energy-demand_contribution_heat-waste', dim='Variables')

    return dm_ind


def simulate_buildings_to_district_heating_input():

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    file = 'All-Countries-interface_from-buildings-to-district-heating.xlsx'
    file_path = os.path.join(current_file_directory, '../_database/data/xls/', file)
    df = pd.read_excel(file_path)
    dm_bld = DataMatrix.create_from_df(df, num_cat=0)
    # Electricity input is not used
    # dm_elec = dm_bld.filter_w_regex({'Variables': 'bld_electricity-demand.*'})
    # dm_elec.deepen_twice()
    dm_dhg = dm_bld.filter_w_regex({'Variables': 'bld_district-heating.*'}, inplace=False)
    dm_dhg.deepen()
    dm_pipe = dm_bld.filter({'Variables': ['bld_new_dh_pipes']})
    dm_pipe.deepen()

    DM_bld = {
        'heat': dm_dhg,
        'pipe': dm_pipe
    }

    return DM_bld


def dhg_energy_demand_workflow(dm_dhg, dm_bld, dm_pow, dm_ind):

    # technology-fuel-% * efficiency-by-fuel
    dm_dhg.operation('bld_heat-district-efficiency', '*', 'bld_heat-district-technology-fuel',
                     out_col='dhg_energy-need-technology-share', unit='%')
    idx_d = dm_dhg.idx
    # Normalise the energy-need
    dm_dhg.array[:, :, idx_d['dhg_energy-need-technology-share'], :] \
        = dm_dhg.array[:, :, idx_d['dhg_energy-need-technology-share'], :]\
        / np.nansum(dm_dhg.array[:, :, idx_d['dhg_energy-need-technology-share'], :], axis=-1, keepdims=True)
    del idx_d

    # Sum space heating residential + non-residential demand
    dm_bld.group_all(dim='Categories1')
    # Compute energy demand by fuel type
    idx_b = dm_bld.idx
    idx_t = dm_dhg.idx
    # note: from GWh to TWh
    arr_energy_by_tech = dm_dhg.array[:, :, idx_t['dhg_energy-need-technology-share'], :] \
                         * dm_bld.array[:, :, idx_b['bld_district-heating-space-heating-supply'], np.newaxis]/1000
    dm_dhg.add(arr_energy_by_tech, dim='Variables', col_label='dhg_energy-need', unit='TWh')

    # !FIXME: we have already multiplied by the efficiency earlier, check if this is correct
    # energy-need-by-fuel * efficiency-by-fuel
    dm_dhg.operation('dhg_energy-need', '*', 'bld_heat-district-efficiency', out_col='dhg_energy-demand-prelim', unit='TWh')
    dm_dhg.filter({'Variables': ['dhg_energy-demand-prelim']}, inplace=True)
    del dm_bld, arr_energy_by_tech, idx_b, idx_t

    # Sum all energy need
    dm_all = dm_dhg.group_all(dim='Categories1', inplace=False)

    # Share of energy need not from waste
    # ! FIXME why are we not removing the one from CHP ?
    dm_all.append(dm_ind, dim='Variables')
    dm_all.append(dm_pow, dim='Variables')
    del dm_ind, dm_pow

    # Energy share not from waste: max(0, (preliminary heat demand - waste heat supply)/preliminary heat demand)
    idx_a = dm_all.idx
    arr_share_non_waste = np.maximum(0, ((dm_all.array[:, :, idx_a['dhg_energy-demand-prelim']]
                                         - dm_all.array[:, :, idx_a['dhg_energy-demand_contribution_heat-waste']])
                                         / dm_all.array[:, :, idx_a['dhg_energy-demand-prelim']]))
    dm_all.add(arr_share_non_waste, dim='Variables', col_label='dhg_energy-demand-share_heat-district-addition', unit='%')
    del arr_share_non_waste, idx_a

    # Multiply preliminary demand by share of energy not from waste
    idx_t = dm_all.idx
    idx_g = dm_dhg.idx
    arr = dm_dhg.array[:, :, idx_g['dhg_energy-demand-prelim'], :] \
          * dm_all.array[:, :, idx_t['dhg_energy-demand-share_heat-district-addition'], np.newaxis]
    dm_dhg.add(arr, dim='Variables', col_label='dhg_energy-demand', unit='TWh')
    dm_dhg.filter({'Variables': ['dhg_energy-demand']}, inplace=True)
    dm_all.filter({'Variables': ['dhg_energy-demand_contribution_CHP', 'dhg_energy-demand_contribution_heat-waste']}, inplace=True)
    del arr, idx_g, idx_t

    # Rename for TPE
    dm_all.rename_col('dhg_energy-demand_contribution_CHP', 'dhg_energy-demand_heat-co-product_from-power', dim='Variables')
    dm_all.rename_col('dhg_energy-demand_contribution_heat-waste', 'dhg_energy-demand_heat-co-product_from-industry', dim='Variables')
    dm_all.units['dhg_energy-demand_heat-co-product_from-industry'] = 'TWh'
    dm_dhg.rename_col('dhg_energy-demand', 'dhg_energy-demand_added-district-heat', dim='Variables')
    DM_energy_out = {}
    DM_energy_out['TPE'] = {
        'heat-co-product': dm_all.copy(),
        'added-district-heating': dm_dhg.copy()
    }
    DM_energy_out['wf_emissions'] = {
        'heat-co-product': dm_all,
        'added-district-heat': dm_dhg
    }
    DM_energy_out['wf_costs'] = dm_dhg.copy()

    return DM_energy_out


def dhg_emissions_workflow(DM_energy, dm_CO2_coef, cdm_emission_fact):
    # cdm_emission_fact: contains emission factor of GHG for various heat sources
    # dm_CO2_coef: contains CO2 emission factor for CHP/waste from the power module
    # DM_energy: comes from the energy workflow and contains the district heating by fuel type and the co-generated one
    #            (i.e. waste and CHP)

    ### ADDED-DISTRICT-HEAT ###

    dm_heat_by_fuel = DM_energy['added-district-heat']
    cdm_emission_fact = cdm_emission_fact.filter({'Categories2': dm_heat_by_fuel.col_labels['Categories1']})
    # Put 0 instead of nan in emission factor
    cdm_emission_fact.array = np.nan_to_num(cdm_emission_fact.array)
    # Mutiply energy-demand by fuel by emission factor by fuel and GHG
    idx_h = dm_heat_by_fuel.idx
    idx_c = cdm_emission_fact.idx
    arr = dm_heat_by_fuel.array[:, :, idx_h['dhg_energy-demand_added-district-heat'], np.newaxis, :] \
          * cdm_emission_fact.array[np.newaxis, np.newaxis, idx_c['cp_emission-factor'], :, :]

    col_labels = {
        'Country': dm_heat_by_fuel.col_labels['Country'],
        'Years': dm_heat_by_fuel.col_labels['Years'],
        'Variables': ['dhg_emissions'],
        'Categories1': cdm_emission_fact.col_labels['Categories1'],
        'Categories2': cdm_emission_fact.col_labels['Categories2']
    }
    dm_emissions = DataMatrix(col_labels, units={'dhg_emissions': 'Mt'})
    dm_emissions.array = arr[:, :, np.newaxis, :, :]

    del cdm_emission_fact, arr, idx_c, idx_h, col_labels, dm_heat_by_fuel

    # Group emissions by GHG + Rename for TPE
    dm_emissions_by_GHG = dm_emissions.group_all(dim='Categories2', inplace=False)
    dm_emissions_by_GHG.rename_col('dhg_emissions', 'dhg_emissions_added-district-heat', dim='Variables')

    ### HEAT-CO-PRODUCT ###

    # For heat-co-product, i.e. waste and CHP
    dm_co_product = DM_energy['heat-co-product']
    dm_co_product.deepen()
    idx_e = dm_CO2_coef.idx
    idx_c = dm_co_product.idx
    arr = dm_co_product.array[:, :, idx_c['dhg_energy-demand_heat-co-product'], :] \
          * dm_CO2_coef.array[:, :, idx_e['elc_CO2-heat_specific'], np.newaxis]
    dm_co_product.add(arr, dim='Variables', col_label='dhg_emissions-CO2_heat-co-product', unit='Mt')
    dm_co_product.filter({'Variables': ['dhg_emissions-CO2_heat-co-product']}, inplace=True)

    # For emissions
    dm_emissions = dm_emissions.flatten().flatten()
    dm_emissions.rename_col_regex('dhg_emissions_', 'dhg_emissions-', 'Variables')

    DM_emissions_out = {
        'TPE': {'added-district-heat': dm_emissions_by_GHG, 'heat-co-product': dm_co_product},
        'emissions': dm_emissions
    }

    return DM_emissions_out


def dhg_costs_workflow(dm_fuel, dm_pipes, dm_cap, dm_rr, dm_price, cdm_cost, baseyear):

    # Capacity factor from daily to yearly
    dm_cap.array = dm_cap.array * 8760
    dm_fuel.append(dm_cap, dim='Variables')
    # capacity [TW] = energy-demand [TWh] /capacity-factor
    dm_fuel.operation('dhg_energy-demand_added-district-heat', '/', 'bld_district-capacity', out_col='dhg_capacity_dh', unit='TW')

    # capacity[TW] = capacity[TW] x replacement-rate[%]
    idx_r = dm_rr.idx
    idx_f = dm_fuel.idx
    # Update capacity value by accounty for replacement rate
    dm_fuel.array[:, :, idx_f['dhg_capacity_dh'], :] = dm_fuel.array[:, :, idx_f['dhg_capacity_dh'], :] \
                                                       * dm_rr.array[:, :, idx_r['bld_district-fixed-assumptions_replacement-rate'], np.newaxis]
    dm_activity = dm_fuel.filter({'Variables': ['dhg_capacity_dh']})

    # ! FIXME: add cost calculation here
    cdm_pipe_cost = cdm_cost.filter_w_regex({'Variables': '.*pipes'})
    cdm_pipe_cost.deepen()
    cdm_cost.drop(dim='Variables', col_label='.*pipes')
    cdm_cost.deepen()
    cdm_cost = cdm_cost.filter({'Categories1': dm_activity.col_labels['Categories1']})

    # Change cost units from EUR/kW to EUR/TW
    idx = cdm_cost.idx
    cdm_cost.array[idx['capex-baseyear'], :] = cdm_cost.array[idx['capex-baseyear'], :]*1000
    cdm_cost.units['capex-baseyear'] = 'EUR/TW'

    dm_capex_heat = cost(dm_activity=dm_activity, dm_price_index=dm_price, cdm_cost=cdm_cost, cost_type='capex',
                         baseyear=baseyear, unit_cost=False)

    # Pipes cost
    # pipes [km] = energy [TWh] * 118300
    # new pipes [km] = pipes [km] * 0.0125
    # dm_energy = dm_fuel.filter({'Variables': ['dhg_energy-demand_added-district-heat']})
    # dm_energy.group_all(dim='Categories1')
    # dm_energy.array = dm_energy.array * 118300 * 0.0125
    # dm_energy.rename_col('dhg_energy-demand_added-district-heat', 'bld_new_dh_pipes', dim='Variables')
    # dm_energy.units['bld_new_dh_pipes'] = 'km'
    # dm_pipes = dm_energy
    # dm_pipes.deepen()

    dm_capex_pipe = cost(dm_activity=dm_pipes, dm_price_index=dm_price, cdm_cost=cdm_pipe_cost, cost_type='capex',
                         baseyear=baseyear, unit_cost=False)

    # Rename for employment
    dm_capex_pipe.rename_col('capex', 'bld_capex_construction', dim='Variables')
    dm_capex_heat.rename_col('capex', 'bld_capex_district-heat-generation-plants', dim='Variables')
    dm_capex_pipe.append(dm_pipes, dim='Variables')
    dm_capex_pipe.rename_col('bld_new_dh', 'bld_new_district-heat-generation-plants', dim='Variables')

    DM_cost_out = {
        'employment': {'cost-capacity': dm_capex_heat, 'pipe': dm_capex_pipe},
    }
    return DM_cost_out


def dhg_TPE_interface(DM_energy, DM_emissions):

    # Compute total energy demand
    dm_energy_added = DM_energy['added-district-heating'].copy()
    dm_energy_added.groupby({'fossil': ['gas-ff-natural', 'liquid-ff-heatingoil', 'solid-ff-coal'] ,
                           'renewable': ['heat-ambient', 'heat-geothermal', 'heat-solar', 'solid-bio']}, dim='Categories1', inplace=True)
    dm_energy_added = dm_energy_added.flatten()

    dm_energy_coproduct = DM_energy['heat-co-product'].groupby({'dhg_energy-demand_co-product-heat': 'dhg_energy-demand_heat-co-product_.*'},
                                                               dim='Variables', inplace=False, regex=True)
    dm_energy_added.append(dm_energy_coproduct, dim='Variables')
    dm_tot_energy = dm_energy_added.groupby({'dhg_energy-demand_heat-district': '.*'}, dim='Variables', regex=True, inplace=False)

    # Merge energy output
    dm_energy = DM_energy['added-district-heating'].flatten()
    dm_energy.append(DM_energy['heat-co-product'], dim='Variables')
    # Merge emission output
    dm_emissions = DM_emissions['added-district-heat'].flatten()
    dm_emissions.append(DM_emissions['heat-co-product'].flatten(), dim='Variables')

    dm_energy.append(dm_emissions, dim='Variables')

    df = dm_energy.write_df()
    df2 = dm_tot_energy.write_df()
    df3 = dm_energy_added.write_df()

    df = pd.concat([df, df2.drop(columns=['Country', 'Years'])], axis=1)
    df = pd.concat([df, df3.drop(columns=['Country', 'Years'])], axis=1)

    return df


def district_heating(lever_setting, years_setting, interface=Interface()):

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    district_heating_data_file = os.path.join(current_file_directory,
                                              '../_database/data/datamatrix/geoscale/district-heating.pickle')
    dm_dhg, dm_rr, dm_capacity, dm_price, cdm_emission, cdm_cost = read_data(district_heating_data_file, lever_setting)
    cntr_list = dm_capacity.col_labels['Country']

    if interface.has_link(from_sector='power', to_sector='district-heating'):
        DM_pow = interface.get_link(from_sector='power', to_sector='district-heating')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing power to district-heating interface')
        DM_pow = simulate_power_to_district_heating_input()
        for key in DM_pow.keys():
            DM_pow[key].filter({'Country': cntr_list}, inplace=True)

    if interface.has_link(from_sector='industry', to_sector='district-heating'):
        dm_ind = interface.get_link(from_sector='industry', to_sector='district-heating')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing industry to district-heating interface')
        dm_ind = simulate_industry_to_district_heating_input()
        dm_ind.filter({'Country': cntr_list}, inplace=True)

    if interface.has_link(from_sector='buildings', to_sector='district-heating'):
        DM_bld = interface.get_link(from_sector='buildings', to_sector='district-heating')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing buildings to district-heating interface')
        DM_bld = simulate_buildings_to_district_heating_input()
        for key in DM_bld.keys():
            DM_bld[key].filter({'Country': cntr_list}, inplace=True)

    # Input: raw energy demand by fuel; efficiency by fuel; heat co-generation from waste/CHP
    # Output: Energy demand district heating by fuel-type + waste/CHP co-generation
    DM_energy_out = dhg_energy_demand_workflow(dm_dhg, DM_bld['heat'], DM_pow['wf_energy'], dm_ind)
    # Input: Energy demand district-heating by fuel + waste/CHP; GHG emission factors; CO2 emission factor for waste/CHP
    # Output: emissions by GHG for district-heating by fuel, emissions for CO2 for waste and CHP heat
    DM_emissions_out = dhg_emissions_workflow(DM_energy_out['wf_emissions'], DM_pow['wf_emissions'], cdm_emission)
    # Input: district-capacity (by fuel), replacement-rate, energy-demand (by fuel)
    # Output:
    # baseyear = years_setting[1]
    # DM_cost_out = dhg_costs_workflow(DM_energy_out['wf_costs'], DM_bld['pipe'], dm_capacity, dm_rr, dm_price, cdm_cost, baseyear)

    # Emissions interface
    interface.add_link(from_sector='district-heating', to_sector='emissions', dm=DM_emissions_out['emissions'])

    #!FIXME: some dhg output to TPE are computed during the 'cube' but it is not working,...
    # ....fix this by computing the variables directly here
    results_run = dhg_TPE_interface(DM_energy_out['TPE'], DM_emissions_out['TPE'])

    return results_run


def district_heating_local_run():
    # Function to run module as stand alone without other modules/converter or TPE
    years_setting, lever_setting = init_years_lever()
    district_heating(lever_setting, years_setting)
    return

# dummy_countries_fxa()
# database_from_csv_to_datamatrix()
#district_heating_local_run()




