import pandas as pd

from model.common.data_matrix_class import DataMatrix
from model.common.interface_class import Interface
from model.common.constant_data_matrix_class import ConstantDataMatrix
from model.common.io_database import read_database, read_database_fxa, read_database_to_ots_fts_dict
from model.common.auxiliary_functions import compute_stock, filter_geoscale
from model.common.auxiliary_functions import read_level_data, create_years_list, linear_fitting
import pickle
import json
import os
import numpy as np
import time


def read_data(data_file, lever_setting):

    with open(data_file, 'rb') as handle:
        DM_forestry = pickle.load(handle)

    dict_const = DM_forestry['constant']
    dict_fxa = DM_forestry['fxa']

    # Split of wood used for each use (e.g., share of coniferous used for fuelwood)
    dm_wood_type_split_per_use = DM_forestry['fxa']['coniferous-share']
    dm_wood_type_split_per_use.filter(({'Variables': ['fst_production-t_share']}))
    cdm_wood_conversion = DM_forestry['constant']['wood-category-conversion-factors']
    cdm_wood_yields = DM_forestry['constant']['industry-byproducts']
    cdm_wood_density = DM_forestry['constant']['wood-density']
    cdm_energy_density = DM_forestry['constant']['energy-density']
    cdm_other_industrial_wood_demand = DM_forestry['fxa']['any-other-industrial-wood']

    # Read fts based on lever_setting
    DM_ots_fts = read_level_data(DM_forestry, lever_setting)

    DM_wood_conversion = {'coniferous-share':dm_wood_type_split_per_use,
                          'industry-yields':cdm_wood_yields,
                          'wood-yields':cdm_wood_conversion,
                          'wood-density':cdm_wood_density,
                          'any-other-wood-demand':cdm_other_industrial_wood_demand,
                          'energy-density': cdm_energy_density}

    DM_forestry = {'fxa': dict_fxa,
                   'constant': dict_const}

    return DM_forestry, DM_wood_conversion, DM_ots_fts

#####################################################################################################################
#####################################################################################################################
# Calculation Tree - Forestry:
#####################################################################################################################
#####################################################################################################################

#####################################################################################################################
# Calculation Tree - Wood demand in m3:
#####################################################################################################################
def wood_demand_m3 (dm_wood_demand, DM_wood_conversion):

    #################################################################################################################
    # Provides the Wood demand in m3
    # 1. Wood product in tonnes (e.g., timber) to express in wood category (e.g., saw-logs) in tonnes given the yields
    # 2. Wood category to split between soft/hard wood (coniferous and non-coniferous)
    # 3. Express the wood demand in m3 given the hard/soft wood density
    # 4. Compute the wood-fuel supply from industrial byproducts
    #################################################################################################################

    # Wood demand from industry [t]
    dm_wood_demand_industry_m3 = dm_wood_demand

    # Wood to wood product yields [t]
    dm_wood_yields = DM_wood_conversion['wood-yields']

    # Conversion from wood products to wood [t]
    ay_wood_conversion = dm_wood_demand_industry_m3[:, :, :, :] *\
                         dm_wood_yields[np.newaxis, np.newaxis,:,:]
    dm_wood_demand_industry_m3.add(ay_wood_conversion, col_label='wood-category-eq', dim='Variables', unit='t')
    # Rename to match the wood category names
    dm_wood_demand_industry_m3.rename_col(
        ['other-industrial', 'pulp', 'timber'],
        ['industrial-wood','any-other-wood','sawlogs'],
        dim='Categories1')

    # Adding the wood demand for calibration / uncovered by the model
    dm_fxa = DM_wood_conversion['any-other-wood-demand']
    dm_fxa.rename_col('fxa_any-other-wood', 'wood-category-eq', dim='Variables')
    arr = dm_fxa.array
    dm_fxa.add(arr, dim='Variables', col_label='ind_wood', unit='t')
    dm_wood_demand_industry_m3.append(dm_fxa, dim='Categories1')
    dm_wood_demand_industry_m3.groupby({'any-other-wood':['any-other-wood','any-other-industrial-demand']}, dim='Categories1',inplace=True)


    # Split between coniferous and non-coniferous wood
    dm_wood_split = DM_wood_conversion['coniferous-share'].filter_w_regex({'Variables': 'fst_production-t_share'})
    dm_wood_split.drop(dim='Categories1', col_label='woodfuel')
    dm_wood_demand_t = dm_wood_split
    ay_wood_split = dm_wood_demand_industry_m3[:, :, 'wood-category-eq' , :, np.newaxis] *\
                         dm_wood_split[:,:,'fst_production-t_share',:,:]
    dm_wood_demand_t.add(ay_wood_split, col_label='wood-demand-per-type', dim='Variables', unit='t',dummy = True)

    # Turn wood demand in tonnes to wood demand in m3
    dm_wood_density = DM_wood_conversion['wood-density'].filter({'Categories1': ['any-other-wood', 'industrial-wood', 'sawlogs','woodfuel']})
    dm_wood_density.drop(dim='Categories1', col_label='woodfuel')
    dm_wood_use_m3 = dm_wood_demand_t.filter({'Variables': ['wood-demand-per-type']})
    ay_wood_density = dm_wood_use_m3[:, :, :, :, :] * \
                    dm_wood_density[np.newaxis, np.newaxis,:, :, :]
    dm_wood_use_m3.add(ay_wood_density, col_label='wood-use', dim='Variables', unit='m3', dummy=True)
    dm_wood_total = dm_wood_use_m3.flatten()
    dm_wood_total = dm_wood_total.filter({'Variables': ['wood-use']})
    dm_wood_total.groupby({'total-industry': '.*'}, regex=True, inplace=True, dim='Categories1')


    # Industrial by-production for fuelwood
    cdm_byproduct = DM_wood_conversion['industry-yields']#.filter({'Categories1': ['wood-fuel-byproducts']})
    cdm_byproduct = cdm_byproduct.filter({'Categories1': ['wood-fuel-byproducts']})
    ay_byproduct= dm_wood_use_m3[:, :, 'wood-use', :, :] * \
                    cdm_byproduct[np.newaxis, np.newaxis, np.newaxis:, :]
    dm_wood_use_m3.add(ay_byproduct, col_label='woodfuel-byproducts', dim='Variables', unit='m3', dummy=True)
    dm_supply_byproduct = dm_wood_use_m3.filter({'Variables': ['woodfuel-byproducts']}).flatten()
    dm_supply_byproduct.groupby({'total-industry': '.*'}, regex=True, inplace=True, dim='Categories1')
    dm_supply_byproduct.flatten()

    DM_industry = {
        'woodfuel-supply': dm_supply_byproduct,
        'wood-demand': dm_wood_use_m3,
        'total-demand': dm_wood_total
    }

    return DM_industry

def fuelwood_demand_m3 (dm_fuelwood_demand, DM_wood_conversion):

    #################################################################################################################
    # Provides the Fuel wood demand in m3
    # 1. Wood product per wood type (coniferous and non-coniferous) in GWh
    # 2. Express the wood demand in m3 given the hard/soft wood energy density (GWh/m3)
    #################################################################################################################

    # Wood demand from industry [t]
    dm_fuelwood_demand = dm_fuelwood_demand

    # Split between coniferous and non-coniferous wood
    dm_wood_split = DM_wood_conversion['coniferous-share'].filter_w_regex({'Variables': 'fst_production-t_share'})
    dm_wood_split = dm_wood_split.filter({'Categories1': ['woodfuel']})
    # Split between coniferous and non-coniferous wood
    ay_wood_split = dm_fuelwood_demand[:, :, 'pow_energy-supply' , :, np.newaxis] *\
                         dm_wood_split[:,:,'fst_production-t_share',:,:]
    dm_wood_split.add(ay_wood_split, col_label='wood-demand-per-type', dim='Variables', unit='TWh',dummy = True)

    # Wood fuel to wood [TWh to tonnes]
    # look for factors TWh to tonnes
    dm_wood_yields = DM_wood_conversion['energy-density'].filter(({'Categories1': ['coniferous','non-coniferous']}))
    dm_fuelwood_t = dm_wood_split.filter(({'Variables': ['wood-demand-per-type']}))
    ay_fuelwood_t = dm_fuelwood_t[:, :, 'wood-demand-per-type', :, :] *\
                    1000000/dm_wood_yields[np.newaxis,np.newaxis, :,:]
    dm_fuelwood_t.add(ay_fuelwood_t, col_label='wood-demand-per-species', dim='Variables', unit='m3', dummy=True)

    ###################################
    # Checks
    ###################################
    #dm_fuelwood_t.flatten().datamatrix_plot({'Country': ['Switzerland'], 'Variables': ['wood-demand-per-type']},
    #                                        stacked=True)
    #dm_fuelwood_t.flatten().datamatrix_plot({'Country': ['Switzerland'], 'Variables': ['wood-demand-per-species']},
    #                                        stacked=True)

    dm_fuelwood_m3 = dm_fuelwood_t.filter(({'Variables': ['wood-demand-per-species']}))
    dm_woodfuel_demand = dm_fuelwood_m3.flatten()
    dm_woodfuel_demand.groupby({'total-demand': '.*'}, regex=True, inplace=True, dim='Categories1')
    dm_woodfuel_demand.rename_col(['wood-demand-per-species'], ['fuelwood'], dim='Variables')


    DM_power = {
        'total-demand': dm_woodfuel_demand,
        'tpe': dm_fuelwood_m3
    }

    return DM_power

def wood_supply (dm_forest_area, DM_forestry, DM_ots_fts):

    #################################################################################################################
    # Provides the Fuel wood supply in m3
    # 1. Share of forest exploited (% of ha)
    # 2. Wood produced given the harvest rate
    #################################################################################################################

    # Exploited forest[ha] = Forest are [ha] * Share of exploited forest [%]

    dm_exploited = DM_forestry['fxa']['forest-exploited-share']
    dm_exploited.append(dm_forest_area, dim='Variables')
    dm_exploited.operation('productive-share', '*', 'total-forest-area',
                      out_col='exploited-forest', unit='ha')
    dm_exploited.operation('unproductive-share', '*', 'total-forest-area',
                           out_col='unexploited-forest', unit='ha')

    dm_forest_land = dm_exploited.filter(({'Variables': ['exploited-forest','unexploited-forest']}))

    # Harvested wood [m3] = Exploited forest [ha] * Harvest rate [m3/ha]
    dm_harvested_wood = DM_ots_fts['harvest-rate']
    dm_productive_area = dm_forest_land.filter(({'Variables':['exploited-forest']}))
    ay_harvest = dm_productive_area[:, :, 'exploited-forest' , np.newaxis] *\
                         dm_harvested_wood[:,:,'harvest-rate',:]
    dm_harvested_wood.add(ay_harvest, col_label='wood-harvested', dim='Variables', unit='m3',dummy = True)

    # Exogeneous wood supply & append the matrix for overall supply

    dm_wood_supply = DM_forestry['fxa']['wood-waste-energy']

    # Append

    dm_wood_supply.deepen()
    dm_wood_supply.rename_col(['fst'], ['wood-supply'], dim='Variables')
    dm_harvested_wood = dm_harvested_wood.filter(({'Variables':['wood-harvested']}))
    dm_harvested_wood.rename_col(['wood-harvested'],['wood-supply'],dim='Variables')


    DM_supply = {
        'wood-supply-per type': dm_wood_supply,
        'forest-supply': dm_forest_land,
        'harvested': dm_harvested_wood
    }

    return DM_supply

def forestry_to_tpe(DM_supply, DM_power, DM_industry, DM_ots_fts):
    #############################################################
    # DM to plot
    #############################################################

    # Wood demand per use
    tpe_wood_demand = DM_industry['wood-demand'].filter(({'Variables': ['wood-use']})).flatten().flatten()
    tpe_supply_species = tpe_wood_demand.copy()
    tpe_supply_species.groupby({'wood-supply_coniferous': '.*_coniferous',' wood-supply_non-coniferous': '.*_non-coniferous'}, regex=True, inplace=True, dim='Variables')
    tpe_wood_demand.groupby({'any-other-wood': '.*any-other-wood.*', 'industrial-wood': '.*industrial-wood.*', 'sawlogs': '.*sawlogs.*'},
                                 dim='Variables', regex=True, inplace=True)

    tpe_energy_wood_demand = DM_power['total-demand'].filter(({'Variables': ['fuelwood']})).flatten()
    tpe_wood_demand.append(tpe_energy_wood_demand, dim='Variables')
    tpe_wood_demand.rename_col(
        ['fuelwood_total-demand'],
        ['woodfuel'], dim='Variables')

    #tpe_wood_demand.datamatrix_plot({'Country': ['Switzerland']}, stacked=True)

    # Wood demand per use per species
    tpe_industry_wood_species = DM_industry['wood-demand'].filter(({'Variables': ['wood-use']})).flatten()
    tpe_energy_wood_species = DM_power['tpe'].flatten().flatten()

    #tpe_industry_wood_species.datamatrix_plot({'Country': ['Switzerland']}, stacked=True)

    # Harvested wood
    dm_harvested_wood = DM_supply['harvested'].filter(({'Categories1':['coniferous','non-coniferous']}))
    dm_harvested_wood.rename_col(
        ['wood-supply'],
        ['harvested-wood'],
        dim='Variables')
    #dm_harvested_wood.datamatrix_plot({'Country': ['Switzerland']}, stacked=True)

    # Exploited forest area
    dm_forest_area = DM_supply['forest-supply']
    #dm_forest_area.datamatrix_plot({'Country': ['Switzerland']}, stacked=True)

    # Wood supply per source
    tpe_wood_supply = DM_industry['woodfuel-supply'].flatten()
    dm_waste_supply = DM_supply['wood-supply-per type'].flatten()

    dm_harvest_supply = dm_harvested_wood.copy()
    dm_harvest_supply.groupby({'harvested-wood': '.*'}, regex=True, inplace=True, dim='Categories1')
    dm_harvest_supply = dm_harvest_supply.flatten()
    tpe_wood_supply.append(dm_harvest_supply, dim='Variables')
    tpe_wood_supply.append(dm_waste_supply, dim='Variables')
    tpe_wood_supply.rename_col(
        ['woodfuel-byproducts_total-industry','harvested-wood_harvested-wood'],
        ['wood-supply_industrial-byproducts','wood-supply_harvested-wood'],
        dim='Variables')

    # Harvest rate
    tpe_harvest_rate = DM_ots_fts['harvest-rate'].filter(({'Variables': ['harvest-rate']}))
    #tpe_harvest_rate.flatten().datamatrix_plot({'Country': ['Switzerland']}, stacked=True)
    #tpe_harvest_rate = tpe_harvest_rate.filter(({'Categories1': [ 'coniferous', 'non-coniferous']}))

    #############################################################
    # DM to df for tpe
    #############################################################
    dm_tpe = dm_harvested_wood.flattest()
    dm_tpe.append(dm_forest_area.flattest(), dim='Variables')
    dm_tpe.append(tpe_wood_demand.flattest(), dim='Variables')
    dm_tpe.append(tpe_industry_wood_species.flattest(), dim='Variables')
    dm_tpe.append(tpe_energy_wood_species.flattest(), dim='Variables')
    dm_tpe.append(tpe_wood_supply.flattest(), dim='Variables')
    dm_tpe.append(tpe_supply_species.flattest(), dim='Variables')
    dm_tpe.append(tpe_harvest_rate.flattest(), dim='Variables')

    return dm_tpe


def forestry(lever_setting, years_setting, interface=Interface()):

    ##############################################################
    # Load the datamatrix of Forestry: the following is computed in the pre-processing, and includes ots, fts, fxa, cp
    ##############################################################

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    forestry_data_file = os.path.join(current_file_directory, '../_database/data/datamatrix/geoscale/forestry.pickle')
    # Read forestry pickle data based on lever setting and return data in a structured DM
    DM_forestry, DM_wood_conversion, DM_ots_fts = read_data(forestry_data_file, lever_setting)
    cntr_list = DM_ots_fts['harvest-rate'].col_labels['Country']
    #cntr_list = ['Switzerland']

    ###################################################################
    # Interface - Energy to Forestry: If the input from Power/Energy are available in the interface, read them, else read from pickle
    ###################################################################

    if interface.has_link(from_sector='energy', to_sector='forestry'):
        dm_fuelwood_demand = interface.get_link(from_sector='energy', to_sector='forestry')
    else:
        if len(interface.list_link()) != 0:
            print("You are missing " + 'energy' + " to " + 'forestry' + " interface")
        lfs_interface_data_file = os.path.join(current_file_directory,
                                               '../_database/data/interface/energy_to_forestry.pickle')
        with open(lfs_interface_data_file, 'rb') as handle:
            dm_fuelwood_demand = pickle.load(handle)
        dm_fuelwood_demand.filter({'Country': cntr_list}, inplace=True)
    ####################################################################################################################
    # Interface - Industry to Forestry:
    ####################################################################################################################

    if interface.has_link(from_sector='industry', to_sector='forestry'):
        dm_wood_demand = interface.get_link(from_sector='industry', to_sector='forestry')
    else:
        if len(interface.list_link()) != 0:
            print("You are missing " + 'industry' + " to " + 'forestry' + " interface")
        lfs_interface_data_file = os.path.join(current_file_directory,
                                               '../_database/data/interface/industry_to_forestry.pickle')
        with open(lfs_interface_data_file, 'rb') as handle:
            dm_wood_demand = pickle.load(handle)
        dm_wood_demand.filter({'Country': cntr_list}, inplace=True)

    ####################################################################################################################
    # Interface - Land to Forestry:
    ####################################################################################################################

    if interface.has_link(from_sector='land', to_sector='forestry'):
        dm_forest_area = interface.get_link(from_sector='land', to_sector='forestry')
    else:
        if len(interface.list_link()) != 0:
            print("You are missing " + 'land' + " to " + 'forestry' + " interface")
        lfs_interface_data_file = os.path.join(current_file_directory,
                                               '../_database/data/interface/land_to_forestry.pickle')
        with open(lfs_interface_data_file, 'rb') as handle:
            dm_forest_area = pickle.load(handle)
        dm_forest_area.filter({'Country': cntr_list}, inplace=True)

    ####################################################################################################################
    # Calculation tree - Wood demand in m3 (Industry & Power)
    ####################################################################################################################
    # Wood demand in Industry
    DM_industry = wood_demand_m3(dm_wood_demand,DM_wood_conversion)
    ind_wood_demand = DM_industry['total-demand']

    # Wood demand in Power
    DM_power = fuelwood_demand_m3(dm_fuelwood_demand,DM_wood_conversion)
    dm_rwe_demand = DM_power['total-demand']

    ####################################################################################################################
    # Calculation tree - Wood supply in m3 (Industry & Power)
    ####################################################################################################################

    # Wood supply in Industry
    ind_byproduct_fuelwood = DM_industry['woodfuel-supply']

    # Wood supply from wastes (exogenous)
    fxa_waste_wood = DM_forestry['fxa']['wood-waste-energy']

    # Harvested wood
    DM_supply = wood_supply(dm_forest_area, DM_forestry, DM_ots_fts)
    fst_harvested_wood = DM_supply['harvested'].filter(({'Categories1': ['total']}))

    ####################################################################################################################
    # Calculation tree - Net demand of wood supply in m3
    ####################################################################################################################

    # Power demand given industrial wood-byproducts
    ay_net_balance = fst_harvested_wood[:, :, :, :] +\
                    ind_byproduct_fuelwood[:, :, :, :] +\
                    fxa_waste_wood[:, :, :, :]-\
                    ind_wood_demand[:, :, :, :]-\
                     dm_rwe_demand[:, :, :, :]

    dm_rwe_demand.add(ay_net_balance, col_label='net-balance', dim='Variables', unit='m3', dummy=True)
    dm_wood_balance = dm_rwe_demand.filter(({'Variables': ['net-balance']}))


    ####################################################################################################################
    # Interface - Forestry to TPE :
    ####################################################################################################################


    results_run = forestry_to_tpe(DM_supply, DM_power, DM_industry, DM_ots_fts)


    return results_run


def local_forestry_run():
    # Function to run only transport module without converter and tpe
    years_setting = [1990, 2023, 2025, 2050, 5]
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(current_file_directory, '../config/lever_position.json'))
    lever_setting = json.load(f)[0]

    global_vars = {'geoscale': 'Vaud'}
    filter_geoscale(global_vars['geoscale'])

    results_run = forestry(lever_setting, years_setting)

    return results_run


local_forestry_run()