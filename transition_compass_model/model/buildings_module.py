import pandas as pd
from model.common.data_matrix_class import DataMatrix
from model.common.interface_class import Interface

from model.common.auxiliary_functions import read_level_data, \
  filter_country_and_load_data_from_pickles, create_years_list, \
  dm_add_missing_variables
import pickle
import json
import os
import warnings
import model.buildings.workflows as wkf
import model.buildings.interfaces as inter

warnings.simplefilter("ignore")


def init_years_lever():
    # function that can be used when running the module as standalone to initialise years and levers
    years_setting = [1990, 2023, 2025, 2050, 5]
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(current_file_directory, '../config/lever_position.json'))
    lever_setting = json.load(f)[0]
    return years_setting, lever_setting


def read_data(DM_buildings, lever_setting):
    # Read fts based on lever_setting
    DM_ots_fts = read_level_data(DM_buildings, lever_setting)

    DM_floor_area = DM_ots_fts['building-renovation-rate']

    DM_floor_area['bld_type'] = DM_buildings['fxa']['bld_type']
    DM_floor_area['bld_age'] = DM_buildings['fxa']['bld_age']
    DM_floor_area['floor-intensity'] = DM_ots_fts['floor-intensity']

    DM_appliances = {'demand': DM_buildings['fxa']['appliances'],
                     'household-size': DM_ots_fts['floor-intensity'].filter({'Variables':['lfs_household-size']}),
                     'other-electricity-demand': DM_buildings['fxa']['other-electricity-demand']}

    DM_hotwater = {'demand': DM_buildings['fxa']['hot-water']['hw-energy-demand'],
                   'efficiency': DM_buildings['fxa']['hot-water']['hw-efficiency'],
                   'tech-mix': DM_buildings['fxa']['hot-water']['hw-tech-mix']}

    DM_services = DM_buildings['fxa']['services']

    dm_light = DM_buildings['fxa']['lighting']

    DM_energy = {'heating-efficiency': DM_ots_fts['heating-efficiency'],
                 'heating-technology': DM_ots_fts['heating-technology-fuel']['bld_heating-technology'],
                 'heatcool-behaviour': DM_ots_fts['heatcool-behaviour'],
                 'heating-calibration': DM_buildings['fxa']['heating-energy-calibration'],
                 'electricity-emission': DM_buildings['fxa']['emission-factor-electricity'],
                 "u-value" :  DM_buildings['fxa']["u-value"],
                 "surface-to-floorarea" : DM_buildings['fxa']["surface-to-floorarea"]}


    cdm_const = DM_buildings['constant']

    return DM_floor_area, DM_appliances, DM_energy, DM_hotwater, DM_services, dm_light, cdm_const



def buildings(lever_setting, years_setting, DM_input, interface=Interface()):
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    # Read data into workflow datamatrix dictionaries
    DM_floor_area, DM_appliances, DM_energy, DM_hotwater, DM_services, dm_light, cdm_const = read_data(DM_input, lever_setting)
    years_ots = create_years_list(years_setting[0], years_setting[1], 1)
    years_fts = create_years_list(years_setting[2], years_setting[3], 5)

    # Simulate lifestyle input
    if interface.has_link(from_sector='lifestyles', to_sector='buildings'):
        DM_lfs = interface.get_link(from_sector='lifestyles', to_sector='buildings')
        dm_lfs = DM_lfs['pop']
    else:
        if len(interface.list_link()) != 0:
            print('You are missing lifestyles to buildings interface')
        data_file = os.path.join(current_file_directory, '../_database/data/interface/lifestyles_to_buildings.pickle')
        with open(data_file, 'rb') as handle:
            DM_lfs = pickle.load(handle)
        dm_lfs = DM_lfs['pop']
        cntr_list = DM_floor_area['floor-intensity'].col_labels['Country']
        dm_lfs.filter({'Country': cntr_list}, inplace=True)

    if interface.has_link(from_sector='climate', to_sector='buildings'):
        dm_clm = interface.get_link(from_sector='climate', to_sector='buildings')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing lifestyles to buildings interface')
        data_file = os.path.join(current_file_directory, '../_database/data/interface/climate_to_buildings.pickle')
        with open(data_file, 'rb') as handle:
            DM_clm = pickle.load(handle)
        dm_clm = DM_clm["cdd-hdd"]
        cntr_list = DM_floor_area['floor-intensity'].col_labels['Country']
        dm_clm.filter({'Country': cntr_list}, inplace=True)

    # Floor Area, Comulated floor area, Construction material
    DM_floor_out = wkf.bld_floor_area_workflow(DM_floor_area, dm_lfs, cdm_const, years_ots, years_fts)

    DM_appliances_out = wkf.bld_appliances_workflow(DM_appliances, dm_lfs)

    #print('You are missing appliances (that should run before energy, so that you have the missing term of the equation')
    #print('You are missing the calibration of the energy results.')
    #print('The heating-mix between 2000 and 2018 or so is bad, maybe it should be improved before calibrating '
    #      'or you calibrate only on missing years')
    #print('You are missing the costs')
    # Total Energy demand, Renovation and Construction per depth, GHG emissions (for Space Heating)
    DM_energy_out = wkf.bld_energy_workflow(DM_energy, dm_clm, DM_floor_out['wf-energy'], cdm_const)

    DM_hotwater_out = wkf.bld_hotwater_workflow(DM_hotwater, DM_energy_out['TPE']['energy-demand-heating'].copy(), dm_lfs, years_ots, years_fts)

    DM_services_out = wkf.bld_services_workflow(DM_services, DM_energy_out['TPE']['energy-demand-heating'].copy(), years_ots, years_fts)

    DM_light_out = {'TPE': dm_light.copy(), 'energy': dm_light.copy()}

    # Group household (residential) energy demand
    # Electricity is off (check heat-pump as well and renewable)
    check_residential_energy_demand = False
    if check_residential_energy_demand:
      dm_heat_cool = DM_energy_out['power'].filter({'Variables': ['bld_energy-demand_heating', 'bld_energy-demand_cooling']})
      dm_hot_water = DM_hotwater_out['power'].filter({'Variables': ['bld_hot-water_energy-demand']})
      dm_hot_water.rename_col('bld_hot-water_energy-demand', 'bld_energy-demand_hot-water', 'Variables')
      dm_add_missing_variables(dm_hot_water,{'Categories1': dm_heat_cool.col_labels['Categories1']} )
      dm_appliances = DM_appliances_out['power'].copy()
      dm_appliances.rename_col('bld_appliances_tot-elec-demand', 'bld_energy-demand_appliances_electricity', 'Variables')
      dm_appliances.deepen()
      dm_add_missing_variables(dm_appliances, {'Categories1': dm_heat_cool.col_labels['Categories1']})
      dm_light = DM_light_out['TPE'].copy()
      dm_light.rename_col('bld_residential-lighting', 'bld_energy-demand_light_electricity', 'Variables')
      dm_light.deepen()
      dm_add_missing_variables(dm_light, {'Categories1': dm_heat_cool.col_labels['Categories1']})
      dm_residential_energy = dm_heat_cool
      dm_residential_energy.append(dm_hot_water, dim='Variables')
      dm_residential_energy.append(dm_appliances, dim='Variables')
      dm_residential_energy.append(dm_light, dim='Variables')
      dm_residential_energy.deepen(based_on ='Variables')
      dm_residential_energy.change_unit('bld_energy-demand', old_unit='TWh', new_unit='TJ', factor=3600)

    # Group services energy demand
    # TPE
    results_run, KPI = inter.bld_TPE_interface(DM_energy_out['TPE'], DM_floor_out['TPE'], DM_services_out['TPE'], DM_appliances_out['power'], DM_light_out['TPE'], DM_hotwater_out['power'])

    # 'District-heating' module interface
    interface.add_link(from_sector='buildings', to_sector='district-heating', dm=DM_energy_out['district-heating'])

    DM_inter_energy = {'households_heating': DM_energy_out['power'],
                       'households_hot-water': DM_hotwater_out['power'],
                       'households_lighting': DM_light_out['energy'],
                       'households_electricity': DM_appliances_out['power'],
                       'services_all': DM_services_out['energy']}
    interface.add_link(from_sector='buildings', to_sector='energy', dm=DM_inter_energy)
    
    # this_dir = os.path.dirname(os.path.abspath(__file__))
    # file = os.path.join(this_dir, '../_database/data/interface/buildings_to_lca.pickle')
    # with open(file, "wb") as handle:
    #     pickle.dump(DM_floor_out['industry'], handle, protocol=pickle.HIGHEST_PROTOCOL )

    interface.add_link(from_sector='buildings', to_sector='emissions', dm=DM_energy_out['TPE']['emissions'])

    interface.add_link(from_sector='buildings', to_sector='industry', dm=DM_floor_out['industry'])
    # TODO: probably add non residential (offices) to interface to industry

    # interface.add_link(from_sector='buildings', to_sector='minerals', dm=DM_floor_out['industry'])
    
    interface.add_link(from_sector='buildings', to_sector='lca', dm=DM_floor_out['industry'])

    #interface.add_link(from_sector='buildings', to_sector='agriculture', dm=DM_energy_out['agriculture'])

    interface.add_link(from_sector='buildings', to_sector='oil-refinery', dm=DM_energy_out['refinery'])

    return results_run, KPI


def buildings_local_run():
    # Function to run module as stand alone without other modules/converter or TPE
    years_setting, lever_setting = init_years_lever()
    # Function to run only transport module without converter and tpe

    # get geoscale
    country_list = ['EU27','Switzerland','Vaud']
    DM_input = filter_country_and_load_data_from_pickles(country_list= country_list, modules_list = 'buildings')

    buildings(lever_setting, years_setting, DM_input['buildings'])
    return

if __name__ == "__main__":
  buildings_local_run()
