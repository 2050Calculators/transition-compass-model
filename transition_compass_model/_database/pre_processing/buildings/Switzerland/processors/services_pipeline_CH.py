# Non Residential aka Services
import pandas as pd

import _database.pre_processing.buildings.Switzerland.get_data_functions.services_CH as fser
from model.common.auxiliary_functions import linear_fitting, create_years_list, \
  load_pop, dm_add_missing_variables, save_url_to_file
from model.common.data_matrix_class import DataMatrix

from _database.pre_processing.buildings.Switzerland.processors.hot_water_pipeline_CH import run as hotwater_run
from _database.pre_processing.buildings.Switzerland.processors.heating_technology_pipeline_CH import run as heating_tech_run

import numpy as np
import re
import os

def rename_cantons(dm):
  dm.sort('Country')
  dm.rename_col_regex(" /.*", "", dim='Country')
  dm.rename_col_regex("-", " ", dim='Country')
  cantons_en = ['Aargau', 'Appenzell Ausserrhoden', 'Appenzell Innerrhoden', 'Basel Landschaft', 'Basel Stadt', 'Bern', 'Fribourg', 'Geneva', 'Glarus', 'Graubünden', 'Jura', 'Lucerne', 'Neuchâtel', 'Nidwalden', 'Obwalden', 'Schaffhausen', 'Schwyz', 'Solothurn', 'St. Gallen', 'Thurgau', 'Ticino', 'Uri', 'Valais', 'Vaud', 'Zug', 'Zurich']
  cantons_fr = ['Aargau', 'Appenzell Ausserrhoden', 'Appenzell Innerrhoden', 'Basel Landschaft', 'Basel Stadt', 'Bern', 'Fribourg', 'Genève', 'Glarus', 'Graubünden', 'Jura', 'Luzern', 'Neuchâtel', 'Nidwalden', 'Obwalden', 'Schaffhausen', 'Schwyz', 'Solothurn', 'St. Gallen', 'Thurgau', 'Ticino', 'Uri', 'Valais', 'Vaud', 'Zug', 'Zürich']
  dm.rename_col(cantons_fr, cantons_en, dim='Country')

  return


def has_numbers(inputString):
  return any(char.isdigit() for char in inputString)

def determine_mapping_dict(cat_list_ref, cat_list_match):
  mapping_dict = {}
  mapping_sectors = {'agriculture': [], 'industry-w-process-heat': [], 'services': [], 'industry-wo-process-heat': []}
  for cat in cat_list_ref:
      if has_numbers(cat):
          cat_num = re.findall(r'\d+', cat)[0]
          if len(cat_num) == 2:
              matching_cat = [c for c in cat_list_match if cat_num in c]
              mapping_dict[cat] = matching_cat
              if int(cat_num) <= 3:
                  mapping_sectors['agriculture'].append(cat)
              elif 3 < int(cat_num) < 41:
                  mapping_sectors['industry-w-process-heat'].append(cat)
              elif 41 <= int(cat_num) < 45:
                  mapping_sectors['industry-wo-process-heat'].append(cat)
              elif int(cat_num) >= 45:
                  mapping_sectors['services'].append(cat)
          elif len(cat_num) == 4:
              first_num = cat_num[0:2]
              second_num = cat_num[2:4]
              matching_cat = []
              for i in range(int(first_num), int(second_num) + 1):
                  str_i = f"{i:02}"  # pad with zeros
                  matching_cat_i = [c for c in cat_list_match if str_i in c]
                  matching_cat.append(matching_cat_i[0])
              if int(first_num) <= 3:
                  mapping_sectors['agriculture'].append(cat)
              elif 3 < int(first_num) < 41:
                  mapping_sectors['industry-w-process-heat'].append(cat)
              elif 41 <= int(first_num) < 45:
                  mapping_sectors['industry-wo-process-heat'].append(cat)
              elif int(first_num) >= 45:
                  mapping_sectors['services'].append(cat)
              mapping_dict[cat] = matching_cat
  return mapping_dict, mapping_sectors


def map_national_energy_demand_by_sector_to_cantons(dm_energy, dm_employees, mapping_dict, mapping_sectors):


  dm_employees_mapped = dm_employees.groupby(mapping_dict, dim='Categories1', inplace=False)
  dm_employees_mapped.drop('Country', 'Suisse')
  dm_employees_mapped.sort('Categories1')

  dm_energy_mapped = dm_energy.filter({'Categories1': dm_employees_mapped.col_labels['Categories1']}, inplace=False)
  dm_energy_mapped.sort('Categories1')
  dm_employees_mapped.normalise(dim='Country')
  linear_fitting(dm_employees_mapped, years_ots=dm_energy.col_labels['Years'], min_t0=0)
  dm_employees_mapped.normalise(dim='Country')

  new_arr = dm_employees_mapped.array[:, :, :, :, np.newaxis] * dm_energy_mapped.array[:, :, :, :, :]
  dm_energy_mapped.add(new_arr, dim='Country', col_label=dm_employees_mapped.col_labels['Country'])

  dm_energy_mapped.groupby(mapping_sectors, dim='Categories1', inplace=True)
  dm_employees_mapped.groupby(mapping_sectors, dim='Categories1', inplace=True)
  dm_employees_mapped.add(np.nan, dummy=True, dim='Years',
                          col_label=list(set(years_ots) - set(dm_employees_mapped.col_labels['Years'])))
  dm_employees_mapped.sort('Years')
  dm_employees_mapped.fill_nans('Years')

  return dm_energy_mapped, dm_employees_mapped


def map_services_eud_by_canton(dm_employees_mapped, dm_services_end_use):
    # First map all end-use except process heat
    dm_employees = dm_employees_mapped.filter({'Categories1': ['services']}, inplace=False)
    dm_employees.normalise('Country')
    arr = (dm_services_end_use['Switzerland', np.newaxis, :, 'enr_services-energy-eud', :]
           * dm_employees[:, :, 'ind_employees', 'services', np.newaxis])
    dm = DataMatrix.based_on(arr[:, :, np.newaxis, ...], format=dm_services_end_use,
                                                 change={'Country': dm_employees.col_labels['Country']},
                                                 units={'srv_energy-end-use': 'TWh'})

    return dm


def split_fuel_demand_by_eud(dm_water, dm_space_heat, dm_industry_eud_canton, cantonal = True):

    dm_fuel_split = dm_water.copy()
    dm_fuel_split.append(dm_space_heat, dim='Categories1')

    # Electricity and Lighting
    dm_elec = dm_industry_eud_canton.filter({'Categories1': ['elec', 'lighting']})
    var = dm_elec.col_labels['Variables'][0]
    dm_elec.rename_col(var, var + '_electricity', dim='Variables')
    dm_elec.deepen(based_on='Variables')

    # Space-heat and Hot water
    if cantonal:
        dm_fuel_split.drop('Country', 'Switzerland')
    dm_hw_sp = dm_industry_eud_canton.filter({'Categories1': ['hot-water', 'space-heating']})
    var = dm_hw_sp.col_labels['Variables'][0]
    assert dm_fuel_split.col_labels['Country'] == dm_hw_sp.col_labels['Country']
    arr = dm_hw_sp[:, :, var, :, np.newaxis] \
          * dm_fuel_split[:, :, 'bld_households', :, :]
    dm_fuel_split.add(arr, dim='Variables', col_label=var, unit='TWh')

    dm_fuels_eud_cantons = dm_elec.copy()
    dummy_cat = list(set(dm_fuel_split.col_labels['Categories2']) - set(dm_fuels_eud_cantons.col_labels['Categories2']))
    dm_fuels_eud_cantons.add(0, dim='Categories2', col_label=dummy_cat, dummy=True)
    dm_fuels_eud_cantons.append(dm_fuel_split.filter({'Variables': [var]}), dim='Categories1')

    return dm_fuels_eud_cantons


def adjust_based_on_FSO_energy_consumption(dm_fuels_cantons, dm_services_fuels_eud_cantons, years_ots):
    # Each fuel is split by its use in hot-water or space-heating, at canton level.
    dm_eud_shares = dm_services_fuels_eud_cantons.normalise('Categories1', inplace=False)
    dm_OFS_fuels = dm_fuels_cantons.groupby({'total': 'services'}, regex=True, inplace=False, dim='Categories1')
    dm_OFS_fuels.rename_col('bld_energy-by-sector', 'srv_energy-end-use', dim='Variables')
    missing_fuels = list(set(dm_OFS_fuels.col_labels['Categories2']) - set(dm_eud_shares.col_labels['Categories2']))
    dm_eud_shares.add(0, dim='Categories2', dummy=True, col_label=missing_fuels)
    # Attribute the missing fuels to space-heat
    for fuel in missing_fuels:
        dm_eud_shares[:, :, :, 'space-heating', fuel] = 1
    dm_eud_shares.sort('Categories2')
    missing_fuels_2 = list(set(dm_eud_shares.col_labels['Categories2']) - set(dm_OFS_fuels.col_labels['Categories2']))
    dm_OFS_fuels.add(np.nan, dim='Categories2', dummy=True, col_label=missing_fuels_2)
    dm_OFS_fuels.sort('Categories2')

    dm_OFS_fuels.filter({'Country': dm_eud_shares.col_labels['Country']}, inplace=True)

    # Add missing years
    dm_add_missing_variables(dm_OFS_fuels, {'Years': years_ots}, fill_nans=True)

    arr = (dm_OFS_fuels[:, :, 'srv_energy-end-use', 'total', np.newaxis, :]
           * dm_eud_shares[:, :, 'enr_services-energy-eud_share', :, :])

    dm_eud_shares.add(arr, dim='Variables', col_label='enr_services-energy-eud', unit='TWh')
    dm_eud_shares.filter({'Variables': ['enr_services-energy-eud']}, inplace=True)

    for fuel in missing_fuels_2:
        dm_eud_shares[:, :, :, :, fuel] = dm_services_fuels_eud_cantons[:, :, :, :, fuel]
    return dm_eud_shares


def energy_split_from_fuel_to_tech(dm_energy, dm_water_CH, dm_space_heat_CH, dm_efficiency, dm_services_eud):

  dm_solar = split_fuel_demand_by_eud(dm_water_CH, dm_space_heat_CH, dm_services_eud.copy(), cantonal = False)
  dm_solar.group_all('Categories1', inplace=True)
  dm_eff = dm_efficiency.filter({'Years': dm_solar.col_labels['Years'], 'Country': ['Switzerland']})
  arr_renew = dm_solar[:, :, :, 'heat-pump']*(dm_eff[:, :, :, 'heat-pump'] - 1 )
  dm_solar.add(arr_renew, dim='Categories1', col_label='renewable')
  dm_solar.filter({'Categories1': ['solar', 'renewable']}, inplace=True)
  dm_solar.normalise('Categories1')
  dm_solar.fill_nans('Years')

  dm_solar.filter({'Years': dm_energy.col_labels['Years']}, inplace=True)
  arr_solar = dm_energy[:, :, :, :, 'renewables'] * dm_solar[:, :, :, np.newaxis, 'solar']
  dm_energy.add(arr_solar, dim='Categories2', col_label='solar')
  dm_energy.operation('renewables', '-', 'solar', out_col='renewable', dim='Categories2')
  dm_energy.drop('Categories2', 'renewables')
  dm_eff.filter({'Years': dm_energy.col_labels['Years']}, inplace=True)
  arr_HP = dm_energy[:, :, :, :, 'renewable'] /(dm_eff[:, :, :, np.newaxis,'heat-pump'] - 1 )
  dm_energy.add(arr_HP, dim='Categories2', col_label = 'heat-pump')
  dm_energy[:, :, :, :, 'electricity']  = dm_energy[:, :, :, :, 'electricity']  - dm_energy[:, :, :, :, 'heat-pump']
  dm_energy.drop('Categories2', 'renewable')

  return dm_energy

#####################################
##   SERVICES TO ENERGY INTERFACE  ##
#####################################
def run(years_ots):

  this_dir = os.path.dirname(os.path.abspath(__file__))

  # Get Water data for all cantons
  cantons_en = ['Aargau', 'Appenzell Ausserrhoden', 'Appenzell Innerrhoden',
                'Basel Landschaft', 'Basel Stadt', 'Bern', 'Fribourg', 'Geneva',
                'Glarus', 'Graubünden', 'Jura', 'Lucerne', 'Neuchâtel',
                'Nidwalden', 'Obwalden', 'Schaffhausen', 'Schwyz', 'Solothurn',
                'St. Gallen', 'Thurgau', 'Ticino', 'Uri', 'Valais', 'Vaud',
                'Zug', 'Zurich'] + ['Switzerland']
  DM_water = hotwater_run(country_list=cantons_en, years_ots=years_ots)

  dm_water = DM_water['hw-tech-mix']  # hot water mix for useful energy
  dm_efficiency = DM_water['hw-efficiency'].copy()
  dm_water.append(dm_efficiency, dim='Variables')  # hot water tech efficiency
  # Tech mix for final energy consumption
  dm_water.operation('bld_hw_tech-mix', '/', 'bld_hot-water_efficiency', out_col='bld_households_hot-water', unit='%')
  dm_water.filter({'Variables': ['bld_households_hot-water']}, inplace=True)
  dm_water.normalise('Categories1')
  dm_water.deepen(based_on='Variables')
  dm_water.switch_categories_order()
  # !FIXME: DUMMY dm_space_heat - replace!

  dm_space_heat = dm_water.copy()
  dm_space_heat.rename_col('hot-water','space-heating', 'Categories1')


  # Infras, TEP, Prognos, 2021. Analyse des schweizerischen Energieverbrauchs 2000–2020 - Auswertung nach Verwendungszwecken.
  # Table 26 - Endenergieverbrauch im Dienstleistungssektor nach Verwendungszwecken Entwicklung von 2000 bis 2020, in PJ, inkl. Landwirtschaft
  # Final energy consumption in the service sector by purpose Development from 2000 to 2020, in PJ, incl. agriculture
  # https://www.bfe.admin.ch/bfe/de/home/versorgung/statistik-und-geodaten/energiestatistiken/energieverbrauch-nach-verwendungszweck.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvMTA2OTM%3D.html&ved=2ahUKEwiC4OjJvpGOAxWexgIHHdyFGVMQFnoECB0QAQ&usg=AOvVaw1a9deGMbwSdNvV0aVLEBPj
  services_agri_split = {
      "space-heating": {2000: 82.1, 2014: 67.1, 2015: 73.2, 2016: 77.3, 2017: 75.0, 2018: 67.0, 2019: 69.7, 2020: 65.9},
      "hot-water": {2000: 12.7, 2014: 12.1, 2015: 12.1, 2016: 12.0, 2017: 12.0, 2018: 12.0, 2019: 11.9, 2020: 11.9},
      "process-heat": {2000: 2.3, 2014: 2.5, 2015: 2.5, 2016: 2.5, 2017: 2.5, 2018: 2.6, 2019: 2.7, 2020: 2.2},
      "lighting": {2000: 16.8, 2014: 17.0, 2015: 17.0, 2016: 17.0, 2017: 17.0, 2018: 16.9, 2019: 16.8, 2020: 15.9},
      "HVAC and building tech": {2000: 11.2, 2014: 13.8, 2015: 15.0, 2016: 15.1, 2017: 15.5, 2018: 15.5, 2019: 15.8, 2020: 15.0},
      "ICT and entertainment media": {2000: 6.1, 2014: 6.9, 2015: 6.9, 2016: 6.9, 2017: 6.9, 2018: 7.0, 2019: 7.0, 2020: 6.8},
      "Drives and processes": {2000: 14.4, 2014: 16.1, 2015: 16.0, 2016: 16.0, 2017: 15.9, 2018: 16.1, 2019: 16.2, 2020: 15.7},
      "Other": {2000: 4.0, 2014: 4.3, 2015: 4.4, 2016: 4.4, 2017: 4.3, 2018: 4.4, 2019: 4.3, 2020: 4.1},
      "Total": {2000: 149.7, 2014: 139.7, 2015: 147.2, 2016: 151.2, 2017: 149.2, 2018: 141.5, 2019: 144.4, 2020: 137.5}
  }
  dm_services_eud = fser.load_services_energy_demand_eud(services_agri_split, years_ots)

  # Extract energy demand by sector at national level by fuel
  table_id = 'px-x-0204000000_106'
  local_filename = os.path.join(this_dir, '../data/energy_accounts_economy_households.pickle')
  # Industry sectors linked to energy, like energy production and waste management, have been removed or edited
  # Remove gasoline and diesel which are for transport / machinery
  dm_energy = fser.extract_national_energy_demand(table_id, local_filename)

  # Extract number of employees per industry and service sector by canton
  # This is in order to map the national energy demand to cantons
  table_id = 'px-x-0602010000_101'
  local_filename = os.path.join(this_dir, '../data/employees_per_sector_canton.pickle')
  dm_employees = fser.extract_employees_per_sector_canton(table_id, local_filename)

  cat_list_ref = dm_energy.col_labels['Categories1']
  cat_list_match = dm_employees.col_labels['Categories1']
  mapping_dict, mapping_sectors = determine_mapping_dict(cat_list_ref, cat_list_match)
  dm_energy.filter({'Categories1': list(mapping_dict.keys())}, inplace=True)

  # Adjust energy carriers categories to heating technologies
  # In particular renewables = ambient heat + geothermal + solar (energy consumption)
  # I want to split it, using water energy consumption split
  # I know heat-pump energy consumption and I know the COP so I can get the energy consumption of ambient-heat / geothermal
  # the problem is that solar is used for hot-water heating but not much for space-heating.
  # Go from fuel split to tech split Detarmine Solar share to then infer Heat-pump
  dm_water_CH = dm_water.filter({'Country': ['Switzerland']})
  dm_space_heat_CH = dm_space_heat.filter({'Country': ['Switzerland']})

  dm_energy = energy_split_from_fuel_to_tech(dm_energy, dm_water_CH, dm_space_heat_CH, dm_efficiency, dm_services_eud)

  # Actually my energy consumption of heat-pump is already the electrical share of the energy consumption.
  # energy consumption (heat-pump) = COP * useful energy

  # Group employees by sector and canton (dm_employees_mapped)
  dm_fuels_cantons, dm_employees_mapped = map_national_energy_demand_by_sector_to_cantons(dm_energy, dm_employees, mapping_dict, mapping_sectors)
  rename_cantons(dm_fuels_cantons)

  # Agiculture demand is << than services, I will not split it here. I do have agriculture data by fuel
  # !FIXME: Consider assigning Drives and processes here and in Industry to not only electricity but also diesel and gasoline.
  # Basically remove from Drives and processes the diesel and gasoline demand. and the remainder is electricity.
  # Also HVAC could be heat pumps and not electricity
  dm_services_eud_cantons = map_services_eud_by_canton(dm_employees_mapped, dm_services_eud)
  rename_cantons(dm_services_eud_cantons)

  dm_services_fuels_eud_cantons = split_fuel_demand_by_eud(dm_water, dm_space_heat, dm_services_eud_cantons)

  # I use the OFS data on fuels consumption by service and by canton to adjust the results.
  # Concretely, for each fuel, I compute the share by end-use and then I multiply by the fuel OFS consumption
  dm_services_fuels_eud_cantons_FSO = adjust_based_on_FSO_energy_consumption(dm_fuels_cantons, dm_services_fuels_eud_cantons, years_ots)

  # Add Switzerland
  dm_services_fuels_eud_cantons_CH = dm_services_fuels_eud_cantons_FSO.groupby({'Switzerland': '.*'}, dim='Country', regex=True, inplace=False)
  dm_services_fuels_eud_cantons_FSO.append(dm_services_fuels_eud_cantons_CH, dim='Country')
  dm_services_fuels_eud_cantons_FSO.sort('Country')

  # Replace 1990-2000 flat extrapolation with linear fitting
  idx = dm_services_fuels_eud_cantons_FSO.idx
  dm_services_fuels_eud_cantons_FSO.array[:, idx[1990]: idx[2000], ...] = np.nan

  dm_services_fuels_eud_cantons_FSO.drop(col_label='waste', dim='Categories2')
  dm_services_fuels_eud_cantons_FSO.drop(col_label='nuclear-fuel', dim='Categories2')

  # Compute useful energy demand from energy consumption and efficiency
  dm_eff = DM_water['hw-efficiency'].copy()
  # Add missing efficiencies
  arr_eff_gas = dm_eff[:, :, :, 'gas']
  dm_eff.add(arr_eff_gas, dim='Categories1', col_label='biogas')
  arr_eff_wood = dm_eff[:, :, :, 'wood']
  dm_eff.add(arr_eff_wood, dim='Categories1', col_label='biomass')

  dm_add_missing_variables(dm_eff, {'Categories1': dm_services_fuels_eud_cantons_FSO.col_labels['Categories2']})
  dm_eff.array = np.nan_to_num(x=dm_eff.array, nan=1)
  dm_demand = dm_services_fuels_eud_cantons_FSO.copy()
  dm_demand.filter({'Years': years_ots}, inplace=True)
  assert dm_eff.col_labels['Country'] == dm_demand.col_labels['Country']
  dm_demand[:, :, :, :, :] = dm_demand[:, :, :, :, :] * dm_eff[:, :, :, np.newaxis, : ]

  dm_tot_demand = dm_demand.group_all('Categories2', inplace=False)
  dm_tech_mix = dm_demand.normalise('Categories2', inplace=False, keep_original=False)

  # Extrapolate 1990-2000 years
  linear_fitting(dm_tech_mix, years_ots)
  dm_tech_mix.array = np.maximum(dm_tech_mix.array, 0)
  dm_tech_mix.normalise('Categories2')

  idx = dm_tot_demand.idx
  dm_tot_demand[:, 0:idx[2001], ...] = np.nan
  linear_fitting(dm_tot_demand, years_ots)
  dm_tot_demand.array = np.maximum(dm_tot_demand.array, 0)


  #dm_fuels_eud_cantons.flattest().datamatrix_plot({'Country': ['Switzerland']})
  DM = {'services_demand': dm_tot_demand,
        'services_tech-mix': dm_tech_mix,
        'services_efficiencies': dm_efficiency}

  return DM


if __name__ == '__main__':
  years_ots = create_years_list(1990, 2023, 1)

  DM = run(years_ots)
