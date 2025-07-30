import pickle
import os

from model.common.auxiliary_functions import create_years_list, save_url_to_file, load_pop, dm_add_missing_variables
import numpy as np

import _database.pre_processing.buildings.Switzerland.get_data_functions.hot_water_CH as hw


def run(country_list, years_ots, years_fts):

  this_dir = os.path.dirname(os.path.abspath(__file__))

  ##################################
  ####      TECHNOLOGY-MIX     #####
  ##################################
  # Get Hot water fuel split at household level per canton
  table_id = 'px-x-0902010000_102'
  file_hw = os.path.join(this_dir, '../data/bld_hotwater_technology_2021-2023.pickle')
  # Extract water tech share based on number of buildings
  dm_tech_mix = hw.extract_hotwater_technologies(table_id, file_hw)  # tech share
  # Add missing years
  dm_add_missing_variables(dm_tech_mix, {'Years': years_ots}, fill_nans=True)


  ############################################
  ####      ENERGY CONSUMPTION - CH      #####
  ############################################
  ## Get water tech share based on energy consumption
  # Hot water demand in CH - use efficiency to determine useful energy demand
  # From Prognos EP2050 Analysis: https://www.bfe.admin.ch/bfe/fr/home/approvisionnement/statistiques-et-geodonnees/statistiques-de-lenergie/consommation-energetique-en-fonction-de-lapplication.html
  file_url = "https://www.bfe.admin.ch/bfe/fr/home/versorgung/statistik-und-geodaten/energiestatistiken/energieverbrauch-nach-verwendungszweck.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvMTE5MzU=.html"
  local_filename = os.path.join(this_dir, '../data/EP2050_Households_CH.xlsx')
  save_url_to_file(file_url, local_filename)
  file_pickle = os.path.join(this_dir, '../data/EP2050_hot_water_households_CH.pickle')
  dm_water_CH_consumption = hw.extract_EP2050_hot_water_energy_consumption(file_raw=local_filename, file_pickle=file_pickle)

  ##################################
  ####        EFFICIENCY       #####
  ##################################
  # I use the space heating efficiency as a proxy for the hot water efficiency
  file_url = 'https://www.bfe.admin.ch/bfe/de/home/politik/energieperspektiven-2050-plus.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvMTA0NDE=.html'
  zip_name = os.path.join(this_dir, '../data/EP2050_sectors.zip')
  file_pickle = os.path.join(this_dir, '../data/bld_heating_efficiencies.pickle')
  dm_efficiencies = hw.extract_heating_efficiencies_EP2050(file_url, zip_name, file_pickle)
  dm_efficiencies.add(1, dim='Categories1', col_label=['solar', 'electricity'], dummy=True)

  ######################################
  ####      USEFUL-ENERGY - CH     #####
  ######################################
  # Go from energy consumption to useful energy
  dm_water_CH_consumption.drop(dim='Categories1', col_label='ambient-heat')
  dm_water_CH_consumption.append(dm_efficiencies.filter({'Years': dm_water_CH_consumption.col_labels['Years']}), dim='Variables')

  dm_water_CH_consumption.operation('bld_hot-water_energy-consumption', '*', 'bld_heating_efficiency', out_col='bld_hw_useful-energy', unit='TWh')
  dm_water_CH_tot = dm_water_CH_consumption.filter({'Variables': ['bld_hw_useful-energy']})
  dm_water_CH_tot.group_all('Categories1', inplace=True)


  ##################################
  ####      CANTONAL SPLIT      ####
  ##################################
  # Split hot water useful energy demand in CH to canton using same per capita
  dm_pop = load_pop(country_list=country_list, years_list=years_ots+years_fts)

  dm_pop_CH = dm_pop.filter({'Country': ['Switzerland'],
                            'Years': dm_water_CH_tot.col_labels['Years']})

  arr = dm_water_CH_tot[:, :, 'bld_hw_useful-energy'] / dm_pop_CH[:, :, 'lfs_population_total']
  dm_water_CH_tot.add(arr, col_label='bld_hw_demand', unit='TWh/cap', dim='Variables')
  dm_water_CH_tot.change_unit('bld_hw_demand', old_unit='TWh/cap', new_unit='MWh/cap', factor=1e6)

  # Apply tech share by bld and then adjust using efficiency

  # Adjust so that sum of canton of energy consumption by fuel is correct


  # Get lighting
  return


if __name__ == "__main__":
  cantons_en = ['Aargau', 'Appenzell Ausserrhoden', 'Appenzell Innerrhoden',
                'Basel Landschaft', 'Basel Stadt', 'Bern', 'Fribourg', 'Geneva',
                'Glarus', 'Graubünden', 'Jura', 'Lucerne', 'Neuchâtel',
                'Nidwalden', 'Obwalden', 'Schaffhausen', 'Schwyz', 'Solothurn',
                'St. Gallen', 'Thurgau', 'Ticino', 'Uri', 'Valais', 'Vaud',
                'Zug', 'Zurich']
  years_ots = create_years_list(1990, 2023, 1)
  years_fts = create_years_list(2025, 2050, 5)
  run(cantons_en+['Switzerland'], years_ots, years_fts)
