import pickle
import os

from model.common.auxiliary_functions import create_years_list, \
  save_url_to_file, load_pop, dm_add_missing_variables, linear_fitting
import numpy as np

import _database.pre_processing.buildings.Switzerland.get_data_functions.hot_water_CH as hw
from _database.pre_processing.buildings.Switzerland.get_data_functions.floor_area_CH import extract_nb_of_apartments_per_building_type
from model.common.data_matrix_class import DataMatrix

def adjust_tech_mix(dm_tech_mix, dm_apt):

  dm_tech_mix.filter({'Years': dm_apt.col_labels['Years']}, inplace=True)
  dm_tech_mix.drop('Categories2', 'other')
  dm_apt.sort('Country')
  dm_tech_mix.sort('Country')
  if dm_apt.col_labels['Country'] != dm_tech_mix.col_labels['Country']:
    raise ValueError('dm_apt and dm_tech_mix do not have the same countries')

  arr = (dm_apt[:, :, 'bld_apartments', :, np.newaxis]
         * dm_tech_mix[:, :, 'bld_hot-water_tech', :, :])

  dm_tech_mix.add(arr, dim='Variables', col_label= 'bld_hot-water_tech-mix', unit='number')
  dm_tech_mix.filter({'Variables': ['bld_hot-water_tech-mix']}, inplace=True)
  dm_tech_mix.group_all('Categories1', inplace=True)
  dm_tech_mix.normalise(dim='Categories1', keep_original=False, inplace=True)

  return dm_tech_mix


def distribute_hw_energy_consumption_to_cantons(dm_water_CH_consumption, dm_tech_mix, dm_pop):

  dm_tech_mix.filter({'Categories1': dm_water_CH_consumption.col_labels[
                        'Categories1']}, inplace=True)
  dm_tech_mix.sort('Categories1')
  dm_water_CH_consumption.sort('Categories1')

  dm_tech = dm_tech_mix.copy()

  assert dm_pop.col_labels['Country'] == dm_tech.col_labels['Country']

  arr = (dm_tech[:, :, 'bld_hot-water_tech-mix', :]
         * dm_pop[:, :, 'lfs_population_total', np.newaxis])
  dm_tech.add(arr, dim='Variables', col_label='bld_hw_pop_by_tech', unit='people')



  dm_tech.drop('Country', 'Switzerland')
  dm_tech.normalise(dim='Country')
  dm_tech.filter({'Years': dm_water_CH_consumption.col_labels['Years']}, inplace=True)

  assert dm_water_CH_consumption.col_labels['Categories1'] == \
         dm_tech.col_labels['Categories1']

  arr = (dm_tech[:, :, 'bld_hw_pop_by_tech', :]
         * dm_water_CH_consumption['Switzerland', np.newaxis, :,'bld_hot-water_energy-consumption',:])

  dm_tech.add(arr, col_label='bld_hot-water_energy-consumption',
                  dim='Variables', unit='TWh')
  dm_canton_share = dm_tech.filter({'Variables': ['bld_hw_pop_by_tech']})
  dm_water_consumption = dm_tech.filter({'Variables': ['bld_hot-water_energy-consumption']})

  return dm_water_consumption, dm_canton_share


def extrapolate_missing_years_based_on_per_capita(dm_apt, dm_pop, years_ots):
  assert dm_pop.col_labels['Country'] == dm_apt.col_labels['Country']
  dm_add_missing_variables(dm_apt, {'Years': years_ots}, fill_nans=False)
  arr = dm_apt[:, :, 'bld_apartments', :] / dm_pop[:, :, 'lfs_population_total', np.newaxis]
  dm_apt.add(arr, dim='Variables', col_label='bld_apt_per_cap', unit='apt/cap')
  linear_fitting(dm_apt, years_ots)
  dm_apt[:, :, 'bld_apartments', :] =(dm_apt[:, :, 'bld_apt_per_cap', :] *
                                      dm_pop[:, :, 'lfs_population_total', np.newaxis])
  dm_apt.filter({'Variables': ['bld_apartments']}, inplace=True)

  return dm_apt


def extrapolate_missing_years_based_on_tech_mix(dm_water_CH_consumption, dm_pop, years_ots):

  dm_cons = dm_water_CH_consumption.group_all('Categories1', inplace=False)
  dm_add_missing_variables(dm_cons, {'Years': years_ots}, fill_nans=False)
  dm_cons.append(dm_pop.filter({'Country': ['Switzerland']}), dim='Variables')
  dm_cons.operation('bld_hot-water_energy-consumption', '/', 'lfs_population_total', out_col='bld_energy-demand-cap', unit='TWh/cap')
  linear_fitting(dm_cons, years_ots)
  dm_cons[:, :, 'bld_hot-water_energy-consumption'] = (
    dm_cons[:, :, 'lfs_population_total'] * dm_cons[:, :, 'bld_energy-demand-cap'])

  dm_cons.filter({'Variables': ['bld_hot-water_energy-consumption']}, inplace = True)

  dm_tech = dm_water_CH_consumption.normalise(dim='Categories1', inplace=False)
  linear_fitting(dm_tech, years_ots)
  dm_tech.array = np.maximum(0, dm_tech.array)
  dm_tech.normalise(dim='Categories1', inplace=True)
  dm_tech.rename_col('bld_hot-water_energy-consumption_share', 'bld_hot-water_tech', dim='Variables')

  dm_add_missing_variables(dm_water_CH_consumption, {'Years': years_ots}, fill_nans=False)
  dm_water_CH_consumption[:, :, 'bld_hot-water_energy-consumption', :] \
    = dm_tech[:, :, 'bld_hot-water_tech', :] * dm_cons[:, :, 'bld_hot-water_energy-consumption', np.newaxis]

  return dm_water_CH_consumption

def adjust_heat_pumps_hot_water_COP(dm_efficiencies, COP_HP_HW):
  dm = dm_efficiencies.copy()
  COP_HP_HW = {2015: 2.8,  2030: 3.0,	2040: 3.2, 2050: 3.5, 2060:3.6}
  dm.add(np.nan, dim='Categories1', col_label='heat-pump-adj', dummy=True)
  for yr, val in COP_HP_HW.items():
    dm['Switzerland', yr, 'bld_heating_efficiency', 'heat-pump-adj'] = val
  dm.operation('heat-pump-adj', '/', 'heat-pump', out_col='ratio', dim='Categories1')
  dm.drop(dim='Categories1', col_label='heat-pump-adj')
  linear_fitting(dm, years_ots=dm.col_labels['Years'])
  dm.operation('heat-pump', '*', 'ratio', out_col='heat-pump-adj', dim='Categories1')
  dm.drop(dim='Categories1', col_label=['heat-pump', 'ratio'])
  dm.rename_col('heat-pump-adj', 'heat-pump', 'Categories1')
  return dm

def run(country_list, years_ots):

  cantons_en = ['Aargau', 'Appenzell Ausserrhoden', 'Appenzell Innerrhoden',
                'Basel Landschaft', 'Basel Stadt', 'Bern', 'Fribourg', 'Geneva',
                'Glarus', 'Graubünden', 'Jura', 'Lucerne', 'Neuchâtel',
                'Nidwalden', 'Obwalden', 'Schaffhausen', 'Schwyz', 'Solothurn',
                'St. Gallen', 'Thurgau', 'Ticino', 'Uri', 'Valais', 'Vaud',
                'Zug', 'Zurich']
  cantons_fr =  ['Argovie', 'Appenzell Rh. Ext.', 'Appenzell Rh. Int.',
                 'Bâle Campagne', 'Bâle Ville', 'Berne', 'Fribourg', 'Genève',
                 'Glaris', 'Grisons', 'Jura', 'Lucerne', 'Neuchâtel', 'Nidwald',
                 'Obwald', 'Schaffhouse', 'Schwytz', 'Soleure', 'Saint Gall',
                 'Thurgovie', 'Tessin', 'Uri', 'Valais', 'Vaud', 'Zoug', 'Zurich']

  this_dir = os.path.dirname(os.path.abspath(__file__))

  dm_pop = load_pop(country_list=cantons_en + ['Switzerland'], years_list=years_ots)

  ##################################
  ####      TECHNOLOGY-MIX     #####
  ##################################
  # Get Hot water fuel split at household level per canton
  table_id = 'px-x-0902010000_102'
  file_hw = os.path.join(this_dir, '../data/bld_hotwater_technology_2021-2023.pickle')
  # Extract water tech share based on number of buildings
  # It only has data for the last 3 years
  #!FIXME extract 2000 and 1990
  dm_tech_mix = hw.extract_hotwater_technologies(table_id, file_hw)

  # Tech mix 1990 and 2000
  table_id = 'px-x-0902020100_112'
  file_hw = os.path.join(this_dir, '../data/bld_hotwater_technology_1990_2000_old.pickle')
  dm_tech_mix_old = hw.extract_hotwater_technologies_old(table_id, file_hw)
  dm_tech_mix_old.drop('Categories2', 'coal')
  dm_tech_mix.append(dm_tech_mix_old, dim='Years')
  # tech share
  # Add missing years
  dm_add_missing_variables(dm_tech_mix, {'Years': years_ots}, fill_nans=True)
  #dm_tech_mix.normalise('Categories2', inplace=True, keep_original=False)

  # Extract number of apartments per building type
  table_id = 'px-x-0902020200_103'
  file = os.path.join(this_dir, '../data/bld_apartments_per_bld_type.pickle')
  dm_apt = extract_nb_of_apartments_per_building_type(table_id, file, cantons_fr, cantons_en)
  # Extrapolate number of apartments based on apt/pop
  dm_apt = extrapolate_missing_years_based_on_per_capita(dm_apt, dm_pop, years_ots)

  # Weight the technology mix by the number of households/apartments
  dm_tech_mix = adjust_tech_mix(dm_tech_mix, dm_apt)
  dm_add_missing_variables(dm_tech_mix, {'Years': years_ots}, fill_nans=True)

  ##################################
  ####        EFFICIENCY       #####
  ##################################
  # I use the space heating efficiency as a proxy for the hot water efficiency
  file_url = 'https://www.bfe.admin.ch/bfe/de/home/politik/energieperspektiven-2050-plus.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvMTA0NDE=.html'
  zip_name = os.path.join(this_dir, '../data/EP2050_sectors.zip')
  file_pickle = os.path.join(this_dir, '../data/bld_heating_efficiencies.pickle')
  dm_efficiencies = hw.extract_heating_efficiencies_EP2050(file_url, zip_name, file_pickle)
  dm_efficiencies.add(1, dim='Categories1', col_label=['electricity'], dummy=True)
  dm_efficiencies.add(0.6, dim='Categories1', col_label=['solar'], dummy=True)
  linear_fitting(dm_efficiencies, years_ots, based_on=[2000, 2010])
  # For heat-pump, according to EP2050+
  # source: EP2050+_TechnsicherBericht_DatenAbbildungen_Kap 1-7_2022-04-12, Abb. 17
  # Wärmenutzungsgrad von Wärmepumpen für Warmwasser
  # The efficiency for Heat-pumps (air to water) for hot water is
  COP_HP_HW = {2015: 2.8,  2030: 3.0,	2040: 3.2, 2050: 3.5, 2060:3.6}
  dm_efficiencies = adjust_heat_pumps_hot_water_COP(dm_efficiencies, COP_HP_HW)
  dm_efficiencies.sort('Categories1')


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
  dm_water_CH_consumption.drop('Categories1', 'ambient-heat')
  dm_water_CH_consumption = extrapolate_missing_years_based_on_tech_mix(dm_water_CH_consumption, dm_pop, years_ots)
  # The energy consumption by fuel and the apt share by fuel do not match. In particular,
  # heating oil share of energy consumption is lower than apt share while gas share is higher
  # This does not make sense as gas is a more efficient technology,
  # and between energy consumption share and fuel share only efficiency should play a role
  # But we are considering that all mfh buildings have the same number of apt.
  # In reality it is likely that gas-heated buildings are bigger than heating-oil heated buildings


  ##################################
  ####      CANTONAL SPLIT      ####
  ##################################
  # Split hot water useful energy demand in CH to canton using same per capita
  dm_water_consumption, dm_canton_share = distribute_hw_energy_consumption_to_cantons(dm_water_CH_consumption, dm_tech_mix, dm_pop)
  dm_water_consumption.append(dm_water_CH_consumption, dim='Country')

  ######################################
  ####      USEFUL-ENERGY - CH     #####
  ######################################
  # Go from energy consumption to useful energy
  dm_efficiencies.sort('Categories1')
  assert dm_water_consumption.col_labels['Categories1'] == dm_efficiencies.col_labels['Categories1']
  dm_efficiencies.filter({'Years': dm_water_CH_consumption.col_labels['Years']}, inplace=True)

  arr = (dm_water_consumption[:, :, 'bld_hot-water_energy-consumption', :]
         * dm_efficiencies['Switzerland', np.newaxis, :, 'bld_heating_efficiency', :])
  dm_water_consumption.add(arr, dim='Variables', col_label='bld_hw_useful-energy', unit='TWh')

  dm_water_tot = dm_water_consumption.filter({'Variables': ['bld_hw_useful-energy']})
  dm_water_tot.group_all('Categories1', inplace=True)


  # Compute hot water useful energy per capita
  dm_water_tot.append(dm_pop.filter({'Years': dm_water_tot.col_labels['Years']}), dim='Variables')
  dm_water_tot.operation('bld_hw_useful-energy', '/', 'lfs_population_total',  out_col='bld_hw_demand', unit='TWh/cap')
  dm_water_tot.change_unit('bld_hw_demand', old_unit='TWh/cap', new_unit='MWh/cap', factor=1e6)


  # Compute demand per technology based on tech-mix of useful energy
  # These data and the tech_mix from the households do not match at all,
  # I think this data are better: more years available to begin with
  dm_tech_mix_useful = dm_water_consumption.filter({'Variables': ['bld_hw_useful-energy']})
  dm_tech_mix_useful.normalise('Categories1')
  dm_tech_mix_useful.rename_col('bld_hw_useful-energy', 'bld_hw_tech-mix', dim='Variables')
  linear_fitting(dm_tech_mix_useful, years_ots)
  dm_tech_mix_useful.array = np.maximum(0, dm_tech_mix_useful.array)
  dm_tech_mix_useful.normalise('Categories1')

  dm_add_missing_variables(dm_efficiencies, dict_all={'Country': cantons_en}, fill_nans=True)

  dm_efficiencies.rename_col('bld_heating_efficiency', 'bld_hot-water_efficiency', dim='Variables')

  DM = {'hw-tech-mix': dm_tech_mix_useful.filter({'Country': country_list}),
        'hw-efficiency': dm_efficiencies.filter({'Country': country_list}),
        'hw-energy-demand': dm_water_tot.filter({'Variables': ['bld_hw_demand'], 'Country': country_list})}

  return DM


if __name__ == "__main__":
  cantons_en = ['Aargau', 'Appenzell Ausserrhoden', 'Appenzell Innerrhoden',
                'Basel Landschaft', 'Basel Stadt', 'Bern', 'Fribourg', 'Geneva',
                'Glarus', 'Graubünden', 'Jura', 'Lucerne', 'Neuchâtel',
                'Nidwalden', 'Obwalden', 'Schaffhausen', 'Schwyz', 'Solothurn',
                'St. Gallen', 'Thurgau', 'Ticino', 'Uri', 'Valais', 'Vaud',
                'Zug', 'Zurich']
  country_list = cantons_en + ['Switzerland']
  years_ots = create_years_list(1990, 2023, 1)
  years_fts = create_years_list(2025, 2050, 5)
  DM = run(country_list, years_ots)
