import os.path

import numpy as np
import pickle

from model.common.auxiliary_functions import linear_fitting, create_years_list, my_pickle_dump, add_dummy_country_to_DM

from _database.pre_processing.buildings.Switzerland.get_data_functions.appliances_CH import (
  save_url_to_file, get_appliances_stock_new_energy, get_households_number,
  clean_country_names, households_fill_missing_years, appliances_fill_missing_years)

from model.common.auxiliary_functions import load_pop

def run(dm_pop, country_list, years_ots, years_fts):

  this_dir = os.path.dirname(os.path.abspath(__file__))
  years_fts_orig = years_fts
  years_fts = create_years_list(years_ots[-1]+1, years_fts_orig[-1], 1)

  missing_years = list(set(years_fts) - set(years_fts_orig))
  dm_pop.add(np.nan, dim='Years', col_label=missing_years, dummy = True)
  dm_pop.sort('Years')
  dm_pop.fill_nans('Years')

  ########################################################
  ###    APPLIANCES NEW, STOCK, ENERGY CONSUMPTION    ####
  ########################################################

  # Opensuisse database with data on large appliances and electronics in Switzerland
  # Containing data on stock, new, and their energy consumption
  # https://opendata.swiss/de/dataset/absatz-und-stromverbrauchswerte-von-elektro-und-elektronischen-gerate-in-der-schweiz
  # DATA - Absatz- und Stromverbrauchswerte von Elektro- und elektronischen Geräte in der Schweiz
  file_url = "https://www.uvek-gis.admin.ch/BFE/ogd/109/ogd109_absatz_verbrauch_elektrogeraete.csv"

  raw_filename = os.path.join(this_dir, '../data/appliances_stock_new_energy_CH_raw.csv')
  save_url_to_file(file_url, raw_filename)
  clean_filename = os.path.join(this_dir, '../data/appliances_stock_new_energy_CH.pickle')

  dm_appliances = get_appliances_stock_new_energy(raw_filename, clean_filename)

  # Filter & Rename only the appliances you want to keep
  app_map = {'freezer': ['Freezers'], 'refrigerator': ['Refrigerators'],
             'dishwasher': ['Dishwasher'], 'washing-machine': ['Washing machines'],
             'oven-and-stove': ['Electric cookers and built-in ovens'],
             'tumble-dryer': ['Tumble dryer'], 'TV': ['Televisions'],
             'PC': ['Personal Computer'], 'laptop': ['Laptops'],
             'monitor': ['Monitor'], 'set-top-box': ['Set-top boxes']}
  dm_appliances = dm_appliances.groupby(app_map, dim='Categories1', inplace=False)

  missing_years = list(set(years_ots + years_fts) - set(dm_appliances.col_labels['Years']))
  dm_appliances.add(np.nan, dim='Years', col_label=missing_years, dummy=True)
  dm_appliances.sort('Years')

  # Mobile phones are missing but their electricity consumption seems minimal. 31 kWh/year vs 2000 kWh/year

  # Number of households
  file_url = 'https://dam-api.bfs.admin.ch/hub/api/dam/assets/27965837/master'
  raw_filename= os.path.join(this_dir, '../data/households_number.xlsx')
  save_url_to_file(file_url, raw_filename)
  clean_filename = os.path.join(this_dir, '../data/households_number.pickle')

  dm_households = get_households_number(raw_filename, clean_filename)
  dm_households = clean_country_names(dm_households)
  dm_households.filter({'Country': country_list}, inplace=True)

  # Compute number of households from population by linear fitting of ppl/households
  dm_households = households_fill_missing_years(dm_households, dm_pop, years_ots, years_fts)

  # Map appliances from CH to cantons based on number of households
  dm_households_CH = dm_households.filter({'Country': ['Switzerland']})

  # Linear fitting based on appliances / households from the 2002-2005 trend
  dm_appliances.operation('bld_appliances_electricity-demand_stock', '/', 'bld_appliances_stock', out_col='bld_appliances_electricity-demand', unit='kWh/unit')
  dm_appliances.filter({'Variables': ['bld_appliances_stock', 'bld_appliances_new', 'bld_appliances_electricity-demand']}, inplace=True)

  # Determine the waste
  # s(t) = s(t-1) + n(t) - w(t)
  # w(t) = s(t-1) + n(t) - s(t)
  dm_appliances.lag_variable('bld_appliances_stock', shift=1, subfix='_tm1')
  dm_appliances.operation('bld_appliances_stock_tm1', '-', 'bld_appliances_stock', out_col='bld_delta_stock', unit='unit')
  dm_appliances.operation('bld_delta_stock', '+', 'bld_appliances_new', out_col='bld_appliances_waste', unit='unit')
  dm_appliances.operation('bld_appliances_waste', '/', 'bld_appliances_stock_tm1', out_col='bld_appliances_retirement-rate', unit='%')

  dm_rr = dm_appliances.filter({'Variables': ['bld_appliances_retirement-rate']})
  dm_rr.fill_nans('Years')

  dm_appliances.filter({'Variables': ['bld_appliances_stock', 'bld_appliances_electricity-demand']}, inplace=True)

  # Fill nans for stock and energy demand/unit
  dm_appliances = appliances_fill_missing_years(dm_appliances, dm_households_CH, years_ots, years_fts)
  dm_appliances.append(dm_rr, dim='Variables')

  dm_appliances.lag_variable('bld_appliances_stock', shift=1, subfix='_tm1')
  dm_appliances.operation('bld_appliances_retirement-rate', '*', 'bld_appliances_stock_tm1', out_col='bld_appliances_waste', unit='%')
  # Determine the waste
  # s(t) = s(t-1) + n(t) - w(t)
  # n(t) = w(t) - s(t-1) + s(t)
  arr = (dm_appliances[:, :, 'bld_appliances_waste', :]
         - dm_appliances[:, :, 'bld_appliances_stock_tm1', :]
         + dm_appliances[:, :, 'bld_appliances_stock', :])
  dm_appliances.add(arr, dim='Variables', col_label='bld_appliances_new', unit='unit')

  dm_appliances[:, :, 'bld_appliances_new', :] \
    = np.maximum(dm_appliances[:, :, 'bld_appliances_new', :], 0)

  for yr in dm_appliances.col_labels['Years'][1:]:
    dm_appliances[:, yr, 'bld_appliances_stock', :] \
      = (dm_appliances[:, yr-1, 'bld_appliances_stock', :]
         + dm_appliances[:, yr, 'bld_appliances_new', :]
         - dm_appliances[:, yr-1, 'bld_appliances_stock', :]
         * dm_appliances[:, yr, 'bld_appliances_retirement-rate', :])

  dm_appliances.filter({'Variables': ['bld_appliances_stock', 'bld_appliances_retirement-rate', 'bld_appliances_electricity-demand']}, inplace=True)

  # Compute stock per households
  dm_appliances['Switzerland', :, 'bld_appliances_stock', :] \
    = (dm_appliances['Switzerland', :, 'bld_appliances_stock', :]
       / dm_households_CH['Switzerland', :, 'bld_households', np.newaxis])
  dm_appliances.change_unit('bld_appliances_stock', factor=1, old_unit='number', new_unit='unit/household')

  # Add all cantons
  cantons_list = list(set(country_list) - {'Switzerland'})
  dm_appliances.add(np.nan, dim='Country', dummy=True, col_label=cantons_list)
  dm_appliances.fill_nans('Country')

  # Prepare as output the stock/household, the retirement-rate,
  # the households size, the energy-demand
  idx = dm_appliances.idx
  dm_appliances.array[:, idx[2023]:, idx['bld_appliances_electricity-demand'], idx['oven-and-stove']] = np.nan
  linear_fitting(dm_appliances, years_ots+years_fts)

  dm_households.filter({'Variables':['lfs_household-size']})

  DM = {'fxa':
          {
            'appliances': dm_appliances.filter({'Years':years_ots + years_fts_orig}),
            'household': dm_households.filter({'Years':years_ots + years_fts_orig})
          }
  }

  # !FIXME add this to the pickle
  return DM


if __name__ == "__main__":

  years_ots = create_years_list(1990, 2023, 1)
  years_fts = create_years_list(2025, 2050, 5)

  country_list = ['Aargau', 'Appenzell Ausserrhoden', 'Appenzell Innerrhoden',
                  'Basel Landschaft', 'Basel Stadt', 'Bern', 'Fribourg', 'Geneva',
                  'Glarus', 'Graubünden', 'Jura', 'Lucerne', 'Neuchâtel',
                  'Nidwalden', 'Obwalden', 'Schaffhausen', 'Schwyz', 'Solothurn',
                  'St. Gallen', 'Thurgau', 'Ticino', 'Uri', 'Valais', 'Vaud',
                  'Zug', 'Zurich']+ ['Switzerland']

  dm_pop = load_pop(country_list=country_list, years_list=years_ots+years_fts)

  DM = run(dm_pop, country_list, years_ots, years_fts)
