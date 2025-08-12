import os

import numpy as np

from model.common.auxiliary_functions import save_url_to_file, linear_fitting, create_years_list, load_pop
import _database.pre_processing.buildings.Switzerland.get_data_functions.lighting_CH as lt
from src.api.routes import country_list


def run(country_list, years_ots):
  this_dir = os.path.dirname(os.path.abspath(__file__))
  file_url = "https://www.bfe.admin.ch/bfe/fr/home/versorgung/statistik-und-geodaten/energiestatistiken/energieverbrauch-nach-verwendungszweck.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvMTE5MzU=.html"
  local_filename = os.path.join(this_dir, '../data/EP2050_Households_CH.xlsx')
  save_url_to_file(file_url, local_filename)
  file_pickle = os.path.join(this_dir, '../data/EP2050_lighting_households_CH.pickle')
  dm_lighting_CH_consumption = lt.extract_EP2050_lighting_energy_consumption(file_raw=local_filename, file_pickle=file_pickle)
  linear_fitting(dm_lighting_CH_consumption, years_ots, based_on=list(range(2000, 2004)))


  dm_pop = load_pop(country_list, years_ots)
  if 'Switzerland' in dm_pop.col_labels['Country']:
    dm_pop.drop('Country', 'Switzerland')

  dm_pop_CH = load_pop(['Switzerland'], years_list=years_ots)
  dm_pop.array = dm_pop.array / dm_pop_CH.array
  dm_shares = dm_pop.copy()
  dm_shares.sort('Country')

  arr = dm_shares[:, :, :] * dm_lighting_CH_consumption['Switzerland', np.newaxis, :, :]
  dm_lighting_CH_consumption.add(arr, dim='Country', col_label = dm_shares.col_labels['Country'])

  dm_lighting_CH_consumption.sort('Country')

  return dm_lighting_CH_consumption

if __name__ == '__main__':
  years_ots = create_years_list(1990,2023,1)
  cantons_en = ['Aargau', 'Appenzell Ausserrhoden', 'Appenzell Innerrhoden',
                'Basel Landschaft', 'Basel Stadt', 'Bern', 'Fribourg', 'Geneva',
                'Glarus', 'Graubünden', 'Jura', 'Lucerne', 'Neuchâtel',
                'Nidwalden', 'Obwalden', 'Schaffhausen', 'Schwyz', 'Solothurn',
                'St. Gallen', 'Thurgau', 'Ticino', 'Uri', 'Valais', 'Vaud',
                'Zug', 'Zurich']
  country_list = cantons_en + ['Switzerland']
  dm_lighting = run(country_list, years_ots)
