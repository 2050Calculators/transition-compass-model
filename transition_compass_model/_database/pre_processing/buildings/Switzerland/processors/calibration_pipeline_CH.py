import os
from _database.pre_processing.buildings.Switzerland.get_data_functions import energy_demand_for_calibration as get_data
from model.common.auxiliary_functions import create_years_list
import numpy as np

def run(country_list, years_ots):
  # Extract energy demand of households (incl. hot-water, heating, lighting, appliances)
  this_dir = os.path.dirname(os.path.abspath(__file__))
  file_url = 'https://www.bfe.admin.ch/bfe/fr/home/versorgung/statistik-und-geodaten/energiestatistiken/gesamtenergiestatistik.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZnIvcHVibGljYX/Rpb24vZG93bmxvYWQvNzUxOQ==.html'
  local_filename =  os.path.join(this_dir, '../data/statistique_globale_suisse_energie.xlsx')
  parameters = dict()
  mapping = {'heating-oil': '.*Produits pétroliers.*',
             'gas': '.*Gaz.*',
             'coal': '.*Charbon.*',
             'wood': '.*bois.*',
             'district-heating': '.*distance.*',
             'renewables': '.*renouvelables.*',
             'electricity': '.*Electricité.*'}
  # renewables include: soleil, énergie éolienne, biogaz, chaleur ambiante.
  # It is weird because we also have electricity, that includes wind,
  # and ok maybe solar here is not PV but solar heat, but wind ??
  # It would be useful to know biogaz as a separate
  parameters['mapping'] = mapping  # dictionary,  to rename column headers
  parameters['var name'] = 'bld_energy-demand_residential'  # string, dm variable name
  parameters['headers indexes'] = (8, 9, 10)  # tuple with index of rows to keep for header
  parameters['first row'] = 11  # integer with the first row to keep
  parameters['unit'] = None
  parameters['cols to drop'] = '.*%.*|.*extra.*'
  # !FIXME: il y a un problème avec déchets industriels
  dm_residential_energy = get_data.extract_energy_statistics_data(file_url, local_filename,sheet_name='T17a', parameters=parameters, years_ots=years_ots)


  # Extract energy demand of services (incl. hot-water, heating, lighting, appliances)
  file_url = 'https://www.bfe.admin.ch/bfe/fr/home/versorgung/statistik-und-geodaten/energiestatistiken/gesamtenergiestatistik.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZnIvcHVibGljYX/Rpb24vZG93bmxvYWQvNzUxOQ==.html'
  local_filename =  os.path.join(this_dir, '../data/statistique_globale_suisse_energie.xlsx')
  parameters = dict()
  mapping = {'heating-oil': '.*Produits pétroliers.*',
             'gas': '.*Gaz.*',
             'coal': '.*Charbon.*',
             'wood': '.*bois.*',
             'district-heating': '.*distance.*',
             'renewables': '.*renouvelables.*',
             'electricity': '.*Electricité.*'}
  # renewables include: soleil, énergie éolienne, biogaz, chaleur ambiante.
  parameters['mapping'] = mapping  # dictionary,  to rename column headers
  parameters['var name'] = 'bld_energy-demand_services'  # string, dm variable name
  parameters['headers indexes'] = (8, 9, 10)  # tuple with index of rows to keep for header
  parameters['first row'] = 11  # integer with the first row to keep
  parameters['unit'] = None
  parameters['cols to drop'] = '.*%.*|.*extra.*'
  dm_services_energy = get_data.extract_energy_statistics_data(file_url, local_filename,sheet_name='T17c', parameters=parameters, years_ots=years_ots)

  dm_cal_energy = dm_residential_energy.copy()
  dm_cal_energy.append(dm_services_energy, dim='Variables')
  #dm_cal_energy.datamatrix_plot()

  for cntr in country_list:
    if cntr not in dm_cal_energy.col_labels['Country']:
      dm_cal_energy.add(np.nan, col_label=cntr, dim='Country', dummy=True)

  dm_cal_energy.sort('Country')
  dm_cal_energy.sort('Categories1')
  return dm_cal_energy

if __name__ == '__main__':
  years_ots = create_years_list(1990, 2023, 1)
  run(country_list=['Switzerland'], years_ots=years_ots)
