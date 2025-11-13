from model.common.auxiliary_functions import save_url_to_file
from _database.pre_processing.transport.Switzerland.get_data_functions import utils
import pandas as pd

def get_pkm_cap_aviation(file_url, local_filename):
  # Extract CH pkm/cap (data available every 5 years)
  save_url_to_file(file_url, local_filename)
  df = pd.read_excel(local_filename)
  filter_1 = [
    'Average annual mobility per person3 by aeroplane (within Switzerland and abroad), in km']
  new_names_1 = ['aviation']
  names_map = dict()
  for i, row in enumerate(filter_1):
    names_map[row] = new_names_1[i]
  var_name = 'tra_pkm-cap'
  unit = 'pkm/cap'
  header_row = 2
  dm_pkm = utils.df_fso_excel_to_dm(df, header_row, names_map, var_name, unit,
                              num_cat=1)
  return dm_pkm

def get_world_pop(pop_url, local_filename):
  save_url_to_file(pop_url, local_filename)
  df_pop = pd.read_excel(local_filename, sheet_name='Data')
  filter = ['World']
  new_names = ['lfs_population_total']
  names_map = dict()
  for i, row in enumerate(filter):
    names_map[row] = new_names[i]
  var_name = 'tra_passenger'
  unit = 'number'
  header_row = 2
  dm_pop = utils.df_fso_excel_to_dm(df_pop, header_row, names_map, var_name, unit,
                              num_cat=0, keep_first=True,
                              country='World')
  return dm_pop

def get_aviation_fleet(file_url, local_filename):
  # Extract CH pkm/cap (data available every 5 years)
  save_url_to_file(file_url, local_filename)
  df = pd.read_excel(local_filename)
  filter_1 = ['MTOM2 > 5700 kg']
  new_names_1 = ['aviation']
  names_map = dict()
  for i, row in enumerate(filter_1):
    names_map[row] = new_names_1[i]
  var_name = 'tra_passenger_vehicle-fleet'
  unit = 'number'
  header_row = 2
  dm_fleet_aviation = utils.df_fso_excel_to_dm(df, header_row, names_map, var_name,
                                         unit, num_cat=1, keep_first=True)

  return dm_fleet_aviation
