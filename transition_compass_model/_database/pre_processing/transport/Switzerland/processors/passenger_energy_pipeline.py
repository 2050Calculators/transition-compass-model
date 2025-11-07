#################################
####     ENERGY DEMAND      #####
#################################
import os
import _database.pre_processing.transport.Switzerland.get_data_functions.transport_energy as get_data
from model.common.auxiliary_functions import dm_add_missing_variables, create_years_list

def run(years_ots):

  this_dir = os.path.dirname(os.path.abspath(__file__))

  # Energy demand for LDV, 2W, bus by technology from EP2050
  # EP2050+_Detailergebnisse 2020-2060_Verkehrssektor_alle Szenarien_2022-04-12
  file_url = 'https://www.bfe.admin.ch/bfe/de/home/politik/energieperspektiven-2050-plus.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvMTA0NDE=.html'
  zip_name = os.path.join(this_dir, '../data/EP2050_sectors.zip')
  file_pickle = os.path.join(this_dir, '../data/tra_EP2050_energy_demand_private.pickle')
  dm_energy_private = get_data.extract_EP2050_transport_energy_demand(file_url, zip_name, file_pickle)

  # Energy demand for rail from EP2050
  # EP2050+_TechnsicherBericht_DatenAbbildungen_Kap 8_2022-04-12, table Abb. 112
  file_url = 'https://www.bfe.admin.ch/bfe/de/home/politik/energieperspektiven-2050-plus.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvMTA5MDQ=.html'
  zip_name = os.path.join(this_dir, '../data/EP2050_tables.zip')
  file_pickle = os.path.join(this_dir, '../data/tra_EP2050_energy_demand_rail.pickle')
  dm_energy_rail = get_data.extract_EP2050_transport_energy_demand_rail(file_url, zip_name, file_pickle)

  dm_passenger_rail = dm_energy_rail.filter({'Categories2': ['passenger']})
  # I allocate all energy demand for passenger rail transport to CEV
  dm_passenger_rail.rename_col('passenger', 'CEV', 'Categories2')

  # From private transport, drop freight to have only passenger
  dm_passenger_energy_all = dm_energy_private.filter({'Categories1': ['bus', '2W', 'LDV']})

  # Append rail to private transport
  all_cat = list(set(dm_passenger_energy_all.col_labels['Categories2']).union(
    set(dm_passenger_rail.col_labels['Categories2'])))
  dm_add_missing_variables(dm_passenger_energy_all, {'Categories2': all_cat}, fill_nans=False)
  dm_add_missing_variables(dm_passenger_rail, {'Categories2': all_cat}, fill_nans=False)
  dm_passenger_energy_all.append(dm_passenger_rail, dim='Categories1')
  dm_passenger_energy_all.filter({'Years': years_ots}, inplace=True)

  return dm_passenger_energy_all


if __name__ == "__main__":
  years_ots = create_years_list(1990, 2023, 1)
  run(years_ots)
