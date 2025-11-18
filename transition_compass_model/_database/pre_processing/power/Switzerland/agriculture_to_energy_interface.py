import numpy as np
import os
import pickle

from model.common.auxiliary_functions import create_years_list, linear_fitting, rename_cantons, sort_pickle, dm_add_missing_variables
from industry_to_energy_interface import (extract_national_energy_demand,
                                          extract_employees_per_sector_canton,
                                          map_national_energy_demand_by_sector_to_cantons)


######################################
##   AGRICULTURE TO ENERGY INTERFACE   ##
######################################

def run():
  years_ots = create_years_list(1990, 2023, 1)
  years_fts = create_years_list(2025, 2050, 5)

  # Extract energy demand by sector at national level by fuel
  table_id = 'px-x-0204000000_106'
  local_filename = 'data/energy_accounts_economy_households.pickle'
  # Remove gasoline and diesel which are for transport / machinery
  dm_energy = extract_national_energy_demand(table_id, local_filename)

  dm_energy_agriculture = dm_energy.groupby({'agriculture': [' 0103 Agriculture, sylviculture et pêche']}, dim='Categories1', inplace=False)

  # Extract number of employees per industry and service sector by canton
  # This is in order to map the national energy demand to cantons
  table_id = 'px-x-0602010000_101'
  local_filename = 'data/employees_per_sector_canton.pickle'
  dm_employees = extract_employees_per_sector_canton(table_id, local_filename)
  dm_employees_agriculture = dm_employees.groupby({'agriculture': ['01 Culture et production animale, chasse et services annexes',
                                                                   '02 Sylviculture et exploitation forestière',
                                                                   '03 Pêche et aquaculture'] }, dim='Categories1', inplace=False)


  ## Distribute agriculture energy demand to canton based on agriculture employees by canton
  # normalise employees dm
  rename_cantons(dm_employees_agriculture)
  dm_employees_agriculture.drop('Country', 'Suisse')
  linear_fitting(dm_employees_agriculture, years_ots=years_ots+years_fts, min_t0=0)
  dm_employees_agriculture.normalise(dim='Country', inplace=True)

  # energy_canton = energy_CH * employees (%)
  linear_fitting(dm_energy_agriculture, years_ots=years_ots+years_fts)

  dm_energy_agriculture.array = np.maximum(0, dm_energy_agriculture.array)
  arr_energy_canton = (dm_energy_agriculture[:, :, 'bld_energy-by-sector', 'agriculture', :]
                       * dm_employees_agriculture[:, :, 'ind_employees', 'agriculture', np.newaxis] )
  dm_energy_agriculture.add(arr_energy_canton[:, :, np.newaxis, np.newaxis], dim='Country', col_label=dm_employees_agriculture.col_labels['Country'])


  check = False
  if check:
    dm_CH = dm_energy_agriculture.filter({'Country': ['Switzerland']})
    dm_tmp = dm_energy_agriculture.copy()
    dm_tmp.drop('Country', 'Switzerland')
    dm_tmp.groupby({'All': '.*'}, dim='Country', inplace=True, regex=True)
    dm_tmp.append(dm_CH, dim='Country')

  dm_energy_agriculture.rename_col('bld_energy-by-sector', 'agr_energy-consumption', dim='Variables')
  dm_energy_agriculture.group_all('Categories1', inplace=True)
  DM = {'power': dm_energy_agriculture}

  # Fill missing years
  this_dir = os.path.dirname(os.path.abspath(__file__))
  file_agriculture = os.path.join(this_dir, '../../../data/interface/agriculture_to_energy.pickle')
  with open(file_agriculture, 'wb') as handle:
      pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)

  #my_pickle_dump(dm_fuels_eud_cantons, file_industry)
  sort_pickle(file_agriculture)
  return

if __name__ == "__main__":
  run()
