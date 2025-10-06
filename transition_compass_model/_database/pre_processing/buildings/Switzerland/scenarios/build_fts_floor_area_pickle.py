import numpy as np
import os
from  model.common.auxiliary_functions import my_pickle_dump, sort_pickle, create_years_list, load_pop, filter_country_and_load_data_from_pickles
import pickle

from src.api.routes import country_list


def run(DM_buildings, years_ots, years_fts):
  var_floor = 'lfs_floor-intensity_space-cap'
  var_pop = 'lfs_population_total'
  country_list = DM_buildings['ots']['floor-intensity'].col_labels['Country']
  # Find min pop forecast
  last_year = years_fts[-1]
  dm_pop_min = load_pop(country_list, years_list=years_ots+years_fts, lev=1)
  for lev in [2,3,4]:
    dm_pop = load_pop(country_list, years_list=years_ots+years_fts, lev=lev)
    if  (dm_pop[:, last_year, var_pop] < dm_pop_min[:, last_year, var_pop]).any():
      dm_pop_min = dm_pop

  dm_floor_ots = DM_buildings['ots']['floor-intensity'].copy()
  current_floor_area = dm_pop[:, years_ots[-1], var_pop] * dm_floor_ots[:, years_ots[-1], var_floor]

  # Area_min = Area_2023
  # Area_min = pop_min x area-cap_min
  # -> area_cap_min = Area_2023/ pop_min_2050
  # decrease_max = area_cap_min/area_cap_current
  area_cap_min = current_floor_area/dm_pop_min[:, last_year, var_pop]
  area_cap_current = dm_floor_ots[:, years_ots[-1], var_floor]
  decrease_max = area_cap_min/area_cap_current
  decrease_max = 1 - (1-decrease_max)*2
  #
  lev_setting = {2: 1, 3: 1-(1-decrease_max)*1/2, 4: decrease_max}
  for lev in [2,3,4]:
    decrease_lev =lev_setting[lev]
    dm_floor_area_ots = DM_buildings['ots']['floor-intensity'].copy()
    dm_floor_area_ots.add(np.nan, dim='Years', col_label=years_fts, dummy=True)
    dm_floor_area_ots[:, years_fts[-1], var_floor] = decrease_lev*dm_floor_ots[:, years_ots[-1], var_floor]
    dm_floor_area_ots.fill_nans('Years')
    DM_buildings['fts']['floor-intensity'][lev] = dm_floor_area_ots.filter({'Years':years_fts})

  this_dir = os.path.dirname(os.path.abspath(__file__))
  file = os.path.join(this_dir, '../../../../data/datamatrix/buildings.pickle')

  my_pickle_dump(DM_buildings, file)
  sort_pickle(file)

  return DM_buildings


if __name__ == "__main__":
  country_list = ['Switzerland', 'Vaud']

  DM_buildings = filter_country_and_load_data_from_pickles(country_list, 'buildings')
  DM_buildings = DM_buildings['buildings']

  years_ots = create_years_list(1990, 2023, 1)
  years_fts = create_years_list(2025, 2050, 5)

  run(DM_buildings, years_ots, years_fts)
