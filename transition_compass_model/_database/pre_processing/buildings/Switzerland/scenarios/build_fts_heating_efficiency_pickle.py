import numpy as np
import os
from  model.common.auxiliary_functions import my_pickle_dump, sort_pickle, create_years_list, filter_country_and_load_data_from_pickles
import pickle

def run(DM_buildings, years_fts):
  # From EP2050+_TechnsicherBericht_DatenAbbildungen_Kap 1-7_2022-04-12
  # Abbildung 16: Wärmenutzungsgrad von Wärmepumpen für Raumwärme
  # I know the expected 2050 values for heatpumps split by type of heatpump
  # (air/water or soil/water) and by type of building (new or old)
  # I assign the air/water COP values to level 3 (new -> A,B , old-> E)
  #    the other classes 2050 are determined by taking the average
  #    increase for A,B and E from level 1 to level 3 2050 values
  # I assign a 50%/50% mix of air/water and soil/water COP values to level 4 (new -> A,B , old-> E)
  #    the other classes 2050 are determined by taking the average
  #    increase for A,B and E from level 1 to level 4 2050 values

  dict_2050_values = {3: {'B': 5.1, 'C': 4.7, 'D': 4.3, 'E': 3.6, 'F': 3.3},
                      4: {'B': 5.7, 'C': 5.2, 'D': 4.7, 'E': 3.9, 'F': 3.6}}
  for lev in [3,4]:
    dm = DM_buildings['ots']['heating-efficiency'].copy()
    dm_fts = DM_buildings['fts']['heating-efficiency'][1].copy()
    dm.append(dm_fts, dim='Years')
    idx_0 = dm.idx[years_fts[0]]
    dm[:, idx_0:, :, :, 'heat-pump'] = np.nan
    for cat in dm.col_labels['Categories1']:
      dm[:, years_fts[-1], 'bld_heating-efficiency', cat, 'heat-pump'] = dict_2050_values[lev][cat]
    dm.fill_nans('Years')
    DM_buildings['fts']['heating-efficiency'][lev] = dm.filter({'Years': years_fts})

  # Level 2 defined as middle point between 1 and 3
  dm_fts_1 = DM_buildings['fts']['heating-efficiency'][1].copy()
  dm_fts_3 = DM_buildings['fts']['heating-efficiency'][3].copy()
  dm_fts_2 = dm_fts_1
  dm_fts_2[...] = (dm_fts_1[...] + dm_fts_3[...])/2
  DM_buildings['fts']['heating-efficiency'][2] = dm_fts_2

  this_dir = os.path.dirname(os.path.abspath(__file__))
  file = os.path.join(this_dir, '../../../../data/datamatrix/buildings.pickle')

  my_pickle_dump(DM_buildings, file)
  sort_pickle(file)

  return

if __name__ == "__main__":
  country_list = ['Switzerland', 'Vaud']

  DM_buildings = filter_country_and_load_data_from_pickles(country_list, 'buildings')
  DM_buildings = DM_buildings['buildings']

  years_fts = create_years_list(2025, 2050, 5)

  run(DM_buildings, years_fts)
