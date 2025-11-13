import numpy as np
import os
from  model.common.auxiliary_functions import my_pickle_dump, sort_pickle, create_years_list
import pickle


def run(DM_buildings, years_ots, years_fts):
  # Lev 2 - heating temperature setting decrease of 0.5 not below 21
  # Lev 3 - heating temperature setting decrease of 1, not below 20
  # Lev 4 - heating temperature setting decrease of 1.5, not below 19.5

  lev_setting = {2: [21, -0.5], 3: [20, -1], 4: [19.5, -1.5]}
  for lev in [2,3,4]:
    min_T = lev_setting[lev][0]
    deltaT = lev_setting[lev][1]
    dm_heat_cool_ots = DM_buildings['ots']['heatcool-behaviour'].copy()
    dm_heat_cool_ots.add(np.nan, dim='Years', col_label=years_fts, dummy=True)
    dm_heat_cool_ots[:, 2050, 'bld_Tint-heating', :, :] = np.maximum(min_T, dm_heat_cool_ots[:, years_ots[-1], 'bld_Tint-heating', :, :]+ deltaT)
    dm_heat_cool_ots[:, 2050, 'bld_Tint-heating', :, :] = np.minimum(dm_heat_cool_ots[:, years_ots[-1], 'bld_Tint-heating', :, :], dm_heat_cool_ots[:, 2050, 'bld_Tint-heating', :, :])
    dm_heat_cool_ots.fill_nans('Years')
    dm_heat_cool_fts = dm_heat_cool_ots.filter({'Years': years_fts})
    DM_buildings['fts']['heatcool-behaviour'][lev] = dm_heat_cool_fts.copy()

  this_dir = os.path.dirname(os.path.abspath(__file__))
  file = os.path.join(this_dir, '../../../../data/datamatrix/buildings.pickle')

  my_pickle_dump(DM_buildings, file)
  sort_pickle(file)

  return DM_buildings


if __name__ == "__main__":
  this_dir = os.path.dirname(os.path.abspath(__file__))
  # !FIXME: use the actual values and not the calibration factor
  file = os.path.join(this_dir, '../../../../data/datamatrix/buildings.pickle')
  with open(file, 'rb') as handle:
    DM_buildings = pickle.load(handle)

  years_ots = create_years_list(1990, 2023, 1)
  years_fts = create_years_list(2025, 2050, 5)

  run(DM_buildings, years_ots, years_fts)
