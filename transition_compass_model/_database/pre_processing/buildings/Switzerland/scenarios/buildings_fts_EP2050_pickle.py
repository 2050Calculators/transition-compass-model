import os
import pickle
import numpy as np
from model.common.auxiliary_functions import my_pickle_dump, sort_pickle


def run(DM_buildings, lev=4):

  # SECTION: Loi energy - Heating tech
  # Plus de gaz, mazout, charbon dans les prochain 15-20 ans. Pas de gaz, mazout, charbon dans les nouvelles constructions
  dm_heating_cat_fts_2 = \
  DM_buildings['fts']['heating-technology-fuel']['bld_heating-technology'][1].copy()

  idx = dm_heating_cat_fts_2.idx
  dm_heating_cat_fts_2.array[idx['Vaud'], 1:idx[2050], idx['bld_heating-mix'], :, :, :] = np.nan

  # Electricity outphasing as BAU
  idx_old_cat = [idx['E'], idx['F']]
  idx_new_cat = [idx['B'], idx['C'], idx['D']]
  dm_heating_cat_fts_2.array[idx['Vaud'], idx[2035]:, :, :, idx_old_cat, idx['electricity']] = 0
  dm_heating_cat_fts_2.array[idx['Vaud'], idx[2040]:, :, :, idx_new_cat, idx['electricity']] = 0
  # Fossil heating outphasing as LoiEnergie (leave bio-gas
  idx_fossil = [idx['coal'], idx['heating-oil']]
  dm_heating_cat_fts_2.array[idx['Vaud'], :, idx['bld_heating-mix'], :, idx['B'], idx_fossil] = 0
  dm_heating_cat_fts_2.array[idx['Vaud'], 1:idx[2045], idx['bld_heating-mix'], :, :, idx_fossil] = np.nan
  dm_heating_cat_fts_2.array[idx['Vaud'], idx[2045]:, idx['bld_heating-mix'], :, :, idx_fossil] = 0
  # Allow some biogas (EP2050)
  dm_heating_cat_fts_2.array[idx['Vaud'], idx[2050], idx['bld_heating-mix'], :, :, idx['gas']] = 0.058
  # Wood (EP2050)
  dm_heating_cat_fts_2.array[idx['Vaud'], idx[2050], idx['bld_heating-mix'], :, :, idx['wood']] = 0.028
  # District-heating (EP2050)
  dm_heating_cat_fts_2.array[idx['Vaud'], idx[2050], idx['bld_heating-mix'], idx['multi-family-households']:, :, idx['district-heating']] = 0.32
  dm_heating_cat_fts_2.array[idx['Vaud'], idx[2050], idx['bld_heating-mix'], idx['single-family-households']:, :, idx['district-heating']] = 0.09
  # Heat-pump (EP2050)
  dm_heating_cat_fts_2.array[idx['Vaud'], idx[2050], idx['bld_heating-mix'], idx['multi-family-households']:, :, idx['heat-pump']] = 0.59
  dm_heating_cat_fts_2.array[idx['Vaud'], idx[2050], idx['bld_heating-mix'], idx['single-family-households']:, :, idx['heat-pump']] = 0.82


  dm_heating_cat_fts_2.fill_nans('Years')
  dm_heating_cat_fts_2.normalise('Categories3')
  DM_buildings['fts']['heating-technology-fuel']['bld_heating-technology'][lev] = dm_heating_cat_fts_2


  this_dir = os.path.dirname(os.path.abspath(__file__))
  # !FIXME: use the actual values and not the calibration factor
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
  run(DM_buildings, lev=3)
