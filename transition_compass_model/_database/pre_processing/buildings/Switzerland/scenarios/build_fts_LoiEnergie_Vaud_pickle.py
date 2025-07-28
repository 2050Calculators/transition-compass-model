import numpy as np
import pickle

from model.common.auxiliary_functions import my_pickle_dump, create_years_list
from _database.pre_processing.api_routines_CH import get_data_api_CH
from _database.pre_processing.buildings.Switzerland.processors.floor_area_pipeline_CH import load_pop

import os

def extract_stock_floor_area(table_id, file):
  try:
    with open(file, 'rb') as handle:
      dm_floor_area = pickle.load(handle)
  except OSError:
    structure, title = get_data_api_CH(table_id, mode='example', language='fr')
    # Extract buildings floor area
    filter = {'Année': structure['Année'],
              'Canton (-) / District (>>) / Commune (......)': ['Suisse',
                                                                '- Vaud'],
              'Catégorie de bâtiment': structure['Catégorie de bâtiment'],
              'Surface du logement': structure['Surface du logement'],
              'Époque de construction': structure['Époque de construction']}
    mapping_dim = {'Country': 'Canton (-) / District (>>) / Commune (......)',
                   'Years': 'Année',
                   'Variables': 'Surface du logement',
                   'Categories1': 'Catégorie de bâtiment',
                   'Categories2': 'Époque de construction'}
    unit_all = ['number'] * len(structure['Surface du logement'])
    # Get api data
    dm_floor_area = get_data_api_CH(table_id, mode='extract', filter=filter,
                                    mapping_dims=mapping_dim,
                                    units=unit_all,
                                    language='fr')
    dm_floor_area.rename_col(['Suisse', '- Vaud'], ['Switzerland', 'Vaud'],
                             dim='Country')

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, file)
    with open(f, 'wb') as handle:
      pickle.dump(dm_floor_area, handle, protocol=pickle.HIGHEST_PROTOCOL)

  dm_floor_area.groupby({'single-family-households': ['Maisons individuelles'],
                         'multi-family-households': [
                           'Maisons à plusieurs logements',
                           "Bâtiments d'habitation avec usage annexe",
                           "Bâtiments partiellement à usage d'habitation"]},
                        dim='Categories1', inplace=True)

  # There is something weird happening where the number of buildings with less than 30m2 built before
  # 1919 increases over time. Maybe they are re-arranging the internal space?
  # Save number of bld (to compute avg size)
  dm_num_bld = dm_floor_area.groupby({'bld_stock-number-bld': '.*'},
                                     dim='Variables',
                                     regex=True, inplace=False)

  ## Compute total floor space
  # Drop split by size
  dm_floor_area.rename_col_regex(' m2', '', 'Variables')
  # The average size for less than 30 is a guess, as is the average size for 150+,
  # we will use the data from bfs to calibrate
  avg_size = {'<30': 25, '30-49': 39.5, '50-69': 59.5, '70-99': 84.5,
              '100-149': 124.5, '150+': 175}
  idx = dm_floor_area.idx
  for size in dm_floor_area.col_labels['Variables']:
    dm_floor_area.array[:, :, idx[size], :, :] = avg_size[
                                                   size] * dm_floor_area.array[
                                                           :, :, idx[size], :,
                                                           :]
  dm_floor_area.groupby({'bld_floor-area_stock': '.*'}, dim='Variables',
                        regex=True, inplace=True)
  dm_floor_area.change_unit('bld_floor-area_stock', 1, 'number', 'm2')

  return dm_floor_area, dm_num_bld


def compute_renovation_loi_energie(dm_stock_area, dm_num_bld, dm_stock_cat, env_cat_mfh, env_cat_sfh, DM_buildings):
    dm_num_bld.append(dm_stock_area, dim='Variables')
    dm_num_bld_sfh = dm_num_bld.filter({'Categories1': ['single-family-households']})
    dm_num_bld_sfh.groupby(env_cat_sfh, dim='Categories2', inplace=True)
    dm_num_bld_mfh = dm_num_bld.filter({'Categories1': ['multi-family-households']})
    dm_num_bld_mfh.groupby(env_cat_mfh, dim='Categories2', inplace=True)
    dm_bld = dm_num_bld_sfh
    dm_bld.append(dm_num_bld_mfh, dim='Categories1')
    dm_bld_adj = dm_stock_cat.filter({'Variables': ['bld_floor-area_stock']})
    dm_bld_adj.rename_col('bld_floor-area_stock', 'bld_floor-area_stock_adj', dim='Variables')
    dm_bld.append(dm_bld_adj.filter({'Years': dm_bld.col_labels['Years']}), dim='Variables')
    dm_bld.operation('bld_floor-area_stock_adj', '/', 'bld_floor-area_stock', out_col='ratio_area', unit='%')
    dm_bld.operation('bld_stock-number-bld', '*', 'ratio_area', out_col='bld_stock-number-bld_adj', unit='number')
    idx = dm_bld.idx
    ren_goal_2035 = 90000/np.sum(dm_bld.array[idx['Vaud'], -1, idx['bld_stock-number-bld_adj'], idx['multi-family-households'], :])
    dm_rr_fts_2 = DM_buildings['fts']['building-renovation-rate']['bld_renovation-rate'][2].copy()
    idx = dm_rr_fts_2.idx
    yrs_fts = [yr for yr in dm_rr_fts_2.col_labels['Years'] if yr <= 2035]
    idx_fts = [idx[yr] for yr in yrs_fts]
    dm_rr_fts_2.array[idx['Vaud'], idx_fts, idx['bld_renovation-rate'], idx['multi-family-households']] = \
        ren_goal_2035/(yrs_fts[-1] - yrs_fts[0]+1)
    return dm_rr_fts_2



def run(DM_buildings, dm_pop, global_var, country_list, lev=4):

  construction_period_envelope_cat_sfh = global_var['envelope construction sfh']
  construction_period_envelope_cat_mfh = global_var['envelope construction mfh']

  # SECTION: Loi Energie - Renovation fts
  # LEVEL 2 Vaud: Loi Energie + Plan Climat
  # According to the Loi Energie, buildings in categories F,G > 750 m2 will have to be renovated before 2035,
  # and the other F,G before 2040. They estimate this corresponds to 90'000 multi-family-households being renovated before 2035.
  table_id = 'px-x-0902020200_103'
  this_dir = os.path.dirname(os.path.abspath(__file__))
  file = os.path.join(this_dir, '../data/bld_floor-area_stock.pickle')
  dm_stock_area, dm_num_bld = extract_stock_floor_area(table_id, file)
  env_cat_mfh = construction_period_envelope_cat_mfh
  env_cat_sfh = construction_period_envelope_cat_sfh

  # Recompute stock_cat from DM_buildings
  dm_floor_cap = DM_buildings['ots']['floor-intensity'].filter({'Variables': ['lfs_floor-intensity_space-cap'],
                                                                'Country': country_list})
  dm_bld_mix = DM_buildings['ots']['building-renovation-rate']['bld_building-mix'].filter({'Country': country_list}).copy()
  arr_stock = (dm_floor_cap[:, :, :, np.newaxis, np.newaxis]
               * dm_pop[:, :, :, np.newaxis, np.newaxis]
               * dm_bld_mix[:, :, :, :, :])
  dm_bld_mix.add(arr_stock, dim='Variables', col_label = 'bld_floor-area_stock', unit='m2')
  dm_stock_cat = dm_bld_mix.filter({'Variables': ['bld_floor-area_stock']})

  # Compute renovation rate loi energie
  dm_rr_fts_2 = compute_renovation_loi_energie(dm_stock_area, dm_num_bld, dm_stock_cat, env_cat_mfh, env_cat_sfh, DM_buildings)
  DM_buildings['fts']['building-renovation-rate']['bld_renovation-rate'][lev] = dm_rr_fts_2

  # SECTION: Loi energy - Heating tech
  # Plus de gaz, mazout, charbon dans les prochain 15-20 ans. Pas de gaz, mazout, charbon dans les nouvelles constructions
  dm_heating_cat_fts_2 = DM_buildings['fts']['heating-technology-fuel']['bld_heating-technology'][1].copy()
  idx = dm_heating_cat_fts_2.idx
  idx_fossil = [idx['coal'], idx['heating-oil'], idx['gas'], idx['electricity']]
  dm_heating_cat_fts_2.array[idx['Vaud'], :, idx['bld_heating-mix'], :, idx['B'], idx_fossil] = 0
  dm_heating_cat_fts_2.array[idx['Vaud'], 1:idx[2045], idx['bld_heating-mix'], :, :, idx_fossil] = np.nan
  dm_heating_cat_fts_2.array[idx['Vaud'], idx[2045]:, idx['bld_heating-mix'], :, :, idx_fossil] = 0
  dm_heating_cat_fts_2.fill_nans('Years')
  dm_heating_cat_fts_2.normalise('Categories3')
  DM_buildings['fts']['heating-technology-fuel']['bld_heating-technology'][lev] = dm_heating_cat_fts_2

  this_dir = os.path.dirname(os.path.abspath(__file__))
  # !FIXME: use the actual values and not the calibration factor
  file = os.path.join(this_dir, '../../../../data/datamatrix/buildings.pickle')

  my_pickle_dump(DM_buildings, file)

  return DM_buildings


if __name__ == "__main__":
  this_dir = os.path.dirname(os.path.abspath(__file__))
  # !FIXME: use the actual values and not the calibration factor
  file = os.path.join(this_dir, '../../../../data/datamatrix/buildings.pickle')
  with open(file, 'rb') as handle:
    DM_buildings = pickle.load(handle)

  construction_period_envelope_cat_sfh = {
    'F': ['Avant 1919', '1919-1945', '1946-1960', '1961-1970'],
    'E': ['1971-1980'],
    'D': ['1981-1990', '1991-2000'],
    'C': ['2001-2005', '2006-2010'],
    'B': ['2011-2015', '2016-2020', '2021-2023']}
  construction_period_envelope_cat_mfh = {
    'F': ['Avant 1919', '1919-1945', '1946-1960', '1961-1970', '1971-1980'],
    'E': ['1981-1990'],
    'D': ['1991-2000'],
    'C': ['2001-2005', '2006-2010'],
    'B': ['2011-2015', '2016-2020', '2021-2023']}

  global_var = {
    'envelope construction sfh': construction_period_envelope_cat_sfh,
    'envelope construction mfh': construction_period_envelope_cat_mfh}

  this_dir = os.path.dirname(os.path.abspath(__file__))
  filepath = os.path.join(this_dir, "../../../../data/datamatrix/lifestyles.pickle")

  years_ots = create_years_list(1990, 2023, 1)
  country_list = ['Switzerland', 'Vaud']

  dm_pop = load_pop(filepath, country_list, years_ots)

  DM_buildings = run(DM_buildings, dm_pop, global_var, country_list, lev=4)
