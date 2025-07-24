import numpy as np
import pickle

from _database.pre_processing.api_routines_CH import get_data_api_CH
from model.common.data_matrix_class import DataMatrix


import os

def extract_number_of_buildings(table_id, file):
    try:
        with open(file, 'rb') as handle:
            dm_nb_bld = pickle.load(handle)
    except OSError:
        structure, title = get_data_api_CH(table_id, 'example', language='fr')
        filter = {
            'Canton (-) / District (>>) / Commune (......)': ['Suisse', '- Vaud'],
            'Catégorie de bâtiment': structure['Catégorie de bâtiment'],
            'Époque de construction': structure['Époque de construction'],
            'Année': structure['Année']
        }
        mapping = {'Country': 'Canton (-) / District (>>) / Commune (......)',
                   'Years': 'Année',
                   'Variables': 'Époque de construction',
                   'Categories1': 'Catégorie de bâtiment'}
        unit = ['number'] * len(structure['Époque de construction'])
        dm_nb_bld = get_data_api_CH(table_id, mode='extract', filter=filter, mapping_dims=mapping, language='fr',
                                    units=unit)
        dm_nb_bld.groupby({'bld_nb-bld': '.*'}, regex=True, inplace=True, dim='Variables')
        dm_nb_bld.groupby({'single-family-households': ['Maisons individuelles'],
                           'multi-family-households': ['Maisons à plusieurs logements',
                                                       "Bâtiments d'habitation avec usage annexe",
                                                       "Bâtiments partiellement à usage d'habitation"]},
                          dim='Categories1', inplace=True)
        dm_nb_bld.rename_col(['Suisse', '- Vaud'], ['Switzerland', 'Vaud'], dim='Country')
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, 'wb') as handle:
            pickle.dump(dm_nb_bld, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm_nb_bld


def compute_renovated_buildings(dm_bld, nb_buildings_renovated, VD_share,
                                share_by_bld):
  dm_bld.add(np.nan, dim='Variables', dummy=True,
             col_label='bld_nb-bld-renovated', unit='number')
  idx = dm_bld.idx
  for yr in nb_buildings_renovated.keys():
    for cat in ['single-family-households', 'multi-family-households']:
      dm_bld.array[
        idx['Switzerland'], idx[yr], idx['bld_nb-bld-renovated'], idx[cat]] \
        = nb_buildings_renovated[yr] * share_by_bld[cat]
      dm_bld.array[idx['Vaud'], idx[yr], idx['bld_nb-bld-renovated'], idx[cat]] \
        = nb_buildings_renovated[yr] * VD_share[yr] * share_by_bld[cat]
  return dm_bld


def compute_renovation_rate(dm_renovation, years_ots):

    dm_renovation.operation('bld_nb-bld-renovated', '/', 'bld_nb-bld',
                            out_col='bld_renovation-rate', unit='%')
    years_missing = list(
      set(years_ots) - set(dm_renovation.col_labels['Years']))
    dm_renovation.add(np.nan, dummy=True, col_label=years_missing, dim='Years')
    dm_renovation.sort('Years')
    dm_renovation.fill_nans(dim_to_interp='Years')
    dm_renovation.filter({'Variables': ['bld_renovation-rate']}, inplace=True)

    return dm_renovation



def extract_renovation_redistribuition(ren_map_in, ren_map_out, years_ots):
    dm = DataMatrix(col_labels={'Country': ['Switzerland', 'Vaud'],
                                            'Years': years_ots,
                                            'Variables': ['bld_renovation-redistribution-in',
                                                          'bld_renovation-redistribution-out'],
                                            'Categories1': ['B', 'C', 'D', 'E', 'F']},
                    units={'bld_renovation-redistribution-in': '%', 'bld_renovation-redistribution-out': '%'})
    dm.array = np.nan * np.ones((len(dm.col_labels['Country']),
                                 len(dm.col_labels['Years']),
                                 len(dm.col_labels['Variables']),
                                 len(dm.col_labels['Categories1'])))
    idx = dm.idx
    for year_period, map_values in ren_map_out.items():
        for key, val in map_values.items():
            dm.array[:, idx[year_period[1]], idx['bld_renovation-redistribution-out'], idx[key]] = val
    dm.array[:, idx[1990], ...] = dm.array[:, idx[2000], ...]

    idx = dm.idx
    for year_period, map_values in ren_map_in.items():
        idx_year_period = [idx[yr] for yr in range(year_period[0], year_period[1]+1)]
        for key, val in map_values.items():
            dm.array[:, idx_year_period, idx['bld_renovation-redistribution-in'], idx[key]] = val

    dm.fill_nans(dim_to_interp='Years')
    dm.normalise('Categories1')

    return dm


def compute_floor_area_renovated(dm_stock_tot, dm_renovation, dm_renov_distr):

    # r_ct(t) = Redistr_ct(t) * (ren-rate_t(t) * stock_t(t))

    dm = dm_renovation.copy()
    dm.append(dm_stock_tot, dim='Variables')
    dm.lag_variable('bld_floor-area_stock', shift=1, subfix='_tm1')
    dm.lag_variable('bld_renovation-rate', shift=1, subfix='_tm1')
    dm.operation('bld_renovation-rate_tm1', '*', 'bld_floor-area_stock_tm1', out_col='bld_floor-area_renovated', unit='m2')

    idx = dm.idx
    idx_d = dm_renov_distr.idx
    arr = dm.array[:, :, idx['bld_floor-area_renovated'], :, np.newaxis] \
          * (- dm_renov_distr.array[:, :, idx_d['bld_renovation-redistribution-out'], np.newaxis, :] +
             dm_renov_distr.array[:, :, idx_d['bld_renovation-redistribution-in'], np.newaxis, :] )
    dm_renovated_cat = DataMatrix.based_on(arr[:, :, np.newaxis, ...], dm, change={'Variables': ['bld_floor-area_renovated'],
                                                            'Categories2': dm_renov_distr.col_labels['Categories1']},
                                           units={'bld_floor-area_renovated': 'm2'})

    return dm_renovated_cat


def fix_negative_stock(dm_all_cat, dm_stock_tot):
    dm_tmp = dm_all_cat.filter({'Variables': ['bld_floor-area_stock']})
    dm_tmp.drop(col_label='F', dim='Categories2')
    shift = -np.min(dm_tmp.array[:, :, 0, :, :], axis=1)
    shift = np.maximum(0, shift)
    # Replace 0 with nan
    # Step 1: Replace zeros with NaN
    shift[shift == 0] = np.nan
    # Step 2: Compute the minimum across axis 2, ignoring NaNs
    min_values = np.nanmin(shift, axis=2, keepdims=True)
    # Step 3: Replace NaNs with the corresponding minimum values
    shift = np.where(np.isnan(shift), min_values, shift)
    dm_tmp.array[:, :, 0, :, :] = dm_tmp.array[:, :, 0, :, :] + shift[:, np.newaxis, :, :]
    dm_stock_wo_F = dm_tmp.group_all(dim='Categories2', inplace=False)
    dm_stock_wo_F.rename_col('bld_floor-area_stock', 'bld_floor-area_stock_woF', dim='Variables')
    dm_stock_tot.append(dm_stock_wo_F, dim='Variables')
    dm_stock_tot.operation('bld_floor-area_stock', '-', 'bld_floor-area_stock_woF', out_col='bld_floor-area_stock_F', unit='m2')
    dm_stock_F = dm_stock_tot.filter({'Variables': ['bld_floor-area_stock_F']})
    dm_stock_F.deepen(based_on='Variables')

    dm_tmp.append(dm_stock_F, dim='Categories2')
    dm_all_cat.drop(col_label='bld_floor-area_stock', dim='Variables')

    dm_all_cat.append(dm_tmp, dim='Variables')
    dm_stock_tot.filter({'Variables': ['bld_floor-area_stock']}, inplace=True)
    return dm_all_cat


def compute_stock_area_by_cat(dm_stock_cat, dm_new_cat, dm_renov_cat, dm_waste_cat, dm_stock_tot):
    # s_{c,t}(t-1) &= s_{c,t}(t) - n_{c,t}(t) - r_{c,t}(t) + w_{c,t}(t)
    dm_all_cat = dm_stock_cat
    dm_all_cat.append(dm_new_cat, dim='Variables')
    dm_all_cat.append(dm_renov_cat, dim='Variables')
    dm_all_cat.append(dm_waste_cat, dim='Variables')
    idx = dm_all_cat.idx

    for t in reversed(dm_all_cat.col_labels['Years'][1:]):
        dm_all_cat.array[:, idx[t-1], idx['bld_floor-area_stock'], ...] \
            = dm_all_cat.array[:, idx[t], idx['bld_floor-area_stock'], ...] \
              - dm_all_cat.array[:, idx[t], idx['bld_floor-area_new'], ...] \
              - dm_all_cat.array[:, idx[t], idx['bld_floor-area_renovated'], ...] \
              + dm_all_cat.array[:, idx[t], idx['bld_floor-area_waste'], ...]


    # FIX Negative Stock
    # Adjusts the stock such that new, waste, renovated are unchanged (stock slope is fixed)
    # This is done by shifting up categories B, C, D, E and shifting down category F
    dm_all_cat = fix_negative_stock(dm_all_cat, dm_stock_tot)
    # IF the fix is done correctly the following run should leave things unchanged
    idx = dm_all_cat.idx
    for t in reversed(dm_all_cat.col_labels['Years'][1:]):
        dm_all_cat.array[:, idx[t-1], idx['bld_floor-area_stock'], ...] \
            = dm_all_cat.array[:, idx[t], idx['bld_floor-area_stock'], ...] \
              - dm_all_cat.array[:, idx[t], idx['bld_floor-area_new'], ...] \
              - dm_all_cat.array[:, idx[t], idx['bld_floor-area_renovated'], ...] \
              + dm_all_cat.array[:, idx[t], idx['bld_floor-area_waste'], ...]

    return dm_all_cat
