import numpy as np
import pickle

from model.common.auxiliary_functions import moving_average, linear_fitting, create_years_list
from _database.pre_processing.api_routines_CH import get_data_api_CH
from model.common.data_matrix_class import DataMatrix

import os


def compute_avg_floor_area(dm_floor_area, years_ots):

    dm_avg_floor_area = dm_floor_area.filter({'Variables': ['bld_avg-floor-area-new']})
    years_to_keep = ['1991-2000', '2001-2005', '2006-2010', '2011-2015', '2016-2020', '2021-2023']
    #years_to_keep = dm_avg_floor_area.col_labels['Categories2'].copy()
    #years_to_keep.remove('Avant 1919')
    dm_avg_floor_area.filter({'Categories2': years_to_keep}, inplace=True)
    # Compute the avg floor area based on construction period
    arr_avg_area = np.nanmean(dm_avg_floor_area.array, axis=1, keepdims=True)
    #years_all = create_years_list(1919, 2023, 1)
    years_missing = list(set(years_ots) - set(dm_avg_floor_area.col_labels['Years']))
    dm_avg_floor_area.add(np.nan, dummy=True, dim='Years', col_label=years_missing)
    dm_avg_floor_area.sort('Years')
    dm = dm_avg_floor_area.copy()
    dm.group_all('Categories2')
    dm.array[...] = np.nan
    idx_out = dm.idx
    idx_in = dm_avg_floor_area.idx
    for interval in dm_avg_floor_area.col_labels['Categories2']:
        y_start = int(interval.split('-')[0])
        y_end = int(interval.split('-')[1])
        y_mean = int((y_start + y_end)/2)
        # Assign categories 2 value to years in the middle of the interval
        dm.array[:, idx_out[y_mean], :, :] = arr_avg_area[:, 0, :, :, idx_in[interval]]
    # Linear interpolation
    dm.fill_nans(dim_to_interp='Years')
    # Moving average to smooth
    window_size = 3  # Change window size to control the smoothing effect
    data_smooth = moving_average(dm.array, window_size, axis=dm.dim_labels.index('Years'))
    dm.array[:, 1:-1, ...] = data_smooth

    return dm


def extract_stock_floor_area(table_id, file):

    try:
        with open(file, 'rb') as handle:
            dm_floor_area = pickle.load(handle)
    except OSError:
        structure, title = get_data_api_CH(table_id, mode='example', language='fr')
        # Extract buildings floor area
        filter = {'Année': structure['Année'],
                  'Canton (-) / District (>>) / Commune (......)': ['Suisse', '- Vaud'],
                  'Catégorie de bâtiment': structure['Catégorie de bâtiment'],
                  'Surface du logement': structure['Surface du logement'],
                  'Époque de construction': structure['Époque de construction']}
        mapping_dim = {'Country': 'Canton (-) / District (>>) / Commune (......)', 'Years': 'Année',
                       'Variables': 'Surface du logement', 'Categories1': 'Catégorie de bâtiment',
                       'Categories2': 'Époque de construction'}
        unit_all = ['number'] * len(structure['Surface du logement'])
        # Get api data
        dm_floor_area = get_data_api_CH(table_id, mode='extract', filter=filter, mapping_dims=mapping_dim,
                                        units=unit_all,
                                        language='fr')
        dm_floor_area.rename_col(['Suisse', '- Vaud'], ['Switzerland', 'Vaud'], dim='Country')

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, 'wb') as handle:
            pickle.dump(dm_floor_area, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dm_floor_area.groupby({'single-family-households': ['Maisons individuelles'],
                           'multi-family-households': ['Maisons à plusieurs logements',
                                                       "Bâtiments d'habitation avec usage annexe",
                                                       "Bâtiments partiellement à usage d'habitation"]},
                          dim='Categories1', inplace=True)

    # There is something weird happening where the number of buildings with less than 30m2 built before
    # 1919 increases over time. Maybe they are re-arranging the internal space?
    # Save number of bld (to compute avg size)
    dm_num_bld = dm_floor_area.groupby({'bld_stock-number-bld': '.*'}, dim='Variables',
                                       regex=True, inplace=False)

    ## Compute total floor space
    # Drop split by size
    dm_floor_area.rename_col_regex(' m2', '', 'Variables')
    # The average size for less than 30 is a guess, as is the average size for 150+,
    # we will use the data from bfs to calibrate
    avg_size = {'<30': 25, '30-49': 39.5, '50-69': 59.5, '70-99': 84.5, '100-149': 124.5, '150+': 175}
    idx = dm_floor_area.idx
    for size in dm_floor_area.col_labels['Variables']:
        dm_floor_area.array[:, :, idx[size], :, :] = avg_size[size] * dm_floor_area.array[:, :, idx[size], :, :]
    dm_floor_area.groupby({'bld_floor-area_stock': '.*'}, dim='Variables', regex=True, inplace=True)
    dm_floor_area.change_unit('bld_floor-area_stock', 1, 'number', 'm2')


    return dm_floor_area, dm_num_bld


def compute_floor_area_stock_v2(table_id, file, dm_pop, cat_map_sfh, cat_map_mfh, years_ots):
    # Computes:
    #   floor-area stock in m2 by sfh and mfh,
    #   2023 split also by envelope category
    dm_stock_area, dm_num_bld = extract_stock_floor_area(table_id, file)

    # Remove 2010 data because they are odd
    dm_stock_area.drop(dim='Years', col_label=2010)
    dm_num_bld.drop(dim='Years', col_label=2010)

    # Compute average floor area
    dm = dm_stock_area.copy()
    dm.append(dm_num_bld, dim='Variables')
    dm.operation('bld_floor-area_stock', '/', 'bld_stock-number-bld', out_col='bld_avg-floor-area-new', unit='m2/bld')
    # Extracting the average new built floor area for sfh and mfh
    # (Basically you want the category construction period to become the year category)
    dm_avg_floor_area = compute_avg_floor_area(dm, years_ots)

    # Turn construction period to envelope category
    dm_sfh = dm_stock_area.filter({'Categories1': ['single-family-households']})
    dm_mfh = dm_stock_area.filter({'Categories1': ['multi-family-households']})
    dm_sfh.groupby(cat_map_sfh, dim='Categories2', inplace=True)
    dm_mfh.groupby(cat_map_mfh, dim='Categories2', inplace=True)
    dm = dm_sfh
    dm.append(dm_mfh, dim='Categories1')

    # Remove split by envelope category (see Building module Overleaf)
    dm_stock_tot = dm.group_all('Categories2', inplace=False)
    # Keep only s_c,t(2023), set the rest to np.nan
    dm.array[:, 0:-1, ...] = np.nan
    dm_stock_cat = dm.filter({'Variables': ['bld_floor-area_stock']})

    # Extrapolate to missing ots years using population and per capita values
    years_missing = list(set(years_ots) - set(dm_stock_tot.col_labels['Years']))
    dm_stock_tot.add(np.nan, dim='Years', col_label=years_missing, dummy=True)
    dm_stock_tot.sort('Years')
    idx = dm_stock_tot.idx
    idx_p = dm_pop.idx
    arr = dm_stock_tot.array[:, :, idx['bld_floor-area_stock']]/dm_pop.array[:, :, idx_p['lfs_population_total'], np.newaxis]
    dm_stock_tot.add(arr, dim='Variables', col_label='bld_floor-area_stock-cap', unit='m2/cap')
    linear_fitting(dm_stock_tot, years_missing)
    dm_stock_tot.array[:, :, idx['bld_floor-area_stock'], :] = dm_stock_tot.array[:, :, idx['bld_floor-area_stock-cap'], :] *\
                                                               dm_pop.array[:, :, idx_p['lfs_population_total'], np.newaxis]
    dm_stock_tot.filter({'Variables': ['bld_floor-area_stock']}, inplace=True)

    dm_stock_cat.add(np.nan, dim='Years', col_label=years_missing, dummy=True)
    dm_stock_cat.sort('Years')


    return dm_stock_tot, dm_stock_cat, dm_avg_floor_area


def extract_bld_new_buildings_1(table_id, file):
    try:
        with open(file, 'rb') as handle:
            dm_new_area = pickle.load(handle)
    except OSError:
        structure, title = get_data_api_CH(table_id, mode='example', language='fr')
        # Extract buildings floor area
        filter = {'Année': structure['Année'],
                  'Grande région (<<) / Canton (-) / Commune (......)': ['Suisse', '- Canton de Vaud'],
                  'Type de bâtiment': structure['Type de bâtiment']}
        mapping_dim = {'Country': 'Grande région (<<) / Canton (-) / Commune (......)',
                       'Years': 'Année',
                       'Variables': 'Type de bâtiment'}
        unit_all = ['number'] * len(structure['Type de bâtiment'])
        # Get api data
        dm_new_area = get_data_api_CH(table_id, mode='extract', filter=filter, mapping_dims=mapping_dim,
                                      units=unit_all, language='fr')
        dm_new_area.rename_col(['Suisse', '- Canton de Vaud'], ['Switzerland', 'Vaud'], dim='Country')
        dm_new_area.groupby({'bld_new-buildings_single-family-households':
                                 ['Maisons individuelles'],
                             'bld_new-buildings_multi-family-households':
                                 ['Maisons à plusieurs logements', "Bâtiments d'habitation avec usage annexe",
                                  "Bâtiments partiellement à usage d'habitation"]}, dim='Variables', inplace=True)

        dm_new_area.filter_w_regex({'Variables': '.*households'}, inplace=True)
        dm_new_area.sort('Years')
        dm_new_area.deepen()

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, 'wb') as handle:
            pickle.dump(dm_new_area, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm_new_area


def extract_bld_new_buildings_2(table_id, file):
    try:
        with open(file, 'rb') as handle:
            dm_new_area = pickle.load(handle)
    except OSError:
        structure, title = get_data_api_CH(table_id, mode='example', language='fr')
        # Extract buildings floor area
        filter = {'Année': structure['Année'],
                  'Canton (-) / Commune (......)': ['Suisse', '- Canton de Vaud'],
                  'Type de bâtiment': structure['Type de bâtiment']}
        mapping_dim = {'Country': 'Canton (-) / Commune (......)',
                       'Years': 'Année',
                       'Variables': 'Type de bâtiment'}
        unit_all = ['number'] * len(structure['Type de bâtiment'])
        # Get api data
        dm_new_area = get_data_api_CH(table_id, mode='extract', filter=filter, mapping_dims=mapping_dim,
                                      units=unit_all, language='fr')
        dm_new_area.rename_col(['Suisse', '- Canton de Vaud'], ['Switzerland', 'Vaud'], dim='Country')
        dm_new_area.groupby({'bld_new-buildings_single-family-households':
                                 ['Maisons individuelles à un logement, isolées',
                                  'Maisons individuelles à un logement, mitoyennes'],
                             'bld_new-buildings_multi-family-households':
                                 ["Maisons à plusieurs logements, à usage exclusif d'habitation",
                                  "Bâtiments à usage mixte, principalement à usage d'habitation",
                                  "Bâtiments partiellement à usage d'habitation"]},
                            dim='Variables', inplace=True)

        dm_new_area.filter_w_regex({'Variables': '.*households'}, inplace=True)
        dm_new_area.sort('Years')
        dm_new_area.deepen()

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, 'wb') as handle:
            pickle.dump(dm_new_area, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm_new_area



def compute_bld_floor_area_new(dm_bld_new_buildings_1, dm_bld_new_buildings_2, dm_bld_area_stock, dm_pop, years_ots):
    dm_bld_area_new = dm_bld_new_buildings_2.copy()
    dm_bld_area_new.append(dm_bld_new_buildings_1, dim='Years')
    dm_bld_area_new.sort('Years')

    # Extrapolate new buildings to missing years using per capita approach
    years_missing = list(set(years_ots) - set(dm_bld_area_new.col_labels['Years']))
    dm_bld_area_new.add(np.nan, col_label=years_missing, dim='Years', dummy=True)
    arr_bld_cap = dm_bld_area_new.array/dm_pop.array[..., np.newaxis]
    dm_bld_area_new.add(arr_bld_cap, dim='Variables', col_label='bld_new-buildings-cap', unit='bld/cap')
    linear_fitting(dm_bld_area_new, years_ots, based_on=create_years_list(1995, 2002, 1))
    dm_bld_area_new.array[:, -1, ...] = np.nan
    linear_fitting(dm_bld_area_new, years_ots, based_on=create_years_list(2015, 2022, 1))
    idx = dm_bld_area_new.idx
    dm_bld_area_new.array[:, :, idx['bld_new-buildings'], :] = dm_bld_area_new.array[:, :, idx['bld_new-buildings-cap'], :] \
                                                               * dm_pop.array[:, :, 0, np.newaxis]
    # floor-area_new = new-bld x avg-floor-areabdl
    dm_bld_area_new.append(dm_bld_area_stock, dim='Variables')
    dm_bld_area_new.operation('bld_new-buildings', '*', 'bld_avg-floor-area-new', out_col='bld_floor-area_new', unit='m2')

    dm_bld_area_new.filter({'Variables': ['bld_floor-area_new']}, inplace=True)

    return dm_bld_area_new


def compute_waste(dm_stock_tot, dm_new_tot, years_ots):

    dm = dm_stock_tot.copy()
    dm.append(dm_new_tot, dim='Variables')

    # w(t) = s(t-1) - s(t) + n(t)
    dm.lag_variable('bld_floor-area_stock', shift=1, subfix='_tm1')
    dm.operation('bld_floor-area_stock_tm1', '-', 'bld_floor-area_stock', out_col='bld_floor-area_delta', unit='m2')
    dm.operation('bld_floor-area_delta', '+', 'bld_floor-area_new', out_col='bld_floor-area_waste', unit='m2')

    # FIX WASTE
    # Set minimum demolition-rate at 0.2%
    # dem-rate(t-1) = max( w(t)/s(t-1), 0.2% )
    # w(t) = dem-rate(t-1)*s(t-1)
    idx = dm.idx
    min_waste = 0.002 * dm.array[:, :, idx['bld_floor-area_stock_tm1'], ...]
    dm.array[:, :, idx['bld_floor-area_waste'], ...] = \
        np.maximum(min_waste, dm.array[:, :, idx['bld_floor-area_waste'], ...])
    # Fix 1990 value
    dm.array[:, 0, idx['bld_floor-area_waste'], ...] = np.nan
    dm.fill_nans('Years')

    # FIX NEW
    # n(t) = s(t) - s(t-1) + w(t)
    dm.drop(dim='Variables', col_label='bld_floor-area_new')
    dm.operation('bld_floor-area_waste', '-', 'bld_floor-area_delta', out_col='bld_floor-area_new', unit='m2')

    dm_waste = dm.filter({'Variables': ['bld_floor-area_waste']})
    dm_new_tot = dm.filter({'Variables': ['bld_floor-area_new']})

    # Replace 1990 with nan and fill
    dm_new_tot.array[:, 0, :, :] = np.nan
    linear_fitting(dm_new_tot, years_ots, based_on=list(range(1991,1995)))
    return dm_waste, dm_new_tot


def compute_floor_area_waste_cat(dm_waste_tot):
    # Assuming the lifetime of a building is 50 years and buildings of categories D are built starting in 1981
    # the first waste of building in category D will by in 2031, before than is all category F.
    dm = dm_waste_tot.copy()
    dm.rename_col('single-family-households', 'single-family-households_F', dim='Categories1')
    dm.rename_col('multi-family-households', 'multi-family-households_F', dim='Categories1')
    dm.deepen()
    dm.add(0, dummy=True, dim='Categories2', col_label=['B', 'C', 'D', 'E'])
    dm.sort("Categories2")

    return dm



def compute_floor_area_new_cat(dm_new_tot, cat_map):

    env_cat = ['B', 'C', 'D', 'E', 'F']
    arr = np.zeros(np.shape(dm_new_tot.array) + (len(env_cat),))
    dm_new_cat = DataMatrix.based_on(arr, dm_new_tot, change={'Categories2': env_cat}, units=dm_new_tot.units)

    idx = dm_new_cat.idx
    for cat, period in cat_map.items():
        idx_period = [idx[yr] for yr in range(period[0], period[-1]+1)]
        dm_new_cat.array[:, idx_period, idx['bld_floor-area_new'], :, idx[cat]] = 1

    idx_t = dm_new_tot.idx
    dm_new_cat.array[:, :, idx['bld_floor-area_new'], :, :]  \
        = dm_new_cat.array[:, :, idx['bld_floor-area_new'], :, :] \
          * dm_new_tot.array[:, :, idx_t['bld_floor-area_new'], :, np.newaxis]

    return dm_new_cat


