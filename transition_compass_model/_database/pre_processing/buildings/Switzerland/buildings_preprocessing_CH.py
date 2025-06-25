import numpy as np
import pandas as pd
import pickle


from model.common.auxiliary_functions import moving_average, linear_fitting, create_years_list, my_pickle_dump
from model.common.io_database import update_database_from_dm, csv_database_reformat, read_database_to_dm
from _database.pre_processing.api_routines_CH import get_data_api_CH
from model.common.data_matrix_class import DataMatrix
from model.common.constant_data_matrix_class import ConstantDataMatrix

import os


def df_excel_to_dm(df, names_dict, var_name, unit, num_cat, keep_first=False, country='Switzerland'):
    # df from excel to dm
    # Remove nans and empty columns/rows
    if np.nan in df.columns:
        df.drop(columns=np.nan, inplace=True)
    # Change headers
    df.rename(columns={df.columns[0]: 'Variables'}, inplace=True)
    df.set_index('Variables', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    # Filter rows that contain at least one number (integer or float)
    df = df[df.apply(lambda row: row.map(pd.api.types.is_number), axis=1).any(axis=1)]
    df_clean = df.loc[:, df.apply(lambda col: col.map(pd.api.types.is_number)).any(axis=0)].copy()
    # Extract only the data we are interested in:
    df_filter = df_clean.loc[names_dict.keys()].copy()
    df_filter = df_filter.apply(lambda col: pd.to_numeric(col, errors='coerce'))
    #df_filter = df_filter.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
    df_filter.reset_index(inplace=True)
    # Keep only first 10 caracters
    df_filter['Variables'] = df_filter['Variables'].replace(names_dict)
    if keep_first:
        df_filter = df_filter.drop_duplicates(subset=['Variables'], keep='first')
    df_filter = df_filter.groupby(['Variables']).sum()
    df_filter.reset_index(inplace=True)

    # Pivot the dataframe
    df_filter['Country'] = country
    df_T = pd.melt(df_filter, id_vars=['Variables', 'Country'], var_name='Years', value_name='values')
    df_pivot = df_T.pivot_table(index=['Country', 'Years'], columns=['Variables'], values='values', aggfunc='sum')
    df_pivot = df_pivot.add_suffix('[' + unit + ']')
    df_pivot = df_pivot.add_prefix(var_name + '_')
    df_pivot.reset_index(inplace=True)

    # Drop non numeric values in Years col
    df_pivot['Years'] = pd.to_numeric(df_pivot['Years'], errors='coerce')
    df_pivot = df_pivot.dropna(subset=['Years'])

    dm = DataMatrix.create_from_df(df_pivot, num_cat=num_cat)
    return dm


def compute_new_area_transformed(dm_floor_area, years_ots, cat_map_sfh, cat_map_mfh):

    dm = dm_floor_area.copy()
    idx = dm.idx
    # removing 2010 value because it is off
    dm.array[:, idx[2010], ...] = np.nan
    linear_fitting(dm, dm.col_labels['Years'])

    dm.lag_variable('bld_floor-area_stock', shift=1, subfix='_tm1')
    dm.operation('bld_floor-area_stock', '-', 'bld_floor-area_stock_tm1', out_col='bld_floor-area_transformed', unit='m2')

    idx = dm.idx
    dm.array[:, 0, idx['bld_floor-area_transformed'], ...] = np.nan

    # Remove negative values
    dm.array[:, :, idx['bld_floor-area_transformed'], ...] \
        = np.maximum(dm.array[:, :, idx['bld_floor-area_transformed'], ...], 0)

    dm.filter({'Variables': ['bld_floor-area_transformed']}, inplace=True)
    # E.g. an increase in stock for a building in category 2016-2020, in 2019, is not a transformation,
    # it is simply a new construction. Therefore, we set it to 0
    construction_period = ['2016-2020', '2021-2023', '2011-2015']
    idx = dm.idx
    for period in construction_period:
        y0 = dm.col_labels['Years'][0]
        end_yr = int(period.split('-')[1])
        for yr in range(y0, end_yr+1):
            if yr in dm.col_labels['Years']:
                dm.array[:, idx[yr], :, :, idx[period]] = 0

    # Fill in previous years with mean values
    arr_mean = np.nanmean(dm.array[...], axis=1)

    years_missing = list(set(years_ots) - set(dm.col_labels['Years']))
    dm.add(0, col_label=years_missing, dummy=True, dim='Years')
    dm.sort('Years')
    idx = dm.idx
    for yr in years_missing:
        dm.array[:, idx[yr], :, :, :] = arr_mean

    dm.fill_nans(dim_to_interp='Years')

    construction_period = ['2016-2020', '2021-2023', '1991-2000', '2001-2005', '2006-2010', '2011-2015']
    for period in construction_period:
        end_yr = int(period.split('-')[1])
        y0 = dm.col_labels['Years'][0]
        # I think the + 3 is arbitrary, but basically to say that one would not transform right away a new building
        for yr in range(y0, end_yr+3):
            if yr in dm.col_labels['Years']:
                dm.array[:, idx[yr], :, :, idx[period]] = 0

    dm_sfh = dm.filter({'Categories1': ['single-family-households']})
    dm_sfh.groupby(cat_map_sfh, dim='Categories2', inplace=True)
    dm_mfh = dm.filter({'Categories1': ['multi-family-households']})
    dm_mfh.groupby(cat_map_mfh, dim='Categories2', inplace=True)

    dm = dm_mfh
    dm.append(dm_sfh, dim='Categories1')

    return dm


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


def extrapolate_energy_categories_to_missing_years(dm_floor_area, cat_map_sfh, cat_map_mfh):

    dm = dm_floor_area.copy()
    dm.filter({'Variables': ['bld_floor-area_stock']}, inplace=True)
    dm.array[:, 0, ...] = np.nan
    #dm_floor_area.group_all('Categories2')
    years_missing = list(set(years_ots) - set(dm.col_labels['Years']))
    dm.add(np.nan, dummy=True, dim='Years', col_label=years_missing)
    dm.sort('Years')

    # Forecasting
    arr_floor_cap = dm.array / dm_pop.array[..., np.newaxis, np.newaxis]
    dm.add(arr_floor_cap, dim='Variables', col_label='bld_floor-area_stock-cap', unit='m2/cap')
    linear_fitting(dm, dm.col_labels['Years'])
    idx = dm.idx
    dm.array[:, :, idx['bld_floor-area_stock'], ...] = \
        dm.array[:, :, idx['bld_floor-area_stock-cap'], ...] * dm_pop.array[..., np.newaxis]

    dm.filter({'Variables': ['bld_floor-area_stock']}, inplace=True)
    # From 1990 to start of the construction period, there cannot be building of said period
    construction_period = ['2016-2020', '2021-2023', '1991-2000', '2001-2005', '2006-2010', '2011-2015']
    for period in construction_period:
        start_yr = int(period.split('-')[0])
        y0 = dm.col_labels['Years'][0]
        for yr in range(y0, start_yr):
            if yr in dm.col_labels['Years']:
                dm.array[:, idx[yr], :, :, idx[period]] = 0

    # put nan during the construction period
    construction_period = ['1991-2000', '2001-2005', '2006-2010']
    for period in construction_period:
        start_yr = int(period.split('-')[0])
        end_yr = int(period.split('-')[1])
        for yr in range(start_yr, end_yr):
            if yr in dm.col_labels['Years']:
                dm.array[:, idx[yr], :, :, idx[period]] = np.nan

    dm.fill_nans(dim_to_interp='Years')

    dm_sfh = dm.filter({'Categories1': ['single-family-households']})
    dm_sfh.groupby(cat_map_sfh, dim='Categories2', inplace=True)
    dm_mfh = dm.filter({'Categories1': ['multi-family-households']})
    dm_mfh.groupby(cat_map_mfh, dim='Categories2', inplace=True)

    dm = dm_mfh
    dm.append(dm_sfh, dim='Categories1')
    dm.normalise(dim='Categories2', inplace=True, keep_original=True)
    dm.drop(col_label=['bld_floor-area_stock'], dim='Variables')

    return dm


def extrapolate_floor_area_to_missing_years(dm):
    dm.group_all('Categories2')

    dm.filter({'Variables': ['bld_floor-area_stock']}, inplace=True)
    dm.array[:, 0, ...] = np.nan
    years_missing = list(set(years_ots) - set(dm.col_labels['Years']))
    dm.add(np.nan, dummy=True, dim='Years', col_label=years_missing)
    dm.sort('Years')

    # Forecasting
    arr_floor_cap = dm.array / dm_pop.array[..., np.newaxis]
    dm.add(arr_floor_cap, dim='Variables', col_label='bld_floor-area_stock-cap', unit='m2/cap')
    linear_fitting(dm, dm.col_labels['Years'])
    idx = dm.idx
    dm.array[:, :, idx['bld_floor-area_stock'], ...] = \
        dm.array[:, :, idx['bld_floor-area_stock-cap'], ...] * dm_pop.array[...]

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


def compute_bld_floor_area_stock_tranformed_avg_new_area(table_id, file, years_ots, cat_map_sfh, cat_map_mfh):

    dm_floor_area, dm_num_bld = extract_stock_floor_area(file, table_id)

    #### Section to look into what happens to old buildings in recent years: i.e. renovation by going from mfh to sfh
    dm_area_transformed = compute_new_area_transformed(dm_floor_area, years_ots, cat_map_sfh, cat_map_mfh)

    dm_floor_area.append(dm_num_bld, dim='Variables')
    dm_floor_area.operation('bld_floor-area_stock', '/', 'bld_stock-number-bld',
                            out_col='bld_avg-floor-area-new', unit='m2/bld')

    # Extracting the average new built floor area for sfh and mfh
    # (Basically you want the category construction period to become the year category)
    dm_avg_floor_area = compute_avg_floor_area(dm_floor_area, years_ots)

    # Drop split by construction year for floor-area stock
    # Add missing years
    dm_energy_cat = extrapolate_energy_categories_to_missing_years(dm_floor_area, cat_map_sfh, cat_map_mfh)
    dm_energy_cat.append(dm_area_transformed, dim='Variables')

    dm_floor_area = extrapolate_floor_area_to_missing_years(dm_floor_area)

    idx = dm_floor_area.idx
    idx_e = dm_energy_cat.idx
    arr = dm_energy_cat.array[:, :, idx_e['bld_floor-area_stock_share'], :, :]\
          * dm_floor_area.array[:, :, idx['bld_floor-area_stock'], :, np.newaxis]
    dm_energy_cat.add(arr, dim='Variables', col_label='bld_floor-area_stock', unit='m2')

    dm_floor_area.append(dm_avg_floor_area, dim='Variables')

    return dm_floor_area, dm_energy_cat


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


def compute_bld_floor_area_new(dm_bld_new_buildings_1, dm_bld_new_buildings_2, dm_bld_area_stock, dm_pop):
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

def clean_floor_area_stock(dm, cat_map):
    # You cannot have stock of a future construction period (i.e. envelope class / energy category)
    idx = dm.idx
    start_yr = dm.col_labels['Years'][0]
    for cat, period in cat_map.items():
        end_yr = period[0]
        period_list = list(range(start_yr, end_yr))
        idx_period = [idx[yr] for yr in period_list]
        # Set stock to 0 until beginning of construction period
        dm.array[:, idx_period, idx['bld_floor-area_stock'], :, idx[cat]] = 0
        # Smooth data in the rest of the time frame
        not_period = list(set(range(len(dm.col_labels['Years']))) - set(idx_period[:-1]))
        not_period.sort()
        if not_period is not None:
            window_size = 3  # Change window size to control the smoothing effect
            nb_c = len(dm.col_labels['Country'])
            nb_y = len(not_period)
            len_cat1 = len(dm.col_labels['Categories1'])
            for i in range(2):
                arr = dm.array[:, not_period, idx['bld_floor-area_stock'], :, idx[cat]].reshape((nb_y, nb_c, len_cat1))
                data_smooth = moving_average(arr, window_size, axis=0)
                dm.array[:, not_period[1:-1], idx['bld_floor-area_stock'], :, idx[cat]] = data_smooth
    return


def clean_floor_area_new(dm, cat_map):
    # You can only have new buildings during the construction period
    idx = dm.idx
    for cat, period in cat_map.items():
        start_yr = period[0]
        end_yr = period[1]
        period_list = list(range(start_yr, end_yr))
        idx_not_period = [idx[yr] for yr in dm.col_labels['Years'] if yr not in period_list]
        dm.array[:, idx_not_period, idx['bld_floor-area_new'], :, idx[cat]] = 0
    return

def compute_bld_demolition_rate(dm_in, cat_map):

    dm = dm_in.filter({'Variables': ['bld_floor-area_stock', 'bld_floor-area_new']})

    clean_floor_area_stock(dm, cat_map)

    clean_floor_area_new(dm, cat_map)

    # s(t) = s(t-1) + n(t) - w(t)
    # w(t) = s(t-1) + n(t) + tr(t) + r(t) - s(t)
    # r(t) = R s(t-1)
    dm.append(dm_in.filter({'Variables': ['bld_floor-area_transformed']}, inplace=False), dim='Variables')

    # dem-rate(t-1) = w(t) / s(t-1)
    idx = dm.idx
    dm.lag_variable('bld_floor-area_stock', shift=1, subfix='_tm1')
    arr_waste = dm.array[:, :, idx['bld_floor-area_stock_tm1'], ...] \
                + dm.array[:, :, idx['bld_floor-area_new'], ...] \
                + dm.array[:, :, idx['bld_floor-area_transformed'], ...] \
                - dm.array[:, :, idx['bld_floor-area_stock'], ...]

    #dm.operation('bld_floor-area_stock_tm1', '-', 'bld_floor-area_stock', out_col='bld_floor-area_deltas', unit='m2')
    dm.add(arr_waste, dim='Variables', unit='m2', col_label='bld_floor-area_waste')

    idx = dm.idx
    start_yr = dm.col_labels['Years'][0]
    for cat, period in cat_map.items():
        end_yr = min(period[1] + 3, dm.col_labels['Years'][-1]+1)
        period_list = list(range(start_yr, end_yr))
        idx_period = [idx[yr] for yr in period_list]
        dm.array[:, idx_period, idx['bld_floor-area_waste'], :, idx[cat]] = 0

    dm.operation('bld_floor-area_waste', '/', 'bld_floor-area_stock_tm1',
                                out_col='bld_demolition-rate_tm1', unit='%')
    dm.lag_variable('bld_demolition-rate_tm1', shift=-1, subfix='_tp1')
    dm.rename_col('bld_demolition-rate_tm1_tp1', 'bld_demolition-rate', dim='Variables')


    # Compute demolition rate
    dm_demolition_rate = dm.filter({'Variables': ['bld_demolition-rate']}, inplace=False)
    # Demolition rate cannot be negative. We impose that it has to be at least 0.1%
    dm_demolition_rate.array = np.maximum(0.001, dm_demolition_rate.array)
    # Smooth the demolition-rate
    window_size = 3  # Change window size to control the smoothing effect
    data_smooth = moving_average(dm_demolition_rate.array, window_size, axis=dm_demolition_rate.dim_labels.index('Years'))
    dm_demolition_rate.array[:, 1:-1, ...] = data_smooth

    idx = dm_demolition_rate.idx
    start_yr = dm_demolition_rate.col_labels['Years'][0]
    for cat, period in cat_map.items():
        end_yr = min(period[1] + 3, dm_demolition_rate.col_labels['Years'][-1]+1)
        period_list = list(range(start_yr, end_yr))
        idx_period = [idx[yr] for yr in period_list]
        dm_demolition_rate.array[:, idx_period, idx['bld_demolition-rate'], :, idx[cat]] = 0

    # Recompute new fleet so that it matches the demolition rate
    # n(t) = s(t) - s(t-1) + w(t)
    # w(t) = dem-rate(t-1) * s(t-1)
    # n(t) = s(t) - s(t-1) - dem-rate(t-1) * s(t-1)
    dm.filter({'Variables': ['bld_floor-area_stock', 'bld_floor-area_stock_tm1']}, inplace=True)
    dm.append(dm_demolition_rate, dim='Variables')
    dm.operation('bld_floor-area_stock', '-', 'bld_floor-area_stock_tm1', out_col='bld_delta-stock', unit='m2')
    dm.lag_variable('bld_demolition-rate', shift=1, subfix='_tm1')
    dm.operation('bld_demolition-rate_tm1', '*', 'bld_floor-area_stock_tm1', out_col='bld_floor-area_waste', unit='m2')
    dm.operation('bld_delta-stock', '+', 'bld_floor-area_waste', out_col='bld_floor-area_new', unit='m2')

    # Where new < 0 -> new(t) = s(t) - s(t-1) - w(t) e.g  -2 = 8 - 10 + 0
    # change waste so that new = 0
    dm_new = dm.filter({'Variables': ['bld_floor-area_new']})
    dm_waste = dm.filter({'Variables': ['bld_floor-area_waste']})
    dm_stock = dm.filter({'Variables': ['bld_floor-area_stock']})
    dm_rate_tm1 = dm.filter({'Variables': ['bld_demolition-rate_tm1']})

    mask = dm_new.array < 0
    dm_stock.array[mask] = dm_stock.array[mask] - dm_new.array[mask]
    dm_new.array[mask] = 0

    dm_out = dm_stock.copy()
    dm_out.append(dm_new, dim='Variables')
    dm_out.lag_variable('bld_floor-area_stock', shift=1, subfix='_tm1')
    dm_out.operation('bld_floor-area_stock_tm1', '-', 'bld_floor-area_stock', out_col='bld_delta-stock', unit='m2')
    dm_out.operation('bld_delta-stock', '+', 'bld_floor-area_new', out_col='bld_floor-area_waste', unit='m2')
    dm_out.operation('bld_floor-area_waste', '/', 'bld_floor-area_stock_tm1', out_col='bld_demolition-rate_tm1', unit='%')
    dm_out.lag_variable('bld_demolition-rate_tm1', shift=-1, subfix='_tp1')
    dm_out.rename_col('bld_demolition-rate_tm1_tp1', 'bld_demolition-rate', dim='Variables')

    #dm_stock_tm1.array[mask] = dm_stock_tm1.array[mask] * dm_rate_tm1.array[mask]
    #dm_stock_tm1.lag_variable('bld_floor-area_stock_tm1', shift=-1, subfix='_tp1')
    #dm_stock_tm1.filter({'Variables': ['bld_floor-area_stock_tm1_tp1']}, inplace=True)
    #dm.append(dm_stock_tm1, dim='Variables')

    dm_out.filter({'Variables': ['bld_floor-area_stock', 'bld_floor-area_new',
                                 'bld_demolition-rate', 'bld_floor-area_waste']}, inplace=True)
    idx = dm_out.idx
    dm_out.array[:, 0, idx['bld_floor-area_new'], ...] = np.nan
    dm_out.fill_nans(dim_to_interp='Years')

    return dm_out


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


def compute_renovated_buildings(dm_bld, nb_buildings_renovated, VD_share, share_by_bld):
    dm_bld.add(np.nan, dim='Variables', dummy=True, col_label='bld_nb-bld-renovated', unit='number')
    idx = dm_bld.idx
    for yr in nb_buildings_renovated.keys():
        for cat in ['single-family-households', 'multi-family-households']:
            dm_bld.array[idx['Switzerland'], idx[yr], idx['bld_nb-bld-renovated'], idx[cat]] \
                = nb_buildings_renovated[yr] * share_by_bld[cat]
            dm_bld.array[idx['Vaud'], idx[yr], idx['bld_nb-bld-renovated'], idx[cat]] \
                = nb_buildings_renovated[yr] * VD_share[yr] * share_by_bld[cat]
    return dm_bld


def extract_renovation_1990_2000(table_id, file, share_thermal):
    try:
        with open(file, 'rb') as handle:
            dm_renovation = pickle.load(handle)
    except OSError:
        structure, title = get_data_api_CH(table_id, mode='example', language='fr')

        dm_renovation = None
        for e in structure['Epoque de construction']:
            filter = {'Canton': ['Suisse', 'Vaud'],
                      'Type de bâtiment': structure['Type de bâtiment'],
                      "Nombre d'étages": structure["Nombre d'étages"],
                      'Nombre de logements': structure['Nombre de logements'],
                      'Epoque de construction': e,
                      'Epoque de rénovation': structure['Epoque de rénovation'],
                      'Année': structure['Année']
                      }

            mapping = {'Country': 'Canton',
                       'Years': 'Année',
                       'Variables': 'Epoque de construction',
                       'Categories1': 'Type de bâtiment',
                       'Categories2': 'Epoque de rénovation'}
            unit = ['number']
            dm_renovation_e = get_data_api_CH(table_id, mode='extract', filter=filter, mapping_dims=mapping, units=unit,
                                            language='fr')

            if dm_renovation is not None:
                dm_renovation.array = dm_renovation.array + dm_renovation_e.array
            else:
                dm_renovation = dm_renovation_e.groupby({'bld_nb-bld-renovated': '.*'}, regex=True, inplace=False, dim='Variables')

        dm_renovation.groupby({'single-family-households': ['Maisons individuelles'],
                               'multi-family-households': ['Maisons à plusieurs logements',
                                                           "Bâtiments d'habitation avec usage annexe",
                                                           "Bâtiments partiellement à usage d'habitation"]},
                              dim='Categories1', inplace=True)
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, 'wb') as handle:
            pickle.dump(dm_renovation, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # Renovation here includes all types of renovation
    # According to the Programme Bâtiments, usually 30%-35% of renovation are for insulation
    dm_renovation.array = dm_renovation.array * share_thermal
    map_to_years = {'Rénovés dans les 4 dernières années': [2000, 4],
                    'Rénovés dans les 5 à 9 années précédentes': [1990, 5]}
    # Add missing years
    dm_tot_bld = dm_renovation.group_all('Categories2', inplace=False)
    dm_tot_bld.add(np.nan, dummy=True, dim='Years', col_label=create_years_list(1991, 1999, 1))
    dm_tot_bld.sort('Years')
    dm_tot_bld.rename_col('bld_nb-bld-renovated', 'bld_nb-bld', dim='Variables')

    dm_renovation.add(np.nan, dummy=True, dim='Years', col_label=create_years_list(1991, 1999, 1))
    dm_renovation.sort('Years')

    dm_renovation.add(np.nan, dummy=True, dim='Categories2', col_label='total')
    idx = dm_renovation.idx
    for cat, values in map_to_years.items():
        yr = values[0]
        interval = values[1]
        dm_renovation.array[:, idx[yr], idx['bld_nb-bld-renovated'], :, idx['total']] =\
            dm_renovation.array[:, idx[yr], idx['bld_nb-bld-renovated'], :, idx[cat]] / interval

    dm_renovation.filter({'Categories2': ['total']}, inplace=True)
    #dm_renovation.group_all('Categories2', inplace=True)
    dm_renovation.fill_nans(dim_to_interp='Years')
    dm_renovation.group_all('Categories2', inplace=True)
    dm_tot_bld.fill_nans(dim_to_interp='Years')

    dm_renovation.append(dm_tot_bld, dim='Variables')
    dm_renovation.operation('bld_nb-bld-renovated', '/', 'bld_nb-bld', out_col='bld_renovation-rate', unit='%')

    return dm_renovation


def compute_renovation_rate(dm_renovation, years_ots):

    dm_renovation.operation('bld_nb-bld-renovated', '/', 'bld_nb-bld', out_col='bld_renovation-rate', unit='%')
    years_missing = list(set(years_ots) - set(dm_renovation.col_labels['Years']))
    dm_renovation.add(np.nan, dummy=True, col_label=years_missing, dim='Years')
    dm_renovation.sort('Years')
    dm_renovation.fill_nans(dim_to_interp='Years')
    dm_renovation.filter({'Variables': ['bld_renovation-rate']}, inplace=True)

    return dm_renovation

def extract_bld_floor_area_stock_envelope_cat(table_id, file, dm_pop, years_ots, cat_dict):

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

    dm_floor_area.append(dm_num_bld, dim='Variables')

    dm_floor_area.groupby(cat_dict, dim='Categories2', inplace=True)

    dm_floor_area.normalise(dim='Categories2', inplace=True, keep_original=True)

    dm_floor_area.filter({'Variables': ['bld_floor-area_stock_share']}, inplace=True)

    return dm_floor_area


def compute_new_area_by_energy_cat(dm_bld_area_new, dm_energy_cat, cat_map):

    dm_energy_cat.add(0, dummy=True, dim='Variables', col_label='bld_floor-area_new', unit='m2')

    idx = dm_energy_cat.idx
    idx_n = dm_bld_area_new.idx
    for cat, period in cat_map.items():
        start_yr = period[0]
        end_yr = period[1]
        period_list = list(range(start_yr, end_yr+1))
        idx_period = [idx[yr] for yr in period_list]
        dm_energy_cat.array[:, idx_period, idx['bld_floor-area_new'], :, idx[cat]] = 1

    dm_energy_cat.array[:, :, idx['bld_floor-area_new'], ...] = \
        dm_energy_cat.array[:, :, idx['bld_floor-area_new'], :, :] \
        * dm_bld_area_new.array[:, :, idx_n['bld_floor-area_new'], :, np.newaxis]

    return dm_energy_cat


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


def adjust_based_on_renovation(dm_in, dm_rr, dm_renov_distr):

    dm = dm_in.copy()
    idx_r = dm_rr.idx
    idx_d = dm_renov_distr.idx

    #arr_tot_ren = np.nansum(dm.array[:, :, idx_s['bld_floor-area_stock'], :, :]
    #                        * dm_rr.array[:, :, idx_r['bld_renovation-rate'], :, np.newaxis], axis=-1)
    #arr = arr_tot_ren[..., np.newaxis] * dm_renov_distr.array[:, :, idx_d['bld_renovation-redistribution'], np.newaxis, :]
    #dm.add(arr, dim='Variables', unit='m2', col_label='bld_floor-area_renovated')
    #dm_rr.add(arr_tot_ren, dim='Variables', col_label='bld_floor-area_renovated', unit='m2')

    # s(t) = s(t-1) + R(t)s(t-1) + n(t) - w(t) -> s(t-1) (1 + R(t)) = s(t) - n(t) + w(t)
    idx = dm.idx
    for ti in reversed(dm.col_labels['Years'][1:]):
        stock_t = dm.array[:, idx[ti], idx['bld_floor-area_stock'], :, :]
        new_t = dm.array[:, idx[ti], idx['bld_floor-area_new'], :, :]
        #ren_t = np.nansum(dm.array[:, idx[ti-1], idx['bld_floor-area_stock'], :, :]
        #                  * dm_rr.array[:, idx_r[ti], idx_r['bld_renovation-rate'], :, np.newaxis], axis=-1, keepdims=True) \
        #        * dm_renov_distr.array[:, idx_d[ti], idx_d['bld_renovation-redistribution'], np.newaxis, :]
        waste_t = dm.array[:, idx[ti], idx['bld_floor-area_waste'], :, :]
        tmp_t = stock_t - new_t + waste_t
        stock_tm1 = tmp_t / (1 + dm_rr.array[:, idx_r[ti], idx_r['bld_renovation-rate'], :, np.newaxis]
                               * (- dm_renov_distr.array[:, idx_d[ti], idx_d['bld_renovation-redistribution-out'], np.newaxis, :]
                                + dm_renov_distr.array[:, idx_d[ti], idx_d['bld_renovation-redistribution-in'], np.newaxis, :]))
        dm.array[:, idx[ti - 1], idx['bld_floor-area_stock'], :, :] = stock_tm1
        dm.array[:, idx[ti - 1], idx['bld_demolition-rate'], :, :] = waste_t / stock_tm1


    dm_dem_rate = dm.filter({'Variables': ['bld_demolition-rate']})
    mask = np.isnan(dm_dem_rate.array)
    dm_dem_rate.array[mask] = 0
    dm.array[:, :, idx['bld_demolition-rate'], :, :] = dm_dem_rate.array[:, :, 0, ...]

    return dm


def recompute_floor_area_per_capita(dm_all, dm_pop):

    dm_floor_stock = dm_all.filter({'Variables': ['bld_floor-area_stock'],
                                    'Categories': {'single-family-households', 'multi-family-households'}}, inplace=False)

    # Computer m2/cap for lifestyles
    dm_floor_stock.group_all(dim='Categories2')
    dm_floor_stock.group_all(dim='Categories1')
    dm_floor_stock.append(dm_pop, dim='Variables')

    dm_floor_stock.operation('bld_floor-area_stock', '/', 'lfs_population_total',
                             out_col='lfs_floor-intensity_space-cap', unit='m2/cap')

    dm_floor_stock.filter({'Variables': ['lfs_floor-intensity_space-cap']}, inplace=True)

    return dm_floor_stock


def compute_building_mix(dm_all):

    dm_building_mix = dm_all.filter({'Variables': ['bld_floor-area_stock', 'bld_floor-area_new']}, inplace=False).flatten()
    dm_building_mix.normalise('Categories1', keep_original=True)
    dm_building_mix.deepen()
    dm_building_mix.rename_col(['bld_floor-area_stock_share', 'bld_floor-area_new_share'],
                               ['bld_building-mix_stock', 'bld_building-mix_new'], dim='Variables')
    dm_building_mix.filter({'Variables': ['bld_building-mix_stock', 'bld_building-mix_new']}, inplace=True)

    return dm_building_mix


######################
### HOUSEHOLD-SIZE ###
######################
def extract_lfs_household_size(years_ots, table_id, file):
    try:
        with open(file, 'rb') as handle:
            dm_household_size = pickle.load(handle)
    except OSError:
        structure, title = get_data_api_CH(table_id, mode='example')
        # Extract buildings floor area
        filter = {'Year': structure['Year'],
                  'Canton (-) / District (>>) / Commune (......)': ['Schweiz / Suisse / Svizzera / Switzerland', '- Vaud'],
                  'Household size': ['1 person', '2 persons', '3 persons', '4 persons', '5 persons', '6 persons or more']}
        mapping_dim = {'Country': 'Canton (-) / District (>>) / Commune (......)', 'Years': 'Year',
                       'Variables': 'Household size'}
        unit_all = ['people'] * len(filter['Household size'])
        # Get api data
        dm_household = get_data_api_CH(table_id, mode='extract', filter=filter, mapping_dims=mapping_dim, units=unit_all)

        dm_household.rename_col(['Schweiz / Suisse / Svizzera / Switzerland', '- Vaud'], ['Switzerland', 'Vaud'],
                                dim='Country')
        drop_strings = [' persons or more', ' persons', ' person']
        for drop_str in drop_strings:
            dm_household.rename_col_regex(drop_str, '', dim='Variables')
        # dm_household contains the number of household per each household-size
        # Compute the average household size by doing the weighted average
        sizes = np.array([int(num_ppl) for num_ppl in dm_household.col_labels['Variables']])
        arr_weighted_size = dm_household.array * sizes[np.newaxis, np.newaxis, :]
        arr_avg_size = np.nansum(arr_weighted_size, axis=-1, keepdims=True) / np.nansum(dm_household.array, axis=-1,
                                                                                        keepdims=True)
        # Create new datamatrix
        dm_household_size = DataMatrix.based_on(arr_avg_size, dm_household, change={'Variables': ['lfs_household-size']},
                                                units={'lfs_household-size': 'people'})
        linear_fitting(dm_household_size, years_ots)
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, 'wb') as handle:
            pickle.dump(dm_household_size, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm_household_size


def extract_heating_technologies_old(table_id, file, cat_sfh, cat_mfh):
    # Domaine de l'énergie: bâtiments selon le canton, le type de bâtiment, l'époque de construction, le type de chauffage,
    # la production d'eau chaude, les agents énergétiques utilisés pour le chauffage et l'eau chaude, 1990 et 2000
    try:
        with open(file, 'rb') as handle:
            dm_heating_old = pickle.load(handle)
    except OSError:
        structure, title = get_data_api_CH(table_id, mode='example', language='fr')
        # Extract buildings floor area
        filter = structure.copy()
        filter['Canton'] = ['Suisse', 'Vaud']
        mapping_dim = {'Country': 'Canton', 'Years': 'Année',
                       'Variables': 'Epoque de construction', 'Categories1': 'Type de bâtiment',
                       'Categories2': 'Agent énergétique pour le chauffage'}
        dm_heating_old = None
        tot_bld = 0
        for t in structure['Type de chauffage']:
            for a in structure["Agent énergétique pour l'eau chaude"]:
                filter['Type de chauffage'] = [t]
                filter["Agent énergétique pour l'eau chaude"] = [a]
                unit_all = ['number'] * len(structure['Epoque de construction'])
                dm_heating_old_t = get_data_api_CH(table_id, mode='extract', filter=filter,
                                                mapping_dims=mapping_dim, units=unit_all, language='fr')
                if dm_heating_old is None:
                    dm_heating_old = dm_heating_old_t.copy()
                else:
                    dm_heating_old.array = dm_heating_old_t.array + dm_heating_old.array
                partial_bld = np.nansum(dm_heating_old_t.array[0, 0, ...])
                tot_bld = tot_bld + partial_bld
                print(t, a, partial_bld, tot_bld)

        dm_heating_old.rename_col(['Suisse'], ['Switzerland'], dim='Country')
        dm_heating_old.groupby({'single-family-households': ['Maisons individuelles'],
                               'multi-family-households': ['Maisons à plusieurs logements',
                                                           "Bâtiments d'habitation avec usage annexe",
                                                           "Bâtiments partiellement à usage d'habitation"]},
                                dim='Categories1', inplace=True)

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, 'wb') as handle:
            pickle.dump(dm_heating_old, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dm_heating_old.rename_col_regex('Construits ', '', dim='Variables')
    dm_heating_old.rename_col_regex('entre ', '', dim='Variables')
    dm_heating_old.rename_col_regex(' et ', '-', dim='Variables')
    dm_heating_old.rename_col_regex('avant ', 'Avant ', dim='Variables')
    dm_heating_old.groupby({'1991-2000': ['1991-1995', '1996-2000']}, dim='Variables', inplace=True)

    # Group by construction period
    dm_heating_sfh = dm_heating_old.filter({'Categories1': ['single-family-households']}, inplace=False)
    dm_heating_sfh.groupby({'bld_heating-mix_F':  cat_sfh['F']}, dim='Variables', inplace=True)
    dm_heating_sfh.groupby({'bld_heating-mix_E': cat_sfh['E']}, dim='Variables', inplace=True)
    dm_heating_sfh.groupby({'bld_heating-mix_D': cat_sfh['D']}, dim='Variables', inplace=True)

    dm_heating_mfh = dm_heating_old.filter({'Categories1': ['multi-family-households']}, inplace=False)
    dm_heating_mfh.groupby({'bld_heating-mix_F': cat_mfh['F']}, dim='Variables', inplace=True)
    dm_heating_mfh.groupby({'bld_heating-mix_E': cat_mfh['E']}, dim='Variables', inplace=True)
    dm_heating_mfh.groupby({'bld_heating-mix_D': cat_mfh['D']}, dim='Variables', inplace=True)

    # Merge sfh and mfh
    dm_heating_mfh.append(dm_heating_sfh, dim='Categories1')
    dm_heating_old = dm_heating_mfh

    dm_heating_old.groupby({'other': ['Autre agent énergétique (chauf.)', 'Sans chauffage']},
                           dim='Categories2', inplace=True)
    dm_heating_old.rename_col(['Mazout (chauf.)', 'Bois (chauf.)', 'Pompe à chaleur (chauf.)',
                               'Electricité (chauf.)', 'Gaz (chauf.)', 'Chaleur à distance (chauf.)',
                               'Charbon (chauf.)', 'Capteur solaire (chauf.)'],
                              ['heating-oil', 'wood', 'heat-pump', 'electricity', 'gas', 'district-heating', 'coal', 'solar'],
                              dim='Categories2')

    dm_heating_old.deepen(based_on='Variables')
    return dm_heating_old



def extract_heating_technologies(table_id, file, cat_sfh, cat_mfh):
    try:
        with open(file, 'rb') as handle:
            dm_heating = pickle.load(handle)
    except OSError:
        structure, title = get_data_api_CH(table_id, mode='example', language='fr')
        # Extract buildings floor area
        filter = {'Année': structure['Année'],
                  'Canton': ['Suisse', 'Vaud'],
                  "Source d'énergie du chauffage": structure["Source d'énergie du chauffage"],
                  "Source d'énergie de l'eau chaude": structure["Source d'énergie de l'eau chaude"],
                  'Époque de construction': structure['Époque de construction'],
                  'Catégorie de bâtiment': structure['Catégorie de bâtiment']}
        mapping_dim = {'Country': 'Canton', 'Years': 'Année',
                       'Variables': 'Époque de construction', 'Categories1': 'Catégorie de bâtiment',
                       'Categories2': "Source d'énergie du chauffage"}
        unit_all = ['number'] * len(structure['Époque de construction'])
        # Get api data
        dm_heating = get_data_api_CH(table_id, mode='extract', filter=filter,
                                        mapping_dims=mapping_dim, units=unit_all, language='fr')
        dm_heating.rename_col(['Suisse'], ['Switzerland'], dim='Country')
        dm_heating.groupby({'single-family-households': ['Maisons individuelles'],
                               'multi-family-households': ['Maisons à plusieurs logements',
                                                           "Bâtiments d'habitation avec usage annexe",
                                                           "Bâtiments partiellement à usage d'habitation"]},
                              dim='Categories1', inplace=True)

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, 'wb') as handle:
            pickle.dump(dm_heating, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Group by construction period
    dm_heating_sfh = dm_heating.filter({'Categories1': ['single-family-households']}, inplace=False)
    cat_sfh = {'bld_heating-mix_' + key: value for key, value in cat_sfh.items()}
    dm_heating_sfh.groupby(cat_sfh, dim='Variables', inplace=True)

    dm_heating_mfh = dm_heating.filter({'Categories1': ['multi-family-households']}, inplace=False)
    cat_sfh = {'bld_heating-mix_' + key: value for key, value in cat_mfh.items()}
    dm_heating_mfh.groupby(cat_sfh, dim='Variables', inplace=True)
    # Merge sfh and mfh
    dm_heating_mfh.append(dm_heating_sfh, dim='Categories1')
    dm_heating = dm_heating_mfh

    dm_heating.groupby({'Other': ['Autre', 'Aucune']}, dim='Categories2', inplace=True)
    dm_heating.rename_col(
        ['Bois', 'Chaleur produite à distance', 'Electricité', 'Gaz', 'Mazout', 'Other', 'Pompe à chaleur', 'Soleil (thermique)'],
        ['wood', 'district-heating', 'electricity', 'gas', 'heating-oil', 'other', 'heat-pump', 'solar'], dim='Categories2')

    dm_heating.deepen(based_on='Variables')

    return dm_heating


def compute_heating_tech_mix_fts(dm_heating_tech):
    # dm_heating_tech.add(np.nan, dim='Years', dummy=True, col_label=years_fts)
    #dm_heating_tech_ots = dm_heating_tech.filter({'Years': years_ots})
    linear_fitting(dm_heating_tech, years_fts, based_on=[2021, 2022, 2023])
    dm_heating_tech.array = np.maximum(dm_heating_tech.array, 0)
    dm_heating_tech.normalise('Categories2')
    dm_heating_tech_fts = dm_heating_tech.filter({'Years': [years_ots[-1]] + years_fts})
    for i in range(3):
        window_size = 3  # Change window size to control the smoothing effect
        data_smooth = moving_average(dm_heating_tech_fts.array, window_size,
                                     axis=dm_heating_tech.dim_labels.index('Years'))
        dm_heating_tech_fts.array[:, 1:-1, ...] = data_smooth
    idx = dm_heating_tech_fts.idx
    dm_heating_tech_fts.array[:, idx[2025], ...] = np.nan
    dm_heating_tech_fts.fill_nans(dim_to_interp='Years')
    dm_heating_tech_fts.filter({'Years': years_fts}, inplace=True)
    dm_heating_tech_fts.normalise('Categories2')
    #dm_heating_tech_ots.append(dm_heating_tech_fts, dim='Years')
    #dm_heating_tech_ots.flatten().datamatrix_plot()
    return dm_heating_tech_fts


def calculate_heating_eff_fts(dm_heating_eff, years_fts, maximum_eff):
    dm_heat_pump = dm_heating_eff.filter({'Categories1': ['heat-pump']})
    dm_heating_eff.drop(dim='Categories1', col_label='heat-pump')
    linear_fitting(dm_heating_eff, years_fts, based_on=list(range(2015, 2023)))
    dm_heating_eff.array = np.minimum(dm_heating_eff.array, maximum_eff)
    linear_fitting(dm_heat_pump, years_fts, based_on=list(range(2015, 2023)))
    dm_heating_eff.append(dm_heat_pump, dim='Categories1')
    dm_heating_eff_fts = dm_heating_eff.filter({'Years': years_fts})

    return dm_heating_eff_fts


def harmonise_stock_new_renovated_transformed(dm, dm_rr, dm_renov_distr, cat_map):
    # dm contains floor-area_stock, floor-area_new, floor-area_transformed
    # s_c(t) = s_c(t - 1) + n_c(t) - w_c(t) + r_c(t) + t_c(t)
    # w_c(t) =  s_c(t - 1) - s_c(t)  + n_c(t) + r_c(t) + t_c(t)
    clean_floor_area_stock(dm, cat_map)
    clean_floor_area_new(dm, cat_map)

    dm.lag_variable('bld_floor-area_stock', shift=1, subfix='_tm1')

    # floor-area_renovated
    idx = dm.idx
    idx_r = dm_rr.idx
    idx_d = dm_renov_distr.idx
    ren_t = np.nansum(dm.array[:, :, idx['bld_floor-area_stock_tm1'], :, :]
                      * dm_rr.array[:, :, idx_r['bld_renovation-rate'], :, np.newaxis], axis=-1, keepdims=True) \
            * ( - dm_renov_distr.array[:, :, idx_d['bld_renovation-redistribution-out'], np.newaxis, :] +
                dm_renov_distr.array[:, :, idx_d['bld_renovation-redistribution-in'], np.newaxis, :])
    dm.add(ren_t, dim='Variables', unit='m2', col_label='bld_floor-area_renovated')

    # compute waste
    #  w_c(t) =  s_c(t - 1) - s_c(t)  + n_c(t) + r_c(t) + t_c(t)
    idx = dm.idx
    waste = dm.array[:, :, idx['bld_floor-area_stock_tm1'], ...] - dm.array[:, :, idx['bld_floor-area_stock'], ...]\
            + dm.array[:, :, idx['bld_floor-area_new'], ...] + dm.array[:, :, idx['bld_floor-area_renovated'], ...] \
            + dm.array[:, :, idx['bld_floor-area_transformed'], ...]
    waste = np.maximum(waste, dm.array[:, :, idx['bld_floor-area_stock_tm1'], ...]*0.001)
    dm.add(waste, dim='Variables', unit='m2', col_label='bld_floor-area_waste')

    # Adjust Transformed
    # t_c(t) = w_c(t) - n_c(t) - r_c(t) + s_c(t) - s_c(t-1)
    idx = dm.idx
    transf = - dm.array[:, :, idx['bld_floor-area_stock_tm1'], ...] + dm.array[:, :, idx['bld_floor-area_stock'], ...] \
             - dm.array[:, :, idx['bld_floor-area_new'], ...] - dm.array[:, :, idx['bld_floor-area_renovated'], ...] \
             + dm.array[:, :, idx['bld_floor-area_waste'], ...]
    dm.add(transf, dim='Variables', unit='m2', col_label='bld_floor-area_transformed_new')

    #dm.operation('bld_floor-area_transformed_new', '+', 'bld_floor-area_new', out_col='bld_floor-area_new2', unit='m2')
    dm.array[:, :, idx['bld_floor-area_new'], ...] = transf + dm.array[:, :, idx['bld_floor-area_new'], ...]

    dm.operation('bld_floor-area_waste', '/', 'bld_floor-area_stock_tm1', out_col='bld_demolition-rate_tm1', unit='%')
    dm.lag_variable('bld_demolition-rate_tm1', shift=-1, subfix='_tp1')
    dm.rename_col('bld_demolition-rate_tm1_tp1', 'bld_demolition-rate', dim='Variables')

    dm_all = dm.filter({'Variables': ['bld_floor-area_stock', 'bld_floor-area_new',
                                      'bld_demolition-rate', 'bld_floor-area_renovated']})

    return dm_all


def compute_floor_area_stock_v2(table_id, file, dm_pop, cat_map_sfh, cat_map_mfh):
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


def compute_waste(dm_stock_tot, dm_new_tot):

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


def compute_floor_area_waste_cat(dm_waste_tot):
    # Assuming the lifetime of a building is 50 years and buildings of categories D are built starting in 1981
    # the first waste of building in category D will by in 2031, before than is all category F.
    dm = dm_waste_tot.copy()
    dm.rename_col('single-family-households', 'single-family-households_F', dim='Categories1')
    dm.rename_col('multi-family-households', 'multi-family-households_F', dim='Categories1')
    dm.deepen()
    dm.add(0, dummy=True, dim='Categories2', col_label=['B', 'C', 'D', 'E'])

    return dm


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


def prepare_heating_mix_by_archetype():
    dict_sfh = {'F': {'heating-oil': 0.48,
                      'wood': 0.19,
                      'gas': 0.14,
                      'heat-pump': 0.09,
                      'electricity': 0.08,
                      'district-heating': 0.02,
                      'solar': 0},
                'E': {'heating-oil': 0.51,
                      'wood': 0.05,
                      'gas': 0.07,
                      'heat-pump': 0.15,
                      'electricity': 0.20,
                      'district-heating': 0.02,
                      'solar': 0},
                'D': {'heating-oil': 0.26,
                      'wood': 0.05,
                      'gas': 0.19,
                      'heat-pump': 0.23,
                      'electricity': 0.26,
                      'district-heating': 0.01,
                      'solar': 0},
                'C': {'heating-oil': 0.25,
                      'wood': 0.04,
                      'gas': 0.36,
                      'heat-pump': 0.32,
                      'electricity': 0.02,
                      'district-heating': 0.01,
                      'solar': 0},
                'B': {'heating-oil': 0.01,
                      'wood': 0.07,
                      'gas': 0.13,
                      'heat-pump': 0.73,
                      'electricity': 0.0,
                      'district-heating': 0.05,
                      'solar': 0.01}
                }

    dict_mfh = {'F': {'heating-oil': 0.57,
                      'wood': 0.02,
                      'gas': 0.27,
                      'heat-pump': 0.03,
                      'electricity': 0.03,
                      'district-heating': 0.08,
                      'solar': 0},
                'E': {'heating-oil': 0.36,
                      'wood': 0.01,
                      'gas': 0.43,
                      'heat-pump': 0.04,
                      'electricity': 0.11,
                      'district-heating': 0.05,
                      'solar': 0},
                'D': {'heating-oil': 0.37,
                      'wood': 0.02,
                      'gas': 0.50,
                      'heat-pump': 0.04,
                      'electricity': 0.03,
                      'district-heating': 0.04,
                      'solar': 0},
                'C': {'heating-oil': 0.16,
                      'wood': 0.04,
                      'gas': 0.64,
                      'heat-pump': 0.09,
                      'electricity': 0,
                      'district-heating': 0.07,
                      'solar': 0},
                'B': {'heating-oil': 0.01,
                      'wood': 0.10,
                      'gas': 0.27,
                      'heat-pump': 0.47,
                      'electricity': 0.0,
                      'district-heating': 0.14,
                      'solar': 0.01}
                }

    dict_archetype = {'single-family-households': dict_sfh, 'multi-family-households': dict_mfh}

    categories1 = list(dict_archetype.keys())
    categories1.sort()
    categories2 = list(dict_sfh.keys())
    categories2.sort()
    categories3 = list(dict_sfh['F'].keys())
    categories3.sort()

    cdm = ConstantDataMatrix(col_labels={'Variables': ['bld_heating-mix-archetype'],
                                         'Categories1': categories1,
                                         'Categories2': categories2,
                                         'Categories3': categories3},
                             units={'bld_heating-mix-archetype': '%'})
    idx = cdm.idx
    cdm.array = np.zeros((1, len(categories1), len(categories2), len(categories3)))
    for bld_type, dict_type in dict_archetype.items():
        for env_cat, dict_fuel in dict_type.items():
            for fuel, value in dict_fuel.items():
                cdm.array[idx['bld_heating-mix-archetype'], idx[bld_type], idx[env_cat], idx[fuel]] = value

    return cdm


def compute_heating_mix_by_category(dm_heating_tech, cdm_heating_archetypes, dm_all):
    # dm_heating_tech: share of building per heating technology, by sfh and mfh
    # cdm_heating_archetypes: portion of households per heating technology, envelope, sfh and mfh
    # I use the stock from dm_all
    # I want to compute the heating mix by sfh/mfh and envelope.

    dm_stock = dm_all.filter({'Variables': ['bld_floor-area_stock']})

    # STEP 1: obtain a heating-mix that sums to 1 over the envelope categories
    idx_h = cdm_heating_archetypes.idx
    idx_a = dm_stock.idx
    arr_tmp = cdm_heating_archetypes.array[np.newaxis, np.newaxis, idx_h['bld_heating-mix-archetype'], :, :, :] \
              * dm_stock.array[:, :, idx_a['bld_floor-area_stock'], :, :, np.newaxis]
    arr = arr_tmp/np.nansum(arr_tmp, axis=-2, keepdims=True)
    dm_heating_mix = DataMatrix.based_on(arr[:, :, np.newaxis, ...], dm_stock, units={'bld_heating-mix-tmp': '%'},
                                         change={'Variables': ['bld_heating-mix-tmp'],
                                                 'Categories3': cdm_heating_archetypes.col_labels['Categories3']})

    dm_stock_norm = dm_stock.normalise('Categories2', inplace=False)
    fuel_missing = list(set(dm_heating_tech.col_labels['Categories2']) - set(dm_heating_mix.col_labels['Categories3']))
    dm_heating_mix.add(np.nan, dim='Categories3', col_label=fuel_missing, dummy=True)
    idx = dm_heating_mix.idx
    idx_n = dm_stock_norm.idx
    for f in fuel_missing:
        dm_heating_mix.array[:, :, idx['bld_heating-mix-tmp'], :, :, idx[f]] = \
            dm_stock_norm.array[:, :, idx_n['bld_floor-area_stock_share'], :, :]
    dm_heating_mix.sort('Categories3')
    dm_heating_tech.sort('Categories2')
    idx = dm_heating_mix.idx
    idx_t = dm_heating_tech.idx
    arr = dm_heating_mix.array[:, :, idx['bld_heating-mix-tmp'], :, :, :] \
          * dm_heating_tech.array[:, :, idx_t['bld_heating-mix'], :, np.newaxis, :]
    dm_heating_mix.array[:, :, idx['bld_heating-mix-tmp'], :, :, :] = arr

    # Now it should sum to 1 over envelope and fuel-type
    dm_heating_mix.normalise(dim='Categories3')
    dm_heating_mix.rename_col('bld_heating-mix-tmp', 'bld_heating-mix', dim='Variables')

    return dm_heating_mix

def clean_heating_cat(dm_heating_cat, envelope_cat_new):

    years_all = dm_heating_cat.col_labels['Years']
    idx = dm_heating_cat.idx
    for cat, period in envelope_cat_new.items():
        years_period = list(range(period[0], period[1]+1))
        years_not_in_period = list(set(years_all) - set(years_period))
        idx_yrs = [idx[yr] for yr in years_not_in_period]
        dm_heating_cat.array[:, idx_yrs, idx['bld_heating-mix'], :, idx[cat], :] = np.nan
    dm_heating_cat.fill_nans('Years')

    return dm_heating_cat


def compute_building_age(dm_stock_cat, years_fts, first_bld_sfh, first_bld_mfh):
    dm_age = dm_stock_cat.filter({'Variables': ['bld_floor-area_stock']})
    dm_age.rename_col('bld_floor-area_stock', 'bld_age', dim='Variables')
    dm_age.change_unit('bld_age', 1, 'm2', 'years')
    dm_age.add(np.nan, dim='Years', col_label=years_fts, dummy=True)
    years_all = np.array(dm_age.col_labels['Years'])
    nb_cntr = len(dm_age.col_labels['Country'])
    idx = dm_age.idx
    for cat, start_yr in first_bld_sfh.items():
        arr_age = years_all - start_yr
        arr_age = np.maximum(arr_age, 0)
        for idx_c in range(nb_cntr):
            dm_age.array[idx_c, :, idx['bld_age'], idx['single-family-households'], idx[cat]] = arr_age
    for cat, start_yr in first_bld_mfh.items():
        arr_age = years_all - start_yr
        arr_age = np.maximum(arr_age, 0)
        for idx_c in range(nb_cntr):
            dm_age.array[idx_c, :, idx['bld_age'], idx['multi-family-households'], idx[cat]] = arr_age
    return dm_age

def compute_renovation_loi_energie(dm_stock_area, dm_num_bld, dm_stock_cat, env_cat_mfh, env_cat_sfh, DM_buildings):
    dm_num_bld.append(dm_stock_area, dim='Variables')
    dm_num_bld_sfh = dm_num_bld.filter({'Categories1': ['single-family-households']})
    dm_num_bld_sfh.groupby(env_cat_sfh, dim='Categories2', inplace=True)
    dm_num_bld_mfh = dm_num_bld.filter({'Categories1': ['multi-family-households']})
    dm_num_bld_mfh.groupby(env_cat_mfh, dim='Categories2', inplace=True)
    dm_bld = dm_num_bld_sfh
    dm_bld.append(dm_num_bld_mfh, dim='Categories1')
    dm_bld_adj = dm_stock_cat.filter({'Variables': ['bld_floor-area_stock']}, inplace=False)
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


def compute_heating_mix_F_E_D_categories(dm_heating_tech, dm_heating_tech_old):
    # For categories existing in before 2000 (in dm_heating_tech_old) merge with new data and normalise
    dm_heating_tech.switch_categories_order('Categories3', 'Categories1')
    dm_heating_tech.switch_categories_order('Categories2', 'Categories3')
    dm_heating_tech_old.switch_categories_order('Categories3', 'Categories1')
    dm_heating_tech_old.switch_categories_order('Categories2', 'Categories3')
    dm_tmp = dm_heating_tech.filter({'Categories1': dm_heating_tech_old.col_labels['Categories1']})
    dm_heating_tech_old.append(dm_tmp, dim='Years')
    dm_heating_tech_old.normalise(dim='Categories3')
    # Remove "D" values at 0 in 1990 and use fill_nans to fill
    idx = dm_heating_tech_old.idx
    dm_heating_tech_old.array[:, idx[1990], :, idx['D'], idx['multi-family-households'], :] = np.nan
    dm_heating_tech_old.fill_nans('Years')
    linear_fitting(dm_heating_tech_old, years_ots)
    dm_heating_tech_old.normalise('Categories3')
    return dm_heating_tech_old


def compute_heating_mix_C_B_categories(dm_heating_tech, cdm_heating_archetypes):

    dm_heating_tech_new = dm_heating_tech.filter({'Categories1': ['B', 'C']}, inplace=False)
    # In order to extrapolate C category for the missing years, since things change rapidely in 2021-2023,
    # I use the archetype paper to fix the values at the beginning of the construction period (this is a bit of a misuse)

    # normalise
    dm_heating_tech_new.normalise('Categories3')
    # add missing years as nan
    years_missing = list(set(years_ots) - set(dm_heating_tech_new.col_labels['Years']))
    dm_heating_tech_new.add(np.nan, dim='Years', dummy=True, col_label=years_missing)
    dm_heating_tech_new.sort('Years')
    # replace the values at the beginning of the construction period for C with the archetypes values
    extra_cat = list(set(dm_heating_tech_new.col_labels['Categories3']) - set(cdm_heating_archetypes.col_labels['Categories3']))
    dm_heating_tech_new.filter({'Categories3': cdm_heating_archetypes.col_labels['Categories3']}, inplace=True)
    cdm_heating_archetypes.sort('Categories3')
    dm_heating_tech_new.sort('Categories3')
    idx = dm_heating_tech_new.idx
    idx_c = cdm_heating_archetypes.idx
    for cat in dm_heating_tech_new.col_labels['Categories1']:
        start_year = envelope_cat_new[cat][0]
        dm_heating_tech_new.array[:, idx[start_year], 0, idx[cat], idx['multi-family-households'], :] \
            = cdm_heating_archetypes.array[np.newaxis, 0, idx['multi-family-households'], idx_c[cat], :]
        dm_heating_tech_new.array[:, idx[start_year], :, idx[cat], idx['single-family-households'], :] \
            = cdm_heating_archetypes.array[np.newaxis, 0, idx['single-family-households'], idx_c[cat], :]
    linear_fitting(dm_heating_tech_new, years_ots)

    idx = dm_heating_tech_new.idx
    for cat in dm_heating_tech_new.col_labels['Categories1']:
        period = envelope_cat_new[cat]
        start_year = period[0]
        before_period = [idx[yr] for yr in years_ots if yr < start_year]
        dm_heating_tech_new.array[:, before_period, :, idx[cat], :, :] = np.nan
    dm_heating_tech_new.normalise('Categories3')

    dm_heating_tech_new.add(0, dummy=True, dim='Categories3', col_label=extra_cat)
    return dm_heating_tech_new


def compute_heating_efficiency_by_archetype(dm_heating_eff, dm_stock_cat, envelope_cat_new, categories):

    arr_w_cat = np.repeat(dm_heating_eff.array[..., np.newaxis], repeats=len(categories), axis=-1)
    dm_eff_cat = DataMatrix.based_on(arr_w_cat, format=dm_heating_eff, change={'Categories2': categories},
                                     units=dm_heating_eff.units)
    dm_eff_cat_raw = dm_eff_cat.copy()
    # Keep only stock split by categories
    dm_stock = dm_stock_cat.group_all('Categories1', inplace=False)
    idx = dm_eff_cat.idx
    idx_s = dm_stock.idx
    for cat in categories:
        if cat not in envelope_cat_new.keys():
            dm_eff_cat.array[:, :, :, :, idx[cat]] = dm_eff_cat.array[:, 0, np.newaxis, :, :, idx[cat]]
        else:
            # Compute efficiency as the weighted average of stock efficiency and new efficiency
            start_yr = envelope_cat_new[cat][0]
            end_yr = envelope_cat_new[cat][1]
            for yr in range(start_yr+1, end_yr+1):
                eff_s_tm1 = dm_eff_cat.array[:, idx[yr-1], 0, :, idx[cat]]
                s_tm1 = dm_stock.array[:, idx_s[yr-1], idx_s['bld_floor-area_stock'], np.newaxis, idx_s[cat]]
                new_t = dm_stock.array[:, idx_s[yr], idx_s['bld_floor-area_new'],  np.newaxis, idx_s[cat]]
                eff_n_t = dm_eff_cat.array[:, idx[yr], 0, :, idx[cat]]
                eff_s_t = (eff_n_t * new_t + s_tm1 * eff_s_tm1)/(new_t + s_tm1)
                dm_eff_cat.array[:, idx[yr], 0, :, idx[cat]] = eff_s_t
            # after construction period, fix efficiency to end of construction period
            if (end_yr + 1) in dm_eff_cat.col_labels['Years']:
                dm_eff_cat.array[:, idx[end_yr]:, 0, :, idx[cat]] = dm_eff_cat.array[:, idx[end_yr], np.newaxis, 0, :, idx[cat]]
            if start_yr > dm_eff_cat.col_labels['Years'][0]:
                dm_eff_cat.array[:, 0:idx[start_yr], 0, :, idx[cat]] = np.nan

    # for heating-oil keep original data
    dm_eff_cat.array[:, :, :, idx['heating-oil'], :] = dm_eff_cat_raw.array[:, :, :, idx['heating-oil'], :]

    dm_eff_cat_raw.rename_col('bld_heating-efficiency', 'bld_heating-efficiency-JRC', dim='Variables')
    dm_eff_cat.append(dm_eff_cat_raw, dim='Variables')
    return dm_eff_cat


def extract_heating_efficiency(file, sheet_name, years_ots):
    df = pd.read_excel(file, sheet_name=sheet_name)
    df = df[0:13].copy()
    names_map = {'Ratio of energy service to energy consumption': 'remove', 'Space heating': 'other', 'Solids': 'coal',
                 'Liquified petroleum gas (LPG)': 'remove', 'Diesel oil': 'heating-oil', 'Natural gas': 'gas',
                 'Biomass': 'wood', 'Geothermal': 'geothermal', 'Distributed heat': 'district-heating',
                 'Advanced electric heating': 'heat-pump', 'Conventional electric heating': 'electricity',
                 'Electricity in circulation': 'remove'}
    dm_heating_eff = df_excel_to_dm(df, names_map, var_name='bld_heating-efficiency', unit='%', num_cat=1)
    dm_heating_eff.drop(col_label='remove', dim='Categories1')
    dm_heating_eff.add(0.8, dim='Categories1', col_label='solar', dummy=True)
    years_missing = list(set(years_ots) - set(dm_heating_eff.col_labels['Years']))
    years_missing_e = [yr for yr in years_missing if yr > dm_heating_eff.col_labels['Years'][0]]
    years_missing_i = [yr for yr in years_missing if yr < dm_heating_eff.col_labels['Years'][0]]
    linear_fitting(dm_heating_eff, years_missing_e, based_on=list(range(2015, dm_heating_eff.col_labels['Years'][-1]+1)))
    dm_heating_eff.add(np.nan, dim='Years', col_label=years_missing_i,  dummy=True)
    dm_heating_eff.sort('Years')
    dm_heating_eff.fill_nans('Years')
    # Add Vaud
    arr = dm_heating_eff.array
    dm_heating_eff.add(arr, dim='Country', col_label='Vaud')
    dm_heating_eff.sort('Country')
    return dm_heating_eff

# Stock
years_ots = create_years_list(1990, 2023, 1)
years_fts = create_years_list(2025, 2050, 5)

dict_lfs_ots, dict_lfs_fts = read_database_to_dm('lifestyles_population.csv', filter={'geoscale': ['Vaud', 'Switzerland']},
                                                 baseyear=years_ots[-1], num_cat=0, level='all')
dm_pop = dict_lfs_ots['pop'].filter({'Variables': ['lfs_population_total']}, inplace=False)
dm_pop.sort('Country')

filename = 'data/bld_household_size.pickle'
dm_lfs_household_size = extract_lfs_household_size(years_ots, table_id='px-x-0102020000_402', file=filename)

# SECTION Floor area Stock ots
construction_period_envelope_cat_sfh = {'F': ['Avant 1919', '1919-1945', '1946-1960', '1961-1970'],
                                        'E': ['1971-1980'],
                                        'D': ['1981-1990', '1991-2000'],
                                        'C': ['2001-2005', '2006-2010'],
                                        'B': ['2011-2015', '2016-2020', '2021-2023']}
construction_period_envelope_cat_mfh = {'F': ['Avant 1919', '1919-1945', '1946-1960', '1961-1970', '1971-1980'],
                                        'E': ['1981-1990'],
                                        'D': ['1991-2000'],
                                        'C': ['2001-2005', '2006-2010'],
                                        'B': ['2011-2015', '2016-2020', '2021-2023']}
envelope_cat_new = {'D': (1990, 2000), 'C': (2001, 2010), 'B': (2011, 2023)}

# Floor area stock
# Logements selon les niveaux géographiques institutionnels, la catégorie de bâtiment,
# la surface du logement et l'époque de construction
# https://www.pxweb.bfs.admin.ch/pxweb/fr/px-x-0902020200_103/-/px-x-0902020200_103.px/
table_id = 'px-x-0902020200_103'
file = 'data/bld_floor-area_stock.pickle'
#dm_bld_area_stock, dm_energy_cat = compute_bld_floor_area_stock_tranformed_avg_new_area(table_id, file,
#                                                                years_ots, construction_period_envelope_cat_sfh,
#                                                                construction_period_envelope_cat_mfh)
dm_stock_tot, dm_stock_cat, dm_avg_floor_area = compute_floor_area_stock_v2(table_id, file, dm_pop=dm_pop,
                                                         cat_map_sfh=construction_period_envelope_cat_sfh,
                                                         cat_map_mfh=construction_period_envelope_cat_mfh)

# SECTION Floor area New ots
# New residential buildings by sfh, mfh
# Nouveaux logements selon la grande région, le canton, la commune et le type de bâtiment, depuis 2013
table_id = 'px-x-0904030000_107'
file = 'data/bld_new_buidlings_2013_2023.pickle'
dm_bld_new_buildings_1 = extract_bld_new_buildings_1(table_id, file)

# Nouveaux logements selon le type de bâtiment, 1995-2012
table_id = 'px-x-0904030000_103'
file = 'data/bld_new_buildings_1995_2012.pickle'
dm_bld_new_buildings_2 = extract_bld_new_buildings_2(table_id, file)
# Floor-area new by sfh, mfh
dm_new_tot_raw = compute_bld_floor_area_new(dm_bld_new_buildings_1, dm_bld_new_buildings_2, dm_avg_floor_area, dm_pop)
del dm_bld_new_buildings_2, dm_bld_new_buildings_1

# SECTION Floor-area Waste + Recompute New
dm_waste_tot, dm_new_tot = compute_waste(dm_stock_tot, dm_new_tot_raw)
dm_waste_cat = compute_floor_area_waste_cat(dm_waste_tot)
# Floor-area new by sfh, mfh and envelope categories
dm_new_cat = compute_floor_area_new_cat(dm_new_tot, envelope_cat_new)

# dm_energy_cat has variables: bld_floor-area_stock_share, bld_floor-area_transformed, bld_floor-area_stock, bld_floor-area_new
#dm_energy_cat = compute_new_area_by_energy_cat(dm_bld_area_new, dm_energy_cat, envelope_cat_new)

# Empty apartments
# Logements vacants selon la grande région, le canton, la commune, le nombre de pièces d'habitation
# et le type de logement vacant
# https://www.pxweb.bfs.admin.ch/pxweb/fr/px-x-0902020300_101/-/px-x-0902020300_101.px/

# SECTION Floor area Renovated ots
# Number of buildings
# Bâtiments selon les niveaux géographiques institutionnels, la catégorie de bâtiment et l'époque de construction
table_id = 'px-x-0902010000_103'
file = 'data/bld_nb-buildings_2010_2022.pickle'
dm_bld = extract_number_of_buildings(table_id, file)

print('Maybe you should considered the buildings undergoing systemic renovation as well')
# Number of renovated-buildings (thermal insulation)
# https://www.newsd.admin.ch/newsd/message/attachments/82234.pdf
# "Programme bâtiments" rapports annuels 2014-2022, focus sur isolation thérmique
nb_buildings_isolated = {2022: 8148, 2021: 8400, 2020: 8050, 2019: 8500,
                          2018: 7500, 2017: 8100, 2016: 7900, 2014: 8303}
nb_buildings_systemic_renovation \
    = {2022: 2326, 2021: 2320, 2020: 2240, 2019: 1900, 2018: 1200,
       2017: 374, 2016: 0, 2014: 0}
nb_buildings_renovated = dict()
for yr in nb_buildings_isolated.keys():
    nb_buildings_renovated[yr] = nb_buildings_isolated[yr] + nb_buildings_systemic_renovation[yr]

# For 2014 - 2016 we assume VD share = VD share 2017
VD_share = {2014: 0.11, 2015: 0.11, 2016: 0.11, 2017: 0.110, 2018: 0.103,
            2019: 0.154, 2020: 0.16, 2021: 0.193, 2022: 0.15}
share_by_bld = {'single-family-households': 0.55, 'multi-family-households': 0.35, 'other': 0.1}
dm_renovation = compute_renovated_buildings(dm_bld, nb_buildings_renovated, VD_share, share_by_bld)

# Compute renovation-rate
dm_renovation = compute_renovation_rate(dm_renovation, years_ots)

# SECTION Renovation by envelope cat ots
# According to the Programme Batiments the assenissment is
# Amélioration de +1 classes CECB 57%
# Amélioration de +2 classes CECB 15%
# Amélioration de +3 classes CECB 15%
# Amélioration de +4 classes CECB 13%
ren_map_in = {(1990, 2000): {'F': 0, 'E': 0.85, 'D': 0.15, 'C': 0, 'B': 0},
               (2001, 2010): {'F': 0, 'E': 0.69, 'D': 0.16, 'C': 0.15, 'B': 0},
               (2011, 2023): {'F': 0, 'E': 0.46, 'D': 0.23, 'C': 0.16, 'B': 0.15}}
ren_map_out = {(1990, 2000): {'F': -0.8, 'E': -0.2, 'D': 0, 'C': 0, 'B': 0},
              (2001, 2010): {'F': -0.8, 'E': -0.2, 'D': 0, 'C': 0, 'B': 0},
              (2011, 2023): {'F': -0.8, 'E': -0.2, 'D': 0, 'C': 0, 'B': 0}}
dm_renov_distr = extract_renovation_redistribuition(ren_map_in, ren_map_out, years_ots)

# Harmonise floor-area stock, new and demolition rate
#dm_cat = compute_bld_demolition_rate(dm_energy_cat, envelope_cat_new)

# SECTION Floor-area Renovated by envelope cat
# r_ct (t) = Ren-disr_ct(t) ren-rate(t) s(t-1)
dm_renov_cat = compute_floor_area_renovated(dm_stock_tot, dm_renovation, dm_renov_distr)

# SECTION Stock by envelope cat
# s_{c,t}(t-1) &= s_{c,t}(t) - n_{c,t}(t) - r_{c,t}(t) + w_{c,t}(t)
dm_all = compute_stock_area_by_cat(dm_stock_cat, dm_new_cat, dm_renov_cat, dm_waste_cat, dm_stock_tot)

# Harmonise stock, new, waste, and renovation ots, demolition-rate
# The analysis before accounts for the stock, new, demolition-rate of building stock construction period.
# the equation I have been working with is: s(t) = s(t-1) + n(t) - w(t)
# Now I want to account for the renovation that redistributes the energy categories.
# The total equation does not change but for each energy class we have
# s_c(t) = s_c(t-1) + n_c(t) - w_c(t) + r_c(t), where r_c(t) can be positive or negative
# I want to assume that n_c, w_c and r_c are given as well as s_c(t), and I compute s_c(t-1).
# I will need to re-compute the demolition rate
#dm_all = harmonise_stock_new_renovated_transformed(dm_energy_cat, dm_renovation, dm_renov_distr, envelope_cat_new)

# SECTION U-values - constant
# Definition of Building Archetypes Based on the Swiss Energy Performance Certificates Database
# by Alessandro Pongelli et al.
# U-value is computed as the average of the house element u-value (roof, wall, windows, ..) weighted by their area
# U-value in: W/m^2 K
# Single-family-households
envelope_cat_u_value = {'single-family-households':
                            {'F': 0.82, 'E': 0.69, 'D': 0.53, 'C': 0.41, 'B': 0.25},
                        'multi-family-households':
                            {'F': 0.93, 'E': 0.70, 'D': 0.63, 'C': 0.48, 'B': 0.29}}
cdm_u_value = ConstantDataMatrix(col_labels={'Variables': ['bld_u-value'],
                                             'Categories1': ['multi-family-households', 'single-family-households'],
                                             'Categories2': ['B', 'C', 'D', 'E', 'F']},
                                 units={'bld_u-value': 'W/m2K'})
arr = np.zeros((len(cdm_u_value.col_labels['Variables']), len(cdm_u_value.col_labels['Categories1']),
                len(cdm_u_value.col_labels['Categories2'])))
cdm_u_value.array = arr
idx = cdm_u_value.idx
for bld, dict_val in envelope_cat_u_value.items():
    for cat, val in dict_val.items():
        cdm_u_value.array[idx['bld_u-value'], idx[bld], idx[cat]] = val

# SECTION Surface to Floorarea factor - constant
# From the same dataset we obtain also the floor to surface area
surface_to_floorarea = {'single-family-households': 2.0, 'multi-family-households': 1.3}
cdm_s2f = ConstantDataMatrix(col_labels={'Variables': ['bld_surface-to-floorarea'],
                                        'Categories1': ['multi-family-households', 'single-family-households']})
arr = np.zeros((len(cdm_s2f.col_labels['Variables']), len(cdm_s2f.col_labels['Categories1'])))
cdm_s2f.array = arr
idx = cdm_s2f.idx
for cat, val in surface_to_floorarea.items():
    cdm_s2f.array[idx['bld_surface-to-floorarea'], idx[cat]] = val

# 2018: Type of renovation 4% windows, 51% roof, 38% facade, 2% floor, 5% other
# shallow: 11% (windows, floor, other) -> uvalue improvement 15%
# medium; 51% (roof) -> uvalue improvement 41%
# deep: 38% (facade) -> uvalue improvement 66%



#dm_energy.datamatrix_plot()
# if I want to save 2.2 TWh, and the maximum improvement I can do is 150 kWh/m2, I need to renovate 15 million m2.
# The current park is 137 + 337 = 474 million m2
# If I want to fix the renovation at 1%, then you renovate 4.7 million m2, in order to save 2.2 TWh,
# you need to save: 470 kWh/m2

file = '../../../data/datamatrix/lifestyles.pickle'
with open(file, 'rb') as handle:
    DM_lifestyles_old = pickle.load(handle)

DM_buildings = {'ots': dict(), 'fts': dict(), 'fxa': dict(), 'constant': dict()}


file = '../../../data/datamatrix/buildings.pickle'
with open(file, 'rb') as handle:
    DM_bld = pickle.load(handle)


# SECTION: Heating technology
##########   HEATING TECHNOLOGY     #########
# You need to extract the heating technology (you only have the last 3 years
# but you have the energy mix for the historical period)
# https://www.pxweb.bfs.admin.ch/pxweb/fr/px-x-0902010000_102/-/px-x-0902010000_102.px/
# In order to check the result the things I can validate are the 1990, 2000 value and the 2021-2023 values
# You can run the check to see if the allocation by envelope category is well done and matches with the original data
# The problem is that at the end the energy demand decreases.

table_id = 'px-x-0902010000_102'
file = 'data/bld_heating_technology.pickle'
dm_heating_tech = extract_heating_technologies(table_id, file, construction_period_envelope_cat_sfh, construction_period_envelope_cat_mfh)
if 'gaz' in dm_heating_tech.col_labels['Categories2']:
    dm_heating_tech.rename_col('gaz', 'gas', 'Categories2')
dm_heating_tech.add(0, dummy=True, dim='Categories2', col_label='coal')

table_id = 'px-x-0902020100_112'
file = 'data/bld_heating_technology_1990-2000.pickle'
dm_heating_tech_old = extract_heating_technologies_old(table_id, file, construction_period_envelope_cat_sfh, construction_period_envelope_cat_mfh)

# Heating categories from Archetypes paper for B and C categories
cdm_heating_archetypes = prepare_heating_mix_by_archetype()

# Reconstruct heating-mix for older heating categories
dm_heating_tech_old = compute_heating_mix_F_E_D_categories(dm_heating_tech, dm_heating_tech_old)
# Reconstruct heating-mix for new categories using archetypes for B and C
# !FIXME Vaud behave differently than Switzerland
dm_heating_tech_new = compute_heating_mix_C_B_categories(dm_heating_tech, cdm_heating_archetypes)

# Merge old and new heating tech
dm_heating_cat = dm_heating_tech_old.copy()
dm_heating_cat.append(dm_heating_tech_new, dim='Categories1')
dm_heating_cat.switch_categories_order('Categories1', 'Categories2')

dm_heating_cat.sort('Categories2')

alternative_computation = False
if alternative_computation:
    # SECTION: Heating technology according to archetypes
    # Use Archetype paper to extract heating mix by archetype
    cdm_heating_archetypes = prepare_heating_mix_by_archetype()

    dm_heating_cat = compute_heating_mix_by_category(dm_heating_tech, cdm_heating_archetypes, dm_all)
    # Before and after construction period keep shares flat
    dm_heating_cat = clean_heating_cat(dm_heating_cat, envelope_cat_new)

    for cat in dm_heating_cat.col_labels['Categories2']:
        dm_heating_cat.filter(
            {'Categories1': ['multi-family-households'], 'Categories2': [cat]}).flatten().flatten().datamatrix_plot(
            {'Country': ['Switzerland']}, stacked=True)

# CHECK:
check = False
if check:
    arr_tmp = dm_heating_cat.array[:, :, 0, ...] * dm_stock_cat.array[:, :, -1, :, :, np.newaxis]
    dm_heating_cat.add(arr_tmp, dim='Variables', unit='m2', col_label='bld_heating-mix-area')
    dm_heating_new = dm_heating_cat.filter({'Variables': ['bld_heating-mix-area']})
    dm_heating_new.group_all('Categories2')
    dm_heating_new.normalise('Categories2')

# SECTION: Heating efficiency
#######      HEATING EFFICIENCY     ###########
file = '../Europe/data/JRC-IDEES-2021_Residential_EU27.xlsx'
sheet_name = 'RES_hh_eff'
dm_heating_eff = extract_heating_efficiency(file, sheet_name, years_ots)
dm_heating_eff_cat = compute_heating_efficiency_by_archetype(dm_heating_eff, dm_stock_cat, envelope_cat_new,
                                                             categories=dm_stock_cat.col_labels['Categories2'])

# SECTION: Lifestyles to Buildings intereface
#########################################
#####   INTERFACE: LFS to BLD     #######
#########################################
file = '../../../data/datamatrix/lifestyles.pickle'
with open(file, 'rb') as handle:
    DM_lifestyles = pickle.load(handle)

dm_pop_ots = DM_lifestyles['ots']['pop']['lfs_population_'].copy()
dm_pop_fts = DM_lifestyles['fts']['pop']['lfs_population_'][1]
dm_pop_ots.append(dm_pop_fts, dim='Years')
DM_interface_lfs_to_bld = {'pop': dm_pop_ots}


file = '../../../data/interface/lifestyles_to_buildings.pickle'
my_pickle_dump(DM_new = DM_interface_lfs_to_bld, local_pickle_file=file)

# SECTION: Climate to Buildings intereface
#########################################
#####   INTERFACE: CLM to BLD     #######
#########################################
file = '../../../data/datamatrix/climate.pickle'
with open(file, 'rb') as handle:
    DM_clm = pickle.load(handle)

dm_clm_ots = DM_clm['ots']['temp']['bld_climate-impact-space'].copy()
dm_clm_fts = DM_clm['fts']['temp']['bld_climate-impact-space'][1]
dm_clm_ots.append(dm_clm_fts, dim='Years')
DM_interface_clm_to_bld = {'cdd-hdd': dm_clm_ots}

file = '../../../data/interface/climate_to_buildings.pickle'
my_pickle_dump(DM_new=DM_interface_clm_to_bld, local_pickle_file=file)

# SECTION: FTS + PREPARE OUTPUT

# SECTION: Calibration from existing buildings.pickle
DM_buildings['fxa']['heating-energy-calibration'] = DM_bld['fxa']['heating-energy-calibration']

# SECTION: Floor intensity
#########################################
#####  FLOOR INTENSITY - SPACE/CAP  #####
#########################################
dm_space_cap = recompute_floor_area_per_capita(dm_all, dm_pop)
dm_space_cap.append(dm_lfs_household_size, dim='Variables')
DM_buildings['ots']['floor-intensity'] = dm_space_cap.copy()
linear_fitting(dm_space_cap, years_fts)
DM_buildings['fts']['floor-intensity'] = dict()
for lev in range(4):
    lev = lev + 1
    DM_buildings['fts']['floor-intensity'][lev] = dm_space_cap.filter({'Years': years_fts})

# SECTION: Heating-cooling behaviour (Temperature)
#########################################
#####   HEATING-COOLING BEHAVIOUR   #####
#########################################
col_label = {'Country': dm_stock_tot.col_labels['Country'],
             'Years': years_ots+years_fts,
             'Variables': ['bld_Tint-heating', 'bld_Tint-cooling'],
             'Categories1': dm_stock_tot.col_labels['Categories1'],
             'Categories2': dm_stock_cat.col_labels['Categories2']}
dm_Tint_heat = DataMatrix(col_labels=col_label, units={'bld_Tint-heating': 'C', 'bld_Tint-cooling': 'C'})
arr_shape = (len(dm_Tint_heat.col_labels['Country']), len(years_ots+years_fts),
             len(dm_Tint_heat.col_labels['Variables']), len(dm_Tint_heat.col_labels['Categories1']),
             len(dm_stock_cat.col_labels['Categories2']))
dm_Tint_heat.array = 20*np.ones(arr_shape)
idx = dm_Tint_heat.idx
cat_Tint = {'F': 19, 'E': 20, 'D': 21, 'C': 22, 'B': 23}
for cat, tint in cat_Tint.items():
    dm_Tint_heat.array[:, :, idx['bld_Tint-heating'], idx['multi-family-households'], idx[cat]] = tint
    dm_Tint_heat.array[:, :, idx['bld_Tint-heating'], idx['single-family-households'], idx[cat]] = tint - 1
DM_buildings['ots']['heatcool-behaviour'] = dm_Tint_heat.filter({'Years': years_ots})
DM_buildings['fts']['heatcool-behaviour'] = dict()
for lev in range(4):
    lev = lev + 1
    DM_buildings['fts']['heatcool-behaviour'][lev] = dm_Tint_heat.filter({'Years': years_fts})

# SECTION: Building-mix for stock and new (by sfh, mfh and energy cat)
#########################################
#####        BUILDING MIX          ######
#########################################
# Used to go from m2/cap to m2 of floor area
# building-mix_stock -> fxa, building-mix_new -> fts
dm_building_mix = compute_building_mix(dm_all)
dm_building_mix.add(np.nan, dummy=True, dim='Years', col_label=years_fts)
DM_buildings['fxa']['bld_type'] = dm_building_mix.filter({'Variables': ['bld_building-mix_stock']})
DM_buildings['ots']['building-renovation-rate'] = dict()
dm_tmp = dm_building_mix.filter({'Variables': ['bld_building-mix_new']})
DM_buildings['ots']['building-renovation-rate']['bld_building-mix'] = dm_tmp.filter({'Years': years_ots})
# FTS
dm_tmp.fill_nans(dim_to_interp='Years')
DM_buildings['fts']['building-renovation-rate'] = dict()
DM_buildings['fts']['building-renovation-rate']['bld_building-mix'] = dict()
for lev in range(4):
    lev = lev + 1
    DM_buildings['fts']['building-renovation-rate']['bld_building-mix'][lev] = dm_tmp.filter({'Years': years_fts})

# SECTION: Renovation rate fts
#########################################
#####         RENOVATION           ######
#########################################

dm_rr = dm_renovation.filter({'Variables': ['bld_renovation-rate']})
DM_buildings['ots']['building-renovation-rate']['bld_renovation-rate'] = dm_rr.copy()
# FTS
DM_buildings['fts']['building-renovation-rate']['bld_renovation-rate'] = dict()
dm_rr.add(np.nan, dim='Years', dummy=True, col_label=years_fts)
dm_rr.fill_nans(dim_to_interp='Years')
for lev in range(4):
    lev = lev + 1
    DM_buildings['fts']['building-renovation-rate']['bld_renovation-rate'][lev] = dm_rr.filter({'Years': years_fts})
##
DM_buildings['ots']['building-renovation-rate']['bld_renovation-redistribution'] = dm_renov_distr.copy()
# FTS
DM_buildings['fts']['building-renovation-rate']['bld_renovation-redistribution'] = dict()
dm_renov_distr.add(np.nan, dim='Years', dummy=True, col_label=years_fts)
dm_renov_distr.fill_nans(dim_to_interp='Years')
for lev in range(4):
    lev = lev + 1
    DM_buildings['fts']['building-renovation-rate']['bld_renovation-redistribution'][lev] = \
        dm_renov_distr.filter({'Years': years_fts})

# SECTION: Loi Energie - Renovation fts
# LEVEL 2 Vaud: Loi Energie + Plan Climat
# According to the Loi Energie, buildings in categories F,G > 750 m2 will have to be renovated before 2035,
# and the other F,G before 2040. They estimate this corresponds to 90'000 multi-family-households being renovated before 2035.
table_id = 'px-x-0902020200_103'
file = 'data/bld_floor-area_stock.pickle'
dm_stock_area, dm_num_bld = extract_stock_floor_area(table_id, file)
env_cat_mfh = construction_period_envelope_cat_mfh
env_cat_sfh = construction_period_envelope_cat_sfh
dm_rr_fts_2 = compute_renovation_loi_energie(dm_stock_area, dm_num_bld, dm_stock_cat, env_cat_mfh, env_cat_sfh, DM_buildings)
DM_buildings['fts']['building-renovation-rate']['bld_renovation-rate'][2] = dm_rr_fts_2

# SECTION: Demolition rate fts
#########################################
#####         DEMOLITION          #######
#########################################
# Compute demolition rate by bld type
dm_tot = dm_stock_tot.copy()
dm_tot.append(dm_waste_tot, dim='Variables')
dm_tot.operation('bld_floor-area_waste', '/', 'bld_floor-area_stock', out_col='bld_demolition-rate', unit='%')
dm_demolition_rate = dm_tot.filter({'Variables': ['bld_demolition-rate']})
DM_buildings['ots']['building-renovation-rate']['bld_demolition-rate'] = dm_demolition_rate.copy()
# Compute average demolition rate in the last 10 years and forecast to future
idx = dm_demolition_rate.idx
idx_yrs = [idx[yr] for yr in create_years_list(2012, 2023, 1)]
val_mean = np.mean(dm_demolition_rate.array[:, idx_yrs, ...], axis=1)
dm_demolition_rate.add(np.nan, dim='Years', dummy=True, col_label=years_fts)
for yr in years_fts:
    dm_demolition_rate.array[:, idx[yr], ...] = val_mean
# FTS
DM_buildings['fts']['building-renovation-rate']['bld_demolition-rate'] = dict()
for lev in range(4):
    lev = lev + 1
    DM_buildings['fts']['building-renovation-rate']['bld_demolition-rate'][lev] = \
        dm_demolition_rate.filter({'Years': years_fts})

# Create a bld age matrix to be used with demolition-rate
first_bld_sfh = {'F': 1900, 'E': 1971, 'D': 1981, 'C': 2001, 'B': 2011}
first_bld_mfh = {'F': 1900, 'E': 1981, 'D': 1991, 'C': 2001, 'B': 2011}
dm_age = compute_building_age(dm_stock_cat, years_fts, first_bld_sfh, first_bld_mfh)
DM_buildings['fxa']['bld_age'] = dm_age

#########################################
#####          U-VALUE            #######
#########################################
DM_buildings['constant']['u-value'] = cdm_u_value

#########################################
#####       SURFACE-2-FLOOR       #######
#########################################
DM_buildings['constant']['surface-to-floorarea'] = cdm_s2f


# SECTION: Heating technology mix fts
###########################################
#####    HEATING TECHNOLOGY MIX     #######
###########################################
DM_buildings['ots']['heating-technology-fuel'] = dict()
dm_heating_cat.sort('Categories3')
DM_buildings['ots']['heating-technology-fuel']['bld_heating-technology'] = dm_heating_cat.copy()
dm_heating_cat.add(np.nan, dim='Years', dummy=True, col_label=years_fts)
dm_heating_cat.fill_nans('Years')
dm_heating_cat_fts = dm_heating_cat.filter({'Years': years_fts}, inplace=False)
#dm_heating_cat_fts = compute_heating_tech_mix_fts(dm_heating_cat)
DM_buildings['fts']['heating-technology-fuel'] = dict()
DM_buildings['fts']['heating-technology-fuel']['bld_heating-technology'] = dict()
for lev in range(4):
    lev = lev + 1
    DM_buildings['fts']['heating-technology-fuel']['bld_heating-technology'][lev] = dm_heating_cat_fts


# SECTION: Loi energy - Heating tech
# Plus de gaz, mazout, charbon dans les prochain 15-20 ans. Pas de gaz, mazout, charbon dans les nouvelles constructions
dm_heating_cat_fts_2 = DM_buildings['fts']['heating-technology-fuel']['bld_heating-technology'][2].copy()
idx = dm_heating_cat_fts_2.idx
idx_fossil = [idx['coal'], idx['heating-oil'], idx['gas'], idx['electricity']]
dm_heating_cat_fts_2.array[idx['Vaud'], :, idx['bld_heating-mix'], :, idx['B'], idx_fossil] = 0
dm_heating_cat_fts_2.array[idx['Vaud'], 1:idx[2045], idx['bld_heating-mix'], :, :, idx_fossil] = np.nan
dm_heating_cat_fts_2.array[idx['Vaud'], idx[2045]:, idx['bld_heating-mix'], :, :, idx_fossil] = 0
dm_heating_cat_fts_2.fill_nans('Years')
dm_heating_cat_fts_2.normalise('Categories3')
DM_buildings['fts']['heating-technology-fuel']['bld_heating-technology'][2] = dm_heating_cat_fts_2

# SECTION: Heating efficiency fts
############################################
######       HEATING EFFICIENCY       ######
############################################
efficiency_by_categories = True
if efficiency_by_categories:
    dm_heating_eff = dm_heating_eff_cat
    dm_heating_eff.sort('Categories2')
dm_heating_eff.filter({'Categories1': dm_heating_cat.col_labels['Categories3']}, inplace=True)
dm_heating_eff.sort('Categories1')
dm_heating_eff_fts = calculate_heating_eff_fts(dm_heating_eff.copy(), years_fts, maximum_eff=0.98)
dm_heating_eff.switch_categories_order()
dm_heating_eff_fts.switch_categories_order()
DM_buildings['ots']['heating-efficiency'] = dm_heating_eff.copy()
DM_buildings['fts']['heating-efficiency'] = dict()
for lev in range(4):
    lev = lev + 1
    DM_buildings['fts']['heating-efficiency'][lev] = dm_heating_eff_fts


# SECTION: Emission factors
####################################
#####     EMISSION FACTORS    ######
####################################
# Obtained dividing emission by energy demand in file file = '../Europe/data/JRC-IDEES-2021_Residential_EU27.xlsx'
JRC_emissions_fact = {'coal': 350, 'heating-oil': 267, 'gas': 200, 'wood': 0, 'solar': 0}
cdm_emission_fact = ConstantDataMatrix(col_labels={'Variables': ['bld_CO2-factors'],
                                                   'Categories1': ['coal', 'heating-oil', 'gas', 'wood', 'solar']},
                                       units={'bld_CO2-factors': 'kt/TWh'})
cdm_emission_fact.array = np.zeros((len(cdm_emission_fact.col_labels['Variables']),
                                    len(cdm_emission_fact.col_labels['Categories1'])))
idx = cdm_emission_fact.idx
for key, value in JRC_emissions_fact.items():
    cdm_emission_fact.array[0, idx[key]] = value

cdm_emission_fact.sort('Categories1')
DM_buildings['constant']['emissions'] = cdm_emission_fact


# SECTION: Electricity emission factors
col_dict = {
    'Country': ['Switzerland', 'Vaud'],
    'Years': years_ots+years_fts,
    'Variables': ['bld_CO2-factor'],
    'Categories1': ['electricity']
}
dm_elec = DataMatrix(col_labels=col_dict, units={'bld_CO2-factor': 'kt/TWh'})

arr_elec = np.zeros((2, 40, 1, 1))
idx = dm_elec.idx
arr_elec[:, idx[1990]: idx[2023]+1, 0, 0] = 112
arr_elec[:, idx[2025]: idx[2050], 0, 0] = np.nan
arr_elec[:, idx[2050], 0, 0] = 0
dm_elec.array = arr_elec
dm_elec.fill_nans(dim_to_interp="Years")
DM_buildings['fxa']['emission-factor-electricity'] = dm_elec

file = '../../../data/datamatrix/buildings.pickle'
my_pickle_dump(DM_buildings, file)

