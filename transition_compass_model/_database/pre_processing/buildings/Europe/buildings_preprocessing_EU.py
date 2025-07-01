
from model.common.data_matrix_class import DataMatrix
from model.common.constant_data_matrix_class import ConstantDataMatrix
from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat
from model.common.auxiliary_functions import linear_fitting, moving_average, create_years_list, eurostat_iso2_dict, my_pickle_dump, cdm_to_dm

import pickle
import numpy as np
import pandas as pd

import warnings
warnings.simplefilter("ignore")
import plotly.io as pio
pio.renderers.default='browser'

#################################################################
########################### FUNCTIONS ###########################
#################################################################

EU27_cntr_list = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland',
                  'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg',
                  'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden']

def sub_routine_get_uvalue_by_element():
    # Load u-values by element, construction-period, building type
    file = 'U_value_Europe.xlsx'
    data_path = 'data/'
    rows_to_skip = [1, 191, 192]
    df_u_wall = pd.read_excel(data_path + file, sheet_name='U-value - wall', skiprows=rows_to_skip)
    df_u_window = pd.read_excel(data_path + file, sheet_name='U-value - window', skiprows=rows_to_skip)
    df_u_roof = pd.read_excel(data_path + file, sheet_name='U-value - roof', skiprows=rows_to_skip)
    df_u_ground = pd.read_excel(data_path + file, sheet_name='U-value - ground floor', skiprows=rows_to_skip)
    # Rename dictionary
    dict_ren = {'Building use': 'Country', 'Unnamed: 1': 'Construction period'}
    df_u_wall.rename(dict_ren, axis=1, inplace=True)
    df_u_window.rename(dict_ren, axis=1, inplace=True)
    df_u_roof.rename(dict_ren, axis=1, inplace=True)
    df_u_ground.rename(dict_ren, axis=1, inplace=True)
    # Remove useless cols
    drop_cols = ['All uses']
    df_u_wall.drop(drop_cols, axis=1, inplace=True)
    df_u_window.drop(drop_cols, axis=1, inplace=True)
    df_u_roof.drop(drop_cols, axis=1, inplace=True)
    df_u_ground.drop(drop_cols, axis=1, inplace=True)
    return df_u_wall, df_u_window, df_u_roof, df_u_ground


def compute_weighted_u_value(df_u_wall, df_u_window, df_u_roof, df_u_ground):
    # Load u-values by element, construction-period, building type
    file_weight = 'Uvalues_literature.xlsx'
    data_path = 'data/'
    df_u_weight = pd.read_excel(data_path + file_weight, sheet_name='weight_element')
    df_u_weight.set_index('Element', inplace=True)

    # Weight u-values by area of element for single-family-households
    w_sfh = df_u_weight['bld_area_weight_sfh[%]']
    sfh_col = 'Single-family buildings'
    df_u_wall[sfh_col] = w_sfh['Facade'] * df_u_wall[sfh_col]
    df_u_roof[sfh_col] = w_sfh['Roof'] * df_u_roof[sfh_col]
    df_u_window[sfh_col] = w_sfh['Windows'] * df_u_roof[sfh_col]
    df_u_ground[sfh_col] = w_sfh['Cellar'] * df_u_ground[sfh_col]

    # Weight u-values by area of element for other buildings
    # We apply the multi-family house area weight to all buildings except sfh
    others_col = [col for col in df_u_wall.columns if col not in
                  ['Country', 'Construction period', 'Single-family buildings']]
    w_mfh = df_u_weight['bld_area_weight_mfh[%]']
    for col in others_col:
        df_u_wall[col] = w_mfh['Facade'] * df_u_wall[col]
        df_u_roof[col] = w_mfh['Roof'] * df_u_roof[sfh_col]
        df_u_window[col] = w_mfh['Windows'] * df_u_roof[sfh_col]
        df_u_ground[col] = w_mfh['Cellar'] * df_u_ground[sfh_col]

    # Compute uvalue as weighted average of element u-value
    df_uvalue = df_u_wall
    u_cols = [col for col in df_u_wall.columns if col not in ['Country', 'Construction period']]
    for col in u_cols:
        df_uvalue[col] = df_uvalue[col] + df_u_roof[col] + df_u_window[col] + df_u_ground[col]
    return df_uvalue


def extract_u_value(df_uvalue):

    # To obtain the u-value of new buildings replace the construction period with the middle year
    df_uvalue[['start_y', 'end_y']] = df_uvalue['Construction period'].str.split('-', expand=True)
    # Replace now with 2020
    df_uvalue['end_y'] = df_uvalue['end_y'].str.replace('now', '2020')
    df_uvalue[['start_y', 'end_y']] = df_uvalue[['start_y', 'end_y']].astype(int)
    df_uvalue['Years'] = ((df_uvalue['end_y'] + df_uvalue['start_y'])/2).astype(int)
    df_uvalue.loc[df_uvalue["end_y"] == 1945,"Years"] = 1900 # TODO: check this modif with Paola for CH
    df_uvalue.drop(['start_y', 'end_y'], axis=1, inplace=True)

    # df_uvalue
    df_uvalue.drop(['Construction period'], axis=1, inplace=True)
    # add unit
    df_uvalue = df_uvalue.add_suffix('[W/(m2K)]')
    df_uvalue = df_uvalue.add_prefix('bld_uvalue_')
    df_uvalue.rename({'bld_uvalue_Country[W/(m2K)]': 'Country',
                      'bld_uvalue_Years[W/(m2K)]': 'Years'}, axis=1, inplace=True)

    return df_uvalue

def extract_floor_area_stock():
    
    file = 'BSO_floor_area_2020.xlsx'
    data_path = 'data/'
    rows_to_skip = [1, 198, 199]
    df_area = pd.read_excel(data_path + file, sheet_name='Export', skiprows=rows_to_skip)
    dict_ren = {'Building use': 'Construction period', 'Unnamed: 1': 'Country'}
    df_area.rename(dict_ren, axis=1, inplace=True)
    
    # To obtain the u-value of new buildings replace the construction period with the middle year
    df_area[['start_y', 'end_y']] = df_area['Construction period'].str.split('-', expand=True)
    
    # Replace now with 2020
    df_area['end_y'] = df_area['end_y'].str.replace('now', '2020')
    df_area[['start_y', 'end_y']] = df_area[['start_y', 'end_y']].astype(int)
    df_area['Years'] = ((df_area['end_y'] + df_area['start_y'])/2).astype(int)
    df_area.loc[df_area["end_y"] == 1945,"Years"] = 1900 # TODO: check this modif with Paola for CH
    df_area_new = df_area.loc[df_area['Years']>=1990].copy()
    df_area.drop(['start_y', 'end_y'], axis=1, inplace=True)

    # Keep only buildings after 1990
    df_area.drop(['Construction period'], axis=1, inplace=True)
    
    # add unit
    df_area = df_area.add_suffix('[m2]')
    df_area = df_area.add_prefix('bld_floor-area_')
    df_area.rename({'bld_floor-area_Country[m2]': 'Country',
                        'bld_floor-area_Years[m2]': 'Years'}, axis=1, inplace=True)
    dm_area = DataMatrix.create_from_df(df_area, num_cat=1)

    # Compute the average yearly new floor-area constructed as the floor-area/construction period lenght
    df_area_new['period_length'] = df_area_new['end_y'] - df_area_new['start_y']
    df_area_new.drop(['start_y', 'end_y', 'Construction period'], axis=1, inplace=True)
    value_cols = set(df_area_new.columns) - {'Country', 'Years', 'period_length'}
    for col in value_cols:
        df_area_new[col] = df_area_new[col]/df_area_new['period_length']
    df_area_new.drop(['period_length'], axis=1, inplace=True)
    # add unit
    df_area_new = df_area_new.add_suffix('[m2]')
    df_area_new = df_area_new.add_prefix('bld_floor-area_new_')
    df_area_new.rename({'bld_floor-area_new_Country[m2]': 'Country',
                        'bld_floor-area_new_Years[m2]': 'Years'}, axis=1, inplace=True)
    dm_area_new = DataMatrix.create_from_df(df_area_new, num_cat=1)

    return dm_area, dm_area_new

def get_uvalue_new_stock0(years_ots):
    # Gets the u-value for the new buildings, as well as the u-value of the building stock at t=baseyear
    
    # Load u-values by element, construction-period, building type
    df_u_wall, df_u_window, df_u_roof, df_u_ground = sub_routine_get_uvalue_by_element()
    
    # Weight u-values of element by area of element
    df_uvalue = compute_weighted_u_value(df_u_wall, df_u_window, df_u_roof, df_u_ground)
    
    # From df_uvalue keep only new built for construction period > 1990
    df_uvalue = extract_u_value(df_uvalue)
    # TODO: there is no selection here > 1900, see with Paola
    
    # From df to dm
    dm_uvalue = DataMatrix.create_from_df(df_uvalue, num_cat=1)
    
    # Extract floor-area of building stock
    dm_area, dm_area_new = extract_floor_area_stock()
    dm_area.drop(dim='Country', col_label = 'EU27')
    
    # Get multi-family household value as weighted average of 'Apartment buildings', 'Multi-family buildings'
    dm_uvalue.append(dm_area, dim='Variables')
    dm_uvalue.operation('bld_uvalue', '*', 'bld_floor-area', out_col='bld_uxarea', unit='m2')
    dm_uvalue.groupby({'multi-family-households': ['Apartment buildings', 'Multi-family buildings']}, dim='Categories1', inplace=True)
    idx = dm_uvalue.idx
    dm_uvalue.array[:, :, idx['bld_uvalue'], idx['multi-family-households']] = \
        dm_uvalue.array[:, :, idx['bld_uxarea'], idx['multi-family-households']] /\
        dm_uvalue.array[:, :, idx['bld_floor-area'], idx['multi-family-households']]
    
    # Rename using Calculator names:
    cols_in = ['Educational buildings', 'Health buildings', 'Hotels and Restaurants', 'Offices',
               'Other non-residential buildings', 'Trade buildings', 'Single-family buildings']
    cols_out = ['education', 'health', 'hotels', 'offices', 'other', 'trade', 'single-family-households']
    dm_uvalue.rename_col(cols_in, cols_out, dim='Categories1')

    # Get right categories for new floor area
    dm_area_new.groupby({'multi-family-households': ['Apartment buildings', 'Multi-family buildings']},
                        dim='Categories1', inplace=True)
    dm_area_new.rename_col(cols_in, cols_out, dim='Categories1')

    # Compute dm_uvalue for initial stock
    dm_uvalue_stock0 = dm_uvalue.filter({'Years': [1900, 1957, 1974, 1984]})
    dm_uvalue_stock0.groupby({1990: '.*'}, dim='Years', regex=True, inplace=True)
    dm_uvalue_stock0.array[:, :, idx['bld_uvalue'], :] = dm_uvalue_stock0.array[:, :, idx['bld_uxarea'], :]\
                                                         / dm_uvalue_stock0.array[:, :, idx['bld_floor-area'], :]
    dm_uvalue_stock0.filter({'Variables': ['bld_uvalue']}, inplace=True)

    # Extract dm_uvalue new
    dm_uvalue_new = dm_uvalue.filter({'Years': [1994, 2005, 2015]})
    dm_uvalue_new.filter({'Variables': ['bld_uvalue']}, inplace=True)
    dm_uvalue_new.rename_col('bld_uvalue', 'bld_uvalue_new', dim='Variables')
    
    # Linear fitting for missing years
    idx = dm_uvalue_stock0.idx
    max_start = dm_uvalue_stock0.array[:, 0, idx['bld_uvalue'], np.newaxis, :]
    min_end = np.min(dm_uvalue_new.array)*np.ones(shape=max_start.shape)
    linear_fitting(dm_uvalue_new, years_ots, max_t0=max_start, min_tb=min_end)

    # Compute share of floor area by building type, to determine floor-area stock for non-residential buildings
    dm_area_2020 = dm_uvalue.filter({'Variables': ['bld_floor-area']}, inplace=False)
    dm_area_2020.groupby({2020: '.*'}, dim='Years', regex=True, inplace=True)

    dm_uvalue_stock0.rename_col('Czechia', 'Czech Republic', dim='Country')
    dm_area_2020.rename_col('Czechia', 'Czech Republic', dim='Country')
    dm_area_new.rename_col('Czechia', 'Czech Republic', dim='Country')
    dm_uvalue_new.rename_col('Czechia', 'Czech Republic', dim='Country')

    # Compute EU data for area in 2020
    dm_area_2020_EU27 = dm_area_2020.groupby({'EU27': EU27_cntr_list}, dim='Country', inplace=False)
    dm_area_2020.append(dm_area_2020_EU27, dim='Country')

    return dm_uvalue_new, dm_area_2020, dm_uvalue_stock0, dm_area_new

def get_rooms_cap_eustat(dict_iso2, years_ots):

    # Extracts the number of rooms per capita for the period available (2003-2023)
    # The data are extrapolated with linear fitting until 1990
    ##### Extract rooms per capita
    filter = {'geo\\TIME_PERIOD': list(dict_iso2.keys()),
              'building': ['HOUSE', 'FLAT'],
              'tenure': 'TOTAL'}
    mapping_dim = {'Country': 'geo\\TIME_PERIOD',
                   'Variables': 'tenure',
                   'Categories1': 'building'}
    dm_rooms = get_data_api_eurostat('ilc_lvho03', filter, mapping_dim, 'rooms/cap', years_ots)

    dm_rooms.rename_col(['FLAT', 'HOUSE'], ['multi-family-households', 'single-family-households'], dim='Categories1')
    # Compute moving average
    window_size = 3  # Change window size to control the smoothing effect
    data_smooth = moving_average(dm_rooms.array, window_size, axis=dm_rooms.dim_labels.index('Years'))
    dm_rooms.array[:, 1:-1, ...] = data_smooth
    window_size = 3  # Change window size to control the smoothing effect
    data_smooth = moving_average(dm_rooms.array, window_size, axis=dm_rooms.dim_labels.index('Years'))
    dm_rooms.array[:, 1:-1, ...] = data_smooth
    window_size = 3  # Change window size to control the smoothing effect
    data_smooth = moving_average(dm_rooms.array, window_size, axis=dm_rooms.dim_labels.index('Years'))
    dm_rooms.array[:, 1:-1, ...] = data_smooth

    # Fill nans
    linear_fitting(dm_rooms, years_ots, min_t0=0, min_tb=0)

    dm_rooms.rename_col('TOTAL', 'lfs_rooms-cap', dim='Variables')

    #dm.fill_nans(dim_to_interp='Years')

    return dm_rooms

def get_pop_by_bld_type(code_eustat, dict_iso2, years_ots):
    # code_eustat = 'ilc_lvho01'
    filter = {'deg_urb': ['TOTAL'], 'geo\\TIME_PERIOD': dict_iso2.keys(), 'incgrp': ['TOTAL'],
              'building': ['FLAT', 'HOUSE'], 'freq': 'A'}
    mapping_dim = {'Country': 'geo\\TIME_PERIOD', 'Variables': 'freq', 'Categories1': 'building'}
    dm_pop_share = get_data_api_eurostat('ilc_lvho01', filter, mapping_dim, unit='%', years=years_ots)
    dm_pop_share.rename_col(['FLAT', 'HOUSE'], ['multi-family-households', 'single-family-households'], dim='Categories1')
    dm_pop_share.rename_col('A', 'lfs_pop-by-bld-type_share', dim='Variables')
    dm_pop_share.drop('Years', 2003)

    for i in range(2):
        window_size = 3  # Change window size to control the smoothing effect
        data_smooth = moving_average(dm_pop_share.array, window_size, axis=dm_pop_share.dim_labels.index('Years'))
        dm_pop_share.array[:, 1:-1, ...] = data_smooth

    linear_fitting(dm_pop_share, years_ots)
    dm_pop_share.normalise(dim='Categories1')

    return dm_pop_share

def estimate_stock_res_from_average_room_size(dm_rooms, dm_area_2020, dm_pop, dm_pop_bld_type):
    # 1) lfs_rooms-cap [rooms/cap] x lfs_pop-by-bld-type [habitants] = lfs_rooms [rooms] (by bld type)
    # 2) bld_floor-area_2020 [m2] / lfs_rooms [rooms] (t=2020) = lfs_avg-room-size [m2/rooms] (t=2020)
    # 3) bld_floor-area_stock_tmp [m2] = lfs_rooms [rooms] x lfs_avg-room-size [m2/rooms]
    # (by bld type, value applied to all years)

    # lfs_pop-by-bld-type = pop_bld_type_share x dm_pop
    dm_pop_bld_type = dm_pop_bld_type.flatten()
    # dm_pop_bld_type.drop(dim='Years', col_label=[2023])
    vars = dm_pop_bld_type.col_labels['Variables']
    dm_pop_bld_type.append(dm_pop, dim='Variables')
    for var in vars:
        var_out = str.replace(var, '_share', '')
        dm_pop_bld_type.operation(var, '*', 'lfs_population_total', out_col=var_out, unit='inhabitants')
    dm_pop_bld_type.drop(dim='Variables', col_label='lfs_population_total')
    dm_pop_bld_type.deepen()

    dm_all = dm_pop_bld_type.filter({'Variables': ['lfs_pop-by-bld-type']})
    dm_all.append(dm_rooms, dim='Variables')
    # rooms/cap * pop = rooms (by bld type)
    dm_all.operation('lfs_pop-by-bld-type', '*', 'lfs_rooms-cap', out_col='lfs_rooms', unit='m2')

    # Put all 2020 data together
    dm_all_2020 = dm_area_2020.filter_w_regex({'Categories1': '.*households'})
    dm_all_2020.filter({'Variables': ['bld_floor-area']}, inplace=True)
    dm_all_2020.append(dm_all.filter({'Years': [2020]}, inplace=False), dim='Variables')

    # floor-araa_2020/rooms_2020 = room_size_2020 [m2/room]
    dm_all_2020.operation('bld_floor-area', '/', 'lfs_rooms', out_col='lfs_room-size', unit='m2/room')

    # bld_floor-area_stock_tmp [m2] = lfs_rooms [rooms] x lfs_room-size [m2/rooms]
    idx = dm_all.idx
    idx_2 = dm_all_2020.idx
    arr_stock = dm_all.array[:, :, idx['lfs_rooms'], :] * dm_all_2020.array[:, 0, np.newaxis, idx_2['lfs_room-size'], :]
    dm_all.add(arr_stock, dim='Variables', col_label='bld_floor-area_stock_tmp', unit='m2')

    # Filter bld_floor-area_stock_tmp
    dm_all.filter({'Variables': ['bld_floor-area_stock_tmp']}, inplace=True)

    return dm_all

def estimate_stock_non_res(dm_area_2020, dm_stock_res):

    dm_area_2020.sort('Country')
    dm_stock_res.sort('Country')
    dm_area_2020.sort('Categories1')
    dm_stock_res.sort('Categories1')
    dm_area_2020.normalise('Categories1', inplace=True,  keep_original=True)

    dm_2020 = dm_area_2020.copy()
    dm_2020.groupby({'residential': '.*households'}, dim='Categories1', regex=True, inplace=True)
    dm_stock = dm_stock_res.filter({'Variables': ['bld_floor-area_stock_tmp']})
    dm_stock.groupby({'residential': '.*households'}, dim='Categories1', regex=True, inplace=True)
    # Compute floor-stock share
    cats = ['education', 'health', 'hotels', 'offices', 'other', 'trade']
    dm_stock.add(0, dummy=True, dim='Categories1', col_label=cats)
    idx = dm_2020.idx
    idx_a = dm_stock.idx
    # stock_tot (t) = stock_res(t) / share_res(t)
    # stock_non-res (t) = stock_tot(t)*share_non-res(t)
    # for share res and non-res we use the 2020 values
    for year in dm_stock.col_labels['Years']:
        arr_tot_area_y = dm_stock.array[:, idx_a[year], idx_a['bld_floor-area_stock_tmp'], idx_a['residential']] \
                         / dm_2020.array[:, idx[2020], idx['bld_floor-area_share'], idx['residential']]
        for cat in cats:
            dm_stock.array[:, idx_a[year], idx_a['bld_floor-area_stock_tmp'], idx_a[cat]] = \
                arr_tot_area_y * dm_2020.array[:, idx[2020], idx['bld_floor-area_share'], idx[cat]]

    dm_stock.drop(dim='Categories1', col_label='residential')
    dm_stock.append(dm_stock_res.filter_w_regex({'Variables': 'bld_floor-area_stock_tmp',
                                                'Categories1': '.*households'}), dim='Categories1')

    return dm_stock

def get_new_building():
    file = 'Floor-area_new_built.xlsx'
    data_path = 'data/'
    rows_to_skip = [0, 506, 507]
    df_new = pd.read_excel(data_path+file, sheet_name='Export', skiprows=rows_to_skip)
    df_new.rename({'Year': 'Years', 'Value': 'bld_floor-area_new_residential[m2]',
                   'Value.1': 'bld_floor-area_new_non-residential[m2]'}, axis=1, inplace=True)
    dm_new = DataMatrix.create_from_df(df_new, num_cat=1)
    dm_new.rename_col('Czechia', 'Czech Republic', dim='Country')
    return dm_new

def estimate_floor_area(dm_new_group, dm_new_type, years_ots):
    dm_new_type.drop('Years', 1994)
    # Smooth new construction permits
    window_size = 3  # Change window size to control the smoothing effect
    data_smooth = moving_average(dm_new_group.array, window_size, axis=dm_new_group.dim_labels.index('Years'))
    dm_new_group.array[:, 1:-1, ...] = data_smooth
    window_size = 5  # Change window size to control the smoothing effect
    data_smooth = moving_average(dm_new_group.array, window_size, axis=dm_new_group.dim_labels.index('Years'))
    dm_new_group.array[:, 2:-2, ...] = data_smooth

    # Compute share for sfh and mfh
    dm_new_res_type = dm_new_type.filter({'Categories1': ['single-family-households', 'multi-family-households']}, inplace=False)
    arr_res_share = dm_new_res_type.array / np.nansum(dm_new_res_type.array, axis=-1, keepdims=True)
    dm_new_res_type.add(arr_res_share, dim='Variables', col_label='bld_floor-area_share', unit='%')
    # Compute shares for non residential (commercial)
    dm_new_comm_type = dm_new_type.filter({'Categories1': ['education', 'health', 'hotels', 'offices', 'other', 'trade']}, inplace=False)
    arr_comm_share = dm_new_comm_type.array / np.nansum(dm_new_comm_type.array, axis=-1, keepdims=True)
    dm_new_comm_type.add(arr_comm_share, dim='Variables', col_label='bld_floor-area_share', unit='%')
    # Extrapolate to all the years available in dm_new_group
    years_tmp = dm_new_group.col_labels['Years']
    linear_fitting(dm_new_comm_type, years_tmp)
    linear_fitting(dm_new_res_type, years_tmp)
    # Multiply new floor-area group by the shares
    idx_g = dm_new_group.idx
    idx_c = dm_new_comm_type.idx
    dm_new_comm_type.array[:, :, idx_c['bld_floor-area_new'], :] \
        = dm_new_comm_type.array[:, :, idx_c['bld_floor-area_share'], :] \
          * dm_new_group.array[:, :, idx_g['bld_floor-area_new'], idx_g['non-residential'], np.newaxis]
    idx_r = dm_new_res_type.idx
    dm_new_res_type.array[:, :, idx_r['bld_floor-area_new'], :] \
        = dm_new_res_type.array[:, :, idx_r['bld_floor-area_share'], :] \
          * dm_new_group.array[:, :, idx_g['bld_floor-area_new'], idx_g['non-residential'], np.newaxis]
    # Join residential and commercial new floor area and apply linear extrapolation
    dm_new_res_type.append(dm_new_comm_type, dim='Categories1')
    dm_new_res_type.drop(dim='Variables', col_label='bld_floor-area_share')
    linear_fitting(dm_new_res_type, years_ots, min_t0=0, min_tb=0)
    window_size = 3
    data_smooth = moving_average(dm_new_res_type.array, window_size, axis=dm_new_res_type.dim_labels.index('Years'))
    dm_new_res_type.array[:, 1:-1, ...] = data_smooth
    data_smooth = moving_average(dm_new_res_type.array, window_size, axis=dm_new_res_type.dim_labels.index('Years'))
    dm_new_res_type.array[:, 1:-1, ...] = data_smooth
    # Compute EU data for area in 2020
    dm_new_res_type_EU27 = dm_new_res_type.groupby({'EU27': EU27_cntr_list}, dim='Country', inplace=False)
    dm_new_res_type.drop('Country', 'EU27')
    dm_new_res_type.append(dm_new_res_type_EU27, dim='Country')
    return dm_new_res_type

def estimate_waste_fix_stock(dm_area_stock_tmp, dm_area_new):
    
    # Filter & join residential households data
    # dm_area_new.drop('Years', 2023)
    dm_area = dm_area_new.copy()
    dm_area.append(dm_area_stock_tmp, dim='Variables')

    # Demolition rate
    dm_area_out = dm_area.copy()
    
    # Set the demolition rate at 0.2% by default
    dm_area_out.add(0.002, dummy=True, dim='Variables', col_label='bld_demolition-rates', unit='m2')
    idx = dm_area_out.idx
    x = np.array(dm_area.col_labels['Years'])
    #breakpoints = x[::5]
    # Improved initial guess for demolition-rate
    for cntr in dm_area.col_labels['Country']:
        for cat in dm_area.col_labels['Categories1']:
            y = dm_area.array[idx[cntr], :, idx['bld_floor-area_stock_tmp'], idx[cat]]
            ds, q = np.polyfit(x, y, 1)  # 1 is for a linear fit (degree 1)
            s0 = np.polyfit(x, y, 0)
            # ds = stock(t) - stock(t-1) = new(t) - waste(t)
            n = dm_area.array[idx[cntr], :, idx['bld_floor-area_new'], idx[cat]]
            n0 = np.polyfit(x, n, 0)
            if n0 > ds:
                # new = n0
                # waste = n0 - ds
                dm_area_out.array[idx[cntr], :, idx['bld_demolition-rates'], idx[cat]] = (n0 - ds)/s0
            #else:
            #    dm_area_out.array[idx[cntr], :, idx['bld_floor-area_new'], idx[cat]] = n/n0*ds

    #dm_area_out.filter({'Variables': ['bld_floor-area_dem-rates', 'bld_floor-area_new', 'bld_floor-area_stock']}, inplace=True)
    
    ###### Add dem-rates from literature
    # Sandberg, Nina Holck, Igor Sartori, Oliver Heidrich, Richard Dawson,
    # Elena Dascalaki, Stella Dimitriou, Tomáš Vimm-r, et al.
    # “Dynamic Building Stock Modelling: Application to 11 European Countries to Support the Energy Efficiency
    # and Retrofit Ambitions of the EU.” Energy and Buildings 132 (November 2016): 26–38.
    # https://doi.org/10.1016/j.enbuild.2016.05.100.
    dem_rates_literature = {'Cyprus': 0.002, 'Czech Republic': 0.006, 'France': 0.005,
                            'Germany': 0.006, 'Greece': 0.009, 'Hungary': 0.005,
                            'Netherlands': 0.005, 'Slovenia': 0.006}

    for c, dem_rate in dem_rates_literature.items():
        dm_area_out.array[idx[c], :, idx['bld_demolition-rates'], :] = dem_rate

    dm_area_out.rename_col('bld_floor-area_stock_tmp', 'bld_floor-area_stock', dim='Variables')
    idx = dm_area_out.idx
    for c in dm_area_out.col_labels['Categories1']:
        for ti in dm_area_out.col_labels['Years'][-1:0:-1]:
            stock_t = dm_area_out.array[:, idx[ti], idx['bld_floor-area_stock'], idx[c]]
            new_t = dm_area_out.array[:, idx[ti], idx['bld_floor-area_new'], idx[c]]
            dem_rate_t = dm_area_out.array[:, idx[ti], idx['bld_demolition-rates'], idx[c]]
            stock_tm1 = (stock_t - new_t)/(1-dem_rate_t)
            dm_area_out.array[:, idx[ti - 1], idx['bld_floor-area_stock'], idx[c]] = stock_tm1

    #dm_area_stock_orig = dm_area.filter({'Variables': ['bld_floor-area_stock']}, inplace=False)
    #dm_area_stock_orig.rename_col('bld_floor-area_stock', 'bld_floor-area_stock_orig', dim='Variables')
    #dm_area_out.append(dm_area_stock_orig, dim='Variables')
    
    # Fix Malta
    # TODO: this is a temporary fix, to understand why there was that problem
    idx2 = dm_area_out.idx
    dm_temp = dm_area_stock_tmp.flatten()
    idx1 = dm_temp.idx
    dm_area_out.array[idx2["Malta"],:,idx2['bld_floor-area_stock'],:] = dm_temp.array[idx1["Malta"],...]
    
    # Make stock in t minus 1
    dm_area_out.lag_variable('bld_floor-area_stock', shift=1, subfix='_tm1')
    
    # make waste
    # TODO: For the moment I do not aggregate and I keep EU27 as unit
    # to be decided if to do the aggregation
    # stock(t) - stock(t - 1) = new(t) - waste(t)
    idx = dm_area_out.idx
    waste_t = dm_area_out.array[:, :, idx['bld_floor-area_new'], :] - \
              ( dm_area_out.array[:, :, idx['bld_floor-area_stock'], :] \
              - dm_area_out.array[:, :, idx['bld_floor-area_stock_tm1'], :])
    dm_area_out.add(waste_t, "Variables", "bld_floor-area_waste", unit="m2", dummy=True)
    dm_area_out.drop(dim='Variables', col_label=['bld_floor-area_stock_tm1'])
    # dm_area_out.flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()
    
    # fix waste for 1990
    idx = dm_area_out.idx
    dm_area_out.array[:,idx[1990],idx["bld_floor-area_waste"],:] = np.nan
    dm_area_out = linear_fitting(dm_area_out, years_ots, based_on=list(range(1990,1995+1,1)))
    # dm_area_out.flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()
    
    # # EU27
    # dm_area_out_EU27 = dm_area_out.groupby({'EU27': EU27_cntr_list}, dim='Country', inplace=False)
    # dm_area_out.drop('Country', 'EU27')
    # dm_area_out.append(dm_area_out_EU27, dim='Country')
    # dm_area_out.sort('Country')
    # idx = dm_area_out.idx
    # # stock(t) - stock(t - 1) = new(t) - waste(t)
    # waste_t = dm_area_out.array[idx['EU27'], :, idx['bld_floor-area_new'], :] - \
    #           ( dm_area_out.array[idx['EU27'], :, idx['bld_floor-area_stock'], :] \
    #           - dm_area_out.array[idx['EU27'], :, idx['bld_floor-area_stock_tm1'], :])
    # # dm_area_out.array[idx['EU27'], :, idx['bld_demolition-rates'], :] = \
    # #     waste_t/dm_area_out.array[idx['EU27'], :, idx['bld_floor-area_stock_tm1'], :]
    # dm_area_out.add(np.nan, "Variables", "bld_floor-area_waste", unit="m2", dummy=True)
    # dm_area_out.array[idx['EU27'], :, idx['bld_floor-area_waste'], :] = waste_t
    # dm_area_out.drop(dim='Variables', col_label=['bld_floor-area_stock_tm1'])

    return dm_area_out

def compute_floor_area_waste_cat(dm_waste_tot):
    
    # Assumption: following EPBD on renovation, non-residential buildings in Class G are required 
    # to reach at least Class F by 2027 and Class E by 2030. Residential buildings 
    # have until 2030 to reach Class F and until 2033 to reach Class E.
    # buildings that are structurally unsound or economically unviable to retrofit 
    # may be candidates for demolition. Given that Classes G and F encompass the least 
    # energy-efficient buildings, it's plausible that a significant portion of demolitions 
    # since 1990 involved structures within these categories. In our case we do not have G,
    # so we assume that 80% of demolished is F (majority), and 20% is E.
    
    dm = dm_waste_tot.copy()
    variabs = dm.col_labels["Categories1"]
    for v in variabs:
        dm.rename_col(v,v+"_F","Categories1")
    dm.deepen()
    arr_temp = dm.array * 0.2
    dm.add(arr_temp, dim='Categories2', col_label=['E'])
    idx = dm.idx
    dm.array[...,idx["F"]] = dm.array[...,idx["F"]] * 0.8
    dm.add(0, dummy=True, dim='Categories2', col_label=['B', 'C', 'D'])
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

def get_renovation_rate():
    
    file = 'renovation_rates.xlsx'
    data_path = 'data/'
    df_rr_res = pd.read_excel(data_path + file, sheet_name='Renovation_rates_residential')
    df_rr_nonres = pd.read_excel(data_path + file, sheet_name='Renovation_rates_non_residentia')
    df_rr_res.rename({'Energy related: “Light” ': 'bld_ren-rate_sha_residential[%]',
                      'Energy related: “Medium” ': 'bld_ren-rate_med_residential[%]',
                      'Energy related: “Deep” ': 'bld_ren-rate_dep_residential[%]'}, axis=1, inplace=True)
    df_rr_nonres.rename({'Energy related: “Light” ': 'bld_ren-rate_sha_non-residential[%]',
                         'Energy related: “Medium” ': 'bld_ren-rate_med_non-residential[%]',
                         'Energy related: “Deep” ': 'bld_ren-rate_dep_non-residential[%]'}, axis=1, inplace=True)
    
    # Drop useless cols
    df_rr_res.drop(['Energy related: “Total” ', 'Energy related: “below Threshold” '], axis=1, inplace=True)
    df_rr_nonres.drop(['Energy related: “Total” ', 'Energy related: “below Threshold” '], axis=1, inplace=True)
    
    # Remove space in Country col
    df_rr_res['Country'] = df_rr_res['Country'].str.strip()
    df_rr_nonres['Country'] = df_rr_nonres['Country'].str.strip()
    
    # These data are the average between 2012 - 2016
    df_rr_res['Years'] = 2014
    df_rr_nonres['Years'] = 2014
    df_rr = pd.merge(df_rr_res, df_rr_nonres, on=['Country', 'Years'], how='inner')
    dm_rr = DataMatrix.create_from_df(df_rr, num_cat=2)
    
    # # Duplicate 2014 values for all years
    # for y in list(range(1990, 2023+1)):
    #     if y != 2014:
    #         # Take first year value
    #         new_array = dm_rr.array[:, 0, ...]
    #         dm_rr.add(new_array, dim='Years', col_label=[y], unit='%')
    # dm_rr.sort(dim='Country')
    
    # Drop United Kingdom
    dm_rr.drop("Country","United Kingdom")
    # dm_rr_EU27 = dm_rr.groupby({'EU27': EU27_cntr_list}, dim='Country', inplace=False, aggregation="mean")
    # dm_rr.append(dm_rr_EU27,"Country")
    # dm_rr.sort("Country")
    
    # Take sum medium and deep
    # from this: https://www.eea.europa.eu/publications/building-renovation-where-circular-economy/modelling-the-renovation-of-buildings/view
    # They mention that medium is mostly insulation, and deep is insulation + heating
    # I will consider the sum of medium and deep for now (even though deep also includes heating)
    # dm_rr.flatten().flatten().datamatrix_plot()
    dm_rr = dm_rr.filter({"Categories1" : ['med', 'dep']}).group_all("Categories1",inplace=False)
    # dm_rr.flatten().datamatrix_plot()
    
    # Make EU
    dm_rr_EU27 = dm_rr.groupby({'EU27': EU27_cntr_list}, dim='Country', inplace=False, aggregation="mean")
    dm_rr.append(dm_rr_EU27,"Country")
    dm_rr.sort("Country")
    # numbers are close to Table 1.3 pag 30 
    # from this: https://www.eea.europa.eu/publications/building-renovation-where-circular-economy/modelling-the-renovation-of-buildings/view
    
    # Assign to different bld types
    mapping_bld_type = {'education':'non-residential', 
                        'health':'non-residential', 
                        'hotels':'non-residential', 
                        'multi-family-households':'residential', 
                        'offices':'non-residential', 
                        'other':'non-residential', 
                        'single-family-households':'residential', 
                        'trade':'non-residential'}
    idx = dm_rr.idx
    for key in mapping_bld_type.keys():
        arr_temp = dm_rr.array[...,idx[mapping_bld_type[key]]]
        dm_rr.add(arr_temp,"Categories1",key,"%")
    dm_rr.drop("Categories1",'non-residential')
    dm_rr.drop("Categories1",'residential')
    dm_rr.sort("Categories1")
    dm_rr.rename_col("bld_ren-rate", "bld_renovation-rate", "Variables")
    
    # Add year dimension
    
    # Assumption: insulation started to gain traction in 2002, when the Energy Performance of Buildings Directive (EPBD) came into force
    years_missing = list(range(1990,2023+1))
    years_missing.remove(2014)
    dm_rr.add(np.nan, "Years", years_missing, dummy = True)
    dm_rr.sort("Years")
    idx = dm_rr.idx
    dm_rr.array[:,0:idx[2002],...] = 0
    years_same = list(range(2015,2023+1))
    for y in years_same:
        dm_rr.array[:,idx[y],...] = dm_rr.array[:,idx[2014],...]
    linear_fitting(dm_rr,list(range(2003,2014+1)))
    # df = dm_rr.write_df()
    
    # # Fix: for the moment assume renovation rate for non residential to be the same 
    # # of residential (otherwise we will get negative stocks for F with current parameters)
    # idx = dm_rr.idx
    # for c in ['education', 'health', 'hotels', 'offices', 'other', 'trade']:
    #     dm_rr.array[...,idx[c]] = dm_rr.array[...,idx['multi-family-households']]
    
    return dm_rr

def make_ren_maps():
    
    # According to the Programme Batiments the assenissment is
    # Amélioration de +1 classes CECB 57%
    # Amélioration de +2 classes CECB 15%
    # Amélioration de +3 classes CECB 15%
    # Amélioration de +4 classes CECB 13%

    # classes that get renovated
    ren_map_out = {(1990, 2000): {'F': -0.8, 'E': -0.2, 'D': 0, 'C': 0, 'B': 0},
                  (2001, 2010): {'F': -0.8, 'E': -0.2, 'D': 0, 'C': 0, 'B': 0},
                  (2011, 2023): {'F': -0.8, 'E': -0.2, 'D': 0, 'C': 0, 'B': 0}}

    # improvements
    improv_1classes = 0.57
    improv_2classes = 0.15
    improv_3classes = 0.15
    improv_4classes = 0.13

    # create dictionary for new classes
    ren_map_in = {(1990, 2000) : {'F':0, 'E': 0, 'D': 0, 'C': 0, 'B': 0},
                  (2001, 2010): {'F':0, 'E': 0, 'D': 0, 'C': 0, 'B': 0},
                  (2011, 2023): {'F':0, 'E': 0, 'D': 0, 'C': 0, 'B': 0}}

    # 1990
    f_to_e = (-ren_map_out[(1990, 2000)]["F"])*improv_1classes # F -> E
    f_to_d = (-ren_map_out[(1990, 2000)]["F"])*improv_2classes # F -> D
    f_to_c = (-ren_map_out[(1990, 2000)]["F"])*improv_3classes # F -> C
    e_to_d = (-ren_map_out[(1990, 2000)]["E"])*improv_1classes # E -> D
    e_to_c = (-ren_map_out[(1990, 2000)]["E"])*improv_2classes # E -> C
    e = f_to_e
    d = f_to_d + e_to_d
    c = f_to_c + e_to_c
    tot = e + d + c 
    ren_map_in[(1990, 2000)]["E"] = e/tot
    ren_map_in[(1990, 2000)]["D"] = d/tot
    ren_map_in[(1990, 2000)]["C"] = c/tot

    # 2001
    f_to_e = (-ren_map_out[(2001, 2010)]["F"])*improv_1classes # F -> E
    f_to_d = (-ren_map_out[(2001, 2010)]["F"])*improv_2classes # F -> D
    f_to_c = (-ren_map_out[(2001, 2010)]["F"])*improv_3classes # F -> C
    f_to_b = (-ren_map_out[(2001, 2010)]["F"])*improv_4classes # F -> B
    e_to_d = (-ren_map_out[(2001, 2010)]["E"])*improv_1classes # E -> D
    e_to_c = (-ren_map_out[(2001, 2010)]["E"])*improv_2classes # E -> C
    e_to_b = (-ren_map_out[(2001, 2010)]["E"])*improv_3classes # E -> B
    e = f_to_e
    d = f_to_d + e_to_d
    c = f_to_c + e_to_c
    b = f_to_b + e_to_b
    b = 2 # adjustment to get to get steep slope to get to zero before 2000
    tot = e + d + c + b
    ren_map_in[(2001, 2010)]["E"] = e/tot
    ren_map_in[(2001, 2010)]["D"] = d/tot
    ren_map_in[(2001, 2010)]["C"] = c/tot
    ren_map_in[(2001, 2010)]["B"] = b/tot
    
    # 2011
    ren_map_in[(2011, 2023)] = ren_map_in[(2001, 2010)]
    
    return ren_map_in, ren_map_out


def extract_renovation_redistribuition(ren_map_in, ren_map_out, years_ots):
    dm = DataMatrix(col_labels={'Country': EU27_cntr_list + ["EU27"],
                                'Years': years_ots,
                                'Variables': ['bld_renovation-redistribution-in',
                                              'bld_renovation-redistribution-out'],
                                'Categories1': ['B', 'C', 'D', 'E', 'F']},
                    units={'bld_renovation-redistribution-in': '%', 'bld_renovation-redistribution-out': '%'})
    dm.array = np.nan * np.ones((len(dm.col_labels['Country']),
                                 len(dm.col_labels['Years']),
                                 len(dm.col_labels['Variables']),
                                 len(dm.col_labels['Categories1'])))
    dm.sort("Country")
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
    
    # drop F
    dm_tmp = dm_all_cat.filter({'Variables': ['bld_floor-area_stock']})
    dm_tmp.drop(col_label='F', dim='Categories2')
    
    # # filter temp
    # dm_tmp = dm_tmp.filter({"Country" : ["EU27"]})
    
    # Get minimum of each time series (axis 1 is year) and change the sign
    shift = -np.min(dm_tmp.array[:, :, 0, :, :], axis=1)
    
    # Set to zero the negative numbers
    shift = np.maximum(0, shift)
    
    # Replace 0 with nan
    
    # Step 1: Replace zeros with NaN
    shift[shift == 0] = np.nan
    # note: nothing changes for EU countries (there are no nan at this step)
    
    # Step 2: Compute the minimum across axis 2, ignoring NaNs
    min_values = np.nanmin(shift, axis=2, keepdims=True)
    # note: axis 2 is the energy classes
    
    # Step 3: Replace NaNs with the corresponding minimum values
    shift = np.where(np.isnan(shift), min_values, shift)
    
    # Sum the shift to dm_tmp
    dm_tmp.array[:, :, 0, :, :] = dm_tmp.array[:, :, 0, :, :] + shift[:, np.newaxis, :, :]
    
    # Get stock without F
    dm_stock_wo_F = dm_tmp.group_all(dim='Categories2', inplace=False)
    dm_stock_wo_F.rename_col('bld_floor-area_stock', 'bld_floor-area_stock_woF', dim='Variables')
    dm_stock_tot.append(dm_stock_wo_F, dim='Variables')
    
    # Get F by subtracting full stock minus stock without F
    dm_stock_tot.operation('bld_floor-area_stock', '-', 'bld_floor-area_stock_woF', out_col='bld_floor-area_stock_F', unit='m2')
    dm_stock_F = dm_stock_tot.filter({'Variables': ['bld_floor-area_stock_F']})
    dm_stock_F.deepen(based_on='Variables')
    dm_tmp.append(dm_stock_F, dim='Categories2')
    
    # Put this new stock inside dm_all_cat
    dm_all_cat.drop(col_label='bld_floor-area_stock', dim='Variables')
    dm_all_cat.append(dm_tmp, dim='Variables')
    
    # In dm_stock_tot keep only the stock
    dm_stock_tot.filter({'Variables': ['bld_floor-area_stock']}, inplace=True)
    
    return dm_all_cat

def compute_stock_area_by_cat(dm_stock_cat, dm_new_cat, dm_renov_cat, dm_waste_cat, dm_stock_tot):
    
    # Checks
    # df_temp = dm_stock_cat.filter({"Country" : ["EU27"],"Years":[2020], "Categories1" : ['multi-family-households','single-family-households']}).group_all("Categories2",inplace=False).write_df()
    # df_temp = pd.melt(df_temp, ["Country","Years"])
    # df_temp = dm_stock_tot.filter({"Country" : ["EU27"],"Years":[2020], "Categories1" : ['multi-family-households','single-family-households']}).write_df()
    # df_temp = pd.melt(df_temp, ["Country","Years"])
    # dm_all_cat.filter({"Country" : ["EU27"], "Variables" : ["bld_floor-area_stock"]}).flatten().flatten().datamatrix_plot() 
    
    # Adapt dm_stock_cat 2020 to be equal in aggregate to dm_stock_tot 2020
    idx1 = dm_stock_cat.idx
    arr_temp1 = dm_stock_cat.array[:,idx1[2020],:,:,:]
    idx2 = dm_stock_tot.idx
    arr_temp2 = dm_stock_tot.array[:,idx2[2020],:,:]
    dm_stock_cat.array[:,idx1[2020],:,:,:] = arr_temp1 * arr_temp2[...,np.newaxis] / np.sum(arr_temp1,axis=3,keepdims=True)
    # TODO: recheck here above
    
    # # Try to assign it to 2023
    # dm_stock_cat.array[:,idx1[2023],:,:,:] = dm_stock_cat.array[:,idx1[2020],:,:,:]
    
    # s_{c,t}(t-1) &= s_{c,t}(t) - n_{c,t}(t) - r_{c,t}(t) + w_{c,t}(t)
    dm_all_cat = dm_stock_cat.copy()
    dm_all_cat.append(dm_new_cat, dim='Variables')
    dm_all_cat.append(dm_renov_cat, dim='Variables')
    dm_all_cat.append(dm_waste_cat, dim='Variables')
    idx = dm_all_cat.idx
    
    # Checks
    # dm_all_cat.filter({'Country' : ["EU27"],'Variables': ['bld_floor-area_new'], "Categories1" : ['multi-family-households','single-family-households']}).flatten().datamatrix_plot()
    # dm_all_cat.filter({'Country' : ["EU27"],'Variables': ['bld_floor-area_renovated'], "Categories1" : ['multi-family-households','single-family-households']}).flatten().datamatrix_plot()
    # dm_all_cat.filter({'Country' : ["EU27"],'Variables': ['bld_floor-area_waste'], "Categories1" : ['multi-family-households','single-family-households']}).flatten().datamatrix_plot()

    # From 2020 backwards
    for t in reversed(range(1991,2020+1)):
        dm_all_cat.array[:, idx[t-1], idx['bld_floor-area_stock'], ...] \
            = dm_all_cat.array[:, idx[t], idx['bld_floor-area_stock'], ...] \
              - dm_all_cat.array[:, idx[t], idx['bld_floor-area_new'], ...] \
              - dm_all_cat.array[:, idx[t], idx['bld_floor-area_renovated'], ...] \
              + dm_all_cat.array[:, idx[t], idx['bld_floor-area_waste'], ...]
    
    # From 2020 onwards
    for t in range(2021,2023+1):
        dm_all_cat.array[:, idx[t], idx['bld_floor-area_stock'], ...] \
            = dm_all_cat.array[:, idx[t-1], idx['bld_floor-area_stock'], ...] \
              + dm_all_cat.array[:, idx[t], idx['bld_floor-area_new'], ...] \
              + dm_all_cat.array[:, idx[t], idx['bld_floor-area_renovated'], ...] \
              - dm_all_cat.array[:, idx[t], idx['bld_floor-area_waste'], ...]

    # FIX Negative Stock
    # Adjusts the stock such that new, waste, renovated are unchanged (stock slope is fixed)
    # This is done by shifting up categories B, C, D, E and shifting down category F
    dm_all_cat = fix_negative_stock(dm_all_cat, dm_stock_tot)
    
    # In some cases F can be negative for non residential, as renovation is higher for those.
    # I will put F = 0 when negative for now
    dm = dm_all_cat.filter({"Variables" : ['bld_floor-area_stock']})
    dm.array[dm.array<0]=0
    dm_all_cat.drop("Variables", ["bld_floor-area_stock"])
    dm_all_cat.append(dm,"Variables")
    dm_all_cat.sort("Variables")
    
    # IF the fix is done correctly the following run should leave things unchanged for residential,
    # and change things for non residential (given that we did the fix F = 0 above)
    idx = dm_all_cat.idx
    for t in reversed(dm_all_cat.col_labels['Years'][1:]):
        dm_all_cat.array[:, idx[t-1], idx['bld_floor-area_stock'], ...] \
            = dm_all_cat.array[:, idx[t], idx['bld_floor-area_stock'], ...] \
              - dm_all_cat.array[:, idx[t], idx['bld_floor-area_new'], ...] \
              - dm_all_cat.array[:, idx[t], idx['bld_floor-area_renovated'], ...] \
              + dm_all_cat.array[:, idx[t], idx['bld_floor-area_waste'], ...]
              
    # Checks
    # dm_all_cat.filter({"Country" : ["EU27"], "Variables" : ["bld_floor-area_stock"]}).flatten().flatten().datamatrix_plot()
    # dm_all_cat.group_all("Categories2",inplace=False).filter({"Country" : ["EU27"], "Variables" : ["bld_floor-area_stock"]}).flatten().datamatrix_plot() 

    return dm_all_cat

def extract_stock_byconstrperiod_2020():
    
    # get data
    file = 'BSO_floor_area_2020.xlsx'
    data_path = 'data/'
    rows_to_skip = [1, 198, 199]
    df_area = pd.read_excel(data_path + file, sheet_name='Export', skiprows=rows_to_skip)
    dict_ren = {'Building use': 'Construction period', 'Unnamed: 1': 'Country'}
    df_area.rename(dict_ren, axis=1, inplace=True)
    df_area["Years"] = 2020
    
    # make dm
    indexes = ["Country","Years"]
    variabs_surface = ['Educational buildings', 'Health buildings', 'Hotels and Restaurants',
                       'Multi-family buildings', 'Offices', 'Other non-residential buildings',
                       'Single-family buildings', 'Trade buildings']
    variab_yofconstr = ["Construction period"]
    df_area = df_area.loc[:,indexes + variabs_surface + variab_yofconstr]
    variabs_surface_new = ['education', 'health', 'hotels', 
                            'multi-family-households', 'offices', 'other', 
                            'single-family-households', 'trade']
    for i in range(len(variabs_surface)):
        df_area.rename(columns={variabs_surface[i] : variabs_surface_new[i]},
                       inplace=True)
    df_area.rename(columns={"Construction period" : "constr_period"},inplace=True)
    df_area = pd.melt(df_area, id_vars = indexes + ["constr_period"])
    df_area["variable"] = df_area["variable"] + "_" + df_area["constr_period"]
    df_area = df_area.pivot(index=indexes, columns="variable", values="value").reset_index()
    columns_to_rename = df_area.columns[2:len(df_area.columns)]
    for c in columns_to_rename:
        df_area.rename(columns={c:"bld_floor-area_stock_" + c + "[m2]"},inplace=True)
    dm_stock_constr_2020 = DataMatrix.create_from_df(df_area, num_cat=2)
    
    return dm_stock_constr_2020

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

def extract_heating_mix():
    
    # Logic
    # categories are ['district-heating', 'electricity', 'gas', 'heat-pump', 'heating-oil', 'solar', 'wood', 'coal', 'other']
    # I will consider both space and water heating (I do not consider cooling and cooking)
    # Source: https://gitlab.com/hotmaps/building-stock/-/tree/master/data?ref_type=heads
    # This is based on Tabula app from 2017
    # Percentages of space heating (SH) are grouped by ["individual", "central", "district heating"], ["boiler", "combined", "stove", "electric heating", "heat pump"], ["solid", "liquid", "gas", "electricity", "biomass"]
    # Percentages of domestic hot water (DHW) are grouped by ["individual", "central", "district heating"], ["boiler", "Combined", "Solar collectors", "heat pump"], ["solid", "liquid", "gas", "electricity", "biomass"]
    # I will assume that the second group is the split of electricity
    # This will give me the picture for 2017 by construction period, which I will then split in envelope classes.

    # get data
    df = pd.read_excel("data/HOTMAPS_heating-mix_2017.xlsx")
    
    # clean
    df["Country"] = "EU27"
    df["Years"] = 2017
    bldtype_old = list(df["bld-type"].unique())
    bldtype_new = ["total","single-family-households","multi-family-households","apartment-blocks"]
    for i in range(0,len(bldtype_old)):
        df.loc[df["bld-type"] == bldtype_old[i],"bld-type"] = bldtype_new[i]
    df.loc[df["construction-year"] == "Before 1945","construction-year"] = "0-1945"
    df.loc[df["construction-year"] == "Post 2010","construction-year"] = "2011-now"
    df["construction-year"] = [s.replace(" ", "") for s in df["construction-year"]]
    df.loc[df["construction-year"] == "1945-1969","construction-year"] = "1946-1969"
    
    # make dm
    indexes = ["Country","Years"]
    df = pd.melt(df, id_vars = indexes + ['bld-type','construction-year'])
    df["variable"] = df["bld-type"] + "_" + df["variable"] + "_" + df["construction-year"]
    df = df.pivot(index=indexes, columns="variable", values="value").reset_index()
    columns_to_rename = df.columns[2:len(df.columns)]
    for c in columns_to_rename:
        df.rename(columns={c:c + "[%]"},inplace=True)
    dm = DataMatrix.create_from_df(df, num_cat=3)
    
    # drop space cooling
    dm.drop("Categories1",["sc"]) # drop space cooling
    dm.drop("Categories2",["space-cooling","no-space-cooling"]) # drop space cooling subs
    
    # # Checks
    # mycat = ['boiler-condensing', 'boiler-non-condensing',
    #          'combined','electric-heating', 'stove','heat-pump','solar-collectors']
    # dm_temp = dm.filter({"Variables" : ["total"], "Categories1" : ["sh"],
    #                      "Categories2" : mycat, "Categories3" : ['2010-today']})
    # dm_temp.group_all("Categories2",inplace=False).array
    # dm_temp = dm.filter({"Variables" : ["total"], "Categories1" : ["dhw"],
    #                      "Categories2" : mycat, "Categories3" : ['2010-today']})
    # dm_temp.group_all("Categories2",inplace=False).array
    
    # aggregate categories withi electricity (this is % of electricity categoriy)
    dict_agg = {"electricity-sub-main" : ['boiler-condensing', 'boiler-non-condensing','electric-heating', 'stove'],
                "electricity-sub-other" : ["combined"]}
    dm_oth = dm.groupby(dict_agg, "Categories2", "sum")
    dm.drop("Categories2",['boiler-condensing', 'boiler-non-condensing','electric-heating', 'stove',"combined"])
    dm.append(dm_oth,"Categories2")
    dm.sort("Categories2")
    
    # # Checks
    # mycat = ["electricity-sub-main","electricity-sub-other",'heat-pump','solar-collectors']
    # dm_temp = dm.filter({"Variables" : ["total"], "Categories1" : ["sh"],
    #                       "Categories2" : mycat, "Categories3" : ['2010-today']})
    # dm_temp.group_all("Categories2",inplace=False).array
    # dm_temp = dm.filter({"Variables" : ["total"], "Categories1" : ["dhw"],
    #                       "Categories2" : mycat, "Categories3" : ['2010-today']})
    # dm_temp.group_all("Categories2",inplace=False).array
    
    # get district heating
    dm_out = dm.filter({"Categories2" : ["district-heating"]})
    dm_notdh = dm.filter({"Categories2" : ["central","individual"]})
    dm_notdh.group_all("Categories2")
    
    # get carrier-type techs
    dm_temp = dm.filter({"Categories2" : ["biomass","electricity","fossil-fuels-gas",
                                          "fossil-fuels-liquid","fossil-fuels-solid"]})
    # dm_temp.group_all("Categories2",inplace=False).array
    dm_temp.normalise("Categories2")
    dm_temp.array = dm_temp.array * dm_notdh.array[:,:,:,:,np.newaxis,:] / 1
    dm_out.append(dm_temp, "Categories2")
    # dm_out.group_all("Categories2",inplace=False).array
    # dm_out.array
    
    # get electricity techs
    dm_temp = dm.filter({"Categories2" : ["electricity-sub-main","electricity-sub-other",
                                          'heat-pump', 'solar-collectors']})
    # dm_temp.group_all("Categories2",inplace=False).array
    dm_temp.normalise("Categories2")
    idx = dm_out.idx
    arr_temp = dm_out.array[:,:,:,:,idx["electricity"],:]
    dm_temp.array = dm_temp.array * arr_temp[:,:,:,:,np.newaxis,:] / 1
    # np.allclose(dm_temp.group_all("Categories2",inplace=False).array, dm_out.array[:,:,:,:,idx["electricity"],:])
    dm_out.drop("Categories2",["electricity"])
    dm_out.append(dm_temp, "Categories2")
    dm_out.rename_col('electricity-sub-main','electricity',"Categories2")
    dm_out.rename_col('electricity-sub-other','other',"Categories2")
    dm_out.sort("Categories2")
    # dm_out.group_all("Categories2",inplace=False).array
    
    # rename
    dm_out.rename_col(
        ["biomass", "fossil-fuels-gas", "fossil-fuels-liquid", "fossil-fuels-solid", "other",      "solar-collectors"],
        ["wood",    "gas",              "heating-oil",         "coal",               "other-tech", "solar"],
        "Categories2")
    dm_out.sort("Categories2")
    # TODO: here I need to call "other" as "other-tech", as there is the category "other" in building type
    # (and we cannot have 2 others). See what to do for CH, but these will have to be conform to be in the
    # same pickle.
    
    # add other bld types
    # Assumption: assigning values from total to non-residential buildings
    dm_temp = dm_out.filter({"Variables" : ['multi-family-households', 'single-family-households',
                                            'total']})
    type_missing = ['education', 'health', 'hotels', 'offices', 'other', 'trade']
    idx = dm_temp.idx
    for t in type_missing:
        dm_temp.add(dm_temp.array[:,:,idx["total"],:,:,:], dim='Variables', col_label=t, unit='%')
    dm_temp.drop("Variables","total")
    dm_temp.sort("Variables")
    dm_out = dm_temp.copy()
    
    # check
    # dm_out.group_all("Categories2",inplace=False).array
    # df = dm_out.filter({"Categories2" : ["solar"]}).write_df()
    
    # Add other years
    
    # get JRC
    file = "data/databases_full/JRC/JRC-IDEES-2021_Residential_EU27.xlsx"
    sheet = "RES_hh_num"
    df = pd.read_excel(file, sheet_name=sheet)
    
    # TODO: see with Paola why for CH we are considering only space heating and not water heating here
    
    # Get space heating
    df_temp = df[0:12].copy()
    names_map = {'Stock of households': 'remove', 'Space heating': 'total', 
                 'Solids': 'coal',
                 'Liquified petroleum gas (LPG)': 'other', 'Diesel oil': 'heating-oil', 'Natural gas': 'gas',
                 'Biomass': 'wood', 'Geothermal': 'other', 'Distributed heat': 'district-heating',
                 'Advanced electric heating': 'heat-pump', 'Conventional electric heating': 'electricity'}
    dm_jrc_sh = df_excel_to_dm(df_temp, names_map, var_name='bld_heating-mix', unit='number', num_cat=1, country="EU27")
    dm_jrc_sh.drop(col_label='remove', dim='Categories1')
    dm_jrc_sh.add(0, dim='Categories1', col_label='solar', dummy=True)
    dm_jrc_sh.sort("Categories1")
    dm_jrc_sh.rename_col("bld_heating-mix","bld_heating-mix_sh","Variables")
    dm_jrc_sh.deepen("_", based_on="Variables")
    dm_jrc_sh.switch_categories_order("Categories2","Categories1")
    
    # Get water heating
    df_temp = pd.concat([df[0:2],df[15:25]])
    names_map = {'Stock of households': 'remove', 'Water heating': 'total', 
                 'Solids': 'coal',
                 'Liquified petroleum gas (LPG)': 'other', 'Diesel oil': 'heating-oil', 'Natural gas': 'gas',
                 'Biomass': 'wood', 'Geothermal': 'other', 'Distributed heat': 'district-heating',
                 'Electricity': 'electricity', 'Solar': 'solar'}
    dm_jrc_dwh = df_excel_to_dm(df_temp, names_map, var_name='bld_heating-mix', unit='number', num_cat=1, country="EU27")
    dm_jrc_dwh.drop(col_label='remove', dim='Categories1')
    dm_jrc_dwh.add(0, dim='Categories1', col_label='heat-pump', dummy=True)
    dm_jrc_dwh.sort("Categories1")
    dm_jrc_dwh.rename_col("bld_heating-mix","bld_heating-mix_dwh","Variables")
    dm_jrc_dwh.deepen("_", based_on="Variables")
    dm_jrc_dwh.switch_categories_order("Categories2","Categories1")
    
    # Put together
    dm_jrc = dm_jrc_sh.copy()
    dm_jrc.append(dm_jrc_dwh, "Categories1")
    dm_jrc.sort("Categories1")
    dm_jrc.drop("Categories2", "total")
    dm_jrc.normalise("Categories2")
    # dm_jrc.group_all("Categories2",inplace=False).array
    # df_temp = dm_jrc.write_df()
    
    # Compute rates of change wrt to 2017
    idx = dm_jrc.idx
    arr_2017 = dm_jrc.array[:,idx[2017],:,:,:]
    arr_temp = (dm_jrc.array - arr_2017[:,np.newaxis,:,:,:])/arr_2017[:,np.newaxis,:,:,:]
    dm_jrc_ch = dm_jrc.copy()
    dm_jrc_ch.array = arr_temp
    dm_jrc_ch.array[dm_jrc_ch.array == np.nan] = 0
    dm_jrc_ch.array = dm_jrc_ch.array + 1
    # df_temp = dm_jrc_ch.write_df()
    
    # Use rates of change to compute time series
    dm_temp = dm_out.copy()
    years_missing = dm_jrc_ch.col_labels["Years"]
    years_missing.remove(2017)
    for y in years_missing:
        dm_temp.add(np.nan, "Years", [y], dummy=True)
    dm_temp.sort("Years")
    dm_temp.array = dm_out.array * dm_jrc_ch.array[...,np.newaxis]
    # df_temp = dm_temp.write_df()
    # dm_temp.flatten().flatten().flatten().datamatrix_plot()
    dm_out = dm_temp.copy()
    
    # df = dm_jrc_ch.filter({"Categories2" : ["solar"]}).write_df()
    # df = dm_out.filter({"Categories2" : ["solar"]}).write_df()
    # df = dm_temp.filter({"Categories2" : ["solar"]}).write_df()
    
    # get average between space heating and domestic hot water
    dm_out.group_all("Categories1",aggregation="mean")
    dm_out.switch_categories_order("Categories2","Categories1")
    dm_out.normalise("Categories2")
    # dm_out.group_all("Categories2",inplace=False).array
    # dm_out.flatten().flatten().datamatrix_plot()
    
    # Do other years
    years_missing = list(set(years_ots) - set(dm_out.col_labels['Years']))
    years_missing_e = [yr for yr in years_missing if yr > dm_out.col_labels['Years'][0]]
    years_missing_i = [yr for yr in years_missing if yr < dm_out.col_labels['Years'][0]]
    linear_fitting(dm_out, years_missing_e, based_on=list(range(2015, dm_out.col_labels['Years'][-1]+1)))
    dm_out.add(np.nan, dim='Years', col_label=years_missing_i,  dummy=True)
    dm_out.sort('Years')
    dm_out.fill_nans('Years')
    
    # Generate other countries
    # TODO: For the moment I generate the other countries as EU27, when we'll do another country we will have
    # to revisit this
    idx = dm_out.idx
    for c in EU27_cntr_list:
        dm_out.add(dm_out.array[idx["EU27"],...], "Country", c)
    dm_out.sort("Country")
    
    # Add bld_heating-mix category
    variabs = dm_out.col_labels["Variables"]
    for v in variabs :
        dm_out.rename_col(v, "bld_heating-mix_" + v, "Variables")
    dm_out.deepen(based_on="Variables")
    dm_out.switch_categories_order("Categories3", "Categories2")
    dm_out.switch_categories_order("Categories1", "Categories2")
    # dm_out.group_all("Categories3",inplace=False).array
    
    return dm_out

def extract_heating_efficiency(file, sheet_name, years_ots):
    
    # TODO: to see with Paola if it's fine to consider both space and water heating
    
    # get data
    df = pd.read_excel(file, sheet_name=sheet_name)
    # df = df[0:13].copy()
    # names_map = {'Ratio of energy service to energy consumption': 'remove', 'Space heating': 'other', 'Solids': 'coal',
    #              'Liquified petroleum gas (LPG)': 'remove', 'Diesel oil': 'heating-oil', 'Natural gas': 'gas',
    #              'Biomass': 'wood', 'Geothermal': 'geothermal', 'Distributed heat': 'district-heating',
    #              'Advanced electric heating': 'heat-pump', 'Conventional electric heating': 'electricity',
    #              'Electricity in circulation': 'remove'}
    # dm_heating_eff = df_excel_to_dm(df, names_map, var_name='bld_heating-efficiency', unit='%', num_cat=1, country="EU27")
    # dm_heating_eff.drop(col_label='remove', dim='Categories1')
    # dm_heating_eff.add(0.8, dim='Categories1', col_label='solar', dummy=True)
    
    # Get space heating
    df_temp = df[0:12].copy()
    names_map = {'Ratio of energy service to energy consumption': 'remove', 'Space heating': 'total', 
                 'Solids': 'coal',
                 'Liquified petroleum gas (LPG)': 'other-tech', 'Diesel oil': 'heating-oil', 'Natural gas': 'gas',
                 'Biomass': 'wood', 'Geothermal': 'other-tech', 'Distributed heat': 'district-heating',
                 'Advanced electric heating': 'heat-pump', 'Conventional electric heating': 'electricity'}
    dm_jrc_sh = df_excel_to_dm(df_temp, names_map, var_name='bld_heating-efficiency_sh', unit='%', num_cat=1, country="EU27")
    dm_jrc_sh.drop(col_label='remove', dim='Categories1')
    dm_jrc_sh.add(0, dim='Categories1', col_label='solar', dummy=True)
    dm_jrc_sh.sort("Categories1")
    dm_jrc_sh.deepen("_", based_on="Variables")
    dm_jrc_sh.switch_categories_order("Categories2","Categories1")
    
    # Get water heating
    df_temp = pd.concat([df[0:2],df[15:25]])
    names_map = {'Ratio of energy service to energy consumption': 'remove', 'Water heating': 'total', 
                 'Solids': 'coal',
                 'Liquified petroleum gas (LPG)': 'other-tech', 'Diesel oil': 'heating-oil', 'Natural gas': 'gas',
                 'Biomass': 'wood', 'Geothermal': 'other-tech', 'Distributed heat': 'district-heating',
                 'Electricity': 'electricity', 'Solar': 'solar'}
    dm_jrc_dwh = df_excel_to_dm(df_temp, names_map, var_name='bld_heating-efficiency_dhw', unit='%', num_cat=1, country="EU27")
    dm_jrc_dwh.drop(col_label='remove', dim='Categories1')
    dm_jrc_dwh.add(0, dim='Categories1', col_label='heat-pump', dummy=True)
    dm_jrc_dwh.sort("Categories1")
    dm_jrc_dwh.deepen("_", based_on="Variables")
    dm_jrc_dwh.switch_categories_order("Categories2","Categories1")
    
    # Put together
    dm_heating_eff = dm_jrc_sh.copy()
    dm_heating_eff.append(dm_jrc_dwh, "Categories1")
    dm_heating_eff.drop("Categories2", "total")
    dm_heating_eff.group_all("Categories1",aggregation="mean")
    # df_temp = dm_heating_eff.write_df()
    
    # Do other years
    years_missing = list(set(years_ots) - set(dm_heating_eff.col_labels['Years']))
    years_missing_e = [yr for yr in years_missing if yr > dm_heating_eff.col_labels['Years'][0]]
    years_missing_i = [yr for yr in years_missing if yr < dm_heating_eff.col_labels['Years'][0]]
    linear_fitting(dm_heating_eff, years_missing_e, based_on=list(range(2015, dm_heating_eff.col_labels['Years'][-1]+1)))
    dm_heating_eff.add(np.nan, dim='Years', col_label=years_missing_i,  dummy=True)
    dm_heating_eff.sort('Years')
    dm_heating_eff.fill_nans('Years')
    
    # Generate other countries
    # TODO: For the moment I generate the other countries as EU27, when we'll do another country we will have
    # to revisit this
    idx = dm_heating_eff.idx
    for c in EU27_cntr_list:
        dm_heating_eff.add(dm_heating_eff.array[idx["EU27"],...], "Country", c)
    dm_heating_eff.sort("Country")
    
    return dm_heating_eff

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

def recompute_floor_area_per_capita(dm_all, dm_pop):

    dm_floor_stock = dm_all.filter({'Variables': ['bld_floor-area_stock'],
                                    'Categories1': ['single-family-households', 'multi-family-households']}, inplace=False)

    # Computer m2/cap for lifestyles
    dm_floor_stock.group_all(dim='Categories2')
    dm_floor_stock.group_all(dim='Categories1')
    dm_floor_stock.append(dm_pop, dim='Variables')

    dm_floor_stock.operation('bld_floor-area_stock', '/', 'lfs_population_total',
                             out_col='lfs_floor-intensity_space-cap', unit='m2/cap')

    dm_floor_stock.filter({'Variables': ['lfs_floor-intensity_space-cap']}, inplace=True)

    return dm_floor_stock

def get_household_size_eustat():
        
    filter = {'geo\\TIME_PERIOD': dict_iso2.keys(), 'freq': 'A'}
    mapping_dim = {'Country': 'geo\\TIME_PERIOD', 'Variables': 'freq'}
    dm_hh_size = get_data_api_eurostat('ilc_lvph01', filter, mapping_dim, unit='people', years=years_ots)
    dm_hh_size.rename_col('A', 'lfs_household-size', dim='Variables')
    linear_fitting(dm_hh_size, years_ots)    
    
    return dm_hh_size

def compute_building_mix(dm_all):
    
    # TODO: drop the filter on categories1 when you do all buildings
    dm_building_mix = dm_all.filter({'Variables': ['bld_floor-area_stock', 'bld_floor-area_new'],
                                     'Categories1': ['single-family-households', 'multi-family-households']}, 
                                    inplace=False).flatten()
    dm_building_mix.normalise('Categories1', keep_original=True)
    dm_building_mix.deepen()
    dm_building_mix.rename_col(['bld_floor-area_stock_share', 'bld_floor-area_new_share'],
                               ['bld_building-mix_stock', 'bld_building-mix_new'], dim='Variables')
    dm_building_mix.filter({'Variables': ['bld_building-mix_stock', 'bld_building-mix_new']}, inplace=True)

    return dm_building_mix

def compute_building_age(dm_stock_cat, years_fts, first_bld):
    dm_age = dm_stock_cat.filter({'Variables': ['bld_floor-area_stock']})
    dm_age.rename_col('bld_floor-area_stock', 'bld_age', dim='Variables')
    dm_age.change_unit('bld_age', 1, 'm2', 'years')
    dm_age.add(np.nan, dim='Years', col_label=years_fts, dummy=True)
    years_all = np.array(dm_age.col_labels['Years'])
    nb_cntr = len(dm_age.col_labels['Country'])
    idx = dm_age.idx
    for bld_type in dm_stock_cat.col_labels["Categories1"]:
        for cat, start_yr in first_bld.items():
            arr_age = years_all - start_yr
            arr_age = np.maximum(arr_age, 0)
            for idx_c in range(nb_cntr):
                dm_age.array[idx_c, :, idx['bld_age'], idx[bld_type], idx[cat]] = arr_age
    return dm_age


def calculate_heating_eff_fts(dm_heating_eff, years_fts, maximum_eff):
    dm_heat_pump = dm_heating_eff.filter({'Categories1': ['heat-pump']})
    dm_heating_eff.drop(dim='Categories1', col_label='heat-pump')
    linear_fitting(dm_heating_eff, years_fts, based_on=list(range(2015, 2023)))
    dm_heating_eff.array = np.minimum(dm_heating_eff.array, maximum_eff)
    linear_fitting(dm_heat_pump, years_fts, based_on=list(range(2015, 2023)))
    dm_heating_eff.append(dm_heat_pump, dim='Categories1')
    dm_heating_eff_fts = dm_heating_eff.filter({'Years': years_fts})

    return dm_heating_eff_fts

#######################################################################
########################### CHECK VARIABLES ###########################
#######################################################################

# Load pickle for CH
filepath = "../../../data/datamatrix/buildings.pickle"
with open(filepath, 'rb') as handle:
    DM_bld = pickle.load(handle)

##################
##### LEVERS #####
##################

list(DM_bld["ots"])
DM_bld["ots"]["floor-intensity"].units 
# lfs_floor-intensity_space-cap is country-year m2/cap
# lfs_household-size is  country-year people (number of people per hh, which is like 2.1)
# PLAN: download it from Eurostat

DM_bld["ots"]["heatcool-behaviour"].units
# bld_Tint-heating and bld_Tint-cooling are country-year-bldtype-bldclass Celsius (celsius of internal temperature)
# PLAN: tbd

list(DM_bld["ots"]["building-renovation-rate"])

DM_bld["ots"]["building-renovation-rate"]["bld_building-mix"].units
# bld_building-mix_new is country-year-bldtype-bldclass % (% of new buildings that are in that class)
# PLAN: 

DM_bld["ots"]["building-renovation-rate"]["bld_renovation-rate"].units
# building-renovation-rate is country-year-bldtype % (% of buildings stock that are renovated)

DM_bld["ots"]["building-renovation-rate"]["bld_renovation-redistribution"].units
# bld_renovation-redistribution-in and bld_renovation-redistribution-out is country-year-bldclass % (% of not clear, to be undertsood)

DM_bld["ots"]["building-renovation-rate"]["bld_demolition-rate"].units
# bld_demolition-rate is country-year-bldtype % (% of buildings stock that are demolished)

DM_bld["ots"]["heating-technology-fuel"]["bld_heating-technology"].units
# bld_heating-mix is country-year-bldtype-bldclass-enesystem % (% of buldings stock that run on that energy)

DM_bld["ots"]["heating-efficiency"].units
# bld_heating-efficiency and bld_heating-efficiency-JRC are country-year-bldclass-enesystem % (% of energy efficiency by energy system)

###############
##### FXA #####
###############

list(DM_bld["fxa"])
DM_bld["fxa"]["heating-energy-calibration"].units
# bld_heating-energy-calibration is country-year-enesystem % (I guess some sorta calibration on efficiency, to be understood)
DM_bld["fxa"]["bld_type"].units
# bld_building-mix_stock is country-year-bldtype-bldclass % (% of buildings stock by building class)
DM_bld["fxa"]["bld_type"].filter({"Country":["Switzerland"]}).group_all("Categories2",inplace=False).group_all("Categories1",inplace=False).array
DM_bld["fxa"]["bld_age"].units
# bld_age is country-year-bldtype-bldclass years (age in years of buildings stock by building class)
DM_bld["fxa"]["emission-factor-electricity"].units
# bld_CO2-factor is country-year kt/TWh (I guess this is kt of co2 per TWh of electricity consumed, but to be understood)

#######################################################################
########################### PRE PROCESSING ############################
#######################################################################

years_ots = create_years_list(start_year=1990, end_year=2023, step=1, astype=int)
years_fts = create_years_list(start_year=2025, end_year=2050, step=5, astype=int)
dict_iso2 = eurostat_iso2_dict()
dict_iso2.pop('CH')  # Remove Switzerland

##########################################
########## Floor area Stock ots ##########
##########################################

# Load U-value for new buildings()
dm_uvalue_new, dm_area_2020, dm_uvalue_stock0, dm_new_2 = get_uvalue_new_stock0(years_ots)

# Compute total floor area from rooms/cap and average room size in 2020
dm_rooms = get_rooms_cap_eustat(dict_iso2, years_ots=years_ots)
# lfs_rooms-cap is country-year-bldtype rooms/cap from eurostat

# Share of population by building type
dm_pop_bld_type = get_pop_by_bld_type('ilc_lvho01', dict_iso2, years_ots)
# lfs_pop-by-bld-type_share is country-year-bldtype % (share of population by bld type)

# Load population data
filepath = "../../lifestyles/Europe/data/lifestyles_allcountries.pickle"
with open(filepath, 'rb') as handle:
    DM_lfs = pickle.load(handle)
dm_pop = DM_lfs["ots"]["pop"]["lfs_population_"].copy()
# dm_pop.append(DM_lfs["fts"]["pop"]["lfs_population_"][1],"Years")
dm_pop = dm_pop.filter({"Country" : list(dict_iso2.values())})
dm_pop.sort("Years")
del DM_lfs

# Drop UK
dm_rooms.drop("Country", ["United Kingdom"])
dm_pop.drop("Country", ["United Kingdom"])
dm_pop_bld_type.drop("Country", ["United Kingdom"])

# Compute floor-area stock (tmp) from avg room size and nb of rooms (sfh, mfh)
dm_residential_stock_tmp = estimate_stock_res_from_average_room_size(dm_rooms, dm_area_2020, dm_pop, dm_pop_bld_type)
# bld_floor-area_stock_tmp is country-year-bldtype m2 (surface of stock by bld type in residential in all years)
dm_all_stock_tmp = estimate_stock_non_res(dm_area_2020, dm_residential_stock_tmp)
# bld_floor-area_stock_tmp is country-year-bldtype m2 (surface of stock by bld type in both residential and office in all years)
del dm_rooms, dm_pop_bld_type

# Checks
# dm_all_stock_tmp.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()
# dm_pop.datamatrix_plot()
# arr_temp = dm_all_stock_tmp.array / dm_pop.array[...,np.newaxis]
# dm_temp = dm_all_stock_tmp.copy()
# dm_temp.array = arr_temp
# dm_temp.datamatrix_plot({"Country" : ["EU27"]}, stacked=True)
# df_temp = dm_all_stock_tmp.write_df()
# df_temp = pd.melt(df_temp, id_vars = ["Country","Years"])
# df_temp = df_temp.loc[df_temp["value"]<0,:]
# Ireland multi family hh is negative over 1990-1998, replacing it with zero
dm_all_stock_tmp.array[dm_all_stock_tmp.array<0]=0

##########################################
########### Floor area New ots ###########
##########################################

# Load new building floor area
dm_new_1 = get_new_building()
# bld_floor-area_new is country-year-bldtype m2 (this is area of new buildings in 2006-2022)

# Reconcile two new build area estimates
dm_area_new = estimate_floor_area(dm_new_1, dm_new_2, years_ots) # this is over 1990-2023
del dm_new_1, dm_new_2

# Checks
# dm_area_new.flatten().datamatrix_plot()
# df_temp = dm_area_new.write_df()
# df_temp = pd.melt(df_temp, id_vars = ["Country","Years"])
# df_temp = df_temp.loc[df_temp["value"]<0,:]
# For negative values I put them as np.nan and fill them with linear fitting
dm_area_new.array[dm_area_new.array<0] = np.nan
dm_area_new = linear_fitting(dm_area_new, years_ots, min_t0=0,min_tb=0)


##########################################################
########### Floor-area Waste + Recompute Stock ###########
##########################################################

# Note: here we recompute the stock (instead of the new) as the stock data is "generated"
# while the new data is raw (on the other hand, for CH we trusted more the stock data
# so we adjusted the new)

dm_all = estimate_waste_fix_stock(dm_all_stock_tmp, dm_area_new)
del dm_residential_stock_tmp, dm_area_new, dm_area_2020
# dm_all.filter({"Country" : ["EU27"], "Variables" : ["bld_demolition-rates"]}).flatten().datamatrix_plot()

# Checks
# dm_all.flatten().datamatrix_plot()
# df_temp = dm_all.write_df()
# df_temp = pd.melt(df_temp, id_vars = ["Country","Years"])
# df_temp = df_temp.loc[df_temp["value"]<0,:]
# Negative values are not for EU27, so for the moment I will just fill them in with linear fitting (there can ne inconsistencies with the stock-flow equations)
dm_all.array[dm_all.array<0] = np.nan
dm_all = linear_fitting(dm_all, years_ots, min_t0=0,min_tb=0)

# assign
dm_stock_tot = dm_all.filter({"Variables" : ["bld_floor-area_stock"]})
dm_new_tot = dm_all.filter({"Variables" : ["bld_floor-area_new"]})
dm_waste_tot = dm_all.filter({"Variables" : ["bld_floor-area_waste"]})

# get waste by envelope (in ots it will be all F as first )
dm_waste_cat = compute_floor_area_waste_cat(dm_waste_tot)
# dm_waste_cat.flatten().flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()

# Floor-area new by sfh, mfh and envelope categories
# source: https://www.bpie.eu/wp-content/uploads/2017/12/State-of-the-building-stock-briefing_Dic6.pdf
# note: I merge together A and B to keep it conform to data on CH
envelope_cat_new = {'C': (1990, 2000), 'B': (2001, 2023)}
dm_new_cat = compute_floor_area_new_cat(dm_new_tot, envelope_cat_new)
# dm_new_cat.flatten().flatten().filter({"Country" : ["EU27"]}).datamatrix_plot(stacked=True)

###############################################
########### Floor area Renovated ots ##########
###############################################

# Load renovation rates
dm_renovation = get_renovation_rate()
# bld_ren-rate is country-year-something-bldtypebroad % (bldtypebroad is non-residential and residential)
# dm_renovation.flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()

# Renovation by envelope cat ots
ren_map_in, ren_map_out = make_ren_maps()
dm_renov_distr = extract_renovation_redistribuition(ren_map_in, ren_map_out, years_ots)
# dm_renov_distr.flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()

# Floor-area Renovated by envelope cat
# r_ct (t) = Ren-disr_ct(t) ren-rate(t) s(t-1)
dm_renov_cat = compute_floor_area_renovated(dm_stock_tot, dm_renovation, dm_renov_distr)
# dm_renov_cat.flatten().flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()
# note: renovated can be negative
# df_temp = dm_renov_cat.group_all("Categories2",inplace=False).write_df()

##################################################################
########### Floor area Stock ots by envelope categories ##########
##################################################################

# get stock per construction year in 2020
dm_stock_byconsper_2020 = extract_stock_byconstrperiod_2020()
# dm_temp = dm_stock_byconsper_2020.copy()
# dm_temp.normalise("Categories2")
# df = dm_temp.filter({"Country" : ["EU27"], "Categories1" : ["single-family-households"]}).write_df()
# df = df.melt(["Country","Years"])

# do the mapping
# source: https://www.bpie.eu/wp-content/uploads/2017/12/State-of-the-building-stock-briefing_Dic6.pdf
# this document does not have F, and color shaded are not super clear
# I will assume F is 0-1969, E is 1970-1979, D is 1980-1989 (these are the inefficient)
# And then is as per the doc, i.e., C is 1990-1999, B is 2000-2010, and A is >2010.
# To make the data conform, I will also merge B and A
construction_period_envelope_cat = {"F" : ['0-1945','1946-1969'],
                                    "E" : ['1970-1979'],
                                    "D" : ['1980-1989'],
                                    "C" : ['1990-1999'],
                                    "B" : ['2000-2010', '2011-now']}
dm_stock_cat_2020 = dm_stock_byconsper_2020.groupby(construction_period_envelope_cat, 
                                dim='Categories2')

# Check
dm_temp = dm_stock_cat_2020.groupby({"B" : ["B"], "C" : ["C"], "D": ["D"],
                                      ">D" : ["F","E"]}, "Categories2", inplace=False)
dm_temp.filter({"Country" : ["EU27"], "Categories1" : ['single-family-households', 'multi-family-households']}, inplace=True)
dm_temp.group_all("Categories1")
dm_temp.normalise("Categories1")
df = dm_temp.write_df()
df = df.melt(["Country","Years"])

# add other years as missing
years_new = dm_stock_tot.col_labels["Years"].copy()
years_new.remove(2020)
dm_stock_cat = dm_stock_cat_2020.copy()
dm_stock_cat.add(np.nan, "Years", years_new, dummy=True)
dm_stock_cat.sort("Years")
dm_stock_cat.rename_col('Czechia','Czech Republic',"Country")

# Stock by envelope cat
# s_{c,t}(t-1) &= s_{c,t}(t) - n_{c,t}(t) - r_{c,t}(t) + w_{c,t}(t)
dm_all = compute_stock_area_by_cat(dm_stock_cat, dm_new_cat, dm_renov_cat, dm_waste_cat, dm_stock_tot)

# Checks
# dm_all.filter({"Country" : ["EU27"], "Variables" : ["bld_floor-area_stock"]}).flatten().flatten().datamatrix_plot() 
# dm_stock_cat.filter({"Country" : ["EU27"], 
#                 "Variables" : ["bld_floor-area_stock"], 
#                 "Categories1" : ["single-family-households","multi-family-households"]}).group_all("Categories1",inplace=False).flatten().datamatrix_plot(stacked=True) 
# df_temp = dm_all.write_df()
# df_temp = pd.melt(df_temp, id_vars = ["Country","Years"])
# df_temp = df_temp.loc[df_temp["value"]<0,:]

#################################################
########### U Value - Fixed assumption ##########
#################################################

# Source: https://www.bpie.eu/wp-content/uploads/2017/12/State-of-the-building-stock-briefing_Dic6.pdf
envelope_cat_u_value = {"F" : (2.07 + 1.95)/2, 
                        "E" : 1.74,
                        "D" : 1.44, 
                        "C" : 1.2,
                        "B" : (0.89 + 0.49)/2}
cdm_u_value = ConstantDataMatrix(col_labels={'Variables': ['bld_u-value'],
                                             'Categories1': dm_all.col_labels["Categories1"],
                                             'Categories2': ['B', 'C', 'D', 'E', 'F']},
                                 units={'bld_u-value': 'W/m2K'})
arr = np.zeros((len(cdm_u_value.col_labels['Variables']), len(cdm_u_value.col_labels['Categories1']),
                len(cdm_u_value.col_labels['Categories2'])))
cdm_u_value.array = arr
idx = cdm_u_value.idx
for key in envelope_cat_u_value.keys():
    cdm_u_value.array[...,idx[key]] = envelope_cat_u_value[key]
dm_u_value = cdm_to_dm(cdm_u_value, dm_stock_cat.col_labels["Country"], ["All"])

#############################################################
########### Surface to Floorarea factor - Constant ##########
#############################################################

# Source: https://www.episcope.eu/downloads/public/docs/report/TABULA_FinalReport_AppendixVolume.pdf
# Page 131 Table 62
surface_to_floorarea = {'single-family-households': np.round(0.8+0.18+1.17+0.72,1), 
                        'multi-family-households': np.round(0.36+0.22+0.78+0.36,1)}

# Assuming the rest has the same of apartment blocks
apart_blocks = 0.22+0.22+0.64+0.22
others = ['education', 'health','hotels', 'offices', 'other', 'trade']
for o in others:
    surface_to_floorarea[o] = apart_blocks

# Put in cdm
cdm_s2f = ConstantDataMatrix(col_labels={'Variables': ['bld_surface-to-floorarea'],
                                        'Categories1': dm_all.col_labels["Categories1"]})
arr = np.zeros((len(cdm_s2f.col_labels['Variables']), len(cdm_s2f.col_labels['Categories1'])))
cdm_s2f.array = arr
idx = cdm_s2f.idx
for cat, val in surface_to_floorarea.items():
    cdm_s2f.array[idx['bld_surface-to-floorarea'], idx[cat]] = val
cdm_s2f.units["bld_surface-to-floorarea"] = "%" 
dm_s2f = cdm_to_dm(cdm_s2f, dm_stock_cat.col_labels["Country"], ["All"])

# TODO: take only heating space and not heating water

###########################################
########### Heating Technologies ##########
###########################################

dm_heating_mix = extract_heating_mix()
dm_heating_cat = dm_heating_mix.groupby(construction_period_envelope_cat, dim='Categories2', aggregation="mean")

# Put zero for years where building was not present yet
for y in range(1990,2010+1):
    dm_heating_cat[:,y,:,:,"B",:] = 0

# check
# dm_heating_cat.group_all("Categories3",inplace=False).array
# dm_heating_cat.filter({"Country" : ["EU27"], "Categories1" : ['multi-family-households', 'single-family-households']}).flatten().flatten().flatten().datamatrix_plot(stacked=True)

#########################################
########### Heating Efficiency ##########
#########################################

file = '../Europe/data/databases_full/JRC/JRC-IDEES-2021_Residential_EU27.xlsx'
sheet_name = 'RES_hh_eff'
dm_heating_eff = extract_heating_efficiency(file, sheet_name, years_ots)
# df = dm_heating_eff.write_df()

dm_heating_eff_cat = compute_heating_efficiency_by_archetype(dm_heating_eff, dm_all, envelope_cat_new,
                                                             categories=dm_stock_cat.col_labels['Categories2'])
# TODO: in the function above, instead of dm_all Paola is using dm_stock_cat, to be understood why

# check
# dm_heating_eff_cat.filter({"Country" : ["EU27"]}).flatten().flatten().datamatrix_plot()
# TODO: for oil we have it also before the start of the envelope, see if to keep it like this or fix it (it
# should not affect the results).

#########################################################
###################### MAKE PICKLE ######################
#########################################################

#########################################
#####   INTERFACE: LFS to BLD     #######
#########################################

file = '../../../data/datamatrix/lifestyles.pickle'
with open(file, 'rb') as handle:
    DM_lifestyles = pickle.load(handle)

dm_pop_ots = DM_lifestyles['ots']['pop']['lfs_population_'].filter({"Country" : ["EU27"]})
dm_pop_fts = DM_lifestyles['fts']['pop']['lfs_population_'][1].filter({"Country" : ["EU27"]})
dm_pop_ots.append(dm_pop_fts, dim='Years')
DM_interface_lfs_to_bld = {'pop': dm_pop_ots}

file = '../../../data/interface/lifestyles_to_buildings.pickle'
my_pickle_dump(DM_new = DM_interface_lfs_to_bld, local_pickle_file=file)


#########################################
#####   INTERFACE: CLM to BLD     #######
#########################################

file = '../../../data/datamatrix/climate.pickle'
with open(file, 'rb') as handle:
    DM_clm = pickle.load(handle)

dm_clm_ots = DM_clm['ots']['temp']['bld_climate-impact-space'].filter({"Country" : ["EU27"]})
dm_clm_fts = DM_clm['fts']['temp']['bld_climate-impact-space'][1].filter({"Country" : ["EU27"]})
dm_clm_ots.append(dm_clm_fts, dim='Years')
DM_interface_clm_to_bld = {'cdd-hdd': dm_clm_ots}

file = '../../../data/interface/climate_to_buildings.pickle'
my_pickle_dump(DM_new=DM_interface_clm_to_bld, local_pickle_file=file)

###########################################
#####  LOAD OLD AND MAKE DM_buildings #####
###########################################

file = '../../../data/datamatrix/lifestyles.pickle'
with open(file, 'rb') as handle:
    DM_lifestyles_old = pickle.load(handle)

file = '../../../data/datamatrix/buildings.pickle'
with open(file, 'rb') as handle:
    DM_bld = pickle.load(handle)
    
DM_buildings = {'ots': dict(), 'fts': dict(), 'fxa': dict(), 'constant': dict()}

############################################
#####  Calibration for heating energy  #####
############################################

dm_temp = DM_bld['fxa']['heating-energy-calibration'].filter({"Country" : ["Switzerland"]})
dm_temp.rename_col("Switzerland","EU27","Country")
country_missing = dm_heating_eff_cat.col_labels["Country"].copy()
country_missing.remove("EU27")
for c in country_missing:
    dm_temp.add(dm_temp["EU27",...], dim='Country', col_label=c)
dm_temp.sort("Country")
DM_buildings['fxa']['heating-energy-calibration'] = dm_temp.copy()
# TODO: make values for EU27, for the moment this is the same of CH. See with Paola about post processing.

#########################################
#####  FLOOR INTENSITY - SPACE/CAP  #####
#########################################

dm_space_cap = recompute_floor_area_per_capita(dm_all, dm_pop)
dm_lfs_household_size = get_household_size_eustat()
dm_lfs_household_size.drop("Country","United Kingdom")
dm_space_cap.append(dm_lfs_household_size, dim='Variables')

DM_buildings['ots']['floor-intensity'] = dm_space_cap.copy()
linear_fitting(dm_space_cap, years_fts)
DM_buildings['fts']['floor-intensity'] = dict()
for lev in range(4):
    lev = lev + 1
    DM_buildings['fts']['floor-intensity'][lev] = dm_space_cap.filter({'Years': years_fts})
    
# # check
# dm = DM_buildings['ots']["floor-intensity"]
# dm.append(DM_buildings['fts']['floor-intensity'][1], "Years")
# dm.filter({"Country" : ["EU27"]}).datamatrix_plot()

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
    
# # check
# variab = "heatcool-behaviour"
# dm = DM_buildings['ots'][variab]
# dm.append(DM_buildings['fts'][variab][1], "Years")
# dm.filter({"Country" : ["EU27"], "Categories1" : ['multi-family-households', 'single-family-households']}).flatten().flatten().datamatrix_plot()


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

# # check
# variab = "bld_type"
# dm = DM_buildings['fxa'][variab]
# dm.filter({"Country" : ["EU27"]}).flatten().flatten().datamatrix_plot(stacked=True)
# dm = DM_buildings['ots']["building-renovation-rate"]["bld_building-mix"]
# dm.append(DM_buildings['fts']["building-renovation-rate"]["bld_building-mix"][1], "Years")
# dm.filter({"Country" : ["EU27"], "Categories1" : ['multi-family-households', 'single-family-households']}).flatten().flatten().datamatrix_plot(stacked=True)

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

# # check
# dm = DM_buildings['ots']["building-renovation-rate"]["bld_renovation-rate"]
# dm.append(DM_buildings['fts']["building-renovation-rate"]["bld_renovation-rate"][1], "Years")
# dm.filter({"Country" : ["EU27"], "Categories1" : ['multi-family-households', 'single-family-households']}).flatten().datamatrix_plot()
# dm = DM_buildings['ots']["building-renovation-rate"]["bld_renovation-redistribution"]
# dm.append(DM_buildings['fts']["building-renovation-rate"]["bld_renovation-redistribution"][1], "Years")
# dm.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

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
first_bld = {'F': 1900, 'E': 1970, 'D': 1980, 'C': 1990, 'B': 2000}
dm_age = compute_building_age(dm_stock_cat, years_fts, first_bld)
DM_buildings['fxa']['bld_age'] = dm_age

# # check
# dm = DM_buildings['ots']['building-renovation-rate']['bld_demolition-rate']
# dm.append(DM_buildings['fts']['building-renovation-rate']['bld_demolition-rate'][1], "Years")
# dm.filter({"Country" : ["EU27"], "Categories1" : ['multi-family-households', 'single-family-households']}).flatten().datamatrix_plot()
# dm = DM_buildings['fxa']["bld_age"]
# dm.filter({"Country" : ["EU27"], "Categories1" : ['multi-family-households', 'single-family-households']}).flatten().flatten().datamatrix_plot()

#########################################
#####          U-VALUE            #######
#########################################

DM_buildings['fxa']['u-value'] = dm_u_value

# # Check
# dm = DM_buildings['fxa']['u-value']
# df = dm.filter({"Country" : ["EU27"], "Categories1" : ['multi-family-households', 'single-family-households']}).write_df()

#########################################
#####       SURFACE-2-FLOOR       #######
#########################################

DM_buildings['fxa']['surface-to-floorarea'] = dm_s2f

# # Check
# dm = DM_buildings['fxa']['surface-to-floorarea']
# df = dm.filter({"Country" : ["EU27"], "Categories1" : ['multi-family-households', 'single-family-households']}).write_df()

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

# # check
# dm = DM_buildings['ots']['heating-technology-fuel']['bld_heating-technology'].copy()
# dm.append(DM_buildings['fts']['heating-technology-fuel']['bld_heating-technology'][1], "Years")
# dm.filter({"Country" : ["EU27"], "Categories1" : ['multi-family-households', 'single-family-households']}).flatten().flatten().flatten().datamatrix_plot(stacked=True)

############################################
######       HEATING EFFICIENCY       ######
############################################

dm_heating_eff = dm_heating_eff_cat.copy()
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

# # check
# dm = DM_buildings['ots']["heating-efficiency"]
# dm.append(DM_buildings['fts']['heating-efficiency'][1], "Years")
# dm.filter({"Country" : ["EU27"]}).flatten().flatten().datamatrix_plot()

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

# Electricity emission factors
col_dict = {
    'Country': dm_heating_eff.col_labels["Country"],
    'Years': years_ots+years_fts,
    'Variables': ['bld_CO2-factor'],
    'Categories1': ['electricity']
}
dm_elec = DataMatrix(col_labels=col_dict, units={'bld_CO2-factor': 'kt/TWh'})
arr_elec = np.zeros((len(dm_heating_eff.col_labels["Country"]), 40, 1, 1))
idx = dm_elec.idx
arr_elec[:, idx[1990]: idx[2023]+1, 0, 0] = 112
arr_elec[:, idx[2025]: idx[2050], 0, 0] = np.nan
arr_elec[:, idx[2050], 0, 0] = 0
dm_elec.array = arr_elec
dm_elec.fill_nans(dim_to_interp="Years")
DM_buildings['fxa']['emission-factor-electricity'] = dm_elec

##########################
#####     FILTER    ######
##########################

def filter_nested_structure(data, countries = ["EU27"], bld_types = ['multi-family-households', 'single-family-households']):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, DataMatrix):
                value = value.filter({"Country": countries})
                if len(value.dim_labels) > 3:
                    if "education" in value.col_labels["Categories1"]:
                        value = value.filter({"Categories1": bld_types})
                data[key] = value
            else:
                filter_nested_structure(value)

filter_nested_structure(DM_buildings["ots"])
filter_nested_structure(DM_buildings["fts"])
filter_nested_structure(DM_buildings["fxa"])


########################
#####     SAVE    ######
########################

file = '../../../data/datamatrix/buildings.pickle'
my_pickle_dump(DM_buildings, file)





