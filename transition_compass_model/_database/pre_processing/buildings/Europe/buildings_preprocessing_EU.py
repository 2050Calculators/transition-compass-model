import pandas as pd
from model.common.data_matrix_class import DataMatrix
import numpy as np
from model.common.auxiliary_functions import linear_fitting, linear_forecast_BAU, moving_average, create_years_list
from model.common.auxiliary_functions import eurostat_iso2_dict
from model.common.io_database import read_database_to_dm, edit_database, update_database_from_dm
import eurostat
import json
from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat
from scipy.optimize import minimize

EU27_cntr_list = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland',
                  'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg',
                  'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden']


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
    # Duplicate 2014 values for all years
    for y in list(range(1990, 2022)):
        if y != 2014:
            # Take first year value
            new_array = dm_rr.array[:, 0, ...]
            dm_rr.add(new_array, dim='Years', col_label=[y], unit='%')
    dm_rr.sort(dim='Country')
    return dm_rr


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


def create_ots_years_list(years_setting, astype):
    startyear: int = years_setting[0]  # Start year is argument [0], i.e., 1990
    baseyear: int = years_setting[1]  # Base/Reference year is argument [1], i.e., 2015
    lastyear: int = years_setting[2]  # End/Last year is argument [2], i.e., 2050
    step_fts = years_setting[3]  # Timestep for scenario is argument [3], i.e., 5 years
    years_ots = list(
        np.linspace(start=startyear, stop=baseyear, num=(baseyear - startyear) + 1).astype(int).astype(astype))
    return years_ots


def get_uvalue_new_stock0(years_ots):
    # Gets the u-value for the new buildings, as well as the u-value of the building stock at t=baseyear
    # Load u-values by element, construction-period, building type
    df_u_wall, df_u_window, df_u_roof, df_u_ground = sub_routine_get_uvalue_by_element()
    # Weight u-values of element by area of element
    df_uvalue = compute_weighted_u_value(df_u_wall, df_u_window, df_u_roof, df_u_ground)
    # From df_uvalue keep only new built for construction period > 1990
    df_uvalue = extract_u_value(df_uvalue)
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
    dm_uvalue_stock0 = dm_uvalue.filter({'Years': [972, 1957, 1974, 1984]})
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


def get_rooms_cap_eustat(dict_iso2, years_ots):

    # Extracts the number of rooms per capita for the period available (2003-2023)
    # The data are extrapolated with linear fitting until 1990
    ##### Extract rooms per capita
    filter = {'geo\TIME_PERIOD': list(dict_iso2.keys()),
              'building': ['HOUSE', 'FLAT'],
              'tenure': 'TOTAL'}
    mapping_dim = {'Country': 'geo\TIME_PERIOD',
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


def get_household_area_eustat(EU27_cntr_list, dict_iso2):

    ##### Extract average m2/household
    filter = {'geo\TIME_PERIOD': list(dict_iso2.keys()),
              'deg_urb': 'TOTAL',
              'hhcomp': 'TOTAL'}
    mapping_dim = {'Country': 'geo\TIME_PERIOD',
                   'Variables': 'deg_urb'}
    dm_size = get_data_api_eurostat('ilc_lvho31', filter, mapping_dim, 'm2/household', years_ots)

    # Fill in Ireland data
    # https://www.finfacts-blog.com/2021/04/irish-house-size-climate-change-and.html#:~:text=The%20average%20household%20size%20in,living%20space%20is%2036.5m².
    idx = dm_size.idx
    dm_size.array[idx['Ireland'], idx[2023], :] = 112
    print('Ireland  average floor area MISSING and set to 112 m2, see code for data source')

    dm_size.drop(dim='Years', col_label=2020)
    dm_size.rename_col('TOTAL', 'lfs_floor-area', dim='Variables')

    return dm_size


def estimate_stock_res_from_average_room_size(dm_rooms, dm_area_2020, dm_pop, dm_pop_bld_type):
    # 1) lfs_rooms-cap [rooms/cap] x lfs_pop-by-bld-type [habitants] = lfs_rooms [rooms] (by bld type)
    # 2) bld_floor-area_2020 [m2] / lfs_rooms [rooms] (t=2020) = lfs_avg-room-size [m2/rooms] (t=2020)
    # 3) bld_floor-area_stock_tmp [m2] = lfs_rooms [rooms] x lfs_avg-room-size [m2/rooms]
    # (by bld type, value applied to all years)

    # lfs_pop-by-bld-type = pop_bld_type_share x dm_pop
    dm_pop_bld_type = dm_pop_bld_type.flatten()
    dm_pop_bld_type.drop(dim='Years', col_label=[2023])
    vars = dm_pop_bld_type.col_labels['Variables']
    dm_pop_bld_type.append(dm_pop.filter({'Years': dm_pop_bld_type.col_labels['Years']}, inplace=False), dim='Variables')
    for var in vars:
        var_out = str.replace(var, '_share', '')
        dm_pop_bld_type.operation(var, '*', 'lfs_population_total', out_col=var_out, unit='inhabitants')
    dm_pop_bld_type.drop(dim='Variables', col_label='lfs_population_total')
    dm_pop_bld_type.deepen()

    dm_all = dm_pop_bld_type.filter({'Variables': ['lfs_pop-by-bld-type']})
    dm_all.append(dm_rooms.filter({'Years': dm_all.col_labels['Years']}), dim='Variables')
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


def compute_floor_area_per_capita(dm_room_size_2020, dm_rooms):
    # 3) bld_floor-area_stock_tmp [m2] = lfs_rooms [rooms] x lfs_room-size [m2/rooms]
    # m2/room [2020] * room/cap = m2/cap
    dm_room_size_2020.sort('Country')
    dm_rooms.sort('Country')

    idx_2 = dm_room_size_2020.idx
    idx = dm_rooms.idx
    arr_m2_cap = dm_room_size_2020.array[:, :, idx_2['lfs_room-size'], ...] \
                 * dm_rooms.array[:, :, idx['lfs_rooms-cap'], ...]
    dm_rooms.add(arr_m2_cap, dim='Variables', col_label='lfs_floor-intensity_space-cap', unit='m2/cap')
    dm_floor = dm_rooms.filter({'Variables': ['lfs_floor-intensity_space-cap']})

    return dm_floor


def estimate_demolition_rate(dm_area_cap_res, dm_area_2020, dm_pop, dm_area_new):

    dm_area_cap_res.drop('Years', 2023)
    dm_area_new.drop('Years', 2023)

    # Sort country/years for all dm
    dm_area_cap_res.sort('Country')
    dm_area_cap_res.sort('Years')
    dm_area_2020.sort('Country')
    dm_area_2020.sort('Years')
    dm_pop.sort('Country')
    dm_pop.sort('Years')
    dm_area_new.sort('Country')
    dm_area_new.sort('Years')

    # Equations to use
    # stock(t) = stock(t-1) + n(t) - w(t)
    # dem-rate(t) = w(t)/s(t-1)

    if dm_area_2020.col_labels['Country'] != dm_area_cap_res.col_labels['Country']:
        raise ValueError('Country lists not matching')
    dm_area_share_res_2020 = dm_area_2020.filter_w_regex({'Categories1': '.*households',
                                                          'Variables': 'bld_floor-area_share'}, inplace=False)
    dm_area_share_res_2020.group_all(dim='Categories1', inplace=True)
    # Compute m2/cap by bld type
    # m2/cap_by bld type = m2/cap_res / %_res x %_by bld type
    idx = dm_area_2020.idx
    arr_area_cap = dm_area_cap_res.array[:, :, :, np.newaxis]/dm_area_share_res_2020.array[:, 0, np.newaxis, :, np.newaxis] \
                   * dm_area_2020.array[:, 0, np.newaxis, idx['bld_floor-area_share'], np.newaxis, :]

    # Compute m2 total per bld type
    dm_pop.filter({'Country': dm_area_2020.col_labels['Country'], 'Years': dm_area_cap_res.col_labels['Years']})
    arr_area = arr_area_cap[:, :, :, :]*dm_pop.array[:, :, :, np.newaxis]
    dm_area = dm_area_new.copy()
    dm_area.add(arr_area, dim='Variables', col_label='bld_floor-area_stock', unit='m2')

    # w(t) = stock(t-1) + n(t) - stock(t)
    dm_area.lag_variable('bld_floor-area_stock', shift=1, subfix='_tm1')
    dm_area.operation('bld_floor-area_stock_tm1', '+', 'bld_floor-area_new', out_col='tmp', unit='m2')
    dm_area.operation('tmp', '-', 'bld_floor-area_stock', out_col='bld_floor-area_waste', unit='m2')

    # dem-rate(t) = w(t)/s(t-1)
    dm_area.operation('bld_floor-area_waste', '/', 'bld_floor-area_stock_tm1', out_col='bld_demolition-rate', unit='%')

    dm_area.datamatrix_plot({'Variables': 'bld_demolition-rate'})

    return


def solve_optimisation_problem(dm_area_res, country, cat):

    idx = dm_area_res.idx
    ### Set initial guesses to correspond to the objective functions
    stock_init = dm_area_res.array[idx[country], :, idx['bld_floor-area_stock'], idx[cat]]
    new_init = dm_area_res.array[idx[country], :, idx['bld_floor-area_new'], idx[cat]]
    dem_rates_init = dm_area_res.array[idx[country], :, idx['bld_dem-rates'], idx[cat]]
    # Equations to use
    # stock(t) = stock(t-1) + n(t) - w(t)
    # dem-rate(t) = w(t)/s(t-1)
    # --> dem-rate(t) = (stock(t-1) + n(t) - stock(t))/s(t-1)

    # Define the objective function to minimize, with normalization
    def objective(variables, t_values):
        stock = variables[:len(t_values)]  # First part is stock(t)
        new = variables[len(t_values):2 * len(t_values)]  # Second part is new(t)

        # Calculate the demolition rate from the stock and new variables
        computed_dem_rate = (stock[-1] + new[1:] - stock[1:]) / stock[-1]

        # Calculate the normalized residuals
        residual_stock = np.linalg.norm(stock - stock_init) / np.linalg.norm(stock_init)  # Normalized stock residual
        residual_new = np.linalg.norm(new - new_init) / np.linalg.norm(new_init)  # Normalized new residual
        residual_dem_rate = np.linalg.norm(computed_dem_rate - dem_rates_init[1:]) / np.linalg.norm(dem_rates_init[1:])  # Normalized dem-rate residual

        tot_residual = residual_stock + residual_new + residual_dem_rate

        print('Residual', tot_residual)
        # Return the total residual sum
        return tot_residual

    # Define the constraint: Ensure that computed_dem_rate(t) >= 0
    def dem_rate_constraint(variables, t_values):
        stock = variables[:len(t_values)]  # First part is stock(t)
        new = variables[len(t_values):2 * len(t_values)]  # Second part is new(t)

        # Calculate the demolition rate
        computed_dem_rate = (stock[:-1] + new[1:] - stock[1:]) / stock[:-1]

        # Return the demolition rates to ensure all are non-negative
        return computed_dem_rate  # We want this to be >= 0

    # Example time range (years)
    t_values = np.array(dm_area_res.col_labels['Years'])  # Years 1 to 10

    # Initial guess for the variables (stock and new for each year)
    initial_guess = np.ones(len(t_values)*2)  # Stock and new combined into one vector

    # Define the constraint in the format for minimize
    constraints = {'type': 'ineq', 'fun': dem_rate_constraint, 'args': (t_values,)}  # 'ineq' means >= 0

    # Solve the optimization problem using minimize with the constraint
    result = minimize(objective, initial_guess, args=(t_values,), constraints=constraints, method='L-BFGS-B', options={'ftol': 1e-4})

    # Extract the optimized variables
    optimized_stock = result.x[:len(t_values)]
    optimized_new = result.x[len(t_values):2 * len(t_values)]

    # Calculate the demolition rate based on optimized stock and new
    optimized_dem_rate = (optimized_stock[:-1] + optimized_new[1:] - optimized_stock[1:]) \
                         / optimized_stock[-1]

    dm_area_res.array[idx[country], :, idx['bld_floor-area_stock'], idx[cat]] = optimized_stock
    dm_area_res.array[idx[country], :, idx['bld_floor-area_new'], idx[cat]] = optimized_new
    dm_area_res.array[idx[country], 1:, idx['bld_dem-rates'], idx[cat]] = optimized_dem_rate

    return


def estimate_demolition_rates_fix_stock(dm_area_stock_tmp, dm_area_new):
    # Filter & join residential households data
    dm_area_new.drop('Years', 2023)
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

    dm_area_out.lag_variable('bld_floor-area_stock', shift=1, subfix='_tm1')

    # EU27
    dm_area_out_EU27 = dm_area_out.groupby({'EU27': EU27_cntr_list}, dim='Country', inplace=False)
    dm_area_out.drop('Country', 'EU27')
    dm_area_out.append(dm_area_out_EU27, dim='Country')
    dm_area_out.sort('Country')
    idx = dm_area_out.idx
    # stock(t) - stock(t - 1) = new(t) - waste(t)
    waste_t = dm_area_out.array[idx['EU27'], :, idx['bld_floor-area_new'], :] - \
              ( dm_area_out.array[idx['EU27'], :, idx['bld_floor-area_stock'], :] \
              - dm_area_out.array[idx['EU27'], :, idx['bld_floor-area_stock_tm1'], :])
    dm_area_out.array[idx['EU27'], :, idx['bld_demolition-rates'], :] = \
        waste_t/dm_area_out.array[idx['EU27'], :, idx['bld_floor-area_stock_tm1'], :]

    dm_area_out.drop(dim='Variables', col_label=['bld_floor-area_stock_tm1'])

    return dm_area_out


def get_pop_by_bld_type(code_eustat, dict_iso2, years_ots):
    # code_eustat = 'ilc_lvho01'
    filter = {'deg_urb': ['TOTAL'], 'geo\TIME_PERIOD': dict_iso2.keys(), 'incgrp': ['TOTAL'],
              'building': ['FLAT', 'HOUSE'], 'freq': 'A'}
    mapping_dim = {'Country': 'geo\TIME_PERIOD', 'Variables': 'freq', 'Categories1': 'building'}
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


def recompute_floor_area_per_capita(dm_all, dm_pop):

    dm_floor_stock = dm_all.filter({'Variables': ['bld_floor-area_stock'],
                                    'Categories': {'single-family-households', 'multi-family-households'}}, inplace=False)

    # Computer m2/cap for lifestyles
    dm_floor_stock.group_all(dim='Categories1')
    dm_floor_stock.append(dm_pop, dim='Variables')

    dm_floor_stock.operation('bld_floor-area_stock', '/', 'lfs_population_total',
                             out_col='lfs_floor-intensity_space-cap', unit='m2/cap')

    dm_floor_stock.filter({'Variables': ['lfs_floor-intensity_space-cap']}, inplace=True)

    return dm_floor_stock


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


def forecast_floor_int_fts_from_ots(dm_floor_ots, years_ots, years_fts):

    dm_floor_fts_1 = linear_forecast_BAU(dm_floor_ots, start_t=2012, years_ots=years_ots, years_fts=years_fts)
    # level 2: 7.3% higher than 2015
    # level 3: 6% lower than 2015
    # level 4: 19% lower than 2015
    # Rates based on EUcalc
    dict_2050 = {2: 0.073, 3: -0.06, 4: -0.19}
    idx_o = dm_floor_ots.idx
    idx_b = dm_floor_fts_1.idx
    arr_2015 = dm_floor_ots.array[:, idx_o[2015], idx_o['lfs_floor-intensity_space-cap']]
    arr_2050_1 = dm_floor_fts_1.array[:, idx_b[2050], idx_b['lfs_floor-intensity_space-cap']]
    dict_fts = dict()
    for level in [2, 3, 4]:
        dm_floor_fts_lev = dm_floor_ots.copy()
        dm_floor_fts_lev.add(np.nan, dim='Years', col_label=years_fts, dummy=True)
        idx = dm_floor_fts_lev.idx
        # Get increase/reduction level for 2050 wrt 2015
        rate = 1 + dict_2050[level]
        arr_2050_tmp = arr_2015 * rate
        arr_2050_lev = np.minimum(arr_2050_tmp, arr_2050_1)
        for cntr in dm_floor_ots.col_labels['Country']:
            arr_2050_lev_cntr = np.maximum(arr_2050_lev[idx_b[cntr]], 25)
            dm_floor_fts_lev.array[idx[cntr], idx[2050], idx['lfs_floor-intensity_space-cap']] = arr_2050_lev_cntr
        # Fill nans
        dm_floor_fts_lev.fill_nans(dim_to_interp='Years')
        dict_fts[level] = dm_floor_fts_lev.filter({'Years': years_fts}, inplace=False)

    dict_fts[1] = dm_floor_fts_1

    return dict_fts


#######################
#### RUN ROUTINES #####
#######################
years_ots = create_years_list(start_year=1990, end_year=2023, step=1, astype=int)
years_fts = create_years_list(start_year=2025, end_year=2050, step=5, astype=int)

dict_iso2 = eurostat_iso2_dict()
dict_iso2.pop('CH')  # Remove Switzerland


# Load renovation rates
dm_rr = get_renovation_rate()
# Load U-value for new buildings()
dm_uvalue_new, dm_area_2020, dm_uvalue_stock0, dm_new_2 = get_uvalue_new_stock0(years_ots)
# Load new building floor area
dm_new_1 = get_new_building()
# Reconcile two new build area estimates
dm_area_new = estimate_floor_area(dm_new_1, dm_new_2, years_ots)
del dm_new_1, dm_new_2
# Compute total floor area from rooms/cap and average room size in 2020
dm_rooms = get_rooms_cap_eustat(dict_iso2, years_ots=years_ots)
# Share of population by building type
dm_pop_bld_type = get_pop_by_bld_type('ilc_lvho01', dict_iso2, years_ots)
# Load population data
dict_lfs, tmp = read_database_to_dm('lifestyles_population.csv', num_cat=0, level=0)
dm_pop = dict_lfs['pop'].filter({'Variables': ['lfs_population_total'], 'Country': list(dict_iso2.values())}, inplace=False)
del dict_lfs, tmp
# Compute floor-area stock (tmp) from avg room size and nb of rooms (sfh, mfh)
dm_residential_stock_tmp = estimate_stock_res_from_average_room_size(dm_rooms, dm_area_2020, dm_pop, dm_pop_bld_type)
dm_all_stock_tmp = estimate_stock_non_res(dm_area_2020, dm_residential_stock_tmp)
del dm_rooms, dm_pop_bld_type
# Determine demolition rate and adjust floor-area stock (sfh, mfh)
dm_all = estimate_demolition_rates_fix_stock(dm_all_stock_tmp, dm_area_new)
del dm_residential_stock_tmp, dm_area_new, dm_area_2020
# Floor area per capita and share of stock
dm_res_area_cap = recompute_floor_area_per_capita(dm_all, dm_pop)

# Update floor-intensity space-cap data
update_floor_intensity = False
if update_floor_intensity:
    file = 'lifestyles_floor-intensity.csv'
    years_ots = create_years_list(start_year=1990, end_year=2022, step=1, astype=int)
    # csv_database_reformat(file)
    update_database_from_dm(dm_res_area_cap, filename=file, lever='floor-intensity', level=0, module='lifestyles')
    # It uses EUcalc rates for level 2, 3, 4
    dict_dm_floor_int_fts = forecast_floor_int_fts_from_ots(dm_res_area_cap, years_ots, years_fts)
    for lev, dm_fts in dict_dm_floor_int_fts.items():
        update_database_from_dm(dm_fts, filename=file, lever='floor-intensity', level=lev, module='lifestyles')


# Let us load historical data for population and average m2/cap to compute the demolition-rate.

dm_rr.normalise('Categories1', keep_original=True)
dm_rr.flatten().datamatrix_plot({'Country': ['Austria']})
print('Hello')
