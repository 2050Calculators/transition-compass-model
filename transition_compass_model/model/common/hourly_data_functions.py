import pandas as pd
import os
import numpy as np
from model.common.data_matrix_class import DataMatrix
from model.common.io_database import dm_lever_dict_from_df


def hourly_data_reader(file, years_setting, lever=None, dict_ots={}, dict_fts={}):

    # This function reads hourly files for baseyear and fts years and puts them in a datamatrix of shape with 3
    # categories: weeks (from w00 to w53), days (from day d1 to d7), hours (from h00 to h23). Where d1 to d7 are the
    # weekdays from Monday to Sunday.
    # The file in input is expected to be a .csv file with columns 'Country', 'Years', 'var-name_xxx[unit]', ...,
    # where xxx is usually a date. The xxx data are not read, we assume that the columns are ordered from the first
    # to the last hour of the year.

    def remove_leap_day(date_range, year):
        # Remove the 29th of February
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            # Generate a boolean mask to identify February 29th
            is_leap_day = (date_range.month == 2) & (date_range.day == 29)
            # Remove those dates from the date range
            date_range = date_range[~is_leap_day]

        return date_range

    def generate_hourly_strings(year):
        # Create a date range for every hour in the given year
        date_range = pd.date_range(start=f'{year}-01-01', end=f'{year + 1}-01-01', freq='H', inclusive='left')
        # Remove leap day
        date_range = remove_leap_day(date_range, year)
        # Create a Series for date range to access datetime-related attributes
        date_series = date_range.to_series()
        # Get week number in %W style (week 0 if the year doesn't start on a Monday)
        week_number = date_series.dt.strftime('%W').astype(int)
        # Get day of the week (Monday = 0, Sunday = 6)
        day_of_week = date_series.dt.weekday
        # Create formatted string
        formatted_date = (
            'w' + week_number.astype(str).str.zfill(2) +  # Week number in %W style
            '_d' + (day_of_week + 1).astype(str) +  # Add 1 to match Monday (1), Sunday (7)
            '_h' + date_series.dt.hour.astype(str).str.zfill(2)  # Hour
        )
        return formatted_date.to_list()

    def check_columns(df, lever):
        # lever should not have 'lever_' in the name
        if lever is not None and 'lever_' in lever:
            raise ValueError('Remove "lever_" from levername')
        # Check column structure
        fixed_cols = [col for col in df.columns if "[" not in col]
        # Check that 'Country' and 'Years' are in fixed cols
        if {'Country', 'Years'} - set(fixed_cols):
            raise ValueError(f'Columns Country and Years are expected in {file}')
        # Check if there is a lever col
        lever_col = set(fixed_cols) - {'Country', 'Years'}
        if len(lever_col) == 1:
            lever_str = lever_col.pop()
            lever_str = lever_str.replace('lever_', '')
            # If there is a lever in the file but not in the call or they don't match
            if lever is None or lever_str != lever:
                raise ValueError(f'You have the lever column "{lever_str}" in {file}, use lever="{lever_str}" in the call.')
            else:
                df.rename(columns={'lever_'+lever: lever}, inplace=True)

        elif len(lever_col) > 1:
            raise ValueError(f'The columns {lever_col} in file {file} do not have the right pattern.')

        return df

    def reorder_columns(df, lever):
        # Make sure column order is correct
        if lever is not None:
            # Create a list of columns after removing the ones to move
            remaining_cols = df.columns.difference(['Country', 'Years', lever], sort=False)
            # Create a new column order with the selected columns at the beginning
            new_order = ['Country', 'Years', lever] + list(remaining_cols)
            # Reorder the DataFrame based on the new order
            df = df[new_order]
        else:
            # Create a list of columns after removing the ones to move
            remaining_cols = df.columns.difference(['Country', 'Years'], sort=False)
            # Create a new column order with the selected columns at the beginning
            new_order = ['Country', 'Years'] + list(remaining_cols)
            # Reorder the DataFrame based on the new order
            df = df[new_order]

        return df

    def remove_duplicates(df, file, lever):
        # Remove duplicates
        len_init = len(df)
        if lever is None:
            df = df.drop_duplicates(subset=['Country', 'Years'])
        else:
            df = df.drop_duplicates(subset=['Country', 'Years', lever])
        if len(df) - len_init < 0:
            print(f"Duplicates found in: {file}, use .duplicated on dataframe to check which lines are repeated")
        return df

    def df_new_hourly_cols(df, year, lever):
        all_countries = set(df.Country)
        if year not in set(df.Years):
            raise ValueError(f'Year {year} missing from database')
        df_year = df.loc[df['Years'] == year]  # Keep only the year you want
        if len(all_countries - set(df_year.Country)):
            missing_countries = all_countries - set(df_year.Country)
            raise ValueError(f'Year {year} is missing the countries: {missing_countries}')

        # Extract hourly column string
        hourly_col_str = generate_hourly_strings(year)
        # Extract variable from last column
        col_tmp = df_year.columns[-1]
        last_underscore_index = col_tmp.rfind('_')
        var = col_tmp[0:last_underscore_index]
        # Extract unit from last column
        last_L_bracket_index = col_tmp.rfind('[')
        last_R_bracket_index = col_tmp.rfind(']')
        unit = col_tmp[last_L_bracket_index + 1: last_R_bracket_index]
        new_columns = [var + '_' + col + '[' + unit + ']' for col in hourly_col_str]
        if lever is not None:
            df_year.columns = ['Country', 'Years', lever] + new_columns
        # If there is not a lever
        else:
            df_year.columns = ['Country', 'Years'] + new_columns
        return df_year

    def df_to_dm_hourly_w_lever(df, baseyear, years_fts, lever, dict_ots, dict_fts):
        # Read ots (only baseyear)
        df_ots = df_new_hourly_cols(df, baseyear, lever).copy()
        # Read fts
        i = 0
        for year in years_fts:
            df_year = df_new_hourly_cols(df, year, lever)
            if i == 0:
                df_fts = df_year
            else:
                # Concatenate with `join='outer'` to retain all columns
                df_fts = pd.concat([df_fts, df_year], axis=0, ignore_index=True, join='outer')
            i = i + 1

        dict_lever = dm_lever_dict_from_df(df_fts, lever, num_cat=3)
        # Remove 'ots_' and drop lever
        df_ots.drop(columns=[lever], inplace=True)
        # Sort by country years
        df_ots.sort_values(by=['Country', 'Years'], axis=0, inplace=True)
        dm_ots = DataMatrix.create_from_df(df_ots, num_cat=3)

        # Make sure dm_ots has all on the weeks (from w00 to w53)
        dm_fts = dict_lever[1]
        missing_weeks = list(set(dm_fts.col_labels['Categories1']) - set(dm_ots.col_labels['Categories1']))
        dm_ots.add(np.nan, dim='Categories1', col_label=missing_weeks, dummy=True)
        dm_ots.sort(dim='Categories1')

        dict_ots[lever] = dm_ots
        dict_fts[lever] = dict_lever

        return dict_ots, dict_fts


    def df_to_dm_hourly_fxa(df, baseyear, years_fts):
        keep_years = [baseyear] + years_fts

        i = 0
        for year in keep_years:
            df_year = df_new_hourly_cols(df, year, lever)
            if i == 0:
                df_all = df_year
            else:
                # Concatenate with `join='outer'` to retain all columns
                df_all = pd.concat([df_all, df_year], axis=0, ignore_index=True, join='outer')
            i = i + 1

        dm = DataMatrix.create_from_df(df_all, num_cat=3)

        return dm



    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    # If extension was missing in file, add it
    if '.' not in file:
        file = file + '.csv'
    file_path = '../../_database/data/csv/' + file
    file_path = os.path.join(current_file_directory, file_path)

    if '.csv' in file:
        df = pd.read_csv(file_path, sep=';')
    else:
        raise ValueError(f'Only .csv file allowed')

    # Check if df columns are good
    df = check_columns(df, lever)
    df = reorder_columns(df, lever)
    df = remove_duplicates(df, file, lever)

    # Keep only baseyear and fts
    baseyear = years_setting[1]
    lastyear = years_setting[2]
    step_fts = years_setting[3]
    years_fts = list(
        np.linspace(start=baseyear + step_fts, stop=lastyear, num=int((lastyear - baseyear) / step_fts)).astype(int))

    if lever is None:
        dm = df_to_dm_hourly_fxa(df, baseyear, years_fts)
        return dm
    else:
        dict_ots, dict_fts = df_to_dm_hourly_w_lever(df, baseyear, years_fts, lever, dict_ots, dict_fts)
        return dict_ots, dict_fts


