import pandas as pd
import os
import warnings
from model.common.data_matrix_class import DataMatrix
import numpy as np


def find_git_root():
    path = os.getcwd()

    while True:
        if os.path.isdir(os.path.join(path, '.git')):
            return path
        new_path = os.path.abspath(os.path.join(path, '..'))
        if new_path == path:
            # We've reached the root of the filesystem without finding a .git directory
            return None
        path = new_path


def read_database(filename, lever, folderpath="default", db_format=False, level='all'):
    # Reads csv file in database/data/csv and extracts it in df format with columns
    # "Country, Years, lever-name, variable-columns"
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    if folderpath == "default":
        folderpath = os.path.join(current_file_directory, "../../_database/data/csv/")
    file = folderpath + filename + '.csv'
    df_db = pd.read_csv(file, sep=";")


    # Remove duplicates
    len_init = len(df_db)
    df_db = df_db.drop_duplicates(subset=['geoscale', 'timescale', 'level', 'string-pivot', 'eucalc-name'])
    if len(df_db) - len_init < 0:
        print(f"Duplicates found in: {filename}, use .duplicated on dataframe to check which lines are repeated")

    if db_format:
        return df_db
    else:
        df_db_ots = (df_db.loc[(df_db["level"] == 0) & (df_db['lever'] == lever)]).copy()
        if level == 'all':
            df_db_fts = (df_db.loc[(df_db["level"] != 0) & (df_db['lever'] == lever)]).copy()
        else:
            df_db_fts = (df_db.loc[(df_db["level"] == level) & (df_db['lever'] == lever)]).copy()

        if (df_db_ots['string-pivot'] != 'none').any():
            df_ots = df_db_ots.pivot(index=['geoscale', 'timescale', 'level', 'string-pivot'], columns="eucalc-name",
                                     values='value')
        else:
            df_ots = df_db_ots.pivot(index=['geoscale', 'timescale', 'level'], columns="eucalc-name", values='value')
        df_ots.reset_index(inplace=True)
        if (df_db_fts['string-pivot'] != 'none').any():
            df_fts = df_db_fts.pivot(index=['geoscale', 'timescale', 'level', 'string-pivot'], columns="eucalc-name",
                                     values='value')
        else:
            df_fts = df_db_fts.pivot(index=['geoscale', 'timescale', 'level'], columns="eucalc-name", values='value')



        df_fts.reset_index(inplace=True)
        rename_cols = {'geoscale': "Country", 'timescale': 'Years', 'level': lever}
        df_ots.rename(columns=rename_cols, inplace=True)
        df_fts.rename(columns=rename_cols, inplace=True)

    return df_ots, df_fts


def edit_database(filename: str, lever: str, column: str, pattern, mode: str, level=None, filter_dict=None):
    # it edits the database either by renaming or removing strings in the database
    # it requires as input the 'filename' as a string, the 'lever' containing the lever name,
    # 'column' indicating the columns in the database that you want to edit, 'mode' is either 'remove' or 'rename',
    # if 'mode'=='remove', then 'pattern' is a regex pattern and the algorithm will remove the entire row
    # e.g. edit_database('lifestyles_population', 'pop', column='geoscale', pattern='Norway|Vaud', mode='remove')
    # it will remove Norway and Vaud from the country list
    # if 'mode'=='rename', then 'pattern' is a dictionary to replace the substring in key with the substring in value.
    # e.g. edit_database('lifestyles_population', 'pop', column='eucalc-name',
    #                    pattern={'population':'pop', 'lfs:'lifestyles'}, mode='rename')
    # if it find for example 'lfs_urban_population' in 'eucalc-name', this would become 'lifestyle_urban_population'
    # name-filter allows to rename only for a specific set of rows before applying the rename
    assert mode in ('rename', 'remove'), f"Invalid mode: {mode}, mode should be rename or remove"

    filename = filename.replace('.csv', '')  # drop .csv extension
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    folderpath = os.path.join(current_file_directory, "../../_database/data/csv/")
    file = folderpath + filename + '.csv'
    df_db = pd.read_csv(file, sep=";")
    df_db_lever = (df_db[df_db['lever'] == lever]).copy()
    df_db_other = (df_db[df_db['lever'] != lever]).copy()
    if mode == "rename":
        if column == "eucalc-name":
            col_to_rename = ["eucalc-name", "element", "item", "unit"]
        else:
            col_to_rename = [column]
        df_db_unchanged = pd.DataFrame()
        if filter_dict is not None:
            # allows to only filter a set of row before applying the rename
            filter_col = list(filter_dict.keys())[0]
            filter_pattern = filter_dict[filter_col]
            mask = df_db_lever[filter_col].astype(str).str.contains(filter_pattern)
            df_db_unchanged = df_db_lever.loc[~mask].copy()
            df_db_lever = df_db_lever.loc[mask].copy()
        for str1 in pattern:
            str2 = pattern[str1]
            for col in col_to_rename:
                df_db_lever[col] = df_db_lever[col].str.replace(str1, str2)
        if not df_db_unchanged.empty:
            df_db_lever = pd.concat([df_db_lever, df_db_unchanged], axis=0)
    if mode == "remove":
        if level is None:
            mask = df_db_lever[column].str.contains(pattern)
            df_db_lever = df_db_lever[~mask]
        # Remove line conditioned to level value (used to remove 2015 in fts)
        if level is not None:
            mask = (df_db_lever[column].astype(str).str.contains(pattern)) & (df_db_lever['level'] == level)
            df_db_lever = df_db_lever[~mask]
    df_db_new = pd.concat([df_db_lever, df_db_other], axis=0)
    df_db_new.sort_values(by=['geoscale', 'timescale'], axis=0, inplace=True)
    df_db_new.to_csv(file, sep=";", index=False)
    return


def change_unit_database(filename, target_col_pattern, new_unit):
    # Given the name of the csv file in the database and the pattern for the column (do not use .*),
    # it replaces the units in the unit column
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    folderpath = os.path.join(current_file_directory, "../../_database/data/csv/")
    file = folderpath + filename + '.csv'
    df_db = pd.read_csv(file, sep=";")
    mask = df_db['eucalc-name'].str.contains(target_col_pattern)
    # Define a regular expression to extract the unit between '[' and ']'
    df_db.loc[mask, 'eucalc-name'] = df_db.loc[mask, 'eucalc-name'].replace(to_replace=r'\[.*\]', value="["+new_unit+"]", regex=True)
    df_db.loc[mask, 'unit'] = new_unit
    df_db.to_csv(file, sep=";", index=False)
    return


def update_database(filename, df_new, lever=None):
    # Update csv file in database/data/csv based on a dataframe with columns
    # "Country, Years, lever-name, (col1, col2, col3)"
    if lever is not None:
        rename_cols = {"Country": 'geoscale', 'Years': 'timescale', lever: 'level'}
    else:
        rename_cols = {"Country": 'geoscale', 'Years': 'timescale'}
        df_new['level'] = 0
    df_new.rename(columns=rename_cols, inplace=True)
    df_new = pd.melt(df_new, id_vars=['geoscale', 'timescale', 'level'], var_name='eucalc-name', value_name='value')
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    folderpath = os.path.join(current_file_directory, "../../_database/data/csv/")
    file = folderpath + filename + '.csv'
    df_db = pd.read_csv(file, sep=";")

    # Merge DataFrame A with DataFrame B based on common columns
    merged_df = df_db.merge(df_new, how='outer', on=['geoscale', 'timescale', 'level', 'eucalc-name'],
                            suffixes=('_old', '_new'))
    merged_df['value'] = merged_df['value_new']

    # check for NaN values
    mask = pd.isna(merged_df['value'])
    merged_df.loc[mask, 'value'] = merged_df.loc[mask, 'value_old']
    # Copy merged on value_old, delete value and value_new and rename value_old as value
    # (this is to preserve the column order)
    merged_df['value_old'] = merged_df['value']
    merged_df.drop(columns=["value_new", "value"], inplace=True)
    merged_df.rename(columns={'value_old': 'value'}, inplace=True)
    merged_df.sort_values(by=['geoscale', 'timescale'], axis=0, inplace=True)
    merged_df.to_csv(file, sep=";", index=False)

    return


def levers_in_file(filename, folderpath="default"):
    # Returns all the lever names in a file
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    if folderpath == "default":
        folderpath = os.path.join(current_file_directory, "../../_database/data/csv/")
    file = folderpath + filename + '.csv'
    df_db = pd.read_csv(file, sep=";")
    levers = list(set(df_db['lever']))
    return levers


def read_database_w_filter(filename, lever, filter_dict, folderpath="default", db_format=False, level='all'):
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    if folderpath == "default":
        folderpath = os.path.join(current_file_directory, "../../_database/data/csv/")
    file = folderpath + filename + '.csv'
    df_db = pd.read_csv(file, sep=";")
    for column, pattern in filter_dict.items():
        mask = df_db[column].astype(str).str.contains(pattern)
        df_db = df_db.loc[mask]

    # Remove duplicates
    len_init = len(df_db)
    df_db = df_db.drop_duplicates(subset=['geoscale', 'timescale', 'level', 'string-pivot', 'eucalc-name'])
    if len(df_db) - len_init < 0:
        print(f"Duplicates found in: {filename}, use .duplicated on dataframe to check which lines are repeated")

    if db_format:
        return df_db
    else:
        df_db_ots = (df_db.loc[(df_db["level"] == 0) & (df_db['lever'] == lever)]).copy()
        if level == 'all':
            df_db_fts = (df_db.loc[(df_db["level"] != 0) & (df_db['lever'] == lever)]).copy()
        else:
            df_db_fts = (df_db.loc[(df_db["level"] == level) & (df_db['lever'] == lever)]).copy()
        if (df_db_ots['string-pivot'] != 'none').any():
            df_ots = df_db_ots.pivot(index=['geoscale', 'timescale', 'level', 'string-pivot'], columns="eucalc-name", values='value')
        else:
            df_ots = df_db_ots.pivot(index=['geoscale', 'timescale', 'level'], columns="eucalc-name", values='value')
        df_ots.reset_index(inplace=True)
        if (df_db_fts['string-pivot'] != 'none').any():
            df_fts = df_db_fts.pivot(index=['geoscale', 'timescale', 'level', 'string-pivot'], columns="eucalc-name", values='value')
        else:
            df_fts = df_db_fts.pivot(index=['geoscale', 'timescale', 'level'], columns="eucalc-name", values='value')
        df_fts.reset_index(inplace=True)
        rename_cols = {'geoscale': "Country", 'timescale': 'Years', 'level': lever}
        df_ots.rename(columns=rename_cols, inplace=True)
        df_fts.rename(columns=rename_cols, inplace=True)

        return df_ots, df_fts


def update_database_from_db_old(filename, db_new, folderpath="default"):
    # Update csv file in database/data/csv based on a database with columns
    # "geoscale, timescale, eucalc-name, level, value"
    if folderpath == "default":
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        folderpath = os.path.join(current_file_directory, "../../_database/data/csv/")
    file = folderpath + filename + '.csv'
    df_db = pd.read_csv(file, sep=";")

    # Merge DataFrame A with DataFrame B based on common columns
    on_cols = list(df_db.columns.drop(['value']))
    merged_df = df_db.merge(db_new, how='outer', on=on_cols,
                            suffixes=('_old', '_new'))
    merged_df['value'] = merged_df['value_new']

    # check for NaN values
    mask = pd.isna(merged_df['value'])
    merged_df.loc[mask, 'value'] = merged_df.loc[mask, 'value_old']
    # Copy merged on value_old, delete value and value_new and rename value_old as value
    # (this is to preserve the column order)
    merged_df['value_old'] = merged_df['value']
    merged_df.drop(columns=["value_new", "value"], inplace=True)
    merged_df.rename(columns={'value_old': 'value'}, inplace=True)
    merged_df.sort_values(by=['geoscale', 'timescale'], axis=0, inplace=True)
    merged_df.to_csv(file, sep=";", index=False)

    return


def read_database_fxa(filename, folderpath="default", db_format=False, filter_dict=None):

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    if folderpath == "default":
        folderpath = os.path.join(current_file_directory, "../../_database/data/csv/")
    file = folderpath + filename + '.csv'
    df_db = pd.read_csv(file, sep=";")
    if filter_dict is not None:
        for column, pattern in filter_dict.items():
            mask = df_db[column].astype(str).str.contains(pattern)
            df_db = df_db.loc[mask]
    # Remove duplicates
    len_init = len(df_db)
    df_db = df_db.drop_duplicates(subset=['geoscale', 'timescale', 'level', 'string-pivot', 'eucalc-name'])
    if len(df_db) - len_init < 0:
        print(f"Duplicates found in: {filename}, use .duplicated on dataframe to check which lines are repeated")

    # Check occurrences of countries:
    countries_counts = df_db['geoscale'].value_counts()
    wrong_countries = countries_counts - countries_counts.median()
    if not (wrong_countries == 0).all():
        wrong = list(wrong_countries[wrong_countries != 0].index)
        warnings.warn(f'The country {wrong} has not the right number of rows')
    
    if db_format:
        return df_db
    else:
        if (df_db['string-pivot'] != 'none').any():
            df = df_db.pivot(index=['geoscale', 'timescale', 'string-pivot'], columns="eucalc-name", values='value')
        else:
            df = df_db.pivot(index=['geoscale', 'timescale'], columns="eucalc-name", values='value')
        df.reset_index(inplace=True)
        rename_cols = {'geoscale': "Country", 'timescale': 'Years'}
        df.rename(columns=rename_cols, inplace=True)
        return df


def dm_lever_dict_from_df(df_fts, levername, num_cat):
    levels = list(set(df_fts[levername]))
    dict_dm = {}
    for i in levels:
        df_fts_i = df_fts.loc[df_fts[levername] == i].copy()
        df_fts_i.drop(columns=[levername], inplace=True)
        rename_fts = {}
        for col in df_fts_i.columns:
            rename_fts[col] = col.replace('fts_', '')
        df_fts_i.rename(columns=rename_fts, inplace=True)
        df_fts_i.sort_values(by=['Country', 'Years'], axis=0, inplace=True)
        dict_dm[i] = DataMatrix.create_from_df(df_fts_i, num_cat=num_cat)
    return dict_dm


def read_database_to_ots_fts_dict(file, lever, num_cat, baseyear, years, dict_ots, dict_fts, df_ots=None, df_fts=None,
                                  filter_dict=None):
    # It reads the database in data/csv with name file and returns the ots and the fts in form
    # of datamatrix accessible by dictionaries:
    # e.g.  dict_ots = {lever: dm_ots}
    #       dict_fts = {lever: {1: dm_fts_level_1, 2: dm_fts_level_2, 3: dm_fts_level_3, 4: dm_fts_level_4}}
    # where file is the name of the file and lever is the levername
    if df_ots is None and df_fts is None:
        if filter_dict is None:
            df_ots, df_fts = read_database(file, lever, level='all')
        else:
            df_ots, df_fts = read_database_w_filter(file, lever, filter_dict)

    if not df_fts.empty:
        # Drop from fts the baseyear and earlier years if any
        df_fts.drop(df_fts[df_fts.Years <= baseyear].index, inplace=True)
        # Keep fts only one every five years
        df_fts = df_fts[df_fts['Years'].isin(years)].copy()

    dict_lever = dm_lever_dict_from_df(df_fts, lever, num_cat)
    dict_fts[lever] = dict_lever

    # Keep only years from 1990
    df_ots.drop(df_ots[df_ots.Years > baseyear].index, inplace=True)
    df_ots = df_ots[df_ots['Years'].isin(years)].copy()
    # Remove 'ots_' and drop lever
    df_ots.drop(columns=[lever], inplace=True)
    rename_ots = {}
    for col in df_ots.columns:
        rename_ots[col] = col.replace('ots_', '')
    df_ots.rename(columns=rename_ots, inplace=True)
    # Sort by country years
    df_ots.sort_values(by=['Country', 'Years'], axis=0, inplace=True)
    dm_ots = DataMatrix.create_from_df(df_ots, num_cat)
    dict_ots[lever] = dm_ots

    return dict_ots, dict_fts


def read_database_to_ots_fts_dict_w_groups(file, lever, num_cat_list, baseyear, years, dict_ots, dict_fts, column: str,
                                           group_list: list):
    # It reads the database in data/csv with name file and returns the ots and the fts in form
    # of datamatrix accessible by dictionaries:
    # e.g.  dict_ots = {lever: {group1: dm_1, group2: dm_2, grou}}
    #       dict_fts = {lever: [dm_fts_a, dm_fts_b, dm_fts_c]}
    # where file is the name of the file and lever is the levername
    dm_ots_groups = {}
    dm_fts_groups = {}
    for (i, group) in enumerate(group_list):
        filter_dict = {column: group}
        num_cat = num_cat_list[i]
        dict_tmp_ots = {}
        dict_tmp_fts = {}
        read_database_to_ots_fts_dict(file, lever, num_cat, baseyear, years, dict_tmp_ots, dict_tmp_fts,
                                      filter_dict=filter_dict)
        group = group.replace('.*', '')
        dm_ots_groups[group] = dict_tmp_ots[lever]
        dm_fts_groups[group] = dict_tmp_fts[lever]

    dict_ots[lever] = dm_ots_groups
    dict_fts[lever] = dm_fts_groups

    return dict_ots, dict_fts


def database_to_df(df_db, lever, level='all'):
    # Given a dataframe in database format and a lever name, it extracts df_ots and df_fts as dataframes with columns:
    # Country, Years, var1, var2, var3 etc
    # Impose correct type
    df_db['timescale'] = df_db['timescale'].astype(int)
    df_db['level'] = df_db['level'].astype(int)
    df_db['value'] = df_db['value'].astype(float)
    df_db['geoscale'] = df_db['geoscale'].astype(str)
    df_db['lever'] = df_db['lever'].astype(str)
    df_db['variables'] = df_db['variables'].astype(str)
    # Remove duplicates
    len_init = len(df_db)
    df_db = df_db.drop_duplicates(subset=['geoscale', 'timescale', 'level', 'variables'])
    if len(df_db) - len_init < 0:
        print(f"Duplicates found, use .duplicated on dataframe to check which lines are repeated")
    # Extract ots
    df_db_ots = (df_db.loc[(df_db["level"] == 0) & (df_db['lever'] == lever)]).copy()
    # Extract fts
    if level == 'all':
        df_db_fts = (df_db.loc[(df_db["level"] != 0) & (df_db['lever'] == lever)]).copy()
    else:
        df_db_fts = (df_db.loc[(df_db["level"] == level) & (df_db['lever'] == lever)]).copy()
    # Pivot ots
    df_ots = df_db_ots.pivot(index=['geoscale', 'timescale', 'level'], columns="variables", values='value')
    df_ots.reset_index(inplace=True)
    # Pivot fts
    df_fts = df_db_fts.pivot(index=['geoscale', 'timescale', 'level'], columns="variables", values='value')

    df_fts.reset_index(inplace=True)
    rename_cols = {'geoscale': "Country", 'timescale': 'Years', 'level': lever}
    df_ots.rename(columns=rename_cols, inplace=True)
    df_fts.rename(columns=rename_cols, inplace=True)

    return df_ots, df_fts


def database_to_dm(df_db, lever, num_cat, baseyear, years, level='all'):
    # Given a dataframe in database format, it extracts the ots and the fts dictionaries
    # e.g.  dict_ots = {lever: dm_ots}
    #       dict_fts = {lever: {1: dm_fts_level_1, 2: dm_fts_level_2, 3: dm_fts_level_3, 4: dm_fts_level_4}}
    df_ots, df_fts = database_to_df(df_db, lever, level)
    if isinstance(years[0], str):
        years = [int(y) for y in years]
    dict_ots = dict()
    dict_fts = dict()
    dict_ots, dict_fts = read_database_to_ots_fts_dict(file=None, lever=lever, num_cat=num_cat, baseyear=baseyear,
                                                       years=years, dict_ots=dict_ots, dict_fts=dict_fts, df_ots=df_ots,
                                                       df_fts=df_fts)
    return dict_ots, dict_fts


def dm_to_database(dm, lever, module, level=0):
    df = dm.write_df()
    rename_cols = {'Country': "geoscale", 'Years': 'timescale'}
    df.rename(columns=rename_cols, inplace=True)
    df['level'] = level
    df['lever'] = lever
    df['module'] = module
    df_db = df.melt(id_vars=['geoscale', 'timescale', 'lever', 'level', 'module'], var_name='variables', value_name='value')

    return df_db


def update_database_from_db(db_old, db_new):
    # Replace
    # Merge DataFrame A with DataFrame B based on common columns
    on_cols = list(db_old.columns.drop(['value']))
    merged_df = db_old.merge(db_new, how='outer', on=on_cols, suffixes=('_old', '_new'))
    merged_df['value'] = merged_df['value_new']

    # check for NaN values
    mask = pd.isna(merged_df['value'])
    merged_df.loc[mask, 'value'] = merged_df.loc[mask, 'value_old']
    # Copy merged on value_old, delete value and value_new and rename value_old as value
    # (this is to preserve the column order)
    merged_df['value_old'] = merged_df['value']
    merged_df.drop(columns=["value_new", "value"], inplace=True)
    merged_df.rename(columns={'value_old': 'value'}, inplace=True)
    merged_df.sort_values(by=['geoscale', 'timescale'], axis=0, inplace=True)

    return merged_df


def csv_database_reformat(filename):
    # Change format of database .csv file (drop useless colums)
    root = find_git_root()
    folder = '_database/data/csv/'
    file = folder + filename
    file_path = os.path.join(root, file)
    df = pd.read_csv(file_path, sep=';')
    df = df[['geoscale', 'timescale', 'module', 'eucalc-name', 'lever', 'level', 'value']].copy()
    df.rename({'eucalc-name': 'variables'}, axis=1, inplace=True)
    df.to_csv(file_path, sep=';', index=False)

    return


def update_database_from_dm(dm, filename, lever, level, module):
    root = find_git_root()
    path = '_database/data/csv/'
    file = path + filename
    file_path = os.path.join(root, file)
    db_new = dm_to_database(dm, lever=lever, module=module, level=level)
    db_old = pd.read_csv(file_path, sep=';')
    # Impose correct type
    db_old['timescale'] = db_old['timescale'].astype(int)
    db_old['level'] = db_old['level'].astype(int)
    db_old['value'] = db_old['value'].astype(float)
    db_old['geoscale'] = db_old['geoscale'].astype(str)
    db_old['lever'] = db_old['lever'].astype(str)
    db_old['variables'] = db_old['variables'].astype(str)
    df_merged = update_database_from_db(db_old, db_new)
    df_merged.to_csv(file_path, index=False, sep=';')
    return


def read_database_to_dm(filename=None, df_db=None, lever=None, num_cat=0, baseyear=2023, years=None, level='all', filter=dict()):
    # csv file columns: geoscale;timescale;module;variables;lever;level;value
    if filename is not None:
        root = find_git_root()
        path = '_database/data/csv/'
        file = path + filename
        file_path = os.path.join(root, file)
        df_db = pd.read_csv(file_path, sep=';')
    if lever is None:
        levers = list(set(df_db['lever']))
        if len(levers) != 1:
            raise ValueError(f'the file {filename} contains more than one lever: {levers}')
        else:
            lever = levers[0]
    if years is None:
        years_ots = list(np.linspace(start=1990, stop=baseyear, num=(baseyear-1990)+1))
        years_fts = list(np.linspace(start=2025, stop=2050, num=int((2050-2025)/5+1)))
        years = years_ots + years_fts
    if len(filter.items()) > 0:
        for col_name, values_to_keep in filter.items():
            df_db = df_db[df_db[col_name].isin(values_to_keep)].copy()

    dict_ots, dict_fts = database_to_dm(df_db, lever, num_cat, baseyear, years, level=level)
    return dict_ots, dict_fts
