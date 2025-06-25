import deepl
import pandas as pd
import requests
import os
import pickle
import zipfile
import io
from model.common.io_database import database_to_dm
from model.common.data_matrix_class import DataMatrix
from model.common.auxiliary_functions import create_years_list
import numpy as np

# Initialize the Deepl Translator
deepl_api_key = '9ecffb3f-5386-4254-a099-8bfc47167661:fx'
translator = deepl.Translator(deepl_api_key)

def df_fso_excel_to_dm(df, header_row, names_dict, var_name, unit, num_cat, keep_first=False, country='Switzerland'):
    # Federal statistical office df from excel to dm
    # Change headers
    new_header = df.iloc[header_row]
    new_header.values[0] = 'Variables'
    df.columns = new_header
    df = df[header_row+1:].copy()
    # Remove nans and empty columns/rows
    if np.nan in df.columns:
        df.drop(columns=np.nan, inplace=True)
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


def translate_text(text):

    translation = translator.translate_text(text, target_lang='EN-GB')
    return translation.text

def create_ots_years_list(years_setting):
    startyear: int = years_setting[0]  # Start year is argument [0], i.e., 1990
    baseyear: int = years_setting[1]  # Base/Reference year is argument [1], i.e., 2015
    lastyear: int = years_setting[2]  # End/Last year is argument [2], i.e., 2050
    step_fts = years_setting[3]  # Timestep for scenario is argument [3], i.e., 5 years
    years_ots = list(np.linspace(start=startyear, stop=baseyear, num=(baseyear - startyear) + 1).astype(int).astype(str))
    return years_ots


def extract_energy_data(file_url, local_filename, baseyear, years_ots, outfile_dm):

    # If the routine had already run and the dm was created, it skips everything and it just loads the dm
    # this is mainly done to avoid calling deepl on repeat which only allows a limited number of calls
    if not os.path.exists(outfile_dm):
        response = requests.get(file_url, stream=True)
        if not os.path.exists(local_filename):
            # Check if the request was successful
            if response.status_code == 200:
                with open(local_filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"File downloaded successfully as {local_filename}")
            else:
                print(f"Error: {response.status_code}, {response.text}")
        else:
            print(f'File {local_filename} already exists. If you want to download again delete the file')

        df = pd.read_csv(local_filename)

        # Clean file format
        # Add """ before the first comma
        df['Jahr'] = df['Jahr'].str.replace(',', '",', 1)
        df = df[['Jahr']]
        # Split the cell into different columns based on ", " separator
        df[['timescale', 'Variable', 'Description', 'value']] = df['Jahr'].str.split('",', expand=True)
        # Drop the original column
        df.drop(columns=['Jahr'], inplace=True)
        for col in df.columns:
            df[col] = df[col].str.lstrip('"')

        # Use deepl to translate variables from de to en
        variables_de = list(set(df['Variable']))
        variables_en = [translate_text(var) for var in variables_de]
        var_dict = dict(zip(variables_de, variables_en))
        # Use deepl to translate description from de to en
        description_de = list(set(df['Description']))
        description_en = [translate_text(var) for var in description_de]
        cat_dict = dict(zip(description_de, description_en))

        df['Use'] = df['Variable'].map(var_dict)
        df['Energy sources'] = df['Description'].map(cat_dict)

        df.drop(['Variable', 'Description'], axis=1, inplace=True)

        # Create database format
        lever = 'energy_tmp'  # temporary lever name
        df['variables'] = df['Use'] + '_' + df['Energy sources'] + '[TJ]'
        df.drop(['Use', 'Energy sources'], axis=1, inplace=True)
        df['level'] = 0
        df['lever'] = 'energy_tmp'
        df['string-pivot'] = 'none'
        df['geoscale'] = 'Switzerland'

        # Replace empty strings with NaN
        df['value'].replace('', np.nan, inplace=True)

        dict_ots, dict_fts = database_to_dm(df, lever, num_cat=1, baseyear=baseyear, years=years_ots, level='all')

        dm = dict_ots[lever]

        with open(outfile_dm, 'wb') as handle:
            pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(outfile_dm, 'rb') as handle:
            dm = pickle.load(handle)

    return dm

def extract_nexuse_capacity_data(file):
    dm = None
    for sheet_yr in ['2020', '2030', '2040', '2050']:
        df_yr = pd.read_excel(file, sheet_name=sheet_yr)

        df_yr = df_yr.loc[df_yr['Country'] == 'CH']
        df_yr = df_yr[
            ['idGen', 'GenName', 'Technology', 'GenEffic', 'CO2Rate', 'Pmax', 'Pmin', 'StartYr', 'EndYr', 'Emax',
             'SubRegion']]

        df_eff_CO2 = df_yr.groupby(['Technology'])[['GenEffic', 'CO2Rate']].mean()

        years_all = years_ots + years_fts
        df = None
        for y in years_all:
            df_all = df_yr[(df_yr['StartYr'] <= y) & (df_yr['EndYr'] >= y)].copy()
            df_all['Years'] = y
            if df is None:
                df = df_all
            else:
                df = pd.concat([df, df_all], axis=0)

        df_P_E = df.groupby(['Technology', 'SubRegion', 'Years'])[['Pmax', 'Pmin', 'Emax']].sum()

        df_P_E.reset_index(inplace=True)
        df_P_E.rename({'SubRegion': 'Country'}, axis=1, inplace=True)

        df_T = df_P_E.pivot(index=['Country', 'Years'], columns='Technology',
                            values=['Pmax', 'Pmin', 'Emax'])
        #df_T.columns = df_T.columns.swaplevel(0, 1)
        df_T.columns = ['_'.join(col).strip() for col in df_T.columns.values]

        df_T.columns = ['pow_capacity-' + col for col in df_T.columns.values]
        cols = []
        for col in df_T.columns.values:
            if 'Pmax' or 'Pmin' in col:
                cols.append(col+'[MW]')
            elif 'Emax' in col:
                cols.append(col + '[MWh]')
        df_T.columns = cols
        df_T.reset_index(inplace=True)

        # Make sure you have all years for all countries
        # Create a DataFrame with all combinations of countries and years
        countries = df_T['Country'].unique()
        complete_index = pd.MultiIndex.from_product([countries, years_all], names=['Country', 'Years'])
        complete_df = pd.DataFrame(index=complete_index).reset_index()

        # Merge with the original DataFrame
        df_T = pd.merge(complete_df, df_T, on=['Country', 'Years'], how='left')

        if dm is None:
            dm = DataMatrix.create_from_df(df_T, num_cat=1)
        else:
            dm_yr = DataMatrix.create_from_df(df_T, num_cat=1)
            extra_cntr = list(set(dm.col_labels['Country']) - set(dm_yr.col_labels['Country']))
            if len(extra_cntr) > 0:
                dm_yr.add(0, dim='Country', col_label=extra_cntr, dummy=True)
            extra_cntr = list(set(dm_yr.col_labels['Country']) - set(dm.col_labels['Country']))
            if len(extra_cntr) > 0:
                dm.add(0, dim='Country', col_label=extra_cntr, dummy=True)
            extra_tech = list(set(dm_yr.col_labels['Categories1']) - set(dm.col_labels['Categories1']))
            if len(extra_tech) > 0:
                dm.add(0, dim='Categories1', col_label=extra_tech, dummy=True)
            extra_tech = list(set(dm.col_labels['Categories1']) - set(dm_yr.col_labels['Categories1']))
            if len(extra_tech) > 0:
                dm_yr.add(0, dim='Categories1', col_label=extra_tech, dummy=True)
            dm_yr.sort('Country')
            dm_yr.sort('Categories1')
            dm.sort('Country')
            dm.sort('Categories1')
            dm.array = np.fmax(dm.array, dm_yr.array)
    dm.sort('Years')
    dm_CH = dm.groupby({'Switzerland': '.*'}, dim='Country', regex=True, inplace=False)
    dm.append(dm_CH, dim='Country')
    return dm, df_eff_CO2


def extract_production_data(file_url, local_filename):
    if not os.path.exists(local_filename):
        response = requests.get(file_url, stream=True)
        # Check if the request was successful
        if response.status_code == 200:
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"File downloaded successfully as {local_filename}")
        else:
            print(f"Error: {response.status_code}, {response.text}")
    else:
        print(f'File {local_filename} already exists. If you want to download again delete the file')

    df = pd.read_excel(local_filename, sheet_name="T24")
    combined_headers = []
    for col1, col2, col3 in zip(df.iloc[4], df.iloc[5], df.iloc[6]):
        combined_headers.append(str(col1) + '-' + str(col2) + '[' + str(col3) + ']')
    # Set the new header
    df.columns = combined_headers
    df = df[7:].copy()

    def is_valid_number(val):
        return isinstance(val, (int, float)) and not pd.isna(val)

    # Apply the function to filter out rows with no valid numeric values
    df = df[df.apply(lambda row: row.map(is_valid_number).any(), axis=1)]
    # Apply similarly for columns if needed
    df = df.loc[:, df.apply(lambda col: col.map(is_valid_number).any())]

    df.rename({'Année-nan[nan]': 'Years'}, axis=1, inplace=True)
    df['Country'] = 'Switzerland'
    df.set_index(['Country', 'Years'], inplace=True)
    # df.columns = ['pow_production_'+col for col in df.columns]
    df = df[df.columns.drop(list(df.filter(regex='[%]')))]
    df.columns = [str.replace(col, 'nan-', '') for col in df.columns]
    df.columns = [str.replace(col, 'nan', '') for col in df.columns]
    df.replace('-', 0, inplace=True)
    df.reset_index(inplace=True)
    df = df.drop('Total[GWh]', axis=1)
    dm = DataMatrix.create_from_df(df, num_cat=0)
    cols = ["Centrales hydrauliques-Centrales au fil de l'eau",
            'Centrales nucléaires-',
            'Centrales thermiques class. et centrales chaleur-force1-Total',
            'Centrales à accumulation',
            'Energies renouvelables diverses3-Chauffages au bois et en partie au bois',
            'Eoliennes', 'Installations au biogaz', 'Installations photo-voltaïques', "Pompage d'accumu-lation-",
            'Production nationale (brute)-', 'Production nette (pompage déduit)-', 'dont\nnon renouvelable',
            'dont renouvelable 2']

    dm_out = dm.groupby({'pow_production_RoR': ".*au fil de l'eau.*", 'pow_production_Nuclear': '.*nucléaire.*',
                         'pow_production_Oil-Gas': '.*class.*|.*biogaz.*', 'pow_production_Dam': '.*accumulation.*',
                         'pow_production_WindOn': '.*Eoliennes.*',
                         'pow_production_PV-roof': '.*photo.*', 'pow_production_Pump-Open': '.*Pompage.*'},
                        regex=True, dim='Variables', inplace=False)
    dm_out.deepen()
    dm_out.filter({'Years': years_ots}, inplace=True)
    dm_out.change_unit('pow_production', 1000, 'GWh', 'MWh')
    return dm_out


def compute_capacity_factor(dm_capacity, dm_production):

    dm_capacity_ots = dm_capacity.filter({'Years': years_ots, 'Country': ['Switzerland']}, inplace=False)
    dm_capacity_ots.groupby({'Oil-Gas': 'Oil|Gas.*'}, regex=True, dim='Categories1', inplace=True)
    missing_cat = set(dm_capacity_ots.col_labels['Categories1']) - set(dm_production.col_labels['Categories1'])
    dm_production.add(0, dummy=True, dim='Categories1', col_label=missing_cat)
    dm_production.append(dm_capacity_ots, dim='Variables')

    idx = dm_production.idx
    # arr = dm_production.array[:, :, idx['pow_capacity-Emax'], :]
    # arr[arr == 0] = np.nan
    # dm_production.array[:, :, idx['pow_capacity-Emax'], :] = arr
    # arr_tmp = np.fmin(dm_production.array[:, :, idx['pow_capacity-Pmax'], :]*24*365,
    #                  dm_production.array[:, :, idx['pow_capacity-Emax'], :])
    arr_tmp = dm_production.array[:, :, idx['pow_capacity-Pmax'], :] * 24 * 365
    dm_production.add(arr_tmp, dim='Variables', col_label='pow_capacity-E', unit='MWh')
    dm_production.operation('pow_production', '/', 'pow_capacity-E', out_col='pow_capacity-factor', unit='%')

    dm_cap_factor = dm_production.filter({'Variables': ['pow_capacity-factor']})
    arr = dm_cap_factor.array
    arr[np.isinf(arr)] = np.nan
    dm_cap_factor.array = arr
    mean_val = np.nanmean(dm_cap_factor.array, axis=1)
    arr = dm_cap_factor.array[0, :, 0, :]
    nan_indices = np.where(np.isnan(arr))
    arr[nan_indices] = np.take(mean_val, nan_indices[1])
    dm_cap_factor.array[0, :, 0, :] = arr
    dm_production.filter({'Variables': ['pow_production']}, inplace=True)

    return dm_cap_factor, dm_production


def extract_hydro_capacity_at_year(df, yr):
    df.rename({'ZE-Kanton': 'Country'}, axis=1, inplace=True)
    # Keep only active power-plants
    df = df.loc[df['ZE-Status'] == 'im Normalbetrieb']
    # Filter useful variables
    df_P = df[['Country', 'Inst. Pumpenleistung']].copy()
    df = df.loc[df['WKA-Typ'] != 'U']  # Pump capacity only
    df = df[['WKA-Typ', 'Country', 'Max. Leistung ab Generator']]  # RoR and Dam
    df = df.loc[df['WKA-Typ'] != 'P']

    # Group Pump capacity by canton and add years column
    df_P = df_P.groupby(['Country']).sum()
    df_P['Years'] = yr
    df_P.rename({'Inst. Pumpenleistung': 'pow_capacity-Pmax_Pump-Open[MW]'}, axis=1, inplace=True)
    df_P.reset_index(inplace=True)
    dm_P = DataMatrix.create_from_df(df_P, num_cat=1)

    # Group capacity by canton and type
    df = df.groupby(['Country', 'WKA-Typ']).sum()
    df.reset_index(inplace=True)
    df['Years'] = yr
    df['WKA-Typ'] = df['WKA-Typ'].str.replace('L', 'pow_capacity-Pmax_RoR[MW]', 1)
    df['WKA-Typ'] = df['WKA-Typ'].str.replace('S', 'pow_capacity-Pmax_Dam[MW]', 1)

    # Dam RoR
    df.rename({'Max. Leistung ab Generator': 'Pmax'}, axis=1, inplace=True)
    df = df.pivot_table(index=['Country', 'Years'], columns=['WKA-Typ'], values='Pmax', aggfunc='sum')
    df.reset_index(inplace=True)
    dm = DataMatrix.create_from_df(df, num_cat=1)

    dm.append(dm_P, dim='Categories1')

    return dm

def extract_old_hydro_capacity_data(url_dict):

    dm_all = None
    for yr in url_dict.keys():
        local_filename = url_dict[yr]['local_filename']
        file_url = url_dict[yr]['file_url']
        if not os.path.exists(local_filename):
            response = requests.get(file_url, stream=True)
            # Check if the request was successful
            if response.status_code == 200:
                with open(local_filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"File downloaded successfully as {local_filename}")
            else:
                print(f"Error: {response.status_code}, {response.text}")
        else:
            print(f'File {local_filename} already exists. If you want to download again delete the file')

        df = pd.read_excel(local_filename)
        dm = extract_hydro_capacity_at_year(df, yr)

        if dm_all is None:
            dm_all = dm.copy()
        else:
            dm_all.append(dm, dim='Years')

    dm_all.sort(dim='Years')

    dm_CH = dm_all.groupby({'Switzerland': '.*'}, regex=True, inplace=False, dim='Country')
    dm_all.append(dm_CH, dim='Country')

    return dm_all


def extract_old_hydro_capacity_zip(url_dict):

    dm_all = None
    for yr in url_dict.keys():

        df = pd.read_excel(url_dict[yr]['local_filename'])
        dm = extract_hydro_capacity_at_year(df, yr)

        if dm_all is None:
            dm_all = dm.copy()
        else:
            dm_all.append(dm, dim='Years')

    dm_all.sort(dim='Years')

    dm_CH = dm_all.groupby({'Switzerland': '.*'}, regex=True, inplace=False, dim='Country')
    dm_all.append(dm_CH, dim='Country')

    return dm_all


def extract_nuclear_capacity_data(reactor_list, dm_capacity):
    # 'Muhleberg': {'Canton': 'BE', 'Pmax': 373, 'StartYr': 1971, 'EndYr': 2019},

    dm = dm_capacity.filter({'Variables': ['pow_capacity-Pmax'], 'Categories1': ['Nuclear']})
    # Reset values to 0
    idx = dm.idx
    dm.array[...] = 0
    years_missing = list(set(range(dm.col_labels['Years'][0], dm.col_labels['Years'][-1])) - set(dm.col_labels['Years']))
    dm.add(0, dummy=True, dim='Years', col_label=years_missing)
    dm.sort('Years')
    for name, properties in reactor_list.items():
        cntr = properties['Canton']
        startyr = max(dm.col_labels['Years'][0], properties['StartYr'])
        endyr = min(dm.col_labels['Years'][-1], properties['EndYr'])
        for yr in range(startyr, endyr+1):
            dm.array[idx[cntr], idx[yr], idx['pow_capacity-Pmax'], idx['Nuclear']] += properties['Pmax']
    dm.array[idx['Switzerland'], ...] = np.nansum(dm.array, axis=0)
    return dm


def fill_missing_years_capacity_hydro(dm_capacity_hydro_ots, years_map, years_ots):
    years_missing = list(set(years_ots) - set(dm_capacity_hydro_ots.col_labels['Years']))
    dm_capacity_hydro_ots.add(np.nan, dim='Years', col_label=years_missing, dummy=True)
    idx = dm_capacity_hydro_ots.idx
    for ref_yr, yr_range in years_map.items():
        for yr in yr_range:
            dm_capacity_hydro_ots.array[:, idx[yr], ...] = dm_capacity_hydro_ots.array[:, idx[ref_yr], ...]

    dm_capacity_hydro_ots.sort('Years')
    return dm_capacity_hydro_ots


def adjust_based_on_nexuse(dm, dm_nexuse):
    dm.sort('Categories1')
    dm_nexuse.sort('Categories1')
    dm_filter = dm_nexuse.filter({'Country': ['Switzerland'], 'Years': years_ots, 'Variables': ['pow_capacity-Pmax'],
                                             'Categories1': dm.col_labels['Categories1']})
    dm.rename_col('pow_capacity-Pmax', 'pow_capacity-Pmax-old', dim='Variables')
    dm_tmp = dm.filter({'Country': ['Switzerland']})
    dm_filter.append(dm_tmp, dim='Variables')
    dm_filter.operation('pow_capacity-Pmax', '/', 'pow_capacity-Pmax-old', out_col='factor', unit='%')
    dm_filter.array[dm_filter.array == 0] = np.nan
    idx = dm_filter.idx
    avg_factor = np.nanmean(dm_filter.array[:, :, idx['factor'], :], axis=1, keepdims=True)
    idx = dm.idx
    dm.array[:, :, idx['pow_capacity-Pmax-old'], :] = dm.array[:, :, idx['pow_capacity-Pmax-old'], :] * avg_factor
    dm.rename_col('pow_capacity-Pmax-old', 'pow_capacity-Pmax', dim='Variables')
    return dm


def extract_waste_capacity_ots(local_filename, years_ots):
    df = pd.read_excel(local_filename)
    df['Kanton'] = df['Kanton'].str.replace('\xa0', '')
    # Rename cols
    dict_rename = {'Kanton': 'Country', 'Täglicher Abfalldurchsatz (Tonnen)': 'ref_capacity-Pmax[t]',
                   'Elektrische Leistung (MW)': 'pow_capacity-Pmax[MW]', 'Inbetriebnahme': 'Years'}
    df.rename(dict_rename, axis=1, inplace=True)
    # Keep only important cols
    df = df[list(dict_rename.values())]
    all_years = range(df['Years'].min(), df['Years'].max() + 1)
    all_countries = df['Country'].unique()
    # 2. Create a MultiIndex for all combinations of countries and years
    multi_index = pd.MultiIndex.from_product([all_countries, all_years], names=['Country', 'Years'])
    # 3. Reindex the DataFrame with this MultiIndex
    df = df.set_index(['Country', 'Years']).reindex(multi_index)
    # 4. Fill missing values with 0
    df = df.fillna(0).astype(int)
    # Reset index if needed
    df = df.reset_index()
    dm = DataMatrix.create_from_df(df, num_cat=0)
    years_missing = list(set(years_ots) - set(dm.col_labels['Years']))
    dm.add(0, dummy=True, dim='Years', col_label=years_missing)
    dm.sort('Years')
    dm.array = np.cumsum(dm.array, axis=1)
    dm.filter({'Years': years_ots}, inplace=True)
    dm_CH = dm.groupby({'Switzerland': '.*'}, regex=True, dim='Country', inplace=False)
    dm.append(dm_CH, dim='Country')
    dm.rename_col(['pow_capacity-Pmax', 'ref_capacity-Pmax'], ['pow_capacity-Pmax_Waste', 'ref_capacity-Pmax_Waste'], dim='Variables')
    dm.deepen()
    return dm


def extract_oil_gas_capacity_data(local_filename):
    df = pd.read_excel(local_filename)
    dict_name = {'Inst. Leistung (MW)': 'Pmax', 'Brennstoff': 'Type'}
    df.rename(dict_name, axis=1, inplace=True)
    df = df[['Canton', 'Pmax', 'Type', 'StartYr', 'EndYr']]
    #df = df.pivot(columns='Type', index=['Canton', 'StartYr', 'EndYr'], values='Pmax')
    #df.reset_index(inplace=True)
    df['EndYr'] = df['EndYr'].fillna(2050)
    col_labels_dict = {'Country': list(df['Canton'].unique()),
                       'Years': years_ots,
                       'Variables': ['pow_capacity-Pmax'],
                       'Categories1': ['Oil', 'Gas']}
    dm = DataMatrix(col_labels_dict, units={'pow_capacity-Pmax': 'MW'})
    dm.array = np.zeros(tuple([len(list) for list in col_labels_dict.values()]))
    idx = dm.idx
    for yr in years_ots:
        for row in range(len(df)):
            StartYr = df.iloc[row]['StartYr']
            EndYr = df.iloc[row]['EndYr']
            if StartYr <= yr and yr <= EndYr:
                cntr = df.iloc[row]['Canton']
                cat = df.iloc[row]['Type']
                dm.array[idx[cntr], idx[yr], idx['pow_capacity-Pmax'], idx[cat]] += df.iloc[row]['Pmax']
    dm_CH = dm.groupby({'Switzerland': '.*'}, dim='Country', regex=True, inplace=False)
    dm.append(dm_CH,dim='Country')
    return dm

#years_setting = [1990, 2022, 2050, 5]  # Set the timestep for historical years & scenarios
years_ots = create_years_list(1990, 2023, 1)
years_fts = create_years_list(2025, 2050, 5)
baseyear = 2023

energy_url = 'https://www.uvek-gis.admin.ch/BFE/ogd/115/ogd115_gest_bilanz.csv'
local_path = 'data/Energy_statistic.csv'
outfile_dm = 'data/energy.pickle'

# Extract energy data
dm_energy = extract_energy_data(energy_url, local_path, baseyear, years_ots, outfile_dm)

# change units from TJ to TWh
for var in dm_energy.col_labels['Variables']:
    dm_energy.change_unit(var, factor=2.778e-4, old_unit='TJ', new_unit='TWh')

# Some visualisation
plotting = False
if plotting:
    dm_energy.datamatrix_plot(title='All energy variables in TWh')
    dm_energy.datamatrix_plot({'Variables': 'Import'}, title='Import [TWh]')
    dm_energy.datamatrix_plot({'Variables': 'Energy conversion - Nuclear power plants'},
                              title='Energy conversion - Nuclear power plants [TWh]')
    dm_energy.datamatrix_plot({'Variables': 'Energy conversion - conventional-thermal power, district heating and district heating power plants'},
                              title='Energy conversion - conventional-thermal power, district heating and district heating power plants [TWh]')

# SECTION - Extract Capacity ots, from Nexus-e installed capacity (2020-2050)
file = 'data/Capacity_Nexuse.xlsx'
dm_capacity, dm_const = extract_nexuse_capacity_data(file)
dm_capacity_group = dm_capacity.copy()
dm_capacity_group.groupby({'Gas': 'Gas.*', 'Hydro': 'Pump-Open|Dam|RoR'}, regex=True, dim='Categories1', inplace=True)

# SECTION - Extract Hydro Capacity ots, 1990-2020 from OFS
# Hydro-power
# Source:  https://www.bfe.admin.ch/bfe/fr/home/approvisionnement/energies-renouvelables/force-hydraulique.html#kw-96906
# 1991 - Statistik der Wasserkraftanlagen der Schweiz. Stand 1.1.1991
# 1996 - Statistik der Wasserkraftanlagen der Schweiz. Stand 1.1.1996
# 2001 - Statistik der Wasserkraftanlagen der Schweiz. Stand 1.1.2001
url_dict = {
            1990: {'file_url': 'https://www.bfe.admin.ch/bfe/fr/home/versorgung/erneuerbare-energien/wasserkraft.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvNzkyNA==.html',
                   'local_filename': 'data/hydro_power_1990.xlsx'},
            1995: {'file_url': 'https://www.bfe.admin.ch/bfe/fr/home/versorgung/erneuerbare-energien/wasserkraft.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvNzkyMw==.html',
                   'local_filename': 'data/hydro_power_1995.xlsx'},
            2000: {'file_url': 'https://www.bfe.admin.ch/bfe/fr/home/versorgung/erneuerbare-energien/wasserkraft.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvNzkyMg==.html',
                   'local_filename': 'data/hydro_power_2000.xlsx'},
            2005: {'file_url': 'nan',
                   'local_filename': 'data/hydro_power_2005.xlsx'},
            2015: {'file_url': 'nan',
                   'local_filename': 'data/hydro_power_2015.xlsx'},
            2019: {'file_url': 'nan',
                   'local_filename': 'data/hydro_power_2019.xlsx'},
            2020: {'file_url': 'nan',
                   'local_filename': 'data/hydro_power_2020.xlsx'},
            2021: {'file_url': 'nan',
                   'local_filename': 'data/hydro_power_2021.xlsx'},
            2022: {'file_url': 'nan',
                   'local_filename': 'data/hydro_power_2022.xlsx'},
            2023: {'file_url': 'nan',
                   'local_filename': 'data/hydro_power_2023.xlsx'}
            }
dm_capacity_hydro_ots = extract_old_hydro_capacity_data(url_dict)
# Create step function profile
# Allocate missing years to existing year
years_map = {1990: range(1990, 1993+1), 1995: range(1994, 1997+1), 2000: range(1998, 2000+1),
             2005: range(2001, 2009+1), 2015: range(2010, 2017+1), 2019: range(2018, 2019+1)}
dm_capacity_hydro_ots = fill_missing_years_capacity_hydro(dm_capacity_hydro_ots, years_map, years_ots)

dm_capacity_hydro_ots = adjust_based_on_nexuse(dm_capacity_hydro_ots, dm_capacity)

# SECTION - Extract Nuclear Capacity ots, 1990-2020 from OFS
# Source:  https://de.wikipedia.org/wiki/Liste_der_Kernreaktoren_in_der_Schweiz
reactor_list = {'Muhleberg': {'Canton': 'BE', 'Pmax': 373, 'StartYr': 1971, 'EndYr': 2019},
                'Beznau1': {'Canton': 'AG', 'Pmax': 365, 'StartYr': 1969, 'EndYr': 2033},
                'Beznau2': {'Canton': 'AG', 'Pmax': 365, 'StartYr': 1971, 'EndYr': 2032},
                'Gosgen': {'Canton': 'SO', 'Pmax': 1010, 'StartYr': 1979, 'EndYr': 2060},
                'Liebstadt': {'Canton': 'AG', 'Pmax': 1233, 'StartYr': 1984, 'EndYr': 2060}}

dm_capacity_nuclear = extract_nuclear_capacity_data(reactor_list, dm_capacity)

# SECTION - Waste capacity ots (power + heat)
# https://de.wikipedia.org/wiki/Liste_von_Kehrichtverbrennungsanlagen_in_der_Schweiz
local_filename = 'data/waste_power.xlsx'
# This has both the capacity in MW and the tonnes of waste incinerated every day (capacity)
dm_capacity_waste_ots = extract_waste_capacity_ots(local_filename, years_ots)

dm_capacity_waste_ots = adjust_based_on_nexuse(dm=dm_capacity_waste_ots, dm_nexuse=dm_capacity)

# SECTION - Oil & Gas capacity ots (power)
local_filename = 'data/oil_gas_power_plants.xlsx'
dm_capacity_oilgas_ots = extract_oil_gas_capacity_data(local_filename)


# SECTION - Electricity production ots
file_url = 'https://www.bfe.admin.ch/bfe/fr/home/versorgung/statistik-und-geodaten/energiestatistiken/gesamtenergiestatistik.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZnIvcHVibGljYX/Rpb24vZG93bmxvYWQvNzUxOQ==.html'
local_filename = 'data/statistique_globale_suisse_energie.xlsx'

dm_production = extract_production_data(file_url, local_filename)

dm_cap_factor, dm_production = compute_capacity_factor(dm_capacity, dm_production)

dm = dm_production.copy()
dm.append(dm_cap_factor, dim='Variables')
dm.drop(dim='Categories1', col_label='Oil-Gas')
dm.operation('pow_production', '/', 'pow_capacity-factor', out_col='pow_capacity-Pmax-est', unit='MW')
# Divide by h to actually have Pmax
idx = dm.idx
dm.array[:, :, idx['pow_capacity-Pmax-est'], :] = dm.array[:, :, idx['pow_capacity-Pmax-est'], :] / (365*24)
dm_capacity_filter = dm_capacity.filter({'Country': ['Switzerland'], 'Years': years_ots, 'Variables': ['pow_capacity-Pmax'],
                                         'Categories1': dm.col_labels['Categories1']})
dm.append(dm_capacity_filter, dim='Variables')

dm_CH = dm_capacity_oilgas_ots.filter({'Country': ['Switzerland']})
dm_CH.append(dm_capacity_waste_ots.filter({'Country': ['Switzerland'], 'Variables': ['pow_capacity-Pmax']}), dim='Categories1')
dm_CH.append(dm_capacity_nuclear.filter({'Country': ['Switzerland'], 'Years': years_ots}), dim='Categories1')
dm_CH.append(dm_capacity_hydro_ots.filter({'Country': ['Switzerland'], 'Years': years_ots}), dim='Categories1')

print('Hello')