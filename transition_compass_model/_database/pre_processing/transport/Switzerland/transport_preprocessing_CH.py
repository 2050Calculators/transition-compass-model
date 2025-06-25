import numpy as np
import pandas as pd
import numpy as np
from _database.pre_processing.api_routines_CH import get_data_api_CH
from model.common.constant_data_matrix_class import ConstantDataMatrix
from model.common.data_matrix_class import DataMatrix
from model.common.io_database import read_database_to_dm
from model.common.auxiliary_functions import create_years_list, linear_fitting, add_missing_ots_years, moving_average
from model.common.auxiliary_functions import my_pickle_dump, sort_pickle
from _database.pre_processing.WorldBank_data_extract import get_WB_data
import pickle
import os
import requests

print('In order for this routine to run you need to download a couple of files and save them locally in ./data/:'
      '- Aviation demand data (pkm) from  "Our World in Data": https://ourworldindata.org/grapher/aviation-demand-efficiency'
      '- Microrecensement analysis for 2005 and 2000 from EUCalc drive PathwayCalc/_database/pre-processing/transport/Switzerland/data/')

def fill_var_nans_based_on_var_curve(dm, var_nan, var_ref):
    # Fills nan in a variable dm based on another variable curve
    dm.operation(var_nan, '/', var_ref, out_col='ratio', unit='%')
    linear_fitting(dm, dm.col_labels['Years'])
    dm.operation('ratio', '*', var_ref, out_col=var_nan+'_ok', unit=dm.units[var_ref])
    dm.filter({'Variables': [var_nan + '_ok']}, inplace=True)
    dm.rename_col(var_nan + '_ok', var_nan, dim='Variables')
    return dm


def save_url_to_file(file_url, local_filename):
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

    return


def compute_passenger_new_fleet(table_id_new_veh, file_new_veh_ots1, file_new_veh_ots2):

    #### New fleet Switzerland + Vaud: 2005 - now
    def get_new_fleet_by_tech_raw(table_id, file):
        # New fleet data are heavy, download them only once
        try:
            with open(file, 'rb') as handle:
                dm_new_fleet = pickle.load(handle)
        except OSError:
            structure, title = get_data_api_CH(table_id, mode='example')
            i = 0
            for month in structure['Month']:
                i = i + 1
                filtering = {'Year': structure['Year'],
                          'Month': [month],
                          'Vehicle group / type': structure['Vehicle group / type'],
                          'Canton': ['Switzerland', 'Vaud'],
                          'Fuel': structure['Fuel']}

                mapping_dim = {'Country': 'Canton',
                               'Years': 'Year',
                               'Variables': 'Month',
                               'Categories1': 'Vehicle group / type',
                               'Categories2': 'Fuel'}

                # Extract new fleet
                dm_new_fleet_month = get_data_api_CH(table_id, mode='extract', filter=filtering, mapping_dims=mapping_dim,
                                                    units=['number'])
                if dm_new_fleet_month is None:
                    raise ValueError(f'API returned None for {month}')
                if i == 1:
                    dm_new_fleet = dm_new_fleet_month.copy()
                else:
                    dm_new_fleet.append(dm_new_fleet_month, dim='Variables')

                current_file_directory = os.path.dirname(os.path.abspath(__file__))
                f = os.path.join(current_file_directory, file)
                with open(f, 'wb') as handle:
                    pickle.dump(dm_new_fleet, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return dm_new_fleet


    ### Passenger new fleet Switzerland + Vaud: 2005 - new
    def extract_passenger_new_fleet_by_tech(dm_new_fleet):
        # Sum all months
        dm_new_fleet.groupby({'tra_passenger_new-vehicles': '.*'}, dim='Variables', regex=True, inplace=True)

        # Keep only passenger car main categories
        main_cat = [cat for cat in dm_new_fleet.col_labels['Categories1'] if '>' in cat]
        passenger_cat = [cat for cat in main_cat if 'Passenger' in cat or 'Motorcycles' in cat]

        # Filter for Passenger vehicles
        dm_pass_new_fleet = dm_new_fleet.filter({'Categories1': passenger_cat}, inplace=False)
        dm_pass_new_fleet.groupby({'LDV': '.*Passenger.*'}, dim='Categories1', regex=True, inplace=True)
        dm_pass_new_fleet.groupby({'2W': '.*Motorcycles.*'}, dim='Categories1', regex=True, inplace=True)

        # Filter new technologies
        # (this is needed to later allocate the vehicle fleet "Other" category to the new technologies)
        new_technologies = ['Hydrogen', 'Diesel-electricity: conventional hybrid',
                            'Petrol-electricity: conventional hybrid',
                            'Petrol-electricity: plug-in hybrid', 'Diesel-electricity: plug-in hybrid',
                            'Gas (monovalent and bivalent)']
        dm_new_tech = dm_pass_new_fleet.filter({'Categories2': new_technologies})

        # Map fuel technology to transport module category
        dict_tech = {'FCEV': ['Hydrogen'], 'BEV': ['Electricity'],
                     'ICE-diesel': ['Diesel', 'Diesel-electricity: conventional hybrid'],
                     'ICE-gasoline': ['Petrol', 'Petrol-electricity: conventional hybrid'],
                     'PHEV-diesel': ['Diesel-electricity: plug-in hybrid'],
                     'PHEV-gasoline': ['Petrol-electricity: plug-in hybrid'],
                     'ICE-gas': ['Gas (monovalent and bivalent)']}
        dm_pass_new_fleet.groupby(dict_tech, dim='Categories2', regex=False, inplace=True)
        dm_pass_new_fleet.drop(col_label='Without motor', dim='Categories2')
        # Check that other categories are only a small contribution
        dm_tmp = dm_pass_new_fleet.normalise(dim='Categories2', inplace=False)
        dm_tmp.filter({'Categories2': ['Other']}, inplace=True)
        # If Other and Without motor are more than 0.1% you should account for it
        if (dm_tmp.array > 0.01).any():
            raise ValueError('"Other" category is greater than 1% of the fleet, it cannot be discarded')

        dm_pass_new_fleet.drop(col_label='Other', dim='Categories2')


        return dm_pass_new_fleet, dm_new_tech


    ### New fleet Switzerland: 1990 - now
    # New registration of road model vehicles
    # download csv file FSO number gr-e-11.03.02.02.01a
    # https://www.bfs.admin.ch/asset/en/30305446
    def get_new_fleet(file, first_year):
        df = pd.read_csv(file)
        for col in df.columns:
            df.rename(columns={col: col + '[number]'}, inplace=True)
        df.rename(columns={'X.1[number]': 'Years'}, inplace=True)
        df['Country'] = 'Switzerland'
        dm_new_fleet_CH = DataMatrix.create_from_df(df, num_cat=0)
        dm_pass_new_fleet_CH = dm_new_fleet_CH.groupby({'tra_passenger_new-vehicles_LDV': 'passenger.*',
                                                        'tra_passenger_new-vehicles_2W': 'motorcycles'},
                                                       dim='Variables', regex=True, inplace=False)
        dm_pass_new_fleet_CH.deepen()

        # Keep only years before 2005
        old_yrs_series = [yr for yr in dm_pass_new_fleet_CH.col_labels['Years'] if yr < first_year]
        dm_pass_new_fleet_CH.filter({'Years': old_yrs_series}, inplace=True)

        return dm_pass_new_fleet_CH


    ### Add new fleet Vaud 1990 - 2004
    def compute_new_fleet_vaud(dm_CH, dm_tech):
        # Extract the cantonal % of the swiss new vehicles in 2005 and uses it to determine Vaud fleet in 1990-2004
        dm_tmp = dm_tech.group_all(dim='Categories2', inplace=False)
        idx = dm_tmp.idx
        arr_shares = dm_tmp.array[idx['Vaud'], 0, :, :] / dm_tmp.array[idx['Switzerland'], 0, :, :]
        idx_ch = dm_CH.idx
        arr_VD = dm_CH.array[idx_ch['Switzerland'], :, :, :] * arr_shares[np.newaxis, :, :]
        dm = dm_CH.copy()
        dm.add(arr_VD, dim='Country', col_label='Vaud')
        return dm


    ### New Passenger fleet for Switzerland and Vaud from 1990-2023
    def compute_new_fleet_tech_all_ots(dm_new_fleet_tech_ots1, dm_pass_new_fleet_ots2):
        # Applied 2005 share by technology to 1990-2005 period
        dm_new_fleet_tech_ots1.normalise(dim='Categories2', inplace=True, keep_original=True)

        if dm_new_fleet_tech_ots1.col_labels['Categories1'] != dm_pass_new_fleet_ots2.col_labels['Categories1']:
            raise ValueError('Make sure categories match')
        if dm_new_fleet_tech_ots1.col_labels['Country'] != dm_pass_new_fleet_ots2.col_labels['Country']:
            raise ValueError('Make sure Country match')
        # Multiply historical data on new fleet by 2005 technology share to obtain fleet by techology
        idx_n = dm_pass_new_fleet_ots2.idx
        idx_s = dm_new_fleet_tech_ots1.idx
        arr = dm_pass_new_fleet_ots2.array[:, :, idx_n['tra_passenger_new-vehicles'], :, np.newaxis] \
              * dm_new_fleet_tech_ots1.array[:, idx_s[first_year], np.newaxis, idx_s['tra_passenger_new-vehicles_share'], :, :]
        arr = arr[:, :, np.newaxis, :, :]

        dm_new_fleet_tech_ots1.drop(dim='Variables', col_label='tra_passenger_new-vehicles_share')
        dm_new_fleet_tech = dm_new_fleet_tech_ots1.copy()
        dm_new_fleet_tech.add(arr, dim='Years', col_label=dm_pass_new_fleet_ots2.col_labels['Years'])
        dm_new_fleet_tech.sort('Years')

        return dm_new_fleet_tech

    # New fleet Switzerland + Vaud: 2005 - now (by technology)
    dm_new_fleet_tech_ots1 = get_new_fleet_by_tech_raw(table_id_new_veh, file_new_veh_ots1)
    # Passenger new fleet Switzerland + Vaud: 2005 - new (by technology)
    dm_pass_new_fleet_tech_ots1, dm_new_tech = extract_passenger_new_fleet_by_tech(dm_new_fleet_tech_ots1)
    first_year = dm_pass_new_fleet_tech_ots1.col_labels['Years'][0]
    # New fleet Switzerland 1990 - 2004

    dm_pass_new_fleet_CH_ots2 = get_new_fleet(file_new_veh_ots2, first_year)
    # Add new fleet Vaud 1990 - 2004
    dm_pass_new_fleet_ots2 = compute_new_fleet_vaud(dm_pass_new_fleet_CH_ots2, dm_pass_new_fleet_tech_ots1)
    # Compute technology shares 1990 - 2004
    dm_new_fleet_tech = compute_new_fleet_tech_all_ots(dm_pass_new_fleet_tech_ots1, dm_pass_new_fleet_ots2)

    return dm_new_fleet_tech, dm_new_tech


def get_passenger_stock_fleet_by_tech_raw(table_id, file):
    # New fleet data are heavy, download them only once
    try:
        with open(file, 'rb') as handle:
            dm_fleet = pickle.load(handle)
    except OSError:
        structure, title = get_data_api_CH(table_id, mode='example')
        # Keep only passenger car main categories
        main_cat = [cat for cat in structure['Vehicle group / type'] if '>' in cat]
        passenger_cat = [cat for cat in main_cat if 'Passenger' in cat or 'Motorcycles' in cat]

        filtering = {'Year': structure['Year'],
                     'Year of first registration': structure['Year of first registration'],
                     'Vehicle group / type': passenger_cat,
                     'Canton': ['Switzerland', 'Vaud'],
                     'Fuel': structure['Fuel']}

        mapping_dim = {'Country': 'Canton',
                       'Years': 'Year',
                       'Variables': 'Year of first registration',
                       'Categories1': 'Vehicle group / type',
                       'Categories2': 'Fuel'}

        # Extract new fleet
        dm_fleet = get_data_api_CH(table_id, mode='extract', filter=filtering, mapping_dims=mapping_dim,
                                            units=['number']*len(structure['Year of first registration']))

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, 'wb') as handle:
            pickle.dump(dm_fleet, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Group all vehicles independently of immatriculation data
    dm_fleet.groupby({'tra_passenger_vehicle-fleet': '.*'}, dim='Variables', regex=True, inplace=True)
    # Group passenger vehicles as LDV and motorcycles as 2W
    dm_fleet.groupby({'LDV': '.*Passenger.*', '2W': '.*Motorcycles'}, dim='Categories1', regex=True, inplace=True)
    # Map fuel technology to transport module category. Other category cannot be removed as it is above 1%
    dict_tech = {'BEV': ['Electricity'],
                 'ICE-diesel': ['Diesel'],
                 'ICE-gasoline': ['Petrol']}
    dm_fleet.groupby(dict_tech, dim='Categories2', regex=False, inplace=True)
    dm_fleet.drop(dim='Categories2', col_label='Without motor')

    return dm_fleet


def allocate_other_to_new_technologies(dm_fleet, dm_new_tech):

    dm_fleet_other = dm_fleet.filter({'Categories2': ['Other']})
    #dm_fleet_other.group_all('Categories2')
    dm_fleet_other.filter({'Years': dm_new_tech.col_labels['Years']}, inplace=True)
    dm_fleet.drop(dim='Categories2', col_label='Other')

    # Assuming none of the vehicles from 2005 has gone to waste (simplification),
    # the fleet at year Y will be the sum of new_fleet for years <= Y
    # The results are then normalised and the shares are used to allocate other
    dm_new_tech_cumul = dm_new_tech.copy()
    dm_new_tech_cumul.array = np.cumsum(dm_new_tech.array, axis=1)
    dm_new_tech_cumul.normalise(dim='Categories2', inplace=True, keep_original=False)
    # The normalisation returns nan if all values are 0. Then replace with 0
    np.nan_to_num(dm_new_tech_cumul.array, nan=0.0, copy=False)

    # Allocate
    idx = dm_fleet_other.idx
    arr = dm_fleet_other.array[:, :, :, :, idx['Other'], np.newaxis]*dm_new_tech_cumul.array
    dm_fleet_other.add(arr, dim='Categories2', col_label=dm_new_tech_cumul.col_labels['Categories2'])
    dm_fleet_other.drop(dim='Categories2', col_label='Other')

    # Map fuel technology to transport module category
    dict_tech = {'FCEV': ['Hydrogen'],
                 'ICE-diesel': ['Diesel-electricity: conventional hybrid'],
                 'ICE-gasoline': ['Petrol-electricity: conventional hybrid'],
                 'PHEV-diesel': ['Diesel-electricity: plug-in hybrid'],
                 'PHEV-gasoline': ['Petrol-electricity: plug-in hybrid'],
                 'ICE-gas': ['Gas (monovalent and bivalent)']}
    dm_fleet_other.groupby(dict_tech, dim='Categories2', regex=False, inplace=True)

    dm_fleet_new = dm_fleet.filter({'Years': dm_fleet_other.col_labels['Years']}, inplace=False)
    dm_fleet.drop(dim='Years', col_label=dm_fleet_other.col_labels['Years'])

    idx_f = dm_fleet.idx
    idx_o = dm_fleet_other.idx
    # Diesel
    dm_fleet_new.array[:, :, :, :, idx_f['ICE-diesel']] = \
        dm_fleet_new.array[:, :, :, :, idx_f['ICE-diesel']] \
        + dm_fleet_other.array[:, :, :, :, idx_o['ICE-diesel']]
    # Petrol
    dm_fleet_new.array[:, :, :, :, idx_f['ICE-gasoline']] = \
        dm_fleet_new.array[:, :, :, :, idx_f['ICE-gasoline']] \
        + dm_fleet_other.array[:, :, :, :, idx_o['ICE-gasoline']]
    dm_fleet_other.drop(dim='Categories2', col_label=['ICE-gasoline', 'ICE-diesel'])

    dm_fleet_new.append(dm_fleet_other, dim='Categories2')

    dm_fleet.add(0.0, dummy=True, dim='Categories2', col_label=dm_fleet_other.col_labels['Categories2'])
    dm_fleet.append(dm_fleet_new, dim='Years')

    return dm_fleet


def get_passenger_transport_demand_SweetCROSS(file_url, local_filename, years_ots):
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
    df.columns = df.columns.str.replace(' (Bpkm)', '')
    df_T = pd.melt(df, id_vars=['Scenario', 'Mode'], var_name='Years', value_name='tra_passenger_demand')
    df_T['Country'] = 'Switzerland'
    # Pivot the dataframe
    df_pivot = df_T.pivot_table(index=['Country', 'Years'], columns=['Scenario', 'Mode'], values='tra_passenger_demand',
                                aggfunc='sum')  # Use 'sum' or 'first', depending on how you want to aggregate

    # Flatten the multi-level columns
    df_pivot.columns = [f'tra_passenger_transport-demand_{sce}_{mod}[Bpkm]' for sce, mod in df_pivot.columns]

    df_pivot.reset_index(inplace=True)
    dm_demand = DataMatrix.create_from_df(df_pivot, num_cat=2)
    dm_demand.switch_categories_order()

    # Rename categories to match calculator's
    map_cat = {'bike': ['Bikes', 'Mopeds and fast e-bikes'], '2W': ['Motorcycles'], 'walk': ['On foot'],
               'bus': ['Buses', 'Trolleybuses'], 'rail': ['Passenger rail'], 'LDV': ['Personal cars'],
               'metrotram': ['Trams']}
    dm_demand.groupby(map_cat, dim='Categories1', inplace=True)
    dm_demand.drop(col_label='Other private', dim='Categories1')

    # Extract ots years
    years_tmp = [y for y in dm_demand.col_labels['Years'] if y < 2025]
    dm_demand_ots = dm_demand.filter({'Years': years_tmp}, inplace=False)
    dm_demand_ots.filter({'Categories2': ['Reference']}, inplace=True)
    dm_demand_ots.group_all('Categories2')

    # Extract fts scenarios
    dm_demand_fts = dm_demand
    dm_demand_fts.drop(dim='Years', col_label=dm_demand_ots.col_labels['Years'])
    dm_demand_fts.operation('Low', '+', 'Reference', out_col='Medium-Low', dim='Categories2')
    idx = dm_demand_fts.idx
    dm_demand_fts.array[:, :, :, :, idx['Medium-Low']] = dm_demand_fts.array[:, :, :, :, idx['Medium-Low']]/2

    dict_demand_fts = dict()
    dict_demand_fts[1] = dm_demand_fts.filter({'Categories2': ['Reference']}, inplace=False)
    dict_demand_fts[1].group_all('Categories2')
    dict_demand_fts[2] = dm_demand_fts.filter({'Categories2': ['High']}, inplace=False)
    dict_demand_fts[2].group_all('Categories2')
    dict_demand_fts[4] = dm_demand_fts.filter({'Categories2': ['Low']}, inplace=False)
    dict_demand_fts[4].group_all('Categories2')

    dm_demand_ots.change_unit('tra_passenger_transport-demand', 1e9, old_unit='Bpkm', new_unit='pkm')

    # Extrapolate years before 2000, but skip 2020 (because of Covid)
    dm_demand_ots_2020 = dm_demand_ots.filter({'Years': [2020]})
    dm_demand_ots.drop(dim='Years', col_label=2020)
    linear_fitting(dm_demand_ots, years_ots)
    dm_demand_ots.drop(dim='Years', col_label=2020)
    dm_demand_ots.append(dm_demand_ots_2020, dim='Years')
    dm_demand_ots.sort('Years')

    return dm_demand_ots, dict_demand_fts



def get_public_transport_data(file_url, local_filename, years_ots):

    def get_excel_file_sheets(file_url, local_filename):
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
        # The excel file contains multiple sheets
        # load index sheet
        df_index = pd.read_excel(local_filename)
        df_index.drop(columns=['Unnamed: 0', 'Unnamed: 3', 'Unnamed: 5'], inplace=True)
        df_index.dropna(how='any', inplace=True)
        # Change colummns header
        df_index.rename(columns={'Unnamed: 1': 'Sheet', 'Unnamed: 2': 'Theme'}, inplace=True)
        df_index = df_index[1:]  # take the data less the header row
        sheet_fleet = list(df_index.loc[df_index['Theme'] == 'Moyens de transport: véhicules '].Sheet)[0]
        sheet_passenger = list(df_index.loc[df_index['Theme'] == 'Voyageurs transportés'].Sheet)[0]
        sheet_pkm = list(df_index.loc[df_index['Theme'] == 'Voyageurs-kilomètres'].Sheet)[0]
        sheet_vkm = list(df_index.loc[df_index['Theme'] == 'Utilisation du système: prestations kilométriques, ponctualité et indices des prix'].Sheet)[0]

        df_fleet = pd.read_excel(local_filename, sheet_name=sheet_fleet.replace('.', '_'))
        df_nb_passenger = pd.read_excel(local_filename, sheet_name=sheet_passenger.replace('.', '_'))
        df_pkm = pd.read_excel(local_filename, sheet_name=sheet_pkm.replace('.', '_'))
        df_vkm = pd.read_excel(local_filename, sheet_name=sheet_vkm.replace('.', '_'))

        DF_dict = {'Passenger fleet': df_fleet,
                   'Passenger transported': df_nb_passenger,
                   'Passenger pkm': df_pkm,
                   'Passenger vkm': df_vkm}

        return DF_dict

    def extract_public_passenger_fleet(df, years_ots):
        # Change headers
        new_header = df.iloc[2]
        new_header.values[0] = 'Variables'
        df.columns = new_header
        df = df[3:].copy()
        # Remove nans and empty columns/rows
        df.drop(columns=np.nan, inplace=True)
        df.set_index('Variables', inplace=True)
        df.dropna(axis=0, how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        # Filter rows that contain at least one number (integer or float)
        df = df[df.apply(lambda row: row.map(pd.api.types.is_number), axis=1).any(axis=1)]
        df = df.loc[:, df.apply(lambda col: col.map(pd.api.types.is_number)).any(axis=0)].copy()
        # Extract only the data we are interested in:
        # Electrique and Diesel here below refers to the engine type of rails
        vehicles_vars = ['Voitures voyageurs (voitures de commande isolées, automotrices et éléments de rames automotrices inclus)',
                         'Trolleybus', 'Tram', 'Autobus', 'Électrique', 'Diesel ']
        df_pass_public_veh = df.loc[vehicles_vars].copy()
        df_pass_public_veh = df_pass_public_veh.apply(lambda col: pd.to_numeric(col, errors='coerce'))
        #df_pass_public_veh = df_pass_public_veh.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
        df_pass_public_veh.reset_index(inplace=True)
        df_pass_public_veh['Variables'] = df_pass_public_veh['Variables'].str[:10]
        df_pass_public_veh = df_pass_public_veh.groupby(['Variables']).sum()
        df_pass_public_veh.reset_index(inplace=True)

        # Pivot the dataframe
        df_pass_public_veh['Country'] = 'Switzerland'
        df_T = pd.melt(df_pass_public_veh, id_vars=['Variables', 'Country'], var_name='Years', value_name='values')
        df_pivot = df_T.pivot_table(index=['Country', 'Years'], columns=['Variables'], values='values', aggfunc='sum')
        df_pivot = df_pivot.add_suffix('[number]')
        df_pivot = df_pivot.add_prefix('tra_passenger_vehicle-fleet_')
        df_pivot.reset_index(inplace=True)

        dm_fleet = DataMatrix.create_from_df(df_pivot, num_cat=1)
        map_cat = {'bus_CEV': ['Trolleybus'], 'bus_ICE-diesel': ['Autobus'],
                   'metrotram_mt': ['Tram'], 'rail': ['Voitures v'], 'rail_CEV': ['Électrique'],
                   'rail_ICE-diesel': ['Diesel ']}
        dm_fleet.groupby(map_cat, dim='Categories1', inplace=True)
        mask = dm_fleet.array == 0
        dm_fleet.array[mask] = np.nan
        add_missing_ots_years(dm_fleet, startyear=dm_fleet.col_labels['Years'][0], baseyear=dm_fleet.col_labels['Years'][-1])
        dm_fleet.fill_nans(dim_to_interp='Years')
        # Extrapolate based on 2010 values onwards
        years_init = [y for y in dm_fleet.col_labels['Years'] if y >= 2010]
        dm_tmp = dm_fleet.filter({'Years': years_init})
        years_extract = [y for y in years_ots if y >= 2010]
        linear_fitting(dm_tmp, years_extract)
        # Join historical values with extrapolated ones
        dm_fleet.drop(dim='Years', col_label=years_init)
        dm_fleet.append(dm_tmp, dim='Years')

        # We have extracted the total number of wagon (as 'rail')
        # and then the number of motorised wagon by electric and diesel (as 'rail_CEV', 'rail_ICE-diesel')
        # we want to distribute the number of wagon by diesel and electric
        dm_rail_tech = dm_fleet.filter({'Categories1': ['rail_CEV', 'rail_ICE-diesel']})
        dm_rail_tech.normalise(dim='Categories1')

        idx_f = dm_fleet.idx
        idx_r = dm_rail_tech.idx
        dm_fleet.array[:, :, :, idx_f['rail_ICE-diesel']] = dm_fleet.array[:, :, :, idx_f['rail']]\
                                                            * dm_rail_tech.array[:, :, :, idx_r['rail_ICE-diesel']]
        dm_fleet.array[:, :, :, idx_f['rail_CEV']] = dm_fleet.array[:, :, :, idx_f['rail']]\
                                                     * dm_rail_tech.array[:, :, :, idx_r['rail_CEV']]
        dm_fleet.drop(col_label=['rail'], dim='Categories1')
        dm_fleet.deepen()

        return dm_fleet

    def extract_public_passenger_pkm(df, years_ots):
        # Change headers
        new_header = df.iloc[2]
        new_header.values[0] = 'Variables'
        df.columns = new_header
        df = df[3:].copy()
        # Remove nans and empty columns/rows
        df.drop(columns=np.nan, inplace=True)
        df.set_index('Variables', inplace=True)
        df.dropna(axis=0, how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        # Filter rows that contain at least one number (integer or float)
        df = df[df.apply(lambda row: row.map(pd.api.types.is_number), axis=1).any(axis=1)]
        df = df.loc[:, df.apply(lambda col: col.map(pd.api.types.is_number)).any(axis=0)].copy()
        # Vars to keep
        vars_to_keep = ['Chemins de fer', 'Chemins de fer à crémaillère', 'Tram', 'Trolleybus', 'Autobus']
        df_pkm = df.loc[vars_to_keep]
        # Convert ... to numerics
        df_pkm = df_pkm.apply(lambda col: pd.to_numeric(col, errors='coerce'))
        #df_pkm = df_pkm.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
        df_pkm.reset_index(inplace=True)

        # Pivot the dataframe
        df_pkm['Country'] = 'Switzerland'
        df_T = pd.melt(df_pkm, id_vars=['Variables', 'Country'], var_name='Years', value_name='values')
        df_pivot = df_T.pivot_table(index=['Country', 'Years'], columns=['Variables'], values='values', aggfunc='sum')
        df_pivot = df_pivot.add_suffix('[Mpkm]')
        df_pivot = df_pivot.add_prefix('tra_passenger_transport-demand_')
        df_pivot.reset_index(inplace=True)

        # Create datamatrix
        dm = DataMatrix.create_from_df(df_pivot, num_cat=1)
        # Convert 0 to np.nan
        cat_map = {'bus': ['Autobus', 'Trolleybus'], 'metrotram': ['Tram'],
                   'rail': ['Chemins de fer', 'Chemins de fer à crémaillère']}
        dm.groupby(cat_map, dim='Categories1', inplace=True)
        mask = dm.array == 0
        dm.array[mask] = np.nan

        # Extrapolate based on 2020 values onwards
        years_init = [y for y in dm.col_labels['Years'] if y >= 2020]
        dm_tmp = dm.filter({'Years': years_init})
        years_extract = [y for y in years_ots if y >= 2020]
        linear_fitting(dm_tmp, years_extract)
        # Join historical values with extrapolated ones
        dm.drop(dim='Years', col_label=years_init)
        dm.append(dm_tmp, dim='Years')
        # Back-extrapolate (use data until 2019 - Covid-19)
        years_tmp = [y for y in dm.col_labels['Years'] if y <= 2019]
        dm_tmp = dm.filter({'Years': years_tmp})
        years_extract = [y for y in years_ots if y <= 2019]
        linear_fitting(dm_tmp, years_extract)
        dm.drop(dim='Years', col_label=years_tmp)
        dm.append(dm_tmp, dim='Years')
        dm.sort('Years')
        dm.change_unit('tra_passenger_transport-demand', 1e6, old_unit='Mpkm', new_unit='pkm')

        return dm

    def extract_public_passenger_vkm(df, years_ots):
        # Change headers
        new_header = df.iloc[2]
        new_header.values[0] = 'Variables'
        df.columns = new_header
        df = df[3:].copy()
        # Remove nans and empty columns/rows
        df.drop(columns=np.nan, inplace=True)
        df.set_index('Variables', inplace=True)
        df.dropna(axis=0, how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        # Filter rows that contain at least one number (integer or float)
        df = df[df.apply(lambda row: row.map(pd.api.types.is_number), axis=1).any(axis=1)]
        df = df.loc[:, df.apply(lambda col: col.map(pd.api.types.is_number)).any(axis=0)].copy()
        # Vars to keep
        vars_to_keep = ['Chemins de fer', 'Chemins de fer à crémaillère', 'Tram', 'Trolleybus', 'Autobus']
        df_vkm = df.loc[vars_to_keep].copy()
        # Convert ... to numerics
        df_vkm = df_vkm.apply(lambda col: pd.to_numeric(col, errors='coerce'))
        #df_vkm = df_vkm.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
        df_vkm.reset_index(inplace=True)

        # Pivot the dataframe
        df_vkm['Country'] = 'Switzerland'
        df_T = pd.melt(df_vkm, id_vars=['Variables', 'Country'], var_name='Years', value_name='values')
        df_pivot = df_T.pivot_table(index=['Country', 'Years'], columns=['Variables'], values='values', aggfunc='sum')
        df_pivot = df_pivot.add_suffix('[1000vkm]')
        df_pivot = df_pivot.add_prefix('tra_passenger_transport-demand-vkm_')
        df_pivot.reset_index(inplace=True)

        # Create datamatrix
        dm = DataMatrix.create_from_df(df_pivot, num_cat=1)

        # Convert 0 to np.nan
        cat_map = {'bus': ['Autobus', 'Trolleybus'], 'metrotram': ['Tram'],
                   'rail': ['Chemins de fer', 'Chemins de fer à crémaillère']}
        dm.groupby(cat_map, dim='Categories1', inplace=True)
        mask = dm.array == 0
        dm.array[mask] = np.nan

        # Extrapolate based on 2010 values onwards
        years_init = [y for y in dm.col_labels['Years'] if y >= 2015 and y != 2020]
        dm_tmp = dm.filter({'Years': years_init})
        years_extract = [y for y in years_ots if y >= 2015]
        linear_fitting(dm_tmp, years_extract)
        dm_tmp.drop(dim='Years', col_label=2020)
        # Join historical values with extrapolated ones
        dm.drop(dim='Years', col_label=years_init)
        dm.append(dm_tmp, dim='Years')
        dm.sort('Years')
        linear_fitting(dm, years_ots)
        dm.change_unit('tra_passenger_transport-demand-vkm', 1000, old_unit='1000vkm', new_unit='vkm')

        return dm

    DF_dict = get_excel_file_sheets(file_url, local_filename)
    dm_public_fleet = extract_public_passenger_fleet(DF_dict['Passenger fleet'], years_ots)
    dm_public_demand_pkm = extract_public_passenger_pkm(DF_dict['Passenger pkm'], years_ots)
    dm_public_demand_vkm = extract_public_passenger_vkm(DF_dict['Passenger vkm'], years_ots)

    DM_public = {'public_fleet': dm_public_fleet, 'public_demand-pkm': dm_public_demand_pkm,
                 'public_demand-vkm': dm_public_demand_vkm}

    return DM_public


def get_vehicle_efficiency(table_id, file, years_ots, var_name):
    # New fleet data are heavy, download them only once
    try:
        with open(file, 'rb') as handle:
            dm_veh_eff = pickle.load(handle)
            print(f'The vehicle efficienty is read from file {file}. Delete it if you want to update data from api.')
    except OSError:
        structure, title = get_data_api_CH(table_id, mode='example', language='fr')
        i = 0
        # The table is too big to be downloaded at once
        for eu_class in structure["Classe d'émission selon l'UE"]:
            for part in structure['Filtre à particules']:
                i = i + 1
                filtering = {'Année': structure['Année'],
                             'Carburant': structure['Carburant'],
                             'Puissance': structure['Puissance'],
                             'Canton': ['Suisse', 'Vaud'],
                             "Classe d'émission selon l'UE": eu_class,
                             'Émissions de CO2 par km (NEDC)': structure['Émissions de CO2 par km (NEDC)'],
                             'Filtre à particules': part}

                mapping_dim = {'Country': 'Canton',
                               'Years': 'Année',
                               'Variables': 'Puissance',
                               'Categories1': 'Carburant',
                               'Categories2': 'Émissions de CO2 par km (NEDC)'}

                # Extract new fleet
                dm_veh_eff_cl = get_data_api_CH(table_id, mode='extract', filter=filtering, mapping_dims=mapping_dim,
                                                units=['gCO2/km']*len(structure['Puissance']), language='fr')
                dm_veh_eff_cl.array = np.nan_to_num(dm_veh_eff_cl.array)

                if dm_veh_eff_cl is None:
                    raise ValueError(f'API returned None for {eu_class}')
                if i == 1:
                    dm_veh_eff = dm_veh_eff_cl.copy()
                else:
                    dm_veh_eff.array = dm_veh_eff.array + dm_veh_eff_cl.array

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, 'wb') as handle:
            pickle.dump(dm_veh_eff, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Distribute Inconnu on other categories based on their share
    cat_other = [cat for cat in dm_veh_eff.col_labels['Categories2'] if cat != 'Inconnu']
    dm_other = dm_veh_eff.filter({'Categories2': cat_other}, inplace=False)
    dm_other.normalise(dim='Categories2', inplace=True)
    idx = dm_veh_eff.idx
    arr_inc = dm_veh_eff.array[:, :, :, :, idx['Inconnu'], np.newaxis] * dm_other.array
    dm_veh_eff.drop(dim='Categories2', col_label='Inconnu')
    dm_veh_eff.array = dm_veh_eff.array + arr_inc

    # Remove fuel type "Autre" (there are only very few car in this category)
    dm_veh_eff.drop(dim='Categories1', col_label='Autre')

    # Group categories1 according to model
    map_cat = {'ICE-diesel': ['Diesel', 'Diesel-électrique: hybride normal'],
               'ICE-gasoline': ['Essence', 'Essence-électrique: hybride normal'],
               'ICE-gas': ['Gaz (monovalent et bivalent)'],
               'BEV': ['Électrique'],
               'FCEV': ['Hydrogène'],
               'PHEV-diesel': ['Diesel-électrique: hybride rechargeable'],
               'PHEV-gasoline': ['Essence-électrique: hybride rechargeable']
               }
    dm_veh_eff.groupby(map_cat, dim='Categories1', inplace=True)

    # Do this to have realistic curves
    mask = dm_veh_eff.array == 0
    dm_veh_eff.array[mask] = np.nan

    # Flat extrapolation
    years_to_add = [year for year in years_ots if year not in dm_veh_eff.col_labels['Years']]
    dm_veh_eff.add(np.nan, dummy=True, col_label=years_to_add, dim='Years')
    dm_veh_eff.sort(dim='Years')
    dm_veh_eff.fill_nans(dim_to_interp='Years')

    dm_veh_eff.groupby({var_name: '.*'}, dim='Variables', regex=True, inplace=True)

    # Clean grams CO2 category and perform weighted average
    # cols are e.g '0 - 50 g' -> '0-50' -> 25
    dm_veh_eff.rename_col_regex(' g', '', dim='Categories2')
    dm_veh_eff.rename_col_regex(' ', '', dim='Categories2')
    dm_veh_eff.rename_col('Plusde300', '300-350', dim='Categories2')
    cat2_list_old = dm_veh_eff.col_labels['Categories2']
    co2_km = []
    for i in range(len(cat2_list_old)):
        old_cat = cat2_list_old[i]
        new_cat = float(old_cat.split('-')[0]) + float(old_cat.split('-')[1])/2
        co2_km.append(new_cat)
    co2_arr = np.array(co2_km)
    dm_veh_eff.normalise(dim='Categories2', inplace=True)
    dm_veh_eff.array = dm_veh_eff.array*co2_arr[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    dm_veh_eff.group_all(dim='Categories2')

    dm_veh_eff.change_unit(var_name, 1, old_unit='%', new_unit='gCO2/km')


    for i in range(2):
        window_size = 3  # Change window size to control the smoothing effect
        data_smooth = moving_average(dm_veh_eff.array, window_size, axis=dm_veh_eff.dim_labels.index('Years'))
        dm_veh_eff.array[:, 1:-1, ...] = data_smooth

    # Add LDV
    dm_veh_eff_LDV = DataMatrix.based_on(dm_veh_eff.array[..., np.newaxis], dm_veh_eff, change={'Categories2': ['LDV']},
                                         units=dm_veh_eff.units)
    dm_veh_eff_LDV.switch_categories_order()
    dm_veh_eff_LDV.rename_col('Suisse', 'Switzerland', dim='Country')

    return dm_veh_eff_LDV


def get_new_vehicle_efficiency(table_id, file, years_ots, var_name):
    # New fleet data are heavy, download them only once
    try:
        with open(file, 'rb') as handle:
            dm_veh_eff = pickle.load(handle)
            print(f'The vehicle efficienty is read from file {file}. Delete it if you want to update data from api.')
    except OSError:
        structure, title = get_data_api_CH(table_id, mode='example', language='fr')
        i = 0
        # The table is too big to be downloaded at once
        for eu_class in structure["Classe d'émission selon l'UE"]:
            i = i + 1
            filtering = {'Année': structure['Année'],
                         'Carburant': structure['Carburant'],
                         'Puissance': structure['Puissance'],
                         'Canton': ['Suisse', 'Vaud'],
                         "Classe d'émission selon l'UE": eu_class,
                         'Émissions de CO2 par km (NEDC/WLTP)': structure['Émissions de CO2 par km (NEDC/WLTP)']}

            mapping_dim = {'Country': 'Canton',
                           'Years': 'Année',
                           'Variables': 'Puissance',
                           'Categories1': 'Carburant',
                           'Categories2': 'Émissions de CO2 par km (NEDC/WLTP)'}

            # Extract new fleet
            dm_veh_eff_cl = get_data_api_CH(table_id, mode='extract', filter=filtering, mapping_dims=mapping_dim,
                                            units=['gCO2/km']*len(structure['Puissance']), language='fr')
            dm_veh_eff_cl.array = np.nan_to_num(dm_veh_eff_cl.array)

            if dm_veh_eff_cl is None:
                raise ValueError(f'API returned None for {eu_class}')
            if i == 1:
                dm_veh_eff = dm_veh_eff_cl.copy()
            else:
                dm_veh_eff.array = dm_veh_eff.array + dm_veh_eff_cl.array

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, 'wb') as handle:
            pickle.dump(dm_veh_eff, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Do this to have realistic curves
    mask = dm_veh_eff.array == 0
    dm_veh_eff.array[mask] = np.nan

    # Flat extrapolation
    years_to_add = [year for year in years_ots if year not in dm_veh_eff.col_labels['Years']]
    dm_veh_eff.add(np.nan, dummy=True, col_label=years_to_add, dim='Years')
    dm_veh_eff.sort(dim='Years')
    dm_veh_eff.fill_nans(dim_to_interp='Years')

    # Explore Inconnu category
    # -> The data seem to be good only from 2016 to 2020, still the "Inconnu" share is big
    dm_norm = dm_veh_eff.normalise(dim='Categories2', inplace=False)
    idx = dm_norm.idx
    for country in dm_veh_eff.col_labels['Country']:
        for year in dm_veh_eff.col_labels['Years']:
            for cat in dm_veh_eff.col_labels['Categories1']:
                # If "Inconnu" is more than 20% remove the data points
                if dm_norm.array[idx[country], idx[year], 0, idx[cat], idx['Inconnu']] > 0.2:
                    dm_veh_eff.array[idx[country], idx[year], 0, idx[cat], :] = np.nan

    for i in range(2):
        window_size = 3  # Change window size to control the smoothing effect
        data_smooth = moving_average(dm_veh_eff.array, window_size, axis=dm_veh_eff.dim_labels.index('Years'))
        dm_veh_eff.array[:, 1:-1, ...] = data_smooth

    # Distribute Inconnu on other categories based on their share
    cat_other = [cat for cat in dm_veh_eff.col_labels['Categories2'] if cat != 'Inconnu']
    dm_other = dm_veh_eff.filter({'Categories2': cat_other}, inplace=False)
    dm_other.normalise(dim='Categories2', inplace=True)
    dm_other.array = np.nan_to_num(dm_other.array)
    idx = dm_veh_eff.idx
    arr_inc = np.nan_to_num(dm_veh_eff.array[:, :, :, :, idx['Inconnu'], np.newaxis]) * dm_other.array
    dm_veh_eff.drop(dim='Categories2', col_label='Inconnu')
    dm_veh_eff.array = dm_veh_eff.array + arr_inc

    # Remove fuel type "Autre" (there are only very few car in this category)
    dm_veh_eff.drop(dim='Categories1', col_label='Autre')

    # Group categories1 according to model
    map_cat = {'ICE-diesel': ['Diesel', 'Diesel-électrique: hybride normal'],
               'ICE-gasoline': ['Essence', 'Essence-électrique: hybride normal'],
               'ICE-gas': ['Gaz (monovalent et bivalent)'],
               'BEV': ['Électrique'],
               'FCEV': ['Hydrogène'],
               'PHEV-diesel': ['Diesel-électrique: hybride rechargeable'],
               'PHEV-gasoline': ['Essence-électrique: hybride rechargeable']
               }
    dm_veh_eff.groupby(map_cat, dim='Categories1', inplace=True)

    dm_veh_eff.groupby({var_name: '.*'}, dim='Variables', regex=True, inplace=True)

    # Clean grams CO2 category and perform weighted average
    # cols are e.g '0 - 50 g' -> '0-50' -> 25
    dm_veh_eff.rename_col_regex(' g', '', dim='Categories2')
    dm_veh_eff.rename_col_regex(' ', '', dim='Categories2')
    dm_veh_eff.rename_col('Plusde300', '300-350', dim='Categories2')
    cat2_list_old = dm_veh_eff.col_labels['Categories2']
    co2_km = []
    for i in range(len(cat2_list_old)):
        old_cat = cat2_list_old[i]
        new_cat = float(old_cat.split('-')[0]) + float(old_cat.split('-')[1])/2
        co2_km.append(new_cat)
    dm_veh_eff.normalise(dim='Categories2', inplace=True)
    dm_veh_eff.array = dm_veh_eff.array*np.array(co2_km)
    dm_veh_eff.group_all(dim='Categories2')
    dm_veh_eff.change_unit(var_name, 1, old_unit='%', new_unit='gCO2/km')

    # Add LDV
    dm_veh_eff_LDV = DataMatrix.based_on(dm_veh_eff.array[..., np.newaxis], dm_veh_eff, change={'Categories2': ['LDV']},
                                         units=dm_veh_eff.units)
    dm_veh_eff_LDV.switch_categories_order()
    dm_veh_eff_LDV.rename_col('Suisse', 'Switzerland', dim='Country')

    return dm_veh_eff_LDV


def get_travel_demand_canton_microrecencement_2021(file_url, local_filename, year):
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

    df_VD = pd.read_excel(local_filename, sheet_name='VD')
    df_CH = pd.read_excel(local_filename, sheet_name='CH')

    DF = {'Vaud': df_VD, 'Switzerland': df_CH}
    for country in DF.keys():
        df = DF[country]
        # Only keep column with pkm/cap
        cols = [df.columns[i] for i in [0, 11, 13, 15]]
        df = df[cols].copy()
        df.columns = ['Demographic', 'tra_pkm-cap_Bike-Walk[km/cap]', 'tra_pkm-cap_LDV-2W[km/cap]', 'tra_pkm-cap_Public[km/cap]']
        df.set_index('Demographic', inplace=True)
        df = df.loc['Total'].copy()
        df['Years'] = year
        df['Country'] = country
        df.drop(columns=['Demographic'])
        df = df.to_frame().T
        DF[country] = df

    df = pd.concat((DF['Switzerland'], DF['Vaud']), axis=0)
    dm = DataMatrix.create_from_df(df, num_cat=1)

    return dm


def get_travel_demand_canton_microrecencement_2015(file_url_dict, local_filename_dict, year):

    DF = dict()
    for country in file_url_dict.keys():
        file_url = file_url_dict[country]
        local_filename = local_filename_dict[country]
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

        DF[country] = pd.read_excel(local_filename)

    for country in DF.keys():
        df = DF[country]
        cols = [df.columns[i] for i in [0, 11, 13, 15]]
        df = df[cols].copy()
        df.columns = ['Demographic', 'tra_pkm-cap_Bike-Walk[km/cap]', 'tra_pkm-cap_LDV-2W[km/cap]', 'tra_pkm-cap_Public[km/cap]']
        df.set_index('Demographic', inplace=True)
        df = df.loc['Total'].copy()
        df['Years'] = year
        df['Country'] = country
        df.drop(columns=['Demographic'])
        df = df.to_frame().T
        DF[country] = df

    df = pd.concat((DF['Switzerland'], DF['Vaud']), axis=0)
    dm = DataMatrix.create_from_df(df, num_cat=1)

    return dm


def get_canton_pkm_cap_fact(dm_pkm_cap_groups, years_ots):
    dm_pkm_cap_groups.sort('Years')
    # pkm-cap-fact = pkm-cap_VD/ pkm-cap_CH
    dm_pkm_cap_groups.operation('Vaud', '/', 'Switzerland', out_col='Ratio', dim='Country')
    dm_pkm_cap_groups.drop(col_label=['Vaud', 'Switzerland'], dim='Country')
    dm_pkm_cap_groups.rename_col('Ratio', 'Vaud', dim='Country')
    dm_pkm_cap_groups.rename_col('tra_pkm-cap', 'tra_pkm-cap_fact', dim='Variables')
    dm_pkm_cap_fact = dm_pkm_cap_groups.groupby({'bike': ['Bike-Walk'], 'walk': ['Bike-Walk'],
                                                 'LDV': ['LDV-2W'], '2W': ['LDV-2W'],
                                                 'bus': ['Public'], 'rail': ['Public'], 'metrotram': ['Public']},
                                                dim='Categories1', inplace=False)
    years_to_add = [y for y in years_ots if y not in dm_pkm_cap_fact.col_labels['Years']]
    dm_pkm_cap_fact.add(np.nan, dummy=True, dim='Years', col_label=years_to_add)
    dm_pkm_cap_fact.sort('Years')
    dm_pkm_cap_fact.fill_nans(dim_to_interp='Years')
    return dm_pkm_cap_fact


def compute_canton_passenger_transport_demand_pkm(DF, var_name):

    dm_pkm_CH = DF['km_CH']
    dm_pkm_cap_fact = DF['pkm_cap_ratio']
    dm_pop = DF['pop']

    dm_pkm_CH.sort('Categories1')
    idx = dm_pkm_CH.idx
    idx_p = dm_pop.idx
    arr_cap = dm_pkm_CH.array[idx['Switzerland'], :, idx[var_name], np.newaxis, :] \
              / dm_pop.array[idx_p['Switzerland'], np.newaxis, :, idx_p['lfs_population_total'], np.newaxis, np.newaxis]
    dm_pkm_cap = DataMatrix.based_on(arr_cap, dm_pkm_CH, change={'Variables': ['tra_km-cap']}, units={'tra_pkm-cap': 'pkm/cap'})
    # Compute Vaud pkm/cap
    ## pkm/cap_VD = pkm/cap_CH x ratioVD/CH
    dm_pkm_cap_fact.sort('Categories1')
    idx = dm_pkm_cap_fact.idx
    arr_cap_VD = arr_cap * dm_pkm_cap_fact.array[idx['Vaud'], np.newaxis, :, idx['tra_pkm-cap_fact'], np.newaxis, :]
    dm_pkm_cap.add(arr_cap_VD, dim='Country', col_label='Vaud')
    # Compute Vaud pkm
    idx = dm_pkm_cap.idx
    idx_p = dm_pop.idx
    arr_pkm_VD = dm_pkm_cap.array[idx['Vaud'], :, idx['tra_km-cap'], np.newaxis, :]\
                 * dm_pop.array[idx_p['Vaud'], :, idx_p['lfs_population_total'], np.newaxis, np.newaxis]
    dm_pkm_CH.add(arr_pkm_VD, dim='Country', col_label='Vaud')
    return dm_pkm_CH


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


def get_transport_demand_vkm(file_url, local_filename, years_ots):
    # If file does not exist, it downloads it and creates it
    rows_to_keep = ['en millions de trains-km', 'Tram', 'Trolleybus', 'Autobus', 'Voitures de tourisme',
                    'Cars privés', 'Motocycles']
    new_name = ['rail', 'metrotram', 'bus', 'bus', 'LDV', 'LDV', '2W']
    var_name = 'tra_passenger_transport-demand-vkm'
    unit = 'Mvkm'
    header_row = 0

    save_url_to_file(file_url, local_filename)

    df_latest = pd.read_excel(local_filename)
    df_earlier = pd.read_excel(local_filename, sheet_name='1990-2004')

    df_latest[df_latest.columns[0]] = df_latest[df_latest.columns[0]].str.replace(r'\d+\)|\(\d+\)', '', regex=True).str.strip()
    df_earlier[df_earlier.columns[0]] = df_earlier[df_earlier.columns[0]].str.replace(r'\d+\)|\(\d+\)', '', regex=True).str.strip()

    # Clean df from excel file
    names_map = dict()
    for i, row in enumerate(rows_to_keep):
        names_map[row] = new_name[i]
    dm_latest = df_fso_excel_to_dm(df_latest, header_row, names_map, var_name, unit, num_cat=1)
    names_map.pop('Autobus')
    names_map['Transport par bus'] = 'bus'
    dm_earlier = df_fso_excel_to_dm(df_earlier, header_row, names_map, var_name, unit, num_cat=1)
    dm_earlier.append(dm_latest, dim='Years')
    dm = dm_earlier.copy()

    # Fix 2023 is missing for various transport types
    ## Replace 0 with nan
    mask = (dm.array == 0)
    dm.array[mask] = np.nan

    # Extrapolate for 2023 starting from 2020
    years_gt_2020 = [y for y in years_ots if y >= 2020]
    linear_fitting(dm, years_gt_2020, based_on=years_gt_2020)

    dm.change_unit(var_name, factor=1e6, old_unit=unit, new_unit='vkm')

    # Metrotram has a weird spike in 2004 that is there in the raw data, but I want to remove
    idx = dm.idx
    dm.array[:, idx[2004], :, idx['metrotram']] = np.nan
    dm.fill_nans(dim_to_interp='Years')

    return dm


def get_transport_demand_pkm(file_url, local_filename, years_ots):

    header_row = 1
    rows_to_keep = ['Chemins de fer', 'Chemins de fer à crémaillère', 'Trams', 'Trolleybus', 'Autobus',
                    'Voitures de tourisme', 'Motocycles', 'Cars', 'Bicyclettes, y. c. vélos électriques lents', 'À pied']
    new_name = ['rail', 'rail', 'metrotram', 'bus', 'bus', 'LDV', '2W', 'LDV', 'bike', 'walk']
    var_name = 'tra_passenger_transport-demand'
    unit = 'Mpkm'

    # If file does not exist, it downloads it and creates it
    save_url_to_file(file_url, local_filename)

    df_latest = pd.read_excel(local_filename)
    df_earlier = pd.read_excel(local_filename, sheet_name='1990-2004')

    df_latest[df_latest.columns[0]] = df_latest[df_latest.columns[0]].str.replace(r'\d+\)|\(\d+\)', '', regex=True).str.strip()
    df_earlier[df_earlier.columns[0]] = df_earlier[df_earlier.columns[0]].str.replace(r'\d+\)|\(\d+\)', '', regex=True).str.strip()

    # Clean df from excel file
    names_map = dict()
    for i, row in enumerate(rows_to_keep):
        names_map[row] = new_name[i]
    dm_latest = df_fso_excel_to_dm(df_latest, header_row, names_map, var_name, unit, num_cat=1)
    # The names change from 1990-2004 to 2005-2023
    names_map.pop('Autobus')
    names_map['Transport par bus'] = 'bus'
    names_map.pop('Bicyclettes, y. c. vélos électriques lents')
    names_map['Bicyclettes'] = 'bike'
    names_map.pop('À pied')
    names_map['à pied'] = 'walk'
    dm_earlier = df_fso_excel_to_dm(df_earlier, header_row, names_map, var_name, unit, num_cat=1)
    dm_earlier.append(dm_latest, dim='Years')
    dm = dm_earlier.copy()

    # Fix 2023 is missing for various transport types
    ## Replace 0 with nan
    mask = (dm.array == 0)
    dm.array[mask] = np.nan
    # Extrapolate for 2023 starting from 2020
    years_gt_2020 = [y for y in years_ots if y >= 2020]
    dm_gt_2020 = dm.filter({'Years': years_gt_2020})
    linear_fitting(dm_gt_2020, years_gt_2020)
    years_lt_2008 = [y for y in years_ots if y < 2008]
    dm_lt_2008 = dm.filter({'Years': years_lt_2008})
    linear_fitting(dm_lt_2008, years_lt_2008)
    idx = dm.idx
    dm.array[:, idx[2020]:, ...] = dm_gt_2020.array
    dm.array[:, 0:idx[2008], ...] = dm_lt_2008.array

    dm.change_unit(var_name, factor=1e6, old_unit=unit, new_unit='pkm')

    return dm


def get_pkm_cap_aviation(file_url, local_filename):
    # Extract CH pkm/cap (data available every 5 years)
    save_url_to_file(file_url, local_filename)
    df = pd.read_excel(local_filename)
    filter_1 = ['Average annual mobility per person3 by aeroplane (within Switzerland and abroad), in km']
    new_names_1 = ['aviation']
    names_map = dict()
    for i, row in enumerate(filter_1):
        names_map[row] = new_names_1[i]
    var_name = 'tra_pkm-cap'
    unit = 'pkm/cap'
    header_row = 2
    dm_pkm = df_fso_excel_to_dm(df, header_row, names_map, var_name, unit, num_cat=1)
    return dm_pkm


def get_world_pop(pop_url, local_filename):
    save_url_to_file(pop_url, local_filename)
    df_pop = pd.read_excel(local_filename)
    filter = ['World']
    new_names = ['lfs_population_total']
    names_map = dict()
    for i, row in enumerate(filter):
        names_map[row] = new_names[i]
    var_name = 'tra_passenger'
    unit = 'number'
    header_row = 2
    dm_pop = df_fso_excel_to_dm(df_pop, header_row, names_map, var_name, unit, num_cat=0, keep_first=True,
                                country='World')
    return dm_pop


def compute_pkm_cap_aviation(dm_pkm_cap_aviation_CH_raw, dm_pkm_aviation_WLD, dm_pop_WLD, years_ots):
    dm_pkm_cap_aviation_CH = dm_pkm_cap_aviation_CH_raw.copy()
    dm_pkm_aviation_WLD.filter({'Years': years_ots}, inplace=True)
    dm_pkm_aviation_WLD.change_unit('tra_passenger_transport-demand', factor=1e9, old_unit='Bpkm', new_unit='pkm')
    dm_pkm_aviation_WLD = dm_pkm_aviation_WLD.flatten()

    # pkm_cap_WLD = pkm/pop
    years_WLD = dm_pkm_aviation_WLD.col_labels['Years']
    dm_pop_WLD.filter({'Years': years_WLD}, inplace=True)
    dm_pkm_aviation_WLD.append(dm_pop_WLD, dim='Variables')
    dm_pkm_aviation_WLD.operation('tra_passenger_transport-demand_aviation', '/', 'tra_passenger_lfs_population_total',
                                  out_col='tra_pkm-cap_aviation', unit='pkm/cap')
    dm_pkm_cap_WLD = dm_pkm_aviation_WLD.filter({'Variables': ['tra_pkm-cap_aviation']})
    dm_pkm_cap_WLD.deepen()

    # Make years compatible
    years_CH = dm_pkm_cap_aviation_CH.col_labels['Years']
    years_to_add = [y for y in years_WLD if y not in years_CH]
    dm_pkm_cap_aviation_CH.add(np.nan, dim='Years', col_label=years_to_add, dummy=True)
    dm_pkm_cap_aviation_CH.sort('Years')
    dm_pkm_cap_aviation_CH.filter({'Years': years_WLD}, inplace=True)

    # Replace 0s with np.nans in CH data
    mask = dm_pkm_cap_aviation_CH.array == 0
    dm_pkm_cap_aviation_CH.array[mask] = np.nan

    dm_pkm_cap_WLD.rename_col('World', 'Switzerland', dim='Country')
    dm_pkm_cap_WLD.rename_col('tra_pkm-cap', 'reference_curve', dim='Variables')

    # ratio = pkm_cap_CH / pkm_cap_WLD, for available years (e
    dm_2021 = dm_pkm_cap_aviation_CH.filter({'Years': [2021]})
    idx = dm_pkm_cap_aviation_CH.idx
    dm_pkm_cap_aviation_CH.array[:, idx[2021], ...] = np.nan
    dm_pkm_cap_aviation_CH.append(dm_pkm_cap_WLD, dim='Variables')
    dm_pkm_cap_aviation_CH = fill_var_nans_based_on_var_curve(dm_pkm_cap_aviation_CH, 'tra_pkm-cap', 'reference_curve')

    idx = dm_pkm_cap_aviation_CH.idx
    dm_pkm_cap_aviation_CH.array[:, idx[2021], ...] = dm_2021.array

    # Fix 2020 value
    ## In reality flights in 2020 were lower than in 2021. We take pkm_cap_2020/pkm_cap_2021 for world and apply it to CH
    idx = dm_pkm_cap_WLD.idx
    ratio_2020_2019 = dm_pkm_cap_WLD.array[0, idx[2020], idx['reference_curve'], idx['aviation']] / \
                      dm_pkm_cap_WLD.array[0, idx[2021], idx['reference_curve'], idx['aviation']]
    idx = dm_pkm_cap_aviation_CH.idx
    dm_pkm_cap_aviation_CH.array[:, idx[2020], ...] = dm_pkm_cap_aviation_CH.array[:, idx[2021], ...]*ratio_2020_2019

    dm_pkm_cap_aviation_VD = dm_pkm_cap_aviation_CH.copy()
    dm_pkm_cap_aviation_VD.rename_col('Switzerland', 'Vaud', dim='Country')
    dm_pkm_cap_aviation_CH.append(dm_pkm_cap_aviation_VD, dim='Country')

    linear_fitting(dm_pkm_cap_aviation_CH, years_ots=[2020, 2021, 2022, 2023], based_on=[2020, 2021])

    return dm_pkm_cap_aviation_CH


def get_aviation_fleet(file_url, local_filename):
    # Extract CH pkm/cap (data available every 5 years)
    save_url_to_file(file_url, local_filename)
    df = pd.read_excel(local_filename)
    filter_1 = ['MTOM2 > 5700 kg']
    new_names_1 = ['aviation']
    names_map = dict()
    for i, row in enumerate(filter_1):
        names_map[row] = new_names_1[i]
    var_name = 'tra_passenger_vehicle-fleet'
    unit = 'number'
    header_row = 2
    dm_fleet_aviation = df_fso_excel_to_dm(df, header_row, names_map, var_name, unit, num_cat=1, keep_first=True)

    return dm_fleet_aviation


def downscale_public_fleet_VD(dm_public_fleet, dm_pkm):
    dm_public_fleet.sort('Categories1')
    dm_public_pkm = dm_pkm.filter({'Categories1': dm_public_fleet.col_labels['Categories1']})
    idx = dm_public_pkm.idx
    arr_ratio_pkm = dm_public_pkm.array[idx['Vaud'], :, :, :] / dm_public_pkm.array[idx['Switzerland'], :, :, :]
    idx = dm_public_fleet.idx
    arr_VD = dm_public_fleet.array[idx['Switzerland'], :, :, :, :] * arr_ratio_pkm[..., np.newaxis]
    dm_public_fleet.add(arr_VD, dim='Country', col_label='Vaud')
    return dm_public_fleet


def get_travel_demand_region_microrecencement(file_url=None, local_filename="", year=2000):
    if file_url is not None:
        save_url_to_file(file_url, local_filename)
    df = pd.read_excel(local_filename)
    df = df[['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 5']]
    df.columns = ['Variables', 'Reason', 'Switzerland', 'Vaud']
    df['Variables'] = df['Variables'].ffill()
    # Keep only the sum of all reasons to travel
    df = df.loc[df['Reason'] == 'Tous les motifs'].copy()
    df = df[['Variables', 'Switzerland', 'Vaud']]
    df = df.dropna(subset=['Variables'])

    # Add years col
    df['Years'] = year

    # Clean names for dm
    df['Variables'] = df['Variables'].str.replace("\n", " ")
    df['Variables'] = df['Variables'].str.split(',').str[0]
    df['Variables'] = df['Variables'].str.replace(r'\s*\(.*?\)\s*', '', regex=True)

    groupby_dict = {'walk': 'pied', 'bus': 'Autocar|Car|Bus', 'metrotram': 'Tram', 'bike': 'Vélo',
                    'rail': 'Train', 'LDV': 'Voiture|Taxi', 'aviation': 'Avion',
                    '2W': 'Motocycle|Cyclomoteur'}

    for new_cat, old_cat in groupby_dict.items():
        # Use word boundaries to match full words only
        # Use str.contains to check if old_cat is in the Variables
        mask = df['Variables'].str.contains(old_cat, regex=True)
        # Replace entire cell with new_cat if old_cat is found
        df.loc[mask, 'Variables'] = new_cat

    df = df[df['Variables'].isin(groupby_dict.keys())].copy()

    df_T = pd.melt(df, id_vars=['Variables', 'Years'], var_name='Country', value_name='values')
    df_pivot = df_T.pivot_table(index=['Country', 'Years'], columns=['Variables'], values='values', aggfunc='sum')

    # Add variable name
    df_pivot = df_pivot.add_suffix('[pkm/cap/day]')
    df_pivot = df_pivot.add_prefix('tra_pkm-cap_')
    df_pivot.reset_index(inplace=True)
    # Convert to dm
    dm = DataMatrix.create_from_df(df_pivot, num_cat=1)
    dm.change_unit('tra_pkm-cap', factor=365, old_unit='pkm/cap/day', new_unit='pkm/cap')

    return dm


def extrapolate_missing_pkm_cap_based_on_pkm_CH(dm_pkm_cap, dm_pkm_CH, dm_pop):

    years_to_add = [y for y in years_ots if y not in dm_pkm_cap.col_labels['Years']]
    dm_pkm_cap.add(np.nan, col_label=years_to_add, dummy=True, dim='Years')
    dm_pkm_cap.sort('Years')
    dm_pkm_cap.sort('Categories1')
    idx = dm_pop.idx
    arr_pkm_cap = dm_pkm_CH.array / dm_pop.array[idx['Switzerland'], np.newaxis, :, :, np.newaxis]
    dm_pkm_CH.add(arr_pkm_cap, dim='Variables', col_label='tra_pkm-cap_full', unit='pkm/cap')
    dm_CH = dm_pkm_CH.filter({'Variables': ['tra_pkm-cap_full']})
    dm_CH.append(dm_pkm_cap.filter({'Country': ['Switzerland'], 'Categories1': dm_CH.col_labels['Categories1']}), dim='Variables')
    dm_pkm_cap_new = fill_var_nans_based_on_var_curve(dm_CH, var_nan='tra_pkm-cap', var_ref='tra_pkm-cap_full')
    dm_tmp = dm_pkm_cap_new.copy()
    dm_tmp.rename_col('Switzerland', 'Vaud', dim='Country')
    dm_tmp.rename_col('tra_pkm-cap', 'tra_pkm-cap_CH', dim='Variables')
    dm_tmp.append(dm_pkm_cap.filter({'Country': ['Vaud'], 'Categories1': dm_tmp.col_labels['Categories1']}), dim='Variables')
    dm_pkm_cap_new_VD = fill_var_nans_based_on_var_curve(dm_tmp, var_nan='tra_pkm-cap', var_ref='tra_pkm-cap_CH')
    dm_pkm_cap_new.append(dm_pkm_cap_new_VD, dim='Country')

    return dm_pkm_cap_new


def compute_pkm_aviation_CH(dm_pkm_cap_aviation, dm_pop):
    print('You need to write this better, CODE AV1')
    linear_fitting(dm_pkm_cap_aviation, years_ots=[2020, 2021, 2022, 2023], based_on=[2020, 2021])
    idx = dm_pop.idx
    arr_pkm = dm_pkm_cap_aviation.array * dm_pop.array[idx['Switzerland'], np.newaxis, :, :, np.newaxis]
    dm_pkm_aviation = DataMatrix.based_on(arr_pkm, dm_pkm_cap_aviation,
                                          change={'Variables': ['tra_passenger_transport-demand']},
                                          units={'tra_passenger_transport-demand': 'pkm'})

    return dm_pkm_aviation


def compute_pkm_from_pkm_cap(dm_pkm_cap, dm_pop):
    dm_pkm_cap.sort('Country')
    dm_pop.sort('Country')
    arr = dm_pkm_cap.array * dm_pop.array[:, :, :, np.newaxis]
    dm_pkm = DataMatrix.based_on(arr, dm_pkm_cap, change={'Variables': ['tra_passenger_transport-demand']},
                                units={'tra_passenger_transport-demand': 'pkm'})
    return dm_pkm


def compute_vkm_CH_VD(dm_vkm_CH, dm_pkm_CH, dm_pkm):

    dm_vkm_CH.append(dm_pkm_CH.filter({'Categories1': dm_vkm_CH.col_labels['Categories1']}),
                     dim='Variables')
    dm_vkm_CH.operation('tra_passenger_transport-demand', '/', 'tra_passenger_transport-demand-vkm',
                        out_col='tra_passenger_occupancy', unit='pkm/vkm')
    dm_vkm_CH.sort('Categories1')

    # Extract occupancy and set same occupancy for CH and VD
    dm_occupancy = dm_vkm_CH.filter({'Variables': ['tra_passenger_occupancy']})
    dm_occupancy_VD = dm_occupancy.copy()
    dm_occupancy_VD.rename_col('Switzerland', 'Vaud', dim='Country')
    dm_occupancy.append(dm_occupancy_VD, dim='Country')

    dm_occupancy.append(dm_pkm.filter({'Categories1': dm_vkm_CH.col_labels['Categories1']}), dim='Variables')
    dm_occupancy.operation('tra_passenger_transport-demand', '/', 'tra_passenger_occupancy',
                           out_col='tra_passenger_transport-demand-vkm', unit='vkm')

    dm_vkm = dm_occupancy.filter({'Variables': ['tra_passenger_transport-demand-vkm']})

    return dm_vkm


def create_emissions_factors_cdm(emis, mapping_cat):
    col_labels = dict()
    col_labels['Variables'] = ['cp_tra_emission-factor']
    col_labels['Categories1'] = ['CH4', 'CO2', 'N2O']
    col_labels['Categories2'] = list(mapping_cat.keys())
    arr = np.ones((len(col_labels['Variables']), len(col_labels['Categories1']), len(col_labels['Categories2'])))
    for g, ghg in enumerate(col_labels['Categories1']):
        for f, fuel in enumerate(col_labels['Categories2']):
            simple_fuel_name = mapping_cat[fuel]
            arr[0, g, f] = emis[ghg][simple_fuel_name]
    cdm_emissions = ConstantDataMatrix(col_labels, units={'cp_tra_emission-factor': 'g/MJ'})  # kg/TJ -> Mt/TWh
    # (Mt/TWh -> 10^6 tonnes / 10^9 kWh -> 10^9 kg / 10^9 kWh -> kg / kWh -> kg / (3.6e-6 TJ) -> 1/(3.6e-6) kg/TJ)
    # Turn kg/TJ to kg/(1e6 x MJ) = 1e3 g / 1e6 MJ -> kg/TJ = 1e-3 g/MJ
    cdm_emissions.array = arr * 1e-3
    return cdm_emissions


def convert_eff_from_gCO2_km_to_MJ_km(dm_veh_eff_LDV, cdm_emissions_factors, new_var_name):
    dm_veh_eff_LDV.drop('Categories2', ['BEV', 'FCEV'])
    var_name = dm_veh_eff_LDV.col_labels['Variables'][0]
    dm_veh_eff_LDV.rename_col(var_name, 'tmp_name', dim='Variables')
    cdm_emissions_CO2_LDV = cdm_emissions_factors.filter({'Categories1': ['CO2'],
                                                          'Categories2': dm_veh_eff_LDV.col_labels['Categories2']})
    cdm_emissions_CO2_LDV.sort('Categories2')
    dm_veh_eff_LDV.sort('Categories2')
    # I want to have an efficiency in MJ/km -> then I have to do  veh-eff(g/km) / emission-fact(g/MJ) = eff (MJ/km)
    arr_eff_MJ_km = dm_veh_eff_LDV.array / cdm_emissions_CO2_LDV.array[np.newaxis, :, :, :]
    dm_veh_eff_LDV.add(arr_eff_MJ_km, dim='Variables', col_label=new_var_name, unit='MJ/km')
    dm_veh_eff_LDV.filter({'Variables': [new_var_name]}, inplace=True)
    return dm_veh_eff_LDV


def replace_LDV_efficiency_with_new(dm_veh_eff, dm_veh_new_eff, dm_veh_eff_LDV, dm_veh_new_eff_LDV, baseyear_old):
    # Vaud efficiency = Swiss efficiency
    dm_veh_eff_VD = dm_veh_eff.copy()
    dm_veh_eff_VD.rename_col('Switzerland', 'Vaud', dim='Country')
    dm_veh_eff.append(dm_veh_eff_VD, dim='Country')
    # Vaud efficiency = Swiss efficiency
    dm_veh_new_eff_VD = dm_veh_new_eff.copy()
    dm_veh_new_eff_VD.rename_col('Switzerland', 'Vaud', dim='Country')
    dm_veh_new_eff.append(dm_veh_new_eff_VD, dim='Country')

    # Remove fts years and add ots years missing
    years_match = [y for y in dm_veh_eff.col_labels['Years'] if y <= baseyear_old]
    years_missing = list(set(dm_veh_eff_LDV.col_labels['Years']) - set(years_match))
    dm_veh_eff.filter({'Years': years_match}, inplace=True)
    dm_veh_new_eff.filter({'Years': years_match}, inplace=True)
    dm_veh_eff.add(np.nan, dummy=True, dim='Years', col_label=years_missing)
    dm_veh_new_eff.add(np.nan, dummy=True, dim='Years', col_label=years_missing)

    # Rename 2W_PHEV as 2W_PHEV-diesel and 2W_PHEV-gasoline
    DM = {'veh-eff': dm_veh_eff, 'veh-eff-new': dm_veh_new_eff}
    for key, dm in DM.items():
        dm = dm.flatten()
        if '2W_PHEV' in dm.col_labels['Categories1']:
            dm_tmp = dm.filter({'Categories1': ['2W_PHEV']})
            dm.rename_col('2W_PHEV', '2W_PHEV-diesel', dim='Categories1')
            dm_tmp.rename_col('2W_PHEV', '2W_PHEV-gasoline', dim='Categories1')
            dm.append(dm_tmp, dim='Categories1')
            dm.deepen()
            DM[key] = dm

    dm_veh_eff = DM['veh-eff']
    dm_veh_new_eff = DM['veh-eff-new']

    idx_t = dm_veh_eff.idx
    idx_l = dm_veh_eff_LDV.idx
    for cat in dm_veh_eff_LDV.col_labels['Categories2']:
        dm_veh_eff.array[:, :, :, idx_t['LDV'], idx_t[cat]] = dm_veh_eff_LDV.array[:, :, :, idx_l['LDV'], idx_l[cat]]
        dm_veh_new_eff.array[:, :, :, idx_t['LDV'], idx_t[cat]] = dm_veh_new_eff_LDV.array[:, :, :, idx_l['LDV'], idx_l[cat]]

    dm_veh_eff.fill_nans(dim_to_interp='Years')
    dm_veh_new_eff.fill_nans(dim_to_interp='Years')
    # Remove aviation
    dm_veh_eff.drop(dim='Categories1', col_label='aviation')
    dm_veh_new_eff.drop(dim='Categories1', col_label='aviation')
    drop_cat = ['PHEV', 'ICE']
    for cat in drop_cat:
        if cat in dm_veh_eff.col_labels['Categories2']:
            dm_veh_eff.drop(col_label=[cat], dim='Categories2')
            dm_veh_new_eff.drop(col_label=[cat], dim='Categories2')

    return dm_veh_eff, dm_veh_new_eff


def fix_freight_tech_shares(DM_transport_new):
    dm_tech = DM_transport_new['fxa']['freight_tech'].filter({'Variables': ['tra_freight_technology-share_fleet']})
    # Fix the shares
    idx = dm_tech.idx
    dm_tech.array[:, :, :, idx['aviation'], idx['BEV']] = 0
    dm_tech.array[:, :, :, idx['aviation'], idx['ICE']] = 1
    dm_tech.array[:, :, :, idx['marine'], idx['FCEV']] = 0
    dm_tech.array[:, :, :, idx['marine'], idx['BEV']] = 0
    dm_tech.array[:, :, :, idx['marine'], idx['ICE']] = 1
    dm_tech.array[:, :, :, idx['IWW'], idx['FCEV']] = 0
    dm_tech.array[:, :, :, idx['IWW'], idx['BEV']] = 0
    dm_tech.array[:, :, :, idx['IWW'], idx['ICE']] = 1
    dm_tech.normalise(dim='Categories2')
    idx = DM_transport_new['fxa']['freight_tech'].idx
    DM_transport_new['fxa']['freight_tech'].array[:, :, idx['tra_freight_technology-share_fleet'], np.newaxis, ...] = dm_tech.array

    dm_tech = DM_transport_new['ots']['freight_technology-share_new']
    # Fix the shares
    idx = dm_tech.idx
    dm_tech.array[:, :, :, idx['aviation'], idx['BEV']] = 0
    dm_tech.array[:, :, :, idx['aviation'], idx['ICE']] = 1
    dm_tech.array[:, :, :, idx['marine'], idx['FCEV']] = 0
    dm_tech.array[:, :, :, idx['marine'], idx['BEV']] = 0
    dm_tech.array[:, :, :, idx['marine'], idx['ICE']] = 1
    dm_tech.array[:, :, :, idx['IWW'], idx['FCEV']] = 0
    dm_tech.array[:, :, :, idx['IWW'], idx['BEV']] = 0
    dm_tech.array[:, :, :, idx['IWW'], idx['ICE']] = 1
    dm_tech.normalise(dim='Categories2')


    dm_tech = DM_transport_new['fts']['freight_technology-share_new'][1]
    # Fix the shares
    idx = dm_tech.idx
    dm_tech.array[:, :, :, idx['aviation'], idx['BEV']] = 0
    dm_tech.array[:, :, :, idx['aviation'], idx['ICE']] = 1
    dm_tech.array[:, :, :, idx['marine'], idx['FCEV']] = 0
    dm_tech.array[:, :, :, idx['marine'], idx['BEV']] = 0
    dm_tech.array[:, :, :, idx['marine'], idx['ICE']] = 1
    dm_tech.array[:, :, :, idx['IWW'], idx['FCEV']] = 0
    dm_tech.array[:, :, :, idx['IWW'], idx['BEV']] = 0
    dm_tech.array[:, :, :, idx['IWW'], idx['ICE']] = 1
    dm_tech.normalise(dim='Categories2')
    return


def compute_renewal_rate_and_adjust(dm, var_names, max_rr):
    """
    It computes the renewal rate and it adjusts the new-vehicles before 2005, where the fleet split was not known
    """
    # Extract variable names
    s_col = var_names['stock']
    new_col = var_names['new']
    waste_col = var_names['waste']
    rr_col = var_names['renewal-rate']

    stock_unit = dm.units[s_col]

    # COMPUTE RENEWAL-RATE
    # Lag stock
    dm.lag_variable(pattern=s_col, shift=1, subfix='_tm1')
    # waste(t) = fleet(t-1) - fleet(t) + new-veh(t)
    dm.operation(s_col + '_tm1', '-', s_col, out_col='tra_delta_stock', unit=stock_unit)
    dm.operation('tra_delta_stock', '+', new_col, out_col=waste_col,
                 unit=stock_unit)
    # rr(t-1) = waste(t) / fleet(t-1)
    dm.operation(waste_col, '/', s_col + '_tm1', out_col='tmp', unit='%')
    dm.lag_variable(pattern='tmp', shift=-1, subfix='_rr')
    dm.rename_col('tmp_rr', rr_col, dim='Variables')
    dm.filter({'Variables': [s_col, s_col + '_tm1', rr_col]}, inplace=True)

    # FIX RENEWAL-RATE
    # move variables col to end
    dm_rr = dm.filter({'Variables': [rr_col]}, inplace=False)
    mask = (dm_rr.array < 0) | (dm_rr.array > max_rr)
    dm_rr.array[mask] = np.nan
    dm_rr.fill_nans('Years')
    dm.drop(dim='Variables', col_label=rr_col)
    dm.append(dm_rr, dim='Variables')

    # RECOMPUTE NEW FLEET
    dm.lag_variable(pattern=rr_col, shift=1, subfix='_tm1')
    # waste(t) = rr(t-1) * fleet(t-1)
    dm.operation(rr_col + '_tm1', '*', s_col + '_tm1', out_col=waste_col, unit=stock_unit)
    # new(t) = fleet(t) - fleet(t-1) + waste(t)
    dm.operation(s_col, '-', s_col + '_tm1', out_col='tra_delta_stock', unit=stock_unit)
    dm.operation('tra_delta_stock', '+', waste_col, out_col=new_col, unit=stock_unit)
    dm.filter({'Variables': [s_col, new_col, waste_col, rr_col]}, inplace=True)

    # FIX NEW FLEET
    dm_new = dm.filter({'Variables': [new_col]}, inplace=False)
    mask = (dm_new.array < 0)
    dm_new.array[mask] = np.nan
    dm_new.fill_nans('Years')
    dm.drop(dim='Variables', col_label=new_col)
    dm.append(dm_new, dim='Variables')

    # RECOMPUTE STOCK
    idx = dm.idx
    for t in dm.col_labels['Years'][1:]:
        s_tm1 = dm.array[:, idx[t-1], idx[s_col], ...]
        new_t = dm.array[:, idx[t], idx[new_col], ...]
        waste_t = dm.array[:, idx[t], idx[waste_col], ...]
        s_t = s_tm1 + new_t - waste_t
        dm.array[:, idx[t], idx[s_col], ...] = s_t

    return

def compute_new_public_fleet_ots(dm, var_names):
    # Extract variable names
    s_col = var_names['stock']
    new_col = var_names['new']
    waste_col = var_names['waste']
    rr_col = var_names['renewal-rate']

    stock_unit = dm.units[s_col]

    # COMPUTE RENEWAL-RATE
    # Lag stock
    dm.lag_variable(pattern=s_col, shift=1, subfix='_tm1')
    dm.lag_variable(pattern=rr_col, shift=1, subfix='_tm1')
    # waste(t) = rr(t-1) * fleet(t-1)
    dm.operation(rr_col + '_tm1', '*', s_col + '_tm1', out_col=waste_col, unit=stock_unit)
    # new(t) = fleet(t) - fleet(t-1) + waste(t)
    dm.operation(s_col, '-', s_col + '_tm1', out_col='tra_delta_stock', unit=stock_unit)
    dm.operation('tra_delta_stock', '+', waste_col, out_col=new_col, unit=stock_unit)
    dm.filter({'Variables': [s_col, new_col, waste_col, rr_col]}, inplace=True)

    # FIX NEW FLEET
    dm_new = dm.filter({'Variables': [new_col]}, inplace=False)
    mask = (dm_new.array < 0)
    dm_new.array[mask] = np.nan
    dm_new.fill_nans('Years')
    dm.drop(dim='Variables', col_label=new_col)
    dm.append(dm_new, dim='Variables')

    # RECOMPUTE STOCK
    idx = dm.idx
    for t in dm.col_labels['Years'][1:]:
        s_tm1 = dm.array[:, idx[t-1], idx[s_col], ...]
        new_t = dm.array[:, idx[t], idx[new_col], ...]
        waste_t = dm.array[:, idx[t], idx[waste_col], ...]
        s_t = s_tm1 + new_t - waste_t
        dm.array[:, idx[t], idx[s_col], ...] = s_t

    return

print('In order for this routine to run you need to download a couple of files and save them locally:'
      '- Aviation demand data (pkm) from  "Our World in Data": https://ourworldindata.org/grapher/aviation-demand-efficiency'
      '- Microrecensement analysis for 2005 and 2000 from EUCalc drive')


#################################################
#######    PRE-PROCESS TRANSPORT DATA     #######
#################################################

years_ots = create_years_list(start_year=1990, end_year=2023, step=1, astype=int)
years_fts = create_years_list(start_year=2025, end_year=2050, step=5)

new_tech_linear_fts = False
new_tech_flat_fts = True

##### Population
file = '../../../data/datamatrix/lifestyles.pickle'
with open(file, 'rb') as handle:
    DM_lifestyles = pickle.load(handle)

dm_pop = DM_lifestyles['ots']['pop']['lfs_population_'].copy()
dm_pop.sort('Country')
dm_pop.filter({'Country': ['Switzerland', 'Vaud']}, inplace=True)
dm_pop_fts = DM_lifestyles['fts']['pop']['lfs_population_'][1].copy()
dm_pop_fts.filter({'Country': ['Switzerland', 'Vaud']}, inplace=True)

#################################
########    AVIATION    #########
#################################
# SECTION Aviation ots

##### Transport demand aviation - Switzerland only
# Civil Aviation
# ! Data available only every 5 years
file_url = 'https://dam-api.bfs.admin.ch/hub/api/dam/assets/32013522/master'
local_filename = 'data/tra_aviation_CH.xlsx'
dm_pkm_cap_aviation_CH_raw = get_pkm_cap_aviation(file_url, local_filename)

##### Transport pkm aviation (World)
# Download data from "Our World in Data"
# https://ourworldindata.org/grapher/aviation-demand-efficiency
df = pd.read_csv('data/tra_global_aviation-demand.csv')
df = df[['Entity', 'Year', 'Passenger demand']]
df.columns = ['Country', 'Years', 'tra_passenger_transport-demand_aviation[Bpkm]']
dm_pkm_aviation_WLD = DataMatrix.create_from_df(df, num_cat=1)

##### Extract world population from World Bank
wb_pop_url = 'https://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=excel'
local_filename = 'data/lfs_world_population_WB.xlsx'
dm_pop_WLD = get_world_pop(wb_pop_url, local_filename)

##### Complete historical aviation pkm CH using World trend
dm_pkm_cap_aviation = compute_pkm_cap_aviation(dm_pkm_cap_aviation_CH_raw, dm_pkm_aviation_WLD, dm_pop_WLD, years_ots)
del df, wb_pop_url, local_filename, file_url, dm_pop_WLD, dm_pkm_cap_aviation_CH_raw, dm_pkm_aviation_WLD


##### Vehicle fleet aviation - Switzerland only
# Civil Aviation
# ! Data available only every 5 years
file_url = 'https://dam-api.bfs.admin.ch/hub/api/dam/assets/32013522/master'
local_filename = 'data/tra_aviation_CH.xlsx'
dm_pkm_fleet_aviation = get_aviation_fleet(file_url, local_filename)


######################################################################
#################################
#####   TRANSPORT DEMAND   ######
#################################
# SECTION Transport demand ots for pkm, vkm
#### Passenger transport demand pkm - Switzerland only
# Data source: FSO, 2024. Transport de personnes: prestations de transport. FSO number: je-f-11.04.01.02
file_url = 'https://dam-api.bfs.admin.ch/hub/api/dam/assets/32253177/master'
local_filename = 'data/tra_pkm_CH.xlsx'
dm_pkm_CH = get_transport_demand_pkm(file_url, local_filename, years_ots)

#### Passenger transport demand vkm - Switzerland only
## Transport de personnes: prestations kilométriques et mouvements des véhicules
file_url = 'https://dam-api.bfs.admin.ch/hub/api/dam/assets/32253171/master'
local_filename = 'data/tra_vkm_CH.xlsx'
dm_vkm_CH = get_transport_demand_vkm(file_url, local_filename, years_ots)

#region Alternative methodology to compute pkm/vkm Vaud - expand to explore
### Factor to downscale from Switzerland to Canton
# Data source: Microrecensement mobilité et transport MRMT (USE NEW DATA)
# file_url_dict = {2021: 'https://dam-api.bfs.admin.ch/hub/api/dam/assets/24025445/master',
#                 2015: {'Vaud': 'https://dam-api.bfs.admin.ch/hub/api/dam/assets/2081714/master',
#                        'Switzerland': 'https://dam-api.bfs.admin.ch/hub/api/dam/assets/2004971/master'},
#                 2010: 'https://dam-api.bfs.admin.ch/hub/api/dam/assets/291635/master'}
# local_filename_dict = {2021: 'data/tra_pkm_CH_VD_2021.xlsx',
#                       2015: {'Vaud': 'data/tra_pkm_VD_2015.xls', 'Switzerland': 'data/tra_pkm_CH_2015.xls'},
#                       2010: 'data/tra_pkm_CH_reg_2010.xls',
#                       2005: 'data/tra_pkm_CH_reg_2005.xls',
#                       2000: 'data/tra_pkm_CH_reg_2000.xls'}
# Check to make sure pkm/cap match with Microrecensement mobilité et transport
# dm_2021 = get_travel_demand_canton_microrecencement_2021(file_url_dict[2021], local_filename_dict[2021], 2021)
# dm_2015 = get_travel_demand_canton_microrecencement_2015(file_url_dict[2015], local_filename_dict[2015], 2015)
# dm_pkm_cap_fact = get_canton_pkm_cap_fact(dm_pkm_cap, years_ots)
# del local_filename_dict, file_url_dict
# ### Transport demand pkm - downscale Vaud
# Compute Swiss pkm/cap
# # pkm/cap_CH = pkm_CH / pop_CH
# DF = {'km_CH': dm_pkm_CH,
#      'pkm_cap_ratio': dm_pkm_cap_fact.copy(),
#      'pop': dm_pop.copy()}
# dm_pkm = compute_canton_passenger_transport_demand_pkm(DF, 'tra_passenger_transport-demand')
# del dm_pkm_CH, DF
#
# ### Transport demand vkm - downscale Vaud
# DF = {'km_CH': dm_vkm_CH.copy(),
#      'pkm_cap_ratio': dm_pkm_cap_fact.filter({'Categories1': ['LDV', '2W', 'bus', 'metrotram', 'rail']}),
#      'pop': dm_pop.copy()}
# dm_vkm = compute_canton_passenger_transport_demand_pkm(DF, 'tra_passenger_transport-demand-vkm')
# del dm_vkm_CH, file_url, local_filename, DF, dm_pkm_cap_fact
#endregion

# 2021: https://www.bfs.admin.ch/bfs/de/home/statistiken/kataloge-datenbanken.assetdetail.24267706.html
# 2015: https://www.bfs.admin.ch/asset/de/2503926
# 2010: https://www.bfs.admin.ch/asset/de/291635
file_url_dict = {2021: 'https://dam-api.bfs.admin.ch/hub/api/dam/assets/24267706/master',
                 2015: 'https://dam-api.bfs.admin.ch/hub/api/dam/assets/2503926/master',
                 2010: 'https://dam-api.bfs.admin.ch/hub/api/dam/assets/291635/master',
                 2005: None,
                 2000: None}

local_filename_dict = {2021: 'data/tra_pkm_CH_reg_2021.xlsx',
                       2015: 'data/tra_pkm_CH_reg_2015.xls',
                       2010: 'data/tra_pkm_CH_reg_2010.xls',
                       2005: 'data/tra_pkm_CH_reg_2005.xls',
                       2000: 'data/tra_pkm_CH_reg_2000.xls'}

dm_pkm_cap_raw = None
for year in local_filename_dict.keys():
    dm = get_travel_demand_region_microrecencement(file_url_dict[year], local_filename_dict[year], year)
    if dm_pkm_cap_raw is None:
        dm_pkm_cap_raw = dm.copy()
    else:
        dm_pkm_cap_raw.append(dm, dim='Years')
dm_pkm_cap_raw.sort('Years')

# For Vaud, adjust pkm/cap for 2015 and 2021 with actual values (split unchanged)
VD_pkm_day = {2015: 38.2, 2021: 32.1}
idx = dm_pkm_cap_raw.idx
arr_tot_pkm_cap_raw = np.nansum(dm_pkm_cap_raw.array, axis=-1)
corr_fact = dict()
for yr in VD_pkm_day.keys():
    corr_fact[yr] = arr_tot_pkm_cap_raw[idx['Vaud'], idx[yr], idx['tra_pkm-cap']] / (VD_pkm_day[yr]*365)
avg_fact = sum(corr_fact.values())/len(corr_fact.values())

for yr in dm_pkm_cap_raw.col_labels['Years']:
    if yr not in VD_pkm_day.keys():
        corr_fact[yr] = avg_fact
    dm_pkm_cap_raw.array[idx['Vaud'], idx[yr], idx['tra_pkm-cap'], :] = \
        dm_pkm_cap_raw.array[idx['Vaud'], idx[yr], idx['tra_pkm-cap'], :] / corr_fact[yr]

dm_tot = dm_pkm_cap_raw.copy()
dm_tot.change_unit('tra_pkm-cap', 365, 'pkm/cap', 'pkm/cap/day', operator='/')
dm_tot.group_all('Categories1')

# For the missing years extrapolate the pkm/cap value base on the pkm curve of Switzerland
dm_pkm_cap = extrapolate_missing_pkm_cap_based_on_pkm_CH(dm_pkm_cap_raw, dm_pkm_CH, dm_pop)
dm_pkm = compute_pkm_from_pkm_cap(dm_pkm_cap, dm_pop)
del dm_pkm_cap_raw, dm, local_filename_dict, file_url_dict, year

# Re-compute vkm CH and extrapolate VD by enforcing vkm/pkm_VD = vkm/pkm_CH
dm_vkm = compute_vkm_CH_VD(dm_vkm_CH, dm_pkm_CH, dm_pkm)
del dm_pkm_CH, dm_vkm_CH

################################
#####   VEHICLE FLEET  #########
################################
# SECTION New vehicle fleet and technology share LDV, 2W ots
##### New passenger fleet by technology LDV, 2W
table_id_new_veh = 'px-x-1103020200_120'
# file is created if it doesn't exist
file_new_veh_ots1 = 'data/tra_new_fleet.pickle'
# download this from https://www.bfs.admin.ch/asset/en/30305446, download csv file FSO number gr-e-11.03.02.02.01a
file_new_veh_ots2 = 'data/tra_new-vehicles_CH_1990-2023.csv'
# dm_new_tech is the number of new vehicles for new technologies (used to allocate "Other" category in dm_pass_fleet
dm_pass_new_fleet, dm_new_tech = compute_passenger_new_fleet(table_id_new_veh, file_new_veh_ots1, file_new_veh_ots2)
del table_id_new_veh, file_new_veh_ots1, file_new_veh_ots2

# SECTION Vehicle fleet and technology share LDV, 2W ots
#### Passenger fleet by technology (stock) LDV, 2W
table_id_tot_veh = 'px-x-1103020100_101'
file_tot_veh = 'data/tra_tot_fleet.pickle'
dm_pass_fleet_raw = get_passenger_stock_fleet_by_tech_raw(table_id_tot_veh, file_tot_veh)
# Allocate "Other" category to new technologies
dm_pass_fleet = allocate_other_to_new_technologies(dm_pass_fleet_raw, dm_new_tech)
del table_id_tot_veh, file_tot_veh, dm_pass_fleet_raw, dm_new_tech

# SECTION Vehicle fleet bus, rail, metrotram ots
#### Passenger fleet by technology (stock) bus, rail, metrotram - Switzerland only
# Note that this data are better for ots than
file_url = 'https://dam-api.bfs.admin.ch/hub/api/dam/assets/32253175/master'
# Transports publics (trafic marchandises rail inclus) - séries chronologiques détaillées
local_filename = 'data/tra_public_transport.xlsx'
DM_public = get_public_transport_data(file_url, local_filename, years_ots)
dm_public_fleet = DM_public['public_fleet'].copy()
del file_url, local_filename, DM_public

#### Passenger fleet by technology (stock) bus, rail, metrotram - Downscale to Vaud
dm_public_fleet = downscale_public_fleet_VD(dm_public_fleet, dm_pkm)

# SECTION Renewal-rate % - New-vehicles - vehicles Waste (2W, LDV) ots
dm_fleet_private = dm_pass_fleet.filter({'Years': years_ots})
dm_fleet_private.append(dm_pass_new_fleet.filter({'Years': years_ots}), dim='Variables')
var_names = {'stock': 'tra_passenger_vehicle-fleet', 'new': 'tra_passenger_new-vehicles',
             'waste': 'tra_passenger_vehicle-waste', 'renewal-rate': 'tra_passenger_renewal-rate'}
compute_renewal_rate_and_adjust(dm_fleet_private, var_names, max_rr=0.1)
dm_pass_fleet = dm_fleet_private.filter({'Variables': [var_names['stock']]})
dm_new_private_fleet = dm_fleet_private.filter({'Variables': [var_names['new']]})
dm_renewal_rate = dm_fleet_private.filter({'Variables': [var_names['renewal-rate']]})
dm_waste_private = dm_fleet_private.filter({'Variables': [var_names['waste']]})

# SECTION Renewal-rate % - New vehicles - vehicles Waste (bus, rail, metrotram) ots
# Use renewal-rate (1/lifetime) to compute the new public fleet
missing_cat = set(dm_public_fleet.col_labels['Categories2']) - set(dm_renewal_rate.col_labels['Categories2'])
dm_renewal_rate.add(np.nan, dim='Categories2', col_label=missing_cat, dummy=True)
dm_renewal_rate.add(np.nan, dim='Categories1', col_label=dm_public_fleet.col_labels['Categories1'], dummy=True)
idx = dm_renewal_rate.idx
idx_cat2_public = [idx[cat] for cat in dm_public_fleet.col_labels['Categories2']]
dm_renewal_rate.array[:, :, idx['tra_passenger_renewal-rate'], idx['rail'], idx_cat2_public] = 1/30
dm_renewal_rate.array[:, :, idx['tra_passenger_renewal-rate'], idx['metrotram'], idx['mt']] = 1/20
dm_renewal_rate.array[:, :, idx['tra_passenger_renewal-rate'], idx['bus'], idx_cat2_public] = 1/10

dm_public_fleet.append(dm_renewal_rate.filter({'Categories1': dm_public_fleet.col_labels['Categories1'],
                                               'Categories2': dm_public_fleet.col_labels['Categories2']}), dim='Variables')
var_names = {'renewal-rate': 'tra_passenger_renewal-rate', 'stock': 'tra_passenger_vehicle-fleet',
             'new': 'tra_passenger_new-vehicles', 'waste': 'tra_passenger_vehicle-waste'}
compute_new_public_fleet_ots(dm_public_fleet, var_names)
dm_new_public_fleet = dm_public_fleet.filter({'Variables': [var_names['new']]})
dm_waste_public = dm_public_fleet.filter({'Variables': [var_names['waste']]})
dm_public_fleet.filter({'Variables': [var_names['stock']]}, inplace=True)

# Join private and public fleet new and waste
cat_private_only = list(set(dm_new_private_fleet.col_labels['Categories2'])
                        - set(dm_new_public_fleet.col_labels['Categories2']))
cat_public_only = list(set(dm_new_public_fleet.col_labels['Categories2'])
                       - set(dm_new_private_fleet.col_labels['Categories2']))
dm_new_fleet = dm_new_private_fleet.copy()
dm_new_fleet.add(np.nan, dummy=True, dim='Categories2', col_label=cat_public_only)
dm_new_public_fleet.add(np.nan, dummy=True, dim='Categories2', col_label=cat_private_only)
dm_new_fleet.append(dm_new_public_fleet, dim='Categories1')

dm_waste_fleet = dm_waste_private.copy()
dm_waste_fleet.add(np.nan, dummy=True, dim='Categories2', col_label=cat_public_only)
dm_waste_public.add(np.nan, dummy=True, dim='Categories2', col_label=cat_private_only)
dm_waste_fleet.append(dm_waste_public, dim='Categories1')


################################
####    EMISSION FACTORS   #####
################################
# SECTION Emission factors - constants
#region Literature on emissions factors
# Source cited in EUCalc doc
# CO2 emissions from table 3.2.1 "Road transport default CO2 emission factors and uncertainty ranges"
# CH4 and N2O emissions from table 3.2.2 "Road transport N2O and CH4 default emission factors and uncertainty ranges"
# [IPCC] International Panel on Climate Change (2006). IPCC Guidelines for National Greenhouse Gas Inventories
# - Volume 2: Energy - Mobile Combustion
# For marinefueloil we consider the emission of the residual fuel oil (aka heavy fuel oil) in table 3.5.2 (CO2) 3.5.3 (CH4, N2O)
# For kerosene (aviation) we use Table 3.6.4 (CO2) and 3.6.5 (CH4, N2O)
# https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/2_Volume2/V2_3_Ch3_Mobile_Combustion.pdf
# Similar values can be found in the National Inventory Document of Switzerland 2024 (Table 3-13).
#endregion
# Emission factors in kg/TJ
emis = {'CO2': {'diesel': 74100, 'gas': 56100, 'gasoline': 69300, 'hydrogen': 0, 'kerosene': 71500, 'marinefueloil': 77400},
        'CH4': {'diesel': 3.9, 'gas': 92, 'gasoline': 3.9, 'hydrogen': 0, 'kerosene': 0.5, 'marinefueloil': 7},
        'N2O': {'diesel': 3.9, 'gas': 3, 'gasoline': 3.9, 'hydrogen': 0, 'kerosene': 2, 'marinefueloil': 2}}
mapping_cat = {'ICE-diesel': 'diesel', 'PHEV-diesel': 'diesel', 'ICE-gasoline': 'gasoline',
               'PHEV-gasoline': 'gasoline', 'ICE-gas': 'gas', 'kerosene': 'kerosene', 'marinefueloil': 'marinefueloil'}

cdm_emissions_factors = create_emissions_factors_cdm(emis, mapping_cat)

###################################
#####  VEHICLE EFFICIENCY LDV  ####
###################################
# SECTION Vehicle efficiency LDV, stock and new, ots
#### Vehicle efficiency - LDV - CO2/km
# FCEV (Hydrogen) data are off - BEV too
# !!! Attention: The data are bad before 2016 and after 2020, backcasting to 1990 from 2016 done with linear fitting.
table_id_veh_eff = 'px-x-1103020100_106'
local_filename_veh = 'data/tra_veh_efficiency.pickle'  # The file is created if it doesn't exist
dm_veh_eff_LDV = get_vehicle_efficiency(table_id_veh_eff, local_filename_veh,
                                        var_name='tra_passenger_veh-efficiency_fleet', years_ots=years_ots)
del table_id_veh_eff, local_filename_veh

#### Vehicle efficiency new - LDV - CO2/km
# FCEV data are off, BEV = 25 gCO2/km independently of car power
table_id_new_eff = 'px-x-1103020200_201'
local_filename_new = 'data/tra_new-veh_efficiency.pickle'  # The file is created if it doesn't exist3#
dm_veh_new_eff_LDV = get_new_vehicle_efficiency(table_id_new_eff, local_filename_new,
                                                var_name='tra_passenger_veh-efficiency_new', years_ots=years_ots)
del table_id_new_eff, local_filename_new

# The Swiss efficiency for the fleet is given in gCO2/km. We convert it to MJ/km
dm_veh_eff_LDV = convert_eff_from_gCO2_km_to_MJ_km(dm_veh_eff_LDV, cdm_emissions_factors,
                                                   new_var_name='tra_passenger_veh-efficiency_fleet')
dm_veh_new_eff_LDV = convert_eff_from_gCO2_km_to_MJ_km(dm_veh_new_eff_LDV, cdm_emissions_factors,
                                                       new_var_name='tra_passenger_veh-efficiency_new')

### We should do a separate flow for aviation. from pkm/cap -> pkm -> technology share (applied to pkm) -> emissions/pkm
data_file = '/Users/paruta/2050-Calculators/PathwayCalc/_database/data/datamatrix/transport.pickle'
with open(data_file, 'rb') as handle:
    DM_transport = pickle.load(handle)

lev = list(DM_transport['ots'].keys())[0]
baseyear_old = DM_transport['ots'][lev].col_labels['Years'][-1]

# Add to the LDV efficiency the efficiency of other means of transport, taken from EUCalc data
print('You are missing vehicle efficiency for LDV FCEV and BEV, but also 2W, bus, aviation, metrotram, rail.'
      ' Bus efficiency looks very wrong')
dm_veh_eff = DM_transport['fxa']['passenger_tech'].filter({'Variables': ['tra_passenger_veh-efficiency_fleet'],
                                                           'Country': ['Switzerland']})
dm_veh_new_eff = DM_transport['ots']['passenger_veh-efficiency_new'].filter(
    {'Variables': ['tra_passenger_veh-efficiency_new'], 'Country': ['Switzerland']})

# Adjust efficiency for metrotram and rail
idx = dm_veh_eff.idx
idx_n = dm_veh_new_eff.idx
public_eff = {('metrotram', 'mt'): 3, ('rail', 'CEV'): 1, ('rail', 'ICE-diesel'): 1*0.8/0.3, ('rail', 'FCEV'): 1*0.8/0.3}
for key, value in public_eff.items():
    dm_veh_eff.array[:, :, idx['tra_passenger_veh-efficiency_fleet'], idx[key[0]], idx[key[1]]] = value
    dm_veh_new_eff.array[:, :, idx_n['tra_passenger_veh-efficiency_new'], idx_n[key[0]], idx_n[key[1]]] = value

# It also removes aviation
dm_veh_eff, dm_veh_new_eff = replace_LDV_efficiency_with_new(dm_veh_eff, dm_veh_new_eff, dm_veh_eff_LDV, dm_veh_new_eff_LDV, baseyear_old)

del dm_veh_new_eff_LDV, dm_veh_eff_LDV, mapping_cat, emis

#######################################################################


# SECTION Modal-share and Transport demand pkm fts
based_on_years = create_years_list(1990, 2019, 1)
linear_fitting(dm_pkm_cap, years_fts, based_on=based_on_years, min_tb=0)
# For Switzerland metrotram use flat extrapolation
idx = dm_pkm_cap.idx
dm_pkm_cap.array[idx['Switzerland'], idx[2025]:idx[2050]+1, idx['tra_pkm-cap'], idx['metrotram'] ] = \
    dm_pkm_cap.array[idx['Switzerland'], idx[2025], idx['tra_pkm-cap'], idx['metrotram'] ]
dm_modal_share = dm_pkm_cap.normalise(dim='Categories1', inplace=False)
dm_modal_share.rename_col('tra_pkm-cap_share', 'tra_passenger_modal-share', dim='Variables')


# SECTION Technology share fleet (2W, LDV) fts
# For tech share we don't need the forecasting because it is computed from new_fleet
arr_fleet_cap = dm_pass_fleet.array / dm_pop.array[..., np.newaxis, np.newaxis]
dm_pass_fleet_cap = DataMatrix.based_on(arr_fleet_cap, dm_pass_fleet,
                                        change={'Variables': ['tra_passenger_vehicle-fleet_cap']},
                                        units={'tra_passenger_vehicle-fleet_cap': 'number/cap'})
based_on_years = create_years_list(2015, 2020, 1)
linear_fitting(dm_pass_fleet_cap, years_fts, based_on=based_on_years, min_tb=0)
dm_fleet_tech_share = dm_pass_fleet_cap.normalise(dim='Categories2', inplace=False)
dm_fleet_tech_share.rename_col('tra_passenger_vehicle-fleet_cap_share', 'tra_passenger_technology-share_fleet', dim='Variables')

# SECTION Technology share fleet (bus, metrotram, rail) fts
arr_fleet_cap = dm_public_fleet.array / dm_pop.array[..., np.newaxis, np.newaxis]
dm_public_fleet_cap = DataMatrix.based_on(arr_fleet_cap, dm_public_fleet,
                                          change={'Variables': ['tra_passenger_vehicle-fleet_cap']},
                                          units={'tra_passenger_vehicle-fleet_cap': 'number/cap'})
based_on_years = create_years_list(2000, 2020, 1)
linear_fitting(dm_public_fleet_cap, years_fts, based_on=based_on_years, min_tb=0)
dm_public_tech_share = dm_public_fleet_cap.normalise('Categories2', inplace=False)

dm_public_tech_share.rename_col('tra_passenger_vehicle-fleet_cap_share', 'tra_passenger_technology-share_fleet', dim='Variables')
# Join private and public technology
dm_public_tech_share.add(np.nan, dummy=True, dim='Categories2', col_label=cat_private_only)
dm_fleet_tech_share.add(np.nan, dummy=True, dim='Categories2', col_label=cat_public_only)
dm_fleet_tech_share.append(dm_public_tech_share, dim='Categories1')
del dm_public_tech_share

# SECTION Technology share new fleet fts
if new_tech_linear_fts:
    # Compute per capita values
    arr_fleet_cap = dm_new_fleet.array / dm_pop.array[..., np.newaxis, np.newaxis]
    dm_pass_fleet_new_cap = DataMatrix.based_on(arr_fleet_cap, dm_new_fleet,
                                                change={'Variables': ['tra_passenger_new-vehicles_cap']},
                                                units={'tra_passenger_new-vehicles_cap': 'number/cap'})
    # Data source (https://www.bfs.admin.ch/bfs/en/home/news/whats-new.assetdetail.32306059.html)
    #BEV_2024_cap = 36506/10*12/8921981
    #PHEV_gasoline_2024_cap = 17109/10*12/8921981

    dm_pass_fleet_new_cap_std = dm_pass_fleet_new_cap.filter_w_regex({'Categories2': 'ICE|CEV|mt'})
    dm_pass_fleet_new_cap_alt = dm_pass_fleet_new_cap.filter_w_regex({'Categories2': 'BEV|FCEV|PHEV'})
    based_on_years_std = create_years_list(2010, 2022, 1)
    linear_fitting(dm_pass_fleet_new_cap_std, years_fts, based_on=based_on_years_std)
    based_on_years_alt = create_years_list(2015, 2024, 1)
    idx = dm_pass_fleet_new_cap_alt.idx
    var = 'tra_passenger_new-vehicles_cap'
    dm_pass_fleet_new_cap_alt.add(np.nan, dim='Years', col_label=[2024], dummy=True)
    #dm_pass_fleet_new_cap_alt.array[idx['Switzerland'], idx[2024], idx[var], idx['LDV'], idx['BEV']] = BEV_2024_cap
    #dm_pass_fleet_new_cap_alt.array[idx['Switzerland'], idx[2024], idx[var], idx['LDV'], idx['BEV']] = PHEV_gasoline_2024_cap
    linear_fitting(dm_pass_fleet_new_cap_alt, years_fts, based_on=based_on_years_alt)
    dm_pass_fleet_new_cap_alt.filter({'Years': years_ots + years_fts}, inplace=True)
    dm_pass_fleet_new_cap_std.append(dm_pass_fleet_new_cap_alt, dim='Categories2')
    dm_pass_fleet_new_cap = dm_pass_fleet_new_cap_std
    dm_pass_fleet_new_cap.array = np.maximum(dm_pass_fleet_new_cap.array, 0)

    # Normalise to obtain new-tech-share
    dm_fleet_new_tech_share = dm_pass_fleet_new_cap.normalise(dim='Categories2', inplace=False)
    dm_fleet_new_tech_share.rename_col('tra_passenger_new-vehicles_cap_share', 'tra_passenger_technology-share_new',
                                       dim='Variables')
    dm_fleet_new_tech_share.fill_nans('Years')

if new_tech_flat_fts:
    dm_fleet_new_tech_share = dm_new_fleet.normalise(dim='Categories2', inplace=False)
    dm_fleet_new_tech_share.rename_col('tra_passenger_new-vehicles_share', 'tra_passenger_technology-share_new',
                                       dim='Variables')
    dm_fleet_new_tech_share.add(np.nan, dim='Years', col_label=years_fts, dummy=True)
    dm_fleet_new_tech_share.fill_nans('Years')

# For rail, use 2021 values
idx = dm_fleet_new_tech_share.idx
idx_fts = [idx[y] for y in years_fts]
dm_fleet_new_tech_share.array[:, idx_fts, idx['tra_passenger_technology-share_new'], idx['rail'], :] = \
    dm_fleet_new_tech_share.array[:, idx[2021], np.newaxis, idx['tra_passenger_technology-share_new'], idx['rail'], :]


##### Compute pkm and fleet based on per capita fts values
dm_pop_all = dm_pop.copy()
dm_pop_all.append(dm_pop_fts, dim='Years')


#region Reflection on the renewal-rate
# When the stock is constant, the renewal-rate is 1/lifetime. This gives a logical sense to the renewal-rate.
# For the ots, the renewal-rate is simply computed as waste/stock, and its value is often far from what we believe
# is a realistic 1/lifetime value, for an average lifetime of 1/13 years the renewal-rate should be 7.6%, and instead we
# find values oscillating between 2% to 9%. This can be due to many things, if for example we are talking about a new
# technology like BEV, FCEV etc the waste is still relatively small because the car that were put on the market have
# not yet arrived to their end-of-life. For thecnologies that are going out of production the renewal rate can be high
# (a lot of waste from previous years and the stock going to zero). To complicate matters though there is the fact that
# the boundary are not closed, cars can be moved across borders before reaching their end of life. But we could assume
# that this is random noise on top of the data that should cancel out. But still in a scenario where we are trying to
# reproduce drastical changes to the vehicle stock, it does not make sense to use a constant renewal-rate for forecasting.
# For forecasting we could assume that the waste is equal to the new vehicles that were put on the market 13 years ago.
# This could create discontinuities between ots and fts... But we can try. So let's assume that for fts the renewal-rate
# is 1/lifetime and instead of computing the waste / stock vehicle fleet in the classic way we use the renewal-rate to go
# in time and determine the waste.
# In order to chose the average lifetime, we look at the trends of LDV ICE-diesel, where the trends show an average
# lifetime of 13.5 years. For EV the current lifetime seems to be rather 5.5 years.
# We could start with 5.5 years and then increase the lifetime to reach 13.5 years after 10 years. For 2W we use 8 years.
#endregion and lif and lifetime

# SECTION Lifetime ots fts
# ots are not used
arr = dm_fleet_new_tech_share.array*np.nan
dm_lifetime = DataMatrix.based_on(arr, format=dm_fleet_new_tech_share, change={'Variables': ['tra_passenger_lifetime']},
                                  units={'tra_passenger_lifetime': 'years'})
idx = dm_lifetime.idx
# LDV: ICE and PHEV vehicles have lifetime of 13.5 years
idx_fts = [idx[yr] for yr in years_fts]
for cat in dm_lifetime.col_labels['Categories2']:
    if ("ICE" in cat) or ('PHEV' in cat):
        dm_lifetime.array[:, idx_fts, idx['tra_passenger_lifetime'], idx['LDV'], idx[cat]] = 14
dm_lifetime.array[:, idx_fts, idx['tra_passenger_lifetime'], idx['LDV'], idx['ICE-gasoline']] = 15
# LDV: New technology like BEV and PHEV have lifetimes initially of 5.5, and the 13.5
for cat in dm_lifetime.col_labels['Categories2']:
    if ("BEV" in cat) or ('FCEV' in cat):
        dm_lifetime.array[:, idx[years_fts[0]], idx['tra_passenger_lifetime'], idx['LDV'], idx[cat]] = 5.5
        dm_lifetime.array[:, idx[2035], idx['tra_passenger_lifetime'], idx['LDV'], idx[cat]] = 14
        dm_lifetime.array[:, idx[years_fts[-1]], idx['tra_passenger_lifetime'], idx['LDV'], idx[cat]] = 14
# 2W: 8 years
dm_lifetime.array[:, idx_fts, idx['tra_passenger_lifetime'], idx['2W'], :] = 8
# We assume rail lifetime is 30 years, metrotram lifetime is 20 years and bus lifetime is 10 years
idx_cat2_public = [idx[cat] for cat in dm_public_fleet.col_labels['Categories2']]
dm_lifetime.array[:, :, idx['tra_passenger_lifetime'], idx['rail'], idx_cat2_public] = 30
dm_lifetime.array[:, :, idx['tra_passenger_lifetime'], idx['metrotram'], idx['mt']] = 20
dm_lifetime.array[:, :, idx['tra_passenger_lifetime'], idx['bus'], idx_cat2_public] = 10
dm_lifetime.fill_nans('Years')


# SECTION Transport demand vkm fts
arr_vkm_cap = dm_vkm.array / dm_pop.array[..., np.newaxis]
dm_vkm_cap = DataMatrix.based_on(arr_vkm_cap, dm_vkm,
                                 change={'Variables': ['tra_vkm-cap']},
                                 units={'tra_vkm-cap': 'vkm/cap'})
based_on_years = create_years_list(1990, 2019, 1)
linear_fitting(dm_vkm_cap, years_fts, based_on=based_on_years, min_tb=0)
# For metrotram extrapolate with flat line
idx = dm_vkm_cap.idx
dm_vkm_cap.array[idx['Switzerland'], idx[2025]:idx[2050]+1, idx['tra_vkm-cap'], idx['metrotram']] = \
    dm_vkm_cap.array[idx['Switzerland'], idx[2025], idx['tra_vkm-cap'], idx['metrotram']]
# Make sure 2W vkm <= pkm (occupancy > 1)
idx_p = dm_pkm_cap.idx
mask = dm_vkm_cap.array[:, :, idx['tra_vkm-cap'], idx['2W']] > dm_pkm_cap.array[:, :, idx_p['tra_pkm-cap'], idx_p['2W']]
dm_vkm_cap.array[:, :, idx['tra_vkm-cap'], idx['2W']][mask] = dm_pkm_cap.array[:, :, idx_p['tra_pkm-cap'], idx_p['2W']][mask]


# SECTION Public fleet avg-pkm-veh [pkm/veh] (bus, metrotram, rail) fts
# by construction avg-pkm-veh CH and VD are the same
# Re-define dm_public_fleet as public_fleet_cap * pop
dm_public = dm_public_fleet_cap.group_all('Categories2', inplace=False)
dm_public_pkm = dm_pkm_cap.filter({'Categories1': ['bus', 'metrotram', 'rail']}, inplace=False)
dm_public.append(dm_public_pkm, dim='Variables')
dm_public.operation('tra_pkm-cap', '/', 'tra_passenger_vehicle-fleet_cap',
                    out_col='tra_passenger_avg-pkm-by-veh', unit='pkm')
dm_public_avg_pkm = dm_public.filter({'Variables': ['tra_passenger_avg-pkm-by-veh']})
del dm_public_pkm, dm_public


# SECTION Occupancy pkm/vkm (LDV/2W) fts
dm_km = dm_pkm_cap.filter({'Categories1': dm_vkm_cap.col_labels['Categories1']})
dm_km.append(dm_vkm_cap, dim='Variables')
dm_km.operation('tra_pkm-cap', '/', 'tra_vkm-cap',
                        out_col='tra_passenger_occupancy', unit='pkm/vkm')
dm_occupancy = dm_km.filter({'Variables': ['tra_passenger_occupancy']})


# SECTION Utilisation rate vkm/veh fts
dm_fleet_cap = dm_pass_fleet_cap.group_all('Categories2', inplace=False)
dm_fleet_cap.append(dm_public_fleet_cap.group_all('Categories2', inplace=False), dim='Categories1')
dm_km.append(dm_fleet_cap, dim='Variables')
dm_km.operation('tra_vkm-cap', '/', 'tra_passenger_vehicle-fleet_cap',
                        out_col='tra_passenger_utilisation-rate', unit='vkm/veh')
dm_utilisation = dm_km.filter({'Variables': ['tra_passenger_utilisation-rate']})
del dm_km


# SECTION Efficiency fleet
# For veh-fleet efficiency we can leave the fts to nan because this get re-computed
dm_veh_eff.add(np.nan, dim='Years', dummy=True, col_label=years_fts)
dm_fleet_tech_share.append(dm_veh_eff, dim='Variables')
dm_veh_new_eff_fts = dm_veh_new_eff.copy()
dm_veh_new_eff_fts.add(np.nan, dim='Years', dummy=True, col_label=years_fts)
dm_veh_new_eff_fts.fill_nans(dim_to_interp='Years')
dm_veh_new_eff_fts.filter({'Years': years_fts}, inplace=True)

# SECTION New Fleet dummy fts
dm_new_fleet.add(np.nan, dim='Years', col_label=years_fts, dummy=True)
dm_fleet_tech_share.append(dm_new_fleet, dim='Variables')

# SECTION Waste Fleet dummy fts
dm_waste_fleet.add(np.nan, dim='Years', col_label=years_fts, dummy=True)
dm_fleet_tech_share.append(dm_waste_fleet, dim='Variables')

# SECTION Aviation
dm_pkm_cap_aviation_fts = dm_pkm_cap_aviation.copy()
dm_pkm_cap_aviation_fts.add(np.nan, dim='Years', col_label=years_fts, dummy=True)
linear_fitting(dm_pkm_cap_aviation_fts, years_fts)
dm_pkm_cap_aviation_fts.filter({'Years': years_fts}, inplace=True)

# Compute vehicle lifetime in vkm for LDV and 2W instead of renewal rate
#dm_tmp = dm_utilisation.filter({'Categories1': ['2W', 'LDV']})
#dm_tmp.append(dm_renewal_rate.filter({'Categories1': ['2W', 'LDV']}), dim='Variables')
#dm_tmp.operation('tra_passenger_utilisation-rate', '/', 'tra_passenger_renewal-rate',
#                             out_col='tra_passenger_vehicle-lifetime', unit='vkm')
#dm_veh_lifetime = dm_tmp.filter({'Variables': ['tra_passenger_vehicle-lifetime']}, inplace=False)
#dm_renewal_rate.drop(col_label=['2W', 'LDV'], dim='Categories1')
## Compute the avg veh-lifetime from 2009-2019
#linear_fitting(dm_veh_lifetime, years_fts, based_on=create_years_list(2004, 2019, 1))

# SECTION Electricity emission factor
col_dict = {
    'Country': ['Vaud', 'Switzerland'],
    'Years': years_ots+years_fts,
    'Variables': ['tra_emission-factor'],
    'Categories1': ['CH4', 'CO2', 'N2O'],
    'Categories2': ['electricity']
}
dm_elec = DataMatrix(col_labels=col_dict, units={'tra_emission-factor': 'g/MJ'})

arr_elec = np.zeros((2, 40, 1, 3, 1))
idx = dm_elec.idx
arr_elec[:, idx[1990]: idx[2023]+1, 0, idx['CO2'], 0] = 31.1
arr_elec[:, idx[2025]: idx[2050], 0, idx['CO2'], 0] = np.nan
arr_elec[:, idx[2050], 0, idx['CO2'], 0] = 0
dm_elec.array = arr_elec
dm_elec.fill_nans(dim_to_interp="Years")


# FXA
DM_transport_new = {'fxa': dict(), 'ots': dict(), 'fts': dict(), 'constant': dict()}
#DM_transport_new['fxa']['passenger_renewal-rate'] = dm_renewal_rate
#DM_transport_new['fxa']['passenger_avg-pkm-by-veh'] = dm_public_avg_pkm
DM_transport_new['fxa']['passenger_tech'] = dm_fleet_tech_share
DM_transport_new['fxa']['passenger_vehicle-lifetime'] = dm_lifetime
DM_transport_new['fxa']['emission-factor-electricity'] = dm_elec

# OTS
# Filter aviation pkm/cap and rename
DM_transport_new['ots']['passenger_aviation-pkm'] = dm_pkm_cap_aviation
DM_transport_new['ots']['passenger_modal-share'] = dm_modal_share.filter({'Years': years_ots})
DM_transport_new['ots']['passenger_occupancy'] = dm_occupancy.filter({'Years': years_ots})
DM_transport_new['ots']['passenger_technology-share_new'] = dm_fleet_new_tech_share.filter({'Years': years_ots})
DM_transport_new['ots']['passenger_utilization-rate'] = dm_utilisation.filter({'Years': years_ots})
DM_transport_new['ots']['passenger_veh-efficiency_new'] = dm_veh_new_eff


# FTS
DM_transport_new['fts']['passenger_aviation-pkm'] = dict()
DM_transport_new['fts']['passenger_modal-share'] = dict()
DM_transport_new['fts']['passenger_occupancy'] = dict()
DM_transport_new['fts']['passenger_technology-share_new'] = dict()
DM_transport_new['fts']['passenger_utilization-rate'] = dict()
DM_transport_new['fts']['passenger_veh-efficiency_new'] = dict()

DM_transport_new['fts']['passenger_aviation-pkm'][1] = dm_pkm_cap_aviation_fts
DM_transport_new['fts']['passenger_modal-share'][1] = dm_modal_share.filter({'Years': years_fts})
DM_transport_new['fts']['passenger_occupancy'][1] = dm_occupancy.filter({'Years': years_fts})
DM_transport_new['fts']['passenger_technology-share_new'][1] = dm_fleet_new_tech_share.filter({'Years': years_fts})
DM_transport_new['fts']['passenger_utilization-rate'][1] = dm_utilisation.filter({'Years': years_fts})
DM_transport_new['fts']['passenger_veh-efficiency_new'][1] = dm_veh_new_eff_fts

# CONSTANT
DM_transport_new['constant'] = cdm_emissions_factors


# LIFESTYLES - TRANSPORT  INTERFACE
dm_pop.append(dm_pop_fts, dim='Years')

DM_interface_lfs_to_tra = {'pop': dm_pop}

file = '../../../data/interface/lifestyles_to_transport.pickle'
my_pickle_dump(DM_new=DM_interface_lfs_to_tra, local_pickle_file=file)

# tot pkm-cap
dm_pkm_cap_tot = dm_pkm_cap.group_all(dim='Categories1', inplace=False)

DM_transport_new['ots']['pkm'] = dm_pkm_cap_tot.filter({'Years': years_ots})
DM_transport_new['fts']['pkm'] = dict()
DM_transport_new['fts']['pkm'][1] = dm_pkm_cap_tot.filter({'Years': years_fts})

types = ['ots', 'fts', 'fxa']

lev = list(DM_transport['ots'].keys())[0]
years_ots_old = DM_transport['ots'][lev].col_labels['Years']
years_fts_old = DM_transport['fts'][lev][1].col_labels['Years']
years_missing = list(set(set(years_ots) - set(years_ots_old)))
years_to_drop = list(set(years_ots).intersection(set(years_fts_old)))
for t in types:
    labels = DM_transport[t].keys()
    for key in labels:
        if ('freight' in key or 'fuel' in key) and key not in DM_transport_new[t].keys():
            dm = DM_transport[t][key].filter({'Country': ['Switzerland', 'Vaud']})
            if t == 'fxa':
                dm.drop(col_label=years_to_drop, dim='Years')
                dm.add(np.nan, dim='Years', col_label=years_missing, dummy=True)
                linear_fitting(dm, years_ots+years_fts)
                DM_transport_new['fxa'][key] = dm.filter({'Years': years_ots+years_fts})
            if t == 'ots':
                # Fill in the missing years based on the BAU scenario (level=1)
                dm_fts = DM_transport['fts'][key][1].filter({'Country': ['Switzerland', 'Vaud']})
                dm_fts.drop(col_label=years_to_drop, dim='Years')
                dm_all = dm.copy()
                dm_all.append(dm_fts, dim='Years')
                dm_all.add(np.nan, dim='Years', col_label=years_missing, dummy=True)
                linear_fitting(dm_all, years_ots+years_fts)
                dm_ots = dm_all.filter({'Years': years_ots})
                DM_transport_new['ots'][key] = dm_ots
                DM_transport_new['fts'][key] = {1: dm_all.filter({'Years': years_fts})}


# Fix freight tech-share
fix_freight_tech_shares(DM_transport_new)
# Fix freight modal-share
dm_modal = DM_transport_new['ots']['freight_modal-share']
idx = dm_modal.idx
dm_modal.array[:, :, :, idx['marine']] = 0
dm_modal.normalise('Categories1')

dm_modal_fts = DM_transport_new['fts']['freight_modal-share'][1]
idx = dm_modal_fts.idx
dm_modal_fts.array[:, :, :, idx['marine']] = 0
dm_modal_fts.array[:, 0:-1, ...] = np.nan
dm_modal_all = dm_modal.copy()
dm_modal_all.append(dm_modal_fts, dim='Years')
linear_fitting(dm_modal_all, years_ots+years_fts)
dm_modal_all.normalise('Categories1')
DM_transport_new['fts']['freight_modal-share'][1] = dm_modal_all.filter({'Years': years_fts})

pickle_file = '../../../data/datamatrix/transport.pickle'

my_pickle_dump(DM_new=DM_transport_new, local_pickle_file=pickle_file)
sort_pickle(pickle_file)
