import pickle
from _database.pre_processing.api_routines_CH import get_data_api_CH
import os
import pandas as pd
import numpy as np
import requests
from model.common.data_matrix_class import DataMatrix
from model.common.auxiliary_functions import linear_fitting, add_missing_ots_years

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
      dm_new_fleet_month = get_data_api_CH(table_id, mode='extract',
                                           filter=filtering,
                                           mapping_dims=mapping_dim,
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
  dm_new_fleet.groupby({'tra_passenger_new-vehicles': '.*'},
                       dim='Variables', regex=True, inplace=True)

  # Keep only passenger car main categories
  main_cat = [cat for cat in dm_new_fleet.col_labels['Categories1'] if
              '>' in cat]
  passenger_cat = [cat for cat in main_cat if
                   'Passenger' in cat or 'Motorcycles' in cat]

  # Filter for Passenger vehicles
  dm_pass_new_fleet = dm_new_fleet.filter({'Categories1': passenger_cat},
                                          inplace=False)
  dm_pass_new_fleet.groupby({'LDV': '.*Passenger.*'}, dim='Categories1',
                            regex=True, inplace=True)
  dm_pass_new_fleet.groupby({'2W': '.*Motorcycles.*'}, dim='Categories1',
                            regex=True, inplace=True)

  # Filter new technologies
  # (this is needed to later allocate the vehicle fleet "Other" category to the new technologies)
  new_technologies = ['Hydrogen', 'Diesel-electricity: conventional hybrid',
                      'Petrol-electricity: conventional hybrid',
                      'Petrol-electricity: plug-in hybrid',
                      'Diesel-electricity: plug-in hybrid',
                      'Gas (monovalent and bivalent)']
  dm_new_tech = dm_pass_new_fleet.filter({'Categories2': new_technologies})

  # Map fuel technology to transport module category
  dict_tech = {'FCEV': ['Hydrogen'], 'BEV': ['Electricity'],
               'ICE-diesel': ['Diesel',
                              'Diesel-electricity: conventional hybrid'],
               'ICE-gasoline': ['Petrol',
                                'Petrol-electricity: conventional hybrid'],
               'PHEV-diesel': ['Diesel-electricity: plug-in hybrid'],
               'PHEV-gasoline': ['Petrol-electricity: plug-in hybrid'],
               'ICE-gas': ['Gas (monovalent and bivalent)']}
  dm_pass_new_fleet.groupby(dict_tech, dim='Categories2', regex=False,
                            inplace=True)
  dm_pass_new_fleet.drop(col_label='Without motor', dim='Categories2')
  # Check that other categories are only a small contribution
  dm_tmp = dm_pass_new_fleet.normalise(dim='Categories2', inplace=False)
  dm_tmp.filter({'Categories2': ['Other']}, inplace=True)
  # If Other and Without motor are more than 0.1% you should account for it
  if (dm_tmp.array > 0.01).any():
    raise ValueError(
      '"Other" category is greater than 1% of the fleet, it cannot be discarded')

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
  dm_pass_new_fleet_CH = dm_new_fleet_CH.groupby(
    {'tra_passenger_new-vehicles_LDV': 'passenger.*',
     'tra_passenger_new-vehicles_2W': 'motorcycles'},
    dim='Variables', regex=True, inplace=False)
  dm_pass_new_fleet_CH.deepen()

  # Keep only years before 2005
  old_yrs_series = [yr for yr in dm_pass_new_fleet_CH.col_labels['Years'] if
                    yr < first_year]
  dm_pass_new_fleet_CH.filter({'Years': old_yrs_series}, inplace=True)

  return dm_pass_new_fleet_CH

def get_passenger_stock_fleet_by_tech_raw(table_id, file):
  # New fleet data are heavy, download them only once
  try:
    with open(file, 'rb') as handle:
      dm_fleet = pickle.load(handle)
  except OSError:
    structure, title = get_data_api_CH(table_id, mode='example')
    # Keep only passenger car main categories
    main_cat = [cat for cat in structure['Vehicle group / type'] if
                '>' in cat]
    passenger_cat = [cat for cat in main_cat if
                     'Passenger' in cat or 'Motorcycles' in cat]

    filtering = {'Year': structure['Year'],
                 'Year of first registration': structure[
                   'Year of first registration'],
                 'Vehicle group / type': passenger_cat,
                 'Canton': ['Switzerland', 'Vaud'],
                 'Fuel': structure['Fuel']}

    mapping_dim = {'Country': 'Canton',
                   'Years': 'Year',
                   'Variables': 'Year of first registration',
                   'Categories1': 'Vehicle group / type',
                   'Categories2': 'Fuel'}

    # Extract new fleet
    dm_fleet = get_data_api_CH(table_id, mode='extract', filter=filtering,
                               mapping_dims=mapping_dim,
                               units=['number'] * len(
                                 structure['Year of first registration']))

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, file)
    with open(f, 'wb') as handle:
      pickle.dump(dm_fleet, handle, protocol=pickle.HIGHEST_PROTOCOL)
  # Group all vehicles independently of immatriculation data
  dm_fleet.groupby({'tra_passenger_vehicle-fleet': '.*'}, dim='Variables',
                   regex=True, inplace=True)
  # Group passenger vehicles as LDV and motorcycles as 2W
  dm_fleet.groupby({'LDV': '.*Passenger.*', '2W': '.*Motorcycles'},
                   dim='Categories1', regex=True, inplace=True)
  # Map fuel technology to transport module category. Other category cannot be removed as it is above 1%
  dict_tech = {'BEV': ['Electricity'],
               'ICE-diesel': ['Diesel'],
               'ICE-gasoline': ['Petrol']}
  dm_fleet.groupby(dict_tech, dim='Categories2', regex=False, inplace=True)
  dm_fleet.drop(dim='Categories2', col_label='Without motor')

  return dm_fleet

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
    print(
      f'File {local_filename} already exists. If you want to download again delete the file')
  # The excel file contains multiple sheets
  # load index sheet
  df_index = pd.read_excel(local_filename)
  df_index.drop(columns=['Unnamed: 0', 'Unnamed: 3', 'Unnamed: 5'],
                inplace=True)
  df_index.dropna(how='any', inplace=True)
  # Change colummns header
  df_index.rename(columns={'Unnamed: 1': 'Sheet', 'Unnamed: 2': 'Theme'},
                  inplace=True)
  df_index = df_index[1:]  # take the data less the header row
  sheet_fleet = list(df_index.loc[df_index[
                                    'Theme'] == 'Moyens de transport: véhicules '].Sheet)[
    0]
  sheet_passenger = \
    list(df_index.loc[df_index['Theme'] == 'Voyageurs transportés'].Sheet)[0]
  sheet_pkm = \
    list(df_index.loc[df_index['Theme'] == 'Voyageurs-kilomètres'].Sheet)[0]
  sheet_vkm = list(df_index.loc[df_index[
                                  'Theme'] == 'Utilisation du système: prestations kilométriques, ponctualité et indices des prix'].Sheet)[
    0]

  df_fleet = pd.read_excel(local_filename,
                           sheet_name=sheet_fleet.replace('.', '_'))
  df_nb_passenger = pd.read_excel(local_filename,
                                  sheet_name=sheet_passenger.replace('.',
                                                                     '_'))
  df_pkm = pd.read_excel(local_filename,
                         sheet_name=sheet_pkm.replace('.', '_'))
  df_vkm = pd.read_excel(local_filename,
                         sheet_name=sheet_vkm.replace('.', '_'))

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
  df = df[df.apply(lambda row: row.map(pd.api.types.is_number), axis=1).any(
    axis=1)]
  df = df.loc[:, df.apply(lambda col: col.map(pd.api.types.is_number)).any(
    axis=0)].copy()
  # Extract only the data we are interested in:
  # Electrique and Diesel here below refers to the engine type of rails
  vehicles_vars = [
    'Voitures voyageurs (voitures de commande isolées, automotrices et éléments de rames automotrices inclus)',
    'Trolleybus', 'Tram', 'Autobus']
  df_pass_public_veh = df.loc[vehicles_vars].copy()
  df_pass_public_veh = df_pass_public_veh.apply(
    lambda col: pd.to_numeric(col, errors='coerce'))
  # df_pass_public_veh = df_pass_public_veh.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
  df_pass_public_veh.reset_index(inplace=True)
  df_pass_public_veh['Variables'] = df_pass_public_veh['Variables'].str[:10]
  df_pass_public_veh = df_pass_public_veh.groupby(['Variables']).sum()
  df_pass_public_veh.reset_index(inplace=True)

  # Pivot the dataframe
  df_pass_public_veh['Country'] = 'Switzerland'
  df_T = pd.melt(df_pass_public_veh, id_vars=['Variables', 'Country'],
                 var_name='Years', value_name='values')
  df_pivot = df_T.pivot_table(index=['Country', 'Years'],
                              columns=['Variables'], values='values',
                              aggfunc='sum')
  df_pivot = df_pivot.add_suffix('[number]')
  df_pivot = df_pivot.add_prefix('tra_passenger_vehicle-fleet_')
  df_pivot.reset_index(inplace=True)

  dm_fleet = DataMatrix.create_from_df(df_pivot, num_cat=1)
  map_cat = {'bus_CEV': ['Trolleybus'], 'bus_ICE-diesel': ['Autobus'],
             'metrotram_mt': ['Tram'], 'rail_CEV': ['Voitures v']}
  dm_fleet.groupby(map_cat, dim='Categories1', inplace=True)
  mask = dm_fleet.array == 0
  dm_fleet.array[mask] = np.nan
  add_missing_ots_years(dm_fleet, startyear=dm_fleet.col_labels['Years'][0],
                        baseyear=dm_fleet.col_labels['Years'][-1])
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
  df = df[df.apply(lambda row: row.map(pd.api.types.is_number), axis=1).any(
    axis=1)]
  df = df.loc[:, df.apply(lambda col: col.map(pd.api.types.is_number)).any(
    axis=0)].copy()
  # Vars to keep
  vars_to_keep = ['Chemins de fer', 'Chemins de fer à crémaillère', 'Tram',
                  'Trolleybus', 'Autobus']
  df_pkm = df.loc[vars_to_keep]
  # Convert ... to numerics
  df_pkm = df_pkm.apply(lambda col: pd.to_numeric(col, errors='coerce'))
  # df_pkm = df_pkm.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
  df_pkm.reset_index(inplace=True)

  # Pivot the dataframe
  df_pkm['Country'] = 'Switzerland'
  df_T = pd.melt(df_pkm, id_vars=['Variables', 'Country'], var_name='Years',
                 value_name='values')
  df_pivot = df_T.pivot_table(index=['Country', 'Years'],
                              columns=['Variables'], values='values',
                              aggfunc='sum')
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
  dm.change_unit('tra_passenger_transport-demand', 1e6, old_unit='Mpkm',
                 new_unit='pkm')

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
  df = df[df.apply(lambda row: row.map(pd.api.types.is_number), axis=1).any(
    axis=1)]
  df = df.loc[:, df.apply(lambda col: col.map(pd.api.types.is_number)).any(
    axis=0)].copy()
  # Vars to keep
  vars_to_keep = ['Chemins de fer', 'Chemins de fer à crémaillère', 'Tram',
                  'Trolleybus', 'Autobus']
  df_vkm = df.loc[vars_to_keep].copy()
  # Convert ... to numerics
  df_vkm = df_vkm.apply(lambda col: pd.to_numeric(col, errors='coerce'))
  # df_vkm = df_vkm.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
  df_vkm.reset_index(inplace=True)

  # Pivot the dataframe
  df_vkm['Country'] = 'Switzerland'
  df_T = pd.melt(df_vkm, id_vars=['Variables', 'Country'], var_name='Years',
                 value_name='values')
  df_pivot = df_T.pivot_table(index=['Country', 'Years'],
                              columns=['Variables'], values='values',
                              aggfunc='sum')
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
  years_init = [y for y in dm.col_labels['Years'] if
                y >= 2015 and y != 2020]
  dm_tmp = dm.filter({'Years': years_init})
  years_extract = [y for y in years_ots if y >= 2015]
  linear_fitting(dm_tmp, years_extract)
  dm_tmp.drop(dim='Years', col_label=2020)
  # Join historical values with extrapolated ones
  dm.drop(dim='Years', col_label=years_init)
  dm.append(dm_tmp, dim='Years')
  dm.sort('Years')
  linear_fitting(dm, years_ots)
  dm.change_unit('tra_passenger_transport-demand-vkm', 1000,
                 old_unit='1000vkm', new_unit='vkm')

  return dm
