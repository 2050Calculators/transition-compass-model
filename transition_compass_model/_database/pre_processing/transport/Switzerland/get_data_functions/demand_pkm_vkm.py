# get_transport_demand_pkm, get_transport_demand_vkm, get_travel_demand_region_microrecencement
from model.common.auxiliary_functions import save_url_to_file, linear_fitting
import pandas as pd
from _database.pre_processing.transport.Switzerland.get_data_functions import utils
import numpy as np
from model.common.data_matrix_class import DataMatrix

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
    dm_latest = utils.df_fso_excel_to_dm(df_latest, header_row, names_map, var_name, unit, num_cat=1)
    # The names change from 1990-2004 to 2005-2023
    names_map.pop('Autobus')
    names_map['Transport par bus'] = 'bus'
    names_map.pop('Bicyclettes, y. c. vélos électriques lents')
    names_map['Bicyclettes'] = 'bike'
    names_map.pop('À pied')
    names_map['à pied'] = 'walk'
    dm_earlier = utils.df_fso_excel_to_dm(df_earlier, header_row, names_map, var_name, unit, num_cat=1)
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


def get_transport_demand_vkm(file_url, local_filename, years_ots):
  # If file does not exist, it downloads it and creates it
  rows_to_keep = ['en millions de trains-km', 'Tram', 'Trolleybus', 'Autobus',
                  'Voitures de tourisme',
                  'Cars privés', 'Motocycles']
  new_name = ['rail', 'metrotram', 'bus', 'bus', 'LDV', 'LDV', '2W']
  var_name = 'tra_passenger_transport-demand-vkm'
  unit = 'Mvkm'
  header_row = 0

  save_url_to_file(file_url, local_filename)

  df_latest = pd.read_excel(local_filename)
  df_earlier = pd.read_excel(local_filename, sheet_name='1990-2004')

  df_latest[df_latest.columns[0]] = df_latest[df_latest.columns[0]].str.replace(
    r'\d+\)|\(\d+\)', '', regex=True).str.strip()
  df_earlier[df_earlier.columns[0]] = df_earlier[
    df_earlier.columns[0]].str.replace(r'\d+\)|\(\d+\)', '',
                                       regex=True).str.strip()

  # Clean df from excel file
  names_map = dict()
  for i, row in enumerate(rows_to_keep):
    names_map[row] = new_name[i]
  dm_latest = utils.df_fso_excel_to_dm(df_latest, header_row, names_map, var_name,
                                 unit, num_cat=1)
  names_map.pop('Autobus')
  names_map['Transport par bus'] = 'bus'
  dm_earlier = utils.df_fso_excel_to_dm(df_earlier, header_row, names_map, var_name,
                                  unit, num_cat=1)
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


