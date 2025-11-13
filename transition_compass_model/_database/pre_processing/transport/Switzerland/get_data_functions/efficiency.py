from _database.pre_processing.api_routines_CH import get_data_api_CH
from model.common.auxiliary_functions import moving_average
from model.common.data_matrix_class import DataMatrix

import numpy as np
import os
import pickle

def get_vehicle_efficiency(table_id, file, years_ots, var_name):
  # New fleet data are heavy, download them only once
  try:
    with open(file, 'rb') as handle:
      dm_veh_eff = pickle.load(handle)
      print \
        (f'The vehicle efficienty is read from file {file}. Delete it if you want to update data from api.')
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
                     'Émissions de CO2 par km (NEDC)': structure
                       ['Émissions de CO2 par km (NEDC)'],
                     'Filtre à particules': part}

        mapping_dim = {'Country': 'Canton',
                       'Years': 'Année',
                       'Variables': 'Puissance',
                       'Categories1': 'Carburant',
                       'Categories2': 'Émissions de CO2 par km (NEDC)'}

        # Extract new fleet
        dm_veh_eff_cl = get_data_api_CH(table_id, mode='extract', filter=filtering, mapping_dims=mapping_dim,
                                        units=['gCO2/km' ] *len
                                          (structure['Puissance']), language='fr')
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
    new_cat = float(old_cat.split('-')[0]) + float(old_cat.split('-')[1] ) /2
    co2_km.append(new_cat)
  co2_arr = np.array(co2_km)
  dm_veh_eff.normalise(dim='Categories2', inplace=True)
  dm_veh_eff.array = dm_veh_eff.array *co2_arr[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
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
      print(
        f'The vehicle efficienty is read from file {file}. Delete it if you want to update data from api.')
  except OSError:
    structure, title = get_data_api_CH(table_id, mode='example',
                                       language='fr')
    i = 0
    # The table is too big to be downloaded at once
    for eu_class in structure["Classe d'émission selon l'UE"]:
      i = i + 1
      filtering = {'Année': structure['Année'],
                   'Carburant': structure['Carburant'],
                   'Puissance': structure['Puissance'],
                   'Canton': ['Suisse', 'Vaud'],
                   "Classe d'émission selon l'UE": eu_class,
                   'Émissions de CO2 par km (NEDC/WLTP)': structure[
                     'Émissions de CO2 par km (NEDC/WLTP)']}

      mapping_dim = {'Country': 'Canton',
                     'Years': 'Année',
                     'Variables': 'Puissance',
                     'Categories1': 'Carburant',
                     'Categories2': 'Émissions de CO2 par km (NEDC/WLTP)'}

      # Extract new fleet
      dm_veh_eff_cl = get_data_api_CH(table_id, mode='extract',
                                      filter=filtering,
                                      mapping_dims=mapping_dim,
                                      units=['gCO2/km'] * len(
                                        structure['Puissance']),
                                      language='fr')
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
  years_to_add = [year for year in years_ots if
                  year not in dm_veh_eff.col_labels['Years']]
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
        if dm_norm.array[
          idx[country], idx[year], 0, idx[cat], idx['Inconnu']] > 0.2:
          dm_veh_eff.array[idx[country], idx[year], 0, idx[cat], :] = np.nan

  for i in range(2):
    window_size = 3  # Change window size to control the smoothing effect
    data_smooth = moving_average(dm_veh_eff.array, window_size,
                                 axis=dm_veh_eff.dim_labels.index('Years'))
    dm_veh_eff.array[:, 1:-1, ...] = data_smooth

  # Distribute Inconnu on other categories based on their share
  cat_other = [cat for cat in dm_veh_eff.col_labels['Categories2'] if
               cat != 'Inconnu']
  dm_other = dm_veh_eff.filter({'Categories2': cat_other}, inplace=False)
  dm_other.normalise(dim='Categories2', inplace=True)
  dm_other.array = np.nan_to_num(dm_other.array)
  idx = dm_veh_eff.idx
  arr_inc = np.nan_to_num(
    dm_veh_eff.array[:, :, :, :, idx['Inconnu'], np.newaxis]) * dm_other.array
  dm_veh_eff.drop(dim='Categories2', col_label='Inconnu')
  dm_veh_eff.array = dm_veh_eff.array + arr_inc

  # Remove fuel type "Autre" (there are only very few car in this category)
  dm_veh_eff.drop(dim='Categories1', col_label='Autre')

  # Group categories1 according to model
  map_cat = {'ICE-diesel': ['Diesel', 'Diesel-électrique: hybride normal'],
             'ICE-gasoline': ['Essence',
                              'Essence-électrique: hybride normal'],
             'ICE-gas': ['Gaz (monovalent et bivalent)'],
             'BEV': ['Électrique'],
             'FCEV': ['Hydrogène'],
             'PHEV-diesel': ['Diesel-électrique: hybride rechargeable'],
             'PHEV-gasoline': ['Essence-électrique: hybride rechargeable']
             }
  dm_veh_eff.groupby(map_cat, dim='Categories1', inplace=True)

  dm_veh_eff.groupby({var_name: '.*'}, dim='Variables', regex=True,
                     inplace=True)

  # Clean grams CO2 category and perform weighted average
  # cols are e.g '0 - 50 g' -> '0-50' -> 25
  dm_veh_eff.rename_col_regex(' g', '', dim='Categories2')
  dm_veh_eff.rename_col_regex(' ', '', dim='Categories2')
  dm_veh_eff.rename_col('Plusde300', '300-350', dim='Categories2')
  cat2_list_old = dm_veh_eff.col_labels['Categories2']
  co2_km = []
  for i in range(len(cat2_list_old)):
    old_cat = cat2_list_old[i]
    new_cat = float(old_cat.split('-')[0]) + float(old_cat.split('-')[1]) / 2
    co2_km.append(new_cat)
  dm_veh_eff.normalise(dim='Categories2', inplace=True)
  dm_veh_eff.array = dm_veh_eff.array * np.array(co2_km)
  dm_veh_eff.group_all(dim='Categories2')
  dm_veh_eff.change_unit(var_name, 1, old_unit='%', new_unit='gCO2/km')

  # Add LDV
  dm_veh_eff_LDV = DataMatrix.based_on(dm_veh_eff.array[..., np.newaxis],
                                       dm_veh_eff,
                                       change={'Categories2': ['LDV']},
                                       units=dm_veh_eff.units)
  dm_veh_eff_LDV.switch_categories_order()
  dm_veh_eff_LDV.rename_col('Suisse', 'Switzerland', dim='Country')

  return dm_veh_eff_LDV
