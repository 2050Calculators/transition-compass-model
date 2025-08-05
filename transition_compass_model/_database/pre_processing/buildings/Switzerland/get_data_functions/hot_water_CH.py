import pickle
import os
import numpy as np
import pandas as pd
import zipfile

from _database.pre_processing.api_routines_CH import get_data_api_CH
from model.common.auxiliary_functions import translate_text, save_url_to_file, df_excel_to_dm, linear_fitting
from model.common.data_matrix_class import DataMatrix


def clean_country_names(dm):
  cantons_en = ['Aargau', 'Appenzell Ausserrhoden', 'Appenzell Innerrhoden',
                'Basel Landschaft', 'Basel Stadt', 'Bern', 'Fribourg', 'Geneva',
                'Glarus', 'Graubünden', 'Jura', 'Lucerne', 'Neuchâtel',
                'Nidwalden', 'Obwalden', 'Schaffhausen', 'Schwyz', 'Solothurn',
                'St. Gallen', 'Thurgau', 'Ticino', 'Uri', 'Valais', 'Vaud',
                'Zug', 'Zurich']
  cantons_fr =  ['Argovie', 'Appenzell Rh. Ext.', 'Appenzell Rh. Int.',
                 'Bâle Campagne', 'Bâle Ville', 'Berne', 'Fribourg', 'Genève',
                 'Glaris', 'Grisons', 'Jura', 'Lucerne', 'Neuchâtel', 'Nidwald',
                 'Obwald', 'Schaffhouse', 'Schwytz', 'Soleure', 'Saint Gall',
                 'Thurgovie', 'Tessin', 'Uri', 'Valais', 'Vaud', 'Zoug', 'Zurich']

  dm.rename_col_regex('Suisse', 'Switzerland', 'Country')
  dm.rename_col_regex(" /.*", "", dim='Country')
  dm.rename_col_regex("-", " ", dim='Country')
  dm.rename_col('Luzern','Lucerne', dim='Country')
  dm.rename_col('Genève','Geneva', dim='Country')
  dm.rename_col('Zürich','Zurich', dim='Country')

  dm.sort('Country')
  return dm

def extract_hotwater_technologies(table_id, file):
    # Domaine de l'énergie: bâtiments selon le canton, le type de bâtiment, l'époque de construction, le type de chauffage,
    # la production d'eau chaude, les agents énergétiques utilisés pour le chauffage et l'eau chaude
    try:
        with open(file, 'rb') as handle:
            dm_hw = pickle.load(handle)
    except OSError:
        structure, title = get_data_api_CH(table_id, mode='example', language='fr')
        # Extract buildings floor area
        filter = structure.copy()
        #filter['Catégorie de bâtiment'] = ["Bâtiments partiellement à usage d'habitation", "Bâtiments d'habitation avec usage annexe"]
        mapping_dim = {'Country': 'Canton', 'Years': 'Année',
                       'Variables': 'Catégorie de bâtiment',
                       'Categories1': "Source d'énergie de l'eau chaude"}
        unit_all = ['number'] * len(structure['Catégorie de bâtiment'])
        dm_hw = None
        tot_bld = 0
        for a in structure["Source d'énergie du chauffage"]:
            filter["Source d'énergie du chauffage"] = [a]
            dm_hw_t = get_data_api_CH(table_id, mode='extract', filter=filter,
                                      mapping_dims=mapping_dim, units=unit_all, language='fr')
            if dm_hw is None:
                dm_hw = dm_hw_t.copy()
            else:
                dm_hw.array += dm_hw_t.array

        dm_hw.rename_col(['Suisse'], ['Switzerland'], dim='Country')
        dm_hw.groupby({'bld_hot-water_tech_single-family-house': ['Maisons individuelles'],
                       'bld_hot-water_tech_multi-family-house': ['Maisons à plusieurs logements',
                                              "Bâtiments d'habitation avec usage annexe",
                                              "Bâtiments partiellement à usage d'habitation"]},
                      dim='Variables', inplace=True)
        dm_hw.deepen(based_on='Variables')
        dm_hw.switch_categories_order()

        with open(file, 'wb') as handle:
            pickle.dump(dm_hw, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dm_hw.groupby({'other': ['Autre', 'Aucune']}, dim='Categories2', inplace=True)
    dm_hw.rename_col(['Mazout', 'Bois', 'Pompe à chaleur', 'Electricité', 'Gaz', 'Chaleur produite à distance',
                           'Soleil (thermique)'],
                          ['heating-oil', 'wood', 'heat-pump', 'electricity', 'gas', 'district-heating', 'solar'],
                          dim='Categories2')
    clean_country_names(dm_hw)
    
    return dm_hw


def extract_hotwater_technologies_old(table_id, file):
  # Domaine de l'énergie: bâtiments selon le canton, le type de bâtiment, l'époque de construction, le type de chauffage,
  # la production d'eau chaude, les agents énergétiques utilisés pour le chauffage et l'eau chaude, 1990 et 2000
  try:
    with open(file, 'rb') as handle:
      dm_heating_old = pickle.load(handle)
  except OSError:
    structure, title = get_data_api_CH(table_id, mode='example', language='fr')
    # Extract buildings floor area
    filter = structure.copy()
    mapping_dim = {'Country': 'Canton', 'Years': 'Année',
                   'Variables': 'Type de bâtiment',
                   'Categories1': "Agent énergétique pour l'eau chaude"}
    dm_heating_old = None
    tot_bld = 0
    for t in structure['Type de chauffage']:
      for a in structure["Agent énergétique pour le chauffage"]:
        filter['Type de chauffage'] = [t]
        filter["Agent énergétique pour le chauffage"] = [a]
        unit_all = ['number'] * len(structure['Type de bâtiment'])
        dm_heating_old_t = get_data_api_CH(table_id, mode='extract',
                                           filter=filter,
                                           mapping_dims=mapping_dim,
                                           units=unit_all, language='fr')
        if dm_heating_old is None:
          dm_heating_old = dm_heating_old_t.copy()
        else:
          assert dm_heating_old.col_labels['Country'] == dm_heating_old_t.col_labels['Country']
          assert dm_heating_old.col_labels['Categories1'] == dm_heating_old_t.col_labels['Categories1']

          dm_heating_old.array = dm_heating_old_t.array + dm_heating_old.array


    #dm_heating_old.rename_col(['Suisse'], ['Switzerland'], dim='Country')
    dm_heating_old.groupby(
      {'bld_hot-water_tech_single-family-households': ['Maisons individuelles'],
       'bld_hot-water_tech_multi-family-households': ['Maisons à plusieurs logements',
                                   "Bâtiments d'habitation avec usage annexe",
                                   "Bâtiments partiellement à usage d'habitation"]},
      dim='Variables', inplace=True)
    dm_heating_old.deepen(based_on='Variables')
    dm_heating_old.switch_categories_order()

    with open(file, 'wb') as handle:
      pickle.dump(dm_heating_old, handle, protocol=pickle.HIGHEST_PROTOCOL)

  dm_heating_old.groupby(
    {'other': ['Autre agent énergétique (eau ch.)', "Sans production d'eau chaude"]},
    dim='Categories2', inplace=True)
  dm_heating_old.rename_col(
    ['Mazout (eau ch.)', 'Bois (eau ch.)', 'Pompe à chaleur (eau ch.)',
     'Electricité (eau ch.)', 'Gaz (eau ch.)', 'Chaleur à distance (eau ch.)',
     'Charbon (eau ch.)', 'Capteur solaire (eau ch.)'],
    ['heating-oil', 'wood', 'heat-pump', 'electricity', 'gas',
     'district-heating', 'coal', 'solar'],
    dim='Categories2')
  clean_country_names(dm_heating_old)

  dm_heating_old.rename_col('single-family-households', 'single-family-house', dim='Categories1')
  dm_heating_old.rename_col('multi-family-households', 'multi-family-house', dim='Categories1')

  return dm_heating_old


def extract_EP2050_hot_water_energy_consumption(file_raw, file_pickle):

  try:
    with open(file_pickle, 'rb') as handle:
      dm = pickle.load(handle)
  except OSError:
    df = pd.read_excel(file_raw, 'Tabelle14')

    df = df[list(df.columns)[1:-1]]
    df.columns = df.iloc[3]
    df = df.iloc[4:12]
    df.set_index(['Energieträger'], inplace=True)
    df.columns = df.columns.astype(int)

    df.reset_index(inplace=True)

    df.rename(columns = {'Energieträger': 'Energy_source'}, inplace=True)
    full_name = ['bld_hot-water_energy-consumption_' + var + '[PJ]' for var in df['Energy_source']]
    df['Energy_source'] = full_name
    # Pivot
    df_T = df.T
    df_T.columns = df_T.iloc[0]
    df_T = df_T.iloc[1:]
    df_T.reset_index(inplace=True)
    df_T.rename(columns = {3: 'Years'}, inplace=True)
    df_T['Country'] = 'Switzerland'

    dm = DataMatrix.create_from_df(df_T, num_cat=1)

    col_de = ["Heizöl", "Erdgas", "Elektrisch Ohm'sche Anlagen",
              "Elektrische Wärmepumpen", "Fernwärme", "Holz",
              "Umweltwärme", "Solar"]
    col_en = ['heating-oil', 'gas', 'electricity', 'heat-pump', 'district-heating',
              'wood', 'ambient-heat', 'solar']
    dm.rename_col(col_de, col_en, dim='Categories1')
    dm.change_unit('bld_hot-water_energy-consumption', old_unit='PJ', new_unit='TWh', factor=3.6, operator='/')

    with open(file_pickle, 'wb') as handle:
      pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

  dm.sort('Categories1')

  return dm


def extract_heating_efficiencies_EP2050(file_url, zip_name, file_pickle):

  try:
    with open(file_pickle, 'rb') as handle:
      dm = pickle.load(handle)

  except OSError:

    extract_dir = os.path.splitext(zip_name)[0]  # 'data/EP2050_sectors'
    if not os.path.exists(extract_dir):
      save_url_to_file(file_url, zip_name)

      # Extract the file
      os.makedirs(extract_dir, exist_ok=True)
      with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    file_industry = extract_dir + '/EP2050+_Szenarienergebnisse_Details_Nachfragesektoren/EP2050+_Detailergebnisse 2020-2060_Private Haushalte_alle Szenarien_2022-05-17.xlsx'
    df = pd.read_excel(file_industry, sheet_name='01 Haushalte & Bestand')

    df.drop(columns=[df.columns[0], df.columns[2]], inplace = True)

    table_title = 'Tabelle 01-08: Wirkungsgrade von Raumheizungsanlagen im Szenario ZERO Basis'
    start_table_row = df.index[df['Unnamed: 1'] == table_title].tolist()[1]
    df.columns = df.iloc[start_table_row+2]

    df = df.iloc[start_table_row + 3:start_table_row + 8]

    # Years as int
    df.set_index(['Anlagenart'], inplace=True)
    df.columns = df.columns.astype(int)
    df.reset_index(inplace=True)

    # Change variables names
    df.rename(columns={'Anlagenart': 'Energy_source'}, inplace=True)
    full_name = ['bld_heating_efficiency_' + var + '[%]' for var in
                 df['Energy_source']]
    df['Energy_source'] = full_name

    # Pivot
    df_T = df.T
    df_T.columns = df_T.iloc[0]
    df_T = df_T.iloc[1:]
    df_T.reset_index(inplace=True)
    df_T.rename(columns={121: 'Years'}, inplace=True)
    df_T['Country'] = 'Switzerland'

    dm = DataMatrix.create_from_df(df_T, num_cat=1)

    # Translate
    col_in = ['Heizöl Zentral', 'Gas Zentral', 'Holz Zentral', 'Wärmepumpe', 'Fernwärme']
    col_out = ['heating-oil', 'gas', 'wood', 'heat-pump', 'district-heating']

    dm.rename_col(col_in, col_out, dim='Categories1')

    with open(file_pickle, 'wb') as handle:
      pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return dm


def extract_hot_water_efficiency_JRC(file, sheet_name, years_ots):
  df = pd.read_excel(file, sheet_name=sheet_name)
  df = df[15:25].copy()
  names_map = {'Water heating': 'other-tech', 'Solids': 'coal',
               'Liquified petroleum gas (LPG)': 'remove',
               'Diesel oil': 'heating-oil', 'Natural gas': 'gas',
               'Biomass': 'wood', 'Geothermal': 'geothermal',
               'Distributed heat': 'district-heating',
               'Electricity': 'electricity', 'Solar': 'solar'}
  dm_heating_eff = df_excel_to_dm(df, names_map,
                                  var_name='bld_hot-water-efficiency', unit='%',
                                  num_cat=1)
  dm_heating_eff.drop(col_label='remove', dim='Categories1')

  return dm_heating_eff
