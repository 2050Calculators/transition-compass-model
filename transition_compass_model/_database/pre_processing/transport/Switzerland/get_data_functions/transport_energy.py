from model.common.data_matrix_class import DataMatrix
import pickle
from model.common.auxiliary_functions import save_url_to_file
import os
import pandas as pd
import zipfile


def extract_EP2050_transport_energy_demand(file_url, zip_name, file_pickle):

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

    file_tra = extract_dir + '/EP2050+_Szenarienergebnisse_Details_Nachfragesektoren/EP2050+_Detailergebnisse 2020-2060_Verkehrssektor_alle Szenarien_2022-04-12.xlsx'
    df = pd.read_excel(file_tra, sheet_name='04 Energieverbrauch Strasse')

    df.drop(columns=[df.columns[0], df.columns[3]], inplace=True)

    table_title = 'Tabelle 04-05: Entwicklung des Energieverbrauchs im Szenario Weiter wie bisher'
    start_table_row = df.index[df['Unnamed: 1'] == table_title].tolist()[1]
    df.columns = df.iloc[start_table_row + 2]

    df = df.iloc[start_table_row + 3:start_table_row + 29]

    # Years as int
    col_mode_name = df.columns[0]
    col_tech_name = df.columns[1]
    df.set_index([col_mode_name, col_tech_name], inplace=True)
    df.columns = df.columns.astype(int)
    df.reset_index(inplace=True)

    # Change variables names
    full_name = ['tra_energy_demand_' + var for var in
                 df[col_mode_name]]
    df[col_mode_name] = full_name
    df['Full_name'] = df[col_mode_name] + '_' + df[col_tech_name] + ['[PJ]']
    df.drop(columns = [col_mode_name, col_tech_name], inplace=True)
    # Move "Full_name" column at the beginning
    first = df['Full_name']
    df.drop(labels=['Full_name'], axis=1, inplace=True)
    df.insert(0, 'Full_name', first)

    # Pivot
    df_T = df.T
    df_T.columns = df_T.iloc[0]
    df_T = df_T.iloc[1:]
    df_T.reset_index(inplace=True)
    df_T.rename(columns={158: 'Years'}, inplace=True)
    df_T['Country'] = 'Switzerland'

    dm = DataMatrix.create_from_df(df_T, num_cat=2)

    # Rename mode of transport
    dm.rename_col(['HGV', 'LCV', 'motorcycle', 'pass. car'],
                  ['HDVH', 'HDVL',  '2W', 'LDV'], dim='Categories1')
    dm.groupby({'bus': ['coach', 'urban bus']}, dim='Categories1', inplace=True)
    # Rename tech transport
    dm.groupby({'BEV': ['electricity'], 'biogasoline': ['E85'],
                'ICE-gas': ['CNG', 'LPG'], 'gasoline': ['petrol', 'petrol 2S'],
                'FCEV': ['hydrogen'], 'ICE-diesel': ['diesel']}, dim='Categories2', inplace=True)

    with open(file_pickle, 'wb') as handle:
      pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

  dm.sort('Categories1')
  dm.sort('Categories2')
  dm.change_unit('tra_energy_demand', old_unit='PJ', new_unit='TWh', factor=3.6, operator='/')

  return dm


def extract_EP2050_transport_energy_demand_rail(file_url, zip_name, file_pickle):

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

    file_tra = extract_dir + '/EP2050+_TechnsicherBericht_DatenAbbildungen_Kap 8_2022-04-12.xlsx'
    df = pd.read_excel(file_tra, sheet_name='Abb. 112')

    df.drop(columns=[df.columns[0]], inplace=True)

    # Headers
    cols = list(df.iloc[28])
    cols[0] = 'Variables'
    df.columns = cols

    df = df.iloc[29:-1]

    # Years as int

    col_name = 'Variables'
    df.set_index([col_name], inplace=True)
    df.columns = df.columns.astype(int)
    df.reset_index(inplace=True)

    # Change variables names
    full_name = ['tra_energy_demand_' + var + '[PJ]' for var in
                 df[col_name]]
    df[col_name] = full_name

    # Pivot
    df_T = df.T
    df_T.columns = df_T.iloc[0]
    df_T = df_T.iloc[1:]
    df_T.reset_index(inplace=True)
    df_T.rename(columns={'index': 'Years'}, inplace=True)
    df_T['Country'] = 'Switzerland'

    dm = DataMatrix.create_from_df(df_T, num_cat=0)

    # Rename mode of transport
    dm.rename_col_regex('Flugverkehr', 'aviation', dim='Variables')
    dm.rename_col_regex('Güterverkehr', 'freight', dim='Variables')
    dm.rename_col_regex('Personenverkehr', 'passenger', dim='Variables')
    dm.rename_col_regex(' Schiene', '_rail', dim='Variables')
    dm.rename_col_regex(' Strasse', '_road', dim='Variables')

    with open(file_pickle, 'wb') as handle:
      pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

  dm_aviation = dm.filter_w_regex({'Variables': '.*aviation.*'})
  dm_rail = dm.filter_w_regex({'Variables': '.*rail'})
  dm_rail.deepen()
  dm_rail.deepen(based_on='Variables')

  dm_rail.change_unit('tra_energy_demand', old_unit='PJ', new_unit='TWh', factor=3.6, operator='/')

  return dm_rail
