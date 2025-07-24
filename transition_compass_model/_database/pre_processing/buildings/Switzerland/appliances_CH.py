import numpy as np
import pandas as pd
import pickle
import deepl
import requests
import os

from model.common.auxiliary_functions import linear_fitting, create_years_list, my_pickle_dump, add_dummy_country_to_DM
from model.common.io_database import update_database_from_dm, csv_database_reformat, read_database_to_dm
from _database.pre_processing.api_routines_CH import get_data_api_CH
from model.common.data_matrix_class import DataMatrix
from model.common.constant_data_matrix_class import ConstantDataMatrix

# Initialize the Deepl Translator
deepl_api_key = '9ecffb3f-5386-4254-a099-8bfc47167661:fx'
translator = deepl.Translator(deepl_api_key)

def translate_text(text):
    if isinstance(text, str):
        translation = translator.translate_text(text, target_lang='EN-GB')
        out = translation.text
    else:
        out = text
    return out

##########################################################################################################
# Download of files from URL
##########################################################################################################
def save_url_to_file(file_url, local_filename):
    # Loop for URL
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

def get_appliances_stock_new_energy(raw_filename, clean_filename):
  if not os.path.exists(clean_filename):
    df = pd.read_csv(raw_filename)

    cols_de_clean = [col.replace('_', ' ') for col in list(df.columns)]
    df.columns = cols_de_clean
    cols_en = [translate_text(col) for col in list(df.columns)]
    df.columns = cols_en
    # Use deepl to translate variables from de to en
    variables_de = list(set(df['Device category']))
    variables_en = [translate_text(var.replace('_', ' ').replace('(', '').replace(')', '')) for var in variables_de]
    var_dict = dict(zip(variables_de, variables_en))
    df["Categories1"] = df['Device category'].map(var_dict)
    df.rename(columns={'Year': 'Years'}, inplace=True)
    df.drop(columns='Those', inplace=True)
    df.drop(columns='Device category', inplace=True)
    # Step 1: Melt the dataframe to long format
    var_cols = list(set(df.columns) - {'Years', 'Categories1'})
    df_melted = df.melt(id_vars=["Years", "Categories1"],
                        value_vars=var_cols,
                        var_name="Variable", value_name="Value")
    # Step 2: Pivot it
    df_pivoted = df_melted.pivot_table(index="Years",
                                       columns=["Variable", "Categories1"],
                                       values="Value")

    # Step 3: Flatten the column MultiIndex
    df_pivoted.columns = [f"{var}_{cat}" for var, cat in df_pivoted.columns]

    # Step 4: Reset index to get "Years" back as a column
    df_final = df_pivoted.reset_index()

    df_final['Country'] = 'Switzerland'

    for col in set(df_final.columns) - {'Years', 'Country'}:
      if 'kWh' not in col:
        df_final.rename(columns={col: col+'[number]'}, inplace=True)
      else:
        df_final.rename(columns={col: col + '[kWh]'}, inplace=True)

    dm = DataMatrix.create_from_df(df_final, num_cat=1)
    vars_in = ['Consumption of appliances kWh', 'Geraetebestand Stk',
               'New appliance consumption kWh', 'New product sales pcs']
    vars_out = ['bld_appliances_electricity-demand_stock', 'bld_appliances_stock',
                'bld_appliances_electricity-demand_new', 'bld_appliances_new']
    dm.rename_col(vars_in, vars_out, dim='Variables')
    dm.sort('Variables')

    with open(clean_filename, 'wb') as handle:
      pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)
  else:
    print(
      f'File {clean_filename} already exists. If you want to download again delete the file')

    with open(clean_filename, 'rb') as handle:
      dm = pickle.load(handle)

  return dm

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
  dm.rename_col(cantons_fr, cantons_en, dim='Country')
  dm.sort('Country')
  return dm


def get_households_number(raw_filename, clean_filename):
  if not os.path.exists(clean_filename):
    sheets_dict = pd.read_excel(raw_filename, sheet_name=None)

    dm = None
    for name, sheet in sheets_dict.items():
      print(name)
      df = sheets_dict[name]
      df.columns = df.iloc[0]
      df = df.iloc[2:]
      df = df[['Canton', 'Total']]
      # Drop rows if they contain a nan
      df.dropna(axis=0, how='any', inplace=True)
      df.rename(columns={'Canton': 'Country', 'Total': 'bld_households[number]'}, inplace=True)
      df['Years'] = int(name)
      df = df.replace('()', np.nan)
      if dm is None:
        dm = DataMatrix.create_from_df(df, num_cat=0)
      else:
        dm_yr = DataMatrix.create_from_df(df, num_cat=0)
        dm.append(dm_yr, dim='Years')
    dm.sort('Years')
    with open(clean_filename, 'wb') as handle:
      pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)
  else:
    print(
      f'File {clean_filename} already exists. If you want to download again delete the file')
    with open(clean_filename, 'rb') as handle:
      dm = pickle.load(handle)

  return dm


def households_fill_missing_years(dm_households, dm_pop, years_all):
  # Add missing years as nans
  missing_years = list(set(years_all) - set(dm_households.col_labels['Years']))
  dm_households.add(np.nan, col_label = missing_years, dummy=True, dim='Years')
  dm_households.sort('Years')

  # Join dm_households and dm_pop
  dm_households.append(dm_pop.filter({'Country': dm_households.col_labels['Country']}), dim='Variables')
  dm_households.operation('lfs_population_total', '/', 'bld_households', out_col='lfs_household-size', unit='people')

  linear_fitting(dm_households, years_ots)

  dm_households.drop('Variables', 'bld_households')

  dm_households.operation('lfs_population_total', '/', 'lfs_household-size', out_col='bld_households', unit='number')

  dm_households.drop('Variables', 'lfs_population_total')

  return dm_households


def appliances_fill_missing_years(dm_appliances, dm_households):
  def tailored_fitting(dm):
    linear_fitting(dm, years_ots=list(range(1990, 2010)), based_on=list(range(2002, 2006)))
    dm.fill_nans('Years')
    idx = dm.idx
    dm.array[:, idx[2022]:, :, :] = np.nan
    linear_fitting(dm, years_ots=years_ots+years_fts, based_on=list(range(2018, 2021)))
    dm.array = np.maximum(0, dm.array)
    return dm
  dm_households_only = dm_households.filter({'Variables': ['bld_households'], 'Years': years_ots+years_fts})
  # Filter out efficiency
  dm_eff = dm_appliances.filter({'Variables': ['bld_appliances_electricity-demand']})
  dm_appliances.drop('Variables', 'bld_appliances_electricity-demand')
  # Compute appliances per household
  dm_appliances.sort('Country')
  dm_households_only.sort('Country')
  dm_appliances.array = dm_appliances.array / dm_households_only.array[..., np.newaxis]
  #dm_appliances.change_unit('bld_appliances_stock', old_unit='number', new_unit='unit/household', factor=1)
  # Impose ownership for PC and monitors
  # Source US labor: https://www.bls.gov/opub/btn/archive/computer-ownership-up-sharply-in-the-1990s.pdf
  dm_appliances[:, 1990, 'bld_appliances_stock', 'PC'] = 0.15
  dm_appliances[:, 1990, 'bld_appliances_stock', 'monitor'] = 0.15
  dm_PC = dm_appliances.filter({'Categories1': ['PC', 'monitor']})
  dm_PC.fill_nans('Years')
  dm_appliances.drop('Categories1', ['PC', 'monitor'])
  dm_appliances.append(dm_PC, 'Categories1')
  # Fill nans
  dm_appliances = tailored_fitting(dm_appliances)
  idx =dm_appliances.idx
  dm_appliances.array[:, idx[2022]:, :, idx['monitor']] = np.nan
  dm_appliances.fill_nans('Years')
  dm_appliances.array = dm_appliances.array * dm_households_only.array[..., np.newaxis]
  dm_eff =tailored_fitting(dm_eff)
  dm_appliances.append(dm_eff, dim='Variables')

  return dm_appliances

#------------------------------------------------------------------
########################################################
###    APPLIANCES NEW, STOCK, ENERGY CONSUMPTION    ####
########################################################

years_ots = create_years_list(1990,2023, 1)
years_fts = create_years_list(2025, 2050, 5)

# Load population
file = '../../../data/datamatrix/lifestyles.pickle'
with open(file, 'rb') as handle:
  DM_lfs = pickle.load(handle)
dm_pop_ots = DM_lfs['ots']['pop']['lfs_population_'].copy()
dm_pop = dm_pop_ots.copy()
dm_pop.append(DM_lfs['fts']['pop']['lfs_population_'][1], dim='Years')

# Opensuisse database with data on large appliances and electronics in Switzerland
# Containing data on stock, new, and their energy consumption
# https://opendata.swiss/de/dataset/absatz-und-stromverbrauchswerte-von-elektro-und-elektronischen-gerate-in-der-schweiz
# DATA - Absatz- und Stromverbrauchswerte von Elektro- und elektronischen Geräte in der Schweiz
file_url = "https://www.uvek-gis.admin.ch/BFE/ogd/109/ogd109_absatz_verbrauch_elektrogeraete.csv"
raw_filename = 'data/appliances_stock_new_energy_CH_raw.csv'
save_url_to_file(file_url, raw_filename)
clean_filename = 'data/appliances_stock_new_energy_CH.pickle'

dm_appliances = get_appliances_stock_new_energy(raw_filename, clean_filename)

# Filter & Rename only the appliances you want to keep
app_map = {'freezer': ['Freezers'], 'refrigerator': ['Refrigerators'],
           'dishwasher': ['Dishwasher'], 'washing-machine': ['Washing machines'],
           'oven-and-stove': ['Electric cookers and built-in ovens'],
           'tumble-dryer': ['Tumble dryer'], 'TV': ['Televisions'],
           'PC': ['Personal Computer'], 'laptop': ['Laptops'],
           'monitor': ['Monitor'], 'set-top-box': ['Set-top boxes']}
dm_appliances = dm_appliances.groupby(app_map, dim='Categories1', inplace=False)

missing_years = list(set(years_ots + years_fts) - set(dm_appliances.col_labels['Years']))
dm_appliances.add(np.nan, dim='Years', col_label=missing_years, dummy=True)
dm_appliances.sort('Years')

# Mobile phones are missing but their electricity consumption seems minimal. 31 kWh/year vs 2000 kWh/year

# Number of households
file_url = 'https://dam-api.bfs.admin.ch/hub/api/dam/assets/27965837/master'
raw_filename = 'data/households_number.xlsx'
save_url_to_file(file_url, raw_filename)
clean_filename = 'data/households_number.pickle'

dm_households = get_households_number(raw_filename, clean_filename)
dm_households = clean_country_names(dm_households)

# Compute number of households from population by linear fitting of ppl/households
dm_households = households_fill_missing_years(dm_households, dm_pop, years_ots + years_fts)
dm_households_ots = dm_households.filter({'Years':years_ots})

# Map appliances from CH to cantons based on number of households
dm_households_CH = dm_households.filter({'Country': ['Switzerland']})
#dm_households_ots.normalise('Country', inplace=True)

#arr_appl_cantons = (dm_appliances['Switzerland', np.newaxis, :, :, :]
#                    * dm_households_ots[:, :, 'bld_households', np.newaxis, np.newaxis])
#dm_appliances.add(arr_appl_cantons, dim='Country', col_label=dm_households_ots.col_labels['Country'])

# Linear fitting based on appliances / households from the 2002-2005 trend
dm_appliances.operation('bld_appliances_electricity-demand_stock', '/', 'bld_appliances_stock', out_col='bld_appliances_electricity-demand', unit='kWh/unit')
dm_appliances.filter({'Variables': ['bld_appliances_stock', 'bld_appliances_new', 'bld_appliances_electricity-demand']}, inplace=True)

# Determine the waste
# s(t) = s(t-1) + n(t) - w(t)
# w(t) = s(t-1) + n(t) - s(t)
dm_appliances.lag_variable('bld_appliances_stock', shift=1, subfix='_tm1')
dm_appliances.operation('bld_appliances_stock_tm1', '-', 'bld_appliances_stock', out_col='bld_delta_stock', unit='unit')
dm_appliances.operation('bld_delta_stock', '+', 'bld_appliances_new', out_col='bld_appliances_waste', unit='unit')
dm_appliances.operation('bld_appliances_waste', '/', 'bld_appliances_stock_tm1', out_col='bld_appliances_retirement-rate', unit='%')

dm_rr = dm_appliances.filter({'Variables': ['bld_appliances_retirement-rate']})
dm_rr.fill_nans('Years')

dm_appliances.filter({'Variables': ['bld_appliances_stock', 'bld_appliances_electricity-demand']}, inplace=True)

# Fill nans for stock and energy demand/unit
dm_appliances = appliances_fill_missing_years(dm_appliances, dm_households_CH)
dm_appliances.append(dm_rr, dim='Variables')

dm_appliances.lag_variable('bld_appliances_stock', shift=1, subfix='_tm1')
dm_appliances.operation('bld_appliances_retirement-rate', '*', 'bld_appliances_stock_tm1', out_col='bld_appliances_waste', unit='%')
# Determine the waste
# s(t) = s(t-1) + n(t) - w(t)
# n(t) = w(t) - (s(t-1) -s(t))
dm_appliances.operation('bld_appliances_stock_tm1', '-', 'bld_appliances_stock', out_col='bld_delta_stock', unit='unit')
dm_appliances.operation('bld_appliances_waste' , '-', 'bld_delta_stock', out_col='bld_appliances_new', unit='unit')
dm_appliances.filter({'Variables': ['bld_appliances_stock', 'bld_appliances_retirement-rate', 'bld_appliances_electricity-demand']}, inplace=True)

# Compute stock per households
dm_appliances['Switzerland', :, 'bld_appliances_stock', :] \
  = (dm_appliances['Switzerland', :, 'bld_appliances_stock', :]
     / dm_households_CH['Switzerland', :, 'bld_households', np.newaxis])
dm_appliances.change_unit('bld_appliances_stock', factor=1, old_unit='number', new_unit='unit/household')

# Add all cantons
cantons_list = list(set(dm_households.col_labels['Country']) - {'Switzerland'})
dm_appliances.add(np.nan, dim='Country', dummy=True, col_label=cantons_list)
dm_appliances.fill_nans('Country')

# Prepare as output the stock/household, the retirement-rate,
# the households size, the energy-demand
idx = dm_appliances.idx
dm_appliances.array[:, idx[2023]:, idx['bld_appliances_electricity-demand'], idx['oven-and-stove']] = np.nan
linear_fitting(dm_appliances, years_ots+years_fts)

dm_households.filter({'Variables':['lfs_household-size']})

DM = {'fxa': {'appliances': dm_appliances, 'household': dm_households}}
add_dummy_country_to_DM(DM, 'EU27', 'Switzerland')

# !FIXME add this to the pickle
print('Hello')
