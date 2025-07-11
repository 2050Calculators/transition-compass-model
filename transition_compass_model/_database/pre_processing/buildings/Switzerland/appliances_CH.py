import numpy as np
import pandas as pd
import pickle
import deepl
import requests
import os

from model.common.auxiliary_functions import moving_average, linear_fitting, create_years_list, my_pickle_dump, cdm_to_dm
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
    new_names = ['Years']
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

#------------------------------------------------------------------
########################################################
###    APPLIANCES NEW, STOCK, ENERGY CONSUMPTION    ####
########################################################
# Opensuisse database with data on large appliances and electronics in Switzerland
# Containing data on stock, new, and their energy consumption
# https://opendata.swiss/de/dataset/absatz-und-stromverbrauchswerte-von-elektro-und-elektronischen-gerate-in-der-schweiz
# DATA - Absatz- und Stromverbrauchswerte von Elektro- und elektronischen Geräte in der Schweiz
file_url = "https://www.uvek-gis.admin.ch/BFE/ogd/109/ogd109_absatz_verbrauch_elektrogeraete.csv"
raw_filename = 'data/appliances_stock_new_energy_CH_raw.csv'
save_url_to_file(file_url, raw_filename)
clean_filename = 'data/appliances_stock_new_energy_CH.pickle'

dm_appliances = get_appliances_stock_new_energy(raw_filename, clean_filename)

print('Hello')
