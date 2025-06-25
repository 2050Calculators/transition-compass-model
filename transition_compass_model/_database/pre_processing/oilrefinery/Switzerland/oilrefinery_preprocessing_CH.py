import numpy as np
import pandas as pd
import pickle
import faostat

from model.common.auxiliary_functions import moving_average, linear_fitting, create_years_list
from model.common.io_database import update_database_from_dm, csv_database_reformat, read_database_to_dm
from _database.pre_processing.api_routines_CH import get_data_api_CH
from model.common.data_matrix_class import DataMatrix
from model.common.constant_data_matrix_class import ConstantDataMatrix
import eurostat
import math
import requests
from _database.pre_processing.WorldBank_data_extract import get_WB_data
import os


def extract_oil_product_demand(file_url, local_filename, sheet_name, years_ots):
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

    df = pd.read_excel(local_filename, sheet_name=sheet_name)
    # Merge two headers rows in a single header
    combined_headers = []
    for col1, col2 in zip(df.iloc[4], df.iloc[5]):
        combined_headers.append(str(col1) + '-' + str(col2))
    # Set the new header
    df.columns = combined_headers
    df = df[6:].copy()  # data start at line 6
    # Remove nan- in column name
    df.columns = [str.replace(col, 'nan-', '') for col in df.columns]
    df.columns = [str.replace(col, 'nan', '') for col in df.columns]
    def is_valid_number(val):
        return isinstance(val, (int, float)) and not pd.isna(val)

    # Apply the function to filter out rows with no valid numeric values
    df = df[df.apply(lambda row: row.map(is_valid_number).any(), axis=1)]
    # Apply similarly for columns if needed
    df = df.loc[:, df.apply(lambda col: col.map(is_valid_number).any())]

    # Rename columns
    df.rename({'Année-': 'Years'}, axis=1, inplace=True)
    df['Country'] = 'Switzerland'
    rename_dict = {
        'Huile extra-légère-': 'lightfueloil',
        'Huile moyenne et lourde-': 'heavierfueloil',
        "Essence2-Total": "gasoline",
        "Carburants d'aviation2-": "jetfuel",
        "Carburant diesel2-": "diesel",
        "Coke de pétrole3-": "oilcoke",
        "Autres produits pétroliers énerg.4-": "other"
        }
    df.rename(rename_dict, axis=1, inplace=True)
    df.set_index(['Years', 'Country'], inplace=True)  # put Years, Country as index
    df = df[rename_dict.values()]  # Keep only columns that were renamed

    # Add "ory_oil-product-demand_" as prefix and units to columns
    df = df.add_prefix('ory_oil-product-demand_')
    df = df.add_suffix('[kt]')
    df.reset_index(inplace=True)  # put Years, Country back to columns
    dm = DataMatrix.create_from_df(df, num_cat=1)

    dm.filter({'Years': years_ots}, inplace=True)  # keep only ots years
    # dm.datamatrix_plot()  # plotting
    return dm


def create_BAU_oil_product_demand_fts(dm, years_fts):
    #dm_opt1 = dm.filter({'Categories1': ['jetfueloil']})
    #dm_opt2 = dm.copy()
    #dm_opt2.drop(dim='Categories1', col_label='jetfueloil')
    #dm_opt1.append(dm_opt2, dim='Categories1')
    #dm = dm_opt1.copy()

    # dm.fill_nans('Years')  #flat extrapolation
    ref_years = list(range(2000, 2023 + 1))
    linear_fitting(dm, years_fts, based_on=ref_years)

    dm.array = np.maximum(dm.array, 0)

    dm_fts = dm.filter({'Years': years_fts})

    return dm_fts


years_ots = create_years_list(1990, 2023, 1)
years_fts = create_years_list(2025, 2050, 5)
# Initialise oil-refinery pickle structure
DM_refinery_new = {'fxa': dict(), 'ots': dict(), 'fts': dict()}

# SECTION: Calculationleaf: Extract oil-product demand for calibration
file_url = 'https://www.bfe.admin.ch/bfe/fr/home/versorgung/statistik-und-geodaten/energiestatistiken/gesamtenergiestatistik.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZnIvcHVibGljYX/Rpb24vZG93bmxvYWQvNzUxOQ==.html'
#local_filename = 'data/7519-gest_2023_tabellen.xlsx'
local_filename = 'data/energy_table.xlsx'
sheet_name = 'T20'
dm_oil_product_demand = extract_oil_product_demand(file_url, local_filename, sheet_name, years_ots)
DM_refinery_new['fxa']['oil-demand'] = dm_oil_product_demand.copy()

# SECTION: Calculationleaf: Forecast oil-product demand for fts years BAU
dm_oil_product_BAU_fts = create_BAU_oil_product_demand_fts(dm_oil_product_demand, years_fts)
DM_refinery_new['fts']['oil-demand'] = dict()
for lev in range(4):
    lev = lev+1
    DM_refinery_new['fts']['oil-demand'][lev] = dm_oil_product_BAU_fts


# If you want to define another fts level
#dm_oil_product_demand_ambitious = dm_oil_product_BAU_fts.copy()
#dm_oil_product_demand_ambitious.array[...] = 0
#DM_refinery_new['fts']['oil-demand'][4] = dm_oil_product_demand_ambitious

# If you want to plot together
#dm_oil_product_demand_ambitious.rename_col('ory_oil-product-demand', 'ambitious', dim='Variables')
#dm_oil_product_BAU_fts.append(dm_oil_product_demand_ambitious, dim='Variables')
#dm_oil_product_BAU_fts.datamatrix_plot()

# If you want to save your pickle
#file = '../../../data/datamatrix/oil-refinery.pickle'
#with open(file, 'rb') as handle:
#    DM_refinery = pickle.load(handle)

#file = '../../../data/datamatrix/oil-refinery.pickle'
#with open(file, 'wb') as handle:
#    pickle.dump(DM_refinery_new, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Hello')