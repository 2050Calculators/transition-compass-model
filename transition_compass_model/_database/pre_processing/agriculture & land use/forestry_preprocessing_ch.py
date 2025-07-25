import pandas as pd
import numpy as np
from _database.pre_processing.api_routines_CH import get_data_api_CH
import pickle
import os
import requests
import deepl
import faostat
from model.common.data_matrix_class import DataMatrix
from model.common.constant_data_matrix_class import ConstantDataMatrix
from model.common.auxiliary_functions import create_years_list, linear_fitting, my_pickle_dump
from model.common.auxiliary_functions import sort_pickle, add_dummy_country_to_DM

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

##########################################################################################################
# Wood production in CH and Cantons
##########################################################################################################
def get_wood_production(table_id, file):
    try:
        # Try to look for the pickle
        with open(file, 'rb') as handle:
            dm = pickle.load(handle)

    # No pickle available
    except OSError:
        structure, title = get_data_api_CH(table_id, mode='example')

        filtering = {'Year': structure['Year'],
                     'Forest zone': ['Switzerland'],
                     'Canton': ['Switzerland', 'Vaud'],
                     'Type of owner': ['Type of owners - total'],
                     'Wood species': structure['Wood species'],
                     'Observation unit': structure['Observation unit']}

        mapping_dim = {'Country': 'Canton',
                       'Years': 'Year',
                       'Variables': 'Observation unit',
                       'Categories1': 'Wood species'}

        # Extract new fleet
        dm = get_data_api_CH(table_id, mode='extract', filter=filtering, mapping_dims=mapping_dim,
                                   units=['m3'] * len(structure['Observation unit']))
        dm.sort('Years')

        #Write the pickle
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, 'wb') as handle:
            pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm

##########################################################################################################
# Wood production in CH and Cantons
##########################################################################################################

def get_wood_energy_by_sector(file_url, local_filename):
    ### Creates the file
    save_url_to_file(file_url, local_filename)
    ### Read the file
    df = pd.read_excel(local_filename, sheet_name='M')
    #filter lines
    df = df.iloc[0:9]
    #filter column:
    df = df.drop(df.columns[0], axis=1)

    # header change
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    # Nan remove
    df = df.dropna(how='all')
    df.columns = [str(int(col)) if isinstance(col, float) else str(col) for col in df.columns]
    ### translate
    new_names=['Years']
    for i in range(1):
        # Use deepl to translate variables from de to en
        variables_de = list(set(df[df.columns[i]]))
        variables_en = [translate_text(var) for var in variables_de]
        var_dict = dict(zip(variables_de, variables_en))
        df[new_names[i]] = df[df.columns[i]].map(var_dict)

    df.drop([df.columns[0]], axis=1, inplace=True)

    # change the column position
    df = df[[df.columns[-1]] + list(df.columns[:-1])]

    ##Transpose
    #df = df.T
    df = df.T
    df = df.reset_index()
    df = df.rename(columns={'index': 'year'})
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)

    #Country
    df.insert(0, "Country", "Switzerland")

    # New names
    df = df.rename(columns={
        'Households': 'fst_wood-energy-demand_households[TJ]',
        'Agriculture / Forestry': 'fst_wood-energy-demand_agriculture-forestry[TJ]',
        'District heating':'fst_wood-energy-demand_district-heating[TJ]',
        'services': 'fst_wood-energy-demand_services[TJ]',
        'Industry / Trade':'fst_wood-energy-demand_industry[TJ]',
        'All system categories (Cat. 1 - 20)':'fst_wood-energy-demand_total[TJ]',
        'Electricity':'fst_wood-energy-demand_electricity[TJ]'
    })
    # Convert to DM
    dm_wood_demand_energy = DataMatrix.create_from_df(df, num_cat=1)

    # Conversion from TJ to GWh:
    dm_wood_demand_energy.change_unit('fst_wood-energy-demand', factor=0.27778, old_unit='TJ', new_unit='GWh')
    #dm_wood_demand_energy.datamatrix_plot()
    return dm_wood_demand_energy

##########################################################################################################
# Wood consumption in CH and Cantons
##########################################################################################################

def get_wood_energy_by_use(file_url, local_filename, clean_local_filename):

    def clean_excel_bloc(df):
        # header change
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        # Nan remove
        df = df.dropna(how='all')
        df.columns = [str(int(col)) if isinstance(col, float) else str(col) for col in df.columns]
        ### translate
        new_names = ['Years']
        for i in range(1):
            # Use deepl to translate variables from de to en
            variables_de = list(set(df[df.columns[i]]))
            variables_en = [translate_text(var) for var in variables_de]
            var_dict = dict(zip(variables_de, variables_en))
            df[new_names[i]] = df[df.columns[i]].map(var_dict)

        df.drop([df.columns[0]], axis=1, inplace=True)

        # change the column position
        df = df[[df.columns[-1]] + list(df.columns[:-1])]

        ##Transpose
        # df = df.T
        df = df.T
        df = df.reset_index()
        df = df.rename(columns={'index': 'year'})
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)

        return df

    ### Creates the file
    save_url_to_file(file_url, local_filename)

    if not os.path.exists(clean_local_filename['energy-use-m3']):
        ### Read the file
        df_raw = pd.read_excel(local_filename, sheet_name='R')

        # Filtering the matrix for wood use in m3 (TJ to follow)
        df = df_raw.copy()
        df = df.iloc[0:11]

        # Clean dataframe
        df = clean_excel_bloc(df)

        # Country
        df.insert(0, "Country", "Switzerland")

        # New names
        df = df.rename(columns={
            'Total incl. KVA (Cat 1-20)': 'fst_wood-energy-use_total[m3]',
            'Waste wood without MSWI (without cat. 20)': 'fst_wood-energy-use_waste-without-incineration[m3]',
            'Waste wood in MSWI (only cat. 20)': 'fst_wood-energy-use_waste-incineration[m3]',
            'Residual wood from wood processing plants': 'fst_wood-energy-use_wood-byproducts[m3]',
            'Wood pellets *)': 'fst_wood-energy-use_pellets[m3]',
            'Natural logs': 'fst_wood-energy-use_natural-logs[m3]',
            'Natural non-chunky wood': 'fst_wood-energy-use_natural-non-chunky-wood[m3]',
            'Total without MWIP (Cat 1-19)': 'fst_wood-energy-use_total-without-incineration[m3]',
            'Total without KVA (Cat 1-19)': 'fst_wood-energy-use_total-without-incineration[m3]'
        })

        df.to_csv(clean_local_filename['energy-use-m3'], sep=",", index=False)

    if not os.path.exists(clean_local_filename['energy-use-ghw']):
        ### Read the file
        df_raw = pd.read_excel(local_filename, sheet_name='R')

        # Filtering the matrix for wood use in TJ
        df = df_raw.copy()
        df = df.iloc[14:25]

        # Clean dataframe
        df = clean_excel_bloc(df)

        # Country
        df.insert(0, "Country", "Switzerland")

        # New names
        df = df.rename(columns={
            'Total incl. KVA (Cat 1-20)': 'fst_wood-energy-use_total[TJ]',
            'Waste wood without MSWI (without cat. 20)': 'fst_wood-energy-use_waste-without-incineration[TJ]',
            'Waste wood in MSWI (only cat. 20)': 'fst_wood-energy-use_waste-incineration[TJ]',
            'Residual wood from wood processing plants': 'fst_wood-energy-use_wood-byproducts[TJ]',
            'Wood pellets *)': 'fst_wood-energy-use_pellets[TJ]',
            'Natural logs': 'fst_wood-energy-use_natural-logs[TJ]',
            'Natural non-chunky wood': 'fst_wood-energy-use_natural-non-chunky-wood[TJ]',
            'Total without MWIP (Cat 1-19)': 'fst_wood-energy-use_total-without-incineration[TJ]',
            'Total without KVA (Cat 1-19)': 'fst_wood-energy-use_total-without-incineration[TJ]'
        })

        df.to_csv(clean_local_filename['energy-use-ghw'], sep=",", index=False)

    df = pd.read_csv(clean_local_filename['energy-use-m3'])
    # Convert to DM
    dm_wood_energy_use_m3 = DataMatrix.create_from_df(df, num_cat=1)

    df = pd.read_csv(clean_local_filename['energy-use-ghw'])
    # Convert to DM
    dm_wood_energy_use_gwh = DataMatrix.create_from_df(df, num_cat=1)

    # Conversion from TJ to GWh:
    dm_wood_energy_use_gwh.change_unit('fst_wood-energy-use', factor=0.27778, old_unit='TJ', new_unit='GWh')
    dm_wood_energy_use_gwh.rename_col("fst_wood-energy-use","fst_wood-energy-use-gwh", "Variables")
    dm_wood_energy_use_m3.rename_col("fst_wood-energy-use", "fst_wood-energy-use-m3", "Variables")
    #dm_wood_demand_energy.datamatrix_plot()

    dm_wood_energy_use=dm_wood_energy_use_gwh
    dm_wood_energy_use.append(dm_wood_energy_use_m3, dim='Variables')
    dm_wood_energy_use.operation('fst_wood-energy-use-gwh', '/', 'fst_wood-energy-use-m3', out_col='fst_energy-density', unit='gwh/m3')
    #dm_wood_energy_use.datamatrix_plot({'Variables': ['fst_energy-density']})

    return dm_wood_energy_use

##########################################################################################################
# Wood production in CH and Cantons
##########################################################################################################
def get_forest_area(table_id, file):
    try:
        with open(file, 'rb') as handle:
            dm = pickle.load(handle)
    except OSError:
        structure, title = get_data_api_CH(table_id, mode='example')

        filtering = {'Year': structure['Year'],
                     'Forest zone': ['Switzerland'],
                     'Canton': ['Switzerland', 'Vaud'],
                     'Type of owner':['Type of owners - total'],
                     'Observation unit':['Total forest area', 'Productive forest area']}

        mapping_dim = {'Country': 'Canton',
                       'Years': 'Year',
                       'Variables': 'Observation unit',
                       'Categories1': 'Type of owner'}

        # Extract new fleet
        dm = get_data_api_CH(table_id, mode='extract', filter=filtering, mapping_dims=mapping_dim,
                                   units=['ha'] * len(structure['Observation unit']))
        dm= dm.flatten()
        df = dm.write_df()
        dm.sort('Years')

        with open(file, 'wb') as handle:
            pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm

def get_wood_trade_balance(file):
    try:
        with open(file, 'rb') as handle:
            dm = pickle.load(handle)
    except OSError:
        # My PARSE -------------------------------------------------
        code = 'FO'

        # Metadata
        list_data = faostat.list_datasets()
        list_data
        list_items = faostat.get_par(code, 'items')
        list_items
        list_itemsagg = faostat.get_par(code, 'itemsagg')
        list_itemsagg
        list_area = faostat.get_par(code, 'area')
        list_area
        list_pars = faostat.list_pars(code)
        list_pars

        # My selection
        areas = ['Switzerland']
        years = [str(y) for y in range(1990, 2024)]
        elements = ['Export quantity','Import quantity','Production Quantity']
        itemsagg = ['Roundwood + (Total)','Roundwood > (List)',
                    'Roundwood, coniferous > (List)',
                    'Roundwood, non-coniferous > (List)']

        my_areas = [faostat.get_par(code, 'area')[c] for c in areas]
        my_elements = [faostat.get_par(code, 'elements')[e] for e in elements]
        my_itemsagg = [faostat.get_par(code, 'itemsagg')[i] for i in itemsagg]
        my_years = [faostat.get_par(code, 'year')[y] for y in years]

        my_pars = {
            'area': my_areas,
            'element': my_elements,
            'year':my_years,
            'itemsagg':my_itemsagg
        }
        dm = faostat.get_data_df(code, pars=my_pars, strval=False)

        # Filter out FAOSTAT columns
        dm = dm[['Area', 'Year', 'Value', 'Element', 'Item', 'Unit']]

        # Shaping to turn into DM
        dm = dm.rename(columns={'Year': 'Years', 'Area': 'Country'})

        dm['Variable'] = dm.apply(
            lambda row: f"{row['Item'].lower()}_{row['Element'].lower()}[{row['Unit'].lower()}]", axis=1)
        dm = dm.drop(columns=['Item', 'Element', 'Unit'])
        dm = dm.pivot(index=['Country', 'Years'], columns='Variable', values='Value').reset_index()
        dm = DataMatrix.create_from_df(dm, num_cat=1)
        dm = dm.filter_w_regex({'Variables': 'industrial roundwood|industrial roundwood, coniferous|industrial roundwood, non-coniferous|\
        other industrial roundwood|pulp for paper|\
        roundwood|roundwood, coniferous|roundwood, non-coniferous|\
        sawlogs and veneer logs|sawlogs and veneer logs, coniferous|\
        sawlogs and veneer logs, non-coniferous|\
        wood fuel|wood fuel, coniferous|wood fuel, non-coniferous'})

        with open(file, 'wb') as handle:
            pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm

################################################################
# Simulate Industry Interface as a Pickle
################################################################
def simulate_industry_input(write_pickle= True):
    if write_pickle is True:
        # Path for the input from industry
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, "data/fake_ind-to-fst.xlsx")
        # Read the file:
        df = pd.read_excel(f, sheet_name="ind-to-fst")
        # Build DataMatrix
        dm = DataMatrix.create_from_df(df, num_cat=1)
        dm.add(np.nan, dim='Country', col_label='EU27', dummy=True)
        dm['EU27', ...] = dm['Switzerland', ...]
        # Write Pickle
        f = os.path.join(current_file_directory, '../../data/interface/industry_to_forestry.pickle')
        my_pickle_dump(DM_new=dm, local_pickle_file=f)

    return



################################################################
# Simulate Industry Interface as a Pickle
################################################################
def simulate_land_input(dm_forest_area, write_pickle= True):
    if write_pickle is True:
        # Build DataMatrix
        dm = dm_forest_area.filter({'Variables': ['total-forest-area']})
        # Extract OTS & Add FTS
        dm.filter({'Years': years_ots}, inplace=True)
        dm.add(np.nan, col_label=years_fts, dim='Years', dummy=True)
        # Linear extrapolation on future years
        linear_fitting(dm, years_fts)
        dm.add(np.nan, dim='Country', col_label = 'EU27', dummy=True)
        dm['EU27', ...] = dm['Switzerland', ...]
        # Write Pickle
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, '../../data/interface/land_to_forestry.pickle')
        my_pickle_dump(DM_new=dm, local_pickle_file=f)

    return


def simulate_industry_other_wood(refresh = True ):
    if refresh is True:
        # Path for the input from industry
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, "data/fake_ind-to-fst.xlsx")
        # Read the file:
        df_industry_calibration = pd.read_excel(f, sheet_name="ind-to-fst")
        dm_industry_calibration = DataMatrix.create_from_df(df_industry_calibration, num_cat=1)

        cntr_list = ['Switzerland','Vaud']

        lfs_interface_data_file = os.path.join(current_file_directory,
                                               '../../data/interface/industry_to_forestry.pickle')
        with open(lfs_interface_data_file, 'rb') as handle:
            dm_industry_wood_demand = pickle.load(handle)
        dm_industry_wood_demand.filter({'Country': cntr_list}, inplace=True)

        # Compute the FXA/Calibration for "any-other-wood"
        # that is any wood that is not covered by EUCalc and that is used a biomaterial (not energy)
        # Gap between EUCalc and Calibration
        ay_fxa_wood_other = dm_industry_calibration[...] -\
                            dm_industry_wood_demand[...]
        dm_industry_wood_demand.add(ay_fxa_wood_other, col_label='fxa_any-other-wood', dim='Variables', unit='t')
        dm_fxa_wood_demand = dm_industry_wood_demand
        # Sum-up the missing wood
        dm_fxa_wood_demand.filter({'Variables': ['fxa_any-other-wood']})
        dm_fxa_wood_demand.groupby({'any-other-industrial-demand': ['other-industrial', 'pulp','timber']},
                                 dim='Categories1',  inplace=True)
        dm_fxa_wood_demand.drop(dim='Variables',  col_label= 'ind_wood')

    return dm_fxa_wood_demand

def add_dummy_EU27_energy_to_forestry_interface(file):
  with open(file, 'rb') as handle:
    dm = pickle.load(handle)

  if "EU27" not in dm.col_labels['Country']:
    dm.add(np.nan, col_label='EU27', dim='Country', dummy=True)
    dm['EU27', ...] = dm['Switzerland', ...]

  my_pickle_dump(dm, file)

  return


def create_dummy_Vaud_waste(dm_forest_area, dm_wood_wastes, years_ots):
    dm_tmp = dm_forest_area.copy()
    dm_tmp.operation('Vaud', '/', 'Switzerland', out_col='Vaud-share', dim='Country')
    dm_tmp.filter({'Country': ['Vaud-share'], 'Variables': ['productive-forest-area'], 'Years': years_ots},
                  inplace=True)
    dm_tmp.rename_col('Vaud-share', 'Vaud', dim='Country')
    dm_wood_wastes_VD = dm_wood_wastes.copy()
    dm_wood_wastes_VD.rename_col('Switzerland', 'Vaud', 'Country')
    dm_wood_wastes_VD.append(dm_tmp, dim='Variables')
    dm_wood_wastes_VD.operation('fst_waste-wood', '*', 'productive-forest-area', out_col='fst_waste-wood_VD', unit='m3',
                                dim='Variables')
    dm_wood_wastes_VD.filter({'Variables': ['fst_waste-wood_VD']}, inplace=True)
    dm_wood_wastes_VD.rename_col('fst_waste-wood_VD', 'fst_waste-wood', dim='Variables')

    return dm_wood_wastes_VD

#####################################################################
# SIMULATE INTERFACE
#####################################################################
# Find excel on google drive:

simulate_industry_input()
dm_fxa_wood_demand = simulate_industry_other_wood()
# Energy to forestry pickle (add EU27)
this_dir = os.path.dirname(os.path.abspath(__file__))
lfs_interface_data_file = os.path.join(this_dir,
                                       '../../data/interface/energy_to_forestry.pickle')
add_dummy_EU27_energy_to_forestry_interface(lfs_interface_data_file)


######################################################################
######################################################################
# DATA
######################################################################
######################################################################
years_ots = create_years_list(1990, 2023, 1)
years_fts = create_years_list(2025, 2050, 5)

################################################################
# Trade Wood - FAOSTAT (Switzerland)
################################################################

file = 'data/forestry-trade.pickle'
dm = get_wood_trade_balance(file)
dm_roundwood_export = dm.filter_w_regex({'Variables':'roundwood'})
dm_roundwood_export = dm.filter_w_regex({'Categories1':'export quantity'})
df = dm_roundwood_export.write_df()
#dm_roundwood_export.datamatrix_plot({'Categories1': ['export quantity']})

################################################################
# Calibration: Energy demand per sector (Switzerland)
################################################################

file_url = 'https://www.bfe.admin.ch/bfe/en/home/versorgung/statistik-und-geodaten/energiestatistiken/teilstatistiken.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvMTE0NDA=.html'
local_filename = 'data/Swiss-wood-energy-statistics.xlsx'
dm_wood_energy_calibration = get_wood_energy_by_sector(file_url, local_filename)

################################################################
# Wood production in CH and Cantons
################################################################

# Create the data matrix out of the SwissSTAT Tab
file = "data/wood_production_ch.pickle"
table_id = "px-x-0703010000_102"
dm = get_wood_production(table_id, file)

# Renaming
dm.rename_col(
    ['Wood harvest - total', ' Sawlogs and veneer logs',' Industrial roundwood',' Wood fuel - total',' >> Chopped wood',' >> Wood chips',' Other types of wood'],
    ['total', 'sawlogs','industrial-wood','woodfuel','chopped-wood','wood-chips','any-other-wood'],
    dim='Variables')

dm.rename_col(
    ['Species - total', 'Softwood (conifers)', 'Hardwood (deciduous)'],
    ['overall','coniferous','non-coniferous'],
    dim='Categories1')

# Merge the category to match FAOSTAT classification

# Filter out the column that we do not use
dm.drop(dim='Variables', col_label='chopped-wood|wood-chips')
dm.drop(dim='Categories1', col_label='overall')

#dm.datamatrix_plot()
for col in dm.col_labels['Variables']:
    dm.rename_col(col, 'fst_production-m3_'+col, dim='Variables')
dm.deepen(based_on='Variables')
dm.switch_categories_order()
dm_harvest_rate = dm.filter_w_regex({'Categories1': 'total'})
dm_harvest_rate_all =dm_harvest_rate.groupby({'total': ['coniferous','non-coniferous']},dim='Categories2', inplace=False)
dm_harvest_rate_all.append(dm_harvest_rate, dim='Categories2')
dm.drop(dim='Categories1', col_label='total')

################################################################
# Constants - Wood density
################################################################

"""forestry_wood_density = {'pulp_coniferous': 1.4,
                         'pulp_non-coniferous': 1.25,
                         'timber_coniferous': 1.82,
                         'timber_non-coniferous': 1.43,
                         'industrial-wood_coniferous': 1.43,
                         'industrial-wood_non-coniferous': 1.25,
                         'woodfuel_coniferous': 1.43,
                         'woodfuel_non-coniferous': 1.25,
                         }"""
forestry_wood_density = {'pulp_coniferous': 1.4,
                         'pulp_non-coniferous': 1.25,
                         'sawlogs_coniferous': 1.82,
                         'sawlogs_non-coniferous': 1.43,
                         'industrial-wood_coniferous': 1.43,
                         'industrial-wood_non-coniferous': 1.25,
                         'woodfuel_coniferous': 1.43,
                         'woodfuel_non-coniferous': 1.25,
                         'any-other-wood_coniferous': 1.43,
                         'any-other-wood_non-coniferous': 1.25,
                         'timber_coniferous': 1.82,
                         'timber_non-coniferous': 1.43
                         }

cdm_wood_density = ConstantDataMatrix(col_labels={'Variables': ['fst_wood-density'],
                                              'Categories1': list(forestry_wood_density.keys())},
                                  units={'fst_wood-density': 'm3/t'})

cdm_wood_density.array = np.zeros((len(cdm_wood_density.col_labels['Variables']),
                                    len(cdm_wood_density.col_labels['Categories1'])))
idx = cdm_wood_density.idx
for key, value in forestry_wood_density.items():
    cdm_wood_density.array[0, idx[key]] = value

cdm_wood_density.deepen(based_on='Categories1')

cdm_conv = cdm_wood_density.filter({'Categories1': dm.col_labels['Categories1']})

#### Convert m3 to tonnes
arr_tonnes = dm[:, :, 'fst_production-m3', :, :] / cdm_conv[np.newaxis, np.newaxis, 'fst_wood-density', :, :]
dm.add(arr_tonnes, dim='Variables', col_label='fst_production-t', unit='t')

## Normalise
dm.normalise('Categories2',  keep_original=True)
dm_wood_production = dm
dm_wood_type = dm.filter_w_regex({'Variables': '.*_share'})

#checks
df = dm.write_df()
#dm.datamatrix_plot({'Variables': 'total_share'}, stacked=True)
#dm.datamatrix_plot()

################################################################
# Forest area per Canton
################################################################

# Create the data matrix out of the SwissSTAT Tab
file = "data/forest_area_ch.pickle"
table_id = "px-x-0703010000_101"
dm = get_forest_area(table_id, file)

# Renaming
dm.rename_col(
    ['Total forest area_Type of owners - total', 'Productive forest area_Type of owners - total'],
    ['total-forest-area', 'productive-forest-area'],
    dim='Variables')

dm.operation('productive-forest-area', '/', 'total-forest-area', out_col='productive-share', unit='%')
dm.operation('total-forest-area', '-', 'productive-forest-area', out_col='unproductive-forest-area', unit='ha')
dm.operation('unproductive-forest-area', '/', 'total-forest-area', out_col='unproductive-share', unit='%')

dm_forest_area=dm

simulate_land_input(dm_forest_area)

#dm_forest_area.datamatrix_plot()
#dm_forest_area.datamatrix_plot({'Variables': ['productive-share','unproductive-share']}, stacked=True)

################################################################
# Harvest rate
################################################################

dm_harvest_rate_all = dm_harvest_rate_all.flatten()
dm_harvest_rate_all.rename_col(
    ['total_total', 'total_coniferous', 'total_non-coniferous'],
    ['total', 'coniferous', 'non-coniferous'],
    dim='Categories1')

ay_harvest_rate = dm_harvest_rate_all[:, :, :,:] \
                             / dm_forest_area[:, :,'productive-forest-area',np.newaxis, np.newaxis]
dm_harvest_rate_all.add(ay_harvest_rate,col_label='harvest-rate',dim='Variables',unit='m3/ha')

#dm_wood_production.datamatrix_plot({'Variables': ['harvest-rate']})

#dm_forest_area.datamatrix_plot()

################################################################
# FXA: Wood fuel demand per use (Switzerland)
################################################################

file_url = 'https://www.bfe.admin.ch/bfe/en/home/versorgung/statistik-und-geodaten/energiestatistiken/teilstatistiken.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvMTE0NDA=.html'
local_filename = 'data/Swiss-wood-energy-statistics.xlsx'
clean_files = {'energy-use-m3': 'data/wood-energy-m3.csv', 'energy-use-ghw': 'data/wood-energy-ghw.csv'}
dm_wood_energy_use = get_wood_energy_by_use(file_url, local_filename, clean_local_filename=clean_files)

################################################################
# DM pre-processing Forestry
################################################################

DM_preprocessing_forestry = {'calibration': dm_wood_energy_calibration,
                             'production': dm_wood_production,
                             'wood-energy-use': dm_wood_energy_use,
                             'forest-area': dm_forest_area,
                             'wood-trade':dm_roundwood_export,
                             'any-other-industrial-wood':dm_fxa_wood_demand}

DM_preprocessing_forestry

################################################################
################################################################
# Constants
################################################################
################################################################



################################################################
# Constants - Wood energy density
################################################################

forestry_energy_density = {'average': 2.326,
                         'non-coniferous': 2.6749,
                         'coniferous': 2.50045
                         }
cdm_energy_density = ConstantDataMatrix(col_labels={'Variables': ['fst_energy-density'],
                                                    'Categories1': ['average',
                                                                    'non-coniferous',
                                                                    'coniferous']},
                                        units={'fst_energy-density': 'MWh/m3'})

cdm_energy_density.array = np.zeros((len(cdm_energy_density.col_labels['Variables']),
                                    len(cdm_energy_density.col_labels['Categories1'])))
idx = cdm_energy_density.idx
for key, value in forestry_energy_density.items():
    cdm_energy_density.array[0, idx[key]] = value

cdm_energy_density.sort('Categories1')

################################################################
# Constants - Industry yields
################################################################

forestry_industry_yields = {'wood-fuel-byproducts': 0.5,
                          'timber-to-sawlogs:':1.67,
                          'pulp-to-industrial-wood':2.5
                          }

cdm_industry_yields = ConstantDataMatrix(col_labels={'Variables': ['fst_industry-yields'],
                                                    'Categories1': ['wood-fuel-byproducts',
                                                                    'timber-to-sawlogs:',
                                                                    'pulp-to-industrial-wood']},
                                        units={'fst_industry-yields': '%'})

cdm_industry_yields.array = np.zeros((len(cdm_industry_yields.col_labels['Variables']),
                                    len(cdm_industry_yields.col_labels['Categories1'])))
idx = cdm_industry_yields.idx
for key, value in forestry_industry_yields.items():
    cdm_industry_yields.array[0, idx[key]] = value

cdm_industry_yields.sort('Categories1')

################################################################
# Constants - Conversion to match STATS categories
################################################################

forestry_conversion_industry_to_wood = {'other-industrial': 1,
                          'timber:':1.67,
                          'pulp':2.5
                          }

cdm_forestry_conversion_industry_to_wood = ConstantDataMatrix(col_labels={'Variables': ['fst_industry-to-wood-category'],
                                                    'Categories1': list(forestry_conversion_industry_to_wood.keys())},
                                        units={'fst_industry-to-wood-category': '%'})

cdm_forestry_conversion_industry_to_wood.array = np.zeros((len(cdm_forestry_conversion_industry_to_wood.col_labels['Variables']),
                                    len(cdm_forestry_conversion_industry_to_wood.col_labels['Categories1'])))
idx = cdm_forestry_conversion_industry_to_wood.idx
for key, value in forestry_conversion_industry_to_wood.items():
    cdm_forestry_conversion_industry_to_wood.array[0, idx[key]] = value

cdm_forestry_conversion_industry_to_wood.sort('Categories1')

################################################################
# Missing Values - Wood type share - Any other industrial wood
################################################################

linear_fitting(dm_wood_type,dm_wood_type.col_labels['Years'],based_on=create_years_list(2004,2023,1))

################################################################
# FXA - Wood type
################################################################

# Extract OTS & Add FTS
dm_wood_type.filter({'Years': years_ots}, inplace=True)
dm_wood_type.add(np.nan, col_label=years_fts, dim='Years', dummy=True)
# Linear extrapolation on future years
linear_fitting(dm_wood_type, years_fts)

################################################################
# FXA - Exploited forest shares
################################################################

# Extract OTS & Add FTS
dm_forest_exploited_share=dm_forest_area.filter_w_regex({'Variables': '.*-share'})
dm_forest_exploited_share.filter({'Years': years_ots}, inplace=True)
dm_forest_exploited_share.add(np.nan, col_label=years_fts, dim='Years', dummy=True)
# Linear extrapolation on future years
linear_fitting(dm_forest_exploited_share, years_fts)
#dm_forest_exploited_share.datamatrix_plot(stacked=True)

################################################################
# FXA - Exogenous wood supply from wastes incineration
################################################################

# Extract OTS & Add FTS
dm_wood_wastes =dm_wood_energy_use.flatten()
dm_wood_wastes =dm_wood_wastes.filter({'Variables': ['fst_wood-energy-use-m3_waste-incineration','fst_wood-energy-use-m3_waste-without-incineration']})
dm_wood_wastes.groupby({'fst_waste-wood': '.*'}, regex=True, inplace=True, dim='Variables')
dm_wood_wastes.add(np.nan, col_label=[2023], dim='Years', dummy=True)
dm_wood_wastes.filter({'Years': years_ots}, inplace=True)

# Add Vaud based on productive-forest-area
# Map wood-waste from Switzerland to Vaud using productive forest area as a proxy
dm_wood_wastes_VD = create_dummy_Vaud_waste(dm_forest_area, dm_wood_wastes, years_ots)
dm_wood_wastes.append(dm_wood_wastes_VD, dim='Country')

# Linear extrapolation on future years
dm_wood_wastes.add(np.nan, col_label=years_fts, dim='Years', dummy=True)
linear_fitting(dm_wood_wastes, years_fts)


#dm_wood_wastes_incineration.datamatrix_plot(stacked=True)

################################################################
# FXA - Optimal harvest rate
################################################################

# Coniferous
#values_dict ={1990: 4.7,2007:4.6,2017:5.7, 2027:4.4,2047:4.6,2056:4.6}
values_dict ={1990: 4.7,2006:4.7, 2007:4.6,2016:4.6,2017:5.7, 2026:5.7, 2027:4.4,2046:4.4, 2047:4.6,2056:4.6}
years_all = create_years_list(1990, 2056, 1)
dm_harvest_coniferous = DataMatrix( col_labels ={'Country': ['Switzerland'], 'Years': years_all,
                                   'Variables' : ['sustainable-harvest-rate_coniferous']}, units = {'sustainable-harvest-rate_coniferous': 'm3/ha'})
for yr, val in values_dict.items():
    dm_harvest_coniferous['Switzerland', yr, 'sustainable-harvest-rate_coniferous'] = val

linear_fitting(dm_harvest_coniferous, years_all)
dm_sustainable_harvest_rate = dm_harvest_coniferous
dm_sustainable_harvest_rate.deepen()

# Non-Coniferous
#values_dict ={1990: 2.7,2007:2.4,2017: 2.9, 2027:2.6,2047:3.3,2056:3.3}
values_dict ={1990: 2.7,2006:2.7,2007:2.4,2016:2.4,2017: 2.9, 2026:2.9,2027:2.6,2046:2.6,2047:3.3,2056:3.3}
years_all = create_years_list(1990, 2056, 1)
dm_harvest_non_coniferous = DataMatrix( col_labels ={'Country': ['Switzerland'], 'Years': years_all,
                                   'Variables' : ['sustainable-harvest-rate_non-coniferous']}, units = {'sustainable-harvest-rate_non-coniferous': 'm3/ha'})
for yr, val in values_dict.items():
    dm_harvest_non_coniferous['Switzerland', yr, 'sustainable-harvest-rate_non-coniferous'] = val

linear_fitting(dm_harvest_non_coniferous, years_all)
dm_harvest_non_coniferous.deepen()
dm_sustainable_harvest_rate.append(dm_harvest_non_coniferous, dim='Categories1')

# Total
dm_sustainable_harvest_rate.operation('coniferous', '+', 'non-coniferous', out_col='total', unit='m3/ha', dim='Categories1')
dm_sustainable_harvest_rate.filter({'Years': years_ots+years_fts}, inplace=True)

################################################################
# Harvest rate
################################################################

# Extract harvest-rate
dm_harvest = dm_harvest_rate_all.filter({'Variables': ['harvest-rate']})
dm_harvest.filter({'Years': years_ots}, inplace=True)
dm_harvest_clean = dm_harvest.copy()
# Remove anomalies for extrapolation
dm_harvest_clean[:, 2000, ...] = np.nan
dm_harvest_clean[:, 1990, ...] = np.nan
dm_harvest_clean.add(np.nan, col_label=years_fts, dim='Years', dummy=True)
# Linear extrapolation on future years
linear_fitting(dm_harvest_clean, years_fts)
#dm_harvest_clean.array = np.maximum(1, dm_harvest_clean.array)  # Example if you want to have at least 1 as harvest-rate
# Keep only fts years
dm_harvest_clean.filter({'Years': years_fts}, inplace=True)
# Append to original data
dm_harvest.append(dm_harvest_clean, dim='Years')

dm_fts_1 = dm_harvest.copy()
dm_fts_1.filter({'Years': years_fts}, inplace=True)

################################################################
# FTS - Harvest rate
################################################################

# Level 2 (growth)
## Coniferous
values_dict ={1990: 4.7,2006:4.7, 2007:4.6,2016:7.0,2017:6.2, 2026:6.2, 2027:4.6,2046:4.6, 2047:4.2,2056:4.2}
years_all = create_years_list(1990, 2056, 1)
dm_fts = DataMatrix( col_labels ={'Country': ['Switzerland'], 'Years': years_all,
                                   'Variables' : ['harvest-rate_coniferous']}, units = {'harvest-rate_coniferous': 'm3/ha'})
for yr, val in values_dict.items():
    dm_fts['Switzerland', yr, 'harvest-rate_coniferous'] = val

linear_fitting(dm_fts, years_all)
dm_fts_2 = dm_fts
dm_fts_2.filter({'Years': years_fts}, inplace=True)
dm_fts_2.deepen()

## Non-Coniferous
values_dict ={1990: 2.7,2006:2.7, 2007:3.6,2016:3.6,2017:3.4, 2026:3.4, 2027:2.7,2046:2.7, 2047:3.1,2056:3.1}
years_all = create_years_list(1990, 2056, 1)
dm_fts = DataMatrix( col_labels ={'Country': ['Switzerland'], 'Years': years_all,
                                   'Variables' : ['harvest-rate_non-coniferous']}, units = {'harvest-rate_non-coniferous': 'm3/ha'})
for yr, val in values_dict.items():
    dm_fts['Switzerland', yr, 'harvest-rate_non-coniferous'] = val

linear_fitting(dm_fts, years_all)
dm_fts.filter({'Years': years_fts}, inplace=True)
dm_fts.deepen()
dm_fts_2.append(dm_fts, dim='Categories1')
dm_fts_2.operation('coniferous', '+', 'non-coniferous', out_col='total', unit='m3/ha', dim='Categories1')

# Level 3 (Kyoto)
values_dict ={1990: 4.7,2006:4.7, 2007:4.1,2016:4.1,2017:4.6, 2026:4.6, 2027:4.1,2046:4.1, 2047:4.5,2056:4.5}
years_all = create_years_list(1990, 2056, 1)
dm_fts = DataMatrix( col_labels ={'Country': ['Switzerland'], 'Years': years_all,
                                   'Variables' : ['harvest-rate_coniferous']}, units = {'harvest-rate_coniferous': 'm3/ha'})
for yr, val in values_dict.items():
    dm_fts['Switzerland', yr, 'harvest-rate_coniferous'] = val

linear_fitting(dm_fts, years_all)
dm_fts_3 = dm_fts
dm_fts_3.filter({'Years': years_fts}, inplace=True)
dm_fts_3.deepen()

## Non-Coniferous
values_dict ={1990: 2.7,2006:2.7, 2007:2.2,2016:2.2,2017:2.3, 2026:2.3, 2027:2.4,2046:2.4, 2047:3.2,2056:3.2}
years_all = create_years_list(1990, 2056, 1)
dm_fts = DataMatrix( col_labels ={'Country': ['Switzerland'], 'Years': years_all,
                                   'Variables' : ['harvest-rate_non-coniferous']}, units = {'harvest-rate_non-coniferous': 'm3/ha'})
for yr, val in values_dict.items():
    dm_fts['Switzerland', yr, 'harvest-rate_non-coniferous'] = val

linear_fitting(dm_fts, years_all)
dm_fts.filter({'Years': years_fts}, inplace=True)
dm_fts.deepen()
dm_fts_3.append(dm_fts, dim='Categories1')
dm_fts_3.operation('coniferous', '+', 'non-coniferous', out_col='total', unit='m3/ha', dim='Categories1')

# Level 4 (Strong demand)
values_dict ={1990: 4.7,2006:4.7, 2007:7.3,2016:7.3,2017:7.3, 2026:7.3, 2027:4.0,2046:4.0, 2047:3.3,2056:3.3}
years_all = create_years_list(1990, 2056, 1)
dm_fts = DataMatrix( col_labels ={'Country': ['Switzerland'], 'Years': years_all,
                                   'Variables' : ['harvest-rate_coniferous']}, units = {'harvest-rate_coniferous': 'm3/ha'})
for yr, val in values_dict.items():
    dm_fts['Switzerland', yr, 'harvest-rate_coniferous'] = val

linear_fitting(dm_fts, years_all)
dm_fts_4 = dm_fts
dm_fts_4.filter({'Years': years_fts}, inplace=True)
dm_fts_4.deepen()

## Non-Coniferous
values_dict ={1990: 2.7,2006:2.7, 2007:4.4,2016:4.4,2017:4.4, 2026:4.4, 2027:2.5,2046:2.5, 2047:2.6,2056:2.6}
years_all = create_years_list(1990, 2056, 1)
dm_fts = DataMatrix( col_labels ={'Country': ['Switzerland'], 'Years': years_all,
                                   'Variables' : ['harvest-rate_non-coniferous']}, units = {'harvest-rate_non-coniferous': 'm3/ha'})
for yr, val in values_dict.items():
    dm_fts['Switzerland', yr, 'harvest-rate_non-coniferous'] = val

linear_fitting(dm_fts, years_all)
dm_fts.filter({'Years': years_fts}, inplace=True)
dm_fts.deepen()
dm_fts_4.append(dm_fts, dim='Categories1')
dm_fts_4.operation('coniferous', '+', 'non-coniferous', out_col='total', unit='m3/ha', dim='Categories1')

################################################################
# Pickle for Forestry
################################################################

DM_forestry = {'ots': dict(), 'fts': dict(), 'fxa': dict(), 'constant': dict()}
DM_forestry['constant']['energy-density'] = cdm_energy_density
DM_forestry['constant']['wood-density'] = cdm_wood_density
DM_forestry['constant']['industry-byproducts'] = cdm_industry_yields
DM_forestry['constant']['wood-category-conversion-factors'] = cdm_forestry_conversion_industry_to_wood
DM_forestry['fxa']['coniferous-share'] = dm_wood_type
DM_forestry['fxa']['any-other-industrial-wood'] = dm_fxa_wood_demand
DM_forestry['fxa']['forest-exploited-share'] = dm_forest_exploited_share
DM_forestry['fxa']['wood-waste-energy'] = dm_wood_wastes
DM_forestry

### Final DM is ots:, fts:, fxa:, (keys of ots must be lever name)

DM_forestry['ots']['harvest-rate'] = dm_harvest.filter({'Years': years_ots})
DM_forestry['fts']['harvest-rate'] = dict()
for lev in range(4):
    DM_forestry['fts']['harvest-rate'][lev+1] = dm_harvest.filter({'Years': years_fts})

DM_forestry['fts']['harvest-rate'][1] = dm_fts_1
DM_forestry['fts']['harvest-rate'][2] = dm_fts_2
DM_forestry['fts']['harvest-rate'][3] = dm_fts_3
DM_forestry['fts']['harvest-rate'][4] = dm_fts_4

add_dummy_country_to_DM(DM_forestry, ref_country='Switzerland', new_country='Vaud')
add_dummy_country_to_DM(DM_forestry, ref_country='Switzerland', new_country='EU27')


# save
f = '../../data/datamatrix/forestry.pickle'
with open(f, 'wb') as handle:
    pickle.dump(DM_forestry, handle, protocol=pickle.HIGHEST_PROTOCOL)

sort_pickle(f)
