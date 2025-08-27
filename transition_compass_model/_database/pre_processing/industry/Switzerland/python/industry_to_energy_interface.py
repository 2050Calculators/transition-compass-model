import pandas as pd

from _database.pre_processing.api_routines_CH import get_data_api_CH
from model.common.auxiliary_functions import create_years_list, linear_fitting, my_pickle_dump, sort_pickle
from model.common.data_matrix_class import DataMatrix
import pickle
import os
import numpy as np
import re
import requests
import zipfile



def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def extract_national_energy_demand(table_id, file):

    try:
        with open(file, 'rb') as handle:
            dm_energy = pickle.load(handle)
    except OSError:

        structure, title = get_data_api_CH(table_id, mode='example', language='fr')

        # Remove freight transport energy demand
        exclude_list = ['49', '50', '51', '52', '53']
        keep_sectors = [s for s in structure['Économie et ménages'] if not any(n in s for n in exclude_list)]
        # Remove household energy demand and passenger transport
        keep_sectors = [s for s in keep_sectors if 'énages' not in s]
        # Drop too detailed split
        keep_sectors = [s for s in keep_sectors if '---- ' not in s]

        filter = {'Économie et ménages': keep_sectors,
                  'Unité de mesure': ['Térajoules'],
                  'Année': structure['Année'],
                  'Agent énergétique': structure['Agent énergétique']}

        mapping = {'Country': 'Unité de mesure',
                   'Years': 'Année',
                   'Variables': 'Économie et ménages',
                   'Categories1': 'Agent énergétique'}

        dm_energy = get_data_api_CH(table_id, mode='extract', mapping_dims=mapping, filter=filter,
                                    units=['TJ']*len(keep_sectors), language='fr')

        #dm_heating.rename_col('--- Chauffage des ménages', 'bld_heating-demand', dim='Variables')
        dm_energy.rename_col('Térajoules', 'Switzerland', dim='Country')

        # We drop the fuels fro transport
        dict_rename = {'heating-oil': ['1.1.2. Huile de chauffage extra-légère'],
                       'coal': ['1.2. Charbon'], 'gas': ['1.3. Gaz naturel'],
                       'district-heating': ['6. Chaleur à distance'],
                       'nuclear-fuel': ['4. Combustibles nucléaires'], 'waste': ['2. Déchets (hors biomasse)'],
                       'biomass': ['3.1. Déchets (biomasse)'],
                       'wood': ['3.2. Bois et charbon de bois'], 'biogas': ['3.3. Biogaz et biocarburants'],
                       'renewables': ['3.4. Géothermie, chaleur ambiante et énergie solaire thermique'],
                       'electricity': ['5. Electricité']}

        dm_energy = dm_energy.groupby(dict_rename, dim='Categories1', inplace=False)

        for var in dm_energy.col_labels['Variables']:
            dm_energy.rename_col(var, 'bld_energy-by-sector_'+ var, dim='Variables')

        dm_energy.deepen(based_on='Variables')
        dm_energy.switch_categories_order()

        dm_energy.change_unit('bld_energy-by-sector', 3600, old_unit='TJ', new_unit='TWh', operator='/')

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, 'wb') as handle:
            pickle.dump(dm_energy, handle, protocol=pickle.HIGHEST_PROTOCOL)
    dm_energy.rename_col_regex('-', '', dim='Categories1')
    dm_energy.drop('Categories1', " 35 Production et distribution d'énergie")
    # Remove demand of waste for waste management sector because it is computed in EnergyScope
    # Biomass is also from waste
    dm_energy[:, :, :, " 3639 Production et distribution d'eau; gestion des déchets", 'waste'] = 0
    dm_energy[:, :, :, " 3639 Production et distribution d'eau; gestion des déchets", 'biomass'] = 0
    return dm_energy


def extract_employees_per_sector_canton(table_id, file):

    try:
        with open(file, 'rb') as handle:
            dm_employees = pickle.load(handle)
    except OSError:

        structure, title = get_data_api_CH(table_id, mode='example', language='fr')

        filter = {'Division économique': structure['Division économique'],
                  "Unité d'observation": ['Equivalents plein temps'],
                  'Année': structure['Année'],
                  'Canton': structure['Canton']}

        mapping = {'Country': 'Canton',
                   'Years': 'Année',
                   'Variables': "Unité d'observation",
                   'Categories1': 'Division économique'}

        dm_employees = get_data_api_CH(table_id, mode='extract', mapping_dims=mapping, filter=filter,
                                       units=['EPT'], language='fr')

        #dm_heating.rename_col('--- Chauffage des ménages', 'bld_heating-demand', dim='Variables')
        dm_employees.rename_col('Equivalents plein temps', 'ind_employees', dim='Variables')
        dm_employees.drop(col_label='Division économique - total', dim='Categories1')

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, 'wb') as handle:
            pickle.dump(dm_employees, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm_employees


def extract_service_surface_area_by_canton(table_id, file):
    try:
        with open(file, 'rb') as handle:
            dm_service_area = pickle.load(handle)
    except OSError:

        structure, title = get_data_api_CH(table_id, mode='example', language='fr')

        cantons_list = [c for c in structure['Grande région (<<) / Canton (-)'] if '- ' in c]
        categories_list = ["<->a.1.1 Aires industrielles et artisanales",
                           "<->a.2.3 Aires de bâtiments publics",
                           "<->a.2.5 Aires de bâtiments non déterminés"]

        filter = {'Grande région (<<) / Canton (-)': cantons_list + ['Suisse'],
                  "Période": structure['Période'],
                  'Nomenclature standard (NOAS04)': categories_list}

        mapping = {'Country': 'Grande région (<<) / Canton (-)',
                   'Years': 'Période',
                   'Variables': 'Nomenclature standard (NOAS04)'}

        dm_service_area = get_data_api_CH(table_id, mode='extract', mapping_dims=mapping, filter=filter,
                                          units=['ha']* len(categories_list), language='fr')

        dm_service_area.groupby({'bld_service-floor-area': '.*'}, dim='Variables',  regex=True, inplace=True)

        dm_service_area.array[dm_service_area.array == 0] = np.nan
        dm_service_area.rename_col_regex('- ', '', dim='Country')

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, 'wb') as handle:
            pickle.dump(dm_service_area, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm_service_area


def map_national_energy_demand_by_sector_to_cantons(dm_energy, dm_employees):
    mapping_dict = {}
    mapping_sectors = {'agriculture': [], 'industry-w-process-heat': [], 'services': [], 'industry-wo-process-heat': []}
    for cat in dm_energy.col_labels['Categories1']:
        if has_numbers(cat):
            cat_num = re.findall(r'\d+', cat)[0]
            if len(cat_num) == 2:
                matching_cat = [c for c in dm_employees.col_labels['Categories1'] if cat_num in c]
                mapping_dict[cat] = matching_cat
                if int(cat_num) <= 3:
                    mapping_sectors['agriculture'].append(cat)
                elif 3 < int(cat_num) < 41:
                    mapping_sectors['industry-w-process-heat'].append(cat)
                elif 41 <= int(cat_num) < 45:
                    mapping_sectors['industry-wo-process-heat'].append(cat)
                elif int(cat_num) >= 45:
                    mapping_sectors['services'].append(cat)
            elif len(cat_num) == 4:
                first_num = cat_num[0:2]
                second_num = cat_num[2:4]
                matching_cat = []
                for i in range(int(first_num), int(second_num) + 1):
                    str_i = f"{i:02}"  # pad with zeros
                    matching_cat_i = [c for c in dm_employees.col_labels['Categories1'] if str_i in c]
                    matching_cat.append(matching_cat_i[0])
                if int(first_num) <= 3:
                    mapping_sectors['agriculture'].append(cat)
                elif 3 < int(first_num) < 41:
                    mapping_sectors['industry-w-process-heat'].append(cat)
                elif 41 <= int(first_num) < 45:
                    mapping_sectors['industry-wo-process-heat'].append(cat)
                elif int(first_num) >= 45:
                    mapping_sectors['services'].append(cat)
                mapping_dict[cat] = matching_cat

    dm_employees_mapped = dm_employees.groupby(mapping_dict, dim='Categories1', inplace=False)
    dm_employees_mapped.drop('Country', 'Suisse')
    dm_employees_mapped.sort('Categories1')

    dm_energy_mapped = dm_energy.filter({'Categories1': dm_employees_mapped.col_labels['Categories1']}, inplace=False)
    dm_energy_mapped.sort('Categories1')
    dm_employees_mapped.normalise(dim='Country')
    linear_fitting(dm_employees_mapped, years_ots=dm_energy.col_labels['Years'], min_t0=0)
    dm_employees_mapped.normalise(dim='Country')

    new_arr = dm_employees_mapped.array[:, :, :, :, np.newaxis] * dm_energy_mapped.array[:, :, :, :, :]
    dm_energy_mapped.add(new_arr, dim='Country', col_label=dm_employees_mapped.col_labels['Country'])

    dm_energy_mapped.groupby(mapping_sectors, dim='Categories1', inplace=True)
    dm_employees_mapped.groupby(mapping_sectors, dim='Categories1', inplace=True)
    dm_employees_mapped.add(np.nan, dummy=True, dim='Years',
                            col_label=list(set(years_ots) - set(dm_employees_mapped.col_labels['Years'])))
    dm_employees_mapped.sort('Years')
    dm_employees_mapped.fill_nans('Years')

    return dm_energy_mapped, dm_employees_mapped


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


def clean_df_EP2050(df, keep_list):

    # Set the new header
    df.drop(columns=['zurück', 'Unnamed: 2'], inplace=True)
    df.columns = df.iloc[10]
    df = df[11:31].copy()

    def is_valid_number(val):
        return isinstance(val, (int, float)) and not pd.isna(val)

    # Apply the function to filter out rows with no valid numeric values
    df = df[df.apply(lambda row: row.map(is_valid_number).any(), axis=1)]

    df_clean = df.loc[df['Verwendungszweck'].isin(keep_list)]
    df_pivot = df_clean.T.reset_index()
    df_pivot.columns = df_pivot.iloc[0]
    df_pivot = df_pivot[1:]  # Drop the first row
    df_pivot.reset_index(drop=True, inplace=True)
    df_pivot.rename(columns={'Verwendungszweck': 'Years'}, inplace=True)
    df_pivot.Years = df_pivot.Years.astype(int)
    df_pivot.set_index('Years', inplace=True)
    return df_pivot

def extract_EP2050_industry_data(file_url, zip_name, keep_years):

    extract_dir = os.path.splitext(zip_name)[0]  # 'data/EP2050_sectors'
    if not os.path.exists(extract_dir):
        save_url_to_file(file_url, zip_name)

        # Extract the file
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(local_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

    file_industry = extract_dir + '/EP2050+_Szenarienergebnisse_Details_Nachfragesektoren/EP2050+_Detailergebnisse 2020-2060_Industriessektor_alle Szenarien_2022-04-12.xlsx'
    df = pd.read_excel(file_industry, sheet_name='02 Industrie WWB')

    keep_list = ['Raumwärme', 'Warmwasser', 'Prozesswärme', 'Beleuchtung', 'Klima, Lüftung und Haustechnik',
                 'Antriebe, Prozesse', 'Sonstige Verwendungszwecke']

    df_clean = clean_df_EP2050(df, keep_list)
    mapping = {'space-heating': ['Raumwärme'], 'hot-water': ['Warmwasser'], 'process-heat': ['Prozesswärme', 'Sonstige Verwendungszwecke'],
               'lighting': ['Beleuchtung'],
               'elec': ['Klima, Lüftung und Haustechnik', 'Antriebe, Prozesse']}

    new_cols = []
    for col in list(df_clean.columns):
        new_cols.append('ind_energy-end-use_' + col + '[PJ]')
    df_clean.columns = new_cols
    df_clean.reset_index(inplace=True)
    df_clean['Country'] = 'Switzerland'
    dm = DataMatrix.create_from_df(df_clean, num_cat=1)

    dm.groupby(mapping, dim='Categories1', inplace=True)
    dm.filter({'Years': keep_years}, inplace=True)

    return dm


def extract_EP2050_industry_data(file_url, zip_name, keep_years):

    extract_dir = os.path.splitext(zip_name)[0]  # 'data/EP2050_sectors'
    if not os.path.exists(extract_dir):
        save_url_to_file(file_url, zip_name)

        # Extract the file
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(local_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

    file_industry = extract_dir + '/EP2050+_Szenarienergebnisse_Details_Nachfragesektoren/EP2050+_Detailergebnisse 2020-2060_Industriessektor_alle Szenarien_2022-04-12.xlsx'
    df = pd.read_excel(file_industry, sheet_name='02 Industrie WWB')

    keep_list = ['Raumwärme', 'Warmwasser', 'Prozesswärme', 'Beleuchtung', 'Klima, Lüftung und Haustechnik',
                 'Antriebe, Prozesse', 'Sonstige Verwendungszwecke']

    df_clean = clean_df_EP2050(df, keep_list)
    mapping = {'space-heating': ['Raumwärme'], 'hot-water': ['Warmwasser'], 'process-heat': ['Prozesswärme', 'Sonstige Verwendungszwecke'],
               'lighting': ['Beleuchtung'],
               'elec': ['Klima, Lüftung und Haustechnik', 'Antriebe, Prozesse']}

    new_cols = []
    for col in list(df_clean.columns):
        new_cols.append('ind_energy-end-use_' + col + '[PJ]')
    df_clean.columns = new_cols
    df_clean.reset_index(inplace=True)
    df_clean['Country'] = 'Switzerland'
    dm = DataMatrix.create_from_df(df_clean, num_cat=1)

    dm.groupby(mapping, dim='Categories1', inplace=True)
    dm.filter({'Years': keep_years}, inplace=True)

    return dm


def distribute_fuels_by_end_use(dm_cantons_ind_wo_process, dm_end_use_others):
    # Electricity first
    dm_cantons_elec = dm_cantons_ind_wo_process.filter({'Categories2': ['electricity']})
    dm_light_elec_end_use = dm_end_use_others.filter({'Categories1': ['lighting', 'elec']})
    dm_light_elec_end_use.normalise('Categories1', inplace=True)
    arr = dm_cantons_elec[:, :, 'bld_energy-by-sector', 'industry-wo-process-heat', np.newaxis, :, np.newaxis]\
          * dm_light_elec_end_use[0, np.newaxis, :, 'ind_energy-end-use', np.newaxis, np.newaxis, :]
    dm_cantons_end_use_tmp = DataMatrix.based_on(arr[:, :, np.newaxis, ...], format=dm_cantons_elec,
                                                 change={'Categories3': dm_light_elec_end_use.col_labels['Categories1'],
                                                         'Variables': ['ind_energy-end-use']},
                                                 units={'ind_energy-end-use': 'TWh'})

    # Distribute all other fuels by end-use
    dm_cantons_ind_wo_process.drop('Categories2', 'electricity')
    dm_end_use_not_elec = dm_end_use_others.filter({'Categories1': ['hot-water', 'space-heating']})
    dm_end_use_not_elec.normalise('Categories1', inplace=True)
    arr = dm_cantons_ind_wo_process[:, :, 'bld_energy-by-sector', 'industry-wo-process-heat', np.newaxis, :, np.newaxis]\
          * dm_end_use_not_elec[0, np.newaxis, :, 'ind_energy-end-use', np.newaxis, np.newaxis, :]
    dm_cantons_end_use = DataMatrix.based_on(arr[:, :, np.newaxis, ...], format=dm_cantons_ind_wo_process,
                                             change={'Categories3': dm_end_use_not_elec.col_labels['Categories1'],
                                                     'Variables': ['ind_energy-end-use']},
                                             units={'ind_energy-end-use': 'TWh'})

    fuels_list = dm_cantons_end_use.col_labels['Categories2']
    dm_cantons_end_use_tmp.add(0, dummy=True, col_label=fuels_list, dim='Categories2')
    dm_cantons_end_use.add(0, dummy=True, col_label='electricity', dim='Categories2')
    dm_cantons_end_use.append(dm_cantons_end_use_tmp, dim='Categories3')
    return dm_cantons_end_use


def create_industry_energy_demand_by_canton_fuel_enduse(dm_c_s_f, dm_end_use):
    # dm_c_f_s contains energy demand by canton, by sector and by fuel type
    # dm_ind_end_use contains information on the final end-use of energy
    # (lighting,electricity,hotwater,space-heating,process-heating) for the industrial sector
    # Here we are going to focus on industry only
    dm_end_use.filter({'Years': dm_c_s_f.col_labels['Years']}, inplace=True)
    dm_cantons_ind_wo_process = dm_c_s_f.filter({'Categories1': ['industry-wo-process-heat']})
    dm_end_use_process = dm_end_use.filter({'Categories1': ['process-heat']})
    dm_end_use_others = dm_end_use.copy()
    dm_end_use_others.drop('Categories1', 'process-heat')

    # For OTHERS (non process heat)
    dm_cantons_end_use_wo_process = distribute_fuels_by_end_use(dm_cantons_ind_wo_process, dm_end_use_others)

    # Compute Space-Heating and Hot-water demand by fuel in industry-w-process-heat
    # using industry-wo-process-heat as a proxy
    # Ewop_end-use -> Ewp_end-use
    # I could also then have the split of hot-water and space-heating by fuel


    return


def extract_heating_technologies(table_id, file):
    # Domaine de l'énergie: bâtiments selon le canton, le type de bâtiment, l'époque de construction, le type de chauffage,
    # la production d'eau chaude, les agents énergétiques utilisés pour le chauffage et l'eau chaude, 1990 et 2000
    try:
        with open(file, 'rb') as handle:
            dm_heating = pickle.load(handle)
    except OSError:
        structure, title = get_data_api_CH(table_id, mode='example', language='fr')
        # Extract buildings floor area
        filter = structure.copy()
        filter['Catégorie de bâtiment'] = ["Bâtiments partiellement à usage d'habitation", "Bâtiments d'habitation avec usage annexe"]
        mapping_dim = {'Country': 'Canton', 'Years': 'Année',
                       'Variables': 'Catégorie de bâtiment',
                       'Categories1': "Source d'énergie du chauffage"}
        unit_all = ['number'] * len(structure['Catégorie de bâtiment'])
        dm_heating = None
        tot_bld = 0
        for a in structure["Source d'énergie de l'eau chaude"]:
            filter["Source d'énergie de l'eau chaude"] = [a]
            dm_heating_t = get_data_api_CH(table_id, mode='extract', filter=filter,
                                           mapping_dims=mapping_dim, units=unit_all, language='fr')
            if dm_heating is None:
                dm_heating = dm_heating_t.copy()
            else:
                dm_heating.array += dm_heating_t.array

        dm_heating.rename_col(['Suisse'], ['Switzerland'], dim='Country')
        dm_heating.groupby({'bld_households_space-heating': '.*'}, regex=True,
                           dim='Variables', inplace=True)

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, 'wb') as handle:
            pickle.dump(dm_heating, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dm_heating.groupby({'other': ['Autre', 'Aucune']}, dim='Categories1', inplace=True)
    dm_heating.rename_col(['Mazout', 'Bois', 'Pompe à chaleur', 'Electricité', 'Gaz', 'Chaleur produite à distance',
                           'Soleil (thermique)'],
                          ['heating-oil', 'wood', 'heat-pump', 'electricity', 'gas', 'district-heating', 'solar'],
                          dim='Categories1')
    #dm_heating.normalise('Categories1', inplace=True)
    return dm_heating


def extract_hotwater_technologies(table_id, file):
    # Domaine de l'énergie: bâtiments selon le canton, le type de bâtiment, l'époque de construction, le type de chauffage,
    # la production d'eau chaude, les agents énergétiques utilisés pour le chauffage et l'eau chaude, 1990 et 2000
    try:
        with open(file, 'rb') as handle:
            dm_hw = pickle.load(handle)
    except OSError:
        structure, title = get_data_api_CH(table_id, mode='example', language='fr')
        # Extract buildings floor area
        filter = structure.copy()
        filter['Catégorie de bâtiment'] = ["Bâtiments partiellement à usage d'habitation", "Bâtiments d'habitation avec usage annexe"]
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
        dm_hw.groupby({'bld_households_hot-water': '.*'}, regex=True,
                      dim='Variables', inplace=True)

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, 'wb') as handle:
            pickle.dump(dm_hw, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dm_hw.groupby({'other': ['Autre', 'Aucune']}, dim='Categories1', inplace=True)
    dm_hw.rename_col(['Mazout', 'Bois', 'Pompe à chaleur', 'Electricité', 'Gaz', 'Chaleur produite à distance',
                           'Soleil (thermique)'],
                          ['heating-oil', 'wood', 'heat-pump', 'electricity', 'gas', 'district-heating', 'solar'],
                          dim='Categories1')
    #dm_hw.normalise('Categories1', inplace=True)
    return dm_hw


def map_eud_by_canton(dm_employees_mapped, dm_industry_energy_end_use):
    # First map all end-use except process heat
    dm_employees_ind = dm_employees_mapped.groupby({'industry': ['industry-w-process-heat', 'industry-wo-process-heat']},
                                                   inplace=False, dim='Categories1')
    dm_employees_ind.normalise('Country')
    arr = dm_industry_energy_end_use['Switzerland', np.newaxis, :, 'ind_energy-end-use', :] * dm_employees_ind[:, :, 'ind_employees', 'industry', np.newaxis]
    dm_industry_eud_canton = DataMatrix.based_on(arr[:, :, np.newaxis, ...], format=dm_industry_energy_end_use,
                                                 change={'Country': dm_employees_ind.col_labels['Country']},
                                                 units={'ind_energy-end-use': 'TWh'})
    # Drop process heat
    dm_industry_eud_canton.drop('Categories1', 'process-heat')
    # Secondly map process heat
    """dm_employees_process = dm_employees_mapped.filter({'Categories1': ['industry-w-process-heat']})
    dm_employees_process.normalise('Country')
    arr = dm_industry_energy_end_use['Switzerland', np.newaxis, :, 'ind_energy-end-use', :] \
          * dm_employees_process[:, :, 'ind_employees', 'industry-w-process-heat', np.newaxis]
    dm_industry_process_canton = DataMatrix.based_on(arr[:, :, np.newaxis, ...], format=dm_industry_energy_end_use,
                                                     change={'Country': dm_employees_process.col_labels['Country']},
                                                     units={'ind_energy-end-use': 'TWh'})
    dm_industry_process_canton.filter({'Categories1': ['process-heat']}, inplace=True)

    dm_industry_eud_canton.append(dm_industry_process_canton, dim='Categories1')"""

    return dm_industry_eud_canton


def map_services_eud_by_canton(dm_employees_mapped, dm_services_end_use):
    # First map all end-use except process heat
    dm_employees = dm_employees_mapped.filter({'Categories1': ['services']}, inplace=False)
    dm_employees.normalise('Country')
    arr = (dm_services_end_use['Switzerland', np.newaxis, :, 'enr_services-energy-eud', :]
           * dm_employees[:, :, 'ind_employees', 'services', np.newaxis])
    dm = DataMatrix.based_on(arr[:, :, np.newaxis, ...], format=dm_services_end_use,
                                                 change={'Country': dm_employees.col_labels['Country']},
                                                 units={'srv_energy-end-use': 'TWh'})

    return dm



def split_fuel_demand_by_eud(dm_water, dm_space_heat, dm_industry_eud_canton, cantonal = True):

    dm_fuel_split = dm_water.copy()
    dm_fuel_split.append(dm_space_heat, dim='Categories1')

    # Electricity and Lighting
    dm_elec = dm_industry_eud_canton.filter({'Categories1': ['elec', 'lighting']})
    var = dm_elec.col_labels['Variables'][0]
    dm_elec.rename_col(var, var + '_electricity', dim='Variables')
    dm_elec.deepen(based_on='Variables')

    # Space-heat and Hot water
    if cantonal:
        dm_fuel_split.drop('Country', 'Switzerland')
    dm_hw_sp = dm_industry_eud_canton.filter({'Categories1': ['hot-water', 'space-heating']})
    var = dm_hw_sp.col_labels['Variables'][0]
    if dm_fuel_split.col_labels['Country'] == dm_hw_sp.col_labels['Country']:
        arr = dm_hw_sp[:, :, var, :, np.newaxis] \
              * dm_fuel_split[:, :, 'bld_households', :, :]
        dm_fuel_split.add(arr, dim='Variables', col_label=var, unit='TWh')
    else:
        raise ValueError("dm_water and dm_industry_eud_canton don't have the same country list")

    dm_fuels_eud_cantons = dm_elec.copy()
    dummy_cat = list(set(dm_fuel_split.col_labels['Categories2']) - set(dm_fuels_eud_cantons.col_labels['Categories2']))
    dm_fuels_eud_cantons.add(0, dim='Categories2', col_label=dummy_cat, dummy=True)
    dm_fuels_eud_cantons.append(dm_fuel_split.filter({'Variables': [var]}), dim='Categories1')

    return dm_fuels_eud_cantons


def add_process_heat_demand(dm_fuels_eud_cantons, dm_fuels_cantons, cantonal=True):
    dm_ind_fuels = dm_fuels_cantons.groupby({'total': 'industry.*'}, regex=True, inplace=False, dim='Categories1')
    dm_ind_fuels.rename_col('bld_energy-by-sector', 'ind_energy-end-use', dim='Variables')
    missing_fuels = list(set(dm_ind_fuels.col_labels['Categories2']) - set(dm_fuels_eud_cantons.col_labels['Categories2']))
    dm_fuels_eud_cantons.add(0, dim='Categories2', dummy=True, col_label=missing_fuels)
    dm_fuels_eud_cantons.sort('Categories2')
    missing_fuels = list(set(dm_fuels_eud_cantons.col_labels['Categories2']) - set(dm_ind_fuels.col_labels['Categories2']))
    dm_ind_fuels.add(0, dim='Categories2', dummy=True, col_label=missing_fuels)
    dm_ind_fuels.sort('Categories2')
    dm_non_proc = dm_fuels_eud_cantons.groupby({'non-process': '.*'}, dim='Categories1', inplace=False, regex=True)
    if cantonal:
        dm_ind_fuels.drop('Country', 'Switzerland')
    dm_ind_fuels.add(np.nan, dim='Years', col_label=list(set(years_ots) - set(dm_ind_fuels.col_labels['Years'])), dummy=True)
    dm_ind_fuels.fill_nans('Years')
    dm_ind_fuels.append(dm_non_proc, dim='Categories1')
    dm_ind_fuels.operation('total', '-', 'non-process', dim='Categories1', out_col='process-heat')
    dm_fuels_eud_cantons.append(dm_ind_fuels.filter({'Categories1': ['process-heat']}), dim='Categories1')
    return dm_fuels_eud_cantons


def adjust_based_on_efficiency(dm, dm_eff, years_ots):
    dm_eff.filter({'Categories1': dm.col_labels['Categories1'], 'Years': years_ots}, inplace=True)
    dm_eff.sort('Categories1')
    dm.sort('Categories1')
    var_name = dm.col_labels['Variables'][0]
    if dm_eff.col_labels['Categories1'] == dm.col_labels['Categories1']:
        arr = dm[:, :, 0, :] / dm_eff[:, :, 'bld_efficiency', :]
        dm.add(arr, dim='Variables', col_label='bld_adj', unit='number')
        dm.filter({'Variables': ['bld_adj']}, inplace=True)
        dm.normalise('Categories1')
        dm.rename_col('bld_adj', var_name, dim='Variables')
    else:
        raise ValueError('dm_eff does not have the same fuel categories')
    return dm


def fix_negative_values(dm_fuels_eud_cantons):
    err = np.minimum(0, dm_fuels_eud_cantons.array[...])
    err[np.isnan(err)] = 0
    dm_fuels_eud_cantons.array[...] = np.maximum(0, dm_fuels_eud_cantons.array[...])
    dm_fuels_eud_cantons.fill_nans('Years')
    mask = np.isnan(dm_fuels_eud_cantons.array)
    dm_fuels_eud_cantons.array[mask] = 0
    return dm_fuels_eud_cantons


def compute_ind_energy_eud_fuels(dm_energy, dm_industry_energy_end_use, dm_water_CH, dm_space_heat_CH):
    # Group all industrial energy demand
    dm_ind_fuels = dm_energy.filter({'Categories1': [' 0509 Industries extractives',
                                                     ' 1033 Industrie manufacturière',
                                                     " 3639 Production et distribution d'eau; gestion des déchets",
                                                     ' 4143 Construction']})
    dm_ind_fuels.groupby({'industry': '.*'}, regex=True, dim='Categories1', inplace=True)

    dm = split_fuel_demand_by_eud(dm_water_CH, dm_space_heat_CH, dm_industry_energy_end_use, cantonal = False)
    dm_out = add_process_heat_demand(dm, dm_ind_fuels, cantonal=False)

    return dm_out


def load_services_energy_demand_eud(dict_services, years_ots):
    dict_services.pop('Total')
    dm = DataMatrix(col_labels={'Country': ['Switzerland'], 'Years': years_ots,
                                'Variables': ['enr_services-energy-eud'],
                                'Categories1': list(dict_services.keys())}, units={'enr_services-energy-eud': 'PJ'})
    for key, years_values in dict_services.items():
        for year, value in years_values.items():
            dm[:, year, :, key] = value

    dm.fill_nans('Years')
    dm.groupby({'elec': ['ICT and entertainment media', 'HVAC and building tech', 'Drives and processes']},
               inplace=True, dim='Categories1')
    dm.drop(dim='Categories1', col_label=['Other', 'process-heat'])

    dm.change_unit('enr_services-energy-eud', 3.6, 'PJ', 'TWh', operator='/')
    return dm


def adjust_based_on_FSO_energy_consumption(dm_fuels_cantons, dm_services_fuels_eud_cantons):
    dm_eud_shares = dm_services_fuels_eud_cantons.normalise('Categories1', inplace=False)
    dm_OFS_fuels = dm_fuels_cantons.groupby({'total': 'services'}, regex=True, inplace=False, dim='Categories1')
    dm_OFS_fuels.rename_col('bld_energy-by-sector', 'srv_energy-end-use', dim='Variables')
    missing_fuels = list(set(dm_OFS_fuels.col_labels['Categories2']) - set(dm_eud_shares.col_labels['Categories2']))
    dm_eud_shares.add(0, dim='Categories2', dummy=True, col_label=missing_fuels)
    # Attribute the missing fuels to space-heat
    for fuel in missing_fuels:
        dm_eud_shares[:, :, :, 'space-heating', fuel] = 1
    dm_eud_shares.sort('Categories2')
    missing_fuels_2 = list(set(dm_eud_shares.col_labels['Categories2']) - set(dm_OFS_fuels.col_labels['Categories2']))
    dm_OFS_fuels.add(np.nan, dim='Categories2', dummy=True, col_label=missing_fuels_2)
    dm_OFS_fuels.sort('Categories2')

    dm_OFS_fuels.filter({'Country': dm_eud_shares.col_labels['Country']}, inplace=True)

    # Add missing years
    missing_years = list(set(years_ots) - set(dm_OFS_fuels.col_labels['Years']))
    dm_OFS_fuels.add(np.nan, dim='Years', col_label=missing_years, dummy=True)
    dm_OFS_fuels.fill_nans('Years')
    dm_OFS_fuels.sort('Years')

    arr = dm_OFS_fuels[:, :, 'srv_energy-end-use', 'total', np.newaxis, :] * dm_eud_shares[:, :,
                                                                             'enr_services-energy-eud_share', :, :]

    dm_eud_shares.add(arr, dim='Variables', col_label='enr_services-energy-eud', unit='TWh')
    dm_eud_shares.filter({'Variables': ['enr_services-energy-eud']}, inplace=True)

    for fuel in missing_fuels_2:
        dm_eud_shares[:, :, :, :, fuel] = dm_services_fuels_eud_cantons[:, :, :, :, fuel]
    return dm_eud_shares


#---------------------------------------------------------------------------------
years_ots = create_years_list(1990, 2023, 1)
years_fts = create_years_list(2025, 2050, 5)

######################################
##   INDUSTRY TO ENERGY INTERFACE   ##
######################################

# Extract energy demand by sector at national level by fuel
table_id = 'px-x-0204000000_106'
local_filename = 'data/energy_accounts_economy_households.pickle'
# Industry sectors linked to energy, like energy production and waste management, have been removed or edited
# Remove gasoline and diesel which are for transport / machinery
dm_energy = extract_national_energy_demand(table_id, local_filename)


# Industry energy demand by end-use (lighting, electricity, space-heating, process-heat, hot-water)
# Energy Perspective 2050 data
file_url = 'https://www.bfe.admin.ch/bfe/de/home/politik/energieperspektiven-2050-plus.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvMTA0NDE=.html'
zip_name = 'data/EP2050_sectors.zip'
dm_industry_energy_end_use = extract_EP2050_industry_data(file_url, zip_name, years_ots+years_fts)
dm_industry_energy_end_use.change_unit('ind_energy-end-use', old_unit='PJ', new_unit='TWh', factor=3.6, operator='/')
dm_industry_energy_end_use_fts = dm_industry_energy_end_use.filter({'Years': years_fts})
dm_industry_energy_end_use.filter({'Years': years_ots}, inplace=True)

# Extract number of employees per industry and service sector by canton
# This is in order to map the national energy demand to cantons
table_id = 'px-x-0602010000_101'
local_filename = 'data/employees_per_sector_canton.pickle'
dm_employees = extract_employees_per_sector_canton(table_id, local_filename)

# Group employees by sector and canton (dm_employees_mapped)
dm_fuels_cantons, dm_employees_mapped = map_national_energy_demand_by_sector_to_cantons(dm_energy, dm_employees)

# Distribute energy by end-use by canton based on number of employees (except for process-heat)
dm_industry_eud_canton = map_eud_by_canton(dm_employees_mapped, dm_industry_energy_end_use)


# Get Hot water and Space heating fuel split at household level per canton
# !FIXME extract also 1990 and 2000
table_id = 'px-x-0902010000_102'
file_sh = 'data/bld_heating_technology_2021-2023.pickle'
dm_space_heat = extract_heating_technologies(table_id, file_sh)
# Add missing years
dm_space_heat.add(np.nan, dummy=True, dim='Years', col_label=list(set(years_ots)-set(dm_space_heat.col_labels['Years'])))
dm_space_heat.sort('Years')
dm_space_heat.fill_nans('Years')


file_hw = 'data/bld_hotwater_technology_2021-2023.pickle'
dm_water = extract_hotwater_technologies(table_id, file_hw)
# Add missing years
dm_water.add(np.nan, dummy=True, dim='Years', col_label=list(set(years_ots)-set(dm_water.col_labels['Years'])))
dm_water.sort('Years')
dm_water.fill_nans('Years')

# Add efficiencies
#data_file = "../../../data/datamatrix/buildings.pickle"
#with open(data_file, 'rb') as handle:
#    DM_bld = pickle.load(handle)

# Add efficiencies
data_file = "../../../data/interface/buildings_to_energy.pickle"
with open(data_file, 'rb') as handle:
    dm_eff = pickle.load(handle)
dm_eff.filter({'Country': ['Switzerland']}, inplace=True)
dm_eff.operation('bld_heating', '/', 'bld_energy-demand_heating', out_col='bld_efficiency', unit='%')
dm_eff.filter({'Variables': ['bld_efficiency']}, inplace=True)
dm_eff.fill_nans('Years')
dm_eff[:, :, 'bld_efficiency', 'electricity'] = 1.0

# It helps going from number of buildings to an estimate of the demand by fuel
dm_space_heat = adjust_based_on_efficiency(dm_space_heat, dm_eff, years_ots)
dm_water = adjust_based_on_efficiency(dm_water, dm_eff, years_ots)

dm_water.deepen(based_on='Variables')
dm_water.switch_categories_order()
dm_space_heat.deepen(based_on='Variables')
dm_space_heat.switch_categories_order()

# Filter for CH
dm_water_CH = dm_water.filter({'Country': ['Switzerland']})
dm_space_heat_CH = dm_space_heat.filter({'Country': ['Switzerland']})

dm_fuels_eud_cantons = split_fuel_demand_by_eud(dm_water, dm_space_heat, dm_industry_eud_canton)

# !FIXME heat-pump and electricity are off, use COP to normalise - Also compute Switzerland
dm_fuels_eud_cantons = add_process_heat_demand(dm_fuels_eud_cantons, dm_fuels_cantons)

dm_fuels_eud_cantons = fix_negative_values(dm_fuels_eud_cantons)

## Compute for all of CH
dm_fuels_eud_CH = compute_ind_energy_eud_fuels(dm_energy, dm_industry_energy_end_use, dm_water_CH, dm_space_heat_CH)


dm_fuels_eud_cantons.append(dm_fuels_eud_CH, dim='Country')

#####################################
##   SERVICES TO ENERGY INTERFACE  ##
#####################################

# Infras, TEP, Prognos, 2021. Analyse des schweizerischen Energieverbrauchs 2000–2020 - Auswertung nach Verwendungszwecken.
# Table 26 - Endenergieverbrauch im Dienstleistungssektor nach Verwendungszwecken Entwicklung von 2000 bis 2020, in PJ, inkl. Landwirtschaft
# Final energy consumption in the service sector by purpose Development from 2000 to 2020, in PJ, incl. agriculture
# https://www.bfe.admin.ch/bfe/de/home/versorgung/statistik-und-geodaten/energiestatistiken/energieverbrauch-nach-verwendungszweck.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvMTA2OTM%3D.html&ved=2ahUKEwiC4OjJvpGOAxWexgIHHdyFGVMQFnoECB0QAQ&usg=AOvVaw1a9deGMbwSdNvV0aVLEBPj
services_agri_split = {
    "space-heating": {2000: 82.1, 2014: 67.1, 2015: 73.2, 2016: 77.3, 2017: 75.0, 2018: 67.0, 2019: 69.7, 2020: 65.9},
    "hot-water": {2000: 12.7, 2014: 12.1, 2015: 12.1, 2016: 12.0, 2017: 12.0, 2018: 12.0, 2019: 11.9, 2020: 11.9},
    "process-heat": {2000: 2.3, 2014: 2.5, 2015: 2.5, 2016: 2.5, 2017: 2.5, 2018: 2.6, 2019: 2.7, 2020: 2.2},
    "lighting": {2000: 16.8, 2014: 17.0, 2015: 17.0, 2016: 17.0, 2017: 17.0, 2018: 16.9, 2019: 16.8, 2020: 15.9},
    "HVAC and building tech": {2000: 11.2, 2014: 13.8, 2015: 15.0, 2016: 15.1, 2017: 15.5, 2018: 15.5, 2019: 15.8, 2020: 15.0},
    "ICT and entertainment media": {2000: 6.1, 2014: 6.9, 2015: 6.9, 2016: 6.9, 2017: 6.9, 2018: 7.0, 2019: 7.0, 2020: 6.8},
    "Drives and processes": {2000: 14.4, 2014: 16.1, 2015: 16.0, 2016: 16.0, 2017: 15.9, 2018: 16.1, 2019: 16.2, 2020: 15.7},
    "Other": {2000: 4.0, 2014: 4.3, 2015: 4.4, 2016: 4.4, 2017: 4.3, 2018: 4.4, 2019: 4.3, 2020: 4.1},
    "Total": {2000: 149.7, 2014: 139.7, 2015: 147.2, 2016: 151.2, 2017: 149.2, 2018: 141.5, 2019: 144.4, 2020: 137.5}
}

dm_services_eud = load_services_energy_demand_eud(services_agri_split, years_ots)


# Agiculture demand is << than services, I will not split it here. I do have agriculture data by fuel
# !FIXME: Consider assigning Drives and processes here and in Industry to not only electricity but also diesel and gasoline.
# Basically remove from Drives and processes the diesel and gasoline demand. and the remainder is electricity.
# Also HVAC could be heat pumps and not electricity
dm_services_eud_cantons = map_services_eud_by_canton(dm_employees_mapped, dm_services_eud)

dm_services_fuels_eud_cantons = split_fuel_demand_by_eud(dm_water, dm_space_heat, dm_services_eud_cantons)

# I use the OFS data on fuels consumption by service and by canton to adjust the results.
# Concretely, for each fuel, I compute the share by end-use and then I multiply by the fuel OFS consumption
dm_services_fuels_eud_cantons_FSO = adjust_based_on_FSO_energy_consumption(dm_fuels_cantons, dm_services_fuels_eud_cantons)

# Add Switzerland
dm_services_fuels_eud_cantons_CH = dm_services_fuels_eud_cantons_FSO.groupby({'Switzerland': '.*'}, dim='Country', regex=True, inplace=False)
dm_services_fuels_eud_cantons_FSO.append(dm_services_fuels_eud_cantons_CH, dim='Country')

# Replace 1990-2000 flat extrapolation with linear fitting
idx = dm_services_fuels_eud_cantons_FSO.idx
dm_services_fuels_eud_cantons_FSO.array[:, idx[1990]: idx[2000], ...] = np.nan
linear_fitting(dm_services_fuels_eud_cantons_FSO, years_ots, based_on=create_years_list(2000, 2010, 1))
dm_services_fuels_eud_cantons_FSO.array[...] = np.maximum(0, dm_services_fuels_eud_cantons_FSO.array[...])

# Join services and industry
dm_services_fuels_eud_cantons_FSO.add(0, dummy=True, dim='Categories1', col_label='process-heat')
dm_fuels_eud_cantons.append(dm_services_fuels_eud_cantons_FSO, dim='Variables')



# Add FTS forecasting
file_lfs = '../../../data/datamatrix/lifestyles.pickle'
with open(file_lfs, 'rb') as handle:
    dm_lfs = pickle.load(handle)

dm_fuels_eud_cantons.sort('Country')
dm_fuels_eud_cantons.rename_col_regex(" /.*", "", dim='Country')
dm_fuels_eud_cantons.rename_col_regex("-", " ", dim='Country')
cantons_en = ['Aargau', 'Appenzell Ausserrhoden', 'Appenzell Innerrhoden', 'Basel Landschaft', 'Basel Stadt', 'Bern', 'Fribourg', 'Geneva', 'Glarus', 'Graubünden', 'Jura', 'Lucerne', 'Neuchâtel', 'Nidwalden', 'Obwalden', 'Schaffhausen', 'Schwyz', 'Solothurn', 'St. Gallen', 'Thurgau', 'Ticino', 'Uri', 'Valais', 'Vaud', 'Zug', 'Zurich']
cantons_fr = ['Aargau', 'Appenzell Ausserrhoden', 'Appenzell Innerrhoden', 'Basel Landschaft', 'Basel Stadt', 'Bern', 'Fribourg', 'Genève', 'Glarus', 'Graubünden', 'Jura', 'Luzern', 'Neuchâtel', 'Nidwalden', 'Obwalden', 'Schaffhausen', 'Schwyz', 'Solothurn', 'St. Gallen', 'Thurgau', 'Ticino', 'Uri', 'Valais', 'Vaud', 'Zug', 'Zürich']
dm_fuels_eud_cantons.rename_col(cantons_fr, cantons_en, dim='Country')
dm_fuels_eud_cantons.add(np.nan, dummy=True, dim='Years', col_label=years_fts)

dm_pop = dm_lfs['ots']['pop']['lfs_population_']
dm_pop.append(dm_lfs['fts']['pop']['lfs_population_'][1], dim='Years')
dm_pop.filter({'Country': dm_fuels_eud_cantons.col_labels['Country']}, inplace=True)
# Forecast based on linear extrapolation of TWh/cap x population
arr = dm_fuels_eud_cantons[:, :, :, :, :] / dm_pop[:, :, 0, np.newaxis, np.newaxis, np.newaxis]
dm_tmp = DataMatrix.based_on(arr, format=dm_fuels_eud_cantons, change= {'Variables': ['ind_cap', 'srv_cap']},
                             units={'ind_cap': 'TWh/cap', 'srv_cap': 'TWh/cap'})
#dm_tmp.fill_nans('Years')
linear_fitting(dm_tmp, years_fts, based_on=create_years_list(2010, 2023, 1))
dm_tmp.array = np.maximum(0, dm_tmp.array)

dm_fuels_eud_cantons[...] = dm_tmp[...] * dm_pop[:, :, 0, np.newaxis, np.newaxis, np.newaxis]

#dm_fuels_eud_cantons.flattest().datamatrix_plot({'Country': ['Switzerland']})
DM = {'ind-serv-energy-demand': dm_fuels_eud_cantons}

file_industry = '../../../data/interface/industry_to_energy.pickle'
with open(file_industry, 'wb') as handle:
    pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)

#my_pickle_dump(dm_fuels_eud_cantons, file_industry)
sort_pickle(file_industry)



print('Hello')