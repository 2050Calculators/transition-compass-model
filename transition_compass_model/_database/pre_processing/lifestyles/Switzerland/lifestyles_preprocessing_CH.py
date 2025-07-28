import numpy as np
import pandas as pd
import pickle


from model.common.auxiliary_functions import linear_fitting, linear_fitting_ots_db, create_years_list, my_pickle_dump
from model.common.io_database import update_database_from_dm, csv_database_reformat
from _database.pre_processing.api_routines_CH import get_data_api_CH
from model.common.data_matrix_class import DataMatrix

import math
import requests

from _database.pre_processing.WorldBank_data_extract import get_WB_data

import os


def create_ots_years_list(years_setting):
    startyear: int = years_setting[0]  # Start year is argument [0], i.e., 1990
    baseyear: int = years_setting[1]  # Base/Reference year is argument [1], i.e., 2015
    lastyear: int = years_setting[2]  # End/Last year is argument [2], i.e., 2050
    step_fts = years_setting[3]  # Timestep for scenario is argument [3], i.e., 5 years
    years_ots = list(
        np.linspace(start=startyear, stop=baseyear, num=(baseyear - startyear) + 1).astype(int).astype(str))
    return years_ots


def create_fts_years_list():

    years_fts = list(
        np.linspace(start=2025, stop=2050, num=6).astype(int))
    return years_fts

###############################
### POPULATION - deprecated ###
###############################
def deprecated_extract_lfs_population_total(years_ots, all_cantons):
    # Demographic balance by institutional units
    table_id = "px-x-0102020000_201"
    structure = get_data_api_CH(table_id, mode='example')

    filter = {'Year': years_ots,
              'Canton (-) / District (>>) / Commune (......)': ['Switzerland', '- Vaud'],
              'Citizenship (category)': 'Citizenship (category) - total',
              'Sex': 'Sex - total',
              'Demographic component': 'Population on 1 January'}

    mapping_dim = {'Country': 'Canton (-) / District (>>) / Commune (......)',
                   'Years': 'Year',
                   'Variables': 'Demographic component'}

    dm_lfs_pop = get_data_api_CH(table_id, mode='extract', filter=filter,
                                 mapping_dims=mapping_dim, units=['inhabitants'])

    dm_lfs_pop.rename_col_regex('- ', '', dim='Country')
    dm_lfs_pop.rename_col('Population on 1 January', 'lfs_population_total', dim='Variables')

    return dm_lfs_pop


######################################
### POPULATION by age - deprecated ###
######################################
def deprecated_extract_lfs_demography_age():
    # ! FIXME: this database is missing many years, find another one model the missing years
    table_id = 'px-x-0102010000_103'
    structure = get_data_api_CH(table_id, mode='example')

    # Extract all age classes
    filter = {'Year': structure['Year'],
              'Canton (-) / District (>>) / Commune (......)': ['Switzerland', '- Vaud'],
              'Population type': ['Permanent resident population', 'Non permanent resident population'],
              'Sex': ['Male', 'Female'],
              'Marital status': 'Marital status - total',
              'Age class': structure['Age class']}

    mapping_dim = {'Country': 'Canton (-) / District (>>) / Commune (......)',
                   'Years': 'Year',
                   'Variables': 'Population type',
                   'Categories1': 'Sex',
                   'Categories2': 'Age class'}
    # Extract population by age group data
    dm_lfs_pop_age = get_data_api_CH(table_id, mode='extract', filter=filter, mapping_dims=mapping_dim,
                                     units=['inhabitants', 'inhabitants'])
    # Group age categories
    dm_lfs_pop_age.groupby({'below19': ['0-4 years', '5-9 years', '10-14 years', '15-19 years'],
                            'age20-29': ['20-24 years', '25-29 years'],
                            'age30-54': ['30-34 years', '35-39 years', '40-44 years', '45-49 years', '50-54 years'],
                            'age55-64': ['55-59 years', '60-64 years'],
                            'above65': ['65-69 years', '70-74 years', '75-79 years', '80-84 years', '85-89 years',
                                        '90-94 years', '95-99 years', '100 years or older']},
                            dim='Categories2', inplace=True, regex=False)
    # Rename sex
    dm_lfs_pop_age.rename_col('Male', 'male', dim='Categories1')
    dm_lfs_pop_age.rename_col('Female', 'female', dim='Categories1')
    # Rename Vaud
    dm_lfs_pop_age.rename_col_regex('- ', '', 'Country')
    # Group permanent and non-permanent (permis C / citizenship and other)
    dm_lfs_pop_age.groupby({'lfs_demography': ['Permanent resident population', 'Non permanent resident population']},
                           dim='Variables', inplace=True, regex=False)
    # Extract age class total to compare with tot population data
    dm_tot_resident_pop = dm_lfs_pop_age.filter({'Categories2': ['Age class - total']}, inplace=False)
    dm_lfs_pop_age.drop('Categories2', 'Age class - total')
    # Join sex and age group
    dm_lfs_pop_age = dm_lfs_pop_age.flatten(sep='-')

    # Drop sex
    dm_tot_resident_pop.group_all('Categories1', inplace=True)

    return dm_lfs_pop_age


########################
###    POPULATION    ###
### tot & by age new ###
########################
def extract_lfs_pop(years_ots, table_id, file):

    try:
        with open(file, 'rb') as handle:
            dm_lfs_pop_age = pickle.load(handle)
    except OSError:
        # Demographic balance by age and canton
        structure, title = get_data_api_CH(table_id, mode='example')

        # Extract all age classes
        years_ots_str = [str(y) for y in years_ots]
        dm_lfs_pop_age = None
        for canton in structure['Canton']:
            filter = {'Year': years_ots_str,
                      'Canton': canton,
                      'Citizenship (category)': 'Citizenship (category) - total',  # Swiss and non-Swiss resident
                      'Sex': ['Male', 'Female'],
                      'Age': structure['Age'],
                      'Demographic component': 'Population on 1 January'}

            mapping_dim = {'Country': 'Canton',
                           'Years': 'Year',
                           'Variables': 'Demographic component',
                           'Categories1': 'Sex',
                           'Categories2': 'Age'}
            # Extract population by age group data
            dm_lfs_pop_age_canton = get_data_api_CH(table_id, mode='extract', filter=filter, mapping_dims=mapping_dim,
                                                    units=['inhabitants'])
            if dm_lfs_pop_age is None:
                dm_lfs_pop_age = dm_lfs_pop_age_canton
            else:
                dm_lfs_pop_age.append(dm_lfs_pop_age_canton, dim='Country')

        dm_lfs_pop_age.sort('Country')
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, 'wb') as handle:
            pickle.dump(dm_lfs_pop_age, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dm_lfs_pop_age.rename_col_regex('Confederation', 'Switzerland', 'Country')
    dm_lfs_pop_age.rename_col_regex(" /.*", "", dim='Country')
    dm_lfs_pop_age.rename_col_regex("-", " ", dim='Country')
    dm_lfs_pop_age.rename_col(cantons_fr, cantons_en, dim='Country')
    dm_lfs_pop_age.sort('Country')

    dm_lfs_pop_age.drop(col_label='No indication', dim='Country')
    dm_lfs_pop_tot = dm_lfs_pop_age.filter({'Categories2': ['Age - total']})
    dm_lfs_pop_age.drop(dim='Categories2', col_label='Age - total')
    dm_lfs_pop_age.drop(col_label='No indication', dim='Categories2')
    dm_lfs_pop_age.rename_col_regex(' years', '', dim='Categories2')
    dm_lfs_pop_age.rename_col_regex(' year', '', dim='Categories2')
    dm_lfs_pop_age.rename_col('99 or older', '99', dim='Categories2')
    dm_lfs_pop_age.drop(dim='Categories2', col_label='No indication')

    # Group ages by category
    group_dict = {
        'below19': [],
        'age20-29': [],
        'age30-54': [],
        'age55-64': [],
        'above65': [],
    }
    for age in dm_lfs_pop_age.col_labels['Categories2']:
        if int(age) <= 19:
            group_dict['below19'].append(age)
        elif (int(age) >= 20) and (int(age) <= 29):
            group_dict['age20-29'].append(age)
        elif (int(age) >= 30) and (int(age) <= 54):
            group_dict['age30-54'].append(age)
        elif (int(age) >= 55) and (int(age) <= 64):
            group_dict['age55-64'].append(age)
        elif int(age) >= 65:
            group_dict['above65'].append(age)

    dm_lfs_pop_age.groupby(group_dict, 'Categories2', inplace=True, regex=False)
    # Rename sex
    dm_lfs_pop_age.rename_col('Male', 'male', dim='Categories1')
    dm_lfs_pop_age.rename_col('Female', 'female', dim='Categories1')
    dm_lfs_pop_age.sort("Categories1")
    dm_lfs_pop_age.rename_col('Population on 1 January', 'lfs_demography', dim='Variables')
    dm_lfs_pop_age = dm_lfs_pop_age.flatten(sep='-')

    dm_lfs_pop_tot.group_all('Categories2')
    dm_lfs_pop_tot.group_all('Categories1')
    dm_lfs_pop_tot.rename_col('Population on 1 January', 'lfs_population_total', dim='Variables')

    # Sort Years
    dm_lfs_pop_age.sort('Years')
    dm_lfs_pop_tot.sort('Years')

    return dm_lfs_pop_age, dm_lfs_pop_tot


def extract_lfs_pop_fts(years_fts, table_id, file):

    try:
        with open(file, 'rb') as handle:
            dm_pop_fts = pickle.load(handle)
    except OSError:
        structure, title = get_data_api_CH(table_id, mode='example', language='fr')
        # Extract buildings floor area
        scenarios = ['Scénario de référence A-00-2025', "Scénario B-00-2025 'haut'",
                     "Variante A-03-2025 'plus haute espérance de vie à la naissance'", "Scénario C-00-2025 'bas'"]
        filter = {'Scénario-variante': scenarios,
                  'Nationalité (catégorie)': ['Nationalité - total'],
                  'Sexe': ['Homme', 'Femme'],
                  "Classe d'âge": structure["Classe d'âge"],
                  "Unité d'observation": ['Population au 1er janvier'],
                  "Année": structure["Année"]}
        mapping_dim = {'Country': 'Nationalité (catégorie)', 'Years': "Année", 'Variables': 'Scénario-variante',
                       'Categories1': 'Sexe', 'Categories2': "Classe d'âge"}
        unit_all = ['inhabitants'] * len(filter['Scénario-variante'])
        # Get api data
        dm_pop_fts = get_data_api_CH(table_id, mode='extract', filter=filter, mapping_dims=mapping_dim, units=unit_all,
                                     language='fr')
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, 'wb') as handle:
            pickle.dump(dm_pop_fts, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Rename
    dm_pop_fts.rename_col('Nationalité - total', 'Switzerland', dim='Country')

    # Filter only fts years
    dm_pop_fts.filter({'Years': years_fts}, inplace=True)
    dm_pop_fts.sort(dim='Years')

    dm_pop_fts_tot = dm_pop_fts.filter({'Categories2': ["Classe d'âge - total"]})
    dm_pop_fts_tot.group_all(dim='Categories2')
    dm_pop_fts_tot.group_all(dim='Categories1')

    dm_pop_fts.drop(dim='Categories2', col_label="Classe d'âge - total")
    dm_pop_fts.rename_col_regex(' ans', '', dim='Categories2')

    # Group ages by category
    group_dict = {
        'below19': [],
        'age20-29': [],
        'age30-54': [],
        'age55-64': [],
        'above65': [],
    }
    min_age = [col.split('-')[0] for col in dm_pop_fts.col_labels['Categories2']]
    dm_pop_fts.rename_col(col_in=dm_pop_fts.col_labels['Categories2'].copy(), col_out=min_age, dim='Categories2')

    for age in dm_pop_fts.col_labels['Categories2']:
        if int(age) <= 19:
            group_dict['below19'].append(age)
        elif (int(age) >= 20) and (int(age) <= 29):
            group_dict['age20-29'].append(age)
        elif (int(age) >= 30) and (int(age) <= 54):
            group_dict['age30-54'].append(age)
        elif (int(age) >= 55) and (int(age) <= 64):
            group_dict['age55-64'].append(age)
        elif int(age) >= 65:
            group_dict['above65'].append(age)

    dm_pop_fts.groupby(group_dict, 'Categories2', inplace=True, regex=False)
    # Rename sex
    dm_pop_fts.rename_col('Homme', 'male', dim='Categories1')
    dm_pop_fts.rename_col('Femme', 'female', dim='Categories1')
    dm_pop_fts.sort("Categories1")
    dm_pop_fts = dm_pop_fts.flatten(sep='-')

    # Assign levers
    level_dict = {1: "Scénario de référence A-00-2025",
                  2: "Scénario B-00-2025 'haut'",
                  3: "Variante A-03-2025 'plus haute espérance de vie à la naissance'",
                  4: "Scénario C-00-2025 'bas'"}
    dict_dm_pop_fts = dict()
    dict_dm_pop_fts_tot = dict()
    for k, v in level_dict.items():
        dict_dm_pop_fts[k] = dm_pop_fts.filter({'Variables': [v]})
        dict_dm_pop_fts[k].rename_col(v, 'lfs_demography', dim='Variables')

        dict_dm_pop_fts_tot[k] = dm_pop_fts_tot.filter({'Variables': [v]})
        dict_dm_pop_fts_tot[k].rename_col(v, 'lfs_population_total', dim='Variables')

    return dict_dm_pop_fts, dict_dm_pop_fts_tot


def add_vaud_fts_pop(dm_lfs_age, dm_lfs_tot_pop, dict_lfs_age_fts, dict_lfs_tot_pop_fts):
    idx = dm_lfs_tot_pop.idx
    # vaud share for last ots year available
    for canton in (set(dm_lfs_tot_pop.col_labels['Country']) - {'Switzerland'}):
        vaud_share = dm_lfs_tot_pop.array[idx[canton], -1, ...] / dm_lfs_tot_pop.array[idx['Switzerland'], -1, ...]
        # Check that ots and fts are harmonised
        for l, dm_fts in dict_lfs_tot_pop_fts.items():
            vaud_arr = vaud_share * dm_fts['Switzerland', :, :]
            dm_fts.add(vaud_arr, dim='Country', col_label=canton)
            # Remove comment to check smoothness
            # dm_fts.append(dm_lfs_tot_pop, dim='Years')
            # dm_fts.sort(dim='Years')
            # dm_fts.datamatrix_plot(title=l)

        idx = dm_lfs_tot_pop.idx
        vaud_share = dm_lfs_age.array[idx[canton], -1, ...] / dm_lfs_age.array[idx['Switzerland'], -1, ...]
        for l, dm_fts in dict_lfs_age_fts.items():
            vaud_arr = vaud_share[np.newaxis, np.newaxis, ...] * dm_fts['Switzerland', :, :, :]
            dm_fts.add(vaud_arr, dim='Country', col_label=canton)

            # Remove comment to check smoothness
            # dm_fts.append(dm_lfs_age, dim='Years')
            # dm_fts.sort(dim='Years')
            # dm_fts.datamatrix_plot(title=l)
    for l, dm_fts in dict_lfs_tot_pop_fts.items():
        dm_fts.sort('Country')
    for l, dm_fts in dict_lfs_age_fts.items():
        dm_fts.sort('Country')

    return


########################
### URBAN POPULATION ###
########################
# NOT USED
def extract_lfs_urban_share(years_ots, table_id):
    # Suisse urbaine: sélection de variables selon la typologie urbain-rural
    structure, title = get_data_api_CH(table_id, mode='example', language='fr')

    # Extract all age classes
    filter = {'Année': structure['Année'],
              'Résultat': 'Valeur',
              'Variable': 'Population résidante permanente, total',  # Swiss and non-Swiss resident
              'Typologie urbain-rural': structure['Typologie urbain-rural']}

    mapping_dim = {'Country': 'Résultat',
                   'Years': 'Année',
                   'Variables': 'Typologie urbain-rural'}

    # Extract urban / rural pop
    dm_lfs_urban = get_data_api_CH(table_id, mode='extract', filter=filter, mapping_dims=mapping_dim,
                                   units=['inhabitants', 'inhabitants', 'inhabitants'], language='fr')

    dm_lfs_urban.rename_col('Valeur', 'Switzerland', dim='Country')
    dm_lfs_urban.groupby({'non-urban': ['Intermédiaire (périurbain dense et centres ruraux)', 'Rural'],
                          'urban': ['Urbain']}, dim='Variables', inplace=True, regex=False)

    # Perform linear extrapolation all the way back to 1990
    linear_fitting(dm_lfs_urban, years_ots)

    # Compute urban share of total population
    dm_lfs_urban.operation('urban', '+', 'non-urban', out_col='total', unit='inhabitants', dim='Variables')
    dm_lfs_urban.operation('urban', '/', 'total', out_col='lfs_demography_urban-population', unit='%', dim='Variables',
                           div0='error')
    dm_lfs_urban.filter({'Variables': ['lfs_demography_urban-population']}, inplace=True)

    # Vaud urban pop rate as CH
    dm_lfs_urban_VD = dm_lfs_urban.copy()
    dm_lfs_urban_VD.rename_col('Switzerland', 'Vaud', 'Country')

    dm_lfs_urban.append(dm_lfs_urban_VD, dim='Country')

    return dm_lfs_urban


########################
####   FLOOR-AREA   ####
########################
def extract_lfs_floor_space(years_ots, dm_lfs_tot_pop, table_id):
    structure, title = get_data_api_CH(table_id, mode='example', language='fr')

    # Extract buildings floor area
    filter = {'Année': structure['Année'],
              'Canton (-) / District (>>) / Commune (......)': ['Suisse', '- Vaud'],
              'Catégorie de bâtiment': structure['Catégorie de bâtiment'],
              'Surface du logement': structure['Surface du logement'],
              'Époque de construction': structure['Époque de construction']}
    mapping_dim = {'Country': 'Canton (-) / District (>>) / Commune (......)', 'Years': 'Année',
                   'Variables': 'Catégorie de bâtiment', 'Categories1': 'Surface du logement',
                   'Categories2': 'Époque de construction'}
    unit_all = ['number'] * len(structure['Catégorie de bâtiment'])
    # Get api data
    dm_floor_area = get_data_api_CH(table_id, mode='extract', filter=filter, mapping_dims=mapping_dim, units=unit_all,
                                    language='fr')

    # Pre-processing

    ## Rename & Group
    dm_floor_area.rename_col(['Suisse'], ['Switzerland'], dim='Country')
    dm_floor_area.rename_col_regex('- ', '', dim='Country')
    dm_floor_area.group_all('Categories2')
    dm_floor_area.groupby({'lfs_dwellings': '.*'}, dim='Variables', regex=True, inplace=True)
    dm_num_bld = dm_floor_area.group_all('Categories1', inplace=False)

    ## Compute total floor space
    dm_floor_area.rename_col_regex(' m2', '', 'Categories1')
    # The average size for less than 30 is a guess, as is the average size for 150+,
    # we will use the data from bfs to calibrate
    avg_size = {'<30': 25, '30-49': 39.5, '50-69': 59.5, '70-99': 84.5, '100-149': 124.5, '150+': 175}
    idx = dm_floor_area.idx
    for size in dm_floor_area.col_labels['Categories1']:
        dm_floor_area.array[:, :, :, idx[size]] = avg_size[size] * dm_floor_area.array[:, :, :, idx[size]]
    dm_floor_area.rename_col('lfs_dwellings', 'lfs_floor-area', 'Variables')
    dm_floor_area.units['lfs_floor-area'] = 'm2'
    dm_floor_area.group_all('Categories1')

    ## Calibrate
    # From https://www.bfs.admin.ch/bfs/en/home/statistics/construction-housing/dwellings/size.html#accordion1719560162958
    # In 2022 the average floor space per dwelling was 99m2. The relative stability observed since 2000 (97m2)
    # can be explained by the fact that dwellings built prior to 1981 (60% of the housing stock) have an average floor space
    # of less than 100m2. The size of more recent dwellings, however, is larger, and dwellings built between 2001 and 2005
    # have an average floor space of 131m2.
    # Compute average size of dwelling
    avg_floor_space_CH_2022_m2 = 99
    dm_floor_area.append(dm_num_bld, dim='Variables')
    dm_floor_area.operation('lfs_floor-area', '/', 'lfs_dwellings', dim='Variables', out_col='lfs_avg-floor-size',
                            unit='m2')
    cal_factor = avg_floor_space_CH_2022_m2 / dm_floor_area.array[
        idx['Switzerland'], idx[2022], idx['lfs_avg-floor-size']]
    dm_floor_area.filter({'Variables': ['lfs_avg-floor-size']}, inplace=True)
    dm_floor_area.array = dm_floor_area.array * cal_factor

    # Linear fitting
    linear_fitting(dm_floor_area, years_ots)
    linear_fitting(dm_num_bld, years_ots)

    # Recompute total floor-area = avg-floor-size * nb_dwellings
    dm_floor_area.append(dm_num_bld, dim='Variables')
    dm_floor_area.operation('lfs_avg-floor-size', '*', 'lfs_dwellings', out_col='lfs_floor-area', unit='m2')

    # Compute floor-area per capita
    dm_floor_area.append(dm_lfs_tot_pop, dim='Variables')
    dm_floor_area.operation('lfs_floor-area', '/', 'lfs_population_total', out_col='lfs_floor-intensity_space-cap',
                            unit='m2/cap')

    dm_floor_area.filter({'Variables': ['lfs_floor-intensity_space-cap']}, inplace=True)

    return dm_floor_area


#######################
### COOL-AREA-SHARE ###
#######################
def extract_lfs_floor_area_cool_share():
    warning = 'The floor_area_cool_share routine is not complete, temperature data and GDP data are missing'
    # "Space cooling technology in Europe" by HeatRoadmapEU
    # https://heatroadmap.eu/wp-content/uploads/2018/11/HRE4_D3.2.pdf
    # Equation "share of residential floor area cooled" at page 21
    # Household size as number of people
    hhd = 2.2
    # GNI is the average per capita purchasing power parity
    GNI = 92980
    # Cooled Days Degrees
    CDD = 1400
    # Market saturation = all need covered
    share_of_residential_floor_area_cooled = 0.815 * (1 - math.exp(-0.00225 * CDD)) / (
                1 + 126.8 * math.exp(-0.000069 * GNI / hhd))
    # To gather temperature data:
    # At EPFL you can access MeteoSwiss measurements through CLIMAP app available on Windows or on mac by virtual machine
    # Visit https://enacserver.epfl.ch/Servers/RemoteDesktop/enacvm-climap for info
    print(warning)
    return


############################
### GDP (PPP) per capita ###
############################
def extract_per_capita_gdp_ppp():
    file_url = 'https://api.worldbank.org/v2/en/indicator/NY.GDP.PCAP.PP.CD?downloadformat=excel'
    # Define the local filename to save the downloaded file
    local_filename = 'data/GDP_World.xlsx'
    var_name = 'GDP[USD/cap]'
    dm_GDP = get_WB_data(file_url, local_filename, var_name, years_ots)

    # In case you want to have simply GDP per capita data in CHF you have
    read_CH_GDP = False
    if read_CH_GDP:
        # URL of the file to be downloaded
        file_url = 'https://dam-api.bfs.admin.ch/hub/api/dam/assets/27065040/master'
        # Define the local filename to save the downloaded file
        local_filename = 'data/GDP_Switzerland.xlsx'

        if not os.path.exists(local_filename):
            # Send a GET request to the URL
            response = requests.get(file_url, stream=True)
            # Check if the request was successful
            if response.status_code == 200:
                # Open the local file in write-binary mode and write the response content to it
                with open(local_filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"File downloaded successfully as {local_filename}")
            else:
                print(f"Error: {response.status_code}, {response.text}")

        df = pd.read_excel(local_filename)

        # Clean df
        # Rename columns
        rename_dict = {'PIB par habitant à prix courants': 'Years', 'Unnamed: 1': 'GDP[CHF/cap]'}
        df.rename(columns=rename_dict, inplace=True)
        df = df[['Years', 'GDP[CHF/cap]']]
        # Only keep ots years
        int_years_ots = [int(year_str) for year_str in years_ots]
        df_years = df[df['Years'].isin(int_years_ots)]
        # Add Country column = Switzerland
        df_years['Country'] = 'Switzerland'
        # Create datamatrix
        dm_GDP = DataMatrix.create_from_df(df_years, num_cat=0)

    return dm_GDP


def dummy_update_DM_module_baseyear(DM_old, years_ots, years_fts):

    def dummy_update_dm_ots_baseyear(dm_ots, dm_fts, years_ots):
        years_ots_missing = list(set(years_ots) - set(dm_ots.col_labels['Years']))
        dm_ots.add(np.nan, dummy=True, dim='Years', col_label=years_ots_missing)
        dm_ots.sort('Years')
        first_fts_year = dm_fts.col_labels['Years'][0]
        if first_fts_year in years_ots:
            idx = dm_ots.idx
            dm_ots.array[:, idx[first_fts_year], ...] = dm_fts.array[:, 0, ...]
        linear_fitting(dm_ots, dm_ots.col_labels['Years'])
        return dm_ots


    DM_new = dict()
    for key, value in DM_old.items():
        if isinstance(value, dict):
            DM_new[key] = dict()
    # key = 'fts', 'ots'
    # Update ots to new baseyear using old fts value (usually 2020)
    for lever in DM_old['ots']:
        DM_new['ots'][lever] = dict()
        DM_new['fts'][lever] = dict()
        var = DM_old['ots'][lever]
        # If it is a dictionary
        if isinstance(var, dict):
            for dm_name in DM_old['ots'][lever]:
                dm_ots = DM_old['ots'][lever][dm_name].copy()
                dm_fts = DM_old['fts'][lever][dm_name][1]
                dm_ots = dummy_update_dm_ots_baseyear(dm_ots, dm_fts, years_ots)
                DM_new['ots'][lever][dm_name] = dm_ots
                DM_new['fts'][lever][dm_name] = dict()
                for level in range(4):
                    level = level + 1
                    dm_fts_old = DM_old['fts'][lever][dm_name][level]
                    DM_new['fts'][lever][dm_name][level] = dm_fts_old.filter({'Years': years_fts})
        else:
            dm_ots = DM_old['ots'][lever].copy()
            dm_fts = DM_old['fts'][lever][1]
            dm_ots = dummy_update_dm_ots_baseyear(dm_ots, dm_fts, years_ots)
            DM_new['ots'][lever] = dm_ots
            DM_new['fts'][lever] = dict()
            for level in range(4):
                level = level + 1
                dm_fts_old = DM_old['fts'][lever][level]
                DM_new['fts'][lever][level] = dm_fts_old.filter({'Years': years_fts})

    if 'fxa' in DM_old.keys():
        for name in DM_old['fxa'].keys():
            dm_new = DM_old['fxa'][name].copy()
            linear_fitting(dm_new, years_ots+years_fts)
            DM_new['fxa'][name] = dm_new

    if 'constant' in DM_old.keys():
        DM_new['constant'] = DM_old['constant']

    return DM_new


def filter_country_DM(cntr_list, DM):

    for lever in DM['ots'].keys():
        if isinstance(DM['ots'][lever], dict):
            for dm_name in DM['ots'][lever].keys():
                dm_ots = DM['ots'][lever][dm_name]
                dm_ots.filter({'Country': cntr_list}, inplace=True)
                DM['ots'][lever][dm_name] = dm_ots
                for level in range(4):
                    level = level + 1
                    dm_fts = DM['fts'][lever][dm_name][level]
                    dm_fts.filter({'Country': cntr_list}, inplace=True)
                    DM['fts'][lever][dm_name][level] = dm_fts
        else:
            dm_ots = DM['ots'][lever]
            dm_ots.filter({'Country': cntr_list}, inplace=True)
            DM['ots'][lever]= dm_ots
            for level in range(4):
                level = level + 1
                dm_fts = DM['fts'][lever][level]
                dm_fts.filter({'Country': cntr_list}, inplace=True)
                DM['fts'][lever][level] = dm_fts

    if 'fxa' in DM.keys():
        for name in DM['fxa'].keys():
            dm = DM['fxa'][name]
            dm.filter({'Country': cntr_list}, inplace=True)
            DM['fxa'][name] = dm

    return DM

# __file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/lifestyles/Switzerland/lifestyles_preprocessing_CH.py"
years_setting = [1990, 2023, 2050, 5]  # Set the timestep for historical years & scenarios
years_ots = create_years_list(start_year=1990, end_year=2023, step=1)
years_fts = create_years_list(start_year=2025, end_year=2050, step=5)

cantons_en = ['Aargau', 'Appenzell Ausserrhoden', 'Appenzell Innerrhoden', 'Basel Landschaft', 'Basel Stadt', 'Bern', 'Fribourg', 'Geneva', 'Glarus', 'Graubünden', 'Jura', 'Lucerne', 'Neuchâtel', 'Nidwalden', 'Obwalden', 'Schaffhausen', 'Schwyz', 'Solothurn', 'St. Gallen', 'Thurgau', 'Ticino', 'Uri', 'Valais', 'Vaud', 'Zug', 'Zurich']
cantons_fr = ['Aargau', 'Appenzell Ausserrhoden', 'Appenzell Innerrhoden', 'Basel Landschaft', 'Basel Stadt', 'Bern', 'Fribourg', 'Genève', 'Glarus', 'Graubünden', 'Jura', 'Luzern', 'Neuchâtel', 'Nidwalden', 'Obwalden', 'Schaffhausen', 'Schwyz', 'Solothurn', 'St. Gallen', 'Thurgau', 'Ticino', 'Uri', 'Valais', 'Vaud', 'Zug', 'Zürich']

# Get population total and by age group (ots)
filename = 'data/lfs_pop_ots_all_cantons.pickle'
dm_lfs_age, dm_lfs_tot_pop = extract_lfs_pop(years_ots, table_id='px-x-0102020000_104', file=filename)
# Get raw fts pop data (fts)
filename = 'data/lfs_pop_fts.pickle'
dict_lfs_age_fts, dict_lfs_tot_pop_fts = extract_lfs_pop_fts(years_fts, table_id='px-x-0104000000_101', file=filename)
add_vaud_fts_pop(dm_lfs_age, dm_lfs_tot_pop, dict_lfs_age_fts, dict_lfs_tot_pop_fts)

# Store
DM_lfs = {"ots": {"pop": {"lfs_demography_": [],
                          "lfs_population_": []}},
          "fts": {"pop": {"lfs_demography_": dict(),
                          "lfs_population_": dict()}}}
DM_lfs['ots']['pop']['lfs_demography_'] = dm_lfs_age
for lev in range(4):
    lev = lev + 1
    DM_lfs['fts']['pop']['lfs_demography_'][lev] = dict_lfs_age_fts[lev]
DM_lfs['ots']['pop']['lfs_population_'] = dm_lfs_tot_pop
for lev in range(4):
    lev = lev + 1
    DM_lfs['fts']['pop']['lfs_population_'][lev] = dict_lfs_tot_pop_fts[lev]

# Save
current_file_directory = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(current_file_directory, '../../../data/datamatrix/lifestyles.pickle')

#with open(file, 'wb') as handle:
#  pickle.dump(DM_lfs, handle, protocol=pickle.HIGHEST_PROTOCOL)

my_pickle_dump(DM_new=DM_lfs, local_pickle_file=file)


