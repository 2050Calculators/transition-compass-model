from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat
from model.common.data_matrix_class import DataMatrix
import eurostat
import pandas as pd
from model.common.auxiliary_functions import create_years_list, eurostat_iso2_dict, linear_forecast_BAU_w_noise
import numpy as np
from model.common.io_database import csv_database_reformat, update_database_from_dm, edit_database


def get_CDD_data():
    # The datasource for the CDD is Eurostat, base temperature is 21, threshold is 24.
    file = 'Eurostat_CDD.xlsx'
    data_path = 'data/'
    rows_to_skip = list(range(8)) + [9] + [39, 40, 41]
    df_cdd = pd.read_excel(data_path+file, sheet_name='Sheet 1', skiprows=rows_to_skip)
    df_cdd.rename({'TIME': 'Country'}, axis=1, inplace=True)
    # It melts a short format to a long format, var_name is all of the columns except the one specified in id_vars
    df_melted = pd.melt(df_cdd, id_vars=['Country'], var_name='Years', value_name='clm_CDD[daysK]')
    dm_cdd = DataMatrix.create_from_df(df_melted, num_cat=0)
    dm_cdd.rename_col('European Union - 27 countries (from 2020)', 'EU27', dim='Country')
    dm_cdd.rename_col('Czechia', 'Czech Republic', dim='Country')
    return dm_cdd


def get_HDD_data():
    file = 'Eurostat_HDD.xlsx'
    data_path = 'data/'
    rows_to_skip = list(range(8)) + [9] + [39, 40, 41]
    df_hdd = pd.read_excel(data_path+file, sheet_name='Sheet 1', skiprows=rows_to_skip)
    df_hdd.rename({'TIME': 'Country'}, axis=1, inplace=True)
    # It melts a short format to a long format, var_name is all of the columns except the one specified in id_vars
    df_melted = pd.melt(df_hdd, id_vars=['Country'], var_name='Years', value_name='clm_HDD[daysK]')
    dm_hdd = DataMatrix.create_from_df(df_melted, num_cat=0)
    dm_hdd.rename_col('European Union - 27 countries (from 2020)', 'EU27', dim='Country')
    dm_hdd.rename_col('Czechia', 'Czech Republic', dim='Country')
    return dm_hdd

# Load CDD
#dm_cdd = get_CDD_data()
# Load HDD
#dm_hdd = get_HDD_data()

dict_iso2 = eurostat_iso2_dict()
years_ots = create_years_list(start_year=1990, end_year=2023, step=1)
years_fts = create_years_list(start_year=2025, end_year=2050, step=5)

filter = {'geo\TIME_PERIOD': dict_iso2.keys()}
mapping_dim = {'Country': 'geo\TIME_PERIOD', 'Variables': 'indic_nrg'}
eustat_code = 'nrg_chdd_a'
dm_hdd_cdd = get_data_api_eurostat(eustat_code, filter, mapping_dim, unit='daysK', years=years_ots)
dm_hdd_cdd.rename_col(['CDD', 'HDD'], ['clm_cdd', 'clm_hdd'], dim='Variables')

dm_hdd_cdd_fts = linear_forecast_BAU_w_noise(dm_hdd_cdd, start_t=1990, years_ots=years_ots, years_fts=years_fts)
dm_hdd_cdd_fts.array = np.maximum(0, dm_hdd_cdd_fts.array)
dm_hdd_cdd.append(dm_hdd_cdd_fts, dim='Years')
# dm_hdd_cdd.datamatrix_plot()

##### Switzerland & Vaud set to Austria
dm_hdd_cdd.add(dm_hdd_cdd.filter({'Country': ['Austria']}).array, dim='Country', col_label='Switzerland')
dm_hdd_cdd.add(dm_hdd_cdd.filter({'Country': ['Austria']}).array, dim='Country', col_label='Vaud')

update_hdd_cdd = True
if update_hdd_cdd:
    dm_hdd_cdd_ots = dm_hdd_cdd.filter({'Years': years_ots})
    dm_hdd_cdd_fts = dm_hdd_cdd.filter({'Years': years_fts})
    file = 'climate_temperature.csv'
    # Run the following two lines if you are starting from the "original" climate_temperature.csv
    #csv_database_reformat(file)
    #edit_database('climate_temperature', 'temp', column='variable', pattern='bld_*', mode='remove')

    update_database_from_dm(dm_hdd_cdd_ots, filename=file, lever='temp', level=0, module='climate')
    # Level 1 to 4 is the same for the moment
    for lev in range(4):
        lev = lev+1
        update_database_from_dm(dm_hdd_cdd_fts, filename=file, lever='temp', level=lev, module='climate')


print('Hello')