import pandas as pd
from model.common.data_matrix_class import DataMatrix
from model.common.auxiliary_functions import create_years_list, moving_average, linear_fitting
import pickle
import numpy as np

####################################
######        CDD, HDD       #######
####################################
def extract_Yasser_dataset(filename, variable, country):
    df = pd.read_csv(filename)
    if 'canton' in df.columns:
        df = df.loc[df['canton'] == country]
        df.drop(columns='canton', inplace=True)
    df = df.T
    df.reset_index(inplace=True)
    df.columns = ['Years', variable]
    df['Country'] = country
    dm = DataMatrix.create_from_df(df, 0)
    return dm


def join_ots_fts_Yasser_dataset(dm_ots, dm_fts, years_fts):
    dm = dm_fts.copy()

    for i in range(3):
        window_size = 5  # Change window size to control the smoothing effect
        data_smooth = moving_average(dm.array, window_size, axis=dm.dim_labels.index('Years'))
        dm.array[:, 2:-2, ...] = data_smooth

    dm.filter({'Years': years_fts}, inplace=True)

    dm.append(dm_ots, dim='Years')
    dm.sort('Years')

    return dm


def extract_CDD_HDD_from_Yasser_dataset(years_fts):

    ots_folder = 'data/historical_meteoswiss/'
    fts_folder = 'data/ch2018_climate_models/'
    swiss_file_CDD = ots_folder + 'historical_meteoswiss_national_CDD.csv'
    swiss_file_HDD = ots_folder + 'historical_meteoswiss_national_HDD.csv'
    swiss_file_CDD_fts = fts_folder + 'CLMCOM-CCLM4_ECEARTH_EUR11_RCP85_national_CDD.csv'
    swiss_file_HDD_fts = fts_folder + 'CLMCOM-CCLM4_ECEARTH_EUR11_RCP85_national_HDD.csv'
    canton_file_CDD = ots_folder + 'historical_meteoswiss_cantonal_CDD.csv'
    canton_file_HDD = ots_folder + 'historical_meteoswiss_cantonal_HDD.csv'
    canton_file_CDD_fts = fts_folder + 'CLMCOM-CCLM4_ECEARTH_EUR11_RCP85_cantonal_CDD.csv'
    canton_file_HDD_fts = fts_folder + 'CLMCOM-CCLM4_ECEARTH_EUR11_RCP85_cantonal_HDD.csv'

    mapping = {
        'CDD': {'Switzerland': (swiss_file_CDD, swiss_file_CDD_fts), 'Vaud': (canton_file_CDD, canton_file_CDD_fts)},
        'HDD': {'Switzerland': (swiss_file_HDD, swiss_file_HDD_fts), 'Vaud': (canton_file_HDD, canton_file_HDD_fts)}}
    name_var = {'CDD': 'clm_CDD[daysK]', 'HDD': 'clm_HDD[daysK]'}

    for HCDD, country_file in mapping.items():
        for country, files in country_file.items():
            dm_ots = extract_Yasser_dataset(files[0], name_var[HCDD], country)
            dm_fts = extract_Yasser_dataset(files[1], name_var[HCDD], country)
            dm = join_ots_fts_Yasser_dataset(dm_ots, dm_fts, years_fts)
            if 'dm_all' in locals():
                dm_all.append(dm, dim='Country')
            else:
                dm_all = dm
        if 'dm_HCDD' in locals():
            dm_HCDD.append(dm_all, dim='Variables')
        else:
            dm_HCDD = dm_all.copy()
        del dm_all

    return dm_HCDD


def extract_days_Tbase_from_Yasser_dataset(years_fts):
    ots_folder = 'data/historical_meteoswiss/'
    fts_folder = 'data/ch2018_climate_models/'
    swiss_file_Nabove = ots_folder + 'historical_meteoswiss_national_above_tbase.csv'
    swiss_file_Nbelow = ots_folder + 'historical_meteoswiss_national_below_tbase.csv'
    swiss_file_Nabove_fts = fts_folder + 'CLMCOM-CCLM4_ECEARTH_EUR11_RCP85_national_above_tbase.csv'
    swiss_file_Nbelow_fts = fts_folder + 'CLMCOM-CCLM4_ECEARTH_EUR11_RCP85_national_below_tbase.csv'
    canton_file_Nabove = ots_folder + 'historical_meteoswiss_cantonal_above_tbase.csv'
    canton_file_Nbelow = ots_folder + 'historical_meteoswiss_cantonal_below_tbase.csv'
    canton_file_Nabove_fts = fts_folder + 'CLMCOM-CCLM4_ECEARTH_EUR11_RCP85_cantonal_above_tbase.csv'
    canton_file_Nbelow_fts = fts_folder + 'CLMCOM-CCLM4_ECEARTH_EUR11_RCP85_cantonal_below_tbase.csv'

    mapping = {'Nabove': {'Switzerland': (swiss_file_Nabove, swiss_file_Nabove_fts),
                          'Vaud': (canton_file_Nabove, canton_file_Nabove_fts)},
               'Nbelow': {'Switzerland': (swiss_file_Nbelow, swiss_file_Nbelow_fts),
                          'Vaud': (canton_file_Nbelow, canton_file_Nbelow_fts)}}
    name_var = {'Nabove': 'clm_days-above-24[days]', 'Nbelow': 'clm_days-below-15[days]'}

    for N, country_file in mapping.items():
        for country, files in country_file.items():
            dm_ots = extract_Yasser_dataset(files[0], name_var[N], country)
            dm_fts = extract_Yasser_dataset(files[1], name_var[N], country)
            dm = join_ots_fts_Yasser_dataset(dm_ots, dm_fts, years_fts)
            if 'dm_all' in locals():
                dm_all.append(dm, dim='Country')
            else:
                dm_all = dm
        if 'dm_N' in locals():
            dm_N.append(dm_all, dim='Variables')
        else:
            dm_N = dm_all.copy()
        del dm_all
    return dm_N


def dummy_update_DM_module_baseyear(DM_old):

    DM_new = {'ots': dict(), 'fts': dict()}
    # key = 'fts', 'ots'
    for lever in DM_old['ots']:
        DM_new['ots'][lever] = dict()
        DM_new['fts'][lever] = dict()
        for dm_name in DM_old['ots'][lever]:
            dm_ots = DM_old['ots'][lever][dm_name].copy()
            dm_fts = DM_old['fts'][lever][dm_name][1]
            years_ots_missing = list(set(years_ots) - set(dm_ots.col_labels['Years']))
            dm_ots.add(np.nan, dummy=True, dim='Years', col_label=years_ots_missing)
            dm_ots.sort('Years')
            first_fts_year = dm_fts.col_labels['Years'][0]
            if first_fts_year in years_ots:
                idx = dm_ots.idx
                dm_ots.array[:, idx[first_fts_year], ...] = dm_fts.array[:, 0, ...]
            linear_fitting(dm_ots, dm_ots.col_labels['Years'])
            DM_new['ots'][lever][dm_name] = dm_ots
            DM_new['fts'][lever][dm_name] = dict()
            for level in range(4):
                level = level + 1
                dm_fts_old = DM_old['fts'][lever][dm_name][level]
                DM_new['fts'][lever][dm_name][level] = dm_fts_old.filter({'Years': years_fts})

    return DM_new


def filter_country_DM(cntr_list, DM):

    for lever in DM['ots']:
        for dm_name in DM['ots'][lever]:
            dm_ots = DM['ots'][lever][dm_name]
            dm_ots.filter({'Country': cntr_list}, inplace=True)
            DM['ots'][lever][dm_name] = dm_ots
            for level in range(4):
                level = level + 1
                dm_fts = DM['fts'][lever][dm_name][level]
                dm_fts.filter({'Country': cntr_list}, inplace=True)
                DM['fts'][lever][dm_name][level] = dm_fts

    return DM


def compute_cooling_uptake(dm_CDD_ots, dict_CDD_fts):
    def strictly_increasing(dm):
        idx = dm.idx
        for yr in dm.col_labels['Years'][0:-1]:
            dm.array[:, idx[yr]+1, ...] = np.maximum(dm.array[:, idx[yr]+1, ...], dm.array[:, idx[yr], ...])
        return dm

    ####   Cooling uptake   ####
    # uptake = 0.95/ (1 + e−((CDD−700) / 150))
    # uptake = 0.815 x (1 − e−0.00225 CDD)
    # arr = 0.95/(1 + np.exp(-(dm_clm.array[:, :, idx_c['clm_CDD']] - 700)/150))
    idx = dm_CDD_ots.idx
    arr = 0.815 * (1 - np.exp(-0.00225 * dm_CDD_ots.array[:, :, idx['clm_CDD']]))
    idx_ots = [idx[yr] for yr in years_ots]
    arr_ots = arr[:, idx_ots]
    for i in range(10):
        window_size = 5
        data_smooth = moving_average(arr_ots, window_size, axis=1)
        arr_ots[:, 2:-2, ...] = data_smooth
        arr_ots[:, 0, ...] = data_smooth[:, 0]
        arr_ots[:, 1, ...] = data_smooth[:, 0]
        arr_ots[:, -1, ...] = data_smooth[:, -1]
        arr_ots[:, -2, ...] = data_smooth[:, -1]
    arr[:, idx_ots[2:-2]] = arr_ots[:, 2:-2, ...]
    for i in [0, 1, -1, -2]:
        arr[:, idx_ots[i]] = np.nan
    dm_CDD_ots.add(arr, dim='Variables', col_label='clm_AC-uptake', dummy=True, unit='%')
    linear_fitting(dm_CDD_ots, dm_CDD_ots.col_labels['Years'])
    dm_uptake_ots = dm_CDD_ots.filter({'Variables': ['clm_AC-uptake']})
    dm_uptake_ots = strictly_increasing(dm_uptake_ots)

    dict_uptake_fts = dict()
    for lev in range(4):
        lev = lev + 1
        dm_CDD_fts = dict_CDD_fts[lev].copy()
        idx = dm_CDD_fts.idx
        arr = 0.815 * (1 - np.exp(-0.00225 * dm_CDD_fts.array[:, :, idx['clm_CDD']]))
        arr[:, 0:-1] = np.nan
        dm_CDD_fts.add(arr, dim='Variables', col_label='clm_AC-uptake', unit='%')
        dm_uptake_fts = dm_CDD_fts.filter({'Variables': ['clm_AC-uptake']})
        dm_uptake_fts.append(dm_uptake_ots, dim='Years')
        dm_uptake_fts.sort('Years')
        dm_uptake_fts.fill_nans('Years')
        dm_uptake_fts = strictly_increasing(dm_uptake_fts)
        dm_uptake_fts.filter({'Years': years_fts}, inplace=True)
        dict_uptake_fts[lev] = dm_uptake_fts

    return dm_uptake_ots, dict_uptake_fts



years_ots = create_years_list(1990, 2023, 1)
years_fts = create_years_list(2025, 2050, 5)

dm_HCDD = extract_CDD_HDD_from_Yasser_dataset(years_fts)

dm_N = extract_days_Tbase_from_Yasser_dataset(years_fts)

dm_all = dm_HCDD.copy()
dm_all.append(dm_N, dim='Variables')


file = '../../../data/datamatrix/climate.pickle'
with open(file, 'rb') as handle:
    DM_climate_old = pickle.load(handle)

DM_climate_new = dummy_update_DM_module_baseyear(DM_climate_old)

DM_climate_new = filter_country_DM(['Switzerland', 'Vaud'], DM_climate_new)

DM_climate_new['ots']['temp']['bld_climate-impact-space'] = dm_all.filter({'Years': years_ots})
for lev in range(4):
    lev = lev + 1
    DM_climate_new['fts']['temp']['bld_climate-impact-space'][lev] = dm_all.filter({'Years': years_fts})

#########################
####    AC UPTAKE    ####
#########################
dict_fts = DM_climate_new['fts']['temp']['bld_climate-impact-space'].copy()
dm_ots = DM_climate_new['ots']['temp']['bld_climate-impact-space'].copy()
dm_uptake_ots, dict_uptake_fts = compute_cooling_uptake(dm_ots, dict_fts)
DM_climate_new['ots']['temp']['bld_climate-impact-space'].append(dm_uptake_ots, dim='Variables')
for lev in range(4):
    lev = lev + 1
    DM_climate_new['fts']['temp']['bld_climate-impact-space'][lev].append(dict_uptake_fts[lev], dim='Variables')

file = '../../../data/datamatrix/climate.pickle'
with open(file, 'wb') as handle:
    pickle.dump(DM_climate_new, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Hello')