import numpy as np
import pandas as pd
import pickle

from model.common.auxiliary_functions import my_pickle_dump
from _database.pre_processing.api_routines_CH import get_data_api_CH
from model.common.data_matrix_class import DataMatrix
from model.common.constant_data_matrix_class import ConstantDataMatrix

import os

def extract_heating_demand(table_id, file):

    try:
        with open(file, 'rb') as handle:
            dm_heating = pickle.load(handle)
    except OSError:

        structure, title = get_data_api_CH(table_id, mode='example', language='fr')

        filter = {'Économie et ménages': ['--- Chauffage des ménages'],
                  'Unité de mesure': ['Térajoules'],
                  'Année': structure['Année'],
                  'Agent énergétique': structure['Agent énergétique']}

        mapping = {'Country': 'Unité de mesure',
                   'Years': 'Année',
                   'Variables': 'Économie et ménages',
                   'Categories1': 'Agent énergétique'}

        dm_heating = get_data_api_CH(table_id, mode='extract', mapping_dims=mapping, filter=filter, units=['TJ'], language='fr')

        dm_heating.rename_col('--- Chauffage des ménages', 'bld_heating-demand', dim='Variables')
        dm_heating.rename_col('Térajoules', 'Switzerland', dim='Country')

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, 'wb') as handle:
            pickle.dump(dm_heating, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dict_rename = {'heating-oil': ['1.1.2. Huile de chauffage extra-légère'],
                   'coal': ['1.2. Charbon'], 'gas': ['1.3. Gaz naturel'],
                   'district-heating': ['2. Déchets (hors biomasse)', '3.1. Déchets (biomasse)',
                                        '4. Combustibles nucléaires', '6. Chaleur à distance'],
                   'wood': ['3.2. Bois et charbon de bois'], 'biofuel': ['3.3. Biogaz et biocarburants'],
                   'renewables': ['3.4. Géothermie, chaleur ambiante et énergie solaire thermique'],
                   'electricity': ['5. Electricité']}

    dm_heating = dm_heating.groupby(dict_rename, dim='Categories1', inplace=False)

    dm_heating.change_unit('bld_heating-demand', 3600, old_unit='TJ', new_unit='TWh', operator='/')

    return dm_heating


# SECTION calibration Heating demand
# Energy demand for heating in building sector
table_id = 'px-x-0204000000_106'
file = 'data/bld_heating-energy-demand.pickle'
dm_heating = extract_heating_demand(table_id, file)

# Read run output
file = 'data/heating_energy.pickle'
with open(file, 'rb') as handle:
    dm_energy = pickle.load(handle)

dm_energy.group_all('Categories2')
dm_energy.group_all('Categories1')
dm_energy.filter({'Country': ['Switzerland']}, inplace=True)
matching_cat = list(set(dm_energy.col_labels['Categories1']).intersection(set(dm_heating.col_labels['Categories1'])))
dm_energy.filter({'Categories1': matching_cat, 'Years': dm_heating.col_labels['Years']}, inplace=True)
dm_heating.filter({'Categories1': matching_cat}, inplace=True)
dm_energy.append(dm_heating, dim='Variables')
dm_energy.rename_col(['bld_heating-demand', 'bld_energy-demand_heating'],
                    ['FSO_energy-demand-heating', 'My_energy-demand-heating'], dim='Variables')
dm_energy.operation('FSO_energy-demand-heating', '/', 'My_energy-demand-heating', out_col='bld_heating-energy-calibration', unit='%')

dm_compare = dm_energy.filter({'Variables': ['FSO_energy-demand-heating', 'My_energy-demand-heating'],
                               'Categories1': ['gas', 'heating-oil']}, inplace=False)
dm_compare.datamatrix_plot()
dm_energy.datamatrix_plot()

calibrate = False
if calibrate:
    file = '../../../data/datamatrix/buildings.pickle'
    with open(file, 'rb') as handle:
        DM_bld = pickle.load(handle)

    years_all = DM_bld['ots']['heating-efficiency'].col_labels['Years'] + DM_bld['fts']['heating-efficiency'][1].col_labels['Years']
    dm_energy_calibration = dm_energy.filter({'Variables': ['bld_heating-energy-calibration']}, inplace=False)
    dm_avg_calib = dm_energy_calibration.groupby({0: dm_energy_calibration.col_labels['Years']}, dim='Years', inplace=False)
    dm_avg_calib.array = dm_avg_calib.array/len(dm_energy_calibration.col_labels['Years'])
    # Do not calibrate coal, district-heating, electricity put 1
    idx = dm_avg_calib.idx
    fuels = ['coal', 'district-heating', 'electricity']
    for fuel in fuels:
        dm_avg_calib.array[:, :, :, idx[fuel]] = 1
    all_fuels = DM_bld['ots']['heating-efficiency'].col_labels['Categories2']
    missing_fuels = list(set(all_fuels) - set(dm_avg_calib.col_labels['Categories1']))
    dm_avg_calib.add(1, dummy=True, dim='Categories1', col_label=missing_fuels)
    dm_avg_calib.sort('Categories1')
    # (self, col_labels={}, units={}, idx={})
    dm_avg_calib.add(np.nan, dummy=True, dim='Country', col_label='Vaud')
    dm_avg_calib.fill_nans(dim_to_interp='Country')
    dm_avg_calib.add(np.nan, dummy=True, dim='Years', col_label=years_all)
    dm_avg_calib.fill_nans(dim_to_interp='Years')
    dm_avg_calib.drop(col_label=0, dim='Years')
    DM_bld['fxa']['heating-energy-calibration'] = dm_avg_calib


    file = '../../../data/datamatrix/buildings.pickle'
    my_pickle_dump(DM_bld, file)


# If the values are off and the calibration factors vary a lot, try to use the heating demand to back calculate the efficiencies
# E_out = E_in * eff
# heat_demand = E_in * eff ->    eff = heat_demand / E_in
#dm_eff_JRC = DM_bld['ots']['heating-efficiency'].copy()
#dm_eff_JRC.filter({'Years': dm_energy.col_labels['Years'], 'Categories1': dm_energy.col_labels['Categories1'],
#                   'Country': dm_energy.col_labels['Country']}, inplace=True)
#dm_eff_JRC.rename_col('bld_heating-efficiency', 'bld_efficiency_JRC', 'Variables')
#dm_eff_JRC.append(dm_energy.filter({'Variables': ['bld_efficiency']}), dim='Variables')

#dm_eff_JRC.datamatrix_plot()

print('Hello')