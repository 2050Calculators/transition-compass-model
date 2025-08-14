import pickle
import os
from _database.pre_processing.api_routines_CH import get_data_api_CH
from model.common.auxiliary_functions import create_years_list
import numpy as np

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
        #filter['Catégorie de bâtiment'] = ["Bâtiments partiellement à usage d'habitation", "Bâtiments d'habitation avec usage annexe"]
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


years_ots = create_years_list(1990, 2023, 1)
years_fts = create_years_list(2025, 2050, 5)


file_bld = '../../../data/interface/buildings_to_energy.pickle'
with open(file_bld, 'rb') as handle:
    dm_bld = pickle.load(handle)

dm_eff = dm_bld.copy()
dm_eff.filter({'Country': ['Switzerland']}, inplace=True)
dm_eff.operation('bld_heating', '/', 'bld_energy-demand_heating', out_col='bld_efficiency', unit='%')
dm_eff.filter({'Variables': ['bld_efficiency']}, inplace=True)
dm_eff.fill_nans('Years')
dm_eff[:, :, 'bld_efficiency', 'electricity'] = 1.0


# Get Hot water fuel split at household level per canton
table_id = 'px-x-0902010000_102'
file_hw = 'data/bld_hotwater_technology_2021-2023.pickle'
# Extract water tech share based on number of buildings
dm_water = extract_hotwater_technologies(table_id, file_hw)  # tech share
# Add missing years
dm_water.add(np.nan, dummy=True, dim='Years', col_label=list(set(years_ots)-set(dm_water.col_labels['Years'])))
dm_water.sort('Years')
dm_water.fill_nans('Years')
## Get water tech share based on energy consumption
#dm_water = adjust_based_on_efficiency(dm_water, dm_eff, years_ots)
# Hot water demand in CH - use efficiency to determine useful energy demand

# Split hot water useful energy demand in CH to canton using same per capita

# Apply tech share by bld and then adjust using efficiency

# Adjust so that sum of canton of energy consumption by fuel is correct


# Get lighting



print('Hello')