from model.common.data_matrix_class import DataMatrix
import numpy as np
from _database.pre_processing.api_routines_CH import get_data_api_CH
import pickle
import os


def extract_national_energy_demand(table_id, file):
  try:
    with open(file, 'rb') as handle:
      dm_energy = pickle.load(handle)
  except OSError:

    structure, title = get_data_api_CH(table_id, mode='example', language='fr')

    # Remove freight transport energy demand
    exclude_list = ['49', '50', '51', '52', '53']
    keep_sectors = [s for s in structure['Économie et ménages'] if
                    not any(n in s for n in exclude_list)]
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

    dm_energy = get_data_api_CH(table_id, mode='extract', mapping_dims=mapping,
                                filter=filter,
                                units=['TJ'] * len(keep_sectors), language='fr')

    # dm_heating.rename_col('--- Chauffage des ménages', 'bld_heating-demand', dim='Variables')
    dm_energy.rename_col('Térajoules', 'Switzerland', dim='Country')

    # We drop the fuels fro transport
    dict_rename = {'heating-oil': ['1.1.2. Huile de chauffage extra-légère'],
                   'coal': ['1.2. Charbon'], 'gas': ['1.3. Gaz naturel'],
                   'district-heating': ['6. Chaleur à distance'],
                   'nuclear-fuel': ['4. Combustibles nucléaires'],
                   'waste': ['2. Déchets (hors biomasse)'],
                   'biomass': ['3.1. Déchets (biomasse)'],
                   'wood': ['3.2. Bois et charbon de bois'],
                   'biogas': ['3.3. Biogaz et biocarburants'],
                   'renewables': [
                     '3.4. Géothermie, chaleur ambiante et énergie solaire thermique'],
                   'electricity': ['5. Electricité']}

    dm_energy = dm_energy.groupby(dict_rename, dim='Categories1', inplace=False)

    for var in dm_energy.col_labels['Variables']:
      dm_energy.rename_col(var, 'bld_energy-by-sector_' + var, dim='Variables')

    dm_energy.deepen(based_on='Variables')
    dm_energy.switch_categories_order()

    dm_energy.change_unit('bld_energy-by-sector', 3600, old_unit='TJ',
                          new_unit='TWh', operator='/')

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, file)
    with open(f, 'wb') as handle:
      pickle.dump(dm_energy, handle, protocol=pickle.HIGHEST_PROTOCOL)
  dm_energy.rename_col_regex('-', '', dim='Categories1')
  dm_energy.drop('Categories1', " 35 Production et distribution d'énergie")
  # Remove demand of waste for waste management sector because it is computed in EnergyScope
  # Biomass is also from waste
  dm_energy[:, :, :,
  " 3639 Production et distribution d'eau; gestion des déchets", 'waste'] = 0
  dm_energy[:, :, :,
  " 3639 Production et distribution d'eau; gestion des déchets", 'biomass'] = 0
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

