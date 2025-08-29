import numpy as np
import pandas as pd
import pickle
import time
from model.common.auxiliary_functions import linear_fitting, rename_cantons, dm_add_missing_variables, save_url_to_file
from _database.pre_processing.api_routines_CH import get_data_api_CH
from model.common.data_matrix_class import DataMatrix
from model.common.constant_data_matrix_class import ConstantDataMatrix

import os
import zipfile


def df_excel_to_dm(df, names_dict, var_name, unit, num_cat, keep_first=False, country='Switzerland'):
    # df from excel to dm
    # Remove nans and empty columns/rows
    if np.nan in df.columns:
        df.drop(columns=np.nan, inplace=True)
    # Change headers
    df.rename(columns={df.columns[0]: 'Variables'}, inplace=True)
    df.set_index('Variables', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    # Filter rows that contain at least one number (integer or float)
    df = df[df.apply(lambda row: row.map(pd.api.types.is_number), axis=1).any(axis=1)]
    df_clean = df.loc[:, df.apply(lambda col: col.map(pd.api.types.is_number)).any(axis=0)].copy()
    # Extract only the data we are interested in:
    df_filter = df_clean.loc[names_dict.keys()].copy()
    df_filter = df_filter.apply(lambda col: pd.to_numeric(col, errors='coerce'))
    #df_filter = df_filter.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
    df_filter.reset_index(inplace=True)
    # Keep only first 10 caracters
    df_filter['Variables'] = df_filter['Variables'].replace(names_dict)
    if keep_first:
        df_filter = df_filter.drop_duplicates(subset=['Variables'], keep='first')
    df_filter = df_filter.groupby(['Variables']).sum()
    df_filter.reset_index(inplace=True)

    # Pivot the dataframe
    df_filter['Country'] = country
    df_T = pd.melt(df_filter, id_vars=['Variables', 'Country'], var_name='Years', value_name='values')
    df_pivot = df_T.pivot_table(index=['Country', 'Years'], columns=['Variables'], values='values', aggfunc='sum')
    df_pivot = df_pivot.add_suffix('[' + unit + ']')
    df_pivot = df_pivot.add_prefix(var_name + '_')
    df_pivot.reset_index(inplace=True)

    # Drop non numeric values in Years col
    df_pivot['Years'] = pd.to_numeric(df_pivot['Years'], errors='coerce')
    df_pivot = df_pivot.dropna(subset=['Years'])

    dm = DataMatrix.create_from_df(df_pivot, num_cat=num_cat)
    return dm

def extract_heating_technologies_old(table_id, file, cat_sfh, cat_mfh):
  # Domaine de l'énergie: bâtiments selon le canton, le type de bâtiment, l'époque de construction, le type de chauffage,
  # la production d'eau chaude, les agents énergétiques utilisés pour le chauffage et l'eau chaude, 1990 et 2000
  try:
    with open(file, 'rb') as handle:
        dm_heating_old = pickle.load(handle)
  except OSError:
    structure, title = get_data_api_CH(table_id, mode='example', language='fr')
    # Extract buildings floor area
    dm_heating_old = None
    for cntr in structure['Canton']:
      filter = structure.copy()
      filter['Canton'] = [cntr]
      mapping_dim = {'Country': 'Canton', 'Years': 'Année',
                     'Variables': 'Epoque de construction', 'Categories1': 'Type de bâtiment',
                     'Categories2': 'Agent énergétique pour le chauffage'}
      dm_heating_old_cntr = None
      tot_bld = 0
      for t in structure['Type de chauffage']:
        for a in structure["Agent énergétique pour l'eau chaude"]:
          filter['Type de chauffage'] = [t]
          filter["Agent énergétique pour l'eau chaude"] = [a]
          unit_all = ['number'] * len(structure['Epoque de construction'])
          dm_heating_old_t = get_data_api_CH(table_id, mode='extract', filter=filter,
                                          mapping_dims=mapping_dim, units=unit_all, language='fr')

          if dm_heating_old_cntr is None:
              dm_heating_old_cntr = dm_heating_old_t.copy()
          else:
              dm_heating_old_cntr.array = dm_heating_old_t.array + dm_heating_old_cntr.array
          partial_bld = np.nansum(dm_heating_old_t.array[0, 0, ...])
          tot_bld = tot_bld + partial_bld
          print(t, a, partial_bld, tot_bld)
          time.sleep(1.5)
      #dm_heating_old_cntr.rename_col(['Suisse'], ['Switzerland'], dim='Country')
      dm_heating_old_cntr.groupby({'single-family-households': ['Maisons individuelles'],
                             'multi-family-households': ['Maisons à plusieurs logements',
                                                         "Bâtiments d'habitation avec usage annexe",
                                                         "Bâtiments partiellement à usage d'habitation"]},
                              dim='Categories1', inplace=True)

      if dm_heating_old is None:
        dm_heating_old = dm_heating_old_cntr.copy()
      else:
        dm_heating_old.append(dm_heating_old_cntr, dim='Country')

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, file)
    with open(f, 'wb') as handle:
        pickle.dump(dm_heating_old, handle, protocol=pickle.HIGHEST_PROTOCOL)

  dm_heating_old.rename_col_regex('Construits ', '', dim='Variables')
  dm_heating_old.rename_col_regex('entre ', '', dim='Variables')
  dm_heating_old.rename_col_regex(' et ', '-', dim='Variables')
  dm_heating_old.rename_col_regex('avant ', 'Avant ', dim='Variables')
  dm_heating_old.groupby({'1991-2000': ['1991-1995', '1996-2000']}, dim='Variables', inplace=True)

  # Group by construction period
  dm_heating_sfh = dm_heating_old.filter({'Categories1': ['single-family-households']}, inplace=False)
  dm_heating_sfh.groupby({'bld_heating-mix_F':  cat_sfh['F']}, dim='Variables', inplace=True)
  dm_heating_sfh.groupby({'bld_heating-mix_E': cat_sfh['E']}, dim='Variables', inplace=True)
  dm_heating_sfh.groupby({'bld_heating-mix_D': cat_sfh['D']}, dim='Variables', inplace=True)

  dm_heating_mfh = dm_heating_old.filter({'Categories1': ['multi-family-households']}, inplace=False)
  dm_heating_mfh.groupby({'bld_heating-mix_F': cat_mfh['F']}, dim='Variables', inplace=True)
  dm_heating_mfh.groupby({'bld_heating-mix_E': cat_mfh['E']}, dim='Variables', inplace=True)
  dm_heating_mfh.groupby({'bld_heating-mix_D': cat_mfh['D']}, dim='Variables', inplace=True)

  # Merge sfh and mfh
  dm_heating_mfh.append(dm_heating_sfh, dim='Categories1')
  dm_heating_old = dm_heating_mfh

  dm_heating_old.groupby({'other-tech': ['Autre agent énergétique (chauf.)', 'Sans chauffage']},
                         dim='Categories2', inplace=True)
  dm_heating_old.rename_col(['Mazout (chauf.)', 'Bois (chauf.)', 'Pompe à chaleur (chauf.)',
                             'Electricité (chauf.)', 'Gaz (chauf.)', 'Chaleur à distance (chauf.)',
                             'Charbon (chauf.)', 'Capteur solaire (chauf.)'],
                            ['heating-oil', 'wood', 'heat-pump', 'electricity', 'gas', 'district-heating', 'coal', 'solar'],
                            dim='Categories2')

  dm_heating_old.deepen(based_on='Variables')

  rename_cantons(dm_heating_old)
  dm_heating_old.rename_col('Suisse', 'Switzerland', 'Country')
  return dm_heating_old



def extract_heating_technologies(table_id, file, cat_sfh, cat_mfh):

    def extract_cntr_heating(cntr_list):
      filter = {'Année': structure['Année'],
                'Canton': cntr_list,
                "Source d'énergie du chauffage": structure[
                  "Source d'énergie du chauffage"],
                "Source d'énergie de l'eau chaude": structure[
                  "Source d'énergie de l'eau chaude"],
                'Époque de construction': structure['Époque de construction'],
                'Catégorie de bâtiment': structure['Catégorie de bâtiment']}
      mapping_dim = {'Country': 'Canton', 'Years': 'Année',
                     'Variables': 'Époque de construction',
                     'Categories1': 'Catégorie de bâtiment',
                     'Categories2': "Source d'énergie du chauffage"}
      unit_all = ['number'] * len(structure['Époque de construction'])
      # Get api data
      dm_heating_cntr = get_data_api_CH(table_id, mode='extract', filter=filter,
                                        mapping_dims=mapping_dim,
                                        units=unit_all, language='fr')
      return dm_heating_cntr

    try:
        with open(file, 'rb') as handle:
            dm_heating = pickle.load(handle)
    except OSError:
        structure, title = get_data_api_CH(table_id, mode='example', language='fr')
        # Extract buildings floor area
        dm_heating=None
        # There is a problem with Neuchatel and it cannot be extracted as standalone
        cantons_list = list(set(structure['Canton']) - {'Suisse',  'Neuchâtel', 'St. Gallen'})
        for cntr in cantons_list:
          cntr_list = [cntr]

          dm_heating_cntr = extract_cntr_heating(cntr_list)
          if dm_heating is None:
            dm_heating = dm_heating_cntr
          else:
            dm_heating.append(dm_heating_cntr, dim='Country')

        dm_heating_cntr = extract_cntr_heating(['Suisse',  'Neuchâtel', 'St. Gallen'])
        dm_heating.append(dm_heating_cntr, dim='Country')

        dm_heating.rename_col(['Suisse'], ['Switzerland'], dim='Country')
        dm_heating.groupby({'single-family-households': ['Maisons individuelles'],
                               'multi-family-households': ['Maisons à plusieurs logements',
                                                           "Bâtiments d'habitation avec usage annexe",
                                                           "Bâtiments partiellement à usage d'habitation"]},
                              dim='Categories1', inplace=True)

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, 'wb') as handle:
            pickle.dump(dm_heating, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Group by construction period
    dm_heating_sfh = dm_heating.filter({'Categories1': ['single-family-households']}, inplace=False)
    cat_sfh = {'bld_heating-mix_' + key: value for key, value in cat_sfh.items()}
    dm_heating_sfh.groupby(cat_sfh, dim='Variables', inplace=True)

    dm_heating_mfh = dm_heating.filter({'Categories1': ['multi-family-households']}, inplace=False)
    cat_sfh = {'bld_heating-mix_' + key: value for key, value in cat_mfh.items()}
    dm_heating_mfh.groupby(cat_sfh, dim='Variables', inplace=True)
    # Merge sfh and mfh
    dm_heating_mfh.append(dm_heating_sfh, dim='Categories1')
    dm_heating = dm_heating_mfh

    dm_heating.groupby({'Other': ['Autre', 'Aucune']}, dim='Categories2', inplace=True)
    dm_heating.rename_col(
        ['Bois', 'Chaleur produite à distance', 'Electricité', 'Gaz', 'Mazout', 'Other', 'Pompe à chaleur', 'Soleil (thermique)'],
        ['wood', 'district-heating', 'electricity', 'gas', 'heating-oil', 'other-tech', 'heat-pump', 'solar'], dim='Categories2')

    dm_heating.deepen(based_on='Variables')

    rename_cantons(dm_heating)
    return dm_heating




def prepare_heating_mix_by_archetype():
    dict_sfh = {'F': {'heating-oil': 0.48,
                      'wood': 0.19,
                      'gas': 0.14,
                      'heat-pump': 0.09,
                      'electricity': 0.08,
                      'district-heating': 0.02,
                      'solar': 0},
                'E': {'heating-oil': 0.51,
                      'wood': 0.05,
                      'gas': 0.07,
                      'heat-pump': 0.15,
                      'electricity': 0.20,
                      'district-heating': 0.02,
                      'solar': 0},
                'D': {'heating-oil': 0.26,
                      'wood': 0.05,
                      'gas': 0.19,
                      'heat-pump': 0.23,
                      'electricity': 0.26,
                      'district-heating': 0.01,
                      'solar': 0},
                'C': {'heating-oil': 0.25,
                      'wood': 0.04,
                      'gas': 0.36,
                      'heat-pump': 0.32,
                      'electricity': 0.02,
                      'district-heating': 0.01,
                      'solar': 0},
                'B': {'heating-oil': 0.01,
                      'wood': 0.07,
                      'gas': 0.13,
                      'heat-pump': 0.73,
                      'electricity': 0.0,
                      'district-heating': 0.05,
                      'solar': 0.01}
                }

    dict_mfh = {'F': {'heating-oil': 0.57,
                      'wood': 0.02,
                      'gas': 0.27,
                      'heat-pump': 0.03,
                      'electricity': 0.03,
                      'district-heating': 0.08,
                      'solar': 0},
                'E': {'heating-oil': 0.36,
                      'wood': 0.01,
                      'gas': 0.43,
                      'heat-pump': 0.04,
                      'electricity': 0.11,
                      'district-heating': 0.05,
                      'solar': 0},
                'D': {'heating-oil': 0.37,
                      'wood': 0.02,
                      'gas': 0.50,
                      'heat-pump': 0.04,
                      'electricity': 0.03,
                      'district-heating': 0.04,
                      'solar': 0},
                'C': {'heating-oil': 0.16,
                      'wood': 0.04,
                      'gas': 0.64,
                      'heat-pump': 0.09,
                      'electricity': 0,
                      'district-heating': 0.07,
                      'solar': 0},
                'B': {'heating-oil': 0.01,
                      'wood': 0.10,
                      'gas': 0.27,
                      'heat-pump': 0.47,
                      'electricity': 0.0,
                      'district-heating': 0.14,
                      'solar': 0.01}
                }

    dict_archetype = {'single-family-households': dict_sfh, 'multi-family-households': dict_mfh}

    categories1 = list(dict_archetype.keys())
    categories1.sort()
    categories2 = list(dict_sfh.keys())
    categories2.sort()
    categories3 = list(dict_sfh['F'].keys())
    categories3.sort()

    cdm = ConstantDataMatrix(col_labels={'Variables': ['bld_heating-mix-archetype'],
                                         'Categories1': categories1,
                                         'Categories2': categories2,
                                         'Categories3': categories3},
                             units={'bld_heating-mix-archetype': '%'})
    idx = cdm.idx
    cdm.array = np.zeros((1, len(categories1), len(categories2), len(categories3)))
    for bld_type, dict_type in dict_archetype.items():
        for env_cat, dict_fuel in dict_type.items():
            for fuel, value in dict_fuel.items():
                cdm.array[idx['bld_heating-mix-archetype'], idx[bld_type], idx[env_cat], idx[fuel]] = value

    return cdm

def compute_heating_mix_F_E_D_categories(dm_heating_tech, dm_heating_tech_old, years_ots):
    # For categories existing in before 2000 (in dm_heating_tech_old) merge with new data and normalise
    dm_heating_tech.switch_categories_order('Categories3', 'Categories1')
    dm_heating_tech.switch_categories_order('Categories2', 'Categories3')
    dm_heating_tech_old.switch_categories_order('Categories3', 'Categories1')
    dm_heating_tech_old.switch_categories_order('Categories2', 'Categories3')
    dm_tmp = dm_heating_tech.filter({'Categories1': dm_heating_tech_old.col_labels['Categories1']})
    dm_heating_tech_old.append(dm_tmp, dim='Years')
    #dm_heating_tech_old.normalise(dim='Categories3')
    # Remove "D" values at 0 in 1990 and use fill_nans to fill
    idx = dm_heating_tech_old.idx
    dm_heating_tech_old.array[:, idx[1990], :, idx['D'], idx['multi-family-households'], :] = np.nan
    dm_heating_tech_old.fill_nans('Years')
    linear_fitting(dm_heating_tech_old, years_ots)
    #dm_heating_tech_old.normalise('Categories3')
    return dm_heating_tech_old

def compute_heating_mix_C_B_categories(dm_heating_tech, cdm_heating_archetypes, years_ots, envelope_cat_new):

    dm_heating_tech_new = dm_heating_tech.filter({'Categories1': ['B', 'C']}, inplace=False)
    # In order to extrapolate C category for the missing years, since things change rapidely in 2021-2023,
    # I use the archetype paper to fix the values at the beginning of the construction period (this is a bit of a misuse)

    # normalise over fuel technology
    dm_heating_tech_new.normalise('Categories3')
    # add missing years as nan
    dm_add_missing_variables(dm_heating_tech_new, {'Years': years_ots}, fill_nans=False)
    # replace the values at the beginning of the construction period for C with the archetypes values
    extra_cat = list(set(dm_heating_tech_new.col_labels['Categories3']) - set(cdm_heating_archetypes.col_labels['Categories3']))
    dm_heating_tech_new.filter({'Categories3': cdm_heating_archetypes.col_labels['Categories3']}, inplace=True)
    cdm_heating_archetypes.sort('Categories3')
    dm_heating_tech_new.sort('Categories3')
    idx = dm_heating_tech_new.idx
    idx_c = cdm_heating_archetypes.idx
    for cat in dm_heating_tech_new.col_labels['Categories1']:
        start_year = envelope_cat_new[cat][0]
        dm_heating_tech_new.array[:, idx[start_year], 0, idx[cat], idx['multi-family-households'], :] \
            = cdm_heating_archetypes.array[np.newaxis, 0, idx['multi-family-households'], idx_c[cat], :]
        dm_heating_tech_new.array[:, idx[start_year], :, idx[cat], idx['single-family-households'], :] \
            = cdm_heating_archetypes.array[np.newaxis, 0, idx['single-family-households'], idx_c[cat], :]
    linear_fitting(dm_heating_tech_new, years_ots)

    idx = dm_heating_tech_new.idx
    for cat in dm_heating_tech_new.col_labels['Categories1']:
        period = envelope_cat_new[cat]
        start_year = period[0]
        before_period = [idx[yr] for yr in years_ots if yr < start_year]
        dm_heating_tech_new.array[:, before_period, :, idx[cat], :, :] = np.nan
    dm_heating_tech_new.normalise('Categories3')

    dm_heating_tech_new.add(0, dummy=True, dim='Categories3', col_label=extra_cat)
    return dm_heating_tech_new


def compute_heating_mix_by_category(dm_heating_tech, cdm_heating_archetypes, dm_all):
    # dm_heating_tech: share of building per heating technology, by sfh and mfh
    # cdm_heating_archetypes: portion of households per heating technology, envelope, sfh and mfh
    # I use the stock from dm_all
    # I want to compute the heating mix by sfh/mfh and envelope.

    dm_stock = dm_all.filter({'Variables': ['bld_floor-area_stock']})

    # STEP 1: obtain a heating-mix that sums to 1 over the envelope categories
    idx_h = cdm_heating_archetypes.idx
    idx_a = dm_stock.idx
    arr_tmp = cdm_heating_archetypes.array[np.newaxis, np.newaxis, idx_h['bld_heating-mix-archetype'], :, :, :] \
              * dm_stock.array[:, :, idx_a['bld_floor-area_stock'], :, :, np.newaxis]
    arr = arr_tmp/np.nansum(arr_tmp, axis=-2, keepdims=True)
    dm_heating_mix = DataMatrix.based_on(arr[:, :, np.newaxis, ...], dm_stock, units={'bld_heating-mix-tmp': '%'},
                                         change={'Variables': ['bld_heating-mix-tmp'],
                                                 'Categories3': cdm_heating_archetypes.col_labels['Categories3']})

    dm_stock_norm = dm_stock.normalise('Categories2', inplace=False)
    fuel_missing = list(set(dm_heating_tech.col_labels['Categories2']) - set(dm_heating_mix.col_labels['Categories3']))
    dm_heating_mix.add(np.nan, dim='Categories3', col_label=fuel_missing, dummy=True)
    idx = dm_heating_mix.idx
    idx_n = dm_stock_norm.idx
    for f in fuel_missing:
        dm_heating_mix.array[:, :, idx['bld_heating-mix-tmp'], :, :, idx[f]] = \
            dm_stock_norm.array[:, :, idx_n['bld_floor-area_stock_share'], :, :]
    dm_heating_mix.sort('Categories3')
    dm_heating_tech.sort('Categories2')
    idx = dm_heating_mix.idx
    idx_t = dm_heating_tech.idx
    arr = dm_heating_mix.array[:, :, idx['bld_heating-mix-tmp'], :, :, :] \
          * dm_heating_tech.array[:, :, idx_t['bld_heating-mix'], :, np.newaxis, :]
    dm_heating_mix.array[:, :, idx['bld_heating-mix-tmp'], :, :, :] = arr

    # Now it should sum to 1 over envelope and fuel-type
    dm_heating_mix.normalise(dim='Categories3')
    dm_heating_mix.rename_col('bld_heating-mix-tmp', 'bld_heating-mix', dim='Variables')

    return dm_heating_mix


def clean_heating_cat(dm_heating_cat, envelope_cat_new):

    years_all = dm_heating_cat.col_labels['Years']
    idx = dm_heating_cat.idx
    for cat, period in envelope_cat_new.items():
        years_period = list(range(period[0], period[1]+1))
        years_not_in_period = list(set(years_all) - set(years_period))
        idx_yrs = [idx[yr] for yr in years_not_in_period]
        dm_heating_cat.array[:, idx_yrs, idx['bld_heating-mix'], :, idx[cat], :] = np.nan
    dm_heating_cat.fill_nans('Years')

    return dm_heating_cat


def extract_heating_efficiency_JRC(file, sheet_name, years_ots):
    df = pd.read_excel(file, sheet_name=sheet_name)
    df = df[0:13].copy()
    names_map = {'Ratio of energy service to energy consumption': 'remove', 'Space heating': 'other-tech', 'Solids': 'coal',
                 'Liquified petroleum gas (LPG)': 'remove', 'Diesel oil': 'heating-oil', 'Natural gas': 'gas',
                 'Biomass': 'wood', 'Geothermal': 'geothermal', 'Distributed heat': 'district-heating',
                 'Advanced electric heating': 'heat-pump', 'Conventional electric heating': 'electricity',
                 'Electricity in circulation': 'remove'}
    dm_heating_eff = df_excel_to_dm(df, names_map, var_name='bld_heating-efficiency', unit='%', num_cat=1)
    dm_heating_eff.drop(col_label='remove', dim='Categories1')
    dm_heating_eff.add(0.8, dim='Categories1', col_label='solar', dummy=True)
    years_missing = list(set(years_ots) - set(dm_heating_eff.col_labels['Years']))
    years_missing_e = [yr for yr in years_missing if yr > dm_heating_eff.col_labels['Years'][0]]
    years_missing_i = [yr for yr in years_missing if yr < dm_heating_eff.col_labels['Years'][0]]
    linear_fitting(dm_heating_eff, years_missing_e, based_on=list(range(2015, dm_heating_eff.col_labels['Years'][-1]+1)))
    dm_heating_eff.add(np.nan, dim='Years', col_label=years_missing_i,  dummy=True)
    dm_heating_eff.sort('Years')
    dm_heating_eff.fill_nans('Years')
    # Add Vaud
    arr = dm_heating_eff.array
    dm_heating_eff.add(arr, dim='Country', col_label='Vaud')
    dm_heating_eff.sort('Country')
    return dm_heating_eff


def compute_heating_efficiency_by_archetype(dm_heating_eff, dm_all, envelope_cat_new, categories):

    arr_w_cat = np.repeat(dm_heating_eff.array[..., np.newaxis], repeats=len(categories), axis=-1)
    dm_eff_cat = DataMatrix.based_on(arr_w_cat, format=dm_heating_eff, change={'Categories2': categories},
                                     units=dm_heating_eff.units)
    dm_eff_cat_raw = dm_eff_cat.copy()
    # Keep only stock split by categories
    dm_stock = dm_all.group_all('Categories1', inplace=False)
    idx = dm_eff_cat.idx
    idx_s = dm_stock.idx
    for cat in categories:
        if cat not in envelope_cat_new.keys():
            dm_eff_cat.array[:, :, :, :, idx[cat]] = dm_eff_cat.array[:, 0, np.newaxis, :, :, idx[cat]]
        else:
            # Compute efficiency as the weighted average of stock efficiency and new efficiency
            start_yr = envelope_cat_new[cat][0]
            end_yr = envelope_cat_new[cat][1]
            for yr in range(start_yr+1, end_yr+1):
                eff_s_tm1 = dm_eff_cat.array[:, idx[yr-1], 0, :, idx[cat]]
                s_tm1 = dm_stock.array[:, idx_s[yr-1], idx_s['bld_floor-area_stock'], np.newaxis, idx_s[cat]]
                new_t = dm_stock.array[:, idx_s[yr], idx_s['bld_floor-area_new'],  np.newaxis, idx_s[cat]]
                eff_n_t = dm_eff_cat.array[:, idx[yr], 0, :, idx[cat]]
                eff_s_t = (eff_n_t * new_t + s_tm1 * eff_s_tm1)/(new_t + s_tm1)
                dm_eff_cat.array[:, idx[yr], 0, :, idx[cat]] = eff_s_t
            # after construction period, fix efficiency to end of construction period
            if (end_yr + 1) in dm_eff_cat.col_labels['Years']:
                dm_eff_cat.array[:, idx[end_yr]:, 0, :, idx[cat]] = dm_eff_cat.array[:, idx[end_yr], np.newaxis, 0, :, idx[cat]]
            if start_yr > dm_eff_cat.col_labels['Years'][0]:
                dm_eff_cat.array[:, 0:idx[start_yr], 0, :, idx[cat]] = np.nan

    # for heating-oil keep original data
    dm_eff_cat.array[:, :, :, idx['heating-oil'], :] = dm_eff_cat_raw.array[:, :, :, idx['heating-oil'], :]

    dm_eff_cat_raw.rename_col('bld_heating-efficiency', 'bld_heating-efficiency-JRC', dim='Variables')
    dm_eff_cat.append(dm_eff_cat_raw, dim='Variables')
    return dm_eff_cat


def extract_heating_technologies_EP2050(file_url, zip_name, file_pickle):

  try:
    with open(file_pickle, 'rb') as handle:
      dm = pickle.load(handle)

  except OSError:

    def format_df_tech_EP250(df, start_row, var_name, unit):
      # Years as int
      df.set_index([df.columns[0]], inplace=True)
      df = df.loc[:, ~df.columns.isna()]
      df.columns = df.columns.astype(int)
      df.reset_index(inplace=True)

      # Change variables names
      df.rename(columns={df.columns[0]: 'Energy_source'}, inplace=True)
      full_name = [var_name + '_'+ cat + '[' + unit + ']' for cat in
                   df['Energy_source']]
      df['Energy_source'] = full_name

      # Pivot
      df_T = df.T
      df_T.columns = df_T.iloc[0]
      df_T = df_T.iloc[1:]
      df_T.reset_index(inplace=True)
      df_T.rename(columns={start_row: 'Years'}, inplace=True)
      df_T['Country'] = 'Switzerland'

      dm = DataMatrix.create_from_df(df_T, num_cat=1)
      return dm

    extract_dir = os.path.splitext(zip_name)[0]  # 'data/EP2050_sectors'
    if not os.path.exists(extract_dir):
      save_url_to_file(file_url, zip_name)

      # Extract the file
      os.makedirs(extract_dir, exist_ok=True)
      with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    file_industry = extract_dir + '/EP2050+_Szenarienergebnisse_Details_Nachfragesektoren/EP2050+_Detailergebnisse 2020-2060_Private Haushalte_alle Szenarien_2022-05-17.xlsx'
    df = pd.read_excel(file_industry, sheet_name='02 Wohnungen')

    df.drop(columns=[df.columns[0], df.columns[2]], inplace = True)

    # Extract shares
    table_title = 'Tabelle 02-01: Entwicklung der Beheizungsstruktur im Gebäudebestand im Szenario ZERO Basis'
    start_table_row = df.index[df['Unnamed: 1'] == table_title].tolist()[1]
    col_row = start_table_row+2
    df.columns = df.iloc[col_row]

    df_single = df.iloc[start_table_row + 4:start_table_row + 4+7]
    dm_single = format_df_tech_EP250(df_single, col_row, var_name = 'bld_heating-mix_single-family-households', unit='%')
    dm_single.deepen(based_on='Variables')
    dm_single.switch_categories_order()
    dm_single.rename_col(['Fernwärme/Nahwärme', 'Gas', 'Heizöl', 'Holz', 'Strom', 'Wärmepumpen', 'sonstige'],
                        ['district-heating', 'gas', 'heating-oil', 'wood', 'electricity', 'heat-pump', 'other'], dim='Categories2')
    dm_single.sort('Categories1')

    df_multi = df.iloc[start_table_row+12: start_table_row+12+7]
    dm_multi = format_df_tech_EP250(df_multi, col_row, var_name = 'bld_heating-mix_multi-family-households', unit='%')
    dm_multi.deepen(based_on='Variables')
    dm_multi.switch_categories_order()
    dm_multi.rename_col(['Fernwärme/Nahwärme', 'Gas', 'Heizöl', 'Holz', 'Strom', 'Wärmepumpen', 'sonstige'],
                        ['district-heating', 'gas', 'heating-oil', 'wood', 'electricity', 'heat-pump', 'other'], dim='Categories2')
    dm_multi.sort('Categories1')

    dm_shares = dm_multi
    dm_shares.append(dm_single, dim='Categories1')


  return dm_shares
