import numpy as np
import pickle
import os
import pandas as pd

from model.common.auxiliary_functions import create_years_list, load_pop, dm_add_missing_variables
from model.common.data_matrix_class import DataMatrix

import _database.pre_processing.buildings.Switzerland.get_data_functions.floor_area_CH as fla

from _database.pre_processing.buildings.Switzerland.get_data_functions.construction_period_param import load_construction_period_param
from model.common.data_matrix_class import DataMatrix


def fill_missing_years_using_FSO_data(dm, dm_raw):

  dm_add_missing_variables(dm, {'Years': dm_raw.col_labels['Years']}, fill_nans=False)
  dm.filter({'Years': dm_raw.col_labels['Years']}, inplace=True)
  dm_raw.rename_col('bld_floor-area_stock', 'bld_floor-area_stock_raw', dim='Variables')
  dm.append(dm_raw, dim='Variables')
  dm.operation('bld_floor-area_stock', '/', 'bld_floor-area_stock_raw', out_col='cal_factor', unit='%')
  idx = dm.idx
  dm[:, :idx[2007], 'cal_factor', :] = np.nan
  dm.fill_nans('Years')
  dm[:, :, 'bld_floor-area_stock', :] = dm[:, :, 'cal_factor', :] * dm[:, :, 'bld_floor-area_stock_raw', :]
  dm.filter({'Variables': ['bld_floor-area_stock']}, inplace=True)

  return dm

def clean_WP_ERA_file(df, cantons_name_list):
  # Select only residential
  df_residential = df.loc[df['NAME'].isin(['EFH', 'MFH', 'Sonstige_Wohngeb'])].T
  df_residential.reset_index(inplace=True)
  df_residential.columns = df_residential.iloc[0]
  df_residential = df_residential.iloc[1:]
  df_residential.rename(columns={'NAME': 'Country', 'EFH': 'bld_floor-area_stock_single-family-households[m2]',
                                 'MFH': 'bld_floor-area_stock_multi-family-households[m2]',
                                 'Sonstige_Wohngeb': 'bld_floor-area_stock_other[m2]'}, inplace=True)
  df_residential['Years'] = 2023
  dm = DataMatrix.create_from_df(df_residential, num_cat=1)
  dm_other = dm.filter({'Categories1': ['other']})
  dm.drop('Categories1', 'other')
  dm_other_split= dm.normalise('Categories1', inplace=False)
  dm_other_split[:, 2023, :, :] = dm_other_split[:, 2023, :, :] * dm_other[:, 2023, :, :]
  dm[...] = dm_other_split[...] + dm[...]

  dm.groupby({'single-family-households': 'single-family.*'}, regex=True, dim='Categories1', inplace=True)
  dm_CH = dm.groupby({'Switzerland':'.*'}, regex=True, dim='Country')
  dm.append(dm_CH,dim='Country')
  dm.sort('Country')
  dm.sort('Categories1')
  WP_cantons_name_list = dm.col_labels['Country'].copy()
  dm.rename_col(WP_cantons_name_list, cantons_name_list, dim='Country')

  return dm


def run(global_vars, country_list, years_ots):
  # Stock

  cantons_en = ['Aargau', 'Appenzell Ausserrhoden', 'Appenzell Innerrhoden',
                'Basel Landschaft', 'Basel Stadt', 'Bern', 'Fribourg', 'Geneva',
                'Glarus', 'Graubünden', 'Jura', 'Lucerne', 'Neuchâtel',
                'Nidwalden', 'Obwalden', 'Schaffhausen', 'Schwyz', 'Solothurn',
                'St. Gallen', 'Thurgau', 'Ticino', 'Uri', 'Valais', 'Vaud',
                'Zug', 'Zurich']


  dm_pop = load_pop(cantons_en+['Switzerland'], years_ots)
  dm_pop.sort('Country')
  # __file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/buildings/Switzerland/buildings_preprocessing_CH.py"
  #filename = 'data/bld_household_size.pickle'
  #dm_lfs_household_size = extract_lfs_household_size(years_ots, table_id='px-x-0102020000_402', file=filename)
  this_dir = os.path.dirname(os.path.abspath(__file__))

  construction_period_envelope_cat_sfh = global_vars['envelope construction sfh']
  construction_period_envelope_cat_mfh = global_vars['envelope construction mfh']
  envelope_cat_new = global_vars['envelope cat new']

  # SECTION Floor area Stock ots

  # Floor area stock
  # Logements selon les niveaux géographiques institutionnels, la catégorie de bâtiment,
  # la surface du logement et l'époque de construction
  # https://www.pxweb.bfs.admin.ch/pxweb/fr/px-x-0902020200_103/-/px-x-0902020200_103.px/
  table_id = 'px-x-0902020200_103'
  file = os.path.join(this_dir, '../data/bld_floor-area_stock_all_cantons.pickle')
  #dm_bld_area_stock, dm_energy_cat = compute_bld_floor_area_stock_tranformed_avg_new_area(table_id, file,
  #                                                                years_ots, construction_period_envelope_cat_sfh,
  #                                                                construction_period_envelope_cat_mfh)
  dm_stock_tot, dm_stock_cat, dm_avg_floor_area = (
    fla.compute_floor_area_stock_v2(table_id, file, dm_pop=dm_pop,
                                cat_map_sfh=construction_period_envelope_cat_sfh,
                                cat_map_mfh=construction_period_envelope_cat_mfh,
                                years_ots= years_ots))

  # SECTION Floor area New ots
  # New residential buildings by sfh, mfh
  # Nouveaux logements selon la grande région, le canton, la commune et le type de bâtiment, depuis 2013
  table_id = 'px-x-0904030000_107'
  file = os.path.join(this_dir, '../data/bld_new_buidlings_2013_2023_all_cantons.pickle')
  dm_bld_new_buildings_1 = fla.extract_bld_new_buildings_1(table_id, file)

  # Nouveaux logements selon le type de bâtiment, 1995-2012
  table_id = 'px-x-0904030000_103'
  file = os.path.join(this_dir, '../data/bld_new_buildings_1995_2012_all_cantons.pickle')
  dm_bld_new_buildings_2 = fla.extract_bld_new_buildings_2(table_id, file)
  # Floor-area new by sfh, mfh
  dm_new_tot_raw = fla.compute_bld_floor_area_new(dm_bld_new_buildings_1, dm_bld_new_buildings_2, dm_avg_floor_area, dm_pop, years_ots)
  del dm_bld_new_buildings_2, dm_bld_new_buildings_1

  # SECTION Adjust Cantonal stock based on National stock by Prognos
  file_url = 'https://www.bfe.admin.ch/bfe/de/home/politik/energieperspektiven-2050-plus.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvMTA0NDE=.html'
  zip_name = os.path.join(this_dir, '../data/EP2050_sectors.zip')
  file_pickle = os.path.join(this_dir, '../data/bld_EP2050_floor_area_type.pickle')
  dm_stock_tot_CH = fla.extract_EP2050_floor_area_by_type(file_url, zip_name, file_pickle)

  # Find missing years in Prognos EP2050 dataset using FSO data (incl. 2000-2007 period)
  dm_stock_tot_CH_raw = dm_stock_tot.filter({'Country': ['Switzerland']})
  dm_stock_tot_CH = fill_missing_years_using_FSO_data(dm_stock_tot_CH, dm_stock_tot_CH_raw)

  # Project to cantonal data based on shares
  dm_stock_tot.drop('Country', 'Switzerland')
  dm_stock_tot.normalise(dim='Country', inplace=True)
  assert dm_stock_tot.col_labels['Categories1'] == dm_stock_tot_CH.col_labels['Categories1']
  dm_stock_tot[...] = dm_stock_tot[...] * dm_stock_tot_CH[...]
  dm_stock_tot.change_unit('bld_floor-area_stock', old_unit='%', new_unit='m2', factor=1)
  dm_stock_tot.append(dm_stock_tot_CH, dim='Country')
  dm_stock_tot.sort('Country')

  # Adjust stock by canton for based on the 2023 data by Wuest & Partner
  file_path = os.path.join(this_dir, '../data/WP_EBF_cantons_2023.xlsx')
  df_WP = pd.read_excel(file_path, sheet_name='EBF')
  dm_WP = clean_WP_ERA_file(df_WP, cantons_name_list=dm_stock_tot.col_labels['Country'])
  # !FIXME: You are here!
  arr_adj_factor = dm_WP[:, 2023, 'bld_floor-area_stock', :] / dm_stock_tot[:, 2023, 'bld_floor-area_stock', :]
  dm_stock_tot[:, :, 'bld_floor-area_stock', :] = dm_stock_tot[:, :, 'bld_floor-area_stock', :] * arr_adj_factor[:, np.newaxis, :]

  # Adjust cantonal stock by energy class
  dm_stock_cat.normalise('Categories2')
  assert dm_stock_tot.col_labels['Categories1'] == dm_stock_cat.col_labels['Categories1']
  assert dm_stock_tot.col_labels['Country'] == dm_stock_cat.col_labels['Country']
  dm_stock_cat[:, :, :, :, :] = dm_stock_tot[:, :, :, :, np.newaxis] * dm_stock_cat[:, :, :, :, :]
  dm_stock_cat.change_unit('bld_floor-area_stock', old_unit='%', new_unit='m2', factor=1)

  # SECTION Floor-area Waste + Recompute New
  dm_waste_tot, dm_new_tot = fla.compute_waste(dm_stock_tot, dm_new_tot_raw, years_ots)
  dm_waste_cat = fla.compute_floor_area_waste_cat(dm_waste_tot)
  # Floor-area new by sfh, mfh and envelope categories
  dm_new_cat = fla.compute_floor_area_new_cat(dm_new_tot, envelope_cat_new)

  DM = {'stock tot': dm_stock_tot.filter({'Country': country_list}),
        'stock cat': dm_stock_cat.filter({'Country': country_list}),
        'new cat': dm_new_cat.filter({'Country': country_list}),
        'waste cat': dm_waste_cat.filter({'Country': country_list})}

  return DM


if __name__ == "__main__":

  global_vars = load_construction_period_param()

  years_ots = create_years_list(1990, 2023, 1)
  cantons_en = ['Aargau', 'Appenzell Ausserrhoden', 'Appenzell Innerrhoden',
                'Basel Landschaft', 'Basel Stadt', 'Bern', 'Fribourg', 'Geneva',
                'Glarus', 'Graubünden', 'Jura', 'Lucerne', 'Neuchâtel',
                'Nidwalden', 'Obwalden', 'Schaffhausen', 'Schwyz', 'Solothurn',
                'St. Gallen', 'Thurgau', 'Ticino', 'Uri', 'Valais', 'Vaud',
                'Zug', 'Zurich']
  country_list = cantons_en + ['Switzerland']

  DM = run(global_vars, country_list, years_ots)
