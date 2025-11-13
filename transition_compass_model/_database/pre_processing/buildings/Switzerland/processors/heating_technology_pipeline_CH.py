import numpy as np

from model.common.auxiliary_functions import create_years_list, load_pop, \
  dm_add_missing_variables, linear_fitting, \
  extrapolate_missing_years_based_on_per_capita, add_dummy_country_to_DM
import _database.pre_processing.buildings.Switzerland.get_data_functions.heating_technology_CH as ht

from _database.pre_processing.buildings.Switzerland.get_data_functions.construction_period_param import load_construction_period_param
from _database.pre_processing.buildings.Switzerland.get_data_functions.floor_area_CH import extract_nb_of_apartments_per_building_type
from _database.pre_processing.buildings.Switzerland.get_data_functions.hot_water_CH import extract_heating_efficiencies_EP2050

import os


def add_dummy_envelope_cat(dm_heating_eff):
  dm_heating_eff.rename_col('bld_heating_efficiency', 'bld_heating-efficiency_B', 'Variables')
  dm_heating_eff.deepen(based_on='Variables')
  dm_heating_eff.add(np.nan, dim='Categories2', col_label= ['C', 'D', 'E', 'F'], dummy=True)
  for cat in dm_heating_eff.col_labels['Categories2']:
    dm_heating_eff[:, :, :, :, cat] =  dm_heating_eff[:, :, :, :, 'B']
  dm_heating_eff.switch_categories_order()
  dm_heating_eff.sort('Categories1')
  dm_heating_eff.sort('Categories2')
  return dm_heating_eff

def adjust_COP_based_on_envelope_cat(dm):
  # This is taken from https://www.flumroc.ch/fileadmin/Dateiliste/flumroc/Bilder/400_steinwolle/Stromsparen-HSLU/d_250716_Kurzbericht_Studie_Flumroc_final.pdf
  # Which is originally taken from  Döring & Richter, 2024
  # And the information of the average energy demand per building category (SFH) from the archetype paper
  # Pongelli, A.; Priore, Y.D.; Bacher, J.-P.; Jusselme, T. Definition of Building Archetypes Based on the Swiss Energy Performance Certificates Database. Buildings 2023, 13, 40. https://doi.org/10.3390/buildings13010040
  avg_2023_COP = {'F': 2.3, 'E': 2.5, 'D': 3, 'C': 3.3, 'B': 3.6 }
  for cat in dm.col_labels['Categories1']:
    #corr_fact = avg_2023_COP[cat]/dm[:, 2023, np.newaxis, :, cat, 'heat-pump']
    #dm[:, :, :, cat, 'heat-pump'] =  corr_fact * dm[:, :, :, cat, 'heat-pump']
    dm[:, :, :, cat, 'heat-pump'] =  np.minimum(avg_2023_COP[cat] , dm[:, :, :, cat, 'heat-pump'])

  return dm


def calibrate_cantons_heating_tech_based_on_EP2050(dm_heating_tech_cantons, dm_heating_CH):

  def calibrate_heating(dm_raw, dm_cal):
    dm_shares = dm_raw.copy()

    # Compute the shares by dividing by canton and envelope category
    dm_shares.array = dm_shares.array / np.nansum(dm_shares.array, axis=(0, 3),
                                                  keepdims=True)
    dm_shares.drop('Categories3', ['solar', 'coal'])
    assert dm_heating_CH.col_labels['Categories1'] == dm_shares.col_labels['Categories2']
    assert dm_heating_CH.col_labels['Categories2'] == dm_shares.col_labels['Categories3']

    arr = (dm_shares[:, :, 'bld_heating-mix', :, :, :]
           * dm_cal['Switzerland', np.newaxis, :,'bld_heating-tech', np.newaxis, :, :])
    dm_shares.add(arr, dim='Variables', col_label='bld_heating-tech', unit='m2')

    dm_raw.add(np.nan, dim='Variables',col_label='bld_heating-tech', unit='m2',dummy=True)

    idx = dm_raw.idx
    for tech in dm_shares.col_labels['Categories3']:
      dm_raw[:, idx[2000]:, 'bld_heating-tech', :, :,
      tech] = dm_shares[:, idx[2000]:, 'bld_heating-tech', :, :, tech]

    dm_raw.operation('bld_heating-tech', '/', 'bld_heating-mix', out_col='ratio', unit='%')
    dm_raw[:, 0:idx[2000], 'ratio', ...] = np.nan
    dm_raw[:, :, 'ratio', :, :, 'coal'] = dm_raw[:, :, 'ratio', :, :, 'wood']
    dm_raw[:, :, 'ratio', :, :, 'solar'] = dm_raw[:, :, 'ratio', :, :, 'heat-pump']
    dm_raw[:, idx[2001]:, 'bld_heating-mix', :, :, 'coal'] = np.nan
    dm_raw[:, idx[2021]:, 'bld_heating-mix', :, :, 'coal'] = 0
    linear_fitting(dm_raw, based_on=list(range(1990, 2000)),
                   years_ots=list(range(2001, 2021)))
    dm_raw.array = np.maximum(0, dm_raw.array)
    dm_raw.fill_nans('Years')

    dm_raw.drop('Variables', 'bld_heating-tech')
    dm_raw.operation('bld_heating-mix', '*', 'ratio',
                                      out_col='bld_heating-tech', unit='m2')

    dm_out = dm_raw.filter({'Variables': ['bld_heating-tech']})

    return dm_out

  dm_heating_CH_raw = dm_heating_tech_cantons.filter({'Country': ['Switzerland']})
  dm_heating_tech_cantons.drop('Country', 'Switzerland')
  dm_heating_CH.sort('Categories2')
  dm_heating_CH.sort('Categories1')
  dm_heating_CH.drop('Categories2', 'other')

  dm_out_cantons = calibrate_heating(dm_heating_tech_cantons, dm_heating_CH)
  dm_out_CH = calibrate_heating(dm_heating_CH_raw, dm_heating_CH)

  dm_out_cantons.append(dm_out_CH, dim='Country')
  dm_out_cantons.sort('Country')

  return dm_out_cantons



def adjust_tech_mix(dm_tech_mix, dm_apt):

  dm_tech_mix[...] = dm_tech_mix[...] / np.nansum(dm_tech_mix[...], axis=(-1,-2), keepdims=True)
  dm_apt_loc = dm_apt.filter({'Years': dm_tech_mix.col_labels['Years']})

  dm_apt_loc.sort('Country')
  dm_tech_mix.sort('Country')
  assert dm_apt_loc.col_labels['Country'] == dm_tech_mix.col_labels['Country']

  arr = (dm_apt_loc[:, :, 0, :, np.newaxis, np.newaxis]
         * dm_tech_mix[:, :, 0, :, :, :])

  dm_tech_mix.add(arr, dim='Variables', col_label= 'tmp', unit='number')
  var_name = dm_tech_mix.col_labels['Variables'][0]
  dm_tech_mix.filter({'Variables': ['tmp']}, inplace=True)
  dm_tech_mix.rename_col('tmp', var_name, dim='Variables')

  return dm_tech_mix


def run(global_var, dm_all, country_list, years_ots):

  cantons_en = ['Aargau', 'Appenzell Ausserrhoden', 'Appenzell Innerrhoden',
                'Basel Landschaft', 'Basel Stadt', 'Bern', 'Fribourg', 'Geneva',
                'Glarus', 'Graubünden', 'Jura', 'Lucerne', 'Neuchâtel',
                'Nidwalden', 'Obwalden', 'Schaffhausen', 'Schwyz', 'Solothurn',
                'St. Gallen', 'Thurgau', 'Ticino', 'Uri', 'Valais', 'Vaud',
                'Zug', 'Zurich']
  cantons_fr =  ['Argovie', 'Appenzell Rh. Ext.', 'Appenzell Rh. Int.',
                 'Bâle Campagne', 'Bâle Ville', 'Berne', 'Fribourg', 'Genève',
                 'Glaris', 'Grisons', 'Jura', 'Lucerne', 'Neuchâtel', 'Nidwald',
                 'Obwald', 'Schaffhouse', 'Schwytz', 'Soleure', 'Saint Gall',
                 'Thurgovie', 'Tessin', 'Uri', 'Valais', 'Vaud', 'Zoug', 'Zurich']

  construction_period_envelope_cat_sfh = global_var['envelope construction sfh']
  construction_period_envelope_cat_mfh = global_var['envelope construction mfh']
  envelope_cat_new = global_var['envelope cat new']

  # SECTION: Heating technology
  ##########   HEATING TECHNOLOGY     #########
  # You need to extract the heating technology (you only have the last 3 years
  # but you have the energy mix for the historical period)
  # https://www.pxweb.bfs.admin.ch/pxweb/fr/px-x-0902010000_102/-/px-x-0902010000_102.px/
  # In order to check the result the things I can validate are the 1990, 2000 value and the 2021-2023 values
  # You can run the check to see if the allocation by envelope category is well done and matches with the original data
  # The problem is that at the end the energy demand decreases.

  table_id = 'px-x-0902010000_102'
  this_dir = os.path.dirname(os.path.abspath(__file__))
  file =  os.path.join(this_dir,'../data/bld_heating_technology_all_cantons.pickle')
  dm_heating_tech = ht.extract_heating_technologies(table_id, file, construction_period_envelope_cat_sfh, construction_period_envelope_cat_mfh)
  if 'gaz' in dm_heating_tech.col_labels['Categories2']:
      dm_heating_tech.rename_col('gaz', 'gas', 'Categories2')
  dm_heating_tech.add(np.nan, dummy=True, dim='Categories2', col_label='coal')
  # !FIXME Geneva data are missing
  dm_pop = load_pop(cantons_en + ['Switzerland'], dm_heating_tech.col_labels['Years'])
  dm_pop.sort('Country')
  pop_ratio  = (dm_pop['Geneva', :, :, np.newaxis, np.newaxis, np.newaxis]/
                dm_pop['Basel Stadt', :, :, np.newaxis, np.newaxis, np.newaxis])
  dm_heating_tech['Geneva', ...] = dm_heating_tech['Basel Stadt', ...]* pop_ratio

  dm_heating_tech.drop('Categories2', 'other')

  table_id = 'px-x-0902020100_112'
  file = os.path.join(this_dir, '../data/bld_heating_technology_1990-2000_all_cantons.pickle')
  dm_heating_tech_old = ht.extract_heating_technologies_old(table_id, file, construction_period_envelope_cat_sfh, construction_period_envelope_cat_mfh)
  dm_heating_tech_old.drop('Categories2', 'other')

  adjust_based_on_apt = False    # This does not work well for Vaud
  if adjust_based_on_apt:
    dm_pop = load_pop(cantons_en + ['Switzerland'], years_ots)
    # Extract number of apartments per building type
    table_id = 'px-x-0902020200_103'
    file = os.path.join(this_dir, '../data/bld_apartments_per_bld_type_new.pickle')
    dm_apt = extract_nb_of_apartments_per_building_type(table_id, file, cantons_fr, cantons_en)

    # Extrapolate number of apartments based on apt/pop
    dm_apt = extrapolate_missing_years_based_on_per_capita(dm_apt, dm_pop, years_ots, var_name='bld_apartments')

    # Weight the technology mix by the number of households/apartments
    dm_heating_tech = adjust_tech_mix(dm_heating_tech, dm_apt)
    dm_heating_tech_old = adjust_tech_mix(dm_heating_tech_old, dm_apt)

  # !FIXME: Adjust based on dm_all floor-area by category once you have all the cantons

  # Reconstruct heating-mix for older heating categories
  dm_heating_tech_old = ht.compute_heating_mix_F_E_D_categories(dm_heating_tech, dm_heating_tech_old, years_ots)

  dm_heating_tech_AB = dm_heating_tech.filter({'Categories1': ['B', 'C']})
  dm_add_missing_variables(dm_heating_tech_AB, {'Years': years_ots}, fill_nans=False)
  start_B = global_var['envelope cat new']['B'][0]
  start_C = global_var['envelope cat new']['C'][0]
  dm_heating_tech_AB[:, start_B-1, 'bld_heating-mix', 'B', :] = 0
  dm_heating_tech_AB[:, start_C-1, 'bld_heating-mix', 'C', :] = 0
  dm_heating_tech_AB.fill_nans('Years')

  dm_heating_tech_old.append(dm_heating_tech_AB, dim='Categories1')
  dm_heating_tech_cantons = dm_heating_tech_old

  #del dm_heating_tech, dm_heating_tech_AB

  file_url = 'https://www.bfe.admin.ch/bfe/de/home/politik/energieperspektiven-2050-plus.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvMTA0NDE=.html'
  zip_name = os.path.join(this_dir, '../data/EP2050_sectors.zip')
  file_pickle = os.path.join(this_dir, '../data/bld_heating_tech_EP2050.pickle')
  # Heating technology mix for single and multi in Switzerland from EP2050
  dm_heating_EP2050 = ht.extract_heating_technologies_EP2050(file_url, zip_name, file_pickle)
  # Keep only years_ots
  dm_add_missing_variables(dm_heating_EP2050, {'Years': years_ots}, fill_nans=False)
  dm_heating_EP2050.filter({'Years': years_ots}, inplace=True)

  # Floor area CH
  dm_floor_area_CH = dm_all.filter({'Variables': ['bld_floor-area_stock'], 'Country': ['Switzerland']}).group_all('Categories2', inplace=False)

  assert dm_floor_area_CH.col_labels['Categories1'] == dm_heating_EP2050.col_labels['Categories1']
  arr = dm_heating_EP2050[:, :, 'bld_heating-mix', :, :] * dm_floor_area_CH[:, :, 'bld_floor-area_stock', :, np.newaxis]
  dm_heating_EP2050.add(arr, dim='Variables', col_label='bld_heating-tech', unit='m2')

  # SECTION: Adjust Cantons heating technology to Swiss one
  dm_heating_tech = calibrate_cantons_heating_tech_based_on_EP2050(dm_heating_tech_cantons, dm_heating_EP2050)
  dm_heating_tech.normalise('Categories3', inplace=True)
  dm_heating_tech.rename_col('bld_heating-tech', 'bld_heating-mix', dim='Variables')
  dm_heating_tech.switch_categories_order('Categories1', 'Categories2')
  dm_heating_tech.add(0, dummy=True, col_label='other-tech', dim='Categories3')
  dm_heating_tech.sort('Categories2')
  dm_heating_tech.sort('Categories3')
  dm_heating_tech.sort('Categories1')


  # Reconstruct heating-mix for new categories using archetypes for B and C


  # SECTION: Heating efficiency
  # !FIXME : USE EP2050 for the efficiencies
  #######      HEATING EFFICIENCY     ###########
  file_url = 'https://www.bfe.admin.ch/bfe/de/home/politik/energieperspektiven-2050-plus.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvMTA0NDE=.html'
  zip_name = os.path.join(this_dir, '../data/EP2050_sectors.zip')
  file_pickle = os.path.join(this_dir, '../data/bld_heating_efficiencies.pickle')
  dm_heating_eff = extract_heating_efficiencies_EP2050(file_url, zip_name, file_pickle)
  dm_heating_eff.add(1, dim='Categories1', col_label=['electricity'], dummy=True)
  dm_heating_eff.add(0.6, dim='Categories1', col_label=['solar', 'other-tech', 'coal'], dummy=True)
  dm_heating_eff.sort('Categories1')
  linear_fitting(dm_heating_eff, years_ots, based_on=[2000, 2010])

  # Add Envelope categories (in this case there is not difference by envelope category)
  dm_heating_eff = add_dummy_envelope_cat(dm_heating_eff)

  dm_heating_eff = adjust_COP_based_on_envelope_cat(dm_heating_eff)

  DM_heating = {'efficiency': dm_heating_eff.filter({'Years': years_ots})}
  for cntr in country_list:
    add_dummy_country_to_DM(DM_heating, ref_country='Switzerland', new_country=cntr)

  DM_heating['heating-tech-split'] =  dm_heating_tech.filter({'Country': country_list})

  return DM_heating


if __name__ == "__main__":
  from floor_area_pipeline_CH import run as floor_area_run
  from renovation_pipeline_CH import run as renovation_run

  years_ots = create_years_list(1990, 2023, 1)

  global_var = load_construction_period_param()

  country_list =  ['Switzerland', 'Vaud']

  print("Running floor area pipeline")
  DM_floor = floor_area_run(global_var, country_list, years_ots)
  dm_stock_tot = DM_floor['stock tot']
  dm_stock_cat = DM_floor['stock cat']
  dm_new_cat = DM_floor['new cat']
  dm_waste_cat = DM_floor['waste cat']

  print("Running renovation pipeline")
  DM_renov = renovation_run(dm_stock_tot, dm_stock_cat, dm_new_cat, dm_waste_cat, years_ots)
  dm_all = DM_renov['floor-area-cat']
  run(global_var, dm_all, country_list, years_ots)
