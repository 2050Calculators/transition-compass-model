import numpy as np

from model.common.auxiliary_functions import create_years_list, load_pop
from _database.pre_processing.buildings.Switzerland.get_data_functions.heating_technology_CH import (
  extract_heating_technologies_old, extract_heating_technologies, prepare_heating_mix_by_archetype,
  compute_heating_mix_F_E_D_categories, compute_heating_mix_C_B_categories, compute_heating_mix_by_category,
  clean_heating_cat, extract_heating_efficiency, compute_heating_efficiency_by_archetype)

from _database.pre_processing.buildings.Switzerland.get_data_functions.construction_period_param import load_construction_period_param

import os


def run(global_var, dm_all, years_ots):

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
  file =  os.path.join(this_dir,'../data/bld_heating_technology.pickle')
  dm_heating_tech = extract_heating_technologies(table_id, file, construction_period_envelope_cat_sfh, construction_period_envelope_cat_mfh)
  if 'gaz' in dm_heating_tech.col_labels['Categories2']:
      dm_heating_tech.rename_col('gaz', 'gas', 'Categories2')
  dm_heating_tech.add(0, dummy=True, dim='Categories2', col_label='coal')

  table_id = 'px-x-0902020100_112'
  file = os.path.join(this_dir, '../data/bld_heating_technology_1990-2000.pickle')
  dm_heating_tech_old = extract_heating_technologies_old(table_id, file, construction_period_envelope_cat_sfh, construction_period_envelope_cat_mfh)

  # Heating categories from Archetypes paper for B and C categories
  cdm_heating_archetypes = prepare_heating_mix_by_archetype()

  # Reconstruct heating-mix for older heating categories
  dm_heating_tech_old = compute_heating_mix_F_E_D_categories(dm_heating_tech, dm_heating_tech_old, years_ots)
  # Reconstruct heating-mix for new categories using archetypes for B and C
  # !FIXME Vaud behave differently than Switzerland
  dm_heating_tech_new = compute_heating_mix_C_B_categories(dm_heating_tech, cdm_heating_archetypes, years_ots, envelope_cat_new)

  # Merge old and new heating tech
  dm_heating_cat = dm_heating_tech_old.copy()
  dm_heating_cat.append(dm_heating_tech_new, dim='Categories1')
  dm_heating_cat.switch_categories_order('Categories1', 'Categories2')

  dm_heating_cat.sort('Categories2')

  alternative_computation = False
  if alternative_computation:
      # SECTION: Heating technology according to archetypes
      # Use Archetype paper to extract heating mix by archetype
      cdm_heating_archetypes = prepare_heating_mix_by_archetype()

      dm_heating_cat = compute_heating_mix_by_category(dm_heating_tech, cdm_heating_archetypes, dm_all)
      # Before and after construction period keep shares flat
      dm_heating_cat = clean_heating_cat(dm_heating_cat, envelope_cat_new)

      for cat in dm_heating_cat.col_labels['Categories2']:
          dm_heating_cat.filter(
              {'Categories1': ['multi-family-households'], 'Categories2': [cat]}).flatten().flatten().datamatrix_plot(
              {'Country': ['Switzerland']}, stacked=True)


  # SECTION: Heating efficiency
  # !FIXME : USE EP2050 for the efficiencies
  #######      HEATING EFFICIENCY     ###########
  file =os.path.join(this_dir, '../data/JRC-IDEES-2021_Residential_EU27.xlsx')
  sheet_name = 'RES_hh_eff'
  dm_heating_eff = extract_heating_efficiency_JRC(file, sheet_name, years_ots)
  dm_heating_eff[:, :, 'bld_heating-efficiency', 'electricity'] = 1
  dm_heating_eff_cat = compute_heating_efficiency_by_archetype(dm_heating_eff, dm_all, envelope_cat_new,
                                                               categories=dm_all.col_labels['Categories2'])

  dm_heating_eff = dm_heating_eff_cat
  dm_heating_eff.sort('Categories2')
  dm_heating_eff.filter(
    {'Categories1': dm_heating_cat.col_labels['Categories3']}, inplace=True)
  dm_heating_eff.sort('Categories1')
  dm_heating_eff.switch_categories_order()

  DM_heating = {'efficiency' : dm_heating_eff_cat,
                'heating-tech-split': dm_heating_cat}
  return DM_heating


if __name__ == "__main__":
  from floor_area_pipeline_CH import run as floor_area_run
  from renovation_pipeline_CH import run as renovation_run

  years_ots = create_years_list(1990, 2023, 1)

  global_var = load_construction_period_param()
  country_list = ['Switzerland', 'Vaud']
  dm_pop =load_pop(country_list=country_list, years_list=years_ots)

  print("Running floor area pipeline")
  DM_floor = floor_area_run(dm_pop, global_var, country_list, years_ots)
  dm_stock_tot = DM_floor['stock tot']
  dm_stock_cat = DM_floor['stock cat']
  dm_new_cat = DM_floor['new cat']
  dm_waste_cat = DM_floor['waste cat']

  print("Running renovation pipeline")
  DM_renov = renovation_run(dm_stock_tot, dm_stock_cat, dm_new_cat, dm_waste_cat, years_ots)
  dm_all = DM_renov['floor-area-cat']
  run(global_var, dm_all, years_ots)
