import numpy as np
import pickle
import os

from model.common.auxiliary_functions import create_years_list, load_pop

from _database.pre_processing.buildings.Switzerland.get_data_functions.floor_area_CH import (
  compute_floor_area_stock_v2, extract_bld_new_buildings_2,
  compute_bld_floor_area_new, extract_bld_new_buildings_1, compute_waste,
  compute_floor_area_waste_cat, compute_floor_area_new_cat)

from _database.pre_processing.buildings.Switzerland.get_data_functions.construction_period_param import load_construction_period_param


def run(dm_pop, global_vars, country_list, years_ots):
  # Stock

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
  file = os.path.join(this_dir, '../data/bld_floor-area_stock.pickle')
  #dm_bld_area_stock, dm_energy_cat = compute_bld_floor_area_stock_tranformed_avg_new_area(table_id, file,
  #                                                                years_ots, construction_period_envelope_cat_sfh,
  #                                                                construction_period_envelope_cat_mfh)
  dm_stock_tot, dm_stock_cat, dm_avg_floor_area = (
    compute_floor_area_stock_v2(table_id, file, dm_pop=dm_pop,
                                cat_map_sfh=construction_period_envelope_cat_sfh,
                                cat_map_mfh=construction_period_envelope_cat_mfh,
                                years_ots= years_ots))

  # SECTION Floor area New ots
  # New residential buildings by sfh, mfh
  # Nouveaux logements selon la grande région, le canton, la commune et le type de bâtiment, depuis 2013
  table_id = 'px-x-0904030000_107'
  file = os.path.join(this_dir, '../data/bld_new_buidlings_2013_2023.pickle')
  dm_bld_new_buildings_1 = extract_bld_new_buildings_1(table_id, file)

  # Nouveaux logements selon le type de bâtiment, 1995-2012
  table_id = 'px-x-0904030000_103'
  file = os.path.join(this_dir, '../data/bld_new_buildings_1995_2012.pickle')
  dm_bld_new_buildings_2 = extract_bld_new_buildings_2(table_id, file)
  # Floor-area new by sfh, mfh
  dm_new_tot_raw = compute_bld_floor_area_new(dm_bld_new_buildings_1, dm_bld_new_buildings_2, dm_avg_floor_area, dm_pop, years_ots)
  del dm_bld_new_buildings_2, dm_bld_new_buildings_1

  # SECTION Floor-area Waste + Recompute New
  dm_waste_tot, dm_new_tot = compute_waste(dm_stock_tot, dm_new_tot_raw, years_ots)
  dm_waste_cat = compute_floor_area_waste_cat(dm_waste_tot)
  # Floor-area new by sfh, mfh and envelope categories
  dm_new_cat = compute_floor_area_new_cat(dm_new_tot, envelope_cat_new)

  DM = {'stock tot': dm_stock_tot,
        'stock cat': dm_stock_cat,
        'new cat': dm_new_cat,
        'waste cat': dm_waste_cat}

  return DM


if __name__ == "__main__":

  global_vars = load_construction_period_param()

  years_ots = create_years_list(1990, 2023, 1)
  country_list = ['Switzerland', 'Vaud']
  dm_pop = load_pop(country_list, years_ots)

  DM = run(dm_pop, global_vars, country_list, years_ots)
