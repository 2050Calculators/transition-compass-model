from model.common.auxiliary_functions import create_years_list, load_pop

from _database.pre_processing.buildings.Switzerland.get_data_functions.construction_period_param import load_construction_period_param

from _database.pre_processing.buildings.Switzerland.get_data_functions.renovation_CH import (
  extract_number_of_buildings, compute_renovated_buildings, compute_renovation_rate,
  extract_renovation_redistribuition, compute_floor_area_renovated, compute_stock_area_by_cat)
import os

def run(dm_stock_tot, dm_stock_cat, dm_new_cat, dm_waste_cat, years_ots):

  # SECTION Floor area Renovated ots
  # Number of buildings
  # Bâtiments selon les niveaux géographiques institutionnels, la catégorie de bâtiment et l'époque de construction
  table_id = 'px-x-0902010000_103'
  this_dir = os.path.dirname(os.path.abspath(__file__))
  file = os.path.join(this_dir, '../data/bld_nb-buildings_2010_2022.pickle')
  dm_bld = extract_number_of_buildings(table_id, file)

  # Number of renovated-buildings (thermal insulation)
  # https://www.newsd.admin.ch/newsd/message/attachments/82234.pdf
  # "Programme bâtiments" rapports annuels 2014-2022, focus sur isolation thérmique
  nb_buildings_isolated = {2022: 8148, 2021: 8400, 2020: 8050, 2019: 8500,
                            2018: 7500, 2017: 8100, 2016: 7900, 2014: 8303}
  nb_buildings_systemic_renovation \
      = {2022: 2326, 2021: 2320, 2020: 2240, 2019: 1900, 2018: 1200,
         2017: 374, 2016: 0, 2014: 0}
  nb_buildings_renovated = dict()
  for yr in nb_buildings_isolated.keys():
      nb_buildings_renovated[yr] = nb_buildings_isolated[yr] + nb_buildings_systemic_renovation[yr]

  # For 2014 - 2016 we assume VD share = VD share 2017
  VD_share = {2014: 0.11, 2015: 0.11, 2016: 0.11, 2017: 0.110, 2018: 0.103,
              2019: 0.154, 2020: 0.16, 2021: 0.193, 2022: 0.15}
  share_by_bld = {'single-family-households': 0.55, 'multi-family-households': 0.35, 'other': 0.1}
  dm_renovation = compute_renovated_buildings(dm_bld, nb_buildings_renovated, VD_share, share_by_bld)

  # Compute renovation-rate
  dm_renovation = compute_renovation_rate(dm_renovation, years_ots)

  # SECTION Renovation by envelope cat ots
  # According to the Programme Batiments the assenissment is
  # Amélioration de +1 classes CECB 57%
  # Amélioration de +2 classes CECB 15%
  # Amélioration de +3 classes CECB 15%
  # Amélioration de +4 classes CECB 13%
  ren_map_in = {(1990, 2000): {'F': 0, 'E': 0.85, 'D': 0.15, 'C': 0, 'B': 0},
                 (2001, 2010): {'F': 0, 'E': 0.69, 'D': 0.16, 'C': 0.15, 'B': 0},
                 (2011, 2023): {'F': 0, 'E': 0.46, 'D': 0.23, 'C': 0.16, 'B': 0.15}}
  ren_map_out = {(1990, 2000): {'F': -0.8, 'E': -0.2, 'D': 0, 'C': 0, 'B': 0},
                (2001, 2010): {'F': -0.8, 'E': -0.2, 'D': 0, 'C': 0, 'B': 0},
                (2011, 2023): {'F': -0.8, 'E': -0.2, 'D': 0, 'C': 0, 'B': 0}}
  dm_renov_distr = extract_renovation_redistribuition(ren_map_in, ren_map_out, years_ots)

  # Harmonise floor-area stock, new and demolition rate
  #dm_cat = compute_bld_demolition_rate(dm_energy_cat, envelope_cat_new)

  # SECTION Floor-area Renovated by envelope cat
  # r_ct (t) = Ren-disr_ct(t) ren-rate(t) s(t-1)
  dm_renov_cat = compute_floor_area_renovated(dm_stock_tot, dm_renovation, dm_renov_distr)

  # SECTION Stock by envelope cat
  # s_{c,t}(t-1) &= s_{c,t}(t) - n_{c,t}(t) - r_{c,t}(t) + w_{c,t}(t)
  dm_all = compute_stock_area_by_cat(dm_stock_cat, dm_new_cat, dm_renov_cat, dm_waste_cat, dm_stock_tot)

  DM_renov = {
    'floor-area-cat': dm_all,
    'renovation-rate': dm_renovation,
    'renovation-redistribution': dm_renov_distr
  }

  return DM_renov

if __name__ == "__main__":

  from floor_area_pipeline_CH import run as floor_area_run
  years_ots = create_years_list(1990, 2023, 1)

  country_list = ['Switzerland', 'Vaud']
  dm_pop = load_pop(country_list=country_list, years_list=years_ots)

  global_vars = load_construction_period_param()

  # Run floor area pipeline
  print("Running floor area pipeline")
  DM = floor_area_run(dm_pop, global_vars=global_vars, years_ots=years_ots, country_list=country_list)
  dm_stock_tot = DM['stock tot']
  dm_stock_cat = DM['stock cat']
  dm_new_cat = DM['new cat']
  dm_waste_cat = DM['waste cat']

  # Run renovation pipeline
  print("Running renovation pipeline")
  DM_renov = run(dm_stock_tot, dm_stock_cat, dm_new_cat, dm_waste_cat, years_ots)
