import numpy as np
import pickle

from model.common.auxiliary_functions import linear_fitting, create_years_list, my_pickle_dump, cdm_to_dm
from model.common.data_matrix_class import DataMatrix
from model.common.constant_data_matrix_class import ConstantDataMatrix

from floor_area_CH import (extract_stock_floor_area, compute_floor_area_stock_v2,
                           extract_bld_new_buildings_2, compute_bld_floor_area_new,
                           extract_bld_new_buildings_1, compute_waste,
                           compute_floor_area_waste_cat, compute_floor_area_new_cat)

from renovation_CH import (extract_number_of_buildings, compute_renovated_buildings,
                           compute_renovation_rate, extract_renovation_redistribuition,
                           compute_floor_area_renovated, compute_stock_area_by_cat)

from heating_technology_CH import (extract_heating_technologies_old, extract_heating_technologies,
                                   prepare_heating_mix_by_archetype, compute_heating_mix_F_E_D_categories,
                                   compute_heating_mix_C_B_categories, compute_heating_mix_by_category,
                                   clean_heating_cat, extract_heating_efficiency, compute_heating_efficiency_by_archetype)

# Stock
years_ots = create_years_list(1990, 2023, 1)
years_fts = create_years_list(2025, 2050, 5)

def load_pop():
  # population
  filepath = "../../../data/datamatrix/lifestyles.pickle"
  with open(filepath, 'rb') as handle:
      DM_lfs = pickle.load(handle)
  dm_pop = DM_lfs["ots"]["pop"]["lfs_population_"].copy()
  dm_pop.append(DM_lfs["fts"]["pop"]["lfs_population_"][1],"Years")
  dm_pop = dm_pop.filter({"Country" : ['Vaud', 'Switzerland']})
  dm_pop.sort("Years")
  dm_pop.filter({"Years" : years_ots},inplace=True)
  return dm_pop

# __file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/buildings/Switzerland/buildings_preprocessing_CH.py"
#filename = 'data/bld_household_size.pickle'
#dm_lfs_household_size = extract_lfs_household_size(years_ots, table_id='px-x-0102020000_402', file=filename)
dm_pop = load_pop()

# SECTION Floor area Stock ots
construction_period_envelope_cat_sfh = {'F': ['Avant 1919', '1919-1945', '1946-1960', '1961-1970'],
                                        'E': ['1971-1980'],
                                        'D': ['1981-1990', '1991-2000'],
                                        'C': ['2001-2005', '2006-2010'],
                                        'B': ['2011-2015', '2016-2020', '2021-2023']}
construction_period_envelope_cat_mfh = {'F': ['Avant 1919', '1919-1945', '1946-1960', '1961-1970', '1971-1980'],
                                        'E': ['1981-1990'],
                                        'D': ['1991-2000'],
                                        'C': ['2001-2005', '2006-2010'],
                                        'B': ['2011-2015', '2016-2020', '2021-2023']}
envelope_cat_new = {'D': (1990, 2000), 'C': (2001, 2010), 'B': (2011, 2023)}

# Floor area stock
# Logements selon les niveaux géographiques institutionnels, la catégorie de bâtiment,
# la surface du logement et l'époque de construction
# https://www.pxweb.bfs.admin.ch/pxweb/fr/px-x-0902020200_103/-/px-x-0902020200_103.px/
table_id = 'px-x-0902020200_103'
file = 'data/bld_floor-area_stock.pickle'
#dm_bld_area_stock, dm_energy_cat = compute_bld_floor_area_stock_tranformed_avg_new_area(table_id, file,
#                                                                years_ots, construction_period_envelope_cat_sfh,
#                                                                construction_period_envelope_cat_mfh)
dm_stock_tot, dm_stock_cat, dm_avg_floor_area = compute_floor_area_stock_v2(table_id, file, dm_pop=dm_pop,
                                                         cat_map_sfh=construction_period_envelope_cat_sfh,
                                                         cat_map_mfh=construction_period_envelope_cat_mfh)

# SECTION Floor area New ots
# New residential buildings by sfh, mfh
# Nouveaux logements selon la grande région, le canton, la commune et le type de bâtiment, depuis 2013
table_id = 'px-x-0904030000_107'
file = 'data/bld_new_buidlings_2013_2023.pickle'
dm_bld_new_buildings_1 = extract_bld_new_buildings_1(table_id, file)

# Nouveaux logements selon le type de bâtiment, 1995-2012
table_id = 'px-x-0904030000_103'
file = 'data/bld_new_buildings_1995_2012.pickle'
dm_bld_new_buildings_2 = extract_bld_new_buildings_2(table_id, file)
# Floor-area new by sfh, mfh
dm_new_tot_raw = compute_bld_floor_area_new(dm_bld_new_buildings_1, dm_bld_new_buildings_2, dm_avg_floor_area, dm_pop)
del dm_bld_new_buildings_2, dm_bld_new_buildings_1

# SECTION Floor-area Waste + Recompute New
dm_waste_tot, dm_new_tot = compute_waste(dm_stock_tot, dm_new_tot_raw)
dm_waste_cat = compute_floor_area_waste_cat(dm_waste_tot)
# Floor-area new by sfh, mfh and envelope categories
dm_new_cat = compute_floor_area_new_cat(dm_new_tot, envelope_cat_new)

# Empty apartments
# Logements vacants selon la grande région, le canton, la commune, le nombre de pièces d'habitation
# et le type de logement vacant
# https://www.pxweb.bfs.admin.ch/pxweb/fr/px-x-0902020300_101/-/px-x-0902020300_101.px/

# SECTION Floor area Renovated ots
# Number of buildings
# Bâtiments selon les niveaux géographiques institutionnels, la catégorie de bâtiment et l'époque de construction
table_id = 'px-x-0902010000_103'
file = 'data/bld_nb-buildings_2010_2022.pickle'
dm_bld = extract_number_of_buildings(table_id, file)

print('Maybe you should considered the buildings undergoing systemic renovation as well')
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

# Harmonise stock, new, waste, and renovation ots, demolition-rate
# The analysis before accounts for the stock, new, demolition-rate of building stock construction period.
# the equation I have been working with is: s(t) = s(t-1) + n(t) - w(t)
# Now I want to account for the renovation that redistributes the energy categories.
# The total equation does not change but for each energy class we have
# s_c(t) = s_c(t-1) + n_c(t) - w_c(t) + r_c(t), where r_c(t) can be positive or negative
# I want to assume that n_c, w_c and r_c are given as well as s_c(t), and I compute s_c(t-1).
# I will need to re-compute the demolition rate
#dm_all = harmonise_stock_new_renovated_transformed(dm_energy_cat, dm_renovation, dm_renov_distr, envelope_cat_new)


#dm_energy.datamatrix_plot()
# if I want to save 2.2 TWh, and the maximum improvement I can do is 150 kWh/m2, I need to renovate 15 million m2.
# The current park is 137 + 337 = 474 million m2
# If I want to fix the renovation at 1%, then you renovate 4.7 million m2, in order to save 2.2 TWh,
# you need to save: 470 kWh/m2


DM_buildings = {'ots': dict(), 'fts': dict(), 'fxa': dict(), 'constant': dict()}


file = '../../../data/datamatrix/buildings.pickle'
with open(file, 'rb') as handle:
    DM_bld = pickle.load(handle)


# SECTION: Heating technology
##########   HEATING TECHNOLOGY     #########
# You need to extract the heating technology (you only have the last 3 years
# but you have the energy mix for the historical period)
# https://www.pxweb.bfs.admin.ch/pxweb/fr/px-x-0902010000_102/-/px-x-0902010000_102.px/
# In order to check the result the things I can validate are the 1990, 2000 value and the 2021-2023 values
# You can run the check to see if the allocation by envelope category is well done and matches with the original data
# The problem is that at the end the energy demand decreases.

table_id = 'px-x-0902010000_102'
file = 'data/bld_heating_technology.pickle'
dm_heating_tech = extract_heating_technologies(table_id, file, construction_period_envelope_cat_sfh, construction_period_envelope_cat_mfh)
if 'gaz' in dm_heating_tech.col_labels['Categories2']:
    dm_heating_tech.rename_col('gaz', 'gas', 'Categories2')
dm_heating_tech.add(0, dummy=True, dim='Categories2', col_label='coal')

table_id = 'px-x-0902020100_112'
file = 'data/bld_heating_technology_1990-2000.pickle'
dm_heating_tech_old = extract_heating_technologies_old(table_id, file, construction_period_envelope_cat_sfh, construction_period_envelope_cat_mfh)

# Heating categories from Archetypes paper for B and C categories
cdm_heating_archetypes = prepare_heating_mix_by_archetype()

# Reconstruct heating-mix for older heating categories
dm_heating_tech_old = compute_heating_mix_F_E_D_categories(dm_heating_tech, dm_heating_tech_old)
# Reconstruct heating-mix for new categories using archetypes for B and C
# !FIXME Vaud behave differently than Switzerland
dm_heating_tech_new = compute_heating_mix_C_B_categories(dm_heating_tech, cdm_heating_archetypes)

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

# CHECK:
check = False
if check:
    arr_tmp = dm_heating_cat.array[:, :, 0, ...] * dm_stock_cat.array[:, :, -1, :, :, np.newaxis]
    dm_heating_cat.add(arr_tmp, dim='Variables', unit='m2', col_label='bld_heating-mix-area')
    dm_heating_new = dm_heating_cat.filter({'Variables': ['bld_heating-mix-area']})
    dm_heating_new.group_all('Categories2')
    dm_heating_new.normalise('Categories2')

# SECTION: Heating efficiency
#######      HEATING EFFICIENCY     ###########
file = '../Europe/data/databases_full/JRC/JRC-IDEES-2021_Residential_EU27.xlsx'
sheet_name = 'RES_hh_eff'
dm_heating_eff = extract_heating_efficiency(file, sheet_name, years_ots)
dm_heating_eff_cat = compute_heating_efficiency_by_archetype(dm_heating_eff, dm_stock_cat, envelope_cat_new,
                                                             categories=dm_stock_cat.col_labels['Categories2'])
