from model.common.auxiliary_functions import create_years_list
from processors.floor_area_pipeline_CH import run as floor_area_run
from processors.renovation_pipeline_CH import run as renovation_run
from processors.heating_technology_pipeline_CH import run as heating_tech_run
from processors.others_pipeline_CH import run as other_run
from processors.build_ots_pickle import run as ots_pickle_run
from scenarios.build_fts_BAU_pickle import run as fts_bau_pickle_run
from scenarios.build_fts_LoiEnergie_Vaud_pickle import run as fts_loi_energie_vaud_run

years_ots = create_years_list(1990, 2023, 1)
years_fts = create_years_list(2025, 2050, 5)
construction_period_envelope_cat_sfh = {
  'F': ['Avant 1919', '1919-1945', '1946-1960', '1961-1970'],
  'E': ['1971-1980'],
  'D': ['1981-1990', '1991-2000'],
  'C': ['2001-2005', '2006-2010'],
  'B': ['2011-2015', '2016-2020', '2021-2023']}
construction_period_envelope_cat_mfh = {
  'F': ['Avant 1919', '1919-1945', '1946-1960', '1961-1970', '1971-1980'],
  'E': ['1981-1990'],
  'D': ['1991-2000'],
  'C': ['2001-2005', '2006-2010'],
  'B': ['2011-2015', '2016-2020', '2021-2023']}
envelope_cat_new = {'D': (1990, 2000), 'C': (2001, 2010), 'B': (2011, 2023)}

global_var = {
  'envelope construction sfh': construction_period_envelope_cat_sfh,
  'envelope construction mfh': construction_period_envelope_cat_mfh,
  'envelope cat new': envelope_cat_new}

country_list = ['Switzerland', 'Vaud']

print("Running floor area pipeline")
DM_floor = floor_area_run(years_ots)
# Extract floor area output
dm_stock_tot = DM_floor['stock tot']
dm_stock_cat = DM_floor['stock cat']
dm_new_cat = DM_floor['new cat']
dm_waste_cat = DM_floor['waste cat']
dm_pop = DM_floor['pop']

print("Running renovation pipeline")
DM_renov = renovation_run(dm_stock_tot, dm_stock_cat, dm_new_cat, dm_waste_cat, years_ots)
# Extract renovation output
dm_all = DM_renov['floor-area-cat']

print("Running heating technology pipeline")
DM_heating = heating_tech_run(global_var, dm_all, years_ots)

print("Running other pipeline")
DM_other = other_run(country_list, years_ots, years_fts)

print('Compile pickle ots')
DM_buildings = ots_pickle_run(DM_floor, DM_renov, DM_heating, DM_other, years_ots)

print('Compile pickle fts - all BAU')
DM_buildings = fts_bau_pickle_run(DM_buildings, country_list, years_fts)

print('Compile Scenario Loi Energie 2025 - Vaud - level 4')
DM_buildings = fts_loi_energie_vaud_run(DM_buildings, global_var, dm_pop, lev=4)
