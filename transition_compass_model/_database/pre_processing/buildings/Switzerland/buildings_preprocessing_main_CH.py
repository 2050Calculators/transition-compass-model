from model.common.auxiliary_functions import create_years_list
from processors.floor_area_pipeline_CH import run as floor_area_run
from processors.renovation_pipeline_CH import run as renovation_run
from processors.heating_technology_pipeline_CH import run as heating_tech_run
from processors.others_pipeline_CH import run as other_run
from processors.hot_water_pipeline_CH import run as hotwater_run
from processors.build_ots_pickle import run as ots_pickle_run
from processors.buildings_interfaces_CH import run as load_interface_run
from processors.appliances_pipeline_CH import run as appliances_run
from processors.lighting_pipeline_CH import run as lighting_run
from processors.services_pipeline_CH import run as services_run
from processors.calibration_pipeline_CH import run as calibration_run
from scenarios.build_fts_BAU_pickle import run as fts_bau_pickle_run
from scenarios.build_fts_LoiEnergie_Vaud_pickle import run as fts_loi_energie_vaud_run
from scenarios.buildings_fts_EP2050_pickle import run as fts_Vaud_EP2050_run
from scenarios.build_fts_Tint_heating_pickle import run as fts_Tint_heating_run
from scenarios.build_fts_floor_area_pickle import run as fts_floor_area_run
from scenarios.build_fts_heating_efficiency_pickle import run as fts_efficiency_run
from get_data_functions.construction_period_param import load_construction_period_param


years_ots = create_years_list(1990, 2023, 1)
years_fts = create_years_list(2025, 2050, 5)

global_var = load_construction_period_param()

country_list = ['Switzerland', 'Vaud']

print("Running load interface")
DM_pop = load_interface_run(country_list)
dm_pop = DM_pop['pop']
dm_pop_ots = DM_pop['pop'].filter({'Years': years_ots})

print("Running floor area pipeline")
DM_floor = floor_area_run(global_var, country_list, years_ots)
# Extract floor area output
dm_stock_tot = DM_floor['stock tot']
dm_stock_cat = DM_floor['stock cat']
dm_new_cat = DM_floor['new cat']
dm_waste_cat = DM_floor['waste cat']

print("Running renovation pipeline")
DM_renov = renovation_run(dm_stock_tot, dm_stock_cat, dm_new_cat, dm_waste_cat, years_ots)
# Extract renovation output
dm_all = DM_renov['floor-area-cat']

print("Running heating technology pipeline")
DM_heating = heating_tech_run(global_var, dm_all, country_list, years_ots)

print("Running other pipeline")
DM_other = other_run(country_list, years_ots, years_fts)

print('Running appliances pipeline')
DM_appliances = appliances_run(dm_pop.copy(), country_list, years_ots, years_fts)

print('Running hot water pipeline')
DM_hotwater = hotwater_run(country_list, years_ots)

print('Running lighting pipeline')
dm_light = lighting_run(country_list, years_ots)

print('Running services / non-residential pipeline')
DM_services = services_run(country_list, years_ots)

print('Extract Buildings energy demand for Calibration')
dm_energy_cal = calibration_run(country_list, years_ots)

DM_all = {
  'floor_renov': DM_renov,
  'space-heat': DM_heating,
  'misc': DM_other,
  'appliances': DM_appliances,
  'hot-water': DM_hotwater,
  'lighting': dm_light,
  'services': DM_services
  }

print('Compile pickle ots')
DM_buildings = ots_pickle_run(dm_pop_ots, DM_all, years_ots, years_fts)

print('Compile pickle fts - all BAU')
DM_buildings = fts_bau_pickle_run(DM_buildings, country_list, years_fts)

print('Compile Scenario EP2050 - Vaud - level 3')
DM_buildings = fts_Vaud_EP2050_run(DM_buildings, lev=3)

print('Compile Scenario Loi Energie 2025 - Vaud - level 4')
DM_buildings = fts_loi_energie_vaud_run(DM_buildings, dm_pop_ots, global_var, country_list, lev=4)

print('Add scenarios for internal temperature setting')
DM_buildings = fts_Tint_heating_run(DM_buildings, years_ots, years_fts)

print('Add scenarios for floor area/cap')
DM_buildings = fts_floor_area_run(DM_buildings, years_ots, years_fts)

print('Add scenarios for heating efficiency (heat-pumps)')
DM_buildings = fts_efficiency_run(DM_buildings, years_fts)

print('Hello')

