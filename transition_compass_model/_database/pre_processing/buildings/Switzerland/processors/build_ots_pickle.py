
import numpy as np
import pickle

from model.common.auxiliary_functions import linear_fitting, my_pickle_dump, \
  add_dummy_country_to_DM, dm_add_missing_variables, sort_pickle
from _database.pre_processing.api_routines_CH import get_data_api_CH
from model.common.data_matrix_class import DataMatrix

import os


def recompute_floor_area_per_capita(dm_all, dm_pop):
  dm_floor_stock = dm_all.filter({'Variables': ['bld_floor-area_stock'],
                                  'Categories': {'single-family-households',
                                                 'multi-family-households'}},
                                 inplace=False)

  # Computer m2/cap for lifestyles
  dm_floor_stock.group_all(dim='Categories2')
  dm_floor_stock.group_all(dim='Categories1')
  dm_floor_stock.append(dm_pop, dim='Variables')

  dm_floor_stock.operation('bld_floor-area_stock', '/', 'lfs_population_total',
                           out_col='lfs_floor-intensity_space-cap',
                           unit='m2/cap')

  dm_floor_stock.filter({'Variables': ['lfs_floor-intensity_space-cap']},
                        inplace=True)

  return dm_floor_stock


def extract_lfs_household_size(years_ots, table_id, file):
  try:
    with open(file, 'rb') as handle:
      dm_household_size = pickle.load(handle)
  except OSError:
    structure, title = get_data_api_CH(table_id, mode='example')
    # Extract buildings floor area
    filter = {'Year': structure['Year'],
              'Canton (-) / District (>>) / Commune (......)': [
                'Schweiz / Suisse / Svizzera / Switzerland', '- Vaud'],
              'Household size': ['1 person', '2 persons', '3 persons',
                                 '4 persons', '5 persons', '6 persons or more']}
    mapping_dim = {'Country': 'Canton (-) / District (>>) / Commune (......)',
                   'Years': 'Year',
                   'Variables': 'Household size'}
    unit_all = ['people'] * len(filter['Household size'])
    # Get api data
    dm_household = get_data_api_CH(table_id, mode='extract', filter=filter,
                                   mapping_dims=mapping_dim, units=unit_all)

    dm_household.rename_col(
      ['Schweiz / Suisse / Svizzera / Switzerland', '- Vaud'],
      ['Switzerland', 'Vaud'],
      dim='Country')
    drop_strings = [' persons or more', ' persons', ' person']
    for drop_str in drop_strings:
      dm_household.rename_col_regex(drop_str, '', dim='Variables')
    # dm_household contains the number of household per each household-size
    # Compute the average household size by doing the weighted average
    sizes = np.array(
      [int(num_ppl) for num_ppl in dm_household.col_labels['Variables']])
    arr_weighted_size = dm_household.array * sizes[np.newaxis, np.newaxis, :]
    arr_avg_size = np.nansum(arr_weighted_size, axis=-1,
                             keepdims=True) / np.nansum(dm_household.array,
                                                        axis=-1,
                                                        keepdims=True)
    # Create new datamatrix
    dm_household_size = DataMatrix.based_on(arr_avg_size, dm_household, change={
      'Variables': ['lfs_household-size']},
                                            units={
                                              'lfs_household-size': 'people'})
    linear_fitting(dm_household_size, years_ots)
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, file)
    with open(f, 'wb') as handle:
      pickle.dump(dm_household_size, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return dm_household_size


def compute_building_mix(dm_all):

    dm_building_mix = dm_all.filter(
      {'Variables': ['bld_floor-area_stock', 'bld_floor-area_new']},
      inplace=False).flatten()
    dm_building_mix.normalise('Categories1', keep_original=True)
    dm_building_mix.deepen()
    dm_building_mix.rename_col(
      ['bld_floor-area_stock_share', 'bld_floor-area_new_share'],
      ['bld_building-mix_stock', 'bld_building-mix_new'], dim='Variables')
    dm_building_mix.filter(
      {'Variables': ['bld_building-mix_stock', 'bld_building-mix_new']},
      inplace=True)

    return dm_building_mix


def run(dm_pop, DM_all, years_ots, years_fts):

  DM_renov = DM_all['floor_renov']
  DM_heating = DM_all['space-heat']
  DM_other = DM_all['misc']
  DM_appliances = DM_all['appliances']
  DM_hotwater = DM_all['hot-water']
  dm_light = DM_all['lighting'].copy()
  DM_services = DM_all['services']

  this_dir = os.path.dirname(os.path.abspath(__file__))
  # !FIXME: use the actual values and not the calibration factor
  file = os.path.join(this_dir, '../../../../data/datamatrix/buildings.pickle')
  with open(file, 'rb') as handle:
    DM_bld = pickle.load(handle)

  DM_buildings = {'ots': dict(), 'fts': dict(), 'fxa': dict(),
                  'constant': dict()}

  # Extract DM inputs

  # OTHER
  dm_Tint_heat = DM_other['Tint-heat']
  dm_age = DM_other['age']
  dm_u_value = DM_other['u-value']
  cdm_emission_fact = DM_other['emission-factors']
  dm_s2f = DM_other['surface-to-floor']
  dm_elec = DM_other['emission-fact-elec']

  # RENOVATION
  dm_all = DM_renov['floor-area-cat']
  dm_renovation = DM_renov['renovation-rate']
  dm_renov_distr = DM_renov['renovation-redistribution']

  # HEATING
  dm_heating_eff_cat = DM_heating['efficiency']
  dm_heating_cat = DM_heating['heating-tech-split']

  # FLOOR AREA

  # SECTION: fxa - bld_age
  DM_buildings['fxa']['bld_age'] = dm_age

  # SECTION: fxa - u-value
  DM_buildings['fxa']['u-value'] = dm_u_value

  # SECTION: fxa - surface-to-floorarea
  DM_buildings['fxa']['surface-to-floorarea'] = dm_s2f

  # SECTION: constant - emissions
  DM_buildings['constant']['emissions'] = cdm_emission_fact

  # SECTION: fxa - emission-factor-electricity
  DM_buildings['fxa']['emission-factor-electricity'] = dm_elec

  # SECTION: fxa - appliances
  DM_buildings['fxa']['appliances'] = DM_appliances['fxa']['appliances'].copy()

  # SECTION: fxa - hot water demand
  # Determine fts years for hot water variables
  dm_hw_demand = DM_hotwater['hw-energy-demand']
  linear_fitting(dm_hw_demand, years_fts)
  dm_hw_efficiency =  DM_hotwater['hw-efficiency']
  dm_add_missing_variables(dm_hw_efficiency, {'Years': years_fts}, fill_nans=False)
  dm_hw_tech_mix = DM_hotwater['hw-tech-mix'].copy()
  dm_add_missing_variables(dm_hw_tech_mix, {'Years': years_fts}, fill_nans=False)

  DM_buildings['fxa']['hot-water'] = \
    {
      'hw-energy-demand': dm_hw_demand.copy(),
      'hw-efficiency':  dm_hw_efficiency.copy(),
      'hw-tech-mix': dm_hw_tech_mix.copy()
     }

  # SECTION: fxa - lighting
  # According to EP2050 "Consommation d’énergie finale en fonction de l’application"
  # https://www.uvek-gis.admin.ch/BFE/storymaps/AP_Energieperspektiven/index.html?lang=de&selectedSzenario=ZB&selectedSektor=HH&selectedDimension=ET&selectedFly=01
  fts_light = {2025: 2.5/3.6, 2030: 2.1/3.6, 2035: 2/3.6, 2040: 1.9/3.6, 2045: 1.8/3.6, 2050: 1.7/3.6}
  arr = (dm_light[:, :, 'bld_residential-lighting']
         / dm_light['Switzerland', np.newaxis, :, 'bld_residential-lighting'])
  shares = arr.mean(axis=1)

  dm_add_missing_variables(dm_light, {'Years': years_fts}, fill_nans=False)
  for yr, value in fts_light.items():
    dm_light['Switzerland', yr, 'bld_residential-lighting'] = value
    dm_light[:, yr, 'bld_residential-lighting'] = shares * dm_light['Switzerland', np.newaxis, yr, 'bld_residential-lighting']

  DM_buildings['fxa']['lighting'] = dm_light

  # SECTION: fxa - services
  linear_fitting(DM_services['services_demand'], years_fts, based_on=list(range(2012, 2023)))
  dm_add_missing_variables(DM_services['services_tech-mix'], {'Years': years_fts}, fill_nans=False)
  dm_add_missing_variables(DM_services['services_efficiencies'], {'Years': years_fts}, fill_nans=False)
  DM_buildings['fxa']['services'] = DM_services

  # add_dummy_country_to_DM(DM_appliances, 'EU27', 'Switzerland')
  #file = os.path.join(this_dir , '../../../../data/datamatrix/buildings.pickle')
  #with open(file, 'rb') as handle:
  #  DM_B = pickle.load(handle)
  #DM_B['fxa']['appliances'] = DM_appliances['fxa']['appliances'].copy()
  #with open(file, 'wb') as handle:
  #  pickle.dump(DM_B, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # CALIBRATION
  # SECTION: fxa - heating-energy-calibration
  DM_buildings['fxa']['heating-energy-calibration'] = DM_bld['fxa'][
    'heating-energy-calibration'].filter({"Country": ["Switzerland", "Vaud"]})


  # OTS
  # SECTION: ots - floor-intensity
  #filename = os.path.join(this_dir, '../data/bld_household_size.pickle')
  dm_lfs_household_size = DM_appliances['fxa']['household'].filter({'Variables': ['lfs_household-size']})

  dm_space_cap = recompute_floor_area_per_capita(dm_all, dm_pop)
  dm_space_cap.append(dm_lfs_household_size.filter({'Years': years_ots}), dim='Variables')
  DM_buildings['ots']['floor-intensity'] = dm_space_cap.copy()

  DM_buildings['ots']['heatcool-behaviour'] = dm_Tint_heat.filter(
    {'Years': years_ots})

  # SECTION: fxa - bld_type
  dm_building_mix = compute_building_mix(dm_all)
  dm_bld_type = dm_building_mix.filter({'Variables': ['bld_building-mix_stock']})
  dm_bld_type.add(np.nan, dummy=True, dim='Years', col_label=years_fts)

  DM_buildings['fxa']['bld_type'] =  dm_bld_type.copy()

  # SECTION: ots - renovation-rate -> building-mix
  DM_buildings['ots']['building-renovation-rate'] = dict()
  DM_buildings['ots']['building-renovation-rate'][
    'bld_building-mix'] = dm_building_mix.filter({'Variables': ['bld_building-mix_new']})

  # SECTION: ots - renovation-rate -> renovation-rate
  dm_rr = dm_renovation.filter({'Variables': ['bld_renovation-rate']})
  DM_buildings['ots']['building-renovation-rate']['bld_renovation-rate'] = dm_rr

  # SECTION: ots - renovation-rate -> renovation-redistribution
  DM_buildings['ots']['building-renovation-rate'][
    'bld_renovation-redistribution'] = dm_renov_distr.copy()

  # SECTION: ots - renovation-rate -> demolition-rate
  dm_tot = dm_all.filter({'Variables': ['bld_floor-area_stock', 'bld_floor-area_waste']})
  dm_tot.group_all('Categories2')
  dm_tot.operation('bld_floor-area_waste', '/', 'bld_floor-area_stock',
                   out_col='bld_demolition-rate', unit='%')
  dm_demolition_rate = dm_tot.filter({'Variables': ['bld_demolition-rate']})
  DM_buildings['ots']['building-renovation-rate'][
    'bld_demolition-rate'] = dm_demolition_rate.copy()

  # SECTION: ots - heating-technology-fuel -> bld_heating-technology
  DM_buildings['ots']['heating-technology-fuel'] = dict()
  dm_heating_cat.sort('Categories3')
  DM_buildings['ots']['heating-technology-fuel'][
    'bld_heating-technology'] = dm_heating_cat.copy()

  # SECTION: ots - heating-efficiency
  DM_buildings['ots']['heating-efficiency'] \
    = dm_heating_eff_cat.copy()


  my_pickle_dump(DM_buildings, file)
  sort_pickle(file)

  #add_dummy_country_to_DM(DM_buildings, new_country='EU27', ref_country='Switzerland')
  #DM_bld['fxa']['services'] = DM_buildings['fxa']['services']
  #DM_bld['fxa']['lighting'] = DM_buildings['fxa']['lighting']

  #with open(file, 'wb') as handle:
  #  pickle.dump(DM_bld, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return DM_buildings
