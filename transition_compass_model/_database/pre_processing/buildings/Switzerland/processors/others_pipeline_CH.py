import numpy as np

from model.common.auxiliary_functions import cdm_to_dm, create_years_list
from model.common.constant_data_matrix_class import ConstantDataMatrix
from model.common.data_matrix_class import DataMatrix


def run(country_list, years_ots, years_fts):
  # SECTION U-values - fixed assumption
  # Definition of Building Archetypes Based on the Swiss Energy Performance Certificates Database
  # by Alessandro Pongelli et al.
  # U-value is computed as the average of the house element u-value (roof, wall, windows, ..) weighted by their area
  # U-value in: W/m^2 K
  # Single-family-households
  envelope_cat_u_value = {'single-family-households':
                              {'F': 0.82, 'E': 0.69, 'D': 0.53, 'C': 0.41, 'B': 0.25},
                          'multi-family-households':
                              {'F': 0.93, 'E': 0.70, 'D': 0.63, 'C': 0.48, 'B': 0.29}}
  cdm_u_value = ConstantDataMatrix(col_labels={'Variables': ['bld_u-value'],
                                               'Categories1': ['multi-family-households', 'single-family-households'],
                                               'Categories2': ['B', 'C', 'D', 'E', 'F']},
                                   units={'bld_u-value': 'W/m2K'})
  arr = np.zeros((len(cdm_u_value.col_labels['Variables']), len(cdm_u_value.col_labels['Categories1']),
                  len(cdm_u_value.col_labels['Categories2'])))
  cdm_u_value.array = arr
  idx = cdm_u_value.idx
  for bld, dict_val in envelope_cat_u_value.items():
      for cat, val in dict_val.items():
          cdm_u_value.array[idx['bld_u-value'], idx[bld], idx[cat]] = val
  dm_u_value = cdm_to_dm(cdm_u_value, country_list, ['All'])

  # SECTION Surface to Floorarea factor - fixed assumption
  # From the same dataset we obtain also the floor to surface area
  surface_to_floorarea = {'single-family-households': 2.0, 'multi-family-households': 1.3}
  cdm_s2f = ConstantDataMatrix(col_labels={'Variables': ['bld_surface-to-floorarea'],
                                          'Categories1': ['multi-family-households', 'single-family-households']})
  arr = np.zeros((len(cdm_s2f.col_labels['Variables']), len(cdm_s2f.col_labels['Categories1'])))
  cdm_s2f.array = arr
  idx = cdm_s2f.idx
  for cat, val in surface_to_floorarea.items():
      cdm_s2f.array[idx['bld_surface-to-floorarea'], idx[cat]] = val
  cdm_s2f.units["bld_surface-to-floorarea"] = "%"
  dm_s2f = cdm_to_dm(cdm_s2f, country_list, ['All'])

  # SECTION: Heating-cooling behaviour (Temperature)
  #########################################
  #####   HEATING-COOLING BEHAVIOUR   #####
  #########################################
  col_label = {'Country': country_list,
               'Years': years_ots + years_fts,
               'Variables': ['bld_Tint-heating', 'bld_Tint-cooling'],
               'Categories1': ['multi-family-households', 'single-family-households'],
               'Categories2': ['B', 'C', 'D', 'E', 'F']}
  dm_Tint_heat = DataMatrix(col_labels=col_label,
                            units={'bld_Tint-heating': 'C',
                                   'bld_Tint-cooling': 'C'})
  dm_Tint_heat.array[...] = 20
  idx = dm_Tint_heat.idx
  cat_Tint = {'F': 19, 'E': 20, 'D': 21, 'C': 22, 'B': 23}
  for cat, tint in cat_Tint.items():
    dm_Tint_heat.array[:, :, idx['bld_Tint-heating'],
    idx['multi-family-households'], idx[cat]] = tint
    dm_Tint_heat.array[:, :, idx['bld_Tint-heating'],
    idx['single-family-households'], idx[cat]] = tint - 1


  # SECION: Building age
  first_bld_sfh = {'F': 1900, 'E': 1971, 'D': 1981, 'C': 2001, 'B': 2011}
  first_bld_mfh = {'F': 1900, 'E': 1981, 'D': 1991, 'C': 2001, 'B': 2011}
  col_label = {'Country': country_list,
               'Years': years_ots + years_fts,
               'Variables': ['bld_age'],
               'Categories1': ['multi-family-households', 'single-family-households'],
               'Categories2': ['B', 'C', 'D', 'E', 'F']}
  dm_age = DataMatrix(col_labels=col_label,
                            units={'bld_age': 'years'})
  years_all = np.array(dm_age.col_labels['Years'])
  nb_cntr = len(dm_age.col_labels['Country'])
  idx = dm_age.idx
  for cat, start_yr in first_bld_sfh.items():
    arr_age = years_all - start_yr
    arr_age = np.maximum(arr_age, 0)
    for idx_c in range(nb_cntr):
      dm_age.array[idx_c, :, idx['bld_age'], idx['single-family-households'],
      idx[cat]] = arr_age
  for cat, start_yr in first_bld_mfh.items():
    arr_age = years_all - start_yr
    arr_age = np.maximum(arr_age, 0)
    for idx_c in range(nb_cntr):
      dm_age.array[idx_c, :, idx['bld_age'], idx['multi-family-households'],
      idx[cat]] = arr_age

  ####################################
  #####     EMISSION FACTORS    ######
  ####################################
  # Obtained dividing emission by energy demand in file file = '../Europe/data/JRC-IDEES-2021_Residential_EU27.xlsx'
  JRC_emissions_fact = {'coal': 350, 'heating-oil': 267, 'gas': 200, 'wood': 0,
                        'solar': 0}
  cdm_emission_fact = ConstantDataMatrix(
    col_labels={'Variables': ['bld_CO2-factors'],
                'Categories1': ['coal', 'heating-oil', 'gas', 'wood', 'solar']},
    units={'bld_CO2-factors': 'kt/TWh'})
  cdm_emission_fact.array = np.zeros(
    (len(cdm_emission_fact.col_labels['Variables']),
     len(cdm_emission_fact.col_labels['Categories1'])))
  idx = cdm_emission_fact.idx
  for key, value in JRC_emissions_fact.items():
    cdm_emission_fact.array[0, idx[key]] = value

  cdm_emission_fact.sort('Categories1')

  # SECTION: Electricity emission factors
  col_dict = {
    'Country': country_list,
    'Years': years_ots + years_fts,
    'Variables': ['bld_CO2-factor'],
    'Categories1': ['electricity']
  }
  dm_elec = DataMatrix(col_labels=col_dict, units={'bld_CO2-factor': 'kt/TWh'})

  arr_elec = np.zeros((2, 40, 1, 1))
  idx = dm_elec.idx
  arr_elec[:, idx[1990]: idx[2023] + 1, 0, 0] = 112
  arr_elec[:, idx[2025]: idx[2050], 0, 0] = np.nan
  arr_elec[:, idx[2050], 0, 0] = 0
  dm_elec.array = arr_elec
  dm_elec.fill_nans(dim_to_interp="Years")

  DM_other = {'u-value': dm_u_value,
              'surface-to-floor': dm_s2f,
              'Tint-heat': dm_Tint_heat,
              'age':dm_age,
              'emission-factors': cdm_emission_fact,
              'emission-fact-elec': dm_elec}

  return DM_other

if __name__ == '__main__':
  print('Running U-value and Surface to floor area factor')
  country_list = ['Switzerland', 'Vaud']
  years_ots = create_years_list(1990, 2023, 1)
  years_fts = create_years_list(2025, 2050, 5)
  DM_other = run(country_list, years_ots, years_fts)
  print('Done')
