import pandas as pd

from model.common.data_matrix_class import DataMatrix
from model.common.constant_data_matrix_class import ConstantDataMatrix
from model.common.io_database import dm_to_database
from model.common.interface_class import Interface
from model.common.auxiliary_functions import calibration_rates, \
  create_years_list, linear_fitting
from model.common.auxiliary_functions import read_level_data, filter_country_and_load_data_from_pickles
import pickle
import json
import os
import numpy as np
from collections import Counter
import time


def init_years_lever():
  # function that can be used when running the module as standalone to initialise years and levers
  years_setting = [1990, 2023, 2025, 2050, 5]
  f = open('../config/lever_position.json')
  lever_setting = json.load(f)[0]
  return years_setting, lever_setting


# CalculationLeaf READ PICKLE
def read_data(DM_TCAF, lever_setting, years_all):

    # Read fts based on lever_setting
    #DM_ots_fts = read_level_data(DM_TCAF, lever_setting)

    # Sub-matrix for TCAF health-diet
    dm_tcaf_paf = DM_TCAF['fxa']['health-diet_paf']
    dm_tcaf_dalys = DM_TCAF['fxa']['health-diet_dalys']

    # Aggregate Data Matrix - DIETARY HABITS
    DM_TCAF_health_diet = {
        'health-diet_paf': dm_tcaf_paf,
        'health-diet_dalys': dm_tcaf_dalys
    }

    # Aggregate Data Matrix - BIODIVERSITY
    DM_TCAF_biodiversity = {
        'biodiversity-ch': DM_TCAF['fxa']['biodiversity']['TCAF-biodiversity-CH'],
        'biodiversity-world': DM_TCAF['fxa']['biodiversity']['TCAF-biodiversity-world']
    }
    for key in DM_TCAF_biodiversity.keys():
      linear_fitting(DM_TCAF_biodiversity[key], years_all)
      DM_TCAF_biodiversity[key].filter({'Years':years_all}, inplace=True)

    # Constants
    CDM_MF = {}
    # For health-diet
    cdm_temp = DM_TCAF['constant'].filter_w_regex({'Variables': 'tcaf_mf_health-diet.*'})
    CDM_MF['health-diet'] = cdm_temp

    return DM_TCAF_health_diet, DM_TCAF_biodiversity, CDM_MF

# SimulateInteractions

def simulate_diet_to_TCAF_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../_database/data/interface/dietary-habits_to_TCAF.pickle")
    with open(f, 'rb') as handle:
        DM_diet = pickle.load(handle)
    return DM_diet

def simulate_landuse_to_TCAF_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    f = os.path.normpath(
      os.path.join(current_file_directory,
                  "../_database/data/interface/land-use_to_TCAF.pickle")
    )
    with open(f, 'rb') as handle:
        dm_cropland= pickle.load(handle)
    return dm_cropland

# CalculationLeaf TCAF HEALTH DIET
def TCAF_health_diet_workflow(DM_diet, DM_TCAF_health_diet, CDM_MF):
  # For both BAU & SCENARIO
  dm_diet_bau = DM_diet['diet-consumed_bau']
  dm_diet_sce = DM_diet['diet-consumed_scenario']

  # Pre-processing
  dm_data_paf = DM_TCAF_health_diet['health-diet_paf']
  dm_data_dalys = DM_TCAF_health_diet['health-diet_dalys']

  # pondération genre PAF

  # Step 0 - Groupby categories relevant for health ----------------------------
  # Red meat
  pattern = 'pro-liv-meat-bovine|pro-liv-meat-pig|pro-liv-meat-sheep|pro-liv-meat-oth-animal'
  dm_diet_bau.groupby({'pro-liv-meat-red': pattern}, dim='Categories1',
                      inplace=True, regex=True)
  dm_diet_sce.groupby({'pro-liv-meat-red': pattern}, dim='Categories1',
                      inplace=True, regex=True)


  # Step 1 - Link the intakes in DM_diet and DM_TCAF_health_diet ---------------
  # Note: link them with the closest value

  # Round up the intake
  #dm_diet_bau[:, :, :, :] = np.round(dm_diet_bau[:, :, :, :])
  #dm_diet_sce[:, :, :, :] = np.round(dm_diet_sce[:, :, :, :])

  # Health categories that we consider
  cat_health = ['crop-fruit',
                'crop-pulse',
                'pro-liv-abp-dairy-milk',
                'crop-oilcrop',
                'pro-liv-meat-processed',
                'pro-liv-meat-red',
                'crop-veg',
                'crop-cereal-whole']
  cat_temp = ['tcaf_health-diet_paf_crop-fruit',
                'tcaf_health-diet_paf_crop-pulse',
                'tcaf_health-diet_paf_pro-liv-abp-dairy-milk',
                'tcaf_health-diet_paf_crop-oilcrop',
                'tcaf_health-diet_paf_pro-liv-meat-processed',
                'tcaf_health-diet_paf_pro-liv-meat-red',
                'tcaf_health-diet_paf_crop-veg',
                'tcaf_health-diet_paf_crop-cereal-whole']

  # Filter diet according to health categories
  dm_diet_bau.filter({'Categories1':cat_health}, inplace=True)
  dm_diet_sce.filter({'Categories1': cat_health}, inplace=True)

  # Create a dm with only the relevant intakes
  dm_paf = dm_data_dalys.copy()
  dm_data_dalys_temp = dm_data_dalys.copy()
  dm_paf_year = dm_data_dalys.copy()
  # Add dummies for processing
  dm_paf_year.add(0.0, dummy=True, col_label=cat_temp, dim='Variables', unit='--------')
  # Drop the DALYs because we only want the structure
  dm_paf.drop(dim='Variables', col_label='tcaf_health-diet_dalys')
  dm_paf_year.drop(dim='Variables', col_label='tcaf_health-diet_dalys')

  n_countries = 1

  for cat in cat_health:
    if cat not in dm_diet_bau.col_labels['Categories1']:
      print(f"Warning: {cat} not in dm_diet_bau")
    if cat not in dm_data_paf:
      print(f"Warning: {cat} not in dm_data_paf")

  for cat in cat_health:
    variable_name = 'tcaf_health-diet_paf_' + cat
    for year in dm_diet_bau.col_labels['Years']:
      # 0: initialize arrays
      arr_diet_intake = dm_diet_bau[:, year, 'lfs_consumers-diet', cat]
      arr_paf_intake = dm_data_paf[cat][:,:,:]
      list_paf_intake = np.array(dm_data_paf[cat].col_labels['Years'])
      # 1: Compute absolute difference between actual intake and available PAF intake levels
      diff = np.abs(list_paf_intake[None, :] - arr_diet_intake[:,None])  # shape (n_countries, n_intakes)

      # 2: Find the index of the closest intake level for each country
      closest_idx = np.argmin(diff, axis=1)  # shape (n_countries,)

      # 3: Select the corresponding row across all 9 variables
      arr_filtered = arr_paf_intake[np.arange(arr_paf_intake.shape[0]),
                     closest_idx, :]  # shape (n_countries, 9)
      # 4: format as a datamatrix
      year = np.int64(year)
      dm_paf_temp = DataMatrix.based_on(arr_filtered[:, np.newaxis, np.newaxis, :],
                                         dm_data_dalys_temp.filter({'Years':[year]},inplace=False),
                                         change={
                                           'Variables': [variable_name]},
                                         units={
                                           variable_name: '-'})
      # 5: Append years together
      dm_paf_year[:,year,variable_name,:] = dm_paf_temp[:,:,variable_name,:]
    # 6: Append variables together
    dm_paf.append(dm_paf_year.filter({'Variables':[variable_name]},inplace=False), dim='Variables')


  # Step 2 - Associated DALYs per disease d for total country = PAF d,r * DALYs d ------------------
  dm_data_dalys = dm_data_dalys.flatten()
  dm_data_dalys.rename_col_regex(str1="tcaf_health-diet_dalys", str2="", dim="Variables")
  array_temp = dm_paf[:,:,:,:] * dm_data_dalys[:,:,np.newaxis,:]
  dm_paf.deepen(based_on='Variables')
  dm_paf.switch_categories_order(cat1='Categories2',
                                         cat2='Categories1')  # Switch categories
  dm_paf.add(array_temp[:,:,np.newaxis,:,:], dummy=True, col_label='tcaf_health-diet_dalys', dim='Variables', unit='DALYs')

  # Step 3 - Total DALYs = sum(DALYs d) ----------------------------------------
  dm_dalys_tot = dm_paf.copy()
  #dm_dalys_tot.drop(dim='Categories2', col_label='combined')
  dm_dalys_tot.groupby({'total': '.*'}, dim='Categories2',inplace=True, regex=True)
  dm_dalys_tot.switch_categories_order(cat1='Categories2',cat2='Categories1')
  dm_dalys_tot = dm_dalys_tot.flatten()

  # Step 4 - Calibration: normalise according to the total DALYs ---------------
  # Use combined PAF




  return dm_paf, dm_dalys_tot


# CalculationLeaf TCAF BIODIVERSITY

def TCAF_biodiversity_workflow(DM_TCAF_biodiversity, DM_landuse_to_TCAF):
  DM_TCAF_biodiversity = DM_TCAF_biodiversity.copy()

  # Step Biodiversity Switzerland
  # Drop treenut cat because not in cropland
  DM_TCAF_biodiversity['biodiversity-ch'].drop(dim='Categories2', col_label='treenut')
  # Add mean value for starch missing in biodiv
  dm_temp = DM_TCAF_biodiversity['biodiversity-ch'].groupby({'starch': '.*'},
                      dim='Categories2',
                      aggregation='mean',
                      regex=True, inplace=False)
  DM_TCAF_biodiversity['biodiversity-ch'].append(dm_temp, dim='Categories2')
  # Append cropland to biodiversity for relevant geoscale
  DM_TCAF_biodiversity['biodiversity-ch'].append(DM_landuse_to_TCAF['cropland-ch'], dim='Variables')

  # Biodiversity costs [CHF/ha] = cropland [ha] * eco-costs [CHF/ha]
  DM_TCAF_biodiversity['biodiversity-ch'].operation('agr_cropland', '*', 'eco-cost',
             dim='Variables',
             out_col='tcaf_biodiversity',
             unit='CHF')

  # Step Biodiversity World
  # Drop Switzerland and differing countries if any
  DM_landuse_to_TCAF['cropland-world'].drop(dim='Country',col_label='Switzerland')
  set_countries = set(DM_TCAF_biodiversity['biodiversity-world'].col_labels['Country']) - set(DM_landuse_to_TCAF['cropland-world'].col_labels['Country'])
  DM_TCAF_biodiversity['biodiversity-world'].drop(dim='Country', col_label=list(set_countries))
  # Sort countries
  DM_TCAF_biodiversity['biodiversity-world'].sort(dim='Country')
  DM_landuse_to_TCAF['cropland-world'].sort(dim='Country')

  # Sum total cropland
  DM_landuse_to_TCAF['cropland-world'].groupby({'total': '.*'},
                      dim='Categories1',
                      aggregation='sum',
                      regex=True, inplace=True)
  DM_landuse_to_TCAF['cropland-world'] = DM_landuse_to_TCAF['cropland-world'].flatten()
  DM_landuse_to_TCAF['cropland-world'].rename_col_regex(str1="agr_cropland_total_total", str2="agr_cropland", dim="Variables")

  # Append cropland to biodiversity for relevant geoscale
  DM_TCAF_biodiversity['biodiversity-world'].append(DM_landuse_to_TCAF['cropland-world'], dim='Variables')

  # Biodiversity costs [EUR2024/ha] = cropland [ha] * eco-costs [EUR2024/ha]
  DM_TCAF_biodiversity['biodiversity-world'].operation('agr_cropland', '*', 'eco-cost',
             dim='Variables',
             out_col='tcaf_biodiversity',
             unit='EUR2024')

  return DM_TCAF_biodiversity

# CalculationLeaf TPE INTERFACE
def TCAF_TPE_interface(dm_health_diet_detailed, dm_health_diet_tot):

  # health-diet detailed
  dm_health_diet_detailed.filter({'Variables':['tcaf_health-diet_dalys']}, inplace=True)
  dm_tpe = dm_health_diet_detailed.flattest()

  # health-diet total
  dm_health_diet_tot.filter({'Variables': ['tcaf_health-diet_dalys']},
                                 inplace=True)
  dm_tpe.append(dm_health_diet_tot.flattest(), dim='Variables')

  return dm_tpe

def TCAF(lever_setting, years_setting, DM_input, interface=Interface()):

    years_ots = create_years_list(years_setting[0], years_setting[1],1)  # make list with years from 1990 to 2015
    years_fts = create_years_list(years_setting[2], years_setting[3], years_setting[4])
    years_all = years_ots + years_fts

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    DM_TCAF_health_diet, DM_TCAF_biodiversity, CDM_MF = read_data(DM_input, lever_setting, years_all)
    country_list = ['Switzerland']


    # INTERFACES IN ---------------------------------------------------------------------------------------------------

    # CalculationLeaf Link interface or Simulate data from other modules
    # dietary-habits
    if interface.has_link(from_sector='dietary-habits', to_sector='TCAF_health-diet'):
      DM_diet = interface.get_link(from_sector='dietary-habits', to_sector='TCAF_health-diet')
    else:
      if len(interface.list_link()) != 0:
        print('You are missing dietary-habits to TCAF interface')
      DM_diet = simulate_diet_to_TCAF_input()
      for key in DM_diet.keys():
        DM_diet[key].filter({'Country': country_list}, inplace=True)

    # land-use
    if interface.has_link(from_sector='land-use', to_sector='TCAF'):
      DM_landuse_to_TCAF = interface.get_link(from_sector='land-use', to_sector='TCAF')
    else:
      if len(interface.list_link()) != 0:
        print('You are missing land-use to TCAF interface')
      DM_landuse_to_TCAF = simulate_landuse_to_TCAF_input()



    # CalculationTree ---------------------------------------------------------------------------------------------------
    dm_health_diet_detailed, dm_health_diet_tot = TCAF_health_diet_workflow(DM_diet, DM_TCAF_health_diet, CDM_MF)
    DM_TCAF_biodiversity = TCAF_biodiversity_workflow(DM_TCAF_biodiversity, DM_landuse_to_TCAF)
    # CalculationTree TPE OUTPUT -------------------------------------------------------------------------------------------------------
    results_run = TCAF_TPE_interface(dm_health_diet_detailed, dm_health_diet_tot)

    # INTERFACES OUT ---------------------------------------------------------------------------------------------------

    # interface to Land use
    #DM_lus = agriculture_landuse_interface(DM_bioenergy, dm_lgn, dm_land_use)
    #interface.add_link(from_sector='agriculture', to_sector='land-use',
    #                   dm=DM_lus)

    return results_run


def TCAF_module_local_run():
  country_list = ['Switzerland']
  DM_input = filter_country_and_load_data_from_pickles \
    (country_list= country_list, modules_list = 'TCAF', filter_country=False)
  years_setting, lever_setting = init_years_lever()
  TCAF(lever_setting, years_setting, DM_input['TCAF'])
  return

if __name__ == "__main__":
  TCAF_module_local_run()
