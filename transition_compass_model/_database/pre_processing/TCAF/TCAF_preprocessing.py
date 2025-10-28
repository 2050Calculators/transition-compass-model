import numpy as np
from model.common.auxiliary_functions import interpolate_nans, add_missing_ots_years, linear_fitting_ots_db, linear_fitting, create_years_list
#from _database.pre_processing.api_routines_CH import get_data_api_CH
from scipy.stats import linregress
import pandas as pd
import faostat
import os
import re
from model.common.data_matrix_class import DataMatrix
from model.common.constant_data_matrix_class import ConstantDataMatrix
from model.common.io_database import read_database, read_database_fxa, edit_database, database_to_df, dm_to_database, database_to_dm
from model.common.io_database import read_database_to_ots_fts_dict, read_database_to_ots_fts_dict_w_groups, read_database_to_dm
from model.common.interface_class import Interface
from model.common.auxiliary_functions import compute_stock,  filter_geoscale, calibration_rates, filter_DM, add_dummy_country_to_DM
from model.common.auxiliary_functions import read_level_data, simulate_input
from scipy.optimize import linprog
import pickle
import json
import os
import numpy as np
import time

# CalculationLeaf TCAF HEALTH DIET
def TCAF_health_diet_preprocessing():

  # Data -----------------------------------------------------------------------
  df_data = pd.read_excel('data/PAF_Idriss.xlsx',
                            sheet_name='Sheet1')

  # Preprocessing --------------------------------------------------------------

  # Average PAF per risk factor, cause, grams
  df_data_grouped = df_data.groupby(['Risk_Factor','cause','grams'])['paf'].mean().reset_index()

  # Combined PAF = 1 - PROD(1-PAFi)
  df_paf_comb = df_data_grouped.copy()
  df_paf_comb = (
    df_paf_comb
    .groupby(['Risk_Factor', 'grams'])['paf']
    .apply(lambda x: 1 - np.prod(1 - x))
    .reset_index()
  )
  df_paf_comb['cause'] = 'Combined'

  # Concat dfs
  df_tcaf_health_diet = pd.concat([df_data_grouped, df_paf_comb])

  # Formatting -----------------------------------------------------------------

  # Add country 'Switzerland'
  df_tcaf_health_diet['Country'] = 'Switzerland'

  # Rename cols
  df_tcaf_health_diet.rename(columns={'paf': 'value', 'grams':'Years'}, inplace=True)

  # Rename terms
  risk_factor_map = {
    'Fruits': 'crop-fruit',
    'Whole_Grains': 'crop-cereal-whole',
    'Calcium': 'calcium',
    'Fiber': 'fiber',
    'Legumes': 'crop-pulses',
    'Milk': 'pro-liv-abp-dairy-milk',
    'Nuts': 'crop-nuts-seeds',
    'Omega_3': 'omega',
    'PUFA': 'pufa',
    'Processed_Meat': 'pro-liv-meat-processed',
    'Red_meat': 'pro-liv-meat-red',
    'SSB': 'pro-bev-ssb',
    'Vegetables': 'crop-veg'
  }
  df_tcaf_health_diet['Risk_Factor'] = df_tcaf_health_diet['Risk_Factor'].replace(risk_factor_map)

  # Rename terms
  cause_map = {
    'Colon and rectum cancer': 'CRC',
    'Diabetes mellitus type 2': 'DT2',
    'Intracerebral hemorrhage': 'ICH',
    'Ischemic heart disease': 'IHD',
    'Ischemic stroke': 'IS',
    'Subarachnoid hemorrhage': 'SH',
    'Tracheal, bronchus, and lung cancer': 'TBLC',
    'Esophageal cancer': 'EC',
    'Combined': 'combined'
  }
  df_tcaf_health_diet['cause'] = df_tcaf_health_diet['cause'].replace(cause_map)

  # Create variables name
  df_tcaf_health_diet['variables'] = 'tcaf_health-diet_paf_' + \
                                      df_tcaf_health_diet['Risk_Factor'] \
                                      + '_' + df_tcaf_health_diet['cause'] \
                                      + '[-]'

  # Format as separate dm, according to the risk factor (or food categories)
  # Note : here, the intake is processed as the 'Years' dimensions, and renamed
  # afterwards. Therefore, this DM has not timescale

  DM_TCAF_health_diet = {}

  for rf in df_tcaf_health_diet["Risk_Factor"].unique():
    sub_df = df_tcaf_health_diet[df_tcaf_health_diet["Risk_Factor"] == rf].copy()
    sub_df_pivot = sub_df.pivot_table(index=['Country', 'Years'], columns='variables', values='value').reset_index()
    dm = DataMatrix.create_from_df(sub_df_pivot, num_cat=0)
    dm.dim_labels[1] = 'Intake [g/day/cap]'
    DM_TCAF_health_diet[rf] = dm

  return DM_TCAF_health_diet

# CalculationLeaf CREATE PICKLE
def database_from_csv_to_datamatrix(years_ots, years_fts, DM_TCAF_health_diet):

  # Make list with years from 2020 to 2050 (steps of 5 years)
  years_all = years_ots + years_fts

  # FixedAssumptionsToDatamatrix -----------------------------------------------

  # Initialise
  dict_fxa = {}

  # Add in fxa
  dict_fxa['health-diet'] = DM_TCAF_health_diet

  # CalibrationDataToDatamatrix ------------------------------------------------

  # LeversToDatamatrix OTS -----------------------------------------------------
  dict_ots = {}


  # LeversToDatamatrix FTS -----------------------------------------------------
  dict_fts = {}

  # FTS linear fitting of ots
  '''DM_ots = DM_agriculture_old['ots'].copy()
  DM_fts = DM_agriculture_old['fts'].copy()

  # To do once when adding a new lever
  # DM_fts['climate-smart-crop']['processing-net-import'] = {'processing-net-import': dict()}

  # Levers to be normalised
  list_norm = ['climate-smart-livestock_ration']

  for key in DM_ots.keys():
    if isinstance(DM_ots[key], dict):
      for subkey in DM_ots[key].keys():
        dm = DM_ots[key][subkey].copy()
        linear_fitting(dm, years_fts)

        for lev in range(1, 5):  # 1 to 4
          if subkey in list_norm:  # ✅ check subkey, not key
            dm_norm = dm.copy()
            # Replace negative values with 0
            array_temp = dm_norm.array[:, :, :, :]
            array_temp[array_temp < 0] = 0.0
            dm_norm.array[:, :, :, :] = array_temp
            # Normalise
            dm_norm.normalise(dim='Categories1', inplace=True)
            DM_fts[key][subkey][lev] = dm_norm.filter(
              {'Years': years_fts}, inplace=False
            )
          else:
            DM_fts[key][subkey][lev] = dm.filter(
              {'Years': years_fts}, inplace=False
            )
    else:
      dm = DM_ots[key].copy()
      linear_fitting(dm, years_fts)
      for lev in range(1, 5):
        DM_fts[key][lev] = dm.filter({'Years': years_fts}, inplace=False)

  # file
  __file__ = "agriculture_landuse_preprocessing_EU.py"

  # directories
  current_file_directory = os.path.dirname(os.path.abspath(__file__))'''

  # ConstantsToDatamatrix ------------------------------------------------------
  dict_const = {}

  # Group all datamatrix in a single structure ---------------------------------
  DM_TCAF = {
    'fxa': dict_fxa,
    'constant': dict_const,
    'fts': dict_fts,
    'ots': dict_ots
  }

  # Write datamatrix to pickle -------------------------------------------------
  f = '../../data/datamatrix/TCAF.pickle'
  with open(f, 'wb') as handle:
    pickle.dump(DM_TCAF, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return


# CalculationTree RUNNING PREPROCESSING ----------------------------------------
DM_TCAF_health_diet = TCAF_health_diet_preprocessing()

# CalculationTree RUNNING PICKLE CREATION --------------------------------------
years_ots = create_years_list(1990, 2023, 1)  # make list with years from 1990 to 2015
years_fts = create_years_list(2025, 2050, 5)
years_all = years_ots + years_fts
database_from_csv_to_datamatrix(years_ots, years_fts, DM_TCAF_health_diet)
