import numpy as np
from model.common.auxiliary_functions import interpolate_nans, add_missing_ots_years, linear_fitting_ots_db, linear_fitting, create_years_list, dm_match_countries
#from _database.pre_processing.api_routines_CH import get_data_api_CH
from scipy.stats import linregress
import pandas as pd
import pycountry
import unicodedata
from rapidfuzz import process
import faostat
import copy
from _database.pre_processing.api_routines_CH import get_data_api_CH
import os
import re
from model.common.data_matrix_class import DataMatrix
from model.common.constant_data_matrix_class import ConstantDataMatrix
from model.common.io_database import read_database, read_database_fxa, edit_database, database_to_df, dm_to_database, database_to_dm, database_to_df_robust
from model.common.io_database import read_database_to_ots_fts_dict, read_database_to_ots_fts_dict_w_groups, read_database_to_dm
from model.common.interface_class import Interface
from model.common.auxiliary_functions import compute_stock,  filter_geoscale, calibration_rates, filter_DM, add_dummy_country_to_DM, my_pickle_dump
from model.common.auxiliary_functions import read_level_data, simulate_input, harmonize_countries, country_to_iso3
from scipy.optimize import linprog
import pickle
import json
import os
import numpy as np
import time

# CalculationLeaf other functions


def normalize(name):
    """Remove accents and normalize string."""
    name = name.strip()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return name


def name_to_iso3(name):
    """Convert country name to ISO3 code."""
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def match_countries_iso3(list_faostat, list_biodiversity):
    """
    Convert both country lists to ISO3 codes
    and return mapping biodiversity_name -> faostat_name
    """

    # --- Convert FAOSTAT countries to ISO3 ---
    faostat_iso = {}
    for country in list_faostat:
        iso = name_to_iso3(normalize(country))
        if iso:
            faostat_iso[iso] = country

    # --- Convert biodiversity countries to ISO3 and match ---
    mapping = {}
    unmatched = []

    for country in list_biodiversity:
        iso = name_to_iso3(normalize(country))

        if iso and iso in faostat_iso:
            mapping[country] = faostat_iso[iso]
        else:
            mapping[country] = None
            unmatched.append(country)

    return mapping, unmatched

# SimulateInteractions crop to TCAF
def simulate_crop_to_TCAF_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../_database/data/interface/crop_to_TCAF.pickle")
    with open(f, 'rb') as handle:
        dm_production = pickle.load(handle)
    return dm_production

# SimulateInteractions land-use to TCAF
def simulate_landuse_to_TCAF_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    f = os.path.normpath(
      os.path.join(current_file_directory,
                  "../../data/interface/land-use_to_TCAF.pickle")
    )
    with open(f, 'rb') as handle:
        dm_cropland= pickle.load(handle)
    return dm_cropland

# CalculationLeaf TCAF MONETIZATION FACTORS
def TCAF_MF_preprocessing():

  # Data -----------------------------------------------------------------------
  df_data = pd.read_excel('data/monetization-factors/TCAF_monetization-factors.xlsx',
                          sheet_name='MFs')
  df_data = df_data[['name', 'value']]

  # Format as constant datamatrix
  CDM_MF = ConstantDataMatrix.create_from_constant(df_data, num_cat=0)
  return CDM_MF

# CalculationLeaf TCAF - Health diet
def TCAF_health_diet_preprocessing():

  # ----------------------------------------------------------------------------
  # DALYs
  # ----------------------------------------------------------------------------

  # Data -----------------------------------------------------------------------
  df_dalys = pd.read_csv('data/health-diet/Disease_sex_DALYs_2023.csv')

  # Preprocessing --------------------------------------------------------------

  # Filter
  df_dalys = df_dalys[['location', 'sex', 'cause', 'year', 'val']]

  # Rename cols
  df_dalys.rename(columns={'location':'Country', 'year':'Years', 'val':'value'}, inplace=True)

  # Groupby gender (sum total DALYs)
  df_dalys = df_dalys.groupby(['Country', 'Years', 'cause'], as_index=False)[
    'value'].sum()

  # Rename terms
  cause_map = {
    'Colon and rectum cancer': 'CRC',
    'Diabetes mellitus type 2': 'DT2',
    'Intracerebral hemorrhage': 'ICH',
    'Ischemic heart disease': 'IHD',
    'Ischemic stroke': 'IS',
    'Subarachnoid hemorrhage': 'SH',
    'Tracheal, bronchus, and lung cancer': 'TBLC',
    'Esophageal cancer': 'EC'
  }
  df_dalys['cause'] = df_dalys['cause'].replace(cause_map)

  # Create variables name
  df_dalys['variables'] = 'tcaf_health-diet_dalys_' + df_dalys['cause'] \
                                      + '[DALYs/y]'

  # Filter
  df_dalys = df_dalys[['Country', 'Years', 'variables', 'value']]

  # Format as dm  --------------------------------------------------------------

  df_dalys_pivot = df_dalys.pivot_table(index=['Country', 'Years'],
                                    columns='variables',
                                    values='value').reset_index()
  dm_health_dalys = DataMatrix.create_from_df(df_dalys_pivot, num_cat=1)

  """# Compute total DALYs
  dm_temp = dm_health_dalys.groupby({'combined': '.*'}, dim='Categories1', regex=True, inplace=False)
  dm_health_dalys.append(dm_temp, dim='Categories1')"""

  # Linear fitting to expand the constant value
  linear_fitting(dm_health_dalys, years_all)


  # ----------------------------------------------------------------------------
  # PAF
  # ----------------------------------------------------------------------------

  # Data -----------------------------------------------------------------------
  df_data = pd.read_excel('data/health-diet/PAF_Idriss.xlsx',
                            sheet_name='Sheet1')

  # Preprocessing --------------------------------------------------------------

  # Average PAF per risk factor, cause, grams
  df_data_grouped = df_data.groupby(['Risk_Factor','cause','grams'])['paf'].mean().reset_index()

  # Add gender

  """# Combined PAF = 1 - PROD(1-PAFi)
  df_paf_comb = df_data_grouped.copy()
  df_paf_comb = (
    df_paf_comb
    .groupby(['Risk_Factor', 'grams'])['paf']
    .apply(lambda x: 1 - np.prod(1 - x))
    .reset_index()
  )
  df_paf_comb['cause'] = 'Combined'"""

  # Concat dfs
  df_tcaf_health_diet = df_data_grouped
  #df_tcaf_health_diet = pd.concat([df_data_grouped, df_paf_comb])

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
    'Legumes': 'crop-pulse',
    'Milk': 'pro-liv-abp-dairy-milk',
    'Nuts': 'crop-oilcrop',
    'Omega_3': 'omega',
    'PUFA': 'pufa',
    'Processed_Meat': 'pro-liv-meat-processed',
    'Red_Meat': 'pro-liv-meat-red',
    'SSB': 'pro-bev-ssb',
    'Vegetables': 'crop-veg'
  }
  df_tcaf_health_diet['Risk_Factor'] = df_tcaf_health_diet['Risk_Factor'].replace(risk_factor_map)

  # Rename terms
  df_tcaf_health_diet['cause'] = df_tcaf_health_diet['cause'].replace(cause_map)

  # Create variables name
  df_tcaf_health_diet['variables'] = 'tcaf_health-diet_paf_' + \
                                     df_tcaf_health_diet['cause'] \
                                      + '[-]'

  # Format as separate dm, according to the risk factor (or food categories)
  # Note : here, the intake is processed as the 'Years' dimensions, and renamed
  # afterwards. Therefore, this DM has not timescale
  DM_TCAF_health_diet_paf = {}

  var_total = [
    'tcaf_health-diet_paf_CRC','tcaf_health-diet_paf_DT2',
    'tcaf_health-diet_paf_ICH', 'tcaf_health-diet_paf_IHD',
    'tcaf_health-diet_paf_IS', 'tcaf_health-diet_paf_SH',
    'tcaf_health-diet_paf_EC', 'tcaf_health-diet_paf_TBLC'
  ]

  for rf in df_tcaf_health_diet["Risk_Factor"].unique():
    sub_df = df_tcaf_health_diet[df_tcaf_health_diet["Risk_Factor"] == rf].copy()
    sub_df_pivot = sub_df.pivot_table(index=['Country', 'Years'], columns='variables', values='value').reset_index()
    dm = DataMatrix.create_from_df(sub_df_pivot, num_cat=0)
    dm.dim_labels[1] = 'Intake [g/day/cap]'
    # Add dummies
    var_rf = dm.col_labels['Variables']
    var_missing = set(var_total) - set(var_rf)
    for var in var_missing:
      dm.add(0.0, dummy=True, col_label=var,dim='Variables', unit='-')
    DM_TCAF_health_diet_paf[rf] = dm

  return DM_TCAF_health_diet_paf, dm_health_dalys


# CalculationLeaf TCAF - Biodiversity

def TCAF_biodiversity_preprocessing():
  import sys
  import model.common.data_matrix_class as dmc

  sys.modules["common.data_matrix_class"] = dmc
  
  # Read biodiversity_world.csv from TCAF Datapool
  current_file_directory = os.path.dirname(os.path.abspath(__file__))
  f = os.path.join(
    current_file_directory,
    "data/data_pool/biodiversity_world.csv"
  )
  df_biodiversity_world = pd.read_csv(f)

  # Read biodiversity_switzerland.csv from TCAF Datapool
  current_file_directory = os.path.dirname(os.path.abspath(__file__))
  f = os.path.join(
    current_file_directory,
    "data/data_pool/biodiversity_switzerland.csv"
  )
  df_biodiversity_ch = pd.read_csv(f)

  # Format as Datamatrix (CH)
  lever = 'dummy'
  df_biodiversity_ch['lever'] = lever
  df_ots, df_fts = database_to_df_robust(df_biodiversity_ch, lever, level='all')
  df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
  dm_biodiversity_ch = DataMatrix.create_from_df(df_ots, num_cat=2)

  # Format as Datamatrix (world)
  lever = 'dummy'
  df_biodiversity_ch['lever'] = lever
  df_ots, df_fts = database_to_df_robust(df_biodiversity_world, lever, level='all')
  df_ots = df_ots.drop(columns=[lever])  # Drop column with lever name
  dm_biodiversity_world = DataMatrix.create_from_df(df_ots, num_cat=0)

  # Group same country when necessary using mean values
  dm_biodiversity_world.groupby({'united states of america': 'united states.*'}, dim='Country', aggregation='mean', regex=True, inplace=True)
  dm_biodiversity_world.groupby({'indonesia': 'indonesia.*'},dim='Country', aggregation='mean', regex=True, inplace=True)

  # Create copies for to divide  "baltic states" in ["Estonia", "Latvia", "Lithuania"]
  for country_baltic in ["Estonia", "Latvia", "Lithuania"]:
    dm_biodiversity_world.add(0.0, dummy=True, col_label=country_baltic,dim='Country')
    dm_biodiversity_world[country_baltic,:,:] = dm_biodiversity_world['baltic states',:,:]
  dm_biodiversity_world.drop(dim='Country', col_label='baltic states')

  # Read pickle from landuse_module to TCAF
  dm_cropland = simulate_landuse_to_TCAF_input()

  # Format country names to match the ones in dm_production

  # Manual fixes for typos & alternative names
  # -----------------------------
  mapping_manual = {
    "hungaria": "Hungary",
    "sri lanca": "Sri Lanka",
    "mauretania": "Mauritania",
    "tunesia": "Tunisia",
    "ivory coast": "Côte d'Ivoire",
    "zaire": "Democratic Republic of the Congo",
    "south korea": "Democratic People's Republic of Korea",
    # if you mean DPRK; otherwise see below
    "north korea": "Democratic People's Republic of Korea",
    "russia": "Russian Federation",
    "bolivia": "Bolivia",
    "netherlands": "Netherlands (kingdom of the)",
    # depending on your ISO mapping, could be "Bolivia (Plurinational State of)"
    "iran": "Iran",
    "venezuela": "Venezuela (Bolivarian Republic of)",
    "turkey": "Türkiye",
    "us": "United States of America",
    "uk": "United Kingdom of Great Britain and Northern Ireland",
    "dominican rep": "Dominican Republic",
    "greenland": "Greenland",
    "new guinea": "Papua New Guinea",
    "surinam": "Suriname",
    "china": "China, mainland"
  }

  # Rename with manual fixes
  for key in mapping_manual.keys():
    dm_biodiversity_world.rename_col(key, mapping_manual[key], 'Country')

  list_faostat = dm_cropland.col_labels['Country']
  list_biodiversity = dm_biodiversity_world.col_labels['Country']

  mapping, unmatched = harmonize_countries(list_biodiversity, list_faostat)

  # Format country names to match the ones in dm_production
  for key in mapping.keys():
    if mapping[key] is not None:
      dm_biodiversity_world.rename_col(key, mapping[key], 'Country')

  # Add missing countries with dummy values
  dm_match_countries(dm_biodiversity_world, dm_cropland, parameter='perfect match')

  # Format separately between Switzerland and other countries
  dm_biodiversity_world.drop(dim='Country', col_label='Switzerland')
  DM_TCAF_biodiversity = {
    'TCAF-biodiversity-CH': dm_biodiversity_ch,
    'TCAF-biodiversity-world': dm_biodiversity_world
  }

  return DM_TCAF_biodiversity

# CalculationLeaf CREATE PICKLE
def database_from_csv_to_datamatrix(years_ots, years_fts):

  # Make list with years from 2020 to 2050 (steps of 5 years)
  years_all = years_ots + years_fts

  # FixedAssumptionsToDatamatrix -----------------------------------------------

  # Initialise
  dict_fxa = {}

  # Add in fxa
  dict_fxa['health-diet_paf'] = DM_TCAF_health_diet
  dict_fxa['health-diet_dalys'] = dm_health_dalys
  dict_fxa['biodiversity'] = DM_TCAF_biodiversity

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
  dict_const = CDM_MF

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
years_ots = create_years_list(1990, 2023, 1)  # make list with years from 1990 to 2015
years_fts = create_years_list(2025, 2050, 5)
years_all = years_ots + years_fts
DM_TCAF_health_diet, dm_health_dalys = TCAF_health_diet_preprocessing()
DM_TCAF_biodiversity = TCAF_biodiversity_preprocessing()
CDM_MF = TCAF_MF_preprocessing()


# CalculationTree RUNNING PICKLE CREATION --------------------------------------
database_from_csv_to_datamatrix(years_ots, years_fts)
