# ======================  IMPORT PACKAGES & DATA  ===================================================================================================================================================

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from plot import plot_transport_variable_2
from model.common.auxiliary_functions import linear_fitting, my_pickle_dump, create_years_list, filter_DM, add_dummy_country_to_DM, sort_pickle
import os

from model.common.data_matrix_class import DataMatrix


def update_from_subdm(dm, subdm):
  # Copy or reference
  idx_all = []

  # Sort both data models by each dimension
  for dim in subdm.dim_labels:
    dm.sort(dim)
    subdm.sort(dim)

    # Validate that all update labels exist in the target
    if not set(subdm.col_labels[dim]).issubset(dm.col_labels[dim]):
      raise ValueError(
        f"Labels in dim '{dim}' from 'from_dm' not found in 'of_dm'")

  # Build indexing tuple
  for dim in subdm.dim_labels:
    # Get the list of indices for this dimension in dm_out
    idx_dim = [dm.idx[label] for label in subdm.col_labels[dim]]
    idx_all.append(np.array(idx_dim))

  # Convert idx_all into a tuple of index arrays for fancy indexing
  # Each idx_dim must be broadcasted into a meshgrid to match the shape of from_dm.array
  idx_mesh = np.ix_(*idx_all)

  # Update the corresponding subarray
  dm.array[idx_mesh] = subdm.array

  return dm


def merge_dm(dm_A, dm_B, same_dim = ['Country', 'Years', 'Variables'], union_dim=['Categories2'], along_dim='Categories1'):

    if isinstance(along_dim, list):
        raise ValueError("You can only merge along one dimension")

    for dim in same_dim:
        assert set(dm_A.col_labels[dim]) == set(dm_B.col_labels[dim])

    for dim in union_dim:
        cols_A = list(set(dm_A.col_labels[dim]) - set(dm_B.col_labels[dim]))
        dm_B.add(np.nan, dim=dim, col_label=cols_A, dummy=True)
        cols_B = list(set(dm_B.col_labels[dim]) - set(dm_A.col_labels[dim]))
        dm_A.add(np.nan, dim=dim, col_label=cols_B, dummy=True)

    dm = dm_A.copy()
    if not set(dm_B.col_labels[along_dim]).issubset(set(dm.col_labels[along_dim])):
      dm.append(dm_B, dim=along_dim)

    return dm


def pkm_monde_ots(dm_pkm_orig, value_2024):

    dm_pkm = dm_pkm_orig
    dm_pkm.add(value_2024, dim='Years', col_label=[2024], dummy=True)
    dm_pkm[0, 2023, 0, 0] = np.nan
    dm_pkm[0, 2022, 0, 0] = np.nan
    dm_pkm.fill_nans(dim_to_interp='Years')
    dm_pkm.drop(col_label = [2024], dim='Years')
    # Impose 2020 value from US 2019/2020 decrease rate
    dm_pkm[0, 2020, 0, 0] = 3103.0

    return dm_pkm


def pkm_monde_fts(value_2024, pkm_rates):
    DM = dict()
    # années de 2024 à 2050
    years = np.arange(2025, 2051)

    i = 1 #pour nommer dans la boucle les scénarios 1,2,3,4
    for rate in pkm_rates:
        # calcul exponentiel : value[t] = 1000 * (1+rate)^(t-2023)
        n_years = years - 2024
        values_pk = value_2024 * (1 + rate) ** n_years

        # construire le DataFrame pour ce scénario
        df_pkm = pd.DataFrame({
            'Country': 'Switzerland',
            'Years': years,
            'tra_pkm-cap_aviation[pkm/cap]': values_pk,
        })

        # On ajoute au dictionnaire sous une clé descriptive
        key = i
        i = i+1
        dm_tmp = DataMatrix.create_from_df(df_pkm, num_cat=1)
        DM[key] = dm_tmp.filter({'Years': years_fts}) #num cat il lit pkm_aviation tout ce qui est avant _ ca devient variable et tout apres cest la catégorie
    return DM


def adj_pkm_monde_ots(dm_pkmsuisse_ots, dm_pkm_ots):
  idx = dm_pkm_ots.idx
  dm_pkm_ots.array[:, 0:idx[2005], ...] = np.nan
  dm_pkm_ots.append(dm_pkmsuisse_ots, dim='Variables')
  dm_pkm_ots.operation('tra_pkm-cap', '/', 'tra_pkm-suisse-cap',
                       out_col='ratio', unit='%')
  linear_fitting(dm_pkm_ots, years_ots,
                 based_on=create_years_list(2006, 2017, 1))
  dm_pkm_ots[:, :, 'ratio', ...] = np.maximum(1.05,
                                              dm_pkm_ots[:, :, 'ratio', ...])

  dm_pkm_ots.operation('ratio', '*', 'tra_pkm-suisse-cap',
                       out_col='tra_pkm-cap_adj', unit='pkm/cap')
  dm_pkm_ots.filter({'Variables': ['tra_pkm-cap_adj']}, inplace=True)
  dm_pkm_ots.rename_col('tra_pkm-cap_adj', 'tra_pkm-cap', dim='Variables')
  return dm_pkm_ots



def pkm_suisse_fts(pkm_rates, dm_pkmsuisse_ots, years_ots, years_fts):

    missing_years = set(years_ots) - set(dm_pkmsuisse_ots.col_labels['Years'])
    dm_pkmsuisse_ots.add(np.nan, 'Years', list(missing_years), dummy=True)
    dm_pkmsuisse_ots.sort('Years')

    # 4) Dans l’array, la forme est (pays, années, cat1, variables)
    value_2023 = dm_pkmsuisse_ots['Switzerland', 2023, 0, 0]

    # années de 2024 à 2050
    years = np.arange(2024, 2051)

    DM = dict()
    i = 1 #pour nommer dans la boucle les scénarios 1,2,3,4
    for rate in pkm_rates:
        # calcul exponentiel : value[t] = 1000 * (1+rate)^(t-2023)
        n_years = years - 2023
        values_pk = value_2023 * (1 + rate) ** n_years

        # construire le DataFrame pour ce scénario
        df_pkm = pd.DataFrame({
            'Country': 'Switzerland',
            'Years': years,
            'tra_pkm-cap_aviation[pkm/cap]': values_pk
        })

        # On ajoute au dictionnaire sous une clé descriptive
        key = i
        i = i+1
        dm_pkm_tmp = DataMatrix.create_from_df(df_pkm, num_cat=1).filter({'Years': years_fts})
        dm_pkm_tmp.rename_col('tra_pkm-cap', 'tra_pkm-suisse-cap', dim='Variables')
        DM[key] = dm_pkm_tmp

    return DM


def occupancy_monde_ots(dm_pkm_suisse, dm_pkm_tot, dm_occupancy_ots,
                        dm_occupancyworld_ots):
  # occupancy = (pkm_CH * occupancy_CH + (pkm_tot - pkm_CH) * occupancy_world) / pkm_tot
  array_occupancyponderee = (dm_pkm_suisse[...] * dm_occupancy_ots[...] + (
      dm_pkm_tot[...] - dm_pkm_suisse[...]) * dm_occupancyworld_ots[...]) / (
                            dm_pkm_tot[...])
  dm_occupancyworld_ots.add(array_occupancyponderee, dim='Variables',
                            col_label='tra_passenger_occupancy_ponderee',
                            unit='%')
  dm_occupancyworld_ots.drop(dim='Variables',
                             col_label=['tra_passenger_occupancy'])
  dm_occupancyworld_ots.rename_col('tra_passenger_occupancy_ponderee',
                                   'tra_passenger_occupancy', dim='Variables')
  dm_occupancyworld_ots.fill_nans(dim_to_interp='Years')
  dm_occupancyworld_ots.change_unit('tra_passenger_occupancy', old_unit='%',
                                    new_unit='pkm/vkm', factor=1)

  return dm_occupancyworld_ots

def occupancy_suisse_ots(file, years_ots):
  # création de la datamatrix pour occupancy_suisse_ots à partir du fichier excel
  df_occupancy_ots = pd.read_excel(file)
  dm_occupancy_ots = DataMatrix.create_from_df(df_occupancy_ots, num_cat=1)

  # rajout des années manquantes et extrapolation constante
  missing_years = set(years_ots) - set(dm_occupancy_ots.col_labels['Years'])
  dm_occupancy_ots.add(np.nan, 'Years', list(missing_years), dummy=True)
  dm_occupancy_ots.sort('Years')
  dm_occupancy_ots.fill_nans('Years')

  # For aviation it is actually in pkm/skm
  dm_occupancy_ots.change_unit('tra_passenger_occupancy', factor=1,
                               old_unit='%', new_unit='pkm/vkm')
  return dm_occupancy_ots

def occupancy_fts(value_2023, scenarios):
  # 3) Années à générer
  years = list(range(2024, 2051))
  DM = dict()
  # pour nommer dans la boucle les scénarios 1,2,3,4
  for lev in range(4):
    # Transformer le scénario en valeur finale d'occupancy (%) en 2050
    value_2050 = scenarios[lev]

    # On crée d'abord une grille linéaire de 2023→2050
    all_years = [2023] + years
    all_values = np.linspace(value_2023, value_2050, num=len(all_years))

    # On construit le DataFrame, puis on retire la ligne 2023
    df_scenario = pd.DataFrame({
      'Country': 'Switzerland',
      'Years': all_years,
      'tra_passenger_occupancy[%]': all_values,
    }).query("Years >= 2024").reset_index(drop=True)

    # On convertit en DataMatrix
    dm_scenario = DataMatrix.create_from_df(df_scenario, num_cat=0)

    # On ajoute au dictionnaire sous une clé descriptive
    dm_scenario.rename_col('tra_passenger_occupancy',
                           'tra_passenger_occupancy_aviation',
                           dim='Variables')
    dm_scenario.change_unit('tra_passenger_occupancy_aviation', factor=1,
                            old_unit='%', new_unit='pkm/vkm')
    dm_scenario.filter({'Years': years_fts}, inplace=True)
    dm_scenario.deepen()
    DM[lev + 1] = dm_scenario
  return DM


def occupancy_global_ots(file, years_ots):
    # création de la datamatrix pour occupancy_monde_ots à partir du fichier excel
    df_occupancyworld_ots = pd.read_excel(file)
    dm_occupancyworld_ots = DataMatrix.create_from_df(df_occupancyworld_ots, num_cat=1)

    # rajout des années manquantes et extrapolation constante
    missing_years = set(years_ots) - set(dm_occupancyworld_ots.col_labels['Years'])
    dm_occupancyworld_ots.add(np.nan, 'Years', list(missing_years), dummy = True)
    dm_occupancyworld_ots.sort('Years')
    dm_occupancyworld_ots.fill_nans('Years')
    return dm_occupancyworld_ots


def compute_seats(dm_pkm, dm_utilirate, dm_occupancy, dm_pop, subfix):
    # seat ots = pkm_ots / occupancy_ots (pkm/skm) / utilisation-rate
    arr_seat = dm_pkm.array/dm_occupancy.array/dm_utilirate.array*dm_pop.array[..., np.newaxis]
    arr_skm = dm_pkm.array/dm_occupancy.array*dm_pop.array[..., np.newaxis]
    dm_occupancy.add(arr_seat, dim='Variables', col_label='tra_passenger_seats'+subfix, unit='seat')
    dm_occupancy.add(arr_skm, dim='Variables', col_label='tra_passenger-demand-skm'+subfix, unit='skm')
    dm_seats = dm_occupancy.filter({'Variables': ['tra_passenger_seats'+subfix]})
    dm_skm = dm_occupancy.filter({'Variables': ['tra_passenger-demand-skm'+subfix]})
    return dm_seats, dm_skm


def compute_new_veh_max(dm_max):
  years_fts_all = create_years_list(2025, 2050, 1)
  missing_years = list(set(years_fts_all) - set(years_fts))
  dm_max.add(np.nan, dim='Years', col_label=missing_years, dummy=True)
  dm_max.sort(dim='Years')
  dm_max.fill_nans('Years')
  # Compute new fleet as s(t) = s(t-1) + new(t) - waste(t), by assuming waste(t) = 0 for H2, BEV
  # then n(t) = s(t) - s(t-1)
  dm_max.lag_variable('tra_vehicles-max', shift=1, subfix='_tm1')
  dm_max.operation('tra_vehicles-max', '-', 'tra_vehicles-max_tm1',
                   out_col='tra_new-vehicles-max', unit='seat')
  dm_max.add(0, dim='Variables', col_label='tra_vehicles-waste-max', dummy=True,
             unit='seat')
  dm_max.drop(col_label='tra_vehicles-max_tm1', dim='Variables')
  dm_max.filter({'Years': years_fts_all}, inplace=True)
  return dm_max


def compute_share_emissions(dm_skm_CH, dm_skm_abroad):
    # Emissions = skm x MJ/skm x CO2/MJ
    # Emissions_CH = skm_CH x (MJ/skm x CO2/MJ)_CH
    # Emissions_abroad = skm_abroad x (MJ/skm x CO2/MJ)_abroad
    # Emission_CH / Emissions_abroad = skm_CH / skm_abroad ----> seats_CH / seats_abroad
    ## Multiply by 2 the Swiss seats to obtain emissions according to sold quantity principle
    #dm_skm_CH[:, :, 'tra_passenger-demand-skm_CH', ...] = 2 * dm_skm_CH[:, :, 'tra_passenger-demand-skm_CH', ...]
    dm_skm_CH.append(dm_skm_abroad, dim='Variables')
    dm_skm_CH.operation('tra_passenger-demand-skm_CH', '/', 'tra_passenger-demand-skm',
                        out_col='tra_share-emissions-local', unit='%')
    dm_emiss_share = dm_skm_CH.filter({'Variables': ['tra_share-emissions-local']})
    # dm_emiss_share.add(np.nan, col_label=years_fts, dim='Years', dummy=True)
    #inear_fitting(dm_emiss_share, years_fts, based_on=create_years_list(2015, 2019, 1))
    return dm_emiss_share


def compute_tech_new_from_newfleet(dm_fleet_ofts, dm_max):
  # Extract new, fleet, waste from pickle saved during run
  dm_new_veh_demand = dm_fleet_ofts.filter(
    {'Variables': ['tra_passenger_new-vehicles', 'tra_passenger_vehicle-fleet',
                   'tra_passenger_vehicle-waste']})
  dm_new_veh_demand.rename_col(
    ['tra_passenger_new-vehicles', 'tra_passenger_vehicle-fleet',
     'tra_passenger_vehicle-waste'],
    ['tra_new-vehicles-max_tot', 'tra_vehicles-max_tot',
     'tra_vehicles-waste-max_tot'], dim='Variables')
  for var in dm_new_veh_demand.col_labels['Variables']:
    dm_new_veh_demand.change_unit(var, factor=1, old_unit='number',
                                  new_unit='seat')

  dm_new_veh_demand.deepen(based_on='Variables')

  # Add all Years from 2025 to 2050
  years_fts_all = create_years_list(2025, 2050, 1)
  missing_years = list(set(years_fts_all) - set(years_fts))
  dm_new_veh_demand.add(np.nan, dim='Years', dummy=True,
                        col_label=missing_years)
  dm_new_veh_demand.sort('Years')
  dm_new_veh_demand.fill_nans('Years')
  dm_new_veh_demand.filter({'Years': years_fts_all}, inplace=True)

  # Append max fleet of new tech with fleet of all tech from run
  dm_max.append(dm_new_veh_demand, dim='Categories2')

  dm_max.add(0, col_label='kerosene', dim='Categories2', dummy=True)
  # ICE = tot - BEV - H2
  dm_max[0, :, :, 'aviation', 'kerosene'] = dm_max[0, :, :, 'aviation',
                                            'tot'] - dm_max[0, :, :, 'aviation',
                                                     'BEV'] \
                                            - dm_max[0, :, :, 'aviation', 'H2']

  # If the demand for new planes is less than the new tech available
  mask = np.any(dm_max.array < 0, axis=-1)
  if mask.any():
    dm_tot = dm_max.filter({'Categories2': ['tot']})
    dm_max.drop('Categories2', 'tot')
    idx = dm_max.idx
    dm_max.array[mask, idx['kerosene']] = 0
    dm_max.normalise('Categories2')
    dm_max.array[...] = dm_max.array[...] * dm_tot.array[...]

  # LEVEL 4 - HIGHEST PENETRATION OF BEV, H2
  dm_new_tech = dm_max.filter({'Variables': ['tra_new-vehicles-max'],
                               'Categories2': ['kerosene', 'BEV', 'H2']})
  dm_new_tech.normalise(dim='Categories2', inplace=True, keep_original=False)
  dm_new_tech.rename_col('tra_new-vehicles-max',
                         'tra_passenger_technology-share_new', dim='Variables')
  dm_new_tech.filter({'Years': years_fts}, inplace=True)
  dm_new_tech.fill_nans('Years')
  return dm_new_tech, dm_max

def add_missing_cat_DM(DM_orig, DM_new):
  for key in DM_orig.keys():
    if isinstance(DM_orig[key], dict):
      if 'freight' not in key and key != 'passenger_aviation-pkm':
        add_missing_cat_DM(DM_orig[key], DM_new[key])
    else:
      dm_orig = DM_orig[key]
      dm_new = DM_new[key]
      if 'Categories1' in dm_new.dim_labels:
        if 'aviation' in dm_new.col_labels[
          'Categories1'] and 'freight' not in str(
          key) and key != 'passenger_aviation-pkm':
          dm_new.filter({'Categories1': ['aviation']}, inplace=True)
          dm = update_from_subdm(dm_orig, dm_new)
          DM_orig[key] = dm.copy()
  return


###################################################################
#########          AVIATION - SWITZERLAND         #################
###################################################################

# from training.transport_module_notebook import col_labels
# 2. Liste des années OTS
years_ots = create_years_list(start_year= 1990, end_year= 2023, step = 1)
# 2. Liste des années FTS uniquement (2025 à 2050)
years_fts = create_years_list(2025, 2050, 5)

# Read PICKLEs
data_file = '../../../data/datamatrix/transport.pickle'
with open(data_file, 'rb') as handle:
    DM_transport = pickle.load(handle)

data_file = '../../../data/datamatrix/lifestyles.pickle'
with open(data_file, 'rb') as handle:
    DM_lifestyles = pickle.load(handle)

# Filter Switzerland
filter_DM(DM_transport, {'Country': ['Switzerland']})
# importation des données sur la population suisse
dm_pop = DM_lifestyles['ots']['pop']['lfs_population_'].filter({'Country': ['Switzerland']})
dm_pop_fts = DM_lifestyles['fts']['pop']['lfs_population_'][1].filter({'Country': ['Switzerland']})

# Dictionaries to store temporary variables
DM_fts = {'fts': dict()}
DM_ots = {'ots': dict()}

# ======================  DATA MATRIX CREATION  ===========================================================================================================================================================================

# -------------------- PKM (ots et fts) -------------------------------------------------------------------------------------------------------------------------
# section DONNEES PKM-MONDE OTS
# 1) Récupérer les datas ots et les placer dans DM_ots
dm_pkm_orig = DM_transport['ots']['passenger_aviation-pkm']
# IATA, Global Air Passenger Demand Reaches Record High in 2024
# https://www.iata.org/en/pressroom/2025-releases/2025-01-30-01/
value_2024 = dm_pkm_orig[0, 2019, 0, 0] * 1.038
# Adjust dm_pkm_orig for post-covid
dm_pkm_ots = pkm_monde_ots(dm_pkm_orig, value_2024)
DM_transport['ots']['passenger_aviation-pkm'] = dm_pkm_ots
del dm_pkm_ots, dm_pkm_orig


#section DONNEES PKM-MONDE FTS
# les trois taux annuels à tester
pkm_rates = [0.04, 0.029, 0.02, -0.02]  #scénario 1 et 2 sont les mêmes
DM = pkm_monde_fts(value_2024, pkm_rates)
DM_transport['fts']['passenger_aviation-pkm'] = DM
del pkm_rates, DM, value_2024

# -------------------PASSENGER_VEC_EFFICIENCY_NEW OTS ------------------------------------------------------------------------------

#section  PASSENGER_VEC_EFFICIENCY_NEW OTS
#ICE, BEV, H2
df_passengereff_ots = pd.read_excel('data/aviation_energy_intensity_fleet.xlsx', sheet_name = 'Feuil1')
dm_passengereff_ots = DataMatrix.create_from_df(df_passengereff_ots, num_cat=2)
dm_passengereff_orig = DM_transport['ots']['passenger_veh-efficiency_new'].copy()
#dm_passengereff = merge_dm(dm_passengereff_orig, dm_passengereff_ots)
dm_passengereff = update_from_subdm(dm_passengereff_orig, dm_passengereff_ots)
DM_transport['ots']['passenger_veh-efficiency_new'] = dm_passengereff
del dm_passengereff_orig, dm_passengereff_ots


#section DONNES PKM-SUISSE OTS
#file = 'data/pkm_suisse.xlsx'
file = 'data/aviation_pkm_suisse.csv'
dm_pkm_ots = DM_transport['ots']['passenger_aviation-pkm'].copy()
df = pd.read_csv(file, sep=';', decimal=',')
df.drop(['Unnamed: 0'], axis=1, inplace=True)
dm_pkmsuisse_ots = DataMatrix.create_from_df(df, num_cat=1)
dm_pkmsuisse_ots.rename_col('tra_pkm-cap', 'tra_pkm-suisse-cap', dim='Variables')

dm_pkm_ots = adj_pkm_monde_ots(dm_pkmsuisse_ots, dm_pkm_ots)
DM_transport['ots']['passenger_aviation-pkm'] = dm_pkm_ots.copy()
DM_ots['ots']['passenger_aviation-pkm-suisse'] = dm_pkmsuisse_ots.copy()


#section DONNES PKM-SUISSE FTS
pkm_rates = [0.04, 0.029, 0.02, 0.02]
DM = pkm_suisse_fts(pkm_rates, dm_pkmsuisse_ots, years_ots, years_fts)
DM_fts['fts']['passenger_aviation-pkm-suisse'] = DM
del DM


# -------------------- OCCUPANCY_OTS --------------------------------------------------------------------------------------

#section DONNEES OCCUPANCY SUISSE OTS

file = 'data/aviation_occupancy_ots.xlsx'
dm_occupancy_ots = occupancy_suisse_ots(file, years_ots)
DM_ots['ots']['passenger_occupancy-suisse'] = dm_occupancy_ots


#section DONNEES OCCUPANCY MONDE OTS


file = 'data/aviation_occupancy_pondere_ots.xlsx'
dm_occupancyworld_ots = occupancy_global_ots(file, years_ots)

#section CALCUL OCCUPANCY PONDEREE OTS
dm_pkm_suisse = DM_ots['ots']['passenger_aviation-pkm-suisse'].copy()
dm_pkm_tot = DM_transport['ots']['passenger_aviation-pkm'].copy()
dm_occupancy_monde_ots = occupancy_monde_ots(dm_pkm_suisse, dm_pkm_tot, dm_occupancy_ots, dm_occupancyworld_ots)
dm_occupancy_orig = DM_transport['ots']['passenger_occupancy']
dm_occupancy_update = update_from_subdm(dm_occupancy_orig, dm_occupancy_monde_ots)

del dm_pkm_tot, dm_pkm_suisse, dm_occupancy_monde_ots, dm_occupancyworld_ots, dm_occupancy_ots

#ATTENTION, l'occupancy monde dans les OTS est utilisée pour calculer l'occupancy ponderee qui une fois calculée est appelée occupancy monde!!!!

# -------------------- OCCUPANCY_FTS -----------------------------------------------------------------------------------------------

# création des occupancys dans le dictionnaire fts
DM_fts['fts']['passenger_occupancy-suisse'] = {}
DM_fts['fts']['passenger_occupancy-monde'] = {}

#section DONNEES OCCUPANCY SUISSE FTS
# 1) Valeur connue en 2023 (différente façon de faire)
dm_occupancy_ots = DM_ots['ots']['passenger_occupancy-suisse'].copy()
value_2023 = dm_occupancy_ots[0, 2023, 0, 0]
# 2) Les scénarios en fraction (0.75, 0.80, 0.85, 0.90)
scenarios = [0.75, 0.80, 0.85, 0.90]
DM_fts['fts']['passenger_occupancy-suisse'] = occupancy_fts(value_2023, scenarios)


#section DONNEES OCCUPANCY MONDE FTS
dm_occupancyworld_ots = DM_transport['ots']['passenger_occupancy'].filter({'Categories1': ['aviation']})
value_2023 = dm_occupancyworld_ots[0, 2023, 0, 0]
DM = occupancy_fts(value_2023, scenarios)
for lev in DM.keys():
    dm_orig = DM_transport['fts']['passenger_occupancy'][lev].copy()
    dm_orig = update_from_subdm(dm_orig, DM[lev])
    DM_transport['fts']['passenger_occupancy'][lev] = dm_orig

del dm_occupancyworld_ots, dm_occupancy_ots, value_2023, file, DM

# -------------------- PASSENGER_TECHNOLOGY_SHARE_NEW OTS ----------------------------------------------------------------------------------------------------------------------------

#section  PASSENGER_TECHNOLOGY_SHARE_NEW OTS
# création de la datamatrix pour technology_share à partir du fichier excel
df_technoshare_new = pd.read_excel('data/aviation_passenger_technology_share_new_ots.xlsx')
dm_technoshare_new = DataMatrix.create_from_df(df_technoshare_new, num_cat=2)

# Merge aviation tech-share-new with tech-share-new for other mode of transport
dm_technoshare_orig = DM_transport['ots']['passenger_technology-share_new'].copy()
dm_technoshare = update_from_subdm(dm_technoshare_orig, dm_technoshare_new)

DM_transport['ots']['passenger_technology-share_new'] = dm_technoshare


# -------------------PASSENGER_VEC_EFFICIENCY_NEW FTS ------------------------------------------------------------------------------

#section PASSENGER_VEC_EFFICIENCY_NEW FTS

# Étape 0 : accéder aux valeurs 2023 depuis DM_ots
dm = DM_transport['ots']['passenger_veh-efficiency_new']

# Extraire les valeurs 2023 pour chaque techno
value_2023_ICE = dm[0, 2023, 0, 'aviation', 'kerosene']  # ICE = catégorie 2
value_2023_BEV = dm[0, 2023, 0, 'aviation', 'BEV']  # BEV = catégorie 0
value_2023_H2 = dm[0, 2023, 0, 'aviation', 'H2']  # H2  = catégorie 1

# Étape 1 : définir les taux de réduction annuels (%/an) pour ICE
reduction_rates = [0.01, 0.0125, 0.015, 0.02]

# Étape 2 : années futures
years = list(range(2024, 2051))

# Étape 3 : initialiser le dictionnaire
DM_fts['fts']['passenger_veh-efficiency_new'] = {}
i = 1

DM = dict()
# Étape 4 : boucle sur les scénarios
for i, rate in enumerate(reduction_rates, start=1):
    values = [value_2023_ICE * ((1 - rate) ** (year - 2023)) for year in years]

    df = pd.DataFrame({
        'Country': 'Switzerland',
        'Years': years,
        'tra_passenger_veh-efficiency_new_aviation_kerosene[MJ/km]': values #in reality the unit is MJ/seat-km but we put MJ/km for the merging with DM_transport
    })
    # Construire le DataFrame (ordre : BEV, H2, ICE)
    df['tra_passenger_veh-efficiency_new_aviation_BEV[MJ/km]'] = 0.34
    df['tra_passenger_veh-efficiency_new_aviation_H2[MJ/km]'] = 0.78

    # Convertir en DataMatrix
    dm_scenario = DataMatrix.create_from_df(df, num_cat=2)

    # Ajouter au dictionnaire
    DM[i] = dm_scenario.filter({'Years': years_fts})
    i += 1

for lev in DM.keys():
    dm_eff_orig = DM_transport['fts']['passenger_veh-efficiency_new'][lev]
    dm_eff_aviation = DM[lev]
    dm_eff = update_from_subdm(dm_eff_orig, dm_eff_aviation)
    DM_transport['fts']['passenger_veh-efficiency_new'][lev] = dm_eff

# -------------------PASSENGER_UTILISATION_RATE OTS ---------------------------------------------------------------------------------

#section PASSENGER_UTILISATION_RATE OTS
df_utilirate_ots = pd.read_excel('data/aviation_utilisation-rate-OTS.xlsx')
dm_utilirate_ots = DataMatrix.create_from_df(df_utilirate_ots, num_cat=1)
dm_pkm_tmp = DM_ots['ots']['passenger_aviation-pkm-suisse']
dm_utilirate_ots.append(dm_pkm_tmp, dim='Variables')
dm_utilirate_ots.operation('tra_passenger_utilisation-rate', '/', 'tra_pkm-suisse-cap', out_col='ratio', dim='Variables', unit='%')

for yr in [2020, 2021, 2022, 2023]:
    dm_utilirate_ots[0, yr, 'ratio', 0] = np.nan

# !FIXME THIS IS WRONG
linear_fitting(dm_utilirate_ots, years_ots=years_ots, based_on=create_years_list(2005, 2019, 1))

dm_utilirate_ots.operation('ratio', '*', 'tra_pkm-suisse-cap', out_col='tra_passenger_utilisation-rate_new', dim='Variables', unit='vkm/veh')
dm_utilirate_ots[0, 2022, 'tra_passenger_utilisation-rate', 0] = dm_utilirate_ots[0, 2022, 'tra_passenger_utilisation-rate_new', 0]
dm_utilirate_ots[0, 2023, 'tra_passenger_utilisation-rate', 0] = dm_utilirate_ots[0, 2023, 'tra_passenger_utilisation-rate_new', 0]
dm_utilirate_ots.filter({'Variables': ['tra_passenger_utilisation-rate']}, inplace=True)
DM_ots['ots']['passenger_utilization-rate'] = dm_utilirate_ots

# -------------------PASSENGER_UTILISATION_RATE FTS ----------------------------------------------------------------------------------

#section PASSENGER_UTILISATION_RATE FTS
df_utilirate_fts = pd.read_excel('data/aviation_utilisation-rate-FTS.xlsx')
dm_utilirate_fts = DataMatrix.create_from_df(df_utilirate_fts, num_cat=1)
dm_utilirate_fts.filter({'Years': years_fts}, inplace=True)
for lev in range(4):
    dm_orig = DM_transport['fts']['passenger_utilization-rate'][lev+1]
    dm_orig = update_from_subdm(dm_orig, dm_utilirate_fts)
    DM_transport['fts']['passenger_utilization-rate'][lev + 1] = dm_orig

# ------------------- CONSTANT : EMISSION FACTOR ------------------------------------------------------------------------------

#section EMISSION FACTOR
dm_const = DM_transport['constant']
#dm_const.add( np.nan, dim='Categories2', col_label=['SAF', 'H2'], dummy=True)
idx = dm_const.idx
dm_const.array[0, idx['CO2'], idx['kerosene']] = 73.3*3  # gCO2/MJ
dm_const.array[0, idx['CO2'], idx['SAF']] = 73.3*2  # gCO2/MJ
dm_const.array[0, idx['CO2'], idx['H2']] = 73.3*2*0.25  # gCO2/MJ
DM_transport['constant'] = dm_const

# ------------------- FXA : lifetime ----------------------------------------------------------------------------------

#section LIFETIME

df_lifetime = pd.read_excel('data/aviation_vehicles_lifetime.xlsx')
dm_lifetime = DataMatrix.create_from_df(df_lifetime, num_cat=2)
dm_lifetime.add(25, dim='Years', dummy=True, col_label=years_fts)
dm_lifetime_orig = DM_transport['fxa']['passenger_vehicle-lifetime']
dm_lifetime_merge = update_from_subdm(dm_lifetime_orig, dm_lifetime)
DM_transport['fxa']['passenger_vehicle-lifetime'] = dm_lifetime_merge

# ------------------ FXA : PASSENGER_TECH -----------------------------------------------------------------------------

#section tra_passenger_veh-efficiency_fleet
dm_fleet_eff = DM_transport['ots']['passenger_veh-efficiency_new'].filter({'Categories1': ['aviation']})
dm_fleet_eff.rename_col_regex('tra_passenger_veh-efficiency_new', 'tra_passenger_veh-efficiency_fleet', dim='Variables')
dm_fleet_eff.add(np.nan, dim='Years', dummy=True, col_label=years_fts)

#section tra_passenger_technology-share_fleet
dm_fleetshare = DM_transport['ots']['passenger_technology-share_new'].filter({'Categories1': ['aviation']})
dm_fleetshare.rename_col_regex('tra_passenger_technology-share_new', 'tra_passenger_technology-share_fleet', dim='Variables')
dm_fleetshare.add(np.nan, dim='Years', dummy=True, col_label=years_fts)

#section tra_vehicules-max_aviation

df_new_vehicules_max = pd.read_excel('data/aviation_number_seats_max.xlsx', sheet_name='1')
dm_new_vehicules_max = DataMatrix.create_from_df(df_new_vehicules_max, num_cat=2)
years_ots = create_years_list(1990, 2023, 1)
dm_new_vehicules_max.rename_col('tra_new-vehicules-max', 'tra_vehicles-max', dim='Variables')
dm_new_vehicules_max.change_unit('tra_vehicles-max', factor=0.1, old_unit='seat', new_unit='seat')
dm_new_vehicules_max.add(0, dim='Years', dummy=True, col_label=years_ots)
dm_new_vehicules_max.sort('Years')
DM_transport['fxa']['vehicles-max'] = dm_new_vehicules_max


# SEATS ABROAD - OTS
dm_pkm_tmp = DM_transport['ots']['passenger_aviation-pkm'].copy()
dm_utilirate = dm_utilirate_ots
dm_occupancy_ots = DM_transport['ots']['passenger_occupancy'].filter({'Categories1': ['aviation']})
dm_seats_ots, dm_skm_ots = compute_seats(dm_pkm_tmp, dm_utilirate, dm_occupancy_ots, dm_pop, subfix='')

# SEATS ABROAD - FTS
dm_pkm_tmp = DM_transport['fts']['passenger_aviation-pkm'][1].copy()
dm_utilirate = DM_transport['fts']['passenger_utilization-rate'][1].filter({'Categories1': ['aviation']})
dm_occupancy = DM_transport['fts']['passenger_occupancy'][1].filter({'Categories1': ['aviation']})
dm_seats_fts, dm_skm_fts = compute_seats(dm_pkm_tmp, dm_utilirate, dm_occupancy, dm_pop_fts, subfix='')
del dm_pkm_tmp, dm_utilirate, dm_occupancy_ots


# SEATS SUISSE - OTS
# seat ots = pkm_ots / occupancy_ots (pkm/skm) / utilisation-rate
dm_pkm_tmp = DM_ots['ots']['passenger_aviation-pkm-suisse'].copy()
dm_utilirate = dm_utilirate_ots
dm_occupancy_ots = DM_ots['ots']['passenger_occupancy-suisse'].copy()
dm_seats_ots_CH, dm_skm_ots_CH = compute_seats(dm_pkm_tmp, dm_utilirate, dm_occupancy_ots, dm_pop, subfix='_CH')


# SEATS SUISSE - FTS
dm_pkm_tmp = DM_fts['fts']['passenger_aviation-pkm-suisse'][2].copy()
dm_utilirate = DM_transport['fts']['passenger_utilization-rate'][1].filter({'Categories1': ['aviation']})
dm_occupancy = DM_fts['fts']['passenger_occupancy-suisse'][1].copy()
dm_seats_fts_CH, dm_skm_fts_CH = compute_seats(dm_pkm_tmp, dm_utilirate, dm_occupancy, dm_pop_fts, subfix='_CH')
del dm_pkm_tmp, dm_utilirate, dm_occupancy_ots


dm_skm_CH = dm_skm_ots_CH.copy()
dm_skm_abroad = dm_skm_ots.copy()
dm_skm_abroad.append(dm_skm_fts, dim='Years')
dm_skm_CH.append(dm_skm_fts_CH, dim='Years')
dm_emiss_share = compute_share_emissions(dm_skm_CH, dm_skm_abroad)
DM_transport['fxa']['share-local-emissions'] = dm_emiss_share

del dm_skm_CH


# SECTION: VEHICLE-WASTE
# Waste = seats(t-1) * retirement-rate(t)
# The datas of the retirement rate were computed using the JRC statistics on European aviation.
# In particular using the stock of aircarfts and the vehicule retirement.
df_retirrate = pd.read_excel('data/aviation_waste_suisse.xlsx', sheet_name='Feuil2')
dm_retirrate = DataMatrix.create_from_df(df_retirrate, num_cat=1)
dm_retirrate.append(dm_seats_ots, dim='Variables')
dm_retirrate.lag_variable('tra_passenger_seats', shift=1, subfix='_tm1')
dm_retirrate.operation('tra_passenger_seats_tm1', '*', 'tra_retirement-rate', out_col='tra_passenger_vehicle-waste', unit='number')
dm_waste = dm_retirrate.filter({'Variables': ['tra_passenger_vehicle-waste']})
dm_waste.add(np.nan, dim='Years', col_label=years_fts, dummy=True)
dm_waste.rename_col('tra_passenger_vehicle-waste', 'tra_passenger_vehicle-waste_kerosene', dim='Variables')
dm_waste.deepen(based_on='Variables')
dm_waste.add(0, dummy=True, col_label=['BEV', 'H2'], dim='Categories2')


# SECTION: VEHICLE-FLEET-NEW
# s(t) = s(t-1) + n(t) - w(t)
# n(t) = s(t) - s(t-1) + w(t)
dm_retirrate.operation('tra_passenger_seats', '-', 'tra_passenger_seats_tm1', out_col='delta', unit='number')
dm_retirrate.operation('delta', '+', 'tra_passenger_vehicle-waste', out_col='tra_passenger_new-vehicles', unit='number')
dm_retirrate[:, :, 'tra_passenger_new-vehicles', :] = np.maximum(0, dm_retirrate[:, :, 'tra_passenger_new-vehicles', :])
dm_retirrate[:, 2022, 'tra_passenger_new-vehicles', :] = np.nan
dm_retirrate.fill_nans(dim_to_interp='Years')
for t in create_years_list(2020, 2023, 1):
    dm_retirrate[0, t, 'tra_passenger_seats', 0] = dm_retirrate[0, t, 'tra_passenger_new-vehicles', 0] \
                                                   - dm_retirrate[0, t, 'tra_passenger_vehicle-waste', 0]\
                                                   + dm_retirrate[0, t-1, 'tra_passenger_seats', 0]

dm_new = dm_retirrate.filter({'Variables': ['tra_passenger_new-vehicles']})
dm_new.add(np.nan, dim='Years', col_label=years_fts, dummy=True)
dm_new.rename_col('tra_passenger_new-vehicles', 'tra_passenger_new-vehicles_kerosene', dim='Variables')
dm_new.deepen(based_on='Variables')
dm_new.add(0, dummy=True, col_label=['BEV', 'H2'], dim='Categories2')


# SECTION: REDO UTILISATION-RATE
# seat ots seats = pkm_ots (pkm-cap) / (occupancy_ots (pkm/skm) * utilisation-rate skm/seat)
# utilisation-rate skm/seat =  pkm_ots (pkm-cap)*pop / ( occupancy_ots (pkm/skm) * seat (seat) )
dm_pkm_tmp = DM_transport['ots']['passenger_aviation-pkm'].copy()
dm_seats = dm_retirrate.filter({'Variables': ['tra_passenger_seats']})
dm_occupancy_ots = DM_transport['ots']['passenger_occupancy'].filter({'Categories1': ['aviation']})
arr_util_rate = dm_pkm_tmp.array * dm_pop.array[..., np.newaxis]/(dm_occupancy_ots.array * dm_seats.array)
dm_utilirate = DM_ots['ots']['passenger_utilization-rate']
dm_utilirate.array = arr_util_rate
dm_utilirate_orig = DM_transport['ots']['passenger_utilization-rate']
dm_utilrate_new = update_from_subdm(dm_utilirate_orig, dm_utilirate)
DM_transport['ots']['passenger_utilization-rate'] = dm_utilrate_new

# SECTION: PASSENGER TECH
dm_pass_tech_aviation = dm_new.copy()
dm_pass_tech_aviation.append(dm_waste, dim='Variables')
tech_aviation = dm_pass_tech_aviation.col_labels['Categories2']
dm_pass_tech_aviation.append(dm_fleetshare.filter({'Categories2': tech_aviation}), dim='Variables')
dm_pass_tech_aviation.append(dm_fleet_eff.filter({'Categories2': tech_aviation}), dim='Variables')
dm_pass_tech_orig = DM_transport['fxa']['passenger_tech']
dm_pass_tech = update_from_subdm(dm_pass_tech_orig, dm_pass_tech_aviation)
DM_transport['fxa']['passenger_tech'] = dm_pass_tech.copy()


# SECTION: NEW-TECH SHARE FTS
dm_max = DM_transport['fxa']['vehicles-max'].copy()

dm_max = compute_new_veh_max(dm_max)

file_tech_lev4 = 'data/tra_aviation_fleet_lev4.pickle'
with open(file_tech_lev4, 'rb') as handle:
    dm_tech_4 = pickle.load(handle)

dm_new_tech_4, dm_tmp = compute_tech_new_from_newfleet(dm_tech_4, dm_max.copy())

file_tech_lev3 = 'data/tra_aviation_fleet_lev3.pickle'
with open(file_tech_lev3, 'rb') as handle:
    dm_tech_3 = pickle.load(handle)
dm_new_tech_3, dm_max = compute_tech_new_from_newfleet(dm_tech_3, dm_max.copy())

dm_new_tech_orig = DM_transport['fts']['passenger_technology-share_new'][3]
dm_3 = update_from_subdm(dm_new_tech_orig, dm_new_tech_3)
DM_transport['fts']['passenger_technology-share_new'][3] = dm_3

dm_new_tech_orig = DM_transport['fts']['passenger_technology-share_new'][4]
dm_4 = update_from_subdm(dm_new_tech_orig, dm_new_tech_4)
DM_transport['fts']['passenger_technology-share_new'][4] = dm_4

# LEVEL 1 - NO BEV, H2
dm_new_tech_1 = dm_new_tech_3.copy()
dm_new_tech_1[...] = 0
dm_new_tech_1[..., 'kerosene'] = 1
dm_new_tech_orig = DM_transport['fts']['passenger_technology-share_new'][1]
dm_1 = update_from_subdm(dm_new_tech_orig, dm_new_tech_1)
DM_transport['fts']['passenger_technology-share_new'][1] = dm_1

# LEVEL 2 / 3 intermediate
dm_new_tech_2 = dm_new_tech_1.copy()
dm_new_tech_2.array = 1/2*dm_new_tech_1.array + 1/2*dm_new_tech_3.array
dm_new_tech_orig = DM_transport['fts']['passenger_technology-share_new'][2]
dm_2 = update_from_subdm(dm_new_tech_orig, dm_new_tech_2)
DM_transport['fts']['passenger_technology-share_new'][2] = dm_2



# FUEL-MIX
# According to ATAG, Waypoint 2050 (2021) (plot page 47)
# https://aviationbenefits.org/media/167417/w2050_v2021_27sept_full.pdf
# There will be 600-140 = 460 Mtoe of SAF available world-wide.
# 1 toe = 41868 MJ
# Of these, we allocate 0.6% to Switzerland, for a total of
val_SAF_2050_ATAG_MJ = 450*1e6 * 41868 *0.006
# Alternatively, the paper Abrantes et al. (2021) "Sustainable aviation fuels and imminent technologies - CO2 emissions evolution towards 2050"
# https://doi.org/10.1016/j.jclepro.2021.127937
# Gives as max value for SAF in 2050 200 Mt. To convert to energy we use 43 MJ/kg
val_SAF_2050_paper_MJ = 200*1e9 * 43 * 0.006

dm_SAF = dm_utilirate_fts.copy()
dm_SAF.add(np.nan, dim='Variables', col_label='tra_passenger-max-SAF', unit='MJ', dummy=True)
dm_SAF['Switzerland', 2050, 'tra_passenger-max-SAF', 'aviation'] = val_SAF_2050_paper_MJ
dm_SAF['Switzerland', 2025, 'tra_passenger-max-SAF', 'aviation'] = 0
dm_SAF['Switzerland', 2030, 'tra_passenger-max-SAF', 'aviation'] = 0
dm_SAF.fill_nans('Years')

# Extract maximum fuel available
dm_tmp = dm_SAF.filter({'Variables': ['tra_passenger-max-SAF']})
dm_tmp.rename_col('tra_passenger-max-SAF', 'tra_passenger_available-fuel-mix_biofuel', dim='Variables')
dm_tmp.deepen(based_on='Variables')
dm_tmp.add(0, dim='Categories2', col_label='efuel', dummy=True)
dm_tmp.switch_categories_order()
dm_tmp.add(0, dummy=True, dim='Years', col_label=years_ots)
dm_tmp.sort('Years')
DM_transport['fxa']['fuel-mix-availability'] = dm_tmp

dm_new_eff = DM_transport['fts']['passenger_veh-efficiency_new'][4].filter({'Categories1': ['aviation'], 'Categories2': ['kerosene']})
dm_new_eff.group_all('Categories2')
dm_SAF.append(dm_new_eff, dim='Variables')
# seats_max = Energy / (eff * util-rate)
dm_SAF.operation('tra_passenger-max-SAF', '/', 'tra_passenger_veh-efficiency_new', out_col='tra_skm_SAF', unit='skm')
dm_SAF.operation('tra_skm_SAF', '/', 'tra_passenger_utilisation-rate', out_col='tra_vehicles-max_SAF', unit='seat')
dm_max_seat_ICE = dm_max.filter({'Categories2': ['kerosene'], 'Variables': ['tra_vehicles-max'], 'Years': years_fts})
dm_max_seat_ICE.group_all('Categories2')
dm_SAF.append(dm_max_seat_ICE, dim='Variables')
dm_SAF.operation('tra_vehicles-max_SAF', '/', 'tra_vehicles-max', out_col='tra_fuel-mix_biofuel', unit='%')
dm_fuel_mix_SAF = dm_SAF.filter({'Variables': ['tra_fuel-mix_biofuel']})
dm_fuel_mix_SAF.deepen(based_on='Variables')
dm_fuel_mix_SAF.switch_categories_order()
dm_fuel_mix_SAF.add(0, dim='Categories1', col_label='efuel', dummy=True)
dm_fuel_mix_SAF.sort('Categories1')
DM = dict()
for lev in range(4):
    DM[lev+1] = dm_fuel_mix_SAF.copy()
# Create less ambitious scenarios
DM[1].array[...] = 0
DM[2].array[...] = 1/2*DM[1].array[...] + 1/2*DM[3].array[...]
DM[4].array[...] = DM[3].array[...]

for lev in range(4):
    dm_fuel_mix_orig = DM_transport['fts']['fuel-mix'][lev+1]
    dm_fuel_mix_orig.sort('Categories1')
    dm_new = DM[lev+1]
    dm_fuel_mix_orig['Switzerland', :, :, :,  'aviation'] = dm_new['Switzerland', :, :, :, 'aviation']
    for cat in list(set(dm_fuel_mix_orig.col_labels['Categories2']) - {'aviation'}):
        dm_fuel_mix_orig['Switzerland', :, :, :, cat] = 0
    DM_transport['fts']['fuel-mix'][lev + 1] = dm_fuel_mix_orig.copy()

# Set other biofuels to 0

del dm_new_eff, dm_max_seat_ICE, dm_SAF, dm_new, dm_fuel_mix_orig, dm_fuel_mix_SAF



# seats = energy MJ / (efficiency MJ/skm * utilisation-rate skm/seat)
# efficiency_SAF =
# utilisation-rate 2050 ~ 1.65 M skm/seats
#seats_SAF_2050 = val_SAF_2050/43

# Create dummy values for other countries
data_file = '../../../data/datamatrix/transport.pickle'
with open(data_file, 'rb') as handle:
    DM_transport_orig = pickle.load(handle)

all_countries = set(DM_transport_orig['ots']['passenger_aviation-pkm'].col_labels['Country']) - {'Switzerland'}
for new_country in all_countries:
    add_dummy_country_to_DM(DM_transport, new_country, 'Switzerland')

# Add aviation to DM_transport_orig
add_missing_cat_DM(DM_transport_orig, DM_transport)

# Overwrite passenger_aviation-pkm
DM_transport_orig['ots']['passenger_aviation-pkm']['Switzerland', ...] = DM_transport['ots']['passenger_aviation-pkm']['Switzerland', ...]
DM_transport_orig['ots']['passenger_aviation-pkm']['Vaud', ...] = DM_transport['ots']['passenger_aviation-pkm']['Switzerland', ...]

for lev in range(4):
    lev = lev + 1
    dm_CH = DM_transport['fts']['passenger_aviation-pkm'][lev].copy()
    dm_orig = DM_transport_orig['fts']['passenger_aviation-pkm'][lev].copy()
    dm_orig['Switzerland', :, :, :] = dm_CH['Switzerland', :, :, :]
    dm_orig['Vaud', :, :, :] = dm_CH['Switzerland', :, :, :]
    DM_transport_orig['fts']['passenger_aviation-pkm'][lev] = dm_orig

# Add new fxa
DM_transport_orig['fxa']['vehicles-max'] = DM_transport['fxa']['vehicles-max']
DM_transport_orig['fxa']['share-local-emissions'] = DM_transport['fxa']['share-local-emissions']
DM_transport_orig['fxa']['fuel-mix-availability'] = DM_transport['fxa']['fuel-mix-availability']

for lev in range(4):
    lev = lev + 1
    dm_CH = DM_transport['fts']['fuel-mix'][lev].copy()
    dm_orig = DM_transport_orig['fts']['fuel-mix'][lev].copy()
    dm_orig['Switzerland', ...] = dm_CH['Switzerland', ...]
    dm_orig['Vaud', ...] = dm_CH['Switzerland', ...]
    DM_transport_orig['fts']['fuel-mix'][lev] = dm_orig

DM_transport_orig['constant'] = DM_transport['constant']

f = '../../../data/datamatrix/transport.pickle'
#with open(f, 'wb') as handle:
#    pickle.dump(DM_transport_orig, handle, protocol=pickle.HIGHEST_PROTOCOL)
my_pickle_dump(DM_transport_orig, f)
sort_pickle('../../../data/datamatrix/transport.pickle')



print('Hello')
