from model.common.auxiliary_functions import create_years_list, linear_fitting
from model.common.data_matrix_class import DataMatrix
import pandas as pd
import numpy as np
import os
from _database.pre_processing.transport.Switzerland.get_data_functions.aviation_part1 import get_pkm_cap_aviation, get_world_pop, get_aviation_fleet
from _database.pre_processing.transport.Switzerland.get_data_functions import utils


def compute_pkm_cap_aviation(dm_pkm_cap_aviation_CH_raw, dm_pkm_aviation_WLD,
                             dm_pop_WLD, years_ots):
  dm_pkm_cap_aviation_CH = dm_pkm_cap_aviation_CH_raw.copy()
  dm_pkm_aviation_WLD.filter({'Years': years_ots}, inplace=True)
  dm_pkm_aviation_WLD.change_unit('tra_passenger_transport-demand', factor=1e9,
                                  old_unit='Bpkm', new_unit='pkm')
  dm_pkm_aviation_WLD = dm_pkm_aviation_WLD.flatten()

  # pkm_cap_WLD = pkm/pop
  years_WLD = dm_pkm_aviation_WLD.col_labels['Years']
  dm_pop_WLD.filter({'Years': years_WLD}, inplace=True)
  dm_pkm_aviation_WLD.append(dm_pop_WLD, dim='Variables')
  dm_pkm_aviation_WLD.operation('tra_passenger_transport-demand_aviation', '/',
                                'tra_passenger_lfs_population_total',
                                out_col='tra_pkm-cap_aviation', unit='pkm/cap')
  dm_pkm_cap_WLD = dm_pkm_aviation_WLD.filter(
    {'Variables': ['tra_pkm-cap_aviation']})
  dm_pkm_cap_WLD.deepen()

  # Make years compatible
  years_CH = dm_pkm_cap_aviation_CH.col_labels['Years']
  years_to_add = [y for y in years_WLD if y not in years_CH]
  dm_pkm_cap_aviation_CH.add(np.nan, dim='Years', col_label=years_to_add,
                             dummy=True)
  dm_pkm_cap_aviation_CH.sort('Years')
  dm_pkm_cap_aviation_CH.filter({'Years': years_WLD}, inplace=True)

  # Replace 0s with np.nans in CH data
  mask = dm_pkm_cap_aviation_CH.array == 0
  dm_pkm_cap_aviation_CH.array[mask] = np.nan

  dm_pkm_cap_WLD.rename_col('World', 'Switzerland', dim='Country')
  dm_pkm_cap_WLD.rename_col('tra_pkm-cap', 'reference_curve', dim='Variables')

  # ratio = pkm_cap_CH / pkm_cap_WLD, for available years (e
  dm_2021 = dm_pkm_cap_aviation_CH.filter({'Years': [2021]})
  idx = dm_pkm_cap_aviation_CH.idx
  dm_pkm_cap_aviation_CH.array[:, idx[2021], ...] = np.nan
  dm_pkm_cap_aviation_CH.append(dm_pkm_cap_WLD, dim='Variables')
  dm_pkm_cap_aviation_CH = utils.fill_var_nans_based_on_var_curve(
    dm_pkm_cap_aviation_CH, 'tra_pkm-cap', 'reference_curve')

  idx = dm_pkm_cap_aviation_CH.idx
  dm_pkm_cap_aviation_CH.array[:, idx[2021], ...] = dm_2021.array

  # Fix 2020 value
  ## In reality flights in 2020 were lower than in 2021. We take pkm_cap_2020/pkm_cap_2021 for world and apply it to CH
  idx = dm_pkm_cap_WLD.idx
  ratio_2020_2019 = dm_pkm_cap_WLD.array[
                      0, idx[2020], idx['reference_curve'], idx['aviation']] / \
                    dm_pkm_cap_WLD.array[
                      0, idx[2021], idx['reference_curve'], idx['aviation']]
  idx = dm_pkm_cap_aviation_CH.idx
  dm_pkm_cap_aviation_CH.array[:, idx[2020],
  ...] = dm_pkm_cap_aviation_CH.array[:, idx[2021], ...] * ratio_2020_2019

  dm_pkm_cap_aviation_VD = dm_pkm_cap_aviation_CH.copy()
  dm_pkm_cap_aviation_VD.rename_col('Switzerland', 'Vaud', dim='Country')
  dm_pkm_cap_aviation_CH.append(dm_pkm_cap_aviation_VD, dim='Country')

  linear_fitting(dm_pkm_cap_aviation_CH, years_ots=[2020, 2021, 2022, 2023],
                 based_on=[2020, 2021])

  return dm_pkm_cap_aviation_CH

def pkm_monde_ots(dm_pkm_orig, value_2024):
  dm_pkm = dm_pkm_orig
  dm_pkm.add(value_2024, dim='Years', col_label=[2024], dummy=True)
  dm_pkm[0, 2023, 0, 0] = np.nan
  dm_pkm[0, 2022, 0, 0] = np.nan
  dm_pkm.fill_nans(dim_to_interp='Years')
  dm_pkm.drop(col_label=[2024], dim='Years')
  # Impose 2020 value from US 2019/2020 decrease rate
  dm_pkm[0, 2020, 0, 0] = 3103.0

  return dm_pkm


def adj_pkm_monde_ots(dm_pkmsuisse_ots, dm_pkm_ots):
  # dm_pkm_ots: represents the pkm/cap demand of swiss resident population territorial and not
  # dm_pkmsuisse_ots: represents the pkm demand of swiss residents within switzerland?
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


def run(years_ots):

  this_dir = os.path.dirname(os.path.abspath(__file__))

  ##### Transport demand aviation - of Swiss residents (> 6 years)
  # Civil Aviation
  # ! Data available only every 5 years
  file_url = 'https://dam-api.bfs.admin.ch/hub/api/dam/assets/32013522/master'
  local_filename = os.path.join(this_dir, '../data/tra_aviation_CH.xlsx')
  dm_pkm_cap_aviation_CH_raw = get_pkm_cap_aviation(file_url, local_filename)

  ##### Transport pkm aviation (World)
  # Download data from "Our World in Data"
  # https://ourworldindata.org/grapher/aviation-demand-efficiency
  local_filename = os.path.join(this_dir, '../data/tra_global_aviation-demand.csv')
  df = pd.read_csv(local_filename)
  df = df[['Entity', 'Year', 'Passenger demand']]
  df.columns = ['Country', 'Years',
                'tra_passenger_transport-demand_aviation[Bpkm]']
  dm_pkm_aviation_WLD = DataMatrix.create_from_df(df, num_cat=1)

  ##### Extract world population from World Bank
  wb_pop_url = 'https://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=excel'
  local_filename = os.path.join(this_dir, '../data/lfs_world_population_WB.xls')
  dm_pop_WLD = get_world_pop(wb_pop_url, local_filename)

  ##### Complete historical aviation pkm CH using World trend
  dm_pkm_cap_aviation = compute_pkm_cap_aviation(dm_pkm_cap_aviation_CH_raw,
                                                 dm_pkm_aviation_WLD,
                                                 dm_pop_WLD, years_ots)

  integrate_full_aviation = False

  if integrate_full_aviation:
    # IATA, Global Air Passenger Demand Reaches Record High in 2024
    # https://www.iata.org/en/pressroom/2025-releases/2025-01-30-01/
    value_2024 = dm_pkm_cap_aviation[0, 2019, 0, 0] * 1.038
    # Adjust dm_pkm_orig for post-covid (2020 value from US 2019/2020 decrease rate)
    dm_pkm_cap = pkm_monde_ots(dm_pkm_cap_aviation, value_2024)

    # section DONNES PKM-SUISSE Territorial
    # file = 'data/pkm_suisse.xlsx'
    # Pkm suisse are linked to the Sold Fuel Principle (aviation fuel sold on swiss soil)
    # It is assumed that 50% of the travelers are swiss residents, we would like

    file = 'data/aviation_pkm_suisse.csv'
    df = pd.read_csv(file, sep=';', decimal=',')
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    dm_pkmsuisse_ots = DataMatrix.create_from_df(df, num_cat=1)
    dm_pkmsuisse_ots.rename_col('tra_pkm-cap', 'tra_pkm-suisse-cap',
                                dim='Variables')

    dm_pkm_cap = adj_pkm_monde_ots(dm_pkmsuisse_ots, dm_pkm_cap)


  ##### Vehicle fleet aviation - Switzerland only
  # Civil Aviation
  # ! Data available only every 5 years
  #file_url = 'https://dam-api.bfs.admin.ch/hub/api/dam/assets/32013522/master'
  #local_filename = os.path.join(this_dir, '../data/tra_aviation_CH.xlsx')
  #dm_pkm_fleet_aviation = get_aviation_fleet(file_url, local_filename)

  return dm_pkm_cap_aviation


if __name__ == "__main__":

  years_ots = create_years_list(1990, 2023, 1)

  print('Running preliminary aviation pipeline')
  dm_demand_cap = run(years_ots)
