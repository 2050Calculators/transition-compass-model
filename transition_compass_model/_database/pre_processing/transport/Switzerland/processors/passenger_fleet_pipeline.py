################################
#####   VEHICLE FLEET  #########
################################

import os
import numpy as np
from _database.pre_processing.transport.Switzerland.get_data_functions import passenger_fleet as get_data
from model.common.auxiliary_functions import create_years_list, load_pop
from _database.pre_processing.transport.Switzerland.processors.transport_demand_pipeline import run as demand_pkm_vkm_run


def downscale_public_fleet_VD(dm_public_fleet, dm_pkm):
  dm_public_fleet.sort('Categories1')
  dm_public_pkm = dm_pkm.filter(
    {'Categories1': dm_public_fleet.col_labels['Categories1']})
  idx = dm_public_pkm.idx
  arr_ratio_pkm = dm_public_pkm.array[idx['Vaud'], :, :,
                  :] / dm_public_pkm.array[idx['Switzerland'], :, :, :]
  idx = dm_public_fleet.idx
  arr_VD = dm_public_fleet.array[idx['Switzerland'], :, :, :, :] * \
           arr_ratio_pkm[..., np.newaxis]
  dm_public_fleet.add(arr_VD, dim='Country', col_label='Vaud')
  return dm_public_fleet


def compute_passenger_new_fleet(table_id_new_veh, file_new_veh_ots1,
                                file_new_veh_ots2):
  ### Add new fleet Vaud 1990 - 2004
  def compute_new_fleet_vaud(dm_CH, dm_tech):
    # Extract the cantonal % of the swiss new vehicles in 2005 and uses it to determine Vaud fleet in 1990-2004
    dm_tmp = dm_tech.group_all(dim='Categories2', inplace=False)
    idx = dm_tmp.idx
    arr_shares = dm_tmp.array[idx['Vaud'], 0, :, :] / dm_tmp.array[
                                                      idx['Switzerland'], 0, :,
                                                      :]
    idx_ch = dm_CH.idx
    arr_VD = dm_CH.array[idx_ch['Switzerland'], :, :, :] * arr_shares[
                                                           np.newaxis, :, :]
    dm = dm_CH.copy()
    dm.add(arr_VD, dim='Country', col_label='Vaud')
    return dm

  ### New Passenger fleet for Switzerland and Vaud from 1990-2023
  def compute_new_fleet_tech_all_ots(dm_new_fleet_tech_ots1,
                                     dm_pass_new_fleet_ots2, first_year):
    # Applied 2005 share by technology to 1990-2005 period
    dm_new_fleet_tech_ots1.normalise(dim='Categories2', inplace=True,
                                     keep_original=True)

    if dm_new_fleet_tech_ots1.col_labels['Categories1'] != \
      dm_pass_new_fleet_ots2.col_labels['Categories1']:
      raise ValueError('Make sure categories match')
    if dm_new_fleet_tech_ots1.col_labels['Country'] != \
      dm_pass_new_fleet_ots2.col_labels['Country']:
      raise ValueError('Make sure Country match')
    # Multiply historical data on new fleet by 2005 technology share to obtain fleet by techology
    idx_n = dm_pass_new_fleet_ots2.idx
    idx_s = dm_new_fleet_tech_ots1.idx
    arr = dm_pass_new_fleet_ots2.array[:, :,
          idx_n['tra_passenger_new-vehicles'], :, np.newaxis] \
          * dm_new_fleet_tech_ots1.array[:, idx_s[first_year], np.newaxis,
            idx_s['tra_passenger_new-vehicles_share'], :, :]
    arr = arr[:, :, np.newaxis, :, :]

    dm_new_fleet_tech_ots1.drop(dim='Variables',
                                col_label='tra_passenger_new-vehicles_share')
    dm_new_fleet_tech = dm_new_fleet_tech_ots1.copy()
    dm_new_fleet_tech.add(arr, dim='Years',
                          col_label=dm_pass_new_fleet_ots2.col_labels['Years'])
    dm_new_fleet_tech.sort('Years')

    return dm_new_fleet_tech


  # New fleet Switzerland + Vaud: 2005 - now (by technology)
  dm_new_fleet_tech_ots1 = get_data.get_new_fleet_by_tech_raw(table_id_new_veh, file_new_veh_ots1)
  # Passenger new fleet Switzerland + Vaud: 2005 - new (by technology)
  dm_pass_new_fleet_tech_ots1, dm_new_tech = get_data.extract_passenger_new_fleet_by_tech(dm_new_fleet_tech_ots1)
  first_year = dm_pass_new_fleet_tech_ots1.col_labels['Years'][0]
  # New fleet Switzerland 1990 - 2004
  dm_pass_new_fleet_CH_ots2 = get_data.get_new_fleet(file_new_veh_ots2, first_year)
  # Add new fleet Vaud 1990 - 2004
  dm_pass_new_fleet_ots2 = compute_new_fleet_vaud(dm_pass_new_fleet_CH_ots2, dm_pass_new_fleet_tech_ots1)
  # Compute technology shares 1990 - 2004
  dm_new_fleet_tech = compute_new_fleet_tech_all_ots(dm_pass_new_fleet_tech_ots1, dm_pass_new_fleet_ots2, first_year)
  return dm_new_fleet_tech, dm_new_tech

def allocate_other_to_new_technologies(dm_fleet, dm_new_tech):

  dm_fleet_other = dm_fleet.filter({'Categories2': ['Other']})
  # dm_fleet_other.group_all('Categories2')
  dm_fleet_other.filter({'Years': dm_new_tech.col_labels['Years']},
                        inplace=True)
  dm_fleet.drop(dim='Categories2', col_label='Other')

  # Assuming none of the vehicles from 2005 has gone to waste (simplification),
  # the fleet at year Y will be the sum of new_fleet for years <= Y
  # The results are then normalised and the shares are used to allocate other
  dm_new_tech_cumul = dm_new_tech.copy()
  dm_new_tech_cumul.array = np.cumsum(dm_new_tech.array, axis=1)
  dm_new_tech_cumul.normalise(dim='Categories2', inplace=True,
                              keep_original=False)
  # The normalisation returns nan if all values are 0. Then replace with 0
  np.nan_to_num(dm_new_tech_cumul.array, nan=0.0, copy=False)

  # Allocate
  idx = dm_fleet_other.idx
  arr = dm_fleet_other.array[:, :, :, :, idx['Other'],
        np.newaxis] * dm_new_tech_cumul.array
  dm_fleet_other.add(arr, dim='Categories2',
                     col_label=dm_new_tech_cumul.col_labels['Categories2'])
  dm_fleet_other.drop(dim='Categories2', col_label='Other')

  # Map fuel technology to transport module category
  dict_tech = {'FCEV': ['Hydrogen'],
               'ICE-diesel': ['Diesel-electricity: conventional hybrid'],
               'ICE-gasoline': ['Petrol-electricity: conventional hybrid'],
               'PHEV-diesel': ['Diesel-electricity: plug-in hybrid'],
               'PHEV-gasoline': ['Petrol-electricity: plug-in hybrid'],
               'ICE-gas': ['Gas (monovalent and bivalent)']}
  dm_fleet_other.groupby(dict_tech, dim='Categories2', regex=False,
                         inplace=True)

  dm_fleet_new = dm_fleet.filter(
    {'Years': dm_fleet_other.col_labels['Years']}, inplace=False)
  dm_fleet.drop(dim='Years', col_label=dm_fleet_other.col_labels['Years'])

  idx_f = dm_fleet.idx
  idx_o = dm_fleet_other.idx
  # Diesel
  dm_fleet_new.array[:, :, :, :, idx_f['ICE-diesel']] = \
    dm_fleet_new.array[:, :, :, :, idx_f['ICE-diesel']] \
    + dm_fleet_other.array[:, :, :, :, idx_o['ICE-diesel']]
  # Petrol
  dm_fleet_new.array[:, :, :, :, idx_f['ICE-gasoline']] = \
    dm_fleet_new.array[:, :, :, :, idx_f['ICE-gasoline']] \
    + dm_fleet_other.array[:, :, :, :, idx_o['ICE-gasoline']]
  dm_fleet_other.drop(dim='Categories2',
                      col_label=['ICE-gasoline', 'ICE-diesel'])

  dm_fleet_new.append(dm_fleet_other, dim='Categories2')

  dm_fleet.add(0.0, dummy=True, dim='Categories2',
               col_label=dm_fleet_other.col_labels['Categories2'])
  dm_fleet.append(dm_fleet_new, dim='Years')

  return dm_fleet

def get_public_transport_data(file_url, local_filename, years_ots):
  DF_dict = get_data.get_excel_file_sheets(file_url, local_filename)
  dm_public_fleet = get_data.extract_public_passenger_fleet(DF_dict['Passenger fleet'],
                                                   years_ots)
  dm_public_demand_pkm = get_data.extract_public_passenger_pkm(
    DF_dict['Passenger pkm'], years_ots)
  dm_public_demand_vkm = get_data.extract_public_passenger_vkm(
    DF_dict['Passenger vkm'], years_ots)

  DM_public = {'public_fleet': dm_public_fleet,
               'public_demand-pkm': dm_public_demand_pkm,
               'public_demand-vkm': dm_public_demand_vkm}

  return DM_public


def run(dm_pkm, years_ots):
  this_dir = os.path.dirname(os.path.abspath(__file__))

  # SECTION New vehicle fleet and technology share LDV, 2W ots
  ##### NEW passenger fleet by technology LDV, 2W
  table_id_new_veh = 'px-x-1103020200_120'
  # file is created if it doesn't exist
  file_new_veh_ots1 = os.path.join(this_dir, '../data/tra_new_fleet.pickle')
  # download this from https://www.bfs.admin.ch/asset/en/30305446, download csv file FSO number gr-e-11.03.02.02.01a
  file_new_veh_ots2 = os.path.join(this_dir, '../data/tra_new-vehicles_CH_1990-2023.csv')
  # dm_new_tech is the number of new vehicles for new technologies (used to allocate "Other" category in dm_pass_fleet
  dm_pass_new_fleet, dm_new_tech = compute_passenger_new_fleet(table_id_new_veh,
                                                               file_new_veh_ots1,
                                                               file_new_veh_ots2)

  # SECTION Vehicle fleet and technology share LDV, 2W ots
  #### Passenger fleet by technology (stock) LDV, 2W
  table_id_tot_veh = 'px-x-1103020100_101'
  file_tot_veh = os.path.join(this_dir, '../data/tra_tot_fleet.pickle')
  dm_pass_fleet_raw = get_data.get_passenger_stock_fleet_by_tech_raw(table_id_tot_veh,
                                                            file_tot_veh)
  # Allocate "Other" category to new technologies
  dm_pass_fleet = allocate_other_to_new_technologies(dm_pass_fleet_raw,
                                                     dm_new_tech)

  # SECTION Vehicle fleet bus, rail, metrotram ots
  #### Passenger fleet by technology (stock) bus, rail, metrotram - Switzerland only
  # Note that this data are better for ots than
  file_url = 'https://dam-api.bfs.admin.ch/hub/api/dam/assets/32253175/master'
  # Transports publics (trafic marchandises rail inclus) - séries chronologiques détaillées
  local_filename = os.path.join(this_dir, '../data/tra_public_transport.xlsx')
  DM_public = get_public_transport_data(file_url, local_filename, years_ots)
  dm_public_fleet = DM_public['public_fleet'].copy()

  #### Passenger fleet by technology (stock) bus, rail, metrotram - Downscale to Vaud
  dm_public_fleet = downscale_public_fleet_VD(dm_public_fleet, dm_pkm)

  dm_private_fleet = dm_pass_fleet.filter({'Years': years_ots})
  dm_private_fleet.append(dm_pass_new_fleet.filter({'Years': years_ots}), dim='Variables')

  return dm_private_fleet, dm_public_fleet


if __name__ == "__main__":

  country_list = ['Switzerland', 'Vaud']
  years_ots = create_years_list(1990, 2023, 1)

  dm_pop_ots = load_pop(country_list, years_list=years_ots)

  dm_pkm_cap, dm_pkm, dm_vkm = demand_pkm_vkm_run(dm_pop_ots, years_ots)

  run(dm_pkm, years_ots)
