import numpy as np
import os
import copy
from model.common.auxiliary_functions import my_pickle_dump, filter_DM, \
  create_years_list, linear_fitting, dm_add_missing_variables, sort_pickle
from model.common.data_matrix_class import DataMatrix
import pickle
from _database.pre_processing.transport.Switzerland.get_data_functions import utils



def forecast_vkm_cap(dm_km, years_fts):
  based_on_years = create_years_list(2010, 2019, 1)
  linear_fitting(dm_km, years_fts, based_on=based_on_years, min_tb=0)
  # For metrotram extrapolate with flat line
  idx = dm_km.idx
  dm_km.array[idx['Switzerland'], idx[2025]:idx[2050] + 1, idx['tra_vkm-cap'], idx['metrotram']] \
    = dm_km['Switzerland', 2025, 'tra_vkm-cap', 'metrotram']
  # Make sure 2W vkm <= pkm (occupancy > 1)
  mask = dm_km[:, :, 'tra_vkm-cap', '2W'] > dm_km[:, :, 'tra_pkm-cap', '2W']
  dm_km[:, :, 'tra_vkm-cap', '2W'][mask] = dm_km[:, :, 'tra_pkm-cap', '2W'][mask]
  return dm_km


def run(DM_transport_new, country_list, years_ots, years_fts):


  # SECTION Modal-share and Transport demand pkm fts
  # pkm/cap * modal-share[%]
  dm_pkm_cap_tot = DM_transport_new['ots']['pkm'].copy()
  dm_share = DM_transport_new['ots']['passenger_modal-share'].copy()
  arr_pkm_cap = dm_pkm_cap_tot[..., np.newaxis] * dm_share[...]
  dm_pkm_cap = DataMatrix.based_on(arr_pkm_cap, format = dm_share,
                                   change={'Variables': ['tra_pkm-cap']},
                                   units={'tra_pkm-cap': 'pkm/cap'})
  based_on_years = create_years_list(2010, 2019, 1)
  linear_fitting(dm_pkm_cap, years_fts, based_on=based_on_years)

  # For Switzerland metrotram use flat extrapolation
  idx = dm_pkm_cap.idx
  dm_pkm_cap[idx['Switzerland'], idx[2025]:idx[2050] + 1,
  idx['tra_pkm-cap'], idx['metrotram']] = \
    dm_pkm_cap.array[idx['Switzerland'], idx[2025], idx['tra_pkm-cap'], idx['metrotram']]
  dm_modal_share = dm_pkm_cap.normalise(dim='Categories1', inplace=False)
  dm_modal_share.rename_col('tra_pkm-cap_share', 'tra_passenger_modal-share',
                            dim='Variables')

  DM_transport_new['fts']['passenger_modal-share'] = dict()
  for lev in range(4):
    DM_transport_new['fts']['passenger_modal-share'][lev+1] = (
      dm_modal_share.filter({'Years': years_fts}))


  # SECTION Pkm
  dm_pkm_cap_tot = dm_pkm_cap.group_all('Categories1', inplace=False)
  DM_transport_new['fts']['pkm'] = dict()
  for lev in range(4):
    DM_transport_new['fts']['pkm'][lev+1] = dm_pkm_cap_tot.filter({'Years': years_fts})


  # SECTION Technology share new fleet fts
  # For tech share we don't need the forecasting because it is computed from new_fleet
  dm_fleet_new_tech_share = DM_transport_new['ots']['passenger_technology-share_new'].copy()
  dm_fleet_new_tech_share.add(np.nan, dim='Years', col_label=years_fts, dummy=True)
  dm_fleet_new_tech_share.fill_nans('Years')

  DM_transport_new['fts']['passenger_technology-share_new'] = dict()
  for lev in range(4):
    DM_transport_new['fts']['passenger_technology-share_new'][lev+1] = (
      dm_fleet_new_tech_share.filter({'Years': years_fts}))


  # SECTION Occupancy pkm/vkm  fts
  dm_occupancy = DM_transport_new['ots']['passenger_occupancy'].copy()
  dm_add_missing_variables(dm_occupancy, {'Years': years_fts})
  dm_km =  dm_occupancy
  dm_km.append(dm_pkm_cap.filter({'Categories1': dm_occupancy.col_labels['Categories1']}), dim='Variables')
  # compute vkm/cap
  dm_km.operation('tra_pkm-cap', '/', 'tra_passenger_occupancy',
                  out_col='tra_vkm-cap', unit='vkm/cap')
  dm_km.drop(dim='Variables', col_label='tra_passenger_occupancy')
  # Create vkm-cap forecasting
  dm_km = forecast_vkm_cap(dm_km, years_fts)
  dm_km.operation('tra_pkm-cap', '/', 'tra_vkm-cap',
                  out_col='tra_passenger_occupancy', unit='pkm/vkm')
  dm_occupancy = dm_km.filter({'Variables': ['tra_passenger_occupancy']})

  DM_transport_new['fts']['passenger_occupancy'] = dict()
  for lev in range(4):
    DM_transport_new['fts']['passenger_occupancy'][lev+1] = (
      dm_occupancy.filter({'Years': years_fts}))


  # SECTION Utilisation rate vkm/veh fts
  dm_utilisation = DM_transport_new['ots']['passenger_utilization-rate'].copy()
  linear_fitting(dm_utilisation, years_fts, based_on=create_years_list(2010, 2019, 1))

  DM_transport_new['fts']['passenger_utilization-rate'] = dict()
  for lev in range(4):
    DM_transport_new['fts']['passenger_utilization-rate'][lev+1] = dm_utilisation.filter({'Years': years_fts})


  # SECTION Efficiency fleet
  # For veh-fleet efficiency we can leave the fts to nan because this get re-computed
  dm_veh_new_eff = DM_transport_new['ots']['passenger_veh-efficiency_new'].copy()
  dm_veh_new_eff.add(np.nan, dim='Years', dummy=True, col_label=years_fts)
  dm_veh_new_eff.fill_nans(dim_to_interp='Years')
  dm_veh_new_eff.filter({'Years': years_fts}, inplace=True)

  DM_transport_new['fts']['passenger_veh-efficiency_new'] = dict()
  for lev in range(4):
    DM_transport_new['fts']['passenger_veh-efficiency_new'][lev+1] = dm_veh_new_eff.filter({'Years': years_fts})

  DM_transport_wo_aviation = copy.deepcopy(DM_transport_new)

  # Load existing DM_transport
  this_dir = os.path.dirname(os.path.abspath(__file__))
  pickle_file = os.path.join(this_dir, '../../../../data/datamatrix/transport.pickle')
  with open(pickle_file, 'rb') as handle:
    DM_transport = pickle.load(handle)

  DM_transport_new = {'fts': DM_transport_new['fts']}
  # ! FIXME: you should re-implement here aviation data
  # DM_transport_new is missing "aviation" so I cannot simply run my_pickle_dump()
  # I need to copy DM_transport aviation
  utils.add_aviation_data_to_DM(DM_transport_new, DM_transport)

  my_pickle_dump(DM_new=DM_transport_new, local_pickle_file=pickle_file)
  sort_pickle(pickle_file)

  return DM_transport_wo_aviation


