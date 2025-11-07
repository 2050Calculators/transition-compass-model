from _database.pre_processing.transport.Switzerland.processors.passenger_efficiency_pipeline import compute_tech_share
from model.common.auxiliary_functions import dm_add_missing_variables, my_pickle_dump, sort_pickle
import os
import pickle
import numpy as np

def run(DM_input, years_ots, years_fts):

  DM_transport_new = {'ots': dict(), 'fts': dict(), 'fxa': dict(),
                  'constant': dict()}

  ######################################
  ####    TECHNOLOGY SHARE FLEET   #####
  ######################################
  dm_private_fleet = DM_input['passenger_private-fleet'].copy()
  dm_public_fleet = DM_input['passenger_public-fleet'].copy()
  dm_fleet_tech_share = compute_tech_share(dm_private_fleet, dm_public_fleet)

  ##############################
  ####     WASTE FLEET     #####
  ##############################
  dm_waste_fleet = DM_input['passenger_waste-fleet'].copy()
  dm_fleet_tech_share.append(dm_waste_fleet, dim='Variables')

  ############################
  ####     NEW FLEET     #####
  ############################
  dm_new_fleet = DM_input['passenger_new-vehicles'].copy()
  dm_fleet_tech_share.append(dm_new_fleet, dim='Variables')

  ###################################
  ####     EFFICIENCY FLEET     #####
  ###################################
  dm_veh_eff = DM_input['efficiency'].filter({'Variables': ['tra_passenger_veh-efficiency_fleet']})
  dm_fleet_tech_share.append(dm_veh_eff, dim='Variables')

  # fts values are left to nan because they get recomputed
  dm_fleet_tech_share.add(np.nan, dim='Years', col_label=years_fts, dummy=True)
  DM_transport_new['fxa']['passenger_tech'] = dm_fleet_tech_share

  #################################
  ####     EFFICIENCY NEW     #####
  #################################
  dm_veh_new_eff = DM_input['efficiency'].filter({'Variables': ['tra_passenger_veh-efficiency_new']})
  DM_transport_new['ots']['passenger_veh-efficiency_new'] = dm_veh_new_eff

  ################################
  ####    FLEET LIFETIME     #####
  ################################
  dm_lifetime = DM_input['lifetime'].copy()
  DM_transport_new['fxa']['passenger_vehicle-lifetime'] = dm_lifetime

  ####################################
  ####    ELECTRICITY EMISSION   #####
  ####################################
  dm_elec = DM_input['emissions_electricity'].copy()
  DM_transport_new['fxa']['emission-factor-electricity'] = dm_elec

  ##############################
  ####    DEMAND PKM/CAP   #####
  ##############################
  dm_pkm_cap = DM_input['pkm_cap'].copy()
  dm_pkm_cap_tot = dm_pkm_cap.group_all(dim='Categories1', inplace=False)
  DM_transport_new['ots']['pkm'] = dm_pkm_cap_tot


  #######################################
  ####    DEMAND PKM/CAP - AVIATION #####
  #######################################
  dm_pkm_cap_aviation = DM_input['pkm_cap_aviation']
  DM_transport_new['ots']['passenger_aviation-pkm'] = dm_pkm_cap_aviation

  ###########################
  ####    MODAL SHARE   #####
  ###########################
  dm_modal_share = dm_pkm_cap.normalise(dim='Categories1', inplace=False)
  DM_transport_new['ots']['passenger_modal-share'] = dm_modal_share

  #############################################
  ####     TECHNOLOGY SHARE NEW FLEET     #####
  #############################################
  dm_fleet_new_tech_share = dm_new_fleet.normalise(dim='Categories2',
                                                   inplace=False)
  dm_fleet_new_tech_share.rename_col('tra_passenger_new-vehicles_share',
                                     'tra_passenger_technology-share_new',
                                     dim='Variables')
  DM_transport_new['ots']['passenger_technology-share_new'] = dm_fleet_new_tech_share


  ############################
  ####     OCCUPANCY     #####
  ############################
  dm_pkm = DM_input['pkm_demand'].copy()
  dm_vkm = DM_input['vkm_demand'].copy()
  dm_km = dm_pkm.filter({'Categories1': dm_vkm.col_labels['Categories1']})
  dm_km.append(dm_vkm, dim='Variables')
  dm_km.operation('tra_passenger_transport-demand', '/', 'tra_passenger_transport-demand-vkm',
                  out_col='tra_passenger_occupancy', unit='pkm/vkm')
  dm_occupancy = dm_km.filter({'Variables': ['tra_passenger_occupancy']})
  DM_transport_new['ots']['passenger_occupancy'] = dm_occupancy

  ###################################
  ####     UTILISATION RATE     #####
  ###################################
  cat_tech = set(dm_private_fleet.col_labels['Categories2']).union(set(dm_public_fleet.col_labels['Categories2']))
  # Join private and public
  dm_add_missing_variables(dm_public_fleet, {'Categories2': cat_tech})
  dm_add_missing_variables(dm_private_fleet, {'Categories2': cat_tech})
  dm_private_fleet.append(dm_public_fleet, dim='Categories1')
  dm_fleet = dm_private_fleet.group_all('Categories2', inplace=False)
  dm_fleet.append(dm_vkm, dim='Variables')
  dm_fleet.operation('tra_passenger_transport-demand-vkm', '/', 'tra_passenger_vehicle-fleet',
                  out_col='tra_passenger_utilisation-rate', unit='vkm/veh')
  dm_utilisation = dm_fleet.filter({'Variables': ['tra_passenger_utilisation-rate']})
  DM_transport_new['ots']['passenger_utilization-rate'] = dm_utilisation.filter({'Years': years_ots})

  ###################################
  ####     EMISSION FACTORS     #####
  ###################################
  cdm_emissions_factors = DM_input['emission_factors']
  DM_transport_new['constant'] = cdm_emissions_factors

  # Load existing DM_transport
  #this_dir = os.path.dirname(os.path.abspath(__file__))
  #pickle_file = os.path.join(this_dir, '../../../../data/datamatrix/transport.pickle')
  #with open(pickle_file, 'rb') as handle:
  #  DM_transport = pickle.load(handle)

  #my_pickle_dump(DM_new=DM_transport_new, local_pickle_file=pickle_file)
  #sort_pickle(pickle_file)
  return DM_transport_new
