# ======================  IMPORT PACKAGES & DATA  ========================================================
import copy
import pickle
import numpy as np
import pandas as pd
from model.common.auxiliary_functions import linear_fitting, my_pickle_dump, \
  sort_pickle, dm_add_missing_variables
import os
from _database.pre_processing.transport.Switzerland.get_data_functions import utils

def run(DM_transport):

  DM_fts = {'fts': dict()}

  # ======================  MODAL_SHARE  ========================================================
  dm_modal_share_2 = DM_transport['fts']['passenger_modal-share'][2].filter({'Country': ['Vaud']})
  dm_modal_share_4 = DM_transport['fts']['passenger_modal-share'][4].filter({'Country': ['Vaud']})
  dm_modal_share_ots = DM_transport['ots']['passenger_modal-share'].filter({'Country': ['Vaud']})

  cat_dict = {'TIM': ['LDV', '2W'],
              'TP': ['rail', 'metrotram', 'bus'],
              'MA': ['walk', 'bike']}

  #Scénario 2 and 4: PCV, DLS

  #Calcul coefficient 2050 selon OTS. On prend les parts modales de 2050,
  # mais on garde les proportions de 2023 entre les différents modes dans TP, MA et TIM
  #(Source: vision 2050)
  CITEC_2050_val_dict = {'TIM': 0.55, 'TP': 0.38, 'MA': 0.07}
  DLS_2050_val_dict = {'TIM': 587/3241, 'TP': 1143/3241, 'MA': (3241-587-1143)/3241}

  # Assing CITEC values to 2050 using the 2023 split within categories
  idx_ots = dm_modal_share_ots.idx
  idx_fts = dm_modal_share_2.idx
  dm_modal_share_2.array[idx_fts['Vaud'], idx_fts[2025]+1:, :, :] = np.nan  # clear level 2 except for 2025
  dm_modal_share_4.array[idx_fts['Vaud'], idx_fts[2025]+1:, :, :] = np.nan  # clear level 4 except for 2025
  dm_modal_share_2_tmp = None
  dm_modal_share_4_tmp = None
  for key, cat in cat_dict.items():
      dm_modal_ots_cat = dm_modal_share_ots.filter({'Categories1': cat})
      dm_modal_ots_cat.normalise(dim='Categories1', inplace=True, keep_original=False)
      # level 2
      dm_modal_fts_cat_2 = dm_modal_share_2.filter({'Categories1': cat})
      dm_modal_fts_cat_2.array[idx_fts['Vaud'], idx_fts[2050], ...] \
          = CITEC_2050_val_dict[key] * dm_modal_ots_cat.array[idx_ots['Vaud'], idx_ots[2023], ...]
      # level 4
      dm_modal_fts_cat_4 = dm_modal_share_4.filter({'Categories1': cat})
      dm_modal_fts_cat_4.array[idx_fts['Vaud'], idx_fts[2050], ...] \
          = DLS_2050_val_dict[key] * dm_modal_ots_cat.array[idx_ots['Vaud'], idx_ots[2023], ...]
      if dm_modal_share_2_tmp is None:
          dm_modal_share_2_tmp = dm_modal_fts_cat_2.copy()
          dm_modal_share_4_tmp = dm_modal_fts_cat_4.copy()
      else:
          dm_modal_share_2_tmp.append(dm_modal_fts_cat_2, dim='Categories1')
          dm_modal_share_4_tmp.append(dm_modal_fts_cat_4, dim='Categories1')

  dm_modal_share_2 = dm_modal_share_2_tmp.copy()
  dm_modal_share_2.sort('Categories1')
  dm_modal_share_4 = dm_modal_share_4_tmp.copy()
  dm_modal_share_4.sort('Categories1')

  #On a les valeurs du PCV pour 2030, on les introduit pour 2030.
  values_2030 = {'rail': 0.25, 'metrotram': 0.02, 'bus':0.03, 'walk': 0.05, 'bike': 0.03, 'LDV': 0.6, '2W':0.03} #Source: PCV
  for key, values in values_2030.items():
      dm_modal_share_2.array[idx_fts['Vaud'], idx_fts[2030], idx_fts['tra_passenger_modal-share'], idx_fts[key]] = values

  linear_fitting(dm_modal_share_2, dm_modal_share_2.col_labels['Years'])
  dm_modal_share_2.normalise(dim='Categories1', inplace=True)

  linear_fitting(dm_modal_share_4, dm_modal_share_4.col_labels['Years'])
  dm_modal_share_4.normalise(dim='Categories1', inplace=True)

  DM_fts['fts']['passenger_modal-share'] = {2: dm_modal_share_2, 4: dm_modal_share_4}

  # FIXME ! Level 4 is missing the LDV reduction in 2025 (should it be there?)

  # ======================  OCCUPANCY  ========================================================
  dm_occupancy_2 = DM_transport['fts']['passenger_occupancy'][2].filter({'Country': ['Vaud']})
  dm_occupancy_ots = DM_transport['ots']['passenger_occupancy'].filter({'Country': ['Vaud']})

  #Scénario PCV:
  idx = dm_occupancy_2.idx
  dm_occupancy_2.array[idx['Vaud'], idx[2030]:, idx['tra_passenger_occupancy'], :] = np.nan
  array_PCV_occupancy = dm_occupancy_2.array

  dm_occupancy_2.array[idx['Vaud'], idx[2030], idx['tra_passenger_occupancy'], idx['LDV']] = 1.9
  dm_occupancy_2.fill_nans(dim_to_interp='Years')

  DM_fts['fts']['passenger_occupancy'] = {2: dm_occupancy_2, 4: dm_occupancy_2.copy()}

  # ======================  NEW FUEL EFF  ========================================================

  dm_new_eff_2 = DM_transport['fts']['passenger_veh-efficiency_new'][2].filter({'Country': ['Vaud']})
  dm_new_eff_ots = DM_transport['ots']['passenger_veh-efficiency_new'].filter({'Country': ['Vaud']})
  dm_new_eff_4 = DM_transport['fts']['passenger_veh-efficiency_new'][4].filter({'Country': ['Vaud']})

  #PCV: on prend les hypothèses d'amélioration ci dessous (source: canton de Vaud).
  reduction_2050_thermique = 1 - 0.39
  reduction_2050_electrique = 1 - 0.13
  reduction_2050_PHEV = 0.5*reduction_2050_electrique + 0.5*reduction_2050_thermique
  reduction_map = {'BEV': reduction_2050_electrique, 'ICE-diesel': reduction_2050_thermique,
                   'ICE-gasoline': reduction_2050_thermique, 'PHEV-diesel': reduction_2050_PHEV,
                   'PHEV-gasoline': reduction_2050_PHEV}

  idx = dm_new_eff_2.idx
  idx0 = dm_new_eff_ots.idx
  for cat, reduction_2050 in reduction_map.items():
      dm_new_eff_2.array[idx['Vaud'], idx[2050], idx['tra_passenger_veh-efficiency_new'], idx['LDV'], idx[cat]] = (
          dm_new_eff_ots.array[idx0['Vaud'], idx0[2023], idx0['tra_passenger_veh-efficiency_new'], idx0['LDV'], idx0[cat]] * reduction_2050)
      dm_new_eff_2.array[idx['Vaud'], 1:idx[2050], idx['tra_passenger_veh-efficiency_new'], idx['LDV'], idx[cat]] = np.nan

  linear_fitting(dm_new_eff_2, dm_new_eff_2.col_labels['Years'])

  DM_fts['fts']['passenger_veh-efficiency_new'] = {2: dm_new_eff_2}

  #Scénario 4:
  #on applique la réduction de 2/3 pour 2025 et 2050 due à la réduction de la taille des véhicules.
  reduction_map_4 = {cle: valeur * 2/3 for cle, valeur in reduction_map.items()}
  idx = dm_new_eff_4.idx
  idx0 = dm_new_eff_ots.idx
  for cat, reduction_2050 in reduction_map_4.items():
      dm_new_eff_4.array[idx['Vaud'], idx[2050], idx['tra_passenger_veh-efficiency_new'], idx['LDV'], idx[cat]] = (
          dm_new_eff_ots.array[idx0['Vaud'], idx0[2023], idx0['tra_passenger_veh-efficiency_new'], idx0['LDV'], idx0[cat]]* reduction_2050)
      dm_new_eff_4.array[idx['Vaud'], 1:idx[2050], idx['tra_passenger_veh-efficiency_new'], idx['LDV'], idx[cat]] = np.nan
      dm_new_eff_4.array[idx['Vaud'], idx[2025], idx['tra_passenger_veh-efficiency_new'], idx['LDV'], idx[cat]] = 2/3*dm_new_eff_ots.array[idx0['Vaud'], idx0[2023], idx0['tra_passenger_veh-efficiency_new'], idx0['LDV'], idx0[cat]]

  #on réduit l'intensité énergétique des BEV car on vend 50% de véhicules intermédiaires dès 2025
  prop_VUS = 0.5
  efficiency_VUS = 0.11

  efficiency_2025_bev = dm_new_eff_4.array[idx['Vaud'], idx[2025], idx['tra_passenger_veh-efficiency_new'], idx['LDV'], idx['BEV']]
  efficiency_2025_bev_vus = efficiency_2025_bev*(1-prop_VUS) + efficiency_VUS*prop_VUS
  dm_new_eff_4.array[idx['Vaud'], idx[2025], idx['tra_passenger_veh-efficiency_new'], idx['LDV'], idx['BEV']] = efficiency_2025_bev_vus

  efficiency_2050_bev = dm_new_eff_4.array[idx['Vaud'], idx[2050], idx['tra_passenger_veh-efficiency_new'], idx['LDV'], idx['BEV']]
  efficiency_2050_bev_vus = efficiency_2050_bev*2/3*(1-prop_VUS) + efficiency_VUS*prop_VUS
  dm_new_eff_4.array[idx['Vaud'], idx[2050], idx['tra_passenger_veh-efficiency_new'], idx['LDV'], idx['BEV']] = efficiency_2050_bev_vus

  linear_fitting(dm_new_eff_4, dm_new_eff_4.col_labels['Years'])
  DM_fts['fts']['passenger_veh-efficiency_new'][4] = dm_new_eff_4

  # ======================  NEW SALES VEHICLES  ========================================================
  dm_new_tech_share_2 = DM_transport['fts']['passenger_technology-share_new'][2].filter({'Country': ['Vaud']})
  dm_new_tech_share_4 = DM_transport['fts']['passenger_technology-share_new'][4].filter({'Country': ['Vaud']})
  dm_new_tech_share_ots = DM_transport['ots']['passenger_technology-share_new'].filter({'Country': ['Vaud']})

  #PCV: ci dessous les valeurs fixées par le PCV pour 2035.
  prop_EV_PHEV_2035_PCV = 0.65
  prop_FCEV_2035_PCV = 0.00
  prop_ICE_2035_PCV = 0.35

  array_PCV_new_tech_share = dm_new_tech_share_2.array
  idx = dm_new_tech_share_ots.idx

  #on fait les calculs suivants pour garder les proportions entre les différentes motorisations
  denominator_EV_PHEV = ( dm_new_tech_share_ots.array[idx['Vaud'], idx[2023], idx['tra_passenger_technology-share_new'], idx['LDV'], idx['BEV']] +
                        dm_new_tech_share_ots.array[idx['Vaud'], idx[2023], idx['tra_passenger_technology-share_new'], idx['LDV'], idx['PHEV-gasoline']] +
                        dm_new_tech_share_ots.array[idx['Vaud'], idx[2023], idx['tra_passenger_technology-share_new'], idx['LDV'], idx['PHEV-diesel']] )
  prop_BEV_EV_2023 = dm_new_tech_share_ots.array[idx['Vaud'], idx[2023], idx['tra_passenger_technology-share_new'], idx['LDV'], idx['BEV']] / denominator_EV_PHEV
  prop_PHEV_diesel_EV_2023 = dm_new_tech_share_ots.array[idx['Vaud'], idx[2023], idx['tra_passenger_technology-share_new'], idx['LDV'], idx['PHEV-diesel']] / denominator_EV_PHEV
  prop_PHEV_gasoline_EV_2023 = dm_new_tech_share_ots.array[idx['Vaud'], idx[2023], idx['tra_passenger_technology-share_new'], idx['LDV'], idx['PHEV-gasoline']] / denominator_EV_PHEV
  prop_gasoline_ICE_2023 = ( dm_new_tech_share_ots.array[idx['Vaud'], idx[2023], idx['tra_passenger_technology-share_new'], idx['LDV'], idx['ICE-gasoline']]  /
                            ( dm_new_tech_share_ots.array[idx['Vaud'], idx[2023], idx['tra_passenger_technology-share_new'], idx['LDV'], idx['ICE-gasoline']] +
                              dm_new_tech_share_ots.array[idx['Vaud'], idx[2023], idx['tra_passenger_technology-share_new'], idx['LDV'], idx['ICE-diesel']] ) )
  prop_diesel_ICE_2023 = 1 - prop_gasoline_ICE_2023

  idx = dm_new_tech_share_2.idx
  dm_new_tech_share_2.array[idx['Vaud'], :, idx['tra_passenger_technology-share_new'], :] = np.nan

  #On trouve les proportions de motorisation pour 2035.
  prop_BEV_2035_PCV = prop_EV_PHEV_2035_PCV*prop_BEV_EV_2023
  prop_PHEV_diesel_2035_PCV = prop_EV_PHEV_2035_PCV * prop_PHEV_diesel_EV_2023
  prop_PHEV_gasoline_2035_PCV = prop_EV_PHEV_2035_PCV * prop_PHEV_gasoline_EV_2023
  prop_diesel_ICE_2035_PCV = prop_ICE_2035_PCV * prop_diesel_ICE_2023
  prop_gasoline_ICE_2035_PCV = prop_ICE_2035_PCV * prop_gasoline_ICE_2023

  values_2035_new_tech_share = {'BEV': prop_BEV_2035_PCV, 'ICE-diesel': prop_diesel_ICE_2035_PCV, 'ICE-gasoline': prop_gasoline_ICE_2035_PCV, 'PHEV-diesel': prop_PHEV_diesel_2035_PCV, 'PHEV-gasoline': prop_PHEV_gasoline_2035_PCV}
  for key,values in values_2035_new_tech_share.items():
      dm_new_tech_share_2.array[idx['Vaud'], idx[2035], idx['tra_passenger_technology-share_new'], idx['LDV'], idx[key]] = values

  dm_new_tech_share_trend_PCV = dm_new_tech_share_ots.copy()
  dm_new_tech_share_trend_PCV.append(dm_new_tech_share_2, dim = 'Years')
  linear_fitting(dm_new_tech_share_trend_PCV, dm_new_tech_share_trend_PCV.col_labels['Years'])
  idx = dm_new_tech_share_trend_PCV.idx
  dm_new_tech_share_trend_PCV.array[idx['Vaud'], idx[2035]+1: , idx['tra_passenger_technology-share_new'], :] = np.nan
  linear_fitting(dm_new_tech_share_trend_PCV, dm_new_tech_share_trend_PCV.col_labels['Years'], based_on= dm_new_tech_share_2.col_labels['Years'], min_tb=0)

  dm_new_tech_share_trend_PCV.normalise(dim='Categories2', inplace=True)
  dm_new_tech_share_2_PVC = dm_new_tech_share_trend_PCV.filter({"Years": dm_new_tech_share_2.col_labels['Years']})
  DM_fts['fts']['passenger_technology-share_new'] = {2: dm_new_tech_share_2_PVC}

  #Scénario 3:
  values_2025_new_tech_share_3 = {'BEV': 1, 'ICE-diesel': 0, 'ICE-gasoline': 0, 'PHEV-diesel': 0, 'PHEV-gasoline': 0}
  idx = dm_new_tech_share_4.idx
  dm_new_tech_share_4.array[idx['Vaud'], :, idx['tra_passenger_technology-share_new'], :] = np.nan
  for key,values in values_2025_new_tech_share_3.items():
      dm_new_tech_share_4.array[idx['Vaud'], idx[2025], idx['tra_passenger_technology-share_new'], idx['LDV'], idx[key]] = values
      dm_new_tech_share_4.array[idx['Vaud'], idx[2050], idx['tra_passenger_technology-share_new'], idx['LDV'], idx[key]] = values

  dm_new_tech_share_trend_4 = dm_new_tech_share_ots.copy()
  dm_new_tech_share_trend_4.append(dm_new_tech_share_4, dim='Years')
  linear_fitting(dm_new_tech_share_trend_4, dm_new_tech_share_trend_4.col_labels['Years'])

  dm_new_tech_share_trend_4.array = np.maximum(dm_new_tech_share_trend_4.array, 0)
  dm_new_tech_share_trend_4.normalise('Categories2', inplace=True)
  dm_new_tech_share_trend_4_fts = dm_new_tech_share_trend_4.filter({"Years": dm_new_tech_share_4.col_labels['Years']})
  DM_fts['fts']['passenger_technology-share_new'][4] = dm_new_tech_share_trend_4_fts

  #======================  DEMANDE  ========================================================
  dm_pkm_2 = DM_transport['fts']['pkm'][2].filter({'Country': ['Vaud']})
  dm_pkm_4 = DM_transport['fts']['pkm'][4].filter({'Country': ['Vaud']})
  dm_pkm_ots = DM_transport['ots']['pkm'].filter({'Country': ['Vaud']})

  idx = dm_pkm_ots.idx
  demande_transport_2019 = dm_pkm_ots.array[idx['Vaud'], idx[2019], 0]
  demande_transport_2023 = dm_pkm_ots.array[idx['Vaud'], idx[2023], 0]
  croissance_demande_annuelle = 0.0091

  idx = dm_pkm_2.idx
  dm_pkm_2.array[idx['Vaud'], idx[2025]:idx[2050]+1, 0] = np.nan
  dm_pkm_2.array[idx['Vaud'], idx[2030], 0] = demande_transport_2019*(1+croissance_demande_annuelle)**5
  dm_pkm_2.array[idx['Vaud'], idx[2025], 0] = demande_transport_2019
  dm_pkm_2.array[idx['Vaud'], idx[2050], 0] = dm_pkm_2.array[idx['Vaud'], idx[2030], 0]
  linear_fitting(dm_pkm_2, dm_pkm_2.col_labels['Years'])
  dm_pkm_PCV = dm_pkm_ots.copy()
  dm_pkm_PCV.append(dm_pkm_2, dim ='Years')

  DM_fts['fts']['pkm'] = {2: dm_pkm_2}

  #SCENARIO 3:
  idx = dm_pkm_4.idx
  dm_pkm_4.array[idx['Vaud'], 0:idx[2050]+1, 0] = np.nan
  dm_pkm_4.array[idx['Vaud'], idx[2025], 0] = demande_transport_2023
  dm_pkm_4.array[idx['Vaud'], idx[2050], 0] = 3241
  linear_fitting(dm_pkm_4, dm_pkm_4.col_labels['Years'])
  dm_pkm_4.array[idx['Vaud'], idx[2025], 0] *= 0.95
  linear_fitting(dm_pkm_4, dm_pkm_4.col_labels['Years'])

  DM_fts['fts']['pkm'] = {4: dm_pkm_4}

  #======================  MESURE 5  ==================================== (CALCULER LES EMISSIONS MOYENNES DU NOUVEAU PARC EN 2021)
  categories_transport =['BEV', 'CEV', 'FCEV', 'ICE-diesel', 'ICE-gas', 'ICE-gasoline', 'PHEV-diesel', 'PHEV-gasoline', 'mt']
  df_mesure_5 = pd.DataFrame(index= categories_transport)

  idx = dm_new_eff_ots.idx
  efficiency_2018 = dm_new_eff_ots.array[idx['Vaud'], idx[2018], idx['tra_passenger_veh-efficiency_new'], idx['LDV'], :]
  df_mesure_5['efficiency 2018'] = efficiency_2018

  idx = dm_new_tech_share_ots.idx
  market_shares_2018 = dm_new_tech_share_ots.array[idx['Vaud'], idx[2018], idx['tra_passenger_technology-share_new'], idx['LDV'], :]
  df_mesure_5['part de marché 2018'] = market_shares_2018

  idx = dm_new_eff_2.idx
  efficiency_2035 = dm_new_eff_2.array[idx['Vaud'], idx[2035], idx['tra_passenger_veh-efficiency_new'], idx['LDV'], :]
  df_mesure_5['efficiency 2035'] = efficiency_2035

  idx = dm_new_tech_share_2.idx
  market_shares_2035 = dm_new_tech_share_2.array[idx['Vaud'], idx[2035], idx['tra_passenger_technology-share_new'], idx['LDV'], :]
  df_mesure_5['part de marché 2035'] = market_shares_2035

  df_mesure_5.drop('ICE-gas', inplace = True)
  df_mesure_5.drop('FCEV', inplace = True)
  df_mesure_5.drop('CEV', inplace = True)
  df_mesure_5.drop('mt', inplace = True)

  efficiency_2018 = (df_mesure_5['efficiency 2018']*df_mesure_5['part de marché 2018']).sum()
  efficiency_2035 = (df_mesure_5['efficiency 2035']*df_mesure_5['part de marché 2035']).sum()
  #print(f"Efficiency: 2018 :" + str(round(efficiency_2018, 3)) + " ; 2035: "+ str(round(efficiency_2035, 3)))

  df_mesure_5['facteurs_CO2_2035'] = df_mesure_5['efficiency 2035']
  df_mesure_5['facteurs_CO2_2018'] = df_mesure_5['efficiency 2018']

  dm_carbon_intensity = DM_transport['constant']
  idx = dm_carbon_intensity.idx

  df_mesure_5.loc['ICE-diesel', 'facteurs_CO2_2035'] *= dm_carbon_intensity.array[0, idx['CO2'], idx['ICE-diesel']]
  df_mesure_5.loc['ICE-diesel', 'facteurs_CO2_2018'] *= dm_carbon_intensity.array[0, idx['CO2'], idx['ICE-diesel']]
  df_mesure_5.loc['PHEV-diesel', 'facteurs_CO2_2035'] *= dm_carbon_intensity.array[0, idx['CO2'], idx['PHEV-diesel']]
  df_mesure_5.loc['PHEV-diesel', 'facteurs_CO2_2018'] *= dm_carbon_intensity.array[0, idx['CO2'], idx['PHEV-diesel']]
  df_mesure_5.loc['ICE-gasoline', 'facteurs_CO2_2035'] *= dm_carbon_intensity.array[0, idx['CO2'], idx['ICE-gasoline']]
  df_mesure_5.loc['ICE-gasoline', 'facteurs_CO2_2018'] *= dm_carbon_intensity.array[0, idx['CO2'], idx['ICE-gasoline']]
  df_mesure_5.loc['PHEV-gasoline', 'facteurs_CO2_2035'] *= dm_carbon_intensity.array[0, idx['CO2'], idx['PHEV-gasoline']]
  df_mesure_5.loc['PHEV-gasoline', 'facteurs_CO2_2018'] *= dm_carbon_intensity.array[0, idx['CO2'], idx['PHEV-gasoline']]

  dm_elec_intensity = DM_transport['fxa']['emission-factor-electricity']
  idx = dm_elec_intensity.idx
  electricite_2035_intensity = dm_elec_intensity.array[idx['Vaud'], idx[2035], 0, idx['CO2'], 0]
  df_mesure_5.loc['BEV', 'facteurs_CO2_2035'] *= dm_elec_intensity.array[idx['Vaud'], idx[2035], 0, idx['CO2'], 0] #dm.array[0, idx['CO2'], idx['PHEV-gasoline']]
  df_mesure_5.loc['BEV', 'facteurs_CO2_2018'] *= dm_elec_intensity.array[idx['Vaud'], idx[2035], 0, idx['CO2'], 0]

  pd.set_option('display.max_columns', 10)

  emissions_moyennes_2018 = (df_mesure_5['facteurs_CO2_2018']*df_mesure_5['part de marché 2018']).sum()
  emissions_moyennes_2035 = (df_mesure_5['facteurs_CO2_2035']*df_mesure_5['part de marché 2035']).sum()

  df_mesure_5.loc['Total', 'facteurs_CO2_2018'] = emissions_moyennes_2018
  df_mesure_5.loc['Total', 'facteurs_CO2_2035'] = emissions_moyennes_2035
  #df_mesure_5.to_excel('mesure_5_nv.xlsx', index=True)

  ratio_2018_2035 = emissions_moyennes_2035/ emissions_moyennes_2018

  idx = dm_new_eff_4.idx
  efficiency_bev_2035_DLS = dm_new_eff_4.array[idx['Vaud'], idx[2035], idx['tra_passenger_veh-efficiency_new'], idx['LDV'], idx['BEV']]
  emissions_moyennes_2035_DLS = electricite_2035_intensity * efficiency_bev_2035_DLS


  # ======================  EXPORTS FINAUX   ===========================
  # Load existing DM_transport
  this_dir = os.path.dirname(os.path.abspath(__file__))
  pickle_file = os.path.join(this_dir, '../../../../data/datamatrix/transport.pickle')
  with open(pickle_file, 'rb') as handle:
    DM_transport_old = pickle.load(handle)

  utils.add_aviation_data_to_DM(DM_fts, DM_transport_old)

  my_pickle_dump(DM_new=DM_fts, local_pickle_file=pickle_file)
  sort_pickle(pickle_file)

  return
