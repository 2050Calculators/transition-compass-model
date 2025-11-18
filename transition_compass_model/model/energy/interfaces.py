import numpy as np
import pyomo.environ as pyo

from model.common.auxiliary_functions import dm_add_missing_variables

def extract_transport_demand(DM_tra):
  dm_demand_trend = DM_tra['freight'].filter({'Categories2': ['BEV', 'CEV']})
  dm_demand_trend.operation('tra_freight_transport-demand-tkm', '*', 'tra_freight_energy-intensity', out_col='freight_energy-consumption', unit='kWh')
  dm_demand_trend.group_all('Categories1')
  dm_demand_trend.filter({'Variables': ['freight_energy-consumption']}, inplace=True)
  dm_demand_trend.change_unit('freight_energy-consumption', old_unit='kWh', new_unit='TWh', factor=1e-9)
  dm_demand_trend.groupby({'electricity': '.*'}, regex=True, inplace=True, dim='Categories1')


  dm_tmp = DM_tra['passenger'].filter({'Categories2': ['BEV', 'CEV', 'mt']})
  # !FIXME: I'm dropping 2W for consistency since EnergyScope does not allow it
  dm_tmp.drop('Categories1', '2W')
  dm_tmp.operation('tra_passenger_transport-demand', '*', 'tra_passenger_energy-intensity', out_col='pass_energy-consumption', unit='kWh')
  dm_tmp.change_unit('pass_energy-consumption', old_unit='kWh', new_unit='TWh', factor=1e-9)
  dm_tmp.group_all('Categories1')
  dm_tmp.filter({'Variables': ['pass_energy-consumption']}, inplace=True)
  dm_tmp.groupby({'electricity': '.*'}, regex=True, inplace=True, dim='Categories1')

  dm_demand_trend.append(dm_tmp, dim='Variables')
  dm_demand_trend.groupby({'tra_energy-consumption': '.*'}, regex=True, inplace=True, dim='Variables')

  return dm_demand_trend


def impose_transport_demand_pyomo(m, endyr, share_pop, DM_tra, cntr):
  eps = 1e-5

  dm_demand_trend = DM_tra['freight'].filter({'Categories2': ['BEV', 'CEV']})
  dm_demand_trend.operation('tra_freight_transport-demand-tkm', '*', 'tra_freight_energy-intensity', out_col='freight_energy-consumption', unit='kWh')
  dm_demand_trend.group_all('Categories1')
  dm_demand_trend.filter({'Variables': ['freight_energy-consumption']}, inplace=True)
  dm_demand_trend.change_unit('freight_energy-consumption', old_unit='kWh', new_unit='TWh', factor=1e-9)
  dm_demand_trend.groupby({'electricity': '.*'}, regex=True, inplace=True, dim='Categories1')


  dm_tmp = DM_tra['passenger'].filter({'Categories2': ['BEV', 'CEV', 'mt']})
  dm_tmp.drop('Categories1', '2W')
  dm_tmp.operation('tra_passenger_transport-demand', '*', 'tra_passenger_energy-intensity', out_col='pass_energy-consumption', unit='kWh')
  dm_tmp.change_unit('pass_energy-consumption', old_unit='kWh', new_unit='TWh', factor=1e-9)
  dm_tmp.group_all('Categories1')
  dm_tmp.filter({'Variables': ['pass_energy-consumption']}, inplace=True)
  dm_tmp.groupby({'electricity': '.*'}, regex=True, inplace=True, dim='Categories1')

  dm_demand_trend.append(dm_tmp, dim='Variables')
  dm_demand_trend.groupby({'tra_energy-consumption': '.*'}, regex=True, inplace=True, dim='Variables')

  # Set public transport pkm share
  # !FIXME add aviation to EnergyScope
  DM_tra['passenger'].change_unit('tra_passenger_transport-demand',
                                  old_unit='pkm', new_unit='Mpkm', factor=1e-6)
  DM_tra['passenger'].filter({'Categories2': ['BEV', 'CEV',  'mt']}, inplace=True) # 'PHEV-diesel', 'PHEV-gasoline',
  dm_pass_demand = DM_tra['passenger'].filter(
    {'Years': [endyr], 'Variables': ['tra_passenger_transport-demand']})
  # !FIXME: I'm dropping two wheels because it is not allowed by Energy Scope
  dm_pass_demand.drop('Categories1', '2W')
  dm_pub_pri = dm_pass_demand.group_all('Categories2', inplace=False)
  dm_pub_pri.groupby(
    {'public': ['metrotram', 'bus', 'rail'], 'private': ['LDV']},
    dim='Categories1', inplace=True)

  dm_pub_pri.normalise('Categories1', inplace=True)
  m.share_mobility_public_min.set_value(dm_pub_pri[cntr, 0, 0, 'public'] - eps)
  m.share_mobility_public_max.set_value(dm_pub_pri[cntr, 0, 0, 'public'] + eps)


  # Set rail transport in freight share
  DM_tra['freight'].filter({'Categories2': ['BEV', 'CEV']}, inplace=True) # 'PHEV-diesel', 'PHEV-gasoline'

  m.share_freight_train_min = 0.99- eps
  m.share_freight_train_max = 1

  # Set total Passenger mobility in pkm
  tot_Mpkm = np.nansum(
    dm_pass_demand[0, 0, 'tra_passenger_transport-demand', :, :],
    axis=(-1, -2)) / share_pop
  m.end_uses_demand_year["MOBILITY_PASSENGER", "TRANSPORTATION"] = tot_Mpkm

  # Set total Freight mobility in pkm
  DM_tra['freight'].change_unit('tra_freight_transport-demand-tkm',
                                old_unit='tkm', new_unit='Mtkm', factor=1e-6)
  tot_Mtkm = np.nansum(
    DM_tra['freight'][0, 0, 'tra_freight_transport-demand-tkm', :, :],
    axis=(-1, -2)) / share_pop
  m.end_uses_demand_year["MOBILITY_FREIGHT", "TRANSPORTATION"] =  tot_Mtkm

  # Passenger private technology share
  # It should be both 2W and LDV according to the Calculator, but energyscope only has LDV
  dm_private = DM_tra['passenger'].filter({'Categories1': ['LDV']})
  dm_private.normalise(dim='Categories2', inplace=True, keep_original=True)
  dm_private.groupby({'CAR_BEV': ['BEV']},
                     dim='Categories2', inplace=True)
  dm_private.filter_w_regex({'Categories2': 'CAR.*'}, inplace=True)
  dm_private = dm_private.flatten()
  dm_private.rename_col_regex('LDV_', '', dim='Categories1')
  private_mob_tech = list(m.TECHNOLOGIES_OF_END_USES_TYPE["MOB_PRIVATE"])
  for cat in private_mob_tech:
    if cat in dm_private.col_labels['Categories1']:
      val_perc = dm_private[0, endyr, 'tra_passenger_transport-demand_share', cat]
      val_abs = dm_private[
                  0, endyr, 'tra_passenger_transport-demand', cat] / share_pop
    else:  # If it is not electricity technology, set shares to 0
      val_perc = 0
      val_abs = 0
    m.fmin_perc[cat] = max(val_perc * (1 - eps), 0)
    m.fmax_perc[cat] = min(val_perc * (1 + eps), 1)
    m.f_min[cat] = 0
    m.f_max[cat] = val_abs * (1 + eps)

  # Passenger public technology share
  dm_public = DM_tra['passenger'].filter(
    {'Categories1': ['metrotram', 'bus', 'rail']})
  dm_public = dm_public.flatten()
  dm_public.groupby(
    {'TRAIN_PUB': ['rail_CEV'],
     'TRAMWAY_TROLLEY': ['metrotram_mt', 'bus_CEV']}, dim='Categories1',
    inplace=True)
  dm_public.normalise('Categories1', inplace=True, keep_original=True)
  public_mob_tech = list(m.TECHNOLOGIES_OF_END_USES_TYPE["MOB_PUBLIC"])
  for cat in public_mob_tech:
    if cat in ['TRAIN_PUB', 'TRAMWAY_TROLLEY']:
      val_perc = dm_public[0, endyr, 'tra_passenger_transport-demand_share', cat]
      val_abs = dm_public[
                  0, endyr, 'tra_passenger_transport-demand', cat] / share_pop
    else:
      val_perc = 0
      val_abs = 0
    m.fmin_perc[cat] = max(val_perc * (1 - eps), 0)
    m.fmax_perc[cat] = min(val_perc * (1 + eps), 1)
    m.f_min[cat] = 0
    m.f_max[cat] = val_abs * (1 + eps)

  # Efficiency - Passenger
  mapping = {'TRAMWAY_TROLLEY': 'ELECTRICITY',
             'TRAIN_PUB': 'ELECTRICITY',
             'CAR_BEV': 'ELECTRICITY'}
  dm_eff = dm_private.copy()
  dm_eff.append(dm_public, dim='Categories1')
  for veh, fuel in mapping.items():
    val = dm_eff[0, endyr, 'tra_passenger_energy-intensity', veh]
    m.layers_in_out[veh, fuel] = -val

  # Efficiency - Freight
  mapping = {'TRAIN_FREIGHT': 'ELECTRICITY'}
  DM_tra['freight'].operation('tra_freight_transport-demand-tkm', '*',
                              'tra_freight_energy-intensity',
                              out_col='tra_freight_energy-demand', unit='GWh')
  dm_eff_freight = DM_tra['freight']
  dm_eff_freight.drop(col_label='tra_freight_energy-intensity', dim='Variables')
  dm_eff_freight.groupby(
    {'TRAIN_FREIGHT': ['rail']},
    dim='Categories1', inplace=True)
  dm_eff_freight.group_all('Categories2')
  dm_eff_freight.operation('tra_freight_energy-demand', '/',
                           'tra_freight_transport-demand-tkm',
                           out_col='tra_freight_energy-intensity',
                           unit='kWh/tkm')
  dm_eff_freight.fill_nans('Years')
  for veh, fuel in mapping.items():
    val = dm_eff_freight[0, endyr, 'tra_freight_energy-intensity', veh]
    m.layers_in_out[veh, fuel] = -val

  return dm_demand_trend


def impose_space_heating_pyomo(m, endyr, share_of_pop, DM_bld, cntr, eps):
  # Useful energy demand
  dm_heating = DM_bld.filter(
    {'Variables': ['bld_heating_useful-energy'], 'Years': [endyr]})
  dm_heating.normalise(dim='Categories1', inplace=True, keep_original=True)
  val = dm_heating[
    cntr, endyr, 'bld_heating_useful-energy_share', 'district-heating']
  m.share_heat_dhn_min = val
  m.share_heat_dhn_max = val

  # Set decentralised heating shares by technology (- district heating)
  # DEC_HP_ELEC	DEC_THHP_GAS	DEC_COGEN_GAS	DEC_COGEN_OIL	DEC_ADVCOGEN_GAS
  # DEC_ADVCOGEN_H2	DEC_BOILER_GAS	DEC_BOILER_WOOD	DEC_BOILER_OIL	DEC_SOLAR	DEC_DIRECT_ELEC
  dm_heating = DM_bld.filter({'Variables': ['bld_heating_useful-energy',
                                            'bld_heating_energy-consumption']})
  dm_heating.drop(dim='Categories1', col_label='district-heating')
  dm_heating.normalise(dim='Categories1', inplace=True, keep_original=True)
  dm_heating.rename_col(
    ['heat-pump', 'electricity'],
    ['DEC_HP_ELEC', 'DEC_DIRECT_ELEC'],  # IND_DIRECT_ELEC
    dim='Categories1')
  decen_heat = list(m.TECHNOLOGIES_OF_END_USES_TYPE["HEAT_LOW_T_DECEN"])
  for cat in decen_heat:
    if cat in dm_heating.col_labels['Categories1']:
      val_perc = dm_heating[cntr, endyr, 'bld_heating_useful-energy_share', cat]
      val_abs = dm_heating[
                  cntr, endyr, 'bld_heating_useful-energy', cat] / share_of_pop
    else:
      val_perc = 0
      val_abs = 0
    m.fmin_perc[cat] = max(val_perc * (1 - eps), 0)
    m.fmax_perc[cat] = min(val_perc * (1 + eps), 1)
    m.f_min[cat] = val_abs * (1 - eps)
    m.f_max[cat] = val_abs * (1 + eps)

  # Set efficiencies
  # mapping = {'DEC_HP_ELEC': 'ELECTRICITY',
  #            'DEC_DIRECT_ELEC': 'ELECTRICITY'}
  mapping = {'DEC_HP_ELEC': 'ELECTRICITY'}
  # mapping = {'DEC_HP_ELEC': 'ELECTRICITY', 'DEC_BOILER_GAS': 'NG', 'DEC_BOILER_OIL': 'LFO',
  #           'DEC_DIRECT_ELEC': 'ELECTRICITY'}
  # !FIXME : there is a problem here when bld_heating is 0!
  dm_heating.operation('bld_heating_energy-consumption', '/',
                       'bld_heating_useful-energy',
                       out_col='bld_rev_eff', unit='%')
  dm_heating.fill_nans('Years')
  # dm_heating[cntr, endyr, 'bld_rev_eff', 'DEC_BOILER_WOOD'] = dm_heating[cntr, endyr, 'bld_rev_eff', 'DEC_BOILER_OIL']
  for veh, fuel in mapping.items():
    val = dm_heating[cntr, endyr, 'bld_rev_eff', veh]
    m.layers_in_out[veh, fuel] = -val

  return


def reorganise_space_heat_hot_water(DM_bld, DM_ind):
  # Extract household hot-water heating
  dm_house_hotwater = DM_bld['households_hot-water'].filter({'Variables': ['bld_hw_useful-energy', 'bld_hot-water_energy-demand'],
                                                             'Categories1': ['district-heating', 'electricity', 'heat-pump']})
  dm_house_hotwater.rename_col(['bld_hw_useful-energy', 'bld_hot-water_energy-demand'],
                               ['bld_useful-energy_hot-water_households', 'bld_energy-consumption_hot-water_households'], 'Variables')
  dm_house_hotwater.deepen(based_on='Variables')
  dm_house_hotwater.deepen(based_on='Variables')

  # Extract household space-heating
  dm_house_heat = DM_bld['households_heating'].filter({'Variables': ['bld_heating', 'bld_energy-demand_heating'],
                                                       'Categories1': ['district-heating', 'electricity', 'heat-pump']})
  dm_house_heat.rename_col('bld_heating', 'bld_useful-energy_space-heating_households', 'Variables')
  dm_house_heat.rename_col('bld_energy-demand_heating', 'bld_energy-consumption_space-heating_households', 'Variables')
  dm_house_heat.deepen(based_on='Variables')
  dm_house_heat.deepen(based_on='Variables')

  dm_house_heat.append(dm_house_hotwater, dim='Categories3')
  dm_house_heat.switch_categories_order('Categories3', 'Categories2')
  dm_house_heat.switch_categories_order('Categories3', 'Categories1')

  # Extract service space-heating & hot-water
  dm_service_heat = DM_bld['services_all'].filter({'Variables': ['bld_services_useful-energy', 'bld_services_energy-consumption'],
                                                   'Categories1': ['space-heating', 'hot-water'],
                                                   'Categories2': ['district-heating', 'electricity', 'heat-pump']})
  dm_service_heat.rename_col('bld_services_useful-energy', 'bld_useful-energy_services', 'Variables')
  dm_service_heat.rename_col('bld_services_energy-consumption', 'bld_energy-consumption_services', 'Variables')
  dm_service_heat.deepen(based_on='Variables')
  dm_service_heat.switch_categories_order('Categories3', 'Categories1')
  dm_service_heat.switch_categories_order('Categories3', 'Categories2')

  # Extract service hot-water heating
  dm_heat = dm_house_heat.copy()
  dm_heat.append(dm_service_heat, dim='Categories1')

  # Extract industry space-heat
  dm_ind_heat = DM_ind['ind-energy-demand'].filter({'Variables': ['ind_energy-end-use'],
                                               'Categories1': ['space-heating', 'hot-water'],
                                               'Categories2': ['district-heating', 'electricity', 'heat-pump']})
  dm_ind_heat.rename_col('ind_energy-end-use', 'ind_heat_energy-consumption', 'Variables')

  return dm_heat, dm_ind_heat

def impose_buildings_demand_pyomo(m, endyr, share_of_pop, DM_bld, DM_ind, cntr):
  eps = 1e-5

  validation = False
  if validation:
    DM_bld['households_heating'].filter(
      {'Variables': ['bld_energy-demand_heating', 'bld_energy-demand_cooling']},
      inplace=True)
    DM_bld['households_heating'].filter(
      {'Categories1': ['electricity', 'heat-pump']},
      inplace=True)
    DM_bld['households_heating'].group_all('Categories1')
    DM_bld['households_hot-water'].filter(
      {'Variables': ['bld_hot-water_energy-demand'],
       'Categories1': ['electricity', 'heat-pump']}, inplace=True)
    DM_bld['households_hot-water'].group_all('Categories1')
    dm_household_elec = DM_bld['households_heating'].copy()
    dm_household_elec.append(DM_bld['households_hot-water'], dim='Variables')
    dm_household_elec.append(DM_bld['households_lighting'], dim='Variables')
    dm_household_elec.append(DM_bld['households_electricity'], dim='Variables')


    DM_ind['ind-energy-demand'].filter({'Categories2': ['electricity', 'heat-pump']}, inplace=True)
    DM_ind['ind-energy-demand'].group_all('Categories2')
    DM_ind['ind-energy-demand'].group_all('Categories1')

    DM_bld['services_all'].filter({'Variables': ['bld_services_energy-consumption'], 'Categories2': ['electricity', 'heat-pump']}, inplace=True)
    DM_bld['services_all'].group_all('Categories2')
    DM_bld['services_all'].group_all('Categories1')


  # SPACE HEATING AND HOT WATER (LOW TEMPERATURE HEAT)
  dm_heat, dm_ind_heat = reorganise_space_heat_hot_water(DM_bld, DM_ind)

  # !FIXME: temporary put district heating to 0
  dm_heat[..., 'district-heating'] = 0
  dm_ind_heat[..., 'district-heating'] = 0

  # Group energy demand for output
  # Here I'm only grouping the heating..
  dm_demand_trend = dm_heat.group_all('Categories2', inplace=False)
  dm_demand_trend.group_all('Categories1', inplace=True)
  #dm_tmp = dm_ind_heat.group_all('Categories1', inplace=False)
  #dm_demand_trend.append(dm_tmp, dim='Variables')

  dm_heat.change_unit('bld_useful-energy', old_unit='TWh', new_unit='GWh',
                      factor=1000)
  dm_heat.change_unit('bld_energy-consumption', old_unit='TWh', new_unit='GWh',
                      factor=1000)
  dm_ind_heat.change_unit('ind_heat_energy-consumption', old_unit='TWh',
                          new_unit='GWh', factor=1000)
  # Section: LOW T HEAT
  dm_house_tot = dm_heat.group_all('Categories3', inplace=False).filter(
    {'Categories1': ['households'], 'Categories2': ['space-heating']})
  tot_house_heat = dm_house_tot[
                     0, endyr, 'bld_useful-energy', 'households', 'space-heating'] / share_of_pop
  m.end_uses_demand_year['HEAT_LOW_T_SH', "HOUSEHOLDS"] = tot_house_heat

  dm_house_hw_tot = dm_heat.group_all('Categories3', inplace=False).filter(
    {'Categories1': ['households'], 'Categories2': ['hot-water']})
  tot_house_hw = dm_house_hw_tot[
                   0, endyr, 'bld_useful-energy', 'households', 'hot-water'] / share_of_pop
  m.end_uses_demand_year['HEAT_LOW_T_HW', 'HOUSEHOLDS'] = tot_house_hw

  dm_service_tot = dm_heat.group_all('Categories3', inplace=False).filter(
    {'Categories1': ['services'], 'Categories2': ['space-heating']})
  tot_service_heat = dm_service_tot[
                       0, endyr, 'bld_useful-energy', 'services', 'space-heating'] / share_of_pop
  m.end_uses_demand_year['HEAT_LOW_T_SH', "SERVICES"] = tot_service_heat

  dm_service_hw_tot = dm_heat.group_all('Categories3', inplace=False).filter(
    {'Categories1': ['services'], 'Categories2': ['hot-water']})
  tot_service_hw = dm_service_hw_tot[
                     0, endyr, 'bld_useful-energy', 'services', 'hot-water'] / share_of_pop
  m.end_uses_demand_year['HEAT_LOW_T_HW', 'SERVICES'] = tot_service_hw

  dm_ind_space_tot = dm_ind_heat.group_all('Categories2', inplace=False).filter(
    {'Categories1': ['space-heating']})
  tot_ind_heat = dm_ind_space_tot[
                   0, endyr, 'ind_heat_energy-consumption', 'space-heating'] / share_of_pop
  m.end_uses_demand_year['HEAT_LOW_T_SH', 'INDUSTRY'] = tot_ind_heat

  dm_ind_hw_tot = dm_ind_heat.group_all('Categories2', inplace=False).filter(
    {'Categories1': ['hot-water']})
  tot_ind_hw = dm_ind_hw_tot[
                 0, endyr, 'ind_heat_energy-consumption', 'hot-water'] / share_of_pop
  m.end_uses_demand_year['HEAT_LOW_T_HW', 'INDUSTRY'] = tot_ind_hw

  dm_heat.group_all('Categories2')
  dm_heat.group_all('Categories1')
  dm_heat.rename_col(['bld_energy-consumption', 'bld_useful-energy'],
                     ['bld_heating_energy-consumption',
                      'bld_heating_useful-energy'], 'Variables')
  impose_space_heating_pyomo(m, endyr, share_of_pop, dm_heat, cntr, eps)

  # Section: ELECTRICITY
  dm_house_elec_tot = DM_bld['households_electricity'].copy()
  tot_house_elec = dm_house_elec_tot[
                     0, endyr, 'bld_appliances_tot-elec-demand'] / share_of_pop * 1000
  m.end_uses_demand_year['ELECTRICITY', 'HOUSEHOLDS'] = tot_house_elec

  dm_service_elec_tot = DM_bld['services_all'].filter(
    {'Variables': ['bld_services_energy-consumption'],
     'Categories1': ['elec'], 'Categories2': ['electricity']})
  tot_service_elec = dm_service_elec_tot[
                       0, endyr, 'bld_services_energy-consumption', 'elec', 'electricity'] / share_of_pop * 1000
  m.end_uses_demand_year['ELECTRICITY', 'SERVICES'] = tot_service_elec

  # Section: LIGHTING
  dm_house_light_tot = DM_bld['households_lighting'].copy()
  tot_house_light = dm_house_light_tot[
                      0, endyr, 'bld_residential-lighting'] / share_of_pop * 1000
  m.end_uses_demand_year['LIGHTING', 'HOUSEHOLDS'] = tot_house_light

  dm_service_light_tot = DM_bld['services_all'].filter(
    {'Variables': ['bld_services_energy-consumption'],
     'Categories1': ['lighting'], 'Categories2': ['electricity']})
  tot_service_light = dm_service_light_tot[
                        0, endyr, 'bld_services_energy-consumption', 'lighting', 'electricity'] / share_of_pop * 1000
  m.end_uses_demand_year['LIGHTING', 'SERVICES'] = tot_service_light

  # Section: add electricity and lighting to energy demand
  dm_elec_light = dm_house_elec_tot.copy()
  dm_elec_light.append(dm_house_light_tot, dim='Variables')
  dm_elec_light.append(dm_service_elec_tot.flattest(), dim='Variables')
  dm_elec_light.append(dm_service_light_tot.flattest(), dim='Variables')
  dm_elec_light.groupby({'bld_energy-consumption-elec_electricity': '.*'}, regex=True,
                          inplace=True, dim='Variables')
  dm_elec_light.deepen()
  dm_add_missing_variables(dm_elec_light, {'Categories1': dm_demand_trend.col_labels['Categories1']})
  dm_demand_trend.filter({'Variables': ['bld_energy-consumption']}, inplace=True)
  dm_demand_trend.append(dm_elec_light, dim='Variables')
  dm_demand_trend.groupby({'bld_energy-consumption': '.*'}, regex=True,
                          inplace=True, dim='Variables')

  return dm_demand_trend


def extract_buildings_demand(DM_bld, DM_ind):
  dm_heat, dm_ind_heat = reorganise_space_heat_hot_water(DM_bld, DM_ind)

  dm_heat[..., 'district-heating'] = 0
  dm_ind_heat[..., 'district-heating'] = 0

  # Group energy demand for output
  # Here I'm only grouping the heating..
  dm_demand_trend = dm_heat.group_all('Categories2', inplace=False)
  dm_demand_trend.group_all('Categories1', inplace=True)

  dm_house_light_tot = DM_bld['households_lighting'].copy()
  dm_house_elec_tot = DM_bld['households_electricity'].copy()
  dm_service_elec_tot = DM_bld['services_all'].filter(
    {'Variables': ['bld_services_energy-consumption'],
     'Categories1': ['elec'], 'Categories2': ['electricity']})
  dm_service_light_tot = DM_bld['services_all'].filter(
    {'Variables': ['bld_services_energy-consumption'],
     'Categories1': ['lighting'], 'Categories2': ['electricity']})

  dm_elec_light = dm_house_elec_tot.copy()
  dm_elec_light.append(dm_house_light_tot, dim='Variables')
  dm_elec_light.append(dm_service_elec_tot.flattest(), dim='Variables')
  dm_elec_light.append(dm_service_light_tot.flattest(), dim='Variables')
  dm_elec_light.groupby({'bld_energy-consumption-elec_electricity': '.*'}, regex=True,
                          inplace=True, dim='Variables')
  dm_elec_light.deepen()
  dm_add_missing_variables(dm_elec_light, {'Categories1': dm_demand_trend.col_labels['Categories1']})
  dm_demand_trend.filter({'Variables': ['bld_energy-consumption']}, inplace=True)
  dm_demand_trend.append(dm_elec_light, dim='Variables')
  dm_demand_trend.groupby({'bld_energy-consumption': '.*'}, regex=True,
                          inplace=True, dim='Variables')

  return dm_demand_trend


def impose_capacity_constraints_pyomo(m, endyr, dm_capacity, country):
  # Overwrite ampl parameter to account for existing capacity and Nexus-e forecast capacity
  dm_CH = dm_capacity.filter({'Country': [country]})
  dm_CH.change_unit('pow_existing-capacity', old_unit='MW', new_unit='GW',
                    factor=1e-3)
  dm_CH.change_unit('pow_capacity-Pmax', old_unit='MW', new_unit='GW',
                    factor=1e-3)

  # fmax is sometimes in GW and sometimes in number of power plants
  # Hydro, PV, Wind existing it is in GW
  # Nuclear, Gas is in number of plants (Nuclear is 1 GW per plant, Gas is 0.5 GW)
  # Renewable energies + Nuclear and Gas
  existing_dam_cap = dm_CH[0, endyr, 'pow_existing-capacity', 'Dam']
  existing_ror_cap = dm_CH[0, endyr, 'pow_existing-capacity', 'RoR']
  for param_name in ['f_min', 'f_max', 'ref_size']:
    param = getattr(m, param_name)
    param['HYDRO_DAM'] = existing_dam_cap
    param['HYDRO_RIVER'] = existing_ror_cap

  category_mapping = {'CCGT': ['GasCC'], 'CCGT_CCS': ['GasCC-CCS'],
                      'NUCLEAR': ['Nuclear'],
                      'PV': ['PV-roof'], 'WIND': ['WindOn'],
                      'NEW_HYDRO_DAM': ['Dam'],
                      'NEW_HYDRO_RIVER': ['RoR'],
                      'PUMPED_HYDRO': ['Pump-Open'], 'POWER2GAS': ['GasCC-Syn'],
                      'OIL': ['Oil'], 'WASTE': ['Waste']}
  dm_CH.groupby(category_mapping, dim='Categories1', inplace=True)

  for renewable in ['WIND', 'PV']:
    existing_cap = dm_CH[0, endyr, 'pow_existing-capacity', renewable]
    max_cap = dm_CH[0, endyr, 'pow_capacity-Pmax', renewable]
    m.f_min[renewable] = existing_cap
    m.f_max[renewable] = max_cap
    m.fmax_perc[renewable] = 0.3

  max_cap = dm_CH[0, endyr, 'pow_capacity-Pmax', 'NEW_HYDRO_RIVER']
  m.f_min['NEW_HYDRO_RIVER'] = 0
  m.f_max['NEW_HYDRO_RIVER'] = max_cap - existing_ror_cap

  max_cap = dm_CH[0, endyr, 'pow_capacity-Pmax', 'NEW_HYDRO_DAM']
  m.f_min['NEW_HYDRO_DAM'] = 0
  #m.f_max['NEW_HYDRO_DAM'] = max_cap - existing_dam_cap

  for non_ren in ['NUCLEAR', 'CCGT', 'CCGT_CCS']:
    ref_size = pyo.value(m.ref_size[non_ren])
    existing_cap = dm_CH[0, endyr, 'pow_existing-capacity', non_ren] / ref_size
    max_cap = dm_CH[0, endyr, 'pow_capacity-Pmax', non_ren] / ref_size
    m.f_min[non_ren] = existing_cap
    m.f_max[non_ren] = max_cap

  # For hydro historical efficiency and forecasted efficiency do not match
  # The adjusting factors I am using here are chosen so that the final efficiency
  # (capacity factor) look coherent
  for period in m.PERIODS:
    m.c_p_t["HYDRO_DAM", period] = 1.3 * m.c_p_t["HYDRO_DAM", period]
    m.c_p_t["NEW_HYDRO_DAM", period] =  1.3 * m.c_p_t["NEW_HYDRO_DAM", period]

  for period in m.PERIODS:
    m.c_p_t["HYDRO_RIVER", period] = 1.4 * m.c_p_t["HYDRO_RIVER", period]
    m.c_p_t["NEW_HYDRO_RIVER", period] =  1.4 * m.c_p_t["NEW_HYDRO_RIVER", period]

  return


def impose_capacity_constraints(ampl, endyr, dm_capacity, country):
  # Overwrite ampl parameter to account for existing capacity and Nexus-e forecast capacity
  dm_CH = dm_capacity.filter({'Country': [country]})
  dm_CH.change_unit('pow_existing-capacity', old_unit='MW', new_unit='GW',
                    factor=1e-3)
  dm_CH.change_unit('pow_capacity-Pmax', old_unit='MW', new_unit='GW',
                    factor=1e-3)

  # fmax is sometimes in GW and sometimes in number of power plants
  # Hydro, PV, Wind existing it is in GW
  # Nuclear, Gas is in number of plants (Nuclear is 1 GW per plant, Gas is 0.5 GW)
  # Renewable energies + Nuclear and Gas
  existing_dam_cap = dm_CH[0, endyr, 'pow_existing-capacity', 'Dam']
  existing_ror_cap = dm_CH[0, endyr, 'pow_existing-capacity', 'RoR']
  for param in ['f_min', 'f_max', 'ref_size']:
    ampl.getParameter(param).setValues({'HYDRO_DAM': existing_dam_cap})
    ampl.getParameter(param).setValues({'HYDRO_RIVER': existing_ror_cap})

  category_mapping = {'CCGT': ['GasCC'], 'CCGT_CCS': ['GasCC-CCS'],
                      'NUCLEAR': ['Nuclear'],
                      'PV': ['PV-roof'], 'WIND': ['WindOn'],
                      'NEW_HYDRO_DAM': ['Dam'],
                      'NEW_HYDRO_RIVER': ['RoR'],
                      'PUMPED_HYDRO': ['Pump-Open'], 'POWER2GAS': ['GasCC-Syn'],
                      'OIL': ['Oil'], 'WASTE': ['Waste']}
  dm_CH.groupby(category_mapping, dim='Categories1', inplace=True)
  for renewable in ['WIND', 'PV']:
    existing_cap = dm_CH[0, endyr, 'pow_existing-capacity', renewable]
    max_cap = dm_CH[0, endyr, 'pow_capacity-Pmax', renewable]
    ampl.getParameter('f_min').setValues({renewable: existing_cap})
    ampl.getParameter('f_max').setValues({renewable: max_cap})
    ampl.getParameter('fmax_perc').setValues({renewable: 0.3})

  for renewable in ['NEW_HYDRO_RIVER', 'NEW_HYDRO_DAM']:
    max_cap = dm_CH[0, endyr, 'pow_capacity-Pmax', renewable]
    ampl.getParameter('f_min').setValues({renewable: 0})
    ampl.getParameter('f_max').setValues({renewable: max_cap})

  for non_ren in ['NUCLEAR', 'CCGT', 'CCGT_CCS']:
    ref_size = ampl.getParameter('ref_size').get(non_ren)
    existing_cap = dm_CH[0, endyr, 'pow_existing-capacity', non_ren] / ref_size
    max_cap = dm_CH[0, endyr, 'pow_capacity-Pmax', non_ren] / ref_size
    ampl.getParameter('f_min').setValues({non_ren: existing_cap})
    ampl.getParameter('f_max').setValues({non_ren: max_cap})

  return


def extract_industry_demand(DM_ind):
  dm_demand_trend = DM_ind['ind-energy-demand'].filter({'Categories1': ['elec', 'lighting', 'process-heat'], 'Categories2': ['electricity']})
  dm_demand_trend.group_all('Categories1', inplace=True)
  dm_demand_trend.add(0, dim='Categories1', col_label=['heat-pump', 'district-heating'], dummy=True)
  return dm_demand_trend


def extract_agriculture_demand(DM_agr):
  # AGRICULTURE electricity demand
  dm_agr_demand_trend = DM_agr['power'].filter({'Categories1': ['electricity', 'district-heating']})
  dm_agr_demand_trend.add(0, dim='Categories1', col_label='heat-pump', dummy=True)
  return dm_agr_demand_trend


def impose_industry_demand_pyomo(m, endyr, share_of_pop, DM_ind, DM_agr, cntr):
  eps = 1e-5

  # Prepare demand trend for post-processing energy-scope result
  dm_demand_trend = DM_ind['ind-energy-demand'].filter({'Categories1': ['elec', 'lighting', 'process-heat'], 'Categories2': ['electricity']})
  dm_demand_trend.group_all('Categories1', inplace=True)
  dm_demand_trend.add(0, dim='Categories1', col_label=['heat-pump', 'district-heating'], dummy=True)

  # AGRICULTURE electricity demand
  dm_agr_demand_trend = DM_agr['power'].filter({'Categories1': ['electricity', 'district-heating']})
  dm_agr_demand_trend.add(0, dim='Categories1', col_label='heat-pump', dummy=True)

  tot_agr_elec = dm_agr_demand_trend[cntr, endyr, 'agr_energy-consumption', 'electricity'] / share_of_pop*1000

  # ELECTRICITY (industry + agriculture)
  dm_ind_elec_tot = DM_ind['ind-energy-demand'].filter({'Categories1': ['elec'], 'Categories2': ['electricity']})
  tot_ind_elec = dm_ind_elec_tot[cntr, endyr, 'ind_energy-end-use', 'elec', 'electricity'] / share_of_pop*1000
  m.end_uses_demand_year['ELECTRICITY', 'INDUSTRY'] = tot_ind_elec + tot_agr_elec

  # LIGHTING
  dm_ind_light_tot = DM_ind['ind-energy-demand'].filter({'Categories1': ['lighting'], 'Categories2': ['electricity']})
  tot_ind_light = dm_ind_light_tot[cntr, endyr, 'ind_energy-end-use', 'lighting', 'electricity'] / share_of_pop*1000
  m.end_uses_demand_year['LIGHTING', 'INDUSTRY'] = tot_ind_light

  m.end_uses_demand_year['ELECTRICITY', "INDUSTRY"] = tot_ind_elec
  m.end_uses_demand_year['LIGHTING', "INDUSTRY"] = tot_ind_light

  # HIGH TEMPERATURE HEAT
  dm_high_heat = DM_ind['ind-energy-demand'].filter({'Categories1': ['process-heat'],
                                                     'Categories2': ['electricity']})
  tot_high_heat = dm_high_heat[cntr, endyr, 'ind_energy-end-use', 'process-heat', 'electricity']/share_of_pop*1000
  m.end_uses_demand_year['HEAT_HIGH_T', "INDUSTRY"] = tot_high_heat

  dm_high_heat.group_all('Categories1', inplace=True)
  dm_high_heat.normalise(dim='Categories1', inplace=True, keep_original=True)
  dm_high_heat.rename_col(['electricity'], ['IND_DIRECT_ELEC'], dim='Categories1')
  decen_heat = list(m.TECHNOLOGIES_OF_END_USES_TYPE["HEAT_HIGH_T"])
  for cat in decen_heat:
    if cat in dm_high_heat.col_labels['Categories1']:
      val_perc = dm_high_heat[cntr, endyr, 'ind_energy-end-use_share', cat]
      val_abs = dm_high_heat[cntr, endyr, 'ind_energy-end-use', cat] / share_of_pop
    else:
      val_perc = 0
      val_abs = 0
    m.fmin_perc[cat] = max(val_perc * (1 - eps), 0)
    m.fmax_perc[cat] = min(val_perc * (1 + eps), 1)
    m.f_min[cat] = val_abs * (1 - eps)
    m.f_max[cat] = val_abs * (1 + eps)

  return dm_demand_trend, dm_agr_demand_trend


def prepare_TPE_output(dm_prod_cap_cntr, dm_demand):

  dm_out = dm_prod_cap_cntr.filter({'Variables': ['pow_production', 'pow_capacity']})
  dm_out.groupby({'Gas': ['GasCC', 'GasCC-Syn', 'GasSC']}, dim='Categories1')
  dm_out.rename_col('Switzerland', 'Vaud', dim='Country')
  dm_out = dm_out.flattest()

  # Energy demand by sector
  dm_tmp = dm_demand.flattest()
  dm_out.append(dm_tmp, dim='Variables')

  return dm_out
