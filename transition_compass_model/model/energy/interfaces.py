import numpy as np


def impose_transport_demand(ampl, endyr, share_pop, DM_tra, cntr):
  eps = 1e-5
  # Set public transport pkm share
  DM_tra['passenger'].change_unit('tra_passenger_transport-demand',
                                  old_unit='pkm', new_unit='Mpkm', factor=1e-6)
  DM_tra['passenger'].filter({'Categories2': ['BEV', 'CEV',  'mt']}, inplace=True) # 'PHEV-diesel', 'PHEV-gasoline',
  dm_pass_demand = DM_tra['passenger'].filter(
    {'Years': [endyr], 'Variables': ['tra_passenger_transport-demand']})
  dm_pub_pri = dm_pass_demand.group_all('Categories2', inplace=False)
  dm_pub_pri.groupby(
    {'public': ['metrotram', 'bus', 'rail'], 'private': ['2W', 'LDV']},
    dim='Categories1', inplace=True)
  dm_pub_pri.normalise('Categories1', inplace=True)
  ampl.getParameter("share_mobility_public_min").setValues(
    [dm_pub_pri[cntr, 0, 0, 'public'] - eps])
  ampl.getParameter("share_mobility_public_max").setValues(
    [dm_pub_pri[cntr, 0, 0, 'public'] + eps])

  # Set rail transport in freight share
  DM_tra['freight'].filter({'Categories2': ['BEV', 'CEV']}, inplace=True) # 'PHEV-diesel', 'PHEV-gasoline'
  dm_freight_demand = DM_tra['freight'].filter(
    {'Years': [endyr], 'Variables': ['tra_freight_transport-demand-tkm']})
  dm_rail_freight = dm_freight_demand.group_all('Categories2', inplace=False)
  dm_rail_freight.normalise('Categories1', inplace=True)
  ampl.getParameter("share_freight_train_min").setValues(
    [0.99- eps])
  ampl.getParameter("share_freight_train_max").setValues(
    [1])

  # Set total Passenger mobility in pkm
  tot_Mpkm = np.nansum(
    dm_pass_demand[0, 0, 'tra_passenger_transport-demand', :, :],
    axis=(-1, -2)) / share_pop
  ampl.getParameter("end_uses_demand_year").setValues(
    {("MOBILITY_PASSENGER", "TRANSPORTATION"): tot_Mpkm})

  # Set total Freight mobility in pkm
  DM_tra['freight'].change_unit('tra_freight_transport-demand-tkm',
                                old_unit='tkm', new_unit='Mtkm', factor=1e-6)
  tot_Mtkm = np.nansum(
    DM_tra['freight'][0, 0, 'tra_freight_transport-demand-tkm', :, :],
    axis=(-1, -2)) / share_pop
  ampl.getParameter("end_uses_demand_year").setValues(
    {("MOBILITY_FREIGHT", "TRANSPORTATION"): tot_Mtkm})

  # Passenger private technology share
  # It should be both 2W and LDV according to the Calculator, but energyscope only has LDV
  dm_private = DM_tra['passenger'].filter({'Categories1': ['LDV']})
  dm_private.normalise(dim='Categories2', inplace=True, keep_original=True)
  dm_private.groupby({'CAR_BEV': ['BEV']},
                     dim='Categories2', inplace=True)
  dm_private.filter_w_regex({'Categories2': 'CAR.*'}, inplace=True)
  dm_private = dm_private.flatten()
  dm_private.rename_col_regex('LDV_', '', dim='Categories1')
  private_mob_tech = list(ampl.get_set("TECHNOLOGIES_OF_END_USES_TYPE").get("MOB_PRIVATE"))
  for cat in private_mob_tech:
    if cat in dm_private.col_labels['Categories1']:
      val_perc = dm_private[0, endyr, 'tra_passenger_transport-demand_share', cat]
      val_abs = dm_private[
                  0, endyr, 'tra_passenger_transport-demand', cat] / share_pop
    else:  # If it is not electricity technology, set shares to 0
      val_perc = 0
      val_abs = 0
    ampl.getParameter("fmin_perc").setValues(
        {cat: max(val_perc * (1 - eps), 0)})
    ampl.getParameter("fmax_perc").setValues({cat: min(val_perc * (1 + eps), 1)})
    ampl.getParameter("f_min").setValues({cat: 0})
    ampl.getParameter("f_max").setValues({cat: val_abs * (1 + eps)})

  # Passenger public technology share
  dm_public = DM_tra['passenger'].filter(
    {'Categories1': ['metrotram', 'bus', 'rail']})
  dm_public = dm_public.flatten()
  dm_public.groupby(
    {'TRAIN_PUB': ['rail_CEV'],
     'TRAMWAY_TROLLEY': ['metrotram_mt', 'bus_CEV']}, dim='Categories1',
    inplace=True)
  dm_public.normalise('Categories1', inplace=True, keep_original=True)
  public_mob_tech = list(ampl.get_set("TECHNOLOGIES_OF_END_USES_TYPE").get("MOB_PUBLIC"))
  for cat in public_mob_tech:
    if cat in ['TRAIN_PUB', 'TRAMWAY_TROLLEY']:
      val_perc = dm_public[0, endyr, 'tra_passenger_transport-demand_share', cat]
      val_abs = dm_public[
                  0, endyr, 'tra_passenger_transport-demand', cat] / share_pop
    else:
      val_perc = 0
      val_abs = 0
    ampl.getParameter("fmin_perc").setValues({cat: max(val_perc * (1 - eps), 0)})
    ampl.getParameter("fmax_perc").setValues({cat: min(val_perc * (1 + eps), 1)})
    ampl.getParameter("f_min").setValues({cat: 0})
    ampl.getParameter("f_max").setValues({cat: val_abs * (1 + eps)})

  # Efficiency - Passenger
  mapping = {'TRAMWAY_TROLLEY': 'ELECTRICITY',
             'TRAIN_PUB': 'ELECTRICITY',
             'CAR_BEV': 'ELECTRICITY'}
  dm_eff = dm_private.copy()
  dm_eff.append(dm_public, dim='Categories1')
  for veh, fuel in mapping.items():
    val = dm_eff[0, endyr, 'tra_passenger_energy-intensity', veh]
    ampl.getParameter("layers_in_out").setValues({(veh, fuel): -val})

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
    ampl.getParameter("layers_in_out").setValues({(veh, fuel): -val})

  return


def impose_buildings_demand(ampl, endyr, share_of_pop, DM_bld, cntr):
  eps = 1e-5
  # Set district-heating share
  #DM_bld[:, :, 'bld_heating', ...] = DM_bld[:, :, 'bld_heating', ...]
  #DM_bld[:, :, 'bld_energy-demand_heating', ...] = DM_bld[:, :,'bld_energy-demand_heating',...]

  DM_bld.filter({'Categories1': ['electricity', 'heat-pump', 'district-heating']}, inplace=True)
  DM_bld.change_unit('bld_heating', old_unit='TWh', new_unit='GWh', factor=1000)
  DM_bld.change_unit('bld_energy-demand_heating', old_unit='TWh',
                     new_unit='GWh', factor=1000)
  # Useful energy demand
  dm_heating = DM_bld.filter({'Variables': ['bld_heating'], 'Years': [endyr]})
  dm_heating.normalise(dim='Categories1', inplace=True, keep_original=True)
  val = dm_heating[cntr, endyr, 'bld_heating_share', 'district-heating']
  ampl.getParameter("share_heat_dhn_min").setValues([val])
  ampl.getParameter("share_heat_dhn_max").setValues([val])

  # Set heating demand in GWh
  # !FIXME Change here the services heat demand
  dm_tot = dm_heating.group_all('Categories1', inplace=False)
  tot_heat = dm_tot[:, :, 'bld_heating'] / share_of_pop
  ampl.getParameter("end_uses_demand_year").setValues(
    {('HEAT_LOW_T_SH', "HOUSEHOLDS"): tot_heat})
  ampl.getParameter("end_uses_demand_year").setValues(
    {('HEAT_LOW_T_SH', "SERVICES"): 0})
  # ampl.getParameter("end_uses_demand_year").setValues({('HEAT_LOW_T_SH', "INDUSTRY"): 0})
  ampl.getParameter("end_uses_demand_year").setValues(
    {('HEAT_LOW_T_HW', 'HOUSEHOLDS'): 0})
  ampl.getParameter("end_uses_demand_year").setValues(
    {('HEAT_LOW_T_HW', 'SERVICES'): 0})
  # ampl.getParameter("end_uses_demand_year").setValues({('HEAT_LOW_T_HW', 'INDUSTRY'): 0})

  # Set decentralised heating shares by technology (- district heating)
  # DEC_HP_ELEC	DEC_THHP_GAS	DEC_COGEN_GAS	DEC_COGEN_OIL	DEC_ADVCOGEN_GAS
  # DEC_ADVCOGEN_H2	DEC_BOILER_GAS	DEC_BOILER_WOOD	DEC_BOILER_OIL	DEC_SOLAR	DEC_DIRECT_ELEC
  dm_heating = DM_bld.filter(
    {'Variables': ['bld_heating', 'bld_energy-demand_heating']})
  dm_heating.drop(dim='Categories1', col_label='district-heating')
  dm_heating.normalise(dim='Categories1', inplace=True, keep_original=True)
  dm_heating.rename_col(
    ['heat-pump', 'electricity'],
    ['DEC_HP_ELEC', 'DEC_DIRECT_ELEC'],
    dim='Categories1')
  decen_heat = list(ampl.get_set("TECHNOLOGIES_OF_END_USES_TYPE").get("HEAT_LOW_T_DECEN"))
  for cat in decen_heat:
    if cat in dm_heating.col_labels['Categories1']:
      val_perc = dm_heating[cntr, endyr, 'bld_heating_share', cat]
      val_abs = dm_heating[cntr, endyr, 'bld_heating', cat] / share_of_pop
    else:
      val_perc = 0
      val_abs = 0
    ampl.getParameter("fmin_perc").setValues({cat: max(val_perc * (1 - eps), 0)})
    ampl.getParameter("fmax_perc").setValues({cat: min(val_perc * (1 + eps), 1)})
    ampl.getParameter("f_min").setValues({cat: val_abs * (1 - eps)})
    ampl.getParameter("f_max").setValues({cat: val_abs * (1 + eps)})


  # Set efficiencies
  mapping = {'DEC_HP_ELEC': 'ELECTRICITY',
             'DEC_DIRECT_ELEC': 'ELECTRICITY'}
  # mapping = {'DEC_HP_ELEC': 'ELECTRICITY', 'DEC_BOILER_GAS': 'NG', 'DEC_BOILER_OIL': 'LFO',
  #           'DEC_DIRECT_ELEC': 'ELECTRICITY'}
  # !FIXME : there is a problem here when bld_heating is 0!
  dm_heating.operation('bld_energy-demand_heating', '/', 'bld_heating',
                       out_col='bld_rev_eff', unit='%')
  dm_heating.fill_nans('Years')
  # dm_heating[cntr, endyr, 'bld_rev_eff', 'DEC_BOILER_WOOD'] = dm_heating[cntr, endyr, 'bld_rev_eff', 'DEC_BOILER_OIL']
  for veh, fuel in mapping.items():
    val = dm_heating[cntr, endyr, 'bld_rev_eff', veh]
    ampl.getParameter("layers_in_out").setValues({(veh, fuel): -val})

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
