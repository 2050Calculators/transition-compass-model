###################################
#####  VEHICLE EFFICIENCY LDV  ####
###################################

import pickle
import os
import numpy as np
import _database.pre_processing.transport.Switzerland.get_data_functions.efficiency as get_data
from model.common.auxiliary_functions import dm_add_missing_variables, \
  linear_fitting


def convert_eff_from_gCO2_km_to_MJ_km(dm_veh_eff_LDV, cdm_emissions_factors, new_var_name):
    dm_veh_eff_LDV.drop('Categories2', ['BEV', 'FCEV'])
    var_name = dm_veh_eff_LDV.col_labels['Variables'][0]
    dm_veh_eff_LDV.rename_col(var_name, 'tmp_name', dim='Variables')
    cdm_emissions_CO2_LDV = cdm_emissions_factors.filter({'Categories1': ['CO2'],
                                                          'Categories2': dm_veh_eff_LDV.col_labels['Categories2']})
    cdm_emissions_CO2_LDV.sort('Categories2')
    dm_veh_eff_LDV.sort('Categories2')
    # I want to have an efficiency in MJ/km -> then I have to do  veh-eff(g/km) / emission-fact(g/MJ) = eff (MJ/km)
    arr_eff_MJ_km = dm_veh_eff_LDV.array / cdm_emissions_CO2_LDV.array[np.newaxis, :, :, :]
    dm_veh_eff_LDV.add(arr_eff_MJ_km, dim='Variables', col_label=new_var_name, unit='MJ/km')
    dm_veh_eff_LDV.filter({'Variables': [new_var_name]}, inplace=True)
    return dm_veh_eff_LDV


def replace_LDV_efficiency_with_new(dm_veh_eff, dm_veh_new_eff, dm_veh_eff_LDV, dm_veh_new_eff_LDV, baseyear_old):
    # Vaud efficiency = Swiss efficiency
    dm_veh_eff_VD = dm_veh_eff.copy()
    dm_veh_eff_VD.rename_col('Switzerland', 'Vaud', dim='Country')
    dm_veh_eff.append(dm_veh_eff_VD, dim='Country')
    # Vaud efficiency = Swiss efficiency
    dm_veh_new_eff_VD = dm_veh_new_eff.copy()
    dm_veh_new_eff_VD.rename_col('Switzerland', 'Vaud', dim='Country')
    dm_veh_new_eff.append(dm_veh_new_eff_VD, dim='Country')

    # Remove fts years and add ots years missing
    years_match = [y for y in dm_veh_eff.col_labels['Years'] if y <= baseyear_old]
    years_missing = list(set(dm_veh_eff_LDV.col_labels['Years']) - set(years_match))
    dm_veh_eff.filter({'Years': years_match}, inplace=True)
    dm_veh_new_eff.filter({'Years': years_match}, inplace=True)
    dm_veh_eff.add(np.nan, dummy=True, dim='Years', col_label=years_missing)
    dm_veh_new_eff.add(np.nan, dummy=True, dim='Years', col_label=years_missing)

    # Rename 2W_PHEV as 2W_PHEV-diesel and 2W_PHEV-gasoline
    DM = {'veh-eff': dm_veh_eff, 'veh-eff-new': dm_veh_new_eff}
    for key, dm in DM.items():
        dm = dm.flatten()
        if '2W_PHEV' in dm.col_labels['Categories1']:
            dm_tmp = dm.filter({'Categories1': ['2W_PHEV']})
            dm.rename_col('2W_PHEV', '2W_PHEV-diesel', dim='Categories1')
            dm_tmp.rename_col('2W_PHEV', '2W_PHEV-gasoline', dim='Categories1')
            dm.append(dm_tmp, dim='Categories1')
            dm.deepen()
            DM[key] = dm

    dm_veh_eff = DM['veh-eff']
    dm_veh_new_eff = DM['veh-eff-new']

    idx_t = dm_veh_eff.idx
    idx_l = dm_veh_eff_LDV.idx
    for cat in dm_veh_eff_LDV.col_labels['Categories2']:
        dm_veh_eff.array[:, :, :, idx_t['LDV'], idx_t[cat]] = dm_veh_eff_LDV.array[:, :, :, idx_l['LDV'], idx_l[cat]]
        dm_veh_new_eff.array[:, :, :, idx_t['LDV'], idx_t[cat]] = dm_veh_new_eff_LDV.array[:, :, :, idx_l['LDV'], idx_l[cat]]

    dm_veh_eff.fill_nans(dim_to_interp='Years')
    dm_veh_new_eff.fill_nans(dim_to_interp='Years')
    # Remove aviation
    dm_veh_eff.drop(dim='Categories1', col_label='aviation')
    dm_veh_new_eff.drop(dim='Categories1', col_label='aviation')
    drop_cat = ['PHEV', 'ICE']
    for cat in drop_cat:
        if cat in dm_veh_eff.col_labels['Categories2']:
            dm_veh_eff.drop(col_label=[cat], dim='Categories2')
            dm_veh_new_eff.drop(col_label=[cat], dim='Categories2')

    return dm_veh_eff, dm_veh_new_eff

def compute_tech_share(dm_private_fleet, dm_public_fleet):
  # All technologies
  dm_private = dm_private_fleet.copy()
  dm_public = dm_public_fleet.copy()
  cat_tech = set(dm_private_fleet.col_labels['Categories2']).union(set(dm_public_fleet.col_labels['Categories2']))
  # Join private and public
  dm_add_missing_variables(dm_public, {'Categories2': cat_tech})
  dm_add_missing_variables(dm_private, {'Categories2': cat_tech})
  dm_private.append(dm_public, dim='Categories1')
  # Technology share
  dm_tech_share = dm_private.normalise('Categories2', inplace=False)

  return dm_tech_share


def compute_vehicle_efficiency_from_energy_demand(dm_energy, dm_vkm, dm_private_fleet, dm_public_fleet):
  # Drop bio-fuels energy
  dm_pass_energy = dm_energy.copy()
  dm_pass_energy.groupby(
    {'ICE-gasoline': ['biogasoline', 'gasoline']}, dim='Categories2', inplace=True)

  # Compute Demand by technology = Demand(vkm) * tech_share (%)
  dm_fleet = dm_private_fleet.filter({'Variables': ['tra_passenger_vehicle-fleet']})
  ## join private and public fleet
  dm_tmp = dm_public_fleet.copy()
  ### COMPUTE TECH SHARE
  dm_tech_share = compute_tech_share(dm_private_fleet=dm_fleet, dm_public_fleet=dm_tmp)
  ## demand by tech
  arr_demand_tech = dm_vkm[..., np.newaxis] * dm_tech_share[...]
  dm_tech_share.add(arr_demand_tech, dim='Variables', col_label='tra_passenger_demand-vkm', unit='vkm')

  # efficiency = energy/demand
  dm_demand_tech = dm_tech_share.filter({'Variables': ['tra_passenger_demand-vkm'],
                                         'Categories1': dm_pass_energy.col_labels['Categories1'],
                                         'Categories2': dm_pass_energy.col_labels['Categories2'],
                                         'Country': dm_pass_energy.col_labels['Country']})

  dm_demand_tech.drop(col_label='LDV', dim='Categories1')
  dm_pass_energy.drop(col_label='LDV', dim='Categories1')
  dm_demand_tech.append(dm_pass_energy, dim='Variables')
  dm_demand_tech.change_unit('tra_energy_demand', old_unit='TWh', new_unit='MJ', factor=3.6*1e9)
  dm_demand_tech.operation('tra_energy_demand', '/', 'tra_passenger_demand-vkm', out_col='tra_passenger_veh-efficiency_fleet', unit='MJ/km')

  dm_veh_eff = dm_demand_tech.filter({'Variables': ['tra_passenger_veh-efficiency_fleet']})

  return dm_veh_eff

def add_metrotram_efficiency_from_JRC(dm_veh_eff):
  # JRC-IDEES-2021, TrRail_ene sheet (Vehicle-efficiency kgoe/100 km)
  # metrotram values go from 45 kgoe/100km in 2000 to 32.2 kgoe/100km  in 2021, roughly linearly
  var_name = 'tra_passenger_veh-efficiency_fleet'
  dm_veh_eff.add(np.nan, col_label='metrotram', dim='Categories1', dummy=True)
  dm_veh_eff.add(np.nan, col_label='mt', dim='Categories2', dummy=True)
  dm_veh_eff['Switzerland', 2000, var_name, 'metrotram', 'mt'] = 45 * 41.868 / 100
  dm_veh_eff['Switzerland', 2021, var_name, 'metrotram', 'mt'] = 32.2 * 41.868 / 100
  dm_metrotram = dm_veh_eff.filter({'Categories1': ['metrotram'], 'Categories2': ['mt']})
  years_ots = dm_metrotram.col_labels['Years']
  linear_fitting(dm_metrotram, years_ots=years_ots)
  dm_veh_eff['Switzerland', :, var_name, 'metrotram', 'mt'] \
    = dm_metrotram['Switzerland', :, var_name, 'metrotram', 'mt']
  return dm_veh_eff

def run(dm_energy, dm_vkm, dm_private_fleet, dm_public_fleet, cdm_emissions_factors, years_ots):

  this_dir = os.path.dirname(os.path.abspath(__file__))  # creates local path variable


  ###############################
  ####  Efficiency for LDV   ####
  ###############################
  # SECTION Vehicle efficiency LDV, stock and new, ots
  #### Vehicle efficiency - LDV - CO2/km
  # FCEV (Hydrogen) data are off - BEV too
  # !!! Attention: The data are bad before 2016 and after 2020, backcasting to 1990 from 2016 done with linear fitting.
  table_id_veh_eff = 'px-x-1103020100_106'
  local_filename_veh = os.path.join(this_dir, '../data/tra_veh_efficiency.pickle')  # The file is created if it doesn't exist
  dm_veh_eff_LDV = get_data.get_vehicle_efficiency(table_id_veh_eff, local_filename_veh,
                                          var_name='tra_passenger_veh-efficiency_fleet', years_ots=years_ots)
  del table_id_veh_eff, local_filename_veh

  #### Vehicle efficiency new - LDV - CO2/km
  # FCEV data are off, BEV = 25 gCO2/km independently of car power
  table_id_new_eff = 'px-x-1103020200_201'
  local_filename_new = os.path.join(this_dir, '../data/tra_new-veh_efficiency.pickle')  # The file is created if it doesn't exist3#
  dm_veh_new_eff_LDV = get_data.get_new_vehicle_efficiency(table_id_new_eff, local_filename_new,
                                                  var_name='tra_passenger_veh-efficiency_new', years_ots=years_ots)
  del table_id_new_eff, local_filename_new

  # The Swiss efficiency for the fleet is given in gCO2/km. We convert it to MJ/km
  dm_veh_eff_LDV = convert_eff_from_gCO2_km_to_MJ_km(dm_veh_eff_LDV, cdm_emissions_factors,
                                                     new_var_name='tra_passenger_veh-efficiency_fleet')
  dm_veh_new_eff_LDV = convert_eff_from_gCO2_km_to_MJ_km(dm_veh_new_eff_LDV, cdm_emissions_factors,
                                                         new_var_name='tra_passenger_veh-efficiency_new')

  ################################################
  ####  Efficiency for modes other than LDV   ####
  ################################################

  # Efficiency = MJ/vkm = Energy(MJ)/Demand(vkm)
  dm_veh_eff = compute_vehicle_efficiency_from_energy_demand(dm_energy, dm_vkm, dm_private_fleet, dm_public_fleet)
  # Add metrotram efficiency from JRC
  dm_veh_eff = add_metrotram_efficiency_from_JRC(dm_veh_eff)

  # new-veh-eff = veh-eff & vaud  = swiss efficiency
  dm_veh_eff.add(np.nan, dim='Variables', col_label = 'tra_passenger_veh-efficiency_new', unit='MJ/km', dummy=True)
  dm_veh_eff.add(np.nan, dim='Country', col_label = 'Vaud', dummy=True)
  dm_veh_eff.fill_nans(dim_to_interp='Variables')
  dm_veh_eff.fill_nans(dim_to_interp='Country')
  dm_veh_eff.fill_nans('Years')

  # If it is all 0 for all years, replace with nans
  arr = dm_veh_eff.array
  mask = (arr == 0).all(axis=1, keepdims=True)
  dm_veh_eff.array = np.where(mask, np.nan, arr)

  # Join LDV and other modes
  all_tech = set(dm_veh_eff_LDV.col_labels['Categories2']).union(set(dm_veh_eff.col_labels['Categories2']))
  dm_veh_eff_LDV.append(dm_veh_new_eff_LDV, dim='Variables')
  dm_add_missing_variables(dm_veh_eff_LDV,{'Categories2': all_tech})
  dm_add_missing_variables(dm_veh_eff,{'Categories2': all_tech})
  dm_veh_eff.append(dm_veh_eff_LDV.filter({'Years': years_ots}), dim='Categories1')


  return dm_veh_eff


def run_old(cdm_emissions_factors, years_ots):

  this_dir = os.path.dirname(os.path.abspath(__file__))  # creates local path variable

  # SECTION Vehicle efficiency LDV, stock and new, ots
  #### Vehicle efficiency - LDV - CO2/km
  # FCEV (Hydrogen) data are off - BEV too
  # !!! Attention: The data are bad before 2016 and after 2020, backcasting to 1990 from 2016 done with linear fitting.
  table_id_veh_eff = 'px-x-1103020100_106'
  local_filename_veh = os.path.join(this_dir, '../data/tra_veh_efficiency.pickle')  # The file is created if it doesn't exist
  dm_veh_eff_LDV = get_data.get_vehicle_efficiency(table_id_veh_eff, local_filename_veh,
                                          var_name='tra_passenger_veh-efficiency_fleet', years_ots=years_ots)
  del table_id_veh_eff, local_filename_veh

  #### Vehicle efficiency new - LDV - CO2/km
  # FCEV data are off, BEV = 25 gCO2/km independently of car power
  table_id_new_eff = 'px-x-1103020200_201'
  local_filename_new = os.path.join(this_dir, '../data/tra_new-veh_efficiency.pickle')  # The file is created if it doesn't exist3#
  dm_veh_new_eff_LDV = get_data.get_new_vehicle_efficiency(table_id_new_eff, local_filename_new,
                                                  var_name='tra_passenger_veh-efficiency_new', years_ots=years_ots)
  del table_id_new_eff, local_filename_new

  # The Swiss efficiency for the fleet is given in gCO2/km. We convert it to MJ/km
  dm_veh_eff_LDV = convert_eff_from_gCO2_km_to_MJ_km(dm_veh_eff_LDV, cdm_emissions_factors,
                                                     new_var_name='tra_passenger_veh-efficiency_fleet')
  dm_veh_new_eff_LDV = convert_eff_from_gCO2_km_to_MJ_km(dm_veh_new_eff_LDV, cdm_emissions_factors,
                                                         new_var_name='tra_passenger_veh-efficiency_new')


  ### We should do a separate flow for aviation. from pkm/cap -> pkm -> technology share (applied to pkm) -> emissions/pkm
  data_file = '/Users/paruta/2050-Calculators/leure-speed-to-zero/backend/_database/data/datamatrix/transport.pickle'
  with open(data_file, 'rb') as handle:
      DM_transport = pickle.load(handle)

  lev = list(DM_transport['ots'].keys())[0]
  baseyear_old = DM_transport['ots'][lev].col_labels['Years'][-1]

  # Add to the LDV efficiency the efficiency of other means of transport, taken from EUCalc data
  print('You are missing vehicle efficiency for LDV FCEV and BEV, but also 2W, bus, aviation, metrotram, rail.'
        ' Bus efficiency looks very wrong')
  dm_veh_eff = DM_transport['fxa']['passenger_tech'].filter({'Variables': ['tra_passenger_veh-efficiency_fleet'],
                                                             'Country': ['Switzerland']})
  dm_veh_new_eff = DM_transport['ots']['passenger_veh-efficiency_new'].filter(
      {'Variables': ['tra_passenger_veh-efficiency_new'], 'Country': ['Switzerland']})

  # Adjust efficiency for metrotram and rail
  idx = dm_veh_eff.idx
  idx_n = dm_veh_new_eff.idx
  public_eff = {('metrotram', 'mt'): 3, ('rail', 'CEV'): 1, ('rail', 'ICE-diesel'): 1*0.8/0.3, ('rail', 'FCEV'): 1*0.8/0.3}
  for key, value in public_eff.items():
      dm_veh_eff.array[:, :, idx['tra_passenger_veh-efficiency_fleet'], idx[key[0]], idx[key[1]]] = value
      dm_veh_new_eff.array[:, :, idx_n['tra_passenger_veh-efficiency_new'], idx_n[key[0]], idx_n[key[1]]] = value

  # Determine efficiency
  # It also removes aviation
  dm_veh_eff, dm_veh_new_eff = replace_LDV_efficiency_with_new(dm_veh_eff, dm_veh_new_eff, dm_veh_eff_LDV, dm_veh_new_eff_LDV, baseyear_old)

  return dm_veh_eff, dm_veh_new_eff
