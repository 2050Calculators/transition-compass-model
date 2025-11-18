from model.energy.energyscopepyomo.ses_pyomo import load_data, build_model, \
  make_highs, attach, solve, extract_results, build_model_structure, set_constraints
import pyomo.environ as pyo
from model.common.interface_class import Interface
from model.common.data_matrix_class import DataMatrix
import os
from model.common.auxiliary_functions import filter_DM, create_years_list, \
  filter_geoscale, filter_country_and_load_data_from_pickles, \
  dm_add_missing_variables, return_lever_data
import pickle
import numpy as np
import model.energy.interfaces as inter
import model.energy.utils as utils
import re
import json



def capture_model_state(model, filename):
  """Capture all model parameters and their values"""
  state = {}

  # Capture all parameters
  state['parameters'] = {}
  for param in model.component_objects(pyo.Param):
    param_data = {}
    if param.is_indexed():
      param_data['values'] = {}
      for idx in param:
        try:
          # Use pyo.value() to handle expressions
          val = pyo.value(param[idx])
          param_data['values'][str(idx)] = val
        except:
          # If it still fails, store as string
          param_data['values'][str(idx)] = str(param[idx])
      param_data['count'] = len(param)
    else:
      try:
        param_data['values'] = pyo.value(
          param) if param.value is not None else None
      except:
        param_data['values'] = str(param)
      param_data['count'] = 1
    param_data['mutable'] = param.mutable
    state['parameters'][param.name] = param_data

  # Capture constraint info (not expressions, just structure)
  state['constraints'] = {}
  for con in model.component_objects(pyo.Constraint):
    con_data = {
      'count': len(con) if con.is_indexed() else 1,
      'active_count': sum(
        1 for idx in con if con[idx].active) if con.is_indexed() else (
        1 if con.active else 0)
    }
    state['constraints'][con.name] = con_data

  # Capture variable info
  state['variables'] = {}
  for var in model.component_objects(pyo.Var):
    var_data = {
      'count': len(var) if var.is_indexed() else 1
    }
    state['variables'][var.name] = var_data

  with open(filename, 'w') as f:
    json.dump(state, f, indent=2)

  return state

# Compare
def compare_states(before, after):
  """Find differences between two model states"""
  issues = []

  # Check parameters
  for param_name in before['parameters']:
    if param_name not in after['parameters']:
      issues.append(f"Parameter {param_name} disappeared!")
      continue

    before_param = before['parameters'][param_name]
    after_param = after['parameters'][param_name]

    if before_param['count'] != after_param['count']:
      issues.append(
        f"Parameter {param_name}: count changed from {before_param['count']} to {after_param['count']}")

    if before_param['mutable'] != after_param['mutable']:
      issues.append(
        f"Parameter {param_name}: mutable changed from {before_param['mutable']} to {after_param['mutable']}")

  # Check constraints
  for con_name in before['constraints']:
    if con_name not in after['constraints']:
      issues.append(f"Constraint {con_name} disappeared!")
      continue

    before_con = before['constraints'][con_name]
    after_con = after['constraints'][con_name]

    if before_con['count'] != after_con['count']:
      issues.append(
        f"Constraint {con_name}: count changed from {before_con['count']} to {after_con['count']}")

    if before_con['active_count'] != after_con['active_count']:
      issues.append(
        f"⚠️ Constraint {con_name}: ACTIVE count changed from {before_con['active_count']} to {after_con['active_count']}")

  return issues




def extract_sankey_energy_flow(DM):
  # Sankey structure
    # 	printf "%s,%s,%.2f,%s,%s,%s\n", "NG" , "Mob priv", sum{t in PERIODS}(-layers_in_out["CAR_NG","NG"] * F_Mult_t ["CAR_NG", t] * t_op [t]) / 1000 , "NG", "#FFD700", "TWh" >> "energyscope-MILP/output/sankey/input2sankey.csv";
  dm_eff = DM['efficiency']
  dm_hours = DM['hours_month']
  dm_operation = DM['monthly_operation_GW']
  dm_operation.sort('Categories1')
  dm_eff.sort('Categories1')
  common_cat = set(dm_operation.col_labels['Categories1']).intersection(dm_eff.col_labels['Categories1'])
  dm_operation.filter({'Categories1': common_cat}, inplace=True)
  dm_eff.filter({'Categories1': common_cat}, inplace=True)

  # Production capacity = Sum monthly operation capacity in GW x hours in a month h
  arr_prod_cap_yr = np.nansum(dm_operation.array * dm_hours.array[:, :, :, np.newaxis, :], axis=-1, keepdims=True)
  # Energy = yearly operation by effieciency
  arr_energy = dm_eff.array * arr_prod_cap_yr/1000
  dm_eff.add(arr_energy, dim='Variables', col_label='pow_production', unit='TWh')
  dm_energy_full = DataMatrix.based_on(arr_energy, dm_eff, change={'Variables': ['pow_production']},
                                  units={'pow_production': 'TWh'})
  # Energy production
  # Use the fact that the values are positive or negative to split
  cat1, cat2 = np.where(dm_energy_full.array[0, 0, 0, :, :]>10)
  new_arr = np.zeros((1, 1, 1, len(cat1)))
  new_arr[0, 0, 0, :] = dm_energy_full.array[0, 0, 0, cat1, cat2]
  new_categories = [dm_energy_full.col_labels['Categories1'][c1] + '-' +
                    dm_energy_full.col_labels['Categories2'][c2] for c1, c2 in zip(cat1, cat2)]
  dm_energy_prod = DataMatrix(col_labels={'Country': dm_eff.col_labels['Country'], 'Years': dm_eff.col_labels['Years'],
                                     'Variables': ['pow_production'], 'Categories1': new_categories},
                         units={'pow_production': 'TWh'})
  dm_energy_prod.array = new_arr

  # Power production
  #! FIXME: check this natural gas situation
  dm_power_prod = dm_energy_prod.filter_w_regex({'Categories1': '.*ELECTRICITYv2'})
  #if 'NG_CCS-NG_CCSv2' in dm_energy_prod.col_labels['Categories1']:
  #  dm_natural_gas_elec = dm_energy_prod.filter({'Categories1': ['NG_CCS-NG_CCSv2']})
  #  dm_power_prod.append(dm_natural_gas_elec, dim='Categories1')
  #if 'CCGT_CCS-ELECTRICITYv2' in dm_power_prod.col_labels['Categories1']:
  #  dm_power_prod.drop('Categories1', 'CCGT_CCS-ELECTRICITYv2')

  #cat_elec = dm_power_prod.col_labels['Categories1']
  dm_power_prod.deepen(sep='-')
  dm_power_prod.group_all('Categories2')

  # Energy production other than electricity
 # dm_energy_prod.drop(col_label=cat_elec, dim='Categories1')
 # dm_energy_prod.groupby({'oil-oiltmp': '.*OIL.*'}, inplace=True, regex=True, dim='Categories1')
 # col_to_drop = [col for col in dm_energy_prod.col_labels['Categories1'] if 'MOB_' in col or 'HEAT_' in col]
 # dm_energy_prod.drop('Categories1', col_to_drop)
 # dm_energy_prod.deepen(sep='-')
 # dm_energy_prod.group_all(dim='Categories2', inplace=True)

  # Energy consumption
  cat1, cat2 = np.where(dm_energy_full.array[0, 0, 0, :, :] < 0)
  new_arr = np.zeros((1, 1, 1, len(cat1)))
  new_arr[0, 0, 0, :] = - dm_energy_full.array[0, 0, 0, cat1, cat2]
  new_categories = [dm_energy_full.col_labels['Categories1'][c1] + '-' +
                    dm_energy_full.col_labels['Categories2'][c2] for c1, c2 in zip(cat1, cat2)]
  dm_energy_use = DataMatrix(col_labels={'Country': dm_eff.col_labels['Country'], 'Years': dm_eff.col_labels['Years'],
                                     'Variables': ['pow_production'], 'Categories1': new_categories},
                         units={'pow_production': 'TWh'})
  dm_energy_use.drop('Categories1', 'DIESEL-DIESELv2')
  dm_energy_use.array = new_arr
  col_to_drop = [col for col in new_categories if 'MOB_' in col or 'HEAT_' in col]
  dm_energy_use.drop('Categories1', col_to_drop)
  dm_energy_use.rename_col_regex('v2', '', dim='Categories1')

  dm_energy_use.deepen(sep='-')
  group_dict = {'passenger_LDV': 'CAR.*', 'passenger_bus': 'BUS.*', 'passenger_metrotram': 'TRAMWAY.*',
                'passenger_rail': 'TRAIN_PUB.*', 'freight_rail': 'TRAIN_FREIGHT.*',
                'freight_HDV': 'TRUCK.*', 'decentralised-heating': 'DEC_.*', 'district-heating': 'DHN_.*',
                'industrial-heat': 'IND_.*'}
  # Remove groups that are not in output
  to_remove = []
  for group, expr in group_dict.items():
    pattern = re.compile(expr)
    keep = [col for col in dm_energy_use.col_labels['Categories1'] if re.match(pattern, str(col))]
    if not keep:
      to_remove.append(group)

  for group in to_remove:
    group_dict.pop(group)

  dm_energy_use.groupby(group_dict, dim='Categories1', regex=True, inplace=True)

  for col in ['H2_ELECTROLYSIS', 'H2_NG']:
    if col in dm_energy_use.col_labels['Categories1']:
      dm_energy_use.drop('Categories1', col)

  rename_dict = {'DIESEL': 'diesel', 'GASOLINE': 'gasoline', 'H2_ELECTROLYSIS': 'green-hydrogen',
                 'H2_NG': 'grey-hydrogen', 'LFO': 'heating-oil', 'NG': 'gas'}
  #dm_energy_use.rename_col([])

  DM['power-production'] = dm_power_prod
  DM['energy-demand-final-use'] = dm_energy_use
  #DM['oil-gas-supply'] = dm_energy_prod

  return DM

def extract_2050_output_pyomo(m, country_prod, endyr, years_fts, DM_energy):

  #DM.keys = {'installed_GW', 'installed_N', 'emissions', 'storage_in',
  # 'storage_out', 'monthly_operation_GW', 'Losses'}
  DM = utils.get_pyomo_output(m, country_prod, endyr)

  # From ses_eval.mod
  # Hours in a month
  DM['hours_month'] = DM_energy['index0'].filter(
    {'Variables': ['t_op'], 'Years': [endyr]})
  # Efficiency (layers_in_out)
  resources = set(m.RESOURCES)
  technologies = set(m.TECHNOLOGIES)
  storage = set(m.STORAGE_TECH)
  index_list = list((resources | technologies) - storage)
  DM['efficiency'] = utils.pyomo_param_to_dm(m,
                                            pyomo_var_name='layers_in_out',
                                            cntr_name=country_prod,
                                            end_yr=endyr,
                                            indexes=['explicit', 'LAYERS'],
                                            unit_dict={'efficiency': '%'},
                                            explicit=index_list)
  # Sankey / Energy flows
  DM_tmp = extract_sankey_energy_flow(DM)
  DM = DM | DM_tmp

  # Losses:
  # Sum-product over hours
  arr_losses = np.nansum(DM['Losses'].array * DM['hours_month'].array[:, :, :, np.newaxis, :], axis=-1, keepdims=False)
  DM['Losses'].group_all('Categories2')
  DM['Losses'].array = - arr_losses/1000
  DM['Losses'].units = {'Losses': 'TWh'}
  # Rename power-production DM

  # If I'm using natural gas, then it's GasCC, else if I'm using NG_CCS it's GasCC-CCS
  # if 'NG' in DM['energy-demand-final-use'].col_labels['Categories2']:
  # DM['power-production'].groupby({'CHP': '.*COGEN.*'}, regex=True, dim='Categories1', inplace=True)
  # elif 'NG_CCS' in DM['energy-demand-final-use'].col_labels['Categories2']:
  #    DM['power-production'].groupby({'CHP-CCS': '.*COGEN.*'}, regex=True, dim='Categories1', inplace=True)
  map_prod = {'Net-import': ['ELECTRICITY'], 'PV-roof': ['PV'],
              'WindOn': ['WIND'], 'Dam': ['HYDRO_DAM'], 'Dam_new': ['NEW_HYDRO_DAM'],
              'RoR': ['HYDRO_RIVER'],  "RoR_new": ['NEW_HYDRO_RIVER'], 'GasCC-CCS': ['CCGT_CCS'],
              'GasCC': ['CCGT']}
  for key, value in list(map_prod.items()):
    if value[0] not in DM['power-production'].col_labels['Categories1']:
      map_prod.pop(key)
  DM['power-production'].groupby(map_prod, dim='Categories1', inplace=True)
  # Work-around for when new_hydro_river or new_hydro_dam are not available
  DM['power-production'].groupby({'RoR': 'RoR.*', 'Dam': 'Dam.*'}, regex=True, inplace=True, dim='Categories1')
  # Append Losses to power production
  dm_losses = DM['Losses'].copy()
  dm_losses.rename_col('Losses', 'pow_production', 'Variables')
  dm_losses.rename_col('ELECTRICITY', 'Losses', 'Categories1')
  DM['power-production'].append(dm_losses, dim='Categories1')

  # Drop from installed GW
  power_categories = list(m.TECHNOLOGIES_OF_END_USES_TYPE["ELECTRICITY"])
  cogen_categories = list(m.COGEN)
  DM['installed_GW'].filter(
    {'Categories1': power_categories + cogen_categories}, inplace=True)

  reversed_mapping = {'GasCC': ['CCGT'], 'GasCC-CCS': ['CCGT_CCS'],
                      'Nuclear': ['NUCLEAR'],
                      'PV-roof': ['PV'], 'WindOn': ['WIND'],
                      'Dam': ['NEW_HYDRO_DAM', 'HYDRO_DAM'],
                      'RoR': ['NEW_HYDRO_RIVER', 'HYDRO_RIVER'],
                      'Coal': ['COAL_US', 'COAL_IGCC', 'COAL_US_CCS',
                               'COAL_IGCC_CCS'],
                      'Geothermal': ['GEOTHERMAL']}

  # !FIXME: This is probably not all in the same units. Why is GasCC zero here?
  DM['installed_GW'].groupby({'CHP': '.*COGEN.*'}, regex=True,
                             dim='Categories1', inplace=True)
  DM['installed_GW'].groupby(reversed_mapping, dim='Categories1',
                             inplace=True)

  # Rename installed N
  DM['installed_N'].filter({'Categories1': power_categories + cogen_categories}, inplace=True)
  DM['installed_N'].groupby({'CHP': '.*COGEN.*'}, regex=True, dim='Categories1', inplace=True)
  DM['installed_N'] = DM['installed_N'].groupby(reversed_mapping, dim='Categories1', inplace=False)


  # Rename fuel-supply
  # mapping = {'diesel': ['DIESEL'], 'H2': ['H2_NG', 'H2_ELECTROLYSIS'], 'gasoline': ['GASOLINE'],
  #           'gas': ['NG', 'NG_CCS'], 'heating-oil': ['LFO', 'oil'], 'waste': ['WASTE'], 'wood': ['WOOD']}
  # DM['oil-gas-supply'].groupby(mapping, inplace=True, dim='Categories1')

  return DM


def create_future_country_production_trend(DM_2050, DM_input, years_ots, years_fts):

    # Capacity trend - Country level
    dm_cap_2050 = DM_2050['installed_GW'].copy()
    dm_cap = DM_input['cal-capacity'].copy()
    dm_cap_sto = dm_cap.filter({'Categories1': ['Battery-TSO', 'DAC', 'Pump-Open']})
    dm_cap.drop('Categories1', ['Battery-TSO', 'DAC', 'Pump-Open'])
    missing_cat = list(set(dm_cap.col_labels['Categories1']) - set(dm_cap_2050.col_labels['Categories1']))
    dm_cap_2050.add(0, dim='Categories1', col_label=missing_cat, dummy=True)
    missing_cat = list(set(dm_cap_2050.col_labels['Categories1']) - set(dm_cap.col_labels['Categories1']))
    dm_cap.add(0, dim='Categories1', col_label=missing_cat, dummy=True)
    dm_cap_2050.add(np.nan, dummy=True, dim='Years', col_label=dm_cap.col_labels['Years'][:-1])
    dm_cap_2050.sort('Years')
    dm_cap_2050.rename_col('F_Mult', 'pow_capacity', dim='Variables')
    dm_cap_2050.change_unit('pow_capacity', old_unit='GW', new_unit='MW', factor=1000)
    dm_cap.filter({'Country': dm_cap_2050.col_labels['Country']}, inplace=True)
    dm_cap.append(dm_cap_2050, dim='Variables')
    idx = dm_cap.idx
    idx_ots = [idx[yr] for yr in years_ots]
    dm_cap.array[0, idx_ots, idx['pow_capacity'], :] = dm_cap.array[0, idx_ots, idx['pow_existing-capacity'], :]

    # Decommissioning
    cap_latest_ots = dm_cap.array[0, idx_ots[-1], idx['pow_capacity'], :]
    cap_final = dm_cap.array[0, -1, idx['pow_capacity'], :]
    cap_max = dm_cap.array[0, -1, idx['pow_capacity-Pmax'], :]
    # Check that there is decommissioning happening
    # And that it is hitting the maximal capacity limit
    decommissioned_mask = (cap_final < cap_latest_ots) & (cap_max == cap_final)
    idx_fts = [idx[yr] for yr in years_fts]
    idx_fts = np.array(idx_fts)
    dm_cap.array[0, idx_fts[:, None], idx['pow_capacity'], decommissioned_mask] = \
        dm_cap.array[0, idx_fts[:, None], idx['pow_capacity-Pmax'], decommissioned_mask]

    dm_cap.fill_nans('Years')
    # The capacity installed cannot be higher than Pmax
    # The capacity in 2050 should already be below the max
    dm_cap.array[0, idx_fts, idx['pow_capacity'], -1] = np.minimum(dm_cap.array[0, idx_fts, idx['pow_capacity'], -1] ,
                                                                  dm_cap.array[0, idx_fts, idx['pow_capacity-Pmax'], -1] )
    #dm_cap_hist.append(, dim='Years')

    # Production trend - Country level
    dm_prod_2050 = DM_2050['power-production']
    dm_prod_hist = DM_input['cal-production'].copy()

    # Append prod EnergyScope 2050 to historical production at Country level
    dm_prod_hist.drop(dim='Categories1', col_label='Pump-Open')
    missing_cat = list(set(dm_prod_hist.col_labels['Categories1']) - set(dm_prod_2050.col_labels['Categories1']))
    dm_prod_2050.add(0, dummy=True, dim='Categories1', col_label=missing_cat)
    missing_cat = list(set(dm_prod_2050.col_labels['Categories1']) - set(dm_prod_hist.col_labels['Categories1']))
    dm_prod_hist.add(0, dummy=True, dim='Categories1', col_label=missing_cat)
    dm_prod_hist.append(dm_prod_2050, dim='Years')
    dm_add_missing_variables(dm_prod_hist, {'Years': years_fts}, fill_nans=False)
    dm_prod_trend = dm_prod_hist

    ## Use Capacity to create a Pathway at Country level
    dm_cap_tmp = dm_cap.filter({'Variables': ['pow_capacity']})
    # Create fts trend by using the pow_cap-fact = production / capacity
    #fake_net_import_cap = dm_prod_hist[:, :, 'pow_production', 'Net-import', np.newaxis]
    #dm_cap_tmp.add(fake_net_import_cap, dummy=True, dim='Categories1', col_label='Net-import')
    # Remove losses
    dm_losses = dm_prod_trend.filter({'Categories1': ['Losses']})
    dm_net_import = dm_prod_trend.filter({'Categories1': ['Net-import']})
    dm_prod_trend.drop('Categories1', ['Losses', 'Net-import'])
    # Compute capacity factor
    dm_prod_trend.append(dm_cap_tmp.filter({'Categories1': dm_prod_trend.col_labels['Categories1']}), dim='Variables')
    dm_prod_trend.operation('pow_production', '/', 'pow_capacity', out_col='pow_cap-fact', unit='TWh/MW', div0='interpolate')
    dm_prod_trend.fill_nans('Years')
    idx = dm_prod_trend.idx
    idx_fts = [idx[yr] for yr in years_fts]
    dm_prod_trend.array[0, idx_fts, idx['pow_production'], :] = dm_prod_trend.array[0, idx_fts, idx['pow_capacity'], :]\
                                                         * dm_prod_trend.array[0, idx_fts, idx['pow_cap-fact'], :]

    dm_prod_trend.change_unit('pow_cap-fact', old_unit='TWh/MW', new_unit='%', factor=8.760*1e-3, operator='/')
    dm_prod_trend.change_unit('pow_capacity', old_unit='MW', new_unit='GW', factor=1e-3, operator='*')

    ## Compute Losses trend
    # Compute total production
    dm_tot_prod = dm_prod_trend.groupby({'Total': '.*'}, dim='Categories1', regex=True, inplace=False)
    dm_tot_prod.filter({'Variables': ['pow_production']}, inplace=True)
    # Losses / Prod = Loss-rate
    dm_losses.append(dm_tot_prod, dim='Categories1')
    dm_losses.operation('Losses', '/', 'Total', out_col='Loss-rate', unit='%', dim='Categories1')
    # fill-nan loss rate
    dm_losses.fill_nans('Years')
    # Losses = Prod * Loss-rate
    dm_losses.drop('Categories1', 'Losses')
    dm_losses.operation('Loss-rate', '*', 'Total', out_col='Losses', unit='TWh', dim='Categories1')
    dm_losses.filter({'Categories1': ['Losses']}, inplace=True)

    return dm_prod_hist, dm_losses, dm_net_import


def downscale_country_to_canton(dm_prod_cap_cntr, dm_cal_capacity, country_dem, share_of_pop):
    country_prod = dm_prod_cap_cntr.col_labels['Country'][0]
    dm_cal_capacity.add(0, col_label='Net-import', dim='Categories1', dummy=True)
    dm_cal_capacity.filter({'Variables': ['pow_capacity-Pmax']}, inplace=True)
    canton_share = np.where(dm_cal_capacity[country_prod, ...] > 0, dm_cal_capacity[country_dem, ...]
                            / dm_cal_capacity[country_prod, ...], 0)
    dm_cal_capacity.drop(dim='Country', col_label=country_prod)
    dm_cal_capacity.add(canton_share[np.newaxis, ...], dim='Variables', col_label='share', unit='%')
    dm_cal_capacity.add(share_of_pop, col_label='CHP', dim='Categories1', dummy=True)
    dm_cal_capacity.filter({'Categories1': dm_prod_cap_cntr.col_labels['Categories1']}, inplace=True)
    dm_cal_capacity.sort('Categories1')
    dm_prod_cap_cntr.sort('Categories1')
    arr_canton_prod = dm_prod_cap_cntr[country_prod, :, 'pow_production', :] * dm_cal_capacity[country_dem, :, 'share', :]
    arr_canton_cap = dm_prod_cap_cntr[country_prod, :, 'pow_capacity', :] * dm_cal_capacity[country_dem, :, 'share', :]
    arr_canton_cap_fact = dm_prod_cap_cntr[country_prod, :, 'pow_cap-fact', :]
    arr_canton = np.concatenate([arr_canton_prod[np.newaxis, :, np.newaxis, :],
                                 arr_canton_cap[np.newaxis, :, np.newaxis, :],
                                 arr_canton_cap_fact[np.newaxis, :, np.newaxis, :]], axis=2)
    dm_prod_cap_cntr.add(arr_canton, dim='Country', col_label=country_dem)



    return dm_prod_cap_cntr


def balance_demand_prod_with_net_import(dm_prod_cap_cntr, dm_losses, dm_net_import, dm_demand_trend, share_of_pop):

  dm_prod = dm_prod_cap_cntr.filter({'Variables': ['pow_production']})
  #dm_prod.drop('Categories1', ['Net-import', 'Waste'])
  dm_prod.group_all('Categories1', inplace=True)
  # Compute demand by country
  dm_demand_trend.drop('Categories1', 'district-heating')
  dm_demand_trend.group_all('Categories1', inplace=True)
  dm_demand_trend.array = dm_demand_trend.array / share_of_pop
  # demand = prod - losses + net_import
  # net_import = demand - (prod - losses) (NOTE: losses is already negative!)

  arr_net_import = dm_demand_trend.array - (dm_prod.array + dm_losses[:, :, :, 'Losses'])
  #dm_net_import.add(arr_net_import, dim='Categories1', col_label='Net-import-computed')
  idx = dm_net_import.idx
  dm_net_import[:, idx[2023]:, 'pow_production', 'Net-import'] = arr_net_import[:, idx[2023]:, 0]
  dm_losses.append(dm_net_import, dim='Categories1')
  dm_losses.add(np.nan, dim='Variables', col_label=['pow_capacity', 'pow_cap-fact'], dummy=True, unit=['GW', '%'])
  dm_prod_cap_cntr.append(dm_losses, dim='Categories1')

  return dm_prod_cap_cntr


def energyscope_pyomo(data_path, DM_tra, DM_bld, DM_ind, DM_agr, years_ots, years_fts, country_list):
  with open(data_path, 'rb') as handle:
    DM_energy = pickle.load(handle)

  dm_capacity = DM_energy.pop('capacity')
  dm_production = DM_energy.pop('production')
  dm_fuels_supply = DM_energy.pop('fuels')
  DM_input = {'cal-capacity': dm_capacity,
              'cal-production': dm_production,
              'hist-fuels-supply': dm_fuels_supply,
              'demand-bld': DM_bld,
              'demand-tra': DM_tra,
              'demand-ind': DM_ind,
              'demand-agr': DM_agr}

  this_dir =  os.path.dirname(os.path.abspath(__file__))
  data_file_path = os.path.join(this_dir, 'energy/energyscopepyomo/ses_main.json')
  data = load_data(data_file_path)
  m = build_model_structure(data)
  #set_constraints(m, objective = "cost")
  #opt = make_highs()
  #attach(opt, m)

  dm_tra_demand_trend = inter.extract_transport_demand(DM_tra)
  dm_bld_demand_trend = inter.extract_buildings_demand(DM_bld, DM_ind)
  dm_ind_demand_trend = inter.extract_industry_demand(DM_ind)
  dm_agr_demand_trend = inter.extract_agriculture_demand(DM_agr)

  #res = solve(opt, m, warmstart=True)

  endyr = years_fts[-1]
  if ['EU27'] == country_list:  # If you are running for EU27
    country_prod = 'EU27'
    country_dem = 'EU27'
    inter.impose_capacity_constraints_pyomo(m, endyr, dm_capacity,
                                      country=country_prod)
    share_of_pop = 1
  else:  # Else you are running for a canton, a canton + Switzerland, or just Switzerland
    country_prod = 'Switzerland'
    country_dem = 'Switzerland'
    inter.impose_capacity_constraints_pyomo(m, endyr, dm_capacity,
                                      country=country_prod)
    if country_prod in country_list:
      share_of_pop = 1
    else:
      country_dem = country_list[0]
      # You should also check that you are not running with more than a canton at the time if Switzerland
      # is not in the mix
      dm_tmp = dm_production.copy()
      dm_tmp.drop('Categories1', 'Pump-Open')
      country_demand = dm_tmp[0, years_ots[-1], 'pow_production', :].sum(axis=-1)
      canton_demand = (dm_tra_demand_trend[country_dem, years_ots[-1], 'tra_energy-consumption', 'electricity']
                       + dm_bld_demand_trend[country_dem, years_ots[-1], 'bld_energy-consumption', 'electricity']
                       + dm_bld_demand_trend[country_dem, years_ots[-1], 'bld_energy-consumption', 'heat-pump']
                       + dm_ind_demand_trend[country_dem, years_ots[-1], 'ind_energy-end-use', 'electricity']
                       + dm_ind_demand_trend[country_dem, years_ots[-1], 'ind_energy-end-use', 'heat-pump']
                       + dm_agr_demand_trend[country_dem, years_ots[-1], 'agr_energy-consumption', 'electricity']
                       + dm_agr_demand_trend[country_dem, years_ots[-1], 'agr_energy-consumption', 'heat-pump'])

      share_of_pop = 0.07885490043172043  # canton_demand/country_demand #


  dm_tra_demand_trend = inter.impose_transport_demand_pyomo(m, endyr, share_of_pop, DM_tra, country_dem)
  dm_bld_demand_trend = inter.impose_buildings_demand_pyomo(m, endyr, share_of_pop, DM_bld, DM_ind,country_dem)
  dm_ind_demand_trend, dm_agr_demand_trend = inter.impose_industry_demand_pyomo(m, endyr, share_of_pop, DM_ind, DM_agr, country_dem)

  # Avail is in GWh
  # No nuclear
  m.avail['URANIUM'] = 0
  # ampl.getParameter('avail').setValues({'WOOD': 1.5*12279})
  m.f_max['CCGT'] = 0
  m.f_min['CCGT'] = 0
  # ampl.getParameter('avail').setValues({'NG_CCS': 0})
  m.avail['COAL_CCS'] = 0
  m.avail['ELECTRICITY'] = 5000  # Import capped to 5 TWh

  set_constraints(m, objective = "cost")
  # Put show_log to True to see the results of the optimisation
  opt = make_highs(show_log=False)
  attach(opt, m)
  res = solve(opt, m, warmstart=True)


  DM_2050 = extract_2050_output_pyomo(m, country_prod, endyr, years_fts, DM_energy)

  # I should map the losses based on the canton share of the country production
  dm_prod_cap_cntr, dm_losses, dm_net_import \
    = create_future_country_production_trend(DM_2050, DM_input, years_ots, years_fts)


  # Compare Demand = prod - losses + net_import  with the Calculator demand
  #dm_balance = dm_prod_cap_cntr.filter({'Variables': ['pow_production']})

  #dm_tmp = dm_production.copy()
  #dm_tmp.drop('Categories1', ['Losses', 'Net-import'])
  #dm_tmp.rename_col('pow_production', 'pow_production_original', dim='Variables')
  #dm_add_missing_variables(dm_tmp, {'Categories1': dm_balance.col_labels['Categories1']})
  #dm_add_missing_variables(dm_balance, {'Categories1': dm_tmp.col_labels['Categories1']})

  #dm_tmp.append(dm_balance.filter({'Years': years_ots}), dim='Variables')

  #dm_balance.append(dm_losses, dim='Categories1')
  #dm_balance.append(dm_net_import, dim='Categories1')
  #dm_balance.group_all('Categories1')
  # Group Calculator demand
  #dm_demand_trend.drop('Categories1', 'district-heating')
  #dm_demand_trend.group_all('Categories1')
  #dm_balance.append(dm_demand_trend, dim='Variables')

  # Group all the demand fts trends
  dm_demand_trend = dm_bld_demand_trend
  dm_demand_trend.append(dm_ind_demand_trend, dim='Variables')
  dm_add_missing_variables(dm_tra_demand_trend, {'Categories1': dm_demand_trend.col_labels['Categories1']}, fill_nans=False)
  dm_demand_trend.append(dm_tra_demand_trend, dim='Variables')
  dm_demand_trend.append(dm_agr_demand_trend, dim='Variables')
  dm_demand_trend_by_sector = dm_demand_trend.group_all('Categories1', inplace=False)
  dm_demand_trend_by_sector.rename_col('ind_energy-end-use', 'ind_energy-consumption', dim='Variables')
  dm_demand_trend.groupby({'total-energy-consumption': '.*'}, dim='Variables', regex=True, inplace=True)

  # Add demand - production balancing through net import & losses
  dm_prod_cap_cntr = balance_demand_prod_with_net_import(dm_prod_cap_cntr,
                                                         dm_losses,
                                                         dm_net_import,
                                                         dm_demand_trend,
                                                         share_of_pop)

  results_run = inter.prepare_TPE_output(dm_prod_cap_cntr, dm_demand_trend_by_sector)

  return results_run


def energy(lever_setting, years_setting, country_list, interface=Interface()):

  current_file_directory = os.path.dirname(os.path.abspath(__file__))
  years_fts = create_years_list(years_setting[2], years_setting[3], years_setting[4])
  years_ots = create_years_list(years_setting[0], years_setting[1], 1)
  # Read transport input
  if interface.has_link(from_sector='transport', to_sector='energy'):
      DM_transport = interface.get_link(from_sector='transport', to_sector='energy')
  else:
      if len(interface.list_link()) != 0:
          print("You are missing " + 'transport' + " to " + 'energy' + " interface")
      tra_interface_data_file = os.path.join(current_file_directory,
                                             '../_database/data/interface/transport_to_energy.pickle')
      with open(tra_interface_data_file, 'rb') as handle:
          DM_transport = pickle.load(handle)
      for key in DM_transport.keys():
          DM_transport[key].filter({'Country': country_list}, inplace=True)
  # !FIXME: I'm dropping aviation
  DM_transport['passenger'].drop('Categories1', 'aviation')

  # Check country selection for energy module run
  if 'EU27' in country_list:
      if country_list != ['EU27']:
          raise RuntimeError("If you want to solve the energy module for EU27, set geoscale=EU27")
  else:
      list_wo_CH = set(country_list) - {'Switzerland'}
      if len(list_wo_CH) > 1:
          raise RuntimeError("You are trying to solve the energy module for 2 cantons at the same time, "
                             "pick only one canton and eventually Switzerland (see geoscale variable)")

  if interface.has_link(from_sector='buildings', to_sector='energy'):
    DM_buildings = interface.get_link(from_sector='buildings', to_sector='energy')
  else:
    if len(interface.list_link()) != 0:
        print("You are missing " + 'buildings' + " to " + 'energy' + " interface")
    bld_file = os.path.join(current_file_directory, '../_database/data/interface/buildings_to_energy.pickle')
    with open(bld_file, 'rb') as handle:
        DM_buildings = pickle.load(handle)
    filter_DM(DM_buildings, {'Country': country_list})

  if interface.has_link(from_sector='industry', to_sector='energy'):
    DM_industry = interface.get_link(from_sector='industry', to_sector='energy')
  else:
    if len(interface.list_link()) != 0:
        print("You are missing " + 'industry' + " to " + 'energy' + " interface")
    bld_file = os.path.join(current_file_directory, '../_database/data/interface/industry_to_energy.pickle')
    with open(bld_file, 'rb') as handle:
        DM_industry = pickle.load(handle)
    filter_DM(DM_industry, {'Country': country_list})

  if interface.has_link(from_sector='agriculture', to_sector='energy'):
    DM_agriculture = interface.get_link(from_sector='agriculture', to_sector='energy')
  else:
    if len(interface.list_link()) != 0:
        print("You are missing " + 'agriculture' + " to " + 'energy' + " interface")
    agr_file = os.path.join(current_file_directory, '../_database/data/interface/agriculture_to_energy.pickle')
    with open(agr_file, 'rb') as handle:
        DM_agriculture = pickle.load(handle)
    filter_DM(DM_agriculture, {'Country': country_list})


  current_file_directory = os.path.dirname(os.path.abspath(__file__))
  data_filepath = os.path.join(current_file_directory, '../_database/data/datamatrix/energy.pickle')
  results_run = energyscope_pyomo(data_filepath, DM_transport, DM_buildings, DM_industry, DM_agriculture, years_ots, years_fts, country_list)

  return results_run


def local_energy_run():
    # Function to run module as stand alone without other modules/converter or TPE
    years_setting = [1990, 2023, 2025, 2050, 5]
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(current_file_directory, '../config/lever_position.json'))
    lever_setting = json.load(f)[0]
    # Function to run only transport module without converter and tpe

    # get geoscale
    country_list = ['Vaud']

    results_run = energy(lever_setting, years_setting, country_list)

    return results_run

# database_from_csv_to_datamatrix()
#print('In transport, the share of waste by fuel/tech type does not seem right. Fix it.')
#print('Apply technology shares before computing the stock')
#print('For the efficiency, use the new methodology developped for Building (see overleaf on U-value)')
if __name__ == "__main__":
  results_run = local_energy_run()
#local_energy_run()

#with open('/Users/paruta/Desktop/transport_EU.pickle', 'rb') as handle:
#    DM_transport = pickle.load(handle)


#with open('/Users/paruta/Desktop/transport_EU.pickle', 'rb') as handle:
#    DM_transport = pickle.load(handle)
#print('Hello')
