#from amplpy import AMPL, add_to_path
#from typing import List, Dict
from model.energy.energyscopepyomo.ses_pyomo import load_data, build_model, make_highs, attach, solve, extract_results
import pyomo.environ as pyo
from model.common.interface_class import Interface
from model.common.data_matrix_class import DataMatrix
import os
from model.common.auxiliary_functions import filter_DM, create_years_list, \
  filter_geoscale, filter_country_and_load_data_from_pickles, \
  dm_add_missing_variables, return_lever_data
import pickle
import json
import numpy as np
import pandas as pd
import model.energy.interfaces as inter
import model.energy.utils as utils
import re


def define_sets_pyomo(m: pyo.ConcreteModel):
  # --- Base / main sets (explicit values) ---
  m.PERIODS = pyo.Set(initialize=range(1, 13), ordered=True)

  m.SECTORS = pyo.Set(initialize=[
      "HOUSEHOLDS", "SERVICES", "INDUSTRY", "TRANSPORTATION"
  ])

  m.END_USES_INPUT = pyo.Set(initialize=[
      "ELECTRICITY", "LIGHTING", "HEAT_HIGH_T", "HEAT_LOW_T_SH",
      "HEAT_LOW_T_HW", "MOBILITY_PASSENGER", "MOBILITY_FREIGHT"
  ])

  m.END_USES_CATEGORIES = pyo.Set(initialize=[
      "ELECTRICITY", "HEAT_HIGH_T", "HEAT_LOW_T",
      "MOBILITY_PASSENGER", "MOBILITY_FREIGHT"
  ])

  m.EXPORT = pyo.Set(initialize=["ELEC_EXPORT"])
  m.BIOFUELS = pyo.Set(initialize=["BIOETHANOL", "BIODIESEL", "SNG"])

  m.RESOURCES = pyo.Set(initialize=[
      "ELECTRICITY", "GASOLINE", "DIESEL", "BIOETHANOL", "BIODIESEL", "LFO", "LNG",
      "NG", "NG_CCS", "SNG", "WOOD", "COAL", "COAL_CCS", "URANIUM", "WASTE", "H2",
      "ELEC_EXPORT"
  ])

  m.STORAGE_TECH = pyo.Set(initialize=["PUMPED_HYDRO", "POWER2GAS"])

  m.INFRASTRUCTURE = pyo.Set(initialize=[
      "EFFICIENCY", "DHN", "GRID", "POWER2GAS_1", "POWER2GAS_2", "POWER2GAS_3",
      "H2_ELECTROLYSIS", "H2_NG", "H2_BIOMASS", "GASIFICATION_SNG", "PYROLYSIS"
  ])

  m.COGEN = pyo.Set(initialize=[
      "IND_COGEN_GAS", "IND_COGEN_WOOD", "IND_COGEN_WASTE", "DHN_COGEN_GAS",
      "DHN_COGEN_WOOD", "DHN_COGEN_WASTE", "DEC_COGEN_GAS", "DEC_COGEN_OIL",
      "DEC_ADVCOGEN_GAS", "DEC_ADVCOGEN_H2"
  ])

  m.BOILERS = pyo.Set(initialize=[
      "IND_BOILER_GAS", "IND_BOILER_WOOD", "IND_BOILER_OIL", "IND_BOILER_COAL",
      "IND_BOILER_WASTE", "DHN_BOILER_GAS", "DHN_BOILER_WOOD", "DHN_BOILER_OIL",
      "DEC_BOILER_GAS", "DEC_BOILER_WOOD", "DEC_BOILER_OIL"
  ])

  # END_USES_TYPES_OF_CATEGORY is a set-of-sets indexed by END_USES_CATEGORIES
  _types_of_cat = {
      "ELECTRICITY": ["ELECTRICITY"],
      "HEAT_HIGH_T": ["HEAT_HIGH_T"],
      "HEAT_LOW_T": ["HEAT_LOW_T_DHN", "HEAT_LOW_T_DECEN"],
      "MOBILITY_PASSENGER": ["MOB_PUBLIC", "MOB_PRIVATE"],
      "MOBILITY_FREIGHT": ["MOB_FREIGHT_RAIL", "MOB_FREIGHT_ROAD"],
  }
  m.END_USES_TYPES_OF_CATEGORY = pyo.Set(
      m.END_USES_CATEGORIES, initialize=_types_of_cat
  )

  # Secondary: END_USES_TYPES := union over categories
  m.END_USES_TYPES = pyo.Set(
      initialize=lambda m: {j for i in m.END_USES_CATEGORIES for j in m.END_USES_TYPES_OF_CATEGORY[i]}
  )

  # TECHNOLOGIES_OF_END_USES_TYPE is a set-of-sets indexed by END_USES_TYPES
  _tech_of_type = {
      "ELECTRICITY": [
          "NUCLEAR", "CCGT", "CCGT_CCS", "COAL_US", "COAL_IGCC", "COAL_US_CCS", "COAL_IGCC_CCS",
          "PV", "WIND", "HYDRO_DAM", "NEW_HYDRO_DAM", "HYDRO_RIVER", "NEW_HYDRO_RIVER", "GEOTHERMAL"
      ],
      "HEAT_HIGH_T": [
          "IND_COGEN_GAS", "IND_COGEN_WOOD", "IND_COGEN_WASTE", "IND_BOILER_GAS", "IND_BOILER_WOOD",
          "IND_BOILER_OIL", "IND_BOILER_COAL", "IND_BOILER_WASTE", "IND_DIRECT_ELEC"
      ],
      "HEAT_LOW_T_DHN": [
          "DHN_HP_ELEC", "DHN_COGEN_GAS", "DHN_COGEN_WOOD", "DHN_COGEN_WASTE", "DHN_BOILER_GAS",
          "DHN_BOILER_WOOD", "DHN_BOILER_OIL", "DHN_DEEP_GEO"
      ],
      "HEAT_LOW_T_DECEN": [
          "DEC_HP_ELEC", "DEC_THHP_GAS", "DEC_COGEN_GAS", "DEC_COGEN_OIL", "DEC_ADVCOGEN_GAS",
          "DEC_ADVCOGEN_H2", "DEC_BOILER_GAS", "DEC_BOILER_WOOD", "DEC_BOILER_OIL", "DEC_SOLAR",
          "DEC_DIRECT_ELEC"
      ],
      "MOB_PUBLIC": [
          "TRAMWAY_TROLLEY", "BUS_COACH_DIESEL", "BUS_COACH_HYDIESEL", "BUS_COACH_CNG_STOICH",
          "BUS_COACH_FC_HYBRIDH2", "TRAIN_PUB"
      ],
      "MOB_PRIVATE": [
          "CAR_GASOLINE", "CAR_DIESEL", "CAR_NG", "CAR_HEV", "CAR_PHEV", "CAR_BEV", "CAR_FUEL_CELL"
      ],
      "MOB_FREIGHT_RAIL": ["TRAIN_FREIGHT"],
      "MOB_FREIGHT_ROAD": ["TRUCK"],
  }
  m.TECHNOLOGIES_OF_END_USES_TYPE = pyo.Set(
      m.END_USES_TYPES, initialize=_tech_of_type
  )

  # Secondary: LAYERS := (RESOURCES \ BIOFUELS \ EXPORT) ∪ END_USES_TYPES
  m.LAYERS = pyo.Set(
      initialize=lambda m: (set(m.RESOURCES) - set(m.BIOFUELS) - set(m.EXPORT)) | set(m.END_USES_TYPES)
  )

  # Secondary: TECHNOLOGIES := (⋃ over types TECHNOLOGIES_OF_END_USES_TYPE[type]) ∪ STORAGE_TECH ∪ INFRASTRUCTURE
  def _tech_init(m):
      techs = set().union(*[set(m.TECHNOLOGIES_OF_END_USES_TYPE[t]) for t in m.END_USES_TYPES])
      techs |= set(m.STORAGE_TECH) | set(m.INFRASTRUCTURE)
      return techs
  m.TECHNOLOGIES = pyo.Set(initialize=_tech_init)

  # Secondary: TECHNOLOGIES_OF_END_USES_CATEGORY[i] within TECHNOLOGIES
  # For each category, union of all technologies belonging to that category’s end-use types
  def _tech_of_cat_init(m, cat):
      types = list(m.END_USES_TYPES_OF_CATEGORY[cat])
      techs = set().union(*[set(m.TECHNOLOGIES_OF_END_USES_TYPE[t]) for t in types])
      # Enforce the "within TECHNOLOGIES" intent by intersecting (defensive)
      return techs & set(m.TECHNOLOGIES)

  m.TECHNOLOGIES_OF_END_USES_CATEGORY = pyo.Set(
      m.END_USES_CATEGORIES, within=m.TECHNOLOGIES, initialize=_tech_of_cat_init
  )

  return


def define_sets(ampl):

    # Declare sets using declareSet()
    # Assign values to sets
    ampl.getSet("PERIODS").setValues(list(range(1, 13)))
    ampl.getSet("SECTORS").setValues(["HOUSEHOLDS", "SERVICES", "INDUSTRY", "TRANSPORTATION"])
    ampl.getSet("END_USES_INPUT").setValues(["ELECTRICITY", "LIGHTING", "HEAT_HIGH_T", "HEAT_LOW_T_SH",
                                             "HEAT_LOW_T_HW", "MOBILITY_PASSENGER", "MOBILITY_FREIGHT"])
    ampl.getSet("END_USES_CATEGORIES").setValues(["ELECTRICITY", "HEAT_HIGH_T", "HEAT_LOW_T",
                                                  "MOBILITY_PASSENGER", "MOBILITY_FREIGHT"])

    ampl.getSet("EXPORT").setValues(["ELEC_EXPORT"])
    ampl.getSet("BIOFUELS").setValues(["BIOETHANOL", "BIODIESEL", "SNG"])
    ampl.getSet("RESOURCES").setValues(["ELECTRICITY", "GASOLINE", "DIESEL", "BIOETHANOL", "BIODIESEL", "LFO", "LNG",
                                        "NG", "NG_CCS", "SNG", "WOOD", "COAL", "COAL_CCS","URANIUM", "WASTE", "H2", "ELEC_EXPORT"])
    ampl.getSet("STORAGE_TECH").setValues(["PUMPED_HYDRO", "POWER2GAS"])
    ampl.getSet("INFRASTRUCTURE").setValues(["EFFICIENCY", "DHN", "GRID", "POWER2GAS_1", "POWER2GAS_2", "POWER2GAS_3",
                                             "H2_ELECTROLYSIS", "H2_NG", "H2_BIOMASS", "GASIFICATION_SNG", "PYROLYSIS"])
    ampl.getSet("COGEN").setValues(["IND_COGEN_GAS", "IND_COGEN_WOOD", "IND_COGEN_WASTE", "DHN_COGEN_GAS",
                                     "DHN_COGEN_WOOD", "DHN_COGEN_WASTE", "DEC_COGEN_GAS", "DEC_COGEN_OIL",
                                     "DEC_ADVCOGEN_GAS", "DEC_ADVCOGEN_H2"])
    ampl.getSet("BOILERS").setValues(["IND_BOILER_GAS", "IND_BOILER_WOOD", "IND_BOILER_OIL", "IND_BOILER_COAL",
                                      "IND_BOILER_WASTE", "DHN_BOILER_GAS", "DHN_BOILER_WOOD", "DHN_BOILER_OIL",
                                      "DEC_BOILER_GAS", "DEC_BOILER_WOOD", "DEC_BOILER_OIL"])

    ampl.getSet("END_USES_TYPES_OF_CATEGORY").setValues({
        "ELECTRICITY": ["ELECTRICITY"],
        "HEAT_HIGH_T": ["HEAT_HIGH_T"],
        "HEAT_LOW_T": ["HEAT_LOW_T_DHN", "HEAT_LOW_T_DECEN"],
        "MOBILITY_PASSENGER": ["MOB_PUBLIC", "MOB_PRIVATE"],
        "MOBILITY_FREIGHT": ["MOB_FREIGHT_RAIL", "MOB_FREIGHT_ROAD"]
    })

    ampl.getSet("TECHNOLOGIES_OF_END_USES_TYPE").setValues({
            "ELECTRICITY": [
                "NUCLEAR", "CCGT", "CCGT_CCS", "COAL_US", "COAL_IGCC", "COAL_US_CCS", "COAL_IGCC_CCS",
                "PV", "WIND", "HYDRO_DAM", "NEW_HYDRO_DAM", "HYDRO_RIVER", "NEW_HYDRO_RIVER", "GEOTHERMAL"
            ],
            "HEAT_HIGH_T": [
                "IND_COGEN_GAS", "IND_COGEN_WOOD", "IND_COGEN_WASTE", "IND_BOILER_GAS", "IND_BOILER_WOOD",
                "IND_BOILER_OIL", "IND_BOILER_COAL", "IND_BOILER_WASTE", "IND_DIRECT_ELEC"
            ],
            "HEAT_LOW_T_DHN": [
                "DHN_HP_ELEC", "DHN_COGEN_GAS", "DHN_COGEN_WOOD", "DHN_COGEN_WASTE", "DHN_BOILER_GAS",
                "DHN_BOILER_WOOD", "DHN_BOILER_OIL", "DHN_DEEP_GEO"
            ],
            "HEAT_LOW_T_DECEN": [
                "DEC_HP_ELEC", "DEC_THHP_GAS", "DEC_COGEN_GAS", "DEC_COGEN_OIL", "DEC_ADVCOGEN_GAS",
                "DEC_ADVCOGEN_H2", "DEC_BOILER_GAS", "DEC_BOILER_WOOD", "DEC_BOILER_OIL", "DEC_SOLAR",
                "DEC_DIRECT_ELEC"
            ],
            "MOB_PUBLIC": [
                "TRAMWAY_TROLLEY", "BUS_COACH_DIESEL", "BUS_COACH_HYDIESEL", "BUS_COACH_CNG_STOICH",
                "BUS_COACH_FC_HYBRIDH2", "TRAIN_PUB"
            ],
            "MOB_PRIVATE": [
                "CAR_GASOLINE", "CAR_DIESEL", "CAR_NG", "CAR_HEV", "CAR_PHEV", "CAR_BEV", "CAR_FUEL_CELL"
            ],
            "MOB_FREIGHT_RAIL": ["TRAIN_FREIGHT"],
            "MOB_FREIGHT_ROAD": ["TRUCK"]
    })

    return


def read_2050_data(ampl, DM, country, endyr):

    define_sets(ampl)
    for key in DM.keys():
        dm = DM[key].filter({'Country': [country], 'Years': [endyr]})
        idx = dm.idx
        if key == 'param':
            for var in dm.col_labels['Variables']:
                ampl.getParameter(var).setValues([dm.array[0, 0, idx[var]]])
        elif 'index' in key:
            for var in dm.col_labels['Variables']:
                if '1' in dm.col_labels['Categories1']:
                    index = list(map(int, dm.col_labels['Categories1']))
                else:
                    index = dm.col_labels['Categories1']
                data = dm.array[0, 0, idx[var], :]
                data_dict = dict(zip(index, data))
                ampl.getParameter(var).setValues(data_dict)
        else:
            index = dm.col_labels['Categories1']
            if '1' in dm.col_labels['Categories2']:
                columns = list(map(int, dm.col_labels['Categories2']))
            else:
                columns = dm.col_labels['Categories2']
            data = dm.array[0, 0, 0, :, :]
            df = pd.DataFrame(data, index=index, columns=columns)
            ampl.getParameter(key).setValues(df)

    return


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
  cat1, cat2 = np.where(dm_energy_full.array[0, 0, 0, :, :]>0)
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
  if 'NG_CCS-NG_CCSv2' in dm_energy_prod.col_labels['Categories1']:
    dm_natural_gas_elec = dm_energy_prod.filter({'Categories1': ['NG_CCS-NG_CCSv2']})
    dm_power_prod.append(dm_natural_gas_elec, dim='Categories1')
  if 'CCGT_CCS-ELECTRICITYv2' in dm_power_prod.col_labels['Categories1']:
    dm_power_prod.drop('Categories1', 'CCGT_CCS-ELECTRICITYv2')

  cat_elec = dm_power_prod.col_labels['Categories1']
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


def extract_2050_output(ampl, country_prod, endyr, years_fts, DM_energy):

    DM = utils.get_ampl_output(ampl, country_prod, endyr)

    # From ses_eval.mod
    # Hours in a month
    DM['hours_month'] = DM_energy['index0'].filter({'Variables': ['t_op'], 'Years': [endyr]})
    # Efficiency (layers_in_out)
    resources = set(r[0] for r in ampl.get_set('RESOURCES').get_values())
    technologies = set(t[0] for t in ampl.get_set('TECHNOLOGIES').get_values())
    storage = set(s[0] for s in ampl.get_set('STORAGE_TECH').get_values())
    index_list = list((resources | technologies) - storage)
    DM['efficiency'] = utils.ampl_param_to_dm(ampl, ampl_var_name='layers_in_out', cntr_name=country_prod, end_yr=endyr,
                                        indexes=['explicit', 'LAYERS'], unit_dict={'efficiency': '%'},
                                        explicit=index_list)
    # Sankey / Energy flows
    DM_tmp = extract_sankey_energy_flow(DM)
    DM = DM | DM_tmp

    # Rename power-production DM

    # If I'm using natural gas, then it's GasCC, else if I'm using NG_CCS it's GasCC-CCS
    #if 'NG' in DM['energy-demand-final-use'].col_labels['Categories2']:
    #DM['power-production'].groupby({'CHP': '.*COGEN.*'}, regex=True, dim='Categories1', inplace=True)
    #elif 'NG_CCS' in DM['energy-demand-final-use'].col_labels['Categories2']:
    #    DM['power-production'].groupby({'CHP-CCS': '.*COGEN.*'}, regex=True, dim='Categories1', inplace=True)
    map_prod = {'Net-import': ['ELECTRICITY'], 'PV-roof': ['PV'], 'WindOn': ['WIND'], 'Dam': ['HYDRO_DAM'],
                'RoR': ['HYDRO_RIVER'], 'GasCC-CCS': ['NG_CCS'], 'GasCC': ['CCGT']}
    for key, value in list(map_prod.items()):
        if value[0] not in DM['power-production'].col_labels['Categories1']:
            map_prod.pop(key)
    DM['power-production'].groupby(map_prod, dim='Categories1', inplace=True)

    # Drop from installed GW
    power_categories = list(ampl.get_set("TECHNOLOGIES_OF_END_USES_TYPE").get("ELECTRICITY"))
    cogen_categories = list(ampl.getSet("COGEN"))
    DM['installed_GW'].filter({'Categories1': power_categories+ cogen_categories}, inplace=True)

    reversed_mapping = {'GasCC': ['CCGT'], 'GasCC-CCS': ['CCGT_CCS'], 'Nuclear': ['NUCLEAR'],
                        'PV-roof': ['PV'], 'WindOn': ['WIND'], 'Dam': ['NEW_HYDRO_DAM', 'HYDRO_DAM'],
                        'RoR': ['NEW_HYDRO_RIVER', 'HYDRO_RIVER'],
                        'Coal': ['COAL_US', 'COAL_IGCC', 'COAL_US_CCS', 'COAL_IGCC_CCS'],
                        'Geothermal': ['GEOTHERMAL']}

    # !FIXME: This is probably not all in the same units. Why is GasCC zero here?
    DM['installed_GW'].groupby({'CHP': '.*COGEN.*'}, regex=True, dim='Categories1', inplace=True)
    DM['installed_GW'].groupby(reversed_mapping, dim='Categories1', inplace=True)

    # Rename installed N
    DM['installed_N'].filter({'Categories1': power_categories+ cogen_categories}, inplace=True)
    DM['installed_N'].groupby({'CHP': '.*COGEN.*'}, regex=True, dim='Categories1', inplace=True)
    DM['installed_N'] = DM['installed_N'].groupby(reversed_mapping, dim='Categories1', inplace=False)

    # Rename fuel-supply
    #mapping = {'diesel': ['DIESEL'], 'H2': ['H2_NG', 'H2_ELECTROLYSIS'], 'gasoline': ['GASOLINE'],
    #           'gas': ['NG', 'NG_CCS'], 'heating-oil': ['LFO', 'oil'], 'waste': ['WASTE'], 'wood': ['WOOD']}
    #DM['oil-gas-supply'].groupby(mapping, inplace=True, dim='Categories1')

    return DM


def create_future_country_trend(DM_2050, DM_input, years_ots, years_fts):

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
    dm_prod_hist = DM_input['cal-production']
    #dm_prod_hist.rename_col('Oil-Gas', 'GasCC', dim='Categories1')
    #dm_prod_stor = dm_prod_hist.filter({'Categories1': ['Pump-Open']})
    dm_prod_hist.drop(dim='Categories1', col_label='Pump-Open')
    missing_cat = list(set(dm_prod_hist.col_labels['Categories1']) - set(dm_prod_2050.col_labels['Categories1']))
    dm_prod_2050.add(0, dummy=True, dim='Categories1', col_label=missing_cat)
    missing_cat = list(set(dm_prod_2050.col_labels['Categories1']) - set(dm_prod_hist.col_labels['Categories1']))
    dm_prod_hist.add(0, dummy=True, dim='Categories1', col_label=missing_cat)
    dm_prod_hist.append(dm_prod_2050, dim='Years')
    years_missing = list(set(years_fts) - set(dm_prod_hist.col_labels['Years']))
    dm_prod_hist.add(np.nan, dim='Years', col_label=years_missing, dummy=True)
    dm_prod_hist.sort('Years')
    dm_cap_tmp = dm_cap.filter({'Variables': ['pow_capacity']})
    # Create fts trend by using the pow_cap-fact
    fake_net_import_cap = dm_prod_hist[:, :, 'pow_production', 'Net-import', np.newaxis]
    dm_cap_tmp.add(fake_net_import_cap, dummy=True, dim='Categories1', col_label='Net-import')
    dm_prod_hist.append(dm_cap_tmp.filter({'Categories1': dm_prod_hist.col_labels['Categories1']}), dim='Variables')
    dm_prod_hist.operation('pow_production', '/', 'pow_capacity', out_col='pow_cap-fact', unit='TWh/MW', div0='interpolate')
    dm_prod_hist.fill_nans('Years')
    idx = dm_prod_hist.idx
    idx_fts = [idx[yr] for yr in years_fts]
    dm_prod_hist.array[0, idx_fts, idx['pow_production'], :] = dm_prod_hist.array[0, idx_fts, idx['pow_capacity'], :]\
                                                         * dm_prod_hist.array[0, idx_fts, idx['pow_cap-fact'], :]

    dm_prod_hist.change_unit('pow_cap-fact', old_unit='TWh/MW', new_unit='%', factor=8.760*1e-3, operator='/')
    dm_prod_hist.change_unit('pow_capacity', old_unit='MW', new_unit='GW', factor=1e-3, operator='*')

    # Fossil-fuels
    # !FIXME: you are dropping Kerosene - get kerosene demand from transport, or add aviation to the model
   # dm_fuel = DM_input['hist-fuels-supply']
   # dm_2050_fuel = DM_2050['oil-gas-supply'].copy()
   # dm_2050_fuel.rename_col('pow_production', 'pow_fuel-supply', dim='Variables')
   # dm_fuel.add(0, dim='Categories1', col_label='H2', dummy=True)
   # dm_fuel.drop(dim='Categories1', col_label='kerosene')
   # dm_fuel.append(dm_2050_fuel, dim='Years')
   # missing_years = list(set(years_fts) - set(dm_fuel.col_labels['Years']))

    # !FIXME: big discontinuity between historical and future values on heating oil and gas. Check historical building.
    #  Calibration missing? Check Swiss data, what did you calibrate against? Consumption or supply ?
 #   dm_fuel.add(np.nan, dim='Years', col_label=missing_years, dummy=True)

    return dm_prod_hist


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


def balance_demand_prod_with_net_import(dm_prod_cap_cntr, dm_demand_trend, share_of_pop):

  dm_prod = dm_prod_cap_cntr.filter({'Variables': ['pow_production']})
  dm_prod.drop('Categories1', ['Net-import', 'Waste'])
  dm_prod.group_all('Categories1', inplace=True)
  dm_demand_trend.drop('Categories1', 'district-heating')
  dm_demand_trend.group_all('Categories1', inplace=True)
  dm_demand_trend.array = dm_demand_trend.array / share_of_pop
  arr_net_import = dm_demand_trend.array - dm_prod.array

  idx = dm_prod_cap_cntr.idx
  dm_prod_cap_cntr[:, :, 'pow_production', 'Net-import'] = arr_net_import[:, :, 0]

  return dm_prod_cap_cntr


def energyscope_pyomo(data_path, DM_tra, DM_bld, DM_ind, years_ots, years_fts, country_list):
  with open(data_path, 'rb') as handle:
    DM_energy = pickle.load(handle)

  data = load_data("ses_main.json")
  m = build_model(data, objective="gwp")
  opt = make_highs()
  attach(opt, m)
  res = solve(opt, m, warmstart=True)

  return



def energyscope(data_path, DM_tra, DM_bld, DM_ind, years_ots, years_fts, country_list):
    endyr = years_fts[-1]

    add_to_path(r'/Applications/AMPL')

    with open(data_path, 'rb') as handle:
        DM_energy = pickle.load(handle)

    #DM_energy['index3'][:, :, 'c_inv', 'PV'] = 400
    # Create an AMPL object
    ampl = AMPL()
    ampl.eval('reset;')
    ampl.setOption("presolve_eps", 1e-12)


    # Use glpk solver
    ampl.option["solver"] = 'highs'

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "energy/energyscope-MILP/ses_main.mod")
    # Read the model
    ampl.read(f)
    #.readData("energyscope-MILP/ses_main.dat")
    # Extract existing capacity data + Nexus-e forecast
    dm_capacity = DM_energy.pop('capacity')
    dm_production = DM_energy.pop('production')
    dm_fuels_supply = DM_energy.pop('fuels')
    DM_input = {'cal-capacity': dm_capacity,
                'cal-production': dm_production,
                'hist-fuels-supply': dm_fuels_supply,
                'demand-bld': DM_bld,
                'demand-tra': DM_tra,
                'demand-ind': DM_ind}

    if ['EU27'] == country_list:  # If you are running for EU27
        country_prod = 'EU27'
        country_dem = 'EU27'
        read_2050_data(ampl, DM_energy, country=country_prod, endyr=endyr)
        inter.impose_capacity_constraints(ampl, endyr, dm_capacity, country=country_prod)
        share_of_pop = 1
    else:  # Else you are running for a canton, a canton + Switzerland, or just Switzerland
        country_prod = 'Switzerland'
        country_dem = 'Switzerland'
        read_2050_data(ampl, DM_energy, country=country_prod, endyr=endyr)
        inter.impose_capacity_constraints(ampl, endyr, dm_capacity, country=country_prod)
        if country_prod in country_list:
            share_of_pop = 1
        else:
            # FIXME: This needs to be given as input from lifestyles and it depends on the canton
            country_dem = country_list[0]
            # You should also check that you are not running with more than a canton at the time if Switzerland
            # is not in the mix
            share_of_pop = 0.095

    # Impose the 2050 demand and return the full fts demand for electricity
    dm_tra_demand_trend = inter.impose_transport_demand(ampl, endyr, share_of_pop, DM_tra, country_dem)
    dm_bld_demand_trend = inter.impose_buildings_demand(ampl, endyr, share_of_pop, DM_bld, DM_ind, country_dem)
    dm_ind_demand_trend = inter.impose_industry_demand(ampl, endyr, share_of_pop, DM_ind, country_dem)

    # Group all the demand fts trends
    dm_demand_trend = dm_bld_demand_trend
    dm_demand_trend.append(dm_ind_demand_trend, dim='Variables')
    dm_add_missing_variables(dm_tra_demand_trend, {'Categories1': dm_demand_trend.col_labels['Categories1']}, fill_nans=False)
    dm_demand_trend.append(dm_tra_demand_trend, dim='Variables')
    dm_demand_trend.groupby({'total-energy-consumption': '.*'}, dim='Variables', regex=True, inplace=True)

    # No nuclear
    ampl.getParameter('avail').setValues({'URANIUM': 0})
    #ampl.getParameter('avail').setValues({'WOOD': 1.5*12279})
    ampl.getParameter('f_max').setValues({'CCGT': 0})
    ampl.getParameter('f_min').setValues({'CCGT': 0})
    #ampl.getParameter('avail').setValues({'NG_CCS': 0})
    ampl.getParameter('avail').setValues({'COAL_CCS': 0})


    # Solve the model (togliere comando “solve" dal mod)
    print(f"Before solve")
    ampl.solve()
    print(f"After solve")

    DM_2050 = extract_2050_output(ampl, country_prod, endyr, years_fts, DM_energy)

    dm_prod_cap_cntr = create_future_country_trend(DM_2050, DM_input, years_ots, years_fts)

    # Add demand - production balancing through net import
    dm_prod_cap_cntr = balance_demand_prod_with_net_import(dm_prod_cap_cntr, dm_demand_trend, share_of_pop)

    results_run = inter.prepare_TPE_output(dm_prod_cap_cntr)

    # Downscale from Country_prod to Country_dem
    #dm_prod_cap_canton = downscale_country_to_canton(dm_prod_cap_cntr, DM_input['cal-capacity'], country_dem, share_of_pop)

   # current_file_directory = os.path.dirname(os.path.abspath(__file__))
   # file = os.path.join(current_file_directory, 'energy/energyscope-MILP/ses_eval.mod')
    # Print output
    #ampl.read(file)

    # Close AMPL
   # ampl.close()
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

  current_file_directory = os.path.dirname(os.path.abspath(__file__))
  data_filepath = os.path.join(current_file_directory, '../_database/data/datamatrix/energy.pickle')
  results_run = energyscope_pyomo(data_filepath, DM_transport, DM_buildings, DM_industry, years_ots, years_fts, country_list)

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
