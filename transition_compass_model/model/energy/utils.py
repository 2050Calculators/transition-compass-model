from model.common.data_matrix_class import DataMatrix
import numpy as np
from typing import List, Dict


def pyomo_var_to_dm(m, pyomo_var_name: str, cntr_name: str, end_yr: int,
                   indexes: List[str], unit_dict: Dict[str, str]):
  var = list(unit_dict.keys())[0]
  col_labels = {'Country': [cntr_name], 'Years': [end_yr], 'Variables': [var]}
  i = 1
  for cat_group in indexes:
    col_labels['Categories' + str(i)] = list(getattr(m, cat_group))
    i = i + 1

  dm = DataMatrix(col_labels, unit_dict)
  shape = tuple(len(col_labels[k]) for k in dm.dim_labels)
  arr = np.zeros(shape)
  dm.array = arr
  idx = dm.idx

  variable = getattr(m, pyomo_var_name)
  if len(indexes) == 1:
    for cat1 in variable:
      value = variable[cat1].value
      if value != 0:
        dm.array[0, 0, 0, idx[cat1]] = value
  if len(indexes) == 2:
    for cat1, cat2 in variable:
      value = variable[cat1, cat2].value
      if value != 0:
        dm.array[0, 0, 0, idx[cat1], idx[cat2]] = value
  if len(indexes) == 3:
    for cat1, cat2, cat3 in variable:
      value = variable[cat1, cat2, cat3].value
      if value != 0:
        dm.array[0, 0, 0, idx[cat1], idx[cat2], idx[cat3]] = value

  return dm


def ampl_var_to_dm(ampl, ampl_var_name: str, cntr_name: str, end_yr: int, indexes: List[str], unit_dict: Dict[str, str]):
    var = list(unit_dict.keys())[0]
    col_labels = {'Country': [cntr_name], 'Years': [end_yr], 'Variables': [var]}
    i = 1
    for cat_group in indexes:
        col_labels['Categories'+str(i)] = [row[0] for row in ampl.get_set(cat_group).get_values()]
        i = i+1

    dm = DataMatrix(col_labels, unit_dict)
    shape = tuple(len(col_labels[k]) for k in dm.dim_labels)
    arr = np.zeros(shape)
    dm.array = arr
    idx = dm.idx

    if len(indexes) == 1:
        for cat1, value in ampl.get_variable(ampl_var_name).get_values():
            if value != 0:
                dm.array[0, 0, 0, idx[cat1]] = value
    if len(indexes) == 2:
        for cat1, cat2, value in ampl.get_variable(ampl_var_name).get_values():
            if value != 0:
                dm.array[0, 0, 0, idx[cat1], idx[cat2]] = value

    return dm


def f_mult_t_pyomo_var_to_dm(m, cntr_name: str, end_yr: int):
  # Extract Operation in each period. multiplication factor with respect to the values in layers_in_out table. Takes into account c_p
  var = 'pow_monthly-operation'
  col_labels = {'Country': [cntr_name], 'Years': [end_yr], 'Variables': [var]}

  resources_col = list(m.RESOURCES)
  technologies_col = list(m.TECHNOLOGIES)
  col_labels['Categories1'] = resources_col + technologies_col

  col_labels['Categories2'] = list(m.PERIODS)

  dm = DataMatrix(col_labels, {'pow_monthly-operation': 'GW'})
  shape = tuple(len(col_labels[k]) for k in dm.dim_labels)
  arr = np.zeros(shape)
  dm.array = arr
  idx = dm.idx

  for cat1, cat2 in m.F_Mult_t:
    value = m.F_Mult_t[cat1, cat2].value
    if value != 0:
      dm.array[0, 0, 0, idx[cat1], idx[cat2]] = value

  return dm


def f_mult_t_ampl_var_to_dm(ampl, cntr_name: str, end_yr: int):
    # Extract Operation in each period. multiplication factor with respect to the values in layers_in_out table. Takes into account c_p
    var = 'pow_monthly-operation'
    col_labels = {'Country': [cntr_name], 'Years': [end_yr], 'Variables': [var]}

    resources_col = [row[0] for row in ampl.get_set('RESOURCES').get_values()]
    technologies_col = [row[0] for row in ampl.get_set('TECHNOLOGIES').get_values()]
    col_labels['Categories1'] = resources_col + technologies_col

    col_labels['Categories2'] = [row[0] for row in ampl.get_set('PERIODS').get_values()]

    dm = DataMatrix(col_labels, {'pow_monthly-operation': 'GW'})
    shape = tuple(len(col_labels[k]) for k in dm.dim_labels)
    arr = np.zeros(shape)
    dm.array = arr
    idx = dm.idx

    for cat1, cat2, value in ampl.get_variable('F_Mult_t').get_values():
        if value != 0:
            dm.array[0, 0, 0, idx[cat1], idx[cat2]] = value

    return dm


def get_pyomo_output(m, country, endyr):
  dm_f_mult = pyomo_var_to_dm(m, pyomo_var_name='F_Mult', cntr_name=country,
                             end_yr=endyr,
                             indexes=['TECHNOLOGIES'],
                             unit_dict={'F_Mult': 'GW'})

  # Sto_in: Power [GW] input to the storage in a certain period
  dm_storage_in = pyomo_var_to_dm(m, pyomo_var_name='Storage_In',
                                 cntr_name=country, end_yr=endyr,
                                 indexes=['STORAGE_TECH', 'LAYERS', 'PERIODS'],
                                 unit_dict={'Storage_In': 'GW'})

  # Sto_out: Power [GW] output from the storage in a certain period
  dm_storage_out = pyomo_var_to_dm(m, pyomo_var_name='Storage_Out',
                                  cntr_name=country, end_yr=endyr,
                                  indexes=['STORAGE_TECH', 'LAYERS', 'PERIODS'],
                                  unit_dict={'Storage_Out': 'GW'})

  # N: number of units of size ref_size which are installed
  dm_installed_n = pyomo_var_to_dm(m, pyomo_var_name='Number_Of_Units',
                                  cntr_name=country, end_yr=endyr,
                                  indexes=['TECHNOLOGIES'],
                                  unit_dict={'Number_Of_Units': '-'})

  # Total yearly emissions of the resources [ktCO2-eq./y]
  dm_GWP = pyomo_var_to_dm(m, pyomo_var_name='GWP_op', cntr_name=country,
                          end_yr=endyr,
                          indexes=['RESOURCES'],
                          unit_dict={'GWP_op': 'ktCO2-eq./y'})

  dm_f_mult_t = f_mult_t_pyomo_var_to_dm(m, cntr_name=country, end_yr=endyr)
  DM = {'installed_GW': dm_f_mult, 'installed_N': dm_installed_n,
        'emissions': dm_GWP, 'storage_in': dm_storage_in,
        'storage_out': dm_storage_out, 'monthly_operation_GW': dm_f_mult_t}
  # You now have to do 2 things, do a linear fitting for the installed technologies to cover all years,
  # look at the demand to compute the difference and use import instead.
  # Assign a portion of the installed technology to Vaud based on the relative Max potential (Nexus-e)
  # The most basic way you can assign is by allocating based on potential,
  # but the best way to do it would be based on number of installations.
  return DM


def get_ampl_output(ampl, country, endyr):

    dm_f_mult = ampl_var_to_dm(ampl, ampl_var_name='F_Mult', cntr_name=country, end_yr=endyr,
                               indexes=['TECHNOLOGIES'], unit_dict={'F_Mult': 'GW'})

    # Sto_in: Power [GW] input to the storage in a certain period
    dm_storage_in = ampl_var_to_dm(ampl, ampl_var_name='Storage_In', cntr_name=country, end_yr=endyr,
                                   indexes=['STORAGE_TECH', 'LAYERS', 'PERIODS'], unit_dict={'Storage_In': 'GW'})

    # Sto_out: Power [GW] output from the storage in a certain period
    dm_storage_out = ampl_var_to_dm(ampl, ampl_var_name='Storage_Out', cntr_name=country, end_yr=endyr,
                                    indexes=['STORAGE_TECH', 'LAYERS', 'PERIODS'], unit_dict={'Storage_Out': 'GW'})

    # N: number of units of size ref_size which are installed
    dm_installed_n = ampl_var_to_dm(ampl, ampl_var_name='Number_Of_Units', cntr_name=country, end_yr=endyr,
                                    indexes=['TECHNOLOGIES'], unit_dict={'Number_Of_Units': '-'})

    # Total yearly emissions of the resources [ktCO2-eq./y]
    dm_GWP = ampl_var_to_dm(ampl, ampl_var_name='GWP_op', cntr_name=country, end_yr=endyr,
                            indexes=['RESOURCES'], unit_dict={'GWP_op': 'ktCO2-eq./y'})

    dm_f_mult_t = f_mult_t_ampl_var_to_dm(ampl, cntr_name=country, end_yr=endyr)
    DM = {'installed_GW': dm_f_mult, 'installed_N': dm_installed_n, 'emissions': dm_GWP, 'storage_in': dm_storage_in,
          'storage_out': dm_storage_out, 'monthly_operation_GW': dm_f_mult_t}
    # You now have to do 2 things, do a linear fitting for the installed technologies to cover all years,
    # look at the demand to compute the difference and use import instead.
    # Assign a portion of the installed technology to Vaud based on the relative Max potential (Nexus-e)
    # The most basic way you can assign is by allocating based on potential,
    # but the best way to do it would be based on number of installations.
    return DM


def pyomo_param_to_dm(m, pyomo_var_name: str, cntr_name: str, end_yr: int, indexes: List[str], unit_dict: Dict[str, str], explicit=None):
    var = list(unit_dict.keys())[0]
    col_labels = {'Country': [cntr_name], 'Years': [end_yr], 'Variables': [var]}
    i = 1
    for cat_group in indexes:
        if cat_group == 'explicit':
            col_labels['Categories' + str(i)] = explicit
        else:
            col_labels['Categories' + str(i)] = list(getattr(m, cat_group))
        i = i + 1

    # Check that categories1 and categories2 do not have the same names, if so change them
    rename_dict = {}
    if len(indexes) == 2:
        common_cat = set(col_labels['Categories1']).intersection(set(col_labels['Categories2']))
        rename_dict = {k: k+'v2' for k in common_cat}
        new_cat_2 = list((set(col_labels['Categories2']) - common_cat).union(set(rename_dict.values())))
        col_labels['Categories2'] = new_cat_2

    dm = DataMatrix(col_labels, unit_dict)
    shape = tuple(len(col_labels[k]) for k in dm.dim_labels)
    arr = np.zeros(shape)
    dm.array = arr
    idx = dm.idx

    variable = getattr(m, pyomo_var_name)
    if len(indexes) == 1:
        for cat1 in variable:
          value = variable[cat1].value
          if value != 0:
              dm.array[0, 0, 0, idx[cat1]] = value
    if len(indexes) == 2:
        for cat1, cat2 in variable:
          value = variable[cat1, cat2].value
          if value != 0:
            if cat2 in rename_dict.keys():
                dm.array[0, 0, 0, idx[cat1], idx[rename_dict[cat2]]] = value
            else:
                dm.array[0, 0, 0, idx[cat1], idx[cat2]] = value
    return dm


def ampl_param_to_dm(ampl, ampl_var_name: str, cntr_name: str, end_yr: int, indexes: List[str], unit_dict: Dict[str, str], explicit=None):
    var = list(unit_dict.keys())[0]
    col_labels = {'Country': [cntr_name], 'Years': [end_yr], 'Variables': [var]}
    i = 1
    for cat_group in indexes:
        if cat_group == 'explicit':
            col_labels['Categories' + str(i)] = explicit
        else:
            col_labels['Categories' + str(i)] = [row[0] for row in ampl.get_set(cat_group).get_values()]
        i = i + 1

    # Check that categories1 and categories2 do not have the same names, if so change them
    rename_dict = {}
    if len(indexes) == 2:
        common_cat = set(col_labels['Categories1']).intersection(set(col_labels['Categories2']))
        rename_dict = {k: k+'v2' for k in common_cat}
        new_cat_2 = list((set(col_labels['Categories2']) - common_cat).union(set(rename_dict.values())))
        col_labels['Categories2'] = new_cat_2

    dm = DataMatrix(col_labels, unit_dict)
    shape = tuple(len(col_labels[k]) for k in dm.dim_labels)
    arr = np.zeros(shape)
    dm.array = arr
    idx = dm.idx

    if len(indexes) == 1:
        for cat1, value in ampl.get_parameter(ampl_var_name).get_values():
            if value != 0:
                dm.array[0, 0, 0, idx[cat1]] = value
    if len(indexes) == 2:
        for cat1, cat2, value in ampl.get_parameter(ampl_var_name).get_values():
            if value != 0:
                if cat2 in rename_dict.keys():
                    dm.array[0, 0, 0, idx[cat1], idx[rename_dict[cat2]]] = value
                else:
                    dm.array[0, 0, 0, idx[cat1], idx[cat2]] = value
    return dm
