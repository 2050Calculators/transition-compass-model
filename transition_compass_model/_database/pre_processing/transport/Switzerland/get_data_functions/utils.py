from model.common.auxiliary_functions import linear_fitting
import pandas as pd
from model.common.data_matrix_class import DataMatrix
import numpy as np

def fill_var_nans_based_on_var_curve(dm, var_nan, var_ref, keep_all_vars=False):
  # Fills nan in a variable dm based on another variable curve
  dm.operation(var_nan, '/', var_ref, out_col='ratio', unit='%')
  linear_fitting(dm, dm.col_labels['Years'])
  dm.operation('ratio', '*', var_ref, out_col=var_nan + '_ok',
               unit=dm.units[var_ref])
  if not keep_all_vars:
    dm.filter({'Variables': [var_nan + '_ok']}, inplace=True)
  else:
    dm.drop(dim='Variables', col_label=[var_nan, 'ratio'])
  dm.rename_col(var_nan + '_ok', var_nan, dim='Variables')

  return dm

def df_fso_excel_to_dm(df, header_row, names_dict, var_name, unit, num_cat,
                       keep_first=False, country='Switzerland'):
  # Federal statistical office df from excel to dm
  # Change headers
  new_header = df.iloc[header_row]
  new_header.values[0] = 'Variables'
  df.columns = new_header
  df = df[header_row + 1:].copy()
  # Remove nans and empty columns/rows
  if np.nan in df.columns:
    df.drop(columns=np.nan, inplace=True)
  df.set_index('Variables', inplace=True)
  df.dropna(axis=0, how='all', inplace=True)
  df.dropna(axis=1, how='all', inplace=True)
  # Filter rows that contain at least one number (integer or float)
  df = df[
    df.apply(lambda row: row.map(pd.api.types.is_number), axis=1).any(axis=1)]
  df_clean = df.loc[:,
             df.apply(lambda col: col.map(pd.api.types.is_number)).any(
               axis=0)].copy()
  # Extract only the data we are interested in:
  df_filter = df_clean.loc[names_dict.keys()].copy()
  df_filter = df_filter.apply(lambda col: pd.to_numeric(col, errors='coerce'))
  # df_filter = df_filter.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
  df_filter.reset_index(inplace=True)
  # Keep only first 10 caracters
  df_filter['Variables'] = df_filter['Variables'].replace(names_dict)
  if keep_first:
    df_filter = df_filter.drop_duplicates(subset=['Variables'], keep='first')
  df_filter = df_filter.groupby(['Variables']).sum()
  df_filter.reset_index(inplace=True)

  # Pivot the dataframe
  df_filter['Country'] = country
  df_T = pd.melt(df_filter, id_vars=['Variables', 'Country'], var_name='Years',
                 value_name='values')
  df_pivot = df_T.pivot_table(index=['Country', 'Years'], columns=['Variables'],
                              values='values', aggfunc='sum')
  df_pivot = df_pivot.add_suffix('[' + unit + ']')
  df_pivot = df_pivot.add_prefix(var_name + '_')
  df_pivot.reset_index(inplace=True)

  # Drop non numeric values in Years col
  df_pivot['Years'] = pd.to_numeric(df_pivot['Years'], errors='coerce')
  df_pivot = df_pivot.dropna(subset=['Years'])

  dm = DataMatrix.create_from_df(df_pivot, num_cat=num_cat)
  return dm
