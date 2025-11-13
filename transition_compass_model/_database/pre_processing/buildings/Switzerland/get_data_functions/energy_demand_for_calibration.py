import os
import requests
from model.common.data_matrix_class import DataMatrix
import pandas as pd
from _database.pre_processing.buildings.Switzerland.get_data_functions import utils

def extract_energy_statistics_data(file_url, local_filename, sheet_name,
                                   parameters, years_ots):
  mapping = parameters['mapping']  # dictionary,  to rename column headers
  var_name = parameters['var name']  # string, dm variable name
  headers_idx = parameters[
    'headers indexes']  # tuple with index of rows to keep for header
  first_row = parameters[
    'first row']  # integer with the first row to keep # regex expression with cols to drop
  unit = parameters['unit']  # Put None if unit is in table, else str
  col_to_drop = parameters[
    'cols to drop']  # None if no need to drop, else string (for dm.drop)

  if not os.path.exists(local_filename):
    response = requests.get(file_url, stream=True)
    # Check if the request was successful
    if response.status_code == 200:
      with open(local_filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
          if chunk:
            f.write(chunk)
      print(f"File downloaded successfully as {local_filename}")
    else:
      print(f"Error: {response.status_code}, {response.text}")
  else:
    print(
      f'File {local_filename} already exists. If you want to download again delete the file')

  df = utils.read_excel_with_merged_cells(local_filename, sheet_name)
  combined_headers = []
  if unit is None:
    for col1, col2, col3 in zip(df.iloc[headers_idx[0]],
                                df.iloc[headers_idx[1]],
                                df.iloc[headers_idx[2]]):
      combined_headers.append(
        str(col1) + '-' + str(col2) + '[' + str(col3) + ']')
  else:
    for col1, col2 in zip(df.iloc[headers_idx[0]], df.iloc[headers_idx[1]]):
      combined_headers.append(str(col1) + '-' + str(col2) + '[' + unit + ']')
  # Set the new header
  df.columns = combined_headers
  df = df[first_row:].copy()

  def is_valid_number(val):
    return isinstance(val, (int, float)) and not pd.isna(val)

  # Apply the function to filter out rows with no valid numeric values
  df = df[df.apply(lambda row: row.map(is_valid_number).any(), axis=1)]
  # Apply similarly for columns if needed
  df = df.loc[:, df.apply(lambda col: col.map(is_valid_number).any())]

  df.rename({df.columns[0]: 'Years'}, axis=1, inplace=True)
  df['Country'] = 'Switzerland'
  df.replace('-', 0, inplace=True)
  if col_to_drop is not None:
    df = df.drop(columns=df.filter(regex=col_to_drop).columns)
  dm = DataMatrix.create_from_df(df, num_cat=0)
  #if col_to_drop is not None:
    #dm.drop(dim='Variables', col_label=col_to_drop)

  for key in list(mapping.keys()):
    mapping[var_name + '_' + key] = mapping.pop(key)

  dm_out = dm.groupby(mapping, regex=True, dim='Variables', inplace=False)
  dm_out.deepen()
  dm_out.filter({'Years': years_ots}, inplace=True)
  # dm_out.change_unit(var_name, 277.8, 'TJ', 'MWh')

  return dm_out
