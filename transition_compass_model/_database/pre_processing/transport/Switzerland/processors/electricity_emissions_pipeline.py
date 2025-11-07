from model.common.data_matrix_class import DataMatrix
import numpy as np

def run(country_list, years_ots, years_fts):
  col_dict = {
    'Country': country_list,
    'Years': years_ots + years_fts,
    'Variables': ['tra_emission-factor'],
    'Categories1': ['CH4', 'CO2', 'N2O'],
    'Categories2': ['electricity']
  }
  dm_elec = DataMatrix(col_labels=col_dict,
                       units={'tra_emission-factor': 'g/MJ'})

  arr_elec = np.zeros((2, 40, 1, 3, 1))
  idx = dm_elec.idx
  arr_elec[:, idx[1990]: idx[2023] + 1, 0, idx['CO2'], 0] = 31.1
  arr_elec[:, idx[2025]: idx[2050], 0, idx['CO2'], 0] = np.nan
  arr_elec[:, idx[2050], 0, idx['CO2'], 0] = 0
  dm_elec.array = arr_elec
  dm_elec.fill_nans(dim_to_interp="Years")
  return dm_elec
