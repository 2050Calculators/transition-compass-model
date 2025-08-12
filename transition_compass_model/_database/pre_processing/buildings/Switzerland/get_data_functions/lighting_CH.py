import pickle
import pandas as pd
from model.common.data_matrix_class import DataMatrix


def extract_EP2050_lighting_energy_consumption(file_raw, file_pickle):

  try:
    with open(file_pickle, 'rb') as handle:
      dm = pickle.load(handle)
  except OSError:
    df = pd.read_excel(file_raw, 'Tabelle1')

    df = df[list(df.columns)[1:-1]]
    df.columns = df.iloc[3]
    df = df.iloc[4:14]
    df.set_index(['Verwendungszweck'], inplace=True)
    df.columns = df.columns.astype(int)

    df = df.loc['Beleuchtung'].to_frame()
    df.reset_index(inplace=True)

    df.rename(columns={3: 'Years', 'Beleuchtung': 'bld_residential-lighting[PJ]'}, inplace=True)
    df['Country'] = 'Switzerland'

    dm = DataMatrix.create_from_df(df, num_cat=0)

    dm.change_unit('bld_residential-lighting', old_unit='PJ', new_unit='TWh', factor=3.6, operator='/')

    with open(file_pickle, 'wb') as handle:
      pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return dm
