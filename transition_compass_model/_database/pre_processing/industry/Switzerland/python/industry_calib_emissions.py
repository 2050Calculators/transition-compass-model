
import pandas as pd
import numpy as np
from model.common.data_matrix_class import DataMatrix

# get data
filepath = "../data/emissions/Evolution_GHG_since_1990_2025-04.xlsx"
def get_emissions_data(ghg):
    df = pd.read_excel(filepath, sheet_name=ghg)
    df.columns = df.iloc[3,:]
    df = df.iloc[31:32,:]
    df = pd.melt(df, id_vars = ['Cat. (1)', np.nan], var_name='Years')
    df = df.loc[:,["Years","value"]]
    df["Years"] = df["Years"].astype(int)
    df["value"] = df["value"].astype(float)
    df["ghg"] = ghg
    return df
df = pd.concat([get_emissions_data(ghg) for ghg in ["CO2","CH4","N2O"]])
# note: for now calibration is done on Mt and not MtCO2eq

# make dm
df["ghg"] = df["ghg"] + "[Mt]"
df = df.pivot(index=["Years"], columns="ghg", values='value').reset_index()
df["Country"] = "Switzerland"
dm = DataMatrix.create_from_df(df, 0)
# dm.datamatrix_plot()

# add missing years as nan
years = list(range(1990,2023+1)) + list(range(2025,2050+5,5))
missing = np.array(years)[[y not in dm.col_labels["Years"] for y in years]].tolist()
dm.add(np.nan, "Years", missing, dummy=True)
dm.sort("Years")

# format and save
for v in dm.col_labels["Variables"]: dm.rename_col(v, 'calib-emissions_' + v, "Variables")
dm.deepen()
import os
current_file_directory = os.getcwd()
f = os.path.join(current_file_directory, '../data/datamatrix/calibration_emissions.pickle')
import pickle
with open(f, 'wb') as handle:
    pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)



