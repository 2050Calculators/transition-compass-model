
import pandas as pd
import numpy as np
from model.common.data_matrix_class import DataMatrix
from model.common.auxiliary_functions import linear_fitting
import pickle
import plotly.io as pio
pio.renderers.default='browser'
import os

# 'energy-demand': DataMatrix with shape (29, 22, 1, 10), variables ['calib-energy-demand-excl-feedstock'] and 
# categories1 ['electricity', 'gas-bio', 'gas-ff-natural', 'hydrogen', 'liquid-bio', 'liquid-ff-diesel', 'liquid-ff-oil', 'solid-bio', 'solid-ff-coal', 'solid-waste']

# mapping of carriers
mapping = {"Elektrizität" : "electricity",
 "Erdgas" : "gas-ff-natural",
 "Heizöl extra-leicht" : 'liquid-ff-diesel',
 "Heizöl mittel und schwer" : 'liquid-ff-oil',
 "Industrieabfälle": 'solid-waste',
 "Kohle": "solid-ff-coal",
 "Fernwärme (Bezug)" : "electricity", # i assign district heating to electricity to be consistent on what has been done in EU
 "Holz" : "solid-bio"}
 
# note: uncovered are 'gas-bio', 'hydrogen', 'liquid-bio', I will put them to zero for now

# read data
# note: data from https://www.bfe.admin.ch/bfe/fr/home/approvisionnement/statistiques-et-geodonnees/statistiques-de-lenergie/statistiques-sectorielles.html
def get_energy_demand_data(filepath, mapping, key):
    
    df = pd.read_excel(filepath, sheet_name=key)
    df.columns = df.iloc[2,:]
    df = df.iloc[4:5,:]
    df = pd.melt(df, id_vars = ['N° de branche', 'Nom de la branche','Secteur'], var_name='Years')
    df = df.loc[:,["Years","value"]]
    df["energy-carrier"] = mapping[key]
    
    return df
filepath = "../data/energy-demand/8788-Publikationstabellen_DE_FR_IT_2013_bis_2024.xlsx"
df1 = pd.concat([get_energy_demand_data(filepath, mapping, key) for key in mapping.keys()])
filepath = "../data/energy-demand/12231-Publikationstabellen_DE_FR_IT_1999_bis_2013.xlsx"
df2 = pd.concat([get_energy_demand_data(filepath, mapping, key) for key in mapping.keys()])
df2 = df2.loc[df2["Years"] != 2013,:]
df = pd.concat([df1,df2])
df = df.groupby(["energy-carrier","Years"], as_index=False)['value'].agg("sum")
df.sort_values(["energy-carrier","Years"],inplace=True)

# fix zeros as nan
df.loc[df["value"] == 0,"value"] = np.nan

# make dm
df["energy-carrier"] = df["energy-carrier"] + "[TJ]"
df = df.pivot(index=["Years"], columns="energy-carrier", values='value').reset_index()
df["Country"] = "Switzerland"
dm = DataMatrix.create_from_df(df, 0)
dm_tot = dm.groupby({"total" : ['electricity', 'liquid-ff-diesel', 'liquid-ff-oil', 'gas-ff-natural', 'solid-bio', 'solid-ff-coal', 'solid-waste']},
                    "Variables", inplace=False)

# make shares to build missing on those, and then reconvert at the end
dm.normalise("Variables")
# dm.datamatrix_plot()

# note: for calib, we do not need to make ots and fts

# # make ots
# years_ots = list(range(1990,2024+1))
# dm = linear_fitting(dm, years_ots, min_t0=0,min_tb=0)
# # dm.datamatrix_plot()

# # make fts
# years_fts = list(range(2025,2050+5,5))
# dm = linear_fitting(dm, years_fts, min_t0=0,min_tb=0)
# # dm.datamatrix_plot()

# # make missing tot
# dm_tot = linear_fitting(dm_tot, years_ots, based_on=list(range(1999,2008+1,1)))
# # dm_tot.datamatrix_plot()
# dm_tot = linear_fitting(dm_tot, years_fts, based_on=list(range(2013,2024+1,1)))
# # dm_tot.datamatrix_plot()

# make missing absolute values per carriers
dm.array = dm.array = dm.array * dm_tot.array
for v in dm.col_labels["Variables"]: dm.units[v] = "MJ"
# dm.datamatrix_plot()

# add missing
missing = ['gas-bio', 'hydrogen', 'liquid-bio']
for m in missing:
    dm.add(0, "Variables", m, unit="MJ", dummy=True)
dm.sort("Variables")

# format and save
current_file_directory = os.getcwd()
dm.drop("Years",2024)
for v in dm.col_labels["Variables"]: dm.rename_col(v, 'calib-energy-demand-excl-feedstock_' + v, "Variables")
dm.deepen()
factor = 2.7778e-10
dm.change_unit("calib-energy-demand-excl-feedstock", factor, "MJ", "TWh")
dm.add(np.nan,"Years",list(range(1990,1998+1,1)), dummy=True)
dm.add(np.nan,"Years",list(range(2025,2050+5,5)), dummy=True)
dm.sort("Years")
dm.sort("Categories1")
dm.rename_col("calib-energy-demand-excl-feedstock", "calib-energy-demand", "Variables")
f = os.path.join(current_file_directory, '../data/datamatrix/calibration_energy-demand.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)
