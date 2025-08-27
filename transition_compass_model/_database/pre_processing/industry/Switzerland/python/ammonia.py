
import numpy as np
import warnings
warnings.simplefilter("ignore")
import os
import pandas as pd
from model.common.data_matrix_class import DataMatrix
from model.common.auxiliary_functions import linear_fitting
import pickle
import plotly.io as pio
pio.renderers.default='browser'

# directories
current_file_directory = os.getcwd()

##################################
##### AMMONIA AND FERTILIZER #####
##################################

# years
years_ots = list(range(1990,2023+1))
years_fts = list(range(2025,2055,5))

# get data
filepath = os.path.join(current_file_directory, '../data/ammonia/FAOSTAT_data_en_8-22-2025.csv')
df = pd.read_csv(filepath)
df.columns
df["Item"].unique()
df = df.loc[:,["Element","Item","Year","Unit","Value"]]

# aggregate
fao_mapping = {
    # Ammonia (raw material)
    "Ammonia, anhydrous": "ammonia",
    
    # fertilizers
    "Ammonium nitrate (AN)": "fertilizer",
    "Ammonium sulphate": "fertilizer",
    "Calcium ammonium nitrate (CAN) and other mixtures with calcium carbonate": "fertilizer",
    "Diammonium phosphate (DAP)": "fertilizer",
    "Monoammonium phosphate (MAP)": "fertilizer",
    "NPK fertilizers": "fertilizer",
    "Other nitrogenous fertilizers, n.e.c.": "fertilizer",
    "Other NP compounds": "fertilizer",
    "Urea": "fertilizer",
    "Urea and ammonium nitrate solutions (UAN)": "fertilizer",
    
    # Not ammonia-based (others)
    "Fertilizers n.e.c.": "other",
    "Other phosphatic fertilizers, n.e.c.": "other",
    "Other potassic fertilizers, n.e.c.": "other",
    "Phosphate rock": "other",
    "PK compounds": "other",
    "Potassium chloride (muriate of potash) (MOP)": "other",
    "Potassium nitrate": "other",  # can involve ammonia indirectly, but generally treated as potassic
    "Potassium sulphate (sulphate of potash) (SOP)": "other",
    "Sodium nitrate": "other",
    "Superphosphates above 35%": "other",
    "Superphosphates, other": "other",
}
for key in fao_mapping.keys():
    df.loc[df["Item"] == key,"Item"] = fao_mapping[key]
df = df.groupby(["Element","Item","Year","Unit"], as_index=False)['Value'].agg("sum")
df["Item"] = df["Element"] + "_" + df["Item"] + "[" + df["Unit"] + "]"
df = df.loc[:,["Item","Year","Value"]]
df = df.pivot(index=["Year"], columns="Item", values='Value').reset_index()
df["Country"] = "Switzerland"
df.rename(columns={"Year":"Years"},inplace=True)
dm = DataMatrix.create_from_df(df, 1)
dm.rename_col(['Export quantity', 'Import quantity'], ["export","import"], "Variables")
dm.drop("Categories1","other")
# dm.flatten().datamatrix_plot()

# make ots
def linear_fitting_per_variab(dm, variable, year_start, year_end, years):
    dm_temp = dm.filter({"Variables" : [variable]})
    dm_temp = linear_fitting(dm_temp, years, based_on=list(range(year_start,year_end+1)))
    dm.drop("Variables",variable)
    dm.append(dm_temp,"Variables")

dm.add(np.nan,"Years",list(range(1990,2001+1)),dummy=True)
dm = dm.flatten()
linear_fitting_per_variab(dm, 'export_ammonia', 2002, 2002, years_ots)
linear_fitting_per_variab(dm, 'import_ammonia', 2015, 2023, years_ots)
linear_fitting_per_variab(dm, 'export_fertilizer', 2005, 2014, years_ots)
linear_fitting_per_variab(dm, 'import_fertilizer', 2002, 2023, years_ots)
# dm.datamatrix_plot()

# make fts
dm.add(np.nan,"Years",years_fts,dummy=True)
linear_fitting_per_variab(dm, 'export_ammonia', 2023, 2023, years_fts)
linear_fitting_per_variab(dm, 'import_ammonia', 2023, 2023, years_fts)
dm.array[dm[...]<0]=0
linear_fitting_per_variab(dm, 'export_fertilizer', 2005, 2014, years_fts)
linear_fitting_per_variab(dm, 'import_fertilizer', 2002, 2023, years_fts)
# dm.datamatrix_plot()
dm.deepen()

# add production
# for production between 2015-2019 (page 45.3)
# https://pubs.usgs.gov/myb/vol3/2019/myb3-2019-switzerland.pdf
dm.add(np.nan, "Variables", "production", "t", True)
dm[:,:,"production","fertilizer"] = 0 # will assume that fertilizer production is zero as export is very limited
data_n = {
    "Year": [2015, 2016, 2017, 2018, 2019],
    "Production_N_kt": [34, 34, 34, 14, 14]  # kt of nitrogen content
}
df = pd.DataFrame(data_n)
factor = 17 / 14  # molecular weight ratio
df["Production_NH3_kt"] = df["Production_N_kt"] * factor
for y in [2015, 2016, 2017, 2018, 2019]:
    dm[:,y,"production","ammonia"] = df.loc[df["Year"] == y,"Production_NH3_kt"]
for y in list(range(1990,2014+1)):
    dm[:,y,"production","ammonia"] = dm[:,2015,"production","ammonia"]
for y in list(range(2020,2023+1)):
    dm[:,y,"production","ammonia"] = dm[:,2019,"production","ammonia"]
for y in years_fts:
    dm[:,y,"production","ammonia"] = dm[:,2019,"production","ammonia"]
# dm.flatten().datamatrix_plot()

# make demand
arr_temp = dm[:,:,"production",:] + dm[:,:,"import",:] - dm[:,:,"export",:]
dm.add(arr_temp, "Variables", "demand", "t")
dm.array[dm[...]<0]=1000 # we keep ammonia demand to 1000 (level of 2018)
# dm.flatten().datamatrix_plot()

# make product net import share
dm_temp = dm.filter({"Variables" : ['export', 'import', 'demand']})
arr_temp = (dm_temp[:,:,"import",:] - dm_temp[:,:,"export",:])/dm_temp[:,:,"demand",:]
dm_temp.add(arr_temp, "Variables", "net-import", "%")
dm_amm_prod_net_import = dm_temp.filter({"Variables" : ['net-import'], "Categories1" : ["fertilizer"]})
dm_amm_prod_net_import.rename_col("net-import", "product-net-import", "Variables")
# dm_amm_prod_net_import.flatten().datamatrix_plot()

# make material net import share
dm_amm_mat_net_import = dm_temp.filter({"Variables" : ['net-import'], "Categories1" : ["ammonia"]})
dm_amm_mat_net_import.rename_col("net-import", "material-net-import", "Variables")
# dm_amm_mat_net_import.flatten().datamatrix_plot()

# make material production
dm_amm_prod = dm.filter({"Variables" : ["production"], "Categories1" : ["ammonia"]})
dm_amm_prod.rename_col("production", "material-production", "Variables")
# dm_amm_prod.flatten().datamatrix_plot()

# save
dm_ots = dm_amm_prod_net_import.filter({"Years" : years_ots})
dm_fts = dm_amm_prod_net_import.filter({"Years" : years_fts})
DM_fts = {1: dm_fts.copy(), 2: dm_fts.copy(), 3: dm_fts.copy(), 4: dm_fts.copy()} # for now we set all levels to be the same
DM = {"ots" : dm_ots,
      "fts" : DM_fts}
f = os.path.join(current_file_directory, '../data/datamatrix/lever_product-net-import_ammonia.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
dm_ots = dm_amm_mat_net_import.filter({"Years" : years_ots})
dm_fts = dm_amm_mat_net_import.filter({"Years" : years_fts})
DM_fts = {1: dm_fts.copy(), 2: dm_fts.copy(), 3: dm_fts.copy(), 4: dm_fts.copy()} # for now we set all levels to be the same
DM = {"ots" : dm_ots,
      "fts" : DM_fts}
f = os.path.join(current_file_directory, '../data/datamatrix/lever_material-net-import_ammonia.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)

dm_amm_prod_cal = dm_amm_prod.filter({"Years" : [2015,2016,2017,2018,2019]})
years = list(range(1990,2023+1)) + list(range(2025,2050+5,5))
missing = np.array(years)[[y not in dm_amm_prod_cal.col_labels["Years"] for y in years]].tolist()
dm_amm_prod_cal.add(np.nan, "Years", missing, dummy=True)
dm_amm_prod_cal.sort("Years")
dm_amm_prod_cal.change_unit("material-production", 1e-3, "t", "kt")
f = os.path.join(current_file_directory, '../data/datamatrix/calibration_material-production_ammonia.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_amm_prod_cal, handle, protocol=pickle.HIGHEST_PROTOCOL)
