
from model.common.data_matrix_class import DataMatrix
from model.common.constant_data_matrix_class import ConstantDataMatrix
from model.common.io_database import read_database, read_database_fxa, update_database_from_db
from model.common.interface_class import Interface
from model.common.auxiliary_functions import filter_geoscale, cdm_to_dm, read_level_data, simulate_input, calibration_rates, cost, create_years_list
import pandas as pd
import pickle
import json
import os
import numpy as np
import re
import warnings
from tqdm import tqdm
import eurostat
from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat
warnings.simplefilter("ignore")

__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/industry/eu/industry_preprocessing_EU.py"

# directories
current_file_directory = os.path.dirname(os.path.abspath(__file__))
xls_directory = os.path.join(current_file_directory, "data")

###########################################################
#################### FIXED ASSUMPTIONS ####################
###########################################################

#################
##### COSTS #####
#################

# costs
file = "costs.xlsx"
xls_file_directory = xls_directory +  "/" + file
df_costs = pd.read_excel(xls_file_directory)

# price index
file = "costs_priceindex.xlsx"
xls_file_directory = xls_directory +  "/" + file
df_price = pd.read_excel(xls_file_directory)

# add extra countries in price index
df_price_extra = pd.DataFrame({"Country" : ["EU27", "EU28","Paris", "Vaud"], 
                               "Years" : [2015,2015,2015,2015], 
                               "ots_tec_price-indices_pli[%]" : [100,100, 107.6, 154.0]})
df_price = pd.concat([df_price, df_price_extra]).reset_index(level=0, drop=True)
df_price = df_price.sort_values("Country")

# add other countries to costs
countries = df_price["Country"].tolist()
countries.remove("EU28")
df_costs_new = df_costs.copy()
for c in countries:
    df_temp = df_costs.copy()
    df_temp["Country"] = c
    df_costs_new = pd.concat([df_costs_new, df_temp]).reset_index(level=0, drop=True)

# merge with price data
df_costs_new = pd.merge(df_costs_new, df_price, how="left", on=["Country","Years"])

# get new capex and opex 2050 and baseyear by weighting for price index
variabs = ["capex_2050","capex_baseyear","opex_2050","opex_baseyear"]
for v in variabs:
    df_costs_new[v] = df_costs_new[v] * df_costs_new["ots_tec_price-indices_pli[%]"] / 100

# b factor = log(1 - linear rate)/log(2)
df_costs_new["capex_b_factor"] = np.log(1 - df_costs_new["capex_lr"]) / np.log(2)
df_costs_new["opex_b_factor"] = np.log(1 - df_costs_new["opex_lr"]) / np.log(2)

# d factor = (cost 2050 - cost 2015) / 2050 - 2015
df_costs_new["capex_d_factor"] = (df_costs_new["capex_2050"] - df_costs_new["capex_baseyear"]) / (2050 - 2015)
df_costs_new["opex_d_factor"] = (df_costs_new["opex_2050"] - df_costs_new["opex_baseyear"]) / (2050 - 2015)

# long format
df = df_costs_new.copy()
df1 = df.loc[df["Country"] != "EU28",:]
indexes = ["Country","Years", "sector", "technology_code","capex_unit"
           #"opex_unit"
           ]
variabs = ['capex_b_factor', 'capex_baseyear', 'capex_d_factor', 
           # 'opex_b_factor','opex_baseyear', 'opex_d_factor',
           'evolution_method']
df1 = df1.loc[:,indexes + variabs]
df1 = pd.melt(df1, id_vars = indexes)

# drop na
df1 = df1.dropna(subset=["value"])

# make variables
df1 = df1.rename(columns={"Country": "geoscale", "Years": "timescale"})
df1["module"] = "cost"
df1["element"] = df1["variable"]
df1["element"] = [i.replace("_", "-") for i in df1["element"].values.tolist()]
df1["item"] = df1["technology_code"]
# df1["unit"] = "num"
df1.loc[df1["element"] == "capex-baseyear","unit"] = df1.loc[df1["element"] == "capex-baseyear","capex_unit"]
df1.loc[df1["element"].isin(['capex-d-factor', 'evolution-method']),"unit"] = "num"
df1["eucalc-name"] = df1["sector"] + "_" + df1["element"] + "_" + df1["item"] + "[" + df1["unit"] + "]"
df1["lever"] = "none"
df1["level"] = 0
df1["string-pivot"] = "none"
df1["type-prefix"] = "none"
df1["module-prefix"] = "cost-" + df1["sector"]
df1["reference-id"] = "missing-reference"
df1["interaction-file"] = "cost_fixed-assumptions"

# drop extra variables
df1 = df1.drop(['sector','technology_code','capex_unit','variable'], axis = 1)

# order and sort
df1 = df1.loc[:,['geoscale', "timescale", "module", "eucalc-name", "lever", "level", "string-pivot",
                  "type-prefix", "module-prefix", "element", "item", "unit", "value", "reference-id", 
                  "interaction-file"]]
df1 = df1.sort_values(by=['geoscale', "timescale", "module", "eucalc-name"])

# # check for doubles
# indexes = ['geoscale', "timescale", "module", "eucalc-name", "lever", "level", "string-pivot",
#            "type-prefix", "module-prefix", "element", "item", "unit", "reference-id", 
#            "interaction-file"]
# A = pd.DataFrame({'geoscale' : [str(i) for i in df1.groupby(indexes)['value'].agg(len).index], 
#                    'value_count' : df1.groupby(indexes)['value'].agg(len).values
#                   })
# A = A[A['value_count']>1]
# A.shape # OK A has 0 rows

# save
filepath = os.path.join(current_file_directory, '../../../data/csv/costs_fixed-assumptions.csv')
df1.to_csv(filepath, sep = ";", index = False)

# get countries
countries = df1["geoscale"].unique().tolist()

# clean
del c, df, df1, df_costs, df_costs_new, df_price, df_price_extra, df_temp, file,\
    filepath, indexes, v, variabs, xls_file_directory

####################################################
##### MATERIAL DECOMPOSITION OF WASTE PRODUCTS #####
####################################################

# get data
file = "waste-product-material-content.xlsx"
xls_file_directory = xls_directory +  "/" + file
df_matdec = pd.read_excel(xls_file_directory)

# long format
indexes = ["Country","Years"]
df1 = pd.melt(df_matdec, id_vars = indexes)

# fix countries
df1 = df1.sort_values(by=['Country'])
np.array(countries)
df1["Country"].unique()
df1.loc[df1["Country"] == "Czechia","Country"] = "Czech Republic"
df1 = df1.loc[df1["Country"] != "Iceland",:]
df1 = df1.loc[df1["Country"] != "Liechtenstein",:]
df1 = df1.loc[df1["Country"] != "Norway",:]
df_temp = df1.loc[df1["Country"] == "Germany",:]
df_temp.loc[:,"Country"] = "Paris"
df1 = pd.concat([df1, df_temp])
df_temp = df1.loc[df1["Country"] == "Germany",:]
df_temp.loc[:,"Country"] = "Vaud"
df1 = pd.concat([df1, df_temp])

# set([i.split("_")[0] for i in df5["variable"]])
# set([i.split("_")[1] for i in df5["variable"]])
# set([i.split("_")[2] for i in df5["variable"]])
# set([i.split("_")[3] for i in df1["variable"]])

# fix variable's names
df1["variable"] = [i.replace("larger-appliances_","domapp_") for i in df1["variable"]]
df1["variable"] = [i.replace("composition_","") for i in df1["variable"]]
df1["variable"] = [i.replace("unit","num") for i in df1["variable"]]

# electronics
df2 = df1.loc[[i.split("_")[0] == "electronics" for i in df1["variable"]],:]
df2["element"] = [i.split("_")[1] for i in df2["variable"]]
items = [i.split("_")[2] for i in df2["variable"]]
df2["item"] = [i.split("[")[0] for i in items]
units = [i.split("[")[1] for i in df2["variable"]]
df2["unit"] = [i.split("]")[0] for i in units]

# vehicles
df3 = df1.loc[[i.split("_")[0] in ['hdvh', 'hdvl', 'hdvm', 'ldv'] for i in df1["variable"]],:]
df3["variable"] = [i.replace("hdvh_","trucks-high-") for i in df3["variable"]]
df3["variable"] = [i.replace("hdvl_","trucks-low-") for i in df3["variable"]]
df3["variable"] = [i.replace("hdvm_","trucks-medium-") for i in df3["variable"]]
df3["variable"] = [i.replace("ldv_","cars-") for i in df3["variable"]]
df3["variable"] = [i.replace("bev","BEV") for i in df3["variable"]]
df3["variable"] = [i.replace("fcev","FCEV") for i in df3["variable"]]
df3["variable"] = [i.replace("ice","ICE") for i in df3["variable"]]
df3["variable"] = [i.replace("phev","PHEV") for i in df3["variable"]]
df3["element"] = [i.split("_")[0] for i in df3["variable"]]
items = [i.split("_")[1] for i in df3["variable"]]
df3["item"] = [i.split("[")[0] for i in items]
units = [i.split("[")[1] for i in df3["variable"]]
df3["unit"] = [i.split("]")[0] for i in units]

# batteries
df4 = df1.loc[[i.split("_")[0] in ['lib'] for i in df1["variable"]],:]
materials = [i.split("_")[-1] for i in df4["variable"]]
materials = list(set([i.split("[")[0] for i in materials]))
df4["variable"] = [i.replace("lib_","batteries-") for i in df4["variable"]]
df4["variable"] = [i.replace("hdvh_","trucks-high-") for i in df4["variable"]]
df4["variable"] = [i.replace("hdvl_","trucks-low-") for i in df4["variable"]]
df4["variable"] = [i.replace("hdvm_","trucks-medium-") for i in df4["variable"]]
df4["variable"] = [i.replace("ldv_","cars-") for i in df4["variable"]]
df4["variable"] = [i.replace("bev","BEV") for i in df4["variable"]]
df4["variable"] = [i.replace("fcev","FCEV") for i in df4["variable"]]
df4["variable"] = [i.replace("ice","ICE") for i in df4["variable"]]
df4["variable"] = [i.replace("phev","PHEV") for i in df4["variable"]]
search = ["batteries-" + i for i in materials]
new = ["batteries_" + i for i in materials]
for s in range(len(search)):
    df4["variable"] = [i.replace(search[s],new[s]) for i in df4["variable"]]
df4["element"] = [i.split("_")[0] for i in df4["variable"]]
df4["element"] = [i.replace("batteries-","batteries_") for i in df4["element"]]
items = [i.split("_")[1] for i in df4["variable"]]
df4["item"] = [i.split("[")[0] for i in items]
units = [i.split("[")[1] for i in df4["variable"]]
df4["unit"] = [i.split("]")[0] for i in units]

# domestic appliances
df5 = df1.loc[[i.split("_")[0] == "domapp" for i in df1["variable"]],:]
df5["variable"] = [i.replace("domapp_","") for i in df5["variable"]]
df5["element"] = [i.split("_")[0] for i in df5["variable"]]
items = [i.split("_")[1] for i in df5["variable"]]
df5["item"] = [i.split("[")[0] for i in items]
units = [i.split("[")[1] for i in df5["variable"]]
df5["unit"] = [i.split("]")[0] for i in units]

# put together
df1 = pd.concat([df2, df3, df4, df5]).reset_index(level=0, drop=True)
df1 = df1.rename(columns={"variable": "eucalc-name"})
df1["eucalc-name"] = df1["element"] + "_" + df1["item"] + "[" + df1["unit"] + "]"
df1 = df1.rename(columns={"Country": "geoscale", "Years": "timescale"})
df1["module"] = "industry-endoflife"
df1["lever"] = "none"
df1["level"] = 0
df1["string-pivot"] = "none"
df1["type-prefix"] = "none"
df1["module-prefix"] = "ind-eol"
df1["reference-id"] = "missing-reference"
df1["interaction-file"] = "ind-eol_fixed-assumptions"

# # make variables
# df1 = df1.rename(columns={"Country": "geoscale", "Years": "timescale"})
# df1["module"] = "industry-endoflife"
# df1["element"] = [i.split("_")[0] for i in df1["variable"]]
# items = [i.split("_")[2] for i in df1["variable"]]
# df1["item"] = [i.split("[")[0] for i in items]
# units = [i.split("[")[1] for i in items]
# units = [i.split("]")[0] for i in units]
# df1["unit"] = [i.replace("unit","num") for i in units]
# df1 = df1.rename(columns={"variable": "eucalc-name"})
# df1["eucalc-name"] = ["ind_" + i.replace("composition_","") for i in df1["eucalc-name"]]
# df1["lever"] = "none"
# df1["level"] = 0
# df1["string-pivot"] = "none"
# df1["type-prefix"] = "none"
# df1["module-prefix"] = "ind-eol"
# df1["reference-id"] = "missing-reference"
# df1["interaction-file"] = "ind-eol_fixed-assumptions"

# # add countries
# countries = np.array(countries)[[i not in "EU27" for i in countries]].tolist()
# for c in tqdm(range(len(countries))):
#     df_temp = df1.loc[df1["geoscale"] == "EU27",:]
#     df_temp["geoscale"] = countries[c]
#     df1 = pd.concat([df1, df_temp]).reset_index(level=0, drop=True)

# # order and sort
# df1 = df1.loc[:,['geoscale', "timescale", "module", "eucalc-name", "lever", "level", "string-pivot",
#                   "type-prefix", "module-prefix", "element", "item", "unit", "value", "reference-id", 
#                   "interaction-file"]]
# df1 = df1.sort_values(by=['geoscale', "timescale", "module", "eucalc-name"])

# # check for doubles
# indexes = ['geoscale', "timescale", "module", "eucalc-name", "lever", "level", "string-pivot",
#             "type-prefix", "module-prefix", "element", "item", "unit", "reference-id", 
#             "interaction-file"]
# A = pd.DataFrame({'geoscale' : [str(i) for i in df1.groupby(indexes)['value'].agg(len).index], 
#                     'value_count' : df1.groupby(indexes)['value'].agg(len).values
#                   })
# A = A[A['value_count']>1]
# A.shape # OK A has 0 rows

# save
filepath = os.path.join(current_file_directory, '../../../data/csv/eol_fixed-assumptions.csv')
df1.to_csv(filepath, sep = ";", index = False)

# clean
del df1, df_matdec, file, indexes, items, units, xls_file_directory, filepath,\
    df2, df3, df4, df5, materials, new, search, df_temp

################################################
#################### LEVERS ####################
################################################

############################
##### WASTE MANAGEMENT #####
############################

eol_xls_directory = os.path.join(xls_directory, "eol_intermediate_files")
df_pc = pd.read_excel(eol_xls_directory + '/pc.xlsx') # computers & electronics
df_LA = pd.read_excel(eol_xls_directory + '/LA.xlsx') # larger appliances
df_tv = pd.read_excel(eol_xls_directory + '/tv.xlsx') # TV&PV
df_elv = pd.read_excel(eol_xls_directory + '/elv.xlsx') # end-of-life vehicles

# # checks
# subset1 = ["littered[%]","export[%]","waste-collected[%]","waste-uncollected[%]"]
# subset2 = ['recycling[%]', 'energy-recovery[%]','reuse[%]', 'landfill[%]', 'incineration[%]']
# def check(df, indexes, subset):

#     df_temp = pd.melt(df, indexes)
#     df_temp = df_temp.sort_values(indexes + ["variable"])
#     df_temp2 = df_temp.loc[df_temp["variable"].isin(subset),:]
#     df_temp2 = df_temp2.groupby(indexes, as_index=False)['value'].agg(sum)
    
#     return print(df_temp2["value"].unique())

# indexes = ["geoscale","timescale"]
# check(df_pc, indexes, subset1)
# check(df_pc, indexes, subset2)
# check(df_LA, indexes, subset1)
# check(df_LA, indexes, subset2)
# check(df_tv, indexes, subset1)
# check(df_tv, indexes, subset2)
# check(df_elv, indexes, subset1)
# check(df_elv, indexes, subset2)

# TODO: fix the zeroes (when things are all zero, it should be that one is 1)

# put in one dataset
indexes = ["geoscale","timescale"]
df_temp = pd.melt(df_pc, id_vars = indexes)
df_temp1 = df_temp.copy()
df_temp["variable"] = ["pc_" + i for i in df_temp["variable"]]
df_temp1["variable"] = ["phone_" + i for i in df_temp1["variable"]]
df_wst = pd.concat([df_temp, df_temp1])
df_temp = pd.melt(df_LA, id_vars = indexes)
df_temp["variable"] = ["domapp_" + i for i in df_temp["variable"]]
df_wst = pd.concat([df_wst, df_temp])
df_temp = pd.melt(df_tv, id_vars = indexes)
df_temp["variable"] = ["tv_" + i for i in df_temp["variable"]]
df_wst = pd.concat([df_wst, df_temp])
df_temp = pd.melt(df_elv, id_vars = indexes)
df_temp["variable"] = ["elv_" + i for i in df_temp["variable"]]
df_wst = pd.concat([df_wst, df_temp])
df_wst = df_wst.rename(columns={"geoscale": "Country","timescale" : "Years"})

# fix countries
df_wst = df_wst.sort_values(by=['Country'])
np.array(countries)
df_wst["Country"].unique()
df_wst.loc[df_wst["Country"] == "Czechia","Country"] = "Czech Republic"
df_wst = df_wst.loc[df_wst["Country"] != "Iceland",:]
df_wst = df_wst.loc[df_wst["Country"] != "Liechtenstein",:]
df_wst = df_wst.loc[df_wst["Country"] != "Norway",:]
df_wst.loc[df_wst["Country"] == "UK","Country"] = "United Kingdom"
df_temp = df_wst.loc[df_wst["Country"] == "Germany",:]
df_temp.loc[:,"Country"] = "Paris"
df_wst = pd.concat([df_wst, df_temp])
df_temp = df_wst.loc[df_wst["Country"] == "Germany",:]
df_temp.loc[:,"Country"] = "Vaud"
df_wst = pd.concat([df_wst, df_temp])
df_wst = df_wst.loc[df_wst["Country"] != "Türkiye",]
df_wst = df_wst.sort_values(by=["variable","Country","Years"])

# expand df to include missing years
countries = df_wst["Country"].unique()
years = range(2025,2055,5)
variables = df_wst["variable"].unique()
levels = [1,2,3,4]
panel_countries = np.repeat(countries, len(variables) * len(years) * 4)
panel_years = np.tile(np.tile(np.repeat(years, len(levels)), len(variables)), len(countries))
panel_variables = np.tile(np.repeat(variables, len(years) * len(levels)), len(countries))
panel_levels = np.tile(np.tile(np.tile(levels, len(years)), len(variables)), len(countries))
df_temp = pd.DataFrame({"Country" : panel_countries, 
                        "Years" : panel_years, 
                        "variable" : panel_variables,
                        "level" : panel_levels})
years = range(1990,2025,1)
panel_countries = np.repeat(countries, len(variables) * len(years))
panel_years = np.tile(np.tile(years, len(variables)), len(countries))
panel_variables = np.tile(np.repeat(variables, len(years)), len(countries))
df_temp1 = pd.DataFrame({"Country" : panel_countries, 
                        "Years" : panel_years, 
                        "variable" : panel_variables,
                        "level" : 0})
df_temp = pd.concat([df_temp1, df_temp])
df_temp = df_temp.sort_values(["Country","variable","Years","level"])
df_wst = pd.merge(df_temp, df_wst, how="left", on=["Country","Years","variable"])



















# get data
file = "ind_eol_waste-management.xlsx"
xls_file_directory = xls_directory +  "/" + file
df_lever1 = pd.read_excel(xls_file_directory)

# long format
indexes = ["Country","Years","lever_eol"]
df1 = pd.melt(df_lever1, id_vars = indexes)

# ["recycling","energy-recovery","reuse","landfill","incineration"]
# variables = ["ldv_bev_energy-recovery[%]","ldv_bev_recycling[%]","ldv_bev_reuse[%]","ldv_bev_landfill[%]","ldv_bev_incineration[%]"]
# df2 = df1.loc[df1["variable"].isin(variables),:]
# indexes = ["Country","Years","lever_eol"]
# A = df2.groupby(indexes)['value'].agg(sum)

# fix countries
df1 = df1.sort_values(by=['Country'])
np.array(countries)
df1["Country"].unique()
df1.loc[df1["Country"] == "Czechia","Country"] = "Czech Republic"
df1 = df1.loc[df1["Country"] != "Iceland",:]
df1 = df1.loc[df1["Country"] != "Liechtenstein",:]
df1 = df1.loc[df1["Country"] != "Norway",:]
df1.loc[df1["Country"] == "UK","Country"] = "United Kingdom"
df_temp = df1.loc[df1["Country"] == "Germany",:]
df_temp.loc[:,"Country"] = "Paris"
df1 = pd.concat([df1, df_temp])
df_temp = df1.loc[df1["Country"] == "Germany",:]
df_temp.loc[:,"Country"] = "Vaud"
df1 = pd.concat([df1, df_temp])
df1 = df1.loc[df1["Country"] != "Türkiye",]
df1 = df1.sort_values(by=["variable","Country","Years","lever_eol"])

# set([i.split("_")[0] for i in df1["variable"]])
# set([i.split("_")[1] for i in df1["variable"]])
# set([i.split("_")[2] for i in df1["variable"]])
# set([i.split("_")[3] for i in df1["variable"]])

# fix variable's names
df1["variable"] = [i.replace("larger-appliances_","domapp_") for i in df1["variable"]]

# electronics
df2 = df1.loc[[i.split("_")[0] == "electronics" for i in df1["variable"]],:]
df2["element"] = [i.split("_")[1] for i in df2["variable"]]
items = [i.split("_")[2] for i in df2["variable"]]
df2["item"] = [i.split("[")[0] for i in items]
units = [i.split("[")[1] for i in df2["variable"]]
df2["unit"] = [i.split("]")[0] for i in units]

# vehicles
df3 = df1.loc[[i.split("_")[0] in ['hdvh', 'hdvl', 'hdvm', 'ldv'] for i in df1["variable"]],:]
df3["variable"] = [i.replace("hdvh_","trucks-high-") for i in df3["variable"]]
df3["variable"] = [i.replace("hdvl_","trucks-low-") for i in df3["variable"]]
df3["variable"] = [i.replace("hdvm_","trucks-medium-") for i in df3["variable"]]
df3["variable"] = [i.replace("ldv_","cars-") for i in df3["variable"]]
df3["variable"] = [i.replace("bev","BEV") for i in df3["variable"]]
df3["variable"] = [i.replace("fcev","FCEV") for i in df3["variable"]]
df3["variable"] = [i.replace("ice","ICE") for i in df3["variable"]]
df3["variable"] = [i.replace("phev","PHEV") for i in df3["variable"]]
df3["element"] = [i.split("_")[0] for i in df3["variable"]]
items = [i.split("_")[1] for i in df3["variable"]]
df3["item"] = [i.split("[")[0] for i in items]
units = [i.split("[")[1] for i in df3["variable"]]
df3["unit"] = [i.split("]")[0] for i in units]

# domestic appliances
df5 = df1.loc[[i.split("_")[0] == "domapp" for i in df1["variable"]],:]
df5["variable"] = [i.replace("domapp_","") for i in df5["variable"]]
df5["element"] = [i.split("_")[0] for i in df5["variable"]]
items = [i.split("_")[1] for i in df5["variable"]]
df5["item"] = [i.split("[")[0] for i in items]
units = [i.split("[")[1] for i in df5["variable"]]
df5["unit"] = [i.split("]")[0] for i in units]

# put together
df1 = pd.concat([df2, df3, df5]).reset_index(level=0, drop=True)
df1 = df1.rename(columns={"variable": "eucalc-name"})
df1["eucalc-name"] = df1["element"] + "_" + df1["item"] + "[" + df1["unit"] + "]"
df1 = df1.rename(columns={"Country": "geoscale", "Years": "timescale"})
df1["module"] = "industry-endoflife"
df1["lever"] = "eol-waste-management"
df1 = df1.rename(columns={"lever_eol": "level"})
df1["string-pivot"] = "none"
df1["type-prefix"] = "none"
df1["module-prefix"] = "ind-eol"
df1["reference-id"] = "missing-reference"
df1["interaction-file"] = "eol_levers"

# order and sort
df1 = df1.loc[:,['geoscale', "timescale", "module", "eucalc-name", "lever", "level", "string-pivot",
                  "type-prefix", "module-prefix", "element", "item", "unit", "value", "reference-id", 
                  "interaction-file"]]
df1 = df1.sort_values(by=['geoscale', "timescale", "module", "eucalc-name"])

# # check for doubles
# indexes = ['geoscale', "timescale", "module", "eucalc-name", "lever", "level", "string-pivot",
#             "type-prefix", "module-prefix", "element", "item", "unit", "reference-id", 
#             "interaction-file"]
# A = pd.DataFrame({'geoscale' : [str(i) for i in df1.groupby(indexes)['value'].agg(len).index], 
#                     'value_count' : df1.groupby(indexes)['value'].agg(len).values
#                   })
# A = A[A['value_count']>1]
# A.shape # OK A has 0 rows

# save
filepath = os.path.join(current_file_directory, '../../../data/csv/eol_waste-management.csv')
df1.to_csv(filepath, sep = ";", index = False)

# clean
del df1, df2, df3, df5, df_lever1, file, filepath, indexes, items, units,\
    xls_file_directory


#############################
##### MATERIAL RECOVERY #####
#############################

# get data
file = "ind_eol_materoial-recovery.xlsx"
xls_file_directory = xls_directory +  "/" + file
df_lever2 = pd.read_excel(xls_file_directory)

# long format
indexes = ["Country","Years","lever_eol_recovery"]
df1 = pd.melt(df_lever2, id_vars = indexes)

# fix countries
df1.loc[df1["Country"] == "Czechia","Country"] = "Czech Republic"
df1 = df1.loc[df1["Country"] != "Iceland",:]
df1 = df1.loc[df1["Country"] != "Liechtenstein",:]
df1 = df1.loc[df1["Country"] != "Norway",:]
df_temp = df1.loc[df1["Country"] == "Germany",:]
df_temp.loc[:,"Country"] = "Paris"
df1 = pd.concat([df1, df_temp])
df_temp = df1.loc[df1["Country"] == "Germany",:]
df_temp.loc[:,"Country"] = "Vaud"
df1 = pd.concat([df1, df_temp])

# # issues
# set([i.split("_")[0] for i in df1["variable"]])
# df2 = df1.loc[[i.split("_")[0] == "elv" for i in df1["variable"]],:]
# df2 = df1.loc[[i.split("_")[0] == "hdvh" for i in df1["variable"]],:]
# df2 = df1.loc[[i.split("_")[1] == "hdvh" for i in df1["variable"]],:]

# drop elv
df1 = df1.loc[[i.split("_")[0] != "elv" for i in df1["variable"]],:]

# drop material-recovery from names
df1["variable"] = [i.replace("_material-recovery","") for i in df1["variable"]]

# electronics
df2 = df1.loc[[i.split("_")[0] == "electronics" for i in df1["variable"]],:]
df2["element"] = [i.split("_")[1] for i in df2["variable"]]
items = [i.split("_")[2] for i in df2["variable"]]
df2["item"] = [i.split("[")[0] for i in items]
units = [i.split("[")[1] for i in df2["variable"]]
df2["unit"] = [i.split("]")[0] for i in units]

# vehicles
df3 = df1.loc[[i.split("_")[0] in ['hdvh', 'hdvl', 'hdvm', 'ldv'] for i in df1["variable"]],:]
df3["variable"] = [i.replace("hdvh_","trucks-high-") for i in df3["variable"]]
df3["variable"] = [i.replace("hdvl_","trucks-low-") for i in df3["variable"]]
df3["variable"] = [i.replace("hdvm_","trucks-medium-") for i in df3["variable"]]
df3["variable"] = [i.replace("ldv_","cars-") for i in df3["variable"]]
df3["variable"] = [i.replace("bev","BEV") for i in df3["variable"]]
df3["variable"] = [i.replace("fcev","FCEV") for i in df3["variable"]]
df3["variable"] = [i.replace("ice","ICE") for i in df3["variable"]]
df3["variable"] = [i.replace("phev","PHEV") for i in df3["variable"]]
df3["element"] = [i.split("_")[0] for i in df3["variable"]]
items = [i.split("_")[1] for i in df3["variable"]]
df3["item"] = [i.split("[")[0] for i in items]
units = [i.split("[")[1] for i in df3["variable"]]
df3["unit"] = [i.split("]")[0] for i in units]

# batteries
df4 = df1.loc[[i.split("_")[0] in ['lib'] for i in df1["variable"]],:]
materials = [i.split("_")[-1] for i in df4["variable"]]
materials = list(set([i.split("[")[0] for i in materials]))
df4["variable"] = [i.replace("lib_","batteries-") for i in df4["variable"]]
df4["variable"] = [i.replace("hdvh_","trucks-high-") for i in df4["variable"]]
df4["variable"] = [i.replace("hdvl_","trucks-low-") for i in df4["variable"]]
df4["variable"] = [i.replace("hdvm_","trucks-medium-") for i in df4["variable"]]
df4["variable"] = [i.replace("ldv_","cars-") for i in df4["variable"]]
df4["variable"] = [i.replace("bev","BEV") for i in df4["variable"]]
df4["variable"] = [i.replace("fcev","FCEV") for i in df4["variable"]]
df4["variable"] = [i.replace("ice","ICE") for i in df4["variable"]]
df4["variable"] = [i.replace("phev","PHEV") for i in df4["variable"]]
search = ["batteries-" + i for i in materials]
new = ["batteries_" + i for i in materials]
for s in range(len(search)):
    df4["variable"] = [i.replace(search[s],new[s]) for i in df4["variable"]]
df4["element"] = [i.split("_")[0] for i in df4["variable"]]
df4["element"] = [i.replace("batteries-","batteries_") for i in df4["element"]]
items = [i.split("_")[1] for i in df4["variable"]]
df4["item"] = [i.split("[")[0] for i in items]
units = [i.split("[")[1] for i in df4["variable"]]
df4["unit"] = [i.split("]")[0] for i in units]

# domestic appliances
df5 = df1.loc[[i.split("_")[0] == "larger-appliances" for i in df1["variable"]],:]
df5["variable"] = [i.replace("larger-appliances_","") for i in df5["variable"]]
df5 = df5.loc[df5["variable"] != 'dishwasher_total[%]',:]
df5["element"] = [i.split("_")[0] for i in df5["variable"]]
items = [i.split("_")[1] for i in df5["variable"]]
df5["item"] = [i.split("[")[0] for i in items]
units = [i.split("[")[1] for i in df5["variable"]]
df5["unit"] = [i.split("]")[0] for i in units]

# put together
df1 = pd.concat([df2, df3, df4, df5]).reset_index(level=0, drop=True)
df1 = df1.rename(columns={"variable": "eucalc-name"})
df1["eucalc-name"] = df1["element"] + "_" + df1["item"] + "[" + df1["unit"] + "]"
df1 = df1.rename(columns={"Country": "geoscale", "Years": "timescale"})
df1["module"] = "industry-endoflife"
df1["lever"] = "eol-material-recovery"
df1 = df1.rename(columns={"lever_eol_recovery": "level"})
df1["string-pivot"] = "none"
df1["type-prefix"] = "none"
df1["module-prefix"] = "ind-eol"
df1["reference-id"] = "missing-reference"
df1["interaction-file"] = "eol_levers"

# order and sort
df1 = df1.loc[:,['geoscale', "timescale", "module", "eucalc-name", "lever", "level", "string-pivot",
                  "type-prefix", "module-prefix", "element", "item", "unit", "value", "reference-id", 
                  "interaction-file"]]
df1 = df1.sort_values(by=['geoscale', "timescale", "module", "eucalc-name"])

# # check for doubles
# indexes = ['geoscale', "timescale", "module", "eucalc-name", "lever", "level", "string-pivot",
#             "type-prefix", "module-prefix", "element", "item", "unit", "reference-id", 
#             "interaction-file"]
# A = pd.DataFrame({'geoscale' : [str(i) for i in df1.groupby(indexes)['value'].agg(len).index], 
#                     'value_count' : df1.groupby(indexes)['value'].agg(len).values
#                   })
# A = A[A['value_count']>1]
# A.shape # OK A has 0 rows

# save
filepath = os.path.join(current_file_directory, '../../../data/csv/eol_material-recovery.csv')
df1.to_csv(filepath, sep = ";", index = False)

# clean
del df1, df2, df3, df4, df5, df_lever2, file, filepath, indexes,\
    items, materials, new, s, search, units, xls_file_directory

############################
##### TECHNOLOGY SHARE #####
############################

# get data
file = "technology_share.xlsx"
xls_file_directory = xls_directory +  "/" + file
df1 = pd.read_excel(xls_file_directory)

# drop technology readiness level
df1.drop("TRL",1, inplace = True)

# make years 2020-2045 with linear interpolation

# expand df to include missing years
years = range(2020,2055,5)
variables = df1["variable"].unique()
panel_years = np.tile(np.repeat(years, 4), len(variables))
panel_levels = np.tile(np.tile([1,2,3,4], len(years)), len(variables))
panel_variables = np.repeat(variables, len(np.tile([1,2,3,4], len(years))))
df_temp = pd.DataFrame({"Country" : "EU27", 
                        "Years" : panel_years, 
                        "variable" : panel_variables,
                        "level" : panel_levels})
years = range(1990,2015+1,1)
panel_years = np.tile(years, len(variables))
panel_variables = np.repeat(variables, len(years))
df_temp2 = pd.DataFrame({"Country" : "EU27", 
                        "Years" : panel_years, 
                        "variable" : panel_variables,
                        "level" : 0})
df_temp = pd.concat([df_temp, df_temp2])
df_temp = df_temp.sort_values(by=["variable", "Country", "Years", "level"])
df1 = pd.merge(df_temp, df1, how="left", on=["Country","Years","variable","level"])

# make values ots before 2015 the same as 2015
for v in variables:
    df1.loc[(df1["variable"] == v) & (df1["Years"] < 2015),"value"] =\
        float(df1.loc[(df1["variable"] == v) & (df1["Years"] == 2015),"value"])

# linear interpolation
def interpolate(df, level):
    df_temp = df.loc[df["level"].isin([0,level]),:]
    df_temp = df_temp.pivot(index=["Country","Years","level"], columns="variable", values='value').reset_index()
    df_temp = df_temp.interpolate(method='linear', axis=0)
    df_temp = pd.melt(df_temp, id_vars = ["Country","Years","level"])
    df_temp = df_temp.loc[df_temp["Years"] > 2015]
    return df_temp
df_temp = pd.concat([interpolate(df1, i) for i in range(1,5)])
df1 = pd.concat([df_temp, df1.loc[df1["Years"] <= 2015,:]])
df1 = df1.sort_values(by=["variable", "level", "Country", "Years"])
df1 = df1.reset_index(drop = True)

# make all countries the same
def make_country(df, country):
    df_temp = df.copy()
    df_temp["Country"] = country
    return df_temp
df1 = pd.concat([make_country(df1, i) for i in countries])

# make dataframe
df1["item"] = df1["variable"]
df1["unit"] = "%"
df1["element"] = "ind_technology-share"
df1 = df1.rename(columns={"variable": "eucalc-name"})
df1["eucalc-name"] = df1["element"] + "_" + df1["item"] + "[" + df1["unit"] + "]"
df1 = df1.rename(columns={"Country": "geoscale", "Years": "timescale"})
df1["module"] = "industry"
df1["lever"] = "technology-share"
df1["string-pivot"] = "none"
df1["type-prefix"] = "none"
df1["module-prefix"] = "ind"
df1["reference-id"] = "missing-reference"
df1["interaction-file"] = "ind_levers"

# order and sort
df1 = df1.loc[:,['geoscale', "timescale", "module", "eucalc-name", "lever", "level", "string-pivot",
                  "type-prefix", "module-prefix", "element", "item", "unit", "value", "reference-id", 
                  "interaction-file"]]
df1 = df1.sort_values(by=['geoscale', "timescale", "module", "eucalc-name"])

# # check for doubles
# indexes = ['geoscale', "timescale", "module", "eucalc-name", "lever", "level", "string-pivot",
#             "type-prefix", "module-prefix", "element", "item", "unit", "reference-id", 
#             "interaction-file"]
# A = pd.DataFrame({'geoscale' : [str(i) for i in df1.groupby(indexes)['value'].agg(len).index], 
#                     'value_count' : df1.groupby(indexes)['value'].agg(len).values
#                   })
# A = A[A['value_count']>1]
# A.shape # OK A has 0 rows

# save
filepath = os.path.join(current_file_directory, '../../../data/csv/industry_technology-share.csv')
df1.to_csv(filepath, sep = ";", index = False)

# clean
del df1, df_temp, df_temp2, file, filepath, panel_levels, panel_variables, panel_years,\
    variables, xls_file_directory, years
    
    
    
