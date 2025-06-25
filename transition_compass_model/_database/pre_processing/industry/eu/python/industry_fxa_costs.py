
# packages
from model.common.data_matrix_class import DataMatrix
import pandas as pd
import pickle
import os
import numpy as np
import warnings
import eurostat
# from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat
warnings.simplefilter("ignore")
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/industry/eu/python/industry_const_costs.py"

# directories
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# get cost data
filepath = os.path.join(current_file_directory, '../data/Literature/costs.xlsx')
df_costs = pd.read_excel(filepath)

# get price index data
filepath = os.path.join(current_file_directory, '../data/Literature/costs_priceindex.xlsx')
df_price = pd.read_excel(filepath)

# add extra countries in price index
df_price_extra = pd.DataFrame({"Country" : ["EU27"], 
                               "Years" : [2015], 
                               "ots_tec_price-indices_pli[%]" : [100]})
df_price = pd.concat([df_price, df_price_extra]).reset_index(level=0, drop=True)
df_price = df_price.sort_values("Country")

# add other countries to costs
countries = df_price["Country"].tolist()
df_costs_new = df_costs.copy()
for c in countries:
    df_temp = df_costs.copy()
    df_temp["Country"] = c
    df_costs_new = pd.concat([df_costs_new, df_temp]).reset_index(level=0, drop=True)

# merge with price data
df_costs_new = pd.merge(df_costs_new, df_price, how="left", on=["Country","Years"])

# put evolution method nan if capex unit is nan
df_costs_new.loc[df_costs_new["capex_unit"].isnull(),"evolution_method"] = np.nan

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

# select industry
df1 = df1.loc[df1["sector"] == "ind",:]

# fix technology name
tech_current = ['CC_ammonia_amm-tech', 'CC_cement_dry-kiln', 'CC_cement_geopolym',
                'CC_cement_wet-kiln', 'CC_chem_chem-tech', 'CC_paper_woodpulp','CC_steel_BF-BOF', 
                'CC_steel_DRI-EAF', 'CC_steel_scrap-EAF',
                'aluminium_prim', 'aluminium_sec', 'amm-tech', 'cement_dry-kiln',
                'cement_wet-kiln', 'chem_chem-tech', 'glass_glass',
                'paper_recycled', 'paper_woodpulp', 'steel_BF-BOF',
                'steel_scrap-EAF', 'primary_copper', 'CC_steel_hisarna',
                'cement_geopolym', 'lime_lime', 'steel_DRI-EAF', 'steel_hisarna']
tech_new = ['CC_ammonia-tech', 'CC_cement-dry-kiln', 'CC_cement-geopolym',
            'CC_cement-wet-kiln', 'CC_chem-chem-tech', 'CC_pulp-tech','CC_steel-BF-BOF',
            'CC_steel-hydrog-DRI', 'CC_steel-scrap-EAF',
            'aluminium-prim', 'aluminium-sec', 'ammonia-tech', 'cement-dry-kiln',
            'cement-wet-kiln', 'chem-chem-tech', 'glass-glass',
            'paper-tech', 'pulp-tech', 'steel-BF-BOF',
            'steel-scrap-EAF', 'copper-tech', 'CC_steel-hisarna',
            'cement-geopolym', 'lime-lime', 'steel-hydrog-DRI', 'steel-hisarna']
for i in range(0,len(tech_current)):
    df1.loc[df1["technology_code"] == tech_current[i],"technology_code"] = tech_new[i]

# make variables
df1["module"] = "cost"
df1["element"] = df1["variable"]
df1["element"] = [i.replace("_", "-") for i in df1["element"].values.tolist()]
df1["item"] = df1["technology_code"]
# df1["unit"] = "num"
df1.loc[df1["element"] == "capex-baseyear","unit"] = df1.loc[df1["element"] == "capex-baseyear","capex_unit"]
df1.loc[df1["element"].isin(['capex-d-factor', 'evolution-method']),"unit"] = "num"
df1["eucalc-name"] = df1["element"] + "_" + df1["item"] + "[" + df1["unit"] + "]"

# order and sort
df1 = df1.loc[:,["Country","Years","eucalc-name","value"]]
df1.columns = ["Country","Years","variable","value"]
df1 = df1.sort_values(by=['Country', "Years", "variable"])

# # check for doubles
# indexes = ['geoscale', "timescale", "module", "eucalc-name", "lever", "level", "string-pivot",
#            "type-prefix", "module-prefix", "element", "item", "unit", "reference-id", 
#            "interaction-file"]
# A = pd.DataFrame({'geoscale' : [str(i) for i in df1.groupby(indexes)['value'].agg(len).index], 
#                    'value_count' : df1.groupby(indexes)['value'].agg(len).values
#                   })
# A = A[A['value_count']>1]
# A.shape # OK A has 0 rows

# correct base year
df1["Years"] = 2023 # put baseyear as 2023

# make data matrix for not CC
idx = [i.split("_")[1] != "CC" for i in df1["variable"]]
df_temp = df1.loc[idx,:]
df_temp = df_temp.pivot(index=["Country","Years"], columns="variable", values='value').reset_index()
dm = DataMatrix.create_from_df(df_temp, 1)
dm.sort("Categories1")

# make the secondary techs (simple assumptions for now)
# TODO: check literature and re-do this
idx = dm.idx
techs = ['aluminium-sec','cement-dry-kiln','chem-chem-tech', 'copper-tech', 'glass-glass', 'paper-tech','steel-scrap-EAF']
techs_sec = ['aluminium-sec-post-consumer','cement-sec-post-consumer','chem-sec-post-consumer', 
             'copper-sec-post-consumer', 'glass-sec-post-consumer', 'paper-sec-post-consumer','steel-sec-post-consumer']
for i in range(0,len(techs)):
    arr_temp = dm.array[...,idx[techs[i]]]
    dm.add(arr_temp, "Categories1", techs_sec[i])
dm.sort("Categories1")

# make data matrix for CC
idx = [i.split("_")[1] == "CC" for i in df1["variable"]]
df_temp = df1.loc[idx,:]
df_temp = df_temp.pivot(index=["Country","Years"], columns="variable", values='value').reset_index()
dm_cc = DataMatrix.create_from_df(df_temp, 1)
dm_cc.sort("Categories1")

# make the secondary techs for cc (simple assumptions for now)
# TODO: check literature and re-do this
idx = dm_cc.idx
techs = ['cement-dry-kiln','chem-chem-tech','steel-scrap-EAF']
techs_sec = ['cement-sec-post-consumer','chem-sec-post-consumer', 'steel-sec-post-consumer']
for i in range(0,len(techs)):
    arr_temp = dm_cc.array[...,idx[techs[i]]]
    dm_cc.add(arr_temp, "Categories1", techs_sec[i])
dm_cc.sort("Categories1")
dm_cc.rename_col_regex("_CC", "", "Variables")

# put together
DM_costs = {"costs" : dm,
            "costs-cc" : dm_cc}

# save
f = os.path.join(current_file_directory, '../data/datamatrix/fxa_costs.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM_costs, handle, protocol=pickle.HIGHEST_PROTOCOL)

# df = dm_cc.write_df()
# df = df.loc[df["Country"] == "EU27",:]
# df_temp = pd.melt(df, id_vars = ['Country','Years'], var_name='variable')
# name = "temp.xlsx"
# df_temp.to_excel("~/Desktop/" + name)


