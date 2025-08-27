
import requests
import re
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

# you need to do material net import (%), production for sectors not considered (kt), packages (something/capita),  
# product net import (%), waste management operation (%), calibration material production (kt),
# calibration energy production (TWh), calibration emissions (tCO2eq)

# Strategy for  material net import (%), production for sectors not considered (kt),  product net import (%), calibration material production (kt)
# 1. Get CHF and Tonnes data of import and export for 2024 (seems only available), and get CHF/Tonne (and if you want adjust with price
# index to get time dimension)
# 2. Get IO Table, transform in Tonnes, get % of net import for modelled products, get tonnes of production of unmodelled materials
# 3. For canton: get net imports and exports in CHF and compute them in tonnes, and from net import difference in tonnes get local
# production and demand (using the same ratios at the national level)

#####################################
##### IMPORT-EXPORT DATA IN CHF #####
#####################################

def get_import_export_chf(trade_flow, df_map_cpa):

    # the data is not on the API, get manually
    file = f"2_5_{trade_flow}_cpa_knt_en"
    filepath = os.path.join(current_file_directory, f'../data/import-export/{file}.xlsx')
    df = pd.read_excel(filepath, sheet_name="TOTAL")
    
    # reshape
    df = df.iloc[:,0:11]
    df.iloc[5,0:2] = df.iloc[4,0:2]
    df.columns = df.iloc[5,:]
    df = df.iloc[6:-5,:]
    df.rename(columns={'CPA code':"code",'Statistical classification of products by activity (CPA) ':"variable"},inplace=True)
    df = pd.melt(df, id_vars = ["code","variable"], var_name='year')
    df.loc[df["value"] == "-","value"] = np.nan
    df['value'] = df['value'].astype(float)
    
    # rename variables with mapping
    df = df.merge(df_map_cpa.loc[:,["code","variable","variable-calc"]], "left", ["code","variable"])
    df = df.loc[~df["variable-calc"].isna(),:]
    df = df.loc[:,["variable-calc","year","value"]]
    df["trade-flow"] = trade_flow
    df = df.groupby(["variable-calc","year","trade-flow"], as_index=False)['value'].agg(sum)

    return df

filepath_map = os.path.join(current_file_directory, '../data/products_materials_mapping/CPA_calc_mapping.xlsx')
df_map = pd.read_excel(filepath_map)
df_map = df_map.iloc[:,0:4]
df_map.columns = df_map.iloc[0,:]
df_map = df_map.iloc[1:,:]
df_map.rename(columns={'CPA code':"code",'Statistical classification of products by activity (CPA) ':"variable"},inplace=True)
df_map["variable-calc"] = df_map["calc_industry_material"]
df_map.loc[df_map["calc_industry_material"].isna(),"variable-calc"] = \
    df_map.loc[df_map["calc_industry_material"].isna(),"calc_industry_product"]
df_map_cpa = df_map.copy()

df_imp_exp = pd.concat([get_import_export_chf(t, df_map_cpa) for t in ["exp","imp"]])


###############################
##### DATA ON CHF / TONNE #####
###############################

def get_price_data(flow):

    # get data
    file = f"2_8_VT_NST_{flow}_en"
    filepath = os.path.join(current_file_directory, f'../data/import-export/{file}.xlsx')
    df = pd.read_excel(filepath)
    
    # reshape
    df = df.iloc[3:-3,0:3]
    df.columns = ["variable","kt","chf-mio"]
    df['kt'] = df['kt'].astype(float)
    df['chf-mio'] = df['chf-mio'].astype(float)
    df['t'] = df['kt'] * 1000
    df['chf'] = df['chf-mio'] * 1000000
    df["price[chf/t]"] = df["chf"]/df["t"]
    df = df.loc[:,["variable","price[chf/t]"]]
    
    # rename variables with mapping
    filepath_map = os.path.join(current_file_directory, '../data/products_materials_mapping/NST_calc_mapping.xlsx')
    df_map = pd.read_excel(filepath_map)
    df_map = df_map.iloc[:,0:3]
    df_map.columns = df_map.iloc[0,:]
    df_map = df_map.iloc[1:,:]
    df_map.rename(columns={'Product groups':"variable"},inplace=True)
    df_map["variable-calc"] = df_map["calc_industry_material"]
    df_map.loc[df_map["calc_industry_material"].isna(),"variable-calc"] = \
        df_map.loc[df_map["calc_industry_material"].isna(),"calc_industry_product"]
    df = df.merge(df_map.loc[:,["variable","variable-calc"]], "left", ["variable"])
    df = df.loc[~df["variable-calc"].isna(),["variable-calc","price[chf/t]"]]
    df = df.groupby(["variable-calc"], as_index=False)['price[chf/t]'].agg(np.mean)
    df["trade-flow"] = flow
    
    return df

df_price = pd.concat([get_price_data(flow) for flow in ["EXP","IMP"]])

###############################
##### DATA ON PRICE INDEX #####
###############################

# get mapping data with 2-digit codes
df_map_cpa["code-2digit"] = [s[-2:] for s in df_map_cpa["code"]]

# get data
file = "su-q-05.04.03.02-ppi-ipp-det"
filepath = os.path.join(current_file_directory, f'../data/price-index/{file}.xlsx')
df = pd.read_excel(filepath, sheet_name="INDEX_y")
df.columns = df.iloc[6,:]
df = df.iloc[:,[3] + list(range(15,len(df.columns)))]
df = df.loc[df["Product code"].isin(df_map_cpa["code-2digit"]),:]
df.rename(columns={'Product code':"code"},inplace=True)
df = pd.melt(df, id_vars = ["code"], var_name='year')
df.loc[df["value"] == '…',"value"] = np.nan
df['value'] = df['value'].astype(float)
df_temp = df_map_cpa.loc[:,["code-2digit","variable-calc"]]
df_temp.rename(columns={'code-2digit':"code"},inplace=True)
df = df.merge(df_temp, "left", ["code"])
df.loc[df["code"] == "B","variable-calc"] = "other"
df = df.loc[~df["variable-calc"].isna(),:]
df = df.loc[:,["variable-calc","year","value"]]
df = df.groupby(["variable-calc","year"], as_index=False)['value'].agg(np.mean)
df["year"] = df["year"].astype(int)
df["year"] = df["year"].astype(str)
df_price_index = df.copy()

del df, df_temp

###########################################
##### DATA ON CHF / TONNE TIME SERIES #####
###########################################

# fix df_price
df_price["year"] = "2024"
df_temp = df_price.loc[df_price["variable-calc"] == "steel, aluminium, copper",:]
df_temp["variable-calc"] = "aluminium-pack"
df_price = pd.concat([df_price, df_temp])
df_temp = df_price.loc[df_price["variable-calc"] == "wwp",:]
df_temp["variable-calc"] = "paper"
df_price = pd.concat([df_price, df_temp])
df_temp = df_price.loc[df_price["variable-calc"] == "computer, phone, tv, fridge, freezer, dishwasher, dryer, wmachine",:]
df_temp["variable-calc"] = "computer, phone, tv"
df_price = pd.concat([df_price, df_temp])
df_price.loc[df_price["variable-calc"] == "computer, phone, tv, fridge, freezer, dishwasher, dryer, wmachine","variable-calc"] = "fridge, freezer, dishwasher, dryer, wmachine"
df_temp = df_price.loc[df_price["variable-calc"] == "vehicles, marine, rail, aviation",:]
df_temp["variable-calc"] = "vehicles"
df_price = pd.concat([df_price, df_temp])
df_price.loc[df_price["variable-calc"] == "vehicles, marine, rail, aviation, tra-equip","variable-calc"] = "marine, rail, aviation, tra-equip"

# fix price index
df_temp = df_price_index.loc[df_price_index["variable-calc"] == "vehicles",:]
df_temp["variable-calc"] = "marine, rail, aviation, tra-equip"
df_price_index = pd.concat([df_price_index, df_temp])
df_price_index["trade-flow"] = "IMP"
df_temp = df_price_index.copy()
df_temp["trade-flow"] = "EXP"
df_price_index = pd.concat([df_price_index,df_temp])
df_price_index.rename(columns={"value":"price-index"},inplace=True)

# merge
df = df_price_index.copy()
df = df.merge(df_price, "left", ["variable-calc","year","trade-flow",])
df.sort_values(["variable-calc","trade-flow","year"],inplace=True)
df.loc[df["variable-calc"] == "mae","price[chf/t]"] = np.array(df.loc[df["variable-calc"] == "computer, phone, tv","price[chf/t]"])
df["price-index-pch"] = np.nan
df["year"] = df["year"].astype(int)
for y in list(range(2023,2004-1,-1)):
    t1 = np.array(df.loc[df["year"] == 2024,"price-index"])
    t0 = np.array(df.loc[df["year"] == y,"price-index"])
    df.loc[df["year"] == y,"price-index-pch"] = (t0-t1)/t1
    price_t1 = np.array(df.loc[df["year"] == 2024,"price[chf/t]"])
    delta_t0 = np.array(df.loc[df["year"] == y,"price-index-pch"])
    df.loc[df["year"] == y,"price[chf/t]"] = price_t1 * (1 + delta_t0)
df = df.loc[:,["variable-calc","trade-flow","year","price[chf/t]"]]
df_price_ts = df.copy()

del df, df_temp, delta_t0, price_t1, t0, t1, y

################################
##### NET IMPORT SHARE (%) #####
################################

# note: we make this share with IO data in CHF
# note: domestic demand should be supply at basic prices - export

def get_io_data(year):

    filepath = os.path.join(current_file_directory, f'../data/input-output/su-e-04.03-IOT-{year}.xlsx')
    df_io = pd.read_excel(filepath, sheet_name="siot")
    
    # IMPORTS AND SUPPLY
    
    df = df_io.copy()
    
    # rows
    word = "PRODUCTS (CPA)"
    mask = df.applymap(lambda x: word in str(x) if pd.notna(x) else False)
    column_with_match = df.columns[mask.any(axis=0)].tolist()[0]
    row_with_match = df[column_with_match][mask.any(axis=1)].tolist()[0]
    rows = [row_with_match,"Output at basic prices", "Imports cif", "Supply at basic prices"]
    df = df.loc[df[column_with_match].isin(rows),:]
    
    # columns
    df = df.iloc[:,df.columns.get_loc(column_with_match):]
    arr = np.array(df.iloc[0,:])
    nan_first_index = np.where(pd.isna(arr))[0][0]
    df = df.iloc[:,0:nan_first_index]
    codes_raw = list(np.array(df.iloc[0,1:]))
    arr = np.array(df.iloc[0,:].astype(str))
    df.iloc[0,:] = np.array([x.strip() if isinstance(x, str) else x for x in arr], dtype=object)
    # code_first = df.iloc[0,1]
    # code_last = df.iloc[0,len(df.columns)-1]
    
    # reshape
    df.columns = df.iloc[0,:]
    df = df.iloc[1:,:]
    df.rename(columns={"PRODUCTS (CPA)":"variable"},inplace=True)
    df = pd.melt(df, id_vars = ["variable"], var_name='code')
    df_main = df.copy()
    
    
    # EXPORTS
    
    df = df_io.copy()
    
    # columns
    word = "Exports"
    mask = df.applymap(lambda x: word in str(x) if pd.notna(x) else False)
    columns_with_exports = df.columns[mask.any(axis=0)].tolist()[0]
    word = "Code"
    mask = df.applymap(lambda x: word in str(x) if pd.notna(x) else False)
    columns_with_code = df.columns[mask.any(axis=0)].tolist()[0]
    df = df.loc[:,[columns_with_code,columns_with_exports]]
    
    # rows
    df = df.loc[df.iloc[:,0].isin(codes_raw),:]
    
    # reshape
    df.columns = ["code","Exports"]
    df = pd.melt(df, id_vars = ["code"], var_name='variable')
    arr = np.array(df["code"].astype(str))
    df["code"] = np.array([x.rstrip() if isinstance(x, str) else x for x in arr], dtype=object)
    
    # concat
    df_main = pd.concat([df_main,df])
    df_main["year"] = year
    df_main = df_main.loc[:,["variable","code","year","value"]]
    df_main.sort_values(["variable","code","year"],inplace=True)
    df_main["value"] = df_main["value"].astype(float)
    
    return df_main

# get data
df_io = pd.concat([get_io_data(y) for y in ["2005","2008","2011","2014","2017"]])

# rename codes wrt to calc name
df_temp = df_map_cpa.loc[:,["code-2digit","variable-calc"]]
df_temp.rename(columns={"code-2digit":"code"},inplace=True)
df_temp = df_temp.loc[~df_temp["variable-calc"].isna(),:]
df_io["code"] = df_io["code"].str.replace(r"\s*-\s*", " - ", regex=True)
df_io["variable-calc"] = np.nan
df_io.loc[df_io["code"] == "05","variable-calc"] = "other"
df_io.loc[df_io["code"] == "05 - 09","variable-calc"] = "other"
df_io.loc[df_io["code"] == "10 - 12","variable-calc"] = "fbt"
df_io.loc[df_io["code"] == "13 - 15","variable-calc"] = "textiles"
df_io.loc[df_io["code"] == "19 - 20","variable-calc"] = "chem"
df_io = df_io.loc[df_io["year"].isin(["2011","2014","2017"]),:] # I am going to take only 2011, 2014 and 2017 (as it seems there are large differences with previous years)
for code in df_temp["code"].unique().tolist():
    df_io.loc[df_io["code"] == code,"variable-calc"] = list(df_temp.loc[df_temp["code"] == code,"variable-calc"])[0]
df_io = df_io.loc[~df_io["variable-calc"].isna(),:]
df_io = df_io.groupby(["variable","variable-calc","year"], as_index=False)['value'].agg(sum)

# add yearly data on import-export post 2016
df_temp = df_imp_exp.copy()
df_temp.rename(columns={"trade-flow":"variable"},inplace=True)
df_temp["value"] = df_temp["value"]/1000
df_temp.loc[df_temp["variable"] == "imp","variable"] = "Imports alternative"
df_temp.loc[df_temp["variable"] == "exp","variable"] = "Exports alternative"
df_temp["year"] = df_temp["year"].astype(str)
df_io = pd.concat([df_io, df_temp])
# df_io = df_io.pivot(index=["variable-calc","year"], columns="variable", values='value').reset_index()

# make dm
df_io["Country"] = "Switzerland"
df_io["variable"].unique()
names_old = ['Exports', 'Imports cif', 'Output at basic prices',
             'Supply at basic prices', 'Exports alternative',
             'Imports alternative']
names_new = ['exports', 'imports', 'output',
             'supply', 'exports-alt', 'imports-alt']
for o, n in zip(names_old,names_new):
    df_io.loc[df_io["variable"] == o,"variable"] = n
df_io["variable"] = df_io["variable"] + "_" + df_io["variable-calc"] + "[mio-chf]"
df_io.rename(columns={"year":"Years"},inplace=True)
df_io = df_io.loc[:,["Country","Years","variable","value"]]
countries = df_io["Country"].unique()
years = list(range(2011,2024+1,1))
variables = df_io["variable"].unique()
panel_countries = np.repeat(countries, len(variables) * len(years))
panel_years = np.tile(np.tile(years, len(variables)), len(countries))
panel_variables = np.tile(np.repeat(variables, len(years)), len(countries))
df_temp = pd.DataFrame({"Country" : panel_countries, 
                        "Years" : panel_years, 
                        "variable" : panel_variables})
df_io["Years"] = df_io["Years"].astype(int)
df_io = pd.merge(df_temp, df_io, how="left", on=["Country","Years","variable"])
df_io = df_io.pivot(index=["Country","Years"], columns="variable", values='value').reset_index()
dm_io = DataMatrix.create_from_df(df_io, 1)

# # adjust demand for timber (right now it's constant at 2017 levels)
# dm_temp = dm_io.filter({"Variables" : ["demand"],"Categories1" : ["timber","wwp"]})
# # df_temp = dm_temp.write_df()
# # myyears = list(range(2011,2024+1))
# # myyears.remove(2017)
# # for y in myyears: dm_temp[:,y,:,"timber"] = np.nan
# # df_temp = dm_temp.write_df()
# arr_temp = 1+(dm_temp[:,:,:,"wwp"] - dm_temp[:,2017,:,"wwp"])/dm_temp[:,2017,:,"wwp"]
# dm_temp.add(arr_temp, "Categories1", "delta-wwp", "%")
# dm_temp.operation("timber", "*", "delta-wwp", "Categories1", "timber-adj")
# dm_temp.datamatrix_plot()

# make demand and net import
dm_netimp = dm_io.copy()
dm_netimp = linear_fitting(dm_netimp,dm_netimp.col_labels["Years"], min_t0=0,min_tb=0)
# dm_netimp.filter({"Variables" : ["exports","exports-alt"]}).flatten().datamatrix_plot()
dm_netimp.operation("supply", "-", "exports", "Variables", "demand", "mio-chf")
dm_netimp.operation("supply", "-", "exports-alt", "Variables", "demand-alt", "mio-chf")
arr_temp = (dm_netimp[:,:,"imports",:] - dm_netimp[:,:,"exports",:])/dm_netimp[:,:,"demand",:]
dm_netimp.add(arr_temp, "Variables", "net-import", "%")
arr_temp = (dm_netimp[:,:,"imports-alt",:] - dm_netimp[:,:,"exports-alt",:])/dm_netimp[:,:,"demand-alt",:]
dm_netimp.add(arr_temp, "Variables", "net-import-alt", "%")
# dm_netimp.filter({"Variables" : ["net-import","net-import-alt"]}).flatten().datamatrix_plot()

# note: the time series are a bit different but not too much, I take net-import-alt as a reference
dm_wwp_demand = dm_netimp.filter({"Variables" : ["demand-alt"], "Categories1" : ["wwp"]}) # I need this for fxa
dm_wwp_demand.rename_col("demand-alt","material-demand","Variables")
# dm_wwp_demand.flatten().datamatrix_plot()
dm_netimp = dm_netimp.filter({"Variables" : ["net-import-alt"]})
dm_netimp.rename_col("net-import-alt","net-import","Variables")
dm_netimp.array[dm_netimp.array == -np.inf] = np.nan

# make ots
years_ots = list(range(1990,2024+1,1))
dm_netimp = linear_fitting(dm_netimp, years_ots, based_on=list(range(2016,2024+1)))
# dm_netimp.flatten().datamatrix_plot()

# make fts
years_fts = list(range(2025,2050+5,5))
dm_netimp = linear_fitting(dm_netimp, years_fts, based_on=list(range(2016,2024+1)))
# dm_netimp.flatten().datamatrix_plot()
dm_netimp.drop("Years",[2024])

# make all goods categories
dm_netimp_goods = dm_netimp.filter({"Categories1" : ["vehicles",'marine, rail, aviation, tra-equip', 
                                                     "aluminium-pack","paper",'computer, phone, tv',
                                                     'fridge, freezer, dishwasher, dryer, wmachine']})
dict_map = {'HDV_BEV' : "vehicles", 'HDV_FCEV' : "vehicles", 'HDV_ICE-diesel' : "vehicles", 
 'HDV_ICE-gas' : "vehicles", 'HDV_ICE-gasoline' : "vehicles", 'HDV_PHEV-diesel' : "vehicles", 
 'HDV_PHEV-gasoline' : "vehicles", 'LDV_BEV' : "vehicles", 'LDV_FCEV' : "vehicles", 
 'LDV_ICE-diesel' : "vehicles", 'LDV_ICE-gas' : "vehicles", 'LDV_ICE-gasoline' : "vehicles", 
 'LDV_PHEV-diesel' : "vehicles", 'LDV_PHEV-gasoline' : "vehicles", 'bus_ICE-diesel' : "vehicles", 
 'bus_ICE-gas' : "vehicles", 'planes_ICE' : 'marine, rail, aviation, tra-equip', 
 'ships_ICE-diesel' : 'marine, rail, aviation, tra-equip', 'trains_CEV' : 'marine, rail, aviation, tra-equip', 
 'trains_ICE-diesel' : 'marine, rail, aviation, tra-equip',
 'glass-pack' : 'aluminium-pack', 'paper-pack' : "paper", 'paper-print' : "paper", 
 'paper-san' : "paper", 'plastic-pack' : 'aluminium-pack', 
 'computer' : 'computer, phone, tv', 'phone': 'computer, phone, tv', 'tv': 'computer, phone, tv',
 'dishwasher' : 'fridge, freezer, dishwasher, dryer, wmachine', 
 'dryer' : 'fridge, freezer, dishwasher, dryer, wmachine', 
 'freezer' : 'fridge, freezer, dishwasher, dryer, wmachine', 
 'fridge' : 'fridge, freezer, dishwasher, dryer, wmachine', 
 'wmachine' : 'fridge, freezer, dishwasher, dryer, wmachine',
 }
for key in dict_map.keys():
    dm_temp = dm_netimp_goods.filter({"Categories1" : [dict_map[key]]})
    dm_temp.rename_col(dict_map[key], key, "Categories1")
    dm_netimp_goods.append(dm_temp,"Categories1")
dm_netimp_goods.drop("Categories1",["vehicles",'marine, rail, aviation, tra-equip',
                                    "aluminium-pack","paper",'computer, phone, tv',
                                    'fridge, freezer, dishwasher, dryer, wmachine'])
dm_netimp_goods.append(dm_netimp.filter({"Categories1" : ["aluminium-pack"]}), "Categories1")

# zeroes
zeroes = ['floor-area-new-non-residential', 'floor-area-new-residential', 
          'floor-area-reno-non-residential', 'floor-area-reno-residential', 
          'rail', 'road',  'trolley-cables', 'new-dhg-pipe']
dm_netimp_goods.add(0, "Categories1", zeroes, "%", dummy=True)
dm_netimp_goods.sort("Categories1")
dm_netimp_goods.rename_col("net-import","product-net-import","Variables")

# save
years_ots = list(range(1990,2023+1,1))
dm_ots = dm_netimp_goods.filter({"Years" : years_ots})
dm_fts = dm_netimp_goods.filter({"Years" : years_fts})
DM_fts = {1: dm_fts.copy(), 2: dm_fts.copy(), 3: dm_fts.copy(), 4: dm_fts.copy()} # for now we set all levels to be the same
DM = {"ots" : dm_ots,
      "fts" : DM_fts}
f = os.path.join(current_file_directory, '../data/datamatrix/lever_product-net-import.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)

# make material net import
# ['aluminium', 'cement', 'chem', 'copper', 'glass', 'lime', 'other', 'paper', 'steel', 'timber']
dm_netimp_materials = dm_netimp.filter({"Categories1" : ['glass, cement, lime, chem, other',
                                                 'steel, aluminium, copper',
                                                 'timber']})
dict_map = {'glass' : 'glass, cement, lime, chem, other',
            'cement' : 'glass, cement, lime, chem, other',
            'lime' : 'glass, cement, lime, chem, other',
            'chem' : 'glass, cement, lime, chem, other',
            'other' : 'glass, cement, lime, chem, other',
            'steel' : 'steel, aluminium, copper',
            'aluminium' : 'steel, aluminium, copper',
            'copper' : 'steel, aluminium, copper',
            }
for key in dict_map.keys():
    dm_temp = dm_netimp_materials.filter({"Categories1" : [dict_map[key]]})
    dm_temp.rename_col(dict_map[key], key, "Categories1")
    dm_netimp_materials.append(dm_temp,"Categories1")
dm_netimp_materials.drop("Categories1",['glass, cement, lime, chem, other',
                                    'steel, aluminium, copper'])
dm_netimp_materials.append(dm_netimp.filter({"Categories1" : ["paper"]}), "Categories1")
dm_netimp_materials.sort("Categories1")
dm_netimp_materials.rename_col("net-import","material-net-import","Variables")

# save
dm_ots = dm_netimp_materials.filter({"Years" : years_ots})
dm_fts = dm_netimp_materials.filter({"Years" : years_fts})
DM_fts = {1: dm_fts.copy(), 2: dm_fts.copy(), 3: dm_fts.copy(), 4: dm_fts.copy()} # for now we set all levels to be the same
DM = {"ots" : dm_ots,
      "fts" : DM_fts}
f = os.path.join(current_file_directory, '../data/datamatrix/lever_material-net-import.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)

del arr_temp, code, countries, df_map, df_temp, dict_map, DM, dm_fts, DM_fts, \
    dm_netimp_goods, dm_netimp_materials, dm_ots, dm_temp, f, file, filepath_map, \
    handle, key, n, names_new, names_old, o, panel_countries, panel_variables, \
    panel_years, variables, years, zeroes


####################################
##### MATERIAL PRODUCTION (kt) #####
####################################

# note: this will be used for unmodelled sectors, and for materials
my_aggregates = ['chem', 'glass, cement, lime, chem, other',
                 'fbt','mae','ois','textiles','wwp', 'timber',
                 'other', 'paper', 'steel, aluminium, copper',
                 'marine, rail, aviation, tra-equip']

# get output in chf
dm_out = dm_io.filter({"Variables" : ["output"], "Categories1" : my_aggregates})
dm_out.drop("Years",[2024])
dm_out.rename_col("output","material-production","Variables")
dm_out.change_unit("material-production", 1e6, "mio-chf", "chf")
# df_temp = dm_out.write_df()

# for timber, get the same trend of wwp
dm_out[:,2014,:,"timber"] = dm_out[:,2017,:,"timber"] * (1+(dm_out[:,2014,:,"wwp"]-dm_out[:,2017,:,"wwp"])/dm_out[:,2017,:,"wwp"])
dm_out[:,2011,:,"timber"] = dm_out[:,2017,:,"timber"] * (1+(dm_out[:,2011,:,"wwp"]-dm_out[:,2017,:,"wwp"])/dm_out[:,2017,:,"wwp"])
# df_temp = dm_out.write_df()

# get price in chf/t
df_temp = df_price_ts.copy()
df_temp = df_temp.loc[df_temp["trade-flow"] == "EXP",:] 
df_temp.rename(columns={"year":"Years","price[chf/t]":"value"},inplace=True)
df_temp["Country"] = "Switzerland"
df_temp = df_temp.loc[:,["Country","Years","variable-calc","value"]]
df_temp["variable-calc"] = "price_" + df_temp["variable-calc"] + "[chf/t]"
df_temp = df_temp.pivot(index=["Country","Years"], columns="variable-calc", values='value').reset_index()
dm_price = DataMatrix.create_from_df(df_temp, 1)
dm_price = dm_price.filter({"Years" : dm_out.col_labels["Years"], "Categories1" : my_aggregates})
# df_temp = dm_price.write_df()

# make output kt
dm_out.append(dm_price, "Variables")
dm_out.operation("material-production", "/", "price", "Variables", "material-production-kt", "t")
dm_out = dm_out.filter({"Variables" : ["material-production-kt"]})
dm_out.rename_col("material-production-kt","material-production","Variables")
dm_out.change_unit("material-production", 1e-3, "t", "kt")

# make individual material with shares from EU27
filepath = os.path.join(current_file_directory,  '../../eu/data/datamatrix/calibration_material-production.pickle')
with open(filepath, 'rb') as handle:
    dm_calib_matprod_eu = pickle.load(handle)
dict_map = {'glass, cement, lime, chem, other' : ['glass', 'cement', 'lime'],
            'steel, aluminium, copper' : ['steel', 'aluminium', 'copper']}
for key in dict_map.keys():
    dm_temp = dm_calib_matprod_eu.filter({"Country" : ["EU27"], "Years" : dm_out.col_labels["Years"],
                                          "Categories1" : dict_map[key]})
    dm_temp.normalise("Categories1")
    dm_temp1 = dm_out.filter({"Categories1" : [key]})
    dm_temp.array = dm_temp.array * dm_temp1.array
    dm_temp.units["material-production"] = "kt"
    dm_temp.rename_col("EU27","Switzerland","Country")
    dm_out.append(dm_temp, "Categories1")
    dm_out.drop("Categories1", [key])
dm_out.sort("Categories1")

# make tra equip
# as in CH C30 is mostly rail + planes, I assume rail + aircraft 90%, and ships + other 10%
# so tra equp around 5%
dm_out[...,"marine, rail, aviation, tra-equip"] = dm_out[...,"marine, rail, aviation, tra-equip"] * 0.05
dm_out.rename_col("marine, rail, aviation, tra-equip", "tra-equip", "Categories1")
dm_out.sort("Categories1")
    
# make ots
dm_out = linear_fitting(dm_out, years_ots, min_t0=0, min_tb=0)
# dm_out.flatten().datamatrix_plot()

# make fts
dm_out = linear_fitting(dm_out, years_fts, min_t0=0, min_tb=0)
# dm_out.flatten().datamatrix_plot()
# df_temp = dm_out.filter({"Years" : [2023]}).write_df().melt(("Country","Years"))

# make calibration data
# note: for calib data, we do not need to make missing ots and fts
dm_out_calib = dm_out.filter({"Years" : list(range(2011,2023,+1))})
years = list(range(1990,2023+1)) + list(range(2025,2050+5,5))
missing = np.array(years)[[y not in dm_out_calib.col_labels["Years"] for y in years]].tolist()
dm_out_calib.add(np.nan, "Years", missing, dummy=True)
dm_out_calib.sort("Years")
# TODO: some of this maybe too low (cement, steel, lime, glass) and some too high (wwp, other),
# consider what to do for calibration.

# make material demand wwp (fxa)
# dm_wwp_demand.flatten().datamatrix_plot()
dm_wwp_demand.change_unit("material-demand", 1e6, "mio-chf", "chf")
dm_wwp_demand.drop("Years",[2024])
dm_wwp_demand.array = dm_wwp_demand.array / dm_price.filter({"Categories1" : ["wwp"]}).array
dm_wwp_demand.units["material-demand"] = "t"
dm_wwp_demand = linear_fitting(dm_wwp_demand, years_ots, based_on=list(range(2011,2017+1)))
dm_wwp_demand = linear_fitting(dm_wwp_demand, years_fts, based_on=list(range(2017,2019+1)))
# dm_wwp_demand.flatten().datamatrix_plot()

# save
dm_out_notmodelled = dm_out.filter({"Categories1" : ['fbt','mae', 'ois', 'textiles', 'wwp', "tra-equip"]})
f = os.path.join(current_file_directory, '../data/datamatrix/fxa_material-production.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_out_notmodelled, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# wwp demand
f = os.path.join(current_file_directory, '../data/datamatrix/fxa_material-demand.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_wwp_demand, handle, protocol=pickle.HIGHEST_PROTOCOL)

# calib material production
f = os.path.join(current_file_directory, '../data/datamatrix/calibration_material-production.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_out_calib, handle, protocol=pickle.HIGHEST_PROTOCOL)


# # check with material flow data
# # Direct Material Input (DMI) = Domestic Extraction (DE) + Imports
# # Domestic Material Consumption (DMC) = Domestic Extraction (DE) + Imports - Exports
# # so DE is my material production, and DMC is my material demand
# # Domestic Processed Output (DPO) are the amount of products produced in the economy
# # Net Additional Stock (NAS) = DE + Imports + Balancing items: input side - Exports - DPO - Balancing items: output side
# # so basically all materials that are not used for producing products, add up to the stock
# # to check the domestic production of material (what in the IO is called output at basic prices),
# # I consider DE, which in the tables is called DEU. And I do not consider fossil fuels.
# file = "je-e-02.04.10.01"
# filepath = os.path.join(current_file_directory, f'../data/material-flow/{file}.xlsx')
# df = pd.read_excel(filepath)
# years = list(df.iloc[2,1:].astype(int).astype(str))
# df.columns = ["variable"] + years
# df = df.iloc[5:11,:]
# df = df.loc[~df.iloc[:,0].isin(["Main categories in %"]),:]
# df = pd.melt(df, id_vars = ["variable"], var_name='year')
# df.loc[df["variable"] == "Domestic extraction used (DEU), million tonnes","variable"] = "Total"
# df.sort_values(["variable","year"],inplace=True)
# df = df.pivot(index=["year"], columns="variable", values='value').reset_index()
# df = df.iloc[0:-1,:]
# df_matflow = df.copy()

# dm_temp = dm_out.copy()
# material_categories = {
#     'Metal ores': ['aluminium', 'copper', 'steel'],
#     'Non metallic minerals': ['cement', 'glass', 'lime'],
#     'Fossil energy materials': ['chem'],
#     'Biomass': ['paper', 'timber', 'wwp']
# }
# dm_temp.groupby(material_categories, "Categories1", inplace=True)
# dm_temp.drop("Categories1", ["other",'fbt', 'mae', 'ois', 'textiles'])
# dm_temp = dm_temp.filter({"Years" : list(df_matflow["year"].astype(int))})
# dm_temp.change_unit("material-production", 1e-3, "kt", "Mt")
# dm_temp.append(dm_temp.groupby({"Total" : ["Metal ores","Non metallic minerals","Fossil energy materials","Biomass"]},
#                                "Categories1"), "Categories1")
# df_check = dm_temp.write_df()
# # so the factor is fine, just I can be low or high depending on what I put in (for example "other", which is huge).
# # TODO: emissions are about 90 mega tonnes a year, let's see where we get and we'll think about what to do with the
# # material calibration after. Also, note that there is another file, https://www.bfs.admin.ch/bfs/en/home/statistics/territory-environment/environmental-accounting/material-flows.assetdetail.35975770.html
# # which has material flows in raw material equivalent, and that's higher (if considering also exports, is around 200 mil tonnes)
# # to be understood what is what, and what to consider.


# note: for demand of packages per capita, you can take the waste packages statistics, and do a per capita thing
# (assuming that all packages wasted are all packages demanded per year)

# note: eurostat has data for municipal waste in CH (https://ec.europa.eu/eurostat/databrowser/view/env_wasmun__custom_17697260/default/table)
# which in case could be used for waste management operations around packages)

#######################################
##### PACK CONSUMPTION PER CAPITA #####
#######################################

# paper
# note: we will take the data of overall paper and split them in the packaging, print, and sanitary
# so from je-f-02.03.02.11, assuming that the taux is the taux de collecte, we will multiply the waste
# number by this taux de collecte

# from https://www.bafu.admin.ch/bafu/en/home/topics/waste/guide-to-waste-a-z/paper-and-cardboard.html:
# produce 1.2 million tonnes of paper. 

# filepath = os.path.join(current_file_directory, '../data/waste/0_Déchets Quantités produites et recyclées/je-f-02.03.02.11.xlsx')
# df = pd.read_excel(filepath)
# df.columns = df.iloc[3,:]
# df = df.iloc[8:10,:]
# df = pd.melt(df, id_vars = ["Matériaux ",'Unité'], var_name='year')
# df.loc[df["value"] == '…',"value"] = np.nan
# df["value"] = df["value"].astype(float)
# df.columns = ["material","unit","year","value"]
# df["material"] = "paper"
# df = df.pivot(index=["material","year"], columns="unit", values='value').reset_index()
# df.columns = ['material', 'year', '%', 't']
# df["%"] = df["%"]/100
# df["value"] = df["t"] * (1 - df["%"] + 1)
# df = df.loc[:,["material","year","value"]]

# fao has the split for paper, so I rely on that
filepath = os.path.join(current_file_directory, '../data/production-fao/FAOSTAT_data_en_8-21-2025.csv')
df = pd.read_csv(filepath)
df.columns
df = df.loc[:,["Item","Year","Unit","Value"]]
df["Item"] = df["Item"] + "[" + df["Unit"] + "]"
df = df.loc[:,["Item","Year","Value"]]
df = df.pivot(index=["Year"], columns="Item", values='Value').reset_index()
df["Country"] = "Switzerland"
df.rename(columns={"Year":"Years"},inplace=True)
dm = DataMatrix.create_from_df(df, 0)
dm.groupby({"paper-pack-pre" : ['Cartonboard', 'Case materials', 'Other papers mainly for packaging', 'Wrapping papers']}, 
           "Variables", inplace=True)
dm.groupby({"paper-pack" : ['paper-pack-pre','Wrapping and packaging paper and paperboard (1961-1997)']}, 
           "Variables", inplace=True)
df_temp = dm.write_df()
dm.groupby({'paper-print': ['Newsprint','Printing and writing papers'],
           'paper-san' : ['Household and sanitary papers']}, "Variables", inplace=True)
dm.drop("Variables",['Other paper and paperboard', 'Other paper and paperboard, not elsewhere specified'])


# plastic
# take the series of PET and increase it by factor of total consumption of plastic today
# https://www.bafu.admin.ch/bafu/en/home/topics/waste/guide-to-waste-a-z/plastics.html: 
# Around one million tonnes of plastics are consumed in Switzerland every year – that's 120 kilograms per capita (reference year 2017). 
# Around 790,000 tonnes of plastic waste are generated every year, almost half of which is used for less than a year, e.g. as packaging. 
# so we can say that around 395000 are the plastic packaging waste, and we can assume that that's the plastic packaging consumption.

filepath = os.path.join(current_file_directory, '../data/waste/0_Déchets Quantités produites et recyclées/je-f-02.03.02.11.xlsx')
df = pd.read_excel(filepath)
df.columns = df.iloc[3,:]
df = df.iloc[29:31,:]
df = pd.melt(df, id_vars = ["Matériaux ",'Unité'], var_name='year')
df.loc[df["value"] == '…',"value"] = np.nan
df["value"] = df["value"].astype(float)
df.columns = ["material","unit","year","value"]
df["material"] = "pet"
df = df.pivot(index=["material","year"], columns="unit", values='value').reset_index()
df.columns = ['material', 'year', '%', 't']
df["%"] = df["%"]/100
df["value"] = df["t"]/df["%"]
df = df.loc[:,["material","year","value"]]

plastic_packaging_consumption_2023 = 395000
df["factor"] = float((plastic_packaging_consumption_2023-df.loc[df["year"] == 2023,"value"])/df.loc[df["year"] == 2023,"value"])
df["value"] = df["value"]*(1+df["factor"])
df = df.loc[:,["material","year","value"]]
df["material"] = "plastic-pack[t]"

df = df.pivot(index=["year"], columns="material", values='value').reset_index()
df["Country"] = "Switzerland"
df.rename(columns={"year":"Years"},inplace=True)
dm_temp = DataMatrix.create_from_df(df, 0)
dm_temp.drop("Years",[1985])
dm_temp.add(np.nan, "Years", [1991,1992], dummy=True)
dm_temp.sort("Years")
for y in [1990,1991,1992]:
    dm_temp[:,y,...] = dm_temp[:,1993,...]
df_temp = dm_temp.write_df()
dm.append(dm_temp, "Variables")

# aluminium
# note: I get the amount of waste of aluminium packages and assuming that that's the same of consumption
filepath = os.path.join(current_file_directory, '../data/waste/0_Déchets Quantités produites et recyclées/je-f-02.03.02.11.xlsx')
df = pd.read_excel(filepath)
df.columns = df.iloc[3,:]
df = df.iloc[23:24,:]
df = pd.melt(df, id_vars = ["Matériaux ",'Unité'], var_name='year')
df["value"] = df["value"].astype(float)
df.columns = ["material","unit","year","value"]
df["material"] = "aluminium-pack[t]"
df = df.loc[:,["material","year","value"]]
df = df.pivot(index=["year"], columns="material", values='value').reset_index()
df["Country"] = "Switzerland"
df.rename(columns={"year":"Years"},inplace=True)
dm_temp = DataMatrix.create_from_df(df, 0)
dm_temp.drop("Years",[1985])
dm_temp.add(np.nan, "Years", [1991,1992], dummy=True)
dm_temp.sort("Years")
for y in [1990,1991,1992]:
    dm_temp[:,y,...] = dm_temp[:,1993,...]
df_temp = dm_temp.write_df()
dm.append(dm_temp, "Variables")

# glass pack
filepath = os.path.join(current_file_directory, '../data/waste/0_Déchets Quantités produites et recyclées/je-f-02.03.02.11.xlsx')
df = pd.read_excel(filepath)
df.columns = df.iloc[3,:]
df = df.iloc[14:16,:]
df = pd.melt(df, id_vars = ["Matériaux ",'Unité'], var_name='year')
df["value"] = df["value"].astype(float)
df.columns = ["material","unit","year","value"]
df["material"] = "glass-pack[t]"
df = df.pivot(index=["material","year"], columns="unit", values='value').reset_index()
df.columns = ['material', 'year', '%', 't']
df["%"] = df["%"]/100
df["value"] = df["t"]/df["%"]
df = df.loc[:,["material","year","value"]]
df = df.pivot(index=["year"], columns="material", values='value').reset_index()
df["Country"] = "Switzerland"
df.rename(columns={"year":"Years"},inplace=True)
dm_temp = DataMatrix.create_from_df(df, 0)
dm_temp.drop("Years",[1985])
dm_temp.add(np.nan, "Years", [1991,1992], dummy=True)
dm_temp.sort("Years")
for y in [1990,1991,1992]:
    dm_temp[:,y,...] = dm_temp[:,1993,...]
df_temp = dm_temp.write_df()
dm.append(dm_temp, "Variables")

# load DM_pack
filepath = os.path.join(current_file_directory, '../../../../data/datamatrix/lifestyles.pickle')
with open(filepath, 'rb') as handle:
    DM_lfs = pickle.load(handle)
dm_pop = DM_lfs["ots"]["pop"]["lfs_population_"].copy()
dm_pop = dm_pop.filter({"Country" : ["Switzerland"]})

# make per capita
for v in dm.col_labels["Variables"]:
    dm.rename_col(v,"product-demand_" + v, "Variables")
dm.deepen()
dm.array = dm.array / dm_pop.array[...,np.newaxis]
dm.units["product-demand"] = "t/cap"
df_temp = dm.write_df()
# df_temp["product-demand_plastic-pack[t/cap]"]*1000
# mayble only plastic packaging is a bit low at 44kg, as it should be around 60kg per capita (it's 120 kg all plastic, and half
# should be packaging), but ok

# do fts
# for y in years_fts:
#     dm.add(dm[:,2023,...], "Years", [y])
# dm = linear_fitting(dm, years_fts, based_on=list(range(2010,2019+1)))
def linear_fitting_per_variab(dm, variable, year_start, year_end, years_fts):
    dm_temp = dm.filter({"Categories1" : [variable]})
    dm_temp = linear_fitting(dm_temp, years_fts, based_on=list(range(year_start,year_end+1)))
    dm.drop("Categories1",variable)
    dm.append(dm_temp,"Categories1")

dm.add(np.nan,"Years",years_fts,dummy=True)
linear_fitting_per_variab(dm, 'aluminium-pack', 2000, 2019, years_fts)
linear_fitting_per_variab(dm, 'glass-pack', 2000, 2019, years_fts)
linear_fitting_per_variab(dm, 'plastic-pack', 2003, 2019, years_fts)
linear_fitting_per_variab(dm, 'paper-pack', 2011, 2019, years_fts)
linear_fitting_per_variab(dm, 'paper-print', 2012, 2023, years_fts)
linear_fitting_per_variab(dm, 'paper-san', 2020, 2023, years_fts)

# dm.flatten().datamatrix_plot()

# save
years_ots = list(range(1990,2023+1))
years_fts = list(range(2025,2055,5))
dm_ots = dm.filter({"Years" : years_ots})
dm_fts = dm.filter({"Years" : years_fts})
DM_fts = {1: dm_fts.copy(), 2: dm_fts.copy(), 3: dm_fts.copy(), 4: dm_fts.copy()} # for now we set all levels to be the same
DM = {"ots" : dm_ots,
      "fts" : DM_fts}
f = os.path.join(current_file_directory, '../data/datamatrix/lever_paperpack.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)

del df, df_imp_exp, df_io, df_map_cpa, df_price, df_price_index,\
    df_price_ts, df_temp, dict_map, dm, DM, dm_calib_matprod_eu,\
    DM_fts, dm_io, DM_lfs, dm_netimp,\
    dm_ots, dm_out, dm_out_calib, dm_pop, dm_price, dm_temp, \
    dm_temp1, f, filepath, handle, key, my_aggregates, plastic_packaging_consumption_2023, \
    v, y, years_fts, years_ots, dm_fts









