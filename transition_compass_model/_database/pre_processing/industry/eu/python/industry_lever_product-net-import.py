
# packages
from model.common.data_matrix_class import DataMatrix
from model.common.auxiliary_functions import linear_fitting

import pandas as pd
import pickle
import os
import numpy as np
import warnings

# from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat
warnings.simplefilter("ignore")
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

# file
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/industry/eu/python/industry_lever_product-net-import.py"

# directories
current_file_directory = os.path.dirname(os.path.abspath(__file__))

#########################################
##### GET CLEAN DATAFRAME WITH DATA #####
#########################################

# get data
# df = eurostat.get_data_df("ds-056120")
filepath = os.path.join(current_file_directory, '../data/eurostat/ds-056120.csv')
# df.to_csv(filepath, index = False)
df = pd.read_csv(filepath)

# NOTE: as ds-056120 is on sold production, then we assume sold production = demand
# import and export should be always on "sold", as a country exports what's demanded
# and it imports what it demands.

# explore data
df.columns
product_code = "29102100"
variabs = ["PRODVAL","PRODQNT","EXPVAL","EXPQNT","IMPVAL","IMPQNT"]
df_sub = df.loc[df["prccode"].isin([product_code]),:]
df_sub = df_sub.loc[df_sub["indicators\\TIME_PERIOD"].isin(variabs),:]
len(df_sub["decl"].unique())
df_sub = df_sub.loc[df_sub["decl"].isin([4,2027])] # get germnay and EU27_2020
# it seems values are generally there for all variables
# I will do: 
# product-net-import[%] = (IMPQNT - EXPQNT) / PRODQNT
# For now I will do the adjustments (nan filling, jumps, predictions) on the variables 
# EXPQNT, IMPQNT and PRODQNT, and then I will make the variable product-net-import[%] at the end.
# The alternative would be to make product-net-import[%] from the
# beginning, and do all the adjustments on that variable. TBC.

# get "PRODQNT", "EXPQNT", "IMPQNT", "QNTUNIT"
variabs = ["PRODQNT", "EXPQNT", "IMPQNT", "QNTUNIT"]
df = df.loc[df["indicators\\TIME_PERIOD"].isin(variabs),:]

# keep only things in mapping for industry
filepath = os.path.join(current_file_directory, '../data/eurostat/PRODCOM2024_PRODCOM2023_Table.csv')
df_map = pd.read_csv(filepath)
df_map = df_map.loc[:,['PRODCOM2024_KEY','calc_industry_product']]
df_map = df_map.rename(columns= {"PRODCOM2024_KEY" : "prccode"})
df_map = df_map.dropna()
df = pd.merge(df, df_map, how="left", on=["prccode"])
df_sub = df.loc[~df["calc_industry_product"].isnull(),:]
df_sub = df_sub.loc[~df_sub["calc_industry_product"].isin(["battery"]),:] # drop battery for now

# fix countries
# sources:
# DECL drop down menu: https://ec.europa.eu/eurostat/databrowser/view/DS-056120/legacyMultiFreq/table?lang=en  
# https://ec.europa.eu/eurostat/documents/120432/0/Quick+guide+on+accessing+PRODCOM+data+DS-056120.pdf/484b8bbf-e371-49f3-6fa7-6a2514ebfcc9?t=1696602916356
decl_mapping = {1: "France", 3: "Netherlands", 4: "Germany", 5: "Italy", 6: "United Kingdom",
                7: "Ireland", 8: "Denmark", 9: "Greece", 10: "Portugal", 11: "Spain",
                17: "Belgium", 18: "Luxembourg", 24: "Iceland", 28: "Norway", 30: "Sweden",
                32: "Finland", 38: "Austria", 46: "Malta", 52: "Turkiye", 53: "Estonia",
                54: "Latvia", 55: "Lithuania", 60: "Poland", 61: "Czech Republic", 
                63: "Slovakia", 64: "Hungary", 66: "Romania", 68: "Bulgaria",
                70: "Albania", 91: "Slovenia", 92: "Croatia", 93: "Bosnia and Herzegovina",
                96: "North Macedonia", 97: "Montenegro", 98: "Serbia", 600: "Cyprus",
                1110: "EU15", 1111: "EU25", 1112: "EU27_2007", 2027: "EU27_2020", 2028: "EU28"}
df_sub["country"] = np.nan
for key in decl_mapping.keys():
    df_sub.loc[df_sub["decl"] == key,"country"] = decl_mapping[key]
    
# make long format
df_sub.rename(columns={"indicators\\TIME_PERIOD":"variable"}, inplace = True)
df_sub_unit = df_sub.loc[df_sub["variable"].isin(["QNTUNIT"]),:]
df_sub = df_sub.loc[~df_sub["variable"].isin(["QNTUNIT"]),:]
drops = ['freq', 'decl']
df_sub.drop(drops,axis=1, inplace = True)
indexes = ['prccode', 'variable', 'country', 'calc_industry_product']
df_sub = pd.melt(df_sub, id_vars = indexes, var_name='year')

# make unit as column
drops = ['freq', 'decl']
df_sub_unit.drop(drops,axis=1, inplace = True)
indexes = ['prccode', 'variable', 'country', 'calc_industry_product']
df_sub_unit = pd.melt(df_sub_unit, id_vars = indexes, var_name='year')
df_sub_unit.rename(columns={"value":"unit"}, inplace = True)
keep = ['prccode', 'country', 'calc_industry_product','year','unit']
indexes = ['prccode', 'country', 'calc_industry_product','year']
df_sub = pd.merge(df_sub, df_sub_unit.loc[:,keep], how="left", on=indexes)

# fix unit
df_sub["unit"].unique()
old_unit = [np.nan, 'kg ', 'm2 ', 'kg N ', 'kg P2O5 ', 'kg K2O ', 'kg effect ', 'p/st ', 'ct/l ', 'CGT ', 'NA ']
new_unit = [np.nan, 'kg', 'm2', 'kg N', 'kg P2O5', 'kg K2O', 'kg effect', 'p/st', 'ct/l', 'CGT', 'NA']
for i in range(0, len(old_unit)):
    df_sub.loc[df_sub["unit"] == old_unit[i],"unit"] = new_unit[i]
df_sub["unit"].unique()

# fix value
df_sub["value"] = [float(i) for i in df_sub["value"]]

# order and sort
indexes = ['country', 'variable', 'prccode', 'calc_industry_product', 'year']
variabs = ['value', 'unit']
df_sub = df_sub.loc[:,indexes + variabs]
df_sub = df_sub.sort_values(by=indexes)

# check
df_check = df_sub.loc[df_sub["prccode"] == "29102100",:]
df_check = df_check.loc[df_sub["country"].isin(["Germany","EU27_2020"])]
# ok

# aggregate by calc_industry_product
df_sub = df_sub.reset_index()
indexes = ['country', 'variable', 'calc_industry_product', 'year','unit']
df_sub = df_sub.groupby(indexes, as_index=False)['value'].agg(sum)

# keep right units
# df_sub["calc_industry_product"].unique()
# df_sub["unit"].unique()
# df_sub.loc[df_sub["calc_industry_product"].isin(["HDVH_ICE-diesel"]),"unit"].unique()
# ["aluminium-pack","glass-pack", "paper-pack", "paper-print", "paper-san", "plastic-pack"]
# product_check = ["plastic-pack"]
# df_check = df_sub.loc[df_sub["calc_industry_product"].isin(product_check),:]
# df_check = df_check.loc[df_check["country"].isin(["EU27_2020"]),:]
units_dict = {'HDVH_ICE-diesel' : ['p/st'], 'HDVL_ICE-diesel' : ['p/st'], 'HDVM_ICE-diesel' : ['p/st'], 
              'HDV_BEV' : ['p/st'], 'HDV_ICE-diesel' : ['p/st'], 'HDV_ICE-gasoline' : ['p/st'], 
              'HDV_PHEV-diesel' : ['p/st'], 'LDV_BEV' : ['p/st'], 'LDV_ICE-diesel' : ['p/st'], 
              'LDV_ICE-gasoline' : ['p/st'], 'LDV_PHEV-gasoline' : ['p/st'],
              'aluminium-pack' : ['p/st'], 'bus_ICE-diesel' : ['p/st'],
              'computer' : ['p/st'], 'dishwasher' : ['p/st'],
              'fertilizer' : ['kg', 'kg K2O', 'kg N', 'kg P2O5', 'kg effect'],
              'freezer' : ['p/st'], 'fridge' : ['p/st'], 'glass-pack' : ['kg'],
              'paper-pack' : ['kg'], 'paper-print' : ['kg'], 'paper-san' : ['kg'],
              'phone' : ['p/st'], 'aviation_ICE' : ['p/st'], 
              'plastic-pack' : ['kg'], 'rail_CEV' : ['p/st'],
              'rail_ICE-diesel' : ['p/st'],
              'marine_ICE-diesel' : ['p/st'], 
              'tv' : ['p/st'], 'wmachine' : ['p/st']}
df_sub = pd.concat([df_sub.loc[(df_sub["calc_industry_product"] == key) & \
                               (df_sub["unit"].isin(units_dict[key])),:] \
                    for key in units_dict.keys()])
df_sub.loc[df_sub["unit"] == 'p/st',"unit"] = "num"

# groupby for fertilizer
df_fert = df_sub.loc[df_sub["calc_industry_product"] == "fertilizer",:]
indexes = ['country', 'variable', 'calc_industry_product', 'year']
df_fert = df_fert.groupby(indexes, as_index=False)['value'].agg(sum)
df_fert["unit"] = "kg"
df_sub = df_sub.loc[~df_sub["calc_industry_product"].isin(["fertilizer"]),:]
df_sub = pd.concat([df_sub, df_fert])

# fix countries
countries_calc = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 
                  'Czech Republic', 'Denmark', 'EU27', 'Estonia', 'Finland', 
                  'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 
                  'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 
                  'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 
                  'Slovenia', 'Spain', 'Sweden', 'United Kingdom']
df_sub["country"].unique()
drops = ['Albania','Bosnia and Herzegovina','EU15','EU28','Iceland','Montenegro',
         'North Macedonia', 'Norway', 'Serbia', 'Turkiye']
df_sub = df_sub.loc[~df_sub["country"].isin(drops),:]
countries = df_sub["country"].unique()

# I assume that trade extra eu is from the data EU27_2020
df_sub = df_sub.loc[~df_sub["country"].isin(['EU27_2007']),:]
df_sub.loc[df_sub["country"] == 'EU27_2020',"country"] = "EU27"
df_temp = df_sub.loc[df_sub["country"] == "EU27",:]
countries = df_sub["country"].unique()

##################################
##### CONVERT TO DATA MATRIX #####
##################################

# make df ready for conversion to dm
df_sub.loc[df_sub["variable"] == "IMPQNT","variable"] = "product-import"
df_sub.loc[df_sub["variable"] == "EXPQNT","variable"] = "product-export"
df_sub.loc[df_sub["variable"] == "PRODQNT","variable"] = "product-demand"
df_sub["variable"] = df_sub["variable"] + "_" + df_sub["calc_industry_product"] + "[" + df_sub["unit"] + "]"
df_sub = df_sub.rename(columns={"country": "Country", "year" : "Years"})
drops = ["calc_industry_product","unit"]
df_sub.drop(drops,axis=1, inplace = True)
countries = df_sub["Country"].unique()
years = df_sub["Years"].unique()
variables = df_sub["variable"].unique()
panel_countries = np.repeat(countries, len(variables) * len(years))
panel_years = np.tile(np.tile(years, len(variables)), len(countries))
panel_variables = np.tile(np.repeat(variables, len(years)), len(countries))
df_temp = pd.DataFrame({"Country" : panel_countries, 
                        "Years" : panel_years, 
                        "variable" : panel_variables})
df_sub = pd.merge(df_temp, df_sub, how="left", on=["Country","Years","variable"])

# put nan where is 0
df_sub.loc[df_sub["value"] == 0,"value"] = np.nan

# split in dms
import re
df_sub["selection"] = [re.split("product-demand_|product-export_|product-import_", 
                                re.split("\\[",i)[0])[1] for i in df_sub["variable"]]

# dm_bld_domapp
df_temp = df_sub.loc[df_sub["selection"].isin(['computer', 'dishwasher', 'freezer', 'fridge',
                                               'phone', 'tv', 'wmachine']),["Country","Years","variable","value"]]
df_temp = df_temp.pivot(index=["Country","Years"], columns="variable", values='value').reset_index()
dm_bld_domapp = DataMatrix.create_from_df(df_temp, 1)

# dm_tra_veh
df_temp = df_sub.loc[df_sub["selection"].isin(['HDVH_ICE-diesel', 'HDVL_ICE-diesel', 'HDVM_ICE-diesel', 'HDV_BEV',
                                               'HDV_ICE-diesel', 'HDV_ICE-gasoline', 'HDV_PHEV-diesel', 'LDV_BEV',
                                               'LDV_ICE-diesel', 'LDV_ICE-gasoline', 'LDV_PHEV-gasoline', 'aviation_ICE',
                                               'bus_ICE-diesel', 'marine_ICE-diesel', 'rail_CEV', 'rail_ICE-diesel']),
                     ["Country","Years","variable","value"]]
df_temp = df_temp.pivot(index=["Country","Years"], columns="variable", values='value').reset_index()
dm_tra_veh = DataMatrix.create_from_df(df_temp, 2)
dm_tra_veh = dm_tra_veh.flatten()
dm_tra_veh.groupby({"HDV_ICE-diesel" : ["HDVH_ICE-diesel","HDVM_ICE-diesel","HDVL_ICE-diesel","HDV_ICE-diesel"]}, 
                   "Categories1", inplace=True)
idx = dm_tra_veh.idx
for y in range(1995,2002+1):
    dm_tra_veh.array[:,idx[y],:,idx["HDV_ICE-diesel"]] = np.nan
dm_tra_veh.deepen()

# dm_pack
df_temp = df_sub.loc[df_sub["selection"].isin(['glass-pack', 'plastic-pack', 
                                               'paper-pack', 'paper-print', 
                                               'paper-san']),["Country","Years","variable","value"]]
df_temp = df_temp.pivot(index=["Country","Years"], columns="variable", values='value').reset_index()
dm_pack_kg = DataMatrix.create_from_df(df_temp, 1)
df_temp = df_sub.loc[df_sub["selection"].isin(['aluminium-pack']),
                     ["Country","Years","variable","value"]]
df_temp = df_temp.pivot(index=["Country","Years"], columns="variable", values='value').reset_index()
dm_pack_unit = DataMatrix.create_from_df(df_temp, 1)

# put together
dm_trade = dm_bld_domapp.flatten()
dm_trade.append(dm_tra_veh.flatten().flatten(),"Variables")
dm_trade.append(dm_pack_kg.flatten(),"Variables")
dm_trade.append(dm_pack_unit.flatten(),"Variables")
dm_trade.sort("Variables")

# check
# dm_trade.filter({"Country" : ["EU27"]}).datamatrix_plot()

###################
##### FIX OTS #####
###################

# Set years range
years_setting = [1990, 2023, 2050, 5]
startyear = years_setting[0]
baseyear = years_setting[1]
lastyear = years_setting[2]
step_fts = years_setting[3]
years_ots = list(range(startyear, baseyear+1, 1))
years_fts = list(range(baseyear+2, lastyear+1, step_fts))
years_all = years_ots + years_fts

# new variabs list
dict_new = {}

# put missing values where needed
idx = dm_trade.idx
for y in range(1995,2010+1):
    dm_trade.array[...,idx[y],idx["product-demand_aluminium-pack"]] = np.nan
    dm_trade.array[...,idx[y],idx["product-export_aluminium-pack"]] = np.nan
    dm_trade.array[...,idx[y],idx["product-export_glass-pack"]] = np.nan
for y in range(2022,2023+1):
    dm_trade.array[...,idx[y],idx["product-demand_fridge"]] = np.nan
    dm_trade.array[...,idx[y],idx["product-export_fridge"]] = np.nan
    dm_trade.array[...,idx[y],idx["product-import_fridge"]] = np.nan
for v in ["phone","aviation_ICE"]:
    dm_trade.array[:,idx[2023],idx["product-demand_" + v]] = np.nan
    dm_trade.array[:,idx[2023],idx["product-export_" + v]] = np.nan
    dm_trade.array[:,idx[2023],idx["product-import_" + v]] = np.nan
    dm_trade.array[:,idx[1995],idx["product-demand_" + v]] = dm_trade.array[:,idx[2022],idx["product-demand_" + v]]/(2022-1995)
    dm_trade.array[:,idx[1995],idx["product-export_" + v]] = dm_trade.array[:,idx[2022],idx["product-export_" + v]]/(2022-1995)
    dm_trade.array[:,idx[1995],idx["product-import_" + v]] = dm_trade.array[:,idx[2022],idx["product-import_" + v]]/(2022-1995)
for v in ["HDV_BEV","HDV_ICE-gasoline","HDV_PHEV-diesel"]:
    dm_trade.array[:,idx[2022],idx["product-demand_" + v]] = np.nan
    dm_trade.array[:,idx[2022],idx["product-export_" + v]] = np.nan
    dm_trade.array[:,idx[2022],idx["product-import_" + v]] = np.nan
    dm_trade.array[:,idx[1995],idx["product-demand_" + v]] = dm_trade.array[:,idx[2023],idx["product-demand_" + v]]/(2023-1995)
    dm_trade.array[:,idx[1995],idx["product-export_" + v]] = dm_trade.array[:,idx[2023],idx["product-export_" + v]]/(2023-1995)
    dm_trade.array[:,idx[1995],idx["product-import_" + v]] = dm_trade.array[:,idx[2023],idx["product-import_" + v]]/(2023-1995)
for y in range(2009,2021+1):
    dm_trade.array[...,idx[y],idx["product-demand_HDV_ICE-diesel"]] = np.nan
    dm_trade.array[...,idx[y],idx["product-export_HDV_ICE-diesel"]] = np.nan
    dm_trade.array[...,idx[y],idx["product-import_HDV_ICE-diesel"]] = np.nan
for y in [2003,2015,2019,2020]:
    dm_trade.array[...,idx[y],idx["product-demand_rail_CEV"]] = np.nan
dm_trade.array[...,idx[2017],idx["product-demand_rail_ICE-diesel"]] = np.nan
for y in range(1995,2006+1):
    dm_trade.array[...,idx[y],idx["product-import_rail_CEV"]] = np.nan
for y in [2019,2020]:
    dm_trade.array[...,idx[y],idx["product-import_bus_ICE-diesel"]] = np.nan
dm_trade.array[:,idx[2022],idx["product-demand_marine_ICE-diesel"]] = np.nan
dm_trade.array[:,idx[2022],idx["product-export_marine_ICE-diesel"]] = np.nan
# dm_trade.array[:,idx[2023],idx["product-import_HDV_PHEV-diesel"]] = np.nan
dm_trade.array[:,idx[1995],idx["product-import_HDV_PHEV-diesel"]] = dm_trade.array[:,idx[2022],idx["product-import_HDV_PHEV-diesel"]]/(2022-1995)
for y in range(1996,2021+1):
    dm_trade.array[:,idx[y],idx["product-import_HDV_PHEV-diesel"]] = dm_trade.array[:,idx[1995],idx["product-import_HDV_PHEV-diesel"]]
    

# function to adjust ots
def make_ots(variable, based_on):
    dm_temp = dm_trade.filter({"Variables" : [variable]})
    dm_temp = linear_fitting(dm_temp, years_ots, based_on=based_on, min_t0=0.1,min_tb=0.1)
    return dm_temp

dict_call = {"product-demand_HDV_BEV" : None,
             "product-demand_HDV_ICE-diesel" : None,
             "product-demand_HDV_ICE-gasoline" : None,
             "product-demand_HDV_PHEV-diesel" : None,
             "product-demand_LDV_BEV" : list(range(2017,2018+1)),
             "product-demand_LDV_ICE-diesel" : None,
             "product-demand_LDV_ICE-gasoline" : None,
             "product-demand_LDV_PHEV-gasoline" : list(range(2017,2019+1)),
             "product-demand_aluminium-pack": None,
             "product-demand_aviation_ICE": None,
             "product-demand_bus_ICE-diesel": None,
             "product-demand_computer" : list(range(2003,2011+1)),
             "product-demand_dishwasher" : None,
             "product-demand_freezer" : None,
             "product-demand_fridge" : None,
             "product-demand_glass-pack" : list(range(2010,2023+1)),
             "product-demand_marine_ICE-diesel" : None,
             "product-demand_paper-pack" : None,
             "product-demand_paper-print" : list(range(2003,2009+1)),
             "product-demand_paper-san" : None,
             "product-demand_phone" : None,
             "product-demand_plastic-pack" : None,
             "product-demand_rail_CEV" : None,
             "product-demand_rail_ICE-diesel" : None,
             "product-demand_tv" : list(range(2003,2010+1)),
             "product-demand_wmachine" : None,
             "product-export_HDV_BEV" : None,
             "product-export_HDV_ICE-diesel" : None,
             "product-export_HDV_ICE-gasoline" : None,
             "product-export_HDV_PHEV-diesel" : None,
             "product-export_LDV_BEV" : list(range(2017,2018+1)),
             "product-export_LDV_ICE-diesel" : list(range(2003,2019+1)),
             "product-export_LDV_ICE-gasoline" : list(range(2003,2007+1)),
             "product-export_LDV_PHEV-gasoline" : list(range(2017,2018+1)),
             "product-export_aluminium-pack": None,
             "product-export_aviation_ICE": None,
             "product-export_bus_ICE-diesel": None,
             "product-export_computer" : list(range(2003,2011+1)),
             "product-export_dishwasher" : None,
             "product-export_freezer" : None,
             "product-export_fridge" : None,
             "product-export_glass-pack" : list(range(2011,2014+1)),
             "product-export_marine_ICE-diesel" : list(range(2003,2007+1)),
             "product-export_paper-pack" : None,
             "product-export_paper-print" : list(range(2003,2009+1)),
             "product-export_paper-san" : None,
             "product-export_phone" : None,
             "product-export_plastic-pack" : None,
             "product-export_rail_CEV" : None,
             "product-export_tv" : list(range(2003,2008+1)),
             "product-export_wmachine" : None,
             "product-import_HDV_BEV" : None,
             "product-import_HDV_ICE-diesel" : None,
             "product-import_HDV_ICE-gasoline" : None,
             "product-import_HDV_PHEV-diesel" : None,
             "product-import_LDV_BEV" : list(range(2017,2018+1)),
             "product-import_LDV_ICE-diesel" : list(range(2003,2007+1)),
             "product-import_LDV_ICE-gasoline" : None,
             "product-import_LDV_PHEV-gasoline" : list(range(2017,2019+1)),
             "product-import_aluminium-pack": list(range(2003,2016+1)),
             "product-import_aviation_ICE": None,
             "product-import_bus_ICE-diesel": None,
             "product-import_computer" : list(range(2003,2011+1)),
             "product-import_dishwasher" : None,
             "product-import_freezer" : None,
             "product-import_fridge" : None,
             "product-import_glass-pack" : None,
             "product-import_marine_ICE-diesel" : None,
             "product-import_paper-pack" : None,
             "product-import_paper-print" : None,
             "product-import_paper-san" : None,
             "product-import_phone" : None,
             "product-import_plastic-pack" : None,
             "product-import_rail_CEV" : None,
             "product-import_tv" : None,
             "product-import_wmachine" : None}

for key in dict_call.keys(): dict_new[key] = make_ots(key, based_on=dict_call[key])

# append
dm_trade_temp = dict_new["product-demand_aluminium-pack"].copy()
mylist = list(dict_call.keys())
mylist.remove("product-demand_aluminium-pack")
for v in mylist:
    dm_trade_temp.append(dict_new[v],"Variables")
dm_trade_temp.sort("Variables")
dm_trade = dm_trade_temp.copy()

# check
# dm_trade.filter({"Country" : ["EU27"]}).datamatrix_plot()

#############################################
##### GENERATE VARIABLES WE DO NOT HAVE #####
#############################################

# put together
DM_trade = {}
dm_temp = dm_trade.filter({"Variables" : ['product-demand_HDV_BEV', 'product-demand_HDV_ICE-diesel', 
                                          'product-demand_HDV_ICE-gasoline', 'product-demand_HDV_PHEV-diesel', 
                                          'product-demand_LDV_BEV', 'product-demand_LDV_ICE-diesel', 
                                          'product-demand_LDV_ICE-gasoline', 'product-demand_LDV_PHEV-gasoline',
                                          'product-demand_aviation_ICE', 'product-demand_bus_ICE-diesel',
                                          'product-demand_marine_ICE-diesel', 
                                          'product-demand_rail_CEV', 'product-demand_rail_ICE-diesel',
                                          'product-export_HDV_BEV', 'product-export_HDV_ICE-diesel', 
                                          'product-export_HDV_ICE-gasoline', 'product-export_HDV_PHEV-diesel', 
                                          'product-export_LDV_BEV', 'product-export_LDV_ICE-diesel', 
                                          'product-export_LDV_ICE-gasoline', 'product-export_LDV_PHEV-gasoline',
                                          'product-export_aviation_ICE', 'product-export_bus_ICE-diesel',
                                          'product-export_marine_ICE-diesel', 
                                          'product-export_rail_CEV',
                                          'product-import_HDV_ICE-gasoline', 'product-import_HDV_PHEV-diesel', 
                                          'product-import_LDV_BEV', 'product-import_LDV_ICE-diesel', 
                                          'product-import_LDV_ICE-gasoline', 'product-import_LDV_PHEV-gasoline',
                                          'product-import_aviation_ICE', 'product-import_bus_ICE-diesel',
                                          'product-import_marine_ICE-diesel', 
                                          'product-import_rail_CEV']})
dm_temp.deepen_twice()
DM_trade["tra-veh"] = dm_temp
dm_temp = dm_trade.filter({"Variables" : ['product-demand_aluminium-pack','product-export_aluminium-pack',
                                          'product-import_aluminium-pack']})
dm_temp.deepen()
DM_trade["pack-alu"] = dm_temp
dm_temp = dm_trade.filter({"Variables" : ['product-demand_computer', 'product-demand_dishwasher', 
                                          'product-demand_freezer', 'product-demand_fridge',
                                          'product-demand_phone', 'product-demand_tv', 'product-demand_wmachine',
                                          'product-export_computer', 'product-export_dishwasher', 
                                          'product-export_freezer', 'product-export_fridge',
                                          'product-export_phone', 'product-export_tv', 'product-export_wmachine',
                                          'product-import_computer', 'product-import_dishwasher', 
                                          'product-import_freezer', 'product-import_fridge',
                                          'product-import_phone', 'product-import_tv', 'product-import_wmachine']})
dm_temp.deepen()
DM_trade["domapp"] = dm_temp
dm_temp = dm_trade.filter({"Variables" : ['product-demand_glass-pack','product-demand_paper-pack', 
                                          'product-demand_paper-print', 'product-demand_paper-san',
                                          'product-demand_plastic-pack',
                                          'product-export_glass-pack','product-export_paper-pack', 
                                          'product-export_paper-print', 'product-export_paper-san',
                                          'product-export_plastic-pack',
                                          'product-import_glass-pack','product-import_paper-pack', 
                                          'product-import_paper-print', 'product-import_paper-san',
                                          'product-import_plastic-pack']})
dm_temp.deepen()
DM_trade["pack"] = dm_temp

# note: for the variables that we do not have, in general import and export will be set
# to zero, and demand will be set to nan

# generate cars-FCV and trucks-FCV
# we assume that imports remain zero throughout
idx = DM_trade["tra-veh"].idx
DM_trade["tra-veh"].add(0, "Categories2", "FCEV", unit="num", dummy=True)
DM_trade["tra-veh"].array[:,:,idx["product-demand"],:,idx["FCEV"]] = np.nan
DM_trade["tra-veh"].add(0, "Categories2", "ICE-gas", unit="num", dummy=True)
DM_trade["tra-veh"].array[:,:,idx["product-demand"],:,idx["ICE-gas"]] = np.nan
DM_trade["tra-veh"].sort("Categories2")

# generate new-dhg-pipe, rail, road, trolley-cables, floor-area-new-non-residential, 
# floor-area-new-residential, floor-area-reno-non-residential, floor-area-reno-residential
# we assume imports of these are all zero
DM_trade["domapp"].add(0, "Categories1", "new-dhg-pipe", unit="num", dummy=True)
dm_bld_pipe = DM_trade["domapp"].filter({"Categories1" : ["new-dhg-pipe"]})
dm_bld_pipe.units['product-export'] = "km"
dm_bld_pipe.units['product-import'] = "km"
dm_bld_pipe.units['product-demand'] = "km"
idx = dm_bld_pipe.idx
dm_bld_pipe.array[:,:,idx["product-demand"],idx["new-dhg-pipe"]] = np.nan
DM_trade["domapp"].drop("Categories1", ["new-dhg-pipe"])
DM_trade["pipe"] = dm_bld_pipe

dm_tra_infra = dm_bld_pipe.copy()
dm_tra_infra.rename_col("new-dhg-pipe","rail","Categories1")
dm_temp = dm_tra_infra.copy()
dm_temp.rename_col("rail","road","Categories1")
dm_tra_infra.append(dm_temp, "Categories1")
dm_temp = dm_tra_infra.filter({"Categories1" : ["rail"]})
dm_temp.rename_col("rail","trolley-cables","Categories1")
dm_tra_infra.append(dm_temp, "Categories1")
DM_trade["tra-infra"] = dm_tra_infra.copy()

DM_trade["domapp"].add(0, "Categories1", "floor-area-new-non-residential", unit="m2", dummy=True)
DM_trade["domapp"].add(0, "Categories1", "floor-area-new-residential", unit="m2", dummy=True)
DM_trade["domapp"].add(0, "Categories1", "floor-area-reno-non-residential", unit="m2", dummy=True)
DM_trade["domapp"].add(0, "Categories1", "floor-area-reno-residential", unit="m2", dummy=True)
dm_bld_floor = DM_trade["domapp"].filter({"Categories1" : ["floor-area-new-non-residential","floor-area-new-residential", 
                                                           "floor-area-reno-non-residential", "floor-area-reno-residential"]})
dm_bld_floor.units['product-export'] = "m2"
dm_bld_floor.units['product-import'] = "m2"
dm_bld_floor.units['product-demand'] = "m2"
idx = dm_bld_floor.idx
dm_bld_floor.array[:,:,idx["product-demand"],:] = np.nan
DM_trade["domapp"].drop("Categories1", ["floor-area-new-non-residential","floor-area-new-residential", 
                                        "floor-area-reno-non-residential", "floor-area-reno-residential"])
dm_bld_floor.sort("Categories1")
DM_trade["bld-floor"] = dm_bld_floor

# dryer
# I assume dryers are 1% of exports and imports of w machine (check excel file in WITS folder called "percentage_dryers_export_EU")
idx = DM_trade["domapp"].idx
arr_temp = DM_trade["domapp"].array[...,idx["wmachine"]] * 0.01
DM_trade["domapp"].add(arr_temp, col_label="dryer", dim='Categories1', unit = "num")
DM_trade["domapp"].sort("Categories1")

# check
# DM_trade['tra-veh'].filter({"Country" : ["EU27"]}).datamatrix_plot()
# DM_trade['pack-alu'].filter({"Country" : ["EU27"]}).datamatrix_plot()
# DM_trade['domapp'].filter({"Country" : ["EU27"]}).datamatrix_plot()
# DM_trade['pack'].filter({"Country" : ["EU27"]}).datamatrix_plot()

####################
##### MAKE FTS #####
####################

# flatten tra veh and make electric total and total which will be used for the projections
DM_trade["tra-veh"] = DM_trade["tra-veh"].flatten()
dm_temp = DM_trade["tra-veh"].groupby({"electric_total" : ['HDV_BEV','HDV_PHEV-diesel','LDV_BEV','LDV_PHEV-gasoline'], 
                                       "total" : ['HDV_BEV', 'HDV_FCEV', 'HDV_ICE-diesel', 'HDV_ICE-gas', 'HDV_ICE-gasoline', 
                                                  'HDV_PHEV-diesel', 'LDV_BEV', 'LDV_FCEV', 'LDV_ICE-diesel', 'LDV_ICE-gas', 
                                                  'LDV_ICE-gasoline', 'LDV_PHEV-gasoline']},"Categories1",inplace=False)
DM_trade["tra-veh"].append(dm_temp,"Categories1")
DM_trade["tra-veh"].drop("Categories1",['aviation_FCEV','aviation_ICE-gas','bus_FCEV','marine_FCEV',
                                        'rail_FCEV','rail_ICE-gas', 'marine_ICE-gas'])

# make function to fill in missing years fts for EU27 with linear fitting
def make_fts(dm, variable, year_start, year_end, country = "EU27", dim = "Categories1", 
             min_t0=0, min_tb=0, years_fts = years_fts): # I put minimum to 1 so it does not go to zero
    dm = dm.copy()
    idx = dm.idx
    based_on_yars = list(range(year_start, year_end + 1, 1))
    dm_temp = linear_fitting(dm.filter({"Country" : [country], dim : [variable]}), 
                             years_ots = years_fts, min_t0=min_t0, min_tb=min_tb, based_on = based_on_yars)
    idx_temp = dm_temp.idx
    if dim == "Variables":
        dm.array[idx[country],:,idx[variable],...] = \
            np.round(dm_temp.array[idx_temp[country],:,idx_temp[variable],...],0)
    if dim == "Categories1":
        dm.array[idx[country],:,:,idx[variable]] = \
            np.round(dm_temp.array[idx_temp[country],:,:,idx_temp[variable]], 0)
    if dim == "Categories2":
        dm.array[idx[country],:,:,:,idx[variable]] = \
            np.round(dm_temp.array[idx_temp[country],:,:,:,idx_temp[variable]], 0)
    if dim == "Categories3":
        dm.array[idx[country],:,:,:,:,idx[variable]] = \
            np.round(dm_temp.array[idx_temp[country],:,:,:,:,idx_temp[variable]], 0)
    
    return dm

# add missing years fts
for key in DM_trade.keys():
    DM_trade[key].add(np.nan, col_label=years_fts, dummy=True, dim='Years')

# set default time window for linear trend
# assumption: best is taking longer trend possible to make predictions to 2050 (even if earlier data is generated)
baseyear_start = 1990
baseyear_end = 2023

# packages
DM_trade["pack-alu"] = make_fts(DM_trade["pack-alu"], "aluminium-pack", baseyear_start, baseyear_end)
DM_trade["pack"] = make_fts(DM_trade["pack"], "glass-pack", 2012, baseyear_end) # here upwatd trend in import and demand starts in 2012
DM_trade["pack"] = make_fts(DM_trade["pack"], "plastic-pack", baseyear_start, baseyear_end)
DM_trade["pack"] = make_fts(DM_trade["pack"], "paper-pack", 2009, baseyear_end)
DM_trade["pack"] = make_fts(DM_trade["pack"], "paper-print", baseyear_start, baseyear_end)
DM_trade["pack"] = make_fts(DM_trade["pack"], "paper-san", baseyear_start, baseyear_end)
# product = "plastic-pack"
# (make_fts(DM_trade["pack"], product, baseyear_start, baseyear_end).
#   datamatrix_plot(selected_cols={"Country" : ["EU27"],
#                                 "Categories1" : [product]}))


# electric vehicles

# note: assuming 8% of total fleet being electric in 2050
# source: https://www.eea.europa.eu/publications/electric-vehicles-and-the-energy/download
DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "total", baseyear_start, baseyear_end)
idx = DM_trade["tra-veh"].idx
electric_2050 = np.round(DM_trade["tra-veh"].array[idx["EU27"],idx[2050],:,idx["total"]] * 0.20) # in the end I put 20% here as with this data electric things are already close to 8% in 2023
dm_share = DM_trade["tra-veh"].filter({"Country" : ["EU27"], "Years" : [2023], 
                                       "Categories1" : ['HDV_BEV','HDV_PHEV-diesel','LDV_BEV','LDV_PHEV-gasoline']})
dm_share.normalise("Categories1")
idx_share = dm_share.idx
HDV_BEV_2050 = np.round(dm_share.array[...,idx_share["HDV_BEV"]] * electric_2050,0)
HDV_PHEV_diesel_2050 = np.round(dm_share.array[...,idx_share["HDV_PHEV-diesel"]] * electric_2050,0)
LDV_BEV_2050 = np.round(dm_share.array[...,idx_share["LDV_BEV"]] * electric_2050,0)
LDV_PHEV_gasoline_2050 = np.round(dm_share.array[...,idx_share["LDV_PHEV-gasoline"]] * electric_2050,0)
DM_trade["tra-veh"].array[idx["EU27"],idx[2050],:,idx["HDV_BEV"]] = HDV_BEV_2050
DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "HDV_BEV", 2023, 2050)
DM_trade["tra-veh"].array[idx["EU27"],idx[2050],:,idx["HDV_PHEV-diesel"]] = HDV_PHEV_diesel_2050
DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "HDV_PHEV-diesel", 2023, 2050)
DM_trade["tra-veh"].array[idx["EU27"],idx[2050],:,idx["LDV_BEV"]] = LDV_BEV_2050
DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "LDV_BEV", 2023, 2050)
DM_trade["tra-veh"].array[idx["EU27"],idx[2050],:,idx["LDV_PHEV-gasoline"]] = LDV_PHEV_gasoline_2050
DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "LDV_PHEV-gasoline", 2023, 2050)
DM_trade["tra-veh"].drop("Categories1", ["electric_total","total"])

# DM_trade['tra-veh'].filter({"Country" : ["EU27"]}).datamatrix_plot()

# rest of transport
DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "HDV_FCEV", baseyear_start, baseyear_end)
DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "HDV_ICE-diesel", baseyear_start, baseyear_end)
DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "HDV_ICE-gas", baseyear_start, baseyear_end)
DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "HDV_ICE-gasoline", baseyear_start, baseyear_end)
DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "LDV_FCEV", baseyear_start, baseyear_end)
DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "LDV_ICE-diesel", baseyear_start, baseyear_end)
DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "LDV_ICE-gas", baseyear_start, baseyear_end)
DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "LDV_ICE-gasoline", baseyear_start, baseyear_end)
DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "aviation_ICE", baseyear_start, baseyear_end)
DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "bus_ICE-diesel", baseyear_start, baseyear_end)
DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "bus_ICE-gas", baseyear_start, baseyear_end)
DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "marine_ICE-diesel", baseyear_start, baseyear_end)
DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "rail_CEV", baseyear_start, baseyear_end)
DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "rail_ICE-diesel", baseyear_start, baseyear_end)

# rename rail to train, aviation to planes, marine to ships
DM_trade["tra-veh"].rename_col_regex("rail","trains","Categories1")
DM_trade["tra-veh"].rename_col_regex("aviation","planes","Categories1")
DM_trade["tra-veh"].rename_col_regex("marine","ships","Categories1")
DM_trade["tra-veh"].sort("Categories1")

# transport infra
DM_trade["tra-infra"] = make_fts(DM_trade["tra-infra"], "rail", baseyear_start, baseyear_end)
DM_trade["tra-infra"] = make_fts(DM_trade["tra-infra"], "road", baseyear_start, baseyear_end)
DM_trade["tra-infra"] = make_fts(DM_trade["tra-infra"], "trolley-cables", baseyear_start, baseyear_end)

# buildings
DM_trade["bld-floor"] = make_fts(DM_trade["bld-floor"], "floor-area-new-non-residential", baseyear_start, baseyear_end)
DM_trade["bld-floor"] = make_fts(DM_trade["bld-floor"], "floor-area-new-residential", baseyear_start, baseyear_end)
DM_trade["bld-floor"] = make_fts(DM_trade["bld-floor"], "floor-area-reno-non-residential", baseyear_start, baseyear_end)
DM_trade["bld-floor"] = make_fts(DM_trade["bld-floor"], "floor-area-reno-residential", baseyear_start, baseyear_end)

# domestic appliances
DM_trade["domapp"] = make_fts(DM_trade["domapp"], "computer", baseyear_start, baseyear_end)
DM_trade["domapp"] = make_fts(DM_trade["domapp"], "dishwasher", baseyear_start, baseyear_end)
DM_trade["domapp"] = make_fts(DM_trade["domapp"], "dryer", 2000, 2007) # here I assume there is some problem with the data after 2008
DM_trade["domapp"] = make_fts(DM_trade["domapp"], "freezer", baseyear_start, baseyear_end)
DM_trade["domapp"] = make_fts(DM_trade["domapp"], "fridge", baseyear_start, baseyear_end)
DM_trade["domapp"] = make_fts(DM_trade["domapp"], "phone", baseyear_start, baseyear_end)
DM_trade["domapp"] = make_fts(DM_trade["domapp"], "tv", 2012, baseyear_end) # downward trend in demand since 2012
DM_trade["domapp"] = make_fts(DM_trade["domapp"], "wmachine", 2000, 2007) # here I assume there is some problem with the data after 2008

# pipes
DM_trade["pipe"] = make_fts(DM_trade["pipe"], "new-dhg-pipe", baseyear_start, baseyear_end)

# check
# DM_trade['tra-veh'].filter({"Country" : ["EU27"]}).datamatrix_plot()
# DM_trade['pack-alu'].filter({"Country" : ["EU27"]}).datamatrix_plot()
# DM_trade['domapp'].filter({"Country" : ["EU27"]}).datamatrix_plot()
# DM_trade['pack'].filter({"Country" : ["EU27"]}).datamatrix_plot()

###################################
##### MAKE PRODUCT NET IMPORT #####
###################################

# dm_temp = DM_trade["tra-veh"].filter_w_regex({"Categories1": "LDV"})
# dm_temp.group_all("Categories1")
# dm_temp.operation("product-export","/","product-demand",'Variables',
#                   'product-export-share',unit="%")
# dm_temp.operation("product-import","/","product-demand",'Variables',
#                   'product-import-share',unit="%")
# idx = dm_temp.idx
# dm_temp.array[idx["EU27"],idx[2021],idx["product-import-share"]] # 0.14
# dm_temp.array[idx["EU27"],idx[2021],idx["product-export-share"]] # 0.19

# product-net-import[%] = (product-import - product-export)/product-demand
DM_trade_net_share = {}
keys = ['domapp', 'tra-veh', 'pack', 'pack-alu', 'pipe', 'tra-infra', 'bld-floor']
for key in keys:
    dm_temp = DM_trade[key].copy() 

    # make product-net-import[%] = (product-import - product-export)/product-demand
    idx = dm_temp.idx
    arr_temp = dm_temp.array
    arr_temp[np.isnan(arr_temp)] = 0 # put zero where nan is
    arr_net = (arr_temp[:,:,idx["product-import"],:] - arr_temp[:,:,idx["product-export"],:]) / arr_temp[:,:,idx["product-demand"],:]
    
    # when both import and export are zero, assign a zero
    arr_net[(arr_temp[:,:,idx["product-import"],:] == 0) & (arr_temp[:,:,idx["product-export"],:] == 0)] = 0
    dm_temp.add(arr_net[:,:,np.newaxis,:], "Variables", "product-net-import", unit="%")
    
    # drop
    dm_temp.drop("Variables", ["product-import","product-export","product-demand"])
    
    # store
    DM_trade_net_share[key] = dm_temp

dm_trade_netshare = DM_trade_net_share["domapp"].copy()
keys = ['tra-veh', 'pack', 'pack-alu', 'pipe', 'tra-infra', 'bld-floor']
for key in keys:
    dm_trade_netshare.append(DM_trade_net_share[key], "Categories1")
dm_trade_netshare.sort("Categories1")

# fill in missing values for product-net-import (coming from dividing by zero)
idx = dm_trade_netshare.idx
dm_trade_netshare.array[np.isinf(dm_trade_netshare.array)] = np.nan
years_fitting = dm_trade_netshare.col_labels["Years"]
dm_trade_netshare = linear_fitting(dm_trade_netshare, years_fitting)

# for the variables that we generated as all zero, re-put zeroes
variabs = ['HDV_FCEV', 'HDV_ICE-gas', 'LDV_FCEV', 'LDV_ICE-gas', 'bus_ICE-gas',
           "new-dhg-pipe", "rail",
           "road", "trolley-cables", "floor-area-new-non-residential", 
           "floor-area-new-residential", "floor-area-reno-non-residential", 
           "floor-area-reno-residential"]
idx = dm_trade_netshare.idx
for v in variabs:
    dm_trade_netshare.array[:,:,:,idx[v]] = 0
    
# # fix jumps in product-net-import
# dm_trade_netshare = fix_jumps_in_dm(dm_trade_netshare)

# Should having values above and below, respectively, 1 and -1 be a problem? probably
# it is if it's larger than 1, as it would mean that we are importing more than the demand ... the
# only reason why that could be is that a the EU27 is importing just to re-export, though
# I guess this is not the norm, and I would probably rule out this situation ...
# on the other hand if it's less than -1 should be fine, as it would mean we are producing
# more than the local demand, and the rest is exported.
# for trains, we have values well above and below, respectively, 1 and -1.
# For computers, we have values well above 1.
# For dryer, we have values below -1 (gets to -3.5).
# For freezer, fridge, phone, we have values well above 1.
# Let's cap everything to max 1
dm_trade_netshare.array[dm_trade_netshare.array>1]=1

# add HDV_PHEV-gasoline and LDV_PHEV-diesel
idx = dm_trade_netshare.idx
arr_temp = dm_trade_netshare.array[:,:,:,idx["HDV_PHEV-diesel"]]
dm_trade_netshare.add(arr_temp, "Categories1", "HDV_PHEV-gasoline", unit="%")
arr_temp = dm_trade_netshare.array[:,:,:,idx["LDV_PHEV-gasoline"]]
dm_trade_netshare.add(arr_temp, "Categories1", "LDV_PHEV-diesel", unit="%")
dm_trade_netshare.sort("Categories1")

# check
# dm_trade_netshare.filter({"Country" : ["EU27"]}).datamatrix_plot()
# DM_trade["tra-veh"].filter({"Country" : ["EU27"]}).datamatrix_plot()
# df_check = dm_trade_netshare.filter({"Country" : ["EU27"]}).write_df()


################################
##### MAKE PAPERPACK LEVER #####
################################

# the paper pack lever is the demand for packages per capita. The unit is ton/cap.
# for the population, we upload the population data in lifestyles

# load DM_pack
filepath = os.path.join(current_file_directory, '../../../../data/datamatrix/lifestyles.pickle')
with open(filepath, 'rb') as handle:
    DM_pack = pickle.load(handle)
    
# get population data
dm_pop = DM_pack["ots"]["pop"]["lfs_population_"].copy()
dm_pop.append(DM_pack["fts"]["pop"]["lfs_population_"][1], "Years")

# get aluminium package data
dm_alu = DM_trade["pack-alu"].filter({"Variables" : ["product-demand"]})

# assuming an average of 30 g per unit, so 0.03 kg per unit
dm_alu.array = dm_alu.array * 0.03
dm_alu.units["product-demand"] = "kg"

# put together with rest of packaging (which is already in kg)
dm_pack = DM_trade["pack"].filter({"Variables" : ["product-demand"]})
dm_pack.append(dm_alu, "Categories1")
dm_pack.sort("Categories1")

# make kg to tonnes
dm_pack.change_unit('product-demand', factor=1e-3, old_unit='kg', new_unit='t')

# make tonne per capita
dm_pop.drop("Country",['Switzerland','Vaud'])
dm_pack.array = dm_pack.array / dm_pop.array[...,np.newaxis]
dm_pack.units["product-demand"] = "t/cap"

################
##### SAVE #####
################

# save dm_trade_netshare
years_ots = list(range(1990,2023+1))
years_fts = list(range(2025,2055,5))
dm_ots = dm_trade_netshare.filter({"Years" : years_ots})
dm_fts = dm_trade_netshare.filter({"Years" : years_fts})
DM_fts = {1: dm_fts.copy(), 2: dm_fts.copy(), 3: dm_fts.copy(), 4: dm_fts.copy()} # for now we set all levels to be the same
DM = {"ots" : dm_ots,
      "fts" : DM_fts}
f = os.path.join(current_file_directory, '../data/datamatrix/lever_product-net-import.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)

# save paperpack
years_ots = list(range(1990,2023+1))
years_fts = list(range(2025,2055,5))
dm_ots = dm_pack.filter({"Years" : years_ots})
dm_fts = dm_pack.filter({"Years" : years_fts})
DM_fts = {1: dm_fts, 2: dm_fts, 3: dm_fts, 4: dm_fts} # for now we set all levels to be the same
DM = {"ots" : dm_ots,
      "fts" : DM_fts}
f = os.path.join(current_file_directory, '../data/datamatrix/lever_paperpack.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)







