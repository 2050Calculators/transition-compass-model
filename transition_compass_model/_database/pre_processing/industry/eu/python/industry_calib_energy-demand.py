
# packages
from model.common.data_matrix_class import DataMatrix
import pandas as pd
import pickle
import os
import numpy as np
import warnings
import eurostat
import re
# from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat
warnings.simplefilter("ignore")
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

# directories
current_file_directory = os.getcwd()

###########################################################
############## CATEGORIES OF ENERGY CARRIERS ##############
###########################################################

enercarr_jrc = ['Ambient heat', 'Electricity', 'Hard coal and others', 'Fuel oil',
                'Derived gases','Coke', 'LPG', 'Refinery gas', 'Other liquids', 
                'Biomass and waste', 'Distributed steam', 'Diesel oil (without biofuels)',
                'Liquid biofuels', 'Natural gas', 'Biogas', 'Solar', 'Geothermal']
enercarr_calc_map = ["electricity", "electricity", "solid-ff-coal", "liquid-ff-oil",
                     "gas-ff-natural", "solid-ff-coal", "gas-ff-natural", "gas-ff-natural", "liquid-ff-oil", 
                     "Biomass and waste", "electricity", "liquid-ff-diesel", 
                     "liquid-bio", "gas-ff-natural", "gas-bio", "electricity", "electricity"]

# For splitting Biomass and waste between solid biomass and waste, I apply the
# following adjustments, taken from the global bioenergy statistics report 2023.
# https://www.worldbioenergy.org/uploads/231219%20GBS%20Report.pdf?utm_source=chatgpt.com
waste_municipal = 76.7
waste_industrial = 36.6
waste = waste_municipal + waste_industrial
solid_biomass = 471
tot = waste + solid_biomass
dict_adj_biomass_waste = {"solid-waste" : np.round(waste/tot,2), 
                          "solid-bio" : np.round(solid_biomass/tot,2)}

# conversion factors
ktoe_to_twh = 1000*11.63/1000000

#############

def get_data(country):
    
    filepath = os.path.join(current_file_directory, '../data/JRC-IDEES-2021/' + country + '/JRC-IDEES-2021_Industry_' + country + '.xlsx')
    df = pd.read_excel(filepath, "Ind_Summary")

    def my_search(search, x):
        if x is np.nan:
            return False
        else:
            return bool(re.search(search, x, re.IGNORECASE))

    # get data
    ls_temp = [my_search("Energy consumption", i) for i in df.iloc[:,0]]
    index_row_start = [i +2 for i, x in enumerate(ls_temp) if x][0]
    ls_temp = [my_search("by sector", i) for i in df.iloc[:,0]]
    index_row_end = [i for i, x in enumerate(ls_temp) if x][0]
    df = df.iloc[range(index_row_start, index_row_end),:]

    # keep only disaggregated ener carrier
    id_var = df.columns[0]
    df = df.loc[df[id_var].isin(enercarr_jrc),:]

    # melt
    df = pd.melt(df, id_vars = id_var, var_name='year')
    df.columns = ["energy_carrier","year","value"]
    df["unit"] = "ktoe"

    # apply calc energy carriers
    for i in range(0,len(enercarr_jrc)):
        df.loc[df["energy_carrier"] == enercarr_jrc[i],"energy_carrier"] = enercarr_calc_map[i]
    df = df.groupby(["energy_carrier","year","unit"], as_index=False)['value'].agg(sum)

    # fix biomass and waste
    for key in dict_adj_biomass_waste.keys():
        df_temp = df.loc[df["energy_carrier"] == "Biomass and waste",:]
        df_temp["value"] = df_temp["value"] * dict_adj_biomass_waste[key]
        df_temp["energy_carrier"] = key
        df = pd.concat([df, df_temp])
    df = df.loc[df["energy_carrier"] != "Biomass and waste",:]

    # convert from ktoe to twh
    df["value"] = df["value"]*ktoe_to_twh
    df["unit"] = "TWh"
    df_out = df.copy()
    
    # # get feedstock
    filepath = os.path.join(current_file_directory, '../data/JRC-IDEES-2021/' + country + '/JRC-IDEES-2021_Industry_' + country + '.xlsx')
    df = pd.read_excel(filepath, "Ind_Summary")
    ls_temp = [my_search("Non-energy use", i) for i in df.iloc[:,0]]
    index_row_start = [i +2 for i, x in enumerate(ls_temp) if x][0]
    ls_temp = [my_search("by sector", i) for i in df.iloc[:,0]]
    index_row_end = [i for i, x in enumerate(ls_temp) if x][1]
    df = df.iloc[range(index_row_start, index_row_end),:]
    id_var = df.columns[0]
    df = df.loc[~df[id_var].isin(["Liquids","Gas"]),:]
    df = pd.melt(df, id_vars = id_var, var_name='year')
    df.columns = ["energy_carrier","year","value"]
    df["unit"] = "ktoe"
    for i in range(0,len(enercarr_jrc)):
        df.loc[df["energy_carrier"] == enercarr_jrc[i],"energy_carrier"] = enercarr_calc_map[i]
    df["energy_carrier"].unique()
    df.loc[df["energy_carrier"] == 'Diesel oil',"energy_carrier"] = 'liquid-ff-diesel'
    df.loc[df['energy_carrier'] == 'Naphtha',"energy_carrier"] = 'liquid-ff-diesel'
    df.loc[df['energy_carrier'] == 'Solids',"energy_carrier"] = 'solid-ff-coal'
    for key in dict_adj_biomass_waste.keys():
        df_temp = df.loc[df["energy_carrier"] == "RES and wastes",:]
        df_temp["value"] = df_temp["value"] * dict_adj_biomass_waste[key]
        df_temp["energy_carrier"] = key
        df = pd.concat([df, df_temp])
    df = df.loc[df["energy_carrier"] != "RES and wastes",:]
    df = df.groupby(["energy_carrier","year","unit"], as_index=False)['value'].agg(sum)
    df["value"] = df["value"]*ktoe_to_twh
    df["unit"] = "TWh"
    
    # # put together
    df_out = pd.concat([df_out, df])
    df = df_out.copy()
    df = df.groupby(["energy_carrier","year","unit"], as_index=False)['value'].agg(sum)
    df["Country"] = country
    df["variable"] = ["energy-demand_" + carrier + "[TWh]" for carrier in df["energy_carrier"]]
    df = df.loc[:,["Country","year","variable","value"]]
    df.columns = ['Country', 'Years', 'variable', 'value']
    df = df.pivot(index=["Country","Years"], columns="variable", values='value').reset_index()
    
    # return
    return df

# get eu27
df = get_data("EU27")

# get other countries
countries_codes = ["AT","BE","BG","HR","CY","CZ","DK",
                   "EE","FI","FR","DE","EL","HU","IE","IT",
                   "LV","LT","LU","MT","NL","PL","PT",
                   "RO","SK","SI","ES","SE"]
for c in countries_codes:
    df = pd.concat([df, get_data(c)])

# change country names
countries = ['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czech Republic','Denmark',
             'Estonia','Finland','France','Germany','Greece','Hungary','Ireland','Italy',
             'Latvia','Lithuania','Luxembourg','Malta','Netherlands','Poland','Portugal',
             'Romania','Slovakia','Slovenia','Spain','Sweden']
for i in range(0,len(countries_codes)):
    df.loc[df["Country"] == countries_codes[i],"Country"] = countries[i]

# make data matrix
dm = DataMatrix.create_from_df(df, 1)

# # create united kingdom
# idx = dm.idx
# arr_temp = dm.array[idx["Germany"],...]
# dm.add(arr_temp, "Country", "United Kinddom")
# dm.sort("Country")

# add hydrogen
dm.add(0, "Categories1", "hydrogen", dummy=True)
dm.sort("Categories1")

# rename
dm.rename_col("energy-demand", "calib-energy-demand", "Variables")

# add missing years as nan
years = list(range(1990,2023+1)) + list(range(2025,2050+5,5))
missing = np.array(years)[[y not in dm.col_labels["Years"] for y in years]].tolist()
dm.add(np.nan, "Years", missing, dummy=True)
dm.sort("Years")

# save
f = os.path.join(current_file_directory, '../data/datamatrix/calibration_energy-demand.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

# df = dm.write_df()
# df = df.loc[df["Country"] == "EU27",:]
# df_temp = pd.melt(df, id_vars = ['Country','Years'], var_name='variable')
# df_temp = df_temp.loc[df_temp["Years"] == 2021,:]
# name = "temp.xlsx"
# df_temp.to_excel("~/Desktop/" + name)
















