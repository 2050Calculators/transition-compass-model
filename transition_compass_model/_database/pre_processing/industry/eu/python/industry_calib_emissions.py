
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

# file
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/industry/eu/python/industry_calib_emissions.py"

# directories
current_file_directory = os.path.dirname(os.path.abspath(__file__))

#########################################
##### GET CLEAN DATAFRAME WITH DATA #####
#########################################

# get data
df = eurostat.get_data_df("env_ac_ainah_r2")
# filepath = os.path.join(current_file_directory, '../data/eurostat/env_ac_ainah_r2.csv')
# df.to_csv(filepath, index = False)
# df = pd.read_csv(filepath)

# get manufacturing and gases in tonnes
df = df.loc[df["nace_r2"] == "C",:]
df = df.loc[df["airpol"].isin(["CH4","CO2","N2O"]),:]
df = df.loc[df["unit"] == "T",:]

# fix country names
countries_codes = ["AT","BE","BG","HR","CY","CZ","DK",
                   "EE","FI","FR","DE","EL","HU","IE","IT",
                   "LV","LT","LU","MT","NL","PL","PT",
                   "RO","SK","SI","ES","SE","EU27_2020"]
df = df.loc[df["geo\\TIME_PERIOD"].isin(countries_codes),:]
countries = ['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czech Republic','Denmark',
             'Estonia','Finland','France','Germany','Greece','Hungary','Ireland','Italy',
             'Latvia','Lithuania','Luxembourg','Malta','Netherlands','Poland','Portugal',
             'Romania','Slovakia','Slovenia','Spain','Sweden',"EU27"]
for i in range(0, len(countries_codes)):
    df.loc[df["geo\\TIME_PERIOD"] == countries_codes[i],"geo\\TIME_PERIOD"] = countries[i]
len(df["geo\\TIME_PERIOD"].unique())

# clean df
df.drop(['freq', 'nace_r2'], axis=1, inplace=True)
df = pd.melt(df, id_vars = ["geo\\TIME_PERIOD","airpol","unit"], var_name='year')
df["year"] = [int(i) for i in df["year"]]
df = df.loc[df["year"] >= 2008,:] # most data is from 2008 onwards
df["value"] = df["value"]/1000000
df["variable"] = ["calib-emissions_" + gas + "[Mt]" for gas in df["airpol"]]
df.columns = ["Country","gas","unit","Years","value","variable"]
df = df.loc[:,["Country","Years","variable","value"]]
df = df.pivot(index=["Country","Years"], columns="variable", values='value').reset_index()

# make data matrix
dm = DataMatrix.create_from_df(df, 1)

# create united kingdom
idx = dm.idx
arr_temp = dm.array[idx["Germany"],...]
dm.add(arr_temp, "Country", "United Kinddom")
dm.sort("Country")

# save
f = os.path.join(current_file_directory, '../data/datamatrix/calibration_emissions.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

# df = dm.write_df()
# df = df.loc[df["Country"] == "EU27",:]
# df_temp = pd.melt(df, id_vars = ['Country','Years'], var_name='variable')
# df_temp = df_temp.loc[df_temp["Years"] == 2021,:]
# name = "temp.xlsx"
# df_temp.to_excel("~/Desktop/" + name)














