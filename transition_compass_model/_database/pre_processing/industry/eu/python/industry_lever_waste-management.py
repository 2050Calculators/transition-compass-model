
# packages
from model.common.data_matrix_class import DataMatrix
from model.common.auxiliary_functions import linear_fitting
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


# Eurostat databases:
# env_waselv for vehicles
# env_waselee for larger appliances, pc and electronics
# env_waspac for packaging waste

##############################################################################
################################## VEHICLES ##################################
##############################################################################

###############################################
##### GET DATA ON END OF LIFE OF VEHICLES #####
###############################################

# get data
df = eurostat.get_data_df("env_waselv")
# filepath = os.path.join(current_file_directory, '../data/eurostat/env_waselv.csv')
# df.to_csv(filepath, index = False)
# df = pd.read_csv(filepath)

df_total = eurostat.get_data_df("env_waselvt")
# filepath = os.path.join(current_file_directory, '../data/eurostat/env_waselvt.csv')
# df_total.to_csv(filepath, index = False)
# df_total = pd.read_csv(filepath)

# get geo column
df.rename(columns={'geo\\TIME_PERIOD': 'geoscale'}, inplace=True)
df_total.rename(columns={'geo\\TIME_PERIOD': 'geoscale'}, inplace=True)

# filter for unit of measure: Tonne
df = df.loc[df['unit'] == 'T',:]
df_total = df_total.loc[df_total['unit'] == 'T',:]

# checks
A = df.copy()
A["combo"] = [i + " - " + j for i, j in zip(df["wst_oper"], df["waste"])]
list(A["combo"].unique())
df_temp = df.copy()
df_temp = df_temp.loc[df_temp["geoscale"] == "AT",:]
df_temp = df_temp.loc[:,['freq', 'wst_oper', 'waste', 'unit', 'geoscale',"2022"]]
df_temp = df_temp.loc[df_temp["wst_oper"] == "REU",:]
df_temp_tot = df_total.copy()
df_temp_tot = df_temp_tot.loc[df_temp_tot["geoscale"] == "AT",:]
df_temp_tot = df_temp_tot.loc[:,['freq', 'wst_oper', 'unit', 'geoscale',"2022"]]
df_temp_tot = df_temp_tot.loc[df_temp_tot["wst_oper"] == "RCY",:]
# RCY-ENV is total recycled
# ENV is always TOTAL
df_temp = df.copy()
df_temp = df_temp.loc[df_temp["geoscale"] == "AT",:]
df_temp = df_temp.loc[:,['freq', 'wst_oper', 'waste', 'unit', 'geoscale',"2022"]]
df_temp = df_temp.loc[df_temp["wst_oper"] == "RCV_E",:]
# W1910 = W191001 + W191002 + W1910A + W1910B

# in general for us:
# total = littered + exported + collected + uncollected
# collected = recycling + energy recovery + reuse + landfill + incineration

# in general in eurostat
# Waste generation: The quantity of waste, whereby ‘waste’ means any substance or object which the holder discards or intends or is required to discard.
# Waste management: Waste management refers to the collection, transport, recovery (including sorting), and disposal of waste.
# Treatment: Treatment means recovery or disposal operations, including preparation prior to recovery or disposal.
# Recovery: Recovery means any operation whose main result is that waste serves a useful purpose. For example, by replacing other materials that would have been used for a particular function.
# Disposal: Disposal means any operation that is not recovery (personal note: I guess this is either incineration or landfill)

# specifically for env_waselv
# in "wst_oper":  disposed (DSP), generated (GEN), recovered (RCV), energy recovery (RCV-E), recycled (RCY), 
# reused (REU)
# in "waste": 
# dismantling and de-pollution (DMDP), 
# exported (EXP), 
# liquids (LIQ), 
# end of life vehicles (ELV),
# End-of-life vehicles: tyres (W160103), 
# End-of-life vehicles: oil filters (W160107),
# End-of-life vehicles: other materials arising from depollution (excluding fuel) (W1601A),
# End-of-life vehicles: metal components (LoW: 160117+160118) (W1601B), 
# End-of-life vehicles: large plastic parts (W160119), 
# End-of-life vehicles: glass (W160120),
# Batteries and accumulators (W1606), 
# Catalysts (W1608),
# Total shredding (W1910),
# Ferrous scrap (steel) from shredding (W191001),
# Non-ferrous materials (aluminium, copper, zinc, lead, etc.) from shredding (W191002),
# Shredder Light Fraction (SLF) (W1910A),
# Other materials arising from shredding (W1910B)

# so, formulas:
# littered: 0
# exported: among “waste”, EXP
# collected: recycling + energy recovery + reuse + landfill + incineration
# uncollected: collected / 0.8 * 0.2
# recycling: among “wst_oper”, RCY
# energy recovery: among “wst_oper”, RCV-E
# reuse: among “wst_oper”, REU
# landfill: among “waste” and “wst_oper”, DSP
# incineration: 0

# mapping
dict_mapping = {"recycling": ["RCY - DMDP", "RCY - W1910"],
                "energy-recovery": ['RCV_E - DMDP', 'RCV_E - W1910'],
                "reuse": ["REU - DMDP", "REU - W1910"],
                "landfill": ['DSP - DMDP', 'DSP - W1910'],
                "export" : ["GEN - EXP"]
                }

# make long format
indexes = ['freq', 'wst_oper', 'waste', 'unit', 'geoscale']
df = pd.melt(df, id_vars = indexes, var_name='year')

# create column with combos
df["combo"] = [i + " - " + j for i, j in zip(df["wst_oper"], df["waste"])]

# aggregate
indexes = ['freq', 'wst_oper', 'unit', 'geoscale','year']
key = "energy-recovery"
def my_aggregation(key, df, dict_mapping, indexes):
    df_temp = df.loc[df["combo"].isin(dict_mapping[key]),:]
    df_temp = df_temp.groupby(indexes, as_index=False)['value'].agg(sum)
    df_temp["variable"] = key
    return df_temp
df_elv = pd.concat([my_aggregation(key, df, dict_mapping, indexes) for key in dict_mapping.keys()])

# select countries
df_elv = df_elv.loc[:,['geoscale', 'year', 'variable', 'unit', 'value']]
country_list = {'AT' : "Austria", 'BE' : "Belgium", 'BG': "Bulgaria", 'HR' : "Croatia", 'CY' : "Cyprus", 
                'CZ' : "Czech Republic", 'DK' : "Denmark", 'EE' : "Estonia", 'EU27_2020': "EU27", 'FI' : "Finland",
                'FR' : "France", 'DE' : "Germany", 'EL' : "Greece", 'HU' : "Hungary", 'IE' : "Ireland",
                'IT' : "Italy", 'LV' : "Latvia", 'LT' : "Lithuania", 'LU' : "Luxembourg", 
                'MT' : "Malta", 'NL' : "Netherlands",
                'PL' : "Poland", 'PT' : "Portugal", 'RO' : "Romania", 'SK' : "Slovakia", 'SI' : "Slovenia", 
                'ES' : "Spain", 'SE' : "Sweden"}
for c in country_list.keys():
    df_elv.loc[df_elv["geoscale"] == c,"geoscale"] = country_list[c]
drops = ['IS', 'LI', 'NO']
df_elv = df_elv.loc[~df_elv["geoscale"].isin(drops),:]
len(df_elv["geoscale"].unique()) # note that we are missing UK, to do at the end

# make EU27
df_temp = df_elv.loc[df_elv["geoscale"] == "EU27",:] # for the moment I will keep this export, but it has only 1 value
df_temp = df.loc[df["geoscale"] == "EU27_2020",:] # yes there are no other export values
indexes = ['year', 'variable', 'unit']
countries = ['Austria', 'Belgium', 'Bulgaria', 'Cyprus', 'Czech Republic',
             'Germany', 'Denmark', 'Estonia', 'Greece', 'Spain', 'Finland',
             'France', 'Croatia', 'Hungary', 'Ireland', 'Italy', 'Lithuania',
             'Luxembourg', 'Latvia', 'Malta', 'Netherlands', 'Poland',
             'Portugal', 'Romania', 'Sweden', 'Slovenia', 'Slovakia']
df_temp = df_elv.loc[df_elv["geoscale"].isin(countries),:]
df_temp = df_temp.groupby(indexes, as_index=False)['value'].agg(sum)
df_temp["geoscale"] = "EU27"
df_temp = df_temp.loc[df_temp["variable"]!="export",:]
df_elv = pd.concat([df_elv,df_temp])
indexes = ['geoscale', 'variable', 'year', 'unit']
df_elv.sort_values(indexes, inplace=True)

# fix variable name with unit
df_elv["variable"] = [i + "[t]" for i in df_elv["variable"]]
df_elv.drop(columns="unit",inplace=True)

# checks
df_temp = df_elv.pivot(index=["geoscale","year"], columns="variable", values='value').reset_index()

# clean
del A, c, countries, country_list, df, df_temp, df_temp_tot, df_total, dict_mapping, drops,\
    indexes, key

##################################
##### CONVERT TO DATA MATRIX #####
##################################

# rename
df_elv.rename(columns={"geoscale":"Country","year":"Years"},inplace=True)

# put nan where is 0 (this is done to then generate values and replacing the zeroes)
df_elv.loc[df_elv["value"] == 0,"value"] = np.nan

# make dm
df_temp = df_elv.copy()
df_temp = df_temp.pivot(index=["Country","Years"], columns="variable", values='value').reset_index()
dm_elv = DataMatrix.create_from_df(df_temp, 0)

# # plot
# dm_elv.filter({"Country" : ["EU27"], "Variables" : ["reuse"]}).datamatrix_plot()
# dm_elv.filter({"Country" : ["EU27"]}).datamatrix_plot()
# landfill: probably consider until 2010 (when it's growing) for backward interpolation
# df_temp = dm_elv.write_df()

# clean
del df_temp, df_elv

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

# fill nas (including 2023)
years_fitting = dm_elv.col_labels["Years"]
dm_elv = linear_fitting(dm_elv, years_fitting, min_t0=0, min_tb=0)
dm_elv.array = np.round(dm_elv.array,0)

# there are still some countries that for one variable have nan at all times, for these we put 0
dm_elv.array[np.isnan(dm_elv.array)] = 0

# # plot
# dm_elv.filter({"Country" : ["EU27"]}).datamatrix_plot() 
# export is constant as we only have 1 value
# df_temp = dm_elv.write_df()

# # fix jumps
# dm_elv = fix_jumps_in_dm(dm_elv)

# # plot
# dm_elv.filter({"Country" : ["EU27"]}).datamatrix_plot() 

# add missing years ots
years_missing = list(set(years_ots) - set(dm_elv.col_labels['Years']))
dm_elv.add(np.nan, col_label=years_missing, dummy=True, dim='Years')
dm_elv.sort('Years')

# fill in missing years ots with linear fitting
# I am simply putting the value of 2005 for all past years, as this works best for shares later
# (with trends, things can get mixed up, like more reuse than recycling, or zero landfilling)
# dm_temp1 = linear_fitting(dm_elv.filter({"Variables" : ["landfill","recycling"]}), 
#                           years_ots = years_missing, min_t0=0, min_tb=0, based_on = list(range(2005,2010+1,1)))
dm_elv = linear_fitting(dm_elv.filter({"Variables" : ['landfill','recycling', 'energy-recovery', 'export', 'reuse']}), 
                        years_ots = years_missing, min_t0=0, min_tb=0, based_on = [2005])
# dm_elv.append(dm_temp1, "Variables")
dm_elv.array = np.round(dm_elv.array,0)
dm_elv.sort("Variables")

# # plot
# dm_elv.filter({"Country" : ["EU27"], "Variables" : ["reuse"]}).datamatrix_plot()
# dm_elv.filter({"Country" : ["EU27"]}).datamatrix_plot()
# df_temp = dm_elv.write_df()

# clean
del years_fitting, years_missing

####################
##### MAKE FTS #####
####################

# make function to fill in missing years fts for EU27 with linear fitting
def make_fts(dm, variable, year_start, year_end, country = "EU27", dim = "Variables", 
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
dm_elv.add(np.nan, col_label=years_fts, dummy=True, dim='Years')

# set default time window for linear trend
# assumption: best is taking longer trend possible to make predictions to 2050 (even if earlier data is generated)
baseyear_start = 1990
baseyear_end = 2019

# compute fts
dm_elv = make_fts(dm_elv, "energy-recovery", baseyear_start, baseyear_end)
dm_elv = make_fts(dm_elv, "export", baseyear_start, baseyear_end)
dm_elv = make_fts(dm_elv, "landfill", 2010, baseyear_end) # we keep the decreasing trend since 2010
dm_elv = make_fts(dm_elv, "recycling", baseyear_start, baseyear_end)
dm_elv = make_fts(dm_elv, "reuse", baseyear_start, baseyear_end)
# variable = "reuse"
# (make_fts(dm_elv, variable, baseyear_start, baseyear_end).
#   datamatrix_plot(selected_cols={"Country" : ["EU27"],
#                                 "Variables" : [variable]}))

# # check
# df_temp = dm_elv.write_df()

################################
##### MAKE FINAL VARIABLES #####
################################

# make dm for collected waste
dm_elv.add(0, col_label="incineration", dummy=True, dim='Variables', unit="t") # incineration = 0 as it's forbidden in EU to incinerate cars
dm_elv_col = dm_elv.filter({"Variables" : ['energy-recovery', 'incineration', 'landfill', 'recycling', 'reuse']})
dm_elv_col.sort("Variables")
dm_elv_tot = dm_elv_col.groupby(group_cols={"collected" : ['energy-recovery', 'incineration', 
                                                           'landfill', 'recycling', 'reuse']}, 
                                dim="Variables", aggregation = "sum", regex=False, inplace=False)

# re-put nan in years fts for all countries but EU27
idx = dm_elv_tot.idx
countries = dm_elv_tot.col_labels["Country"]
countries = list(np.array(countries)[[i != "EU27" for i in countries]])
for c in countries:
    for y in years_fts:
            dm_elv_tot.array[idx[c],idx[y],:] = np.nan

# compute the shares rounded at 2 decimals
dm_elv_col.array = dm_elv_col.array / dm_elv_tot.array
def adjust_shares(shares):
    # Round shares to two decimal places
    rounded_shares = [round(share, 2) for share in shares]
    total = sum(rounded_shares)
    
    # Calculate the discrepancy
    discrepancy = round(1.0 - total, 2)
    
    # Adjust the share with the largest absolute value
    if discrepancy != 0:
        # Find the index of the share to adjust
        idx_to_adjust = max(range(len(rounded_shares)), key=lambda i: rounded_shares[i])
        rounded_shares[idx_to_adjust] += discrepancy
        # Ensure the adjustment doesn't break the two-decimal constraint
        rounded_shares[idx_to_adjust] = round(rounded_shares[idx_to_adjust], 2)
    
    return rounded_shares
idx = dm_elv_col.idx
for c in dm_elv_col.col_labels["Country"]:
    for y in years_all:
            dm_elv_col.array[idx[c],idx[y],:] = adjust_shares(dm_elv_col.array[idx[c],idx[y],:])
for v in dm_elv_col.col_labels["Variables"]: 
    dm_elv_col.units[v] = "%"

# # checks
# dm_temp = dm_elv_col.groupby(group_cols={"total" : dm_elv_col.col_labels["Variables"]}, 
#                               dim="Variables", aggregation = "sum", regex=False, inplace=False)
# for c in countries:
#     for y in years_fts:
#             dm_temp.array[idx[c],idx[y],:] = np.nan
# dm_temp.array
# df_temp = dm_temp.write_df()
# df_temp = dm_elv_col.write_df()
# df_temp = dm_elv_tot.write_df()
# dm_elv_col.filter({"Country" : ["EU27"]}).datamatrix_plot()

# make dm for total waste
dm_elv_tot.append(dm_elv.filter({"Variables" : ["export"]}), "Variables")
dm_elv_tot.rename_col("collected","waste-collected","Variables")
dm_temp = dm_elv_tot.groupby({"waste-uncollected" : ["waste-collected","export"]}, "Variables", inplace=False)
dm_temp.array = dm_temp.array * 0.36/0.64
dm_elv_tot.append(dm_temp,"Variables")
dm_elv_tot.add(0, col_label="littered", dummy=True, dim='Variables', unit="t")
dm_elv_tot.sort("Variables")
idx = dm_elv_tot.idx
countries = dm_elv_tot.col_labels["Country"]
countries = list(np.array(countries)[[i != "EU27" for i in countries]])
for c in countries:
    for y in years_fts:
            dm_elv_tot.array[idx[c],idx[y],idx["littered"]] = np.nan
            
# dm_elv_tot.filter({"Country" : ["EU27"]}).datamatrix_plot()

# make shares in dm
# compute the shares rounded at 2 decimals
dm_temp = dm_elv_tot.groupby(group_cols={"total" : ['waste-collected', 'export', 'waste-uncollected', 'littered']}, 
                             dim="Variables", aggregation = "sum", regex=False, inplace=False)
idx = dm_temp.idx
for c in countries:
    for y in years_fts:
            dm_temp.array[idx[c],idx[y],idx["total"]] = np.nan
dm_elv_tot.array = dm_elv_tot.array / dm_temp.array
for c in dm_elv_tot.col_labels["Country"]:
    for y in years_all:
            dm_elv_tot.array[idx[c],idx[y],:] = adjust_shares(dm_elv_tot.array[idx[c],idx[y],:])
for v in dm_elv_tot.col_labels["Variables"]: 
    dm_elv_tot.units[v] = "%"

# df_temp = dm_elv_tot.write_df()
# df_temp["total"] = df_temp["waste-collected[%]"] + df_temp["waste-uncollected[%]"] + df_temp["export[%]"] + df_temp["littered[%]"]
# dm_elv_tot.filter({"Country" : ["EU27"]}).datamatrix_plot()

# create UK
# TODO: for the moment i make UK as Germany, to be done later to get data for UK
idx = dm_elv_col.idx
arr_temp = dm_elv_col.array[idx["Germany"],:,:]
dm_elv_col.add(arr_temp[np.newaxis,...], dim = "Country", col_label = "United Kingdom")
idx = dm_elv_tot.idx
arr_temp = dm_elv_tot.array[idx["Germany"],:,:]
dm_elv_tot.add(arr_temp[np.newaxis,...], dim = "Country", col_label = "United Kingdom")

# create industry products for vehicles (use same rates for all products)
# vehicles = ['HDV', 'LDV', 'bus']
# engines = ['BEV', 'FCEV', 'ICE', 'ICE-diesel', 'ICE-gas', 'ICE-gasoline', 'PHEV-diesel', 'PHEV-gasoline']
# dict_out = {}
# for v in vehicles:
#     dict_out[v] = [v + "_" + e for e in engines]
# products = dict_out["HDV"].copy()
# for v in ['LDV', 'bus']:
#     products = products + dict_out[v].copy()
products = ["vehicles"]

dm_temp = dm_elv_tot.copy()
variabs = dm_temp.col_labels["Variables"]
for v in variabs:
    dm_temp.rename_col(v, products[0] + "_" + v, "Variables")
dm_temp.deepen()
dm_elv_tot_new = dm_temp.copy()
for i in range(1, len(products)):
    dm_temp = dm_elv_tot.copy()
    variabs = dm_temp.col_labels["Variables"]
    for v in variabs:
        dm_temp.rename_col(v, products[i] + "_" + v, "Variables")
    dm_temp.deepen()
    dm_elv_tot_new.append(dm_temp, "Variables")
dm_elv_tot = dm_elv_tot_new.copy()

dm_temp = dm_elv_col.copy()
variabs = dm_temp.col_labels["Variables"]
for v in variabs:
    dm_temp.rename_col(v, products[0] + "_" + v, "Variables")
dm_temp.deepen()
dm_elv_col_new = dm_temp.copy()
for i in range(1, len(products)):
    dm_temp = dm_elv_col.copy()
    variabs = dm_temp.col_labels["Variables"]
    for v in variabs:
        dm_temp.rename_col(v, products[i] + "_" + v, "Variables")
    dm_temp.deepen()
    dm_elv_col_new.append(dm_temp, "Variables")
dm_elv_col = dm_elv_col_new.copy()

# clean
del arr_temp, baseyear, baseyear_end, baseyear_start, c, countries, dm_elv,\
    dm_temp, idx, lastyear, startyear, step_fts, v, years_all, \
    years_fts, years_ots, years_setting, y, dm_elv_col_new, dm_elv_tot_new, \
    products

################
##### SAVE #####
################

# store
DM_wst_mgt = {"elv-total" : dm_elv_tot,
              "elv-col" : dm_elv_col}

# clean
del dm_elv_tot, dm_elv_col

##############################################################################
############################## TRUCKS AND BUSES ##############################
##############################################################################

# DM_wst_mgt["elv-total"].filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()
# DM_wst_mgt["elv-col"].filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

# For buses and trucks:
# Assumption: same of vehicles

# # Source: https://horizoneuropencpportal.eu/sites/default/files/2023-09/acea-position-paper-end-of-life-vehicles-directive-trucks-buses-2020.pdf
# Page 3:
# Industry believes that the re-use and recycling of second raw materials is important as well. In fact,
# this is already part of the business models of many vehicle manufacturers today. Throughout the 19
# years that HDVs have been outside the scope of the ELV Directive, the vehicle recycling industry
# has handled, treated and de-polluted trucks and buses in a way similar to passenger cars and thus
# basically already applies existing environmental legislation to HDVs.

# Source: https://cms.uitp.org/wp/wp-content/uploads/2021/05/Knowledge-Brief-Second-hand-bus_final.pdf
# Page 2: Only 10–20% of the dismissed buses find a
# second life. This is simply because most buses are entirely used up to their technical life duration (15-20 years)
# and then scrapped. In cases where buses are taken out
# of service at early ages, buyers could be found, mostly in
# distant or significantly poorer countries.

##############################################################################
############################ TRAINS AND METROTRAM ############################
##############################################################################

# Assumption: most of the trains that are collected go to recycling
# Source: https://www.researchgate.net/publication/267762517_Recycling_guidelines_of_the_rolling_stock/link/5784f3d908ae36ad40a4b0e6/download?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19
# Page 8:
# The subway rolling stock is characterized by an almost 95% recovery rate [10]. 85% of the mass is subject to recycling and another 10% is combusted with energy recovery.
# Eurostar - has also implemented assumptions of environment-friendly traveling. One of the elements of the strategy was to achieve an 80% recycling rate of the end-of- life rolling stock by 2012 with the assumption that no waste will be forwarded for landfilling

dm_rail_tot = DM_wst_mgt["elv-total"].copy()
dm_rail_tot[:,:,:,"export"] = np.nanmean(dm_rail_tot[:,:,:,"export"]) # assuming similar export than vehicles
dm_rail_tot[:,:,:,"waste-collected"] = 1-dm_rail_tot[:,:,:,"export"] # assuming all rest is collected
dm_rail_tot[:,:,:,"waste-uncollected"] = 0
dm_rail_tot[:,:,:,"littered"] = 0
dm_rail_tot.rename_col("vehicles","trains","Variables")

dm_rail_col = DM_wst_mgt["elv-col"].copy()
dm_rail_col["reuse"] = 0 # assuming trains are used until the end of their lifetime
dm_rail_col["incineration"] = 0 # assuming trains are not incinerated
dm_rail_col["landfill"] = 0.05 # assuming 95% recovery rate. so 5% landfilled
dm_rail_col["recycling"] = 0.85
dm_rail_col["energy-recovery"] = 0.1
dm_rail_col.rename_col("vehicles","trains","Variables")

DM_wst_mgt["trains-total"] = dm_rail_tot
DM_wst_mgt["trains-col"] = dm_rail_col

##############################################################################
################################### PLANES ###################################
##############################################################################

# Source: https://www.easa.europa.eu/en/document-library/research-reports/study-assessment-environmental-sustainability-status-aviation
# Page 106 table 23: reuse 30, recycling 60, energy recovery 8, landfill 2
# Page 91: Litter likely to be produced, but unlikely that is generate beyond the exterior fencing
# I will assume litter, uncollected and exported to zero.

dm_planes_tot = DM_wst_mgt["elv-total"].copy()
dm_planes_tot[:,:,:,"waste-collected"] = 1
dm_planes_tot[:,:,:,"export"] = 0
dm_planes_tot[:,:,:,"waste-uncollected"] = 0
dm_planes_tot[:,:,:,"littered"] = 0
dm_planes_tot.rename_col("vehicles","planes","Variables")

dm_planes_col = DM_wst_mgt["elv-col"].copy()
dm_planes_col["reuse"] = 0.3
dm_planes_col["incineration"] = 0
dm_planes_col["landfill"] = 0.02
dm_planes_col["recycling"] = 0.6
dm_planes_col["energy-recovery"] = 0.08
dm_planes_col.rename_col("vehicles","planes","Variables")

DM_wst_mgt["planes-total"] = dm_planes_tot
DM_wst_mgt["planes-col"] = dm_planes_col

#############################################################################
################################### SHIPS ###################################
#############################################################################

# EU study: https://environment.ec.europa.eu/document/download/f717f65b-7293-43eb-9b52-2890277bc6d8_en?filename=Support%20study%20to%20the%20evaluation%20.pdf
# table 3.4: in 2018 (before directive) around 25 recycling facilities, today around 48 facilities

# source: https://shipbreakingplatform.org/platform-publishes-list-2024/
# following the number in the excel for 2024, of the EU owned ships, around 30% were
# dismantled in EU facilities, while around 70% were exported to third countries (Turkey, South East Easia, etc)
# assuming that before 2018 they were half

# this is beaching vs scrapping around the world, mostly beaching: https://www.offthebeach.org/
# we will assume that there is no "beaching" in the EU, so it's mostly scrapping, and probably we can use similar percentages
# than planes (most of it is recycled-reused)

dm_ship_tot = DM_wst_mgt["elv-total"].copy()

years_before_regulation = list(range(1990,2018+1))
years_after_regulation = list(range(2019,2023+1)) + list(range(2025,2050+5,5))

for y in years_before_regulation:
    dm_ship_tot[:,y,:,"waste-collected"] = 0.15
    dm_ship_tot[:,y,:,"export"] = 0.85
    dm_ship_tot[:,y,:,"waste-uncollected"] = 0
    dm_ship_tot[:,y,:,"littered"] = 0

for y in years_after_regulation:
    dm_ship_tot[:,y,:,"waste-collected"] = 0.3
    dm_ship_tot[:,y,:,"export"] = 0.7
    dm_ship_tot[:,y,:,"waste-uncollected"] = 0
    dm_ship_tot[:,y,:,"littered"] = 0
dm_ship_tot.rename_col("vehicles","ships","Variables")

dm_ship_col = DM_wst_mgt["elv-col"].copy()
dm_ship_col["reuse"] = 0.3
dm_ship_col["incineration"] = 0
dm_ship_col["landfill"] = 0.02
dm_ship_col["recycling"] = 0.6
dm_ship_col["energy-recovery"] = 0.08
dm_ship_col.rename_col("vehicles","ships","Variables")

DM_wst_mgt["ship-total"] = dm_ship_tot
DM_wst_mgt["ship-col"] = dm_ship_col

# tomorrow: do end of life of buildings and rail - road- trolly cables, and see what to do for all fts (probably keep the levels
# all the same for now)

###############################################################################
################################## BUILDINGS ##################################
###############################################################################

# get data
df = eurostat.get_data_df("env_wastrt")

# get geo column
df.rename(columns={'geo\\TIME_PERIOD': 'geoscale'}, inplace=True)

# filter for unit of measure: Tonne
df = df.loc[df['unit'] == 'T',:]

# get only construction waste, both hazardous and non hazardous
df = df.loc[df["waste"] == "W121",:]
df = df.loc[df["hazard"] == "HAZ_NHAZ",:]
df.drop(columns=["hazard"],inplace=True)

# checks
A = df.copy()
A["combo"] = [i + " - " + j for i, j in zip(df["wst_oper"], df["waste"])]
list(A["combo"].unique())


# in general for us:
# total = littered + exported + collected + uncollected
# collected = recycling + energy recovery + reuse + landfill + incineration

# in general in eurostat
# Waste generation: The quantity of waste, whereby ‘waste’ means any substance or object which the holder discards or intends or is required to discard.
# Waste management: Waste management refers to the collection, transport, recovery (including sorting), and disposal of waste.
# Treatment: Treatment means recovery or disposal operations, including preparation prior to recovery or disposal.
# Recovery: Recovery means any operation whose main result is that waste serves a useful purpose. For example, by replacing other materials that would have been used for a particular function.
# Disposal: Disposal means any operation that is not recovery (personal note: I guess this is either incineration or landfill)

# specifically for env_wastrt
# 'DSP_I - W121': incineration (D10)
# 'DSP_L - W121': landfill (D1, D5, D12)
# 'DSP_L_OTH - W121': landfill and other (D1-D7, D12)
# 'DSP_OTH - W121': other (D2-D4, D6-D7)
# 'RCV_B - W121': backfilling
# 'RCV_E - W121': energy recovery (R1)
# 'RCV_R - W121': recycling
# 'RCV_R_B - W121': recycling and backfilling (R2-R11)
# 'TRT - W121': waste treatment

# so, formulas:
# littered: 0
# exported: 0
# collected: recycling + energy recovery + reuse (backfilling) + landfill + incineration
# uncollected: 0
# recycling: RCV_R - W121
# energy recovery: RCV_E - W121
# reuse: RCV_B - W121
# landfill: DSP_L - W121
# incineration: DSP_I - W121

# mapping
dict_mapping = {"recycling": ["RCV_R - W121"],
                "energy-recovery": ['RCV_E - W121'],
                "reuse": ["RCV_B - W121"],
                "landfill": ['DSP_L - W121'],
                "incineration" : ["DSP_I - W121"]
                }

# make long format
indexes = ['freq', 'wst_oper', 'waste', 'unit', 'geoscale']
df = pd.melt(df, id_vars = indexes, var_name='year')

# create column with combos
df["combo"] = [i + " - " + j for i, j in zip(df["wst_oper"], df["waste"])]

# aggregate
indexes = ['freq', 'wst_oper', 'unit', 'geoscale','year']
key = "energy-recovery"
def my_aggregation(key, df, dict_mapping, indexes):
    df_temp = df.loc[df["combo"].isin(dict_mapping[key]),:]
    df_temp = df_temp.groupby(indexes, as_index=False)['value'].agg(sum)
    df_temp["variable"] = key
    return df_temp
df_bld = pd.concat([my_aggregation(key, df, dict_mapping, indexes) for key in dict_mapping.keys()])

# select countries
df_bld = df_bld.loc[:,['geoscale', 'year', 'variable', 'unit', 'value']]
country_list = {'AT' : "Austria", 'BE' : "Belgium", 'BG': "Bulgaria", 'HR' : "Croatia", 'CY' : "Cyprus", 
                'CZ' : "Czech Republic", 'DK' : "Denmark", 'EE' : "Estonia", 'EU27_2020': "EU27", 'FI' : "Finland",
                'FR' : "France", 'DE' : "Germany", 'EL' : "Greece", 'HU' : "Hungary", 'IE' : "Ireland",
                'IT' : "Italy", 'LV' : "Latvia", 'LT' : "Lithuania", 'LU' : "Luxembourg", 
                'MT' : "Malta", 'NL' : "Netherlands",
                'PL' : "Poland", 'PT' : "Portugal", 'RO' : "Romania", 'SK' : "Slovakia", 'SI' : "Slovenia", 
                'ES' : "Spain", 'SE' : "Sweden", 'UK' : 'United Kingdom'}
for c in country_list.keys():
    df_bld.loc[df_bld["geoscale"] == c,"geoscale"] = country_list[c]
drops = ['AL','BA','IS', 'ME', 'EU28','MK', 'LI', 'NO', 'RS', 'TR', 'XK']
df_bld = df_bld.loc[~df_bld["geoscale"].isin(drops),:]
len(df_bld["geoscale"].unique())

# fix variable name with unit
df_bld["variable"] = [i + "[t]" for i in df_bld["variable"]]
df_bld.drop(columns="unit",inplace=True)

# checks
df_temp = df_bld.pivot(index=["geoscale","year"], columns="variable", values='value').reset_index()

# drop before 2008 (all zero)
df_bld = df_bld.loc[df_bld["year"] > "2008",:]

# clean
del A, c, country_list, df, df_temp, dict_mapping, drops,\
    indexes, key
    
##################################
##### CONVERT TO DATA MATRIX #####
##################################

# rename
df_bld.rename(columns={"geoscale":"Country","year":"Years"},inplace=True)

# # put nan where is 0 (this is done to then generate values and replacing the zeroes)
# df_bld.loc[df_bld["value"] == 0,"value"] = np.nan

# make dm
df_temp = df_bld.copy()
df_temp = df_temp.pivot(index=["Country","Years"], columns="variable", values='value').reset_index()
dm_bld = DataMatrix.create_from_df(df_temp, 0)

# # plot
# dm_bld.filter({"Country" : ["EU27"], "Variables" : ["recycling"]}).datamatrix_plot()
# dm_bld.filter({"Country" : ["EU27"]}).datamatrix_plot()
# landfill: probably consider until 2010 (when it's growing) for backward interpolation
# df_temp = dm_bld.write_df()

# clean
del df_temp, df_bld

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

# fill nas (including 2023)
years_missing = np.array(years_ots)[[y not in dm_bld.col_labels["Years"] for y in years_ots]].tolist()
for y in years_missing:    
    dm_bld.add(np.nan, col_label=[y], dummy=True, dim='Years')
dm_bld.sort("Years")
years_fitting = dm_bld.col_labels["Years"]
dm_bld = linear_fitting(dm_bld, years_fitting, min_t0=0, min_tb=0)
dm_bld.array = np.round(dm_bld.array,0)

# # plot
# dm_bld.filter({"Country" : ["EU27"], "Variables" : ["reuse"]}).datamatrix_plot()
# dm_bld.filter({"Country" : ["EU27"]}).datamatrix_plot()
# df_temp = dm_bld.write_df()

# clean
del years_fitting, years_missing

####################
##### MAKE FTS #####
####################

# make function to fill in missing years fts for EU27 with linear fitting
def make_fts(dm, variable, year_start, year_end, country = "EU27", dim = "Variables", 
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
dm_bld.add(np.nan, col_label=years_fts, dummy=True, dim='Years')

# set default time window for linear trend
baseyear_start = 2010
baseyear_end = 2023

# compute fts
dm_bld = make_fts(dm_bld, "energy-recovery", baseyear_start, baseyear_end)
dm_bld = make_fts(dm_bld, "landfill", baseyear_start, baseyear_end)
dm_bld = make_fts(dm_bld, "recycling", baseyear_start, baseyear_end)
dm_bld = make_fts(dm_bld, "reuse", baseyear_start, baseyear_end)
dm_bld = make_fts(dm_bld, "incineration", baseyear_start, baseyear_end)

# # check
# df_temp = dm_bld.write_df()
# dm_bld.filter({"Country" : ["EU27"]}).datamatrix_plot()

################################
##### MAKE FINAL VARIABLES #####
################################

# make dm for collected waste
dm_bld_col = dm_bld.copy()
dm_bld_col.sort("Variables")
dm_bld_tot = dm_bld_col.groupby(group_cols={"collected" : ['energy-recovery', 'incineration', 
                                                           'landfill', 'recycling', 'reuse']}, 
                                dim="Variables", aggregation = "sum", regex=False, inplace=False)

# re-put nan in years fts for all countries but EU27
idx = dm_bld_tot.idx
countries = dm_bld_tot.col_labels["Country"]
countries = list(np.array(countries)[[i != "EU27" for i in countries]])
for c in countries:
    for y in years_fts:
            dm_bld_tot.array[idx[c],idx[y],:] = np.nan

# compute the shares rounded at 2 decimals
dm_bld_col.array = dm_bld_col.array / dm_bld_tot.array
def adjust_shares(shares):
    # Round shares to two decimal places
    rounded_shares = [round(share, 2) for share in shares]
    total = sum(rounded_shares)
    
    # Calculate the discrepancy
    discrepancy = round(1.0 - total, 2)
    
    # Adjust the share with the largest absolute value
    if discrepancy != 0:
        # Find the index of the share to adjust
        idx_to_adjust = max(range(len(rounded_shares)), key=lambda i: rounded_shares[i])
        rounded_shares[idx_to_adjust] += discrepancy
        # Ensure the adjustment doesn't break the two-decimal constraint
        rounded_shares[idx_to_adjust] = round(rounded_shares[idx_to_adjust], 2)
    
    return rounded_shares
idx = dm_bld_col.idx
for c in dm_bld_col.col_labels["Country"]:
    for y in years_all:
            dm_bld_col.array[idx[c],idx[y],:] = adjust_shares(dm_bld_col.array[idx[c],idx[y],:])
for v in dm_bld_col.col_labels["Variables"]: 
    dm_bld_col.units[v] = "%"

# # checks
# dm_temp = dm_bld_col.groupby(group_cols={"total" : dm_bld_col.col_labels["Variables"]}, 
#                               dim="Variables", aggregation = "sum", regex=False, inplace=False)
# for c in countries:
#     for y in years_fts:
#             dm_temp.array[idx[c],idx[y],:] = np.nan
# dm_temp.array
# df_temp = dm_temp.write_df()
# df_temp = dm_bld_col.write_df()
# df_temp = dm_bld_tot.write_df()
# dm_bld_col.filter({"Country" : ["EU27"]}).datamatrix_plot()

# make dm for total waste
dm_bld_tot.rename_col("collected","waste-collected","Variables")
dm_bld_tot.add(0, col_label="waste-uncollected", dummy=True, dim='Variables', unit="t")
dm_bld_tot.add(0, col_label="littered", dummy=True, dim='Variables', unit="t")
dm_bld_tot.add(0, col_label="export", dummy=True, dim='Variables', unit="t")
dm_bld_tot.sort("Variables")
idx = dm_bld_tot.idx
countries = dm_bld_tot.col_labels["Country"]
countries = list(np.array(countries)[[i != "EU27" for i in countries]])
for c in countries:
    for y in years_fts:
            dm_bld_tot.array[idx[c],idx[y],idx["littered"]] = np.nan
            dm_bld_tot.array[idx[c],idx[y],idx["export"]] = np.nan
            dm_bld_tot.array[idx[c],idx[y],idx["waste-uncollected"]] = np.nan

# dm_bld_tot.filter({"Country" : ["EU27"]}).datamatrix_plot()

# make shares in dm
# compute the shares rounded at 2 decimals
dm_temp = dm_bld_tot.groupby(group_cols={"total" : ['waste-collected', 'export', 'waste-uncollected', 'littered']}, 
                             dim="Variables", aggregation = "sum", regex=False, inplace=False)
idx = dm_temp.idx
for c in countries:
    for y in years_fts:
            dm_temp.array[idx[c],idx[y],idx["total"]] = np.nan
dm_bld_tot.array = dm_bld_tot.array / dm_temp.array
for c in dm_bld_tot.col_labels["Country"]:
    for y in years_all:
            dm_bld_tot.array[idx[c],idx[y],:] = adjust_shares(dm_bld_tot.array[idx[c],idx[y],:])
for v in dm_bld_tot.col_labels["Variables"]: 
    dm_bld_tot.units[v] = "%"

# dm_bld_tot.filter({"Country" : ["EU27"]}).datamatrix_plot()

# rename floor-area-new-residential and floor-area-new-non-residential
for v in dm_bld_col.col_labels["Variables"]:
    dm_bld_col.rename_col(v,"floor-area-new-residential" + "_" + v,"Variables")
dm_bld_col.deepen()
dm_temp = dm_bld_col.copy()
dm_temp.rename_col("floor-area-new-residential","floor-area-new-non-residential","Variables")
dm_bld_col.append(dm_temp, "Variables")
dm_bld_col.sort("Variables")
for v in dm_bld_tot.col_labels["Variables"]:
    dm_bld_tot.rename_col(v,"floor-area-new-residential" + "_" + v,"Variables")
dm_bld_tot.deepen()
dm_temp = dm_bld_tot.copy()
dm_temp.rename_col("floor-area-new-residential","floor-area-new-non-residential","Variables")
dm_bld_tot.append(dm_temp, "Variables")
dm_bld_tot.sort("Variables")

# clean
del baseyear, baseyear_end, baseyear_start, c, countries, dm_bld,\
    dm_temp, idx, lastyear, startyear, step_fts, v, y

################
##### SAVE #####
################

# store
DM_wst_mgt["bld-total"] = dm_bld_tot
DM_wst_mgt["bld-col"] = dm_bld_col

# clean
del dm_bld_tot, dm_bld_col

###############################################################################
#################################### ROADS ####################################
###############################################################################

# assuming same of buldings, as buildnigs are based on construction eol, which includes roads

dm_road_tot = DM_wst_mgt["bld-total"].filter({"Variables" : ['floor-area-new-non-residential']})
dm_road_tot.rename_col("floor-area-new-non-residential","road","Variables")
DM_wst_mgt["road-total"] = dm_road_tot

dm_road_col = DM_wst_mgt["bld-col"].filter({"Variables" : ['floor-area-new-non-residential']})
dm_road_col.rename_col("floor-area-new-non-residential","road","Variables")
DM_wst_mgt["road-col"] = dm_road_col

###############################################################################
#################################### RAILS ####################################
###############################################################################

# source: https://www.researchgate.net/publication/388232821_A_life_cycle_model_for_high-speed_rail_infrastructure_environmental_inventories_and_assessment_of_the_Tours-Bordeaux_railway_in_France
# table 2: Components have different lifespans and possible second lives that need to be considered carefully as well as 
# transparently explained. Assumptions regarding component replacement, maintenance and End-of-Life (EoL) are presented in TABLE 2, 
# based on data provided by French railway experts.
# rail: 80% recycling - 20% reuse

dm_rail_tot = DM_wst_mgt["bld-total"].filter({"Variables" : ['floor-area-new-non-residential']})
dm_rail_tot.rename_col("floor-area-new-non-residential","rail","Variables")
DM_wst_mgt["rail-total"] = dm_rail_tot

dm_rail_col = DM_wst_mgt["bld-col"].filter({"Variables" : ['floor-area-new-non-residential']})
dm_rail_col.rename_col("floor-area-new-non-residential","rail","Variables")
for y in years_ots:
    dm_rail_col[:,y,:,:] = 0
    dm_rail_col[:,y,:,"recycling"] = 0.8
    dm_rail_col[:,y,:,"reuse"] = 0.2
for y in years_fts:
    dm_rail_col["EU27",y,:,:] = 0
    dm_rail_col["EU27",y,:,"recycling"] = 0.8
    dm_rail_col["EU27",y,:,"reuse"] = 0.2
DM_wst_mgt["rail-col"] = dm_rail_col
    
del dm_rail_col, dm_rail_tot, years_ots, years_fts, years_after_regulation, years_all, years_before_regulation, years_setting

###############################################################################
################################ TROLLEY CABLES ###############################
###############################################################################

# assuming that eol of trolly cables is similar to the one of copper, which is
# contained in env_wastrt, the variable metal wastes - non ferrous (W062)

# get data
df = eurostat.get_data_df("env_wastrt")
df.rename(columns={'geo\\TIME_PERIOD': 'geoscale'}, inplace=True)
df = df.loc[df['unit'] == 'T',:]
df = df.loc[df["waste"] == "W062",:]
df = df.loc[df["hazard"] == "HAZ_NHAZ",:]
df.drop(columns=["hazard"],inplace=True)

# formulas:
# littered: 0
# exported: 0
# collected: recycling + energy recovery + reuse (backfilling) + landfill + incineration
# uncollected: 0
# recycling: RCV_R - W121
# energy recovery: RCV_E - W121
# reuse: RCV_B - W121
# landfill: DSP_L - W121
# incineration: DSP_I - W121

# mapping
dict_mapping = {"recycling": ["RCV_R - W062"],
                "energy-recovery": ['RCV_E - W062'],
                "reuse": ["RCV_B - W062"],
                "landfill": ['DSP_L - W062'],
                "incineration" : ["DSP_I - W062"]
                }
indexes = ['freq', 'wst_oper', 'waste', 'unit', 'geoscale']
df = pd.melt(df, id_vars = indexes, var_name='year')
df["combo"] = [i + " - " + j for i, j in zip(df["wst_oper"], df["waste"])]
indexes = ['freq', 'wst_oper', 'unit', 'geoscale','year']
key = "energy-recovery"
def my_aggregation(key, df, dict_mapping, indexes):
    df_temp = df.loc[df["combo"].isin(dict_mapping[key]),:]
    df_temp = df_temp.groupby(indexes, as_index=False)['value'].agg(sum)
    df_temp["variable"] = key
    return df_temp
df_trc = pd.concat([my_aggregation(key, df, dict_mapping, indexes) for key in dict_mapping.keys()])

# select countries
df_trc = df_trc.loc[:,['geoscale', 'year', 'variable', 'unit', 'value']]
country_list = {'AT' : "Austria", 'BE' : "Belgium", 'BG': "Bulgaria", 'HR' : "Croatia", 'CY' : "Cyprus", 
                'CZ' : "Czech Republic", 'DK' : "Denmark", 'EE' : "Estonia", 'EU27_2020': "EU27", 'FI' : "Finland",
                'FR' : "France", 'DE' : "Germany", 'EL' : "Greece", 'HU' : "Hungary", 'IE' : "Ireland",
                'IT' : "Italy", 'LV' : "Latvia", 'LT' : "Lithuania", 'LU' : "Luxembourg", 
                'MT' : "Malta", 'NL' : "Netherlands",
                'PL' : "Poland", 'PT' : "Portugal", 'RO' : "Romania", 'SK' : "Slovakia", 'SI' : "Slovenia", 
                'ES' : "Spain", 'SE' : "Sweden", 'UK' : 'United Kingdom'}
for c in country_list.keys():
    df_trc.loc[df_trc["geoscale"] == c,"geoscale"] = country_list[c]
drops = ['AL','BA','IS', 'ME', 'EU28','MK', 'LI', 'NO', 'RS', 'TR', 'XK']
df_trc = df_trc.loc[~df_trc["geoscale"].isin(drops),:]
len(df_trc["geoscale"].unique())

# fix variable name with unit
df_trc["variable"] = [i + "[t]" for i in df_trc["variable"]]
df_trc.drop(columns="unit",inplace=True)

# checks
df_temp = df_trc.pivot(index=["geoscale","year"], columns="variable", values='value').reset_index()

# drop before 2008 (all zero)
df_trc = df_trc.loc[df_trc["year"] > "2008",:]

# clean
del c, country_list, df, df_temp, dict_mapping, drops,\
    indexes, key
    
##################################
##### CONVERT TO DATA MATRIX #####
##################################

# rename
df_trc.rename(columns={"geoscale":"Country","year":"Years"},inplace=True)

# make dm
df_temp = df_trc.copy()
df_temp = df_temp.pivot(index=["Country","Years"], columns="variable", values='value').reset_index()
dm_trc = DataMatrix.create_from_df(df_temp, 0)

# # plot
# dm_trc.filter({"Country" : ["EU27"], "Variables" : ["recycling"]}).datamatrix_plot()
# dm_trc.filter({"Country" : ["EU27"]}).datamatrix_plot()
# landfill: probably consider until 2010 (when it's growing) for backward interpolation
# df_temp = dm_trc.write_df()

# clean
del df_temp, df_trc

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

# fill nas (including 2023)
years_missing = np.array(years_ots)[[y not in dm_trc.col_labels["Years"] for y in years_ots]].tolist()
for y in years_missing:    
    dm_trc.add(np.nan, col_label=[y], dummy=True, dim='Years')
dm_trc.sort("Years")
years_fitting = dm_trc.col_labels["Years"]
dm_trc = linear_fitting(dm_trc, years_fitting, min_t0=0, min_tb=0)
dm_trc.array = np.round(dm_trc.array,0)

# # plot
# dm_trc.filter({"Country" : ["EU27"], "Variables" : ["reuse"]}).datamatrix_plot()
# dm_trc.filter({"Country" : ["EU27"]}).datamatrix_plot()
# df_temp = dm_trc.write_df()

# clean
del years_fitting, years_missing

####################
##### MAKE FTS #####
####################

# make function to fill in missing years fts for EU27 with linear fitting
def make_fts(dm, variable, year_start, year_end, country = "EU27", dim = "Variables", 
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
dm_trc.add(np.nan, col_label=years_fts, dummy=True, dim='Years')

# set default time window for linear trend
baseyear_start = 2010
baseyear_end = 2023

# compute fts
dm_trc = make_fts(dm_trc, "energy-recovery", baseyear_start, baseyear_end)
dm_trc = make_fts(dm_trc, "landfill", baseyear_start, baseyear_end)
dm_trc = make_fts(dm_trc, "recycling", baseyear_start, baseyear_end)
dm_trc = make_fts(dm_trc, "reuse", baseyear_start, baseyear_end)
dm_trc = make_fts(dm_trc, "incineration", baseyear_start, baseyear_end)

# # check
# df_temp = dm_trc.write_df()
# dm_trc.filter({"Country" : ["EU27"]}).datamatrix_plot()

################################
##### MAKE FINAL VARIABLES #####
################################

# make dm for collected waste
dm_trc_col = dm_trc.copy()
dm_trc_col.sort("Variables")
dm_trc_tot = dm_trc_col.groupby(group_cols={"collected" : ['energy-recovery', 'incineration', 
                                                           'landfill', 'recycling', 'reuse']}, 
                                dim="Variables", aggregation = "sum", regex=False, inplace=False)

# re-put nan in years fts for all countries but EU27
idx = dm_trc_tot.idx
countries = dm_trc_tot.col_labels["Country"]
countries = list(np.array(countries)[[i != "EU27" for i in countries]])
for c in countries:
    for y in years_fts:
            dm_trc_tot.array[idx[c],idx[y],:] = np.nan

# compute the shares rounded at 2 decimals
dm_trc_col.array = dm_trc_col.array / dm_trc_tot.array
def adjust_shares(shares):
    # Round shares to two decimal places
    rounded_shares = [round(share, 2) for share in shares]
    total = sum(rounded_shares)
    
    # Calculate the discrepancy
    discrepancy = round(1.0 - total, 2)
    
    # Adjust the share with the largest absolute value
    if discrepancy != 0:
        # Find the index of the share to adjust
        idx_to_adjust = max(range(len(rounded_shares)), key=lambda i: rounded_shares[i])
        rounded_shares[idx_to_adjust] += discrepancy
        # Ensure the adjustment doesn't break the two-decimal constraint
        rounded_shares[idx_to_adjust] = round(rounded_shares[idx_to_adjust], 2)
    
    return rounded_shares
idx = dm_trc_col.idx
for c in dm_trc_col.col_labels["Country"]:
    for y in years_all:
            dm_trc_col.array[idx[c],idx[y],:] = adjust_shares(dm_trc_col.array[idx[c],idx[y],:])
for v in dm_trc_col.col_labels["Variables"]: 
    dm_trc_col.units[v] = "%"

# # checks
# dm_trc_col.filter({"Country" : ["EU27"]}).datamatrix_plot()

# make dm for total waste
dm_trc_tot.rename_col("collected","waste-collected","Variables")
dm_trc_tot.add(0, col_label="waste-uncollected", dummy=True, dim='Variables', unit="t")
dm_trc_tot.add(0, col_label="littered", dummy=True, dim='Variables', unit="t")
dm_trc_tot.add(0, col_label="export", dummy=True, dim='Variables', unit="t")
dm_trc_tot.sort("Variables")
idx = dm_trc_tot.idx
countries = dm_trc_tot.col_labels["Country"]
countries = list(np.array(countries)[[i != "EU27" for i in countries]])
for c in countries:
    for y in years_fts:
            dm_trc_tot.array[idx[c],idx[y],idx["littered"]] = np.nan
            dm_trc_tot.array[idx[c],idx[y],idx["export"]] = np.nan
            dm_trc_tot.array[idx[c],idx[y],idx["waste-uncollected"]] = np.nan

# dm_trc_tot.filter({"Country" : ["EU27"]}).datamatrix_plot()

# make shares in dm
# compute the shares rounded at 2 decimals
dm_temp = dm_trc_tot.groupby(group_cols={"total" : ['waste-collected', 'export', 'waste-uncollected', 'littered']}, 
                             dim="Variables", aggregation = "sum", regex=False, inplace=False)
idx = dm_temp.idx
for c in countries:
    for y in years_fts:
            dm_temp.array[idx[c],idx[y],idx["total"]] = np.nan
dm_trc_tot.array = dm_trc_tot.array / dm_temp.array
for c in dm_trc_tot.col_labels["Country"]:
    for y in years_all:
            dm_trc_tot.array[idx[c],idx[y],:] = adjust_shares(dm_trc_tot.array[idx[c],idx[y],:])
for v in dm_trc_tot.col_labels["Variables"]: 
    dm_trc_tot.units[v] = "%"

# dm_trc_tot.filter({"Country" : ["EU27"]}).datamatrix_plot()

# rename trolley-cables
for v in dm_trc_col.col_labels["Variables"]:
    dm_trc_col.rename_col(v,"trolley-cables" + "_" + v,"Variables")
dm_trc_col.deepen()
for v in dm_trc_tot.col_labels["Variables"]:
    dm_trc_tot.rename_col(v,"trolley-cables" + "_" + v,"Variables")
dm_trc_tot.deepen()

# clean
del baseyear, baseyear_end, baseyear_start, c, countries, dm_trc,\
    dm_temp, idx, lastyear, startyear, step_fts, v, years_all, \
    years_fts, years_ots, years_setting, y

################
##### SAVE #####
################

# store
DM_wst_mgt["trc-total"] = dm_trc_tot
DM_wst_mgt["trc-col"] = dm_trc_col

# clean
del dm_trc_tot, dm_trc_col


###############################################################################
################### LARGER APPLIANCES, AND PC & ELECTRONICS ###################
###############################################################################

####################
##### GET DATA #####
####################

# get data
df = eurostat.get_data_df("env_waselee")
# filepath = os.path.join(current_file_directory, '../data/eurostat/env_waselee.csv')
# df.to_csv(filepath, index = False)
# df = pd.read_csv(filepath)

df_mun = eurostat.get_data_df("env_wasmun")
# filepath = os.path.join(current_file_directory, '../data/eurostat/env_wasmun.csv')
# df_mun.to_csv(filepath, index = False)
# df_mun = pd.read_csv(filepath)
df_mun["wst_oper"].unique()

# get geo column
df.rename(columns={'geo\\TIME_PERIOD': 'geoscale'}, inplace=True)
df_mun.rename(columns={'geo\\TIME_PERIOD': 'geoscale'}, inplace=True)

# filter for unit of measure: Tonne
df = df.loc[df['unit'] == 'T',:]
df_mun = df_mun.loc[df_mun['unit'] == 'THS_T',:]

# get larger appliances and pc&electronics
df = df.loc[df["waste"].isin(['EE_LHA','EE_ITT']),:]

# checks
A = df.copy()
A["combo"] = [i + " - " + j for i, j in zip(df["waste"], df["wst_oper"])]
list(A["combo"].unique())
df_temp = df.copy()
df_temp = df_temp.loc[df_temp["geoscale"] == "AT",:]
df_temp = df_temp.loc[:,['freq', 'wst_oper', 'waste', 'unit', 'geoscale',"2018"]]
df_temp_mun = df_mun.copy()
df_temp_mun = df_temp_mun.loc[df_temp_mun["geoscale"] == "AT",:]
df_temp_mun = df_temp_mun.loc[:,['freq', 'wst_oper', 'unit', 'geoscale',"2018"]]


# in general for us:
# total = littered + exported + collected + uncollected
# collected = recycling + energy recovery + reuse + landfill + incineration

# in general in eurostat
# Waste generation: The quantity of waste, whereby ‘waste’ means any substance or object which the holder discards or intends or is required to discard.
# Waste management: Waste management refers to the collection, transport, recovery (including sorting), and disposal of waste.
# Treatment: Treatment means recovery or disposal operations, including preparation prior to recovery or disposal.
# Recovery: Recovery means any operation whose main result is that waste serves a useful purpose. For example, by replacing other materials that would have been used for a particular function.
# Disposal: Disposal means any operation that is not recovery (personal note: I guess this is either incineration or landfill)

# specifically for env_waselee
# COL = COL_HH + COL_OTH: collection (and from private households and other than private households)
# MKT: put on the market (this is no waste it's EEE)
# PRP_REU: preparing for reuse as whole appliances 
# RCV: recovery
# RCY_PRP_REU: recycling and preparing for reuse
# TRT = COL = TRT_NAT + TRT_EU_FOR + TRT_NEU: waste treatment (and waste treated in member state, waste treated
# in another member state, waste treated outside EU)
# probably TRT_NAT - RCV = DSP = landfill + incineration
# probably RCV = PRP_REU + RCY_PRP_REU + energy recovery

# specifically for env_wasmun
# DSP_I: incineration
# DSP_I_RCV_E: energy recovery (in disposal)
# DSP_L_OTH: landfill and other
# GEN: generated
# PRP_REU: preparing for reuse
# RCV_E: energy recovery (in recovery)
# RCY = RCY_C_D + RCY_M: recycling (and composting and digestion, and material)
# TRT: treated
# usually DSP_I_RCV_E = RCV_E
# probably GEN - TRT = uncollected + littered
# TRT = DSP_I + DSP_L_OTH + PRP_REU + RCV_E + RCY

# so, formulas:
# littered: 0
# exported: among “wst_oper”, TRT_NEU + TRT_NAT (aside for EU27, where we do only TRT_NEU)
# collected: recycling + energy recovery + reuse + landfill + incineration
# uncollected: collected / (1 - 0.5963) * 0.5963
# recycling: among “wst_oper”, RCY_PRP_REU
# energy recovery: RCV - PRP_REU - RCY_PRP_EU
# reuse: among “wst_oper”, PRP_REU
# landfill: (TRT_EU_FOR - RCV) * (DSP_L_OTH / (DSP_I + DSP_L_OTH))
# incineration: (TRT_EU_FOR - RCV) * (DSP_I / (DSP_I + DSP_L_OTH))

# mapping
dict_mapping = {"RCY_PRP_REU" : "recycling",
                "PRP_REU": "reuse",
                "TRT_NEU": "export", # for EU27 it will be only NEU (and not TRT_EU_FOR)
                "TRT_EU_FOR" : "export",
                "TRT_NAT" : "waste-collected", # this will be used as check and to create disposal
                "RCV" : "recovery" # this will be used as check and to create energy recovery and disposal
                }

# make long format
indexes = ['freq', 'wst_oper', 'waste', 'unit', 'geoscale']
df = pd.melt(df, id_vars = indexes, var_name='year')

# select
df = df.loc[df["wst_oper"].isin(list(dict_mapping.keys())),:]
df["variable"] = np.nan
for key in dict_mapping.keys():
    df.loc[df["wst_oper"] == key,"variable"] = dict_mapping[key]

# select countries
df = df.loc[:,['geoscale', 'year', 'variable', 'wst_oper', 'waste', 'unit', 'value']]
country_list = {'AT' : "Austria", 'BE' : "Belgium", 'BG': "Bulgaria", 'HR' : "Croatia", 'CY' : "Cyprus", 
                'CZ' : "Czech Republic", 'DK' : "Denmark", 'EE' : "Estonia", 'FI' : "Finland",
                'FR' : "France", 'DE' : "Germany", 'EL' : "Greece", 'HU' : "Hungary", 'IE' : "Ireland",
                'IT' : "Italy", 'LV' : "Latvia", 'LT' : "Lithuania", 'LU' : "Luxembourg", 
                'MT' : "Malta", 'NL' : "Netherlands",
                'PL' : "Poland", 'PT' : "Portugal", 'RO' : "Romania", 'SK' : "Slovakia", 'SI' : "Slovenia", 
                'ES' : "Spain", 'SE' : "Sweden", 'UK' : "United Kingdom"}
for c in country_list.keys():
    df.loc[df["geoscale"] == c,"geoscale"] = country_list[c]
drops = ['IS', 'LI', 'NO', 'EU27_2020']
df = df.loc[~df["geoscale"].isin(drops),:]
len(df["geoscale"].unique())

# make EU27
indexes = ['year', 'waste', 'variable', 'wst_oper', 'unit']
countries = ['Austria', 'Belgium', 'Bulgaria', 'Cyprus', 'Czech Republic',
             'Germany', 'Denmark', 'Estonia', 'Greece', 'Spain', 'Finland',
             'France', 'Croatia', 'Hungary', 'Ireland', 'Italy', 'Lithuania',
             'Luxembourg', 'Latvia', 'Malta', 'Netherlands', 'Poland',
             'Portugal', 'Romania', 'Sweden', 'Slovenia', 'Slovakia']
df_temp = df.loc[df["geoscale"].isin(countries),:]
df_temp = df_temp.loc[df_temp["wst_oper"] != "TRT_EU_FOR",:]
df_temp = df_temp.groupby(indexes, as_index=False)['value'].agg(sum)
df_temp["geoscale"] = "EU27"
df = pd.concat([df, df_temp])
indexes = ['geoscale', 'waste', 'variable', 'wst_oper', 'year', 'unit']
df.sort_values(indexes, inplace=True)

# aggregate by variable (so it will compute export as sum of within EU and extra EU)
indexes = ['geoscale', 'waste', 'variable', 'year', 'unit']
df = df.groupby(indexes, as_index=False)['value'].agg(sum)

# pivot to create variables
df = df.pivot(index=['geoscale', 'waste', 'year', 'unit'], 
              columns="variable", values='value').reset_index()

# fill in nans for Czech Republic, Italy and Lithuania.
df_temp = df.loc[df["geoscale"].isin(["Czech Republic","Italy","Lithuania"]),:]
df.loc[df['reuse'].isnull(),"reuse"] = df.loc[df['reuse'].isnull(),"recovery"] - \
    df.loc[df['reuse'].isnull(),"recycling"]
df.loc[df["reuse"] <0,"reuse"] = 0

# fix: when recycling + reuse > recovery -> recovery = recycling + reuse
df.loc[df["recycling"] + df["reuse"] > df["recovery"], "recovery"] = \
    df.loc[df["recycling"] + df["reuse"] > df["recovery"], "recycling"] + \
    df.loc[df["recycling"] + df["reuse"] > df["recovery"], "reuse"]

# fix when recovery > waste-collected -> waste-collected = recovery
df.loc[df["recovery"] > df["waste-collected"],"waste-collected"] =\
    df.loc[df["recovery"] > df["waste-collected"],"recovery"]

# for waste-collected, when is zero fill in with next valid value
df.loc[df["waste-collected"] == 0,"waste-collected"] = np.nan
df = df.bfill()

# clean df_mun (to get (DSP_L_OTH / (DSP_I + DSP_L_OTH)) and (DSP_I / (DSP_I + DSP_L_OTH)) to compute landfill and incineration)
df_mun  = df_mun.loc[df_mun["wst_oper"].isin(["DSP_L_OTH","DSP_I"]),:]
indexes = ['geoscale', 'freq','wst_oper', 'unit']
df_mun = pd.melt(df_mun, id_vars = indexes, var_name='year')
df_mun = df_mun.loc[:,['geoscale', 'year', 'wst_oper', 'unit', 'value']]
for c in country_list.keys():
    df_mun.loc[df_mun["geoscale"] == c,"geoscale"] = country_list[c]
df_mun["geoscale"].unique()
drops = ['AL','BA','CH','EU27_2020','IS','ME','MK','NO','RS','TR','XK']
df_mun = df_mun.loc[~df_mun["geoscale"].isin(drops),:]
indexes = ['year', 'wst_oper', 'unit']
df_temp = df_mun.groupby(indexes, as_index=False)['value'].agg(sum)
df_temp["geoscale"] = "EU27"
df_mun = pd.concat([df_mun, df_temp])
indexes = ['geoscale', 'wst_oper', 'year', 'unit']
df_mun.sort_values(indexes, inplace=True)
df_mun["value"] = df_mun["value"] * 1000
df_mun["unit"] = "T"
df_mun = df_mun.pivot(index=['geoscale', 'year', 'unit'], columns="wst_oper", 
                      values='value').reset_index()
df_mun["incineration-coeff"] = df_mun["DSP_I"] / (df_mun["DSP_I"] + df_mun["DSP_L_OTH"])
df_mun["landfill-coeff"] = df_mun["DSP_L_OTH"] / (df_mun["DSP_I"] + df_mun["DSP_L_OTH"])
df_mun = df_mun.bfill() # fill in missing with next available

# for UK, put the values of Germany. 
# TODO: find values for UK
df_temp = df_mun.loc[df_mun["geoscale"] == "Germany",:]
df_temp["geoscale"] = "United Kingdom"
df_mun = pd.concat([df_mun, df_temp])

# compute incineration and landfill in df
df = pd.merge(df, df_mun, how="left", on=['geoscale', 'year', 'unit'])
df["incineration"] = np.round((df["waste-collected"] - df["recovery"])*df["incineration-coeff"],0)
df["landfill"] = np.round((df["waste-collected"] - df["recovery"])*df["landfill-coeff"],0)

# check
check = df["waste-collected"] != df["recovery"] + df["incineration"] + df["landfill"]
df_temp = df.loc[check,:]

# make energy recovery
df["energy-recovery"] = df['recovery'] - df['recycling'] - df['reuse']

# check
check = df["recovery"] != df["recycling"] + df["reuse"] + df["energy-recovery"]
df_temp = df.loc[check,:]
check = df["waste-collected"] != df["recovery"] + df["incineration"] + df["landfill"]
df_temp = df.loc[check,:]

# clean
df.drop(["incineration-coeff","landfill-coeff",'DSP_I', 'DSP_L_OTH','recovery',
         'waste-collected'],axis=1,inplace=True)
df = df.melt(id_vars=['geoscale', 'waste', 'year', 'unit'], var_name='variable',value_name='value')

# fix variable name
df.loc[df["waste"] == "EE_ITT","waste"] = "electronics"
df.loc[df["waste"] == "EE_LHA","waste"] = "domapp"
df["variable"] = [w + "_" + v + "[t]" for w, v in zip(df["waste"], df["variable"])]
df.drop(columns=["waste","unit"],inplace=True)
df.sort_values(by=['geoscale','variable','year'], inplace=True)

# clean
del A, c, check, countries, country_list, df_temp, df_temp_mun, dict_mapping, drops,\
    indexes, key, df_mun

##################################
##### CONVERT TO DATA MATRIX #####
##################################

# rename
df.rename(columns={"geoscale":"Country","year":"Years"},inplace=True)

# # put nan where is 0
# df.loc[df["value"] == 0,"value"] = np.nan

# make dm
df_temp = df.copy()
df_temp = df_temp.pivot(index=["Country","Years"], columns="variable", values='value').reset_index()
dm_domapp = DataMatrix.create_from_df(df_temp, 1)

# # plot
# dm_domapp.filter({"Country" : ["EU27"], "Variables" : ["reuse"]}).datamatrix_plot()
# dm_domapp.filter({"Country" : ["EU27"]}).datamatrix_plot()
# df_temp = dm_domapp.write_df()

# clean
del df_temp, df

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

# fill 2019-2023 based on 2018
dm_domapp.add(np.nan, col_label=list(range(2019,2023+1,1)), dummy=True, dim='Years')
years_fitting = dm_domapp.col_labels["Years"]
dm_domapp = linear_fitting(dm_domapp, years_fitting, min_t0=0, min_tb=0, based_on=[2018])

# # plot
# dm_domapp.filter({"Country" : ["EU27"]}).datamatrix_plot() 
# df_temp = dm_domapp.write_df()

# # fix jumps
# dm_domapp = fix_jumps_in_dm(dm_domapp)

# # plot
# dm_domapp.filter({"Country" : ["EU27"]}).datamatrix_plot()
# df_temp = dm_domapp.write_df()

# add missing years ots
years_missing = list(set(years_ots) - set(dm_domapp.col_labels['Years']))
dm_domapp.add(np.nan, col_label=years_missing, dummy=True, dim='Years')
dm_domapp.sort('Years')

# fill in missing years ots with linear fitting
# I am simply putting the value of 2005 for all past years, as this works best for shares later
# (with trends, things can get mixed up, like more reuse than recycling, or zero landfilling)
# incineration data before 2007 is low and there is a weird jump, so i start from 2007
dm_temp = dm_domapp.filter({"Categories1" : ['incineration','landfill']})
idx = dm_temp.idx
dm_temp.array[:,idx[2005],:] = np.nan
dm_temp.array[:,idx[2006],:] = np.nan
dm_temp = linear_fitting(dm_temp, years_ots = years_missing, min_t0=0, min_tb=0, based_on = [2007])
dm_domapp = linear_fitting(dm_domapp.filter({"Categories1" : ['energy-recovery', 'export', 
                                                              'recycling', 'reuse']}), 
                           years_ots = years_missing, min_t0=0, min_tb=0, based_on = [2005])
dm_domapp.append(dm_temp, "Categories1")

# # plot
# dm_domapp.filter({"Country" : ["EU27"], "Variables" : ["reuse"]}).datamatrix_plot()
# dm_domapp.filter({"Country" : ["EU27"]}).datamatrix_plot()
# df_temp = dm_domapp.write_df()

# clean
del years_missing, idx

####################
##### MAKE FTS #####
####################

# make function to fill in missing years fts for EU27 with linear fitting
def make_fts(dm, variable, year_start, year_end, country = "EU27", dim = "Variables", 
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
dm_domapp.add(np.nan, col_label=years_fts, dummy=True, dim='Years')

# set default time window for linear trend
# assumption: best is taking longer trend possible to make predictions to 2050 (even if earlier data is generated)
baseyear_start = 1990
baseyear_end = 2018

# compute fts
dm_domapp = dm_domapp.flatten()
dm_domapp = make_fts(dm_domapp, "domapp_energy-recovery", baseyear_start, baseyear_end)
dm_domapp = make_fts(dm_domapp, "domapp_export", baseyear_start, baseyear_end)
dm_domapp = make_fts(dm_domapp, "domapp_landfill", 2010, baseyear_end) # we keep the decreasing trend since 2010
dm_domapp = make_fts(dm_domapp, "domapp_incineration", 2010, baseyear_end) # we keep the decreasing trend since 2010
dm_domapp = make_fts(dm_domapp, "domapp_recycling", baseyear_start, baseyear_end)
dm_domapp = make_fts(dm_domapp, "domapp_reuse", baseyear_start, baseyear_end)
dm_domapp = make_fts(dm_domapp, "electronics_energy-recovery", baseyear_start, baseyear_end)
dm_domapp = make_fts(dm_domapp, "electronics_export", baseyear_start, baseyear_end)
dm_domapp = make_fts(dm_domapp, "electronics_landfill", 2010, baseyear_end) # we keep the decreasing trend since 2010
dm_domapp = make_fts(dm_domapp, "electronics_incineration", 2010, baseyear_end) # we keep the decreasing trend since 2010
dm_domapp = make_fts(dm_domapp, "electronics_recycling", baseyear_start, baseyear_end)
dm_domapp = make_fts(dm_domapp, "electronics_reuse", baseyear_start, baseyear_end)
# variable = "electronics_export"
# (make_fts(dm_domapp, variable, baseyear_start, baseyear_end).
#   datamatrix_plot(selected_cols={"Country" : ["EU27"],
#                                 "Variables" : [variable]}))

# # check
# dm_domapp.filter({"Country" : ["EU27"]}).datamatrix_plot()
# df_temp = dm_domapp.write_df()

# deepen back
dm_domapp.deepen()

################################
##### MAKE FINAL VARIABLES #####
################################

# make dm for collected waste
dm_domapp_col = dm_domapp.filter({"Categories1" : ['energy-recovery', 'incineration', 'landfill', 'recycling', 'reuse']})
dm_domapp_tot = dm_domapp_col.groupby(group_cols={"collected" : ['energy-recovery', 'incineration', 
                                                                 'landfill', 'recycling', 'reuse']}, 
                                dim="Categories1", aggregation = "sum", regex=False, inplace=False)

# re-put nan in years fts for all countries but EU27
idx = dm_domapp_tot.idx
countries = dm_domapp_tot.col_labels["Country"]
countries = list(np.array(countries)[[i != "EU27" for i in countries]])
for c in countries:
    for y in years_fts:
            dm_domapp_tot.array[idx[c],idx[y],:] = np.nan

# compute the shares rounded at 2 decimals
dm_domapp_col.array = dm_domapp_col.array / dm_domapp_tot.array
idx = dm_domapp_col.idx
for c in dm_domapp_col.col_labels["Country"]:
    for y in years_all:
        for v in dm_domapp_col.col_labels["Variables"]:
            dm_domapp_col.array[idx[c],idx[y],idx[v],:] = adjust_shares(dm_domapp_col.array[idx[c],idx[y],idx[v],:])
for v in dm_domapp_col.col_labels["Variables"]: 
    dm_domapp_col.units[v] = "%"

# # checks
# dm_temp = dm_domapp_col.groupby(group_cols={"total" : dm_domapp_col.col_labels["Categories1"]}, 
#                                 dim="Categories1", aggregation = "sum", regex=False, inplace=False)
# for c in countries:
#     for y in years_fts:
#             dm_temp.array[idx[c],idx[y],:] = np.nan
# dm_temp.array
# df_temp = dm_temp.write_df() # there are some countries with potential missing values before 2005, fine for now
# df_temp = dm_domapp_col.write_df()
# df_temp = dm_domapp_tot.write_df()
# dm_domapp_col.filter({"Country" : ["EU27"]}).datamatrix_plot()

# the nan in ots are because of division by 0, for the moment I keep them as nan (as we'll use only EU27)
# when other countries are implemented, we should re-consider.

# make dm for total waste
dm_domapp_tot.append(dm_domapp.filter({"Categories1" : ["export"]}), "Categories1")
dm_domapp_tot.rename_col("collected","waste-collected","Categories1")
idx = dm_domapp_tot.idx
arr_temp = dm_domapp_tot.array[:,:,:,idx["waste-collected"], np.newaxis] / 0.8 * 0.2
dm_domapp_tot.add(arr_temp, dim = "Categories1", col_label = "waste-uncollected", unit="t")
dm_domapp_tot.add(0, col_label="littered", dummy=True, dim='Categories1', unit="t")
idx = dm_domapp_tot.idx
countries = dm_domapp_tot.col_labels["Country"]
countries = list(np.array(countries)[[i != "EU27" for i in countries]])
for c in countries:
    for y in years_fts:
            dm_domapp_tot.array[idx[c],idx[y],:,idx["littered"]] = np.nan

# dm_domapp_tot.filter({"Country" : ["EU27"]}).datamatrix_plot()

# make shares in dm
# compute the shares rounded at 2 decimals
dm_temp = dm_domapp_tot.groupby(group_cols={"total" : ['waste-collected', 'export', 'waste-uncollected', 'littered']}, 
                                dim="Categories1", aggregation = "sum", regex=False, inplace=False)
idx = dm_temp.idx
for c in countries:
    for y in years_fts:
            dm_temp.array[idx[c],idx[y],:,idx["total"]] = np.nan
dm_domapp_tot.array = dm_domapp_tot.array / dm_temp.array
for c in dm_domapp_tot.col_labels["Country"]:
    for y in years_all:
        for v in dm_domapp_tot.col_labels["Variables"]:
            dm_domapp_tot.array[idx[c],idx[y],idx[v],:] = adjust_shares(dm_domapp_tot.array[idx[c],idx[y],idx[v],:])
for v in dm_domapp_tot.col_labels["Variables"]: 
    dm_domapp_tot.units[v] = "%"

# df_temp = dm_domapp_tot.write_df()
# dm_domapp_tot.filter({"Country" : ["EU27"]}).datamatrix_plot()
# note: export goes to 0 because of the rounding and adjust_shares

# clean
del arr_temp, baseyear, baseyear_end, baseyear_start, c, countries, dm_domapp, \
    dm_temp, idx, lastyear, startyear, step_fts, v, years_all, \
    years_fts, years_ots, years_setting, y

################
##### SAVE #####
################

# store
DM_wst_mgt["domapp-total"] = dm_domapp_tot.filter({"Variables" : ["domapp"]})
DM_wst_mgt["electronics-total"] = dm_domapp_tot.filter({"Variables" : ["electronics"]})
DM_wst_mgt["domapp-col"] = dm_domapp_col.filter({"Variables" : ["domapp"]})
DM_wst_mgt["electronics-col"] = dm_domapp_col.filter({"Variables" : ["electronics"]})

# clean
del dm_domapp_tot, dm_domapp_col


#####################################################
################### PACKAGE WASTE ###################
#####################################################

####################
##### GET DATA #####
####################

# get data
df = eurostat.get_data_df("env_waspac")
# filepath = os.path.join(current_file_directory, '../data/eurostat/env_waspac.csv')
# df.to_csv(filepath, index = False)
# df = pd.read_csv(filepath)

df_mun = eurostat.get_data_df("env_wasmun")
# filepath = os.path.join(current_file_directory, '../data/eurostat/env_wasmun.csv')
# df_mun.to_csv(filepath, index = False)
# df_mun = pd.read_csv(filepath)
df_mun["wst_oper"].unique()

# get geo column
df.rename(columns={'geo\\TIME_PERIOD': 'geoscale'}, inplace=True)
df_mun.rename(columns={'geo\\TIME_PERIOD': 'geoscale'}, inplace=True)

# filter for unit of measure: Tonne
df = df.loc[df['unit'] == 'T',:]
df_mun = df_mun.loc[df_mun['unit'] == 'THS_T',:]

# get waste packages: paper, plastic, aluminium, glass
df = df.loc[df["waste"].isin(['W150101','W150102','W15010401', 'W150107']),:]

# checks
A = df.copy()
A["combo"] = [i + " - " + j for i, j in zip(df["waste"], df["wst_oper"])]
list(A["combo"].unique())
df_temp = df.copy()
df_temp = df_temp.loc[df_temp["geoscale"] == "AT",:]
df_temp = df_temp.loc[:,['freq', 'wst_oper', 'waste', 'unit', 'geoscale',"2018"]]
df_temp_mun = df_mun.copy()
df_temp_mun = df_temp_mun.loc[df_temp_mun["geoscale"] == "AT",:]
df_temp_mun = df_temp_mun.loc[:,['freq', 'wst_oper', 'unit', 'geoscale',"2018"]]


# in general for us:
# total = littered + exported + collected + uncollected
# collected = recycling + energy recovery + reuse + landfill + incineration

# in general in eurostat
# Waste generation: The quantity of waste, whereby ‘waste’ means any substance or object which the holder discards or intends or is required to discard.
# Waste management: Waste management refers to the collection, transport, recovery (including sorting), and disposal of waste.
# Treatment: Treatment means recovery or disposal operations, including preparation prior to recovery or disposal.
# Recovery: Recovery means any operation whose main result is that waste serves a useful purpose. For example, by replacing other materials that would have been used for a particular function.
# Disposal: Disposal means any operation that is not recovery (personal note: I guess this is either incineration or landfill)

# specifically for env_waspac
# GEN: generated
# RCV = RCV_E_PAC + RCV_OTH + RCY: recovery (energy recovery + recovery other + recycling)
# RCY = RCY_EU_FOR + RCY_NAT + RCY_NEU: recycling (recycling in other EU member state + recycling in the EU member state + recycling outside EU)
# RPR: repair

# specifically for env_wasmun
# DSP_I: incineration
# DSP_I_RCV_E: energy recovery (in disposal)
# DSP_L_OTH: landfill and other
# GEN: generated
# PRP_REU: preparing for reuse
# RCV_E: energy recovery (in recovery)
# RCY = RCY_C_D + RCY_M: recycling (and composting and digestion, and material)
# TRT: treated
# usually DSP_I_RCV_E = RCV_E
# probably GEN - TRT = uncollected + littered
# TRT = DSP_I + DSP_L_OTH + PRP_REU + RCV_E + RCY

# so, formulas:
# littered: 0
# TODO: consdier what to do for littering, as we did for cars
# exported: RCY_EU_FOR + RCY_NEU (only RCY_NEU for EU27)
# collected: recycling + energy recovery + reuse + landfill + incineration
# uncollected: GEN - collected
# recycling: RCY - RCY_EU_FOR + RCY_NEU (only RCY - RCY_NEU for EU27)
# energy recovery: RCV_E_PAC
# reuse: RPR
# landfill: (GEN - RCV) * (DSP_L_OTH / DSP_L_OTH + DSP_I)
# incineration: (GEN - RCV) * (DSP_I / DSP_L_OTH + DSP_IY)
# note that I assume that anything that is left from GEN - RCV, it's either incineration or landfill

# mapping
dict_mapping = {"RCY_NAT" : "recycling-national",
                "RCV_E_PAC" : "energy recovery",
                "RPR": "reuse",
                "GEN" : "generated", # this one will be used to obtain landfill and incineration
                "RCV" : "recovery", # this one will be used to obtain landfill and incineration
                "RCY" : "recycling", # this one will be used to obtain recycling
                "RCY_NEU": "recycling-neu", # for EU27 it will be only NEU (and not RCY_EU_FOR)
                "RCY_EU_FOR" : "recycling-eufor"
                }

# make long format
indexes = ['freq', 'wst_oper', 'waste', 'unit', 'geoscale']
df = pd.melt(df, id_vars = indexes, var_name='year')

# select
df = df.loc[df["wst_oper"].isin(list(dict_mapping.keys())),:]
df["variable"] = np.nan
for key in dict_mapping.keys():
    df.loc[df["wst_oper"] == key,"variable"] = dict_mapping[key]

# select countries
df = df.loc[:,['geoscale', 'year', 'variable', 'wst_oper', 'waste', 'unit', 'value']]
country_list = {'AT' : "Austria", 'BE' : "Belgium", 'BG': "Bulgaria", 'HR' : "Croatia", 'CY' : "Cyprus", 
                'CZ' : "Czech Republic", 'DK' : "Denmark", 'EE' : "Estonia", 'FI' : "Finland",
                'FR' : "France", 'DE' : "Germany", 'EL' : "Greece", 'HU' : "Hungary", 'IE' : "Ireland",
                'IT' : "Italy", 'LV' : "Latvia", 'LT' : "Lithuania", 'LU' : "Luxembourg", 
                'MT' : "Malta", 'NL' : "Netherlands",
                'PL' : "Poland", 'PT' : "Portugal", 'RO' : "Romania", 'SK' : "Slovakia", 'SI' : "Slovenia", 
                'ES' : "Spain", 'SE' : "Sweden", 'UKN' : "United Kingdom"}
for c in country_list.keys():
    df.loc[df["geoscale"] == c,"geoscale"] = country_list[c]
drops = ['IS', 'LI', 'NO', 'EU27_2020']
df = df.loc[~df["geoscale"].isin(drops),:]
len(df["geoscale"].unique())

# pivot to create variables
df = df.pivot(index=['geoscale', 'waste', 'year', 'unit'], 
              columns="variable", values='value').reset_index()

# do a back fill for nan
df.fillna(0, inplace=True)
# df = df.bfill()

# make recycling as recycling - recycling-neu - recycling-eufor
df["recycling"] = df["recycling"] - df["recycling-neu"] - df["recycling-eufor"]

# make EU27
indexes = ['year', 'waste', 'unit']
countries = ['Austria', 'Belgium', 'Bulgaria', 'Cyprus', 'Czech Republic',
             'Germany', 'Denmark', 'Estonia', 'Greece', 'Spain', 'Finland',
             'France', 'Croatia', 'Hungary', 'Ireland', 'Italy', 'Lithuania',
             'Luxembourg', 'Latvia', 'Malta', 'Netherlands', 'Poland',
             'Portugal', 'Romania', 'Sweden', 'Slovenia', 'Slovakia']
df_temp = df.loc[df["geoscale"].isin(countries),:]
df_temp = df_temp.groupby(indexes, as_index=False).agg(sum)
df_temp["geoscale"] = "EU27"
df = pd.concat([df, df_temp])
indexes = ['geoscale', 'waste', 'year', 'unit']
df.sort_values(indexes, inplace=True)

# make export as df["recycling-neu"] + df["recycling-eufor"]
df["export"] = df["recycling-neu"] + df["recycling-eufor"]
df.loc[df["geoscale"] == "EU27","export"] = df.loc[df["geoscale"] == "EU27","recycling-neu"]

# keep only needed variables
df = df.loc[:,['geoscale', 'waste', 'year', 'unit', 'energy recovery','recycling','reuse', 'export', 'recovery', 'generated']]

# make recovery as the sum of recovery operations (recucling + energy recovery + reuse)
df["recovery"] = df["recycling"] + df["energy recovery"] + df["reuse"]

# check
check = df["generated"] < df["energy recovery"] + df["recycling"] + df["reuse"]
df_temp = df.loc[check,:]

# fix by replacing generated with sum of 3 components
df = df.loc[[not i for i in list(check)],:]
df_temp["generated"] = df_temp["energy recovery"] + df_temp["recycling"] + df_temp["reuse"]
df = pd.concat([df, df_temp])

# clean df_mun (to get (DSP_L_OTH / GEN) and (DSP_I / GEN) to compute landfill and incineration)
df_mun  = df_mun.loc[df_mun["wst_oper"].isin(["DSP_L_OTH","DSP_I"]),:]
indexes = ['geoscale', 'freq','wst_oper', 'unit']
df_mun = pd.melt(df_mun, id_vars = indexes, var_name='year')
df_mun = df_mun.loc[:,['geoscale', 'year', 'wst_oper', 'unit', 'value']]
for c in country_list.keys():
    df_mun.loc[df_mun["geoscale"] == c,"geoscale"] = country_list[c]
df_mun["geoscale"].unique()
drops = ['AL','BA','CH','EU27_2020','IS','ME','MK','NO','RS','TR','XK']
df_mun = df_mun.loc[~df_mun["geoscale"].isin(drops),:]
indexes = ['year', 'wst_oper', 'unit']
df_temp = df_mun.groupby(indexes, as_index=False)['value'].agg(sum)
df_temp["geoscale"] = "EU27"
df_mun = pd.concat([df_mun, df_temp])
indexes = ['geoscale', 'wst_oper', 'year', 'unit']
df_mun.sort_values(indexes, inplace=True)
df_mun["value"] = df_mun["value"] * 1000
df_mun["unit"] = "T"
df_mun = df_mun.pivot(index=['geoscale', 'year', 'unit'], columns="wst_oper", 
                      values='value').reset_index()
df_mun["incineration-coeff"] = df_mun["DSP_I"] / (df_mun["DSP_I"] + df_mun["DSP_L_OTH"])
df_mun["landfill-coeff"] = df_mun["DSP_L_OTH"] / (df_mun["DSP_I"] + df_mun["DSP_L_OTH"])
df_mun = df_mun.bfill() # fill in missing with next available

# for UK, put the values of Germany. 
# TODO: find values for UK
df_temp = df_mun.loc[df_mun["geoscale"] == "Germany",:]
df_temp["geoscale"] = "United Kingdom"
df_mun = pd.concat([df_mun, df_temp])

# compute incineration and landfill in df
df = pd.merge(df, df_mun, how="left", on=['geoscale', 'year', 'unit'])
df["incineration"] = np.round((df["generated"] - df["recovery"])*df["incineration-coeff"],0)
df["landfill"] = np.round((df["generated"] - df["recovery"])*df["landfill-coeff"],0)

# check
check = df["generated"] != df["energy recovery"] + df["recycling"] + df["incineration"] + df["landfill"] + df["reuse"]
df_temp = df.loc[check,["geoscale","waste","year","generated","recovery","energy recovery","recycling","incineration","landfill","reuse"]]
df_temp["generated"]
df_temp["energy recovery"] + df_temp["recycling"] + df_temp["incineration"] + df_temp["landfill"] + df_temp["reuse"]

# fix 
index = (df["geoscale"] == "Finland") & (df["year"] == "2021") & (df["waste"] == "W150102")
df.loc[index, "generated"] = df.loc[index,"energy recovery"] + df.loc[index,"recycling"] + \
    df.loc[index,"incineration"] + df.loc[index,"landfill"] + df.loc[index,"reuse"]

# clean
df.drop(['generated','recovery','DSP_I', 'DSP_L_OTH','incineration-coeff', 'landfill-coeff'],axis=1,inplace=True)
df = df.melt(id_vars=['geoscale', 'waste', 'year', 'unit'], var_name='waste_new',value_name='value')
df.rename(columns={'waste': 'variable', "waste_new" : "wst-oper"}, inplace=True)
df.loc[df["wst-oper"] == "energy recovery","wst-oper"] = "energy-recovery"
df.sort_values(by=['geoscale','variable',"wst-oper",'year'], inplace=True)

# fix variable name
current_name = ['W150101','W150102','W15010401', 'W150107']
new_name = ["paper-pack", "plastic-pack", "aluminium-pack", "glass-pack"]
for current, new in zip(current_name, new_name):
    df.loc[df["variable"] == current,"variable"] = new
df["variable"] = [w + "_" + v + "[t]" for w, v in zip(df["variable"], df["wst-oper"])]
df.drop(columns=["wst-oper","unit"],inplace=True)
df.sort_values(by=['geoscale','variable','year'], inplace=True)

# clean
del A, c, check, countries, country_list, df_temp, df_temp_mun, dict_mapping, drops,\
    indexes, key, df_mun, current, index, new, new_name, years_fitting, current_name

##################################
##### CONVERT TO DATA MATRIX #####
##################################

# rename
df.rename(columns={"geoscale":"Country","year":"Years"},inplace=True)

# # put nan where is 0
# df.loc[df["value"] == 0,"value"] = np.nan

# make dm
df_temp = df.copy()
df_temp = df_temp.pivot(index=["Country","Years"], columns="variable", values='value').reset_index()
dm_pack = DataMatrix.create_from_df(df_temp, 1)

# # plot
# dm_pack.filter({"Country" : ["EU27"], "Variables" : ["reuse"]}).datamatrix_plot()
# dm_pack.filter({"Country" : ["EU27"]}).datamatrix_plot()
# df_temp = dm_pack.write_df()

# clean
del df_temp, df

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

# # fill 2023 based on 2022
# dm_pack.add(np.nan, col_label=[2023], dummy=True, dim='Years')
# years_fitting = [2023]
# dm_pack = linear_fitting(dm_pack, years_fitting, min_t0=0, min_tb=0, based_on=[2022])

# # plot
# dm_pack.filter({"Country" : ["EU27"]}).datamatrix_plot() 
# df_temp = dm_pack.write_df()

# # fix jumps
# dm_domapp = fix_jumps_in_dm(dm_domapp)

# # plot
# dm_domapp.filter({"Country" : ["EU27"]}).datamatrix_plot()
# df_temp = dm_domapp.write_df()

# add missing years ots
years_missing = list(set(years_ots) - set(dm_pack.col_labels['Years']))
dm_pack.add(np.nan, col_label=years_missing, dummy=True, dim='Years')
dm_pack.sort('Years')

# fill in missing years ots with linear fitting
# I am simply putting the value of 1997 for all past years, as this works best for shares later
dm_pack = linear_fitting(dm_pack, years_ots = years_missing, min_t0=0, min_tb=0, based_on = [1997])

# # plot
# dm_pack.filter({"Country" : ["EU27"], "Variables" : ["aluminium-pack"]}).datamatrix_plot()
# dm_pack.filter({"Country" : ["EU27"]}).datamatrix_plot()
# df_temp = dm_pack.write_df()

# clean
del years_missing

####################
##### MAKE FTS #####
####################

# make function to fill in missing years fts for EU27 with linear fitting
def make_fts(dm, variable, year_start, year_end, country = "EU27", dim = "Variables", 
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
dm_pack.add(np.nan, col_label=years_fts, dummy=True, dim='Years')

# set default time window for linear trend
# assumption: best is taking longer trend possible to make predictions to 2050 (even if earlier data is generated)
baseyear_start = 1990
baseyear_end = 2023

# compute fts
dm_pack = dm_pack.flatten()
dm_pack = make_fts(dm_pack, "aluminium-pack_energy-recovery", baseyear_start, baseyear_end)
dm_pack = make_fts(dm_pack, "aluminium-pack_export", baseyear_start, baseyear_end)
dm_pack = make_fts(dm_pack, "aluminium-pack_landfill", baseyear_start, baseyear_end)
dm_pack = make_fts(dm_pack, "aluminium-pack_incineration", baseyear_start, baseyear_end)
dm_pack = make_fts(dm_pack, "aluminium-pack_recycling", baseyear_start, baseyear_end)
dm_pack = make_fts(dm_pack, "aluminium-pack_reuse", baseyear_start, baseyear_end)

dm_pack = make_fts(dm_pack, "glass-pack_energy-recovery", baseyear_start, baseyear_end)
dm_pack = make_fts(dm_pack, "glass-pack_export", baseyear_start, baseyear_end)
dm_pack = make_fts(dm_pack, "glass-pack_landfill", baseyear_start, baseyear_end)
dm_pack = make_fts(dm_pack, "glass-pack_incineration", baseyear_start, baseyear_end)
dm_pack = make_fts(dm_pack, "glass-pack_recycling", baseyear_start, baseyear_end)
dm_pack = make_fts(dm_pack, "glass-pack_reuse", baseyear_start, baseyear_end)

dm_pack = make_fts(dm_pack, "paper-pack_energy-recovery", baseyear_start, baseyear_end)
dm_pack = make_fts(dm_pack, "paper-pack_export", 2017, baseyear_end)
dm_pack = make_fts(dm_pack, "paper-pack_landfill", baseyear_start, baseyear_end)
dm_pack = make_fts(dm_pack, "paper-pack_incineration", baseyear_start, baseyear_end)
dm_pack = make_fts(dm_pack, "paper-pack_recycling", baseyear_start, baseyear_end)
dm_pack = make_fts(dm_pack, "paper-pack_reuse", baseyear_start, baseyear_end)

dm_pack = make_fts(dm_pack, "plastic-pack_energy-recovery", baseyear_start, baseyear_end)
dm_pack = make_fts(dm_pack, "plastic-pack_export", 2017, baseyear_end)
dm_pack = make_fts(dm_pack, "plastic-pack_landfill", baseyear_start, baseyear_end)
dm_pack = make_fts(dm_pack, "plastic-pack_incineration", baseyear_start, baseyear_end)
dm_pack = make_fts(dm_pack, "plastic-pack_recycling", baseyear_start, baseyear_end)
dm_pack = make_fts(dm_pack, "plastic-pack_reuse", baseyear_start, baseyear_end)

# variable = "plastic-pack_reuse"
# (make_fts(dm_pack, variable, baseyear_start, baseyear_end).
#   datamatrix_plot(selected_cols={"Country" : ["EU27"],
#                                 "Variables" : [variable]}))

# # check
# dm_pack.filter({"Country" : ["EU27"]}).datamatrix_plot()
# df_temp = dm_pack.write_df()

# deepen back
dm_pack.deepen()

# make paper-print and paper-san (same of paper-pack)
variables = ["paper-print","paper-san"]
for v in variables:
    dm_temp = dm_pack.filter({"Variables" : ["paper-pack"]})
    dm_temp.rename_col("paper-pack",v,"Variables")
    dm_pack.append(dm_temp,"Variables")
dm_pack.sort("Variables")

################################
##### MAKE FINAL VARIABLES #####
################################

# make dm for collected waste
dm_pack_col = dm_pack.filter({"Categories1" : ['energy-recovery', 'incineration', 'landfill', 'recycling', 'reuse']})
dm_pack_tot = dm_pack_col.groupby(group_cols={"collected" : ['energy-recovery', 'incineration', 
                                                                 'landfill', 'recycling', 'reuse']}, 
                                dim="Categories1", aggregation = "sum", regex=False, inplace=False)

# re-put nan in years fts for all countries but EU27
idx = dm_pack_tot.idx
countries = dm_pack_tot.col_labels["Country"]
countries = list(np.array(countries)[[i != "EU27" for i in countries]])
for c in countries:
    for y in years_fts:
            dm_pack_tot.array[idx[c],idx[y],:] = np.nan

# compute the shares rounded at 2 decimals
dm_pack_col.array = dm_pack_col.array / dm_pack_tot.array
idx = dm_pack_col.idx
for c in dm_pack_col.col_labels["Country"]:
    for y in years_all:
        for v in dm_pack_col.col_labels["Variables"]:
            dm_pack_col.array[idx[c],idx[y],idx[v],:] = adjust_shares(dm_pack_col.array[idx[c],idx[y],idx[v],:])
for v in dm_pack_col.col_labels["Variables"]: 
    dm_pack_col.units[v] = "%"

# # checks
# dm_temp = dm_pack_col.groupby(group_cols={"total" : dm_pack_col.col_labels["Categories1"]}, 
#                                 dim="Categories1", aggregation = "sum", regex=False, inplace=False)
# for c in countries:
#     for y in years_fts:
#             dm_temp.array[idx[c],idx[y],:] = np.nan
# dm_temp.array
# df_temp = dm_temp.write_df() # there are some countries with potential missing values before 2005, fine for now
# df_temp = dm_pack_col.write_df()
# df_temp = dm_pack_tot.write_df()
# dm_pack_col.filter({"Country" : ["EU27"]}).datamatrix_plot()

# the nan in ots are because of division by 0, for the moment I keep them as nan (as we'll use only EU27)
# when other countries are implemented, we should re-consider.

# make dm for total waste
dm_pack_tot.append(dm_pack.filter({"Categories1" : ["export"]}), "Categories1")
dm_pack_tot.rename_col("collected","waste-collected","Categories1")
dm_pack_tot.add(0, col_label="waste-uncollected", dummy=True, dim='Categories1', unit="t")
dm_pack_tot.add(0, col_label="littered", dummy=True, dim='Categories1', unit="t")
# TODO: for the moment I put waste-uncollected to 0, TBD with constants from the literature
idx = dm_pack_tot.idx
countries = dm_pack_tot.col_labels["Country"]
countries = list(np.array(countries)[[i != "EU27" for i in countries]])
for c in countries:
    for y in years_fts:
            dm_pack_tot.array[idx[c],idx[y],:,idx["littered"]] = np.nan

# dm_pack_tot.filter({"Country" : ["EU27"]}).datamatrix_plot()

# make shares in dm
# compute the shares rounded at 2 decimals
dm_temp = dm_pack_tot.groupby(group_cols={"total" : ['waste-collected', 'export', 'waste-uncollected', 'littered']}, 
                                dim="Categories1", aggregation = "sum", regex=False, inplace=False)
idx = dm_temp.idx
for c in countries:
    for y in years_fts:
            dm_temp.array[idx[c],idx[y],:,idx["total"]] = np.nan
dm_pack_tot.array = dm_pack_tot.array / dm_temp.array
for c in dm_pack_tot.col_labels["Country"]:
    for y in years_all:
        for v in dm_pack_tot.col_labels["Variables"]:
            dm_pack_tot.array[idx[c],idx[y],idx[v],:] = adjust_shares(dm_pack_tot.array[idx[c],idx[y],idx[v],:])
for v in dm_pack_tot.col_labels["Variables"]: 
    dm_pack_tot.units[v] = "%"

# df_temp = dm_pack_tot.write_df()
# dm_pack_tot.filter({"Country" : ["EU27"]}).datamatrix_plot()
# note: export goes to 0 because of the rounding and adjust_shares

# clean
del baseyear, baseyear_end, baseyear_start, c, countries, dm_pack, \
    dm_temp, idx, lastyear, startyear, step_fts, v, years_all, \
    years_fts, years_ots, years_setting, y

################
##### SAVE #####
################

# store
DM_wst_mgt["pack-total"] = dm_pack_tot
DM_wst_mgt["pack-col"] = dm_pack_col

# clean
del dm_pack_tot, dm_pack_col

########################################################
################### RENAME VARIABLES ###################
########################################################

# # rename elv
# variabs = ['waste-collected', 'export', 'waste-uncollected', 'littered']
# for v in variabs:
#     DM_wst_mgt["elv-total"].rename_col(v,"waste-mgt_elv_" + v,"Variables")
# DM_wst_mgt["elv-total"].deepen()
# variabs = ['energy-recovery', 'incineration', 'landfill', 'recycling', 'reuse']
# for v in variabs:
#     DM_wst_mgt["elv-col"].rename_col(v,"waste-mgt_elv_" + v,"Variables")
# DM_wst_mgt["elv-col"].deepen()

# # rename rest
# DM_wst_mgt["domapp-total"].rename_col("domapp","waste-mgt_domapp","Variables")
# DM_wst_mgt["domapp-col"].rename_col("domapp","waste-mgt_domapp","Variables")
# DM_wst_mgt["electronics-total"].rename_col("electronics","waste-mgt_electronics","Variables")
# DM_wst_mgt["electronics-col"].rename_col("electronics","waste-mgt_electronics","Variables")
# variabs = ['aluminium-pack', 'glass-pack', 'paper-pack', 'plastic-pack']
# for v in variabs:
#     DM_wst_mgt["pack-total"].rename_col(v,"waste-mgt_" + v,"Variables")
#     DM_wst_mgt["pack-col"].rename_col(v,"waste-mgt_" + v,"Variables")

# order
keys = ['elv-total','elv-col','domapp-total','domapp-col','electronics-total','electronics-col','pack-total','pack-col']
for k in keys:
    DM_wst_mgt[k].sort("Variables")
    DM_wst_mgt[k].sort("Categories1")


# keys = ['elv-total','elv-col','domapp-total','domapp-col','electronics-total','electronics-col','pack-total','pack-col']
# for k in keys:
#     df_temp = DM_wst_mgt[k].write_df()
#     df_temp = pd.melt(df_temp, id_vars = ['Country', 'Years'], var_name='variable')
#     df_temp = df_temp.loc[df_temp["Country"].isin(["Austria","France"]),:]
#     df_temp = df_temp.loc[df_temp["Years"]==1990,:]
#     name = k + "_temp.xlsx"
#     df_temp.to_excel("~/Desktop/" + name)

########################################################
##################### PUT TOGETHER #####################
########################################################

# put together
# list(DM_wst_mgt) 
dm = DM_wst_mgt["elv-total"].copy()
keys = list(DM_wst_mgt)
keys = [s for s in keys if "-total" in s]
keys.remove('elv-total')
for key in keys:
    dm.append(DM_wst_mgt[key], dim="Variables")
dm_col = DM_wst_mgt["elv-col"].copy()
keys = list(DM_wst_mgt)
keys = [s for s in keys if "-col" in s]
keys.remove('elv-col')
for key in keys:
    dm_col.append(DM_wst_mgt[key], dim="Variables")
dm.append(dm_col,"Categories1")

# set years
years_ots = list(range(1990,2023+1))
years_fts = list(range(2025,2055,5))

###############
##### OTS #####
###############

dm_ots = dm.filter({"Years" : years_ots})

#######################
##### FTS LEVEL 1 #####
#######################

# level 1: continuing as is
dm_fts_level1 = dm.filter({"Years" : years_fts})
# dm_fts_level1.filter({"Country" : ["EU27"], "Variables" : ["vehicles"]}).flatten().datamatrix_plot(stacked=True)

#######################
##### FTS LEVEL 4 #####
#######################

# TODO: for now we do it only for vehicles, to be done for others

# for layer 1 we put waste collected to zero
dm_level4_veh = dm.filter({"Variables" : ["vehicles"]})
dm_level4_tot = dm_level4_veh.filter({"Categories1" : ["export","littered","waste-collected","waste-uncollected"]})
idx = dm_level4_tot.idx
for y in range(2030,2055,5):
    dm_level4_tot.array[idx["EU27"],idx[y],:,:] = np.nan
export = dm_level4_tot.array[idx["EU27"],idx[2023],:,idx["export"]]
collected = dm_level4_tot.array[idx["EU27"],idx[2023],:,idx["waste-collected"]]
tot = export + collected
dm_level4_tot.array[idx["EU27"],idx[2050],:,idx["export"]] = np.round(export/tot,2)
dm_level4_tot.array[idx["EU27"],idx[2050],:,idx["waste-collected"]] = np.round(collected/tot,2)
dm_level4_tot.array[idx["EU27"],idx[2050],:,idx["littered"]] = 0
dm_level4_tot.array[idx["EU27"],idx[2050],:,idx["waste-uncollected"]] = 0
dm_level4_tot = linear_fitting(dm_level4_tot, years_fts)
# dm_level4_tot.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot(stacked=True)

# for layer 2
dm_level4_col = dm_level4_veh.filter({"Categories1" : ["energy-recovery","landfill","incineration","reuse","recycling"]})
idx = dm_level4_col.idx
for y in years_fts:
    dm_level4_col.array[idx["EU27"],idx[y],:,:] = np.nan
dm_level4_col.array[idx["EU27"],idx[2050],:,idx["recycling"]] = 1
dm_level4_col.array[idx["EU27"],idx[2050],:,idx["energy-recovery"]] = 0
dm_level4_col.array[idx["EU27"],idx[2050],:,idx["landfill"]] = 0
dm_level4_col.array[idx["EU27"],idx[2050],:,idx["incineration"]] = 0
dm_level4_col.array[idx["EU27"],idx[2050],:,idx["reuse"]] = 0
dm_level4_col = linear_fitting(dm_level4_col, years_fts)
# dm_level4_col.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot(stacked=True)

# substitute back in
dm_level4 = dm.copy()
dm_level4_veh = dm_level4_tot.copy()
dm_level4_veh.append(dm_level4_col,"Categories1")
dm_level4_veh.sort("Categories1")
dm_level4.drop("Variables","vehicles")
dm_level4.append(dm_level4_veh,"Variables")
dm_level4.sort("Variables")
# dm_level4.filter({"Country" : ["EU27"], "Variables" : ["vehicles"]}).flatten().datamatrix_plot(stacked=True)
dm_fts_level4 = dm_level4.filter({"Years" : years_fts})
# dm_fts_level4.filter({"Country" : ["EU27"], "Variables" : ["vehicles"]}).flatten().datamatrix_plot(stacked=True)

# get levels for 2 and 3
variabs = dm_fts_level1.col_labels["Categories1"]
df_temp = pd.DataFrame({"level" : np.tile(range(1,4+1),len(variabs)), 
                        "variable": np.repeat(variabs, 4)})
df_temp["value"] = np.nan
df_temp = df_temp.pivot(index=['level'], 
                        columns=['variable'], values="value").reset_index()
for v in variabs:
    idx = dm_fts_level1.idx
    level1 = dm_fts_level1.array[idx["EU27"],idx[2050],idx["vehicles"],idx[v]]
    idx = dm_fts_level4.idx
    level4 = dm_fts_level4.array[idx["EU27"],idx[2050],idx["vehicles"],idx[v]]
    arr = np.array([level1,np.nan,np.nan,level4])
    df_temp[v] = pd.Series(arr).interpolate().to_numpy()

#######################
##### FTS LEVEL 2 #####
#######################

dm_level2 = dm.copy()
idx = dm_level2.idx
for y in range(2030,2055,5):
    dm_level2.array[idx["EU27"],idx[y],idx["vehicles"],:] = np.nan
for v in variabs:
    dm_level2.array[idx["EU27"],idx[2050],idx["vehicles"],idx[v]] = df_temp.loc[df_temp["level"] == 2,v]
dm_level2 = linear_fitting(dm_level2, years_fts)
# dm_level2.filter({"Country" : ["EU27"], "Categories1":["vehicles"]}).flatten().datamatrix_plot()
dm_fts_level2 = dm_level2.filter({"Years" : years_fts})
# dm_fts_level2.filter({"Country" : ["EU27"], "Variables" : ["vehicles"]}).flatten().datamatrix_plot(stacked=True)

#######################
##### FTS LEVEL 3 #####
#######################

dm_level3 = dm.copy()
idx = dm_level3.idx
for y in range(2030,2055,5):
    dm_level3.array[idx["EU27"],idx[y],idx["vehicles"],:] = np.nan
for v in variabs:
    dm_level3.array[idx["EU27"],idx[2050],idx["vehicles"],idx[v]] = df_temp.loc[df_temp["level"] == 3,v]
dm_level3 = linear_fitting(dm_level3, years_fts)
# dm_level3.filter({"Country" : ["EU27"], "Categories1":["LDV"]}).flatten().datamatrix_plot()
dm_fts_level3 = dm_level3.filter({"Years" : years_fts})
# dm_fts_level3.filter({"Country" : ["EU27"], "Variables" : ["vehicles"]}).flatten().datamatrix_plot(stacked=True)


################
##### SAVE #####
################

DM_fts = {1: dm_fts_level1.copy(), 2: dm_fts_level2.copy(), 3: dm_fts_level3.copy(), 4: dm_fts_level4.copy()}
DM = {"ots" : dm_ots,
      "fts" : DM_fts}
f = '../data/datamatrix/lever_waste-management.pickle'
with open(f, 'wb') as handle:
    pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # total
# dm = DM_wst_mgt["elv-total"].copy()
# keys = ['domapp-total','electronics-total','pack-total']
# for key in keys:
#     dm.append(DM_wst_mgt[key], dim="Variables")
# years_ots = list(range(1990,2023+1))
# years_fts = list(range(2025,2055,5))
# dm_ots = dm.filter({"Years" : years_ots})
# dm_fts = dm.filter({"Years" : years_fts})
# DM_fts = {1: dm_fts, 2: dm_fts, 3: dm_fts, 4: dm_fts} # for now we set all levels to be the same
# DM = {"ots" : dm_ots,
#       "fts" : DM_fts}
# f = os.path.join(current_file_directory, '../data/datamatrix/lever_waste-management_layer1.pickle')
# with open(f, 'wb') as handle:
#     pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # collected
# dm = DM_wst_mgt["elv-col"].copy()
# keys = ['domapp-col','electronics-col','pack-col']
# for key in keys:
#     dm.append(DM_wst_mgt[key], dim="Variables")
# dm_ots = dm.filter({"Years" : years_ots})
# dm_fts = dm.filter({"Years" : years_fts})
# DM_fts = {1: dm_fts, 2: dm_fts, 3: dm_fts, 4: dm_fts} # for now we set all levels to be the same
# DM = {"ots" : dm_ots,
#       "fts" : DM_fts}
# f = os.path.join(current_file_directory, '../data/datamatrix/lever_waste-management_layer2.pickle')
# with open(f, 'wb') as handle:
#     pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)



