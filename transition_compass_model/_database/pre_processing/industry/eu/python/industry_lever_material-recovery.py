
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

# get data
filepath = '../data/literature/literature_review_material_recovery.xlsx'
df = pd.read_excel(filepath)

# name first 2 columns
df.rename(columns={"Unnamed: 0": "material", "Unnamed: 1" : "material-sub"}, inplace=True)

# melt
indexes = ["material","material-sub"]
df = pd.melt(df, id_vars = indexes, var_name='variable')

# save materials lists
material_sub_alu = ["cast-aluminium","wrought-aluminium"]
material_sub_steel = ["cast-iron","iron","steel","galvanized-steel","stainless-steel"]
material_sub_pla = ["plastics-ABS", "plastics-PP", "plastics-PA", "plastics-PBT", "plastics-PE",
                    "plastics-PMMA", "plastics-POM", "plastics-EPMD", "plastics-EPS", "plastics-PS",
                    "plastics-PU", "plastics-PUR", "plastics-PET", "plastics-PVC", 
                    "plastics-carbon-fiber-reinforced", "plastics-glass-fiber-reinforced",
                    "plastics-mixture","plastics-other"]
material_current = ['aluminium', 'ammonia', 'concrete-and-inert', 'plastics-total', 'copper', 'glass', 
                    'lime', 'paper', 'iron_&_steel', 'wood', 'HDPE', 'latex', 'paint', 'resin', 'rubber', 'fibreglass-composites',
                    'fluids-and-lubricants','refrigerant-R-134a','high-impact polystyrene','polychlorinated biphenyl']
material_current_correct_name = ['aluminium', 'ammonia', 'cement', 'chem', 'copper', 'glass', 
                                  'lime', 'paper', 'steel', 'timber', 'chem', 'chem', 'chem', 'chem', 'chem', 'chem', 
                                  'chem', 'chem', 'chem', 'chem']

def aggregate_materials(df, variable, material_current, material_current_correct_name):
    
    # get df for one variable
    df_temp = df.loc[df["variable"] == variable,:]

    # drop na in value
    df_temp = df_temp.dropna(subset=['value'])

    # rename missing material with sub material and drop sub material
    df_temp.loc[df_temp["material"].isnull(),"material"] = df_temp.loc[df_temp["material"].isnull(),"material-sub"]
    df_temp = df_temp.loc[:,["material","variable","value"]]

    # aggregate sub materials if any
    df_temp.loc[df_temp["material"].isin(material_sub_pla),"material"] = "plastics-total"
    df_temp.loc[df_temp["material"].isin(material_sub_alu),"material"] = "Aluminium"
    df_temp.loc[df_temp["material"].isin(material_sub_steel),"material"] = "iron_&_steel"
    df_temp.loc[df_temp["value"] == 0,"value"] = np.nan
    df_temp = df_temp.groupby(["material","variable"], as_index=False)['value'].agg(np.mean)

    # get df with materials of current model and change their names
    df_temp1 = df_temp.loc[df_temp["material"].isin(material_current),:]
    for i in range(0, len(material_current)):
        df_temp1.loc[df_temp1["material"] == material_current[i],"material"] = material_current_correct_name[i]
    df_temp1 = df_temp1.groupby(["material","variable"], as_index=False)['value'].agg(np.mean)
    
    # get other materials, sum them and concat with others
    df_temp2 = df_temp.loc[~df_temp["material"].isin(material_current),:]
    df_temp2 = df_temp2.groupby(["variable"], as_index=False)['value'].agg(np.mean)
    df_temp2["material"] = "other"
    df_temp = pd.concat([df_temp1, df_temp2])
    
    # return
    return df_temp

variabs = df["variable"].unique()
DF = {}
for v in variabs:
    DF[v] = aggregate_materials(df, v, 
                                material_current = material_current, 
                                material_current_correct_name = material_current_correct_name)
df_agg = pd.concat(DF.values(), ignore_index=True)

# if na put zero
df_agg.loc[df_agg["value"].isnull(),"value"] = 0

# check
df_check = df_agg.groupby(["variable"], as_index=False)['value'].agg(np.mean)

# substitue nan with zero
df_agg.loc[df_agg["value"].isnull(),"value"] = 0

# Assumptions

# trucks and buses, I will assume that they are the same of vehicles
# # Source: https://horizoneuropencpportal.eu/sites/default/files/2023-09/acea-position-paper-end-of-life-vehicles-directive-trucks-buses-2020.pdf
# Page 3:
# Industry believes that the re-use and recycling of second raw materials is important as well. In fact,
# this is already part of the business models of many vehicle manufacturers today. Throughout the 19
# years that HDVs have been outside the scope of the ELV Directive, the vehicle recycling industry
# has handled, treated and de-polluted trucks and buses in a way similar to passenger cars and thus
# basically already applies existing environmental legislation to HDVs.

# trains and mt: Table 2 https://www.sciencedirect.com/science/article/pii/S0956053X16305396?casa_token=URoJ4M0WLRAAAAAA:0TjVLbKhEiDy3Il7b9CmbTjEDelPNlZpF5SBRZWb_mvNeayULwxjW3BW_wAHfKQR-_8tioFj6_HQ#b0255

# planes: https://www.easa.europa.eu/en/document-library/research-reports/study-assessment-environmental-sustainability-status-aviation
# High recovery rates are reported by manufacturers for the case of aircraft that have been retired in the last decade. 
# For aircraft reaching their EoL during the last decade (with a high share of metals in their structure), recovery rates of 
# around 95% can be achieved by recyclers. These high recovery rates involve the use of downcycling and dealing with aircraft 
# that have a high percentage of metallic parts. Thus, for the case of recent aircraft models containing a significant 
# percentage of composite structural parts, which will enter the EoL phase in the future, new recycling technologies will 
# be required in order to keep the high reusability and recyclability rates. 
# put same recovery rates than trains (the recovery rates will be similar by material, i.e. aluminium high, composites low, etc).

df["variable"].unique()

# map to products we have in the calc (by taking the mean across products)
dict_map = {"vehicles" : ['ELV_shredding-and-dismantling_recycling-best',
                          'ELV_shredding-and-dismantling_recovery-network-lowest',
                          'ELV_shredding-and-dismantling_recovery-network-highest',
                          'ELV_dismantling-mechanical-separation-and-recycling_recycling-best',
                          'ELV_dismantling-mechanical-separation-and-recycling_recovery-network-lowest',
                          'ELV_dismantling-mechanical-separation-and-recycling_recovery-network-highest',
                          'ELV_dismantling-separation-and-dedicated-recycling_processes-recycling-best', 
                          'ELV_dismantling-separation-and-dedicated-recycling_processes-recovery-network-lowest',
                          "ELV_dismantling-separation-and-dedicated-recycling_processes-recovery-network-highest"],
            "battery-lion" : ['LIB_pyrometallurgy-smelting_lowest',
                              'LIB_pyrometallurgy-smelting_highest',
                              'LIB_pyrometallurgy_carbothermal-reduction-roasting_lowest',
                              'LIB_pyrometallurgy_carbothermal-reduction-roasting_highest',
                              'LIB_hydrometallurgy_leaching-organic_recovery-network-lowest',
                              'LIB_hydrometallurgy_leaching-organic_recovery-network-highest',
                              'LIB_hydrometallurgy_leaching-inorganic_recycling-best',
                              'LIB_hydrometallurgy_bio-leaching',
                              'LIB_hydrometallurgy_deep-eutectic-solvents'],
            "computer" : ["PC_recycling"],
            "fridge" : ["fridge_total-recovery"],
            "dishwasher" : ["dishwasher_combined-treatment"], # I take the combined treatment for now, as otherwise we would need to see how to combine the 4 we have (sometimes take average, sometimes take max, etc)
            "electronics" : ["WEEE"],
            "mt" : ["metrotram_light-dismantling", "	metrotram_deep-dismantling"],
            "train" : ["train_ICE-Diesel_light-dismantling", "train_ICE-Diesel_deep-dismantling", 
                       "train_CEV_light-dismantling", "train_CEV_deep-dismantling"],
            "plastic-pack" : ["plastic-packaging"],
            "glass-pack" : ["glass-packaging"],
            "paper-pack" : ["paper-packaging"],
            "aluminium-pack" : ["aluminium-pack_low","aluminium-pack_high"],
            "floor-area" : ["floor-area-new-residential"]}

for key in dict_map.keys():
    df_agg.loc[df_agg["variable"].isin(dict_map[key]),"variable"] = key
df_agg.loc[df_agg["value"] == 0,"value"] = np.nan
df_agg = df_agg.groupby(["variable","material"], as_index=False)['value'].agg(np.mean)
df_agg = df_agg.loc[df_agg["variable"].isin(list(dict_map.keys())),:]

# fix units
df_agg["value"] = df_agg["value"]/100

# check
df_check = df_agg.groupby(["variable"], as_index=False)['value'].agg(np.mean)

# make function to make dm
def make_dm(df):
    
    # create dm
    countries = ['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czech Republic','Denmark',
                 'EU27','Estonia','Finland','France','Germany','Greece','Hungary','Ireland','Italy',
                 'Latvia','Lithuania','Luxembourg','Malta','Netherlands','Poland','Portugal',
                 'Romania','Slovakia','Slovenia','Spain','Sweden','United Kingdom']
    years = list(range(1990,2023+1,1))
    years = years + list(range(2025, 2050+1, 5))
    variabs = list(df["variable"])
    units = list(np.repeat("%", len(variabs)))
    units_dict = dict()
    for i in range(0, len(variabs)):
        units_dict[variabs[i]] = units[i]
    index_dict = dict()
    for i in range(0, len(countries)):
        index_dict[countries[i]] = i
    for i in range(0, len(years)):
        index_dict[years[i]] = i
    for i in range(0, len(variabs)):
        index_dict[variabs[i]] = i

    dm = DataMatrix(empty=True)
    dm.col_labels = {"Country" : countries, "Years" : years, "Variables" : variabs}
    dm.units = units_dict
    dm.idx = index_dict
    dm.array = np.zeros((len(countries), len(years), len(variabs)))
    idx = dm.idx
    for i in variabs:
        dm.array[:,:,idx[i]] = df.loc[df["variable"]==i,"value"]
    # df_check = dm.write_df()

    # make nan for other than EU27 for fts
    countries_oth = np.array(countries)[[i not in "EU27" for i in countries]].tolist()
    idx = dm.idx
    years = list(range(2025, 2050+1, 5))
    for c in countries_oth:
        for y in years:
            for v in variabs:
                dm.array[idx[c],idx[y],idx[v]] = np.nan
    # df_check = dm.write_df()

    # rename
    dm.deepen()
    variabs = dm.col_labels["Variables"]
    for i in variabs:
        dm.rename_col(i, "waste-material-recovery_" + i, "Variables")
    dm.deepen(based_on="Variables")
    dm.switch_categories_order("Categories1","Categories2")

    # check
    # dm.filter({"Country" : ["EU27"]}).flatten().flatten().datamatrix_plot()

    # # drop ammonia
    # dm.drop("Categories2", ["ammonia"])
    
    # dm units
    dm.units["waste-material-recovery"] = "%"
    
    return dm

####################
##### VEHICLES #####
####################

# select only vehicles
df_elv = df_agg.loc[df_agg["variable"].isin(["vehicles","battery-lion"]),:]

# fix variables
df_elv["variable"] = [v + "_" + m for v,m in zip(df_elv["variable"],df_elv["material"])]
df_elv.drop(["material"],axis=1,inplace=True)

# as we intend batteries as battery packs, I assign the same recovery rates of the car recycling techs
# to steel and aluminium that can be in the pack
df_elv.loc[df_elv["variable"] == "battery-lion_aluminium","value"] = \
    df_elv.loc[df_elv["variable"] == "vehicles_aluminium","value"].values[0]
df_elv.loc[df_elv["variable"] == "battery-lion_steel","value"] = \
    df_elv.loc[df_elv["variable"] == "vehicles_steel","value"].values[0]

# now assign 0 to nan
df_elv.loc[df_elv["value"].isnull(),"value"] = 0

# make dm
dm_veh = make_dm(df_elv)
# df_check = dm_veh.write_df()

#################################
##### TRAINS, SHIPS, PLANES #####
#################################

# select only trains and metrotram
df_temp = df_agg.loc[df_agg["variable"].isin(["train","mt"]),:]

# for mt, put steel same of train
df_temp.loc[(df_temp["variable"] == "mt") & (df_temp["material"] == "steel"),"value"] = 0.2

# rename
df_temp.loc[df_temp["variable"] == "mt","variable"] = "metrotram"
df_temp.loc[df_temp["variable"] == "train","variable"] = "trains"
    
# fix variables
df_temp["variable"] = [v + "_" + m for v,m in zip(df_temp["variable"],df_temp["material"])]
df_temp.drop(["material"],axis=1,inplace=True)

# now assign 0 to nan
df_temp.loc[df_temp["value"].isnull(),"value"] = 0

# make dm
dm_train = make_dm(df_temp)

# add ships and planes (assumed to be the same of trains)
dm_temp = dm_train.filter({"Categories1" : ["trains"]})
dm_temp.rename_col("trains","ships","Categories1")
dm_train.append(dm_temp,"Categories1")
dm_temp.rename_col("ships","planes","Categories1")
dm_train.append(dm_temp,"Categories1")
dm_train.sort("Categories1")

####################
##### PACKAGES #####
####################

# select
df_temp = df_agg.loc[df_agg["variable"].isin(['aluminium-pack','glass-pack','paper-pack','plastic-pack']),:]

# fix variables
df_temp["variable"] = [v + "_" + m for v,m in zip(df_temp["variable"],df_temp["material"])]
df_temp.drop(["material"],axis=1,inplace=True)

# now assign 0 to nan
df_temp.loc[df_temp["value"].isnull(),"value"] = 0

# make dm
dm_pack = make_dm(df_temp)

# add paper-print and paper-san (assume to be same of paper pack)
dm_temp = dm_pack.filter({"Categories1" : ["paper-pack"]})
dm_temp.rename_col("paper-pack","paper-print","Categories1")
dm_pack.append(dm_temp,"Categories1")
dm_temp.rename_col("paper-print","paper-san","Categories1")
dm_pack.append(dm_temp,"Categories1")
dm_pack.sort("Categories1")

###############################
##### DOMESTIC APPLIANCES #####
###############################

# select
df_temp = df_agg.loc[df_agg["variable"].isin(['fridge','dishwasher']),:]

# fridge chem seem too high, assigning the ones of dishwasher for now
df_temp.loc[(df_temp["variable"] == "fridge") & (df_temp["material"] == "chem"),"value"] = 0.423
    
# fix variables
df_temp["variable"] = [v + "_" + m for v,m in zip(df_temp["variable"],df_temp["material"])]
df_temp.drop(["material"],axis=1,inplace=True)

# now assign 0 to nan
df_temp.loc[df_temp["value"].isnull(),"value"] = 0

# make dm
dm_domapp = make_dm(df_temp)

# add dryer and wmachine (as dishwasher), and freezer (as fridge)
dm_temp = dm_domapp.filter({"Categories1" : ["dishwasher"]})
dm_temp.rename_col("dishwasher","dryer","Categories1")
dm_domapp.append(dm_temp,"Categories1")
dm_temp.rename_col("dryer","wmachine","Categories1")
dm_domapp.append(dm_temp,"Categories1")
dm_temp = dm_domapp.filter({"Categories1" : ["fridge"]})
dm_temp.rename_col("fridge","freezer","Categories1")
dm_domapp.append(dm_temp,"Categories1")
dm_domapp.sort("Categories1")

#######################
##### ELECTRONICS #####
#######################

# TODO: for the moment in the weee source considered, there is no mention of batteries
# for the moment I assume that they are included, at some point we will have to
# separate them as done for cars

# select
df_temp = df_agg.loc[df_agg["variable"].isin(['electronics']),:]
    
# fix variables
df_temp["variable"] = [v + "_" + m for v,m in zip(df_temp["variable"],df_temp["material"])]
df_temp.drop(["material"],axis=1,inplace=True)

# now assign 0 to nan
df_temp.loc[df_temp["value"].isnull(),"value"] = 0

# make dm
dm_elec = make_dm(df_temp)

#####################
##### BUILDINGS #####
#####################

# select
df_temp = df_agg.loc[df_agg["variable"].isin(['floor-area']),:]
    
# fix variables
df_temp["variable"] = [v + "_" + m for v,m in zip(df_temp["variable"],df_temp["material"])]
df_temp.drop(["material"],axis=1,inplace=True)

# now assign 0 to nan
df_temp.loc[df_temp["value"].isnull(),"value"] = 0

# make dm
dm_bld = make_dm(df_temp)

##########################
##### INFRASTRUCTURE #####
##########################

# assumed to be the same of buildings

dm_infra = dm_bld.copy()
dm_infra.rename_col("floor-area", "rail", "Categories1")
dm_temp = dm_infra.filter({"Categories1" : ["rail"]})
dm_temp.rename_col("rail","road","Categories1")
dm_infra.append(dm_temp,"Categories1")
dm_temp.rename_col("road","trolley-cables","Categories1")
dm_infra.append(dm_temp,"Categories1")
dm_infra.sort("Categories1")

# for trolley cables keep only copper (for the moment i put 70%, to be rechecked)
materials = ['aluminium', 'cement', 'chem', 'glass', 'lime', 'other', 'paper', 'steel', 'timber']
for m in materials:
    dm_infra["EU27",:,:,"trolley-cables",m] = 0
dm_infra["EU27",:,:,"trolley-cables","copper"] = 0.7

# fix rail (keep only steel and timber) and road (put glass and timber to zero)
materials = ['aluminium', 'cement', 'chem', 'copper', 'glass', 'lime', 'other', 'paper']
for m in materials:
    dm_infra["EU27",:,:,"rail",m] = 0
dm_infra["EU27",:,:,"road","glass"] = 0
dm_infra["EU27",:,:,"road","timber"] = 0

########################
##### PUT TOGETHER #####
########################

dm = dm_veh.copy()
dm.append(dm_train,"Categories1")
dm.append(dm_pack,"Categories1")
dm.append(dm_domapp,"Categories1")
dm.append(dm_elec,"Categories1")
dm.append(dm_bld,"Categories1")
dm.append(dm_infra,"Categories1")
dm.sort("Categories1")

# dm.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

###############
##### OTS #####
###############

# set years
years_ots = list(range(1990,2023+1))
years_fts = list(range(2025,2055,5))

dm_ots = dm.filter({"Years" : years_ots})

#######################
##### FTS LEVEL 1 #####
#######################

# level 1: continuing as is
dm_fts_level1 = dm.filter({"Years" : years_fts})
# dm_fts_level1.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

#######################
##### FTS LEVEL 4 #####
#######################

# TODO: for the moment we put max values following own knowledge, to be re-done with literature
dm_level4 = dm.copy()
for y in range(2030,2055,5):
    dm_level4["EU27",y,:,:,:] = np.nan
dm_level4["EU27",2050,:,"battery-lion","aluminium"] = 1
dm_level4["EU27",2050,:,"battery-lion","other"] = 1
dm_level4["EU27",2050,:,"battery-lion","steel"] = 1
products = ["vehicles","dishwasher","dryer","wmachine"]
for p in products:
    dm_level4["EU27",2050,:,p,"aluminium"] = 1
    dm_level4["EU27",2050,:,p,"chem"] = 0.9
    dm_level4["EU27",2050,:,p,"copper"] = 1
    dm_level4["EU27",2050,:,p,"other"] = 0.9
    dm_level4["EU27",2050,:,p,"steel"] = 1
products = ["electronics"]
for p in products:
    dm_level4["EU27",2050,:,p,"aluminium"] = 1
    dm_level4["EU27",2050,:,p,"copper"] = 1
    dm_level4["EU27",2050,:,p,"other"] = 0.9
    dm_level4["EU27",2050,:,p,"steel"] = 1
products = ["floor-area"]
for p in products:
    dm_level4["EU27",2050,:,p,"aluminium"] = 1
    dm_level4["EU27",2050,:,p,"cement"] = 0.9
    dm_level4["EU27",2050,:,p,"chem"] = 0.9
    dm_level4["EU27",2050,:,p,"glass"] = 1
    dm_level4["EU27",2050,:,p,"other"] = 0.9
    dm_level4["EU27",2050,:,p,"steel"] = 1
    dm_level4["EU27",2050,:,p,"timber"] = 1
products = ["freezer","fridge"]
for p in products:
    dm_level4["EU27",2050,:,p,"aluminium"] = 1
    dm_level4["EU27",2050,:,p,"chem"] = 0.9
    dm_level4["EU27",2050,:,p,"copper"] = 1
dm_level4["EU27",2050,:,"aluminium-pack","aluminium"] = 1
dm_level4["EU27",2050,:,"glass-pack","glass"] = 1
dm_level4["EU27",2050,:,"paper-pack","paper"] = 1
dm_level4["EU27",2050,:,"paper-print","paper"] = 1
dm_level4["EU27",2050,:,"paper-san","paper"] = 1
dm_level4["EU27",2050,:,"plastic-pack","chem"] = 0.9
products = ["metrotram","trains","planes","ships"]
for p in products:
    dm_level4["EU27",2050,:,p,"aluminium"] = 1
    dm_level4["EU27",2050,:,p,"chem"] = 0.9
    dm_level4["EU27",2050,:,p,"glass"] = 1
    dm_level4["EU27",2050,:,p,"other"] = 0.9
    dm_level4["EU27",2050,:,p,"steel"] = 1
dm_level4["EU27",2050,:,"rail","steel"] = 1
dm_level4["EU27",2050,:,"rail","timber"] = 1
dm_level4["EU27",2050,:,"road","aluminium"] = 1
dm_level4["EU27",2050,:,"road","cement"] = 1
dm_level4["EU27",2050,:,"road","chem"] = 0.9
dm_level4["EU27",2050,:,"road","steel"] = 1
dm_level4["EU27",2050,:,"road","other"] = 0.9
dm_level4["EU27",2050,:,"trolley-cables","copper"] = 1

dm_level4 = linear_fitting(dm_level4, years_fts)
# dm_level4.filter({"Country" : ["EU27"]}).flatten().flatten().datamatrix_plot()
dm_fts_level4 = dm_level4.filter({"Years" : years_fts})
# dm_fts_level4.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

#############################
##### FTS LEVEL 2 and 3 #####
#############################

dm_level2 = dm_level4.copy()
dm_level3 = dm_level4.copy()
for y in range(2030,2050+5,5):
    dm_level2["EU27",y,:,:,:] = np.nan
    dm_level3["EU27",y,:,:,:] = np.nan

p = "aluminium-pack"
m = "aluminium"
for p in dm_fts_level1.col_labels["Categories1"]:
    for m in dm_fts_level1.col_labels["Categories2"]:
        level1 = dm_fts_level1["EU27",2050,:,p,m][0]
        level4 = dm_fts_level4["EU27",2050,:,p,m][0]
        arr = np.array([level1,np.nan,np.nan,level4])
        arr = pd.Series(arr).interpolate().to_numpy()
        level2 = np.round(arr[1],2)
        level3 = np.round(arr[2],2)
        dm_level2["EU27",2050,:,p,m] = level2
        dm_level3["EU27",2050,:,p,m] = level3

dm_level2 = linear_fitting(dm_level2, years_fts)
# dm_level2.filter({"Country" : ["EU27"]}).flatten().flatten().datamatrix_plot()
dm_fts_level2 = dm_level2.filter({"Years" : years_fts})
dm_level3 = linear_fitting(dm_level3, years_fts)
# dm_level3.filter({"Country" : ["EU27"]}).flatten().flatten().datamatrix_plot()
dm_fts_level3 = dm_level3.filter({"Years" : years_fts})

################
##### SAVE #####
################

# put together
DM_fts = {1: dm_fts_level1.copy(), 2: dm_fts_level2.copy(), 3: dm_fts_level3.copy(), 4: dm_fts_level4.copy()}
DM = {"ots" : dm_ots,
      "fts" : DM_fts}
f = '../data/datamatrix/lever_material-recovery.pickle'
with open(f, 'wb') as handle:
    pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)

# df = dm.write_df()
# df_temp = pd.melt(df, id_vars = ['Country', 'Years'], var_name='variable')
# df_temp = df_temp.loc[df_temp["Country"].isin(["Austria","France"]),:]
# df_temp = df_temp.loc[df_temp["Years"]==1990,:]
# name = "temp.xlsx"
# df_temp.to_excel("~/Desktop/" + name)



