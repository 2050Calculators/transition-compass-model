
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
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/minerals/eu/python/minerals_lever_material-recovery.py"

# directories
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# get data
filepath = os.path.join(current_file_directory, '../../../industry/eu/data/literature/literature_review_material_recovery.xlsx')
df = pd.read_excel(filepath)

# name first 2 columns
df.rename(columns={"Unnamed: 0": "material", "Unnamed: 1" : "material-sub"}, inplace=True)

# melt
indexes = ["material","material-sub"]
df = pd.melt(df, id_vars = indexes, var_name='variable')

# save materials lists
material_sub_alu = ["Cast-aluminium","Wrought-aluminium"]
material_sub_steel = ["cast-iron","iron","steel","galvanized-steel","stainless-steel"]
material_sub_pla = ["plastics-ABS", "plastics-PP", "plastics-PA", "plastics-PBT", "plastics-PE",
                    "plastics-PMMA", "plastics-POM", "plastics-EPMD", "plastics-EPS", "plastics-PS",
                    "plastics-PU", "plastics-PUR", "plastics-PET", "plastics-PVC", 
                    "plastics-carbon-fiber-reinforced", "plastics-glass-fiber-reinforced",
                    "plastics-mixture","plastics-other"]

def aggregate_materials(df, variable):
    
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
    
    # return
    return df_temp

variabs = df["variable"].unique()
df_agg = pd.concat([aggregate_materials(df, variable) 
                    for variable in variabs])

# check
df_check = df_agg.groupby(["variable"], as_index=False)['value'].agg(np.mean)

# map to products we have in the calc (by taking the mean across products)
dict_map = {"vehicles" : ['ELV_shredding-and-dismantling_recycling-best',
                          'ELV_shredding-and-dismantling_recovery-network-lowest',
                          'ELV_shredding-and-dismantling_recovery-network-highest',
                          'ELV_dismantling-mechanical-separation-and-recycling_recycling-best',
                          'ELV_dismantling-mechanical-separation-and-recycling_recovery-network-lowest',
                          'ELV_dismantling-mechanical-separation-and-recycling_recovery-network-highest'],
            "battery-lion" : ['LIB_pyrometallurgy-smelting_lowest',
                              'LIB_pyrometallurgy-smelting_highest',
                              'LIB_pyrometallurgy_carbothermal-reduction-roasting_lowest',
                              'LIB_pyrometallurgy_carbothermal-reduction-roasting_highest',
                              'LIB_hydrometallurgy_leaching-organic_recovery-network-lowest',
                              'LIB_hydrometallurgy_leaching-organic_recovery-network-highest',
                              'LIB_hydrometallurgy_leaching-inorganic_recycling-best',
                              'LIB_hydrometallurgy_bio-leaching',
                              'LIB_hydrometallurgy_deep-eutectic-solvents']}

for key in dict_map.keys():
    df_agg.loc[df_agg["variable"].isin(dict_map[key]),"variable"] = key
df_agg = df_agg.groupby(["variable","material"], as_index=False)['value'].agg(np.mean)

# check
df_check = df_agg.groupby(["variable"], as_index=False)['value'].agg(np.mean)

# fix units
df_agg["value"] = df_agg["value"]/100

# select only vehicles and batteries
df_elv = df_agg.loc[df_agg["variable"].isin(["vehicles","battery-lion"]),:]

# fix material names
materials_fix = {"Aluminium" : "aluminium", "carbon paper" : "carbon-paper",
                 "concrete-and-inert" : "cement", "iron_&_steel" : "iron-and-steel",
                 "plastics-total" : "plastics", 'silver ' : 'silver', "wood" : "timber"}
for key in materials_fix.keys():
    df_elv.loc[df_elv["material"] == key,"material"] = materials_fix[key]
df_elv.sort_values(["variable","material"],inplace=True)

# fix variables
df_elv["variable"] = [v + "_" + m for v,m in zip(df_elv["variable"],df_elv["material"])]
df_elv.drop(["material"],axis=1,inplace=True)

# put zeroes where it's missing
df_elv.loc[df_elv["value"].isnull(),"value"] = 0

# create dm
countries = ['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czech Republic','Denmark',
             'EU27','Estonia','Finland','France','Germany','Greece','Hungary','Ireland','Italy',
             'Latvia','Lithuania','Luxembourg','Malta','Netherlands','Poland','Portugal',
             'Romania','Slovakia','Slovenia','Spain','Sweden','United Kingdom']
years = list(range(1990,2023+1,1))
years = years + list(range(2025, 2050+1, 5))
variabs = list(df_elv["variable"])
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

dm = DataMatrix()
dm.col_labels = {"Country" : countries, "Years" : years, "Variables" : variabs}
dm.units = units_dict
dm.idx = index_dict
dm.array = np.zeros((len(countries), len(years), len(variabs)))
idx = dm.idx
for i in variabs:
    dm.array[:,:,idx[i]] = df_elv.loc[df_elv["variable"]==i,"value"]
df_check = dm.write_df()

# make nan for other than EU27 for fts
countries_oth = np.array(countries)[[i not in "EU27" for i in countries]].tolist()
idx = dm.idx
years = list(range(2025, 2050+1, 5))
for c in countries_oth:
    for y in years:
        for v in variabs:
            dm.array[idx[c],idx[y],idx[v]] = np.nan
df_check = dm.write_df()

# rename
dm.deepen()
variabs = dm.col_labels["Variables"]
for i in variabs:
    dm.rename_col(i, "waste-material-recovery_" + i, "Variables")
dm.deepen(based_on="Variables")
dm.switch_categories_order("Categories1","Categories2")

# drop ammonia and other
dm.drop("Categories2", ["ammonia","other"])

# save
years_ots = list(range(1990,2023+1))
years_fts = list(range(2025,2055,5))
dm_ots = dm.filter({"Years" : years_ots})
dm_fts = dm.filter({"Years" : years_fts})
DM_fts = {1: dm_fts.copy(), 2: dm_fts.copy(), 3: dm_fts.copy(), 4: dm_fts.copy()} # for now we set all levels to be the same
DM = {"ots" : dm_ots,
      "fts" : DM_fts}
f = os.path.join(current_file_directory, '../data/datamatrix/lever_material-recovery.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)

# df = dm.write_df()
# df_temp = pd.melt(df, id_vars = ['Country', 'Years'], var_name='variable')
# df_temp = df_temp.loc[df_temp["Country"].isin(["Austria","France"]),:]
# df_temp = df_temp.loc[df_temp["Years"]==1990,:]
# name = "temp.xlsx"
# df_temp.to_excel("~/Desktop/" + name)



