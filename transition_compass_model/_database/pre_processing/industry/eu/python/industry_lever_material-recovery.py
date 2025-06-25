
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
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/industry/eu/python/industry_lever_material-recovery.py"

# directories
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# get data
filepath = os.path.join(current_file_directory, '../data/literature/literature_review_material_recovery.xlsx')
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
    df_temp1 = df_temp1.groupby(["material","variable"], as_index=False)['value'].agg(sum)
    
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
                              'LIB_hydrometallurgy_deep-eutectic-solvents']}

for key in dict_map.keys():
    df_agg.loc[df_agg["variable"].isin(dict_map[key]),"variable"] = key
df_agg.loc[df_agg["value"] == 0,"value"] = np.nan
df_agg = df_agg.groupby(["variable","material"], as_index=False)['value'].agg(np.mean)

# check
df_check = df_agg.groupby(["variable"], as_index=False)['value'].agg(np.mean)

# fix units
df_agg["value"] = df_agg["value"]/100

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

# check
# dm.filter({"Country" : ["EU27"]}).flatten().flatten().datamatrix_plot()

# drop ammonia
dm.drop("Categories2", ["ammonia"])

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
dm_fts_level1 = dm.filter({"Years" : years_fts}).flatten()
# dm_fts_level1.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

#######################
##### FTS LEVEL 4 #####
#######################

# we set all to 1 but for chemicals and others
dm_level4 = dm.flatten()
idx = dm_level4.idx
for y in range(2030,2055,5):
    dm_level4.array[idx["EU27"],idx[y],:,:] = np.nan
dm_level4.array[idx["EU27"],idx[2050],:,idx["battery-lion_aluminium"]] = 1
dm_level4.array[idx["EU27"],idx[2050],:,idx["battery-lion_other"]] = 1
dm_level4.array[idx["EU27"],idx[2050],:,idx["battery-lion_steel"]] = 1
dm_level4.array[idx["EU27"],idx[2050],:,idx["vehicles_aluminium"]] = 1
dm_level4.array[idx["EU27"],idx[2050],:,idx["vehicles_chem"]] = 0.9
dm_level4.array[idx["EU27"],idx[2050],:,idx["vehicles_copper"]] = 1
dm_level4.array[idx["EU27"],idx[2050],:,idx["vehicles_other"]] = 0.9
dm_level4.array[idx["EU27"],idx[2050],:,idx["vehicles_steel"]] = 1

dm_level4 = linear_fitting(dm_level4, years_fts)
# dm_level4.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()
dm_fts_level4 = dm_level4.filter({"Years" : years_fts})
# dm_fts_level4.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

# get levels for 2 and 3
variabs = dm_fts_level1.col_labels["Categories1"]
df_temp = pd.DataFrame({"level" : np.tile(range(1,4+1),len(variabs)), 
                        "variable": np.repeat(variabs, 4)})
df_temp["value"] = np.nan
df_temp = df_temp.pivot(index=['level'], 
                        columns=['variable'], values="value").reset_index()
for v in variabs:
    idx = dm_fts_level1.idx
    level1 = dm_fts_level1.array[idx["EU27"],idx[2050],:,idx[v]][0]
    idx = dm_fts_level4.idx
    level4 = dm_fts_level4.array[idx["EU27"],idx[2050],:,idx[v]][0]
    arr = np.array([level1,np.nan,np.nan,level4])
    df_temp[v] = pd.Series(arr).interpolate().to_numpy()

#######################
##### FTS LEVEL 2 #####
#######################

dm_level2 = dm.flatten()
idx = dm_level2.idx
for y in range(2030,2055,5):
    dm_level2.array[idx["EU27"],idx[y],:,:] = np.nan
for v in variabs:
    dm_level2.array[idx["EU27"],idx[2050],:,idx[v]] = df_temp.loc[df_temp["level"] == 2,v]
dm_level2 = linear_fitting(dm_level2, years_fts)
# dm_level2.filter({"Country" : ["EU27"], "Categories1":["LDV"]}).flatten().datamatrix_plot()
dm_fts_level2 = dm_level2.filter({"Years" : years_fts})
# dm_fts_level2.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

#######################
##### FTS LEVEL 3 #####
#######################

dm_level3 = dm.flatten()
idx = dm_level3.idx
for y in range(2030,2055,5):
    dm_level3.array[idx["EU27"],idx[y],:,:] = np.nan
for v in variabs:
    dm_level3.array[idx["EU27"],idx[2050],:,idx[v]] = df_temp.loc[df_temp["level"] == 3,v]
dm_level3 = linear_fitting(dm_level3, years_fts)
# dm_level3.filter({"Country" : ["EU27"], "Categories1":["LDV"]}).flatten().datamatrix_plot()
dm_fts_level3 = dm_level3.filter({"Years" : years_fts})
# dm_fts_level3.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

################
##### SAVE #####
################

# deepen
dm_fts_level1.deepen(based_on="Categories1")
dm_fts_level2.deepen(based_on="Categories1")
dm_fts_level3.deepen(based_on="Categories1")
dm_fts_level4.deepen(based_on="Categories1")

# put together
DM_fts = {1: dm_fts_level1.copy(), 2: dm_fts_level2.copy(), 3: dm_fts_level3.copy(), 4: dm_fts_level4.copy()}
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



