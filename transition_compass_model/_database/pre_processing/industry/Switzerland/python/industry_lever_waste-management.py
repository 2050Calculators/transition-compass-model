
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

###########################################################################################
########################################## WASTE ##########################################
###########################################################################################

# import requests
# import re

# def get_organization_names(search_string):
    
#     url = 'https://ckan.opendata.swiss/api/3/action/organization_list'
#     response_structure = requests.get(url)
#     data_structure = response_structure.json()
#     org_list = data_structure["result"]
#     bool_idx = [bool(re.search(search_string,s)) for s in org_list]
    
#     return np.array(org_list)[bool_idx].tolist()
    
# def get_databases_names_by_organization(organization_id):

#     base_url = "https://ckan.opendata.swiss/api/3/action/organization_show"
#     url = f"{base_url}?id={organization_id}&include_datasets=True"
#     response_structure = requests.get(url)
#     data_structure = response_structure.json()
    
#     packages = data_structure["result"]["packages"]
#     return [packages[idx]["title"]["fr"] for idx in range(len(packages))]

# # get name of bazg
# get_organization_names("bafu")

# # get all databases of bazg
# mylist = get_databases_names_by_organization("bundesamt-fur-umwelt-bafu")
# np.array(mylist)[[bool(re.search("chet",m)) for m in mylist]]

# note: it seems nor here nor online there is a database (excel) on waste

# note: I will assume that in switzerland all incineration is going for energy recovery
# https://opendata.swiss/en/dataset/b4fa710a-136e-476c-aa71-58bb7f89bea9?

####################
##### VEHICLES #####
####################

# source on website: https://www.bafu.admin.ch/bafu/en/home/topics/waste/guide-to-waste-a-z/end-of-life-vehicles.html
# "Approximately 200,000 vehicles are withdrawn from circulation every year in Switzerland. 
# A large proportion of them are completely roadworthy or only have minor damage or faults. 
# These vehicles are classified as second-hand articles. They are often exported and sold on used car markets abroad."
# so exports should be quite large (including the second hand), and within collected
# re-use could be set of being low (as most of the re-use are exported). Though in our case re-use also include
# re-use of parts, so possibly a value that is similar to EU27 might be good.

# Other source: https://files.designer.hoststar.ch/68/86/68867f3a-1338-425f-a1e3-ec7dd88eb254.pdf

# The number of vehicles processed in Swiss shredder plants continues to decline. Only a little over
# 37,000 end-of-life vehicles were recycled in the year
# under review – ten years ago the figure was as high
# as 100,000. The fall can be attributed both to the
# increasing age of the vehicles, as they are recycled
# later, and to the higher share of exports compared
# with before.

# While the number of recycled vehicles is falling, the
# declared quantity of automobile shredder residue
# (ASR) is actually rising and now stands at 41,000
# tonnes, 22.4 per cent of which comes from end-oflife vehicles. ASR is processed by a thermal technique
# in waste incineration plants only, with 91 per cent
# of incinerations taking place in Swiss plants. Overall,
# ASR only accounts for one per cent of total incinerated waste. The recycling process does not end with
# incineration, however: further valuable metals are
# recovered from the filter dust and slag. At the same
# time, the waste heat from the flue gases is used to
# generate electricity and district heating.

# TAKEN OFFROAD: all vehicles that exit stock
# VEHICLES CANCELLED IN SWITZERLAND: all vehicles that become waste
# VEHICLES SHREDDED IN SWITZERLAND: all vehicles that are shredded
# difference between 2 above: An unknown number of deregistered vehicles are parked in garages, second-hand dealers and scrap yards

# layer 1: TAKEN OFFROAD should be total, then VEHICLES CANCELLED should be collected, then uncollected like EU27, 
# then rest is export (littered zero)
# layer 2: SHREDDED IN SWITZERLAND is all incineration (if we assume that a car weights 1.2 tonnes, then shredded 2024 is
# 44.4 tonnes, and data on incineration in 2024 is 41 tonnes).
# For recycling there is something but it must be little, so remaining of difference between cancelled and shredded can be assigned
# to either second hand or landfil (with numbers similar to EU, or can use Déchets spéciaux suisses)

# make dm
filepath = "../data/waste/vehicles/vehicle_statistics.xlsx"
df = pd.read_excel(filepath)
df = df.loc[:,["Year","Taken Offroad","Vehicles Cancelled in Switzerland",
               "Vehicles Shredded in Switzerland","Difference Cancelled vs Shredded"]]
df.columns = ["Years","waste-tot[num]","waste-collected[num]","energy-recovery[num]","layer2-else[num]"]
df["Country"] = "Switzerland"
dm = DataMatrix.create_from_df(df, 0)

# make layer 1
dm_layer1 = dm.filter({"Variables" : ['waste-tot','waste-collected']})
dm_layer1.operation("waste-collected", "/", "waste-tot", "Variables", "waste-collected-share","%")
df_temp = dm_layer1.write_df()

# make uncollected
filepath = os.path.join(current_file_directory,  '../../eu/data/datamatrix/lever_waste-management.pickle')
with open(filepath, 'rb') as handle:
    DM_eu = pickle.load(handle)
dm_eu = DM_eu["ots"].filter({"Variables" : ["vehicles"]})
years_selection = dm_layer1.col_labels["Years"].copy()
years_selection.remove(2024)
dm_temp = dm_eu.filter({"Country" : ["EU27"], "Years" : years_selection, "Categories1" : ["waste-uncollected"]})
dm_temp = dm_temp.flatten()
dm_temp.add(dm_temp[:,2023,...], "Years", [2024], dummy=True)
dm_temp.sort("Years")
dm_layer1.add(dm_temp.array, "Variables", "waste-uncollected-share", "%")
df_temp = dm_layer1.write_df()

# make export
arr_temp = 1 - dm_layer1[...,"waste-collected-share"] - dm_layer1[...,"waste-uncollected-share"]
dm_layer1.add(arr_temp, "Variables","export-share","%")
df_temp = dm_layer1.write_df()

# clean layer 1
dm_layer1.drop("Variables", ['waste-collected', 'waste-tot'])
dm_layer1.rename_col_regex("-share", "", "Variables")
dm_layer1.add(0,"Variables","littered","%",True)

# make layer 2
dm_layer2 = dm.filter({"Variables" : ['waste-collected','energy-recovery', 'layer2-else']})
dm_layer2.operation("energy-recovery", "/", "waste-collected", "Variables", "energy-recovery-share", "%")
dm_temp = dm_layer2.filter({"Variables" : ["energy-recovery-share"]})
dm_temp.array[dm_temp.array>1] = np.nan
dm_layer2.drop("Variables","energy-recovery-share")
dm_layer2.append(dm_temp,"Variables")
dm_layer2.add(0, "Variables", "incineration-share", "%", True)
dm_layer2 = linear_fitting(dm_layer2,dm_layer2.col_labels["Years"])
df_temp = dm_layer2.write_df()

# get data dechet speciaux
filepath = "../data/waste/2_Statistique des déchets spéciaux/ALL.xlsx"
df = pd.read_excel(filepath)
def subset_with_key_word(df, word):
    myindex = [bool(re.search(word, s, re.IGNORECASE)) for s in df["Sorte de déchet"]]
    return df.loc[myindex,:]
df = subset_with_key_word(df, "batteries")
df = df.loc[df["type"] == "traités sur le territoire national",:]
df = df.loc[:,["year","Total"]]
df = pd.concat([df, pd.DataFrame({"year" : [2013,2024], "Total" : [5, 0]})])
df.sort_values(["year"],inplace=True)
dm_temp = dm_layer2.filter({"Variables" : ['waste-collected']})
dm_temp.array = dm_temp.array * 1.2
dm_temp.units["waste-collected"] = "t"
np.array(df["Total"])[np.newaxis,:,np.newaxis] / dm_temp.array
# ok so almost zero recycled batteries so far, so we put zero recycled
dm_layer2.add(0, "Variables", "recycling-share", "%", True)

# for reuse and landfill, use same percentages than eu27
years_selection = dm_layer1.col_labels["Years"].copy()
years_selection.remove(2024)
dm_temp = dm_eu.filter({"Country" : ["EU27"], "Years" : years_selection, "Categories1" : ['landfill','reuse']})
dm_temp = dm_temp.flatten()
dm_temp.rename_col(['vehicles_landfill', 'vehicles_reuse'],['landfill-share', 'reuse-share'],"Variables")
dm_temp.add(dm_temp[:,2023,...], "Years", [2024], dummy=True)
dm_temp.sort("Years")
dm_temp.rename_col(["EU27"],["Switzerland"], "Country")
dm_temp.append(dm_layer2.filter({"Variables" : ['energy-recovery-share', 'incineration-share', 'recycling-share']}), "Variables")
arr_temp = 1 - dm_temp[...,"energy-recovery-share"] - dm_temp[...,"incineration-share"] - dm_temp[...,'recycling-share']
dm_temp.add(arr_temp, "Variables", "else-share", "%")
dm_temp = dm_temp.filter({"Variables" : ['landfill-share', 'reuse-share',"else-share"]})
df_temp = dm_temp.write_df()
mysum = dm_temp[...,"landfill-share"] + dm_temp[...,"reuse-share"]
arr_temp = dm_temp[...,"landfill-share"] * dm_temp[...,"else-share"] / mysum
dm_temp.drop("Variables","landfill-share")
dm_temp.add(arr_temp, "Variables", "landfill-share", "%")
arr_temp = dm_temp[...,"reuse-share"] * dm_temp[...,"else-share"] / mysum
dm_temp.drop("Variables","reuse-share")
dm_temp.add(arr_temp, "Variables", "reuse-share", "%")
df_temp = dm_temp.write_df()
dm_layer2.append(dm_temp.filter({"Variables" : ["landfill-share","reuse-share"]}), "Variables")
dm_layer2.drop("Variables", ['energy-recovery', 'layer2-else', 'waste-collected'])
dm_layer2.rename_col_regex("-share", "", "Variables")
np.sum(dm_layer2.array,2)

# put together
dm_waste = dm_layer1.copy()
variables = dm_waste.col_labels["Variables"]
for v in variables:
    dm_waste.rename_col(v,"vehicles_" + v,"Variables")
dm_waste.deepen()
variables = dm_layer2.col_labels["Variables"]
for v in variables:
    dm_layer2.rename_col(v,"vehicles_" + v,"Variables")
dm_layer2.deepen()
dm_waste.append(dm_layer2,"Categories1")
dm_waste.sort("Categories1")

# make ots and fts
years_ots = list(range(1990,2024+1,1))
years_fts = list(range(2025,2050+5,5))
dm_waste = linear_fitting(dm_waste,years_ots, based_on=[2012])
dm_waste = linear_fitting(dm_waste,years_fts, based_on=[2024])
# dm_waste.datamatrix_plot()
dm_waste.drop("Years",[2024])

############################
##### TRUCKS AND BUSES #####
############################

# same of vehicles

################################
##### TRAINS AND METROTRAM #####
################################

# same of EU27
dm_eu = DM_eu["ots"].filter({"Country" : ["EU27"], "Variables" : ["trains"]})
dm_eu.append(DM_eu["fts"][1].filter({"Country" : ["EU27"], "Variables" : ["trains"]}), "Years")
dm_eu.rename_col("EU27","Switzerland","Country")
dm_waste.append(dm_eu, "Variables")

##################
##### PLANES #####
##################

# same of EU27
dm_eu = DM_eu["ots"].filter({"Country" : ["EU27"], "Variables" : ["planes"]})
dm_eu.append(DM_eu["fts"][1].filter({"Country" : ["EU27"], "Variables" : ["planes"]}), "Years")
dm_eu.rename_col("EU27","Switzerland","Country")
dm_waste.append(dm_eu, "Variables")

#################
##### SHIPS #####
#################

# source: https://shipbreakingplatform.org/platform-publishes-list-2024/
# in 2024, 100% of swiss ships being dismantled were dismantled in India or other countries
# so we will put export 100% and rest to zero

dm_eu = DM_eu["ots"].filter({"Country" : ["EU27"], "Variables" : ["ships"]})
dm_eu.append(DM_eu["fts"][1].filter({"Country" : ["EU27"], "Variables" : ["ships"]}), "Years")
dm_eu.rename_col("EU27","Switzerland","Country")
dm_eu[...,"export"] = 1
variabs = ['littered', 'waste-collected', 'waste-uncollected', 'energy-recovery', 'incineration', 'landfill', 'recycling', 'reuse']
for v in variabs:
    dm_eu[...,v] = 0
dm_waste.append(dm_eu, "Variables")

#####################
##### BUILDINGS #####
#####################

# yearly pdfs on déchets spéciaux (probably Déchets minéraux or Déchets de chantier non triés problématiques)
# layer 1 should be collected or exported 
# layer 2 from pdf

# note: the class "Déchets de chantier non triés problématiques" is only about polluted waste from construction sites
# not sure how representative can be of the eol of a demolished building.

def extract_waste_data_from_dechets_speciaux(word, variable):

    filepath = "../data/waste/2_Statistique des déchets spéciaux/ALL.xlsx"
    df = pd.read_excel(filepath)
    df = subset_with_key_word(df, word)
    df = df.loc[df["type"].isin(["traités sur le territoire national","exportation"]),:]
    df = df.dropna()
    df.columns = ['Years', 'variable', 'Sorte de déchet', 'total', 'landfill', 'energy-recovery','treated-chem-bio', 'recycling']
    df = df.loc[:,df.columns != 'Sorte de déchet']
    df.loc[df["variable"] == "traités sur le territoire national","variable"] = "domestic"
    df.loc[df["variable"] == "exportation","variable"] = "export"
    df = pd.melt(df, id_vars = ["Years","variable"], var_name='type')
    df["variable"] = df["variable"] + "_" + df["type"] + "[t]"
    df = df.loc[:,["Years","variable","value"]]
    df = df.pivot(index=["Years"], columns="variable", values='value').reset_index()
    df["Country"] = "Switzerland"
    dm = DataMatrix.create_from_df(df, 1)
    dm_temp = dm.filter({"Variables" : ["export"], "Categories1" : ["total"]})
    dm_temp.rename_col("export",variable,"Variables")
    dm_temp.rename_col("total",'export',"Categories1")
    dm.drop("Variables","export")
    dm.rename_col("domestic",variable,"Variables")
    dm.append(dm_temp,"Categories1")
    dm.groupby({"landfill" : ["landfill","treated-chem-bio"]}, "Categories1",inplace=True)
    
    return dm

dm_bld = extract_waste_data_from_dechets_speciaux("Déchets de chantier non triés problématiques", 'floor-area-new-residential')

# make layer1
dm_layer1 = dm_bld.filter({"Categories1" : ['total','export']})
dm_layer1.operation("total", "+", "export", "Categories1", "waste-collected", "t")
dm_layer1.add(0, "Categories1", "waste-uncollected", "t", True)
dm_layer1.add(0, "Categories1", "littered", "t", True)
dm_layer1.drop("Categories1","total")
dm_layer1.normalise("Categories1")
df_temp = dm_layer1.write_df()
df_temp2 = DM_eu["ots"].filter({"Country" : ["EU27"], "Years" : dm_layer1.col_labels["Years"], 
                                "Variables" : ["floor-area-new-residential"], 
                                "Categories1" : dm_layer1.col_labels["Categories1"]}).write_df()
# seems ok

# make layer2
dm_layer2 = dm_bld.filter({"Categories1" : ['energy-recovery','landfill','recycling']})
dm_layer2.add(0, "Categories1", "reuse", "t", True)
dm_layer2.add(0, "Categories1", "incineration", "t", True)
dm_layer2.normalise("Categories1")
df_temp = dm_layer2.write_df()
df_temp2 = DM_eu["ots"].filter({"Country" : ["EU27"], "Years" : dm_layer2.col_labels["Years"], 
                                "Variables" : ["floor-area-new-residential"], 
                                "Categories1" : dm_layer2.col_labels["Categories1"]}).write_df()
# seems ok

# put together
dm_waste_bld = dm_layer1.copy()
dm_waste_bld.append(dm_layer2,"Categories1")
dm_waste_bld.sort("Categories1")

# make ots and fts
years_ots = list(range(1990,2023+1,1))
years_fts = list(range(2025,2050+5,5))
dm_waste_bld = linear_fitting(dm_waste_bld,years_ots, based_on=[2014])
dm_waste_bld = linear_fitting(dm_waste_bld,years_fts, based_on=[2023])
# dm_waste_bld.datamatrix_plot()

# put together with overall waste
arr_temp = dm_waste_bld.array
dm_waste_bld.add(arr_temp, "Variables", "floor-area-new-non-residential", "%")
dm_waste.append(dm_waste_bld, "Variables")

#################
##### ROADS #####
#################

# either like buildings or Matériaux bitumineux de démolition des routes > 20'000 mg/kg HAP
# note: this is polluted waste from road demolition, so possibly the numbers will be a bit different for overall
# waste from road demolition. We consider it as an approximation (probably higher bound)

dm_roads = extract_waste_data_from_dechets_speciaux("Matériaux bitumineux de démolition des routes >","road")
df_temp = dm_roads.write_df()

# make layer1
dm_layer1 = dm_roads.filter({"Categories1" : ['total','export']})
dm_layer1.operation("total", "+", "export", "Categories1", "waste-collected", "t")
dm_layer1.add(0, "Categories1", "waste-uncollected", "t", True)
dm_layer1.add(0, "Categories1", "littered", "t", True)
dm_layer1.drop("Categories1","total")
dm_layer1.normalise("Categories1")
df_temp = dm_layer1.write_df()
df_temp2 = DM_eu["ots"].filter({"Country" : ["EU27"], "Years" : dm_layer1.col_labels["Years"], 
                                "Variables" : ["road"], 
                                "Categories1" : dm_layer1.col_labels["Categories1"]}).write_df()
# seems ok

# make layer2
dm_layer2 = dm_roads.filter({"Categories1" : ['energy-recovery','landfill','recycling']})
dm_layer2.add(0, "Categories1", "reuse", "t", True)
dm_layer2.add(0, "Categories1", "incineration", "t", True)
dm_layer2.normalise("Categories1")
df_temp = dm_layer2.write_df()
df_temp2 = DM_eu["ots"].filter({"Country" : ["EU27"], "Years" : dm_layer2.col_labels["Years"], 
                                "Variables" : ["road"], 
                                "Categories1" : dm_layer2.col_labels["Categories1"]}).write_df()
# this one seems a lot towards landfilling and little recycling

# so I will take the EU values for roads
dm_eu = DM_eu["ots"].filter({"Country" : ["EU27"], "Variables" : ["road"]})
dm_eu.append(DM_eu["fts"][1].filter({"Country" : ["EU27"], "Variables" : ["road"]}), "Years")
dm_eu.rename_col("EU27","Switzerland","Country")
dm_waste.append(dm_eu, "Variables")

################
##### RAIL #####
################

# assumed like EU
dm_eu = DM_eu["ots"].filter({"Country" : ["EU27"], "Variables" : ["rail"]})
dm_eu.append(DM_eu["fts"][1].filter({"Country" : ["EU27"], "Variables" : ["rail"]}), "Years")
dm_eu.rename_col("EU27","Switzerland","Country")
dm_waste.append(dm_eu, "Variables")

##########################
##### TROLLEY CABLES #####
##########################

# assumed like EU
dm_eu = DM_eu["ots"].filter({"Country" : ["EU27"], "Variables" : ["trolley-cables"]})
dm_eu.append(DM_eu["fts"][1].filter({"Country" : ["EU27"], "Variables" : ["trolley-cables"]}), "Years")
dm_eu.rename_col("EU27","Switzerland","Country")
dm_waste.append(dm_eu, "Variables")

###################################################
##### LARGER APPLIANCES, AND PC & ELECTRONICS #####
###################################################

# electronics
# from here: https://www.bafu.admin.ch/bafu/en/home/topics/waste/guide-to-waste-a-z/electrical-and-electronic-equipment.html
# "The dismantling and separation of equipment into fractions is mainly carried out in Switzerland. 
# The other processing stages are often carried out abroad because non-ferrous metals processing systems, in particular, are not available in Switzerland."

# so probably when it comes to the recycling of aluminium and copper (non-ferrous metal), I will have to put a zero
# everywhere for Switzerland. To be done in the run.
# however, documents on aluminium packaging are saying that that's recycled (recyclage_des_emballagespourboissonsen2014)
# maybe recycling of standard aluminium is not currently done, but packaging yes ... to be understood

# there is some steel in electronics and appliances, so for those two you can probably put the same of Déchets métalliques

# "Consumers, in turn, are obliged to return equipment. The disposal of used equipment through municipal solid waste or 
# bulk waste collections is prohibited. These regulations are contained in the Ordinance on the Return, 
# Taking Back and Disposal of Electrical and Electronic Equipment (ORDEE). 
# The following categories of electrically operated equipment are regulated by the ORDEE:
# Electronic entertainment equipment
# Office, information, communications technology equipment
# Refrigeration equipment
# Household equipment
# Tools (excluding large-scale, stationary industrial tools)
# Sport and leisure equipment and toys
# Luminaries and lighting control equipment

# Note that from 0_Déchets 2023  Quantités produites et recyclées
# you know how much is recycled, i.e. 132’100 t in 2023

# Ok so summary:
# main report is this: https://www.swico.ch/media/filer_public/4b/cf/4bcf42d4-60f6-4fd0-8c3c-4dbcef4e8475/220613-se-fachbericht-en-rz.pdf
# numbers of overall waste are generally aligned with Déchets Quantités produites et recyclées, under Appareils électriques et électroniques
# Figure 1 of the report says that around 95% of electronic waste is collected, and about 75% is recycled.
# Let's see how these numbers compare to eu

# electronics
dm_eu = DM_eu["ots"].filter({"Country" : ["EU27"], "Variables" : ["electronics"]})
dm_eu.append(DM_eu["fts"][1].filter({"Country" : ["EU27"], "Variables" : ["electronics"]}), "Years")
dm_eu.rename_col("EU27","Switzerland","Country")
df_temp = dm_eu.filter({"Categories1" : ["waste-collected","waste-uncollected","export","littered"]}).write_df()
# all years: collected 0.8, uncollected 0.2, can be changed to 0.95 and 0.05
df_temp = dm_eu.filter({"Categories1" : ["recycling","incineration","energy-recovery","reuse","landfill"]}).write_df()
# in 2023: 87% recucling, 5% energy recovery, 5% landfilling, 3% reuse (not too far from the 75% - rest from the graph)

# fixes
dm_eu[...,"waste-collected"] = 0.95
dm_eu[...,"waste-uncollected"] = 0.05
dm_eu[...,"export"] = 0 # assuming export zero: The export and import of such waste requires the authorisation of the FOEN. Export to states that are not members of the OECD or EU is prohibited.
dm_eu[...,"littered"] = 0

# put together
dm_waste.append(dm_eu, "Variables")

# appliances
dm_eu = DM_eu["ots"].filter({"Country" : ["EU27"], "Variables" : ["domapp"]})
dm_eu.append(DM_eu["fts"][1].filter({"Country" : ["EU27"], "Variables" : ["domapp"]}), "Years")
dm_eu.rename_col("EU27","Switzerland","Country")
df_temp = dm_eu.filter({"Categories1" : ["waste-collected","waste-uncollected","export","littered"]}).write_df()
# same than electronics
df_temp = dm_eu.filter({"Categories1" : ["recycling","incineration","energy-recovery","reuse","landfill"]}).write_df()
# 2023: similar to electronics, so not too far from the 75% - rest from the graph

# fixes
dm_eu[...,"waste-collected"] = 0.95
dm_eu[...,"waste-uncollected"] = 0.05
dm_eu[...,"export"] = 0 # assuming export zero: The export and import of such waste requires the authorisation of the FOEN. Export to states that are not members of the OECD or EU is prohibited.
dm_eu[...,"littered"] = 0

# put together
dm_waste.append(dm_eu, "Variables")

    
##########################
##### GLASS PACKAGES #####
##########################

# for glass: 
# https://aureverre.ch/faits-et-chiffres
# https://www.vetroswiss.ch/fr/vetroswiss/rapport-annuel/

# Parmi ces bouteilles en verre, 44% sont produites en Suisse, tandis que les 56% restantes sont importés

# Sur l’ensemble du verre collecté en 2020, 
# environ 64% ont été exportés, 
# 24% recyclés (voir encadré pour la définition du taux de recyclage), 
# 9% décyclés en sable de verre, 
# 2% a été incinéré 
# et seulement 0.6% a été réutilisé

# so the import-export can be relevant, as for example from Recyclage des emballages pour boissons
# we see that in 2023 Quantité consommée of verre is 294’737 tonnes, and recycled is 295’753 tonnes
# but so there must be quite some import (and potentially export, judging from the figures of reports
# above). For the time dimension, you can assume it's been like this since the 90's.

# for other packaging: reuse will be set to zero, for export of waste of alu, paper and plastic
# find data with chat. Then, for layer 1, you can probably take eu data (to see if we have littered
# there, but i think so), and adapt the export. For layer 2, you can take recycling from je-f-02.03.02.11
# and set rest to energy-recovery

dm = DM_eu["ots"].filter({"Country" : ["EU27"], "Variables" : ['glass-pack']})
dm.append(DM_eu["fts"][1].filter({"Country" : ["EU27"], "Variables" : ['glass-pack']}), "Years")
dm.rename_col("EU27","Switzerland","Country")

dm[...,"export"] = 0.64
dm[...,"waste-collected"] = 1 - 0.64
dm[...,"waste-uncollected"] = 0
dm[...,"littered"] = 0

tot = 0.24+0.09+0.02+0.006
dm[...,"recycling"] = (0.24+0.09)/tot
dm[...,"energy-recovery"] = (0.02)/tot
dm[...,"reuse"] = (0.006)/tot
dm[...,"incineration"] = 0
dm[...,"landfill"] = 0

dm_waste.append(dm, "Variables")

############################
##### PLASTIC PACKAGES #####
############################

# plastic pack

# for plastic packaging waste management, we will need to see if to consider the PET data (which is only one share of plastic)
# or rather info from https://www.bafu.admin.ch/bafu/en/home/topics/waste/guide-to-waste-a-z/plastics.html:
# "Around 790,000 tonnes of plastic waste are generated every year, almost half of which is used for less 
# than a year, e.g. as packaging. Around 83% per cent (660,000 tonnes) of plastic waste is recovered for 
# energy in waste incineration plants and around 2% (10,000 tonnes) in cement works. 
# Around nine per cent (70,000 tonnes) is processed into recycled material. 
# A further six per cent (50,000 tonnes) of plastic waste is reused, for example textiles."

# So basically only 9% is recycled (while around 80% of PET is recycled, so most of those 9% will be PET)
# And we could say that around 790,000/2 is the packaging, and 790,000/2/population can be the
# packaging per capita number (considering only PET would underestimate it)

# https://plasticrecycler.ch/wp-content/uploads/2025/07/250701_Monitoringbericht_2024_FR_final.pdf
# Figure 5: we export all of it for tri, then re-import for recycling and energy recovery

# littering: https://www.news.admin.ch/en/nsb?id=75798
# https://www.empa.ch/web/s604/mikroplastik-bafu
# 5000 tons of plastic released into the environment every year
# Overall, around 5,120 tons of the seven types of plastic are discharged into the environment each year. This is around 0.7% of the total amount of the seven plastics consumed in Switzerland each year (amounting to a total of around 710,000 tons). According to Empa’s modelling, around 4,400 tons of macroplastic are deposited on soils every year.

dm = DM_eu["ots"].filter({"Country" : ["EU27"], "Variables" : ['plastic-pack']})
dm.append(DM_eu["fts"][1].filter({"Country" : ["EU27"], "Variables" : ['plastic-pack']}), "Years")
dm.rename_col("EU27","Switzerland","Country")
# dm.add(0, "Years", [2024], "%", dummy=True)
dm.array[...] = np.nan
# dm.units['plastic-pack'] = "t"

export = 0.3 # (11678-2507-3460)/11678 = 0.49 from the plastic recycler report above seems high, so we put 30%
littered = 0.007
collected = 1 - export - littered
uncollected = 0 # we assume that in CH the uncollected is in littered, and that there are no communes without collection service
dm[...,"export"] = export
dm[...,"littered"] = littered
dm[...,"waste-collected"] = collected
dm[...,"waste-uncollected"] = uncollected

dm[...,"energy-recovery"] = 0.85
dm[...,"recycling"] = 0.09
dm[...,"landfill"] = 0
dm[...,"reuse"] = 0.06
dm[...,"incineration"] = 0

# put together
dm_waste.append(dm, "Variables")

# # 2024
# factor_for_littered = 0.03 # as the report covers only around 3% of PET / plastic waste
# layer1_dict = {"littered" : 5000*factor_for_littered, "export" : 11678-2507-3460, 
#                "waste-collected" : 11695, "waste-uncollected" : 0}
# layer2_dict = {"energy-recovery" : 199+3460, "recycling" : 2315, 
#                "incineration" : 0, "reuse" : 0, "landfill" : 0}
# total_layer1 = layer1_dict["export"] + layer1_dict["waste-collected"] + layer1_dict["waste-uncollected"] + layer1_dict["littered"]
# total_layer2 = layer2_dict["energy-recovery"] + layer2_dict["recycling"] + layer2_dict["incineration"] + layer2_dict["reuse"] + layer2_dict["landfill"]
# for key in layer1_dict.keys(): dm[:,2024,:,key] = layer1_dict[key]/total_layer1
# for key in layer2_dict.keys(): dm[:,2024,:,key] = layer2_dict[key]/total_layer2

# # get time series
# dm = linear_fitting(dm, years_ots)
# # df_temp = dm.write_df()
# # dm.flatten().datamatrix_plot()

# alternative with total time trends
# =============================================================================
# # 2022
# # https://plasticrecycler.ch/wp-content/uploads/2024/07/230629_Monitoringbericht_2022_final_FR.pdf
# layer1_dict = {"littered" : 3800, # I set 3500 to have a constant trend with linear fitting
#                "export" : 9525-2431-2760, 
#                "waste-collected" : 9553, "waste-uncollected" : 0}
# layer2_dict = {"energy-recovery" : 193+2760, "recycling" : 2069, 
#                "incineration" : 0, "reuse" : 0, "landfill" : 0}
# total_layer1 = layer1_dict["export"] + layer1_dict["waste-collected"] + layer1_dict["waste-uncollected"] + layer1_dict["littered"]
# total_layer2 = layer2_dict["energy-recovery"] + layer2_dict["recycling"] + layer2_dict["incineration"] + layer2_dict["reuse"] + layer2_dict["landfill"]
# for key in layer1_dict.keys(): dm[:,2022,:,key] = layer1_dict[key]/total_layer1
# for key in layer2_dict.keys(): dm[:,2022,:,key] = layer2_dict[key]/total_layer2
# 
# # get pet through time for making trends
# filepath = os.path.join(current_file_directory, '../data/waste/0_Déchets Quantités produites et recyclées/je-f-02.03.02.11.xlsx')
# df = pd.read_excel(filepath)
# df.columns = df.iloc[3,:]
# df = df.loc[df["Matériaux "].isin(["PET "]),:]
# df = pd.melt(df, id_vars = ["Matériaux ",'Unité'], var_name='year')
# df.loc[df["value"] == '…',"value"] = np.nan
# df["value"] = df["value"].astype(float)
# val_2022 = df.loc[df["year"] == 2022,"value"].values
# df["change"] = (df["value"] - val_2022)/val_2022
# for y in list(range(1993,2023)):
#     dm[:,y,...] = dm[:,2022,...] * (1+df.loc[df["year"]==y,"change"].values)
# 
# # fill nas
# dm.drop("Years",years_fts)
# dm = linear_fitting(dm, years_ots)
# # dm.flatten().datamatrix_plot()
# dm.drop("Years",[1990,1991,1992])
# dm = linear_fitting(dm, years_ots,based_on=[1993])
# # dm.flatten().datamatrix_plot()
# dm = linear_fitting(dm, years_fts)
# # dm.flatten().datamatrix_plot()
# =============================================================================

# # put together
# dm.drop("Years",2024)
# dm_waste.append(dm, "Variables")

##############################
##### ALUMINIUM PACKAGES #####
##############################

# note: Switzerland does not have facilities for the recycling of aluminium (https://alu-recycling.ch/fr/le-circuit-de-lalu/ and https://www.bafu.admin.ch/bafu/en/home/topics/waste/guide-to-waste-a-z/electrical-and-electronic-equipment.html)

dm = DM_eu["ots"].filter({"Country" : ["EU27"], "Variables" : ['aluminium-pack']})
dm.append(DM_eu["fts"][1].filter({"Country" : ["EU27"], "Variables" : ['aluminium-pack']}), "Years")
dm.rename_col("EU27","Switzerland","Country")

dm[...,"export"] = 1
dm[...,"waste-collected"] = 0
dm[...,"waste-uncollected"] = 0
dm[...,"littered"] = 0

dm[...,"recycling"] = 0
dm[...,"energy-recovery"] = 0
dm[...,"reuse"] = 0
dm[...,"incineration"] = 0
dm[...,"landfill"] = 0

dm_waste.append(dm, "Variables")

##########################
##### PAPER PACKAGES #####
##########################

# note: in theory the recyclable ones are paper pack and paper print, paper san goes all to water / not recyclable

# https://spkf.ch/wp-content/uploads/2024/06/bk-240521-statistischer-Jahresbericht-RPK-2023.pdf

dm = DM_eu["ots"].filter({"Country" : ["EU27"], "Variables" : ['paper-pack','paper-san','paper-print']})
dm.append(DM_eu["fts"][1].filter({"Country" : ["EU27"], "Variables" : ['paper-pack','paper-san','paper-print']}), "Years")
dm.rename_col("EU27","Switzerland","Country")

# figures for 2023 in tonnes
# pack_collected_separated = 277930
# pack_collected_mixedbags = 17005
# print_collected_separated = 343685
# print_collected_mixedbags = 21029
# san_consumed = 149706
# recycled_ch = 680790
# export_waste_paper = 460357
# energy_recovery = 40136
# export_share = export_waste_paper/(pack_collected_separated + pack_collected_mixedbags + print_collected_separated + print_collected_mixedbags)

# consumption = 1330989
# export = 460357
# dm[...,"export"] = export/consumption
# dm[...,"littered"] = 0.07 # we assume same littered than plastics
# dm[...,"waste-uncollected"] = 0.03 # RP+K reports recyclable paper still found in residual waste (hence incinerated, not littered) at 40,136 t, which is 3.02 % of total paper consumption (40,136 ÷ 1,330,989). That’s not littering, but it does quantify the mis-sorted share that misses separate collection
# dm[...,"waste-collected"] = 1 - dm[...,"export"] - dm[...,"littered"] - dm[...,"waste-uncollected"]

# collected = 1141147
# recycling = 835333
# dm[...,"recycling"] = recycling/collected
# dm[...,"energy-recovery"] = 1 - dm[...,"recycling"]
# dm[...,"landfill"] = 0
# dm[...,"incineration"] = 0
# dm[...,"reuse"] = 0

# # assume same for paper pack and paper print, and fix paper san (we assume layer 1 same of others, and layer 2 all used for energy recovery)
# dm[...,"paper-san","recycling"] = 0
# dm[...,"paper-san","energy-recovery"] = 1

consumption = 1_330_989      # total paper & cardboard generated (RP+K "consumption")
collected   = 1_141_147      # separately collected for recycling (FOEN/RP+K)
export      = 460_357        # exported recovered paper (subset of collected)
imports     = 154_543        # (not used in mass balance below)
mills_input = 835_333        # domestic mills' recovered paper input (includes imports)
hygiene     = 149_706        # non-recyclable tissue/hygiene
mis_sorted  = 40_136         # recyclable paper left in residual waste (incinerated)

dm[...,"export"] = export / consumption  
dm[...,"littered"] = 0.07 # we assume same littered than plastics
dm[...,"waste-uncollected"] = 0 # assuming in CH all waste is collected
dm[...,"waste-collected"] = 1 - dm[...,"export"] - dm[...,"littered"] - dm[...,"waste-uncollected"]
dm[...,"recycling"]       = 1.0
dm[...,"incineration"]    = 0.0
dm[...,"landfill"]        = 0.0
dm[...,"energy-recovery"] = 0.0
dm[...,"reuse"]           = 0.0

# assume same for paper pack and paper print, and fix paper san (we assume layer 1 same of others, and layer 2 all used for energy recovery)
dm[...,"paper-san","recycling"] = 0
dm[...,"paper-san","energy-recovery"] = 1

dm_waste.append(dm, "Variables")
dm_waste.sort("Variables")

################
##### SAVE #####
################

dm_ots = dm_waste.filter({"Years" : years_ots})
dm_fts = dm_waste.filter({"Years" : years_fts})
DM_fts = {1: dm_fts.copy(), 2: dm_fts.copy(), 3: dm_fts.copy(), 4: dm_fts.copy()} # for now we set all levels to be the same
DM = {"ots" : dm_ots,
      "fts" : DM_fts}
f = os.path.join(current_file_directory, '../data/datamatrix/lever_waste-management.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)
