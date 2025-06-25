
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

# file
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/industry/eu/python/industry_const_emission-factors.py"

# directories
current_file_directory = os.path.dirname(os.path.abspath(__file__))

###################################################################################
############################### COMBUSTION EMISSIONS ##############################
###################################################################################

# source: https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/2_Volume2/V2_2_Ch2_Stationary_Combustion.pdf
# Table 2.3 page 2.18

# filepath
filepath = os.path.join(current_file_directory, '../data/Literature/literature_review_ghg_emission_factors.xlsx')
df = pd.read_excel(filepath)

# order
df = df.sort_values("carrier_calc")

# melt
df = pd.melt(df, id_vars = ["carrier_calc"], var_name='gas')

# make variable
df["variable"] = [ec + "_" + gas + "[TWh/Mt]" for ec, gas in zip(df["carrier_calc"], df["gas"])]

# subset
df = df.loc[:,["variable","value"]]

# make cdm
from model.common.constant_data_matrix_class import ConstantDataMatrix
def create_constant(df, variables):
    
    df_temp = df.loc[df["variable"].isin(variables),:]
    
    # put unit
    df_temp["unit"] = [i.split("[")[1].split("]")[0] for i in df_temp["variable"]]
    
    const = {
        'name': list(df_temp['variable']),
        'value': list(df_temp['value']),
        'idx': dict(zip(list(df_temp['variable']), range(len(df_temp['variable'])))),
        'units': dict(zip(list(df_temp['variable']), list(df_temp['unit'])))
    }
    
    # return
    return const

tmp = create_constant(df, df["variable"])
cdm = ConstantDataMatrix.create_from_constant(tmp, 0)
cdm.deepen()

# store
CDM_emissions = {"combustion-emissions" : cdm}

df = cdm.write_df()
df["country"] = "all"
df_temp = pd.melt(df, id_vars = ['country'], var_name='variable')
name = "temp.xlsx"
df_temp.to_excel("~/Desktop/" + name)

###################################################################################
################################ PROCESS EMISSIONS ################################
###################################################################################

# From chatGPT:
# Materials with non-negligible CH₄ and/or N₂O process emissions
# Ammonia – N₂O emissions are significant.
# Ammonia production via the Haber-Bosch process itself does not emit much N₂O.
# However, N₂O is a major byproduct in nitric acid production, which is used to make fertilizers from ammonia.
# CH₄ is not a major process emission, but it is a key feedstock.
# Chemicals (varies by process, but often N₂O emissions occur)
# Adipic acid and nitric acid production:
# These processes generate large amounts of N₂O as a byproduct.
# Adipic acid production (used in nylon) is one of the biggest industrial sources of N₂O.
# Some chemical processes can emit CH₄, but this depends on the specific reaction pathways.
# Integrated steelworks (blast furnace steel production) – CH₄ emissions can be relevant.
# Blast furnaces produce coke oven gas, which contains CH₄.
# N₂O can also form in sintering and iron ore reduction processes, but emissions are usually lower than CO₂.
# Pulp and Paper – CH₄ and N₂O can be emitted, though at lower levels.
# CH₄: Can be emitted from the anaerobic breakdown of organic material in wastewater treatment.
# N₂O: Can form in the combustion of black liquor (a byproduct of pulping).
# Materials where CH₄ and N₂O process emissions are negligible
# Primary aluminum – No significant CH₄ or N₂O emissions; CO₂ and PFCs dominate.
# Secondary aluminum – No significant CH₄ or N₂O emissions.
# Dry-kiln cement & Wet-kiln cement – Mostly CO₂ emissions from limestone calcination; CH₄ and N₂O are minimal.
# Copper – CO₂ is the main process emission; CH₄ and N₂O are negligible.
# Glass – CO₂ from carbonate decomposition is the main concern; CH₄ and N₂O are minimal.
# Lime – Almost exclusively CO₂ emissions from limestone decomposition.
# Electric-arc furnace steel – Mainly CO₂; CH₄ and N₂O emissions are not significant.
# Summary
# Significant N₂O sources: Ammonia (via nitric acid), adipic acid (in chemicals).
# Significant CH₄ sources: Integrated steelworks, pulp & paper (wastewater).
# Negligible CH₄ and N₂O: Aluminum, cement, copper, glass, lime, EAF steel.

# So for CO2 I'll get as much as I can from JRC, and for the rest, for the non
# negliblible ones I will get the
# process emission factors from IPCC: https://www.ipcc-nggip.iges.or.jp/public/2006gl/vol3.html

###############
##### CO2 #####
###############

# get year of selection
year_start = 2021
year_end = 2021
years = list(range(year_start,year_end+1))

# filepath
filepath = os.path.join(current_file_directory, '../data/JRC-IDEES-2021/EU27/JRC-IDEES-2021_Industry_EU27.xlsx')

# def my search
def my_search(search, x):
    if x is np.nan:
        return False
    else:
        return bool(re.search(search, x, re.IGNORECASE))

def get_data(filepath, tech_code, tech, tech_calc, years):
    
    # get df
    df = pd.read_excel(filepath, tech_code + "_emi")

    # get data
    ls_temp = [my_search(tech, i) for i in df.iloc[:,0]]
    index_row_start = [i for i, x in enumerate(ls_temp) if x][0]
    df_temp = df.iloc[range(index_row_start, len(df)),:]
    ls_temp = [my_search("process emissions", i) for i in df_temp.iloc[:,0]]
    if not any(ls_temp):
        df_pe = pd.DataFrame({"variable" : ["process-emissions_" + tech_calc + "_CO2[Mt/Mt]"], "value" : [0], 
                              "process-emissions": [0], "production" : [np.nan]})
        return df_pe
    
    else:
        index_row_end = [i for i, x in enumerate(ls_temp) if x][0]
        df_temp = df_temp.iloc[[0, index_row_end],:]
    
        # melt
        id_var = df_temp.columns[0]
        ls_temp = [my_search("Process emissions", i) for i in df_temp.loc[:,id_var]]
        df_temp = df_temp.loc[ls_temp,[id_var] + years]
        df_temp = pd.melt(df_temp, id_vars = [id_var], var_name='year')
        df_temp = df_temp.groupby([id_var], as_index=False)['value'].agg(np.mean)
        df_temp.columns = ["tech","value"]
        df_temp["tech"] = tech
        df_pe = df_temp.copy()
    
        # physical output
        df = pd.read_excel(filepath, tech_code)
    
        # get data
        ls_temp = [my_search("physical output", i) for i in df.iloc[:,0]]
        index_row_start = [i for i, x in enumerate(ls_temp) if x][0]
        df_temp = df.loc[range(index_row_start,len(df)),:]
        ls_temp = [my_search(tech, i) for i in df_temp.iloc[:,0]]
        index_row_end = [i for i, x in enumerate(ls_temp) if x][0]
        df_temp = df_temp.iloc[[0,index_row_end],:]
        ls_temp = [my_search(tech, i) for i in df_temp.iloc[:,0]]
        df_temp = df_temp.loc[ls_temp,:]
    
        # melt
        id_var = df_temp.columns[0]
        df_temp = df_temp.loc[:,[id_var] + years]
        df_temp = pd.melt(df_temp, id_vars = [id_var], var_name='year')
        df_temp = df_temp.groupby([id_var], as_index=False)['value'].agg(np.mean)
        df_temp.columns = ["tech","production"]
        df_temp["tech"] = tech
    
        # merge
        df_pe = pd.merge(df_pe, df_temp, how="left", on=["tech"])
        df_pe.loc[df_pe["tech"] == tech,"tech"] = tech_calc
        df_check = df_pe.copy()
    
        # convert units
        df_pe["value"] = df_pe["value"]/1000
        df_pe["production"] = df_pe["production"]/1000
    
        # get emission factors
        df_pe["value"] = df_pe["value"]/df_pe["production"]
    
        # clean
        df_pe = df_pe.loc[:,["tech","value"]]
        df_check.columns = ["tech","process-emissions","production"]
        df_pe = pd.merge(df_pe, df_check, how="left", on=["tech"])
        df_pe.columns = ["variable","value","process-emissions","production"]
        df_pe["variable"] = ["process-emissions_" + v + "_CO2[Mt/Mt]" for v in df_pe["variable"]]
        
        # return
        return df_pe


dict_input = {"Integrated steelworks" : ["ISI", "steel-BF-BOF"],
              "Electric arc" : ["ISI", 'steel-scrap-EAF'],
              "Cement" : ["NMM", "cement"],
              "Glass production": ["NMM", "glass-glass-tech"],
              "Basic chemicals": ['CHI','chem-chem-tech'],
              "Pulp production": ['PPA','pulp-tech'],
              "Paper production": ['PPA','paper-tech'],
              "Aluminium - primary production" : ["NFM","aluminium-prim"],
              "Aluminium - secondary production": ["NFM","aluminium-sec"],
              "Other non-ferrous metals" : ["NFM","copper-tech"],
              "Food, beverages and tobacco": ["FBT", "fbt-tech"],
              "Machinery equipment": ["MAE", "mae-tech"],
              "Other industrial sectors": ["OIS","ois-tech"],
              "Textiles and leather" : ["TEL","textiles-tech"],
              "Transport equipment" : ["TRE","tra-equip-tech"],
              "Wood and wood products" : ["WWP", "wwp-tech"]}

df = pd.concat([get_data(filepath, dict_input[key][0], key, dict_input[key][1], years) for key in dict_input.keys()])
# OK values are good

# clean
df = df.loc[:,["variable","value"]]
df.sort_values(by=["variable"],inplace=True)

# assign same process emissions to cement wet-kiln and dry-kiln, and 0 to cement-gepolym
value = np.array(df.loc[df["variable"] == "process-emissions_cement_CO2[Mt/Mt]","value"])[0]
df_temp = pd.DataFrame({"variable" : ["process-emissions_cement-dry-kiln_CO2[Mt/Mt]",
                                      "process-emissions_cement-wet-kiln_CO2[Mt/Mt]",
                                      "process-emissions_cement-geopolym_CO2[Mt/Mt]"],
                        "value" : [value,value,0]})
df = pd.concat([df, df_temp])
df = df.loc[df["variable"] != "process-emissions_cement_CO2[Mt/Mt]",:]

# ammonia
# source: https://dechema.de/dechema_media/Downloads/Positionspapiere/Technology_study_Low_carbon_energy_and_feedstock_for_the_European_chemical_industry.pdf
# page 57 Table 11
value = 0.5
df_temp = pd.DataFrame({"variable" : ["process-emissions_ammonia-tech_CO2[Mt/Mt]"],
                        "value" : [value]})
df = pd.concat([df, df_temp])

# lime
# source: https://www.eula.eu/wp-content/uploads/2019/02/A-Competitive-and-Efficient-Lime-Industry-Technical-report-by-Ecofys_0.pdf
# page 31
quicklime = 0.751
dolime = 0.807
sintered_dolime = 0.913
value = np.mean([quicklime, dolime, sintered_dolime])
df_temp = pd.DataFrame({"variable" : ["process-emissions_lime-lime_CO2[Mt/Mt]"],
                        "value" : [value]})
df = pd.concat([df, df_temp])
df.sort_values(by=["variable"],inplace=True)

# steel dri
# source: https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/3_Volume3/V3_4_Ch4_Metal_Industry.pdf
# page 4.25
# assumption: process emissions of hydrogen DRI are the same of DRI
value = 0.7
df_temp = pd.DataFrame({"variable" : ["process-emissions_steel-hydrog-DRI_CO2[Mt/Mt]"],
                        "value" : [value]})
df = pd.concat([df, df_temp])
df.sort_values(by=["variable"],inplace=True)

# steel hisarna
# assumption: 20% less than BF-BOF
value = np.array(df.loc[df["variable"] == "process-emissions_steel-BF-BOF_CO2[Mt/Mt]","value"])[0]*(1-0.2)
df_temp = pd.DataFrame({"variable" : ["process-emissions_steel-hisarna_CO2[Mt/Mt]"],
                        "value" : [value]})
df = pd.concat([df, df_temp])
df.sort_values(by=["variable"],inplace=True)

# aluminium-sec-post-consumer
# assumption: same of alluminium sec
# TODO: check the literature and re-do this
df_temp = df.loc[df["variable"] == "process-emissions_aluminium-sec_CO2[Mt/Mt]",:]
df_temp["variable"] = "process-emissions_aluminium-sec-post-consumer_CO2[Mt/Mt]"
df = pd.concat([df, df_temp])

# cement-sec-post-consumer
# source: https://www.sciencedirect.com/science/article/pii/S235255412300044X
# from abstract: first tech no emission reduction, with other 2 techs, up until 80% reduction in emissions
# assuming that this is translated to process emissions (which make up for a large part of emissions from clinker)
# I make an average between 0% and 80%
# TODO: check the literature and re-do this
ec_perc_less = np.mean(np.array([0,0.8]))
df_temp = df.loc[df["variable"] == "process-emissions_cement-dry-kiln_CO2[Mt/Mt]",:]
df_temp["value"] = df_temp["value"]*(1-ec_perc_less)
df_temp["variable"] = "process-emissions_cement-sec-post-consumer_CO2[Mt/Mt]"
df = pd.concat([df, df_temp])

# chem-sec-post-consumer
# it seems that energy consumption and emissions in post consumer recycling can differ a lot from chemical to chemial
# so for the moment I will put it the same of chemicals primary
# TODO: check the literature and re-do this
df_temp = df.loc[df["variable"] == "process-emissions_chem-chem-tech_CO2[Mt/Mt]",:]
df_temp["variable"] = "process-emissions_chem-sec-post-consumer_CO2[Mt/Mt]"
df = pd.concat([df, df_temp])

# copper-sec-post-consumer
# source: https://internationalcopper.org/policy-focus/climate-environment/recycling/#:~:text=Recycled%20copper%20requires%2085%20percent,production%20and%20reduces%20CO2%20emissions
# here they do not mention process emissions, and it seems that the reduciton in emissions
# is just due to the lower energy demand to make secondary copper. So process emissions
# I will assign the same of process emissions for primary copper
# TODO: check the literature and re-do this
df_temp = df.loc[df["variable"] == "process-emissions_copper-tech_CO2[Mt/Mt]",:]
df_temp["variable"] = "process-emissions_copper-sec-post-consumer_CO2[Mt/Mt]"
df = pd.concat([df, df_temp])

# glass-sec-post-consumer
# sources:
# https://www.nrel.gov/docs/legosti/old/5703.pdf
# https://www.gpi.org/facts-about-glass-recycling
# https://www.agc-glass.eu/en/sustainability/decarbonisation/recycling
# no clear answer, so I will put the same of glass production for now
# TODO: check the literature and re-do this
df_temp = df.loc[df["variable"] == "process-emissions_glass-glass-tech_CO2[Mt/Mt]",:]
df_temp["variable"] = "process-emissions_glass-sec-post-consumer_CO2[Mt/Mt]"
df = pd.concat([df, df_temp])

# paper-sec-post-consumer
# source: https://ocshredding.com/blog/does-it-take-more-energy-to-produce-recycled-paper/#:~:text=According%20to%20the%20Environmental%20Paper,takes%20about%2022%20million%20BTUs.
# here they do not mention process emissions, so I assume that all the reduction in emissions
# comes from the lower energy demand, and that process emissions are the same
# TODO: check the literature and re-do this
df_temp = df.loc[df["variable"] == "process-emissions_paper-tech_CO2[Mt/Mt]",:]
df_temp["variable"] = "process-emissions_paper-sec-post-consumer_CO2[Mt/Mt]"
df = pd.concat([df, df_temp])

# steel-sec-post-consumer
# assumption: same process emissions of scrap EAF
df_temp = df.loc[df["variable"] == "process-emissions_steel-scrap-EAF_CO2[Mt/Mt]",:]
df_temp["variable"] = "process-emissions_steel-sec-post-consumer_CO2[Mt/Mt]"
df = pd.concat([df, df_temp])

# wwp-sec-post-consumer: assuming no wooden products post consumer for now

# sort
df.sort_values(by=["variable"],inplace=True)

###############
##### N2O #####
###############

# ammonia
# source: https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/3_Volume3/V3_3_Ch3_Chemical_Industry.pdf
# Table 3.3 page 3.23

value = np.mean(np.array([2, 2.5, 5, 7, 9])/1000)
df_N2O = pd.DataFrame({"variable" : ["process-emissions_ammonia-tech_N2O[Mt/Mt]"],
                           "value" : [value]})

# chemicals
# source: https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/3_Volume3/V3_3_Ch3_Chemical_Industry.pdf
# Table 3.4 page 3.30
# value = 300/1000
# note: the 300/1000 is for nitrict acid, which is very high, and we have chemicals in general, 
# which for the moment is mostly types of plastics, which do not have much of N2O emissions,
# aside from PVC, but at this stage it would be difficult to do the split, so
# we assign 0 N2O process emissions to chemicals.
value = 0
df_temp = pd.DataFrame({"variable" : ["process-emissions_chem-chem-tech_N2O[Mt/Mt]",
                                      "process-emissions_chem-sec-post-consumer_N2O[Mt/Mt]"], # I assume post consumer it's the same for process emissions N2O
                        "value" : [value, value]})
df_N2O = pd.concat([df_N2O, df_temp])

# assign 0 to others
techs = [i.split("_")[1].split("_")[0] for i in df["variable"]]
drops = ["ammonia-tech","chem-chem-tech","chem-sec-post-consumer"]
idx = [i not in drops for i in techs]
techs = list(np.array(techs)[idx])
variabs = ["process-emissions_" + t + "_N2O[Mt/Mt]" for t in techs]
df_temp = pd.DataFrame({"variable" : variabs, "value" : 0})
df_N2O = pd.concat([df_N2O, df_temp])
df_N2O.sort_values(by=["variable"],inplace=True)

# put together
df = pd.concat([df, df_N2O])


###############
##### CH4 #####
###############

# for steel with BF-BOF
# souce: https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/3_Volume3/V3_4_Ch4_Metal_Industry.pdf
# Table 4.2 page 4.26: very small quantities of CH4 per tonne, so I assume them to be zero

# for paper and pulp
# source: https://www.ipcc-nggip.iges.or.jp/public/gp/bgp/5_2_CH4_N2O_Waste_Water.pdf
# page 454: while the pulp and paper industry is the largest contributor to CH4
# from industrial processes (in theory from waste waster), I currently cannot find the factors quickly.
# Some sources are:
# https://www.sciencedirect.com/science/article/pii/S0921344916300088
# So for the moment I will assume them to be zero, and will come back to this later

# so for the moment I assign 0 to all
techs = set([i.split("_")[1].split("_")[0] for i in df["variable"]])
variabs = ["process-emissions_" + t + "_CH4[Mt/Mt]" for t in techs]
df_temp = pd.DataFrame({"variable" : variabs, "value" : 0})
df = pd.concat([df, df_temp])
df.sort_values(by=["variable"],inplace=True)

###########################################################
############# CONVERT TO CONSTANT DATA MATRIX #############
###########################################################

from model.common.constant_data_matrix_class import ConstantDataMatrix

# create dms
def create_constant(df, variables):
    
    df_temp = df.loc[df["variable"].isin(variables),:]
    
    # put unit
    df_temp["unit"] = [i.split("[")[1].split("]")[0] for i in df_temp["variable"]]
    
    const = {
        'name': list(df_temp['variable']),
        'value': list(df_temp['value']),
        'idx': dict(zip(list(df_temp['variable']), range(len(df_temp['variable'])))),
        'units': dict(zip(list(df_temp['variable']), list(df_temp['unit'])))
    }
    
    # return
    return const

# excluding feedstock
tmp = create_constant(df, df["variable"])
cdm = ConstantDataMatrix.create_from_constant(tmp, 0)
cdm.deepen_twice()

# store
CDM_emissions["process-emissions"] = cdm

# save
f = os.path.join(current_file_directory, '../data/datamatrix/const_emissions-factors.pickle')
with open(f, 'wb') as handle:
    pickle.dump(CDM_emissions, handle, protocol=pickle.HIGHEST_PROTOCOL)


# df = cdm.write_df()
# df["country"] = "all"
# df_temp = pd.melt(df, id_vars = ['country'], var_name='variable')
# name = "temp.xlsx"
# df_temp.to_excel("~/Desktop/" + name)





















