
# note: # fuel vs feedstock: machines eat energy to work (such as oil, etc), 
# but for some industrial processes they eat energy as material input
# (such as oil to make chemicals). Feedstock is this second thing.
# to compute feedstock, we'll use the table at page 61 of eucalc documentation, 
# with % of total

# JRC-2021
# Iron and Steel (ISI)
# ISI_fec: detailed split of final energy consumption
# ISI_ued: detailed split of useful energy demand
# ISI_emi: detailed split of CO2 emissions

# ISI
# Physical output (kt steel): by tech (integrated steelwork, electric arc)
# Energy consumption (ktoe): by fuel (solids, liquids, gas, res and wastes, distributed steam, electricity) or by tech (integrated steelwork, electric arc), not both
# CO2 emissions (kt CO2): by type (energy-use related, process emissions) or by tech (integrated steelwork, electric arc), not both

# ISI_fec, ISI_ued, ISI_emi are the same if ISI just with the split between the different
# phases of the technology, such as lighting, air compressor, sinter making, 
# basic oxygen furnance, product finishing, etc. Here there are the split
# by energy carrier for each tech.

# The difference between energy consumption and useful enerfy demand is that energy
# consumption is the total amount of energy needed to make the machine work, while
# useful energy demand is the amount of energy needed excluding the energy loss.
# I will consider the energy consumption.

# Energy consumption is given in tonne of oil equivalent (toe).
# The International Energy Agency defines one tonne of oil equivalent (toe) to be equal to: 1 toe = 11.63 megawatt-hours (MWh)
# https://www.iea.org/reports/glossary-of-energy-units

# In the calc, here are the techs and energy carriers
# ['steel-BF-BOF', 'steel-scrap-EAF', 'steel-hisarna', 'steel-hydrog-DRI', 'cement-dry-kiln', 'cement-wet-kiln', 'cement-geopolym', 'chem-chem-tech', 'ammonia-tech', 'paper-woodpulp', 'paper-recycled', 'aluminium-prim', 'aluminium-sec', 'glass-glass','lime-lime','copper-tech','fbt-tech', 'mae-tech', 'ois-tech', 'textiles-tech', 'tra-equip-tech', 'wwp-tech']
# ['electricity', 'gas-bio', 'gas-ff-natural', 'hydrogen', 'liquid-bio', 'liquid-ff-oil', 'solid-bio', 'solid-ff-coal', 'solid-waste']

# packages
import pandas as pd
import pickle
import os
import numpy as np
import warnings
warnings.simplefilter("ignore")
import re

# directories
current_file_directory = os.getcwd()

###########################################################
############## CATEGORIES OF ENERGY CARRIERS ##############
###########################################################

# categories jrc (unaggregated)
enercarr_jrc = ["Diesel oil and liquid biofuels", "Natural gas and biogas", "Solar and geothermal",
                "Ambient heat", "Electricity", "Solids", "Fuel oil", "Derived gases", "Coke", 
                "LPG", "Refinery gas", "Other liquids", "Biomass and waste", "Distributed steam"]
electric_usual = ["Lighting", "Air compressors", "Motor drives", "Fans and pumps"]

# categories jrc aggregated
enercarr_jrc_agg_map = ["Diesel oil and liquid biofuels", "Natural gas and biogas", "Solar and geothermal",
                        "Ambient heat", "Electricity", "Hard coal and others", "Fuel oil", "Derived gases", "Coke",
                        "LPG", "Refinery gas", "Other liquids", "Biomass and waste", "Distributed steam"]

# this is an adjustment needed to split some categories using the jrc data across techs (as for the specific tech they are missing)
enercarr_jrc_agg_map_adj = {"Diesel oil and liquid biofuels" : ["Diesel oil (without biofuels)", "Liquid biofuels"],
                            "Natural gas and biogas" : ["Natural gas", "Biogas"],
                            "Solar and geothermal" : ["Solar", "Geothermal"]}
enercarr_jrc_agg = enercarr_jrc_agg_map.copy()
for key in enercarr_jrc_agg_map_adj.keys():
    enercarr_jrc_agg = enercarr_jrc_agg + enercarr_jrc_agg_map_adj[key]
    enercarr_jrc_agg.remove(key)

# categories calculator
enercarr_calc_map = ["electricity", "electricity", "solid-ff-coal", "liquid-ff-oil", "gas-ff-natural",
                     "solid-ff-coal", "gas-ff-natural", "gas-ff-natural", "liquid-ff-oil", "Biomass and waste", "electricity",
                     "liquid-ff-diesel", "liquid-bio", "gas-ff-natural", "gas-bio", "electricity", "electricity"]
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
gj_to_twh = 2.7777777777778*1e-7

# get year of selection
year_start = 2018
year_end = 2021
years = list(range(year_start,year_end+1))

# file
filepath = os.path.join(current_file_directory, '../data/JRC-IDEES-2021/EU27/JRC-IDEES-2021_Industry_EU27.xlsx')

# get df of material production for 'fbt', 'mae', 'ois', 'textiles', 'tra-equip', 'wwp-tech'
f = os.path.join(current_file_directory, '../data/datamatrix/calibration_material-production.pickle')
with open(f, 'rb') as handle:
    dm_matprod_calib = pickle.load(handle)
df_prod_extramat = dm_matprod_calib.filter({"Country" : ["EU27"], 
                                            "Categories1" : ['fbt','mae','ois','textiles', 'tra-equip', 'wwp']}).write_df()
df_prod_extramat = pd.melt(df_prod_extramat, id_vars = ["Country","Years"], var_name='variable')
df_prod_extramat = df_prod_extramat.loc[df_prod_extramat["Years"].isin(years),:]
df_prod_extramat = df_prod_extramat.groupby(["variable"], as_index=False)['value'].agg(np.mean)
df_prod_extramat["tech"] = [i.split("_")[1].split("[")[0] + "-tech" for i in df_prod_extramat["variable"]]
df_prod_extramat = df_prod_extramat.loc[:,["tech","value"]]
df_prod_extramat.columns = ["tech","production"]
df_prod_extramat["production"] = df_prod_extramat["production"]/1000000

# clean
del waste_municipal, waste_industrial, waste, solid_biomass, tot

# function to get energy intensities from JRC files
def get_energy_intensity(filepath, techs_code, techs, techs_calc, dict_subtechs_elec, years):
    
    # definitions
    # filepath: path of JRC file to be uploaded
    # techs_code: code of technology in JRC file (e.g. ISI)
    # techs: name of technologies for techs_code (e.g. ["Integrated steelworks","Electric arc"])
    # techs_calc: names of technologies in our calculator (e.g. ['steel-BF-BOF', 'steel-scrap-EAF'])
    # dict_subtechs_elec: names of energy carriers and sub techs that are aggregated under electricity
    # years: which years to select to build the constants
    
    # get energy consumption aggregated across techs (toe)
    df = pd.read_excel(filepath, techs_code)
    id_var = df.columns[0]
    df = df.loc[:,[id_var] + years]
    ls_temp = list(df.iloc[:,0].isin(["Energy consumption (ktoe)"]))
    if techs_code in ["OIS"]:
        ls_temp = list(df.iloc[:,0].isin(["Energy consumption (ktoe)*"]))
    index_row_start = [i +1 for i, x in enumerate(ls_temp) if x]
    df_temp = df.iloc[range(index_row_start[0],len(df)),:]
    ls_temp = list(df_temp.iloc[:,0].isin(["by subsector (calibration output)"]))
    index_row_end = [i +1 for i, x in enumerate(ls_temp) if x]
    if len(index_row_end) > 0:
        index_row_end = index_row_end[0]-1
    else:
        ls_temp = list(df_temp.iloc[:,0].isin(["CO2 emissions (kt CO2)"]))
        index_row_end = [i-1 for i, x in enumerate(ls_temp) if x][0]
    df_ec = df_temp.iloc[range(1,index_row_end),:]
    df_ec = pd.melt(df_ec, id_vars = id_var, var_name='year')
    df_ec = df_ec.groupby([id_var], as_index=False)['value'].agg(np.mean)
    df_ec.columns = ["energy_carrier","value"]
    df_ec = df_ec.loc[~df_ec["energy_carrier"].isin(["Solids","Liquids","Gas","RES and wastes"])]
    # index_row_end_totals = index_row_end+1
    # df_ec_tot = df_temp.iloc[range(index_row_end_totals,index_row_end_totals + len(techs)),:]

    # get physical output
    ls_temp = list(df.iloc[:,0])
    ls_temp = [str(i) for i in ls_temp]
    ls_temp = [bool(re.search("Physical output", i, re.IGNORECASE)) for i in ls_temp]
    index_row_start = [i +1 for i, x in enumerate(ls_temp) if x][0]
    if techs_code in ["FBT","TRE","MAE","TEL","WWP","OIS"]:
        index_row_start = [i for i, x in enumerate(ls_temp) if x][0]
    index_row_end = index_row_start + len(techs)
    if techs_code == "NFM": # this is necessary as for NFM they also have overall aluminium production (which is an aggregate) in the production part
        index_row_end = index_row_start + len(techs)+1
    df_po = df.iloc[range(index_row_start,index_row_end),:]
    df_po = pd.melt(df_po, id_vars = id_var, var_name='year')
    df_po = df_po.groupby([id_var], as_index=False)['value'].agg(np.mean)
    df_po.columns = ["tech","value"]
    for i in range(0, len(techs)):
        idx = [bool(re.search(techs[i], string)) for string in df_po["tech"]]
        df_po.loc[idx,"tech"] = techs_calc[i]
    if techs_code in ["FBT","TRE","MAE","TEL","WWP","OIS"]:
        df_po["tech"] = techs_calc
    df_po["unit"] = "Kt"

    # get specific data on final energy consumption (fec)
    df = pd.read_excel(filepath, techs_code + "_fec")
    id_var = df.columns[0]
    df = df.loc[:,[id_var] + years]
    ls_temp = list(df.iloc[:,0].isin(["Detailed split of energy consumption by subsector (ktoe)"]))
    index_row_start = [i +1 for i, x in enumerate(ls_temp) if x]
    if techs_code in ["FBT","TRE","MAE","TEL","WWP","OIS"]:
        ls_temp = list(df.iloc[:,0].isin(["Detailed split of energy consumption (ktoe)"]))
        index_row_start = [i +2 for i, x in enumerate(ls_temp) if x]
    ls_temp = list(df.iloc[:,0].isin(["Market shares of energy uses by subsector (%)"]))
    index_row_end = [i-2 for i, x in enumerate(ls_temp) if x]
    if techs_code in ["FBT","TRE","MAE","TEL","WWP","OIS"]:
        ls_temp = list(df.iloc[:,0].isin(["Market shares of energy uses (%)"]))
        index_row_end = [i-1 for i, x in enumerate(ls_temp) if x]
    df = df.loc[range(index_row_start[0], index_row_end[0]),:]
    DF = {"fec" : df.copy()}
    
    # get specific data on useful energy demand (ued)
    df = pd.read_excel(filepath, techs_code + "_ued")
    id_var = df.columns[0]
    df = df.loc[:,[id_var] + years]
    ls_temp = list(df.iloc[:,0].isin(["Detailed split of useful energy demand by subsector (ktoe)"]))
    index_row_start = [i +1 for i, x in enumerate(ls_temp) if x]
    if techs_code in ["FBT","TRE","MAE","TEL","WWP","OIS"]:
        ls_temp = list(df.iloc[:,0].isin(["Detailed split of useful energy demand (ktoe)"]))
        index_row_start = [i +2 for i, x in enumerate(ls_temp) if x]
    ls_temp = list(df.iloc[:,0].isin(["Market shares of useful energy demand by subsector (%)"]))
    index_row_end = [i-2 for i, x in enumerate(ls_temp) if x]
    if techs_code in ["FBT","TRE","MAE","TEL","WWP","OIS"]:
        ls_temp = list(df.iloc[:,0].isin(["Market shares of useful energy demand (%)"]))
        index_row_end = [i-1 for i, x in enumerate(ls_temp) if x]
    df = df.loc[range(index_row_start[0], index_row_end[0]),:]
    DF["ued"] = df.copy()

    # do operations on each df
    DF_techs = {}
    for energy_key in DF.keys():
        
        df = DF[energy_key].copy()
        
        for i in range(0,len(techs)):
            
            # subset
            ls_temp = list(df.iloc[:,0].isin([techs[i]]))
            index_row_start = [i +1 for i, x in enumerate(ls_temp) if x]
            if i != len(techs)-1:
                ls_temp = list(df.iloc[:,0].isin([techs[i+1]]))
                if techs_code == "PPA": # this is because with PPA_fec there is a row less between techs
                    index_row_end = [i -1 for i, x in enumerate(ls_temp) if x]
                else:
                    index_row_end = [i -2 for i, x in enumerate(ls_temp) if x]
            else:
                index_row_end = list()
                index_row_end.append(len(df))
            df_temp = df.iloc[range(index_row_start[0],index_row_end[0]),:]
            
            # get aggregates
            # note: you can take Lighting out of dict_subtechs_elec (to avoid aggregation together with elec)
            id_var = df_temp.columns[0]
            df_temp = df_temp.loc[df_temp[id_var].isin(enercarr_jrc + dict_subtechs_elec[techs[i]]),:]
            mysubset = dict_subtechs_elec[techs[i]].copy()
            mysubset.remove('Lighting')
            df_temp.loc[df_temp[id_var].isin(mysubset),id_var] = "Electricity"
            df_temp = df_temp.groupby([id_var], as_index=False).agg(sum)
            
            # take average across years
            df_temp = pd.melt(df_temp, id_vars = id_var, var_name='year')
            df_temp = df_temp.groupby([id_var], as_index=False)['value'].agg(np.mean)
            df_temp.columns = ["energy_carrier","value"]
            
            # check
            df_check = df_temp.copy()
            df_check[id_var] = "total"
            df_check = df_check.groupby([id_var], as_index=False)["value"].agg(sum)
            
            # apply aggrgates JRC
            for idx in range(0,len(enercarr_jrc)):
                df_temp.loc[df_temp["energy_carrier"] == enercarr_jrc[idx],"energy_carrier"] = enercarr_jrc_agg_map[idx]
            
            # adjust some categories with total shares
            for key in enercarr_jrc_agg_map_adj.keys():
                for idx in range(0, len(enercarr_jrc_agg_map_adj[key])):
                    adj_value = df_temp.loc[df_temp["energy_carrier"] == key,"value"] * \
                        np.array(df_ec.loc[df_ec["energy_carrier"] == enercarr_jrc_agg_map_adj[key][idx],"value"]) / \
                            np.array(sum(df_ec.loc[df_ec["energy_carrier"].isin(enercarr_jrc_agg_map_adj[key]),"value"]))
                    df_temp = pd.concat([df_temp, pd.DataFrame({"energy_carrier" : enercarr_jrc_agg_map_adj[key][idx], 
                                                                "value" : adj_value})])
                df_temp = df_temp.loc[df_temp["energy_carrier"] != key,:]
                
            # check
            df_check = df_temp.copy()
            df_check[id_var] = "total"
            df_check = df_check.groupby([id_var], as_index=False)["value"].agg(sum)
            
            # make aggregates as in calculator
            for idx in range(0, len(enercarr_jrc_agg)):
                df_temp.loc[df_temp["energy_carrier"] == enercarr_jrc_agg[idx],"energy_carrier"] = enercarr_calc_map[idx]
            df_temp = df_temp.groupby(["energy_carrier"], as_index=False).agg(sum)
            df_temp.loc[df_temp["energy_carrier"] == "Lighting","energy_carrier"] = "lighting"
            
            # split biomass and waste
            for key in dict_adj_biomass_waste.keys():
                adj_value = df_temp.loc[df_temp["energy_carrier"] == "Biomass and waste","value"] * \
                    dict_adj_biomass_waste[key]
                df_temp = pd.concat([df_temp, pd.DataFrame({"energy_carrier" : key, "value" : adj_value})])
            df_temp = df_temp.loc[df_temp["energy_carrier"] != "Biomass and waste",:]
    
            # check
            df_check = df_temp.copy()
            df_check[id_var] = "total"
            df_check = df_check.groupby([id_var], as_index=False)["value"].agg(sum)
            
            # add hydrogen
            df_temp = pd.concat([df_temp, pd.DataFrame({"energy_carrier" : ["hydrogen"], "value" : [0]})])
            
            # store
            DF_techs[energy_key + "_" + techs[i]] = df_temp

    # put together
    def get_df(t, t_calc, e):
        df_temp = DF_techs[e + "_" + t].copy()
        df_temp["tech"] = t
        df_temp["energy_demand_type"] = e
        df_temp.loc[df_temp["tech"] == t,"tech"] = t_calc
        return df_temp
    df_fec = pd.concat([get_df(t, t_calc, e = "fec") for t, t_calc in zip(techs, techs_calc)])
    df_ued = pd.concat([get_df(t, t_calc, e = "ued") for t, t_calc in zip(techs, techs_calc)])
    df = pd.concat([df_fec,df_ued])
        
    # make df to check code output
    df_tot = df.copy()
    df_po.columns = ["tech","total","unit_production"]
    df_temp = df_fec.groupby(["tech"], as_index=False)['value'].agg(sum)
    df_temp = pd.merge(df_temp, df_po, how="left", on=["tech"])
    df_temp["ratio"] = df_temp["value"]/df_temp["total"]
    df_energyint_check = df_temp.copy()

    # convert toe to TWh, and divide by mega tonnes of production (objective is TWh/Mt)
    df_po["total"] = df_po["total"]/1000
    df_po["unit_production"] = "Mt"
    df["value"] = df["value"] * ktoe_to_twh
    df["unit"] = "TWh"
    df = pd.merge(df, df_po, how="left", on=["tech"])
    df["value"] = df["value"]/df["total"]

    # clean df
    df["variable"] = [tech + "_" + enercarr + "[" + unit + "/" + unit_production + "]" for \
                      tech, enercarr, unit, unit_production in \
                          zip(df["tech"], df["energy_carrier"], df["unit"], df["unit_production"])]
    df = df.loc[:,["variable","value","tech","energy_demand_type"]]
    
    # return
    DF = {"tot" : df_tot,
          "energy-intensity-check" : df_energyint_check,
          "energy-intensity" : df}
    return DF

###########################################################
########################## STEEL ##########################
###########################################################

# tech jrc code
techs_code = "ISI" # iron and steel

# techs
techs = ["Integrated steelworks","Electric arc"]
techs_calc = ['steel-BF-BOF', 'steel-scrap-EAF']

# this is aggregated into electricity
dict_subtechs_elec = {"Integrated steelworks" : electric_usual + ["Steel: Furnaces, refining and rolling - Electric", 
                                                                  "Steel: Product finishing - Electric"], 
                      "Electric arc" : electric_usual + ["Steel: Electric arc", 
                                                         "Steel: Furnaces, refining and rolling - Electric", 
                                                         "Steel: Product finishing - Electric"]}

# get energy intensity
DF = get_energy_intensity(filepath = filepath, techs_code = techs_code, techs = techs, 
                          techs_calc = techs_calc, dict_subtechs_elec = dict_subtechs_elec,
                          years = years)
df = DF["energy-intensity"].copy()

# check
df_check = DF["tot"].copy()
df_check = df_check.groupby(["tech"], as_index=False)["value"].agg(sum)
df_check = DF["energy-intensity-check"].copy()

# create steel Hisarna: reduction of energy demand of 20% with respect to steel BF-BOF
# from page 60 of https://www.european-calculator.eu/wp-content/uploads/2020/04/D3.1-Raw-materials-module-and-manufacturing.pdf
df_hi = df.loc[df["tech"] == "steel-BF-BOF",:]
df_hi["value"] = df_hi["value"]*(1-0.2)
df_hi["tech"] = "steel-hisarna"
df_hi["variable"] = [i.replace("steel-BF-BOF", "steel-hisarna") for i in df_hi["variable"]]
df = pd.concat([df, df_hi])

# create Steel - hydrogen-DRI: final energy consumption of 3.48 TWh/Mt,
# from page 60 of https://www.european-calculator.eu/wp-content/uploads/2020/04/D3.1-Raw-materials-module-and-manufacturing.pdf
# I am assuming 50% hydrogen and 50% electricity, reference to be found
df_dri = df_hi.copy()
df_dri["value"] = 0
df_dri["variable"] = [i.replace("steel-hisarna", "steel-hydrog-DRI") for i in df_dri["variable"]]
df_dri["tech"] = "steel-hydrog-DRI"
df_dri["total"] = 3.48
df_dri.loc[df_dri["variable"] == "steel-hydrog-DRI_electricity[TWh/Mt]","value"] = 0.5
df_dri.loc[df_dri["variable"] == "steel-hydrog-DRI_hydrogen[TWh/Mt]","value"] = 0.5
df_dri["value"] = df_dri["value"]*df_dri["total"]
df = pd.concat([df, df_dri.loc[:,["variable","value","tech","energy_demand_type"]]])

# # create steel post consumer
# # assumption: same energy demand of electric arc furnace for scrap
# # TODO: check the literature and re-do this
# df_temp = df.loc[df["tech"] == "steel-scrap-EAF",:]
# df_temp["tech"] = "steel-sec-post-consumer"
# df_temp["variable"] = [i.replace("steel-scrap-EAF","steel-sec-post-consumer") for i in df_temp["variable"]]
# df = pd.concat([df, df_temp])

# store
df_final = df.copy()

# clean
del DF, df, df_check, df_dri, df_hi, key, techs, techs_calc, techs_code, year_end, year_start

######################################################################
########################## CEMENT AND GLASS ##########################
######################################################################

# tech jrc code
techs_code = "NMM" # non metallic mineral products

# techs
techs = ["Cement","Ceramics & other NMM","Glass production"]
techs_calc = ['cement', 'ceramics','glass-glass']

# this is aggregated into electricity
dict_subtechs_elec = {"Cement" : electric_usual + ["Cement: Grinding, milling of raw material",
                                                   "Cement: Grinding, packaging and precasting (electricity)"],
                      "Ceramics & other NMM" : electric_usual + ["Ceramics: Mixing of raw material", 
                                                                 "Ceramics: Microwave drying and sintering", 
                                                                 "Ceramics: Electric kiln", 
                                                                 "Ceramics: Electric furnace"],
                      "Glass production" : electric_usual + ["Glass: Electric melting tank", "Glass: Forming",
                                                             "Glass: Annealing - electric", "Glass: Finishing processes"]}


# get energy intensity
DF = get_energy_intensity(filepath = filepath, techs_code = techs_code, techs = techs, 
                          techs_calc = techs_calc, dict_subtechs_elec = dict_subtechs_elec,
                          years = years)
df = DF["energy-intensity"].copy()

# check
df_check = DF["tot"].copy()
df_check = df_check.groupby(["tech"], as_index=False)["value"].agg(sum)
df_check = DF["energy-intensity-check"].copy()

# dropping ceramics
df = df.loc[df["tech"] != "ceramics",:]

# adding 0 for solid waste and solid bio for glass
df = pd.concat([df, pd.DataFrame({"variable" : ["glass-glass_solid-waste[TWh/Mt]",
                                               "glass-glass_solid-bio[TWh/Mt]",
                                               "glass-glass_solid-waste[TWh/Mt]",
                                               "glass-glass_solid-bio[TWh/Mt]"],
                                 "value" : [0,0,0,0],
                                 "tech" : ["glass-glass","glass-glass","glass-glass","glass-glass"],
                                 "energy_demand_type" : ["fec","fec","ued","ued"]})])
df.sort_values(["energy_demand_type","tech"],inplace=True)

# making 'cement-dry-kiln', 'cement-wet-kiln', with values from
# JRC (2013) page 137: https://op.europa.eu/en/publication-detail/-/publication/f5b60dd9-a0a1-447d-a64a-11c436f84492/language-en 
# and 'cement-geopolym' from EUCalc documentation page 60 https://www.european-calculator.eu/wp-content/uploads/2020/04/D3.1-Raw-materials-module-and-manufacturing.pdf
ls_cement_tech_ec = {'cement-dry-kiln' : 3.38 * gj_to_twh * 1000000, 
                     'cement-wet-kiln' : 6.34 * gj_to_twh * 1000000, 
                     'cement-geopolym' : 0.65}
for key in ls_cement_tech_ec.keys():
    df_temp = df.loc[(df["tech"] == "cement") & (df["energy_demand_type"] == "fec"),:]
    df_temp["total"] = np.sum(df_temp["value"])
    df_temp["share"] = df_temp["value"]/df_temp["total"]
    df_temp["literature"] = ls_cement_tech_ec[key]
    df_temp["value"] = df_temp["share"]*df_temp["literature"]
    df_temp = df_temp.loc[:,["variable","value","tech","energy_demand_type"]]
    df_temp["tech"] = key
    df_temp["variable"] = [i.replace("cement",key) for i in df_temp["variable"]]
    df = pd.concat([df, df_temp])

# make ued for 'cement-dry-kiln', 'cement-wet-kiln' and 'cement-geopolym' using efficiency of cement
df_efficiency = df.loc[df["tech"] == "cement",:]
df_efficiency = df_efficiency.pivot(index=['variable'], 
                                    columns=['energy_demand_type'], values="value").reset_index()
df_efficiency["efficiency"] = df_efficiency["ued"]/df_efficiency["fec"]
df_efficiency.loc[df_efficiency["variable"] == "cement_hydrogen[TWh/Mt]","efficiency"] = 0.5
df.sort_values(["tech","energy_demand_type","variable"],inplace=True)
for key in ls_cement_tech_ec.keys():
    df_temp = df.loc[df["tech"] == key,:]
    df_temp["value"] = df_temp.loc[:,"value"] * df_efficiency.loc[:,"efficiency"]
    df_temp["energy_demand_type"] = "ued"
    df = pd.concat([df, df_temp])
df = df.loc[df["tech"] != "cement",:]

# create cement secondary
# cement: https://www.sciencedirect.com/science/article/pii/S235255412300044X
# from abstract: there are marginal energy and emissions reduction with the wet method (so zero), while dry and air methods consume only 30%–40% of the energy for clinker production..
ec_perc_less = np.mean(np.array([0,0.30,0.40]))
df_temp = df.loc[df["tech"] == "cement-dry-kiln",:]
df_temp["value"] = df_temp["value"]*(1-ec_perc_less)
df_temp["tech"] = "cement-sec"
df_temp["variable"] = [i.replace("cement-dry-kiln","cement-sec") for i in df_temp["variable"]]
df = pd.concat([df, df_temp])

# create glass secondary
# sources:
# https://www.nrel.gov/docs/legosti/old/5703.pdf
# https://www.gpi.org/facts-about-glass-recycling
# https://www.agc-glass.eu/en/sustainability/decarbonisation/recycling
# no clear answer, so I will put the same of glass production for now
# TODO: check the literature and re-do this
df_temp = df.loc[df["tech"] == "glass-glass",:]
df_temp["tech"] = "glass-sec"
df_temp["variable"] = [i.replace("glass-glass","glass-sec") for i in df_temp["variable"]]
df = pd.concat([df, df_temp])

# store
df_final = pd.concat([df_final, df])

# clean
del DF, df, df_temp, key, ls_cement_tech_ec


######################################################################
############################# CHEMICALS ##############################
######################################################################

# tech jrc code
techs_code = "CHI" # chemical industry

# techs
techs = ["Basic chemicals"]
techs_calc = ['chem-chem-tech']

# this is aggregated into electricity
dict_subtechs_elec = {"Basic chemicals" : electric_usual + ["Chemicals: Furnaces - Electric",
                                                            "Chemicals: Process cooling - Electric",
                                                            "Chemicals: Generic electric process"]}

# get energy consumption aggregated across techs (toe)
df = pd.read_excel(filepath, techs_code)
id_var = df.columns[0]
df = df.loc[:,[id_var] + years]
ls_temp = list(df.iloc[:,0].isin(["Energy consumption (ktoe)"]))
index_row_start = [i +1 for i, x in enumerate(ls_temp) if x]
df_temp = df.iloc[range(index_row_start[0],len(df)),:]
df_temp = df_temp.reset_index(drop=True)
df_ec = df_temp.iloc[range(1,20),:]
df_ec = pd.melt(df_ec, id_vars = [id_var], var_name='year')
df_ec = df_ec.groupby([id_var], as_index=False)['value'].agg(np.mean)
df_ec.columns = ["energy_carrier","value"]
df_ec_fs = df_temp.iloc[range(27,39),:]
df_ec_fs = pd.melt(df_ec_fs, id_vars = [id_var], var_name='year')
df_ec_fs = df_ec_fs.groupby([id_var], as_index=False)['value'].agg(np.mean)
df_ec_fs.columns = ["energy_carrier","value"]

# get physical output
ls_temp = list(df.iloc[:,0])
ls_temp = [str(i) for i in ls_temp]
ls_temp = [bool(re.search("Physical output", i, re.IGNORECASE)) for i in ls_temp]
index_row_start = [i +1 for i, x in enumerate(ls_temp) if x][0]
index_row_end = index_row_start + len(techs)
df_po = df.iloc[range(index_row_start,index_row_end),:]
df_po = pd.melt(df_po, id_vars = id_var, var_name='year')
df_po = df_po.groupby([id_var], as_index=False)['value'].agg(np.mean)
df_po.columns = ["tech","value"]
for i in range(0, len(techs)):
    idx = [bool(re.search(techs[i], string)) for string in df_po["tech"]]
    df_po.loc[idx,"tech"] = techs_calc[i]
df_po["unit"] = "Kt"


# get specific data on final energy consumption (fec)
df = pd.read_excel(filepath, techs_code + "_fec")
id_var = df.columns[0]
df = df.loc[:,[id_var] + years]
ls_temp = list(df.iloc[:,0].isin(["Detailed split of energy consumption by subsector (ktoe)"]))
index_row_start = [i +1 for i, x in enumerate(ls_temp) if x]
ls_temp = list(df.iloc[:,0].isin(["Market shares of energy uses by subsector (%)"]))
index_row_end = [i-2 for i, x in enumerate(ls_temp) if x]
df = df.loc[range(index_row_start[0], index_row_end[0]),:]
DF_temp = {}
DF_temp["fec"] = df.copy()

# get specific data on useful energy demand (ued)
df = pd.read_excel(filepath, techs_code + "_ued")
id_var = df.columns[0]
df = df.loc[:,[id_var] + years]
ls_temp = list(df.iloc[:,0].isin(["Detailed split of useful energy demand by subsector (ktoe)"]))
index_row_start = [i +1 for i, x in enumerate(ls_temp) if x]
ls_temp = list(df.iloc[:,0].isin(["Market shares of useful energy demand by subsector (%)"]))
index_row_end = [i-2 for i, x in enumerate(ls_temp) if x]
df = df.loc[range(index_row_start[0], index_row_end[0]),:]
DF_temp["ued"] = df.copy()

# do operations for "fec","ued"
df_final_feedstock = pd.DataFrame()
for energy_demand_type in ["fec","ued"]:

    # do operations
    df = DF_temp[energy_demand_type].copy()
    
    # subset
    i = 0
    ls_temp = list(df.iloc[:,0].isin([techs[i]]))
    index_row_start = [i +1 for i, x in enumerate(ls_temp) if x]
    ls_temp = list(df.iloc[:,0].isin(["Other chemicals"]))
    index_row_end = [i -2 for i, x in enumerate(ls_temp) if x]
    df_temp = df.iloc[range(index_row_start[0],index_row_end[0]),:]
    
    # get feedstock
    ls_temp = list(df.iloc[:,0].isin(["Chemicals: Feedstock (energy used as raw material)"]))
    index_row_start = [i +1 for i, x in enumerate(ls_temp) if x][0]
    index_row_end = index_row_start + 8
    df_temp_fs = df.iloc[range(index_row_start,index_row_end),:]
    ls_temp = list(df_temp.iloc[:,0].isin(["Chemicals: Feedstock (energy used as raw material)"]))
    index_row_start = [i +1 for i, x in enumerate(ls_temp) if x][0]
    index_row_end = index_row_start + 8
    df_temp = df_temp.loc[[i not in range(index_row_start-1,index_row_end) for i in range(0,len(df_temp))],:]
    
    # separate Chemicals: Process cooling - Natural gas and biogas
    df_temp_coolinggas = df_temp.loc[df_temp.iloc[:,0] == "Chemicals: Process cooling - Natural gas and biogas",:]
    df_temp = df_temp.loc[df_temp.iloc[:,0] != "Chemicals: Process cooling - Natural gas and biogas",:]
    
    # get aggregates
    id_var = df_temp.columns[0]
    df_temp = df_temp.loc[df_temp[id_var].isin(enercarr_jrc + dict_subtechs_elec[techs[i]]),:]
    mytechs = dict_subtechs_elec[techs[i]].copy()
    mytechs.remove("Lighting")
    df_temp.loc[df_temp[id_var].isin(mytechs),id_var] = "Electricity"
    df_temp = df_temp.groupby([id_var], as_index=False).agg(sum)
    
    # add Chemicals: Process cooling - Natural gas and biogas to natural gas and biogas
    for y in years:
        df_temp.loc[df_temp[id_var] == "Natural gas and biogas",y] = \
            sum(np.array(df_temp.loc[df_temp[id_var] == "Natural gas and biogas",y]),
                np.array(df_temp_coolinggas.loc[:,y]))
    
    # take average across years
    df_temp = pd.melt(df_temp, id_vars = id_var, var_name='year')
    df_temp = df_temp.groupby([id_var], as_index=False)['value'].agg(np.mean)
    df_temp.columns = ["energy_carrier","value"]
    df_temp_fs = pd.melt(df_temp_fs, id_vars = id_var, var_name='year')
    df_temp_fs = df_temp_fs.groupby([id_var], as_index=False)['value'].agg(np.mean)
    df_temp_fs.columns = ["energy_carrier","value"]
    
    # # check
    # df_check = df_temp.copy()
    # df_check[id_var] = "total"
    # df_check = df_check.groupby([id_var], as_index=False)["value"].agg(sum)
    
    # apply aggrgates JRC
    for idx in range(0,len(enercarr_jrc)):
        df_temp.loc[df_temp["energy_carrier"] == enercarr_jrc[idx],"energy_carrier"] = enercarr_jrc_agg_map[idx]
    
    # adjust some categories with total shares
    for key in enercarr_jrc_agg_map_adj.keys():
        for idx in range(0, len(enercarr_jrc_agg_map_adj[key])):
            adj_value = df_temp.loc[df_temp["energy_carrier"] == key,"value"] * \
                np.array(df_ec.loc[df_ec["energy_carrier"] == enercarr_jrc_agg_map_adj[key][idx],"value"]) / \
                    np.array(sum(df_ec.loc[df_ec["energy_carrier"].isin(enercarr_jrc_agg_map_adj[key]),"value"]))
            df_temp = pd.concat([df_temp, pd.DataFrame({"energy_carrier" : enercarr_jrc_agg_map_adj[key][idx], 
                                                        "value" : adj_value})])
        df_temp = df_temp.loc[df_temp["energy_carrier"] != key,:]
        
    # # check
    # df_check = df_temp.copy()
    # df_check[id_var] = "total"
    # df_check = df_check.groupby([id_var], as_index=False)["value"].agg(sum)
    
    # make aggregates as in calculator
    for idx in range(0, len(enercarr_jrc_agg)):
        df_temp.loc[df_temp["energy_carrier"] == enercarr_jrc_agg[idx],"energy_carrier"] = enercarr_calc_map[idx]
    df_temp = df_temp.groupby(["energy_carrier"], as_index=False).agg(sum)
    df_temp.loc[df_temp["energy_carrier"] == "Lighting","energy_carrier"] = "lighting"
    
    # split biomass and waste
    for key in dict_adj_biomass_waste.keys():
        adj_value = df_temp.loc[df_temp["energy_carrier"] == "Biomass and waste","value"] * \
            dict_adj_biomass_waste[key]
        df_temp = pd.concat([df_temp, pd.DataFrame({"energy_carrier" : key, "value" : adj_value})])
    df_temp = df_temp.loc[df_temp["energy_carrier"] != "Biomass and waste",:]
    
    # # check
    # df_check = df_temp.copy()
    # df_check[id_var] = "total"
    # df_check = df_check.groupby([id_var], as_index=False)["value"].agg(sum)
    
    # add hydrogen
    df_temp = pd.concat([df_temp, pd.DataFrame({"energy_carrier" : ["hydrogen"], "value" : [0]})])
    
    # store
    df_temp["tech"] = techs_calc[0]
    df_exclfeedstock = df_temp.copy()
    
    # make df to check code output
    df = df_exclfeedstock.copy()
    df_tot = df.copy()
    df_po.columns = ["tech","total","unit_production"]
    df_temp = df.groupby(["tech"], as_index=False)['value'].agg(sum)
    df_temp = pd.merge(df_temp, df_po, how="left", on=["tech"])
    df_temp["ratio"] = df_temp["value"]/df_temp["total"]
    df_energyint_check = df_temp.copy()
    
    # convert toe to TWh, and divide by mega tonnes of production (objective is TWh/Mt)
    df_po_temp = df_po.copy()
    df_po_temp["total"] = df_po_temp["total"]/1000
    df_po_temp["unit_production"] = "Mt"
    df["value"] = df["value"] * ktoe_to_twh
    df["unit"] = "TWh"
    df = pd.merge(df, df_po_temp, how="left", on=["tech"])
    df["value"] = df["value"]/df["total"]
    
    # clean df
    df["variable"] = [tech + "_" + enercarr + "[" + unit + "/" + unit_production + "]" for \
                      tech, enercarr, unit, unit_production in \
                          zip(df["tech"], df["energy_carrier"], df["unit"], df["unit_production"])]
    df = df.loc[:,["variable","value","tech"]]
    
    # create chemicals secondary
    # it seems that energy consumption in post consumer recycling can differ a lot from chemical to chemial
    # so for the moment I will put it the same of chemicals primary
    # TODO: check the literature and re-do this
    df_temp = df.loc[df["tech"] == "chem-chem-tech",:]
    df_temp["tech"] = "chem-sec"
    df_temp["variable"] = [i.replace("chem-chem-tech","chem-sec") for i in df_temp["variable"]]
    df = pd.concat([df, df_temp])
    
    # make energy_demand_type
    df["energy_demand_type"] = energy_demand_type
    
    # append
    df_final = pd.concat([df_final, df])
    
    # do feedstock
    df_temp = df_temp_fs.copy()
    for idx in range(0,len(enercarr_jrc)):
        df_temp.loc[df_temp["energy_carrier"] == enercarr_jrc[idx],"energy_carrier"] = enercarr_jrc_agg_map[idx]
    for key in enercarr_jrc_agg_map_adj.keys():
        for idx in range(0, len(enercarr_jrc_agg_map_adj[key])):
            adj_value = df_temp.loc[df_temp["energy_carrier"] == key,"value"] * \
                np.array(df_ec_fs.loc[df_ec_fs["energy_carrier"] == enercarr_jrc_agg_map_adj[key][idx],"value"]) / \
                    np.array(sum(df_ec_fs.loc[df_ec_fs["energy_carrier"].isin(enercarr_jrc_agg_map_adj[key]),"value"]))
            df_temp = pd.concat([df_temp, pd.DataFrame({"energy_carrier" : enercarr_jrc_agg_map_adj[key][idx], 
                                                        "value" : adj_value})])
        df_temp = df_temp.loc[df_temp["energy_carrier"] != key,:]
    for idx in range(0, len(enercarr_jrc_agg)):
        df_temp.loc[df_temp["energy_carrier"] == enercarr_jrc_agg[idx],"energy_carrier"] = enercarr_calc_map[idx]
    df_temp = df_temp.groupby(["energy_carrier"], as_index=False).agg(sum)
    for key in dict_adj_biomass_waste.keys():
        adj_value = df_temp.loc[df_temp["energy_carrier"] == "Biomass and waste","value"] * \
            dict_adj_biomass_waste[key]
        df_temp = pd.concat([df_temp, pd.DataFrame({"energy_carrier" : key, "value" : adj_value})])
    df_temp = df_temp.loc[df_temp["energy_carrier"] != "Biomass and waste",:]
    df_temp = pd.concat([df_temp, pd.DataFrame({"energy_carrier" : ["hydrogen"], "value" : [0]})])
    df_temp.loc[df_temp["energy_carrier"] == "Diesel oil","energy_carrier"] = "liquid-ff-diesel"
    df_temp.loc[df_temp["energy_carrier"] == "Naphtha","energy_carrier"] = "liquid-ff-oil"
    df_temp = df_temp.groupby(["energy_carrier"], as_index=False).agg(sum)
    # df_check = df_temp.copy()
    # df_check[id_var] = "total"
    # df_check = df_check.groupby([id_var], as_index=False)["value"].agg(sum)
    df_temp = pd.concat([df_temp, pd.DataFrame({"energy_carrier" : ["electricity","gas-bio","liquid-bio","solid-waste","solid-bio"],
                                                "value" : [0,0,0,0,0]})])
    df_temp["tech"] = techs_calc[0]
    df_feedstock = df_temp.copy()
    df = df_feedstock.copy()
    df["value"] = df["value"] * ktoe_to_twh
    df["unit"] = "TWh"
    df = pd.merge(df, df_po_temp, how="left", on=["tech"])
    df["value"] = df["value"]/df["total"]
    df["variable"] = [tech + "_" + enercarr + "[" + unit + "/" + unit_production + "]" for \
                      tech, enercarr, unit, unit_production in \
                          zip(df["tech"], df["energy_carrier"], df["unit"], df["unit_production"])]
    df = df.loc[:,["variable","value","tech"]]
    
    # create chemicals feedstock secondary
    # it seems that energy consumption in post consumer recycling can differ a lot from chemical to chemial
    # so for the moment I will put it the same of chemicals primary
    # TODO: check the literature and re-do this
    df_temp = df.loc[df["tech"] == "chem-chem-tech",:]
    df_temp["tech"] = "chem-sec"
    df_temp["variable"] = [i.replace("chem-chem-tech","chem-sec") for i in df_temp["variable"]]
    df = pd.concat([df, df_temp])
    
    # make energy_demand_type
    df["energy_demand_type"] = energy_demand_type
    
    df_final_feedstock = pd.concat([df_final_feedstock, df])

# clean
del adj_value, df, df_ec, df_ec_fs, df_exclfeedstock, df_feedstock, df_po,\
    df_temp, df_temp_coolinggas, df_temp_fs, df_tot, i, id_var, idx, index_row_end, \
    index_row_start, key, ls_temp, techs, techs_calc, techs_code, y

####################################################################
############################# AMMONIA ##############################
####################################################################

# source: https://dechema.de/dechema_media/Downloads/Positionspapiere/Technology_study_Low_carbon_energy_and_feedstock_for_the_European_chemical_industry.pdf
# page 57 Table 11

# ['electricity', 'gas-bio', 'gas-ff-natural', 'hydrogen', 'liquid-bio', 'liquid-ff-oil', 'solid-bio', 'solid-ff-coal', 'solid-waste']

# compressors produce steam, assuming they run on electricity for simplicity
# assuming that other utilities run also on electricity

# do excluding feedstock
df = pd.DataFrame({"energy_carrier" : ['lighting','electricity', 'gas-bio', 'gas-ff-natural', 'hydrogen', 
                     'liquid-bio', 'liquid-ff-oil', 'liquid-ff-diesel', 'solid-bio', 
                     'solid-ff-coal', 'solid-waste'], 
                   "value" : [0, 0.74+5+1.7-4.3,0,0,0,
                              0,10.9,0,0,
                              0,0]})
df["value"] = df["value"] * gj_to_twh * 1000000
df["tech"] = "ammonia-tech"
df["energy_carrier"] = [tech + "_" + ec + "[TWh/Mt]" for tech, ec in zip(df["tech"], df["energy_carrier"])]
df.columns = ['variable', 'value', 'tech']
df["energy_demand_type"] = "fec"

df_final = pd.concat([df_final, df])

# do feedstock
# source for the percentages: https://www.european-calculator.eu/wp-content/uploads/2020/04/D3.1-Raw-materials-module-and-manufacturing.pdf
# page 61 Table 19. As they are same percentages across 3 carriers, I will just
# split what we have (from page 57 Table 11 of above link) among those 3 carriers (regardless of the percentage reported,
# which I am not sure on what should be applied)
energy_feedstock_gj = 21
df = pd.DataFrame({"energy_carrier" : ['lighting','electricity', 'gas-bio', 'gas-ff-natural', 'hydrogen', 
                     'liquid-bio', 'liquid-ff-oil', 'liquid-ff-diesel', 'solid-bio', 
                     'solid-ff-coal', 'solid-waste'], 
                   "value" : [0, 0, energy_feedstock_gj/3, energy_feedstock_gj/3, energy_feedstock_gj/3,
                              0, 0, 0, 0,
                              0, 0]})
df["value"] = df["value"] * gj_to_twh * 1000000
df["tech"] = "ammonia-tech"
df["energy_carrier"] = [tech + "_" + ec + "[TWh/Mt]" for tech, ec in zip(df["tech"], df["energy_carrier"])]
df.columns = ['variable', 'value', 'tech']
df["energy_demand_type"] = "fec"
df_final_feedstock = pd.concat([df_final_feedstock, df])

# # check
# sum([sum(df_final.loc[df_final["tech"] == "ammonia-tech","value"]),
#     sum(df_final_feedstock.loc[df_final_feedstock["tech"] == "ammonia-tech","value"])])
# # yes same of EUCalc

# clean
del df, energy_feedstock_gj

# make ued using same difference that there is for chemicals
df_efficiency = df_final.loc[df_final["tech"] == "chem-chem-tech",:]
df_efficiency = df_efficiency.pivot(index=['variable'], 
                                    columns=['energy_demand_type'], values="value").reset_index()
df_efficiency["efficiency"] = df_efficiency["ued"]/df_efficiency["fec"]
df_efficiency.loc[df_efficiency["variable"] == "chem-chem-tech_hydrogen[TWh/Mt]","efficiency"] = 0.5

df_final.sort_values(["tech","energy_demand_type","variable"],inplace=True)
df_final_feedstock.sort_values(["tech","energy_demand_type","variable"],inplace=True)

df_temp = df_final.loc[df_final["tech"] == "ammonia-tech",:]
df_temp["value"] = df_temp.loc[df_temp["tech"] == "ammonia-tech","value"] * df_efficiency.loc[:,"efficiency"]
df_temp["energy_demand_type"] = "ued"
df_final = pd.concat([df_final, df_temp])

df_temp = df_final_feedstock.loc[df_final_feedstock["tech"] == "ammonia-tech",:]
df_temp["value"] = df_temp.loc[df_temp["tech"] == "ammonia-tech","value"] * df_efficiency.loc[:,"efficiency"]
df_temp["energy_demand_type"] = "ued"
df_final_feedstock = pd.concat([df_final_feedstock, df_temp])

###########################################################
################ PULP AND PAPER PRODUCTION ################
###########################################################

# NOTE: EUcalc had feedstock for pulp (biomass), as JRC does not report it
# I will not do it for now.

# tech jrc code
techs_code = "PPA" # pulp, paper and printing

# techs
techs = ["Pulp production","Paper production","Printing and media reproduction"]
techs_calc = ['pulp-tech', 'paper-tech','printing-media-tech']

# this is aggregated into electricity
dict_subtechs_elec = {"Pulp production" : electric_usual + ["Pulp: Wood preparation, grinding",
                                                            "Pulp: Pulping electric", 
                                                            "Pulp: Cleaning"],
                      "Paper production" : electric_usual + ["Paper: Stock preparation - Mechanical", 
                                                             "Paper: Paper machine - Electricity",
                                                             "Paper: Product finishing - Electricity"],
                      "Printing and media reproduction" : electric_usual + ["Printing and publishing"]}


# get energy intensity
DF = get_energy_intensity(filepath = filepath, techs_code = techs_code, techs = techs, 
                          techs_calc = techs_calc, dict_subtechs_elec = dict_subtechs_elec,
                          years = years)
df = DF["energy-intensity"].copy()

# check
df_check = DF["tot"].copy()
df_check = df_check.groupby(["tech"], as_index=False)["value"].agg(sum)
# note: printing media is wrong but i will drop it so for now i do not fix the function
df_check = DF["energy-intensity-check"].copy()
df_check = df.groupby(["tech"], as_index=False)["value"].agg(np.mean)

# drop printing and media
df = df.loc[df["tech"] != "printing-media-tech",:]

# # create paper post consumer
# # source: https://ocshredding.com/blog/does-it-take-more-energy-to-produce-recycled-paper/#:~:text=According%20to%20the%20Environmental%20Paper,takes%20about%2022%20million%20BTUs.
# ec_perc_less = 0.31
# df_temp = df.loc[df["tech"] == "paper-tech",:]
# df_temp["value"] = df_temp["value"]*(1-ec_perc_less)
# df_temp["tech"] = "paper-sec-post-consumer"
# df_temp["variable"] = [i.replace("paper-tech","paper-sec-post-consumer") for i in df_temp["variable"]]
# df = pd.concat([df, df_temp])

# save
df_final = pd.concat([df_final, df])

# clean
del DF, df, df_check, df_energyint_check, techs, techs_calc, techs_code


###########################################################
################### ALUMINIUM AND COPPER ##################
###########################################################

# NOTE: I HAD TO MANUALLY ADD ROWS 33 AND 115 IN EXCEL AS OTHERWISE THE STRUCTURE WAS INCONSISTENT

# tech jrc code
techs_code = "NFM" # non ferrous metals

# techs
techs = ["Alumina production","Aluminium - primary production","Aluminium - secondary production", "Other non-ferrous metals"]
techs_calc = ['alumina-tech', 'aluminium-prim','aluminium-sec','other-nfm']

# this is aggregated into electricity
dict_subtechs_elec = {"Alumina production" : electric_usual,
                      "Aluminium - primary production" : electric_usual + ["Aluminium electrolysis (smelting)",
                                                                           "Aluminium processing - Electric",
                                                                           "Aluminium finishing - Electric"],
                      "Aluminium - secondary production" : electric_usual + ["Secondary aluminium - Electric",
                                                                             "Aluminium processing - Electric",
                                                                             "Aluminium finishing - Electric"],
                      "Other non-ferrous metals" : electric_usual + ["Metal production - Electric",
                                                                     "Metal processing - Electric",
                                                                     "Metal finishing - Electric"]}


# get energy intensity
DF = get_energy_intensity(filepath = filepath, techs_code = techs_code, techs = techs, 
                          techs_calc = techs_calc, dict_subtechs_elec = dict_subtechs_elec,
                          years = years)
df = DF["energy-intensity"].copy()

# check
df_check = DF["tot"].copy()
df_check = df_check.groupby(["tech"], as_index=False)["value"].agg(sum)
# note: printing media is wrong but i will drop it so for now i do not fix the function
df_check = DF["energy-intensity-check"].copy()
df_check = df.groupby(["tech"], as_index=False)["value"].agg(np.mean)

# drop the ones we do not use
df = df.loc[~df["tech"].isin(["alumina-tech"]),:]

# rename other-nfm as copper (I am using the energy consumption of other nfm as 
# proxy for the energy consumption of copper)
df.loc[df["tech"] == "other-nfm","tech"] = 'copper-tech'
df["variable"] = [i.replace("other-nfm",'copper-tech') for i in df["variable"]]
# sum(df.loc[df["tech"] == "copper","value"])

# create copper post consumer
# source: https://internationalcopper.org/policy-focus/climate-environment/recycling/#:~:text=Recycled%20copper%20requires%2085%20percent,production%20and%20reduces%20CO2%20emissions.
ec_perc_less = 0.85
df_temp = df.loc[df["tech"] == "copper-tech",:]
df_temp["value"] = df_temp["value"]*(1-ec_perc_less)
df_temp["tech"] = "copper-sec"
df_temp["variable"] = [i.replace("copper-tech","copper-sec") for i in df_temp["variable"]]
df = pd.concat([df, df_temp])

# put together
df_final = pd.concat([df_final, df])

# clean
del DF, df, df_check, techs, techs_calc, techs_code

###########################################################
########################## LIME ###########################
###########################################################

# source: https://www.eula.eu/wp-content/uploads/2019/02/A-Competitive-and-Efficient-Lime-Industry-Technical-report-by-Ecofys_0.pdf
# page 27

energy_consumption_gj_per_tonne = 4.25
electricity = 4.25*0.05
ec_minus_electricity = energy_consumption_gj_per_tonne - electricity

df = pd.DataFrame({"energy_carrier" : ['lighting','gas-bio','gas-ff-natural', 'hydrogen', 
                                       'liquid-bio','liquid-ff-diesel','liquid-ff-oil',
                                       'solid-bio', 'solid-ff-coal', 'solid-waste'],
                   "value" : [0, 0, 0.34, 0, 
                              0, 0.025, 0.025, # I am assuming equal split between diesel and oil
                              0.02, 0.51, 0.08]}) # I am also assuming that fossil solid fuels is coal (solid ff)
df["value"] = df["value"] * ec_minus_electricity
df = pd.concat([df, pd.DataFrame({"energy_carrier" : ["electricity"], "value" : electricity})])
df["value"] = df["value"] * gj_to_twh * 1000000
# note: same value of eucalc
df.columns = ["variable","value"]
df["variable"] = ["lime-lime_" + v + "[TWh/Mt]" for v in df["variable"]]
df["tech"] = "lime-lime"
df["energy_demand_type"] = "fec"
df.sort_values(["tech","energy_demand_type","variable"],inplace=True)
df_final = pd.concat([df_final, df])

# get ued with efficiency factor of cement
df_efficiency = df_final.loc[df_final["tech"] == "cement-dry-kiln",:]
df_efficiency = df_efficiency.pivot(index=['variable'], 
                                    columns=['energy_demand_type'], values="value").reset_index()
df_efficiency["efficiency"] = df_efficiency["ued"]/df_efficiency["fec"]
df_efficiency.loc[df_efficiency["variable"] == "cement-dry-kiln_hydrogen[TWh/Mt]","efficiency"] = 0.5
df = df.reset_index(drop=True)
df_temp = df.copy()
df_temp["value"] = df_temp.loc[df_temp["tech"] == "lime-lime","value"] * df_efficiency.loc[:,"efficiency"]
df_temp["energy_demand_type"] = "ued"
df_final = pd.concat([df_final, df_temp])

# note for recycling: we will just drop lime post consumer recycling for now
# as it does not seem feasible / largely done at the moment
# some sources here:
# https://www.buildinglimesforum.org.uk/recycle-week/
# https://www.sciencedirect.com/science/article/pii/S2352710224005035

# clean
del df, energy_consumption_gj_per_tonne, electricity, ec_minus_electricity


# remainings: 'fbt-tech', 'mae-tech', 'ois-tech', 'textiles-tech', 'tra-equip-tech', 'wwp-tech'

# note: here the unit of total production is "index", which is the value added, 
# rather than the tonnes of production ... this is the best thing I can do
# for the moment, as production data in weight is missing (and the production data
# that we have computed for calibration is still a by product of demand minus
# import plus export, and for example for fbt it would give an energy intensity
# of 29, which is a different order of magnitude than 0. something, which is 
# what a low emission sector should have).

###########################################################
################ FOOD BEVERAGES AND TOBACCO ###############
###########################################################

# tech jrc code
techs_code = "FBT" # non ferrous metals

# techs
techs = ["Food, beverages and tobacco"]
techs_calc = ['fbt-tech']

# this is aggregated into electricity
dict_subtechs_elec = {"Food, beverages and tobacco" : electric_usual + ["Food: Direct Heat - Electric",
                                                                        "Food: Direct Heat - Microwave",
                                                                        "Food: Process Heat - Electric",
                                                                        "Food: Process Heat - Microwave",
                                                                        "Food: Electric drying",
                                                                        "Food: Freeze drying",
                                                                        "Food: Microwave drying",
                                                                        "Food: Thermal cooling",
                                                                        "Food: Electric cooling",
                                                                        "Food: Electric machinery"]}

# get energy intensity
DF = get_energy_intensity(filepath = filepath, techs_code = techs_code, techs = techs, 
                          techs_calc = techs_calc, dict_subtechs_elec = dict_subtechs_elec,
                          years = years)
df = DF["energy-intensity"].copy()

# check
df_check = DF["tot"].copy()
df_check = df_check.groupby(["tech"], as_index=False)["value"].agg(sum)
df_check = DF["energy-intensity-check"].copy()
df_check = df.groupby(["tech"], as_index=False)["value"].agg(np.mean)

# note: with computed production from other database, the energy intesity here would
# be around 29, which is way too high ... so I'll simply take this index thing
# which gives something that is at least closer to a value that seems reasonable 
# (which should be low, as all of these should be low energy / emission intensity)

# save
df_final = pd.concat([df_final, df])

# clean
del DF, df, df_check, techs, techs_calc, techs_code


###########################################################
################### MACHINERY EQUIPMENT ###################
###########################################################

# tech jrc code
techs_code = "MAE" # non ferrous metals

# techs
techs = ["Machinery equipment"]
techs_calc = ['mae-tech']

# this is aggregated into electricity
dict_subtechs_elec = {"Machinery equipment" : electric_usual + ["Mach. Eq.: Electric Foundries",
                                                                "Mach. Eq.: Thermal connection",
                                                                "Mach. Eq.: Electric connection",
                                                                "Mach. Eq.: Heat treatment - Electric",
                                                                "Mach. Eq.: General machinery",
                                                                "Mach. Eq.: Product finishing"]}

# get energy intensity
DF = get_energy_intensity(filepath = filepath, techs_code = techs_code, techs = techs, 
                          techs_calc = techs_calc, dict_subtechs_elec = dict_subtechs_elec,
                          years = years)
df = DF["energy-intensity"].copy()

# check
df_check = DF["tot"].copy()
df_check = df_check.groupby(["tech"], as_index=False)["value"].agg(sum)
df_check = DF["energy-intensity-check"].copy()
df_check = df.groupby(["tech"], as_index=False)["value"].agg(np.mean)

# save
df_final = pd.concat([df_final, df])

# clean
del DF, df, df_check, techs, techs_calc, techs_code


###########################################################
################# OTHER INDUSTRIAL SECTORS ################
###########################################################

# tech jrc code
techs_code = "OIS" # other industrial sectors

# techs
techs = ["Other industrial sectors"]
techs_calc = ['ois-tech']

# this is aggregated into electricity
dict_subtechs_elec = {"Other industrial sectors" : electric_usual + ["Other Industrial sectors: Electric processing",
                                                                     "Other Industries: Electric drying",
                                                                     "Other Industries: Thermal cooling",
                                                                     "Other Industrial sectors: Diesel motors (incl. biofuels)", # this is a bad assumption as this should go in diesel and biofuels (not electricity), for now i keep it like that
                                                                     "Other Industrial sectors: Electric machinery"]}

# get energy intensity
DF = get_energy_intensity(filepath = filepath, techs_code = techs_code, techs = techs, 
                          techs_calc = techs_calc, dict_subtechs_elec = dict_subtechs_elec,
                          years = years)
df = DF["energy-intensity"].copy()

# check
df_check = DF["tot"].copy()
df_check = df_check.groupby(["tech"], as_index=False)["value"].agg(sum)
# note: there is a slight difference with total, also in excel there is this difference
# I can't find where the problem is for now so I'll leave it like that
df_check = DF["energy-intensity-check"].copy()
df_check = df.groupby(["tech"], as_index=False)["value"].agg(np.mean)

# make ois secondary
# assume: same of primary
df_temp = df.copy()
df_temp["tech"] = "ois-sec"
df_temp["variable"] = [i.replace("ois-tech","ois-sec") for i in df_temp["variable"]]
df = pd.concat([df,df_temp])

# save
df_final = pd.concat([df_final, df])

# clean
del DF, df, df_check, techs, techs_calc, techs_code

###########################################################
######################## TEXTILES #########################
###########################################################

# tech jrc code
techs_code = "TEL" # Textiles and leather

# techs
techs = ["Textiles and leather"]
techs_calc = ['textiles-tech']

# this is aggregated into electricity
dict_subtechs_elec = {"Textiles and leather" : electric_usual + ["Textiles: Electric general machinery",
                                                                 "Textiles: Electric drying",
                                                                 "Textiles: Microwave drying",
                                                                 "Textiles: Finishing Electric"]}

# get energy intensity
DF = get_energy_intensity(filepath = filepath, techs_code = techs_code, techs = techs, 
                          techs_calc = techs_calc, dict_subtechs_elec = dict_subtechs_elec,
                          years = years)
df = DF["energy-intensity"].copy()

# check
df_check = DF["tot"].copy()
df_check = df_check.groupby(["tech"], as_index=False)["value"].agg(sum)
df_check = DF["energy-intensity-check"].copy()
df_check = df.groupby(["tech"], as_index=False)["value"].agg(np.mean)

# save
df_final = pd.concat([df_final, df])

# clean
del DF, df, df_check, techs, techs_calc, techs_code

###########################################################
################### TRANSPORT EQUIPMENT ###################
###########################################################

# tech jrc code
techs_code = "TRE" # transport equipment

# techs
techs = ["Transport equipment"]
techs_calc = ['tra-equip-tech']

# this is aggregated into electricity
dict_subtechs_elec = {"Transport equipment" : electric_usual + ["Trans. Eq.: Electric Foundries",
                                                                "Trans. Eq.: Thermal connection",
                                                                "Trans. Eq.: Electric connection",
                                                                "Trans. Eq.: Heat treatment - Electric",
                                                                "Trans. Eq.: General machinery",
                                                                "Trans. Eq.: Product finishing"]}

# get energy intensity
DF = get_energy_intensity(filepath = filepath, techs_code = techs_code, techs = techs, 
                          techs_calc = techs_calc, dict_subtechs_elec = dict_subtechs_elec,
                          years = years)
df = DF["energy-intensity"].copy()

# check
df_check = DF["tot"].copy()
df_check = df_check.groupby(["tech"], as_index=False)["value"].agg(sum)
df_check = DF["energy-intensity-check"].copy()
df_check = df.groupby(["tech"], as_index=False)["value"].agg(np.mean)

# save
df_final = pd.concat([df_final, df])

# clean
del DF, df, df_check, techs, techs_calc, techs_code

###########################################################
################### WOOD AND WOOD PRODUCTS ################
###########################################################

# tech jrc code
techs_code = "WWP" # Wood and wood products

# techs
techs = ["Wood and wood products"]
techs_calc = ['wwp-tech']

# this is aggregated into electricity
dict_subtechs_elec = {"Wood and wood products" : electric_usual + ["Wood: Electric mechanical processes",
                                                                   "Wood: Electric drying",
                                                                   "Wood: Microwave drying",
                                                                   "Wood: Finishing Electric"]}

# get energy intensity
DF = get_energy_intensity(filepath = filepath, techs_code = techs_code, techs = techs, 
                          techs_calc = techs_calc, dict_subtechs_elec = dict_subtechs_elec,
                          years = years)
df = DF["energy-intensity"].copy()

# check
df_check = DF["tot"].copy()
df_check = df_check.groupby(["tech"], as_index=False)["value"].agg(sum)
df_check = DF["energy-intensity-check"].copy()
df_check = df.groupby(["tech"], as_index=False)["value"].agg(np.mean)

# for post consumer recycling
# it does not seem to be very spread at the moment (it's mostly used to be burned)
# one article is here: https://www.nature.com/articles/s41467-023-42499-6
# they say it requires less energy, but not how much less
# for the moment i will assume the same of wwp primary
df_temp = df.copy()
df_temp["tech"] = "wwp-sec"
df_temp["variable"] = [i.replace("wwp-tech","wwp-sec") for i in df_temp["variable"]]
df = pd.concat([df,df_temp])

# save
df_final = pd.concat([df_final, df])

# clean
del DF, df, df_check, techs, techs_calc, techs_code


df_final = df_final.sort_values(["tech","energy_demand_type","variable"])

# # checks
# df_efficiency = df_final.copy()
# df_efficiency = df_efficiency.pivot(index=['variable'], 
#                                     columns=['energy_demand_type'], values="value").reset_index()
# df_efficiency["efficiency"] = df_efficiency["ued"]/df_efficiency["fec"]

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

# reshape for efficiency ratios
def reshape_energy_constant(cdm):
    cdm_temp = cdm.copy()
    cdm_temp.drop("Categories2","lighting")
    for c in cdm_temp.col_labels["Categories1"]:
        cdm_temp.rename_col(c, c + "_process-heat", "Categories1")
    cdm_temp.deepen("_","Categories1")
    cdm_temp.switch_categories_order("Categories2","Categories3")
    cdm_temp[:,:,:,"electricity"] = 0
    cdm_temp1 = cdm.filter({"Categories2" : ["lighting","electricity"]})
    cdm_temp1.rename_col(["lighting","electricity"], ["lighting_electricity","elec_electricity"], "Categories2")
    cdm_temp1.deepen()
    missing = cdm_temp.col_labels["Categories3"].copy()
    missing.remove("electricity")
    for m in missing:
        cdm_temp1.add(0, "Categories3", m, dummy=True)
    cdm_temp1.sort("Categories3")
    cdm_temp.append(cdm_temp1, "Categories2")
    cdm_temp.sort("Categories2")
    return cdm_temp

####################################
##### FINAL ENERGY CONSUMPTION #####
####################################

# excluding feedstock
df_temp = df_final.loc[df_final["energy_demand_type"] == "fec",:]
tmp = create_constant(df_temp, df_temp["variable"])
cdm_enerdem_exclfeed = ConstantDataMatrix.create_from_constant(tmp, 0)
variabs = cdm_enerdem_exclfeed.col_labels["Variables"]
for v in variabs:
    cdm_enerdem_exclfeed.rename_col(v, "energy-demand-excl-feedstock_" + v, "Variables")
cdm_enerdem_exclfeed.deepen_twice()

# reshape to be used for efficiency ratios
cdm_enerdem_exclfeed_reshaped = reshape_energy_constant(cdm_enerdem_exclfeed)

# aggregate lighting and electricity, and get split
cdm_temp = cdm_enerdem_exclfeed.filter({"Categories2" : ["lighting","electricity"]})
cdm_enerdem_exclfeed.groupby({"electricity" : ["lighting","electricity"]}, "Categories2", inplace=True)
cdm_temp.append(cdm_temp.groupby({"total" : ["lighting","electricity"]}, "Categories2"),"Categories2")
cdm_temp.group_all("Categories1")
cdm_temp[...,"lighting"] = cdm_temp[...,"lighting"]/cdm_temp[...,"total"]
cdm_temp[...,"electricity"] = cdm_temp[...,"electricity"]/cdm_temp[...,"total"]
cdm_temp.drop("Categories1","total")
cdm_temp.units["energy-demand-excl-feedstock"] = "%"
df_check = cdm_temp.write_df()
cdm_enerdem_exclfeed_eleclight_split = cdm_temp.copy()

# feedstock
df_temp = df_final_feedstock.loc[df_final_feedstock["energy_demand_type"] == "fec",:]
tmp = create_constant(df_temp, df_temp["variable"])
cdm_enerdem_feedstock = ConstantDataMatrix.create_from_constant(tmp, 1)
variabs_missing = cdm_enerdem_feedstock.col_labels["Variables"]
variabs_missing = [i not in variabs_missing for i in cdm_enerdem_exclfeed.col_labels["Categories1"]]
variabs_missing = list(np.array(cdm_enerdem_exclfeed.col_labels["Categories1"])[variabs_missing])
for v in variabs_missing:
    cdm_enerdem_feedstock.add(0, "Variables", v, dummy=True)
cdm_enerdem_feedstock.sort("Variables")
cdm_enerdem_feedstock = cdm_enerdem_feedstock.flatten()
variabs = cdm_enerdem_feedstock.col_labels["Variables"]
for v in variabs:
    cdm_enerdem_feedstock.rename_col(v, "energy-demand-feedstock_" + v, "Variables")
cdm_enerdem_feedstock.deepen_twice()
cdm_enerdem_feedstock.units["energy-demand-feedstock"] = 'TWh/Mt'

# reshape to be used for efficiency ratios
cdm_enerdem_feedstock_reshaped = reshape_energy_constant(cdm_enerdem_feedstock)

# aggregate lighting and electricity
cdm_enerdem_feedstock.groupby({"electricity" : ["lighting","electricity"]}, "Categories2", inplace=True)

# put together excl feedstock and feedstock (to be used for efficiency rations)
cdm_enerdem_fec = cdm_enerdem_exclfeed_reshaped.copy()
cdm_enerdem_fec.append(cdm_enerdem_feedstock_reshaped,"Variables")
cdm_enerdem_fec.groupby({"energy-demand-fec" : ['energy-demand-excl-feedstock','energy-demand-feedstock']},
                        "Variables", inplace=True)
cdm_enerdem_fec.group_all("Categories1")

# save
CDM_energy_demand = {
    "energy-demand-excl-feedstock" : cdm_enerdem_exclfeed,
    "energy-demand-feedstock" : cdm_enerdem_feedstock,
    "energy-demand-excl-feedstock-eleclight-split" : cdm_enerdem_exclfeed_eleclight_split
    }

####################################
##### ENERGY EFFICIENCY RATIOS #####
####################################

# ued excluding feedstock
df_temp = df_final.loc[df_final["energy_demand_type"] == "ued",:]
tmp = create_constant(df_temp, df_temp["variable"])
cdm_enerdem_exclfeed = ConstantDataMatrix.create_from_constant(tmp, 0)
variabs = cdm_enerdem_exclfeed.col_labels["Variables"]
for v in variabs:
    cdm_enerdem_exclfeed.rename_col(v, "energy-demand-excl-feedstock_" + v, "Variables")
cdm_enerdem_exclfeed.deepen_twice()

# reshape
cdm_enerdem_exclfeed_reshaped = reshape_energy_constant(cdm_enerdem_exclfeed)

# ued feedstock
df_temp = df_final_feedstock.loc[df_final_feedstock["energy_demand_type"] == "ued",:]
tmp = create_constant(df_temp, df_temp["variable"])
cdm_enerdem_feedstock = ConstantDataMatrix.create_from_constant(tmp, 1)
variabs_missing = cdm_enerdem_feedstock.col_labels["Variables"]
variabs_missing = [i not in variabs_missing for i in cdm_enerdem_exclfeed.col_labels["Categories1"]]
variabs_missing = list(np.array(cdm_enerdem_exclfeed.col_labels["Categories1"])[variabs_missing])
for v in variabs_missing:
    cdm_enerdem_feedstock.add(0, "Variables", v, dummy=True)
cdm_enerdem_feedstock.sort("Variables")
cdm_enerdem_feedstock = cdm_enerdem_feedstock.flatten()
variabs = cdm_enerdem_feedstock.col_labels["Variables"]
for v in variabs:
    cdm_enerdem_feedstock.rename_col(v, "energy-demand-feedstock_" + v, "Variables")
cdm_enerdem_feedstock.deepen_twice()
cdm_enerdem_feedstock.units["energy-demand-feedstock"] = 'TWh/Mt'

# reshape to be used for efficiency ratios
cdm_enerdem_feedstock_reshaped = reshape_energy_constant(cdm_enerdem_feedstock)

# put together excl feedstock and feedstock (to be used for efficiency rations)
cdm_enerdem_ued = cdm_enerdem_exclfeed_reshaped.copy()
cdm_enerdem_ued.append(cdm_enerdem_feedstock_reshaped,"Variables")
cdm_enerdem_ued.groupby({"energy-demand-fec" : ['energy-demand-excl-feedstock','energy-demand-feedstock']},
                        "Variables", inplace=True)
cdm_enerdem_ued.group_all("Categories1")

# make ratios
cdm_enerdem_eff = cdm_enerdem_fec.copy()
cdm_enerdem_eff.array = cdm_enerdem_ued.array / cdm_enerdem_fec.array
cdm_enerdem_eff.rename_col("energy-demand-fec", "energy-efficiency", "Variables")
cdm_enerdem_eff.units["energy-efficiency"] = "%"
df_check = cdm_enerdem_eff.write_df()
df_check.melt()

# save
CDM_energy_demand["energy-efficiency"] = cdm_enerdem_eff.copy()

# save
f = os.path.join(current_file_directory, '../data/datamatrix/const_energy-demand.pickle')
with open(f, 'wb') as handle:
    pickle.dump(CDM_energy_demand, handle, protocol=pickle.HIGHEST_PROTOCOL)


# df = cdm_enerdem_exclfeed.write_df()
# df["country"] = "all"
# df_temp = pd.melt(df, id_vars = ['country'], var_name='variable')
# name = "temp.xlsx"
# df_temp.to_excel("~/Desktop/" + name)



