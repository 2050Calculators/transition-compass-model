
# materials: ['aluminium', 'ammonia, 'cement', 'chem', 'copper', 'glass', 'lime', 'other', 'paper', 'steel', 'timber']

# packages
from model.common.data_matrix_class import DataMatrix
from model.common.auxiliary_functions import linear_fitting
from _database.pre_processing.fix_jumps import fix_jumps_in_dm
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

# directories
current_file_directory = os.getcwd()

#########################################
##### GET CLEAN DATAFRAME WITH DATA #####
#########################################



def get_prodcom_data(file_name):
    
    # get data
    # df = eurostat.get_data_df({file_name})
    filepath = os.path.join(current_file_directory, f'../data/eurostat/{file_name}.csv')
    # df.to_csv(filepath, index = False)
    df = pd.read_csv(filepath)

    # NOTE: as ds-056120 is on sold production, then we assume sold production = demand
    # import and export should be always on "sold", as a country exports what's demanded
    # and it imports what it demands.

    # get "PRODQNT", "EXPQNT", "IMPQNT", "QNTUNIT"
    variabs = ["PRODQNT", "EXPQNT", "IMPQNT", "QNTUNIT"]
    df = df.loc[df["indicators\\TIME_PERIOD"].isin(variabs),:]

    # apply mapping with our variable names
    filepath = os.path.join(current_file_directory, '../data/eurostat/PRODCOM2024_PRODCOM2023_Table.csv')
    df_map = pd.read_csv(filepath)
    df_map = df_map.rename(columns= {"PRODCOM2024_KEY" : "prccode"})
    df_map_sub = df_map.filter(items=['prccode', 'calc_industry_material',"primary_material_flag"])
    df_map_sub = df_map_sub.dropna()
    df_map_sub["calc_industry_material"].unique()
    materials = ['aluminium', 'ammonia', 'cement', 'chem', 'copper', 'glass', 'lime', 'paper', 'steel', 'timber',
                 'fbt', 'mae', 'ois', 'other', 'textiles', 'tra-equip', 'wwp']
    # note: 
    # for (food, beverages and tobacco), machinery equipment (mae)
    # transport equipment (tra-equip), textiles and leather (textiles), wood and wood products (wwp),
    # and other industries (ois  ), I am getting total production = sold production (demand) + export - import 
    # from ds-056120 rather than getting
    # directly total production from ds-056121 as ds-056121 has less data availability
    # these materials will enter as fxa. And I will also obtain total production for
    # the other materials, which can be used for calibration.
    df_map_sub = df_map_sub.loc[df_map_sub["calc_industry_material"].isin(materials),:]
    df_map_sub = df_map_sub.loc[df_map_sub["primary_material_flag"] == True,:] # this is done to avoid double counting (as for example some aluminium codes can be input of aluminium production)
    df = pd.merge(df, df_map_sub.loc[:,['prccode', 'calc_industry_material']], how="left", on=["prccode"])
    df_sub = df.loc[~df["calc_industry_material"].isnull(),:]

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
    indexes = ['prccode', 'variable', 'country', 'calc_industry_material']
    df_sub = pd.melt(df_sub, id_vars = indexes, var_name='year')

    # make unit as column
    drops = ['freq', 'decl']
    df_sub_unit.drop(drops,axis=1, inplace = True)
    indexes = ['prccode', 'variable', 'country', 'calc_industry_material']
    df_sub_unit = pd.melt(df_sub_unit, id_vars = indexes, var_name='year')
    df_sub_unit.rename(columns={"value":"unit"}, inplace = True)
    keep = ['prccode', 'country', 'calc_industry_material','year','unit']
    indexes = ['prccode', 'country', 'calc_industry_material','year']
    df_sub = pd.merge(df_sub, df_sub_unit.loc[:,keep], how="left", on=indexes)

    # fix unit
    df_sub["unit"].unique()
    df_sub["calc_industry_material"].unique()
    old_unit = ['kg ', 'p/st ', 'l ', 'l alc 100% ', 'm3 ', 'pa ', 'm2 ', 'm ', 'kg 90% sdt ',
                'kg TiO2 ', 'kg HCl ', 'kg P2O5 ', 'kg HF ', 'kg SiO2 ', 'kg SO2 ',
                'kg NaOH ', 'kg Al2O3 ', 'kg Cl ', 'kg Na2S2O5 ', 'kg Na2CO3 ',
                'kg B2O3 ', 'kg H2O2 ', 'g ', 'kg N ', 'kg act.subst ', 'km ',
                'c/k ', 'pa ', 'kg act. subst. ', 'kW ', 'l alc. 100% ', 'kg KOH ',
                'kg H2SO4 ', 'NA ', 'kg act.subst. ', 'kg F ']
    new_unit = ['kg', 'p/st', 'l', 'l alc 100%', 'm3', 'pa', 'm2', 'm', 'kg 90% sdt',
                'kg TiO2', 'kg HCl', 'kg P2O5', 'kg HF', 'kg SiO2', 'kg SO2',
                'kg NaOH', 'kg Al2O3', 'kg Cl', 'kg Na2S2O5', 'kg Na2CO3',
                'kg B2O3', 'kg H2O2', 'g', 'kg N', 'kg act.subst', 'km',
                'c/k', 'pa', 'kg act. subst.', 'kW', 'l alc. 100%', 'kg KOH',
                'kg H2SO4', 'NA', 'kg act.subst.', 'kg F']
    for i in range(0, len(old_unit)):
        df_sub.loc[df_sub["unit"] == old_unit[i],"unit"] = new_unit[i]
    df_sub["unit"].unique()
    df_sub["calc_industry_material"].unique()

    # fix value
    df_sub["value"] = [float(i) for i in df_sub["value"]]

    # order and sort
    indexes = ['country', 'variable', 'prccode', 'calc_industry_material', 'year']
    variabs = ['value', 'unit']
    df_sub = df_sub.loc[:,indexes + variabs]
    df_sub = df_sub.sort_values(by=indexes)

    # check
    df_check = df_sub.loc[df_sub["prccode"] == "24421130",:]
    df_check = df_check.loc[df_sub["country"].isin(["Germany","EU27_2020"])]
    # ok

    # aggregate by calc_industry_material
    df_sub = df_sub.reset_index()
    indexes = ['country', 'variable', 'calc_industry_material', 'year','unit']
    df_sub = df_sub.groupby(indexes, as_index=False)['value'].agg(sum)

    # keep right units
    df_sub["calc_industry_material"].unique()
    df_sub["unit"].unique()
    df_check = df_sub.loc[df_sub["calc_industry_material"] == "paper",:]
    df_check["unit"].unique()
    units_dict = {'aluminium' : ['kg'], 'ammonia' : ["kg N"], 'cement' : ["kg"], 
                  'copper' : ['kg'], 'glass' : ['kg'], 
                  'lime' : ['kg'], 'mae' : ['kg'], 'ois' : ['kg'], 'other' :['kg'],
                  'steel' : ['kg'], "textiles" : ['kg'],
                  'tra-equip' : ['kg']}
    # NOTE: for large groups of materials, we consider only kg, but 
    # we are missing other products in other categories (for example for ois, there are 
    # things under 'm2', 'm3', 'p/st', 'pa', 'c/k', 'm', 'NA')
    df_sub_temp = pd.concat([df_sub.loc[(df_sub["calc_industry_material"] == key) & \
                                   (df_sub["unit"].isin(units_dict[key])),:] \
                        for key in units_dict.keys()])
    df_sub_temp.loc[df_sub_temp["unit"] == 'kg N',"unit"] = "kg"

    # chem
    df_sub_chem = df_sub.loc[df_sub["calc_industry_material"].isin(["chem"]),:]
    df_sub_chem["unit"].unique()
    df_sub_chem.loc[df_sub_chem["unit"] == "g","value"] = df_sub_chem.loc[df_sub_chem["unit"] == "g","value"] / 1000
    df_sub_chem.loc[df_sub_chem["unit"] == "g","unit"] = "kg"
    ls_temp = ['kg Al2O3', 'kg B2O3', 'kg F', 'kg H2O2', 'kg H2SO4',
               'kg HCl', 'kg HF', 'kg KOH', 'kg N', 'kg Na2CO3', 'kg NaOH',
               'kg P2O5', 'kg SO2', 'kg SiO2', 'kg TiO2 ', 'kg act. subst.',
               'kg Cl', 'kg Na2S2O5', 'kg act.subst', 'kg act.subst.']
    for l in ls_temp:
        df_sub_chem.loc[df_sub_chem["unit"] == l,"unit"] = "kg"
    indexes = ['country', 'variable', 'calc_industry_material', 'year', 'unit']
    df_sub_chem = df_sub_chem.groupby(indexes, as_index=False)['value'].agg(sum)
    df_sub_chem = df_sub_chem.loc[df_sub_chem["unit"] == "kg",:]
    df_sub_temp = pd.concat([df_sub_temp, df_sub_chem])

    # timber
    # assumption: 1 m3 of wood weights 600 kg/m3 (between 0.55 t/m3 to 0.65 t/m3: https://www.fao.org/4/w4095e/w4095e06.htm#3.1.4%20examples%20of%20calculations%20of%20biomass%20density)
    df_sub_timber = df_sub.loc[df_sub["calc_industry_material"].isin(["timber"]),:]
    df_sub_timber["unit"].unique()
    df_sub_timber = df_sub_timber.loc[df_sub_timber["unit"] == "m3",:]
    df_sub_timber["value"] = df_sub_timber["value"] * 600
    df_sub_timber["unit"] = "kg"
    df_sub_temp = pd.concat([df_sub_temp, df_sub_timber])

    # fbt
    df_sub_fbt = df_sub.loc[df_sub["calc_industry_material"].isin(["fbt"]),:]
    df_sub_fbt["unit"].unique()
    # assumption: Pure ethanol (alcohol) has a density of approximately 0.789 kilograms per liter (kg/L) at room temperature (20°C)
    df_sub_fbt.loc[df_sub_fbt["unit"] == 'l alc. 100%',"value"] = \
        df_sub_fbt.loc[df_sub_fbt["unit"] == 'l alc. 100%',"value"] * 0.789
    df_sub_fbt.loc[df_sub_fbt["unit"] == 'l alc 100%',"value"] = \
        df_sub_fbt.loc[df_sub_fbt["unit"] == 'l alc 100%',"value"] * 0.789
    df_sub_fbt = df_sub_fbt.loc[df_sub_fbt["unit"].isin(['kg', 'l', 'l alc. 100%', 'l alc 100%']),:]
    df_sub_fbt["unit"] = "kg"
    indexes = ['country', 'variable', 'calc_industry_material', 'year', 'unit']
    df_sub_fbt = df_sub_fbt.groupby(indexes, as_index=False)['value'].agg(sum)
    df_sub_temp = pd.concat([df_sub_temp, df_sub_fbt])

    # wwp
    # assumption: 1 m3 of wood weights 600 kg/m3 (between 0.55 t/m3 to 0.65 t/m3: https://www.fao.org/4/w4095e/w4095e06.htm#3.1.4%20examples%20of%20calculations%20of%20biomass%20density)
    df_sub_wwp = df_sub.loc[df_sub["calc_industry_material"].isin(["wwp"]),:]
    df_sub_wwp["unit"].unique()
    df_sub_wwp.loc[df_sub_wwp["unit"] == "m3","value"] = df_sub_wwp.loc[df_sub_wwp["unit"] == "m3","value"] * 600
    df_sub_wwp = df_sub_wwp.loc[df_sub_wwp["unit"].isin(["kg","m3"]),:]
    df_sub_wwp.loc[df_sub_wwp["unit"] == "m3","unit"] = "kg"
    indexes = ['country', 'variable', 'calc_industry_material', 'year', 'unit']
    df_sub_wwp = df_sub_wwp.groupby(indexes, as_index=False)['value'].agg(sum)
    df_sub_temp = pd.concat([df_sub_temp, df_sub_wwp])

    # paper
    df_sub_paper = df_sub.loc[df_sub["calc_industry_material"].isin(["paper"]),:]
    df_sub_paper["unit"].unique()
    df_sub_paper.loc[df_sub_paper["unit"] == 'kg 90% sdt',"unit"] = "kg"
    df_sub_paper = df_sub_paper.groupby(indexes, as_index=False)['value'].agg(sum)
    df_sub = pd.concat([df_sub_temp, df_sub_paper])

    # sort
    indexes = ['country', 'variable', 'calc_industry_material', 'year', 'unit']
    df_sub.sort_values(by=indexes, inplace=True)

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
    df_sub["variable"] = df_sub["variable"] + "_" + df_sub["calc_industry_material"] + "[" + df_sub["unit"] + "]"
    df_sub = df_sub.rename(columns={"country": "Country", "year" : "Years"})
    drops = ["calc_industry_material","unit"]
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

    # make dm
    df_temp = df_sub.pivot(index=["Country","Years"], columns="variable", values='value').reset_index()
    dm_mat = DataMatrix.create_from_df(df_temp, 1)
    # dm_mat.datamatrix_plot(selected_cols={"Country" : ["EU27"], 
    #                                       "Variables" : ["product-export","product-import","product-demand"],
    #                                       "Categories1" : ["timber"]})
    # note: all values before 2006 will be put to nan and re-built, and timber has weird jump in 2022 for import and demand,
    # so can be generated before

    # fix names
    dm_mat.rename_col_regex("product", "material", "Variables")

    # check
    # dm_mat.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()
    
    return dm_mat

dm_mat = get_prodcom_data("ds-056120")

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

# set years before 2006 as missing
idx = dm_mat.idx
for y in range(1995, 2006+1):
    dm_mat.array[:,idx[y],:,:] = np.nan

# make zeroes as nans
dm_mat.array[dm_mat.array == 0] = np.nan

# make timber before 2022 as nan for EU27 (for some reason there is a big jump in 2022 for the checked countries)
idx = dm_mat.idx
for y in range(1995,2022):
    dm_mat.array[idx["EU27"],idx[y],idx["material-demand"],idx["timber"]] = np.nan
for y in range(1995,2022):
    dm_mat.array[idx["EU27"],idx[y],idx["material-export"],idx["timber"]] = np.nan
dm_mat.array[idx["EU27"],idx[2021],idx["material-export"],idx["timber"]] = 20000000000

# for wwp, before 2016 as nan for EU27 (same principle of timber)
idx = dm_mat.idx
for y in range(1995,2016):
    dm_mat.array[idx["EU27"],idx[y],idx["material-demand"],idx["wwp"]] = np.nan
for y in range(1995,2016):
    dm_mat.array[idx["EU27"],idx[y],idx["material-export"],idx["wwp"]] = np.nan
dm_mat.array[idx["EU27"],idx[2022],idx["material-export"],idx["wwp"]] = np.nan
for y in range(1995,2016):
    dm_mat.array[idx["EU27"],idx[y],idx["material-import"],idx["wwp"]] = np.nan
dm_mat.array[idx["EU27"],idx[2022],idx["material-import"],idx["wwp"]] = np.nan

# for tra equipment put missing before 2009
idx = dm_mat.idx
for y in range(1995,2009):
    dm_mat.array[idx["EU27"],idx[y],:,idx["tra-equip"]] = np.nan
    
# for ois put missing before 2010
idx = dm_mat.idx
for y in range(1995,2010):
    dm_mat.array[idx["EU27"],idx[y],:,idx["ois"]] = np.nan
    
# for other, put 2022-2023 as missing
idx = dm_mat.idx
for y in range(2022,2023+1):
    dm_mat.array[idx["EU27"],idx[y],:,idx["other"]] = np.nan

# check
# dm_mat.flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()

# fix jumps
dm_mat = fix_jumps_in_dm(dm_mat)

# check
# dm_mat.flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()

# # put nas for 2008 crisis when needed
# idx = dm_mat.idx
# for y in range(2007,2011+1):
#     dm_mat.array[idx["EU27"],idx[y],:,idx["copper"]] = np.nan
#     dm_mat.array[idx["EU27"],idx[y],:,idx["cement"]] = np.nan
#     dm_mat.array[idx["EU27"],idx[y],:,idx["lime"]] = np.nan
# for y in range(2007,2009+1):
#     dm_mat.array[idx["EU27"],idx[y],:,idx["steel"]] = np.nan
# dm_mat.array[idx["EU27"],idx[2018],:,idx["paper"]] = np.nan
# dm_mat.array[idx["EU27"],idx[2008],:,idx["paper"]] = np.nan
# dm_mat.array[idx["EU27"],idx[2022],:,idx["lime"]] = np.nan
# dm_mat.array[idx["EU27"],idx[2023],:,idx["lime"]] = np.nan

# flatten
dm_mat = dm_mat.flatten()

# check
# dm_mat.filter({"Country" : ["EU27"]}).datamatrix_plot()

# new variabs list
dict_new = {}

# function to adjust ots
def make_ots(dm, variable, based_on):
    dm_temp = dm.filter({"Variables" : [variable]})
    dm_temp = linear_fitting(dm_temp, years_ots, based_on=based_on, min_t0=0.1,min_tb=0.1)
    return dm_temp

dict_call = {"material-demand_aluminium" : None,
             "material-demand_ammonia" : None,
             "material-demand_cement" :  None,
             "material-demand_chem" : None,
             "material-demand_copper" :  None,
             "material-demand_fbt" : None,
             "material-demand_glass" : None,
             "material-demand_lime" : None,
             "material-demand_mae" : None,
             "material-demand_ois" : None,
             "material-demand_other" : range(2010,2018+1),
             "material-demand_paper" : range(2007,2017+1),
             "material-demand_steel" : range(2010,2018+1),
             "material-demand_textiles" : None,
             "material-demand_timber" : None,
             "material-demand_tra-equip" : None,
             "material-demand_wwp" : None,
             "material-export_aluminium" : None,
             "material-export_cement" : range(2012,2014+1),
             "material-export_chem" : None,
             "material-export_copper" : None,
             "material-export_fbt" : None,
             "material-export_glass" : None,
             "material-export_lime" : range(2012,2018+1),
             "material-export_mae" : range(2007,2017+1),
             "material-export_ois" : range(2010,2018+1),
             "material-export_other" : None,
             "material-export_paper" : None,
             "material-export_steel" : None,
             "material-export_textiles" :None,
             "material-export_timber" : None,
             "material-export_tra-equip" : None,
             "material-export_wwp" : None,
             "material-import_aluminium" : None,
             "material-import_cement" : None,
             "material-import_chem" : None,
             "material-import_copper" : None,
             "material-import_fbt" : None,
             "material-import_glass" : None,
             "material-import_lime" : range(2014,2023+1),
             "material-import_mae" : None,
             "material-import_ois" : None,
             "material-import_other" : None,
             "material-import_paper" : None,
             "material-import_steel" : None,
             "material-import_textiles" : None,
             "material-import_timber" : None,
             "material-import_tra-equip" : None,
             "material-import_wwp" : None}

for key in dict_call.keys(): dict_new[key] = make_ots(dm_mat, key, based_on=dict_call[key])

# append
dm_mat_temp = dict_new["material-demand_aluminium"].copy()
mylist = list(dict_call.keys())
mylist.remove("material-demand_aluminium")
for v in mylist:
    dm_mat_temp.append(dict_new[v],"Variables")
dm_mat_temp.sort("Variables")
dm_mat = dm_mat_temp.copy()
dm_mat.deepen()

# check
# dm_mat_temp.filter({"Country" : ["EU27"]}).datamatrix_plot()

# # fix jumps
# dm_mat = fix_jumps_in_dm(dm_mat)

# check
# dm_mat_temp.filter({"Country" : ["EU27"]}).datamatrix_plot()

####################
##### MAKE FTS #####
####################

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
dm_mat.add(np.nan, col_label=years_fts, dummy=True, dim='Years')

# set default time window for linear trend
# assumption: best is taking longer trend possible to make predictions to 2050 (even if earlier data is generated)
baseyear_start = 1990
baseyear_end = 2023

# fill in
dm_mat = make_fts(dm_mat, "aluminium", baseyear_start, baseyear_end)
dm_mat = make_fts(dm_mat, "ammonia", baseyear_start, baseyear_end)
dm_mat = make_fts(dm_mat, "cement", 2014, 2023) # import on upward trend and export on downward trend since 2014 (demand predictions dont change much if we start from 2014)
dm_mat = make_fts(dm_mat, "chem", baseyear_start, baseyear_end)
dm_mat = make_fts(dm_mat, "copper", baseyear_start, baseyear_end)
dm_mat = make_fts(dm_mat, "fbt", baseyear_start, baseyear_end)
dm_mat = make_fts(dm_mat, "glass", 2020, 2023)
dm_mat = make_fts(dm_mat, "lime", baseyear_start, baseyear_end)
dm_mat = make_fts(dm_mat, "mae", baseyear_start, baseyear_end)
dm_mat = make_fts(dm_mat, "ois", baseyear_start, baseyear_end)
dm_mat = make_fts(dm_mat, "other", baseyear_start, baseyear_end)
dm_mat = make_fts(dm_mat, "paper", baseyear_start, baseyear_end)
dm_mat = make_fts(dm_mat, "steel", baseyear_start, baseyear_end)
dm_mat = make_fts(dm_mat, "textiles", baseyear_start, baseyear_end)
dm_mat = make_fts(dm_mat, "timber", baseyear_start, baseyear_end)
dm_mat = make_fts(dm_mat, "tra-equip", baseyear_start, baseyear_end)
dm_mat = make_fts(dm_mat, "wwp", baseyear_start, baseyear_end)

# check
# dm_mat.filter({"Country" : ["EU27"]}).datamatrix_plot()

####################################
##### MAKE MATERIAL NET IMPORT #####
####################################

# material-net-import[%] = (material-import - material-export)/material-demand

# subset for main materials
materials = ['aluminium', 'ammonia', 'cement', 'chem', 'copper', 'glass', 'lime', 'other', 'paper', 'steel', 'timber']
dm_temp = dm_mat.filter({"Categories1" : materials})

# make material-net-import[%] = (material-import - material-export)/material-demand
idx = dm_temp.idx
arr_temp = dm_temp.array
arr_net = (arr_temp[:,:,idx["material-import"],:] - arr_temp[:,:,idx["material-export"],:]) / arr_temp[:,:,idx["material-demand"],:]

# when both import and export are zero, assign a zero
arr_net[(arr_temp[:,:,idx["material-import"],:] == 0) & (arr_temp[:,:,idx["material-export"],:] == 0)] = 0
dm_temp.add(arr_net[:,:,np.newaxis,:], "Variables", "material-net-import", unit="%")

# drop
dm_temp.drop("Variables", ["material-import","material-export","material-demand"])

# store
dm_trade_netshare = dm_temp.copy()
dm_trade_netshare.sort("Categories1")

# fill in missing values for material-net-import (coming from dividing by zero)
idx = dm_trade_netshare.idx
dm_trade_netshare.array[dm_trade_netshare.array == np.inf] = np.nan
years_fitting = dm_trade_netshare.col_labels["Years"]
dm_trade_netshare = linear_fitting(dm_trade_netshare, years_fitting)
    
# # fix jumps in material-net-import
# dm_trade_netshare = fix_jumps_in_dm(dm_trade_netshare)

# make ammonia as missing
dm_trade_netshare.drop("Categories1","ammonia")
dm_trade_netshare.add(np.nan, col_label="ammonia", dummy=True, dim='Categories1')
dm_trade_netshare.sort("Categories1")

# let's cap everything to 1
dm_trade_netshare.array[dm_trade_netshare.array > 1] = 1

# check
# dm_trade_netshare.filter({"Country" : ["EU27"]}).datamatrix_plot()

####################################
##### MAKE MATERIAL PRODUCTION #####
####################################

# material-production[kg] = material-demand[kg] + material-export[kg] - material-import[kg]

dm_temp = dm_mat.copy()

# make material-production
idx = dm_temp.idx
arr_temp = dm_temp.array
arr_net = arr_temp[:,:,idx["material-demand"],:] + arr_temp[:,:,idx["material-export"],:] - \
    arr_temp[:,:,idx["material-import"],:]

# assign zero when production is negative
# material production < 0 when material import > demand + export
# when this happens, I assume that material production is zero (a country that imports a lot to the point 
# that the material net import is larger than domestic demand)
# whatever people do not consume of all this import, can be added to a measure of material stock in case
arr_net[arr_net<0] = 0

# make dm with material production
dm_temp.add(arr_net[:,:,np.newaxis,:], "Variables", "material-production", unit="kg")
dm_temp.drop("Variables", ["material-import","material-export","material-demand"])
dm_matprod = dm_temp.copy()

# # fix jumps in material-production
# dm_matprod = fix_jumps_in_dm(dm_matprod)

# # make ammonia as demand
# idx = dm_mat.idx
# arr_temp = dm_mat.array[:,:,idx["material-demand"],idx["ammonia"]]
# dm_matprod.add(arr_temp[:,:,np.newaxis,np.newaxis], col_label="ammonia", dim='Categories1', unit="kg")
# dm_matprod.sort("Categories1")

# make it in kilo tonnes
dm_matprod.array = dm_matprod.array / 1000000
dm_matprod.units["material-production"] = "kt"

# make fxa for non-modelled sectors
dm_matprod_fxa = dm_matprod.filter({"Categories1" : ["fbt","mae","ois","textiles","tra-equip", "wwp"]})

# # make calibration data
# dm_matprod_calib = dm_matprod.filter({"Years" : years_ots})
# years = list(range(1990,2023+1)) + list(range(2025,2050+5,5))
# missing = np.array(years)[[y not in dm_matprod_calib.col_labels["Years"] for y in years]].tolist()
# dm_matprod_calib.add(np.nan, "Years", missing, dummy=True)
# dm_matprod_calib.sort("Years")

# check
# dm_matprod_fxa.filter({"Country" : ["EU27"]}).datamatrix_plot()
# dm_matprod_calib.filter({"Country" : ["EU27"]}).datamatrix_plot()

#########################################################
##### MAKE CALIBRATION DATA FOR MATERIAL PRODUCTION #####
#########################################################

dm_matprod_calib = get_prodcom_data("ds-056121")
dm_matprod_calib.rename_col("material-demand", "material-production", "Variables")
dm_matprod_calib.change_unit("material-production", 1e-6, "kg", "kt")
# dm_matprod_calib.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()
# df_temp = dm_matprod_calib.filter({"Country" : ["EU27"],"Years" : [2023]}).write_df()
# df_temp = df_temp.melt(["Country","Years"])
# df_temp
# note: it could be that aluminium data is too high, but also after filtering for non primary materials it stays the same, so not sure how to change it further
# also other probably has an issue (spike post 2022), but not too sure how to deal with it
# in general, material production data seems bad, and not sure if calibrating on it is a good idea


#######################################
##### MAKE MATERIAL DEMAND OF WWP #####
#######################################

dm_temp = dm_mat.filter({"Variables" : ["material-demand"], "Categories1" : ["wwp"]})
dm_temp.change_unit('material-demand', factor=1e-3, old_unit='kg', new_unit='t')
dm_matdem_fxa = dm_temp.copy()

################
##### SAVE #####
################

# lever: trade net share
years_ots = list(range(1990,2023+1))
years_fts = list(range(2025,2055,5))
dm_ots = dm_trade_netshare.filter({"Years" : years_ots})
dm_fts = dm_trade_netshare.filter({"Years" : years_fts})
DM_fts = {1: dm_fts.copy(), 2: dm_fts.copy(), 3: dm_fts.copy(), 4: dm_fts.copy()} # for now we set all levels to be the same
DM = {"ots" : dm_ots,
      "fts" : DM_fts}
f = os.path.join(current_file_directory, '../data/datamatrix/lever_material-net-import.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)

# fxa material production
f = os.path.join(current_file_directory, '../data/datamatrix/fxa_material-production.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_matprod_fxa, handle, protocol=pickle.HIGHEST_PROTOCOL)

# fxa material demand
f = os.path.join(current_file_directory, '../data/datamatrix/fxa_material-demand.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_matdem_fxa, handle, protocol=pickle.HIGHEST_PROTOCOL)

# calib material production
f = os.path.join(current_file_directory, '../data/datamatrix/calibration_material-production.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_matprod_calib, handle, protocol=pickle.HIGHEST_PROTOCOL)
