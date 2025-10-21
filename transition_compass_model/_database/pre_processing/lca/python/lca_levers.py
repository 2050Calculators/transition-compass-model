# packages
from model.common.data_matrix_class import DataMatrix
from model.common.auxiliary_functions import linear_fitting
from scipy.interpolate import interp1d
import pandas as pd
import pickle
import os
import numpy as np
import re
import warnings
#import eurostat
# from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat
warnings.simplefilter("ignore")
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

# directories
current_file_directory = os.getcwd()

# get data
filepath = "../data/psi/SPEED2ZERO. D.4.1.2. LCA data for calculator v.16.12.2024.xlsx"
df = pd.read_excel(filepath, sheet_name="Data")
df.columns

# rename
df.rename(columns={"EPFL name" : "product"},inplace=True)

# checks
len(df["product"].unique())

# fix product names
product_rename_dict = {"smartphone":"phone", "desktop computer" : "computer_desktop",
                       "TV":"tv", "fridge":"fridge", "dishwasher":"dishwasher", "oven":"oven",
                       "washing machine":"wmachine", "computer production, laptop" : "computer_laptop",
                       
                       "battery electric vehicles" : "LDV_BEV",
                       "plug-in hybrid electric vehicles - diesel" : "LDV_PHEV-diesel",
                       "plug-in hybrid electric vehicles - gasoline" : "LDV_PHEV-gasoline",
                       "fuel-cell electric vehicles" : "LDV_FCEV", 
                       "internal-combustion engine vehicles - diesel": "LDV_ICE-diesel",                     
                       "internal-combustion engine vehicles - gasoline" : "LDV_ICE-gasoline",
                       "internal-combustion engine vehicles - gas" : "LDV_ICE-gas",
                       "heavy-duty vehicles (freight trucks) light (payload = 7.5 tons), battery electric vehicles" : "HDVL_BEV",
                       "heavy-duty vehicles (freight trucks) light (payload = 7.5 tons), plug-in hybrid electric vehicles - diesel":"HDVL_PHEV-diesel",
                       "heavy-duty vehicles (freight trucks) light (payload = 7.5 tons), fuel-cell electric vehicles":"HDVL_FCEV",
                       "heavy-duty vehicles (freight trucks) light (payload = 7.5 tons), internal-combustion engine vehcles - diesel":"HDVL_ICE-diesel",
                       "heavy-duty vehicles (freight trucks) light (payload = 7.5 tons), internal-combustion engine vehcles - gas": "HDVL_ICE-gas",
                       "heavy-duty vehicles (freight trucks) medium (payload = 18 tons), battery electric vehicles":"HDVM_BEV",
                       "heavy-duty vehicles (freight trucks) medium (payload = 18 tons), plug-in hybrid electric vehicles - diesel":"HDVM_PHEV-diesel",
                       "heavy-duty vehicles (freight trucks) medium (payload = 18 tons), fuel-cell electric vehicles":"HDVM_FCEV",
                       "heavy-duty vehicles (freight trucks) medium (payload = 18 tons), internal-combustion engine vehcles - diesel":"HDVM_ICE-diesel",
                       "heavy-duty vehicles (freight trucks) medium (payload = 18 tons), internal-combustion engine vehcles - gas":"HDVM_ICE-gas",
                       "heavy-duty vehicles (freight trucks) high (payload = 32 tons), battery electric vehicles":"HDVH_BEV",
                       "heavy-duty vehicles (freight trucks) high (payload = 32 tons), plug-in hybrid electric vehicles - diesel":"HDVH_PHEV-diesel",
                       "heavy-duty vehicles (freight trucks) high (payload = 32 tons), fuel-cell electric vehicles":"HDVH_FCEV",
                       "heavy-duty vehicles (freight trucks) high (payload = 32 tons), internal-combustion engine vehcles - diesel":"HDVH_ICE-diesel",
                       "heavy-duty vehicles (freight trucks) high (payload = 32 tons), internal-combustion engine vehcles - gas":"HDVH_ICE-gas",
                       "scooter, electric, 4-11kW, NMC battery" : "2W_BEV",
                       "scooter, gasoline, 4-11kW, EURO-5" : "2W_ICE-gasoline", 
                       "passenger bus, battery electric - opportunity charging, LTO battery, 13m single deck urban bus" : "bus_BEV",
                       "passenger bus, diesel hybrid, 13m single deck urban bus, EURO-VI" : "bus_PHEV-diesel",
                       "passenger bus, fuel cell electric, 13m single deck urban bus" : "bus_FCEV",
                       "Passenger bus, diesel, 13m double deck urban bus, EURO-VI" : "bus_ICE-diesel",
                       "Passenger bus, compressed gas, 13m single deck urban bus, EURO-VI" : "bus_ICE-gas",
                       "goods wagon production" : "trains_CEV",
                       "tram production":"metrotram_mt",
                       "aircraft production, belly-freight aircraft, long haul":"planes_ICE",
                       "barge tanker production" :"ships_ICE",
                       
                       "battery, lithium-ion battery for vehicles":"battery-lion_vehicles",
                       "battery, lithium-ion battery for electronics": "battery-lion_electronics",
                       
                       "mastic asphalt production" : "road",
                       "railway track construction" : "rail",
                       "market for aluminium around steel bi-metal wire, 3.67mm external diameter" : "trolley-cables",
                       
                       "photovoltaic slanted-roof installation, 3 kWp, CIS, laminated, integrated, on roof" : "RES-solar-Pvroof_csi",
                       "photovoltaic slanted-roof installation, 3kWp, a-Si, laminated, integrated, on roof" : "RES-solar-Pvroof_asi",
                       "collector field area construction, solar thermal parabolic trough, 50 MW" : "RES-solar-csp",
                       "wind turbine construction, 2MW, onshore" : "RES-wind-onshore",
                       "wind power plant construction, 2MW, offshore, fixed parts":"RES-wind-offshore_fixed",
                       "wind power plant construction, 2MW, offshore, moving parts":"RES-wind-offshore_moving",
                       "hydropower plant construction, reservoir":"RES-other-hydroelectric_reservoir",
                       "hydropower plant construction, run-of-river":"RES-other-hydroelectric_runofriver",
                       "wave energy converter platform production":"RES-other-marine",
                       "electricity production, at hard coal-fired IGCC power plant":"fossil-coal",
                       "gas turbine construction, 10MW electrical":"fossil-gas",
                       "electricity production, oil":"fossil-oil",
                       "electricity production, nuclear, boiling water reactor":"nuclear",
                       "electricity production, deep geothermal":"RES-other-geothermal",
                       "battery production, Li-ion, NMC811, rechargeable, prismatic" : "battery-lion_general", 
                       "heat pump production, 30kW":"heat-pump",
                       "HVAC ventilator":"HVAC-ventilator",
                       "heat production, natural gas, at boiler fan burner low-NOx non-modulating <100kW":"boiler-gas_undefined",
                       "gas boiler production":"boiler-gas",
                       "solar collector system installation, Cu flat plate collector, one-family house, hot water" : "solar-collector",
                       "heat production, wood pellet, at furnace 9kW, state-of-the-art 2014" : "stove_wood",
                       
                       "pipeline, supercritical carbon dioxide" : "pipes_CO2",
                       "carbon dioxide, captured at synthetic natural gas plant, post, 200km pipeline, storage 1000m" : "deep-saline-formation_natural-gas-plant",
                       "carbon dioxide, captured from hard coal-fired power plant, post, pipeline 200km, storage 1000m" : "deep-saline-formation_coal-plant",
                       
                       "Desktop computer end of life" : "computer-EOL",
                       "Smartphone end of life, mechanical treatment":"phone-EOL",
                       "Passenger car, glider end of life, shredding":"LDV-EOL_glider",
                       "Passenger car, internal combustio engine end of life, shredding":"LDV-EOL_engine",
                       "Passenger car, electric car power train end of life, manual dismantling":"LDV-EOL_electric-power-train",
                       "Li-ion battery end of life, pyrometallurgical treatment":"battery-lion-EOL_pyromettalurgical",
                       "Li-ion battery end of life, hydrometallurgical treatment":"battery-lion-EOL_hydromettalurgical"}

for key in product_rename_dict.keys():
    df.loc[df["product"] == key,"product"] = product_rename_dict[key]

df_full = df.copy()

# TODO: as it will be for buildings, also for roads and infrastructure in general 
# run ecoinvent at material level and obtain
# estimates in combination with material composition (rather than doing at product level)
# probably you can apply this methodology for all products (so all materials in
# rows and columns). This will also solve the problems for the products that ecoinvent
# might not have, like dryer and freezer in domapp

#############################################################
######################### MATERIALS #########################
#############################################################

#################
##### CLEAN #####
#################

# select only material footprint
df = df_full.iloc[:,0:-17]

# fix materials in column names
materials = list(df.columns[6:])
for i in range(0,len(materials)):
    if "Plastic" not in materials[i]:
        materials[i]=(materials[i][0:-5]).lower().replace(" ","-")
    else:
        materials[i]=materials[i][0].lower()+(materials[i][1:-5]).replace(" ","-")
materials_dict = dict(zip(list(df.columns[6:]), materials)) 
for key in materials_dict.keys():
    df.rename(columns={key:materials_dict[key]}, inplace=True)
    
# add missing materials
# TODO: add in ecoinvent these materials as columns
missing_mat = ['cement', 'lime', 'paper', 'timber']
for m in missing_mat:
    df[m] = np.nan

# reshape
df.drop(['Ecoinvent name', 'Reference product', 'Location'], axis = 1,inplace=True)
df = pd.melt(df, ['product','Scenario_year','Reference unit'], var_name="material")

# fix units
units = df["Reference unit"].unique()
units_dict_fix = dict(zip(units, 
                          ['num', 'kg', 'meter-year', 'meter', 'KWh',
                           'MJ', 'km']))
for key in units_dict_fix.keys():
    df.loc[df["Reference unit"] == key,"Reference unit"] = units_dict_fix[key]
    
# from kg to tonnes
df["value"] = df["value"]*1e-3

# rename product
df["product"] = df["product"] + "_" + df["material"] + "[t/" + df["Reference unit"] + "]"
df = df.loc[:,["product","Scenario_year","value"]]

df_mat = df.copy()

# function to make dm
def make_dm(df, first_year, years_start = None, years_end = None, years_gap = None, 
            agg_prod_dict = None, agg_mat_dict = None, deepen_n_cat = 1):
    
    # make dm
    df["Country"] = "Switzerland"
    df["Years"] = first_year
    df = df.loc[:,["Country","Years","product","value"]]
    df = df.pivot(index=["Country","Years"], columns="product", values='value').reset_index()
    dm = DataMatrix.create_from_df(df, deepen_n_cat)

    # make all countries and years
    countries = ['EU27','Vaud']
    arr_temp = dm.array
    for c in countries:
        dm.add(arr_temp, "Country", c)
    if years_start is not None and years_end is not None and years_gap is not None:
        years_missing = list(range(years_start,years_end+years_gap,years_gap))
        arr_temp = dm.array
        for y in years_missing:
            dm.add(arr_temp, "Years", [y])
        dm.sort("Years")
        
    if agg_prod_dict is not None:
        dm.groupby(agg_prod_dict, "Variables", "mean", regex=True, inplace=True)
        
    if agg_mat_dict is not None:
        dm.groupby(agg_mat_dict, "Categories1", "sum", regex=False, inplace=True)
    
    # 'rail': 't/meter-year', 'road': 't/kg', 'trolley-cables': 't/meter'
    unit_rail = dm.units["rail"]
    unit_rail_numerator = unit_rail.split("/")[0]
    dm.change_unit('rail', factor=1e3, old_unit = unit_rail, new_unit = unit_rail_numerator + "/km")
    
    unit_road = dm.units["road"]
    unit_road_numerator = unit_road.split("/")[0]
    dm.change_unit('road', factor=1e3, old_unit = unit_road, new_unit = unit_road_numerator + "/t")
    # assumption: 25 000 t of total mass per kilometer (typical for a multilayer 2-lane road)
    dm.change_unit('road', factor=25000, old_unit=unit_road_numerator+"/t", new_unit= unit_road_numerator+'/km')
    
    unit_cable = dm.units["trolley-cables"]
    unit_cable_numerator = unit_rail.split("/")[0]
    dm.change_unit('trolley-cables', factor=1e3, old_unit = unit_cable, new_unit = unit_cable_numerator + "/km")
    
    return dm

def make_dm_fts(df, scenario, agg_prod_dict=None, agg_mat_dict=None, deepen_n_cat = 1):

    # get fts data
    df_fts = df.loc[df['Scenario_year'].isin([scenario + '_2025'])]
    dm_fts = make_dm(df_fts, 2025, deepen_n_cat = deepen_n_cat)
    df_fts_temp = df.loc[df['Scenario_year'].isin([scenario + '_2050'])]
    dm_fts_temp = make_dm(df_fts_temp, 2050, deepen_n_cat = deepen_n_cat)
    dm_fts.append(dm_fts_temp, "Years")
    dm_fts.sort("Years")
    
    # group products
    if agg_prod_dict is not None:
        dm_fts.groupby(agg_prod_dict, "Variables", "mean", regex=True,inplace=True)
        
    # group materials
    if agg_mat_dict is not None:
        dm_fts.groupby(agg_mat_dict, "Categories1", "sum", regex=False, inplace=True)
    
    # add missing years
    years_missing = list(range(2030, 2045+5, 5))
    dm_fts.add(np.nan, "Years", years_missing, dummy=True)
    dm_fts.sort("Years")
    years_fts = list(range(2025, 2050+ 5, 5))
    dm_fts = linear_fitting(dm_fts, years_fts)
    
    return dm_fts

agg_prod_dict = {"HDV_BEV" : "HDV.*_BEV", "HDV_FCEV" : "HDV.*_FCEV", 
                 "HDV_ICE-diesel" : "HDV.*_ICE-diesel", "HDV_ICE-gas" : "HDV.*_ICE-gas",
                 "HDV_PHEV-diesel" : "HDV.*_PHEV-diesel",
                 "LDV-EOL" : "LDV-EOL.*",
                 "battery-lion" : "battery-lion_.*",
                 "battery-lion-EOL" : "battery-lion-EOL.*",
                 "RES-other-hydroelectric" : "RES-other-hydroelectric.*",
                 "RES-solar-Pvroof" : "RES-solar-Pvroof.*",
                 "RES-wind-offshore" : "RES-wind-offshore.*",
                 "computer" : "computer_.*", 
                 "deep-saline-formation" : "deep-saline-formation.*",
                 "trains_CEV" : "metrotram_mt|trains_CEV"
                 }

agg_mat_dict = {'aluminium':['aluminium'],
                'copper':['copper'],
                'other':['antimony', 'arsenic', 'barium', 'cadmium', 'chromium', 'cobalt', 
                          'fibreglass', 'gallium', 'gold', 'indium', 'lead',
                          'lubricating-oil', 'magnesium', 'mercury', 'niobium', 'nylon-66', 
                          'palladium', 'platinum', 'silicon', 'silver', 'tantalum', 'tin', 
                          'titanium', 'vanadium', 'zinc',
                          'graphite', 'lithium', 'manganese', 'nickel',
                          'cerium','europium', 'gadolinium','lanthanum','neodymium','praseodymium', 'terbium', 'yttrium'],
                 # 'REEs':['cerium','europium', 'gadolinium','lanthanum','neodymium','praseodymium', 'terbium', 'yttrium'],
                 'chem' : ['plastic-PE', 'plastic-PET', 'plastic-PP', 'plastic-PU', 'plastic-PVC',
                           'rubber'],
                'steel':['steel','iron','cast-iron']}

###############
##### OTS #####
###############

# get ots data
df_ots = df_mat.loc[df_mat['Scenario_year'].isin(['SSP5-Base_2025'])]

# make dm
dm_ots = make_dm(df_ots, 2023, 1990, 2022, 1, 
                 agg_prod_dict = agg_prod_dict, agg_mat_dict = agg_mat_dict)

###############
##### FTS #####
###############

dm_fts_level1 = make_dm_fts(df_mat, "SSP5-Base", agg_prod_dict=agg_prod_dict, agg_mat_dict = agg_mat_dict)
dm_fts_level2 = make_dm_fts(df_mat, "SSP2-NPi", agg_prod_dict=agg_prod_dict, agg_mat_dict = agg_mat_dict)
dm_fts_level3 = make_dm_fts(df_mat, "SSP2-NDC", agg_prod_dict=agg_prod_dict, agg_mat_dict = agg_mat_dict)
dm_fts_level4 = make_dm_fts(df_mat, "SSP1-PkBudg1150", agg_prod_dict=agg_prod_dict, agg_mat_dict = agg_mat_dict)

DM_mat = {"ots":dm_ots,
          "fts1":dm_fts_level1,
          "fts2":dm_fts_level2,
          "fts3":dm_fts_level3,
          "fts4":dm_fts_level4}


########################################################
######################### ELSE #########################
########################################################

#################
##### CLEAN #####
#################

# select ELSE
ncol = len(df_full.columns)
df = df_full.iloc[:,[0,1,4] + list(range(ncol-17,ncol))]

# rename
dict_rename = {'Energy demand: electricity [kWh]' : "energy-demand-elec[KWh]",
               'Cumulative energy demand; non-renewable, fossil [MJ-eq]' : "energy-demand-ff[MJeq]",
               'Ecological Footprint, total [square meter-year]' : "ecological-footprint[sqm-year]",
               'Global warming potential, 100 years [kgCO2-eq]' : "gwp-100years[kgCO2eq]",
               'Water Consumption [cubic meter]' : "water-consumption[m3]", 
               'Air pollutants: PM10 [kg]' : "air-pollutant-pm10[kg]",
               'Air pollutants: PM2.5 [kg]' : "air-pollutant-pm25[kg]",
               'Air pollutants: SO2 [kg]' : "air-pollutant-so2[kg]",
               'Air pollutants: Ammonia [kg]' : "air-pollutant-ammonia[kg]",
               'Air pollutants: NOx [kg]' : "air-pollutant-nox[kg]",
               'Air pollutants: NMVOC [kg]' : "air-pollutant-nmvoc[kg]",
               'Heavy metals, to soil: Arsenic [kg]' : "heavy-metals-to-soil-arsenic[kg]",
               'Heavy metals, to soil: Cadmium [kg]' : "heavy-metals-to-soil-cadmium[kg]",
               'Heavy metals, to soil: Chromium [kg]' : "heavy-metals-to-soil-chromium[kg]",
               'Heavy metals, to soil: Lead [kg]' : "heavy-metals-to-soil-lead[kg]",
               'Heavy metals, to soil: Mercury [kg]' : "heavy-metals-to-soil-mercury[kg]",
               'Heavy metals, to soil: Nickel [kg]' : "heavy-metals-to-soil-nickel[kg]"}
for key in dict_rename.keys():
    df.rename(columns={key : dict_rename[key]},inplace=True)
    
# reshape
df = pd.melt(df, ['product','Scenario_year','Reference unit'], var_name="variable")

# fix units
units = df["Reference unit"].unique()
units_dict_fix = dict(zip(units, 
                          ['num', 'kg', 'meter-year', 'meter', 'KWh',
                           'MJ', 'km']))
for key in units_dict_fix.keys():
    df.loc[df["Reference unit"] == key,"Reference unit"] = units_dict_fix[key]

# rename product
unit_numerator = [s.split("[")[1].split("]")[0] for s in df["variable"]]
variables_without_unit = [s.split("[")[0] for s in df["variable"]]
df["product"] = df["product"] + "_" + variables_without_unit + "[" + unit_numerator + "/" + df["Reference unit"] + "]"
df = df.loc[:,["product","Scenario_year","value"]]

df_else = df.copy()

def make_dm_dict(df, pattern, deepen_n_cat = 1):
    
    # subset
    index = [bool(re.search(pattern, s)) for s in df["product"]]
    df_sub = df.loc[index,:]

    # get ots data
    df_ots = df_sub.loc[df_sub['Scenario_year'].isin(['SSP5-Base_2025'])]

    # ots
    dm_ots = make_dm(df_ots, 2023, 1990, 2022, 1, agg_prod_dict = agg_prod_dict, deepen_n_cat = deepen_n_cat)

    # fts
    dm_fts_level1 = make_dm_fts(df_sub, "SSP5-Base", agg_prod_dict=agg_prod_dict, deepen_n_cat = deepen_n_cat)
    dm_fts_level2 = make_dm_fts(df_sub, "SSP2-NPi", agg_prod_dict=agg_prod_dict, deepen_n_cat = deepen_n_cat)
    dm_fts_level3 = make_dm_fts(df_sub, "SSP2-NDC", agg_prod_dict=agg_prod_dict, deepen_n_cat = deepen_n_cat)
    dm_fts_level4 = make_dm_fts(df_sub, "SSP1-PkBudg1150", agg_prod_dict=agg_prod_dict, deepen_n_cat = deepen_n_cat)

    # dict
    DM = {"ots":dm_ots,
          "fts1":dm_fts_level1,
          "fts2":dm_fts_level2,
          "fts3":dm_fts_level3,
          "fts4":dm_fts_level4}
    
    return DM

####################
##### MAKE DMS #####
####################

# energy demand electricity
DM_ene_dem_elec = make_dm_dict(df_else, "energy-demand-elec")

# energy demand ff
DM_ene_dem_ff = make_dm_dict(df_else, "energy-demand-ff")

# ecological footprint
DM_eco = make_dm_dict(df_else, "ecological")

# global warming potential
DM_gwp = make_dm_dict(df_else, "gwp")

# water consumption
DM_water = make_dm_dict(df_else, "water")

# air pollutant
DM_air = make_dm_dict(df_else, "air-pollutant")

# metals in soil
DM_heavy_metals = make_dm_dict(df_else, "heavy-metals")


############################
##### PUT TOGETHER DMS #####
############################

# note: the groups all have different units, and different "categories", so
# the only way to put them all in one dm is to flatten at variable level

DM_all = {}

for element in ["ots","fts1","fts2","fts3","fts4"]:

    dm = DM_mat[element].copy()
    for c in dm.col_labels["Categories1"]:
        dm.rename_col(c, "material-footprint-" + c, "Categories1")
    dm = dm.flatten()
    
    list_DM = [DM_ene_dem_elec, DM_ene_dem_ff, DM_eco, DM_gwp, DM_water, DM_air, DM_heavy_metals]
    for DM in list_DM: 
        dm.append(DM[element].flatten(), "Variables")
    
    DM_all[element] = dm.copy()
    

################
##### SAVE #####
################

DM_fts = {1: DM_all["fts1"].copy(), 2: DM_all["fts2"].copy(), 3: DM_all["fts3"].copy(), 4: DM_all["fts4"].copy()}
DM = {"ots" : DM_all["ots"].copy(),
      "fts" : DM_fts}
f = '../data/datamatrix/lever_footprint.pickle'
with open(f, 'wb') as handle:
    pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ########################################
# ##### SPLIT BETWEEN PRODUCT GROUPS #####
# ########################################


# def make_prod_groups(dm):
    
#     DM = {}

#     # vehicles
#     dm_temp = dm.filter({"Variables" : ['2W_BEV', '2W_ICE-gasoline', 'HDV_BEV', 
#                                         'HDV_FCEV', 'HDV_ICE-diesel', 'HDV_ICE-gas', 
#                                         'HDV_PHEV-diesel',
#                                         'LDV_BEV', 'LDV_FCEV', 'LDV_ICE-diesel', 
#                                         'LDV_ICE-gas', 'LDV_ICE-gasoline', 'LDV_PHEV-diesel', 
#                                         'LDV_PHEV-gasoline',
#                                         'bus_BEV', 'bus_FCEV', 'bus_ICE-diesel', 'bus_ICE-gas', 
#                                         'bus_PHEV-diesel', 'planes_ICE', 'ships_ICE',
#                                         'trains_CEV']})
#     dm_temp.deepen(based_on="Variables")
#     dm_temp.switch_categories_order("Categories2","Categories1")
#     DM["vehicles"] = dm_temp.copy()
#     DM["vehicles-eol"] = dm.filter({"Variables" : ["LDV-EOL"]})
    
#     # batteries
#     DM["batteries"] = dm.filter({"Variables" : ["battery-lion"]})
#     DM["batteries-eol"] = dm.filter({"Variables" : ["battery-lion-EOL"]})
    
#     # transport infrastructure
#     DM["transport-infra"] = dm.filter({"Variables" : ['rail', 'road', 'trolley-cables']})
    
#     # energy infrastructure
#     DM["energy-infra"] = dm.filter({"Variables" : ['RES-other-geothermal', 'RES-other-hydroelectric', 
#                                                    'RES-other-marine', 'RES-solar-Pvroof', 'RES-solar-csp', 
#                                                    'RES-wind-offshore', 'RES-wind-onshore', 
#                                                    'fossil-coal', 'fossil-gas', 'fossil-oil',
#                                                    'nuclear',
                                                   
#                                                    'HVAC-ventilator', 'boiler-gas', 'heat-pump', 'solar-collector']})
#     # TODO: see how we treat things like boiler etc, if it's something we link to bld or elc, etc.
#     # for now I put them together
    
#     # ccus
#     DM["ccus"] = dm.filter({"Variables" : ["deep-saline-formation"]})
    
#     # domestic appliances
#     DM["domapp"] = dm.filter({"Variables" : ['dishwasher', 'fridge', 'oven', 'stove_wood', 'wmachine']})
    
#     # electronics
#     DM['electronics'] = ['computer', 'phone', 'tv']
#     DM['electronics-eol'] = ['computer-EOL', 'phone-EOL']
    
#     return DM


# DM_ots = make_prod_groups(dm_ots)

# DM_fts = {}
# DM_fts[1] = make_prod_groups(dm_fts_level1)
# DM_fts[2] = make_prod_groups(dm_fts_level2)
# DM_fts[3] = make_prod_groups(dm_fts_level3)
# DM_fts[4] = make_prod_groups(dm_fts_level4)

# tomorrow: re-structure this to make it fit

