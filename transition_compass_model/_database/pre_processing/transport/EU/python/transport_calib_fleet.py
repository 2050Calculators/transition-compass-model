
# packages
import os
import warnings
warnings.simplefilter("ignore")
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'
import numpy as np
import pickle

from _database.pre_processing.json_routine_Eurostat import get_data_json_eurostat, get_data_api_eurostat_via_json
from _database.pre_processing.routine_JRC import get_jrc_data
from transition_compass_model.model.common.auxiliary_functions import eurostat_iso2_dict, jrc_iso2_dict

# directories
current_file_directory = os.getcwd()

# get iso codes
dict_iso2 = eurostat_iso2_dict()
dict_iso2.pop('CH')  # Remove Switzerland
dict_iso2.pop('UK')  # Remove UK
dict_iso2_jrc = jrc_iso2_dict()
country_codes = list(dict_iso2.keys())

years = list(range(1990,2023+1))

def add_missing(dm, full, category, unit = None):
    
    dm_temp = dm.copy()
    
    missing = np.array(full)[[c not in dm_temp.col_labels[category]for c in full]].tolist()
    if unit is None:
        dm_temp.add(np.nan, category, missing, dummy=True)
    else:
        dm_temp.add(np.nan, category, missing, unit=unit, dummy=True)
    dm_temp.sort(category)
    
    return dm_temp

######################################################################
############################## EUROSTAT ##############################
######################################################################

###############################
########## PASSENGER ##########
###############################

vehicle_types = ['2W', 'LDV', 'aviation', 'bus', 'metrotram', 'rail']
enginge_types = ['BEV', 'CEV', 'FCEV', 'H2', 'ICE-diesel', 'ICE-gas', 'ICE-gasoline', 'PHEV-diesel', 'PHEV-gasoline', 'kerosene', 'mt', 'total']

###############
##### LDV #####
###############

# mapping
powertrain_to_eurostat = {
    
    "total" : ["TOTAL"],
    
    "BEV": [
        "ELC"
    ],

    "FCEV": [
        "HYD_FCELL"
    ],

    "ICE-gasoline": [
        "PET_X_HYB",
        "ELC_PET_HYB",
        "BIOETH"
    ],

    "ICE-diesel": [
        "DIE_X_HYB",
        "ELC_DIE_HYB",
        "BIODIE"
    ],

    "ICE-gas": [
        "GAS",
        "LPG",
        "BIFUEL"
    ],

    "PHEV-gasoline": [
        "ELC_PET_PI"
    ],

    "PHEV-diesel": [
        "ELC_DIE_PI"
    ]
}
all_codes = [
    code
    for codes in powertrain_to_eurostat.values()
    for code in codes
]

# get data
filter = {'geo': list(dict_iso2.keys()),
          'time' : list(range(1990,2023+1)),
          'mot_nrg' : all_codes,
          'unit' : ['NR']}
mapping_dim = {'Country': 'geo',
               'Years' : 'time',
               'Variables': 'mot_nrg'}
# file_path = "../data/eurostat/road_eqr_carmot__custom_19549512_jsonstat.json"
# dm_new_ldv = get_data_json_eurostat(file_path, filter, mapping_dim, unit)
dm_new_ldv = get_data_api_eurostat_via_json("road_eqr_carpda", filter, mapping_dim, "num")
dm_fleet_ldv = get_data_api_eurostat_via_json("road_eqs_carpda", filter, mapping_dim, "num")

# reshape
dm_new_ldv.groupby(powertrain_to_eurostat, "Variables", inplace=True)
for v in dm_new_ldv.col_labels["Variables"]: dm_new_ldv.rename_col(v, "new-vehicles_LDV_" + v, "Variables")
dm_new_ldv.deepen_twice()
dm_fleet_ldv.groupby(powertrain_to_eurostat, "Variables", inplace=True)
for v in dm_fleet_ldv.col_labels["Variables"]: dm_fleet_ldv.rename_col(v, "stock-vehicles_LDV_" + v, "Variables")
dm_fleet_ldv.deepen_twice()
dm_fleet_ldv.append(dm_new_ldv, "Variables")

# check
# dm_fleet_ldv.filter({"Country" : ["EU27_2020"], "Variables" : ["stock-vehicles"]}).flatten().datamatrix_plot() # EU2020 has only BEV, FCEV, PHEV-diesel, PHEV-gasoline, the rest needs to be built by aggregating
# dm_fleet_ldv.filter({"Country" : ["DE"], "Variables" : ["stock-vehicles"]}).flatten().datamatrix_plot()
# dm_fleet_ldv.filter({"Country" : ["EU27_2020"], "Variables" : ["new-vehicles"]}).flatten().datamatrix_plot() # EU2020 has BEV and FCEV, and from 2021 onwards diesel, gasoline and PHEV, the rest needs to be built by aggregating
# dm_fleet_ldv.filter({"Country" : ["DE"], "Variables" : ["stock-vehicles"]}).flatten().datamatrix_plot()

country_codes_temp = dm_fleet_ldv.col_labels["Country"].copy()
country_codes_temp.remove("EU27_2020")
dm_fleet_ldv_eu = dm_fleet_ldv.groupby({"EU27_2020" : country_codes_temp}, "Country")
# dm_fleet_ldv_eu.filter({"Country" : ["EU27_2020"]}).flatten().datamatrix_plot()
dm_fleet_ldv.drop("Country", "EU27_2020")
dm_fleet_ldv.append(dm_fleet_ldv_eu, "Country")
dm_fleet_ldv.sort("Country")

dm_fleet_ldv = add_missing(dm_fleet_ldv, enginge_types, "Categories2")
dm_fleet_ldv = add_missing(dm_fleet_ldv, years, "Years")

# check
# dm_fleet_ldv.flatten().filter({"Country" : ["EU27_2020"]}).datamatrix_plot()

dm_fleet_total = dm_fleet_ldv.copy()

##############
##### 2W #####
##############

# get data
filter = {'geo': list(dict_iso2.keys()),
          'time' : list(range(1990,2023+1)),
          'vehicle': ['MOP','MOTO'],
          'mot_nrg' : ['TOTAL','FOSS','ZEMIS'],
          'unit' : ['NR']}
mapping_dim = {'Country': 'geo',
               'Years' : 'time',
               'Variables': 'vehicle', 
               'Categories1' : 'mot_nrg'}
dm_new = get_data_api_eurostat_via_json("road_eqr_mopeds", filter, mapping_dim, "num")
dm_fleet = get_data_api_eurostat_via_json("road_eqs_mopeds", filter, mapping_dim, "num")

# dm_fleet.flatten().filter({"Country" : ["EU27_2020"]}).datamatrix_plot()

# reshape
dm_new.groupby({"2W" : ['MOP', 'MOTO']}, "Variables", inplace=True)
dm_fleet.groupby({"2W" : ['MOP', 'MOTO']}, "Variables", inplace=True)
dm_new.rename_col("2W", "new-vehicles_2W", "Variables")
dm_fleet.rename_col("2W", "stock-vehicles_2W", "Variables")
dm_new = add_missing(dm_new, years, "Years")
dm_fleet.append(dm_new, "Variables")
dm_fleet.deepen(based_on="Variables")
dm_fleet.switch_categories_order("Categories1","Categories2")
dm_fleet.rename_col(['FOSS', 'TOTAL', 'ZEMIS'], ['ICE-gasoline', 'total', 'BEV'], "Categories2")
dm_fleet = add_missing(dm_fleet, enginge_types, "Categories2")
dm_fleet = add_missing(dm_fleet, years, "Years")

# dm_fleet.flatten().filter({"Country" : ["EU27_2020"]}).datamatrix_plot()

dm_fleet_total.append(dm_fleet, "Categories1")

#################
##### BUSES #####
#################

powertrain_to_eurostat = {
    
    "total" : ["TOTAL"],
    
    "BEV": [
        "ELC"
    ],

    "FCEV": [
        "HYD_FCELL"
    ],

    "ICE-diesel": [
        "DIE_X_HYB",
        "ELC_DIE_HYB"
    ],

    "PHEV-diesel": [
        "ELC_DIE_PI"
    ],

    "ICE-gas": [
        "LPG",
        "CNG",
        "LNG"
    ],

    "ICE-gasoline": [
        "PET"   # must be used, no subcategories available
    ]
}

all_codes = [
    code
    for codes in powertrain_to_eurostat.values()
    for code in codes
]

# get data
filter = {'geo': list(dict_iso2.keys()),
          'time' : list(range(1990,2023+1)),
          'mot_nrg' : all_codes,
          'unit' : ['NR']}
mapping_dim = {'Country': 'geo',
               'Years' : 'time',
               'Variables': 'mot_nrg'}
dm_new = get_data_api_eurostat_via_json("road_eqr_busmot", filter, mapping_dim, "num")
dm_fleet = get_data_api_eurostat_via_json("road_eqs_busmot", filter, mapping_dim, "num")

# dm_fleet.filter({"Country" : ["EU27_2020"]}).datamatrix_plot()
# dm_fleet.filter({"Country" : ["DE"]}).datamatrix_plot()

# reshape
dm_new.groupby(powertrain_to_eurostat, "Variables", inplace=True)
for v in dm_new.col_labels["Variables"]: dm_new.rename_col(v, "new-vehicles_bus_" + v, "Variables")
dm_new.deepen_twice()
dm_fleet.groupby(powertrain_to_eurostat, "Variables", inplace=True)
for v in dm_fleet.col_labels["Variables"]: dm_fleet.rename_col(v, "stock-vehicles_bus_" + v, "Variables")
dm_fleet.deepen_twice()
dm_fleet.append(dm_new, "Variables")

# dm_fleet.filter({"Country" : ["EU27_2020"]}).flatten().datamatrix_plot()
# dm_fleet.filter({"Country" : ["DE"]}).flatten().datamatrix_plot()

country_codes_temp = dm_fleet.col_labels["Country"].copy()
country_codes_temp.remove("EU27_2020")
dm_fleet_eu = dm_fleet.groupby({"EU27_2020" : country_codes_temp}, "Country")
# dm_fleet.filter({"Country" : ["EU27_2020"]}).flatten().datamatrix_plot()
dm_fleet.drop("Country", "EU27_2020")
dm_fleet.append(dm_fleet_eu, "Country")
dm_fleet.sort("Country")

dm_fleet = add_missing(dm_fleet, enginge_types, "Categories2")
dm_fleet = add_missing(dm_fleet, years, "Years")

# dm_fleet.flatten().filter({"Country" : ["EU27_2020"]}).datamatrix_plot()

dm_fleet_total.append(dm_fleet, "Categories1")

#####################
##### METROTRAM #####
#####################

# get data
filter = {'geo': list(dict_iso2.keys()),
          'time' : list(range(1990,2023+1)),
          'unit' : ['NR']}
mapping_dim = {'Country': 'geo',
               'Years' : 'time',
               'Variables': 'metrotram'}
dm_new = get_data_api_eurostat_via_json("road_eqr_trams", filter, mapping_dim, "num")
dm_fleet = get_data_api_eurostat_via_json("road_eqs_trams", filter, mapping_dim, "num")

# reshape
dm_new = add_missing(dm_new, country_codes, "Country") # here i only have 12 countries so EU27 will be missing
dm_fleet_eu = dm_fleet.groupby({"EU27_2020" : dm_fleet.col_labels["Country"]}, "Country")
dm_fleet.append(dm_fleet_eu, "Country")
dm_fleet.sort("Country")
dm_fleet = add_missing(dm_fleet, country_codes, "Country")
dm_new.rename_col("metrotram", "new-vehicles_metrotram_CEV", "Variables")
dm_new.deepen_twice()
dm_fleet.rename_col("metrotram", "stock-vehicles_metrotram_CEV", "Variables")
dm_fleet.deepen_twice()
dm_new = add_missing(dm_new, years, "Years")
dm_fleet.append(dm_new, "Variables")
dm_fleet = add_missing(dm_fleet, enginge_types, "Categories2")
dm_fleet_total.append(dm_fleet, "Categories1")

################
##### RAIL #####
################

# get data
filter = {'geo': list(dict_iso2.keys()),
          'time' : list(range(1990,2023+1)),
          'mot_nrg' : ['DIE','ELC','TOTAL'],
          'vehicle' : ['LOC'], # assumption 1 locomotive = 1 train
          'unit' : ['NR']}
mapping_dim = {'Country': 'geo',
               'Years' : 'time',
               'Variables': 'vehicle',
               'Categories1' : 'mot_nrg'}
dm_fleet = get_data_api_eurostat_via_json("rail_eq_locon", filter, mapping_dim, "num")

# reshape
dm_fleet.rename_col(['DIE', 'ELC', 'TOTAL'], ['ICE-diesel', 'CEV', 'total'], "Categories1")
dm_fleet.rename_col('LOC', 'stock-vehicles_train', "Variables")
dm_fleet.deepen(based_on="Variables")
dm_fleet.switch_categories_order("Categories1","Categories2")
dm_fleet_eu = dm_fleet.groupby({"EU27_2020" : dm_fleet.col_labels["Country"]}, "Country") # I have 25 countries, I assume fine to aggregate to EU27
dm_fleet.append(dm_fleet_eu, "Country")
dm_fleet.sort("Country")
dm_fleet.add(np.nan, "Variables","new-vehicles", "num", True)
dm_fleet = add_missing(dm_fleet, country_codes, "Country")
dm_fleet = add_missing(dm_fleet, enginge_types, "Categories2")
# dm_fleet.flatten().filter({"Country" : ["EU27_2020"]}).datamatrix_plot()
dm_fleet_total.append(dm_fleet, "Categories1")

####################
##### AVIATION #####
####################

# get data
filter = {'geo': list(dict_iso2.keys()),
          'time' : list(range(1990,2023+1)),
          'airc_cat' : ['PAS'], # this does not include private jets
          'unit' : ['NR']}
mapping_dim = {'Country': 'geo',
               'Years' : 'time',
               'Variables': 'aviation'}
dm_fleet = get_data_api_eurostat_via_json("avia_eq_arc_typ", filter, mapping_dim, "num")

# reshape
dm_fleet.rename_col('aviation', 'stock-vehicles_aviation_kerosene', "Variables")
dm_fleet.deepen_twice()
dm_fleet_eu = dm_fleet.groupby({"EU27_2020" : dm_fleet.col_labels["Country"]}, "Country") # I have 25 countries, I assume fine to aggregate to EU27
dm_fleet.append(dm_fleet_eu, "Country")
dm_fleet.sort("Country")
dm_fleet.add(np.nan, "Variables","new-vehicles", "num", True)
dm_fleet = add_missing(dm_fleet, country_codes, "Country")
dm_fleet = add_missing(dm_fleet, enginge_types, "Categories2")
dm_fleet = add_missing(dm_fleet, years, "Years")
# dm_fleet.flatten().filter({"Country" : ["EU27_2020"]}).datamatrix_plot()
dm_fleet_total.append(dm_fleet, "Categories1")


###############
##### FIX #####
###############

# fix
dm_fleet_total.sort("Categories1")
for c in dict_iso2.keys():
    dm_fleet_total.rename_col(c,dict_iso2[c],"Country")
dm_fleet_total.sort("Country")
arr_temp = dm_fleet_total["EU27",...]
arr_temp[arr_temp == 0] = np.nan
dm_fleet_total["EU27",...] = arr_temp
for v in dm_fleet_total.col_labels["Variables"]: dm_fleet_total.change_unit(v, 1, "num","number")

# check
# dm_fleet_total.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot(stacked=True)

# since train is both passenger and freight, I save it separate
dm_fleet_total_rail = dm_fleet_total.filter({"Categories1" : ["train"]})
dm_fleet_total_rail.rename_col("train","rail","Categories1")
dm_fleet_total.drop("Categories1", "train")

# since aviation in the model is seats and here is number of vehicles, I separate it for now
dm_fleet_aviation = dm_fleet_total.filter({"Categories1" : ["aviation"]})
dm_fleet_total.drop("Categories1", "aviation")

DM_fleet = {}
DM_fleet["eurostat_passenger"] = dm_fleet_total.copy()
DM_fleet["eurostat_passenger_aviation_number_of_planes"] = dm_fleet_aviation.copy()
DM_fleet["eurostat_total_rail"] = dm_fleet_total_rail.copy()

#############################
########## FREIGHT ##########
#############################

vehicle_types = ['trucks', 'IWW', 'aviation', 'marine', 'rail']
enginge_types = ['BEV', 'CEV', 'FCEV', 'H2', 'ICE', 'ICE-diesel', 'ICE-gas', 'ICE-gasoline', 'PHEV-diesel', 'PHEV-gasoline', 'kerosene', 'total']

##################
##### TRUCKS #####
##################

['ALT', 'BIODIE', 'BIOETH', 'CNG', 'DIE', 'DIE_X_HYB', 'ELC',
       'ELC_DIE_HYB', 'ELC_PET_HYB', 'GAS', 'LNG', 'LPG', 'OTH', 'PET',
       'PET_X_HYB', 'TOTAL']

powertrain_to_eurostat_trucks = {
    
    "total" : ["TOTAL"],
    
    "BEV": [
        "ELC"
    ],

    "ICE-diesel": [
        "DIE_X_HYB",
        "ELC_DIE_HYB",
        "BIODIE"
    ],

    "ICE-gasoline": [
        "PET_X_HYB",
        "ELC_PET_HYB",
        "BIOETH"
    ],

    "ICE-gas": [
        "CNG",
        "LNG",
        "LPG"
    ]
}

all_codes = [
    code
    for codes in powertrain_to_eurostat_trucks.values()
    for code in codes
]

# get data
filter = {'geo': list(dict_iso2.keys()),
          'time' : list(range(1990,2023+1)),
          'mot_nrg' : all_codes,
          'vehicle' : ["LOR_GT3P5","VG_LE3P5"],
          'unit' : ['NR']}
mapping_dim = {'Country': 'geo',
               'Years' : 'time',
               'Variables' : 'vehicle',
               'Categories1': 'mot_nrg'}
dm_new = get_data_api_eurostat_via_json("road_eqr_lormot", filter, mapping_dim, "num")
dm_fleet = get_data_api_eurostat_via_json("road_eqs_lormot", filter, mapping_dim, "num")

# dm_fleet.filter({"Country" : ["EU27_2020"]}).datamatrix_plot()
# dm_fleet.filter({"Country" : ["DE"]}).datamatrix_plot()

# reshape
dm_new.groupby(powertrain_to_eurostat_trucks, "Categories1", inplace=True)
dm_new.groupby({'new-vehicles_trucks' : ['LOR_GT3P5', 'VG_LE3P5']}, "Variables", inplace=True)
dm_new.deepen(based_on="Variables")
dm_new.switch_categories_order("Categories1", "Categories2")
dm_fleet.groupby(powertrain_to_eurostat_trucks, "Categories1", inplace=True)
dm_fleet.groupby({'stock-vehicles_trucks' : ['LOR_GT3P5', 'VG_LE3P5']}, "Variables", inplace=True)
dm_fleet.deepen(based_on="Variables")
dm_fleet.switch_categories_order("Categories1", "Categories2")
dm_fleet.append(dm_new, "Variables")
country_codes_temp = dm_fleet.col_labels["Country"].copy()
country_codes_temp.remove("EU27_2020")
dm_fleet_eu = dm_fleet.groupby({"EU27_2020" : country_codes_temp}, "Country")
# dm_fleet_eu.filter({"Country" : ["EU27_2020"]}).flatten().datamatrix_plot()
dm_fleet.drop("Country", "EU27_2020")
dm_fleet.append(dm_fleet_eu, "Country")
dm_fleet.sort("Country")

dm_fleet = add_missing(dm_fleet, enginge_types, "Categories2")
dm_fleet = add_missing(dm_fleet, years, "Years")

# check
# dm_fleet_ldv.flatten().filter({"Country" : ["EU27_2020"]}).datamatrix_plot()

dm_fleet_total = dm_fleet.copy()


################
##### RAIL #####
################

# in eurostat not possible to split passenger and freight, so I will create a new dm with train
# for both and that's it

####################
##### AVIATION #####
####################

# in eurostat no data on fleet / new of freight aviation

###############
##### IWW #####
###############

# get data
filter = {'geo': list(dict_iso2.keys()),
          'time' : list(range(1990,2023+1)),
          'weight' : ["TOTAL"],
          'vessel' : ['BAR_SP', 'DUM_PUSV'],
          'unit' : ['NR']}
mapping_dim = {'Country': 'geo',
               'Years' : 'time',
               'Variables' : 'weight',
               'Categories1': 'vessel'}
dm_fleet = get_data_api_eurostat_via_json("iww_eq_loadcap", filter, mapping_dim, "num")

# reshape
dm_fleet.group_all("Categories1")
dm_fleet.rename_col("TOTAL", "stock-vehicles_IWW_total", "Variables")
dm_fleet.deepen_twice()
dm_fleet_eu = dm_fleet.groupby({"EU27_2020" : dm_fleet.col_labels["Country"]}, "Country") # I have 18 countries, major are in, I assume I can make EU27
dm_fleet.append(dm_fleet_eu, "Country")
dm_fleet.sort("Country")

dm_fleet.add(np.nan, "Variables", "new-vehicles", "num", True)
dm_fleet.sort("Variables")
dm_fleet = add_missing(dm_fleet, enginge_types, "Categories2")
dm_fleet = add_missing(dm_fleet, country_codes, "Country")

# check
# dm_fleet.flatten().filter({"Country" : ["EU27_2020"]}).datamatrix_plot()

dm_fleet_total.append(dm_fleet, "Categories1")

##################
##### MARINE #####
##################

# in eurostat no data on fleet / new of freight marine

###############
##### FIX #####
###############

# fix
dm_fleet_total.sort("Categories1")
for c in dict_iso2.keys():
    dm_fleet_total.rename_col(c,dict_iso2[c],"Country")
dm_fleet_total.sort("Country")
arr_temp = dm_fleet_total["EU27",...]
arr_temp[arr_temp == 0] = np.nan
dm_fleet_total["EU27",...] = arr_temp
for v in dm_fleet_total.col_labels["Variables"]: dm_fleet_total.change_unit(v, 1, "num", "number")

# check
# dm_fleet_total.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot(stacked=True)

DM_fleet["eurostat_freight"] = dm_fleet_total.copy()


######################################################################
################################# JRC ################################
######################################################################

###############################
########## PASSENGER ##########
###############################

f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/jrc_passenger_fleet_calib.pickle')
with open(f, 'rb') as handle: dm_jrc_pass_fleet = pickle.load(handle)

f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/jrc_passenger_new_calib.pickle')
with open(f, 'rb') as handle: dm_jrc_pass_new = pickle.load(handle)

for v in dm_jrc_pass_fleet.col_labels["Variables"]: dm_jrc_pass_fleet.rename_col(v, "stock-vehicles_" + v, "Variables")
dm_jrc_pass_fleet.deepen(based_on="Variables")
dm_jrc_pass_fleet.switch_categories_order("Categories1","Categories2")
for v in dm_jrc_pass_new.col_labels["Variables"]: dm_jrc_pass_new.rename_col(v, "new-vehicles_" + v, "Variables")
dm_jrc_pass_new.deepen(based_on="Variables")
dm_jrc_pass_new.switch_categories_order("Categories1","Categories2")
dm_jrc_pass_fleet.append(dm_jrc_pass_new, "Variables")
dm_jrc_pass_fleet.change_unit("stock-vehicles", 1, "vehicles", "number")

# check
# dm_jrc_pass_fleet.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot(stacked=True)

DM_fleet["jrc_passenger"] = dm_jrc_pass_fleet.copy()

#############################
########## FREIGHT ##########
#############################

# fleet from pre processing of freight
f = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/jrc_freight_fleet_calib.pickle')
with open(f, 'rb') as handle: dm_jrc_freight_fleet = pickle.load(handle)

vehicle_types = ['trucks', 'IWW', 'aviation', 'marine', 'rail']
enginge_types = ['BEV', 'CEV', 'FCEV', 'H2', 'ICE', 'ICE-diesel', 'ICE-gas', 'ICE-gasoline', 'PHEV-diesel', 'PHEV-gasoline', 'kerosene']

################
##### HDVL #####
################

# get data
dict_extract = {"database" : "Transport",
                "sheet" : "TrRoad_tech",
                "variable" : "New vehicle-registrations",
                "categories" : "Light commercial vehicles",
                "sheet_last_row" : "Battery electric vehicles",
                "sub_variables" : ["Gasoline engine","Diesel oil engine","LPG engine",
                                    "Natural gas engine","Battery electric vehicles"],
                "calc_names" : ["HDVL_ICE-gasoline","HDVL_ICE-diesel","HDVL_ICE-gas-lpg",
                                "HDVL_ICE-gas-natural","HDVL_BEV"]}
dm_hdvl = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# substitute 0 with nans (to avoid that zeroes get in the averages)
dm_hdvl.array[dm_hdvl.array==0] = np.nan

# aggregate gas
dm_hdvl.groupby({"HDVL_ICE-gas" : ["HDVL_ICE-gas-lpg","HDVL_ICE-gas-natural"]}, 
                dim='Variables', aggregation = "sum", regex=False, inplace=True)
# df_light_eu = dm_hdvl.filter({"Country" : ["EU27"]}).write_df()

# make other variables
dm_hdvl.deepen()
dm_hdvl = add_missing(dm_hdvl, enginge_types, "Categories1")

################
##### HDVH #####
################

# get data
dict_extract = {"database" : "Transport",
                "sheet" : "TrRoad_tech",
                "variable" : "New vehicle-registrations",
                "sheet_last_row" : "Heavy goods vehicles",
                "sub_variables" : ["Heavy goods vehicles"],
                "calc_names" : ["HDVH"]}
dm_hdvh = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# substitute 0 with nans (to avoid that zeroes get in the averages)
dm_hdvh.array[dm_hdvh.array==0] = np.nan

# df_heavy_eu = dm_hdvh.filter({"Country" : ["EU27"]}).write_df()

# use same ratios of HDVL to make the split between types of engines
dm_temp = dm_hdvl.flatten()
dm_temp1 = dm_temp.groupby({"HDVL" : ['HDVL_BEV', 'HDVL_ICE-diesel', 'HDVL_ICE-gas', 'HDVL_ICE-gasoline']}, 
                           dim='Variables', aggregation = "sum", regex=False, inplace=False)
dm_temp.append(dm_temp1,"Variables")
idx = dm_temp.idx
dm_temp.array = dm_temp.array/dm_temp.array[...,idx["HDVL"],np.newaxis]
dm_temp.drop("Variables",["HDVL"])
dm_temp.array = dm_temp.array * dm_hdvh.array
dm_temp.rename_col_regex("HDVL","HDVH","Variables")
dm_hdvh = dm_temp.copy()
# df_heavy_eu = dm_hdvh.filter({"Country" : ["EU27"]}).write_df()

# make other variables
dm_hdvh.deepen()
dm_hdvh = add_missing(dm_hdvh, enginge_types, "Categories1")

############################
##### AGGREGATE TRUCKS #####
############################

# put together
dm_new = dm_hdvl.copy()
dm_new.append(dm_hdvh,"Variables")
# df_check = dm_fleet.filter({"Country" : ["EU27"]}).write_df()
dm_new.groupby({"trucks" : ['HDVL', 'HDVH']}, "Variables", inplace=True)


####################
##### aviation #####
####################

import pandas as pd
from transition_compass_model.model.common.data_matrix_class import DataMatrix

def get_specific_jrc_data(country_code, country_name, row_start, row_end, unit, variable = "aviation_kerosene", 
                          database = "JRC-IDEES-2021_x1990_Aviation_EU"):
    
    filepath_jrc = os.path.join(current_file_directory, f"../../../industry/eu/data/JRC-IDEES-2021/EU27/{database}.xlsx")
    df_temp = pd.read_excel(filepath_jrc, sheet_name=country_code)
    df_temp = df_temp.iloc[row_start:row_end,:]
    indexes = df_temp.columns[0]
    df_temp = pd.melt(df_temp, id_vars = indexes, var_name='year')
    df_temp.columns = ["Country","Years",f"{variable}[{unit}]"]
    df_temp["Country"] = country_name
    
    return df_temp

# get fleet in number of planes
country_codes = list(dict_iso2_jrc.keys())
country_names = list(dict_iso2_jrc.values())
df_new_avi = pd.concat([get_specific_jrc_data(code, name, 106, 107, "unit") for code,name in zip(country_codes, country_names)],ignore_index=True)
dm_avi = DataMatrix.create_from_df(df_new_avi, 0)
dm_avi.array = np.round(dm_avi.array, 0)

# # get data
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrAvia_act",
#                 "variable" : "Stock of aircrafts - total",
#                 "sheet_last_row" : "Freight transport",
#                 "sub_variables" : ["Freight transport"],
#                 "calc_names" : ["aviation"]}
# dm_avi = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# assuming most planes are kerosene, which here I call ICE
dm_avi.deepen()

# assuming that all else is nan
dm_avi = add_missing(dm_avi, enginge_types, "Categories1")

################
##### RAIL #####
################

# get data
dict_extract = {"database" : "Transport",
                "sheet" : "TrRail_act",
                "variable" : "New vehicles - total (representative train configuration)",
                "sheet_last_row" : "Electric",
                "categories": "Freight transport",
                "sub_variables" : ["Diesel oil",
                                    "Electric"],
                "calc_names" : ["rail_ICE-diesel","rail_CEV"]}
dm_rail = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)

# make rest of the variables (assuming they are all missing for now)
dm_rail.deepen()
dm_rail = add_missing(dm_rail, enginge_types, "Categories1")

###############
##### IWW #####
###############

# get data from jrc
df_iww = pd.concat(
    [get_specific_jrc_data(code, name, 99, 100, "unit", "IWW_ICE", "JRC-IDEES-2021_x1990_Navigation_Domestic") 
     for code,name in zip(country_codes, country_names)],
    ignore_index=True)
dm_iww = DataMatrix.create_from_df(df_iww, 0)
dm_iww.array = np.round(dm_iww.array, 0)

# # get data on total fleet from eurostat
# code = "iww_eq_loadcap"
# eurostat.get_pars(code)
# filter = {'geo\\TIME_PERIOD': list(dict_iso2.keys()),
#           'vessel': ['BAR_SP'],
#           'unit' : ['NR'],
#           'weight' : ['TOTAL']}
# mapping_dim = {'Country': 'geo\\TIME_PERIOD',
#                 'Variables': 'vessel'}
# dm_iww_fleet = get_data_api_eurostat(code, filter, mapping_dim, 'num')
# dm_iww_fleet = dm_iww_fleet.filter({"Years" : list(range(1990,2023+1,1))})
# dm_iww_fleet.drop("Country","United Kingdom")
# dm_iww_fleet = dm_iww_fleet.groupby({"IWW" : ['BAR_SP']}, "Variables")
# # df = dm_iww_fleet.write_df()

# make techs (we say they are all ICE, as vessels usually are some diesel and some Heavy Fuel Oil, which we do not have)
# dm_iww_fleet.rename_col("IWW","IWW_ICE","Variables")
dm_iww.deepen()

# assuming that all else is nan
dm_iww = add_missing(dm_iww, enginge_types, "Categories1")

# # make EU27
# dm_iww_fleet.append(dm_iww_fleet.groupby({"EU27":dm_iww_fleet.col_labels["Country"]},"Country"),"Country")

# # add missing countries
# countries = dm_avi.col_labels["Country"]
# missing_countries = np.array(countries)[[c not in dm_iww_fleet.col_labels["Country"] for c in countries]]
# dm_iww_fleet.add(np.nan,"Country",missing_countries,"number",True)
# dm_iww_fleet.sort("Country")

##################
##### MARINE #####
##################

# get data from jrc
# note: I assume that inside international EU file, the voice "intra-EU27" includes also the 
# voice "coastal shipping" mentioned in the domestic EU file
df_mar = pd.concat(
    [get_specific_jrc_data(code, name, 99, 100, "unit", "marine_ICE", "JRC-IDEES-2021_x1990_Navigation_International_EU") 
     for code,name in zip(country_codes, country_names)],
    ignore_index=True)
dm_mar = DataMatrix.create_from_df(df_mar, 0)
dm_mar.array = np.round(dm_mar.array, 0)


# # get data
# df = pd.read_csv("../data/unctad/US_MerchantFleet.csv")
# df["Economy Label"].unique()
# countries = dm_avi.col_labels["Country"]
# missing_countries = np.array(countries)[[c not in df["Economy Label"].unique() for c in countries]]
# countries = countries + ['European Union (2020 …)','Czechia','Netherlands (Kingdom of the)']
# df = df.loc[df["Economy Label"].isin(countries),:]
# df = df.loc[df["ShipType Label"] == 'Total fleet',:]
# old_names = ['European Union (2020 …)','Czechia','Netherlands (Kingdom of the)']
# new_names = ["EU27", "Czech Republic", "Netherlands"]
# for o,n in zip(old_names, new_names):
#     df.loc[df["Economy Label"] == o,"Economy Label"] = n

# # make dm
# df.columns
# df = df.loc[:,["Year","Economy Label","Number of ships"]]
# df.rename(columns={"Economy Label":"Country","Year" : "Years","Number of ships":"marine[number]"},inplace=True)
# df = df.loc[df["Years"].isin(list(range(1990,2023+1))),:]
# df_temp = pd.DataFrame({"Country":np.repeat(df["Country"].unique(), len(df["Years"].unique())),
#                         "Years":np.tile(df["Years"].unique(), len(df["Country"].unique()))})
# df = df_temp.merge(df, "left", ["Country","Years"])
# dm_mar_fleet = DataMatrix.create_from_df(df, 0)

# make techs (we say they are all ICE, as vessels usually are some diesel and some Heavy Fuel Oil, which we do not have)
# dm_mar_fleet.rename_col("marine","marine_ICE","Variables")
dm_mar.deepen()

# assuming that all else is nan
dm_mar = add_missing(dm_mar, enginge_types, "Categories1")


########################
##### PUT TOGETHER #####
########################

dm_new.change_unit("trucks", 1, "unit", "num")
dm_new.add(np.nan, "Years", list(range(1990,1999+1)), "number", True)
dm_new.sort("Years")
dm_avi.sort("Years")
dm_rail.add(np.nan, "Years", list(range(1990,1999+1)), "number", True)
dm_rail.sort("Years")
dm_new.append(dm_rail,"Variables")
dm_iww.sort("Years")
dm_new.append(dm_iww,"Variables")
dm_mar.sort("Years")
dm_new.append(dm_mar,"Variables")
dm_new.append(dm_avi,"Variables")
dm_new.sort("Variables")
dm_new.sort("Country")

# substitute zero values with missing
dm_new.array[dm_new.array==0] = np.nan


########################
##### PUT TOGETHER #####
########################

dm_jrc_freight_fleet

dm_jrc_freight_fleet.groupby({"trucks" : ['HDVH', 'HDVL', 'HDVM']}, "Variables", inplace=True)
dm_fleet = dm_jrc_freight_fleet.copy()

for v in dm_fleet.col_labels["Variables"]: dm_fleet.rename_col(v, "stock-vehicles_" + v, "Variables")
dm_fleet.deepen(based_on="Variables")
dm_fleet.switch_categories_order("Categories1","Categories2")
for v in dm_new.col_labels["Variables"]: dm_new.rename_col(v, "new-vehicles_" + v, "Variables")
dm_new.deepen(based_on="Variables")
dm_new.switch_categories_order("Categories1","Categories2")
dm_fleet.append(dm_new, "Variables")

dm_fleet.array[dm_fleet.array == 0] = np.nan
for v in dm_fleet.col_labels["Variables"]: dm_fleet.change_unit(v, 1, "unit", "number")

# check
# dm_fleet.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot(stacked=True)

DM_fleet["jrc_freight"] = dm_fleet.copy()

######################################################################
################################# SAVE ###############################
######################################################################

# save
f = os.path.join(current_file_directory, '../data/datamatrix/calibration_fleet.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM_fleet, handle, protocol=pickle.HIGHEST_PROTOCOL)


