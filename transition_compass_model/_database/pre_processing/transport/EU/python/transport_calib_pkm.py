
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

################
##### ROAD #####
################

# get data
filter = {'geo': list(dict_iso2.keys()),
          'time' : list(range(1990,2023+1)),
          'vehicle' : ["MOTO_MOP","CAR","BUS_MCO_TRO"],
          'unit' : ['MIO_PKM']}
mapping_dim = {'Country': 'geo',
               'Years' : 'time',
               'Variables': 'vehicle'}
dm = get_data_api_eurostat_via_json("road_pa_mov", filter, mapping_dim, "mio-pkm")

# reshape
dm.rename_col(['BUS_MCO_TRO', 'CAR', 'MOTO_MOP', 'TOTAL'], ["bus","LDV","2W", "road-total"], "Variables")
dm_eu = dm.groupby({"EU27_2020" : dm.col_labels["Country"]}, "Country")
dm.append(dm_eu, "Country")
dm.sort("Country")
dm = add_missing(dm, country_codes, "Country")
dm_pkm = dm.copy()

#################
##### TRAIN #####
#################

# get data
filter = {'geo': list(dict_iso2.keys()),
          'time' : list(range(1990,2023+1)),
          'unit' : ['MIO_PKM']}
mapping_dim = {'Country': 'geo',
               'Years' : 'time',
               'Variables': 'train'}
dm = get_data_api_eurostat_via_json("rail_pa_total", filter, mapping_dim, "mio-pkm")

# reshape
dm = add_missing(dm, country_codes, "Country")
dm = add_missing(dm, years, "Years")
dm_pkm.append(dm, "Variables")


####################
##### AVIATION #####
####################

# get data
filter = {'geo': list(dict_iso2.keys()),
          'time' : list(range(1990,2023+1)),
          'tra_cov' : ["NAT_INTL_IEU27_2020","INTL_XEU27_2020"],
          'unit' : ['MIO_PKM']}
mapping_dim = {'Country': 'geo',
               'Years' : 'time',
               'Variables': 'tra_cov'}
dm = get_data_api_eurostat_via_json("avia_tppa", filter, mapping_dim, "mio-pkm")

# reshape
dm.groupby({"aviation" : ['INTL_XEU27_2020', 'NAT_INTL_IEU27_2020']}, "Variables", inplace=True)
dm = add_missing(dm, years, "Years")
dm_pkm.append(dm, "Variables")


##################
##### CHECKS #####
##################

dm_pkm.sort("Variables")
# dm_pkm.filter({"Country" : ["EU27_2020"]}).datamatrix_plot()

for v in dm_pkm.col_labels["Variables"]: dm_pkm.rename_col(v, "transport-demand-pkm_" + v, "Variables")
dm_pkm.deepen()
dm_pkm.change_unit("transport-demand-pkm", 1e6, "mio-pkm", "pkm")

for c in dict_iso2.keys():
    dm_pkm.rename_col(c,dict_iso2[c],"Country")

# note: here train includes also metrotram

DM = {}
DM["eurostat_passenger_pkm"] = dm_pkm.copy()

#############################
########## FREIGHT ##########
#############################

##################
##### TRUCKS #####
##################

# get data
filter = {'geo': list(dict_iso2.keys()),
          'time' : list(range(1990,2023+1)),
          'nst07' : ["TOTAL"],
          'tra_type' : ["TOTAL"],
          'unit' : ['MIO_TKM']}
mapping_dim = {'Country': 'geo',
               'Years' : 'time',
               'Variables': 'nst07',
               "Categories1" : 'tra_type'}
dm = get_data_api_eurostat_via_json("road_go_ta_tg", filter, mapping_dim, "mio-tkm")

# reshape
dm = dm.flatten()
dm.rename_col("TOTAL_TOTAL", "trucks", "Variables")
dm = add_missing(dm, country_codes, "Country")
dm = add_missing(dm, years, "Years")
dm_tkm = dm.copy()


####################
##### AVIATION #####
####################

# get data
filter = {'geo': list(dict_iso2.keys()),
          'time' : list(range(1990,2023+1)),
          'tra_cov' : ["NAT_INTL_IEU27_2020","INTL_XEU27_2020"],
          'unit' : ['MIO_TKM']}
mapping_dim = {'Country': 'geo',
               'Years' : 'time',
               'Variables': 'tra_cov'}
dm = get_data_api_eurostat_via_json("avia_tpgo", filter, mapping_dim, "mio-tkm")

# reshape
dm.groupby({"aviation" : ['INTL_XEU27_2020', 'NAT_INTL_IEU27_2020']}, "Variables", inplace=True)
dm = add_missing(dm, years, "Years")
dm_tkm.append(dm, "Variables")

###############
##### IWW #####
###############

# get data
filter = {'geo': list(dict_iso2.keys()),
          'time' : list(range(1990,2023+1)),
          'nst07' : ["TOTAL"],
          'tra_cov' : ["TOTAL"],
          'typpack' : ["TOTAL"],
          'unit' : ['MIO_TKM']}
mapping_dim = {'Country': 'geo',
               'Years' : 'time',
               'Variables': 'nst07',
               "Categories1" : 'tra_cov',
               "Categories2" : 'typpack'}
dm = get_data_api_eurostat_via_json("iww_go_atygo", filter, mapping_dim, "mio-tkm")

# reshape
dm = dm.flatten().flatten()
dm.rename_col("TOTAL_TOTAL_TOTAL", "IWW", "Variables")
dm = add_missing(dm, country_codes, "Country")
dm = add_missing(dm, years, "Years")
dm_tkm.append(dm, "Variables")

##################
##### MARINE #####
##################

# get data
filter = {'geo': list(dict_iso2.keys()),
          'time' : list(range(1990,2023+1)),
          'tra_cov' : ["NAT","INTL_IEU27_2020","INTL_XEU27_2020"],
          'unit' : ['MIO_TKM']}
mapping_dim = {'Country': 'geo',
               'Years' : 'time',
               'Variables': 'tra_cov'}
dm = get_data_api_eurostat_via_json("mar_tp_go", filter, mapping_dim, "mio-tkm")

# reshape
dm.groupby({"marine" : ['INTL_IEU27_2020', 'INTL_XEU27_2020', 'NAT']}, "Variables", inplace=True)
dm = add_missing(dm, country_codes, "Country")
dm = add_missing(dm, years, "Years")
dm_tkm.append(dm, "Variables")

################
##### RAIL #####
################

# get data
filter = {'geo': list(dict_iso2.keys()),
          'time' : list(range(1990,2023+1)),
          'unit' : ['MIO_TKM']}
mapping_dim = {'Country': 'geo',
               'Years' : 'time',
               'Variables': 'rail'}
dm = get_data_api_eurostat_via_json("rail_go_total", filter, mapping_dim, "mio-tkm")
# dm.filter({"Country" : ["EU27_2020"]}).datamatrix_plot()
# dm.filter({"Country" : ["DE"]}).datamatrix_plot() # will build EU27

# reshape
country_codes_temp = dm.col_labels["Country"].copy()
country_codes_temp.remove("EU27_2020")
dm_eu = dm.groupby({"EU27_2020" : country_codes_temp}, "Country")
# dm_eu.filter({"Country" : ["EU27_2020"]}).flatten().datamatrix_plot()
dm.drop("Country", "EU27_2020")
dm.append(dm_eu, "Country")
dm.sort("Country")
dm = add_missing(dm, country_codes, "Country")
dm = add_missing(dm, years, "Years")
dm_tkm.append(dm, "Variables")

##################
##### CHECKS #####
##################

dm_tkm.sort("Variables")
# dm_tkm.filter({"Country" : ["EU27_2020"]}).datamatrix_plot()

for v in dm_tkm.col_labels["Variables"]: dm_tkm.rename_col(v, "transport-demand-tkm_" + v, "Variables")
dm_tkm.deepen()
dm_tkm.change_unit("transport-demand-tkm", 1e6, "mio-tkm", "tkm")

for c in dict_iso2.keys():
    dm_tkm.rename_col(c,dict_iso2[c],"Country")

DM["eurostat_freight_tkm"] = dm_tkm.copy()


######################################################################
################################# SAVE ###############################
######################################################################

# save
f = os.path.join(current_file_directory, '../data/datamatrix/calibration_pkm_tkm.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)










