
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

vehicle_types_passenger = ['2W', 'LDV', 'aviation', 'bus', 'metrotram', 'rail']
engine_types_passenger = ['BEV', 'CEV', 'FCEV', 'H2', 'ICE-diesel', 'ICE-gas', 'ICE-gasoline', 'PHEV-diesel', 'PHEV-gasoline', 'kerosene', 'mt', 'total']
vehicle_types_freight = ['trucks', 'IWW', 'aviation', 'marine', 'rail']
engine_types_freight = ['BEV', 'CEV', 'FCEV', 'H2', 'ICE', 'ICE-diesel', 'ICE-gas', 'ICE-gasoline', 'PHEV-diesel', 'PHEV-gasoline', 'kerosene', 'total']


################
##### ROAD #####
################

vehicleclass_to_eurostat = {
    # # Two-wheelers (leaf-only)
    # "2W": [
    #     "MOP_MOTO"
    # ],

    # Light-duty vehicles (leaf-only)
    "LDV": [
        "CAR"
    ],

    # Buses/coaches (leaf-only)
    "bus": [
        "BUS"
    ],

    # Trucks (leaf-only)
    "trucks": [
        "LOR",
        "TRC"
    ]
}

all_codes = [
    code
    for codes in vehicleclass_to_eurostat.values()
    for code in codes
]

# get data
filter = {'geo': list(dict_iso2.keys()),
          'time' : list(range(1990,2023+1)),
          'regisveh' : ["TER_REGNAT"],
          'vehicle' : all_codes,
          'mot_nrg' : ['TOTAL','PET','DIE','OTH'],
          'unit' : ['MIO_VKM']}
mapping_dim = {'Country': 'geo',
               'Years' : 'time',
               'Variables': 'regisveh',
               'Categories1': 'vehicle',
               'Categories2' : 'mot_nrg'}
dm = get_data_api_eurostat_via_json("road_tf_veh", filter, mapping_dim, "mio-vkm")

# reshape
dm.groupby(vehicleclass_to_eurostat, "Categories1", inplace=True)
dm.rename_col(['DIE', 'OTH', 'PET', 'TOTAL'],["ICE-diesel","other","ICE-gasoline","total"], "Categories2")
dm.drop("Categories2","other")
dm.rename_col(['TER_REGNAT'],["demand-vkm"], "Variables")
dm = add_missing(dm, country_codes, "Country")
dm = add_missing(dm, years, "Years")
dm_road_passenger = dm.filter({"Categories1" : ["LDV"]})
dm_road_passenger.rename_col("demand-vkm","passenger-demand-vkm","Variables")
dm_road_passenger = add_missing(dm_road_passenger, engine_types_passenger, "Categories2")
dm_road_freight = dm.filter({"Categories1" : ["trucks"]})
dm_road_freight.rename_col("demand-vkm","freight-demand-vkm","Variables")
dm_road_freight = add_missing(dm_road_freight, engine_types_freight, "Categories2")
dm_passenger = dm_road_passenger.copy()
dm_freight = dm_road_freight.copy()

################
##### RAIL #####
################

# get data
filter = {'geo': list(dict_iso2.keys()),
          'time' : list(range(1990,2023+1)),
          'train' : ["TRN_GD","TRN_PAS"],
          'vehicle' : ["LOC"],
          'mot_nrg' : ['TOTAL','DIE','ELC'],
          'unit' : ['THS_TRKM']}
mapping_dim = {'Country': 'geo',
               'Years' : 'time',
               'Variables': 'train',
               'Categories1': 'vehicle',
               'Categories2' : 'mot_nrg'}
dm = get_data_api_eurostat_via_json("rail_tf_traveh", filter, mapping_dim, "ths-trkm")

# reshape
dm.rename_col(['DIE', 'ELC', 'TOTAL'],["ICE-diesel","CEV","total"], "Categories2")
dm.rename_col(['TRN_GD', 'TRN_PAS'],["freight-demand-vkm","passenger-demand-vkm"], "Variables")
dm.rename_col(['LOC'],["train"], "Categories1")
dm.change_unit("freight-demand-vkm", 1e-3, "ths-trkm", "mio-vkm")
dm.change_unit("passenger-demand-vkm", 1e-3, "ths-trkm", "mio-vkm")
dm = add_missing(dm, country_codes, "Country")
dm = add_missing(dm, years, "Years")
dm_train_passenger = dm.filter({"Variables" : ["passenger-demand-vkm"]})
dm_train_passenger = add_missing(dm_train_passenger, engine_types_passenger, "Categories2")
dm_passenger.append(dm_train_passenger, "Categories1")
dm_train_freight = dm.filter({"Variables" : ["freight-demand-vkm"]})
dm_train_freight = add_missing(dm_train_freight, engine_types_freight, "Categories2")
dm_freight.append(dm_train_freight, "Categories1")

################
##### SAVE #####
################

for c in dict_iso2.keys():
    dm_passenger.rename_col(c,dict_iso2[c],"Country")
dm_passenger.change_unit("passenger-demand-vkm", 1e6, "mio-vkm","vkm")
for c in dict_iso2.keys():
    dm_freight.rename_col(c,dict_iso2[c],"Country")
dm_freight.change_unit("freight-demand-vkm", 1e6, "mio-vkm","vkm")

DM = {}
DM["eurostat_passenger_vkm"] = dm_passenger.copy() # there is no EU27
DM["eurostat_freight_vkm"] = dm_freight.copy() # there is no EU27
f = os.path.join(current_file_directory, '../data/datamatrix/calibration_vkm.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)




