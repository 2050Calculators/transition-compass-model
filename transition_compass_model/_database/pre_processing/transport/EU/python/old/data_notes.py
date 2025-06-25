
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
import re
pio.renderers.default='browser'

from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat
from _database.pre_processing.routine_JRC import get_jrc_data
from model.common.auxiliary_functions import eurostat_iso2_dict, jrc_iso2_dict

# file
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/transport/EU/python/transport_build-pickle.py"

# directories
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# load old pickle
filepath = os.path.join(current_file_directory, '../../../../data/datamatrix/transport.pickle')
with open(filepath, 'rb') as handle:
    DM = pickle.load(handle)
list(DM)

# constants
DM["constant"]

# fxa
list(DM['fxa'])

DM['fxa']["emission-factor-electricity"].units
DM['fxa']["freight_mode_other"].units
DM['fxa']["passenger_tech"].filter({"Variables" : ["tra_passenger_technology-share_fleet"]})

# levers
list(DM['ots'])

dict_iso2 = eurostat_iso2_dict()
dict_iso2.pop('CH')  # Remove Switzerland

##########################################################################################
######################################## EUROSTAT ########################################
##########################################################################################

toc_df = eurostat.get_toc_df()

# # try to get data in dm
# code = "avia_tppa"
# eurostat.get_pars(code)
# filter = {'geo\TIME_PERIOD': list(dict_iso2.keys()),
#           'tra_cov': 'TOTAL',
#           'unit' : 'MIO_PKM'}
# mapping_dim = {'Country': 'geo\TIME_PERIOD',
#                'Variables': 'tra_cov'}
# dm = get_data_api_eurostat(code, filter, mapping_dim, 'mi-pkm')

#################################################################
########################### PASSENGER ###########################
#################################################################

##########################
##### AVIATION (pkm) #####
##########################

DM['ots']['passenger_aviation-pkm']
DM['ots']['passenger_aviation-pkm'].units # pkm/cap
# Passenger-kilometres (pkm) is the total distance travelled by all the passengers. 
# For instance, one person travelling for 20km contributes for 20 passenger-kilometres; 
# four people, travelling for 20km each, contribute for 80 passenger- kilometres
# note: probably find pkm directly
# eurostat.get_pars("avia_tppa")
# filter = {'geo\TIME_PERIOD': list(dict_iso2.keys()),
#           'tra_cov': 'TOTAL',
#           'unit' : 'MIO_PKM'}
# mapping_dim = {'Country': 'geo\TIME_PERIOD',
#                 'Variables': 'tra_cov'}
# mapping_calc = {'tra_pkm-cap_aviation' : ['TOTAL']}

#######################
##### MODAL SHARE #####
#######################

DM['ots']['passenger_modal-share']
DM['ots']['passenger_modal-share'].units # %
# % of mode of transport used by people between ['2W', 'LDV', 'bike', 'bus', 'metrotram', 'rail', 'walk']
# eurostat.get_pars("road_pa_mov")
# filter = {'geo\TIME_PERIOD': list(dict_iso2.keys()),
#           'vehicle': ['MOTO_MOP','CAR','BUS_MCO_TRO'],
#           'unit' : 'MIO_PKM'}
# mapping_dim = {'Country': 'geo\TIME_PERIOD',
#                 'Variables': 'vehicle'}
# mapping_calc = {'MOTO_MOP': ['2W'], 'CAR' : ['LDV'], 'BUS_MCO_TRO' : ['bus']}
# eurostat.get_pars("rail_pa_total")
# filter = {'geo\TIME_PERIOD': list(dict_iso2.keys()),
#           'unit' : 'MIO_PKM'}
# mapping_dim = {'Country': 'geo\TIME_PERIOD',
#                 'Variables': 'unit'}
# mapping_calc = {'rail': ['MIO_PKM']}
# missing for the moment is bike, metrotram and walk
# alternative is tran_hv_ms_psmod (%) for LDV, bus, rail (or tran_hv_psmod if more data availability)
# TODO: possibly this can be obtained from fleet data (and with that you can get metro tram)
df = eurostat.subset_toc_df(toc_df, 'cycl') # nothing on bikes 
df = eurostat.subset_toc_df(toc_df, 'tram') # only in new and stocks, no modal share or pkm
df = eurostat.subset_toc_df(toc_df, 'walk') # for walking and cycling, there is only hlth_ehis_pe6e, which is % of people that are walking and cycling at least 30 mins a day in 2019


#####################
##### OCCUPANCY #####
#####################

DM['ots']['passenger_occupancy']
DM['ots']['passenger_occupancy'].units # pkm/vkm
# Vehicle-kilometre (vkm) is the total distance travelled by all vehicles.
# pkm/vkm is total distance by passenger over totak distance by all vehicles,
# so how many passengers per vehicle by ['2W', 'LDV', 'bus', 'metrotram', 'rail']
# note: tbd how to get vkm, probably either raw or n of vehicles * n of kilometers travelled by vehicle on average ?
# for numerator (pkm): road_pa_mov (mi-pkm) for LDV, bus, 2W and rail_pa_total (mi-pkm) for rail (both mentioned in modal share), while metrotram missing.
# for denominator (vkm): road_tf_vehmov (mi-vkm) for LDV, bus, 2W, rail, while metrotram missing.
# eurostat.get_pars("road_tf_vehmov")
# filter = {'geo\TIME_PERIOD': list(dict_iso2.keys()),
#           'regisveh': "TERNAT_REG", # TODO: this is traffic performed on the national territory by vehicles registered in the reporting country or in foreign countries (there is another option of exlcuding "registered in foreign countries", see with Paola)
#           'vehicle' : ["MOTO_MOP","CAR","BUS_MCO_TRO"], # note: I am not taking lorries and road train for now, as these are something like trucks
#           'unit' : 'MIO_VKM'}
# mapping_dim = {'Country': 'geo\TIME_PERIOD',
#                 'Variables': 'vehicle'}
# mapping_calc = {'2W': ['MOTO_MOP'], 'LDV' : ['CAR'], 'bus' : ['BUS_MCO_TRO']}
# eurostat.get_pars("rail_tf_traveh")
# filter = {'geo\TIME_PERIOD': list(dict_iso2.keys()),
#           'mot_nrg' : ["TOTAL"], # other alternatives were diesel and eletric
#           'train' : ["TRN_PAS"], # taking only passenger trains
#           'vehicle' : ["TOTAL"], # other other alternarives were locomotives and railcars
#           'unit' : 'THS_TRKM'} # thousands of train-kilometers
# mapping_dim = {'Country': 'geo\TIME_PERIOD',
#                 'Variables': 'vehicle'}
# mapping_calc = {'rail': ['TOTAL']}

############################
##### TECHNOLOGY SHARE #####
############################

# TODO: why there is no passenger aviation among these? Should I add one variable that is gasoline 100%?

DM['ots']['passenger_technology-share_new']
DM['ots']['passenger_technology-share_new'].units # %
list(DM['ots']['passenger_technology-share_new'].write_df().columns)
# for each mode of transport, split between type of mode of transport (so it's by
# mode and by type of mode). Types of mode, or technologies, are
# ['BEV', 'CEV', 'FCEV', 'ICE-diesel', 'ICE-gas', 'ICE-gasoline', 'PHEV-diesel', 'PHEV-gasoline', 'mt']
DM['ots']['passenger_technology-share_new'].group_all("Categories2",inplace=False).array
# the sum of the shares of the techs is 1
# note: in theory it will be computed in the same way of modal share, so maybe with pkm tech / pkm mode
# but there are no data on Eurostat of tech with pkm. So possible strategy: get shares of fleet new.

# 2W_BEV, 2W_ICE-diesel, 2W_ICE-gas, 2W_ICE-gasoline, 2W_PHEV-gasoline
# eurostat.get_pars("road_eqr_mopeds")
# filter = {'geo\TIME_PERIOD': list(dict_iso2.keys()),
#           'vehicle': ['MOTO','MOP'],
#           'mot_nrg': ['FOSS','ZEMIS'], # FOSS are fossil fuels, and ZEMIS should be electric.
#           'unit' : 'NR'}
# mapping_dim = {'Country': 'geo\TIME_PERIOD',
#                'Variables': 'vehicle',
#                'Categories1' : 'mot_nrg'}
# mapping_calc = {'Variables' : {'2W': ['MOTO','MOP']},
#                 'Categories1' : {'ICE' : ['FOSS'], 'BEV' : ['ZEMIS']}}
# TODO: do the split between ICE-diesel, ICE-gas, ICE-gasoline and 2W_PHEV-gasoline

# LDV_BEV, LDV_FCEV, LDV_ICE-diesel, LDV_ICE-gas, LDV_ICE-gasoline, LDV_PHEV-diesel, LDV_PHEV-gasoline
# eurostat.get_pars("road_eqr_carpda")
# filter = {'geo\TIME_PERIOD': list(dict_iso2.keys()),
#           'mot_nrg' : ['PET','LPG','DIE','GAS','ELC','PET_X_HYB','ELC_PET_PI',
#                         'DIE_X_HYB','ELC_DIE_PI','HYD_FCELL'], # note: I get both total PET and PET excluding hybdrid to check data availab, but in theory the right one is the one excluding hybrids
#           'unit' : 'NR'}
# mapping_dim = {'Country': 'geo\TIME_PERIOD',
#                 'Variables': 'mot_nrg'}
# mapping_calc = {'LDV_ICE-gasoline' : ['PET_X_HYB','ELC_PET_HYB'], 'LDV_ICE-gas' : ['LPG','GAS'], 
#                 'LDV_ICE-diesel' : ['DIE_X_HYB', 'ELC_DIE_HYB'], 'LDV_BEV' : ['ELC'], 
#                 'LDV_PHEV-gasoline' : ['ELC_PET_PI'], 'LDV_PHEV-diesel' : ['ELC_DIE_PI'],
#                 'LDV_FCEV' : ['HYD_FCELL']}
# TODO: for now I have put normal hybrid in ICE, see with Paola if that's fine

# bus_CEV, bus_FCEV, bus_ICE-diesel, bus_ICE-gas, bus_ICE-gasoline, bus_PHEV-diesel
# eurostat.get_pars("road_eqr_busmot")
# filter = {'geo\TIME_PERIOD': list(dict_iso2.keys()),
#           'mot_nrg' : ['PET','LPG','DIE','GAS','ELC',
#                        'DIE_X_HYB','ELC_DIE_PI','HYD_FCELL','CNG','LNG'],
#           'unit' : 'NR'}
# mapping_dim = {'Country': 'geo\TIME_PERIOD',
#                 'Variables': 'mot_nrg'}
# mapping_calc = {'bus_ICE-gasoline' : ['PET_X_HYB'], 'bus_ICE-gas' : ['LPG','GAS','CNG','LNG'], 
#                 'bus_ICE-diesel' : ['DIE_X_HYB'], 'bus_CEV' : ['ELC'], 
#                 'bus_PHEV-diesel' : ['ELC_DIE_PI'],
#                 'bus_FCEV' : ['HYD_FCELL']}

# metrotram_mt
# eurostat.get_pars("road_eqr_trams")
# filter = {'geo\TIME_PERIOD': list(dict_iso2.keys()),
#           'unit' : 'NR'}
# mapping_dim = {'Country': 'geo\TIME_PERIOD',
#                 'Variables': 'unit'}
# mapping_calc = {'metrotram_mt' : ['NR']}

# rail_CEV and rail_ICE-diesel
# eurostat.get_pars("rail_eq_locon")
# filter = {'geo\TIME_PERIOD': list(dict_iso2.keys()),
#           'vehicle' : ['LOC','RCA'],
#           'unit' : 'NR'}
# mapping_dim = {'Country': 'geo\TIME_PERIOD',
#                 'Variables': 'vehicle'}
# mapping_calc = {'rail' : ['LOC','RCA']}
# TODO: not sure if these are stock or flow data. If they are stocks, for flows we can probably approximate sotck t - stock t-1 (assuming out eol) to get the percentages we need. TBD what to do for the fleet, probably merging with production data.
# TODO: split between cev and diesel, you can probably use rail_eq_locop, which is in Megawatt but has the split and make some shares
# TODO: this is all locomotives and railcars, there is no split between passenger and freight, probably check JRC

############################
##### UTILIZATION RATE #####
############################

DM['ots']['passenger_utilization-rate']
DM['ots']['passenger_utilization-rate'].units # vkm/veh
# this should be the vkm divided number of vehicles in stock
# get data from passenger occupancy and fleet stock and compute it

# vehicle efficiency
DM['ots']["passenger_veh-efficiency_new"]
DM['ots']["passenger_veh-efficiency_new"].units # MJ/km
# # this will be from JRC

# pkm
DM['ots']["pkm"]
DM['ots']["pkm"].units # pkm/cap
# I guess this is pkm divided by our own population data

###############################################################
########################### FREIGHT ###########################
###############################################################

#######################
##### MODAL SHARE #####
#######################

DM['ots']['freight_modal-share']
DM['ots']['freight_modal-share'].units # %
# this should be tkm over total tkm
# source:
# tran_hv_ms_frmod (%) for 'IWW', 'aviation', 'marine', 'rail'. Then you have a general "roads" but split
# between 'HDVH', 'HDVL', 'HDVM' is missing.
# road_go_ta_mplw (tkm) or road_go_ta_lc (tkm) for 'HDVH', 'HDVL', 'HDVM' (mapping to be done, but easy)

# # aviation
# eurostat.get_pars("avia_tpgo")
# filter = {'geo\TIME_PERIOD': list(dict_iso2.keys()),
#           'tra_cov': ['TOTAL'],
#           'unit' : 'MIO_TKM'}
# mapping_dim = {'Country': 'geo\TIME_PERIOD',
#                 'Variables': 'tra_cov'}
# mapping_calc = {'aviation': ['TOTAL']}

# # IWW
# # there is no data on tkm, only data on shares directly here below (for tkm check JRC)
# eurostat.get_pars("tran_hv_ms_frmod")
# filter = {'geo\TIME_PERIOD': list(dict_iso2.keys()),
#           'tra_mode': ['IWW'],
#           'unit' : 'PC'}
# mapping_dim = {'Country': 'geo\TIME_PERIOD',
#                 'Variables': 'tra_mode'}
# mapping_calc = {'iww': ['IWW']}

# # marine
# eurostat.get_pars("mar_tp_go")
# filter = {'geo\TIME_PERIOD': list(dict_iso2.keys()),
#           'tra_cov': ['TOTAL'],
#           'unit' : 'MIO_TKM'}
# mapping_dim = {'Country': 'geo\TIME_PERIOD',
#                 'Variables': 'tra_cov'}
# mapping_calc = {'marine': ['TOTAL']}

# # rail
# eurostat.get_pars("rail_go_total")
# filter = {'geo\TIME_PERIOD': list(dict_iso2.keys()),
#           'unit' : 'MIO_TKM'}
# mapping_dim = {'Country': 'geo\TIME_PERIOD',
#                 'Variables': 'unit'}
# mapping_calc = {'rail': ['MIO_TKM']}

# # trucks
# eurostat.get_pars("road_go_ta_lc")
# filter = {'geo\TIME_PERIOD': list(dict_iso2.keys()),
#           'tra_type' : ['TOTAL'],
#           'weight' : ['T_LE9P5','T9P6-15P5','T15P6-20P5','T20P6-25P5','T25P6-30P5','T_GT30P5'],
#           'unit' : 'MIO_TKM'}
# mapping_dim = {'Country': 'geo\TIME_PERIOD',
#                 'Variables': 'weight'}
# mapping_calc = {'HDVL': ['T_LE9P5','T9P6-15P5'],
#                 'HDVM': ['T15P6-20P5','T20P6-25P5'],
#                 'HDVH': ['T25P6-30P5','T_GT30P5']}


############################
##### technology share #####
############################

DM['ots']['freight_technology-share_new']
DM['ots']['freight_technology-share_new'].units # %
DM['ots']['freight_technology-share_new'].write_df().columns
# this is num/num of fleet new

# HDVL, CEV, FCEV, ICE-diesel, ICE-gas, ICE-gasoline, PHEV-diesel, PHEV-gasoline
# eurostat.get_pars("road_eqr_lormot")
# filter = {'geo\TIME_PERIOD': list(dict_iso2.keys()),
#           'vehicle': ['VG_LE3P5'],
#           'mot_nrg' : ['LPG','GAS','ELC','PET_X_HYB','ELC_PET_PI',
#                        'DIE_X_HYB','ELC_DIE_PI','HYD_FCELL'], # note: I get both total PET and PET excluding hybdrid to check data availab, but in theory the right one is the one excluding hybrids
#           'unit' : 'NR'}
# mapping_dim = {'Country': 'geo\TIME_PERIOD',
#                 'Variables': 'mot_nrg'}
# mapping_calc = {'HDVL_ICE-gasoline' : ['PET_X_HYB','ELC_PET_HYB'], 'HDVL_ICE-gas' : ['LPG','GAS','CNG','LNG'], 
#                 'HDVL_ICE-diesel' : ['DIE_X_HYB','ELC_DIE_HYB'], 'HDVL_BEV' : ['ELC']}
# note: there are no PHEV nor FCEV

# HDVH, HDVM: CEV, FCEV, ICE-diesel, ICE-gas, ICE-gasoline, PHEV-diesel, PHEV-gasoline
# eurostat.get_pars("road_eqr_lormot")
# filter = {'geo\TIME_PERIOD': list(dict_iso2.keys()),
#           'vehicle': ['LOR_GT3P5'],
#           'mot_nrg' : ['LPG','GAS','ELC','PET_X_HYB','ELC_PET_PI',
#                        'DIE_X_HYB','ELC_DIE_PI','HYD_FCELL'], # note: I get both total PET and PET excluding hybdrid to check data availab, but in theory the right one is the one excluding hybrids
#           'unit' : 'NR'}
# mapping_dim = {'Country': 'geo\TIME_PERIOD',
#                 'Variables': 'mot_nrg'}
# mapping_calc = {'HDVM_ICE-gasoline' : ['PET_X_HYB','ELC_PET_HYB'], 'HDVM_ICE-gas' : ['LPG','GAS','CNG','LNG'], 
#                 'HDVM_ICE-diesel' : ['DIE_X_HYB','ELC_DIE_HYB'], 'HDVM_BEV' : ['ELC']}
# TODO: I am missing the split at least between HDVM and HDVH (and to be udnerstood if we need to keep
# homogeneous weight split between three categories or not, as in this case the HDVL is below 3.5 tonnes)

# IWW
# eurostat.get_pars("iww_eq_loadcap")
# filter = {'geo\TIME_PERIOD': list(dict_iso2.keys()),
#           'vessel': ['BAR_SP'],
#           'weight' : ['TOTAL'], # note: I get both total PET and PET excluding hybdrid to check data availab, but in theory the right one is the one excluding hybrids
#           'unit' : 'NR'}
# mapping_dim = {'Country': 'geo\TIME_PERIOD',
#                 'Variables': 'vessel'}
# mapping_calc = {'IWW_ICE' : ['BAR_SP']}
# note: there is no FCEV nor BEV, and probably all ICE is ICE-diesel
# TODO: this is data on equipment, not sure if it's stock or flows (it's probably stock), to be decided what to do

# aviation
# TODO: data no available by type of engine, I can just add gasoline 100%

# marine
# TODO: data not available on eurostat, check JRC

# rail
# TODO: rail_eq_locon is all locomotives and railcars, there is no split between passenger and freight, probably check JRC

############################
##### UTILIZATION RATE #####
############################

DM['ots']['freight_utilization-rate']
DM['ots']['freight_utilization-rate'].units # load factor is tkm/vkm and utilisation rate is vkm/year
# TODO: this is just for trucks, and to be understood what is year

##############################
##### VEHICLE EFFICIENCY #####
##############################

DM['ots']["freight_vehicle-efficiency_new"]
DM['ots']["freight_vehicle-efficiency_new"].units # MJ/km
# get it from JRC

###############
##### TKM #####
###############

# tkm
DM['ots']["freight_tkm"]
DM['ots']["freight_tkm"].units # bn-tkm
# this is sum of tkm from above

####################
##### FUEL MIX #####
####################

DM['ots']["fuel-mix"]
DM['ots']["fuel-mix"].units # %
DM['ots']["fuel-mix"].group_all("Categories2",inplace=False).array
# TODO: this one I am not sure


#####################################################################################
######################################## JRC ########################################
#####################################################################################

dict_iso2 = jrc_iso2_dict()
list(DM['ots'])

##################################
##### passenger_aviation-pkm #####
##################################

DM['ots']["passenger_aviation-pkm"]

# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrAvia_act",
#                 "variable" : "Passenger transport (mio pkm)",
#                 "sheet_last_row" : "International - Extra-EEAwUK",
#                 "sub_variables" : ["Domestic"],
#                 "calc_names" : ["aviation"]}
# dm_passenger_aviation_pkm = get_jrc_data(dict_extract, dict_iso2, current_file_directory)
f = os.path.join(current_file_directory, '../data/datamatrix/passenger_aviation-pkm.pickle')
# with open(f, 'wb') as handle:
#     pickle.dump(dm_passenger_aviation_pkm, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(f, 'rb') as handle:
    dm_passenger_aviation_pkm = pickle.load(handle)
    
# rename and deepen
dm_passenger_aviation_pkm.rename_col("aviation","tra_pkm-cap_aviation","Variables")
dm_passenger_aviation_pkm.deepen()

#################################
##### passenger_modal-share #####
#################################

DM['ots']["passenger_modal-share"].units

# '2W', 'LDV', 'bike', 'bus', 'metrotram', 'rail', 'walk'
# dict_extract = {"database" : "Transport",
#                 "sheet" : "Transport",
#                 "variable" : "Passenger transport (mio pkm)",
#                 "sheet_last_row" : "International - Extra-EEAwUK",
#                 "sub_variables" : ["Powered two-wheelers",
#                                     "Passenger cars",
#                                     "Motor coaches, buses and trolley buses",
#                                     "Metro and tram, urban light rail", 
#                                     "Conventional passenger trains",
#                                     "High speed passenger trains"],
#                 "calc_names" : ["2W", "LDV", "bus", "metrotram", "rail-conv", "rail-highspeed"]}
# dm_passenger_modal_share = get_jrc_data(dict_extract, dict_iso2, current_file_directory)
f = os.path.join(current_file_directory, '../data/datamatrix/passenger_modal-share.pickle')
# with open(f, 'wb') as handle:
#     pickle.dump(dm_passenger_modal_share, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(f, 'rb') as handle:
    dm_passenger_modal_share = pickle.load(handle)

# get rail
dm_passenger_modal_share.operation("rail-conv", "+", "rail-highspeed","Variables",out_col="rail",unit="mio pkm")
dm_passenger_modal_share.drop("Variables",["rail-conv","rail-highspeed"])

# rename and deepen
for v in dm_passenger_modal_share.col_labels["Variables"]:
    dm_passenger_modal_share.rename_col(v,"tra_passenger_modal-share_" + v, "Variables")
dm_passenger_modal_share.deepen()

# get it in pkm
dm_passenger_modal_share.change_unit("tra_passenger_modal-share", 1e6, "mio pkm", "pkm")
dm_passenger_pkm = dm_passenger_modal_share.copy()

# # do the percentages
# dm_passenger_modal_share.normalise("Variables", inplace=True, keep_original=False)
# # note: to understand if to do this after the ots and fts have been computed

# TODO: bike and walk

###############################
##### passenger_occupancy #####
###############################

DM['ots']["passenger_occupancy"].units

# '2W', 'LDV', 'bus'
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrRoad_act",
#                 "variable" : "Vehicle-km driven (mio km)",
#                 "sheet_last_row" : "Motor coaches, buses and trolley buses",
#                 "sub_variables" : ["Powered two-wheelers",
#                                    "Passenger cars",
#                                    "Motor coaches, buses and trolley buses"],
#                 "calc_names" : ["2W", "LDV", "bus"]}
# dm_passenger_occupancy_road = get_jrc_data(dict_extract, dict_iso2, current_file_directory)
f = os.path.join(current_file_directory, '../data/datamatrix/passenger_occupancy_road.pickle')
# with open(f, 'wb') as handle:
#     pickle.dump(dm_passenger_occupancy_road, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(f, 'rb') as handle:
    dm_passenger_occupancy_road = pickle.load(handle)

# 'metrotram', 'rail'
dict_extract = {"database" : "Transport",
                "sheet" : "TrRail_act",
                "variable" : "Vehicle-km (mio km)",
                "sheet_last_row" : "High speed passenger trains",
                "sub_variables" : ["Metro and tram, urban light rail",
                                   "Conventional passenger trains",
                                   "High speed passenger trains"],
                "calc_names" : ["metrotram", "rail-conv", "rail-highspeed"]}
dm_passenger_occupancy_rail = get_jrc_data(dict_extract, dict_iso2, current_file_directory)
f = os.path.join(current_file_directory, '../data/datamatrix/passenger_occupancy_rail.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_passenger_occupancy_rail, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(f, 'rb') as handle:
    dm_passenger_occupancy_rail = pickle.load(handle)

# get rail
dm_passenger_occupancy_rail.operation("rail-conv", "+", "rail-highspeed","Variables",out_col="rail",unit="mio pkm")
dm_passenger_occupancy_rail.drop("Variables",["rail-conv","rail-highspeed"])

# put together
dm_passenger_occupancy = dm_passenger_occupancy_road.copy()
dm_passenger_occupancy.append(dm_passenger_occupancy_rail, "Variables")
dm_passenger_occupancy.sort("Variables")

# rename and deepen
for v in dm_passenger_occupancy.col_labels["Variables"]:
    dm_passenger_occupancy.rename_col(v,"tra_passenger_occupancy_" + v, "Variables")
dm_passenger_occupancy.deepen()

# get it in pkm
dm_passenger_occupancy.change_unit("tra_passenger_occupancy", 1e6, "mio km", "vkm")

##########################################
##### passenger_technology-share_new #####
##########################################

DM['ots']["passenger_technology-share_new"].units

# 2W 
dict_extract = {"database" : "Transport",
                "sheet" : "TrRoad_act",
                "variable" : "Vehicle-km driven (mio km)",
                "sheet_last_row" : "Battery electric vehicles",
                "sub_variables" : ["Powered two-wheelers",
                                   "Passenger cars",
                                   "Gasoline engine", "Diesel oil engine", "LPG engine", "Natural gas engine", 
                                   "Plug-in hybrid electric", "Battery electric vehicles"],
                "calc_names" : ["2W"]}
dm_passenger_technology_share_new_2w = get_jrc_data(dict_extract, dict_iso2, current_file_directory)

# LDV


# on Monday: add a unit option in the get_jrc_data() function, as there are things like new registration vehicles
# for which the unit is not reported. Then keep going with the LDV.










