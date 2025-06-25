
# packages
from model.common.data_matrix_class import DataMatrix
from model.common.auxiliary_functions import my_pickle_dump
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
import subprocess

# file
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/transport/EU/python/transport_build-pickle.py"

# current file directory
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# load current transport pickle
filepath = os.path.join(current_file_directory, '../../../../data/datamatrix/transport.pickle')
with open(filepath, 'rb') as handle:
    DM_tra = pickle.load(handle)

###############################################################################
############################### EXECUTE SCRIPTS ###############################
###############################################################################

# subprocess.run(['python', os.path.join(current_file_directory, 'transport_fxa_passenger_tech.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'transport_fxa_passenger_vehicle-lifetime.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'transport_fxa_emission-factor-electricity.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'transport_fxa_freight_tech.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'transport_fxa_freight_mode_road.py')])

# subprocess.run(['python', os.path.join(current_file_directory, 'transport_lever_passenger_aviation-pkm.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'transport_lever_passenger_modal-share.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'transport_lever_passenger_occupancy.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'transport_lever_passenger_technology-share_new.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'transport_lever_passenger_utilization-rate.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'transport_lever_passenger_veh-efficiency_new.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'transport_lever_pkm.py')])

# subprocess.run(['python', os.path.join(current_file_directory, 'transport_lever_freight_modal-share.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'transport_fxa_freight_mode_other.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'transport_lever_freight_technology-share_new.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'transport_lever_freight_utilization-rate.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'transport_lever_freight_vehicle-efficiency_new.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'transport_lever_freight_tkm.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'transport_lever_fuel-mix.py')])

###############################################################################
################################ BUILD PICKLE #################################
###############################################################################

# files
files_directory = os.path.join(current_file_directory, '../data/datamatrix')
files = os.listdir(files_directory)

# create DM_transport
DM_ots = {}
DM_fts = {}
DM_fxa = {}
DM_cal = {}
CDM_const = {}
DM_transport = {}

##################
##### LEVERS #####
##################

# list(np.array(files)[[bool(re.search("lever", i)) for i in files]])
lever_files = ['lever_passenger_aviation-pkm.pickle','lever_passenger_modal-share.pickle',
               'lever_passenger_occupancy.pickle','lever_passenger_technology-share_new.pickle',
               'lever_passenger_utilization-rate.pickle','lever_passenger_veh-efficiency_new.pickle',
               'lever_pkm.pickle', 'lever_freight_vehicle-efficiency_new.pickle',
               'lever_freight_utilization-rate.pickle','lever_freight_modal-share.pickle',
               'lever_freight_technology-share_new.pickle','lever_freight_tkm.pickle',
               'lever_fuel-mix.pickle']
lever_names = ['passenger_aviation-pkm','passenger_modal-share',
               'passenger_occupancy','passenger_technology-share_new',
               'passenger_utilization-rate','passenger_veh-efficiency_new',
               'pkm', 'freight_vehicle-efficiency_new',
               'freight_utilization-rate', 'freight_modal-share',
               'freight_technology-share_new','freight_tkm',
               'fuel-mix']

# load dms
for i in range(0, len(lever_files)):
    filepath = os.path.join(current_file_directory, '../data/datamatrix/' + lever_files[i])
    with open(filepath, 'rb') as handle:
        DM = pickle.load(handle)
    DM_ots[lever_names[i]] = DM["ots"][lever_names[i]]
    DM_fts[lever_names[i]] = DM["fts"][lever_names[i]]

# save
DM_transport["ots"] = DM_ots.copy()
DM_transport["fts"] = DM_fts.copy()

#############################
##### FIXED ASSUMPTIONS #####
#############################

# list(np.array(files)[[bool(re.search("fxa", i)) for i in files]])
fxa_files = ['fxa_passenger_tech.pickle','fxa_passenger_vehicle-lifetime.pickle',
             'fxa_emission-factor-electricity.pickle','fxa_freight_tech.pickle',
             'fxa_freight_mode_other.pickle','fxa_freight_mode_road.pickle']
fxa_names = ['passenger_tech','passenger_vehicle-lifetime',
             'emission-factor-electricity','freight_tech',
             'freight_mode_other','freight_mode_road']

# load dms
for i in range(0, len(fxa_files)):
    filepath = os.path.join(current_file_directory, '../data/datamatrix/' + fxa_files[i])
    with open(filepath, 'rb') as handle:
        dm = pickle.load(handle)
        DM_fxa[fxa_names[i]] = dm

# save
DM_transport["fxa"] = DM_fxa.copy()

##########################
##### KEEP ONLY EU27 #####
##########################

for key in DM_transport["ots"].keys():
    DM_transport["ots"][key].filter({"Country" : ["EU27"]},inplace=True)
for key in DM_transport["fts"].keys():
    for level in list(range(1,4+1)):
        DM_transport["fts"][key][level].filter({"Country" : ["EU27"]},inplace=True)
for key in DM_transport["fxa"].keys():
    DM_transport["fxa"][key].filter({"Country" : ["EU27"]},inplace=True)
    
# #############################################################
# ##### PUT ZERO FOR FCEV IN FREIGHT (INSTEAD OF MISSING) #####
# #############################################################

# # note: I am doing this as otherwise the function rename_and_group() at line 453 in transport_module
# # gets blocked (as in EU27 there is no FCEV, so line 965 does not make FCV-hydrogen, and
# # then hydrogen is not found at line 453)

# levers = ["passenger_technology-share_new", "passenger_veh-efficiency_new", 
#           'freight_vehicle-efficiency_new', 'freight_technology-share_new']
# for lever in levers:
#     idx = DM_transport["ots"][lever].idx
#     DM_transport["ots"][lever].array[:,:,:,:,idx["FCEV"]] = 0
#     for i in range(1,4+1):
#         DM_transport["fts"][lever][i].array[:,:,:,:,idx["FCEV"]] = 0
# variabs = ["passenger_tech","passenger_vehicle-lifetime","freight_tech"]
# for v in variabs:
#     idx = DM_transport["fxa"][v].idx
#     DM_transport["fxa"][v].array[:,:,:,:,idx["FCEV"]] = 0

###############
#### SAVE #####
###############

# save
f = os.path.join(current_file_directory, '../../../../data/datamatrix/transport.pickle')
my_pickle_dump(DM_transport, f)

# # check

# list(DM_transport["fxa"])
# DM_transport["fxa"]["passenger_tech"].filter({"Variables" : ['tra_passenger_technology-share_fleet']}).flatten().flatten().datamatrix_plot()
# DM_transport["fxa"]["passenger_tech"].filter({"Variables" : ['tra_passenger_new-vehicles']}).flatten().flatten().datamatrix_plot()
# DM_transport["fxa"]["passenger_tech"].filter({"Variables" : ['tra_passenger_vehicle-waste']}).flatten().flatten().datamatrix_plot()

# list(DM_transport["ots"])

# level = 1

# dm_temp = DM_transport["ots"]["passenger_modal-share"].copy()
# dm_temp.append(DM_transport["fts"]["passenger_modal-share"][level],"Years")
# dm_temp.flatten().datamatrix_plot(stacked=True)

# dm_temp = DM_transport["ots"]["passenger_occupancy"].copy()
# dm_temp.append(DM_transport["fts"]["passenger_occupancy"][level],"Years")
# dm_temp.flatten().datamatrix_plot()

# dm_temp = DM_transport["ots"]['passenger_technology-share_new'].copy()
# dm_temp.append(DM_transport["fts"]['passenger_technology-share_new'][level], "Years")
# dm_temp.filter({"Categories1" : ["LDV"]}).flatten().datamatrix_plot(stacked=True)

# dm_temp = DM_transport["ots"]["passenger_utilization-rate"].copy()
# dm_temp.append(DM_transport["fts"]["passenger_utilization-rate"][level],"Years")
# dm_temp.flatten().datamatrix_plot()

# dm_temp = DM_transport["ots"]["pkm"].copy()
# dm_temp.append(DM_transport["fts"]["pkm"][level],"Years")
# dm_temp.datamatrix_plot()




