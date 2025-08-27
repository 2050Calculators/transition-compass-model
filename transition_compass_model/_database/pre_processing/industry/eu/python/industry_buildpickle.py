

# packages
from model.common.data_matrix_class import DataMatrix
from model.common.auxiliary_functions import my_pickle_dump
import pandas as pd
import pickle
import os
import numpy as np

# from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat

import plotly.express as px
import plotly.io as pio
import re
pio.renderers.default='browser'
import subprocess
import warnings
warnings.simplefilter("ignore")

# directories
current_file_directory = os.getcwd()

###############################################################################
############################### EXECUTE SCRIPTS ###############################
###############################################################################

# subprocess.run(['python', os.path.join(current_file_directory, 'industry_lever_material-switch.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'industry_lever_material-efficiency.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'industry_lever_technology-development.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'industry_lever_carbon-capture.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'industry_lever_technology-share.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'industry_lever_material-net-import.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'industry_lever_product-net-import.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'industry_lever_energy-switch.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'industry_lever_waste-management.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'industry_lever_material-recovery.py')])

# subprocess.run(['python', os.path.join(current_file_directory, 'industry_fxa_costs.py')])

# subprocess.run(['python', os.path.join(current_file_directory, 'industry_calib_emissions.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'industry_calib_energy-demand.py')])

# subprocess.run(['python', os.path.join(current_file_directory, 'industry_const_emission-factors.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'industry_const_energy-demand.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'industry_const_material-decomposition.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'industry_const_material-switch-ratio.py')])


###############################################################################
################################ BUILD PICKLE #################################
###############################################################################

# files
files_directory = os.path.join(current_file_directory, '../data/datamatrix')
files = os.listdir(files_directory)

# create DM_industry
DM_ots = {}
DM_fts = {}
DM_fxa = {}
DM_cal = {}
CDM_const = {}
DM_industry = {}

# create DM_ammonia
DM_ots_amm = {}
DM_fts_amm = {}
DM_fxa_amm = {}
DM_cal_amm = {}
CDM_const_amm = {}
DM_ammonia = {}

##################
##### LEVERS #####
##################

# list(np.array(files)[[bool(re.search("lever", i)) for i in files]])
lever_files = ['lever_material-switch.pickle', 'lever_material-efficiency.pickle', 
               'lever_technology-development.pickle', 'lever_carbon-capture.pickle', 
               'lever_technology-share.pickle', 'lever_material-net-import.pickle',
               'lever_product-net-import.pickle', 'lever_energy-switch.pickle',
               'lever_waste-management.pickle', 'lever_material-recovery.pickle',
               'lever_paperpack.pickle']
lever_names = ['material-switch', 'material-efficiency', 
               'technology-development',  'cc', 
               'technology-share', 'material-net-import', 
               'product-net-import', 'energy-carrier-mix', 
               'eol-waste-management', 'eol-material-recovery',
               'paperpack']

# load dms
for i in range(0, len(lever_files)):
    filepath = os.path.join(current_file_directory, '../data/datamatrix/' + lever_files[i])
    with open(filepath, 'rb') as handle:
        DM = pickle.load(handle)
    DM_ots[lever_names[i]] = DM["ots"]
    DM_fts[lever_names[i]] = DM["fts"]

# drop fertilizer
n = 'product-net-import'
DM_ots_amm[n] = DM_ots[n].filter({"Categories1" : ["fertilizer"]})
DM_ots[n].drop("Categories1","fertilizer")
for i in range(1,4+1):
    if n not in DM_fts_amm: DM_fts_amm[n] = {}
    DM_fts_amm[n][i] = DM_fts[n][i].filter({"Categories1" : ["fertilizer"]})
    DM_fts[n][i].drop("Categories1","fertilizer")

# drop ammonia
lever_names = ['material-efficiency','material-net-import']
for n in lever_names:
    DM_ots_amm[n] = DM_ots[n].filter({"Categories1" : ["ammonia"]})
    DM_ots[n].drop("Categories1","ammonia")
    for i in range(1,4+1):
        if n not in DM_fts_amm: DM_fts_amm[n] = {}
        DM_fts_amm[n][i] = DM_fts[n][i].filter({"Categories1" : ["ammonia"]})
        DM_fts[n][i].drop("Categories1","ammonia")

n = 'eol-material-recovery'
DM_ots_amm[n] = DM_ots[n].filter({"Categories2" : ["ammonia"]})
DM_ots[n].drop("Categories2","ammonia")
for i in range(1,4+1):
    if n not in DM_fts_amm: DM_fts_amm[n] = {}
    DM_fts_amm[n][i] = DM_fts[n][i].filter({"Categories2" : ["ammonia"]})
    DM_fts[n][i].drop("Categories2","ammonia")

lever_names = ['technology-development','cc',
               'energy-carrier-mix']
for n in lever_names:
    DM_ots_amm[n] = DM_ots[n].filter({"Categories1" : ["ammonia-tech"]})
    DM_ots[n].drop("Categories1","ammonia-tech")
    for i in range(1,4+1):
        if n not in DM_fts_amm: DM_fts_amm[n] = {}
        DM_fts_amm[n][i] = DM_fts[n][i].filter({"Categories1" : ["ammonia-tech"]})
        DM_fts[n][i].drop("Categories1","ammonia-tech")

# # drop products that have not being re-inserted in the calc yet
# drops = ['floor-area-new-non-residential','floor-area-reno-non-residential',
#          'computer', 'dishwasher', 'dryer', 'freezer', 'fridge', 'new-dhg-pipe',
#          'phone', 'tv', 'wmachine']
# DM_ots["product-net-import"].drop("Categories1",drops)
# for i in range(1,4+1):
#     DM_fts["product-net-import"][i].drop("Categories1",drops)
# drops = ['domapp','electronics']
# DM_ots["eol-waste-management"].drop("Variables",drops)
# for i in range(1,4+1):
#     DM_fts["eol-waste-management"][i].drop("Variables",drops)

# # save
# DM_industry["ots"] = DM_ots.copy()
# DM_industry["fts"] = DM_fts.copy()

#############################
##### FIXED ASSUMPTIONS #####
#############################

# files_temp = list(np.array(files)[[bool(re.search("fxa", i)) for i in files]])
# names_temp = [i.split("fxa_")[1].split(".pickle")[0] for i in files_temp]
# ['prod', 'cost-matprod', 'cost-CC']

# costs
filepath = os.path.join(current_file_directory, '../data/datamatrix/' + 'fxa_costs.pickle')
with open(filepath, 'rb') as handle:
    DM = pickle.load(handle)
DM_fxa["cost-matprod"] = DM["costs"]
DM_fxa["cost-CC"] = DM["costs-cc"]

# material production
filepath = os.path.join(current_file_directory, '../data/datamatrix/' + 'fxa_material-production.pickle')
with open(filepath, 'rb') as handle:
    DM = pickle.load(handle)
DM_fxa["prod"] = DM

# material demand
filepath = os.path.join(current_file_directory, '../data/datamatrix/' + 'fxa_material-demand.pickle')
with open(filepath, 'rb') as handle:
    DM = pickle.load(handle)
DM_fxa["demand"] = DM

# drop ammonia-tech
DM_fxa_amm["cost-matprod"] = DM_fxa["cost-matprod"].filter({"Categories1" : ["ammonia-tech"]})
DM_fxa["cost-matprod"].drop("Categories1","ammonia-tech")
DM_fxa_amm["cost-CC"] = DM_fxa["cost-CC"].filter({"Categories1" : ["ammonia-tech"]})
DM_fxa["cost-CC"].drop("Categories1","ammonia-tech")

#######################
##### CALIBRATION #####
#######################

files_temp = list(np.array(files)[[bool(re.search("calibration", i)) for i in files]])
names_temp = [i.split("calibration_")[1].split(".pickle")[0] for i in files_temp]

for i in range(0, len(files_temp)):
    filepath = os.path.join(current_file_directory, '../data/datamatrix/' + files_temp[i])
    with open(filepath, 'rb') as handle:
        dm = pickle.load(handle)
    DM_cal[names_temp[i]] = dm
    
# TODO: here we need to add the calibration of energy demand and emissions of ammonia manufacturing.
# for energy demand: probably can be inferred from emissions and constants, though we would need the energy mix (probably we can use the one of chemicals in JRC)
# for emissions: FAOSTAT -> Climate Change -> Totals and Indicators -> Emissions totals -> Fertilizer manufacturing

# drop ammonia
DM_cal_amm["material-production"] = DM_cal["material-production"].filter({"Categories1" : ["ammonia"]})
DM_cal["material-production"].drop("Categories1","ammonia")

#####################
##### CONSTANTS #####
#####################

# files_temp = list(np.array(files)[[bool(re.search("const", i)) for i in files]])
# names_temp = [i.split("const_")[1].split(".pickle")[0] for i in files_temp]

# material switch ratios
filepath = os.path.join(current_file_directory, '../data/datamatrix/' + 'const_material-switch-ratios.pickle')
with open(filepath, 'rb') as handle:
    cdm = pickle.load(handle)
CDM_const["material-switch"] = cdm

# material decomposition
filepath = os.path.join(current_file_directory, '../data/datamatrix/' + 'const_material-decomposition.pickle')
with open(filepath, 'rb') as handle:
    CDM = pickle.load(handle)
# CDM_const["material-decomposition_pipe"] = CDM["bld_pipe"] # do not load pipes for dh for now as this needs to be implemented in buildings
CDM_const["material-decomposition_floor"] = CDM["bld_floor"].filter({"Categories1" : ['floor-area-new-residential', 'floor-area-reno-residential']}) # keep only residential for now as non residential need to be implemented in buildings
CDM_const["material-decomposition_domapp"] = CDM["bld_domapp"].filter({"Categories1":['dishwasher', 'dryer', 'freezer', 'fridge','wmachine']})
CDM_const["material-decomposition_electronics"] = CDM["bld_domapp"].filter({"Categories1":['computer', 'phone', 'tv']})
CDM_const["material-decomposition_infra"] = CDM["tra_infra"]
CDM_const["material-decomposition_veh"] = CDM["tra_veh"]
CDM_const["material-decomposition_bat"] = CDM["tra_bat"]
CDM_const["material-decomposition_pack"] = CDM["pack"]
CDM_const_amm["material-decomposition_fertilizer"] = CDM["fertilizer"]

# energy demand
filepath = os.path.join(current_file_directory, '../data/datamatrix/' + 'const_energy-demand.pickle')
with open(filepath, 'rb') as handle:
    CDM = pickle.load(handle)
CDM_const["energy_excl-feedstock"] = CDM["energy-demand-excl-feedstock"].copy()
CDM_const["energy_feedstock"] = CDM["energy-demand-feedstock"].copy()
CDM_const["energy_excl-feedstock_eleclight-split"] = CDM["energy-demand-excl-feedstock-eleclight-split"].copy()
CDM_const["energy_efficiency"] = CDM["energy-efficiency"].copy()
CDM_const_amm['energy_excl-feedstock_eleclight-split'] = CDM["energy-demand-excl-feedstock-eleclight-split"].copy()
CDM_const_amm['energy_efficiency'] = CDM["energy-efficiency"].copy()

# emission factors
filepath = os.path.join(current_file_directory, '../data/datamatrix/' + 'const_emissions-factors.pickle')
with open(filepath, 'rb') as handle:
    CDM = pickle.load(handle)
CDM_const["emission-factor"] = CDM["combustion-emissions"].copy()
CDM_const["emission-factor-process"] = CDM["process-emissions"].copy()

# drop ammonia
lever_names = ['material-decomposition_floor', 'material-decomposition_infra',
               'material-decomposition_pack','material-decomposition_domapp',
               'material-decomposition_electronics']
for n in lever_names:
    CDM_const_amm[n] = CDM_const[n].filter({"Categories2" : ["ammonia"]})
    CDM_const[n].drop("Categories2","ammonia")
CDM_const_amm['material-decomposition_veh'] = CDM_const['material-decomposition_veh'].filter({"Categories3" : ["ammonia"]})
CDM_const['material-decomposition_veh'].drop("Categories3","ammonia")
CDM_const_amm['material-decomposition_bat'] = CDM_const['material-decomposition_bat'].filter({"Categories3" : ["ammonia"]})
CDM_const['material-decomposition_bat'].drop("Categories3","ammonia")
lever_names = ['energy_excl-feedstock', 'energy_feedstock', 
               'emission-factor-process']
for n in lever_names:
    CDM_const_amm[n] = CDM_const[n].filter({"Categories1" : ["ammonia-tech"]})
    CDM_const[n].drop("Categories1","ammonia-tech")
CDM_const_amm['emission-factor'] = CDM["combustion-emissions"].copy()

########################
##### PUT TOGETHER #####
########################

DM_industry = {
    'fxa': DM_fxa,
    'fts': DM_fts,
    'ots': DM_ots,
    'calibration': DM_cal,
    "constant" : CDM_const
}

DM_ammonia = {
    'fxa': DM_fxa_amm,
    'fts': DM_fts_amm,
    'ots': DM_ots_amm,
    'calibration': DM_cal_amm,
    "constant" : CDM_const_amm
}


##########################
##### KEEP ONLY EU27 #####
##########################

for key in DM_industry["ots"].keys():
    DM_industry["ots"][key].filter({"Country" : ["EU27"]},inplace=True)
for key in DM_industry["fts"].keys():
    for level in list(range(1,4+1)):
        DM_industry["fts"][key][level].filter({"Country" : ["EU27"]},inplace=True)
for key in DM_industry["fxa"].keys():
    DM_industry["fxa"][key].filter({"Country" : ["EU27"]},inplace=True)
for key in DM_industry["calibration"].keys():
    DM_industry["calibration"][key].filter({"Country" : ["EU27"]},inplace=True)

for key in DM_ammonia["ots"].keys():
    DM_ammonia["ots"][key].filter({"Country" : ["EU27"]},inplace=True)
for key in DM_ammonia["fts"].keys():
    for level in list(range(1,4+1)):
        DM_ammonia["fts"][key][level].filter({"Country" : ["EU27"]},inplace=True)
for key in DM_ammonia["fxa"].keys():
    DM_ammonia["fxa"][key].filter({"Country" : ["EU27"]},inplace=True)
for key in DM_ammonia["calibration"].keys():
    DM_ammonia["calibration"][key].filter({"Country" : ["EU27"]},inplace=True)

################
##### SAVE #####
################

# save
f = os.path.join(current_file_directory, '../../../../data/datamatrix/industry.pickle')
my_pickle_dump(DM_industry, f)

f = os.path.join(current_file_directory, '../../../../data/datamatrix/ammonia.pickle')
my_pickle_dump(DM_ammonia, f)

# # save
# f = os.path.join(current_file_directory, '../../../../data/datamatrix/industry.pickle')
# with open(f, 'wb') as handle:
#     pickle.dump(DM_industry, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# f = os.path.join(current_file_directory, '../../../../data/datamatrix/ammonia.pickle')
# with open(f, 'wb') as handle:
#     pickle.dump(DM_ammonia, handle, protocol=pickle.HIGHEST_PROTOCOL)

# filepath = os.path.join(current_file_directory, '../../../../data/interface/industry_to_energy.pickle')
# with open(filepath, 'rb') as handle:
#     DM = pickle.load(handle)
    
# DM
# DM["ind-serv-energy-demand"].units
# df = DM["ind-serv-energy-demand"].write_df()

# filepath = os.path.join(current_file_directory, '../../../../data/interface/industry_to_forestry.pickle')
# with open(filepath, 'rb') as handle:
#     dm = pickle.load(handle)
# dm.units