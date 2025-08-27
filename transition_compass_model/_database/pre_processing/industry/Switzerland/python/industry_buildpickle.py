
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

# tomorrow: 
# for EU: in calibration data, make sure all 3 calib variables have 40 years (decide what
# to do with fts, probably put missing values, and understand how calibration is done
# for fts when levers change, if any ... probably calibration should be done only in baseline)
# also revisit calibration data and if you need to extrapolate it, rather put nan
# and finally do the ammonia emission / energy demand thing

###############################################################################
############################### EXECUTE SCRIPTS ###############################
###############################################################################

# subprocess.run(['python', os.path.join(current_file_directory, 'industry_lever_product-net-import.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'industry_lever_waste-management.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'industry_calib_energy-demand.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'industry_calib_emissions.py')])
# subprocess.run(['python', os.path.join(current_file_directory, 'ammonia.py')])

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
# CDM_const = {}
DM_industry = {}

# create DM_ammonia
DM_ots_amm = {}
DM_fts_amm = {}
DM_fxa_amm = {}
DM_cal_amm = {}
# CDM_const_amm = {}
DM_ammonia = {}

# load europe and make copies for switzerland that are missing
filepath = os.path.join(current_file_directory,  '../../../../data/datamatrix/industry.pickle')
with open(filepath, 'rb') as handle:
    DM_industry_copies = pickle.load(handle)
for key in DM_industry_copies["ots"].keys():
    DM_industry_copies["ots"][key].filter({"Country" : ["EU27"]}, inplace=True)
    DM_industry_copies["ots"][key].rename_col("EU27","Switzerland","Country")
for key in DM_industry_copies["fts"].keys():
    for level in list(range(1,4+1)):
        DM_industry_copies["fts"][key][level].filter({"Country" : ["EU27"]}, inplace=True)
        DM_industry_copies["fts"][key][level].rename_col("EU27","Switzerland","Country")
for key in DM_industry_copies["fxa"].keys():
    DM_industry_copies["fxa"][key].filter({"Country" : ["EU27"]}, inplace=True)
    DM_industry_copies["fxa"][key].rename_col("EU27","Switzerland","Country")

filepath = os.path.join(current_file_directory,  '../../../../data/datamatrix/ammonia.pickle')
with open(filepath, 'rb') as handle:
    DM_ammonia_copies = pickle.load(handle)
for key in DM_ammonia_copies["ots"].keys():
    DM_ammonia_copies["ots"][key].filter({"Country" : ["EU27"]}, inplace=True)
    DM_ammonia_copies["ots"][key].rename_col("EU27","Switzerland","Country")
for key in DM_ammonia_copies["fts"].keys():
    for level in list(range(1,4+1)):
        DM_ammonia_copies["fts"][key][level].filter({"Country" : ["EU27"]}, inplace=True)
        DM_ammonia_copies["fts"][key][level].rename_col("EU27","Switzerland","Country")
for key in DM_ammonia_copies["fxa"].keys():
    DM_ammonia_copies["fxa"][key].filter({"Country" : ["EU27"]}, inplace=True)
    DM_ammonia_copies["fxa"][key].rename_col("EU27","Switzerland","Country")


##################
##### LEVERS #####
##################

# list(np.array(files)[[bool(re.search("lever", i)) for i in files]])
lever_files = ['lever_product-net-import.pickle',
               'lever_material-net-import.pickle',
               'lever_paperpack.pickle',
               'lever_waste-management.pickle']
lever_names = ['product-net-import',
               'material-net-import',
               'paperpack',
               'eol-waste-management']

# load dms
for i in range(0, len(lever_files)):
    filepath = os.path.join(current_file_directory, '../data/datamatrix/' + lever_files[i])
    with open(filepath, 'rb') as handle:
        DM = pickle.load(handle)
    DM_ots[lever_names[i]] = DM["ots"]
    DM_fts[lever_names[i]] = DM["fts"]
    
# add missing, copying from europe
present = list(DM_ots)
europe = list(DM_industry_copies["ots"])
missing = np.array(europe)[[e not in present for e in europe]].tolist()
for m in missing:
    DM_ots[m] = DM_industry_copies["ots"][m].copy()
    DM_fts[m] = DM_industry_copies["fts"][m].copy()
DM_ots = {key: DM_ots[key] for key in europe}
DM_fts = {key: DM_fts[key] for key in europe}

# make ammonia
lever_files = ['lever_product-net-import_ammonia.pickle',
               'lever_material-net-import_ammonia.pickle']
lever_names = ['product-net-import',
               'material-net-import']
for i in range(0, len(lever_files)):
    filepath = os.path.join(current_file_directory, '../data/datamatrix/' + lever_files[i])
    with open(filepath, 'rb') as handle:
        DM = pickle.load(handle)
    DM_ots_amm[lever_names[i]] = DM["ots"]
    DM_fts_amm[lever_names[i]] = DM["fts"]
present = list(DM_ots_amm)
europe = list(DM_ammonia_copies["ots"])
missing = np.array(europe)[[e not in present for e in europe]].tolist()
for m in missing:
    DM_ots_amm[m] = DM_ammonia_copies["ots"][m].copy()
    DM_fts_amm[m] = DM_ammonia_copies["fts"][m].copy()
DM_ots_amm = {key: DM_ots_amm[key] for key in europe}
DM_fts_amm = {key: DM_fts_amm[key] for key in europe}


#############################
##### FIXED ASSUMPTIONS #####
#############################

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

# costs
filepath = os.path.join(current_file_directory, '../data/datamatrix/' + 'fxa_costs.pickle')
with open(filepath, 'rb') as handle:
    DM = pickle.load(handle)
DM_fxa["cost-matprod"] = DM["costs"]
DM_fxa["cost-CC"] = DM["costs-cc"]

# drop ammonia-tech
DM_fxa_amm["cost-matprod"] = DM_fxa["cost-matprod"].filter({"Categories1" : ["ammonia-tech"]})
DM_fxa["cost-matprod"].drop("Categories1","ammonia-tech")
DM_fxa_amm["cost-CC"] = DM_fxa["cost-CC"].filter({"Categories1" : ["ammonia-tech"]})
DM_fxa["cost-CC"].drop("Categories1","ammonia-tech")

#######################
##### CALIBRATION #####
#######################

files_temp = ['calibration_energy-demand.pickle', 'calibration_material-production.pickle',
              'calibration_emissions.pickle']
names_temp = ['energy-demand', 'material-production',
              'emissions']

for i in range(0, len(files_temp)):
    filepath = os.path.join(current_file_directory, '../data/datamatrix/' + files_temp[i])
    with open(filepath, 'rb') as handle:
        dm = pickle.load(handle)
    DM_cal[names_temp[i]] = dm.copy()

# ammonia
filepath = os.path.join(current_file_directory, '../data/datamatrix/' + "calibration_material-production_ammonia.pickle")
with open(filepath, 'rb') as handle:
    dm = pickle.load(handle)
DM_cal_amm["material-production"] = dm.copy()

# TODO: here we need to add the calibration of energy demand and emissions of ammonia manufacturing.
# for energy demand: probably can be inferred from emissions and constants, though we would need the energy mix (probably we can use the one of chemicals in JRC)
# for emissions: FAOSTAT -> Climate Change -> Totals and Indicators -> Emissions totals -> Fertilizer manufacturing

# note: In practice, ammonia production is often responsible for the bulk of fertilizer manufacturing emissions, 
# but the FAO (and IPCC inventories) keep the broader category since other steps matter too (notably N₂O from nitric acid)
# in my case, for emission factors of ammonia-tech I have taken both CO2 and N2O, so probably in my case ammonia-tech
# is closer to fertilizer manufacturing in general. So it probably makes sense to take emissions from fertilizer
# manufacturing from FAO as calibration for the emission of the Ammonia module. Do this tomorrow.

# #####################
# ##### CONSTANTS #####
# #####################

# CDM_const = DM_industry_copies["constant"].copy()
# CDM_const_amm = DM_ammonia_copies["constant"].copy()

########################
##### PUT TOGETHER #####
########################

DM_industry = {
    'fxa': DM_fxa,
    'fts': DM_fts,
    'ots': DM_ots,
    'calibration': DM_cal,
    # "constant" : CDM_const
}

DM_ammonia = {
    'fxa': DM_fxa_amm,
    'fts': DM_fts_amm,
    'ots': DM_ots_amm,
    'calibration': DM_cal_amm,
    # "constant" : CDM_const_amm
}

################################
###### GENERATE FAKE VAUD ######
################################

def make_fake_country(DM, country):

    for key in ['fxa', 'ots', 'calibration']:
        dm_names = list(DM[key])
        for name in dm_names:
            dm_temp = DM[key][name]
            if country not in dm_temp.col_labels["Country"]:
                idx = dm_temp.idx
                arr_temp = dm_temp.array[idx["Switzerland"],...]
                dm_temp.add(arr_temp[np.newaxis,...], "Country", country)
                dm_temp.sort("Country")
                
    dm_names = list(DM["fts"])
    for name in dm_names:
        for i in range(1,4+1):
            dm_temp = DM["fts"][name][i]
            if country not in dm_temp.col_labels["Country"]:
                idx = dm_temp.idx
                arr_temp = dm_temp.array[idx["Switzerland"],...]
                dm_temp.add(arr_temp[np.newaxis,...], "Country", country)
                dm_temp.sort("Country")

make_fake_country(DM_industry, "Vaud")
make_fake_country(DM_ammonia, "Vaud")

################
##### SAVE #####
################

# save
f = os.path.join(current_file_directory, '../../../../data/datamatrix/industry.pickle')
my_pickle_dump(DM_industry, f)

f = os.path.join(current_file_directory, '../../../../data/datamatrix/ammonia.pickle')
my_pickle_dump(DM_ammonia, f)









