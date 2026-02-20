

# packages
from model.common.auxiliary_functions import my_pickle_dump
import pickle
import os

# from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat

# import plotly.express as px
import plotly.io as pio
# import re
pio.renderers.default='browser'
# import subprocess
import warnings
warnings.simplefilter("ignore")

# directories
current_file_directory = os.getcwd()

###############################################################################
############################### EXECUTE SCRIPTS ###############################
###############################################################################

# subprocess.run(['python', os.path.join(current_file_directory, 'lca_levers.py')])

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
DM_lca = {}

##################
##### LEVERS #####
##################

# list(np.array(files)[[bool(re.search("lever", i)) for i in files]])
lever_files = ['lever_footprint.pickle']
lever_names = ['footprint']

# load dms
for i in range(0, len(lever_files)):
    filepath = os.path.join(current_file_directory, '../data/datamatrix/' + lever_files[i])
    with open(filepath, 'rb') as handle:
        DM = pickle.load(handle)
    DM_ots[lever_names[i]] = DM["ots"]
    DM_fts[lever_names[i]] = DM["fts"]

#############################
##### FIXED ASSUMPTIONS #####
#############################

#######################
##### CALIBRATION #####
#######################

#####################
##### CONSTANTS #####
#####################

########################
##### PUT TOGETHER #####
########################

DM_lca = {
    'fxa': DM_fxa,
    'fts': DM_fts,
    'ots': DM_ots,
    'calibration': DM_cal,
    "constant" : CDM_const
}

################
##### SAVE #####
################

# save
f = os.path.join(current_file_directory, '../../../data/datamatrix/lca.pickle')
my_pickle_dump(DM_lca, f)

