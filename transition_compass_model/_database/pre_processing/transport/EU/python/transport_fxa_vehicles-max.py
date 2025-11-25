
# import packages
import os
import warnings
import pickle
import numpy as np
warnings.simplefilter("ignore")
from model.common.auxiliary_functions import linear_fitting

# directories
current_file_directory = os.getcwd()

# load current transport pickle
filepath = os.path.join(current_file_directory, '../../../../data/datamatrix/transport.pickle')
with open(filepath, 'rb') as handle:
    DM_tra = pickle.load(handle)

# load seat km
filepath = os.path.join(current_file_directory, '../data/datamatrix/intermediate_files/passenger_vkm.pickle')
with open(filepath, 'rb') as handle:
    dm_vkm = pickle.load(handle)
dm_skm = dm_vkm.filter({"Country" : ["EU27"], "Categories1" : ["aviation"]})

# get max number of seats pickle
dm_max = DM_tra["fxa"]["vehicles-max"].filter({"Country" : ["EU27"]})

# Battery-electric (BEV) aircraft
# Max ~5% of total EU seat-km in 2050 (more realistic central value: 1–2%).
# https://te-cdn.ams3.cdn.digitaloceanspaces.com/files/TE-aviation-decarbonisation-roadmap-FINAL.pdf

# Hydrogen aircraft
# Max ~35–40% of total EU seat-km in 2050.
# https://te-cdn.ams3.cdn.digitaloceanspaces.com/files/Study-Analysing-the-costs-of-hydrogen-aircraft.pdf?utm_source=chatgpt.com

# get 2050 levels
skm_perseat_peryear_bev = 0.7*1e6
skm_perseat_peryear_h2 = 2.5*1e6
dm_max[:,2050,:,:,"BEV"] = dm_skm[:,2050,:,:] * 0.05 / skm_perseat_peryear_bev
dm_max[:,2050,:,:,"H2"] = dm_skm[:,2050,:,:] * 0.40 / skm_perseat_peryear_h2

# put nan in non 2050
# for y in list(range(2030,2045+5,5)):
#     dm_max[:,y,...] = np.nan

# put zeroes in non 2050
for y in list(range(2030,2045+5,5)):
    dm_max[:,y,...] = 0
    
# put nan in 2045
dm_max[:,2045,...] = np.nan

# linear fit
dm_max = linear_fitting(dm_max, list(range(2030,2050+5,5)))

# check
# dm_max.flatten().datamatrix_plot()
# dm_max.filter({"Country" : ["EU27"]}).write_df().to_csv("/Users/echiarot/Desktop/check.csv")

# save
f = os.path.join(current_file_directory, '../data/datamatrix/fxa_vehicles-max.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_max, handle, protocol=pickle.HIGHEST_PROTOCOL)


