
# import packages
import os
import warnings
import pickle
import numpy as np
warnings.simplefilter("ignore")

# directories
current_file_directory = os.getcwd()

# load current transport pickle
filepath = os.path.join(current_file_directory, '../../../../data/datamatrix/transport.pickle')
with open(filepath, 'rb') as handle:
    DM_tra = pickle.load(handle)

# load utilization rate
filepath = os.path.join(current_file_directory, '../data/datamatrix/fxa_share-local-emissions.pickle')
with open(filepath, 'rb') as handle:
    dm_fxa = pickle.load(handle)

# FUEL-MIX
# According to ATAG, Waypoint 2050 (2021) (plot page 47)
# https://aviationbenefits.org/media/167417/w2050_v2021_27sept_full.pdf
# There will be 600-140 = 460 Mtoe of SAF available world-wide.
# 1 toe = 41868 MJ
# Of these, we allocate 0.6% to Switzerland. As EU27+UK accounted for roughly 15–17% of global CO₂ emissions from aviation, for EU27 we put 13%
val_SAF_2050_ATAG_MJ = 450*1e6 * 41868 *0.13
# Alternatively, the paper Abrantes et al. (2021) "Sustainable aviation fuels and imminent technologies - CO2 emissions evolution towards 2050"
# https://doi.org/10.1016/j.jclepro.2021.127937
# Gives as max value for SAF in 2050 200 Mt. To convert to energy we use 43 MJ/kg
val_SAF_2050_paper_MJ = 200*1e9 * 43 * 0.13

dm_SAF = dm_fxa.flatten()
dm_SAF.add(np.nan, dim='Variables', col_label='tra_passenger-max-SAF_aviation', unit='MJ', dummy=True)
dm_SAF = dm_SAF.filter({'Variables': ['tra_passenger-max-SAF_aviation']})
dm_SAF.deepen()
dm_SAF['EU27', 2050, 'tra_passenger-max-SAF', 'aviation'] = val_SAF_2050_paper_MJ
dm_SAF['EU27', 2025, 'tra_passenger-max-SAF', 'aviation'] = 0
dm_SAF['EU27', 2030, 'tra_passenger-max-SAF', 'aviation'] = 0
dm_SAF.fill_nans('Years')

# Extract maximum fuel available
dm_tmp = dm_SAF.filter({'Variables': ['tra_passenger-max-SAF']})
dm_tmp.rename_col('tra_passenger-max-SAF', 'tra_passenger_available-fuel-mix_biofuel', dim='Variables')
dm_tmp.deepen(based_on='Variables')
dm_tmp.add(0, dim='Categories2', col_label='efuel', dummy=True)
dm_tmp.switch_categories_order()
# df = dm_tmp.write_df()

# save
f = os.path.join(current_file_directory, '../data/datamatrix/fxa_fuel-mix-availability.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_tmp, handle, protocol=pickle.HIGHEST_PROTOCOL)



