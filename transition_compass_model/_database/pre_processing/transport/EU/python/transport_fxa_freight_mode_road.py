
# packages
import pickle
import os
import warnings
warnings.simplefilter("ignore")

# directories
current_file_directory = os.getcwd()

# load current transport pickle
filepath = os.path.join(current_file_directory, '../../../../data/datamatrix/transport.pickle')
with open(filepath, 'rb') as handle:
    DM_tra = pickle.load(handle)

# load tkm pickle
filepath = os.path.join(current_file_directory, '../data/datamatrix/fxa_passenger_vehicle-lifetime.pickle')
with open(filepath, 'rb') as handle:
    dm_lifetime = pickle.load(handle)

####################
##### lifetime #####
####################

# TODO: i do not know where these numbers are coming from, so for the moment I assign
# the ones of CH to my countries (all the same), and then to be checked with Paola

# I assume that in all countries it's the same than it is in CH
dm_ch = DM_tra["fxa"]["freight_mode_road"].filter({"Country" : ["Switzerland"]})
countries = ['Belgium','Bulgaria','Croatia','Cyprus','Czech Republic','Denmark',
             'EU27','Estonia','Finland','France','Germany','Greece','Hungary','Ireland','Italy',
             'Latvia','Lithuania','Luxembourg','Malta','Netherlands','Poland','Portugal',
             'Romania','Slovakia','Slovenia','Spain','Sweden','United Kingdom']
dm_temp = dm_ch.copy()
dm_temp.rename_col("Switzerland","Austria","Country")
dm = dm_temp.copy()
for c in countries:
    dm_temp = dm_ch.copy()
    dm_temp.rename_col("Switzerland",c,"Country")
    dm.append(dm_temp,"Country")

# save
f = os.path.join(current_file_directory, '../data/datamatrix/fxa_freight_mode_road.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)
