
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

# I assume that in all countries it's the same than it is in CH
dm_emifact_ch = DM_tra["fxa"]["emission-factor-electricity"].filter({"Country" : ["Switzerland"]})
countries = ['Belgium','Bulgaria','Croatia','Cyprus','Czech Republic','Denmark',
             'EU27','Estonia','Finland','France','Germany','Greece','Hungary','Ireland','Italy',
             'Latvia','Lithuania','Luxembourg','Malta','Netherlands','Poland','Portugal',
             'Romania','Slovakia','Slovenia','Spain','Sweden','United Kingdom']
dm_temp = dm_emifact_ch.copy()
dm_temp.rename_col("Switzerland","Austria","Country")
dm_emifact = dm_temp.copy()
for c in countries:
    dm_temp = dm_emifact_ch.copy()
    dm_temp.rename_col("Switzerland",c,"Country")
    dm_emifact.append(dm_temp,"Country")

# save
f = os.path.join(current_file_directory, '../data/datamatrix/fxa_emission-factor-electricity.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_emifact, handle, protocol=pickle.HIGHEST_PROTOCOL)

