
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

# lifetime is a set of constants with the shape of country and time (so fxa), with all fts
# LDV: ICE and PHEV vehicles have lifetime of 13.5 years
# LDV: New technology like BEV and PHEV have lifetimes initially of 5.5, and the 13.5
# 2W: 8 years
# We assume rail lifetime is 30 years, metrotram lifetime is 20 years and bus lifetime is 10 years

# I assume that in all countries it's the same than it is in CH
dm_lifetime_ch = DM_tra["fxa"]["passenger_vehicle-lifetime"].filter({"Country" : ["Switzerland"]})
countries = ['Belgium','Bulgaria','Croatia','Cyprus','Czech Republic','Denmark',
             'EU27','Estonia','Finland','France','Germany','Greece','Hungary','Ireland','Italy',
             'Latvia','Lithuania','Luxembourg','Malta','Netherlands','Poland','Portugal',
             'Romania','Slovakia','Slovenia','Spain','Sweden','United Kingdom']
dm_temp = dm_lifetime_ch.copy()
dm_temp.rename_col("Switzerland","Austria","Country")
dm_lifetime = dm_temp.copy()
for c in countries:
    dm_temp = dm_lifetime_ch.copy()
    dm_temp.rename_col("Switzerland",c,"Country")
    dm_lifetime.append(dm_temp,"Country")
# dm_lifetime.flatten().flatten().filter({"Country" : ["EU27"]}).datamatrix_plot()

# save
f = os.path.join(current_file_directory, '../data/datamatrix/fxa_passenger_vehicle-lifetime.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_lifetime, handle, protocol=pickle.HIGHEST_PROTOCOL)





