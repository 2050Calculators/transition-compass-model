
# packages
from model.common.data_matrix_class import DataMatrix
import pickle
import warnings
warnings.simplefilter("ignore")

# load current transport pickle
filepath = '../../../../data/datamatrix/transport.pickle'
with open(filepath, 'rb') as handle:
    DM_tra = pickle.load(handle)

# load tkm pickle
filepath = '../data/datamatrix/intermediate_files/freight_tkm.pickle'
with open(filepath, 'rb') as handle:
    DM_tkm = pickle.load(handle)
    
# load freight fleet
filepath = '../data/datamatrix/intermediate_files/freight_fleet.pickle'
with open(filepath, 'rb') as handle:
    dm_fleet = pickle.load(handle)
    
# load renewal rate pickle
filepath = '../data/datamatrix/intermediate_files/passenger_renewal-rate.pickle'
with open(filepath, 'rb') as handle:
    dm_renrate = pickle.load(handle)

###############
##### TKM #####
###############

DM_tra["fxa"]["freight_mode_other"].units

# get total tkm
dm_tkm = DM_tkm["ots"]["freight_tkm"].copy()
dm_tkm.append(DM_tkm["fts"]["freight_tkm"][1],"Years")

# check
# dm_tkm.filter({"Country" : ["EU27"]}).datamatrix_plot()

# get fleet
dm_fleet_agg = dm_fleet.group_all("Categories2", inplace=False)
# df = dm_fleet_agg.write_df()

# get tkm by vehicle
dm_tkm.append(dm_fleet_agg,"Variables")
# df_tkm = dm_tkm.write_df()
dm_tkm.operation("tra_freight_tkm","/","tra_freight_technology-share_fleet",
                 out_col="tra_freight_tkm-by-veh",unit="tkm")
dm_tkm.drop("Variables",['tra_freight_tkm', 'tra_freight_technology-share_fleet'])

# filter
dm_tkm.filter({"Categories1" : ['IWW', 'aviation', 'marine', 'rail']}, inplace=True)

# TODO: even though this is tkm/num, i leave it tkm as this is how the pickle is
# currently built (to be changed at some point)

########################
##### RENEWAL RATE #####
########################

# TODO: for the moment I take some values from CH

dm_renrate = DM_tra["fxa"]["freight_mode_other"].filter({"Variables" : ["tra_freight_renewal-rate"]})
df = dm_renrate.write_df()
dm_renrate = DataMatrix.create_from_df(df, 1) # doing this to fix the years

# take renewal rate rail CEV as proxy for renewal rate of rail and aviation, and for IWW and marine put missing
dm_renrate_freight = dm_renrate.copy()
dm_renrate_freight["EU27",...] = dm_renrate_freight["Switzerland",...]
dm_renrate_freight = dm_renrate_freight.filter({"Country" : ["EU27"]})
countries = dm_tkm.col_labels["Country"].copy()
countries.remove("EU27")
for c in countries:
    arr_temp = dm_renrate_freight["EU27",...]
    dm_renrate_freight.add(arr_temp, "Country", c)
dm_renrate_freight.sort("Country")

# fix marine
dm_renrate_freight[:,:,:,"marine"] = dm_renrate_freight[:,:,:,"marine"]-1

# # check
# dm_renrate_freight.filter({"Country" : ["EU27"]}).datamatrix_plot()
# df = dm_renrate_freight

# TODO: check the renewal rate, 15% seems a bit high for ships

########################
##### PUT TOGETHER #####
########################

dm_mode_oth = dm_tkm.copy()
dm_mode_oth.append(dm_renrate_freight,"Variables")
dm_mode_oth.sort("Variables")

# save
f = '../data/datamatrix/fxa_freight_mode_other.pickle'
with open(f, 'wb') as handle:
    pickle.dump(dm_mode_oth, handle, protocol=pickle.HIGHEST_PROTOCOL)

