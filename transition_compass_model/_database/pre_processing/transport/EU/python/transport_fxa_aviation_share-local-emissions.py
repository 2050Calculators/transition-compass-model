
import os
import warnings
import pickle
warnings.simplefilter("ignore")
from model.common.auxiliary_functions import jrc_iso2_dict, linear_fitting
from _database.pre_processing.routine_JRC import get_jrc_data

# directories
current_file_directory = os.getcwd()

# load current transport pickle
filepath = os.path.join(current_file_directory, '../../../../data/datamatrix/transport.pickle')
with open(filepath, 'rb') as handle:
    DM_tra = pickle.load(handle)

####################
##### GET DATA #####
####################

dict_iso2_jrc = jrc_iso2_dict()

# # get data
# dict_extract = {"database" : "Transport",
#                 "sheet" : "TrAvia_act",
#                 "variable" : "Passenger transport (mio pkm)",
#                 "sheet_last_row" : "International - Extra-EEAwUK",
#                 "sub_variables" : ["Domestic",
#                                     "International - Intra-EEAwUK",
#                                     "International - Extra-EEAwUK"],
#                 "calc_names" : ["domestic","international-int","international-extra"]}
# dm_avi = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# for v in dm_avi.col_labels["Variables"]:
#     dm_avi.rename_col(v, "aviation_" + v, "Variables")
    
# get data vkm
dict_extract = {"database" : "Transport",
                "sheet" : "TrAvia_act",
                "variable" : "Vehicle-km (mio km)",
                "sheet_last_row" : "International - Extra-EEAwUK",
                "sub_variables" : ["Domestic",
                                    "International - Intra-EEAwUK",
                                    "International - Extra-EEAwUK"],
                "calc_names" : ["vkm_domestic","vkm_international-int","vkm_international-extra"]}
dm_vkm_avi = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
# dm_vkm_avi.filter({"Country": ["EU27"]}).datamatrix_plot()
dm_vkm_avi.deepen()

# get data seats
dict_extract = {"database" : "Transport",
                "sheet" : "TrAvia_png",
                "variable" : "Seats available per flight",
                "sheet_last_row" : "International - Extra-EEAwUK",
                "sub_variables" : ["Domestic",
                                    "International - Intra-EEAwUK",
                                    "International - Extra-EEAwUK"],
                "calc_names" : ["seats_domestic","seats_international-int","seats_international-extra"]}
dm_seats_avi = get_jrc_data(dict_extract, dict_iso2_jrc, current_file_directory)
dm_seats_avi.deepen()
dm_seats_avi.units["seats"] = "number"

# get skm
dm_avi = dm_vkm_avi.copy()
dm_avi.append(dm_seats_avi,"Variables")
dm_avi.operation("vkm", "*", "seats","Variables","skm","vkm")

# get share local emissions as vkm domestic + vkm intra / (vkm domestic + vkm intra + vkm extra)
dm_avi.filter({"Variables":["skm"]},inplace=True)
dm_avi.append(dm_avi.groupby({"numerator" : ["domestic","international-int"]}, "Categories1"), "Categories1")
dm_avi.append(dm_avi.groupby({"denominator" : ["domestic","international-int","international-extra"]}, 
                             "Categories1"), "Categories1")
dm_avi.operation("numerator", "/", "denominator", "Categories1", "tra_share-emissions-local","%")
dm_avi = dm_avi.filter({"Categories1" : ["tra_share-emissions-local"]})
dm_avi = dm_avi.flatten()
dm_avi.rename_col("skm_tra_share-emissions-local", "tra_share-emissions-local_aviation", "Variables")
dm_avi.deepen()
dm_avi.units["tra_share-emissions-local"] = "%"

# dm_avi.filter({"Country": ["EU27"]}).datamatrix_plot()


####################
##### MAKE OTS #####
####################

years_fitting = list(range(1990,2023+1))
dm_avi = linear_fitting(dm_avi, years_fitting)
# dm_avi.filter({"Country": ["EU27"]}).datamatrix_plot()

####################
##### MAKE FTS #####
####################

years_fitting = list(range(2025,2050+5,5))
dm_avi = linear_fitting(dm_avi, years_fitting)
# dm_avi.filter({"Country": ["EU27"]}).datamatrix_plot()
# DM_tra["fxa"]["share-local-emissions"].filter({"Country": ["Switzerland"]}).datamatrix_plot()

################
##### SAVE #####
################

# save
f = os.path.join(current_file_directory, '../data/datamatrix/fxa_share-local-emissions.pickle')
with open(f, 'wb') as handle:
    pickle.dump(dm_avi, handle, protocol=pickle.HIGHEST_PROTOCOL)


