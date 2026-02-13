
import _database.pre_processing.transport.Switzerland.get_data_functions.demand_pkm_vkm as get_pkm_vkm_data
import os
import pickle
from model.common.auxiliary_functions import create_years_list

def get_ep2050_data(this_dir, years_ots):

    # VKM demand for LDV, 2W, bus by technology from EP2050
    # EP2050+_Detailergebnisse 2020-2060_Verkehrssektor_alle Szenarien_2022-04-12
    file_url = 'https://www.bfe.admin.ch/bfe/de/home/politik/energieperspektiven-2050-plus.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvMTA0NDE=.html'
    zip_name = os.path.join(this_dir, '../data/EP2050_sectors.zip')
    file_pickle = os.path.join(this_dir, '../data/tra_EP2050_vkm_demand_private.pickle')
    dm_vkm_private = get_pkm_vkm_data.extract_EP2050_transport_vkm_demand(file_url, zip_name, file_pickle)
    dm_vkm_private.filter({"Years" : years_ots}, inplace=True)
    DM = {}
    DM["passenger-and-freight_road_EP-2050"] = dm_vkm_private.copy()
    
    return DM


def run(years_ots):
    
    # get dir
    this_dir = os.path.dirname(os.path.abspath(__file__))
    
    # get ep 2050
    DM = get_ep2050_data(this_dir, years_ots)
    
    # save
    f = os.path.join(this_dir, '../data/datamatrix/calibration_vkm.pickle')
    with open(f, 'wb') as handle:
        pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return DM

if __name__ == "__main__":
    
    # get years ots
    years_ots = create_years_list(1990, 2023, 1)
    
    # run
    run(years_ots)
