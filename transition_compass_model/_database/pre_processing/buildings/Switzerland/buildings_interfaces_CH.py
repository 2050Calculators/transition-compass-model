import pickle

from model.common.auxiliary_functions import my_pickle_dump

def run():
  # SECTION: Lifestyles to Buildings intereface
  #########################################
  #####   INTERFACE: LFS to BLD     #######
  #########################################
  file = '../../../data/datamatrix/lifestyles.pickle'
  with open(file, 'rb') as handle:
      DM_lifestyles = pickle.load(handle)

  dm_pop_ots = DM_lifestyles['ots']['pop']['lfs_population_'].filter({"Country" : ["Switzerland","Vaud"]})
  dm_pop_fts = DM_lifestyles['fts']['pop']['lfs_population_'][1].filter({"Country" : ["Switzerland","Vaud"]})
  dm_pop_ots.append(dm_pop_fts, dim='Years')
  DM_interface_lfs_to_bld = {'pop': dm_pop_ots}


  file = '../../../data/interface/lifestyles_to_buildings.pickle'
  my_pickle_dump(DM_new = DM_interface_lfs_to_bld, local_pickle_file=file)

  # SECTION: Climate to Buildings intereface
  #########################################
  #####   INTERFACE: CLM to BLD     #######
  #########################################
  file = '../../../data/datamatrix/climate.pickle'
  with open(file, 'rb') as handle:
      DM_clm = pickle.load(handle)

  dm_clm_ots = DM_clm['ots']['temp']['bld_climate-impact-space'].filter({"Country" : ["Switzerland","Vaud"]})
  dm_clm_fts = DM_clm['fts']['temp']['bld_climate-impact-space'][1].filter({"Country" : ["Switzerland","Vaud"]})
  dm_clm_ots.append(dm_clm_fts, dim='Years')
  DM_interface_clm_to_bld = {'cdd-hdd': dm_clm_ots}

  file = '../../../data/interface/climate_to_buildings.pickle'
  my_pickle_dump(DM_new=DM_interface_clm_to_bld, local_pickle_file=file)
  return

if __name__ == "__main__":
  run()

