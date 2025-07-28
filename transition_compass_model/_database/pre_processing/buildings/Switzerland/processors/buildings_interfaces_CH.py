import os.path
import pickle

from model.common.auxiliary_functions import my_pickle_dump

def run(country_list):
  # SECTION: Lifestyles to Buildings intereface
  #########################################
  #####   INTERFACE: LFS to BLD     #######
  #########################################
  this_dir = os.path.dirname(os.path.abspath(__file__))
  file = os.path.join(this_dir, '../../../../data/datamatrix/lifestyles.pickle')
  with open(file, 'rb') as handle:
      DM_lifestyles = pickle.load(handle)

  dm_pop_ots = DM_lifestyles['ots']['pop']['lfs_population_'].filter({"Country" : country_list})
  dm_pop_fts = DM_lifestyles['fts']['pop']['lfs_population_'][1].filter({"Country" : country_list})
  dm_pop_ots.append(dm_pop_fts, dim='Years')
  DM_interface_lfs_to_bld = {'pop': dm_pop_ots}


  file = os.path.join(this_dir, '../../../../data/interface/lifestyles_to_buildings.pickle')
  my_pickle_dump(DM_new = DM_interface_lfs_to_bld, local_pickle_file=file)

  # SECTION: Climate to Buildings intereface
  #########################################
  #####   INTERFACE: CLM to BLD     #######
  #########################################
  file = os.path.join(this_dir, '../../../../data/datamatrix/climate.pickle')
  with open(file, 'rb') as handle:
      DM_clm = pickle.load(handle)

  dm_clm_ots = DM_clm['ots']['temp']['bld_climate-impact-space'].filter({"Country" : country_list})
  dm_clm_fts = DM_clm['fts']['temp']['bld_climate-impact-space'][1].filter({"Country" : country_list})
  dm_clm_ots.append(dm_clm_fts, dim='Years')
  DM_interface_clm_to_bld = {'cdd-hdd': dm_clm_ots}

  file = os.path.join(this_dir,  '../../../../data/interface/climate_to_buildings.pickle')
  my_pickle_dump(DM_new=DM_interface_clm_to_bld, local_pickle_file=file)

  return DM_interface_lfs_to_bld

if __name__ == "__main__":
  run(country_list=['Switzerland', 'Vaud'])

