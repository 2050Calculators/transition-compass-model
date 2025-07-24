import numpy as np
import os

from model.common.auxiliary_functions import linear_fitting, create_years_list, my_pickle_dump


def calculate_heating_eff_fts(dm_heating_eff, years_fts, maximum_eff):
  dm_heat_pump = dm_heating_eff.filter({'Categories1': ['heat-pump']})
  dm_heating_eff.drop(dim='Categories1', col_label='heat-pump')
  linear_fitting(dm_heating_eff, years_fts, based_on=list(range(2015, 2023)))
  dm_heating_eff.array = np.minimum(dm_heating_eff.array, maximum_eff)
  linear_fitting(dm_heat_pump, years_fts, based_on=list(range(2015, 2023)))
  dm_heating_eff.append(dm_heat_pump, dim='Categories1')
  dm_heating_eff_fts = dm_heating_eff.filter({'Years': years_fts})

  return dm_heating_eff_fts


def run(DM_buildings, years_fts):

  this_dir = os.path.dirname(os.path.abspath(__file__))
  # !FIXME: use the actual values and not the calibration factor
  file = os.path.join(this_dir, '../../../../data/datamatrix/buildings.pickle')

  #########################################
  #####  FLOOR INTENSITY - SPACE/CAP  #####
  #########################################
  dm_space_cap = DM_buildings['ots']['floor-intensity'].copy()
  linear_fitting(dm_space_cap, years_fts)
  DM_buildings['fts']['floor-intensity'] = dict()
  for lev in range(4):
    lev = lev + 1
    DM_buildings['fts']['floor-intensity'][lev] = dm_space_cap.filter(
      {'Years': years_fts})

  #########################################
  #####   HEATING-COOLING BEHAVIOUR   #####
  #########################################
  dm_Tint_heat = DM_buildings['ots']['heatcool-behaviour'].copy()
  linear_fitting(dm_Tint_heat, years_fts)
  DM_buildings['fts']['heatcool-behaviour'] = dict()
  for lev in range(4):
    lev = lev + 1
    DM_buildings['fts']['heatcool-behaviour'][lev] = dm_Tint_heat.filter(
      {'Years': years_fts})

  #########################################
  #####        BUILDING MIX          ######
  #########################################
  dm_building_mix_new = DM_buildings['ots']['building-renovation-rate']['bld_building-mix'].copy()
  dm_building_mix_new.add(np.nan, dim='Years', col_label=years_fts, dummy=True)
  dm_building_mix_new.fill_nans('Years')
  DM_buildings['fts']['building-renovation-rate'] = dict()
  DM_buildings['fts']['building-renovation-rate']['bld_building-mix'] = dict()
  for lev in range(4):
    lev = lev + 1
    DM_buildings['fts']['building-renovation-rate']['bld_building-mix'][
      lev] = dm_building_mix_new.filter({'Years': years_fts})


  #########################################
  #####       RENOVATION-RATE        ######
  #########################################
  dm_rr = DM_buildings['ots']['building-renovation-rate'][
    'bld_renovation-rate'].copy()
  DM_buildings['fts']['building-renovation-rate'][
    'bld_renovation-rate'] = dict()
  dm_rr.add(np.nan, dim='Years', dummy=True, col_label=years_fts)
  dm_rr.fill_nans(dim_to_interp='Years')
  for lev in range(4):
    lev = lev + 1
    DM_buildings['fts']['building-renovation-rate']['bld_renovation-rate'][
      lev] = dm_rr.filter({'Years': years_fts})


  ###########################################
  #####    RENOVATION-REDISTRIBUTION    #####
  ###########################################
  dm_renov_distr = DM_buildings['ots']['building-renovation-rate'][
    'bld_renovation-redistribution'].copy()
  DM_buildings['fts']['building-renovation-rate'][
    'bld_renovation-redistribution'] = dict()
  dm_renov_distr.add(np.nan, dim='Years', dummy=True, col_label=years_fts)
  dm_renov_distr.fill_nans(dim_to_interp='Years')
  for lev in range(4):
    lev = lev + 1
    DM_buildings['fts']['building-renovation-rate'][
      'bld_renovation-redistribution'][lev] = \
      dm_renov_distr.filter({'Years': years_fts})

  #########################################
  #####         DEMOLITION          #######
  #########################################

  dm_demolition_rate = DM_buildings['ots']['building-renovation-rate'][
    'bld_demolition-rate'].copy()
  # Compute average demolition rate in the last 10 years and forecast to future
  idx = dm_demolition_rate.idx
  idx_yrs = [idx[yr] for yr in create_years_list(2012, 2023, 1)]
  val_mean = np.mean(dm_demolition_rate.array[:, idx_yrs, ...], axis=1)
  dm_demolition_rate.add(np.nan, dim='Years', dummy=True, col_label=years_fts)
  for yr in years_fts:
    dm_demolition_rate.array[:, idx[yr], ...] = val_mean
  # FTS
  DM_buildings['fts']['building-renovation-rate'][
    'bld_demolition-rate'] = dict()
  for lev in range(4):
    lev = lev + 1
    DM_buildings['fts']['building-renovation-rate']['bld_demolition-rate'][lev] = \
      dm_demolition_rate.filter({'Years': years_fts})

  ###########################################
  #####    HEATING TECHNOLOGY MIX     #######
  ###########################################
  dm_heating_cat = DM_buildings['ots']['heating-technology-fuel']['bld_heating-technology'].copy()
  dm_heating_cat.add(np.nan, dim='Years', dummy=True, col_label=years_fts)
  dm_heating_cat.fill_nans('Years')
  DM_buildings['fts']['heating-technology-fuel'] = dict()
  DM_buildings['fts']['heating-technology-fuel'][
    'bld_heating-technology'] = dict()
  for lev in range(4):
    lev = lev + 1
    DM_buildings['fts']['heating-technology-fuel']['bld_heating-technology'][
      lev] = dm_heating_cat.filter({'Years': years_fts})

  ############################################
  ######       HEATING EFFICIENCY       ######
  ############################################
  dm_heating_eff = DM_buildings['ots']['heating-efficiency'].copy()
  dm_heating_eff_fts = calculate_heating_eff_fts(dm_heating_eff.copy(),
                                                 years_fts, maximum_eff=0.98)
  DM_buildings['fts']['heating-efficiency'] = dict()
  for lev in range(4):
    lev = lev + 1
    DM_buildings['fts']['heating-efficiency'][lev] = dm_heating_eff_fts

  my_pickle_dump(DM_buildings, file)

  return DM_buildings
