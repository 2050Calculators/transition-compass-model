import numpy as np
from model.common.data_matrix_class import DataMatrix
from model.common.auxiliary_functions import create_years_list, dm_add_missing_variables, moving_average

def compute_stock_fts(DM, years_ots, years_fts):
  dm = DM['ots'].copy()
  del DM['ots']
  # Have value every years
  years_missing = list(
    set(range(years_ots[0], years_fts[-1] + 1)) - set(years_ots + years_fts))
  for key, dm_k in DM.items():
    dm_k.add(np.nan, dummy=True, dim='Years', col_label=years_missing)
    dm_k.sort('Years')
    dm_k.fill_nans(dim_to_interp='Years')
    if key == 'building-mix-new':
      # Normalise:
      dm_k = dm_k.flatten()
      dm_k.normalise(dim='Categories1')
      dm_k.deepen()

  dm.filter({'Variables': ['bld_floor-area_stock', 'bld_floor-area_renovated',
                           'bld_floor-area_new', 'bld_floor-area_waste']},
            inplace=True)
  years_missing = list(
    set(range(years_ots[0], years_fts[-1] + 1)) - set(years_ots))
  dm.add(np.nan, dummy=True, dim='Years', col_label=years_missing)

  dm_rr = DM['renovation-rate']
  dm_renov_distr = DM['renovation-distribution']
  dm_demand = DM['total-stock']

  dm.append(DM['building-mix-new'], dim='Variables')
  dm_dem_rate = DM['demolition-rate']
  dm_dem_distr = DM['demolition-distribution']

  idx = dm.idx
  idx_r = dm_rr.idx
  idx_rd = dm_renov_distr.idx
  idx_s = dm_demand.idx
  idx_d = dm_dem_rate.idx
  idx_dr = dm_dem_distr.idx

  future_years = list(set(dm.col_labels['Years']) - set(years_ots))
  future_years.sort()
  for ti in future_years:
    # ni = n[idx[ti]]
    # s_tm1 = ((ni - 1) / ni * s_t + 1 / n * s_tmn
    s_t_tot = dm_demand.array[:, idx_s[ti], idx_s['bld_floor-area_total']]
    s_tm1 = dm.array[:, idx[ti - 1], idx['bld_floor-area_stock'], ...]
    dem_tm1 = dm_dem_rate.array[:, idx_d[ti - 1], idx_d['bld_demolition-rate'],
              ...]
    dem_distr_tm1 = dm_dem_distr.array[:, idx_dr[ti - 1],
                    idx_dr['bld_to-demolish'], :, :] * s_tm1
    dem_distr_tm1 = dem_distr_tm1 / np.nansum(dem_distr_tm1, axis=-1,
                                              keepdims=True)
    w_t = np.nansum(s_tm1, axis=-1, keepdims=True) * dem_tm1[
      ..., np.newaxis] * dem_distr_tm1
    s_tm1_w_t = s_tm1 - w_t
    arr_rr = np.nansum(
      s_tm1_w_t * dm_rr.array[:, idx_r[ti], idx_r['bld_renovation-rate'], :,
                  np.newaxis], axis=-1,
      keepdims=True)
    r_t = arr_rr \
          * (dm_renov_distr.array[:, idx_rd[ti],
             idx_rd['bld_renovation-redistribution-in'], np.newaxis, :] +
             - dm_renov_distr.array[:, idx_rd[ti],
               idx_rd['bld_renovation-redistribution-out'], np.newaxis, :])
    # check for s_tm1 + r_t < 0 ( I'm renovating more building than what is available)
    mask_negative = np.any(s_tm1_w_t + r_t < 0, axis=2, keepdims=True)
    if mask_negative.any():
      mask_negative = mask_negative.repeat(r_t.shape[-1], axis=-1)
      # this most likely mean that I have renovated/demolished all cat.F buildings
      ren_distr_in = dm_renov_distr.array[:, idx_rd[ti],
                     idx_rd['bld_renovation-redistribution-in'], np.newaxis, :]
      ren_distr_in = ren_distr_in.repeat(r_t.shape[1], axis=1)
      arr_rr = arr_rr.repeat(ren_distr_in.shape[-1], axis=-1)
      r_t[mask_negative] = arr_rr[mask_negative] * (
          -dem_distr_tm1[mask_negative] + ren_distr_in[mask_negative])

    # TOTAL: n_t = s(t) - s(t-1) + w(t) (without cat split)
    n_t_tot = s_t_tot - np.nansum(s_tm1_w_t, axis=(-1, -2))
    n_t = n_t_tot[..., np.newaxis, np.newaxis] * dm.array[:, idx[ti],
                                                 idx['bld_building-mix_new'],
                                                 ...]
    s_t = s_tm1_w_t + n_t + r_t
    dm.array[:, idx[ti], idx['bld_floor-area_stock'], ...] = s_t
    dm.array[:, idx[ti], idx['bld_floor-area_renovated'], ...] = r_t
    dm.array[:, idx[ti], idx['bld_floor-area_new'], ...] = n_t
    dm.array[:, idx[ti], idx['bld_floor-area_waste'], ...] = w_t

  # dm.filter({'Years': years_ots + years_fts}, inplace=True)

  return dm


def bld_floor_area_workflow(DM_floor_area, dm_lfs, cdm_const, years_ots,
                            years_fts):
  # SECTION FLOOR AREA
  # SECTION Floor demand = pop x floor-intensity
  # Floor area and material workflow
  # s(t)[m2] = pop(t)[ppl] x area-demand(t)[m2/cap]
  dm_floor_demand = DM_floor_area['floor-intensity'].filter(
    {'Variables': ['lfs_floor-intensity_space-cap']},
    inplace=False)
  dm_floor_demand.append(dm_lfs, dim='Variables')
  dm_floor_demand.operation('lfs_population_total', '*',
                            'lfs_floor-intensity_space-cap',
                            out_col='bld_floor-area_total', unit='m2')

  # SECTION OTS
  #################
  ####   OTS   ####
  #################
  # SECTION Floor area stock by bld type and cat = floor demand x building-mix
  # Compute building stock by energy category as floor-area m2 x building-stock-mix
  dm_floor_demand_ots = dm_floor_demand.filter({'Years': years_ots})
  dm_bld_ots = DM_floor_area['bld_type'].filter({'Years': years_ots})
  idx_d = dm_floor_demand_ots.idx
  idx_b = dm_bld_ots.idx
  arr = dm_floor_demand_ots.array[:, :, idx_d['bld_floor-area_total'],
        np.newaxis, np.newaxis] \
        * dm_bld_ots.array[:, :, idx_b['bld_building-mix_stock'], :, :]
  dm_bld_ots.add(arr, dim='Variables', col_label='bld_floor-area_stock',
                 unit='m2')

  # compute waste
  # SECTION Waste = dem-rate x Stock
  # I want to distribute the demolition rate based on the construction year of the envelope.
  # first I do dem-rate_b(t-1) x s_b(t-1) = w_b(t) (b = sfh, mfh)
  # Then I distribute the waste by energy category, based on stock, but only for those categories built at least
  # 50 years ago ( I need this matrix in fxa )
  # w(t) = dem-rate(t-1) x s(t-1)
  dm_demolition = DM_floor_area['bld_demolition-rate'].filter(
    {'Years': years_ots})
  dm_stock_cat = dm_bld_ots.filter({'Variables': ['bld_floor-area_stock']})
  dm_stock_tot = dm_stock_cat.group_all('Categories2', inplace=False)
  dm_demolition.append(dm_stock_tot, dim='Variables')
  # dm_bld_ots.append(dm_demolition_rate.filter({'Years': years_ots}), dim='Variables')
  dm_demolition.lag_variable('bld_demolition-rate', shift=1, subfix='_tm1')
  dm_demolition.lag_variable('bld_floor-area_stock', shift=1, subfix='_tm1')
  dm_demolition.operation('bld_floor-area_stock_tm1', '*',
                          'bld_demolition-rate_tm1',
                          out_col='bld_floor-area_waste',
                          unit='m2')
  # !FIXME consider moving this to pre-processing if it is only for ots
  # Create 0/1 array of bld that can be demolished for each year, bld-type, energy category
  dm_dem_distr = DM_floor_area['bld_age']
  dm_dem_distr.rename_col('bld_age', 'bld_to-demolish', 'Variables')
  mask_gt_50 = (dm_dem_distr.array > 50)
  dm_dem_distr.array[mask_gt_50] = 1
  dm_dem_distr.array[~mask_gt_50] = 0
  dm_demolished = dm_dem_distr.filter({'Years': years_ots})
  # distribute waste by envelope category based on stock available of bld that can be demolished
  dm_demolished.append(dm_stock_cat, dim='Variables')
  dm_demolished.operation('bld_to-demolish', '*', 'bld_floor-area_stock',
                          out_col='bld_demolition-distribution', unit='m2')
  dm_demolished.filter({'Variables': ['bld_demolition-distribution']},
                       inplace=True)
  dm_demolished.normalise(dim='Categories2')
  idx_1 = dm_demolished.idx
  idx_2 = dm_demolition.idx
  arr = dm_demolished.array[:, :, idx_1['bld_demolition-distribution'], ...] \
        * dm_demolition.array[:, :, idx_2['bld_floor-area_waste'], :,
          np.newaxis]
  dm_bld_ots.add(arr, dim='Variables', col_label='bld_floor-area_waste',
                 unit='m2')

  # SECTION Renovated = rr x Redistribution x Stock
  # r(t) = R(t) x s(t-1)
  dm_rr_ots = DM_floor_area['bld_renovation-rate'].filter({'Years': years_ots})
  dm_renov_redistr_ots = DM_floor_area['bld_renovation-redistribution'].filter(
    {'Years': years_ots})
  idx = dm_bld_ots.idx
  idx_r = dm_rr_ots.idx
  idx_d = dm_renov_redistr_ots.idx
  dm_bld_ots.lag_variable('bld_floor-area_stock', shift=1, subfix='_tm1')
  arr = np.nansum(dm_bld_ots.array[:, :, idx['bld_floor-area_stock_tm1'], :, :]
                  * dm_rr_ots.array[:, :, idx_r['bld_renovation-rate'], :,
                    np.newaxis], axis=-1, keepdims=True) \
        * (dm_renov_redistr_ots.array[:, :,
           idx_d['bld_renovation-redistribution-in'], np.newaxis, :]
           - dm_renov_redistr_ots.array[:, :,
             idx_d['bld_renovation-redistribution-out'], np.newaxis, :])
  dm_bld_ots.add(arr, dim='Variables', col_label='bld_floor-area_renovated',
                 unit='m2')

  # SECTION New(t) = Stock(t) - Stock(t-1) + Waste(t) - Renovated(t)
  # s(t) = s(t-1) + n(t) - w(t) + r(t)
  # n(t) = s(t) - s(t-1) + w(t) - r(t)
  idx = dm_bld_ots.idx
  arr_new = dm_bld_ots.array[:, :, idx['bld_floor-area_stock'], ...] \
            - dm_bld_ots.array[:, :, idx['bld_floor-area_stock_tm1'], ...] \
            + dm_bld_ots.array[:, :, idx['bld_floor-area_waste'], ...] \
            - dm_bld_ots.array[:, :, idx['bld_floor-area_renovated'], ...]
  dm_bld_ots.add(arr_new, dim='Variables', unit='m2',
                 col_label='bld_floor-area_new')

  # SECTION FTS
  #################
  ####   FTS   ####
  #################
  # SECTION Stock fts
  # I need to know how many m2 are needed in 2024. Then I look at 2023 and I know the stock split in 2023.
  # I also apply the demolition rate and obtain the waste generated. I can obtain the total new m2 per mfh and sfh.
  # You can take the new m2 and apply the share by energy cat. Now I have the s_c(t-1), n_c(t), w_c(t)
  # I take s_c(t-1), I apply the renovation-rate and I obtain the renovated stock r_c(t). I can now use the following:
  # s_c(t) = s_c(t-1) + n_c(t) - w_c(t) + r_c(t)
  DM = {'ots': dm_bld_ots,
        'demolition-rate': DM_floor_area['bld_demolition-rate'],
        'demolition-distribution': dm_dem_distr,
        'building-mix-new': DM_floor_area['bld_building-mix'],
        'renovation-rate': DM_floor_area['bld_renovation-rate'],
        'renovation-distribution': DM_floor_area[
          'bld_renovation-redistribution'],
        'total-stock': dm_floor_demand.filter(
          {'Variables': ['bld_floor-area_total']})}

  dm_bld_tot = compute_stock_fts(DM, years_ots, years_fts)
  del DM, dm_bld_ots

  ########################
  ####   CUMULATED    ####
  ########################
  # SECTION Cumulated floor area
  dm_cumulated = dm_bld_tot.filter(
    {'Variables': ['bld_floor-area_stock', 'bld_floor-area_new',
                   'bld_floor-area_renovated']})
  # Interpolate to have data every year
  idx = dm_cumulated.idx
  arr = np.cumsum(dm_cumulated.array[:, :, idx['bld_floor-area_new'], :, :],
                  axis=1)
  dm_cumulated.add(arr, dim='Variables',
                   col_label='bld_floor-area_new-cumulated', unit='m2')
  arr = np.cumsum(
    dm_cumulated.array[:, :, idx['bld_floor-area_renovated'], :, :], axis=1)
  arr = np.maximum(arr, 0)
  dm_cumulated.add(arr, dim='Variables',
                   col_label='bld_floor-area_renovated-cumulated', unit='m2')
  arr = dm_cumulated.array[:, :, idx['bld_floor-area_stock'], :, :] \
        - dm_cumulated.array[:, :, idx['bld_floor-area_new-cumulated'], :, :] \
        - dm_cumulated.array[:, :, idx['bld_floor-area_renovated-cumulated'], :,
          :]
  dm_cumulated.add(arr, dim='Variables',
                   col_label='bld_floor-area_unrenovated-cumulated', unit='m2')

  dm_cumulated.group_all('Categories2')
  dm_cumulated.group_all('Categories1')
  dm_cumulated.filter({'Variables': ['bld_floor-area_unrenovated-cumulated',
                                     'bld_floor-area_new-cumulated',
                                     'bld_floor-area_renovated-cumulated'],
                       'Years': years_ots + years_fts},
                      inplace=True)

  dm_bld_tot.filter({'Years': years_ots + years_fts}, inplace=True)

  # SECTION Prepare output
  dm_industry = dm_bld_tot.filter({'Variables': ['bld_floor-area_new',
                                                 'bld_floor-area_renovated',
                                                 'bld_floor-area_waste',
                                                 'bld_floor-area_stock']})
  dm_industry[:, :, 'bld_floor-area_renovated', ...] = np.maximum(0,
                                                                  dm_industry[:,
                                                                  :,
                                                                  'bld_floor-area_renovated',
                                                                  ...])  # Remove negative renovation
  dm_industry.group_all('Categories2')
  dm_industry.groupby({'residential': '.*'}, dim='Categories1', regex=True,
                      inplace=True)
  DM_industry = {}
  DM_industry["floor-area"] = dm_industry.copy()

  # make dummy domapp
  dm_domapp = dm_industry.filter({"Variables": ['bld_floor-area_new',
                                                'bld_floor-area_waste',
                                                'bld_floor-area_stock']})
  dm_domapp.rename_col_regex("floor-area", "domapp", "Variables")
  dm_domapp.rename_col("residential", "fridge", "Categories1")
  missing = ["freezer", "dishwasher", "wmachine", "dryer"]
  for m in missing:
    dm_domapp.add(np.nan, "Categories1", m, "number", True)
  dm_domapp.sort("Categories1")
  dm_domapp[...] = 100
  for v in dm_domapp.col_labels["Variables"]:
    dm_domapp.units[v] = "number"
  DM_industry["domapp"] = dm_domapp.copy()

  # make dummy electronics
  dm_elec = dm_domapp.filter({"Categories1": ["dishwasher"]})
  dm_elec.rename_col_regex("domapp", "electronics", "Variables")
  dm_elec.rename_col("dishwasher", "phone", "Categories1")
  missing = ["computer", "tv"]
  for m in missing:
    dm_elec.add(np.nan, "Categories1", m, "number", True)
  dm_elec.sort("Categories1")
  dm_elec[...] = 100
  DM_industry["electronics"] = dm_elec.copy()

  dm_stock = dm_bld_tot.filter({'Variables': ['bld_floor-area_stock']})
  dm_stock.array = np.maximum(dm_stock.array, 0)

  dm_stock = dm_bld_tot.filter({'Variables': ['bld_floor-area_stock']})
  dm_stock.change_unit('bld_floor-area_stock', old_unit='m2', new_unit='Mm2',
                       factor=1e-6)

  DM_floor_out = \
    {'TPE': {'floor-area-cumulated': dm_cumulated,
             'floor-area-cat': dm_stock.group_all('Categories1', inplace=False),
             'floor-area-bld-type': dm_stock.group_all('Categories2',
                                                       inplace=False)},
     'wf-energy': dm_bld_tot,
     'industry': DM_industry}

  return DM_floor_out


def bld_energy_workflow(DM_energy, dm_clm, dm_floor_area, cdm_const):
  # SECTION ENERGY
  # SECTION Adjusted HDD, CDD
  ####    Modify HDD and CDD   ####
  dm_Tint = DM_energy['heatcool-behaviour']
  idx_t = dm_Tint.idx
  idx_c = dm_clm.idx
  # HDD = HDD_ref + N_15 x (Tint - 18)
  arr = dm_clm.array[:, :, idx_c['clm_HDD'], np.newaxis, np.newaxis] \
        + dm_clm.array[:, :, idx_c['clm_days-below-15'], np.newaxis, np.newaxis] \
        * (dm_Tint.array[:, :, idx_t['bld_Tint-heating'], :, :] - 18)
  dm_Tint.add(arr, dim='Variables', col_label='bld_HDD-adj', dummy=True,
              unit='daysK')
  # CDD = CDD_ref + N_24 x (21 - Tint)
  arr = dm_clm.array[:, :, idx_c['clm_CDD'], np.newaxis, np.newaxis] \
        + dm_clm.array[:, :, idx_c['clm_days-above-24'], np.newaxis, np.newaxis] \
        * (21 - dm_Tint.array[:, :, idx_t['bld_Tint-cooling'], :, :])
  dm_Tint.add(arr, dim='Variables', col_label='bld_CDD-adj', dummy=True,
              unit='daysK')

  # SECTION Heating energy demand
  # dH = kH (24 HDD Us A − IG)
  dm_uvalue = DM_energy['u-value']
  dm_A = DM_energy['surface-to-floorarea']
  idx_a = dm_A.idx
  idx_u = dm_uvalue.idx
  idx = dm_floor_area.idx
  # Power per degree
  # 24 * Us * A * m2
  arr_W_K = dm_floor_area.array[:, :, idx['bld_floor-area_stock'], :, :] \
            * dm_uvalue.array[:, :, idx_u['bld_u-value'], :, :] \
            * dm_A.array[:, :, idx_a['bld_surface-to-floorarea'], :, np.newaxis]
  dm_floor_area.add(arr_W_K, dim='Variables', col_label='bld_power-per-K',
                    unit='W/K')
  # Yearly heating energy demand
  # W/K x HDD x 24
  idx = dm_floor_area.idx
  idx_c = dm_Tint.idx
  arr_energy = dm_floor_area.array[:, :, idx['bld_power-per-K'], :, :] \
               * dm_Tint.array[:, :, idx_c['bld_HDD-adj'], :, :] * 24
  dm_floor_area.add(arr_energy, dim='Variables', col_label='bld_heating',
                    unit='Wh')
  dm_floor_area.change_unit('bld_heating', 1e-3, 'Wh', 'kWh')

  # SECTION Energy demand = heating demand x tech x efficiency
  # Energy demand
  # heating demand x tech x efficiency
  dm_tech = DM_energy['heating-technology']
  dm_eff = DM_energy['heating-efficiency']
  idx = dm_floor_area.idx
  idx_e = dm_eff.idx
  idx_t = dm_tech.idx
  arr = dm_floor_area.array[:, :, idx['bld_heating'], :, :, np.newaxis] \
        * dm_tech.array[:, :, idx_t['bld_heating-mix'], :, :, :]
  dm_energy = DataMatrix.based_on(arr[:, :, np.newaxis, ...], dm_floor_area,
                                  change={'Variables': ['bld_heating'],
                                          'Categories3': dm_tech.col_labels[
                                            'Categories3']},
                                  units={'bld_heating': 'kWh'})
  dm_energy.change_unit('bld_heating', 1e-9, 'kWh', 'TWh')

  idx = dm_energy.idx
  arr = dm_energy.array[:, :, idx['bld_heating'], :, :, :] / \
        dm_eff.array[:, :, idx_e['bld_heating-efficiency'], np.newaxis, :, :]
  dm_energy.add(arr, dim='Variables', col_label='bld_energy-demand_heating',
                unit='TWh')

  # SECTION Calibrate heating energy demand
  dm_calib = DM_energy['heating-calibration']
  dm_calib.sort('Categories1')
  dm_energy.array[...] = dm_energy.array[...] * dm_calib.array[:, :, :,
                                                np.newaxis, np.newaxis, :]

  # write datamatrix to pickle
  # current_file_directory = os.path.dirname(os.path.abspath(__file__))
  # directory = os.path.join(current_file_directory, '../_database/pre_processing/buildings/Switzerland/data')
  # if not os.path.exists(directory):
  #    os.makedirs(directory)
  # f = os.path.join(directory, 'heating_energy.pickle')
  # with open(f, 'wb') as handle:
  #    pickle.dump(dm_energy, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # SECTION Cooling
  # dC = gamma kC (24 CDD Us A − IG)
  # gamma x W/K x CDD x 24
  idx = dm_floor_area.idx
  idx_c = dm_Tint.idx
  idx_m = dm_clm.idx
  arr_energy = dm_floor_area.array[:, :, idx['bld_power-per-K'], :, :] \
               * dm_Tint.array[:, :, idx_c['bld_CDD-adj'], :, :] * 24 \
               * dm_clm.array[:, :, idx_m['clm_AC-uptake'], np.newaxis,
                 np.newaxis]
  dm_floor_area.add(arr_energy, dim='Variables', col_label='bld_cooling',
                    unit='Wh')
  dm_floor_area.change_unit('bld_cooling', 1e-3, 'Wh', 'kWh')

  # Energy demand
  # cooling demand x tech x efficiency (we assume the technology is the heat-pump)
  idx = dm_floor_area.idx
  arr = dm_floor_area.array[:, :, idx['bld_cooling'], :, :] \
        / dm_eff.array[:, :, idx_e['bld_heating-efficiency'], np.newaxis, :,
          idx_e['heat-pump']]
  idx = dm_energy.idx
  dm_energy.add(np.nan, dim='Variables', dummy=True,
                col_label='bld_energy-demand_cooling', unit='kWh')
  dm_energy.array[:, :, idx['bld_energy-demand_cooling'], :, :,
  idx['heat-pump']] = arr
  dm_energy.change_unit('bld_energy-demand_cooling', 1e-9, 'kWh', 'TWh')

  # SECTION Emissions
  ###################
  ###  EMISSIONS  ###
  ###################
  # Filter fossil fuels
  cdm_emission = cdm_const['emissions']
  dm_emissions = dm_energy.filter(
    {'Categories3': cdm_emission.col_labels['Categories1'],
     'Variables': ['bld_energy-demand_heating']})
  # Compute emissions fossil fuels
  dm_emissions.sort('Categories3')
  cdm_emission.sort('Categories1')
  idx = dm_emissions.idx
  idx_e = cdm_emission.idx
  arr = dm_emissions.array[:, :, idx['bld_energy-demand_heating'], :, :, :] * \
        cdm_emission.array[np.newaxis, np.newaxis, idx_e['bld_CO2-factors'],
        np.newaxis, np.newaxis, :]
  dm_emissions.add(arr, dim='Variables', col_label='bld_CO2-emissions_heating',
                   unit='kt')
  dm_emissions.change_unit('bld_CO2-emissions_heating', 1e-3, 'kt', 'Mt')
  dm_emissions.filter({'Variables': ['bld_CO2-emissions_heating']},
                      inplace=True)

  # Compute emissions electricity and heat-pump
  dm_elec = dm_energy.filter({'Categories3': ['electricity', 'heat-pump'],
                              'Variables': ['bld_energy-demand_heating']})
  dm_emis_elec = DM_energy['electricity-emission']
  idx = dm_elec.idx
  idx_e = dm_emis_elec.idx
  arr = dm_elec.array[:, :, idx['bld_energy-demand_heating'], :, :, :] \
        * dm_emis_elec.array[:, :, idx_e['bld_CO2-factor'],
          idx_e['electricity'], np.newaxis, np.newaxis, np.newaxis]
  dm_elec.add(arr, dim='Variables', col_label='bld_CO2-emissions_heating',
              unit='kt')
  dm_elec.change_unit('bld_CO2-emissions_heating', 1e-3, 'kt', 'Mt')
  dm_elec.filter({'Variables': ['bld_CO2-emissions_heating']}, inplace=True)

  # Join fossil and electricity
  dm_emissions.append(dm_elec, dim='Categories3')

  dm_emiss_by_class = dm_emissions.group_all('Categories3', inplace=False)
  dm_emiss_by_class.group_all('Categories1')
  dm_emissions.group_all('Categories2')
  dm_emissions.group_all('Categories1')

  # SECTION Prepare output
  #########################
  ###   PREPARE OUTPUT  ###
  #########################
  # Energy demand by type of fuel
  dm_fuel = dm_energy.group_all('Categories2', inplace=False)
  dm_fuel.group_all('Categories1', inplace=True)
  dm_fuel.filter({'Variables': ['bld_energy-demand_heating', 'bld_heating']})
  dm_fuel.add(np.nan, dummy=True, dim='Categories1', col_label='ambient-heat')
  dm_fuel[:, :, 'bld_energy-demand_heating', 'ambient-heat'] \
    = dm_fuel[:, :,'bld_heating', 'heat-pump'] - dm_fuel[:, :, 'bld_energy-demand_heating', 'heat-pump']

  # Heating demand by type of building
  dm_class = dm_floor_area.group_all('Categories1', inplace=False)
  dm_class.filter({'Variables': ['bld_heating']}, inplace=True)
  dm_class.change_unit('bld_heating', 1e-9, 'kWh', 'TWh')
  dm_class.append(dm_emiss_by_class, dim='Variables')

  #
  dm_power = dm_energy.group_all('Categories2', inplace=False)
  dm_power.group_all('Categories1')
  dm_refinery = dm_power.filter({'Variables': ['bld_energy-demand_heating'],
                                 'Categories1': ['gas', 'heating-oil', 'coal']})
  dm_district_heating = dm_power.filter(
    {'Variables': ['bld_energy-demand_heating'],
     'Categories1': ['district-heating']})
  dm_power.filter(
    {'Variables': ['bld_energy-demand_heating', 'bld_energy-demand_cooling'],
     'Categories1': ['electricity', 'heat-pump']}, inplace=True)

  DM_energy_out = {'TPE': {'energy-emissions-by-class': dm_class,
                           'energy-demand-heating': dm_fuel.filter({'Variables': ['bld_energy-demand_heating', 'bld_heating']}),
                           'energy-demand-cooling': dm_fuel.filter(
                             {'Variables': ['bld_energy-demand_cooling'],
                              'Categories1': ['heat-pump']}),
                           'emissions': dm_emissions},
                   'power': dm_fuel.copy(),
                   'district-heating': dm_district_heating,
                   'agriculture': dm_fuel.filter(
                     {'Variables': ['bld_energy-demand_heating'],
                      'Categories1': ['wood']}),
                   'refinery': dm_refinery}

  return DM_energy_out


def bld_appliances_workflow(DM_appliances, dm_pop):
  dm_households = DM_appliances['household-size']
  dm_households.append(dm_pop, dim='Variables')
  dm_households.operation('lfs_population_total', '/', 'lfs_household-size',
                          out_col='bld_households', unit='household')

  dm_appliance = DM_appliances['demand']

  # Total number of appliances = appliances * nb households
  dm_appliance[:, :, 'bld_appliances_stock', :] = (
      dm_appliance[:, :, 'bld_appliances_stock', :]
      * dm_households[:, :, 'bld_households', np.newaxis])
  dm_appliance.change_unit('bld_appliances_stock', old_unit='unit/household',
                           new_unit='unit', factor=1)

  # Total electricity demand = appliances * appliance elec demand
  dm_appliance.operation('bld_appliances_stock', '*',
                         'bld_appliances_electricity-demand',
                         out_col='bld_appliances_tot-elec-demand', unit='kWh')

  # w(t) = rr(t) * s(t-1)
  # Add missing years
  start_yr = dm_pop.col_labels['Years'][0]
  end_yr = dm_pop.col_labels['Years'][-1]
  years_all = create_years_list(start_yr, end_yr, 1)
  dm_add_missing_variables(dm_appliance, dict_all={'Years': years_all},
                           fill_nans=True)
  # Lag variable
  dm_appliance.lag_variable('bld_appliances_stock', 1, '_tm1')
  dm_appliance.operation('bld_appliances_stock_tm1', '*',
                         'bld_appliances_retirement-rate',
                         out_col='bld_appliances_waste', unit='unit')
  # s(t) = s(t-1) + n(t) - w(t) -> n(t) = s(t) - s(t-1) + w(t)
  arr = (dm_appliance[:, :, 'bld_appliances_stock', :]
         - dm_appliance[:, :, 'bld_appliances_stock_tm1', :]
         + dm_appliance[:, :, 'bld_appliances_waste', :])
  dm_appliance.add(arr, dim='Variables', col_label='bld_appliances_new',
                   unit='unit')
  dm_appliance[:, :, 'bld_appliances_new', :] = np.maximum(0, dm_appliance[:, :,
                                                              'bld_appliances_new',
                                                              :])

  dm_appliance.filter({'Years': dm_pop.col_labels['Years']}, inplace=True)
  dm_appliance.change_unit('bld_appliances_tot-elec-demand', old_unit='kWh', new_unit='TWh', factor=1e-9 )

  DM_appliance_out = {
    'power': dm_appliance.filter(
      {'Variables': ['bld_appliances_tot-elec-demand']}).group_all(
      'Categories1', inplace=False),
    'industry': dm_appliance.filter(
      {'Variables': ['bld_appliances_new', 'bld_appliances_waste']})
  }

  return DM_appliance_out


def bld_costs_workflow(DM_costs, dm_district_heat_supply, dm_new_appliance,
                       dm_floor_renovated):
  # Compute total energy-need for district heating of households
  dm_district_heat_supply = dm_district_heat_supply['households-dh']
  dm_other = DM_costs['other']
  idx_d = dm_district_heat_supply.idx
  idx = dm_other.idx
  arr_tot_en_dh = np.nansum(
    dm_district_heat_supply.array[:, :,
    idx_d['bld_district-heating-space-heating-supply'], ...], axis=(-1, -2, -3)) \
                  + dm_other.array[:, :, idx[
                                           'bld_heat-district_energy-demand_residential_hot-water']]
  dm_other.add(arr_tot_en_dh, dim='Variables',
               col_label='bld_energy-need_district-heating', unit='GWh')
  # Total pipe needs
  dm_other.operation('cp_district-heating_pipe-factor', '*',
                     'bld_energy-need_district-heating',
                     out_col='bld_district-heating_total-pipe-need', unit='km')
  # New pipe need
  dm_other.operation('bld_district-heating_total-pipe-need', '*',
                     'cp_district-heating_new-pipe-factor',
                     out_col='bld_district-heating_new-pipe-need', unit='km')

  # New pipes cost
  dm_other.operation('bld_district-heating_new-pipe-need', '*',
                     'bld_capex_new-pipes',
                     out_col='bld_district-heating_costs', unit='MEUR')

  # Appliance cost
  DM_costs['appliances-capex'].drop(col_label='ac', dim='Categories1')
  dm_new_appliance.append(DM_costs['appliances-capex'], dim='Variables')
  dm_new_appliance.operation('bld_appliance-new', '*', 'bld_capex',
                             out_col='bld_appliances_costs', unit='MEUR')

  # Renovation cost
  dm_cost_renov = DM_costs['renovation-capex']
  idx_c = dm_cost_renov.idx
  idx_r = dm_floor_renovated.idx
  # Cost by building and renovation type based on area renovated
  arr_cost_bld_ren = dm_cost_renov.array[:, :, idx_c['bld_capex'], :, :] \
                     * dm_floor_renovated.array[:, :,
                       idx_r['bld_floor-area-renovated'], :, np.newaxis]
  # Cost of bld renovation by bld type (sum over renovation type)
  arr_cost_bld = np.nansum(arr_cost_bld_ren, axis=-1)
  dm_floor_renovated.add(arr_cost_bld, dim='Variables',
                         col_label='bld_capex_reno', unit='MEUR')
  # Cost of bld renovation by renovation type (sum over building type)
  arr_cost_ren = np.nansum(arr_cost_bld_ren, axis=-2)
  ref_col = dm_cost_renov.col_labels
  col_labels = {'Country': ref_col['Country'], 'Years': ref_col['Years'],
                'Variables': ['bld_capex_reno'],
                'Categories1': ref_col['Categories2']}
  dm_cost_renov_by_depth = DataMatrix(col_labels,
                                      units={'bld_capex_reno': 'MEUR'})
  dm_cost_renov_by_depth.array = arr_cost_ren

  DM_costs_out = {}
  DM_costs_out['TPE'] = {
    'cost-renovation_bld': dm_floor_renovated.filter(
      {'Variables': ['bld_capex_reno']}),
    'cost-renovation_depth': dm_cost_renov_by_depth,
    'cost-appliances': dm_new_appliance.filter(
      {'Variables': ['bld_appliances_costs']})
  }
  DM_costs_out['industry'] = dm_other.filter(
    {'Variables': ['bld_district-heating_new-pipe-need']})

  return DM_costs_out


def bld_light_heat_cool_workflow(DM_light_heat, DM_lfs, DM_clm, baseyear):
  # Extract relevant lfs data
  dm_floor = DM_lfs['floor'].filter({'Variables': ['lfs_floor-space_cool']})
  dm_lfs = DM_lfs['other'].filter(
    {'Variables': ['lfs_heatcool-behaviour_degrees']})
  dm_lfs.append(dm_floor, dim='Variables')
  del dm_floor
  dm_clm = DM_clm['climate-impact-average']
  idx_l = dm_lfs.idx
  idx_c = dm_clm.idx
  # bld_index_lfs_heatcool-behaviour_degrees[#]|bld_index_lfs_floor-space_cool[#]|bld_climate-impact_average[%]
  arr_tmp = dm_clm.array[:, :, idx_c['bld_climate-impact_average']] * \
            dm_lfs.array[:, :, idx_l['lfs_floor-space_cool']] / dm_lfs.array[:,
                                                                np.newaxis,
                                                                idx_l[baseyear],
                                                                idx_l[
                                                                  'lfs_floor-space_cool']] * \
            dm_lfs.array[:, :,
            idx_l['lfs_heatcool-behaviour_degrees']] / dm_lfs.array[:,
                                                       np.newaxis,
                                                       idx_l[baseyear],
                                                       idx_l[
                                                         'lfs_heatcool-behaviour_degrees']]
  # !FIXME this should be the same unit as climate-impact_average
  dm_lfs.add(arr_tmp, dim='Variables', col_label='bld_lifestyles-impact-factor',
             unit='#')

  dm_water_light = DM_light_heat['hot-water']
  dm_ac = DM_light_heat['ac-efficiency']
  # Rescale to reference year
  idx = dm_ac.idx
  arr_tmp = dm_ac.array[:, :, idx['bld_appliance-efficiency'], :] \
            / dm_ac.array[:, idx[baseyear], np.newaxis,
              idx['bld_appliance-efficiency'], :]
  dm_ac.array[:, :, idx['bld_appliance-efficiency'], :] = arr_tmp
  dm_ac = dm_ac.flatten()

  # lighting demand * rescaled efficiency
  dm_water_light.append(dm_ac, dim='Variables')
  del dm_ac
  dm_water_light.operation('bld_lighting-demand_non-residential_electricity',
                           '/', 'bld_appliance-efficiency_ac',
                           out_col='bld_lighting-energy-demand_non-residential_electricity',
                           unit='GWh',
                           div0='interpolate')
  dm_water_light.operation('bld_lighting-demand_residential_electricity', '/',
                           'bld_appliance-efficiency_ac',
                           out_col='bld_lighting-energy-demand_residential_electricity',
                           unit='GWh',
                           div0='interpolate')
  # Remove raw lighting demand
  # dm_water_light.drop(col_label='bld_lighting-demand.*', dim='Variables')

  # Extract cooling and hot water
  # !FIXME this rename should happen in the csv directly
  dm_cooling = DM_light_heat['energy'].filter(
    {'Variables': ['bld_space-cooling-energy-demand_non-residential',
                   'bld_space-cooling-energy-demand_residential'],
     'Categories1': ['electricity', 'gas-bio', 'gas-ff-natural']})
  dm_cooling.deepen(based_on='Variables')

  # Correct space cooling by climate and lifestyle factor
  idx = dm_cooling.idx
  idx_l = dm_lfs.idx
  idx_w = dm_water_light.idx
  arr_tmp = dm_cooling.array[:, :, idx['bld_space-cooling-energy-demand'], :, :] \
            * dm_lfs.array[:, :, idx_l['bld_lifestyles-impact-factor'],
              np.newaxis, np.newaxis] \
            * dm_water_light.array[:, :, idx_w['bld_appliance-efficiency_ac'],
              np.newaxis, np.newaxis]
  dm_cooling.array[:, :, idx['bld_space-cooling-energy-demand'], :, :] = arr_tmp

  # Adjust heatcool-efficiency to reference year
  dm_other = DM_light_heat['energy'].filter(
    {'Variables': ['bld_heatcool-efficiency',
                   'bld_residential-cooking-energy-demand']})
  idx = dm_other.idx
  dm_other.array[:, :, idx['bld_heatcool-efficiency'], :] = dm_other.array[:, :,
                                                            idx[
                                                              'bld_heatcool-efficiency'],
                                                            :] \
                                                            / dm_other.array[:,
                                                              idx[baseyear],
                                                              np.newaxis,
                                                              idx[
                                                                'bld_heatcool-efficiency'],
                                                              :]

  # Adj hot water demand by heat-cool efficiency factor
  dm_water = DM_light_heat['energy'].filter(
    {'Variables': ['bld_hot-water-demand-non-residential',
                   'bld_hot-water-demand-residential']})
  dm_water.rename_col_regex('demand-non-residential', 'demand_non-residential',
                            dim='Variables')
  dm_water.rename_col_regex('demand-residential', 'demand_residential',
                            dim='Variables')
  dm_water.deepen(based_on='Variables')
  idx_c = dm_water.idx
  dm_water.array[:, :, idx_c['bld_hot-water-demand'], :, :] = dm_water.array[:,
                                                              :, idx_c[
                                                                   'bld_hot-water-demand'],
                                                              :,
                                                              :] \
                                                              * dm_other.array[
                                                                :, :, idx[
                                                                        'bld_heatcool-efficiency'],
                                                                :, np.newaxis]

  ### Prepare wf_emissions output
  dm_cooking = dm_other.filter(
    {'Variables': ['bld_residential-cooking-energy-demand'],
     'Categories1': ['gas-bio', 'gas-ff-natural', 'solid-bio',
                     'solid-ff-coal']})
  dm_cool = dm_cooling.filter({'Variables': ['bld_space-cooling-energy-demand'],
                               'Categories1': ['gas-bio', 'gas-ff-natural']})

  # Prepare energy output
  # cooking electricity
  dm_energy = dm_other.filter(
    {'Variables': ['bld_residential-cooking-energy-demand'],
     'Categories1': ['electricity']})
  dm_energy = dm_energy.flatten()
  dm_energy.rename_col('bld_residential-cooking-energy-demand_electricity',
                       'bld_power-demand_residential_cooking',
                       dim='Variables')
  # lighting electricity
  dm_lighting = dm_water_light.filter(
    {'Variables': ['bld_lighting-energy-demand_non-residential_electricity',
                   'bld_lighting-energy-demand_residential_electricity']})
  dm_lighting.rename_col(
    'bld_lighting-energy-demand_non-residential_electricity',
    'bld_power-demand_non-residential_lighting', dim='Variables')
  dm_lighting.rename_col('bld_lighting-energy-demand_residential_electricity',
                         'bld_power-demand_residential_lighting',
                         dim='Variables')
  dm_energy.append(dm_lighting, dim='Variables')
  # cooling electricity
  dm_pow_cooling = dm_cooling.filter({'Categories1': ['electricity']})
  dm_pow_cooling.switch_categories_order()
  dm_pow_cooling.rename_col('bld_space-cooling-energy-demand',
                            'bld_power-demand', dim='Variables')
  dm_pow_cooling.rename_col('electricity', 'space-cooling', dim='Categories2')
  dm_pow_cooling = dm_pow_cooling.flatten()
  dm_pow_cooling = dm_pow_cooling.flatten()
  dm_energy.append(dm_pow_cooling, dim='Variables')
  del dm_lighting, dm_pow_cooling

  # (cooking), (space-heating), appliances, hot-water, (lighting), (space-cooling),
  DM_light_heat_out = {
    'wf_fuel_switch': dm_water,
    'wf_emissions_appliances': {'cooking': dm_cooking, 'cooling': dm_cool},
    'power': dm_energy
  }
  return DM_light_heat_out


def bld_fuel_switch_workflow(DM_fuel_switch, dm_fuel_switch, baseyear):
  lastyear = dm_fuel_switch.col_labels['Years'][-1]
  dm_renewable = DM_fuel_switch['heatcool-shares-renew'].copy()
  # Compute percentage change
  dm_renewable.operation('bld_heatcool-technology-fuel_residential_current',
                         '-',
                         'bld_heatcool-technology-fuel_residential_reference-year',
                         out_col='bld_percentage-change', unit='%')
  # % increase-normalised = % increase / sum( % increase)
  idx = dm_renewable.idx
  arr_sum_increase = np.nansum(
    dm_renewable.array[:, :, idx['bld_percentage-change'], :], axis=-1,
    keepdims=True)
  arr_norm = dm_renewable.array[:, :, idx['bld_percentage-change'],
             :] / arr_sum_increase
  dm_renewable.add(arr_norm,
                   col_label='bld_substitution-per-renewable_residential',
                   dim='Variables', unit='%')
  # Drop unnecessary columns
  dm_renewable = dm_renewable.filter(
    {'Variables': ['bld_substitution-per-renewable_residential']})
  # !FIXME the fact that we are using the ratio for fossil fuel and the difference for renewable doesn't make sense
  dm_fossil = DM_fuel_switch['heatcool-shares-fossil'].copy()
  dm_fossil.operation('bld_heatcool-technology-fuel_residential_current', '/',
                      'bld_heatcool-technology-fuel_residential_reference-year',
                      out_col='bld_percentage-change_residential', unit='%')
  dm_fossil.operation('bld_heatcool-technology-fuel_nonresidential_current',
                      '/',
                      'bld_heatcool-technology-fuel_nonresidential_reference-year',
                      out_col='bld_percentage-change_nonresidential', unit='%')
  idx = dm_fossil.idx
  arr_max_res_nonres = np.maximum(
    dm_fossil.array[:, :, idx['bld_percentage-change_residential'], :],
    dm_fossil.array[:, :, idx['bld_percentage-change_nonresidential'], :])
  # !FIXME I'm also adding the normalisation here because it makes sense, similarly to what done for renewables
  dm_fossil.add(arr_max_res_nonres, col_label='bld_space-heating-fuel-mix',
                dim='Variables', unit='%')
  # Drop unnecessary columns
  dm_fossil = dm_fossil.filter({'Variables': ['bld_space-heating-fuel-mix']})

  # Sum residential and non-residential hot water and compute share by fuel
  arr_hot_water = np.nansum(dm_fuel_switch.array, axis=-1)
  tot_hot_water = np.nansum(arr_hot_water, axis=-1)
  shares_hot_water = arr_hot_water / tot_hot_water[:, :, :, np.newaxis]
  new_col = dm_fuel_switch.col_labels.copy()
  new_col.pop('Categories2')
  new_col['Variables'] = ['bld_hot-water-energy-demand']
  dm_hot_water = DataMatrix(new_col, units={'bld_hot-water-energy-demand': '%'})
  dm_hot_water.array = shares_hot_water

  # !FIXME I don't think this makes sense, it is using baseyear data for 2050
  #  (also somehow this should not apply to electricity)
  # Multiplies hot water energy fossil share by the space heating fuel mix for the baseyear
  # Also here electricity is not included, but probably then it should be excluded also
  # when doing the assessment above for fossil fuel
  dm_space_heating = dm_fossil.copy()
  dm_space_heating.drop(col_label=['electricity'], dim='Categories1')
  dm_hot_water_fossil = dm_hot_water.filter(
    {'Categories1': dm_space_heating.col_labels['Categories1']})
  idx = dm_hot_water_fossil.idx
  idx_f = dm_fossil.idx
  # sum.fuel-type ( hot-water-demand-fossil (t=2050) [%]
  #               - hot-water-demand-fossil (t=2015) [%] * heating-fuel-mix (t=2015) [%])
  # --> bld_hot-water_total-substitution
  arr_tmp = dm_hot_water_fossil.array[:, idx[baseyear],
            idx['bld_hot-water-energy-demand'], :] \
            * dm_space_heating.array[:, idx_f[baseyear],
              idx_f['bld_space-heating-fuel-mix'], :]
  dm_hot_water_fossil_2050 = dm_hot_water_fossil.filter({'Years': [lastyear]})
  dm_hot_water_fossil_2050.array = arr_tmp[:, np.newaxis, np.newaxis, :]
  arr_tmp = np.nansum(dm_hot_water_fossil.array[:, idx[lastyear],
                      idx['bld_hot-water-energy-demand'], :] - \
                      arr_tmp, axis=-1)
  # bld_substitution-per-renewable_residential.by_RES (t=2050) =
  # bld_hot-water_total-substitution (t=2050) * bld_substitution-per-renewable_residential.by_RES (t=2050)
  idx = dm_renewable.idx
  arr_renew = dm_renewable.array[:, idx[lastyear],
              idx['bld_substitution-per-renewable_residential'], :] \
              * arr_tmp[:, np.newaxis]
  dm_hot_water_renew = dm_hot_water.filter(
    {'Categories1': dm_renewable.col_labels['Categories1']})
  # bld_hot-water-fuel-mix-2050 = bld_hot-water-energy-demand +
  #                               bld_substitution-per-renewable_residential * bld_hot-water_total-substitution
  idx = dm_hot_water_renew.idx
  arr_hot_water_renew = dm_hot_water_renew.array[:, idx[lastyear],
                        idx['bld_hot-water-energy-demand'], :] + arr_renew[
                                                                 :, :]
  dm_hot_water_renew_2050 = dm_hot_water_renew.filter({'Years': [lastyear]})
  dm_hot_water_renew_2050.array = arr_hot_water_renew[:, np.newaxis, np.newaxis,
                                  :]
  del arr_hot_water_renew, arr_hot_water, arr_norm, arr_max_res_nonres, arr_renew, arr_sum_increase, arr_tmp, new_col

  dm_hot_water_2050 = dm_hot_water_renew_2050
  dm_hot_water_2050.append(dm_hot_water_fossil_2050, dim='Categories1')
  # Extract electricity
  dm_hot_water_elect_2050 = dm_hot_water.filter(
    {'Categories1': ['electricity'], 'Years': [lastyear]})
  dm_hot_water_2050.append(dm_hot_water_elect_2050, dim='Categories1')
  dm_hot_water_2050.sort(dim='Categories1')
  # Put 2050 data into hot_water datamatrix
  idx = dm_hot_water.idx
  dm_hot_water.array[:, idx[lastyear], :, :] = dm_hot_water_renew_2050.array[:,
                                               0, :, :]
  # Perform linear interpolation between 2015 - 2050
  dm_hot_water.array[:, idx[baseyear] + 1:idx[lastyear], :, :] = np.nan
  dm_hot_water.fill_nans(dim_to_interp='Years')

  # Use these newly computed shares to project the demand_hot_water in GWh for both residential and non-residential
  idx = dm_fuel_switch.idx
  # sum over fuel type, mantain residential & non-residential split
  arr = np.nansum(dm_fuel_switch.array[:, :, idx['bld_hot-water-demand'], :, :],
                  axis=-2)
  idx_h = dm_hot_water.idx
  # for FTS : hot-water-demand.by_fuel_res_type = hot-water-demand.by_fuel_type[%] * hot-water-demand.by_res_non-res[GWh]
  arr_tmp = dm_hot_water.array[:, :, idx_h['bld_hot-water-energy-demand'],
            np.newaxis, :] * arr[:, :, :, np.newaxis]
  dm_fuel_switch.switch_categories_order()
  idx = dm_fuel_switch.idx
  dm_fuel_switch.array[:, idx[baseyear]:, idx['bld_hot-water-demand'], :,
  :] = arr_tmp[:, idx[baseyear]:, :, :]

  # Prepare power module output
  dm_water_pow = dm_fuel_switch.filter({'Categories2': ['electricity']})
  dm_water_pow.rename_col('bld_hot-water-demand', 'bld_power-demand',
                          dim='Variables')
  dm_water_pow.rename_col('electricity', 'hot-water', dim='Categories2')
  dm_water_pow = dm_water_pow.flatten()
  dm_water_pow = dm_water_pow.flatten()

  dm_energy = dm_fuel_switch.group_all('Categories1', inplace=False)

  DM_fuel_switch_out = {
    'wf_emissions_appliances': dm_fuel_switch,
    'power': dm_water_pow,
    'TPE': dm_energy
  }

  return DM_fuel_switch_out


def bld_emissions_appliances_workflow(DM_cooking_cooling, dm_hot_water,
                                      cdm_const):
  # In order to have split by fuel in categories1
  dm_cooking = DM_cooking_cooling['cooking']
  dm_cooking.rename_col('bld_residential-cooking-energy-demand',
                        'bld_cooking-energy-demand_residential',
                        dim='Variables')
  dm_cooking.deepen(based_on='Variables')

  dm_hot_water.switch_categories_order()
  DM_emissions = {
    'cooking': dm_cooking,
    'cooling': DM_cooking_cooling['cooling'],
    'hot_water': dm_hot_water
  }
  # Initialize numpy array to gather residential and non-residential CO2 emissions
  arr_CO2_res = np.zeros((len(dm_hot_water.col_labels['Country']),
                          len(dm_hot_water.col_labels['Years'])))
  arr_CO2_nonres = np.zeros((len(dm_hot_water.col_labels['Country']),
                             len(dm_hot_water.col_labels['Years'])))
  for key in DM_emissions.keys():
    dm_tmp = DM_emissions[key]
    # From GWh to TWh
    for var in dm_tmp.col_labels['Variables']:
      dm_tmp.change_unit(var, factor=1e-3, old_unit='GWh', new_unit='TWh')
    cdm_const_tmp = cdm_const.filter(
      {'Categories1': dm_tmp.col_labels['Categories1']})
    assert cdm_const_tmp.col_labels['Categories1'] == dm_tmp.col_labels[
      'Categories1'], f"Fuels categories do not match"
    # Multiply energy * emissions-factors
    arr_emission = dm_tmp.array[:, :, 0, ...] * cdm_const_tmp.array[np.newaxis,
                                                np.newaxis, 0, :, np.newaxis]
    new_var = var + '_CO2-emissions'
    dm_tmp.add(arr_emission, col_label=new_var, dim='Variables', unit='Mt')
    DM_emissions[key] = dm_tmp
    # Sum emissions for all fuel types by residential and non-residential
    idx = dm_tmp.idx
    arr_CO2_res = arr_CO2_res + np.nansum(
      DM_emissions[key].array[:, :, idx[new_var], :, idx['residential']],
      axis=-1)
    if 'non-residential' in idx.keys():
      arr_CO2_nonres = arr_CO2_nonres + np.nansum(
        DM_emissions[key].array[:, :, idx[new_var], :, idx['non-residential']],
        axis=-1)

  # Gather tot 'appliances' (incl. hot water) emissions by residential and not residential in new datamatrix
  ay_em_appliances = np.concatenate(
    (arr_CO2_nonres[..., np.newaxis, np.newaxis],
     arr_CO2_res[..., np.newaxis, np.newaxis]), axis=-1)
  dm_format = dm_cooking.group_all(dim='Categories2', inplace=False)
  dm_em_appliances = DataMatrix.based_on(ay_em_appliances, format=dm_format,
                                         change={'Variables': [
                                           'bld_CO2-emissions_appliances'],
                                                 'Categories1': [
                                                   'non-residential',
                                                   'residential']},
                                         units={
                                           'bld_CO2-emissions_appliances': 'Mt'})

  DM_emissions_appliances_out = {'emissions': dm_em_appliances}

  return DM_emissions_appliances_out

def compute_tech_fts_based_on_heat_tech(dm_heating, dm_hw_tech, years_ots, years_fts, var_name='bld_hw_tech-mix'):
  dm_heat_tech  = dm_heating.filter({'Variables': ['bld_heating'],
                                     'Categories1': dm_hw_tech.col_labels['Categories1']})
  window_size = 3  # Change window size to control the smoothing effect
  data_smooth = moving_average(dm_heat_tech.array, window_size,
                               axis=dm_heat_tech.dim_labels.index('Years'))
  dm_heat_tech.array[:, 1:-1, ...] = data_smooth

  dm_heat_tech.normalise(dim='Categories1')

  dm_tech = dm_hw_tech.copy()
  window_size = 3  # Change window size to control the smoothing effect
  data_smooth = moving_average(dm_tech.array, window_size,
                               axis=dm_tech.dim_labels.index('Years'))
  dm_tech.array[:, 1:-1, ...] = data_smooth

  dm_heat_tech.append(dm_tech, dim='Variables')

  # Normalise hw and heat tech so that are comparable in 2023
  dm_heat_tech_norm = dm_heat_tech.copy()
  baseyear = years_ots[-2]
  dm_heat_tech_norm[:, :, :, :] = (dm_heat_tech_norm[:, :, :, :]
                                  / dm_heat_tech_norm[:, baseyear, np.newaxis, :, :])

  # Normalised hot-water fts trends matchning heating trends
  idx = dm_heat_tech_norm.idx
  dm_heat_tech_norm[:, idx[years_fts[0]]:, var_name, :] \
    = dm_heat_tech_norm[:, idx[years_fts[0]]:, 'bld_heating', :]

  # Return to actual values
  dm_heat_tech[:, idx[years_fts[0]]:, var_name, :] \
    = (dm_heat_tech_norm[:, idx[years_fts[0]]:, var_name, :]
       * dm_heat_tech[:, baseyear, np.newaxis, var_name, :])

  assert dm_hw_tech.col_labels['Categories1'] == dm_heat_tech.col_labels['Categories1']
  dm_hw_tech[:, idx[years_fts[0]]:, var_name, :] = dm_heat_tech[:, idx[years_fts[0]]:, var_name, :]

  dm_hw_tech.normalise('Categories1', inplace=True)
  dm_hw_tech.filter({'Variables': [var_name]}, inplace=True)

  return dm_hw_tech


def compute_eff_fts_based_on_heat_eff(dm_heating, dm_hw_eff,  years_ots, years_fts, var_name='bld_hot-water_efficiency'):
  # Hot water efficiency fts scenario is created based on the heating efficiency fts

  # Compute heat efficiency
  dm_heating.operation('bld_heating', '/', 'bld_energy-demand_heating',
                       out_col='bld_heating-efficiency', unit='%')

  matching_cat = set(dm_hw_eff.col_labels['Categories1']).intersection(set(dm_heating.col_labels['Categories1']))

  dm_heat_eff = dm_heating.filter({'Variables': ['bld_heating-efficiency'],
                                   'Categories1': list(matching_cat)})
  dm_hw_eff.filter({'Categories1': list(matching_cat)}, inplace=True)

  dm_heat_eff.append(dm_hw_eff, dim='Variables')

  # Normalise hw and heat efficiency so that they are comparable in 2023
  dm_heat_eff_norm = dm_heat_eff.copy()
  baseyear = years_ots[-1]
  dm_heat_eff_norm[:, :, :, :] = (dm_heat_eff_norm[:, :, :, :]
                                  / dm_heat_eff_norm[:, baseyear, np.newaxis, :, :])

  # Normalised hot-water fts trends matching heating trends
  idx = dm_heat_eff_norm.idx
  dm_heat_eff_norm[:, idx[years_fts[0]]:, var_name, :] \
    = dm_heat_eff_norm[:, idx[years_fts[0]]:, 'bld_heating-efficiency', :]

  dm_heat_eff.filter({'Variables': [var_name]}, inplace=True)

  # Return to actual values
  dm_heat_eff[:, idx[years_fts[0]]:, var_name, :] \
    = (dm_heat_eff_norm[:, idx[years_fts[0]]:, var_name, :]
       * dm_heat_eff[:, baseyear, np.newaxis, var_name, :])

  # Efficiency can not be above 0.98 (except for electricity and heat-pump)
  standard_cat = [cat for cat in dm_heat_eff.col_labels['Categories1'] if
                  ('heat-pump' not in cat) and ('electricity' not in cat)]
  dm_heat_eff_std = dm_heat_eff.filter({'Categories1': standard_cat})
  dm_heat_eff.drop(col_label=standard_cat, dim='Categories1')
  dm_heat_eff_std.array = np.minimum(dm_heat_eff_std.array, 0.98)
  dm_heat_eff.append(dm_heat_eff_std, dim='Categories1')

  dm_heat_eff.sort('Categories1')

  return dm_heat_eff

def bld_hotwater_workflow(DM_hotwater, dm_heating, dm_lfs, years_ots, years_fts):

  dm_hw_eff = DM_hotwater['efficiency'].copy()
  dm_hw_eff = compute_eff_fts_based_on_heat_eff(dm_heating, dm_hw_eff,  years_ots, years_fts)

  dm_hw_tech = DM_hotwater['tech-mix'].copy()
  dm_hw_tech = compute_tech_fts_based_on_heat_tech(dm_heating, dm_hw_tech, years_ots, years_fts)

  dm_hw_demand = DM_hotwater['demand'].copy()

  arr = (dm_hw_tech[:, :, 'bld_hw_tech-mix', :]
         * dm_hw_demand[:, :, 'bld_hw_demand', np.newaxis]
         * dm_lfs[:, :, 'lfs_population_total', np.newaxis])
  dm_hw_tech.add(arr, dim='Variables', col_label='bld_hw_useful-energy', unit='MWh')
  dm_hw_tech.change_unit('bld_hw_useful-energy', old_unit='MWh', new_unit='TWh', factor=1e-6)

  dm_hw_tech.append(dm_hw_eff, dim='Variables')
  dm_hw_tech.operation('bld_hw_useful-energy', '/', 'bld_hot-water_efficiency', out_col='bld_hot-water_energy-demand', unit='TWh')

  DM_hotwater_out = {'power': dm_hw_tech}

  return DM_hotwater_out


def bld_services_workflow(DM_services, dm_heating, years_ots, years_fts):

  dm_eff = DM_services['services_efficiencies'].copy()
  dm_eff = compute_eff_fts_based_on_heat_eff(dm_heating, dm_eff,  years_ots, years_fts, var_name='bld_services_efficiency')
  dm_eff.fill_nans('Years')

  # Forecast how water tech-mix based on household heating
  dm_srv_hw_tech = DM_services['services_tech-mix'].filter({'Categories1': ['hot-water']})
  dm_srv_hw_tech.group_all('Categories1', inplace=True)
  matching_cat = set(dm_srv_hw_tech.col_labels['Categories1']).intersection(set(dm_heating.col_labels['Categories1']))
  dm_srv_hw_tech.filter({'Categories1': list(matching_cat)}, inplace=True)
  dm_srv_hw_tech = compute_tech_fts_based_on_heat_tech(dm_heating, dm_srv_hw_tech, years_ots, years_fts, var_name='bld_services_tech-mix')
  dm_srv_hw_tech.rename_col('bld_services_tech-mix', 'bld_services_tech-mix_hot-water', dim='Variables')
  dm_srv_hw_tech.deepen(based_on='Variables')
  dm_srv_hw_tech.switch_categories_order()

  # Forecast space heating services tech-mix based on household heating
  dm_srv_sh_tech = DM_services['services_tech-mix'].filter({'Categories1': ['space-heating']})
  dm_srv_sh_tech.group_all('Categories1', inplace=True)
  matching_cat = set(dm_srv_sh_tech.col_labels['Categories1']).intersection(set(dm_heating.col_labels['Categories1']))
  dm_srv_sh_tech.filter({'Categories1': list(matching_cat)}, inplace=True)
  dm_srv_sh_tech = compute_tech_fts_based_on_heat_tech(dm_heating, dm_srv_sh_tech, years_ots, years_fts, var_name='bld_services_tech-mix')
  dm_srv_sh_tech.rename_col('bld_services_tech-mix', 'bld_services_tech-mix_space-heating', dim='Variables')
  dm_srv_sh_tech.deepen(based_on='Variables')
  dm_srv_sh_tech.switch_categories_order()

  dm_elec = DM_services['services_tech-mix'].filter({'Categories1': ['elec', 'lighting']})
  dm_elec.fill_nans('Years')

  dm_tech_mix = dm_elec.copy()
  dm_tech_mix.filter({'Categories2': dm_srv_sh_tech.col_labels['Categories2']}, inplace=True)
  dm_tech_mix.append(dm_srv_sh_tech, dim='Categories1')
  dm_tech_mix.append(dm_srv_hw_tech, dim='Categories1')
  dm_tech_mix.sort('Categories1')

  dm_demand = DM_services['services_demand']
  dm_demand.sort('Categories1')

  arr_useful_energy = (dm_demand[:, :, 'bld_services_useful-energy', :, np.newaxis]
                       * dm_tech_mix[:, :, 'bld_services_tech-mix', :, :])
  arr_energy_consumption = (arr_useful_energy
                            / dm_eff[:, :, 'bld_services_efficiency', np.newaxis, :])

  dm_tech_mix.add(arr_useful_energy, dim='Variables', col_label='bld_services_useful-energy', unit='TWh')
  dm_tech_mix.add(arr_energy_consumption, dim='Variables', col_label='bld_services_energy-consumption', unit='TWh')


  DM_services_out = {
    'TPE' : dm_tech_mix.filter({'Variables': ['bld_services_energy-consumption']}),
    'energy': dm_tech_mix}

  return DM_services_out
