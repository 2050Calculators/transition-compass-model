########################################
####    RENEWAL-RATE  &  WASTE     #####
########################################

import numpy as np


def compute_renewal_rate_and_adjust(dm, var_names, max_rr):
  """
  It computes the renewal rate and it adjusts the new-vehicles before 2005, where the fleet split was not known
  """
  # Extract variable names
  s_col = var_names['stock']
  new_col = var_names['new']
  waste_col = var_names['waste']
  rr_col = var_names['renewal-rate']

  stock_unit = dm.units[s_col]

  # COMPUTE RENEWAL-RATE
  # Lag stock
  dm.lag_variable(pattern=s_col, shift=1, subfix='_tm1')
  # waste(t) = fleet(t-1) - fleet(t) + new-veh(t)
  dm.operation(s_col + '_tm1', '-', s_col, out_col='tra_delta_stock',
               unit=stock_unit)
  dm.operation('tra_delta_stock', '+', new_col, out_col=waste_col,
               unit=stock_unit)
  # rr(t-1) = waste(t) / fleet(t-1)
  dm.operation(waste_col, '/', s_col + '_tm1', out_col='tmp', unit='%')
  dm.lag_variable(pattern='tmp', shift=-1, subfix='_rr')
  dm.rename_col('tmp_rr', rr_col, dim='Variables')
  dm.filter({'Variables': [s_col, s_col + '_tm1', rr_col]}, inplace=True)

  # FIX RENEWAL-RATE
  # move variables col to end
  dm_rr = dm.filter({'Variables': [rr_col]}, inplace=False)
  mask = (dm_rr.array < 0) | (dm_rr.array > max_rr)
  dm_rr.array[mask] = np.nan
  dm_rr.fill_nans('Years')
  dm.drop(dim='Variables', col_label=rr_col)
  dm.append(dm_rr, dim='Variables')

  # RECOMPUTE NEW FLEET
  dm.lag_variable(pattern=rr_col, shift=1, subfix='_tm1')
  # waste(t) = rr(t-1) * fleet(t-1)
  dm.operation(rr_col + '_tm1', '*', s_col + '_tm1', out_col=waste_col,
               unit=stock_unit)
  # new(t) = fleet(t) - fleet(t-1) + waste(t)
  dm.operation(s_col, '-', s_col + '_tm1', out_col='tra_delta_stock',
               unit=stock_unit)
  dm.operation('tra_delta_stock', '+', waste_col, out_col=new_col,
               unit=stock_unit)
  dm.filter({'Variables': [s_col, new_col, waste_col, rr_col]}, inplace=True)

  # FIX NEW FLEET
  dm_new = dm.filter({'Variables': [new_col]}, inplace=False)
  mask = (dm_new.array < 0)
  dm_new.array[mask] = np.nan
  dm_new.fill_nans('Years')
  dm.drop(dim='Variables', col_label=new_col)
  dm.append(dm_new, dim='Variables')

  # RECOMPUTE STOCK
  idx = dm.idx
  for t in dm.col_labels['Years'][1:]:
    s_tm1 = dm.array[:, idx[t - 1], idx[s_col], ...]
    new_t = dm.array[:, idx[t], idx[new_col], ...]
    waste_t = dm.array[:, idx[t], idx[waste_col], ...]
    s_t = s_tm1 + new_t - waste_t
    dm.array[:, idx[t], idx[s_col], ...] = s_t

  return

def compute_new_public_fleet_ots(dm, var_names):
  # Extract variable names
  s_col = var_names['stock']
  new_col = var_names['new']
  waste_col = var_names['waste']
  rr_col = var_names['renewal-rate']

  stock_unit = dm.units[s_col]

  # COMPUTE RENEWAL-RATE
  # Lag stock
  dm.lag_variable(pattern=s_col, shift=1, subfix='_tm1')
  dm.lag_variable(pattern=rr_col, shift=1, subfix='_tm1')
  # waste(t) = rr(t-1) * fleet(t-1)
  dm.operation(rr_col + '_tm1', '*', s_col + '_tm1', out_col=waste_col,
               unit=stock_unit)
  # new(t) = fleet(t) - fleet(t-1) + waste(t)
  dm.operation(s_col, '-', s_col + '_tm1', out_col='tra_delta_stock',
               unit=stock_unit)
  dm.operation('tra_delta_stock', '+', waste_col, out_col=new_col,
               unit=stock_unit)
  dm.filter({'Variables': [s_col, new_col, waste_col, rr_col]}, inplace=True)

  # FIX NEW FLEET
  dm_new = dm.filter({'Variables': [new_col]}, inplace=False)
  mask = (dm_new.array < 0)
  dm_new.array[mask] = np.nan
  dm_new.fill_nans('Years')
  dm.drop(dim='Variables', col_label=new_col)
  dm.append(dm_new, dim='Variables')

  # RECOMPUTE STOCK
  idx = dm.idx
  for t in dm.col_labels['Years'][1:]:
    s_tm1 = dm.array[:, idx[t - 1], idx[s_col], ...]
    new_t = dm.array[:, idx[t], idx[new_col], ...]
    waste_t = dm.array[:, idx[t], idx[waste_col], ...]
    s_t = s_tm1 + new_t - waste_t
    dm.array[:, idx[t], idx[s_col], ...] = s_t

  return


def run(dm_private_fleet, dm_public_fleet):
  var_names = {'stock': 'tra_passenger_vehicle-fleet',
               'new': 'tra_passenger_new-vehicles',
               'waste': 'tra_passenger_vehicle-waste',
               'renewal-rate': 'tra_passenger_renewal-rate'}
  compute_renewal_rate_and_adjust(dm_private_fleet, var_names, max_rr=0.1)
  dm_pass_fleet = dm_private_fleet.filter({'Variables': [var_names['stock']]})
  dm_new_private_fleet = dm_private_fleet.filter(
    {'Variables': [var_names['new']]})
  dm_renewal_rate = dm_private_fleet.filter(
    {'Variables': [var_names['renewal-rate']]})
  dm_waste_private = dm_private_fleet.filter(
    {'Variables': [var_names['waste']]})
  dm_private_fleet = dm_private_fleet.filter(
    {'Variables': [var_names['stock']]})

  # SECTION Renewal-rate % - New vehicles - vehicles Waste (bus, rail, metrotram) ots
  # Use renewal-rate (1/lifetime) to compute the new public fleet
  missing_cat = set(dm_public_fleet.col_labels['Categories2']) - set(
    dm_renewal_rate.col_labels['Categories2'])
  dm_renewal_rate.add(np.nan, dim='Categories2', col_label=missing_cat,
                      dummy=True)
  dm_renewal_rate.add(np.nan, dim='Categories1',
                      col_label=dm_public_fleet.col_labels['Categories1'],
                      dummy=True)
  idx = dm_renewal_rate.idx
  idx_cat2_public = [idx[cat] for cat in
                     dm_public_fleet.col_labels['Categories2']]
  dm_renewal_rate.array[:, :, idx['tra_passenger_renewal-rate'], idx['rail'],
  idx_cat2_public] = 1 / 30
  dm_renewal_rate.array[:, :, idx['tra_passenger_renewal-rate'],
  idx['metrotram'], idx['mt']] = 1 / 20
  dm_renewal_rate.array[:, :, idx['tra_passenger_renewal-rate'], idx['bus'],
  idx_cat2_public] = 1 / 10

  dm_public_fleet.append(dm_renewal_rate.filter(
    {'Categories1': dm_public_fleet.col_labels['Categories1'],
     'Categories2': dm_public_fleet.col_labels['Categories2']}),
                         dim='Variables')
  var_names = {'renewal-rate': 'tra_passenger_renewal-rate',
               'stock': 'tra_passenger_vehicle-fleet',
               'new': 'tra_passenger_new-vehicles',
               'waste': 'tra_passenger_vehicle-waste'}
  compute_new_public_fleet_ots(dm_public_fleet, var_names)
  dm_new_public_fleet = dm_public_fleet.filter(
    {'Variables': [var_names['new']]})
  dm_waste_public = dm_public_fleet.filter({'Variables': [var_names['waste']]})
  dm_public_fleet.filter({'Variables': [var_names['stock']]}, inplace=True)

  # Join private and public fleet new and waste
  cat_private_only = list(set(dm_new_private_fleet.col_labels['Categories2'])
                          - set(dm_new_public_fleet.col_labels['Categories2']))
  cat_public_only = list(set(dm_new_public_fleet.col_labels['Categories2'])
                         - set(dm_new_private_fleet.col_labels['Categories2']))
  dm_new_fleet = dm_new_private_fleet.copy()
  dm_new_fleet.add(np.nan, dummy=True, dim='Categories2',
                   col_label=cat_public_only)
  dm_new_public_fleet.add(np.nan, dummy=True, dim='Categories2',
                          col_label=cat_private_only)
  dm_new_fleet.append(dm_new_public_fleet, dim='Categories1')

  dm_waste_fleet = dm_waste_private.copy()
  dm_waste_fleet.add(np.nan, dummy=True, dim='Categories2',
                     col_label=cat_public_only)
  dm_waste_public.add(np.nan, dummy=True, dim='Categories2',
                      col_label=cat_private_only)
  dm_waste_fleet.append(dm_waste_public, dim='Categories1')

  DM = {'passenger_private-fleet': dm_private_fleet,
        'passenger_public-fleet': dm_public_fleet,
        'passenger_renewal-rate': dm_renewal_rate,
        'passenger_new-vehicles': dm_new_fleet,
        'passenger_waste-fleet': dm_waste_fleet}

  return DM
