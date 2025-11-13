########################
####    LIFETIME    ####
########################
from model.common.data_matrix_class import DataMatrix
from src.api.routes import country_list


def run(mode_cat, tech_cat, years_ots, years_fts, country_years):
  # region Reflection on the renewal-rate and lifetime
  # When the stock is constant, the renewal-rate is 1/lifetime. This gives a logical sense to the renewal-rate.
  # For the ots, the renewal-rate is simply computed as waste/stock, and its value is often far from what we believe
  # is a realistic 1/lifetime value, for an average lifetime of 1/13 years the renewal-rate should be 7.6%, and instead we
  # find values oscillating between 2% to 9%. This can be due to many things, if for example we are talking about a new
  # technology like BEV, FCEV etc the waste is still relatively small because the car that were put on the market have
  # not yet arrived to their end-of-life. For thecnologies that are going out of production the renewal rate can be high
  # (a lot of waste from previous years and the stock going to zero). To complicate matters though there is the fact that
  # the boundary are not closed, cars can be moved across borders before reaching their end of life. But we could assume
  # that this is random noise on top of the data that should cancel out. But still in a scenario where we are trying to
  # reproduce drastical changes to the vehicle stock, it does not make sense to use a constant renewal-rate for forecasting.
  # For forecasting we could assume that the waste is equal to the new vehicles that were put on the market 13 years ago.
  # This could create discontinuities between ots and fts... But we can try. So let's assume that for fts the renewal-rate
  # is 1/lifetime and instead of computing the waste / stock vehicle fleet in the classic way we use the renewal-rate to go
  # in time and determine the waste.

  # In order to chose the average lifetime, we look at the trends of LDV ICE-diesel, where the trends show an average
  # lifetime of 13.5 years. For EV the current lifetime seems to be rather 5.5 years.
  # We could start with 5.5 years and then increase the lifetime to reach 13.5 years after 10 years. For 2W we use 8 years.
  # endregion

  # ots are not used
  dm_lifetime = DataMatrix(col_labels={'Country': country_list,
                                       'Years': years_ots+years_fts,
                                       'Variables': ['tra_passenger_lifetime'],
                                       'Categories1': mode_cat,
                                       'Categories2': tech_cat},
                           units= {'tra_passenger_lifetime': 'years'})
  dm_lifetime.sort('Categories1')
  dm_lifetime.sort('Categories2')

  # LDV: ICE and PHEV vehicles have lifetime of 13.5 years
  for cat in dm_lifetime.col_labels['Categories2']:
      if ("ICE" in cat) or ('PHEV' in cat):
          dm_lifetime[:, :, 'tra_passenger_lifetime', 'LDV', cat] = 14
  dm_lifetime[:, :, 'tra_passenger_lifetime', 'LDV', 'ICE-gasoline'] = 15
  # LDV: New technology like BEV and PHEV have lifetimes initially of 5.5, and the 13.5
  for cat in dm_lifetime.col_labels['Categories2']:
      if ("BEV" in cat) or ('FCEV' in cat):
          dm_lifetime[:, years_fts[0], 'tra_passenger_lifetime', 'LDV', cat] = 5.5
          dm_lifetime[:, 2035, 'tra_passenger_lifetime', 'LDV', cat] = 14
          dm_lifetime[:, years_fts[-1], 'tra_passenger_lifetime', 'LDV', cat] = 14
  # 2W: 8 years
  dm_lifetime[:, :, 'tra_passenger_lifetime', '2W', :] = 8
  # We assume rail lifetime is 30 years, metrotram lifetime is 20 years and bus lifetime is 10 years
  dm_lifetime[:, :, 'tra_passenger_lifetime', 'rail', 'CEV'] = 30
  dm_lifetime[:, :, 'tra_passenger_lifetime', 'metrotram', 'mt'] = 20
  dm_lifetime[:, :, 'tra_passenger_lifetime', 'bus', 'CEV'] = 10
  dm_lifetime[:, :, 'tra_passenger_lifetime', 'bus', 'ICE-diesel'] = 10

  dm_lifetime.fill_nans('Years')

  return dm_lifetime
