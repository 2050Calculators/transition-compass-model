################################
####    EMISSION FACTORS   #####
################################
from model.common.constant_data_matrix_class import ConstantDataMatrix
import numpy as np

def create_emissions_factors_cdm(emis, mapping_cat):
  col_labels = dict()
  col_labels['Variables'] = ['cp_tra_emission-factor']
  col_labels['Categories1'] = ['CH4', 'CO2', 'N2O']
  col_labels['Categories2'] = list(mapping_cat.keys())
  arr = np.ones((len(col_labels['Variables']), len(col_labels['Categories1']),
                 len(col_labels['Categories2'])))
  for g, ghg in enumerate(col_labels['Categories1']):
    for f, fuel in enumerate(col_labels['Categories2']):
      simple_fuel_name = mapping_cat[fuel]
      arr[0, g, f] = emis[ghg][simple_fuel_name]
  cdm_emissions = ConstantDataMatrix(col_labels, units={
    'cp_tra_emission-factor': 'g/MJ'})  # kg/TJ -> Mt/TWh
  # (Mt/TWh -> 10^6 tonnes / 10^9 kWh -> 10^9 kg / 10^9 kWh -> kg / kWh -> kg / (3.6e-6 TJ) -> 1/(3.6e-6) kg/TJ)
  # Turn kg/TJ to kg/(1e6 x MJ) = 1e3 g / 1e6 MJ -> kg/TJ = 1e-3 g/MJ
  cdm_emissions.array = arr * 1e-3
  return cdm_emissions



def run():
# SECTION Emission factors - constants
  #region Literature on emissions factors
  # Source cited in EUCalc doc
  # CO2 emissions from table 3.2.1 "Road transport default CO2 emission factors and uncertainty ranges"
  # CH4 and N2O emissions from table 3.2.2 "Road transport N2O and CH4 default emission factors and uncertainty ranges"
  # [IPCC] International Panel on Climate Change (2006). IPCC Guidelines for National Greenhouse Gas Inventories
  # - Volume 2: Energy - Mobile Combustion
  # For marinefueloil we consider the emission of the residual fuel oil (aka heavy fuel oil) in table 3.5.2 (CO2) 3.5.3 (CH4, N2O)
  # For kerosene (aviation) we use Table 3.6.4 (CO2) and 3.6.5 (CH4, N2O)
  # https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/2_Volume2/V2_3_Ch3_Mobile_Combustion.pdf
  # Similar values can be found in the National Inventory Document of Switzerland 2024 (Table 3-13).
  #endregion
  # Emission factors in kg/TJ
  emis = {'CO2': {'diesel': 74100, 'gas': 56100, 'gasoline': 69300, 'hydrogen': 0, 'kerosene': 71500, 'marinefueloil': 77400},
          'CH4': {'diesel': 3.9, 'gas': 92, 'gasoline': 3.9, 'hydrogen': 0, 'kerosene': 0.5, 'marinefueloil': 7},
          'N2O': {'diesel': 3.9, 'gas': 3, 'gasoline': 3.9, 'hydrogen': 0, 'kerosene': 2, 'marinefueloil': 2}}
  mapping_cat = {'ICE-diesel': 'diesel', 'PHEV-diesel': 'diesel', 'ICE-gasoline': 'gasoline',
                 'PHEV-gasoline': 'gasoline', 'ICE-gas': 'gas', 'kerosene': 'kerosene', 'marinefueloil': 'marinefueloil'}

  cdm_emissions_factors = create_emissions_factors_cdm(emis, mapping_cat)

  return cdm_emissions_factors

if __name__ == "__main__":
  run()
