
# packages
from model.common.constant_data_matrix_class import ConstantDataMatrix
import pickle
import os
import numpy as np
import warnings
warnings.simplefilter("ignore")

# file
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/industry/eu/python/industry_const_material-switch-ratio.py"

# directories
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# make constant
names = ['tec_material-switch-ratios_cement-to-timber', 'tec_material-switch-ratios_chem-to-natfibers', 
        'tec_material-switch-ratios_chem-to-paper', 'tec_material-switch-ratios_steel-to-aluminium', 
        'tec_material-switch-ratios_steel-to-chem', 'tec_material-switch-ratios_steel-to-timber']
names = [i + "[kg/kg]" for i in names]
values = [3.87, 1.  , 1.  , 0.55, 0.4 , 1.04]
units = np.repeat("kg/kg", len(values)).tolist()
const = {
    'name': names,
    'value': values,
    'idx': dict(zip(names, range(len(values)))),
    'units': dict(zip(names, units))
}

# make cdm
cdm = ConstantDataMatrix.create_from_constant(const, 0)

# rename
cdm.rename_col_regex("tec_", "", "Variables")

# save
f = os.path.join(current_file_directory, '../data/datamatrix/const_material-switch-ratios.pickle')
with open(f, 'wb') as handle:
    pickle.dump(cdm, handle, protocol=pickle.HIGHEST_PROTOCOL)





