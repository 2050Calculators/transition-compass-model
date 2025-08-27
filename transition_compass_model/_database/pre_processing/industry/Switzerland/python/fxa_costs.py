

import os
import pickle

# directories
current_file_directory = os.getcwd()

# get cost data
f = os.path.join(current_file_directory, '../../eu/data/datamatrix/fxa_costs.pickle')
with open(f, 'rb') as handle:
    DM_costs = pickle.load(handle)

# get switzerland
DM_costs["costs"] = DM_costs["costs"].filter({"Country" : ["Switzerland"]})
DM_costs["costs-cc"] = DM_costs["costs-cc"].filter({"Country" : ["Switzerland"]})

# save
f = os.path.join(current_file_directory, '../data/datamatrix/fxa_costs.pickle')
with open(f, 'wb') as handle:
    pickle.dump(DM_costs, handle, protocol=pickle.HIGHEST_PROTOCOL)


