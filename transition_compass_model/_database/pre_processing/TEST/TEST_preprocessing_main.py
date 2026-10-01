# from processors.TEST_levers import run as levers_run
# from processors.TEST_ots_pickle import run as ots_pickle_run
# from scenarios.TEST_fts_BAU_pickle import run as fts_bau_pickle_run
import pickle

from transition_compass_model.model.common.auxiliary_functions import create_years_list

# years
years_ots = create_years_list(1990, 2023, 1)
years_fts = create_years_list(2025, 2050, 5)

# EXERCISE
# dm building
# floor-intensity, open in preprocessing test.py, change the 4 ?? (i think, multiply, etc sth like that
# --> change each level individually, TO DO, e.g. muliptly level 2 by 2, level 3 by 3, level 4 by 4)
# create test pickle wit my pickle dump
# open it in test
# let the datamatrixes from test with buildings.pickle interact (e.g. add them together) to test it
# send the combined datamatrix to energy
# use dm.array (np.array)
# dm.index

# to find out the path for buildings.pickle
# import os

# for root, dirs, files in os.walk("."):
#     if "buildings.pickle" in files:
#         print(os.path.join(root, "buildings.pickle"))


data_file = "transition_compass_model/_database/data/datamatrix/buildings.pickle"

with open(data_file, "rb") as handle:
    DM_buildings = pickle.load(handle)

print(DM_buildings["ots"]["floor-intensity"])


# creating a copy to experiment with the wanted DataMatrix
dm_floor_intensity_ots = DM_buildings["ots"]["floor-intensity"]
dm_floor_intensity_fts = DM_buildings["fts"]["floor-intensity"]

# test to see what type fts is
# print(type(dm_floor_intensity_fts))
# print(dm_floor_intensity_fts.keys())


# Multiply all OTS values by 4
dm_floor_intensity_ots.array = dm_floor_intensity_ots.array * 4

# Multiply all FTS values by 4 (also FTS has the 4 levels 1-4 --> 4 x multiplicating, for each scenario)
for key in dm_floor_intensity_fts:
    dm_floor_intensity_fts[key].array = dm_floor_intensity_fts[key].array * 4

print(type(dm_floor_intensity_ots))
print(type(dm_floor_intensity_ots.array))

print(type(dm_floor_intensity_fts))
print(type(dm_floor_intensity_fts[1].array))  # [key]: from 1 to 4

# --------------------------------------------------------------------------
# Creating a TEST.pickle
# --------------------------------------------------------------------------

from transition_compass_model.model.common.auxiliary_functions import my_pickle_dump

DM_TEST = {
    "TEST": {
        "ots": {"floor-intensity": dm_floor_intensity_ots},
        "fts": {"floor-intensity": dm_floor_intensity_fts},
    }
}

file = "transition_compass_model/_database/data/datamatrix/TEST.pickle"

my_pickle_dump(DM_TEST, file)

# testing if it worked:
# with open(file, "rb") as handle:
#     TEST = pickle.load(handle)

# print(TEST.keys())

# print(TEST["ots"])
# print(TEST["fts"].keys())

# ----------------------------------------------------------------------------
