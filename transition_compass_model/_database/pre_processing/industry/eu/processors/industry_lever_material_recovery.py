# packages
import pickle
import warnings

warnings.simplefilter("ignore")
import numpy as np
import plotly.io as pio

pio.renderers.default = "browser"
import os

# from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat
from transition_compass_model._database.pre_processing.industry.eu.get_data_functions.data_material_recovery import (
    get_material_recovery_data,
)
from transition_compass_model.model.common.auxiliary_functions import create_years_list
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def make_material_recovery_dm(current_file_directory, lever_file, years_ots):
    df_agg = get_material_recovery_data()

    # make function to make dm
    def make_dm(df, years_ots):
        # create dm
        countries = [
            "Austria",
            "Belgium",
            "Bulgaria",
            "Croatia",
            "Cyprus",
            "Czech Republic",
            "Denmark",
            "EU27",
            "Estonia",
            "Finland",
            "France",
            "Germany",
            "Greece",
            "Hungary",
            "Ireland",
            "Italy",
            "Latvia",
            "Lithuania",
            "Luxembourg",
            "Malta",
            "Netherlands",
            "Poland",
            "Portugal",
            "Romania",
            "Slovakia",
            "Slovenia",
            "Spain",
            "Sweden",
            "United Kingdom",
        ]

        variabs = list(df["variable"])
        units = list(np.repeat("%", len(variabs)))
        units_dict = dict()
        for i in range(0, len(variabs)):
            units_dict[variabs[i]] = units[i]
        index_dict = dict()
        for i in range(0, len(countries)):
            index_dict[countries[i]] = i
        for i in range(0, len(years_ots)):
            index_dict[years_ots[i]] = i
        for i in range(0, len(variabs)):
            index_dict[variabs[i]] = i

        dm = DataMatrix(empty=True)
        dm.col_labels = {"Country": countries, "Years": years_ots, "Variables": variabs}
        dm.units = units_dict
        dm.idx = index_dict
        dm.array = np.zeros((len(countries), len(years_ots), len(variabs)))
        idx = dm.idx
        for i in variabs:
            dm.array[:, :, idx[i]] = df.loc[df["variable"] == i, "value"]
        # df_check = dm.write_df()

        # # make nan for other than EU27 for fts
        # countries_oth = np.array(countries)[[i not in "EU27" for i in countries]].tolist()
        # idx = dm.idx
        # years = list(range(2025, 2050 + 1, 5))
        # for c in countries_oth:
        #     for y in years:
        #         for v in variabs:
        #             dm.array[idx[c], idx[y], idx[v]] = np.nan
        # # df_check = dm.write_df()

        # rename
        dm.deepen()
        variabs = dm.col_labels["Variables"]
        for i in variabs:
            dm.rename_col(i, "waste-material-recovery_" + i, "Variables")
        dm.deepen(based_on="Variables")
        dm.switch_categories_order("Categories1", "Categories2")

        # check
        # dm.filter({"Country" : ["EU27"]}).flatten().flatten().datamatrix_plot()

        # # drop ammonia
        # dm.drop("Categories2", ["ammonia"])

        # dm units
        dm.units["waste-material-recovery"] = "%"

        return dm

    ####################
    ##### VEHICLES #####
    ####################

    # select only vehicles
    df_elv = df_agg.loc[df_agg["variable"].isin(["vehicles", "battery-lion"]), :]

    # fix variables
    df_elv["variable"] = [
        v + "_" + m for v, m in zip(df_elv["variable"], df_elv["material"])
    ]
    df_elv.drop(["material"], axis=1, inplace=True)

    # as we intend batteries as battery packs, I assign the same recovery rates of the car recycling techs
    # to steel and aluminium that can be in the pack
    df_elv.loc[df_elv["variable"] == "battery-lion_aluminium", "value"] = df_elv.loc[
        df_elv["variable"] == "vehicles_aluminium", "value"
    ].values[0]
    df_elv.loc[df_elv["variable"] == "battery-lion_steel", "value"] = df_elv.loc[
        df_elv["variable"] == "vehicles_steel", "value"
    ].values[0]

    # now assign 0 to nan
    df_elv.loc[df_elv["value"].isnull(), "value"] = 0

    # make dm
    dm_veh = make_dm(df_elv, years_ots)
    # df_check = dm_veh.write_df()

    #################################
    ##### TRAINS, SHIPS, PLANES #####
    #################################

    # select only trains and metrotram
    df_temp = df_agg.loc[
        df_agg["variable"].isin(
            [
                "train",
                # "mt"
            ]
        ),
        :,
    ]

    # # for mt, put steel same of train
    # df_temp.loc[(df_temp["variable"] == "mt") & (df_temp["material"] == "steel"),"value"] = 0.2

    # rename
    # df_temp.loc[df_temp["variable"] == "mt","variable"] = "metrotram"
    df_temp.loc[df_temp["variable"] == "train", "variable"] = "trains"

    # fix variables
    df_temp["variable"] = [
        v + "_" + m for v, m in zip(df_temp["variable"], df_temp["material"])
    ]
    df_temp.drop(["material"], axis=1, inplace=True)

    # now assign 0 to nan
    df_temp.loc[df_temp["value"].isnull(), "value"] = 0

    # make dm
    dm_train = make_dm(df_temp, years_ots)

    # add ships and planes (assumed to be the same of trains)
    dm_temp = dm_train.filter({"Categories1": ["trains"]})
    dm_temp.rename_col("trains", "ships", "Categories1")
    dm_train.append(dm_temp, "Categories1")
    dm_temp.rename_col("ships", "planes", "Categories1")
    dm_train.append(dm_temp, "Categories1")
    dm_train.sort("Categories1")

    ####################
    ##### PACKAGES #####
    ####################

    # select
    df_temp = df_agg.loc[
        df_agg["variable"].isin(
            ["aluminium-pack", "glass-pack", "paper-pack", "plastic-pack"]
        ),
        :,
    ]

    # fix variables
    df_temp["variable"] = [
        v + "_" + m for v, m in zip(df_temp["variable"], df_temp["material"])
    ]
    df_temp.drop(["material"], axis=1, inplace=True)

    # now assign 0 to nan
    df_temp.loc[df_temp["value"].isnull(), "value"] = 0

    # make dm
    dm_pack = make_dm(df_temp, years_ots)

    # add paper-print (same recovery as paper-pack) and paper-san (0%: hygiene paper is
    # hygienically contaminated and not collectable for recycling)
    dm_temp = dm_pack.filter({"Categories1": ["paper-pack"]})
    dm_temp.rename_col("paper-pack", "paper-print", "Categories1")
    dm_pack.append(dm_temp, "Categories1")
    dm_temp.rename_col("paper-print", "paper-san", "Categories1")
    dm_temp["EU27", :, :, "paper-san", "paper"] = 0
    dm_pack.append(dm_temp, "Categories1")
    dm_pack.sort("Categories1")

    ###############################
    ##### DOMESTIC APPLIANCES #####
    ###############################

    # select
    df_temp = df_agg.loc[df_agg["variable"].isin(["fridge", "dishwasher"]), :]

    # fridge chem seem too high, assigning the ones of dishwasher for now
    df_temp.loc[
        (df_temp["variable"] == "fridge") & (df_temp["material"] == "chem"), "value"
    ] = 0.423

    # fix variables
    df_temp["variable"] = [
        v + "_" + m for v, m in zip(df_temp["variable"], df_temp["material"])
    ]
    df_temp.drop(["material"], axis=1, inplace=True)

    # now assign 0 to nan
    df_temp.loc[df_temp["value"].isnull(), "value"] = 0

    # make dm
    dm_domapp = make_dm(df_temp, years_ots)

    # add dryer and wmachine (as dishwasher), and freezer (as fridge)
    dm_temp = dm_domapp.filter({"Categories1": ["dishwasher"]})
    dm_temp.rename_col("dishwasher", "dryer", "Categories1")
    dm_domapp.append(dm_temp, "Categories1")
    dm_temp.rename_col("dryer", "wmachine", "Categories1")
    dm_domapp.append(dm_temp, "Categories1")
    dm_temp = dm_domapp.filter({"Categories1": ["fridge"]})
    dm_temp.rename_col("fridge", "freezer", "Categories1")
    dm_domapp.append(dm_temp, "Categories1")
    dm_domapp.sort("Categories1")

    #######################
    ##### ELECTRONICS #####
    #######################

    # TODO: for the moment in the weee source considered, there is no mention of batteries
    # for the moment I assume that they are included, at some point we will have to
    # separate them as done for cars

    # select
    df_temp = df_agg.loc[df_agg["variable"].isin(["electronics"]), :]

    # fix variables
    df_temp["variable"] = [
        v + "_" + m for v, m in zip(df_temp["variable"], df_temp["material"])
    ]
    df_temp.drop(["material"], axis=1, inplace=True)

    # now assign 0 to nan
    df_temp.loc[df_temp["value"].isnull(), "value"] = 0

    # make dm
    dm_elec = make_dm(df_temp, years_ots)

    #####################
    ##### BUILDINGS #####
    #####################

    # select
    df_temp = df_agg.loc[df_agg["variable"].isin(["floor-area"]), :]

    # fix variables
    df_temp["variable"] = [
        v + "_" + m for v, m in zip(df_temp["variable"], df_temp["material"])
    ]
    df_temp.drop(["material"], axis=1, inplace=True)

    # now assign 0 to nan
    df_temp.loc[df_temp["value"].isnull(), "value"] = 0

    # make dm
    dm_bld = make_dm(df_temp, years_ots)

    ##########################
    ##### INFRASTRUCTURE #####
    ##########################

    # assumed to be the same of buildings

    dm_infra = dm_bld.copy()
    dm_infra.rename_col("floor-area", "rail", "Categories1")
    dm_temp = dm_infra.filter({"Categories1": ["rail"]})
    dm_temp.rename_col("rail", "road", "Categories1")
    dm_infra.append(dm_temp, "Categories1")
    dm_temp.rename_col("road", "trolley-cables", "Categories1")
    dm_infra.append(dm_temp, "Categories1")
    dm_infra.sort("Categories1")

    # for trolley cables keep only copper (for the moment i put 70%, to be rechecked)
    materials = [
        "aluminium",
        "cement",
        "chem",
        "glass",
        "lime",
        "other",
        "paper",
        "steel",
        "timber",
    ]
    for m in materials:
        dm_infra["EU27", :, :, "trolley-cables", m] = 0
    dm_infra["EU27", :, :, "trolley-cables", "copper"] = 0.7

    # fix rail (keep only steel and timber) and road (put glass and timber to zero)
    materials = [
        "aluminium",
        "cement",
        "chem",
        "copper",
        "glass",
        "lime",
        "other",
        "paper",
    ]
    for m in materials:
        dm_infra["EU27", :, :, "rail", m] = 0
    dm_infra["EU27", :, :, "road", "glass"] = 0
    dm_infra["EU27", :, :, "road", "timber"] = 0

    ########################
    ##### PUT TOGETHER #####
    ########################

    dm = dm_veh.copy()
    dm.append(dm_train, "Categories1")
    dm.append(dm_pack, "Categories1")
    dm.append(dm_domapp, "Categories1")
    dm.append(dm_elec, "Categories1")
    dm.append(dm_bld, "Categories1")
    dm.append(dm_infra, "Categories1")
    dm.sort("Categories1")

    # Asphalt RAP (Reclaimed Asphalt Pavement) recovery for road infrastructure.
    # EU RAP utilisation rate ~88% in 2023 (https://eapa.org/asphalt-industry/statistics/);
    # 85% used as conservative EU-wide average across all road types.
    # All other product categories get 0 (asphalt is not a material in vehicles, buildings, etc.).
    dm.add(0, "Categories2", "asphalt", dummy=True)
    dm["EU27", :, :, "road", "asphalt"] = 0.85

    # nmm (aggregate), met (critical metals), ois-modelled (construction ceramics): 0 recovery.
    # nmm: recycled concrete aggregate is downcycled to road sub-base, not recovered as primary
    # aggregate for new concrete — no closed-loop recycling pathway in current model.
    # met: critical metals (Li, Co, Ni, Mn) in EoL products have no sec-tech in current model;
    # battery recycling is a known gap, flagged as future work.
    # ois-modelled: broken ceramics/tiles go to inert waste, not re-fired into new ceramics.
    dm.add(0, "Categories2", "nmm", dummy=True)
    dm.add(0, "Categories2", "met", dummy=True)
    dm.add(0, "Categories2", "ois-modelled", dummy=True)

    # dm.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

    # save
    f = os.path.join(current_file_directory, "../data/datamatrix/", lever_file)
    with open(f, "wb") as handle:
        pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm


def run(years_ots):
    # directories
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # if exists, load, else make
    lever_file = "lever_material-recovery.pickle"
    filepath = os.path.join(current_file_directory, "../data/datamatrix/" + lever_file)
    if os.path.exists(filepath):
        with open(filepath, "rb") as handle:
            dm = pickle.load(handle)
    else:
        dm = make_material_recovery_dm(current_file_directory, lever_file, years_ots)

    return dm


if __name__ == "__main__":
    years_ots = create_years_list(1990, 2023, 1)

    run(years_ots)

# #######################
# ##### FTS LEVEL 1 #####
# #######################

# # level 1: continuing as is
# dm_fts_level1 = dm.filter({"Years": years_fts})
# # dm_fts_level1.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

# #######################
# ##### FTS LEVEL 4 #####
# #######################

# # TODO: for the moment we put max values following own knowledge, to be re-done with literature
# dm_level4 = dm.copy()
# for y in range(2030, 2055, 5):
#     dm_level4["EU27", y, :, :, :] = np.nan
# dm_level4["EU27", 2050, :, "battery-lion", "aluminium"] = 1
# dm_level4["EU27", 2050, :, "battery-lion", "other"] = 1
# dm_level4["EU27", 2050, :, "battery-lion", "steel"] = 1
# products = ["vehicles", "dishwasher", "dryer", "wmachine"]
# for p in products:
#     dm_level4["EU27", 2050, :, p, "aluminium"] = 1
#     dm_level4["EU27", 2050, :, p, "chem"] = 0.9
#     dm_level4["EU27", 2050, :, p, "copper"] = 1
#     dm_level4["EU27", 2050, :, p, "other"] = 0.9
#     dm_level4["EU27", 2050, :, p, "steel"] = 1
# products = ["electronics"]
# for p in products:
#     dm_level4["EU27", 2050, :, p, "aluminium"] = 1
#     dm_level4["EU27", 2050, :, p, "copper"] = 1
#     dm_level4["EU27", 2050, :, p, "other"] = 0.9
#     dm_level4["EU27", 2050, :, p, "steel"] = 1
# products = ["floor-area"]
# for p in products:
#     dm_level4["EU27", 2050, :, p, "aluminium"] = 1
#     dm_level4["EU27", 2050, :, p, "cement"] = 0.9
#     dm_level4["EU27", 2050, :, p, "chem"] = 0.9
#     dm_level4["EU27", 2050, :, p, "glass"] = 1
#     dm_level4["EU27", 2050, :, p, "other"] = 0.9
#     dm_level4["EU27", 2050, :, p, "steel"] = 1
#     dm_level4["EU27", 2050, :, p, "timber"] = 1
# products = ["freezer", "fridge"]
# for p in products:
#     dm_level4["EU27", 2050, :, p, "aluminium"] = 1
#     dm_level4["EU27", 2050, :, p, "chem"] = 0.9
#     dm_level4["EU27", 2050, :, p, "copper"] = 1
# dm_level4["EU27", 2050, :, "aluminium-pack", "aluminium"] = 1
# dm_level4["EU27", 2050, :, "glass-pack", "glass"] = 1
# dm_level4["EU27", 2050, :, "paper-pack", "paper"] = 1
# dm_level4["EU27", 2050, :, "paper-print", "paper"] = 1
# dm_level4["EU27", 2050, :, "paper-san", "paper"] = 1
# dm_level4["EU27", 2050, :, "plastic-pack", "chem"] = 0.9
# products = [
#     "trains",
#     "planes",
#     "ships",
#     # "metrotram",
# ]
# for p in products:
#     dm_level4["EU27", 2050, :, p, "aluminium"] = 1
#     dm_level4["EU27", 2050, :, p, "chem"] = 0.9
#     dm_level4["EU27", 2050, :, p, "glass"] = 1
#     dm_level4["EU27", 2050, :, p, "other"] = 0.9
#     dm_level4["EU27", 2050, :, p, "steel"] = 1
# dm_level4["EU27", 2050, :, "rail", "steel"] = 1
# dm_level4["EU27", 2050, :, "rail", "timber"] = 1
# dm_level4["EU27", 2050, :, "road", "aluminium"] = 1
# dm_level4["EU27", 2050, :, "road", "cement"] = 1
# dm_level4["EU27", 2050, :, "road", "chem"] = 0.9
# dm_level4["EU27", 2050, :, "road", "steel"] = 1
# dm_level4["EU27", 2050, :, "road", "other"] = 0.9
# dm_level4["EU27", 2050, :, "trolley-cables", "copper"] = 1

# dm_level4 = linear_fitting(dm_level4, years_fts)
# # dm_level4.filter({"Country" : ["EU27"]}).flatten().flatten().datamatrix_plot()
# dm_fts_level4 = dm_level4.filter({"Years": years_fts})
# # dm_fts_level4.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

# #############################
# ##### FTS LEVEL 2 and 3 #####
# #############################

# dm_level2 = dm_level4.copy()
# dm_level3 = dm_level4.copy()
# for y in range(2030, 2050 + 5, 5):
#     dm_level2["EU27", y, :, :, :] = np.nan
#     dm_level3["EU27", y, :, :, :] = np.nan

# p = "aluminium-pack"
# m = "aluminium"
# for p in dm_fts_level1.col_labels["Categories1"]:
#     for m in dm_fts_level1.col_labels["Categories2"]:
#         level1 = dm_fts_level1["EU27", 2050, :, p, m][0]
#         level4 = dm_fts_level4["EU27", 2050, :, p, m][0]
#         arr = np.array([level1, np.nan, np.nan, level4])
#         arr = pd.Series(arr).interpolate().to_numpy()
#         level2 = np.round(arr[1], 2)
#         level3 = np.round(arr[2], 2)
#         dm_level2["EU27", 2050, :, p, m] = level2
#         dm_level3["EU27", 2050, :, p, m] = level3

# dm_level2 = linear_fitting(dm_level2, years_fts)
# # dm_level2.filter({"Country" : ["EU27"]}).flatten().flatten().datamatrix_plot()
# dm_fts_level2 = dm_level2.filter({"Years": years_fts})
# dm_level3 = linear_fitting(dm_level3, years_fts)
# # dm_level3.filter({"Country" : ["EU27"]}).flatten().flatten().datamatrix_plot()
# dm_fts_level3 = dm_level3.filter({"Years": years_fts})

# ################
# ##### SAVE #####
# ################

# # put together
# DM_fts = {
#     1: dm_fts_level1.copy(),
#     2: dm_fts_level2.copy(),
#     3: dm_fts_level3.copy(),
#     4: dm_fts_level4.copy(),
# }
# DM = {"ots": dm_ots, "fts": DM_fts}
# f = "../data/datamatrix/lever_material-recovery.pickle"
# with open(f, "wb") as handle:
#     pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # df = dm.write_df()
# # df_temp = pd.melt(df, id_vars = ['Country', 'Years'], var_name='variable')
# # df_temp = df_temp.loc[df_temp["Country"].isin(["Austria","France"]),:]
# # df_temp = df_temp.loc[df_temp["Years"]==1990,:]
# # name = "temp.xlsx"
# # df_temp.to_excel("~/Desktop/" + name)
