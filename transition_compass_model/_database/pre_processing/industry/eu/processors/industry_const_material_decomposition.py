# packages
import pickle
import warnings

warnings.simplefilter("ignore")
import os
import re

import numpy as np
import pandas as pd

from transition_compass_model._database.pre_processing.industry.eu.get_data_functions.data_const_material_decomposition import (
    get_material_decomposition_data,
)
from transition_compass_model.model.common.constant_data_matrix_class import (
    ConstantDataMatrix,
)


def make_material_decomposition_dm(current_file_directory, lever_file):
    #############################################
    ##### NEW CONSTANTS FROM LIT REV BY E4S #####
    #############################################

    # get data
    df_agg = get_material_decomposition_data()

    # make datamatrixes
    def create_constant(df, variables):
        df_temp = df.loc[df["variable"].isin(variables), :]

        # rename variables
        df_temp["variable"] = [
            v.split("[")[0] + "_" + m + "[" + v.split("[")[1]
            for v, m in zip(df_temp["variable"], df_temp["material"])
        ]
        df_temp.drop(["material"], axis=1, inplace=True)

        # put unit
        df_temp["unit"] = [i.split("[")[1].split("]")[0] for i in df_temp["variable"]]

        const = {
            "name": list(df_temp["variable"]),
            "value": list(df_temp["value"]),
            "idx": dict(
                zip(list(df_temp["variable"]), range(len(df_temp["variable"])))
            ),
            "units": dict(zip(list(df_temp["variable"]), list(df_temp["unit"]))),
        }

        # return
        return const

    # cdm_bld_floor
    tmp = create_constant(
        df_agg,
        [
            "floor-area-new-residential[t/m2]",
            "floor-area-new-non-residential[t/m2]",
            "floor-area-reno-residential[t/m2]",
            "floor-area-reno-non-residential[t/m2]",
        ],
    )
    cdm_bld_floor = ConstantDataMatrix.create_from_constant(tmp, 1)
    cdm_check = cdm_bld_floor.group_all("Categories1", inplace=False)
    df_check = pd.melt(cdm_check.write_df())

    # cdm_bld_pipe
    tmp = create_constant(df_agg, ["new-dhg-pipe[t/km]"])
    cdm_bld_pipe = ConstantDataMatrix.create_from_constant(tmp, 1)
    cdm_check = cdm_bld_pipe.group_all("Categories1", inplace=False)
    df_check = pd.melt(cdm_check.write_df())

    # cdm_domapp
    tmp = create_constant(
        df_agg,
        [
            "fridge[t/num]",
            "dishwasher[t/num]",
            "wmachine[t/num]",
            "freezer[t/num]",
            "dryer[t/num]",
            "tv[t/num]",
            "phone[t/num]",
            "computer[t/num]",
        ],
    )
    cdm_domapp = ConstantDataMatrix.create_from_constant(tmp, 1)
    cdm_check = cdm_domapp.group_all("Categories1", inplace=False)
    df_check = pd.melt(cdm_check.write_df())

    # cdm_tra_veh
    variabs = df_agg["variable"].unique()
    variabs = list(
        np.array(variabs)[
            [bool(re.search("HDV|LDV|planes|trains|ships", i)) for i in variabs]
        ]
    )
    variabs = list(
        np.array(variabs)[
            [not bool(re.search("battery", i, re.IGNORECASE)) for i in variabs]
        ]
    )
    tmp = create_constant(df_agg, variabs)
    cdm_tra_veh = ConstantDataMatrix.create_from_constant(tmp, 1)

    # add missing veh
    # I assume buses to be similar to trucks, as the overall weight and material composition are similar
    # source for buses (it's a thesis): https://www.theseus.fi/bitstream/handle/10024/52377/Karna_Paivi.pdf.pdf?sequence=1

    idx = cdm_tra_veh.idx
    cdm_tra_veh.add(
        cdm_tra_veh.array[idx["HDV_BEV"], :], "Variables", "bus_BEV", unit="t/num"
    )
    cdm_tra_veh.add(
        cdm_tra_veh.array[idx["HDV_FCEV"], :], "Variables", "bus_FCEV", unit="t/num"
    )
    cdm_tra_veh.add(
        cdm_tra_veh.array[idx["HDV_ICE-diesel"], :],
        "Variables",
        "bus_ICE-diesel",
        unit="t/num",
    )
    cdm_tra_veh.add(
        cdm_tra_veh.array[idx["HDV_PHEV-diesel"], :],
        "Variables",
        "bus_PHEV-diesel",
        unit="t/num",
    )
    cdm_tra_veh.add(
        cdm_tra_veh.array[idx["HDV_ICE-diesel"], :],
        "Variables",
        "HDV_ICE-gas",
        unit="t/num",
    )
    cdm_tra_veh.add(
        cdm_tra_veh.array[idx["LDV_ICE-gasoline"], :],
        "Variables",
        "LDV_ICE-gas",
        unit="t/num",
    )
    cdm_tra_veh.add(
        cdm_tra_veh.array[idx["LDV_PHEV-gasoline"], :],
        "Variables",
        "LDV_PHEV-diesel",
        unit="t/num",
    )
    cdm_tra_veh.add(
        cdm_tra_veh.array[idx["bus_ICE-diesel"], :],
        "Variables",
        "bus_ICE-gas",
        unit="t/num",
    )
    cdm_tra_veh.add(
        cdm_tra_veh.array[idx["HDV_PHEV-diesel"], :],
        "Variables",
        "HDV_PHEV-gasoline",
        unit="t/num",
    )
    cdm_tra_veh.add(
        cdm_tra_veh.array[idx["HDV_ICE-diesel"], :],
        "Variables",
        "HDV_ICE-gasoline",
        unit="t/num",
    )
    cdm_tra_veh.add(
        cdm_tra_veh.array[idx["bus_ICE-diesel"], :],
        "Variables",
        "bus_ICE-gasoline",
        unit="t/num",
    )
    cdm_tra_veh.add(
        cdm_tra_veh.array[idx["bus_PHEV-diesel"], :],
        "Variables",
        "bus_PHEV-gasoline",
        unit="t/num",
    )
    # trains: diesel copy of CEV (same mass, no catenary)
    cdm_tra_veh.add(
        cdm_tra_veh.array[idx["trains_CEV"], :],
        "Variables",
        "trains_ICE-diesel",
        unit="t/num",
    )

    # Override trains_CEV with a weighted average of passenger and freight EU-configs.
    # CH OTS average: 57 new passenger EU-configs/yr, 45 freight EU-configs/yr
    # (passenger EU-config = 6.2 BFS carriages ÷ 6.2; freight EU-config = 63.2 wagons ÷ 63.2).
    # Passenger values: 310 t per config (steel 186, Al 46.5, other 65.1, chem 6.2, glass 3.1, Cu 3.1)
    # Freight values: 2788 t per config from literature (steel 1801, Al 279, other 620, chem 84, Cu 4)
    # Weighted average (w_pass=0.557, w_fre=0.443):
    mat_labels = cdm_tra_veh.col_labels["Categories1"]
    mat_idx_map = {m: i for i, m in enumerate(mat_labels)}
    avg_arr = np.zeros(len(mat_labels))
    avg_arr[mat_idx_map["steel"]] = 901.4  # 0.557×186 + 0.443×1801
    avg_arr[mat_idx_map["aluminium"]] = 149.4  # 0.557×46.5 + 0.443×279
    avg_arr[mat_idx_map["other"]] = 311.0  # 0.557×65.1 + 0.443×620
    avg_arr[mat_idx_map["chem"]] = 40.5  # 0.557×6.2  + 0.443×84
    avg_arr[mat_idx_map["glass"]] = 1.7  # 0.557×3.1  + 0.443×0
    avg_arr[mat_idx_map["copper"]] = 3.6  # 0.557×3.1  + 0.443×4.3
    cdm_tra_veh.array[idx["trains_CEV"], :] = avg_arr
    cdm_tra_veh.array[idx["trains_ICE-diesel"], :] = avg_arr

    # Add body copper for HDV and bus: wiring harness, alternator/starter motor.
    # Not reported in the literature review (data gap) but physically present in all powertrains.
    # Values based on automotive industry data: HDV ~25 kg, bus ~30 kg.
    idx = cdm_tra_veh.idx  # refresh after all add() calls
    idx_cu = mat_idx_map["copper"]
    for var in cdm_tra_veh.col_labels["Variables"]:
        if var.startswith("HDV_"):
            cdm_tra_veh.array[idx[var], idx_cu] = 0.025
        elif var.startswith("bus_"):
            cdm_tra_veh.array[idx[var], idx_cu] = 0.030

    cdm_tra_veh.sort("Variables")
    cdm_tra_veh.deepen(based_on="Variables")
    cdm_tra_veh.switch_categories_order("Categories1", "Categories2")
    cdm_check = cdm_tra_veh.group_all("Categories2", inplace=False)
    df_check = pd.melt(cdm_check.write_df())

    # batteries
    tmp = create_constant(
        df_agg,
        [
            "battery-lion-HDV_BEV[t/num]",
            "battery-lion-HDV_PHEV[t/num]",
            "battery-lion-LDV_BEV[t/num]",
            "battery-lion-LDV_PHEV[t/num]",
        ],
    )
    cdm_tra_bat = ConstantDataMatrix.create_from_constant(tmp, 2)
    cdm_temp = cdm_tra_bat.filter({"Variables": ["battery-lion-HDV"]})
    cdm_temp.rename_col("battery-lion-HDV", "battery-lion-bus", "Variables")
    cdm_tra_bat.append(cdm_temp, "Variables")
    cdm_tra_bat.sort("Variables")

    # add missing categories for PHEV
    idx = cdm_tra_bat.idx
    cdm_tra_bat.rename_col("PHEV", "PHEV-gasoline", "Categories1")
    cdm_tra_bat.add(
        cdm_tra_bat.array[:, idx["PHEV-gasoline"], :],
        "Categories1",
        "PHEV-diesel",
        unit="t/num",
    )
    cdm_tra_bat.add(
        cdm_tra_bat.array[:, idx["PHEV-gasoline"], :],
        "Categories1",
        "FCEV",
        unit="t/num",
    )

    # add other missing categories
    missing = ["CEV", "ICE", "ICE-diesel", "ICE-gas", "ICE-gasoline"]
    cdm_tra_bat.add(0, col_label=missing, dummy=True, dim="Categories1")
    cdm_tra_bat.sort("Categories1")
    cdm_check = cdm_tra_bat.group_all("Categories2", inplace=False)
    df_check = pd.melt(cdm_check.write_df())

    # cdm_tra_infra
    tmp = create_constant(df_agg, ["road[t/km]", "rail[t/km]", "trolley-cables[t/km]"])
    cdm_tra_infra = ConstantDataMatrix.create_from_constant(tmp, 1)
    # Rail steel: literature review stores 0.060 t/m (one UIC-60 rail at 60 kg/m), not t/km.
    # Corrected value: 0.060 t/m × 1000 m/km × 2 rails = 120 t/km.
    idx = cdm_tra_infra.idx
    cdm_tra_infra.array[idx["rail"], idx["steel"]] = 120.0
    cdm_check = cdm_tra_infra.group_all("Categories1", inplace=False)
    df_check = pd.melt(cdm_check.write_df())

    # cdm_fert
    tmp = create_constant(df_agg, ["fertilizer[t/t]"])
    cdm_fert = ConstantDataMatrix.create_from_constant(tmp, 1)
    cdm_check = cdm_fert.group_all("Categories1", inplace=False)
    df_check = pd.melt(cdm_check.write_df())

    # cdm_pack
    tmp = create_constant(
        df_agg,
        [
            "plastic-pack[t/t]",
            "paper-pack[t/t]",
            "aluminium-pack[t/t]",
            "glass-pack[t/t]",
            "paper-print[t/t]",
            "paper-san[t/t]",
        ],
    )
    cdm_pack = ConstantDataMatrix.create_from_constant(tmp, 1)
    cdm_check = cdm_pack.group_all("Categories1", inplace=False)
    # df_check = pd.melt(cdm_check.write_df())

    # put together
    CDM_matdec = {
        "pack": cdm_pack,
        "tra_veh": cdm_tra_veh,
        "tra_bat": cdm_tra_bat,
        "tra_infra": cdm_tra_infra,
        "bld_floor": cdm_bld_floor,
        "bld_pipe": cdm_bld_pipe,
        "bld_domapp": cdm_domapp,
        "fertilizer": cdm_fert,
    }

    # rename
    for key in [
        "pack",
        "tra_infra",
        "bld_floor",
        "bld_pipe",
        "bld_domapp",
        "fertilizer",
    ]:
        variabs = CDM_matdec[key].col_labels["Variables"]
        for v in variabs:
            CDM_matdec[key].rename_col(v, "material-decomp_" + v, "Variables")
        CDM_matdec[key] = CDM_matdec[key].flatten()
        CDM_matdec[key].deepen_twice()

    for key in ["tra_veh", "tra_bat"]:
        variabs = CDM_matdec[key].col_labels["Variables"]
        for v in variabs:
            CDM_matdec[key].rename_col(v, "material-decomp_" + v, "Variables")
        CDM_matdec[key] = CDM_matdec[key].flatten()
        CDM_matdec[key].deepen(based_on="Variables")
        CDM_matdec[key].switch_categories_order("Categories1", "Categories2")
        CDM_matdec[key].deepen(based_on="Categories2")

    # # drop other
    # # note: in general we drop other as we do not have a general technology for other materials
    # # we could keep "other" and use it in industry module until the technology part, though we would need to adjust
    # # net import to add other raw materials ... we'll do it only if we decide
    # # to add a general tech for other at some point.
    # for key in ['pack', 'tra_infra', 'bld_floor', 'bld_pipe', 'bld_domapp', 'fertilizer']:
    #     CDM_matdec[key].drop("Categories2","other")
    # CDM_matdec["tra_veh"].drop("Categories3","other")
    # CDM_matdec["tra_bat"].drop("Categories3","other")

    # get electronics
    elecs = ["computer", "phone", "tv"]
    CDM_matdec["bld_elec"] = CDM_matdec["bld_domapp"].filter({"Categories1": elecs})
    CDM_matdec["bld_domapp"].drop("Categories1", elecs)

    # save
    f = os.path.join(current_file_directory, "../data/datamatrix/" + lever_file)
    with open(f, "wb") as handle:
        pickle.dump(CDM_matdec, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # cdm_temp = CDM_matdec["bld_domapp"].copy()
    # idx = cdm_temp.idx
    # cdm_temp.array[cdm_temp.array == 0] = np.nan
    # cdm_temp.write_df().columns

    cdm_pack = CDM_matdec["pack"].copy()
    cdm_tra_veh = CDM_matdec["tra_veh"].copy()
    cdm_tra_bat = CDM_matdec["tra_bat"].copy()
    cdm_tra_infra = CDM_matdec["tra_infra"].copy()
    cdm_bld_floor = CDM_matdec["bld_floor"].copy()
    cdm_bld_pipe = CDM_matdec["bld_pipe"].copy()
    cdm_domapp = CDM_matdec["bld_domapp"].copy()
    cdm_elec = CDM_matdec["bld_elec"].copy()
    cdm_fert = CDM_matdec["fertilizer"].copy()

    return (
        cdm_pack,
        cdm_tra_veh,
        cdm_tra_bat,
        cdm_tra_infra,
        cdm_bld_floor,
        cdm_bld_pipe,
        cdm_domapp,
        cdm_elec,
        cdm_fert,
    )


def run():
    # directories
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # if exists, load, else make
    lever_file = "const_material-decomposition.pickle"
    filepath = os.path.join(current_file_directory, "../data/datamatrix/" + lever_file)
    if os.path.exists(filepath):
        with open(filepath, "rb") as handle:
            CDM_matdec = pickle.load(handle)
            cdm_pack = CDM_matdec["pack"].copy()
            cdm_tra_veh = CDM_matdec["tra_veh"].copy()
            cdm_tra_bat = CDM_matdec["tra_bat"].copy()
            cdm_tra_infra = CDM_matdec["tra_infra"].copy()
            cdm_bld_floor = CDM_matdec["bld_floor"].copy()
            cdm_bld_pipe = CDM_matdec["bld_pipe"].copy()
            cdm_domapp = CDM_matdec["bld_domapp"].copy()
            cdm_elec = CDM_matdec["bld_elec"].copy()
            cdm_fert = CDM_matdec["fertilizer"].copy()
    else:
        (
            cdm_pack,
            cdm_tra_veh,
            cdm_tra_bat,
            cdm_tra_infra,
            cdm_bld_floor,
            cdm_bld_pipe,
            cdm_domapp,
            cdm_elec,
            cdm_fert,
        ) = make_material_decomposition_dm(current_file_directory, lever_file)

    return (
        cdm_pack,
        cdm_tra_veh,
        cdm_tra_bat,
        cdm_tra_infra,
        cdm_bld_floor,
        cdm_bld_pipe,
        cdm_domapp,
        cdm_elec,
        cdm_fert,
    )


if __name__ == "__main__":
    run()


# =============================================================================
# ##################
# ##### EUCALC #####
# ##################
#
# # get data
# filepath = os.path.join(current_file_directory, '../data/EUCalc/products_material_composition.xlsx')
# df = pd.read_excel(filepath)
#
# # fix product names
# df.loc[:,"product"]
# name_old = ["Residential buildings [kg/m2 floor]", "Non-residential buildings [kg/m2 floor]",
#             "Insulation residential buildings [t/m2 wall]", "Insulation non- residential buildings [t/m2 wall]",
#             "District heating pipes [t/km]","Fridges [kg/num]", "Dishwashers [kg/num]",
#             "Washing machines [kg/num]", "Freezers [kg/num]", "Dryer [kg/num]",
#             "TV [kg/num]", "Smartphone [kg/num]", "Computer [kg/num]",
#             "ICE cars [t/num]", "ICE trucks [t/num]", "FCV cars [t/num]",
#             "FCV trucks [t/num]", "EV cars [t/num]", "EV trucks [t/num]",
#             "Ships [t/num]", "Trains [t/num]", "Planes [t/num]", "Road [t/km]",
#             "Rail [t/km]", "Trolley-cables [t/km]", "Fertilizer [t/t]",
#             "Plastic packaging [t/t]", "Paper packaging [t/t]", "Aluminium packaging [t/t]",
#             "Glass packaging [t/t]", "Paper printing and graphic [t/t]",
#             "Paper sanitary and household [t/t]"]
# name_new = ["floor-area-new-residential[kg/m2]", "floor-area-new-non-residential[kg/m2]",
#             "floor-area-reno-residential[t/m2]", "floor-area-reno-non-residential[t/m2]",
#             "new-dhg-pipe[t/km]", "fridge[kg/num]", "dishwasher[kg/num]",
#             "wmachine[kg/num]", "freezer[kg/num]", "dryer[kg/num]",
#             "tv[kg/num]", "phone[kg/num]", "computer[kg/num]",
#             "cars-ICE[t/num]", "trucks-ICE[t/num]", "cars-FCV[t/num]",
#             "trucks-FCV[t/num]", "cars-EV[t/num]", "trucks-EV[t/num]",
#             "ships[t/num]", "trains[t/num]", "planes[t/num]", "road[t/km]",
#             "rail[t/km]", "trolly-cables[t/km]", "fertilizer[t/t]",
#             "plastic-pack[t/t]", "paper-pack[t/t]","aluminium-pack[t/t]",
#             "glass-pack[t/t]", "paper-print[t/t]",
#             "paper-san[t/t]"]
# for i in range(0,len(name_old)):
#     df.loc[df["product"] == name_old[i],"product"] = name_new[i]
#
# # fix columns
# df.rename(columns={"other chemicals" : "chem", "product" : "variable"}, inplace=True)
#
# # melt
# indexes = ["variable"]
# df = pd.melt(df, id_vars = indexes, var_name='material')
#
# # fix units
# df.loc[df["variable"] == "floor-area-new-residential[kg/m2]","value"] = \
#     df.loc[df["variable"] == "floor-area-new-residential[kg/m2]","value"] / 1000
# df.loc[df["variable"] == "floor-area-new-non-residential[kg/m2]","value"] = \
#     df.loc[df["variable"] == "floor-area-new-non-residential[kg/m2]","value"] / 1000
# df.loc[df["variable"] == "floor-area-new-residential[kg/m2]","variable"] = "floor-area-new-residential[t/m2]"
# df.loc[df["variable"] == "floor-area-new-non-residential[kg/m2]","variable"] = "floor-area-new-non-residential[t/m2]"
# ls_temp = ["fridge[kg/num]", "dishwasher[kg/num]", "wmachine[kg/num]",
#            "freezer[kg/num]", "dryer[kg/num]", "tv[kg/num]", "phone[kg/num]",
#            "computer[kg/num]"]
# ls_temp1 = ["fridge[t/num]", "dishwasher[t/num]", "wmachine[t/num]",
#            "freezer[t/num]", "dryer[t/num]", "tv[t/num]", "phone[t/num]",
#            "computer[t/num]"]
# for i in range(0, len(ls_temp)):
#     df.loc[df["variable"] == ls_temp[i],"value"] = \
#         df.loc[df["variable"] == ls_temp[i],"value"] / 1000
#     df.loc[df["variable"] == ls_temp[i],"variable"] = ls_temp1[i]
#
# # drop other
# # note: in general we drop other as we do not have a general technology for other materials
# # we could keep "other" and use it in industry module until the technology part, though we would need to adjust
# # net import to add other raw materials ... we'll do it only if we decide
# # to add a general tech for other at some point.
# df = df.loc[df["material"] != "other",:]
#
# # create dms
# def create_constant(df, variables):
#
#     df_temp = df.loc[df["variable"].isin(variables),:]
#
#     # rename variables
#     df_temp["variable"] = [v.split("[")[0] + "_" + m + "[" + v.split("[")[1] for v, m in zip(df_temp["variable"],df_temp["material"])]
#     df_temp.drop(["material"], axis=1, inplace=True)
#
#     # put unit
#     df_temp["unit"] = [i.split("[")[1].split("]")[0] for i in df_temp["variable"]]
#
#     const = {
#         'name': list(df_temp['variable']),
#         'value': list(df_temp['value']),
#         'idx': dict(zip(list(df_temp['variable']), range(len(df_temp['variable'])))),
#         'units': dict(zip(list(df_temp['variable']), list(df_temp['unit'])))
#     }
#
#     # return
#     return const
#
# # cdm_bld_floor
# tmp = create_constant(df, ["floor-area-new-residential[t/m2]", "floor-area-new-non-residential[t/m2]",
#                            "floor-area-reno-residential[t/m2]", "floor-area-reno-non-residential[t/m2]"])
# cdm_bld_floor = ConstantDataMatrix.create_from_constant(tmp, 1)
#
# # cdm_bld_pipe
# tmp = create_constant(df, ["new-dhg-pipe[t/km]"])
# cdm_bld_pipe = ConstantDataMatrix.create_from_constant(tmp, 1)
#
# # cdm_domapp
# tmp = create_constant(df, ["fridge[t/num]", "dishwasher[t/num]","wmachine[t/num]",
#                            "freezer[t/num]", "dryer[t/num]", "tv[t/num]",
#                            "phone[t/num]", "computer[t/num]"])
# cdm_domapp = ConstantDataMatrix.create_from_constant(tmp, 1)
#
# # cdm_tra_veh
# tmp = create_constant(df, ["cars-ICE[t/num]", "trucks-ICE[t/num]", "cars-FCV[t/num]",
#                            "ships[t/num]", "trains[t/num]", "planes[t/num]",
#                            "trucks-FCV[t/num]", "cars-EV[t/num]", "trucks-EV[t/num]"])
# cdm_tra_veh = ConstantDataMatrix.create_from_constant(tmp, 1)
#
# # cdm_tra_infra
# tmp = create_constant(df, ["road[t/km]", "rail[t/km]", "trolly-cables[t/km]"])
# cdm_tra_infra = ConstantDataMatrix.create_from_constant(tmp, 1)
#
# # cdm_fert
# tmp = create_constant(df, ["fertilizer[t/t]"])
# cdm_fert = ConstantDataMatrix.create_from_constant(tmp, 1)
#
# # cdm_lfs
# tmp = create_constant(df, ["plastic-pack[t/t]", "paper-pack[t/t]", "aluminium-pack[t/t]",
#                            "glass-pack[t/t]", "paper-print[t/t]", "paper-san[t/t]"])
# cdm_lfs = ConstantDataMatrix.create_from_constant(tmp, 1)
#
# # put together
# CDM_matdec = {
#     "lfs" : cdm_lfs,
#     "tra_veh" : cdm_tra_veh,
#     "tra_infra" : cdm_tra_infra,
#     "bld_floor" : cdm_bld_floor,
#     "bld_pipe" : cdm_bld_pipe,
#     "bld_domapp" : cdm_domapp,
#     "fertilizer" : cdm_fert
#     }
#
# # rename
# for key in CDM_matdec.keys():
#     variabs = CDM_matdec[key].col_labels["Variables"]
#     for v in variabs:
#         CDM_matdec[key].rename_col(v, "material-decomp_" + v, "Variables")
#     CDM_matdec[key] = CDM_matdec[key].flatten()
#     CDM_matdec[key].deepen_twice()
#
# # save
# f = os.path.join(current_file_directory, '../data/datamatrix/const_material-decomposition.pickle')
# with open(f, 'wb') as handle:
#     pickle.dump(CDM_matdec, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# # cdm_temp = CDM_matdec["bld_domapp"].copy()
# # idx = cdm_temp.idx
# # cdm_temp.array[cdm_temp.array == 0] = np.nan
# # cdm_temp.write_df().columns
# =============================================================================
