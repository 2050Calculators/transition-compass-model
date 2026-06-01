# packages
import os
import pickle
import warnings

warnings.simplefilter("ignore")
import numpy as np

from transition_compass_model._database.pre_processing.industry.eu.get_data_functions.data_const_energy_demand import (
    get_const_energy_demand_dm,
)
from transition_compass_model.model.common.constant_data_matrix_class import (
    ConstantDataMatrix,
)


def make_const_energy_demand(current_file_directory, lever_file):
    df_final, df_final_feedstock = get_const_energy_demand_dm(current_file_directory)

    ###########################################################
    ############# CONVERT TO CONSTANT DATA MATRIX #############
    ###########################################################

    # create dms
    def create_constant(df, variables):
        df_temp = df.loc[df["variable"].isin(variables), :]

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

    # reshape for efficiency ratios
    def reshape_energy_constant(cdm):
        cdm_temp = cdm.copy()
        cdm_temp.drop("Categories2", "lighting")
        cdm_temp.drop("Categories2", "electricity-else")
        for c in cdm_temp.col_labels["Categories1"]:
            cdm_temp.rename_col(c, c + "_process-heat", "Categories1")
        cdm_temp.deepen("_", "Categories1")
        cdm_temp.switch_categories_order("Categories2", "Categories3")

        cdm_temp1 = cdm.filter({"Categories2": ["lighting", "electricity-else"]})
        cdm_temp1.rename_col(
            ["lighting", "electricity-else"],
            ["lighting_electricity", "elec_electricity"],
            "Categories2",
        )
        cdm_temp1.deepen()
        missing = cdm_temp.col_labels["Categories3"].copy()
        missing.remove("electricity")
        for m in missing:
            cdm_temp1.add(0, "Categories3", m, dummy=True)
        cdm_temp1.sort("Categories3")
        cdm_temp.append(cdm_temp1, "Categories2")
        cdm_temp.sort("Categories2")
        return cdm_temp

    ####################################
    ##### FINAL ENERGY CONSUMPTION #####
    ####################################

    # excluding feedstock
    df_temp = df_final.loc[df_final["energy_demand_type"] == "fec", :]
    tmp = create_constant(df_temp, df_temp["variable"])
    cdm_enerdem_exclfeed = ConstantDataMatrix.create_from_constant(tmp, 0)
    variabs = cdm_enerdem_exclfeed.col_labels["Variables"]
    for v in variabs:
        cdm_enerdem_exclfeed.rename_col(
            v, "energy-demand-excl-feedstock_" + v, "Variables"
        )
    cdm_enerdem_exclfeed.deepen_twice()

    # reshape to be used for efficiency ratios
    cdm_enerdem_exclfeed_reshaped = reshape_energy_constant(cdm_enerdem_exclfeed)
    # cdm_temp = cdm_enerdem_exclfeed_reshaped.filter({"Categories1" : ['steel-BF-BOF']})
    # df_temp = cdm_temp.write_df()

    # aggregate lighting, electricity (from process heat) and electricity-else, and get split
    cdm_temp = cdm_enerdem_exclfeed.filter(
        {"Categories2": ["lighting", "electricity", "electricity-else"]}
    )
    cdm_enerdem_exclfeed.groupby(
        {"electricity": ["lighting", "electricity", "electricity-else"]},
        "Categories2",
        inplace=True,
    )
    cdm_temp.append(
        cdm_temp.groupby(
            {"total": ["lighting", "electricity", "electricity-else"]}, "Categories2"
        ),
        "Categories2",
    )
    cdm_temp.group_all("Categories1")
    cdm_temp[..., "lighting"] = cdm_temp[..., "lighting"] / cdm_temp[..., "total"]
    cdm_temp[..., "electricity"] = cdm_temp[..., "electricity"] / cdm_temp[..., "total"]
    cdm_temp[..., "electricity-else"] = (
        cdm_temp[..., "electricity-else"] / cdm_temp[..., "total"]
    )
    cdm_temp.drop("Categories1", "total")
    cdm_temp.units["energy-demand-excl-feedstock"] = "%"
    df_check = cdm_temp.write_df()
    cdm_enerdem_exclfeed_eleclight_split = cdm_temp.copy()

    # feedstock
    df_temp = df_final_feedstock.loc[
        df_final_feedstock["energy_demand_type"] == "fec", :
    ]
    tmp = create_constant(df_temp, df_temp["variable"])
    cdm_enerdem_feedstock = ConstantDataMatrix.create_from_constant(tmp, 1)
    variabs_missing = cdm_enerdem_feedstock.col_labels["Variables"]
    variabs_missing = [
        i not in variabs_missing for i in cdm_enerdem_exclfeed.col_labels["Categories1"]
    ]
    variabs_missing = list(
        np.array(cdm_enerdem_exclfeed.col_labels["Categories1"])[variabs_missing]
    )
    for v in variabs_missing:
        cdm_enerdem_feedstock.add(0, "Variables", v, dummy=True)
    cdm_enerdem_feedstock.sort("Variables")
    cdm_enerdem_feedstock = cdm_enerdem_feedstock.flatten()
    variabs = cdm_enerdem_feedstock.col_labels["Variables"]
    for v in variabs:
        cdm_enerdem_feedstock.rename_col(v, "energy-demand-feedstock_" + v, "Variables")
    cdm_enerdem_feedstock.deepen_twice()
    cdm_enerdem_feedstock.units["energy-demand-feedstock"] = "TWh/Mt"
    cdm_enerdem_feedstock.add(0, "Categories2", "electricity-else", dummy=True)
    cdm_enerdem_feedstock.sort("Categories2")

    # reshape to be used for efficiency ratios
    cdm_enerdem_feedstock_reshaped = reshape_energy_constant(cdm_enerdem_feedstock)

    # aggregate lighting, electricity and electricity-else
    cdm_enerdem_feedstock.groupby(
        {"electricity": ["lighting", "electricity", "electricity-else"]},
        "Categories2",
        inplace=True,
    )

    # put together excl feedstock and feedstock (to be used for efficiency rations)
    cdm_enerdem_fec = cdm_enerdem_exclfeed_reshaped.copy()
    cdm_enerdem_fec.append(cdm_enerdem_feedstock_reshaped, "Variables")
    cdm_enerdem_fec.groupby(
        {
            "energy-demand-fec": [
                "energy-demand-excl-feedstock",
                "energy-demand-feedstock",
            ]
        },
        "Variables",
        inplace=True,
    )
    cdm_enerdem_fec.group_all("Categories1")

    # save
    CDM_energy_demand = {
        "energy-demand-excl-feedstock-eleclight-split": cdm_enerdem_exclfeed_eleclight_split,
    }

    ####################################
    ##### ENERGY EFFICIENCY RATIOS #####
    ####################################

    # ued excluding feedstock
    df_temp = df_final.loc[df_final["energy_demand_type"] == "ued", :]
    tmp = create_constant(df_temp, df_temp["variable"])
    cdm_enerdem_exclfeed = ConstantDataMatrix.create_from_constant(tmp, 0)
    variabs = cdm_enerdem_exclfeed.col_labels["Variables"]
    for v in variabs:
        cdm_enerdem_exclfeed.rename_col(
            v, "energy-demand-excl-feedstock_" + v, "Variables"
        )
    cdm_enerdem_exclfeed.deepen_twice()

    # reshape
    cdm_enerdem_exclfeed_reshaped = reshape_energy_constant(cdm_enerdem_exclfeed)

    # ued feedstock
    df_temp = df_final_feedstock.loc[
        df_final_feedstock["energy_demand_type"] == "ued", :
    ]
    tmp = create_constant(df_temp, df_temp["variable"])
    cdm_enerdem_feedstock = ConstantDataMatrix.create_from_constant(tmp, 1)
    variabs_missing = cdm_enerdem_feedstock.col_labels["Variables"]
    variabs_missing = [
        i not in variabs_missing for i in cdm_enerdem_exclfeed.col_labels["Categories1"]
    ]
    variabs_missing = list(
        np.array(cdm_enerdem_exclfeed.col_labels["Categories1"])[variabs_missing]
    )
    for v in variabs_missing:
        cdm_enerdem_feedstock.add(0, "Variables", v, dummy=True)
    cdm_enerdem_feedstock.sort("Variables")
    cdm_enerdem_feedstock = cdm_enerdem_feedstock.flatten()
    variabs = cdm_enerdem_feedstock.col_labels["Variables"]
    for v in variabs:
        cdm_enerdem_feedstock.rename_col(v, "energy-demand-feedstock_" + v, "Variables")
    cdm_enerdem_feedstock.deepen_twice()
    cdm_enerdem_feedstock.units["energy-demand-feedstock"] = "TWh/Mt"
    cdm_enerdem_feedstock.add(0, "Categories2", "electricity-else", dummy=True)
    cdm_enerdem_feedstock.sort("Categories2")

    # reshape to be used for efficiency ratios
    cdm_enerdem_feedstock_reshaped = reshape_energy_constant(cdm_enerdem_feedstock)

    # put together excl feedstock and feedstock (to be used for efficiency rations)
    cdm_enerdem_ued = cdm_enerdem_exclfeed_reshaped.copy()
    cdm_enerdem_ued.append(cdm_enerdem_feedstock_reshaped, "Variables")
    cdm_enerdem_ued.groupby(
        {
            "energy-demand-fec": [
                "energy-demand-excl-feedstock",
                "energy-demand-feedstock",
            ]
        },
        "Variables",
        inplace=True,
    )
    cdm_enerdem_ued.group_all("Categories1")

    # make ratios
    cdm_enerdem_eff = cdm_enerdem_fec.copy()
    cdm_enerdem_eff.array = cdm_enerdem_ued.array / cdm_enerdem_fec.array
    cdm_enerdem_eff.rename_col("energy-demand-fec", "energy-efficiency", "Variables")
    cdm_enerdem_eff.units["energy-efficiency"] = "%"
    df_check = cdm_enerdem_eff.write_df()
    df_check.melt()

    # save
    CDM_energy_demand["energy-efficiency"] = cdm_enerdem_eff.copy()

    # save
    f = os.path.join(current_file_directory, "../data/datamatrix/" + lever_file)
    with open(f, "wb") as handle:
        pickle.dump(CDM_energy_demand, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return cdm_enerdem_exclfeed_eleclight_split, cdm_enerdem_eff


def run():
    # directories
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # if exists, load, else make
    lever_file = "const_energy-demand.pickle"
    filepath = os.path.join(current_file_directory, "../data/datamatrix/" + lever_file)
    if os.path.exists(filepath):
        with open(filepath, "rb") as handle:
            CDM = pickle.load(handle)
            cdm_enerdem_exclfeed_eleclight_split = CDM[
                "energy-demand-excl-feedstock-eleclight-split"
            ].copy()
            cdm_enerdem_eff = CDM["energy-efficiency"].copy()
    else:
        cdm_enerdem_exclfeed_eleclight_split, cdm_enerdem_eff = (
            make_const_energy_demand(current_file_directory, lever_file)
        )

    return cdm_enerdem_exclfeed_eleclight_split, cdm_enerdem_eff


if __name__ == "__main__":
    run()


# df = cdm_enerdem_exclfeed.write_df()
# df["country"] = "all"
# df_temp = pd.melt(df, id_vars = ['country'], var_name='variable')
# name = "temp.xlsx"
# df_temp.to_excel("~/Desktop/" + name)
