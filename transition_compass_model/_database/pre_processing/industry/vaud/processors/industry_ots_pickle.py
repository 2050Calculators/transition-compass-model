import os
import pickle

import numpy as np

from transition_compass_model.model.common.auxiliary_functions import create_years_list


def run(DM_input, years_ots):
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    DM = {"ots": dict(), "fxa": dict(), "calibration": dict()}

    DM["ots"]["product-net-import"] = DM_input["product-net-import"]
    DM["ots"]["material-net-import"] = DM_input["material-net-import"]
    DM["ots"]["paperpack"] = DM_input["packaging"]
    DM["ots"]["eol-waste-management"] = DM_input["waste-management"]

    DM["fxa"]["prod"] = DM_input["material-production-not-modelled"]
    DM["fxa"]["demand"] = DM_input["material-demand-wpp"]

    # Load CH industry pickle for levers that are copied unchanged
    pickle_file = os.path.join(
        current_file_directory, "../../../../data/datamatrix/industry.pickle"
    )
    with open(pickle_file, "rb") as handle:
        DM_ch = pickle.load(handle)

    # Policy levers: copy from CH (same national/federal policy framework applies to Vaud)
    other_levers = [
        "material-switch",
        "material-efficiency",
        "technology-development",
        "cc",
        "technology-share",
        "energy-carrier-mix",
        "eol-material-recovery",
    ]
    for lev in other_levers:
        dm_temp = DM_ch["ots"][lev].filter({"Country": ["Switzerland"]}).copy()
        dm_temp.rename_col("Switzerland", "Vaud", dim="Country")
        DM["ots"][lev] = dm_temp

    # Cost FXA: copy from CH
    for fxa_key in ["cost-matprod", "cost-CC"]:
        dm_temp = DM_ch["fxa"][fxa_key].filter({"Country": ["Switzerland"]}).copy()
        dm_temp.rename_col("Switzerland", "Vaud", dim="Country")
        DM["fxa"][fxa_key] = dm_temp

    # Energy demand FXA (Vaud-specific carrier shares already applied)
    DM["fxa"]["energy-demand-excl-feedstock"] = DM_input["fxa-energy-exclfeedstock"]
    DM["fxa"]["energy-demand-feedstock"] = DM_input["fxa-energy-feedstock"]

    # Calibration: all NaN (no Vaud-specific calibration data available)
    for calib_key in ["emissions", "energy-demand", "material-production"]:
        dm_temp = (
            DM_ch["calibration"][calib_key].filter({"Country": ["Switzerland"]}).copy()
        )
        dm_temp.rename_col("Switzerland", "Vaud", dim="Country")
        dm_temp[...] = np.nan
        DM["calibration"][calib_key] = dm_temp

    # Save intermediate OTS pickle
    os.makedirs(
        os.path.join(current_file_directory, "../data/datamatrix"), exist_ok=True
    )
    f = os.path.join(current_file_directory, "../data/datamatrix/industry_ots.pickle")
    with open(f, "wb") as handle:
        pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # ---- Ammonia OTS ----
    DM_amm = {"ots": dict(), "fxa": dict(), "calibration": dict()}

    ammonia_pickle_file = os.path.join(
        current_file_directory, "../../../../data/datamatrix/ammonia.pickle"
    )
    with open(ammonia_pickle_file, "rb") as handle:
        DM_amm_ch = pickle.load(handle)

    # Product/material net-import: copy from CH (no Vaud ammonia production)
    for key in ["product-net-import", "material-net-import"]:
        dm_temp = DM_amm_ch["ots"][key].filter({"Country": ["Switzerland"]}).copy()
        dm_temp.rename_col("Switzerland", "Vaud", dim="Country")
        DM_amm["ots"][key] = dm_temp

    # Policy levers for ammonia: copy from CH
    amm_other_levers = [
        "material-efficiency",
        "eol-material-recovery",
        "technology-development",
        "cc",
        "energy-carrier-mix",
    ]
    for lev in amm_other_levers:
        dm_temp = DM_amm_ch["ots"][lev].filter({"Country": ["Switzerland"]}).copy()
        dm_temp.rename_col("Switzerland", "Vaud", dim="Country")
        DM_amm["ots"][lev] = dm_temp

    # Cost FXA for ammonia: copy from CH
    for fxa_key in ["cost-matprod", "cost-CC"]:
        dm_temp = DM_amm_ch["fxa"][fxa_key].filter({"Country": ["Switzerland"]}).copy()
        dm_temp.rename_col("Switzerland", "Vaud", dim="Country")
        DM_amm["fxa"][fxa_key] = dm_temp

    # Energy demand FXA for ammonia
    DM_amm["fxa"]["energy-demand-excl-feedstock"] = DM_input[
        "fxa-amm-energy-exclfeedstock"
    ]
    DM_amm["fxa"]["energy-demand-feedstock"] = DM_input["fxa-amm-energy-feedstock"]

    # Calibration for ammonia: all NaN
    for calib_key in ["emissions", "material-production"]:
        dm_temp = (
            DM_amm_ch["calibration"][calib_key]
            .filter({"Country": ["Switzerland"]})
            .copy()
        )
        dm_temp.rename_col("Switzerland", "Vaud", dim="Country")
        dm_temp[...] = np.nan
        DM_amm["calibration"][calib_key] = dm_temp

    # Save intermediate ammonia OTS pickle
    f = os.path.join(current_file_directory, "../data/datamatrix/ammonia_ots.pickle")
    with open(f, "wb") as handle:
        pickle.dump(DM_amm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return DM, DM_amm


if __name__ == "__main__":
    years_ots = create_years_list(1990, 2023, 1)

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(
        current_file_directory, "../data/datamatrix/industry_pre_processing.pickle"
    )
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            "You need to run industry_preprocessing_main_VD.py first"
        )
    with open(filepath, "rb") as f:
        DM_input = pickle.load(f)

    run(DM_input, years_ots)
