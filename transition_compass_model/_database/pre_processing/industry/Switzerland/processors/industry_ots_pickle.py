# import pickle
import os
import pickle
from transition_compass_model.model.common.auxiliary_functions import create_years_list

# import numpy as np


def run(DM_input, years_ots):

    # directories
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    DM = {"ots": dict(), "fxa": dict(), "calibration": dict()}

    DM["ots"]["product-net-import"] = DM_input["product-net-import"]
    DM["ots"]["material-net-import"] = DM_input["material-net-import"]
    DM["ots"]["paperpack"] = DM_input["packaging"]
    DM["ots"]["eol-waste-management"] = DM_input["waste-management"]

    DM["fxa"]["prod"] = DM_input["material-production-not-modelled"]
    DM["fxa"]["demand"] = DM_input["material-demand-wpp"]

    DM["calibration"]["material-production"] = DM_input["calib-matprod"]
    DM["calibration"]["emissions"] = DM_input["calib-emissions"]
    DM["calibration"]["energy-demand"] = DM_input["calib-energy"]
    # ['energy-demand', 'material-production', 'emissions']

    # Load existing DM_industry
    pickle_file = os.path.join(
        current_file_directory, "../../../../data/datamatrix/industry.pickle"
    )
    with open(pickle_file, "rb") as handle:
        DM_industry_current = pickle.load(handle)

    # make other levers same from EU
    other_levers = [
        "material-switch",
        "material-efficiency",
        "technology-development",
        "cc",
        "technology-share",
        "energy-carrier-mix",
        "eol-material-recovery",
    ]
    for l in other_levers:
        dm_temp = DM_industry_current["ots"][l].filter({"Country": ["EU27"]})
        dm_temp.rename_col("EU27", "Switzerland", "Country")
        DM["ots"][l] = dm_temp.copy()

    # for fxa on costs, for the moment put as EU
    other_fxa = ["cost-matprod", "cost-CC"]
    for f in other_fxa:
        dm_temp = DM_industry_current["fxa"][f].filter({"Country": ["EU27"]})
        dm_temp.rename_col("EU27", "Switzerland", "Country")
        DM["fxa"][f] = dm_temp.copy()

    # for calibration, for the moment put material production as missing
    # NOTE: we'll see later if to calibrate this against CHF data on material production (converted to tonne)
    dm_temp = DM_industry_current["calibration"]["material-production"].filter(
        {"Country": ["EU27"]}
    )
    dm_temp.rename_col("EU27", "Switzerland", "Country")
    import numpy as np

    dm_temp[...] = np.nan
    DM["calibration"]["material-production"] = dm_temp.copy()

    # save intermediate
    f = os.path.join(current_file_directory, "../data/datamatrix/industry_ots.pickle")
    with open(f, "wb") as handle:
        pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # make ammonia
    DM_amm = {"ots": dict(), "fxa": dict(), "calibration": dict()}

    DM_amm["ots"]["product-net-import"] = DM_input["fert-product-net-import"]
    DM_amm["ots"]["material-net-import"] = DM_input["amm-material-net-import"]

    # Load existing DM_ammonia
    pickle_file = os.path.join(
        current_file_directory, "../../../../data/datamatrix/ammonia.pickle"
    )
    with open(pickle_file, "rb") as handle:
        DM_amm_current = pickle.load(handle)

    # make other levers same from EU
    other_levers = [
        "material-efficiency",
        "eol-material-recovery",
        "technology-development",
        "cc",
        "energy-carrier-mix",
    ]
    for l in other_levers:
        dm_temp = DM_amm_current["ots"][l].filter({"Country": ["EU27"]})
        dm_temp.rename_col("EU27", "Switzerland", "Country")
        DM_amm["ots"][l] = dm_temp.copy()

    # make other fxa same from EU
    other_fxa = ["cost-matprod", "cost-CC"]
    for f in other_fxa:
        dm_temp = DM_amm_current["fxa"][f].filter({"Country": ["EU27"]})
        dm_temp.rename_col("EU27", "Switzerland", "Country")
        DM_amm["fxa"][f] = dm_temp.copy()

    # for calibration for now put all to nan (we have material production in case for later)
    DM_amm["calibration"]["material-production"] = DM_input[
        "calib-amm-material-production"
    ].copy()
    calibs = ["emissions"]
    for c in calibs:
        dm_temp = DM_amm_current["calibration"][c].filter({"Country": ["EU27"]})
        dm_temp.rename_col("EU27", "Switzerland", "Country")
        dm_temp[...] = np.nan
        DM_amm["calibration"][c] = dm_temp.copy()

    # save intermediate
    f = os.path.join(current_file_directory, "../data/datamatrix/ammonia_ots.pickle")
    with open(f, "wb") as handle:
        pickle.dump(DM_amm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return DM, DM_amm


if __name__ == "__main__":

    # get country ots fts
    years_ots = create_years_list(1990, 2023, 1)

    # load pickle transport ots
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(
        current_file_directory, "../data/datamatrix/industry_pre_processing.pickle"
    )
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            "You need to run the first part of industry_preprocessing_main_CH first"
        )
    with open(filepath, "rb") as f:
        DM_input = pickle.load(f)

    run(DM_input, years_ots)
