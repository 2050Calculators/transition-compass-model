import os
import pickle

from transition_compass_model.model.common.auxiliary_functions import create_years_list


def run(DM_input, years_ots):
    # directories
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # make empty DMs
    DM = {"ots": dict(), "fxa": dict(), "calibration": dict(), "constant": dict()}
    DM_amm = {"ots": dict(), "fxa": dict(), "calibration": dict(), "constant": dict()}

    # make ammonia
    DM_amm["ots"]["product-net-import"] = DM_input["product-net-import"].filter(
        {"Categories1": ["fertilizer"]}
    )
    DM_input["product-net-import"].drop("Categories1", "fertilizer")
    DM_amm["ots"]["material-net-import"] = DM_input["material-net-import"].filter(
        {"Categories1": ["ammonia"]}
    )
    DM_input["material-net-import"].drop("Categories1", "ammonia")
    DM_amm["ots"]["material-efficiency"] = DM_input["material-efficiency"].filter(
        {"Categories1": ["ammonia"]}
    )
    DM_input["material-efficiency"].drop("Categories1", "ammonia")
    DM_amm["ots"]["eol-material-recovery"] = DM_input["material-recovery"].filter(
        {"Categories2": ["ammonia"]}
    )
    DM_input["material-recovery"].drop("Categories2", "ammonia")
    DM_amm["ots"]["technology-development"] = DM_input["tech-development"].filter(
        {"Categories1": ["ammonia-tech"]}
    )
    DM_input["tech-development"].drop("Categories1", "ammonia-tech")
    DM_amm["ots"]["cc"] = DM_input["cc"].filter({"Categories1": ["ammonia-tech"]})
    DM_input["cc"].drop("Categories1", "ammonia-tech")
    DM_amm["ots"]["energy-carrier-mix"] = DM_input["energy-switch"].filter(
        {"Categories1": ["ammonia-tech"]}
    )
    DM_input["energy-switch"].drop("Categories1", "ammonia-tech")

    DM_amm["fxa"]["cost-matprod"] = DM_input["costs"].filter(
        {"Categories1": ["ammonia-tech"]}
    )
    DM_input["costs"].drop("Categories1", "ammonia-tech")
    DM_amm["fxa"]["cost-CC"] = DM_input["costs-cc"].filter(
        {"Categories1": ["ammonia-tech"]}
    )
    DM_input["costs-cc"].drop("Categories1", "ammonia-tech")
    DM_amm["fxa"]["energy-demand-excl-feedstock"] = DM_input[
        "fxa-energy-exclfeedstock"
    ].filter({"Categories1": ["ammonia-tech"]})
    DM_input["fxa-energy-exclfeedstock"].drop("Categories1", "ammonia-tech")
    DM_amm["fxa"]["energy-demand-feedstock"] = DM_input["fxa-energy-feedstock"].filter(
        {"Categories1": ["ammonia-tech"]}
    )
    DM_input["fxa-energy-feedstock"].drop("Categories1", "ammonia-tech")

    DM_amm["calibration"]["emissions"] = DM_input["calib-emissions-ammonia"].copy()
    DM_amm["calibration"]["material-production"] = DM_input[
        "calib-material-production-ammonia"
    ].copy()

    DM_amm["constant"]["energy_excl-feedstock_eleclight-split"] = DM_input[
        "const-energy-exclfeedstock-eleclightsplit"
    ].copy()
    DM_amm["constant"]["energy_efficiency"] = DM_input["const-energy-efficiency"].copy()
    DM_amm["constant"]["material-decomposition_fertilizer"] = DM_input[
        "const-material-decomp-fert"
    ].copy()
    products = ["floor", "infra", "pack", "domapp", "electronics"]
    for p in products:
        DM_amm["constant"]["material-decomposition_" + p] = DM_input[
            "const-material-decomp-" + p
        ].filter({"Categories2": ["ammonia"]})
        DM_input["const-material-decomp-" + p].drop("Categories2", "ammonia")
    DM_amm["constant"]["material-decomposition_veh"] = DM_input[
        "const-material-decomp-veh"
    ].filter({"Categories3": ["ammonia"]})
    DM_input["const-material-decomp-veh"].drop("Categories3", "ammonia")
    DM_amm["constant"]["material-decomposition_bat"] = DM_input[
        "const-material-decomp-batteries"
    ].filter({"Categories3": ["ammonia"]})
    DM_input["const-material-decomp-batteries"].drop("Categories3", "ammonia")
    DM_amm["constant"]["emission-factor-process"] = DM_input[
        "const-emission-process"
    ].filter({"Categories1": ["ammonia-tech"]})
    DM_input["const-emission-process"].drop("Categories1", "ammonia-tech")
    DM_amm["constant"]["emission-factor"] = DM_input["const-emission-combustion"].copy()

    # make industry
    DM["ots"]["material-switch"] = DM_input["material-switch"].copy()
    DM["ots"]["material-efficiency"] = DM_input["material-efficiency"].copy()
    DM["ots"]["technology-development"] = DM_input["tech-development"].copy()
    DM["ots"]["cc"] = DM_input["cc"].copy()
    DM["ots"]["technology-share"] = DM_input["tech-share"].copy()
    DM["ots"]["material-net-import"] = DM_input["material-net-import"].copy()
    DM["ots"]["product-net-import"] = DM_input["product-net-import"].copy()
    DM["ots"]["energy-carrier-mix"] = DM_input["energy-switch"].copy()
    DM["ots"]["eol-waste-management"] = DM_input["waste-management"].copy()
    DM["ots"]["eol-material-recovery"] = DM_input["material-recovery"].copy()
    DM["ots"]["paperpack"] = DM_input["packaging"].copy()

    DM["fxa"]["cost-matprod"] = DM_input["costs"].copy()
    DM["fxa"]["cost-CC"] = DM_input["costs-cc"].copy()
    DM["fxa"]["prod"] = DM_input["material-production-not-modelled"].copy()
    DM["fxa"]["demand"] = DM_input["material-demand-wpp"].copy()
    DM["fxa"]["energy-demand-excl-feedstock"] = DM_input[
        "fxa-energy-exclfeedstock"
    ].copy()
    DM["fxa"]["energy-demand-feedstock"] = DM_input["fxa-energy-feedstock"].copy()

    DM["calibration"]["emissions"] = DM_input["calib-emissions"]
    DM["calibration"]["energy-demand"] = DM_input["calib-energy"]
    DM["calibration"]["material-production"] = DM_input["calib-material-production"]

    DM["constant"]["emission-factor"] = DM_input["const-emission-combustion"].copy()
    DM["constant"]["emission-factor-process"] = DM_input[
        "const-emission-process"
    ].copy()
    DM["constant"]["energy_excl-feedstock_eleclight-split"] = DM_input[
        "const-energy-exclfeedstock-eleclightsplit"
    ].copy()
    DM["constant"]["energy_efficiency"] = DM_input["const-energy-efficiency"].copy()
    DM["constant"]["material-decomposition_pack"] = DM_input[
        "const-material-decomp-pack"
    ].copy()
    DM["constant"]["material-decomposition_veh"] = DM_input[
        "const-material-decomp-veh"
    ].copy()
    DM["constant"]["material-decomposition_bat"] = DM_input[
        "const-material-decomp-batteries"
    ].copy()
    DM["constant"]["material-decomposition_infra"] = DM_input[
        "const-material-decomp-infra"
    ].copy()
    DM["constant"]["material-decomposition_floor"] = DM_input[
        "const-material-decomp-floor"
    ].copy()
    # DM["constant"]["material-decomposition_pipe"] = DM_input["const-material-decomp-dhgpipes"].copy() # omit for the moment
    DM["constant"]["material-decomposition_domapp"] = DM_input[
        "const-material-decomp-domapp"
    ].copy()
    DM["constant"]["material-decomposition_electronics"] = DM_input[
        "const-material-decomp-electronics"
    ].copy()
    DM["constant"]["material-switch"] = DM_input["const-material-switch"].copy()

    # get EU27 only
    for key in DM["ots"].keys():
        DM["ots"][key].filter({"Country": ["EU27"]}, inplace=True)
    for key in DM["fxa"].keys():
        DM["fxa"][key].filter({"Country": ["EU27"]}, inplace=True)
    for key in DM["calibration"].keys():
        DM["calibration"][key].filter({"Country": ["EU27"]}, inplace=True)

    for key in DM_amm["ots"].keys():
        DM_amm["ots"][key].filter({"Country": ["EU27"]}, inplace=True)
    for key in DM_amm["fxa"].keys():
        DM_amm["fxa"][key].filter({"Country": ["EU27"]}, inplace=True)
    for key in DM_amm["calibration"].keys():
        DM_amm["calibration"][key].filter({"Country": ["EU27"]}, inplace=True)

    # save intermediate
    f = os.path.join(current_file_directory, "../data/datamatrix/industry_ots.pickle")
    with open(f, "wb") as handle:
        pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)
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
            "You need to run the first part of industry_preprocessing_main_EU first"
        )
    with open(filepath, "rb") as f:
        DM_input = pickle.load(f)

    run(DM_input, years_ots)
