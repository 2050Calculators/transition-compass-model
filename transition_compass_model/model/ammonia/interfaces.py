import os
import pickle
import numpy as np


def get_interface(
    current_file_directory, interface, from_sector, to_sector, country_list
):

    if interface.has_link(from_sector=from_sector, to_sector=to_sector):
        DM = interface.get_link(from_sector=from_sector, to_sector=to_sector)
    else:
        if len(interface.list_link()) != 0:
            print("You are missing " + from_sector + " to " + to_sector + " interface")
        filepath = os.path.join(
            current_file_directory,
            "../_database/data/interface/"
            + from_sector
            + "_to_"
            + to_sector
            + ".pickle",
        )
        with open(filepath, "rb") as handle:
            DM = pickle.load(handle)
        if type(DM) is dict:
            for key in DM.keys():
                DM[key].filter({"Country": country_list}, inplace=True)
        else:
            DM.filter({"Country": country_list}, inplace=True)
    return DM


# def variables_for_tpe(dm_material_production_bymat, dm_ind_material_production, dm_energy_demand_bymat,
#                       dm_ind_energy_demand, dm_energy_demand_bymatcarr):

#     # production of chemicals (chem in ind + chem in ammonia)
#     dm_tpe = dm_material_production_bymat.copy()
#     dm_tpe.change_unit('material-production', factor=1e-3, old_unit='kt', new_unit='Mt')
#     dm_tpe.append(dm_ind_material_production.copy(), "Categories1")
#     dm_tpe.group_all("Categories1")
#     dm_tpe.rename_col("material-production", "ind_material-production_chemicals", "Variables")

#     # energy demand chemicals
#     dm_temp = dm_energy_demand_bymat.copy()
#     dm_temp.append(dm_ind_energy_demand.group_all("Categories2", inplace=False), "Categories1")
#     dm_temp.group_all("Categories1")
#     dm_temp.rename_col("energy-demand", "ind_energy-demand_chemicals", "Variables")
#     dm_tpe.append(dm_temp, "Variables")

#     # energy demand chemicals by energy carriers
#     dm_temp = dm_energy_demand_bymatcarr.copy()
#     dm_temp.append(dm_ind_energy_demand, "Categories1")
#     dm_temp.group_all("Categories1")
#     dm_temp.rename_col("energy-demand", "ind_energy-demand_chemicals", "Variables")
#     dm_tpe.append(dm_temp.flatten(), "Variables")

#     # # NOTE: FOR THE MOMENT THE CODE BELOW IS COMMENTED OUT, TO KEEP UNTIL WE FINALIZE THE TPE
#     # # adjust variables' names
#     # DM_cost["material-production_capex"].rename_col_regex("capex", "investment", "Variables")
#     # DM_cost["material-production_capex"].rename_col('ammonia-amm-tech','amm-tech',"Categories1")
#     # DM_cost["CO2-capt-w-cc_capex"].rename_col_regex("capex", "investment_CC", "Variables")
#     # DM_cost["CO2-capt-w-cc_capex"].rename_col_regex("ammonia-amm-tech", "amm-tech", "Categories1")
#     # DM_cost["material-production_opex"].rename_col_regex("opex", "operating-costs", "Variables")
#     # DM_cost["material-production_opex"].rename_col('ammonia-amm-tech','amm-tech',"Categories1")
#     # DM_cost["CO2-capt-w-cc_opex"].rename_col_regex("opex", "operating-costs_CC", "Variables")
#     # DM_cost["CO2-capt-w-cc_opex"].rename_col_regex("ammonia-amm-tech", "amm-tech", "Categories1")
#     # DM_emissions["bygas"] = DM_emissions["bygas"].flatten()
#     # DM_emissions["bygas"].rename_col_regex("_","-","Variables")
#     # variables = DM_material_production["bytech"].col_labels["Categories1"]
#     # variables_new = [rename_tech_fordeepen(i) for i in variables]
#     # for i in range(len(variables)):
#     #     DM_material_production["bytech"].rename_col(variables[i], variables_new[i], dim = "Categories1")
#     # DM_material_production["bymat"].array = DM_material_production["bymat"].array / 1000
#     # DM_material_production["bymat"].units["material-production"] = "Mt"

#     # # dm_tpe
#     # dm_tpe = DM_emissions["bygas"].copy()
#     # dm_tpe.append(DM_energy_demand["bymat"].flatten(), "Variables")
#     # dm_tpe.append(DM_energy_demand["bycarr"].flatten(), "Variables")
#     # dm_tpe.append(DM_cost["CO2-capt-w-cc_capex"].filter({"Variables" : ["investment_CC"]}).flatten(), "Variables")
#     # dm_tpe.append(DM_cost["material-production_capex"].filter({"Variables" : ["investment"]}).flatten(), "Variables")
#     # dm_tpe.append(DM_cost["CO2-capt-w-cc_opex"].filter({"Variables" : ["operating-costs_CC"]}).flatten(), "Variables")
#     # dm_tpe.append(DM_cost["material-production_opex"].filter({"Variables" : ["operating-costs"]}).flatten(), "Variables")
#     # dm_tpe.append(DM_material_production["bymat"].flatten(), "Variables")
#     # variables = dm_tpe.col_labels["Variables"]
#     # for i in variables:
#     #     dm_tpe.rename_col(i, "amm_" + i, "Variables")
#     # variables = ['amm_investment_CC_amm-tech', 'amm_investment_amm-tech',
#     #              'amm_operating-costs_CC_amm-tech', 'amm_operating-costs_amm-tech']
#     # variables_new = ['ind_investment_CC_amm-tech', 'ind_investment_amm-tech',
#     #                  'ind_operating-costs_CC_amm-tech', 'ind_operating-costs_amm-tech']
#     # for i in range(len(variables)):
#     #     dm_tpe.rename_col(variables[i], variables_new[i], "Variables")
#     # dm_tpe.sort("Variables")

#     return dm_tpe


def variables_for_tpe(dm_matprod, dm_emi_bygas):

    dm_out = dm_matprod.flatten()
    dm_out.append(dm_emi_bygas.flatten(), "Variables")

    return dm_out


def ammonia_energy_interface(
    dm_energy_demand_by_carr, cdm_split, cdm_eneff, write_pickle=False
):

    # split between electricity and lighting
    dm_temp = dm_energy_demand_by_carr.filter({"Categories1": ["electricity"]})
    dm_temp.rename_col("electricity", "lighting", "Categories1")
    dm_energy_demand_by_carr.append(dm_temp, "Categories1")

    # reshape
    dm_temp = dm_energy_demand_by_carr.copy()
    dm_temp.drop("Categories1", "lighting")
    for c in dm_temp.col_labels["Variables"]:
        dm_temp.rename_col(c, c + "_process-heat", "Variables")
    dm_temp.deepen("_", "Variables")
    dm_temp.switch_categories_order("Categories1", "Categories2")
    dm_temp[:, :, :, "electricity"] = 0
    dm_temp1 = dm_energy_demand_by_carr.filter(
        {"Categories1": ["lighting", "electricity"]}
    )
    dm_temp1.rename_col(
        ["lighting", "electricity"],
        ["lighting_electricity", "elec_electricity"],
        "Categories1",
    )
    dm_temp1.deepen()
    missing = dm_temp.col_labels["Categories2"].copy()
    missing.remove("electricity")
    for m in missing:
        dm_temp1.add(0, "Categories2", m, dummy=True)
    dm_temp1.sort("Categories2")
    dm_temp.append(dm_temp1, "Categories1")
    dm_temp.sort("Categories1")
    dm_energy_demand_by_carr_reshaped = dm_temp.copy()

    # get useful enery demand
    dm_useful_energy_demand_by_carr = dm_energy_demand_by_carr_reshaped.copy()
    dm_useful_energy_demand_by_carr.array = (
        dm_useful_energy_demand_by_carr.array * cdm_eneff[np.newaxis, np.newaxis, ...]
    )

    # rename energy carriers to match energy ones
    dm_useful_energy_demand_by_carr.rename_col(
        [
            "electricity",
            "gas-bio",
            "gas-ff-natural",
            "hydrogen",
            "liquid-bio",
            "solid-bio",
            "solid-ff-coal",
            "solid-waste",
        ],
        [
            "electricity",
            "biogas",
            "gas",
            "other",
            "renewables",
            "biomass",
            "coal",
            "waste",
        ],
        "Categories2",
    )
    dm_useful_energy_demand_by_carr.groupby(
        {"heating-oil": ["liquid-ff-diesel", "liquid-ff-oil"]},
        "Categories2",
        inplace=True,
    )

    # for the moment add zero for the missing carrier
    # TODO: in pre-processing, you could get constants to get 'district-heating', 'heat-pump' and 'solar'
    # out of electricity (for the moment they have been aggregated to electricity). For 'wood', probably
    # a constant that separates it from biomass, but that's not in JRC. And 'nuclear-fuel' also not in JRC,
    # not sure how we could do itn (probably with some nuclear mix of how electricity is produced by country).
    missing = ["district-heating", "heat-pump", "nuclear-fuel", "solar", "wood"]
    for m in missing:
        dm_useful_energy_demand_by_carr.add(0, "Categories2", m, dummy=True)
    dm_useful_energy_demand_by_carr.sort("Categories2")

    # add zero for 'hot-water' and 'space-heating'
    # TODO: they will either come from bld or we'll do something by tonne in industry
    dm_useful_energy_demand_by_carr.add(0, "Categories1", "hot-water", dummy=True)
    dm_useful_energy_demand_by_carr.add(0, "Categories1", "space-heating", dummy=True)
    dm_useful_energy_demand_by_carr.sort("Categories1")
    dm_useful_energy_demand_by_carr.rename_col(
        "energy-demand", "ind_energy-end-use", "Variables"
    )

    # pass energy efficiency ratios
    cmd_temp = cdm_eneff.copy()
    missing = [
        "district-heating",
        "heat-pump",
        "nuclear-fuel",
        "solar",
    ]  # for these ones assumed to be same of electricity
    for m in missing:
        cmd_temp.add(cmd_temp[..., "electricity"], "Categories2", m)
    cmd_temp.add(
        cmd_temp[..., "solid-bio"], "Categories2", "wood"
    )  # for wood assumed to be the same of biomass
    cmd_temp.rename_col(
        [
            "electricity",
            "gas-bio",
            "gas-ff-natural",
            "hydrogen",
            "liquid-bio",
            "solid-bio",
            "solid-ff-coal",
            "solid-waste",
        ],
        [
            "electricity",
            "biogas",
            "gas",
            "other",
            "renewables",
            "biomass",
            "coal",
            "waste",
        ],
        "Categories2",
    )
    cmd_temp.groupby(
        {"heating-oil": ["liquid-ff-diesel", "liquid-ff-oil"]},
        "Categories2",
        inplace=True,
        aggregation="mean",
    )
    cmd_temp.sort("Categories2")
    cmd_temp.add(
        cmd_temp[:, "process-heat", :], "Categories1", "hot-water"
    )  # put same of process heat for now
    cmd_temp.add(
        cmd_temp[:, "process-heat", :], "Categories1", "space-heating"
    )  # put same of process heat for now
    cmd_temp.sort("Categories1")

    DM_ene = {
        "ind-energy-demand": dm_useful_energy_demand_by_carr,
        "ind-energy-efficiency-const": cmd_temp,
    }

    # of write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../_database/data/interface/ammonia_to_energy.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(DM_ene, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # return
    return DM_ene


def ammonia_emissions_interface(DM_emissions, write_pickle=False):

    # emissions by gas
    dm_ems = DM_emissions["bygas"].copy()
    dm_ems.rename_col("emissions", "ammonia", "Variables")

    # negative emissions
    # TODO: invert here if positive
    dm_temp = DM_emissions["capt_w_cc_bytech"].group_all("Categories1", inplace=False)
    dm_temp.rename_col("CO2-capt-w-cc", "ammonia-captured-emissions_CO2", "Variables")
    dm_temp.deepen()
    dm_temp.add(0, "Categories1", "CH4", dummy=True)
    dm_temp.add(0, "Categories1", "N2O", dummy=True)
    dm_ems.append(dm_temp, "Variables")

    # # adjust variables' names
    # dm_temp = DM_emissions["bygasmat"].flatten().flatten()
    # dm_temp.deepen()
    # dm_temp.rename_col_regex("_","-","Variables")

    # # dm_cli
    # dm_ems = dm_temp.flatten()
    # variables = dm_ems.col_labels["Variables"]
    # for i in variables:
    #     dm_ems.rename_col(i, "amm_" + i, "Variables")
    # dm_ems.sort("Variables")

    # write
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../../_database/data/interface/ammonia_to_emissions.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(dm_ems, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # return
    return dm_ems
