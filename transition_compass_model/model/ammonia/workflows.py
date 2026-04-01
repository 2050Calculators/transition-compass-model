import numpy as np
import re
from transition_compass_model.model.common.auxiliary_functions import cdm_to_dm
from transition_compass_model.model.common.auxiliary_functions import calibration_rates, cost, energy_switch


def product_production(dm_agriculture, dm_import):

    # net import [%] is net import [unit] / demand [unit]
    # production [unit] = demand [unit] - net import [unit]

    # buildings
    dm_netimport_fert = dm_import.copy()
    dm_netimport_fert.array = dm_netimport_fert.array * dm_agriculture.array
    dm_netimport_fert.units["product-net-import"] = dm_agriculture.units[
        "agr_product-demand"
    ]
    dm_prod_fert = dm_agriculture.copy()
    dm_prod_fert.array = dm_prod_fert.array - dm_netimport_fert.array
    dm_prod_fert.rename_col("agr_product-demand", "product-production", "Variables")

    # return
    return dm_prod_fert


def apply_material_decomposition(dm_production_fert, cdm_matdec_fert):

    countries = dm_production_fert.col_labels["Country"]
    years = dm_production_fert.col_labels["Years"]

    # material demand [t] = product production [unit] * material decomposition coefficient [t/unit]

    dm_bld_fert_matdec = cdm_to_dm(cdm_matdec_fert, countries, years)
    dm_bld_fert_matdec.array = (
        dm_production_fert.array[..., np.newaxis] * dm_bld_fert_matdec.array
    )
    dm_bld_fert_matdec.units["material-decomp"] = "t"
    dm_bld_fert_matdec = dm_bld_fert_matdec.filter(
        {"Categories2": ["ammonia"]}
    )  # for now it's 100% ammonia, we'll see later if we'll add other types of fertilizers

    # return
    return dm_bld_fert_matdec


def material_production(
    dm_material_efficiency, dm_material_net_import, dm_material_demand
):

    ######################
    ##### EFFICIENCY #####
    ######################

    dm_temp = dm_material_efficiency.copy()
    dm_material_demand.array = dm_material_demand.array * (
        1 - dm_temp.array[..., np.newaxis]
    )

    ############################
    ##### AGGREGATE DEMAND #####
    ############################

    # get aggregate demand
    dm_matdec_agg = dm_material_demand.group_all(dim="Categories1", inplace=False)
    dm_matdec_agg.change_unit(
        "material-decomp", factor=1e-3, old_unit="t", new_unit="kt"
    )

    ######################
    ##### PRODUCTION #####
    ######################

    # material production [kt] = material demand [kt] * (1 - material net import [%])
    # TODO: add this quantity to the material stock

    # get net import % and make production %
    dm_temp = dm_material_net_import.copy()
    dm_temp.array = 1 - dm_temp.array

    # get material production in units
    dm_material_production_bymat = dm_matdec_agg.copy()
    dm_material_production_bymat.array = dm_matdec_agg.array * dm_temp.array
    dm_material_production_bymat.rename_col(
        col_in="material-decomp", col_out="material-production", dim="Variables"
    )

    # get material net import in kilo tonnes
    dm_material_net_import_kt = dm_matdec_agg.copy()
    dm_material_net_import_kt.array = (
        dm_material_net_import_kt.array * dm_material_net_import.array
    )

    # put together
    DM_material_production = {
        "bymat": dm_material_production_bymat,
        "material-net-import": dm_material_net_import_kt,
    }

    # return
    return DM_material_production


def calibration_material_production(
    DM_cal, dm_material_production_bymat, DM_material_production, years_setting
):

    # get calibration series
    dm_cal_sub = DM_cal["material-production"].copy()
    materials = dm_material_production_bymat.col_labels["Categories1"]
    dm_cal_sub.filter({"Categories1": materials}, inplace=True)

    # get calibration rates
    DM_material_production["calib_rates_bymat"] = calibration_rates(
        dm=dm_material_production_bymat,
        dm_cal=dm_cal_sub,
        calibration_start_year=1990,
        calibration_end_year=2023,
        years_setting=years_setting,
    )

    # do calibration
    dm_material_production_bymat.array = (
        dm_material_production_bymat.array
        * DM_material_production["calib_rates_bymat"].array
    )

    # clean
    del dm_cal_sub, materials

    return


def energy_demand(dm_material_production_bytech, CDM_const):

    # this is by material-technology and carrier

    feedstock = ["excl-feedstock", "feedstock"]
    DM_energy_demand = {}

    for f in feedstock:

        # get constants for energy demand for material production by technology
        cdm_temp = CDM_const["energy_" + f]

        # create dm for energy demand for material production by technology
        dm_energy_demand = cdm_to_dm(
            cdm_temp,
            dm_material_production_bytech.col_labels["Country"],
            dm_material_production_bytech.col_labels["Years"],
        )

        # get energy demand for material production by technology
        dm_temp = dm_material_production_bytech.copy()
        dm_temp.change_unit(
            "material-production", factor=1e-3, old_unit="kt", new_unit="Mt"
        )
        dm_energy_demand.array = dm_energy_demand.array * dm_temp.array[..., np.newaxis]
        dm_energy_demand.units["energy-demand-" + f] = "TWh"
        DM_energy_demand[f + "_bytechcarr"] = dm_energy_demand

    # # get overall energy demand
    # dm_energy_demand_temp = DM_energy_demand["excl-feedstock_bytechcarr"].copy()
    # dm_energy_demand_temp.append(DM_energy_demand["feedstock_bytechcarr"], dim = "Variables")
    # dm_energy_demand_bytechcarr = DM_energy_demand["excl-feedstock_bytechcarr"].copy()
    # dm_energy_demand_bytechcarr.array = np.nansum(dm_energy_demand_temp.array, axis = -3, keepdims= True) # here we are summing feedstock and excluding feedstock together
    # dm_energy_demand_bytechcarr.rename_col(col_in = 'energy-demand-excl-feedstock', col_out = "energy-demand", dim = "Variables")
    # DM_energy_demand["total_bytechcarr"] = dm_energy_demand_bytechcarr.copy()
    # DM_energy_demand["total_bycarr"] = DM_energy_demand["total_bytechcarr"].group_all(dim='Categories1', inplace=False)

    # aggregate energy demand by energy carrier
    DM_energy_demand["excl-feedstock_bycarr"] = DM_energy_demand[
        "excl-feedstock_bytechcarr"
    ].group_all(dim="Categories1", inplace=False)

    # return
    return DM_energy_demand


def calibration_energy_demand(
    DM_cal,
    dm_energy_demand_bycarr,
    dm_energy_demand_bytechcarr,
    DM_energy_demand,
    years_setting,
):

    # this is by material-technology and carrier

    # get calibration rates
    dm_energy_demand_calib_rates_bycarr = calibration_rates(
        dm=dm_energy_demand_bycarr.copy(),
        dm_cal=DM_cal["energy-demand"].copy(),
        calibration_start_year=2000,
        calibration_end_year=2021,
        years_setting=years_setting,
    )

    # FIXME!: before 2000, instead of 1 put the calib rate of 2000 (it's done like this in the KNIME for industry, tbc what to do)
    idx = dm_energy_demand_calib_rates_bycarr.idx
    years_bef2000 = np.array(range(1990, 2000, 1)).tolist()
    for i in years_bef2000:
        dm_energy_demand_calib_rates_bycarr.array[:, idx[i], ...] = (
            dm_energy_demand_calib_rates_bycarr.array[:, idx[2000], ...]
        )

    # store dm_energy_demand_calib_rates_bycarr
    DM_energy_demand["calib_rates_bycarr"] = dm_energy_demand_calib_rates_bycarr

    # do calibration
    dm_energy_demand_bycarr.array = (
        dm_energy_demand_bycarr.array * dm_energy_demand_calib_rates_bycarr.array
    )

    # do calibration for each technology (by applying aggregate calibration rates)
    dm_energy_demand_bytechcarr.array = (
        dm_energy_demand_bytechcarr.array
        * dm_energy_demand_calib_rates_bycarr.array[:, :, :, np.newaxis, :]
    )

    # clean
    del idx, years_bef2000, dm_energy_demand_calib_rates_bycarr

    # return
    return


def technology_development(dm_technology_development, dm_energy_demand_bytechcarr):

    # get energy demand after technology development (tech dev improves energy efficiency)
    dm_energy_demand_bytechcarr.array = dm_energy_demand_bytechcarr.array * (
        1 - dm_technology_development.array[..., np.newaxis]
    )

    # return
    return


def apply_energy_switch(dm_energy_carrier_mix, dm_energy_demand_bytechcarr):

    # this is by material-technology and carrier

    # energy demand for electricity [TWh] = (energy demand [TWh] * electricity share) + energy demand coming from switch to electricity [TWh]

    # get energy mix
    dm_temp = dm_energy_carrier_mix.copy()

    #######################
    ##### ELECTRICITY #####
    #######################

    carrier_in = dm_energy_demand_bytechcarr.col_labels["Categories2"].copy()
    carrier_in.remove("electricity")
    carrier_in.remove("hydrogen")
    energy_switch(
        dm_energy_demand=dm_energy_demand_bytechcarr,
        dm_energy_carrier_mix=dm_temp,
        carrier_in=carrier_in,
        carrier_out="electricity",
        dm_energy_carrier_mix_prefix="to-electricity",
    )

    ####################
    ##### HYDROGEN #####
    ####################

    carrier_in = dm_energy_demand_bytechcarr.col_labels["Categories2"].copy()
    carrier_in.remove("electricity")
    carrier_in.remove("hydrogen")
    energy_switch(
        dm_energy_demand=dm_energy_demand_bytechcarr,
        dm_energy_carrier_mix=dm_temp,
        carrier_in=carrier_in,
        carrier_out="hydrogen",
        dm_energy_carrier_mix_prefix="to-hydrogen",
    )

    ###############
    ##### GAS #####
    ###############

    energy_switch(
        dm_energy_demand=dm_energy_demand_bytechcarr,
        dm_energy_carrier_mix=dm_temp,
        carrier_in=["solid-ff-coal"],
        carrier_out="gas-ff-natural",
        dm_energy_carrier_mix_prefix="solid-to-gas",
    )

    energy_switch(
        dm_energy_demand=dm_energy_demand_bytechcarr,
        dm_energy_carrier_mix=dm_temp,
        carrier_in=["liquid-ff-oil"],
        carrier_out="gas-ff-natural",
        dm_energy_carrier_mix_prefix="liquid-to-gas",
    )

    ###########################
    ##### SYNTHETIC FUELS #####
    ###########################

    # TODO: TO BE DONE

    #####################
    ##### BIO FUELS #####
    #####################

    energy_switch(
        dm_energy_demand=dm_energy_demand_bytechcarr,
        dm_energy_carrier_mix=dm_temp,
        carrier_in=["solid-ff-coal"],
        carrier_out="solid-bio",
        dm_energy_carrier_mix_prefix="to-biomass",
    )

    energy_switch(
        dm_energy_demand=dm_energy_demand_bytechcarr,
        dm_energy_carrier_mix=dm_temp,
        carrier_in=["liquid-ff-oil"],
        carrier_out="liquid-bio",
        dm_energy_carrier_mix_prefix="to-biomass",
    )

    energy_switch(
        dm_energy_demand=dm_energy_demand_bytechcarr,
        dm_energy_carrier_mix=dm_temp,
        carrier_in=["gas-ff-natural"],
        carrier_out="gas-bio",
        dm_energy_carrier_mix_prefix="to-biomass",
    )

    # clean
    del dm_temp, carrier_in

    # return
    return


def add_specific_energy_demands(
    dm_energy_demand_exclfeedstock_bytechcarr,
    dm_energy_demand_feedstock_bytechcarr,
    DM_energy_demand,
    dict_groupby,
):

    # get demand for biomaterial from feedstock
    dm_energy_demand_feedstock_bycarr = dm_energy_demand_feedstock_bytechcarr.group_all(
        "Categories1", inplace=False
    )
    dm_energy_demand_feedstock_bybiomat = dm_energy_demand_feedstock_bycarr.filter(
        {"Categories1": ["solid-bio", "gas-bio", "liquid-bio"]}
    )

    # get total energy demand
    dm_energy_demand_bytechcarr = dm_energy_demand_exclfeedstock_bytechcarr.copy()
    dm_energy_demand_bytechcarr.append(
        dm_energy_demand_feedstock_bytechcarr, "Variables"
    )
    dm_energy_demand_bytechcarr.groupby(
        {"energy-demand": ["energy-demand-excl-feedstock", "energy-demand-feedstock"]},
        "Variables",
        inplace=True,
    )

    # get demand for industrial waste
    dm_energy_demand_bycarr = dm_energy_demand_bytechcarr.group_all(
        "Categories1", inplace=False
    )
    dm_energy_demand_indwaste = dm_energy_demand_bycarr.filter(
        {"Categories1": ["solid-waste"]}
    )

    # get demand for bioenergy solid, bioenergy gas, bioenergy liquid
    dm_energy_demand_bioener_bybiomat = dm_energy_demand_bycarr.filter(
        {"Categories1": ["solid-bio", "gas-bio", "liquid-bio"]}
    )
    dm_energy_demand_bioener_bybiomat.rename_col(
        "energy-demand", "energy-demand_bioenergy", "Variables"
    )
    dm_energy_demand_bioener = dm_energy_demand_bioener_bybiomat.group_all(
        "Categories1", inplace=False
    )

    # get demand by material
    dm_energy_demand_bymatcarr = dm_energy_demand_bytechcarr.groupby(
        dict_groupby, dim="Categories1", aggregation="sum", regex=True, inplace=False
    )
    dm_energy_demand_bymat = dm_energy_demand_bymatcarr.group_all(
        "Categories2", inplace=False
    )

    # get demand by carrier
    dm_energy_demand_bycarr = dm_energy_demand_bymatcarr.group_all(
        "Categories1", inplace=False
    )

    # get energy demand by tech
    dm_energy_demand_bytech = dm_energy_demand_bytechcarr.group_all(
        "Categories2", inplace=False
    )

    # put in DM
    DM_energy_demand["bymatcarr"] = dm_energy_demand_bymatcarr
    DM_energy_demand["feedstock_bybiomat"] = dm_energy_demand_feedstock_bybiomat
    DM_energy_demand["indwaste"] = dm_energy_demand_indwaste
    DM_energy_demand["bioener_bybiomat"] = dm_energy_demand_bioener_bybiomat
    DM_energy_demand["bioener"] = dm_energy_demand_bioener

    DM_energy_demand["bymat"] = dm_energy_demand_bymat
    DM_energy_demand["bycarr"] = dm_energy_demand_bycarr
    DM_energy_demand["bytech"] = dm_energy_demand_bytech

    # clean
    del (
        dm_energy_demand_bymatcarr,
        dm_energy_demand_feedstock_bybiomat,
        dm_energy_demand_indwaste,
        dm_energy_demand_bioener,
        dm_energy_demand_bymat,
        dm_energy_demand_bioener_bybiomat,
        dm_energy_demand_bycarr,
        dm_energy_demand_feedstock_bycarr,
    )

    # return
    return


def emissions(
    cdm_const_emission_factor_process,
    cdm_const_emission_factor,
    dm_energy_demand_exclfeedstock_bytechcarr,
    dm_material_production_bytech,
):

    # get emission factors
    cdm_temp1 = cdm_const_emission_factor_process
    cdm_temp2 = cdm_const_emission_factor

    # emissions = energy demand * emission factor

    # combustion
    dm_emissions_combustion = dm_energy_demand_exclfeedstock_bytechcarr.copy()
    dm_emissions_combustion.rename_col(
        "energy-demand-excl-feedstock", "emissions", "Variables"
    )
    dm_emissions_combustion.units["emissions"] = "Mt"
    dm_emissions_combustion.rename_col("emissions", "emissions_CH4", "Variables")
    dm_emissions_combustion.deepen("_", based_on="Variables")
    arr_temp = dm_emissions_combustion.array
    dm_emissions_combustion.add(arr_temp, "Categories3", "CO2")
    dm_emissions_combustion.add(arr_temp, "Categories3", "N2O")
    dm_emissions_combustion.array = (
        dm_emissions_combustion.array
        * cdm_temp2.array[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    )

    # biogenic total
    bio = ["gas-bio", "liquid-bio", "solid-bio"]
    dm_emissions_combustion_bio = dm_emissions_combustion.filter(
        {"Categories2": bio}, inplace=False
    )
    dm_emissions_combustion_bio.group_all("Categories2")
    dm_emissions_combustion_bio.switch_categories_order("Categories2", "Categories1")
    dm_emissions_combustion_bio.rename_col(
        "emissions", "emissions-biogenic", dim="Variables"
    )

    # process
    dm_emissions_process = dm_material_production_bytech.copy()
    dm_emissions_process.change_unit(
        "material-production", factor=1e-3, old_unit="kt", new_unit="Mt"
    )
    dm_emissions_process.rename_col(
        "material-production", "emissions-process_CH4", "Variables"
    )
    dm_emissions_process.deepen("_", based_on="Variables")
    arr_temp = dm_emissions_process.array
    dm_emissions_process.add(arr_temp, "Categories2", "CO2")
    dm_emissions_process.add(arr_temp, "Categories2", "N2O")
    dm_emissions_process.array = (
        dm_emissions_process.array * cdm_temp1.array[np.newaxis, np.newaxis, ...]
    )

    # total emissions per technology
    dm_emissions_bygastech = dm_emissions_combustion.group_all(
        "Categories2", inplace=False
    )
    dm_emissions_bygastech.append(dm_emissions_process, dim="Variables")
    dm_emissions_bygastech.add(
        np.nansum(dm_emissions_bygastech.array, -3, keepdims=True),
        dim="Variables",
        col_label="emissions-total",
        unit="Mt",
    )
    dm_emissions_bygastech.drop("Variables", ["emissions", "emissions-process"])
    dm_emissions_bygastech.rename_col("emissions-total", "emissions", "Variables")
    dm_emissions_bygastech.switch_categories_order("Categories1", "Categories2")

    # put in dict
    DM_emissions = {
        "combustion": dm_emissions_combustion,
        "process": dm_emissions_process,
        "bygastech": dm_emissions_bygastech,
        "combustion_bio": dm_emissions_combustion_bio,
        "bygastech_beforecc": dm_emissions_bygastech,
    }

    # return
    return DM_emissions


def carbon_capture(
    dm_ots_fts_cc,
    dm_emissions_bygastech,
    dm_emissions_combustion_bio,
    DM_emissions,
    dict_groupby,
):

    # get carbon capture
    dm_temp = dm_ots_fts_cc.copy()

    # subtract carbon captured to total CO2 emissions per technology
    idx = dm_emissions_bygastech.idx
    arr_temp = dm_emissions_bygastech.array[:, :, :, idx["CO2"], :] * (
        1 - dm_temp.array
    )
    dm_emissions_bygastech.add(
        arr_temp[:, :, :, np.newaxis, :], dim="Categories1", col_label="after-cc"
    )

    # get emissions captured with carbon capture
    idx = dm_emissions_bygastech.idx
    arr_temp = (
        dm_emissions_bygastech.array[:, :, :, idx["CO2"], :]
        - dm_emissions_bygastech.array[:, :, :, idx["after-cc"], :]
    )
    dm_emissions_bygastech.add(
        arr_temp[:, :, :, np.newaxis, :], dim="Categories1", col_label="CO2-capt-w-cc"
    )
    dm_emissions_capt_w_cc_bytech = dm_emissions_bygastech.filter(
        {"Categories1": ["CO2-capt-w-cc"]}
    )
    dm_emissions_capt_w_cc_bytech = dm_emissions_capt_w_cc_bytech.flatten()
    dm_emissions_capt_w_cc_bytech.rename_col_regex(
        "CO2-capt-w-cc_", "", dim="Categories1"
    )
    dm_emissions_capt_w_cc_bytech.rename_col("emissions", "CO2-capt-w-cc", "Variables")
    dm_emissions_bygastech.drop("Categories1", "CO2")
    dm_emissions_bygastech.rename_col(
        col_in="after-cc", col_out="CO2", dim="Categories1"
    )
    dm_emissions_bygastech.sort("Categories1")

    # get captured biogenic emissions
    dm_emissions_combustion_bio_capt_w_cc = dm_emissions_combustion_bio.copy()
    idx = dm_emissions_combustion_bio_capt_w_cc.idx
    arr_temp = dm_emissions_combustion_bio_capt_w_cc.array[:, :, :, idx["CO2"], :] * (
        1 - dm_temp.array
    )
    dm_emissions_combustion_bio_capt_w_cc.add(
        arr_temp[:, :, :, np.newaxis, :], dim="Categories1", col_label="after-cc"
    )
    idx = dm_emissions_combustion_bio_capt_w_cc.idx
    arr_temp = (
        dm_emissions_combustion_bio_capt_w_cc.array[:, :, :, idx["CO2"], :]
        - dm_emissions_combustion_bio_capt_w_cc.array[:, :, :, idx["after-cc"], :]
    )
    dm_emissions_combustion_bio_capt_w_cc.add(
        arr_temp[:, :, :, np.newaxis, :], dim="Categories1", col_label="capt-w-cc"
    )
    dm_emissions_combustion_bio_capt_w_cc = (
        dm_emissions_combustion_bio_capt_w_cc.filter({"Categories1": ["capt-w-cc"]})
    )
    dm_emissions_combustion_bio_capt_w_cc = (
        dm_emissions_combustion_bio_capt_w_cc.flatten().flatten()
    )
    dm_emissions_combustion_bio_capt_w_cc.deepen()
    dm_emissions_combustion_bio_capt_w_cc.rename_col(
        col_in="emissions-biogenic_capt-w-cc",
        col_out="emissions-biogenic_CO2-capt-w-cc",
        dim="Variables",
    )

    # get these captured biogenic emissions by material
    dm_emissions_combustion_bio_capt_w_cc = (
        dm_emissions_combustion_bio_capt_w_cc.groupby(
            dict_groupby,
            dim="Categories1",
            aggregation="sum",
            regex=True,
            inplace=False,
        )
    )

    # make negative captured biogenic emissions to supply to the climate module
    dm_emissions_combustion_bio_capt_w_cc_neg_bymat = (
        dm_emissions_combustion_bio_capt_w_cc.copy()
    )
    dm_emissions_combustion_bio_capt_w_cc_neg_bymat.array = (
        dm_emissions_combustion_bio_capt_w_cc_neg_bymat.array * -1
    )
    dm_emissions_combustion_bio_capt_w_cc_neg_bymat.rename_col(
        "emissions-biogenic_CO2-capt-w-cc",
        "emissions-biogenic_CO2-capt-w-cc-negative",
        "Variables",
    )

    # store
    DM_emissions["combustion_bio_capt_w_cc_neg_bymat"] = (
        dm_emissions_combustion_bio_capt_w_cc_neg_bymat
    )
    DM_emissions["capt_w_cc_bytech"] = dm_emissions_capt_w_cc_bytech

    # store also bygas (which is used in calibration if it's done)
    DM_emissions["bygas"] = dm_emissions_bygastech.group_all(
        "Categories2", inplace=False
    )

    # return
    return


def calibration_emissions(
    DM_cal, dm_emissions_bygas, dm_emissions_bygastech, DM_emissions, years_setting
):

    # get calibration rates
    DM_emissions["calib_rates_bygas"] = calibration_rates(
        dm=dm_emissions_bygas,
        dm_cal=DM_cal["emissions"],
        calibration_start_year=2008,
        calibration_end_year=2023,
        years_setting=years_setting,
    )

    # do calibration
    dm_emissions_bygas.array = (
        dm_emissions_bygas.array * DM_emissions["calib_rates_bygas"].array
    )

    # do calibration for each technology (by applying aggregate calibration rates)
    dm_emissions_bygastech.array = (
        dm_emissions_bygastech.array
        * DM_emissions["calib_rates_bygas"].array[:, :, :, :, np.newaxis]
    )

    # return
    return


def compute_costs(
    dm_fxa_cost_matprod,
    dm_fxa_cost_cc,
    dm_material_production_bytech,
    dm_emissions_capt_w_cc_bytech,
):

    ###############################
    ##### MATERIAL PRODUCTION #####
    ###############################

    # subset costs
    dm_cost_sub = dm_fxa_cost_matprod.copy()

    # get material production by technology
    keep = dm_fxa_cost_matprod.col_labels["Categories1"]
    dm_material_techshare_sub = dm_material_production_bytech.filter(
        {"Categories1": keep}
    )
    dm_cost_sub.change_unit(
        "capex-baseyear", factor=1e3, old_unit="EUR/t", new_unit="EUR/kt"
    )
    dm_cost_sub.change_unit(
        "capex-d-factor", factor=1e3, old_unit="num", new_unit="num"
    )

    # get costs
    dm_material_techshare_sub_capex = cost(
        dm_activity=dm_material_techshare_sub, dm_cost=dm_cost_sub, cost_type="capex"
    )

    ######################################
    ##### EMISSIONS CAPTURED WITH CC #####
    ######################################

    # subset cdm
    dm_cost_sub = dm_fxa_cost_cc.copy()

    # get emissions captured with carbon capture
    keep = dm_fxa_cost_cc.col_labels["Categories1"]
    dm_emissions_capt_w_cc_sub = dm_emissions_capt_w_cc_bytech.filter(
        {"Categories1": keep}
    )
    dm_emissions_capt_w_cc_sub.change_unit(
        "CO2-capt-w-cc", factor=1e6, old_unit="Mt", new_unit="t"
    )

    # get costs
    dm_emissions_capt_w_cc_sub_capex = cost(
        dm_activity=dm_emissions_capt_w_cc_sub, dm_cost=dm_cost_sub, cost_type="capex"
    )

    ########################
    ##### PUT TOGETHER #####
    ########################

    DM_cost = {
        "material-production_capex": dm_material_techshare_sub_capex,
        "CO2-capt-w-cc_capex": dm_emissions_capt_w_cc_sub_capex,
    }

    # fix names
    for key in DM_cost.keys():
        cost_type = re.split("_", key)[1]
        activity_type = re.split("_", key)[0]
        DM_cost[key].filter({"Variables": ["unit-cost", cost_type]}, inplace=True)
        DM_cost[key].rename_col(
            "unit-cost", activity_type + "_" + cost_type + "-unit", "Variables"
        )
        DM_cost[key].rename_col(cost_type, activity_type + "_" + cost_type, "Variables")

    # make datamatrixes by material
    keys = list(DM_cost)
    for key in keys:
        materials = [i.split("-")[0] for i in DM_cost[key].col_labels["Categories1"]]
        materials = list(dict.fromkeys(materials))
        dict_groupby = {}
        for m in materials:
            dict_groupby[m] = ".*" + m + ".*"
        DM_cost[key + "_bymat"] = DM_cost[key].groupby(
            dict_groupby,
            dim="Categories1",
            aggregation="sum",
            regex=True,
            inplace=False,
        )

    # return
    return DM_cost
