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


# def variables_for_tpe(dm_cost_material_production_capex, dm_cost_CO2_capt_w_cc_capex,
#                       dm_emissions_bygas,
#                       dm_material_production_bytech, dm_material_production_bymat,
#                       dm_energy_demand_bymat, dm_energy_demand_bymatcarr,
#                       dm_energy_demand_bioener
#                       ):

#     # TODO: recheck variables' names compared to TPE (while running TPE)

#     # adjust variables' names
#     dm_cost_material_production_capex.rename_col_regex("material-production_capex", "investment", "Variables")
#     dm_cost_CO2_capt_w_cc_capex.rename_col_regex("CO2-capt-w-cc_capex", "investment_CC", "Variables")
#     dm_emissions_bygas = dm_emissions_bygas.flatten()
#     dm_emissions_bygas.rename_col_regex("_","-","Variables")
#     # variables = dm_material_production_bytech.col_labels["Categories1"]
#     # variables_new = [rename_tech_fordeepen(i) for i in variables]
#     # for i in range(len(variables)):
#     #     dm_material_production_bytech.rename_col(variables[i], variables_new[i], dim = "Categories1")

#     # convert kt to mt
#     dm_material_production_bytech.change_unit('material-production', factor=1e-3, old_unit='kt', new_unit='Mt')
#     dm_material_production_bymat.change_unit('material-production', factor=1e-3, old_unit='kt', new_unit='Mt')

#     # material production total (chemicals done in ammonia)
#     dm_mat_prod = dm_material_production_bymat.filter({"Categories1" : ["aluminium","cement","copper",
#                                                                       "glass","lime","paper","steel"]})
#     dm_mat_prod.rename_col('material-production', 'ind_material-production', 'Variables')

#     # energy demand by material
#     dm_energy_by_mat = dm_energy_demand_bymat.copy()
#     dm_energy_by_mat.rename_col('energy-demand', 'ind_energy-demand', 'Variables')

#     # emissions (done in emissions)

#     # production technologies (aluminium, cement, paper, steel)
#     dm_prod_tech = dm_material_production_bytech.copy()
#     # dm_temp.groupby({'aluminium_sec': 'aluminium_sec.*',
#     #                  'steel_scrap-EAF': 'steel_scrap-EAF-precons.*|teel_sec-postcons'},
#     #                 dim='Categories1', regex=True, inplace=True)
#     # dm_prod_tech = dm_temp.filter({"Categories1" : ['aluminium_prim', 'aluminium_sec',
#     #                                                 'cement_dry-kiln', 'cement_geopolym',
#     #                                                 'cement_wet-kiln','paper_recycled',
#     #                                                 'paper_woodpulp', 'steel_BF-BOF',
#     #                                                 'steel_hisarna', 'steel_hydrog-DRI',
#     #                                                 'steel_scrap-EAF']})
#     dm_prod_tech.rename_col('material-production', 'ind_material-production', 'Variables')

#     # energy demand for material production by ener carrier (aluminium, cement, chem, glass, lime, paper, steel)
#     dm_energy_by_carrier = dm_energy_demand_bymatcarr.filter({"Categories1": ['aluminium', 'cement', 'glass',
#                                                                               'lime', 'paper', 'steel']})
#     dm_energy_by_carrier.rename_col('energy-demand', 'ind_energy-demand', 'Variables')


#     # dm_tpe
#     # TODO: check if you need to pass emissions in MtCO2eq (rather than in Mt, so if CH4 and N2O should be weighted up)
#     dm_tpe = dm_emissions_bygas.copy()
#     dm_tpe.append(dm_energy_by_mat.flatten(), "Variables")
#     dm_tpe.append(dm_energy_by_carrier.flatten().flatten(), "Variables")
#     dm_tpe.append(dm_energy_demand_bioener, "Variables")
#     dm_tpe.append(dm_cost_CO2_capt_w_cc_capex.flatten(), "Variables")
#     dm_tpe.append(dm_cost_material_production_capex.flatten(), "Variables")
#     dm_tpe.append(dm_mat_prod.flatten(), "Variables")
#     dm_tpe.append(dm_prod_tech.flatten(), "Variables")

#     # return
#     return dm_tpe


def variables_for_tpe(dm_matprod, dm_emi_bygas):

    dm_out = dm_matprod.flatten()
    dm_out.append(dm_emi_bygas.flatten(), "Variables")

    return dm_out


def industry_agriculture_interface(
    DM_material_production, DM_energy_demand, write_pickle=False
):

    DM_agr = {}

    # natfibers
    dm_temp = DM_material_production["natfiber"].copy()
    dm_temp.rename_col("material-decomp", "ind_dem", "Variables")
    DM_agr["natfibers"] = dm_temp.flatten()

    # bioenergy
    dm_temp = DM_energy_demand["bioener_bybiomat"].copy()
    dm_temp.rename_col("energy-demand_bioenergy", "ind_bioenergy", "Variables")
    dm_temp = dm_temp.filter({"Categories1": ["gas-bio", "solid-bio"]})
    DM_agr["bioenergy"] = dm_temp

    # biomaterial
    dm_temp = DM_energy_demand["feedstock_bybiomat"].copy()
    dm_temp.rename_col("energy-demand-feedstock", "ind_biomaterial", "Variables")
    dm_temp = dm_temp.filter({"Categories1": ["gas-bio"]})
    DM_agr["biomaterial"] = dm_temp

    # if write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../../_database/data/interface/industry_to_agriculture.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(DM_agr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return DM_agr


def industry_ammonia_interface(
    DM_material_production, DM_energy_demand, write_pickle=False
):

    dm_amm_matprod = DM_material_production["bymat"].filter({"Categories1": ["chem"]})
    dm_amm_endem = DM_energy_demand["bymatcarr"].filter({"Categories1": ["chem"]})
    DM_amm = {"material-production": dm_amm_matprod, "energy-demand": dm_amm_endem}

    # of write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../../_database/data/interface/industry_to_ammonia.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(DM_amm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return DM_amm


def industry_landuse_interface(
    DM_material_production, DM_energy_demand, write_pickle=False
):

    DM_lus = {}

    # timber
    dm_timber = DM_material_production["bymat"].filter({"Categories1": ["timber"]})
    dm_timber.rename_col("material-production", "ind_material-production", "Variables")
    dm_timber = dm_timber.flatten()
    DM_lus["timber"] = dm_timber

    # woodpuplp
    dm_woodpulp = DM_material_production["bytech"].filter(
        {"Categories1": ["pulp-tech"]}
    )
    dm_woodpulp = dm_woodpulp.flatten()
    dm_woodpulp.rename_col(
        "material-production_pulp-tech",
        "ind_material-production_paper_woodpulp",
        "Variables",
    )
    DM_lus["woodpulp"] = dm_woodpulp.copy()

    # biomaterial solid bio
    dm_temp = DM_energy_demand["feedstock_bybiomat"].copy()
    dm_temp.rename_col("energy-demand-feedstock", "ind_biomaterial", "Variables")
    dm_temp = dm_temp.filter({"Categories1": ["solid-bio"]})
    DM_lus["biomaterial"] = dm_temp.flatten()

    # of write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../../_database/data/interface/industry_to_land-use.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(DM_lus, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # return
    return DM_lus


def industry_energy_interface(
    dm_energy_demand_by_carr, cdm_split, cdm_eneff, write_pickle=False
):

    # TODO: here give to energy only electricity (so total is same of excluding feedstock, as there
    # is no feedstock for electricity). For the other energy carriers, fossil fuels should be
    # given to refinery, wood to forestry, and bio to agriculture possibly

    # split between electricity and lighting
    dm_temp = dm_energy_demand_by_carr.filter({"Categories1": ["electricity"]})
    dm_temp.add(dm_temp.array, "Categories1", "lighting")
    dm_temp.array = dm_temp[...] * cdm_split[np.newaxis, np.newaxis, ...]
    dm_energy_demand_by_carr.drop("Categories1", "electricity")
    dm_energy_demand_by_carr.append(dm_temp, "Categories1")
    dm_energy_demand_by_carr.sort("Categories1")

    # reshape
    dm_temp = dm_energy_demand_by_carr.copy()
    dm_temp.drop("Categories1", "lighting")
    for c in dm_temp.col_labels["Variables"]:
        dm_temp.rename_col(c, c + "_process-heat", "Variables")
    dm_temp.deepen("_", "Variables")
    dm_temp.switch_categories_order("Categories1", "Categories2")
    dm_temp[:, :, :, "electricity"] = (
        0  # note: to understand why we said zero electricity for process heat
    )
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
            "../../_database/data/interface/industry_to_energy.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(DM_ene, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # return
    return DM_ene


def industry_refinery_interface(DM_energy_demand, write_pickle=False):

    # dm_elc
    dm_ref = DM_energy_demand["bycarr"].filter(
        {
            "Categories1": [
                "liquid-ff-diesel",
                "liquid-ff-oil",
                "gas-ff-natural",
                "solid-ff-coal",
            ]
        }
    )
    dm_ref.rename_col("energy-demand", "ind_energy-demand", "Variables")
    dm_ref.rename_col_regex("liquid-ff-oil_", "", dim="Categories1")
    dm_ref.sort("Categories1")

    # of write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../_database/data/interface/industry_to_refinery.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(dm_ref, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # return
    return dm_ref


def industry_district_heating_interface(DM_energy_demand, write_pickle=False):

    # FIXME: fix dummy values for dh with real values
    dm_dh = DM_energy_demand["bycarr"].filter({"Categories1": ["electricity"]})
    dm_dh = dm_dh.flatten()
    dm_dh.add(
        0,
        dim="Variables",
        col_label="dhg_energy-demand_contribution_heat-waste",
        unit="TWh/year",
        dummy=True,
    )
    dm_dh.drop("Variables", "energy-demand_electricity")

    # of write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../_database/data/interface/industry_to_district-heating.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(dm_dh, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # return
    return dm_dh


def industry_emissions_interface(DM_emissions, write_pickle=False):

    # emissions by gas
    dm_ems = DM_emissions["bygas"].copy()
    dm_ems.rename_col("emissions", "industry", "Variables")

    # negative emissions
    # TODO: invert here if positive
    dm_temp = DM_emissions["capt_w_cc_bytech"].group_all("Categories1", inplace=False)
    dm_temp.rename_col("CO2-capt-w-cc", "industry-captured-emissions_CO2", "Variables")
    dm_temp.deepen()
    dm_temp.add(0, "Categories1", "CH4", dummy=True)
    dm_temp.add(0, "Categories1", "N2O", dummy=True)
    dm_ems.append(dm_temp, "Variables")

    # # adjust variables' names
    # dm_temp = DM_emissions["bygasmat"].flatten().flatten()
    # dm_temp.deepen()
    # dm_temp.rename_col_regex("_","-","Variables")
    # DM_emissions["combustion_bio_capt_w_cc_neg_bymat"].rename_col("emissions-biogenic_CO2-capt-w-cc-negative","emissions-CO2_biogenic","Variables")
    # dm_temp1 = DM_emissions["combustion_bio_capt_w_cc_neg_bymat"].flatten()
    # # TODO: re-check what it's doing here above. In principle, it would make sense
    # # to gather negative emissions, to then be put together with other emissions below ...
    # # but why only gater biogenic negative emissions, and not all negative emissions?
    # # to be re-checked

    # # dm_ems
    # dm_ems = dm_temp.flatten()
    # dm_ems.append(dm_temp1, "Variables")
    # variables = dm_ems.col_labels["Variables"]
    # for i in variables:
    #     dm_ems.rename_col(i, "ind_" + i, "Variables")
    # dm_ems.sort("Variables")

    # of write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../../_database/data/interface/industry_to_emissions.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(dm_ems, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # return
    return dm_ems


def industry_water_inferface(
    DM_energy_demand, DM_material_production, write_pickle=False
):

    # dm_water
    dm_water = (
        DM_energy_demand["bycarr"]
        .filter(
            {
                "Categories1": [
                    "electricity",
                    "gas-ff-natural",
                    "hydrogen",
                    "liquid-ff-oil",
                    "solid-ff-coal",
                ]
            }
        )
        .flatten()
    )
    dm_water.append(
        DM_material_production["bytech"]
        .filter(
            {
                "Categories1": [
                    "aluminium-prim",
                    "aluminium-sec",
                    "cement-dry-kiln",
                    "cement-geopolym",
                    "cement-wet-kiln",
                    "chem-chem-tech",
                    "copper-tech",
                    "glass-glass",
                    "lime-lime",
                    "paper-recycled",
                    "paper-woodpulp",
                    "steel-BF-BOF",
                    "steel-hisarna",
                    "steel-hydrog-DRI",
                    "steel-scrap-EAF",
                ]
            }
        )
        .flatten(),
        "Variables",
    )
    variables = dm_water.col_labels["Variables"]
    for i in variables:
        dm_water.rename_col(i, "ind_" + i, "Variables")
    dm_water.sort("Variables")

    # of write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../_database/data/interface/industry_to_water.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(dm_water, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # return
    return dm_water


def industry_ccus_interface(DM_emissions, write_xls=False):

    # adjust variables' names
    DM_emissions["capt_w_cc_bytech"].rename_col(
        "CO2-capt-w-cc", "ind_CO2-emissions-CC", "Variables"
    )

    # dm_ccus
    dm_ccus = (
        DM_emissions["capt_w_cc_bytech"]
        .filter(
            {
                "Categories1": [
                    "aluminium_prim",
                    "aluminium_sec",
                    "cement_dry-kiln",
                    "cement_geopolym",
                    "cement_wet-kiln",
                    "chem_chem-tech",
                    "lime_lime",
                    "paper_recycled",
                    "paper_woodpulp",
                    "steel_BF-BOF",
                    "steel_hisarna",
                    "steel_hydrog-DRI",
                    "steel_scrap-EAF",
                ]
            }
        )
        .flatten()
    )
    dm_ccus.sort("Variables")

    # df_ccus
    if write_xls is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        df_ccus = dm_ccus.write_df()
        df_ccus.to_excel(
            current_file_directory
            + "/../_database/data/xls/"
            + "industry-to-ccus.xlsx",
            index=False,
        )

    # return
    return dm_ccus


# def industry_minerals_interface(DM_material_production, DM_production, DM_ots_fts, write_xls = False):

#     DM_ind = {}

#     # aluminium pack
#     dm_alupack = DM_production["pack"].filter({"Categories1" : ["aluminium-pack"]})
#     DM_ind["aluminium-pack"] = dm_alupack.flatten()

#     # material production
#     dm_matprod = DM_material_production["bymat"].filter({"Categories1": ["timber", 'glass', 'cement']})
#     dm_paper_woodpulp = DM_material_production["bytech"].filter({"Categories1": ['paper_woodpulp']})
#     dm_matprod.append(dm_paper_woodpulp, "Categories1")
#     dm_matprod.rename_col("material-production", "ind_material-production", "Variables")
#     DM_ind["material-production"] = dm_matprod.flatten()

#     # technology development
#     dm_techdev = DM_ots_fts['technology-development'].filter(
#         {"Categories1" : ['aluminium-prim', 'aluminium-sec','copper-tech',
#                           'steel-BF-BOF', 'steel-hisarna', 'steel-hydrog-DRI',
#                           'steel-scrap-EAF']})
#     variables = dm_techdev.col_labels["Categories1"]
#     variables_new = ['aluminium_primary', 'aluminium_secondary','copper_secondary',
#                       'steel_BF-BOF', 'steel_hisarna', 'steel_hydrog-DRI',
#                       'steel_scrap-EAF']
#     for i in range(len(variables)):
#         dm_techdev.rename_col(variables[i], variables_new[i], dim = "Categories1")
#     dm_techdev.rename_col("ind_technology-development","ind_proportion","Variables")
#     DM_ind["technology-development"] = dm_techdev.flatten()

#     # material efficiency
#     DM_ind["material-efficiency"] = DM_ots_fts['material-efficiency'].filter(
#         {"Variables" : ['ind_material-efficiency'],
#          "Categories1" : ['aluminium','copper','steel']})

#     # material switch
#     dm_temp = DM_ots_fts['material-switch'].filter(
#         {"Categories1" : ['build-steel-to-timber', 'cars-steel-to-chem',
#                           'trucks-steel-to-aluminium', 'trucks-steel-to-chem']}).flatten()
#     dm_temp.rename_col_regex("material-switch_","material-switch-","Variables")
#     DM_ind["material-switch"] = dm_temp

#     # product net import
#     dm_temp = DM_ots_fts["product-net-import"].filter(
#         {"Variables" : ["ind_product-net-import"],
#          "Categories1" : ['cars-EV', 'cars-FCV', 'cars-ICE', 'computer', 'dishwasher', 'dryer',
#                           'freezer', 'fridge','phone','planes','rail','road', 'ships', 'trains',
#                           'trolley-cables', 'trucks-EV', 'trucks-FCV', 'trucks-ICE', 'tv',
#                           'wmachine','new-dhg-pipe']})
#     dm_temp.rename_col_regex("cars","LDV","Categories1")
#     dm_temp.rename_col_regex("trucks","HDVL","Categories1")
#     dm_temp.rename_col("computer","electronics-computer","Categories1")
#     dm_temp.rename_col("phone","electronics-phone","Categories1")
#     dm_temp.rename_col("tv","electronics-tv","Categories1")
#     dm_temp.rename_col("dishwasher","dom-appliance-dishwasher","Categories1")
#     dm_temp.rename_col("dryer","dom-appliance-dryer","Categories1")
#     dm_temp.rename_col("freezer","dom-appliance-freezer","Categories1")
#     dm_temp.rename_col("fridge","dom-appliance-fridge","Categories1")
#     dm_temp.rename_col("wmachine","dom-appliance-wmachine","Categories1")
#     dm_temp.rename_col("new-dhg-pipe","infra-pipe","Categories1")
#     dm_temp.rename_col("rail","infra-rail","Categories1")
#     dm_temp.rename_col("road","infra-road","Categories1")
#     dm_temp.rename_col("trolley-cables","infra-trolley-cables","Categories1")
#     dm_temp.rename_col("planes","other-planes","Categories1")
#     dm_temp.rename_col("ships","other-ships","Categories1")
#     dm_temp.rename_col("trains","other-trains","Categories1")
#     dm_temp.rename_col_regex('FCV', 'FCEV', 'Categories1')
#     dm_temp.sort("Categories1")
#     DM_ind["product-net-import"] = dm_temp.flatten()

#     # df_min
#     if write_xls is True:

#         current_file_directory = os.path.dirname(os.path.abspath(__file__))

#         dm_min = DM_ind['aluminium-pack']
#         dm_min.append(DM_ind['material-production'], "Variables")
#         dm_min.append(DM_ind['technology-development'], "Variables")
#         dm_min.append(DM_ind['material-efficiency'].flatten(), "Variables")
#         dm_min.append(DM_ind['material-switch'], "Variables")
#         dm_min.append(DM_ind['product-net-import'], "Variables")
#         dm_min.sort("Variables")

#         df_min = dm_min.write_df()
#         df_min.to_excel(current_file_directory + "/../_database/data/xls/" + 'industry-to-minerals.xlsx', index=False)

#     # return
#     return DM_ind


def industry_minerals_interface(
    DM_production, veh_eol_to_recycling, write_pickle=False
):

    DM_ind = {"production": DM_production, "veh-to-recycling": veh_eol_to_recycling}

    # of write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../_database/data/interface/industry_to_minerals.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(DM_ind, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # return
    return DM_ind


def industry_employment_interface(
    DM_material_demand,
    DM_energy_demand,
    DM_material_production,
    DM_cost,
    DM_ots_fts,
    write_xls=False,
):

    # get material demand for appliances
    DM_material_demand["appliances"] = (
        DM_material_demand["material-demand"]
        .filter(
            {
                "Categories1": [
                    "computer",
                    "dishwasher",
                    "dryer",
                    "freezer",
                    "fridge",
                    "tv",
                ]
            }
        )
        .group_all("Categories1", inplace=False)
    )
    DM_material_demand["appliances"].rename_col(
        "material-decomposition", "material-demand_appliances", "Variables"
    )

    # get material demand for transport
    DM_material_demand["transport"] = (
        DM_material_demand["material-demand"]
        .filter(
            {
                "Categories1": [
                    "cars-EV",
                    "cars-FCV",
                    "cars-ICE",
                    "trucks-EV",
                    "trucks-FCV",
                    "trucks-ICE",
                    "planes",
                    "ships",
                    "trains",
                ]
            }
        )
        .group_all("Categories1", inplace=False)
    )
    DM_material_demand["transport"].rename_col(
        "material-decomposition", "material-demand_transport", "Variables"
    )

    # get material demand for construction
    DM_material_demand["construction"] = (
        DM_material_demand["material-demand"]
        .filter(
            {
                "Categories1": [
                    "floor-area-new-non-residential",
                    "floor-area-new-residential",
                    "floor-area-reno-non-residential",
                    "floor-area-reno-residential",
                    "rail",
                    "road",
                    "trolley-cables",
                ]
            }
        )
        .group_all("Categories1", inplace=False)
    )
    DM_material_demand["construction"].rename_col(
        "material-decomposition", "material-demand_construction", "Variables"
    )

    # dm_emp
    dm_emp = DM_material_demand["appliances"].flatten()
    dm_emp.append(DM_material_demand["transport"].flatten(), "Variables")
    dm_emp.append(DM_material_demand["construction"].flatten(), "Variables")
    dm_emp.append(DM_energy_demand["bymat"].flatten(), "Variables")
    dm_emp.append(DM_energy_demand["bymatcarr"].flatten().flatten(), "Variables")
    dm_emp.append(
        DM_material_production["bymat"]
        .filter({"Categories1": DM_energy_demand["bymat"].col_labels["Categories1"]})
        .flatten(),
        "Variables",
    )
    dm_emp.append(DM_cost["material-production_capex"].flatten(), "Variables")
    dm_emp.append(DM_cost["material-production_opex"].flatten(), "Variables")
    dm_emp.append(DM_cost["CO2-capt-w-cc_capex"].flatten(), "Variables")
    dm_emp.append(DM_cost["CO2-capt-w-cc_opex"].flatten(), "Variables")
    variables = dm_emp.col_labels["Variables"]
    for i in variables:
        dm_emp.rename_col(i, "ind_" + i, "Variables")
    dm_emp.append(
        DM_ots_fts["material-net-import"]
        .filter(
            {
                "Categories1": [
                    "aluminium",
                    "cement",
                    "chem",
                    "copper",
                    "glass",
                    "lime",
                    "paper",
                    "steel",
                    "timber",
                ],
            }
        )
        .flatten(),
        "Variables",
    )
    dm_emp.sort("Variables")

    # df_emp
    if write_xls is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        df_emp = dm_emp.write_df()
        df_emp.to_excel(
            current_file_directory
            + "/../_database/data/xls/"
            + "industry-to-employment.xlsx",
            index=False,
        )

    # return
    return dm_emp


def industry_airpollution_interface(
    DM_material_production, DM_energy_demand, write_xls=False
):

    # dm_airpoll
    dm_airpoll = DM_material_production["bytech"].flatten()
    dm_airpoll.append(DM_energy_demand["bymatcarr"].flatten().flatten(), "Variables")
    variables = dm_airpoll.col_labels["Variables"]
    for i in variables:
        dm_airpoll.rename_col(i, "ind_" + i, "Variables")
    dm_airpoll.sort("Variables")

    # write
    if write_xls is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        df_airpoll = dm_airpoll.write_df()
        df_airpoll.to_excel(
            current_file_directory
            + "/../_database/data/xls/"
            + "industry-to-air_pollution.xlsx",
            index=False,
        )

    # return
    return dm_airpoll


def industry_forestry_interface(
    dm_material_demand, dm_fxa_demand_wwp, write_pickle=False
):

    # get pulp and timber
    dm_temp = dm_material_demand.group_all("Categories1", inplace=False)
    dm_temp = dm_temp.filter({"Categories1": ["paper", "timber"]})
    dm_temp[..., "paper"] = (
        dm_temp[..., "paper"] * 0.95
    )  # for the moment I assume this coefficient to get pulp
    dm_temp.rename_col("paper", "pulp", "Categories1")

    # get other-industrial (demand)
    dm_temp1 = dm_fxa_demand_wwp.copy()
    dm_temp1.rename_col("material-demand", "material-decomp", "Variables")
    dm_temp.append(dm_temp1, "Categories1")
    dm_temp.rename_col("wwp", "other-industrial", "Categories1")
    dm_temp.rename_col("material-decomp", "ind_wood", "Variables")
    dm_temp.sort("Categories1")

    # of write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../../_database/data/interface/industry_to_forestry.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(dm_temp, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm_temp


def industry_lca_interface(cdm_matdec_veh, veh_eol_to_recycling, write_pickle=False):

    # DM_ind = {"prod-domestic-production_bld-floor" : DM_production["bld-floor"],
    #           "prod-domestic-production_bld-domapp" : DM_production["bld-domapp"],
    #           "prod-domestic-production_bld-electronics" : DM_production["bld-electronics"],
    #           "prod-domestic-production_tra-infra" : DM_production["tra-infra"],
    #           "prod-domestic-production_tra-veh" : DM_production["tra-veh"],
    #           "prod-domestic-production_pack" : DM_production["pack"],
    #           "mat-demand" : DM_material_demand["material-demand"],
    #           "veh-to-recycling" : veh_eol_to_recycling}

    DM_ind = {"veh-matdec": cdm_matdec_veh, "veh-to-recycling": veh_eol_to_recycling}

    # of write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory, "../_database/data/interface/industry_to_lca.pickle"
        )
        with open(f, "wb") as handle:
            pickle.dump(DM_ind, handle, protocol=pickle.HIGHEST_PROTOCOL)
