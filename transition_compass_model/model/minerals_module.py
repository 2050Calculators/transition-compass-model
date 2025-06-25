#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:29:51 2024

@author: echiarot
"""

from model.common.data_matrix_class import DataMatrix
from model.common.constant_data_matrix_class import ConstantDataMatrix
from model.common.io_database import read_database_fxa
from model.common.interface_class import Interface
from model.common.auxiliary_functions import filter_geoscale, cdm_to_dm, simulate_input, calibration_rates
from model.common.auxiliary_functions import material_switch, material_decomposition
import pandas as pd
import pickle
import os
import numpy as np
import re
import warnings

warnings.simplefilter("ignore")


def relative_reserve(minerals, dm, reserve_starting_year, mineral_type, range_max):
    prefix_1 = "min_reserve_"
    if mineral_type == "fossil_fuel":
        prefix_2 = "min_energy_"
    if mineral_type == "mineral":
        prefix_2 = "min_extraction_"
    warning_name = "min_" + mineral_type + "_warning"

    # create empty array for warning
    index_warning = np.empty(0)

    # create dictionary where to save things
    variabs_relresleft = ["min_relative_reserves_left_" + i + "[%]" for i in minerals]
    output = dict.fromkeys(variabs_relresleft, None)
    output[warning_name] = None
    output["Country"] = "Europe"
    output["Years"] = 2050

    for k in range(len(minerals)):

        # get names of reserves and mineral variables
        variabs_reserve = [prefix_1 + i for i in minerals]
        variabs_mineral = [prefix_2 + i for i in minerals]

        # get indexes
        idx = dm.idx

        # get last year of considered reserves
        reserve_starting = dm.array[:, idx[reserve_starting_year], idx[variabs_reserve[k]]]

        # get indexes for years after start reserves
        years = dm.col_labels["Years"]
        years = np.array(years)[[i > reserve_starting_year for i in years]].tolist()
        idx_years = [idx[i] for i in years]

        # make cumulative use of minerals
        mineral_yearly = dm.array[:, idx_years, idx[variabs_mineral[k]]] * 5
        mineral_cum = np.cumsum(mineral_yearly)

        # compute series for reserves left
        reserve_left = np.append(reserve_starting, reserve_starting - mineral_cum)

        # get warning if there is no more reserves left in 2050 
        relative_reserve_left = (reserve_left[len(reserve_left) - 1] / reserve_left[0] - 1) * -100
        output[variabs_relresleft[k]] = relative_reserve_left

        if 100 <= relative_reserve_left <= range_max:
            index_warning = np.append(index_warning, 1)

    # count how many warnings for minerals
    index_warning = np.count_nonzero(index_warning)
    if index_warning == 0:
        index_warning = 0
    if index_warning == 1:
        index_warning = 1
    if index_warning >= 2:
        index_warning = 2

    # store warnings
    output[warning_name] = index_warning

    return output


def database_from_csv_to_datamatrix():
    # Read database
    # Set years range
    years_setting = [1990, 2015, 2050, 5]
    startyear = years_setting[0]
    baseyear = years_setting[1]
    lastyear = years_setting[2]
    step_fts = years_setting[3]
    years_ots = list(np.linspace(start=startyear, stop=baseyear, num=(baseyear - startyear) + 1).astype(
        int))  # make list with years from 1990 to 2015
    years_fts = list(
        np.linspace(start=baseyear + step_fts, stop=lastyear, num=int((lastyear - baseyear) / step_fts)).astype(
            int))  # make list with years from 2020 to 2050 (steps of 5 years)
    years_all = years_ots + years_fts

    #####################
    # FIXED ASSUMPTIONS #
    #####################

    # Read fixed assumptions to datamatrix
    df = read_database_fxa('minerals_fixed-assumptions')
    dm = DataMatrix.create_from_df(df, num_cat=0)

    # Keep only ots and fts years
    dm = dm.filter(selected_cols={'Years': years_all})
    dm.col_labels

    # make data matrixes with specific data using regular expression (regex)
    dm_elec_new = dm.filter_w_regex({'Variables': 'elc_new_RES.*|elc_new.*|.*solar.*'})
    dm_elec_new.col_labels
    dm_min_other = dm.filter_w_regex({'Variables': 'min_other.*'})
    dm_min_proportion = dm.filter_w_regex({'Variables': 'min_proportion.*'})

    # modify elec_new

    # rename
    dm_elec_new.rename_col_regex("_tech", "", dim="Variables")
    dm_elec_new.rename_col_regex("_new_RES", "", dim="Variables")
    dm_elec_new.rename_col_regex("_new_fossil", "", dim="Variables")
    dm_elec_new.rename_col_regex("elc_", "elc_energy-", dim="Variables")
    # FIXME: at the moment we do not have ots for oil and coal

    # make all zeroes for oil and coal ots for the moment
    c, y = len(dm_elec_new.col_labels["Country"]), len(dm_elec_new.col_labels["Years"])
    arr_temp = np.zeros((c, y))
    dm_elec_new.add(arr_temp, dim="Variables", col_label="elc_energy-coal", unit="GW")
    dm_elec_new.add(arr_temp, dim="Variables", col_label="elc_energy-oil", unit="GW")

    # deepen
    dm_elec_new.deepen()

    # save
    dict_fxa = {
        'elec_new': dm_elec_new,
        'min_other': dm_min_other,
        'min_proportion': dm_min_proportion
    }

    ###############
    # CALIBRATION #
    ###############

    # Read calibration
    df = read_database_fxa('minerals_calibration')
    dm_cal = DataMatrix.create_from_df(df, num_cat=0)

    #############
    # CONSTANTS #
    #############

    # Load constants
    cdm_const = ConstantDataMatrix.extract_constant('interactions_constants',
                                                    pattern='cp_ind_material-efficiency.*|cp_min.*', num_cat=0)
    cdm_const.rename_col_regex("cp_", "", "Variables")

    # split
    dict_const = {}

    # batteries vehicles
    cdm_temp = cdm_const.filter_w_regex({"Variables": ".*batveh.*"})
    cdm_temp.deepen()
    dict_const["batveh"] = cdm_temp

    # trade
    cdm_temp = cdm_const.filter_w_regex({"Variables": ".*trade*"})
    # cdm_temp = cdm_to_dm(cdm_temp, countries_list, years_list)
    cdm_temp.deepen_twice()
    cdm_temp.rename_col(col_in="min_trade", col_out="product-demand-split-share", dim="Variables")
    dict_const["trade"] = cdm_temp

    # batteries
    cdm_temp = cdm_const.filter_w_regex({"Variables": ".*battery*"})
    cdm_temp = cdm_temp.filter_w_regex({"Variables": "^((?!trade|energytech).)*$"})
    cdm_temp.deepen_twice()
    dict_const["battery"] = cdm_temp

    # LDV
    cdm_temp = cdm_const.filter_w_regex({"Variables": ".*LDV*"})
    cdm_temp = cdm_temp.filter_w_regex({"Variables": "^((?!batveh).)*$"})
    cdm_temp.deepen_twice()
    dict_const["LDV"] = cdm_temp

    # HDV
    cdm_temp = cdm_const.filter_w_regex({"Variables": ".*HDV*"})
    cdm_temp = cdm_temp.filter_w_regex({"Variables": "^((?!batveh).)*$"})
    cdm_temp.deepen_twice()
    dict_const["HDV"] = cdm_temp

    # other transport
    tra_oth = ['2W-EV', '2W-FCEV', '2W-ICE', '2W-PHEV', 'bus-EV', 'bus-FCEV', 'bus-ICE', 'bus-PHEV',
               'other-planes', 'other-ships', 'other-trains']
    find = [".*" + i + ".*" for i in tra_oth]
    cdm_temp = cdm_const.filter_w_regex({"Variables": "|".join(find)})
    cdm_temp = cdm_temp.filter_w_regex({"Variables": "^((?!batveh).)*$"})
    cdm_temp.deepen_twice()
    dict_const["tra-other"] = cdm_temp

    # infrastructure
    cdm_temp = cdm_const.filter_w_regex({"Variables": ".*infra*"})
    cdm_temp.rename_col_regex(str1="infra_", str2="infra-", dim="Variables")
    cdm_temp.deepen_twice()
    dict_const["infra"] = cdm_temp

    # domestic appliances
    cdm_temp = cdm_const.filter_w_regex({"Variables": ".*appliance*"})
    cdm_temp = cdm_temp.filter_w_regex({"Variables": "^((?!min_other).)*$"})
    cdm_temp.rename_col_regex(str1="appliance", str2="dom-appliance", dim="Variables")
    cdm_temp.deepen_twice()
    dict_const["domapp"] = cdm_temp

    # domestic appliances other
    cdm_temp = cdm_const.filter_w_regex({"Variables": ".*min_other_dom*"})
    cdm_temp.rename_col_regex(str1="other_dom-appliances", str2="other-dom-appliance", dim="Variables")
    cdm_temp.deepen_twice()
    cdm_temp.sort("Categories2")
    dict_const["domapp-other"] = cdm_temp

    # electronics
    cdm_temp = cdm_const.filter_w_regex({"Variables": ".*electr*"})
    cdm_temp = cdm_temp.filter_w_regex({"Variables": "^((?!trade|batveh|battery).)*$"})
    cdm_temp.rename_col_regex(str1="electronics_", str2="electronics-", dim="Variables")
    cdm_temp.deepen_twice()
    dict_const["electronics"] = cdm_temp

    # buildings
    cdm_temp = cdm_const.filter_w_regex({"Variables": ".*building*"})
    cdm_temp.rename_col_regex(str1="building_", str2="floor-area-", dim="Variables")
    cdm_temp.rename_col_regex(str1="new_", str2="new-", dim="Variables")
    cdm_temp.rename_col_regex(str1="reno_", str2="reno-", dim="Variables")
    cdm_temp.deepen_twice()
    cdm_temp.sort("Categories1")
    dict_const["buildings"] = cdm_temp

    # energy
    cdm_temp = cdm_const.filter_w_regex({"Variables": ".*energy*"})
    cdm_temp = cdm_temp.filter_w_regex({"Variables": "^((?!trade|battery).)*$"})
    cdm_temp.deepen_twice()
    dict_const["energy"] = cdm_temp

    # share pv
    cdm_temp = cdm_const.filter_w_regex({"Variables": ".*min_share_pv_*"})
    cdm_temp.rename_col_regex(str1="pv_", str2="pv-", dim="Variables")
    dict_const["pv-share"] = cdm_temp

    # wire copper
    cdm_temp = cdm_const.filter_w_regex({"Variables": ".*wire_copper*"})
    dict_const["wire-copper"] = cdm_temp

    # unaccounted minerals
    minerals_sub1 = ["aluminium", "copper", "lead", "steel"]
    minerals_sub2 = ["graphite", "lithium", "manganese", "nickel"]
    cdm_temp = cdm_const.filter_w_regex({"Variables": ".*other*"})
    cdm_temp = cdm_temp.filter_w_regex({"Variables": "^((?!vehicle|appliance).)*$"})
    cdm_temp.deepen_twice()
    cdm_temp1 = cdm_temp.filter({"Categories2": minerals_sub1})
    dict_const["minerals-unaccounted-sub1"] = cdm_temp1
    cdm_temp2 = cdm_temp.filter({"Categories2": minerals_sub2})
    dict_const["minerals-unaccounted-sub2"] = cdm_temp2
    # dm_temp = cdm_to_dm(cdm_temp, countries_list = dm_other_mindec.col_labels["Country"], 
    #                     years_list = dm_other_mindec.col_labels["Years"])

    # switches
    conversions_old = ["min_industry_aluminium_lithium", "min_industry_steel_nickel", "min_industry_steel_manganese",
                       "min_industry_steel_graphite", "min_industry_glass_lithium", "min_industry_aluminium_manganese"]
    cdm_temp = cdm_const.filter({"Variables": conversions_old})
    conversions = ["min_aluminium-lithium", "min_steel-nickel", "min_steel-manganese",
                   "min_steel-graphite", "min_glass-lithium", "min_aluminium-manganese"]
    # conversions = ["min_material-switch-ratios_aluminium-to-lithium", "min_material-switch-ratios_steel-to-nickel", 
    #                "min_material-switch-ratios_steel-to-manganese", "min_material-switch-ratios_steel-to-graphite",
    #                "min_material-switch-ratios_glass-to-lithium", "min_material-switch-ratios_aluminium-to-manganese"]
    for i in range(len(conversions)):
        cdm_temp.rename_col(col_in=conversions_old[i], col_out=conversions[i], dim="Variables")
    cdm_temp.deepen()
    dict_const["material-switch"] = cdm_temp

    # efficiency
    cdm_temp = cdm_const.filter_w_regex({"Variables": ".*eff*"})
    dict_const["efficiency"] = cdm_temp

    # factors for primary production
    cdm_temp = cdm_const.filter_w_regex({"Variables": ".*recy*"})
    cdm_temp.sort("Variables")
    dict_const["factor-primary"] = cdm_temp

    # factors extraction
    cdm_temp = cdm_const.filter_w_regex({"Variables": ".*param*"})
    dict_const["factor-extraction"] = cdm_temp

    # reserves
    cdm_temp = cdm_const.filter_w_regex({"Variables": ".*reserve.*"})
    # dm_reserves = cdm_to_dm(cdm_reserves, countries_list = dm_extraction.col_labels["Country"], 
    #                         years_list = dm_extraction.col_labels["Years"])
    dict_const["reserves"] = cdm_temp

    ########
    # SAVE #
    ########

    DM_minerals = {
        'fxa': dict_fxa,
        'calibration': dm_cal,
        'constant': dict_const
    }

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, '../_database/data/datamatrix/minerals.pickle')
    with open(f, 'wb') as handle:
        pickle.dump(DM_minerals, handle, protocol=pickle.HIGHEST_PROTOCOL)

    del baseyear, cdm_const, df, dict_fxa, dm, dm_elec_new, dm_min_other, dm_min_proportion, DM_minerals, f, handle, lastyear, \
        startyear, step_fts, years_all, years_fts, years_ots, years_setting

    return


def read_data(data_file):
    # load datamatrixes
    with open(data_file, 'rb') as handle:
        DM_minerals = pickle.load(handle)

    # get constants
    CDM_const = DM_minerals["constant"].copy()

    # return
    return DM_minerals, CDM_const

def product_demand(DM_minerals, DM_buildings, DM_pow, DM_tra, CDM_const):
    # get fxa
    DM_fxa = DM_minerals['fxa']

    # get datamatrixes
    dm_tra_veh = DM_tra['tra_veh']
    dm_infra = DM_tra['tra_infra']

    dm_infra_temp = DM_buildings['bld-pipe']
    dm_constr = DM_buildings['bld-floor']
    dm_domapp = DM_buildings['bld-appliance']
    dm_electr = DM_buildings['bld-electr']

    # note that in dm_tra_veh.idx now product-demand appears as last, but this is fine as the order of idx is not important, what's important
    # is the value of the key

    #####################
    ##### BATTERIES #####
    #####################

    # this is demand for installed capacity of batteries, expressed in kWh

    # get demand for batteries from energy sector
    dm_battery = DM_pow["battery"].copy()

    # !FIXME this conversion doesn't make sense, at the same time, we need kWh because the other batteries are in kWh
    # convert from GW to kWh
    dm_battery.change_unit('str_energy-battery', factor=1e6, old_unit='GW', new_unit='kWh')

    # deepen
    dm_battery.deepen()
    dm_battery.rename_col(col_in="str", col_out="product-demand", dim="Variables")

    # get demand for batteries from transport and electronics

    # get kWh from constants
    cdm_temp = CDM_const["batveh"]

    # Compute battery kwh for electronics
    cdm_electronics = cdm_temp.filter_w_regex({'Categories1': 'electronics.*'})
    dm_electr_battery = dm_electr.filter({'Categories1': cdm_electronics.col_labels['Categories1']})
    ay_battery_electronics = dm_electr_battery.array * cdm_electronics.array[np.newaxis, np.newaxis, ...]
    dm_electr_battery = DataMatrix.based_on(ay_battery_electronics, format=dm_electr_battery,
                                            change={'Variables': ['product-demand']}, units={'product-demand': 'kWh'})
    # sum all categories together
    dm_electr_battery.groupby({'electronics-battery': '.*'}, dim='Categories1', regex=True, inplace=True)
    del cdm_electronics, ay_battery_electronics

    # Compute battery kwh for electronics
    cdm_temp.drop(col_label='electronics', dim='Categories1')
    dm_veh_battery = dm_tra_veh.filter({'Categories1': cdm_temp.col_labels['Categories1']})
    ay_battery_veh = dm_veh_battery.array * cdm_temp.array[np.newaxis, np.newaxis, ...]
    dm_veh_battery = DataMatrix.based_on(ay_battery_veh, format=dm_veh_battery,
                                         change={'Variables': ['product-demand']}, units={'product-demand': 'kWh'})
    # sum all categories together
    dm_veh_battery.groupby({'transport-battery': '.*'}, dim='Categories1', regex=True, inplace=True)
    del ay_battery_veh, cdm_temp

    # join transport batteries and electronics batteries
    dm_battery.append(dm_veh_battery, dim='Categories1')
    dm_battery.append(dm_electr_battery, dim='Categories1')


    ##########################
    ##### INFRASTRUCTURE #####
    ##########################

    dm_infra.rename_col('tra_new_infrastructure', 'product-demand', dim='Variables')
    # append
    dm_infra.append(dm_infra_temp, dim="Categories1")


    ##################
    ##### ENERGY #####
    ##################

    # this is the demand for energy that comes from different energy sources, expressed in GW (power)

    # Get new capacity installation
    dm_energy = DM_pow["energy"].copy()


    ########################
    ##### PUT TOGETHER #####
    ########################

    DM_demand = {"vehicles": dm_tra_veh,
                 "electronics": dm_electr,
                 "batteries": dm_battery,
                 "infrastructure": dm_infra,
                 "dom-appliance": dm_domapp,
                 "construction": dm_constr,
                 "energy": dm_energy}

    # return
    return DM_demand


def product_import(DM_ind):

    # get imports and rename
    dm_import = DM_ind["product-net-import"]
    dm_import.rename_col_regex(str1="_product-net-import", str2="", dim="Variables")

    # add net imports for categories of vehicles we do not have
    # !FIXME: this categories correspondance is very odd
    variabs = ["LDV-ICE", "HDVL-ICE", "LDV-ICE", "LDV-ICE", "LDV-ICE", "LDV-ICE", "HDVL-ICE",
               "HDVL-ICE",
               "HDVL-ICE", "HDVL-ICE", "HDVL-ICE", "HDVL-ICE", "HDVL-ICE", "HDVL-ICE", "HDVL-ICE", "HDVL-ICE",
               "HDVL-ICE", "HDVL-ICE"]
    variabs = ["ind_" + i for i in variabs]
    variabs_new = ["LDV-PHEV", "HDVL-PHEV", "2W-EV", "2W-ICE", "2W-FCEV", "2W-PHEV", "bus-EV",
                   "bus-ICE",
                   "bus-FCEV", "bus-PHEV", "HDVM-EV", "HDVM-ICE", "HDVM-FCEV", "HDVM-PHEV", "HDVH-EV", "HDVH-ICE",
                   "HDVH-FCEV", "HDVH-PHEV"]
    variabs_new = ["ind_" + i for i in variabs_new]

    idx = dm_import.idx
    for i in range(len(variabs)):
        dm_import.add(dm_import.array[:, :, idx[variabs[i]]], dim="Variables", col_label=variabs_new[i], unit="%")

    # sort
    dm_import.sort(dim="Variables")

    # clean
    del i, idx, variabs, variabs_new

    # return
    return dm_import


def product_demand_split(DM_demand, dm_import, CDM_const):
    ###################################
    ##### SHARE OF PRODUCT DEMAND #####
    ###################################

    # demand split
    dm_demand_split_share = dm_import.copy()

    # product indirect demand
    dm_demand_split_share.deepen()
    dm_demand_split_share.rename_col(col_in="ind", col_out="product-demand-split-share", dim="Variables")
    # Add a category 'indir'
    dm_demand_split_share.array = dm_demand_split_share.array[..., np.newaxis]
    dm_demand_split_share.idx['indir'] = 0
    dm_demand_split_share.col_labels['Categories2'] = ['indir']
    dm_demand_split_share.dim_labels.append('Categories2')

    # product net export = - indir
    arr_temp = -dm_demand_split_share.array
    dm_demand_split_share.add(arr_temp, dim="Categories2", col_label="exp", unit="%")

    # product direct demand
    idx = dm_demand_split_share.idx
    arr_temp = dm_demand_split_share.array[:, :, :, :, idx["indir"]] - dm_demand_split_share.array[:, :, :, :,
                                                                       idx["indir"]] + 1
    dm_demand_split_share.add(arr_temp, dim="Categories2", col_label="dir")

    # add to dm_trade the share for other products from constants
    countries_list = dm_demand_split_share.col_labels["Country"]
    years_list = dm_demand_split_share.col_labels["Years"]
    cdm_temp = CDM_const["trade"]
    cdm_temp = cdm_to_dm(cdm_temp, countries_list, years_list)

    # append
    dm_demand_split_share.append(cdm_temp, dim="Categories1")

    # sort
    dm_demand_split_share.sort(dim="Categories1")
    # note: careful that here sorts first the variables with cap letters

    # clean
    del arr_temp, cdm_temp, idx

    #################
    ##### UNITS #####
    #################

    # create empty dictionary without constructions (they are assumed not to be traded)
    DM_demand_split = dict.fromkeys(["vehicles", "electronics", "batteries",
                                     "infrastructure", "dom-appliance", "energy"])

    for key in DM_demand_split.keys():
        # get demand
        dm_demand_temp = DM_demand[key]

        # get corresponding split share
        dm_demand_split_temp = dm_demand_split_share.filter({"Categories1": dm_demand_temp.col_labels["Categories1"]})
        arr_temp = dm_demand_temp.array[..., np.newaxis] * dm_demand_split_temp.array

        # add split in unit as a variable
        dm_demand_split_temp.add(arr_temp, dim="Variables",
                                 col_label="product-demand-split-unit", unit=dm_demand_temp.units['product-demand'])

        # drop product demand split share
        dm_demand_split_temp.drop(dim="Variables", col_label=["product-demand-split-share"])

        # put dm in dictionary
        DM_demand_split[key] = dm_demand_split_temp

    # clean
    del key, dm_demand_temp, dm_demand_split_temp, arr_temp

    # return
    return DM_demand_split


def mineral_demand_split(DM_minerals, DM_demand, DM_demand_split, CDM_const, DM_ind, DM_pow):
    # name of minerals
    minerals = ['aluminium', 'copper', 'graphite', 'lead', 'lithium', 'manganese', 'nickel', 'steel']

    # get data
    DM_fxa = DM_minerals['fxa']

    #####################
    ##### BATTERIES #####
    #####################

    # get product demand split unit
    dm_temp = DM_demand_split["batteries"]

    # get constants for mineral decomposition
    cdm_temp = CDM_const["battery"]

    # get minderal decomposition
    dm_battery_mindec = material_decomposition(dm=dm_temp, cdm=cdm_temp)

    # clean
    del dm_temp, cdm_temp

    ######################
    ##### CARS (LDV) #####
    ######################

    # get names
    tra_veh = ['HDVH-EV', 'HDVH-FCEV', 'HDVH-ICE', 'HDVH-PHEV', 'HDVL-EV', 'HDVL-FCEV',
               'HDVL-ICE', 'HDVL-PHEV', 'HDVM-EV', 'HDVM-FCEV', 'HDVM-ICE', 'HDVM-PHEV', 'LDV-EV', 'LDV-FCEV',
               'LDV-ICE', 'LDV-PHEV',
               '2W-EV', '2W-FCEV', '2W-ICE', '2W-PHEV', 'bus-EV', 'bus-FCEV', 'bus-ICE', 'bus-PHEV',
               'other-planes', 'other-ships', 'other-subways', 'other-trains']
    tra_ldv = np.array(tra_veh)[[bool(re.search("LDV", str(i), flags=re.IGNORECASE)) for i in tra_veh]].tolist()

    # get product demand split unit
    dm_temp = DM_demand_split["vehicles"]
    dm_temp = dm_temp.filter_w_regex({"Categories1": ".*LDV*"})

    # get constants for mineral decomposition
    cdm_temp = CDM_const["LDV"]

    # get minderal decomposition
    dm_veh_ldv_mindec = material_decomposition(dm_temp, cdm_temp)

    # get sum across vehicles
    arr_temp = np.nansum(dm_veh_ldv_mindec.array, axis=-3, keepdims=True)
    dm_veh_ldv_mindec.add(arr_temp, dim="Categories1", col_label="LDV")
    dm_veh_ldv_mindec.drop(dim="Categories1", col_label=tra_ldv)

    # get mineral switch parameter
    dm_temp = DM_ind["material-switch"]
    dm_temp = dm_temp.filter_w_regex({"Variables": ".*switch-cars*"})

    # # FIXME: for the material switch, here in minerals they just use switch percentages, 
    # # but then they do not apply the ratio that says how much material is lost in the process (which is from constants).
    # # This is a bit weird, as these ratios are loaded in constants (but for the one material-to-other), 
    # # but then are not used. For the moment I keep this as it's in KNIME (no constant ratio applied)
    # # In case we want to change this later, here below is the code that makes it work with
    # # the function material_switch():
    # dm_temp.rename_col("ind_material-switch-cars-steel-other","ind_material-switch_cars-steel-to-other","Variables")
    # dm_temp.deepen()
    # CDM_const["material-switch"].add(1, dim = "Variables", col_label = "min_material-switch-ratios_steel-to-other", 
    #                                  unit = "%", dummy = True)
    # dm_veh_ldv_mindec_sub = dm_veh_ldv_mindec.flatten().flatten()
    # dm_veh_ldv_mindec_sub.deepen()
    # dm_veh_ldv_mindec_sub = dm_veh_ldv_mindec_sub.filter({"Categories1" : ['LDV_dir', 'LDV_exp']})
    # material_switch(dm = dm_veh_ldv_mindec_sub, dm_ots_fts = dm_temp,
    #                 cdm_const = CDM_const["material-switch"], material_in="steel", material_out=["other"],
    #                 product="LDV_dir", switch_percentage_prefix="cars-",
    #                 switch_ratio_prefix="min_material-switch-ratios_")

    # set variables with mineral that is switched
    mineral_in = "steel"
    mineral_out = "other"

    mineral_in_unadj = mineral_in + "-unadj"
    mineral_switched = mineral_in + "-switched-to-" + mineral_out

    dm_veh_ldv_mindec.rename_col(col_in=mineral_in, col_out=mineral_in_unadj, dim="Categories3")

    # for mineral that is switched, get how much is switched and add it as new variable
    idx = dm_veh_ldv_mindec.idx
    arr_temp = dm_veh_ldv_mindec.array[:, :, :, :, :, idx[mineral_in_unadj]]
    arr_temp = arr_temp[..., np.newaxis] * dm_temp.array[:, :, np.newaxis, np.newaxis, np.newaxis, :]
    dm_veh_ldv_mindec.add(arr_temp, dim="Categories3", col_label=mineral_switched)

    # do mineral unadjusted - mineral switched
    dm_veh_ldv_mindec.operation(col1=mineral_in_unadj, operator="-", col2=mineral_switched, dim="Categories3",
                                out_col=mineral_in)

    # for indir, substitute back the unadjusted one (adjustment is only done on exp and dir)
    dm_veh_ldv_mindec.array[:, :, :, :, idx["indir"], idx[mineral_in]] = dm_veh_ldv_mindec.array[:, :, :, :,
                                                                         idx["indir"], idx[mineral_in_unadj]]

    # drop
    dm_veh_ldv_mindec.drop(dim="Categories3", col_label=[mineral_in_unadj, mineral_switched])

    # clean
    del dm_temp, cdm_temp, mineral_in, mineral_out, mineral_switched, mineral_in_unadj, idx, arr_temp

    ########################
    ##### TRUCKS (HDV) #####
    ########################

    # get names
    tra_hdv = np.array(tra_veh)[[bool(re.search("HDV", str(i), flags=re.IGNORECASE)) for i in tra_veh]].tolist()

    # get product demand split unit
    dm_temp = DM_demand_split["vehicles"]
    dm_temp = dm_temp.filter_w_regex({"Categories1": ".*HDV*"})

    # get constants for mineral decomposition
    cdm_temp = CDM_const["HDV"]

    # get minderal decomposition
    dm_veh_hdv_mindec = material_decomposition(dm_temp, cdm_temp)

    # get sum across vehicles
    arr_temp = np.nansum(dm_veh_hdv_mindec.array, axis=-3, keepdims=True)
    dm_veh_hdv_mindec.add(arr_temp, dim="Categories1", col_label="HDV")
    dm_veh_hdv_mindec.drop(dim="Categories1", col_label=tra_hdv)

    # do the switch steel to other and steel to aluminium

    # get mineral switch parameter
    dm_temp = DM_ind["material-switch"]
    dm_temp_alu = dm_temp.filter_w_regex({"Variables": ".*switch-trucks-steel-to-aluminium*"})
    dm_temp = dm_temp.filter_w_regex({"Variables": ".*switch-trucks-steel-to-chem*"})

    # set variables with mineral that is switched
    mineral_in = "steel"
    mineral_out = "chem"
    mineral_out2 = "aluminium"

    mineral_in_unadj = mineral_in + "-unadj"
    mineral_out2_unadj = mineral_out2 + "-unadj"

    mineral_switched = mineral_in + "-switched-to-" + mineral_out
    mineral_switched2 = mineral_in + "-switched-to-" + mineral_out2

    dm_veh_hdv_mindec.rename_col(col_in=mineral_in, col_out=mineral_in_unadj, dim="Categories3")
    dm_veh_hdv_mindec.rename_col(col_in=mineral_out2, col_out=mineral_out2_unadj, dim="Categories3")

    # for mineral that is switched, get how much is switched and add it as new variable
    idx = dm_veh_hdv_mindec.idx

    arr_temp = dm_veh_hdv_mindec.array[:, :, :, :, :, idx[mineral_in_unadj]]
    arr_temp = arr_temp[..., np.newaxis] * dm_temp.array[:, :, np.newaxis, np.newaxis, np.newaxis, :]
    dm_veh_hdv_mindec.add(arr_temp, dim="Categories3", col_label=mineral_switched)

    arr_temp = dm_veh_hdv_mindec.array[:, :, :, :, :, idx[mineral_in_unadj]]
    arr_temp = arr_temp[..., np.newaxis] * dm_temp_alu.array[:, :, np.newaxis, np.newaxis, np.newaxis, :]
    dm_veh_hdv_mindec.add(arr_temp, dim="Categories3", col_label=mineral_switched2)

    # do mineral unadjusted - mineral switched
    idx = dm_veh_hdv_mindec.idx
    arr_temp = dm_veh_hdv_mindec.array[..., idx[mineral_in_unadj]] - \
               dm_veh_hdv_mindec.array[..., idx[mineral_switched]] - \
               dm_veh_hdv_mindec.array[..., idx[mineral_switched2]]
    dm_veh_hdv_mindec.add(arr_temp, dim="Categories3", col_label=mineral_in)

    # do aluminium + steel-switched-to-aluminium
    dm_veh_hdv_mindec.operation(col1=mineral_out2_unadj, operator="+", col2=mineral_switched2, dim="Categories3",
                                out_col=mineral_out2)

    # for indir, substitute back the unadjusted ones (adjustment is only done on exp and dir)
    dm_veh_hdv_mindec.array[:, :, :, :, idx["indir"], idx[mineral_in]] = dm_veh_hdv_mindec.array[:, :, :, :,
                                                                         idx["indir"], idx[mineral_in_unadj]]
    dm_veh_hdv_mindec.array[:, :, :, :, idx["indir"], idx[mineral_out2]] = dm_veh_hdv_mindec.array[:, :, :, :,
                                                                           idx["indir"], idx[mineral_out2_unadj]]

    # drop
    dm_veh_hdv_mindec.drop(dim="Categories3",
                           col_label=[mineral_in_unadj, mineral_out2_unadj, mineral_switched, mineral_switched2])

    # sort
    dm_veh_hdv_mindec.sort("Categories3")

    # clean
    del dm_temp, cdm_temp, mineral_in, mineral_out, mineral_out2, mineral_switched, \
        mineral_switched2, mineral_in_unadj, mineral_out2_unadj, idx, arr_temp, dm_temp_alu

    ##########################
    ##### OTHER VEHICLES #####
    ##########################

    # get other vehicles
    tra_oth = ['2W-EV', '2W-FCEV', '2W-ICE', '2W-PHEV', 'bus-EV', 'bus-FCEV', 'bus-ICE', 'bus-PHEV',
               'other-planes', 'other-ships', 'other-trains']

    # get product demand split unit
    dm_temp = DM_demand_split["vehicles"]
    dm_temp = dm_temp.filter({"Categories1": tra_oth})

    # get constants for mineral decomposition
    cdm_temp = CDM_const["tra-other"]

    # get minderal decomposition
    dm_veh_oth_mindec = material_decomposition(dm_temp, cdm_temp)

    # get sum across vehicles
    arr_temp = np.nansum(dm_veh_oth_mindec.array, axis=-3, keepdims=True)
    dm_veh_oth_mindec.add(arr_temp, dim="Categories1", col_label="other")
    dm_veh_oth_mindec.drop(dim="Categories1", col_label=tra_oth)

    # clean
    del dm_temp, cdm_temp, arr_temp

    ###########################
    ##### TRANSPORT TOTAL #####
    ###########################

    # get batteries for transport
    dm_veh_batt_mindec = dm_battery_mindec.filter({"Categories1": ['transport-battery']})

    # add missing minerals to the dms
    DM_temp = {"ldv": dm_veh_ldv_mindec,
               "hdv": dm_veh_hdv_mindec,
               "oth": dm_veh_oth_mindec,
               "batt": dm_veh_batt_mindec}

    for key in DM_temp.keys():

        variables = DM_temp[key].col_labels["Categories3"]
        variables_missing = np.array(minerals)[[i not in variables for i in minerals]].tolist()

        for variable in variables_missing:
            DM_temp[key].add(np.nan, dim="Categories3", col_label=variable, dummy=True)

    # sum across vehicles and batteries
    dm_veh_mindec = dm_veh_ldv_mindec.copy()
    dm_veh_mindec.append(dm_veh_hdv_mindec, dim="Categories1")
    dm_veh_mindec.append(dm_veh_oth_mindec, dim="Categories1")
    dm_veh_mindec.append(dm_veh_batt_mindec, dim="Categories1")
    arr_temp = np.nansum(dm_veh_mindec.array, axis=-3, keepdims=True)
    dm_veh_mindec.add(arr_temp, dim="Categories1", col_label="transport")
    dm_veh_mindec.drop(dim="Categories1", col_label=['LDV', 'HDV', 'other', 'transport-battery'])

    del dm_veh_ldv_mindec, dm_veh_hdv_mindec, dm_veh_oth_mindec, arr_temp, DM_temp, variable, variables, variables_missing

    ##########################
    ##### INFRASTRUCTURE #####
    ##########################

    # names for infra
    tra_inf = ['infra-rail', 'infra-road', 'infra-trolley-cables']
    bld_infra = ['infra-pipe']
    infra = tra_inf + bld_infra

    # get product demand split unit
    dm_temp = DM_demand_split["infrastructure"]

    # get constants for mineral decomposition
    cdm_temp = CDM_const["infra"]

    # get minderal decomposition
    dm_infra_mindec = material_decomposition(dm_temp, cdm_temp)

    # get sum across infra
    arr_temp = np.nansum(dm_infra_mindec.array, axis=-3, keepdims=True)
    dm_infra_mindec.add(arr_temp, dim="Categories1", col_label="infra")
    dm_infra_mindec.drop(dim="Categories1", col_label=infra)

    # clean
    del dm_temp, cdm_temp, arr_temp

    ##############################
    ##### DOMESTIC APPLIANCE #####
    ##############################

    # names for domapp
    domapp = ['dom-appliance-dishwasher', 'dom-appliance-dryer', 'dom-appliance-freezer',
              'dom-appliance-fridge', 'dom-appliance-wmachine']

    # get product demand split unit
    dm_temp = DM_demand_split["dom-appliance"]

    # get constants for mineral decomposition
    cdm_temp = CDM_const["domapp"]

    # get minderal decomposition
    dm_domapp_mindec = material_decomposition(dm_temp, cdm_temp)

    # get sum across dom appliance
    arr_temp = np.nansum(dm_domapp_mindec.array, axis=-3, keepdims=True)
    dm_domapp_mindec.add(arr_temp, dim="Categories1", col_label="dom-appliance")
    dm_domapp_mindec.drop(dim="Categories1", col_label=domapp)

    # get factor for materials coming from unaccounted appliances
    cdm_temp = CDM_const["domapp-other"]

    # divide mineral split by this factor (to get mineral + extra mineral from unaccounted sectors)
    dm_domapp_mindec.array = dm_domapp_mindec.array / cdm_temp.array[np.newaxis, np.newaxis, np.newaxis, ...]

    # get aluminium packages (t) and add it to aluminium from dom appliance (only for dir)
    dm_temp = DM_ind["aluminium-pack"]
    dm_temp.array = dm_temp.array * 1000  # make kg
    idx = dm_domapp_mindec.idx
    dm_domapp_mindec.array[:, :, :, :, idx["dir"], idx["aluminium"]] = dm_domapp_mindec.array[:, :, :, :, idx["dir"],
                                                                       idx["aluminium"]] + \
                                                                       dm_temp.array[..., np.newaxis]

    # clean
    del dm_temp, cdm_temp, arr_temp, idx

    #######################
    ##### ELECTRONICS #####
    #######################

    # names of electronics
    electr = ['electronics-computer', 'electronics-phone', 'electronics-tv']

    # get product demand split unit
    dm_temp = DM_demand_split["electronics"]

    # get constants for mineral decomposition
    cdm_temp = CDM_const["electronics"]

    # get minderal decomposition
    dm_electr_cotvph_mindec = material_decomposition(dm_temp, cdm_temp)

    # get batteries for electronics
    dm_electr_batt_mindec = dm_battery_mindec.filter({"Categories1": ['electronics-battery']})

    # add missing minerals to the dms
    DM_temp = {"electr": dm_electr_cotvph_mindec,
               "batt": dm_electr_batt_mindec}

    for key in DM_temp.keys():

        variables = DM_temp[key].col_labels["Categories3"]
        variables_missing = np.array(minerals)[[i not in variables for i in minerals]].tolist()

        for variable in variables_missing:
            DM_temp[key].add(np.nan, dim="Categories3", col_label=variable, dummy=True)

    # append
    dm_electr_cotvph_mindec.append(dm_electr_batt_mindec, dim="Categories1")

    # get sum across electr
    arr_temp = np.nansum(dm_electr_cotvph_mindec.array, axis=-3, keepdims=True)
    dm_electr_cotvph_mindec.add(arr_temp, dim="Categories1", col_label="electronics")
    electr = electr + ["electronics-battery"]
    dm_electr_cotvph_mindec.drop(dim="Categories1", col_label=electr)

    # copy
    dm_electr_mindec = dm_electr_cotvph_mindec.copy()

    # clean
    del dm_electr_cotvph_mindec, dm_electr_batt_mindec, dm_temp, cdm_temp, arr_temp, DM_temp, key

    ########################
    ##### CONSTRUCTION #####
    ########################

    # names of construction
    constr = ['floor-area-new-non-residential', 'floor-area-new-residential',
              'floor-area-reno-non-residential', 'floor-area-reno-residential']

    # get product demand split unit
    dm_temp = DM_demand["construction"].copy()

    # convert Mm2 to m2
    dm_temp.change_unit('product-demand', factor=1e6, old_unit='Mm2', new_unit='m2')

    # expand of 1 dimension
    dm_temp.array = dm_temp.array[..., np.newaxis]
    dm_temp.col_labels["Categories2"] = ["dir"]
    dm_temp.dim_labels = dm_temp.dim_labels + ["Categories2"]
    dm_temp.idx["Categories2"] = 0

    # add exp and indir as nan 
    variables_missing = ["exp", "indir"]
    for variable in variables_missing:
        dm_temp.add(np.nan, dim="Categories2", col_label=variable, dummy=True)

    # get constants for mineral decomposition
    cdm_temp = CDM_const["buildings"]

    # get mineral decomposition
    dm_constr_mindec = material_decomposition(dm_temp, cdm_temp)

    # get sum across buildings
    arr_temp = np.nansum(dm_constr_mindec.array, axis=-3, keepdims=True)
    dm_constr_mindec.add(arr_temp, dim="Categories1", col_label="construction")
    dm_constr_mindec.drop(dim="Categories1", col_label=constr)

    # get mineral switch parameter
    dm_temp = DM_ind["material-switch"]
    dm_temp = dm_temp.filter_w_regex({"Variables": ".*switch-build*"})

    # set variables with mineral that is switched
    mineral_in = "steel"
    mineral_out = "timber"

    mineral_in_unadj = mineral_in + "-unadj"
    mineral_switched = mineral_in + "-switched-to-" + mineral_out

    dm_constr_mindec.rename_col(col_in=mineral_in, col_out=mineral_in_unadj, dim="Categories3")

    # for mineral that is switched, get how much is switched and add it as new variable
    idx = dm_constr_mindec.idx
    arr_temp = dm_constr_mindec.array[:, :, :, :, :, idx[mineral_in_unadj]]
    arr_temp = arr_temp[..., np.newaxis] * dm_temp.array[:, :, np.newaxis, np.newaxis, np.newaxis, :]
    dm_constr_mindec.add(arr_temp, dim="Categories3", col_label=mineral_switched)

    # do mineral unadjusted - mineral switched
    dm_constr_mindec.operation(col1=mineral_in_unadj, operator="-", col2=mineral_switched, dim="Categories3",
                               out_col=mineral_in)

    # for indir and exp, substitute back the unadjusted one (adjustment is only done on exp and dir)
    dm_constr_mindec.array[:, :, :, :, idx["indir"], idx[mineral_in]] = dm_constr_mindec.array[:, :, :, :, idx["indir"],
                                                                        idx[mineral_in_unadj]]
    dm_constr_mindec.array[:, :, :, :, idx["exp"], idx[mineral_in]] = dm_constr_mindec.array[:, :, :, :, idx["exp"],
                                                                      idx[mineral_in_unadj]]

    # drop
    dm_constr_mindec.drop(dim="Categories3", col_label=[mineral_in_unadj, mineral_switched])

    # clean
    del dm_temp, cdm_temp, mineral_in, mineral_out, mineral_switched, mineral_in_unadj, idx, arr_temp, variables, variable, \
        variables_missing

    ##################
    ##### ENERGY #####
    ##################

    # names for energy
    energy = ['energy-coal', 'energy-csp', 'energy-gas', 'energy-geo', 'energy-hydro',
              'energy-marine', 'energy-nuclear', 'energy-off-wind', 'energy-oil',
              'energy-on-wind', 'energy-pv']

    # get product demand split unit
    dm_temp = DM_demand_split["energy"].copy()

    # get constants for mineral decomposition
    cdm_temp = CDM_const["energy"]

    # get constants for thin film
    cdm_temp2 = CDM_const["pv-share"]

    # get indir, exp and dir for energy-pv-csi (by multipilication with the thin film and csi factors)
    idx = dm_temp.idx
    idx2 = cdm_temp2.idx

    arr_temp = dm_temp.array[:, :, :, idx["energy-pv"], :] * cdm_temp2.array[idx2["min_share_pv-csi"]]
    dm_temp.add(arr_temp, dim="Categories1", col_label="energy-pv-csi")

    arr_temp = dm_temp.array[:, :, :, idx["energy-pv"], :] * cdm_temp2.array[idx2["min_share_pv-thinfilm"]]
    dm_temp.add(arr_temp, dim="Categories1", col_label="energy-pv-thinfilm")

    dm_temp.drop(dim="Categories1", col_label=["energy-pv"])
    energy = energy[0:-1]
    energy = energy + ['energy-pv-csi', 'energy-pv-thinfilm']

    # get mineral decomposition
    dm_energy_tech_mindec = material_decomposition(dm_temp, cdm_temp)

    # get batteries for energy
    dm_energy_batt_mindec = dm_battery_mindec.filter({"Categories1": ['energy-battery']})

    # add missing minerals to the dms
    DM_temp = {"energy": dm_energy_tech_mindec,
               "batt": dm_energy_batt_mindec}

    for key in DM_temp.keys():

        variables = DM_temp[key].col_labels["Categories3"]
        variables_missing = np.array(minerals)[[i not in variables for i in minerals]].tolist()

        for variable in variables_missing:
            DM_temp[key].add(np.nan, dim="Categories3", col_label=variable, dummy=True)

    # append
    dm_energy_tech_mindec.append(dm_energy_batt_mindec, dim="Categories1")
    dm_energy_mindec = dm_energy_tech_mindec.copy()

    # get sum across energy
    arr_temp = np.nansum(dm_energy_mindec.array, axis=-3, keepdims=True)
    dm_energy_mindec.add(arr_temp, dim="Categories1", col_label="energy")
    energy = energy + ["energy-battery"]
    dm_energy_mindec.drop(dim="Categories1", col_label=energy)
    # not that here for example Austria 2020 for dir_energy_aluminium differs slightly from KNIME, supposedly for rounding differences (numbers are generally fine)

    # !FIXME: there is a link between copper demand and electricity infrastructure,
    # !FIXME: but here we are multiplying the total electricity demand by a constant
    # that gives us the total copper-wire demand, and documentation for this is missing.
    # get electricity demand total (GWh) and constant for amount of copper in wires (kg/GWh)
    dm_temp = DM_pow["electricity-demand"]
    cdm_temp = CDM_const["wire-copper"]

    # multiply direct demand times amount of copper in wires to get amount of copper in wires (kg)
    arr_temp = dm_temp.array * cdm_temp.array
    dm_temp = dm_energy_mindec.filter({"Categories2": ["dir"]})
    dm_temp = dm_temp.filter({"Categories3": ["copper"]})
    dm_temp.add(arr_temp[..., np.newaxis, np.newaxis, np.newaxis], col_label="copper-wire", dim="Categories3")

    # add amount of coppers in wires to copper from energy
    idx = dm_temp.idx
    dm_temp.array[..., idx["copper"]] = np.nansum(dm_temp.array, axis=-1)
    dm_temp.drop(dim="Categories3", col_label=["copper-wire"])

    idx1 = dm_energy_mindec.idx
    idx2 = dm_temp.idx
    dm_energy_mindec.array[:, :, :, :, idx1["dir"], idx1["copper"]] = dm_temp.array[:, :, :, :, idx2["dir"],
                                                                      idx2["copper"]]

    # clean
    del dm_energy_tech_mindec, dm_energy_batt_mindec, dm_temp, cdm_temp, arr_temp, cdm_temp2, idx, idx2, \
        DM_temp, key, variable, variables, variables_missing, idx1

    ########################
    ##### ALL MINERALS #####
    ########################

    # add minerals as nans for those dms which do not have all minerals

    DM_mindec = {"transport": dm_veh_mindec,
                 "infra": dm_infra_mindec,
                 "dom-appliance": dm_domapp_mindec,
                 "electronics": dm_electr_mindec,
                 "construction": dm_constr_mindec,
                 "energy": dm_energy_mindec}

    for key in DM_mindec.keys():

        variables = DM_mindec[key].col_labels["Categories3"]
        variables_missing = np.array(minerals)[[i not in variables for i in minerals]].tolist()

        for variable in variables_missing:
            DM_mindec[key].add(np.nan, dim="Categories3", col_label=variable, dummy=True)

    # sum minerals across all sectors
    dm_mindec = dm_veh_mindec.copy()
    mylist = ['infra', 'dom-appliance', 'electronics', 'construction', 'energy']
    for key in mylist:
        dm_mindec.append(DM_mindec[key], dim="Categories1")
    arr_temp = np.nansum(dm_mindec.array, axis=-3, keepdims=True)
    drop = list(DM_mindec)
    dm_mindec.add(arr_temp, dim="Categories1", col_label="all-sectors")
    dm_mindec.drop(dim="Categories1", col_label=drop)

    # clean
    del key, arr_temp, variables, variable, variables_missing, mylist

    #######################################
    ##### OTHER (UNACCOUNTED) SECTORS #####
    #######################################

    minerals_sub1 = ["aluminium", "copper", "lead", "steel"]

    # create dm for other minerals
    dm_other_mindec = dm_mindec.filter({"Categories3": minerals_sub1})
    dm_other_mindec = dm_other_mindec.filter({"Categories2": ["dir"]})

    # get constants
    cdm_temp = CDM_const["minerals-unaccounted-sub1"]

    # expand constants
    dm_temp = cdm_to_dm(cdm_temp, countries_list=dm_other_mindec.col_labels["Country"],
                        years_list=dm_other_mindec.col_labels["Years"])

    # multiply total times factors and add them to dm_other_mindec
    arr_temp = dm_other_mindec.array * dm_temp.array[..., np.newaxis, :]
    dm_other_mindec.add(arr_temp, dim="Categories1", col_label="other")
    dm_other_mindec.drop(dim="Categories1", col_label="all-sectors")

    # clean
    del cdm_temp, dm_temp, arr_temp

    ####################
    ##### INDUSTRY #####
    ####################

    # Note: this is done only for direct demand, so industries in foreign countries are not considered

    minerals_sub2 = ["graphite", "lithium", "manganese", "nickel"]

    # add other aluminium and steel temporarely to total (this is just for computing industry, you'll redo the addition of everything at the end)
    dm_mindec_temp = dm_mindec.filter({"Categories3": ["aluminium", "steel"], "Categories2": ["dir"]})
    dm_other_mindec_temp = dm_other_mindec.filter({"Categories3": ["aluminium", "steel"], "Categories2": ["dir"]})
    dm_mindec_temp.append(dm_other_mindec_temp, "Categories1")
    arr_temp = np.nansum(dm_mindec_temp.array, axis=-3, keepdims=True)
    dm_mindec_temp.drop(dim="Categories1", col_label="all-sectors")
    dm_mindec_temp.add(arr_temp, dim="Categories1", col_label="all-sectors")
    dm_mindec_temp.drop(dim="Categories1", col_label="other")

    # create dm for industry: take direct demand for steel and aluminium and convert from kg to mt
    dm_industry_mindec = dm_mindec_temp.copy()
    dm_industry_mindec.change_unit('material-decomposition', factor=1e-9, old_unit='kg', new_unit='Mt')

    # take glass
    dm_temp2 = DM_ind["material-production"].filter({"Variables": ["ind_material-production_glass"]})
    dm_industry_mindec.add(dm_temp2.array[:, :, np.newaxis, np.newaxis, np.newaxis, :], dim="Categories3",
                           col_label="glass")

    # get constants for switches
    cdm_temp = CDM_const["material-switch"]

    # multiply direct demand by these "switch factors"
    # FIXME: in theory these are not switch factors (which are a lever coming from industry)
    # but are recovery factors (which are usually applied after switch factors). For the moment
    # I leave it like this (as it is in KNIME), to be checked later.
    idx_dm = dm_industry_mindec.idx
    idx_cdm = cdm_temp.idx
    col1 = ["aluminium", "aluminium", "steel", "steel", "steel", "glass"]
    col2 = ["aluminium-lithium", "aluminium-manganese", "steel-nickel", "steel-manganese", "steel-graphite",
            "glass-lithium"]
    for i in range(len(col1)):
        arr_temp = dm_industry_mindec.array[..., idx_dm[col1[i]]] * cdm_temp.array[..., idx_cdm[col2[i]]]
        dm_industry_mindec.add(arr_temp, dim="Categories3", col_label=col2[i])

    # drop starting point minerals
    dm_industry_mindec.drop(dim="Categories3", col_label=["aluminium", "glass", "steel"])

    # transform in kg
    dm_industry_mindec.change_unit('material-decomposition', factor=1e9, old_unit='Mt', new_unit='kg')

    # sum over end point minerals
    dm_temp = dm_industry_mindec.filter({"Categories3": ["aluminium-lithium", "glass-lithium"]})
    arr_temp = np.nansum(dm_temp.array, axis=-1, keepdims=True)
    dm_industry_mindec.add(arr_temp, dim="Categories3", col_label="lithium")
    dm_industry_mindec.drop(dim="Categories3", col_label=["aluminium-lithium", "glass-lithium"])

    dm_temp2 = dm_industry_mindec.filter({"Categories3": ["aluminium-manganese", "steel-manganese"]})
    arr_temp = np.nansum(dm_temp2.array, axis=-1, keepdims=True)
    dm_industry_mindec.add(arr_temp, dim="Categories3", col_label="manganese")
    dm_industry_mindec.drop(dim="Categories3", col_label=["aluminium-manganese", "steel-manganese"])

    dm_industry_mindec.rename_col(col_in='steel-nickel', col_out="nickel", dim="Categories3")
    dm_industry_mindec.rename_col(col_in='steel-graphite', col_out="graphite", dim="Categories3")

    # adjust graphite for how much graphite is used in electric arc furnace to make steel
    dm_temp = DM_fxa["min_proportion"].filter({"Variables": ["min_proportion_eu_steel_EAF"]})
    idx = dm_industry_mindec.idx
    dm_industry_mindec.array[..., idx["graphite"]] = dm_industry_mindec.array[..., idx["graphite"]] * dm_temp.array[:,
                                                                                                      :, np.newaxis,
                                                                                                      np.newaxis, :]

    # sort and rename
    dm_industry_mindec.sort("Categories3")
    dm_industry_mindec.rename_col(col_in="all-sectors", col_out="industry", dim="Categories1")

    # add industry graphite, lithium, manganese and nickel temporarely to total to compute other graphite, lithium, manganese and nickel
    dm_mindec_temp = dm_mindec.filter({"Categories3": minerals_sub2})
    dm_mindec_temp = dm_mindec_temp.filter({"Categories2": ["dir"]})
    dm_mindec_temp.add(dm_industry_mindec.array, col_label="industry", dim="Categories1")
    arr_temp = np.nansum(dm_mindec_temp.array, axis=-3, keepdims=True)
    dm_mindec_temp.drop(dim="Categories1", col_label="all-sectors")
    dm_mindec_temp.add(arr_temp, dim="Categories1", col_label="all-sectors")
    dm_mindec_temp.drop(dim="Categories1", col_label="industry")

    # multiply these total with factors for other unaccounted sectos 

    # get constants
    cdm_temp = CDM_const["minerals-unaccounted-sub2"]
    dm_temp = cdm_to_dm(cdm_temp, countries_list=dm_other_mindec.col_labels["Country"],
                        years_list=dm_other_mindec.col_labels["Years"])

    # multiply total from dm_mindec_temp times factors and add them to dm_mindec_temp
    arr_temp = dm_mindec_temp.array * dm_temp.array[..., np.newaxis, :]
    dm_mindec_temp.add(arr_temp, dim="Categories1", col_label="other")

    # add other to dm_other_mindec
    dm_temp = dm_mindec_temp.filter({"Categories1": ["other"]})
    dm_other_mindec.append(dm_temp, dim="Categories3")
    dm_other_mindec.sort("Categories3")

    # add exp and indir to other
    dm_other_mindec.add(np.nan, dim="Categories2", col_label="exp", dummy=True)
    dm_other_mindec.add(np.nan, dim="Categories2", col_label="indir", dummy=True)

    # add exp and indir, and other materials, to industry
    dm_industry_mindec.add(np.nan, dim="Categories2", col_label="exp", dummy=True)
    dm_industry_mindec.add(np.nan, dim="Categories2", col_label="indir", dummy=True)
    for i in minerals_sub1:
        dm_industry_mindec.add(np.nan, dim="Categories3", col_label=i, dummy=True)
    dm_industry_mindec.sort(dim="Categories3")

    # sum industry and other to total
    dm_mindec.append(dm_other_mindec, dim="Categories1")
    dm_mindec.append(dm_industry_mindec, dim="Categories1")
    arr_temp = np.nansum(dm_mindec.array, axis=-3, keepdims=True)
    dm_mindec.drop(dim="Categories1", col_label="all-sectors")
    dm_mindec.add(arr_temp, dim="Categories1", col_label="all-sectors")
    dm_mindec.drop(dim="Categories1", col_label="industry")
    dm_mindec.drop(dim="Categories1", col_label="other")

    # clean
    del cdm_temp, dm_temp, arr_temp, col1, col2, dm_temp2, drop, i, idx, \
        idx_cdm, idx_dm, minerals_sub1, minerals_sub2, dm_mindec_temp

    ########################
    ##### PUT TOGETHER #####
    ########################

    # put dms together
    DM_mindec = {"transport": dm_veh_mindec,
                 "infrastructure": dm_infra_mindec,
                 "domestic-appliance": dm_domapp_mindec,
                 "electronics": dm_electr_mindec,
                 "construction": dm_constr_mindec,
                 "energy": dm_energy_mindec,
                 "industry": dm_industry_mindec,
                 "other": dm_other_mindec}

    # put sectors in one dm
    dm_mindec_sect = DM_mindec["transport"].copy()
    mykey = list(DM_mindec)[1:]
    for i in mykey:
        dm_mindec_sect.append(DM_mindec[i], dim="Categories1")

    #################################
    ##### SECTORIAL PERCENTAGES #####
    #################################

    # make percentages
    # note: I have to to this here as these percenteges are done on dm_mindec before computing effifiency (not clear why)
    dm_mindec_sect.array = dm_mindec_sect.array / dm_mindec.array
    dm_mindec_sect.units['material-decomposition'] = "%"

    # make nan for indir (as at the moment percentages are not done for indir)
    idx = dm_mindec_sect.idx
    dm_mindec_sect.array[:, :, idx["material-decomposition"], :, idx["indir"], :] = np.nan

    # clean
    del idx, mykey

    ################################################
    ##### EFFICIENCY FOR MINERAL DIRECT DEMAND #####
    ################################################

    # get material-efficiency from constant and industry
    cdm_temp = CDM_const["efficiency"]
    dm_temp = DM_ind["material-efficiency"].copy()

    # expand constants and add
    dm_temp2 = cdm_to_dm(cdm_temp, countries_list=dm_temp.col_labels["Country"], years_list=dm_temp.col_labels["Years"])
    dm_temp2.deepen()
    dm_temp.append(dm_temp2, dim="Categories1")
    
    # do 1 - ind efficiency and substitute back in
    dm_temp.array = 1 - dm_temp.array

    # mutltiply total times ind efficiency and substitute back in
    dm_temp2 = dm_mindec.filter({"Categories2": ["dir"]})
    idx = dm_mindec.idx
    idx2 = dm_temp2.idx
    dm_mindec.array[:, :, :, :, idx["dir"], :] = dm_temp2.array[..., idx2["dir"], :] * dm_temp.array[:, :, np.newaxis, :]

    # clean
    del cdm_temp, dm_temp, dm_temp2, idx, idx2

    # return
    return dm_mindec, dm_mindec_sect


def mineral_demand_calibration(DM_minerals, dm_mindec):
    # get calibration series for direct demand
    dm_cal = DM_minerals["calibration"]
    dm_cal.deepen_twice()
    dm_cal.rename_col("min", "material-decomposition-calib", "Variables")
    dm_cal.rename_col("calib", "all-sectors", "Categories1")

    # get only direct demand
    dm_temp = dm_mindec.filter({"Categories2": ["dir"]})
    dm_temp = dm_temp.flatten()
    dm_temp.rename_col_regex(str1="dir_", str2="", dim="Categories2")

    # obtain calibration rates
    dm_mindec_dir_calib_rates = calibration_rates(dm=dm_temp, dm_cal=dm_cal)

    # use the same calibration rates on indirect demand, net exports and direct demand
    dm_mindec.array = dm_mindec.array * dm_mindec_dir_calib_rates.array[:, :, :, :, np.newaxis, :]

    # clean
    del dm_cal, dm_temp

    # return
    return dm_mindec, dm_mindec_dir_calib_rates


def mineral_extraction(DM_minerals, DM_ind, dm_mindec, CDM_const):
    # name of minerals
    minerals = ['aluminium', 'copper', 'graphite', 'lead', 'lithium', 'manganese', 'nickel', 'steel']

    # get data
    DM_fxa = DM_minerals['fxa']

    ###################################
    ##### MINERAL PRODUCTION (KG) #####
    ###################################

    # NOTE: in the knime this is dir - exp, here we fixed it to dir + exp.

    # mineral production at home
    dm_production = dm_mindec.copy()
    idx = dm_production.idx
    dm_production.array[..., idx["exp"]] = np.nan_to_num(dm_production.array[..., idx["exp"]])
    dm_production.operation('dir', "+", 'exp', dim="Categories2", out_col="mineral-production-home", div0="error")

    # mineral production abroad
    idx = dm_mindec.idx
    arr_temp = dm_production.array[..., idx["indir"], :]
    dm_production.add(arr_temp[..., np.newaxis, :, :], dim="Categories2", col_label="mineral-production-abroad")
    # note: in theory mineral demand = mineral produced at home + mineral produced abroad

    # clean
    dm_production.drop(dim="Categories2", col_label=["dir", "exp", "indir"])
    del idx, arr_temp

    ################################################
    ##### SCRAP USE IN MINERAL PRODUCTION (KG) #####
    ################################################

    # get variables
    dm_proportion = DM_fxa["min_proportion"].copy()
    dm_temp = DM_ind["technology-development"].copy()

    # adjust steel eaf for scraps
    idx1 = dm_proportion.idx
    idx2 = dm_temp.idx

    ay_steel_EAF = dm_proportion.array[:, :, idx1["min_proportion_eu_steel_EAF"]] \
                   + dm_temp.array[:, :, idx2['ind_proportion_steel_scrap-EAF']] \
                   - dm_temp.array[:, :, idx2['ind_proportion_steel_hydrog-DRI']] \
                   - (dm_temp.array[:, :, idx2['ind_proportion_steel_hisarna']] / 2) \
                   - dm_temp.array[:, :, idx2['ind_proportion_steel_BF-BOF']]

    dm_proportion.array[:, :, idx1["min_proportion_eu_steel_EAF"]] = ay_steel_EAF

    # adjust steel bof for scraps
    ay_steel_BOF = dm_proportion.array[:, :, idx1["min_proportion_eu_steel_BOF"]] \
                   + dm_temp.array[:, :, idx2['ind_proportion_steel_BF-BOF']] \
                   - (dm_temp.array[:, :, idx2['ind_proportion_steel_hisarna']] / 2) \
                   - dm_temp.array[:, :, idx2['ind_proportion_steel_scrap-EAF']]

    dm_proportion.array[:, :, idx1["min_proportion_eu_steel_BOF"]] = ay_steel_BOF

    # make proportion of primary copper in industry
    dm_temp.add(dm_temp.array[:, :, idx2["ind_proportion_copper_secondary"]],
                dim="Variables", col_label="ind_proportion_copper_primary", unit='%')

    # adjust proportions of aluminium and copper
    dm_proportion.array[:, :, idx1["min_proportion_eu_aluminium_primary"]] = \
        dm_proportion.array[:, :, idx1["min_proportion_eu_aluminium_primary"]] - dm_temp.array[:, :, idx2[
                                                                                                         'ind_proportion_aluminium_primary']]
    dm_proportion.array[:, :, idx1["min_proportion_eu_copper_primary"]] = \
        dm_proportion.array[:, :, idx1["min_proportion_eu_copper_primary"]] - dm_temp.array[:, :,
                                                                              idx2['ind_proportion_copper_primary']]
    dm_proportion.array[:, :, idx1["min_proportion_eu_aluminium_secondary"]] = \
        dm_proportion.array[:, :, idx1["min_proportion_eu_aluminium_secondary"]] - dm_temp.array[:, :, idx2[
                                                                                                           'ind_proportion_aluminium_secondary']]
    dm_proportion.array[:, :, idx1["min_proportion_eu_copper_secondary"]] = \
        dm_proportion.array[:, :, idx1["min_proportion_eu_copper_secondary"]] - dm_temp.array[:, :,
                                                                                idx2['ind_proportion_copper_secondary']]

    # get proportions for hisarna and dri from temp into dm_proportion
    dm_proportion.add(dm_temp.array[:, :, idx2["ind_proportion_steel_hisarna"]], dim="Variables",
                      col_label="min_proportion_eu_steel_hisarna", unit='%')
    dm_proportion.add(dm_temp.array[:, :, idx2["ind_proportion_steel_hydrog-DRI"]], dim="Variables",
                      col_label="min_proportion_eu_steel_DRI", unit='%')

    # fix dimensions of dm_production
    dm_production = dm_production.flatten()
    dm_production = dm_production.flatten()
    dm_production = dm_production.flatten()
    dm_production.rename_col_regex(str1="material-decomposition_all-sectors_mineral-production-", str2="",
                                   dim="Variables")
    dm_production.deepen()

    # fix dimensions of dm_proportion
    dm_proportion.deepen_twice()
    dm_proportion.rename_col(col_in="min_proportion_eu", col_out="home", dim="Variables")
    dm_proportion.rename_col(col_in="min_proportion_row", col_out="abroad", dim="Variables")
    dm_proportion.sort("Variables")
    dm_proportion.drop(dim="Categories1", col_label=['cobalt'])

    # clean
    del idx1, idx2

    ###################################
    ##### MINERAL EXTRACTION (KG) #####
    ###################################

    # multiply production with proportion
    dm_primsec = dm_proportion.copy()
    dm_primsec.array = dm_production.array[..., np.newaxis] * dm_proportion.array
    dm_primsec.units["home"] = "kg"
    dm_primsec.units["abroad"] = "kg"

    # get factor to keep only primary
    cdm_temp = CDM_const["factor-primary"]

    # apply factor to keep only primary
    dm_extraction = dm_primsec.copy()
    dm_extraction.array = dm_extraction.array * cdm_temp.array[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]

    # sum over prim and sec, and home and abroad
    dm_extraction.add(np.nansum(dm_extraction.array, axis=-1, keepdims=True), dim="Categories2", col_label="total-sub")
    drops = ['BOF', 'DRI', 'EAF', 'hisarna', 'primary', 'secondary']
    dm_extraction.drop(dim="Categories2", col_label=drops)
    dm_extraction.groupby({'total': '.*'}, dim='Variables', regex=True, inplace=True)

    # reshape
    dm_extraction = dm_extraction.flatten()
    dm_extraction = dm_extraction.flatten()
    dm_extraction.rename_col_regex(str1="total_", str2="", dim="Variables")
    dm_extraction.rename_col_regex(str1="_total-sub", str2="", dim="Variables")
    for i in minerals:
        dm_extraction.units[i] = "kg"

    # multiply by extraction parameters
    cdm_temp = CDM_const["factor-extraction"]
    dm_extraction.array = dm_extraction.array * cdm_temp.array[np.newaxis, np.newaxis, :]

    # rename aluminium to bauxite and steel to iron
    dm_extraction.rename_col("aluminium", "bauxite", "Variables")
    dm_extraction.rename_col("steel", "iron", "Variables")
    dm_extraction.sort("Variables")
    minerals = ['bauxite', 'copper', 'graphite', "iron", 'lead', 'lithium', 'manganese', 'nickel', 'phosphate',
                'potash']

    # convert to mt
    dm_extraction.array = dm_extraction.array * 0.000000001
    for i in minerals:
        dm_extraction.units[i] = "Mt"

    # clean
    del cdm_temp, dm_proportion, dm_temp, drops, i

    # return
    return dm_extraction


def mineral_reserves(DM_minerals, dm_mindec, dm_mindec_sect, dm_extraction, 
                     CDM_const, dm_lfs, dm_agr, dm_ref):
    # name of minerals
    minerals = ['bauxite', 'copper', 'graphite', "iron", 'lead', 'lithium', 'manganese', 'nickel', 'phosphate',
                'potash']

    # get data
    DM_fxa = DM_minerals['fxa']

    ###############################################################
    #################### MINERAL RESERVES (Mt) ####################
    ###############################################################

    # get reserves
    cdm_reserves = CDM_const["reserves"]
    dm_reserves = cdm_to_dm(cdm_reserves, countries_list=dm_extraction.col_labels["Country"],
                            years_list=dm_extraction.col_labels["Years"])

    # scale reserves by population share to make it country level
    # dm_lfs = DM_lfs.copy()
    idx_lfs = dm_lfs.idx
    arr_temp = dm_lfs.array[..., idx_lfs['lfs_population_total']] / dm_lfs.array[..., idx_lfs['lfs_macro-scenarii_iiasa-ssp2']]
    dm_reserves.array = dm_reserves.array * arr_temp[..., np.newaxis]

    # clean
    del idx_lfs, arr_temp, dm_lfs, cdm_reserves

    ########################
    ##### FOSSIL FUELS #####
    ########################

    # demand for oil, gas and coal
    dm_fossil = dm_ref.copy()
    dm_fossil.rename_col_regex(str1="fos_primary-demand_", str2="min_energy_", dim="Variables")
    idx = dm_fossil.idx
    dm_fossil.array[..., idx["min_energy_coal"]] = dm_fossil.array[..., idx["min_energy_coal"]] * 0.123
    dm_fossil.array[..., idx["min_energy_gas"]] = dm_fossil.array[..., idx["min_energy_gas"]] * 0.076
    dm_fossil.array[..., idx["min_energy_oil"]] = dm_fossil.array[..., idx["min_energy_oil"]] * 0.086
    fossils = ["coal", "gas", "oil"]
    variables = ["min_energy_" + i for i in fossils]
    for i in variables:
        dm_fossil.units[i] = "Mt"

    # # adjust min_energy_gas for ccus_gas
    # dm_gas = dm_fossil.filter({"Variables": ["min_energy_gas"]})
    # dm_ccus.array[dm_gas.array < 0] = dm_ccus.array[dm_gas.array < 0] + dm_gas.array[
    #     dm_gas.array < 0]  # in the ambitious pathway, the supply of ccus (which include biogas) is more than the demand for gaz leading to gas demand being negative. The following operation serves to correct this difference by substracting the negative gas demand by the over supply of ccus.
    # idx = dm_fossil.idx
    # dm_fossil.array[:, :, idx["min_energy_gas"], np.newaxis] = dm_fossil.array[:, :, idx["min_energy_gas"],
    #                                                            np.newaxis] + dm_ccus.array

    # relative reserves for fossil fuels 
    variables = ["min_reserve_" + i for i in fossils]
    dm_relres_fossil = dm_reserves.filter({"Variables": variables})
    dm_relres_fossil.append(dm_fossil, dim="Variables")

    # get yearly (sum across countries)
    dm_relres_fossil.add(np.nansum(dm_relres_fossil.array, axis=-3, keepdims=True), dim="Country", col_label="total")
    drops = dm_fossil.col_labels["Country"]
    dm_relres_fossil.drop(dim="Country", col_label=drops)

    # make relative reserves
    dict_relres_fossil = relative_reserve(minerals=["coal", "gas", "oil"], dm=dm_relres_fossil.copy(),
                                          reserve_starting_year=2015, mineral_type="fossil_fuel",
                                          range_max=200)

    # clean
    del idx, fossils, variables, drops

    ####################
    ##### MINERALS #####
    ####################

    # add phosphate and potash extraction
    dm_min_other = DM_fxa["min_other"].copy()

    dm_temp = dm_agr.filter_w_regex({"Variables": ".*phosphate.*"})
    dm_temp.append(dm_min_other.filter_w_regex({"Variables": ".*phosphate.*"}), dim="Variables")
    dm_temp.add(np.nansum(dm_temp.array, axis=-1, keepdims=True), dim="Variables", col_label="phosphate", unit="Mt")
    dm_extraction.append(dm_temp.filter({"Variables": ["phosphate"]}), dim="Variables")

    dm_temp = dm_agr.filter_w_regex({"Variables": ".*potash.*"})
    dm_temp.append(dm_min_other.filter_w_regex({"Variables": ".*potash.*"}), dim="Variables")
    dm_temp.add(np.nansum(dm_temp.array, axis=-1, keepdims=True), dim="Variables", col_label="potash", unit="Mt")
    dm_extraction.append(dm_temp.filter({"Variables": ["potash"]}), dim="Variables")

    # relative reserves for minerals
    variables = ["min_reserve_" + i for i in minerals]
    dm_relres_mineral = dm_reserves.filter({"Variables": variables})
    dm_temp = dm_extraction.copy()
    for i in minerals:
        dm_temp.rename_col(col_in=i, col_out="min_extraction_" + i, dim="Variables")
    dm_relres_mineral.append(dm_temp, dim="Variables")

    # get yearly (sum across countries)
    dm_relres_mineral.add(np.nansum(dm_relres_mineral.array, axis=-3, keepdims=True), dim="Country", col_label="total")
    drops = dm_extraction.col_labels["Country"]
    dm_relres_mineral.drop(dim="Country", col_label=drops)

    # make relative reserves
    dict_relres_minerals = relative_reserve(
        minerals=["bauxite", "copper", "graphite", "iron", "lead", "lithium", "manganese", "nickel", "phosphate",
                  "potash"],
        dm=dm_relres_mineral.copy(), reserve_starting_year=2015,
        mineral_type="mineral", range_max=300)

    # return
    return dict_relres_fossil, dict_relres_minerals, dm_fossil


def mineral_production_bysector(dm_mindec, dm_mindec_sect, CDM_const):
    # apply extraction parameter to total indir, exp and dir
    # note: not clear why this is applied now directly here, as before we applied after all the modifications in industry, etc ...
    cdm_temp = CDM_const["factor-extraction"]
    dm_mindec.array = dm_mindec.array * cdm_temp.array
    # note: here we apply the parameter to dm_mindec directly as for tpe we need the indir after multiplication

    # multiply exp and dir by sectoral percentages to get sectoral exp and dir (indir will be nan)
    dm_mindec_sect.array = dm_mindec.array * dm_mindec_sect.array
    dm_mindec_sect.units['material-decomposition'] = "kg"

    # NOTE: in the knime this is dir - exp, here we fixed it to dir + exp.

    # mineral production by sector
    dm_production_sect = dm_mindec_sect.copy()
    idx = dm_production_sect.idx

    dm_production_sect.array[:, :, :, :, idx["exp"], :] = np.nan_to_num(
        dm_production_sect.array[:, :, :, :, idx["exp"], :])
    dm_production_sect.operation('dir', "+", 'exp', dim="Categories2", out_col="mineral-production", div0="error")
    dm_production_sect.drop(dim="Categories2", col_label=['dir', 'exp', 'indir'])
    dm_production_sect.rename_col(col_in="aluminium", col_out="bauxite", dim="Categories3")
    dm_production_sect.rename_col(col_in="steel", col_out="iron", dim="Categories3")
    dm_production_sect.sort("Categories3")

    # clean
    del idx, cdm_temp

    # return
    return dm_production_sect


def variables_for_tpe(DM_minerals, dm_production_sect, dm_fossil, dm_mindec, dm_extraction,
                      dict_relres_fossil, dict_relres_minerals, DM_ind, dm_agr):
    # get data
    DM_fxa = DM_minerals['fxa']
    dm_min_other = DM_fxa["min_other"]

    ###########################
    ##### EXTRA MATERIALS #####
    ###########################

    # bioenergy wood
    dm_extramaterials = dm_agr.filter({"Variables": ['agr_bioenergy_biomass-demand_liquid_btl_fuelwood-and-res',
                                           'agr_bioenergy_biomass-demand_solid_fuelwood-and-res']})
    dm_extramaterials.groupby({'bioenergy_wood': '.*'}, dim='Variables', regex=True, inplace=True)

    # from industry
    dm_temp = DM_ind["material-production"]
    idx = dm_temp.idx

    # glass sand
    dm_temp.array[..., idx["ind_material-production_glass"]] = dm_temp.array[..., idx["ind_material-production_glass"]] * 1.9 / 2.4
    dm_temp.rename_col(col_in="ind_material-production_glass", col_out="glass_sand", dim="Variables")

    # timber
    dm_temp.rename_col(col_in="ind_material-production_timber", col_out="construction_wood", dim="Variables")

    # cement sand
    dm_temp.array[..., idx["ind_material-production_cement"]] = dm_temp.array[..., idx[
        "ind_material-production_cement"]] * 90 / 50
    dm_temp.rename_col(col_in="ind_material-production_cement", col_out="cement_sand", dim="Variables")

    # paper wood
    dm_temp.array[..., idx["ind_material-production_paper_woodpulp"]] = dm_temp.array[..., idx[
        "ind_material-production_paper_woodpulp"]] * 2.5
    dm_temp.rename_col(col_in="ind_material-production_paper_woodpulp", col_out="paper_wood", dim="Variables")

    dm_extramaterials.append(dm_temp, dim="Variables")

    # # gas ccus
    # dm_temp = dm_ccus.copy()
    # dm_temp.rename_col(col_in="ccu_ccus_gas-ff-natural", col_out="ccus_gas", dim="Variables")
    # dm_extramaterials.append(dm_temp, dim="Variables")

    # rename
    for i in dm_extramaterials.col_labels["Variables"]:
        dm_extramaterials.rename_col(col_in=i, col_out="min_" + i, dim="Variables")

    # clean
    del dm_temp, idx, i

    ##################################
    ##### PUT VARIABLES TOGETHER #####
    ##################################

    # extra materials
    dm_tpe = dm_extramaterials.copy()

    # mineral production by mineral and sector

    # potash and phosphate from other and agriculture
    dm_temp = dm_agr.filter({"Variables": ['agr_demand_phosphate', 'agr_demand_potash']})
    dm_temp.rename_col_regex(str1="agr_demand", str2="min_agr", dim="Variables")
    dm_tpe.append(dm_temp, dim="Variables")
    dm_temp = dm_min_other.filter({"Variables": ['min_other_phosphate', 'min_other_potash']})
    dm_tpe.append(dm_temp, dim="Variables")

    # minerals from all sectors
    dm_temp = dm_production_sect.copy()
    dm_temp.rename_col(col_in='material-decomposition', col_out="min", dim="Variables")
    dm_temp.rename_col(col_in='construction', col_out="building", dim="Categories1")
    dm_temp = dm_temp.flatten()
    dm_temp = dm_temp.flatten()
    dm_temp = dm_temp.flatten()
    dm_temp.rename_col_regex(str1="mineral-production_", str2="", dim="Variables")
    dm_temp.array = dm_temp.array * 0.000000001
    for key in dm_temp.units.keys():
        dm_temp.units[key] = "Mt"
    dm_tpe.append(dm_temp, dim="Variables")

    # fossil fuels
    dm_tpe.append(dm_fossil, dim="Variables")

    # indirect demand by mineral
    dm_temp = dm_mindec.filter({"Categories2": ["indir"]})
    dm_temp.rename_col(col_in='aluminium', col_out="bauxite", dim="Categories3")
    dm_temp.rename_col(col_in='steel', col_out="iron", dim="Categories3")
    dm_temp.sort("Categories3")
    dm_temp.rename_col(col_in='material-decomposition', col_out="min", dim="Variables")
    dm_temp.rename_col(col_in='indir', col_out="indirect", dim="Categories2")
    dm_temp.array = dm_temp.array * 0.000000001
    dm_temp.units["min"] = "Mt"
    dm_temp = dm_temp.flatten()
    dm_temp = dm_temp.flatten()
    dm_temp = dm_temp.flatten()
    dm_temp.rename_col_regex(str1="all-sectors_", str2="", dim="Variables")
    dm_tpe.append(dm_temp, dim="Variables")

    # extraction
    dm_temp = dm_extraction.copy()
    for i in dm_temp.col_labels["Variables"]:
        dm_temp.rename_col(col_in=i, col_out="min_extraction_" + i, dim="Variables")
    dm_tpe.append(dm_temp, dim="Variables")

    # sort
    dm_tpe.sort("Variables")

    # get as df
    df_tpe = dm_tpe.copy().write_df()

    # get relative reserves
    df_relres_fossil = pd.DataFrame(dict_relres_fossil, index=[0])
    df_relres_minerals = pd.DataFrame(dict_relres_minerals, index=[0])
    df_tpe_relres = pd.merge(df_relres_fossil, df_relres_minerals, how="left", on=["Country", "Years"])
    indexes = ["Country", "Years"]
    variables = df_tpe_relres.columns.tolist()
    variables = np.array(variables)[[i not in indexes for i in variables]].tolist()
    df_tpe_relres = df_tpe_relres.loc[:, indexes + variables]

    # clean
    del dm_temp, key, i, indexes, variables

    # return
    return df_tpe, df_tpe_relres


def simulate_lifestyles_to_minerals_input():
    
    dm = simulate_input(from_sector="lifestyles", to_sector="minerals")
    dm.rename_col("lfs_pop_population", "lfs_population_total", "Variables")

    return dm


def simulate_transport_to_minerals_input():
    
    dm_tra = simulate_input(from_sector="transport", to_sector="minerals")

    # demand for infrastructure [km]
    dm_tra_infra = dm_tra.filter_w_regex({"Variables": 'tra_new_infrastructure'})
    dm_tra_infra.deepen()
    dm_tra_infra.units['tra_new_infrastructure'] = 'km'

    dm_tra_veh = dm_tra.filter_w_regex({'Variables': 'product-demand'})
    dm_tra_veh.deepen()

    DM_tra = {
        'tra_infra': dm_tra_infra,
        'tra_veh': dm_tra_veh
    }
    return DM_tra


def simulate_agriculture_to_minerals_input():
    
    dm = simulate_input(from_sector="agriculture", to_sector="minerals")

    return dm


def simulate_industry_to_minerals_input():
    dm = simulate_input(from_sector="industry", to_sector="minerals")

    # # import
    # dm_imp = dm.filter_w_regex({"Variables" : ".*product-net-import.*"})

    # rename
    dict_rename = {
        "ind_prod_aluminium-pack": "ind_product-production_aluminium-pack",
        "ind_product-net-import_cars-EV": "ind_product-net-import_LDV-EV",
        "ind_product-net-import_cars-FCV": "ind_product-net-import_LDV-FCEV",
        "ind_product-net-import_cars-ICE": "ind_product-net-import_LDV-ICE",
        "ind_product-net-import_computer": "ind_product-net-import_electronics-computer",
        "ind_product-net-import_dishwasher": "ind_product-net-import_dom-appliance-dishwasher",
        "ind_product-net-import_dryer": "ind_product-net-import_dom-appliance-dryer",
        "ind_product-net-import_freezer": "ind_product-net-import_dom-appliance-freezer",
        "ind_product-net-import_fridge": "ind_product-net-import_dom-appliance-fridge",
        "ind_product-net-import_new_dhg_pipe": "ind_product-net-import_infra-pipe",
        "ind_product-net-import_phone": "ind_product-net-import_electronics-phone",
        "ind_product-net-import_planes": "ind_product-net-import_other-planes",
        "ind_product-net-import_rail": "ind_product-net-import_infra-rail",
        "ind_product-net-import_road": "ind_product-net-import_infra-road",
        "ind_product-net-import_ships": "ind_product-net-import_other-ships",
        "ind_product-net-import_trains": "ind_product-net-import_other-trains",
        "ind_product-net-import_trolley-cables": "ind_product-net-import_infra-trolley-cables",
        "ind_product-net-import_trucks-EV": "ind_product-net-import_HDVL-EV",
        "ind_product-net-import_trucks-FCV": "ind_product-net-import_HDVL-FCEV",
        "ind_product-net-import_trucks-ICE": "ind_product-net-import_HDVL-ICE",
        "ind_product-net-import_tv": "ind_product-net-import_electronics-tv",
        "ind_product-net-import_wmachine": "ind_product-net-import_dom-appliance-wmachine",
        "ind_timber": "ind_material-production_timber"
    }

    for k, v in dict_rename.items():
        dm.rename_col(k, v, dim='Variables')

    DM_ind = {}
    
    # aluminium
    DM_ind["aluminium-pack"] = dm.filter({"Variables" : ["ind_product-production_aluminium-pack"]})
    
    # material production
    DM_ind["material-production"] = dm.filter({"Variables" : 
                                               ['ind_material-production_cement', 'ind_material-production_glass', 
                                                'ind_material-production_paper_woodpulp', 'ind_material-production_timber']})
    DM_ind["material-production"].change_unit('ind_material-production_timber', factor=1e-3, old_unit='kt', new_unit='Mt')
    
    # technology development
    dm_temp = dm.filter_w_regex({"Variables" : ".*technology.*"})
    dm_temp.rename_col_regex(str1="technology-development", str2="proportion", dim="Variables")
    dm_temp.rename_col_regex(str1="copper_tech", str2="copper_secondary", dim="Variables")
    dm_temp.rename_col_regex(str1="_prim", str2="_primary", dim="Variables")
    dm_temp.rename_col_regex(str1="aluminium_sec", str2="aluminium_secondary", dim="Variables")
    DM_ind["technology-development"] = dm_temp
    
    # material efficiency
    dm_temp = dm.filter_w_regex({"Variables": ".*eff*"})
    dm_temp.deepen()
    DM_ind["material-efficiency"] = dm_temp
    
    # material switch
    dm_temp = dm.filter_w_regex({"Variables": ".*switch*"})
    dm_temp.rename_col_regex("switch_", "switch-", "Variables")
    DM_ind["material-switch"] = dm_temp
    
    # product net import
    dm_temp = dm.filter_w_regex({"Variables": ".*product-net-import*"})
    dm_temp.sort("Variables")

    DM_ind["product-net-import"] = dm_temp

    return DM_ind


def simulate_power_to_minerals_input():
    
    dm = simulate_input(from_sector="storage", to_sector="minerals")
    
    # rename
    dict_rename = {
        "str_new-capacity_battery" : "str_energy-battery",
        "elc_new-capacity_RES_other_geothermal" : "product-demand_energy-geo",
        "elc_new-capacity_RES_other_hydroelectric" : "product-demand_energy-hydro",
        "elc_new-capacity_RES_other_marine" : "product-demand_energy-marine",
        "elc_new-capacity_RES_solar_Pvroof" : "product-demand_energy-pvroof",
        "elc_new-capacity_RES_solar_Pvutility" : "product-demand_energy-pvutility",
        "elc_new-capacity_RES_solar_csp" : "product-demand_energy-csp",
        "elc_new-capacity_RES_wind_offshore" : "product-demand_energy-off-wind",
        "elc_new-capacity_RES_wind_onshore" : "product-demand_energy-on-wind",
        "elc_new-capacity_fossil_coal" : "product-demand_energy-coal",
        "elc_new-capacity_fossil_gas" : "product-demand_energy-gas",
        "elc_new-capacity_fossil_oil" : "product-demand_energy-oil",
        "elc_new-capacity_nuclear" : "product-demand_energy-nuclear"
    }

    for k, v in dict_rename.items():
        dm.rename_col(k, v, dim='Variables')

    dm.groupby({'product-demand_energy-pv': ['product-demand_energy-pvutility', 'product-demand_energy-pvroof']},
               dim='Variables', inplace=True)
        
    DM_pow = {}
    
    # battery
    DM_pow["battery"] = dm.filter_w_regex({"Variables" : ".*battery.*"})
    
    # energy
    energy = ['energy-coal', 'energy-csp', 'energy-gas', 'energy-geo', 'energy-hydro',
              'energy-marine', 'energy-nuclear', 'energy-off-wind', 'energy-oil',
              'energy-on-wind', 'energy-pv']
    find = ["product-demand_" + i for i in energy]
    dm_new_capacity = dm.filter({"Variables": find})
    dm_new_capacity.deepen()
    dm_new_capacity.sort('Categories1')
    DM_pow["energy"] = dm_new_capacity
    
    # electricity
    DM_pow["electricity-demand"] = dm.filter({"Variables" : ["elc_electricity-demand_total"]})

    return DM_pow


def simulate_buildings_to_minerals_input():

    dm = simulate_input(from_sector="buildings", to_sector="minerals")

    dict_rename = {
        'bld_appliance-new_comp': 'bld_electronics-computer',
        'bld_appliance-new_dishwasher': 'bld_dom-appliance-dishwasher',
        'bld_appliance-new_dryer': 'bld_dom-appliance-dryer',
        'bld_appliance-new_freezer': 'bld_dom-appliance-freezer',
        'bld_appliance-new_fridge': 'bld_dom-appliance-fridge',
        'bld_appliance-new_phone': 'bld_electronics-phone',
        'bld_appliance-new_tv': 'bld_electronics-tv',
        'bld_appliance-new_wmachine': 'bld_dom-appliance-wmachine',
        'bld_district-heating_new-pipe-need': 'bld_infra-pipe',
        'bld_floor-area_new_non-residential': 'bld_floor-area-new-non-residential',
        'bld_floor-area_new_residential': 'bld_floor-area-new-residential',
        'bld_floor-area_reno_non-residential': 'bld_floor-area-reno-non-residential',
        'bld_floor-area_reno_residential': 'bld_floor-area-reno-residential'
    }

    for k, v in dict_rename.items():
        dm.rename_col(k, v, dim='Variables')

    ##############################
    ##### DOMESTIC APPLIANCE #####
    ##############################

    # get domestic appliances in bld
    domapp = ['dom-appliance-dishwasher', 'dom-appliance-dryer', 'dom-appliance-freezer',
              'dom-appliance-fridge', 'dom-appliance-wmachine', ]
    find = ["bld_" + i for i in domapp]
    dm_domapp = dm.filter({"Variables": find})

    # deepen
    dm_domapp.deepen()
    dm_domapp.rename_col(col_in="bld", col_out="product-demand", dim="Variables")

    #######################
    ##### ELECTRONICS #####
    #######################

    # filter
    electr = ['electronics-computer', 'electronics-phone', 'electronics-tv']
    find = ["bld_" + i for i in electr]
    dm_electr = dm.filter(selected_cols={"Variables": find})

    # deepen
    dm_electr.deepen()
    dm_electr.rename_col(col_in="bld", col_out="product-demand", dim="Variables")

    # get infra in bld
    # !FIXME: move this to buildings
    dm_infra_temp = dm.filter({"Variables": ['bld_infra-pipe']})
    dm_infra_temp.rename_col_regex(str1="bld", str2="product-demand", dim="Variables")
    dm_infra_temp.deepen()


    ########################
    ##### CONSTRUCTION #####
    ########################

    # get floor area
    constr = ['floor-area-new-non-residential', 'floor-area-new-residential',
              'floor-area-reno-non-residential', 'floor-area-reno-residential']
    find = ["bld_" + i for i in constr]
    dm_constr = dm.filter({"Variables": find})

    # deepen
    dm_constr.deepen()
    dm_constr.rename_col(col_in="bld", col_out="product-demand", dim="Variables")

    DM_buildings = {
        'bld-pipe': dm_infra_temp,
        'bld-floor': dm_constr,
        'bld-appliance': dm_domapp,
        'bld-electr': dm_electr
    }

    return DM_buildings


def simulate_refinery_to_minerals_input():
    
    dm = simulate_input(from_sector="refinery", to_sector="minerals")

    return dm


def simulate_ccus_to_minerals_input():
    
    dm = simulate_input(from_sector="ccus", to_sector="minerals")

    return dm


def minerals(interface=Interface(), calibration=False):

    # directories
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    minerals_data_file = os.path.join(current_file_directory, '../_database/data/datamatrix/geoscale/minerals.pickle')

    # get data
    DM_minerals, CDM_const = read_data(minerals_data_file)

    # get countries
    cntr_list = DM_minerals["fxa"]["elec_new"].col_labels['Country']

    # get interfaces

    # lifestyles
    if interface.has_link(from_sector='lifestyles', to_sector='minerals'):
        dm_lfs = interface.get_link(from_sector='lifestyles', to_sector='minerals')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing lifestyles to minerals interface')
        dm_lfs = simulate_lifestyles_to_minerals_input()

    # transport
    if interface.has_link(from_sector='transport', to_sector='minerals'):
        DM_tra = interface.get_link(from_sector='transport', to_sector='minerals')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing transport to minerals interface')
        DM_tra = simulate_transport_to_minerals_input()
        for i in DM_tra.keys():
            DM_tra[i] = DM_tra[i].filter({'Country': cntr_list})

    # ! FIXME computing 2020 as an average
    # make 2020 as mean of before and after for subways, planes, ships, trains
    dm_tra_veh = DM_tra['tra_veh']
    idx = dm_tra_veh.idx
    cat = ["other-planes", "other-ships", "other-trains"]
    for i in cat:
        dm_tra_veh.array[:, idx[2020], :, idx[i]] = (dm_tra_veh.array[:, idx[2015], :, idx[i]] + \
                                                     dm_tra_veh.array[:, idx[2025], :, idx[i]]) / 2
    del idx, cat

    # agriculture
    if interface.has_link(from_sector='agriculture', to_sector='minerals'):
        dm_agr = interface.get_link(from_sector='agriculture', to_sector='minerals')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing agriculture to minerals interface')
        dm_agr = simulate_agriculture_to_minerals_input()
        dm_agr.filter({'Country': cntr_list}, inplace=True)

    # industry
    if interface.has_link(from_sector='industry', to_sector='minerals'):
        DM_ind = interface.get_link(from_sector='industry', to_sector='minerals')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing industry to minerals interface')
        DM_ind = simulate_industry_to_minerals_input()
        for i in DM_ind.keys():
            DM_ind[i] = DM_ind[i].filter({'Country': cntr_list})

    # power
    if interface.has_link(from_sector='power', to_sector='minerals'):
        DM_pow = interface.get_link(from_sector='power', to_sector='minerals')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing power to minerals interface')
        DM_pow = simulate_power_to_minerals_input()
        for i in DM_pow.keys():
            DM_pow[i] = DM_pow[i].filter({'Country': cntr_list})

    # buildings
    if interface.has_link(from_sector='buildings', to_sector='minerals'):
        DM_buildings = interface.get_link(from_sector='buildings', to_sector='minerals')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing buildings to minerals interface')
        DM_buildings = simulate_buildings_to_minerals_input()
        for i in DM_buildings.keys():
            DM_buildings[i] = DM_buildings[i].filter({'Country': cntr_list})

    # refinery
    if interface.has_link(from_sector='oil-refinery', to_sector='minerals'):
        dm_ref = interface.get_link(from_sector='oil-refinery', to_sector='minerals')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing oil-refinery to minerals interface')
        dm_ref = simulate_refinery_to_minerals_input()
        dm_ref.filter({'Country': cntr_list}, inplace=True)

    # # ccus
    # if interface.has_link(from_sector='ccus', to_sector='minerals'):
    #     dm_ccus = interface.get_link(from_sector='ccus', to_sector='minerals')
    # else:
    #     if len(interface.list_link()) != 0:
    #         print('You are missing ccus to minerals interface')
    #     dm_ccus = simulate_ccus_to_minerals_input()
    #     dm_ccus.filter({'Country': cntr_list}, inplace=True)

    # get product demand
    DM_demand = product_demand(DM_minerals, DM_buildings, DM_pow, DM_tra, CDM_const)

    # get product import
    dm_import = product_import(DM_ind)

    # get product demand split
    DM_demand_split = product_demand_split(DM_demand, dm_import, CDM_const)

    # get mineral demand split
    dm_mindec, dm_mindec_sect = mineral_demand_split(DM_minerals, DM_demand, DM_demand_split, CDM_const, DM_ind, DM_pow)

    # calibration
    if calibration is True:
        dm_mindec, dm_mindec_calib_rates = mineral_demand_calibration(DM_minerals, dm_mindec)

    # get mineral extraction
    dm_extraction = mineral_extraction(DM_minerals, DM_ind, dm_mindec, CDM_const)

    # get mineral reserves
    dict_relres_fossil, dict_relres_minerals, dm_fossil = mineral_reserves(DM_minerals, dm_mindec, dm_mindec_sect, 
                                                                           dm_extraction, CDM_const, dm_lfs, 
                                                                           dm_agr, dm_ref)

    # get mineral production by sector
    dm_production_sect = mineral_production_bysector(dm_mindec, dm_mindec_sect, CDM_const)

    # get variables for TPE
    df_tpe, df_tpe_relres = variables_for_tpe(DM_minerals, dm_production_sect, dm_fossil, dm_mindec,
                                              dm_extraction,
                                              dict_relres_fossil, dict_relres_minerals, DM_ind, dm_agr)

    return df_tpe, df_tpe_relres


def local_minerals_run():
    # geoscale
    global_vars = {'geoscale': '.*'}
    filter_geoscale(global_vars)

    # run
    results_run = minerals()

    # return
    return results_run


# # run local
# __file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/model/minerals_module.py"
# database_from_csv_to_datamatrix()
# import time
# start = time.time()
# results_run = local_minerals_run()
# end = time.time()
# print(end - start)


