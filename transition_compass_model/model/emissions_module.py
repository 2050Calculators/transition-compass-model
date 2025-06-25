#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:55:39 2024

@author: echiarot
"""

from model.common.data_matrix_class import DataMatrix
from model.common.constant_data_matrix_class import ConstantDataMatrix
from model.common.io_database import read_database_fxa, read_database_to_ots_fts_dict
from model.common.interface_class import Interface
from model.common.auxiliary_functions import filter_geoscale, cdm_to_dm, read_level_data, simulate_input, calibration_rates, cost
import pickle
import json
import os
import numpy as np
import re
import warnings
import time
from scipy import interpolate
warnings.simplefilter("ignore")

def database_from_csv_to_datamatrix():
    
    # Read database
    
    #############################
    ##### FIXED ASSUMPTIONS #####
    #############################

    # # TODO: the code below is to put unpcatured emissions in fixed assumptions (in KNIME they were in calibration), delete at the end
    # # Read calibration
    # df = read_database_fxa('emissions_calibration')
    # dm_cal = DataMatrix.create_from_df(df, num_cat=0)
    # dm_cal = dm_cal.filter({"Variables" : ['cal_clm_emissions-CH4_uncaptured', 
    #                                        'cal_clm_emissions-CO2_uncaptured', 
    #                                        'cal_clm_emissions-N2O_uncaptured']})
    # dm_temp = dm_cal
    # idx = dm_temp.idx
    # arr_temp = dm_temp.array[idx["Germany"],...]
    # dm_temp.add(arr_temp, "Country", "EU27")
    # dm_temp.add(arr_temp, "Country", "Vaud")
    # dm_temp.sort("Country")
    # dm_temp.rename_col_regex(str1 = "cal_clm_", str2 = "ems_", dim = "Variables")
    # df = dm_temp.write_df()
    # import pandas as pd
    # df = pd.melt(df, id_vars=["Country","Years"])
    # df1 = pd.DataFrame({'geoscale' : df["Country"].values,
    #                     "timescale" : df["Years"].values,
    #                     "module" : "emissions",
    #                     "eucalc-name" : df["variable"].values,
    #                     "lever" : "ems_fixed-assumptions",
    #                     "level" : 0,
    #                     "string-pivot" : "none",
    #                     "type-prefix" : "none",
    #                     "module-prefix" : "ems",
    #                     "element" : [re.split("_", i)[1] for i in df["variable"].values],
    #                     "item" : [re.split("_", i)[2] for i in df["variable"].values],
    #                     "unit" : "Mt",
    #                     "value" : df["value"].values,
    #                     "reference-id" : "missing-reference",
    #                     "interaction-file" : "ems_fixed-assumptions"})
    # current_file_directory = os.path.dirname(os.path.abspath(__file__))
    # filepath = os.path.join(current_file_directory, '../_database/data/csv/emissions_fixed-assumptions.csv')
    # df1.to_csv(filepath, sep = ";", index = False)
    
    # Read fixed assumptions to datamatrix
    df = read_database_fxa('emissions_fixed-assumptions') # weird warning as there seems to be no repeated lines
    dm_fxa = DataMatrix.create_from_df(df, num_cat=0)    
    dict_fxa = {"uncaptured-emissions" : dm_fxa}
    
    ##################
    ##### LEVERS #####
    ##################
    
    # TODO: note that ems-after-2050 is used only in the more complex computation of CO2e, which is currently not done
    # here, but it's done in KNIME. We will use it when we'll implement this more complex computation also in python.

    dict_ots = {}
    dict_fts = {}
    
    ##### emissions post 2050 #####
    
    # Set years range
    years_setting = [1990, 2015, 2100, 1]
    startyear = years_setting[0]
    baseyear = years_setting[1]
    lastyear = years_setting[2]
    step_fts = years_setting[3]
    years_ots = list(np.linspace(start=startyear, stop=baseyear, num=(baseyear-startyear)+1).astype(int)) # make list with years from 1990 to 2015
    years_fts = list(np.linspace(start=baseyear+step_fts, stop=lastyear, num=int((lastyear-baseyear)/step_fts)).astype(int)) # make list with years from 2020 to 2050 (steps of 5 years)
    years_all = years_ots + years_fts
    
    # get file
    file = 'climate_post-2050-emissions'
    lever = "ems-after-2050"
    dict_ots, dict_fts = read_database_to_ots_fts_dict(file, lever, num_cat=0, baseyear=baseyear,
                                                        years=years_all, dict_ots=dict_ots, dict_fts=dict_fts)
    
    # add EU27 and Vaud
    dm_temp = dict_ots["ems-after-2050"]
    idx = dm_temp.idx
    arr_temp = dm_temp.array[idx["Germany"],...]
    dm_temp.add(arr_temp, "Country", "EU27")
    dm_temp.add(arr_temp, "Country", "Vaud")
    dm_temp.sort("Country")
    dict_temp = dict_fts["ems-after-2050"]
    for key in dict_temp.keys():
        dm_temp = dict_temp[key]
        idx = dm_temp.idx
        arr_temp = dm_temp.array[idx["Germany"],...]
        dm_temp.add(arr_temp, "Country", "EU27")
        dm_temp.add(arr_temp, "Country", "Vaud")
        dm_temp.sort("Country")
    

    ################
    ##### SAVE #####
    ################

    DM_emissions = {
        'fxa': dict_fxa
        # 'fts': dict_fts,
        # 'ots': dict_ots,
    }

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, '../_database/data/datamatrix/emissions.pickle')
    with open(f, 'wb') as handle:
        pickle.dump(DM_emissions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # clean
    del baseyear, df, dm_fxa, DM_emissions,\
        f, handle, lastyear, startyear,\
        years_all, years_fts, years_ots, years_setting
        
    return

def read_data(data_file, lever_setting):
    
    # load dm
    with open(data_file, 'rb') as handle:
        DM_emissions = pickle.load(handle)
        
    # get fxa
    DM_fxa = DM_emissions['fxa']
        
    # # Get ots fts based on lever_setting (excluded for the moment)
    # DM_ots_fts = read_level_data(DM_emissions, lever_setting)

    # # get calibration
    # dm_cal = DM_emissions['calibration']

    # clean
    del handle, DM_emissions, data_file
    
    # return
    return DM_fxa

# FIXME: add call to this function to compute the change in temperature due to emissions.
def sum_emissions_by_gas(DM_interface):
    # NOTE: this is work in progress, to be recovered when we do temperature

    # buildings
    dm_emi = DM_interface["buildings"].filter_w_regex({"Variables": "bld.*gas-ff-natural.*|bld.*heat-ambient.*|bld.*heat-geothermal.*|bld.*heat-solar.*|bld.*liquid-ff-heatingoil.*|bld.*solid-ff-coal.*|bld.*solid-bio.*"})
    dm_emi.deepen()
    dm_emi.group_all("Categories1")
    dm_emi.deepen()
    dm_emi.add(np.nan, "Categories1", "emissions-CH4", unit="Mt", dummy=True)
    dm_emi.add(np.nan, "Categories1", "emissions-N2O", unit="Mt", dummy=True)
    dm_emi.sort("Categories1")

    # transport
    dm_tra = DM_interface["transport"].filter_w_regex({"Variables": "tra.*LDV.*|tra.*2W.*|tra.*rail.*|tra.*bus.*|tra.*metro-tram.*|tra.*aviation.*|tra.*marine.*|tra.*IWW.*|tra.*HDV.*"})
    dm_tra.deepen_twice()
    dm_tra.group_all("Categories1")
    dm_tra.group_all("Categories1")
    dm_tra.deepen()
    dm_emi.append(dm_tra, "Variables")

    # district heating
    dm_dh = DM_interface["district-heating"].filter_w_regex({"Variables": "dhg.*gas-ff-natural.*|dhg.*heat-ambient.*|dhg.*heat-geothermal.*|dhg.*heat-solar.*|dhg.*liquid-ff-heatingoil.*|dhg.*solid-ff-coal.*|dhg.*solid-bio.*"})
    dm_dh.deepen_twice()
    dm_dh.group_all("Categories2")
    dm_emi.append(dm_dh, "Variables")

    # industry
    dm_ind = DM_interface["industry"]
    dm_ind.deepen()
    dm_temp = dm_ind.filter({"Variables": ['ind_emissions-CO2', 'ind_emissions-CO2_biogenic']})
    dm_ind.drop(dim="Variables", col_label="ind_emissions-CO2_biogenic")
    idx = dm_ind.idx
    dm_ind.array[:, :, idx["ind_emissions-CO2"], :] = np.nansum(dm_temp.array, axis=-2)
    dm_ind.group_all("Categories1")
    dm_ind.deepen()
    dm_emi.append(dm_ind, "Variables")

    # ammonia
    dm_amm = DM_interface["ammonia"]
    dm_amm.deepen_twice()
    dm_amm.group_all("Categories2")
    dm_emi.append(dm_amm, "Variables")

    # land use
    dm_lus = DM_interface["land-use"]
    dm_lus.deepen_twice()
    dm_lus.group_all("Categories2")
    dm_lus.add(np.nan, "Categories1", "emissions-CH4", unit="Mt", dummy=True)
    dm_lus.add(np.nan, "Categories1", "emissions-N2O", unit="Mt", dummy=True)
    dm_lus.sort("Categories1")
    dm_emi.append(dm_lus, "Variables")

    # biodiversity
    dm_bdy = DM_interface["biodiversity"]
    dm_bdy.deepen()
    dm_emi.append(dm_bdy, "Variables")

    # agriculture
    dm_agr = DM_interface["agriculture"]
    dm_agr.deepen()
    dm_agr.group_all("Categories1")
    dm_agr.deepen()
    dm_emi.append(dm_agr, "Variables")

    # dm_agr_sub = dm_agr.filter_w_regex({"Variables": ".*crop.*"})
    # dm_agr_sub.deepen_twice()
    # dm_agr_sub.group_all("Categories2")

    # dm_agr_liv = dm_agr.filter_w_regex({"Variables": ".*liv.*"})
    # dm_agr_liv.deepen()
    # dm_agr_liv.deepen(based_on = "Variables")
    # dm_agr_liv.deepen(based_on = "Variables")
    # dm_agr_liv.group_all("Categories3")
    # dm_agr_liv.group_all("Categories2")
    # dm_agr_liv.group_all("Categories1")
    # dm_agr_liv.deepen(based_on = "Variables")
    # dm_agr_sub.append(dm_agr_liv, "Categories1")
    # dm_agr_sub.group_all("Categories1")
    # dm_agr_sub.deepen()

    # dm_agr_input = dm_agr.filter_w_regex({"Variables": ".*input.*"})
    # dm_agr_input.deepen()
    # dm_agr_input.group_all("Categories1")
    # dm_agr_input.rename_col(col_in = 'agr_input-use_emissions-CO2', col_out = 'agr_emissions-CO2_input-use', dim = "Variables")
    # dm_agr_input.deepen()
    # FIXME: the interface to power has changed, it needs to be re-done
    # electricity
    dm_elc = DM_interface["electricity"].filter({"Variables": ["elc_emissions-CO2"]})
    dm_elc.deepen()
    dm_elc.add(np.nan, "Categories1", "emissions-CH4", unit="Mt", dummy=True)
    dm_elc.add(np.nan, "Categories1", "emissions-N2O", unit="Mt", dummy=True)
    dm_emi.append(dm_elc, "Variables")

    # oil refinery
    dm_ref = DM_interface["oil-refinery"]
    dm_ref.deepen()
    dm_ref.add(np.nan, "Categories1", "emissions-CH4", unit="Mt", dummy=True)
    dm_ref.add(np.nan, "Categories1", "emissions-N2O", unit="Mt", dummy=True)
    dm_emi.append(dm_ref, "Variables")

    # sum
    variables = dm_emi.col_labels["Variables"]
    for i in variables:
        dm_emi.rename_col(i, "clm_" + i, "Variables")
    dm_emi.deepen(based_on="Variables")
    dm_emi.group_all("Categories2")

    del dm_agr, dm_amm, dm_bdy, dm_dh, dm_elc, dm_ind, dm_lus, dm_ref, dm_temp, \
        dm_tra, i, idx, variables

    return dm_emi


def emissions_equivalent(DM_interface, DM_fxa):
    
    # TODO: note that this currently works with dms all flattened (no categories), later on after all modules are finalized we can think of making it work with categories, and avoid deepen done in variables for TPE
    # put together
    dm_ems = DM_interface["buildings"].copy()
    keys = ["district-heating", "power", "land-use",
            "industry", "ammonia", "oil-refinery", "agriculture", "transport"]
    for key in keys:
        dm_ems.append(DM_interface[key], "Variables")

    # put in the uncaptured emissions
    dm_ems.append(DM_fxa["uncaptured-emissions"], "Variables")

    # linear interpolation for nans
    idx = dm_ems.idx
    countries = dm_ems.col_labels["Country"]
    variables = dm_ems.col_labels["Variables"]
    for c in countries:
        for v in variables:
            arr_temp = dm_ems.array[idx[c],:,idx[v]]
            nan_idx = np.isnan(arr_temp)
            if not nan_idx.all():  # If it's not just nans
                nans_pos = np.where(nan_idx)[0]
                nonnan_pos = np.where(~nan_idx)[0]
                nonnan = arr_temp[nonnan_pos]
                arr_temp[nan_idx] = np.interp(nans_pos, nonnan_pos, nonnan)
    
    # apply eq coefficients
    # NOTE: here in knime they were multiplying by (12 / 44) to go grom MtCO2eq to MtC/year, but as then they convert back to MtCO2eq here I compute directly MtCO2eq (so no multiplication by (12 / 44))
    GWP_N2O = 265
    GWP_CH4 = 28
    GWP_SO2 = -40.0
    dm_ems_co2 = dm_ems.filter_w_regex({"Variables" : ".*CO2.*"})
    # dm_ems_co2.array = dm_ems_co2.array * (12 / 44)
    dm_ems_n2o = dm_ems.filter_w_regex({"Variables" : ".*N2O.*"})
    dm_ems_n2o.array = dm_ems_n2o.array * GWP_N2O
    dm_ems_ch4 = dm_ems.filter_w_regex({"Variables" : ".*CH4.*"})
    dm_ems_ch4.array = dm_ems_ch4.array * GWP_CH4
    so2_any = any([re.search("SO2", i) for i in dm_ems.col_labels["Variables"]])
    if so2_any:
        dm_ems_so2 = dm_ems.filter_w_regex({"Variables" : ".*SO2.*"})
        dm_ems_so2.array = dm_ems_so2.array * GWP_SO2
    dm_ems = dm_ems_co2
    dm_ems.append(dm_ems_n2o, "Variables")
    dm_ems.append(dm_ems_ch4, "Variables")
    if so2_any:
        dm_ems.append(dm_ems_so2, "Variables")
        del dm_ems_so2
        
    # # sum to get total CO2e
    # # NOTE: this is commented out for now as modules send variables that are already aggregates / sectorial
    # # the computation of total co2e is done in variables_for_tpe().
    # dm_ems.add(np.nansum(dm_ems.array, axis = -1, keepdims=True), "Variables", "clm_total_CO2e_ems", "Mt")

    return dm_ems

def variables_for_tpe(dm_ems):
    
    # # biodiversity
    # dm_tpe = dm_ems.filter_w_regex({"Variables" : ".*bdy.*"})
    # dm_tpe.deepen()
    # dm_tpe.group_all("Categories1")
    # dm_tpe.rename_col("bdy", "bdy_emissions-CO2e", "Variables")
    
    # refinery
    dm_fos_agg = dm_ems.filter_w_regex({"Variables" : ".*fos_emissions.*"})
    dm_fos_agg.rename_col("fos_emissions-CO2","fos_emissions-CO2e","Variables")
    dm_tpe = dm_fos_agg.copy()
    
    # power bio
    dm_pow_bio = dm_ems.filter_w_regex({"Variables": "pow.*bio.*"})
    dm_ems.drop('Variables', 'pow.*bio.*')
    dm_pow_bio.groupby({'pow_emissions-CO2e_RES_bio': '.*'}, dim='Variables', inplace=True, regex=True)
    dm_tpe.append(dm_pow_bio, "Variables")
    
    # power fos
    dm_pow_fos = dm_ems.filter_w_regex({"Variables": 'pow_.*'})
    dm_pow_fos.rename_col_regex("_CO2","","Variables")
    dm_pow_fos.deepen()
    # dm_elc.group_all('Categories2')
    dm_pow_fos.rename_col('pow_emissions', 'pow_emissions-CO2e_fossil', 'Variables')
    dm_tpe.append(dm_pow_fos.flatten(), "Variables")
    
    # district heating
    dm_dhg = dm_ems.filter_w_regex({"Variables" : ".*dhg_emissions.*"})
    dm_dhg.deepen_twice()
    dm_dhg.group_all("Categories1")
    dm_dhg.rename_col("dhg", "dhg_emissions-CO2e","Variables")
    dm_temp = dm_dhg.flatten()
    dm_tpe.append(dm_temp, "Variables")
    
    # district heating aggregates
    dm_temp = dm_dhg.groupby({'added-district-heat-fossil': ['gas-ff-natural','liquid-ff-heatingoil', 
                                                             'solid-bio', 'solid-ff-coal']}, 
                             dim='Categories1', regex=False, inplace=False)
    dm_tpe.append(dm_temp.flatten(), "Variables")
    dm_temp = dm_dhg.groupby({'added-district-heat-renewable': ['heat-ambient', 'heat-geothermal', 'heat-solar']}, 
                             dim='Categories1', regex=False, inplace=False)
    dm_tpe.append(dm_temp.flatten(), "Variables")
    
    # power aggregates (pow fossiles + dhg)
    dm_pow_agg = dm_pow_fos.group_all("Categories1", inplace = False)
    dm_temp = dm_dhg.group_all("Categories1", inplace = False)
    dm_temp.rename_col("dhg_emissions-CO2e","pow_emissions-CO2e_dhg","Variables")
    dm_pow_agg.append(dm_temp, "Variables")
    dm_pow_agg.deepen()
    dm_pow_agg.group_all("Categories1")
    dm_tpe.append(dm_pow_agg, "Variables")
    
    # agriculture
    dm_agr = dm_ems.filter_w_regex({"Variables": ".*agr_emissions.*"})
    dm_agr.rename_col_regex("_","-","Variables")
    dm_agr.rename_col_regex("agr-emissions-N2O-","agr_emissions-N2O_","Variables")
    dm_agr.rename_col_regex("agr-emissions-CH4-","agr_emissions-CH4_","Variables")
    dm_agr.deepen_twice()
    dm_agr.group_all("Categories1")
    dm_agr = dm_agr.flatten()
    dm_agr.rename_col_regex("agr_", "agr_emissions-CO2e_", "Variables")
    dm_tpe.append(dm_agr, "Variables")
    # dm_agr = dm_ems.filter_w_regex({"Variables" : ".*agr_input-use.*"})
    # for i in dm_agr.col_labels["Variables"]:
    #     dm_agr.units[i] = "Mt"
    # dm_tpe.append(dm_agr, "Variables")
    
    # agriculture aggregates
    dm_agr_agg = dm_agr.groupby({"agr_emissions-CO2e" : ".*"}, dim='Variables', 
                                 regex=True, inplace=False)
    dm_tpe.append(dm_agr_agg, "Variables")
    
    # land use system
    dm_lus = dm_ems.filter_w_regex({"Variables" : ".*lus_emissions.*"})
    dm_lus.deepen()
    dm_lus_agg = dm_lus.group_all("Categories1", inplace = False)
    dm_lus_agg.rename_col("lus_emissions-CO2", "lus_emissions-CO2e", "Variables")
    dm_tpe.append(dm_lus_agg, "Variables")
    
    # industry
    dm_ind = dm_ems.filter_w_regex({"Variables" : ".*ind_emissions.*"})
    dm_ind.deepen()
    dm_ind.drop("Variables", ['ind_emissions-CO2_biogenic'])
    dm_ind.deepen(based_on="Variables")
    dm_ind.group_all("Categories2")
    dm_ind.rename_col("ind","ind_emissions-CO2e","Variables")
    dm_tpe.append(dm_ind.flatten(), "Variables")
    
    # industry biogenic
    dm_ind_biogen = dm_ems.filter_w_regex({"Variables" : ".*ind_emissions-CO2_biogenic.*"})
    dm_ind_biogen.rename_col_regex('CO2', 'CO2e', "Variables")
    dm_tpe.append(dm_ind_biogen, "Variables")
    
    # ammonia
    dm_amm = dm_ems.filter_w_regex({"Variables" : ".*amm.*"})
    dm_amm.rename_col_regex("_ammonia", "", "Variables")
    dm_amm.deepen()
    dm_amm.group_all("Categories1")
    dm_amm.rename_col("amm", "amm_emissions-CO2e", "Variables")
    dm_tpe.append(dm_amm, "Variables")
    
    # industry aggregates
    dm_ind_agg = dm_amm.copy()
    dm_ind_agg.rename_col("amm_emissions-CO2e","ind_emissions-CO2e_amm","Variables")
    dm_ind_agg.deepen()
    dm_ind_agg.append(dm_ind, "Categories1")
    dm_ind_agg.group_all("Categories1")
    dm_tpe.append(dm_ind_agg, "Variables")
    
    # biogen aggregates
    dm_clm_biogen_agg = dm_ind_biogen.copy()
    dm_clm_biogen_agg.append(dm_pow_bio, "Variables")
    dm_clm_biogen_agg.groupby({"clm_emissions-CO2e_biogenic" : ".*"}, "Variables", regex = True, inplace = True)
    dm_tpe.append(dm_clm_biogen_agg, "Variables")
    
    # transport
    dm_tra = dm_ems.filter_w_regex({"Variables" : ".*tra_emissions.*"})
    dm_tra.deepen()
    dm_tra.group_all(dim='Categories1')
    dm_tra.rename_col_regex('emissions', 'emissions-CO2e', dim='Variables')
    dm_tra.groupby({'tra_emissions-CO2e_freight_HDV': '.*HDV.*'}, dim='Variables', regex=True, inplace=True)
    dm_tpe.append(dm_tra, "Variables")
    
    # transport aggregate
    dm_tra.deepen_twice()
    dm_tra.group_all("Categories2")
    dm_tpe.append(dm_tra.flatten(), "Variables")
    dm_tra_agg = dm_tra.group_all("Categories1", inplace = False)
    dm_tpe.append(dm_tra_agg, "Variables")
    
    # buildings
    # Emissions by fuel type
    dm_bld = dm_ems.filter_w_regex({"Variables": "bld_emissions-CO2.*"})
    dm_bld.rename_col_regex("CO2", "CO2e", "Variables")
    dm_tpe.append(dm_bld, "Variables")
    
    # buildings aggregates
    dm_bld_agg = dm_bld.groupby({'bld_emissions-CO2e': 
                    ["bld_emissions-CO2e_gas-ff-natural", "bld_emissions-CO2e_heat-ambient",
                     "bld_emissions-CO2e_heat-geothermal", "bld_emissions-CO2e_heat-solar",
                     "bld_emissions-CO2e_liquid-ff-heatingoil", "bld_emissions-CO2e_solid-bio",
                     "bld_emissions-CO2e_solid-ff-coal"]}, dim='Variables', inplace=False, regex=False)
    dm_tpe.append(dm_bld_agg, "Variables")
    
    # total
    DM_total = {"lus": dm_lus_agg, # lus
                # "agr" : dm_agr_agg, # agr
                "fos": dm_fos_agg, # fos (oil refinery)
                "pow": dm_pow_agg, # pow fos + dhg
                "tra": dm_tra_agg, # tra
                "bld": dm_bld_agg, # bld
                "ind": dm_ind_agg, # ind (includes amm)
                "ind_biogen": dm_clm_biogen_agg} # ind biogenic

    dm_tot = dm_agr_agg.copy()
    for key in DM_total.keys():
        dm_tot.append(DM_total[key],"Variables")
    dm_tot.groupby({"emissions-CO2e" : ".*"}, "Variables", regex=True, inplace=True)
    dm_tpe.append(dm_tot, "Variables")
    
    # sort
    dm_tpe.sort("Variables")
        
    return dm_tpe

def simulate_buildings_to_emissions_input():
    
    dm = simulate_input(from_sector="buildings", to_sector="emissions")
    dm.rename_col("bld_residential-emissions-CO2_non_appliances", "bld_emissions-CO2_appliances_non-residential", "Variables")
    dm.rename_col("bld_residential-emissions-CO2_appliances", "bld_emissions-CO2_appliances_residential", "Variables")
    dm = dm.filter({"Variables" : ["bld_emissions-CO2_gas-ff-natural", "bld_emissions-CO2_heat-ambient",
                                   "bld_emissions-CO2_heat-geothermal", "bld_emissions-CO2_heat-solar", 
                                   "bld_emissions-CO2_liquid-ff-heatingoil", "bld_emissions-CO2_solid-bio", 
                                   "bld_emissions-CO2_solid-ff-coal", "bld_emissions-CO2_appliances_non-residential"]})

    
    return dm

def simulate_district_heating_to_emissions_input():
    
    dm = simulate_input(from_sector="district-heating", to_sector="emissions")
    
    return dm

def simulate_power_to_emissions_input():
    
    dm = simulate_input(from_sector="power", to_sector="emissions")

    # drop variables that are already aggregated
    # TODO: in DM_interface["power"] I have dropped "elc_emissions-CO2_fossil_total" to avoid to double counting in the overall sum, to be reported in the known issues
    dm.drop("Variables", ['elc_stored-CO2_RES_bio_gas', 'elc_stored-CO2_RES_bio_mass', 'elc_emissions-CO2_fossil_total'])

    dm.rename_col_regex('RES_bio', 'biogas', 'Variables')
    dm.rename_col_regex('fossil_', '', 'Variables')
    dm.rename_col_regex('natural-gas', 'gas', 'Variables')
    dm.rename_col_regex('elc', 'pow', 'Variables')
    dm.rename_col_regex('emissions-', 'emissions_', 'Variables')
    dm.deepen_twice()
    dm.switch_categories_order('Categories1', 'Categories2')
    dm = dm.flatten()
    dm = dm.flatten()
    return dm

def simulate_refinery_to_emissions_input():
    
    dm = simulate_input(from_sector="refinery", to_sector="emissions")
    
    return dm

def simulate_agriculture_to_emissions_input():
    
    dm = simulate_input(from_sector="agriculture", to_sector="emissions")
    # import pprint
    # dm.sort("Variables")
    # pprint.pprint(dm.col_labels["Variables"])
    
    return dm

def simulate_land_use_to_emissions_input():
    
    dm = simulate_input(from_sector="land-use", to_sector="emissions")
    dm = dm.filter({"Variables" : ['lus_emissions-CO2_land-to-cropland', 'lus_emissions-CO2_land-to-forest', 
                              'lus_emissions-CO2_land-to-grassland', 'lus_emissions-CO2_land-to-other', 
                              'lus_emissions-CO2_land-to-settlement', 'lus_emissions-CO2_land-to-wetland']})
    # NOTE: in knime we have 12 variables here, while here we consider only 6, as emissions in landuse contain
    # already the remaining emissions
    
    return dm

def simulate_biodiversity_to_emissions_input():
    
    dm = simulate_input(from_sector="biodiversity", to_sector="emissions")
    
    return dm

def simulate_industry_to_emissions_input():
    
    dm = simulate_input(from_sector="industry", to_sector="emissions")
    
    return dm

def simulate_ammonia_to_emissions_input():
    
    dm = simulate_input(from_sector="ammonia", to_sector="emissions")
    
    return dm

def simulate_transport_to_emissions_input():
    
    dm = simulate_input(from_sector="transport", to_sector="emissions")
    
    return dm


def emissions(lever_setting, years_setting, interface = Interface(), calibration = False):
    
    # emissions data file
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    emissions_data_file = os.path.join(current_file_directory, '../_database/data/datamatrix/geoscale/emissions.pickle')
    DM_fxa = read_data(emissions_data_file, lever_setting)

    cntr_list = DM_fxa['uncaptured-emissions'].col_labels['Country']

    # get / simulate interfaces
    DM_interface = {}
        
    if interface.has_link(from_sector = "buildings", to_sector = 'emissions'):
        DM_interface["buildings"] = interface.get_link(from_sector="buildings", to_sector='emissions')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing buildings to emissions interface')
        DM_interface["buildings"] = simulate_buildings_to_emissions_input()
        DM_interface['buildings'].filter({'Country': cntr_list}, inplace=True)
        
    if interface.has_link(from_sector = "district-heating", to_sector = 'emissions'):
        DM_interface["district-heating"] = interface.get_link(from_sector="district-heating", to_sector='emissions')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing district-heating to emissions interface')
        DM_interface["district-heating"] = simulate_district_heating_to_emissions_input()
        DM_interface['district-heating'].filter({'Country': cntr_list}, inplace=True)

    if interface.has_link(from_sector = "power", to_sector = 'emissions'):
        DM_interface["power"] = interface.get_link(from_sector="power", to_sector='emissions')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing power to emissions interface')
        DM_interface["power"] = simulate_power_to_emissions_input()
        DM_interface['power'].filter({'Country': cntr_list}, inplace=True)

    if interface.has_link(from_sector = "oil-refinery", to_sector = 'emissions'):
        DM_interface["oil-refinery"] = interface.get_link(from_sector="oil-refinery", to_sector='emissions')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing refinery to emissions interface')
        DM_interface["oil-refinery"] = simulate_refinery_to_emissions_input()
        DM_interface['oil-refinery'].filter({'Country': cntr_list}, inplace=True)
    
    if interface.has_link(from_sector = "agriculture", to_sector = 'emissions'):
        DM_interface["agriculture"] = interface.get_link(from_sector="agriculture", to_sector='emissions')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing agriculture to emissions interface')
        DM_interface["agriculture"] = simulate_agriculture_to_emissions_input()
        DM_interface['agriculture'].filter({'Country': cntr_list}, inplace=True)
    
    if interface.has_link(from_sector = "land-use", to_sector = 'emissions'):
        DM_interface["land-use"] = interface.get_link(from_sector="land-use", to_sector='emissions')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing land-use to emissions interface')
        DM_interface["land-use"] = simulate_land_use_to_emissions_input()
        DM_interface['land-use'].filter({'Country': cntr_list}, inplace=True)
        
    if interface.has_link(from_sector = "biodiversity", to_sector = 'biodiversity'):
        DM_interface["biodiversity"] = interface.get_link(from_sector="biodiversity", to_sector='emissions')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing biodiversity to emissions interface')
        DM_interface["biodiversity"] = simulate_biodiversity_to_emissions_input()
        DM_interface['biodiversity'].filter({'Country': cntr_list}, inplace=True)

    if interface.has_link(from_sector = "industry", to_sector = 'emissions'):
        DM_interface["industry"] = interface.get_link(from_sector="industry", to_sector='emissions')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing industry to emissions interface')
        DM_interface["industry"] = simulate_industry_to_emissions_input()
        DM_interface['industry'].filter({'Country': cntr_list}, inplace=True)
    
    if interface.has_link(from_sector = "ammonia", to_sector = 'emissions'):
        DM_interface["ammonia"] = interface.get_link(from_sector="ammonia", to_sector='emissions')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing ammonia to emissions interface')
        DM_interface["ammonia"] = simulate_ammonia_to_emissions_input()
        DM_interface['ammonia'].filter({'Country': cntr_list}, inplace=True)
        
    if interface.has_link(from_sector = "transport", to_sector = 'emissions'):
        DM_interface["transport"] = interface.get_link(from_sector="transport", to_sector='emissions')
    else:
        if len(interface.list_link()) != 0:
            print('You are missing transport to emissions interface')
        DM_interface["transport"] = simulate_transport_to_emissions_input()
        DM_interface['transport'].filter({'Country': cntr_list}, inplace=True)

    # get emissions for gas equivalent
    dm_ems = emissions_equivalent(DM_interface, DM_fxa)
    dm_ems.sort("Variables")
    
    # get variables for tpe
    dm_tpe = variables_for_tpe(dm_ems)
    
    results_run = dm_tpe.write_df()

    # return
    return results_run

def local_emissions_run():
    
    # get years and lever setting
    years_setting = [1990, 2015, 2100, 1]
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(current_file_directory, '../config/lever_position.json'))
    lever_setting = json.load(f)[0]
    
    # get geoscale
    global_vars = {'geoscale': '.*'}
    filter_geoscale(global_vars)

    # run
    results_run = emissions(lever_setting, years_setting)
    
    # return
    return results_run

# # run local
# __file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/model/emissions_module.py"
# # database_from_csv_to_datamatrix()
# start = time.time()
# results_run = local_emissions_run()
# end = time.time()
# print(end-start)

