import pandas as pd

from model.common.data_matrix_class import DataMatrix
from model.common.constant_data_matrix_class import ConstantDataMatrix
from model.common.io_database import dm_to_database
from model.common.interface_class import Interface
from model.common.auxiliary_functions import  calibration_rates, create_years_list
from model.common.auxiliary_functions import read_level_data, filter_country_and_load_data_from_pickles
import pickle
import json
import os
import numpy as np
import time


def init_years_lever():
  # function that can be used when running the module as standalone to initialise years and levers
  years_setting = [1990, 2023, 2025, 2050, 5]
  f = open('../config/lever_position.json')
  lever_setting = json.load(f)[0]
  return years_setting, lever_setting


# CalculationLeaf READ PICKLE
def read_data(DM_TCAF, lever_setting):

    DM_ots_fts = read_level_data(DM_TCAF, lever_setting)

    # FXA data matrix
    dm_fxa_cal_diet = DM_agriculture['fxa']['cal_agr_diet']
    dm_fxa_cal_liv_prod = DM_agriculture['fxa']['cal_agr_domestic-production-liv']
    dm_fxa_cal_liv_pop = DM_agriculture['fxa']['cal_agr_liv-population']
    dm_fxa_cal_liv_CH4 = DM_agriculture['fxa']['cal_agr_liv_CH4-emission']
    dm_fxa_cal_liv_N2O = DM_agriculture['fxa']['cal_agr_liv_N2O-emission']
    dm_fxa_cal_demand_feed = DM_agriculture['fxa']['cal_agr_demand_feed']
    # dm_fxa_cal_land = DM_agriculture['fxa']['cal_agr_lus_land']
    dm_fxa_ef_liv_N2O = DM_agriculture['fxa']['ef_liv_N2O-emission']
    dm_fxa_ef_liv_CH4_treated = DM_agriculture['fxa']['ef_liv_CH4-emission_treated']
    dm_fxa_liv_nstock = DM_agriculture['fxa']['liv_manure_n-stock']

    # Extract sub-data-matrices according to the flow
    # Sub-matrix for LIFESTYLE
    # dm_demography = DM_ots_fts['pop']['lfs_demography_']
    dm_diet_requirement = DM_ots_fts['kcal-req']
    dm_diet_split = DM_ots_fts['diet']['lfs_consumers-diet']
    dm_diet_share = DM_ots_fts['diet']['share']
    dm_diet_fwaste = DM_ots_fts['fwaste']
    # dm_population = DM_ots_fts['pop']['lfs_population_']

    # Sub-matrix for the FOOD DEMAND
    dm_food_net_import_pro = DM_ots_fts['food-net-import'].filter_w_regex(
        {'Categories1': 'pro-.*', 'Variables': 'agr_food-net-import'})

    # Sub-matrix for LIVESTOCK
    dm_livestock_losses = DM_ots_fts['climate-smart-livestock']['climate-smart-livestock_losses']
    dm_livestock_yield = DM_ots_fts['climate-smart-livestock']['climate-smart-livestock_yield']
    dm_livestock_slaughtered = DM_ots_fts['climate-smart-livestock']['climate-smart-livestock_slaughtered']
    dm_livestock_density = DM_ots_fts['climate-smart-livestock']['climate-smart-livestock_density']
    dm_fxa_ratio_milk = DM_agriculture['fxa']['ratio_milk']

    # Sub-matrix for ALCOHOLIC BEVERAGES
    dm_alc_bev = DM_ots_fts['biomass-hierarchy']['biomass-hierarchy-bev-ibp-use-oth']


    # Sub-matrix for BIOENERGY
    dm_bioenergy_cap_load_factor = DM_ots_fts['bioenergy-capacity']['bioenergy-capacity_load-factor']
    dm_bioenergy_cap_bgs_mix = DM_ots_fts['bioenergy-capacity']['bioenergy-capacity_bgs-mix']
    dm_bioenergy_cap_efficiency = DM_ots_fts['bioenergy-capacity']['bioenergy-capacity_efficiency']
    dm_bioenergy_cap_liq = DM_ots_fts['bioenergy-capacity']['bioenergy-capacity_liq_b']
    dm_bioenergy_cap_elec = DM_ots_fts['bioenergy-capacity']['bioenergy-capacity_elec']
    dm_bioenergy_mix_digestor = DM_ots_fts['biomass-hierarchy']['biomass-hierarchy_biomass-mix_digestor']
    dm_bioenergy_mix_solid = DM_ots_fts['biomass-hierarchy']['biomass-hierarchy_biomass-mix_solid']
    dm_bioenergy_mix_liquid = DM_ots_fts['biomass-hierarchy']['biomass-hierarchy_biomass-mix_liquid']
    dm_bioenergy_liquid_biodiesel = DM_ots_fts['biomass-hierarchy']['biomass-hierarchy_bioenergy_liquid_biodiesel']
    dm_bioenergy_liquid_biogasoline = DM_ots_fts['biomass-hierarchy']['biomass-hierarchy_bioenergy_liquid_biogasoline']
    dm_bioenergy_liquid_biojetkerosene = DM_ots_fts['biomass-hierarchy'][
        'biomass-hierarchy_bioenergy_liquid_biojetkerosene']
    dm_bioenergy_cap_elec.append(dm_bioenergy_cap_load_factor, dim='Variables')
    dm_bioenergy_cap_elec.append(dm_bioenergy_cap_efficiency, dim='Variables')

    # Sub-matrix for LIVESTOCK MANURE MANGEMENT & GHG EMISSIONS
    dm_livestock_enteric_emissions = DM_ots_fts['climate-smart-livestock']['climate-smart-livestock_enteric']
    dm_livestock_manure = DM_ots_fts['climate-smart-livestock']['climate-smart-livestock_manure']

    # Sub-matrix for FEED
    dm_ration = DM_ots_fts['climate-smart-livestock']['climate-smart-livestock_ration']
    dm_alt_protein = DM_ots_fts['alt-protein']
    dm_ruminant_feed = DM_ots_fts['ruminant-feed']

    # Sub-matrix for CROP
    dm_food_net_import_crop = DM_ots_fts['food-net-import'].filter_w_regex({'Categories1': 'crop-.*',
                                                                            'Variables': 'agr_food-net-import'})  # filtered here on purpose and not in the pickle (other parts of the datamatrix are used)
    dm_food_net_import_crop.rename_col_regex(str1="crop-", str2="", dim="Categories1")
    dm_crop = DM_ots_fts['climate-smart-crop']['climate-smart-crop_losses']
    #dm_food_net_import_crop.drop(dim='Categories1', col_label=['stm'])
    dm_crop.append(dm_food_net_import_crop, dim='Variables')
    dm_residues_yield = DM_agriculture['fxa']['residues_yield']
    dm_hierarchy_residues_cereals = DM_ots_fts['biomass-hierarchy']['biomass-hierarchy_crop_cereal']
    dm_cal_crop = DM_agriculture['fxa']['cal_agr_domestic-production_food']
    dm_cal_crop_bev = DM_agriculture['fxa']['cal_agr_domestic-production_bev']
    # dm_crop.append(dm_cal_crop, dim='Variables')
    dm_ef_residues = DM_agriculture['fxa']['ef_burnt-residues']
    dm_ssr_feed_crop = DM_ots_fts['climate-smart-crop']['feed-net-import']
    dm_processing_yield = DM_agriculture['fxa']['processing-yield']

    # Sub-matrix for LAND
    dm_cal_land = DM_agriculture['fxa']['cal_agr_lus_land']
    dm_yield = DM_ots_fts['climate-smart-crop']['climate-smart-crop_yield']
    dm_fibers = DM_agriculture['fxa']['fibers']
    dm_rice = DM_agriculture['fxa']['rice']
    dm_cal_cropland = DM_agriculture['fxa']['cal_agr_lus_land_cropland']

    # Sub-matrix for NITROGEN BALANCE
    dm_input = DM_ots_fts['climate-smart-crop']['climate-smart-crop_input-use']
    dm_fertilizer_emission = DM_agriculture['fxa']['agr_emission_fertilizer']
    dm_cal_n = DM_agriculture['fxa']['cal_agr_crop_emission_N2O-emission_fertilizer']
    # dm_fertilizer_emission.append(dm_cal_n, dim='Variables')

    # Sub-matrix for ENERGY & GHG EMISSIONS
    dm_cal_energy_demand = DM_agriculture['fxa']['cal_agr_energy-demand']
    dm_energy_demand = DM_ots_fts['climate-smart-crop']['climate-smart-crop_energy-demand']
    dm_cal_GHG = DM_agriculture['fxa']['cal_agr_emissions']
    dm_cal_GHG.deepen()
    dm_cal_input = DM_agriculture['fxa']['cal_agr_input-use_emissions-CO2']

    # Aggregated Data Matrix - ENERGY & GHG EMISSIONS
    DM_energy_ghg = {
        'energy_demand': dm_energy_demand,
        'cal_energy_demand': dm_cal_energy_demand,
        'cal_input': dm_cal_input,
        'cal_GHG': dm_cal_GHG
    }

    # Aggregate Data Matrix - LIFESTYLE
    DM_lifestyle = {
        'energy-requirement': dm_diet_requirement,
        'diet-split': dm_diet_split,
        'diet-share': dm_diet_share,
        'diet-fwaste': dm_diet_fwaste,
        # 'demography': dm_demography,
        # 'population': dm_population,
        'cal_diet': dm_fxa_cal_diet
    }

    # Aggregated Data Matrix - FOOD DEMAND
    DM_food_demand = {
        'food-net-import-pro': dm_food_net_import_pro
    }

    # Aggregated Data Matrix - LIVESTOCK
    DM_livestock = {
        'losses': dm_livestock_losses,
        'yield': dm_livestock_yield,
        'liv_slaughtered_rate': dm_livestock_slaughtered,
        'cal_liv_prod': dm_fxa_cal_liv_prod,
        'cal_liv_population': dm_fxa_cal_liv_pop,
        'ruminant_density': dm_livestock_density,
        'ratio_milk': dm_fxa_ratio_milk
    }

    # Aggregated Data Matrix - ALCOHOLIC BEVERAGES
    DM_alc_bev = {
        'biomass_hierarchy': dm_alc_bev,
        'processing-yields': dm_processing_yield
    }

    # Aggregated Data Matrix - BIOENERGY
    DM_bioenergy = {
        'electricity_production': dm_bioenergy_cap_elec,
        'bgs-mix': dm_bioenergy_cap_bgs_mix,
        'liq': dm_bioenergy_cap_liq,
        'digestor-mix': dm_bioenergy_mix_digestor,
        'solid-mix': dm_bioenergy_mix_solid,
        'liquid-mix': dm_bioenergy_mix_liquid,
        'liquid-biodiesel': dm_bioenergy_liquid_biodiesel,
        'liquid-biogasoline': dm_bioenergy_liquid_biogasoline,
        'liquid-biojetkerosene': dm_bioenergy_liquid_biojetkerosene
    }

    # Aggregated Data Matrix - LIVESTOCK MANURE MANAGEMENT & GHG EMISSIONS
    DM_manure = {
        'enteric_emission': dm_livestock_enteric_emissions,
        'manure': dm_livestock_manure,
        'cal_liv_CH4': dm_fxa_cal_liv_CH4,
        'cal_liv_N2O': dm_fxa_cal_liv_N2O,
        'ef_liv_N2O': dm_fxa_ef_liv_N2O,
        'ef_liv_CH4_treated': dm_fxa_ef_liv_CH4_treated,
        'liv_n-stock': dm_fxa_liv_nstock
    }

    # Aggregated Data Matrix - FEED
    DM_feed = {
        'ration': dm_ration,
        'alt-protein': dm_alt_protein,
        'cal_agr_demand_feed': dm_fxa_cal_demand_feed,
        'ruminant-feed': dm_ruminant_feed
    }

    # Aggregated Data Matrix - CROP
    DM_crop = {
        'crop': dm_crop,
        'cal_crop': dm_cal_crop,
        'cal_bev': dm_cal_crop_bev,
        'ef_residues': dm_ef_residues,
        'residues_yield': dm_residues_yield,
        'hierarchy_residues_cereals': dm_hierarchy_residues_cereals,
        'food-net-import-pro': dm_food_net_import_pro,
        'feed-net-import_crop': dm_ssr_feed_crop,
        'processing-yields': dm_processing_yield
    }

    # Aggregated Data Matrix - LAND
    DM_land = {
        'cal_land': dm_cal_land,
        'cal_cropland': dm_cal_cropland,
        'yield': dm_yield,
        'fibers': dm_fibers,
        'rice': dm_rice
    }

    # Aggregated Data Matrix - NITROGEN BALANCE
    DM_nitrogen = {
        'input': dm_input,
        'emissions': dm_fertilizer_emission,
        'cal_n': dm_cal_n
    }

    CDM_const = DM_agriculture['constant']

    return DM_ots_fts

# SimulateInteractions

def simulate_lifestyles_to_agriculture_input_new():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../_database/data/interface/lifestyles_to_agriculture.pickle")
    with open(f, 'rb') as handle:
        DM_lfs = pickle.load(handle)

    return DM_lfs


#def TCAF_health_diet_TPE_interface():
#  return dm_tpe



def TCAF_health_diet(lever_setting, years_setting, DM_input, interface=Interface()):

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    DM_TCAF = read_data(DM_input, lever_setting)
    country_list = ['Switzerland']

    # INTERFACES IN ---------------------------------------------------------------------------------------------------

    # CalculationLeaf Link interface or Simulate data from other modules
    if interface.has_link(from_sector='dietary-habits', to_sector='TCAF_health-diet'):
      DM_diet = interface.get_link(from_sector='dietary-habits', to_sector='TCAF_health-diet')
    else:
      if len(interface.list_link()) != 0:
        print('You are missing lifestyles to agriculture interface')
      DM_diet = simulate_lifestyles_to_agriculture_input_new()
      for key in DM_diet.keys():
        DM_diet[key].filter({'Country': country_list}, inplace=True)


    # CalculationTree ---------------------------------------------------------------------------------------------------
    results_run = DM_diet.copy()

    # INTERFACES OUT ---------------------------------------------------------------------------------------------------

    # interface to Land use
    #DM_lus = agriculture_landuse_interface(DM_bioenergy, dm_lgn, dm_land_use)
    #interface.add_link(from_sector='agriculture', to_sector='land-use',
    #                   dm=DM_lus)

    # TPE OUTPUT -------------------------------------------------------------------------------------------------------
    #results_run = agriculture_TPE_interface()

    return results_run


def TCAF_health_diet_local_run():
  country_list = ['Switzerland']
  DM_input = filter_country_and_load_data_from_pickles \
    (country_list= country_list, modules_list = 'TCAF')
  years_setting, lever_setting = init_years_lever()
  TCAF_health_diet(lever_setting, years_setting, DM_input['TCAF'])
  return

if __name__ == "__main__":
  TCAF_health_diet_local_run()
