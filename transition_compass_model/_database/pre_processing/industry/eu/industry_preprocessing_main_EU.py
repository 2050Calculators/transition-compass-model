from processors.industry_calib_emissions import run as calib_emissions_run
from processors.industry_calib_energy_demand import run as calib_energy_demand_run
from processors.industry_calib_material_production import (
    run as calib_material_production_run,
)
from processors.industry_const_emission_factors import run as const_emission_factors_run
from processors.industry_const_energy_demand import run as const_energy_demand_run
from processors.industry_const_material_decomposition import (
    run as const_material_decomposition_run,
)
from processors.industry_const_material_switch_ratio import (
    run as const_materia_switch_ratios_run,
)
from processors.industry_fxa_costs import run as costs_run
from processors.industry_fxa_energy_demand import run as fxa_energy_demand_run
from processors.industry_lever_carbon_capture import run as carbon_capture_run
from processors.industry_lever_energy_switch import run as energy_switch_run
from processors.industry_lever_material_efficiency import run as material_efficiency_run
from processors.industry_lever_material_net_import import run as material_net_import_run
from processors.industry_lever_material_recovery import run as material_recovery_run
from processors.industry_lever_material_switch import run as material_switch_run
from processors.industry_lever_product_net_import import run as product_net_import_run
from processors.industry_lever_technology_development import (
    run as technology_development_run,
)
from processors.industry_lever_technology_share import run as technology_share_run
from processors.industry_lever_waste_management import run as waste_management_run
from processors.industry_ots_pickle import run as ots_pickle_run

# from processors.ammonia_make_dm import run as make_ammonia_dms
from processors.industry_pre_processing_save import save_industry_pre_processing_run
from scenarios.industry_fts_BAU_pickle import run as fts_bau_pickle_run

from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
)

# years
years_ots = create_years_list(1990, 2023, 1)
years_fts = create_years_list(2025, 2050, 5)

# Material switch (lever)
print("Material switch")
dm_mat_switch = material_switch_run(years_ots)

# Material efficiency (lever)
print("Material efficiency")
dm_mat_eff = material_efficiency_run(years_ots)

# Technology development (lever)
print("Technology development")
dm_tech_dev = technology_development_run(years_ots)

# Carbon capture (lever)
print("Carbon capture")
dm_cc = carbon_capture_run(years_ots)

# Technology share (lever)
print("Techbology share")
dm_tech_share = technology_share_run(years_ots)

# Material net import share, material production fxa, material demand fxa
print("Material net import share, material production fxa, material demand fxa")
dm_trade_netshare, dm_matprod_fxa, dm_matdem_fxa = material_net_import_run(
    years_ots, years_fts
)

# Product net import share, packaging per capita
print("Product net import share, packaging per capita")
dm_trade_netshare_prod, dm_pack = product_net_import_run(years_ots)

# Energy switch (lever)
print("Energy switch")
dm_ene_switch = energy_switch_run(years_ots)

# Waste management (lever)
print("Waste management")
dm_wst_management = waste_management_run(years_ots)

# Material recovery (lever)
print("Material recovery")
dm_material_recovery = material_recovery_run(years_ots)

# Costs (fxa)
print("Costs")
dm_costs, dm_costs_cc = costs_run(years_ots)

# Energy demand (fxa)
print("Energy demand fxa")
dm_fxa_energy_excl, dm_fxa_energy_feed = fxa_energy_demand_run(years_ots, years_fts)

# Calibration emissions (calib)
print("Calibration emissions")
dm_calib_emissions, dm_calib_emissions_ammonia = calib_emissions_run(
    years_ots, years_fts
)

# Calibration energy demand (calib)
print("Calibration energy demand")
dm_calib_energy_demand = calib_energy_demand_run(years_ots, years_fts)

# Calibration material production (calib)
print("Calibration material production")
dm_calib_matprod, dm_calib_matprod_ammonia = calib_material_production_run(
    years_ots, years_fts
)

# Emission constants
print("Emission constants")
cdm_emi_fact_combustion, cdm_emi_fact_process = const_emission_factors_run()

# Energy demand constants
print("Energy demand constants")
cdm_enerdem_exclfeed_eleclight_split, cdm_enerdem_eff = const_energy_demand_run()

# Material decomposition constants
print("Material decomposition constants")
(
    cdm_pack,
    cdm_tra_veh,
    cdm_tra_bat,
    cdm_tra_infra,
    cdm_bld_floor,
    cdm_bld_pipe,
    cdm_domapp,
    cdm_elec,
    cdm_fert,
) = const_material_decomposition_run()

# Material switch ratios
print("Material switch ratios")
cdm_mat_switch_ratios = const_materia_switch_ratios_run()

# save industry pre-processing
DM_input = {
    "material-switch": dm_mat_switch,
    "material-efficiency": dm_mat_eff,
    "tech-development": dm_tech_dev,
    "cc": dm_cc,
    "tech-share": dm_tech_share,
    "material-net-import": dm_trade_netshare,
    "material-production-not-modelled": dm_matprod_fxa,
    "material-demand-wpp": dm_matdem_fxa,
    "product-net-import": dm_trade_netshare_prod,
    "packaging": dm_pack,
    "energy-switch": dm_ene_switch,
    "waste-management": dm_wst_management,
    "material-recovery": dm_material_recovery,
    "costs": dm_costs,
    "costs-cc": dm_costs_cc,
    "calib-emissions": dm_calib_emissions,
    "calib-emissions-ammonia": dm_calib_emissions_ammonia,
    "calib-energy": dm_calib_energy_demand,
    "calib-material-production": dm_calib_matprod,
    "calib-material-production-ammonia": dm_calib_matprod_ammonia,
    "const-emission-combustion": cdm_emi_fact_combustion,
    "const-emission-process": cdm_emi_fact_process,
    "const-energy-exclfeedstock-eleclightsplit": cdm_enerdem_exclfeed_eleclight_split,
    "const-energy-efficiency": cdm_enerdem_eff,
    "fxa-energy-exclfeedstock": dm_fxa_energy_excl,
    "fxa-energy-feedstock": dm_fxa_energy_feed,
    "const-material-decomp-pack": cdm_pack,
    "const-material-decomp-veh": cdm_tra_veh,
    "const-material-decomp-batteries": cdm_tra_bat,
    "const-material-decomp-infra": cdm_tra_infra,
    "const-material-decomp-floor": cdm_bld_floor,
    "const-material-decomp-dhgpipes": cdm_bld_pipe,
    "const-material-decomp-domapp": cdm_domapp,
    "const-material-decomp-electronics": cdm_elec,
    "const-material-decomp-fert": cdm_fert,
    "const-material-switch": cdm_mat_switch_ratios,
}
save_industry_pre_processing_run(DM_input)

# industry and ammonia ots pickle
DM_industry, DM_ammonia = ots_pickle_run(DM_input, years_ots)

# make fts bau
country_list = ["EU27"]
DM_industry, DM_ammonia = fts_bau_pickle_run(
    DM_industry, DM_ammonia, country_list, years_ots, years_fts
)
