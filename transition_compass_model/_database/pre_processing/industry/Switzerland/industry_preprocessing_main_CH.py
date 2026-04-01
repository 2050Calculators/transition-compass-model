from transition_compass_model.model.common.auxiliary_functions import create_years_list, load_pop
from processors.industry_lever_product_net_import import run as net_import_share_run
from processors.industry_lever_packaging_per_capita import (
    run as packaging_per_capita_run,
)
from processors.industry_lever_waste_management import run as waste_management_run
from processors.industry_calib_energy_demand import run as energy_demand_run
from processors.industry_calib_emissions import run as emissions_run
from processors.ammonia_levers_fxa import run as ammonia_run
from processors.industry_pre_processing_save import (
    run as save_industry_pre_processing_run,
)
from processors.industry_ots_pickle import run as ots_pickle_run
from scenarios.industry_fts_BAU_pickle import run as fts_bau_pickle_run

# years
years_ots = create_years_list(1990, 2023, 1)
years_fts = create_years_list(2025, 2050, 5)

# load pop
country_list = ["Switzerland"]
dm_pop_ots = load_pop(country_list, years_list=years_ots)

# net import share of goods, materials, product production of not modelled sectors, and demand of wwp
print("Net import share and material production")
(
    dm_netimp_goods,
    dm_wwp_demand,
    dm_netimp_materials,
    dm_matprod_notmodelled,
    dm_matprod_calib,
) = net_import_share_run(years_ots, years_fts)

# packaging per capita
print("Packaging per capita")
dm_pack = packaging_per_capita_run(dm_pop_ots)

# waste management
print("Waste management")
dm_waste = waste_management_run(years_ots)

# energy demand
print("Energy demand")
dm_energy = energy_demand_run(years_ots, years_fts)

# emissions
print("Emissions")
dm_emissions = emissions_run(years_ots, years_fts)

# ammonia
dm_amm_prod_net_import, dm_amm_mat_net_import, dm_amm_prod = ammonia_run(
    years_ots, years_fts
)

# save industry pre-processing
DM_input = {
    "product-net-import": dm_netimp_goods,
    "material-net-import": dm_netimp_materials,
    "material-demand-wpp": dm_wwp_demand,
    "material-production-not-modelled": dm_matprod_notmodelled,
    "packaging": dm_pack,
    "waste-management": dm_waste,
    "calib-matprod": dm_matprod_calib,
    "calib-emissions": dm_emissions,
    "calib-energy": dm_energy,
    "fert-product-net-import": dm_amm_prod_net_import,
    "amm-material-net-import": dm_amm_mat_net_import,
    "calib-amm-material-production": dm_amm_prod,
}
save_industry_pre_processing_run(DM_input)

# industry ots pickle
DM_industry, DM_ammonia = ots_pickle_run(DM_input, years_ots)

# make fts bau
DM_industry, DM_ammonia = fts_bau_pickle_run(
    DM_industry, DM_ammonia, country_list, years_ots, years_fts
)
