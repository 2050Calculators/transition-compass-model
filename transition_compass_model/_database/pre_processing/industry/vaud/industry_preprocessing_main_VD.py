from processors.industry_fxa_energy_demand import run as fxa_energy_demand_run
from processors.industry_lever_packaging_per_capita import (
    run as packaging_per_capita_run,
)
from processors.industry_lever_product_net_import import run as product_net_import_run
from processors.industry_lever_waste_management import run as waste_management_run
from processors.industry_ots_pickle import run as ots_pickle_run
from scenarios.industry_fts_BAU_pickle import run as fts_bau_pickle_run

from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    load_pop,
)

years_ots = create_years_list(1990, 2023, 1)
years_fts = create_years_list(2025, 2050, 5)

country_list = ["Vaud"]
dm_pop_ots = load_pop(country_list, years_list=years_ots)

# product/material net-import (Vaud-specific where data available) and
# fxa/prod for non-modelled sectors (scaled by Vaud/CH STATENT FTE ratios)
print("Product/material net-import and fxa/prod (non-modelled sectors)")
dm_netimp_goods, dm_netimp_materials, dm_wwp_demand, dm_matprod_notmodelled = (
    product_net_import_run(years_ots, years_fts)
)

# packaging per capita (same national per-capita values as CH)
print("Packaging per capita")
dm_pack = packaging_per_capita_run(dm_pop_ots, years_ots)

# waste management (same national waste-management regulations as CH)
print("Waste management")
dm_waste = waste_management_run(years_ots)

# energy demand FXA (same production intensities as CH)
print("Energy demand FXA")
DM_fxa_energy = fxa_energy_demand_run(years_ots, years_fts)

DM_input = {
    "product-net-import": dm_netimp_goods,
    "material-net-import": dm_netimp_materials,
    "material-demand-wpp": dm_wwp_demand,
    "material-production-not-modelled": dm_matprod_notmodelled,
    "packaging": dm_pack,
    "waste-management": dm_waste,
    "fxa-energy-exclfeedstock": DM_fxa_energy["industry"][
        "energy-demand-excl-feedstock"
    ],
    "fxa-energy-feedstock": DM_fxa_energy["industry"]["energy-demand-feedstock"],
    "fxa-amm-energy-exclfeedstock": DM_fxa_energy["ammonia"][
        "energy-demand-excl-feedstock"
    ],
    "fxa-amm-energy-feedstock": DM_fxa_energy["ammonia"]["energy-demand-feedstock"],
}

# assemble OTS pickle
print("Industry OTS pickle")
DM_industry, DM_ammonia = ots_pickle_run(DM_input, years_ots)

# build FTS and write Vaud into the main industry/ammonia pickles
print("Industry FTS BAU pickle")
DM_industry, DM_ammonia = fts_bau_pickle_run(
    DM_industry, DM_ammonia, country_list, years_ots, years_fts
)

print("Done — Vaud written to industry.pickle and ammonia.pickle")
