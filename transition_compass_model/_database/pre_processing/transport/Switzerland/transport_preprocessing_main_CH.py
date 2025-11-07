from model.common.auxiliary_functions import create_years_list, load_pop
from processors.aviation_part1_pipeline_CH import run as aviation_pt1_run
from processors.transport_demand_pipeline import run as demand_pkm_vkm_run
from processors.passenger_fleet_pipeline import run as passenger_fleet_run
from processors.passenger_renewal_rate_and_waste_pipeline import run as passenger_ren_rate_waste_adj_run
from processors.passenger_emission_factors import run as passenger_emission_run
from processors.passenger_efficiency_pipeline import run as passenger_efficiency_run
from processors.passenger_energy_pipeline import run as passenger_energy_run
from processors.transport_ots_pickle import run as ots_pickle_run
from processors.passenger_lifetime_pipeline import run as passenger_lifetime_run
from processors.electricity_emissions_pipeline import run as electricity_emission_run

years_ots = create_years_list(1990, 2023, 1)
years_fts = create_years_list(2025, 2050, 5)

country_list = ['Switzerland', 'Vaud']

dm_pop_ots = load_pop(country_list, years_list=years_ots)

##  Demand in pkm and vkm
print('Transport passenger demand pkm, vkm')
dm_pkm_cap, dm_pkm, dm_vkm = demand_pkm_vkm_run(dm_pop_ots, years_ots)

## Total fleet and new-fleet (for private only)
print('Fleet - private and public')
dm_private_fleet, dm_public_fleet = passenger_fleet_run(dm_pkm, years_ots)

## Renewal rate & Waste + adj fleet
print('Renewal-rate & Waste + adj fleet')
# ['passenger_private-fleet', 'passenger_public-fleet', 'passenger_renewal-rate', 'passenger_new-vehicles', 'passenger_waste-fleet']
DM = passenger_ren_rate_waste_adj_run(dm_private_fleet, dm_public_fleet)
dm_private_fleet = DM['passenger_private-fleet'].copy()
dm_public_fleet = DM['passenger_public-fleet'].copy()

## Load lifetime
print('Lifetime')
# This is used just for fts, ots can be at nan
mode_cat = list(set(dm_private_fleet.col_labels['Categories1']).union(set(dm_public_fleet.col_labels['Categories1'])))
tech_cat = list(set(dm_private_fleet.col_labels['Categories2']).union(set(dm_public_fleet.col_labels['Categories2'])))
dm_lifetime = passenger_lifetime_run(mode_cat, tech_cat, years_ots, years_fts, country_list)

print('Electricity emission - TO BE REMOVED')
dm_elec_emission = electricity_emission_run(country_list, years_ots, years_fts)

## Constant Emission factors
print('Load Emission factors')
cdm_emissions_factors = passenger_emission_run()

## Energy demand
print('Energy demand')
dm_energy = passenger_energy_run(years_ots)

## Efficiency
print('Efficiency')
dm_veh_eff = passenger_efficiency_run(dm_energy, dm_vkm, dm_private_fleet, dm_public_fleet, cdm_emissions_factors, years_ots)


##  Aviation part1
print('Aviation - part1')
dm_pkm_cap_aviation, dm_pkm_fleet_aviation = aviation_pt1_run(years_ots)




## Transport ots pickle
print('Create transport pickle ots')
DM_input = {'pkm_demand': dm_pkm,
            'vkm_demand': dm_vkm,
            'pkm_cap': dm_pkm_cap,
            'pkm_cap_aviation': dm_pkm_cap_aviation,
            'efficiency': dm_veh_eff,
            'lifetime': dm_lifetime,
            'emissions_electricity': dm_elec_emission,
            'emission_factors': cdm_emissions_factors}
# DM.keys = ['passenger_private-fleet', 'passenger_public-fleet', 'passenger_renewal-rate', 'passenger_new-vehicles', 'passenger_waste-fleet']
DM_input = DM_input | DM  # join
DM_transport = ots_pickle_run(DM_input, years_ots, years_fts)


print('Hello')
