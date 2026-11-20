from params import country_list, years_fts, years_ots
from processors.build_pickle import run as build_pickle_run
from processors.build_pickle import run_ots as build_ots_run
from processors.freight_modal_share import run as freight_modal_share_run
from processors.freight_tech_share_new import run as freight_tech_share_new_run
from processors.freight_utilization_rate import run as freight_utilization_rate_run
from processors.freight_veh_efficiency_new import run as freight_veh_eff_new_run
from processors.fuel_mix import run as fuel_mix_run
from processors.fxa_aviation_fuel_mix_availability import run as fxa_fuel_mix_avail_run
from processors.fxa_aviation_share_local_emissions import run as fxa_share_local_run
from processors.fxa_emission_factor_electricity import run as fxa_elec_ef_run
from processors.fxa_freight_mode_other import run as fxa_freight_mode_other_run
from processors.fxa_freight_mode_road import run as fxa_freight_mode_road_run
from processors.fxa_freight_tech import run as fxa_freight_tech_run
from processors.fxa_passenger_tech import run as fxa_passenger_tech_run
from processors.fxa_passenger_vehicle_lifetime import run as fxa_passenger_lifetime_run
from processors.fxa_vehicles_max import run as fxa_vehicles_max_run
from processors.passenger_aviation_pkm import run as passenger_aviation_pkm_run
from processors.passenger_modal_share import run as passenger_modal_share_run
from processors.passenger_occupancy import run as passenger_occupancy_run
from processors.passenger_pkm import run as passenger_pkm_run
from processors.passenger_tech_share_new import run as passenger_tech_share_new_run
from processors.passenger_utilization_rate import run as passenger_utilization_rate_run
from processors.passenger_veh_efficiency_new import run as passenger_veh_eff_new_run
from scenarios.transport_fts_EU import run as transport_fts_run

###############################################################################
# PASSENGER
###############################################################################

print("Passenger fleet + tech FXA")
dm_tech, dm_fleet, dm_new, dm_renrate = fxa_passenger_tech_run(years_ots, years_fts)

print("Passenger aviation pkm")
dm_avi, DM_avi = passenger_aviation_pkm_run(years_ots, years_fts)

print("Passenger modal share")
DM_mod, DM_pkm = passenger_modal_share_run(years_ots, years_fts)

print("Passenger occupancy")
dm_occ_ots, dm_vkm_ots, dm_pkm_avi_terr_ots = passenger_occupancy_run(DM_pkm, years_ots)

print("Passenger technology share new")
dm_tech_share_new = passenger_tech_share_new_run(dm_new, years_ots)

print("Passenger utilization rate")
dm_uti_ots = passenger_utilization_rate_run(dm_fleet, dm_vkm_ots, years_ots)

print("Passenger vehicle efficiency new")
dm_veh_eff_new = passenger_veh_eff_new_run(years_ots)

print("Passenger pkm total")
dm_pkm_cap_ots = passenger_pkm_run(DM_pkm, years_ots)

print("FXA passenger vehicle lifetime")
dm_lifetime = fxa_passenger_lifetime_run(dm_tech, years_ots, years_fts)

print("FXA vehicles max (aviation BEV/H2)")
dm_vehicles_max = fxa_vehicles_max_run(years_ots, years_fts, dm_vkm_ots)

print("FXA electricity emission factor")
dm_elec_ef = fxa_elec_ef_run(years_ots, years_fts)

###############################################################################
# FREIGHT
###############################################################################

print("Freight modal share + tkm")
DM_mod_freight, DM_tkm_freight, DM_tkmtot_freight = freight_modal_share_run(
    years_ots, years_fts
)

print("FXA freight tech")
dm_freight_tech, dm_freight_fleet = fxa_freight_tech_run(years_ots, years_fts)

print("FXA freight mode other")
dm_freight_mode_other = fxa_freight_mode_other_run(
    years_ots, years_fts, DM_tkm_freight, dm_freight_fleet
)

print("Freight utilization rate")
dm_freight_utilization = freight_utilization_rate_run(
    years_ots, years_fts, DM_tkm_freight
)

print("FXA freight mode road")
dm_freight_mode_road = fxa_freight_mode_road_run(
    years_ots, years_fts, dm_freight_utilization
)

print("Freight technology share new")
dm_freight_tech_share_new = freight_tech_share_new_run(years_ots, years_fts)

print("Freight vehicle efficiency new")
dm_freight_veh_eff = freight_veh_eff_new_run(years_ots, years_fts)

###############################################################################
# CROSS-SECTOR LEVERS & FXA
###############################################################################

print("Fuel mix")
dm_fuel_mix = fuel_mix_run(years_ots, years_fts)

print("FXA aviation share local emissions")
dm_share_local = fxa_share_local_run(years_ots, years_fts)

print("FXA aviation fuel mix availability")
dm_fuel_mix_avail = fxa_fuel_mix_avail_run(years_ots, years_fts, dm_share_local)


###############################################################################
# FTS SCENARIOS
###############################################################################

print("Transport FTS scenarios")
DM_fts = transport_fts_run(
    dm_occ_ots,
    dm_vkm_ots,
    dm_pkm_avi_terr_ots,
    dm_uti_ots,
    dm_tech_share_new,
    dm_veh_eff_new,
    DM_pkm,
    dm_freight_utilization,
    DM_tkm_freight,
    DM_tkmtot_freight,
    dm_freight_tech_share_new,
    dm_freight_veh_eff,
    dm_fuel_mix,
    years_ots,
    years_fts,
)

###############################################################################
# BUILD PICKLE
###############################################################################

print("Build transport OTS pickle")
DM_transport_ots = build_ots_run(
    DM_avi=DM_avi,
    DM_mod=DM_mod,
    dm_occ_ots=dm_occ_ots,
    dm_new_ots=dm_tech_share_new,
    dm_uti_ots=dm_uti_ots,
    dm_eneff_new_ots=dm_veh_eff_new,
    dm_pkm_cap_ots=dm_pkm_cap_ots,
    dm_eneff_freight_ots=dm_freight_veh_eff,
    dm_freight_utilization=dm_freight_utilization,
    DM_mod_freight=DM_mod_freight,
    DM_tkmtot_freight=DM_tkmtot_freight,
    dm_freight_tech_share_new=dm_freight_tech_share_new,
    dm_fuel_mix=dm_fuel_mix,
    dm_tech=dm_tech,
    dm_lifetime=dm_lifetime,
    dm_emifact=dm_elec_ef,
    dm_freight_tech=dm_freight_tech,
    dm_freight_mode_other=dm_freight_mode_other,
    dm_freight_mode_road=dm_freight_mode_road,
    dm_vehicles_max=dm_vehicles_max,
    dm_share_local=dm_share_local,
    dm_fuel_mix_avail=dm_fuel_mix_avail,
    country_list=country_list,
)

print("Build transport pickle (OTS + FTS)")
build_pickle_run(
    DM_transport_ots=DM_transport_ots,
    DM_avi=DM_avi,
    DM_mod=DM_mod,
    DM_fts=DM_fts,
    country_list=country_list,
)
