import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from params import country_list, years_fts, years_ots
from processors.build_pickle import run as build_pickle_run
from processors.build_pickle import run_ots as build_ots_run
from processors.floor_area import run as floor_area_run
from processors.heating_efficiency import run as heating_efficiency_run
from processors.heating_mix import run as heating_mix_run
from processors.renovation import run as renovation_run
from processors.stock_by_cat import run as stock_by_cat_run
from scenarios.buildings_fts_EU import run as buildings_fts_run

###############################################################################
# FLOOR AREA
###############################################################################

print("Floor area (stock / new / waste / services)")
dm_all, dm_stock_tot, dm_new_tot, dm_waste_tot, dm_services_floor, dm_pop = (
    floor_area_run(years_ots)
)

###############################################################################
# RENOVATION
###############################################################################

print("Renovation (rates / redistribution / renovated floor area)")
dm_renovation, dm_renov_cat, dm_renov_distr = renovation_run(dm_stock_tot, years_ots)

###############################################################################
# STOCK BY CATEGORY
###############################################################################

print("Stock by category (envelope class) + u-value + surface-to-floorarea")
dm_all_cat, dm_u_value, dm_s2f = stock_by_cat_run(
    dm_new_tot, dm_waste_tot, dm_stock_tot, dm_renov_cat, years_ots
)

###############################################################################
# HEATING MIX
###############################################################################

print("Heating mix (space heating + hot water by tech × envelope class)")
dm_heating_cat = heating_mix_run(years_ots)

###############################################################################
# HEATING EFFICIENCY
###############################################################################

print("Heating efficiency (by tech × envelope class)")
dm_heating_eff_cat, dm_heating_eff_all = heating_efficiency_run(dm_all_cat, years_ots)

###############################################################################
# BUILD OTS CACHE
###############################################################################

print("Build OTS cache (assemble OTS + FXA + constant)")
DM_buildings_ots = build_ots_run(
    dm_all_cat,
    dm_pop,
    dm_services_floor,
    dm_renovation,
    dm_renov_distr,
    dm_stock_tot,
    dm_waste_tot,
    dm_heating_cat,
    dm_heating_eff_cat,
    dm_heating_eff_all,
    dm_u_value,
    dm_s2f,
    years_ots,
    years_fts,
)

###############################################################################
# FTS SCENARIOS
###############################################################################

print("FTS scenarios (lever-differentiated future time series)")
DM_fts = buildings_fts_run(DM_buildings_ots, years_fts)

###############################################################################
# BUILD PICKLE
###############################################################################

print("Build pickle (assemble OTS + FTS + FXA + constant → buildings.pickle)")
build_pickle_run(DM_buildings_ots, DM_fts, country_list)
