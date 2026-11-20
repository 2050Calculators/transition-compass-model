from processors.lca_levers import run as levers_run
from processors.lca_ots_pickle import run as ots_pickle_run
from scenarios.lca_fts_BAU_pickle import run as fts_bau_pickle_run

from transition_compass_model._database.pre_processing.params import (
    years_fts,
    years_ots,
)

# footprint lever
print("Footprint")
(
    dm_mat,
    dm_ene_dem_elec,
    dm_ene_dem_ff,
    dm_eco,
    dm_gwp,
    dm_water,
    dm_air,
    dm_heavy_metals,
) = levers_run(years_ots)

# save lca pre-processing
DM_input = {
    "materials": dm_mat,
    "energy-demand-elec": dm_ene_dem_elec,
    "energy-demand-ff": dm_ene_dem_ff,
    "ecological": dm_eco,
    "gwp": dm_gwp,
    "water": dm_water,
    "air-pollutant": dm_air,
    "heavy-metals": dm_heavy_metals,
}

# lca ots pickle
print("Doing ots")
DM_lca = ots_pickle_run(DM_input, years_ots)

# make fts bau
print("Doing fts bau")
DM_lca = fts_bau_pickle_run(DM_lca, years_fts)
