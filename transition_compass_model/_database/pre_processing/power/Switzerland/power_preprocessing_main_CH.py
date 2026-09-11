from processors.capacity_merge_pipeline_CH import redistribute_by_canton
from processors.capacity_merge_pipeline_CH import run as capacity_merge_run
from processors.capacity_nexuse_pipeline_CH import run as capacity_nexuse_run
from processors.energy_statistics_pipeline_CH import run as energy_statistics_run
from processors.fuels_supply_pipeline_CH import run as fuels_supply_run
from processors.hydro_capacity_pipeline_CH import run as hydro_capacity_run
from processors.importexport_pipeline_CH import run as importexport_run
from processors.nuclear_capacity_pipeline_CH import run as nuclear_capacity_run
from processors.oilgas_capacity_pipeline_CH import run as oilgas_capacity_run
from processors.oilgas_capacity_pipeline_CH import split_oilgas_by_capacity_factor
from processors.production_pipeline_CH import run as production_run
from processors.renewable_capacity_pipeline_CH import run as renewable_capacity_run
from processors.waste_capacity_pipeline_CH import run as waste_capacity_run
from scenarios.power_fts_capacity_levers_CH import run as power_levers_run

from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    my_pickle_dump,
)

# years_setting = [1990, 2022, 2050, 5]  # Set the timestep for historical years & scenarios
years_ots = create_years_list(1990, 2023, 1)
years_fts = create_years_list(2025, 2050, 5)
baseyear = 2023

# Canton SubRegion code (as used in the Nexus-e capacity data) -> English name.
# Note: Fribourg's Nexus-e code is "FB", not the standard "FR".
CANTON_NAMES = {
    "AG": "Aargau",
    "AR": "Appenzell Ausserrhoden",
    "AI": "Appenzell Innerrhoden",
    "BL": "Basel-Landschaft",
    "BS": "Basel-Stadt",
    "BE": "Bern",
    "FR": "Fribourg",
    "FB": "Fribourg",
    "GE": "Geneva",
    "GL": "Glarus",
    "GR": "Graubunden",
    "JU": "Jura",
    "LU": "Lucerne",
    "NE": "Neuchatel",
    "NW": "Nidwalden",
    "OW": "Obwalden",
    "SH": "Schaffhausen",
    "SZ": "Schwyz",
    "SO": "Solothurn",
    "SG": "St. Gallen",
    "TI": "Ticino",
    "TG": "Thurgau",
    "UR": "Uri",
    "VD": "Vaud",
    "VS": "Valais",
    "ZG": "Zug",
    "ZH": "Zurich",
}

# Which cantons (besides Switzerland) to keep in energy.pickle, as English names.
cantons_to_add = ["Vaud", "Schwyz", "Fribourg"]
unknown_cantons = set(cantons_to_add) - set(CANTON_NAMES.values())
if unknown_cantons:
    raise ValueError(f"Unknown canton(s) in cantons_to_add: {sorted(unknown_cantons)}")

print("Running energy statistics pipeline")
dm_energy = energy_statistics_run(baseyear, years_ots)

print("Running Nexus-e capacity pipeline")
dm_capacity, dm_const, dm_capacity_group = capacity_nexuse_run(years_ots, years_fts)

print("Running hydro capacity pipeline")
dm_capacity_hydro_ots = hydro_capacity_run(dm_capacity, years_ots)

print("Running nuclear capacity pipeline")
dm_capacity_nuclear, reactor_list = nuclear_capacity_run(dm_capacity)

print("Running waste capacity pipeline")
dm_capacity_waste_ots = waste_capacity_run(dm_capacity, years_ots)

print("Running oil & gas capacity pipeline")
dm_capacity_oilgas_ots = oilgas_capacity_run(years_ots)

print("Running renewable (wind/solar) capacity pipeline")
dm_capacity_PV_wind_ots = renewable_capacity_run()

print("Running electricity production pipeline")
dm_production, file_url, local_filename = production_run(years_ots)

print("Running import/export pipeline")
dm_production = importexport_run(file_url, local_filename, dm_production, years_ots)

print("Running capacity merge pipeline")
dm_capacity = capacity_merge_run(
    dm_capacity,
    dm_capacity_oilgas_ots,
    dm_capacity_waste_ots,
    dm_capacity_nuclear,
    dm_capacity_hydro_ots,
    dm_capacity_PV_wind_ots,
    years_ots,
    years_fts,
)

print("Splitting oil/gas production by capacity factor")
dm_production, dm_capacity = split_oilgas_by_capacity_factor(
    dm_production, dm_capacity, years_ots
)

print("Redistributing capacity by canton")
dm_capacity = redistribute_by_canton(dm_capacity)

print("Running fuels supply pipeline")
dm_fuels_supply, dm_production = fuels_supply_run(
    file_url, local_filename, dm_production, years_ots
)

print("Compiling power capacity levers (nuclear, onshore wind, PV)")
DM_ots, DM_fts = power_levers_run(dm_capacity, reactor_list, years_ots, years_fts)

for code, name in CANTON_NAMES.items():
    dm_capacity.rename_col(code, name, dim="Country")
DM_energy = {
    "fxa": {
        "capacity": dm_capacity.filter({"Country": ["Switzerland"] + cantons_to_add}),
        "production": dm_production.filter({"Country": ["Switzerland"]}),
        "fuels": dm_fuels_supply,
    },
    "ots": DM_ots,
    "fts": DM_fts,
}

file = "../../../data/datamatrix/energy.pickle"
my_pickle_dump(DM_energy, file)
