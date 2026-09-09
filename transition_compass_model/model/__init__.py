"""Model module."""

# Note: interactions_localrun executes code at import time, excluded from package
# from . import interactions_localrun
from . import (
    agriculture_module,
    ammonia_module,
    buildings,
    buildings_module,
    climate_module,
    common,
    district_heating_module,
    emissions_module,
    energy,
    energy_module,
    forestry_module,
    industry_module,
    interactions,
    landuse_module,
    lca_module,
    lifestyles_module,
    minerals_module,
    oilrefinery_module,
    power_module,
    transport,
    transport_module,
)

__all__ = [
    "agriculture_module",
    "ammonia_module",
    "buildings_module",
    "climate_module",
    "district_heating_module",
    "emissions_module",
    "energy_module",
    "forestry_module",
    "industry_module",
    "interactions",
    # "interactions_localrun",  # Excluded - executes code at import time
    "landuse_module",
    "lca_module",
    "lifestyles_module",
    "minerals_module",
    "oilrefinery_module",
    "power_module",
    "transport_module",
    "buildings",
    "common",
    "energy",
    "transport",
]
