"""Model module."""

from . import agriculture_module
from . import ammonia_module
from . import buildings_module
from . import climate_module
from . import district_heating_module
from . import emissions_module
from . import energy_module
from . import energy_module_AMPL
from . import forestry_module
from . import industry_module
from . import interactions

# Note: interactions_localrun executes code at import time, excluded from package
# from . import interactions_localrun
from . import landuse_module
from . import lca_module
from . import lifestyles_module
from . import minerals_module
from . import oilrefinery_module
from . import power_module
from . import transport_module

from . import buildings
from . import common
from . import energy
from . import transport

__all__ = [
    "agriculture_module",
    "ammonia_module",
    "buildings_module",
    "climate_module",
    "district_heating_module",
    "emissions_module",
    "energy_module",
    "energy_module_AMPL",
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
