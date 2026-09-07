"""Model module."""

# Note: interactions_localrun executes code at import time, excluded from package
from . import (
    agriculture_module,
    alcoholic_beverages_module,
    buildings,
    buildings_module,
    common,
    crop_module,
    dietary_habits_module,
    forestry_module,
    interactions,
    land_use_module,
    livestock_module,
    population_module,
    tcaf_module,
)

__all__ = [
    "agriculture_module",
    "alcoholic_beverages_module",
    "buildings_module",
    "crop_module",
    "dietary_habits_module",
    "forestry_module",
    "interactions",
    # "interactions_localrun",  # Excluded - executes code at import time
    "land_use_module",
    "livestock_module",
    "population_module",
    "tcaf_module",
    "buildings",
    "common",
]
