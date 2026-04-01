"""Preprocessing for Scenarios."""

from . import build_fts_BAU_pickle
from . import build_fts_LoiEnergie_Vaud_pickle
from . import build_fts_Tint_heating_pickle
from . import build_fts_floor_area_pickle
from . import build_fts_heating_efficiency_pickle
from . import buildings_fts_EP2050_pickle

__all__ = [
    "build_fts_BAU_pickle",
    "build_fts_LoiEnergie_Vaud_pickle",
    "build_fts_Tint_heating_pickle",
    "build_fts_floor_area_pickle",
    "build_fts_heating_efficiency_pickle",
    "buildings_fts_EP2050_pickle",
]
