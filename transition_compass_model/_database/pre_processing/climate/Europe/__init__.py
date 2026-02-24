"""Preprocessing for Europe."""

# Note: EU_climate_build-fake-pickle cannot be imported due to hyphen in filename
# from . import EU_climate_build-fake-pickle
from . import climate_preprocessing_EU

__all__ = [
    # "EU_climate_build-fake-pickle",  # Cannot import due to hyphen
    "climate_preprocessing_EU",
]
