"""Preprocessing for Old."""

# Note: EU_buildings_build-fake-pickle cannot be imported due to hyphen in filename
# from . import EU_buildings_build-fake-pickle
from . import buildings_preprocessing_EU

__all__ = [
    # "EU_buildings_build-fake-pickle",  # Cannot import due to hyphen
    "buildings_preprocessing_EU",
]
