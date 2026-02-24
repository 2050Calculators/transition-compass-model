"""Preprocessing for Python."""

# Note: CHVD_industry_build-fake-pickle cannot be imported due to hyphen in filename
# from . import CHVD_industry_build-fake-pickle
from . import ammonia
from . import fxa_costs
from . import industry_buildpickle
from . import industry_calib_emissions

# Note: industry_calib_energy-demand cannot be imported due to hyphen in filename
# from . import industry_calib_energy-demand
# Note: industry_lever_product-net-import cannot be imported due to hyphen in filename
# from . import industry_lever_product-net-import
# Note: industry_lever_waste-management cannot be imported due to hyphen in filename
# from . import industry_lever_waste-management
from . import industry_to_energy_interface

__all__ = [
    # "CHVD_industry_build-fake-pickle",  # Cannot import due to hyphen
    "ammonia",
    "fxa_costs",
    "industry_buildpickle",
    "industry_calib_emissions",
    # "industry_calib_energy-demand",  # Cannot import due to hyphen
    # "industry_lever_product-net-import",  # Cannot import due to hyphen
    # "industry_lever_waste-management",  # Cannot import due to hyphen
    "industry_to_energy_interface",
]
