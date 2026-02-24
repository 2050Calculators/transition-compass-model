"""Preprocessing for Python."""

from . import industry_buildpickle
from . import industry_calib_emissions

# Note: industry_calib_energy-demand cannot be imported due to hyphen in filename
# from . import industry_calib_energy-demand
# Note: industry_calib_material-production cannot be imported due to hyphen in filename
# from . import industry_calib_material-production
# Note: industry_const_emission-factors cannot be imported due to hyphen in filename
# from . import industry_const_emission-factors
# Note: industry_const_energy-demand cannot be imported due to hyphen in filename
# from . import industry_const_energy-demand
# Note: industry_const_material-decomposition cannot be imported due to hyphen in filename
# from . import industry_const_material-decomposition
# Note: industry_const_material-switch-ratio cannot be imported due to hyphen in filename
# from . import industry_const_material-switch-ratio
from . import industry_fxa_costs

# Note: industry_fxa_material-production cannot be imported due to hyphen in filename
# from . import industry_fxa_material-production
# Note: industry_lever_carbon-capture cannot be imported due to hyphen in filename
# from . import industry_lever_carbon-capture
# Note: industry_lever_energy-switch cannot be imported due to hyphen in filename
# from . import industry_lever_energy-switch
# Note: industry_lever_material-efficiency cannot be imported due to hyphen in filename
# from . import industry_lever_material-efficiency
# Note: industry_lever_material-net-import cannot be imported due to hyphen in filename
# from . import industry_lever_material-net-import
# Note: industry_lever_material-recovery cannot be imported due to hyphen in filename
# from . import industry_lever_material-recovery
# Note: industry_lever_material-switch cannot be imported due to hyphen in filename
# from . import industry_lever_material-switch
# Note: industry_lever_product-net-import cannot be imported due to hyphen in filename
# from . import industry_lever_product-net-import
# Note: industry_lever_technology-development cannot be imported due to hyphen in filename
# from . import industry_lever_technology-development
# Note: industry_lever_technology-share cannot be imported due to hyphen in filename
# from . import industry_lever_technology-share
# Note: industry_lever_waste-management cannot be imported due to hyphen in filename
# from . import industry_lever_waste-management

__all__ = [
    "industry_buildpickle",
    "industry_calib_emissions",
    # "industry_calib_energy-demand",  # Cannot import due to hyphen
    # "industry_calib_material-production",  # Cannot import due to hyphen
    # "industry_const_emission-factors",  # Cannot import due to hyphen
    # "industry_const_energy-demand",  # Cannot import due to hyphen
    # "industry_const_material-decomposition",  # Cannot import due to hyphen
    # "industry_const_material-switch-ratio",  # Cannot import due to hyphen
    "industry_fxa_costs",
    # "industry_fxa_material-production",  # Cannot import due to hyphen
    # "industry_lever_carbon-capture",  # Cannot import due to hyphen
    # "industry_lever_energy-switch",  # Cannot import due to hyphen
    # "industry_lever_material-efficiency",  # Cannot import due to hyphen
    # "industry_lever_material-net-import",  # Cannot import due to hyphen
    # "industry_lever_material-recovery",  # Cannot import due to hyphen
    # "industry_lever_material-switch",  # Cannot import due to hyphen
    # "industry_lever_product-net-import",  # Cannot import due to hyphen
    # "industry_lever_technology-development",  # Cannot import due to hyphen
    # "industry_lever_technology-share",  # Cannot import due to hyphen
    # "industry_lever_waste-management",  # Cannot import due to hyphen
]
