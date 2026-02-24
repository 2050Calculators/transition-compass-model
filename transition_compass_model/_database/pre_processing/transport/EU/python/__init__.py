"""Preprocessing for Python."""

# Note: transport_build-pickle cannot be imported due to hyphen in filename
# from . import transport_build-pickle
# Note: transport_fxa_aviation_fuel-mix-availability cannot be imported due to hyphen in filename
# from . import transport_fxa_aviation_fuel-mix-availability
# Note: transport_fxa_aviation_share-local-emissions cannot be imported due to hyphen in filename
# from . import transport_fxa_aviation_share-local-emissions
# Note: transport_fxa_emission-factor-electricity cannot be imported due to hyphen in filename
# from . import transport_fxa_emission-factor-electricity
from . import transport_fxa_freight_mode_other
from . import transport_fxa_freight_mode_road
from . import transport_fxa_freight_tech
from . import transport_fxa_passenger_tech

# Note: transport_fxa_passenger_vehicle-lifetime cannot be imported due to hyphen in filename
# from . import transport_fxa_passenger_vehicle-lifetime
# Note: transport_fxa_vehicles-max cannot be imported due to hyphen in filename
# from . import transport_fxa_vehicles-max
# Note: transport_lever_freight_modal-share cannot be imported due to hyphen in filename
# from . import transport_lever_freight_modal-share
# Note: transport_lever_freight_technology-share_new cannot be imported due to hyphen in filename
# from . import transport_lever_freight_technology-share_new
from . import transport_lever_freight_tkm

# Note: transport_lever_freight_utilization-rate cannot be imported due to hyphen in filename
# from . import transport_lever_freight_utilization-rate
# Note: transport_lever_freight_vehicle-efficiency_new cannot be imported due to hyphen in filename
# from . import transport_lever_freight_vehicle-efficiency_new
# Note: transport_lever_fuel-mix cannot be imported due to hyphen in filename
# from . import transport_lever_fuel-mix
# Note: transport_lever_passenger_aviation-pkm cannot be imported due to hyphen in filename
# from . import transport_lever_passenger_aviation-pkm
# Note: transport_lever_passenger_modal-share cannot be imported due to hyphen in filename
# from . import transport_lever_passenger_modal-share
from . import transport_lever_passenger_occupancy

# Note: transport_lever_passenger_technology-share_new cannot be imported due to hyphen in filename
# from . import transport_lever_passenger_technology-share_new
# Note: transport_lever_passenger_utilization-rate cannot be imported due to hyphen in filename
# from . import transport_lever_passenger_utilization-rate
# Note: transport_lever_passenger_veh-efficiency_new cannot be imported due to hyphen in filename
# from . import transport_lever_passenger_veh-efficiency_new
from . import transport_lever_pkm

from . import old

__all__ = [
    # "transport_build-pickle",  # Cannot import due to hyphen
    # "transport_fxa_aviation_fuel-mix-availability",  # Cannot import due to hyphen
    # "transport_fxa_aviation_share-local-emissions",  # Cannot import due to hyphen
    # "transport_fxa_emission-factor-electricity",  # Cannot import due to hyphen
    "transport_fxa_freight_mode_other",
    "transport_fxa_freight_mode_road",
    "transport_fxa_freight_tech",
    "transport_fxa_passenger_tech",
    # "transport_fxa_passenger_vehicle-lifetime",  # Cannot import due to hyphen
    # "transport_fxa_vehicles-max",  # Cannot import due to hyphen
    # "transport_lever_freight_modal-share",  # Cannot import due to hyphen
    # "transport_lever_freight_technology-share_new",  # Cannot import due to hyphen
    "transport_lever_freight_tkm",
    # "transport_lever_freight_utilization-rate",  # Cannot import due to hyphen
    # "transport_lever_freight_vehicle-efficiency_new",  # Cannot import due to hyphen
    # "transport_lever_fuel-mix",  # Cannot import due to hyphen
    # "transport_lever_passenger_aviation-pkm",  # Cannot import due to hyphen
    # "transport_lever_passenger_modal-share",  # Cannot import due to hyphen
    "transport_lever_passenger_occupancy",
    # "transport_lever_passenger_technology-share_new",  # Cannot import due to hyphen
    # "transport_lever_passenger_utilization-rate",  # Cannot import due to hyphen
    # "transport_lever_passenger_veh-efficiency_new",  # Cannot import due to hyphen
    "transport_lever_pkm",
    "old",
]
