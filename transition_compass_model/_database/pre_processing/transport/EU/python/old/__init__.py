"""Preprocessing for Old."""

from . import data_notes

# Note: transport_fxa_freight_vehicle-efficiency_fleet cannot be imported due to hyphen in filename
# from . import transport_fxa_freight_vehicle-efficiency_fleet
# Note: transport_fxa_passenger_new-vehicles cannot be imported due to hyphen in filename
# from . import transport_fxa_passenger_new-vehicles
# Note: transport_fxa_passenger_technology-share_fleet cannot be imported due to hyphen in filename
# from . import transport_fxa_passenger_technology-share_fleet
# Note: transport_fxa_passenger_veh-efficiency_fleet cannot be imported due to hyphen in filename
# from . import transport_fxa_passenger_veh-efficiency_fleet
# Note: transport_fxa_passenger_vehicle-waste cannot be imported due to hyphen in filename
# from . import transport_fxa_passenger_vehicle-waste

__all__ = [
    "data_notes",
    # "transport_fxa_freight_vehicle-efficiency_fleet",  # Cannot import due to hyphen
    # "transport_fxa_passenger_new-vehicles",  # Cannot import due to hyphen
    # "transport_fxa_passenger_technology-share_fleet",  # Cannot import due to hyphen
    # "transport_fxa_passenger_veh-efficiency_fleet",  # Cannot import due to hyphen
    # "transport_fxa_passenger_vehicle-waste",  # Cannot import due to hyphen
]
