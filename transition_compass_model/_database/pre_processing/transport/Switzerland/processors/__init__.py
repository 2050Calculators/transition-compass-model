"""Preprocessing for Processors."""

from . import aviation_part1_pipeline_CH
from . import electricity_emissions_pipeline
from . import passenger_efficiency_pipeline
from . import passenger_emission_factors
from . import passenger_energy_pipeline
from . import passenger_fleet_pipeline
from . import passenger_lifetime_pipeline
from . import passenger_renewal_rate_and_waste_pipeline
from . import transport_demand_pipeline
from . import transport_ots_pickle

__all__ = [
    "aviation_part1_pipeline_CH",
    "electricity_emissions_pipeline",
    "passenger_efficiency_pipeline",
    "passenger_emission_factors",
    "passenger_energy_pipeline",
    "passenger_fleet_pipeline",
    "passenger_lifetime_pipeline",
    "passenger_renewal_rate_and_waste_pipeline",
    "transport_demand_pipeline",
    "transport_ots_pickle",
]
