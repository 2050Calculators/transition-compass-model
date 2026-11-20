"""
Assemble all EU27 transport preprocessing outputs into transport.pickle.

Faithful port of transport_build-pickle.py. Replaces disk-based loading of
intermediate pickles with direct DataMatrix arguments (data flows in memory).

Two-step build
--------------
1. run_ots(...) : Build OTS + FXA + constant → save to _OTS_CACHE_PATH (cached).
                  Returns DM_transport_ots dict.
2. run(...)     : Load OTS dict → add FTS on top → my_pickle_dump to transport.pickle.

Final pickle structure
----------------------
DM_transport = {
    "ots": { lever_name: DataMatrix (OTS 1990-2023) },
    "fts": { lever_name: {1..4: DataMatrix (FTS 2025-2050)} },
    "fxa": { fxa_name: DataMatrix },
    "constant": ConstantDataMatrix (cp_tra_emission-factor),
}

Lever names (OTS + FTS)
-----------------------
passenger_aviation-pkm, passenger_modal-share, passenger_occupancy,
passenger_technology-share_new, passenger_utilization-rate,
passenger_veh-efficiency_new, pkm,
freight_vehicle-efficiency_new, freight_utilization-rate, freight_modal-share,
freight_technology-share_new, freight_tkm, fuel-mix

FXA names
---------
passenger_tech, passenger_vehicle-lifetime, emission-factor-electricity,
freight_tech, freight_mode_other, freight_mode_road,
vehicles-max, share-local-emissions, fuel-mix-availability
"""

import os
import pickle

from transition_compass_model._database.pre_processing.transport.Switzerland.processors.passenger_emission_factors import (
    run as emission_factors_run,
)
from transition_compass_model.model.common.auxiliary_functions import my_pickle_dump

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OTS_CACHE_PATH = os.path.join(_THIS_DIR, "../data/datamatrix/transport_ots_EU.pickle")
_OUTPUT_PATH = os.path.join(_THIS_DIR, "../../../../data/datamatrix/transport.pickle")


def run_ots(
    # Lever data — processors that include their own OTS FTS
    DM_avi,  # passenger_aviation_pkm.run()
    DM_mod,  # passenger_modal_share.run()
    # Lever OTS — FTS comes from run() via scenarios
    dm_occ_ots,  # passenger_occupancy.run()
    dm_new_ots,  # passenger_tech_share_new.run()
    dm_uti_ots,  # passenger_utilization_rate.run()
    dm_eneff_new_ots,  # passenger_veh_efficiency_new.run()
    dm_pkm_cap_ots,  # passenger_pkm.run()
    dm_eneff_freight_ots,  # freight_veh_efficiency_new.run()
    dm_freight_utilization,  # freight_utilization_rate.run()
    DM_mod_freight,  # freight_modal_share.run()
    DM_tkmtot_freight,  # freight_modal_share.run()
    dm_freight_tech_share_new,  # freight_tech_share_new.run()
    dm_fuel_mix,  # fuel_mix.run()
    # FXA data
    dm_tech,  # fxa_passenger_tech.run()
    dm_lifetime,  # fxa_passenger_vehicle_lifetime.run()
    dm_emifact,  # fxa_emission_factor_electricity.run()
    dm_freight_tech,  # fxa_freight_tech.run()
    dm_freight_mode_other,  # fxa_freight_mode_other.run()
    dm_freight_mode_road,  # fxa_freight_mode_road.run()
    dm_vehicles_max,  # fxa_vehicles_max.run()
    dm_share_local,  # fxa_aviation_share_local_emissions.run()
    dm_fuel_mix_avail,  # fxa_aviation_fuel_mix_availability.run()
    # Config
    country_list,
):
    """
    Build OTS + FXA + constant and save to intermediate cache.

    Returns
    -------
    DM_transport_ots : dict with keys "ots", "fxa", "constant"
    """
    if os.path.exists(_OTS_CACHE_PATH):
        print("  (OTS cached)")
        with open(_OTS_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    ##############
    # BUILD OTS  #
    ##############

    DM_ots = {
        "passenger_aviation-pkm": DM_avi["ots"]["passenger_aviation-pkm"],
        "passenger_modal-share": DM_mod["ots"]["passenger_modal-share"],
        "passenger_occupancy": dm_occ_ots,
        "passenger_technology-share_new": dm_new_ots,
        "passenger_utilization-rate": dm_uti_ots,
        "passenger_veh-efficiency_new": dm_eneff_new_ots,
        "pkm": dm_pkm_cap_ots,
        "freight_vehicle-efficiency_new": dm_eneff_freight_ots,
        "freight_utilization-rate": dm_freight_utilization,
        "freight_modal-share": DM_mod_freight["ots"]["freight_modal-share"],
        "freight_technology-share_new": dm_freight_tech_share_new,
        "freight_tkm": DM_tkmtot_freight["ots"]["freight_tkm"],
        "fuel-mix": dm_fuel_mix,
    }

    ##############
    # BUILD FXA  #
    ##############

    DM_fxa = {
        "passenger_tech": dm_tech,
        "passenger_vehicle-lifetime": dm_lifetime,
        "emission-factor-electricity": dm_emifact,
        "freight_tech": dm_freight_tech,
        "freight_mode_other": dm_freight_mode_other,
        "freight_mode_road": dm_freight_mode_road,
        "vehicles-max": dm_vehicles_max,
        "share-local-emissions": dm_share_local,
        "fuel-mix-availability": dm_fuel_mix_avail,
    }

    ##############
    # CONSTANT   #
    ##############

    cdm_const = emission_factors_run()

    ##########################
    # KEEP ONLY country_list #
    ##########################

    for key in DM_ots.keys():
        DM_ots[key].filter({"Country": country_list}, inplace=True)
    for key in DM_fxa.keys():
        DM_fxa[key].filter({"Country": country_list}, inplace=True)
    # constant has no Country dimension — no filter needed

    #################
    # ASSEMBLE      #
    #################

    DM_transport_ots = {
        "ots": DM_ots,
        "fxa": DM_fxa,
        "constant": cdm_const,
    }

    ##########
    # SAVE   #
    ##########

    os.makedirs(os.path.dirname(_OTS_CACHE_PATH), exist_ok=True)
    with open(_OTS_CACHE_PATH, "wb") as f:
        pickle.dump(DM_transport_ots, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved OTS cache → {_OTS_CACHE_PATH}")
    return DM_transport_ots


def run(
    DM_transport_ots,  # output of run_ots()
    DM_avi,  # passenger_aviation_pkm.run() → fts dict
    DM_mod,  # passenger_modal_share.run() → fts dict
    DM_fts,  # scenarios.transport_fts_EU.run()
    country_list,
):
    """
    Add FTS to OTS dict and my_pickle_dump to transport.pickle.

    Parameters
    ----------
    DM_transport_ots : dict
        Output of run_ots() with "ots", "fxa", "constant" keys.
    DM_avi, DM_mod : dict
        Processor dicts containing their own FTS (keyed "fts").
    DM_fts : dict
        Scenarios output from transport_fts_EU.run().
    country_list : list
    """
    ##############
    # BUILD FTS  #
    ##############

    DM_fts_all = {
        # These processors compute their own FTS internally
        "passenger_aviation-pkm": DM_avi["fts"]["passenger_aviation-pkm"],
        "passenger_modal-share": DM_mod["fts"]["passenger_modal-share"],
        # These come from scenarios/transport_fts_EU.py
        "passenger_occupancy": DM_fts["passenger_occupancy"],
        "passenger_technology-share_new": DM_fts["passenger_technology-share_new"],
        "passenger_utilization-rate": DM_fts["passenger_utilization-rate"],
        "passenger_veh-efficiency_new": DM_fts["passenger_veh-efficiency_new"],
        "pkm": DM_fts["pkm"],
        "freight_vehicle-efficiency_new": DM_fts["freight_vehicle-efficiency_new"],
        "freight_utilization-rate": DM_fts["freight_utilization-rate"],
        "freight_modal-share": DM_fts["freight_modal-share"],
        "freight_technology-share_new": DM_fts["freight_technology-share_new"],
        "freight_tkm": DM_fts["freight_tkm"],
        "fuel-mix": DM_fts["fuel-mix"],
    }

    ##########################
    # KEEP ONLY country_list #
    ##########################

    for key in DM_fts_all.keys():
        for level in list(range(1, 4 + 1)):
            DM_fts_all[key][level].filter({"Country": country_list}, inplace=True)

    #################
    # ASSEMBLE      #
    #################

    DM_transport = {**DM_transport_ots, "fts": DM_fts_all}

    ##########
    # SAVE   #
    ##########

    os.makedirs(os.path.dirname(_OUTPUT_PATH), exist_ok=True)
    my_pickle_dump(DM_transport, _OUTPUT_PATH)
    print(f"  Saved transport.pickle → {_OUTPUT_PATH}")
