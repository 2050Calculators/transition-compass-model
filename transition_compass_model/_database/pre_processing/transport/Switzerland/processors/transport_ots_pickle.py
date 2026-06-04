import copy
import os

import numpy as np
from _database.pre_processing.transport.Switzerland.processors.passenger_efficiency_pipeline import (
    compute_tech_share,
)

from transition_compass_model.model.common.auxiliary_functions import (
    dm_add_missing_variables,
    my_pickle_dump,
    sort_pickle,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

# Biofuel blend fraction (%) for road and rail OTS — from EU JRC-IDEES used as Swiss proxy.
# Swiss federal biofuel mandate (Mineralölsteuergesetz) applies uniformly across all cantons,
# so Vaud uses the same values as Switzerland.
# Years before 2000 are 0 (no mandate).
_CH_BIOFUEL_BLEND = {
    2000: 0.0041,
    2001: 0.0059,
    2002: 0.0092,
    2003: 0.0136,
    2004: 0.0177,
    2005: 0.0360,
    2006: 0.0639,
    2007: 0.0734,
    2008: 0.0580,
    2009: 0.0526,
    2010: 0.0571,
    2011: 0.0547,
    2012: 0.0579,
    2013: 0.0520,
    2014: 0.0524,
    2015: 0.0485,
    2016: 0.0491,
    2017: 0.0498,
    2018: 0.0504,
    2019: 0.0510,
    2020: 0.0516,
    2021: 0.0522,
    2022: 0.0528,
    2023: 0.0534,
}


def _expand_ch_to_countries(dm, country_list):
    """Duplicate Switzerland data to any countries in country_list not already present."""
    for country in country_list:
        if country not in dm.col_labels["Country"]:
            dm_c = dm.filter({"Country": ["Switzerland"]})
            dm_c.rename_col("Switzerland", country, dim="Country")
            dm.append(dm_c, dim="Country")


def _add_aviation_to_dm(dm_target, dm_aviation_ch, country_list):
    """Append an aviation sub-DM (Switzerland only) to a multi-modal target DM.

    Duplicates Switzerland data for any extra countries in country_list.
    Aligns Categories2 (road techs vs aviation techs) before appending.
    """
    dm_av = dm_aviation_ch.copy()
    for country in country_list:
        if country not in dm_av.col_labels["Country"]:
            dm_country = dm_av.filter({"Country": ["Switzerland"]})
            dm_country.rename_col("Switzerland", country, dim="Country")
            dm_av.append(dm_country, dim="Country")
    if "Categories2" in dm_target.col_labels and "Categories2" in dm_av.col_labels:
        cat2_all = set(dm_target.col_labels["Categories2"]).union(
            set(dm_av.col_labels["Categories2"])
        )
        dm_add_missing_variables(dm_target, {"Categories2": cat2_all})
        dm_add_missing_variables(dm_av, {"Categories2": cat2_all})
    dm_target.append(dm_av, dim="Categories1")


def run(DM_input, years_ots, years_fts, DM_aviation_ots, country_list):
    DM_transport_new = {"ots": dict(), "fts": dict(), "fxa": dict(), "constant": dict()}

    ######################################
    ####    TECHNOLOGY SHARE FLEET   #####
    ######################################
    dm_private_fleet = DM_input["passenger_private-fleet"].copy()
    dm_public_fleet = DM_input["passenger_public-fleet"].copy()
    dm_fleet_tech_share = compute_tech_share(dm_private_fleet, dm_public_fleet)
    dm_fleet_tech_share.rename_col(
        "tra_passenger_vehicle-fleet_share",
        "tra_passenger_technology-share_fleet",
        dim="Variables",
    )

    ##############################
    ####     WASTE FLEET     #####
    ##############################
    dm_waste_fleet = DM_input["passenger_waste-fleet"].copy()
    dm_fleet_tech_share.append(dm_waste_fleet, dim="Variables")

    ############################
    ####     NEW FLEET     #####
    ############################
    dm_new_fleet = DM_input["passenger_new-vehicles"].copy()
    dm_fleet_tech_share.append(dm_new_fleet, dim="Variables")

    ###################################
    ####     EFFICIENCY FLEET     #####
    ###################################
    dm_veh_eff = DM_input["efficiency"].filter(
        {"Variables": ["tra_passenger_veh-efficiency_fleet"]}
    )
    dm_fleet_tech_share.append(dm_veh_eff, dim="Variables")

    # fts values are left to nan because they get recomputed
    dm_fleet_tech_share.add(np.nan, dim="Years", col_label=years_fts, dummy=True)
    DM_transport_new["fxa"]["passenger_tech"] = dm_fleet_tech_share

    #################################
    ####     EFFICIENCY NEW     #####
    #################################
    dm_veh_new_eff = DM_input["efficiency"].filter(
        {"Variables": ["tra_passenger_veh-efficiency_new"]}
    )
    DM_transport_new["ots"]["passenger_veh-efficiency_new"] = dm_veh_new_eff

    ################################
    ####    FLEET LIFETIME     #####
    ################################
    dm_lifetime = DM_input["lifetime"].copy()
    DM_transport_new["fxa"]["passenger_vehicle-lifetime"] = dm_lifetime

    ####################################
    ####    ELECTRICITY EMISSION   #####
    ####################################
    dm_elec = DM_input["emissions_electricity"].copy()
    DM_transport_new["fxa"]["emission-factor-electricity"] = dm_elec

    ##############################
    ####    DEMAND PKM/CAP   #####
    ##############################
    dm_pkm_cap = DM_input["pkm_cap"].copy()
    dm_pkm_cap_tot = dm_pkm_cap.group_all(dim="Categories1", inplace=False)
    DM_transport_new["ots"]["pkm"] = dm_pkm_cap_tot

    #######################################
    ####    DEMAND PKM/CAP - AVIATION #####
    #######################################
    dm_pkm_av = DM_aviation_ots["ots"]["passenger_aviation-pkm"].copy()
    for country in country_list:
        if country not in dm_pkm_av.col_labels["Country"]:
            dm_c = dm_pkm_av.filter({"Country": ["Switzerland"]})
            dm_c.rename_col("Switzerland", country, dim="Country")
            dm_pkm_av.append(dm_c, dim="Country")
    DM_transport_new["ots"]["passenger_aviation-pkm"] = dm_pkm_av

    ###########################
    ####    MODAL SHARE   #####
    ###########################
    dm_modal_share = dm_pkm_cap.normalise(dim="Categories1", inplace=False)
    dm_modal_share.rename_col(
        "tra_pkm-cap_share", "tra_passenger_modal-share", "Variables"
    )
    DM_transport_new["ots"]["passenger_modal-share"] = dm_modal_share

    #############################################
    ####     TECHNOLOGY SHARE NEW FLEET     #####
    #############################################
    dm_fleet_new_tech_share = dm_new_fleet.normalise(dim="Categories2", inplace=False)
    dm_fleet_new_tech_share.rename_col(
        "tra_passenger_new-vehicles_share",
        "tra_passenger_technology-share_new",
        dim="Variables",
    )
    DM_transport_new["ots"]["passenger_technology-share_new"] = dm_fleet_new_tech_share

    ############################
    ####     OCCUPANCY     #####
    ############################
    dm_pkm = DM_input["pkm_demand"].copy()
    dm_vkm = DM_input["vkm_demand"].copy()
    dm_km = dm_pkm.filter({"Categories1": dm_vkm.col_labels["Categories1"]})
    dm_km.append(dm_vkm, dim="Variables")
    dm_km.operation(
        "tra_passenger_transport-demand",
        "/",
        "tra_passenger_transport-demand-vkm",
        out_col="tra_passenger_occupancy",
        unit="pkm/vkm",
    )
    dm_occupancy = dm_km.filter({"Variables": ["tra_passenger_occupancy"]})
    DM_transport_new["ots"]["passenger_occupancy"] = dm_occupancy

    ###################################
    ####     UTILISATION RATE     #####
    ###################################
    cat_tech = set(dm_private_fleet.col_labels["Categories2"]).union(
        set(dm_public_fleet.col_labels["Categories2"])
    )
    # Join private and public
    dm_add_missing_variables(dm_public_fleet, {"Categories2": cat_tech})
    dm_add_missing_variables(dm_private_fleet, {"Categories2": cat_tech})
    dm_private_fleet.append(dm_public_fleet, dim="Categories1")
    dm_fleet = dm_private_fleet.group_all("Categories2", inplace=False)
    dm_fleet.append(dm_vkm, dim="Variables")
    dm_fleet.operation(
        "tra_passenger_transport-demand-vkm",
        "/",
        "tra_passenger_vehicle-fleet",
        out_col="tra_passenger_utilisation-rate",
        unit="vkm/veh",
    )
    dm_utilisation = dm_fleet.filter({"Variables": ["tra_passenger_utilisation-rate"]})
    DM_transport_new["ots"]["passenger_utilization-rate"] = dm_utilisation.filter(
        {"Years": years_ots}
    )

    ###################################
    ####     EMISSION FACTORS     #####
    ###################################
    cdm_emissions_factors = DM_input["emission_factors"]
    DM_transport_new["constant"] = cdm_emissions_factors

    ###################################
    ####    FREIGHT OTS DATA      #####
    ###################################
    DM_transport_new["ots"]["freight_tkm"] = DM_input["freight_tkm"]
    DM_transport_new["ots"]["freight_modal-share"] = DM_input["freight_modal-share"]

    ###################################
    ####    FREIGHT FXA DATA      #####
    ###################################
    DM_transport_new["fxa"]["freight_mode_other"] = DM_input["freight_mode_other"]
    DM_transport_new["fxa"]["freight_tech"] = DM_input["freight_tech"]

    ###################################
    ####    FREIGHT UTI RATE OTS  #####
    ###################################
    DM_transport_new["ots"]["freight_utilization-rate"] = DM_input[
        "freight_utilization-rate"
    ]

    ###################################
    ####    FREIGHT MODE ROAD FXA #####
    ###################################
    DM_transport_new["fxa"]["freight_mode_road"] = DM_input["freight_mode_road"]

    ###############################################
    ####    FREIGHT EFFICIENCY + TECH SHARE   #####
    ###############################################
    DM_transport_new["ots"]["freight_vehicle-efficiency_new"] = DM_input[
        "freight_vehicle-efficiency_new"
    ]
    DM_transport_new["ots"]["freight_technology-share_new"] = DM_input[
        "freight_technology-share_new"
    ]

    # Snapshot before aviation is merged in — freight is included, aviation rows are not.
    # fts_bau_pickle_run reads from this dict so aviation OTS rows don't bleed into FTS extrapolation.
    # The full DM_transport_new (with aviation in OTS) is also returned so callers have a consistent view.
    DM_transport_wo_aviation = copy.deepcopy(DM_transport_new)

    # --- Integrate aviation OTS ---
    for key in [
        "passenger_veh-efficiency_new",
        "passenger_technology-share_new",
        "passenger_occupancy",
        "passenger_utilization-rate",
    ]:
        _add_aviation_to_dm(
            DM_transport_new["ots"][key], DM_aviation_ots["ots"][key], country_list
        )

    # --- Integrate aviation FXA ---
    _add_aviation_to_dm(
        DM_transport_new["fxa"]["passenger_tech"],
        DM_aviation_ots["fxa"]["passenger_tech"],
        country_list,
    )
    _add_aviation_to_dm(
        DM_transport_new["fxa"]["passenger_vehicle-lifetime"],
        DM_aviation_ots["fxa"]["passenger_vehicle-lifetime"],
        country_list,
    )
    DM_transport_new["fxa"]["vehicles-max"] = DM_aviation_ots["fxa"][
        "vehicles-max"
    ].copy()
    _expand_ch_to_countries(DM_transport_new["fxa"]["vehicles-max"], country_list)
    DM_transport_new["fxa"]["share-local-emissions"] = DM_aviation_ots["fxa"][
        "share-local-emissions"
    ].copy()
    _expand_ch_to_countries(
        DM_transport_new["fxa"]["share-local-emissions"], country_list
    )
    DM_transport_new["fxa"]["fuel-mix-availability"] = DM_aviation_ots["fxa"][
        "fuel-mix-availability"
    ].copy()
    _expand_ch_to_countries(
        DM_transport_new["fxa"]["fuel-mix-availability"], country_list
    )

    ###################################
    ####     FUEL-MIX OTS        #####
    ###################################
    dm_fuel_mix = DataMatrix(
        col_labels={
            "Country": ["Switzerland"],
            "Years": years_ots,
            "Variables": ["tra_fuel-mix"],
            "Categories1": ["biofuel", "efuel"],
            "Categories2": ["IWW", "aviation", "marine", "rail", "road"],
        },
        units={"tra_fuel-mix": "%"},
    )
    idx = dm_fuel_mix.idx
    for yr, val in _CH_BIOFUEL_BLEND.items():
        if yr in idx:
            dm_fuel_mix.array[
                idx["Switzerland"], idx[yr], 0, idx["biofuel"], idx["road"]
            ] = val
            dm_fuel_mix.array[
                idx["Switzerland"], idx[yr], 0, idx["biofuel"], idx["rail"]
            ] = val
    _expand_ch_to_countries(dm_fuel_mix, country_list)
    DM_transport_new["ots"]["fuel-mix"] = dm_fuel_mix

    this_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(this_dir, "../../../../data/datamatrix/transport.pickle")
    my_pickle_dump(DM_new=DM_transport_new, local_pickle_file=pickle_file)
    sort_pickle(pickle_file)

    return DM_transport_wo_aviation, DM_transport_new
