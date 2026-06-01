import os

import numpy as np
from scenarios.aviation_fts_CH import run as aviation_fts_run

from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    dm_add_missing_variables,
    linear_fitting,
    my_pickle_dump,
    sort_pickle,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def _add_aviation_fts_to_dm(dm_target, dm_aviation_ch, country_list):
    """Append an aviation FTS sub-DM (Switzerland) to a multi-modal target DM.

    Aligns Categories2 (road techs vs aviation techs) before appending.
    """
    dm_av = dm_aviation_ch.copy()
    for country in country_list:
        if country not in dm_av.col_labels["Country"]:
            dm_c = dm_av.filter({"Country": ["Switzerland"]})
            dm_c.rename_col("Switzerland", country, dim="Country")
            dm_av.append(dm_c, dim="Country")
    if "Categories2" in dm_target.col_labels and "Categories2" in dm_av.col_labels:
        cat2_all = set(dm_target.col_labels["Categories2"]).union(
            set(dm_av.col_labels["Categories2"])
        )
        dm_add_missing_variables(dm_target, {"Categories2": cat2_all})
        dm_add_missing_variables(dm_av, {"Categories2": cat2_all})
    dm_target.append(dm_av, dim="Categories1")


def forecast_vkm_cap(dm_km, years_fts):
    based_on_years = create_years_list(2010, 2019, 1)
    linear_fitting(dm_km, years_fts, based_on=based_on_years, min_tb=0)
    # For metrotram extrapolate with flat line
    idx = dm_km.idx
    dm_km.array[
        idx["Switzerland"],
        idx[2025] : idx[2050] + 1,
        idx["tra_vkm-cap"],
        idx["metrotram"],
    ] = dm_km["Switzerland", 2025, "tra_vkm-cap", "metrotram"]
    # Make sure 2W vkm <= pkm (occupancy > 1)
    mask = dm_km[:, :, "tra_vkm-cap", "2W"] > dm_km[:, :, "tra_pkm-cap", "2W"]
    dm_km[:, :, "tra_vkm-cap", "2W"][mask] = dm_km[:, :, "tra_pkm-cap", "2W"][mask]
    return dm_km


def run(DM_transport_new, country_list, years_ots, years_fts, DM_aviation_ots):
    DM_aviation_fts = aviation_fts_run(DM_aviation_ots["_state"], years_fts)

    # SECTION Modal-share and Transport demand pkm fts
    # pkm/cap * modal-share[%]
    dm_pkm_cap_tot = DM_transport_new["ots"]["pkm"].copy()
    dm_share = DM_transport_new["ots"]["passenger_modal-share"].copy()
    arr_pkm_cap = dm_pkm_cap_tot[..., np.newaxis] * dm_share[...]
    dm_pkm_cap = DataMatrix.based_on(
        arr_pkm_cap,
        format=dm_share,
        change={"Variables": ["tra_pkm-cap"]},
        units={"tra_pkm-cap": "pkm/cap"},
    )
    based_on_years = create_years_list(2010, 2019, 1)
    linear_fitting(dm_pkm_cap, years_fts, based_on=based_on_years)

    # For Switzerland metrotram use flat extrapolation
    idx = dm_pkm_cap.idx
    dm_pkm_cap[
        idx["Switzerland"],
        idx[2025] : idx[2050] + 1,
        idx["tra_pkm-cap"],
        idx["metrotram"],
    ] = dm_pkm_cap.array[
        idx["Switzerland"], idx[2025], idx["tra_pkm-cap"], idx["metrotram"]
    ]
    dm_modal_share = dm_pkm_cap.normalise(dim="Categories1", inplace=False)
    dm_modal_share.rename_col(
        "tra_pkm-cap_share", "tra_passenger_modal-share", dim="Variables"
    )

    DM_transport_new["fts"]["passenger_modal-share"] = dict()
    for lev in range(4):
        DM_transport_new["fts"]["passenger_modal-share"][lev + 1] = (
            dm_modal_share.filter({"Years": years_fts})
        )

    # SECTION Pkm
    dm_pkm_cap_tot = dm_pkm_cap.group_all("Categories1", inplace=False)
    DM_transport_new["fts"]["pkm"] = dict()
    for lev in range(4):
        DM_transport_new["fts"]["pkm"][lev + 1] = dm_pkm_cap_tot.filter(
            {"Years": years_fts}
        )

    # SECTION Technology share new fleet fts
    # For tech share we don't need the forecasting because it is computed from new_fleet
    dm_fleet_new_tech_share = DM_transport_new["ots"][
        "passenger_technology-share_new"
    ].copy()
    dm_fleet_new_tech_share.add(np.nan, dim="Years", col_label=years_fts, dummy=True)
    dm_fleet_new_tech_share.fill_nans("Years")

    DM_transport_new["fts"]["passenger_technology-share_new"] = dict()
    for lev in range(4):
        DM_transport_new["fts"]["passenger_technology-share_new"][lev + 1] = (
            dm_fleet_new_tech_share.filter({"Years": years_fts})
        )

    # SECTION Occupancy pkm/vkm  fts
    dm_occupancy = DM_transport_new["ots"]["passenger_occupancy"].copy()
    dm_add_missing_variables(dm_occupancy, {"Years": years_fts})
    dm_km = dm_occupancy
    dm_km.append(
        dm_pkm_cap.filter({"Categories1": dm_occupancy.col_labels["Categories1"]}),
        dim="Variables",
    )
    # compute vkm/cap
    dm_km.operation(
        "tra_pkm-cap",
        "/",
        "tra_passenger_occupancy",
        out_col="tra_vkm-cap",
        unit="vkm/cap",
    )
    dm_km.drop(dim="Variables", col_label="tra_passenger_occupancy")
    # Create vkm-cap forecasting
    dm_km = forecast_vkm_cap(dm_km, years_fts)
    dm_km.operation(
        "tra_pkm-cap",
        "/",
        "tra_vkm-cap",
        out_col="tra_passenger_occupancy",
        unit="pkm/vkm",
    )
    dm_occupancy = dm_km.filter({"Variables": ["tra_passenger_occupancy"]})

    DM_transport_new["fts"]["passenger_occupancy"] = dict()
    for lev in range(4):
        DM_transport_new["fts"]["passenger_occupancy"][lev + 1] = dm_occupancy.filter(
            {"Years": years_fts}
        )

    # SECTION Utilisation rate vkm/veh fts
    dm_utilisation = DM_transport_new["ots"]["passenger_utilization-rate"].copy()
    linear_fitting(dm_utilisation, years_fts, based_on=create_years_list(2010, 2019, 1))

    DM_transport_new["fts"]["passenger_utilization-rate"] = dict()
    for lev in range(4):
        DM_transport_new["fts"]["passenger_utilization-rate"][lev + 1] = (
            dm_utilisation.filter({"Years": years_fts})
        )

    # SECTION Efficiency fleet
    # For veh-fleet efficiency we can leave the fts to nan because this get re-computed
    dm_veh_new_eff = DM_transport_new["ots"]["passenger_veh-efficiency_new"].copy()
    dm_veh_new_eff.add(np.nan, dim="Years", dummy=True, col_label=years_fts)
    dm_veh_new_eff.fill_nans(dim_to_interp="Years")
    dm_veh_new_eff.filter({"Years": years_fts}, inplace=True)

    DM_transport_new["fts"]["passenger_veh-efficiency_new"] = dict()
    for lev in range(4):
        DM_transport_new["fts"]["passenger_veh-efficiency_new"][lev + 1] = (
            dm_veh_new_eff.filter({"Years": years_fts})
        )

    this_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Integrate aviation FTS (all 4 levels) into the full transport FTS ---
    fts_av = DM_aviation_fts["fts"]
    fts_keys_with_aviation = [
        "passenger_occupancy",
        "passenger_veh-efficiency_new",
        "passenger_utilization-rate",
        "passenger_technology-share_new",
    ]
    for lev in range(1, 5):
        for key in fts_keys_with_aviation:
            _add_aviation_fts_to_dm(
                DM_transport_new["fts"][key][lev], fts_av[key][lev], country_list
            )

    # Aviation pkm and fuel-mix are separate keys (not multi-modal)
    for lev in range(1, 5):
        # passenger_aviation-pkm: set Switzerland and Vaud
        dm_pkm_av = fts_av["passenger_aviation-pkm"][lev].copy()
        for country in country_list:
            if country not in dm_pkm_av.col_labels["Country"]:
                dm_c = dm_pkm_av.filter({"Country": ["Switzerland"]})
                dm_c.rename_col("Switzerland", country, dim="Country")
                dm_pkm_av.append(dm_c, dim="Country")
        DM_transport_new["fts"]["passenger_aviation-pkm"] = DM_transport_new["fts"].get(
            "passenger_aviation-pkm", {}
        )
        DM_transport_new["fts"]["passenger_aviation-pkm"][lev] = dm_pkm_av

        # fuel-mix: aviation FTS + zeros for non-aviation modes (no biofuel/efuel in BAU)
        dm_fm_av = fts_av["fuel-mix"][lev].copy()
        dm_fm_av.add(
            0,
            col_label=["IWW", "marine", "rail", "road"],
            dim="Categories2",
            dummy=True,
        )
        dm_fm_av.sort("Categories2")
        for country in country_list:
            if country not in dm_fm_av.col_labels["Country"]:
                dm_c = dm_fm_av.filter({"Country": ["Switzerland"]})
                dm_c.rename_col("Switzerland", country, dim="Country")
                dm_fm_av.append(dm_c, dim="Country")
        if "fuel-mix" not in DM_transport_new["fts"]:
            DM_transport_new["fts"]["fuel-mix"] = {}
        DM_transport_new["fts"]["fuel-mix"][lev] = dm_fm_av

    pickle_file = os.path.join(this_dir, "../../../../data/datamatrix/transport.pickle")
    DM_to_dump = {"fts": DM_transport_new["fts"]}
    my_pickle_dump(DM_new=DM_to_dump, local_pickle_file=pickle_file)
    sort_pickle(pickle_file)

    return DM_transport_new
