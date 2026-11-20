import os
import pickle
import warnings

import numpy as np
import pandas as pd

from transition_compass_model.model.common.auxiliary_functions import linear_fitting

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(_THIS_DIR, "../data/datamatrix/transport_fts_EU.pickle")
_LIFESTYLES_PATH = os.path.join(
    _THIS_DIR, "../../../lifestyles/Europe/data/lifestyles_allcountries.pickle"
)


def _make_fts(
    dm,
    variable,
    year_start,
    year_end,
    years_fts,
    country="EU27",
    dim="Categories1",
    min_t0=0,
    min_tb=0,
):
    """Project one variable/category to FTS years via linear fit on [year_start, year_end]."""
    dm = dm.copy()
    idx = dm.idx
    based_on_yars = list(range(year_start, year_end + 1, 1))
    dm_temp = linear_fitting(
        dm.filter({"Country": [country], dim: [variable]}),
        years_ots=years_fts,
        min_t0=min_t0,
        min_tb=min_tb,
        based_on=based_on_yars,
    )
    idx_temp = dm_temp.idx
    if dim == "Variables":
        dm.array[idx[country], :, idx[variable], ...] = dm_temp.array[
            idx_temp[country], :, idx_temp[variable], ...
        ]
    if dim == "Categories1":
        dm.array[idx[country], :, :, idx[variable]] = dm_temp.array[
            idx_temp[country], :, :, idx_temp[variable]
        ]
    if dim == "Categories2":
        dm.array[idx[country], :, :, :, idx[variable]] = dm_temp.array[
            idx_temp[country], :, :, :, idx_temp[variable]
        ]
    if dim == "Categories3":
        dm.array[idx[country], :, :, :, :, idx[variable]] = dm_temp.array[
            idx_temp[country], :, :, :, :, idx_temp[variable]
        ]
    return dm


def run(
    dm_occ_ots,
    dm_vkm_ots,
    dm_pkm_avi_terr_ots,
    dm_uti_ots,
    dm_new_ots,
    dm_eneff_new_ots,
    DM_pkm,
    dm_freight_uti,
    DM_tkm_freight,
    DM_tkmtot_freight,
    dm_new_freight_ots,
    dm_eneff_freight_ots,
    dm_fuelmix_ots,
    years_ots,
    years_fts,
):
    """
    Compute FTS scenarios (levels 1-4) for all EU27 levers.

    Covers: occupancy, utilization-rate, technology-share_new, veh-efficiency_new,
    pkm, freight_utilization-rate, freight_modal-share, freight_tkm,
    freight_technology-share_new, freight_vehicle-efficiency_new, fuel-mix.

    Parameters
    ----------
    dm_occ_ots : DataMatrix
        OTS occupancy from passenger_occupancy.run().
    dm_vkm_ots : DataMatrix
        OTS vehicle-km from passenger_occupancy.run() (Variable='tra_passenger_vkm',
        Cat1=modes, unit=vkm).
    dm_pkm_avi_terr_ots : DataMatrix
        OTS territorial aviation pkm from passenger_occupancy.run()
        (Variable='tra_passenger_pkm', Cat1=['aviation'], unit=pkm).
    dm_uti_ots : DataMatrix
        OTS utilization rate from passenger_utilization_rate.run().
    dm_new_ots : DataMatrix
        OTS passenger tech share new from passenger_tech_share_new.run().
    DM_pkm : dict
        Full pkm dict from passenger_modal_share.run() (OTS + FTS level-1 ground pkm).
    dm_freight_uti : DataMatrix
        OTS freight utilization rate from freight_utilization_rate.run().
    DM_tkm_freight, DM_tkmtot_freight : dict
        OTS freight tkm dicts from freight_modal_share.run().
    dm_new_freight_ots : DataMatrix
        OTS freight tech share new from freight_tech_share_new.run().
        Variable='tra_freight_technology-share_new', Cat1=modes, Cat2=techs.
    dm_eneff_freight_ots : DataMatrix
        OTS freight vehicle efficiency new from freight_veh_efficiency_new.run().
        Variable='tra_freight_vehicle-efficiency_new', Cat1=modes, Cat2=techs.
    dm_fuelmix_ots : DataMatrix
        OTS fuel mix from fuel_mix.run().
        Variable='tra_fuel-mix', Cat1=[biofuel, efuel], Cat2=[IWW, aviation, marine, rail, road].
    years_ots, years_fts : list

    Returns
    -------
    DM_fts : dict
        Keys: passenger_occupancy, passenger_utilization-rate,
              passenger_technology-share_new, passenger_veh-efficiency_new, pkm,
              freight_utilization-rate, freight_modal-share, freight_tkm,
              freight_technology-share_new, freight_vehicle-efficiency_new, fuel-mix.
        Each key maps to {1..4: DataMatrix}.
    """
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    baseyear_start = 2000
    baseyear_end = 2019

    DM_fts = {
        "passenger_occupancy": dict(),
        "passenger_utilization-rate": dict(),
        "passenger_technology-share_new": dict(),
        "passenger_veh-efficiency_new": dict(),
        "pkm": dict(),
        "freight_utilization-rate": dict(),
        "freight_modal-share": dict(),
        "freight_tkm": dict(),
        "freight_technology-share_new": dict(),
        "freight_vehicle-efficiency_new": dict(),
        "fuel-mix": dict(),
    }

    ##############################
    ##### OCCUPANCY FTS      #####
    ##############################

    # Project vkm to FTS (BAU trend, same approach as old EU lever script)
    dm_vkm = dm_vkm_ots.copy()
    dm_vkm.add(np.nan, col_label=years_fts, dummy=True, dim="Years")
    idx_vkm = dm_vkm.idx
    for y in years_fts:
        dm_vkm.array[idx_vkm["EU27"], idx_vkm[y], :, idx_vkm["2W"]] = dm_vkm.array[
            idx_vkm["EU27"], idx_vkm[2023], :, idx_vkm["2W"]
        ]
    for mode in ["LDV", "bus", "metrotram", "rail", "aviation"]:
        dm_vkm = _make_fts(
            dm_vkm, mode, baseyear_start, baseyear_end, years_fts, dim="Categories1"
        )

    # Project territorial aviation pkm to FTS
    dm_pkm_avi_terr = dm_pkm_avi_terr_ots.copy()
    dm_pkm_avi_terr.add(np.nan, col_label=years_fts, dummy=True, dim="Years")
    dm_pkm_avi_terr = _make_fts(
        dm_pkm_avi_terr,
        "aviation",
        baseyear_start,
        baseyear_end,
        years_fts,
        dim="Categories1",
    )

    # Combine ground pkm (OTS + level-1 FTS) with territorial aviation pkm
    dm_pkm = DM_pkm["ots"]["passenger_pkm"].copy()
    dm_pkm.append(DM_pkm["fts"]["passenger_pkm"][1], "Years")
    dm_pkm.sort("Years")
    dm_pkm.append(dm_pkm_avi_terr, "Categories1")
    dm_pkm.sort("Categories1")

    # Compute occupancy OTS+FTS = pkm / vkm
    dm_occ = dm_pkm.copy()
    dm_occ.array = dm_occ.array / dm_vkm.array
    dm_occ.rename_col("tra_passenger_pkm", "tra_passenger_occupancy", "Variables")
    dm_occ.units["tra_passenger_occupancy"] = "pkm/vkm"

    # Normalize aviation per country (same fix as processor OTS)
    idx_norm = dm_occ.idx
    for c in dm_occ.col_labels["Country"]:
        avi = dm_occ.array[
            idx_norm[c], :, idx_norm["tra_passenger_occupancy"], idx_norm["aviation"]
        ]
        max_val = np.nanmax(avi)
        if max_val > 1.0:
            dm_occ.array[
                idx_norm[c],
                :,
                idx_norm["tra_passenger_occupancy"],
                idx_norm["aviation"],
            ] /= max_val

    # Level 1: BAU continuation
    dm_occ_fts_level1 = dm_occ.filter({"Years": years_fts})

    # Level 4: ITF 2023 targets
    # LDV source: page 56-57 https://www.itf-oecd.org/sites/default/files/docs/itf-transport-outlook-2023-launch.pdf
    # bus/metrotram/rail: scale by same relative improvement as bus; aviation: full occupancy = 1
    dm_occ_level4 = dm_occ.copy()
    idx = dm_occ_level4.idx
    for y in range(2030, 2055, 5):
        dm_occ_level4.array[idx["EU27"], idx[y], :, :] = np.nan
    dm_occ_level4.array[idx["EU27"], idx[2050], :, idx["LDV"]] = dm_occ_level4.array[
        idx["EU27"], idx[2023], :, idx["LDV"]
    ] * (1 + 0.0815)
    dm_occ_level4.array[idx["EU27"], idx[2050], :, idx["2W"]] = 1.3
    dm_occ_level4.array[idx["EU27"], idx[2050], :, idx["bus"]] = 27.2
    bus_2023 = dm_occ_level4.array[idx["EU27"], idx[2023], :, idx["bus"]]
    bus_2050 = dm_occ_level4.array[idx["EU27"], idx[2050], :, idx["bus"]]
    rate_bus = (bus_2050 - bus_2023) / bus_2023
    dm_occ_level4.array[idx["EU27"], idx[2050], :, idx["metrotram"]] = (
        dm_occ_level4.array[idx["EU27"], idx[2023], :, idx["metrotram"]]
        * (1 + rate_bus)
    )
    dm_occ_level4.array[idx["EU27"], idx[2050], :, idx["rail"]] = dm_occ_level4.array[
        idx["EU27"], idx[2023], :, idx["rail"]
    ] * (1 + rate_bus)
    dm_occ_level4.array[idx["EU27"], idx[2050], :, idx["aviation"]] = 1
    dm_occ_level4 = linear_fitting(dm_occ_level4, years_fts)
    dm_occ_fts_level4 = dm_occ_level4.filter({"Years": years_fts})

    # Interpolate levels 2 and 3 between level 1 and 4 at 2050
    variabs_occ = dm_occ_fts_level1.col_labels["Categories1"]
    df_temp = pd.DataFrame(
        {
            "level": np.tile(range(1, 4 + 1), len(variabs_occ)),
            "variable": np.repeat(variabs_occ, 4),
        }
    )
    df_temp["value"] = np.nan
    df_temp = df_temp.pivot(
        index=["level"], columns=["variable"], values="value"
    ).reset_index()
    for v in variabs_occ:
        idx = dm_occ_fts_level1.idx
        level1 = dm_occ_fts_level1.array[idx["EU27"], idx[2050], :, idx[v]][0]
        idx = dm_occ_fts_level4.idx
        level4 = dm_occ_fts_level4.array[idx["EU27"], idx[2050], :, idx[v]][0]
        arr = np.array([level1, np.nan, np.nan, level4])
        df_temp[v] = pd.Series(arr).interpolate().to_numpy()

    dm_occ_level2 = dm_occ.copy()
    idx = dm_occ_level2.idx
    for y in range(2030, 2055, 5):
        dm_occ_level2.array[idx["EU27"], idx[y], :, :] = np.nan
    for v in variabs_occ:
        dm_occ_level2.array[idx["EU27"], idx[2050], :, idx[v]] = df_temp.loc[
            df_temp["level"] == 2, v
        ]
    dm_occ_level2 = linear_fitting(dm_occ_level2, years_fts)
    dm_occ_fts_level2 = dm_occ_level2.filter({"Years": years_fts})

    dm_occ_level3 = dm_occ.copy()
    idx = dm_occ_level3.idx
    for y in range(2030, 2055, 5):
        dm_occ_level3.array[idx["EU27"], idx[y], :, :] = np.nan
    for v in variabs_occ:
        dm_occ_level3.array[idx["EU27"], idx[2050], :, idx[v]] = df_temp.loc[
            df_temp["level"] == 3, v
        ]
    dm_occ_level3 = linear_fitting(dm_occ_level3, years_fts)
    dm_occ_fts_level3 = dm_occ_level3.filter({"Years": years_fts})

    DM_fts["passenger_occupancy"][1] = dm_occ_fts_level1
    DM_fts["passenger_occupancy"][2] = dm_occ_fts_level2
    DM_fts["passenger_occupancy"][3] = dm_occ_fts_level3
    DM_fts["passenger_occupancy"][4] = dm_occ_fts_level4

    #####################################
    ##### UTILIZATION RATE FTS      #####
    #####################################

    dm_uti_level4 = dm_uti_ots.copy()
    dm_uti_level4.add(np.nan, col_label=years_fts, dummy=True, dim="Years")
    years_fts_range = list(range(2025, 2055, 5))
    idx = dm_uti_level4.idx
    for y in years_fts_range:
        dm_uti_level4.array[idx["EU27"], idx[y], :, :] = np.nan
    dm_uti_level4.array[idx["EU27"], idx[2050], :, idx["LDV"]] = dm_uti_level4.array[
        idx["EU27"], idx[2023], :, idx["LDV"]
    ] * (1 + 0.30)  # I do not put 900% here as it would blow all the results
    dm_uti_level4.array[idx["EU27"], idx[2050], :, idx["2W"]] = dm_uti_level4.array[
        idx["EU27"], idx[2023], :, idx["2W"]
    ] * (1 + 0.15)
    dm_uti_level4.array[idx["EU27"], idx[2050], :, idx["bus"]] = dm_uti_level4.array[
        idx["EU27"], idx[2023], :, idx["bus"]
    ] * (1 + 0.45)
    dm_uti_level4.array[idx["EU27"], idx[2050], :, idx["aviation"]] = (
        dm_uti_level4.array[idx["EU27"], idx[2023], :, idx["aviation"]] * (1 + 0.45)
    )  # TODO: temporary as bus, re do this later
    dm_uti_level4.array[idx["EU27"], idx[2050], :, idx["rail"]] = dm_uti_level4.array[
        idx["EU27"], idx[2023], :, idx["rail"]
    ] * (1 + 0.45)
    dm_uti_level4.array[idx["EU27"], idx[2050], :, idx["metrotram"]] = (
        dm_uti_level4.array[idx["EU27"], idx[2023], :, idx["metrotram"]] * (1 + 0.45)
    )
    dm_uti_level4 = linear_fitting(dm_uti_level4, years_fts_range)
    dm_uti_fts_level4 = dm_uti_level4.filter({"Years": years_fts})

    # Level 1: BAU continuation (2W flat at 2023 — declining uti_rate trend causes fleet explosion)
    dm_uti_bau = dm_uti_ots.copy()
    dm_uti_bau.add(np.nan, col_label=years_fts, dummy=True, dim="Years")
    dm_uti_bau = linear_fitting(dm_uti_bau, years_fts)
    idx_uti = dm_uti_bau.idx
    for y in years_fts:
        dm_uti_bau.array[idx_uti["EU27"], idx_uti[y], :, idx_uti["2W"]] = (
            dm_uti_bau.array[idx_uti["EU27"], idx_uti[2023], :, idx_uti["2W"]]
        )
    dm_uti_fts_level1 = dm_uti_bau.filter({"Years": years_fts})

    # Interpolate levels 2 and 3
    variabs_uti = dm_uti_fts_level1.col_labels["Categories1"]
    df_temp = pd.DataFrame(
        {
            "level": np.tile(range(1, 4 + 1), len(variabs_uti)),
            "variable": np.repeat(variabs_uti, 4),
        }
    )
    df_temp["value"] = np.nan
    df_temp = df_temp.pivot(
        index=["level"], columns=["variable"], values="value"
    ).reset_index()
    for v in variabs_uti:
        idx = dm_uti_fts_level1.idx
        level1 = dm_uti_fts_level1.array[idx["EU27"], idx[2050], :, idx[v]][0]
        idx = dm_uti_fts_level4.idx
        level4 = dm_uti_fts_level4.array[idx["EU27"], idx[2050], :, idx[v]][0]
        arr = np.array([level1, np.nan, np.nan, level4])
        df_temp[v] = pd.Series(arr).interpolate().to_numpy()

    dm_uti_level2 = dm_uti_ots.copy()
    dm_uti_level2.add(np.nan, col_label=years_fts, dummy=True, dim="Years")
    idx = dm_uti_level2.idx
    for y in range(2030, 2055, 5):
        dm_uti_level2.array[idx["EU27"], idx[y], :, :] = np.nan
    for v in variabs_uti:
        dm_uti_level2.array[idx["EU27"], idx[2050], :, idx[v]] = df_temp.loc[
            df_temp["level"] == 2, v
        ]
    dm_uti_level2 = linear_fitting(dm_uti_level2, years_fts)
    dm_uti_fts_level2 = dm_uti_level2.filter({"Years": years_fts})

    dm_uti_level3 = dm_uti_ots.copy()
    dm_uti_level3.add(np.nan, col_label=years_fts, dummy=True, dim="Years")
    idx = dm_uti_level3.idx
    for y in range(2030, 2055, 5):
        dm_uti_level3.array[idx["EU27"], idx[y], :, :] = np.nan
    for v in variabs_uti:
        dm_uti_level3.array[idx["EU27"], idx[2050], :, idx[v]] = df_temp.loc[
            df_temp["level"] == 3, v
        ]
    dm_uti_level3 = linear_fitting(dm_uti_level3, years_fts)
    dm_uti_fts_level3 = dm_uti_level3.filter({"Years": years_fts})

    DM_fts["passenger_utilization-rate"][1] = dm_uti_fts_level1
    DM_fts["passenger_utilization-rate"][2] = dm_uti_fts_level2
    DM_fts["passenger_utilization-rate"][3] = dm_uti_fts_level3
    DM_fts["passenger_utilization-rate"][4] = dm_uti_fts_level4

    #####################################
    ##### TECH SHARE NEW FTS        #####
    #####################################

    # Level 1: BAU continuation — project OTS tech shares to FTS via linear trend
    dm_new_bau = dm_new_ots.copy()
    dm_new_bau.add(np.nan, col_label=years_fts, dummy=True, dim="Years")
    dm_new_bau = linear_fitting(dm_new_bau, years_fts)
    dm_new_fts_level1 = dm_new_bau.filter({"Years": years_fts}).flatten()

    # Level 4: Green Deal — no new emission vehicles after 2035
    dm_new_level4 = dm_new_ots.copy()
    dm_new_level4.add(np.nan, col_label=years_fts, dummy=True, dim="Years")
    dm_new_level4 = dm_new_level4.flatten()
    idx = dm_new_level4.idx
    for y in list(range(2030, 2055, 5)):
        dm_new_level4.array[idx["EU27"], idx[y], :, :] = np.nan
    for y in range(2035, 2055, 5):
        dm_new_level4.array[idx["EU27"], idx[y], :, idx["LDV_ICE-gasoline"]] = 0
        dm_new_level4.array[idx["EU27"], idx[y], :, idx["LDV_ICE-gas"]] = 0
        dm_new_level4.array[idx["EU27"], idx[y], :, idx["LDV_ICE-diesel"]] = 0
        dm_new_level4.array[idx["EU27"], idx[y], :, idx["LDV_PHEV-gasoline"]] = 0
        dm_new_level4.array[idx["EU27"], idx[y], :, idx["LDV_PHEV-diesel"]] = 0
        dm_new_level4.array[idx["EU27"], idx[y], :, idx["LDV_FCEV"]] = 0.1
        dm_new_level4.array[idx["EU27"], idx[y], :, idx["LDV_BEV"]] = 0.9
        dm_new_level4.array[idx["EU27"], idx[y], :, idx["2W_ICE-gasoline"]] = 1
        dm_new_level4.array[idx["EU27"], idx[y], :, idx["bus_ICE-gasoline"]] = 0
        dm_new_level4.array[idx["EU27"], idx[y], :, idx["bus_ICE-gas"]] = 0
        dm_new_level4.array[idx["EU27"], idx[y], :, idx["bus_ICE-diesel"]] = 0
        dm_new_level4.array[idx["EU27"], idx[y], :, idx["bus_BEV"]] = 1
        dm_new_level4.array[idx["EU27"], idx[y], :, idx["metrotram_mt"]] = 1
        dm_new_level4.array[idx["EU27"], idx[y], :, idx["rail_ICE-diesel"]] = 0
        dm_new_level4.array[idx["EU27"], idx[y], :, idx["rail_CEV"]] = 1
    dm_new_level4.array[idx["EU27"], idx[2050], :, idx["aviation_kerosene"]] = (
        0.87  # TODO: SAF should be in here (it should be around 70%, while remaining 27% should be kerosene), to be checked with paola
    )
    dm_new_level4.array[idx["EU27"], idx[2050], :, idx["aviation_H2"]] = 0.10
    dm_new_level4.array[idx["EU27"], idx[2050], :, idx["aviation_BEV"]] = 0.03
    dm_new_level4 = linear_fitting(dm_new_level4, years_fts)
    dm_new_fts_level4 = dm_new_level4.filter({"Years": years_fts})

    # Interpolate levels 2 and 3
    variabs_new = dm_new_fts_level1.col_labels["Categories1"]
    df_temp = pd.DataFrame(
        {
            "level": np.tile(range(1, 4 + 1), len(variabs_new)),
            "variable": np.repeat(variabs_new, 4),
        }
    )
    df_temp["value"] = np.nan
    df_temp = df_temp.pivot(
        index=["level"], columns=["variable"], values="value"
    ).reset_index()
    for v in variabs_new:
        idx = dm_new_fts_level1.idx
        level1 = dm_new_fts_level1.array[idx["EU27"], idx[2050], :, idx[v]][0]
        idx = dm_new_fts_level4.idx
        level4 = dm_new_fts_level4.array[idx["EU27"], idx[2050], :, idx[v]][0]
        arr = np.array([level1, np.nan, np.nan, level4])
        df_temp[v] = pd.Series(arr).interpolate().to_numpy()

    dm_new_level2 = dm_new_ots.flatten()
    dm_new_level2.add(np.nan, col_label=years_fts, dummy=True, dim="Years")
    idx = dm_new_level2.idx
    for y in range(2030, 2055, 5):
        dm_new_level2.array[idx["EU27"], idx[y], :, :] = np.nan
    for v in variabs_new:
        dm_new_level2.array[idx["EU27"], idx[2050], :, idx[v]] = df_temp.loc[
            df_temp["level"] == 2, v
        ]
    dm_new_level2 = linear_fitting(dm_new_level2, years_fts)
    dm_new_fts_level2 = dm_new_level2.filter({"Years": years_fts})

    dm_new_level3 = dm_new_ots.flatten()
    dm_new_level3.add(np.nan, col_label=years_fts, dummy=True, dim="Years")
    idx = dm_new_level3.idx
    for y in range(2030, 2055, 5):
        dm_new_level3.array[idx["EU27"], idx[y], :, :] = np.nan
    for v in variabs_new:
        dm_new_level3.array[idx["EU27"], idx[2050], :, idx[v]] = df_temp.loc[
            df_temp["level"] == 3, v
        ]
    dm_new_level3 = linear_fitting(dm_new_level3, years_fts)
    dm_new_fts_level3 = dm_new_level3.filter({"Years": years_fts})

    # deepen all FTS levels
    dm_new_fts_level1.deepen()
    dm_new_fts_level2.deepen()
    dm_new_fts_level3.deepen()
    dm_new_fts_level4.deepen()

    DM_fts["passenger_technology-share_new"][1] = dm_new_fts_level1
    DM_fts["passenger_technology-share_new"][2] = dm_new_fts_level2
    DM_fts["passenger_technology-share_new"][3] = dm_new_fts_level3
    DM_fts["passenger_technology-share_new"][4] = dm_new_fts_level4

    #############################################
    ##### PASSENGER VEH EFFICIENCY NEW FTS  #####
    #############################################

    # Faithful port of transport_lever_passenger_veh-efficiency_new.py.
    # OTS dm is deepened (Cat1=mode, Cat2=tech, unit=MJ/km); flatten to mode_tech combos.
    # Old script used dim="Variables" on a flat dm; here mode_tech combos are in Categories1.
    dm_eneff_new = dm_eneff_new_ots.flatten()
    dm_eneff_new.add(np.nan, col_label=years_fts, dummy=True, dim="Years")

    for combo in [
        "2W_ICE-gasoline",
        "LDV_BEV",
        "LDV_ICE-diesel",
        "LDV_ICE-gas",
        "LDV_ICE-gasoline",
        "LDV_PHEV-diesel",
        "LDV_PHEV-gasoline",
        "bus_BEV",
        "bus_ICE-diesel",
        "bus_ICE-gas",
        "bus_ICE-gasoline",
        "metrotram_mt",
        "rail_CEV",
        "rail_ICE-diesel",
    ]:
        dm_eneff_new = _make_fts(
            dm_eneff_new,
            combo,
            baseyear_start,
            baseyear_end,
            years_fts,
            dim="Categories1",
            min_t0=0.1,
            min_tb=0.1,
        )
    dm_eneff_new = _make_fts(
        dm_eneff_new,
        "aviation_kerosene",
        baseyear_start,
        baseyear_end,
        years_fts,
        dim="Categories1",
        min_t0=0,
        min_tb=0,
    )

    dm_eneff_new_fts = dm_eneff_new.filter({"Years": years_fts})
    dm_eneff_new_fts.deepen()
    dm_eneff_new_fts.add(np.nan, col_label="FCEV", dummy=True, dim="Categories2")
    dm_eneff_new_fts.sort("Categories2")
    dm_eneff_new_fts.add(np.nan, "Categories2", "H2", "MJ/km", True)
    dm_eneff_new_fts.sort("Categories2")

    for i in range(1, 5):
        DM_fts["passenger_veh-efficiency_new"][i] = dm_eneff_new_fts

    ###########################
    ##### PKM (pkm/cap) FTS ###
    ###########################

    # Faithful port of transport_lever_pkm.py.
    with open(_LIFESTYLES_PATH, "rb") as f:
        DM_lfs = pickle.load(f)

    dm_pkmtot = DM_pkm["ots"]["passenger_pkm"].copy()
    dm_pkmtot.append(DM_pkm["fts"]["passenger_pkm"][1], "Years")
    dm_pkmtot.sort("Years")
    dm_pkmtot.group_all("Categories1")
    dm_pkmtot = dm_pkmtot.filter({"Country": ["EU27"]})

    dm_pop = DM_lfs["ots"]["pop"]["lfs_population_"].copy()
    dm_pop.append(DM_lfs["fts"]["pop"]["lfs_population_"][1], "Years")
    dm_pop = dm_pop.filter({"Country": ["EU27"]})
    dm_pop.sort("Years")
    dm_pkmtot.sort("Years")
    dm_pkmtot.array = dm_pkmtot.array / dm_pop.array
    dm_pkmtot.units["tra_passenger_pkm"] = "pkm/cap"
    dm_pkmtot.rename_col("tra_passenger_pkm", "tra_pkm-cap", "Variables")

    # Level 1: BAU continuation (already in the combined OTS+FTS[1] series)
    dm_pkmtot_fts_level1 = dm_pkmtot.filter({"Years": years_fts})

    # Level 4: 14% reduction vs 2023 by 2050
    dm_pkmtot_level4 = dm_pkmtot.copy()
    idx = dm_pkmtot_level4.idx
    for y in range(2030, 2055, 5):
        dm_pkmtot_level4.array[idx["EU27"], idx[y], :] = np.nan
    dm_pkmtot_level4.array[idx["EU27"], idx[2050], :] = dm_pkmtot_level4.array[
        idx["EU27"], idx[2023], :
    ] * (1 - 0.14)
    dm_pkmtot_level4 = linear_fitting(dm_pkmtot_level4, years_fts)
    dm_pkmtot_fts_level4 = dm_pkmtot_level4.filter({"Years": years_fts})

    # Interpolate levels 2 and 3
    variabs_pkm = dm_pkmtot_fts_level1.col_labels["Variables"]
    df_temp = pd.DataFrame(
        {
            "level": np.tile(range(1, 4 + 1), len(variabs_pkm)),
            "variable": np.repeat(variabs_pkm, 4),
        }
    )
    df_temp["value"] = np.nan
    df_temp = df_temp.pivot(
        index=["level"], columns=["variable"], values="value"
    ).reset_index()
    for v in variabs_pkm:
        idx = dm_pkmtot_fts_level1.idx
        level1 = dm_pkmtot_fts_level1.array[idx["EU27"], idx[2050], idx[v]]
        idx = dm_pkmtot_fts_level4.idx
        level4 = dm_pkmtot_fts_level4.array[idx["EU27"], idx[2050], idx[v]]
        arr = np.array([level1, np.nan, np.nan, level4])
        df_temp[v] = pd.Series(arr).interpolate().to_numpy()

    dm_pkmtot_level2 = dm_pkmtot.copy()
    idx = dm_pkmtot_level2.idx
    for y in range(2030, 2055, 5):
        dm_pkmtot_level2.array[idx["EU27"], idx[y], :] = np.nan
    for v in variabs_pkm:
        dm_pkmtot_level2.array[idx["EU27"], idx[2050], idx[v]] = df_temp.loc[
            df_temp["level"] == 2, v
        ]
    dm_pkmtot_level2 = linear_fitting(dm_pkmtot_level2, years_fts)
    dm_pkmtot_fts_level2 = dm_pkmtot_level2.filter({"Years": years_fts})

    dm_pkmtot_level3 = dm_pkmtot.copy()
    idx = dm_pkmtot_level3.idx
    for y in range(2030, 2055, 5):
        dm_pkmtot_level3.array[idx["EU27"], idx[y], :] = np.nan
    for v in variabs_pkm:
        dm_pkmtot_level3.array[idx["EU27"], idx[2050], idx[v]] = df_temp.loc[
            df_temp["level"] == 3, v
        ]
    dm_pkmtot_level3 = linear_fitting(dm_pkmtot_level3, years_fts)
    dm_pkmtot_fts_level3 = dm_pkmtot_level3.filter({"Years": years_fts})

    DM_fts["pkm"][1] = dm_pkmtot_fts_level1
    DM_fts["pkm"][2] = dm_pkmtot_fts_level2
    DM_fts["pkm"][3] = dm_pkmtot_fts_level3
    DM_fts["pkm"][4] = dm_pkmtot_fts_level4

    ##########################################
    ##### FREIGHT UTILIZATION RATE FTS   #####
    ##########################################

    # Faithful port of transport_lever_freight_utilization-rate.py.
    # Old script extended dm_hdv (flat, Variables=modes) via make_fts with dim="Variables".
    # Here dm_freight_uti has both variables deepened into Cat1=modes; use dim="Categories1".
    # Trend window 2000-2023 (same as old script's baseyear_end = baseyear = 2023).
    dm_freight_uti_bau = dm_freight_uti.copy()
    dm_freight_uti_bau.add(np.nan, col_label=years_fts, dummy=True, dim="Years")
    for mode in ["HDVH", "HDVL", "HDVM"]:
        dm_freight_uti_bau = _make_fts(
            dm_freight_uti_bau, mode, baseyear_start, 2023, years_fts, dim="Categories1"
        )
    dm_freight_uti_fts = dm_freight_uti_bau.filter({"Years": years_fts})

    for i in range(1, 5):
        DM_fts["freight_utilization-rate"][i] = dm_freight_uti_fts

    ################################################
    ##### FREIGHT MODAL SHARE + TOTAL TKM FTS  #####
    ################################################

    # Faithful port of transport_lever_freight_modal-share.py.
    # Old script extended flat dm_tkm (Variables=modes) via make_fts with dim="Variables".
    # Here DM_tkm_freight["ots"]["freight_tkm"] has Variable='tra_freight_tkm', Cat1=modes.
    # Trend window 2000-2019 (same as old script).
    dm_tkm_f = DM_tkm_freight["ots"]["freight_tkm"].copy()
    dm_tkm_f.add(np.nan, col_label=years_fts, dummy=True, dim="Years")
    for mode in ["HDVH", "HDVL", "HDVM", "IWW", "aviation", "marine", "rail"]:
        dm_tkm_f = _make_fts(
            dm_tkm_f, mode, baseyear_start, baseyear_end, years_fts, dim="Categories1"
        )

    # Modal share: normalize per-mode tkm to fractions
    dm_modal_f = dm_tkm_f.normalise("Categories1", inplace=False, keep_original=False)
    dm_modal_f.rename_col(
        "tra_freight_tkm_share", "tra_freight_modal-share", "Variables"
    )
    dm_modal_fts = dm_modal_f.filter({"Years": years_fts})

    for i in range(1, 5):
        DM_fts["freight_modal-share"][i] = dm_modal_fts

    # Total tkm (bn-tkm): BAU extension of OTS total tkm (2000-2019 trend).
    dm_tkmtot_f = DM_tkmtot_freight["ots"]["freight_tkm"].copy()
    dm_tkmtot_f.add(np.nan, col_label=years_fts, dummy=True, dim="Years")
    dm_tkmtot_f = _make_fts(
        dm_tkmtot_f,
        "tra_freight_tkm-total-demand",
        baseyear_start,
        baseyear_end,
        years_fts,
        dim="Variables",
    )
    dm_tkm_fts = dm_tkmtot_f.filter({"Years": years_fts})

    for i in range(1, 5):
        DM_fts["freight_tkm"][i] = dm_tkm_fts

    ######################################################
    ##### FREIGHT TECHNOLOGY SHARE NEW FTS (BAU)    #####
    ######################################################

    # Faithful BAU port — all 4 levels identical (linear trend continuation).
    # Old script had a more complex BAU (8% BEV target etc.) which is a TODO
    # when differentiating ambition levels.
    dm_new_f = dm_new_freight_ots.copy()
    dm_new_f.add(np.nan, col_label=years_fts, dummy=True, dim="Years")
    dm_new_f = linear_fitting(dm_new_f, years_fts, min_t0=0, min_tb=0)
    dm_new_fts = dm_new_f.filter({"Years": years_fts})

    # Rail FTS: OTS data is too noisy (small absolute numbers, ~1K new vehicles/yr)
    # to trend reliably — the linear fit produces an increasing ICE-diesel share which
    # is physically wrong. Clamp EU27 rail to 2023 OTS endpoint for all FTS years.
    for y in years_fts:
        dm_new_fts["EU27", y, "tra_freight_technology-share_new", "rail", "CEV"] = (
            dm_new_freight_ots[
                "EU27", 2023, "tra_freight_technology-share_new", "rail", "CEV"
            ]
        )
        dm_new_fts[
            "EU27", y, "tra_freight_technology-share_new", "rail", "ICE-diesel"
        ] = dm_new_freight_ots[
            "EU27", 2023, "tra_freight_technology-share_new", "rail", "ICE-diesel"
        ]

    for i in range(1, 5):
        DM_fts["freight_technology-share_new"][i] = dm_new_fts

    ######################################################
    ##### FREIGHT VEHICLE EFFICIENCY NEW FTS (BAU)  #####
    ######################################################

    # BAU — all 4 levels identical (linear trend continuation, min 0.1 to avoid zero).
    dm_eneff_f = dm_eneff_freight_ots.copy()
    dm_eneff_f.add(np.nan, col_label=years_fts, dummy=True, dim="Years")
    dm_eneff_f = linear_fitting(dm_eneff_f, years_fts, min_t0=0.1, min_tb=0.1)
    dm_eneff_fts = dm_eneff_f.filter({"Years": years_fts})

    for i in range(1, 5):
        DM_fts["freight_vehicle-efficiency_new"][i] = dm_eneff_fts

    ##################################
    ##### FUEL-MIX FTS (BAU)    #####
    ##################################

    # BAU — all 4 levels identical (linear trend continuation per mode).
    # Faithful port of make_fts calls in transport_lever_fuel-mix.py:
    #   IWW/marine: trend based on 2021-2022; road/rail/aviation: 2000-2019 trend.
    # Works on final structure: Variable=tra_fuel-mix, Cat1=[biofuel, efuel], Cat2=modes.
    dm_fuelmix = dm_fuelmix_ots.copy()
    dm_fuelmix.add(np.nan, col_label=years_fts, dummy=True, dim="Years")
    dm_fuelmix = _make_fts(dm_fuelmix, "IWW", 2021, 2022, years_fts, dim="Categories2")
    dm_fuelmix = _make_fts(
        dm_fuelmix,
        "aviation",
        baseyear_start,
        baseyear_end,
        years_fts,
        dim="Categories2",
    )
    dm_fuelmix = _make_fts(
        dm_fuelmix, "marine", 2021, 2022, years_fts, dim="Categories2"
    )
    dm_fuelmix = _make_fts(
        dm_fuelmix, "rail", baseyear_start, baseyear_end, years_fts, dim="Categories2"
    )
    dm_fuelmix = _make_fts(
        dm_fuelmix, "road", baseyear_start, baseyear_end, years_fts, dim="Categories2"
    )
    dm_fuelmix_fts = dm_fuelmix.filter({"Years": years_fts})

    for i in range(1, 5):
        DM_fts["fuel-mix"][i] = dm_fuelmix_fts

    ################
    ##### SAVE #####
    ################

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(DM_fts, f)
    return DM_fts


if __name__ == "__main__":
    from transition_compass_model._database.pre_processing.transport.EU.params import (
        years_fts,
        years_ots,
    )
    from transition_compass_model._database.pre_processing.transport.EU.processors.freight_modal_share import (
        run as freight_modal_run,
    )
    from transition_compass_model._database.pre_processing.transport.EU.processors.freight_tech_share_new import (
        run as freight_tech_share_run,
    )
    from transition_compass_model._database.pre_processing.transport.EU.processors.freight_utilization_rate import (
        run as freight_uti_run,
    )
    from transition_compass_model._database.pre_processing.transport.EU.processors.freight_veh_efficiency_new import (
        run as freight_eneff_run,
    )
    from transition_compass_model._database.pre_processing.transport.EU.processors.fuel_mix import (
        run as fuel_mix_run,
    )
    from transition_compass_model._database.pre_processing.transport.EU.processors.fxa_passenger_tech import (
        run as tech_run,
    )
    from transition_compass_model._database.pre_processing.transport.EU.processors.passenger_modal_share import (
        run as modal_run,
    )
    from transition_compass_model._database.pre_processing.transport.EU.processors.passenger_occupancy import (
        run as occ_run,
    )
    from transition_compass_model._database.pre_processing.transport.EU.processors.passenger_tech_share_new import (
        run as tech_share_run,
    )
    from transition_compass_model._database.pre_processing.transport.EU.processors.passenger_utilization_rate import (
        run as uti_run,
    )
    from transition_compass_model._database.pre_processing.transport.EU.processors.passenger_veh_efficiency_new import (
        run as eneff_run,
    )

    _, dm_fleet, dm_new_raw, _ = tech_run(years_ots, years_fts)
    _, DM_pkm = modal_run(years_ots, years_fts)
    dm_occ_ots, dm_vkm_ots, dm_pkm_avi_terr_ots = occ_run(DM_pkm, years_ots)
    dm_uti_ots = uti_run(dm_fleet, dm_vkm_ots, years_ots)
    dm_new_ots = tech_share_run(dm_new_raw, years_ots)
    dm_eneff_new_ots = eneff_run(years_ots)
    _, DM_tkm_freight, DM_tkmtot_freight = freight_modal_run(years_ots, years_fts)
    dm_freight_uti = freight_uti_run(years_ots, years_fts, DM_tkm_freight)
    dm_new_freight_ots = freight_tech_share_run(years_ots, years_fts)
    dm_eneff_freight_ots = freight_eneff_run(years_ots, years_fts)
    dm_fuelmix_ots = fuel_mix_run(years_ots, years_fts)
    run(
        dm_occ_ots,
        dm_vkm_ots,
        dm_pkm_avi_terr_ots,
        dm_uti_ots,
        dm_new_ots,
        dm_eneff_new_ots,
        DM_pkm,
        dm_freight_uti,
        DM_tkm_freight,
        DM_tkmtot_freight,
        dm_new_freight_ots,
        dm_eneff_freight_ots,
        dm_fuelmix_ots,
        years_ots,
        years_fts,
    )
