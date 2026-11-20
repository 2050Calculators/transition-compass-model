"""
EU27 freight OTS: total tkm, modal share, and absolute per-mode tkm (1990-2023).

Accounting basis
----------------
Road HDVL, HDVM, HDVH: **territorial** (JRC TrRoad_act — all trucks on EU27 roads,
  including non-EU transit vehicles). No EU-wide equivalent of the CH BFS GTS
  IMMATRICULATION="CH" filter exists. For EU27 as a large bloc, territorial ≈
  residency (transit trucks a small share of total EU27 road freight).

  HDV split methodology:
  - HDVL (<3.5 t GVW): JRC "Light commercial vehicles" directly — no split needed.
  - HDVM (3.5–20 t MPLW): JRC "Heavy goods vehicles" × Eurostat ROAD_GO_TA_MPLW
    share for (T_LE10 + T10P1-20) / TOTAL — approx. rigid trucks.
  - HDVH (>20 t MPLW): JRC "Heavy goods vehicles" × remaining MPLW share — approx.
    articulated trucks. Eurostat MPLW data available from 2008; pre-2008 years
    backfilled with the 2008 value (share is stable at ~10/90 HDVM/HDVH).

  CH reference: BFS GTS LORRY/HAV split by body type (rigid/articulated); a
  GVW-based revisit of the CH split using the same MPLW methodology is a
  separate pending task.

IWW: territorial (JRC Navigation_Domestic).
Aviation: territorial (JRC Aviation_EU, departures from EU airports).
Maritime: territorial — Intra-EU27 short-sea (JRC Navigation_International, Intra-EU27 row)
  plus domestic coastal (JRC Navigation_Domestic, domestic coastal row). Extra-EU27 deep-sea
  excluded: it captures full voyage distances of ships touching EU27 ports (not territorial).

  Scope rationale: intra-EU27 voyages (e.g. Rotterdam→Barcelona) are treated as
  "domestic" for the EU27 bloc — exactly as a truck driving the same route is
  counted as EU27 road freight. From individual member-state perspectives these
  voyages are "international bunkers" (excluded from national inventories), so
  Eurostat 1.A.3.d "Navigation" (~14 Mt CO2 EU27 2019) is NOT the right benchmark
  here — it sums 27 national inventories, each counting only intra-national voyages.
  The correct benchmark is EMSA/ICCT intra-EU maritime emissions (~60–80 Mt CO2),
  consistent with our ~66 Mt CO2 in 2019.

  Residency assumption: goods moving within EU27 are overwhelmingly demanded by EU27
  residents/businesses, so territorial ≈ residency — same logic as road HDV.
Rail: territorial (JRC TrRail_act).

Freight levers: all 4 FTS ambition levels are identical (BAU linear trend
  continued). FTS is generated in scenarios/transport_fts_EU.py — this
  processor returns OTS only.
  # TODO: differentiate FTS ambition levels for freight demand levers.

Sources
-------
Road HDV:   JRC-IDEES-2021 Transport, sheet TrRoad_act, "Freight transport (mio tkm)"
MPLW split: Eurostat ROAD_GO_TA_MPLW (EU27_2020), MIO_TKM, annual 2008–present
IWW:        JRC-IDEES-2021 Navigation_Domestic, rows 3-4
Aviation:   JRC-IDEES-2021 Aviation_EU, rows 6-7
Maritime:   JRC-IDEES-2021 Navigation_International_EU, Intra-EU27 row (iloc[2:3])
            + JRC-IDEES-2021 Navigation_Domestic, domestic coastal row (iloc[2:3])
Rail:       JRC-IDEES-2021 Transport, sheet TrRail_act, "Freight transport (mio tkm)"
"""

import os
import pickle
import warnings

import eurostat
import numpy as np
import pandas as pd

from transition_compass_model._database.pre_processing.routine_JRC import get_jrc_data
from transition_compass_model.model.common.auxiliary_functions import (
    jrc_iso2_dict,
    linear_fitting,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(_THIS_DIR, "../data/datamatrix/freight_modal_share.pickle")


def _compute_mplw_shares(years_ext: list) -> dict:
    """
    Fetch Eurostat ROAD_GO_TA_MPLW for EU27 and return per-year HDVM/HDVH shares.

    HDVM = (T_LE10 + T10P1-20) / TOTAL  (MPLW <= 20 t, approx. rigid trucks)
    HDVH = (T20P1-30 + T30P1-40 + T_GT40) / TOTAL

    Years before 2008 (Eurostat coverage start) are backfilled with the 2008 value.
    Returns {year: (hdvm_share, hdvh_share)}.
    """
    df = eurostat.get_data_df("ROAD_GO_TA_MPLW")
    df_eu27 = df[
        (df["geo\\TIME_PERIOD"] == "EU27_2020") & (df["unit"] == "MIO_TKM")
    ].copy()
    df_eu27 = df_eu27.drop(columns=["freq", "unit", "geo\\TIME_PERIOD"])
    df_melted = df_eu27.melt(id_vars=["weight"], var_name="year", value_name="mio_tkm")
    df_melted["year"] = pd.to_numeric(df_melted["year"], errors="coerce")
    df_melted = df_melted.dropna(subset=["year", "mio_tkm"])
    df_melted["year"] = df_melted["year"].astype(int)

    piv = df_melted.pivot(index="year", columns="weight", values="mio_tkm")
    piv["hdvm"] = piv["T_LE10"] + piv["T10P1-20"]
    piv["hdvh"] = piv["T20P1-30"] + piv["T30P1-40"] + piv["T_GT40"]
    piv["total"] = piv["hdvm"] + piv["hdvh"]
    piv["hdvm_share"] = piv["hdvm"] / piv["total"]
    piv["hdvh_share"] = piv["hdvh"] / piv["total"]

    share_2008 = (
        float(piv.loc[2008, "hdvm_share"]),
        float(piv.loc[2008, "hdvh_share"]),
    )

    shares = {}
    for y in years_ext:
        if y in piv.index:
            shares[y] = (
                float(piv.loc[y, "hdvm_share"]),
                float(piv.loc[y, "hdvh_share"]),
            )
        else:
            shares[y] = share_2008
    return shares


def run(years_ots: list, years_fts: list):
    """
    Build freight modal share, per-mode tkm, and total tkm DataMatrices for OTS.

    Parameters
    ----------
    years_ots, years_fts : list

    Returns
    -------
    DM_mod : dict  {"ots": {"freight_modal-share": DataMatrix}}
        Modal share % per mode, Cat1=modes, OTS only.
    DM_tkm : dict  {"ots": {"freight_tkm": DataMatrix}}
        Absolute tkm per mode, Cat1=modes, unit=tkm, OTS only.
    DM_tkmtot : dict  {"ots": {"freight_tkm": DataMatrix}}
        Total tkm (all modes summed), unit=bn-tkm, OTS only.
        (Folds logic from old transport_lever_freight_tkm.py.)
    """
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    years_ots_ext = list(range(1989, years_ots[-1] + 1))

    dict_iso2_jrc = jrc_iso2_dict()

    # MPLW shares for HDVM/HDVH split (Eurostat, EU27 aggregate applied to all countries)
    mplw_shares = _compute_mplw_shares(years_ots_ext)

    ###############
    ##### HDV #####
    ###############

    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_act",
        "variable": "Freight transport (mio tkm)",
        "sheet_last_row": "Heavy goods vehicles",
        "sub_variables": ["Light commercial vehicles", "Heavy goods vehicles"],
        "calc_names": ["HDVL", "HDVH"],
    }
    dm_hdvl = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)

    # Split JRC "Heavy goods vehicles" (HDVH) into HDVM and HDVH using MPLW shares.
    # HDVL (LCV, <3.5 t) maps directly to model HDVL — no split needed.
    years_in_dm = dm_hdvl.col_labels["Years"]
    hdvm_sh_arr = np.array([mplw_shares[y][0] for y in years_in_dm])
    hdvh_sh_arr = np.array([mplw_shares[y][1] for y in years_in_dm])
    idx = dm_hdvl.idx
    arr_hgv = dm_hdvl.array[:, :, idx["HDVH"]].copy()  # (n_countries, n_years)
    dm_hdvl.array[:, :, idx["HDVH"]] = arr_hgv * hdvh_sh_arr[np.newaxis, :]
    dm_hdvl.add(arr_hgv * hdvm_sh_arr[np.newaxis, :], "Variables", "HDVM", "vehicles")
    dm_hdvl.sort("Variables")

    ###############
    ##### IWW #####
    ###############

    def get_specific_jrc_data(
        country_code,
        country_name,
        row_start,
        row_end,
        unit,
        variable="aviation_kerosene",
        database="JRC-IDEES-2021_x1990_Aviation_EU",
    ):
        filepath_jrc = os.path.join(
            _THIS_DIR,
            f"../../../industry/eu/data/JRC-IDEES-2021/EU27/{database}.xlsx",
        )
        df_temp = pd.read_excel(filepath_jrc, sheet_name=country_code)
        df_temp = df_temp.iloc[row_start:row_end, :]
        indexes = df_temp.columns[0]
        df_temp = pd.melt(df_temp, id_vars=indexes, var_name="year")
        df_temp.columns = ["Country", "Years", f"{variable}[{unit}]"]
        df_temp["Country"] = country_name
        return df_temp

    country_codes = list(dict_iso2_jrc.keys())
    country_names = list(dict_iso2_jrc.values())

    df_iww = pd.concat(
        [
            get_specific_jrc_data(
                code,
                name,
                3,
                4,
                "mio tkm",
                "IWW",
                "JRC-IDEES-2021_x1990_Navigation_Domestic",
            )
            for code, name in zip(country_codes, country_names)
        ],
        ignore_index=True,
    )
    dm_iww = DataMatrix.create_from_df(df_iww, 0)
    dm_iww.array[dm_iww.array == 0] = np.nan

    ####################
    ##### aviation #####
    ####################

    df_avi = pd.concat(
        [
            get_specific_jrc_data(
                code,
                name,
                6,
                7,
                "mio tkm",
                "aviation",
                "JRC-IDEES-2021_x1990_Aviation_EU",
            )
            for code, name in zip(country_codes, country_names)
        ],
        ignore_index=True,
    )
    dm_avi = DataMatrix.create_from_df(df_avi, 0)

    ####################
    ##### maritime #####
    ####################

    # Intra-EU27 short-sea: maritime between EU27 member-state ports (iloc[2:3]).
    # Excludes Extra-EU27 deep-sea (iloc[3:4]) — full-voyage distances of ships touching
    # EU27 ports but sailing to non-EU destinations are not territorial.
    df_mar = pd.concat(
        [
            get_specific_jrc_data(
                code,
                name,
                2,
                3,
                "mio tkm",
                "marine",
                "JRC-IDEES-2021_x1990_Navigation_International_EU",
            )
            for code, name in zip(country_codes, country_names)
        ],
        ignore_index=True,
    )
    dm_mar = DataMatrix.create_from_df(df_mar, 0)

    # Domestic coastal shipping: add to marine (same territorial scope, separate JRC sheet).
    df_mar_coastal = pd.concat(
        [
            get_specific_jrc_data(
                code,
                name,
                2,
                3,
                "mio tkm",
                "marine",
                "JRC-IDEES-2021_x1990_Navigation_Domestic",
            )
            for code, name in zip(country_codes, country_names)
        ],
        ignore_index=True,
    )
    dm_mar_coastal = DataMatrix.create_from_df(df_mar_coastal, 0)
    dm_mar.array += dm_mar_coastal.array

    ################
    ##### RAIL #####
    ################

    dict_extract = {
        "database": "Transport",
        "sheet": "TrRail_act",
        "variable": "Freight transport (mio tkm)",
        "sheet_last_row": "Freight transport (mio tkm)",
        "sub_variables": ["Freight transport (mio tkm)"],
        "calc_names": ["rail"],
    }
    dm_tkm_rail = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)

    ########################
    ##### PUT TOGETHER #####
    ########################

    dm_tkm = dm_hdvl.copy()
    dm_tkm.add(np.nan, "Years", list(range(1990, 1999 + 1)), "number", True)
    dm_tkm.sort("Years")
    dm_tkm.append(dm_iww, "Variables")
    dm_tkm.append(dm_avi, "Variables")
    dm_tkm.append(dm_mar, "Variables")
    dm_tkm_rail.add(np.nan, "Years", list(range(1990, 1999 + 1)), "number", True)
    dm_tkm.append(dm_tkm_rail, "Variables")
    dm_tkm.sort("Variables")
    dm_tkm.sort("Country")
    dm_tkm.sort("Years")

    ###################
    ##### FIX OTS #####
    ###################

    dm_tkm = linear_fitting(dm_tkm, years_ots_ext)

    ####################################
    ##### MAKE AS FINAL DATAMATRIX #####
    ####################################

    for v in dm_tkm.col_labels["Variables"]:
        dm_tkm.rename_col(v, "tra_freight_modal-share_" + v, "Variables")
    dm_tkm.deepen()

    dm_tkm.change_unit("tra_freight_modal-share", 1e6, "mio tkm", "tkm")

    dm_tkm_pc = dm_tkm.normalise("Categories1", inplace=False, keep_original=False)
    dm_tkm_pc.rename_col(
        "tra_freight_modal-share_share", "tra_freight_modal-share", "Variables"
    )

    ################
    ##### SAVE #####
    ################

    # OTS only — FTS generated in scenarios/transport_fts_EU.py
    dm_tkm_ots = dm_tkm.filter({"Years": years_ots})
    dm_tkm_pc_ots = dm_tkm_pc.filter({"Years": years_ots})

    # Rename variable for tkm output (mirrors old script's rename before saving DM_tkm)
    dm_tkm_ots.rename_col("tra_freight_modal-share", "tra_freight_tkm", "Variables")

    # Total tkm — fold logic from transport_lever_freight_tkm.py
    dm_tkmtot = dm_tkm_ots.copy()
    dm_tkmtot.group_all("Categories1")
    dm_tkmtot.change_unit("tra_freight_tkm", 1e-9, "tkm", "bn-tkm")
    dm_tkmtot.rename_col("tra_freight_tkm", "tra_freight_tkm-total-demand", "Variables")

    DM_mod = {"ots": {"freight_modal-share": dm_tkm_pc_ots}}
    DM_tkm = {"ots": {"freight_tkm": dm_tkm_ots}}
    DM_tkmtot = {"ots": {"freight_tkm": dm_tkmtot}}

    result = (DM_mod, DM_tkm, DM_tkmtot)
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(result, f)
    return result


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_fts, years_ots

    run(years_ots, years_fts)
