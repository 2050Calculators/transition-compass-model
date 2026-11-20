import os
import pickle
import warnings

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
_CACHE_PATH = os.path.join(_THIS_DIR, "../data/datamatrix/passenger_occupancy.pickle")


def run(DM_pkm, years_ots: list):
    """
    Build passenger occupancy OTS for EU27.

    Occupancy = pkm / vkm (pkm / skm for aviation). OTS only (1990-2023).
    FTS scenarios are computed separately in scenarios/transport_fts_EU.py.

    Accounting basis
    ----------------
    Ground modes (road/rail): both pkm and vkm are territorial (JRC), so the
    ratio is internally consistent — a real load factor.

    Aviation: occupancy = territorial_pkm / territorial_skm = true load factor
    (naturally ≤ 1). Both pkm (from passenger_aviation_pkm, extra-EU at ×1) and
    seat-km denominator are territorial — consistent. Model chain:
        territorial_pkm / load_factor = territorial_skm → energy → territorial emissions

    Parameters
    ----------
    DM_pkm : dict
        Ground pkm from passenger_modal_share.run(),
        {"ots": {"passenger_pkm": dm}, "fts": {"passenger_pkm": {1..4: dm}}}.

    Returns
    -------
    dm_occ_ots : DataMatrix
        OTS occupancy (pkm/vkm), 1990-2023.
    dm_vkm : DataMatrix
        Vehicle-km OTS (deepened, Variable='tra_passenger_vkm', Cat1=modes),
        1990-2023 — passed to utilization_rate and veh_efficiency_new.
    dm_pkm_avi_terr : DataMatrix
        Territorial aviation pkm OTS (deepened, Variable='tra_passenger_pkm',
        Cat1=['aviation']), 1990-2023 — passed to scenarios for FTS computation.
    """
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    dict_iso2_jrc = jrc_iso2_dict()

    ################################################
    ################### GET DATA ###################
    ################################################

    ########################
    ##### LDV, 2W, BUS #####
    ########################

    dict_extract = {
        "database": "Transport",
        "sheet": "TrRoad_act",
        "variable": "Vehicle-km driven (mio km)",
        "sheet_last_row": "Freight transport",
        "sub_variables": [
            "Powered two-wheelers",
            "Passenger cars",
            "Motor coaches, buses and trolley buses",
        ],
        "calc_names": ["2W", "LDV", "bus"],
    }
    dm_vkm_ltb = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_vkm_ltb.sort("Variables")

    ################
    ##### RAIL #####
    ################

    dict_extract = {
        "database": "Transport",
        "sheet": "TrRail_act",
        "variable": "Vehicle-km (mio km)",
        "sheet_last_row": "High speed passenger trains",
        "sub_variables": [
            "Metro and tram, urban light rail",
            "Conventional passenger trains",
            "High speed passenger trains",
        ],
        "calc_names": ["metrotram", "train-conv", "train-hs"],
    }
    dm_vkm_r = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    mapping_calc = {"rail": ["train-conv", "train-hs"]}
    dm_vkm_r.groupby(
        mapping_calc, dim="Variables", aggregation="sum", regex=False, inplace=True
    )

    ####################
    ##### AVIATION #####
    ####################

    # aviation occupancy = pkm / skm, where skm = flights × seats per plane

    def get_specific_jrc_data(
        country_code, country_name, row_start, row_end, unit, variable="aviation"
    ):
        database = "JRC-IDEES-2021_x1990_Aviation_EU"
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

    df_vkm_avi = pd.concat(
        [
            get_specific_jrc_data(code, name, 22, 23, "mio vkm", "vkm")
            for code, name in zip(country_codes, country_names)
        ],
        ignore_index=True,
    )
    dm_vkm_avi = DataMatrix.create_from_df(df_vkm_avi, 0)

    df_seats_avi = pd.concat(
        [
            get_specific_jrc_data(code, name, 273, 274, "number", "seats per plane")
            for code, name in zip(country_codes, country_names)
        ],
        ignore_index=True,
    )
    dm_seats_avi = DataMatrix.create_from_df(df_seats_avi, 0)

    dm_skm_avi = dm_vkm_avi.copy()
    dm_seats_avi = dm_seats_avi.filter({"Years": dm_skm_avi.col_labels["Years"]})
    dm_skm_avi.append(dm_seats_avi, "Variables")
    dm_skm_avi.operation(
        "seats per plane", "*", "vkm", "Variables", "aviation", "mio skm"
    )
    dm_skm_avi.filter({"Variables": ["aviation"]}, inplace=True)

    ###################################
    ##### TERRITORIAL AVIATION PKM #####
    ###################################

    # Aviation pkm for occupancy: territorial (domestic + intra-EU + extra-EU ×1),
    # consistent with passenger_aviation_pkm.py and territorial seat-km denominator.
    dict_extract = {
        "database": "Transport",
        "sheet": "TrAvia_act",
        "variable": "Passenger transport (mio pkm)",
        "sheet_last_row": "International - Extra-EEAwUK",
        "sub_variables": [
            "Domestic",
            "International - Intra-EEAwUK",
            "International - Extra-EEAwUK",
        ],
        "calc_names": ["domestic", "international-intra", "international-extra"],
    }
    dm_pkm_avi_terr = get_jrc_data(dict_extract, dict_iso2_jrc, _THIS_DIR)
    dm_pkm_avi_terr.groupby(
        {"aviation": ["domestic", "international-intra", "international-extra"]},
        dim="Variables",
        aggregation="sum",
        regex=False,
        inplace=True,
    )
    dm_pkm_avi_terr.change_unit("aviation", 1e6, "mio pkm", "pkm")

    ########################
    ##### PUT TOGETHER #####
    ########################

    # Append aviation skm to road+rail vkm. Aviation EU xlsx covers 1990-2021,
    # road/rail JRC covers 2000-2021 — add 1990-1999 NaN to road/rail first so
    # all three DMs share the same year range before appending.
    dm_vkm = dm_vkm_ltb.copy()
    dm_vkm.append(dm_vkm_r, "Variables")
    dm_vkm.add(
        np.nan, col_label=list(range(years_ots[0], 2000)), dummy=True, dim="Years"
    )
    dm_vkm.sort("Years")
    dm_vkm.append(dm_skm_avi, "Variables")
    dm_vkm.sort("Variables")
    dm_vkm.sort("Country")

    ###################
    ##### FIX OTS #####
    ###################

    years_fitting = list(range(years_ots[0], 2000))

    # 2W: fit on 2000-2014 (peaked before 2015); others: fit on 2000-2018
    dm_2w = dm_vkm.filter({"Variables": ["2W"]})
    dm_vkm.drop("Variables", ["2W"])
    dm_vkm = linear_fitting(dm_vkm, years_fitting, based_on=list(range(2000, 2019)))
    dm_2w = linear_fitting(dm_2w, years_fitting, based_on=list(range(2000, 2015)))
    dm_vkm.append(dm_2w, "Variables")
    dm_vkm.sort("Variables")

    # 2022-2023: add then extrapolate from pre-COVID trend
    dm_vkm.add(np.nan, col_label=[2022, 2023], dummy=True, dim="Years")
    dm_vkm = linear_fitting(dm_vkm, [2022, 2023], based_on=list(range(2000, 2019 + 1)))

    # same for territorial aviation pkm OTS
    dm_pkm_avi_terr.add(
        np.nan, col_label=list(range(years_ots[0], 2000)), dummy=True, dim="Years"
    )
    dm_pkm_avi_terr.sort("Years")
    dm_pkm_avi_terr = linear_fitting(
        dm_pkm_avi_terr, years_fitting, based_on=list(range(2000, 2019 + 1))
    )
    dm_pkm_avi_terr.add(np.nan, col_label=[2022, 2023], dummy=True, dim="Years")
    dm_pkm_avi_terr = linear_fitting(
        dm_pkm_avi_terr, [2022, 2023], based_on=list(range(2000, 2019 + 1))
    )

    ####################################
    ##### MAKE AS FINAL DATAMATRIX #####
    ####################################

    # rename and deepen vkm
    for v in dm_vkm.col_labels["Variables"]:
        dm_vkm.rename_col(v, "tra_passenger_vkm_" + v, "Variables")
    dm_vkm.deepen()
    dm_vkm.change_unit("tra_passenger_vkm", 1e6, "mio km", "vkm")

    # deepen territorial aviation pkm, rename to match ground pkm variable
    dm_pkm_avi_terr.rename_col("aviation", "tra_aviation", "Variables")
    dm_pkm_avi_terr.deepen()
    dm_pkm_avi_terr.rename_col("tra", "tra_passenger_pkm", "Variables")
    dm_pkm_avi_terr.sort("Country")

    # combine ground pkm OTS with territorial aviation pkm
    dm_pkm = DM_pkm["ots"]["passenger_pkm"].copy()
    dm_pkm.append(dm_pkm_avi_terr, "Categories1")
    dm_pkm.sort("Categories1")

    # compute occupancy = territorial_pkm / territorial_vkm
    dm_occ = dm_pkm.copy()
    dm_occ.array = dm_occ.array / dm_vkm.array
    dm_occ.rename_col("tra_passenger_pkm", "tra_passenger_occupancy", "Variables")
    dm_occ.units["tra_passenger_occupancy"] = "pkm/vkm"

    # normalize aviation per country: small JRC inconsistency between TrAvia_act pkm
    # and Aviation EU xlsx seat-km can push load factor slightly above 1 (EU27: ~1.03,
    # hub countries up to ~2 due to transit passengers). Divide by country max to keep ≤ 1.
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

    ###############
    ##### OTS #####
    ###############

    dm_occ_ots = dm_occ.filter({"Years": years_ots})

    ################
    ##### SAVE #####
    ################

    result = (dm_occ_ots, dm_vkm, dm_pkm_avi_terr)
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(result, f)
    return result


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_fts, years_ots
    from processors.passenger_modal_share import run as modal_run

    _, DM_pkm = modal_run(years_ots, years_fts)
    run(DM_pkm, years_ots)
