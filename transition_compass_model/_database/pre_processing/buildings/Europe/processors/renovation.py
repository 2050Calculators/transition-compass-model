"""
EU27 buildings renovation preprocessing.

Computes renovation rates, renovation redistribution by envelope category,
and renovated floor area by building type × envelope category (OTS).

Returns
-------
dm_renovation : DataMatrix
    Renovation rate by building type (OTS). Also used directly in build_pickle.
dm_renov_cat : DataMatrix
    Renovated floor area by building type × envelope category (OTS).
dm_renov_distr : DataMatrix
    Renovation redistribution in/out by envelope category (OTS).
    Also used directly in build_pickle.
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd

from transition_compass_model.model.common.auxiliary_functions import linear_fitting
from transition_compass_model.model.common.data_matrix_class import DataMatrix

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(_THIS_DIR, "../data/")
_CACHE_PATH = os.path.join(_THIS_DIR, "../data/datamatrix/renovation_EU.pickle")

EU27_CNTR_LIST = [
    "Austria",
    "Belgium",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Ireland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "Poland",
    "Portugal",
    "Romania",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
]


#################################################################
########################### FUNCTIONS ###########################
#################################################################


def get_renovation_rate():
    file = "renovation_rates.xlsx"
    data_path = _DATA_PATH
    df_rr_res = pd.read_excel(
        data_path + file, sheet_name="Renovation_rates_residential"
    )
    df_rr_nonres = pd.read_excel(
        data_path + file, sheet_name="Renovation_rates_non_residentia"
    )
    df_rr_res.rename(
        {
            "Energy related: “Light” ": "bld_ren-rate_sha_residential[%]",
            "Energy related: “Medium” ": "bld_ren-rate_med_residential[%]",
            "Energy related: “Deep” ": "bld_ren-rate_dep_residential[%]",
        },
        axis=1,
        inplace=True,
    )
    df_rr_nonres.rename(
        {
            "Energy related: “Light” ": "bld_ren-rate_sha_non-residential[%]",
            "Energy related: “Medium” ": "bld_ren-rate_med_non-residential[%]",
            "Energy related: “Deep” ": "bld_ren-rate_dep_non-residential[%]",
        },
        axis=1,
        inplace=True,
    )

    # Drop useless cols
    df_rr_res.drop(
        ["Energy related: “Total” ", "Energy related: “below Threshold” "],
        axis=1,
        inplace=True,
    )
    df_rr_nonres.drop(
        ["Energy related: “Total” ", "Energy related: “below Threshold” "],
        axis=1,
        inplace=True,
    )

    # Remove space in Country col
    df_rr_res["Country"] = df_rr_res["Country"].str.strip()
    df_rr_nonres["Country"] = df_rr_nonres["Country"].str.strip()

    # These data are the average between 2012 - 2016
    df_rr_res["Years"] = 2014
    df_rr_nonres["Years"] = 2014
    df_rr = pd.merge(df_rr_res, df_rr_nonres, on=["Country", "Years"], how="inner")
    dm_rr = DataMatrix.create_from_df(df_rr, num_cat=2)

    # Drop United Kingdom
    dm_rr.drop("Country", "United Kingdom")

    # Take sum medium and deep
    # from this: https://www.eea.europa.eu/publications/building-renovation-where-circular-economy/modelling-the-renovation-of-buildings/view
    # They mention that medium is mostly insulation, and deep is insulation + heating
    # I will consider the sum of medium and deep for now (even though deep also includes heating)
    dm_rr = dm_rr.filter({"Categories1": ["med", "dep"]}).group_all(
        "Categories1", inplace=False
    )

    # Make EU
    dm_rr_EU27 = dm_rr.groupby(
        {"EU27": EU27_CNTR_LIST}, dim="Country", inplace=False, aggregation="mean"
    )
    dm_rr.append(dm_rr_EU27, "Country")
    dm_rr.sort("Country")
    # numbers are close to Table 1.3 pag 30
    # from this: https://www.eea.europa.eu/publications/building-renovation-where-circular-economy/modelling-the-renovation-of-buildings/view

    # Assign to different bld types
    mapping_bld_type = {
        "education": "non-residential",
        "health": "non-residential",
        "hotels": "non-residential",
        "multi-family-households": "residential",
        "offices": "non-residential",
        "other": "non-residential",
        "single-family-households": "residential",
        "trade": "non-residential",
    }
    idx = dm_rr.idx
    for key in mapping_bld_type.keys():
        arr_temp = dm_rr.array[..., idx[mapping_bld_type[key]]]
        dm_rr.add(arr_temp, "Categories1", key, "%")
    dm_rr.drop("Categories1", "non-residential")
    dm_rr.drop("Categories1", "residential")
    dm_rr.sort("Categories1")
    dm_rr.rename_col("bld_ren-rate", "bld_renovation-rate", "Variables")

    # Add year dimension
    # Assumption: insulation started to gain traction in 2002, when the Energy Performance of Buildings Directive (EPBD) came into force
    years_missing = list(range(1990, 2023 + 1))
    years_missing.remove(2014)
    dm_rr.add(np.nan, "Years", years_missing, dummy=True)
    dm_rr.sort("Years")
    idx = dm_rr.idx
    dm_rr.array[:, 0 : idx[2002], ...] = 0
    years_same = list(range(2015, 2023 + 1))
    for y in years_same:
        dm_rr.array[:, idx[y], ...] = dm_rr.array[:, idx[2014], ...]
    linear_fitting(dm_rr, list(range(2003, 2014 + 1)))

    return dm_rr


def make_ren_maps():
    # According to the Programme Batiments the assenissment is
    # Amélioration de +1 classes CECB 57%
    # Amélioration de +2 classes CECB 15%
    # Amélioration de +3 classes CECB 15%
    # Amélioration de +4 classes CECB 13%

    # classes that get renovated
    ren_map_out = {
        (1990, 2000): {"F": -0.8, "E": -0.2, "D": 0, "C": 0, "B": 0},
        (2001, 2010): {"F": -0.8, "E": -0.2, "D": 0, "C": 0, "B": 0},
        (2011, 2023): {"F": -0.8, "E": -0.2, "D": 0, "C": 0, "B": 0},
    }

    # improvements
    improv_1classes = 0.57
    improv_2classes = 0.15
    improv_3classes = 0.15
    improv_4classes = 0.13

    # create dictionary for new classes
    ren_map_in = {
        (1990, 2000): {"F": 0, "E": 0, "D": 0, "C": 0, "B": 0},
        (2001, 2010): {"F": 0, "E": 0, "D": 0, "C": 0, "B": 0},
        (2011, 2023): {"F": 0, "E": 0, "D": 0, "C": 0, "B": 0},
    }

    # 1990
    f_to_e = (-ren_map_out[(1990, 2000)]["F"]) * improv_1classes  # F -> E
    f_to_d = (-ren_map_out[(1990, 2000)]["F"]) * improv_2classes  # F -> D
    f_to_c = (-ren_map_out[(1990, 2000)]["F"]) * improv_3classes  # F -> C
    e_to_d = (-ren_map_out[(1990, 2000)]["E"]) * improv_1classes  # E -> D
    e_to_c = (-ren_map_out[(1990, 2000)]["E"]) * improv_2classes  # E -> C
    e = f_to_e
    d = f_to_d + e_to_d
    c = f_to_c + e_to_c
    tot = e + d + c
    ren_map_in[(1990, 2000)]["E"] = e / tot
    ren_map_in[(1990, 2000)]["D"] = d / tot
    ren_map_in[(1990, 2000)]["C"] = c / tot

    # 2001
    f_to_e = (-ren_map_out[(2001, 2010)]["F"]) * improv_1classes  # F -> E
    f_to_d = (-ren_map_out[(2001, 2010)]["F"]) * improv_2classes  # F -> D
    f_to_c = (-ren_map_out[(2001, 2010)]["F"]) * improv_3classes  # F -> C
    f_to_b = (-ren_map_out[(2001, 2010)]["F"]) * improv_4classes  # F -> B
    e_to_d = (-ren_map_out[(2001, 2010)]["E"]) * improv_1classes  # E -> D
    e_to_c = (-ren_map_out[(2001, 2010)]["E"]) * improv_2classes  # E -> C
    e_to_b = (-ren_map_out[(2001, 2010)]["E"]) * improv_3classes  # E -> B
    e = f_to_e
    d = f_to_d + e_to_d
    c = f_to_c + e_to_c
    b = f_to_b + e_to_b
    b = 2  # adjustment to get to get steep slope to get to zero before 2000
    tot = e + d + c + b
    ren_map_in[(2001, 2010)]["E"] = e / tot
    ren_map_in[(2001, 2010)]["D"] = d / tot
    ren_map_in[(2001, 2010)]["C"] = c / tot
    ren_map_in[(2001, 2010)]["B"] = b / tot

    # 2011
    ren_map_in[(2011, 2023)] = ren_map_in[(2001, 2010)]

    return ren_map_in, ren_map_out


def extract_renovation_redistribuition(ren_map_in, ren_map_out, years_ots):
    dm = DataMatrix(
        col_labels={
            "Country": EU27_CNTR_LIST + ["EU27"],
            "Years": years_ots,
            "Variables": [
                "bld_renovation-redistribution-in",
                "bld_renovation-redistribution-out",
            ],
            "Categories1": ["B", "C", "D", "E", "F"],
        },
        units={
            "bld_renovation-redistribution-in": "%",
            "bld_renovation-redistribution-out": "%",
        },
    )
    dm.array = np.nan * np.ones(
        (
            len(dm.col_labels["Country"]),
            len(dm.col_labels["Years"]),
            len(dm.col_labels["Variables"]),
            len(dm.col_labels["Categories1"]),
        )
    )
    dm.sort("Country")
    idx = dm.idx
    for year_period, map_values in ren_map_out.items():
        for key, val in map_values.items():
            dm.array[
                :,
                idx[year_period[1]],
                idx["bld_renovation-redistribution-out"],
                idx[key],
            ] = val
    dm.array[:, idx[1990], ...] = dm.array[:, idx[2000], ...]

    idx = dm.idx
    for year_period, map_values in ren_map_in.items():
        idx_year_period = [idx[yr] for yr in range(year_period[0], year_period[1] + 1)]
        for key, val in map_values.items():
            dm.array[
                :, idx_year_period, idx["bld_renovation-redistribution-in"], idx[key]
            ] = val

    dm.fill_nans(dim_to_interp="Years")
    dm.normalise("Categories1")

    return dm


def compute_floor_area_renovated(dm_stock_tot, dm_renovation, dm_renov_distr):
    # r_ct(t) = Redistr_ct(t) * (ren-rate_t(t) * stock_t(t))

    dm = dm_renovation.copy()
    dm.append(dm_stock_tot, dim="Variables")
    dm.lag_variable("bld_floor-area_stock", shift=1, subfix="_tm1")
    dm.lag_variable("bld_renovation-rate", shift=1, subfix="_tm1")
    dm.operation(
        "bld_renovation-rate_tm1",
        "*",
        "bld_floor-area_stock_tm1",
        out_col="bld_floor-area_renovated",
        unit="m2",
    )

    idx = dm.idx
    idx_d = dm_renov_distr.idx
    arr = dm.array[:, :, idx["bld_floor-area_renovated"], :, np.newaxis] * (
        -dm_renov_distr.array[
            :, :, idx_d["bld_renovation-redistribution-out"], np.newaxis, :
        ]
        + dm_renov_distr.array[
            :, :, idx_d["bld_renovation-redistribution-in"], np.newaxis, :
        ]
    )
    dm_renovated_cat = DataMatrix.based_on(
        arr[:, :, np.newaxis, ...],
        dm,
        change={
            "Variables": ["bld_floor-area_renovated"],
            "Categories2": dm_renov_distr.col_labels["Categories1"],
        },
        units={"bld_floor-area_renovated": "m2"},
    )

    return dm_renovated_cat


#################################################################
########################## MAIN RUN ############################
#################################################################


def run(dm_stock_tot, years_ots):
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    ###############################################
    ########### Floor area Renovated ots ##########
    ###############################################

    # Load renovation rates
    dm_renovation = get_renovation_rate()

    # Renovation by envelope cat ots
    ren_map_in, ren_map_out = make_ren_maps()
    dm_renov_distr = extract_renovation_redistribuition(
        ren_map_in, ren_map_out, years_ots
    )

    # Floor-area Renovated by envelope cat
    # r_ct (t) = Ren-disr_ct(t) ren-rate(t) s(t-1)
    dm_renov_cat = compute_floor_area_renovated(
        dm_stock_tot, dm_renovation, dm_renov_distr
    )

    result = (dm_renovation, dm_renov_cat, dm_renov_distr)
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved cache → {_CACHE_PATH}")
    return result


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_ots

    from processors.floor_area import run as floor_area_run

    _, dm_stock_tot, _, _, _, _ = floor_area_run(years_ots)
    dm_renovation, dm_renov_cat, dm_renov_distr = run(dm_stock_tot, years_ots)
    print("dm_renovation variables:", dm_renovation.col_labels["Variables"])
    print("dm_renovation shape:", dm_renovation.array.shape)
    print("dm_renov_cat shape:", dm_renov_cat.array.shape)
    print("dm_renov_distr shape:", dm_renov_distr.array.shape)
