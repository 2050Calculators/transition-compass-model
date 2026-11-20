"""
EU27 buildings heating mix preprocessing.

Computes space-heating and domestic-hot-water technology mix by building type,
energy class, and year (OTS).

Hotmaps 2017 provides the relative distribution of fuel shares across building
types and construction periods (energy classes B-F).  JRC-IDEES RES_hh_fec
provides the correct energy-based absolute fuel shares per year (2000-2021),
capturing wood stoves that Hotmaps misses.  The two are combined: Hotmaps
relative shape × JRC absolute level, then renormalized.

Returns
-------
dm_heating_cat : DataMatrix
    Heating technology mix by building type × sh/dhw × technology × energy class (OTS).
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
_CACHE_PATH = os.path.join(_THIS_DIR, "../data/datamatrix/heating_mix_EU.pickle")

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

CONSTRUCTION_PERIOD_ENVELOPE_CAT = {
    "F": ["0-1945", "1946-1969"],
    "E": ["1970-1979"],
    "D": ["1980-1989"],
    "C": ["1990-1999"],
    "B": ["2000-2010", "2011-now"],
}


#################################################################
########################### FUNCTIONS ###########################
#################################################################


def df_excel_to_dm(
    df, names_dict, var_name, unit, num_cat, keep_first=False, country="Switzerland"
):
    if np.nan in df.columns:
        df.drop(columns=np.nan, inplace=True)
    df.rename(columns={df.columns[0]: "Variables"}, inplace=True)
    df.set_index("Variables", inplace=True)
    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df = df[df.apply(lambda row: row.map(pd.api.types.is_number), axis=1).any(axis=1)]
    df_clean = df.loc[
        :, df.apply(lambda col: col.map(pd.api.types.is_number)).any(axis=0)
    ].copy()
    df_filter = df_clean.loc[names_dict.keys()].copy()
    df_filter = df_filter.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    df_filter.reset_index(inplace=True)
    df_filter["Variables"] = df_filter["Variables"].replace(names_dict)
    if keep_first:
        df_filter = df_filter.drop_duplicates(subset=["Variables"], keep="first")
    df_filter = df_filter.groupby(["Variables"]).sum()
    df_filter.reset_index(inplace=True)

    df_filter["Country"] = country
    df_T = pd.melt(
        df_filter,
        id_vars=["Variables", "Country"],
        var_name="Years",
        value_name="values",
    )
    df_pivot = df_T.pivot_table(
        index=["Country", "Years"],
        columns=["Variables"],
        values="values",
        aggfunc="sum",
    )
    df_pivot = df_pivot.add_suffix("[" + unit + "]")
    df_pivot = df_pivot.add_prefix(var_name + "_")
    df_pivot.reset_index(inplace=True)

    df_pivot["Years"] = pd.to_numeric(df_pivot["Years"], errors="coerce")
    df_pivot = df_pivot.dropna(subset=["Years"])

    dm = DataMatrix.create_from_df(df_pivot, num_cat=num_cat)
    return dm


def _get_jrc_fec_shares(df_fec, use):
    """Return (jrc_years, shares[n_years, n_fuels]) from RES_hh_fec.

    Fuel order matches sorted Categories2:
    coal, district-heating, electricity, gas, heat-pump, heating-oil, other-tech, solar, wood
    """
    year_cols = [
        c for c in df_fec.columns if isinstance(c, (int, float)) and 2000 <= c <= 2021
    ]
    n_years = len(year_cols)

    # Row indices in RES_hh_fec (0-based, after pd.read_excel)
    if use == "sh":
        row_single = {
            "coal": 3,
            "heating-oil": 5,
            "gas": 6,
            "wood": 7,
            "district-heating": 9,
            "heat-pump": 10,
            "electricity": 11,
        }
        row_multi = {"other-tech": [4, 8, 12]}  # LPG + Geothermal + Elec.circ.
        row_zero = {"solar"}
    else:  # dhw
        row_single = {
            "coal": 16,
            "heating-oil": 18,
            "gas": 19,
            "wood": 20,
            "district-heating": 22,
            "electricity": 23,
            "solar": 24,
        }
        row_multi = {"other-tech": [17, 21]}  # LPG + Geothermal
        row_zero = {"heat-pump"}

    fuels = [
        "coal",
        "district-heating",
        "electricity",
        "gas",
        "heat-pump",
        "heating-oil",
        "other-tech",
        "solar",
        "wood",
    ]
    arr = np.zeros((n_years, len(fuels)))
    for fi, fuel in enumerate(fuels):
        if fuel in row_zero:
            arr[:, fi] = 0.0
        elif fuel in row_multi:
            arr[:, fi] = sum(
                np.array([float(df_fec.iloc[r][y]) for y in year_cols])
                for r in row_multi[fuel]
            )
        else:
            r = row_single[fuel]
            arr[:, fi] = np.array([float(df_fec.iloc[r][y]) for y in year_cols])

    total = arr.sum(axis=1, keepdims=True)
    shares = arr / np.maximum(total, 1e-10)
    return [int(y) for y in year_cols], shares


def extract_heating_mix(years_ots):
    # get data
    df = pd.read_excel(_DATA_PATH + "HOTMAPS_heating-mix_2017.xlsx")

    # clean
    df["Country"] = "EU27"
    df["Years"] = 2017
    bldtype_old = list(df["bld-type"].unique())
    bldtype_new = [
        "total",
        "single-family-households",
        "multi-family-households",
        "apartment-blocks",
    ]
    for i in range(0, len(bldtype_old)):
        df.loc[df["bld-type"] == bldtype_old[i], "bld-type"] = bldtype_new[i]
    df.loc[df["construction-year"] == "Before 1945", "construction-year"] = "0-1945"
    df.loc[df["construction-year"] == "Post 2010", "construction-year"] = "2011-now"
    df["construction-year"] = [s.replace(" ", "") for s in df["construction-year"]]
    df.loc[df["construction-year"] == "1945-1969", "construction-year"] = "1946-1969"

    # make dm
    indexes = ["Country", "Years"]
    df = pd.melt(df, id_vars=indexes + ["bld-type", "construction-year"])
    df["variable"] = (
        df["bld-type"] + "_" + df["variable"] + "_" + df["construction-year"]
    )
    df = df.pivot(index=indexes, columns="variable", values="value").reset_index()
    columns_to_rename = df.columns[2 : len(df.columns)]
    for c in columns_to_rename:
        df.rename(columns={c: c + "[%]"}, inplace=True)
    dm = DataMatrix.create_from_df(df, num_cat=3)

    # drop space cooling
    dm.drop("Categories1", ["sc"])
    dm.drop("Categories2", ["space-cooling", "no-space-cooling"])

    # aggregate categories within electricity
    dict_agg = {
        "electricity-sub-main": [
            "boiler-condensing",
            "boiler-non-condensing",
            "electric-heating",
            "stove",
        ],
        "electricity-sub-other": ["combined"],
    }
    dm_oth = dm.groupby(dict_agg, "Categories2", "sum")
    dm.drop(
        "Categories2",
        [
            "boiler-condensing",
            "boiler-non-condensing",
            "electric-heating",
            "stove",
            "combined",
        ],
    )
    dm.append(dm_oth, "Categories2")
    dm.sort("Categories2")

    # get district heating
    dm_out = dm.filter({"Categories2": ["district-heating"]})
    dm_notdh = dm.filter({"Categories2": ["central", "individual"]})
    dm_notdh.group_all("Categories2")

    # get carrier-type techs
    dm_temp = dm.filter(
        {
            "Categories2": [
                "biomass",
                "electricity",
                "fossil-fuels-gas",
                "fossil-fuels-liquid",
                "fossil-fuels-solid",
            ]
        }
    )
    dm_temp.normalise("Categories2")
    dm_temp.array = dm_temp.array * dm_notdh.array[:, :, :, :, np.newaxis, :] / 1
    dm_out.append(dm_temp, "Categories2")

    # get electricity techs
    dm_temp = dm.filter(
        {
            "Categories2": [
                "electricity-sub-main",
                "electricity-sub-other",
                "heat-pump",
                "solar-collectors",
            ]
        }
    )
    dm_temp.normalise("Categories2")
    idx = dm_out.idx
    arr_temp = dm_out.array[:, :, :, :, idx["electricity"], :]
    dm_temp.array = dm_temp.array * arr_temp[:, :, :, :, np.newaxis, :] / 1
    dm_out.drop("Categories2", ["electricity"])
    dm_out.append(dm_temp, "Categories2")
    dm_out.rename_col("electricity-sub-main", "electricity", "Categories2")
    dm_out.rename_col("electricity-sub-other", "other", "Categories2")
    dm_out.sort("Categories2")

    # rename
    dm_out.rename_col(
        [
            "biomass",
            "fossil-fuels-gas",
            "fossil-fuels-liquid",
            "fossil-fuels-solid",
            "other",
            "solar-collectors",
        ],
        ["wood", "gas", "heating-oil", "coal", "other-tech", "solar"],
        "Categories2",
    )
    dm_out.sort("Categories2")

    # add other bld types (proxy from total for non-residential)
    dm_temp = dm_out.filter(
        {"Variables": ["multi-family-households", "single-family-households", "total"]}
    )
    type_missing = ["education", "health", "hotels", "offices", "other", "trade"]
    idx = dm_temp.idx
    for t in type_missing:
        dm_temp.add(
            dm_temp.array[:, :, idx["total"], :, :, :],
            dim="Variables",
            col_label=t,
            unit="%",
        )
    dm_temp.drop("Variables", "total")
    dm_temp.sort("Variables")
    dm_out = dm_temp.copy()

    # Calibrate Hotmaps shape to JRC RES_hh_fec energy-based absolute levels.
    # Hotmaps provides the relative distribution across building types and
    # construction periods; JRC fec provides the correct EU27 aggregate per year.
    file_fec = _DATA_PATH + "databases_full/JRC/JRC-IDEES-2021_Residential_EU27.xlsx"
    df_fec = pd.read_excel(file_fec, sheet_name="RES_hh_fec")

    jrc_years, jrc_sh_shares = _get_jrc_fec_shares(df_fec, "sh")  # [22, 9]
    _, jrc_dhw_shares = _get_jrc_fec_shares(df_fec, "dhw")  # [22, 9]

    # dm_out.array shape: (1_country, 1_year=2017, n_vars, n_use, n_fuels, n_cp)
    use_labels = dm_out.col_labels["Categories1"]
    idx_sh = use_labels.index("sh")
    idx_dhw = use_labels.index("dhw")
    n_jrc = len(jrc_years)
    n_fuels = dm_out.array.shape[4]

    # JRC shares broadcast array: (1, n_jrc, 1, n_use, n_fuels, 1)
    jrc_by_use = np.zeros((n_jrc, len(use_labels), n_fuels))
    jrc_by_use[:, idx_sh, :] = jrc_sh_shares
    jrc_by_use[:, idx_dhw, :] = jrc_dhw_shares
    jrc_broadcast = jrc_by_use[np.newaxis, :, np.newaxis, :, :, np.newaxis]

    # Hotmaps relative shape: each cell's value divided by its mean across
    # (Variables, Categories3).  Cells where mean=0 keep relative=1 so they
    # multiply cleanly against the (also-zero) JRC share.
    hm_arr = dm_out.array  # (1, 1, n_vars, n_use, n_fuels, n_cp)
    with np.errstate(divide="ignore", invalid="ignore"):
        hm_mean = np.nanmean(
            hm_arr, axis=(2, 5), keepdims=True
        )  # (1,1,1,n_use,n_fuels,1)
        hm_relative = np.where(hm_mean > 1e-10, hm_arr / hm_mean, 1.0)

    # Calibrated = relative shape × JRC absolute shares → (1, n_jrc, n_vars, n_use, n_fuels, n_cp)
    calibrated = hm_relative * jrc_broadcast
    cal_total = calibrated.sum(axis=4, keepdims=True)
    calibrated /= np.maximum(cal_total, 1e-10)

    # Expand dm_out to cover all JRC years and fill with calibrated values
    for y in jrc_years:
        if y not in dm_out.col_labels["Years"]:
            dm_out.add(np.nan, "Years", [y], dummy=True)
    dm_out.sort("Years")
    idx = dm_out.idx
    for ti, y in enumerate(jrc_years):
        dm_out.array[:, idx[y], ...] = calibrated[:, ti, ...]

    # Fill remaining OTS years outside 2000-2021
    years_missing = list(set(years_ots) - set(dm_out.col_labels["Years"]))
    years_missing_e = [
        yr for yr in years_missing if yr > dm_out.col_labels["Years"][-1]
    ]
    years_missing_i = [yr for yr in years_missing if yr < dm_out.col_labels["Years"][0]]
    linear_fitting(
        dm_out,
        years_missing_e,
        based_on=list(range(2015, dm_out.col_labels["Years"][-1] + 1)),
    )
    dm_out.add(np.nan, dim="Years", col_label=years_missing_i, dummy=True)
    dm_out.sort("Years")
    dm_out.fill_nans("Years")

    # Generate other countries (proxy from EU27)
    idx = dm_out.idx
    for c in EU27_CNTR_LIST:
        dm_out.add(dm_out.array[idx["EU27"], ...], "Country", c)
    dm_out.sort("Country")

    return dm_out


#################################################################
########################## MAIN RUN ############################
#################################################################


def run(years_ots):
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    dm_heating_mix = extract_heating_mix(years_ots)

    dm_heating_cat = dm_heating_mix.groupby(
        CONSTRUCTION_PERIOD_ENVELOPE_CAT, dim="Categories3", aggregation="mean"
    )

    # Zero out class B for years before 2010 (no B buildings existed yet)
    for y in range(1990, 2010 + 1):
        dm_heating_cat[:, y, :, :, :, "B"] = 0

    result = dm_heating_cat
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved cache → {_CACHE_PATH}")
    return result


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_ots

    dm_heating_cat = run(years_ots)
    print("dm_heating_cat shape:", dm_heating_cat.array.shape)
    for dim, vals in dm_heating_cat.col_labels.items():
        if dim == "Years":
            print(f"  {dim}: {vals[0]}..{vals[-1]} ({len(vals)})")
        else:
            print(f"  {dim}: {vals}")
