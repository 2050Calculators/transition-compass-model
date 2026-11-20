"""
EU27 buildings heating efficiency preprocessing.

Computes space-heating efficiency by technology × energy class (OTS), using
JRC-IDEES EU27 efficiency time series weighted by stock share per class.

Returns
-------
dm_heating_eff_cat : DataMatrix
    Heating efficiency by technology × energy class (OTS).
    Shape: Country × Years × Variables(bld_heating-efficiency) × Categories1(techs) × Categories2(classes).
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
_CACHE_PATH = os.path.join(_THIS_DIR, "../data/datamatrix/heating_efficiency_EU.pickle")

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

ENVELOPE_CAT_NEW = {"C": (1990, 2000), "B": (2001, 2023)}
ENERGY_CLASSES = ["B", "C", "D", "E", "F"]


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


def extract_heating_efficiency(file, sheet_name, years_ots):
    df = pd.read_excel(file, sheet_name=sheet_name)

    # Get space heating
    df_temp = df[0:12].copy()
    names_map = {
        "Ratio of energy service to energy consumption": "remove",
        "Space heating": "total",
        "Solids": "coal",
        "Liquified petroleum gas (LPG)": "other-tech",
        "Diesel oil": "heating-oil",
        "Natural gas": "gas",
        "Biomass": "wood",
        "Geothermal": "other-tech",
        "Distributed heat": "district-heating",
        "Advanced electric heating": "heat-pump",
        "Conventional electric heating": "electricity",
    }
    dm_jrc_sh = df_excel_to_dm(
        df_temp,
        names_map,
        var_name="bld_heating-efficiency_sh",
        unit="%",
        num_cat=1,
        country="EU27",
    )
    dm_jrc_sh.drop(col_label="remove", dim="Categories1")
    dm_jrc_sh.add(0, dim="Categories1", col_label="solar", dummy=True)
    dm_jrc_sh.sort("Categories1")
    dm_jrc_sh.deepen("_", based_on="Variables")
    dm_jrc_sh.switch_categories_order("Categories2", "Categories1")

    # Get water heating
    df_temp = pd.concat([df[0:2], df[15:25]])
    names_map = {
        "Ratio of energy service to energy consumption": "remove",
        "Water heating": "total",
        "Solids": "coal",
        "Liquified petroleum gas (LPG)": "other-tech",
        "Diesel oil": "heating-oil",
        "Natural gas": "gas",
        "Biomass": "wood",
        "Geothermal": "other-tech",
        "Distributed heat": "district-heating",
        "Electricity": "electricity",
        "Solar": "solar",
    }
    dm_jrc_dwh = df_excel_to_dm(
        df_temp,
        names_map,
        var_name="bld_heating-efficiency_dhw",
        unit="%",
        num_cat=1,
        country="EU27",
    )
    dm_jrc_dwh.drop(col_label="remove", dim="Categories1")
    dm_jrc_dwh.add(0, dim="Categories1", col_label="heat-pump", dummy=True)
    dm_jrc_dwh.sort("Categories1")
    dm_jrc_dwh.deepen("_", based_on="Variables")
    dm_jrc_dwh.switch_categories_order("Categories2", "Categories1")

    # Put together
    dm_heating_eff = dm_jrc_sh.copy()
    dm_heating_eff.append(dm_jrc_dwh, "Categories1")
    dm_heating_eff.drop("Categories2", "total")

    # fix other tech
    dm_heating_eff[..., "other-tech"] = 1

    # Do other years
    years_missing = list(set(years_ots) - set(dm_heating_eff.col_labels["Years"]))
    years_missing_e = [
        yr for yr in years_missing if yr > dm_heating_eff.col_labels["Years"][0]
    ]
    years_missing_i = [
        yr for yr in years_missing if yr < dm_heating_eff.col_labels["Years"][0]
    ]
    linear_fitting(
        dm_heating_eff,
        years_missing_e,
        based_on=list(range(2015, dm_heating_eff.col_labels["Years"][-1] + 1)),
    )
    dm_heating_eff.add(np.nan, dim="Years", col_label=years_missing_i, dummy=True)
    dm_heating_eff.sort("Years")
    dm_heating_eff.fill_nans("Years")

    # Generate other countries (proxy from EU27)
    idx = dm_heating_eff.idx
    for c in EU27_CNTR_LIST:
        dm_heating_eff.add(dm_heating_eff.array[idx["EU27"], ...], "Country", c)
    dm_heating_eff.sort("Country")

    return dm_heating_eff


def compute_heating_efficiency_by_archetype(
    dm_heating_eff, dm_stock_cat, envelope_cat_new, categories
):
    arr_w_cat = np.repeat(
        dm_heating_eff.array[..., np.newaxis], repeats=len(categories), axis=-1
    )
    dm_eff_cat = DataMatrix.based_on(
        arr_w_cat,
        format=dm_heating_eff,
        change={"Categories2": categories},
        units=dm_heating_eff.units,
    )

    # Keep only stock split by categories
    dm_stock = dm_stock_cat.group_all("Categories1", inplace=False)
    idx = dm_eff_cat.idx
    idx_s = dm_stock.idx
    for cat in categories:
        if cat not in envelope_cat_new.keys():
            dm_eff_cat.array[:, :, :, :, idx[cat]] = dm_eff_cat.array[
                :, 0, np.newaxis, :, :, idx[cat]
            ]
        else:
            # Compute efficiency as the weighted average of stock efficiency and new efficiency
            start_yr = envelope_cat_new[cat][0]
            end_yr = envelope_cat_new[cat][1]
            for yr in range(start_yr + 1, end_yr + 1):
                eff_s_tm1 = dm_eff_cat.array[:, idx[yr - 1], 0, :, idx[cat]]
                s_tm1 = dm_stock.array[
                    :,
                    idx_s[yr - 1],
                    idx_s["bld_floor-area_stock"],
                    np.newaxis,
                    idx_s[cat],
                ]
                new_t = dm_stock.array[
                    :, idx_s[yr], idx_s["bld_floor-area_new"], np.newaxis, idx_s[cat]
                ]
                eff_n_t = dm_eff_cat.array[:, idx[yr], 0, :, idx[cat]]
                eff_s_t = (eff_n_t * new_t + s_tm1 * eff_s_tm1) / (new_t + s_tm1)
                dm_eff_cat.array[:, idx[yr], 0, :, idx[cat]] = eff_s_t
            # after construction period, fix efficiency to end of construction period
            if (end_yr + 1) in dm_eff_cat.col_labels["Years"]:
                dm_eff_cat.array[:, idx[end_yr] :, 0, :, idx[cat]] = dm_eff_cat.array[
                    :, idx[end_yr], np.newaxis, 0, :, idx[cat]
                ]
            if start_yr > dm_eff_cat.col_labels["Years"][0]:
                dm_eff_cat.array[:, 0 : idx[start_yr], 0, :, idx[cat]] = np.nan

    return dm_eff_cat


#################################################################
########################## MAIN RUN ############################
#################################################################


def run(dm_all_cat, years_ots):
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    file = _DATA_PATH + "databases_full/JRC/JRC-IDEES-2021_Residential_EU27.xlsx"
    sheet_name = "RES_hh_eff"
    dm_heating_eff_all = extract_heating_efficiency(file, sheet_name, years_ots)

    # no solar for space heating, no heat pump for water heating
    dm_heating_eff_all[:, :, "bld_heating-efficiency", "sh", "solar"] = np.nan
    dm_heating_eff_all[:, :, "bld_heating-efficiency", "dhw", "heat-pump"] = np.nan

    dm_heating_eff_sh = dm_heating_eff_all.filter({"Categories1": ["sh"]})
    dm_heating_eff_sh.group_all("Categories1")
    dm_heating_eff_cat = compute_heating_efficiency_by_archetype(
        dm_heating_eff_sh,
        dm_all_cat,
        ENVELOPE_CAT_NEW,
        categories=ENERGY_CLASSES,
    )

    result = (dm_heating_eff_cat, dm_heating_eff_all)
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
    from processors.renovation import run as renovation_run
    from processors.stock_by_cat import run as stock_by_cat_run

    _, dm_stock_tot, dm_new_tot, dm_waste_tot, _, _ = floor_area_run(years_ots)
    _, dm_renov_cat, _ = renovation_run(dm_stock_tot, years_ots)
    dm_all_cat, _, _ = stock_by_cat_run(
        dm_new_tot, dm_waste_tot, dm_stock_tot, dm_renov_cat, years_ots
    )
    dm_heating_eff_cat = run(dm_all_cat, years_ots)
    print("dm_heating_eff_cat shape:", dm_heating_eff_cat.array.shape)
    for dim, vals in dm_heating_eff_cat.col_labels.items():
        if dim == "Years":
            print(f"  {dim}: {vals[0]}..{vals[-1]} ({len(vals)})")
        else:
            print(f"  {dim}: {vals}")
