import os
import pickle

import numpy as np
import pandas as pd

from transition_compass_model._database.pre_processing.params import (
    country_list,
    years_fts,
    years_ots,
)
from transition_compass_model.model.common.auxiliary_functions import (
    linear_fitting,
    moving_average,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

####################################
######        CDD, HDD       #######
####################################

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Population-weighted degree-day table produced by hdd_cdd_meteoswiss_CH.py
# (MeteoSwiss TabsD 1 km grid, BFS STATPOP weights). One value per region, year,
# variable and base temperature Tb.
DD_FILE = os.path.join(CURRENT_DIR, "data/meteoswiss_tabsd/hdd_cdd_annual_CH.csv")

# variable in the csv -> (variable in the DataMatrix, unit)
DD_VARIABLES = {
    "HDD": ("clm_HDD", "daysK"),
    "N_below": ("clm_days-below", "days"),
    "CDD": ("clm_CDD", "daysK"),
    "N_above": ("clm_days-above", "days"),
}
DAY_COUNT_VARIABLES = ["clm_days-below", "clm_days-above"]

# Canton names in the csv that differ from the names in country_list
CANTON_NAME_IN_FILE = {
    "Basel Landschaft": "Basel-Landschaft",
    "Basel Stadt": "Basel-Stadt",
    "Geneva": "Genève",
    "Lucerne": "Luzern",
    "Zurich": "Zürich",
}


def load_degree_day_table(countries):
    """Degree-day table as a DataMatrix: Country x Years x Variables x Categories1,
    where Categories1 are the base temperatures Tb as strings ("-5" ... "30")."""
    df = pd.read_csv(DD_FILE)
    file_to_model = {v: k for k, v in CANTON_NAME_IN_FILE.items()}
    df["region"] = df["region"].replace(file_to_model)
    df["variable"] = df["variable"].map({k: v[0] for k, v in DD_VARIABLES.items()})
    missing = set(countries) - set(df["region"])
    if missing:
        raise ValueError(f"Regions missing in {DD_FILE}: {sorted(missing)}")

    years = sorted(int(y) for y in df["year"].unique())
    tbs = sorted(df["Tb"].unique())
    variables = [v[0] for v in DD_VARIABLES.values()]
    index = pd.MultiIndex.from_product([countries, years, variables, tbs])
    values = df.set_index(["region", "year", "variable", "Tb"])["value"].reindex(index)
    if values.isna().any():
        raise ValueError("Incomplete degree-day table")

    dm = DataMatrix(
        col_labels={
            "Country": list(countries),
            "Years": years,
            "Variables": variables,
            "Categories1": [f"{tb:g}" for tb in tbs],
        },
        units={v[0]: v[1] for v in DD_VARIABLES.values()},
    )
    dm.array = values.to_numpy().reshape(dm.array.shape)
    # Other sectors combine DataMatrices by position, so keep the alphabetical order
    # used by the other pickles (not the order of country_list)
    dm.sort("Country")
    return dm


def extrapolate_trend(dm, years_out):
    """Placeholder forecast until the CH2025 scenarios are processed: ordinary
    least-squares linear trend over all observed years, per country, variable and Tb,
    evaluated at years_out. Degree-days are clipped at 0, day counts to [0, 365]."""
    years_obs = np.array(dm.col_labels["Years"], dtype=float)
    y = np.moveaxis(dm.array, 1, 0)  # years first
    slope, intercept = np.polyfit(years_obs, y.reshape(len(years_obs), -1), 1)
    out = intercept + slope * np.array(years_out, dtype=float)[:, np.newaxis]
    out = np.moveaxis(out.reshape((len(years_out),) + y.shape[1:]), 0, 1)

    dm_out = DataMatrix(
        col_labels={**dm.col_labels, "Years": list(years_out)}, units=dm.units
    )
    dm_out.array = np.maximum(out, 0)
    idx = dm_out.idx
    for var in DAY_COUNT_VARIABLES:
        dm_out.array[:, :, idx[var], ...] = np.minimum(
            dm_out.array[:, :, idx[var], ...], 365
        )
    return dm_out


def legacy_variables(dm_dd):
    """Two-threshold metrics the model used historically, derived from the table:
    HDD 18/15 = HDD(15) + 3 N_below(15),  CDD 21/24 = CDD(24) + 3 N_above(24)."""
    dm = DataMatrix(
        col_labels={
            "Country": dm_dd.col_labels["Country"],
            "Years": dm_dd.col_labels["Years"],
            "Variables": [
                "clm_CDD",
                "clm_HDD",
                "clm_days-above-24",
                "clm_days-below-15",
            ],
        },
        units={
            "clm_CDD": "daysK",
            "clm_HDD": "daysK",
            "clm_days-above-24": "days",
            "clm_days-below-15": "days",
        },
    )
    dm[:, :, "clm_HDD"] = (
        dm_dd[:, :, "clm_HDD", "15"] + 3 * dm_dd[:, :, "clm_days-below", "15"]
    )
    dm[:, :, "clm_days-below-15"] = dm_dd[:, :, "clm_days-below", "15"]
    dm[:, :, "clm_CDD"] = (
        dm_dd[:, :, "clm_CDD", "24"] + 3 * dm_dd[:, :, "clm_days-above", "24"]
    )
    dm[:, :, "clm_days-above-24"] = dm_dd[:, :, "clm_days-above", "24"]
    return dm


def dummy_update_DM_module_baseyear(DM_old):
    DM_new = {"ots": dict(), "fts": dict()}
    # key = 'fts', 'ots'
    for lever in DM_old["ots"]:
        DM_new["ots"][lever] = dict()
        DM_new["fts"][lever] = dict()
        for dm_name in DM_old["ots"][lever]:
            dm_ots = DM_old["ots"][lever][dm_name].copy()
            dm_fts = DM_old["fts"][lever][dm_name][1]
            years_ots_missing = list(set(years_ots) - set(dm_ots.col_labels["Years"]))
            dm_ots.add(np.nan, dummy=True, dim="Years", col_label=years_ots_missing)
            dm_ots.sort("Years")
            first_fts_year = dm_fts.col_labels["Years"][0]
            if first_fts_year in years_ots:
                idx = dm_ots.idx
                dm_ots.array[:, idx[first_fts_year], ...] = dm_fts.array[:, 0, ...]
            linear_fitting(dm_ots, dm_ots.col_labels["Years"])
            DM_new["ots"][lever][dm_name] = dm_ots
            DM_new["fts"][lever][dm_name] = dict()
            for level in range(4):
                level = level + 1
                dm_fts_old = DM_old["fts"][lever][dm_name][level]
                DM_new["fts"][lever][dm_name][level] = dm_fts_old.filter(
                    {"Years": years_fts}
                )

    return DM_new


def filter_country_DM(cntr_list, DM):
    for lever in DM["ots"]:
        for dm_name in DM["ots"][lever]:
            dm_ots = DM["ots"][lever][dm_name]
            dm_ots.filter({"Country": cntr_list}, inplace=True)
            DM["ots"][lever][dm_name] = dm_ots
            for level in range(4):
                level = level + 1
                dm_fts = DM["fts"][lever][dm_name][level]
                dm_fts.filter({"Country": cntr_list}, inplace=True)
                DM["fts"][lever][dm_name][level] = dm_fts

    return DM


def compute_cooling_uptake(dm_CDD_ots, dict_CDD_fts):
    def strictly_increasing(dm):
        idx = dm.idx
        for yr in dm.col_labels["Years"][0:-1]:
            dm.array[:, idx[yr] + 1, ...] = np.maximum(
                dm.array[:, idx[yr] + 1, ...], dm.array[:, idx[yr], ...]
            )
        return dm

    ####   Cooling uptake   ####
    # uptake = 0.815 x (1 − e−0.00225 CDD)
    arr = 0.815 * (1 - np.exp(-0.00225 * dm_CDD_ots[:, :, "clm_CDD"]))
    idx_ots = [dm_CDD_ots.idx[yr] for yr in years_ots]
    arr_ots = arr[:, idx_ots]
    for i in range(10):
        window_size = 5
        data_smooth = moving_average(arr_ots, window_size, axis=1)
        arr_ots[:, 2:-2, ...] = data_smooth
        arr_ots[:, 0, ...] = data_smooth[:, 0]
        arr_ots[:, 1, ...] = data_smooth[:, 0]
        arr_ots[:, -1, ...] = data_smooth[:, -1]
        arr_ots[:, -2, ...] = data_smooth[:, -1]
    arr[:, idx_ots[2:-2]] = arr_ots[:, 2:-2, ...]
    for i in [0, 1, -1, -2]:
        arr[:, idx_ots[i]] = np.nan
    dm_CDD_ots.add(
        arr, dim="Variables", col_label="clm_AC-uptake", dummy=True, unit="%"
    )
    linear_fitting(dm_CDD_ots, dm_CDD_ots.col_labels["Years"])
    dm_uptake_ots = dm_CDD_ots.filter({"Variables": ["clm_AC-uptake"]})
    dm_uptake_ots = strictly_increasing(dm_uptake_ots)

    dict_uptake_fts = dict()
    for lev in range(4):
        lev = lev + 1
        dm_CDD_fts = dict_CDD_fts[lev].copy()
        idx = dm_CDD_fts.idx
        arr = 0.815 * (1 - np.exp(-0.00225 * dm_CDD_fts.array[:, :, idx["clm_CDD"]]))
        arr[:, 0:-1] = np.nan
        dm_CDD_fts.add(arr, dim="Variables", col_label="clm_AC-uptake", unit="%")
        dm_uptake_fts = dm_CDD_fts.filter({"Variables": ["clm_AC-uptake"]})
        dm_uptake_fts.append(dm_uptake_ots, dim="Years")
        dm_uptake_fts.sort("Years")
        dm_uptake_fts.fill_nans("Years")
        dm_uptake_fts = strictly_increasing(dm_uptake_fts)
        dm_uptake_fts.filter({"Years": years_fts}, inplace=True)
        dict_uptake_fts[lev] = dm_uptake_fts

    return dm_uptake_ots, dict_uptake_fts


dm_dd = load_degree_day_table(country_list)
missing_ots = set(years_ots) - set(dm_dd.col_labels["Years"])
if missing_ots:
    raise ValueError(f"Degree-day table has no data for {sorted(missing_ots)}")

dm_dd_ots = dm_dd.filter({"Years": years_ots})
dm_dd_fts = extrapolate_trend(dm_dd, years_fts)

# Two groups under the "temp" lever: the legacy scalars (cdd-hdd) and the full
# degree-day table by base temperature (degree-days)
DM_climate_new = {
    "ots": {
        "temp": {
            "cdd-hdd": legacy_variables(dm_dd_ots),
            "degree-days": dm_dd_ots,
        }
    },
    "fts": {"temp": {"cdd-hdd": {}, "degree-days": {}}},
}
dm_legacy_fts = legacy_variables(dm_dd_fts)
for lev in range(1, 5):
    DM_climate_new["fts"]["temp"]["cdd-hdd"][lev] = dm_legacy_fts.copy()
    DM_climate_new["fts"]["temp"]["degree-days"][lev] = dm_dd_fts.copy()

#########################
####    AC UPTAKE    ####
#########################
dict_fts = DM_climate_new["fts"]["temp"]["cdd-hdd"].copy()
dm_ots = DM_climate_new["ots"]["temp"]["cdd-hdd"].copy()
dm_uptake_ots, dict_uptake_fts = compute_cooling_uptake(dm_ots, dict_fts)
DM_climate_new["ots"]["temp"]["cdd-hdd"].append(dm_uptake_ots, dim="Variables")
for lev in range(1, 5):
    DM_climate_new["fts"]["temp"]["cdd-hdd"][lev].append(
        dict_uptake_fts[lev], dim="Variables"
    )

file = os.path.join(CURRENT_DIR, "../../../data/datamatrix/climate.pickle")
with open(file, "wb") as handle:
    pickle.dump(DM_climate_new, handle, protocol=pickle.HIGHEST_PROTOCOL)
