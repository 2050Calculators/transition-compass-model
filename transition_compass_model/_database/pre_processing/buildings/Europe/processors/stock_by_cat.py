"""
EU27 buildings stock-by-category preprocessing.

Computes floor area stock / new / waste disaggregated by envelope energy class,
plus u-value and surface-to-floorarea constant DataMatrices.

Returns
-------
dm_all_cat : DataMatrix
    Floor area stock / new / waste / renovated by building type × energy class (OTS).
dm_u_value : DataMatrix
    U-value by building type × energy class (constant, all-years).
dm_s2f : DataMatrix
    Surface-to-floorarea ratio by building type (constant, all-years).
"""

import os
import pickle
import warnings

import eurostat
import numpy as np
import pandas as pd

from processors.floor_area import get_bso_uvalue_by_class
from transition_compass_model.model.common.auxiliary_functions import (
    cdm_to_dm,
)
from transition_compass_model.model.common.constant_data_matrix_class import (
    ConstantDataMatrix,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(_THIS_DIR, "../data/")
_CACHE_PATH = os.path.join(_THIS_DIR, "../data/datamatrix/stock_by_cat_EU.pickle")


#################################################################
########################### FUNCTIONS ###########################
#################################################################


def compute_floor_area_waste_cat(dm_waste_tot):
    # Assumption: following EPBD on renovation, non-residential buildings in Class G are required
    # to reach at least Class F by 2027 and Class E by 2030. Residential buildings
    # have until 2030 to reach Class F and until 2033 to reach Class E.
    # buildings that are structurally unsound or economically unviable to retrofit
    # may be candidates for demolition. Given that Classes G and F encompass the least
    # energy-efficient buildings, it's plausible that a significant portion of demolitions
    # since 1990 involved structures within these categories. In our case we do not have G,
    # so we assume that 80% of demolished is F (majority), and 20% is E.

    dm = dm_waste_tot.copy()
    variabs = dm.col_labels["Categories1"]
    for v in variabs:
        dm.rename_col(v, v + "_F", "Categories1")
    dm.deepen()
    arr_temp = dm.array * 0.2
    dm.add(arr_temp, dim="Categories2", col_label=["E"])
    idx = dm.idx
    dm.array[..., idx["F"]] = dm.array[..., idx["F"]] * 0.8
    dm.add(0, dummy=True, dim="Categories2", col_label=["B", "C", "D"])
    dm.sort("Categories2")

    return dm


def compute_floor_area_new_cat(dm_new_tot, cat_map):
    env_cat = ["B", "C", "D", "E", "F"]
    arr = np.zeros(np.shape(dm_new_tot.array) + (len(env_cat),))
    dm_new_cat = DataMatrix.based_on(
        arr, dm_new_tot, change={"Categories2": env_cat}, units=dm_new_tot.units
    )

    idx = dm_new_cat.idx
    for cat, period in cat_map.items():
        idx_period = [idx[yr] for yr in range(period[0], period[-1] + 1)]
        dm_new_cat.array[:, idx_period, idx["bld_floor-area_new"], :, idx[cat]] = 1

    idx_t = dm_new_tot.idx
    dm_new_cat.array[:, :, idx["bld_floor-area_new"], :, :] = (
        dm_new_cat.array[:, :, idx["bld_floor-area_new"], :, :]
        * dm_new_tot.array[:, :, idx_t["bld_floor-area_new"], :, np.newaxis]
    )

    return dm_new_cat


def extract_stock_byconstrperiod_2020():
    file = "BSO_floor_area_2020.xlsx"
    data_path = _DATA_PATH
    rows_to_skip = [1, 198, 199]
    df_area = pd.read_excel(
        data_path + file, sheet_name="Export", skiprows=rows_to_skip
    )
    dict_ren = {"Building use": "Construction period", "Unnamed: 1": "Country"}
    df_area.rename(dict_ren, axis=1, inplace=True)
    df_area["Years"] = 2020

    indexes = ["Country", "Years"]
    variabs_surface = [
        "Educational buildings",
        "Health buildings",
        "Hotels and Restaurants",
        "Multi-family buildings",
        "Offices",
        "Other non-residential buildings",
        "Single-family buildings",
        "Trade buildings",
    ]
    variab_yofconstr = ["Construction period"]
    df_area = df_area.loc[:, indexes + variabs_surface + variab_yofconstr]
    variabs_surface_new = [
        "education",
        "health",
        "hotels",
        "multi-family-households",
        "offices",
        "other",
        "single-family-households",
        "trade",
    ]
    for i in range(len(variabs_surface)):
        df_area.rename(
            columns={variabs_surface[i]: variabs_surface_new[i]}, inplace=True
        )
    df_area.rename(columns={"Construction period": "constr_period"}, inplace=True)
    df_area = pd.melt(df_area, id_vars=indexes + ["constr_period"])
    df_area["variable"] = df_area["variable"] + "_" + df_area["constr_period"]
    df_area = df_area.pivot(
        index=indexes, columns="variable", values="value"
    ).reset_index()
    columns_to_rename = df_area.columns[2 : len(df_area.columns)]
    for c in columns_to_rename:
        df_area.rename(columns={c: "bld_floor-area_stock_" + c + "[m2]"}, inplace=True)
    dm_stock_constr_2020 = DataMatrix.create_from_df(df_area, num_cat=2)

    return dm_stock_constr_2020


def fix_negative_stock(dm_all_cat, dm_stock_tot):
    # drop F
    dm_tmp = dm_all_cat.filter({"Variables": ["bld_floor-area_stock"]})
    dm_tmp.drop(col_label="F", dim="Categories2")

    # Get minimum of each time series (axis 1 is year) and change the sign
    shift = -np.min(dm_tmp.array[:, :, 0, :, :], axis=1)

    # Set to zero the negative numbers
    shift = np.maximum(0, shift)

    # Replace 0 with nan
    shift[shift == 0] = np.nan

    # Compute the minimum across axis 2, ignoring NaNs
    min_values = np.nanmin(shift, axis=2, keepdims=True)

    # Replace NaNs with the corresponding minimum values
    shift = np.where(np.isnan(shift), min_values, shift)

    shift[np.isnan(shift)] = 0

    # Sum the shift to dm_tmp
    dm_tmp.array[:, :, 0, :, :] = (
        dm_tmp.array[:, :, 0, :, :] + shift[:, np.newaxis, :, :]
    )

    # Get stock without F
    dm_stock_wo_F = dm_tmp.group_all(dim="Categories2", inplace=False)
    dm_stock_wo_F.rename_col(
        "bld_floor-area_stock", "bld_floor-area_stock_woF", dim="Variables"
    )
    dm_stock_tot.append(dm_stock_wo_F, dim="Variables")

    # Get F by subtracting full stock minus stock without F
    dm_stock_tot.operation(
        "bld_floor-area_stock",
        "-",
        "bld_floor-area_stock_woF",
        out_col="bld_floor-area_stock_F",
        unit="m2",
    )
    dm_stock_F = dm_stock_tot.filter({"Variables": ["bld_floor-area_stock_F"]})
    dm_stock_F.deepen(based_on="Variables")
    dm_tmp.append(dm_stock_F, dim="Categories2")

    # Put this new stock inside dm_all_cat
    dm_all_cat.drop(col_label="bld_floor-area_stock", dim="Variables")
    dm_all_cat.append(dm_tmp, dim="Variables")

    # In dm_stock_tot keep only the stock
    dm_stock_tot.filter({"Variables": ["bld_floor-area_stock"]}, inplace=True)

    return dm_all_cat


def compute_stock_area_by_cat(
    dm_stock_cat, dm_new_cat, dm_renov_cat, dm_waste_cat, dm_stock_tot
):
    # Adapt dm_stock_cat 2020 to be equal in aggregate to dm_stock_tot 2020
    idx1 = dm_stock_cat.idx
    arr_temp1 = dm_stock_cat.array[:, idx1[2020], :, :, :]
    idx2 = dm_stock_tot.idx
    arr_temp2 = dm_stock_tot.array[:, idx2[2020], :, :]
    dm_stock_cat.array[:, idx1[2020], :, :, :] = (
        arr_temp1
        * arr_temp2[..., np.newaxis]
        / np.sum(arr_temp1, axis=3, keepdims=True)
    )

    # s_{c,t}(t-1) &= s_{c,t}(t) - n_{c,t}(t) - r_{c,t}(t) + w_{c,t}(t)
    dm_all_cat = dm_stock_cat.copy()
    dm_all_cat.append(dm_new_cat, dim="Variables")
    dm_all_cat.append(dm_renov_cat, dim="Variables")
    dm_all_cat.append(dm_waste_cat, dim="Variables")
    idx = dm_all_cat.idx

    # From 2020 backwards
    for t in reversed(range(1991, 2020 + 1)):
        dm_all_cat.array[:, idx[t - 1], idx["bld_floor-area_stock"], ...] = (
            dm_all_cat.array[:, idx[t], idx["bld_floor-area_stock"], ...]
            - dm_all_cat.array[:, idx[t], idx["bld_floor-area_new"], ...]
            - dm_all_cat.array[:, idx[t], idx["bld_floor-area_renovated"], ...]
            + dm_all_cat.array[:, idx[t], idx["bld_floor-area_waste"], ...]
        )

    # From 2020 onwards
    for t in range(2021, 2023 + 1):
        dm_all_cat.array[:, idx[t], idx["bld_floor-area_stock"], ...] = (
            dm_all_cat.array[:, idx[t - 1], idx["bld_floor-area_stock"], ...]
            + dm_all_cat.array[:, idx[t], idx["bld_floor-area_new"], ...]
            + dm_all_cat.array[:, idx[t], idx["bld_floor-area_renovated"], ...]
            - dm_all_cat.array[:, idx[t], idx["bld_floor-area_waste"], ...]
        )

    # FIX Negative Stock
    dm_all_cat = fix_negative_stock(dm_all_cat, dm_stock_tot)

    # Put F = 0 when negative for non-residential
    dm = dm_all_cat.filter({"Variables": ["bld_floor-area_stock"]})
    dm.array[dm.array < 0] = 0
    dm_all_cat.drop("Variables", ["bld_floor-area_stock"])
    dm_all_cat.append(dm, "Variables")
    dm_all_cat.sort("Variables")

    idx = dm_all_cat.idx
    for t in reversed(dm_all_cat.col_labels["Years"][1:]):
        dm_all_cat.array[:, idx[t - 1], idx["bld_floor-area_stock"], ...] = (
            dm_all_cat.array[:, idx[t], idx["bld_floor-area_stock"], ...]
            - dm_all_cat.array[:, idx[t], idx["bld_floor-area_new"], ...]
            - dm_all_cat.array[:, idx[t], idx["bld_floor-area_renovated"], ...]
            + dm_all_cat.array[:, idx[t], idx["bld_floor-area_waste"], ...]
        )

    return dm_all_cat


#################################################################
########################## MAIN RUN ############################
#################################################################


def run(dm_new_tot, dm_waste_tot, dm_stock_tot, dm_renov_cat, years_ots):
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    ###############################################
    ########### Waste by envelope cat  ############
    ###############################################

    dm_waste_cat = compute_floor_area_waste_cat(dm_waste_tot)

    ###############################################
    ###########  New by envelope cat   ############
    ###############################################

    # source: https://www.bpie.eu/wp-content/uploads/2017/12/State-of-the-building-stock-briefing_Dic6.pdf
    envelope_cat_new = {"C": (1990, 2000), "B": (2001, 2023)}
    dm_new_cat = compute_floor_area_new_cat(dm_new_tot, envelope_cat_new)

    ###############################################
    ##########  Stock by envelope cat   ###########
    ###############################################

    dm_stock_byconsper_2020 = extract_stock_byconstrperiod_2020()

    # source: https://www.bpie.eu/wp-content/uploads/2017/12/State-of-the-building-stock-briefing_Dic6.pdf
    construction_period_envelope_cat = {
        "F": ["0-1945", "1946-1969"],
        "E": ["1970-1979"],
        "D": ["1980-1989"],
        "C": ["1990-1999"],
        "B": ["2000-2010", "2011-now"],
    }
    dm_stock_cat_2020 = dm_stock_byconsper_2020.groupby(
        construction_period_envelope_cat, dim="Categories2"
    )

    years_new = dm_stock_tot.col_labels["Years"].copy()
    years_new.remove(2020)
    dm_stock_cat = dm_stock_cat_2020.copy()
    dm_stock_cat.add(np.nan, "Years", years_new, dummy=True)
    dm_stock_cat.sort("Years")
    dm_stock_cat.rename_col("Czechia", "Czech Republic", "Country")

    # s_{c,t}(t-1) = s_{c,t}(t) - n_{c,t}(t) - r_{c,t}(t) + w_{c,t}(t)
    dm_all_cat = compute_stock_area_by_cat(
        dm_stock_cat, dm_new_cat, dm_renov_cat, dm_waste_cat, dm_stock_tot
    )

    ###############################################
    #######  U-value (fixed assumption)  ##########
    ###############################################

    # Source: EU Building Stock Observatory (BSO) — per-country, per-construction-period,
    # per-building-type U-values, weighted by element area (Almeida & Ferreira 2017,
    # DOI: 10.1016/j.enbuild.2017.07.050) and averaged across EU27 weighted by BSO floor area.
    # Construction-period → energy-class mapping: BPIE 2017
    # (https://www.bpie.eu/wp-content/uploads/2017/12/State-of-the-building-stock-briefing_Dic6.pdf)
    bso_uvalue = get_bso_uvalue_by_class()
    bld_types = sorted(dm_all_cat.col_labels["Categories1"])
    classes = ["B", "C", "D", "E", "F"]
    cdm_u_value = ConstantDataMatrix(
        col_labels={
            "Variables": ["bld_u-value"],
            "Categories1": bld_types,
            "Categories2": classes,
        },
        units={"bld_u-value": "W/m2K"},
    )
    arr = np.zeros(
        (
            len(cdm_u_value.col_labels["Variables"]),
            len(cdm_u_value.col_labels["Categories1"]),
            len(cdm_u_value.col_labels["Categories2"]),
        )
    )
    cdm_u_value.array = arr
    # Utilisation factor η: fraction of transmission losses the heating system must cover.
    #
    # Residential (η=0.75): IEA/Odyssee-Mure value for continuously heated dwellings.
    # Passive solar and internal gains offset ~25% of envelope losses.
    # https://www.odyssee-mure.eu/publications/policy-brief/buildings-energy-efficiency-trends.html
    #
    # Non-residential: commercial buildings are heated part-time — only during occupied
    # hours. η is derived per building type from Eurostat LFS (LFSA_EWHAN2), EU27 2019:
    #   η[type] = 0.75 × (weekly_hours[NACE] × 52 × 2 / 8760)
    # The ×2 factor accounts for the building being heated from before first arrival to
    # after last departure — approximately twice any individual worker's shift span.
    # NACE mapping: education→P, health→Q, hotels→I, offices→M, trade→G, other→S.
    RES_UTILISATION_FACTOR = 0.75
    RES_TYPES = {"single-family-households", "multi-family-households"}

    _NACE_MAP = {
        "education": "P",
        "health": "Q",
        "hotels": "I",
        "offices": "M",
        "trade": "G",
        "other": "S",
    }
    _df_lfs = eurostat.get_data_df("LFSA_EWHAN2")
    _geo_col = [c for c in _df_lfs.columns if "geo" in c.lower()][0]
    _lfs_mask = (
        (_df_lfs[_geo_col] == "EU27_2020")
        & (_df_lfs["sex"] == "T")
        & (_df_lfs["age"] == "Y15-64")
        & (_df_lfs["worktime"] == "TOTAL")
        & (_df_lfs["wstatus"] == "EMP")
    )
    _lfs = _df_lfs[_lfs_mask].set_index("nace_r2")["2019"]
    _nonres_eta = {
        bld: 0.75 * min(float(_lfs[nace]) * 52 * 2 / 8760, 1.0)
        for bld, nace in _NACE_MAP.items()
    }

    idx = cdm_u_value.idx
    for bld in bld_types:
        for cls in classes:
            u = bso_uvalue[bld][cls]
            eta = RES_UTILISATION_FACTOR if bld in RES_TYPES else _nonres_eta[bld]
            cdm_u_value.array[idx["bld_u-value"], idx[bld], idx[cls]] = u * eta
    dm_u_value = cdm_to_dm(cdm_u_value, dm_stock_cat.col_labels["Country"], ["All"])

    ###############################################
    ####  Surface-to-floorarea (constant)  ########
    ###############################################

    # Source: Almeida & Ferreira 2017 (DOI: 10.1016/j.enbuild.2017.07.050),
    # floor-area-weighted mean across reference buildings from 6 EU countries (SFH)
    # and 7 EU countries (MFH). Previous TABULA EPISCOPE values (SFH=2.9, MFH=1.72)
    # were based on a single German 1-story archetype and overestimate the EU average.
    surface_to_floorarea = {
        "single-family-households": 2.7,
        "multi-family-households": 1.4,
    }
    # Non-residential proxied as apartment blocks (TABULA Table 62: wall/roof/floor/window = 0.22+0.22+0.64+0.22)
    apart_blocks = 0.22 + 0.22 + 0.64 + 0.22
    others = ["education", "health", "hotels", "offices", "other", "trade"]
    for o in others:
        surface_to_floorarea[o] = apart_blocks

    cdm_s2f = ConstantDataMatrix(
        col_labels={
            "Variables": ["bld_surface-to-floorarea"],
            "Categories1": sorted(dm_all_cat.col_labels["Categories1"]),
        }
    )
    arr = np.zeros(
        (len(cdm_s2f.col_labels["Variables"]), len(cdm_s2f.col_labels["Categories1"]))
    )
    cdm_s2f.array = arr
    idx = cdm_s2f.idx
    for cat, val in surface_to_floorarea.items():
        cdm_s2f.array[idx["bld_surface-to-floorarea"], idx[cat]] = val
    cdm_s2f.units["bld_surface-to-floorarea"] = "%"
    dm_s2f = cdm_to_dm(cdm_s2f, dm_stock_cat.col_labels["Country"], ["All"])

    result = (dm_all_cat, dm_u_value, dm_s2f)
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

    _, dm_stock_tot, dm_new_tot, dm_waste_tot, _, _ = floor_area_run(years_ots)
    _, dm_renov_cat, _ = renovation_run(dm_stock_tot, years_ots)
    dm_all_cat, dm_u_value, dm_s2f = run(
        dm_new_tot, dm_waste_tot, dm_stock_tot, dm_renov_cat, years_ots
    )
    print("dm_all_cat shape:", dm_all_cat.array.shape)
    print("dm_all_cat variables:", dm_all_cat.col_labels["Variables"])
    print("dm_all_cat Categories1:", dm_all_cat.col_labels["Categories1"])
    print("dm_all_cat Categories2:", dm_all_cat.col_labels["Categories2"])
    print("dm_u_value shape:", dm_u_value.array.shape)
    print("dm_s2f shape:", dm_s2f.array.shape)
