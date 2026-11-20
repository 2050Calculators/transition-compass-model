"""
EU27 buildings floor area preprocessing.

Computes floor area stock, new, waste and non-residential services floor area
for all EU27 countries plus EU27 aggregate.

Returns
-------
dm_all : DataMatrix
    Floor area stock / new / waste / demolition-rates by building type (OTS).
dm_stock_tot : DataMatrix
    Total floor area stock by building type.
dm_new_tot : DataMatrix
    New floor area by building type.
dm_waste_tot : DataMatrix
    Waste floor area by building type.
dm_services_floor : DataMatrix
    Non-residential floor area by building type × energy class.
dm_pop : DataMatrix
    Population OTS (EU27 countries), used downstream for floor intensity.
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd

from transition_compass_model._database.pre_processing.api_routine_Eurostat import (
    get_data_api_eurostat,
)
from transition_compass_model.model.common.auxiliary_functions import (
    eurostat_iso2_dict,
    linear_fitting,
    moving_average,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(_THIS_DIR, "../data/")
_CACHE_PATH = os.path.join(_THIS_DIR, "../data/datamatrix/floor_area_EU.pickle")

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


def sub_routine_get_uvalue_by_element():
    # Load u-values by element, construction-period, building type
    file = "U_value_Europe.xlsx"
    data_path = _DATA_PATH
    rows_to_skip = [1, 191, 192]
    df_u_wall = pd.read_excel(
        data_path + file, sheet_name="U-value - wall", skiprows=rows_to_skip
    )
    df_u_window = pd.read_excel(
        data_path + file, sheet_name="U-value - window", skiprows=rows_to_skip
    )
    df_u_roof = pd.read_excel(
        data_path + file, sheet_name="U-value - roof", skiprows=rows_to_skip
    )
    df_u_ground = pd.read_excel(
        data_path + file, sheet_name="U-value - ground floor", skiprows=rows_to_skip
    )
    # Rename dictionary
    dict_ren = {"Building use": "Country", "Unnamed: 1": "Construction period"}
    df_u_wall.rename(dict_ren, axis=1, inplace=True)
    df_u_window.rename(dict_ren, axis=1, inplace=True)
    df_u_roof.rename(dict_ren, axis=1, inplace=True)
    df_u_ground.rename(dict_ren, axis=1, inplace=True)
    # Remove useless cols
    drop_cols = ["All uses"]
    df_u_wall.drop(drop_cols, axis=1, inplace=True)
    df_u_window.drop(drop_cols, axis=1, inplace=True)
    df_u_roof.drop(drop_cols, axis=1, inplace=True)
    df_u_ground.drop(drop_cols, axis=1, inplace=True)
    return df_u_wall, df_u_window, df_u_roof, df_u_ground


def compute_weighted_u_value(df_u_wall, df_u_window, df_u_roof, df_u_ground):
    # Load u-values by element, construction-period, building type
    file_weight = "Uvalues_literature.xlsx"
    data_path = _DATA_PATH
    df_u_weight = pd.read_excel(data_path + file_weight, sheet_name="weight_element")
    df_u_weight.set_index("Element", inplace=True)

    # Weight u-values by area of element for single-family-households
    w_sfh = df_u_weight["bld_area_weight_sfh[%]"]
    sfh_col = "Single-family buildings"
    df_u_wall[sfh_col] = w_sfh["Facade"] * df_u_wall[sfh_col]
    df_u_roof[sfh_col] = w_sfh["Roof"] * df_u_roof[sfh_col]
    df_u_window[sfh_col] = w_sfh["Windows"] * df_u_window[sfh_col]
    df_u_ground[sfh_col] = w_sfh["Cellar"] * df_u_ground[sfh_col]

    # Weight u-values by area of element for other buildings
    # We apply the multi-family house area weight to all buildings except sfh
    others_col = [
        col
        for col in df_u_wall.columns
        if col not in ["Country", "Construction period", "Single-family buildings"]
    ]
    w_mfh = df_u_weight["bld_area_weight_mfh[%]"]
    for col in others_col:
        df_u_wall[col] = w_mfh["Facade"] * df_u_wall[col]
        df_u_roof[col] = w_mfh["Roof"] * df_u_roof[col]
        df_u_window[col] = w_mfh["Windows"] * df_u_window[col]
        df_u_ground[col] = w_mfh["Cellar"] * df_u_ground[col]

    # Compute uvalue as weighted average of element u-value
    df_uvalue = df_u_wall
    u_cols = [
        col
        for col in df_u_wall.columns
        if col not in ["Country", "Construction period"]
    ]
    for col in u_cols:
        df_uvalue[col] = (
            df_uvalue[col] + df_u_roof[col] + df_u_window[col] + df_u_ground[col]
        )
    return df_uvalue


def extract_u_value(df_uvalue):
    # To obtain the u-value of new buildings replace the construction period with the middle year
    df_uvalue[["start_y", "end_y"]] = df_uvalue["Construction period"].str.split(
        "-", expand=True
    )
    # Replace now with 2020
    df_uvalue["end_y"] = df_uvalue["end_y"].str.replace("now", "2020")
    df_uvalue[["start_y", "end_y"]] = df_uvalue[["start_y", "end_y"]].astype(int)
    df_uvalue["Years"] = ((df_uvalue["end_y"] + df_uvalue["start_y"]) / 2).astype(int)
    df_uvalue.loc[df_uvalue["end_y"] == 1945, "Years"] = (
        1900  # TODO: check this modif with Paola for CH
    )
    df_uvalue.drop(["start_y", "end_y"], axis=1, inplace=True)

    # df_uvalue
    df_uvalue.drop(["Construction period"], axis=1, inplace=True)
    # add unit
    df_uvalue = df_uvalue.add_suffix("[W/(m2K)]")
    df_uvalue = df_uvalue.add_prefix("bld_uvalue_")
    df_uvalue.rename(
        {
            "bld_uvalue_Country[W/(m2K)]": "Country",
            "bld_uvalue_Years[W/(m2K)]": "Years",
        },
        axis=1,
        inplace=True,
    )

    return df_uvalue


def extract_floor_area_stock():
    file = "BSO_floor_area_2020.xlsx"
    data_path = _DATA_PATH
    rows_to_skip = [1, 198, 199]
    df_area = pd.read_excel(
        data_path + file, sheet_name="Export", skiprows=rows_to_skip
    )
    dict_ren = {"Building use": "Construction period", "Unnamed: 1": "Country"}
    df_area.rename(dict_ren, axis=1, inplace=True)

    # To obtain the u-value of new buildings replace the construction period with the middle year
    df_area[["start_y", "end_y"]] = df_area["Construction period"].str.split(
        "-", expand=True
    )

    # Replace now with 2020
    df_area["end_y"] = df_area["end_y"].str.replace("now", "2020")
    df_area[["start_y", "end_y"]] = df_area[["start_y", "end_y"]].astype(int)
    df_area["Years"] = ((df_area["end_y"] + df_area["start_y"]) / 2).astype(int)
    df_area.loc[df_area["end_y"] == 1945, "Years"] = (
        1900  # TODO: check this modif with Paola for CH
    )
    df_area_new = df_area.loc[df_area["Years"] >= 1990].copy()
    df_area.drop(["start_y", "end_y"], axis=1, inplace=True)

    # Keep only buildings after 1990
    df_area.drop(["Construction period"], axis=1, inplace=True)

    # add unit
    df_area = df_area.add_suffix("[m2]")
    df_area = df_area.add_prefix("bld_floor-area_")
    df_area.rename(
        {"bld_floor-area_Country[m2]": "Country", "bld_floor-area_Years[m2]": "Years"},
        axis=1,
        inplace=True,
    )
    dm_area = DataMatrix.create_from_df(df_area, num_cat=1)

    # Compute the average yearly new floor-area constructed as the floor-area/construction period lenght
    df_area_new["period_length"] = df_area_new["end_y"] - df_area_new["start_y"]
    df_area_new.drop(["start_y", "end_y", "Construction period"], axis=1, inplace=True)
    value_cols = set(df_area_new.columns) - {"Country", "Years", "period_length"}
    for col in value_cols:
        df_area_new[col] = df_area_new[col] / df_area_new["period_length"]
    df_area_new.drop(["period_length"], axis=1, inplace=True)
    # add unit
    df_area_new = df_area_new.add_suffix("[m2]")
    df_area_new = df_area_new.add_prefix("bld_floor-area_new_")
    df_area_new.rename(
        {
            "bld_floor-area_new_Country[m2]": "Country",
            "bld_floor-area_new_Years[m2]": "Years",
        },
        axis=1,
        inplace=True,
    )
    dm_area_new = DataMatrix.create_from_df(df_area_new, num_cat=1)

    return dm_area, dm_area_new


def get_uvalue_new_stock0(years_ots):
    # Gets the u-value for the new buildings, as well as the u-value of the building stock at t=baseyear

    # Load u-values by element, construction-period, building type
    df_u_wall, df_u_window, df_u_roof, df_u_ground = sub_routine_get_uvalue_by_element()

    # Weight u-values of element by area of element
    df_uvalue = compute_weighted_u_value(df_u_wall, df_u_window, df_u_roof, df_u_ground)

    # From df_uvalue keep only new built for construction period > 1990
    df_uvalue = extract_u_value(df_uvalue)
    # TODO: there is no selection here > 1900, see with Paola

    # From df to dm
    dm_uvalue = DataMatrix.create_from_df(df_uvalue, num_cat=1)

    # Extract floor-area of building stock
    dm_area, dm_area_new = extract_floor_area_stock()
    dm_area.drop(dim="Country", col_label="EU27")

    # Get multi-family household value as weighted average of 'Apartment buildings', 'Multi-family buildings'
    dm_uvalue.append(dm_area, dim="Variables")
    dm_uvalue.operation(
        "bld_uvalue", "*", "bld_floor-area", out_col="bld_uxarea", unit="m2"
    )
    dm_uvalue.groupby(
        {"multi-family-households": ["Apartment buildings", "Multi-family buildings"]},
        dim="Categories1",
        inplace=True,
    )
    idx = dm_uvalue.idx
    dm_uvalue.array[:, :, idx["bld_uvalue"], idx["multi-family-households"]] = (
        dm_uvalue.array[:, :, idx["bld_uxarea"], idx["multi-family-households"]]
        / dm_uvalue.array[:, :, idx["bld_floor-area"], idx["multi-family-households"]]
    )

    # Rename using Calculator names:
    cols_in = [
        "Educational buildings",
        "Health buildings",
        "Hotels and Restaurants",
        "Offices",
        "Other non-residential buildings",
        "Trade buildings",
        "Single-family buildings",
    ]
    cols_out = [
        "education",
        "health",
        "hotels",
        "offices",
        "other",
        "trade",
        "single-family-households",
    ]
    dm_uvalue.rename_col(cols_in, cols_out, dim="Categories1")

    # Get right categories for new floor area
    dm_area_new.groupby(
        {"multi-family-households": ["Apartment buildings", "Multi-family buildings"]},
        dim="Categories1",
        inplace=True,
    )
    dm_area_new.rename_col(cols_in, cols_out, dim="Categories1")

    # Compute dm_uvalue for initial stock
    dm_uvalue_stock0 = dm_uvalue.filter({"Years": [1900, 1957, 1974, 1984]})
    dm_uvalue_stock0.groupby({1990: ".*"}, dim="Years", regex=True, inplace=True)
    dm_uvalue_stock0.array[:, :, idx["bld_uvalue"], :] = (
        dm_uvalue_stock0.array[:, :, idx["bld_uxarea"], :]
        / dm_uvalue_stock0.array[:, :, idx["bld_floor-area"], :]
    )
    dm_uvalue_stock0.filter({"Variables": ["bld_uvalue"]}, inplace=True)

    # Extract dm_uvalue new
    dm_uvalue_new = dm_uvalue.filter({"Years": [1994, 2005, 2015]})
    dm_uvalue_new.filter({"Variables": ["bld_uvalue"]}, inplace=True)
    dm_uvalue_new.rename_col("bld_uvalue", "bld_uvalue_new", dim="Variables")

    # Linear fitting for missing years
    idx = dm_uvalue_stock0.idx
    max_start = dm_uvalue_stock0.array[:, 0, idx["bld_uvalue"], np.newaxis, :]
    min_end = np.min(dm_uvalue_new.array) * np.ones(shape=max_start.shape)
    linear_fitting(dm_uvalue_new, years_ots, max_t0=max_start, min_tb=min_end)

    # Compute share of floor area by building type, to determine floor-area stock for non-residential buildings
    dm_area_2020 = dm_uvalue.filter({"Variables": ["bld_floor-area"]}, inplace=False)
    dm_area_2020.groupby({2020: ".*"}, dim="Years", regex=True, inplace=True)

    dm_uvalue_stock0.rename_col("Czechia", "Czech Republic", dim="Country")
    dm_area_2020.rename_col("Czechia", "Czech Republic", dim="Country")
    dm_area_new.rename_col("Czechia", "Czech Republic", dim="Country")
    dm_uvalue_new.rename_col("Czechia", "Czech Republic", dim="Country")

    # Compute EU data for area in 2020
    dm_area_2020_EU27 = dm_area_2020.groupby(
        {"EU27": EU27_CNTR_LIST}, dim="Country", inplace=False
    )
    dm_area_2020.append(dm_area_2020_EU27, dim="Country")

    return dm_uvalue_new, dm_area_2020, dm_uvalue_stock0, dm_area_new


# Source: https://www.bpie.eu/wp-content/uploads/2017/12/State-of-the-building-stock-briefing_Dic6.pdf
_CONSTRUCTION_PERIOD_TO_CLASS = {
    "0-1945": "F",
    "1946-1969": "F",
    "1970-1979": "E",
    "1980-1989": "D",
    "1990-1999": "C",
    "2000-2010": "B",
    "2011-now": "B",
}

_BSO_COL_TO_CAT = {
    "Single-family buildings": "single-family-households",
    "Educational buildings": "education",
    "Health buildings": "health",
    "Hotels and Restaurants": "hotels",
    "Offices": "offices",
    "Other non-residential buildings": "other",
    "Trade buildings": "trade",
}


def get_bso_uvalue_by_class():
    """Return EU27 floor-area-weighted composite U-values per (building type, energy class).

    Returns
    -------
    dict : {bld_type: {energy_class: u_value_W_m2K}}
        Building types: single-family-households, multi-family-households,
        education, health, hotels, offices, other, trade.
        Energy classes: B, C, D, E, F.
    """
    df_u_wall, df_u_window, df_u_roof, df_u_ground = sub_routine_get_uvalue_by_element()
    df_comp = compute_weighted_u_value(df_u_wall, df_u_window, df_u_roof, df_u_ground)
    df_comp["class"] = df_comp["Construction period"].map(_CONSTRUCTION_PERIOD_TO_CLASS)

    bso_cols = [
        c
        for c in df_comp.columns
        if c not in ["Country", "Construction period", "class"]
    ]

    # Load BSO floor area for EU27-average weighting
    rows_to_skip = [1, 198, 199]
    df_area = pd.read_excel(
        _DATA_PATH + "BSO_floor_area_2020.xlsx",
        sheet_name="Export",
        skiprows=rows_to_skip,
    )
    df_area.rename(
        {"Building use": "Construction period", "Unnamed: 1": "Country"},
        axis=1,
        inplace=True,
    )
    df_area["class"] = df_area["Construction period"].map(_CONSTRUCTION_PERIOD_TO_CLASS)

    df_m = df_comp.merge(
        df_area[["Country", "Construction period"] + bso_cols].rename(
            columns={c: c + "_area" for c in bso_cols}
        ),
        on=["Country", "Construction period"],
    )

    classes = ["B", "C", "D", "E", "F"]
    result = {}
    for bso_col, cat in _BSO_COL_TO_CAT.items():
        result[cat] = {}
        for cls in classes:
            sub = df_m[df_m["class"] == cls]
            w_a = sub[bso_col + "_area"].fillna(0)
            u_v = sub[bso_col].fillna(sub[bso_col].mean())
            denom = float(w_a.sum())
            result[cat][cls] = (
                float((u_v * w_a).sum() / denom) if denom > 0 else float(u_v.mean())
            )

    # MFH: floor-area-weighted average of Multi-family buildings + Apartment buildings
    result["multi-family-households"] = {}
    for cls in classes:
        sub = df_m[df_m["class"] == cls]
        a_mfh = sub["Multi-family buildings_area"].fillna(0)
        a_apt = sub["Apartment buildings_area"].fillna(0)
        u_mfh = sub["Multi-family buildings"].fillna(
            sub["Multi-family buildings"].mean()
        )
        u_apt = sub["Apartment buildings"].fillna(sub["Apartment buildings"].mean())
        tot = float(a_mfh.sum() + a_apt.sum())
        result["multi-family-households"][cls] = float(
            ((u_mfh * a_mfh).sum() + (u_apt * a_apt).sum()) / tot
        )

    return result


def get_jrc_residential_heated_floor_area(dm_pop, years_ots):
    """
    Return EU27 total residential heated floor area [m²] per year for all years_ots.

    Source: JRC IDEES 2021 Residential EU27, sheet RES_summary,
    'Total households useful surface area (in thousand sqm)'.
    https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/JRC-IDEES/JRC-IDEES-2021_v1/
    JRC coverage: 2000-2021. Values outside that range are extrapolated using a
    per-capita linear trend (backward from 2000-2009, forward from 2015-2021).

    BSO floor area is total (including unheated garages and basements); JRC reports
    heated floor area only. Hotmaps data confirms services heated ≈ constructed (99.9%),
    so only residential requires this correction.
    """
    import openpyxl

    jrc_file = os.path.join(
        _DATA_PATH,
        "databases_full/JRC/JRC-IDEES-2021_Residential_EU27.xlsx",
    )
    wb = openpyxl.load_workbook(jrc_file, read_only=True, data_only=True)
    ws = wb["RES_summary"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    years_jrc = [c for c in rows[0] if isinstance(c, int)]  # 2000–2021
    vals_jrc = [c for c in rows[6] if isinstance(c, (int, float))]
    jrc_area = {y: v * 1e3 for y, v in zip(years_jrc, vals_jrc)}  # thousand sqm → m²

    # EU27 total population per year (sum all countries in dm_pop)
    eu27_pop = {
        yr: float(dm_pop.array[:, dm_pop.idx[yr], 0].sum())
        for yr in dm_pop.col_labels["Years"]
    }

    # Per-capita floor area for JRC years
    jrc_yrs = np.array(sorted(jrc_area.keys()), dtype=float)
    cap_jrc = np.array([jrc_area[int(y)] / eu27_pop[int(y)] for y in jrc_yrs])

    # Two separate linear fits: backward from 2000-2009, forward from 2015-2021
    slope_back, icept_back = np.polyfit(jrc_yrs[:10], cap_jrc[:10], 1)
    slope_fwd, icept_fwd = np.polyfit(jrc_yrs[-7:], cap_jrc[-7:], 1)

    result = {}
    y_min, y_max = int(jrc_yrs[0]), int(jrc_yrs[-1])
    for yr in years_ots:
        if yr in jrc_area:
            result[yr] = jrc_area[yr]
        elif yr < y_min:
            cap = max(0.0, slope_back * yr + icept_back)
            result[yr] = cap * eu27_pop.get(yr, eu27_pop[y_min])
        else:
            cap = max(0.0, slope_fwd * yr + icept_fwd)
            result[yr] = cap * eu27_pop.get(yr, eu27_pop[y_max])

    return result


def get_rooms_cap_eustat(dict_iso2, years_ots):
    # Extracts the number of rooms per capita for the period available (2003-2023)
    # The data are extrapolated with linear fitting until 1990
    ##### Extract rooms per capita
    filter = {
        "geo\\TIME_PERIOD": list(dict_iso2.keys()),
        "building": ["HOUSE", "FLAT"],
        "tenure": "TOTAL",
    }
    mapping_dim = {
        "Country": "geo\\TIME_PERIOD",
        "Variables": "tenure",
        "Categories1": "building",
    }
    dm_rooms = get_data_api_eurostat(
        "ilc_lvho03", filter, mapping_dim, "rooms/cap", years_ots
    )

    dm_rooms.rename_col(
        ["FLAT", "HOUSE"],
        ["multi-family-households", "single-family-households"],
        dim="Categories1",
    )
    # Compute moving average
    window_size = 3  # Change window size to control the smoothing effect
    data_smooth = moving_average(
        dm_rooms.array, window_size, axis=dm_rooms.dim_labels.index("Years")
    )
    dm_rooms.array[:, 1:-1, ...] = data_smooth
    window_size = 3  # Change window size to control the smoothing effect
    data_smooth = moving_average(
        dm_rooms.array, window_size, axis=dm_rooms.dim_labels.index("Years")
    )
    dm_rooms.array[:, 1:-1, ...] = data_smooth
    window_size = 3  # Change window size to control the smoothing effect
    data_smooth = moving_average(
        dm_rooms.array, window_size, axis=dm_rooms.dim_labels.index("Years")
    )
    dm_rooms.array[:, 1:-1, ...] = data_smooth

    # Fill nans
    linear_fitting(dm_rooms, years_ots, min_t0=0, min_tb=0)

    dm_rooms.rename_col("TOTAL", "lfs_rooms-cap", dim="Variables")

    # dm.fill_nans(dim_to_interp='Years')

    return dm_rooms


def get_pop_by_bld_type(code_eustat, dict_iso2, years_ots):
    # code_eustat = 'ilc_lvho01'
    filter = {
        "deg_urb": ["TOTAL"],
        "geo\\TIME_PERIOD": dict_iso2.keys(),
        "rskpovth": ["TOTAL"],
        "building": ["FLAT", "HOUSE"],
        "freq": "A",
    }
    mapping_dim = {
        "Country": "geo\\TIME_PERIOD",
        "Variables": "freq",
        "Categories1": "building",
    }
    dm_pop_share = get_data_api_eurostat(
        "ilc_lvho01", filter, mapping_dim, unit="%", years=years_ots
    )
    dm_pop_share.rename_col(
        ["FLAT", "HOUSE"],
        ["multi-family-households", "single-family-households"],
        dim="Categories1",
    )
    dm_pop_share.rename_col("A", "lfs_pop-by-bld-type_share", dim="Variables")
    dm_pop_share.drop("Years", 2003)

    for i in range(2):
        window_size = 3  # Change window size to control the smoothing effect
        data_smooth = moving_average(
            dm_pop_share.array, window_size, axis=dm_pop_share.dim_labels.index("Years")
        )
        dm_pop_share.array[:, 1:-1, ...] = data_smooth

    linear_fitting(dm_pop_share, years_ots)
    dm_pop_share.normalise(dim="Categories1")

    return dm_pop_share


def estimate_stock_res_from_average_room_size(
    dm_rooms, dm_area_2020, dm_pop, dm_pop_bld_type
):
    # 1) lfs_rooms-cap [rooms/cap] x lfs_pop-by-bld-type [habitants] = lfs_rooms [rooms] (by bld type)
    # 2) bld_floor-area_2020 [m2] / lfs_rooms [rooms] (t=2020) = lfs_avg-room-size [m2/rooms] (t=2020)
    # 3) bld_floor-area_stock_tmp [m2] = lfs_rooms [rooms] x lfs_avg-room-size [m2/rooms]
    # (by bld type, value applied to all years)

    # lfs_pop-by-bld-type = pop_bld_type_share x dm_pop
    dm_pop_bld_type = dm_pop_bld_type.flatten()
    # dm_pop_bld_type.drop(dim='Years', col_label=[2023])
    vars = dm_pop_bld_type.col_labels["Variables"]
    dm_pop_bld_type.append(dm_pop, dim="Variables")
    for var in vars:
        var_out = str.replace(var, "_share", "")
        dm_pop_bld_type.operation(
            var, "*", "lfs_population_total", out_col=var_out, unit="inhabitants"
        )
    dm_pop_bld_type.drop(dim="Variables", col_label="lfs_population_total")
    dm_pop_bld_type.deepen()

    dm_all = dm_pop_bld_type.filter({"Variables": ["lfs_pop-by-bld-type"]})
    dm_all.append(dm_rooms, dim="Variables")
    # rooms/cap * pop = rooms (by bld type)
    dm_all.operation(
        "lfs_pop-by-bld-type", "*", "lfs_rooms-cap", out_col="lfs_rooms", unit="m2"
    )

    # Put all 2020 data together
    dm_all_2020 = dm_area_2020.filter_w_regex({"Categories1": ".*households"})
    dm_all_2020.filter({"Variables": ["bld_floor-area"]}, inplace=True)
    dm_all_2020.append(dm_all.filter({"Years": [2020]}, inplace=False), dim="Variables")

    # floor-araa_2020/rooms_2020 = room_size_2020 [m2/room]
    dm_all_2020.operation(
        "bld_floor-area", "/", "lfs_rooms", out_col="lfs_room-size", unit="m2/room"
    )

    # bld_floor-area_stock_tmp [m2] = lfs_rooms [rooms] x lfs_room-size [m2/rooms]
    idx = dm_all.idx
    idx_2 = dm_all_2020.idx
    arr_stock = (
        dm_all.array[:, :, idx["lfs_rooms"], :]
        * dm_all_2020.array[:, 0, np.newaxis, idx_2["lfs_room-size"], :]
    )
    dm_all.add(
        arr_stock, dim="Variables", col_label="bld_floor-area_stock_tmp", unit="m2"
    )

    # Filter bld_floor-area_stock_tmp
    dm_all.filter({"Variables": ["bld_floor-area_stock_tmp"]}, inplace=True)

    return dm_all


def estimate_stock_non_res(dm_area_2020, dm_stock_res):
    dm_area_2020.sort("Country")
    dm_stock_res.sort("Country")
    dm_area_2020.sort("Categories1")
    dm_stock_res.sort("Categories1")
    dm_area_2020.normalise("Categories1", inplace=True, keep_original=True)

    dm_2020 = dm_area_2020.copy()
    dm_2020.groupby(
        {"residential": ".*households"}, dim="Categories1", regex=True, inplace=True
    )
    dm_stock = dm_stock_res.filter({"Variables": ["bld_floor-area_stock_tmp"]})
    dm_stock.groupby(
        {"residential": ".*households"}, dim="Categories1", regex=True, inplace=True
    )
    # Compute floor-stock share
    cats = ["education", "health", "hotels", "offices", "other", "trade"]
    dm_stock.add(0, dummy=True, dim="Categories1", col_label=cats)
    idx = dm_2020.idx
    idx_a = dm_stock.idx
    # stock_tot (t) = stock_res(t) / share_res(t)
    # stock_non-res (t) = stock_tot(t)*share_non-res(t)
    # for share res and non-res we use the 2020 values
    for year in dm_stock.col_labels["Years"]:
        arr_tot_area_y = (
            dm_stock.array[
                :, idx_a[year], idx_a["bld_floor-area_stock_tmp"], idx_a["residential"]
            ]
            / dm_2020.array[
                :, idx[2020], idx["bld_floor-area_share"], idx["residential"]
            ]
        )
        for cat in cats:
            dm_stock.array[
                :, idx_a[year], idx_a["bld_floor-area_stock_tmp"], idx_a[cat]
            ] = (
                arr_tot_area_y
                * dm_2020.array[:, idx[2020], idx["bld_floor-area_share"], idx[cat]]
            )

    dm_stock.drop(dim="Categories1", col_label="residential")
    dm_stock.append(
        dm_stock_res.filter_w_regex(
            {"Variables": "bld_floor-area_stock_tmp", "Categories1": ".*households"}
        ),
        dim="Categories1",
    )

    return dm_stock


def get_new_building():
    file = "Floor-area_new_built.xlsx"
    data_path = _DATA_PATH
    rows_to_skip = [0, 506, 507]
    df_new = pd.read_excel(data_path + file, sheet_name="Export", skiprows=rows_to_skip)
    df_new.rename(
        {
            "Year": "Years",
            "Value": "bld_floor-area_new_residential[m2]",
            "Value.1": "bld_floor-area_new_non-residential[m2]",
        },
        axis=1,
        inplace=True,
    )
    dm_new = DataMatrix.create_from_df(df_new, num_cat=1)
    dm_new.rename_col("Czechia", "Czech Republic", dim="Country")
    return dm_new


def estimate_floor_area(dm_new_group, dm_new_type, years_ots):
    dm_new_type.drop("Years", 1994)
    # Smooth new construction permits
    window_size = 3  # Change window size to control the smoothing effect
    data_smooth = moving_average(
        dm_new_group.array, window_size, axis=dm_new_group.dim_labels.index("Years")
    )
    dm_new_group.array[:, 1:-1, ...] = data_smooth
    window_size = 5  # Change window size to control the smoothing effect
    data_smooth = moving_average(
        dm_new_group.array, window_size, axis=dm_new_group.dim_labels.index("Years")
    )
    dm_new_group.array[:, 2:-2, ...] = data_smooth

    # Compute share for sfh and mfh
    dm_new_res_type = dm_new_type.filter(
        {"Categories1": ["single-family-households", "multi-family-households"]},
        inplace=False,
    )
    arr_res_share = dm_new_res_type.array / np.nansum(
        dm_new_res_type.array, axis=-1, keepdims=True
    )
    dm_new_res_type.add(
        arr_res_share, dim="Variables", col_label="bld_floor-area_share", unit="%"
    )
    # Compute shares for non residential (commercial)
    dm_new_comm_type = dm_new_type.filter(
        {"Categories1": ["education", "health", "hotels", "offices", "other", "trade"]},
        inplace=False,
    )
    arr_comm_share = dm_new_comm_type.array / np.nansum(
        dm_new_comm_type.array, axis=-1, keepdims=True
    )
    dm_new_comm_type.add(
        arr_comm_share, dim="Variables", col_label="bld_floor-area_share", unit="%"
    )
    # Extrapolate to all the years available in dm_new_group
    years_tmp = dm_new_group.col_labels["Years"]
    linear_fitting(dm_new_comm_type, years_tmp)
    linear_fitting(dm_new_res_type, years_tmp)
    # Multiply new floor-area group by the shares
    idx_g = dm_new_group.idx
    idx_c = dm_new_comm_type.idx
    dm_new_comm_type.array[:, :, idx_c["bld_floor-area_new"], :] = (
        dm_new_comm_type.array[:, :, idx_c["bld_floor-area_share"], :]
        * dm_new_group.array[
            :, :, idx_g["bld_floor-area_new"], idx_g["non-residential"], np.newaxis
        ]
    )
    idx_r = dm_new_res_type.idx
    dm_new_res_type.array[:, :, idx_r["bld_floor-area_new"], :] = (
        dm_new_res_type.array[:, :, idx_r["bld_floor-area_share"], :]
        * dm_new_group.array[
            :, :, idx_g["bld_floor-area_new"], idx_g["residential"], np.newaxis
        ]
    )
    # Join residential and commercial new floor area and apply linear extrapolation
    dm_new_res_type.append(dm_new_comm_type, dim="Categories1")
    dm_new_res_type.drop(dim="Variables", col_label="bld_floor-area_share")
    linear_fitting(dm_new_res_type, years_ots, min_t0=0, min_tb=0)
    window_size = 3
    data_smooth = moving_average(
        dm_new_res_type.array,
        window_size,
        axis=dm_new_res_type.dim_labels.index("Years"),
    )
    dm_new_res_type.array[:, 1:-1, ...] = data_smooth
    data_smooth = moving_average(
        dm_new_res_type.array,
        window_size,
        axis=dm_new_res_type.dim_labels.index("Years"),
    )
    dm_new_res_type.array[:, 1:-1, ...] = data_smooth
    # Compute EU data for area in 2020
    dm_new_res_type_EU27 = dm_new_res_type.groupby(
        {"EU27": EU27_CNTR_LIST}, dim="Country", inplace=False
    )
    dm_new_res_type.drop("Country", "EU27")
    dm_new_res_type.append(dm_new_res_type_EU27, dim="Country")
    return dm_new_res_type


def estimate_waste_fix_stock(dm_area_stock_tmp, dm_area_new, years_ots):
    # Filter & join residential households data
    # dm_area_new.drop('Years', 2023)
    dm_area = dm_area_new.copy()
    dm_area.append(dm_area_stock_tmp, dim="Variables")

    # Demolition rate
    dm_area_out = dm_area.copy()

    # Set the demolition rate at 0.2% by default
    dm_area_out.add(
        0.002, dummy=True, dim="Variables", col_label="bld_demolition-rates", unit="m2"
    )
    idx = dm_area_out.idx
    x = np.array(dm_area.col_labels["Years"])
    # breakpoints = x[::5]
    # Improved initial guess for demolition-rate
    for cntr in dm_area.col_labels["Country"]:
        for cat in dm_area.col_labels["Categories1"]:
            y = dm_area.array[idx[cntr], :, idx["bld_floor-area_stock_tmp"], idx[cat]]
            ds, q = np.polyfit(x, y, 1)  # 1 is for a linear fit (degree 1)
            s0 = np.polyfit(x, y, 0)
            # ds = stock(t) - stock(t-1) = new(t) - waste(t)
            n = dm_area.array[idx[cntr], :, idx["bld_floor-area_new"], idx[cat]]
            n0 = np.polyfit(x, n, 0)
            if n0 > ds:
                # new = n0
                # waste = n0 - ds
                dm_area_out.array[
                    idx[cntr], :, idx["bld_demolition-rates"], idx[cat]
                ] = (n0 - ds) / s0
            # else:
            #    dm_area_out.array[idx[cntr], :, idx['bld_floor-area_new'], idx[cat]] = n/n0*ds

    # dm_area_out.filter({'Variables': ['bld_floor-area_dem-rates', 'bld_floor-area_new', 'bld_floor-area_stock']}, inplace=True)

    # Cap demolition rates to avoid formula-driven outliers (housing-boom noise in
    # countries like Ireland, Spain, Malta pushes residential above 3–5%; non-res
    # formula can reach 100 %+ for small-country/building-type cells).
    # Caps: residential ≤ 1.0 %, non-res ≤ 2.0 % — consistent with the upper end
    # of the EU range in Sandberg et al. (2016) "Dynamic Building Stock Modelling:
    # Application to 11 European Countries", Energy and Buildings 132, 26–38.
    # https://doi.org/10.1016/j.enbuild.2016.05.100
    # Country-specific overrides from that paper (0.2–0.9 % residential) can be
    # reintroduced here if per-country calibration is needed in the future.
    res_cats_set = {"single-family-households", "multi-family-households"}
    for cat in dm_area_out.col_labels["Categories1"]:
        cap = 0.01 if cat in res_cats_set else 0.02
        cat_idx = idx[cat]
        dm_area_out.array[:, :, idx["bld_demolition-rates"], cat_idx] = np.minimum(
            dm_area_out.array[:, :, idx["bld_demolition-rates"], cat_idx], cap
        )

    dm_area_out.rename_col(
        "bld_floor-area_stock_tmp", "bld_floor-area_stock", dim="Variables"
    )
    idx = dm_area_out.idx
    for c in dm_area_out.col_labels["Categories1"]:
        for ti in dm_area_out.col_labels["Years"][-1:0:-1]:
            stock_t = dm_area_out.array[:, idx[ti], idx["bld_floor-area_stock"], idx[c]]
            new_t = dm_area_out.array[:, idx[ti], idx["bld_floor-area_new"], idx[c]]
            dem_rate_t = dm_area_out.array[
                :, idx[ti], idx["bld_demolition-rates"], idx[c]
            ]
            stock_tm1 = (stock_t - new_t) / (1 - dem_rate_t)
            dm_area_out.array[:, idx[ti - 1], idx["bld_floor-area_stock"], idx[c]] = (
                stock_tm1
            )

    # Make stock in t minus 1
    dm_area_out.lag_variable("bld_floor-area_stock", shift=1, subfix="_tm1")

    # make waste
    # TODO: For the moment I do not aggregate and I keep EU27 as unit
    # to be decided if to do the aggregation
    # stock(t) - stock(t - 1) = new(t) - waste(t)
    idx = dm_area_out.idx
    waste_t = dm_area_out.array[:, :, idx["bld_floor-area_new"], :] - (
        dm_area_out.array[:, :, idx["bld_floor-area_stock"], :]
        - dm_area_out.array[:, :, idx["bld_floor-area_stock_tm1"], :]
    )
    dm_area_out.add(waste_t, "Variables", "bld_floor-area_waste", unit="m2", dummy=True)
    dm_area_out.drop(dim="Variables", col_label=["bld_floor-area_stock_tm1"])

    # fix waste for 1990
    idx = dm_area_out.idx
    dm_area_out.array[:, idx[1990], idx["bld_floor-area_waste"], :] = np.nan
    dm_area_out = linear_fitting(
        dm_area_out, years_ots, based_on=list(range(1990, 1995 + 1, 1))
    )

    return dm_area_out


def extract_nonres_class_shares_BSO():
    """Return P(class | type, country) shares from BSO 2020 floor area data.

    Returns a dict {country: DataFrame(index=energy_class, columns=bld_type)}
    where values sum to 1 per column (i.e., sum over classes = 1 for each type).
    """
    file = "BSO_floor_area_2020.xlsx"
    data_path = _DATA_PATH
    df = pd.read_excel(data_path + file, sheet_name="Export", skiprows=[1, 198, 199])
    df.rename(
        {"Building use": "Construction period", "Unnamed: 1": "Country"},
        axis=1,
        inplace=True,
    )
    df["Country"] = df["Country"].replace("Czechia", "Czech Republic")

    non_res_cols = {
        "Educational buildings": "education",
        "Health buildings": "health",
        "Hotels and Restaurants": "hotels",
        "Offices": "offices",
        "Other non-residential buildings": "other",
        "Trade buildings": "trade",
    }
    period_to_class = {
        "0-1945": "F",
        "1946-1969": "E",
        "1970-1979": "D",
        "1980-1989": "D",
        "1990-1999": "C",
        "2000-2010": "C",
        "2011-now": "B",
    }

    df_nr = df[["Country", "Construction period"] + list(non_res_cols.keys())].copy()
    df_nr.rename(columns=non_res_cols, inplace=True)
    df_nr["energy_class"] = df_nr["Construction period"].map(period_to_class)
    df_nr = df_nr.dropna(subset=["energy_class"])

    bld_types = list(non_res_cols.values())
    df_grouped = df_nr.groupby(["Country", "energy_class"])[bld_types].sum()

    shares = {}
    for country in df_grouped.index.get_level_values("Country").unique():
        area_by_class = df_grouped.loc[country]
        type_totals = area_by_class.sum(axis=0)
        # Avoid division by zero: if a type has zero area, assign uniform class distribution
        type_totals = type_totals.replace(0, np.nan)
        country_shares = area_by_class.div(type_totals, axis=1).fillna(
            1.0 / len(area_by_class)
        )
        # Ensure all 5 classes are present
        for cls in ["B", "C", "D", "E", "F"]:
            if cls not in country_shares.index:
                country_shares.loc[cls] = 0.0
        shares[country] = country_shares.loc[["B", "C", "D", "E", "F"]]

    # For EU27 aggregate (if needed), use the EU27 row from BSO
    return shares


#################################################################
########################## MAIN RUN ############################
#################################################################


def run(years_ots):
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    dict_iso2 = eurostat_iso2_dict()
    dict_iso2.pop("CH")  # Remove Switzerland

    ##########################################
    ########## Floor area Stock ots ##########
    ##########################################

    # Load U-value for new buildings()
    dm_uvalue_new, dm_area_2020, dm_uvalue_stock0, dm_new_2 = get_uvalue_new_stock0(
        years_ots
    )

    # Compute total floor area from rooms/cap and average room size in 2020
    dm_rooms = get_rooms_cap_eustat(dict_iso2, years_ots=years_ots)

    # Share of population by building type
    dm_pop_bld_type = get_pop_by_bld_type("ilc_lvho01", dict_iso2, years_ots)

    # Load population data
    filepath = os.path.join(
        _THIS_DIR, "../../../lifestyles/Europe/data/lifestyles_allcountries.pickle"
    )
    with open(filepath, "rb") as handle:
        DM_lfs = pickle.load(handle)
    dm_pop = DM_lfs["ots"]["pop"]["lfs_population_"].copy()
    dm_pop = dm_pop.filter({"Country": list(dict_iso2.values())})
    dm_pop.sort("Years")
    del DM_lfs

    # Drop UK
    dm_rooms.drop("Country", ["United Kingdom"])
    dm_pop.drop("Country", ["United Kingdom"])
    dm_pop_bld_type.drop("Country", ["United Kingdom"])

    # Compute floor-area stock (tmp) from avg room size and nb of rooms (sfh, mfh)
    dm_residential_stock_tmp = estimate_stock_res_from_average_room_size(
        dm_rooms, dm_area_2020, dm_pop, dm_pop_bld_type
    )

    # Anchor EU27 residential total to JRC IDEES heated floor area.
    # BSO reports total floor area (including unheated spaces); JRC provides heated
    # area only. Hotmaps confirms services heated ≈ constructed (99.9%), so only
    # residential requires this correction. Scaling is applied uniformly to all
    # countries (preserving per-country and SFH/MFH shares within each country).
    # dm_area_2020 residential is also scaled so that estimate_stock_non_res()
    # derives non-res from the correct (heated) residential denominator,
    # leaving non-res floor area unchanged (BSO non-res ≈ heated per Hotmaps).
    jrc_res = get_jrc_residential_heated_floor_area(dm_pop, years_ots)
    idx_r = dm_residential_stock_tmp.idx
    # Exclude EU27 aggregate row to avoid double-counting individual countries
    eu27_country_rows = np.array(
        [
            idx_r[c]
            for c in dm_residential_stock_tmp.col_labels["Country"]
            if c != "EU27"
        ]
    )
    arr_model_eu27 = np.nansum(
        dm_residential_stock_tmp.array[
            eu27_country_rows, :, idx_r["bld_floor-area_stock_tmp"], :
        ],
        axis=(0, 2),
    )
    arr_jrc = np.array([jrc_res[yr] for yr in years_ots])
    scale = arr_jrc / np.maximum(arr_model_eu27, 1.0)
    dm_residential_stock_tmp.array[:, :, idx_r["bld_floor-area_stock_tmp"], :] *= scale[
        np.newaxis, :, np.newaxis
    ]
    scale_2020 = float(scale[years_ots.index(2020)])
    idx_a = dm_area_2020.idx
    dm_area_2020.array[
        :, :, idx_a["bld_floor-area"], idx_a["single-family-households"]
    ] *= scale_2020
    dm_area_2020.array[
        :, :, idx_a["bld_floor-area"], idx_a["multi-family-households"]
    ] *= scale_2020

    dm_all_stock_tmp = estimate_stock_non_res(dm_area_2020, dm_residential_stock_tmp)
    del dm_rooms, dm_pop_bld_type

    dm_all_stock_tmp.array[dm_all_stock_tmp.array < 0] = 0

    ##########################################
    ########### Floor area New ots ###########
    ##########################################

    # Load new building floor area
    dm_new_1 = get_new_building()

    # Reconcile two new build area estimates
    dm_area_new = estimate_floor_area(dm_new_1, dm_new_2, years_ots)
    del dm_new_1, dm_new_2

    dm_area_new.array[dm_area_new.array < 0] = np.nan
    dm_area_new = linear_fitting(dm_area_new, years_ots, min_t0=0, min_tb=0)

    ##########################################################
    ########### Floor-area Waste + Recompute Stock ###########
    ##########################################################

    dm_all = estimate_waste_fix_stock(dm_all_stock_tmp, dm_area_new, years_ots)
    del dm_residential_stock_tmp, dm_area_new, dm_area_2020

    dm_all.array[dm_all.array < 0] = np.nan
    dm_all = linear_fitting(dm_all, years_ots, min_t0=0, min_tb=0)

    dm_stock_tot = dm_all.filter({"Variables": ["bld_floor-area_stock"]})
    dm_new_tot = dm_all.filter({"Variables": ["bld_floor-area_new"]})
    dm_waste_tot = dm_all.filter({"Variables": ["bld_floor-area_waste"]})

    # Non-residential (services) floor area by building type and energy class.
    non_res_cats = ["education", "health", "hotels", "offices", "other", "trade"]
    energy_classes = ["B", "C", "D", "E", "F"]
    dm_services_floor_type = dm_stock_tot.filter({"Categories1": non_res_cats})
    dm_services_floor_type.rename_col(
        "bld_floor-area_stock", "bld_floor-area_services", dim="Variables"
    )
    bso_shares = extract_nonres_class_shares_BSO()

    n_countries = len(dm_services_floor_type.col_labels["Country"])
    n_years = len(dm_services_floor_type.col_labels["Years"])
    n_types = len(non_res_cats)
    n_classes = len(energy_classes)
    arr_5d = np.zeros((n_countries, n_years, 1, n_types, n_classes))
    idx_t = dm_services_floor_type.idx
    for ic, country in enumerate(dm_services_floor_type.col_labels["Country"]):
        shares_c = bso_shares.get(country, bso_shares.get("EU27"))
        for it, btype in enumerate(non_res_cats):
            for icl, cls in enumerate(energy_classes):
                share_val = (
                    float(shares_c.loc[cls, btype])
                    if btype in shares_c.columns
                    else 0.0
                )
                arr_5d[ic, :, 0, it, icl] = (
                    dm_services_floor_type.array[
                        ic, :, idx_t["bld_floor-area_services"], idx_t[btype]
                    ]
                    * share_val
                )

    dm_services_floor = DataMatrix(
        col_labels={
            "Country": dm_services_floor_type.col_labels["Country"],
            "Years": dm_services_floor_type.col_labels["Years"],
            "Variables": ["bld_floor-area_services"],
            "Categories1": non_res_cats,
            "Categories2": energy_classes,
        },
        units={"bld_floor-area_services": "m2"},
    )
    dm_services_floor.array = arr_5d

    result = (dm_all, dm_stock_tot, dm_new_tot, dm_waste_tot, dm_services_floor, dm_pop)
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved cache → {_CACHE_PATH}")
    return result


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_ots

    dm_all, dm_stock_tot, dm_new_tot, dm_waste_tot, dm_services_floor, dm_pop = run(
        years_ots
    )
    print("dm_all variables:", dm_all.col_labels["Variables"])
    print("dm_stock_tot shape:", dm_stock_tot.array.shape)
    print("dm_services_floor shape:", dm_services_floor.array.shape)
    print("dm_pop shape:", dm_pop.array.shape)
