import math

import numpy as np
import pandas as pd
from pre_processing.params import country_list

from transition_compass_model.model.common.auxiliary_functions import (
    dm_add_missing_variables,
    linear_fitting,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

# Canton labels of the Swiss Stats SDMX dimension UV_HGDE_KT (vehicle stock and
# registrations, agency CH1.MFZ_IVS) that differ from params.country_list.
_SWISS_STATS_CANTON_LABELS = {
    "Basel Stadt": "Basel-Stadt",
    "Basel Landschaft": "Basel-Landschaft",
}


def swiss_stats_canton_filter():
    """UV_HGDE_KT filter values for Switzerland ("Total") and every canton in country_list."""
    cantons = [c for c in country_list if c != "Switzerland"]
    return ["Total"] + [_SWISS_STATS_CANTON_LABELS.get(c, c) for c in cantons]


def rename_swiss_stats_cantons(dm):
    """Rename UV_HGDE_KT labels to country_list names (in place): Swiss total to
    "Switzerland", hyphenated Basel labels to the country_list spelling."""
    labels = dm.col_labels["Country"]
    for total in ("Total_Country", "Total"):
        if total in labels:
            dm.rename_col(total, "Switzerland", dim="Country")
    for ours, api in _SWISS_STATS_CANTON_LABELS.items():
        if api in dm.col_labels["Country"]:
            dm.rename_col(api, ours, dim="Country")
    return dm


def fill_var_nans_based_on_var_curve(dm, var_nan, var_ref, keep_all_vars=False):
    # Fills nan in a variable dm based on another variable curve
    dm.operation(var_nan, "/", var_ref, out_col="ratio", unit="%")
    linear_fitting(dm, dm.col_labels["Years"])
    dm.operation("ratio", "*", var_ref, out_col=var_nan + "_ok", unit=dm.units[var_ref])
    if not keep_all_vars:
        dm.filter({"Variables": [var_nan + "_ok"]}, inplace=True)
    else:
        dm.drop(dim="Variables", col_label=[var_nan, "ratio"])
    dm.rename_col(var_nan + "_ok", var_nan, dim="Variables")

    return dm


def drop_if_smaller_than_0_01_per_category_country_cat_2(
    dm, col_to_drop, cat_to_drop="Categories2"
):
    if cat_to_drop == "Categories2":
        cat_not_to_drop = "Categories1"
    else:
        cat_not_to_drop = "Categories2"

    dm_norm = dm.normalise(dim=cat_to_drop, inplace=False)
    idx = dm_norm.idx
    dm_missing = dm_norm.filter({cat_to_drop: [col_to_drop]}).copy()
    if (dm_missing.array < 0.01).any():
        dm.drop(col_label=col_to_drop, dim=cat_to_drop)
        return dm

    for country in dm.col_labels["Country"]:
        for year in dm.col_labels["Years"]:
            for cat in dm.col_labels[cat_not_to_drop]:
                # If "Inconnu" is more than 20% remove the data points
                if cat_to_drop == "Categories2":
                    if (
                        dm_norm.array[
                            idx[country], idx[year], 0, idx[cat], idx[col_to_drop]
                        ]
                        > 0.2
                    ):
                        dm.array[idx[country], idx[year], 0, idx[cat], :] = np.nan

    # for i in range(2):
    #         window_size = 3  # Change window size to control the smoothing effect
    #         data_smooth = moving_average(
    #             dm.array, window_size, axis=dm.dim_labels.index("Years")
    #         )
    #         dm.array[:, 1:-1, ...] = data_smooth

    # Distribute Inconnu on other categories based on their share
    cat_other = [cat for cat in dm.col_labels[cat_to_drop] if cat != col_to_drop]
    dm_other = dm.filter({cat_to_drop: cat_other}, inplace=False)
    dm_other.normalise(dim=cat_to_drop, inplace=True)
    dm_other.array = np.nan_to_num(dm_other.array)
    idx = dm.idx
    arr_inc = (
        np.nan_to_num(dm.array[:, :, :, :, idx[col_to_drop], np.newaxis])
        * dm_other.array
    )
    dm.drop(dim=cat_to_drop, col_label=col_to_drop)
    dm.array = dm.array + arr_inc
    return dm


def drop_if_smaller_than_0_01(
    dm_og: DataMatrix,
    cat_to_drop="Categories2",
    col_to_drop="Other",
    ref_country=None,
) -> DataMatrix:
    """Drop a column from a category in a datamatrix

    The dropped column is spread over the other columns in proportion to the country's
    own distribution. Above 15% of the total this is not trusted: it raises, unless
    ref_country is given, in which case the years above 15% use ref_country's
    distribution (of the remaining columns) to spread the dropped column.

    Args:
        dm_og (DataMatrix)
        cat_to_drop (str, optional): Categories with the column you want to drop. Defaults to "Categories2".
        col_to_drop (str, optional): _description_. Defaults to "Other".
        ref_country (str, optional): country whose distribution is used above 15%.

    Returns:
        DataMatrix: dm with the desired column removed
    """
    dm = dm_og.copy()
    dm_ratio = dm_og.copy()

    if cat_to_drop == "Categories2":
        # sum all data of categories 1 and collapse categories 1 so categories 2 becomes category 1
        dm_ratio.group_all("Categories1")

    dm_ratio = dm_ratio.normalise(dim="Categories1", inplace=False)
    idx = dm_ratio.idx

    for country in dm_ratio.col_labels["Country"]:
        dm_missing = dm_ratio.filter(
            {"Country": [country], "Categories1": [col_to_drop]}
        ).copy()

        dm_country = dm.filter({"Country": [country]}).copy()

        if (dm_missing.array > 0.01).any():
            years_over = []
            if (dm_missing.array > 0.15).any():
                if ref_country is None:
                    raise ValueError(
                        f"{col_to_drop} category is greater than 15%  of the fleet for {country}, it cannot be discarded {dm_missing.col_labels['Variables']} it cannot be discarded"
                    )
                years_axis = dm_missing.dim_labels.index("Years")
                other_axes = tuple(
                    a for a in range(dm_missing.array.ndim) if a != years_axis
                )
                years_over = list(
                    np.where((dm_missing.array > 0.15).any(axis=other_axes))[0]
                )
                print(
                    f'"{col_to_drop}" category is greater than 15% of the fleet for {country} in '
                    f"{[dm_missing.col_labels['Years'][y] for y in years_over]}: spread with "
                    f"the {ref_country} distribution in those years"
                )

            print(
                f'"{col_to_drop}" category is greater than 1% of the fleet for {country}, it cannot be discarded {dm_missing.col_labels["Variables"]}'
            )
            dm_without_other = dm_country.copy()
            dm_without_other.drop(col_label=col_to_drop, dim=cat_to_drop)
            dm_without_other_normalised = dm_without_other.normalise(
                dim=cat_to_drop, inplace=False
            )
            if years_over:
                dm_ref = dm_og.filter({"Country": [ref_country]})
                dm_ref.drop(col_label=col_to_drop, dim=cat_to_drop)
                dm_ref.normalise(dim=cat_to_drop, inplace=True)
                dm_without_other_normalised.array[:, years_over, ...] = dm_ref.array[
                    :, years_over, ...
                ]

            dm_without_other.array = (
                dm_without_other.array
                + dm_country.filter({cat_to_drop: [col_to_drop]}).array
                * dm_without_other_normalised.array
            )
            idx_dm = dm.idx
            keep_idx = [
                idx_dm[col]
                for col in dm_country.col_labels[cat_to_drop]
                if col != col_to_drop
            ]
            idx_dm_country = dm_country.idx
            if cat_to_drop == "Categories1":
                # dm.array[c, :, :, keep_idx] has shape (Categories1, Years, Variables,
                # [Categories2]): move Categories1 (axis 2 after taking the country) to
                # the front, not the last axis.
                dm.array[idx_dm[country], :, :, keep_idx] = np.moveaxis(
                    dm_without_other.array[idx_dm_country[country], ...], 2, 0
                )
            elif cat_to_drop == "Categories2":
                dm.array[idx_dm[country], :, :, :, keep_idx] = np.moveaxis(
                    dm_without_other.array[idx_dm_country[country], :, :, :, :], -1, 0
                )
                # dm.array[idx[country],:,:,:,keep_idx] = dm_without_other.array[idx[country],:,:,:,:]

    # Drop
    dm.drop(col_label=col_to_drop, dim=cat_to_drop)
    return dm


def df_fso_excel_to_dm(
    df,
    header_row,
    names_dict,
    var_name,
    unit,
    num_cat,
    keep_first=False,
    country="Switzerland",
):
    # Federal statistical office df from excel to dm
    # Change headers
    new_header = df.iloc[header_row]
    new_header.values[0] = "Variables"
    df.columns = new_header
    df = df[header_row + 1 :].copy()
    # Remove nans and empty columns/rows
    if np.nan in df.columns:
        df.drop(columns=np.nan, inplace=True)
    df.set_index("Variables", inplace=True)
    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    # Filter rows that contain at least one number (integer or float)
    df = df[df.apply(lambda row: row.map(pd.api.types.is_number), axis=1).any(axis=1)]
    df_clean = df.loc[
        :, df.apply(lambda col: col.map(pd.api.types.is_number)).any(axis=0)
    ].copy()
    # Extract only the data we are interested in:
    df_filter = df_clean.loc[names_dict.keys()].copy()
    df_filter = df_filter.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    # df_filter = df_filter.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
    df_filter.reset_index(inplace=True)
    # Keep only first 10 caracters
    df_filter["Variables"] = df_filter["Variables"].replace(names_dict)
    if keep_first:
        df_filter = df_filter.drop_duplicates(subset=["Variables"], keep="first")
    df_filter = df_filter.groupby(["Variables"]).sum()
    df_filter.reset_index(inplace=True)

    # Pivot the dataframe
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

    # Drop non numeric values in Years col
    df_pivot["Years"] = pd.to_numeric(df_pivot["Years"], errors="coerce")
    df_pivot = df_pivot.dropna(subset=["Years"])

    dm = DataMatrix.create_from_df(df_pivot, num_cat=num_cat)
    return dm


def add_aviation_data_to_DM(DM_transport_new, DM_transport):
    for key, DM_new in DM_transport_new.items():
        if isinstance(DM_new, dict):
            add_aviation_data_to_DM(DM_new, DM_transport[key])
        else:
            dm_new = DM_new
            dm = DM_transport[key]
            if "Categories1" in dm.dim_labels:
                if "aviation" in dm.col_labels["Categories1"]:
                    dm_aviation = dm.filter({"Categories1": ["aviation"]})
                    if "Categories2" in dm_aviation.dim_labels:
                        dm_add_missing_variables(
                            dm_new,
                            {"Categories2": dm_aviation.col_labels["Categories2"]},
                        )
                    try:
                        dm_new.append(
                            dm_aviation.filter(
                                {"Country": dm_new.col_labels["Country"]}
                            ),
                            dim="Categories1",
                        )
                    except Exception:
                        raise RuntimeError(
                            f"Warning: Error occurred when trying to add 'aviation' to variable {dm_new.col_labels['Variables']}"
                        )

    return


def assert_len_dimension_for_call_to_api(filtering: dict):
    """Check whether the nb of data being call to the api is smaller than 1000 to ensure successful call

    Args:
        filtering (dict): dimension and list of values to call to the api
    """
    len_dimension = [len(i) for i in filtering.values()]
    res = math.prod(len_dimension)
    assert res < 1000
