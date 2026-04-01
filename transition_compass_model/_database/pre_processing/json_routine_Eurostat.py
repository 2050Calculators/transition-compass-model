import pandas as pd
from transition_compass_model.model.common.data_matrix_class import DataMatrix
import json
import requests


def eurostat_json_to_df(data):
    """
    Convert a Eurostat JSON dataset into a pandas DataFrame.
    """

    # # dimension
    # data["dimension"]["engine"]["category"]["index"] # this is index for total
    # data["dimension"]["time"]["category"]["index"] # this is index for time

    # # value
    # data["value"].keys()

    ids = data["id"]  # dimension order
    sizes = data["size"]  # dimension sizes
    dimensions = data["dimension"]
    values = data.get("value", {})
    status = data.get("status", {})

    # Pre-build index → label maps for each dimension
    dim_maps = {}
    for dim in ids:
        cat = dimensions[dim]["category"]
        index_to_label = {v: k for k, v in cat["index"].items()}
        dim_maps[dim] = index_to_label

    # Helper: decode flattened index
    def decode_index(idx, sizes):
        coords = []
        for size in reversed(sizes):
            coords.append(idx % size)
            idx //= size
        return list(reversed(coords))

    rows = []

    for k, v in values.items():
        idx = int(k)
        coords = decode_index(idx, sizes)

        row = {}
        for dim, coord in zip(ids, coords):
            row[dim] = dim_maps[dim][coord]

        row["value"] = v

        # Optional: include status flags if present
        if k in status:
            row["status"] = status[k]

        rows.append(row)

    return pd.DataFrame(rows)


def eurostat_df_to_dm(df, filter, mapping_dim, unit):

    # change years to numeric
    df["time"] = df["time"].astype(int)

    # if unit not there, add it
    if "unit" not in df.columns:
        df["unit"] = filter["unit"][0]

    # select rows
    for key in filter.keys():
        df = df.loc[df[key].isin(filter[key]), :]

    # Rename column to 'Country' and 'Years'
    df.rename(columns={mapping_dim["Country"]: "Country"}, inplace=True)
    df.rename(columns={mapping_dim["Years"]: "Years"}, inplace=True)

    # Create a single column variables_categories1_categories2[unit] called var_long_name
    num_cat = 0
    concat_list = ["Categories1", "Categories2"]
    if mapping_dim["Variables"] in df.columns:
        df["var_long_name"] = df[mapping_dim["Variables"]]
    else:
        df["var_long_name"] = mapping_dim["Variables"]
    for cat in concat_list:
        if cat in mapping_dim.keys():
            df[mapping_dim[cat]] = df[mapping_dim[cat]].str.replace(
                "_", "", regex=False
            )
            df["var_long_name"] = df["var_long_name"] + "_" + df[mapping_dim[cat]]
            num_cat = num_cat + 1
    df["var_long_name"] = df["var_long_name"] + "[" + unit + "]"

    # reshape
    df = df.loc[:, ["Country", "Years", "var_long_name", "value"]]
    df = df.pivot(
        index=["Country", "Years"], columns="var_long_name", values="value"
    ).reset_index()
    multi_index = pd.MultiIndex.from_product(
        [df["Country"].unique(), df["Years"].unique()], names=["Country", "Years"]
    )
    df_all = df.set_index(["Country", "Years"]).reindex(multi_index).reset_index()
    df_all = df_all.sort_values(["Country", "Years"])

    # get dm
    dm = DataMatrix.create_from_df(df_all, 0)

    # deepen
    size_dim = len(mapping_dim.keys())
    if size_dim == 3:
        dm = dm.copy()
    elif size_dim == 4:
        dm.deepen()
        for c in ["Variables", "Categories1"]:
            names_original = filter[mapping_dim[c]].copy()
            names_original.sort()
            names_new = dm.col_labels[c].copy()
            names_new.sort()
            if len(names_original) == len(names_new):
                for original, new in zip(names_original, names_new):
                    dm.rename_col(new, original, c)
            else:
                raise ValueError("Some variables are missing in the database")
    elif size_dim == 5:
        dm.deepen_twice()
        for c in ["Variables", "Categories1", "Categories2"]:
            names_original = filter[mapping_dim[c]].copy()
            names_original.sort()
            names_new = dm.col_labels[c].copy()
            names_new.sort()
            if names_original == names_new:
                for original, new in zip(names_original, names_new):
                    dm.rename_col(new, original, c)
            else:
                raise ValueError("Some variables are missing in the database")

    # return
    return dm


def get_data_json_eurostat(file_path, filter, mapping_dim, unit):

    # Open and read the JSON file
    with open(file_path, "r") as file:
        data = json.load(file)

    # get df
    df = eurostat_json_to_df(data)

    # get dm
    dm = eurostat_df_to_dm(df, filter, mapping_dim, unit)

    # return
    return dm


def get_eurostat_jsonstat(
    dataset_code, *, compress=False, lang="en", extra_params=None, timeout=120
):
    """
    Fetch Eurostat SDMX 3.0 data as JSON (JSON-stat-like payload, as in your example).
    """
    BASE = (
        "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT"
    )
    url = f"{BASE}/{dataset_code}/1.0"
    params = {
        "compress": "true" if compress else "false",
        "format": "json",  # this is the JSON-stat-style payload you showed
        "lang": lang,
    }
    if extra_params:
        params.update(extra_params)

    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_data_api_eurostat_via_json(file_name, filter, mapping_dim, unit):

    # get data
    data = get_eurostat_jsonstat(file_name, compress=False, lang="en")
    if "warning" in data and data["warning"]:
        raise RuntimeError(f"Eurostat warning: {data['warning']}")

    # get df
    df = eurostat_json_to_df(data)

    # get dm
    dm = eurostat_df_to_dm(df, filter, mapping_dim, unit)

    # return
    return dm
