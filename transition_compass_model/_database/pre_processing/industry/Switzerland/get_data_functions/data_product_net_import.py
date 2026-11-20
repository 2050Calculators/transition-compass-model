import os

import numpy as np
import pandas as pd


def get_import_export_chf(current_file_directory, trade_flow, df_map_cpa):
    # the data is not on the API, get manually
    file = f"2_5_{trade_flow}_cpa_knt_en"
    filepath = os.path.join(
        current_file_directory, f"../data/import-export/{file}.xlsx"
    )
    df = pd.read_excel(filepath, sheet_name="TOTAL")

    # reshape
    df = df.iloc[:, 0:11]
    df.iloc[5, 0:2] = df.iloc[4, 0:2]
    df.columns = df.iloc[5, :]
    df = df.iloc[6:-5, :]
    df.rename(
        columns={
            "CPA code": "code",
            "Statistical classification of products by activity (CPA) ": "variable",
        },
        inplace=True,
    )
    df = pd.melt(df, id_vars=["code", "variable"], var_name="year")
    df.loc[df["value"] == "-", "value"] = np.nan
    df["value"] = df["value"].astype(float)

    # rename variables with mapping
    df = df.merge(
        df_map_cpa.loc[:, ["code", "variable", "variable-calc"]],
        "left",
        ["code", "variable"],
    )
    df = df.loc[~df["variable-calc"].isna(), :]
    df = df.loc[:, ["variable-calc", "year", "value"]]
    df["trade-flow"] = trade_flow
    df = df.groupby(["variable-calc", "year", "trade-flow"], as_index=False)[
        "value"
    ].agg(sum)

    return df


def get_price_data(current_file_directory, flow):
    # get data
    file = f"2_8_VT_NST_{flow}_en"
    filepath = os.path.join(
        current_file_directory, f"../data/import-export/{file}.xlsx"
    )
    df = pd.read_excel(filepath)

    # reshape
    df = df.iloc[3:-3, 0:3]
    df.columns = ["variable", "kt", "chf-mio"]
    df["kt"] = df["kt"].astype(float)
    df["chf-mio"] = df["chf-mio"].astype(float)

    # rename variables with mapping
    filepath_map = os.path.join(
        current_file_directory,
        "../data/products_materials_mapping/NST_calc_mapping.xlsx",
    )
    df_map = pd.read_excel(filepath_map)
    df_map = df_map.iloc[:, 0:3]
    df_map.columns = df_map.iloc[0, :]
    df_map = df_map.iloc[1:, :]
    df_map.rename(columns={"Product groups": "variable"}, inplace=True)
    df_map["variable-calc"] = df_map["calc_industry_material"]
    df_map.loc[df_map["calc_industry_material"].isna(), "variable-calc"] = df_map.loc[
        df_map["calc_industry_material"].isna(), "calc_industry_product"
    ]
    df = df.merge(df_map.loc[:, ["variable", "variable-calc"]], "left", ["variable"])
    df = df.loc[~df["variable-calc"].isna(), ["variable-calc", "kt", "chf-mio"]]

    # volume-weighted average: total chf-mio / total kt across all NST4 groups per
    # model category — correct aggregation for CHF-to-tonne conversion
    agg = df.groupby("variable-calc", as_index=False).agg(
        total_kt=("kt", "sum"),
        total_chf_mio=("chf-mio", "sum"),
    )
    agg["price[chf/t]"] = agg["total_chf_mio"] / agg["total_kt"] * 1000
    df = agg[["variable-calc", "price[chf/t]"]].copy()
    df["trade-flow"] = flow

    return df


def get_price_index_data(current_file_directory, df_map_cpa):
    # get data
    file = "su-q-05.04.03.02-ppi-ipp-det"
    filepath = os.path.join(current_file_directory, f"../data/price-index/{file}.xlsx")
    df = pd.read_excel(filepath, sheet_name="INDEX_y")
    df.columns = df.iloc[6, :]
    df = df.iloc[:, [3] + list(range(15, len(df.columns)))]
    df = df.loc[df["Product code"].isin(df_map_cpa["code-2digit"]), :]
    df.rename(columns={"Product code": "code"}, inplace=True)
    df = pd.melt(df, id_vars=["code"], var_name="year")
    df.loc[df["value"] == "…", "value"] = np.nan
    df["value"] = df["value"].astype(float)
    df_temp = df_map_cpa.loc[:, ["code-2digit", "variable-calc"]]
    df_temp.rename(columns={"code-2digit": "code"}, inplace=True)
    df = df.merge(df_temp, "left", ["code"])
    df.loc[df["code"] == "B", "variable-calc"] = "other"
    df = df.loc[~df["variable-calc"].isna(), :]
    df = df.loc[:, ["variable-calc", "year", "value"]]
    df = df.groupby(["variable-calc", "year"], as_index=False)["value"].agg(np.mean)
    df["year"] = df["year"].astype(int)
    df["year"] = df["year"].astype(str)
    df_price_index = df.copy()

    return df_price_index


def get_io_data(current_file_directory, year):
    filepath = os.path.join(
        current_file_directory, f"../data/input-output/su-e-04.03-IOT-{year}.xlsx"
    )
    df_io = pd.read_excel(filepath, sheet_name="siot")

    # IMPORTS AND SUPPLY

    df = df_io.copy()

    # rows
    word = "PRODUCTS (CPA)"
    mask = df.applymap(lambda x: word in str(x) if pd.notna(x) else False)
    column_with_match = df.columns[mask.any(axis=0)].tolist()[0]
    row_with_match = df[column_with_match][mask.any(axis=1)].tolist()[0]
    rows = [
        row_with_match,
        "Output at basic prices",
        "Imports cif",
        "Supply at basic prices",
    ]
    df = df.loc[df[column_with_match].isin(rows), :]

    # columns
    df = df.iloc[:, df.columns.get_loc(column_with_match) :]
    arr = np.array(df.iloc[0, :])
    nan_first_index = np.where(pd.isna(arr))[0][0]
    df = df.iloc[:, 0:nan_first_index]
    codes_raw = list(np.array(df.iloc[0, 1:]))
    arr = np.array(df.iloc[0, :].astype(str))
    df.iloc[0, :] = np.array(
        [x.strip() if isinstance(x, str) else x for x in arr], dtype=object
    )
    # code_first = df.iloc[0,1]
    # code_last = df.iloc[0,len(df.columns)-1]

    # reshape
    df.columns = df.iloc[0, :]
    df = df.iloc[1:, :]
    df.rename(columns={"PRODUCTS (CPA)": "variable"}, inplace=True)
    df = pd.melt(df, id_vars=["variable"], var_name="code")
    df_main = df.copy()

    # EXPORTS

    df = df_io.copy()

    # columns
    word = "Exports"
    mask = df.applymap(lambda x: word in str(x) if pd.notna(x) else False)
    columns_with_exports = df.columns[mask.any(axis=0)].tolist()[0]
    word = "Code"
    mask = df.applymap(lambda x: word in str(x) if pd.notna(x) else False)
    columns_with_code = df.columns[mask.any(axis=0)].tolist()[0]
    df = df.loc[:, [columns_with_code, columns_with_exports]]

    # rows
    df = df.loc[df.iloc[:, 0].isin(codes_raw), :]

    # reshape
    df.columns = ["code", "Exports"]
    df = pd.melt(df, id_vars=["code"], var_name="variable")
    arr = np.array(df["code"].astype(str))
    df["code"] = np.array(
        [x.rstrip() if isinstance(x, str) else x for x in arr], dtype=object
    )

    # concat
    df_main = pd.concat([df_main, df])
    df_main["year"] = year
    df_main = df_main.loc[:, ["variable", "code", "year", "value"]]
    df_main.sort_values(["variable", "code", "year"], inplace=True)
    df_main["value"] = df_main["value"].astype(float)

    return df_main
