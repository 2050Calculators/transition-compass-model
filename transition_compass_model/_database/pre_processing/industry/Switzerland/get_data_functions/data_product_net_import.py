import os
import pandas as pd
import numpy as np
from transition_compass_model.model.common.data_matrix_class import DataMatrix


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
    df["t"] = df["kt"] * 1000
    df["chf"] = df["chf-mio"] * 1000000
    df["price[chf/t]"] = df["chf"] / df["t"]
    df = df.loc[:, ["variable", "price[chf/t]"]]

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
    df = df.loc[~df["variable-calc"].isna(), ["variable-calc", "price[chf/t]"]]
    df = df.groupby(["variable-calc"], as_index=False)["price[chf/t]"].agg(np.mean)
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


def get_packaging_data(current_file_directory):

    # paper
    # note: we will take the data of overall paper and split them in the packaging, print, and sanitary
    # so from je-f-02.03.02.11, assuming that the taux is the taux de collecte, we will multiply the waste
    # number by this taux de collecte

    # from https://www.bafu.admin.ch/bafu/en/home/topics/waste/guide-to-waste-a-z/paper-and-cardboard.html:
    # produce 1.2 million tonnes of paper.

    # filepath = os.path.join(current_file_directory, '../data/waste/0_Déchets Quantités produites et recyclées/je-f-02.03.02.11.xlsx')
    # df = pd.read_excel(filepath)
    # df.columns = df.iloc[3,:]
    # df = df.iloc[8:10,:]
    # df = pd.melt(df, id_vars = ["Matériaux ",'Unité'], var_name='year')
    # df.loc[df["value"] == '…',"value"] = np.nan
    # df["value"] = df["value"].astype(float)
    # df.columns = ["material","unit","year","value"]
    # df["material"] = "paper"
    # df = df.pivot(index=["material","year"], columns="unit", values='value').reset_index()
    # df.columns = ['material', 'year', '%', 't']
    # df["%"] = df["%"]/100
    # df["value"] = df["t"] * (1 - df["%"] + 1)
    # df = df.loc[:,["material","year","value"]]

    # fao has the split for paper, so I rely on that
    filepath = os.path.join(
        current_file_directory, "../data/production-fao/FAOSTAT_data_en_8-21-2025.csv"
    )
    df = pd.read_csv(filepath)
    df.columns
    df = df.loc[:, ["Item", "Year", "Unit", "Value"]]
    df["Item"] = df["Item"] + "[" + df["Unit"] + "]"
    df = df.loc[:, ["Item", "Year", "Value"]]
    df = df.pivot(index=["Year"], columns="Item", values="Value").reset_index()
    df["Country"] = "Switzerland"
    df.rename(columns={"Year": "Years"}, inplace=True)
    dm = DataMatrix.create_from_df(df, 0)
    dm.groupby(
        {
            "paper-pack-pre": [
                "Cartonboard",
                "Case materials",
                "Other papers mainly for packaging",
                "Wrapping papers",
            ]
        },
        "Variables",
        inplace=True,
    )
    dm.groupby(
        {
            "paper-pack": [
                "paper-pack-pre",
                "Wrapping and packaging paper and paperboard (1961-1997)",
            ]
        },
        "Variables",
        inplace=True,
    )
    # df_temp = dm.write_df()
    dm.groupby(
        {
            "paper-print": ["Newsprint", "Printing and writing papers"],
            "paper-san": ["Household and sanitary papers"],
        },
        "Variables",
        inplace=True,
    )
    dm.drop(
        "Variables",
        [
            "Other paper and paperboard",
            "Other paper and paperboard, not elsewhere specified",
        ],
    )

    # plastic
    # take the series of PET and increase it by factor of total consumption of plastic today
    # https://www.bafu.admin.ch/bafu/en/home/topics/waste/guide-to-waste-a-z/plastics.html:
    # Around one million tonnes of plastics are consumed in Switzerland every year – that's 120 kilograms per capita (reference year 2017).
    # Around 790,000 tonnes of plastic waste are generated every year, almost half of which is used for less than a year, e.g. as packaging.
    # so we can say that around 395000 are the plastic packaging waste, and we can assume that that's the plastic packaging consumption.

    filepath = os.path.join(
        current_file_directory,
        "../data/waste/0_Déchets Quantités produites et recyclées/je-f-02.03.02.11.xlsx",
    )
    df = pd.read_excel(filepath)
    df.columns = df.iloc[3, :]
    df = df.iloc[29:31, :]
    df = pd.melt(df, id_vars=["Matériaux ", "Unité"], var_name="year")
    df.loc[df["value"] == "…", "value"] = np.nan
    df["value"] = df["value"].astype(float)
    df.columns = ["material", "unit", "year", "value"]
    df["material"] = "pet"
    df = df.pivot(
        index=["material", "year"], columns="unit", values="value"
    ).reset_index()
    df.columns = ["material", "year", "%", "t"]
    df["%"] = df["%"] / 100
    df["value"] = df["t"] / df["%"]
    df = df.loc[:, ["material", "year", "value"]]

    plastic_packaging_consumption_2023 = 395000
    df["factor"] = float(
        (plastic_packaging_consumption_2023 - df.loc[df["year"] == 2023, "value"])
        / df.loc[df["year"] == 2023, "value"]
    )
    df["value"] = df["value"] * (1 + df["factor"])
    df = df.loc[:, ["material", "year", "value"]]
    df["material"] = "plastic-pack[t]"

    df = df.pivot(index=["year"], columns="material", values="value").reset_index()
    df["Country"] = "Switzerland"
    df.rename(columns={"year": "Years"}, inplace=True)
    dm_temp = DataMatrix.create_from_df(df, 0)
    dm_temp.drop("Years", [1985])
    dm_temp.add(np.nan, "Years", [1991, 1992], dummy=True)
    dm_temp.sort("Years")
    for y in [1990, 1991, 1992]:
        dm_temp[:, y, ...] = dm_temp[:, 1993, ...]
    # df_temp = dm_temp.write_df()
    dm.append(dm_temp, "Variables")

    # aluminium
    # note: I get the amount of waste of aluminium packages and assuming that that's the same of consumption
    filepath = os.path.join(
        current_file_directory,
        "../data/waste/0_Déchets Quantités produites et recyclées/je-f-02.03.02.11.xlsx",
    )
    df = pd.read_excel(filepath)
    df.columns = df.iloc[3, :]
    df = df.iloc[23:24, :]
    df = pd.melt(df, id_vars=["Matériaux ", "Unité"], var_name="year")
    df["value"] = df["value"].astype(float)
    df.columns = ["material", "unit", "year", "value"]
    df["material"] = "aluminium-pack[t]"
    df = df.loc[:, ["material", "year", "value"]]
    df = df.pivot(index=["year"], columns="material", values="value").reset_index()
    df["Country"] = "Switzerland"
    df.rename(columns={"year": "Years"}, inplace=True)
    dm_temp = DataMatrix.create_from_df(df, 0)
    dm_temp.drop("Years", [1985])
    dm_temp.add(np.nan, "Years", [1991, 1992], dummy=True)
    dm_temp.sort("Years")
    for y in [1990, 1991, 1992]:
        dm_temp[:, y, ...] = dm_temp[:, 1993, ...]
    # df_temp = dm_temp.write_df()
    dm.append(dm_temp, "Variables")

    # glass pack
    filepath = os.path.join(
        current_file_directory,
        "../data/waste/0_Déchets Quantités produites et recyclées/je-f-02.03.02.11.xlsx",
    )
    df = pd.read_excel(filepath)
    df.columns = df.iloc[3, :]
    df = df.iloc[14:16, :]
    df = pd.melt(df, id_vars=["Matériaux ", "Unité"], var_name="year")
    df["value"] = df["value"].astype(float)
    df.columns = ["material", "unit", "year", "value"]
    df["material"] = "glass-pack[t]"
    df = df.pivot(
        index=["material", "year"], columns="unit", values="value"
    ).reset_index()
    df.columns = ["material", "year", "%", "t"]
    df["%"] = df["%"] / 100
    df["value"] = df["t"] / df["%"]
    df = df.loc[:, ["material", "year", "value"]]
    df = df.pivot(index=["year"], columns="material", values="value").reset_index()
    df["Country"] = "Switzerland"
    df.rename(columns={"year": "Years"}, inplace=True)
    dm_temp = DataMatrix.create_from_df(df, 0)
    dm_temp.drop("Years", [1985])
    dm_temp.add(np.nan, "Years", [1991, 1992], dummy=True)
    dm_temp.sort("Years")
    for y in [1990, 1991, 1992]:
        dm_temp[:, y, ...] = dm_temp[:, 1993, ...]
    # df_temp = dm_temp.write_df()
    dm.append(dm_temp, "Variables")

    return dm
