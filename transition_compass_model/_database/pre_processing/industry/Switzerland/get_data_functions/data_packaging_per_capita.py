import os

import numpy as np
import pandas as pd

from transition_compass_model.model.common.data_matrix_class import DataMatrix


def get_packaging_data(current_file_directory, years_ots):
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
    dm = dm.filter(
        {"Years": years_ots, "Variables": ["paper-pack", "paper-print", "paper-san"]}
    )

    # plastic
    # take the series of PET and increase it by factor of total consumption of plastic today
    # https://www.bafu.admin.ch/bafu/en/home/topics/waste/guide-to-waste-a-z/plastics.html:
    # Around one million tonnes of plastics are consumed in Switzerland every year – that's 120 kilograms per capita (reference year 2017).
    # Around 790,000 tonnes of plastic waste are generated every year, almost half of which is used for less than a year, e.g. as packaging.
    # so we can say that around 395000 are the plastic packaging waste, and we can assume that that's the plastic packaging consumption.

    filepath = os.path.join(
        current_file_directory,
        "../data/waste/0_Déchets_Quantités_produites_et_recyclées/je-f-02.03.02.11.xlsx",
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
        "../data/waste/0_Déchets_Quantités_produites_et_recyclées/je-f-02.03.02.11.xlsx",
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
        "../data/waste/0_Déchets_Quantités_produites_et_recyclées/je-f-02.03.02.11.xlsx",
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
