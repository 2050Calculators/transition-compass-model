import os

import numpy as np
import pandas as pd

from transition_compass_model.model.common.data_matrix_class import DataMatrix


def get_prodcom_production_data(current_file_directory):
    #########################################
    ##### GET CLEAN DATAFRAME WITH DATA #####
    #########################################

    # get data
    # df = eurostat.get_data_df("ds-056120")
    filepath = os.path.join(current_file_directory, "../data/eurostat/ds-056120.csv")
    # df.to_csv(filepath, index = False)
    df = pd.read_csv(filepath)

    # NOTE: as ds-056120 is on sold production, then we assume sold production = demand
    # import and export should be always on "sold", as a country exports what's demanded
    # and it imports what it demands.

    # explore data
    df.columns
    product_code = "29102100"
    variabs = ["PRODVAL", "PRODQNT", "EXPVAL", "EXPQNT", "IMPVAL", "IMPQNT"]
    df_sub = df.loc[df["prccode"].isin([product_code]), :]
    df_sub = df_sub.loc[df_sub["indicators\\TIME_PERIOD"].isin(variabs), :]
    len(df_sub["decl"].unique())
    df_sub = df_sub.loc[df_sub["decl"].isin([4, 2027])]  # get germnay and EU27_2020
    # it seems values are generally there for all variables
    # I will do:
    # product-net-import[%] = (IMPQNT - EXPQNT) / PRODQNT
    # For now I will do the adjustments (nan filling, jumps, predictions) on the variables
    # EXPQNT, IMPQNT and PRODQNT, and then I will make the variable product-net-import[%] at the end.
    # The alternative would be to make product-net-import[%] from the
    # beginning, and do all the adjustments on that variable. TBC.

    # get "PRODQNT", "EXPQNT", "IMPQNT", "QNTUNIT"
    variabs = ["PRODQNT", "EXPQNT", "IMPQNT", "QNTUNIT"]
    df = df.loc[df["indicators\\TIME_PERIOD"].isin(variabs), :]

    # keep only things in mapping for industry
    filepath = os.path.join(
        current_file_directory, "../data/eurostat/PRODCOM2024_PRODCOM2023_Table.csv"
    )
    df_map = pd.read_csv(filepath)
    df_map = df_map.loc[:, ["PRODCOM2024_KEY", "calc_industry_product"]]
    df_map = df_map.rename(columns={"PRODCOM2024_KEY": "prccode"})
    df_map = df_map.dropna()
    df = pd.merge(df, df_map, how="left", on=["prccode"])
    df_sub = df.loc[~df["calc_industry_product"].isnull(), :]
    df_sub = df_sub.loc[
        ~df_sub["calc_industry_product"].isin(["battery"]), :
    ]  # drop battery for now

    # fix countries
    # sources:
    # DECL drop down menu: https://ec.europa.eu/eurostat/databrowser/view/DS-056120/legacyMultiFreq/table?lang=en
    # https://ec.europa.eu/eurostat/documents/120432/0/Quick+guide+on+accessing+PRODCOM+data+DS-056120.pdf/484b8bbf-e371-49f3-6fa7-6a2514ebfcc9?t=1696602916356
    decl_mapping = {
        1: "France",
        3: "Netherlands",
        4: "Germany",
        5: "Italy",
        6: "United Kingdom",
        7: "Ireland",
        8: "Denmark",
        9: "Greece",
        10: "Portugal",
        11: "Spain",
        17: "Belgium",
        18: "Luxembourg",
        24: "Iceland",
        28: "Norway",
        30: "Sweden",
        32: "Finland",
        38: "Austria",
        46: "Malta",
        52: "Turkiye",
        53: "Estonia",
        54: "Latvia",
        55: "Lithuania",
        60: "Poland",
        61: "Czech Republic",
        63: "Slovakia",
        64: "Hungary",
        66: "Romania",
        68: "Bulgaria",
        70: "Albania",
        91: "Slovenia",
        92: "Croatia",
        93: "Bosnia and Herzegovina",
        96: "North Macedonia",
        97: "Montenegro",
        98: "Serbia",
        600: "Cyprus",
        1110: "EU15",
        1111: "EU25",
        1112: "EU27_2007",
        2027: "EU27_2020",
        2028: "EU28",
    }
    df_sub["country"] = np.nan
    for key in decl_mapping.keys():
        df_sub.loc[df_sub["decl"] == key, "country"] = decl_mapping[key]

    # make long format
    df_sub.rename(columns={"indicators\\TIME_PERIOD": "variable"}, inplace=True)
    df_sub_unit = df_sub.loc[df_sub["variable"].isin(["QNTUNIT"]), :]
    df_sub = df_sub.loc[~df_sub["variable"].isin(["QNTUNIT"]), :]
    drops = ["freq", "decl"]
    df_sub.drop(drops, axis=1, inplace=True)
    indexes = ["prccode", "variable", "country", "calc_industry_product"]
    df_sub = pd.melt(df_sub, id_vars=indexes, var_name="year")

    # make unit as column
    drops = ["freq", "decl"]
    df_sub_unit.drop(drops, axis=1, inplace=True)
    indexes = ["prccode", "variable", "country", "calc_industry_product"]
    df_sub_unit = pd.melt(df_sub_unit, id_vars=indexes, var_name="year")
    df_sub_unit.rename(columns={"value": "unit"}, inplace=True)
    keep = ["prccode", "country", "calc_industry_product", "year", "unit"]
    indexes = ["prccode", "country", "calc_industry_product", "year"]
    df_sub = pd.merge(df_sub, df_sub_unit.loc[:, keep], how="left", on=indexes)

    # fix unit
    df_sub["unit"].unique()
    old_unit = [
        np.nan,
        "kg ",
        "m2 ",
        "kg N ",
        "kg P2O5 ",
        "kg K2O ",
        "kg effect ",
        "p/st ",
        "ct/l ",
        "CGT ",
        "NA ",
    ]
    new_unit = [
        np.nan,
        "kg",
        "m2",
        "kg N",
        "kg P2O5",
        "kg K2O",
        "kg effect",
        "p/st",
        "ct/l",
        "CGT",
        "NA",
    ]
    for i in range(0, len(old_unit)):
        df_sub.loc[df_sub["unit"] == old_unit[i], "unit"] = new_unit[i]
    df_sub["unit"].unique()

    # fix value
    df_sub["value"] = [float(i) for i in df_sub["value"]]

    # order and sort
    indexes = ["country", "variable", "prccode", "calc_industry_product", "year"]
    variabs = ["value", "unit"]
    df_sub = df_sub.loc[:, indexes + variabs]
    df_sub = df_sub.sort_values(by=indexes)

    # check
    df_check = df_sub.loc[df_sub["prccode"] == "29102100", :]
    df_check = df_check.loc[df_sub["country"].isin(["Germany", "EU27_2020"])]
    # ok

    # aggregate by calc_industry_product
    df_sub = df_sub.reset_index()
    indexes = ["country", "variable", "calc_industry_product", "year", "unit"]
    df_sub = df_sub.groupby(indexes, as_index=False)["value"].agg(sum)

    # keep right units
    # df_sub["calc_industry_product"].unique()
    # df_sub["unit"].unique()
    # df_sub.loc[df_sub["calc_industry_product"].isin(["HDVH_ICE-diesel"]),"unit"].unique()
    # ["aluminium-pack","glass-pack", "paper-pack", "paper-print", "paper-san", "plastic-pack"]
    # product_check = ["plastic-pack"]
    # df_check = df_sub.loc[df_sub["calc_industry_product"].isin(product_check),:]
    # df_check = df_check.loc[df_check["country"].isin(["EU27_2020"]),:]
    units_dict = {
        "HDVH_ICE-diesel": ["p/st"],
        "HDVL_ICE-diesel": ["p/st"],
        "HDVM_ICE-diesel": ["p/st"],
        "HDV_BEV": ["p/st"],
        "HDV_ICE-diesel": ["p/st"],
        "HDV_ICE-gasoline": ["p/st"],
        "HDV_PHEV-diesel": ["p/st"],
        "LDV_BEV": ["p/st"],
        "LDV_ICE-diesel": ["p/st"],
        "LDV_ICE-gasoline": ["p/st"],
        "LDV_PHEV-gasoline": ["p/st"],
        "aluminium-pack": ["p/st"],
        "bus_ICE-diesel": ["p/st"],
        "computer": ["p/st"],
        "dishwasher": ["p/st"],
        "fertilizer": ["kg", "kg K2O", "kg N", "kg P2O5", "kg effect"],
        "freezer": ["p/st"],
        "fridge": ["p/st"],
        "glass-pack": ["kg"],
        "paper-pack": ["kg"],
        "paper-print": ["kg"],
        "paper-san": ["kg"],
        "phone": ["p/st"],
        "aviation_ICE": ["p/st"],
        "plastic-pack": ["kg"],
        "rail_CEV": ["p/st"],
        "rail_ICE-diesel": ["p/st"],
        "marine_ICE-diesel": ["p/st"],
        "tv": ["p/st"],
        "wmachine": ["p/st"],
        "2W_ICE-gasoline": ["p/st"],
    }
    df_sub = pd.concat(
        [
            df_sub.loc[
                (df_sub["calc_industry_product"] == key)
                & (df_sub["unit"].isin(units_dict[key])),
                :,
            ]
            for key in units_dict.keys()
        ]
    )
    df_sub.loc[df_sub["unit"] == "p/st", "unit"] = "num"

    # groupby for fertilizer
    df_fert = df_sub.loc[df_sub["calc_industry_product"] == "fertilizer", :]
    indexes = ["country", "variable", "calc_industry_product", "year"]
    df_fert = df_fert.groupby(indexes, as_index=False)["value"].agg(sum)
    df_fert["unit"] = "kg"
    df_sub = df_sub.loc[~df_sub["calc_industry_product"].isin(["fertilizer"]), :]
    df_sub = pd.concat([df_sub, df_fert])

    # fix countries
    countries_calc = [
        "Austria",
        "Belgium",
        "Bulgaria",
        "Croatia",
        "Cyprus",
        "Czech Republic",
        "Denmark",
        "EU27",
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
        "United Kingdom",
    ]
    df_sub["country"].unique()
    drops = [
        "Albania",
        "Bosnia and Herzegovina",
        "EU15",
        "EU28",
        "Iceland",
        "Montenegro",
        "North Macedonia",
        "Norway",
        "Serbia",
        "Turkiye",
    ]
    df_sub = df_sub.loc[~df_sub["country"].isin(drops), :]
    countries = df_sub["country"].unique()

    # I assume that trade extra eu is from the data EU27_2020
    df_sub = df_sub.loc[~df_sub["country"].isin(["EU27_2007"]), :]
    df_sub.loc[df_sub["country"] == "EU27_2020", "country"] = "EU27"
    df_temp = df_sub.loc[df_sub["country"] == "EU27", :]
    countries = df_sub["country"].unique()

    ##################################
    ##### CONVERT TO DATA MATRIX #####
    ##################################

    # make df ready for conversion to dm
    df_sub.loc[df_sub["variable"] == "IMPQNT", "variable"] = "product-import"
    df_sub.loc[df_sub["variable"] == "EXPQNT", "variable"] = "product-export"
    df_sub.loc[df_sub["variable"] == "PRODQNT", "variable"] = "product-demand"
    df_sub["variable"] = (
        df_sub["variable"]
        + "_"
        + df_sub["calc_industry_product"]
        + "["
        + df_sub["unit"]
        + "]"
    )
    df_sub = df_sub.rename(columns={"country": "Country", "year": "Years"})
    drops = ["calc_industry_product", "unit"]
    df_sub.drop(drops, axis=1, inplace=True)
    countries = df_sub["Country"].unique()
    years = df_sub["Years"].unique()
    variables = df_sub["variable"].unique()
    panel_countries = np.repeat(countries, len(variables) * len(years))
    panel_years = np.tile(np.tile(years, len(variables)), len(countries))
    panel_variables = np.tile(np.repeat(variables, len(years)), len(countries))
    df_temp = pd.DataFrame(
        {"Country": panel_countries, "Years": panel_years, "variable": panel_variables}
    )
    df_sub = pd.merge(df_temp, df_sub, how="left", on=["Country", "Years", "variable"])

    # put nan where is 0
    df_sub.loc[df_sub["value"] == 0, "value"] = np.nan

    # split in dms
    import re

    df_sub["selection"] = [
        re.split(
            "product-demand_|product-export_|product-import_", re.split("\\[", i)[0]
        )[1]
        for i in df_sub["variable"]
    ]

    # dm_bld_domapp
    df_temp = df_sub.loc[
        df_sub["selection"].isin(
            ["computer", "dishwasher", "freezer", "fridge", "phone", "tv", "wmachine"]
        ),
        ["Country", "Years", "variable", "value"],
    ]
    df_temp = df_temp.pivot(
        index=["Country", "Years"], columns="variable", values="value"
    ).reset_index()
    dm_bld_domapp = DataMatrix.create_from_df(df_temp, 1)

    # dm_tra_veh
    df_temp = df_sub.loc[
        df_sub["selection"].isin(
            [
                "HDVH_ICE-diesel",
                "HDVL_ICE-diesel",
                "HDVM_ICE-diesel",
                "HDV_BEV",
                "HDV_ICE-diesel",
                "HDV_ICE-gasoline",
                "HDV_PHEV-diesel",
                "LDV_BEV",
                "LDV_ICE-diesel",
                "LDV_ICE-gasoline",
                "LDV_PHEV-gasoline",
                "aviation_ICE",
                "bus_ICE-diesel",
                "marine_ICE-diesel",
                "rail_CEV",
                "rail_ICE-diesel",
                "2W_ICE-gasoline",
            ]
        ),
        ["Country", "Years", "variable", "value"],
    ]
    df_temp = df_temp.pivot(
        index=["Country", "Years"], columns="variable", values="value"
    ).reset_index()
    dm_tra_veh = DataMatrix.create_from_df(df_temp, 2)
    dm_tra_veh = dm_tra_veh.flatten()
    dm_tra_veh.groupby(
        {
            "HDV_ICE-diesel": [
                "HDVH_ICE-diesel",
                "HDVM_ICE-diesel",
                "HDVL_ICE-diesel",
                "HDV_ICE-diesel",
            ]
        },
        "Categories1",
        inplace=True,
    )
    idx = dm_tra_veh.idx
    for y in range(1995, 2002 + 1):
        dm_tra_veh.array[:, idx[y], :, idx["HDV_ICE-diesel"]] = np.nan
    dm_tra_veh.deepen()

    # dm_pack
    df_temp = df_sub.loc[
        df_sub["selection"].isin(
            ["glass-pack", "plastic-pack", "paper-pack", "paper-print", "paper-san"]
        ),
        ["Country", "Years", "variable", "value"],
    ]
    df_temp = df_temp.pivot(
        index=["Country", "Years"], columns="variable", values="value"
    ).reset_index()
    dm_pack_kg = DataMatrix.create_from_df(df_temp, 1)
    df_temp = df_sub.loc[
        df_sub["selection"].isin(["aluminium-pack"]),
        ["Country", "Years", "variable", "value"],
    ]
    df_temp = df_temp.pivot(
        index=["Country", "Years"], columns="variable", values="value"
    ).reset_index()
    dm_pack_unit = DataMatrix.create_from_df(df_temp, 1)

    # dm_fert
    df_temp = df_sub.loc[
        df_sub["selection"].isin(["fertilizer"]),
        ["Country", "Years", "variable", "value"],
    ]
    df_temp = df_temp.pivot(
        index=["Country", "Years"], columns="variable", values="value"
    ).reset_index()
    dm_fert_kg = DataMatrix.create_from_df(df_temp, 1)

    # put together
    dm_trade = dm_bld_domapp.flatten()
    dm_trade.append(dm_tra_veh.flatten().flatten(), "Variables")
    dm_trade.append(dm_pack_kg.flatten(), "Variables")
    dm_trade.append(dm_pack_unit.flatten(), "Variables")
    dm_trade.append(dm_fert_kg.flatten(), "Variables")
    dm_trade.sort("Variables")

    # check
    # dm_trade.filter({"Country" : ["EU27"]}).datamatrix_plot()

    return dm_trade
