import os

import numpy as np
import pandas as pd

from transition_compass_model.model.common.data_matrix_class import DataMatrix


def get_prodcom_data(file_name, current_file_directory):
    # get data
    # df = eurostat.get_data_df({file_name})
    filepath = os.path.join(current_file_directory, f"../data/eurostat/{file_name}.csv")
    # df.to_csv(filepath, index = False)
    df = pd.read_csv(filepath)

    # NOTE: as ds-056120 is on sold production, then we assume sold production = demand
    # import and export should be always on "sold", as a country exports what's demanded
    # and it imports what it demands.

    # get "PRODQNT", "EXPQNT", "IMPQNT", "QNTUNIT"
    variabs = ["PRODQNT", "EXPQNT", "IMPQNT", "QNTUNIT"]
    df = df.loc[df["indicators\\TIME_PERIOD"].isin(variabs), :]

    # apply mapping with our variable names
    filepath = os.path.join(
        current_file_directory, "../data/eurostat/PRODCOM2024_PRODCOM2023_Table.csv"
    )
    df_map = pd.read_csv(filepath)
    df_map = df_map.rename(columns={"PRODCOM2024_KEY": "prccode"})
    # Remap PRODCOM C23.99.13 (bituminous mixtures) from "ois" to "asphalt".
    # The CSV maps code 23991310 to "ois"; override here to avoid editing the shared CSV.
    df_map.loc[
        df_map["prccode"].astype(str).str.startswith("23991310"),
        "calc_industry_material",
    ] = "asphalt"
    df_map_sub = df_map.filter(
        items=["prccode", "calc_industry_material", "primary_material_flag"]
    )
    df_map_sub = df_map_sub.dropna()
    df_map_sub["calc_industry_material"].unique()
    materials = [
        "aluminium",
        "ammonia",
        "asphalt",
        "cement",
        "chem",
        "copper",
        "glass",
        "lime",
        "met",
        "nmm",
        "paper",
        "steel",
        "timber",
        "fbt",
        "mae",
        "ois",
        "other",
        "textiles",
        "tra-equip",
        "wwp",
    ]
    # note:
    # for (food, beverages and tobacco), machinery equipment (mae)
    # transport equipment (tra-equip), textiles and leather (textiles), wood and wood products (wwp),
    # and other industries (ois  ), I am getting total production = sold production (demand) + export - import
    # from ds-056120 rather than getting
    # directly total production from ds-056121 as ds-056121 has less data availability
    # these materials will enter as fxa. And I will also obtain total production for
    # the other materials, which can be used for calibration.
    df_map_sub = df_map_sub.loc[df_map_sub["calc_industry_material"].isin(materials), :]
    df_map_sub = df_map_sub.loc[
        df_map_sub["primary_material_flag"] == True, :
    ]  # this is done to avoid double counting (as for example some aluminium codes can be input of aluminium production)
    df = pd.merge(
        df,
        df_map_sub.loc[:, ["prccode", "calc_industry_material"]],
        how="left",
        on=["prccode"],
    )
    df_sub = df.loc[~df["calc_industry_material"].isnull(), :]

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
    indexes = ["prccode", "variable", "country", "calc_industry_material"]
    df_sub = pd.melt(df_sub, id_vars=indexes, var_name="year")

    # make unit as column
    drops = ["freq", "decl"]
    df_sub_unit.drop(drops, axis=1, inplace=True)
    indexes = ["prccode", "variable", "country", "calc_industry_material"]
    df_sub_unit = pd.melt(df_sub_unit, id_vars=indexes, var_name="year")
    df_sub_unit.rename(columns={"value": "unit"}, inplace=True)
    keep = ["prccode", "country", "calc_industry_material", "year", "unit"]
    indexes = ["prccode", "country", "calc_industry_material", "year"]
    df_sub = pd.merge(df_sub, df_sub_unit.loc[:, keep], how="left", on=indexes)

    # fix unit
    df_sub["unit"].unique()
    df_sub["calc_industry_material"].unique()
    old_unit = [
        "kg ",
        "p/st ",
        "l ",
        "l alc 100% ",
        "m3 ",
        "pa ",
        "m2 ",
        "m ",
        "kg 90% sdt ",
        "kg TiO2 ",
        "kg HCl ",
        "kg P2O5 ",
        "kg HF ",
        "kg SiO2 ",
        "kg SO2 ",
        "kg NaOH ",
        "kg Al2O3 ",
        "kg Cl ",
        "kg Na2S2O5 ",
        "kg Na2CO3 ",
        "kg B2O3 ",
        "kg H2O2 ",
        "g ",
        "kg N ",
        "kg act.subst ",
        "km ",
        "c/k ",
        "pa ",
        "kg act. subst. ",
        "kW ",
        "l alc. 100% ",
        "kg KOH ",
        "kg H2SO4 ",
        "NA ",
        "kg act.subst. ",
        "kg F ",
    ]
    new_unit = [
        "kg",
        "p/st",
        "l",
        "l alc 100%",
        "m3",
        "pa",
        "m2",
        "m",
        "kg 90% sdt",
        "kg TiO2",
        "kg HCl",
        "kg P2O5",
        "kg HF",
        "kg SiO2",
        "kg SO2",
        "kg NaOH",
        "kg Al2O3",
        "kg Cl",
        "kg Na2S2O5",
        "kg Na2CO3",
        "kg B2O3",
        "kg H2O2",
        "g",
        "kg N",
        "kg act.subst",
        "km",
        "c/k",
        "pa",
        "kg act. subst.",
        "kW",
        "l alc. 100%",
        "kg KOH",
        "kg H2SO4",
        "NA",
        "kg act.subst.",
        "kg F",
    ]
    for i in range(0, len(old_unit)):
        df_sub.loc[df_sub["unit"] == old_unit[i], "unit"] = new_unit[i]
    df_sub["unit"].unique()
    df_sub["calc_industry_material"].unique()

    # fix value
    df_sub["value"] = [float(i) for i in df_sub["value"]]

    # order and sort
    indexes = ["country", "variable", "prccode", "calc_industry_material", "year"]
    variabs = ["value", "unit"]
    df_sub = df_sub.loc[:, indexes + variabs]
    df_sub = df_sub.sort_values(by=indexes)

    # check
    df_check = df_sub.loc[df_sub["prccode"] == "24421130", :]
    df_check = df_check.loc[df_sub["country"].isin(["Germany", "EU27_2020"])]
    # ok

    # aggregate by calc_industry_material
    df_sub = df_sub.reset_index()
    indexes = ["country", "variable", "calc_industry_material", "year", "unit"]
    df_sub = df_sub.groupby(indexes, as_index=False)["value"].agg(sum)

    # keep right units
    df_sub["calc_industry_material"].unique()
    df_sub["unit"].unique()
    df_check = df_sub.loc[df_sub["calc_industry_material"] == "paper", :]
    df_check["unit"].unique()
    units_dict = {
        "aluminium": ["kg"],
        "ammonia": ["kg N"],
        "asphalt": ["kg"],  # PRODCOM C23.99.13 Bituminous mixtures reported in kg
        "cement": ["kg"],
        "copper": ["kg"],
        "glass": ["kg"],
        "lime": ["kg"],
        "mae": ["kg"],
        "met": ["kg"],  # NACE B07 metal ore concentrates reported in kg
        "ois": ["kg"],
        "other": ["kg"],
        "steel": ["kg"],
        "textiles": ["kg"],
        "tra-equip": ["kg"],
    }
    # NOTE: for large groups of materials, we consider only kg, but
    # we are missing other products in other categories (for example for ois, there are
    # things under 'm2', 'm3', 'p/st', 'pa', 'c/k', 'm', 'NA')
    df_sub_temp = pd.concat(
        [
            df_sub.loc[
                (df_sub["calc_industry_material"] == key)
                & (df_sub["unit"].isin(units_dict[key])),
                :,
            ]
            for key in units_dict.keys()
        ]
    )
    df_sub_temp.loc[df_sub_temp["unit"] == "kg N", "unit"] = "kg"

    # chem
    df_sub_chem = df_sub.loc[df_sub["calc_industry_material"].isin(["chem"]), :]
    df_sub_chem["unit"].unique()
    df_sub_chem.loc[df_sub_chem["unit"] == "g", "value"] = (
        df_sub_chem.loc[df_sub_chem["unit"] == "g", "value"] / 1000
    )
    df_sub_chem.loc[df_sub_chem["unit"] == "g", "unit"] = "kg"
    ls_temp = [
        "kg Al2O3",
        "kg B2O3",
        "kg F",
        "kg H2O2",
        "kg H2SO4",
        "kg HCl",
        "kg HF",
        "kg KOH",
        "kg N",
        "kg Na2CO3",
        "kg NaOH",
        "kg P2O5",
        "kg SO2",
        "kg SiO2",
        "kg TiO2 ",
        "kg act. subst.",
        "kg Cl",
        "kg Na2S2O5",
        "kg act.subst",
        "kg act.subst.",
    ]
    for l in ls_temp:
        df_sub_chem.loc[df_sub_chem["unit"] == l, "unit"] = "kg"
    indexes = ["country", "variable", "calc_industry_material", "year", "unit"]
    df_sub_chem = df_sub_chem.groupby(indexes, as_index=False)["value"].agg(sum)
    df_sub_chem = df_sub_chem.loc[df_sub_chem["unit"] == "kg", :]
    df_sub_temp = pd.concat([df_sub_temp, df_sub_chem])

    # timber
    # assumption: 1 m3 of wood weights 600 kg/m3 (between 0.55 t/m3 to 0.65 t/m3: https://www.fao.org/4/w4095e/w4095e06.htm#3.1.4%20examples%20of%20calculations%20of%20biomass%20density)
    df_sub_timber = df_sub.loc[df_sub["calc_industry_material"].isin(["timber"]), :]
    df_sub_timber["unit"].unique()
    df_sub_timber = df_sub_timber.loc[df_sub_timber["unit"] == "m3", :]
    df_sub_timber["value"] = df_sub_timber["value"] * 600
    df_sub_timber["unit"] = "kg"
    df_sub_temp = pd.concat([df_sub_temp, df_sub_timber])

    # fbt
    df_sub_fbt = df_sub.loc[df_sub["calc_industry_material"].isin(["fbt"]), :]
    df_sub_fbt["unit"].unique()
    # assumption: Pure ethanol (alcohol) has a density of approximately 0.789 kilograms per liter (kg/L) at room temperature (20°C)
    df_sub_fbt.loc[df_sub_fbt["unit"] == "l alc. 100%", "value"] = (
        df_sub_fbt.loc[df_sub_fbt["unit"] == "l alc. 100%", "value"] * 0.789
    )
    df_sub_fbt.loc[df_sub_fbt["unit"] == "l alc 100%", "value"] = (
        df_sub_fbt.loc[df_sub_fbt["unit"] == "l alc 100%", "value"] * 0.789
    )
    df_sub_fbt = df_sub_fbt.loc[
        df_sub_fbt["unit"].isin(["kg", "l", "l alc. 100%", "l alc 100%"]), :
    ]
    df_sub_fbt["unit"] = "kg"
    indexes = ["country", "variable", "calc_industry_material", "year", "unit"]
    df_sub_fbt = df_sub_fbt.groupby(indexes, as_index=False)["value"].agg(sum)
    df_sub_temp = pd.concat([df_sub_temp, df_sub_fbt])

    # wwp
    # PRODCOM primary_material_flag=True codes for wwp are: coniferous/non-coniferous wood chips
    # (C16.10.25.03/05, ~40 Mt EU27 2019) plus MDF, particle board, fibreboard, OSB (~3.6 Mt).
    # Construction sawn timber (C16.10.11-12) and builders' carpentry (C16.23) are already
    # primary_material_flag=False — they are tracked separately under 'timber'.
    # Wood chips are sawmill by-products; their energy is embedded in the sawmilling process
    # and is NOT separately attributed to wwp — the energy intensity denominator for wwp
    # uses FAOSTAT total C16 production (sawnwood + chips + boards) to avoid inflating the
    # per-tonne intensity. See data_fxa_energy_demand.py wwp section.
    # assumption: 1 m3 of wood weights 600 kg/m3 (between 0.55 t/m3 to 0.65 t/m3: https://www.fao.org/4/w4095e/w4095e06.htm#3.1.4%20examples%20of%20calculations%20of%20biomass%20density)
    df_sub_wwp = df_sub.loc[df_sub["calc_industry_material"].isin(["wwp"]), :]
    df_sub_wwp["unit"].unique()
    df_sub_wwp.loc[df_sub_wwp["unit"] == "m3", "value"] = (
        df_sub_wwp.loc[df_sub_wwp["unit"] == "m3", "value"] * 600
    )
    df_sub_wwp = df_sub_wwp.loc[df_sub_wwp["unit"].isin(["kg", "m3"]), :]
    df_sub_wwp.loc[df_sub_wwp["unit"] == "m3", "unit"] = "kg"
    indexes = ["country", "variable", "calc_industry_material", "year", "unit"]
    df_sub_wwp = df_sub_wwp.groupby(indexes, as_index=False)["value"].agg(sum)
    df_sub_temp = pd.concat([df_sub_temp, df_sub_wwp])

    # nmm (non-metallic minerals: aggregate — NACE B08 sand/gravel/crushed stone)
    # PRODCOM B08 codes may report in kg or m3; convert m3 using bulk density 1500 kg/m3.
    # Source bulk density: typical loose aggregate 1400–1600 kg/m3 (EN 1097-3).
    df_sub_nmm = df_sub.loc[df_sub["calc_industry_material"].isin(["nmm"]), :]
    df_sub_nmm.loc[df_sub_nmm["unit"] == "m3", "value"] = (
        df_sub_nmm.loc[df_sub_nmm["unit"] == "m3", "value"] * 1500
    )
    df_sub_nmm = df_sub_nmm.loc[df_sub_nmm["unit"].isin(["kg", "m3"]), :]
    df_sub_nmm.loc[df_sub_nmm["unit"] == "m3", "unit"] = "kg"
    indexes = ["country", "variable", "calc_industry_material", "year", "unit"]
    df_sub_nmm = df_sub_nmm.groupby(indexes, as_index=False)["value"].agg(sum)
    df_sub_temp = pd.concat([df_sub_temp, df_sub_nmm])

    # paper
    df_sub_paper = df_sub.loc[df_sub["calc_industry_material"].isin(["paper"]), :]
    df_sub_paper["unit"].unique()
    df_sub_paper.loc[df_sub_paper["unit"] == "kg 90% sdt", "unit"] = "kg"
    df_sub_paper = df_sub_paper.groupby(indexes, as_index=False)["value"].agg(sum)
    df_sub = pd.concat([df_sub_temp, df_sub_paper])

    # sort
    indexes = ["country", "variable", "calc_industry_material", "year", "unit"]
    df_sub.sort_values(by=indexes, inplace=True)

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
        + df_sub["calc_industry_material"]
        + "["
        + df_sub["unit"]
        + "]"
    )
    df_sub = df_sub.rename(columns={"country": "Country", "year": "Years"})
    drops = ["calc_industry_material", "unit"]
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

    # make dm
    df_temp = df_sub.pivot(
        index=["Country", "Years"], columns="variable", values="value"
    ).reset_index()
    dm_mat = DataMatrix.create_from_df(df_temp, 1)
    # dm_mat.datamatrix_plot(selected_cols={"Country" : ["EU27"],
    #                                       "Variables" : ["product-export","product-import","product-demand"],
    #                                       "Categories1" : ["timber"]})
    # note: all values before 2006 will be put to nan and re-built, and timber has weird jump in 2022 for import and demand,
    # so can be generated before

    # fix names
    dm_mat.rename_col_regex("product", "material", "Variables")

    # check
    # dm_mat.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

    return dm_mat


def get_prodcom_cement_production(current_file_directory):
    """Return dict {year: total_cement_kt} for EU27 from PRODCOM.

    Uses CPA 23511210 (Portland cement) + 23511290 (other hydraulic cements).
    Excludes clinker (23511100, intermediate product) and concrete products (C23.6x).
    Source: https://ec.europa.eu/eurostat/web/prodcom
    """
    filepath = os.path.join(current_file_directory, "../data/eurostat/ds-056120.csv")
    df = pd.read_csv(filepath, low_memory=False)
    cement_codes = ["23511210", "23511290"]
    df_eu = df[
        (df["decl"] == 2027)
        & (df["indicators\\TIME_PERIOD"] == "PRODQNT")
        & (df["prccode"].astype(str).isin(cement_codes))
    ].copy()
    result = {}
    for col in df_eu.columns:
        try:
            yr = int(col)
        except ValueError:
            continue
        val_kg = pd.to_numeric(df_eu[col], errors="coerce").sum()
        if pd.notna(val_kg) and val_kg > 0:
            result[yr] = val_kg / 1e6  # kg → kt
    return result


def get_faostat_sawnwood_data(current_file_directory):
    """Return dict {year: total_sawnwood_Mt} for European Union (27).

    Sums coniferous and non-coniferous sawnwood from FAOSTAT FO.
    Source: https://www.fao.org/faostat/en/#data/FO (Forestry Production and Trade)
    Unit conversion: m3 × 600 kg/m3 → Mt
    https://www.fao.org/4/w4095e/w4095e06.htm#3.1.4%20examples%20of%20calculations%20of%20biomass%20density
    """
    filepath = os.path.join(
        current_file_directory, "../data/FAO/FAOSTAT_data_en_11-12-2026.csv"
    )
    df = pd.read_csv(filepath)
    df = df.loc[df["Area"] == "European Union (27)", ["Year", "Value"]]
    sawnwood_by_year = df.groupby("Year")["Value"].sum()  # m3
    return {
        int(yr): float(val) * 600 / 1e9 for yr, val in sawnwood_by_year.items()
    }  # Mt
