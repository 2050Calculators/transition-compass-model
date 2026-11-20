import os

import eurostat
import numpy as np
import pandas as pd

from transition_compass_model.model.common.data_matrix_class import DataMatrix


def get_waste_vehicle_data(current_file_directory):
    # get data — fetch from Eurostat once and cache locally as CSV
    filepath = os.path.join(current_file_directory, "../data/eurostat/env_waselv.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        df = eurostat.get_data_df("env_waselv")
        df.to_csv(filepath, index=False)

    filepath = os.path.join(current_file_directory, "../data/eurostat/env_waselvt.csv")
    if os.path.exists(filepath):
        df_total = pd.read_csv(filepath)
    else:
        df_total = eurostat.get_data_df("env_waselvt")
        df_total.to_csv(filepath, index=False)

    # get geo column
    df.rename(columns={"geo\\TIME_PERIOD": "geoscale"}, inplace=True)
    df_total.rename(columns={"geo\\TIME_PERIOD": "geoscale"}, inplace=True)

    # filter for unit of measure: Tonne
    df = df.loc[df["unit"] == "T", :]
    df_total = df_total.loc[df_total["unit"] == "T", :]

    # checks
    A = df.copy()
    A["combo"] = [i + " - " + j for i, j in zip(df["wst_oper"], df["waste"])]
    list(A["combo"].unique())
    df_temp = df.copy()
    df_temp = df_temp.loc[df_temp["geoscale"] == "AT", :]
    df_temp = df_temp.loc[:, ["freq", "wst_oper", "waste", "unit", "geoscale", "2022"]]
    df_temp = df_temp.loc[df_temp["wst_oper"] == "REU", :]
    df_temp_tot = df_total.copy()
    df_temp_tot = df_temp_tot.loc[df_temp_tot["geoscale"] == "AT", :]
    df_temp_tot = df_temp_tot.loc[:, ["freq", "wst_oper", "unit", "geoscale", "2022"]]
    df_temp_tot = df_temp_tot.loc[df_temp_tot["wst_oper"] == "RCY", :]
    # RCY-ENV is total recycled
    # ENV is always TOTAL
    df_temp = df.copy()
    df_temp = df_temp.loc[df_temp["geoscale"] == "AT", :]
    df_temp = df_temp.loc[:, ["freq", "wst_oper", "waste", "unit", "geoscale", "2022"]]
    df_temp = df_temp.loc[df_temp["wst_oper"] == "RCV_E", :]
    # W1910 = W191001 + W191002 + W1910A + W1910B

    # in general for us:
    # total = littered + exported + collected + uncollected
    # collected = recycling + energy recovery + reuse + landfill + incineration

    # in general in eurostat
    # Waste generation: The quantity of waste, whereby ‘waste’ means any substance or object which the holder discards or intends or is required to discard.
    # Waste management: Waste management refers to the collection, transport, recovery (including sorting), and disposal of waste.
    # Treatment: Treatment means recovery or disposal operations, including preparation prior to recovery or disposal.
    # Recovery: Recovery means any operation whose main result is that waste serves a useful purpose. For example, by replacing other materials that would have been used for a particular function.
    # Disposal: Disposal means any operation that is not recovery (personal note: I guess this is either incineration or landfill)

    # specifically for env_waselv
    # in "wst_oper":  disposed (DSP), generated (GEN), recovered (RCV), energy recovery (RCV-E), recycled (RCY),
    # reused (REU)
    # in "waste":
    # dismantling and de-pollution (DMDP),
    # exported (EXP),
    # liquids (LIQ),
    # end of life vehicles (ELV),
    # End-of-life vehicles: tyres (W160103),
    # End-of-life vehicles: oil filters (W160107),
    # End-of-life vehicles: other materials arising from depollution (excluding fuel) (W1601A),
    # End-of-life vehicles: metal components (LoW: 160117+160118) (W1601B),
    # End-of-life vehicles: large plastic parts (W160119),
    # End-of-life vehicles: glass (W160120),
    # Batteries and accumulators (W1606),
    # Catalysts (W1608),
    # Total shredding (W1910),
    # Ferrous scrap (steel) from shredding (W191001),
    # Non-ferrous materials (aluminium, copper, zinc, lead, etc.) from shredding (W191002),
    # Shredder Light Fraction (SLF) (W1910A),
    # Other materials arising from shredding (W1910B)

    # so, formulas:
    # littered: 0
    # exported: among “waste”, EXP
    # collected: recycling + energy recovery + reuse + landfill + incineration
    # uncollected: collected / 0.8 * 0.2
    # recycling: among “wst_oper”, RCY
    # energy recovery: among “wst_oper”, RCV-E
    # reuse: among “wst_oper”, REU
    # landfill: among “waste” and “wst_oper”, DSP
    # incineration: 0

    # mapping
    dict_mapping = {
        "recycling": ["RCY - DMDP", "RCY - W1910"],
        "energy-recovery": ["RCV_E - DMDP", "RCV_E - W1910"],
        "reuse": ["REU - DMDP", "REU - W1910"],
        "landfill": ["DSP - DMDP", "DSP - W1910"],
        "export": ["GEN - EXP"],
    }

    # make long format
    indexes = ["freq", "wst_oper", "waste", "unit", "geoscale"]
    df = pd.melt(df, id_vars=indexes, var_name="year")

    # create column with combos
    df["combo"] = [i + " - " + j for i, j in zip(df["wst_oper"], df["waste"])]

    # aggregate
    indexes = ["freq", "wst_oper", "unit", "geoscale", "year"]
    key = "energy-recovery"

    def my_aggregation(key, df, dict_mapping, indexes):
        df_temp = df.loc[df["combo"].isin(dict_mapping[key]), :]
        df_temp = df_temp.groupby(indexes, as_index=False)["value"].agg(sum)
        df_temp["variable"] = key
        return df_temp

    df_elv = pd.concat(
        [my_aggregation(key, df, dict_mapping, indexes) for key in dict_mapping.keys()]
    )

    # select countries
    df_elv = df_elv.loc[:, ["geoscale", "year", "variable", "unit", "value"]]
    country_list = {
        "AT": "Austria",
        "BE": "Belgium",
        "BG": "Bulgaria",
        "HR": "Croatia",
        "CY": "Cyprus",
        "CZ": "Czech Republic",
        "DK": "Denmark",
        "EE": "Estonia",
        "EU27_2020": "EU27",
        "FI": "Finland",
        "FR": "France",
        "DE": "Germany",
        "EL": "Greece",
        "HU": "Hungary",
        "IE": "Ireland",
        "IT": "Italy",
        "LV": "Latvia",
        "LT": "Lithuania",
        "LU": "Luxembourg",
        "MT": "Malta",
        "NL": "Netherlands",
        "PL": "Poland",
        "PT": "Portugal",
        "RO": "Romania",
        "SK": "Slovakia",
        "SI": "Slovenia",
        "ES": "Spain",
        "SE": "Sweden",
    }
    for c in country_list.keys():
        df_elv.loc[df_elv["geoscale"] == c, "geoscale"] = country_list[c]
    drops = ["IS", "LI", "NO"]
    df_elv = df_elv.loc[~df_elv["geoscale"].isin(drops), :]
    len(df_elv["geoscale"].unique())  # note that we are missing UK, to do at the end

    # make EU27
    df_temp = df_elv.loc[
        df_elv["geoscale"] == "EU27", :
    ]  # for the moment I will keep this export, but it has only 1 value
    df_temp = df.loc[
        df["geoscale"] == "EU27_2020", :
    ]  # yes there are no other export values
    indexes = ["year", "variable", "unit"]
    countries = [
        "Austria",
        "Belgium",
        "Bulgaria",
        "Cyprus",
        "Czech Republic",
        "Germany",
        "Denmark",
        "Estonia",
        "Greece",
        "Spain",
        "Finland",
        "France",
        "Croatia",
        "Hungary",
        "Ireland",
        "Italy",
        "Lithuania",
        "Luxembourg",
        "Latvia",
        "Malta",
        "Netherlands",
        "Poland",
        "Portugal",
        "Romania",
        "Sweden",
        "Slovenia",
        "Slovakia",
    ]
    df_temp = df_elv.loc[df_elv["geoscale"].isin(countries), :]
    df_temp = df_temp.groupby(indexes, as_index=False)["value"].agg(sum)
    df_temp["geoscale"] = "EU27"
    df_temp = df_temp.loc[df_temp["variable"] != "export", :]
    df_elv = pd.concat([df_elv, df_temp])
    indexes = ["geoscale", "variable", "year", "unit"]
    df_elv.sort_values(indexes, inplace=True)

    # fix variable name with unit
    df_elv["variable"] = [i + "[t]" for i in df_elv["variable"]]
    df_elv.drop(columns="unit", inplace=True)

    # checks
    df_temp = df_elv.pivot(
        index=["geoscale", "year"], columns="variable", values="value"
    ).reset_index()

    # clean
    del (
        A,
        c,
        countries,
        country_list,
        df,
        df_temp,
        df_temp_tot,
        df_total,
        dict_mapping,
        drops,
        indexes,
        key,
    )

    ##################################
    ##### CONVERT TO DATA MATRIX #####
    ##################################

    # rename
    df_elv.rename(columns={"geoscale": "Country", "year": "Years"}, inplace=True)

    # put nan where is 0 (this is done to then generate values and replacing the zeroes)
    df_elv.loc[df_elv["value"] == 0, "value"] = np.nan

    # make dm
    df_temp = df_elv.copy()
    df_temp = df_temp.pivot(
        index=["Country", "Years"], columns="variable", values="value"
    ).reset_index()
    dm_elv = DataMatrix.create_from_df(df_temp, 0)

    # # plot
    # dm_elv.filter({"Country" : ["EU27"], "Variables" : ["reuse"]}).datamatrix_plot()
    # dm_elv.filter({"Country" : ["EU27"]}).datamatrix_plot()
    # landfill: probably consider until 2010 (when it's growing) for backward interpolation
    # df_temp = dm_elv.write_df()

    # clean
    del df_temp, df_elv

    return dm_elv


def get_waste_buildings_data(current_file_directory):
    # get data — fetch from Eurostat once and cache locally as CSV
    filepath = os.path.join(current_file_directory, "../data/eurostat/env_wastrt.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        df = eurostat.get_data_df("env_wastrt")
        df.to_csv(filepath, index=False)

    # get geo column
    df.rename(columns={"geo\\TIME_PERIOD": "geoscale"}, inplace=True)

    # filter for unit of measure: Tonne
    df = df.loc[df["unit"] == "T", :]

    # get only construction waste, both hazardous and non hazardous
    df = df.loc[df["waste"] == "W121", :]
    df = df.loc[df["hazard"] == "HAZ_NHAZ", :]
    df.drop(columns=["hazard"], inplace=True)

    # checks
    A = df.copy()
    A["combo"] = [i + " - " + j for i, j in zip(df["wst_oper"], df["waste"])]
    list(A["combo"].unique())

    # in general for us:
    # total = littered + exported + collected + uncollected
    # collected = recycling + energy recovery + reuse + landfill + incineration

    # in general in eurostat
    # Waste generation: The quantity of waste, whereby ‘waste’ means any substance or object which the holder discards or intends or is required to discard.
    # Waste management: Waste management refers to the collection, transport, recovery (including sorting), and disposal of waste.
    # Treatment: Treatment means recovery or disposal operations, including preparation prior to recovery or disposal.
    # Recovery: Recovery means any operation whose main result is that waste serves a useful purpose. For example, by replacing other materials that would have been used for a particular function.
    # Disposal: Disposal means any operation that is not recovery (personal note: I guess this is either incineration or landfill)

    # specifically for env_wastrt
    # 'DSP_I - W121': incineration (D10)
    # 'DSP_L - W121': landfill (D1, D5, D12)
    # 'DSP_L_OTH - W121': landfill and other (D1-D7, D12)
    # 'DSP_OTH - W121': other (D2-D4, D6-D7)
    # 'RCV_B - W121': backfilling
    # 'RCV_E - W121': energy recovery (R1)
    # 'RCV_R - W121': recycling
    # 'RCV_R_B - W121': recycling and backfilling (R2-R11)
    # 'TRT - W121': waste treatment

    # so, formulas:
    # littered: 0
    # exported: 0
    # collected: recycling + energy recovery + reuse (backfilling) + landfill + incineration
    # uncollected: 0
    # recycling: RCV_R - W121
    # energy recovery: RCV_E - W121
    # reuse: RCV_B - W121
    # landfill: DSP_L - W121
    # incineration: DSP_I - W121

    # mapping
    dict_mapping = {
        "recycling": ["RCV_R - W121"],
        "energy-recovery": ["RCV_E - W121"],
        "reuse": ["RCV_B - W121"],
        "landfill": ["DSP_L - W121"],
        "incineration": ["DSP_I - W121"],
    }

    # make long format
    indexes = ["freq", "wst_oper", "waste", "unit", "geoscale"]
    df = pd.melt(df, id_vars=indexes, var_name="year")

    # create column with combos
    df["combo"] = [i + " - " + j for i, j in zip(df["wst_oper"], df["waste"])]

    # aggregate
    indexes = ["freq", "wst_oper", "unit", "geoscale", "year"]
    key = "energy-recovery"

    def my_aggregation(key, df, dict_mapping, indexes):
        df_temp = df.loc[df["combo"].isin(dict_mapping[key]), :]
        df_temp = df_temp.groupby(indexes, as_index=False)["value"].agg(sum)
        df_temp["variable"] = key
        return df_temp

    df_bld = pd.concat(
        [my_aggregation(key, df, dict_mapping, indexes) for key in dict_mapping.keys()]
    )

    # select countries
    df_bld = df_bld.loc[:, ["geoscale", "year", "variable", "unit", "value"]]
    country_list = {
        "AT": "Austria",
        "BE": "Belgium",
        "BG": "Bulgaria",
        "HR": "Croatia",
        "CY": "Cyprus",
        "CZ": "Czech Republic",
        "DK": "Denmark",
        "EE": "Estonia",
        "EU27_2020": "EU27",
        "FI": "Finland",
        "FR": "France",
        "DE": "Germany",
        "EL": "Greece",
        "HU": "Hungary",
        "IE": "Ireland",
        "IT": "Italy",
        "LV": "Latvia",
        "LT": "Lithuania",
        "LU": "Luxembourg",
        "MT": "Malta",
        "NL": "Netherlands",
        "PL": "Poland",
        "PT": "Portugal",
        "RO": "Romania",
        "SK": "Slovakia",
        "SI": "Slovenia",
        "ES": "Spain",
        "SE": "Sweden",
        "UK": "United Kingdom",
    }
    for c in country_list.keys():
        df_bld.loc[df_bld["geoscale"] == c, "geoscale"] = country_list[c]
    drops = ["AL", "BA", "IS", "ME", "EU28", "MK", "LI", "NO", "RS", "TR", "XK"]
    df_bld = df_bld.loc[~df_bld["geoscale"].isin(drops), :]
    len(df_bld["geoscale"].unique())

    # fix variable name with unit
    df_bld["variable"] = [i + "[t]" for i in df_bld["variable"]]
    df_bld.drop(columns="unit", inplace=True)

    # checks
    df_temp = df_bld.pivot(
        index=["geoscale", "year"], columns="variable", values="value"
    ).reset_index()

    # drop before 2008 (all zero)
    df_bld = df_bld.loc[df_bld["year"] > "2008", :]

    # clean
    del A, c, country_list, df, df_temp, dict_mapping, drops, indexes, key

    ##################################
    ##### CONVERT TO DATA MATRIX #####
    ##################################

    # rename
    df_bld.rename(columns={"geoscale": "Country", "year": "Years"}, inplace=True)

    # # put nan where is 0 (this is done to then generate values and replacing the zeroes)
    # df_bld.loc[df_bld["value"] == 0,"value"] = np.nan

    # make dm
    df_temp = df_bld.copy()
    df_temp = df_temp.pivot(
        index=["Country", "Years"], columns="variable", values="value"
    ).reset_index()
    dm_bld = DataMatrix.create_from_df(df_temp, 0)

    # # plot
    # dm_bld.filter({"Country" : ["EU27"], "Variables" : ["recycling"]}).datamatrix_plot()
    # dm_bld.filter({"Country" : ["EU27"]}).datamatrix_plot()
    # landfill: probably consider until 2010 (when it's growing) for backward interpolation
    # df_temp = dm_bld.write_df()

    # clean
    del df_temp, df_bld

    return dm_bld
