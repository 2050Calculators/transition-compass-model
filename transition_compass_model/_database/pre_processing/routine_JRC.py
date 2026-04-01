import pandas as pd
import numpy as np
import re
import os
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def get_jrc_data(
    dict_extract,
    dict_countries,
    current_file_directory,
    years=list(range(2000, 2021 + 1)),
    levels_to_industry_preproc="../../../Industry",
):

    sub_variables = dict_extract["sub_variables"]
    calc_names = dict_extract["calc_names"]

    # get dataframe by country

    def get_dataframe(dict_extract, country, sub_variables):

        # store things
        database = dict_extract["database"]
        sheet = dict_extract["sheet"]
        variable = dict_extract["variable"]
        sheet_last_row = dict_extract["sheet_last_row"]
        if "categories" in list(dict_extract.keys()):
            categories = dict_extract["categories"]
        else:
            categories = None

        # define function to search
        def my_search(search, x):
            if x is np.nan:
                return False
            else:
                return bool(re.search(search, x, re.IGNORECASE))

        # get data
        filepath_jrc = os.path.join(
            current_file_directory,
            levels_to_industry_preproc
            + "/eu/data/JRC-IDEES-2021/"
            + country
            + "/JRC-IDEES-2021_"
            + database
            + "_"
            + country
            + ".xlsx",
        )
        df = pd.read_excel(filepath_jrc, sheet)
        id_var = df.columns[0]
        df.rename(columns={id_var: "variable"}, inplace=True)

        # subset data
        def get_indexes(vector, string):
            variable = re.sub(
                r"([\(\)])", r"\\\1", string
            )  # this is to accept parentheses in the string
            bool_temp = [bool(my_search(variable, x)) for x in vector]
            idx_temp = [i for i, x in enumerate(bool_temp) if x][0]
            return idx_temp

        string_first = get_indexes(df.loc[:, "variable"], variable)
        df_temp = df.iloc[range(string_first, len(df)), :]
        if categories is not None:
            string_intermediate = get_indexes(df_temp.loc[:, "variable"], categories)
            df_temp = df_temp.iloc[range(string_intermediate, len(df_temp)), :]
        string_second = get_indexes(df_temp.loc[:, "variable"], sheet_last_row)
        df_temp = df_temp.iloc[range(0, string_second + 1), :]
        df_temp = df_temp.loc[df_temp["variable"].isin(sub_variables), :]
        df_temp = df_temp.loc[:, ["variable"] + years]

        # add unit to variables names
        if len(variable.split("(")) == 1:
            df_temp["variable"] = [
                re.sub(r"$", "[unit]", text) for text in df_temp["variable"]
            ]
        else:
            unit = variable.split("(")[1].split(")")[0]
            df_temp["variable"] = [
                re.sub(r"$", "[" + unit + "]", text) for text in df_temp["variable"]
            ]

        # reshape data
        df_temp = pd.melt(df_temp, id_vars="variable", var_name="Years")
        df_temp["Country"] = country
        df_temp = df_temp.pivot(
            index=["Country", "Years"], columns="variable", values="value"
        ).reset_index()

        return df_temp

    df_all = pd.concat(
        [
            get_dataframe(dict_extract, country, sub_variables)
            for country in dict_countries.keys()
        ]
    )

    # make data matrix
    dm = DataMatrix.create_from_df(df_all, 0)

    # change country names
    dict_temp = dict_countries.copy()
    if "EU27" in list(dict_temp.keys()):
        dict_temp.pop("EU27")
    for key in dict_temp.keys():
        dm.rename_col(key, dict_temp[key], "Country")

    # rename variables and aggregate
    for i in range(0, len(sub_variables)):
        dm.rename_col(sub_variables[i], calc_names[i], "Variables")

    # return
    return dm


def get_names_map_for_transport():

    names_map = {
        "INTMARB": "marine-international",
        "INTAVI_PA_IN": "aviation-passenger-intra",
        "INTAVI_PA_EX": "aviation-passenger-extra",
        "INTAVI_FR_IN": "aviation-freight-intra",
        "INTAVI_FR_EX": "aviation-freight-extra",
        "FC_TRA_RAIL_MTU_E": "metrotram",
        "FC_TRA_RAIL_CPT_E": "rail-passenger-normal",
        "FC_TRA_RAIL_HST_E": "rail-passenger-high",
        "FC_TRA_RAIL_FRT_E": "rail-freight",
        "FC_TRA_ROAD_P2W_E": "2W",
        "FC_TRA_ROAD_CAR_E": "LDV",
        "FC_TRA_ROAD_BUS_E": "bus",
        "FC_TRA_ROAD_LCV_E": "trucks-lm",
        "FC_TRA_ROAD_HGV_E": "trucks-h",
        "FC_TRA_DAVI_PA_E": "aviation-passenger-domestic",
        "FC_TRA_DAVI_FR_E": "aviation-freight-domestic",
        "FC_TRA_DNAVI_DCS_E": "marine-domestic",
        "FC_TRA_DNAVI_IWW_E": "IWW",
    }

    return names_map


def get_mapping_fuels_for_transport():

    jrc_to_model_fuels = {
        # --- BIOFUELS ---
        "biodiesel": [
            "R5220P",  # Pure biodiesels
            "R5220B",  # Blended biodiesels (bio portion)
        ],
        "biogas": ["R5300"],  # Biogases
        "biogasoline": [
            "R5210P",  # Pure biogasoline
            "R5210B",  # Blended biogasoline (bio portion)
            "R5290",  # Other liquid biofuels (assigned here by convention)
        ],
        "kerosenebio": [
            "R5230P",  # Pure bio jet kerosene
            "R5230B",  # Blended bio jet kerosene (bio portion)
        ],
        # --- FOSSIL LIQUID FUELS ---
        "diesel": [
            "O4671XR5220B",  # Gas oil and diesel oil (excluding biofuel portion)
            "O4680",  # Fuel oil
        ],
        "gasoline": [
            "O4652XR5210B",  # Motor gasoline (excluding biofuel portion)
            "O4669",  # Other kerosene
            "O4691",  # White spirit & SBP spirits
            "O4640",  # Naphtha
        ],
        "kerosene": [
            "O4661XR5230B",  # Kerosene-type jet fuel (excluding biofuel portion)
            "O4653",  # Gasoline-type jet fuel
            "O4651",  # Aviation gasoline
        ],
        # --- GASES ---
        "gas": [
            "G3000",  # Natural gas
            "O4630",  # Liquefied petroleum gases (LPG)
            "O4610",  # Refinery gas
        ],
        # --- ELECTRICITY ---
        "electricity": ["E7000"],
        # --- EVERYTHING ELSE (non-transport, upstream, aggregates, residuals) ---
        "other": [
            # --- Solid fossil fuels ---
            "C0000X0350-0370",
            "C0100",
            "C0110",
            "C0121",
            "C0129",
            "C0200",
            "C0210",
            "C0220",
            "C0300",
            "C0311",
            "C0312",
            "C0320",
            "C0330",
            "C0340",
            "C0350",
            "C0350-0370",
            "C0360",
            "C0371",
            "C0379",
            # --- Peat ---
            "P1000",
            "P1100",
            "P1200",
            # --- Oil shale ---
            "S2000",
            # --- Upstream oil / aggregates ---
            "O4000XBIO",
            "O4100_TOT",
            "O4100_TOT_4200-4500XBIO",
            "O4200",
            "O4300",
            "O4400X4410",
            "O4500",
            "O4600XBIO",
            # --- Minor oil products not used as fuels ---
            "O4620",  # Ethane
            "O4692",  # Lubricants
            "O4693",  # Paraffin waxes
            "O4694",  # Petroleum coke
            "O4695",  # Bitumen
            "O4699",  # Other oil products
            # --- Renewables aggregates / non-specified ---
            "RA000",
            "RA100",
            "RA200",
            "RA300",
            "RA410",
            "RA420",
            "RA500",
            "RA600",
            "R5110-5150_W6000RI",  # Primary solid biofuels
            "R5160",  # Charcoal
            "W6210",  # Renewable municipal waste
            # --- Waste ---
            "W6100",
            "W6100_6220",
            "W6220",
            # --- Nuclear & heat ---
            "N900H",
            "H8000",
        ],
    }

    return jrc_to_model_fuels


def get_jrc_balance_data_for_transport(
    current_file_directory,
    country_code,
    country_name,
    unit,
    variable,
    database,
    sheet_name,
    names_map,
    jrc_to_model_fuels,
):

    # get dm
    filepath_jrc = os.path.join(
        current_file_directory,
        f"../../../industry/eu/data/JRC-IDEES-2021/{country_code}/{database}_{country_code}.xlsx",
    )
    df_temp = pd.read_excel(filepath_jrc, sheet_name)
    df_temp = df_temp.iloc[1:, 1:]
    df_temp = df_temp.dropna()
    indexes = df_temp.columns[0]
    df_temp = pd.melt(df_temp, id_vars=indexes, var_name="year")
    df_temp.columns = ["variable", "Years", "value"]
    df_temp["variable"] = df_temp["variable"] + "[" + unit + "]"
    df_temp = df_temp.pivot(
        index=["Years"], columns="variable", values="value"
    ).reset_index()
    df_temp["Country"] = country_name
    dm_temp = DataMatrix.create_from_df(df_temp, 0)

    # aggregate
    jrc_to_model_fuels_temp = jrc_to_model_fuels.copy()
    categs = dm_temp.col_labels["Variables"]
    for key in jrc_to_model_fuels_temp.keys():
        list_temp = jrc_to_model_fuels_temp[key].copy()
        list_temp = np.array(list_temp)[[l in categs for l in list_temp]].tolist()
        jrc_to_model_fuels_temp[key] = list_temp
    jrc_to_model_fuels_temp = {k: v for k, v in jrc_to_model_fuels_temp.items() if v}
    dm_temp.groupby(jrc_to_model_fuels_temp, "Variables", inplace=True)
    dm_temp.add(0, "Variables", ["efuel", "hydrogen"], unit, dummy=True)
    dm_temp.sort("Variables")

    # deepen
    for v in dm_temp.col_labels["Variables"]:
        dm_temp.rename_col(v, names_map[sheet_name] + "_" + v, "Variables")
    dm_temp.deepen()

    # return
    return dm_temp
