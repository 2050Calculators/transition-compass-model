# get_transport_demand_pkm, get_transport_demand_vkm, get_travel_demand_region_microrecencement
import os
import pickle
import zipfile

import numpy as np
import pandas as pd

from transition_compass_model._database.pre_processing.constants import (
    CANTONS_NAME,
    GRANDE_REGION_CANTONS,
)
from transition_compass_model._database.pre_processing.params import country_list
from transition_compass_model._database.pre_processing.transport.Switzerland.get_data_functions import (
    utils,
)
from transition_compass_model.model.common.auxiliary_functions import (
    linear_fitting,
    save_url_to_file,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def get_transport_demand_pkm(file_url, local_filename, years_ots):
    header_row = 1
    rows_to_keep = [
        "Chemins de fer",
        "Chemins de fer à crémaillère",
        "Trams",
        "Trolleybus",
        "Autobus",
        "Voitures de tourisme",
        "Motocycles",
        "Cars",
        "Bicyclettes, y. c. vélos électriques lents",
        "À pied",
    ]
    new_name = [
        "rail",
        "rail",
        "metrotram",
        "bus",
        "bus",
        "LDV",
        "2W",
        "LDV",
        "bike",
        "walk",
    ]
    var_name = "tra_passenger_transport-demand"
    unit = "Mpkm"

    # If file does not exist, it downloads it and creates it
    save_url_to_file(file_url, local_filename)

    df_latest = pd.read_excel(local_filename)
    df_earlier = pd.read_excel(local_filename, sheet_name="1990-2004")

    df_latest[df_latest.columns[0]] = (
        df_latest[df_latest.columns[0]]
        .str.replace(r"\d+\)|\(\d+\)", "", regex=True)
        .str.strip()
    )
    df_earlier[df_earlier.columns[0]] = (
        df_earlier[df_earlier.columns[0]]
        .str.replace(r"\d+\)|\(\d+\)", "", regex=True)
        .str.strip()
    )

    # Clean df from excel file
    names_map = dict()
    for i, row in enumerate(rows_to_keep):
        names_map[row] = new_name[i]
    dm_latest = utils.df_fso_excel_to_dm(
        df_latest, header_row, names_map, var_name, unit, num_cat=1
    )
    # The names change from 1990-2004 to 2005-2023
    names_map.pop("Autobus")
    names_map["Transport par bus"] = "bus"
    names_map.pop("Bicyclettes, y. c. vélos électriques lents")
    names_map["Bicyclettes"] = "bike"
    names_map.pop("À pied")
    names_map["à pied"] = "walk"
    dm_earlier = utils.df_fso_excel_to_dm(
        df_earlier, header_row, names_map, var_name, unit, num_cat=1
    )
    dm_earlier.append(dm_latest, dim="Years")
    dm = dm_earlier.copy()

    # Fix 2023 is missing for various transport types
    ## Replace 0 with nan
    mask = dm.array == 0
    dm.array[mask] = np.nan
    # Extrapolate for 2023 starting from 2020
    years_gt_2020 = [y for y in years_ots if y >= 2020]
    dm_gt_2020 = dm.filter({"Years": years_gt_2020})
    linear_fitting(dm_gt_2020, years_gt_2020)
    years_lt_2008 = [y for y in years_ots if y < 2008]
    dm_lt_2008 = dm.filter({"Years": years_lt_2008})
    linear_fitting(dm_lt_2008, years_lt_2008)
    idx = dm.idx
    dm.array[:, idx[2020] :, ...] = dm_gt_2020.array
    dm.array[:, 0 : idx[2008], ...] = dm_lt_2008.array

    dm.change_unit(var_name, factor=1e6, old_unit=unit, new_unit="pkm")

    return dm


def get_transport_demand_vkm(file_url, local_filename, years_ots):
    # If file does not exist, it downloads it and creates it
    rows_to_keep = [
        "en millions de trains-km",
        "Tram",
        "Trolleybus",
        "Autobus",
        "Voitures de tourisme",
        "Cars privés",
        "Motocycles",
    ]
    new_name = ["rail", "metrotram", "bus", "bus", "LDV", "LDV", "2W"]
    var_name = "tra_passenger_transport-demand-vkm"
    unit = "Mvkm"
    header_row = 0

    save_url_to_file(file_url, local_filename)

    df_latest = pd.read_excel(local_filename)
    df_earlier = pd.read_excel(local_filename, sheet_name="1990-2004")

    df_latest[df_latest.columns[0]] = (
        df_latest[df_latest.columns[0]]
        .str.replace(r"\d+\)|\(\d+\)", "", regex=True)
        .str.strip()
    )
    df_earlier[df_earlier.columns[0]] = (
        df_earlier[df_earlier.columns[0]]
        .str.replace(r"\d+\)|\(\d+\)", "", regex=True)
        .str.strip()
    )

    # Clean df from excel file
    names_map = dict()
    for i, row in enumerate(rows_to_keep):
        names_map[row] = new_name[i]
    dm_latest = utils.df_fso_excel_to_dm(
        df_latest, header_row, names_map, var_name, unit, num_cat=1
    )
    names_map.pop("Autobus")
    names_map["Transport par bus"] = "bus"
    dm_earlier = utils.df_fso_excel_to_dm(
        df_earlier, header_row, names_map, var_name, unit, num_cat=1
    )
    dm_earlier.append(dm_latest, dim="Years")
    dm = dm_earlier.copy()

    # Fix 2023 is missing for various transport types
    ## Replace 0 with nan
    mask = dm.array == 0
    dm.array[mask] = np.nan

    # Extrapolate for 2023 starting from 2020
    years_gt_2020 = [y for y in years_ots if y >= 2020]
    linear_fitting(dm, years_gt_2020, based_on=years_gt_2020)

    dm.change_unit(var_name, factor=1e6, old_unit=unit, new_unit="vkm")

    # Metrotram has a weird spike in 2004 that is there in the raw data, but I want to remove
    idx = dm.idx
    dm.array[:, idx[2004], :, idx["metrotram"]] = np.nan
    dm.fill_nans(dim_to_interp="Years")

    return dm


def get_travel_demand_region_microrecencement(
    file_url=None, local_filename="", year=2000
):
    if file_url is not None:
        save_url_to_file(file_url, local_filename)
    df = pd.read_excel(local_filename)
    # Select thye columns of interest
    # TODO fine a cleaner year to only get the region of interest
    # output the row in which the grande region are given
    year_to_row = {
        2000: 2,
        2005: 2,
        2010: 2,
        2015: 2,
        2021: 3,
    }
    # Clean the df from the excel file
    df.replace("\n", "", regex=True, inplace=True)

    # Select only grande region columns
    cols = ["Unnamed: 1", "Unnamed: 2", "Unnamed: 3"] + df.columns[
        df.iloc[year_to_row[year]].isin(GRANDE_REGION_CANTONS.keys())
    ].tolist()
    df = df[cols]

    # Rename columns
    grande_region_cols = df.columns[
        df.iloc[year_to_row[year]].isin(GRANDE_REGION_CANTONS.keys())
    ]

    df = df.rename(
        columns=dict(
            zip(
                ["Unnamed: 1", "Unnamed: 2", "Unnamed: 3"],
                ["Variables", "Reason", "Switzerland"],
            )
        )
    )

    df = df.rename(
        columns={col: df.loc[year_to_row[year], col] for col in grande_region_cols}
    )

    df["Variables"] = df["Variables"].ffill()
    # Keep only the sum of all reasons to travel
    df = df.loc[df["Reason"] == "Tous les motifs"].copy()
    del df["Reason"]
    df = df.dropna(subset=["Variables"])

    # Add years col
    df["Years"] = year

    # Clean names for dm
    df["Variables"] = df["Variables"].str.replace("\n", " ")
    df["Variables"] = df["Variables"].str.split(",").str[0]
    df["Variables"] = df["Variables"].str.replace(r"\s*\(.*?\)\s*", "", regex=True)

    groupby_dict = {
        "walk": "pied",
        "bus": "Autocar|Car|Bus",
        "metrotram": "Tram",
        "bike": "Vélo",
        "rail": "Train",
        "LDV": "Voiture|Taxi",
        "aviation": "Avion",
        "2W": "Motocycle|Cyclomoteur",
    }

    for new_cat, old_cat in groupby_dict.items():
        # Use word boundaries to match full words only
        # Use str.contains to check if old_cat is in the Variables
        mask = df["Variables"].str.contains(old_cat, regex=True)
        # Replace entire cell with new_cat if old_cat is found
        df.loc[mask, "Variables"] = new_cat

    df = df[df["Variables"].isin(groupby_dict.keys())].copy()

    df_T = pd.melt(
        df, id_vars=["Variables", "Years"], var_name="Country", value_name="values"
    )
    df_pivot = df_T.pivot_table(
        index=["Country", "Years"],
        columns=["Variables"],
        values="values",
        aggfunc="sum",
    )

    # Add variable name
    df_pivot = df_pivot.add_suffix("[pkm/cap/day]")
    df_pivot = df_pivot.add_prefix("tra_pkm-cap_")
    df_pivot.reset_index(inplace=True)
    df_pivot.rename(columns={"Country": "Grande_region"}, inplace=True)
    # Create canton-level rows by copying the corresponding Grande Région values
    rows = []

    for _, row in df_pivot.iterrows():
        grande_region = row["Grande_region"]

        if grande_region in GRANDE_REGION_CANTONS:
            for canton in GRANDE_REGION_CANTONS[grande_region]:
                new_row = row.copy()
                new_row["Country"] = canton
                rows.append(new_row)
        else:
            # Keep rows such as Switzerland if they are not a Grande Région
            rows.append(row)

    df_pivot = pd.DataFrame(rows).reset_index(drop=True)
    df_pivot.loc[df_pivot["Grande_region"] == "Switzerland", "Country"] = "Switzerland"
    df_pivot.drop(columns=["Grande_region"], inplace=True)

    #
    # Convert to dm
    dm = DataMatrix.create_from_df(df_pivot, num_cat=1)
    dm.change_unit(
        "tra_pkm-cap", factor=365, old_unit="pkm/cap/day", new_unit="pkm/cap"
    )
    return dm


scenario_table = {
    "WWB": "Tabelle 03-05: Entwicklung der Fahrleistung von Strassenfahrzeugen im Szenario Weiter wie bisher",
    "ZERO-B": "Tabelle 03-03: Entwicklung der Fahrleistung von Strassenfahrzeugen in ZERO-Variante B",
}
scenario_year = {"WWB": 18, "ZERO-B": 104}


def extract_EP2050_transport_vkm_demand(
    file_url, zip_name, file_pickle, scenario="WWB"
):
    try:
        with open(file_pickle, "rb") as handle:
            dm = pickle.load(handle)

    except OSError:
        extract_dir = os.path.splitext(zip_name)[0]  # 'data/EP2050_sectors'
        if not os.path.exists(extract_dir):
            save_url_to_file(file_url, zip_name)

            # Extract the file
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_name, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

        file_tra = (
            extract_dir
            + "/EP2050+_Szenarienergebnisse_Details_Nachfragesektoren/EP2050+_Detailergebnisse 2020-2060_Verkehrssektor_alle Szenarien_2022-04-12.xlsx"
        )
        df = pd.read_excel(file_tra, sheet_name="03 Fahrleistung")

        df.drop(columns=[df.columns[0], df.columns[3]], inplace=True)

        table_title = scenario_table[scenario]
        start_table_row = df.index[df["Unnamed: 1"] == table_title].tolist()[1]
        df.columns = df.iloc[start_table_row + 2]

        df = df.iloc[start_table_row + 3 : start_table_row + 37]

        # Years as int
        col_mode_name = df.columns[0]
        col_tech_name = df.columns[1]
        df.set_index([col_mode_name, col_tech_name], inplace=True)
        df.columns = df.columns.astype(int)
        df.reset_index(inplace=True)

        # Change variables names
        full_name = ["tra_vkm_demand_" + var for var in df[col_mode_name]]
        df[col_mode_name] = full_name
        df["Full_name"] = df[col_mode_name] + "_" + df[col_tech_name] + ["[mio-vkm]"]
        df.drop(columns=[col_mode_name, col_tech_name], inplace=True)
        # Move "Full_name" column at the beginning
        first = df["Full_name"]
        df.drop(labels=["Full_name"], axis=1, inplace=True)
        df.insert(0, "Full_name", first)
        df["Full_name"] = df["Full_name"].str.replace("(", "", regex=False)
        df["Full_name"] = df["Full_name"].str.replace(")", "", regex=False)

        # Pivot
        df_T = df.T
        df_T.columns = df_T.iloc[0]
        df_T = df_T.iloc[1:]
        df_T.reset_index(inplace=True)
        df_T.rename(columns={scenario_year[scenario]: "Years"}, inplace=True)
        df_T["Country"] = "Switzerland"

        dm = DataMatrix.create_from_df(df_T, num_cat=2)

        # Rename mode of transport
        dm.rename_col(
            ["HGV", "LCV", "motorcycle", "pass. car"],
            ["HDVH", "HDVL", "2W", "LDV"],
            dim="Categories1",
        )
        dm.groupby({"bus": ["coach", "urban bus"]}, dim="Categories1", inplace=True)
        # Rename tech transport
        dm.groupby(
            {
                "BEV": ["electricity"],
                "ICE-gas": ["CNG", "LNG", "bifuel CNG/petrol"],
                "ICE-gasoline": [
                    "petrol 2S",
                    "petrol 4S",
                    "bifuel LPG/petrol",
                    "flex-fuel E85",
                ],
                "PHEV-diesel": ["Plug-in Hybrid diesel/electric"],
                "PHEV-gasoline": ["Plug-in Hybrid petrol/electric"],
                "FCEV": ["FuelCell"],
                "ICE-diesel": ["diesel"],
            },
            dim="Categories2",
            inplace=True,
        )

        with open(file_pickle, "wb") as handle:
            pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dm.sort("Categories1")
    dm.sort("Categories2")
    dm.change_unit(
        "tra_vkm_demand", old_unit="mio-vkm", new_unit="vkm", factor=1e6, operator="*"
    )

    return dm


def pkm_MRMT(file_folder_2015, file_path_2021):  # ,
    # Checks that the file for 2021 exists
    save_url_to_file(
        "https://www.bfs.admin.ch/bfsstatic/dam/assets/24025445/master", file_path_2021
    )

    cantons_list = [x for x in country_list if x != "Switzerland"]
    # iterate in canton to get the valeu for each one
    asset_ids_2015 = {"Vaud": "2081714", "Fribourg": "2081306", "Schwyz": "2005567"}
    canton_pkm_day = {}
    for canton in cantons_list:
        canton_pkm_day[canton] = {}
        canton_pkm_day[canton][2021] = pd.read_excel(
            file_path_2021, sheet_name=CANTONS_NAME["name_to_accronym"][canton]
        ).iloc[5, 9]
        file_canton_2015 = (
            file_folder_2015 + CANTONS_NAME["name_to_accronym"][canton] + ".xlsx"
        )

        # If the file don't exist, it downloads it and creates it
        save_url_to_file(
            f"https://dam-api.bfs.admin.ch/hub/api/dam/assets/{asset_ids_2015[canton]}/master",
            file_canton_2015,
        )
        canton_pkm_day[canton][2015] = pd.read_excel(file_canton_2015).iloc[4, 9]

    return canton_pkm_day
