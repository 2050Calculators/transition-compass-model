import os

import numpy as np
import pandas as pd
import requests

from transition_compass_model.model.common.data_matrix_class import DataMatrix


def extract_hydro_capacity_at_year(df, yr):
    df.rename({"ZE-Kanton": "Country"}, axis=1, inplace=True)
    # Keep only active power-plants
    df = df.loc[df["ZE-Status"] == "im Normalbetrieb"]
    # Filter useful variables
    df_P = df[["Country", "Inst. Pumpenleistung"]].copy()
    df = df.loc[df["WKA-Typ"] != "U"]  # Pump capacity only
    df = df[["WKA-Typ", "Country", "Max. Leistung ab Generator"]]  # RoR and Dam
    df = df.loc[df["WKA-Typ"] != "P"]

    # Group Pump capacity by canton and add years column
    df_P = df_P.groupby(["Country"]).sum()
    df_P["Years"] = yr
    df_P.rename(
        {"Inst. Pumpenleistung": "pow_capacity-Pmax_Pump-Open[MW]"},
        axis=1,
        inplace=True,
    )
    df_P.reset_index(inplace=True)
    dm_P = DataMatrix.create_from_df(df_P, num_cat=1)

    # Group capacity by canton and type
    df = df.groupby(["Country", "WKA-Typ"]).sum()
    df.reset_index(inplace=True)
    df["Years"] = yr
    df["WKA-Typ"] = df["WKA-Typ"].str.replace("L", "pow_capacity-Pmax_RoR[MW]", 1)
    df["WKA-Typ"] = df["WKA-Typ"].str.replace("S", "pow_capacity-Pmax_Dam[MW]", 1)

    # Dam RoR
    df.rename({"Max. Leistung ab Generator": "Pmax"}, axis=1, inplace=True)
    df = df.pivot_table(
        index=["Country", "Years"], columns=["WKA-Typ"], values="Pmax", aggfunc="sum"
    )
    df.reset_index(inplace=True)
    dm = DataMatrix.create_from_df(df, num_cat=1)

    dm.append(dm_P, dim="Categories1")

    return dm


def extract_old_hydro_capacity_data(url_dict):
    dm_all = None
    for yr in url_dict.keys():
        local_filename = url_dict[yr]["local_filename"]
        file_url = url_dict[yr]["file_url"]
        if not os.path.exists(local_filename):
            response = requests.get(file_url, stream=True)
            # Check if the request was successful
            if response.status_code == 200:
                with open(local_filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"File downloaded successfully as {local_filename}")
            else:
                print(f"Error: {response.status_code}, {response.text}")
        else:
            print(
                f"File {local_filename} already exists. If you want to download again delete the file"
            )

        df = pd.read_excel(local_filename)
        dm = extract_hydro_capacity_at_year(df, yr)

        if dm_all is None:
            dm_all = dm.copy()
        else:
            dm_all.append(dm, dim="Years")

    dm_all.sort(dim="Years")

    dm_CH = dm_all.groupby(
        {"Switzerland": ".*"}, regex=True, inplace=False, dim="Country"
    )
    dm_all.append(dm_CH, dim="Country")

    return dm_all


def extract_old_hydro_capacity_zip(url_dict):
    dm_all = None
    for yr in url_dict.keys():
        df = pd.read_excel(url_dict[yr]["local_filename"])
        dm = extract_hydro_capacity_at_year(df, yr)

        if dm_all is None:
            dm_all = dm.copy()
        else:
            dm_all.append(dm, dim="Years")

    dm_all.sort(dim="Years")

    dm_CH = dm_all.groupby(
        {"Switzerland": ".*"}, regex=True, inplace=False, dim="Country"
    )
    dm_all.append(dm_CH, dim="Country")

    return dm_all


def fill_missing_years_capacity_hydro(dm_capacity_hydro_ots, years_map, years_ots):
    years_missing = list(
        set(years_ots) - set(dm_capacity_hydro_ots.col_labels["Years"])
    )
    dm_capacity_hydro_ots.add(np.nan, dim="Years", col_label=years_missing, dummy=True)
    idx = dm_capacity_hydro_ots.idx
    for ref_yr, yr_range in years_map.items():
        for yr in yr_range:
            dm_capacity_hydro_ots.array[:, idx[yr], ...] = dm_capacity_hydro_ots.array[
                :, idx[ref_yr], ...
            ]

    dm_capacity_hydro_ots.sort("Years")
    return dm_capacity_hydro_ots


def adjust_based_on_nexuse(dm, dm_nexuse, years_ots):
    dm.sort("Categories1")
    dm_nexuse.sort("Categories1")
    # Filter from Nexus-e the technologies in dm
    dm_filter = dm_nexuse.filter(
        {
            "Country": ["Switzerland"],
            "Years": years_ots,
            "Variables": ["pow_capacity-Pmax"],
            "Categories1": dm.col_labels["Categories1"],
        }
    )

    # Join Nexus-e and dm (for Switzerland)
    dm.rename_col("pow_capacity-Pmax", "pow_capacity-Pmax-old", dim="Variables")
    dm_tmp = dm.filter({"Country": ["Switzerland"]})
    dm_filter.append(dm_tmp, dim="Variables")

    # Determine the ratio between Nexus-e data (Pmax) and our extracted data (Pmax-old)
    dm_filter.operation(
        "pow_capacity-Pmax", "/", "pow_capacity-Pmax-old", out_col="factor", unit="%"
    )
    dm_filter.array[dm_filter.array == 0] = np.nan
    idx = dm_filter.idx
    # Compute the average adjustment throughout the years
    avg_factor = np.nanmean(
        dm_filter.array[:, :, idx["factor"], :], axis=1, keepdims=True
    )
    # Multiply Pmax (old) by the avg adjustment to match Nexus-e
    idx = dm.idx
    dm.array[:, :, idx["pow_capacity-Pmax-old"], :] = (
        dm.array[:, :, idx["pow_capacity-Pmax-old"], :] * avg_factor
    )
    dm.rename_col("pow_capacity-Pmax-old", "pow_capacity-Pmax", dim="Variables")
    return dm


def run(dm_capacity, years_ots):
    # Hydro-power
    # Source:  https://www.bfe.admin.ch/bfe/fr/home/approvisionnement/energies-renouvelables/force-hydraulique.html#kw-96906
    # 1991 - Statistik der Wasserkraftanlagen der Schweiz. Stand 1.1.1991
    # 1996 - Statistik der Wasserkraftanlagen der Schweiz. Stand 1.1.1996
    # 2001 - Statistik der Wasserkraftanlagen der Schweiz. Stand 1.1.2001
    url_dict = {
        1990: {
            "file_url": "https://www.bfe.admin.ch/bfe/fr/home/versorgung/erneuerbare-energien/wasserkraft.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvNzkyNA==.html",
            "local_filename": "data/hydro_power_1990.xlsx",
        },
        1995: {
            "file_url": "https://www.bfe.admin.ch/bfe/fr/home/versorgung/erneuerbare-energien/wasserkraft.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvNzkyMw==.html",
            "local_filename": "data/hydro_power_1995.xlsx",
        },
        2000: {
            "file_url": "https://www.bfe.admin.ch/bfe/fr/home/versorgung/erneuerbare-energien/wasserkraft.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvNzkyMg==.html",
            "local_filename": "data/hydro_power_2000.xlsx",
        },
        2005: {"file_url": "nan", "local_filename": "data/hydro_power_2005.xlsx"},
        2015: {"file_url": "nan", "local_filename": "data/hydro_power_2015.xlsx"},
        2019: {"file_url": "nan", "local_filename": "data/hydro_power_2019.xlsx"},
        2020: {"file_url": "nan", "local_filename": "data/hydro_power_2020.xlsx"},
        2021: {"file_url": "nan", "local_filename": "data/hydro_power_2021.xlsx"},
        2022: {"file_url": "nan", "local_filename": "data/hydro_power_2022.xlsx"},
        2023: {"file_url": "nan", "local_filename": "data/hydro_power_2023.xlsx"},
    }
    dm_capacity_hydro_ots = extract_old_hydro_capacity_data(url_dict)
    # Create step function profile
    # Allocate missing years to existing year
    years_map = {
        1990: range(1990, 1993 + 1),
        1995: range(1994, 1997 + 1),
        2000: range(1998, 2000 + 1),
        2005: range(2001, 2009 + 1),
        2015: range(2010, 2017 + 1),
        2019: range(2018, 2019 + 1),
    }
    dm_capacity_hydro_ots = fill_missing_years_capacity_hydro(
        dm_capacity_hydro_ots, years_map, years_ots
    )

    dm_capacity_hydro_ots = adjust_based_on_nexuse(
        dm_capacity_hydro_ots, dm_capacity, years_ots
    )

    return dm_capacity_hydro_ots
