import os

import numpy as np
import pandas as pd

# Fernwärme (Bezug) → electricity: consistent with EU27 treatment of district heating
CARRIER_MAPPING = {
    "Elektrizität": "electricity",
    "Erdgas": "gas-ff-natural",
    "Heizöl extra-leicht": "liquid-ff-diesel",
    "Heizöl mittel und schwer": "liquid-ff-oil",
    "Industrieabfälle": "solid-waste",
    "Kohle": "solid-ff-coal",
    "Fernwärme (Bezug)": "electricity",
    "Holz": "solid-bio",
}


def _read_sfoe_sheet(filepath, sheet_name, carrier):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df.columns = df.iloc[2, :]
    df = df.iloc[3:, :]
    df = df[df["Secteur"] == "Industrie"]
    df = df[
        df["N° de branche"].apply(
            lambda x: (
                isinstance(x, (int, float))
                and not isinstance(x, bool)
                and 1 <= int(x) <= 11
            )
        )
    ]
    year_cols = [
        c for c in df.columns if isinstance(c, (int, float)) and float(c) > 1900
    ]
    df_melt = df.melt(
        id_vars=["N° de branche"],
        value_vars=year_cols,
        var_name="year",
        value_name="value_TJ",
    )
    df_melt["year"] = df_melt["year"].astype(int)
    df_melt["sector_num"] = df_melt["N° de branche"].astype(int)
    df_melt["carrier"] = carrier
    return df_melt[["year", "sector_num", "carrier", "value_TJ"]]


def get_sfoe_sector_data(current_file_directory):
    """Read SFOE sector-level energy data from both Excel files.

    Returns long-format DataFrame with columns (year, sector_num, carrier, value_TJ).
    Covers industry sectors 1-11 (Construction excluded) for years 1999-2024.
    Carriers mapped to model carrier names; Fernwärme aggregated into electricity.
    """
    filepath_new = os.path.join(
        current_file_directory,
        "../data/energy-demand/8788-Publikationstabellen_DE_FR_IT_2013_bis_2024.xlsx",
    )
    filepath_old = os.path.join(
        current_file_directory,
        "../data/energy-demand/12231-Publikationstabellen_DE_FR_IT_1999_bis_2013.xlsx",
    )

    dfs = []
    for sheet, carrier in CARRIER_MAPPING.items():
        df_new = _read_sfoe_sheet(filepath_new, sheet, carrier)
        df_old = _read_sfoe_sheet(filepath_old, sheet, carrier)
        df_old = df_old[df_old["year"] < 2013]
        dfs.append(pd.concat([df_new, df_old], ignore_index=True))

    df = pd.concat(dfs, ignore_index=True)
    # sum carriers mapping to the same name (electricity + Fernwärme → electricity)
    df = df.groupby(["year", "sector_num", "carrier"], as_index=False)["value_TJ"].sum()
    df = df.sort_values(["sector_num", "carrier", "year"]).reset_index(drop=True)

    # treat exact zeros as missing
    df.loc[df["value_TJ"] == 0, "value_TJ"] = np.nan

    return df
