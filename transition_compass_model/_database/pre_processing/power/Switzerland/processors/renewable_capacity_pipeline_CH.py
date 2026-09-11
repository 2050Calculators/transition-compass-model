import os

import pandas as pd
import requests

from transition_compass_model.model.common.data_matrix_class import DataMatrix


def extract_renewable_capacity_data(file_url, local_filename):
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

    df = pd.read_excel(local_filename, sheet_name="Anhang B")
    # Set the new header
    df.columns = df.iloc[1]
    df = df.loc[[48, 231]].copy()
    df["Technologie"] = df["Technologie"].replace(
        {
            "Photovoltaikanl. (Netz+Insel)": "pow_capacity-Pmax_PV-roof[MW]",
            "Windenergieanlagen": "pow_capacity-Pmax_WindOn[MW]",
        }
    )
    # Keep "Technologie" column and Years columns
    cols_to_keep = []
    for col in df.columns:
        if isinstance(col, int) or col == "Technologie":
            cols_to_keep.append(col)
    df = df[cols_to_keep]

    df_melted = df.melt(id_vars=["Technologie"], var_name="Years", value_name="Value")
    df_pivoted = df_melted.pivot(
        index="Years", columns="Technologie", values="Value"
    ).reset_index()
    df_pivoted["Country"] = "Switzerland"

    dm = DataMatrix.create_from_df(df_pivoted, num_cat=1)

    return dm


def run():
    # https://www.bfe.admin.ch/bfe/en/home/supply/statistics-and-geodata/energy-statistics/sector-statistics.html
    # Excel file under Renewable Energy titled "Swiss Statistics of the Renewable energies" or
    # "Schweizerische Statistik der erneuerbaren Energien 2023 - Datentabellen"
    file_url = "https://www.bfe.admin.ch/bfe/en/home/versorgung/statistik-und-geodaten/energiestatistiken/teilstatistiken.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvODc4Nw==.html"
    local_filename = "data/swiss_statistics_of_the_renewable_energies.xlsx"
    dm_capacity_PV_wind_ots = extract_renewable_capacity_data(file_url, local_filename)

    return dm_capacity_PV_wind_ots
