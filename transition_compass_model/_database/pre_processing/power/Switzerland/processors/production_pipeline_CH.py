import os

import numpy as np
import pandas as pd
import requests
from openpyxl import load_workbook

from transition_compass_model.model.common.data_matrix_class import DataMatrix


def read_excel_with_merged_cells(filepath, sheet_name=0):
    # Load workbook and worksheet
    wb = load_workbook(filename=filepath, data_only=True)
    ws = wb[sheet_name] if isinstance(sheet_name, str) else wb.worksheets[sheet_name]

    # Build a matrix with the values
    max_row = ws.max_row
    max_col = ws.max_column
    data = [[None for _ in range(max_col)] for _ in range(max_row)]

    # Fill values from merged cells
    for merged_range in ws.merged_cells.ranges:
        min_row, max_row_range = merged_range.min_row, merged_range.max_row
        min_col, max_col_range = merged_range.min_col, merged_range.max_col
        top_left_value = ws.cell(row=min_row, column=min_col).value
        for row in range(min_row, max_row_range + 1):
            for col in range(min_col, max_col_range + 1):
                data[row - 1][col - 1] = top_left_value

    # Fill in remaining cells (non-merged or not already filled)
    for row in ws.iter_rows(
        min_row=1, max_row=ws.max_row, min_col=1, max_col=ws.max_column
    ):
        for cell in row:
            if data[cell.row - 1][cell.column - 1] is None:
                data[cell.row - 1][cell.column - 1] = cell.value

    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df


def extract_production_data(file_url, local_filename, years_ots):
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

    df = pd.read_excel(local_filename, sheet_name="T24")
    combined_headers = []
    for col1, col2, col3 in zip(df.iloc[4], df.iloc[5], df.iloc[6]):
        combined_headers.append(str(col1) + "-" + str(col2) + "[" + str(col3) + "]")
    # Set the new header
    df.columns = combined_headers
    df = df[7:].copy()

    def is_valid_number(val):
        return isinstance(val, (int, float)) and not pd.isna(val)

    # Apply the function to filter out rows with no valid numeric values
    df = df[df.apply(lambda row: row.map(is_valid_number).any(), axis=1)]
    # Apply similarly for columns if needed
    df = df.loc[:, df.apply(lambda col: col.map(is_valid_number).any())]

    df.rename({"Année-nan[nan]": "Years"}, axis=1, inplace=True)
    df["Country"] = "Switzerland"
    df.set_index(["Country", "Years"], inplace=True)
    # df.columns = ['pow_production_'+col for col in df.columns]
    df = df[df.columns.drop(list(df.filter(regex="[%]")))]
    df.columns = [str.replace(col, "nan-", "") for col in df.columns]
    df.columns = [str.replace(col, "nan", "") for col in df.columns]
    df.replace("-", 0, inplace=True)
    df.reset_index(inplace=True)
    df = df.drop("Total[GWh]", axis=1)
    dm = DataMatrix.create_from_df(df, num_cat=0)
    cols = [
        "Centrales hydrauliques-Centrales au fil de l'eau",
        "Centrales nucléaires-",
        "Centrales thermiques class. et centrales chaleur-force1-Total",
        "Centrales à accumulation",
        "Energies renouvelables diverses3-Chauffages au bois et en partie au bois",
        "Eoliennes",
        "Installations au biogaz",
        "Installations photo-voltaïques",
        "Pompage d'accumu-lation-",
        "Production nationale (brute)-",
        "Production nette (pompage déduit)-",
        "dont\nnon renouvelable",
        "dont renouvelable 2",
    ]

    dm_out = dm.groupby(
        {
            "pow_production_RoR": ".*au fil de l'eau.*",
            "pow_production_Nuclear": ".*nucléaire.*",
            "pow_production_Oil-Gas-Waste": ".*class.*|.*biogaz.*",
            "pow_production_Dam-gross": ".*accumulation.*",
            "pow_production_WindOn": ".*Eoliennes.*",
            "pow_production_PV-roof": ".*photo.*",
            "pow_production_Pump-Open": ".*Pompage.*",
            "pow_production_Waste": ".*dont renouvelable.*",
        },
        regex=True,
        dim="Variables",
        inplace=False,
    )
    dm_out.deepen()
    dm_out.operation("Dam-gross", "-", "Pump-Open", out_col="Dam", dim="Categories1")
    dm_out.drop(dim="Categories1", col_label=["Dam-gross"])
    dm_out.operation(
        "Oil-Gas-Waste", "-", "Waste", out_col="Oil-Gas", dim="Categories1"
    )
    dm_out.drop(dim="Categories1", col_label="Oil-Gas-Waste")
    dm_out.filter({"Years": years_ots}, inplace=True)
    dm_out.change_unit("pow_production", 1000, "GWh", "MWh")
    return dm_out


def compute_capacity_factor(dm_capacity, dm_production, years_ots):
    dm_capacity_ots = dm_capacity.filter(
        {"Years": years_ots, "Country": ["Switzerland"]}, inplace=False
    )
    dm_capacity_ots.groupby(
        {"Oil-Gas": "Oil|Gas.*"}, regex=True, dim="Categories1", inplace=True
    )
    missing_cat = set(dm_capacity_ots.col_labels["Categories1"]) - set(
        dm_production.col_labels["Categories1"]
    )
    dm_production.add(0, dummy=True, dim="Categories1", col_label=missing_cat)
    dm_production.append(dm_capacity_ots, dim="Variables")

    # Determine the capacity factor
    # capacity-factor = Production / ( Capacity * 24 * 365)
    # Production comes from Statistical Office and Capacity from Nexus-E
    arr_tmp = dm_production[:, :, "pow_capacity-Pmax", :] * 24 * 365
    dm_production.add(arr_tmp, dim="Variables", col_label="pow_capacity-E", unit="MWh")
    dm_production.operation(
        "pow_production", "/", "pow_capacity-E", out_col="pow_capacity-factor", unit="%"
    )

    # Handle
    dm_cap_factor = dm_production.filter({"Variables": ["pow_capacity-factor"]})
    arr = dm_cap_factor.array
    arr[np.isinf(arr)] = np.nan
    dm_cap_factor.array = arr
    mean_val = np.nanmean(dm_cap_factor.array, axis=1)
    arr = dm_cap_factor.array[0, :, 0, :]
    nan_indices = np.where(np.isnan(arr))
    arr[nan_indices] = np.take(mean_val, nan_indices[1])
    dm_cap_factor.array[0, :, 0, :] = arr
    dm_production.filter({"Variables": ["pow_production"]}, inplace=True)

    return dm_cap_factor, dm_production


def run(years_ots):
    file_url = "https://www.bfe.admin.ch/bfe/fr/home/versorgung/statistik-und-geodaten/energiestatistiken/gesamtenergiestatistik.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZnIvcHVibGljYX/Rpb24vZG93bmxvYWQvNzUxOQ==.html"
    local_filename = "data/statistique_globale_suisse_energie.xlsx"

    dm_production = extract_production_data(file_url, local_filename, years_ots)
    dm_production.change_unit(
        "pow_production", old_unit="MWh", new_unit="TWh", factor=1e-6
    )

    return dm_production, file_url, local_filename
