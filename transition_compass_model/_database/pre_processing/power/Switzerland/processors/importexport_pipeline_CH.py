import os

import pandas as pd
import requests

from transition_compass_model._database.pre_processing.power.Switzerland.processors.production_pipeline_CH import (
    read_excel_with_merged_cells,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def extract_importexport_data(
    file_url, local_filename, sheet_name, var_name, mapping, years_ots
):
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

    # df = pd.read_excel(local_filename, sheet_name="T06")

    df = read_excel_with_merged_cells(local_filename, sheet_name)
    combined_headers = []
    for col1, col2 in zip(df.iloc[4], df.iloc[5]):
        combined_headers.append(str(col1) + "[" + str(col2) + "]")
    # Set the new header
    df.columns = combined_headers
    df = df[6:].copy()

    def is_valid_number(val):
        return isinstance(val, (int, float)) and not pd.isna(val)

    # Apply the function to filter out rows with no valid numeric values
    df = df[df.apply(lambda row: row.map(is_valid_number).any(), axis=1)]
    # Apply similarly for columns if needed
    df = df.loc[:, df.apply(lambda col: col.map(is_valid_number).any())]

    df.rename({"Année[Année]": "Years"}, axis=1, inplace=True)
    df["Country"] = "Switzerland"
    df.set_index(["Country", "Years"], inplace=True)
    # df.columns = ['pow_production_'+col for col in df.columns]
    df.replace("-", 0, inplace=True)
    df.replace("None", 0, inplace=True)
    df.reset_index(inplace=True)
    df = df.drop("Total[TJ]", axis=1)
    col_to_keep = df.columns[df.columns.str.contains("TJ", case=False)].tolist()
    filtered_df = df[["Country", "Years"] + col_to_keep].copy()
    dm = DataMatrix.create_from_df(filtered_df, num_cat=0)
    for key in list(mapping.keys()):
        mapping[var_name + "_" + key] = mapping.pop(key)
    dm_out = dm.groupby(mapping, regex=True, dim="Variables", inplace=False)
    dm_out.deepen()
    dm_out.filter({"Years": years_ots}, inplace=True)
    dm_out.change_unit(var_name, 3.6 * 1e-3, "TJ", "MWh", operator="/")

    return dm_out


def run(file_url, local_filename, dm_production, years_ots):
    # SECTION - Energy trade (incl.electricity)
    # Extract import
    mapping = {
        "wood": ".*Bois.*",
        "biofuels": ".*biogènes.*",
        "coal": ".*Charbon.*",
        "electricity": ".*Electricité.*",
        "waste": ".*Ordures.*",
        "gas": ".*Gaz.*",
        "oil": ".*Pétrole.*",
    }
    dm_import = extract_importexport_data(
        file_url,
        local_filename,
        sheet_name="T06",
        var_name="pow_import",
        mapping=mapping,
        years_ots=years_ots,
    )
    dm_import.change_unit("pow_import", old_unit="MWh", new_unit="TWh", factor=1e-6)
    # Extract Export
    mapping = {
        "wood": ".*Bois.*",
        "coal": ".*Charbon.*",
        "electricity": ".*Electricité.*",
        "oil": ".*pétroliers.*",
    }
    dm_export = extract_importexport_data(
        file_url,
        local_filename,
        sheet_name="T07",
        var_name="pow_export",
        mapping=mapping,
        years_ots=years_ots,
    )
    dm_export.change_unit("pow_export", old_unit="MWh", new_unit="TWh", factor=1e-6)
    # Compute Net Import
    cat_missing = list(
        set(dm_import.col_labels["Categories1"])
        - set(dm_export.col_labels["Categories1"])
    )
    dm_export.add(0, col_label=cat_missing, dim="Categories1", dummy=True)
    dm_import.append(dm_export, dim="Variables")
    dm_import.operation(
        "pow_import", "-", "pow_export", out_col="pow_net-import", unit="TWh"
    )

    # Add net-import of electricity as electricity production import
    dm_elec_import = dm_import.filter(
        {"Variables": ["pow_net-import"], "Categories1": ["electricity"]}
    )
    dm_elec_import.rename_col("pow_net-import", "pow_production", dim="Variables")
    dm_elec_import.rename_col("electricity", "Net-import", dim="Categories1")
    dm_production.append(dm_elec_import, dim="Categories1")

    return dm_production
