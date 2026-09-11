import os

import numpy as np
import pandas as pd
import requests

from transition_compass_model._database.pre_processing.power.Switzerland.processors.production_pipeline_CH import (
    read_excel_with_merged_cells,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def df_fso_excel_to_dm(
    df,
    header_row,
    names_dict,
    var_name,
    unit,
    num_cat,
    keep_first=False,
    country="Switzerland",
):
    # Federal statistical office df from excel to dm
    # Change headers
    new_header = df.iloc[header_row]
    new_header.values[0] = "Variables"
    df.columns = new_header
    df = df[header_row + 1 :].copy()
    # Remove nans and empty columns/rows
    if np.nan in df.columns:
        df.drop(columns=np.nan, inplace=True)
    df.set_index("Variables", inplace=True)
    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    # Filter rows that contain at least one number (integer or float)
    df = df[df.apply(lambda row: row.map(pd.api.types.is_number), axis=1).any(axis=1)]
    df_clean = df.loc[
        :, df.apply(lambda col: col.map(pd.api.types.is_number)).any(axis=0)
    ].copy()
    # Extract only the data we are interested in:
    df_filter = df_clean.loc[names_dict.keys()].copy()
    df_filter = df_filter.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    # df_filter = df_filter.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
    df_filter.reset_index(inplace=True)
    # Keep only first 10 caracters
    df_filter["Variables"] = df_filter["Variables"].replace(names_dict)
    if keep_first:
        df_filter = df_filter.drop_duplicates(subset=["Variables"], keep="first")
    df_filter = df_filter.groupby(["Variables"]).sum()
    df_filter.reset_index(inplace=True)

    # Pivot the dataframe
    df_filter["Country"] = country
    df_T = pd.melt(
        df_filter,
        id_vars=["Variables", "Country"],
        var_name="Years",
        value_name="values",
    )
    df_pivot = df_T.pivot_table(
        index=["Country", "Years"],
        columns=["Variables"],
        values="values",
        aggfunc="sum",
    )
    df_pivot = df_pivot.add_suffix("[" + unit + "]")
    df_pivot = df_pivot.add_prefix(var_name + "_")
    df_pivot.reset_index(inplace=True)

    # Drop non numeric values in Years col
    df_pivot["Years"] = pd.to_numeric(df_pivot["Years"], errors="coerce")
    df_pivot = df_pivot.dropna(subset=["Years"])

    dm = DataMatrix.create_from_df(df_pivot, num_cat=num_cat)
    return dm


def extract_energy_statistics_data(
    file_url, local_filename, sheet_name, parameters, years_ots
):
    mapping = parameters["mapping"]  # dictionary,  to rename column headers
    var_name = parameters["var name"]  # string, dm variable name
    headers_idx = parameters[
        "headers indexes"
    ]  # tuple with index of rows to keep for header
    first_row = parameters[
        "first row"
    ]  # integer with the first row to keep # regex expression with cols to drop
    unit = parameters["unit"]  # Put None if unit is in table, else str
    col_to_drop = parameters[
        "cols to drop"
    ]  # None if no need to drop, else string (for dm.drop)

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

    df = read_excel_with_merged_cells(local_filename, sheet_name)
    combined_headers = []
    if unit is None:
        for col1, col2, col3 in zip(
            df.iloc[headers_idx[0]], df.iloc[headers_idx[1]], df.iloc[headers_idx[2]]
        ):
            combined_headers.append(str(col1) + "-" + str(col2) + "[" + str(col3) + "]")
    else:
        for col1, col2 in zip(df.iloc[headers_idx[0]], df.iloc[headers_idx[1]]):
            combined_headers.append(str(col1) + "-" + str(col2) + "[" + unit + "]")
    # Set the new header
    df.columns = combined_headers
    df = df[first_row:].copy()

    def is_valid_number(val):
        return isinstance(val, (int, float)) and not pd.isna(val)

    # Apply the function to filter out rows with no valid numeric values
    df = df[df.apply(lambda row: row.map(is_valid_number).any(), axis=1)]
    # Apply similarly for columns if needed
    df = df.loc[:, df.apply(lambda col: col.map(is_valid_number).any())]

    df.rename({df.columns[0]: "Years"}, axis=1, inplace=True)
    df["Country"] = "Switzerland"
    df.replace("-", 0, inplace=True)
    dm = DataMatrix.create_from_df(df, num_cat=0)
    if col_to_drop is not None:
        dm.drop(dim="Variables", col_label=col_to_drop)

    for key in list(mapping.keys()):
        mapping[var_name + "_" + key] = mapping.pop(key)

    dm_out = dm.groupby(mapping, regex=True, dim="Variables", inplace=False)
    dm_out.deepen()
    dm_out.filter({"Years": years_ots}, inplace=True)
    # dm_out.change_unit(var_name, 277.8, 'TJ', 'MWh')

    return dm_out


def extract_districtheating_demand(
    file_url, local_filename, sheet_name, mapping, var_name, years_ots
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

    df = read_excel_with_merged_cells(local_filename, sheet_name)
    combined_headers = []
    for col1, col2 in zip(df.iloc[5], df.iloc[6]):
        combined_headers.append(str(col1) + "_" + str(col2) + "[TJ]")

    # Set the new header
    df.columns = combined_headers
    df = df[19:].copy()

    def is_valid_number(val):
        return isinstance(val, (int, float)) and not pd.isna(val)

    # Apply the function to filter out rows with no valid numeric values
    df = df[df.apply(lambda row: row.map(is_valid_number).any(), axis=1)]
    # Apply similarly for columns if needed
    df = df.loc[:, df.apply(lambda col: col.map(is_valid_number).any())]

    df.rename({"Année_Année[TJ]": "Years"}, axis=1, inplace=True)
    df["Country"] = "Switzerland"
    df.replace("-", 0, inplace=True)
    dm = DataMatrix.create_from_df(df, num_cat=0)
    dm.filter_w_regex({"Variables": "Energie utilisé.*"}, inplace=True)
    dm.rename_col_regex("Energie utilisé_", "", dim="Variables")

    for key in list(mapping.keys()):
        mapping[var_name + "_" + key] = mapping.pop(key)

    dm_out = dm.groupby(mapping, regex=True, dim="Variables", inplace=False)
    dm_out.deepen()
    dm_out.filter({"Years": years_ots}, inplace=True)
    # dm_out.change_unit(var_name, 277.8, 'TJ', 'MWh')

    return dm_out


def run(file_url, local_filename, dm_production, years_ots):
    ######################################################################
    ##  Extract historical demand of fuels other than electricity prod  ##
    ######################################################################

    # Statistique Global Suisse de l'Energie
    # Bois et charbon de bois1
    # Force hydraulique
    # Ordures ménagères et déchets industriels2
    # Charbon
    # Pétrole brut et produits pétroliers
    # dont pétrole brut
    # dont produits pétroliers
    # Gaz
    # Combustibles nucléaires
    # Autres énergies renouvelables3
    # Utilisation totale d'agents énergé-tiques
    # Elektricité Solde de Import/Export
    # Consommation brute d'énergie dans le pays (100%)
    parameters = dict()
    mapping = {
        "heating-oil": ".*Combustibles.*",
        "transport-oil": ".*Carburants.*",
        "gas": ".*Gaz.*",
        "coal": ".*Charbon.*",
        "wood": ".*bois.*",
        "district-heating": ".*distance.*",
        "waste": ".*industriel.*",
        "biofuels": ".*biogènes.*",
        "biogas": ".*Biogaz.*",
    }
    parameters["mapping"] = mapping  # dictionary,  to rename column headers
    parameters["var name"] = "pow_fuel-demand"  # string, dm variable name
    parameters["headers indexes"] = (
        5,
        6,
    )  # tuple with index of rows to keep for header
    parameters["first row"] = 87  # integer with the first row to keep
    parameters["unit"] = "TJ"
    parameters["cols to drop"] = None

    dm_fuels_demand = extract_energy_statistics_data(
        file_url,
        local_filename,
        sheet_name="T14",
        parameters=parameters,
        years_ots=years_ots,
    )

    ##### Supply
    # SECTION - Energy supply
    parameters = dict()
    mapping = {
        "hydro-power": ".*hydraulique.*",
        "wood": ".*bois.*",
        "waste": ".*déchets.*",
        "coal": ".*charbon.*",
        "oil": ".*Pétrole brut et produits pétroliers.*",
        "gas": ".*Gaz.*",
        "nuclear": ".*nucléaires.*",
        "renewables": ".*renouvelables.*",
    }
    parameters["mapping"] = mapping  # dictionary,  to rename column headers
    parameters["var name"] = "pow_fuel-supply"  # string, dm variable name
    parameters["headers indexes"] = (
        4,
        5,
        5,
    )  # tuple with index of rows to keep for header
    parameters["first row"] = 86  # integer with the first row to keep
    parameters["cols to drop"] = ".*%.*"
    parameters["unit"] = None

    dm_fuels_supply = extract_energy_statistics_data(
        file_url,
        local_filename,
        sheet_name="T10",
        parameters=parameters,
        years_ots=years_ots,
    )

    ##### Losses
    # SECTION - Losses of electricity
    parameters = dict()
    mapping = {"Losses": ".*Centrales électriques.*"}
    parameters["mapping"] = mapping  # dictionary,  to rename column headers
    parameters["var name"] = "pow_production"  # string, dm variable name
    parameters["headers indexes"] = (
        3,
        4,
    )  # tuple with index of rows to keep for header
    parameters["first row"] = 5  # integer with the first row to keep
    parameters["cols to drop"] = (
        ".*Raffineries.*|.*Usines.*|.*Chaleur.*|.*Total.*|.*Consommation.*|.*%.*"
    )
    parameters["unit"] = "TJ"

    dm_losses = extract_energy_statistics_data(
        file_url,
        local_filename,
        sheet_name="T13",
        parameters=parameters,
        years_ots=years_ots,
    )
    dm_losses.change_unit(
        "pow_production", factor=3600, old_unit="TJ", new_unit="TWh", operator="/"
    )
    # Remove Pump-Open losses from total losses to avoid double counting
    dm_losses["Switzerland", :, "pow_production", "Losses"] = (
        dm_losses["Switzerland", :, "pow_production", "Losses"]
        - dm_production["Switzerland", :, "pow_production", "Pump-Open"]
    )
    dm_losses.array = -dm_losses.array
    dm_production.append(dm_losses, dim="Categories1")

    #######################################
    ###      Oil Supply by type     #######
    #######################################
    mapping = {
        "heating-oil": ".*Huile.*",
        "kerosene": ".*aviation.*",
        "diesel": ".*diesel.*",
        "gasoline": ".*Essence2-Total.*",
        "other": ".*Coke.*|.*Autres.*",
    }
    parameters["mapping"] = mapping  # dictionary,  to rename column headers
    parameters["var name"] = "pow_fuel-supply"  # string, dm variable name
    parameters["headers indexes"] = (
        5,
        6,
    )  # tuple with index of rows to keep for header
    parameters["first row"] = 23  # integer with the first row to keep
    parameters["cols to drop"] = None
    parameters["unit"] = "1000t"
    dm_oil_split = extract_energy_statistics_data(
        file_url,
        local_filename,
        sheet_name="T20",
        parameters=parameters,
        years_ots=years_ots,
    )
    # Conversion factor based on Lower Heating Value
    # Reference https://world-nuclear.org/information-library/facts-and-figures/heat-values-of-various-fuels
    # Reference heating-oil https://www.forestresearch.gov.uk/tools-and-resources/fthr/biomass-energy-resources/reference-biomass/facts-figures/typical-calorific-values-of-fuels/
    # Reference kerosene Linstrom, Peter (2021). NIST Chemistry WebBook. NIST Standard Reference Database Number 69. NIST Office of Data and Informatics. doi:10.18434/T4D303.
    # (value reported by wikipedia https://en.wikipedia.org/wiki/Heat_of_combustion#cite_note-NIST-11)
    # For other I'm using 42.
    LHV_MJ_kg = {
        "heating-oil": 42.5,
        "diesel": 44,
        "gasoline": 45,
        "kerosene": 44.1,
        "other": 42,
    }
    # MJ/kg = TJ/1000tonnes
    for var, conv_fact in LHV_MJ_kg.items():
        dm_oil_split[:, :, :, var] = conv_fact * dm_oil_split[:, :, :, var]
    dm_oil_split.change_unit(
        "pow_fuel-supply", factor=1, old_unit="1000t", new_unit="TJ"
    )

    # Map oil supply using dm_oil_split
    dm_fuels_supply.append(dm_oil_split, dim="Categories1")
    dm_fuels_supply.drop(dim="Categories1", col_label=["oil"])
    keep_fuel_cat = [
        "gasoline",
        "gas",
        "waste",
        "heating-oil",
        "wood",
        "diesel",
        "kerosene",
    ]
    dm_fuels_supply.filter({"Categories1": keep_fuel_cat}, inplace=True)
    dm_fuels_supply.change_unit(
        "pow_fuel-supply", factor=3600, old_unit="TJ", new_unit="TWh", operator="/"
    )

    # District-heating
    # mapping = {'wood': 'Bois', 'coal': 'Charbon', 'nuclear': 'Combustibles nucléaires3', 'other': 'Divers4',
    #           'electricity': 'Electricité', 'gas': 'Gaz1', 'heating-oil': 'Huile extra-légère|Huile moyenne et lourde',
    #           'waste': 'Ordures2'}
    # dm_distrheat = extract_districtheating_demand(file_url, local_filename, sheet_name='T26', mapping=mapping,
    #                                              var_name='pow_district-heating')

    #  but they are not appearing correctly in the final dm_capacity
    # Correct this. Then I think the rest is good.

    return dm_fuels_supply, dm_production
