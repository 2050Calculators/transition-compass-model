import os
import pickle

import deepl
import numpy as np
import pandas as pd
import requests

from transition_compass_model.model.common.io_database import database_to_dm

# Initialize the Deepl Translator
deepl_api_key = "9ecffb3f-5386-4254-a099-8bfc47167661:fx"
translator = deepl.Translator(deepl_api_key)


def translate_text(text):
    translation = translator.translate_text(text, target_lang="EN-GB")
    return translation.text


def create_ots_years_list(years_setting):
    startyear: int = years_setting[0]  # Start year is argument [0], i.e., 1990
    baseyear: int = years_setting[1]  # Base/Reference year is argument [1], i.e., 2015
    lastyear: int = years_setting[2]  # End/Last year is argument [2], i.e., 2050
    step_fts = years_setting[3]  # Timestep for scenario is argument [3], i.e., 5 years
    years_ots = list(
        np.linspace(start=startyear, stop=baseyear, num=(baseyear - startyear) + 1)
        .astype(int)
        .astype(str)
    )
    return years_ots


def extract_energy_data(file_url, local_filename, baseyear, years_ots, outfile_dm):
    # If the routine had already run and the dm was created, it skips everything and it just loads the dm
    # this is mainly done to avoid calling deepl on repeat which only allows a limited number of calls
    if not os.path.exists(outfile_dm):
        response = requests.get(file_url, stream=True)
        if not os.path.exists(local_filename):
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

        df = pd.read_csv(local_filename)

        # Clean file format
        # Add """ before the first comma
        df["Jahr"] = df["Jahr"].str.replace(",", '",', 1)
        df = df[["Jahr"]]
        # Split the cell into different columns based on ", " separator
        df[["timescale", "Variable", "Description", "value"]] = df["Jahr"].str.split(
            '",', expand=True
        )
        # Drop the original column
        df.drop(columns=["Jahr"], inplace=True)
        for col in df.columns:
            df[col] = df[col].str.lstrip('"')

        # Use deepl to translate variables from de to en
        variables_de = list(set(df["Variable"]))
        variables_en = [translate_text(var) for var in variables_de]
        var_dict = dict(zip(variables_de, variables_en))
        # Use deepl to translate description from de to en
        description_de = list(set(df["Description"]))
        description_en = [translate_text(var) for var in description_de]
        cat_dict = dict(zip(description_de, description_en))

        df["Use"] = df["Variable"].map(var_dict)
        df["Energy sources"] = df["Description"].map(cat_dict)

        df.drop(["Variable", "Description"], axis=1, inplace=True)

        # Create database format
        lever = "energy_tmp"  # temporary lever name
        df["variables"] = df["Use"] + "_" + df["Energy sources"] + "[TJ]"
        df.drop(["Use", "Energy sources"], axis=1, inplace=True)
        df["level"] = 0
        df["lever"] = "energy_tmp"
        df["string-pivot"] = "none"
        df["geoscale"] = "Switzerland"

        # Replace empty strings with NaN
        df["value"].replace("", np.nan, inplace=True)

        dict_ots, dict_fts = database_to_dm(
            df, lever, num_cat=1, baseyear=baseyear, years=years_ots, level="all"
        )

        dm = dict_ots[lever]

        with open(outfile_dm, "wb") as handle:
            pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(outfile_dm, "rb") as handle:
            dm = pickle.load(handle)

    return dm


def run(baseyear, years_ots):
    energy_url = "https://www.uvek-gis.admin.ch/BFE/ogd/115/ogd115_gest_bilanz.csv"
    local_path = "data/Energy_statistic.csv"
    outfile_dm = "data/energy.pickle"

    # Extract energy data
    dm_energy = extract_energy_data(
        energy_url, local_path, baseyear, years_ots, outfile_dm
    )

    # change units from TJ to TWh
    for var in dm_energy.col_labels["Variables"]:
        dm_energy.change_unit(
            var, factor=3600, old_unit="TJ", new_unit="TWh", operator="/"
        )

    # Some visualisation
    plotting = False
    if plotting:
        dm_energy.datamatrix_plot(title="All energy variables in TWh")
        dm_energy.datamatrix_plot({"Variables": "Import"}, title="Import [TWh]")
        dm_energy.datamatrix_plot(
            {"Variables": "Energy conversion - Nuclear power plants"},
            title="Energy conversion - Nuclear power plants [TWh]",
        )
        dm_energy.datamatrix_plot(
            {
                "Variables": "Energy conversion - conventional-thermal power, district heating and district heating power plants"
            },
            title="Energy conversion - conventional-thermal power, district heating and district heating power plants [TWh]",
        )

    return dm_energy
