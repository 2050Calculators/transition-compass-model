# packages
import os
import pickle
import warnings

warnings.simplefilter("ignore")
import plotly.io as pio

pio.renderers.default = "browser"
import eurostat
import numpy as np
import pandas as pd

from transition_compass_model.model.common.auxiliary_functions import create_years_list
from transition_compass_model.model.common.data_matrix_class import DataMatrix

# from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat


def make_calib_emissions_dm(current_file_directory, lever_files, years_ots, years_fts):
    #########################################
    ##### GET CLEAN DATAFRAME WITH DATA #####
    #########################################

    # get data — fetch from Eurostat once and cache locally as CSV
    filepath = os.path.join(
        current_file_directory, "../data/eurostat/env_ac_ainah_r2.csv"
    )
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        df = eurostat.get_data_df("env_ac_ainah_r2")
        df.to_csv(filepath, index=False)

    # get manufacturing and gases in tonnes
    df = df.loc[df["nace_r2"] == "C", :]
    df = df.loc[df["airpol"].isin(["CH4", "CO2", "N2O"]), :]
    df = df.loc[df["unit"] == "T", :]

    # fix country names
    countries_codes = [
        "AT",
        "BE",
        "BG",
        "HR",
        "CY",
        "CZ",
        "DK",
        "EE",
        "FI",
        "FR",
        "DE",
        "EL",
        "HU",
        "IE",
        "IT",
        "LV",
        "LT",
        "LU",
        "MT",
        "NL",
        "PL",
        "PT",
        "RO",
        "SK",
        "SI",
        "ES",
        "SE",
        "EU27_2020",
    ]
    df = df.loc[df["geo\\TIME_PERIOD"].isin(countries_codes), :]
    countries = [
        "Austria",
        "Belgium",
        "Bulgaria",
        "Croatia",
        "Cyprus",
        "Czech Republic",
        "Denmark",
        "Estonia",
        "Finland",
        "France",
        "Germany",
        "Greece",
        "Hungary",
        "Ireland",
        "Italy",
        "Latvia",
        "Lithuania",
        "Luxembourg",
        "Malta",
        "Netherlands",
        "Poland",
        "Portugal",
        "Romania",
        "Slovakia",
        "Slovenia",
        "Spain",
        "Sweden",
        "EU27",
    ]
    for i in range(0, len(countries_codes)):
        df.loc[df["geo\\TIME_PERIOD"] == countries_codes[i], "geo\\TIME_PERIOD"] = (
            countries[i]
        )
    len(df["geo\\TIME_PERIOD"].unique())

    # clean df
    df.drop(["freq", "nace_r2"], axis=1, inplace=True)
    df = pd.melt(df, id_vars=["geo\\TIME_PERIOD", "airpol", "unit"], var_name="year")
    df["year"] = [int(i) for i in df["year"]]
    df = df.loc[df["year"] >= 2008, :]  # most data is from 2008 onwards
    df["value"] = df["value"] / 1000000
    df["variable"] = ["calib-emissions_" + gas + "[Mt]" for gas in df["airpol"]]
    df.columns = ["Country", "gas", "unit", "Years", "value", "variable"]
    df = df.loc[:, ["Country", "Years", "variable", "value"]]
    df = df.pivot(
        index=["Country", "Years"], columns="variable", values="value"
    ).reset_index()

    # make data matrix
    dm = DataMatrix.create_from_df(df, 1)
    # Drop any years beyond the OTS range (Eurostat may include recent provisional data)
    _max_ots = max(int(y) for y in years_ots)
    _extra = [y for y in dm.col_labels["Years"] if int(y) > _max_ots]
    if _extra:
        dm.drop("Years", _extra)

    # add missing years as nan
    missing = np.array(years_ots)[
        [y not in dm.col_labels["Years"] for y in years_ots]
    ].tolist()
    dm.add(np.nan, "Years", missing, dummy=True)
    dm.sort("Years")
    dm.add(np.nan, "Years", years_fts, dummy=True)
    dm.sort("Years")
    # dm.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

    # get ammonia from fao
    filepath = os.path.join(
        current_file_directory,
        "../data/FAO/FAOSTAT_data_en_8-26-2025_fertilizer_manufacturing_EU27_aggregate.csv",
    )
    df = pd.read_csv(filepath)
    filepath = os.path.join(
        current_file_directory,
        "../data/FAO/FAOSTAT_data_en_8-26-2025_fertilizer_manufacturing_EU27_countries.csv",
    )
    df_temp = pd.read_csv(filepath)
    df = pd.concat([df, df_temp])
    df.columns
    df = df.loc[:, ["Area", "Item", "Element", "Year", "Unit", "Value"]]
    df["Item"] = "fertilizer"
    df.loc[df["Element"] == "Emissions (CO2)", "Element"] = "CO2"
    df.loc[df["Element"] == "Emissions (N2O)", "Element"] = "N2O"
    df["Item"] = df["Item"] + "_" + df["Element"] + "[" + df["Unit"] + "]"
    df = df.loc[:, ["Area", "Item", "Year", "Value"]]
    df = df.pivot(index=["Area", "Year"], columns="Item", values="Value").reset_index()
    df.rename(columns={"Area": "Country", "Year": "Years"}, inplace=True)
    countries = df["Country"].unique()
    years = df["Years"].unique()
    df_full = pd.DataFrame(
        {
            "Country": np.repeat(countries, len(years)),
            "Years": np.tile(years, len(countries)),
        }
    )
    df = pd.merge(df_full, df, how="left", on=["Country", "Years"])
    dm_fert = DataMatrix.create_from_df(df, 1)
    dm_fert.add(0, "Categories1", "CH4", "kt", True)
    dm_fert.sort("Categories1")
    dm_fert.change_unit("fertilizer", 1e-3, "kt", "Mt")
    dm_fert.rename_col(
        ["Czechia", "European Union (27)", "Netherlands (Kingdom of the)"],
        ["Czech Republic", "EU27", "Netherlands"],
        "Country",
    )
    dm_fert.sort("Country")
    country_missing = np.array(dm.col_labels["Country"])[
        [c not in dm_fert.col_labels["Country"] for c in dm.col_labels["Country"]]
    ].tolist()
    dm_fert.add(np.nan, "Country", country_missing, dummy=True)
    dm_fert.sort("Country")
    year_missing = np.array(dm.col_labels["Years"])[
        [c not in dm_fert.col_labels["Years"] for c in dm.col_labels["Years"]]
    ].tolist()
    dm_fert.add(np.nan, "Years", year_missing, dummy=True)
    dm_fert.sort("Years")
    dm_fert.rename_col("fertilizer", "calib-emissions", "Variables")
    # dm_fert.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

    # subtract fertilizer emissions from overall industry
    dm_temp = dm_fert.copy()
    dm_temp.array[np.isnan(dm_temp.array)] = 0
    dm.array = dm.array - dm_temp.array
    # dm.filter({"Country" : ["EU27"]}).flatten().datamatrix_plot()

    # save
    f = os.path.join(current_file_directory, "../data/datamatrix/" + lever_files[0])
    with open(f, "wb") as handle:
        pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save
    f = os.path.join(current_file_directory, "../data/datamatrix/" + lever_files[1])
    with open(f, "wb") as handle:
        pickle.dump(dm_fert, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm, dm_fert


def run(years_ots, years_fts):
    # directories
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # if exists, load, else make
    lever_files = [
        "calibration_emissions.pickle",
        "calibration_emissions_ammonia.pickle",
    ]
    filepaths = [
        os.path.join(current_file_directory, "../data/datamatrix/" + lever_file)
        for lever_file in lever_files
    ]
    true_condition = all([os.path.exists(filepath) for filepath in filepaths])
    if true_condition:
        with open(filepaths[0], "rb") as handle:
            dm = pickle.load(handle)
        with open(filepaths[1], "rb") as handle:
            dm_fert = pickle.load(handle)
    else:
        dm, dm_fert = make_calib_emissions_dm(
            current_file_directory, lever_files, years_ots, years_fts
        )

    return dm, dm_fert


if __name__ == "__main__":
    years_ots = create_years_list(1990, 2023, 1)
    years_fts = create_years_list(2025, 2050, 5)
    run(years_ots, years_fts)


# df = dm.write_df()
# df = df.loc[df["Country"] == "EU27",:]
# df_temp = pd.melt(df, id_vars = ['Country','Years'], var_name='variable')
# df_temp = df_temp.loc[df_temp["Years"] == 2021,:]
# name = "temp.xlsx"
# df_temp.to_excel("~/Desktop/" + name)
