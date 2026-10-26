# from _database.pre_processing.api_routines_CH import get_data_api_CH
import pandas as pd
from get_data_functions.get_data import (
    get_losses_csl,
    get_producing_animals_1990_2022,
    get_slaughtered_1990_2022,
)
from get_data_functions.get_enteric_emissions import get_enteric_1990_2021
from get_data_functions.get_manure_emissions import get_manure_1990_2021
from get_data_functions.utils import ensure_structure

from transition_compass_model._database.pre_processing.params import years_ots
from transition_compass_model.model.common.auxiliary_functions import (
    linear_fitting_ots_db,
)


# CalculationLeaf CLIMATE SMART LIVESTOCK ------------------------------------------------------------------------------
def climate_smart_livestock_processing(
    df_feed_ration, df_liv_pop, df_cropland_density, list_countries, file_dict
):
    # ----------------------------------------------------------------------------------------------------------------------
    # LIVESTOCK DENSITY & GRAZING INTENSITY ---------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # Filter grazing ruminant livestock (cattle meat, sheep, goats) and sum per year
    df_ruminant = df_liv_pop[
        df_liv_pop["Item"].isin(
            ["Cattle, dairy", "Cattle, non-dairy", "Sheep and Goats"]
        )
    ]
    df_ruminant = df_ruminant.groupby(["Area", "Year"], as_index=False)["Value"].sum()

    df_ruminant["Year"] = df_ruminant["Year"].astype(int)
    # Merge with cropland_density
    df_ruminant = pd.merge(df_ruminant, df_cropland_density, on=["Area", "Year"])

    # Compute livestock density of ruminant per area of permanent meadows and pastures
    df_ruminant["Livestock density [lsu/ha]"] = (
        df_ruminant["Value"] / df_ruminant["Permanent meadows and pastures"]
    )

    # Filter and add column density
    df_ruminant = df_ruminant[["Year", "Area", "Livestock density [lsu/ha]"]]

    # Adding an Item column for name
    df_ruminant["Item"] = "Density"

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------

    # Renaming into 'Value'
    df_ruminant.rename(
        columns={
            "Area": "geoscale",
            "Year": "timescale",
            "Livestock density [lsu/ha]": "value",
        },
        inplace=True,
    )

    # Read excel file
    df_dict_csl = pd.read_excel(
        "data/dictionaries/dictionnary_agriculture_landuse.xlsx",
        sheet_name="climate-smart-livestock",
    )

    # Merge based on 'Item'
    df_csl_density_pathwaycalc = pd.merge(df_dict_csl, df_ruminant, on="Item")

    # Drop the 'Item' column
    df_csl_density_pathwaycalc = df_csl_density_pathwaycalc.drop(columns=["Item"])

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_csl_density_pathwaycalc["module"] = "agriculture"
    df_csl_density_pathwaycalc["lever"] = "climate-smart-livestock"
    df_csl_density_pathwaycalc["level"] = 0
    cols = df_csl_density_pathwaycalc.columns.tolist()
    cols.insert(cols.index("value"), cols.pop(cols.index("module")))
    cols.insert(cols.index("value"), cols.pop(cols.index("lever")))
    cols.insert(cols.index("value"), cols.pop(cols.index("level")))
    df_csl_density_pathwaycalc = df_csl_density_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    df_csl_density_pathwaycalc["geoscale"] = df_csl_density_pathwaycalc[
        "geoscale"
    ].replace("United Kingdom of Great Britain and Northern Ireland", "United Kingdom")
    df_csl_density_pathwaycalc["geoscale"] = df_csl_density_pathwaycalc[
        "geoscale"
    ].replace("Netherlands (Kingdom of the)", "Netherlands")
    df_csl_density_pathwaycalc["geoscale"] = df_csl_density_pathwaycalc[
        "geoscale"
    ].replace("Czechia", "Czech Republic")

    # ----------------------------------------------------------------------------------------------------------------------
    # AGROFORESTRY (GRASSLAND & HEDGES) ------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    # Is equal to 0 for all ots for all countries

    # Use density (grouped_df) as a structural basis
    agroforestry_liv = df_ruminant.copy()

    # Drop the column Item
    agroforestry_liv = agroforestry_liv.drop(columns=["Item", "value"])

    # Rename the column in geoscale and timescale
    agroforestry_liv.rename(
        columns={"Area": "geoscale", "Year": "timescale"}, inplace=True
    )

    # Changing data type to numeric (except for the geoscale column)
    agroforestry_liv.loc[:, agroforestry_liv.columns != "geoscale"] = (
        agroforestry_liv.loc[:, agroforestry_liv.columns != "geoscale"].apply(
            pd.to_numeric, errors="coerce"
        )
    )

    # Add rows to have 1990-2022
    # Generate a DataFrame with all combinations of geoscale and timescale
    geoscale_values = agroforestry_liv["geoscale"].unique()
    timescale_values = pd.Series(range(1990, 2023))

    # Create a DataFrame for the cartesian product
    cartesian_product = pd.MultiIndex.from_product(
        [geoscale_values, timescale_values], names=["geoscale", "timescale"]
    ).to_frame(index=False)

    # Merge the original DataFrame with the cartesian product to include all combinations
    agroforestry_liv = pd.merge(
        cartesian_product, agroforestry_liv, on=["geoscale", "timescale"], how="left"
    )

    # Add the variables with a value of 0
    agroforestry_liv["agr_climate-smart-livestock_ef_agroforestry_grassland[tC/ha]"] = 0
    agroforestry_liv["agr_climate-smart-livestock_ef_agroforestry_hedges[tC/ha]"] = 0

    # Melt the df
    agroforestry_liv_pathwaycalc = pd.melt(
        agroforestry_liv,
        id_vars=["timescale", "geoscale"],
        value_vars=[
            "agr_climate-smart-livestock_ef_agroforestry_grassland[tC/ha]",
            "agr_climate-smart-livestock_ef_agroforestry_hedges[tC/ha]",
        ],
        var_name="variables",
        value_name="value",
    )

    # PathwayCalc formatting --------------------------------------------------------------------------------------------
    agroforestry_liv_pathwaycalc["module"] = "agriculture"
    agroforestry_liv_pathwaycalc["lever"] = "climate-smart-livestock"
    agroforestry_liv_pathwaycalc["level"] = 0
    cols = agroforestry_liv_pathwaycalc.columns.tolist()
    cols.insert(cols.index("value"), cols.pop(cols.index("module")))
    cols.insert(cols.index("value"), cols.pop(cols.index("lever")))
    cols.insert(cols.index("value"), cols.pop(cols.index("level")))
    cols.insert(cols.index("timescale"), cols.pop(cols.index("variables")))
    agroforestry_liv_pathwaycalc = agroforestry_liv_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    agroforestry_liv_pathwaycalc["geoscale"] = agroforestry_liv_pathwaycalc[
        "geoscale"
    ].replace("United Kingdom of Great Britain and Northern Ireland", "United Kingdom")
    agroforestry_liv_pathwaycalc["geoscale"] = agroforestry_liv_pathwaycalc[
        "geoscale"
    ].replace("Netherlands (Kingdom of the)", "Netherlands")
    agroforestry_liv_pathwaycalc["geoscale"] = agroforestry_liv_pathwaycalc[
        "geoscale"
    ].replace("Czechia", "Czech Republic")

    # ----------------------------------------------------------------------------------------------------------------------
    # ENTERIC EMISSIONS ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    df_enteric_1990_2021 = get_enteric_1990_2021(file_dict, list_countries)
    # Reading excel lsu equivalent
    df_lsu = pd.read_excel(
        "data/dictionaries/lsu_equivalent.xlsx", sheet_name="lsu_equivalent"
    )
    # Merging
    df_enteric_1990_2021 = pd.merge(df_enteric_1990_2021, df_lsu, on="Item")

    # Converting Animals to lsu
    condition = df_enteric_1990_2021["Unit"] == "An"
    df_enteric_1990_2021.loc[condition, "Value"] *= df_enteric_1990_2021["lsu"]

    # Aggregating
    df_enteric_1990_2021_grouped = df_enteric_1990_2021.groupby(
        ["Aggregation", "Area", "Year", "Element", "Unit"], as_index=False
    )["Value"].sum()

    # Pivot the df
    pivot_df = df_enteric_1990_2021_grouped.pivot_table(
        index=["Area", "Year", "Aggregation"], columns="Element", values="Value"
    ).reset_index()

    # Enteric emissions CH4 [t/lsu] = 1000 * 'Enteric fermentation (Emissions CH4) [kt]'/ 'Stocks [lsu]'
    pivot_df["Enteric emissions CH4 [t/lsu]"] = (
        1000 * pivot_df["Enteric fermentation (Emissions CH4)"] / pivot_df["Stocks"]
    )

    # Create duplicate for fxa
    df_manure_ch4_fxa = pivot_df.copy()
    df_manure_ch4_fxa["Manure emissions CH4 [t/lsu]"] = (
        1000
        * df_manure_ch4_fxa["Manure management (Emissions CH4)"]
        / df_manure_ch4_fxa["Stocks"]
    )
    df_manure_ch4_fxa = df_manure_ch4_fxa[
        ["Area", "Year", "Aggregation", "Manure emissions CH4 [t/lsu]"]
    ].copy()

    # Drop the columns 'Enteric fermentation (Emissions CH4)' 'Stocks'
    pivot_df = pivot_df.drop(
        columns=[
            "Enteric fermentation (Emissions CH4)",
            "Manure management (Emissions CH4)",
            "Stocks",
        ]
    )

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------

    # Renaming into 'Value'
    pivot_df.rename(
        columns={
            "Area": "geoscale",
            "Year": "timescale",
            "Enteric emissions CH4 [t/lsu]": "value",
        },
        inplace=True,
    )

    # Food item name matching with dictionary
    # Read excel file
    df_dict_csl_enteric = pd.read_excel(
        "data/dictionaries/dictionnary_agriculture_landuse.xlsx",
        sheet_name="climate-smart-livestock_enteric",
    )

    # Merge based on 'Item' & 'Aggregation'
    df_enteric_pathwaycalc = pd.merge(
        df_dict_csl_enteric, pivot_df, left_on="Item", right_on="Aggregation"
    )

    # Drop the 'Item' column
    df_enteric_pathwaycalc = df_enteric_pathwaycalc.drop(
        columns=["Item", "Aggregation"]
    )

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_enteric_pathwaycalc["module"] = "agriculture"
    df_enteric_pathwaycalc["lever"] = "climate-smart-livestock"
    df_enteric_pathwaycalc["level"] = 0
    cols = df_enteric_pathwaycalc.columns.tolist()
    cols.insert(cols.index("value"), cols.pop(cols.index("module")))
    cols.insert(cols.index("value"), cols.pop(cols.index("lever")))
    cols.insert(cols.index("value"), cols.pop(cols.index("level")))
    df_enteric_pathwaycalc = df_enteric_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    df_enteric_pathwaycalc["geoscale"] = df_enteric_pathwaycalc["geoscale"].replace(
        "United Kingdom of Great Britain and Northern Ireland", "United Kingdom"
    )
    df_enteric_pathwaycalc["geoscale"] = df_enteric_pathwaycalc["geoscale"].replace(
        "Netherlands (Kingdom of the)", "Netherlands"
    )
    df_enteric_pathwaycalc["geoscale"] = df_enteric_pathwaycalc["geoscale"].replace(
        "Czechia", "Czech Republic"
    )

    # ----------------------------------------------------------------------------------------------------------------------
    # MANURE EMISSIONS (APPLIED, PASTURE & TREATED) ------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    df_manure_1990_2021 = get_manure_1990_2021(file_dict, list_countries)

    # Reading excel lsu equivalent (for aggregation)
    df_lsu = pd.read_excel(
        "data/dictionaries/lsu_equivalent.xlsx", sheet_name="lsu_equivalent"
    )
    # Merging
    df_manure_1990_2021 = pd.merge(df_manure_1990_2021, df_lsu, on="Item")

    # Aggregating
    df_manure_1990_2021 = df_manure_1990_2021.groupby(
        ["Aggregation", "Area", "Year", "Element", "Unit"], as_index=False
    )["Value"].sum()

    # Pivot the df
    pivot_df = df_manure_1990_2021.pivot_table(
        index=["Area", "Year", "Aggregation"], columns="Element", values="Value"
    ).reset_index()

    # Create copy for manure_fxa()
    df_manure_n_fxa = pivot_df.copy()

    # Merge with df_liv_pop
    # Rename for merge (df_liv_pop => pivot_df_slau (meat) or df_slau_eggs_milk (eggs,dairy))
    terms = {
        "Cattle, dairy": "Dairy-milk",
        "Cattle, non-dairy": "Bovine",
        "Chickens, layers": "Hens-egg",
        "Sheep and Goats": "Sheep",
        "Swine": "Pig",
        "Others Stocks": "Other animal",
        "Poultry Stocks": "Poultry",
    }

    # Apply the replacement
    df_liv_pop["Item"] = df_liv_pop["Item"].replace(terms)

    df_liv_pop["Year"] = df_liv_pop["Year"].astype(int)
    # Merge with stock from df_liv_pop
    pivot_df = pd.merge(
        pivot_df,
        df_liv_pop,
        left_on=["Area", "Year", "Aggregation"],
        right_on=["Area", "Year", "Item"],
        how="inner",
    )

    # Manure applied/treated/pasture [%] = Manure applied to soil/treated/left on pasture (N content) [kg] / Amount excreted (N content) [kg]

    pivot_df["Manure applied [%]"] = (
        pivot_df["Manure applied to soils (N content)"]
        / pivot_df["Amount excreted in manure (N content)"]
    )
    pivot_df["Manure treated [%]"] = (
        pivot_df["Manure management (manure treated, N content)"]
        / pivot_df["Amount excreted in manure (N content)"]
    )
    pivot_df["Manure pasture [%]"] = (
        pivot_df["Manure left on pasture (N content)"]
        / pivot_df["Amount excreted in manure (N content)"]
    )

    # Compute manure yield fxa
    pivot_df["Manure yield [tN/lsu]"] = (
        10**-3 * pivot_df["Amount excreted in manure (N content)"] / pivot_df["Value"]
    )

    # Create copy for emission factor per practice

    # Drop the columns
    pivot_df = pivot_df.drop(
        columns=[
            "Manure applied to soils (N content)",
            "Manure management (manure treated, N content)",
            "Manure left on pasture (N content)",
            "Amount excreted in manure (N content)",
            "Value",
            "Item",
        ]
    )

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------

    # Melt the DataFrame
    df_melted = pd.melt(
        pivot_df,
        id_vars=["Area", "Year", "Aggregation"],
        value_vars=[
            "Manure applied [%]",
            "Manure treated [%]",
            "Manure pasture [%]",
            "Manure yield [tN/lsu]",
        ],
        var_name="Item",
        value_name="value",
    )

    # Concatenate the aggregation column with the manure column names
    df_melted["Item"] = df_melted["Aggregation"] + " " + df_melted["Item"]

    # Drop the aggregation column as it's now part of the item column
    df_melted = df_melted.drop(columns=["Aggregation"])

    # Renaming
    df_melted.rename(columns={"Area": "geoscale", "Year": "timescale"}, inplace=True)

    # Food item name matching with dictionary
    # Read excel file
    df_dict_csl = pd.read_excel(
        "data/dictionaries/dictionnary_agriculture_landuse.xlsx",
        sheet_name="climate-smart-livestock",
    )

    # Merge based on 'Item' & 'Aggregation'
    df_manure_pathwaycalc = pd.merge(df_dict_csl, df_melted, on="Item")

    # Drop the 'Item' column
    df_manure_pathwaycalc = df_manure_pathwaycalc.drop(columns=["Item"])

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_manure_pathwaycalc["module"] = "agriculture"
    df_manure_pathwaycalc["lever"] = "climate-smart-livestock"
    df_manure_pathwaycalc["level"] = 0
    cols = df_manure_pathwaycalc.columns.tolist()
    cols.insert(cols.index("value"), cols.pop(cols.index("module")))
    cols.insert(cols.index("value"), cols.pop(cols.index("lever")))
    cols.insert(cols.index("value"), cols.pop(cols.index("level")))
    df_manure_pathwaycalc = df_manure_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    df_manure_pathwaycalc["geoscale"] = df_manure_pathwaycalc["geoscale"].replace(
        "United Kingdom of Great Britain and Northern Ireland", "United Kingdom"
    )
    df_manure_pathwaycalc["geoscale"] = df_manure_pathwaycalc["geoscale"].replace(
        "Netherlands (Kingdom of the)", "Netherlands"
    )
    df_manure_pathwaycalc["geoscale"] = df_manure_pathwaycalc["geoscale"].replace(
        "Czechia", "Czech Republic"
    )

    # Filter for fxa
    df_csl_fxa = df_manure_pathwaycalc[
        df_manure_pathwaycalc["variables"].str.contains("fxa", case=False, na=False)
    ]

    # Drop the rows from original df
    df_manure_pathwaycalc = df_manure_pathwaycalc[
        ~df_manure_pathwaycalc["variables"].str.contains("fxa", case=False, na=False)
    ]

    # ----------------------------------------------------------------------------------------------------------------------
    # LOSSES ---------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # FOOD BALANCE SHEETS (FBS) - For everything  -------------------------------------------------
    df_losses_csl = get_losses_csl(file_dict, list_countries)

    # Compute losses ([%] of production) -----------------------------------------------------------------------------------
    # Losses [%] = 1 / (1 - Losses [1000t] / Production [1000t]) (pre processing for multiplicating the workflow)

    # Step 1: Pivot the DataFrame
    pivot_df = df_losses_csl.pivot_table(
        index=["Area", "Year", "Item"], columns="Element", values="Value"
    ).reset_index()

    # Replace NaN with 0
    pivot_df["Losses"].fillna(0.0, inplace=True)

    # Step 2: Compute the Losses [%] (really it's unit less)
    pivot_df["Losses[%]"] = 1 + (pivot_df["Losses"] / pivot_df["Production"])

    # Drop the columns Production, Import Quantity and Export Quantity
    pivot_df = pivot_df.drop(columns=["Production", "Losses"])

    # Extrapolating for 2022 -----------------------------------------------------------------------------------------------

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------

    # Food item name matching with dictionary
    # Read excel file
    df_dict_csl_losses = pd.read_excel(
        "data/dictionaries/dictionnary_agriculture_landuse.xlsx",
        sheet_name="climate-smart-livestock_losses",
    )

    # Merge based on 'Item'
    df_losses_csl_pathwaycalc = pd.merge(df_dict_csl_losses, pivot_df, on="Item")

    # Drop the 'Item' column
    df_losses_csl_pathwaycalc = df_losses_csl_pathwaycalc.drop(columns=["Item"])

    # Renaming existing columns (geoscale, timsecale, value)
    df_losses_csl_pathwaycalc.rename(
        columns={"Area": "geoscale", "Year": "timescale", "Losses[%]": "value"},
        inplace=True,
    )

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_losses_csl_pathwaycalc["module"] = "agriculture"
    df_losses_csl_pathwaycalc["lever"] = "climate-smart-livestock"
    df_losses_csl_pathwaycalc["level"] = 0
    cols = df_losses_csl_pathwaycalc.columns.tolist()
    cols.insert(cols.index("value"), cols.pop(cols.index("module")))
    cols.insert(cols.index("value"), cols.pop(cols.index("lever")))
    cols.insert(cols.index("value"), cols.pop(cols.index("level")))
    df_losses_csl_pathwaycalc = df_losses_csl_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    df_losses_csl_pathwaycalc["geoscale"] = df_losses_csl_pathwaycalc[
        "geoscale"
    ].replace("United Kingdom of Great Britain and Northern Ireland", "United Kingdom")
    df_losses_csl_pathwaycalc["geoscale"] = df_losses_csl_pathwaycalc[
        "geoscale"
    ].replace("Netherlands (Kingdom of the)", "Netherlands")
    df_losses_csl_pathwaycalc["geoscale"] = df_losses_csl_pathwaycalc[
        "geoscale"
    ].replace("Czechia", "Czech Republic")

    # ----------------------------------------------------------------------------------------------------------------------
    # FEED RATION ----------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------

    # Fill nan with zeros
    df_feed_ration["Feed"].fillna(0, inplace=True)

    # Add a column with the total feed (per country and year)
    df_feed_ration["Total feed"] = df_feed_ration.groupby(["Area", "Year"])[
        "Feed"
    ].transform("sum")

    # Feed ration [%] = Feed from item i / Total feed
    df_feed_ration["Feed ratio"] = df_feed_ration["Feed"] / df_feed_ration["Total feed"]

    # Drop columns
    df_feed_ration = df_feed_ration.drop(columns=["Feed", "Total feed"])

    # For Switzerland add Fruits = 0%
    # Duplicate rows only where Item = 'Pulses' and Area = 'Switzerland'
    duplicated_rows = df_feed_ration[
        (df_feed_ration["Item"] == "Pulses") & (df_feed_ration["Area"] == "Switzerland")
    ].copy()
    # Modify the duplicated rows
    duplicated_rows["Item"] = "Fruits - Excluding Wine"
    duplicated_rows["Feed ratio"] = 0.0
    # Concatenate back to the main DataFrame
    df_feed_ration = pd.concat([df_feed_ration, duplicated_rows], ignore_index=True)

    # For Switzerland add Vegetable oils = 0%
    # Duplicate rows only where Item = 'Pulses' and Area = 'Switzerland'
    duplicated_rows = df_feed_ration[
        (df_feed_ration["Item"] == "Pulses") & (df_feed_ration["Area"] == "Switzerland")
    ].copy()
    # Modify the duplicated rows
    duplicated_rows["Item"] = "Vegetable Oils"
    duplicated_rows["Feed ratio"] = 0.0
    # Concatenate back to the main DataFrame
    df_feed_ration = pd.concat([df_feed_ration, duplicated_rows], ignore_index=True)

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------

    # Renaming into 'Value'
    df_feed_ration.rename(
        columns={"Area": "geoscale", "Year": "timescale", "Feed ratio": "value"},
        inplace=True,
    )

    # Read excel file
    df_dict_csl = pd.read_excel(
        "data/dictionaries/dictionnary_agriculture_landuse.xlsx",
        sheet_name="climate-smart-livestock",
    )

    # Merge based on 'Item'
    df_csl_feed_pathwaycalc = pd.merge(df_dict_csl, df_feed_ration, on="Item")

    # Drop the 'Item' column
    df_csl_feed_pathwaycalc = df_csl_feed_pathwaycalc.drop(columns=["Item"])

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_csl_feed_pathwaycalc["module"] = "agriculture"
    df_csl_feed_pathwaycalc["lever"] = "climate-smart-livestock"
    df_csl_feed_pathwaycalc["level"] = 0
    cols = df_csl_feed_pathwaycalc.columns.tolist()
    cols.insert(cols.index("value"), cols.pop(cols.index("module")))
    cols.insert(cols.index("value"), cols.pop(cols.index("lever")))
    cols.insert(cols.index("value"), cols.pop(cols.index("level")))
    df_csl_feed_pathwaycalc = df_csl_feed_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    df_csl_feed_pathwaycalc["geoscale"] = df_csl_feed_pathwaycalc["geoscale"].replace(
        "United Kingdom of Great Britain and Northern Ireland", "United Kingdom"
    )
    df_csl_feed_pathwaycalc["geoscale"] = df_csl_feed_pathwaycalc["geoscale"].replace(
        "Netherlands (Kingdom of the)", "Netherlands"
    )
    df_csl_feed_pathwaycalc["geoscale"] = df_csl_feed_pathwaycalc["geoscale"].replace(
        "Czechia", "Czech Republic"
    )

    # ----------------------------------------------------------------------------------------------------------------------
    # YIELD (DAIRY & EGGS) -------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    df_producing_animals_1990_2022 = get_producing_animals_1990_2022(
        file_dict, list_countries
    )

    # Reading excel lsu equivalent
    df_lsu = pd.read_excel(
        "data/dictionaries/lsu_equivalent.xlsx", sheet_name="lsu_equivalent"
    )
    # Merging
    df_producing_animals_1990_2022 = pd.merge(
        df_producing_animals_1990_2022, df_lsu, on="Item"
    )

    # Converting Animals to lsu
    condition = (df_producing_animals_1990_2022["Unit"] == "An") | (
        df_producing_animals_1990_2022["Unit"] == "1000 An"
    )
    df_producing_animals_1990_2022.loc[condition, "Value"] *= (
        df_producing_animals_1990_2022["lsu"]
    )

    # Aggregating
    grouped_df = df_producing_animals_1990_2022.groupby(
        ["Aggregation", "Area", "Year", "Element", "Unit"], as_index=False
    )["Value"].sum()

    # Pivot the df
    pivot_df = grouped_df.pivot_table(
        index=["Area", "Year", "Aggregation"], columns="Element", values="Value"
    ).reset_index()

    # "Merging" the columns 'Laying' and 'Milk Animals' into 'Producing Animals'
    # Replace NaN with 0
    pivot_df["Laying"].fillna(0, inplace=True)
    pivot_df["Milk Animals"].fillna(0, inplace=True)

    # Sum the columns to create the 'Producing Animals' column
    pivot_df["Producing Animals"] = pivot_df["Laying"] + pivot_df["Milk Animals"]

    # Yield [t/lsu] = Production quantity / Producing animals/Slaugthered NOW done after using cal values
    pivot_df["Yield [t/lsu]"] = pivot_df["Producing Animals"]
    # pivot_df['Yield [t/lsu]'] = pivot_df['Production'] / pivot_df['Producing Animals']

    # Create a copy
    df_slau_eggs_milk = pivot_df.copy()
    df_slau_eggs_milk = df_slau_eggs_milk.drop(
        columns=["Laying", "Milk Animals", "Production", "Yield [t/lsu]"]
    )

    # Drop the columns to only have Yield and Slaughter rate
    pivot_df = pivot_df.drop(
        columns=["Laying", "Milk Animals", "Production", "Producing Animals"]
    )

    # ----------------------------------------------------------------------------------------------------------------------
    # YIELD (MEAT) --------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    df_slaughtered_1990_2022 = get_slaughtered_1990_2022(file_dict, list_countries)

    # Reading excel lsu equivalent
    df_lsu = pd.read_excel(
        "data/dictionaries/lsu_equivalent.xlsx", sheet_name="lsu_equivalent"
    )
    # Merging
    df_slaughtered_1990_2022 = pd.merge(df_slaughtered_1990_2022, df_lsu, on="Item")

    # Converting Animals to lsu
    condition = (df_slaughtered_1990_2022["Unit"] == "An") | (
        df_slaughtered_1990_2022["Unit"] == "1000 An"
    )
    df_slaughtered_1990_2022.loc[condition, "Value"] *= df_slaughtered_1990_2022["lsu"]

    # Aggregating
    grouped_df = df_slaughtered_1990_2022.groupby(
        ["Aggregation", "Area", "Year", "Element", "Unit"], as_index=False
    )["Value"].sum()

    # Pivot the df
    pivot_df_slau = grouped_df.pivot_table(
        index=["Area", "Year", "Aggregation"], columns="Element", values="Value"
    ).reset_index()

    # Replace NaN with 0
    pivot_df_slau["Producing Animals/Slaughtered"].fillna(0, inplace=True)
    pivot_df_slau["Production"].fillna(0, inplace=True)

    # Create a copy for slau rate
    df_slau_meat = pivot_df_slau.copy()

    # Yield [t/lsu] = Production quantity / Producing animals/Slaugthered NOW DONE AFTER using cal values
    pivot_df_slau["Yield [t/lsu]"] = pivot_df_slau["Producing Animals/Slaughtered"]
    # pivot_df_slau['Yield [t/lsu]'] = pivot_df_slau['Production'] / pivot_df_slau['Producing Animals/Slaughtered']

    # Drop the columns
    pivot_df_slau = pivot_df_slau.drop(
        columns=["Producing Animals/Slaughtered", "Production"]
    )

    # Replace NaN with 0
    pivot_df_slau["Yield [t/lsu]"].fillna(0, inplace=True)

    # ----------------------------------------------------------------------------------------------------------------------
    # SLAUGHTERED RATE (MEAT, EGGS & MILK) --------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # Concat df_slau_meat (meat) and df_slau_eggs_milk (eggs,dairy)
    df_slau_meat.rename(
        columns={"Producing Animals/Slaughtered": "Producing Animals"}, inplace=True
    )
    df_slau_meat = df_slau_meat.drop(columns=["Production"])
    df_slau = pd.concat([df_slau_meat, df_slau_eggs_milk], ignore_index=True)

    # Rename for merge (df_liv_pop => pivot_df_slau (meat) or df_slau_eggs_milk (eggs,dairy))
    terms = {
        "Cattle, dairy": "Dairy-milk",
        "Cattle, non-dairy": "Bovine",
        "Chickens, layers": "Hens-egg",
        "Sheep and Goats": "Sheep",
        "Swine": "Pig",
        "Others Stocks": "Other animal",
        "Poultry Stocks": "Poultry",
    }

    # Apply the replacement
    df_liv_pop["Item"] = df_liv_pop["Item"].replace(terms)

    # Merge with stock from df_liv_pop
    df_slau = pd.merge(
        df_slau,
        df_liv_pop,
        left_on=["Area", "Year", "Aggregation"],
        right_on=["Area", "Year", "Item"],
        how="inner",
    )

    # Slaughtered animals [%] = 'Producing Animals/Slaughtered' / 'Value' (value = stocks [lsu])
    df_slau["Slaughtered animals [%]"] = df_slau["Producing Animals"] / df_slau["Value"]
    df_slau["Slaughtered animals [%]"].fillna(0, inplace=True)

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------

    # Separating between slaugthered animals and yield (for meat)
    df_yield_meat = pivot_df_slau[["Area", "Year", "Aggregation", "Yield [t/lsu]"]]
    df_slau_meat = df_slau[["Area", "Year", "Aggregation", "Slaughtered animals [%]"]]

    # Creating copies
    df_yield_meat = df_yield_meat.copy()
    df_slau_meat = df_slau_meat.copy()

    # Renaming into 'Value'
    df_yield_meat.rename(
        columns={"Area": "geoscale", "Year": "timescale", "Yield [t/lsu]": "value"},
        inplace=True,
    )
    pivot_df.rename(
        columns={"Area": "geoscale", "Year": "timescale", "Yield [t/lsu]": "value"},
        inplace=True,
    )
    df_slau_meat.rename(
        columns={
            "Area": "geoscale",
            "Year": "timescale",
            "Slaughtered animals [%]": "value",
        },
        inplace=True,
    )

    # Concatenating yield (meat, milk & eggs)
    df_yield_liv = pd.concat([df_yield_meat, pivot_df])

    # Read excel
    df_kcal_t = pd.read_excel(
        "data/dictionaries/kcal_to_t.xlsx", sheet_name="kcal_per_100g"
    )
    df_kcal_g = df_kcal_t[["Item livestock yield", "kcal per t"]]
    # Merge
    merged_df = pd.merge(
        df_kcal_g,
        df_yield_liv,  # Only keep the needed columns
        left_on=["Item livestock yield"],
        right_on=["Aggregation"],
    )
    # Operation Unit conversion t => kcal (not necessary since it's the producing animals now)
    # merged_df['value'] = merged_df['value'] * merged_df['kcal per t']
    df_yield_liv = merged_df[["geoscale", "timescale", "Aggregation", "value"]]
    df_yield_liv = df_yield_liv.copy()

    # Food item name matching with dictionary
    # Read excel file
    df_dict_csl_yield = pd.read_excel(
        "data/dictionaries/dictionnary_agriculture_landuse.xlsx",
        sheet_name="climate-smart-livestock_yield",
    )
    df_dict_csl_slau = pd.read_excel(
        "data/dictionaries/dictionnary_agriculture_landuse.xlsx",
        sheet_name="climate-smart-livestock_slau",
    )

    # Merge based on 'Item'
    df_yield_liv_pathwaycalc = pd.merge(
        df_dict_csl_yield, df_yield_liv, left_on="Item", right_on="Aggregation"
    )
    df_slau_liv_pathwaycalc = pd.merge(
        df_dict_csl_slau, df_slau_meat, left_on="Item", right_on="Aggregation"
    )

    # Drop the 'Item' column
    df_yield_liv_pathwaycalc = df_yield_liv_pathwaycalc.drop(
        columns=["Item", "Aggregation"]
    )
    df_slau_liv_pathwaycalc = df_slau_liv_pathwaycalc.drop(
        columns=["Item", "Aggregation"]
    )

    # Concatenating yield and slau
    df_yield_slau_liv_pathwaycalc = pd.concat(
        [df_yield_liv_pathwaycalc, df_slau_liv_pathwaycalc]
    )

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_yield_slau_liv_pathwaycalc["module"] = "agriculture"
    df_yield_slau_liv_pathwaycalc["lever"] = "climate-smart-livestock"
    df_yield_slau_liv_pathwaycalc["level"] = 0
    cols = df_yield_slau_liv_pathwaycalc.columns.tolist()
    cols.insert(cols.index("value"), cols.pop(cols.index("module")))
    cols.insert(cols.index("value"), cols.pop(cols.index("lever")))
    cols.insert(cols.index("value"), cols.pop(cols.index("level")))
    df_yield_pathwaycalc = df_yield_slau_liv_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    df_yield_slau_liv_pathwaycalc["geoscale"] = df_yield_slau_liv_pathwaycalc[
        "geoscale"
    ].replace("United Kingdom of Great Britain and Northern Ireland", "United Kingdom")
    df_yield_slau_liv_pathwaycalc["geoscale"] = df_yield_slau_liv_pathwaycalc[
        "geoscale"
    ].replace("Netherlands (Kingdom of the)", "Netherlands")
    df_yield_slau_liv_pathwaycalc["geoscale"] = df_yield_slau_liv_pathwaycalc[
        "geoscale"
    ].replace("Czechia", "Czech Republic")

    # ----------------------------------------------------------------------------------------------------------------------
    # FINAL RESULTS --------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    df_csl = pd.concat([df_csl_density_pathwaycalc, df_enteric_pathwaycalc])
    df_csl = pd.concat([df_csl, df_manure_pathwaycalc])
    df_csl = pd.concat([df_csl, df_losses_csl_pathwaycalc])
    df_csl = pd.concat([df_csl, df_csl_feed_pathwaycalc])
    df_csl = pd.concat([df_csl, df_yield_slau_liv_pathwaycalc])
    df_csl = pd.concat([df_csl, agroforestry_liv_pathwaycalc])
    df_csl = df_csl.drop_duplicates()

    # Extrapolating
    df_climate_smart_livestock_pathwaycalc = ensure_structure(df_csl)
    df_climate_smart_livestock_pathwaycalc = linear_fitting_ots_db(
        df_climate_smart_livestock_pathwaycalc, years_ots, countries="all"
    )

    return (
        df_climate_smart_livestock_pathwaycalc,
        df_csl_fxa,
        df_manure_n_fxa,
        df_manure_ch4_fxa,
    )
