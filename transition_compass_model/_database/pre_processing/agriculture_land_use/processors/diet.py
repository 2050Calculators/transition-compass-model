import re

# from _database.pre_processing.api_routines_CH import get_data_api_CH
import pandas as pd
from get_data_functions.get_diet import get_diet_faostat
from get_data_functions.utils import ensure_structure

from transition_compass_model._database.pre_processing.params import years_ots
from transition_compass_model.model.common.auxiliary_functions import (
    linear_fitting_ots_db,
)


# CalculationLeaf DIET (LIFESTYLE) ------------------------------------------------------------------------------------
def diet_processing(list_countries, file):
    # ----------------------------------------------------------------------------------------------------------------------
    # CONSUMER DIET Part 1 - including food waste
    # ----------------------------------------------------------------------------------------------------------------------

    # Read data ------------------------------------------------------------------------------------------------------------

    df_diet = get_diet_faostat(file, list_countries)

    # Filter to only have meat, fish, sugar, sweet, veg, fruits (the rest goes to share for further pre-processing)
    list_consumers_diet = [
        "Fruits - Excluding Wine",
        "Vegetables",
        "Demersal Fish",
        "Freshwater Fish",
        "Aquatic Animals, Others",
        "Pelagic Fish",
        "Sugar (Raw Equivalent)",
        "Sweeteners, Other",
        "Bovine Meat",
        "Meat, Other",
        "Pigmeat",
        "Poultry Meat",
        "Mutton & Goat Meat",
        "Fish, Seafood",
    ]
    pattern_consumers_diet = "|".join(re.escape(item) for item in list_consumers_diet)
    list_share = [
        "Cereals - Excluding Beer",
        "Oilcrops",
        "Pulses",
        "Rice and products",
        "Starchy Roots",
        "Beer",
        "Beverages, Alcoholic",
        "Beverages, Fermented",
        "Wine",
        "Vegetable Oils",
        "Milk - Excluding Butter",
        "Eggs",
        "Animal fats",
        "Offals",
        "Coffee and products",
        "Cocoa Beans and products",
        "Tea (including mate)",
        "Grand Total",
    ]
    pattern_share = "|".join(re.escape(item) for item in list_share)
    df_consumers_diet = df_diet[
        df_diet["Item"].str.contains(pattern_consumers_diet, case=False)
    ]
    df_share = df_diet[df_diet["Item"].str.contains(pattern_share, case=False)]

    # Pivot the df
    pivot_df_consumers_diet = df_consumers_diet.pivot_table(
        index=["Area", "Year", "Item"], columns="Element", values="Value"
    ).reset_index()
    # Rename columns
    # pivot_df_consumers_diet.rename(columns={'Food': 'value'}, inplace=True)

    # ----------------------------------------------------------------------------------------------------------------------
    # SHARE (FOR OTHER PRODUCTS)
    # ----------------------------------------------------------------------------------------------------------------------

    # Pivot the df
    pivot_df_share = df_share.pivot_table(
        index=["Area", "Year", "Item"], columns="Element", values="Value"
    ).reset_index()

    # Note : this is now done after converting unit from kt to kcal
    """# Creating a column with the Grand Total kcal/cap/day
    # Step 1: Extract Grand Total for each Year and Area
    grand_totals = pivot_df_share[pivot_df_share['Item'] == 'Grand Total'][
        ['Area', 'Year', 'Food']]
    grand_totals.rename(columns={'Food': 'Grand Total'}, inplace=True)
    # Step 2: Merge the Grand Total back into the original DataFrame
    pivot_df_share = pivot_df_share.merge(grand_totals, on=['Area', 'Year'], how='left')
    # Step 3: Drop rows where Item is 'Grand Total'
    pivot_df_share = pivot_df_share[pivot_df_share['Item'] != 'Grand Total']

    # Divide the consumption by the total kcal to obtain the share
    #pivot_df_share['value'] = pivot_df_share['Food'] / pivot_df_share['Grand Total']
    """

    pivot_df_share["value"] = pivot_df_share["Food"]  # Test with Paola 28.05.2025'

    # Drop the columns
    pivot_df_share = pivot_df_share[["Area", "Year", "Item", "value"]]

    # Normalize so that for each year and country, sum(share) = 1
    # pivot_df_share['value'] = pivot_df_share['value'] / pivot_df_share.groupby(['Area', 'Year'])['value'].transform(
    #    'sum')

    # ----------------------------------------------------------------------------------------------------------------------
    # CONSUMER DIET Part 2 - including food waste
    # ----------------------------------------------------------------------------------------------------------------------

    # Food item name matching with dictionary
    # Read excel file
    df_dict_waste = pd.read_excel(
        "data/dictionaries/dictionnary_agriculture_landuse.xlsx",
        sheet_name="food-waste_lifestyle",
    )

    # Merge based on 'Item'
    pivot_df_consumers_diet = pd.merge(
        df_dict_waste, pivot_df_consumers_diet, on="Item"
    )

    # Diet [kcal/cap/day] = food supply [kcal/cap/day] * (1-food waste [%])
    pivot_df_consumers_diet["value"] = pivot_df_consumers_diet["Food"] * (
        1 - pivot_df_consumers_diet["Proportion"]
    )

    # Drop the unused columns
    pivot_df_consumers_diet = pivot_df_consumers_diet.drop(
        columns=["variables", "Food", "Proportion"]
    )

    # Concatenating consumer diet & share
    pivot_df_diet = pd.concat([pivot_df_consumers_diet, pivot_df_share])

    # PathwayCalc formatting -----------------------------------------------------------------------------------------------
    # Food item name matching with dictionary
    # Read excel file
    df_dict_diet = pd.read_excel(
        "data/dictionaries/dictionnary_agriculture_landuse.xlsx",
        sheet_name="diet_lifestyle",
    )

    # Merge based on 'Item'
    df_diet_pathwaycalc = pd.merge(df_dict_diet, pivot_df_diet, on="Item")

    # Drop the 'Item' column
    df_diet_pathwaycalc = df_diet_pathwaycalc.drop(columns=["Item"])

    # Renaming existing columns (geoscale, timsecale, value)
    df_diet_pathwaycalc.rename(
        columns={"Area": "geoscale", "Year": "timescale"}, inplace=True
    )

    # Adding the columns module, lever, level and string-pivot at the correct places
    df_diet_pathwaycalc["module"] = "agriculture"
    df_diet_pathwaycalc["lever"] = "diet"
    df_diet_pathwaycalc["level"] = 0
    cols = df_diet_pathwaycalc.columns.tolist()
    cols.insert(cols.index("value"), cols.pop(cols.index("module")))
    cols.insert(cols.index("value"), cols.pop(cols.index("lever")))
    cols.insert(cols.index("value"), cols.pop(cols.index("level")))
    df_diet_pathwaycalc = df_diet_pathwaycalc[cols]

    # Rename countries to Pathaywcalc name
    df_diet_pathwaycalc["geoscale"] = df_diet_pathwaycalc["geoscale"].replace(
        "United Kingdom of Great Britain and Northern Ireland", "United Kingdom"
    )
    df_diet_pathwaycalc["geoscale"] = df_diet_pathwaycalc["geoscale"].replace(
        "Netherlands (Kingdom of the)", "Netherlands"
    )
    df_diet_pathwaycalc["geoscale"] = df_diet_pathwaycalc["geoscale"].replace(
        "Czechia", "Czech Republic"
    )

    # Extrapolating
    df_diet_pathwaycalc = ensure_structure(df_diet_pathwaycalc)
    df_diet_pathwaycalc = linear_fitting_ots_db(
        df_diet_pathwaycalc, years_ots, countries="all"
    )
    return df_diet_pathwaycalc, df_diet
