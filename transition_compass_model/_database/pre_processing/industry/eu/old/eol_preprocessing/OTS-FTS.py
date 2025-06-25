from itertools import product
import pandas as pd
import numpy as np
import os
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/industry/eu/eol_preprocessing/OTS-FTS.py"

# import
current_file_directory = os.path.dirname(os.path.abspath(__file__))
saved_files_directory = os.path.join(current_file_directory, '../data/eol_intermediate_files')
df_pc = pd.read_excel(saved_files_directory + '/pc.xlsx') # computers & electronics
df_LA = pd.read_excel(saved_files_directory + '/LA.xlsx') # larger appliances
df_tv = pd.read_excel(saved_files_directory + '/tv.xlsx') # TV&PV
df_elv = pd.read_excel(saved_files_directory + '/elv.xlsx') # end-of-life vehicles

# checks
# subset1 = ["littered[%]","export[%]","waste-collected[%]","waste-uncollected[%]"]
# subset2 = ['recycling[%]', 'energy-recovery[%]','reuse[%]', 'landfill[%]', 'incineration[%]']
# def check(df, indexes, subset):

#     df_temp = pd.melt(df, indexes)
#     df_temp = df_temp.sort_values(indexes + ["variable"])
#     df_temp2 = df_temp.loc[df_temp["variable"].isin(subset),:]
#     df_temp2 = df_temp2.groupby(indexes, as_index=False)['value'].agg(sum)
    
#     return print(df_temp2["value"].unique())

# indexes = ["geoscale","timescale"]
# check(df_pc, indexes, subset1)
# check(df_pc, indexes, subset2)
# check(df_LA, indexes, subset1)
# check(df_LA, indexes, subset2)
# check(df_tv, indexes, subset1)
# check(df_tv, indexes, subset2)
# check(df_elv, indexes, subset1)
# check(df_elv, indexes, subset2)

# put in one dataset
indexes = ["geoscale","timescale"]
df_temp = pd.melt(df_pc, id_vars = indexes)
df_temp1 = df_temp.copy()
df_temp["variable"] = ["pc_" + i for i in df_temp["variable"]]
df_temp1["variable"] = ["phone_" + i for i in df_temp1["variable"]]
df_wst = pd.concat([df_temp, df_temp1])
df_temp = pd.melt(df_LA, id_vars = indexes)
df_temp["variable"] = ["domapp_" + i for i in df_temp["variable"]]
df_wst = pd.concat([df_wst, df_temp])
df_temp = pd.melt(df_tv, id_vars = indexes)
df_temp["variable"] = ["tv_" + i for i in df_temp["variable"]]
df_wst = pd.concat([df_wst, df_temp])
df_temp = pd.melt(df_elv, id_vars = indexes)
df_temp["variable"] = ["elv_" + i for i in df_temp["variable"]]
df_wst = pd.concat([df_wst, df_temp])
df_wst = df_wst.rename(columns={"geoscale": "Country","timescale" : "Years"})

# fix countries
df1 = df_wst.sort_values(by=['Country'])
np.array(countries)
df1["Country"].unique()
df1.loc[df1["Country"] == "Czechia","Country"] = "Czech Republic"
df1 = df1.loc[df1["Country"] != "Iceland",:]
df1 = df1.loc[df1["Country"] != "Liechtenstein",:]
df1 = df1.loc[df1["Country"] != "Norway",:]
df1.loc[df1["Country"] == "UK","Country"] = "United Kingdom"
df_temp = df1.loc[df1["Country"] == "Germany",:]
df_temp.loc[:,"Country"] = "Paris"
df1 = pd.concat([df1, df_temp])
df_temp = df1.loc[df1["Country"] == "Germany",:]
df_temp.loc[:,"Country"] = "Vaud"
df1 = pd.concat([df1, df_temp])
df1 = df1.loc[df1["Country"] != "Türkiye",]
df1 = df1.sort_values(by=["variable","Country","Years","lever_eol"])

# expand df to include missing years
countries = df_wst["Country"].unique()
years = range(2025,2055,5)
variables = df_wst["variable"].unique()
panel_years = np.tile(np.repeat(years, 4), len(variables))
panel_levels = np.tile(np.tile([1,2,3,4], len(years)), len(variables))
panel_variables = np.repeat(variables, len(np.tile([1,2,3,4], len(years))))
df_temp = pd.DataFrame({"Country" : "EU27", 
                        "Years" : panel_years, 
                        "variable" : panel_variables,
                        "level" : panel_levels})
years = range(1990,2015+1,1)
panel_years = np.tile(years, len(variables))
panel_variables = np.repeat(variables, len(years))
df_temp2 = pd.DataFrame({"Country" : "EU27", 
                        "Years" : panel_years, 
                        "variable" : panel_variables,
                        "level" : 0})
df_temp = pd.concat([df_temp, df_temp2])
df_temp = df_temp.sort_values(by=["variable", "Country", "Years", "level"])
df1 = pd.merge(df_temp, df1, how="left", on=["Country","Years","variable","level"])














# Format df in a common way, add ots and fts years
def extend_and_sort_years(df, country_col='geoscale', year_col='timescale'):
    """
    Extend the years range for each country to 1990 - 2050 (including gaps),
    and fill missing values with the closest value.
    Then sort by Country and Years.
    """
    # Rename columns for consistency
    df.rename(columns={country_col: 'Country', year_col: 'Years'}, inplace=True)
    # Get unique countries and define the years range
    unique_countries = df['Country'].unique()
    years_range = list(range(1990, 2021)) + list(range(2025, 2051, 5))
    # Create a list from the product of unique countries and the range of years
    years_product = list(product(unique_countries, years_range))
    # Convert list into a DataFrame
    expanded_years = pd.DataFrame(years_product, columns=['Country', 'Years'])
    # Convert 'Years' to string for consistency during merge
    expanded_years['Years'] = expanded_years['Years'].astype(str)
    df['Years'] = df['Years'].astype(str)
    # Merge the expanded DataFrame with the original data
    merged_df = pd.merge(expanded_years, df, on=['Country', 'Years'], how='left')
    # Fill missing values within each country using forward and backward fill
    merged_df = merged_df.groupby('Country').apply(lambda group: group.fillna(method='ffill').fillna(method='bfill'))
    merged_df.reset_index(drop=True, inplace=True)
    # Convert the 'Years' column to integers
    merged_df['Years'] = merged_df['Years'].astype(int)
    # Sort the DataFrame by 'Country' and 'Years'
    merged_df.sort_values(by=['Country', 'Years'], inplace=True)
    # Sort 'Country' alphabetically but put 'EU27' at the end
    merged_df['Country_sort_key'] = merged_df['Country'].apply(lambda country: 'ZZZ' if country == 'EU27' else country)
    merged_df.sort_values(by=['Country_sort_key', 'Years'], inplace=True)
    # Drop the temporary 'Country_sort_key' column
    merged_df.drop(columns=['Country_sort_key'], inplace=True)
    return merged_df

# Operate for each of df_pc, df_elv, df_tv, and df_LA
df_pc = extend_and_sort_years(df_pc)
df_elv = extend_and_sort_years(df_elv)
df_tv = extend_and_sort_years(df_tv)
df_LA = extend_and_sort_years(df_LA)

# # checks
# indexes = ["Country","Years"]
# check(df_pc, indexes, subset1)
# check(df_pc, indexes, subset2)
# check(df_LA, indexes, subset1)
# check(df_LA, indexes, subset2)
# check(df_tv, indexes, subset1)
# check(df_tv, indexes, subset2)
# check(df_elv, indexes, subset1)
# check(df_elv, indexes, subset2)

# 1.3 Rename columns in df's to have 'pc_' and 'elv_' prefix
df_pc.rename(columns={col: f'pc_{col}' for col in df_pc.columns if col not in ['Country', 'Years']}, inplace=True)
df_elv.rename(columns={col: f'elv_{col}' for col in df_elv.columns if col not in ['Country', 'Years']}, inplace=True)
df_tv.rename(columns={col: f'tv_{col}' for col in df_tv.columns if col not in ['Country', 'Years']}, inplace=True)
df_LA.rename(columns={col: f'larger-appliances_{col}' for col in df_LA.columns if col not in ['Country', 'Years']}, inplace=True)

# 1.4: Adding phones columns
df_pc = extend_and_sort_years(df_pc)
# 1.5 Duplicate 'pc_' columns to make 'phone_' columns with the same dataset
def duplicate_and_rename_columns(df):
    """
    Duplicate columns with the prefix 'pc_' and change the prefix to 'phone_'.
    """
    new_columns = {}
    for col in df.columns:
        if col.startswith('pc_'):
            new_col = 'phone_' + col[3:]
            new_columns[new_col] = df[col]
    for new_col, data in new_columns.items():
        df[new_col] = data
    return df
# Apply the function to the DataFrame
df_pc = duplicate_and_rename_columns(df_pc)

# 1.4 Merge all df's
merged_df = pd.merge(df_pc, df_elv, on=['Country', 'Years'], how='outer')
merged_df = pd.merge(merged_df, df_tv, on=['Country', 'Years'], how='outer')
merged_df = pd.merge(merged_df, df_LA, on=['Country', 'Years'], how='outer')
product_categories = ['elv', 'larger-appliances', 'pc', 'phone', 'tv']

# # Check
# indexes = ["Country","Years"]
# df = pd.melt(merged_df, id_vars = indexes)
# df["wst_man_operation"] = [i.split("_")[1] for i in df["variable"]]
# df["variable"] = [i.split("_")[0] for i in df["variable"]]
# df = df.sort_values(by = ["Country","Years","variable","wst_man_operation"])
# df["wst_man_operation"].unique()

# subset = ["littered[%]","export[%]","waste-collected[%]","waste-uncollected[%]"]
# df1 = df.loc[df["wst_man_operation"].isin(subset),:]
# indexes = ["Country","Years","variable"]
# df1 = df1.groupby(indexes, as_index=False)['value'].agg(sum)
# df1["value"].unique()

# subset = ["recycling[%]", "energy-recovery[%]", "reuse[%]", "landfill[%]", "incineration[%]"]
# df1 = df.loc[df["wst_man_operation"].isin(subset),:]
# indexes = ["Country","Years","variable"]
# df1 = df1.groupby(indexes, as_index=False)['value'].agg(sum)
# df1["value"].unique()

# Part 2: FTS

# TODO: for now I'll do all countries, so some will have issues as they are already at the target,
# and it can be that they worsen it. Later we'll do only EU27, Switzerland and Vaud for level 1, and for the
# moment drop the remaining FTS.

################################################################
##### littered, export, waste-collected, waste-uncollected #####
################################################################

indexes = ["Country","Years"]
df = pd.melt(merged_df, id_vars = indexes)









df = merged_df.copy()
# Step 1: Add 'lever_eol' column
df['lever_eol'] = 0

# Function to expand the df with levels for future years
def expand_country(df_country, future_years):
    expanded_data = pd.DataFrame()
    for year in future_years:
        for lever in range(1, 5):  # Ensure levels 1 to 4 are included
            temp_df = df_country[df_country['Years'] == year].copy()
            if temp_df.empty:
                # Find the nearest year to fill missing values
                nearest_year = df_country['Years'].sub(year).abs().idxmin()  # Find the nearest year
                temp_df = df_country[df_country['Years'] == df_country.loc[nearest_year, 'Years']].copy()
                if temp_df.empty:  # Still empty, use zeros
                    temp_df = pd.DataFrame(np.zeros((1, len(df_country.columns))), columns=df_country.columns)
                temp_df['Years'] = year
                temp_df['Country'] = df_country['Country'].iloc[0]  # Assuming country name is consistent
            temp_df['lever_eol'] = lever
            expanded_data = pd.concat([expanded_data, temp_df], ignore_index=True)
    return expanded_data

# Define future years, 5-year gaps for after 2025, and get unique countries
future_years = list(range(2019, 2021)) + list(range(2025, 2051, 5))
unique_countries = df['Country'].unique()

# Initialize a list to collect df
df_list = []
for country in unique_countries:
    df_country = df[df['Country'] == country].copy()
    expanded_country_df = expand_country(df_country, future_years)
    # Avoid duplicates by checking both Years and lever_eol columns simultaneously
    condition = ~(
            expanded_country_df['Years'].isin(df_country['Years']) &
            expanded_country_df['lever_eol'].isin(df_country['lever_eol'])
    )
    expanded_country_df = expanded_country_df[condition]
    df_country_combined = pd.concat([df_country, expanded_country_df], ignore_index=True, sort=False)
    df_list.append(df_country_combined)

# Concatenate all country df back together
df = pd.concat(df_list, ignore_index=True)
df.sort_values(by=['Country', 'Years', 'lever_eol'], inplace=True)

# Replace all nan with 0
df.fillna(0, inplace=True)

# drop level 0 for future years
df = df.loc[~((df["Years"].isin(future_years)) & (df["lever_eol"] == 0)),:]

# Check
#df.to_excel('/Users/sqiao/Documents/Calculators/Julie_Calc/end-of-life/toy-module/processing-data/Python/Python-OTS.xlsx', index=False)

# List of thresholds (assumptions based on EU Directive targets and Eurostat values)
thresholds_dict = {
    'larger-appliances': {
        'recovery':        [None, 0.85, 0.9, 0.95, 1],
        'recycling':       [None, 0.8, 0.825, 0.875, 0.9],
        'reuse':           [None, None, 0.04, 0.06, 0.1],
        'energy-recovery': [None, None, 0.035, 0.015, 0],
        'landfill':        [None, None, 0.09, 0.05, 0],
        'incineration':    [None, None, 0.01, 0, 0]
    },
    'tv': {
        'recovery':         [None, 0.8, 0.9, 0.95, 1],
        'recycling':        [None, 0.7, 0.8, 0.85, 0.9],
        'reuse':            [None, None, 0.05, 0.08, 0.1],
        'energy-recovery':  [None, None, 0.05, 0.02, 0],
        'landfill':         [None, None, 0.09, 0.05, 0],
        'incineration':     [None, None, 0.01, 0, 0]
    },
    'pc': {
        'recovery':         [None, 0.8, 0.9, 0.95, 1],
        'recycling':        [None, 0.7, 0.8, 0.85, 0.9],
        'reuse':            [None, None, 0.05, 0.08, 0.1],
        'energy-recovery':  [None, None, 0.05, 0.02, 0],
        'landfill':         [None, None, 0.09, 0.05, 0],
        'incineration':     [None, None, 0.01, 0, 0]
    },
    'phone': {
        'recovery':         [None, 0.8, 0.9, 0.95, 1],
        'recycling':        [None, 0.75, 0.8, 0.85, 0.9],
        'reuse':            [None, None, 0.05, 0.08, 0.1],
        'energy-recovery':  [None, None, 0.05, 0.02, 0],
        'landfill':         [None, None, 0.09, 0.05, 0],
        'incineration':     [None, None, 0.01, 0, 0]
    },
    'elv': {
        'recovery':         [None, 0.85, 0.925, 0.95, 1],
        'recycling':        [None, 0.8, 0.825, 0.85, 0.9],
        'reuse':            [None, 0.1, 0.1, 0.1, 0.1],
        'energy-recovery':  [None, None, 0, 0, 0],
        'landfill':         [None, None, 0.075, 0.05, 0],
        'incineration':     [None, None, 0, 0, 0]
    }
}

# Function to apply thresholds
def adjust_waste_columns(df, mask, columns, thresholds, operation):
    for col, threshold in zip(columns, thresholds):
        if threshold is not None:
            if operation == 'max':
                df.loc[mask, col] = np.maximum(df.loc[mask, col], threshold)
            elif operation == 'min':
                df.loc[mask, col] = np.minimum(df.loc[mask, col], threshold)

# Apply thresholds to each category based on lever_eol levels
def apply_thresholds(df, category, thresholds):
    waste_columns = [
        f'{category}_recovery[%]',  # Index 0: Recovery
        f'{category}_recycling[%]',  # Index 1: Recycling
        f'{category}_reuse[%]',  # Index 2: Reuse
        f'{category}_energy-recovery[%]',  # Index 3: Energy Recovery
        f'{category}_landfill[%]',  # Index 4: Landfill
        f'{category}_incineration[%]'  # Index 5: Incineration
    ]
    for level, threshold in enumerate(thresholds['recovery']):  # Loop through each 'lever_eol' level and corresponding threshold
        #print(f"Processing threshold: {threshold} for level {level}")
        mask = df['lever_eol'] == level
        adjust_waste_columns(df, mask, waste_columns[:2], [threshold, thresholds['recycling'][level]], 'max')  # Apply thresholds for recovery and recycling first
        df.loc[mask, waste_columns[3]] = df.loc[mask, waste_columns[0]] - df.loc[mask, waste_columns[1]] - df.loc[mask, waste_columns[2]]  # Dynamically adjust energy-recovery based on the applied recovery and recycling
        df.loc[mask, waste_columns[3]] = np.maximum(df.loc[mask, waste_columns[3]], 0)  # Ensure that energy recovery does not go below zero
        adjust_waste_columns(df, mask, waste_columns[5:], [thresholds['incineration'][level]], 'min')  # Apply minimum adjustment for incineration
        df.loc[mask, waste_columns[4]] = 1 - df.loc[mask, waste_columns[0]] - df.loc[mask, waste_columns[5]]  # Dynamically calculate landfill based on the remaining sum
        df.loc[mask, waste_columns[4]] = np.maximum(df.loc[mask, waste_columns[4]], 0)  # Ensure that landfill does not go below zero
        # Ensure total waste management sums to 1 for the waste categories
        total_waste_management = df.loc[mask, waste_columns].sum(axis=1)
        mismatch_waste = ~(np.isclose(total_waste_management, 1, atol=0.0001)) # Keep mismatch under 0.00001
        mismatch_indices = df.loc[mask].index[mismatch_waste]  # Get the indices of mismatches
        df.loc[mismatch_indices, waste_columns[4]] += (1 - total_waste_management[mismatch_waste])
    return df

# Apply thresholds and corrections for each product category
for category in product_categories:
    df = apply_thresholds(df, category, thresholds_dict[category])

# General function to ensure balances for any product category
def ensure_balances(df, category):
    waste_columns = [f'{category}_recovery[%]', f'{category}_recycling[%]', f'{category}_reuse[%]',
                     f'{category}_energy-recovery[%]', f'{category}_landfill[%]', f'{category}_incineration[%]']
    collected_waste = df[waste_columns[0]] + df[waste_columns[4]] + df[waste_columns[5]] # collected waste = recovery + landfill + and incineration
    mismatch_recovery = np.isclose(collected_waste, 1, atol=0.00001) == False # Identify rows where the collected waste is not ~1 (with a tolerance of 0.00001)
    recovery_difference = 1 - collected_waste # Calculate the difference between 1 and the collected waste for mismatched rows
    df.loc[mismatch_recovery, waste_columns[4]] += recovery_difference[mismatch_recovery] # Adjust the landfill value by adding the recovery difference for mismatched rows
    return df

# Apply balance enforcement for all products
for category in product_categories:
    df = ensure_balances(df, category)
    # Add the debugging block here to check after balance enforcement
    waste_columns = [
        f'{category}_recovery[%]', f'{category}_recycling[%]', f'{category}_reuse[%]',
        f'{category}_energy-recovery[%]', f'{category}_landfill[%]', f'{category}_incineration[%]'
    ]
    total_waste_management = df[waste_columns].sum(axis=1)
    mismatch_indices = df.index[~np.isclose(total_waste_management, 1, atol=0.00001)]
    if len(mismatch_indices) > 0:
        print(f"Mismatch found in {category} after balance enforcement:")
        print(df.loc[mismatch_indices, waste_columns])
# Move the lever column forward
df = df[[c for c in df.columns if c != 'lever_eol'][:df.columns.get_loc('Years') + 1] + ['lever_eol'] + [c for c in df.columns if c != 'lever_eol'][df.columns.get_loc('Years') + 1:]]

# Fill all nan with 0
df.fillna(0, inplace=True)

# Check
#df.to_excel('/Users/sqiao/Documents/Calculators/Julie_Calc/end-of-life/toy-module/processing-data/Python/Python-OTS+FTS.xlsx', index=False)

# Part 3: Duplicate columns for each catergories to expand the product list
df = df.copy()
# Define the prefixes for duplication
larger_appliances_terms = ['fridge_', 'freezer_', 'dishwasher_', 'washing-machine_']
# Prefixes for each vehicle type, including ldv, hdvl, hdvm, and hdvh
vehicle_prefixes = ['ldv_', 'hdvl_', 'hdvm_', 'hdvh_']
vehicle_types = ['bev_', 'fcev_', 'ice-diesel_', 'ice-gasoline_', 'ice-gas_', 'phev-diesel_', 'phev-gasoline_']
# Generate the full list of terms with the new prefixes for elv
elv_terms = [f"{prefix}{vehicle_type}" for prefix in vehicle_prefixes for vehicle_type in vehicle_types]

# Initialize a list to store new columns and their data
new_columns = {}

# Process each column based on the rules
for column in df.columns:
    if column.startswith('pc_') or column.startswith('phone_') or column.startswith('tv_'):
        # New electronics column
        new_column_name = 'electronics_' + column
        new_columns[new_column_name] = df[column].copy()  # Copy the column data
    elif column.startswith('larger-appliances_'):
        base_name = column[len('larger-appliances_'):]
        # Copy data into multiple larger-appliances categories (fridge, freezer, etc.)
        for term in larger_appliances_terms:
            new_column_name = 'larger-appliances_' + term + base_name
            new_columns[new_column_name] = df[column].copy()  # Copy the original column data
    elif column.startswith('elv_'):
        base_name = column[len('elv_'):]
        # Create new columns for each combination of elv terms
        for term in elv_terms:
            new_column_name = term + base_name
            new_columns[new_column_name] = df[column].copy()  # Copy the original column data
    else:
        # For columns that don't match any pattern, retain them as is
        new_columns[column] = df[column].copy()
# Create the new df with the updated columns
df_updated = pd.DataFrame(new_columns)
list(df_updated.columns)

# Ensure that only 'Country' is a string and all other columns are numeric where applicable
for col in df_updated.columns:
    if col != 'Country':
        # Replace string '0' with numeric 0 only if the column has string types
        if df_updated[col].dtype == object:
            df_updated[col] = df_updated[col].replace('0', 0)
        # Use pd.to_numeric to convert mixed types (int/str) to numeric, coercing errors to NaN
        df_updated[col] = pd.to_numeric(df_updated[col], errors='coerce')

# Remove the old columns
columns_to_remove = [col for col in df_updated.columns if col.startswith(('elv_', 'pc_', 'phone_', 'tv_', 'fridge_', 'freezer_', 'dishwasher_', 'washing-machine_'))]
df_filtered = df_updated.drop(columns=columns_to_remove)
list(df_filtered.columns)

# Save
save_file_directory = os.path.join(current_file_directory, '../data/ind_eol_waste-management.xlsx')
df_filtered.to_excel(save_file_directory, index=False)

# Part 4: check if the sums are correct
# littered + export + uncolleted + collected = 1
# recycling + energy recovery + reuse + landfill + (incineration) = 1
df = df_filtered.copy()
indexes = ["Country","Years","lever_eol"]
df = pd.melt(df, id_vars = indexes)
df["wst_man_operation"] = [i.split("_")[2] for i in df["variable"]]
df["variable"] = [i.split("_")[0] + "_" + i.split("_")[1] for i in df["variable"]]
df = df.sort_values(by = ["Country","Years","lever_eol","variable","wst_man_operation"])
df["wst_man_operation"].unique()

subset = ["littered[%]","export[%]","waste-collected[%]","waste-uncollected[%]"]
df1 = df.loc[df["wst_man_operation"].isin(subset),:]
indexes = ["Country","Years","lever_eol","variable"]
df1 = df1.groupby(indexes, as_index=False)['value'].agg(sum)
df1["value"].unique()

subset = ["recycling[%]", "energy-recovery[%]", "reuse[%]", "landfill[%]", "incineration[%]"]
df1 = df.loc[df["wst_man_operation"].isin(subset),:]
indexes = ["Country","Years","lever_eol","variable"]
df1 = df1.groupby(indexes, as_index=False)['value'].agg(sum)
df1["value"].unique()



