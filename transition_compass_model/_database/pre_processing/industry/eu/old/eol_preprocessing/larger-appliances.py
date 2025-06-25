import pandas as pd
import eurostat
import pycountry
import numpy as np
import os
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/industry/eu/eol_preprocessing/larger-appliances.py"

# Function for converting Eurostat country codes to English names using pycountry
def convert_country_code_to_name(country_code):
    """
    Convert a Eurostat country code to a full English country name.
    If the country code is not found in pycountry, return the original code.
    Greece and Kosovo are special cases that are converted manually.
    """
    # Special cases: Convert 'EL' to 'Greece' and 'XK' to 'Kosovo'
    if country_code == 'EL':
        return 'Greece'
    elif country_code == 'XK':
        return 'Kosovo'
    try:
        # Use pycountry to get the country name
        country = pycountry.countries.get(alpha_2=country_code)
        # Return the country name if found, otherwise return the original code
        return country.name if country else country_code
    except KeyError:
        # Return the original code if any error occurs (e.g., code not found)
        return country_code

# Function for converting data to long format
def transform_to_long_format(df, id_vars, value_vars, value_col_name='value', timescale_col_name='timescale', geoscale_col_name='geo\\TIME_PERIOD', wst_oper_col_name='wst_oper', waste_col_name=None):
    """
    Transforms a DataFrame to a long format and pivots the data to create separate columns for each 'wst_oper' value.

    Parameters:
    df (DataFrame): The input df to be transformed.
    id_vars (list): List of columns to be kept as identifiers in the melt operation.
    value_vars (list): List of columns to be unpivoted (e.g., years).
    value_col_name (str): The name of the new column that will contain the melted values.
    timescale_col_name (str): The name of the new column for the unpivoted values (e.g., years).
    geoscale_col_name (str): The name of the column containing geographic data.
    wst_oper_col_name (str): The name of the column containing waste operation types.
    waste_col_name (str or None): The name of the waste category column, or None if not present.

    Returns:
    DataFrame: Transformed df in the desired long format with separate columns for each 'wst_oper' value.
    """
    # Step 1: Melt the DataFrame into a long format
    long_df = df.melt(
        id_vars=id_vars,  # Columns that will remain the same
        value_vars=value_vars,  # Columns to unpivot (years)
        var_name=timescale_col_name,  # New column name for the unpivoted years
        value_name=value_col_name  # New column name for the data values
    )
    # Step 2: Rename the geo\\TIME_PERIOD column to 'geoscale' for consistency
    long_df.rename(columns={geoscale_col_name: 'geoscale'}, inplace=True)
    # Debug: Check if the 'waste' column is present before pivoting
    print("Columns in the df before pivoting:", long_df.columns)
    # Step 3: Add the [t] suffix to each wst_oper value before pivoting
    long_df[wst_oper_col_name] = long_df[wst_oper_col_name] + '[t]'
    # Step 4: Determine the index for pivot_table based on waste_col_name presence
    pivot_index = ['geoscale', timescale_col_name]
    if waste_col_name:  # Include waste_col_name in the pivot index only if it is provided
        pivot_index.append(waste_col_name)
    # Step 5: Pivot the DataFrame to create separate columns for each wst_oper value
    long_df = long_df.pivot_table(
        index=pivot_index,
        columns=wst_oper_col_name,
        values=value_col_name,
        aggfunc='sum'
    ).reset_index()
    # Step 6: Flatten the MultiIndex columns after the pivot
    long_df.columns.name = None  # Remove the name of the columns index
    long_df.columns = [str(col) for col in long_df.columns]  # Flatten the columns
    # Step 7: Sort the DataFrame by 'geoscale' alphabetically and 'timescale' numerically
    long_df = long_df.sort_values(by=['geoscale', timescale_col_name], ascending=[True, True])
    return long_df

# Define a list of EU27 countries
eu27_countries = [
    'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Estonia',
    'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia',
    'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania',
    'Slovakia', 'Slovenia', 'Spain', 'Sweden'
]

# Part 1: Get the WEEE database from Eurostat
df = eurostat.get_data_df("env_waselee")

# Use the correct column name for geographic data
geo_column = 'geo\\TIME_PERIOD'

# Step 1: Filter for unit of measure: Tonne
df = df[df['unit'] == 'T']

# Step 2: Filter only the Waste category: Large household appliances
waste_categories = ['EE_LHA']
df_LA = df[df['waste'].isin(waste_categories)]

# Step 3: Filter the Waste management operations
waste_management = ['COL', 'TRT_NEU', 'RCV', 'RCY_PRP_REU', 'PRP_REU'] # Waste collected, Waste treated outside the EU, Recovery, Recycling and preparing for reuse, Preparing for reuse
df_LA = df_LA[df_LA['wst_oper'].isin(waste_management)]

# Step 4: Rename the specified values using the correct column names
replace_dict = {
    'geo\\TIME_PERIOD': {'EU27_2020': 'EU27'},
    'wst_oper': {
        'COL': 'waste-collected',
        'TRT_NEU': 'export',
        'RCV': 'recovery',
        'RCY_PRP_REU': 'recycling',
        'PRP_REU': 'reuse'
    }
}

# Use the replace function to rename values
df_LA = df_LA.replace(replace_dict)

# Step 5: Use pre-defined function to convert Eurostat country codes to English names
# Apply the function to all country codes except for 'EU27'
df_LA['geo\\TIME_PERIOD'] = df_LA['geo\\TIME_PERIOD'].apply(lambda x: convert_country_code_to_name(x) if isinstance(x, str) and x != 'EU27' else x)

# Step 6: Convert the wide format to long format using the pre-defined function
df_LA = transform_to_long_format(
    df=df_LA,
    id_vars=['geo\\TIME_PERIOD', 'waste', 'wst_oper'],
    value_vars=[str(year) for year in range(2007, 2019)]
)

# Check: Export the final long format data to an Excel file
#df_LA.to_excel('/Users/sqiao/Documents/Calculators/Julie_Calc/end-of-life/pre-processing/_data-processing/data/larger-appliances/Python-converted_LA.xlsx', index=False)

# Part 2: Get municipal waste data from Eurostat
df = eurostat.get_data_df("env_wasmun")
# Step 1: Filter for unit of measure: Thousand tonnes
df_mun = df[df['unit'] == 'THS_T']
# Step 2: Filter the Waste management operations for only Disposal - landfill and other (D1-D7, D12), Disposal - incineration (D10)
waste_management = ['DSP_L_OTH', 'DSP_I']  # landfill, incineration
df_mun = df_mun[df_mun['wst_oper'].isin(waste_management)]
# Step 3: Rename the specified values using the correct column names
replace_dict = {
    'geo\\TIME_PERIOD': {'EU27_2020': 'EU27'},
    'wst_oper': {
        'DSP_L_OTH': 'landfill-mun',
        'DSP_I': 'incineration-mun'
    }
}
# Use the replace function to rename values
df_mun = df_mun.replace(replace_dict)
# Step 4: Convert Eurostat country codes to English names using pycountry
df_mun['geo\\TIME_PERIOD'] = df_mun['geo\\TIME_PERIOD'].apply(lambda x: convert_country_code_to_name(x) if isinstance(x, str) and x != 'EU27' else x)
# Step 5: Call the function to convert df to long format
df_mun = transform_to_long_format(
    df=df_mun,
    id_vars=['geo\\TIME_PERIOD', 'wst_oper'],  # Columns to be kept as identifiers during the melt operation
    value_vars=[str(year) for year in range(2007, 2019)],  # Columns representing the years to unpivot
    value_col_name='value',
    timescale_col_name='timescale',
    geoscale_col_name='geo\\TIME_PERIOD',
    wst_oper_col_name='wst_oper',
    waste_col_name=None  # Since there is no specific waste column in this dataset
)

# Check: Export the final long format data to an Excel file
#df_mun.to_excel('/Users/sqiao/Documents/Calculators/Julie_Calc/end-of-life/pre-processing/_data-processing/data/computers/Python-converted_mun.xlsx', index=False)

# Part 3: Combine both df and perform calculations based on assumptions
# Step 1: Merging df_LA and df_mun based on geoscale and timescale
merged_df = pd.merge(df_mun, df_LA, on=['geoscale', 'timescale'], how='outer')
# Step 2: move EU27 to the top
# 2.1 Create a helper column to prioritize 'EU27' at the top
merged_df['sort_order'] = merged_df['geoscale'].apply(lambda x: 0 if x == 'EU27' else 1)
# Sort the df first by the helper column to move 'EU27' to the top, and then by 'geoscale' and 'timescale'
merged_df = merged_df.sort_values(by=['sort_order', 'geoscale', 'timescale'], ascending=[True, True, True])
# Drop the helper column as it is no longer needed
merged_df = merged_df.drop(columns='sort_order')
# Reset the index of the merged DataFrame for a clean look
merged_df.reset_index(drop=True, inplace=True)

# Check: Export the merged DataFrame to an Excel file
#merged_df.to_excel('/Users/sqiao/Documents/Calculators/Julie_Calc/end-of-life/pre-processing/_data-processing/data/larger-appliances/Python-converted_merge.xlsx', index=False)

# Step 3: Linearly regress waste-collected, landfill, and incineration over time (2007-2018)
df = merged_df.copy()
# Step 3.1 Remove countries with only municipal waste data
df = df[~df['geoscale'].isin(['Albania', 'Bosnia and Herzegovina', 'Montenegro', 'North Macedonia', 'Kosovo', 'Serbia', 'Turkey'])]
# Define countries AFTER filtering
countries = df['geoscale'].unique()  # List of filtered countries in 'geoscale'
# Standardize the 'timescale' column to integers
df['timescale'] = pd.to_numeric(df['timescale'])

# 3.2 Set missing values to 0 for specific countries (non-EU27 and other exceptions)
geoscale_values = ["Liechtenstein", "Norway", "Iceland", "UK", "Portugal"]
df.loc[df['geoscale'].isin(geoscale_values), :] = df.loc[df['geoscale'].isin(geoscale_values), :].fillna(0)
# For all NaN in "reuse[t]", fill with 0 due to insufficient data points, except for EU27, Belgium, Ireland, and Spain
def fill_reuse_t(row):
    excluded_geoscales = ['Belgium', 'Ireland', 'Spain', 'Cyprus']
    if row['geoscale'] == 'EU27' and pd.isna(row['reuse[t]']):
        return row['reuse[t]']
    elif row['geoscale'] in excluded_geoscales:
        return row['reuse[t]']
    else:
        return row['reuse[t]'] if not pd.isna(row['reuse[t]']) else 0
df['reuse[t]'] = df.apply(fill_reuse_t, axis=1)
# Specific exceptions: export[t] for Greece, Spain, Poland
def fillna_specific_columns(df, conditions):
    """
    Fill NaN values in specific columns based on conditions.
    """
    for geoscale, column in conditions.items():
        df.loc[df['geoscale'] == geoscale, column] = df.loc[df['geoscale'] == geoscale, column].fillna(0)
    return df
# Set exception countries and variables, set to '0'
conditions = {
    'Greece': 'incineration-mun[t]',
    'Poland': 'export[t]',
    'Spain': 'export[t]'
}
# Update df
df = fillna_specific_columns(df, conditions)

# 3.3 Linear extrapolation for Italy
italy_data = df[df['geoscale'] == 'Italy'].copy()
if italy_data['waste-collected[t]'].notna().sum() > 1:
    for column in [col for col in italy_data.columns if col not in ['waste-collected[t]', 'timescale', 'geoscale', 'export[t]']]:
        if italy_data[column].isna().any():
            clean_data = italy_data.dropna(subset=[column, 'waste-collected[t]'])
            if len(clean_data) > 1:
                x = clean_data['waste-collected[t]'].values
                y = clean_data[column].values
                slope, intercept = np.polyfit(x, y, 1)
                x_pred = italy_data.loc[italy_data[column].isna(), 'waste-collected[t]'].values
                italy_data.loc[italy_data[column].isna(), column] = x_pred * slope + intercept
# Ensure there are enough data points to perform regression of "export[t]" against time
if italy_data['export[t]'].notna().sum() > 1:
    if italy_data['export[t]'].isna().any():
        clean_data = italy_data.dropna(subset=['export[t]', 'timescale'])
        if len(clean_data) > 1:
            x = clean_data['timescale'].values
            y = clean_data['export[t]'].values
            slope, intercept = np.polyfit(x, y, 1)
            x_pred = italy_data.loc[italy_data['export[t]'].isna(), 'timescale'].values
            italy_data.loc[italy_data['export[t]'].isna(), 'export[t]'] = x_pred * slope + intercept
df.update(italy_data)

# 3.4 Linear extrapolation for other countries
for geo in countries:
    if geo not in ['EU27'] + geoscale_values:
        geo_data = df[df['geoscale'] == geo].copy()
        for column in geo_data.select_dtypes(include=[np.number]).columns:
            if geo_data[column].notna().sum() > 1:
                x = geo_data.loc[geo_data[column].notna(), 'timescale'].values
                y = geo_data.loc[geo_data[column].notna(), column].values
                slope, intercept = np.polyfit(x, y, 1)
                geo_data.loc[geo_data[column].isna(), column] = np.maximum(0, geo_data['timescale'] * slope + intercept)
        df.update(geo_data)
# 3.5 Summation and filling of EU27 data
excluded_countries = ['Liechtenstein', 'Norway', 'Iceland', 'United Kingdom', 'Poland', 'Austria']
sum_data = df[~df['geoscale'].isin(excluded_countries + ['EU27'])].drop(columns=['geoscale'], errors='ignore')
summed_data = sum_data.groupby('timescale').sum(min_count=1).reset_index()
sum_dict = summed_data.set_index('timescale').to_dict('index')
eu27_data = df[df['geoscale'] == 'EU27'].copy()

for year in df['timescale'].unique():
    for column in eu27_data.columns:
        if column not in ['geoscale', 'timescale']:
            # Get the current value for EU27 for the specific column and year
            eu27_value = df.loc[(df['geoscale'] == 'EU27') & (df['timescale'] == year), column].values[0]
            # Check if the value is 0 or NaN, and replace it with the sum of EU27 countries if applicable
            if eu27_value == 0 or pd.isna(eu27_value):
                # Sum the values for all EU27 countries (non-excluded) for the corresponding year and column
                eu27_sum = df[(df['timescale'] == year) & (df['geoscale'].isin(df['geoscale'].unique()))][column].sum()
                # If the sum is greater than 0, replace the EU27 value with the calculated sum
                if eu27_sum > 0:
                    df.loc[(df['geoscale'] == 'EU27') & (df['timescale'] == year), column] = eu27_sum
                    # print(f"Replaced 0/NaN in EU27 for {column} in year {year} with {eu27_sum}")
# Fill any remaining NaN values with 0
df.fillna(0, inplace=True)

# Check
#df.to_excel('/Users/sqiao/Documents/Calculators/Julie_Calc/end-of-life/pre-processing/_data-processing/data/larger-appliances/Python-converted_regress_141024.xlsx', index=False)

# Step 4: Convert all variables into %
# 4.1 For waste collected
# 1. If recycling[t] < reuse[t], impose recycling[t] = reuse[t]
df['recycling[t]'] = df[['recycling[t]', 'reuse[t]']].max(axis=1)
# 2. If RCV[t] < RCY_PRP_REU[t], impose RCV[t] = RCY_PRP_REU[t]
df['recovery[t]'] = df[['recovery[t]', 'recycling[t]']].max(axis=1)
# 3. Recycling calculation
df['recycling[t]'] = df['recycling[t]'] - df['reuse[t]']
# 4. Energy recovery calculation
df['energy-recovery[t]'] = df['recovery[t]'] - df['recycling[t]']
# 5. Reuse calculation
# df['reuse[t]'] = df['PRP_REU[t]']
# 6. If COL[t] < RCV[t], impose COL[t] = RCV[t]
df['waste-collected[t]'] = df[['waste-collected[t]', 'recovery[t]']].max(axis=1)
# 7. Landfill calculation
df['landfill[t]'] = df['landfill-mun[t]'] / (df['landfill-mun[t]'] + df['incineration-mun[t]']) * (df['waste-collected[t]'] - df['recovery[t]'])
# 8. Incineration calculation
df['incineration[t]'] = df['incineration-mun[t]'] / (df['landfill-mun[t]'] + df['incineration-mun[t]']) * (df['waste-collected[t]'] - df['recovery[t]'])
# 9. Calculate the total sum (mysum[t])
df['mysum[t]'] = df['recycling[t]'] + df['energy-recovery[t]'] + df['reuse[t]'] + df['landfill[t]'] + df['incineration[t]']
# 10. Calculate recycling percentage
df['recycling[%]'] = df['recycling[t]'] / df['mysum[t]']
# 11. Calculate energy recovery percentage
df['energy-recovery[%]'] = df['energy-recovery[t]'] / df['mysum[t]']
# 12. Calculate reuse percentage
df['reuse[%]'] = df['reuse[t]'] / df['mysum[t]']
# 13. Calculate landfill percentage
df['landfill[%]'] = df['landfill[t]'] / df['mysum[t]']
# 14. Calculate incineration percentage
df['incineration[%]'] = df['incineration[t]'] / df['mysum[t]']
# 15. Calculate recovery percentage
df['recovery[%]'] = df['recovery[t]'] / df['mysum[t]']
# Select only the numeric columns and replace negative values with 0
df.loc[:, df.select_dtypes(include=[np.number]).columns] = df.select_dtypes(include=[np.number]).clip(lower=0)

# 4.2 For total waste
# 1. Use assumptions from Eurostat and eol model to create and append columns, convert all variables to % for KNIME processing
# 1.1 Create null column for 'littered[t]'
df['littered[t]'] = 0
# 1.2 illegal-export[t] = export[t]/0.3*0.7, based on literature data
df['illegal-export[t]'] = df['export[t]'] / 0.3 * 0.7
# 1.3 waste-uncollected[t] = waste-collected[t] / (1-0.5963) * 0.5963, based on literature data
df['waste-uncollected[t]'] = df['waste-collected[t]'] / (1 - 0.5963) * 0.5963
# 1.4 total-waste[t] = waste-collected[t] + waste-uncollected[t] + illegal-export[t] + export[t] + littered[t]
df['total-waste[t]'] = df['waste-collected[t]'] + df['waste-uncollected[t]'] + df['illegal-export[t]'] + df['export[t]'] + df['littered[t]']
# 1.11 waste-uncollected[%] = waste-uncollected[t] / total-waste[t]
df['waste-uncollected[%]'] = df['waste-uncollected[t]'] / df['total-waste[t]']
# 1.12 waste-collected[%] = waste-collected[t] / total-waste[t]
df['waste-collected[%]'] = df['waste-collected[t]'] / df['total-waste[t]']
# 1.13 export[%] = (export[t] + illegal-export[t]) / total-waste[t] NOTE: changed on 30/09/2024 to sum 'export' and 'illegal-export' to the new 'export' total
df['export[%]'] = (df['export[t]'] + df['illegal-export[t]']) / df['total-waste[t]']
# 1.14 littered[%] = littered[t] / total-waste[t]
df['littered[%]'] = df['littered[t]'] / df['total-waste[t]']
# 1.16 Set all range to be between 0 to 1, and NaN to 0
df[['recovery[%]', 'recycling[%]', 'reuse[%]', 'energy-recovery[%]', 'landfill[%]', 'incineration[%]', 'waste-uncollected[%]', 'waste-collected[%]', 'export[%]', 'littered[%]']] = df[['recovery[%]', 'recycling[%]', 'reuse[%]', 'energy-recovery[%]', 'landfill[%]', 'incineration[%]', 'waste-uncollected[%]', 'waste-collected[%]', 'export[%]', 'littered[%]']].apply(lambda x: x.clip(lower=0).fillna(0))
# 2. Filter columns containing '%' and the 'geoscale' and 'timescale' columns
# Ensure that all NaN values are replaced with 0 before exporting
df.fillna(0, inplace=True)

# Filter columns containing '%' and the 'geoscale' and 'timescale' columns for the final output
columns_to_include = [col for col in df.columns if '%' in col or col in ['geoscale', 'timescale']]

# Save
current_file_directory = os.path.dirname(os.path.abspath(__file__))
save_file_directory = os.path.join(current_file_directory, '../data/eol_intermediate_files/LA.xlsx')
df[columns_to_include].to_excel(save_file_directory, index=False)
