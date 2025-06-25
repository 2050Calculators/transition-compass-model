import pandas as pd
import eurostat
import pycountry
import numpy as np
import os
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/industry/eu/eol_preprocessing/pc-and-electronics.py"

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
    print("Columns in the DataFrame before pivoting:", long_df.columns)
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

# Step 2: Filter only the Waste category: IT and telecommunication equipment
waste_categories = ['EE_ITT']
df_pc = df[df['waste'].isin(waste_categories)]

# Step 3: Filter the Waste management operations
waste_management = ['COL', 'TRT_NEU', 'RCV', 'RCY_PRP_REU', 'PRP_REU'] # Waste collected, Waste treated outside the EU, Recovery, Recycling and preparing for reuse, Preparing for reuse
df_pc = df_pc[df_pc['wst_oper'].isin(waste_management)]

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
df_pc = df_pc.replace(replace_dict)

# Step 5: Use pre-defined function to convert Eurostat country codes to English names
# Apply the function to all country codes except for 'EU27'
df_pc['geo\\TIME_PERIOD'] = df_pc['geo\\TIME_PERIOD'].apply(lambda x: convert_country_code_to_name(x) if isinstance(x, str) and x != 'EU27' else x)

# Step 6: Convert the wide format to long format using the pre-defined function
df_pc = transform_to_long_format(
    df=df_pc,
    id_vars=['geo\\TIME_PERIOD', 'waste', 'wst_oper'],
    value_vars=[str(year) for year in range(2007, 2019)]
)

# Check: Export the final long format data to an Excel file
# df_pc.to_excel('/Users/sqiao/Documents/Calculators/Julie_Calc/end-of-life/pre-processing/_data-processing/data/computers/Python-converted_pc.xlsx', index=False)


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
# Step 1: Merging df_pc and df_mun based on geoscale and timescale
merged_df = pd.merge(df_mun, df_pc, on=['geoscale', 'timescale'], how='outer')

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
#merged_df.to_excel('/Users/sqiao/Documents/Calculators/Julie_Calc/end-of-life/pre-processing/_data-processing/data/computers/Python-converted_merge.xlsx', index=False)

# Step 3: Linearly regress waste-collected, landfill, and incineration over time (2007-2018)
# 3.1 Use regression of existing data to fill in the missing values
#print("Regression through time for columns 'waste-collected[t]', 'landfill-mun[t]', 'incineration-mun[t]'")

# 3.2 Remove countries with only municipal waste data
merged_df = merged_df[~merged_df['geoscale'].isin(['Albania', 'Bosnia and Herzegovina', 'Montenegro', 'North Macedonia', 'Kosovo', 'Serbia', 'Turkey'])]
countries = merged_df['geoscale'].unique()  # List of all countries in 'geoscale'
excluded_countries = ['UK', 'Norway', 'Iceland', 'Liechtenstein', 'EU27']  # List of non-EU27 countries to be summed up in future steps
included_countries = [country for country in countries if country not in excluded_countries]  # List of countries to include in regression

columns_to_regress = ['waste-collected[t]', 'landfill-mun[t]', 'incineration-mun[t]']  # Define the columns to regress against time

for country in included_countries:
    #print(f"Processing data for country: {country}")
    country_data = merged_df[merged_df['geoscale'] == country]  # Create a new df containing only the data for the current country
    if country != 'EU27':  # Skip 'EU27' to avoid double-counting
        X_country = country_data['timescale'].astype(float).values.reshape(-1, 1)  # Convert timescale to float and reshape
        for column in columns_to_regress:  # Loop only through columns of interest
            y = country_data[column].astype(float).values  # Convert the specified column to float array
            valid_indices = ~pd.isna(y)  # Keep rows with valid values
            relative_missing_indices = np.where(pd.isna(y))[0]  # Indices of missing values
            # Check if there are any missing values to fill and if there are >1 data points for regression
            if relative_missing_indices.size > 0 and valid_indices.sum() > 1:
                #print(f"There are missing values, but enough valid data points to proceed for: {column}")
                # Perform linear regression using numpy
                X_valid = X_country[valid_indices]
                y_valid = y[valid_indices]
                # Calculate the coefficients for linear regression: y = mx + c
                A = np.vstack([X_valid.flatten(), np.ones(len(X_valid))]).T
                m, c = np.linalg.lstsq(A, y_valid, rcond=None)[0]
                # Predict missing values using the regression line
                y_pred = m * X_country[relative_missing_indices].flatten() + c
                # Assign predicted values
                merged_df.loc[country_data.index[relative_missing_indices], column] = np.maximum(y_pred, 0)  # Ensure non-negative values

# 3.3 Handling EU27 outside the country loop to prevent repetitive operations, for only the columns of interest
eu27_data = merged_df[merged_df['geoscale'] == 'EU27']
# To avoid duplication: Create a set to track the columns that have already been calculated for EU27
calculated_columns = set()

for year in eu27_data['timescale'].unique():
    for column in columns_to_regress:
        # Check if this column has already been calculated for EU27
        if column not in calculated_columns:
            # Check if there are missing values for the current year and column
            if pd.isna(eu27_data.loc[(eu27_data['timescale'] == year), column]).any():
                # Aggregate value from included countries for the same timescale
                aggregate_value = merged_df[(merged_df['timescale'] == year) & (merged_df['geoscale'].isin(included_countries))][column].sum()
                # Index for the EU27 data for the current timescale
                eu27_index = eu27_data[eu27_data['timescale'] == year].index[0]
                # Assign the aggregated value for the EU27 row
                merged_df.loc[eu27_index, column] = aggregate_value
                # Update all rows for the same EU27 and timescale (to match KNIME logic)
                merged_df.loc[(merged_df['geoscale'] == 'EU27') & (merged_df['timescale'] == year), column] = aggregate_value
            # Mark this column as calculated
            calculated_columns.add(column)

# Check
# merged_df.to_excel('/Users/sqiao/Documents/Calculators/Julie_Calc/end-of-life/pre-processing/_data-processing/data/computers/Python-converted_regress_141024.xlsx', index=False)

# Step 4: Regress other variables against the imputed waste-collected column
# 4.1. Regress recovery, recycling, reuse and selected export values against the imputed 'waste-collected[t]' column as the x-axis
# Use the merged_df from previous steps
df = merged_df.copy()

# Check for column existence and create missing ones if necessary
for col in ['export[t]', 'recovery[t]', 'recycling[t]', 'reuse[t]']:
    if col not in df.columns:
        df[col] = np.nan
        #print(f"'{col}' column not found. Created with NaN values.")

# 4.1. Regress recovery, recycling, reuse and selected export values against the imputed 'waste-collected[t]' column as the x-axis
#print("Regression through 'waste-collected[t]' for columns 'recovery[t]', 'recycling[t]', 'reuse[t]', 'export[t]'")

exception_countries = ['Czechia', 'Ireland', 'France', 'Lithuania', 'Malta', 'Netherlands', 'Finland', 'Liechtenstein'] # These countries do not need to be regressed for 'export[t]'
columns_to_regress_2 = ['recovery[t]', 'recycling[t]', 'reuse[t]', 'export[t]']

for country in df['geoscale'].unique():
    #print(f"Processing data for country: {country}")
    country_data = df[df['geoscale'] == country]

    if country != 'EU27':  # Skip 'EU27' to avoid double-counting
        X_country = country_data['waste-collected[t]'].astype(float).values.reshape(-1, 1)  # Use waste-collected[t] as the independent variable for regression
        for column in columns_to_regress_2:
            if column == 'export[t]' and country in exception_countries:
                # Skip regression for export[t] and replace NaN with 0 for exception countries
                df.loc[country_data.index, 'export[t]'] = df.loc[country_data.index, 'export[t]'].fillna(0)
                #print(f"Skipped regression for 'export[t]' for country: {country}. Replaced NaNs with 0.")
                continue
            # Skip regressing Belgium's reuse[t]
            if column == 'reuse[t]' and country == 'Belgium':
                #print(f"Skipped regression for 'reuse[t]' for Belgium.")
                continue
            # Skip regressing Bulgaria's reuse[t]
            if column == 'reuse[t]' and country == 'Bulgaria':
                #print(f"Skipped regression for 'reuse[t]' for Bulgaria.")
                continue
            #if column not in country_data.columns:
                #print(f"Column {column} missing for country {country}. Skipping.")
                #continue
            y = country_data[column].astype(float).values  # Convert to float array
            valid_indices = ~pd.isna(y)  # Keep rows with valid values
            relative_missing_indices = np.where(pd.isna(y))[0]  # Indices of missing values
            if relative_missing_indices.size > 0 and valid_indices.sum() > 1:
                #print(f"Predicting missing values for column: {column}")
                # Perform linear regression using numpy
                X_valid = X_country[valid_indices]
                y_valid = y[valid_indices]
                # Calculate the coefficients for linear regression: y = mx + c
                A = np.vstack([X_valid.flatten(), np.ones(len(X_valid))]).T
                m, c = np.linalg.lstsq(A, y_valid, rcond=None)[0]
                # Predict missing values using the regression line
                y_pred = m * X_country[relative_missing_indices].flatten() + c
                # Assign predicted or default values
                df.loc[country_data.index[relative_missing_indices], column] = np.maximum(y_pred, 0)  # Ensure non-negative values

# 4.2 Additional data adjustments after inspecting the cleaned data, based on discussed assumptions: Italy
# Regress Italy's 'recovery[t]' data across 'waste-collected[t]' to replace the original '0' in 2013
df_italy = df[df['geoscale'] == 'Italy']
# Define the years for regression and prediction
regression_years = [2014, 2015, 2016, 2017, 2018]
prediction_years = [2007, 2008, 2009, 2010, 2011, 2012, 2013]
# Separate the dataset into the part for regression and the part for prediction
df_regression = df_italy[df_italy['timescale'].isin(regression_years)]
df_prediction = df_italy[df_italy['timescale'].isin(prediction_years)]
# Perform the regression using 'waste-collected[t]' to predict 'recovery[t]'
X_valid = df_regression['waste-collected[t]'].astype(float).values.reshape(-1, 1)
y_valid = df_regression['recovery[t]'].astype(float).values
# Calculate the coefficients for linear regression: y = mx + c
A = np.vstack([X_valid.flatten(), np.ones(len(X_valid))]).T
m, c = np.linalg.lstsq(A, y_valid, rcond=None)[0]
# Predict 'recovery[t]' for the years 2007-2013
X_predict = df_italy[df_italy['timescale'].isin(prediction_years)]['waste-collected[t]'].astype(float).values.reshape(-1, 1)
predicted_recovery = m * X_predict.flatten() + c
# Replace the 'recovery[t]' values in the original df for the prediction years, only for Italy
df.loc[(df['geoscale'] == 'Italy') & (df['timescale'].isin(prediction_years)), 'recovery[t]'] = predicted_recovery.astype(int)

# 4.3. Now do 'EU27' aggregate with new Italy data and replace any remaining 0 in EU27 with sum of all EU27 countries, where applicable
eu27_data = df[df['geoscale'] == 'EU27']
eu27_columns = ['export[t]', 'recovery[t]', 'recycling[t]', 'reuse[t]', 'waste-collected[t]'] # Where there are still 0 values

for year in eu27_data['timescale'].unique():
    for column in eu27_columns:  # Assuming eu27_columns includes columns that might need recalculation
        if column not in calculated_columns:
            # Calculate the aggregate value from all included EU27 countries for the same year (timescale)
            aggregate_value = df[(df['timescale'] == year) & (df['geoscale'].isin(eu27_countries))][column].sum()
            # Find the index for EU27 for the current year
            eu27_index = eu27_data[eu27_data['timescale'] == year].index[0]
            # Replace the value for EU27 in the current year and column with the calculated sum
            df.loc[eu27_index, column] = aggregate_value
            # Mark this column as calculated now
            calculated_columns.add(column)

        #print(f"Replaced value for EU27 in {column} for year {year} with the sum: {aggregate_value}")

# Ensure that 0 values in EU27 are replaced with the corresponding sums per year and column, where applicable
for year in df['timescale'].unique():
    for column in eu27_columns:
        # Check if the EU27 row has a 0 value in the current column for the current year
        eu27_value = df.loc[(df['geoscale'] == 'EU27') & (df['timescale'] == year), column].values[0]
        if eu27_value == 0:
            # Sum the values for all countries within EU27 for the corresponding column and year
            eu27_sum = df[(df['timescale'] == year) & (df['geoscale'].isin(eu27_countries))][column].sum()

            # Replace the 0 value in the EU27 row for that year with the sum if greater than 0
            if eu27_sum > 0:
                df.loc[(df['geoscale'] == 'EU27') & (df['timescale'] == year), column] = eu27_sum
                #print(f"Replaced 0 in EU27 for {column} in year {year} with {eu27_sum}")

# 4.5. Setting all remaining empty cells to 0 due to insufficient values to impute
#print("Setting all remaining empty cells to 0 due to insufficient values to impute")
df.fillna(0, inplace=True)  # Set all remaining NaN values to 0

# Check
#df.to_excel('/Users/sqiao/Documents/Calculators/Julie_Calc/end-of-life/pre-processing/_data-processing/data/computers/Python-converted_regress_141024.xlsx', index=False)

# 4.6. Convert all variables into %
# 4.6.1. For waste collected
# If recycling[t] < reuse[t], impose recycling[t] = reuse[t]
df['recycling[t]'] = df[['recycling[t]', 'reuse[t]']].max(axis=1)
# If recovery[t] < recycling[t], impose recovery[t] = recycling[t]
df['recovery[t]'] = df[['recovery[t]', 'recycling[t]']].max(axis=1)
# Recycling calculation: recycling[t] = recycling[t] - reuse[t]
df['recycling[t]'] = df['recycling[t]'] - df['reuse[t]']
# Energy recovery calculation: energy-recovery[t] = recovery[t] - recycling[t]
df['energy-recovery[t]'] = df['recovery[t]'] - df['recycling[t]']
# If waste-collected[t] < recovery[t], impose waste-collected[t] = recovery[t]
df['waste-collected[t]'] = df[['waste-collected[t]', 'recovery[t]']].max(axis=1)
# Landfill calculation: based on the assumption that municipal waste % is representative of WEEE
df['landfill[t]'] = df['landfill-mun[t]'] / (df['landfill-mun[t]'] + df['incineration-mun[t]']) * (df['waste-collected[t]'] - df['recovery[t]'])
# Incineration calculation: based on the assumption that municipal waste % is representative of WEEE
df['incineration[t]'] = df['incineration-mun[t]'] / (df['landfill-mun[t]'] + df['incineration-mun[t]']) * (df['waste-collected[t]'] - df['recovery[t]'])
# Calculate the total sum (mysum[t])
df['mysum[t]'] = df['recycling[t]'] + df['energy-recovery[t]'] + df['reuse[t]'] + df['landfill[t]'] + df['incineration[t]']
# Calculate recycling percentage
df['recycling[%]'] = df['recycling[t]'] / df['mysum[t]']
# Calculate energy recovery percentage
df['energy-recovery[%]'] = df['energy-recovery[t]'] / df['mysum[t]']
# Calculate reuse percentage
df['reuse[%]'] = df['reuse[t]'] / df['mysum[t]']
# Calculate landfill percentage
df['landfill[%]'] = df['landfill[t]'] / df['mysum[t]']
# Calculate incineration percentage
df['incineration[%]'] = df['incineration[t]'] / df['mysum[t]']
# Calculate recovery percentage
df['recovery[%]'] = df['recovery[t]'] / df['mysum[t]']

# 4.6.2. For total waste
# Create null column for 'littered[t]'
df['littered[t]'] = 0
# illegal-export[t] = export[t]/0.3*0.7, based on literature data
df['illegal-export[t]'] = df['export[t]'] / 0.3 * 0.7
# waste-uncollected[t] = waste-collected[t] / (1-0.5963) * 0.5963, based on literature data
df['waste-uncollected[t]'] = df['waste-collected[t]'] / (1 - 0.5963) * 0.5963
# total-waste[t] = waste-collected[t] + waste-uncollected[t] + illegal-export[t] + export[t] + littered[t]
df['total-waste[t]'] = df['waste-collected[t]'] + df['waste-uncollected[t]'] + df['illegal-export[t]'] + df['export[t]'] + df['littered[t]']
# waste-uncollected[%] = waste-uncollected[t] / total-waste[t]
df['waste-uncollected[%]'] = df['waste-uncollected[t]'] / df['total-waste[t]']
# waste-collected[%] = waste-collected[t] / total-waste[t]
df['waste-collected[%]'] = df['waste-collected[t]'] / df['total-waste[t]']
# export[%] = (export[t] + illegal-export[t]) / total-waste[t]
df['export[%]'] = (df['export[t]'] + df['illegal-export[t]']) / df['total-waste[t]'] # This is the total export
# littered[%] = littered[t] / total-waste[t]
df['littered[%]'] = df['littered[t]'] / df['total-waste[t]']
# Ensure all percentage values are between 0 and 1, and replace NaN values with 0
df[['recovery[%]', 'recycling[%]', 'reuse[%]', 'energy-recovery[%]', 'landfill[%]', 'incineration[%]',
    'waste-uncollected[%]', 'waste-collected[%]', 'export[%]', 'littered[%]']] = df[['recovery[%]',
    'recycling[%]', 'reuse[%]', 'energy-recovery[%]', 'landfill[%]', 'incineration[%]', 'waste-uncollected[%]',
    'waste-collected[%]', 'export[%]', 'littered[%]']].apply(lambda x: x.clip(lower=0).fillna(0))

# Filter columns containing '%' and the 'geoscale' and 'timescale' columns for the final output
columns_to_include = [col for col in df.columns if '%' in col or col in ['geoscale', 'timescale']]

# Save
current_file_directory = os.path.dirname(os.path.abspath(__file__))
save_file_directory = os.path.join(current_file_directory, '../data/eol_intermediate_files/pc_and_electronics.xlsx')
df[columns_to_include].to_excel(save_file_directory, index=False)

