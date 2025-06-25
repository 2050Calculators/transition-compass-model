import pandas as pd
import eurostat
import pycountry
import numpy as np

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
    elif country_code == 'UKN':
        return 'UK'
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

# Part 1: Get the Packaging waste by waste management operations database from Eurostat
df = eurostat.get_data_df("env_waspac")

# Use the correct column name for geographic data
geo_column = 'geo\\TIME_PERIOD'

# Step 1: Filter for unit of measure: Tonne
df = df[df['unit'] == 'T']

# Step 2: Filter only the Waste category: IT and telecommunication equipment
waste_categories = ['W150101', 'W150102', 'W15010401', 'W150107']  # Paper and cardboard packaging, Plastic packaging, Aluminium packaging, Glass packaging
df_pac = df[df['waste'].isin(waste_categories)]

# Step 3: Filter the Waste management operations
waste_management = ['GEN', 'RCV', 'RCV_E_PAC', 'RCY', 'RCY_NEU'] # Waste generated, Recovery, Recovery - energy recovery from packaging waste, Recycling, Recycling outside the EU
df_pac = df_pac[df_pac['wst_oper'].isin(waste_management)]

# Step 4: Rename the specified values using the correct column names
replace_dict = {
    'geo\\TIME_PERIOD': {'EU27_2020': 'EU27'},
    'unit': {'T': 'tonne'},
    'waste': {
        'W150101': 'paper',
        'W150102': 'plastic',
        'W15010401': 'aluminium',
        'W150107': 'glass'
    },
    'wst_oper': {
        'GEN': 'waste-generated',
        'RCV': 'recovery',
        'RCV_E_PAC': 'energy-recovery',
        'RCY': 'recycling',
        'RCY_NEU': 'export'
    }
}

# Use the replace function to rename values
df_pac = df_pac.replace(replace_dict)

# Step 5: Use pre-defined function to convert Eurostat country codes to English names
# Apply the function to all country codes except for 'EU27'
df_pac['geo\\TIME_PERIOD'] = df_pac['geo\\TIME_PERIOD'].apply(lambda x: convert_country_code_to_name(x) if isinstance(x, str) and x != 'EU27' else x)

# Step 6: Convert the wide format to long format using the pre-defined function
df_pac = transform_to_long_format(
    df=df_pac,
    id_vars=['geo\\TIME_PERIOD', 'waste', 'wst_oper'],
    value_vars=[str(year) for year in range(2007, 2019)]
)

# Check: Export the final long format data to an Excel file
#df_pac.to_excel('/Users/sqiao/Documents/Calculators/Julie_Calc/end-of-life/pre-processing/_data-processing/data/packaging/Python-converted_pac.xlsx', index=False)

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
#df_mun.to_excel('/Users/sqiao/Documents/Calculators/Julie_Calc/end-of-life/pre-processing/_data-processing/data/packaging/Python-converted_mun.xlsx', index=False)

# Part 3: Combine both df and perform calculations based on assumptions
# Step 1: Merging df_pac and df_mun based on geoscale and timescale
merged_df = pd.merge(df_mun, df_pac, on=['geoscale', 'timescale'], how='outer')

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
#merged_df.to_excel('/Users/sqiao/Documents/Calculators/Julie_Calc/end-of-life/pre-processing/_data-processing/data/packaging/Python-converted_merge.xlsx', index=False)

# Step 3: Linearly regress waste-generated, landfill, and incineration over time (2007-2018)
# 3.1 Use regression of existing data to fill in the missing values
#print("Regression through time for columns 'waste-generated[t]', 'landfill-mun[t]', 'incineration-mun[t]'")

# 3.2 Remove countries with only municipal waste data
merged_df = merged_df[~merged_df['geoscale'].isin(['Albania', 'Bosnia and Herzegovina', 'Montenegro', 'North Macedonia', 'Kosovo', 'Serbia', 'Turkey'])]
countries = merged_df['geoscale'].unique()  # List of all countries in 'geoscale'
excluded_countries = ['UK', 'Norway', 'Iceland', 'Liechtenstein', 'EU27']  # List of non-EU27 countries to be summed up in future steps
included_countries = [country for country in countries if country not in excluded_countries]  # List of countries to include in regression
columns_to_regress = ['waste-generated[t]', 'landfill-mun[t]', 'incineration-mun[t]']  # Define the columns to regress against time

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

# Set the remaining nan to 0
merged_df.fillna(0, inplace=True)
merged_df_new = merged_df.fillna(0)

# Check
#merged_df.to_excel('/Users/sqiao/Documents/Calculators/Julie_Calc/end-of-life/pre-processing/_data-processing/data/packaging/Python-converted_regress.xlsx', index=False)

# Step 4: Regress other variables against the imputed waste-generated column
# 4.1. Regress recovery, recycling, reuse and selected export values against the imputed 'waste-generated[t]' column as the x-axis
# Use the merged_df from previous steps
df = merged_df.copy()

# Add more database specific regression steps here

# Add 'EU27' aggregate where applicable

# Add steps for conversion of all variables into %