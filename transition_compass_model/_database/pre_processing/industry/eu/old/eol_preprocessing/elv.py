import pandas as pd
import eurostat
import pycountry
import numpy as np
import os
__file__ = "/Users/echiarot/Documents/GitHub/2050-Calculators/PathwayCalc/_database/pre_processing/industry/eu/eol_preprocessing/elv.py"


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
df = eurostat.get_data_df("env_waselv")

# Use the correct column name for geographic data
geo_column = 'geo\\TIME_PERIOD'

# Step 1: Filter for unit of measure: Tonne
df = df[df['unit'] == 'T']

# Step 2: Filter based on different combinations of 'waste' and 'wst_oper'
# 2.1 Select from 'Waste arising only from end-of-life vehicles of type passenger cars (M1), light commercial vehicles (N1) and three wheeled moped vehicles (ELV)', the 'Waste generated (GEN)' & 'Recycling (RCY)'
df_elv_1 = df[(df['waste'] == 'ELV') & (df['wst_oper'].isin(['GEN', 'RCY', 'RCV', 'REU']))]
# 2.2 Select from 'End-of-life vehicles exported (EXP)', 'Waste generated (GEN)' & 'Disposal (DSP)'
df_elv_2 = df[(df['waste'] == 'EXP') & (df['wst_oper'].isin(['GEN', 'DSP']))]
# 2.3 Select from 'Waste from dismantling and de-pollution of end-of-life-vehicles (LIQ+W1601A+W1601B+W1601C+LoW:160103+160107+160119+160120+1606+1608) (DMDP)' & 'Waste arising from shredding of end-of-life vehicles (W191001+W191002+W1910A+W1910B)', 'Recovery - energy recovery (R1)' & 'Disposal (DSP)' & 'Recovery (RCV)'
df_elv_3 = df[(df['waste'].isin(['DMDP', 'W1910'])) & (df['wst_oper'].isin(['RCV_E', 'DSP', 'RCV']))]
# Combine all filtered DataFrames into one
df_elv = pd.concat([df_elv_1, df_elv_2, df_elv_3])

# Step 3: Renaming specific (waste, wst_oper) combinations, EU27, and geoscale
combination_replace_dict = {
    ('ELV', 'GEN'): 'GEN', # Waste generated
    ('ELV', 'RCY'): 'RCY', # Recycling
    ('EXP', 'GEN'): 'EXP-GEN', # Exported
    ('DMDP', 'RCV_E'): 'R1_DMDP', # Energy recovery of waste from dismantling and de-pollution
    ('W1910', 'RCV_E'): 'R1_W1910', # Energy recovery of waste from shredding
    ('DMDP', 'DSP'): 'DSP-DMDP', # Disposal of waste from dismantling and de-pollution
    ('W1910', 'DSP'): 'DSP-W1910', # Disposal of waste from shredding
    ('EXP', 'DSP'): 'DSP-EXP', # Disposal of exported vehicles
    ('ELV', 'RCV'): 'recovery', # recovery
    ('ELV','REU'): 'reuse', # reuse
    ('DMDP', 'RCV'): 'RCV_DMDP', # placeholder
    ('W1910', 'RCV'): 'RCV_W1910' # placeholder
}
# Function to replace based on the combination of 'waste' and 'wst_oper'
def translate_waste_oper(row):
    return combination_replace_dict.get((row['waste'], row['wst_oper']), None)
# Apply the replacement
df_elv['translated'] = df_elv.apply(translate_waste_oper, axis=1)

# Rename the 'geo\\TIME_PERIOD' column to 'geoscale' and replace 'EU27_2020' with 'EU27'
df_elv.rename(columns={'geo\\TIME_PERIOD': 'geoscale'}, inplace=True)
# Also replace 'EU27_2020' with 'EU27'
df_elv['geoscale'] = df_elv['geoscale'].replace({'EU27_2020': 'EU27'})
# Apply the function to all country codes except for 'EU27'
df_elv['geoscale'] = df_elv['geoscale'].apply(lambda x: convert_country_code_to_name(x) if isinstance(x, str) and x != 'EU27' else x)

# Step 5: Reshape the df to long format, but first exclude unnecessary columns like 'freq' and 'unit'
# Drop columns 'freq' and 'unit' before reshaping
df_elv = df_elv.drop(columns=['freq', 'unit'])
# Melt the year columns into a 'timescale' column
df_elv = df_elv.melt(
    id_vars=['geoscale', 'waste', 'wst_oper', 'translated'],  # Columns to keep
    var_name='timescale',  # Name of the new column that will hold the years
    value_name='value'  # Name of the new column that will hold the values
)
# Proceed with the pivot_table step now that the 'timescale' column is created
df_elv = df_elv.pivot_table(
    index=['geoscale', 'timescale'],  # Keeping the identifiers 'geo' and 'timescale'
    columns='translated',  # Pivoting on the 'translated' column, which has new names from combination_replace_dict
    values='value',  # Using the 'value' column created from the melt as the values
    aggfunc='sum'  # Aggregating with sum
).reset_index()
# Flatten the MultiIndex in columns after the pivot operation
df_elv.columns = [str(col) for col in df_elv.columns]

# Check: Export the final long format data to an Excel file
#df_elv.to_excel('/Users/sqiao/Documents/Calculators/Julie_Calc/end-of-life/pre-processing/_data-processing/data/vehicles/Python-converted_tv.xlsx', index=False)

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
# Step 1: Merging df_elv and df_mun based on geoscale and timescale
merged_df = pd.merge(df_mun, df_elv, on=['geoscale', 'timescale'], how='outer')
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
#merged_df.to_excel('/Users/sqiao/Documents/Calculators/Julie_Calc/end-of-life/pre-processing/_data-processing/data/vehicles/Python-converted_merge.xlsx', index=False)

# Step 3: Linearly regress waste-collected, landfill, and incineration over time (2007-2018)
df = merged_df.copy()

# Step 3.1 Remove countries with only municipal waste data
df = df[~df['geoscale'].isin(['Albania', 'Bosnia and Herzegovina', 'Montenegro', 'North Macedonia', 'Kosovo', 'Serbia', 'Turkey'])]

# 3.2 Special operations for ELV: remove years before 2010
excluded_years = [2005, 2006, 2007, 2008, 2009, 2010]  # To remove data gaps
included_years = [year for year in df['timescale'].unique() if year not in excluded_years]  # Include all other years
df = df[df['timescale'].isin(included_years)]  # Keep only the years with proper data
# Standardize the 'timescale' column to integers
df['timescale'] = pd.to_numeric(df['timescale'])

# 3.3 Linear regression for GEN
excluded_countries = ['UK', 'Norway', 'Iceland', 'Liechtenstein', 'EU27']
included_countries = [country for country in df['geoscale'].unique() if country not in excluded_countries]
# Handle EU27 data aggregation and filling
eu27_data = df[df['geoscale'] == 'EU27']
for year in eu27_data['timescale'].unique():
    if pd.isna(eu27_data.loc[eu27_data['timescale'] == year, 'GEN']).any():
        aggregate_value = df[(df['timescale'] == year) & (df['geoscale'].isin(included_countries))]['GEN'].sum()
        df.loc[(df['geoscale'] == 'EU27') & (df['timescale'] == year), 'GEN'] = aggregate_value
        #print(f"Aggregated value for EU27 in {year} for GEN: {aggregate_value}")
# Perform regression for GEN column for each country in included_countries
for country in included_countries:
    #print(f"Processing data for country: {country}")
    country_data = df[df['geoscale'] == country].copy()
    if country_data.empty:
        continue
    X_country = country_data['timescale'].values.reshape(-1, 1)
    y = country_data['GEN'].values
    missing_indices = np.where(pd.isna(y))[0]
    if missing_indices.size > 0 and len(y) - len(missing_indices) > 1:
        #print(f"Predicting missing values for GEN in {country}")
        slope, intercept = np.polyfit(X_country[~pd.isna(y)].flatten(), y[~pd.isna(y)], 1)
        y_pred = slope * X_country[missing_indices].flatten() + intercept
        country_data.iloc[missing_indices, country_data.columns.get_loc('GEN')] = np.maximum(0, y_pred)
        df.update(country_data)

# 3.4 Regress other columns against GEN to fill missing values
# Columns to regress against 'GEN'
columns_to_regress = ['RCY', 'DSP-W1910', 'DSP-DMDP', 'DSP-EXP', 'EXP-GEN', 'R1_DMDP', 'R1_W1910', 'recovery', 'reuse']
# Impute missing values using regression across the 'GEN' column
for country in df['geoscale'].unique():
    country_data = df[df['geoscale'] == country].copy()
    #print(f"Processing data for country: {country}")
    if country != 'EU27':  # Skip 'EU27' to avoid double-counting
        X_country = country_data['GEN'].values.reshape(-1, 1)  # Use 'GEN' column as the independent variable for regression
        for column in columns_to_regress:
            y = country_data[column].values
            valid_indices = ~pd.isna(y)  # Boolean array of valid data points
            missing_indices = np.where(pd.isna(y))[0]  # Indices where data is missing
            # Ensure there are enough valid data points for regression
            if missing_indices.size > 0 and valid_indices.sum() > 1:
                try:
                    #print(f"Predicting missing values for {country} in {column}")
                    # Perform linear regression using numpy's polyfit
                    slope, intercept = np.polyfit(X_country[valid_indices].flatten(), y[valid_indices], 1)
                    y_pred = slope * X_country[missing_indices].flatten() + intercept
                    #print(f"Predicted values for {country} in {column}: {y_pred}")
                    # Fill the missing values in the original dataframe
                    country_data.loc[country_data.index[missing_indices], column] = np.maximum(0, y_pred)
                except np.linalg.LinAlgError:
                    #print(f"Skipping regression for {country} in {column} due to insufficient or collinear data.")
                    df.update(country_data)
# Handle EU27 data aggregation and filling
eu27_data = df[df['geoscale'] == 'EU27']
for year in eu27_data['timescale'].unique():
    for column in columns_to_regress:
        if pd.isna(eu27_data.loc[eu27_data['timescale'] == year, column]).any():
            aggregate_value = df[(df['timescale'] == year) & (df['geoscale'].isin(included_countries))][column].sum()
            df.loc[(df['geoscale'] == 'EU27') & (df['timescale'] == year), column] = aggregate_value
            #print(f"Aggregated value for EU27 in {year} for {column}: {aggregate_value}")

# Replace all NaN values with 0 in the DataFrame
df.fillna(0, inplace=True)

df = df[df['timescale'] >= 2011]
# Check
#df.to_excel('/Users/sqiao/Documents/Calculators/Julie_Calc/end-of-life/pre-processing/_data-processing/data/vehicles/Python-converted_regress.xlsx', index=False)

# Step 4: Convert all variables into %
# 1. Use assumptions from Eurostat and eol model to create and append columns, convert all variables to % for KNIME processing
# 1.1 littered[t] = 0, waste-collected[t] = GEN - EXP-GEN
df['littered[t]'] = 0
# df['waste-collected[t]'] = df['GEN'] - df['EXP-GEN'] # did not work because collected waste not summing to 100% when recycling+energy recovery+reuse+landfill not add up to 1
df['waste-collected[t]'] = df['RCY'] + df['reuse'] + df['R1_W1910'] + df['R1_DMDP'] + df['DSP-DMDP'] + df['DSP-W1910'] # sum to 1, where 1 is waste-collected
# 1.2 illegal-export[t] = EXP-GEN/0.18*0.82, based on literature data
df['illegal-export[t]'] = df['EXP-GEN'] / 0.18 * 0.82
# 1.3 waste-uncollected[t] = waste-collected[t] / 0.8 * 0.2, based on literature data
df['waste-uncollected[t]'] = df['waste-collected[t]'] / 0.8 * 0.2
# 1.4 total-waste[t] = waste-collected[t] + waste-uncollected[t] + illegal-export[t] + EXP-GEN[t] + littered[t] BUT assumption for waste collected changed
# df['total-waste[t]'] = df['waste-collected[t]'] + df['waste-uncollected[t]'] + df['illegal-export[t]'] + df['EXP-GEN'] + df['littered[t]']
df['total-waste[t]'] = df['waste-collected[t]'] + df['waste-uncollected[t]'] + df['illegal-export[t]'] + df['EXP-GEN'] + df['littered[t]']
# 1.5 Converting all variables into % for EUCalc tree-parallel processing
# recovery[%] = recovery / waste-collected[t], based on Eurostat assumption (domestic waste collected as 100%)
df['recovery[%]'] = df['recovery'] / df['waste-collected[t]']
# 1.6 recycling[%] = RCY / waste-collected[t]
df['recycling[%]'] = df['RCY'] / df['waste-collected[t]']
# 1.7 reuse[%] = reuse / waste-collected[t]
df['reuse[%]'] = df['reuse'] / df['waste-collected[t]']
# 1.8 energy-recovery[%] = (R1_W1910 + R1_DMDP) / waste-collected[t]
df['energy-recovery[%]'] = (df['R1_W1910'] + df['R1_DMDP']) / df['waste-collected[t]']
# 1.9 landfill[%] = (DSP-DMDP + DSP-W1910) / total-waste[t], only domestic landfill, incineration of ELV is not allowed in Europe
df['landfill[%]'] = (df['DSP-DMDP'] + df['DSP-W1910']) / df['waste-collected[t]']
df['incineration[%]'] = 0 # Set to 0 as EU regulations required
# The rest of the % use total waste as 100% for overall calculations (incl. outside EU)
# 1.10 waste-uncollected[%] = waste-uncollected[t] / total-waste[t]
df['waste-uncollected[%]'] = df['waste-uncollected[t]'] / df['total-waste[t]']
# 1.11 waste-collected[%] = waste-collected[t] / total-waste[t] BUT since the assumption changed, this will change to new formula
# df['waste-collected[%]'] = df['waste-collected[t]'] / df['total-waste[t]']
df['waste-collected[%]'] = (df['waste-collected[t]']) / df['total-waste[t]']
# 1.13 export[%] = (EXP-GEN + illegal-export[t]) / total-waste[t] NOTE: 300924 update summed 'export' and 'illegal-export' to the new 'export' total
df['export[%]'] = (df['EXP-GEN'] + df['illegal-export[t]']) / df['total-waste[t]']
# illegal-export[%] = illegal-export[t] / total-waste[t] NOTE: Removed on 300924
# df['illegal-export[%]'] = df['illegal-export[t]'] / df['total-waste[t]']
# 1.14 littered[%] = littered[t] / total-waste[t]
df['littered[%]'] = df['littered[t]'] / df['total-waste[t]']
# 1.15 Set all range to be between 0 to 1, and NaN to 0
df[['recovery[%]', 'recycling[%]', 'reuse[%]', 'energy-recovery[%]', 'landfill[%]', 'incineration[%]', 'waste-uncollected[%]', 'waste-collected[%]', 'export[%]']] = df[['recovery[%]', 'recycling[%]', 'reuse[%]', 'energy-recovery[%]', 'landfill[%]', 'incineration[%]', 'waste-uncollected[%]', 'waste-collected[%]', 'export[%]']].apply(lambda x: x.clip(lower=0).fillna(0))
# 1.16 Make sure no value is greater than 1
exclude_columns = ['geoscale', 'timescale']
df.update(df.drop(columns=exclude_columns).apply(pd.to_numeric, errors='coerce').clip(upper=1))

# 2. Filter columns containing '%' and the 'geoscale' and 'timescale' columns
columns_to_include = [col for col in df.columns if '%' in col or col in ['geoscale', 'timescale']]
df = df[columns_to_include]
# Remove rows with timescale 2007, 2008, 2009, or 2010
df = df[~df['timescale'].isin([2005, 2006, 2007, 2008, 2009, 2010])]
# UK had NaN due to 0 value, therefore change to 0
df.fillna(0, inplace=True)

# Save
current_file_directory = os.path.dirname(os.path.abspath(__file__))
save_file_directory = os.path.join(current_file_directory, '../data/eol_intermediate_files/elv.xlsx')
df[columns_to_include].to_excel(save_file_directory, index=False)
