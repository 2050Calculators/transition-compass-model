import requests
from model.common.data_matrix_class import DataMatrix
import os
import pandas as pd


def get_WB_data(file_url, local_filename, var_name, years_ots, country_list=None):
    # Dowload data from the worldbank in excel format.
    # The file_url needs to be the one you obtain when you right click on the dowload XLS option once you selected the
    # data you want.
    # You can try this example
    # file_url = 'https://api.worldbank.org/v2/en/indicator/NY.GDP.PCAP.PP.CD?downloadformat=excel'
    # local_filename = 'data/GDP_World.xlsx'
    # var_name = 'GDP[USD/cap]'
    # years_ots = [1990, 1991, 1992, ..., 2022] # it can be as string
    if country_list is None:
        country_list = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark',
                        'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia',
                        'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland', 'Portugal',
                        'Romania', 'Slovak Republic', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom']


    if not os.path.exists(local_filename):
        response = requests.get(file_url, stream=True)
        # Check if the request was successful
        if response.status_code == 200:
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"File downloaded successfully as {local_filename}")
        else:
            print(f"Error: {response.status_code}, {response.text}")
    else:
        print(f'File {local_filename} already exists. If you want to download again delete the file')

    df = pd.read_excel(local_filename)
    # Replace the columns
    df.columns = df.iloc[2]
    # Filter countries
    df_countries = df[df['Country Name'].isin(country_list)].copy()
    missing_countries = set(country_list).difference(set(df_countries['Country Name']))
    if len(missing_countries) > 0:
        raise ValueError(f'The following countries are not in the database {missing_countries}')
    # Extract the variables name
    var_name_loc = list(set(df_countries['Indicator Name']))
    if len(var_name_loc) > 1:
        raise ValueError('The WorldBank table contains more than one variables')
    else:
        print(f'Extracting from the WorldBank database the variable `{var_name_loc[0]}` as `{var_name}`')
    # Drop useless columns
    df_countries = df_countries.drop(['Indicator Name', 'Indicator Code', 'Country Code'], axis=1).copy()
    # Reshape df to have years as rows
    df_T = pd.melt(df_countries, id_vars=['Country Name'], var_name='Years', value_name=var_name)
    df_T.rename(columns={'Country Name': 'Country'}, inplace=True)
    df_T['Years'] = df_T['Years'].astype(int)
    # Only keep ots years
    int_years_ots = [int(year_str) for year_str in years_ots]
    df_years = df_T[df_T['Years'].isin(int_years_ots)].copy()
    dm = DataMatrix.create_from_df(df_years, num_cat=0)
    # Rename countries
    if 'Slovak Republic' in country_list:
        dm.rename_col('Slovak Republic', 'Slovakia', 'Country')
    return dm
