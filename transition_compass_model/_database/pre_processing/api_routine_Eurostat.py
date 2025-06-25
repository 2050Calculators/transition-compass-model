import eurostat
import pandas as pd
import numpy as np
from model.common.data_matrix_class import DataMatrix
# Check website for more insight https://pypi.org/project/eurostat/
# Use the following lines to search a string of text in a table title in eurostat
# toc = eurostat.get_toc_df(agency='EUROSTAT', lang='en')
# toc_pop = eurostat.subset_toc_df(toc, 'population')
# toc_chdd = eurostat.subset_toc_df(toc, 'Cooling and heating degree days')


def get_data_api_eurostat(eustat_code, filter, mapping_dim, unit, years=None):
    dict_iso2 = {
        "AT": "Austria",
        "BE": "Belgium",
        "BG": "Bulgaria",
        "HR": "Croatia",
        "CY": "Cyprus",
        "CZ": "Czech Republic",
        "DK": "Denmark",
        "EE": "Estonia",
        "FI": "Finland",
        "FR": "France",
        "DE": "Germany",
        "EL": "Greece",
        "HU": "Hungary",
        "IE": "Ireland",
        "IT": "Italy",
        "LV": "Latvia",
        "LT": "Lithuania",
        "LU": "Luxembourg",
        "MT": "Malta",
        "NL": "Netherlands",
        "PL": "Poland",
        "PT": "Portugal",
        "RO": "Romania",
        "SK": "Slovakia",
        "SI": "Slovenia",
        "ES": "Spain",
        "SE": "Sweden",
        "CH": "Switzerland",
        "UK" : "United Kingdom",
        "EU27_2020": "EU27"
    }

    num_cat = 0
    df = eurostat.get_data_df(eustat_code)

    for k, v in filter.items():
        if isinstance(v, str) or isinstance(v, int):
            v = [v]
        df = df[df[k].isin(v)].copy()

    # Rename column to 'Country'
    df.rename(columns={mapping_dim['Country']: 'Country'}, inplace=True)
    # Find all years columns
    year_cols = []
    for col in df.columns:
        try:
            year_cols.append(str(int(col)))
        except ValueError:
            pass

    # Create a single column variables_categories1_categories2[unit] called var_long_name
    concat_list = ['Categories1', 'Categories2']
    df['var_long_name'] = df[mapping_dim['Variables']]
    for cat in concat_list:
        if cat in mapping_dim.keys():
            df[mapping_dim[cat]] = df[mapping_dim[cat]].str.replace('_', '', regex=False)
            df['var_long_name'] = df['var_long_name'] + '_' + df[mapping_dim[cat]]
            num_cat = num_cat + 1
    df['var_long_name'] = df['var_long_name'] + '[' + unit + ']'

    df = df[['Country', 'var_long_name'] + year_cols]

    df_melted = pd.melt(df, id_vars=['Country', 'var_long_name'], var_name='Years', value_name='tmp')
    df_pivoted = df_melted.pivot_table(index=['Country', 'Years'], columns='var_long_name', values='tmp').reset_index()

    # Add NaN where country x year combination is missing
    df_pivoted['Years'] = df_pivoted['Years'].astype(int)
    # Create a MultiIndex with all combinations of countries and years
    multi_index = pd.MultiIndex.from_product([df_pivoted['Country'].unique(), df_pivoted['Years'].unique()], names=['Country', 'Years'])
    # Reindex the original DataFrame with the MultiIndex
    df_all = df_pivoted.set_index(['Country', 'Years']).reindex(multi_index).reset_index()

    # Create datamatrix
    dm = DataMatrix.create_from_df(df_all, num_cat)

    if years is not None:
        dm.filter({'Years': years}, inplace=True)

    dm.sort('Years')

    # Rename iso codes to country names
    for iso in dm.col_labels['Country']:
        cntr_name = dict_iso2[iso]
        dm.rename_col(iso, cntr_name, dim='Country')


    return dm



