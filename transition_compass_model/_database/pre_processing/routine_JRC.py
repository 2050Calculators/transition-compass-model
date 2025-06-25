
import pandas as pd
import numpy as np
import re
import os
from model.common.data_matrix_class import DataMatrix

def get_jrc_data(dict_extract, dict_countries, current_file_directory, years = list(range(2000,2021+1)), 
                 levels_to_industry_preproc = "../../../Industry"):
    
    sub_variables = dict_extract["sub_variables"]
    calc_names = dict_extract["calc_names"]
    
    # get dataframe by country

    def get_dataframe(dict_extract, country, sub_variables):
        
        # store things
        database = dict_extract["database"]
        sheet = dict_extract["sheet"]
        variable = dict_extract["variable"]
        sheet_last_row = dict_extract["sheet_last_row"]
        if "categories" in list(dict_extract.keys()):
            categories = dict_extract["categories"]
        else:
            categories = None
        
        # define function to search
        def my_search(search, x):
            if x is np.nan:
                return False
            else:
                return bool(re.search(search, x, re.IGNORECASE))
        
        # get data
        filepath_jrc = os.path.join(current_file_directory, levels_to_industry_preproc + '/eu/data/JRC-IDEES-2021/' + country + '/JRC-IDEES-2021_' + database + '_' + country + '.xlsx')
        df = pd.read_excel(filepath_jrc, sheet)
        id_var = df.columns[0]
        df.rename(columns={id_var : "variable"},inplace=True)
        
        # subset data
        def get_indexes(vector, string):
            variable = re.sub(r"([\(\)])", r"\\\1", string) # this is to accept parentheses in the string
            bool_temp = [bool(my_search(variable, x)) for x in vector]
            idx_temp = [i for i, x in enumerate(bool_temp) if x][0]
            return idx_temp
        string_first = get_indexes(df.loc[:,"variable"], variable)
        df_temp = df.iloc[range(string_first,len(df)),:]
        if categories is not None:
            string_intermediate = get_indexes(df_temp.loc[:,"variable"], categories)
            df_temp = df_temp.iloc[range(string_intermediate,len(df_temp)),:]
        string_second = get_indexes(df_temp.loc[:,"variable"], sheet_last_row)
        df_temp = df_temp.iloc[range(0,string_second+1),:]
        df_temp = df_temp.loc[df_temp["variable"].isin(sub_variables),:]
        df_temp = df_temp.loc[:,["variable"] + years]
        
        # add unit to variables names
        if len(variable.split("(")) == 1:
            df_temp["variable"] = [re.sub(r"$", "[unit]", text) for text in df_temp["variable"]]
        else:
            unit = variable.split("(")[1].split(")")[0]
            df_temp["variable"] = [re.sub(r"$", "[" + unit + "]", text) for text in df_temp["variable"]]
        
        # reshape data
        df_temp = pd.melt(df_temp, id_vars = "variable", var_name='Years')
        df_temp["Country"] = country
        df_temp = df_temp.pivot(index=["Country","Years"], columns="variable", values='value').reset_index()
        
        return df_temp
    
    df_all = pd.concat([get_dataframe(dict_extract, country, sub_variables) for country in dict_countries.keys()])
    
    # make data matrix
    dm = DataMatrix.create_from_df(df_all, 0)
    
    # change country names
    dict_temp = dict_countries.copy()
    if "EU27" in list(dict_temp.keys()):
        dict_temp.pop("EU27")
    for key in dict_temp.keys():
        dm.rename_col(key, dict_temp[key], "Country")
        
    # rename variables and aggregate
    for i in range(0,len(sub_variables)):
        dm.rename_col(sub_variables[i], calc_names[i], "Variables")
    
    # return
    return dm

