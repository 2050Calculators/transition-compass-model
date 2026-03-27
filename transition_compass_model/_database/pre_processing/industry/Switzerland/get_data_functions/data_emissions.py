import pandas as pd
import os
import numpy as np


def get_emissions_data(current_file_directory, ghg):
    filepath = os.path.join(
        current_file_directory,
        "../data/emissions/Evolution_GHG_since_1990_2025-04.xlsx",
    )
    df = pd.read_excel(filepath, sheet_name=ghg)
    df.columns = df.iloc[3, :]
    df = df.iloc[[11, 31], :]
    df.iloc[0, 1] = "process-emissions"
    df.iloc[1, 1] = "combustion-emissions"
    df = pd.melt(df, id_vars=["Cat. (1)", np.nan], var_name="Years")
    df.columns = ["Cat. (1)", "emissions", "Years", "value"]
    df = df.loc[:, ["emissions", "Years", "value"]]
    df["emissions"] = df["emissions"].astype(str)
    df["Years"] = df["Years"].astype(int)
    df["value"] = df["value"].astype(float)
    df["ghg"] = ghg
    return df
