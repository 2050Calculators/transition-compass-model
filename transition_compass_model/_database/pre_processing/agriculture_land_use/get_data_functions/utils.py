import numpy as np
import pandas as pd


# Ensure structure coherence
def ensure_structure(df):
    # Get unique values for geoscale, timescale, and variables
    df["timescale"] = df["timescale"].astype(int)
    df = df.drop_duplicates(
        subset=["geoscale", "timescale", "level", "variables", "lever", "module"]
    )
    lever_name = list(set(df["lever"]))[0]
    countries = df["geoscale"].unique()
    years = df["timescale"].unique()
    variables = df["variables"].unique()
    level = df["level"].unique()
    lever = df["lever"].unique()
    module = df["module"].unique()
    # Create a complete multi-index from all combinations of unique values
    full_index = pd.MultiIndex.from_product(
        [countries, years, variables, level, lever, module],
        names=["geoscale", "timescale", "variables", "level", "lever", "module"],
    )
    # Reindex the DataFrame to include all combinations, filling missing values with NaN
    df = df.set_index(
        ["geoscale", "timescale", "variables", "level", "lever", "module"]
    )
    df = df.reindex(full_index, fill_value=np.nan).reset_index()

    return df
