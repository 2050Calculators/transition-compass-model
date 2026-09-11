import numpy as np
import pandas as pd

from transition_compass_model._database.pre_processing.power.Switzerland.processors.hydro_capacity_pipeline_CH import (
    adjust_based_on_nexuse,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def extract_waste_capacity_ots(local_filename, years_ots):
    df = pd.read_excel(local_filename)
    df["Kanton"] = df["Kanton"].str.replace("\xa0", "")
    # Rename cols
    dict_rename = {
        "Kanton": "Country",
        "Täglicher Abfalldurchsatz (Tonnen)": "ref_capacity-Pmax[t]",
        "Elektrische Leistung (MW)": "pow_capacity-Pmax[MW]",
        "Inbetriebnahme": "Years",
    }
    df.rename(dict_rename, axis=1, inplace=True)
    # Keep only important cols
    df = df[list(dict_rename.values())]
    all_years = range(df["Years"].min(), df["Years"].max() + 1)
    all_countries = df["Country"].unique()
    # 2. Create a MultiIndex for all combinations of countries and years
    multi_index = pd.MultiIndex.from_product(
        [all_countries, all_years], names=["Country", "Years"]
    )
    # 3. Reindex the DataFrame with this MultiIndex
    df = df.set_index(["Country", "Years"]).reindex(multi_index)
    # 4. Fill missing values with 0
    df = df.fillna(0).astype(int)
    # Reset index if needed
    df = df.reset_index()
    dm = DataMatrix.create_from_df(df, num_cat=0)
    years_missing = list(set(years_ots) - set(dm.col_labels["Years"]))
    dm.add(0, dummy=True, dim="Years", col_label=years_missing)
    dm.sort("Years")
    dm.array = np.cumsum(dm.array, axis=1)
    dm.filter({"Years": years_ots}, inplace=True)
    dm_CH = dm.groupby({"Switzerland": ".*"}, regex=True, dim="Country", inplace=False)
    dm.append(dm_CH, dim="Country")
    dm.rename_col(
        ["pow_capacity-Pmax", "ref_capacity-Pmax"],
        ["pow_capacity-Pmax_Waste", "ref_capacity-Pmax_Waste"],
        dim="Variables",
    )
    dm.deepen()
    return dm


def run(dm_capacity, years_ots):
    # https://de.wikipedia.org/wiki/Liste_von_Kehrichtverbrennungsanlagen_in_der_Schweiz
    local_filename = "data/waste_power.xlsx"
    # This has both the capacity in MW and the tonnes of waste incinerated every day (capacity)
    dm_capacity_waste_ots = extract_waste_capacity_ots(local_filename, years_ots)

    dm_capacity_waste_ots = adjust_based_on_nexuse(
        dm=dm_capacity_waste_ots, dm_nexuse=dm_capacity, years_ots=years_ots
    )

    return dm_capacity_waste_ots
