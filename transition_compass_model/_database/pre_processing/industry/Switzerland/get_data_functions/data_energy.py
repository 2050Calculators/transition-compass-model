import pandas as pd
import os
import numpy as np
from transition_compass_model.model.common.data_matrix_class import DataMatrix


# read data
# note: data from https://www.bfe.admin.ch/bfe/fr/home/approvisionnement/statistiques-et-geodonnees/statistiques-de-lenergie/statistiques-sectorielles.html
def get_energy_demand_data(filepath, mapping, key):

    df = pd.read_excel(filepath, sheet_name=key)
    df.columns = df.iloc[2, :]
    df = df.iloc[4:5, :]
    df = pd.melt(
        df, id_vars=["N° de branche", "Nom de la branche", "Secteur"], var_name="Years"
    )
    df = df.loc[:, ["Years", "value"]]
    df["energy-carrier"] = mapping[key]

    return df


def data_energy(current_working_directory):

    # 'energy-demand': DataMatrix with shape (29, 22, 1, 10), variables ['calib-energy-demand-excl-feedstock'] and
    # categories1 ['electricity', 'gas-bio', 'gas-ff-natural', 'hydrogen', 'liquid-bio', 'liquid-ff-diesel', 'liquid-ff-oil', 'solid-bio', 'solid-ff-coal', 'solid-waste']

    # mapping of carriers
    mapping = {
        "Elektrizität": "electricity",
        "Erdgas": "gas-ff-natural",
        "Heizöl extra-leicht": "liquid-ff-diesel",
        "Heizöl mittel und schwer": "liquid-ff-oil",
        "Industrieabfälle": "solid-waste",
        "Kohle": "solid-ff-coal",
        "Fernwärme (Bezug)": "electricity",  # i assign district heating to electricity to be consistent on what has been done in EU
        "Holz": "solid-bio",
    }

    # note: uncovered are 'gas-bio', 'hydrogen', 'liquid-bio', I will put them to zero for now
    filepath = os.path.join(
        current_working_directory,
        "../data/energy-demand/8788-Publikationstabellen_DE_FR_IT_2013_bis_2024.xlsx",
    )
    df1 = pd.concat(
        [get_energy_demand_data(filepath, mapping, key) for key in mapping.keys()]
    )
    filepath = os.path.join(
        current_working_directory,
        "../data/energy-demand/12231-Publikationstabellen_DE_FR_IT_1999_bis_2013.xlsx",
    )
    df2 = pd.concat(
        [get_energy_demand_data(filepath, mapping, key) for key in mapping.keys()]
    )
    df2 = df2.loc[df2["Years"] != 2013, :]
    df = pd.concat([df1, df2])
    df = df.groupby(["energy-carrier", "Years"], as_index=False)["value"].agg("sum")
    df.sort_values(["energy-carrier", "Years"], inplace=True)

    # fix zeros as nan
    df.loc[df["value"] == 0, "value"] = np.nan

    # make dm
    df["energy-carrier"] = df["energy-carrier"] + "[TJ]"
    df = df.pivot(
        index=["Years"], columns="energy-carrier", values="value"
    ).reset_index()
    df["Country"] = "Switzerland"
    dm = DataMatrix.create_from_df(df, 0)
    # dm_tot = dm.groupby({"total" : ['electricity', 'liquid-ff-diesel', 'liquid-ff-oil', 'gas-ff-natural', 'solid-bio', 'solid-ff-coal', 'solid-waste']},
    #                     "Variables", inplace=False)

    return dm
