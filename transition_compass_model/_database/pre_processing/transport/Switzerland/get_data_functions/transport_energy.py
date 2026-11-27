import os
import pickle
import zipfile

import numpy as np
import pandas as pd

from transition_compass_model._database.pre_processing.params import years_ots
from transition_compass_model._database.pre_processing.transport.Switzerland.get_data_functions import (
    utils as utils,
)
from transition_compass_model.model.common.auxiliary_functions import (
    dm_add_missing_variables,
    save_url_to_file,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

scenario_table = {
    "WWB": "Tabelle 04-05: Entwicklung des Energieverbrauchs im Szenario Weiter wie bisher",
    "ZERO-B": "Tabelle 04-03: Entwicklung des Energieverbrauchs im Szenario ZERO B",
}
scenario_year = {"WWB": 158, "ZERO-B": 88}


def extract_EP2050_transport_energy_demand(
    file_url, zip_name, file_pickle, scenario="WWB"
):
    try:
        with open(file_pickle, "rb") as handle:
            dm = pickle.load(handle)

    except OSError:
        extract_dir = os.path.splitext(zip_name)[0]  # 'data/EP2050_sectors'
        if not os.path.exists(extract_dir):
            save_url_to_file(file_url, zip_name)

            # Extract the file
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_name, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

        file_tra = (
            extract_dir
            + "/EP2050+_Szenarienergebnisse_Details_Nachfragesektoren/EP2050+_Detailergebnisse 2020-2060_Verkehrssektor_alle Szenarien_2022-04-12.xlsx"
        )
        df = pd.read_excel(file_tra, sheet_name="04 Energieverbrauch Strasse")

        df.drop(columns=[df.columns[0], df.columns[3]], inplace=True)

        table_title = scenario_table[scenario]
        start_table_row = df.index[df["Unnamed: 1"] == table_title].tolist()[1]
        df.columns = df.iloc[start_table_row + 2]

        df = df.iloc[start_table_row + 3 : start_table_row + 29]

        # Years as int
        col_mode_name = df.columns[0]
        col_tech_name = df.columns[1]
        df.set_index([col_mode_name, col_tech_name], inplace=True)
        df.columns = df.columns.astype(int)
        df.reset_index(inplace=True)

        # Change variables names
        full_name = ["tra_energy_demand_" + var for var in df[col_mode_name]]
        df[col_mode_name] = full_name
        df["Full_name"] = df[col_mode_name] + "_" + df[col_tech_name] + ["[PJ]"]
        df.drop(columns=[col_mode_name, col_tech_name], inplace=True)
        # Move "Full_name" column at the beginning
        first = df["Full_name"]
        df.drop(labels=["Full_name"], axis=1, inplace=True)
        df.insert(0, "Full_name", first)

        # Pivot
        df_T = df.T
        df_T.columns = df_T.iloc[0]
        df_T = df_T.iloc[1:]
        df_T.reset_index(inplace=True)
        df_T.rename(columns={scenario_year[scenario]: "Years"}, inplace=True)
        df_T["Country"] = "Switzerland"

        dm = DataMatrix.create_from_df(df_T, num_cat=2)

        # Rename mode of transport
        dm.rename_col(
            ["HGV", "LCV", "motorcycle", "pass. car"],
            ["HDVH", "HDVL", "2W", "LDV"],
            dim="Categories1",
        )
        dm.groupby({"bus": ["coach", "urban bus"]}, dim="Categories1", inplace=True)
        # Rename tech transport
        dm.groupby(
            {
                "BEV": ["electricity"],
                "biogasoline": ["E85"],
                "ICE-gas": ["CNG", "LPG"],
                "gasoline": ["petrol", "petrol 2S"],
                "FCEV": ["hydrogen"],
                "ICE-diesel": ["diesel"],
            },
            dim="Categories2",
            inplace=True,
        )

        with open(file_pickle, "wb") as handle:
            pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dm.sort("Categories1")
    dm.sort("Categories2")
    dm.change_unit(
        "tra_energy_demand", old_unit="PJ", new_unit="TWh", factor=3.6, operator="/"
    )

    return dm


def extract_EP2050_transport_energy_demand_rail(file_url, zip_name, file_pickle):
    try:
        with open(file_pickle, "rb") as handle:
            dm = pickle.load(handle)

    except OSError:
        extract_dir = os.path.splitext(zip_name)[0]  # 'data/EP2050_sectors'
        if not os.path.exists(extract_dir):
            save_url_to_file(file_url, zip_name)

            # Extract the file
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_name, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

        file_tra = (
            extract_dir
            + "/EP2050+_TechnsicherBericht_DatenAbbildungen_Kap 8_2022-04-12.xlsx"
        )
        df = pd.read_excel(file_tra, sheet_name="Abb. 112")

        df.drop(columns=[df.columns[0]], inplace=True)

        # Headers
        cols = list(df.iloc[28])
        cols[0] = "Variables"
        df.columns = cols

        df = df.iloc[29:-1]

        # Years as int

        col_name = "Variables"
        df.set_index([col_name], inplace=True)
        df.columns = df.columns.astype(int)
        df.reset_index(inplace=True)

        # Change variables names
        full_name = ["tra_energy_demand_" + var + "[PJ]" for var in df[col_name]]
        df[col_name] = full_name

        # Pivot
        df_T = df.T
        df_T.columns = df_T.iloc[0]
        df_T = df_T.iloc[1:]
        df_T.reset_index(inplace=True)
        df_T.rename(columns={"index": "Years"}, inplace=True)
        df_T["Country"] = "Switzerland"

        dm = DataMatrix.create_from_df(df_T, num_cat=0)

        # Rename mode of transport
        dm.rename_col_regex("Flugverkehr", "aviation", dim="Variables")
        dm.rename_col_regex("Güterverkehr", "freight", dim="Variables")
        dm.rename_col_regex("Personenverkehr", "passenger", dim="Variables")
        dm.rename_col_regex(" Schiene", "_rail", dim="Variables")
        dm.rename_col_regex(" Strasse", "_road", dim="Variables")

        with open(file_pickle, "wb") as handle:
            pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dm_aviation = dm.filter_w_regex({"Variables": ".*aviation.*"})
    dm_rail = dm.filter_w_regex({"Variables": ".*rail"})
    dm_rail.deepen()
    dm_rail.deepen(based_on="Variables")

    dm_rail.change_unit(
        "tra_energy_demand", old_unit="PJ", new_unit="TWh", factor=3.6, operator="/"
    )

    return dm_rail


def get_public_transport_energy_consumption(dm_passenger_energy_all):
    """Read sheet T1 from the FSO database su-f-11-TP-ZR.
    Transports publics (trafic marchandises rail inclus) - séries chronologiques détaillées.
    Tableau T1: Inputs - consommation d'énergie, surface utilisée
    Returns:
        pd.DataFrame: Dataframe with energy consumption for switzerland in TJ from 1990 to 2024
    """
    # get dir
    this_dir = os.path.dirname(os.path.abspath(__file__))
    local_filename = os.path.join(this_dir, "../data/tra_public_transport.xlsx")
    # Total energy consumption per canton in MJ
    # Formatting the df for transformation in datamatrix
    df = pd.read_excel(local_filename, sheet_name="T1", header=3)
    year_cols = [col for col in df.columns if isinstance(col, int)]
    df[year_cols] = df[year_cols].apply(pd.to_numeric, errors="coerce")
    df.reset_index(inplace=True, drop=True)
    df = df[["Unnamed: 0"] + year_cols]
    df.replace("…", np.nan, inplace=True)

    def format_df_for_dm(df_bus, cat1="bus"):
        df_T = df_bus.T
        df_T.columns = df_T.iloc[0]
        df_T = df_T.add_suffix("[TJ]")
        df_T = df_T.add_prefix(f"tra_energy_demand_{cat1}_")
        df_T.reset_index(inplace=True)
        df_T.drop(df.index[0], inplace=True)
        df_T.rename(columns={df_T.columns[0]: "Years"}, inplace=True)
        df_T["Country"] = "Switzerland"
        df_T.drop(df_T[df_T["Years"] == 2025].index, inplace=True)

        return DataMatrix.create_from_df(df_T, num_cat=2)

    df_bus_idx = df.loc[df["Unnamed: 0"] == "Autobus"].index[0]
    # Autobus energy by type: "Électrique ", "Diesel", "Autres"
    df_autobus = df.iloc[df_bus_idx + 2 : df_bus_idx + 5, :].copy()
    df_autobus.set_index("Unnamed: 0", inplace=True)
    # Before the split exists (2010), "Diesel" is the whole Autobus energy: the
    # electric and other rows are 0 in those years, not unknown.
    for row in ["Électrique ", "Autres"]:
        first_valid = df_autobus.loc[row].first_valid_index()
        if first_valid is not None:
            years_before = [y for y in year_cols if y < first_valid]
            df_autobus.loc[row, years_before] = 0
    # "Autres" (other fuels: gas, biofuels) is burnt in combustion buses: add it to
    # the combustion energy (kept in "Diesel" = ICE-diesel here). The bus efficiency
    # then uses one combustion efficiency for diesel, gas and petrol buses.
    df_autobus.loc["Diesel"] = df_autobus.loc["Diesel"] + df_autobus.loc["Autres"]
    df_autobus.drop(index="Autres", inplace=True)
    df_autobus.reset_index(inplace=True)
    df_bus = pd.concat([df_autobus, df.loc[df["Unnamed: 0"] == "Trolleybus"]])
    dm_bus_energy = format_df_for_dm(df_bus, cat1="bus")

    df_train_idx = df.loc[df["Unnamed: 0"] == "Chemins de fer"].index[0]
    df_train = df.iloc[df_train_idx + 2 : df_train_idx + 5, :]
    dm_train_energy = format_df_for_dm(df_train, cat1="rail")

    dm_bus_energy.rename_col(
        ["Trolleybus", "Électrique ", "Diesel"],
        ["CEV", "BEV", "ICE-diesel"],
        dim="Categories2",
    )
    dm_train_energy.rename_col(
        ["Électrique ", "Diesel"],
        ["CEV", "ICE-diesel"],
        dim="Categories2",
    )

    dm_train_energy.add(0, dim="Categories2", col_label="BEV", dummy=True)
    # Bus "Autres" is already in ICE-diesel; rail "Autres" is spread below
    dm_bus_energy.add(0, dim="Categories2", col_label="Autres", dummy=True)
    dm_tra_public_energy = dm_bus_energy.copy()
    dm_tra_public_energy.append(dm_train_energy, dim="Categories1")

    # interpolating missing years
    dm_add_missing_variables(
        dm_tra_public_energy, {"Years": years_ots}, fill_nans=False
    )
    dm_tra_public_energy.sort("Years")
    dm_tra_public_energy.fill_nans("Years")
    dm_tra_public_energy.filter({"Years": years_ots}, inplace=True)

    dm_tra_public_energy = utils.drop_if_smaller_than_0_01(
        dm_tra_public_energy, cat_to_drop="Categories2", col_to_drop="Autres"
    )

    # convert units
    dm_tra_public_energy.change_unit(
        "tra_energy_demand", old_unit="TJ", new_unit="TWh", factor=3600, operator="/"
    )

    dm_tra_public_energy.sort("Categories2")

    return dm_tra_public_energy


# def get_proportion_electric_bus():
#     """As there is no database separating electric bus vkm and diesel bus vkm. The FSO database su-f-11-TP-ZR is used.
#     Therewe we can found the energy consumption by type

#     Args:
#         dm (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     #Valeur obtenue de mobitool V3 environ 170.3 kWh/ 100km pour un bus électrique moyenne de de 2020 https://www.suisseenergie.ch/calculateur-environnemental-transport/
#     avg_electric_car_efficiency = 1.703 * 3.6 *0.000001 #TJ/Vkm
#     avg_diesel_car_efficiency = 3.8 *3.6*0.000001
#     # get dir
#     this_dir = os.path.dirname(os.path.abspath(__file__))
#     local_filename = os.path.join(this_dir, "../data/tra_public_transport.xlsx")
#     #Total energy consumption per canton in MJ
#     df = pd.read_excel(local_filename, sheet_name="T1", header = 3)
#     year_cols = [col for col in df.columns if isinstance(col, int)]
#     df[year_cols] = df[year_cols].apply(
#         pd.to_numeric, errors="coerce"
#     )
#     df.reset_index(inplace=True,drop =True)
#     df = df[['Unnamed: 0'] + year_cols]
#     df.fillna(0,inplace=True)
#     df.replace('…',0,inplace=  True)


#     # Calculate diesel VKM (consider other have similar energy consumption as diesel)
#     mask = df["Unnamed: 0"] == "Autobus"
#     bus_consumption_without_electric = df.loc[mask, year_cols].iloc[0] - df.iloc[23,1:]
#     diesel_vkm = bus_consumption_without_electric / avg_diesel_car_efficiency

#     # Calculate electric VKM
#     electric_vkm = df.iloc[23,1:]/ avg_electric_car_efficiency

#     total_vkm = diesel_vkm + electric_vkm

#     electric_veh_prop = electric_vkm/total_vkm

#     return electric_veh_prop
