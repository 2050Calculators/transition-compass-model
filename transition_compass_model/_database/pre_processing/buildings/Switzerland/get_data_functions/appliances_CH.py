import os
import pickle

import numpy as np
import pandas as pd

from transition_compass_model.model.common.auxiliary_functions import linear_fitting
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def get_appliances_stock_new_energy(raw_filename, clean_filename):
    if not os.path.exists(clean_filename):
        # utf-8-sig: the BFE file starts with a byte-order mark before "Jahr"
        df = pd.read_csv(raw_filename, encoding="utf-8-sig")

        # Fixed German -> model-name mappings. These used to come from DeepL, but
        # its output changes over time (e.g. "Geraetekategorie" became "Category
        # of the riddle"), which broke the column lookups. The English device
        # names reproduce the earlier DeepL output that appliances_pipeline_CH
        # relies on.
        var_map = {
            "Neugeraeteabsatz_Stk": "bld_appliances_new",
            "Geraetebestand_Stk": "bld_appliances_stock",
            "Verbrauch_Neugeraete_kWh": "bld_appliances_electricity-demand_new",
            "Verbrauch_Geraetebestand_kWh": "bld_appliances_electricity-demand_stock",
        }
        device_map = {
            "Beamer": "Beamer",
            "Drucker": "Printer",
            "Dunstabzugshauben": "Extractor bonnets",
            "Elektroherde und Einbaubacköfen": "Electric cookers and built-in ovens",
            "Fernsehgeräte": "Televisions",
            "Geschirrspüler": "Dishwasher",
            "Kaffeemaschinen": "Coffee machines",
            "Kühlschränke": "Refrigerators",
            "Laptops": "Laptops",
            "Monitore": "Monitor",
            "Personal Computer": "Personal Computer",
            "Router / Modems": "Routers / Modems",
            "Server (Small Scale)": "Server Small Scale",
            "Settop-Boxen": "Set-top boxes",
            "Slate Computer (Tablets)": "Slate Computer Tablets",
            "Staubsauger": "Hoover",
            "Tiefkühlschränke": "Freezers",
            "USV": "USV",
            "Video/DVD/Blu-Ray": "Video/DVD/Blu-Ray",
            "Waschmaschinen": "Washing machines",
            "Waschtrockner": "Washer-dryer",
            "Wäschetrockner": "Tumble dryer",
        }
        unknown = set(df["Geraetekategorie"]) - set(device_map)
        if unknown:
            raise ValueError(
                f"New device categories in {raw_filename}: {sorted(unknown)}. "
                "Add them to device_map."
            )
        df["Categories1"] = df["Geraetekategorie"].map(device_map)
        df = df.rename(columns={"Jahr": "Years", **var_map})
        df = df.drop(columns=["Quelle", "Geraetekategorie"])
        # Step 1: Melt the dataframe to long format
        var_cols = list(set(df.columns) - {"Years", "Categories1"})
        df_melted = df.melt(
            id_vars=["Years", "Categories1"],
            value_vars=var_cols,
            var_name="Variable",
            value_name="Value",
        )
        # Step 2: Pivot it
        df_pivoted = df_melted.pivot_table(
            index="Years", columns=["Variable", "Categories1"], values="Value"
        )

        # Step 3: Flatten the column MultiIndex
        df_pivoted.columns = [f"{var}_{cat}" for var, cat in df_pivoted.columns]

        # Step 4: Reset index to get "Years" back as a column
        df_final = df_pivoted.reset_index()

        df_final["Country"] = "Switzerland"

        for col in set(df_final.columns) - {"Years", "Country"}:
            unit = "kWh" if "electricity-demand" in col else "number"
            df_final.rename(columns={col: f"{col}[{unit}]"}, inplace=True)

        dm = DataMatrix.create_from_df(df_final, num_cat=1)
        dm.sort("Variables")

        with open(clean_filename, "wb") as handle:
            pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(
            f"File {clean_filename} already exists. If you want to download again delete the file"
        )

        with open(clean_filename, "rb") as handle:
            dm = pickle.load(handle)

    return dm


def clean_country_names(dm):
    cantons_en = [
        "Aargau",
        "Appenzell Ausserrhoden",
        "Appenzell Innerrhoden",
        "Basel Landschaft",
        "Basel Stadt",
        "Bern",
        "Fribourg",
        "Geneva",
        "Glarus",
        "Graubünden",
        "Jura",
        "Lucerne",
        "Neuchâtel",
        "Nidwalden",
        "Obwalden",
        "Schaffhausen",
        "Schwyz",
        "Solothurn",
        "St. Gallen",
        "Thurgau",
        "Ticino",
        "Uri",
        "Valais",
        "Vaud",
        "Zug",
        "Zurich",
    ]
    cantons_fr = [
        "Argovie",
        "Appenzell Rh. Ext.",
        "Appenzell Rh. Int.",
        "Bâle Campagne",
        "Bâle Ville",
        "Berne",
        "Fribourg",
        "Genève",
        "Glaris",
        "Grisons",
        "Jura",
        "Lucerne",
        "Neuchâtel",
        "Nidwald",
        "Obwald",
        "Schaffhouse",
        "Schwytz",
        "Soleure",
        "Saint Gall",
        "Thurgovie",
        "Tessin",
        "Uri",
        "Valais",
        "Vaud",
        "Zoug",
        "Zurich",
    ]

    dm.rename_col_regex("Suisse", "Switzerland", "Country")
    dm.rename_col_regex(" /.*", "", dim="Country")
    dm.rename_col_regex("-", " ", dim="Country")
    dm.rename_col(cantons_fr, cantons_en, dim="Country")
    dm.sort("Country")
    return dm


def get_households_number(raw_filename, clean_filename):
    if not os.path.exists(clean_filename):
        sheets_dict = pd.read_excel(raw_filename, sheet_name=None)

        dm = None
        for name, sheet in sheets_dict.items():
            print(name)
            df = sheets_dict[name]
            df.columns = df.iloc[0]
            df = df.iloc[2:]
            df = df[["Canton", "Total"]]
            # Drop rows if they contain a nan
            df.dropna(axis=0, how="any", inplace=True)
            df.rename(
                columns={"Canton": "Country", "Total": "bld_households[number]"},
                inplace=True,
            )
            df["Years"] = int(name)
            df = df.replace("()", np.nan)
            if dm is None:
                dm = DataMatrix.create_from_df(df, num_cat=0)
            else:
                dm_yr = DataMatrix.create_from_df(df, num_cat=0)
                dm.append(dm_yr, dim="Years")
        dm.sort("Years")
        with open(clean_filename, "wb") as handle:
            pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(
            f"File {clean_filename} already exists. If you want to download again delete the file"
        )
        with open(clean_filename, "rb") as handle:
            dm = pickle.load(handle)

    return dm


def households_fill_missing_years(dm_households, dm_pop, years_ots, years_fts):
    years_all = years_ots + years_fts
    # Add missing years as nans
    missing_years = list(set(years_all) - set(dm_households.col_labels["Years"]))
    dm_households.add(np.nan, col_label=missing_years, dummy=True, dim="Years")
    dm_households.sort("Years")

    # Join dm_households and dm_pop
    dm_households.append(
        dm_pop.filter({"Country": dm_households.col_labels["Country"]}), dim="Variables"
    )
    dm_households.operation(
        "lfs_population_total",
        "/",
        "bld_households",
        out_col="lfs_household-size",
        unit="people",
    )

    linear_fitting(dm_households, years_ots)

    dm_households.drop("Variables", "bld_households")

    dm_households.operation(
        "lfs_population_total",
        "/",
        "lfs_household-size",
        out_col="bld_households",
        unit="number",
    )

    dm_households.drop("Variables", "lfs_population_total")

    return dm_households


def appliances_fill_missing_years(dm_appliances, dm_households, years_ots, years_fts):
    def tailored_fitting(dm):
        linear_fitting(
            dm, years_ots=list(range(1990, 2010)), based_on=list(range(2002, 2006))
        )
        dm.fill_nans("Years")
        idx = dm.idx
        dm.array[:, idx[2022] :, :, :] = np.nan
        linear_fitting(
            dm, years_ots=years_ots + years_fts, based_on=list(range(2018, 2021))
        )
        dm.array = np.maximum(0, dm.array)
        return dm

    dm_households_only = dm_households.filter(
        {"Variables": ["bld_households"], "Years": years_ots + years_fts}
    )
    # Filter out efficiency
    dm_eff = dm_appliances.filter({"Variables": ["bld_appliances_electricity-demand"]})
    dm_appliances.drop("Variables", "bld_appliances_electricity-demand")
    # Compute appliances per household
    dm_appliances.sort("Country")
    dm_households_only.sort("Country")
    dm_appliances.array = (
        dm_appliances.array / dm_households_only.array[..., np.newaxis]
    )
    # dm_appliances.change_unit('bld_appliances_stock', old_unit='number', new_unit='unit/household', factor=1)
    # Impose ownership for PC and monitors
    # Source US labor: https://www.bls.gov/opub/btn/archive/computer-ownership-up-sharply-in-the-1990s.pdf
    dm_appliances[:, 1990, "bld_appliances_stock", "PC"] = 0.15
    dm_appliances[:, 1990, "bld_appliances_stock", "monitor"] = 0.15
    dm_PC = dm_appliances.filter({"Categories1": ["PC", "monitor"]})
    dm_PC.fill_nans("Years")
    dm_appliances.drop("Categories1", ["PC", "monitor"])
    dm_appliances.append(dm_PC, "Categories1")
    # Fill nans
    dm_appliances = tailored_fitting(dm_appliances)
    idx = dm_appliances.idx
    dm_appliances.array[:, idx[2022] :, :, idx["monitor"]] = np.nan
    dm_appliances.fill_nans("Years")
    dm_appliances.array = (
        dm_appliances.array * dm_households_only.array[..., np.newaxis]
    )
    # According to EP2050+
    # EP2050+_Detailergebnisse 2020-2060_Private Haushalte_alle Szenarien_2022-05-17
    # Sheet 04 Elektrogeräte
    # The unitary energy demand in kWh/unit of appliances in 2050 will reduce to 60%-90% of the 2010 value in 2050, in BAU scenario
    # The average is 73%. We use 70% for simplicity
    # dm_eff =tailored_fitting(dm_eff)
    dm_eff[:, 2050, "bld_appliances_electricity-demand", :] = (
        0.7 * dm_eff[:, 2010, "bld_appliances_electricity-demand", :]
    )
    dm_eff[:, 2050, "bld_appliances_electricity-demand", "tumble-dryer"] = (
        0.35 * dm_eff[:, 2010, "bld_appliances_electricity-demand", "tumble-dryer"]
    )
    dm_eff[:, 2050, "bld_appliances_electricity-demand", "washing-machine"] = (
        0.55 * dm_eff[:, 2010, "bld_appliances_electricity-demand", "washing-machine"]
    )
    linear_fitting(dm_eff, years_ots=years_ots + years_fts)
    dm_appliances.append(dm_eff, dim="Variables")

    return dm_appliances


def extract_EP2050_cooking_energy_consumption(file_raw):
    df = pd.read_excel(file_raw, "Tabelle17")

    df = df[list(df.columns)[1:-1]]
    df.columns = df.iloc[3]
    df = df.iloc[4:14]
    df.set_index(["Verwendungszweck"], inplace=True)
    df.columns = df.columns.astype(int)

    df = df.loc[["Gas (-Herd)", "Holz (-Herd)", "Elektrizität"]]
    df.reset_index(inplace=True)
    df = df.T
    df.reset_index(inplace=True)
    df.columns = df.iloc[0]
    df = df.iloc[1:-1]
    root = "bld_energy-demand_cooking_"
    unit = "[PJ]"
    df.rename(
        columns={
            "Verwendungszweck": "Years",
            "Gas (-Herd)": root + "gas" + unit,
            "Holz (-Herd)": root + "heating-oil" + unit,
            "Elektrizität": root + "electricity" + unit,
        },
        inplace=True,
    )
    df["Country"] = "Switzerland"

    dm = DataMatrix.create_from_df(df, num_cat=1)

    dm.change_unit(
        "bld_energy-demand_cooking",
        old_unit="PJ",
        new_unit="kWh",
        factor=3.6 * 1e-9,
        operator="/",
    )

    return dm


def extract_EP2050_other_elec_demand(file_raw):
    df = pd.read_excel(file_raw, "Tabelle1")

    df = df[list(df.columns)[1:-1]]
    df.columns = df.iloc[3]
    df = df.iloc[4:14]
    df.set_index(["Verwendungszweck"], inplace=True)
    df.columns = df.columns.astype(int)

    df = df.loc["sonstige Elektrogeräte"].to_frame()
    df.reset_index(inplace=True)

    df.rename(
        columns={
            3: "Years",
            "sonstige Elektrogeräte": "bld_energy-demand-total_other_electricity[PJ]",
        },
        inplace=True,
    )
    df["Country"] = "Switzerland"

    dm = DataMatrix.create_from_df(df, num_cat=0)

    dm.change_unit(
        "bld_energy-demand-total_other_electricity",
        old_unit="PJ",
        new_unit="TWh",
        factor=3.6,
        operator="/",
    )

    return dm
