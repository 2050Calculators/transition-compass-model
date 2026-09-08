import numpy as np
import pandas as pd

from transition_compass_model.model.common.data_matrix_class import DataMatrix


def extract_nexuse_capacity_data(file, years_ots, years_fts):
    dm = None
    for sheet_yr in ["2020", "2030", "2040", "2050"]:
        df_yr = pd.read_excel(file, sheet_name=sheet_yr)

        df_yr = df_yr.loc[df_yr["Country"] == "CH"]
        df_yr = df_yr[
            [
                "idGen",
                "GenName",
                "Technology",
                "GenEffic",
                "CO2Rate",
                "Pmax",
                "Pmin",
                "StartYr",
                "EndYr",
                "Emax",
                "SubRegion",
            ]
        ]

        df_eff_CO2 = df_yr.groupby(["Technology"])[["GenEffic", "CO2Rate"]].mean()

        years_all = years_ots + years_fts
        df = None
        for y in years_all:
            df_all = df_yr[(df_yr["StartYr"] <= y) & (df_yr["EndYr"] >= y)].copy()
            df_all["Years"] = y
            if df is None:
                df = df_all
            else:
                df = pd.concat([df, df_all], axis=0)

        df_P_E = df.groupby(["Technology", "SubRegion", "Years"])[
            ["Pmax", "Pmin", "Emax"]
        ].sum()

        df_P_E.reset_index(inplace=True)
        df_P_E.rename({"SubRegion": "Country"}, axis=1, inplace=True)

        df_T = df_P_E.pivot(
            index=["Country", "Years"],
            columns="Technology",
            values=["Pmax", "Pmin", "Emax"],
        )
        # df_T.columns = df_T.columns.swaplevel(0, 1)
        df_T.columns = ["_".join(col).strip() for col in df_T.columns.values]

        df_T.columns = ["pow_capacity-" + col for col in df_T.columns.values]
        cols = []
        for col in df_T.columns.values:
            if "Pmax" or "Pmin" in col:
                cols.append(col + "[MW]")
            elif "Emax" in col:
                cols.append(col + "[MWh]")
        df_T.columns = cols
        df_T.reset_index(inplace=True)

        # Make sure you have all years for all countries
        # Create a DataFrame with all combinations of countries and years
        countries = df_T["Country"].unique()
        complete_index = pd.MultiIndex.from_product(
            [countries, years_all], names=["Country", "Years"]
        )
        complete_df = pd.DataFrame(index=complete_index).reset_index()

        # Merge with the original DataFrame
        df_T = pd.merge(complete_df, df_T, on=["Country", "Years"], how="left")

        if dm is None:
            dm = DataMatrix.create_from_df(df_T, num_cat=1)
        else:
            dm_yr = DataMatrix.create_from_df(df_T, num_cat=1)
            extra_cntr = list(
                set(dm.col_labels["Country"]) - set(dm_yr.col_labels["Country"])
            )
            if len(extra_cntr) > 0:
                dm_yr.add(0, dim="Country", col_label=extra_cntr, dummy=True)
            extra_cntr = list(
                set(dm_yr.col_labels["Country"]) - set(dm.col_labels["Country"])
            )
            if len(extra_cntr) > 0:
                dm.add(0, dim="Country", col_label=extra_cntr, dummy=True)
            extra_tech = list(
                set(dm_yr.col_labels["Categories1"]) - set(dm.col_labels["Categories1"])
            )
            if len(extra_tech) > 0:
                dm.add(0, dim="Categories1", col_label=extra_tech, dummy=True)
            extra_tech = list(
                set(dm.col_labels["Categories1"]) - set(dm_yr.col_labels["Categories1"])
            )
            if len(extra_tech) > 0:
                dm_yr.add(0, dim="Categories1", col_label=extra_tech, dummy=True)
            dm_yr.sort("Country")
            dm_yr.sort("Categories1")
            dm.sort("Country")
            dm.sort("Categories1")
            dm.array = np.fmax(dm.array, dm_yr.array)
    dm.sort("Years")
    dm_CH = dm.groupby({"Switzerland": ".*"}, dim="Country", regex=True, inplace=False)
    dm.append(dm_CH, dim="Country")
    return dm, df_eff_CO2


def run(years_ots, years_fts):
    file = "data/Capacity_Nexuse.xlsx"
    dm_capacity, dm_const = extract_nexuse_capacity_data(file, years_ots, years_fts)
    dm_capacity_group = dm_capacity.copy()
    dm_capacity_group.groupby(
        {"Gas": "Gas.*", "Hydro": "Dam|RoR"},
        regex=True,
        dim="Categories1",
        inplace=True,
    )

    return dm_capacity, dm_const, dm_capacity_group
