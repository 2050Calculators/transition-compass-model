# packages
import os
import pickle
import warnings

warnings.simplefilter("ignore")
import numpy as np
import pandas as pd

from transition_compass_model._database.pre_processing.industry.eu.get_data_functions.data_fxa_energy_demand import (
    get_fxa_energy_demand_df,
)
from transition_compass_model.model.common.auxiliary_functions import create_years_list
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def make_fxa_energy_demand(current_file_directory, lever_file, years_ots, years_fts):
    df_final, df_final_feedstock = get_fxa_energy_demand_df(current_file_directory)

    # full model year range: OTS (1990–2023 yearly) + FTS (2025–2050 quinquennial)
    years_model = years_ots + years_fts

    # JRC year range available in data
    years_jrc = sorted(df_final["year"].unique().tolist())
    year_first = min(years_jrc)
    year_last = max(years_jrc)

    def merge_elec_carriers(df):
        # Collapse lighting and electricity-else into electricity in variable names,
        # then sum duplicate (year, tech, variable) rows.
        df = df.copy()
        df["variable"] = df["variable"].str.replace(
            r"_lighting\[", "_electricity[", regex=True
        )
        df["variable"] = df["variable"].str.replace(
            r"_electricity-else\[", "_electricity[", regex=True
        )
        df = df.groupby(
            ["year", "variable", "tech", "energy_demand_type"], as_index=False
        )["value"].sum()
        return df

    def build_dm(df_src, prefix, fec_or_ued):
        """
        Pivot year-indexed long-format df to wide DataMatrix-ready form,
        extrapolate to full model year range, and return a DataMatrix.
        """
        df = df_src.loc[df_src["energy_demand_type"] == fec_or_ued, :].copy()
        df = merge_elec_carriers(df)

        # prepend variable prefix
        df["variable"] = prefix + "_" + df["variable"]

        # pivot to wide: rows = years, columns = variables
        df_pivot = df.pivot_table(
            index="year", columns="variable", values="value", aggfunc="sum"
        ).reset_index()
        df_pivot.columns.name = None

        # extrapolate: clamp model years to JRC range, then duplicate rows
        rows = []
        for model_yr in years_model:
            jrc_yr = int(np.clip(model_yr, year_first, year_last))
            # if that exact JRC year is missing (gap), pick nearest available
            if jrc_yr not in df_pivot["year"].values:
                available = np.array(df_pivot["year"].tolist())
                jrc_yr = int(available[np.argmin(np.abs(available - jrc_yr))])
            row = df_pivot.loc[df_pivot["year"] == jrc_yr, :].copy()
            row["year"] = model_yr
            rows.append(row)
        df_full = pd.concat(rows, ignore_index=True)

        # add Country column expected by DataMatrix.create_from_df
        df_full["Country"] = "EU27"
        df_full = df_full.rename(columns={"year": "Years"})
        var_cols = sorted([c for c in df_full.columns if c not in ["Country", "Years"]])
        df_full = df_full[["Country", "Years"] + var_cols]

        dm = DataMatrix.create_from_df(df_full, num_cat=2)
        dm.units[prefix] = "TWh/Mt"
        return dm

    dm_excl = build_dm(df_final, "energy-demand-excl-feedstock", "fec")
    dm_feed = build_dm(df_final_feedstock, "energy-demand-feedstock", "fec")

    # pad dm_feed to the same Categories2 (carriers) and Categories1 (techs) as dm_excl,
    # filling absent combinations with zero — matches the old CDM behaviour
    for carrier in dm_excl.col_labels["Categories2"]:
        if carrier not in dm_feed.col_labels["Categories2"]:
            dm_feed.add(0, "Categories2", carrier, dummy=True)
    dm_feed.sort("Categories2")
    for tech in dm_excl.col_labels["Categories1"]:
        if tech not in dm_feed.col_labels["Categories1"]:
            dm_feed.add(0, "Categories1", tech, dummy=True)
    dm_feed.sort("Categories1")

    DM_energy_demand = {
        "energy-demand-excl-feedstock": dm_excl,
        "energy-demand-feedstock": dm_feed,
    }

    f = os.path.join(current_file_directory, "../data/datamatrix/" + lever_file)
    with open(f, "wb") as handle:
        pickle.dump(DM_energy_demand, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm_excl, dm_feed


def run(years_ots, years_fts):
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    lever_file = "fxa_energy-demand.pickle"
    filepath = os.path.join(current_file_directory, "../data/datamatrix/" + lever_file)
    if os.path.exists(filepath):
        with open(filepath, "rb") as handle:
            DM = pickle.load(handle)
        dm_excl = DM["energy-demand-excl-feedstock"].copy()
        dm_feed = DM["energy-demand-feedstock"].copy()
    else:
        dm_excl, dm_feed = make_fxa_energy_demand(
            current_file_directory, lever_file, years_ots, years_fts
        )

    return dm_excl, dm_feed


if __name__ == "__main__":
    years_ots = create_years_list(1990, 2023, 1)
    years_fts = create_years_list(2025, 2050, 5)
    run(years_ots, years_fts)
