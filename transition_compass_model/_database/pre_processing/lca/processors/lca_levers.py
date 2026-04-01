import os
import numpy as np
import pandas as pd
import re
import warnings

warnings.simplefilter("ignore")

from transition_compass_model.model.common.data_matrix_class import DataMatrix

from _database.pre_processing.lca.get_data_functions.data_lca import get_data_lca
from transition_compass_model.model.common.auxiliary_functions import create_years_list


def get_material_footprint_df(current_file_directory, df_full):

    #############################################################
    ######################### MATERIALS #########################
    #############################################################

    #################
    ##### CLEAN #####
    #################

    # select only material footprint
    df = df_full.iloc[:, 0:-17]

    # fix materials in column names
    materials = list(df.columns[6:])
    for i in range(0, len(materials)):
        if "Plastic" not in materials[i]:
            materials[i] = (materials[i][0:-5]).lower().replace(" ", "-")
        else:
            materials[i] = materials[i][0].lower() + (materials[i][1:-5]).replace(
                " ", "-"
            )
    materials_dict = dict(zip(list(df.columns[6:]), materials))
    for key in materials_dict.keys():
        df.rename(columns={key: materials_dict[key]}, inplace=True)

    # add missing materials
    # TODO: add in ecoinvent these materials as columns
    missing_mat = ["cement", "lime", "paper", "timber"]
    for m in missing_mat:
        df[m] = np.nan

    # reshape
    df.drop(["Ecoinvent name", "Reference product", "Location"], axis=1, inplace=True)
    df = pd.melt(
        df, ["product", "Scenario_year", "Reference unit"], var_name="material"
    )

    # fix units
    units = df["Reference unit"].unique()
    units_dict_fix = dict(
        zip(units, ["num", "kg", "meter-year", "meter", "KWh", "MJ", "km"])
    )
    for key in units_dict_fix.keys():
        df.loc[df["Reference unit"] == key, "Reference unit"] = units_dict_fix[key]

    # from kg to tonnes
    df["value"] = df["value"] * 1e-3

    # rename product
    df["product"] = (
        df["product"] + "_" + df["material"] + "[t/" + df["Reference unit"] + "]"
    )
    df = df.loc[:, ["product", "Scenario_year", "value"]]

    df_mat = df.copy()

    # save
    file_path = os.path.join(
        current_file_directory, "../data/intermediate_databases/materials.xlsx"
    )
    if not os.path.exists(file_path):
        df_mat.to_excel(file_path)

    return df_mat


def make_aggregates_footprint():

    agg_prod_dict = {
        "HDV_BEV": "HDV.*_BEV",
        "HDV_FCEV": "HDV.*_FCEV",
        "HDV_ICE-diesel": "HDV.*_ICE-diesel",
        "HDV_ICE-gas": "HDV.*_ICE-gas",
        "HDV_PHEV-diesel": "HDV.*_PHEV-diesel",
        "LDV-EOL": "LDV-EOL.*",
        "battery-lion": "battery-lion_.*",
        "battery-lion-EOL": "battery-lion-EOL.*",
        "RES-other-hydroelectric": "RES-other-hydroelectric.*",
        "RES-solar-Pvroof": "RES-solar-Pvroof.*",
        "RES-wind-offshore": "RES-wind-offshore.*",
        "computer": "computer_.*",
        "deep-saline-formation": "deep-saline-formation.*",
        "trains_CEV": "metrotram_mt|trains_CEV",
    }

    agg_mat_dict = {
        "aluminium": ["aluminium"],
        "copper": ["copper"],
        "other": [
            "antimony",
            "arsenic",
            "barium",
            "cadmium",
            "chromium",
            "cobalt",
            "fibreglass",
            "gallium",
            "gold",
            "indium",
            "lead",
            "lubricating-oil",
            "magnesium",
            "mercury",
            "niobium",
            "nylon-66",
            "palladium",
            "platinum",
            "silicon",
            "silver",
            "tantalum",
            "tin",
            "titanium",
            "vanadium",
            "zinc",
            "graphite",
            "lithium",
            "manganese",
            "nickel",
            "cerium",
            "europium",
            "gadolinium",
            "lanthanum",
            "neodymium",
            "praseodymium",
            "terbium",
            "yttrium",
        ],
        # 'REEs':['cerium','europium', 'gadolinium','lanthanum','neodymium','praseodymium', 'terbium', 'yttrium'],
        "chem": [
            "plastic-PE",
            "plastic-PET",
            "plastic-PP",
            "plastic-PU",
            "plastic-PVC",
            "rubber",
        ],
        "steel": ["steel", "iron", "cast-iron"],
    }

    return agg_prod_dict, agg_mat_dict


def make_footprint_dm(
    df,
    first_year,
    years_start=None,
    years_end=None,
    years_gap=None,
    agg_prod_dict=None,
    agg_mat_dict=None,
    deepen_n_cat=1,
):

    # make dm
    df["Country"] = "Switzerland"
    df["Years"] = first_year
    df = df.loc[:, ["Country", "Years", "product", "value"]]
    df = df.pivot(
        index=["Country", "Years"], columns="product", values="value"
    ).reset_index()
    dm = DataMatrix.create_from_df(df, deepen_n_cat)

    # make all countries and years
    countries = ["EU27", "Vaud"]
    arr_temp = dm.array
    for c in countries:
        dm.add(arr_temp, "Country", c)
    if years_start is not None and years_end is not None and years_gap is not None:
        years_missing = list(range(years_start, years_end + years_gap, years_gap))
        arr_temp = dm.array
        for y in years_missing:
            dm.add(arr_temp, "Years", [y])
        dm.sort("Years")

    if agg_prod_dict is not None:
        dm.groupby(agg_prod_dict, "Variables", "mean", regex=True, inplace=True)

    if agg_mat_dict is not None:
        dm.groupby(agg_mat_dict, "Categories1", "sum", regex=False, inplace=True)

    # 'rail': 't/meter-year', 'road': 't/kg', 'trolley-cables': 't/meter'
    unit_rail = dm.units["rail"]
    unit_rail_numerator = unit_rail.split("/")[0]
    dm.change_unit(
        "rail", factor=1e3, old_unit=unit_rail, new_unit=unit_rail_numerator + "/km"
    )

    unit_road = dm.units["road"]
    unit_road_numerator = unit_road.split("/")[0]
    dm.change_unit(
        "road", factor=1e3, old_unit=unit_road, new_unit=unit_road_numerator + "/t"
    )
    # assumption: 25 000 t of total mass per kilometer (typical for a multilayer 2-lane road)
    dm.change_unit(
        "road",
        factor=25000,
        old_unit=unit_road_numerator + "/t",
        new_unit=unit_road_numerator + "/km",
    )

    unit_cable = dm.units["trolley-cables"]
    unit_cable_numerator = unit_rail.split("/")[0]
    dm.change_unit(
        "trolley-cables",
        factor=1e3,
        old_unit=unit_cable,
        new_unit=unit_cable_numerator + "/km",
    )

    # substitute missing with zeroes so that when flattening / deepening we keep dimensions
    dm.array[np.isnan(dm.array)] = 0

    return dm


def get_other_footprint_df(df_full):

    # select ELSE
    ncol = len(df_full.columns)
    df = df_full.iloc[:, [0, 1, 4] + list(range(ncol - 17, ncol))]

    # rename
    dict_rename = {
        "Energy demand: electricity [kWh]": "energy-demand-elec[KWh]",
        "Cumulative energy demand; non-renewable, fossil [MJ-eq]": "energy-demand-ff[MJeq]",
        "Ecological Footprint, total [square meter-year]": "ecological-footprint[sqm-year]",
        "Global warming potential, 100 years [kgCO2-eq]": "gwp-100years[kgCO2eq]",
        "Water Consumption [cubic meter]": "water-consumption[m3]",
        "Air pollutants: PM10 [kg]": "air-pollutant-pm10[kg]",
        "Air pollutants: PM2.5 [kg]": "air-pollutant-pm25[kg]",
        "Air pollutants: SO2 [kg]": "air-pollutant-so2[kg]",
        "Air pollutants: Ammonia [kg]": "air-pollutant-ammonia[kg]",
        "Air pollutants: NOx [kg]": "air-pollutant-nox[kg]",
        "Air pollutants: NMVOC [kg]": "air-pollutant-nmvoc[kg]",
        "Heavy metals, to soil: Arsenic [kg]": "heavy-metals-to-soil-arsenic[kg]",
        "Heavy metals, to soil: Cadmium [kg]": "heavy-metals-to-soil-cadmium[kg]",
        "Heavy metals, to soil: Chromium [kg]": "heavy-metals-to-soil-chromium[kg]",
        "Heavy metals, to soil: Lead [kg]": "heavy-metals-to-soil-lead[kg]",
        "Heavy metals, to soil: Mercury [kg]": "heavy-metals-to-soil-mercury[kg]",
        "Heavy metals, to soil: Nickel [kg]": "heavy-metals-to-soil-nickel[kg]",
    }
    for key in dict_rename.keys():
        df.rename(columns={key: dict_rename[key]}, inplace=True)

    # reshape
    df = pd.melt(
        df, ["product", "Scenario_year", "Reference unit"], var_name="variable"
    )

    # fix units
    units = df["Reference unit"].unique()
    units_dict_fix = dict(
        zip(units, ["num", "kg", "meter-year", "meter", "KWh", "MJ", "km"])
    )
    for key in units_dict_fix.keys():
        df.loc[df["Reference unit"] == key, "Reference unit"] = units_dict_fix[key]

    # rename product
    unit_numerator = [s.split("[")[1].split("]")[0] for s in df["variable"]]
    variables_without_unit = [s.split("[")[0] for s in df["variable"]]
    df["product"] = (
        df["product"]
        + "_"
        + variables_without_unit
        + "["
        + unit_numerator
        + "/"
        + df["Reference unit"]
        + "]"
    )
    df = df.loc[:, ["product", "Scenario_year", "value"]]

    df_else = df.copy()

    return df_else


def make_other_dm(
    current_file_directory,
    df,
    pattern,
    years_ots,
    agg_prod_dict,
    years_gap=1,
    deepen_n_cat=1,
):

    # subset
    index = [bool(re.search(pattern, s)) for s in df["product"]]
    df_sub = df.loc[index, :]

    # save
    file_path = os.path.join(
        current_file_directory, f"../data/intermediate_databases/{pattern}.xlsx"
    )
    if not os.path.exists(file_path):
        df_sub.to_excel(file_path)

    # get ots data
    df_ots = df_sub.loc[df_sub["Scenario_year"].isin(["SSP5-Base_2025"])]

    # ots
    dm_ots = make_footprint_dm(
        df_ots,
        first_year=years_ots[-1],
        years_start=years_ots[0],
        years_end=years_ots[-2],
        years_gap=years_gap,
        agg_prod_dict=agg_prod_dict,
        deepen_n_cat=deepen_n_cat,
    )

    return dm_ots


def run(years_ots):

    # directories
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # footprint data
    df_full = get_data_lca(current_file_directory)

    # get material footprint df
    df_mat = get_material_footprint_df(current_file_directory, df_full)

    # get ots data for material footprint
    df_ots_mat = df_mat.loc[df_mat["Scenario_year"].isin(["SSP5-Base_2025"])]

    # get aggregates
    agg_prod_dict, agg_mat_dict = make_aggregates_footprint()

    # make dm for material footprint
    dm_mat = make_footprint_dm(
        df_ots_mat,
        years_ots[-1],
        years_ots[0],
        years_ots[-2],
        1,
        agg_prod_dict,
        agg_mat_dict,
    )

    # get other footprint df
    df_other = get_other_footprint_df(df_full)

    # make dm for other footprints
    dm_ene_dem_elec = make_other_dm(
        current_file_directory, df_other, "energy-demand-elec", years_ots, agg_prod_dict
    )
    dm_ene_dem_ff = make_other_dm(
        current_file_directory, df_other, "energy-demand-ff", years_ots, agg_prod_dict
    )
    dm_eco = make_other_dm(
        current_file_directory, df_other, "ecological", years_ots, agg_prod_dict
    )
    dm_gwp = make_other_dm(
        current_file_directory, df_other, "gwp", years_ots, agg_prod_dict
    )
    dm_water = make_other_dm(
        current_file_directory, df_other, "water", years_ots, agg_prod_dict
    )
    dm_air = make_other_dm(
        current_file_directory, df_other, "air-pollutant", years_ots, agg_prod_dict
    )
    dm_heavy_metals = make_other_dm(
        current_file_directory, df_other, "heavy-metals", years_ots, agg_prod_dict
    )

    return (
        dm_mat,
        dm_ene_dem_elec,
        dm_ene_dem_ff,
        dm_eco,
        dm_gwp,
        dm_water,
        dm_air,
        dm_heavy_metals,
    )


if __name__ == "__main__":

    years_ots = create_years_list(1990, 2023, 1)
    # years_fts = create_years_list(2025, 2050, 5)

    run(years_ots)
