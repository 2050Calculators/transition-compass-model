import os
import pickle
import pandas as pd
import numpy as np

from model.common.auxiliary_functions import filter_DM, my_pickle_dump
from _database.pre_processing.api_routines_CH import get_data_api_CH
from model.common.auxiliary_functions import (
    create_years_list,
    linear_fitting,
    filter_DM,
    add_dummy_country_to_DM,
)


###############################################################################
# Read Data
###############################################################################
def read_pickle_agriculture():

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../../data/datamatrix/agriculture.pickle")
    with open(f, "rb") as handle:
        DM_pickle = pickle.load(handle)

        return DM_pickle


def read_pickle_lifestyle():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../../data/datamatrix/lifestyles.pickle")
    with open(f, "rb") as handle:
        DM_pickle = pickle.load(handle)

        return DM_pickle


###############################################################################
# Switzerland to Vaud : Duplicates
###############################################################################


def write_duplicate(dm_duplicate, canton):
    DM_agriculture = read_pickle_agriculture()

    # Result as tuple
    result = {}

    for path in dm_duplicate:
        # Find in DM
        source = DM_agriculture
        try:
            for key in path:
                source = source[key]
        except KeyError:
            continue

        # Filtering Switzerland & Renaming Canton
        if hasattr(source, "filter") and hasattr(source, "rename_col"):
            source = source.filter(({"Country": ["Switzerland"]}))
            source.rename_col("Switzerland", "Vaud", "Country")

        # Build selection
        cur_dst = result
        for key in path[:-1]:
            cur_dst = cur_dst.setdefault(key, {})
        cur_dst[path[-1]] = source

    return result


def dash_to_dm(row):
    # Extract DM specs
    keys = []
    dm_0 = row["DM"]
    if pd.notna(dm_0):
        keys.append(dm_0)
    # Add Sub-DM if present
    dm_1 = row.get("Sub-DM")
    if dm_1 is not None and pd.notna(dm_1):
        keys.append(dm_1)

    # Add Sub-sub-DM if present
    dm_2 = row.get("Sub-sub-DM")
    if dm_2 is not None and pd.notna(dm_2):
        keys.append(dm_2)
    return tuple(keys)


def downscale_dashboard(canton):
    # Read the downscale dashboard
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(
        current_file_directory, "downscale/agriculture_downscale_to_canton.xlsx"
    )
    dash = pd.read_excel(f, sheet_name=canton)
    dash = dash.drop(columns=["Comments"])

    # Extract the DM specs (duplicates)
    filtered_dash = dash[dash["Downscale"] == "duplicate"]
    dm_duplicate = [dash_to_dm(row) for _, row in filtered_dash.iterrows()]
    dm_duplicate = list(dict.fromkeys(dm_duplicate))

    return dm_duplicate


###############################################################################
# Switzerland to Canton : Constant
###############################################################################


def import_constant():
    DM_agriculture = read_pickle_agriculture()
    DM_constant = {}
    DM_constant["constant"] = DM_agriculture["constant"]
    return DM_constant


###############################################################################
# Switzerland to Canton : Calibration
###############################################################################


def diet_calibration(canton):
    # Import data (population & diet)
    DM_lifestyles = read_pickle_lifestyle()
    DM_agriculture = read_pickle_agriculture()

    # Filter canton
    filter_DM(DM_lifestyles, {"Country": [canton]})
    filter_DM(DM_agriculture, {"Country": [canton]})

    # Filter variables
    dm_diet_base = DM_agriculture["ots"]["diet"]["lfs_consumers-diet"]
    dm_diet_side = DM_agriculture["ots"]["diet"]["share"].normalise(
        "Categories1", inplace=False
    )
    dm_kcal = DM_agriculture["ots"]["kcal-req"]
    dm_waste = DM_agriculture["ots"]["fwaste"]
    dm_pop = DM_lifestyles["ots"]["pop"]["lfs_demography_"]
    dm_total_pop = DM_lifestyles["ots"]["pop"]["lfs_population_"]

    # Computation : Food Intake (total)
    dm_kcal.append(dm_pop, dim="Variables")
    dm_kcal.operation(
        "agr_kcal-req", "*", "lfs_demography", out_col="total_kcal", unit="kcal/day"
    )

    # Computation : Food Intake (average)
    dm_kcal.group_all("Categories1")
    dm_kcal.operation(
        "total_kcal",
        "/",
        "lfs_demography",
        out_col="total_kcal-req_avg",
        unit="kcal/cap/day",
    )
    dm_kcal.filter({"Variables": ["total_kcal-req_avg"]}, inplace=True)

    # Computation : Food Intake (side)
    dm_diet_total_base = dm_diet_base.group_all("Categories1", inplace=False)
    arr_kcal_side = (
        dm_kcal.array[:, :, :, np.newaxis]
        - dm_diet_total_base.array[:, :, :, np.newaxis]
    ) * dm_diet_side.array[:, :, :, :]
    dm_diet_side.add(
        arr_kcal_side,
        dim="Variables",
        col_label=["lfs_consumers-diet"],
        unit=["kcal/cap/day"],
    )
    dm_diet_side_split = dm_diet_side.filter({"Variables": ["lfs_consumers-diet"]})

    # Computation : Food Intake (split
    dm_diet_base.append(dm_diet_side_split, "Categories1")
    dm_diet_split = dm_diet_base

    # Food Supply (accounting for waste, per capita)
    dm_supply = dm_waste
    dm_supply.append(dm_diet_split, dim="Variables")
    dm_supply.operation(
        "lfs_consumers-food-wastes",
        "+",
        "lfs_consumers-diet",
        out_col="lfs_supply",
        unit="kcal/cap/day",
    )

    # Food Supply (accounting for waste, total)
    idx = dm_supply.idx
    arr_supply = (
        (dm_total_pop[:, :, "lfs_population_total", np.newaxis])
        * dm_supply[:, :, "lfs_supply", :]
        * 365
    )
    dm_supply.add(
        arr_supply, dim="Variables", col_label=["cal_agr_diet"], unit=["kcal"]
    )

    cal_agr_diet = dm_supply.filter({"Variables": ["cal_agr_diet"]})
    # cal_agr_diet = dm_supply

    return cal_agr_diet


def get_crop_prod(years_ots):
    # Inputs
    table_id = "px-x-0702000000_106"
    file = "data/agr_crop_prod.pickle"

    try:
        with open(file, "rb") as handle:
            dm = pickle.load(handle)
            print(
                f"The livestock units are read from file {file}. Delete it if you want to update data from api."
            )
    except OSError:
        structure, title = get_data_api_CH(table_id, mode="example", language="fr")

        # The table is too big to be downloaded at once
        filtering = {
            "Unité d'observation": structure["Unité d'observation"],
            "Canton": ["Vaud"],
            "Zone de production agricole": ["Zone de production agricole - total"],
            "Système d'exploitation": ["Système d'exploitation - total"],
            "Forme d'exploitation": ["Forme d'exploitation - total"],
            "Année": structure["Année"],
        }
        mapping_dim = {
            "Country": "Canton",
            "Years": "Année",
            "Variables": "Zone de production agricole",
            "Categories1": "Unité d'observation",
        }
        dm = get_data_api_CH(
            table_id,
            mode="extract",
            filter=filtering,
            mapping_dims=mapping_dim,
            units=["ha"],
            language="fr",
        )
        dm.drop(dim="Categories1", col_label=["Exploitations", "SAU - Total (en ha)"])
        dm.rename_col_regex("SAU - ", "", dim="Categories1")
        dm.rename_col_regex(" (en ha)", "", dim="Categories1")

        cat_map = {
            "crop-cereal": [
                "Blé",
                "Orge",
                "Avoine",
                "Seigle",
                "Triticale",
                "Epeautre",
                "Céréales en général",
                "Méteil et autres céréales panifiables",
                "Maïs grain",
                "Autres céréales",
                "Maïs d'ensilage et maïs vert",
            ],
            "crop-fruit": [
                "Baies annuelles",
                "Cultures de baies sous abri",
                "Cultures fruitières en général",
                "Pommes",
                "Poires",
                "Fruits à noyaux",
                "Baies pluriannuelles",
            ],
            "crop-oilcrop": [
                "Colza pour matière première renouvelable",
                "Tournesol pour matière première renouvelable",
                "Lin",
                "Chanvre",
                "Colza pour huile comestible",
                "Tournesol pour huile comestible",
                "Courge à huile",
            ],
            "crop-pulse": [
                "Pois protéagineux",
                "Féveroles",
                "Légumineuses en général",
                "Lupin fourrager",
                "Soja",
            ],
            "crop-starch": ["Pommes de terre"],
            "crop-sugarcrop": ["Betteraves sucrières"],
            "crop-veg": [
                "Cultures maraîchères de plein champ",
                "Cultures maraîchères sous abri",
                "Asperges",
                "Rhubarbe",
            ],
            "pro-bev-beer": ["Houblon"],
            "pro-bev-wine": ["Vigne"],
            "remove": [
                "Plantes aromatiques et médicinales annuelles",
                "Plantes aromatiques et médicinales pluriannuelles",
                "Arbrisseaux ornementaux",
                "Sapins de Noël",
                "Pépinières forestières hors forêt sur SAU",
                "Autres pépinières",
                "Prairies artificielles",
                "Pâturages",
                "Prairies extensives",
                "Prairies peu intensives",
                "Prairies dans la région d'estivage",
                "Autres prairies permanentes",
                "Surfaces à litières",
                "Haies, bosquets champêtres et berges boisées",
                "Betteraves fourragères",
                "Matières premières renouvelables annuelles",
                "Matières premières renouvelables pluriannuelles",
                "Autres SAU",
                "Tabac",
                "Jachère",
                "Autres terres ouvertes",
                "Méteil et autres céréales fourragères",
            ],
            "other": [
                "Cultures horticoles de plein champ annuelles",
                "Cultures horticoles sous abri",
                "Autres cultures pérennes",
                "Autres cultures sous abri",
                "Cultures sous abri en général",
            ],
        }
        # FIXME: Where should bettreve fourrageres go ?
        dm.groupby(cat_map, dim="Categories1", inplace=True)
        dm.drop(dim="Categories1", col_label=["remove"])
        dm.rename_col(
            "Zone de production agricole - total", "agr_land-use", dim="Variables"
        )

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, "wb") as handle:
            pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    linear_fitting(dm, years_ots)
    dm.filter({"Years": years_ots}, inplace=True)
    return dm


###############################################################################
###############################################################################
## Main
###############################################################################
###############################################################################

# Canton selection ############################################################
current_file_directory = os.path.dirname(os.path.abspath(__file__))
f = os.path.join(current_file_directory, "../../data/datamatrix/agriculture.pickle")
# Geoscale
canton = "Vaud"

# Timescale
years_ots = create_years_list(1990, 2023, 1)
years_fts = create_years_list(2025, 2050, 5)

dm = get_crop_prod(years_ots)

# Restore the Pickle ###########################################################

#
# DM_agriculture['constant']['cdm_cp_efficiency'].units = {'cp_efficiency_liv': 'kg DM feed/kg EW'}
# with open(f, 'wb') as handle:
#  pickle.dump(DM_agriculture, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Duplicates ##################################################################

dm_duplicate = downscale_dashboard(canton)
DM_duplicates = write_duplicate(dm_duplicate, canton)
DM_constant = import_constant()

# Calibration ##################################################################

cal_agr_diet = diet_calibration(canton)

# Data Matrix ##################################################################

DM_agriculture = read_pickle_agriculture()
DM_agriculture["fxa"]["cal_agr_diet"] = cal_agr_diet


# Overwriting Pickle ###########################################################
# my_pickle_dump(DM_agriculture, f)


print("Hello")
