import numpy as np

import pandas as pd
import faostat
import os
import re
from model.common.data_matrix_class import DataMatrix
from model.common.constant_data_matrix_class import ConstantDataMatrix
from _database.pre_processing.api_routines_CH import get_data_api_CH
from model.common.auxiliary_functions import (
    create_years_list,
    linear_fitting,
    filter_DM,
    add_dummy_country_to_DM,
    my_pickle_dump,
)

import pickle
import json
import os
import numpy as np


def get_livestock_all(table_id, file, years_ots):
    try:
        with open(file, "rb") as handle:
            dm = pickle.load(handle)
            print(
                f"The livestock units are read from file {file}. Delete it if you want to update data from api."
            )
    except OSError:
        structure, title = get_data_api_CH(table_id, mode="example", language="fr")
        i = 0
        # The table is too big to be downloaded at once
        filtering = {
            "Unité d'observation": [
                "Cheptel - Bovins",
                "Cheptel - Equidés",
                "Cheptel - Moutons",
                "Cheptel - Chèvres",
                "Cheptel - Porcs",
                "Cheptel - Volailles",
                "Cheptel - Autres animaux",
            ],
            "Canton": ["Vaud"],
            "Zone de production agricole": ["Zone de production agricole - total"],
            "Classe de taille": ["Classe de taille - total"],
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

        # Extract new fleet
        dm = get_data_api_CH(
            table_id,
            mode="extract",
            filter=filtering,
            mapping_dims=mapping_dim,
            units=["animals"],
            language="fr",
        )
        dm.rename_col(
            "Zone de production agricole - total", "agr_livestock", "Variables"
        )
        dm.rename_col_regex("Cheptel - ", "", dim="Categories1")
        dict_cat = {
            "bovine": ["Bovins"],
            "sheep": ["Moutons"],
            "pig": ["Porcs"],
            "poultry": ["Volailles"],
            "oth-animals": ["Equidés", "Chèvres", "Autres animaux"],
        }
        dm.groupby(dict_cat, dim="Categories1", inplace=True)
        dm.sort("Years")
        dm.filter({"Years": years_ots}, inplace=True)

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, "wb") as handle:
            pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dm = linear_fitting(dm, years_ots)
    return dm


def get_livestock_dairy_egg(table_id, file, years_ots):
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
            "Unité d'observation": [
                "Cheptel - Vaches laitières",
                "Cheptel - Brebis laitières",
                "Cheptel - Chèvres laitières",
                "Cheptel - Poules de ponte et d'élevage",
            ],
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

        # Extract new fleet
        dm = get_data_api_CH(
            table_id,
            mode="extract",
            filter=filtering,
            mapping_dims=mapping_dim,
            units=["animals"],
            language="fr",
        )
        dm.rename_col(
            "Zone de production agricole - total",
            "agr_livestock-dairy-egg",
            "Variables",
        )
        dm.rename_col_regex("Cheptel - ", "", dim="Categories1")
        dict_cat = {
            "bovine": ["Vaches laitières"],
            "sheep": ["Brebis laitières"],
            "oth-animals": ["Chèvres laitières"],
            "poultry": ["Poules de ponte et d'élevage"],
        }
        dm.groupby(dict_cat, dim="Categories1", inplace=True)
        dm.sort("Years")
        dm.filter({"Years": years_ots}, inplace=True)

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, "wb") as handle:
            pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dm.array[dm.array == 0] = np.nan
    dm = linear_fitting(dm, years_ots)
    return dm


def compute_livestock(dm_all, dm_dairy_egg):
    missing_cat = list(
        set(dm_all.col_labels["Categories1"])
        - set(dm_dairy_egg.col_labels["Categories1"])
    )

    dm_meat = dm_all.filter({"Categories1": dm_dairy_egg.col_labels["Categories1"]})
    dm_meat.append(dm_dairy_egg, dim="Variables")
    dm_meat.operation(
        "agr_livestock",
        "-",
        "agr_livestock-dairy-egg",
        out_col="agr_livestock-meat",
        unit="animals",
    )
    dm_meat.filter({"Variables": ["agr_livestock-meat"]}, inplace=True)

    dm_meat_other = dm_all.filter({"Categories1": missing_cat})
    dm_meat_other.rename_col("agr_livestock", "agr_livestock-meat", dim="Variables")

    dm_meat.append(dm_meat_other, dim="Categories1")
    dm_meat.sort("Categories1")

    # add 'meat-' to categories
    for cat in dm_meat.col_labels["Categories1"]:
        dm_meat.rename_col(cat, "meat-" + cat, dim="Categories1")

    dm_meat.rename_col("agr_livestock-meat", "agr_livestock", dim="Variables")

    dm_dairy_egg.groupby(
        {
            "abp-dairy-milk": ["bovine", "oth-animals", "sheep"],
            "abp-hens-egg": ["poultry"],
        },
        dim="Categories1",
        inplace=True,
    )
    dm_dairy_egg.rename_col("agr_livestock-dairy-egg", "agr_livestock", "Variables")
    dm_lsu = dm_meat
    dm_lsu.append(dm_dairy_egg, dim="Categories1")
    dm_lsu.sort("Categories1")

    return dm_lsu


def get_crop_prod(table_id, file, years_ots):
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
            "grassland": [
                "Prairies artificielles",
                "Pâturages",
                "Prairies extensives",
                "Prairies peu intensives",
                "Prairies dans la région d'estivage",
                "Autres prairies permanentes",
            ],
            "remove": [
                "Plantes aromatiques et médicinales annuelles",
                "Plantes aromatiques et médicinales pluriannuelles",
                "Arbrisseaux ornementaux",
                "Sapins de Noël",
                "Pépinières forestières hors forêt sur SAU",
                "Autres pépinières",
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


def compute_wine_production(dm_crop_wine, wine_hl, conversion_hl_to_kcal):
    years_wine = list(wine_hl.keys())
    years_wine.sort()
    arr_wine = [wine_hl[yr] for yr in years_wine]
    arr_wine = np.array(arr_wine)
    dm_wine = DataMatrix(
        col_labels={
            "Country": ["Vaud"],
            "Years": years_wine,
            "Variables": ["agr_production"],
            "Categories1": ["pro-bev-wine"],
        },
        units={"agr_production": "hl"},
    )
    dm_shape = tuple([len(dm_wine.col_labels[dim]) for dim in dm_wine.dim_labels])
    dm_wine.array = np.zeros(shape=dm_shape)
    dm_wine.array[0, :, 0, 0] = arr_wine
    missing_years = list(set(years_ots) - set(dm_wine.col_labels["Years"]))
    dm_wine.add(np.nan, dummy=True, dim="Years", col_label=missing_years)

    dm_crop_wine.append(dm_wine, dim="Variables")
    dm_crop_wine.operation(
        "agr_production",
        "/",
        "agr_land-use",
        out_col="agr_climate-smart-crop_yield",
        unit="hl/ha",
    )
    dm_crop_wine.drop(col_label="agr_production", dim="Variables")
    dm_crop_wine.fill_nans("Years")
    dm_crop_wine.operation(
        "agr_land-use",
        "*",
        "agr_climate-smart-crop_yield",
        out_col="agr_production",
        unit="hl",
    )
    # Wine calories
    # https://www.calories.info/food/wine ( avg 80 Calories/100 gr = 80 kcal/100 gr = 800 kcal/l -> 80'000 kcal/hl )
    # FIXME you are here
    dm_crop_wine.change_unit(
        "agr_production", old_unit="hl", new_unit="kcal", factor=conversion_hl_to_kcal
    )
    dm_crop_wine.filter({"Variables": ["agr_production"]}, inplace=True)
    return dm_crop_wine


def compute_crop_prod(
    dm_crop_land, dm_yield, wine_hl_VD, conversion_hl_to_kcal, beer_proxy
):
    dm_crop_only = dm_crop_land.filter_w_regex(
        {"Categories1": "crop-.*"}, inplace=False
    )
    dm_crop_only.rename_col_regex("crop-", "", dim="Categories1")
    dm_crop_only.drop(col_label="stm", dim="Categories1")
    dm_yield_crop = dm_yield.filter(
        {"Categories1": dm_crop_only.col_labels["Categories1"]}, inplace=False
    )
    dm_crop_only.append(dm_yield_crop, dim="Variables")
    dm_crop_only.operation(
        "agr_land-use",
        "*",
        "agr_climate-smart-crop_yield",
        out_col="agr_production",
        unit="kcal",
    )
    dm_crop_only.filter({"Variables": ["agr_production"]}, inplace=True)
    for cat in dm_crop_only.col_labels["Categories1"]:
        cat_new = "crop-" + cat
        dm_crop_only.rename_col(cat, cat_new, "Categories1")

    # Wine production
    # The Federal Office of Agriculture, report the hl of wine produced and the ha of land dedicated to vineyards.
    # https://www.blw.admin.ch/fr/vin
    # the reports are available for every year, and every canton, in pdf format. The values from 2021 to 2024
    # one hl of wine is 80'000 kcal
    dm_crop_wine = dm_crop_land.filter({"Categories1": ["pro-bev-wine"]}, inplace=False)
    dm_crop_wine = compute_wine_production(
        dm_crop_wine, wine_hl_VD, conversion_hl_to_kcal
    )

    # Beer production
    # Since the cultivation of hop is small in Vaud and no data could be found on beer production we use crop-cereal
    dm_crop_beer = dm_crop_land.filter({"Categories1": ["pro-bev-beer"]}, inplace=False)
    dm_yield_beer = dm_yield.filter({"Categories1": [beer_proxy]}, inplace=False)
    dm_yield_beer.rename_col(beer_proxy, "pro-bev-beer", "Categories1")
    dm_crop_beer.append(dm_yield_beer, dim="Variables")
    dm_crop_beer.operation(
        "agr_land-use",
        "*",
        "agr_climate-smart-crop_yield",
        out_col="agr_production",
        unit="kcal",
    )
    dm_crop_beer.filter({"Variables": ["agr_production"]}, inplace=True)

    dm_crop_only.append(dm_crop_wine, dim="Categories1")
    dm_crop_only.append(dm_crop_beer, dim="Categories1")
    dm_crop_only.sort("Categories1")

    return dm_crop_only


def rename_categories(dm, cat_list, dummy_val):
    missing_cat = []
    for cat in dm.col_labels["Categories1"]:
        cat_full_name = [full_name for full_name in cat_list if cat in full_name]
        if len(cat_full_name) > 1:
            print(f"{cat} is matching with {cat_full_name}")
        elif len(cat_full_name) == 1:
            dm.rename_col(cat, cat_full_name[0], dim="Categories1")
        else:
            missing_cat.append(cat)

    missing_dummy_cat = list(set(cat_list) - set(dm.col_labels["Categories1"]))
    dm.add(dummy_val, dim="Categories1", col_label=missing_dummy_cat, dummy=True)

    dm.sort("Categories1")

    return dm, missing_cat


def extract_variables_to_file():
    f = "../_database/data/datamatrix/agriculture.pickle"
    with open(f, "rb") as handle:
        DM_agriculture = pickle.load(handle)

    def extract_all_variables_to_list(DM, list_var):
        for key in DM:
            if key == "constant" or key == "fts":
                continue
            if isinstance(DM[key], dict):
                list_var = extract_all_variables_to_list(DM[key], list_var)
            else:
                dm = DM[key].flattest()
                new_vars = [f"{k}[{v}]" for k, v in dm.units.items()]
                list_var.append(new_vars)
        return list_var

    list_agr_var = []
    list_agr_var = extract_all_variables_to_list(DM_agriculture, list_agr_var)
    df = pd.DataFrame(
        {"Column1": [str(lst) for lst in list_agr_var]}
    )  # Convert lists to strings
    df.to_excel(
        "../_database/pre_processing/agriculture & land use/data/agriculture_var_list.xlsx",
        index=False,
        engine="openpyxl",
    )

    return


years_ots = create_years_list(1990, 2023, 1)
years_fts = create_years_list(2025, 2050, 5)


# Population
data_file = "../../data/datamatrix/lifestyles.pickle"
with open(data_file, "rb") as handle:
    DM_lfs = pickle.load(handle)
dm_pop = DM_lfs["ots"]["pop"]["lfs_population_"].filter(
    {"Country": ["Vaud"]}, inplace=False
)
dm_demo = DM_lfs["ots"]["pop"]["lfs_demography_"].filter(
    {"Country": ["Vaud"]}, inplace=False
)

# Load Agriculture pickle to read fake Vaud data (as Switzerland)
data_file = "../../data/datamatrix/agriculture.pickle"
with open(data_file, "rb") as handle:
    DM_agriculture = pickle.load(handle)
filter_DM(DM_agriculture, {"Country": ["Switzerland"]})
add_dummy_country_to_DM(DM_agriculture, "Vaud", "Switzerland")
filter_DM(DM_agriculture, {"Country": ["Vaud"]})

# dm_kcal = DM_agriculture['ots']['kcal-req']
# linear_fitting(dm_kcal, years_ots)
# linear_fitting(dm_kcal, years_fts)

# Section: Production - Animals
# All livestock
table_id = "px-x-0702000000_101"
file = "data/agr_livestock_all.pickle"
dm_lsu_meat = get_livestock_all(table_id, file, years_ots)
# Dairy livestock
table_id = "px-x-0702000000_108"
file = "data/agr_livestock_dairy_egg.pickle"
dm_lsu_dairy_egg = get_livestock_dairy_egg(table_id, file, years_ots)
# All livestock (meat, dairy, egg)
dm_lsu = compute_livestock(dm_lsu_meat, dm_lsu_dairy_egg)
# Convert animals to lsu
lsu_conversion = {
    "meat-bovine": 0.6,
    "meat-oth-animals": 0.8,
    "meat-pig": 0.22,
    "meat-poultry": 0.007,
    "meat-sheep": 0.1,
    "abp-dairy-milk": 0.7,
    "abp-hens-egg": 0.014,
}
idx = dm_lsu.idx
for cat in dm_lsu.col_labels["Categories1"]:
    dm_lsu.array[:, :, idx["agr_livestock"], idx[cat]] = (
        lsu_conversion[cat] * dm_lsu.array[:, :, idx["agr_livestock"], idx[cat]]
    )
dm_lsu.change_unit("agr_livestock", old_unit="animals", new_unit="lsu", factor=1)

# Slaughtered = Stock x slaughtered-rate
dm_slaughtered_rate = DM_agriculture["ots"]["climate-smart-livestock"][
    "climate-smart-livestock_slaughtered"
]
dm_slaughtered_rate.filter_w_regex({"Categories1": "meat.*"}, inplace=True)
dm_lsu_meat = dm_lsu.filter_w_regex({"Categories1": "meat.*"}, inplace=False)
dm_lsu_other = dm_lsu.filter_w_regex({"Categories1": "abp.*"}, inplace=False)
dm_lsu_meat.append(dm_slaughtered_rate, dim="Variables")
dm_lsu_meat.operation(
    "agr_livestock",
    "*",
    "agr_climate-smart-livestock_slaughtered",
    out_col="agr_slaughtered",
    unit="lsu",
)
dm_lsu_meat.filter({"Variables": ["agr_slaughtered"]}, inplace=True)
dm_lsu_meat.rename_col("agr_slaughtered", "agr_livestock", dim="Variables")

dm_lsu = dm_lsu_other
dm_lsu.append(dm_lsu_meat, dim="Categories1")

# Multiply dm_lsu by yield to obtain total kcal produced
dm_yield = DM_agriculture["ots"]["climate-smart-livestock"][
    "climate-smart-livestock_yield"
]
if "meat-oth-animals" not in dm_yield.col_labels["Categories1"]:
    idx = dm_yield.idx
    arr_poultry = dm_yield.array[:, :, :, idx["meat-poultry"]]
    dm_yield.add(arr_poultry, dim="Categories1", col_label=["meat-oth-animals"])
linear_fitting(dm_yield, years_ots)

# Production
dm_lsu.append(dm_yield, dim="Variables")
dm_lsu.operation(
    "agr_livestock",
    "*",
    "agr_climate-smart-livestock_yield",
    out_col="agr_production",
    unit="kcal",
)

# Section: Production - Crop & Co
table_id = "px-x-0702000000_106"
file = "data/agr_crop_prod.pickle"
# Production in *ha* / Land-Use
dm_crop_land = get_crop_prod(table_id, file, years_ots)
# Production in kcal = Production in ha x yield in kcal/ha
dm_yield = DM_agriculture["ots"]["climate-smart-crop"]["climate-smart-crop_yield"]
linear_fitting(dm_yield, years_ots)
# Wine production
# The Federal Office of Agriculture, report the hl of wine produced and the ha of land dedicated to vineyards.
# https://www.blw.admin.ch/fr/vin
# the reports are available for every year, and every canton, in pdf format. The values from 2021 to 2024
wine_hl_VD = {
    2015: 218026,
    2019: 278474,
    2020: 237740,
    2021: 191463,
    2022: 273762,
    2023: 287379,
}  # 2024: 230916}
conversion_hl_to_kcal = 80000
beer_proxy = "cereal"
dm_crop_prod = compute_crop_prod(
    dm_crop_land, dm_yield, wine_hl_VD, conversion_hl_to_kcal, beer_proxy
)

# Production join crop and lsu
dm_prod = dm_lsu.filter({"Variables": ["agr_production"]}, inplace=False)
dm_prod.append(dm_crop_prod, dim="Categories1")

# Section: Supply
# Compute supply = Demand + Waste
# Demand = append( Consumer-diet,  )
# Consumer-diet-other = ( kcal-req - sum(Consumer-diet) )*Share
# Share
dm_share = DM_agriculture["ots"]["diet"]["share"].normalise(
    "Categories1", inplace=False
)
linear_fitting(dm_share, years_ots)
# Consumer-diet
dm_diet = DM_agriculture["ots"]["diet"]["lfs_consumers-diet"]
linear_fitting(dm_diet, years_ots)
# kcal-req
dm_kcal_req = DM_agriculture["ots"]["kcal-req"]
linear_fitting(dm_kcal_req, years_ots)
dm_kcal_req.append(dm_demo, dim="Variables")
dm_kcal_req.operation(
    "agr_kcal-req", "*", "lfs_demography", out_col="agr_kcal", unit="kcal/day"
)
dm_kcal_req.group_all("Categories1")
dm_kcal_req.operation(
    "agr_kcal", "/", "lfs_demography", out_col="agr_kcal-req_avg", unit="kcal/cap/day"
)
dm_kcal_req.filter({"Variables": ["agr_kcal-req_avg"]}, inplace=True)

#  Consumer-diet-other = ( kcal-req - sum(Consumer-diet) )*Share
dm_tot_diet = dm_diet.group_all("Categories1", inplace=False)
arr_kcal_other = (
    dm_kcal_req.array[:, :, :, np.newaxis] - dm_tot_diet.array[:, :, :, np.newaxis]
) * dm_share.array[:, :, :, :]
dm_share.add(
    arr_kcal_other,
    dim="Variables",
    col_label=["lfs_consumers-diet"],
    unit=["kcal/cap/day"],
)
dm_diet_other = dm_share.filter({"Variables": ["lfs_consumers-diet"]})

# Demand = append( Consumer-diet, Consumer-diet-other )
dm_diet.append(dm_diet_other, "Categories1")

# Supply = Waste + Demand
dm_waste = DM_agriculture["ots"]["fwaste"]
linear_fitting(dm_waste, years_ots)
dm_supply = dm_waste
# if 'rice' not in dm_supply.col_labels['Categories1']:
# dm_supply.add(np.nan, dim='Categories1', col_label=['crop-rice'], dummy=True)
dm_supply.append(dm_diet, dim="Variables")
dm_supply.operation(
    "lfs_consumers-food-wastes",
    "+",
    "lfs_consumers-diet",
    out_col="lfs_supply",
    unit="kcal/cap/day",
)

# Net-import = Production/Suppy ( = production/(production + import - export))
dm_netimport_dummy = DM_agriculture["ots"]["food-net-import"]
dm_supply.change_unit(
    "lfs_supply", old_unit="kcal/cap/day", new_unit="kcal/cap", factor=365
)
idx = dm_supply.idx
dm_supply.array[:, :, idx["lfs_supply"], :] = (
    dm_supply.array[:, :, idx["lfs_supply"], :] * dm_pop.array[:, :, :]
)
dm_supply.change_unit("lfs_supply", old_unit="kcal/cap", new_unit="kcal", factor=1)
dm_supply.rename_col(
    ["pigs", "cereals", "fruits", "oilcrops", "pulses", "afats", "sugar"],
    ["pig", "cereal", "fruit", "oilcrop", "pulse", "afat", "pro-crop-processed-sugar"],
    dim="Categories1",
)
dm_supply, missing_cat_sup = rename_categories(
    dm_supply, dm_netimport_dummy.col_labels["Categories1"], dummy_val=np.nan
)
dm_prod, missing_cat_prod = rename_categories(
    dm_prod, dm_netimport_dummy.col_labels["Categories1"], dummy_val=0
)

dm_supply.drop(col_label=missing_cat_sup, dim="Categories1")
dm_supply.append(dm_prod, dim="Variables")
dm_supply.operation(
    "agr_production", "/", "lfs_supply", out_col="agr_food-net-import", unit="%"
)
# FIXME! How to compute processed cake, molasse, sugar etc.
# FIXME! Difference between VD and CH

# Overwrite Vaud net-import in DM
DM_agriculture["ots"]["food-net-import"].drop(col_label="Vaud", dim="Country")
DM_agriculture["ots"]["food-net-import"].append(
    dm_supply.filter({"Variables": ["agr_food-net-import"]}), dim="Country"
)
DM_agriculture["ots"]["food-net-import"].sort("Country")

# Overwrite Vaud calibration ###################################################

## Livestock Production
dm_prod.rename_col_regex("pro-liv-", "", dim="Categories1")
dm_prod.rename_col_regex(
    "agr_production", "cal_agr_domestic-production-liv", dim="Variables"
)
dm_prod.drop(col_label=["abp-processed-afat", "abp-processed-offal"], dim="Categories1")
DM_agriculture["fxa"]["cal_agr_domestic-production-liv"] = dm_prod.filter(
    {
        "Categories1": [
            "abp-dairy-milk",
            "abp-hens-egg",
            "meat-bovine",
            "meat-oth-animals",
            "meat-pig",
            "meat-poultry",
            "meat-sheep",
        ]
    }
)

## Livestock Population
dm_lsu_cal = dm_lsu.copy()
dm_lsu_cal.filter({"Variables": ["agr_livestock"]}, inplace=True)
dm_lsu_cal.rename_col_regex("agr_livestock", "cal_agr_liv-population", dim="Variables")
DM_agriculture["fxa"]["cal_agr_liv-population"] = dm_lsu_cal

## Livestock emission N2O
dm_manure_n = DM_agriculture["fxa"]["liv_manure_n-stock"].copy()
dm_manure_n2o = DM_agriculture["fxa"]["ef_liv_N2O-emission"].copy()
dm_manure_n.filter({"Years": years_ots}, inplace=True)
dm_manure_n2o.filter({"Years": years_ots}, inplace=True)

arr_n2o_cal = (
    dm_manure_n[:, :, "fxa_liv_manure_n-stock", :, np.newaxis]
    * dm_manure_n2o[:, :, "fxa_ef_liv_N2O-emission_ef", :, :]
    * dm_lsu_cal[:, :, "cal_agr_liv-population", :, np.newaxis]
)
dm_manure_n2o.add(
    arr_n2o_cal, dim="Variables", col_label=["cal_agr_liv_N2O-emission"], unit=["kt"]
)

dm_manure_n2o.switch_categories_order("Categories2", "Categories1")
dm_manure_n2o.filter({"Variables": ["cal_agr_liv_N2O-emission"]}, inplace=True)
DM_agriculture["fxa"]["cal_agr_liv_N2O-emission"] = dm_manure_n2o

## Livestock emission CH4 (enteric & treated)
dm_enteric_ch4 = DM_agriculture["ots"]["climate-smart-livestock"][
    "climate-smart-livestock_enteric"
].copy()
dm_manure_ch4 = DM_agriculture["ots"]["climate-smart-livestock"][
    "climate-smart-livestock_manure"
].copy()
dm_ef_ch4 = DM_agriculture["fxa"]["ef_liv_CH4-emission_treated"].copy()
dm_ef_ch4.filter({"Years": years_ots}, inplace=True)
dm_manure_ch4.filter({"Categories2": ["treated"]}, inplace=True)

arr_ch4_cal = (
    dm_manure_n[:, :, "fxa_liv_manure_n-stock", :, np.newaxis]
    * dm_manure_ch4[:, :, "agr_climate-smart-livestock_manure", :, :]
    * dm_lsu_cal[:, :, "cal_agr_liv-population", :, np.newaxis]
    * dm_ef_ch4[:, :, "fxa_ef_liv_CH4-emission_treated", :, np.newaxis]
)
dm_manure_ch4.add(
    arr_ch4_cal, dim="Variables", col_label=["cal_agr_liv_CH4-emission"], unit=["kt"]
)

dm_manure_ch4.switch_categories_order("Categories2", "Categories1")
dm_manure_ch4.filter({"Variables": ["cal_agr_liv_CH4-emission"]}, inplace=True)

arr_enteric_cal = (
    dm_enteric_ch4[:, :, "agr_climate-smart-livestock_enteric", :]
    * dm_lsu_cal[:, :, "cal_agr_liv-population", :]
)
dm_enteric_ch4.add(
    arr_enteric_cal,
    dim="Variables",
    col_label=["cal_agr_liv_CH4-emission"],
    unit=["kt"],
)

### Formating
# dm_manure_ch4.switch_categories_order("Categories2","Categories1")
dm_manure_ch4.filter({"Variables": ["cal_agr_liv_CH4-emission"]}, inplace=True)
dm_enteric_ch4.filter({"Variables": ["cal_agr_liv_CH4-emission"]}, inplace=True)
dm_enteric_ch4.deepen(based_on="Variables")
dm_enteric_ch4.rename_col("cal_agr_liv", "cal_agr_liv_CH4-emission", dim="Variables")
dm_enteric_ch4.rename_col("CH4-emission", "enteric", dim="Categories2")
dm_enteric_ch4.switch_categories_order("Categories2", "Categories1")

dm_manure_ch4.append(dm_enteric_ch4, dim="Categories1")

### Overwriting DM
DM_agriculture["fxa"]["cal_agr_liv_CH4-emission"] = dm_manure_ch4

## Domestic production (crop)
dm_prod_crop = dm_prod.copy()
dm_prod_crop.rename_col_regex("crop-", "", dim="Categories1")
dm_prod_crop.rename_col_regex(
    "cal_agr_domestic-production-liv",
    "cal_agr_domestic-production_food",
    dim="Variables",
)
dm_prod_crop.filter(
    {
        "Categories1": [
            "cereal",
            "fruit",
            "oilcrop",
            "pulse",
            "rice",
            "starch",
            "sugarcrop",
            "veg",
        ]
    },
    inplace=True,
)

### Overwriting DM
DM_agriculture["fxa"]["cal_agr_domestic-production_food"] = dm_prod_crop

## Feed Demand
### Extracting DM
dm_fcr = DM_agriculture["constant"]["cdm_cp_efficiency"]
dm_pasture = DM_agriculture["ots"]["ruminant-feed"]["ruminant-feed"].copy()
dm_feed_split = DM_agriculture["ots"]["climate-smart-livestock"][
    "climate-smart-livestock_ration"
].copy()
dm_meat_prod = dm_prod.copy()
dm_meat_prod.filter(
    {
        "Categories1": [
            "abp-dairy-milk",
            "abp-hens-egg",
            "meat-bovine",
            "meat-oth-animals",
            "meat-pig",
            "meat-poultry",
            "meat-sheep",
        ]
    },
    inplace=True,
)
dm_kcal_to_tons = DM_agriculture["constant"]["cdm_kcal-per-t"].copy()
dm_kcal_to_tons.rename_col_regex(str1="pro-liv-", str2="", dim="Categories1")
dm_kcal_to_tons = dm_kcal_to_tons.filter(
    {
        "Categories1": [
            "abp-dairy-milk",
            "abp-hens-egg",
            "meat-bovine",
            "meat-oth-animals",
            "meat-pig",
            "meat-poultry",
            "meat-sheep",
        ]
    }
)


### Feed total demand
arr_feed_total_demand = (
    dm_meat_prod[:, :, "cal_agr_domestic-production-liv", :]
    / dm_kcal_to_tons[np.newaxis, np.newaxis, "cp_kcal-per-t", :]
    * dm_fcr[np.newaxis, np.newaxis, "cp_efficiency_liv", :]
)
dm_meat_prod.add(
    arr_feed_total_demand,
    dim="Variables",
    col_label=["cal_agr_demand_feed"],
    unit=["t"],
)
dm_feed_demand = dm_meat_prod.copy()
dm_feed_demand.filter({"Variables": ["cal_agr_demand_feed"]}, inplace=True)

### Feed demand excluding grass

dm_feed_demand.groupby(
    {"ruminants": ["abp-dairy-milk", "meat-bovine", "meat-sheep"]},
    inplace=True,
    regex=False,
    dim="Categories1",
)

dm_feed_demand.groupby(
    {"non-ruminants": ["abp-hens-egg", "meat-oth-animals", "meat-pig", "meat-poultry"]},
    inplace=True,
    regex=False,
    dim="Categories1",
)

arr_feed_grass_demand = (
    dm_feed_demand[:, :, "cal_agr_demand_feed", "ruminants"]
    * dm_pasture[:, :, "agr_ruminant-feed_share-grass"]
)
dm_feed_demand.add(
    arr_feed_grass_demand[:, :, np.newaxis, np.newaxis],
    dim="Categories1",
    col_label=["grass-feed"],
)

dm_feed_demand.operation(
    "ruminants", "+", "non-ruminants", out_col="total", unit="t", dim="Categories1"
)
dm_feed_demand.operation(
    "total", "-", "grass-feed", out_col="feed-other", unit="t", dim="Categories1"
)

### Feed Split for calibration
arr_feed_split = (
    dm_feed_demand[:, :, "cal_agr_demand_feed", "feed-other", np.newaxis]
    * dm_feed_split[:, :, "agr_climate-smart-livestock_ration", :]
)
dm_feed_split.add(
    arr_feed_split, dim="Variables", col_label=["cal_agr_demand_feed"], unit=["t"]
)
dm_feed_split.filter({"Variables": ["cal_agr_demand_feed"]}, inplace=True)

### Overwriting DM
DM_agriculture["fxa"]["cal_agr_demand_feed"] = dm_feed_split

## Land use


print("Hello")

################################################################################
# Pickle overwriting
################################################################################

# my_pickle_dump(DM_agriculture, f)
