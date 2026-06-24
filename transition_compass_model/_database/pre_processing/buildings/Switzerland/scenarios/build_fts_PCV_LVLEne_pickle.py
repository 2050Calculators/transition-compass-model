import os
import pickle

import numpy as np

from transition_compass_model._database.pre_processing.api_routines_CH import (
    get_data_api_CH,
)
from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    load_pop,
    my_pickle_dump,
    sort_pickle,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def get_renov_rate_E(DM_buildings, yrs_fts, household_type="multi-family-households"):
    # get the proportion of renovation for E buildings (not influenced by energy law) to add to the renovation rate of F buildings, as we assume that some of the renovation that would have been done in E will be done in F because of the energy law
    ren_redistribution = DM_buildings["fts"]["building-renovation-rate"][
        "bld_renovation-redistribution"
    ][2].copy()
    idx_redistrib = ren_redistribution.idx
    prop_E_renovated = ren_redistribution.array[
        idx_redistrib["Vaud"],
        idx_redistrib[2035],
        idx_redistrib["bld_renovation-redistribution-out"],
        idx_redistrib["E"],
    ]
    dm_rr_fts_2 = DM_buildings["fts"]["building-renovation-rate"][
        "bld_renovation-rate"
    ][2].copy()
    idx = dm_rr_fts_2.idx

    idx_fts = [idx[yr] for yr in yrs_fts]

    renovation_E = (
        dm_rr_fts_2.array[
            idx["Vaud"],
            idx_fts,
            idx["bld_renovation-rate"],
            idx[household_type],
        ]
        * prop_E_renovated
    )
    return renovation_E


def extract_stock_floor_area(table_id, file):
    try:
        with open(file, "rb") as handle:
            dm_floor_area = pickle.load(handle)
    except OSError:
        structure, title = get_data_api_CH(table_id, mode="example", language="fr")

        # Extract buildings floor area
        filter = {
            "Année": structure["Année"],
            "Canton (-) / District (>>) / Commune (......)": ["Suisse", "- Vaud"],
            "Catégorie de bâtiment": structure["Catégorie de bâtiment"],
            "Surface du logement": structure["Surface du logement"],
            "Époque de construction": structure["Époque de construction"],
        }
        mapping_dim = {
            "Country": "Canton (-) / District (>>) / Commune (......)",
            "Years": "Année",
            "Variables": "Surface du logement",
            "Categories1": "Catégorie de bâtiment",
            "Categories2": "Époque de construction",
        }
        unit_all = ["number"] * len(structure["Surface du logement"])
        # Get api data
        dm_floor_area = get_data_api_CH(
            table_id,
            mode="extract",
            filter=filter,
            mapping_dims=mapping_dim,
            units=unit_all,
            language="fr",
        )
        dm_floor_area.rename_col(
            ["Suisse", "- Vaud"], ["Switzerland", "Vaud"], dim="Country"
        )

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, "wb") as handle:
            pickle.dump(dm_floor_area, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dm_floor_area.groupby(
        {
            "single-family-households": ["Maisons individuelles"],
            "multi-family-households": [
                "Maisons à plusieurs logements",
                "Bâtiments d'habitation avec usage annexe",
                "Bâtiments partiellement à usage d'habitation",
            ],
        },
        dim="Categories1",
        inplace=True,
    )

    # There is something weird happening where the number of buildings with less than 30m2 built before
    # 1919 increases over time. Maybe they are re-arranging the internal space?
    # Save number of bld (to compute avg size)
    dm_num_bld = dm_floor_area.groupby(
        {"bld_stock-number-bld": ".*"}, dim="Variables", regex=True, inplace=False
    )

    ## Compute total floor space
    # Drop split by size
    dm_floor_area.rename_col_regex(" m2", "", "Variables")
    # The average size for less than 30 is a guess, as is the average size for 150+,
    # we will use the data from bfs to calibrate
    avg_size = {
        "<30": 25,
        "30-49": 39.5,
        "50-69": 59.5,
        "70-99": 84.5,
        "100-149": 124.5,
        "150+": 175,
    }

    dm_num_bld_per_size_per_type = dm_floor_area.copy()
    idx = dm_floor_area.idx
    for size in dm_floor_area.col_labels["Variables"]:
        dm_floor_area.array[:, :, idx[size], :, :] = (
            avg_size[size] * dm_floor_area.array[:, :, idx[size], :, :]
        )

    dm_floor_area.groupby(
        {"bld_floor-area_stock": ".*"}, dim="Variables", regex=True, inplace=True
    )
    dm_floor_area.change_unit("bld_floor-area_stock", 1, "number", "m2")

    return dm_floor_area, dm_num_bld, dm_num_bld_per_size_per_type


def replace_years_by_corresponding_categories_for_specified_household(
    dm_num_bld, env_cat, type_households="single-family-households"
):
    dm_num_bld_sfh = dm_num_bld.filter({"Categories1": [type_households]})
    dm_num_bld_sfh.groupby(env_cat, dim="Categories2", inplace=True)
    return dm_num_bld_sfh


def replace_years_by_corresponding_categories(dm_num_bld, env_cat_sfh, env_cat_mfh):
    dm_num_bld_sfh = replace_years_by_corresponding_categories_for_specified_household(
        dm_num_bld, env_cat_sfh, type_households="single-family-households"
    )
    dm_num_bld_mfh = replace_years_by_corresponding_categories_for_specified_household(
        dm_num_bld, env_cat_mfh, type_households="multi-family-households"
    )
    dm_bld = dm_num_bld_sfh
    dm_bld.append(dm_num_bld_mfh, dim="Categories1")
    return dm_bld


def compute_renovation_loi_energie(
    dm_stock_area: DataMatrix,
    dm_num_bld: DataMatrix,
    env_cat_mfh: dict,
    env_cat_sfh: dict,
    DM_buildings: DataMatrix,
    dm_num_bld_per_size_per_type: DataMatrix,
):
    """_summary_

    Args:
        dm_stock_area (DataMatrix): _description_
        dm_num_bld (DataMatrix): _description_
        env_cat_mfh (dict): _description_
        env_cat_sfh (dict): _description_
        DM_buildings (DataMatrix): _description_
        dm_num_bld_per_size_per_type (DataMatrix): _description_

    Returns:
        DataMatrix: DM_buildings["fts"]["building-renovation-rate"]["bld_renovation-rate"] for lever 3
        float: proportion between 0 and 1 of the E buildings renovated over the total buoildings renovated before 2035
        float: proportion between 0 and 1 of the E buildings renovated over the total buoildings renovated before 2040
        float: proportion between 0 and 1 of the E proportion of the area for  multi households buildings F and G that must be renovated over the total area of multi households buldings
    """

    dm_num_bld_per_size_per_cat = replace_years_by_corresponding_categories(
        dm_num_bld_per_size_per_type, env_cat_sfh, env_cat_mfh
    )
    dm_num_bld_per_size_F = dm_num_bld_per_size_per_cat.filter(
        {"Country": ["Vaud"], "Categories2": ["F"]}
    )

    # append the area in meter square to the datamatrix with the number of buildings.
    dm_num_bld.append(dm_stock_area, dim="Variables")
    dm_bld = replace_years_by_corresponding_categories(
        dm_num_bld, env_cat_sfh, env_cat_mfh
    )
    ###### NEW VERSION OF LAW : <750 m2 F and G renovated ######
    dm_num_bld_F = dm_bld.filter(
        {
            "Country": ["Vaud"],
            "Variables": ["bld_stock-number-bld"],
            "Categories2": ["F"],
        }
    )
    # Categories 2 is the CECB category as it filtered with only F we just remove a useless dimension
    dm_num_bld_F.group_all(dim="Categories2")
    dm_num_bld_per_size_F.group_all(dim="Categories2")
    dm_num_bld_F.append(dm_num_bld_per_size_F, dim="Variables")

    # We filter only for multi-family household sbecause they are probably the only one with superficy higher than 750 m2

    dm_single_multi_grouped = dm_num_bld_F.copy()
    dm_single_multi_grouped.group_all(dim="Categories1")
    dm_num_bld_F.filter({"Categories1": ["multi-family-households"]}, inplace=True)
    array_ratio = {}
    idx = dm_num_bld_F.idx
    # Iterate in the size of buildings and compute the ratio of number of buildings it correspond to
    for col in dm_num_bld_per_size_F.col_labels["Variables"]:
        array_ratio[col] = (
            dm_num_bld_F.array[0, idx[2023], idx[col], idx["multi-family-households"]]
            / dm_single_multi_grouped.array[0, idx[2023], idx["bld_stock-number-bld"]]
        )

        # dm_num_bld_F.operation(
        #     col,
        #     "/",
        #     "bld_stock-number-bld",
        #     out_col=f"ratio_num_bld_{col}",
        #     unit="%",
        # )

    resting_toget_to_20 = 0.20 - (array_ratio["150+"] + array_ratio["100-149"])

    # we only want the twentypercent  biggest buildings to be renovated
    # For this we need the percentage of dwellings with area between 100-149 meter square to be renovated and we renovate and the buildings
    # with area bigger than 150 meter
    percent_building_renvoated_70_99 = resting_toget_to_20 / array_ratio["70-99"]
    # We want to convert the number of building to the floor area that need to be renovated, and we assume that the average size of building bigger than 150m2 is 175m2
    # and the average size of building between 100 and 149 is 124.5 m2
    area_necessary_renovated = (
        dm_num_bld_F.array[idx["Vaud"], idx[2023], idx["150+"]] * 175
    )
    area_necessary_renovated += (
        dm_num_bld_F.array[idx["Vaud"], idx[2023], idx["100-149"]] * 124.5
    )
    area_necessary_renovated += (
        dm_num_bld_F.array[idx["Vaud"], idx[2023], idx["70-99"]]
        * percent_building_renvoated_70_99
        * 84.5
    )

    idx = dm_bld.idx
    # Get the ratio of renovation needed in 2040
    ren_rate_min_class_F = area_necessary_renovated / np.sum(
        dm_bld.array[
            idx["Vaud"],
            idx[2023],
            idx["bld_floor-area_stock"],
            idx["multi-family-households"],
            :,
        ]
    )
    dm_rr_bau = DM_buildings["fts"]["building-renovation-rate"]["bld_renovation-rate"][
        2
    ].copy()
    dm_rr_fts_2 = DM_buildings["fts"]["building-renovation-rate"][
        "bld_renovation-rate"
    ][2].copy()

    idx = dm_rr_fts_2.idx
    yrs_fts = [yr for yr in dm_rr_fts_2.col_labels["Years"] if yr <= 2040]
    idx_fts = [idx[yr] for yr in yrs_fts]
    renovation_E = get_renov_rate_E(DM_buildings, yrs_fts)

    perc_F = 0.5
    perc_G = 0.5
    # F buildings must be renovated before G buildings
    renovation_rate_F = ren_rate_min_class_F / (yrs_fts[-1] - yrs_fts[0] + 1) * perc_F
    renovation_rate_G = ren_rate_min_class_F / (yrs_fts[-2] - yrs_fts[0] + 1) * perc_G
    # Before 2035 all type G buildings and some type E
    dm_rr_fts_2.array[
        idx["Vaud"],
        idx_fts[:-1],
        idx["bld_renovation-rate"],
        idx["multi-family-households"],
    ] = (renovation_rate_G + renovation_rate_F) * 0.85 + renovation_E[:-1]

    # Renovation objective divided by the number of year to apply it
    dm_rr_fts_2.array[
        idx["Vaud"],
        idx_fts[-1],
        idx["bld_renovation-rate"],
        idx["multi-family-households"],
    ] = renovation_rate_F * 0.85 + renovation_E[-1]

    return dm_rr_fts_2, ren_rate_min_class_F


def update_renovation_out(renov_distrib_fts_3, prop_E_renovated):
    """update renovation out array with proportion of E buildings renovated"""
    idx = renov_distrib_fts_3.idx
    # Renovation out
    renov_distrib_fts_3.array[
        idx["Vaud"],
        idx[2025] : idx[2035] + 1,
        idx["bld_renovation-redistribution-out"],
        idx["E"],
    ] = prop_E_renovated[idx[2025] : idx[2035] + 1]

    renov_distrib_fts_3.array[
        idx["Vaud"],
        idx[2025] : idx[2035] + 1,
        idx["bld_renovation-redistribution-out"],
        idx["F"],
    ] = 1 - prop_E_renovated[idx[2025] : idx[2035] + 1]

    renov_distrib_fts_3.array[
        idx["Vaud"], idx[2040], idx["bld_renovation-redistribution-out"], idx["E"]
    ] = prop_E_renovated[idx[2040]]

    renov_distrib_fts_3.array[
        idx["Vaud"], idx[2040], idx["bld_renovation-redistribution-out"], idx["F"]
    ] = 1 - prop_E_renovated[idx[2040]]

    renov_distrib_fts_3.array[
        idx["Vaud"], idx[2045] :, idx["bld_renovation-redistribution-out"], idx["E"]
    ] = prop_E_renovated[idx[2045] :]

    renov_distrib_fts_3.array[
        idx["Vaud"], idx[2045] :, idx["bld_renovation-redistribution-out"], idx["F"]
    ] = 1 - prop_E_renovated[idx[2045] :]

    return renov_distrib_fts_3


def update_heating_change_proportion(
    dm_heating_cat_fts_2, household_type="multi-family-households"
):
    """when setting some technology to 0 in dm_heating_cat_fts_2 we need to replace them with other technology.
    Here we use the proportion from the study perspectives chaleur to replace them.

    Args:
        dm_heating_cat_fts_2 (Datamatrix): Datamatrix with the ratio of use of each techology.
        household_type (str): _description_. Defaults to "multi-family-households".

    Returns:
        Datamatrix: The updated datamatrix filtered with the relevant household type.
    """

    # update multi households
    dm_heating_fts_mfh = dm_heating_cat_fts_2.filter(
        {"Country": ["Vaud"], "Categories1": [household_type]}
    )

    idx = dm_heating_fts_mfh.idx
    # Once all the old technologies are set to 0 we want to replace the missing proportion with the ideal scenario proportion
    dm_sum = dm_heating_fts_mfh.group_all("Categories3", inplace=False)
    arr_sum = dm_sum.array[..., np.newaxis]
    proportion_replaced_heating_per_cat = 1 - arr_sum

    if household_type == "multi-family-households":
        # Proportion according to the study perspectives chaleur (fig. 1)
        renov_proportion = {
            "district-heating": 0.66,
            "heat-pump": 0.27,
            "solar": 0.04,
            "wood": 0.03,
        }
    else:
        # Proportion according to the study perspectives chaleur (fig 3.)
        renov_proportion = {
            "district-heating": 0.0044,  # 0.004426002766251729
            "heat-pump": 0.8488,
            "solar": 0.0835,
            "wood": 0.0633,
        }

    heating_types = list(renov_proportion.keys())
    prop_vec = np.array([renov_proportion[h] for h in heating_types])
    dm_heating_fts_mfh.array[:, :, :, :, :, [idx[h] for h in heating_types]] += (
        proportion_replaced_heating_per_cat * prop_vec
    )

    return dm_heating_fts_mfh


def update_heating_fts_2(dm_heating_cat_fts_2, dm_heating_cat_ots):
    idx_ots = dm_heating_cat_ots.idx
    idx = dm_heating_cat_fts_2.idx
    idx_old_cat = [idx["E"], idx["F"]]
    idx_new_cat = [idx["B"], idx["C"], idx["D"]]
    # Fossil heating
    # article  40.1
    idx_fossil = [idx["coal"], idx["heating-oil"], idx["gas"]]
    idx_ots_fossil = [idx_ots["coal"], idx_ots["heating-oil"], idx_ots["gas"]]
    idx_ots_new_cat = [idx_ots["B"], idx_ots["C"], idx_ots["D"]]
    # dm_heating_cat_fts_2.array[idx['Vaud'], :, idx['bld_heating-mix'], :, idx['B'], idx_fossil] = 0
    dm_heating_cat_fts_2.array[
        idx["Vaud"],
        1 : idx[2050],
        idx["bld_heating-mix"],
        :,
        :,
        idx_fossil,
    ] = np.nan
    dm_heating_cat_fts_2.array[
        idx["Vaud"],
        idx[2040],
        idx["bld_heating-mix"],
        :,
        :,
        idx_fossil,
    ] = (
        dm_heating_cat_ots.array[
            idx_ots["Vaud"],
            idx_ots[2023],
            idx_ots["bld_heating-mix"],
            :,
            :,
            idx_ots_fossil,
        ]
        * 0.25
    )

    # We suppose 5% of exception
    dm_heating_cat_fts_2.array[
        idx["Vaud"], idx[2045], idx["bld_heating-mix"], :, :, idx_fossil
    ] = (
        dm_heating_cat_ots.array[
            idx_ots["Vaud"],
            idx_ots[2023],
            idx_ots["bld_heating-mix"],
            :,
            :,
            idx_ots_fossil,
        ]
        * 0.05
    )
    dm_heating_cat_fts_2.array[
        idx["Vaud"], idx[2050], idx["bld_heating-mix"], :, :, idx_fossil
    ] = dm_heating_cat_fts_2.array[
        idx["Vaud"], idx[2045], idx["bld_heating-mix"], :, :, idx_fossil
    ]
    dm_heating_cat_fts_2.fill_nans("Years")

    dm_heating_fts_mfh = update_heating_change_proportion(
        dm_heating_cat_fts_2, "multi-family-households"
    )
    dm_heating_cat_fts_2["Vaud", :, :, "multi-family-households", :, :] = (
        dm_heating_fts_mfh["Vaud", :, :, "multi-family-households", :, :]
    )

    dm_heating_fts_sfh = update_heating_change_proportion(
        dm_heating_cat_fts_2, "single-family-households"
    )
    dm_heating_cat_fts_2["Vaud", :, :, "single-family-households", :, :] = (
        dm_heating_fts_sfh["Vaud", :, :, "single-family-households", :, :]
    )

    dm_heating_cat_fts_2.normalise("Categories3")
    dm_heating_cat_fts_2.fill_nans("Years")
    return dm_heating_cat_fts_2


def create_renov_prop_hw(dm_bld_mix):
    # Proportion according to the study perspectives chaleur (fig. 1)
    renov_proportion_multi = {
        "district-heating": 0.563,
        "heat-pump": 0.25,
        "solar": 0.067 + 0.063,
        "wood": 0.057,
    }
    # Proportion according to the study perspectives chaleur (fig 3.)
    renov_proportion_single = {
        "district-heating": 0.004,
        "heat-pump": 0.822,
        "solar": 0.081 + 0.031,
        "wood": 0.061,
    }
    dm_prop_multi_single = dm_bld_mix.filter(
        {"Country": ["Vaud"], "Variables": ["bld_floor-area_stock"]}
    ).copy()
    dm_prop_multi_single.group_all("Categories2")
    dm_prop_multi_single.normalise("Categories1")
    idx_prop_multi_single = dm_prop_multi_single.idx
    array_prop_multi_single = dm_prop_multi_single.array[
        idx_prop_multi_single["Vaud"],
        idx_prop_multi_single[2023],
        idx_prop_multi_single["bld_floor-area_stock"],
        :,
    ]

    # I want to do prop
    renov_prop_hw = {}
    for cat in renov_proportion_single.keys():
        renov_prop_hw[cat] = (
            renov_proportion_multi[cat] * array_prop_multi_single[0]
            + renov_proportion_single[cat] * array_prop_multi_single[1]
        )

    return renov_prop_hw


def compute_renov_fts_mapping(renov_distrib_fts: DataMatrix):
    # the energy law forces to renovate directly to class D

    # Got it from the programme batiment

    ren_nb_class = {
        1: 0.57,  # Amélioration de +1 classes CECB 57% des rénovations
        2: 0.15,  # Amélioration de +2 classes CECB 15%
        3: 0.15,
        4: 0.13,
    }

    idx = renov_distrib_fts.idx

    # et the percentage of buildings renovated that are E or F
    # renov_out_E = renov_distrib_fts.array[
    # idx["Vaud"], :, idx["bld_renovation-redistribution-out"], idx["E"]
    # ]
    # renov_out_F = renov_distrib_fts.array[
    # idx["Vaud"], :, idx["bld_renovation-redistribution-out"], idx["F"]
    # ]

    # renov_distrib_fts.array[
    # idx["Vaud"], :, idx["bld_renovation-redistribution-in"], idx["E"]
    # ] = np.round(renov_out_F* ren_nb_class[1], 2)

    # renov_distrib_fts.array[
    # idx["Vaud"], :, idx["bld_renovation-redistribution-in"], idx["D"]
    # ] = np.round(renov_out_F * ren_nb_class[2] +renov_out_E * ren_nb_class[1], 2)

    # renov_distrib_fts.array[
    # idx["Vaud"], :, idx["bld_renovation-redistribution-in"], idx["C"]
    # ] = np.round(renov_out_F* ren_nb_class[3] + renov_out_E * ren_nb_class[2], 2)

    # renov_distrib_fts.array[
    # idx["Vaud"], :, idx["bld_renovation-redistribution-in"], idx["B"]
    # ] =  np.round(
    # renov_out_F * ren_nb_class[4] + renov_out_E* ren_nb_class[3] +renov_out_E * ren_nb_class[4], 2
    # )

    # Set renovation to 0 for class E and transfer all renovation to class D
    renov_distrib_fts.array[
        idx["Vaud"], :, idx["bld_renovation-redistribution-in"], idx["D"]
    ] += renov_distrib_fts.array[
        idx["Vaud"], :, idx["bld_renovation-redistribution-in"], idx["E"]
    ]
    renov_distrib_fts.array[
        idx["Vaud"], :, idx["bld_renovation-redistribution-in"], idx["E"]
    ] = 0

    return renov_distrib_fts


def run(
    DM_buildings, dm_stock_cat, dm_pop, global_var, country_list, lev=2
):  # lever =2 for energy law and 3 for PCV 4 is perfect world 1 is BAU
    construction_period_envelope_cat_sfh = global_var["envelope construction sfh"]
    construction_period_envelope_cat_mfh = global_var["envelope construction mfh"]

    # SECTION: Loi Energie - Renovation fts
    # LEVEL 2 Vaud: Loi Energie + Plan Climat
    # According to the Loi Energie, buildings in categories F,G > 750 m2 will have to be renovated before 2035,
    # They estimate this corresponds to 90'000 multi-family-households being renovated before 2035.
    table_id = "px-x-0902020200_103"
    this_dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(this_dir, "../data/bld_floor-area_stock.pickle")
    dm_stock_area, dm_num_bld, dm_num_bld_per_size_per_type = extract_stock_floor_area(
        table_id, file
    )

    dm_stock_area = dm_stock_area.filter({"Country": country_list}).copy()
    dm_num_bld = dm_num_bld.filter({"Country": country_list}).copy()

    env_cat_mfh = construction_period_envelope_cat_mfh
    env_cat_sfh = construction_period_envelope_cat_sfh

    # Recompute stock_cat from DM_buildings
    dm_floor_cap = (
        DM_buildings["ots"]["floor-intensity"]
        .filter(
            {"Variables": ["lfs_floor-intensity_space-cap"], "Country": country_list}
        )
        .copy()
    )
    dm_bld_mix = (
        DM_buildings["ots"]["building-renovation-rate"]["bld_building-mix"]
        .filter({"Country": country_list})
        .copy()
    )
    arr_stock = (
        dm_floor_cap[:, :, :, np.newaxis, np.newaxis]
        * dm_pop[:, :, :, np.newaxis, np.newaxis]
        * dm_bld_mix[:, :, :, :, :]
    )
    dm_bld_mix.add(
        arr_stock, dim="Variables", col_label="bld_floor-area_stock", unit="m2"
    )
    # dm_stock_cat_bis = dm_bld_mix.filter({"Variables": ["bld_floor-area_stock"]})

    # Compute renovation rate loi energie
    dm_rr_fts_3, ren_rate_tot_above_750 = compute_renovation_loi_energie(
        dm_stock_area,
        dm_num_bld,
        env_cat_mfh,
        env_cat_sfh,
        DM_buildings,
        dm_num_bld_per_size_per_type,
    )

    ####Ren rate old version###
    renov_copy_old = DM_buildings["fts"]["building-renovation-rate"][
        "bld_renovation-rate"
    ][2].copy()
    # Get the proportion of area in every category by normalising the surface by the enveloppe area category
    dm_stock_copy = dm_stock_cat.copy()
    dm_stock_copy.normalise("Categories2")
    idx = dm_stock_copy.idx
    idx_renov_old = renov_copy_old.idx

    # Get the list of years before the application of the law and the associated indexes
    yrs_fts = [yr for yr in renov_copy_old.col_labels["Years"] if yr <= 2040]
    idx_fts = [idx_renov_old[yr] for yr in yrs_fts]

    renov_tot_F = {}
    renov_yr_E = {}
    for household_type in ["multi-family-households", "single-family-households"]:
        # Proportion of area in F categoory
        renov_tot_F[household_type] = dm_stock_copy.array[
            idx["Vaud"],
            idx[2023],
            idx["bld_floor-area_stock"],
            idx[household_type],
            idx["F"],
        ]
        renov_yr_E[household_type] = get_renov_rate_E(
            DM_buildings, yrs_fts, household_type=household_type
        )

    idx_renov_old = renov_copy_old.idx
    # single family households renvoaton
    renov_copy_old.array[
        idx_renov_old["Vaud"],
        idx_fts,
        idx_renov_old["bld_renovation-rate"],
        idx_renov_old["single-family-households"],
    ] = (
        renov_tot_F["single-family-households"] / (yrs_fts[-1] - yrs_fts[0] + 1)
    ) * 0.85 + renov_yr_E["single-family-households"]

    renov_copy_old.array[
        idx_renov_old["Vaud"],
        idx_renov_old[2045] :,
        idx_renov_old["bld_renovation-rate"],
        idx_renov_old["single-family-households"],
    ] = renov_yr_E["single-family-households"][-2:]

    # multi family household renovation
    ren_rate_tot_under_750 = (
        renov_tot_F["multi-family-households"] - ren_rate_tot_above_750
    )
    # before 2035
    renov_copy_old.array[
        idx_renov_old["Vaud"],
        idx_fts[:-1],
        idx_renov_old["bld_renovation-rate"],
        idx_renov_old["multi-family-households"],
    ] = (
        ren_rate_tot_under_750 / (yrs_fts[-1] - yrs_fts[0] + 1)
        + ren_rate_tot_above_750 / (yrs_fts[-2] - yrs_fts[0] + 1)
    ) * 0.85 + renov_yr_E["multi-family-households"][:-1]

    # 2040
    renov_copy_old.array[
        idx_renov_old["Vaud"],
        idx_fts[-1],
        idx_renov_old["bld_renovation-rate"],
        idx_renov_old["multi-family-households"],
    ] = (ren_rate_tot_under_750 / (yrs_fts[-1] - yrs_fts[0] + 1)) * 0.85 + renov_yr_E[
        "multi-family-households"
    ][-1]

    prop_E_renovated_before_2035_lev_4 = (
        renov_yr_E["single-family-households"][1]
        + renov_yr_E["multi-family-households"][1]
    ) / renov_copy_old.array[
        idx_renov_old["Vaud"],
        idx_fts[1],
        idx_renov_old["bld_renovation-rate"],
        :,
    ].sum()
    prop_E_renovated_in_2040_lev_4 = (
        renov_yr_E["single-family-households"][-1]
        + renov_yr_E["multi-family-households"][-1]
    ) / renov_copy_old.array[
        idx_renov_old["Vaud"],
        idx_fts[-1],
        idx_renov_old["bld_renovation-rate"],
        :,
    ].sum()

    # Add the BAU renovation rate of buildings uuder 750 m to all time series + after 2040 stop renovating above 750 because everything renovated already
    bau_ren_rate = DM_buildings["fts"]["building-renovation-rate"][
        "bld_renovation-rate"
    ][2].copy()

    idx = dm_rr_fts_3.idx
    ren_rate_yr_under_750_bau_multi = (
        (1 - ren_rate_tot_above_750 / renov_tot_F["multi-family-households"])
        * 0.8
        * bau_ren_rate.array[
            idx["Vaud"],
            :,
            idx["bld_renovation-rate"],
            idx["multi-family-households"],
        ]
    )

    dm_rr_fts_3.array[
        idx["Vaud"],
        idx[2045] :,
        idx["bld_renovation-rate"],
        idx["multi-family-households"],
    ] = renov_yr_E["multi-family-households"][-2:]

    dm_rr_fts_3.array[
        idx["Vaud"],
        :,
        idx["bld_renovation-rate"],
        idx["multi-family-households"],
    ] += ren_rate_yr_under_750_bau_multi

    prop_E_renovated_after_2045_lev_3 = (
        renov_yr_E["multi-family-households"][-1]
        / dm_rr_fts_3.array[
            idx["Vaud"],
            -1,
            idx["bld_renovation-rate"],
            idx["multi-family-households"],
        ]
    )

    # Update tthe dic
    DM_buildings["fts"]["building-renovation-rate"]["bld_renovation-rate"][3] = (
        dm_rr_fts_3
    )

    renov_copy_old.array[
        idx["Vaud"],
        idx[2045] :,
        idx["bld_renovation-rate"],
        idx["multi-family-households"],
    ] = renov_yr_E["multi-family-households"][-2:]

    DM_buildings["fts"]["building-renovation-rate"]["bld_renovation-rate"][4] = (
        renov_copy_old
    )

    # renovation redistribution is also affected

    renov_distrib_fts_3 = DM_buildings["fts"]["building-renovation-rate"][
        "bld_renovation-redistribution"
    ][2].copy()
    renov_distrib_fts_4 = DM_buildings["fts"]["building-renovation-rate"][
        "bld_renovation-redistribution"
    ][2].copy()

    prop_E_renovated_fts_3 = (
        renov_yr_E["multi-family-households"][0]
        + renov_yr_E["single-family-households"][0]
    ) / (
        dm_rr_fts_3.array[
            idx["Vaud"],
            :,
            idx["bld_renovation-rate"],
            idx["multi-family-households"],
        ]
        + dm_rr_fts_3.array[
            idx["Vaud"],
            :,
            idx["bld_renovation-rate"],
            idx["single-family-households"],
        ]
    )

    prop_E_renovated_fts_4 = (
        renov_yr_E["multi-family-households"][0]
        + renov_yr_E["single-family-households"][0]
    ) / (
        renov_copy_old.array[
            idx["Vaud"],
            :,
            idx["bld_renovation-rate"],
            idx["multi-family-households"],
        ]
        + renov_copy_old.array[
            idx["Vaud"],
            :,
            idx["bld_renovation-rate"],
            idx["single-family-households"],
        ]
    )

    renov_distrib_fts_3 = update_renovation_out(
        renov_distrib_fts_3, prop_E_renovated_fts_3
    )

    # In the 4 th scanerio all possibly reovable buildings in f and G have been renovated
    renov_distrib_fts_4 = update_renovation_out(
        renov_distrib_fts_4, prop_E_renovated_fts_4
    )

    # Renovation in
    renov_distrib_fts_3 = compute_renov_fts_mapping(renov_distrib_fts_3)
    renov_distrib_fts_4 = compute_renov_fts_mapping(renov_distrib_fts_4)
    # renov_distrib_fts_3.array[
    #     idx["Vaud"], :, idx["bld_renovation-redistribution-in"], idx["D"]
    # ] += renov_distrib_fts_3.array[
    #     idx["Vaud"], :, idx["bld_renovation-redistribution-in"], idx["E"]
    # ]
    # renov_distrib_fts_3.array[
    #     idx["Vaud"], :, idx["bld_renovation-redistribution-in"], idx["E"]
    # ] = 0

    DM_buildings["fts"]["building-renovation-rate"]["bld_renovation-redistribution"][
        3
    ] = renov_distrib_fts_3.copy()

    DM_buildings["fts"]["building-renovation-rate"]["bld_renovation-redistribution"][
        4
    ] = renov_distrib_fts_4.copy()
    # for lever in range(3, 4 + 1):
    #     DM_buildings["fts"]["building-renovation-rate"][
    #         "bld_renovation-redistribution"
    #     ][lever] = renov_distrib_fts_3.copy()

    # SECTION: Loi energy - Heating tech
    # Plus de gaz, mazout, charbon dans les prochain 15-20 ans. Pas de gaz, mazout, charbon dans les nouvelles constructions
    dm_heating_cat_fts_2 = DM_buildings["fts"]["heating-technology-fuel"][
        "bld_heating-technology"
    ][1].copy()

    dm_heating_cat_fts_2 = update_heating_fts_2(
        dm_heating_cat_fts_2,
        DM_buildings["ots"]["heating-technology-fuel"]["bld_heating-technology"],
    )
    for lever in range(lev, 4 + 1):
        DM_buildings["fts"]["heating-technology-fuel"]["bld_heating-technology"][
            lev
        ] = dm_heating_cat_fts_2.copy()

    # Compute renovation rate loi energie_refuse
    # dm_renovation = DM_buildings["fts"]["building-renovation-rate"]["bld_renovation-rate"][2].copy()

    # dm_stock_mix =  DM_buildings["fxa"]["bld_type"].copy()
    # idx_mix = dm_stock_mix.idx
    # renov_rate    =  dm_stock_mix.array[
    #         idx_mix["Vaud"],
    #         idx_mix[2023],
    #         idx_mix["bld_building-mix_stock"],
    #         :,
    #         idx_mix["F"],
    #     ]
    # renov_rate = np.ones_like(renov_rate)
    # #get the number of years to divide the rnovation rate by to apply it gradually between 2025 and 2040
    # idx_renov =dm_renovation.idx
    # yrs_renov = [yr for yr in dm_rr_fts_2.col_labels["Years"] if yr <= 2040]
    # idx_years_renov = [idx_renov[yr] for yr in yrs_renov]
    # E_renov = get_renov_rate_E(DM_buildings)
    # dm_renovation.array[
    #     idx_renov["Vaud"], idx_years_renov, idx_renov["bld_renovation-rate"], :
    # ] = (
    #     renov_rate / (yrs_renov[-1] - yrs_renov[0] + 1) )+ E_renov[0]

    # DM_buildings["fts"]["building-renovation-rate"]["bld_renovation-rate"][4] = (
    #     dm_renovation
    # )

    ##### HOTWATER TECHNOLOGY MIX ######

    dm_hotwater_fts_2 = DM_buildings["fts"]["heating-technology-fuel"][
        "bld_hot-water-technology"
    ][1].copy()
    dm_hotwater_ots = DM_buildings["ots"]["heating-technology-fuel"][
        "bld_hot-water-technology"
    ].copy()

    idx = dm_hotwater_fts_2.idx
    # Fossil heating
    # article  40.1
    idx_fossil = [idx["heating-oil"], idx["gas"]]
    idx_ots = dm_hotwater_ots.idx
    # Replace the use of fossil fuel for hot water to 0 for new buildings from 2025 and for all buildings from 2035, and replace it by the ideal scenario proportion
    # There are no buildings catergoies for hotwater technology so we replace it for all the categories at once
    dm_hotwater_fts_2.array[idx["Vaud"], 1:, idx["bld_hw_tech-mix"], idx_fossil] = (
        np.nan
    )

    dm_hotwater_fts_2.array[
        idx["Vaud"], idx[2040], idx["bld_hw_tech-mix"], idx_fossil
    ] = (
        dm_hotwater_ots.array[
            idx_ots["Vaud"],
            idx_ots[2023],
            idx["bld_hw_tech-mix"],
            [idx_ots["heating-oil"], idx_ots["gas"]],
        ]
        * 0.25
    )
    dm_hotwater_fts_2.array[
        idx["Vaud"], idx[2045], idx["bld_hw_tech-mix"], idx_fossil
    ] = (
        dm_hotwater_ots.array[
            idx_ots["Vaud"],
            idx_ots[2023],
            idx["bld_hw_tech-mix"],
            [idx_ots["heating-oil"], idx_ots["gas"]],
        ]
        * 0.05
    )
    dm_hotwater_fts_2.array[
        idx["Vaud"], idx[2050], idx["bld_hw_tech-mix"], idx_fossil
    ] = dm_hotwater_fts_2.array[
        idx["Vaud"], idx[2045], idx["bld_hw_tech-mix"], idx_fossil
    ]

    dm_hotwater_fts_2.fill_nans("Years")

    renov_prop_hotwater = create_renov_prop_hw(dm_bld_mix)
    heating_types = list(renov_prop_hotwater.keys())
    prop_vec = np.array([renov_prop_hotwater[h] for h in heating_types]).reshape(
        1, 1, -1
    )

    # Once all the old technologies are set to 0 we want to replace the missing proportion with the ideal scenario proportion
    # proportion_heating_to_replace is the missing proportion of heating that need to be replaced, we compute it by doing 1 - the sum of the proportion of heating that is not set to 0 (the one that is not affected by the energy law)
    dm_sum = dm_hotwater_fts_2.filter({"Country": ["Vaud"]}).group_all(
        "Categories1", inplace=False
    )
    arr_sum = dm_sum.array[:, :, :, np.newaxis]
    proportion_heating_to_replace = 1 - arr_sum
    idx = dm_hotwater_fts_2.idx
    heating_idx = [idx[h] for h in heating_types]
    dm_hotwater_fts_2.array[
        np.ix_(
            [idx["Vaud"]],
            np.arange(dm_hotwater_fts_2.array.shape[1]),
            np.arange(dm_hotwater_fts_2.array.shape[2]),
            heating_idx,
        )
    ] += proportion_heating_to_replace * prop_vec
    dm_hotwater_fts_2.normalise("Categories1")
    dm_hotwater_fts_2.fill_nans("Years")

    for lever in range(lev, 4 + 1):
        DM_buildings["fts"]["heating-technology-fuel"]["bld_hot-water-technology"][
            lever
        ] = dm_hotwater_fts_2.copy()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    # !FIXME: use the actual values and not the calibration factor
    file = os.path.join(this_dir, "../../../../data/datamatrix/buildings.pickle")

    my_pickle_dump(DM_buildings, file)
    sort_pickle(file)

    return DM_buildings


if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    # !FIXME: use the actual values and not the calibration factor
    file = os.path.join(this_dir, "../../../../data/datamatrix/buildings.pickle")
    with open(file, "rb") as handle:
        DM_buildings = pickle.load(handle)

    construction_period_envelope_cat_sfh = {
        "F": ["Avant 1919", "1919-1945", "1946-1960", "1961-1970"],
        "E": ["1971-1980"],
        "D": ["1981-1990", "1991-2000"],
        "C": ["2001-2005", "2006-2010"],
        "B": ["2011-2015", "2016-2020", "2021-2023"],
    }
    construction_period_envelope_cat_mfh = {
        "F": ["Avant 1919", "1919-1945", "1946-1960", "1961-1970", "1971-1980"],
        "E": ["1981-1990"],
        "D": ["1991-2000"],
        "C": ["2001-2005", "2006-2010"],
        "B": ["2011-2015", "2016-2020", "2021-2023"],
    }

    global_var = {
        "envelope construction sfh": construction_period_envelope_cat_sfh,
        "envelope construction mfh": construction_period_envelope_cat_mfh,
    }

    years_ots = create_years_list(1990, 2023, 1)
    country_list = ["Vaud"]

    dm_pop = load_pop(country_list, years_ots)

    DM_buildings = run(DM_buildings, dm_pop, global_var, country_list, lev=2)
