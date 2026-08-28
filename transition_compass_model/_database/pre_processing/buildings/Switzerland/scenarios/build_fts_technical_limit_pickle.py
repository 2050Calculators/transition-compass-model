import os
import pickle

import numpy as np
from scenarios.build_fts_PCV_LVLEne_pickle import update_heating_change_proportion

from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    my_pickle_dump,
    sort_pickle,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def create_renov_prop_hw2(dm_bld_mix: DataMatrix) -> dict:
    """Compute the renovation share of each hot-water technology.

    The function combines the renovation technology shares from the multi-family
    and single-family scenarios using the 2023 building-stock split between these
    two household categories. This yields the technology proportions to assign to
    the hot-water renovation portfolio.

    Args:
        dm_bld_mix (DataMatrix): Building mix matrix for the relevant countries,
            grouped by ``Categories2`` and filtered to the
            ``multi-family-households`` and ``single-family-households`` segments.

    Returns:
        dict: Mapping of technology names to their renovation share in the final
        portfolio, with values in the range ``[0, 1]``.
    """
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
    # Group by enveloppe categories
    dm_bld_mix.group_all("Categories2")

    # TODO : do the services
    dm_bld_mix.filter(
        {"Categories1": ["multi-family-households", "single-family-households"]}
    )
    dm_bld_mix.normalise("Categories1")
    idx_prop_multi_single = dm_bld_mix.idx
    array_prop_multi_single = dm_bld_mix.array[
        idx_prop_multi_single["Vaud"],
        idx_prop_multi_single[2023],
        0,
        :,
    ]

    renov_prop_hw = {}
    for cat in renov_proportion_single.keys():
        renov_prop_hw[cat] = (
            renov_proportion_multi[cat] * array_prop_multi_single[0]
            + renov_proportion_single[cat] * array_prop_multi_single[1]
        )

    return renov_prop_hw


def hotwater_sublever(DM_buildings: DataMatrix, lev: int) -> DataMatrix:
    dm_hotwater_fts = DM_buildings["fts"]["heating-technology-fuel"][
        "bld_hot-water-technology"
    ][2].copy()

    idx = dm_hotwater_fts.idx
    idx_fossil = [idx["heating-oil"], idx["gas"]]

    # Set fossil technologies to 0 in 2045 to later replace them.
    dm_hotwater_fts.array[idx["Vaud"], 1:, idx["bld_hw_tech-mix"], idx_fossil] = np.nan
    dm_hotwater_fts.array[
        idx["Vaud"], idx[2045] :, idx["bld_hw_tech-mix"], idx_fossil
    ] = 0

    country_list = dm_hotwater_fts.col_labels["Country"]
    dm_bld_mix = (
        DM_buildings["ots"]["building-renovation-rate"]["bld_building-mix"]
        .filter({"Country": country_list})
        .copy()
    )

    # Matrix with the theoretical proportion replace
    renov_prop_hotwater = create_renov_prop_hw2(dm_bld_mix)
    heating_types = list(renov_prop_hotwater.keys())
    prop_vec = np.array([renov_prop_hotwater[h] for h in heating_types]).reshape(
        1, 1, -1
    )

    # Once all the old technologies are set to 0 we want to replace the missing proportion with the ideal scenario proportion
    # proportion_heating_to_replace is the missing proportion of heating that need to be replaced, we compute it by doing 1 - the sum of the proportion of heating that is not set to 0 (the one that is not affected by the energy law)
    dm_sum = dm_hotwater_fts.group_all("Categories1", inplace=False)
    arr_sum = dm_sum.array[:, :, :, np.newaxis]
    proportion_heating_to_replace = 1 - arr_sum
    idx = dm_hotwater_fts.idx
    heating_idx = [idx[h] for h in heating_types]
    dm_hotwater_fts.array[
        np.ix_(
            [idx[c] for c in country_list],
            np.arange(dm_hotwater_fts.array.shape[1]),
            np.arange(dm_hotwater_fts.array.shape[2]),
            heating_idx,
        )
    ] += proportion_heating_to_replace * prop_vec
    dm_hotwater_fts.normalise("Categories1")
    dm_hotwater_fts.fill_nans("Years")

    DM_buildings["fts"]["heating-technology-fuel"]["bld_hot-water-technology"][lev] = (
        dm_hotwater_fts.copy()
    )

    return DM_buildings


def heating_tech(DM_buildings, lev):
    dm_heating_cat_fts_4 = DM_buildings["fts"]["heating-technology-fuel"][
        "bld_heating-technology"
    ][2].copy()

    idx = dm_heating_cat_fts_4.idx
    idx_fossil = [idx["coal"], idx["heating-oil"], idx["gas"]]

    dm_heating_cat_fts_4.array[
        idx["Vaud"],
        1 : idx[2045],
        idx["bld_heating-mix"],
        :,
        :,
        idx_fossil,
    ] = np.nan
    dm_heating_cat_fts_4.array[
        idx["Vaud"],
        idx[2045] :,
        idx["bld_heating-mix"],
        :,
        :,
        idx_fossil,
    ] = 0

    dm_heating_cat_fts_4.fill_nans("Years")

    dm_heating_fts_mfh = update_heating_change_proportion(
        dm_heating_cat_fts_4, "multi-family-households"
    )
    dm_heating_cat_fts_4["Vaud", :, :, "multi-family-households", :, :] = (
        dm_heating_fts_mfh["Vaud", :, :, "multi-family-households", :, :]
    )

    dm_heating_fts_sfh = update_heating_change_proportion(
        dm_heating_cat_fts_4, "single-family-households"
    )
    dm_heating_cat_fts_4["Vaud", :, :, "single-family-households", :, :] = (
        dm_heating_fts_sfh["Vaud", :, :, "single-family-households", :, :]
    )

    dm_heating_cat_fts_4.normalise("Categories3")
    dm_heating_cat_fts_4.fill_nans("Years")

    DM_buildings["fts"]["heating-technology-fuel"]["bld_heating-technology"][lev] = (
        dm_heating_cat_fts_4
    )

    ###HOTWATER section
    DM_buildings = hotwater_sublever(DM_buildings, lev)

    return DM_buildings


def run(DM_buildings, lev=4):
    ######  HEATING TECHNOLOGY #####
    DM_buildings = heating_tech(DM_buildings, lev)

    this_dir = os.path.dirname(os.path.abspath(__file__))
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

    years_ots = create_years_list(1990, 2023, 1)
    years_fts = create_years_list(2025, 2050, 5)

    run(DM_buildings, years_ots, years_fts)
