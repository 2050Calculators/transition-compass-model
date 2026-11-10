import os

import numpy as np

from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    filter_country_and_load_data_from_pickles,
    init_years_lever,
    linear_fit_ratio,
    linear_fitting,
    mean_fts_level,
    my_pickle_dump,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

# Years whose mean is held to 2050 where a linear trend runs away
YEARS_MEAN_FIRST = 2014
# SSR capped at 1 in the BAU (no exports recorded)
SSR_BAU_CAP_1 = ["crop-sugarcrop"]


def ssr_bau_fts(dm, years_ots, years_fts, DM_fxa, liv_ssr_on_food_supply):
    """BAU fts of the self-sufficiency ratios (lever food-net-import).

    - default: linear trend of the ots (as before), at least 0
    - crops and processed crops (crop-*, pro-crop-*): the 2014-2023 mean. A
      linear trend over 1990-2023 runs through the FBSH/FBS break of 2010 and
      the 2018-2023 beet stock build: Swiss cereal production went to 0 in
      2050, sugar beet above its 2023 level. Sugar beet capped at 1.
    - livestock products, when liv_ssr_on_food_supply (cantons): the SSR is
      production / (food supply x supply-food ratio). The trend is fitted on
      production / food supply (no FBSH/FBS break) and divided by the held
      ratio of fxa ratio_supply-food.
    """
    years_mean = [y for y in years_ots if y >= YEARS_MEAN_FIRST]
    dm_fts = dm.copy()
    linear_fitting(dm_fts, years_fts, based_on=years_ots)
    dm_fts = dm_fts.filter({"Years": years_fts})
    dm_fts.array = np.where(dm_fts.array < 0, 0, dm_fts.array)

    crop_cats = [
        c
        for c in dm.col_labels["Categories1"]
        if c.startswith("crop-") or c.startswith("pro-crop-")
    ]
    dm_mean = mean_fts_level(
        dm.filter({"Categories1": crop_cats}), years_mean, years_fts
    )
    for c in crop_cats:
        dm_fts.array[..., dm_fts.idx[c]] = dm_mean.array[..., dm_mean.idx[c]]
        if c in SSR_BAU_CAP_1:
            dm_fts.array[..., dm_fts.idx[c]] = np.minimum(
                dm_fts.array[..., dm_fts.idx[c]], 1.0
            )

    if liv_ssr_on_food_supply:
        dm_ratio = DM_fxa["ratio_supply-food"].filter(
            {"Country": dm.col_labels["Country"]}
        )
        for c in dm_ratio.col_labels["Categories1"]:
            dm_c = dm.filter({"Categories1": [c]})
            ratio_ots = dm_ratio.filter({"Years": years_ots, "Categories1": [c]})
            dm_c.array = dm_c.array * ratio_ots.array
            linear_fitting(dm_c, years_fts, based_on=years_ots)
            dm_c = dm_c.filter({"Years": years_fts})
            ratio_fts = dm_ratio.filter({"Years": years_fts, "Categories1": [c]})
            dm_fts.array[..., dm_fts.idx[c]] = np.maximum(
                dm_c.array[..., 0] / ratio_fts.array[..., 0], 0
            )
    return dm_fts


def run(
    DM_agriculture: DataMatrix,
    years_ots: list,
    years_fts: list,
    liv_ssr_on_food_supply: bool = False,
) -> DataMatrix:
    """Create fts BAU for agriculture

    Args:
        DM_agriculture (DataMatrix)
        years_ots (list)
        years_fts (list)
        liv_ssr_on_food_supply (bool): True for the cantons, whose livestock
            SSR is production / (food supply x supply-food ratio)

    Returns:
        DataMatrix: DM_agriculture updated with fts BAU for agriculture
    """
    # Levers to be normalised
    list_norm = [
        "climate-smart-livestock_ration",
        "share",
        "biomass-hierarchy_biomass-mix_digestor",
    ]  #'biomass-hierarchy_bioenergy_liquid_biodiesel'
    #'biomass-hierarchy_bioenergy_liquid_biogasoline'#"food-net-import"
    # Levers kept constant at the last ots value
    list_constant = ["climate-smart-livestock_slaughtered"]

    for key in DM_agriculture["ots"].keys():
        if isinstance(DM_agriculture["ots"][key], dict):
            for subkey in DM_agriculture["ots"][key].keys():
                dm = DM_agriculture["ots"][key][subkey].copy()

                # 1 to 4
                if subkey in list_norm:  # lever to stay in 0-1 range
                    linear_fit_ratio(
                        dm, years_fts, years_range=[years_ots[0], years_ots[-1]]
                    )

                elif subkey == "climate-smart-crop_losses":
                    # 2014-2023 mean losses factor, at least 1 (a linear trend
                    # went below 1, negative losses, for Swiss starchy roots)
                    dm = mean_fts_level(
                        dm,
                        [y for y in years_ots if y >= YEARS_MEAN_FIRST],
                        years_fts,
                        min_value=1.0,
                    )

                elif subkey in list_constant:
                    linear_fitting(dm, years_fts, based_on=years_ots)
                    idx = dm.idx
                    dm.array[:, [idx[y] for y in years_fts], ...] = dm.array[
                        :, idx[years_ots[-1]], np.newaxis, ...
                    ]

                else:
                    linear_fitting(dm, years_fts, based_on=years_ots)
                    dm.array = np.where(dm.array < 0, 0, dm.array)
                for lev in range(1, 5):
                    DM_agriculture["fts"][key][subkey][lev] = dm.filter(
                        {"Years": years_fts}
                    )

        else:
            dm = DM_agriculture["ots"][key].copy()
            if key == "food-net-import":
                dm = ssr_bau_fts(
                    dm,
                    years_ots,
                    years_fts,
                    DM_agriculture["fxa"],
                    liv_ssr_on_food_supply,
                )
            else:
                linear_fitting(dm, years_fts, based_on=years_ots)
            if key == "food-net-import" and "Vaud" in dm.col_labels["Country"]:
                idx = dm.idx
                # Ensure cereal production stays above 1 for Vaud
                dm.array[idx["Vaud"], :, :, idx["crop-cereal"]] = np.where(
                    dm.array[idx["Vaud"], :, :, idx["crop-cereal"]] < 1,
                    1,
                    dm.array[idx["Vaud"], :, :, idx["crop-cereal"]],
                )

            dm.array = np.where(dm.array < 0, 0, dm.array)
            for lev in range(1, 5):
                DM_agriculture["fts"][key][lev] = dm.filter({"Years": years_fts})

    ##### SAVE FILE #########
    this_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(this_dir, "../../../data/datamatrix/agriculture.pickle")
    my_pickle_dump(DM_new=DM_agriculture, local_pickle_file=pickle_file)

    return DM_agriculture


if __name__ == "__main__":
    country_list = ["Vaud"]
    DM_input = filter_country_and_load_data_from_pickles(
        country_list=country_list, modules_list="agriculture"
    )
    years_setting, lever_setting = init_years_lever()
    years_ots = create_years_list(years_setting[0], years_setting[1], step=1)
    years_fts = create_years_list(years_setting[2], years_setting[3], years_setting[4])

    # Load the transport data matrix
    this_dir = os.path.dirname(os.path.abspath(__file__))
    # pickle_file = os.path.join(this_dir, "../../../data/datamatrix/agriculture.pickle")
    # with open(pickle_file, "rb") as f:
    #     DM_agriculture = pickle.load(f)

    # Run the function to increase freight rail by 45%
    DM_agriculture_updated = run(DM_input["agriculture"], years_ots, years_fts)
