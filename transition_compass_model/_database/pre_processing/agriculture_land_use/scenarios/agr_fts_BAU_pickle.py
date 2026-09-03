import os

from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    filter_country_and_load_data_from_pickles,
    init_years_lever,
    linear_fit_ratio,
    linear_fitting,
    my_pickle_dump,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def run(DM_agriculture: DataMatrix, years_ots: list, years_fts: list) -> DataMatrix:
    """Create fts BAU for agriculture

    Args:
        DM_agriculture (DataMatrix)
        years_ots (list)
        years_fts (list)

    Returns:
        DataMatrix: DM_agriculture updated with fts BAU for agriculture
    """
    # Levers to be normalised
    list_norm = ["climate-smart-livestock_ration"]

    for key in DM_agriculture["ots"].keys():
        if isinstance(DM_agriculture["ots"][key], dict):
            for subkey in DM_agriculture["ots"][key].keys():
                dm = DM_agriculture["ots"][key][subkey].copy()

                # 1 to 4
                if subkey in list_norm:  # lever to stay in 0-1 range
                    linear_fit_ratio(
                        dm, years_fts, years_range=[years_ots[0], years_ots[-1]]
                    )

                else:
                    linear_fitting(dm, years_fts, based_on=years_ots)
                for lev in range(1, 5):
                    DM_agriculture["fts"][key][subkey][lev] = dm.filter(
                        {"Years": years_fts}
                    )

        else:
            dm = DM_agriculture["ots"][key].copy()
            linear_fitting(dm, years_fts, based_on=years_ots)
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
