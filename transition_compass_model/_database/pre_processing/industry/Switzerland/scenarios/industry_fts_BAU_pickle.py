import os
import pickle
from transition_compass_model.model.common.auxiliary_functions import (
    linear_fitting,
    create_years_list,
    my_pickle_dump,
    sort_pickle,
)

import plotly.io as pio

pio.renderers.default = "browser"


def make_fts(DM_industry, name, years_fts, based_on):

    dm = DM_industry["ots"][name].copy()
    dm = linear_fitting(dm, years_fts, based_on=based_on)
    # dm.datamatrix_plot()
    DM_industry["fts"][name] = {}
    for level in list(range(1, 4 + 1)):
        DM_industry["fts"][name][level] = dm.filter({"Years": years_fts})

    return


def run(DM_industry, DM_ammonia, country_list, years_ots, years_fts):

    # directory
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    DM_industry["fts"] = {}

    # NOTE: for simplicity I keep these ratios constants for future years, to
    # see later if we want to change this.

    # product net import
    make_fts(DM_industry, "product-net-import", years_fts, based_on=[2023])

    # material net import
    make_fts(DM_industry, "material-net-import", years_fts, based_on=[2023])

    # paperpack
    make_fts(DM_industry, "paperpack", years_fts, based_on=[2023])

    # waste management
    make_fts(DM_industry, "eol-waste-management", years_fts, based_on=[2023])

    # Load existing DM_industry
    pickle_file = os.path.join(
        current_file_directory, "../../../../data/datamatrix/industry.pickle"
    )
    with open(pickle_file, "rb") as handle:
        DM_industry_current = pickle.load(handle)

    # make other levers same from EU
    other_levers = [
        "material-switch",
        "material-efficiency",
        "technology-development",
        "cc",
        "technology-share",
        "energy-carrier-mix",
        "eol-material-recovery",
    ]
    for l in other_levers:
        DM_industry["fts"][l] = {}
        for level in list(range(1, 4 + 1)):
            dm_temp = DM_industry_current["fts"][l][1].filter({"Country": ["EU27"]})
            dm_temp.rename_col("EU27", "Switzerland", "Country")
            DM_industry["fts"][l][level] = dm_temp.copy()

    # save
    my_pickle_dump(DM_new=DM_industry, local_pickle_file=pickle_file)
    sort_pickle(pickle_file)

    # ammonia

    DM_ammonia["fts"] = {}

    # product net import
    make_fts(DM_ammonia, "product-net-import", years_fts, based_on=[2023])

    # material net import
    make_fts(DM_ammonia, "material-net-import", years_fts, based_on=[2023])

    # Load existing DM_industry
    pickle_file = os.path.join(
        current_file_directory, "../../../../data/datamatrix/ammonia.pickle"
    )
    with open(pickle_file, "rb") as handle:
        DM_ammonia_current = pickle.load(handle)

    # make other levers same from EU
    other_levers = [
        "material-efficiency",
        "eol-material-recovery",
        "technology-development",
        "cc",
        "energy-carrier-mix",
    ]
    for l in other_levers:
        DM_ammonia["fts"][l] = {}
        for level in list(range(1, 4 + 1)):
            dm_temp = DM_ammonia_current["fts"][l][1].filter({"Country": ["EU27"]})
            dm_temp.rename_col("EU27", "Switzerland", "Country")
            DM_ammonia["fts"][l][level] = dm_temp.copy()

    # save
    my_pickle_dump(DM_new=DM_ammonia, local_pickle_file=pickle_file)
    sort_pickle(pickle_file)

    return DM_industry, DM_ammonia


if __name__ == "__main__":

    # get country ots fts
    years_ots = create_years_list(1990, 2023, 1)
    years_fts = create_years_list(2025, 2050, 5)
    country_list = ["Switzerland"]

    # load pickle industry ots
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(
        current_file_directory, "../data/datamatrix/industry_ots.pickle"
    )
    if not os.path.exists(filepath):
        raise FileNotFoundError("You need to run ots_pickle_run() first")
    with open(filepath, "rb") as f:
        DM_industry = pickle.load(f)
    filepath = os.path.join(
        current_file_directory, "../data/datamatrix/ammonia_ots.pickle"
    )
    if not os.path.exists(filepath):
        raise FileNotFoundError("You need to run ots_pickle_run() first")
    with open(filepath, "rb") as f:
        DM_ammonia = pickle.load(f)

    run(DM_industry, DM_ammonia, country_list, years_ots, years_fts)
