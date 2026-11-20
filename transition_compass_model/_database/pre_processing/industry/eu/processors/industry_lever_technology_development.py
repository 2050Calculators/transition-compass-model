# packages
import os
import pickle
import warnings

import numpy as np

warnings.simplefilter("ignore")

from transition_compass_model.model.common.auxiliary_functions import create_years_list
from transition_compass_model.model.common.data_matrix_class import DataMatrix

# NOTE: for the business as usual, we will put no improvement on the tech development.
# TODO: use documentation EUCalc to do level 1 etc.


def make_technology_development_dm(current_file_directory, lever_file, years_ots):
    # create dm
    countries = [
        "Austria",
        "Belgium",
        "Bulgaria",
        "Croatia",
        "Cyprus",
        "Czech Republic",
        "Denmark",
        "EU27",
        "Estonia",
        "Finland",
        "France",
        "Germany",
        "Greece",
        "Hungary",
        "Ireland",
        "Italy",
        "Latvia",
        "Lithuania",
        "Luxembourg",
        "Malta",
        "Netherlands",
        "Poland",
        "Portugal",
        "Romania",
        "Slovakia",
        "Slovenia",
        "Spain",
        "Sweden",
        "United Kingdom",
    ]

    variabs = [
        "aluminium-prim",
        "aluminium-sec",
        "ammonia-tech",
        "asphalt-sec",
        "asphalt-tech",
        "cement-dry-kiln",
        "cement-geopolym",
        "cement-sec-post-consumer",
        "cement-wet-kiln",
        "chem-chem-tech",
        "chem-sec",
        "copper-sec",
        "copper-tech",
        "fbt-tech",
        "glass-glass",
        "glass-sec",
        "lime-lime",
        "mae-tech",
        "ois-sec",
        "ois-tech",
        "paper-tech",
        "paper-sec",
        "steel-BF-BOF",
        "steel-hisarna",
        "steel-hydrog-DRI",
        "steel-scrap-EAF",
        "textiles-tech",
        "timber-sec",
        "timber-tech",
        "tra-equip-tech",
        "wwp-tech",
    ]
    variabs = ["technology-development_" + i for i in variabs]
    units = list(np.repeat("%", len(variabs)))
    units_dict = dict()
    for i in range(0, len(variabs)):
        units_dict[variabs[i]] = units[i]
    index_dict = dict()
    for i in range(0, len(countries)):
        index_dict[countries[i]] = i
    for i in range(0, len(years_ots)):
        index_dict[years_ots[i]] = i
    for i in range(0, len(variabs)):
        index_dict[variabs[i]] = i
    dm = DataMatrix(empty=True)
    dm.col_labels = {"Country": countries, "Years": years_ots, "Variables": variabs}
    dm.units = units_dict
    dm.idx = index_dict
    dm.array = np.zeros((len(countries), len(years_ots), len(variabs)))

    # # make nan for other than EU27, as for EU for the moment we keep BAU which is 0
    # countries_oth = np.array(countries)[[i not in "EU27" for i in countries]].tolist()
    # idx = dm.idx
    # years = list(range(2025, 2050 + 1, 5))
    # for c in countries_oth:
    #     for y in years:
    #         for v in variabs:
    #             dm.array[idx[c], idx[y], idx[v]] = np.nan
    # df = dm.write_df()

    # deepen
    dm.deepen()

    # # split between ots and fts
    # years_ots = list(range(1990, 2023 + 1))
    # years_fts = list(range(2025, 2055, 5))
    # dm_ots = dm.filter({"Years": years_ots})
    # dm_fts = dm.filter({"Years": years_fts})
    # DM_fts = {
    #     1: dm_fts.copy(),
    #     2: dm_fts.copy(),
    #     3: dm_fts.copy(),
    #     4: dm_fts.copy(),
    # }  # for now we set all levels to be the same
    # DM = {"ots": dm_ots, "fts": DM_fts}

    # save
    f = os.path.join(current_file_directory, "../data/datamatrix/", lever_file)
    with open(f, "wb") as handle:
        pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # df = dm.write_df()
    # df = df.loc[df["Country"] == "EU27",:]
    # df = df.loc[df["Years"].isin([2022,2023]),:]
    # df_temp = pd.melt(df, id_vars = ['Country','Years'], var_name='variable')
    # name = "temp.xlsx"
    # df_temp.to_excel("~/Desktop/" + name)

    return dm


def run(years_ots):
    # directories
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # if exists, load, else make
    lever_file = "lever_technology-development.pickle"
    filepath = os.path.join(current_file_directory, "../data/datamatrix/" + lever_file)
    if os.path.exists(filepath):
        with open(filepath, "rb") as handle:
            dm = pickle.load(handle)
    else:
        dm = make_technology_development_dm(
            current_file_directory, lever_file, years_ots
        )

    return dm


if __name__ == "__main__":
    years_ots = create_years_list(1990, 2023, 1)
    run(years_ots)
