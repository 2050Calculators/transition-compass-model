import pandas as pd
import numpy as np
import os
from transition_compass_model.model.common.data_matrix_class import DataMatrix
from transition_compass_model.model.common.auxiliary_functions import create_years_list
from _database.pre_processing.industry.Switzerland.get_data_functions.data_emissions import (
    get_emissions_data,
)


def emissions_calib(current_file_directory, years_ots, years_fts):

    # get data
    df = pd.concat(
        [
            get_emissions_data(current_file_directory, ghg)
            for ghg in ["CO2", "CH4", "N2O"]
        ]
    )
    # note: for now calibration is done on Mt and not MtCO2eq

    # make dm
    df["ghg_new"] = np.nan
    df.loc[df["ghg"] == "CO2", "ghg_new"] = (
        df.loc[df["ghg"] == "CO2", "emissions"]
        + "_"
        + df.loc[df["ghg"] == "CO2", "ghg"]
        + "[Mt]"
    )
    for gas in ["CH4", "N2O"]:
        df.loc[df["ghg"] == gas, "ghg_new"] = (
            df.loc[df["ghg"] == gas, "emissions"]
            + "_"
            + df.loc[df["ghg"] == gas, "ghg"]
            + "[kt]"
        )
    df = df.loc[:, ["Years", "value", "ghg_new"]]
    df = df.pivot(index=["Years"], columns="ghg_new", values="value").reset_index()
    df["Country"] = "Switzerland"
    dm = DataMatrix.create_from_df(df, 0)
    # dm.datamatrix_plot()

    # kt to Mt for CH4 and N2O
    dm.change_unit("combustion-emissions_CH4", 1e-3, "kt", "Mt")
    dm.change_unit("combustion-emissions_N2O", 1e-3, "kt", "Mt")
    dm.change_unit("process-emissions_CH4", 1e-3, "kt", "Mt")
    dm.change_unit("process-emissions_N2O", 1e-3, "kt", "Mt")
    dm.deepen()

    # for now put together (keep them separate later in case)
    dm.groupby(
        {"calib-emissions": ["combustion-emissions", "process-emissions"]},
        "Variables",
        inplace=True,
    )

    # add missing years as nan
    years = years_ots + years_fts
    missing = np.array(years)[[y not in dm.col_labels["Years"] for y in years]].tolist()
    dm.add(np.nan, "Years", missing, dummy=True)
    dm.sort("Years")

    # # format and save
    # for v in dm.col_labels["Variables"]: dm.rename_col(v, 'calib-emissions_' + v, "Variables")
    # dm.deepen()

    return dm


def run(years_ots, years_fts):

    # directories
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    dm = emissions_calib(current_file_directory, years_ots, years_fts)

    return dm


if __name__ == "__main__":

    years_ots = create_years_list(1990, 2023, 1)
    years_fts = create_years_list(2025, 2050, 5)

    run(years_ots, years_fts)


# import os
# current_file_directory = os.getcwd()
# f = os.path.join(current_file_directory, '../data/datamatrix/calibration_emissions.pickle')
# import pickle
# with open(f, 'wb') as handle:
#     pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)
