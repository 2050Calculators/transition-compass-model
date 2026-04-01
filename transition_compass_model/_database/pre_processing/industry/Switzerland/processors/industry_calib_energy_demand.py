import pandas as pd
import numpy as np
from transition_compass_model.model.common.data_matrix_class import DataMatrix
from transition_compass_model.model.common.auxiliary_functions import linear_fitting
import pickle
import plotly.io as pio

pio.renderers.default = "browser"
import os

from _database.pre_processing.industry.Switzerland.get_data_functions.data_energy import (
    data_energy as get_energy_data,
)
from transition_compass_model.model.common.auxiliary_functions import create_years_list


def energy_calib(current_working_directory, years_ots, years_fts):

    dm = get_energy_data(current_working_directory)

    # make shares to build missing on those, and then reconvert at the end
    # dm.normalise("Variables")
    # dm.datamatrix_plot()

    # note: for calib, we do not need to make ots and fts

    # # make ots
    # years_ots = list(range(1990,2024+1))
    # dm = linear_fitting(dm, years_ots, min_t0=0,min_tb=0)
    # # dm.datamatrix_plot()

    # # make fts
    # years_fts = list(range(2025,2050+5,5))
    # dm = linear_fitting(dm, years_fts, min_t0=0,min_tb=0)
    # # dm.datamatrix_plot()

    # # make missing tot
    # dm_tot = linear_fitting(dm_tot, years_ots, based_on=list(range(1999,2008+1,1)))
    # # dm_tot.datamatrix_plot()
    # dm_tot = linear_fitting(dm_tot, years_fts, based_on=list(range(2013,2024+1,1)))
    # # dm_tot.datamatrix_plot()

    # # make missing absolute values per carriers
    # dm.array = dm.array = dm.array * dm_tot.array
    # for v in dm.col_labels["Variables"]: dm.units[v] = "MJ"
    # # dm.datamatrix_plot()

    # add missing
    missing = ["gas-bio", "hydrogen", "liquid-bio"]
    for m in missing:
        dm.add(0, "Variables", m, unit="TJ", dummy=True)
    dm.sort("Variables")

    # format and save
    dm.drop("Years", 2024)
    for v in dm.col_labels["Variables"]:
        dm.rename_col(v, "calib-energy-demand-excl-feedstock_" + v, "Variables")
    dm.deepen()
    factor = 1 / 3600
    dm.change_unit(
        "calib-energy-demand-excl-feedstock", factor, "TJ", "TWh", operator="*"
    )
    dm.add(np.nan, "Years", list(range(1990, 1998 + 1, 1)), dummy=True)
    dm.add(np.nan, "Years", years_fts, dummy=True)
    dm.sort("Years")
    dm.sort("Categories1")
    dm.rename_col(
        "calib-energy-demand-excl-feedstock", "calib-energy-demand", "Variables"
    )
    # f = os.path.join(current_file_directory, '../data/datamatrix/calibration_energy-demand.pickle')
    # with open(f, 'wb') as handle:
    #     pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm


def run(years_ots, years_fts):

    # directories
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    dm = energy_calib(current_file_directory, years_ots, years_fts)

    return dm


if __name__ == "__main__":

    years_ots = create_years_list(1990, 2023, 1)
    years_fts = create_years_list(2025, 2050, 5)

    run(years_ots, years_fts)
