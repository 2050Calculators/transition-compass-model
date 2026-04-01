import numpy as np
import os

from _database.pre_processing.industry.Switzerland.get_data_functions.data_product_net_import import (
    get_packaging_data,
)
from transition_compass_model.model.common.auxiliary_functions import linear_fitting
from transition_compass_model.model.common.auxiliary_functions import create_years_list, load_pop


def make_dm_packaging(current_file_directory, dm_pop):

    dm = get_packaging_data(current_file_directory)

    # # load DM_pack
    # filepath = os.path.join(current_file_directory, '../../../../data/datamatrix/lifestyles.pickle')
    # with open(filepath, 'rb') as handle:
    #     DM_lfs = pickle.load(handle)
    # dm_pop = DM_lfs["ots"]["pop"]["lfs_population_"].copy()
    # dm_pop = dm_pop.filter({"Country" : ["Switzerland"]})

    # make per capita
    for v in dm.col_labels["Variables"]:
        dm.rename_col(v, "product-demand_" + v, "Variables")
    dm.deepen()
    dm.array = dm.array / dm_pop.array[..., np.newaxis]
    dm.units["product-demand"] = "t/cap"
    # df_temp = dm.write_df()
    # df_temp["product-demand_plastic-pack[t/cap]"]*1000
    # mayble only plastic packaging is a bit low at 44kg, as it should be around 60kg per capita (it's 120 kg all plastic, and half
    # should be packaging), but ok

    # do fts
    # for y in years_fts:
    #     dm.add(dm[:,2023,...], "Years", [y])
    # dm = linear_fitting(dm, years_fts, based_on=list(range(2010,2019+1)))
    # def linear_fitting_per_variab(dm, variable, year_start, year_end, years_fts):
    #     dm_temp = dm.filter({"Categories1" : [variable]})
    #     dm_temp = linear_fitting(dm_temp, years_fts, based_on=list(range(year_start,year_end+1)))
    #     dm.drop("Categories1",variable)
    #     dm.append(dm_temp,"Categories1")

    # dm.add(np.nan,"Years",years_fts,dummy=True)
    # linear_fitting_per_variab(dm, 'aluminium-pack', 2000, 2019, years_fts)
    # linear_fitting_per_variab(dm, 'glass-pack', 2000, 2019, years_fts)
    # linear_fitting_per_variab(dm, 'plastic-pack', 2003, 2019, years_fts)
    # linear_fitting_per_variab(dm, 'paper-pack', 2011, 2019, years_fts)
    # linear_fitting_per_variab(dm, 'paper-print', 2012, 2023, years_fts)
    # linear_fitting_per_variab(dm, 'paper-san', 2020, 2023, years_fts)

    # dm.flatten().datamatrix_plot()

    # # save
    # years_ots = list(range(1990,2023+1))
    # years_fts = list(range(2025,2055,5))
    # dm_ots = dm.filter({"Years" : years_ots})
    # dm_fts = dm.filter({"Years" : years_fts})
    # DM_fts = {1: dm_fts.copy(), 2: dm_fts.copy(), 3: dm_fts.copy(), 4: dm_fts.copy()} # for now we set all levels to be the same
    # DM = {"ots" : dm_ots,
    #       "fts" : DM_fts}
    # f = os.path.join(current_file_directory, '../data/datamatrix/lever_paperpack.pickle')
    # with open(f, 'wb') as handle:
    #     pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm


def run(dm_pop_ots):

    # directory
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # get packaging per capita
    dm_pack = make_dm_packaging(current_file_directory, dm_pop_ots)

    return dm_pack


if __name__ == "__main__":

    country_list = ["Switzerland"]
    years_ots = create_years_list(1990, 2023, 1)
    years_fts = create_years_list(2025, 2050, 5)

    dm_pop_ots = load_pop(country_list, years_list=years_ots)

    run(dm_pop_ots)
