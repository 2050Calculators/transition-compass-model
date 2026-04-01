import os
import numpy as np
import pandas as pd

from transition_compass_model.model.common.auxiliary_functions import create_years_list
from transition_compass_model.model.common.auxiliary_functions import linear_fitting
from _database.pre_processing.industry.Switzerland.get_data_functions.data_ammonia_product_net_import import (
    get_ammonia_trade_data,
)


def linear_fitting_per_variab(dm, variable, year_start, year_end, years):
    dm_temp = dm.filter({"Variables": [variable]})
    dm_temp = linear_fitting(
        dm_temp, years, based_on=list(range(year_start, year_end + 1))
    )
    dm.drop("Variables", variable)
    dm.append(dm_temp, "Variables")


def make_ammonia_dms(dm, years_ots, years_fts):

    # make ots
    dm.add(np.nan, "Years", list(range(1990, 2001 + 1)), dummy=True)
    dm = dm.flatten()
    linear_fitting_per_variab(dm, "export_ammonia", 2002, 2002, years_ots)
    linear_fitting_per_variab(dm, "import_ammonia", 2015, 2023, years_ots)
    linear_fitting_per_variab(dm, "export_fertilizer", 2005, 2014, years_ots)
    linear_fitting_per_variab(dm, "import_fertilizer", 2002, 2023, years_ots)
    # dm.datamatrix_plot()

    # make fts
    dm.add(np.nan, "Years", years_fts, dummy=True)
    linear_fitting_per_variab(dm, "export_ammonia", 2023, 2023, years_fts)
    linear_fitting_per_variab(dm, "import_ammonia", 2023, 2023, years_fts)
    dm.array[dm[...] < 0] = 0
    linear_fitting_per_variab(dm, "export_fertilizer", 2005, 2014, years_fts)
    linear_fitting_per_variab(dm, "import_fertilizer", 2002, 2023, years_fts)
    # dm.datamatrix_plot()
    dm.deepen()

    # add production
    # for production between 2015-2019 (page 45.3)
    # https://pubs.usgs.gov/myb/vol3/2019/myb3-2019-switzerland.pdf
    dm.add(np.nan, "Variables", "production", "t", True)
    dm[:, :, "production", "fertilizer"] = (
        0  # will assume that fertilizer production is zero as export is very limited
    )
    data_n = {
        "Year": [2015, 2016, 2017, 2018, 2019],
        "Production_N_kt": [34, 34, 34, 14, 14],  # kt of nitrogen content
    }
    df = pd.DataFrame(data_n)
    factor = 17 / 14  # molecular weight ratio
    df["Production_NH3_kt"] = df["Production_N_kt"] * factor
    for y in [2015, 2016, 2017, 2018, 2019]:
        dm[:, y, "production", "ammonia"] = df.loc[df["Year"] == y, "Production_NH3_kt"]
    for y in list(range(1990, 2014 + 1)):
        dm[:, y, "production", "ammonia"] = dm[:, 2015, "production", "ammonia"]
    for y in list(range(2020, 2023 + 1)):
        dm[:, y, "production", "ammonia"] = dm[:, 2019, "production", "ammonia"]
    for y in years_fts:
        dm[:, y, "production", "ammonia"] = dm[:, 2019, "production", "ammonia"]
    # dm.flatten().datamatrix_plot()
    dm[..., "production", "ammonia"] = (
        dm[..., "production", "ammonia"] * 1000
    )  # get the right unit in tonnes

    # make demand
    arr_temp = dm[:, :, "production", :] + dm[:, :, "import", :] - dm[:, :, "export", :]
    dm.add(arr_temp, "Variables", "demand", "t")
    dm.array[dm[...] < 0] = 1000  # we keep ammonia demand to 1000 (level of 2018)
    # dm.flatten().datamatrix_plot()

    # make product net import share
    dm_temp = dm.filter({"Variables": ["export", "import", "demand"]})
    arr_temp = (dm_temp[:, :, "import", :] - dm_temp[:, :, "export", :]) / dm_temp[
        :, :, "demand", :
    ]
    dm_temp.add(arr_temp, "Variables", "net-import", "%")
    dm_amm_prod_net_import = dm_temp.filter(
        {"Variables": ["net-import"], "Categories1": ["fertilizer"]}
    )
    dm_amm_prod_net_import.rename_col("net-import", "product-net-import", "Variables")
    dm_amm_prod_net_import = dm_amm_prod_net_import.filter({"Years": years_ots})
    # dm_amm_prod_net_import.flatten().datamatrix_plot()
    dm_amm_prod_net_import[...] = (
        0.95  # making this correction to allow some local production of fertilizer (otherwise there is no
    )
    # ammonia production)

    # make material net import share
    dm_amm_mat_net_import = dm_temp.filter(
        {"Variables": ["net-import"], "Categories1": ["ammonia"]}
    )
    dm_amm_mat_net_import.rename_col("net-import", "material-net-import", "Variables")
    dm_amm_mat_net_import = dm_amm_mat_net_import.filter({"Years": years_ots})
    # dm_amm_mat_net_import.flatten().datamatrix_plot()

    # make material production
    dm_amm_prod = dm.filter({"Variables": ["production"], "Categories1": ["ammonia"]})
    dm_amm_prod.rename_col("production", "material-production", "Variables")
    dm_amm_prod.change_unit("material-production", 1e-3, "t", "kt")
    # dm_amm_prod.flatten().datamatrix_plot()

    return dm_amm_prod_net_import, dm_amm_mat_net_import, dm_amm_prod


# calib data for ammonia emissions

# TODO: here we need to add the calibration of energy demand and emissions of ammonia manufacturing.
# for energy demand: probably can be inferred from emissions and constants, though we would need the energy mix (probably we can use the one of chemicals in JRC)
# for emissions: FAOSTAT -> Climate Change -> Totals and Indicators -> Emissions totals -> Fertilizer manufacturing

# note: In practice, ammonia production is often responsible for the bulk of fertilizer manufacturing emissions,
# but the FAO (and IPCC inventories) keep the broader category since other steps matter too (notably N₂O from nitric acid)
# in my case, for emission factors of ammonia-tech I have taken both CO2 and N2O, so probably in my case ammonia-tech
# is closer to fertilizer manufacturing in general. So it probably makes sense to take emissions from fertilizer
# manufacturing from FAO as calibration for the emission of the Ammonia module.

# note: FAO data has only available data for "synthetic fertilizers" (and only N2O), and not "fertlizier manufacturing"
# I would assume that nitrogen emissions from synthetic fertilizers are from the use and not manufacturing.
# So for the moment I do not take nothing, and I create an empty database for this calibration


def run(years_ots, years_fts):

    # directories
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    dm_trade = get_ammonia_trade_data(current_file_directory)

    dm_amm_prod_net_import, dm_amm_mat_net_import, dm_amm_prod = make_ammonia_dms(
        dm_trade, years_ots, years_fts
    )

    return dm_amm_prod_net_import, dm_amm_mat_net_import, dm_amm_prod


if __name__ == "__main__":

    years_ots = create_years_list(1990, 2023, 1)
    years_fts = create_years_list(2025, 2050, 5)

    run(years_ots, years_fts)
