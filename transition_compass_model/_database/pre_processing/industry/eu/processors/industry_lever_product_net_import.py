# packages
import os
import pickle
import warnings

import numpy as np

warnings.simplefilter("ignore")
import plotly.io as pio

pio.renderers.default = "browser"

from transition_compass_model._database.pre_processing.industry.eu.get_data_functions.data_product_net_import import (
    get_prodcom_production_data,
)
from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    linear_fitting,
)

# from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat


def make_product_net_import_dm(current_file_directory, years_ots):
    #########################################
    ##### GET CLEAN DATAFRAME WITH DATA #####
    #########################################

    dm_trade = get_prodcom_production_data(current_file_directory)

    ###################
    ##### FIX OTS #####
    ###################

    # Set years range
    years_setting = [1990, 2023, 2050, 5]
    startyear = years_setting[0]
    baseyear = years_setting[1]
    lastyear = years_setting[2]
    step_fts = years_setting[3]
    years_ots = list(range(startyear, baseyear + 1, 1))
    years_fts = list(range(baseyear + 2, lastyear + 1, step_fts))
    years_all = years_ots + years_fts

    # new variabs list
    dict_new = {}

    # put missing values where needed
    idx = dm_trade.idx
    for y in range(1995, 2010 + 1):
        dm_trade.array[..., idx[y], idx["product-demand_aluminium-pack"]] = np.nan
        dm_trade.array[..., idx[y], idx["product-export_aluminium-pack"]] = np.nan
        dm_trade.array[..., idx[y], idx["product-export_glass-pack"]] = np.nan
    for y in range(2022, 2023 + 1):
        dm_trade.array[..., idx[y], idx["product-demand_fridge"]] = np.nan
        dm_trade.array[..., idx[y], idx["product-export_fridge"]] = np.nan
        dm_trade.array[..., idx[y], idx["product-import_fridge"]] = np.nan
    for v in ["phone", "aviation_ICE"]:
        dm_trade.array[:, idx[2023], idx["product-demand_" + v]] = np.nan
        dm_trade.array[:, idx[2023], idx["product-export_" + v]] = np.nan
        dm_trade.array[:, idx[2023], idx["product-import_" + v]] = np.nan
        dm_trade.array[:, idx[1995], idx["product-demand_" + v]] = dm_trade.array[
            :, idx[2022], idx["product-demand_" + v]
        ] / (2022 - 1995)
        dm_trade.array[:, idx[1995], idx["product-export_" + v]] = dm_trade.array[
            :, idx[2022], idx["product-export_" + v]
        ] / (2022 - 1995)
        dm_trade.array[:, idx[1995], idx["product-import_" + v]] = dm_trade.array[
            :, idx[2022], idx["product-import_" + v]
        ] / (2022 - 1995)
    for v in ["HDV_BEV", "HDV_ICE-gasoline", "HDV_PHEV-diesel"]:
        dm_trade.array[:, idx[2022], idx["product-demand_" + v]] = np.nan
        dm_trade.array[:, idx[2022], idx["product-export_" + v]] = np.nan
        dm_trade.array[:, idx[2022], idx["product-import_" + v]] = np.nan
        dm_trade.array[:, idx[1995], idx["product-demand_" + v]] = dm_trade.array[
            :, idx[2023], idx["product-demand_" + v]
        ] / (2023 - 1995)
        dm_trade.array[:, idx[1995], idx["product-export_" + v]] = dm_trade.array[
            :, idx[2023], idx["product-export_" + v]
        ] / (2023 - 1995)
        dm_trade.array[:, idx[1995], idx["product-import_" + v]] = dm_trade.array[
            :, idx[2023], idx["product-import_" + v]
        ] / (2023 - 1995)
    for y in range(2009, 2021 + 1):
        dm_trade.array[..., idx[y], idx["product-demand_HDV_ICE-diesel"]] = np.nan
        dm_trade.array[..., idx[y], idx["product-export_HDV_ICE-diesel"]] = np.nan
        dm_trade.array[..., idx[y], idx["product-import_HDV_ICE-diesel"]] = np.nan
    for y in [2003, 2015, 2019, 2020]:
        dm_trade.array[..., idx[y], idx["product-demand_rail_CEV"]] = np.nan
    dm_trade.array[..., idx[2017], idx["product-demand_rail_ICE-diesel"]] = np.nan
    for y in range(1995, 2006 + 1):
        dm_trade.array[..., idx[y], idx["product-import_rail_CEV"]] = np.nan
    for y in [2019, 2020]:
        dm_trade.array[..., idx[y], idx["product-import_bus_ICE-diesel"]] = np.nan
    dm_trade.array[:, idx[2022], idx["product-demand_marine_ICE-diesel"]] = np.nan
    dm_trade.array[:, idx[2022], idx["product-export_marine_ICE-diesel"]] = np.nan
    # dm_trade.array[:,idx[2023],idx["product-import_HDV_PHEV-diesel"]] = np.nan
    dm_trade.array[:, idx[1995], idx["product-import_HDV_PHEV-diesel"]] = (
        dm_trade.array[:, idx[2022], idx["product-import_HDV_PHEV-diesel"]]
        / (2022 - 1995)
    )
    for y in range(1996, 2021 + 1):
        dm_trade.array[:, idx[y], idx["product-import_HDV_PHEV-diesel"]] = (
            dm_trade.array[:, idx[1995], idx["product-import_HDV_PHEV-diesel"]]
        )

    # function to adjust ots
    def make_ots(variable, based_on):
        dm_temp = dm_trade.filter({"Variables": [variable]})
        dm_temp = linear_fitting(
            dm_temp, years_ots, based_on=based_on, min_t0=0.1, min_tb=0.1
        )
        return dm_temp

    dict_call = {
        "product-demand_HDV_BEV": None,
        "product-demand_HDV_ICE-diesel": None,
        "product-demand_HDV_ICE-gasoline": None,
        "product-demand_HDV_PHEV-diesel": None,
        "product-demand_LDV_BEV": list(range(2017, 2018 + 1)),
        "product-demand_LDV_ICE-diesel": None,
        "product-demand_LDV_ICE-gasoline": None,
        "product-demand_LDV_PHEV-gasoline": list(range(2017, 2019 + 1)),
        "product-demand_aluminium-pack": None,
        "product-demand_aviation_ICE": None,
        "product-demand_bus_ICE-diesel": None,
        "product-demand_computer": list(range(2003, 2011 + 1)),
        "product-demand_dishwasher": None,
        "product-demand_fertilizer": None,
        "product-demand_freezer": None,
        "product-demand_fridge": None,
        "product-demand_glass-pack": list(range(2010, 2023 + 1)),
        "product-demand_marine_ICE-diesel": None,
        "product-demand_paper-pack": None,
        "product-demand_paper-print": list(range(2003, 2009 + 1)),
        "product-demand_paper-san": None,
        "product-demand_phone": None,
        "product-demand_plastic-pack": None,
        "product-demand_rail_CEV": None,
        "product-demand_rail_ICE-diesel": None,
        "product-demand_tv": list(range(2003, 2010 + 1)),
        "product-demand_wmachine": None,
        "product-export_HDV_BEV": None,
        "product-export_HDV_ICE-diesel": None,
        "product-export_HDV_ICE-gasoline": None,
        "product-export_HDV_PHEV-diesel": None,
        "product-export_LDV_BEV": list(range(2017, 2018 + 1)),
        "product-export_LDV_ICE-diesel": list(range(2003, 2019 + 1)),
        "product-export_LDV_ICE-gasoline": list(range(2003, 2007 + 1)),
        "product-export_LDV_PHEV-gasoline": list(range(2017, 2018 + 1)),
        "product-export_aluminium-pack": None,
        "product-export_aviation_ICE": None,
        "product-export_bus_ICE-diesel": None,
        "product-export_computer": list(range(2003, 2011 + 1)),
        "product-export_dishwasher": None,
        "product-export_fertilizer": None,
        "product-export_freezer": None,
        "product-export_fridge": None,
        "product-export_glass-pack": list(range(2011, 2014 + 1)),
        "product-export_marine_ICE-diesel": list(range(2003, 2007 + 1)),
        "product-export_paper-pack": None,
        "product-export_paper-print": list(range(2003, 2009 + 1)),
        "product-export_paper-san": None,
        "product-export_phone": None,
        "product-export_plastic-pack": None,
        "product-export_rail_CEV": None,
        "product-export_tv": list(range(2003, 2008 + 1)),
        "product-export_wmachine": None,
        "product-import_HDV_BEV": None,
        "product-import_HDV_ICE-diesel": None,
        "product-import_HDV_ICE-gasoline": None,
        "product-import_HDV_PHEV-diesel": None,
        "product-import_LDV_BEV": list(range(2017, 2018 + 1)),
        "product-import_LDV_ICE-diesel": list(range(2003, 2007 + 1)),
        "product-import_LDV_ICE-gasoline": None,
        "product-import_LDV_PHEV-gasoline": list(range(2017, 2019 + 1)),
        "product-import_aluminium-pack": list(range(2003, 2016 + 1)),
        "product-import_aviation_ICE": None,
        "product-import_bus_ICE-diesel": None,
        "product-import_computer": list(range(2003, 2011 + 1)),
        "product-import_dishwasher": None,
        "product-import_fertilizer": None,
        "product-import_freezer": None,
        "product-import_fridge": None,
        "product-import_glass-pack": None,
        "product-import_marine_ICE-diesel": None,
        "product-import_paper-pack": None,
        "product-import_paper-print": None,
        "product-import_paper-san": None,
        "product-import_phone": None,
        "product-import_plastic-pack": None,
        "product-import_rail_CEV": None,
        "product-import_tv": None,
        "product-import_wmachine": None,
        "product-demand_2W_ICE-gasoline": None,
        "product-export_2W_ICE-gasoline": None,
        "product-import_2W_ICE-gasoline": None,
    }

    for key in dict_call.keys():
        dict_new[key] = make_ots(key, based_on=dict_call[key])

    # append
    dm_trade_temp = dict_new["product-demand_aluminium-pack"].copy()
    mylist = list(dict_call.keys())
    mylist.remove("product-demand_aluminium-pack")
    for v in mylist:
        dm_trade_temp.append(dict_new[v], "Variables")
    dm_trade_temp.sort("Variables")
    dm_trade = dm_trade_temp.copy()

    # check
    # dm_trade.filter({"Country" : ["EU27"]}).datamatrix_plot()

    #############################################
    ##### GENERATE VARIABLES WE DO NOT HAVE #####
    #############################################

    # put together
    DM_trade = {}
    dm_temp = dm_trade.filter(
        {
            "Variables": [
                "product-demand_HDV_BEV",
                "product-demand_HDV_ICE-diesel",
                "product-demand_HDV_ICE-gasoline",
                "product-demand_HDV_PHEV-diesel",
                "product-demand_LDV_BEV",
                "product-demand_LDV_ICE-diesel",
                "product-demand_LDV_ICE-gasoline",
                "product-demand_LDV_PHEV-gasoline",
                "product-demand_aviation_ICE",
                "product-demand_bus_ICE-diesel",
                "product-demand_marine_ICE-diesel",
                "product-demand_rail_CEV",
                "product-demand_rail_ICE-diesel",
                "product-export_HDV_BEV",
                "product-export_HDV_ICE-diesel",
                "product-export_HDV_ICE-gasoline",
                "product-export_HDV_PHEV-diesel",
                "product-export_LDV_BEV",
                "product-export_LDV_ICE-diesel",
                "product-export_LDV_ICE-gasoline",
                "product-export_LDV_PHEV-gasoline",
                "product-export_aviation_ICE",
                "product-export_bus_ICE-diesel",
                "product-export_marine_ICE-diesel",
                "product-export_rail_CEV",
                "product-import_HDV_ICE-gasoline",
                "product-import_HDV_PHEV-diesel",
                "product-import_LDV_BEV",
                "product-import_LDV_ICE-diesel",
                "product-import_LDV_ICE-gasoline",
                "product-import_LDV_PHEV-gasoline",
                "product-import_aviation_ICE",
                "product-import_bus_ICE-diesel",
                "product-import_marine_ICE-diesel",
                "product-import_rail_CEV",
                "product-demand_2W_ICE-gasoline",
                "product-export_2W_ICE-gasoline",
                "product-import_2W_ICE-gasoline",
            ]
        }
    )
    dm_temp.deepen_twice()
    DM_trade["tra-veh"] = dm_temp
    dm_temp = dm_trade.filter(
        {
            "Variables": [
                "product-demand_aluminium-pack",
                "product-export_aluminium-pack",
                "product-import_aluminium-pack",
            ]
        }
    )
    dm_temp.deepen()
    DM_trade["pack-alu"] = dm_temp
    dm_temp = dm_trade.filter(
        {
            "Variables": [
                "product-demand_computer",
                "product-demand_dishwasher",
                "product-demand_freezer",
                "product-demand_fridge",
                "product-demand_phone",
                "product-demand_tv",
                "product-demand_wmachine",
                "product-export_computer",
                "product-export_dishwasher",
                "product-export_freezer",
                "product-export_fridge",
                "product-export_phone",
                "product-export_tv",
                "product-export_wmachine",
                "product-import_computer",
                "product-import_dishwasher",
                "product-import_freezer",
                "product-import_fridge",
                "product-import_phone",
                "product-import_tv",
                "product-import_wmachine",
            ]
        }
    )
    dm_temp.deepen()
    DM_trade["domapp"] = dm_temp
    dm_temp = dm_trade.filter(
        {
            "Variables": [
                "product-demand_glass-pack",
                "product-demand_paper-pack",
                "product-demand_paper-print",
                "product-demand_paper-san",
                "product-demand_plastic-pack",
                "product-export_glass-pack",
                "product-export_paper-pack",
                "product-export_paper-print",
                "product-export_paper-san",
                "product-export_plastic-pack",
                "product-import_glass-pack",
                "product-import_paper-pack",
                "product-import_paper-print",
                "product-import_paper-san",
                "product-import_plastic-pack",
            ]
        }
    )
    dm_temp.deepen()
    DM_trade["pack"] = dm_temp
    dm_temp = dm_trade.filter(
        {
            "Variables": [
                "product-demand_fertilizer",
                "product-export_fertilizer",
                "product-import_fertilizer",
            ]
        }
    )
    dm_temp.deepen()
    DM_trade["fertilizer"] = dm_temp

    # note: for the variables that we do not have, in general import and export will be set
    # to zero, and demand will be set to nan

    # generate cars-FCV and trucks-FCV
    # we assume that imports remain zero throughout
    idx = DM_trade["tra-veh"].idx
    DM_trade["tra-veh"].add(0, "Categories2", "FCEV", unit="num", dummy=True)
    DM_trade["tra-veh"].array[:, :, idx["product-demand"], :, idx["FCEV"]] = np.nan
    DM_trade["tra-veh"].add(0, "Categories2", "ICE-gas", unit="num", dummy=True)
    DM_trade["tra-veh"].array[:, :, idx["product-demand"], :, idx["ICE-gas"]] = np.nan
    DM_trade["tra-veh"].sort("Categories2")

    # generate new-dhg-pipe, rail, road, trolley-cables, floor-area-new-non-residential,
    # floor-area-new-residential, floor-area-reno-non-residential, floor-area-reno-residential
    # we assume imports of these are all zero
    DM_trade["domapp"].add(0, "Categories1", "new-dhg-pipe", unit="num", dummy=True)
    dm_bld_pipe = DM_trade["domapp"].filter({"Categories1": ["new-dhg-pipe"]})
    dm_bld_pipe.units["product-export"] = "km"
    dm_bld_pipe.units["product-import"] = "km"
    dm_bld_pipe.units["product-demand"] = "km"
    idx = dm_bld_pipe.idx
    dm_bld_pipe.array[:, :, idx["product-demand"], idx["new-dhg-pipe"]] = np.nan
    DM_trade["domapp"].drop("Categories1", ["new-dhg-pipe"])
    DM_trade["pipe"] = dm_bld_pipe

    dm_tra_infra = dm_bld_pipe.copy()
    dm_tra_infra.rename_col("new-dhg-pipe", "rail", "Categories1")
    dm_temp = dm_tra_infra.copy()
    dm_temp.rename_col("rail", "road", "Categories1")
    dm_tra_infra.append(dm_temp, "Categories1")
    dm_temp = dm_tra_infra.filter({"Categories1": ["rail"]})
    dm_temp.rename_col("rail", "trolley-cables", "Categories1")
    dm_tra_infra.append(dm_temp, "Categories1")
    DM_trade["tra-infra"] = dm_tra_infra.copy()

    DM_trade["domapp"].add(
        0, "Categories1", "floor-area-new-non-residential", unit="m2", dummy=True
    )
    DM_trade["domapp"].add(
        0, "Categories1", "floor-area-new-residential", unit="m2", dummy=True
    )
    DM_trade["domapp"].add(
        0, "Categories1", "floor-area-reno-non-residential", unit="m2", dummy=True
    )
    DM_trade["domapp"].add(
        0, "Categories1", "floor-area-reno-residential", unit="m2", dummy=True
    )
    dm_bld_floor = DM_trade["domapp"].filter(
        {
            "Categories1": [
                "floor-area-new-non-residential",
                "floor-area-new-residential",
                "floor-area-reno-non-residential",
                "floor-area-reno-residential",
            ]
        }
    )
    dm_bld_floor.units["product-export"] = "m2"
    dm_bld_floor.units["product-import"] = "m2"
    dm_bld_floor.units["product-demand"] = "m2"
    idx = dm_bld_floor.idx
    dm_bld_floor.array[:, :, idx["product-demand"], :] = np.nan
    DM_trade["domapp"].drop(
        "Categories1",
        [
            "floor-area-new-non-residential",
            "floor-area-new-residential",
            "floor-area-reno-non-residential",
            "floor-area-reno-residential",
        ],
    )
    dm_bld_floor.sort("Categories1")
    DM_trade["bld-floor"] = dm_bld_floor

    # dryer
    # I assume dryers are 1% of exports and imports of w machine (check excel file in WITS folder called "percentage_dryers_export_EU")
    idx = DM_trade["domapp"].idx
    arr_temp = DM_trade["domapp"].array[..., idx["wmachine"]] * 0.01
    DM_trade["domapp"].add(arr_temp, col_label="dryer", dim="Categories1", unit="num")
    DM_trade["domapp"].sort("Categories1")

    # check
    # DM_trade['tra-veh'].filter({"Country" : ["EU27"]}).datamatrix_plot()
    # DM_trade['pack-alu'].filter({"Country" : ["EU27"]}).datamatrix_plot()
    # DM_trade['domapp'].filter({"Country" : ["EU27"]}).datamatrix_plot()
    # DM_trade['pack'].filter({"Country" : ["EU27"]}).datamatrix_plot()

    # ####################
    # ##### MAKE FTS #####
    # ####################

    # # flatten tra veh and make electric total and total which will be used for the projections
    # DM_trade["tra-veh"] = DM_trade["tra-veh"].flatten()
    # dm_temp = DM_trade["tra-veh"].groupby(
    #     {
    #         "electric_total": [
    #             "HDV_BEV",
    #             "HDV_PHEV-diesel",
    #             "LDV_BEV",
    #             "LDV_PHEV-gasoline",
    #         ],
    #         "total": [
    #             "HDV_BEV",
    #             "HDV_FCEV",
    #             "HDV_ICE-diesel",
    #             "HDV_ICE-gas",
    #             "HDV_ICE-gasoline",
    #             "HDV_PHEV-diesel",
    #             "LDV_BEV",
    #             "LDV_FCEV",
    #             "LDV_ICE-diesel",
    #             "LDV_ICE-gas",
    #             "LDV_ICE-gasoline",
    #             "LDV_PHEV-gasoline",
    #         ],
    #     },
    #     "Categories1",
    #     inplace=False,
    # )
    # DM_trade["tra-veh"].append(dm_temp, "Categories1")
    # DM_trade["tra-veh"].drop(
    #     "Categories1",
    #     [
    #         "aviation_FCEV",
    #         "aviation_ICE-gas",
    #         "bus_FCEV",
    #         "marine_FCEV",
    #         "rail_FCEV",
    #         "rail_ICE-gas",
    #         "marine_ICE-gas",
    #     ],
    # )

    # # make function to fill in missing years fts for EU27 with linear fitting
    # def make_fts(
    #     dm,
    #     variable,
    #     year_start,
    #     year_end,
    #     country="EU27",
    #     dim="Categories1",
    #     min_t0=0,
    #     min_tb=0,
    #     years_fts=years_fts,
    # ):  # I put minimum to 1 so it does not go to zero
    #     dm = dm.copy()
    #     idx = dm.idx
    #     based_on_yars = list(range(year_start, year_end + 1, 1))
    #     dm_temp = linear_fitting(
    #         dm.filter({"Country": [country], dim: [variable]}),
    #         years_ots=years_fts,
    #         min_t0=min_t0,
    #         min_tb=min_tb,
    #         based_on=based_on_yars,
    #     )
    #     idx_temp = dm_temp.idx
    #     if dim == "Variables":
    #         dm.array[idx[country], :, idx[variable], ...] = np.round(
    #             dm_temp.array[idx_temp[country], :, idx_temp[variable], ...], 0
    #         )
    #     if dim == "Categories1":
    #         dm.array[idx[country], :, :, idx[variable]] = np.round(
    #             dm_temp.array[idx_temp[country], :, :, idx_temp[variable]], 0
    #         )
    #     if dim == "Categories2":
    #         dm.array[idx[country], :, :, :, idx[variable]] = np.round(
    #             dm_temp.array[idx_temp[country], :, :, :, idx_temp[variable]], 0
    #         )
    #     if dim == "Categories3":
    #         dm.array[idx[country], :, :, :, :, idx[variable]] = np.round(
    #             dm_temp.array[idx_temp[country], :, :, :, :, idx_temp[variable]], 0
    #         )

    #     return dm

    # # add missing years fts
    # for key in DM_trade.keys():
    #     DM_trade[key].add(np.nan, col_label=years_fts, dummy=True, dim="Years")

    # # set default time window for linear trend
    # # assumption: best is taking longer trend possible to make predictions to 2050 (even if earlier data is generated)
    # baseyear_start = 1990
    # baseyear_end = 2023

    # # packages
    # DM_trade["pack-alu"] = make_fts(
    #     DM_trade["pack-alu"], "aluminium-pack", baseyear_start, baseyear_end
    # )
    # DM_trade["pack"] = make_fts(
    #     DM_trade["pack"], "glass-pack", 2012, baseyear_end
    # )  # here upwatd trend in import and demand starts in 2012
    # DM_trade["pack"] = make_fts(
    #     DM_trade["pack"], "plastic-pack", baseyear_start, baseyear_end
    # )
    # DM_trade["pack"] = make_fts(DM_trade["pack"], "paper-pack", 2009, baseyear_end)
    # DM_trade["pack"] = make_fts(
    #     DM_trade["pack"], "paper-print", baseyear_start, baseyear_end
    # )
    # DM_trade["pack"] = make_fts(DM_trade["pack"], "paper-san", baseyear_start, baseyear_end)
    # # product = "plastic-pack"
    # # (make_fts(DM_trade["pack"], product, baseyear_start, baseyear_end).
    # #   datamatrix_plot(selected_cols={"Country" : ["EU27"],
    # #                                 "Categories1" : [product]}))

    # # electric vehicles

    # # note: assuming 8% of total fleet being electric in 2050
    # # source: https://www.eea.europa.eu/publications/electric-vehicles-and-the-energy/download
    # DM_trade["tra-veh"] = make_fts(
    #     DM_trade["tra-veh"], "total", baseyear_start, baseyear_end
    # )
    # idx = DM_trade["tra-veh"].idx
    # electric_2050 = np.round(
    #     DM_trade["tra-veh"].array[idx["EU27"], idx[2050], :, idx["total"]] * 0.20
    # )  # in the end I put 20% here as with this data electric things are already close to 8% in 2023
    # dm_share = DM_trade["tra-veh"].filter(
    #     {
    #         "Country": ["EU27"],
    #         "Years": [2023],
    #         "Categories1": ["HDV_BEV", "HDV_PHEV-diesel", "LDV_BEV", "LDV_PHEV-gasoline"],
    #     }
    # )
    # dm_share.normalise("Categories1")
    # idx_share = dm_share.idx
    # HDV_BEV_2050 = np.round(dm_share.array[..., idx_share["HDV_BEV"]] * electric_2050, 0)
    # HDV_PHEV_diesel_2050 = np.round(
    #     dm_share.array[..., idx_share["HDV_PHEV-diesel"]] * electric_2050, 0
    # )
    # LDV_BEV_2050 = np.round(dm_share.array[..., idx_share["LDV_BEV"]] * electric_2050, 0)
    # LDV_PHEV_gasoline_2050 = np.round(
    #     dm_share.array[..., idx_share["LDV_PHEV-gasoline"]] * electric_2050, 0
    # )
    # DM_trade["tra-veh"].array[idx["EU27"], idx[2050], :, idx["HDV_BEV"]] = HDV_BEV_2050
    # DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "HDV_BEV", 2023, 2050)
    # DM_trade["tra-veh"].array[idx["EU27"], idx[2050], :, idx["HDV_PHEV-diesel"]] = (
    #     HDV_PHEV_diesel_2050
    # )
    # DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "HDV_PHEV-diesel", 2023, 2050)
    # DM_trade["tra-veh"].array[idx["EU27"], idx[2050], :, idx["LDV_BEV"]] = LDV_BEV_2050
    # DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "LDV_BEV", 2023, 2050)
    # DM_trade["tra-veh"].array[idx["EU27"], idx[2050], :, idx["LDV_PHEV-gasoline"]] = (
    #     LDV_PHEV_gasoline_2050
    # )
    # DM_trade["tra-veh"] = make_fts(DM_trade["tra-veh"], "LDV_PHEV-gasoline", 2023, 2050)
    # DM_trade["tra-veh"].drop("Categories1", ["electric_total", "total"])

    # # DM_trade['tra-veh'].filter({"Country" : ["EU27"]}).datamatrix_plot()

    # # rest of transport
    # DM_trade["tra-veh"] = make_fts(
    #     DM_trade["tra-veh"], "HDV_FCEV", baseyear_start, baseyear_end
    # )
    # DM_trade["tra-veh"] = make_fts(
    #     DM_trade["tra-veh"], "HDV_ICE-diesel", baseyear_start, baseyear_end
    # )
    # DM_trade["tra-veh"] = make_fts(
    #     DM_trade["tra-veh"], "HDV_ICE-gas", baseyear_start, baseyear_end
    # )
    # DM_trade["tra-veh"] = make_fts(
    #     DM_trade["tra-veh"], "HDV_ICE-gasoline", baseyear_start, baseyear_end
    # )
    # DM_trade["tra-veh"] = make_fts(
    #     DM_trade["tra-veh"], "LDV_FCEV", baseyear_start, baseyear_end
    # )
    # DM_trade["tra-veh"] = make_fts(
    #     DM_trade["tra-veh"], "LDV_ICE-diesel", baseyear_start, baseyear_end
    # )
    # DM_trade["tra-veh"] = make_fts(
    #     DM_trade["tra-veh"], "LDV_ICE-gas", baseyear_start, baseyear_end
    # )
    # DM_trade["tra-veh"] = make_fts(
    #     DM_trade["tra-veh"], "LDV_ICE-gasoline", baseyear_start, baseyear_end
    # )
    # DM_trade["tra-veh"] = make_fts(
    #     DM_trade["tra-veh"], "aviation_ICE", baseyear_start, baseyear_end
    # )
    # DM_trade["tra-veh"] = make_fts(
    #     DM_trade["tra-veh"], "bus_ICE-diesel", baseyear_start, baseyear_end
    # )
    # DM_trade["tra-veh"] = make_fts(
    #     DM_trade["tra-veh"], "bus_ICE-gas", baseyear_start, baseyear_end
    # )
    # DM_trade["tra-veh"] = make_fts(
    #     DM_trade["tra-veh"], "marine_ICE-diesel", baseyear_start, baseyear_end
    # )
    # DM_trade["tra-veh"] = make_fts(
    #     DM_trade["tra-veh"], "rail_CEV", baseyear_start, baseyear_end
    # )
    # DM_trade["tra-veh"] = make_fts(
    #     DM_trade["tra-veh"], "rail_ICE-diesel", baseyear_start, baseyear_end
    # )

    # # rename rail to train, aviation to planes, marine to ships
    # DM_trade["tra-veh"].rename_col_regex("rail", "trains", "Categories1")
    # DM_trade["tra-veh"].rename_col_regex("aviation", "planes", "Categories1")
    # DM_trade["tra-veh"].rename_col_regex("marine", "ships", "Categories1")
    # DM_trade["tra-veh"].sort("Categories1")

    # # transport infra
    # DM_trade["tra-infra"] = make_fts(
    #     DM_trade["tra-infra"], "rail", baseyear_start, baseyear_end
    # )
    # DM_trade["tra-infra"] = make_fts(
    #     DM_trade["tra-infra"], "road", baseyear_start, baseyear_end
    # )
    # DM_trade["tra-infra"] = make_fts(
    #     DM_trade["tra-infra"], "trolley-cables", baseyear_start, baseyear_end
    # )

    # # buildings
    # DM_trade["bld-floor"] = make_fts(
    #     DM_trade["bld-floor"],
    #     "floor-area-new-non-residential",
    #     baseyear_start,
    #     baseyear_end,
    # )
    # DM_trade["bld-floor"] = make_fts(
    #     DM_trade["bld-floor"], "floor-area-new-residential", baseyear_start, baseyear_end
    # )
    # DM_trade["bld-floor"] = make_fts(
    #     DM_trade["bld-floor"],
    #     "floor-area-reno-non-residential",
    #     baseyear_start,
    #     baseyear_end,
    # )
    # DM_trade["bld-floor"] = make_fts(
    #     DM_trade["bld-floor"], "floor-area-reno-residential", baseyear_start, baseyear_end
    # )

    # # domestic appliances
    # DM_trade["domapp"] = make_fts(
    #     DM_trade["domapp"], "computer", baseyear_start, baseyear_end
    # )
    # DM_trade["domapp"] = make_fts(
    #     DM_trade["domapp"], "dishwasher", baseyear_start, baseyear_end
    # )
    # DM_trade["domapp"] = make_fts(
    #     DM_trade["domapp"], "dryer", 2000, 2007
    # )  # here I assume there is some problem with the data after 2008
    # DM_trade["domapp"] = make_fts(
    #     DM_trade["domapp"], "freezer", baseyear_start, baseyear_end
    # )
    # DM_trade["domapp"] = make_fts(
    #     DM_trade["domapp"], "fridge", baseyear_start, baseyear_end
    # )
    # DM_trade["domapp"] = make_fts(DM_trade["domapp"], "phone", baseyear_start, baseyear_end)
    # DM_trade["domapp"] = make_fts(
    #     DM_trade["domapp"], "tv", 2012, baseyear_end
    # )  # downward trend in demand since 2012
    # DM_trade["domapp"] = make_fts(
    #     DM_trade["domapp"], "wmachine", 2000, 2007
    # )  # here I assume there is some problem with the data after 2008

    # # pipes
    # DM_trade["pipe"] = make_fts(
    #     DM_trade["pipe"], "new-dhg-pipe", baseyear_start, baseyear_end
    # )

    # # fertilizer
    # DM_trade["fertilizer"] = make_fts(
    #     DM_trade["fertilizer"], "fertilizer", baseyear_start, baseyear_end
    # )

    # # check
    # # DM_trade['tra-veh'].filter({"Country" : ["EU27"]}).datamatrix_plot()
    # # DM_trade['pack-alu'].filter({"Country" : ["EU27"]}).datamatrix_plot()
    # # DM_trade['domapp'].filter({"Country" : ["EU27"]}).datamatrix_plot()
    # # DM_trade['pack'].filter({"Country" : ["EU27"]}).datamatrix_plot()
    # # DM_trade["fertilizer"].filter({"Country" : ["EU27"]}).datamatrix_plot()

    ###################################
    ##### MAKE PRODUCT NET IMPORT #####
    ###################################

    # dm_temp = DM_trade["tra-veh"].filter_w_regex({"Categories1": "LDV"})
    # dm_temp.group_all("Categories1")
    # dm_temp.operation("product-export","/","product-demand",'Variables',
    #                   'product-export-share',unit="%")
    # dm_temp.operation("product-import","/","product-demand",'Variables',
    #                   'product-import-share',unit="%")
    # idx = dm_temp.idx
    # dm_temp.array[idx["EU27"],idx[2021],idx["product-import-share"]] # 0.14
    # dm_temp.array[idx["EU27"],idx[2021],idx["product-export-share"]] # 0.19

    # product-net-import[%] = (product-import - product-export)/product-demand
    DM_trade["tra-veh"] = DM_trade["tra-veh"].flatten()
    DM_trade["tra-veh"].drop(
        "Categories1",
        [
            "aviation_FCEV",
            "aviation_ICE-gas",
            "bus_FCEV",
            "marine_FCEV",
            "rail_FCEV",
            "rail_ICE-gas",
            "marine_ICE-gas",
            "2W_FCEV",
            "2W_ICE-gas",
        ],
    )

    DM_trade_net_share = {}
    keys = [
        "domapp",
        "tra-veh",
        "pack",
        "pack-alu",
        "pipe",
        "tra-infra",
        "bld-floor",
        "fertilizer",
    ]
    for key in keys:
        dm_temp = DM_trade[key].copy()

        # make product-net-import[%] = (product-import - product-export)/product-demand
        idx = dm_temp.idx
        arr_temp = dm_temp.array
        arr_temp[np.isnan(arr_temp)] = 0  # put zero where nan is
        arr_net = (
            arr_temp[:, :, idx["product-import"], :]
            - arr_temp[:, :, idx["product-export"], :]
        ) / arr_temp[:, :, idx["product-demand"], :]

        # when both import and export are zero, assign a zero
        arr_net[
            (arr_temp[:, :, idx["product-import"], :] == 0)
            & (arr_temp[:, :, idx["product-export"], :] == 0)
        ] = 0
        dm_temp.add(
            arr_net[:, :, np.newaxis, :], "Variables", "product-net-import", unit="%"
        )

        # drop
        dm_temp.drop(
            "Variables", ["product-import", "product-export", "product-demand"]
        )

        # store
        DM_trade_net_share[key] = dm_temp

    dm_trade_netshare = DM_trade_net_share["tra-veh"].copy()
    keys = [
        "domapp",
        "pack",
        "pack-alu",
        "pipe",
        "tra-infra",
        "bld-floor",
        "fertilizer",
    ]
    for key in keys:
        dm_trade_netshare.append(DM_trade_net_share[key], "Categories1")
    dm_trade_netshare.sort("Categories1")

    # fill in missing values for product-net-import (coming from dividing by zero)
    idx = dm_trade_netshare.idx
    dm_trade_netshare.array[np.isinf(dm_trade_netshare.array)] = np.nan
    years_fitting = dm_trade_netshare.col_labels["Years"]
    dm_trade_netshare = linear_fitting(dm_trade_netshare, years_fitting)

    # for the variables that we generated as all zero, re-put zeroes
    variabs = [
        "HDV_FCEV",
        "HDV_ICE-gas",
        "LDV_FCEV",
        "LDV_ICE-gas",
        "bus_ICE-gas",
        "new-dhg-pipe",
        "rail",
        "road",
        "trolley-cables",
        "floor-area-new-non-residential",
        "floor-area-new-residential",
        "floor-area-reno-non-residential",
        "floor-area-reno-residential",
    ]
    idx = dm_trade_netshare.idx
    for v in variabs:
        dm_trade_netshare.array[:, :, :, idx[v]] = 0

    # # fix jumps in product-net-import
    # dm_trade_netshare = fix_jumps_in_dm(dm_trade_netshare)

    # Should having values above and below, respectively, 1 and -1 be a problem? probably
    # it is if it's larger than 1, as it would mean that we are importing more than the demand ... the
    # only reason why that could be is that a the EU27 is importing just to re-export, though
    # I guess this is not the norm, and I would probably rule out this situation ...
    # on the other hand if it's less than -1 should be fine, as it would mean we are producing
    # more than the local demand, and the rest is exported.
    # for trains, we have values well above and below, respectively, 1 and -1.
    # For computers, we have values well above 1.
    # For dryer, we have values below -1 (gets to -3.5).
    # For freezer, fridge, phone, we have values well above 1.
    # Let's cap everything to max 1
    dm_trade_netshare.array[dm_trade_netshare.array > 1] = 1

    # 2W_BEV: same net import as 2W_ICE-gasoline (no PRODCOM data for electric 2W)
    idx = dm_trade_netshare.idx
    arr_2w = dm_trade_netshare.array[:, :, :, idx["2W_ICE-gasoline"]]
    if "2W_BEV" in dm_trade_netshare.col_labels["Categories1"]:
        dm_trade_netshare.array[:, :, :, idx["2W_BEV"]] = arr_2w
    else:
        dm_trade_netshare.add(arr_2w, "Categories1", "2W_BEV", unit="%")

    # add HDV_PHEV-gasoline and LDV_PHEV-diesel
    idx = dm_trade_netshare.idx
    arr_temp = dm_trade_netshare.array[:, :, :, idx["HDV_PHEV-diesel"]]
    dm_trade_netshare.add(arr_temp, "Categories1", "HDV_PHEV-gasoline", unit="%")
    arr_temp = dm_trade_netshare.array[:, :, :, idx["LDV_PHEV-gasoline"]]
    dm_trade_netshare.add(arr_temp, "Categories1", "LDV_PHEV-diesel", unit="%")
    dm_trade_netshare.sort("Categories1")

    # rename to match model variable names
    dm_trade_netshare.rename_col("aviation_ICE", "planes_ICE", "Categories1")
    dm_trade_netshare.rename_col("marine_ICE-diesel", "ships_ICE-diesel", "Categories1")
    dm_trade_netshare.rename_col("rail_CEV", "trains_CEV", "Categories1")
    dm_trade_netshare.rename_col("rail_ICE-diesel", "trains_ICE-diesel", "Categories1")
    dm_trade_netshare.sort("Categories1")

    # check
    # dm_trade_netshare.filter({"Country" : ["EU27"]}).datamatrix_plot()
    # DM_trade["tra-veh"].filter({"Country" : ["EU27"]}).datamatrix_plot()
    # df_check = dm_trade_netshare.filter({"Country" : ["EU27"]}).write_df()

    ################################
    ##### MAKE PAPERPACK LEVER #####
    ################################

    # the paper pack lever is the demand for packages per capita. The unit is ton/cap.
    # for the population, we upload the population data in lifestyles

    # load DM_pack
    filepath = os.path.join(
        current_file_directory,
        "../../../../pre_processing/lifestyles/Europe/data/lifestyles_allcountries.pickle",
    )
    with open(filepath, "rb") as handle:
        DM_pack = pickle.load(handle)

    # get population data
    dm_pop = DM_pack["ots"]["pop"]["lfs_population_"].copy()
    # dm_pop.append(DM_pack["fts"]["pop"]["lfs_population_"][1], "Years")

    # get aluminium package data
    dm_alu = DM_trade["pack-alu"].filter({"Variables": ["product-demand"]})

    # assuming an average of 30 g per unit, so 0.03 kg per unit
    dm_alu.array = dm_alu.array * 0.03
    dm_alu.units["product-demand"] = "kg"

    # put together with rest of packaging (which is already in kg)
    dm_pack = DM_trade["pack"].filter({"Variables": ["product-demand"]})
    dm_pack.append(dm_alu, "Categories1")
    dm_pack.sort("Categories1")

    # make kg to tonnes
    dm_pack.change_unit("product-demand", factor=1e-3, old_unit="kg", new_unit="t")

    # make tonne per capita
    # dm_pop.drop("Country",['Switzerland','Vaud'])
    dm_pack.array = dm_pack.array / dm_pop.array[..., np.newaxis]
    dm_pack.units["product-demand"] = "t/cap"

    ################
    ##### SAVE #####
    ################

    # # save dm_trade_netshare
    # years_ots = list(range(1990, 2023 + 1))
    # years_fts = list(range(2025, 2055, 5))
    # dm_ots = dm_trade_netshare.filter({"Years": years_ots})
    # dm_fts = dm_trade_netshare.filter({"Years": years_fts})
    # DM_fts = {
    #     1: dm_fts.copy(),
    #     2: dm_fts.copy(),
    #     3: dm_fts.copy(),
    #     4: dm_fts.copy(),
    # }  # for now we set all levels to be the same
    # DM = {"ots": dm_ots, "fts": DM_fts}
    f = os.path.join(
        current_file_directory, "../data/datamatrix/lever_product-net-import.pickle"
    )
    with open(f, "wb") as handle:
        pickle.dump(dm_trade_netshare, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # save paperpack
    # years_ots = list(range(1990, 2023 + 1))
    # years_fts = list(range(2025, 2055, 5))
    # dm_ots = dm_pack.filter({"Years": years_ots})
    # dm_fts = dm_pack.filter({"Years": years_fts})
    # DM_fts = {
    #     1: dm_fts.copy(),
    #     2: dm_fts.copy(),
    #     3: dm_fts.copy(),
    #     4: dm_fts.copy(),
    # }  # for now we set all levels to be the same
    # DM = {"ots": dm_ots, "fts": DM_fts}
    f = os.path.join(
        current_file_directory, "../data/datamatrix/lever_paperpack.pickle"
    )
    with open(f, "wb") as handle:
        pickle.dump(dm_pack, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm_trade_netshare, dm_pack


def run(years_ots):
    # directories
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # if exists, load, else make
    lever_files = ["lever_product-net-import.pickle", "lever_paperpack.pickle"]
    filepaths = [
        os.path.join(current_file_directory, "../data/datamatrix/" + lever_file)
        for lever_file in lever_files
    ]
    true_condition = all([os.path.exists(filepath) for filepath in filepaths])
    if true_condition:
        with open(filepaths[0], "rb") as handle:
            dm_trade_netshare = pickle.load(handle)
        with open(filepaths[1], "rb") as handle:
            dm_pack = pickle.load(handle)

    else:
        dm_trade_netshare, dm_pack = make_product_net_import_dm(
            current_file_directory, years_ots
        )

    return dm_trade_netshare, dm_pack


if __name__ == "__main__":
    years_ots = create_years_list(1990, 2023, 1)
    run(years_ots)
