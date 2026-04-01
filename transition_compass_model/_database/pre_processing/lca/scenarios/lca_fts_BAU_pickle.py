import os
import numpy as np
import pandas as pd
import pickle
import warnings

warnings.simplefilter("ignore")

from transition_compass_model.model.common.auxiliary_functions import (
    linear_fitting,
    create_years_list,
    my_pickle_dump,
    sort_pickle,
)
from _database.pre_processing.lca.processors.lca_levers import (
    make_footprint_dm,
    make_aggregates_footprint,
)


def load_data(current_file_directory, pattern):

    filepath = os.path.join(
        current_file_directory, f"../data/intermediate_databases/{pattern}.xlsx"
    )
    df = pd.read_excel(filepath)

    return df


def make_dm_fts(
    df, years_fts, scenario, agg_prod_dict=None, agg_mat_dict=None, deepen_n_cat=1
):

    # get fts data
    df_fts = df.loc[df["Scenario_year"].isin([scenario + "_2025"]), :]
    dm_fts = make_footprint_dm(df_fts, years_fts[0], deepen_n_cat=deepen_n_cat)
    df_fts_temp = df.loc[df["Scenario_year"].isin([scenario + "_2050"])]
    dm_fts_temp = make_footprint_dm(
        df_fts_temp, years_fts[-1], deepen_n_cat=deepen_n_cat
    )
    dm_fts.append(dm_fts_temp, "Years")
    dm_fts.sort("Years")

    # group products
    if agg_prod_dict is not None:
        dm_fts.groupby(agg_prod_dict, "Variables", "mean", regex=True, inplace=True)

    # group materials
    if agg_mat_dict is not None:
        dm_fts.groupby(agg_mat_dict, "Categories1", "sum", regex=False, inplace=True)

    # add missing years
    years_missing = list(range(years_fts[1], years_fts[-2], 5))
    dm_fts.add(np.nan, "Years", years_missing, dummy=True)
    dm_fts.sort("Years")
    dm_fts = linear_fitting(dm_fts, years_fts)

    # substitute missing with zeroes so that when flattening / deepening we keep dimensions
    dm_fts.array[np.isnan(dm_fts.array)] = 0

    return dm_fts


def make_fts(
    current_file_directory,
    years_fts,
    DM_lca,
    dict_scenarios,
    variable,
    agg_prod_dict=None,
    agg_mat_dict=None,
):

    # get data
    df_data = load_data(current_file_directory, variable)

    # make storing space
    DM_lca["fts"]["footprint"][variable] = {}

    # make dm fts
    for level in dict_scenarios.keys():
        DM_lca["fts"]["footprint"][variable][level] = make_dm_fts(
            df_data, years_fts, dict_scenarios[level], agg_prod_dict, agg_mat_dict
        )

    return


def run(DM_lca, years_fts):

    # make space to store fts
    DM_lca["fts"] = {}
    DM_lca["fts"]["footprint"] = {}

    # directory
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # get aggregates
    agg_prod_dict, agg_mat_dict = make_aggregates_footprint()

    # scenarios
    dict_scenarios = {
        1: "SSP5-Base",
        2: "SSP2-NPi",
        3: "SSP2-NDC",
        4: "SSP1-PkBudg1150",
    }

    # make fts
    make_fts(
        current_file_directory,
        years_fts,
        DM_lca,
        dict_scenarios,
        "materials",
        agg_prod_dict,
        agg_mat_dict,
    )
    make_fts(
        current_file_directory,
        years_fts,
        DM_lca,
        dict_scenarios,
        "energy-demand-elec",
        agg_prod_dict,
    )
    make_fts(
        current_file_directory,
        years_fts,
        DM_lca,
        dict_scenarios,
        "energy-demand-ff",
        agg_prod_dict,
    )
    make_fts(
        current_file_directory,
        years_fts,
        DM_lca,
        dict_scenarios,
        "ecological",
        agg_prod_dict,
    )
    make_fts(
        current_file_directory, years_fts, DM_lca, dict_scenarios, "gwp", agg_prod_dict
    )
    make_fts(
        current_file_directory,
        years_fts,
        DM_lca,
        dict_scenarios,
        "water",
        agg_prod_dict,
    )
    make_fts(
        current_file_directory,
        years_fts,
        DM_lca,
        dict_scenarios,
        "air-pollutant",
        agg_prod_dict,
    )
    make_fts(
        current_file_directory,
        years_fts,
        DM_lca,
        dict_scenarios,
        "heavy-metals",
        agg_prod_dict,
    )

    # # check
    # # Load existing DM_lca
    pickle_file = os.path.join(
        current_file_directory, "../../../data/datamatrix/lca.pickle"
    )
    # with open(pickle_file, 'rb') as handle:
    #   DM_lca_current = pickle.load(handle)
    # variable = "LDV_BEV"
    # def myreshape(DM, variable, level):
    #     dm_temp = DM["footprint"]["materials"][level].filter({"Country" : ["Switzerland"], "Variables" : [variable], "Categories1" : ["aluminium"]})
    #     dm_temp.rename_col_regex("_","-", "Variables")
    #     dm_temp.rename_col(dm_temp.col_labels["Variables"][0],str(level) + "_" + dm_temp.col_labels["Variables"][0], "Variables")
    #     dm_temp.deepen(based_on = "Variables", sep = "_")
    #     dm_temp.switch_categories_order("Categories1","Categories2")
    #     return dm_temp
    # dm_temp = myreshape(DM_lca_current["fts"], variable, 1)
    # for i in range(2,4+1): dm_temp.append(myreshape(DM_lca_current["fts"], variable, i), "Variables")
    # dm_temp.flatten().datamatrix_plot()
    # dm_temp = myreshape(DM_lca["fts"], variable, 1)
    # for i in range(2,4+1): dm_temp.append(myreshape(DM_lca["fts"], variable, i), "Variables")
    # dm_temp.flatten().datamatrix_plot()

    # save
    my_pickle_dump(DM_new=DM_lca, local_pickle_file=pickle_file)
    sort_pickle(pickle_file)

    return


if __name__ == "__main__":

    # get country ots fts
    years_fts = create_years_list(2025, 2050, 5)

    # load pickle lca ots
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_file_directory, "../data/datamatrix/lca_ots.pickle")
    if not os.path.exists(filepath):
        raise FileNotFoundError("You need to run ots_pickle_run() first")
    with open(filepath, "rb") as f:
        DM_lca = pickle.load(f)

    run(DM_lca, years_fts)
