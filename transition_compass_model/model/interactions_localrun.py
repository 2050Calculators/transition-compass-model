import logging

from transition_compass_model.model.common.auxiliary_functions import (
    filter_country_and_load_data_from_pickles,
)
from transition_compass_model.model.common.config_loader import load_lever_config
from transition_compass_model.model.interactions import runner


def local_interactions_run():
    # Configure logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    # Configures initial input for model run
    lever_setting = load_lever_config()
    years_setting = [1990, 2023, 2025, 2050, 5]

    # country_list = ["Switzerland","EU27","Vaud"]
    country_list = ["Schwyz"]

    sectors = [
        "climate",
        "lifestyles",
        "transport",
        "buildings",
        "industry",
        "agriculture",
        "ammonia",
        # 'landuse',
        "energy",
        "emissions",
        "forestry",
        "lca",
    ]
    # Filter geoscale
    # from database/data/datamatrix/.* reads the pickles, filters the geoscale, and loads them
    DM_input = filter_country_and_load_data_from_pickles(
        country_list=country_list, modules_list=sectors
    )

    # run
    output, KPI = runner(
        lever_setting,
        years_setting,
        DM_input,
        sectors,
        logger,
    )

    # return
    return output


# run local
results_run = local_interactions_run()

# # checks
# import pprint
# pprint.pprint(results_run["emissions"].columns.tolist())
# import pandas as pd
# df = results_run["emissions"].filter(['Country', 'Years', 'fos_emissions-CO2e[Mt]', 'ind_emissions-CO2e[Mt]'])
# df = pd.melt(df, id_vars = ["Country","Years"])
# df = df.groupby(["variable","Years",])['value'].agg(sum)
