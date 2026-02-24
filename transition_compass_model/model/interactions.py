from .transport_module import transport
from .lifestyles_module import lifestyles
from .buildings_module import buildings

from .forestry_module import forestry

# from .minerals_module import minerals
from .common.interface_class import Interface
from .agriculture_module import agriculture
from .climate_module import climate

from .ammonia_module import ammonia
from .industry_module import industry
from .energy_module import energy

import math
import copy
import time


def runner(lever_setting, years_setting, DM_in, sectors, logger):
    # lever setting dictionary convert float to integer
    lever_setting = {key: math.floor(value) for key, value in lever_setting.items()}
    country_list = DM_in["lifestyles"]["ots"]["pop"]["lfs_population_"].col_labels[
        "Country"
    ]
    # Transport module

    init_time = time.time()
    TPE = {}
    KPI = {}
    interface = Interface()
    DM_input = copy.deepcopy(DM_in)
    if "climate" in sectors:
        start_time = time.time()
        TPE["climate"] = climate(
            lever_setting, years_setting, DM_input["climate"], interface
        )
        logger.info(
            "Execution time Climate: {0:.3g} s".format(time.time() - start_time)
        )
    if "lifestyles" in sectors:
        start_time = time.time()
        TPE["lifestyles"] = lifestyles(
            lever_setting, years_setting, DM_input["lifestyles"], interface
        )
        logger.info(
            "Execution time Lifestyles: {0:.3g} s".format(time.time() - start_time)
        )
    if "transport" in sectors:
        start_time = time.time()
        TPE["transport"], KPI["transport"] = transport(
            lever_setting, years_setting, DM_input["transport"], interface
        )
        logger.info(
            "Execution time Transport: {0:.3g} s".format(time.time() - start_time)
        )
    if "buildings" in sectors:
        start_time = time.time()
        TPE["buildings"], KPI["buildings"] = buildings(
            lever_setting, years_setting, DM_input["buildings"], interface
        )
        logger.info(
            "Execution time Buildings: {0:.3g} s".format(time.time() - start_time)
        )
    if "industry" in sectors:
        start_time = time.time()
        TPE["industry"] = industry(
            lever_setting, years_setting, DM_input["industry"], interface
        )
        logger.info(
            "Execution time Industry: {0:.3g} s".format(time.time() - start_time)
        )
    if "forestry" in sectors:
        start_time = time.time()
        TPE["forestry"] = forestry(
            lever_setting, years_setting, DM_input["forestry"], interface
        )
        logger.info(
            "Execution time Forestry: {0:.3g} s".format(time.time() - start_time)
        )
    if "agriculture" in sectors:
        start_time = time.time()
        TPE["agriculture"], KPI["agriculture"] = agriculture(
            lever_setting, years_setting, DM_input["agriculture"], interface
        )
        logger.info(
            "Execution time Agriculture: {0:.3g} s".format(time.time() - start_time)
        )
    if "ammonia" in sectors:
        start_time = time.time()
        TPE["ammonia"] = ammonia(
            lever_setting, years_setting, DM_input["ammonia"], interface
        )
        logger.info(
            "Execution time Ammonia: {0:.3g} s".format(time.time() - start_time)
        )

    if "energy" in sectors:
        start_time = time.time()
        TPE["energy"] = energy(lever_setting, years_setting, country_list, interface)
        logger.info("Execution time Energy: {0:.3g} s".format(time.time() - start_time))
    # start_time = time.time()
    # TPE['agriculture'] = agriculture(lever_setting, years_setting, interface)
    # logger.info('Execution time Agriculture: {0:.3g} s'.format(time.time() - start_time))
    # start_time = time.time()
    # TPE['ammonia'] = ammonia(lever_setting, years_setting, interface)
    # logger.info('Execution time Ammonia: {0:.3g} s'.format(time.time() - start_time))
    # start_time = time.time()
    # TPE['power'] = power(lever_setting, years_setting, interface)
    # logger.info('Execution time Power: {0:.3g} s'.format(time.time() - start_time))
    # start_time = time.time()
    # TPE['oil-refinery'] = refinery(lever_setting, years_setting, interface)
    # logger.info('Execution time Oil-refinery: {0:.3g} s'.format(time.time() - start_time))
    # start_time = time.time()
    # TPE['district-heating'] = district_heating(lever_setting, years_setting, interface)
    # logger.info('Execution time District-Heating: {0:.3g} s'.format(time.time() - start_time))
    # start_time = time.time()
    # TPE['land-use'] = land_use(lever_setting, years_setting, interface)
    # logger.info('Execution time Land-use: {0:.3g} s'.format(time.time() - start_time))
    # start_time = time.time()
    # TPE['minerals'], TPE['minerals_EU'] = minerals(interface)
    # logger.info('Execution time Minerals: {0:.3g} s'.format(time.time() - start_time))
    # start_time = time.time()
    # TPE['emissions'] = emissions(lever_setting, years_setting, interface)
    # logger.info('Execution time Emissions: {0:.3g} s'.format(time.time() - start_time))
    # start_time = time.time()

    logger.info("Total runtime: {0:.3g} s".format(time.time() - init_time))

    return TPE, KPI
