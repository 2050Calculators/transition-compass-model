from model.population_module import population
from model.buildings_module import buildings

from model.forestry_module import forestry
# from model.minerals_module import minerals
from model.common.interface_class import Interface
from model.agriculture_module import agriculture

from model.landuse_module import land_use


import math
import copy
import time
import os
import json


def runner(lever_setting, years_setting, DM_in, sectors, logger):
    # lever setting dictionary convert float to integer
    lever_setting = {key: math.floor(value) for key, value in lever_setting.items()}
    # Transport module

    init_time = time.time()
    TPE = {}
    KPI = {}
    interface = Interface()
    DM_input = copy.deepcopy(DM_in)
    if 'lifestyles' in sectors:
      start_time = time.time()
      TPE["lifestyles"] = lifestyles(lever_setting, years_setting, DM_input['lifestyles'], interface)
      logger.info("Execution time Lifestyles: {0:.3g} s".format(time.time() - start_time))
    if 'buildings' in sectors:
      start_time = time.time()
      TPE['buildings'], KPI['buildings'] = buildings(lever_setting, years_setting, DM_input['buildings'], interface)
      logger.info('Execution time Buildings: {0:.3g} s'.format(time.time() - start_time))
    if 'forestry' in sectors:
      start_time = time.time()
      TPE['forestry'] = forestry(lever_setting, years_setting, DM_input['forestry'], interface)
      logger.info('Execution time Forestry: {0:.3g} s'.format(time.time() - start_time))
    if 'agriculture' in sectors:
      start_time = time.time()
      TPE['agriculture'] = agriculture(lever_setting, years_setting, DM_input['agriculture'], interface)
      logger.info('Execution time Agriculture: {0:.3g} s'.format(time.time() - start_time))

    logger.info("Total runtime: {0:.3g} s".format(time.time() - init_time))

    return TPE, KPI

