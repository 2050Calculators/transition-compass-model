
from model.climate_module import climate
from model.transport_module import transport
from model.lifestyles_module import lifestyles
from model.buildings_module import buildings
from model.industry_module import industry
from model.agriculture_module import agriculture
from model.forestry_module import forestry
from model.common.interface_class import Interface
from model.common.auxiliary_functions import filter_geoscale
# from model.district_heating_module import district_heating
# from model.agriculture_module import agriculture
# from model.emissions_module import emissions
# from model.ammonia_module import ammonia
# from model.power_module import power
# from model.landuse_module import land_use
# from model.oilrefinery_module import refinery
# from model.minerals_module import minerals

import math
import time
import os
import json



def interactions(lever_setting, years_setting):

    # create dictionary for storing run time
    runtime = {}    

    init_time = time.time()
    TPE = {}
    interface = Interface()
    TPE['climate'] = climate(lever_setting, years_setting, interface)
    start_time = time.time()
    TPE['lifestyles'] = lifestyles(lever_setting, years_setting, interface)
    runtime['Execution time Lifestyles'] = time.time() - start_time
    start_time = time.time()
    TPE['transport'] = transport(lever_setting, years_setting, interface)
    runtime['Execution time Transport'] = time.time() - start_time
    start_time = time.time()
    TPE['buildings'] = buildings(lever_setting, years_setting, interface)
    runtime['Execution time Buldings'] = time.time() - start_time
    start_time = time.time()
    TPE['industry'] = industry(lever_setting, years_setting, interface)
    runtime['Execution time Industry'] = time.time() - start_time
    start_time = time.time()
    TPE['forestry'] = forestry(lever_setting, years_setting, interface)
    runtime['Execution time Forestry'] = time.time() - start_time
    start_time = time.time()
    TPE['agriculture'] = agriculture(lever_setting, years_setting, interface)
    runtime['Execution time Agriculture'] = time.time() - start_time
    start_time = time.time()
    #TPE['ammonia'] = ammonia(lever_setting, years_setting, interface)
    #runtime['Execution time Ammonia'] = time.time() - start_time
    #start_time = time.time()
    #TPE['power'] = power(lever_setting, years_setting, interface)
    #runtime['Execution time Power'] = time.time() - start_time
    #start_time = time.time()
    #TPE['oil-refinery'] = refinery(lever_setting, years_setting, interface)
    #runtime['Execution time Oil-Refinery'] = time.time() - start_time
    #start_time = time.time()
    #TPE['district-heating'] = district_heating(lever_setting, years_setting, interface)
    #runtime['Execution time District-Heating'] = time.time() - start_time
    #start_time = time.time()
    #TPE['land-use'] = land_use(lever_setting, years_setting, interface)
    #runtime['Execution time Land-Use'] = time.time() - start_time
    #start_time = time.time()
    #TPE['minerals'], TPE['minerals_EU'] = minerals(interface)
    #runtime['Execution time Minerals'] = time.time() - start_time
    #start_time = time.time()
    #TPE['emissions'] = emissions(lever_setting, years_setting, interface)
    #runtime['Execution time Emissions'] = time.time() - start_time
    start_time = time.time()
    
    runtime['Execution time'] = time.time() - init_time

    return TPE, runtime

def local_interactions_run():
    
    # get lever setting
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(current_file_directory, '../config/lever_position.json'))
    lever_setting = json.load(f)[0]
    lever_setting = {key: math.floor(value) for key, value in lever_setting.items()}
    
    # get years
    years_setting = [1990, 2023, 2025, 2050, 5]
    
    # geoscale
    global_vars = {'geoscale': 'EU27|Switzerland|Vaud'}
    filter_geoscale(global_vars['geoscale'])

    # run
    results_run, runtime = interactions(lever_setting, years_setting)
    
    # return
    return results_run, runtime

# run local
results_run, runtime = local_interactions_run()

# # checks
# import pprint
# pprint.pprint(results_run["emissions"].columns.tolist())
# import pandas as pd
# df = results_run["emissions"].filter(['Country', 'Years', 'fos_emissions-CO2e[Mt]', 'ind_emissions-CO2e[Mt]'])
# df = pd.melt(df, id_vars = ["Country","Years"])
# df = df.groupby(["variable","Years",])['value'].agg(sum)

