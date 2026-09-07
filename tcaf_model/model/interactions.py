import copy
import math
import time

from tcaf_model.model.agriculture_module import agriculture
from tcaf_model.model.alcoholic_beverages_module import (
    alcoholic_beverages,
)
from tcaf_model.model.buildings_module import buildings
from tcaf_model.model.common.interface_class import Interface
from tcaf_model.model.crop_module import crop
from tcaf_model.model.dietary_habits_module import dietaryhabits
from tcaf_model.model.forestry_module import forestry

# the land-use entry point is still called crop() in land_use_module
from tcaf_model.model.land_use_module import crop as land_use
from tcaf_model.model.livestock_module import livestock
from tcaf_model.model.population_module import population
from tcaf_model.model.tcaf_module import TCAF

# Order matters: a module reads what the ones before it put on the interface.
# population -> dietary-habits -> alcoholic-beverages -> livestock -> crop
#            -> land-use -> TCAF
TCAF_CHAIN = [
    "population",
    "dietary-habits",
    "alcoholic-beverages",
    "livestock",
    "crop",
    "land-use",
    "TCAF",
]


def runner(lever_setting, years_setting, DM_in, sectors, logger):
    # lever setting dictionary convert float to integer
    lever_setting = {key: math.floor(value) for key, value in lever_setting.items()}

    init_time = time.time()
    TPE = {}
    KPI = {}
    interface = Interface()
    DM_input = copy.deepcopy(DM_in)

    def run(name, fn, *args, **kwargs):
        """Run one module, keep its output, never let one stop the others."""
        if name not in sectors:
            return
        if name not in DM_input:
            logger.warning("No input data for %s, skipped", name)
            return
        start_time = time.time()
        try:
            out = fn(lever_setting, years_setting, DM_input[name], *args, **kwargs)
        except Exception:
            logger.exception("%s failed, skipped", name)
            return
        if isinstance(out, tuple):
            # buildings returns (TPE, KPI)
            TPE[name], KPI[name] = out
        elif out is not None:
            TPE[name] = out
        logger.info(
            "Execution time {0}: {1:.3g} s".format(name, time.time() - start_time)
        )

    run("population", population, interface=interface)
    run(
        "dietary-habits",
        dietaryhabits,
        tpe_scenario="diet-split-kcal",
        write_pickle=False,
        interface=interface,
    )
    run("alcoholic-beverages", alcoholic_beverages, False, interface=interface)
    run("livestock", livestock, False, interface=interface)
    run("crop", crop, False, interface=interface)
    run("land-use", land_use, False, interface=interface)
    run("TCAF", TCAF, interface=interface)

    run("buildings", buildings, interface=interface)
    run("forestry", forestry, interface=interface)
    run("agriculture", agriculture, interface=interface)

    logger.info("Total runtime: {0:.3g} s".format(time.time() - init_time))

    return TPE, KPI
