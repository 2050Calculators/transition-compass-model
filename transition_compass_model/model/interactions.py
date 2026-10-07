import copy
import math
import time

# from model.district_heating_module import district_heating
from transition_compass_model.model.agriculture_module import agriculture
from transition_compass_model.model.ammonia_module import ammonia
from transition_compass_model.model.buildings_module import buildings
from transition_compass_model.model.climate_module import climate

# from model.minerals_module import minerals
from transition_compass_model.model.common.interface_class import Interface
from transition_compass_model.model.emissions_module import emissions
from transition_compass_model.model.energy_module import energy
from transition_compass_model.model.forestry_module import forestry
from transition_compass_model.model.industry_module import industry

# from model.landuse_module import land_use
# from model.oilrefinery_module import refinery
from transition_compass_model.model.lca_module import lca
from transition_compass_model.model.lifestyles_module import lifestyles
from transition_compass_model.model.transport_module import transport

# Cross-sector totals shown on the Overall page. Each list is exactly the set of
# bands stacked in the corresponding Overall chart, so the KPI card always equals
# the top of its stack. Buildings and transport only: agriculture and industry are
# not part of the Overall view yet.
OVERALL_KPI_DEFINITIONS = [
    (
        "Total emissions (all sectors)",
        "Mt CO₂eq",
        [
            ("buildings", "bld_emissions-CO2e"),
            ("transport", "tra_emissions-CO2e_passenger-land"),
            ("transport", "tra_emissions-CO2e_freight"),
            ("transport", "tra_emissions-CO2e_passenger_aviation"),
        ],
    ),
    (
        "Total energy demand (all sectors)",
        "TWh",
        [
            ("buildings", "bld_energy-demand_total"),
            ("transport", "tra_energy-demand_passenger-land"),
            ("transport", "tra_energy-demand_freight"),
            ("transport", "tra_passenger_energy-demand-by-mode_aviation"),
        ],
    ),
]


def compute_overall_KPI(TPE, years_setting):
    """Cross-sector KPI cards for the Overall page.

    Returns [] unless every sector involved has been run, so that partial runs
    (a single sector requested by the frontend) do not raise.

    The gauge bounds are derived from the last historical year rather than being
    hard-coded: Vaud and Switzerland differ by roughly a factor ten, and the
    historical value is unaffected by lever settings, so the scale stays put while
    the needle moves. The headroom above the historical value is deliberate --
    under the default lever positions the 2050 total is *above* the historical one,
    driven by aviation growth, so a scale that stopped at the base year would peg.

    The warning/danger fractions are placeholders, not policy targets.
    """
    required = {
        sector for _, _, terms in OVERALL_KPI_DEFINITIONS for sector, _ in terms
    }
    if not required.issubset(TPE.keys()):
        return []

    base_year = years_setting[1]
    end_year = years_setting[3]

    KPI_overall = []
    for title, unit, terms in OVERALL_KPI_DEFINITIONS:
        try:
            value = sum(TPE[sector][0, end_year, var] for sector, var in terms)
            base = sum(TPE[sector][0, base_year, var] for sector, var in terms)
        except KeyError:
            continue
        KPI_overall.append(
            {
                "title": title,
                "value": value,
                "unit": unit,
                "min": 0,
                "max": 1.5 * base,
                "warning": 0.5 * base,
                "danger": 0.8 * base,
            }
        )

    return KPI_overall


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
        TPE["industry"], KPI["industry"] = industry(
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
    if "emissions" in sectors:
        start_time = time.time()
        TPE["emissions"] = emissions(years_setting, interface)
        logger.info(
            "Execution time Emissions: {0:.3g} s".format(time.time() - start_time)
        )
    if "lca" in sectors:
        start_time = time.time()
        TPE["lca"] = lca(lever_setting, years_setting, DM_input["lca"], interface)
        logger.info("Execution time LCA: {0:.3g} s".format(time.time() - start_time))

    # start_time = time.time()
    # TPE['agriculture'] = agriculture(lever_setting, years_setting, interface)
    # logger.info('Execution time Agriculture: {0:.3g} s'.format(time.time() - start_time))
    # start_time = time.time()
    # TPE['ammonia'] = ammonia(lever_setting, years_setting, interface)
    # logger.info('Execution time Ammonia: {0:.3g} s'.format(time.time() - start_time))
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

    KPI_overall = compute_overall_KPI(TPE, years_setting)
    if KPI_overall:
        KPI["overall"] = KPI_overall

    logger.info("Total runtime: {0:.3g} s".format(time.time() - init_time))

    return TPE, KPI
