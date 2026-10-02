# Transport module — accounting basis
#
# Accounting basis: RESIDENCY-BASED (not territorial, not consumption-based).
#   Passenger: Swiss residents' trips wherever they occur (source: MTMC survey).
#   Freight road HDV: Swiss-registered vehicles only (source: BFS GTS IMMATRICULATION=="CH").
#   Freight rail/IWW: territorial ≈ residency in practice (no registration split available).
#
# Emission scope: SCOPE 1 ONLY — direct combustion emissions from fuel burned in engines.
#   EVs appear as zero direct emissions here; their electricity consumption is forwarded
#   to the energy module, where generation-side emissions are handled (scope 2 for transport
#   would mean attributing those electricity emissions back to transport — not done here).
#   Scope 3 (vehicle manufacturing, upstream fuel, infrastructure) is handled by the
#   industry and LCA modules via the industry interface.

import os
import pickle

import numpy as np

from transition_compass_model.model.common.auxiliary_functions import my_pickle_dump
from transition_compass_model.model.transport.workflows import (
    convert_to_cO2eq_emissions,
)


def tra_industry_interface(
    dm_freight_veh, dm_passenger_veh, dm_infrastructure, write_pickle=False
):
    if "aviation" not in dm_passenger_veh.col_labels["Categories1"]:
        dm_passenger_veh.add(
            np.nan, dim="Categories1", col_label="aviation", dummy=True
        )
        dm_passenger_veh.add(np.nan, dim="Categories2", col_label="ICE", dummy=True)
    dm_veh = dm_passenger_veh.copy()
    dm_veh.groupby({"CEV": ["mt", "CEV"]}, "Categories2", inplace=True)
    dm_veh.groupby({"trains": ["metrotram", "rail"]}, "Categories1", inplace=True)
    dm_veh = dm_veh.filter_w_regex({"Categories1": "LDV|aviation|bus|trains"})
    dm_veh.rename_col(["aviation"], ["planes"], "Categories1")
    dm_veh.rename_col(
        [
            "tra_passenger_new-vehicles",
            "tra_passenger_vehicle-waste",
            "tra_passenger_vehicle-fleet",
        ],
        ["tra_product-demand", "tra_product-waste", "tra_product-stock"],
        dim="Variables",
    )

    # freight
    dm_fre = dm_freight_veh.copy()
    dm_fre.groupby({"HDV": "HDV"}, "Categories1", regex=True, inplace=True)
    dm_fre = dm_fre.filter_w_regex({"Categories1": "HDV|aviation|marine|rail"})
    dm_fre.rename_col("marine", "ships", "Categories1")
    dm_fre.rename_col(
        [
            "tra_freight_new-vehicles",
            "tra_freight_vehicle-waste",
            "tra_freight_vehicle-fleet",
        ],
        ["tra_product-demand", "tra_product-waste", "tra_product-stock"],
        dim="Variables",
    )

    # put together
    cat_missing = list(
        set(dm_veh.col_labels["Categories2"]) - set(dm_fre.col_labels["Categories2"])
    )
    dm_veh.add(0, dummy=True, dim="Categories2", col_label=["ICE"])
    # dm_fre[...] = 0
    dm_veh.append(dm_fre, "Categories1")
    # Rename kerosene and H2 as ICE
    dm_veh.rename_col("ICE", "ICE_old", "Categories2")
    dm_veh.groupby(
        {"ICE": ["ICE_old", "H2", "kerosene"]}, dim="Categories2", inplace=True
    )
    dm_veh.groupby({"trains": ["trains", "rail"]}, "Categories1", inplace=True)
    dm_veh.groupby({"planes": ["planes", "aviation"]}, "Categories1", inplace=True)
    dm_veh.sort("Categories1")

    # get infrastructure
    dm_infra_ind = dm_infrastructure.copy()
    dm_infra_ind.rename_col_regex("infra-", "", dim="Categories1")
    dm_infra_ind.rename_col(
        [
            "tra_infrastructure_waste",
            "tra_new_infrastructure",
            "tra_tot-infrastructure",
        ],
        ["tra_product-waste", "tra_product-demand", "tra_product-stock"],
        dim="Variables",
    )

    # fix years in dm_veh
    # TODO: to remove this when we fix it in pre processing
    years = dm_veh.col_labels["Years"].copy()
    for y in years:
        arr_temp = dm_veh[:, y, ...]
        dm_veh.drop("Years", int(y))
        dm_veh.add(arr_temp, "Years", [int(y)])
    dm_veh.sort("Years")

    # ! FIXME add infrastructure in km
    DM_industry = {
        "tra-veh": dm_veh.filter({"Variables": ["tra_product-demand"]}),
        "tra-infra": dm_infra_ind,
        "tra-waste": dm_veh.filter({"Variables": ["tra_product-waste"]}),
        "tra-stock": dm_veh.filter({"Variables": ["tra_product-stock"]}),
    }

    # if write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../../_database/data/interface/transport_to_industry.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(DM_industry, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return DM_industry


def tra_minerals_interface(
    dm_freight_new_veh,
    dm_passenger_new_veh,
    DM_industry,
    dm_infrastructure,
    write_pickle=False,
):
    # Group technologies as PHEV, ICE, EV and FCEV
    dm_freight_new_veh.groupby(
        {"PHEV": "PHEV.*", "ICE": "ICE.*", "EV": "BEV|CEV"},
        regex=True,
        inplace=True,
        dim="Categories2",
    )
    # note that mt is later dropped
    dm_passenger_new_veh.groupby(
        {"PHEV": "PHEV.*", "ICE": "ICE.*", "EV": "BEV|CEV|mt"},
        regex=True,
        inplace=True,
        dim="Categories2",
    )
    # keep only certain vehicles
    keep_veh = "HDV.*|2W|LDV|bus"
    dm_keep_new_veh = dm_passenger_new_veh.filter_w_regex({"Categories1": keep_veh})
    dm_keep_new_veh.rename_col(
        "tra_new-vehicles", "tra_product-demand", dim="Variables"
    )
    dm_keep_freight_new_veh = dm_freight_new_veh.filter_w_regex(
        {"Categories1": keep_veh}
    )
    dm_keep_freight_new_veh.rename_col(
        "tra_new-vehicles", "tra_product-demand", dim="Variables"
    )
    # join passenger and freight

    dm_keep_new_veh.append(dm_keep_freight_new_veh, dim="Categories1")
    # flatten to obtain e.g. LDV-EV or HDVL-FCEV
    dm_keep_new_veh = dm_keep_new_veh.flatten()
    dm_keep_new_veh.rename_col_regex("_", "-", "Categories1")

    dm_other = DM_industry["tra-veh"].filter(
        {"Categories1": ["planes", "ships", "trains"]}
    )
    dm_other.groupby(
        {
            "other-planes": ["planes"],
            "other-ships": ["ships"],
            "other-trains": ["trains"],
        },
        dim="Categories1",
        inplace=True,
    )

    dm_keep_new_veh.append(dm_other, dim="Categories1")
    dm_keep_new_veh.rename_col("tra_product-demand", "product-demand", dim="Variables")

    DM_minerals = {"tra_veh": dm_keep_new_veh, "tra_infra": dm_infrastructure}

    # if write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../../_database/data/interface/transport_to_minerals.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(DM_minerals, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return DM_minerals


def tra_oilrefinery_interface(dm_pass_energy, dm_freight_energy, write_pickle=False):
    cat_missing = ["kerosene", "marinefueloil"]
    for cat in cat_missing:
        if cat not in dm_pass_energy.col_labels["Categories1"]:
            dm_pass_energy.add(0, dummy=True, col_label=cat, dim="Categories1")
    dm_pass_energy.append(dm_freight_energy, dim="Variables")
    dm_tot_energy = dm_pass_energy.groupby(
        {"tra_energy-demand": ".*"}, dim="Variables", inplace=False, regex=True
    )
    dict_rename = {
        "diesel": "liquid-ff-diesel",
        "marinefueloil": "liquid-ff-fuel-oil",
        "gasoline": "liquid-ff-gasoline",
        "gas": "gas-ff-natural",
        "kerosene": "liquid-ff-kerosene",
    }
    for str_old, str_new in dict_rename.items():
        dm_tot_energy.rename_col(str_old, str_new, dim="Categories1")

    # if write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../../_database/data/interface/transport_to_oil-refinery.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(dm_tot_energy, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm_tot_energy


def prepare_KPIs(dm_tpe, DM_kpi_dict, years_setting):
    """Three whole-sector KPI cards for the Transport page.

    The two totals are summed from the *-main-mode_* series, whose eight categories
    partition all of transport (passenger: aviation, LDV, 2W, bus, rail, metrotram;
    freight: aviation, HDV, IWW, marine, rail). Summing exactly the series the charts
    stack means each card equals the top of its stack.

    Gauge bounds are derived from the last historical year rather than hardcoded per
    region: that value is unaffected by lever settings, so the scale stays put while
    the needle moves, and it works for both a canton and the country (~10x apart).
    """
    base_year = years_setting[1]
    end_year = years_setting[3]
    cntr = dm_tpe.col_labels["Country"][0]
    KPI = []

    # The multiplier is fitted per sector, not shared: transport roughly doubles by
    # 2050 under default levers (aviation demand grows ~3.8x), so the max has to clear
    # 38.4 Mt / 152.2 TWh for Switzerland. compute_overall_KPI uses 1.5 for the same
    # reasoning because declining buildings emissions dampen the growth there.
    def _scaled(title, unit, prefix):
        series = [v for v in dm_tpe.col_labels["Variables"] if v.startswith(prefix)]
        value = sum(float(dm_tpe[cntr, end_year, v]) for v in series)
        base = sum(float(dm_tpe[cntr, base_year, v]) for v in series)
        return {
            "title": title,
            "value": value,
            "unit": unit,
            "min": 0,
            "max": 2.0 * base,
            "warning": 0.5 * base,
            "danger": 0.8 * base,
        }

    KPI.append(
        _scaled(
            "Total transport emissions",
            "Mt CO\u2082eq",
            "tra_emissions-CO2e-main-mode_",
        )
    )
    KPI.append(
        _scaled("Total transport energy demand", "TWh", "tra_energy-demand-main-mode_")
    )

    # Share of electric cars in the LDV stock. Bounds are policy levels on a 0-100
    # scale, not scaled history: the share starts near zero and should grow, so
    # deriving a max from the base year would peg the gauge immediately.
    dm_LDV_EV = DM_kpi_dict["stock_EV"]
    value = (
        dm_LDV_EV[0, end_year, "tra_passenger_technology-share-fleet", "LDV", "BEV"]
        * 100
    )
    KPI.append(
        {
            "title": "Electric car share",
            "value": value,
            "unit": "%",
            "min": 0,
            "max": 100,
            "warning": 90,
            "danger": 50,
        }
    )

    return KPI


def compute_aviation_emission_variants(DM_passenger_out):
    """Return DM with outbound and round-trip passenger aviation emissions.

    Variables:
      tra_passenger-emissions-total  — full round-trip (MTMC-based)
      tra_passenger-emissions-local  — outbound only (×share-local)
    Categories1: aviation (collapse with group_all before combining with other modes).

    The share-local factor (~0.35 for CH in 2019) is defined as:
      skm_CH / skm_monde = (pkm_suisse / occ_suisse) / (pkm_monde / occ_monde)
    where pkm_suisse is territorial (flights departing from Swiss airports) and
    pkm_monde is MTMC residency-based (Swiss residents' full flight chains).
    The ratio is ~0.35, not ~0.5, because MTMC captures all legs including
    connections through foreign hubs, while pkm_suisse counts only the first
    Swiss-departing segment. So outbound ≈ Swiss territorial inventory;
    round-trip ≈ 3× outbound (round-trip factor × connecting-flight factor).
    """
    dm = DM_passenger_out["aviation"]["emissions"].copy()
    dm = dm.group_all("Categories2", inplace=False)
    dm.group_all("Categories2", inplace=True)
    dm.append(DM_passenger_out["aviation-share-local"], dim="Variables")
    dm.operation(
        "tra_passenger_emissions",
        "*",
        "tra_share-emissions-local",
        out_col="tra_passenger-emissions-local",
        unit="Mt",
    )
    dm.rename_col(
        "tra_passenger_emissions", "tra_passenger-emissions-total", dim="Variables"
    )
    dm.drop(dim="Variables", col_label="tra_share-emissions-local")
    return dm


# Fuel labels used by passenger and freight, grouped into the families shown
# in the whole-transport "by fuel" charts. Labels not listed keep their name.
FUEL_FAMILIES = {
    "biofuels": [
        "biodiesel",
        "biogas",
        "biogasoline",
        "kerosenebio",
        "biojetfuel",
        "biomarinefueloil",
    ],
    "efuels": ["efuel", "ejetfuel", "emarinefueloil"],
    "marine-fuel-oil": ["marinefueloil"],
    "hydrogen": ["hydrogen", "H2"],
}


def _group_fuel_families(dm):
    labels = dm.col_labels["Categories1"]
    groups = {}
    for family, fuels in FUEL_FAMILIES.items():
        present = [f for f in fuels if f in labels]
        if present:
            groups[family] = present
    dm.groupby(groups, dim="Categories1", inplace=True)
    return dm


def _sum_by_fuel(dm_list, variable):
    # Sum datamatrices (Categories1 = fuel) whose fuel lists may differ
    fuels = sorted(set().union(*[dm.col_labels["Categories1"] for dm in dm_list]))
    dm_out = None
    for dm in dm_list:
        dm.rename_col(dm.col_labels["Variables"][0], variable, dim="Variables")
        missing = [f for f in fuels if f not in dm.col_labels["Categories1"]]
        if missing:
            dm.add(0, dummy=True, dim="Categories1", col_label=missing)
        dm.sort("Categories1")
        if dm_out is None:
            dm_out = dm
        else:
            dm_out.array = np.nansum([dm_out.array, dm.array], axis=0)
    return dm_out


def compute_transport_by_fuel(DM_passenger_out, DM_freight_out):
    """Passenger + freight energy demand [TWh] and CO2e emissions [Mt] by fuel family.

    Energy covers all modes, aviation included. Emissions are direct (scope 1),
    so electricity does not appear; they add up to the by-mode CO2e emissions.
    """
    dm_pass_land = DM_passenger_out["energy"].copy()
    dm_pass_aviation = DM_passenger_out["aviation"]["energy"].copy()
    dm_pass_aviation.group_all("Categories1", inplace=True)
    dm_energy = _sum_by_fuel(
        [
            _group_fuel_families(dm_pass_land),
            _group_fuel_families(dm_pass_aviation),
            _group_fuel_families(DM_freight_out["energy"].copy()),
        ],
        variable="tra_energy-demand-by-fuel",
    )

    dm_emi_pass = convert_to_cO2eq_emissions(
        DM_passenger_out["emissions_by_fuel"].copy()
    )
    dm_emi_freight = convert_to_cO2eq_emissions(
        DM_freight_out["emissions_by_fuel"].copy()
    )
    dm_emissions = _sum_by_fuel(
        [_group_fuel_families(dm_emi_pass), _group_fuel_families(dm_emi_freight)],
        variable="tra_emissions-CO2e-by-fuel",
    )

    return dm_energy, dm_emissions


# Main categories shown in the primary transport charts, as
# {main category: [detailed categories]}. The detailed series stay in the output.
MAIN_MODES = {
    "aviation-passenger": [("passenger", "aviation")],
    "aviation-freight": [("freight", "aviation")],
    "cars": [("passenger", "LDV")],
    "trucks": [("freight", "HDV")],
    "other-passenger": [("passenger", "2W"), ("passenger", "bus")],
    "other-freight": [("freight", "IWW"), ("freight", "marine")],
    "rail-freight": [("freight", "rail")],
    "rail-passenger": [("passenger", "rail"), ("passenger", "metrotram")],
}
MAIN_FUELS = {
    "kerosene": ["kerosene"],
    "diesel": ["diesel"],
    "gasoline": ["gasoline"],
    "other-fossil": ["gas", "marine-fuel-oil"],
    "other-renewable": ["biofuels", "efuels", "hydrogen", "SAF"],
    "electricity": ["electricity"],
}


def group_main_categories(dm_tpe):
    """Sum the detailed by-mode and by-fuel transport variables of the flat TPE
    datamatrix into the main categories of MAIN_MODES and MAIN_FUELS."""
    by_mode = {
        "tra_emissions-CO2e-main-mode": {
            "passenger": "tra_emissions-CO2e_passenger_",
            "freight": "tra_emissions-CO2e_freight_",
        },
        "tra_energy-demand-main-mode": {
            "passenger": "tra_passenger_energy-demand-by-mode_",
            "freight": "tra_freight_energy-demand-by-mode_",
        },
    }
    by_fuel = {
        "tra_emissions-CO2e-main-fuel": "tra_emissions-CO2e-by-fuel_",
        "tra_energy-demand-main-fuel": "tra_energy-demand-by-fuel_",
    }
    variables = dm_tpe.col_labels["Variables"]
    groups = {}
    for out_var, prefix in by_mode.items():
        for main, members in MAIN_MODES.items():
            groups[f"{out_var}_{main}"] = [
                prefix[kind] + mode for kind, mode in members
            ]
    for out_var, prefix in by_fuel.items():
        for main, members in MAIN_FUELS.items():
            # Not every fuel exists for both (e.g. no electricity in emissions)
            present = [prefix + f for f in members if prefix + f in variables]
            if present:
                groups[f"{out_var}_{main}"] = present
    return dm_tpe.groupby(groups, dim="Variables", inplace=False)


def prepare_TPE_output(
    DM_passenger_out, DM_freight_out, dm_aviation_local, years_setting
):
    # Whole-transport (passenger + freight) totals by fuel family. Built first,
    # because the aviation energy below is regrouped in place.
    dm_energy_by_fuel, dm_emissions_by_fuel = compute_transport_by_fuel(
        DM_passenger_out, DM_freight_out
    )

    # Aviation Energy-demand
    dm_keep_aviation_energy = DM_passenger_out["aviation"]["energy"]
    dm_keep_aviation_energy.groupby(
        {"SAF": "kerosenebio|keroseneefuel"},
        dim="Categories2",
        regex=True,
        inplace=True,
    )
    dm_keep_aviation_energy.filter(
        {"Categories2": ["kerosene", "SAF", "hydrogen", "electricity"]}, inplace=True
    )

    dm_keep_aviation_emissions = DM_passenger_out["aviation"]["emissions"].copy()
    dm_keep_aviation_local = dm_aviation_local

    dm_keep_mode = DM_passenger_out["mode"].filter(
        {
            "Variables": [
                "tra_passenger_transport-demand-by-mode",
                "tra_passenger_energy-demand-by-mode",
                "tra_passenger_vehicle-fleet",
                "tra_passenger_new-vehicles",
                "tra_passenger_transport-demand-vkm",
            ]
        }
    )
    dm_keep_mode.change_unit(
        "tra_passenger_transport-demand-by-mode",
        old_unit="pkm",
        new_unit="Bpkm",
        factor=1e-9,
    )
    dm_keep_mode.change_unit(
        "tra_passenger_transport-demand-vkm",
        old_unit="vkm",
        new_unit="Bvkm",
        factor=1e-9,
    )
    dm_keep_mode.change_unit(
        "tra_passenger_vehicle-fleet",
        old_unit="number",
        new_unit="millions",
        factor=1e-6,
    )

    dm_keep_tech = DM_passenger_out["tech"].filter(
        {"Variables": ["tra_passenger_vehicle-fleet"], "Categories1": ["LDV"]}
    )
    # Same conversion as the mode-level fleet above, so the per-technology series is
    # on the same scale as its sibling rather than raw vehicle counts.
    dm_keep_tech.change_unit(
        "tra_passenger_vehicle-fleet",
        old_unit="number",
        new_unit="millions",
        factor=1e-6,
    )

    dm_keep_fuel = DM_passenger_out["fuel"]

    dm_keep_energy = DM_passenger_out["energy"].copy()
    dm_keep_energy.drop(dim="Categories1", col_label=["efuel"])

    dm_freight_energy_by_mode = DM_freight_out["mode"].filter(
        {"Variables": ["tra_freight_energy-demand"]}
    )
    dm_freight_energy_by_mode.rename_col(
        "tra_freight_energy-demand",
        "tra_freight_energy-demand-by-mode",
        dim="Variables",
    )
    dm_freight_energy_by_mode.groupby(
        {"HDV": "HDV.*"}, dim="Categories1", inplace=True, regex=True
    )

    dm_freight_energy_by_fuel = DM_freight_out["energy"].copy()
    dm_freight_energy_by_fuel.drop(dim="Categories1", col_label=["efuel", "ejetfuel"])
    dm_freight_energy_by_fuel.rename_col(
        "tra_freight_total-energy", "tra_freight_energy-demand-by-fuel", dim="Variables"
    )

    dm_tech_HDVH = DM_freight_out["tech"].filter(
        {"Variables": ["tra_freight_technology-share-fleet"], "Categories1": ["HDVH"]}
    )
    dm_tech_HDVH.array = dm_tech_HDVH.array * 100

    dm_freight_emissions = DM_freight_out["emissions"].filter({"Categories2": ["CO2"]})
    dm_freight_emissions.groupby(
        {"HDV": ["HDVH", "HDVM", "HDVL"]}, dim="Categories1", inplace=True
    )

    dm_soft_mobility = DM_passenger_out["soft-mobility"]
    dm_soft_mobility.change_unit(
        "tra_passenger_transport-demand-by-mode",
        old_unit="pkm",
        new_unit="Bpkm",
        factor=1e-9,
    )

    # Transport totals for the cross-sector (Overall) charts: passenger land,
    # freight, and passenger aviation kept apart. Passenger aviation is already
    # published per mode, so only the two land/freight aggregates are built here.
    # Freight aviation stays inside freight: it is territorial, whereas passenger
    # aviation is the residency-based round-trip figure (see emissions/workflows.py).
    land_modes = ["2W", "LDV", "bus", "metrotram", "rail"]

    dm_emi_pass_land = DM_passenger_out["emissions"].filter({"Categories1": land_modes})
    dm_emi_pass_land.group_all("Categories1", inplace=True)
    dm_emi_pass_land.rename_col(
        "tra_emissions-CO2e_passenger",
        "tra_emissions-CO2e_passenger-land",
        dim="Variables",
    )

    # Freight emissions still carry the gas dimension here, unlike passenger which
    # transport_module has already converted, so apply the same GWP helper.
    dm_emi_freight = convert_to_cO2eq_emissions(DM_freight_out["emissions"].copy())
    dm_emi_freight.groupby(
        {"HDV": ["HDVH", "HDVM", "HDVL"]}, dim="Categories1", inplace=True
    )
    dm_emi_freight.rename_col(
        "tra_freight_emissions", "tra_emissions-CO2e_freight", dim="Variables"
    )
    # Per mode, for the combined passenger + freight emissions chart
    dm_emi_freight_by_mode = dm_emi_freight.copy()
    dm_emi_freight.group_all("Categories1", inplace=True)

    dm_energy_pass_land = dm_keep_mode.filter(
        {
            "Variables": ["tra_passenger_energy-demand-by-mode"],
            "Categories1": land_modes,
        }
    )
    dm_energy_pass_land.group_all("Categories1", inplace=True)
    dm_energy_pass_land.rename_col(
        "tra_passenger_energy-demand-by-mode",
        "tra_energy-demand_passenger-land",
        dim="Variables",
    )

    dm_energy_freight_tot = dm_freight_energy_by_mode.group_all(
        "Categories1", inplace=False
    )
    dm_energy_freight_tot.rename_col(
        "tra_freight_energy-demand-by-mode",
        "tra_energy-demand_freight",
        dim="Variables",
    )

    # Merge datamatrices for new-app
    dm_tpe = dm_keep_mode.flattest()
    dm_tpe.append(dm_keep_tech.flattest(), dim="Variables")
    dm_tpe.append(dm_keep_fuel.flattest(), dim="Variables")
    dm_tpe.append(dm_keep_energy.flattest(), dim="Variables")
    dm_tpe.append(dm_freight_energy_by_mode.flattest(), dim="Variables")
    dm_tpe.append(dm_freight_energy_by_fuel.flattest(), dim="Variables")
    dm_tpe.append(DM_passenger_out["soft-mobility"].flattest(), dim="Variables")
    dm_tpe.append(DM_passenger_out["emissions"].flattest(), dim="Variables")
    dm_tpe.append(dm_keep_aviation_emissions.flattest(), dim="Variables")
    dm_tpe.append(dm_keep_aviation_local.flattest(), dim="Variables")
    dm_tpe.append(dm_keep_aviation_energy.flattest(), dim="Variables")
    dm_tpe.append(dm_tech_HDVH.flattest(), dim="Variables")
    dm_tpe.append(dm_freight_emissions.flattest(), dim="Variables")
    dm_tpe.append(dm_emi_pass_land.flattest(), dim="Variables")
    dm_tpe.append(dm_emi_freight.flattest(), dim="Variables")
    dm_tpe.append(dm_energy_pass_land.flattest(), dim="Variables")
    dm_tpe.append(dm_energy_freight_tot.flattest(), dim="Variables")
    dm_tpe.append(dm_emi_freight_by_mode.flattest(), dim="Variables")
    dm_tpe.append(dm_energy_by_fuel.flattest(), dim="Variables")
    dm_tpe.append(dm_emissions_by_fuel.flattest(), dim="Variables")
    dm_tpe.append(group_main_categories(dm_tpe), dim="Variables")

    # Only the electric-car share needs a datamatrix of its own; the two totals are
    # summed from dm_tpe, which by now carries the grouped main-mode series.
    DM_kpi_dict = {
        "stock_EV": DM_passenger_out["tech"].filter(
            {
                "Variables": ["tra_passenger_technology-share-fleet"],
                "Categories1": ["LDV"],
            },
            inplace=False,
        ),
    }

    KPI = prepare_KPIs(dm_tpe, DM_kpi_dict, years_setting)

    # Freight emissions
    return dm_tpe, KPI


def tra_emissions_interface(
    dm_pass_emissions, dm_freight_emissions, dm_aviation_local, write_pickle=False
):
    # Extract outbound share (outbound / round-trip) per country×year as numpy array.
    # dm_aviation_local has dims (n_c, n_y, 2_vars, 1_cat1=aviation).
    dm_share_tmp = dm_aviation_local.copy()
    dm_share_tmp.group_all("Categories1", inplace=True)  # → (n_c, n_y, 2)
    _sidx = dm_share_tmp.idx
    share_np = (
        dm_share_tmp.array[:, :, _sidx["tra_passenger-emissions-local"]]
        / dm_share_tmp.array[:, :, _sidx["tra_passenger-emissions-total"]]
    )  # (n_c, n_y)

    # Build pass and freight aviation retaining the gas Categories1 dimension
    # so the result is compatible with dm_emi (transport-wo-aviation) below.
    dm_pass_avi = dm_pass_emissions.filter({"Categories1": ["aviation"]}).copy()
    dm_pass_avi.group_all("Categories1", inplace=True)  # aviation drops; gases → Cat1
    dm_freight_avi = dm_freight_emissions.filter({"Categories1": ["aviation"]}).copy()
    dm_freight_avi.group_all("Categories1", inplace=True)

    # Round-trip: unscaled pass + freight → "aviation-roundtrip" (supplementary graph)
    dm_avi_rt = dm_pass_avi.copy()
    dm_avi_rt.append(dm_freight_avi, "Variables")
    dm_avi_rt.groupby(
        {"aviation-roundtrip": ["tra_passenger_emissions", "tra_freight_emissions"]},
        "Variables",
        inplace=True,
    )

    # Outbound: scale pass by local share; freight is territorial so kept unchanged
    dm_avi_out = dm_pass_avi.copy()
    dm_avi_out.array *= share_np[:, :, np.newaxis, np.newaxis]
    dm_avi_out.append(dm_freight_avi.copy(), "Variables")
    dm_avi_out.groupby(
        {"aviation": ["tra_passenger_emissions", "tra_freight_emissions"]},
        "Variables",
        inplace=True,
    )

    dm_emi_aviation = dm_avi_out
    dm_emi_aviation.append(dm_avi_rt, "Variables")

    # Transport without aviation
    dm_emi = dm_pass_emissions.filter(
        {"Categories1": ["2W", "LDV", "bus", "metrotram", "rail"]}
    ).group_all("Categories1", inplace=False)
    dm_emi.append(
        dm_freight_emissions.filter(
            {"Categories1": ["HDVH", "HDVL", "HDVM", "IWW", "marine", "rail"]}
        ).group_all("Categories1", inplace=False),
        "Variables",
    )
    dm_emi.groupby(
        {"transport-wo-aviation": ["tra_passenger_emissions", "tra_freight_emissions"]},
        "Variables",
        inplace=True,
    )
    dm_emi.append(dm_emi_aviation, "Variables")

    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../../_database/data/interface/transport_to_emissions.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(dm_emi, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm_emi


def tra_agriculture_interface(
    dm_freight_agriculture, dm_passenger_agriculture, write_pickle=False
):
    # !FIXME: of all of the bio-energy demand, only the biogas one is accounted for in Agriculture
    dm_agriculture = dm_freight_agriculture
    dm_agriculture.array = dm_agriculture.array + dm_passenger_agriculture.array
    dm_agriculture.rename_col(
        "tra_freight_total-energy", "tra_bioenergy", dim="Variables"
    )
    dm_agriculture.rename_col("biogas", "gas", dim="Categories1")

    # if write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../../_database/data/interface/transport_to_agriculture.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(dm_agriculture, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm_agriculture


def tra_power_interface(DM_passenger_power, DM_freight_power, write_pickle=False):
    DM_power = DM_passenger_power
    DM_power["hydrogen"].array = (
        DM_power["hydrogen"].array + DM_freight_power["hydrogen"].array
    )
    DM_power["electricity"].add(
        0, dim="Variables", dummy=True, col_label="tra_power-demand_other", unit="GWh"
    )
    DM_power["electricity"].sort("Variables")
    DM_freight_power["electricity"].add(
        0,
        dim="Variables",
        dummy=True,
        col_label="tra_power-demand_aviation",
        unit="GWh",
    )
    DM_freight_power["electricity"].sort("Variables")
    DM_power["electricity"].array = (
        DM_power["electricity"].array + DM_freight_power["electricity"].array
    )

    # if write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../../_database/data/interface/transport_to_power.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(DM_power, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return DM_power


def tra_energy_interface(DM_passenger_energy, DM_freight_energy, write_pickle=False):
    DM_energy = {"freight": DM_freight_energy, "passenger": DM_passenger_energy}
    # if write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../../_database/data/interface/transport_to_energy.pickle",
        )
        my_pickle_dump(DM_energy, f)

    return DM_energy
