import os
import pickle

import numpy as np
import plotly.io as pio

from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    linear_fitting,
    my_pickle_dump,
    sort_pickle,
)

pio.renderers.default = "browser"


def make_fts(DM, name, years_fts, based_on):
    """All 4 levels identical: freeze at based_on year (BAU)."""
    dm = DM["ots"][name].copy()
    dm = linear_fitting(dm, years_fts, based_on=based_on)
    DM["fts"][name] = {}
    for level in list(range(1, 4 + 1)):
        DM["fts"][name][level] = dm.filter({"Years": years_fts})
    return


def make_fts_leveled(DM, name, years_fts, targets_l4, country="EU27"):
    """
    Level 1 = freeze at 2023 (BAU).
    Level 4 = linear ramp from 2023 value to targets_l4 at 2050.
    Levels 2/3 = linearly interpolated (1/3 and 2/3 of the gap from 1 to 4).

    targets_l4: {cat1_label: value_at_2050}
    DM shape expected: (Country, Years, Variables, Categories1)
    """
    # Level 1: freeze at 2023
    dm_l1_full = DM["ots"][name].copy()
    dm_l1_full = linear_fitting(dm_l1_full, years_fts, based_on=[2023])
    dm_l1 = dm_l1_full.filter({"Years": years_fts})

    # Level 4: ramp to 2050 targets
    dm_l4_full = dm_l1_full.copy()
    idx = dm_l4_full.idx
    for yr in years_fts[:-1]:  # NaN 2025-2045 so fill_nans re-interpolates
        dm_l4_full.array[idx[country], idx[yr], ...] = np.nan
    for cat, val in targets_l4.items():
        dm_l4_full.array[idx[country], idx[2050], :, idx[cat]] = val
    dm_l4_full.fill_nans("Years")
    dm_l4 = dm_l4_full.filter({"Years": years_fts})

    # Levels 2 and 3: evenly spaced between 1 and 4
    dm_l2 = dm_l1.copy()
    dm_l2.array = dm_l1.array + (dm_l4.array - dm_l1.array) / 3
    dm_l3 = dm_l1.copy()
    dm_l3.array = dm_l1.array + (dm_l4.array - dm_l1.array) * 2 / 3

    DM["fts"][name] = {1: dm_l1, 2: dm_l2, 3: dm_l3, 4: dm_l4}


def make_fts_eol_material_recovery(DM_industry, years_fts, country="EU27"):
    """
    Level 1 = freeze at 2023.
    Level 4 = linear ramp to full/near-full recovery rates at 2050 (from EUCalc).
    Levels 2/3 = evenly interpolated.

    DM shape: (Country, Years, Variables=["waste-material-recovery"],
               Categories1=product_type, Categories2=material)
    """
    name = "eol-material-recovery"

    # Level 1: freeze at 2023
    dm_l1_full = DM_industry["ots"][name].copy()
    dm_l1_full = linear_fitting(dm_l1_full, years_fts, based_on=[2023])
    dm_l1 = dm_l1_full.filter({"Years": years_fts})

    # Level 4: set 2050 recovery targets per product × material
    dm_l4_full = dm_l1_full.copy()
    idx = dm_l4_full.idx
    for yr in years_fts[:-1]:
        dm_l4_full.array[idx[country], idx[yr], ...] = np.nan

    # battery-lion
    for m in ["aluminium", "other", "steel"]:
        dm_l4_full.array[idx[country], idx[2050], :, idx["battery-lion"], idx[m]] = 1.0

    # vehicles, dishwasher, dryer, wmachine
    for p in ["vehicles", "dishwasher", "dryer", "wmachine"]:
        dm_l4_full.array[idx[country], idx[2050], :, idx[p], idx["aluminium"]] = 1.0
        dm_l4_full.array[idx[country], idx[2050], :, idx[p], idx["chem"]] = 0.9
        dm_l4_full.array[idx[country], idx[2050], :, idx[p], idx["copper"]] = 1.0
        dm_l4_full.array[idx[country], idx[2050], :, idx[p], idx["other"]] = 0.9
        dm_l4_full.array[idx[country], idx[2050], :, idx[p], idx["steel"]] = 1.0

    # electronics
    dm_l4_full.array[
        idx[country], idx[2050], :, idx["electronics"], idx["aluminium"]
    ] = 1.0
    dm_l4_full.array[idx[country], idx[2050], :, idx["electronics"], idx["copper"]] = (
        1.0
    )
    dm_l4_full.array[idx[country], idx[2050], :, idx["electronics"], idx["other"]] = 0.9
    dm_l4_full.array[idx[country], idx[2050], :, idx["electronics"], idx["steel"]] = 1.0

    # floor-area (buildings)
    for m, val in [
        ("aluminium", 1.0),
        ("cement", 0.9),
        ("chem", 0.9),
        ("glass", 1.0),
        ("other", 0.9),
        ("steel", 1.0),
        ("timber", 1.0),
    ]:
        dm_l4_full.array[idx[country], idx[2050], :, idx["floor-area"], idx[m]] = val

    # freezer, fridge
    for p in ["freezer", "fridge"]:
        dm_l4_full.array[idx[country], idx[2050], :, idx[p], idx["aluminium"]] = 1.0
        dm_l4_full.array[idx[country], idx[2050], :, idx[p], idx["chem"]] = 0.9
        dm_l4_full.array[idx[country], idx[2050], :, idx[p], idx["copper"]] = 1.0

    # packages
    dm_l4_full.array[
        idx[country], idx[2050], :, idx["aluminium-pack"], idx["aluminium"]
    ] = 1.0
    dm_l4_full.array[idx[country], idx[2050], :, idx["glass-pack"], idx["glass"]] = 1.0
    for p in ["paper-pack", "paper-print", "paper-san"]:
        dm_l4_full.array[idx[country], idx[2050], :, idx[p], idx["paper"]] = 1.0
    dm_l4_full.array[idx[country], idx[2050], :, idx["plastic-pack"], idx["chem"]] = 0.9

    # trains, planes, ships
    for p in ["trains", "planes", "ships"]:
        dm_l4_full.array[idx[country], idx[2050], :, idx[p], idx["aluminium"]] = 1.0
        dm_l4_full.array[idx[country], idx[2050], :, idx[p], idx["chem"]] = 0.9
        dm_l4_full.array[idx[country], idx[2050], :, idx[p], idx["glass"]] = 1.0
        dm_l4_full.array[idx[country], idx[2050], :, idx[p], idx["other"]] = 0.9
        dm_l4_full.array[idx[country], idx[2050], :, idx[p], idx["steel"]] = 1.0

    # infrastructure
    dm_l4_full.array[idx[country], idx[2050], :, idx["rail"], idx["steel"]] = 1.0
    dm_l4_full.array[idx[country], idx[2050], :, idx["rail"], idx["timber"]] = 1.0
    for m, val in [
        ("aluminium", 1.0),
        ("cement", 1.0),
        ("chem", 0.9),
        ("steel", 1.0),
        ("other", 0.9),
    ]:
        dm_l4_full.array[idx[country], idx[2050], :, idx["road"], idx[m]] = val
    dm_l4_full.array[
        idx[country], idx[2050], :, idx["trolley-cables"], idx["copper"]
    ] = 1.0

    dm_l4_full.fill_nans("Years")
    dm_l4 = dm_l4_full.filter({"Years": years_fts})

    # Levels 2 and 3: evenly spaced between 1 and 4
    dm_l2 = dm_l1.copy()
    dm_l2.array = dm_l1.array + (dm_l4.array - dm_l1.array) / 3
    dm_l3 = dm_l1.copy()
    dm_l3.array = dm_l1.array + (dm_l4.array - dm_l1.array) * 2 / 3

    DM_industry["fts"][name] = {1: dm_l1, 2: dm_l2, 3: dm_l3, 4: dm_l4}


def make_fts_eol_waste_management(DM_industry, years_fts, country="EU27"):
    """
    Level 1 = freeze at 2023.
    Level 4 (vehicles only):
      - total flows: export × 1.5, waste-collected × 1.25, littered = 0, uncollected = 0
      - collection shares: recycling = 1, energy-recovery/landfill/incineration/reuse = 0
    Levels 2/3 = evenly interpolated.
    All other product variables are identical across all levels.

    DM shape: (Country, Years, Variables=product_types, Categories1=waste_categories)
    """
    name = "eol-waste-management"

    # Level 1: freeze at 2023
    dm_l1_full = DM_industry["ots"][name].copy()
    dm_l1_full = linear_fitting(dm_l1_full, years_fts, based_on=[2023])
    dm_l1 = dm_l1_full.filter({"Years": years_fts})

    # Level 4: vehicles-specific targets, all other products unchanged
    dm_l4_full = dm_l1_full.copy()
    idx = dm_l4_full.idx
    for yr in years_fts[:-1]:
        dm_l4_full.array[idx[country], idx[yr], idx["vehicles"], :] = np.nan

    # Total flow targets scaled from 2023
    export_2023 = dm_l1_full.array[
        idx[country], idx[2023], idx["vehicles"], idx["export"]
    ]
    collected_2023 = dm_l1_full.array[
        idx[country], idx[2023], idx["vehicles"], idx["waste-collected"]
    ]
    dm_l4_full.array[idx[country], idx[2050], idx["vehicles"], idx["export"]] = (
        export_2023 * 1.5
    )
    dm_l4_full.array[
        idx[country], idx[2050], idx["vehicles"], idx["waste-collected"]
    ] = collected_2023 * 1.25
    dm_l4_full.array[idx[country], idx[2050], idx["vehicles"], idx["littered"]] = 0.0
    dm_l4_full.array[
        idx[country], idx[2050], idx["vehicles"], idx["waste-uncollected"]
    ] = 0.0

    # Collection share targets: full recycling
    dm_l4_full.array[idx[country], idx[2050], idx["vehicles"], idx["recycling"]] = 1.0
    dm_l4_full.array[
        idx[country], idx[2050], idx["vehicles"], idx["energy-recovery"]
    ] = 0.0
    dm_l4_full.array[idx[country], idx[2050], idx["vehicles"], idx["landfill"]] = 0.0
    dm_l4_full.array[idx[country], idx[2050], idx["vehicles"], idx["incineration"]] = (
        0.0
    )
    dm_l4_full.array[idx[country], idx[2050], idx["vehicles"], idx["reuse"]] = 0.0

    dm_l4_full.fill_nans("Years")
    dm_l4 = dm_l4_full.filter({"Years": years_fts})

    # Levels 2 and 3: evenly spaced between 1 and 4
    dm_l2 = dm_l1.copy()
    dm_l2.array = dm_l1.array + (dm_l4.array - dm_l1.array) / 3
    dm_l3 = dm_l1.copy()
    dm_l3.array = dm_l1.array + (dm_l4.array - dm_l1.array) * 2 / 3

    DM_industry["fts"][name] = {1: dm_l1, 2: dm_l2, 3: dm_l3, 4: dm_l4}


def make_fts_product_net_import(DM_industry, years_fts):
    dm_ots = DM_industry["ots"]["product-net-import"]
    products = dm_ots.col_labels["Categories1"]

    baseyear_start = 1990
    baseyear_end = 2023

    # custom fitting windows for specific products (rest use baseyear_start to baseyear_end)
    custom_windows = {
        "glass-pack": (2012, baseyear_end),
        "paper-pack": (2009, baseyear_end),
        "tv": (2012, baseyear_end),
        "dryer": (2000, 2007),
        "wmachine": (2000, 2007),
    }

    # constructed all-zero products: keep at zero
    zero_products = {
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
    }

    fts_dms = []
    for p in products:
        dm_p = dm_ots.filter({"Categories1": [p]}).copy()
        if p in zero_products:
            dm_p = linear_fitting(dm_p, years_fts, based_on=[baseyear_end])
            dm_fts_p = dm_p.filter({"Years": years_fts})
            dm_fts_p.array[:] = 0
        else:
            y_start, y_end = custom_windows.get(p, (baseyear_start, baseyear_end))
            dm_p = linear_fitting(
                dm_p, years_fts, based_on=list(range(y_start, y_end + 1))
            )
            dm_fts_p = dm_p.filter({"Years": years_fts})
            dm_fts_p.array[dm_fts_p.array > 1] = 1
        fts_dms.append(dm_fts_p)

    dm_fts = fts_dms[0].copy()
    for d in fts_dms[1:]:
        dm_fts.append(d, "Categories1")
    dm_fts.sort("Categories1")

    DM_industry["fts"]["product-net-import"] = {}
    for level in range(1, 5):
        DM_industry["fts"]["product-net-import"][level] = dm_fts.copy()

    return


def run(DM_industry, DM_ammonia, country_list, years_ots, years_fts):
    # directory
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    DM_industry["fts"] = {}
    DM_ammonia["fts"] = {}

    # levers with no ambition differentiation: BAU = freeze at 2023, all 4 levels identical
    for lever in [
        "technology-development",
        "cc",
        "energy-carrier-mix",
        "material-net-import",
        "paperpack",
    ]:
        make_fts(DM_industry, lever, years_fts, based_on=[2023])

    # levers with differentiated ambition levels (level 1 = BAU, level 4 = EUCalc targets,
    # levels 2/3 = linearly interpolated)
    make_fts_leveled(
        DM_industry,
        "material-efficiency",
        years_fts,
        {
            "steel": 0.33,
            "cement": 0.20,
            # "ammonia": 0.10,
            "chem": 0.30,
            "paper": 0.10,
            "aluminium": 0.14,
            "glass": 0.12,
            "lime": 0.14,
            "copper": 0.14,
        },
    )
    make_fts_leveled(
        DM_industry,
        "material-switch",
        years_fts,
        {
            "cars-steel-to-aluminium": 0.50,
            "trucks-steel-to-aluminium": 0.45,
            "cars-steel-to-chem": 0.20,
            "trucks-steel-to-chem": 0.15,
            "build-steel-to-timber": 0.20,
            "build-cement-to-timber": 0.60,
            "reno-chem-to-paper": 0.10,
            "reno-chem-to-natfibers": 0.20,
        },
    )
    make_fts_leveled(
        DM_industry,
        "technology-share",
        years_fts,
        {
            "steel-BF-BOF": 0.80,
            "steel-hisarna": 0.10,
            "steel-hydrog-DRI": 0.10,
            "cement-dry-kiln": 0.80,
            "cement-wet-kiln": 0.00,
            "cement-geopolym": 0.20,
        },
    )
    make_fts_eol_material_recovery(DM_industry, years_fts)
    make_fts_eol_waste_management(DM_industry, years_fts)

    # normalize technology-share within each technology group (steel and cement sum to 1)
    steel_techs = ["steel-BF-BOF", "steel-hisarna", "steel-hydrog-DRI"]
    cement_techs = ["cement-dry-kiln", "cement-geopolym", "cement-wet-kiln"]
    for level in range(1, 5):
        dm = DM_industry["fts"]["technology-share"][level]
        idx = dm.idx
        for group in [steel_techs, cement_techs]:
            s = sum(dm.array[:, :, :, idx[t]] for t in group)
            for t in group:
                dm.array[:, :, :, idx[t]] = np.where(
                    s > 0, dm.array[:, :, :, idx[t]] / s, 0
                )

    # product-net-import: per-product trend fitting with custom windows
    make_fts_product_net_import(DM_industry, years_fts)

    # ammonia levers: all BAU (freeze at 2023, all 4 levels identical)
    for lever in [
        "product-net-import",
        "material-net-import",
        "material-efficiency",
        "eol-material-recovery",
        "technology-development",
        "cc",
        "energy-carrier-mix",
    ]:
        make_fts(DM_ammonia, lever, years_fts, based_on=[2023])

    # save industry
    pickle_file = os.path.join(
        current_file_directory, "../../../../data/datamatrix/industry.pickle"
    )
    my_pickle_dump(DM_new=DM_industry, local_pickle_file=pickle_file)
    sort_pickle(pickle_file)

    # save ammonia
    pickle_file = os.path.join(
        current_file_directory, "../../../../data/datamatrix/ammonia.pickle"
    )
    my_pickle_dump(DM_new=DM_ammonia, local_pickle_file=pickle_file)
    sort_pickle(pickle_file)

    return DM_industry, DM_ammonia


if __name__ == "__main__":
    # get years
    years_ots = create_years_list(1990, 2023, 1)
    years_fts = create_years_list(2025, 2050, 5)
    country_list = ["EU27"]

    # load current pickles
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(
        current_file_directory, "../../../../data/datamatrix/industry.pickle"
    )
    with open(filepath, "rb") as f:
        DM_industry_current = pickle.load(f)
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(
        current_file_directory, "../../../../data/datamatrix/ammonia.pickle"
    )
    with open(filepath, "rb") as f:
        DM_amm_current = pickle.load(f)

    # load industry ots pickle
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(
        current_file_directory, "../data/datamatrix/industry_ots.pickle"
    )
    if not os.path.exists(filepath):
        raise FileNotFoundError("You need to run ots_pickle_run() first")
    with open(filepath, "rb") as f:
        DM_industry = pickle.load(f)

    # load ammonia ots pickle
    filepath = os.path.join(
        current_file_directory, "../data/datamatrix/ammonia_ots.pickle"
    )
    if not os.path.exists(filepath):
        raise FileNotFoundError("You need to run ots_pickle_run() first")
    with open(filepath, "rb") as f:
        DM_ammonia = pickle.load(f)

    run(DM_industry, DM_ammonia, country_list, years_ots, years_fts)
