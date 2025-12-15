"""Utilities for extracting and preparing lever data for plotting."""


def get_lever_data_to_plot(lever_name, DM_input):
    """Extract lever data formatted for plotting.

    Given the lever_name and a DM_input containing the input used in the run,
    returns a DM with keys 1,2,3,4 and for each, a flat dm covering the whole time series.
    lever_name should be in chosen lever_position.json
    DM_input can be obtained by running:
    DM_input = filter_country_and_load_data_from_pickles(country_list, modules_list)

    Args:
        lever_name: Name of the lever to extract data for
        DM_input: Input data matrix from the model run

    Returns:
        Dictionary with keys 1-4, each containing a flattened datamatrix for the time series

    Note:
        !FIXME: multiply by 100 when %
        !FIXME: add switch-case when there are multiple sublever, return sublever if there is a single one
    """
    from model.common.auxiliary_functions import return_lever_data

    DM_lever = return_lever_data(lever_name, DM_input)
    DM_clean = dict()
    if DM_lever is None:
        print(f"lever_name {lever_name} not found in input DM")
    else:
        dm_ots = DM_lever["ots"]
        if not isinstance(dm_ots, dict):
            match lever_name:
                case "lever_heatcool-behaviour":
                    dm_ots = dm_ots.filter({"Variables": ["bld_Tint-heating"]})
                    dm_ots.group_all("Categories1", aggregation="mean", inplace=True)
                    dm_ots = dm_ots.flattest()
                    for lev in range(4):
                        dm_fts = DM_lever["fts"][lev + 1].filter(
                            {"Variables": ["bld_Tint-heating"]}
                        )
                        dm_fts.group_all(
                            "Categories1", aggregation="mean", inplace=True
                        )
                        dm_fts = dm_fts.flattest()
                        DM_clean[lev + 1] = dm_ots.copy()
                        DM_clean[lev + 1].append(dm_fts, dim="Years")
                case "lever_heating-efficiency":
                    dm_ots = dm_ots.filter({"Categories2": ["heat-pump"]})
                    dm_ots = dm_ots.flattest()
                    for lev in range(4):
                        dm_fts = DM_lever["fts"][lev + 1].filter(
                            {"Categories2": ["heat-pump"]}
                        )
                        dm_fts = dm_fts.flattest()
                        DM_clean[lev + 1] = dm_ots.copy()
                        DM_clean[lev + 1].append(dm_fts, dim="Years")
                case "lever_floor-intensity":
                    dm_ots = dm_ots.filter(
                        {"Variables": ["lfs_floor-intensity_space-cap"]}
                    )
                    for lev in range(4):
                        dm_fts = DM_lever["fts"][lev + 1].filter(
                            {"Variables": ["lfs_floor-intensity_space-cap"]}
                        )
                        DM_clean[lev + 1] = dm_ots.copy()
                        DM_clean[lev + 1].append(dm_fts, dim="Years")
                case "lever_passenger_modal-share":
                    dm_ots = dm_ots.flattest()
                    dm_ots.array = dm_ots.array * 100
                    for lev in range(4):
                        dm_fts = DM_lever["fts"][lev + 1].flattest()
                        dm_fts.array = dm_fts.array * 100
                        DM_clean[lev + 1] = dm_ots.copy()
                        DM_clean[lev + 1].append(dm_fts, dim="Years")
                case "lever_fuel-mix":
                    dm_ots.filter(
                        {
                            "Categories2": ["aviation", "road"],
                            "Categories1": ["biofuel"],
                        },
                        inplace=True,
                    )
                    dm_ots = dm_ots.flattest()
                    dm_ots.array = dm_ots.array * 100
                    for lev in range(4):
                        dm_fts = DM_lever["fts"][lev + 1]
                        dm_fts.filter(
                            {
                                "Categories2": ["aviation", "road"],
                                "Categories1": ["biofuel"],
                            },
                            inplace=True,
                        )
                        dm_fts = dm_fts.flattest()
                        dm_fts.array = dm_fts.array * 100
                        DM_clean[lev + 1] = dm_ots.copy()
                        DM_clean[lev + 1].append(dm_fts, dim="Years")
                case "lever_passenger_technology-share_new":
                    dm_ots_LDV = dm_ots.filter(
                        {"Categories1": ["LDV"], "Categories2": ["BEV"]}
                    )
                    dm_ots_aviation = dm_ots.filter(
                        {"Categories1": ["aviation"], "Categories2": ["BEV", "H2"]}
                    )
                    dm_ots_LDV = dm_ots_LDV.flattest()
                    dm_ots_aviation = dm_ots_aviation.flattest()
                    dm_ots_LDV.append(dm_ots_aviation, dim="Variables")
                    dm_ots = dm_ots_LDV
                    dm_ots.array = dm_ots.array * 100
                    for lev in range(4):
                        dm_fts = DM_lever["fts"][lev + 1]
                        dm_fts_LDV = dm_fts.filter(
                            {"Categories1": ["LDV"], "Categories2": ["BEV"]}
                        )
                        dm_fts_aviation = dm_fts.filter(
                            {"Categories1": ["aviation"], "Categories2": ["BEV", "H2"]}
                        )
                        dm_fts_LDV = dm_fts_LDV.flattest()
                        dm_fts_aviation = dm_fts_aviation.flattest()
                        dm_fts_LDV.append(dm_fts_aviation, dim="Variables")
                        dm_fts = dm_fts_LDV
                        dm_fts.array = dm_fts.array * 100
                        DM_clean[lev + 1] = dm_ots.copy()
                        DM_clean[lev + 1].append(dm_fts, dim="Years")
                case "lever_passenger_veh-efficiency_new":
                    dm_ots.filter(
                        {
                            "Categories1": ["LDV"],
                            "Categories2": [
                                "BEV",
                                "FCEV",
                                "ICE-diesel",
                                "ICE-gas",
                                "ICE-gasoline",
                                "PHEV-diesel",
                                "PHEV-gasoline",
                            ],
                        },
                        inplace=True,
                    )
                    dm_ots = dm_ots.flattest()
                    dm_ots.array = dm_ots.array * 100
                    for lev in range(4):
                        dm_fts = DM_lever["fts"][lev + 1]
                        dm_fts.filter(
                            {
                                "Categories1": ["LDV"],
                                "Categories2": [
                                    "BEV",
                                    "FCEV",
                                    "ICE-diesel",
                                    "ICE-gas",
                                    "ICE-gasoline",
                                    "PHEV-diesel",
                                    "PHEV-gasoline",
                                ],
                            },
                            inplace=True,
                        )
                        dm_fts = dm_fts.flattest()
                        dm_fts.array = dm_fts.array * 100
                        DM_clean[lev + 1] = dm_ots.copy()
                        DM_clean[lev + 1].append(dm_fts, dim="Years")
                case _:
                    dm_ots = dm_ots.flattest()
                    for lev in range(4):
                        dm_fts = DM_lever["fts"][lev + 1].flattest()
                        DM_clean[lev + 1] = dm_ots.copy()
                        DM_clean[lev + 1].append(dm_fts, dim="Years")
        else:
            # If there is only one sublever - return it
            if len(dm_ots) == 1:
                match lever_name:
                    case "lever_heating-technology-fuel":
                        key = next(iter(dm_ots))
                        dm_ots = dm_ots[key]
                        dm_ots.filter(
                            {
                                "Categories2": ["B", "F"],
                                "Categories3": [
                                    "district-heating",
                                    "gas",
                                    "heating-oil",
                                    "wood",
                                    "heat-pump",
                                ],
                            },
                            inplace=True,
                        )
                        dm_ots = dm_ots.flattest()
                        dm_ots.array = dm_ots.array * 100
                        for lev in range(4):
                            dm_fts = DM_lever["fts"][key][lev + 1]
                            dm_fts.filter(
                                {
                                    "Categories2": ["B", "F"],
                                    "Categories3": [
                                        "district-heating",
                                        "gas",
                                        "heating-oil",
                                        "wood",
                                        "heat-pump",
                                    ],
                                },
                                inplace=True,
                            )
                            dm_fts = dm_fts.flattest()
                            dm_fts.array = dm_fts.array * 100
                            DM_clean[lev + 1] = dm_ots.copy()
                            DM_clean[lev + 1].append(dm_fts, dim="Years")
                    case _:
                        key = next(iter(dm_ots))
                        dm_ots = dm_ots[key].flattest()
                        for lev in range(4):
                            dm_fts = DM_lever["fts"][key][lev + 1].flattest()
                            DM_clean[lev + 1] = dm_ots.copy()
                            DM_clean[lev + 1].append(dm_fts, dim="Years")
            else:
                match lever_name:
                    case "lever_building-renovation-rate":
                        key = "bld_renovation-rate"
                        dm_ots = dm_ots[key].flattest()
                        dm_ots.array = dm_ots.array * 100
                        for lev in range(4):
                            dm_fts = DM_lever["fts"][key][lev + 1].flattest()
                            dm_fts.array = dm_fts.array * 100
                            DM_clean[lev + 1] = dm_ots.copy()
                            DM_clean[lev + 1].append(dm_fts, dim="Years")
                    case _:
                        print(
                            f"The lever {lever_name} controls more than one variable and cannot be plotted"
                        )

    return DM_clean
