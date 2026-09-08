import numpy as np


def run(
    dm_capacity,
    dm_capacity_oilgas_ots,
    dm_capacity_waste_ots,
    dm_capacity_nuclear,
    dm_capacity_hydro_ots,
    dm_capacity_PV_wind_ots,
    years_ots,
    years_fts,
):
    # SECTION - Group Capacities
    dm_CH = dm_capacity_oilgas_ots.filter({"Country": ["Switzerland"]})
    dm_CH.append(
        dm_capacity_waste_ots.filter(
            {"Country": ["Switzerland"], "Variables": ["pow_capacity-Pmax"]}
        ),
        dim="Categories1",
    )
    dm_CH.append(
        dm_capacity_nuclear.filter({"Country": ["Switzerland"], "Years": years_ots}),
        dim="Categories1",
    )
    dm_CH.append(
        dm_capacity_hydro_ots.filter({"Country": ["Switzerland"], "Years": years_ots}),
        dim="Categories1",
    )
    dm_CH.append(
        dm_capacity_PV_wind_ots.filter(
            {"Country": ["Switzerland"], "Years": years_ots}
        ),
        dim="Categories1",
    )

    # SECTION - Split Gas into Gas-CC and Gas-GS based on Nexus-e
    # Gas is originally Symple Cycle
    dm_CH.rename_col("Gas", "GasSC", dim="Categories1")
    # But in recent years it becomes combined Cycle
    for yr in dm_CH.col_labels["Years"]:
        dm_CH["Switzerland", yr, "pow_capacity-Pmax", "GasSC"] = (
            dm_CH["Switzerland", yr, "pow_capacity-Pmax", "GasSC"]
            - dm_capacity["Switzerland", yr, "pow_capacity-Pmax", "GasCC"]
        )
    dm_tmp = dm_capacity.filter(
        {
            "Country": ["Switzerland"],
            "Years": years_ots,
            "Variables": ["pow_capacity-Pmax"],
            "Categories1": ["GasCC"],
        }
    )
    dm_CH.append(dm_tmp, dim="Categories1")

    # SECTION - Merge OTS capacity and FTS capacity in CH
    missing_ots_cat = list(
        set(dm_capacity.col_labels["Categories1"])
        - set(dm_CH.col_labels["Categories1"])
    )

    dm_CH.add(0, dummy=True, dim="Categories1", col_label=missing_ots_cat)
    dm_CH.sort("Categories1")
    dm_capacity.sort("Categories1")
    for yr in dm_CH.col_labels["Years"]:
        dm_capacity["Switzerland", yr, "pow_capacity-Pmax", :] = dm_CH[
            "Switzerland", yr, "pow_capacity-Pmax", :
        ]
    # Avoid 2025 discontinuity (especially in PV)
    dm_capacity["Switzerland", 2025, "pow_capacity-Pmax", :] = np.nan
    dm_capacity.fill_nans("Years")

    # SECTION - Add installed capacity to pickle
    dm_capacity.add(
        np.nan,
        dim="Variables",
        col_label="pow_existing-capacity",
        dummy=True,
        unit="MW",
    )
    for yr in years_ots:
        dm_capacity["Switzerland", yr, "pow_existing-capacity", :] = dm_capacity[
            "Switzerland", yr, "pow_capacity-Pmax", :
        ]
    dm_capacity.fill_nans("Years")
    for yr in years_fts:
        dm_capacity["Switzerland", yr, "pow_existing-capacity", :] = np.minimum(
            dm_capacity["Switzerland", yr, "pow_existing-capacity", :],
            dm_capacity["Switzerland", yr, "pow_capacity-Pmax", :],
        )

    return dm_capacity


def redistribute_by_canton(dm_capacity):
    # !FIXME: The ots capacity data for example for oil and gas are available by canton,
    # There is maybe a problem with the Nexus-e capacities
    # SECTION - Redistribute Swiss capacity-Pmax and existing-capacity by Canton
    dm_capacity.drop(
        dim="Variables", col_label=["pow_capacity-Emax", "pow_capacity-Pmin"]
    )
    dm_capacity_CH = dm_capacity.filter({"Country": ["Switzerland"]})
    dm_capacity.drop(dim="Country", col_label="Switzerland")
    dm_capacity[:, :, "pow_existing-capacity", :] = dm_capacity[
        :, :, "pow_capacity-Pmax", :
    ]
    dm_capacity.fill_nans("Years")
    dm_capacity.normalise(inplace=True, dim="Country")
    dm_capacity.array = (
        dm_capacity[:, :, :, :] * dm_capacity_CH["Switzerland", np.newaxis, :, :, :]
    )
    for var in dm_capacity.col_labels["Variables"]:
        dm_capacity.units[var] = dm_capacity_CH.units[var]

    mask = np.isnan(dm_capacity[:, :, :, :])
    dm_capacity[:, :, :, :][mask] = 0

    dm_capacity.append(dm_capacity_CH, dim="Country")

    return dm_capacity
