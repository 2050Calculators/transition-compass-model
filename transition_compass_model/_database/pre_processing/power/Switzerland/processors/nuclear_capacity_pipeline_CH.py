import numpy as np


def extract_nuclear_capacity_data(reactor_list, dm_capacity):
    # 'Muhleberg': {'Canton': 'BE', 'Pmax': 373, 'StartYr': 1971, 'EndYr': 2019},

    dm = dm_capacity.filter(
        {"Variables": ["pow_capacity-Pmax"], "Categories1": ["Nuclear"]}
    )
    # Reset values to 0
    idx = dm.idx
    dm.array[...] = 0
    years_missing = list(
        set(range(dm.col_labels["Years"][0], dm.col_labels["Years"][-1]))
        - set(dm.col_labels["Years"])
    )
    dm.add(0, dummy=True, dim="Years", col_label=years_missing)
    dm.sort("Years")
    for name, properties in reactor_list.items():
        cntr = properties["Canton"]
        startyr = max(dm.col_labels["Years"][0], properties["StartYr"])
        endyr = min(dm.col_labels["Years"][-1], properties["EndYr"])
        for yr in range(startyr, endyr + 1):
            dm.array[idx[cntr], idx[yr], idx["pow_capacity-Pmax"], idx["Nuclear"]] += (
                properties["Pmax"]
            )
    dm.array[idx["Switzerland"], ...] = np.nansum(dm.array, axis=0)
    return dm


def run(dm_capacity):
    # Source:  https://de.wikipedia.org/wiki/Liste_der_Kernreaktoren_in_der_Schweiz
    reactor_list = {
        "Muhleberg": {"Canton": "BE", "Pmax": 373, "StartYr": 1971, "EndYr": 2019},
        "Beznau1": {"Canton": "AG", "Pmax": 365, "StartYr": 1969, "EndYr": 2033},
        "Beznau2": {"Canton": "AG", "Pmax": 365, "StartYr": 1971, "EndYr": 2032},
        "Gosgen": {"Canton": "SO", "Pmax": 1010, "StartYr": 1979, "EndYr": 2060},
        "Liebstadt": {"Canton": "AG", "Pmax": 1233, "StartYr": 1984, "EndYr": 2060},
    }

    dm_capacity_nuclear = extract_nuclear_capacity_data(reactor_list, dm_capacity)

    return dm_capacity_nuclear, reactor_list
