import numpy as np


def get_footprint_by_group(DM_footprint):

    def reshape_and_store(DM_footprint, keyword):

        DM = {}
        dm = DM_footprint[keyword].copy()

        dm_veh = dm.filter_w_regex(
            ({"Variables": "HDV_.*|LDV_.*|bus_.*|planes_.*|ships_.*|trains_.*"})
        )
        for v in dm_veh.col_labels["Variables"]:
            dm_veh.rename_col(v, f"{keyword}_" + v, "Variables")
        dm_veh.deepen(based_on="Variables")
        dm_veh.deepen(based_on="Variables")
        dm_veh.switch_categories_order("Categories1", "Categories3")
        lastcat = list(dm_veh.col_labels.keys())[-1]
        if len(dm_veh.col_labels[lastcat]) == 1:
            dm_veh.group_all(lastcat)
        DM["vehicles"] = dm_veh.copy()

        def deepen_else(dm, keyword=keyword):

            lastcat = dm.dim_labels[-1]
            if len(dm.col_labels[lastcat]) == 1:
                dm.group_all(lastcat)

            for v in dm.col_labels["Variables"]:
                dm.rename_col(v, f"{keyword}_" + v, "Variables")
            dm.deepen(based_on="Variables")

            if len(dm.dim_labels) == 5:
                dm.switch_categories_order("Categories1", "Categories2")

        # transport infrastructure
        dm_temp = dm.filter({"Variables": ["rail", "road", "trolley-cables"]})
        deepen_else(dm_temp)
        DM["infra-tra"] = dm_temp.copy()

        # domapp
        dm_temp = dm.filter(({"Variables": ["dishwasher", "fridge", "wmachine"]}))
        deepen_else(dm_temp)
        DM["domapp"] = dm_temp.copy()

        # electronics
        dm_temp = dm.filter(({"Variables": ["computer", "phone", "tv"]}))
        deepen_else(dm_temp)
        DM["electronics"] = dm_temp.copy()

        return DM

    DM_footprint_split = {}
    for keyword in DM_footprint.keys():
        DM_footprint_split[keyword] = reshape_and_store(DM_footprint, keyword)

    return DM_footprint_split


def get_footprint(footprint, DM_demand, DM_footprint):

    # vehicles
    dm_veh = DM_footprint["vehicles"].copy()
    dm_veh.units[footprint] = dm_veh.units[footprint].split("/")[0]
    if len(dm_veh.dim_labels) == 5:
        dm_veh.array = DM_demand["vehicles"].array * dm_veh.array
        dm_all = dm_veh.flatten()
    elif len(dm_veh.dim_labels) == 6:
        dm_veh.array = DM_demand["vehicles"].array[..., np.newaxis] * dm_veh.array
        dm_all = dm_veh.flatten().flatten()
        dm_all.deepen()

    def make_multiplication(dm_footprint, dm_demand):

        dm_temp = dm_footprint.copy()
        if len(dm_temp.dim_labels) == 4:
            dm_temp.array = dm_temp.array * dm_demand.array
        elif len(dm_temp.dim_labels) == 5:
            dm_temp.array = dm_temp.array * dm_demand.array[..., np.newaxis]
        dm_temp.units[footprint] = dm_temp.units[footprint].split("/")[0]

        return dm_temp

    # transport infra
    dm_temp = make_multiplication(DM_footprint["infra-tra"], DM_demand["tra-infra"])
    dm_all.append(dm_temp, "Categories1")

    # domapp
    dm_temp = make_multiplication(DM_footprint["domapp"], DM_demand["domapp"])
    dm_all.append(dm_temp, "Categories1")

    # electronics
    dm_temp = make_multiplication(DM_footprint["electronics"], DM_demand["electronics"])
    dm_all.append(dm_temp, "Categories1")

    # aggregate
    dm_all_agg = dm_all.copy()
    dm_all_agg.group_all("Categories1")

    return dm_all_agg


def variables_for_tpe(DM_footprint_agg):

    dm_tpe = DM_footprint_agg["materials"].flatten()
    dm_tpe.append(DM_footprint_agg["ecological"], "Variables")
    dm_tpe.append(DM_footprint_agg["gwp"], "Variables")
    dm_tpe.append(DM_footprint_agg["water"], "Variables")
    dm_tpe.append(DM_footprint_agg["air-pollutant"].flatten(), "Variables")
    dm_tpe.append(DM_footprint_agg["heavy-metals"].flatten(), "Variables")
    dm_tpe.append(DM_footprint_agg["energy-demand"], "Variables")

    # # checks
    # DM_footprint_agg["materials"].datamatrix_plot(stacked=True)
    # DM_footprint_agg['ecological'].datamatrix_plot()
    # DM_footprint_agg['gwp'].datamatrix_plot()
    # DM_footprint_agg['water'].datamatrix_plot()
    # DM_footprint_agg['air-pollutant'].datamatrix_plot(stacked=True)
    # DM_footprint_agg['heavy-metals-to-soil'].datamatrix_plot(stacked=True)
    # DM_footprint_agg['energy-demand'].filter({"Variables" : ["energy-demand-elec"]}).datamatrix_plot()
    # DM_footprint_agg['energy-demand'].filter({"Variables" : ["energy-demand-ff"]}).datamatrix_plot()

    return dm_tpe
