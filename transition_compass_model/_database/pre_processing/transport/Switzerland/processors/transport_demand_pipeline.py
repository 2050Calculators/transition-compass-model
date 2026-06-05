import os

import numpy as np

from transition_compass_model._database.pre_processing.transport.Switzerland.get_data_functions import (
    utils,
)
from transition_compass_model._database.pre_processing.transport.Switzerland.get_data_functions.demand_pkm_vkm import (
    get_transport_demand_pkm,
    get_transport_demand_vkm,
    get_travel_demand_region_microrecencement,
)
from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    dm_add_missing_variables,
    load_pop,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

#################################
#####   TRANSPORT DEMAND   ######
#################################


def extrapolate_missing_pkm_cap_based_on_pkm_CH(
    dm_pkm_cap_MRMT, dm_pkm_CH, dm_pop, years_ots
):
    """
    Fills missing years in pkm/cap for Switzerland and Vaud based on the pkm curve of Switzerland, with the following steps:
    1. Interpolate swiss MRMT data for missing year with FSO data (by doing ratio than linear interpolation)
    2. Use swiss MRMt interpolate to interpolate vaud MRMT with the smae method as before
    3. Say that the ratio between swiss OFS and swiss MRMT represent the differences in populations taken into account. MRMt inclunding only residential and OFS territorial and thus multiplying by the ratio previously computed would transform our data from residential to territorial ?

    Args:
        dm_pkm_cap_MRMT (DataMatrix): _data for pkm/cap from the microrecencement, with missing years, for Switzerland and Vaud
        dm_pkm_CH (DataMatrix): pkm for Switzerland from FSO, with all years but no split for Vaud
        dm_pop (DataMatrix): population data for Switzerland and Vaud, in cap
        years_ots (list): list of years for which to compute the demand, in OTS (1990-2023)

    Returns:
        DataMatrix: Adjusted pkm/cap for Switzerland and Vaud, with all years, in pkm/cap
    """
    # pkm_cap = pkm (CH) / pop
    arr_pkm_cap = dm_pkm_CH[...] / dm_pop["Switzerland", np.newaxis, :, :, np.newaxis]
    dm_pkm_CH.add(
        arr_pkm_cap, dim="Variables", col_label="tra_pkm-cap_official", unit="pkm/cap"
    )
    dm_CH = dm_pkm_CH.filter({"Variables": ["tra_pkm-cap_official"]})

    # for Switzerland : merge official pkm/cap with pkm/cap from microresencement (MRMT)
    dm_add_missing_variables(dm_pkm_cap_MRMT, {"Years": years_ots})
    dm_pkm_cap_MRMT.sort("Categories1")
    dm_pkm_cap_MRMT.rename_col("tra_pkm-cap", "tra_pkm-cap_MRMT", "Variables")
    # Append the MRMT data to the official data for Switzerland, to be used as reference for extrapolation of missing years and for Vaud.
    dm_CH.append(
        dm_pkm_cap_MRMT.filter(
            {"Country": ["Switzerland"], "Categories1": dm_CH.col_labels["Categories1"]}
        ),
        dim="Variables",
    )

    # Reconstruct missing years in pkm/cap MRMT based on official pkm/cap
    # computes ratio = MRMT / official
    # Fits a linear trend over years for that ratio (fills missing years consistently).
    # Reconstructs annual MRMT as reconstructed_ratio × official.
    dm_pkm_cap_new_CH = utils.fill_var_nans_based_on_var_curve(
        dm_CH,
        var_nan="tra_pkm-cap_MRMT",
        var_ref="tra_pkm-cap_official",
        keep_all_vars=True,
    )
    dm_pkm_cap_new_CH.operation(
        "tra_pkm-cap_official", "/", "tra_pkm-cap_MRMT", out_col="adj_factor", unit="%"
    )

    # For Vaud, extrapolate pkm/cap for all years based on the official pkm/cap of Switzerland and the ratio between MRMT and official for Switzerland (assuming same ratio applies to Vaud)
    dm_tmp = dm_pkm_cap_new_CH.filter({"Variables": ["tra_pkm-cap_MRMT"]})
    dm_tmp.rename_col("Switzerland", "Vaud", dim="Country")
    dm_tmp.rename_col("tra_pkm-cap_MRMT", "tra_pkm-cap_MRMT_CH", dim="Variables")
    dm_tmp.append(
        dm_pkm_cap_MRMT.filter(
            {"Country": ["Vaud"], "Categories1": dm_tmp.col_labels["Categories1"]}
        ),
        dim="Variables",
    )
    # fill missing years for Vaud based on the ratio of MRMT/official for Switzerland
    dm_pkm_cap_new_VD = utils.fill_var_nans_based_on_var_curve(
        dm_tmp, var_nan="tra_pkm-cap_MRMT", var_ref="tra_pkm-cap_MRMT_CH"
    )
    # Adjust Vaud demand to go from MRMT to official
    dm_adj_fact = dm_pkm_cap_new_CH.filter({"Variables": ["adj_factor"]})
    dm_adj_fact.rename_col("Switzerland", "Vaud", dim="Country")
    dm_pkm_cap_new_VD.append(dm_adj_fact, dim="Variables")
    dm_pkm_cap_new_VD.operation(
        "adj_factor",
        "*",
        "tra_pkm-cap_MRMT",
        out_col="tra_pkm-cap_official",
        unit="pkm/cap",
    )

    # dm_pkm_cap_new_VD.rename_col("tra_pkm-cap_MRMT", "tra_pkm-cap_official", dim="Variables")
    # Keep only "official" data
    dm_pkm_cap_new_CH.filter({"Variables": ["tra_pkm-cap_official"]}, inplace=True)
    dm_pkm_cap_new_VD.filter({"Variables": ["tra_pkm-cap_official"]}, inplace=True)
    dm_pkm_cap_new_CH.append(dm_pkm_cap_new_VD, dim="Country")
    dm_pkm_cap_new_CH.rename_col("tra_pkm-cap_official", "tra_pkm-cap", dim="Variables")

    return dm_pkm_cap_new_CH


def compute_pkm_from_pkm_cap(dm_pkm_cap, dm_pop):
    dm_pkm_cap.sort("Country")
    dm_pop.sort("Country")
    arr = dm_pkm_cap.array * dm_pop.array[:, :, :, np.newaxis]
    dm_pkm = DataMatrix.based_on(
        arr,
        dm_pkm_cap,
        change={"Variables": ["tra_passenger_transport-demand"]},
        units={"tra_passenger_transport-demand": "pkm"},
    )
    return dm_pkm


def compute_vkm_CH_VD(dm_vkm_CH, dm_pkm_CH, dm_pkm):
    """
    Compute vkm for Switzerland and Vaud based on occupancy rate of Switzerland.
    The occupancy rate is computed as pkm/vkm for Switzerland, and then applied to both Switzerland and Vaud to compute vkm from pkm.
    """
    # Occupancy = demand (pkm) / demand (vkm)
    dm_vkm_CH.append(
        dm_pkm_CH.filter({"Categories1": dm_vkm_CH.col_labels["Categories1"]}),
        dim="Variables",
    )
    dm_vkm_CH.operation(
        "tra_passenger_transport-demand",
        "/",
        "tra_passenger_transport-demand-vkm",
        out_col="tra_passenger_occupancy",
        unit="pkm/vkm",
    )
    dm_vkm_CH.sort("Categories1")

    # Extract occupancy and set same occupancy for CH and VD
    dm_occupancy = dm_vkm_CH.filter({"Variables": ["tra_passenger_occupancy"]})
    dm_occupancy_VD = dm_occupancy.copy()
    dm_occupancy_VD.rename_col("Switzerland", "Vaud", dim="Country")
    dm_occupancy.append(dm_occupancy_VD, dim="Country")

    dm_occupancy.append(
        dm_pkm.filter({"Categories1": dm_vkm_CH.col_labels["Categories1"]}),
        dim="Variables",
    )
    dm_occupancy.operation(
        "tra_passenger_transport-demand",
        "/",
        "tra_passenger_occupancy",
        out_col="tra_passenger_transport-demand-vkm",
        unit="vkm",
    )

    dm_vkm = dm_occupancy.filter({"Variables": ["tra_passenger_transport-demand-vkm"]})

    return dm_vkm


def run(dm_pop_ots, years_ots):
    """
    This function runs the whole pipeline to compute transport demand in pkm and vkm for Switzerland and Vaud, based on the following data sources:
    -  FSO, 2024. Transport de personnes: prestations de transport. FSO number :  je-f-11.04.01.02
    -  FSO, 2024. Transport de personnes: prestations kilométriques et mouvements des véhicules. FSO number: je-f-11.04.01.01
    Recensement micro régional de la mobilité et des transports (MRMT). Here Lémanique région is used as a proxy for Vaud.
        - FSO, 2021. Recensement micro régional de la mobilité et des transports (MRMT). FSO number: su-f-11.04.03-MZ-2021-T01.xlsx
        - FSO, 2015. Recensement micro régional de la mobilité et des transports (MRMT). FSO number: su-f-11.04.03-MZ-2015-T01.xls
        - FSO, 2010. Recensement micro régional de la mobilité et des transports (MRMT). FSO number: su-f-11.04.03-MZ-2010-T00.xls
        - FSO, 2005. Recensement micro régional de la mobilité et des transports (MRMT)
        - FSO, 2000. Recensement micro régional de la mobilité et des transports (MRMT)
    The function first reads the data, then adjusts the pkm/cap values for Vaud based on the official values for 2015 and 2021, and finally extrapolates the missing years based on the pkm curve of Switzerland.
    The vkm is then computed based on the occupancy rate (pkm/vkm) of Switzerland, which is applied to both Switzerland and Vaud.

    Args:
        dm_pop_ots: DataMatrix with population data for Switzerland and Vaud, in cap
        years_ots: list of years for which to compute the demand, in OTS (1990-2023)

    Returns:
        dm_pkm_cap: DataMatrix with pkm/cap for Switzerland and Vaud, in pkm/cap
        dm_pkm: DataMatrix with pkm for Switzerland and Vaud, in pkm
        dm_vkm: DataMatrix with vkm for Switzerland and Vaud, in vkm
    """

    this_dir = os.path.dirname(os.path.abspath(__file__))

    # SECTION Transport demand ots for pkm, vkm
    #### Passenger transport demand pkm - Switzerland only
    # Data source: FSO, 2024. Transport de personnes: prestations de transport. FSO number: je-f-11.04.01.02
    file_url = "https://dam-api.bfs.admin.ch/hub/api/dam/assets/36202626/master"
    local_filename = os.path.join(this_dir, "../data/tra_pkm_CH.xlsx")
    dm_pkm_CH = get_transport_demand_pkm(file_url, local_filename, years_ots)

    #### Passenger transport demand vkm - Switzerland only
    ## Transport de personnes: prestations kilométriques et mouvements des véhicules
    file_url = "https://dam-api.bfs.admin.ch/hub/api/dam/assets/32253171/master"
    local_filename = os.path.join(this_dir, "../data/tra_vkm_CH.xlsx")
    dm_vkm_CH = get_transport_demand_vkm(file_url, local_filename, years_ots)

    # Comportement de la population en matière de transports, tableaux de synthèse - Résultats du Microrecensement mobilité et transports 2021
    # Distance journalière moyenne en Suisse par personne selon le moyen de transport et le motif de déplacement (en km)
    # 2021: https://www.bfs.admin.ch/bfs/de/home/statistiken/kataloge-datenbanken.assetdetail.24267706.html
    # 2015: https://www.bfs.admin.ch/asset/de/2503926
    # 2010: https://www.bfs.admin.ch/asset/de/291635
    file_url_dict = {
        2021: "https://dam-api.bfs.admin.ch/hub/api/dam/assets/24267706/master",
        2015: "https://dam-api.bfs.admin.ch/hub/api/dam/assets/2503926/master",
        2010: "https://dam-api.bfs.admin.ch/hub/api/dam/assets/291635/master",
        2005: None,
        2000: None,
    }

    local_filename_dict = {
        2021: os.path.join(this_dir, "../data/tra_pkm_CH_reg_2021.xlsx"),
        2015: os.path.join(this_dir, "../data/tra_pkm_CH_reg_2015.xls"),
        2010: os.path.join(this_dir, "../data/tra_pkm_CH_reg_2010.xls"),
        2005: os.path.join(this_dir, "../data/tra_pkm_CH_reg_2005.xls"),
        2000: os.path.join(this_dir, "../data/tra_pkm_CH_reg_2000.xls"),
    }

    dm_pkm_cap_raw = None
    for year in local_filename_dict.keys():
        dm = get_travel_demand_region_microrecencement(
            file_url_dict[year], local_filename_dict[year], year
        )
        if dm_pkm_cap_raw is None:
            dm_pkm_cap_raw = dm.copy()
        else:
            dm_pkm_cap_raw.append(dm, dim="Years")
    dm_pkm_cap_raw.sort("Years")

    # For Vaud, adjust pkm/cap for 2015 and 2021 with actual values (split unchanged)
    # !FIXME: This should be adjusted to all cantons!
    # Do this by using tra_pkm_CH_cantons_2015.xlsx and tra_pkm_CH_cantons_2021.xlsx
    # FSO, 2023. Comportement de la population en matière de transport, chiffres clés - par canton. FSO number: su-f-11.04.03-MZ-2021-A2_Kant.
    # https://www.bfs.admin.ch/bfs/fr/home/statistiques/mobilite-transports/transport-personnes/comportements-transports/tableaux-2021/cantons.assetdetail.24025445.html
    # FSO, 2017. Comportement de la population en matière de transport, chiffres clés par canton (MRMT).
    # https://www.bfs.admin.ch/bfs/fr/home/statistiques/mobilite-transports/transport-personnes/comportements-transports/tableaux-2015/cantons.html
    VD_pkm_day = {2015: 38.2, 2021: 32.1}
    idx = dm_pkm_cap_raw.idx
    arr_tot_pkm_cap_raw = np.nansum(dm_pkm_cap_raw.array, axis=-1)
    corr_fact = dict()
    for yr in VD_pkm_day.keys():
        corr_fact[yr] = arr_tot_pkm_cap_raw[
            idx["Vaud"], idx[yr], idx["tra_pkm-cap"]
        ] / (VD_pkm_day[yr] * 365)
    avg_fact = sum(corr_fact.values()) / len(corr_fact.values())

    for yr in dm_pkm_cap_raw.col_labels["Years"]:
        if yr not in VD_pkm_day.keys():
            corr_fact[yr] = avg_fact
        dm_pkm_cap_raw.array[idx["Vaud"], idx[yr], idx["tra_pkm-cap"], :] = (
            dm_pkm_cap_raw.array[idx["Vaud"], idx[yr], idx["tra_pkm-cap"], :]
            / corr_fact[yr]
        )

    # For the missing years extrapolate the pkm/cap value base on the pkm curve of Switzerland
    dm_pkm_cap = extrapolate_missing_pkm_cap_based_on_pkm_CH(
        dm_pkm_cap_raw, dm_pkm_CH, dm_pop_ots, years_ots
    )
    # Compute pkm from pkm/cap and population
    dm_pkm = compute_pkm_from_pkm_cap(dm_pkm_cap, dm_pop_ots)
    del dm_pkm_cap_raw, dm, local_filename_dict, file_url_dict, year

    # Re-compute vkm CH and extrapolate VD by enforcing vkm/pkm_VD = vkm/pkm_CH
    dm_vkm = compute_vkm_CH_VD(dm_vkm_CH, dm_pkm_CH, dm_pkm)

    return dm_pkm_cap, dm_pkm, dm_vkm


if __name__ == "__main__":
    years_ots = create_years_list(1990, 2023, 1)
    dm_pop_ots = load_pop(["Switzerland", "Vaud"], years_list=years_ots)

    dm_pkm_cap, dm_pkm, dm_vkm = run(dm_pop_ots, years_ots)
