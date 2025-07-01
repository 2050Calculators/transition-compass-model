import pandas as pd

from model.common.data_matrix_class import DataMatrix
from model.common.interface_class import Interface
from model.common.constant_data_matrix_class import ConstantDataMatrix
from model.common.io_database import (
    read_database,
    read_database_fxa,
    read_database_to_ots_fts_dict,
)
from model.common.auxiliary_functions import compute_stock, filter_geoscale
from model.common.auxiliary_functions import (
    read_level_data,
    create_years_list,
    linear_fitting,
  filter_country_and_load_data_from_pickles,
)
import pickle
import json
import os
import numpy as np
import time


def read_data(DM_transport, lever_setting):

    dict_fxa = DM_transport["fxa"]
    dm_freight_tech = dict_fxa["freight_tech"]
    dm_passenger_tech = dict_fxa["passenger_tech"]
    dm_passenger_tech.append(dict_fxa["passenger_vehicle-lifetime"], dim="Variables")
    dm_freight_mode_other = dict_fxa["freight_mode_other"]
    dm_freight_mode_road = dict_fxa["freight_mode_road"]

    # Read fts based on lever_setting
    DM_ots_fts = read_level_data(DM_transport, lever_setting)

    # PASSENGER
    dm_passenger_aviation = DM_ots_fts["passenger_aviation-pkm"]
    dm_passenger_tech.append(
        DM_ots_fts["passenger_veh-efficiency_new"], dim="Variables"
    )
    dm_passenger_tech.append(
        DM_ots_fts["passenger_technology-share_new"], dim="Variables"
    )
    dm_passenger_modal = DM_ots_fts["passenger_modal-share"]
    dm_passenger = DM_ots_fts["passenger_occupancy"]
    dm_passenger.append(DM_ots_fts["passenger_utilization-rate"], dim="Variables")

    # FREIGHT
    dm_freight_tech.append(
        DM_ots_fts["freight_vehicle-efficiency_new"], dim="Variables"
    )
    dm_freight_tech.append(DM_ots_fts["freight_technology-share_new"], dim="Variables")
    dm_freight_mode_road.append(DM_ots_fts["freight_utilization-rate"], dim="Variables")
    dm_freight_modal_share = DM_ots_fts["freight_modal-share"]
    dm_freight_demand = DM_ots_fts["freight_tkm"]
    # OTHER
    dm_fuels = DM_ots_fts["fuel-mix"]

    DM_passenger = {
        "passenger_tech": dm_passenger_tech,
        "passenger_aviation": dm_passenger_aviation,
        "passenger_modal_split": dm_passenger_modal,
        "passenger_all": dm_passenger,
        "passenger_pkm_demand": DM_ots_fts["pkm"],  # pkm/cap
    }

    DM_freight = {
        "freight_tech": dm_freight_tech,
        "freight_mode_other": dm_freight_mode_other,
        "freight_mode_road": dm_freight_mode_road,
        "freight_demand": dm_freight_demand,
        "freight_modal_split": dm_freight_modal_share,
    }

    DM_other = {
        "fuels": dm_fuels,
        "electricity-emissions": DM_transport["fxa"]["emission-factor-electricity"],
    }

    cdm_const = DM_transport["constant"]

    return DM_passenger, DM_freight, DM_other, cdm_const


def simulate_lifestyles_input():
    # Read input from lifestyle
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(
        current_file_directory,
        "../_database/data/xls/All-Countries-interface_from-lifestyles-to-transport.xlsx",
    )
    df = pd.read_excel(f, sheet_name="default")
    dm = DataMatrix.create_from_df(df, num_cat=0)
    dm_pop = dm.filter_w_regex({"Variables": "lfs_pop.*"})
    dm_passenger_demand = dm.filter_w_regex(
        {"Variables": "lfs_passenger-travel-demand.*"}
    )
    dm_passenger_demand.deepen()
    DM = {"lfs_pop": dm_pop, "lfs_passenger_demand": dm_passenger_demand}
    return DM


def compute_pkm_demand(modal_split, urb_nonurb_demand):
    # It take the datamatrix with the modal split for urban and non urban
    # and it multiplies it with the demand from lifestyle to return the demand in pkm
    idx_d = urb_nonurb_demand.idx
    idx_m = modal_split.idx
    tmp_nonurb = (
        urb_nonurb_demand.array[:, :, :, idx_d["non-urban"], np.newaxis]
        * modal_split.array[:, :, :, idx_m["nonurban"], :]
    )
    tmp_urb = (
        urb_nonurb_demand.array[:, :, :, idx_d["urban"], np.newaxis]
        * modal_split.array[:, :, :, idx_m["urban"], :]
    )
    tmp_demand = tmp_nonurb + tmp_urb
    cols = {
        "Country": modal_split.col_labels["Country"].copy(),
        "Years": modal_split.col_labels["Years"].copy(),
        "Variables": ["tra_passenger_transport-demand"],
        "Categories1": modal_split.col_labels["Categories2"].copy(),
    }
    demand = DataMatrix(
        col_labels=cols, units={"tra_passenger_transport-demand": "pkm"}
    )
    demand.array = tmp_demand
    return demand


def evaluate_at_tmlife(dm, tmlife_arr, col_name):
    """
    Extracts values from a multi-dimensional data array (Country, Years, Variables, Categories1, Categories2)
    for a specific variable at years defined in `tmlife_arr` for each (Country, Categories1, Categories2) combination.

    Returns:
        np.ndarray:
            A 3D array of shape (Country, Categories1, ...) containing the extracted values.
    """
    # Step 1: Build the shape for the output array
    # This excludes the dimensions corresponding to "Years" and "Variables"
    exclude_axes = {dm.dim_labels.index("Years"), dm.dim_labels.index("Variables")}
    arr_shape = [
        size for ax, size in enumerate(dm.array.shape) if ax not in exclude_axes
    ]
    out = np.full(tuple(arr_shape), np.nan)

    idx = dm.idx
    # Step 2: Iterate through all (Country, Categories1) pairs
    for country in dm.col_labels["Country"]:  # Iterate over countries
        for cat1 in dm.col_labels["Categories1"]:  # Iterate over Categories1
            # Step 2.1: Get the year of interest for this (Country, Categories1) from `tmlife_arr`
            year_value = tmlife_arr[idx[country], idx[cat1]]

            # Step 2.2: Extract the values from dm.array for the given indices
            # Fix indices: select (Country, year_index, idx_var, Categories1, :) -> across Categories2
            out[idx[country], idx[cat1], ...] = dm.array[
                idx[country], idx[year_value], idx[col_name], idx[cat1], ...
            ]

    return out


def compute_fts_tech_split(dm_mode, dm_tech, cols):

    rr_col = cols["renewal-rate"]
    tot_col = cols["tot"]
    waste_col = cols["waste"]
    tech_tot_col = cols["tech_tot"]
    eff_tot_col = cols["eff_tot"]
    eff_new_col = cols["eff_new"]
    new_col = cols["new"]

    startyear = dm_mode.col_labels["Years"][0]
    idx_m = dm_mode.idx
    idx_t = dm_tech.idx

    years_orig = dm_mode.col_labels["Years"].copy()
    years_all = set(range(years_orig[0], years_orig[-1] + 1))
    missing_years = list(years_all - set(years_orig))
    dm_mode.add(np.nan, dim="Years", col_label=missing_years, dummy=True)
    dm_tech.add(np.nan, dim="Years", col_label=missing_years, dummy=True)
    dm_mode.sort("Years")
    dm_tech.sort("Years")
    dm_mode.fill_nans("Years")
    dm_tech.fill_nans("Years")
    years_compute = list(range(2010, dm_mode.col_labels["Years"][-1] + 1))

    for t in years_compute:
        # Vehicle fleet at t-1 (e.g. if t = 2020 and tmn = 2015, t-1 = 2019)
        tot_tm1 = (
            dm_mode.array[:, idx_m[t - 1], idx_m[tot_col], :, np.newaxis]
            * dm_tech.array[:, idx_t[t - 1], idx_t[tech_tot_col], ...]
        )

        # Compute the year = t - round(1/RR(t-1)) = t - lifetime
        rr_tm1 = dm_mode.array[:, idx_m[t - 1], idx_m[rr_col], ...]
        tmlife = np.where(rr_tm1 > 0, t - (1 / rr_tm1).astype(int), startyear)
        tmlife[tmlife < startyear] = startyear  # do not allow years before start year

        # The tech share at tmlife corresponds to the tech share of waste at time t
        tech_share_waste_t = evaluate_at_tmlife(dm_tech, tmlife, tech_tot_col)
        eff_waste_t = evaluate_at_tmlife(dm_tech, tmlife, eff_tot_col)

        # Recompute waste by technology type
        waste_t = (
            dm_mode.array[:, idx_m[t], idx_m[waste_col], :, np.newaxis]
            * tech_share_waste_t
        )
        new_t = dm_tech.array[:, idx_t[t], idx_t[new_col], ...]
        # tot(t) = tot(t-1) + new(t) - waste(t)
        tot_t = tot_tm1 + new_t - waste_t

        # Deal with negative numbers
        mask = tot_t < 0
        tot_t[mask] = 0
        waste_t[mask] = tot_tm1[mask] + new_t[mask]

        # Compute efficiency (eff_tot_t is actually eff_tot_t*tot_t (following lines fix this)
        eff_tot_tmn = dm_tech.array[:, idx_t[t - 1], idx_t[eff_tot_col], ...]
        eff_tot_t = (
            dm_tech.array[:, idx_t[t - 1], idx_t[tot_col], ...] * eff_tot_tmn
            + new_t * dm_tech.array[:, idx_t[t], idx_t[eff_new_col], ...]
            - waste_t * eff_waste_t
        )

        mask = tot_t == 0
        # Re-compute the actual efficiency shares with the exception of when tot_t=0
        eff_tot_t[~mask] = eff_tot_t[~mask] / tot_t[~mask]
        eff_tot_t[mask] = eff_tot_tmn[mask]
        # Re-compute the actual technology shares at time t with the exception of when sum_tot_t = 0
        sum_tot_t = np.nansum(tot_t, axis=-1, keepdims=True)
        tech_tot_t = np.divide(
            tot_t, sum_tot_t, out=np.nan * np.ones_like(tot_t), where=sum_tot_t != 0
        )
        # mask_tech = (np.isnan(tech_tot_t))
        # tech_tot_tmp = dm_tech.array[:, idx_t[t-1], idx_t[tech_tot_col], ...]
        # tech_tot_t[mask_tech] = tech_tot_tmp[mask_tech]
        # Update dm_mode for tot_t and waste_t
        dm_mode.array[:, idx_m[t], idx_m[tot_col], :] = np.nansum(tot_t, axis=-1)
        dm_mode.array[:, idx_m[t], idx_m[waste_col], :] = np.nansum(waste_t, axis=-1)
        # Update dm_tech for tot_t, waste_t, eff_tot_t, tech_tot_t
        dm_tech.array[:, idx_t[t], idx_t[tot_col], ...] = tot_t
        dm_tech.array[:, idx_t[t], idx_t[waste_col], ...] = waste_t
        dm_tech.array[:, idx_t[t], idx_t[eff_tot_col], ...] = eff_tot_t
        dm_tech.array[:, idx_t[t], idx_t[tech_tot_col], ...] = tech_tot_t

    dm_mode.filter({"Years": years_orig}, inplace=True)
    dm_tech.filter({"Years": years_orig}, inplace=True)
    return


def add_biofuel_efuel(dm_energy, dm_fuel_shares, mapping_cat):
    # Compute the biofuel and efuel from the demand of PHEV and ICE
    # and outputs them in a new dataframe together with the energy demand
    dm_energy_ICE_PHEV = dm_energy.filter_w_regex({"Categories2": "PHEV.*|ICE.*"})
    idx_f = dm_fuel_shares.idx
    iter = 0
    for cat in mapping_cat.keys():
        # Filter categories in dm_all that correspond to the group category in dm_fuel_shares. e.g. road : LDV, 2W, bus etc
        # Biofuel
        dm_cat = dm_energy_ICE_PHEV.filter({"Categories1": mapping_cat[cat]})
        dm_biofuel_cat = dm_cat.copy()
        dm_biofuel_cat.array = (
            dm_cat.array
            * dm_fuel_shares.array[
                :,
                :,
                0,
                idx_f["biofuel"],
                idx_f[cat],
                np.newaxis,
                np.newaxis,
                np.newaxis,
            ]
        )
        dm_biofuel_cat.col_labels["Categories2"] = [
            c + "bio" for c in dm_biofuel_cat.col_labels["Categories2"]
        ]
        if iter == 0:
            dm_biofuel = dm_biofuel_cat.copy()
        else:
            dm_biofuel.append(dm_biofuel_cat, dim="Categories1")
        # Efuel
        dm_efuel_cat = dm_cat.copy()
        dm_efuel_cat.array = (
            dm_cat.array
            * dm_fuel_shares.array[
                :, :, 0, idx_f["efuel"], idx_f[cat], np.newaxis, np.newaxis, np.newaxis
            ]
        )
        dm_efuel_cat.col_labels["Categories2"] = [
            c + "efuel" for c in dm_efuel_cat.col_labels["Categories2"]
        ]
        if iter == 0:
            dm_efuel = dm_efuel_cat.copy()
        else:
            dm_efuel.append(dm_efuel_cat, dim="Categories1")
        iter = iter + 1

    # Add biofuel and efuel from standard fuel demand to avoid double counting
    dm_energy.append(dm_efuel, dim="Categories2")
    dm_energy.append(dm_biofuel, dim="Categories2")
    # Remove biofuel and efuel from standard fuel demand to avoid double counting
    idx_e = dm_energy.idx
    i = 0
    for c in dm_energy_ICE_PHEV.col_labels["Categories2"]:
        dm_energy.array[..., idx_e[c]] = (
            dm_energy.array[..., idx_e[c]]
            - dm_efuel.array[..., i]
            - dm_biofuel.array[..., i]
        )
        i = i + 1

    return


def rename_and_group(dm_new_cat, groups, dict_end, grouped_var="tra_total-energy"):

    # Sum columns using the same fuel
    i = 0
    for fuel in groups:
        fuel_str = ".*" + fuel
        tmp = dm_new_cat.filter_w_regex({"Categories1": fuel_str})
        tmp_cat1 = np.nansum(tmp.array, axis=-1, keepdims=True)
        if i == 0:
            i = i + 1
            arr = tmp_cat1
        else:
            arr = np.concatenate([arr, tmp_cat1], axis=-1)
    dm_total_energy = DataMatrix(
        col_labels={
            "Country": dm_new_cat.col_labels["Country"],
            "Years": dm_new_cat.col_labels["Years"],
            "Variables": [grouped_var],
            "Categories1": groups,
        },
        units={grouped_var: list(dm_new_cat.units.values())[0]},
    )

    dm_total_energy.array = arr

    for substring, replacement in dict_end.items():
        dm_new_cat.rename_col_regex(substring, replacement, dim="Categories1")
    for substring, replacement in dict_end.items():
        dm_total_energy.rename_col_regex(substring, replacement, dim="Categories1")

    dm_new_cat.deepen()
    for cat in dm_new_cat.col_labels["Categories2"]:
        if "-" not in cat:
            dm_new_cat.rename_col(cat, "none-" + cat, dim="Categories2")
    dm_new_cat.rename_col_regex("-", "_", dim="Categories2")
    dm_new_cat.deepen()

    return dm_total_energy


def compute_stock_from_lifetime(dm_mode, dm_tech, var_names, years_setting):
    # dm_mode contains the vehicle-fleet (fts) by mode (computed from demand, occupancy etc)
    # dm_tech contains: the tech-share of the vehicle fleet (ots), the new vehicles (ots), the vehicle waste (ots),
    #   the lifetime (fts), the tech-share of the new vehicles (fts), the efficiency of new (fts) and of the fleet (ots)

    # I need to do a for loop over the fts years (every year):
    #     - STEP0: fill_nans to have all the fts years
    #     - STEP1: compute waste(t) = new(t-lifetime) by tech
    #     - STEPx: compute eff_w(t) = eff_n(t-lifetime) by tech
    #     - STEP2: compute new(t) = s(t) - s(t-1) + waste(t) by mode
    #     - STEP3: compute new(t) by tech (using new tech shares(t)
    #     - STEP4: compute s(t) = s(t-1) + new(t) - waste(t) by tech
    #     - STEP6: compute eff_s(t) = eff_s(t-1) * s(t-1) + eff_n(t) * new(t) + eff_w(t) * w(t)

    stock_col = var_names["stock"]
    lifetime_col = var_names["lifetime"]
    new_col = var_names["new"]
    waste_col = var_names["waste"]
    eff_new_col = var_names["eff-new"]  # New efficiency
    eff_stock_col = var_names["eff-stock"]
    tech_new_col = var_names["tech-new"]  # New technology shares
    tech_stock_col = var_names["tech-stock"]

    # section STEP0: fill_nans to have all the fts years
    years_orig = dm_mode.col_labels["Years"].copy()
    years_all = set(range(years_setting[0], years_setting[-2]))
    missing_years = list(years_all - set(years_orig))
    dm_stock = dm_mode.filter({"Variables": [stock_col]})
    dm_stock.add(np.nan, dim="Years", col_label=missing_years, dummy=True)
    dm_stock.sort("Years")
    dm_stock.fill_nans("Years")
    dm_tech.add(np.nan, dim="Years", col_label=missing_years, dummy=True)
    dm_tech.sort("Years")
    dm_tech.fill_nans("Years")

    idx_s = dm_stock.idx
    idx_t = dm_tech.idx

    # Create stock_col in dm_tech
    arr = (
        dm_stock.array[:, :, idx_s[stock_col], :, np.newaxis]
        * dm_tech.array[:, :, idx_t[tech_stock_col], :, :]
    )
    dm_tech.add(
        arr, dim="Variables", col_label=stock_col, unit=dm_stock.units[stock_col]
    )

    startyear = years_setting[0]

    # Future years should include 2024
    future_years = list(range(years_setting[1] + 1, years_setting[-2] + 1))
    for t in future_years:
        # section STEP1: compute waste(t) = new(t-lifetime) by tech
        # Compute the year = t - lifetime
        lifetime = dm_tech.array[:, idx_t[t], idx_t[lifetime_col], ...]
        mask = np.isnan(lifetime)
        tmlife = t - lifetime
        tmlife[~mask] = tmlife[~mask].astype(int)
        tmlife[mask] = (
            startyear  # dummy set tmlife = startyear when nan (this happens for the categories that do not exist)
        )
        tmlife[tmlife < startyear] = startyear  # do not allow years before start year
        idx_tmlife = np.vectorize(idx_t.get)(tmlife)
        # The tech share at tmlife corresponds to the tech share of waste at time t
        # Generate index arrays for country, category 1, and category 2
        country_idx, cat1_idx, cat2_idx = np.indices(np.shape(tmlife))
        # Use advanced indexing to extract new(t-lifetime)
        new_tmlife = dm_tech.array[
            country_idx, idx_tmlife, idx_t[new_col], cat1_idx, cat2_idx
        ]
        # Sanity check
        # waste_t = min(new(t-lifetime), stock(t-1)), as the waste can't be higher than what you have in stock
        s_tm1 = dm_tech.array[:, idx_t[t - 1], idx_t[stock_col], :, :]
        # check for new_tmlife > s_tm1 (you want to throw away cars but there aren't any left)
        waste_t = np.minimum(new_tmlife, s_tm1)
        dm_tech.array[:, idx_t[t], idx_t[waste_col], ...] = waste_t

        # section STEPx: compute eff_w(t) =  waste(t) * eff_n(t-lifetime) + extra_waste(t) * (eff_s(t-1)*s_tm1 - eff_n(t-lifetime)  by tech
        # This is wrong now
        eff_new_tmlife = dm_tech.array[
            country_idx, idx_tmlife, idx_t[eff_new_col], cat1_idx, cat2_idx
        ]
        eff_waste_t = eff_new_tmlife.copy()
        eff_stock_tm1 = dm_tech.array[:, idx_t[t - 1], idx_t[eff_stock_col], :, :]
        # if (eff_waste_t < 0).any():
        #    print('Problem')
        # if (eff_stock_tm1 < 0).any():
        #    print('Problem')

        # This is so that vehicle fleet matches the demand.
        # section STEP2: compute new(t) = s(t) - s(t-1) + waste(t) by mode
        waste_t_mode = np.nansum(waste_t, axis=-1)
        new_t_mode = (
            dm_stock.array[:, idx_s[t], idx_s[stock_col], :]
            - dm_stock.array[:, idx_s[t - 1], idx_s[stock_col], :]
            + waste_t_mode
        )
        # If the demand is drastically reduced, you need to throw away vehicles earlier than their lifetime
        mask = new_t_mode < 0
        if mask.any():
            new_t_mode[mask] = 0
            # delta_s[mask] is the new waste
            delta_s = (
                dm_stock.array[:, idx_s[t - 1], idx_s[stock_col], :]
                - dm_stock.array[:, idx_s[t], idx_s[stock_col], :]
            )
            extra_waste_mode = 0 * waste_t_mode
            extra_waste_mode[mask] = delta_s[mask] - waste_t_mode[mask]
            waste_t_mode[mask] = delta_s[mask]
            # assign the exrta waste based on the technology split of s_tm1 - waste_t
            denominator = np.nansum((s_tm1 - waste_t), axis=-1, keepdims=True)
            denominator[denominator == 0] = 1
            tech_share_extra_waste = (s_tm1 - waste_t) / denominator
            eff_extra_waste = eff_waste_t.copy()
            mask2 = s_tm1 - waste_t > 0
            eff_extra_waste[mask2] = (
                s_tm1[mask2] * eff_stock_tm1[mask2]
                - waste_t[mask2] * eff_waste_t[mask2]
            ) / (s_tm1[mask2] - waste_t[mask2])
            extra_waste_t = extra_waste_mode[..., np.newaxis] * tech_share_extra_waste
            mask1 = waste_t + extra_waste_t > 0
            eff_waste_t_tmp = eff_waste_t.copy()
            eff_waste_t_tmp[mask1] = (
                waste_t[mask1] * eff_waste_t[mask1]
                + extra_waste_t[mask1] * eff_extra_waste[mask1]
            ) / (waste_t[mask1] + extra_waste_t[mask1])
            eff_waste_t_tmp = np.maximum(
                eff_waste_t_tmp, np.minimum(eff_waste_t, extra_waste_t)
            )
            eff_waste_t_tmp = np.minimum(
                eff_waste_t_tmp, np.maximum(eff_waste_t, extra_waste_t)
            )
            eff_waste_t = eff_waste_t_tmp
            waste_t = waste_t + extra_waste_t

        # if (eff_waste_t < 0).any():
        #    print('Problem')

        # section STEP3: compute new(t) by tech (using new tech shares(t))
        new_t = (
            new_t_mode[..., np.newaxis]
            * dm_tech.array[:, idx_t[t], idx_t[tech_new_col], :, :]
        )
        dm_tech.array[:, idx_t[t], idx_t[new_col], :, :] = new_t

        # section STEP4: compute s(t) = s(t-1) + new(t) - waste(t) by tech
        s_t = s_tm1 + new_t - waste_t
        dm_tech.array[:, idx_t[t], idx_t[stock_col], :, :] = s_t
        dm_tech.array[:, idx_t[t], idx_t[tech_stock_col], :, :] = s_t / np.nansum(
            s_t, axis=-1, keepdims=True
        )

        # section STEP6: compute eff_s(t) = (eff_s(t-1) * s(t-1) + eff_n(t) * new(t) - eff_w(t) * w(t))/s(t)
        eff_new_t = dm_tech.array[:, idx_t[t], idx_t[eff_new_col], :, :]
        mask = s_t > 0
        eff_stock_t = eff_stock_tm1.copy()  # If stock = 0 then eff(t) = eff(t-1)
        eff_stock_t[mask] = (
            eff_stock_tm1[mask] * s_tm1[mask]
            + eff_new_t[mask] * new_t[mask]
            - eff_waste_t[mask] * waste_t[mask]
        ) / s_t[mask]
        # The stock efficiency cannot be lower than the existing efficiency or the new efficiency
        eff_stock_t = np.maximum(eff_stock_t, np.minimum(eff_new_t, eff_stock_tm1))
        eff_stock_t = np.minimum(eff_stock_t, np.maximum(eff_new_t, eff_stock_tm1))
        dm_tech.array[:, idx_t[t], idx_t[eff_stock_col], :, :] = eff_stock_t
        # if (eff_stock_t < 0).any():
        #    print('Problem')

    dm_tech.filter({"Years": years_orig}, inplace=True)

    return dm_tech


def passenger_fleet_energy(DM_passenger, dm_lfs, DM_other, cdm_const, years_setting):
    # SECTION Passenger - Demand-pkm by mode
    # dm_demand_by_mode [pkm] = modal_shares(urban) * demand_pkm(urban) + modal_shares(non-urban) * demand_pkm(non-urban)
    dm_modal_split = DM_passenger["passenger_modal_split"]
    dm_pkm_demand = DM_passenger["passenger_pkm_demand"]  # pkm/cap
    arr = (
        dm_lfs.array[..., np.newaxis]
        * dm_pkm_demand.array[..., np.newaxis]
        * dm_modal_split.array
    )
    dm_demand_by_mode = DataMatrix.based_on(
        arr,
        dm_modal_split,
        change={"Variables": ["tra_passenger_transport-demand"]},
        units={"tra_passenger_transport-demand": "pkm"},
    )
    # dm_demand_by_mode = compute_pkm_demand(dm_modal_split, dm_lfs_demand)
    del dm_modal_split, dm_pkm_demand
    # Remove walking and biking
    dm_demand_soft = dm_demand_by_mode.filter({"Categories1": ["walk", "bike"]})
    dm_demand_soft.rename_col(
        "tra_passenger_transport-demand",
        "tra_passenger_transport-demand-by-mode",
        dim="Variables",
    )
    dm_demand_by_mode.drop(dim="Categories1", col_label="walk|bike")

    # SECTION Passenger - Aviation Demand-pkm
    # demand_aviation [pkm] = demand aviation [pkm/cap] * pop
    dm_aviation_pkm = DM_passenger["passenger_aviation"]
    dm_pop = dm_lfs
    arr_av_pkm = dm_aviation_pkm.array * dm_pop.array[..., np.newaxis]
    dm_aviation_pkm.add(
        arr_av_pkm,
        dim="Variables",
        unit="pkm",
        col_label="tra_passenger_transport-demand-by-mode",
    )
    dm_aviation_pkm.drop(col_label="tra_pkm-cap", dim="Variables")
    # tmp_aviation = dm_aviation_pkm.array[..., 0] * dm_pop.array[...]
    # dm_demand_by_mode.add(tmp_aviation, dim='Categories1', col_label='aviation')
    del arr_av_pkm

    dm_demand = dm_demand_by_mode.filter(
        {"Categories1": ["LDV", "2W", "bus", "metrotram", "rail"]}
    )
    dm_mode = DM_passenger["passenger_all"]
    dm_mode.append(dm_demand, dim="Variables")
    del dm_demand

    # SECTION Passenger - Demand-vkm by mode
    # demand [vkm] = demand [pkm] / occupancy [pkm/vkm]
    dm_mode.operation('tra_passenger_transport-demand', '/', 'tra_passenger_occupancy',
                      dim="Variables", out_col='tra_passenger_transport-demand-vkm', unit='vkm', div0="error")

    # SECTION Passenger - Vehicle-fleet by mode
    # vehicle-fleet [number] = demand [vkm] / utilisation-rate [vkm/veh/year]
    dm_mode.operation(
        "tra_passenger_transport-demand-vkm",
        "/",
        "tra_passenger_utilisation-rate",
        dim="Variables",
        out_col="tra_passenger_vehicle-fleet",
        unit="number",
        div0="error",
        type=int,
    )

    # SECTION Passenger - Vehicle fleet, new-vehicle, vehicle-waste, efficiency by technology
    dm_tech = DM_passenger["passenger_tech"]
    #
    cols = {
        "lifetime": "tra_passenger_lifetime",
        "stock": "tra_passenger_vehicle-fleet",
        "waste": "tra_passenger_vehicle-waste",
        "new": "tra_passenger_new-vehicles",
        "tech-new": "tra_passenger_technology-share_new",
        "tech-stock": "tra_passenger_technology-share_fleet",
        "eff-stock": "tra_passenger_veh-efficiency_fleet",
        "eff-new": "tra_passenger_veh-efficiency_new",
    }
    dm_tech = compute_stock_from_lifetime(
        dm_mode, dm_tech, var_names=cols, years_setting=years_setting
    )

    # SECTION Passenger - New-vehicles by mode
    dm_new_veh = dm_tech.filter({"Variables": [cols["new"]]})
    dm_mode.append(dm_new_veh.group_all("Categories2", inplace=False), dim="Variables")

    # SECTION Passenger - Energy demand
    # Energy demand [MJ] = Transport demand [vkm] x efficiency [MJ/vkm]
    # Extract passenger transport demand vkm for road and pkm for others,
    # join and compute transport demand by technology
    dm_demand_vkm = dm_mode.filter(
        selected_cols={"Variables": ["tra_passenger_transport-demand-vkm"]}
    )
    dm_demand_vkm.sort(dim="Categories1")
    dm_tech.sort(dim="Categories1")
    idx_t = dm_tech.idx
    tmp = (
        dm_demand_vkm.array[:, :, 0, :, np.newaxis]
        * dm_tech.array[:, :, idx_t["tra_passenger_technology-share_fleet"], ...]
    )
    dm_tech.add(
        tmp, dim="Variables", col_label="tra_passenger_transport-demand-vkm", unit="km"
    )
    del tmp
    # Compute energy consumption
    dm_tech.operation(
        "tra_passenger_veh-efficiency_fleet",
        "*",
        "tra_passenger_transport-demand-vkm",
        out_col="tra_passenger_energy-demand",
        unit="MJ",
    )

    # SECTION Passenger - e-fuel and bio-fuel
    # Add e-fuel and bio-fuel to energy consumption
    dm_fuel = DM_other["fuels"].copy()
    dm_fuel.drop(col_label="aviation", dim="Categories2")
    mapping_cat = {"road": ["LDV", "2W", "rail", "metrotram", "bus"]}
    dm_energy = dm_tech.filter({"Variables": ["tra_passenger_energy-demand"]})
    add_biofuel_efuel(dm_energy, dm_fuel, mapping_cat)

    # SECTION Passenger - GHG Emissions fossil
    # Compute emissions by fuel for fossil fuels, mode, GHG
    cdm_const.drop(col_label=["marinefueloil", "kerosene"], dim="Categories2")
    dm_energy_fossil = dm_energy.copy()
    dm_energy_fossil.filter(
        {"Categories2": cdm_const.col_labels["Categories2"]}, inplace=True
    )
    dm_energy_fossil.sort("Categories2")
    cdm_const.sort("Categories2")
    arr_emis_mode_GHG_fuel = (
        dm_energy_fossil.array[:, :, :, :, np.newaxis, :]
        * cdm_const.array[np.newaxis, np.newaxis, :, np.newaxis, :, :]
    )
    arr_emis_mode_GHG_fuel = np.moveaxis(
        arr_emis_mode_GHG_fuel, -2, -1
    )  # Move GHG to end
    dm_emissions = DataMatrix.based_on(
        arr_emis_mode_GHG_fuel,
        dm_energy_fossil,
        change={
            "Variables": ["tra_passenger_emissions"],
            "Categories3": cdm_const.col_labels["Categories1"],
        },
        units={"tra_passenger_emissions": "g"},
    )

    # SECTION Passenger - GHG Emissions EV
    # Compute emissions from electricity
    dm_energy_EV = dm_energy.filter({"Categories2": ["BEV", "CEV"]})
    dm_fact = DM_other["electricity-emissions"]
    idx_f = dm_fact.idx
    idx_e = dm_energy_EV.idx
    arr = (
        dm_energy_EV.array[:, :, idx_e["tra_passenger_energy-demand"], :, :, np.newaxis]
        * dm_fact.array[
            :,
            :,
            idx_f["tra_emission-factor"],
            np.newaxis,
            np.newaxis,
            :,
            idx_f["electricity"],
        ]
    )
    dm_emissions_elec = DataMatrix.based_on(
        arr[:, :, np.newaxis, ...],
        dm_energy_EV,
        change={
            "Variables": ["tra_passenger_emissions"],
            "Categories3": dm_fact.col_labels["Categories1"],
        },
        units={"tra_passenger_emissions": "g"},
    )
    dm_emissions.append(dm_emissions_elec, dim="Categories2")
    dm_emissions.change_unit("tra_passenger_emissions", 1e-12, "g", "Mt")

    # SECTION Prepare output
    dm_emissions_by_mode = dm_emissions.group_all("Categories2", inplace=False)
    dm_emissions_by_fuel = dm_emissions.group_all("Categories1", inplace=False)
    dm_emissions_by_fuel.groupby(
        {
            "diesel": ".*diesel",
            "gasoline": ".*gasoline",
            "gas": ".*gas",
            "electricity": ".*EV",
        },
        dim="Categories1",
        regex=True,
        inplace=True,
    )
    dm_emissions_by_GHG = dm_emissions.group_all("Categories2", inplace=False)
    dm_emissions_by_GHG.group_all("Categories1")

    dm_energy.change_unit(
        "tra_passenger_energy-demand", 2.77778e-10, old_unit="MJ", new_unit="TWh"
    )

    # Deal with PHEV and electricity. For each mode of transport,
    # sum PHEV energy demand and multiply it by 0.1 to obtain a new category, the PHEV_elec
    dm_energy_phev = dm_energy.filter_w_regex(
        {"Variables": "tra_passenger_energy-demand", "Categories2": "PHEV.*"}
    )
    PHEV_elec = 0.1 * np.nansum(dm_energy_phev.array, axis=-1)
    dm_energy.add(PHEV_elec, dim="Categories2", col_label="PHEV-elec")

    # Rename and group fuel types
    dm_energy.groupby(
        {
            "biodiesel": ".*dieselbio",
            "biogas": ".*gasbio",
            "biogasoline": ".*gasolinebio",
            "efuel": ".*efuel",
        },
        dim="Categories2",
        regex=True,
        inplace=True,
    )
    dm_energy.groupby(
        {
            "diesel": ".*-diesel",
            "gasoline": ".*-gasoline",
            "hydrogen": "FCEV",
            "gas": ".*-gas",
            "electricity": "BEV|CEV|PHEV-elec|mt",
        },
        dim="Categories2",
        regex=True,
        inplace=True,
    )

    # Compute energy demand by mode
    idx = dm_energy.idx
    tmp = np.nansum(
        dm_energy.array[:, :, idx["tra_passenger_energy-demand"], :, :], axis=-1
    )
    dm_mode.add(
        tmp,
        dim="Variables",
        col_label="tra_passenger_energy-demand-by-mode",
        unit="TWh",
    )

    # Prepare output for energy
    dm_electricity = dm_energy.filter({"Categories2": ["electricity"]})
    dm_electricity.group_all(dim="Categories2")
    dm_electricity.groupby(
        {"road": ["2W", "bus", "LDV"], "rail": ["metrotram", "rail"]},
        dim="Categories1",
        inplace=True,
    )
    dm_electricity.rename_col(
        "tra_passenger_energy-demand", "tra_power-demand", dim="Variables"
    )

    # Drop mode split in energy
    dm_energy.group_all("Categories1")

    # Prepare efuel output
    dm_efuel = dm_energy.filter({"Categories1": ["efuel"]})
    dm_efuel.rename_col(
        "tra_passenger_energy-demand", "tra_power-demand", dim="Variables"
    )

    dm_electricity.append(dm_efuel, dim="Categories1")
    dm_electricity.change_unit("tra_power-demand", 1e3, old_unit="TWh", new_unit="GWh")

    DM_passenger_out = {
        "power": {"electricity": dm_electricity.flatten()},
    }

    # Power output (tra_power_demand _ hydrogen)
    dm_pow_hydrogen = dm_energy.filter({"Categories1": ["hydrogen"]})
    dm_pow_hydrogen.rename_col(
        "tra_passenger_energy-demand", "tra_power-demand", dim="Variables"
    )
    dm_pow_hydrogen.change_unit(
        "tra_power-demand", factor=1e3, old_unit="TWh", new_unit="GWh"
    )
    DM_passenger_out["power"]["hydrogen"] = dm_pow_hydrogen.flatten()

    DM_passenger_out["oil-refinery"] = dm_energy.filter(
        {"Categories1": ["gasoline", "diesel", "gas"]}
    )

    # (tra_power_demand _ hydrogen)
    dm_biogas = dm_energy.filter({"Categories1": ["biogas"]})

    dm_tech.rename_col(
        "tra_passenger_technology-share_fleet",
        "tra_passenger_technology-share-fleet",
        dim="Variables",
    )

    # Compute passenger demand by mode
    # dm_mode.append(dm_demand_by_mode, dim='Variables')
    dm_mode.rename_col(
        ["tra_passenger_transport-demand"],
        ["tra_passenger_transport-demand-by-mode"],
        dim="Variables",
    )

    # Add passenger demand vkm to output
    # dm_mode.append(dm_all.filter({'Variables': ['tra_passenger_transport-demand-vkm']}), dim='Variables')

    # Compute CO2 emissions by mode
    idx = dm_emissions_by_mode.idx
    tmp = dm_emissions_by_mode.array[
        :, :, idx["tra_passenger_emissions"], :, idx["CO2"]
    ]
    dm_mode.add(
        tmp, dim="Variables", col_label="tra_passenger_emissions-by-mode_CO2", unit="Mt"
    )

    dm_energy.rename_col(
        col_in="tra_passenger_energy-demand",
        col_out="tra_passenger_energy-demand-by-fuel",
        dim="Variables",
    )

    DM_passenger_out["mode"] = dm_mode
    DM_passenger_out["tech"] = dm_tech
    DM_passenger_out["fuel"] = dm_fuel
    DM_passenger_out["agriculture"] = dm_biogas
    DM_passenger_out["emissions"] = dm_emissions_by_mode
    DM_passenger_out["energy"] = dm_energy
    DM_passenger_out["soft-mobility"] = dm_demand_soft
    DM_passenger_out["aviation"] = dm_aviation_pkm

    return DM_passenger_out


def freight_fleet_energy(DM_freight, DM_other, cdm_const, years_setting):
    # FREIGHT
    dm_tkm = DM_freight["freight_demand"]
    dm_mode = DM_freight["freight_modal_split"]
    # From bn tkm to tkm by mode of transport
    tmp = dm_tkm.array[:, :, 0, np.newaxis] * 1e9 * dm_mode.array[:, :, 0, :]
    dm_mode.add(
        tmp, dim="Variables", col_label="tra_freight_transport-demand", unit="tkm"
    )

    dm_mode_road = DM_freight["freight_mode_road"]
    dm_mode_road.append(
        dm_mode.filter_w_regex(dict_dim_pattern={"Categories1": "HDV.*"}),
        dim="Variables",
    )
    dm_mode_road.operation(
        "tra_freight_transport-demand",
        "/",
        "tra_freight_load-factor",
        out_col="tra_freight_transport-demand-vkm",
        unit="vkm",
    )
    dm_mode_road.operation(
        "tra_freight_transport-demand-vkm",
        "/",
        "tra_freight_utilisation-rate",
        out_col="tra_freight_vehicle-fleet",
        unit="number",
    )
    dm_mode_road.operation(
        "tra_freight_utilisation-rate",
        "/",
        "tra_freight_lifetime",
        out_col="tra_freight_renewal-rate",
        unit="%",
    )

    dm_mode_other = DM_freight["freight_mode_other"]
    dm_mode_other.append(
        dm_mode.filter({"Categories1": ["IWW", "marine", "aviation", "rail"]}),
        dim="Variables",
    )
    dm_mode_other.operation(
        "tra_freight_transport-demand",
        "/",
        "tra_freight_tkm-by-veh",
        out_col="tra_freight_vehicle-fleet",
        unit="number",
    )

    # Compute vehicle waste and new vehicles for both road and other
    dm_other_tmp = dm_mode_other.filter_w_regex(
        dict_dim_pattern={"Variables": ".*vehicle-fleet|.*renewal-rate"}
    )
    dm_road_tmp = dm_mode_road.filter_w_regex(
        dict_dim_pattern={"Variables": ".*vehicle-fleet|.*renewal-rate"}
    )
    dm_mode_other.drop(dim="Variables", col_label=".*vehicle-fleet|.*renewal-rate")
    dm_mode_road.drop(dim="Variables", col_label=".*vehicle-fleet|.*renewal-rate")
    dm_road_tmp.append(dm_other_tmp, dim="Categories1")
    dm_mode.append(dm_road_tmp, dim="Variables")
    del dm_road_tmp, dm_other_tmp

    compute_stock(
        dm_mode,
        rr_regex="tra_freight_renewal-rate",
        tot_regex="tra_freight_vehicle-fleet",
        waste_col="tra_freight_vehicle-waste",
        new_col="tra_freight_new-vehicles",
    )

    # Compute fleet by technology type
    dm_tech = DM_freight["freight_tech"]
    idx_t = dm_tech.idx
    idx_m = dm_mode.idx
    tmp_1 = (
        dm_tech.array[:, :, idx_t["tra_freight_technology-share_fleet"], :, :]
        * dm_mode.array[:, :, idx_m["tra_freight_vehicle-fleet"], :, np.newaxis]
    )
    tmp_2 = (
        dm_tech.array[:, :, idx_t["tra_freight_technology-share_fleet"], :, :]
        * dm_mode.array[:, :, idx_m["tra_freight_vehicle-waste"], :, np.newaxis]
    )
    tmp_3 = (
        dm_tech.array[:, :, idx_t["tra_freight_technology-share_new"], :, :]
        * dm_mode.array[:, :, idx_m["tra_freight_new-vehicles"], :, np.newaxis]
    )
    dm_tech.add(
        tmp_1, col_label="tra_freight_vehicle-fleet", dim="Variables", unit="number"
    )
    dm_tech.add(
        tmp_2, col_label="tra_freight_vehicle-waste", dim="Variables", unit="number"
    )
    dm_tech.add(
        tmp_3, col_label="tra_freight_new-vehicles", dim="Variables", unit="number"
    )
    del tmp_1, tmp_2, tmp_3
    #
    cols = {
        "renewal-rate": "tra_freight_renewal-rate",
        "tot": "tra_freight_vehicle-fleet",
        "waste": "tra_freight_vehicle-waste",
        "new": "tra_freight_new-vehicles",
        "tech_tot": "tra_freight_technology-share_fleet",
        "eff_tot": "tra_freight_vehicle-efficiency_fleet",
        "eff_new": "tra_freight_vehicle-efficiency_new",
    }
    compute_fts_tech_split(dm_mode, dm_tech, cols)

    # Extract freight transport demand vkm for road and tkm for others, join and compute transport demand by technology
    dm_demand_km = dm_mode_road.filter(
        selected_cols={"Variables": ["tra_freight_transport-demand-vkm"]}
    )
    dm_demand_km_other = dm_mode_other.filter(
        selected_cols={"Variables": ["tra_freight_transport-demand"]}
    )
    dm_demand_km_other.units["tra_freight_transport-demand"] = "km"
    dm_demand_km.units["tra_freight_transport-demand-vkm"] = "km"
    dm_demand_km.rename_col(
        "tra_freight_transport-demand-vkm",
        "tra_freight_transport-demand",
        dim="Variables",
    )
    dm_demand_km.append(dm_demand_km_other, dim="Categories1")
    dm_demand_km.sort(dim="Categories1")
    idx_t = dm_tech.idx
    tmp = (
        dm_demand_km.array[:, :, 0, :, np.newaxis]
        * dm_tech.array[:, :, idx_t["tra_freight_technology-share_fleet"], ...]
    )
    dm_tech.add(
        tmp, dim="Variables", col_label="tra_freight_transport-demand", unit="km"
    )
    del tmp, dm_demand_km, dm_demand_km_other
    # Compute energy consumption
    dm_tech.operation(
        "tra_freight_vehicle-efficiency_fleet",
        "*",
        "tra_freight_transport-demand",
        out_col="tra_freight_energy-demand",
        unit="MJ",
    )

    # Compute biofuel and efuel and extract energy as standalone dm
    dm_fuel = DM_other["fuels"]
    mapping_cat = {
        "road": ["HDVH", "HDVM", "HDVL"],
        "aviation": ["aviation"],
        "rail": ["rail"],
        "marine": ["IWW", "marine"],
    }
    dm_energy = dm_tech.filter({"Variables": ["tra_freight_energy-demand"]})
    add_biofuel_efuel(dm_energy, dm_fuel, mapping_cat)

    # Deal with PHEV and electricity. For each mode of transport,
    # sum PHEV energy demand and multiply it by 0.1 to obtain a new category, the PHEV_elec
    dm_energy_phev = dm_energy.filter_w_regex(
        {"Variables": "tra_freight_energy-demand", "Categories2": "PHEV.*"}
    )
    PHEV_elec = 0.1 * np.nansum(dm_energy_phev.array, axis=-1)
    dm_energy.add(PHEV_elec, dim="Categories2", col_label="PHEV-elec")

    dm_energy.array = dm_energy.array * 2.77778e-10
    dm_energy.units["tra_freight_energy-demand"] = "TWh"

    dm_energy_by_mode = dm_energy.group_all(dim="Categories2", inplace=False)
    dm_mode.append(dm_energy_by_mode, dim="Variables")

    # Prepare output for energy
    dm_electricity = dm_energy.groupby(
        {"power-demand": ["BEV", "CEV", "PHEV-elec"]}, dim="Categories2"
    )
    dm_electricity.groupby(
        {
            "road": ["HDVH", "HDVL", "HDVM"],
            "rail": ["rail"],
            "other": ["aviation", "marine", "IWW"],
        },
        dim="Categories1",
        inplace=True,
    )
    dm_electricity.switch_categories_order()
    dm_electricity.rename_col("tra_freight_energy-demand", "tra", dim="Variables")
    dm_electricity = dm_electricity.flatten()
    dm_electricity = dm_electricity.flatten()
    dm_electricity.deepen()

    dm_efuel = dm_energy.groupby({"efuel": ".*efuel"}, dim="Categories2", regex=True)
    dm_efuel.groupby(
        {"power-demand": ".*"}, dim="Categories1", inplace=True, regex=True
    )
    dm_efuel.rename_col("tra_freight_energy-demand", "tra", dim="Variables")
    dm_efuel = dm_efuel.flatten()
    dm_efuel = dm_efuel.flatten()
    dm_efuel.deepen()

    dm_electricity.append(dm_efuel, dim="Categories1")
    dm_electricity.change_unit("tra_power-demand", 1e3, old_unit="TWh", new_unit="GWh")

    DM_freight_out = {
        "power": {"electricity": dm_electricity.flatten()},
    }
    ## end
    all_modes = dm_energy.col_labels['Categories1'].copy()
    dm_energy_aviation = dm_energy.filter({'Categories1': ['aviation']})
    dm_energy_marine = dm_energy.filter({'Categories1': ['marine', 'IWW']})
    dm_energy.drop(dim='Categories1', col_label=['marine', 'IWW', 'aviation'])

    # Rename and group fuel types
    dm_energy.groupby({'biodiesel': '.*dieselbio', 'biogas': '.*gasbio',
                       'biogasoline': '.*gasolinebio', 'efuel': '.*efuel'},
                      dim='Categories2', regex=True, inplace=True)
    dm_energy.groupby({'diesel': '.*-diesel', 'gasoline': '.*-gasoline', 'hydrogen': 'FCEV',
                       'gas': '.*-gas', 'electricity': 'BEV|CEV|PHEV-elec|mt'},
                      dim='Categories2', regex=True, inplace=True)

    dm_energy_aviation = dm_energy_aviation.groupby({'ejetfuel': ['ICEefuel'], 'biojetfuel': ['ICEbio'],
                                                     'kerosene': ['ICE']}, inplace=False, dim='Categories2')

    dm_energy_marine.rename_col(['ICEbio', 'ICEefuel', 'ICE'], ['biomarinefueloil', 'emarinefueloil', 'marinefueloil'], dim='Categories2')
    dm_energy_marine.filter({'Categories2': ['biomarinefueloil', 'emarinefueloil', 'marinefueloil']}, inplace=True)

    # Merge
    missing_cat_aviation = list(set(all_modes)-set(dm_energy_aviation.col_labels['Categories1']))
    dm_energy_aviation.add(np.nan, dummy=True, dim='Categories1', col_label=missing_cat_aviation)
    missing_cat_marine = list(set(all_modes) - set(dm_energy_marine.col_labels['Categories1']))
    dm_energy_marine.add(np.nan, dummy=True, dim='Categories1', col_label=missing_cat_marine)
    missing_cat_road = list(set(all_modes) - set(dm_energy.col_labels['Categories1']))
    dm_energy.add(np.nan, dummy=True, dim='Categories1', col_label=missing_cat_road)

    dm_energy.append(dm_energy_marine, dim='Categories2')
    dm_energy.append(dm_energy_aviation, dim='Categories2')

    # Group together by fuel type, drop mode
    dm_total_energy = dm_energy.group_all('Categories1', inplace=False)
    dm_total_energy.rename_col('tra_freight_energy-demand', 'tra_freight_total-energy', dim='Variables')

    # Output to power:
    dm_pow_hydrogen = dm_total_energy.filter({"Categories1": ["hydrogen"]})
    dm_pow_hydrogen.rename_col(
        "tra_freight_total-energy", "tra_power-demand", dim="Variables"
    )
    dm_pow_hydrogen.change_unit("tra_power-demand", 1e3, old_unit="TWh", new_unit="GWh")
    DM_freight_out["power"]["hydrogen"] = dm_pow_hydrogen.flatten()

    # Prepare output to refinery:
    DM_freight_out["oil-refinery"] = dm_total_energy.filter(
        {"Categories1": ["gasoline", "diesel", "marinefueloil", "gas", "kerosene"]}
    )

    dm_biogas = dm_total_energy.filter({"Categories1": ["biogas"]})

    # Compute emission by fuel
    # Filter fuels for which we have emissions
    cdm_const.drop("Categories2", ["PHEV-diesel", "PHEV-gasoline"])
    cdm_const.rename_col("ICE-diesel", "diesel", dim="Categories2")
    cdm_const.rename_col("ICE-gasoline", "gasoline", dim="Categories2")
    cdm_const.rename_col("ICE-gas", "gas", dim="Categories2")

    dm_energy_em = dm_total_energy.filter(
        {"Categories1": cdm_const.col_labels["Categories2"]}
    )
    # Sort categories to make sure they match
    dm_energy_em.sort(dim="Categories1")
    cdm_const.sort(dim="Categories2")
    idx_e = dm_energy_em.idx
    idx_c = cdm_const.idx
    # emissions = energy * emission-factor
    tmp = (
        dm_energy_em.array[:, :, idx_e["tra_freight_total-energy"], np.newaxis, :]
        * cdm_const.array[np.newaxis, np.newaxis, idx_c["cp_tra_emission-factor"], :, :]
    )
    tmp = np.moveaxis(tmp, -2, -1)
    # Save emissions by fuel in a datamatrix
    col_labels = dm_energy_em.col_labels.copy()
    col_labels["Variables"] = ["tra_freight_emissions"]
    col_labels["Categories2"] = cdm_const.col_labels[
        "Categories1"
    ].copy()  # GHG category
    unit = {"tra_freight_emissions": "Mt"}
    dm_emissions_by_fuel = DataMatrix(col_labels=col_labels, units=unit)
    dm_emissions_by_fuel.array = tmp[
        :, :, np.newaxis, :, :
    ]  # The variable dimension was lost when doing nansum
    del dm_energy_em, tmp, col_labels, unit

    # Compute emissions by mode
    dm_energy_em = dm_energy.filter({'Categories2': cdm_const.col_labels['Categories2']})
    dm_energy_em.sort(dim='Categories2')
    cdm_const.sort(dim='Categories2')
    idx_e = dm_energy_em.idx
    idx_c = cdm_const.idx
    tmp = dm_energy_em.array[:, :, idx_e['tra_freight_energy-demand'], :, np.newaxis, :] \
          * cdm_const.array[np.newaxis, np.newaxis, idx_c['cp_tra_emission-factor'], np.newaxis, :, :]
    tmp = np.nansum(tmp, axis=-1)  # Remove split by fuel
    tmp = tmp[:, :, np.newaxis, :, :]
    dm_emissions_by_mode = DataMatrix.based_on(tmp, format=dm_energy_em,
                                               change={'Variables': ['tra_freight_emissions'],
                                                       'Categories2': cdm_const.col_labels['Categories1']},
                                               units={'tra_freight_emissions': 'Mt'})

    del tmp, idx_e, idx_c, dm_energy_em

    tmp = np.nansum(dm_emissions_by_mode.array, axis=-2)
    col_labels = dm_emissions_by_mode.col_labels.copy()
    col_labels["Categories1"] = col_labels["Categories2"].copy()
    col_labels.pop("Categories2")
    unit = dm_emissions_by_mode.units
    dm_emissions_by_GHG = DataMatrix(col_labels=col_labels, units=unit)
    dm_emissions_by_GHG.array = tmp[:, :, np.newaxis, :]
    del tmp, unit, col_labels

    dm_tech.rename_col('tra_freight_technology-share_fleet', 'tra_freight_techology-share-fleet', dim='Variables')

    DM_freight_out['mode'] = dm_mode
    DM_freight_out['tech'] = dm_tech
    DM_freight_out['energy'] = dm_total_energy
    DM_freight_out['agriculture'] = dm_biogas
    DM_freight_out['emissions'] = dm_emissions_by_mode

    return DM_freight_out


def tra_industry_interface(
    dm_freight_veh, dm_passenger_veh, dm_infrastructure, write_pickle=False
):

    # passenger
    # TODO: check with Paola why aviation is not here
    if "aviation" not in dm_passenger_veh.col_labels["Categories1"]:
        dm_passenger_veh.add(
            np.nan, dim="Categories1", col_label="aviation", dummy=True
        )
        dm_passenger_veh.add(np.nan, dim="Categories2", col_label="ICE", dummy=True)
    dm_veh = dm_passenger_veh.copy()
    dm_veh.groupby({"CEV": ["mt", "CEV"]}, "Categories2", inplace=True)
    dm_veh = dm_veh.filter_w_regex({"Categories1": "LDV|aviation|bus|rail"})
    dm_veh.rename_col(["rail", "aviation"], ["trains", "planes"], "Categories1")
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
    dm_veh.append(dm_fre, "Categories1")
    dm_veh.groupby(
        {"trains": ["trains", "rail"], "planes": ["planes", "aviation"]},
        "Categories1",
        inplace=True,
    )
    dm_veh.sort("Categories1")

    # get infrastructure
    dm_infra_ind = dm_infrastructure.copy()
    dm_infra_ind.rename_col_regex("infra-", "", dim="Categories1")
    dm_infra_ind.rename_col(
        "tra_new_infrastructure", "tra_product-demand", dim="Variables"
    )

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
            "../_database/data/interface/transport_to_industry.pickle",
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
            "../_database/data/interface/transport_to_minerals.pickle",
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
            "../_database/data/interface/transport_to_oil-refinery.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(dm_tot_energy, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm_tot_energy


def prepare_TPE_output(DM_passenger_out, DM_freight_out):

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

    dm_keep_tech = DM_passenger_out["tech"].filter(
        {"Variables": ["tra_passenger_vehicle-fleet"], "Categories1": ["LDV"]}
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

    dm_freight_energy_by_fuel = DM_freight_out['energy'].copy()
    dm_freight_energy_by_fuel.drop(dim='Categories1', col_label=['efuel', 'ejetfuel'])
    dm_freight_energy_by_fuel.rename_col('tra_freight_total-energy', 'tra_freight_energy-demand-by-fuel', dim='Variables')


    # Total energy demand
    dm_energy_tot = DM_passenger_out["energy"].copy()
    dm_energy_tot.group_all(dim="Categories1")
    dm_energy_freight = DM_freight_out["energy"].copy()
    dm_energy_freight.group_all(dim="Categories1")
    dm_energy_tot.append(dm_energy_freight, dim="Variables")
    dm_energy_tot.groupby(
        {"tra_energy-demand_total": ".*"}, inplace=True, regex=True, dim="Variables"
    )

    # Merge datamatrices for new-app
    dm_tpe = dm_keep_mode.flattest()
    dm_tpe.append(dm_keep_tech.flattest(), dim="Variables")
    dm_tpe.append(dm_keep_fuel.flattest(), dim="Variables")
    dm_tpe.append(dm_keep_energy.flattest(), dim="Variables")
    dm_tpe.append(dm_freight_energy_by_mode.flattest(), dim="Variables")
    dm_tpe.append(dm_energy_tot.flattest(), dim="Variables")
    dm_tpe.append(dm_freight_energy_by_fuel.flattest(), dim="Variables")
    dm_tpe.append(DM_passenger_out["soft-mobility"].flattest(), dim="Variables")
    dm_tpe.append(DM_passenger_out["emissions"].flattest(), dim="Variables")

    return dm_tpe


# !FIXME: infrastructure dummy not OK, find real tot infrastructure data and real renewal-rates or new-infrastructure
def dummy_tra_infrastructure_workflow(dm_pop):

    # Industry and Minerals need the new infrastructure in km for rails, roads, and trolley-cables
    # In order to compute the new infrastructure we need the tot infrastructure and a renewal-rate
    # tot_infrastructure = Looking at Swiss data it looks like there are around 10 m of road per capita
    # (Longueurs des routes nationales, cantonales et des autres routes ouvertes aux véhicules à moteur selon le canton)
    # and 0.6 m of rail per capita and 0.0017 of trolley-bus, I'm using this approximation for all countries
    # for the renewal rate eucalc was using 5%, which correspond to a resurfacing every 20 years. I use this for road
    # for rails I use 2.5% (40 years lifetime). For the wires I have no idea,
    # I'm going with 25 that seem to be the rewiring span of electrical cables (rr = 4%)
    # I'm using the stock function to compute the new km and the 'waste' km

    ay_infra_road = dm_pop.array * 10 / 1000  # road infrastructure in km
    ay_infra_rail = dm_pop.array * 0.6 / 1000  # rail infrastructure in km
    ay_infra_trolleybus = dm_pop.array * 0.0017 / 1000  # rail infrastructure in km

    ay_tot = np.concatenate(
        (ay_infra_rail, ay_infra_road, ay_infra_trolleybus), axis=-1
    )

    dm_infra = DataMatrix.based_on(
        ay_tot[:, :, np.newaxis, :],
        format=dm_pop,
        change={
            "Variables": ["tra_tot-infrastructure"],
            "Categories1": ["infra-rail", "infra-road", "infra-trolley-cables"],
        },
        units={"tra_tot-infrastructure": "km"},
    )
    # Add dummy renewal rates
    dm_infra.add(0, dummy=True, dim="Variables", col_label="tra_renewal-rate", unit="%")
    idx = dm_infra.idx
    dm_infra.array[:, :, idx["tra_renewal-rate"], idx["infra-road"]] = 0.05
    dm_infra.array[:, :, idx["tra_renewal-rate"], idx["infra-rail"]] = 0.025
    dm_infra.array[:, :, idx["tra_renewal-rate"], idx["infra-trolley-cables"]] = 0.04

    compute_stock(
        dm_infra,
        "tra_renewal-rate",
        "tra_tot-infrastructure",
        waste_col="tra_infrastructure_waste",
        new_col="tra_new_infrastructure",
    )

    return dm_infra.filter({"Variables": ["tra_new_infrastructure"]})


def tra_emissions_interface(
    dm_pass_emissions, dm_freight_emissions, write_pickle=False
):

    dm_pass_emissions.rename_col(
        "tra_passenger_emissions", "tra_emissions_passenger", dim="Variables"
    )
    dm_pass_emissions = dm_pass_emissions.flatten().flatten()
    dm_freight_emissions.rename_col(
        "tra_freight_emissions", "tra_emissions_freight", dim="Variables"
    )
    dm_freight_emissions = dm_freight_emissions.flatten().flatten()

    dm_pass_emissions.append(dm_freight_emissions, dim="Variables")

    # if write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../_database/data/interface/transport_to_emissions.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(dm_pass_emissions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm_pass_emissions


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
            "../_database/data/interface/transport_to_agriculture.pickle",
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
        np.nan,
        dim="Variables",
        dummy=True,
        col_label="tra_power-demand_other",
        unit="GWh",
    )
    DM_power["electricity"].sort("Variables")
    DM_freight_power["electricity"].sort("Variables")
    DM_power["electricity"].array = (
        DM_power["electricity"].array + DM_freight_power["electricity"].array
    )

    # if write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../_database/data/interface/transport_to_power.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(DM_power, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return DM_power


def transport(lever_setting, years_setting, DM_input, interface=Interface()):

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    DM_passenger, DM_freight, DM_other, cdm_const = read_data(DM_input, lever_setting)

    cntr_list = DM_passenger["passenger_modal_split"].col_labels["Country"]

    # If the input from lifestyles are available in the interface, read them, else read from xls
    if interface.has_link(from_sector="lifestyles", to_sector="transport"):
        DM_lfs = interface.get_link(from_sector="lifestyles", to_sector="transport")
        dm_lfs = DM_lfs["pop"]
    else:
        if len(interface.list_link()) != 0:
            print(
                "You are missing " + "lifestyles" + " to " + "transport" + " interface"
            )
        lfs_interface_data_file = os.path.join(
            current_file_directory,
            "../_database/data/interface/lifestyles_to_transport.pickle",
        )
        with open(lfs_interface_data_file, "rb") as handle:
            DM_lfs = pickle.load(handle)
        dm_lfs = DM_lfs["pop"]
        dm_lfs.filter({"Country": cntr_list}, inplace=True)

    # PASSENGER
    cdm_const_passenger = cdm_const.copy()
    DM_passenger_out = passenger_fleet_energy(
        DM_passenger, dm_lfs, DM_other, cdm_const_passenger, years_setting
    )
    # FREIGHT
    cdm_const_freight = cdm_const.copy()
    DM_freight_out = freight_fleet_energy(
        DM_freight, DM_other, cdm_const_freight, years_setting
    )

    # Power-module
    DM_power = tra_power_interface(DM_passenger_out["power"], DM_freight_out["power"])
    interface.add_link(from_sector="transport", to_sector="power", dm=DM_power)
    # df = dm_power.write_df()
    # df.to_excel('transport-to-power.xlsx', index=False)

    # Storage-module
    dm_oil_refinery = tra_oilrefinery_interface(
        DM_passenger_out["oil-refinery"], DM_freight_out["oil-refinery"]
    )
    interface.add_link(
        from_sector="transport", to_sector="oil-refinery", dm=dm_oil_refinery
    )

    # Agriculture-module
    dm_agriculture = tra_agriculture_interface(
        DM_freight_out["agriculture"], DM_passenger_out["agriculture"]
    )
    interface.add_link(
        from_sector="transport", to_sector="agriculture", dm=dm_agriculture
    )

    # Minerals and Industry
    dm_freight_veh = DM_freight_out["tech"].filter(
        {
            "Variables": [
                "tra_freight_new-vehicles",
                "tra_freight_vehicle-waste",
                "tra_freight_vehicle-fleet",
            ]
        }
    )
    dm_passenger_veh = DM_passenger_out["tech"].filter(
        {
            "Variables": [
                "tra_passenger_new-vehicles",
                "tra_passenger_vehicle-waste",
                "tra_passenger_vehicle-fleet",
            ]
        }
    )
    dm_infrastructure = dummy_tra_infrastructure_workflow(dm_lfs)
    DM_industry = tra_industry_interface(
        dm_freight_veh.copy(), dm_passenger_veh.copy(), dm_infrastructure
    )
    # DM_minerals = tra_minerals_interface(dm_freight_veh, dm_passenger_veh, DM_industry, dm_infrastructure, write_xls=False)
    # !FIXME: add km infrastructure data, using compute_stock with tot_km and renovation rate as input.
    #  data for ch ok, data for eu, backcalculation? dummy based on swiss pop?
    interface.add_link(from_sector="transport", to_sector="industry", dm=DM_industry)
    # interface.add_link(from_sector='transport', to_sector='minerals', dm=DM_minerals)

    # Emissions
    dm_emissions = tra_emissions_interface(
        DM_passenger_out["emissions"], DM_freight_out["emissions"]
    )
    interface.add_link(
        from_sector="transport", to_sector="emissions", dm=dm_emissions.copy()
    )

    # Local transport emissions
    N2O_to_CO2 = 265
    CH4_to_CO2 = 28

    dm_emissions = DM_passenger_out["emissions"]
    idx = dm_emissions.idx
    dm_emissions.array[:, :, :, :, idx["CH4"]] = (
        dm_emissions.array[:, :, :, :, idx["CH4"]] * CH4_to_CO2
    )
    dm_emissions.array[:, :, :, :, idx["N2O"]] = (
        dm_emissions.array[:, :, :, :, idx["N2O"]] * N2O_to_CO2
    )
    dm_emissions.rename_col(
        "tra_emissions_passenger", "tra_emissions-CO2e_passenger", dim="Variables"
    )
    dm_emissions.group_all("Categories2")

    results_run = prepare_TPE_output(DM_passenger_out, DM_freight_out)
    return results_run


def local_transport_run():
    # Function to run only transport module without converter and tpe
    years_setting = [1990, 2023, 2025, 2050, 5]
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(current_file_directory, "../config/lever_position.json"))
    lever_setting = json.load(f)[0]

    # get geoscale
    country_list = ['EU27', 'Switzerland', 'Vaud']
    DM_input = filter_country_and_load_data_from_pickles(country_list= country_list, modules_list = 'transport')

    results_run = transport(lever_setting, years_setting, DM_input['transport'])

    return results_run


# print('In transport, the share of waste by fuel/tech type does not seem right. Fix it.')
# print('Apply technology shares before computing the stock')
# print('For the efficiency, use the new methodology developed for Building (see overleaf on U-value)')
#results_run = local_transport_run()
