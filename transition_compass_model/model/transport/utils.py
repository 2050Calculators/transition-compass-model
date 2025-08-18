import numpy as np
from model.common.data_matrix_class import DataMatrix

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
    dm_energy_ICE_PHEV = dm_energy.filter_w_regex(
        {"Categories2": "PHEV.*|ICE.*|kerosene"}
    )
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
    # if 'aviation' in dm_energy.col_labels['Categories1'] and 'aviation' not in dm_efuel.col_labels['Categories1']:
    #    dm_efuel.add(0, dim='Categories1', col_label='aviation', dummy=True)
    #    dm_biofuel.add(0, dim='Categories1', col_label='aviation', dummy=True)
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
            array = tmp_cat1
        else:
            array = np.concatenate([array, tmp_cat1], axis=-1)
    dm_total_energy = DataMatrix(
        col_labels={
            "Country": dm_new_cat.col_labels["Country"],
            "Years": dm_new_cat.col_labels["Years"],
            "Variables": [grouped_var],
            "Categories1": groups,
        },
        units={grouped_var: list(dm_new_cat.units.values())[0]},
    )

    dm_total_energy.array = array

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
