import os

import numpy as np

from transition_compass_model._database.pre_processing.buildings.Switzerland.get_data_functions.services_CH import (
    extract_services_renovation_rate_EP2050,
)
from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    filter_DM,
    linear_fit_ratio,
    linear_fitting,
    my_pickle_dump,
    sort_pickle,
)


def calculate_heating_eff_fts(
    dm_heating_eff, years_fts, maximum_eff, fuel_cat="Categories2"
):
    # Linear fitting of all efficiencies except based on the 2015-2023 period
    # For all efficiencies except heat-pump, the efficiency is capped at 98%
    dm_heat_pump = dm_heating_eff.filter({fuel_cat: ["heat-pump"]})
    dm_heating_eff.drop(dim=fuel_cat, col_label="heat-pump")
    linear_fitting(dm_heating_eff, years_fts, based_on=list(range(2015, 2023)))
    dm_heating_eff.array = np.minimum(dm_heating_eff.array, maximum_eff)
    linear_fitting(dm_heat_pump, years_fts, based_on=list(range(2015, 2023)))
    dm_heating_eff.append(dm_heat_pump, dim=fuel_cat)
    dm_heating_eff_fts = dm_heating_eff.filter({"Years": years_fts})

    return dm_heating_eff_fts


def adjust_COP_based_on_envelope_cat(dm):
    # This is taken from https://www.flumroc.ch/fileadmin/Dateiliste/flumroc/Bilder/400_steinwolle/Stromsparen-HSLU/d_250716_Kurzbericht_Studie_Flumroc_final.pdf
    # Which is originally taken from  Döring & Richter, 2024
    # And the information of the average energy demand per building category (SFH) from the archetype paper
    # Pongelli, A.; Priore, Y.D.; Bacher, J.-P.; Jusselme, T. Definition of Building Archetypes Based on the Swiss Energy Performance Certificates Database. Buildings 2023, 13, 40. https://doi.org/10.3390/buildings13010040
    avg_2023_COP = {"F": 2.3, "E": 2.5, "D": 3, "C": 3.3, "B": 3.6}
    for cat in dm.col_labels["Categories1"]:
        # corr_fact = avg_2023_COP[cat]/dm[:, 2023, np.newaxis, :, cat, 'heat-pump']
        # dm[:, :, :, cat, 'heat-pump'] =  corr_fact * dm[:, :, :, cat, 'heat-pump']
        dm[:, :, :, cat, "heat-pump"] = np.minimum(
            avg_2023_COP[cat], dm[:, :, :, cat, "heat-pump"]
        )

    return dm


def run(DM_buildings, country_list, years_fts):
    filter_DM(DM_buildings, {"Country": country_list})

    this_dir = os.path.dirname(os.path.abspath(__file__))
    # !FIXME: use the actual values and not the calibration factor
    file = os.path.join(this_dir, "../../../../data/datamatrix/buildings.pickle")

    #########################################
    #####  FLOOR INTENSITY - SPACE/CAP  #####
    #########################################
    dm_space_cap = DM_buildings["ots"]["floor-intensity"].copy()
    # Use recent years only for nonres-cap: trend peaked ~2015 and is flat/declining since
    dm_nonres_cap = dm_space_cap.filter(
        {"Variables": ["bld_floor-intensity_nonres-cap"]}, inplace=False
    )
    linear_fitting(dm_nonres_cap, years_fts, based_on=list(range(2012, 2024)))
    dm_res_cap = dm_space_cap.filter(
        {
            "Variables": [
                v
                for v in dm_space_cap.col_labels["Variables"]
                if v != "bld_floor-intensity_nonres-cap"
            ]
        },
        inplace=False,
    )
    linear_fitting(dm_res_cap, years_fts)
    dm_nonres_cap.append(dm_res_cap, dim="Variables")
    dm_space_cap = dm_nonres_cap
    DM_buildings["fts"]["floor-intensity"] = dict()
    for lev in range(4):
        lev = lev + 1
        DM_buildings["fts"]["floor-intensity"][lev] = dm_space_cap.filter(
            {"Years": years_fts}
        )

    #########################################
    #####   HEATING-COOLING BEHAVIOUR   #####
    #########################################
    dm_Tint_heat = DM_buildings["ots"]["heatcool-behaviour"].copy()
    linear_fitting(dm_Tint_heat, years_fts)
    DM_buildings["fts"]["heatcool-behaviour"] = dict()
    for lev in range(4):
        lev = lev + 1
        DM_buildings["fts"]["heatcool-behaviour"][lev] = dm_Tint_heat.filter(
            {"Years": years_fts}
        )

    #########################################
    #####  SERVICES FLOOR AREA - FTS  #######
    #########################################
    if "services-floor-area" in DM_buildings["ots"]:
        dm_srv_floor = DM_buildings["ots"]["services-floor-area"].copy()
        linear_fitting(dm_srv_floor, years_fts, based_on=list(range(2012, 2024)))
        DM_buildings["fts"]["services-floor-area"] = dict()
        for lev in range(4):
            lev = lev + 1
            DM_buildings["fts"]["services-floor-area"][lev] = dm_srv_floor.filter(
                {"Years": years_fts}
            )

    #########################################
    #####        BUILDING MIX          ######
    #########################################
    dm_building_mix_new = DM_buildings["ots"]["building-renovation-rate"][
        "bld_building-mix"
    ].copy()
    dm_building_mix_new.add(np.nan, dim="Years", col_label=years_fts, dummy=True)
    dm_building_mix_new.fill_nans("Years")
    DM_buildings["fts"]["building-renovation-rate"] = dict()
    DM_buildings["fts"]["building-renovation-rate"]["bld_building-mix"] = dict()
    for lev in range(4):
        lev = lev + 1
        DM_buildings["fts"]["building-renovation-rate"]["bld_building-mix"][lev] = (
            dm_building_mix_new.filter({"Years": years_fts})
        )

    #########################################
    #####       RENOVATION-RATE        ######
    #########################################
    dm_rr = DM_buildings["ots"]["building-renovation-rate"][
        "bld_renovation-rate"
    ].copy()
    DM_buildings["fts"]["building-renovation-rate"]["bld_renovation-rate"] = dict()
    dm_rr.add(np.nan, dim="Years", dummy=True, col_label=years_fts)
    dm_rr.fill_nans(dim_to_interp="Years")

    # For non-residential types, replace the flat OTS extrapolation with EP2050 FTS values.
    # Level 1 (BAU) = WWB; level 4 (ambitious) = ZERO; levels 2-3 = linear interpolation.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    ep2050_services_file = os.path.join(
        this_dir,
        "../data/EP2050_sectors/EP2050+_Szenarienergebnisse_Details_Nachfragesektoren/"
        "EP2050+_Detailergebnisse 2020-2060_Dienstleistung_alle Szenarien_2022-10-20.xlsx",
    )
    nonres_types = ["education", "health", "hotels", "offices", "other", "trade"]
    country_list = dm_rr.col_labels["Country"]
    ep2050_rr = extract_services_renovation_rate_EP2050(
        ep2050_services_file, [], nonres_types, country_list
    )["fts"]
    dm_wwb_fts = ep2050_rr["wwb"]  # shape: (country, fts_years, var, nonres_types)
    dm_zero_fts = ep2050_rr["zero"]

    for lev in range(1, 5):
        dm_lev = dm_rr.filter({"Years": years_fts})
        # Interpolation weight: 0 = WWB, 1 = ZERO
        alpha = (lev - 1) / 3.0
        for yr in dm_wwb_fts.col_labels["Years"]:
            if yr in dm_lev.idx:
                wwb_val = dm_wwb_fts.array[
                    :, dm_wwb_fts.idx[yr], dm_wwb_fts.idx["bld_renovation-rate"], :
                ]
                zero_val = dm_zero_fts.array[
                    :, dm_zero_fts.idx[yr], dm_zero_fts.idx["bld_renovation-rate"], :
                ]
                interp_val = (1 - alpha) * wwb_val + alpha * zero_val
                for i_t, t in enumerate(nonres_types):
                    dm_lev.array[
                        :,
                        dm_lev.idx[yr],
                        dm_lev.idx["bld_renovation-rate"],
                        dm_lev.idx[t],
                    ] = interp_val[:, i_t]
        DM_buildings["fts"]["building-renovation-rate"]["bld_renovation-rate"][lev] = (
            dm_lev
        )
    # Lever 1 = BAU: keep OTS renovation rate (no zeroing)

    ###########################################
    #####    RENOVATION-REDISTRIBUTION    #####
    ###########################################
    dm_renov_distr = DM_buildings["ots"]["building-renovation-rate"][
        "bld_renovation-redistribution"
    ].copy()
    DM_buildings["fts"]["building-renovation-rate"]["bld_renovation-redistribution"] = (
        dict()
    )
    dm_renov_distr.add(np.nan, dim="Years", dummy=True, col_label=years_fts)
    dm_renov_distr.fill_nans(dim_to_interp="Years")
    for lev in range(4):
        lev = lev + 1
        DM_buildings["fts"]["building-renovation-rate"][
            "bld_renovation-redistribution"
        ][lev] = dm_renov_distr.filter({"Years": years_fts})

    #########################################
    #####         DEMOLITION          #######
    #########################################

    dm_demolition_rate = DM_buildings["ots"]["building-renovation-rate"][
        "bld_demolition-rate"
    ].copy()
    # Compute average demolition rate in the last 10 years and forecast to future
    idx = dm_demolition_rate.idx
    idx_yrs = [idx[yr] for yr in create_years_list(2012, 2023, 1)]
    val_mean = np.mean(dm_demolition_rate.array[:, idx_yrs, ...], axis=1)
    dm_demolition_rate.add(np.nan, dim="Years", dummy=True, col_label=years_fts)
    for yr in years_fts:
        dm_demolition_rate.array[:, idx[yr], ...] = val_mean
    # FTS
    DM_buildings["fts"]["building-renovation-rate"]["bld_demolition-rate"] = dict()
    for lev in range(4):
        lev = lev + 1
        DM_buildings["fts"]["building-renovation-rate"]["bld_demolition-rate"][lev] = (
            dm_demolition_rate.filter({"Years": years_fts})
        )

    ###########################################
    #####    HEATING TECHNOLOGY MIX     #######
    ###########################################

    dm_heating_cat = DM_buildings["ots"]["heating-technology-fuel"][
        "bld_heating-technology"
    ].copy()
    dm_heating_cat.add(np.nan, dim="Years", dummy=True, col_label=years_fts)
    # Obligation à enlever les chauffages electriques d'ici 2033
    # https://www.vd.ch/environnement/energie/legislation/chauffages-et-chauffe-eaux-electriques
    idx = dm_heating_cat.idx

    # We want to linearly interpolate for elecetricity and then linearly extrpolate the other base on years 2015 -2023
    def set_elec_to_zero(dm):
        """
        The function sets resisitive electric heating to 0 from 2035 onwards.
        The function first sets the values to NaN from 2025 onwards, and then sets them to 0 from 2035 onwards. The function then fills the NaN values by linear interpolation.
        From 2033 onwards, the DACCE makes electricty heating unauthorized.  (article 9 et 10 )
        Some exceptions exists but are not considered here. For more information, see
        https://www.vd.ch/environnement/energie/legislation/chauffages-et-chauffe-eaux-electriques

        Args:
            dm (_type_): The data matrix containing the heating technology mix for buildings in Vaud, Switzerland. The data matrix is expected to have a dimension for heating technologies, including electricity.

        Returns:
            _type_: The modified data matrix with the values for electricity heating set to 0 from 2035 onwards, and the NaN values filled by linear interpolation.
        """
        idx = dm.idx
        dm_heating_cat_elec = dm.copy()
        # The data will linearly interpolated
        dm_heating_cat_elec["Vaud", idx[2025] :, ..., "electricity"] = np.nan
        #
        # Here it is simpllified to 0 and exception are considerted neglected.
        dm_heating_cat_elec["Vaud", idx[2035] :, ..., "electricity"] = 0
        dm_heating_cat_elec.fill_nans("Years")
        dm["Vaud", ..., "electricity"] = dm_heating_cat_elec["Vaud", ..., "electricity"]
        return dm

    dm_heating_cat = set_elec_to_zero(dm_heating_cat)
    dm_heating_cat = linear_fit_ratio(
        dm_heating_cat.copy(),
        years_fts,
        years_range=[2015, 2023],
        category_to_normalise="Categories3",
    )

    DM_buildings["fts"]["heating-technology-fuel"] = dict()
    DM_buildings["fts"]["heating-technology-fuel"]["bld_heating-technology"] = dict()
    for lev in range(4):
        lev = lev + 1
        DM_buildings["fts"]["heating-technology-fuel"]["bld_heating-technology"][
            lev
        ] = dm_heating_cat.filter({"Years": years_fts})
    ###########################################
    #####    HOTWATER TECHNOLOGY MIX     #######
    ###########################################
    dm_hotwater_cat = DM_buildings["fxa"]["hot-water"]["hw-tech-mix"].copy()

    dm_hotwater_cat = set_elec_to_zero(dm_hotwater_cat)

    # For hot water we do a linear extrapolation of the technology mix.
    # As it is a ratio we force values to be above 0, and then normalise to sum to 1 again.

    dm_hotwater_cat = linear_fit_ratio(
        dm_hotwater_cat,
        years_fts,
        years_range=[2015, 2023],
        category_to_normalise="Categories1",
    )

    # dm_hotwater_cat.fill_nans("Years")
    # dm_hotwater_cat.normalise("Categories1")
    DM_buildings["fts"]["heating-technology-fuel"]["bld_hot-water-technology"] = dict()
    for lev in range(4):
        lev = lev + 1
        DM_buildings["fts"]["heating-technology-fuel"]["bld_hot-water-technology"][
            lev
        ] = dm_hotwater_cat.filter({"Years": years_fts})

    ############################################
    ######       HEATING EFFICIENCY       ######
    ############################################
    dm_heating_eff = DM_buildings["ots"]["heating-efficiency"].copy()
    dm_heating_eff_fts = calculate_heating_eff_fts(
        dm_heating_eff.copy(), years_fts, maximum_eff=0.98
    )
    dm_heating_eff_fts = adjust_COP_based_on_envelope_cat(dm_heating_eff_fts)
    dm_heating_eff_fts[:, :, "bld_heating-efficiency", :, "electricity"] = 1
    DM_buildings["fts"]["heating-efficiency"] = dict()
    for lev in range(4):
        lev = lev + 1
        DM_buildings["fts"]["heating-efficiency"][lev] = dm_heating_eff_fts.copy()

    """dm_hw_eff = DM_buildings['ots']['heating-efficiency']['bld_hot-water-efficiency'].copy()
  dm_hw_eff_fts = calculate_heating_eff_fts(dm_hw_eff.copy(),years_fts, maximum_eff=0.98, fuel_cat='Categories1')
  dm_hw_eff_fts[:, :, 'bld_hot-water_efficiency', 'electricity'] = 1

  DM_buildings['fts']['heating-efficiency']['bld_hot-water-efficiency'] = dict()
  for lev in range(4):
    lev = lev + 1
    DM_buildings['fts']['heating-efficiency']['bld_hot-water-efficiency'][lev] = dm_hw_eff_fts.copy()"""

    my_pickle_dump(DM_buildings, file)
    sort_pickle(file)

    return DM_buildings
