import numpy as np

from transition_compass_model.model.common.auxiliary_functions import (
    cdm_to_dm,
    create_years_list,
)
from transition_compass_model.model.common.constant_data_matrix_class import (
    ConstantDataMatrix,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def run(country_list, years_ots, years_fts):
    # SECTION U-values - fixed assumption
    # Definition of Building Archetypes Based on the Swiss Energy Performance Certificates Database
    # by Alessandro Pongelli et al.
    # U-value is computed as the average of the house element u-value (roof, wall, windows, ..) weighted by their area
    # U-value in: W/m^2 K
    # Single-family-households
    # Non-residential values are proxied from EU Hotmaps data (same envelope U-values by class)
    nonres_u = {"F": 2.01, "E": 1.74, "D": 1.44, "C": 1.20, "B": 0.69}
    envelope_cat_u_value = {
        "single-family-households": {
            "F": 0.82,
            "E": 0.69,
            "D": 0.53,
            "C": 0.41,
            "B": 0.25,
        },
        "multi-family-households": {
            "F": 0.93,
            "E": 0.70,
            "D": 0.63,
            "C": 0.48,
            "B": 0.29,
        },
        "education": nonres_u,
        "health": nonres_u,
        "hotels": nonres_u,
        "offices": nonres_u,
        "other": nonres_u,
        "trade": nonres_u,
    }
    cdm_u_value = ConstantDataMatrix(
        col_labels={
            "Variables": ["bld_u-value"],
            "Categories1": sorted(envelope_cat_u_value.keys()),
            "Categories2": ["B", "C", "D", "E", "F"],
        },
        units={"bld_u-value": "W/m2K"},
    )
    arr = np.zeros(
        (
            len(cdm_u_value.col_labels["Variables"]),
            len(cdm_u_value.col_labels["Categories1"]),
            len(cdm_u_value.col_labels["Categories2"]),
        )
    )
    cdm_u_value.array = arr
    idx = cdm_u_value.idx
    for bld, dict_val in envelope_cat_u_value.items():
        for cat, val in dict_val.items():
            cdm_u_value.array[idx["bld_u-value"], idx[bld], idx[cat]] = val
    dm_u_value = cdm_to_dm(cdm_u_value, country_list, ["All"])

    # SECTION Surface to Floorarea factor - fixed assumption
    # Residential values (sfh, mfh) from Pongelli et al. (2023), Swiss EPC database archetypes.
    # Non-residential values estimated from building physics references:
    #   education=1.2, offices=1.3, health=1.3: Delmastro et al. (2016), "A building stock analysis
    #     for the Italian residential and service sector", Energy Build. 128, 247–263.
    #   hotels=1.4: Fleiter et al. (2017), "A methodology for bottom-up modelling of energy transitions
    #     in the industry and service sectors", Energy Effic. 10, 829–847.
    #   trade=1.1: large-footprint single-storey retail, typical building physics assumption
    #     (low external-surface-to-floor ratio); consistent with Hotmaps project CH building atlas.
    #   other=1.3: generic proxy, same as offices/health.
    surface_to_floorarea = {
        "single-family-households": 2.0,
        "multi-family-households": 1.3,
        "education": 1.2,
        "health": 1.3,
        "hotels": 1.4,
        "offices": 1.3,
        "other": 1.3,
        "trade": 1.1,
    }
    cdm_s2f = ConstantDataMatrix(
        col_labels={
            "Variables": ["bld_surface-to-floorarea"],
            "Categories1": sorted(surface_to_floorarea.keys()),
        }
    )
    arr = np.zeros(
        (len(cdm_s2f.col_labels["Variables"]), len(cdm_s2f.col_labels["Categories1"]))
    )
    cdm_s2f.array = arr
    idx = cdm_s2f.idx
    for cat, val in surface_to_floorarea.items():
        cdm_s2f.array[idx["bld_surface-to-floorarea"], idx[cat]] = val
    cdm_s2f.units["bld_surface-to-floorarea"] = "%"
    dm_s2f = cdm_to_dm(cdm_s2f, country_list, ["All"])

    # SECTION: Heating-cooling behaviour (Temperature)
    #########################################
    #####   HEATING-COOLING BEHAVIOUR   #####
    #########################################
    nonres_types = ["education", "health", "hotels", "offices", "other", "trade"]
    col_label = {
        "Country": country_list,
        "Years": years_ots + years_fts,
        "Variables": ["bld_Tint-heating", "bld_Tint-cooling"],
        "Categories1": sorted(
            ["multi-family-households", "single-family-households"] + nonres_types
        ),
        "Categories2": ["B", "C", "D", "E", "F"],
    }
    dm_Tint_heat = DataMatrix(
        col_labels=col_label, units={"bld_Tint-heating": "C", "bld_Tint-cooling": "C"}
    )
    dm_Tint_heat.array[...] = 20
    idx = dm_Tint_heat.idx
    cat_Tint = {"F": 19, "E": 20, "D": 21, "C": 22, "B": 23}
    for cat, tint in cat_Tint.items():
        dm_Tint_heat.array[
            :, :, idx["bld_Tint-heating"], idx["multi-family-households"], idx[cat]
        ] = tint
        dm_Tint_heat.array[
            :, :, idx["bld_Tint-heating"], idx["single-family-households"], idx[cat]
        ] = tint - 1
    # Non-residential: flat 20°C across all building types and energy classes.
    # Source: EN 15251 / ISO 13790 standard design indoor temperature for non-residential
    # buildings (offices, schools, etc. = 20°C), adopted in Swiss SIA 380/1.
    # No class variation assumed (unlike residential where higher-class buildings tend
    # to be maintained at slightly higher temperatures).

    # SECION: Building age
    first_bld_sfh = {"F": 1900, "E": 1971, "D": 1981, "C": 2001, "B": 2011}
    first_bld_mfh = {"F": 1900, "E": 1981, "D": 1991, "C": 2001, "B": 2011}
    # Non-residential: EP2050 construction period boundaries (vor 1946→F, 1947-1975→E, etc.)
    first_bld_nonres = {"F": 1900, "E": 1947, "D": 1976, "C": 1991, "B": 2020}
    col_label = {
        "Country": country_list,
        "Years": years_ots + years_fts,
        "Variables": ["bld_age"],
        "Categories1": ["multi-family-households", "single-family-households"]
        + nonres_types,
        "Categories2": ["B", "C", "D", "E", "F"],
    }
    dm_age = DataMatrix(col_labels=col_label, units={"bld_age": "years"})
    years_all = np.array(dm_age.col_labels["Years"])
    nb_cntr = len(dm_age.col_labels["Country"])
    idx = dm_age.idx
    for cat, start_yr in first_bld_sfh.items():
        arr_age = years_all - start_yr
        arr_age = np.maximum(arr_age, 0)
        for idx_c in range(nb_cntr):
            dm_age.array[
                idx_c, :, idx["bld_age"], idx["single-family-households"], idx[cat]
            ] = arr_age
    for cat, start_yr in first_bld_mfh.items():
        arr_age = years_all - start_yr
        arr_age = np.maximum(arr_age, 0)
        for idx_c in range(nb_cntr):
            dm_age.array[
                idx_c, :, idx["bld_age"], idx["multi-family-households"], idx[cat]
            ] = arr_age
    for t in nonres_types:
        for cat, start_yr in first_bld_nonres.items():
            arr_age = years_all - start_yr
            arr_age = np.maximum(arr_age, 0)
            for idx_c in range(nb_cntr):
                dm_age.array[idx_c, :, idx["bld_age"], idx[t], idx[cat]] = arr_age

    ####################################
    #####     EMISSION FACTORS    ######
    ####################################
    # Obtained from OFEV file https://www.bafu.admin.ch/dam/fr/sd-web/HnIzzj6OfDUU/EF_CO2_Berichterstattung_Kantone.pdf.
    # Electricity and heating district are treated separately. There is no coal in vaud
    OFEV_emissions_fact = {
        "coal": 350,
        "heating-oil": 265,
        "gas": 201,
        "wood": 0,
        "solar": 0,
    }

    cdm_emission_fact = ConstantDataMatrix(
        col_labels={
            "Variables": ["bld_CO2-factors"],
            "Categories1": [
                "coal",
                "heating-oil",
                "gas",
                "wood",
                "solar",
            ],
        },
        units={"bld_CO2-factors": "kt/TWh"},
    )
    cdm_emission_fact.array = np.zeros(
        (
            len(cdm_emission_fact.col_labels["Variables"]),
            len(cdm_emission_fact.col_labels["Categories1"]),
        )
    )
    idx = cdm_emission_fact.idx
    for key, value in OFEV_emissions_fact.items():
        cdm_emission_fact.array[0, idx[key]] = value

    cdm_emission_fact.sort("Categories1")

    # SECTION: Electricity and district emission factors
    col_dict = {
        "Country": country_list,
        "Years": years_ots + years_fts,
        "Variables": ["bld_CO2-factor"],
        "Categories1": ["electricity", "heating_district"],
    }
    dm_co2_factor = DataMatrix(col_labels=col_dict, units={"bld_CO2-factor": "kt/TWh"})

    arr_co2_factor = np.zeros((2, 40, 1, 2))
    idx = dm_co2_factor.idx
    arr_co2_factor[:, idx[1990] : idx[2023] + 1, :, idx["electricity"]] = 168.64
    arr_co2_factor[:, idx[2025] : idx[2050], :, idx["electricity"]] = np.nan
    arr_co2_factor[:, idx[2050], :, idx["electricity"]] = 0

    arr_co2_factor[:, : idx[2019] + 1, :, idx["heating_district"]] = 73.55
    arr_co2_factor[:, idx[2019] : idx[2023], :, idx["heating_district"]] = np.nan
    arr_co2_factor[:, idx[2023], :, idx["heating_district"]] = 66
    arr_co2_factor[:, idx[2025] : idx[2050], :, idx["heating_district"]] = np.nan
    arr_co2_factor[:, idx[2050], :, idx["heating_district"]] = 20.7
    dm_co2_factor.array = arr_co2_factor
    dm_co2_factor.fill_nans(dim_to_interp="Years")

    DM_other = {
        "u-value": dm_u_value,
        "surface-to-floor": dm_s2f,
        "Tint-heat": dm_Tint_heat,
        "age": dm_age,
        "emission-factors": cdm_emission_fact,
        "emission-fact-elec_district": dm_co2_factor,
    }

    return DM_other


if __name__ == "__main__":
    print("Running U-value and Surface to floor area factor")
    country_list = ["Switzerland", "Vaud"]
    years_ots = create_years_list(1990, 2023, 1)
    years_fts = create_years_list(2025, 2050, 5)
    DM_other = run(country_list, years_ots, years_fts)
    print("Done")
