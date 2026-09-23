import os
import pickle

import numpy as np
from get_data_functions import utils as utils
from params import years_ots

from transition_compass_model._database.pre_processing.api_routines_CH import (
    get_data_api_CH,
)
from transition_compass_model._database.pre_processing.api_routines_swiss_stats import (
    get_data_api_swiss_stats,
)
from transition_compass_model.model.common.auxiliary_functions import moving_average
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def clean_energy_dimension(dm_veh_eff, var_name):
    # Clean grams CO2 category and perform weighted average
    # cols are e.g '0 - 50 g' -> '0-50' -> 25
    dm_veh_eff.rename_col_regex("kWh", "", dim="Categories2")
    dm_veh_eff.rename_col_regex(" ", "", dim="Categories2")
    # TODO : search for real maximum efficiency value.
    dm_veh_eff.rename_col("20.01ormore", "20.01‒27.5", dim="Categories2")
    dm_veh_eff.rename_col("Upto12.50", "0‒12.50", dim="Categories2")
    cat2_list_old = dm_veh_eff.col_labels["Categories2"]
    co2_km = []
    for i in range(len(cat2_list_old)):
        old_cat = cat2_list_old[i]
        new_cat = (float(old_cat.split("‒")[0]) + float(old_cat.split("‒")[1])) / 2
        co2_km.append(new_cat)
    co2_arr = np.array(co2_km)
    dm_veh_eff.normalise(dim="Categories2", inplace=True)
    dm_veh_eff.array = (
        dm_veh_eff.array * co2_arr[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    )
    dm_veh_eff.group_all(dim="Categories2")

    dm_veh_eff.change_unit(var_name, 1, old_unit="%", new_unit="kWh/100km")
    dm_veh_eff.change_unit(var_name, 1 / 100, old_unit="kWh/100km", new_unit="kWh/km")
    return dm_veh_eff


def get_vehicle_electric_efficiency(
    file: str, agency: str, dataflow: str, var_name="tra_passenger_veh-efficiency_fleet"
):
    """
    Get energy consumption data of the database Stock of passenger cars by technical characteristics (2/2):
    By year, canton, type of holder (natural person / legal entity), age of holder, unladen weight, fuel, CO2 emissions per km (NEDC) and electricity consumption per 100km (WLTP)
    Note: No data prior to 2016. At present, CO2 emission values based on the new,
    more realistic WLTP (Worldwide Harmonised Light-Duty Vehicles Test Procedure)
    measurement method exist only for a small share of the vehicle stock.
    For this reason, the emissions in this time series are still reported in
    accordance with the previous NEDC method (New European Driving Cycle).
    Vehicles for which only WLTP values are available, and for which it is not
    possible to perform a back calculation in NEDC, are listed under 'Missing/Undefined/Unknown'.

    Args:
        file (str): filepath where the pickle is saved
        agency (str): Agency to call the api
        dataflow (str): dataflow to call the api
        var_name (str, optional): _description_. Defaults to "tra_passenger_veh-efficiency_fleet".

    Raises:
        ValueError: raise error if call to api failed

    Returns:
        DataMatrix: Efficiency datamatrix
    """
    #
    try:
        with open(file, "rb") as handle:
            dm_veh_eff = pickle.load(handle)
            print(
                f"The vehicle efficienty is read from file {file}. Delete it if you want to update data from api."
            )
    except OSError:
        structure, title = get_data_api_swiss_stats(agency, dataflow, mode="example")
        i = 0
        ibis = 0
        fuel_list = ["Battery electric vehicle (BEV)"]

        consumption_list = [
            cons
            for cons in structure["UV_RV_ECONSUMPTION_WLTP_PC"]
            if cons not in ["Total"]
        ]

        CO2_emission_list = ["Total"]

        filtering = {
            "TIME_PERIOD": structure["TIME_PERIOD"],
            "UV_RV_FUEL": fuel_list,
            "UV_RV_OWNER_TYPE": ["Total"],
            "UV_RV_OWNER_AGE": ["Total"],
            "UV_RV_EMPTY_WEIGHT": ["Total"],
            "UV_RV_ECONSUMPTION_WLTP_PC": consumption_list,
            "UV_HGDE_KT": [
                "Total",
                "Vaud",
                "Fribourg",
                "Schwyz",
            ],
            "UV_RV_CO2_NEDC": CO2_emission_list,
        }

        mapping_dim = {
            "Country": "UV_HGDE_KT",
            "Years": "TIME_PERIOD",
            "Variables": "UV_RV_CO2_NEDC",
            "Categories1": "UV_RV_FUEL",
            "Categories2": "UV_RV_ECONSUMPTION_WLTP_PC",
        }

        # Extract new fleet
        dm_veh_eff = get_data_api_swiss_stats(
            agency,
            dataflow,
            mode="extract",
            filter=filtering,
            mapping_dims=mapping_dim,
            units=["number"] * len(consumption_list),
        )
        if dm_veh_eff is None:
            raise ValueError(f"API returned None for {agency},{dataflow}")

        dm_veh_eff.array = np.nan_to_num(dm_veh_eff.array)
        dm_veh_eff.rename_col("Total_Variables", var_name, dim="Variables")

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, "wb") as handle:
            pickle.dump(dm_veh_eff, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Group categories1 according to model
    dict_tech = {
        "BEV": ["Battery electric vehicle (BEV)"],
    }
    # Rename columns with dict tech
    dm_veh_eff.groupby(dict_tech, dim="Categories1", inplace=True)

    # Distribute Inconnu on other categories based on their share
    # Remove fuel type "Autre" (there are only very few car in this category)
    # Before 2022 data is mainly unknown so it is dropped as it is doesn't give us efficiency information
    dm_veh_eff = utils.drop_if_smaller_than_0_01(
        dm_veh_eff, cat_to_drop="Categories2", col_to_drop="Missing/Undefined/Unknown"
    )

    dm_veh_eff = clean_energy_dimension(dm_veh_eff, var_name)

    for i in range(2):
        window_size = 3  # Change window size to control the smoothing effect
        data_smooth = moving_average(
            dm_veh_eff.array, window_size, axis=dm_veh_eff.dim_labels.index("Years")
        )
        dm_veh_eff.array[:, 1:-1, ...] = data_smooth

    # Add LDV
    dm_veh_eff_LDV = DataMatrix.based_on(
        dm_veh_eff.array[..., np.newaxis],
        dm_veh_eff,
        change={"Categories2": ["LDV"]},
        units=dm_veh_eff.units,
    )
    dm_veh_eff_LDV.switch_categories_order()
    dm_veh_eff_LDV.rename_col("Total_Country", "Switzerland", dim="Country")
    dm_veh_eff_LDV.filter({"Years": years_ots}, inplace=True)

    return dm_veh_eff_LDV


def get_vehicle_efficiency_co2(
    file: str, agency: str, dataflow: str, var_name="tra_passenger_veh-efficiency_fleet"
):
    """Get CO2 emission data of the database Stock of passenger cars by technical characteristics (2/2):
    By year, canton, type of holder (natural person / legal entity), age of holder, unladen weight, fuel, CO2 emissions per km (NEDC) and electricity consumption per 100km (WLTP)
    Note: No data prior to 2016. At present, CO2 emission values based on the new,
    more realistic WLTP (Worldwide Harmonised Light-Duty Vehicles Test Procedure)
    measurement method exist only for a small share of the vehicle stock.
    For this reason, the emissions in this time series are still reported in
    accordance with the previous NEDC method (New European Driving Cycle).
    Vehicles for which only WLTP values are available, and for which it is not
    possible to perform a back calculation in NEDC, are listed under 'Missing/Undefined/Unknown'.

    Args:
        file (str): filepath where the pickle is saved
        agency (str): Agency to call the api
        dataflow (str): dataflow to call the api
        var_name (str, optional): _description_. Defaults to "tra_passenger_veh-efficiency_fleet".

    Raises:
        ValueError: raise error if call to api failed

    Returns:
        DataMatrix: Efficiency datamatrix
    """
    #
    try:
        with open(file, "rb") as handle:
            dm_veh_eff = pickle.load(handle)
            print(
                f"The vehicle efficienty is read from file {file}. Delete it if you want to update data from api."
            )
    except OSError:
        structure, title = get_data_api_swiss_stats(agency, dataflow, mode="example")
        i = 0
        ibis = 0
        fuel_list = [
            fuel
            for fuel in structure["UV_RV_FUEL"]
            if fuel
            not in [
                "Total",
                "No motor",
                "Battery electric vehicle (BEV)",
                "Fuel cell electric vehicle (FCEV)",
            ]
        ]
        consumption_list = ["Total"]

        CO2_emission_list = [
            emis for emis in structure["UV_RV_CO2_NEDC"] if emis not in ["Total"]
        ]

        filtering = {
            "TIME_PERIOD": structure["TIME_PERIOD"],
            "UV_RV_FUEL": fuel_list,
            "UV_RV_OWNER_TYPE": ["Total"],
            "UV_RV_OWNER_AGE": ["Total"],
            "UV_RV_EMPTY_WEIGHT": ["Total"],
            "UV_RV_ECONSUMPTION_WLTP_PC": consumption_list,
            "UV_HGDE_KT": [
                "Total",
                "Vaud",
                "Fribourg",
                "Schwyz",
            ],
            "UV_RV_CO2_NEDC": CO2_emission_list,
        }

        mapping_dim = {
            "Country": "UV_HGDE_KT",
            "Years": "TIME_PERIOD",
            "Variables": "UV_RV_ECONSUMPTION_WLTP_PC",
            "Categories1": "UV_RV_FUEL",
            "Categories2": "UV_RV_CO2_NEDC",
        }

        # Extract new fleet
        dm_veh_eff = get_data_api_swiss_stats(
            agency,
            dataflow,
            mode="extract",
            filter=filtering,
            mapping_dims=mapping_dim,
            units=["number"] * len(consumption_list),
        )
        if dm_veh_eff is None:
            raise ValueError(f"API returned None for {agency},{dataflow}")

        dm_veh_eff.array = np.nan_to_num(dm_veh_eff.array)
        dm_veh_eff.rename_col("Total_Variables", var_name, dim="Variables")

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, "wb") as handle:
            pickle.dump(dm_veh_eff, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Group categories1 according to model
    dict_tech = {
        "ICE-diesel": ["Diesel: conventional", "Diesel: hybrid electric vehicle (HEV)"],
        "ICE-gasoline": [
            "Petrol: conventional",
            "Petrol: hybrid electric vehicle (HEV)",
        ],
        "PHEV-diesel": ["Plug-in hybrid electric vehicle (PHEV): diesel"],
        "PHEV-gasoline": ["Plug-in hybrid electric vehicle (PHEV): petrol"],
        "ICE-gas": ["Gas (mono- and bi-fuel)"],
    }
    # Rename columns with dict tech
    dm_veh_eff.groupby(dict_tech, dim="Categories1", inplace=True)

    # For BEV electric consumption check get_vehicle_electric_efficiency
    dm_veh_eff.drop(dim="Categories1", col_label="BEV")
    # # Do this to have realistic curves
    mask = dm_veh_eff.array == 0
    dm_veh_eff.array[mask] = np.nan

    # Flat extrapolation data is bad before 2016 so backstage it
    dm_veh_eff.filter({"Years": list(range(2017, years_ots[-1]))}, inplace=True)
    years_to_add = [
        year for year in years_ots if year not in dm_veh_eff.col_labels["Years"]
    ]
    dm_veh_eff.add(np.nan, dummy=True, col_label=years_to_add, dim="Years")
    dm_veh_eff.sort(dim="Years")
    dm_veh_eff.fill_nans(dim_to_interp="Years")

    # Distribute Inconnu on other categories based on their share
    # Remove fuel type "Autre" (there are only very few car in this category)
    dm_veh_eff = utils.drop_if_smaller_than_0_01(
        dm_veh_eff, cat_to_drop="Categories2", col_to_drop="No data/unknown"
    )

    dm_veh_eff = utils.drop_if_smaller_than_0_01(
        dm_veh_eff, cat_to_drop="Categories1", col_to_drop="Other"
    )

    # Clean grams CO2 category and perform weighted average
    # cols are e.g '0 - 50 g' -> '0-50' -> 25
    dm_veh_eff.rename_col_regex(" g", "", dim="Categories2")
    dm_veh_eff.rename_col_regex(" ", "", dim="Categories2")
    dm_veh_eff.rename_col("301ormore", "301‒350", dim="Categories2")
    dm_veh_eff.rename_col("Upto50", "0‒50", dim="Categories2")
    cat2_list_old = dm_veh_eff.col_labels["Categories2"]
    co2_km = []
    for i in range(len(cat2_list_old)):
        old_cat = cat2_list_old[i]
        new_cat = (float(old_cat.split("‒")[0]) + float(old_cat.split("‒")[1])) / 2
        co2_km.append(new_cat)
    co2_arr = np.array(co2_km)
    dm_veh_eff.normalise(dim="Categories2", inplace=True)
    dm_veh_eff.array = (
        dm_veh_eff.array * co2_arr[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    )
    dm_veh_eff.group_all(dim="Categories2")

    dm_veh_eff.change_unit(var_name, 1, old_unit="%", new_unit="gCO2/km")

    for i in range(2):
        window_size = 3  # Change window size to control the smoothing effect
        data_smooth = moving_average(
            dm_veh_eff.array, window_size, axis=dm_veh_eff.dim_labels.index("Years")
        )
        dm_veh_eff.array[:, 1:-1, ...] = data_smooth

    # Add LDV
    dm_veh_eff_LDV = DataMatrix.based_on(
        dm_veh_eff.array[..., np.newaxis],
        dm_veh_eff,
        change={"Categories2": ["LDV"]},
        units=dm_veh_eff.units,
    )
    dm_veh_eff_LDV.switch_categories_order()
    dm_veh_eff_LDV.rename_col("Total_Country", "Switzerland", dim="Country")

    dm_veh_eff_LDV.filter({"Years": years_ots}, inplace=True)
    return dm_veh_eff_LDV


def get_vehicle_efficiency_ofs(table_id, file, years_ots, var_name):
    # New fleet data are heavy, download them only once
    try:
        with open(file, "rb") as handle:
            dm_veh_eff = pickle.load(handle)
            print(
                f"The vehicle efficienty is read from file {file}. Delete it if you want to update data from api."
            )
    except OSError:
        structure, title = get_data_api_CH(table_id, mode="example", language="fr")
        i = 0
        # The table is too big to be downloaded at once
        for eu_class in structure["Classe d'émission selon l'UE"]:
            for part in structure["Filtre à particules"]:
                i = i + 1
                filtering = {
                    "Année": structure["Année"],
                    "Carburant": structure["Carburant"],
                    "Puissance": structure["Puissance"],
                    "Canton": ["Suisse", "Vaud"],
                    "Classe d'émission selon l'UE": eu_class,
                    "Émissions de CO2 par km (NEDC)": structure[
                        "Émissions de CO2 par km (NEDC)"
                    ],
                    "Filtre à particules": part,
                }

                mapping_dim = {
                    "Country": "Canton",
                    "Years": "Année",
                    "Variables": "Puissance",
                    "Categories1": "Carburant",
                    "Categories2": "Émissions de CO2 par km (NEDC)",
                }

                # Extract new fleet
                dm_veh_eff_cl = get_data_api_CH(
                    table_id,
                    mode="extract",
                    filter=filtering,
                    mapping_dims=mapping_dim,
                    units=["gCO2/km"] * len(structure["Puissance"]),
                    language="fr",
                )
                dm_veh_eff_cl.array = np.nan_to_num(dm_veh_eff_cl.array)

                if dm_veh_eff_cl is None:
                    raise ValueError(f"API returned None for {eu_class}")
                if i == 1:
                    dm_veh_eff = dm_veh_eff_cl.copy()
                else:
                    dm_veh_eff.array = dm_veh_eff.array + dm_veh_eff_cl.array

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, "wb") as handle:
            pickle.dump(dm_veh_eff, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Distribute Inconnu on other categories based on their share
    cat_other = [
        cat for cat in dm_veh_eff.col_labels["Categories2"] if cat != "Inconnu"
    ]
    dm_other = dm_veh_eff.filter({"Categories2": cat_other}, inplace=False)
    dm_other.normalise(dim="Categories2", inplace=True)
    idx = dm_veh_eff.idx
    arr_inc = dm_veh_eff.array[:, :, :, :, idx["Inconnu"], np.newaxis] * dm_other.array
    dm_veh_eff.drop(dim="Categories2", col_label="Inconnu")
    dm_veh_eff.array = dm_veh_eff.array + arr_inc

    # Remove fuel type "Autre" (there are only very few car in this category)
    dm_veh_eff.drop(dim="Categories1", col_label="Autre")

    # Group categories1 according to model
    map_cat = {
        "ICE-diesel": ["Diesel", "Diesel-électrique: hybride normal"],
        "ICE-gasoline": ["Essence", "Essence-électrique: hybride normal"],
        "ICE-gas": ["Gaz (monovalent et bivalent)"],
        "BEV": ["Électrique"],
        "FCEV": ["Hydrogène"],
        "PHEV-diesel": ["Diesel-électrique: hybride rechargeable"],
        "PHEV-gasoline": ["Essence-électrique: hybride rechargeable"],
    }
    dm_veh_eff.groupby(map_cat, dim="Categories1", inplace=True)

    # Do this to have realistic curves
    mask = dm_veh_eff.array == 0
    dm_veh_eff.array[mask] = np.nan

    # Flat extrapolation
    years_to_add = [
        year for year in years_ots if year not in dm_veh_eff.col_labels["Years"]
    ]
    dm_veh_eff.add(np.nan, dummy=True, col_label=years_to_add, dim="Years")
    dm_veh_eff.sort(dim="Years")
    dm_veh_eff.fill_nans(dim_to_interp="Years")

    dm_veh_eff.groupby({var_name: ".*"}, dim="Variables", regex=True, inplace=True)

    # Clean grams CO2 category and perform weighted average
    # cols are e.g '0 - 50 g' -> '0-50' -> 25
    dm_veh_eff.rename_col_regex(" g", "", dim="Categories2")
    dm_veh_eff.rename_col_regex(" ", "", dim="Categories2")
    dm_veh_eff.rename_col("Plusde300", "300-350", dim="Categories2")
    cat2_list_old = dm_veh_eff.col_labels["Categories2"]
    co2_km = []
    for i in range(len(cat2_list_old)):
        old_cat = cat2_list_old[i]
        new_cat = float(old_cat.split("-")[0]) + float(old_cat.split("-")[1]) / 2
        co2_km.append(new_cat)
    co2_arr = np.array(co2_km)
    dm_veh_eff.normalise(dim="Categories2", inplace=True)
    dm_veh_eff.array = (
        dm_veh_eff.array * co2_arr[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    )
    dm_veh_eff.group_all(dim="Categories2")

    dm_veh_eff.change_unit(var_name, 1, old_unit="%", new_unit="gCO2/km")

    for i in range(2):
        window_size = 3  # Change window size to control the smoothing effect
        data_smooth = moving_average(
            dm_veh_eff.array, window_size, axis=dm_veh_eff.dim_labels.index("Years")
        )
        dm_veh_eff.array[:, 1:-1, ...] = data_smooth

    # Add LDV
    dm_veh_eff_LDV = DataMatrix.based_on(
        dm_veh_eff.array[..., np.newaxis],
        dm_veh_eff,
        change={"Categories2": ["LDV"]},
        units=dm_veh_eff.units,
    )
    dm_veh_eff_LDV.switch_categories_order()
    dm_veh_eff_LDV.rename_col("Suisse", "Switzerland", dim="Country")

    return dm_veh_eff_LDV


def get_difference_between_col_label(dm1, dm2, dim):
    return [i for i in dm1.col_labels[dim] if i not in dm2.col_labels[dim]]


def add_missing_data_for_merging(dm1: DataMatrix, dm2: DataMatrix, dim: str):
    dm1.col_labels[dim]

    only_dm1 = get_difference_between_col_label(dm1, dm2, dim)
    only_dm2 = get_difference_between_col_label(dm2, dm1, dim)

    if len(only_dm1) != 0:
        dm2.add(np.nan, dim, col_label=only_dm1, dummy=True)

    if len(only_dm2) != 0:
        dm1.add(np.nan, dim, col_label=only_dm1, dummy=True)

    return dm1, dm2


def get_new_vehicle_efficiency_elec(file: str, agency: str, dataflow: str, var_name):
    """
    Extract data from https://stats.swiss/vis?lc=en&df[ds]=disseminate&df[id]=DF_IVS_1_EMISSION&df[ag]=CH1.MFZ_IVS&dq=5%2B22._T._T.N._T._T%2BPC%2BPH%2BDC%2BDH%2BHP%2BHD%2BGA%2B_O.1%2B2%2B3%2B4%2B5%2B6%2B7%2B_Z%2B_U%2B_T._T.A&pd=2005%2C2025&to[TIME_PERIOD]=false&pg=0
    Get MJ/100km  for BEVfrom this database.
    However data is available only from 2021.

    Args:
        file (str): filepath where the pickle is saved
        agency (str): Agency to call the api
        dataflow (str): dataflow to call the api
        var_name (str, optional): _description_. Defaults to "tra_passenger_veh-efficiency_fleet".

    Raises:
        ValueError: raise error if call to api failed

    Returns:
        DataMatrix: Efficiency datamatrix
    """
    try:
        with open(file, "rb") as handle:
            dm_veh_eff = pickle.load(handle)
            print(
                f"The vehicle efficienty is read from file {file}. Delete it if you want to update data from api."
            )
    except OSError:
        structure, title = get_data_api_swiss_stats(agency, dataflow, mode="example")
        i = 0
        ibis = 0
        fuel_list = ["Battery electric vehicle (BEV)"]
        consumption_list = [
            cons
            for cons in structure["UV_RV_ECONSUMPTION_WLTP_PC"]
            if cons not in ["Total"]
        ]

        CO2_emission_list = ["Total"]

        canton_list = [
            "Total",
            "Vaud",
            "Fribourg",
            "Schwyz",
        ]

        mapping_dim = {
            "Country": "UV_HGDE_KT",
            "Years": "TIME_PERIOD",
            "Variables": "UV_RV_CO2_WLTP",
            "Categories1": "UV_RV_FUEL",
            "Categories2": "UV_RV_ECONSUMPTION_WLTP_PC",
        }
        for i, canton in enumerate(canton_list):
            filtering = {
                "TIME_PERIOD": structure["TIME_PERIOD"],
                "UV_RV_FUEL": fuel_list,
                "UV_RV_OWNER_TYPE": ["Total"],
                "UV_RV_REGISTRATION_TYPE": ["First registrations of new vehicles"],
                "UV_RV_OWNER_AGE": ["Total"],
                "UV_RV_EMPTY_WEIGHT": ["Total"],
                "UV_RV_ECONSUMPTION_WLTP_PC": consumption_list,
                "UV_HGDE_KT": [canton],
                "UV_RV_CO2_WLTP": CO2_emission_list,
                "FREQ": ["Annual"],
            }
            # Extract new fleet
            dm_veh_eff_canton = get_data_api_swiss_stats(
                agency,
                dataflow,
                mode="extract",
                filter=filtering,
                mapping_dims=mapping_dim,
                units=["number"] * len(CO2_emission_list),
            )
            if dm_veh_eff_canton is None:
                raise ValueError(f"API returned None for {agency},{dataflow}, {canton}")

            dm_veh_eff_canton.rename_col("Total_Variables", var_name, dim="Variables")
            if i == 0:
                dm_veh_eff = dm_veh_eff_canton.copy()
            else:
                dm_veh_eff, dm_veh_eff_canton = add_missing_data_for_merging(
                    dm_veh_eff, dm_veh_eff_canton, "Years"
                )
                dm_veh_eff.append(dm_veh_eff_canton.copy(), dim="Country")

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, "wb") as handle:
            pickle.dump(dm_veh_eff, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Group categories1 according to model
    dict_tech = {
        "BEV": ["Battery electric vehicle (BEV)"],
    }
    # Rename columns with dict tech
    dm_veh_eff.groupby(dict_tech, dim="Categories1", inplace=True)

    # Distribute Inconnu on other categories based on their share
    # Remove fuel type "Autre" (there are only very few car in this category)
    # Before 2022 data is mainly unknown so it is dropped as it is doesn't give us efficiency information
    dm_veh_eff = utils.drop_if_smaller_than_0_01(
        dm_veh_eff, cat_to_drop="Categories2", col_to_drop="Missing/Undefined/Unknown"
    )

    dm_veh_eff = clean_energy_dimension(dm_veh_eff, var_name)

    for i in range(2):
        window_size = 3  # Change window size to control the smoothing effect
        data_smooth = moving_average(
            dm_veh_eff.array, window_size, axis=dm_veh_eff.dim_labels.index("Years")
        )
        dm_veh_eff.array[:, 1:-1, ...] = data_smooth

    # Add LDV
    dm_veh_eff_LDV = DataMatrix.based_on(
        dm_veh_eff.array[..., np.newaxis],
        dm_veh_eff,
        change={"Categories2": ["LDV"]},
        units=dm_veh_eff.units,
    )
    dm_veh_eff_LDV.switch_categories_order()
    dm_veh_eff_LDV.rename_col("Total_Country", "Switzerland", dim="Country")
    dm_veh_eff_LDV.filter({"Years": years_ots}, inplace=True)
    return dm_veh_eff_LDV


def get_new_vehicle_efficiency_co2(file: str, agency: str, dataflow: str, var_name):
    """
    Extract data from https://stats.swiss/vis?lc=en&df[ds]=disseminate&df[id]=DF_IVS_1_EMISSION&df[ag]=CH1.MFZ_IVS&dq=5%2B22._T._T.N._T._T%2BPC%2BPH%2BDC%2BDH%2BHP%2BHD%2BGA%2B_O.1%2B2%2B3%2B4%2B5%2B6%2B7%2B_Z%2B_U%2B_T._T.A&pd=2005%2C2025&to[TIME_PERIOD]=false&pg=0
    Get CO2/km from this database.
    However data is available only from 2021.

    Args:
        file (str): filepath where the pickle is saved
        agency (str): Agency to call the api
        dataflow (str): dataflow to call the api
        var_name (str, optional): _description_. Defaults to "tra_passenger_veh-efficiency_fleet".

    Raises:
        ValueError: raise error if call to api failed

    Returns:
        DataMatrix: Efficiency datamatrix
    """
    try:
        with open(file, "rb") as handle:
            dm_veh_eff = pickle.load(handle)
            print(
                f"The vehicle efficienty is read from file {file}. Delete it if you want to update data from api."
            )
    except OSError:
        structure, title = get_data_api_swiss_stats(agency, dataflow, mode="example")
        i = 0
        fuel_list = [
            fuel
            for fuel in structure["UV_RV_FUEL"]
            if fuel
            not in [
                "Total",
                "No motor",
                "Battery electric vehicle (BEV)",
                "Fuel cell electric vehicle (FCEV)",
            ]
        ]
        consumption_list = ["Total"]

        CO2_emission_list = [
            emis
            for emis in structure["UV_RV_CO2_WLTP"]
            if emis not in ["Total", "Not applicable"]
        ]
        canton_list = [
            "Total",
            "Vaud",
            "Fribourg",
            "Schwyz",
        ]

        mapping_dim = {
            "Country": "UV_HGDE_KT",
            "Years": "TIME_PERIOD",
            "Variables": "UV_RV_ECONSUMPTION_WLTP_PC",
            "Categories1": "UV_RV_FUEL",
            "Categories2": "UV_RV_CO2_WLTP",
        }
        for i, canton in enumerate(canton_list):
            filtering = {
                "TIME_PERIOD": structure["TIME_PERIOD"],
                "UV_RV_FUEL": fuel_list,
                "UV_RV_OWNER_TYPE": ["Total"],
                "UV_RV_REGISTRATION_TYPE": ["First registrations of new vehicles"],
                "UV_RV_OWNER_AGE": ["Total"],
                "UV_RV_EMPTY_WEIGHT": ["Total"],
                "UV_RV_ECONSUMPTION_WLTP_PC": consumption_list,
                "UV_HGDE_KT": [canton],
                "UV_RV_CO2_WLTP": CO2_emission_list,
                "FREQ": ["Annual"],
            }
            # Extract new fleet
            dm_veh_eff_canton = get_data_api_swiss_stats(
                agency,
                dataflow,
                mode="extract",
                filter=filtering,
                mapping_dims=mapping_dim,
                units=["number"] * len(consumption_list),
            )
            if dm_veh_eff_canton is None:
                raise ValueError(f"API returned None for {agency},{dataflow}, {canton}")

            dm_veh_eff_canton.rename_col("Total_Variables", var_name, dim="Variables")
            if i == 0:
                dm_veh_eff = dm_veh_eff_canton.copy()
            else:
                dm_veh_eff.append(dm_veh_eff_canton.copy(), dim="Country")

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, "wb") as handle:
            pickle.dump(dm_veh_eff, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Group categories1 according to model
    dict_tech = {
        "ICE-diesel": ["Diesel: conventional", "Diesel: hybrid electric vehicle (HEV)"],
        "ICE-gasoline": [
            "Petrol: conventional",
            "Petrol: hybrid electric vehicle (HEV)",
        ],
        "PHEV-diesel": ["Plug-in hybrid electric vehicle (PHEV): diesel"],
        "PHEV-gasoline": ["Plug-in hybrid electric vehicle (PHEV): petrol"],
        "ICE-gas": ["Gas (mono- and bi-fuel)"],
    }
    # Rename columns with dict tech
    dm_veh_eff.groupby(dict_tech, dim="Categories1", inplace=True)

    # For BEV electric consumption check get_vehicle_electric_efficiency
    dm_veh_eff.drop(dim="Categories1", col_label="BEV")
    # # Do this to have realistic curves
    mask = dm_veh_eff.array == 0
    dm_veh_eff.array[mask] = np.nan

    # Flat extrapolation data is bad before 2016 so backstage it
    dm_veh_eff.filter({"Years": list(range(2017, years_ots[-1]))}, inplace=True)
    years_to_add = [
        year for year in years_ots if year not in dm_veh_eff.col_labels["Years"]
    ]
    dm_veh_eff.add(np.nan, dummy=True, col_label=years_to_add, dim="Years")
    dm_veh_eff.sort(dim="Years")
    dm_veh_eff.fill_nans(dim_to_interp="Years")

    # Distribute Inconnu on other categories based on their share
    # Remove fuel type "Autre" (there are only very few car in this category)
    dm_veh_eff = utils.drop_if_smaller_than_0_01(
        dm_veh_eff, cat_to_drop="Categories2", col_to_drop="No data/unknown"
    )

    dm_veh_eff = utils.drop_if_smaller_than_0_01(
        dm_veh_eff, cat_to_drop="Categories1", col_to_drop="Other"
    )

    # Clean grams CO2 category and perform weighted average
    # cols are e.g '0 - 50 g' -> '0-50' -> 25
    dm_veh_eff.rename_col_regex(" g", "", dim="Categories2")
    dm_veh_eff.rename_col_regex(" ", "", dim="Categories2")
    dm_veh_eff.rename_col("301ormore", "301‒350", dim="Categories2")
    dm_veh_eff.rename_col("Upto50", "0‒50", dim="Categories2")
    cat2_list_old = dm_veh_eff.col_labels["Categories2"]
    co2_km = []
    for i in range(len(cat2_list_old)):
        old_cat = cat2_list_old[i]
        new_cat = (float(old_cat.split("‒")[0]) + float(old_cat.split("‒")[1])) / 2
        co2_km.append(new_cat)
    co2_arr = np.array(co2_km)
    dm_veh_eff.normalise(dim="Categories2", inplace=True)
    dm_veh_eff.array = (
        dm_veh_eff.array * co2_arr[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    )
    dm_veh_eff.group_all(dim="Categories2")

    dm_veh_eff.change_unit(var_name, 1, old_unit="%", new_unit="gCO2/km")

    for i in range(2):
        window_size = 3  # Change window size to control the smoothing effect
        data_smooth = moving_average(
            dm_veh_eff.array, window_size, axis=dm_veh_eff.dim_labels.index("Years")
        )
        dm_veh_eff.array[:, 1:-1, ...] = data_smooth

    # Add LDV
    dm_veh_eff_LDV = DataMatrix.based_on(
        dm_veh_eff.array[..., np.newaxis],
        dm_veh_eff,
        change={"Categories2": ["LDV"]},
        units=dm_veh_eff.units,
    )
    dm_veh_eff_LDV.switch_categories_order()
    dm_veh_eff_LDV.rename_col("Total_Country", "Switzerland", dim="Country")

    dm_veh_eff_LDV.filter({"Years": years_ots}, inplace=True)
    return dm_veh_eff_LDV


def get_new_vehicle_efficiency_ofs(table_id, file, years_ots, var_name):
    # New fleet data are heavy, download them only once
    try:
        with open(file, "rb") as handle:
            dm_veh_eff = pickle.load(handle)
            print(
                f"The vehicle efficienty is read from file {file}. Delete it if you want to update data from api."
            )
    except OSError:
        structure, title = get_data_api_CH(table_id, mode="example", language="fr")
        i = 0
        # The table is too big to be downloaded at once
        for eu_class in structure["Classe d'émission selon l'UE"]:
            i = i + 1
            filtering = {
                "Année": structure["Année"],
                "Carburant": structure["Carburant"],
                "Puissance": structure["Puissance"],
                "Canton": ["Suisse", "Vaud"],
                "Classe d'émission selon l'UE": eu_class,
                "Émissions de CO2 par km (NEDC/WLTP)": structure[
                    "Émissions de CO2 par km (NEDC/WLTP)"
                ],
            }

            mapping_dim = {
                "Country": "Canton",
                "Years": "Année",
                "Variables": "Puissance",
                "Categories1": "Carburant",
                "Categories2": "Émissions de CO2 par km (NEDC/WLTP)",
            }

            # Extract new fleet
            dm_veh_eff_cl = get_data_api_CH(
                table_id,
                mode="extract",
                filter=filtering,
                mapping_dims=mapping_dim,
                units=["gCO2/km"] * len(structure["Puissance"]),
                language="fr",
            )
            dm_veh_eff_cl.array = np.nan_to_num(dm_veh_eff_cl.array)

            if dm_veh_eff_cl is None:
                raise ValueError(f"API returned None for {eu_class}")
            if i == 1:
                dm_veh_eff = dm_veh_eff_cl.copy()
            else:
                dm_veh_eff.array = dm_veh_eff.array + dm_veh_eff_cl.array

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, "wb") as handle:
            pickle.dump(dm_veh_eff, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Do this to have realistic curves
    mask = dm_veh_eff.array == 0
    dm_veh_eff.array[mask] = np.nan

    # Flat extrapolation
    years_to_add = [
        year for year in years_ots if year not in dm_veh_eff.col_labels["Years"]
    ]
    dm_veh_eff.add(np.nan, dummy=True, col_label=years_to_add, dim="Years")
    dm_veh_eff.sort(dim="Years")
    dm_veh_eff.fill_nans(dim_to_interp="Years")

    # Explore Inconnu category
    # -> The data seem to be good only from 2016 to 2020, still the "Inconnu" share is big
    dm_norm = dm_veh_eff.normalise(dim="Categories2", inplace=False)
    idx = dm_norm.idx
    for country in dm_veh_eff.col_labels["Country"]:
        for year in dm_veh_eff.col_labels["Years"]:
            for cat in dm_veh_eff.col_labels["Categories1"]:
                # If "Inconnu" is more than 20% remove the data points
                if (
                    dm_norm.array[idx[country], idx[year], 0, idx[cat], idx["Inconnu"]]
                    > 0.2
                ):
                    dm_veh_eff.array[idx[country], idx[year], 0, idx[cat], :] = np.nan

    for i in range(2):
        window_size = 3  # Change window size to control the smoothing effect
        data_smooth = moving_average(
            dm_veh_eff.array, window_size, axis=dm_veh_eff.dim_labels.index("Years")
        )
        dm_veh_eff.array[:, 1:-1, ...] = data_smooth

    # Distribute Inconnu on other categories based on their share
    cat_other = [
        cat for cat in dm_veh_eff.col_labels["Categories2"] if cat != "Inconnu"
    ]
    dm_other = dm_veh_eff.filter({"Categories2": cat_other}, inplace=False)
    dm_other.normalise(dim="Categories2", inplace=True)
    dm_other.array = np.nan_to_num(dm_other.array)
    idx = dm_veh_eff.idx
    arr_inc = (
        np.nan_to_num(dm_veh_eff.array[:, :, :, :, idx["Inconnu"], np.newaxis])
        * dm_other.array
    )
    dm_veh_eff.drop(dim="Categories2", col_label="Inconnu")
    dm_veh_eff.array = dm_veh_eff.array + arr_inc

    # Remove fuel type "Autre" (there are only very few car in this category)
    dm_veh_eff.drop(dim="Categories1", col_label="Autre")

    # Group categories1 according to model
    map_cat = {
        "ICE-diesel": ["Diesel", "Diesel-électrique: hybride normal"],
        "ICE-gasoline": ["Essence", "Essence-électrique: hybride normal"],
        "ICE-gas": ["Gaz (monovalent et bivalent)"],
        "BEV": ["Électrique"],
        "FCEV": ["Hydrogène"],
        "PHEV-diesel": ["Diesel-électrique: hybride rechargeable"],
        "PHEV-gasoline": ["Essence-électrique: hybride rechargeable"],
    }
    dm_veh_eff.groupby(map_cat, dim="Categories1", inplace=True)

    dm_veh_eff.groupby({var_name: ".*"}, dim="Variables", regex=True, inplace=True)

    # Clean grams CO2 category and perform weighted average
    # cols are e.g '0 - 50 g' -> '0-50' -> 25
    dm_veh_eff.rename_col_regex(" g", "", dim="Categories2")
    dm_veh_eff.rename_col_regex(" ", "", dim="Categories2")
    dm_veh_eff.rename_col("Plusde300", "300-350", dim="Categories2")
    cat2_list_old = dm_veh_eff.col_labels["Categories2"]
    co2_km = []
    for i in range(len(cat2_list_old)):
        old_cat = cat2_list_old[i]
        new_cat = float(old_cat.split("-")[0]) + float(old_cat.split("-")[1]) / 2
        co2_km.append(new_cat)
    dm_veh_eff.normalise(dim="Categories2", inplace=True)
    dm_veh_eff.array = dm_veh_eff.array * np.array(co2_km)
    dm_veh_eff.group_all(dim="Categories2")
    dm_veh_eff.change_unit(var_name, 1, old_unit="%", new_unit="gCO2/km")

    # Add LDV
    dm_veh_eff_LDV = DataMatrix.based_on(
        dm_veh_eff.array[..., np.newaxis],
        dm_veh_eff,
        change={"Categories2": ["LDV"]},
        units=dm_veh_eff.units,
    )
    dm_veh_eff_LDV.switch_categories_order()
    dm_veh_eff_LDV.rename_col("Suisse", "Switzerland", dim="Country")

    dm_veh_eff_LDV.filter({"Years": years_ots}, inplace=True)
    return dm_veh_eff_LDV
