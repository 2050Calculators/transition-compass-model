import os

import numpy as np
import pandas as pd
from processors.freight_efficiency_tech_share import _EP2050_PATH, EP2050_tech_to_model

import transition_compass_model._database.pre_processing.transport.Switzerland.get_data_functions.transport_energy as get_data
from transition_compass_model._database.pre_processing.transport.Switzerland.get_data_functions.demand_pkm_vkm import (
    extract_EP2050_transport_vkm_demand,
)
from transition_compass_model.model.common.auxiliary_functions import (
    linear_fitting,
    my_pickle_dump,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

rename_veh_cat = {
    "eBike": "2W",
    "motorcycle": "2W",
    "eScooter": "2W",
    "pass. car": "LDV",
    "PC": "LDV",
    "coach": "bus",
    "Coach": "bus",
    "urban bus": "bus",
    "MC": "2W",
    "Moped": "2W",
    "Ubus": "bus",
}


# ---------------------------------------------------------------------------
# EP2050 readers
# ---------------------------------------------------------------------------
def read_ep2050_vkm(scenario="ZERO-Basis") -> pd.DataFrame:
    """Read ZERO-Basis road-freight vehicle kilometers from EP2050.

    The data covers light commercial vehicles (LCV) and heavy goods vehicles
    (HGV), disaggregated by fuel or propulsion technology. Energy values are
    expressed in petajoules (PJ) per year.

    Returns:
        pd.DataFrame : df containing:

        - ``Fahrzeugart``: vehicle category (``LCV`` or ``HGV``)
        - ``Treibstoff``: fuel or propulsion technology
        - year columns from 1990 to 2060: energy consumption in PJ

        Fuel names are normalized to the model terminology, including
        ``ICE-gasoline`` for petrol and ``BEV`` for electricity.
    """
    if scenario == "ZERO-Basis":
        header_scenar = 19
    elif scenario == "ZERO-B":
        header_scenar = 105

    df_passenger = pd.read_excel(
        _EP2050_PATH, sheet_name="03 Fahrleistung", header=header_scenar
    ).iloc[:34, 1:]

    # Only keep passenger vehicle data
    df_passenger.replace(rename_veh_cat, inplace=True)
    df_passenger = df_passenger.loc[
        df_passenger["VehCat"].isin(rename_veh_cat.values()), :
    ]
    df_passenger.replace(EP2050_tech_to_model, inplace=True)

    df_passenger.rename(columns={"VehCat": "mode", "Technology": "tech"}, inplace=True)
    df_grouped = df_passenger.groupby(["mode", "tech"], as_index=False).sum(
        numeric_only=True
    )
    dm_passenger = passenger_df_to_dm(
        df_grouped, var_name="tra_passenger_vkm", unit="Mvkm"
    )

    return dm_passenger


tech_patterns = [
    "PHEV diesel",
    "PHEV petrol",
    "CNG/petrol",
    "FuelCell",
    "Hybrid",
    "Electric",
    "BEV",
    "FFV",
    "LPG/petrol",
    "diesel",
    "petrol",
    "LNG",
    "CNG",
]


def split_segment(segment):
    segment = segment.replace("<br>", "").strip()

    # Vehicle type
    vehicle_types = ["PC", "LCV", "Coach", "Ubus", "MC", "Moped", "eBike", "eScooter"]

    vehicle_type = next(
        (v for v in vehicle_types if segment.startswith(v + " ") or segment == v), None
    )

    remainder = segment[len(vehicle_type) :].strip() if vehicle_type else segment

    # Technology
    tech = next((t for t in tech_patterns if remainder.startswith(t)), None)

    # If no technology is specified, e.g. "Coach Std <=18t"

    return pd.Series([vehicle_type, tech])


def passenger_df_to_dm(
    df_passenger: pd.DataFrame,
    var_name: str,
    unit: str = "MJ/km",
    country: str = "Switzerland",
) -> DataMatrix:
    """
    Convert a wide EP2050 passenger table like:
        VehCat | Treibstoff | 1990 | 1991 | ...
    into a DataMatrix using DataMatrix.create_from_df(..., num_cat=2).
    """
    if df_passenger.empty:
        raise ValueError("df_passenger cannot be empty.")
    value_col = f"{var_name} [{unit}]"
    # 1) Wide -> long
    df_long = df_passenger.melt(
        id_vars=["mode", "tech"],
        var_name="Years",
        value_name=value_col,
    )

    # 2) Add Country + normalize Years
    df_long["Country"] = country
    df_long["Years"] = pd.to_numeric(df_long["Years"], errors="raise").astype(int)

    # Keep only the columns expected by DataMatrix
    df_long = df_long[["Country", "Years", value_col, "mode", "tech"]]

    # 3) Long -> wide again, but with category tuples as column names

    df_dm = df_long.pivot(
        index=["Country", "Years"], columns=["mode", "tech"], values=value_col
    ).reset_index()

    new_cols = ["Country", "Years"]

    for mode, tech in df_dm.columns[2:]:
        new_cols.append(f"{var_name}_{mode}_{tech}[MJ/km]")

    df_dm.columns = new_cols
    # 5) This is the important part
    return DataMatrix.create_from_df(df_dm, num_cat=2)


def read_ep2050_fleet(scenario="ZERO-Basis") -> pd.DataFrame:
    """Read ZERO-Basis road-freight vehicle kilometers from EP2050."""
    if scenario == "ZERO-Basis":
        header_scenar = 19
    elif scenario == "ZERO-B":
        header_scenar = 114

    df_passenger = pd.read_excel(
        _EP2050_PATH, sheet_name="02 Flottenbestand", header=header_scenar
    ).iloc[:56, 1:]

    # Only keep passenger vehicle data

    df_passenger[["mode", "tech"]] = df_passenger["Segment"].apply(split_segment)

    df_passenger.loc[df_passenger["mode"].isin(["eBike", "eScooter"]), "Tech"] = "BEV"
    # Rename according to the dm name
    df_passenger.replace(rename_veh_cat, inplace=True)
    df_passenger = df_passenger.loc[
        df_passenger["mode"].isin(rename_veh_cat.values()), :
    ]

    df_passenger.replace(EP2050_tech_to_model, inplace=True)
    df_passenger["tech"].fillna("ICE-diesel", inplace=True)
    df_grouped = df_passenger.groupby(["mode", "tech"], as_index=False).sum(
        numeric_only=True
    )

    df_grouped.drop(columns=["Unnamed: 2", "Unnamed: 3"], inplace=True)
    dm_passenger = passenger_df_to_dm(
        df_grouped, var_name="tra_passenger_fleet", unit="number"
    )
    return dm_passenger


def read_ep2050_energy(scenario="ZERO-Basis") -> pd.DataFrame:
    if scenario == "ZERO-Basis":
        header_scenar = 19
    elif scenario == "ZERO-B":
        header_scenar = 89
    df_passenger = pd.read_excel(
        _EP2050_PATH, sheet_name="04 Energieverbrauch Strasse", header=header_scenar
    ).iloc[:26, 1:]
    df_passenger.rename(
        columns={"Fahrzeugart": "mode", "Treibstoff": "tech"}, inplace=True
    )

    df_passenger.drop(columns=["Unnamed: 3"], inplace=True)
    df_passenger.replace(rename_veh_cat, inplace=True)
    df_passenger.replace(EP2050_tech_to_model, inplace=True)

    df_grouped = df_passenger.groupby(["mode", "tech"], as_index=False).sum(
        numeric_only=True
    )

    dm_passenger = passenger_df_to_dm(
        df_grouped, var_name="tra_passenger_energy_consumption", unit="PJ"
    )

    #
    return dm_passenger


def get_ep2050_vkm_data_zero_b(this_dir):
    # VKM demand for LDV, 2W, bus by technology from EP2050
    # EP2050+_Detailergebnisse 2020-2060_Verkehrssektor_alle Szenarien_2022-04-12
    file_url = "https://www.bfe.admin.ch/bfe/de/home/politik/energieperspektiven-2050-plus.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvMTA0NDE=.html"
    zip_name = os.path.join(this_dir, "../data/EP2050_sectors.zip")
    file_pickle = os.path.join(this_dir, "../data/tra_EP2050_vkm_demand_private.pickle")
    dm_vkm_private = extract_EP2050_transport_vkm_demand(
        file_url, zip_name, file_pickle, scenario="ZERO-B"
    )
    return dm_vkm_private


def get_ep2050_energy_data_zero_b(this_dir):
    # Get energy
    file_url = "https://www.bfe.admin.ch/bfe/de/home/politik/energieperspektiven-2050-plus.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvMTA0NDE=.html"
    zip_name = os.path.join(this_dir, "../data/EP2050_sectors.zip")
    file_pickle = os.path.join(
        this_dir, "../data/tra_EP2050_energy_demand_private_scen_zero-B.pickle"
    )
    dm_energy_ep2050 = get_data.extract_EP2050_transport_energy_demand(
        file_url, zip_name, file_pickle, scenario="ZERO-B"
    )
    dm_energy_ep2050.groupby(
        {"ICE-gasoline": ["biogasoline", "gasoline"]}, "Categories2", inplace=True
    )
    dm_energy_ep2050.change_unit(
        "tra_energy_demand", old_unit="TWh", new_unit="MJ", factor=3.6e9, operator="*"
    )
    return dm_energy_ep2050


def car_efficiency_techno_ep2050(DM_transport: DataMatrix, lev):
    # lever to modify
    dm_fts = DM_transport["fts"]["passenger_veh-efficiency_new"][lev].copy()

    # Importin data
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # import DM
    dm_energy_ep2050 = get_ep2050_energy_data_zero_b(this_dir)
    dm_vkm_ep2050 = get_ep2050_vkm_data_zero_b(this_dir)

    # Only keep passenger data for fts years
    modes = ["2W", "LDV", "bus"]
    techs = ["BEV", "FCEV", "ICE-diesel", "ICE-gas", "ICE-gasoline"]
    dm_energy_ep2050 = dm_energy_ep2050.filter(
        {
            "Categories1": modes,
            "Categories2": techs,
        }  # , "Years" : dm_fts.col_labels["Years"]
    )
    dm_vkm_ep2050 = dm_vkm_ep2050.filter(
        {
            "Categories1": modes,
            "Categories2": techs,
        }  # , "Years" : dm_fts.col_labels["Years"]
    )

    # Create efficency matrix
    dm_efficiency_ep2050 = dm_energy_ep2050.copy()
    dm_efficiency_ep2050.append(dm_vkm_ep2050, "Variables")
    dm_efficiency_ep2050.operation(
        "tra_energy_demand",
        "/",
        "tra_vkm_demand",
        "Variables",
        "efficiency_fleet_ep2050",
        "MJ/km",
    )

    # new_array, dim, col_label, unit=None, dummy=False)
    # dm_efficiency_ep2050.add(np.nan, "Categories2", col_label = "CEV", dummy = True)

    # idx_efficiency = dm_efficiency_ep2050.idx
    # dm_efficiency_ep2050.array[:,:,:,idx_efficiency["bus"], idx_efficiency["CEV"]] =dm_efficiency_ep2050.array[:,:,:,idx_efficiency["bus"], idx_efficiency["BEV"]]
    # dm_efficiency_ep2050.array[:,:,:,idx_efficiency["bus"], idx_efficiency["BEV"]] = np.nan

    # For this we consider that the efficiency ratio between the different technologies stay constant # TODO : search for better proxy
    tech_to_interpolate = [
        tech
        for tech in dm_fts.col_labels["Categories2"]
        if tech not in dm_efficiency_ep2050.col_labels["Categories2"]
    ]
    tech_to_interpolate.remove("mt")
    tech_to_interpolate.remove("kerosene")

    idx_fts = dm_fts.idx
    ratio_eff = {}
    for tech in tech_to_interpolate:
        ratio_eff[tech] = (
            dm_fts.array[..., idx_fts[tech]] / dm_fts.array[..., idx_fts["ICE-diesel"]]
        )

    dm_efficiency_ep2050_fts = dm_efficiency_ep2050.filter(
        {"Years": dm_fts.col_labels["Years"]}
    )
    idx_ep2050_fts = dm_efficiency_ep2050_fts.idx

    idx_road_fts = [idx_fts["2W"], idx_fts["LDV"], idx_fts["bus"]]
    idx_road_ep2050 = [
        idx_ep2050_fts["2W"],
        idx_ep2050_fts["LDV"],
        idx_ep2050_fts["bus"],
    ]

    common_tech = common_tech = [
        tech
        for tech in dm_vkm_ep2050.col_labels["Categories2"]
        if tech in dm_efficiency_ep2050.col_labels["Categories2"]
    ]
    idx_tech_fts = [idx_fts[tech] for tech in common_tech]
    idx_tech_ep2050 = [idx_ep2050_fts[tech] for tech in common_tech]

    mode_fts, tech_fts = np.ix_(idx_road_fts, idx_tech_fts)

    mode_ep2050, tech_ep2050 = np.ix_(idx_road_ep2050, idx_tech_ep2050)

    # Use ep2050 values for the future scenario
    dm_fts.array[idx_fts["Switzerland"], :, 0, mode_fts, tech_fts] = (
        dm_efficiency_ep2050_fts.array[
            idx_ep2050_fts["Switzerland"],
            :,
            idx_ep2050_fts["tra_passenger_efficiency"],
            mode_ep2050,
            tech_ep2050,
        ]
    )

    # dm_fts.array[idx_fts["Switzerland"], :,0, idx_road_fts[:,None], idx_tech_fts ] = (

    #  dm_efficiency_ep2050_fts.array[idx_ep2050_fts["Switzerland"], :,idx_ep2050_fts["tra_passenger_efficiency"], idx_road_ep2050[:,None], idx_tech_ep2050 ]
    # )

    return DM_transport


def car_efficiency_techno(DM_transport, lev):
    """Scenario developped by martin simon in DLS pespective (smaller car and berline).
    The 1/3 optimization is due to the car size redution"""

    dm_new_eff_4 = DM_transport["fts"]["passenger_veh-efficiency_new"][4].copy()
    dm_new_eff_ots = DM_transport["ots"]["passenger_veh-efficiency_new"].copy()

    # PCV: on prend les hypothèses d'amélioration ci dessous (source: canton de Vaud).
    reduction_2050_thermique = 1 - 0.39
    reduction_2050_electrique = 1 - 0.13
    reduction_2050_PHEV = (
        0.5 * reduction_2050_electrique + 0.5 * reduction_2050_thermique
    )
    reduction_map = {
        "BEV": reduction_2050_electrique,
        "ICE-diesel": reduction_2050_thermique,
        "ICE-gasoline": reduction_2050_thermique,
        "PHEV-diesel": reduction_2050_PHEV,
        "PHEV-gasoline": reduction_2050_PHEV,
    }
    # Scénario 4:
    # on applique la réduction de 2/3 pour 2025 et 2050 due à la réduction de la taille des véhicules.
    reduction_map_4 = {cle: valeur * 2 / 3 for cle, valeur in reduction_map.items()}

    idx = dm_new_eff_4.idx
    idx0 = dm_new_eff_ots.idx
    for cat, reduction_2050 in reduction_map_4.items():
        dm_new_eff_4.array[
            :,
            idx[2050],
            idx["tra_passenger_veh-efficiency_new"],
            idx["LDV"],
            idx[cat],
        ] = (
            dm_new_eff_ots.array[
                :,
                idx0[2023],
                idx0["tra_passenger_veh-efficiency_new"],
                idx0["LDV"],
                idx0[cat],
            ]
            * reduction_2050
        )
        dm_new_eff_4.array[
            :,
            1 : idx[2050],
            idx["tra_passenger_veh-efficiency_new"],
            idx["LDV"],
            idx[cat],
        ] = np.nan
        dm_new_eff_4.array[
            :,
            idx[2025],
            idx["tra_passenger_veh-efficiency_new"],
            idx["LDV"],
            idx[cat],
        ] = (
            2
            / 3
            * dm_new_eff_ots.array[
                :,
                idx0[2023],
                idx0["tra_passenger_veh-efficiency_new"],
                idx0["LDV"],
                idx0[cat],
            ]
        )

    # on réduit l'intensité énergétique des BEV car on vend 50% de véhicules intermédiaires dès 2025
    prop_VUS = 0.5
    efficiency_VUS = 0.11

    efficiency_2025_bev = dm_new_eff_4.array[
        :,
        idx[2025],
        idx["tra_passenger_veh-efficiency_new"],
        idx["LDV"],
        idx["BEV"],
    ]
    efficiency_2025_bev_vus = (
        efficiency_2025_bev * (1 - prop_VUS) + efficiency_VUS * prop_VUS
    )
    dm_new_eff_4.array[
        :,
        idx[2025],
        idx["tra_passenger_veh-efficiency_new"],
        idx["LDV"],
        idx["BEV"],
    ] = efficiency_2025_bev_vus

    efficiency_2050_bev = dm_new_eff_4.array[
        :,
        idx[2050],
        idx["tra_passenger_veh-efficiency_new"],
        idx["LDV"],
        idx["BEV"],
    ]
    efficiency_2050_bev_vus = (
        efficiency_2050_bev * 2 / 3 * (1 - prop_VUS) + efficiency_VUS * prop_VUS
    )
    dm_new_eff_4.array[
        :,
        idx[2050],
        idx["tra_passenger_veh-efficiency_new"],
        idx["LDV"],
        idx["BEV"],
    ] = efficiency_2050_bev_vus

    linear_fitting(dm_new_eff_4, dm_new_eff_4.col_labels["Years"])
    DM_transport["fts"]["passenger_veh-efficiency_new"][4] = dm_new_eff_4

    return DM_transport


def run(DM_transport: DataMatrix, lev: int = 4) -> DataMatrix:
    # DM_transport = car_efficiency_techno_ep2050(DM_transport,lev)

    DM_transport = car_efficiency_techno(DM_transport, lev)

    ##### SAVE DATA #########
    this_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(this_dir, "../../../../data/datamatrix/transport.pickle")
    my_pickle_dump(DM_new=DM_transport, local_pickle_file=pickle_file)

    #### Demande de transport de stat vaud #####

    return DM_transport
