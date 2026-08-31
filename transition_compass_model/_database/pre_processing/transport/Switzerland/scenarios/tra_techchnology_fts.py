import os

import pandas as pd
from processors.freight_efficiency_tech_share import _EP2050_PATH, EP2050_tech_to_model

from transition_compass_model.model.common.auxiliary_functions import (
    my_pickle_dump,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

rename_veh_cat = {
    "eBike": "bike",
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

    df_passenger = pd.read_excel(
        _EP2050_PATH, sheet_name="03 Fahrleistung", header=header_scenar
    ).iloc[:34, 1:]

    # Only keep passenger vehicle data
    df_passenger.replace(rename_veh_cat, inplace=True)
    df_passenger = df_passenger.loc[
        df_passenger["VehCat"].isin(rename_veh_cat.values()), :
    ]
    df_passenger.replace(EP2050_tech_to_model, inplace=True)

    df_grouped = df_passenger.groupby(["VehCat"], as_index=False).sum(numeric_only=True)

    return df_grouped


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


def read_ep2050_fleet(scenario="ZERO-Basis") -> pd.DataFrame:
    """Read ZERO-Basis road-freight vehicle kilometers from EP2050."""
    if scenario == "ZERO-Basis":
        header_scenar = 19

    df_passenger = pd.read_excel(
        _EP2050_PATH, sheet_name="02 Flottenbestand", header=header_scenar
    ).iloc[:56, 1:]

    # Only keep passenger vehicle data

    df_passenger[["VehCat", "Tech"]] = df_passenger["Segment"].apply(split_segment)

    df_passenger.loc[df_passenger["VehCat"].isin(["eBike", "eScooter"]), "Tech"] = "BEV"
    # Rename according to the dm name
    df_passenger.replace(rename_veh_cat, inplace=True)
    df_passenger = df_passenger.loc[
        df_passenger["VehCat"].isin(rename_veh_cat.values()), :
    ]

    df_passenger.replace(EP2050_tech_to_model, inplace=True)
    df_passenger["Tech"].fillna("ICE-diesel", inplace=True)
    df_grouped = df_passenger.groupby(["VehCat"], as_index=False).sum(numeric_only=True)
    return df_grouped


def run(DM_transport: DataMatrix, years_fts):
    ##### SAVE DATA #########
    this_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(this_dir, "../../../../data/datamatrix/transport.pickle")
    my_pickle_dump(DM_new=DM_transport, local_pickle_file=pickle_file)

    #### Demande de transport de stat vaud #####

    return DM_transport
