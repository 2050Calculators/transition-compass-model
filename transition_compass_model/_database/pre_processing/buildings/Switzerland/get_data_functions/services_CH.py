import os
import pickle

import numpy as np
import pandas as pd

from transition_compass_model._database.pre_processing.api_routines_CH import (
    get_data_api_CH,
)
from transition_compass_model.model.common.auxiliary_functions import (
    dm_add_missing_variables,
    linear_fitting,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def _extract_type_class_shares_from_cadastre(this_dir):
    """Return P(type | class) shares from Vaud cadastre, shape (n_types, n_classes)."""
    cadastre_path = os.path.join(this_dir, "../data/cadastre_regener.csv")
    df = pd.read_csv(cadastre_path, low_memory=False)

    non_res_sia = [
        "Administration",
        "Commerce",
        "Ecoles",
        "Hopitaux",
        "Restauration",
        "Lieux_de_rassemblement",
        "Installations_sportives",
    ]
    type_map = {
        "Administration": "offices",
        "Commerce": "trade",
        "Ecoles": "education",
        "Hopitaux": "health",
        "Restauration": "hotels",
        "Lieux_de_rassemblement": "other",
        "Installations_sportives": "other",
    }

    df = df[df["aff_sia"].isin(non_res_sia)].copy()
    df = df[df["gebf"].notna() & (df["gebf"] > 0)]
    df["model_type"] = df["aff_sia"].map(type_map)

    def _year_to_class(yr):
        if pd.isna(yr):
            return None
        yr = int(yr)
        if yr < 1947:
            return "F"
        elif yr <= 1975:
            return "E"
        elif yr <= 1990:
            return "D"
        elif yr <= 2019:
            return "C"
        else:
            return "B"

    df["energy_class"] = df["gbauj"].apply(_year_to_class)
    df = df[df["energy_class"].notna()]

    area_by_type_class = (
        df.groupby(["model_type", "energy_class"])["gebf"]
        .sum()
        .unstack("energy_class")
        .fillna(0)
    )
    # Normalize columns so shares sum to 1 per class
    shares = area_by_type_class.div(area_by_type_class.sum(axis=0), axis=1)
    return shares  # DataFrame indexed by model_type, columns = energy class


def extract_services_floor_area_EP2050(this_dir, years_ots):
    """Extract national non-residential floor area by building type and energy class."""
    file_path = os.path.join(
        this_dir,
        "../data/EP2050_sectors/EP2050+_Szenarienergebnisse_Details_Nachfragesektoren/"
        "EP2050+_Detailergebnisse 2020-2060_Dienstleistung_alle Szenarien_2022-10-20.xlsx",
    )
    df = pd.read_excel(file_path, sheet_name="04 EBF-Baualter", header=None)

    # Locate year columns (header row containing 2000)
    year_row_idx = df.apply(lambda row: 2000 in row.values, axis=1).idxmax()
    year_cols = {
        int(v): i
        for i, v in enumerate(df.iloc[year_row_idx])
        if isinstance(v, (int, float)) and not pd.isna(v) and 2000 <= int(v) <= 2060
    }
    years_ep2050 = sorted(year_cols.keys())

    # Construction period rows map to energy classes (labels in column 1)
    period_to_class = {
        "vor 1946": "F",
        "1947-1975": "E",
        "1976-1990": "D",
        "1991-2019": "C",
        "ab 2020": "B",
    }
    energy_classes = ["B", "C", "D", "E", "F"]

    # Extract floor area (Tsd. m²) per class per year
    class_area = {}  # class → np.array over years_ep2050
    for period, cls in period_to_class.items():
        row_mask = df.iloc[:, 1].astype(str).str.strip() == period
        row_idx = df.index[row_mask][0]
        values = np.array(
            [df.iloc[row_idx, year_cols[yr]] for yr in years_ep2050], dtype=float
        )
        class_area[cls] = values * 1000  # Tsd. m² → m²

    # Get P(type | class) shares from Vaud cadastre
    type_shares = _extract_type_class_shares_from_cadastre(this_dir)
    bld_types = sorted(
        type_shares.index.tolist()
    )  # ['education','health','hotels','offices','other','trade']

    dm_ch = DataMatrix(
        col_labels={
            "Country": ["Switzerland"],
            "Years": years_ep2050,
            "Variables": ["bld_floor-area_services"],
            "Categories1": bld_types,
            "Categories2": energy_classes,
        },
        units={"bld_floor-area_services": "m2"},
    )

    for i, yr in enumerate(years_ep2050):
        for cls in energy_classes:
            total_cls = class_area[cls][i]
            for bld_type in bld_types:
                share = (
                    type_shares.loc[bld_type, cls]
                    if cls in type_shares.columns
                    else 0.0
                )
                dm_ch["Switzerland", yr, "bld_floor-area_services", bld_type, cls] = (
                    total_cls * share
                )

    missing_years = [yr for yr in years_ots if yr not in years_ep2050]
    if missing_years:
        dm_add_missing_variables(dm_ch, {"Years": missing_years})
        dm_ch.sort("Years")
    linear_fitting(dm_ch, years_ots, based_on=list(range(2000, 2010)))
    return dm_ch.filter({"Years": years_ots})


def extract_services_renovation_rate_EP2050(
    file_path, years_ots, nonres_types, country_list
):
    """Extract non-residential energy renovation rate from EP2050 Dienstleistung Excel.

    Source: sheet '01 Sanierung', Table 01-03 (ZERO) and 01-06 (WWB).
    Uses the 'Gesamt' row — total stock weighted across all construction periods.
    Unit: fraction per year (not %).

    Returns a dict with keys 'ots' and 'fts', each a DataMatrix:
      - 'ots': annual rate 1990-2023, same value broadcast to all nonres_types
      - 'fts': dict with keys 'wwb' and 'zero', each covering years_fts (2025-2050)
    """
    df = pd.read_excel(file_path, sheet_name="01 Sanierung", header=None)

    # Locate year columns from the ZERO table header (row 31 / 65 are identical)
    year_row_idx = 31
    year_cols = {
        int(v): i
        for i, v in enumerate(df.iloc[year_row_idx])
        if isinstance(v, (int, float)) and not pd.isna(v) and 2000 <= int(v) <= 2060
    }

    # Row indices for 'Gesamt' in each scenario
    ZERO_GESAMT_ROW = 35
    WWB_GESAMT_ROW = 69

    def _build_dm(row_idx, years):
        dm = DataMatrix(
            col_labels={
                "Country": country_list,
                "Years": years,
                "Variables": ["bld_renovation-rate"],
                "Categories1": nonres_types,
            },
            units={"bld_renovation-rate": "%"},
        )
        for yr in years:
            if yr in year_cols:
                rate = df.iloc[row_idx, year_cols[yr]]
                dm.array[:, dm.idx[yr], dm.idx["bld_renovation-rate"], :] = rate
        return dm

    # OTS: use ZERO row (OTS values are identical across scenarios)
    ots_years_in_ep2050 = [yr for yr in years_ots if yr in year_cols]
    dm_ots = _build_dm(ZERO_GESAMT_ROW, years_ots)
    # Fill 1990-1999 (not in EP2050) by holding the 2000 value constant backward
    val_2000 = df.iloc[ZERO_GESAMT_ROW, year_cols[2000]]
    for yr in years_ots:
        if yr < 2000:
            dm_ots.array[:, dm_ots.idx[yr], dm_ots.idx["bld_renovation-rate"], :] = (
                val_2000
            )

    # FTS: extract both scenarios over 2025-2050
    fts_ep2050_years = [yr for yr in range(2025, 2051, 5) if yr in year_cols]
    dm_wwb = _build_dm(WWB_GESAMT_ROW, fts_ep2050_years)
    dm_zero = _build_dm(ZERO_GESAMT_ROW, fts_ep2050_years)

    return {"ots": dm_ots, "fts": {"wwb": dm_wwb, "zero": dm_zero}}


def extract_national_energy_demand(table_id, file):
    try:
        with open(file, "rb") as handle:
            dm_energy = pickle.load(handle)
    except OSError:
        structure, title = get_data_api_CH(table_id, mode="example", language="fr")

        # Remove freight transport energy demand
        exclude_list = ["49", "50", "51", "52", "53"]
        keep_sectors = [
            s
            for s in structure["Économie et ménages"]
            if not any(n in s for n in exclude_list)
        ]
        # Remove household energy demand and passenger transport
        keep_sectors = [s for s in keep_sectors if "énages" not in s]
        # Drop too detailed split
        keep_sectors = [s for s in keep_sectors if "---- " not in s]

        filter = {
            "Économie et ménages": keep_sectors,
            "Unité de mesure": ["Térajoules"],
            "Année": structure["Année"],
            "Agent énergétique": structure["Agent énergétique"],
        }

        mapping = {
            "Country": "Unité de mesure",
            "Years": "Année",
            "Variables": "Économie et ménages",
            "Categories1": "Agent énergétique",
        }

        dm_energy = get_data_api_CH(
            table_id,
            mode="extract",
            mapping_dims=mapping,
            filter=filter,
            units=["TJ"] * len(keep_sectors),
            language="fr",
        )

        # dm_heating.rename_col('--- Chauffage des ménages', 'bld_heating-demand', dim='Variables')
        dm_energy.rename_col("Térajoules", "Switzerland", dim="Country")

        # We drop the fuels fro transport
        dict_rename = {
            "heating-oil": ["1.1.2. Huile de chauffage extra-légère"],
            "coal": ["1.2. Charbon"],
            "gas": ["1.3. Gaz naturel"],
            "district-heating": ["6. Chaleur à distance"],
            "nuclear-fuel": ["4. Combustibles nucléaires"],
            "waste": ["2. Déchets (hors biomasse)"],
            "biomass": ["3.1. Déchets (biomasse)"],
            "wood": ["3.2. Bois et charbon de bois"],
            "biogas": ["3.3. Biogaz et biocarburants"],
            "renewables": [
                "3.4. Géothermie, chaleur ambiante et énergie solaire thermique"
            ],
            "electricity": ["5. Electricité"],
        }

        dm_energy = dm_energy.groupby(dict_rename, dim="Categories1", inplace=False)

        for var in dm_energy.col_labels["Variables"]:
            dm_energy.rename_col(var, "bld_energy-by-sector_" + var, dim="Variables")

        dm_energy.deepen(based_on="Variables")
        dm_energy.switch_categories_order()

        dm_energy.change_unit(
            "bld_energy-by-sector", 3600, old_unit="TJ", new_unit="TWh", operator="/"
        )

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, "wb") as handle:
            pickle.dump(dm_energy, handle, protocol=pickle.HIGHEST_PROTOCOL)
    dm_energy.rename_col_regex("-", "", dim="Categories1")
    dm_energy.drop("Categories1", " 35 Production et distribution d'énergie")
    # Remove demand of waste for waste management sector because it is computed in EnergyScope
    # Biomass is also from waste
    dm_energy[
        :, :, :, " 3639 Production et distribution d'eau; gestion des déchets", "waste"
    ] = 0
    dm_energy[
        :,
        :,
        :,
        " 3639 Production et distribution d'eau; gestion des déchets",
        "biomass",
    ] = 0
    return dm_energy


def extract_employees_per_sector_canton(table_id, file):
    try:
        with open(file, "rb") as handle:
            dm_employees = pickle.load(handle)
    except OSError:
        structure, title = get_data_api_CH(table_id, mode="example", language="fr")

        filter = {
            "Division économique": structure["Division économique"],
            "Unité d'observation": ["Equivalents plein temps"],
            "Année": structure["Année"],
            "Canton": structure["Canton"],
        }

        mapping = {
            "Country": "Canton",
            "Years": "Année",
            "Variables": "Unité d'observation",
            "Categories1": "Division économique",
        }

        dm_employees = get_data_api_CH(
            table_id,
            mode="extract",
            mapping_dims=mapping,
            filter=filter,
            units=["EPT"],
            language="fr",
        )

        # dm_heating.rename_col('--- Chauffage des ménages', 'bld_heating-demand', dim='Variables')
        dm_employees.rename_col(
            "Equivalents plein temps", "ind_employees", dim="Variables"
        )
        dm_employees.drop(col_label="Division économique - total", dim="Categories1")

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(current_file_directory, file)
        with open(f, "wb") as handle:
            pickle.dump(dm_employees, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm_employees


def load_services_energy_demand_eud(dict_services, years_ots):
    dict_services.pop("Total")
    dm = DataMatrix(
        col_labels={
            "Country": ["Switzerland"],
            "Years": years_ots,
            "Variables": ["enr_services-energy-eud"],
            "Categories1": list(dict_services.keys()),
        },
        units={"enr_services-energy-eud": "PJ"},
    )
    for key, years_values in dict_services.items():
        for year, value in years_values.items():
            dm[:, year, :, key] = value

    dm.fill_nans("Years")
    dm.groupby(
        {
            "elec": [
                "ICT and entertainment media",
                "HVAC and building tech",
                "Drives and processes",
            ]
        },
        inplace=True,
        dim="Categories1",
    )
    dm.drop(dim="Categories1", col_label=["Other", "process-heat"])

    dm.change_unit("enr_services-energy-eud", 3.6, "PJ", "TWh", operator="/")
    return dm
