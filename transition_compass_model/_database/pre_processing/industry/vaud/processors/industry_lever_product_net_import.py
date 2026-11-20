import os
import pickle
import re

import numpy as np

from transition_compass_model._database.pre_processing.api_routines_CH import (
    get_data_api_CH,
)
from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    load_pop,
)

# NOGA 2-digit division → fxa/prod model sector.
# Sectors modelled by material balance (cement, steel, Al, Cu, glass, lime, chem) are
# NOT in fxa/prod; their energy demand comes from DM_production × intensity. Everything
# else in secondary industry that drives energy demand falls into one of these buckets.
NOGA_TO_SECTOR = {
    "fbt": [10, 11, 12],
    "textiles": [13, 14, 15],
    "wwp": [16, 17, 18],
    "mae": [28],
    "tra-equip": [29, 30],
    # ois = residual: mining (5-9), chem/pharma (19-21), rubber/plastics (22-23),
    # non-metallic minerals (23 already partly above), basic/fabricated metals (24-25),
    # electronics/electrical (26-27), other manufacturing/repair (31-33)
    # Note: NOGA 23 includes cement (Holcim Eclépens). Holcim employees inflate the
    # OIS FTE ratio slightly, but this does not cause double-counting: cement energy
    # demand is computed via the material balance, not via fxa/prod OIS.
    "ois": [5, 6, 7, 8, 9, 19, 20, 21, 22, 23, 24, 25, 26, 27, 31, 32, 33],
}

# NOGA 2-digit division → model material (for material-net-import scaling).
# The FTE-based formula:  net_import_VD = clip(1 − (fte/pop) × (1 − net_import_CH), −1, 1)
# Categories absent from this map keep their CH value unchanged (safe fallback).
# Domestic-only materials (asphalt=0, nmm=0, ois-modelled=0) are excluded; their CH
# net-import is defined as 0 and must stay 0 for all cantons.
# Cement is excluded: NOGA 23 covers all non-metallic minerals (ceramics, concrete
# products, cut stone, …). Holcim Eclépens workers are diluted nationally, giving Vaud
# a NOGA 23 FTE share below pop-share and the wrong (import) direction. Cement uses a
# plant-capacity formula instead (see _CH_CEMENT_CONSUMPTION_KT / _ECLÉPENS_CAPACITY_KT).
# Lime in Vaud: Carmeuse Villeneuve existence unverified (TODO); NOGA 23 FTE signal
# applies (same dilution caveat as cement, but lime has no known VD plant to override).
MATERIAL_TO_NOGA = {
    "glass": [23],
    "lime": [23],
    "steel": [24, 25],
    "aluminium": [24, 25],
    "copper": [24, 25],
    "met": [
        24,
        25,
    ],  # critical metals: no VD mining → fte_ratio ≈ pop_ratio → CH retained
    "chem": [20],
    "paper": [17],
    "timber": [16],
}

# CEMENT — Holcim Eclépens plant estimation
# Vaud hosts the Holcim Eclépens cement plant, the only cement production facility
# in the canton and one of the largest in Switzerland.
#
# CH baseline (Cemsuisse Kennzahlen 2020, p.4+10):
#   CH cement production: deliveries 4,215 kt + exports 106 kt = 4,321 kt
#   CH cement consumption: deliveries 4,215 kt + imports 659 kt = 4,874 kt
#   → CH net-import = (659 − 106) / 4,874 ≈ +0.11 (net importer)
#   Source: https://www.cemsuisse.ch/app/uploads/2020/05/Cemsuisse_Kennzahlen_2020_RZ.pdf
#
# Vaud estimation:
#   Eclépens plant cement production ≈ 700 kt/year
#   Source: Holcim CH, "Cimenterie d'Eclépens" — https://www.holcim.ch/fr/cimenterie-eclepens
#   ("ses 115 collaboratrices et collaborateurs contribuent à la production d'environ
#   700 000 tonnes annuelles de ciment")
#
#   VD cement consumption ≈ VD/CH population ratio × CH consumption
#   (population is a standard proxy for construction activity)
#   = 0.089 × 4,874 ≈ 434 kt/year
#
#   VD net-import = (VD_consumption − Eclépens_production) / VD_consumption
#   = (434 − 700) / 434 ≈ −0.61
#   → Vaud produces ~700 kt and consumes ~434 kt: net exporter of ~266 kt.
#
# Note: the model's material-decomposition demand (buildings only) is lower than
# 434 kt because it does not cover all cement uses (industrial, commercial). The
# physical consumption estimate from Cemsuisse is the correct denominator for the
# net-import formula; the model gap is a known structural limitation.
#
# Residual gap: modelled CO2 may still differ from reported plant emissions due to
# the EU-average clinker ratio in the process EF constant (0.512 tCO2/kt), which
# does not capture Holcim's high use of supplementary cementitious materials.
_CH_CEMENT_CONSUMPTION_KT = 4_874.0  # kt/year — Cemsuisse Kennzahlen 2020
_ECLÉPENS_PRODUCTION_KT = (
    700.0  # kt/year cement production — Holcim Eclépens (holcim.ch)
)

# NOGA 2-digit division → model product (for product-net-import scaling).
# Infrastructure products (rail, road, floor-area-*) are excluded — CH value is 0.
PRODUCT_TO_NOGA = {
    # Motor vehicles (C29)
    "2W_BEV": [29],
    "2W_ICE-gasoline": [29],
    "HDV_BEV": [29],
    "HDV_FCEV": [29],
    "HDV_ICE-diesel": [29],
    "HDV_ICE-gas": [29],
    "HDV_ICE-gasoline": [29],
    "HDV_PHEV-diesel": [29],
    "HDV_PHEV-gasoline": [29],
    "LDV_BEV": [29],
    "LDV_FCEV": [29],
    "LDV_ICE-diesel": [29],
    "LDV_ICE-gas": [29],
    "LDV_ICE-gasoline": [29],
    "LDV_PHEV-diesel": [29],
    "LDV_PHEV-gasoline": [29],
    "bus_ICE-diesel": [29],
    "bus_ICE-gas": [29],
    # Other transport equipment (C30): Stadler (Bussnang TG) + Pilatus (Stans NW) → VD FTE ≈ 0
    "planes_ICE": [30],
    "ships_ICE-diesel": [30],
    "trains_CEV": [30],
    "trains_ICE-diesel": [30],
    # Packaging: plastic (C22), glass (C23), aluminium (C24)
    "glass-pack": [23],
    "plastic-pack": [22],
    "aluminium-pack": [24],
    # Paper products (C17-C18)
    "paper-pack": [17],
    "paper-print": [18],
    "paper-san": [17],
    # Consumer electronics (C26): no VD manufacturer → FTE ≈ 0 → net-import ≈ 1
    "computer": [26],
    "phone": [26],
    "tv": [26],
    # Household appliances (C27): V-ZUG in Zug → VD FTE below pop-share
    "dishwasher": [27],
    "dryer": [27],
    "freezer": [27],
    "fridge": [27],
    "wmachine": [27],
}

# Population-share fallback if STATENT API is unavailable
_POP_SHARE_FALLBACK = 0.089  # Vaud ~8.9% of CH (2023 census)


def _extract_employees_per_sector_canton(table_id, cache_file):
    try:
        with open(cache_file, "rb") as handle:
            dm_employees = pickle.load(handle)
    except OSError:
        structure, _ = get_data_api_CH(table_id, mode="example", language="fr")
        filter_dict = {
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
            filter=filter_dict,
            units=["EPT"],
            language="fr",
        )
        dm_employees.rename_col(
            "Equivalents plein temps", "ind_employees", dim="Variables"
        )
        dm_employees.drop(col_label="Division économique - total", dim="Categories1")

        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as handle:
            pickle.dump(dm_employees, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm_employees


def _compute_fte_ratio(dm_employees, noga_list):
    """Return Vaud/CH FTE ratio for the given list of NOGA 2-digit codes.

    Falls back to _POP_SHARE_FALLBACK if data is unavailable or codes not found.
    """

    def _div_num(label):
        nums = re.findall(r"\d+", label)
        return int(nums[0]) if nums else None

    divs = dm_employees.col_labels["Categories1"]
    div_nums = {d: _div_num(d) for d in divs}

    ch_labels = [c for c in dm_employees.col_labels["Country"] if "Suisse" in c]
    vd_labels = [c for c in dm_employees.col_labels["Country"] if "Vaud" in c]

    if not ch_labels or not vd_labels:
        return _POP_SHARE_FALLBACK

    dm_ch = dm_employees.filter({"Country": ch_labels})
    dm_vd = dm_employees.filter({"Country": vd_labels})

    matching = [d for d, n in div_nums.items() if n in noga_list]
    if not matching:
        return _POP_SHARE_FALLBACK

    ch_idx = [dm_ch.idx[d] for d in matching]
    vd_idx = [dm_vd.idx[d] for d in matching]

    vd_total = float(np.nansum(dm_vd.array[0, :, 0, vd_idx]))
    ch_total = float(np.nansum(dm_ch.array[0, :, 0, ch_idx]))
    return vd_total / ch_total if ch_total > 0 else _POP_SHARE_FALLBACK


def _compute_vaud_ch_fte_ratios(dm_employees):
    """Return dict {model_sector: float} — time-averaged Vaud/CH FTE ratio."""
    return {
        sector: _compute_fte_ratio(dm_employees, noga_list)
        for sector, noga_list in NOGA_TO_SECTOR.items()
    }


def _apply_fte_net_import(dm, dim_name, noga_mapping, dm_employees, pop_ratio):
    """Apply FTE-based net-import formula in-place.

    For each category in noga_mapping, replaces the CH value with:
        net_import_VD = clip(1 - (fte_ratio / pop_ratio) × (1 - net_import_CH), -1, 1)
    NaN values are preserved. Categories not in noga_mapping are left unchanged.
    """
    country = dm.col_labels["Country"][0]
    for cat, noga_list in noga_mapping.items():
        if cat not in dm.col_labels["Categories1"]:
            continue
        fte_ratio = _compute_fte_ratio(dm_employees, noga_list)
        ch_vals = np.array(dm[country, :, dim_name, cat])
        new_vals = 1.0 - (fte_ratio / pop_ratio) * (1.0 - ch_vals)
        new_vals = np.where(np.isnan(ch_vals), ch_vals, np.clip(new_vals, -1.0, 1.0))
        dm[country, :, dim_name, cat] = new_vals


def run(years_ots, years_fts):
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # Load CH industry pickle
    pickle_file = os.path.join(
        current_file_directory, "../../../../data/datamatrix/industry.pickle"
    )
    with open(pickle_file, "rb") as handle:
        DM_ch = pickle.load(handle)

    # ---- population ratio (used for wwp demand and civil materials) ----
    dm_pop_ch = load_pop(["Switzerland"], years_list=years_ots)
    dm_pop_vd = load_pop(["Vaud"], years_list=years_ots)
    pop_ratio = float(np.nanmean(dm_pop_vd.array[0, :, 0] / dm_pop_ch.array[0, :, 0]))

    # ---- load STATENT FTE data ----
    table_id = "px-x-0602010000_101"
    cache_file = os.path.join(
        current_file_directory, "../data/employees_per_sector_canton.pickle"
    )
    try:
        dm_employees = _extract_employees_per_sector_canton(table_id, cache_file)
    except Exception:
        # If API is unavailable: net-import keeps CH values; fxa/prod uses pop-share
        dm_employees = None

    # ---- product-net-import: start from CH, apply FTE formula ----
    dm_netimp_goods = (
        DM_ch["ots"]["product-net-import"].filter({"Country": ["Switzerland"]}).copy()
    )
    dm_netimp_goods.rename_col("Switzerland", "Vaud", dim="Country")

    if dm_employees is not None:
        _apply_fte_net_import(
            dm_netimp_goods,
            "product-net-import",
            PRODUCT_TO_NOGA,
            dm_employees,
            pop_ratio,
        )

    # ---- material-net-import: start from CH, apply FTE formula ----
    dm_netimp_materials = (
        DM_ch["ots"]["material-net-import"].filter({"Country": ["Switzerland"]}).copy()
    )
    dm_netimp_materials.rename_col("Switzerland", "Vaud", dim="Country")

    if dm_employees is not None:
        _apply_fte_net_import(
            dm_netimp_materials,
            "material-net-import",
            MATERIAL_TO_NOGA,
            dm_employees,
            pop_ratio,
        )

    # ---- cement: plant-capacity override (replaces FTE formula) ----
    vd_cement_demand_kt = pop_ratio * _CH_CEMENT_CONSUMPTION_KT
    vd_cement_net_import = (
        vd_cement_demand_kt - _ECLÉPENS_PRODUCTION_KT
    ) / vd_cement_demand_kt
    dm_netimp_materials["Vaud", :, "material-net-import", "cement"] = (
        vd_cement_net_import
    )

    # ---- fxa/prod: scale CH by Vaud/CH FTE ratio per model sector ----
    if dm_employees is not None:
        fte_ratios = _compute_vaud_ch_fte_ratios(dm_employees)
    else:
        fte_ratios = {s: _POP_SHARE_FALLBACK for s in NOGA_TO_SECTOR}

    dm_prod_ch = DM_ch["fxa"]["prod"].filter({"Country": ["Switzerland"]}).copy()
    dm_prod_vd = dm_prod_ch.copy()
    dm_prod_vd.rename_col("Switzerland", "Vaud", dim="Country")

    idx = dm_prod_vd.idx
    for sector, ratio in fte_ratios.items():
        if sector in idx:
            dm_prod_vd.array[
                idx["Vaud"], :, idx["material-production"], idx[sector]
            ] *= ratio

    # civil infrastructure materials are not in NOGA_TO_SECTOR (no FTE signal);
    # scale by population ratio as proxy for construction activity
    for civil_mat in ["cement-civil", "nmm-civil", "met-civil"]:
        if civil_mat in idx:
            dm_prod_vd.array[
                idx["Vaud"], :, idx["material-production"], idx[civil_mat]
            ] *= pop_ratio

    # ---- fxa/demand (wood demand): scale by Vaud/CH population ratio ----
    dm_wwp_demand_ch = (
        DM_ch["fxa"]["demand"].filter({"Country": ["Switzerland"]}).copy()
    )
    dm_wwp_demand_vd = dm_wwp_demand_ch.copy()
    dm_wwp_demand_vd.rename_col("Switzerland", "Vaud", dim="Country")
    dm_wwp_demand_vd.array *= pop_ratio

    return dm_netimp_goods, dm_netimp_materials, dm_wwp_demand_vd, dm_prod_vd


if __name__ == "__main__":
    years_ots = create_years_list(1990, 2023, 1)
    years_fts = create_years_list(2025, 2050, 5)
    run(years_ots, years_fts)
