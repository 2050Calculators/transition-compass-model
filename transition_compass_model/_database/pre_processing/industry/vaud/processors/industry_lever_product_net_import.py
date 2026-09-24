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

# Population-share fallback if STATENT API is unavailable
_POP_SHARE_FALLBACK = 0.089  # Vaud ~8.9% of CH (2023 census)

# ---------------------------------------------------------------------------
# material-net-import: Vaud-specific adjustments
# ---------------------------------------------------------------------------

# CEMENT — Holcim Eclépens plant estimation
# Vaud hosts the Holcim Eclépens cement plant, the only cement production facility
# in the canton and one of the largest in Switzerland.
#
# CH baseline (from CH OTS preprocessing):
#   CH cement production ≈ 4,500 kt/year
#   CH cement consumption ≈ 3,500 kt/year
#   → CH net-import = (3,500 − 4,500) / 3,500 ≈ −0.25
#   Source: Cemsuisse annual statistics — https://www.cemsuisse.ch
#
# Vaud estimation:
#   Eclépens plant actual cement output ≈ 850 kt/year (historical average)
#   The nameplate clinker capacity is ~1,500 kt/year, but actual cement output is
#   significantly lower: ~849 kt in 2013–2014 (peak), declining to ~700 kt recently
#   as construction activity fell. Using nameplate clinker capacity would overestimate
#   production by ~1.4× and cement CO2 by ~1.9× vs reported plant emissions.
#   Source: 24heures, "Cimenterie d'Eclépens: moins de béton, mais 10% de CO₂ en moins"
#   https://www.24heures.ch/cimenterie-declepens-moins-de-beton-mais-10-de-co2-en-moins-516839957775
#
#   VD cement demand ≈ VD/CH population ratio × CH consumption
#   (population used as proxy for construction activity; Vaud is ~8.9% of CH)
#
#   VD net-import = (VD_demand − Eclépens_output) / VD_demand
#   ≈ (0.089 × 3,500 − 850) / (0.089 × 3,500)
#   ≈ (311 − 850) / 311 ≈ −1.73
#
# Residual gap: even with corrected production, modelled CO2 remains ~25% above
# reported plant emissions (e.g. ~420 kt vs 339 kt in 2023). The remaining gap
# is attributable to the EU-average clinker ratio embedded in the process EF
# constant (0.512 tCO2/kt), which does not capture Holcim's high use of
# supplementary cementitious materials and alternative fuels.
_CH_CEMENT_CONSUMPTION_KT = 3_500.0  # kt/year — Cemsuisse
_ECLÉPENS_CAPACITY_KT = 850.0  # kt/year cement output — Holcim Eclépens actual

# METALS (steel, aluminium, copper) — T06.03_1b correction
# Source: StatVD / OFDF, "Le commerce extérieur du canton de Vaud et de la Suisse,
# par groupe de marchandises" (T06.03_1b-marchandises.xlsx, available from StatVD:
# https://www.vd.ch/themes/etat-droit-finances/statistiques-vaud).
# Category "Métaux" (rows 9 in VD+CH exports/imports tables, values in MCHF, 2016–2025).
#
# Vaud/CH net-import ratio relative to population share (avg 2016–2025):
#   VD net imports (Métaux): +294 MCHF/yr  → VD/CH share: 7.9%
#   CH net imports (Métaux): +3,723 MCHF/yr
#   Pop share: 8.9%
#   Multiplier = 7.9% / 8.9% = 0.888
#
# Interpretation: Vaud imports proportionally slightly less metals per capita than
# the CH average, consistent with its lower concentration of metal-intensive industry.
# Note: T06.03_1b covers international (customs) trade only, not inter-cantonal flows.
#
# "chem" is NOT adjusted via T06.03_1b: the customs category "Produits chimiques"
# is dominated by pharma exports (Roche/Novartis in Basel), making CH a net exporter
# in T06.03_1b (−3,645 MCHF/yr) despite being a net importer of chemical feedstocks
# in the model (+0.249). Applying the T06.03_1b ratio would flip the sign — invalid.
_METALS_T06031B_MULTIPLIER = 0.888


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


def _compute_vaud_ch_fte_ratios(dm_employees):
    """Return dict {model_sector: float} — time-averaged Vaud/CH FTE ratio."""

    # Extract 2-digit NOGA division codes from category labels
    def _div_num(label):
        nums = re.findall(r"\d+", label)
        return int(nums[0]) if nums else None

    divs = dm_employees.col_labels["Categories1"]
    div_nums = {d: _div_num(d) for d in divs}

    # Use the national "Suisse" row as CH total (official STATENT figure)
    ch_labels = [c for c in dm_employees.col_labels["Country"] if "Suisse" in c]
    vd_labels = [c for c in dm_employees.col_labels["Country"] if "Vaud" in c]

    if not ch_labels or not vd_labels:
        # Fallback: population share for all sectors
        return {s: _POP_SHARE_FALLBACK for s in NOGA_TO_SECTOR}

    dm_ch = dm_employees.filter({"Country": ch_labels})
    dm_vd = dm_employees.filter({"Country": vd_labels})

    ratios = {}
    for sector, noga_list in NOGA_TO_SECTOR.items():
        matching = [d for d, n in div_nums.items() if n in noga_list]
        if not matching:
            ratios[sector] = _POP_SHARE_FALLBACK
            continue

        ch_idx = [dm_ch.idx[d] for d in matching]
        vd_idx = [dm_vd.idx[d] for d in matching]

        # Sum FTE over all matching divisions and all available years
        vd_total = float(np.nansum(dm_vd.array[0, :, 0, vd_idx]))
        ch_total = float(np.nansum(dm_ch.array[0, :, 0, ch_idx]))
        ratios[sector] = vd_total / ch_total if ch_total > 0 else _POP_SHARE_FALLBACK

    return ratios


def run(years_ots, years_fts):
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # Load CH industry pickle
    pickle_file = os.path.join(
        current_file_directory, "../../../../data/datamatrix/industry.pickle"
    )
    with open(pickle_file, "rb") as handle:
        DM_ch = pickle.load(handle)

    # ---- population ratio (used for both wwp demand and cement estimation) ----
    dm_pop_ch = load_pop(["Switzerland"], years_list=years_ots)
    dm_pop_vd = load_pop(["Vaud"], years_list=years_ots)
    pop_ratio = float(np.nanmean(dm_pop_vd.array[0, :, 0] / dm_pop_ch.array[0, :, 0]))

    # ---- product-net-import: copy from CH, then override products not made in Vaud ----
    dm_netimp_goods = (
        DM_ch["ots"]["product-net-import"].filter({"Country": ["Switzerland"]}).copy()
    )
    dm_netimp_goods.rename_col("Switzerland", "Vaud", dim="Country")

    idx_goods = dm_netimp_goods.idx

    # Trains: Stadler Rail is in Bussnang (TG) → Vaud has no train manufacturer → all imported
    # CH value 0.316 reflects Stadler's domestic supply share nationally; for VD it is 0
    for prod in ["trains_CEV", "trains_ICE-diesel"]:
        dm_netimp_goods.array[
            idx_goods["Vaud"], :, idx_goods["product-net-import"], idx_goods[prod]
        ] = 1.0

    # Planes: Pilatus Aircraft is in Stans (NW) → Vaud has no aircraft manufacturer → all imported
    dm_netimp_goods.array[
        idx_goods["Vaud"], :, idx_goods["product-net-import"], idx_goods["planes_ICE"]
    ] = 1.0

    # White goods: V-ZUG is in Zug (ZG) → Vaud has no white-goods manufacturer → all imported
    # CH value −0.021 reflects V-ZUG's small net-export position nationally; for VD it is 0
    for prod in ["dishwasher", "dryer", "freezer", "fridge", "wmachine"]:
        dm_netimp_goods.array[
            idx_goods["Vaud"], :, idx_goods["product-net-import"], idx_goods[prod]
        ] = 1.0

    # ---- material-net-import: start from CH, then apply Vaud-specific corrections ----
    dm_netimp_materials = (
        DM_ch["ots"]["material-net-import"].filter({"Country": ["Switzerland"]}).copy()
    )
    dm_netimp_materials.rename_col("Switzerland", "Vaud", dim="Country")

    idx_mat = dm_netimp_materials.idx

    # Cement: replace CH ratio (−0.25) with Vaud-specific estimate (see constants above)
    vd_cement_demand_kt = pop_ratio * _CH_CEMENT_CONSUMPTION_KT
    vd_cement_net_import = (
        vd_cement_demand_kt - _ECLÉPENS_CAPACITY_KT
    ) / vd_cement_demand_kt
    dm_netimp_materials.array[
        idx_mat["Vaud"], :, idx_mat["material-net-import"], idx_mat["cement"]
    ] = vd_cement_net_import

    # Metals: scale by T06.03_1b "Métaux" multiplier (see constants above)
    for mat in ["steel", "aluminium", "copper"]:
        dm_netimp_materials.array[
            idx_mat["Vaud"], :, idx_mat["material-net-import"], idx_mat[mat]
        ] *= _METALS_T06031B_MULTIPLIER

    # Remaining materials (chem, glass, lime, paper, timber, other): keep CH values
    # — chem: T06.03_1b invalid (pharma-export dominance flips sign); keep CH +0.249
    # — glass/timber/paper/other: insufficient VD-specific data; impact negligible for timber
    # lime — CH is −0.80 (strong net exporter). No active lime kiln exists in Vaud:
    #   industrial lime production consolidated or ceased in the canton during the 20th
    #   century (verified via Swiss E-PRTR / OFEV registry and web search). CH default
    #   net-import ratio retained.

    # ---- fxa/prod: scale CH by Vaud/CH FTE ratio per model sector ----
    table_id = "px-x-0602010000_101"
    # Re-use cached data from industry_to_energy_interface if it already exists
    cache_file = os.path.join(
        current_file_directory, "../data/employees_per_sector_canton.pickle"
    )
    try:
        dm_employees = _extract_employees_per_sector_canton(table_id, cache_file)
        fte_ratios = _compute_vaud_ch_fte_ratios(dm_employees)
    except Exception:
        # If API is unavailable fall back to population share
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
