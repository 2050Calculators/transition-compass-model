"""Build simple_industry.pickle and industry_to_energy.pickle for Switzerland.

Energy demand by canton x industry branch x fuel, and the related emissions.
Method (see PLAN_simple_industry_waste.md next to this file):

1. National energy by branch x fuel: BFE industry statistics give the branch split of
   each fuel (2013-2023). Each fuel is scaled to GEST T17b so the Swiss total matches
   the official energy balance. NACE 36-39 (water-waste) own use comes from the FSO
   energy accounts and is part of the GEST industry total.
2. Emissions: combustion = energy x emission factor (fossil and biogenic separately);
   process emissions by branch from EP2050+ (Tab. 02-04).
3. Cantons: branches with SwissPRTR facilities are allocated point-sources-first
   (PRTR CO2 per canton, remainder by FTE); everything else by FTE. Hard check: canton x
   branch CO2 >= PRTR.
4. Forecast: levers 1-4, level 1 = EP2050+ WWB, level 4 = ZERO Basis, scaled to 2023.
5. industry_to_energy.pickle: same layout as before (end use x fuel), end-use split
   from EP2050+ Tab. 02-01 / 02-02.

STAT-TAB API calls (pxweb.bfs.admin.ch) are flagged with "# STAT-TAB API". They only run
when the cached pickle is missing.
"""

import os
import pickle

import numpy as np
import pandas as pd

from transition_compass_model._database.pre_processing.api_routines_CH import (
    get_data_api_CH,
)
from transition_compass_model._database.pre_processing.constants import (
    CANTONS_RENAME_TARGET,
)
from transition_compass_model._database.pre_processing.simple_industry.Switzerland import (
    get_data_functions as gd,
)
from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    rename_cantons,
)
from transition_compass_model.model.common.constant_data_matrix_class import (
    ConstantDataMatrix,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

DATAMATRIX_DIR = os.path.join(gd.ROOT_DIR, "_database/data/datamatrix")
INTERFACE_DIR = os.path.join(gd.ROOT_DIR, "_database/data/interface")

YEARS_OTS = create_years_list(1990, 2023, 1)
YEARS_FTS = create_years_list(2025, 2050, 5)
BASE_YEAR = 2023
CANTONS = sorted(CANTONS_RENAME_TARGET)
BRANCHES = gd.BRANCHES
INDUSTRY_BRANCHES = BRANCHES[:12]  # covered by the BFE statistics
GASES = ["CH4", "CO2", "N2O"]

FUELS = [
    "electricity",
    "gas",
    "heating-oil",  # extra-light heating oil
    "other-oil",  # all other petroleum products (heavy fuel oil, LPG, petcoke, ...)
    "coal",
    "wood",
    "district-heating",
    "waste",  # industrial waste used as fuel (mostly cement and chemicals)
    "renewables",  # solar thermal, biogas, ambient heat
]
FOSSIL_FUELS = ["gas", "heating-oil", "other-oil", "coal", "waste"]
# Fuels burned on site: their energy follows the emission-based canton shares
COMBUSTION_FUELS = FOSSIL_FUELS + ["wood"]

# Emission factors in Mt per TWh of fuel.
# CO2: BAFU factors used in buildings (coal 350, heating oil 265, gas 201 kt/TWh,
# https://www.bafu.admin.ch/dam/fr/sd-web/HnIzzj6OfDUU/EF_CO2_Berichterstattung_Kantone.pdf).
# Other oil: heavy fuel oil (77.3 t/TJ). Wood and the waste fuel: IPCC 2006 (via
# industry.pickle). ASSUMPTION: 30% of the industrial waste fuel is biogenic.
# CH4 / N2O: IPCC 2006 Vol. 2 Tab. 2.3 (industry.pickle 'emission-factor').
WASTE_BIOGENIC_SHARE = 0.3
EF_FOSSIL = {
    "CO2": {
        "gas": 0.201,
        "heating-oil": 0.265,
        "other-oil": 0.278,
        "coal": 0.350,
        "waste": 0.3696 * (1 - WASTE_BIOGENIC_SHARE),
    },
    "CH4": {
        "gas": 3.6e-6,
        "heating-oil": 1.08e-5,
        "other-oil": 1.08e-5,
        "coal": 3.6e-5,
        "waste": 1.08e-4,
        "wood": 2.367e-4,
    },
    "N2O": {
        "gas": 3.6e-7,
        "heating-oil": 2.16e-6,
        "other-oil": 2.16e-6,
        "coal": 5.4e-6,
        "waste": 1.44e-5,
        "wood": 1.26e-5,
    },
}
EF_BIOGENIC_CO2 = {"wood": 0.3774, "waste": 0.3696 * WASTE_BIOGENIC_SHARE}

# STATENT 2-digit NOGA divisions -> branch (cement/glass share C23, steel/NE share C24)
FTE_NOGA = {
    "food": ["10", "11", "12"],
    "textiles": ["13", "14", "15"],
    "paper": ["17", "18"],
    "chemicals": ["20", "21"],
    "cement": ["23"],
    "glass-ceramics": ["23"],
    "iron-steel": ["24"],
    "non-ferrous-metals": ["24"],
    "metal-products": ["25", "26", "27"],
    "machinery": ["28"],
    "other-industries": [
        "05",
        "06",
        "07",
        "08",
        "09",
        "16",
        "22",
        "29",
        "30",
        "31",
        "32",
    ],
    "construction": ["41", "42", "43"],
    "water-waste": ["36", "37", "38", "39"],
}

EP_SHEETS = {1: "02 Industrie WWB", 4: "03 Industrie ZERO Basis"}

# Water-waste fuels that GEST counts outside industry: sewage-plant biogas is in the
# GEST biogas balance (T34a), not in T17b (T17b 'other renewables' is 0.07 TWh in 1990,
# below the 0.33 TWh of NACE 36-39 biogas). These are added on top of GEST.
WW_OUTSIDE_GEST = ["renewables"]


# ======================================================================================
# STAT-TAB extractions (cached)
# ======================================================================================
def extract_energy_accounts(table_id, file):
    """FSO energy accounts of the economy, TWh. Cached in `file`."""
    try:
        with open(file, "rb") as handle:
            return pickle.load(handle)
    except OSError:
        pass
    print(f"WARNING: calling the stat-tab API for {table_id} (cache {file} missing)")
    # STAT-TAB API
    structure, title = get_data_api_CH(table_id, mode="example", language="fr")
    keep_sectors = [
        s
        for s in structure["Économie et ménages"]
        if "énages" not in s and "---- " not in s
    ]
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
    # STAT-TAB API
    dm = get_data_api_CH(
        table_id,
        mode="extract",
        mapping_dims=mapping,
        filter=filter,
        units=["TJ"] * len(keep_sectors),
        language="fr",
    )
    dm.rename_col("Térajoules", "Switzerland", dim="Country")
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
    dm = dm.groupby(dict_rename, dim="Categories1", inplace=False)
    for var in dm.col_labels["Variables"]:
        dm.rename_col(var, "bld_energy-by-sector_" + var, dim="Variables")
    dm.deepen(based_on="Variables")
    dm.switch_categories_order()
    dm.change_unit(
        "bld_energy-by-sector", 3600, old_unit="TJ", new_unit="TWh", operator="/"
    )
    with open(file, "wb") as handle:
        pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return dm


def extract_employees_per_sector_canton(table_id, file):
    """STATENT full-time equivalents by canton and NOGA division. Cached in `file`."""
    try:
        with open(file, "rb") as handle:
            return pickle.load(handle)
    except OSError:
        pass
    print(f"WARNING: calling the stat-tab API for {table_id} (cache {file} missing)")
    # STAT-TAB API
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
    # STAT-TAB API
    dm = get_data_api_CH(
        table_id,
        mode="extract",
        mapping_dims=mapping,
        filter=filter,
        units=["EPT"],
        language="fr",
    )
    dm.rename_col("Equivalents plein temps", "ind_employees", dim="Variables")
    dm.drop(col_label="Division économique - total", dim="Categories1")
    with open(file, "wb") as handle:
        pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return dm


# ======================================================================================
# Step 1: national energy by branch x fuel
# ======================================================================================
def water_waste_own_use(dm_accounts):
    """NACE 36-39 own energy use by fuel (TWh), years YEARS_OTS.

    Waste and biomass fuels are dropped: they are burned in waste incineration plants,
    which are in simple_waste. Years outside 2000-2022 are held at the edge value.
    """
    cat = [c for c in dm_accounts.col_labels["Categories1"] if "36" in c and "39" in c]
    if len(cat) != 1:
        raise ValueError(f"Cannot find NACE 36-39 in energy accounts: {cat}")
    years = dm_accounts.col_labels["Years"]
    mapping = {
        "electricity": ["electricity"],
        "gas": ["gas"],
        "heating-oil": ["heating-oil"],
        "coal": ["coal"],
        "wood": ["wood"],
        "district-heating": ["district-heating"],
        "renewables": ["biogas", "renewables"],
    }
    df = pd.DataFrame(0.0, index=years, columns=FUELS)
    for fuel, srcs in mapping.items():
        for s in srcs:
            df[fuel] += np.nan_to_num(
                dm_accounts["Switzerland", :, "bld_energy-by-sector", cat[0], s]
            )
    return df.reindex(YEARS_OTS).ffill().bfill()


def national_energy(gest, bfe, ep_comb_wwb, ww):
    """Energy by year x branch x fuel (TWh), summing to GEST T17b per fuel.

    For each fuel, the industry branches share (GEST - water-waste) using the BFE
    branch split. Before 2013, the split is held at 2013, and for fossil fuels it is
    shifted with the EP2050+ combustion-emission trend of each branch.
    """
    nb, nf = len(BRANCHES), len(FUELS)
    E = np.zeros((len(YEARS_OTS), nb, nf))
    bfe_years = sorted(bfe["electricity"].columns)
    first = bfe_years[0]
    # Fuels without their own BFE sheet borrow the split of a similar fuel
    proxy = {"renewables": None, "other-oil": "heating-oil"}
    for iy, y in enumerate(YEARS_OTS):
        yb = min(max(y, first), bfe_years[-1])
        tot_branch = sum(bfe[f][yb] for f in bfe)  # total energy per branch
        for jf, f in enumerate(FUELS):
            if f in bfe and bfe[f][yb].sum() > 0:
                share = bfe[f][yb].copy()
            elif proxy.get(f) and bfe[proxy[f]][yb].sum() > 0:
                share = bfe[proxy[f]][yb].copy()
            else:
                share = tot_branch.copy()
            if y < first and f in FOSSIL_FUELS:
                trend = (
                    ep_comb_wwb[y].reindex(INDUSTRY_BRANCHES)
                    / ep_comb_wwb[first].reindex(INDUSTRY_BRANCHES)
                ).fillna(1.0)
                share = share * trend
            share = share.reindex(INDUSTRY_BRANCHES).fillna(0.0)
            share = share / share.sum()
            target = gest.loc[y, f] - (0.0 if f in WW_OUTSIDE_GEST else ww.loc[y, f])
            if target < 0:
                raise ValueError(f"Water-waste {f} {y} exceeds GEST industry total")
            E[iy, :12, jf] = share.values * target
        E[iy, 12, :] = ww.loc[y, FUELS].values
    return E


# ======================================================================================
# Step 2: emissions
# ======================================================================================
def ef_arrays():
    """Emission-factor arrays [fuel, gas] (Mt/TWh), fossil and biogenic."""
    ef_f = np.zeros((len(FUELS), len(GASES)))
    ef_b = np.zeros((len(FUELS), len(GASES)))
    for jg, gas in enumerate(GASES):
        for jf, f in enumerate(FUELS):
            ef_f[jf, jg] = EF_FOSSIL[gas].get(f, 0.0)
    for jf, f in enumerate(FUELS):
        ef_b[jf, GASES.index("CO2")] = EF_BIOGENIC_CO2.get(f, 0.0)
    return ef_f, ef_b


def emissions_from_energy(E, process):
    """E [..., branch, fuel], process [..., branch] -> dict of arrays [..., branch, gas]."""
    ef_f, ef_b = ef_arrays()
    comb = E @ ef_f
    bio = E @ ef_b
    proc = np.zeros(comb.shape)
    proc[..., GASES.index("CO2")] = process  # CO2-eq, see EP2050+ Tab. 02-04
    return {"combustion": comb, "biogenic": bio, "process": proc}


def total_co2(E, process):
    em = emissions_from_energy(E, process)
    ico2 = GASES.index("CO2")
    return em["combustion"][..., ico2] + em["biogenic"][..., ico2] + process


# ======================================================================================
# Step 3: cantonal allocation
# ======================================================================================
def fte_shares(dm_employees):
    """FTE share of each canton in each branch, array [canton, year, branch].

    STATENT covers 2011-2022; shares are held constant outside that range.
    """
    dm = dm_employees.copy()
    dm.drop("Country", "Suisse")
    rename_cantons(dm)
    dm.sort("Country")
    if dm.col_labels["Country"] != sorted(CANTONS):
        raise ValueError("FTE cantons do not match the canton list")
    codes = [c[:2] for c in dm.col_labels["Categories1"]]
    years = dm.col_labels["Years"]
    arr = np.nan_to_num(dm.array[:, :, 0, :])  # canton, year, noga
    out = np.zeros((len(CANTONS), len(YEARS_OTS), len(BRANCHES)))
    for ib, b in enumerate(BRANCHES):
        idx = [i for i, c in enumerate(codes) if c in FTE_NOGA[b]]
        fte = arr[:, :, idx].sum(axis=2)
        share = fte / fte.sum(axis=0, keepdims=True)
        for iy, y in enumerate(YEARS_OTS):
            yy = min(max(y, years[0]), years[-1])
            out[:, iy, ib] = share[:, years.index(yy)]
    return out


def allocate_to_cantons(E_nat, proc_nat, share_fte, prtr):
    """Split national energy and process emissions to cantons.

    For each branch and year: canton CO2 = PRTR CO2 + remainder x FTE share, and
    combustion fuels and process emissions follow these canton shares. Electricity,
    district heat and renewables follow FTE. Before the first PRTR year the shares of
    the first PRTR year are used. Returns (E_c, proc_c, weights, violations).
    """
    nc, ny, nb = share_fte.shape
    T = total_co2(E_nat, proc_nat)  # [year, branch]
    w = share_fte.copy()
    prtr_years = [iy for iy in range(ny) if not np.isnan(prtr[0, iy, 0])]
    violations = []
    for iy in prtr_years:
        for ib in range(nb):
            P = prtr[:, iy, ib]
            if P.sum() == 0:
                continue
            rem = T[iy, ib] - P.sum()
            if rem < 0:
                violations.append((YEARS_OTS[iy], BRANCHES[ib], T[iy, ib], P.sum()))
            co2_c = P + max(rem, 0.0) * share_fte[:, iy, ib]
            w[:, iy, ib] = co2_c / co2_c.sum()
    if prtr_years:
        first = prtr_years[0]
        for iy in range(first):
            w[:, iy, :] = w[:, first, :]
    E_c = np.zeros((nc,) + E_nat.shape)
    is_comb = np.array([f in COMBUSTION_FUELS for f in FUELS])
    E_c[..., is_comb] = w[..., None] * E_nat[None, ..., is_comb]
    E_c[..., ~is_comb] = share_fte[..., None] * E_nat[None, ..., ~is_comb]
    proc_c = w * proc_nat[None, ...]
    return E_c, proc_c, w, violations


# ======================================================================================
# Step 4: forecast
# ======================================================================================
def ep2050_ratios(sheet):
    """EP2050+ trajectories of one scenario relative to the base year."""
    tot = gd.read_ep2050_table(sheet, 1).loc["Summe"]
    elec = gd.read_ep2050_table(sheet, 2).loc["Summe"]
    comb = gd.read_ep2050_branch_table(sheet, 3)
    proc = gd.read_ep2050_branch_table(sheet, 4)

    def rel(df):
        base = df[BASE_YEAR]
        r = df.div(base.replace(0, np.nan), axis=0).fillna(1.0)
        return r.clip(lower=0.0)  # ZERO nets out CCS: negative -> 0 fuel use

    return {
        "total": tot / tot[BASE_YEAR],
        "electricity": elec / elec[BASE_YEAR],
        "combustion": rel(comb),
        "process": rel(proc),
    }


def project(E_base, proc_base, ratios):
    """Project base-year canton values [canton, branch, fuel] to YEARS_FTS."""
    nc, nb, nf = E_base.shape
    E = np.zeros((nc, len(YEARS_FTS), nb, nf))
    proc = np.zeros((nc, len(YEARS_FTS), nb))
    je = FUELS.index("electricity")
    is_fossil = np.array([f in FOSSIL_FUELS for f in FUELS])
    is_other = ~is_fossil
    is_other[je] = False
    ind = slice(0, 12)
    for iy, y in enumerate(YEARS_FTS):
        r_comb = ratios["combustion"][y].reindex(BRANCHES).values  # per branch
        r_proc = ratios["process"][y].reindex(BRANCHES).values
        E[:, iy, :, je] = E_base[:, :, je] * ratios["electricity"][y]
        E[:, iy][..., is_fossil] = E_base[..., is_fossil] * r_comb[None, :, None]
        # Non-electric, non-fossil fuels of the industry branches take what is left
        # of the EP2050+ total energy trajectory (one factor for all branches).
        target = E_base[:, ind].sum() * ratios["total"][y]
        rest = target - E[:, iy, ind, je].sum() - E[:, iy, ind][..., is_fossil].sum()
        other_base = E_base[:, ind][..., is_other].sum()
        k = max(rest, 0.0) / other_base if other_base > 0 else 1.0
        E[:, iy, ind, :] = np.where(
            is_other[None, None, :], E_base[:, ind, :] * k, E[:, iy, ind, :]
        )
        # water-waste: only electricity follows EP2050+; other fuels held constant
        E[:, iy, 12, :] = np.where(
            is_other | is_fossil, E_base[:, 12, :], E[:, iy, 12, :]
        )
        proc[:, iy, :] = proc_base * r_proc[None, :]
    return E, proc


# ======================================================================================
# Outputs
# ======================================================================================
def make_dm(arr, variables, cat1, cat2, units, countries, years):
    """arr [country, year, variable, cat1, cat2] -> DataMatrix."""
    dm = DataMatrix(
        col_labels={
            "Country": countries,
            "Years": years,
            "Variables": variables,
            "Categories1": cat1,
            "Categories2": cat2,
        },
        units=units,
    )
    dm.array = arr
    dm.sort("Country")  # alphabetical, Switzerland included (as in other pickles)
    return dm


def with_switzerland(arr):
    """Append the Swiss total (sum of cantons) as the last country."""
    return np.concatenate([arr, arr.sum(axis=0, keepdims=True)], axis=0)


def build_dms(E, proc, years):
    """E [canton, year, branch, fuel], proc [canton, year, branch] -> energy, emissions dm."""
    countries = CANTONS + ["Switzerland"]
    dm_energy = make_dm(
        with_switzerland(E)[:, :, None, :, :],
        ["ind_energy-demand"],
        BRANCHES,
        FUELS,
        {"ind_energy-demand": "TWh"},
        countries,
        years,
    )
    em = emissions_from_energy(E, proc)
    names = {
        "combustion": "ind_emissions-combustion",
        "biogenic": "ind_emissions-biogenic",
        "process": "ind_emissions-process",
    }
    arr = np.stack([with_switzerland(em[k]) for k in names], axis=2)
    dm_em = make_dm(
        arr,
        list(names.values()),
        BRANCHES,
        GASES,
        {v: "Mt" for v in names.values()},
        countries,
        years,
    )
    return dm_energy, dm_em


def ef_constant_dm():
    ef_f, ef_b = ef_arrays()
    cdm = ConstantDataMatrix(
        col_labels={
            "Variables": ["ind_emission-factor"],
            "Categories1": ["fossil", "biogenic"],
            "Categories2": FUELS,
            "Categories3": GASES,
        },
        units={"ind_emission-factor": "Mt/TWh"},
    )
    cdm.array = np.stack([ef_f, ef_b], axis=0)[None, ...]
    return cdm


def prtr_dm(prtr_full, years):
    """PRTR CO2 [canton, year, branch] (all PRTR years) as a DataMatrix."""
    arr = with_switzerland(np.nan_to_num(prtr_full))[:, :, None, :]
    dm = DataMatrix(
        col_labels={
            "Country": CANTONS + ["Switzerland"],
            "Years": years,
            "Variables": ["ind_prtr-co2"],
            "Categories1": BRANCHES,
        },
        units={"ind_prtr-co2": "Mt"},
    )
    dm.array = arr
    dm.sort("Country")
    return dm


def end_use_shares(sheet, years):
    """EP2050+ end-use shares of industry energy, for electricity and other fuels.

    Returns two DataFrames (year x end use) that sum to 1 per year.
    """
    t1 = gd.read_ep2050_table(sheet, 1)
    t2 = gd.read_ep2050_table(sheet, 2)
    rows = {
        "space-heating": ["Raumwärme"],
        "hot-water": ["Warmwasser"],
        "process-heat": ["Prozesswärme"],
        "lighting": ["Beleuchtung"],
        "elec": [
            "Klima, Lüftung und Haustechnik",
            "Antriebe, Prozesse",
            "Sonstige Verwendungszwecke",
        ],
    }
    elec = pd.DataFrame({k: t2.loc[v].sum() for k, v in rows.items()})
    other = pd.DataFrame({k: (t1.loc[v] - t2.loc[v]).sum() for k, v in rows.items()})
    # Non-electric 'Sonstige' (mainly environmental protection) is heat: process heat
    other["process-heat"] += other["elec"] + other["lighting"]
    other[["elec", "lighting"]] = 0.0
    elec = elec.clip(lower=0).div(elec.clip(lower=0).sum(axis=1), axis=0)
    other = other.clip(lower=0).div(other.clip(lower=0).sum(axis=1), axis=0)
    return elec.loc[years], other.loc[years]


INTERFACE_FUELS = [
    "biogas",
    "biomass",
    "coal",
    "district-heating",
    "electricity",
    "gas",
    "heat-pump",
    "heating-oil",
    "nuclear-fuel",
    "other",
    "renewables",
    "solar",
    "waste",
    "wood",
]
INTERFACE_END_USES = ["elec", "hot-water", "lighting", "process-heat", "space-heating"]


def build_energy_interface(dm_energy_ots, dm_energy_fts_lev1):
    """industry_to_energy.pickle: {'ind-energy-demand': dm}, end use x fuel, TWh."""
    dm = dm_energy_ots.copy()
    dm.append(dm_energy_fts_lev1, dim="Years")
    years = dm.col_labels["Years"]
    tot = dm.array[:, :, 0, :, :].sum(axis=2)  # country, year, fuel
    sh_elec, sh_other = end_use_shares(EP_SHEETS[1], years)
    out = np.zeros(
        (len(dm.col_labels["Country"]), len(years), 1, 5, len(INTERFACE_FUELS))
    )
    fuel_map = {f: f for f in FUELS}
    fuel_map["other-oil"] = "heating-oil"
    for jf, f in enumerate(FUELS):
        jt = INTERFACE_FUELS.index(fuel_map[f])
        sh = sh_elec if f == "electricity" else sh_other
        for je, eu in enumerate(INTERFACE_END_USES):
            out[:, :, 0, je, jt] += tot[:, :, jf] * sh[eu].values[None, :]
    dm_out = make_dm(
        out,
        ["ind_energy-end-use"],
        INTERFACE_END_USES,
        INTERFACE_FUELS,
        {"ind_energy-end-use": "TWh"},
        dm.col_labels["Country"],
        years,
    )
    return {"ind-energy-demand": dm_out}


# ======================================================================================
# Checks
# ======================================================================================
def check_and_report(E_c, gest, prtr, proc_c, violations, strict):
    ok = True
    # Sum of cantons = GEST industry, per fuel
    E_ch = E_c.sum(axis=0).sum(axis=1)  # year, fuel
    ww_out = [FUELS.index(f) for f in WW_OUTSIDE_GEST]
    print(
        f"[report] water-waste {WW_OUTSIDE_GEST} outside GEST, 2023: "
        f"{E_c.sum(axis=0)[-1, 12, ww_out].round(3)} TWh"
    )
    E_ch[:, ww_out] -= E_c.sum(axis=0)[:, 12, ww_out]
    gest_f = gest.loc[YEARS_OTS, FUELS].values
    err = np.abs(E_ch - gest_f).max()
    print(f"\n[check] max |sum cantons - GEST T17b| over years x fuels: {err:.2e} TWh")
    if err > 1e-6:
        ok = False
    print(
        "[check] 2023 energy by fuel (TWh), model vs GEST:\n",
        pd.DataFrame({"model": E_ch[-1], "GEST": gest_f[-1]}, index=FUELS).round(3),
    )
    # canton x branch CO2 >= PRTR
    T_c = total_co2(E_c, proc_c)  # canton, year, branch
    rows = []
    for iy, y in enumerate(YEARS_OTS):
        if np.isnan(prtr[0, iy, 0]):
            continue
        for ic, c in enumerate(CANTONS):
            for ib, b in enumerate(BRANCHES):
                p = prtr[ic, iy, ib]
                if p > 0:
                    rows.append((y, c, b, T_c[ic, iy, ib], p))
    df = pd.DataFrame(rows, columns=["year", "canton", "branch", "model_Mt", "prtr_Mt"])
    df["margin_Mt"] = df["model_Mt"] - df["prtr_Mt"]
    bad = df[df["margin_Mt"] < -1e-9]
    print(f"\n[check] canton x branch CO2 >= PRTR: {len(df) - len(bad)}/{len(df)} pass")
    print("[check] 2023 canton x branch vs PRTR (Mt CO2, fossil + biogenic + process):")
    print(df[df.year == 2023].round(3).to_string(index=False))
    if violations:
        print("\n[check] NATIONAL branch total below PRTR sum:")
        for v in violations:
            print(f"   {v[0]} {v[1]:20s} model {v[2]:.3f} < PRTR {v[3]:.3f} Mt")
    if len(bad):
        ok = False
        print(bad.round(4).to_string(index=False))
    if not ok and strict:
        raise ValueError("simple_industry checks failed (see report above)")
    return df


def report_inventory(E_nat, proc_nat):
    inv = gd.read_foen_inventory("CO2")
    em = emissions_from_energy(E_nat, proc_nat)
    ico2 = GASES.index("CO2")
    comb = em["combustion"][:, :12, ico2].sum(axis=1)
    proc = proc_nat.sum(axis=1)
    df = pd.DataFrame(
        {
            "model comb. fossil (12 branches)": comb,
            "FOEN 1A2": inv["1A2 Manufacturing industries and construction"].reindex(
                YEARS_OTS
            ),
            "model process (CO2eq)": proc,
            "FOEN 2 IPPU (CO2)": inv["2 Industrial processes and product use"].reindex(
                YEARS_OTS
            ),
        },
        index=YEARS_OTS,
    )
    print("\n[report] national CO2 vs FOEN inventory (Mt):")
    print(df.loc[[1990, 2000, 2010, 2015, 2020, 2023]].round(2))


# ======================================================================================
def run(strict=True, write=True):
    # --- data -------------------------------------------------------------------------
    gest = gd.read_gest_industry()
    bfe = gd.read_bfe_branch_fuel()
    ep_comb_wwb = gd.read_ep2050_branch_table(EP_SHEETS[1], 3)
    ep_proc_wwb = gd.read_ep2050_branch_table(EP_SHEETS[1], 4)
    dm_accounts = extract_energy_accounts(  # STAT-TAB API (cached)
        "px-x-0204000000_106",
        os.path.join(gd.POWER_DATA_DIR, "energy_accounts_economy_households.pickle"),
    )
    dm_employees = extract_employees_per_sector_canton(  # STAT-TAB API (cached)
        "px-x-0602010000_101",
        os.path.join(gd.POWER_DATA_DIR, "employees_per_sector_canton.pickle"),
    )
    df_prtr = gd.read_prtr_co2()

    # --- national ---------------------------------------------------------------------
    ww = water_waste_own_use(dm_accounts)
    E_nat = national_energy(gest, bfe, ep_comb_wwb, ww)  # year, branch, fuel
    proc_nat = ep_proc_wwb[YEARS_OTS].T.values  # year, branch (EP2050+ WWB)

    # --- cantons ----------------------------------------------------------------------
    share_fte = fte_shares(dm_employees)
    prtr = gd.prtr_by_canton_branch(df_prtr, CANTONS, YEARS_OTS, BRANCHES)
    E_c, proc_c, w, violations = allocate_to_cantons(E_nat, proc_nat, share_fte, prtr)

    report_inventory(E_nat, proc_nat)
    check_and_report(E_c, gest, prtr, proc_c, violations, strict)

    # --- forecast ---------------------------------------------------------------------
    ib = YEARS_OTS.index(BASE_YEAR)
    ratios = {lev: ep2050_ratios(sheet) for lev, sheet in EP_SHEETS.items()}
    proj = {lev: project(E_c[:, ib], proc_c[:, ib], ratios[lev]) for lev in (1, 4)}
    for lev, a in ((2, 1 / 3), (3, 2 / 3)):
        proj[lev] = tuple(proj[1][k] + a * (proj[4][k] - proj[1][k]) for k in range(2))

    # --- outputs ----------------------------------------------------------------------
    dm_energy, dm_em = build_dms(E_c, proc_c, YEARS_OTS)
    fts = {}
    for lev in (1, 2, 3, 4):
        e, m = build_dms(proj[lev][0], proj[lev][1], YEARS_FTS)
        fts[lev] = {"energy": e, "emissions": m}
    prtr_years = sorted(df_prtr["year"].unique())
    DM = {
        "ots": {"simple-industry": {"energy": dm_energy, "emissions": dm_em}},
        "fts": {"simple-industry": fts},
        "fxa": {
            "prtr-co2": prtr_dm(
                gd.prtr_by_canton_branch(df_prtr, CANTONS, prtr_years, BRANCHES),
                prtr_years,
            )
        },
        "constant": {"emission-factors": ef_constant_dm()},
    }
    DM_interface = build_energy_interface(dm_energy, fts[1]["energy"])

    if write:
        f_out = os.path.join(DATAMATRIX_DIR, "simple_industry.pickle")
        with open(f_out, "wb") as handle:
            pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\nSaved {f_out}")
        f_int = os.path.join(INTERFACE_DIR, "industry_to_energy.pickle")
        with open(f_int, "wb") as handle:
            pickle.dump(DM_interface, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {f_int}")
    return DM, DM_interface


if __name__ == "__main__":
    run()
