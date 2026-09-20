import os
import pickle
import re

import numpy as np
import pandas as pd

from tcaf_model.model.common.auxiliary_functions import (
    calibration_rates,
    create_years_list,
    dm_match_countries,
    filter_country_and_load_data_from_pickles,
    linear_fitting,
)
from tcaf_model.model.common.config_loader import load_lever_config
from tcaf_model.model.common.data_matrix_class import DataMatrix
from tcaf_model.model.common.interface_class import Interface


def init_years_lever():
    # function that can be used when running the module as standalone to initialise years and levers
    years_setting = [1990, 2023, 2025, 2050, 5]
    lever_setting = load_lever_config()
    return years_setting, lever_setting


# CalculationLeaf READ PICKLE
def read_data(DM_TCAF, lever_setting, years_all):
    # Read fts based on lever_setting
    # DM_ots_fts = read_level_data(DM_TCAF, lever_setting)

    # Sub-matrix for TCAF health-diet
    dm_tcaf_paf = DM_TCAF["fxa"]["health-diet_paf"]
    dm_tcaf_dalys = DM_TCAF["fxa"]["health-diet_dalys"]

    # Aggregate Data Matrix - DIETARY HABITS
    DM_TCAF_health_diet = {
        "health-diet_paf": dm_tcaf_paf,
        "health-diet_dalys": dm_tcaf_dalys,
    }

    # Aggregate Data Matrix - LCA
    DM_TCAF_lca = {
        "lca-switzerland": DM_TCAF["fxa"]["lca"]["lca-switzerland"],
        "lca-world": DM_TCAF["fxa"]["lca"]["lca-world"],
    }
    # Observed Swiss agricultural GHG inventory (IPCC cat. 3), two total
    # CO2-eq series [t CO2-eq]: cat-3 total and 3A+3B livestock sub-total.
    DM_TCAF_lca["cal-ghg"] = DM_TCAF["fxa"].get("cal_lca_ghg", None)
    # Population-based per-animal GHG factors (Table 5) + lsu/head, for the
    # alternative livestock GHG path.
    DM_TCAF_lca["ghg-ef-perhead"] = DM_TCAF["fxa"].get("liv_ghg_ef_perhead", None)
    DM_TCAF_lca["lsu-per-head"] = DM_TCAF["fxa"].get("liv_lsu_per_head", None)
    # Swiss domestic fish (LCA per method, self-sufficiency, wild-catch reference);
    # absent in a pickle built before TCAF_fish_preprocessing.py was run.
    DM_TCAF_lca["fish"] = DM_TCAF["fxa"].get("fish", None)

    # Aggregate Data Matrix - BIODIVERSITY
    DM_TCAF_biodiversity = {
        "biodiversity-ch": DM_TCAF["fxa"]["biodiversity"]["TCAF-biodiversity-CH"],
        "biodiversity-world": DM_TCAF["fxa"]["biodiversity"]["TCAF-biodiversity-world"],
    }
    for key in DM_TCAF_biodiversity.keys():
        linear_fitting(DM_TCAF_biodiversity[key], years_all)
        DM_TCAF_biodiversity[key].filter({"Years": years_all}, inplace=True)

    # Constants
    # Monetization factors
    CDM_MF = {}
    # For health-diet
    cdm_temp = DM_TCAF["constant"]["monetization-factors"].filter_w_regex(
        {"Variables": "tcaf_mf_health-diet.*"}
    )
    CDM_MF["health-diet"] = cdm_temp
    # For environmental LCA (one factor per impact category); the LCA impacts are
    # stored PHYSICAL in pre-processing and monetized in TCAF_lca_workflow.
    CDM_MF["env-lca"] = DM_TCAF["constant"]["monetization-factors-lca"].copy()

    # Other constants
    CDM_const = {}
    CDM_const["cdm_kcal"] = DM_TCAF["constant"]["cdm_kcal"].copy()

    return DM_TCAF_lca, DM_TCAF_health_diet, DM_TCAF_biodiversity, CDM_MF, CDM_const


# SimulateInteractions


def simulate_diet_to_TCAF_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(
        current_file_directory,
        "../_database/data/interface/dietary-habits_to_TCAF.pickle",
    )
    with open(f, "rb") as handle:
        DM_diet = pickle.load(handle)
    return DM_diet


def simulate_landuse_to_TCAF_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    f = os.path.normpath(
        os.path.join(
            current_file_directory,
            "../_database/data/interface/land-use_to_TCAF.pickle",
        )
    )
    with open(f, "rb") as handle:
        dm_cropland = pickle.load(handle)
    return dm_cropland


def simulate_crop_to_TCAF_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(
        current_file_directory, "../_database/data/interface/crop_to_TCAF.pickle"
    )
    with open(f, "rb") as handle:
        DM_crop_to_TCAF = pickle.load(handle)
    return DM_crop_to_TCAF


def simulate_livestock_to_TCAF_input():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(
        current_file_directory, "../_database/data/interface/livestock_to_TCAF.pickle"
    )
    with open(f, "rb") as handle:
        DM_livestock_to_TCAF = pickle.load(handle)
    return DM_livestock_to_TCAF


# CalculationLeaf TCAF LCA
# ---------------------------------------------------------------------------
# Helpers - the only three places that stay in numpy, each because the
# DataMatrix API has no equivalent broadcast operation.
# ---------------------------------------------------------------------------
def _cy_to_t_dm(arr_cy, ref_dm, name):
    """Wrap a (Country, Years) array [kg CO2-eq] as a [t CO2-eq] DataMatrix.

    Structure (Country/Years labels) is inherited from `ref_dm` via
    DataMatrix.based_on, so no hand-built col_labels. The array is copied
    first because change_unit() rescales in place and a bare slice would be a
    view onto the caller's data.
    """
    dm = DataMatrix.based_on(
        np.array(arr_cy, dtype=float)[:, :, np.newaxis],
        format=ref_dm,
        change={
            "Variables": [name],
            "Categories1": None,
            "Categories2": None,
            "Categories3": None,
        },
        units={name: "kg CO2-eq"},
    )
    dm.change_unit(name, 1e-3, old_unit="kg CO2-eq", new_unit="t CO2-eq")
    return dm


def _sum_asf_total(arr_gw, ref_ch_dm):
    """(ASF, crop, total) GHG per (Country, Years) from a food x method slice.

    `arr_gw` is the global-warming physical slice [kg CO2-eq], shape
    (Country, Years, food, method). It is wrapped as a DataMatrix and summed
    with group_all() (== nansum, but label-safe), and the ASF subset is picked
    with filter_w_regex() instead of a positional boolean mask.
    """
    dm = DataMatrix.based_on(
        arr_gw[:, :, np.newaxis, :, :],
        format=ref_ch_dm,
        change={"Variables": ["ghg"], "Categories3": None},
        units={"ghg": "kg CO2-eq"},
    )
    dm_total = dm.group_all("Categories2", inplace=False)  # sum over method
    dm_total = dm_total.group_all("Categories1", inplace=False)  # sum over food
    dm_asf = dm.filter_w_regex({"Categories1": "meat-.*|abp-.*"})
    dm_asf = dm_asf.group_all("Categories2", inplace=False)
    dm_asf = dm_asf.group_all("Categories1", inplace=False)
    total = dm_total.array[:, :, 0]
    asf = dm_asf.array[:, :, 0]
    return asf, total - asf, total


def _monetize_by_impact(phys_array, impact_labels, cdm_mf_lca):
    """Multiply a physical-impact array by the per-impact factor [CHF/impact-unit].

    The LAST axis of `phys_array` must be the impact category, ordered as
    `impact_labels`. The factor is pulled from the ConstantDataMatrix BY LABEL
    (via its __getitem__/idx), so it stays aligned whatever order the constant
    itself stores. This one broadcast multiply is kept in numpy on purpose:
    the class cannot multiply a matrix by a per-category constant that
    broadcasts over the other axes.
    """
    v = cdm_mf_lca.col_labels["Variables"][0]  # 'tcaf_mf_lca'
    mf = np.array([float(cdm_mf_lca[v, c]) for c in impact_labels], dtype=float)
    return phys_array * mf  # broadcasts over leading axes


# ---------------------------------------------------------------------------
def TCAF_lca_workflow(
    DM_TCAF_lca,
    DM_crop_to_TCAF,
    DM_landuse_to_TCAF,
    DM_livestock_to_TCAF,
    CDM_const,
    CDM_MF,
    years_setting=None,
    dm_cal_ghg_lca=None,
    cdm_ghg_ef_perhead=None,
    cdm_lsu_per_head=None,
    return_diagnostics=False,
):
    # Match countries FIXME better match countries
    DM_livestock_to_TCAF["meat-world"].drop(
        col_label=["Switzerland", "Tokelau"], dim="Country"
    )
    DM_livestock_to_TCAF["asf-world"].drop(
        col_label=["Switzerland", "Tokelau"], dim="Country"
    )
    DM_crop_to_TCAF.drop(col_label=["Switzerland"], dim="Country")
    dm_match_countries(
        DM_crop_to_TCAF, DM_TCAF_lca["lca-world"], parameter="perfect match"
    )
    dm_match_countries(
        DM_livestock_to_TCAF["asf-world"],
        DM_TCAF_lca["lca-world"],
        parameter="perfect match",
    )
    dm_match_countries(
        DM_livestock_to_TCAF["meat-world"],
        DM_TCAF_lca["lca-world"],
        parameter="perfect match",
    )

    # (Switzerland & World) Crop - Unit conversion: [kcal] to [kg]
    cdm_kcal = CDM_const["cdm_kcal"].copy()
    cat = DM_crop_to_TCAF.col_labels["Categories1"]
    cdm_kcal = cdm_kcal.filter({"Categories1": cat})
    DM_crop_to_TCAF.sort("Categories1")
    cdm_kcal.sort("Categories1")
    array_temp = (
        10**3
        * DM_crop_to_TCAF[:, :, "agr_domestic-production_afw", :]
        / cdm_kcal[np.newaxis, np.newaxis, "cp_kcal-per-t", :]
    )
    DM_crop_to_TCAF.add(
        array_temp,
        dim="Variables",
        col_label="agr_domestic-production_afw_kg",
        unit="kg",
    )
    DM_landuse_to_TCAF["prod-ch"].add(
        0.0, dummy=True, col_label="rice", dim="Categories2", unit="kcal"
    )
    array_temp = (
        10**3
        * DM_landuse_to_TCAF["prod-ch"][:, :, "agr_domestic-production_afw", :, :]
        / cdm_kcal[np.newaxis, np.newaxis, "cp_kcal-per-t", np.newaxis, :]
    )
    DM_landuse_to_TCAF["prod-ch"].add(
        array_temp,
        dim="Variables",
        col_label="agr_domestic-production_afw_kg",
        unit="kg",
    )

    # FIXME cereals = cereals + rice
    DM_landuse_to_TCAF["prod-ch"].groupby(
        {"crop-cereal": "crop-cereal|crop-rice"},
        dim="Categories2",
        inplace=True,
        regex=True,
    )
    DM_crop_to_TCAF.groupby(
        {"crop-cereal": "crop-cereal|crop-rice"},
        dim="Categories1",
        inplace=True,
        regex=True,
    )

    # FIXME drop extensive for crops CH for now
    DM_landuse_to_TCAF["prod-ch"].drop(dim="Categories1", col_label="extensive")

    # Rename production variables to a common name
    DM_landuse_to_TCAF["prod-ch"].rename_col(
        "agr_domestic-production_afw_kg", "agr_production-lca", dim="Variables"
    )
    DM_crop_to_TCAF.rename_col(
        "agr_domestic-production_afw_kg", "agr_production-lca", dim="Variables"
    )
    # Meat impacts are per kg LIVEWEIGHT -> use liveweight, not edible weight.
    DM_livestock_to_TCAF["meat-ch"].rename_col(
        "liveweight-production", "agr_production-lca", dim="Variables"
    )
    DM_livestock_to_TCAF["meat-world"].rename_col(
        "liveweight-production", "agr_production-lca", dim="Variables"
    )
    # Egg/milk impacts are per kg of product -> use the product mass.
    DM_livestock_to_TCAF["asf-ch"].rename_col(
        "agr_domestic_production_liv_afw_kg", "agr_production-lca", dim="Variables"
    )
    DM_livestock_to_TCAF["asf-world"].rename_col(
        "agr_domestic_production_liv_afw_kg", "agr_production-lca", dim="Variables"
    )

    # LCA monetization factors [CHF/impact-unit]; impacts are physical here.
    cdm_mf_lca = CDM_MF["env-lca"]

    # =====================================================================
    # Step World
    # =====================================================================
    dm_lca_world = DM_livestock_to_TCAF["meat-world"].filter(
        {"Variables": ["agr_production-lca"]}
    )
    dm_lca_world.append(
        DM_livestock_to_TCAF["asf-world"].filter({"Variables": ["agr_production-lca"]}),
        dim="Categories1",
    )
    dm_lca_world.append(
        DM_crop_to_TCAF.filter({"Variables": ["agr_production-lca"]}), dim="Categories1"
    )

    # Keep only food categories present in BOTH matrices, aligned order (the
    # multiply over Categories1 is positional).
    common_cat = sorted(
        set(dm_lca_world.col_labels["Categories1"])
        & set(DM_TCAF_lca["lca-world"].col_labels["Categories1"])
    )
    dm_lca_world.filter({"Categories1": common_cat}, inplace=True)
    dm_lca_world.sort("Categories1")
    DM_TCAF_lca["lca-world"].filter({"Categories1": common_cat}, inplace=True)
    DM_TCAF_lca["lca-world"].sort("Categories1")

    # [ISLAND 1] physical impact = production[kg] (x) impact[/kg] over the impact
    # category - an outer product the class has no method for.
    array_phys_world = (
        dm_lca_world[:, :, "agr_production-lca", :, np.newaxis]
        * DM_TCAF_lca["lca-world"][:, :, "lca-impacts", :, :]
    )
    imp_world = DM_TCAF_lca["lca-world"].col_labels["Categories2"]
    array_temp = _monetize_by_impact(array_phys_world, imp_world, cdm_mf_lca)
    DM_TCAF_lca["lca-world"].add(
        array_temp, dim="Variables", col_label="agr_production-tcaf", unit="CHF"
    )

    # =====================================================================
    # Step Switzerland
    # =====================================================================
    dm_lca_ch = DM_livestock_to_TCAF["meat-ch"].filter(
        {"Variables": ["agr_production-lca"]}
    )
    dm_lca_ch.append(
        DM_livestock_to_TCAF["asf-ch"].filter({"Variables": ["agr_production-lca"]}),
        dim="Categories1",
    )
    DM_landuse_to_TCAF["prod-ch"].switch_categories_order(
        cat1="Categories2", cat2="Categories1"
    )
    dm_lca_ch.append(
        DM_landuse_to_TCAF["prod-ch"].filter({"Variables": ["agr_production-lca"]}),
        dim="Categories1",
    )

    # Align food (Categories1) and method (Categories2) on both matrices.
    common_food = sorted(
        set(dm_lca_ch.col_labels["Categories1"])
        & set(DM_TCAF_lca["lca-switzerland"].col_labels["Categories1"])
    )
    common_method = sorted(
        set(dm_lca_ch.col_labels["Categories2"])
        & set(DM_TCAF_lca["lca-switzerland"].col_labels["Categories2"])
    )
    common_years = sorted(
        set(dm_lca_ch.col_labels["Years"])
        & set(DM_TCAF_lca["lca-switzerland"].col_labels["Years"])
    )
    for dm in (dm_lca_ch, DM_TCAF_lca["lca-switzerland"]):
        dm.filter(
            {
                "Years": common_years,
                "Categories1": common_food,
                "Categories2": common_method,
            },
            inplace=True,
        )
        dm.sort("Categories1")
        dm.sort("Categories2")
        dm.sort("Years")

    # [ISLAND 1] physical impact, method-resolved: production (c,y,food,method)
    # (x) impacts (c,y,food,method,impact).
    array_phys = (
        dm_lca_ch[:, :, "agr_production-lca", :, :, np.newaxis]
        * DM_TCAF_lca["lca-switzerland"][:, :, "lca-impacts", :, :, :]
    )

    # ---- GHG CALIBRATION (Switzerland domestic only) ---------------------
    ref_ch = DM_TCAF_lca["lca-switzerland"]  # structure donor for the helpers
    imp_ch = ref_ch.col_labels["Categories3"]
    food_ch = ref_ch.col_labels["Categories1"]
    method_ch = ref_ch.col_labels["Categories2"]
    countries_ch = list(ref_ch.col_labels["Country"])
    years_ch = list(ref_ch.col_labels["Years"])
    dm_ghg_lca_ch = None
    _diag = {}

    def _calib(dm_raw_single, cal_var):
        if dm_cal_ghg_lca is None or years_setting is None:
            return None
        if cal_var not in dm_cal_ghg_lca.col_labels["Variables"]:
            return None
        dmc = dm_cal_ghg_lca.filter({"Variables": [cal_var]})
        return calibration_rates(
            dm_raw_single,
            dmc,
            years_setting,
            calibration_start_year=1990,
            calibration_end_year=2023,
        )

    if "global-warming" in imp_ch:
        i_gw = imp_ch.index("global-warming")

        # ---- (P) production-based GHG, ASF vs crops (idiomatic sums) --------
        prod_asf, prod_crop, prod_total = _sum_asf_total(
            array_phys[:, :, :, :, i_gw], ref_ch
        )
        is_asf = np.array(
            [f.startswith("meat-") or f.startswith("abp-") for f in food_ch]
        )

        dm_ghg_lca_ch = _cy_to_t_dm(prod_total, ref_ch, "tcaf_lca_ghg-emissions_raw")
        print(
            "\n[TCAF LCA] Swiss domestic agricultural GHG (global-warming), "
            "production-based raw total [t CO2-eq]:"
        )
        for ci, ctry in enumerate(dm_ghg_lca_ch.col_labels["Country"]):
            for yi, yr in enumerate(dm_ghg_lca_ch.col_labels["Years"]):
                print(f"    {ctry} {yr}: {dm_ghg_lca_ch.array[ci, yi, 0]:,.0f}")

        # ---- (S) POPULATION-BASED per-impact ASF impacts (per head) --------
        # [ISLAND 2] head = population[lsu] / lsu-per-head, then head x per-head
        # EF, swapped into the ASF slices of array_phys for every impact. This
        # is a per-(food, method, impact) contraction the class cannot express,
        # so it stays numpy; the rest of the GHG handling below is idiomatic.
        pop_asf = None
        dm_pop = (
            DM_livestock_to_TCAF.get("population-ch", None)
            if isinstance(DM_livestock_to_TCAF, dict)
            else None
        )
        if (
            cdm_ghg_ef_perhead is not None
            and cdm_lsu_per_head is not None
            and dm_pop is not None
        ):
            try:
                keep_c = [c for c in countries_ch if c in dm_pop.col_labels["Country"]]
                keep_y = [y for y in years_ch if y in dm_pop.col_labels["Years"]]
                dm_pop = dm_pop.filter(
                    {"Country": keep_c, "Years": keep_y}, inplace=False
                )
                col_keep = [years_ch.index(y) for y in keep_y]
                ex, lx = cdm_ghg_ef_perhead.idx, cdm_lsu_per_head.idx
                ef_vars = set(cdm_ghg_ef_perhead.col_labels["Variables"])
                ef_foods = set(cdm_ghg_ef_perhead.col_labels["Categories1"])
                has_young_ef = "meat-bovine-young" in ef_foods

                dm_young = (
                    DM_livestock_to_TCAF.get("bovine-young-share-ch", None)
                    if isinstance(DM_livestock_to_TCAF, dict)
                    else None
                )
                if dm_young is not None:
                    dm_young = dm_young.filter(
                        {"Country": keep_c, "Years": keep_y}, inplace=False
                    )

                def _young_share(meth):
                    if (
                        dm_young is None
                        or meth not in dm_young.col_labels["Categories1"]
                    ):
                        return 0.0
                    ys = dm_young.array[
                        :, :, dm_young.idx["bovine_young-share"], dm_young.idx[meth]
                    ]
                    return np.nan_to_num(ys, nan=0.0)

                for fpos, food in enumerate(food_ch):
                    if not is_asf[fpos] or food not in ef_foods:
                        continue
                    lsu_ph = cdm_lsu_per_head.array[lx["liv_lsu-per-head"], lx[food]]
                    if not lsu_ph:
                        continue
                    for mpos, meth in enumerate(method_ch):
                        if meth not in ("intensive", "organic"):
                            continue
                        var = "agr_liv_population_" + meth
                        if var not in dm_pop.col_labels["Variables"]:
                            continue
                        head = (
                            dm_pop.array[:, :, dm_pop.idx[var], dm_pop.idx[food]]
                            / lsu_ph
                        )
                        if food == "meat-bovine" and has_young_ef:
                            ys = _young_share(meth)
                            head_adult, head_young = head * (1.0 - ys), head * ys
                        else:
                            head_adult, head_young = head, None
                        for ipos, imp in enumerate(imp_ch):
                            if imp not in ef_vars:
                                continue
                            val = (
                                head_adult
                                * cdm_ghg_ef_perhead.array[ex[imp], ex[food], ex[meth]]
                            )
                            if head_young is not None:
                                val = (
                                    val
                                    + head_young
                                    * cdm_ghg_ef_perhead.array[
                                        ex[imp], ex["meat-bovine-young"], ex[meth]
                                    ]
                                )
                            full = array_phys[:, :, fpos, mpos, ipos].copy()
                            full[:, col_keep] = np.nan_to_num(val)
                            array_phys[:, :, fpos, mpos, ipos] = full

                # population ASF GHG, read back from the swapped array (idiomatic sum)
                pop_asf, _, _ = _sum_asf_total(array_phys[:, :, :, :, i_gw], ref_ch)
            except Exception as e:
                print(f"  ⚠️ population-based impacts skipped ({type(e).__name__}: {e})")
                pop_asf = None

        # ---- Stage 1: ASF totals vs inventory 3A+3B ------------------------
        dm_prod_asf = _cy_to_t_dm(prod_asf, ref_ch, "ghg_asf_production_raw")
        r_prod_asf = _calib(dm_prod_asf, "cal_tcaf_lca_ghg-emissions-livestock")
        dm_pop_asf = (
            _cy_to_t_dm(pop_asf, ref_ch, "ghg_asf_population_raw")
            if pop_asf is not None
            else None
        )
        r_pop_asf = (
            _calib(dm_pop_asf, "cal_tcaf_lca_ghg-emissions-livestock")
            if dm_pop_asf is not None
            else None
        )

        if r_prod_asf is not None:
            cal_liv = dm_cal_ghg_lca.filter(
                {"Variables": ["cal_tcaf_lca_ghg-emissions-livestock"]}
            )
            oy_liv = cal_liv.col_labels["Years"]
            print("\n[TCAF LCA] STAGE 1 - livestock GHG vs inventory 3A+3B [t CO2-eq]:")
            print(
                f"    {'year':>6} {'observed':>12} {'prod-based':>12} {'x rate':>7} "
                f"{'pop-based':>12} {'x rate':>7}"
            )
            ci = 0
            for yi, yr in enumerate(years_ch):
                obs = (
                    cal_liv.array[ci, oy_liv.index(yr), 0]
                    if yr in oy_liv
                    else float("nan")
                )
                rp = r_prod_asf[ci, yi, "cal_rate"]
                pa = dm_prod_asf.array[ci, yi, 0]
                if dm_pop_asf is not None:
                    sp = dm_pop_asf.array[ci, yi, 0]
                    rs = r_pop_asf[ci, yi, "cal_rate"]
                    pop_str = f"{sp:>12,.0f} {rs:>7.3f}"
                else:
                    pop_str = f"{'-':>12} {'-':>7}"
                obs_str = f"{obs:>12,.0f}" if yr in oy_liv else f"{'-':>12}"
                print(f"    {yr:>6} {obs_str} {pa:>12,.0f} {rp:>7.3f} {pop_str}")

            def _dist(r):
                if r is None:
                    return None
                rr = np.array(
                    [
                        r[ci, years_ch.index(y), "cal_rate"]
                        for y in oy_liv
                        if y in years_ch
                    ]
                )
                return float(np.nanmean(np.abs(rr - 1.0)))

            d_prod, d_pop = _dist(r_prod_asf), _dist(r_pop_asf)
            if d_pop is not None:
                better = "production-based" if d_prod <= d_pop else "population-based"
                print(
                    f"    -> mean |rate-1|: production={d_prod:.3f}, population={d_pop:.3f} "
                    f"=> better match: {better}"
                )
            else:
                print(
                    f"    -> mean |rate-1|: production={d_prod:.3f} (population path unavailable)"
                )

            if return_diagnostics:
                ci0 = 0

                def _byyr(dm, israte):
                    if dm is None:
                        return None
                    return {
                        yr: (
                            float(dm[ci0, years_ch.index(yr), "cal_rate"])
                            if israte
                            else float(dm.array[ci0, years_ch.index(yr), 0])
                        )
                        for yr in years_ch
                    }

                _diag["years"] = list(years_ch)
                _diag["stage1"] = {
                    "observed_3A3B": {
                        yr: (
                            float(cal_liv.array[ci0, oy_liv.index(yr), 0])
                            if yr in oy_liv
                            else float("nan")
                        )
                        for yr in years_ch
                    },
                    "prod_asf_raw": _byyr(dm_prod_asf, False),
                    "prod_rate": _byyr(r_prod_asf, True),
                    "pop_asf_raw": _byyr(dm_pop_asf, False),
                    "pop_rate": _byyr(r_pop_asf, True),
                    "d_prod": d_prod,
                    "d_pop": d_pop,
                    "better": (
                        "production-based"
                        if (d_pop is None or d_prod <= d_pop)
                        else "population-based"
                    ),
                }

        # ---- Stage 2: grand total (crops + stage-1-calibrated ASF) vs cat-3 -
        def _stage1_total(asf_kg, r_asf, name):
            if r_asf is None:
                return None, None
            total_in = asf_kg * r_asf[:, :, "cal_rate"] + prod_crop
            dm_in = _cy_to_t_dm(total_in, ref_ch, name)
            return dm_in, _calib(dm_in, "cal_tcaf_lca_ghg-emissions")

        dm_prod_total_in, r_prod_total = _stage1_total(
            prod_asf, r_prod_asf, "ghg_total_production_stage1"
        )
        dm_pop_total_in, r_pop_total = (
            _stage1_total(pop_asf, r_pop_asf, "tcaf_lca_ghg-emissions_stage1")
            if pop_asf is not None
            else (None, None)
        )

        # Driver: population if available, else production.
        if r_pop_asf is not None and r_pop_total is not None:
            drive_r1, drive_r2 = (
                r_pop_asf[:, :, "cal_rate"],
                r_pop_total[:, :, "cal_rate"],
            )
            drive_total_in = dm_pop_total_in
        elif r_prod_asf is not None and r_prod_total is not None:
            drive_r1, drive_r2 = (
                r_prod_asf[:, :, "cal_rate"],
                r_prod_total[:, :, "cal_rate"],
            )
            drive_total_in = dm_prod_total_in
        else:
            drive_r1 = None

        if drive_r1 is not None:
            # [ISLAND 3] rescale ONLY the GHG slice in place: ASF food x rate1,
            # then all food x rate2. Applying a (c,y) rate that broadcasts over
            # food/method/impact is not expressible with operation(), so it stays
            # numpy on the GHG slice.
            rate1_food = np.ones((len(countries_ch), len(years_ch), len(food_ch)))
            rate1_food[:, :, is_asf] = drive_r1[:, :, np.newaxis]
            array_phys[:, :, :, :, i_gw] = (
                array_phys[:, :, :, :, i_gw] * rate1_food[:, :, :, np.newaxis]
            )
            array_phys[:, :, :, :, i_gw] = (
                array_phys[:, :, :, :, i_gw] * drive_r2[:, :, np.newaxis, np.newaxis]
            )

            # drive_total_in is already in t CO2-eq (built by _cy_to_t_dm), and
            # drive_r2 is a dimensionless rate, so build the calibrated total
            # directly in tonnes (no kg round-trip).
            dm_cal_total = DataMatrix.based_on(
                (drive_total_in.array[:, :, 0] * drive_r2)[:, :, np.newaxis],
                format=ref_ch,
                change={
                    "Variables": ["tcaf_lca_ghg-emissions"],
                    "Categories1": None,
                    "Categories2": None,
                    "Categories3": None,
                },
                units={"tcaf_lca_ghg-emissions": "t CO2-eq"},
            )
            dm_ghg_lca_ch.append(dm_cal_total, dim="Variables")

            cal_tot = dm_cal_ghg_lca.filter(
                {"Variables": ["cal_tcaf_lca_ghg-emissions"]}
            )
            oy_tot = cal_tot.col_labels["Years"]
            print(
                "\n[TCAF LCA] STAGE 2 - grand total (crops + stage-1-calibrated ASF) "
                "vs inventory cat-3 [t CO2-eq]  (population drives monetization):"
            )
            print(
                f"    {'year':>6} {'observed':>12} {'prod-based':>12} {'x rate':>7} "
                f"{'pop-based':>12} {'x rate':>7}"
            )
            ci = 0
            for yi, yr in enumerate(years_ch):
                obs = (
                    cal_tot.array[ci, oy_tot.index(yr), 0]
                    if yr in oy_tot
                    else float("nan")
                )
                if dm_prod_total_in is not None:
                    prod_str = (
                        f"{dm_prod_total_in.array[ci, yi, 0]:>12,.0f} "
                        f"{r_prod_total[ci, yi, 'cal_rate']:>7.3f}"
                    )
                else:
                    prod_str = f"{'-':>12} {'-':>7}"
                if dm_pop_total_in is not None:
                    pop_str = (
                        f"{dm_pop_total_in.array[ci, yi, 0]:>12,.0f} "
                        f"{r_pop_total[ci, yi, 'cal_rate']:>7.3f}"
                    )
                else:
                    pop_str = f"{'-':>12} {'-':>7}"
                obs_str = f"{obs:>12,.0f}" if yr in oy_tot else f"{'-':>12}"
                print(f"    {yr:>6} {obs_str} {prod_str} {pop_str}")

            if return_diagnostics:
                ci0 = 0

                def _byyr2(dm, israte):
                    if dm is None:
                        return None
                    return {
                        yr: (
                            float(dm[ci0, years_ch.index(yr), "cal_rate"])
                            if israte
                            else float(dm.array[ci0, years_ch.index(yr), 0])
                        )
                        for yr in years_ch
                    }

                _diag["stage2"] = {
                    "observed_cat3": {
                        yr: (
                            float(cal_tot.array[ci0, oy_tot.index(yr), 0])
                            if yr in oy_tot
                            else float("nan")
                        )
                        for yr in years_ch
                    },
                    "prod_total_raw": _byyr2(dm_prod_total_in, False),
                    "prod_total_rate": _byyr2(r_prod_total, True),
                    "pop_total_raw": _byyr2(dm_pop_total_in, False),
                    "pop_total_rate": _byyr2(r_pop_total, True),
                }

    # Monetize per impact category (last axis), broadcast over food and method.
    array_temp = _monetize_by_impact(array_phys, imp_ch, cdm_mf_lca)
    DM_TCAF_lca["lca-switzerland"].add(
        array_temp, dim="Variables", col_label="agr_production-tcaf", unit="CHF"
    )

    if return_diagnostics:
        return DM_TCAF_lca, dm_ghg_lca_ch, _diag
    return DM_TCAF_lca, dm_ghg_lca_ch


# CalculationLeaf TCAF LCA - FISH (Swiss domestic production)
# Fish has no crop/livestock-style module, so its Swiss production is built here
# from the diet demand. Kept out of TCAF_lca_workflow on purpose: that function
# calibrates the GHG of crops and livestock against the IPCC agriculture inventory
# (cat. 3), which does not include aquaculture or fishing.
FISH_DEMAND_CATEGORY = "seafood-ffish"  # label in the diet food-demand and in cdm_kcal


def TCAF_fish_lca_workflow(DM_fish, dm_food_demand, CDM_const, CDM_MF):
    """Monetized LCA of Swiss domestic freshwater-fish production [CHF].

    domestic production [t] = demand [kcal] / kcal-per-t x SSR
      demand : dietary-habits agr_demand, calibrated to the FBS food supply, so the
               tonnes are on the FBS (live weight equivalent) basis, the same basis
               as the SSR (FBS production / food) and as the LCA processes (per kg
               live weight at landing or farm gate). No edible-fraction step.
      split  : wild capture is held at its FishStatJ tonnage (lake catch does not
               follow the diet), aquaculture is the rest of the domestic production.
    impacts [impact-unit] = production [kg] x LCA [impact-unit/kg], then monetized
    per impact category with the same factors as crops and livestock.

    Returns a dict of DataMatrices:
      cost       agr_production-tcaf [CHF], food x method x impact
      production agr_production-lca-t [t], food x method
      ghg        agr_ghg-lca-t [t CO2-eq], food x method
      demand     tcaf_fish_demand [t], no categories
    """
    dm_lca = DM_fish["lca-ch"]
    dm_ssr = DM_fish["ssr"]
    dm_cap = DM_fish["capture-ref"]
    food = dm_lca.col_labels["Categories1"][0]
    methods = list(dm_lca.col_labels["Categories2"])
    impacts = list(dm_lca.col_labels["Categories3"])

    years = sorted(
        set(dm_food_demand.col_labels["Years"])
        & set(dm_lca.col_labels["Years"])
        & set(dm_ssr.col_labels["Years"])
    )
    dm_dem = dm_food_demand.filter(
        {
            "Variables": ["agr_demand"],
            "Categories1": [FISH_DEMAND_CATEGORY],
            "Years": years,
        },
        inplace=False,
    )
    dm_lca = dm_lca.filter({"Years": years}, inplace=False)
    dm_ssr = dm_ssr.filter({"Years": years}, inplace=False)
    dm_cap = dm_cap.filter({"Years": years}, inplace=False)

    cdm_kcal = CDM_const["cdm_kcal"]
    kcal_per_t = float(cdm_kcal["cp_kcal-per-t", FISH_DEMAND_CATEGORY])

    demand_t = dm_dem[:, :, "agr_demand", FISH_DEMAND_CATEGORY] / kcal_per_t
    total_t = demand_t * dm_ssr[:, :, "fish_ssr", food]
    capture_t = np.minimum(dm_cap[:, :, "fish_capture-ref", food], total_t)
    t_by_method = {"aquaculture": total_t - capture_t, "capture": capture_t}

    prod_t = np.stack([t_by_method[m] for m in methods], axis=-1)  # (c, y, method)
    dm_production = DataMatrix.based_on(
        prod_t[:, :, np.newaxis, np.newaxis, :],
        format=dm_lca,
        change={"Variables": ["agr_production-lca-t"], "Categories3": None},
        units={"agr_production-lca-t": "t"},
    )

    # [ISLAND] physical impact = production [kg] (x) impact [/kg] over the impact axis
    phys = (1e3 * prod_t)[:, :, :, np.newaxis] * dm_lca[:, :, "lca-impacts", food, :, :]
    cost = _monetize_by_impact(phys, impacts, CDM_MF["env-lca"])
    dm_cost = DataMatrix.based_on(
        cost[:, :, np.newaxis, np.newaxis, :, :],
        format=dm_lca,
        change={"Variables": ["agr_production-tcaf"]},
        units={"agr_production-tcaf": "CHF"},
    )
    ghg_t = phys[:, :, :, impacts.index("global-warming")] / 1e3  # kg -> t CO2-eq
    dm_ghg = DataMatrix.based_on(
        ghg_t[:, :, np.newaxis, np.newaxis, :],
        format=dm_lca,
        change={"Variables": ["agr_ghg-lca-t"], "Categories3": None},
        units={"agr_ghg-lca-t": "t"},
    )
    dm_demand = DataMatrix.based_on(
        demand_t[:, :, np.newaxis],
        format=dm_lca,
        change={
            "Variables": ["tcaf_fish_demand"],
            "Categories1": None,
            "Categories2": None,
            "Categories3": None,
        },
        units={"tcaf_fish_demand": "t"},
    )
    return {
        "cost": dm_cost,
        "production": dm_production,
        "ghg": dm_ghg,
        "demand": dm_demand,
    }


def TCAF_fish_TPE_interface(fish):
    """Swiss domestic fish outputs for the app (Production tab, Blue food).

    Variables, all Switzerland x years:
      tcaf_fish_production_<method> [t]   domestic production, aquaculture / capture
      tcaf_fish_import [t]                demand not met by domestic production
      tcaf_fish_ssr [%]                   domestic production / demand
      tcaf_fish_cost_<method> [CHF]       LCA cost by method
      tcaf_fish_cost_<impact> [CHF]       LCA cost by impact category
      tcaf_fish_ghg_<method> [t]          GHG emissions (t CO2-eq) by method
    """
    dm_prod = fish["production"].group_all("Categories1", inplace=False)  # -> method
    dm_prod.rename_col("agr_production-lca-t", "tcaf_fish_production", dim="Variables")
    dm_tpe = dm_prod.flattest()

    total_t = fish["production"].array.sum(axis=(3, 4))[:, :, 0]  # (c, y)
    demand_t = fish["demand"].array[:, :, 0]
    imports = DataMatrix.based_on(
        (demand_t - total_t)[:, :, np.newaxis],
        format=fish["demand"],
        change={"Variables": ["tcaf_fish_import"]},
        units={"tcaf_fish_import": "t"},
    )
    dm_tpe.append(imports, dim="Variables")
    with np.errstate(divide="ignore", invalid="ignore"):
        ssr = np.where(demand_t > 0, 100.0 * total_t / demand_t, 0.0)
    dm_tpe.append(
        DataMatrix.based_on(
            ssr[:, :, np.newaxis],
            format=fish["demand"],
            change={"Variables": ["tcaf_fish_ssr"]},
            units={"tcaf_fish_ssr": "%"},
        ),
        dim="Variables",
    )

    dm_cost = fish["cost"].copy()
    dm_cost.rename_col("agr_production-tcaf", "tcaf_fish_cost", dim="Variables")
    dm_cost.group_all("Categories1", inplace=True)  # sum over food -> method x impact
    dm_by_method = dm_cost.group_all("Categories2", inplace=False)  # sum over impact
    dm_by_impact = dm_cost.group_all("Categories1", inplace=False)  # sum over method
    dm_tpe.append(dm_by_method.flattest(), dim="Variables")
    dm_tpe.append(dm_by_impact.flattest(), dim="Variables")

    dm_ghg = fish["ghg"].group_all("Categories1", inplace=False)  # -> method
    dm_ghg.rename_col("agr_ghg-lca-t", "tcaf_fish_ghg", dim="Variables")
    dm_tpe.append(dm_ghg.flattest(), dim="Variables")
    return dm_tpe


# CalculationLeaf TCAF HEALTH DIET
def _project_dalys(dm_gbd, dm_demography, base_year=2025):
    """
    Frozen-rate demographic projection of DALYs (mirrors 03_projection_dalys.R).

      rate_{S,d} = DALYs^2023_{S,d} / P_{S}(base_year)
      D_d(y)     = sum_S rate_{S,d} * P_{S}(y)

    S is the stratification carried by the 2023 GBD table: either (age, sex) when
    age bands are available, or (sex) only. The 2023 per-capita rate is held
    constant and applied to the model's projected demography aggregated to S. At
    y = base_year, D_d = sum_S DALYs^2023_{S,d} (the GBD totals).

    dm_gbd        : Country x [2023] x [tcaf_health-diet_dalys] x disease [x age] x sex
    dm_demography : Country x Years  x [lfs_demography]         x (sex-age)   [inhabitants]
    returns       : Country x Years  x [tcaf_health-diet_dalys] x disease     [DALYs/y]
    """
    country = dm_gbd.col_labels["Country"]
    diseases = dm_gbd.col_labels["Categories1"]
    years = list(dm_demography.col_labels["Years"])
    has_age = (
        "Categories3" in dm_gbd.dim_labels
    )  # (disease, age, sex) vs (disease, sex)
    sexes = (
        dm_gbd.col_labels["Categories3"]
        if has_age
        else dm_gbd.col_labels["Categories2"]
    )
    ages = dm_gbd.col_labels["Categories2"] if has_age else None

    # Full demography P[country, year, age, sex] parsed from the sex-age categories
    demo_ages = sorted(
        {t.split("-", 1)[1] for t in dm_demography.col_labels["Categories1"]}
    )
    n_c, n_y, n_s = len(country), len(years), len(sexes)
    arr_demo = dm_demography.array[:, :, 0, :]  # (country, year, sex-age category)
    P_full = np.zeros((n_c, n_y, len(demo_ages), n_s))
    ida = {a: i for i, a in enumerate(demo_ages)}
    for ci, token in enumerate(dm_demography.col_labels["Categories1"]):
        sex, age = token.split("-", 1)  # e.g. 'female-below19' -> ('female', 'below19')
        if sex in sexes and age in ida:
            P_full[:, :, ida[age], sexes.index(sex)] = arr_demo[:, :, ci]

    yb = years.index(base_year)
    if has_age:
        # align demography ages to the GBD age order, then project over (age, sex)
        arr_gbd = dm_gbd.array[:, 0, 0, :, :, :]  # (c, d, a, s)
        P = np.stack([P_full[:, :, ida[a], :] for a in ages], axis=2)  # (c, y, a, s)
        base = P[:, yb, :, :]  # (c, a, s)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(
                base[:, np.newaxis, :, :] > 0, arr_gbd / base[:, np.newaxis, :, :], 0.0
            )  # (c, d, a, s)
        arr_dalys = np.einsum("cdas,cyas->cyd", rate, P)  # (c, y, d)
    else:
        # aggregate demography over age -> P_s(y), then project over sex
        arr_gbd = dm_gbd.array[:, 0, 0, :, :]  # (c, d, s)
        P = P_full.sum(axis=2)  # (c, y, s)
        base = P[:, yb, :]  # (c, s)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(
                base[:, np.newaxis, :] > 0, arr_gbd / base[:, np.newaxis, :], 0.0
            )  # (c, d, s)
        arr_dalys = np.einsum("cds,cys->cyd", rate, P)  # (c, y, d)

    dm_dalys = DataMatrix(
        col_labels={
            "Country": list(country),
            "Years": years,
            "Variables": ["tcaf_health-diet_dalys"],
            "Categories1": list(diseases),
        },
        units={"tcaf_health-diet_dalys": "DALYs/y"},
    )
    dm_dalys.array = arr_dalys[:, :, np.newaxis, :]
    return dm_dalys


def TCAF_health_diet_workflow(DM_diet, DM_TCAF_health_diet, CDM_MF):
    """
    Diet-attributable / avoidable DALYs, following the stratified-adherence logic
    of the R projection script (Projection.R).

    For every food risk-factor r and disease d:
      PAF_r(B) : PAF read off the dose-response curve at the BAU (reference) intake
      PAF_r(T) : PAF read off the curve at the full target intake (full adherent)
      PIF_r    : (PAF_r(B) - PAF_r(T)) / (1 - PAF_r(T))   [full adoption; may be < 0 if the diet worsens]
    PAF is read by linear interpolation with flat extrapolation
    (numpy.interp == R's approx(..., rule = 2)).

    Combine the food risk-factors of a disease multiplicatively (GBD standard):
      AF_ref = 1 - prod_r (1 - PAF_r(B))
      PIF    = 1 - prod_r (1 - PIF_r)

    Population adherence alpha (share of the population adopting the target diet)
    enters as a LINEAR scaling of the disease-level impact fraction - the
    two-strata mixture used by R, PIF(alpha) = alpha * PIF:
      attributable = AF_ref        * DALYs
      avoided      = alpha * PIF    * DALYs
      residual     = attributable - avoided

    Inputs from the dietary-habits interface:
      B     = diet-consumed_bau     (unweighted full BAU diet,     g/cap/day)
      T     = diet-consumed_target  (unweighted full target diet,  g/cap/day)
      alpha = diet-adherence        (share_diet_adherence,         -)
      P     = demography            (population by sex x age,       inhabitants)

    DALYs D_d(y) are projected inside the module (frozen-rate demographic
    projection, see _project_dalys) from the static 2023 GBD table and the model's
    live demography P.

    Monetization:
      MF    = CDM_MF['health-diet'] (value per DALY, CHF/DALY)  ->  cost = DALYs * MF

    FLAG - where alpha is applied: here it scales the COMBINED (disease-level) PIF,
    which is the standard "a fraction alpha of the population fully complies"
    interpretation and reproduces R's par_total at every alpha. If the R script
    instead scales each food's PIF *before* the multiplicative combination, use the
    line marked "ALT" below instead (the two differ because the combination is
    non-linear).
    """
    dm_B = DM_diet["diet-consumed_bau"].copy()  # reference (BAU) diet
    dm_T = DM_diet["diet-consumed_target"].copy()  # full target diet
    dm_alpha = DM_diet["diet-adherence"].copy()  # population adherence share

    # Pre-processing
    dm_data_paf = DM_TCAF_health_diet[
        "health-diet_paf"
    ]  # dict: food -> PAF dose-response curve
    # DALYs are projected from the static 2023 GBD table x the model's demography
    dm_data_dalys = _project_dalys(
        DM_TCAF_health_diet["health-diet_dalys"], DM_diet["demography"], base_year=2025
    )

    # Step 0 - Groupby categories relevant for health ----------------------------
    # Red meat = bovine + pig + sheep + other animal
    pattern = "pro-liv-meat-bovine|pro-liv-meat-pig|pro-liv-meat-sheep|pro-liv-meat-oth-animal"
    dm_B.groupby(
        {"pro-liv-meat-red": pattern}, dim="Categories1", inplace=True, regex=True
    )
    dm_T.groupby(
        {"pro-liv-meat-red": pattern}, dim="Categories1", inplace=True, regex=True
    )

    # Health categories (food risk factors) that we consider
    cat_health = [
        "crop-fruit",
        "crop-pulse",
        "pro-liv-abp-dairy-milk",
        "crop-oilcrop",
        "pro-liv-meat-processed",
        "pro-liv-meat-red",
        "crop-veg",
        "crop-cereal-whole",
    ]

    dm_B.filter({"Categories1": cat_health}, inplace=True)
    dm_T.filter({"Categories1": cat_health}, inplace=True)

    for cat in cat_health:
        if cat not in dm_B.col_labels["Categories1"]:
            print(f"Warning: {cat} not in diet-consumed_bau")
        if cat not in dm_data_paf:
            print(f"Warning: {cat} not in dm_data_paf")

    # Step 1 - Common years across intakes, adherence and (projected) DALYs ------
    years = sorted(
        set(dm_B.col_labels["Years"])
        & set(dm_data_dalys.col_labels["Years"])
        & set(dm_alpha.col_labels["Years"])
    )
    dm_B.filter({"Years": years}, inplace=True)
    dm_T.filter({"Years": years}, inplace=True)
    dm_alpha.filter({"Years": years}, inplace=True)

    country = dm_B.col_labels["Country"]
    dalys_disease = dm_data_dalys.col_labels["Categories1"]  # disease code order
    n_c, n_y, n_f, n_d = len(country), len(years), len(cat_health), len(dalys_disease)
    paf_var_prefix = "tcaf_health-diet_paf_"

    # Step 2 - Read the PAF off the dose-response curve at each intake -----------
    # numpy.interp does linear interpolation with flat extrapolation outside the
    # grid, reproducing R's approx(x, y, xout, rule = 2).
    def eval_paf(dm_intake):
        out = np.zeros((n_c, n_y, n_f, n_d))
        for fi, cat in enumerate(cat_health):
            dm_curve = dm_data_paf[cat]
            x_grid = np.array(
                dm_curve.col_labels["Years"], dtype=float
            )  # intake grid [g/day/cap]
            var_names = [paf_var_prefix + d for d in dalys_disease]
            y_grid = np.stack(
                [dm_curve[:, :, v][0, :] for v in var_names], axis=-1
            )  # (n_intake, n_disease)
            order = np.argsort(x_grid)
            x_s, y_s = x_grid[order], y_grid[order, :]
            xv = dm_intake[:, :, "lfs_consumers-diet", cat]  # (n_c, n_y)
            for di in range(n_d):
                out[:, :, fi, di] = np.interp(xv, x_s, y_s[:, di])
        return out

    arr_paf_B = eval_paf(dm_B)  # PAF at reference intake
    arr_paf_T = eval_paf(dm_T)  # PAF at full target intake

    # Step 3 - Full-adoption PIF per food x disease ------------------------------
    # No floor at 0: a food moving in the harmful direction yields a negative PIF
    # (added burden), so any diet - including worsening ones - can be evaluated.
    with np.errstate(divide="ignore", invalid="ignore"):
        arr_pif = (arr_paf_B - arr_paf_T) / (1.0 - arr_paf_T)
    arr_pif = np.nan_to_num(arr_pif, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 4 - Combine the food risk-factors of a disease: 1 - prod(1 - p) -------
    af_ref_comb = 1.0 - np.prod(1.0 - arr_paf_B, axis=2)  # (n_c, n_y, n_d)
    pif_comb = 1.0 - np.prod(1.0 - arr_pif, axis=2)  # (n_c, n_y, n_d)  full adoption

    # Step 5 - Apply population adherence: alpha * PIF ---------------------------
    alpha = dm_alpha[:, :, "share_diet_adherence"]  # (n_c, n_y)
    pif_alpha = alpha[:, :, np.newaxis] * pif_comb  # alpha on the COMBINED PIF
    # ALT (scale each food's PIF before combining):
    # pif_alpha = 1.0 - np.prod(1.0 - alpha[:, :, np.newaxis, np.newaxis] * arr_pif, axis=2)

    # Step 6 - Attributable / avoided / residual DALYs ---------------------------
    dm_dalys = dm_data_dalys.copy()
    dm_dalys.filter({"Years": years}, inplace=True)
    arr_dalys = dm_dalys[:, :, "tcaf_health-diet_dalys", :]  # (n_c, n_y, n_d)

    arr_attr = af_ref_comb * arr_dalys
    arr_avoid = pif_alpha * arr_dalys
    arr_resid = arr_attr - arr_avoid

    # Detailed DM (per disease): attributable / avoided / residual
    dm_paf = dm_dalys.copy()
    dm_paf[:, :, "tcaf_health-diet_dalys", :] = arr_attr
    dm_paf.add(
        arr_avoid,
        dim="Variables",
        col_label="tcaf_health-diet_dalys-avoided",
        unit="DALYs/y",
    )
    dm_paf.add(
        arr_resid,
        dim="Variables",
        col_label="tcaf_health-diet_dalys-residual",
        unit="DALYs/y",
    )

    # Step 7 - Total across diseases = sum_d ------------------------------------
    dm_dalys_tot = dm_paf.copy()
    dm_dalys_tot.groupby({"total": ".*"}, dim="Categories1", inplace=True, regex=True)

    # Step 8 - Monetization: cost [CHF] = DALYs [DALYs/y] * MF [CHF/DALY] --------
    # CDM_MF['health-diet'] holds a single health-diet monetization factor
    # (monetary value per DALY). Each DALYs component (attributable / avoided /
    # residual) is monetized, both per disease and for the total.
    cdm_mf = CDM_MF["health-diet"]
    mf_var = cdm_mf.col_labels["Variables"][0]
    mf = cdm_mf[mf_var]  # scalar CHF/DALY
    cost_of = {
        "tcaf_health-diet_dalys": "tcaf_health-diet_cost",
        "tcaf_health-diet_dalys-avoided": "tcaf_health-diet_cost-avoided",
        "tcaf_health-diet_dalys-residual": "tcaf_health-diet_cost-residual",
    }
    for dm in (dm_paf, dm_dalys_tot):
        for dalys_var, cost_var in cost_of.items():
            dm.add(
                dm[:, :, dalys_var, :] * mf,
                dim="Variables",
                col_label=cost_var,
                unit="CHF",
            )

    return dm_paf, dm_dalys_tot


# CalculationLeaf TCAF BIODIVERSITY


def TCAF_biodiversity_workflow(DM_TCAF_biodiversity, DM_landuse_to_TCAF):
    DM_TCAF_biodiversity = DM_TCAF_biodiversity.copy()

    # Step Biodiversity Switzerland
    # Drop treenut cat because not in cropland
    DM_TCAF_biodiversity["biodiversity-ch"].drop(dim="Categories2", col_label="treenut")
    # Add mean value for starch missing in biodiv
    dm_temp = DM_TCAF_biodiversity["biodiversity-ch"].groupby(
        {"starch": ".*"},
        dim="Categories2",
        aggregation="mean",
        regex=True,
        inplace=False,
    )
    DM_TCAF_biodiversity["biodiversity-ch"].append(dm_temp, dim="Categories2")
    # Append cropland to biodiversity for relevant geoscale
    # land-use names the crops 'crop-cereal', the biodiversity data 'cereal'
    dm_cropland_ch = DM_landuse_to_TCAF["cropland-ch"].copy()
    dm_cropland_ch.rename_col_regex(str1="crop-", str2="", dim="Categories2")
    DM_TCAF_biodiversity["biodiversity-ch"].append(dm_cropland_ch, dim="Variables")

    # Biodiversity costs [CHF/ha] = cropland [ha] * eco-costs [CHF/ha]
    DM_TCAF_biodiversity["biodiversity-ch"].operation(
        "agr_cropland",
        "*",
        "eco-cost",
        dim="Variables",
        out_col="tcaf_biodiversity",
        unit="CHF",
    )

    # Step Biodiversity World
    # Drop Switzerland and differing countries if any
    DM_landuse_to_TCAF["cropland-world"].drop(dim="Country", col_label="Switzerland")
    set_countries = set(
        DM_TCAF_biodiversity["biodiversity-world"].col_labels["Country"]
    ) - set(DM_landuse_to_TCAF["cropland-world"].col_labels["Country"])
    DM_TCAF_biodiversity["biodiversity-world"].drop(
        dim="Country", col_label=list(set_countries)
    )
    # Sort countries
    DM_TCAF_biodiversity["biodiversity-world"].sort(dim="Country")
    DM_landuse_to_TCAF["cropland-world"].sort(dim="Country")

    # Sum total cropland
    DM_landuse_to_TCAF["cropland-world"].groupby(
        {"total": ".*"}, dim="Categories1", aggregation="sum", regex=True, inplace=True
    )
    DM_landuse_to_TCAF["cropland-world"] = DM_landuse_to_TCAF[
        "cropland-world"
    ].flatten()
    DM_landuse_to_TCAF["cropland-world"].rename_col_regex(
        str1="agr_cropland_total_total", str2="agr_cropland", dim="Variables"
    )

    # Append cropland to biodiversity for relevant geoscale
    DM_TCAF_biodiversity["biodiversity-world"].append(
        DM_landuse_to_TCAF["cropland-world"], dim="Variables"
    )

    # Biodiversity costs [EUR2024/ha] = cropland [ha] * eco-costs [EUR2024/ha]
    DM_TCAF_biodiversity["biodiversity-world"].operation(
        "agr_cropland",
        "*",
        "eco-cost",
        dim="Variables",
        out_col="tcaf_biodiversity",
        unit="EUR2024",
    )

    return DM_TCAF_biodiversity


# CalculationLeaf TPE INTERFACE
def TCAF_TPE_interface(
    dm_health_diet_detailed, dm_health_diet_tot, DM_TCAF_lca, fish=None
):
    # attributable / avoided / residual DALYs, and their monetized costs [CHF]
    vars_out = [
        "tcaf_health-diet_dalys",
        "tcaf_health-diet_dalys-avoided",
        "tcaf_health-diet_dalys-residual",
        "tcaf_health-diet_cost",
        "tcaf_health-diet_cost-avoided",
        "tcaf_health-diet_cost-residual",
    ]

    # health-diet detailed (per disease)
    dm_health_diet_detailed.filter({"Variables": vars_out}, inplace=True)
    dm_tpe = dm_health_diet_detailed.flattest()

    # health-diet total (summed over diseases)
    dm_health_diet_tot.filter({"Variables": vars_out}, inplace=True)
    dm_tpe.append(dm_health_diet_tot.flattest(), dim="Variables")

    # LCA monetized results (Switzerland) [CHF], summed over method (intensive/organic)
    # FIXME add lca world
    dm_lca_ch = DM_TCAF_lca["lca-switzerland"].filter({"Variables": ["agr_production-tcaf"]})
    dm_lca_ch.rename_col("agr_production-tcaf", "tcaf_lca_cost", dim="Variables")
    dm_lca_ch.group_all("Categories2", inplace=True)  # sum over method -> Categories1 food, Categories2 impact

    # Swiss domestic fish, same structure: sum over method, then add as a food
    if fish is not None:
        dm_fish = fish["cost"].filter({"Variables": ["agr_production-tcaf"]})
        dm_fish.rename_col("agr_production-tcaf", "tcaf_lca_cost", dim="Variables")
        dm_fish.group_all("Categories2", inplace=True)  # sum over method
        years = sorted(
            set(dm_fish.col_labels["Years"]) & set(dm_lca_ch.col_labels["Years"])
        )
        dm_fish.filter({"Years": years}, inplace=True)
        dm_lca_ch.filter({"Years": years}, inplace=True)
        dm_lca_ch.append(dm_fish, dim="Categories1")
        dm_lca_ch.sort("Categories1")

    # lca per food category (summed over impact categories)
    dm_lca_ch_food = dm_lca_ch.copy()
    dm_lca_ch_food.group_all("Categories2", inplace=True)  # sum over impact -> Categories1 food
    dm_tpe.append(dm_lca_ch_food.flattest(), dim="Variables")

    # lca per impact category (summed over food categories)
    dm_lca_ch_imp = dm_lca_ch.copy()
    dm_lca_ch_imp.group_all("Categories1", inplace=True)  # sum over food -> Categories1 impact
    dm_tpe.append(dm_lca_ch_imp.flattest(), dim="Variables")

    # lca total (summed over food and impact categories)
    dm_lca_ch_tot = dm_lca_ch_imp.copy()
    dm_lca_ch_tot.group_all("Categories1", inplace=True)  # sum over impact -> no categories left
    dm_lca_ch_tot.rename_col("tcaf_lca_cost", "tcaf_lca_cost_total", dim="Variables")
    dm_tpe.append(dm_lca_ch_tot, dim="Variables")

    # Swiss domestic fish detail (production, imports, SSR, cost, GHG)
    if fish is not None:
        dm_tpe.append(TCAF_fish_TPE_interface(fish), dim="Variables")

    # total true cost (LCA + health, at the moment: biodiversity is currently disabled) [CHF]
    # Uses cost-residual (attributable minus avoided), the net health burden that
    # actually remains under this scenario's adherence and target diet.
    # tcaf_health-diet_cost_total is the constant attributable baseline (computed
    # from the BAU reference diet alone) and does not vary with adherence or diet
    # choice, so it would make true-cost identical across every scenario.
    dm_tpe.operation(
        "tcaf_lca_cost_total",
        "+",
        "tcaf_health-diet_cost-residual_total",
        dim="Variables",
        out_col="tcaf_true-cost_total",
        unit="CHF",
    )

    return dm_tpe


def TCAF(lever_setting, years_setting, DM_input, interface=Interface()):
    years_ots = create_years_list(
        years_setting[0], years_setting[1], 1
    )  # make list with years from 1990 to 2015
    years_fts = create_years_list(years_setting[2], years_setting[3], years_setting[4])
    years_all = years_ots + years_fts

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    DM_TCAF_lca, DM_TCAF_health_diet, DM_TCAF_biodiversity, CDM_MF, CDM_const = (
        read_data(DM_input, lever_setting, years_all)
    )
    country_list = ["Switzerland"]

    # INTERFACES IN ---------------------------------------------------------------------------------------------------

    # CalculationLeaf Link interface or Simulate data from other modules
    # dietary-habits
    if interface.has_link(from_sector="dietary-habits", to_sector="TCAF"):
        DM_diet = interface.get_link(from_sector="dietary-habits", to_sector="TCAF")
    else:
        if len(interface.list_link()) != 0:
            print("You are missing dietary-habits to TCAF interface")
        DM_diet = simulate_diet_to_TCAF_input()
        for key in DM_diet.keys():
            DM_diet[key].filter({"Country": country_list}, inplace=True)

    # land-use
    if interface.has_link(from_sector="land-use", to_sector="TCAF"):
        DM_landuse_to_TCAF = interface.get_link(
            from_sector="land-use", to_sector="TCAF"
        )
    else:
        if len(interface.list_link()) != 0:
            print("You are missing land-use to TCAF interface")
        DM_landuse_to_TCAF = simulate_landuse_to_TCAF_input()

    # crop
    if interface.has_link(from_sector="crop", to_sector="TCAF"):
        DM_crop_to_TCAF = interface.get_link(from_sector="crop", to_sector="TCAF")
    else:
        if len(interface.list_link()) != 0:
            print("You are missing crop to TCAF interface")
        DM_crop_to_TCAF = simulate_crop_to_TCAF_input()

    # livestock
    if interface.has_link(from_sector="livestock", to_sector="TCAF"):
        DM_livestock_to_TCAF = interface.get_link(
            from_sector="livestock", to_sector="TCAF"
        )
    else:
        if len(interface.list_link()) != 0:
            print("You are missing livestock to TCAF interface")
        DM_livestock_to_TCAF = simulate_livestock_to_TCAF_input()

    # CalculationTree ---------------------------------------------------------------------------------------------------
    # Observed Swiss agricultural GHG inventory (from preprocessing 'cal_lca_ghg').
    # Popped out of the LCA dict so the workflow keeps only its lca-* matrices.
    # If absent (older pickle), calibration is skipped and the raw total is shown.
    dm_cal_ghg_lca = DM_TCAF_lca.pop("cal-ghg", None)
    cdm_ghg_ef_perhead = DM_TCAF_lca.pop("ghg-ef-perhead", None)
    cdm_lsu_per_head = DM_TCAF_lca.pop("lsu-per-head", None)
    DM_fish = DM_TCAF_lca.pop("fish", None)
    DM_TCAF_lca, dm_ghg_lca_ch = TCAF_lca_workflow(
        DM_TCAF_lca,
        DM_crop_to_TCAF,
        DM_landuse_to_TCAF,
        DM_livestock_to_TCAF,
        CDM_const,
        CDM_MF,
        years_setting=years_setting,
        dm_cal_ghg_lca=dm_cal_ghg_lca,
        cdm_ghg_ef_perhead=cdm_ghg_ef_perhead,
        cdm_lsu_per_head=cdm_lsu_per_head,
    )
    # Debug test, only prints. It needs lcia_animal_production_recipe.csv, which
    # is not in the repo, so it is not called in the model run.
    # TCAF_ghg_calibration_weight_test()

    # Swiss domestic fish. Needs the fish inputs in the pickle and the food demand
    # from dietary-habits; without either, fish is left out and this says so.
    fish = None
    if DM_fish is not None and "food-demand" in DM_diet:
        fish = TCAF_fish_lca_workflow(DM_fish, DM_diet["food-demand"], CDM_const, CDM_MF)
    else:
        print("TCAF: fish LCA skipped (fish inputs or diet food-demand missing)")

    dm_health_diet_detailed, dm_health_diet_tot = TCAF_health_diet_workflow(
        DM_diet, DM_TCAF_health_diet, CDM_MF
    )
    """DM_TCAF_biodiversity = TCAF_biodiversity_workflow(
        DM_TCAF_biodiversity, DM_landuse_to_TCAF
    )"""
    # CalculationTree TPE OUTPUT -------------------------------------------------------------------------------------------------------
    results_run = TCAF_TPE_interface(
        dm_health_diet_detailed, dm_health_diet_tot, DM_TCAF_lca, fish
    )

    # INTERFACES OUT ---------------------------------------------------------------------------------------------------

    # interface to Land use
    # DM_lus = agriculture_landuse_interface(DM_bioenergy, dm_lgn, dm_land_use)
    # interface.add_link(from_sector='agriculture', to_sector='land-use',
    #                   dm=DM_lus)

    return results_run


def TCAF_module_local_run():
    country_list = ["Switzerland"]
    DM_input = filter_country_and_load_data_from_pickles(
        country_list=country_list, modules_list="TCAF", filter_country=False
    )
    years_setting, lever_setting = init_years_lever()
    TCAF(lever_setting, years_setting, DM_input["TCAF"])
    return


# CalculationLeaf GHG CALIBRATION - WEIGHT SOURCE TEST -------------------------
# Test: does the Swiss livestock GHG calibration change when the per-head animal
# weights come from the DB (coproduct-aware extraction of the LCIA summaries) vs
# the hard-coded Table-5 values? The per-head EF is per-kg EF x Weight / life, so
# only the Weight factor differs between the two runs; population, lsu/head,
# lifetimes, per-kg EFs and the FOEN inventory targets are held identical. Any
# difference in the calibration is therefore attributable purely to the weight.
#
# Run:  python TCAF_module.py --weight-test
# (needs a TCAF.pickle built by TCAF_preprocessing, the livestock/crop/land-use
#  interface pickles, and data/data_pool/lcia_animal_production_recipe.csv.)

# Process selectors mirroring TCAF_preprocessing.TABLE5 (one row per food x
# method) and the Table-5 reference weights [kg/head]. Kept here so the test is
# self-contained and does not import TCAF_preprocessing (which runs a full
# preprocessing pass at import time).
_T5_SELECTORS = {
    "abp-dairy-milk": {
        "intensive": "cull cow, conventional, lowland milk system, silage maize 5 to 10%, at farm gate",
        "organic": "cull cow, organic, lowland milk system, silage maize 5 to 10%, at farm gate",
    },
    "meat-bovine": {
        "intensive": "beef cattle, conventional, national average, at farm gate",
        "organic": "beef cattle, organic, national average, at farm gate",
    },
    "meat-bovine-young": {
        "intensive": "calf, 13 days old, conventional, lowland milk system, silage maize 5 to 10%, at farm gate",
        "organic": "calf, 13 days old, organic, lowland milk system, silage maize 5 to 10%, at farm gate",
    },
    "meat-pig": {
        "intensive": "pig, conventional, national average, at farm gate",
        "organic": "pig, organic, national average, at farm gate",
    },
    "meat-sheep": {
        "intensive": "lamb, conventional, indoor production system, at farm gate",
        "organic": "lamb, organic, system number 1, at farm gate",
    },
    "meat-poultry": {
        "intensive": "broiler, conventional, at farm gate",
        "organic": "broiler, organic, at farm gate",
    },
    "abp-hens-egg": {
        "intensive": "cull hen, conventional, national average, at farm gate",
        "organic": "cull hen, organic, at farm gate",
    },
    "meat-oth-animal": {
        "intensive": "kid goat, conventional, intensive forage area, at farm gate",
        "organic": None,
    },
}
_T5_WEIGHT = {
    "abp-dairy-milk": 670.05,
    "meat-bovine": 650.0,
    "meat-bovine-young": 79.56,
    "meat-pig": 115.4,
    "meat-sheep": 35.0,
    "meat-poultry": 2.04,
    "abp-hens-egg": 1.9,
    "meat-oth-animal": 9.0,
}

# Coproduct-aware weight extraction (same logic as load_data): pick the coproduct
# matching the animal the process is about, most specific first.
_COPRODUCT_RE = re.compile(
    r"([A-Za-z][A-Za-z \-]+?)\s*\(\s*weight\s*([0-9][0-9 ,\.]*?)\s*kg\)", re.IGNORECASE
)
_MAIN_ANIMAL = [
    (
        r"cull cow|dairy cow",
        [r"^cull cow$", r"cull dairy cow", r"dairy cow", r"cull cow"],
    ),
    (r"\bcalf\b", [r"calf for fattening", r"weaned calf", r"^calf", r"calf"]),
    (r"\bpig\b|\bsow\b|pork", [r"fattened pig", r"^pig$", r"pig"]),
    (r"broiler|chicken", [r"^chicken$", r"chicken", r"broiler"]),
    (r"cull hen|laying|\bhen\b", [r"cull hen", r"hen"]),
    (r"lamb|\bewe\b|sheep|mutton", [r"sold lamb", r"^lamb$", r"lamb"]),
    (r"kid goat|\bgoat\b", [r"sold kid goat", r"^kid goat$", r"kid goat"]),
    (r"salmon|trout|\bfish\b", [r"salmon", r"trout", r"fish"]),
]


def _parse_coproducts(summary):
    if not isinstance(summary, str):
        return []
    m = re.search(r"(?i)coproducts?\s*:(.*)", summary)
    if not m:
        return []
    seg = m.group(1).split("\n")[0]
    out = []
    for mm in _COPRODUCT_RE.finditer(seg):
        name = mm.group(1).strip().lower()
        raw = mm.group(2).replace(" ", "").replace(",", ".")
        if raw.count(".") > 1:
            head, _, tail = raw.rpartition(".")
            raw = head.replace(".", "") + "." + tail
        try:
            w = float(raw)
        except ValueError:
            continue
        if 0 < w < 5000:
            out.append((name, w))
    return out


def _main_weight(process, summary):
    pairs = _parse_coproducts(summary)
    if not pairs:
        return None
    p = str(process).lower()
    for proc_pat, tiers in _MAIN_ANIMAL:
        if re.search(proc_pat, p):
            for tier in tiers:
                cands = [w for n, w in pairs if re.search(tier, n)]
                if cands:
                    return round(sum(cands) / len(cands), 2)
            return None
    return round(pairs[0][1], 2) if len(pairs) == 1 else None


def _coproduct_db_weights(csv_path):
    """Return {food: {'intensive': w|None, 'organic': w|None}} from the LCIA CSV,
    using the Table-5 process selectors and coproduct-aware extraction."""
    df = pd.read_csv(csv_path)
    u = df.drop_duplicates(subset=["Process", "Database", "Category"]).copy()
    proc_lower = u["Process"].astype(str).str.lower()
    out = {}
    for food, meth_sel in _T5_SELECTORS.items():
        out[food] = {}
        for meth, sel in meth_sel.items():
            if sel is None:
                out[food][meth] = None
                continue
            hit = u[proc_lower.str.startswith(sel)]
            if len(hit) == 0:  # loosen to a contains-match
                hit = u[proc_lower.str.contains(re.escape(sel[:32]), regex=True)]
            out[food][meth] = (
                _main_weight(hit.iloc[0]["Process"], hit.iloc[0]["Summary"])
                if len(hit)
                else None
            )
    return out


def _scale_perhead_ef(cdm_ef, ratios):
    """Return a copy of the per-head EF with array[:, food, method] *= ratio.
    ratios: {food: {'intensive': r, 'organic': r}}. Weight scales every impact
    variable equally, so the whole Variables axis is multiplied."""
    out = cdm_ef.copy()
    foods = out.col_labels["Categories1"]
    meths = out.col_labels["Categories2"]
    ix = out.idx
    for food in foods:
        if food not in ratios:
            continue
        for meth in meths:
            r = ratios[food].get(meth, 1.0)
            if r and r != 1.0:
                out.array[:, ix[food], ix[meth]] = out.array[:, ix[food], ix[meth]] * r
    return out


def _fmt(x):
    return f"{x:g}" if isinstance(x, (int, float)) else "n/a"


def _print_calibration_comparison(a, b):
    print("\n" + "#" * 78)
    print("# WEIGHT-SOURCE CALIBRATION COMPARISON  (Table-5 vs DB weights)")
    print("#" * 78)
    s1a, s1b = a.get("stage1"), b.get("stage1")
    if not s1a or not s1b:
        print("stage-1 diagnostics unavailable (no 'global-warming' impact category).")
        return
    yrs = a["years"]
    pa, pb = s1a.get("pop_rate"), s1b.get("pop_rate")
    raw_a, raw_b = s1a.get("pop_asf_raw"), s1b.get("pop_asf_raw")
    path = "population"
    if pa is None or pb is None:  # population path unavailable -> production
        pa, pb = s1a.get("prod_rate"), s1b.get("prod_rate")
        raw_a, raw_b = s1a.get("prod_asf_raw"), s1b.get("prod_asf_raw")
        path = "production (population path unavailable)"
    obs = s1a.get("observed_3A3B", {})
    print(
        f"\nStage 1 - livestock ASF GHG vs FOEN inventory 3A+3B  [{path} path, t CO2-eq]"
    )
    print(
        f"{'year':>6} {'observed':>12} {'rawT5':>12} {'rawDB':>12} "
        f"{'rateT5':>8} {'rateDB':>8} {'dRaw%':>8}"
    )
    for y in yrs:
        if not pa or pa.get(y) is None:
            continue
        o = obs.get(y, float("nan"))
        if o != o:  # only print calibration years (obs present)
            continue
        rt, rd = raw_a.get(y), raw_b.get(y)
        draw = (rd / rt - 1) * 100 if (rt and rd) else float("nan")
        print(
            f"{y:>6} {o:>12,.0f} {rt:>12,.0f} {rd:>12,.0f} "
            f"{pa.get(y):>8.3f} {pb.get(y):>8.3f} {draw:>+8.2f}"
        )

    def _mad(p):
        v = [abs(p[y] - 1.0) for y in yrs if p and p.get(y) is not None]
        return float(np.mean(v)) if v else float("nan")

    mt, md = _mad(pa), _mad(pb)
    print(f"\nmean |rate-1|:   Table-5 = {mt:.4f}    DB = {md:.4f}")
    print(
        f"closer to inventory (rate=1): "
        f"{'DB weights' if md < mt else 'Table-5 weights'}"
    )

    s2a, s2b = a.get("stage2"), b.get("stage2")
    if s2a and s2b:
        ta = s2a.get("pop_total_rate") or s2a.get("prod_total_rate")
        tb = s2b.get("pop_total_rate") or s2b.get("prod_total_rate")
        if ta and tb:
            va = np.nanmean([ta[y] for y in yrs if ta.get(y) is not None])
            vb = np.nanmean([tb[y] for y in yrs if tb.get(y) is not None])
            print(
                f"\nStage 2 - grand total vs cat-3: mean rate  Table-5 = {va:.4f}   DB = {vb:.4f}"
            )
    print("#" * 78)


def resolve_lcia_csv(
    explicit=None, filename="lcia_animal_production_recipe.csv", start=None, max_up=6
):
    """Locate the LCIA animal-production CSV without hard-coding the layout.

    Order: (1) an explicit path if given; (2) known relative sub-paths checked at
    the module dir, cwd and each of their ancestors; (3) a bounded os.walk of the
    top ancestor, preferring a copy under 'data_pool' (what the pickle was built
    from). Raises FileNotFoundError listing where it looked."""
    if explicit:
        p = os.path.abspath(explicit)
        if os.path.exists(p):
            return p
        raise FileNotFoundError(f"csv_path does not exist: {p}")

    try:
        module_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:  # __file__ undefined in notebooks / REPL
        module_dir = os.getcwd()
    starts = [s for s in (start, module_dir, os.getcwd()) if s]

    rel_candidates = [
        os.path.join("data", "data_pool", filename),
        os.path.join("pre_processing", "TCAF", "data", "data_pool", filename),
        os.path.join("model", "pre_processing", "TCAF", "data", "data_pool", filename),
        os.path.join(
            "backend", "model", "pre_processing", "TCAF", "data", "data_pool", filename
        ),
        filename,
    ]
    tried = []
    for base in starts:  # ancestors x known subpaths
        d = base
        for _ in range(max_up + 1):
            for rel in rel_candidates:
                cand = os.path.normpath(os.path.join(d, rel))
                tried.append(cand)
                if os.path.exists(cand):
                    return cand
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent

    # Fallback: walk the nearest repo-root ancestor (never the filesystem root).
    markers = (".git", "pyproject.toml", "setup.py", "requirements.txt", "backend")
    repo_root = None
    for base in starts:
        d = base
        for _ in range(max_up + 1):
            if any(os.path.exists(os.path.join(d, m)) for m in markers):
                repo_root = d
                break
            parent = os.path.dirname(d)
            if parent == d:  # hit filesystem root -> stop
                break
            d = parent
        if repo_root is not None:
            break
    skip = {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".idea",
        ".mypy_cache",
    }
    if repo_root is not None:
        matches = []
        for dirpath, dirnames, filenames in os.walk(repo_root):
            dirnames[:] = [dn for dn in dirnames if dn not in skip]
            if filename in filenames:
                matches.append(os.path.join(dirpath, filename))
        if matches:  # prefer the data_pool copy, then shortest path
            matches.sort(key=lambda p: (0 if "data_pool" in p else 1, len(p)))
            return matches[0]

    raise FileNotFoundError(
        f"Could not locate {filename}. Looked at:\n  "
        + "\n  ".join(dict.fromkeys(tried[:10]))
        + (
            f"\n  ...and os.walk('{repo_root}')"
            if repo_root
            else "\n  (no repo-root marker found to anchor a search)"
        )
        + "\nPass csv_path='/abs/path/to/lcia_animal_production_recipe.csv' explicitly."
    )


def TCAF_ghg_calibration_weight_test(
    csv_path=None, years_setting=None, lever_setting=None
):
    """Run the Swiss livestock GHG calibration twice - Table-5 weights vs
    coproduct-aware DB weights - and print the two calibrations side by side.

    Only cdm_ghg_ef_perhead (the per-head EF) differs between runs; it is rebuilt
    for the DB run by rescaling the pickle's Table-5 per-head EF by W_db / W_table5
    per food x method (exact, since per-head EF is linear in Weight and lifetime is
    unchanged). Returns (diag_table5, diag_db)."""
    country_list = ["Switzerland"]
    csv_path = resolve_lcia_csv(csv_path)
    print(f"[weight-test] using CSV: {csv_path}")
    ys, ls = init_years_lever()
    years_setting = years_setting or ys
    lever_setting = lever_setting or ls
    years_ots = create_years_list(years_setting[0], years_setting[1], 1)
    years_fts = create_years_list(years_setting[2], years_setting[3], years_setting[4])
    years_all = years_ots + years_fts

    DM_input = filter_country_and_load_data_from_pickles(
        country_list=country_list, modules_list="TCAF", filter_country=False
    )["TCAF"]
    DM_TCAF_lca, _, _, CDM_MF, CDM_const = read_data(DM_input, lever_setting, years_all)

    # Interfaces (simulate fallback, mirrors TCAF()).
    DM_landuse = simulate_landuse_to_TCAF_input()
    DM_crop = simulate_crop_to_TCAF_input()
    DM_livestock = simulate_livestock_to_TCAF_input()

    dm_cal = DM_TCAF_lca.pop("cal-ghg", None)
    cdm_ef_t5 = DM_TCAF_lca.pop("ghg-ef-perhead", None)
    cdm_lsu = DM_TCAF_lca.pop("lsu-per-head", None)
    if cdm_ef_t5 is None or cdm_lsu is None:
        raise RuntimeError(
            "pickle lacks 'liv_ghg_ef_perhead' / 'liv_lsu_per_head'; "
            "rebuild TCAF.pickle with TCAF_preprocessing first."
        )

    # DB-weight variant of the per-head EF.
    db_w = _coproduct_db_weights(csv_path)
    ratios = {}
    for food in _T5_WEIGHT:
        ratios[food] = {}
        for meth in ("intensive", "organic"):
            w = db_w.get(food, {}).get(meth)
            ratios[food][meth] = (w / _T5_WEIGHT[food]) if w else 1.0
    cdm_ef_db = _scale_perhead_ef(cdm_ef_t5, ratios)

    print("\nWeight source - DB (coproduct-aware) vs Table-5 [kg/head] and ratio:")
    print(f"  {'food':20} {'T5':>8} {'DB int':>8} {'x':>6} {'DB org':>8} {'x':>6}")
    for food in _T5_WEIGHT:
        wi = db_w.get(food, {}).get("intensive")
        wo = db_w.get(food, {}).get("organic")
        print(
            f"  {food:20} {_T5_WEIGHT[food]:>8} {_fmt(wi):>8} "
            f"{ratios[food]['intensive']:>6.3f} {_fmt(wo):>8} {ratios[food]['organic']:>6.3f}"
        )

    def _run(cdm_ef, tag):
        print("\n" + "=" * 78 + f"\n[RUN] {tag}\n" + "=" * 78)
        lca = {k: v.copy() for k, v in DM_TCAF_lca.items()}  # deep-copy mutated inputs
        crop = DM_crop.copy()
        landuse = {k: v.copy() for k, v in DM_landuse.items()}
        livestock = {k: v.copy() for k, v in DM_livestock.items()}
        _, _, diag = TCAF_lca_workflow(
            lca,
            crop,
            landuse,
            livestock,
            CDM_const,
            CDM_MF,
            years_setting=years_setting,
            dm_cal_ghg_lca=(dm_cal.copy() if dm_cal is not None else None),
            cdm_ghg_ef_perhead=cdm_ef,
            cdm_lsu_per_head=cdm_lsu,
            return_diagnostics=True,
        )
        return diag

    diag_t5 = _run(cdm_ef_t5, "Table-5 weights")
    diag_db = _run(cdm_ef_db, "DB (coproduct-aware) weights")
    _print_calibration_comparison(diag_t5, diag_db)
    return diag_t5, diag_db


if __name__ == "__main__":
    TCAF_module_local_run()
