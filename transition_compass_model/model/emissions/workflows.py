# =============================================================================
# Emissions accounting scope (territorial scope 1, Switzerland)
# =============================================================================
# All sectors use territorial scope 1 except where noted.
#
# Transport:       residency-based scope 1 (Swiss residents' fuel combustion,
#                  regardless of where the trip occurs). Territorial deficit vs
#                  SFOE 1A3 is expected and intentional by design.
# Buildings:       residential + services fuel combustion, scope 1 only.
#                  Electricity CO2 excluded (attributed to energy module).
#                  District heating uses a blended EF proxy (see buildings
#                  workflows comment); causes slight overcount vs SFOE 1A4.
# Industry:        territorial scope 1. Covers combustion (energy × EF) and
#                  process emissions (material production × emission-factor-
#                  process: cement calcination, steel reduction, glass, lime,
#                  Al, Cu, chem). Feedstock carbon correctly excluded here
#                  (non-energy use, embedded in products). Model covers ~74.8%
#                  of SFOE energy (S2 textiles, S8 Al primary, S10 tra-equip
#                  structurally missing). Construction (SFOE 1A2 sub-set) and
#                  HFC/PFC/SF6 from electronics (SFOE cat 2) also not modeled.
# Agriculture:     territorial scope 1 (enteric fermentation CH4, manure N2O,
#                  soil N2O). Slight overcount vs SFOE cat 3 (~5 %).
# Ammonia:         territorial scope 1.
# Energy:          territorial scope 1 — Swiss electricity generation CO2 only
#                  (fossil power plants + KVA waste incinerators; OTS via
#                  per-technology EF, FTS via EnergyScope GWP_op[WASTE/NG_CCS]
#                  + per-tech EF for GasCC/Waste). DH boiler CO2 not here
#                  (partially proxied in buildings; see DH module comment).
#                  Imported electricity CO2 excluded (territorial principle).
# Aviation:        residency-based MTMC round-trip demand; far above SFOE 1A3a
#                  (domestic only). Intentional — documents Swiss resident
#                  footprint, not territorial inventory value.
#
# SFOE GHG inventory comparison (2019, MtCO2eq):
#   Electricity (model 2.42 vs SFOE 1A1 3.46): gap = DH gas/oil boilers in 1A1
#   Buildings   (model 13.28 vs SFOE 1A4 11.83): ~1.5 overcount, DH proxy + residency
#   Industry    (model 5.06 vs SFOE 1A2+cat2 8.98): 3.9 gap = coverage + HFC + construction
#   Transport   (model 13.79 vs SFOE 1A3-avia 14.77): ~1.0 residency deficit expected
#   Aviation    (model 12.36 vs SFOE 1A3a 0.12): intentional boundary difference
#   Agriculture (model 6.40 vs SFOE cat3 6.08): good alignment
#   Model total 53.3 vs SFOE 46.6; excl aviation model 40.9 vs SFOE 46.5;
#   excl aviation+IPPU-gap model 40.9 vs SFOE ~42.3 → ~3 % below (well within uncertainty).
# =============================================================================


def put_together_emissions(DM_emi):
    dm_emi = DM_emi["transport"].copy()
    modules = ["buildings", "industry", "agriculture", "ammonia"]
    for m in modules:
        dm_emi.append(DM_emi[m].copy(), "Variables")
    if DM_emi.get("energy") is not None:
        dm_emi.append(DM_emi["energy"].copy(), "Variables")

    # for captured / negative emissions, make sure that they are negative
    for cat in ["industry-captured-emissions", "ammonia-captured-emissions"]:
        arr_temp = dm_emi[:, :, cat, :].copy()
        arr_temp[arr_temp > 0] = -arr_temp[arr_temp > 0]
        dm_emi[:, :, cat, :] = arr_temp.copy()

    # deepen
    for v in dm_emi.col_labels["Variables"]:
        dm_emi.rename_col(v, "emissions_" + v, "Variables")
    dm_emi.deepen(based_on="Variables")
    dm_emi.switch_categories_order("Categories1", "Categories2")

    return dm_emi


def make_co2_equivalent(dm_emi):
    dm_out = dm_emi.copy()

    GWP_N2O = 265
    GWP_CH4 = 28

    dm_out[..., "N2O"] = dm_out[..., "N2O"] * GWP_N2O
    dm_out[..., "CH4"] = dm_out[..., "CH4"] * GWP_CH4

    dm_out.group_all("Categories2")

    dm_out.change_unit(old_unit="Mt", new_unit="MtCO2eq", factor=1, var="emissions")

    return dm_out
