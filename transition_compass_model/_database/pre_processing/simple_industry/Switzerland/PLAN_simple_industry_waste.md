# Plan: simple_industry + simple_waste pickles (cantonal energy by fuel × branch, emissions ≥ PRTR)

## Context
`simple_industry_preprocessing_CH.py` is a copy of `power/Switzerland/industry_to_energy_interface.py`.
It downscales the FSO energy accounts to cantons using employee FTEs, splits by end use with a
residential-heating proxy, and stops without saving anything. Goals:
- `simple_industry.pickle`: energy demand by canton × industry branch × fuel, plus emissions, for all 26 cantons + CH.
- `simple_waste.pickle`: waste incineration (KVA) energy, emissions and electricity/heat output.
- Cantons sum to official Swiss totals (GEST); key-sector emissions per canton ≥ SwissPRTR.
- Regenerate `industry_to_energy.pickle` from the same data so EnergyScope stays consistent.
- The emissions module is out of scope (not used).

## Weaknesses of the current method (to fix)
1. **Manufacturing is one bucket (NACE 10-33).** FTE downscaling gives a pharma FTE the same energy as a cement FTE.
   Cantons with heavy industry are underestimated; office-heavy cantons are overestimated.
2. **Oil products:** only "1.1.2 extra-light heating oil" is kept, so heavy fuel oil, petcoke and LPG are dropped
   (GEST 2022: 11.6 PJ oil, of which 8.8 XL).
3. **Totals ≠ GEST.** The energy accounts follow economic sectors (NACE), cover 2000–2022, and are about 1.6 TWh
   short on electricity vs GEST T17b (2022). 1990–1999 and 2023 are only gap-filled.
4. **Waste:** NACE 36-39 waste and biomass are set to 0, and most KVA activity sits in NACE 35 anyway
   (3.7 TWh vs 12.8 TWh in GEST T27).
5. **FTE shares are linearly extrapolated** outside 2011–2022 (`linear_fitting` on normalised shares), which can drift.
6. **End-use split** uses residential building counts (not energy) × residential efficiencies, applied to industry.
   Process heat is then a residual clipped at 0 (`fix_negative_values`), which adds energy that isn't in the data.
7. **Category matching bug:** `re.findall(...)[0]` on "68-75, 77-82" drops 77-82, and substring matching
   (`cat_num in c`) is fragile.
8. **Forecast** is linear TWh/capita × population, with no link to industrial activity.
9. **No emissions:** process emissions are missing, and cement is ~60% process CO2, so it can never reach PRTR from energy alone.

## Decisions (agreed)
- **Branches:** the 13 EP2050+ branches: food, textiles, paper, chem-pharma, glass-ceramics, cement-lime, iron-steel,
  non-ferrous metals, metal products, machinery, other, water-waste (NACE E own use), construction.
- **National totals:** each fuel scaled to GEST T17b (industry, 1990–2023).
- **KVA:** goes in a separate `simple_waste` script and pickle. It owns all KVA energy and CO2, with fossil and biogenic CO2 separate.
  - KVA electricity and heat output are exported; wiring them into the energy module is a follow-up.
  - EnergyScope can only burn WASTE via CHP or boilers that need a heat sink.
- **PRTR:** add process emissions per branch, site key branches by PRTR facilities, and allocate the rest by FTE.
  Hard check: canton × branch ≥ PRTR.
- **Energy interface:** regenerate `industry_to_energy.pickle` (key `ind-energy-demand`, same layout).
- **Forecast:** levers 1–4. Level 1 = EP2050+ WWB, level 4 = ZERO Basis, levels 2–3 linearly interpolated,
  all scaled to 2023 values.

## Data sources
| Data | Source | Access |
|---|---|---|
| Industry energy by fuel 1990–2023 | GEST T17b (`power/Switzerland/data/statistique_globale_suisse_energie.xlsx`) | on disk |
| KVA waste input, fossil input, heat/elec produced/sold | GEST T27 (same file) | on disk |
| Energy by branch group × fuel (19 groups, sub-statistic of GEST) | BFE "Energieverbrauch in der Industrie und im Dienstleistungssektor" ([BFE Teilstatistiken](https://www.bfe.admin.ch/bfe/de/home/versorgung/statistik-und-geodaten/energiestatistiken/teilstatistiken.html)) | new download (xlsx); **verify tables exist; fallback below** |
| Facility CO2 (≥100 kt threshold), canton, activity | SwissPRTR Excel ([BAFU](https://www.bafu.admin.ch/bafu/en/home/topics/chemicals/swissprtr-pollutant-register.html)), data to 2024 | new download; **verify CO2 incl./excl. biogenic** |
| Process emissions (IPPU 2A/2B/2C), KVA fossil CO2 ("thereof waste incineration") | FOEN inventory `industry/Switzerland/data/emissions/Evolution_GHG_since_1990_2026-04.xlsx` | on disk |
| End-use split, branch GHG trajectories (02-01..02-05), production index | EP2050+ industry workbook (WWB + ZERO Basis sheets) | on disk |
| FTE per NACE 2-digit × canton | STATENT px-x-0602010000_101 | **STAT-TAB API** (cached pickle exists) |
| KVA plant list + canton (for plants below the PRTR threshold) | `power/Switzerland/data/waste_power.xlsx` | on disk |
| Population | `load_pop` | repo |

**Stat-tab API calls (flagged):**
- `px-x-0602010000_101` (FTE): kept.
- `px-x-0204000000_106` (energy accounts): only as a fallback if the BFE branch tables are unusable.
- `px-x-0902010000_102` (residential heating technologies): removed.

Every remaining `get_data_api_CH` call gets a `# STAT-TAB API` comment and prints a warning on cache miss.
API results stay cached in pickles.

## Files
- **First step:** copy this plan to `_database/pre_processing/simple_industry/Switzerland/PLAN_simple_industry_waste.md`
  so it can be re-read next to the code.
- **Rewrite** `_database/pre_processing/simple_industry/Switzerland/simple_industry_preprocessing_CH.py`
  (drop the end-use/residential-proxy functions and the hardcoded canton lists).
- **New** `_database/pre_processing/simple_industry/Switzerland/get_data_prtr.py`: SwissPRTR loader
  (facility, canton, NOGA/activity, year, CO2), shared with waste.
- **New** `_database/pre_processing/simple_waste/Switzerland/simple_waste_preprocessing_CH.py`.
- New raw downloads go in `simple_industry/Switzerland/data/` and `simple_waste/Switzerland/data/`.
  Existing caches in `power/Switzerland/data/` are reused read-only.
- **Outputs:**
  - `_database/data/datamatrix/simple_industry.pickle`
  - `_database/data/datamatrix/simple_waste.pickle`
  - `_database/data/interface/industry_to_energy.pickle` (overwritten; back up the old one in the scratchpad for the diff)
- **Reuse:**
  - `rename_cantons` + `constants.CANTONS_RENAME_TARGET` (canton names, e.g. "Zurich")
  - `get_data_api_CH`, `load_pop`, `create_years_list`, `linear_fitting`, `DataMatrix`
  - GEST reader `buildings/Switzerland/get_data_functions/energy_demand_for_calibration.extract_energy_statistics_data`
  - `clean_df_EP2050` (extend to read any table block / sheet)
- Not touched: `lever_position.json`, `interactions.py`, the energy module, the emissions module.
  No model module reads the new pickles yet.

## simple_industry method
1. **National energy, branch × fuel, 1990–2023.**
   - Map the BFE branch groups onto the 13 branches, then do biproportional (RAS) scaling:
     branch totals from BFE, fuel totals from GEST T17b for each year.
   - Before the first BFE year, keep the branch × fuel shares fixed and scale to GEST.
   - NACE E (water-waste) = own final use only; no KVA waste input.
   - Fuels follow the interface names: electricity, gas, heating-oil (all petroleum products), coal, wood,
     district-heating, waste (industrial waste fuels), renewables (solar/biogas/ambient).
   - *Fallback without BFE:* energy accounts for B/C/E/F (all oil products, fixing weakness 2), with the split
     inside manufacturing from EP2050+ Tab 02-03 combustion GHG for fossil fuels and FTE for electricity. Then scale to GEST.
2. **Emissions (national).**
   - Combustion = energy × emission factors: BAFU CH factors already used in buildings (coal 350, heating-oil 265,
     gas 201 kt CO2/TWh), and IPCC factors from `industry.pickle` `constant/emission-factor` for waste, CH4 and N2O.
   - Biogenic CO2 (wood, biogas, biogenic share of waste fuels) goes in its own variable.
   - Process = FOEN IPPU rows mapped to branches: 2A1/2A2 → cement-lime, 2A3/2A4 → glass-ceramics,
     2B → chem-pharma, 2C1 → iron-steel, 2C3+ → NE metals. F-gas and product use (2D–2G) are excluded and reported.
   - Report national combustion CO2 vs FOEN 1A2 per year (ratio printed; not forced).
3. **Cantonal allocation per branch and year.**
   - *Key branches* (cement-lime, chem-pharma, iron-steel, NE metals, paper, glass-ceramics):
     - Canton emissions = PRTR facility CO2 in that canton, plus the national remainder spread by FTE.
     - Fossil-fuel energy follows the emission shares; electricity and non-emitting fuels follow FTE.
   - *Other branches:* FTE shares, with STATENT 2-digit NACE mapped to branches.
     - C23 and C24 can't be split at 3–4 digit level, so cement and steel are sited through PRTR and the
       remainder uses C23 / C24 FTE.
     - FTE outside 2011–2022 is held constant at the edge year (no extrapolation of shares).
     - PRTR shares are held before its first year.
   - **Hard check:** canton × branch CO2 ≥ PRTR. If the national branch total is below the PRTR sum, the script
     stops and prints the offending facilities. That's the point to discuss. Known suspects:
     - the Cressier refinery (energy sector, not industry in GEST/EP2050)
     - biogenic CO2 in PRTR
     - facilities whose PRTR activity maps to a different branch
4. **Forecast (levers 1–4).**
   - Per branch, ratios to 2023 from EP2050+:
     - electricity: Tab 02-02 (industry total)
     - total energy: Tab 02-01
     - fossil fuels: combustion GHG, Tab 02-03 per branch
     - process emissions: Tab 02-04 per branch
   - Non-electric non-fossil fuels take the remainder.
   - Level 1 uses the WWB sheet, level 4 the ZERO Basis sheet, levels 2–3 interpolated.
   - Applied to 2023 canton values (canton shares frozen at 2023).
   - **Check:** whether the ZERO GHG tables are net of CCS (the "Carbon Capture" row in 02-01); flag it.
5. **Pickle structure.**
   - `{'ots': {'simple-industry': {'energy': dm, 'emissions': dm}}, 'fts': {'simple-industry': {1..4: {...}}}, 'constant': {'emission-factors': cdm}}`
   - energy: `ind_energy-demand[TWh]` × branch × fuel
   - emissions: `ind_emissions-combustion`, `ind_emissions-process`, `ind_emissions-biogenic` [Mt] × branch × gas (CO2/CH4/N2O)
   - Years 1990–2023, then 2025–2050 every 5 years. Countries: 26 cantons + Switzerland.
6. **Energy interface** (`industry_to_energy.pickle`).
   - Electricity split by end use with EP2050+ Tab 02-02 shares: Antriebe + Klima + Sonstige → `elec`;
     Beleuchtung → `lighting`; Prozesswärme → `process-heat`; Raumwärme → `space-heating`; Warmwasser → `hot-water`.
   - Other fuels split with the non-electric shares (02-01 minus 02-02).
   - Same Categories2 list as today; missing fuels (`heat-pump`, `nuclear-fuel`, `other`, `solar`, `biomass`, `biogas`) set to 0.
   - fts uses level 1 (WWB) until a module passes the lever live.

## simple_waste method
- History 1990–2023 from GEST T27: waste input, fossil input, heat and electricity produced, used and sold.
- Fossil CO2 calibrated to the FOEN line "thereof waste incineration plant" (implied fossil emission factor per year).
  Biogenic CO2 = biogenic energy share × biomass emission factor. CH4 and N2O from IPCC (`solid-waste`).
- Canton shares:
  - SwissPRTR KVA facility CO2 (yearly).
  - Plants below the PRTR threshold: capacity from `waste_power.xlsx`.
  - Check: canton ≥ PRTR (comparing like with like on biogenic CO2).
- Forecast levers:
  - Level 1 = waste incinerated per capita held at 2019–2023 average × population.
  - Level 4 = reduced per-capita waste (circular-economy target; **parameter to set with the user**,
    default −20% by 2050). Levels 2–3 interpolated.
  - Fossil share held constant. No CCS in this version (flagged).
- Electricity and heat output = waste × T27 conversion efficiencies (2019–2023 average). Stored for the later energy wiring.
- Pickle: same structure, variables `wst_waste-incinerated`, `wst_electricity-production`, `wst_heat-production` [TWh],
  `wst_emissions-fossil`, `wst_emissions-biogenic` [Mt] × gas.

## Verification
Run:
- `uv run python -m transition_compass_model._database.pre_processing.simple_industry.Switzerland.simple_industry_preprocessing_CH`
- the same for `simple_waste`

Built-in asserts and printed reports:
- Σ cantons = Switzerland for every variable, year and level; tolerance 1e-6.
- CH energy by fuel = GEST T17b (industry) and T27 (KVA), 1990–2023.
- Canton × key branch CO2 ≥ PRTR, all PRTR years. Print a table: canton, branch, model, PRTR, margin.
- National CO2 vs FOEN: 1A2 (combustion), IPPU 2A–2C (process), KVA fossil. Ratios printed per year.
- No negatives or NaNs; fts continuous at 2023→2025.
- Diff the new vs old `industry_to_energy.pickle`: canton electricity totals; national electricity vs GEST.
- Run `uv run python -m transition_compass_model.model.interactions_localrun` (energy sector) to confirm
  EnergyScope still solves with the regenerated interface.

## Follow-ups (not in this change)
- Wire KVA electricity into the energy module (subtract from the demand EnergyScope must cover) and drop `Waste`
  from its CO2 factors (`model/energy/workflows.py:868-881`).
- District-heating module (KVA heat).
- Model modules `simple_industry` / `simple_waste`, plus `lever_*` entries in `lever_position.json`
  (this also affects speed-to-zero).
- CCS at cement plants and KVA.

---

## Implementation notes (2026-11-02)

**Files**
- `simple_industry/Switzerland/get_data_functions.py`: readers for GEST, FOEN, EP2050+, BFE, SwissPRTR
  (no stat-tab calls). Replaces the planned `get_data_prtr.py`.
- `simple_industry/Switzerland/simple_industry_preprocessing_CH.py`: rewritten.
- `simple_waste/Switzerland/simple_waste_preprocessing_CH.py`: new.
- Raw downloads in `simple_industry/Switzerland/data/{bfe,prtr}/`. These are gitignored (`*.xlsx`, `*.csv`) and
  re-downloaded automatically if missing:
  - BFE: https://pubdb.bfe.admin.ch/de/publication/download/8788
  - PRTR: https://www.bafu.admin.ch/dam/en/sd-web/CXgeZKcf8ez8/swissprtr-daten-2007-2024.xlsx

**Deviations from the plan**
- **Branches:** the BFE statistics have 12 industry branches (EP2050+ split, no water/waste).
  `water-waste` (NACE 36-39 own use) comes from the FSO energy accounts (stat-tab, cached).
- **GEST T17b industry includes NACE 36-39 own use.** Evidence: energy accounts B+C+E+F ≈ GEST per fuel.
  So water-waste is subtracted from GEST before the BFE split. Exception: sewage-plant biogas
  (0.33 TWh) is outside T17b and is added on top.
- **No RAS needed:** GEST only constrains fuel totals, so each fuel is scaled on its own (BFE branch shares × GEST).
- **Pre-2013:** BFE branch shares are frozen at 2013. For fossil fuels they are shifted with the EP2050+
  combustion-GHG trend per branch.
- **Interface fts** = level 1 (WWB).

**Results (2023)**
- Cantons sum to GEST T17b for every fuel and year (error < 1e-14 TWh).
- Canton × branch CO2 ≥ PRTR: 405/405 (2007–2023). Waste: 252/252.
- Combustion fossil CO2 of the 12 branches: 3.53 Mt vs FOEN 1A2 3.82 Mt (−8%). 1990: 6.50 vs 6.54.
- Process (EP2050+ WWB, CO2eq): 2.66 Mt vs FOEN IPPU CO2 1.93 Mt. The difference is N2O/PFC in CO2eq and
  scenario values after 2017.
- KVA: fossil CO2 = FOEN (2.13 Mt), biogenic 2.43 Mt. Total 4.56 Mt vs PRTR 3.87 Mt (21 of 29 plants).
- **industry_to_energy:** CH electricity 2023 = 16.25 TWh (old 16.99). 2050 = 15.5 TWh (old 7.6, which was a
  per-capita extrapolation). EnergyScope (standalone, Vaud) solves.

**Open points / weaknesses to discuss**
1. **Cement:** model 3.15 Mt (0.77 fossil combustion + 1.91 process + 0.47 biogenic) vs PRTR 2.09 Mt.
   ≥ PRTR holds, but process CO2 from EP2050+ after 2017 is a scenario value. Calibrating to FOEN 2A1
   (NIR/CRF tables) would tighten it.
2. **BFE → GEST scaling factors (2023):** electricity 1.25, gas 1.19, wood 2.5, oil 1.6.
   The extra wood goes mostly to `other-industries` (sawmills, C16), in proportion to BFE wood use.
3. **Oil products other than extra-light** (0.82 TWh: LPG, petcoke, ...) are spread like heavy fuel oil /
   extra-light heating oil, with the heavy-fuel-oil emission factor.
4. **Industrial waste fuel:** 30% biogenic (assumption).
5. **Electricity, district heat and renewables** are allocated by FTE, even in PRTR branches.
   Only combustion fuels follow PRTR siting.
6. **ZERO Basis** nets CCS in its combustion and process GHG. Fuel ratios are clipped at 0, and negative
   emissions are not modelled.
7. **Waste:** level 4 = −20% waste per capita by 2050 (placeholder; to agree). No CCS at KVA.
8. **KVA CO2 is still counted in the energy module** (`model/energy/workflows.py:868-881`). Follow-up.
9. **Full model runs:** when `industry` is in `sectors`, energy uses the industry module's link,
   not `industry_to_energy.pickle`.
