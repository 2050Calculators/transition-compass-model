# Buildings pre-processing for Switzerland: historical data

A guide to the historical part of `buildings_preprocessing_main_CH.py`: what it builds,
where the data comes from, and how each step works. It assumes you already know the
`DataMatrix` class.

This version covers the historical years only (1990–2023). The script also builds future
scenarios (2025–2050), which are not described here.

---

## What the script does

The buildings module needs a lot of input data: how much floor area exists, how fast it is
renovated, which heating systems are used, how efficient they are, how much electricity
appliances use, and more. It needs this for every region it runs (Switzerland and the
cantons).

`buildings_preprocessing_main_CH.py` builds all of that from public statistics and writes it
to one file:

```
transition_compass_model/_database/data/datamatrix/buildings.pickle
```

The main script itself is short. It calls a series of **pipelines** (`processors/`) that
build the historical data (1990–2023). The data extraction and cleaning helpers used by the
pipelines live in `get_data_functions/`.

```
buildings_preprocessing_main_CH.py
├── processors/          historical pipelines, one per topic, plus build_ots_pickle.py
├── scenarios/           future values (not covered here)
├── get_data_functions/  download, cache and clean each source
└── data/                cached raw files and intermediate pickles
```

---

## Before you run it

**Run it as a script**, so that its own folder is on the Python path (the imports
`from processors...` and `from get_data_functions...` rely on this):

```bash
uv run python transition_compass_model/_database/pre_processing/buildings/Switzerland/buildings_preprocessing_main_CH.py
```

**Settings** come from `_database/pre_processing/params.py`:

| Setting | Value | Meaning |
|---|---|---|
| `country_list` | Switzerland and the 26 cantons | Regions written to the pickle |
| `years_ots` | 1990–2023, yearly | Historical years |

Most pipelines compute all 26 cantons internally anyway (several steps need the full set
to compute shares), then keep only `country_list` at the end.

**Other pickles must already exist and be up to date:**

- `data/datamatrix/lifestyles.pickle`: population, used almost everywhere (through `load_pop`).
- `data/datamatrix/climate.pickle`: heating and cooling degree days, copied into the climate
  interface.
- `data/datamatrix/buildings.pickle`: the existing file is updated, not replaced (see below).

**Downloads are cached.** Every source is read from a local file in `data/` if it exists.
It is downloaded (from Swiss Stats, STAT-TAB, the SFOE website…) only when the file is
missing. To refresh a source, delete its cached file. Some sources are not downloaded at all
and must already be in `data/`: the Wüest & Partner file, the Programme Bâtiments reports and
the RegBL building register.

**The pickle is merged, not overwritten.** Every write goes through `my_pickle_dump`. For
each DataMatrix, it replaces the countries present in the new data and keeps any other
country already in the file. The pickle is written several times during the run, so a crash
halfway leaves a partly updated file.

---

## Words used in this guide

**Building types (`Categories1`).** Residential buildings are split into
**single-family households (SFH)** and **multi-family households (MFH)**. Non-residential
buildings (services) are split into **education, health, hotels, offices, trade and
other**.

**Envelope classes (`Categories2`).** Every building type is also split into five
energy-performance classes, from **B** (best) to **F** (worst). A/B and F/G are merged. The
historical data doesn't record classes directly, so the class is inferred from the
**construction period**. The mapping lives in `get_data_functions/construction_period_param.py`
and is different for the two residential types:

| Class | SFH built | MFH built |
|---|---|---|
| F | before 1971 | before 1981 |
| E | 1971–1980 | 1981–1990 |
| D | 1981–2000 | 1991–2000 |
| C | 2001–2010 | 2001–2010 |
| B | 2011 onwards | 2011 onwards |

For buildings built since 1990, the same file assigns new floor area by year of
construction: D for 1990–2000, C for 2001–2010, B for 2011–2023.

Non-residential buildings use the MFH periods too, since most services are housed in
MFH-type buildings. Their U-values stay non-residential (see `u-value` below).

**Heating technologies.** coal, district-heating, electricity (direct resistive heating),
gas, heating-oil, heat-pump, other-tech, solar, wood.

**ots / fxa / constant.** These are the top-level keys of the pickle that hold historical
data:
- `ots`: historical values (1990–2023) of variables that can change in the scenarios.
- `fxa`: variables that are the same in every scenario.
- `constant`: values with no country or year dimension (a `ConstantDataMatrix`).

---

## The big picture

The historical part runs in two phases.

```
Phase 1: historical pipelines (processors/)
   interfaces ─► population
   floor area ─► renovation ─► space-heating technology
   fixed assumptions, appliances, hot water, lighting, services
                                │
Phase 2: assemble the historical pickle (build_ots_pickle.py)
                                   writes ots, fxa, constant
```

Order matters in two places. Renovation needs the floor-area flows. Space heating needs the
floor area by class that comes out of renovation.

---

## Phase 1: historical pipelines

### 1. Interfaces and population

`processors/buildings_interfaces_CH.py`

**What it answers:** which inputs from other modules the buildings module will receive.

It reads population from `lifestyles.pickle` and degree days from `climate.pickle`. It
writes them to `data/interface/lifestyles_to_buildings.pickle` and
`data/interface/climate_to_buildings.pickle`, so the buildings module can run on its own in
local tests. It returns the population, which the main script uses later.

---

### 2. Floor area: how much residential space exists, and how it changes

`processors/floor_area_pipeline_CH.py` (helpers in `get_data_functions/floor_area_CH.py`)

**What it answers:** for each canton and year from 1990 to 2023, how many m² of SFH and MFH
housing exist, how much is built, and how much is demolished. It also splits these by
envelope class.

**Building the stock.** The starting point is the Swiss Stats building register (SDMX
dataflow `CH1.GWS / DF_GWS_REG7`). It gives floor area and number of buildings by canton,
building type and construction period. Each construction period is mapped to an envelope
class (see the class table in *Words used in this guide*), and summing over classes gives
`stock tot`. The class split is only trusted for **2023**, so every other year of
`stock cat` is left as NaN. Years the register doesn't cover are filled by keeping floor
area per person on its trend.

**Making it agree with known totals.** The register isn't the reference total, so the stock
is rescaled twice:
1. The Swiss total is replaced by the EP2050+ national residential floor area (*Private
   Haushalte*, Tabelle 01-03, holiday homes included). Years EP2050+ doesn't cover are
   filled with the register values, multiplied by a calibration factor. Each canton then
   keeps its share of the Swiss total.
2. Each canton is scaled so its 2023 value matches the Wüest & Partner energy reference area
   (`WP_EBF_cantons_2023.xlsx`). The same factor applies to every year, so the trend doesn't
   change. Wüest & Partner's "other residential" buildings are split between SFH and MFH
   in proportion to their size.

The 2023 class shares are then applied to the rescaled stock.

**Flows.** New floor area starts as *new dwellings × average floor area per building*
(`new raw`). New dwellings come from two STAT-TAB tables (1995–2012 and 2013–2023). Gaps are
filled by extrapolating new dwellings per person (fitted on 1995–2002 for the early years
and 2015–2022 for the recent ones).

The average floor area comes from the register (floor area ÷ number of dwellings, by
construction period), so it is in register units, not in the units of the rescaled stock.
To put both in the same units, `new raw` is multiplied by one factor per canton and type:

```
f = rescaled stock (2023) / register stock (2023)
```

The register measures dwelling floor space and the rescaled stock is energy reference
area, so f is mostly a difference in definition. It is about 0.8–1.05 for MFH and
1.3–1.7 for SFH, and it changes by only about ±2 % between 1990 and 2023. That is why
a single 2023 value is used: it keeps the year-to-year shape of the FSO series. The
factors are printed during the run.

Demolition (`waste`) would follow from the stock balance:

```
w(t) = s(t-1) − s(t) + n(t)
```

That series is noisy, because the register changed method around 2008. A flat rate is used
instead: the 2015–2023 average, never below 0.2 % per year. New floor area is then
recomputed from the same balance, so stock, new and waste always add up.

> ⚠️ The 0.2 % floor is a proxy, not measured demolition data.

**Class of the flows.** All demolished floor area is assigned to class F (buildings are
assumed to last about 50 years, so no D building is demolished before 2031). New floor area
goes to D, C or B depending on the year it was built.

**Output:** a dict, all values in m², filtered to `country_list`:

| Key | Variable | Categories |
|---|---|---|
| `stock tot` | `bld_floor-area_stock` | SFH/MFH |
| `stock cat` | `bld_floor-area_stock` | SFH/MFH × B–F, **2023 only** |
| `new cat` | `bld_floor-area_new` | SFH/MFH × B–F |
| `waste cat` | `bld_floor-area_waste` | SFH/MFH × B–F (all in F) |
| `new raw` | `bld_floor-area_new` | SFH/MFH, rescaled by f, before balancing |

---

### 3. Renovation, and the stock by class for every year

`processors/renovation_pipeline_CH.py` (helpers in `get_data_functions/renovation_CH.py`)

**What it answers:** how much floor area is renovated each year, which classes it leaves and
enters, and what the stock in each class looked like from 1990 to 2023.

**Renovation rate.** The number of renovated SFH and MFH per canton comes from the annual
reports of the *Programme Bâtiments* (Excel files in `data/programme_batiment/`, 2017–2022).
Dividing by the number of buildings (Swiss Stats `DF_GWS_REG1`) gives a renovation rate.
Years before 2017 take the 2017 value, and later missing years take the last known value.
The programme only counts renovations that received a subsidy, so this rate is a lower
bound.

**Where renovated floor area goes.** A renovation moves floor area from one class to a
better one. The pipeline assumes renovated floor area **leaves** class F (80 %) and E (20 %),
and **enters** better classes depending on the period:

| Period | E | D | C | B |
|---|---|---|---|---|
| 1990–2000 | 85 % | 15 % | – | – |
| 2001–2010 | 69 % | 16 % | 15 % | – |
| 2011–2023 | 46 % | 23 % | 15 % | 16 % |

The 2011–2023 split follows the Programme Bâtiments statistics on how many classes a
renovation gains (+1 class 57 %, +2 15 %, +3 15 %, +4 13 %). The older splits are
assumptions, since classes A/B didn't exist yet.

The renovated floor area per class is then:

```
r_c(t) = (in_c − out_c) × renovation-rate(t−1) × stock(t−1)
```

**New floor area by class.** For classes that only exist after 1990, the new-dwellings series
(`new raw`) is scaled so that *everything built or renovated into the class since 1990*
equals its 2023 stock. Each class is empty before its construction period starts. SFH class D
and MFH class E also contain older buildings, so they use the scaling factor of class C.
Per canton, type and class c:

```
k_c     = (s_c(2023) − Σ r_c(t)) / Σ n_raw(t)    sums over 1991–2023, n_raw only in c's period
n_c(t)  = k_c × n_raw(t)                         in c's period, 0 outside
```

This follows from the backward balance with no demolition in c and s_c(1990) = 0. 1990
is left out because its flows only affect 1989.
The factors are printed during the run. Because `new raw` is already in the units of
the stock, they should be close to 1 (about 0.85–1.05). A factor below zero (more renovated
into the class than its 2023 stock) is set to 0, with a message.

**Stock by class for every year.** Starting from the 2023 split, the stock is rebuilt
backwards one year at a time:

```
s_c(t−1) = s_c(t) − n_c(t) − r_c(t) + w_c(t)
```

Any negative stock is corrected by shifting classes B–E up and F down, without changing the
flows. Finally, class F is set to *total stock − the other classes*, so the classes always
add up to `stock tot`. F therefore also absorbs any mismatch, such as buildings that
changed type.

**Output:**

| Key | Content |
|---|---|
| `floor-area-cat` | `bld_floor-area_stock`, `_new`, `_renovated`, `_waste` [m²], SFH/MFH × B–F, every year |
| `renovation-rate` | `bld_renovation-rate` [%], SFH/MFH |
| `renovation-redistribution` | `bld_renovation-redistribution-in` / `-out`, by class B–F |

---

### 4. Space heating: which technology heats which building, and how efficiently

`processors/heating_technology_pipeline_CH.py` (helpers in
`get_data_functions/heating_technology_CH.py`)

**What it answers:** the share of each heating technology for every building type and
class, and the efficiency of each technology.

**Technology mix.** Two sources count buildings by canton, type, construction period and
main heating energy:
- Swiss Stats `DF_GWS_REG4`, for 2021–2023.
- STAT-TAB `px-x-0902020100_112`, for 1990 and 2000.

For classes F, E and D, which existed in 1990, the three points in time are joined and
interpolated. Classes C and B are set to zero in the year before their construction period
starts (2000 and 2010) and interpolated up to the 2021 data.

**Calibration to EP2050+.** The national technology mix of EP2050+ (*Private Haushalte*,
Tabelle 02-01) is multiplied by the Swiss floor area from step 3. This gives m² heated by
each technology for SFH and MFH. That national total is then distributed across cantons and
classes in proportion to their building counts, from 2000 onwards. Before 2000, the
correction is extrapolated. Coal is phased out to zero by 2021. The result is normalised over
technologies, so it becomes a share.

**Non-residential buildings** have no data by technology, so all six types copy the MFH mix.
Apartment blocks are closer to service buildings than detached houses are, especially for
district heating and gas.

**Efficiency.** Efficiencies come from EP2050+ (Tabelle 01-08, space-heating efficiencies in
the ZERO Basis scenario). Electricity is set to 1, and solar, other-tech and coal to 0.6.
Heat-pump COPs are capped by class, based on Döring & Richter (2024) and the Swiss building
archetypes of Pongelli et al. (2023): F 2.3, E 2.5, D 3.0, C 3.3, B 3.6. Otherwise the
efficiency doesn't depend on the class. Switzerland's efficiencies are copied to every
canton.

**Output:**

| Key | Content |
|---|---|
| `heating-tech-split` | `bld_heating-mix` [share], building type (8) × class B–F × technology (9) |
| `efficiency` | `bld_heating-efficiency`, class B–F × technology |

---

### 5. Fixed assumptions

`processors/others_pipeline_CH.py`

**What it answers:** physical and behavioural parameters that come from the literature
rather than from time series.

| Output key | What | Values and source |
|---|---|---|
| `u-value` | Average envelope U-value [W/m²K], type × class | Residential: Pongelli et al. (2023), e.g. SFH F 0.82 → B 0.25. Non-residential: fitted to EP2050+ (see below) |
| `surface-to-floor` | Envelope surface per m² floor area | SFH 2.0, MFH 1.3 (Pongelli et al.). Non-residential 1.1–1.4 (Delmastro et al. 2016, Fleiter et al. 2017) |
| `Tint-heat` | Indoor heating temperature [°C] | MFH: F 19 °C to B 23 °C (better buildings are kept warmer). SFH: 1 °C lower. Non-residential: 20 °C (SIA 380/1) |
| `age` | Age of each class [years] | Year minus the first construction year of the class |
| `emission-factors` | CO₂ per fuel [kt/TWh] | OFEV: coal 350, heating oil 265, gas 201, wood and solar 0 |
| `emission-factor-heating_district` | CO₂ of district heating [kt/TWh] | 73.55 until 2019, 66 in 2023 |

Emissions from electricity (resistive heating, heat pumps, appliances) are not counted in
the buildings module. They are computed in the energy module.

**Non-residential U-values.** All six services types share one set, by class:

| Class | F | E | D | C | B |
|---|---|---|---|---|---|
| U-value [W/m²K] | 1.20 | 1.03 | 0.86 | 0.39 | 0.34 |

These are effective values, worked back from EP2050+ *Dienstleistung*, Tabelle 03-01, which
gives useful space heat per m². Useful heat is the heat the building loses, before boiler
or heat-pump losses. The model computes it as U-value × envelope area per m² × degree days,
so it is linear in the U-value and each class can be fitted with one factor. EP2050+ uses
the same floor area as the model. The fit is done by `scripts/fit_services_u_values_CH.py`
(repo root), which prints the values and writes nothing:

- C (built 2001–2010) matches EP2050+ "Neubau" (new buildings), averaged over 2001–2010
  (about 44 kWh/m²).
- B (built from 2011) matches "Neubau" averaged over 2011–2015 (about 37 kWh/m²).
- D, E and F keep the ratios of the former EU Hotmaps proxy (2.01 / 1.74 / 1.44). They are
  scaled together so that the services stock ("Baujahr < 2020") matches EP2050+ over
  2010–2015, as a ratio of sums (total heat ÷ total floor area over the six years).
  EP2050+ reacts only weakly to the weather, so a six-year average of cold and warm years
  is used rather than a single year.

The former Hotmaps values were 1.6–2× too high: services needed 1.5–1.9× the EP2050+ heat
per m². Being effective values, the fitted ones also absorb the internal gains (heat from
people, lighting, IT) that the model doesn't subtract. To change them, re-run the fit
script and copy its output into `nonres_u`.

---

### 6. Appliances and other electricity use

`processors/appliances_pipeline_CH.py` (helpers in `get_data_functions/appliances_CH.py`)

**What it answers:** how many large appliances each household owns, how long they last, and
how much electricity each one uses.

The SFOE open dataset on appliance sales and consumption (opendata.swiss, `ogd109`) gives
national stock, sales and electricity use for 11 appliances: freezer, refrigerator,
dishwasher, washing machine, oven and stove, tumble dryer, TV, PC, laptop, monitor and
set-top box. The oven-and-stove consumption is replaced by the EP2050+ electricity demand for
cooking (*Analyse nach Verwendungszweck*, Tabelle 17).

The retirement rate follows from the stock balance `w(t) = s(t−1) + n(t) − s(t)`. Stock and
consumption per unit are extrapolated over the full period using the number of households
(from an FSO table), and the stock is then rebuilt from the balance. Stock is expressed per
household. Every canton gets the Swiss values.

Electricity use that isn't covered by these appliances comes from EP2050+ (Tabelle 1). It is
expressed per person and is the same in every canton.

**Output** (all go to `fxa`):

| Key | Content |
|---|---|
| `appliances` | `bld_appliances_stock` [unit/household], `_retirement-rate`, `_electricity-demand` [kWh/unit] |
| `household` | number of households and household size |
| `other-electricity-demand` | [kWh/cap] |

---

### 7. Hot water

`processors/hot_water_pipeline_CH.py` (helpers in `get_data_functions/hot_water_CH.py`)

**What it answers:** how much useful energy per person goes into hot water, which
technologies provide it, and how efficient they are.

**National consumption.** EP2050+ (*Analyse nach Verwendungszweck*, Tabelle 14) gives Swiss
household hot-water consumption by fuel. Missing years are filled by extrapolating
consumption per person and the fuel shares separately.

**Split to cantons.** Each canton's hot-water technology mix comes from the building counts
(Swiss Stats `DF_GWS_REG4` for 2021–2023, STAT-TAB for 1990 and 2000), weighted by the
number of apartments per building type (`DF_GWS_REG7`). The national consumption of each fuel
is then shared out according to *population × technology share*.

**Efficiency.** Hot-water efficiencies are taken from the space-heating efficiencies of step
4. Heat pumps get a hot-water COP from EP2050+ (Abb. 17): 2.8 in 2015. Useful energy is
*consumption × efficiency*.

> ⚠️ The fuel shares from consumption and from building counts disagree (heating oil is
> lower, gas higher), probably because gas-heated buildings are larger on average. The
> consumption-based shares are used for the technology mix.

**Output:**

| Key | Content |
|---|---|
| `hw-energy-demand` | `bld_hw_demand` [MWh/cap], useful energy |
| `hw-tech-mix` | `bld_hw_tech-mix` [share] by technology, from useful energy |
| `hw-efficiency` | `bld_hot-water_efficiency` by technology |

---

### 8. Lighting

`processors/lighting_pipeline_CH.py`

The Swiss residential lighting consumption comes from EP2050+ (*Analyse nach
Verwendungszweck*, Tabelle 1, "Beleuchtung"), converted from PJ to TWh. Each canton gets
its population share. The output is a single DataMatrix with `bld_residential-lighting` [TWh].

---

### 9. Services (non-residential buildings)

`processors/services_pipeline_CH.py` (helpers in `get_data_functions/services_CH.py`)

**What it answers:** how much floor area non-residential buildings have, and how much
energy per m² they use for hot water, lighting and electricity.

**Energy by fuel, per canton.** FSO energy accounts (STAT-TAB `px-x-0204000000_106`) give
national energy use by economic sector and fuel. Gasoline and diesel (transport and machinery)
and energy-producing sectors are removed. "Renewables" is split into heat pumps and solar
using the household hot-water mix. National use is distributed to cantons in proportion to
employees per sector (STAT-TAB `px-x-0602010000_101`).

**Energy by end use.** The national split by end use comes from Prognos/Infras/TEP,
*Analyse des schweizerischen Energieverbrauchs 2000–2020*, Table 26: space heating, hot water,
process heat, lighting, HVAC, ICT, drives and processes, other. The values are hard-coded as
`PROGNOS_SERVICES_EUD_PJ` in `get_data_functions/services_CH.py`, also used by the
validation script. It is applied per canton, then adjusted so that each fuel matches the FSO
consumption. Useful energy is *consumption × efficiency*, using the hot-water efficiencies
(biogas = gas, biomass = wood, everything else 1).

**Intensity per m².** The total useful energy by end use is divided by the Swiss
non-residential floor area to give kWh/m². The same intensity is used for every canton,
because cantonal demand and cantonal floor area are both shared out by employees. Space
heating is dropped: the buildings module computes it from physics (floor area × U-value ×
degree days).

**Floor area.** EP2050+ *Dienstleistung* (sheet "04 EBF-Baualter") gives national
non-residential floor area by construction period (thousand m², from 2000). These periods
are mapped to the MFH classes:
- F = before 1946 + 1947–1975 + 5/15 of 1976–1990
- E = 10/15 of 1976–1990 (the band is split evenly per year)
- The 1991–2019 band is split by when the floor area was added: D is its 2000 stock, C what
  was added 2001–2010, B what was added 2011–2019 (assuming no demolition before 2020).

The floor area is then split into the six building types using the type shares of each
class in the RegBL building register (`data/Regbl/Switzerland/`), and distributed to
cantons by their share of service employees. EP2050+ starts in 2000. For 1990–1999 the
total follows the yearly services floor-area stock of Wüest & Partner (BFE 2004), without
agricultural buildings, scaled to EP2050+ in 2000. Class D (built 1991–2000) grows from 0
in 1990 to its EP2050+ value in 2000, following the timing of the floor area added in the
W&P series; classes E and F take the rest.

**Output:**

| Key | Content |
|---|---|
| `services_demand` | `bld_services_useful-energy` [kWh/m²] by end use |
| `services_tech-mix` | technology share by end use |
| `services_efficiencies` | efficiency by technology |
| `services_floor-area` | `bld_floor-area_services` [m²], type (6) × class B–F |

> ⚠️ To split each FSO fuel between hot water and space heating, the pipeline needs a
> space-heating fuel mix. It currently uses a copy of the household hot-water mix as a
> stand-in. Because both mixes are the same, every fuel is split in the same ratio, so the
> services hot-water mix ends up close to the total services heating fuel mix. The stand-in
> also sets how FSO "renewables" are split into solar and heat pumps. It does **not** affect
> services space heating in the model, which uses the MFH mix from step 4.

---

### 10. Checking the results against official data

This is not a preprocessing step. The comparison of a Swiss buildings run with official
statistics lives in `scripts/buildings_validation_CH.py` (repo root). It runs
climate → lifestyles → buildings for Switzerland and compares final energy by fuel with:

- BFE overall energy statistics (*Gesamtenergiestatistik*): households (T17a) and services
  (T17c), all uses together;
- FSO energy accounts (STAT-TAB `px-x-0204000000_106`): household heating, which includes
  hot water.

It then checks services space heating on its own:

- useful heat per m² against EP2050+ *Dienstleistung*, Tabelle 03-01 (stock built before
  2020). This checks the services U-values. It needs the TPE variables
  `bld_useful-heat_<type>` [TWh] and `bld_floor-area-occupied_<type>` [Mm²], which the
  buildings module publishes for this purpose;
- space heating, final energy, against Prognos Table 26 (includes agriculture);
- the heating mix against EP2050+ Tabelle 02-02. EP2050+ gives a share of heating systems
  in buildings built before 2010, so it is compared with the model's input mix weighted
  by floor area (also a share of systems), not with energy by fuel.

Results are printed as tables and plotted in `scripts/output/buildings_validation_CH.html`.
Official files are cached in `scripts/data/`. The EP2050+ workbook is read from this
folder's `data/EP2050_sectors/` if it exists, otherwise downloaded once.

```bash
uv run python scripts/buildings_validation_CH.py
uv run python scripts/fit_services_u_values_CH.py   # services U-value fit (see step 5)
```

---

## Phase 2: assembling the historical pickle

`processors/build_ots_pickle.py`

This step collects the outputs of steps 3–9 and arranges them into the structure the
buildings module expects. Most outputs are copied as they are, but a few are derived here:

- **Floor area per person** (`lfs_floor-intensity_space-cap`): total residential stock ÷
  population. It is stored with household size and non-residential floor area per person
  under `ots/floor-intensity`.
- **Building mix**: the share of each building type × class in the stock
  (`bld_building-mix_stock`) and in new construction (`bld_building-mix_new`), across all
  eight types. For non-residential buildings, "new" is the yearly increase in floor area,
  assigned to the same class per year as residential (D 1990–2000, C 2001–2010, B from 2011).
- **Demolition rate**: waste ÷ stock for residential buildings. For non-residential
  buildings it follows from the EP2050+ stock balance: new construction is the sum of
  yearly class increases, and waste = new construction − change in total stock.
- **Non-residential renovation rate**: EP2050+ *Dienstleistung*, sheet "01 Sanierung",
  total row.

It then writes `ots`, `fxa` and `constant` to `buildings.pickle`.

---

## Output: `buildings.pickle` (historical part)

```
DM_buildings
├── ots
│   ├── floor-intensity                 m²/cap residential, household size, non-res m²/cap
│   ├── heatcool-behaviour              bld_Tint-heating, bld_Tint-cooling  [type × class]
│   ├── services-floor-area             bld_floor-area_services  [non-res type × class]
│   ├── building-renovation-rate
│   │   ├── bld_building-mix            bld_building-mix_new  [type × class]
│   │   ├── bld_renovation-rate         [type]
│   │   ├── bld_renovation-redistribution  in / out  [class]
│   │   └── bld_demolition-rate         [type]
│   ├── heating-technology-fuel
│   │   ├── bld_heating-technology      bld_heating-mix  [type × class × technology]
│   │   └── bld_hot-water-technology    bld_hw_tech-mix  [technology]
│   └── heating-efficiency              bld_heating-efficiency  [class × technology]
├── fxa
│   ├── bld_age, u-value, surface-to-floorarea, bld_type
│   ├── emission-factor-heating_district
│   ├── appliances, other-electricity-demand, lighting
│   ├── hot-water                       {hw-energy-demand, hw-efficiency}
│   └── services                        {services_demand, services_tech-mix, services_efficiencies}
└── constant
    └── emissions                       bld_CO2-factors  [fuel], ConstantDataMatrix
```

All DataMatrices have `Country` = Switzerland and the 26 cantons (plus any other
country already in the pickle). `ots` matrices cover 1990–2023. The pickle also holds future
values used by the scenarios, which are not listed here.

---

## Data sources

| Source | Used for | Access | Cached as (in `data/`) |
|---|---|---|---|
| Swiss Stats, building register `DF_GWS_REG7` | Floor area, number of buildings and apartments by period | SDMX API | `bld_floor-area_stock_all_cantons_swiss_stat.pickle`, `bld_apartments_per_bld_type_swiss_stat.pickle` |
| Swiss Stats `DF_GWS_REG1` | Number of buildings (renovation rate) | SDMX API | `bld_nb-buildings_2010_2025_swiss_stat.pickle` |
| Swiss Stats `DF_GWS_REG4` | Heating and hot-water energy source, 2021–2023 | SDMX API | `bld_heating_technology_all_cantons_swiss_stat.pickle`, `bld_hotwater_technology_2021-2023_swiss_stat.pickle` |
| STAT-TAB `px-x-0902020100_112` | Heating and hot-water energy source, 1990 and 2000 | PxWeb API | `bld_heating_technology_1990-2000_all_cantons.pickle`, `bld_hotwater_technology_1990_2000_old.pickle` |
| STAT-TAB `px-x-0904030000_103` / `_107` | New dwellings 1995–2012 / 2013–2023 | PxWeb API | `bld_new_buildings_1995_2012_all_cantons.pickle`, `bld_new_buidlings_2013_2023_all_cantons.pickle` |
| STAT-TAB `px-x-0204000000_106` | Energy accounts by sector and fuel | PxWeb API | `energy_accounts_economy_households.pickle` |
| STAT-TAB `px-x-0602010000_101` | Employees by sector and canton | PxWeb API | `employees_per_sector_canton.pickle` |
| FSO households table | Number of households | Download | `households_number.xlsx` |
| EP2050+ detailed results (*Private Haushalte*, *Dienstleistung*) | Residential floor area, heating mix, heating efficiency, non-residential floor area and renovation; services useful heat (U-value fit and validation) and services heating mix (validation) | Download (zip) | `EP2050_sectors/`, `bld_EP2050_floor_area_type.pickle`, `bld_heating_tech_EP2050.pickle`, `bld_heating_efficiencies.pickle` |
| EP2050+ / Prognos, *Analyse nach Verwendungszweck* | Hot water, lighting, cooking, other electricity | Download | `EP2050_Households_CH.xlsx` |
| Prognos/Infras/TEP 2021, Table 26 | Services energy by end use; services space heating (validation) | Hard-coded (`services_CH.py`) | – |
| SFOE open data `ogd109` | Appliance stock, sales, consumption | Download | `appliances_stock_new_energy_CH_raw.csv` |
| SFOE *Gesamtenergiestatistik* T17a, T17c | Validation only (`scripts/buildings_validation_CH.py`) | Download | `scripts/data/statistique_globale_suisse_energie.xlsx` |
| Wüest & Partner, EBF 2023 | Cantonal floor-area anchor | Local file | `WP_EBF_cantons_2023.xlsx` |
| Wüest & Partner for BFE (2004), *Zukünftige Entwicklung der Energiebezugsflächen*, App. A-8 | Non-residential floor area 1990–1999 | [PDF](https://pubdb.bfe.admin.ch/de/publication/download/1077), transcribed | `WP_EBF_services_1990_2003_BFE2004.csv` |
| Programme Bâtiments annual reports 2017–2022 | Renovated buildings | Local files | `programme_batiment/` |
| RegBL building register | Non-residential type shares | Local file | `Regbl/Switzerland/` |
| OFEV CO₂ factors for cantonal reporting | Fuel emission factors | Hard-coded | – |
| Pongelli et al. 2023; Döring & Richter 2024; Delmastro et al. 2016; Fleiter et al. 2017 | Residential U-values, surface ratios, heat-pump COP by class | Hard-coded | – |

---

## Known issues and caveats

- **Renovation rate** counts only subsidised renovations (Programme Bâtiments), so it
  underestimates the total.
- **Demolition** uses a 0.2 % floor rather than measured data.
- **Services hot-water mix** follows the total services heating fuel mix, because the
  hot-water / space-heating split uses a stand-in space-heating mix (see step 9).
- **Non-residential heating mix** copies MFH. Compared with EP2050+, services get too many
  heat pumps (2023: 23 % vs 9 % of systems) and too little oil, gas and district heating.
  This likely explains why services final space heating is about 15 % below Prognos
  although useful heat matches EP2050+.
- **Internal gains are not subtracted** in the heating formula. The residential and the
  fitted services U-values absorb them on average, but the trend over time is off: the
  model falls more slowly than EP2050+ (services: 0.89 × EP2050+ in 2000, 1.07 × in 2023
  after the fit).
- **Services electricity is too low** (about 0.77 × BFE): Prognos "Other" and
  "process-heat" are dropped, and the intensities are rescaled to FSO totals.
