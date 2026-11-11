# Buildings pre-processing for Switzerland

A guide to `buildings_preprocessing_main_CH.py`: what it builds, where the data comes from,
and how each step works. It assumes you already know the `DataMatrix` class.

---

## What the script does

The buildings module needs a lot of input data: how much floor area exists, how fast it is
renovated, which heating systems are used, how efficient they are, how much electricity
appliances use, and more. It needs this for every region it runs (Switzerland and the
cantons), for the historical period and for each lever level in the future.

`buildings_preprocessing_main_CH.py` builds all of that from public statistics and writes it
to one file:

```
transition_compass_model/_database/data/datamatrix/buildings.pickle
```

The main script itself is short. It calls a series of **pipelines** (`processors/`) that
build the historical data (1990–2023), then a series of **scenario builders** (`scenarios/`)
that add the future (2025–2050) for each lever level. The data extraction and cleaning
helpers used by the pipelines live in `get_data_functions/`.

```
buildings_preprocessing_main_CH.py
├── processors/          historical pipelines, one per topic, plus build_ots_pickle.py
├── scenarios/           future values per lever level
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
| `country_list` | Switzerland, Vaud, Fribourg, Schwyz | Regions written to the pickle |
| `years_ots` | 1990–2023, yearly | Historical years |
| `years_fts` | 2025–2050, every 5 years | Future years |

Most pipelines compute **all 26 cantons** internally (several steps need the full set to
compute shares), then keep only `country_list` at the end.

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
country already in the file. The pickle is written several times during the run (after
building the historical data and after most scenario steps), so a crash halfway leaves a
partly updated file.

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

**ots / fts / fxa / constant.** These are the top-level keys of the pickle:
- `ots`: historical values (1990–2023) of variables that have a lever.
- `fts`: future values (2025–2050) of the same variables, one DataMatrix per lever level
  (`fts[key][1]` to `fts[key][4]`).
- `fxa`: variables without a lever. They are the same in every scenario and usually cover
  both historical and future years.
- `constant`: values with no country or year dimension (a `ConstantDataMatrix`).

---

## The big picture

The main script runs in three phases.

```
Phase 1: historical pipelines (processors/)
   interfaces ─► population
   floor area ─► renovation ─► space-heating technology
   fixed assumptions, appliances, hot water, lighting, services, calibration
                                │
Phase 2: assemble the historical pickle (build_ots_pickle.py)
                                │  writes ots, fxa, constant
Phase 3: future scenarios (scenarios/)
   BAU for all levels ─► temperature ─► floor area per person ─► Vaud policies (PCV)
   ─► EP2050 ─► decent living standard (DLS) ─► technical limit ─► fill gaps ─► heat-pump efficiency
                                   writes fts[...][1..4]
```

Order matters in two places. Renovation needs the floor-area flows. Space heating needs the
floor area by class that comes out of renovation. Each scenario builder starts from what the
previous one left in `DM_buildings`, so later steps can overwrite earlier ones.

---

## Phase 1: historical pipelines

### 1. Interfaces and population

`processors/buildings_interfaces_CH.py`

**What it answers:** which inputs from other modules the buildings module will receive.

It reads population from `lifestyles.pickle` and degree days from `climate.pickle`, for
historical years and level-1 future years. It writes them to
`data/interface/lifestyles_to_buildings.pickle` and `data/interface/climate_to_buildings.pickle`,
so the buildings module can run on its own in local tests. It returns the population, which
the main script uses later.

---

### 2. Floor area: how much residential space exists, and how it changes

`processors/floor_area_pipeline_CH.py` (helpers in `get_data_functions/floor_area_CH.py`)

**What it answers:** for each canton and year from 1990 to 2023, how many m² of SFH and MFH
housing exist, how much is built, and how much is demolished. It also splits these by
envelope class.

**Building the stock.** The starting point is the Swiss Stats building register (SDMX
dataflow `CH1.GWS / DF_GWS_REG7`). It gives floor area and number of buildings by canton,
building type and construction period. Each construction period is mapped to an envelope
class (see the table above), and summing over classes gives `stock tot`. The class split is
only trusted for **2023**, so every other year of `stock cat` is left as NaN. Years the
register doesn't cover are filled by keeping floor area per person on its trend.

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

> ⚠️ The 0.2 % floor is a proxy, not measured demolition data (see the `!TODO` in
> `compute_waste`).

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

**Where renovated floor area goes.** A renovation moves floor area from one class to a
better one. The pipeline assumes renovated floor area **leaves** class F (80 %) and E (20 %),
and **enters** better classes depending on the period:

| Period | E | D | C | B |
|---|---|---|---|---|
| 1990–2000 | 85 % | 15 % | – | – |
| 2001–2010 | 69 % | 16 % | 15 % | – |
| 2011–2023 | 46 % | 23 % | 15 % | 16 % |

The 2011–2023 split follows the Programme Bâtiments statistics on how many classes a
renovation gains (+1 class 57 %, +2 15 %, +3 15 %, +4 13 %). The older splits are educated
guesses, since classes A/B didn't exist yet.

The renovated floor area per class is then:

```
r_c(t) = (in_c − out_c) × renovation-rate(t−1) × stock(t−1)
```

**New floor area by class.** For classes that only exist after 1990, the new-dwellings series
(`new raw`) is scaled so that *everything built or renovated into the class since 1990*
equals its 2023 stock. Each class is empty before its construction period starts. SFH class D
and MFH class E also contain older buildings, so they use the scaling factor of class C.
The factors are printed during the run. Because `new raw` is already in the units of
the stock, they should be close to 1 (about 0.85–1.05). A factor below zero (more renovated into the class
than its 2023 stock) is set to 0, with a message.

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

> ⚠️ An option to weight the mix by number of apartments (`adjust_based_on_apt`) is switched
> off because it gave poor results for Vaud.

---

### 5. Fixed assumptions

`processors/others_pipeline_CH.py`

**What it answers:** physical and behavioural parameters that come from the literature
rather than from time series.

| Output key | What | Values and source |
|---|---|---|
| `u-value` | Average envelope U-value [W/m²K], type × class | Residential: Pongelli et al. (2023), e.g. SFH F 0.82 → B 0.25. Non-residential: EU Hotmaps |
| `surface-to-floor` | Envelope surface per m² floor area | SFH 2.0, MFH 1.3 (Pongelli et al.). Non-residential 1.1–1.4 (Delmastro et al. 2016, Fleiter et al. 2017) |
| `Tint-heat` | Indoor heating temperature [°C] | MFH: F 19 °C to B 23 °C (better buildings are kept warmer). SFH: 1 °C lower. Non-residential: 20 °C (SIA 380/1) |
| `age` | Age of each class [years] | Year minus the first construction year of the class |
| `emission-factors` | CO₂ per fuel [kt/TWh] | OFEV: coal 350, heating oil 265, gas 201, wood and solar 0 |
| `emission-fact-elec_district` | CO₂ of electricity and district heating [kt/TWh] | Electricity 168.64 until 2023, falling to 0 in 2050. District heating 73.55 until 2019, 66 in 2023, 20.7 in 2050 |

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

**Output** (all go to `fxa`, historical and future years):

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
4. Heat pumps get a hot-water COP from EP2050+ (Abb. 17): 2.8 in 2015, 3.0 in 2030, 3.5 in
2050. Useful energy is *consumption × efficiency*.

> ⚠️ The fuel shares from consumption and from building counts disagree (heating oil is
> lower, gas higher). The code comment suggests gas-heated buildings are larger on average.
> The consumption-based shares are used for the technology mix.

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
process heat, lighting, HVAC, ICT, drives and processes, other. The values are hard-coded in
the pipeline. It is applied per canton, then adjusted so that each fuel matches the FSO
consumption. Useful energy is *consumption × efficiency*, using the hot-water efficiencies
(biogas = gas, biomass = wood, everything else 1).

**Intensity per m².** The total useful energy by end use is divided by the Swiss
non-residential floor area to give kWh/m². The same intensity is used for every canton,
because cantonal demand and cantonal floor area are both shared out by employees. Space
heating is dropped: the buildings module computes it from physics (floor area × U-value ×
degree days).

**Floor area.** EP2050+ *Dienstleistung* (sheet "04 EBF-Baualter") gives national
non-residential floor area by construction period (thousand m², 2000–2060). These periods are
mapped to the MFH classes:
- F = before 1946 + 1947–1975 + 5/15 of 1976–1990
- E = 10/15 of 1976–1990 (the band is split evenly per year)
- The 1991–2019 band is split by when the floor area was added: D is its 2000 stock, C what
  was added 2001–2010, B what was added 2011–2019 (assuming no demolition before 2020).
  After 2019 the band declines; the decline goes to D and C, the older buildings.
- B also takes the whole "from 2020" band.

The floor area is then split into
the six building types using the type shares of each class in the RegBL building register
(`data/Regbl/Switzerland/`), and distributed to cantons by their share of service employees.
EP2050+ starts in 2000. For 1990–1999 the total follows the yearly services floor-area stock
of Wüest & Partner (BFE 2004), without agricultural buildings, scaled to EP2050+ in 2000.
Class D (built 1991–2000) grows from 0 in 1990 to its EP2050+ value in 2000, following the
timing of the floor area added in the W&P series; classes E and F take the rest.

**Output:**

| Key | Content |
|---|---|
| `services_demand` | `bld_services_useful-energy` [kWh/m²] by end use |
| `services_tech-mix` | technology share by end use |
| `services_efficiencies` | efficiency by technology |
| `services_floor-area` | `bld_floor-area_services` [m²], type (6) × class B–F |

> ⚠️ The space-heating technology mix of services is a placeholder copied from hot water
> (`!FIXME: DUMMY dm_space_heat`). It affects how "renewables" are split.

---

### 10. Calibration data

`processors/calibration_pipeline_CH.py`

This step reads residential (T17a) and services (T17c) final energy by carrier from the Swiss
overall energy statistics (*Gesamtenergiestatistik*), for Switzerland only. Cantons are added
as NaN.

> ⚠️ The result (`dm_energy_cal`) is computed but **not used**: it doesn't go into
> `DM_all` and is not written to the pickle.

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
- **Future values of some `fxa` variables**: hot-water demand and services intensity are
  extrapolated linearly. Future lighting follows the EP2050+ Swiss trajectory (from 2.5 PJ
  in 2025 to 1.7 PJ in 2050), and each canton keeps its average historical share.

It then writes `ots`, `fxa` and `constant` to `buildings.pickle`.

---

## Phase 3: future scenarios (lever levels)

Every lever has four levels, from 1 (business as usual) to 4 (most ambitious). The scenario
builders first fill **all four levels with BAU**, then overwrite specific levels.

Many policies are coded for **Vaud only** (`dm["Vaud", ...]`). For Switzerland, Fribourg
and Schwyz, those levels keep the BAU values unless stated otherwise below.

### Step by step

1. **BAU for all levels** (`build_fts_BAU_pickle.py`). Historical trends are extended to 2050:
   - Floor area per person: linear trend (non-residential fitted on 2012–2023 only, since it
     peaked around 2015).
   - Renovation rate, redistribution and new-building mix: held at the last historical
     value. For non-residential renovation rates, level 1 uses EP2050+ WWB and level 4 uses
     EP2050+ ZERO, with levels 2–3 in between.
   - Demolition rate: the 2012–2023 average.
   - Heating and hot-water technology mix: shares extended linearly from 2015–2023, kept
     non-negative and renormalised. In Vaud, resistive electric heating falls to zero by 2035
     (cantonal ban from 2033).
   - Efficiency: extended from 2015–2023, capped at 98 % (heat pumps not capped, but limited
     by the class COPs).
2. **Indoor temperature** (`build_fts_Tint_heating_pickle.py`). By 2050, levels 2/3/4 lower
   the heating temperature by 0.5/1/1.5 °C, but not below 21/20/19.5 °C.
3. **Floor area per person** (`build_fts_floor_area_pickle.py`). Level 2 holds m²/person at
   its 2023 value. Level 4 cuts m²/person by twice the amount needed to keep the total floor
   area at its 2023 level under the lowest population forecast. Level 3 is halfway between
   levels 2 and 4.
4. **Vaud energy law and climate plan, PCV** (`build_fts_PCV_LVLEne_pickle.py`, Vaud only):
   - Renovation, level 3: MFH in class F/G above 750 m² are renovated by 2035, as the
     *Loi sur l'énergie* requires. Other buildings keep a BAU-like rate.
   - Renovation, level 4: 85 % of the class-F stock is renovated between 2025 and 2040, for
     both SFH and MFH.
   - Renovation redistribution, levels 3–4: the split of renovated floor area leaving F and
     E is recomputed from the new renovation rates. Renovations that would have ended in
     class E go to class D instead, because the law requires at least class D.
   - Heating technology, level 2: fossil heating falls to 25 % of its 2023 share by 2040 and
     to 5 % (exceptions) by 2045. The freed share goes to district heating, heat pumps, solar
     and wood, in the proportions of the *Perspectives chaleur* study (different for SFH and
     MFH).
   - Hot water, levels 2–4: the same fossil phase-out and replacement.
5. **EP2050+ heating mix** (`buildings_fts_EP2050_pickle.py`, Vaud, level 3). The 2050 mix
   is set to EP2050+ values: heat pumps 82 % (SFH) / 59 % (MFH), district heating 9 % / 32 %,
   biogas 5.8 %, wood 2.8 %. Oil and coal reach zero by 2045. Resistive electric heating
   reaches zero by 2035 (classes E–F) or 2040 (B–D).
6. **Decent living standard, DLS** (`build_fts_DLS.py`, level 4):
   - Vaud: residential floor area falls by 23.9 m²/person between 2025 and 2050.
   - All regions: non-residential floor area stays at its 2023 value.
   - All regions: indoor temperature is 19 °C from 2030.
7. **Technical limit** (`build_fts_technical_limit_pickle.py`, Vaud, level 4). Starting from
   level 2, fossil heating and hot water fall to zero by 2045, with the same *Perspectives
   chaleur* replacement.
8. **Fill gaps** (`build_fts_fill.py`). If non-residential floor area is still the same for
   levels 1 and 2, levels 2 and 3 are set at 25 % and 75 % of the way from level 1 to level 4.
9. **Heat-pump efficiency** (`build_fts_heating_efficiency_pickle.py`). 2050 COPs come from
   EP2050+ (Abb. 16). Level 3 uses air-to-water values (B 5.1 … F 3.3). Level 4 uses a 50/50
   mix of air-to-water and ground-source (B 5.7 … F 3.6). Level 2 is the average of levels 1
   and 3.

### Summary: what each level contains after the run

| Lever (pickle key) | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---|---|---|---|
| `floor-intensity` | trend | flat at 2023 | half of max cut | Vaud: DLS (−23.9 m²/cap); others: max cut |
| `heatcool-behaviour` | as 2023 | −0.5 °C (≥ 21) | −1 °C (≥ 20) | 19 °C from 2030 |
| `services-floor-area` | trend | 25 % towards L4 | 75 % towards L4 | flat at 2023 |
| `bld_renovation-rate` | BAU | BAU | Vaud: energy law (MFH > 750 m²) | Vaud: 85 % of F by 2040 |
| `bld_renovation-redistribution` | BAU | BAU | Vaud: straight to D | Vaud: straight to D |
| `bld_demolition-rate` | 2012–23 avg | same | same | same |
| `bld_building-mix` | last value | same | same | same |
| `bld_heating-technology` | trend (+ Vaud no electric heating by 2035) | Vaud: fossil 5 % by 2045 | Vaud: EP2050+ mix | Vaud: fossil 0 by 2045 |
| `bld_hot-water-technology` | trend (+ Vaud no electric by 2035) | Vaud: fossil 5 % by 2045 | same as L2 | Vaud: fossil 0 by 2045 |
| `heating-efficiency` | trend | midpoint L1–L3 | air-source HP 2050 | mixed HP 2050 |

---

## Output: `buildings.pickle`

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
├── fts                                 same keys as ots, each a dict {1: dm, 2: dm, 3: dm, 4: dm}
├── fxa
│   ├── bld_age, u-value, surface-to-floorarea, bld_type
│   ├── emission-factor-electricity, emission-factor-heating_district
│   ├── appliances, other-electricity-demand, lighting
│   ├── hot-water                       {hw-energy-demand, hw-efficiency}
│   └── services                        {services_demand, services_tech-mix, services_efficiencies}
└── constant
    └── emissions                       bld_CO2-factors  [fuel], ConstantDataMatrix
```

All DataMatrices have `Country` = Switzerland, Vaud, Fribourg, Schwyz (plus any other
country already in the pickle). `ots` matrices cover 1990–2023, `fts` matrices 2025–2050 in
5-year steps, and most `fxa` matrices both.

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
| EP2050+ detailed results (*Private Haushalte*, *Dienstleistung*) | Residential floor area, heating mix, heating efficiency, non-residential floor area and renovation | Download (zip) | `EP2050_sectors/`, `bld_EP2050_floor_area_type.pickle`, `bld_heating_tech_EP2050.pickle`, `bld_heating_efficiencies.pickle` |
| EP2050+ / Prognos, *Analyse nach Verwendungszweck* | Hot water, lighting, cooking, other electricity | Download | `EP2050_Households_CH.xlsx` |
| Prognos/Infras/TEP 2021, Table 26 | Services energy by end use | Hard-coded | – |
| SFOE open data `ogd109` | Appliance stock, sales, consumption | Download | `appliances_stock_new_energy_CH_raw.csv` |
| SFOE *Gesamtenergiestatistik* T17a, T17c | Calibration (unused) | Download | `statistique_globale_suisse_energie.xlsx` |
| Wüest & Partner, EBF 2023 | Cantonal floor-area anchor | Local file | `WP_EBF_cantons_2023.xlsx` |
| Wüest & Partner for BFE (2004), *Zukünftige Entwicklung der Energiebezugsflächen*, App. A-8 | Non-residential floor area 1990–1999 | [PDF](https://pubdb.bfe.admin.ch/de/publication/download/1077), transcribed | `WP_EBF_services_1990_2003_BFE2004.csv` |
| Programme Bâtiments annual reports 2017–2022 | Renovated buildings | Local files | `programme_batiment/` |
| RegBL building register | Non-residential type shares | Local file | `Regbl/Switzerland/` |
| OFEV CO₂ factors for cantonal reporting | Fuel emission factors | Hard-coded | – |
| Pongelli et al. 2023; Döring & Richter 2024; Delmastro et al. 2016; Fleiter et al. 2017 | U-values, surface ratios, heat-pump COP by class | Hard-coded | – |
| *Perspectives chaleur* study (Vaud) | Replacement mix when fossil heating is phased out | Hard-coded | – |

---

## Known issues and caveats

- **Several policy levels apply to Vaud only.** Fribourg and Schwyz get BAU for most
  heating, hot-water and renovation levels.
- **Fribourg and Schwyz renovation shares are placeholders** (`FIXME` in the renovation
  pipeline). In the current code that dictionary isn't used, since the rate comes from the
  Programme Bâtiments reports instead.
- **Demolition** uses a 0.2 % floor rather than measured data.
- **Services space-heating mix** is a copy of the hot-water mix.
- **Calibration data** is extracted but not written to the pickle.
- **Non-residential heating mix** copies MFH.
- The **main script's printed messages** don't always match which levels a step writes. For
  example, "PCV (level 2) and PCV+ (level 3)" also writes renovation for levels 3 and 4,
  and hot water for levels 2–4. The table above shows the final state.
