# Transport pre-processing for Switzerland: historical data

A guide to the historical part of `transport_preprocessing_main_CH.py`: what it builds,
where the data comes from, and how each step works. It assumes you already know the
`DataMatrix` class.

This version covers the historical years only (1990–2023). The script also builds future
scenarios (2025–2050), which are not described here.

---

## What the script does

The transport module needs a lot of input data: how far people travel and by which mode,
how many vehicles exist, which technologies they use, how much energy they need per km,
how much freight is moved, and more. It needs this for every region it runs (Switzerland
and the cantons).

`transport_preprocessing_main_CH.py` builds all of that from public statistics and writes
it to one file:

```
transition_compass_model/_database/data/datamatrix/transport.pickle
```

The main script itself is short. It calls a series of **pipelines** (`processors/`) that
build the historical data (1990–2023) for passengers, aviation and freight. The data
extraction and cleaning helpers used by the pipelines live in `get_data_functions/`.

```
transport_preprocessing_main_CH.py
├── processors/          historical pipelines, one per topic, plus transport_ots_pickle.py
├── scenarios/           future values (not covered here)
├── get_data_functions/  download, cache and clean each source
└── data/                cached raw files and intermediate pickles
```

---

## Before you run it

**Run it as a script, from its own folder.** The imports `from processors...` and
`from scenarios...` need the folder on the Python path, and the pre-2005 car fleet is read
from `data/px-x-1103020100_101.csv` with a path relative to the current directory:

```bash
cd transition_compass_model/_database/pre_processing/transport/Switzerland
uv run python transport_preprocessing_main_CH.py
```

**Settings** come from `_database/pre_processing/params.py`:

| Setting | Value | Meaning |
|---|---|---|
| `country_list` | Switzerland and the 26 cantons | Regions written to the pickle |
| `years_ots` | 1990–2023, yearly | Historical years |

Most data is national. Only a few inputs have real cantonal values (passenger km, car
fleet, car efficiency, freight tkm). Everything else is copied from Switzerland to each
canton, see [How cantons get their values](#how-cantons-get-their-values).

**Other pickles must already exist and be up to date:**

- `data/datamatrix/lifestyles.pickle`: population (through `load_pop`).
- `data/datamatrix/transport.pickle`: the existing file is updated, not replaced (see
  below).
- `data/interface/lifestyles_to_transport.pickle`: needed by the scenario part of the
  script.

**Downloads are cached.** Most sources are read from a local file in `data/` if it
exists. They are downloaded (from the FSO website, Swiss Stats, STAT-TAB, the SFOE
website…) only when the file is missing. To refresh a source, delete its cached file.
Some sources are not downloaded at all and must already be in `data/`:

- the FSO table of new vehicles 1990–2023 (`tra_new-vehicles_CH_1990-2023.csv`, download
  it by hand from FSO asset 30305446, number gr-e-11.03.02.02.01a);
- the STAT-TAB car fleet archive (`px-x-1103020100_101.csv`);
- the MRMT regional tables for 2000 and 2005;
- the freight files in `data/Freight/`;
- the FOEN greenhouse-gas inventory (`Evolution_GHG_since_1990_2025-04.xlsx`);
- the aviation Excel and CSV files (`aviation_*.xlsx`, `aviation_pkm_suisse.csv`,
  `tra_global_aviation-demand.csv`).

**The pickle is merged, not overwritten.** Every write goes through `my_pickle_dump`. For
each DataMatrix, it replaces the countries present in the new data and keeps any other
country already in the file (today the pickle also holds `EU27`). The pickle is written
several times during the run, so a crash halfway leaves a partly updated file.

---

## Words used in this guide

**Passenger modes (`Categories1`).** `LDV` (cars), `2W` (motorbikes and mopeds), `bus`,
`rail`, `metrotram`, `bike`, `walk`, and `aviation`, which is built separately.

**Freight modes.** Road trucks are split by body type, not by weight:
- `HDVH`: articulated trucks (tractor and trailer).
- `HDVM`: rigid trucks of 3.5 t and more.
- `HDVL`: vans under 3.5 t.

Non-road freight is `rail`, `IWW` (Rhine barges), `aviation` and `marine` (always zero,
Switzerland has no sea access).

**Technologies (`Categories2`).** `ICE-diesel`, `ICE-gasoline`, `ICE-gas` (combustion
engines; hybrids that can't be plugged in count as combustion), `PHEV-diesel`,
`PHEV-gasoline` (plug-in hybrids), `BEV` (battery electric), `FCEV` (fuel cell), `CEV`
(electric from an overhead line: electric trains, trolleybuses), `mt` (metro and tram).
Aviation uses `kerosene`, `BEV` and `H2`.

**Activity units.**
- **pkm** (passenger-km): one person travelling one km.
- **vkm** (vehicle-km): one vehicle driving one km.
- **tkm** (tonne-km): one tonne of goods moved one km.
- **skm** (seat-km): one aircraft seat flown one km.

**Ratios used everywhere.**
- **Occupancy** = pkm / vkm: people per vehicle.
- **Load factor** = tkm / vkm: tonnes per truck.
- **Utilisation rate** = vkm / vehicles: km driven per vehicle per year.
- **Renewal rate** = vehicles scrapped in a year / fleet of the year before.

**Territorial vs residents.** *Territorial* counts all traffic on Swiss soil, whoever
drives. *Residents* counts the travel of Swiss residents, wherever it happens. Freight is
territorial. Aviation demand is counted for residents, and the territorial part is used to
decide which emissions are local.

**ots / fxa / constant.** These are the top-level keys of the pickle that hold historical
data:
- `ots`: historical values (1990–2023) of variables that can change in the scenarios.
- `fxa`: variables that are the same in every scenario.
- `constant`: emission factors, with no country or year dimension (a
  `ConstantDataMatrix`).

---

## The big picture

The historical part runs in two phases.

```
Phase 1: historical pipelines (processors/)
   passenger: demand (pkm, vkm) ─► fleet ─► renewal and waste
              emission factors, energy (EP2050+) ─► efficiency
              EP2050+ vkm (for the efficiency correction)
   aviation:  part 1 (downloads) ─► aviation history and fixed inputs
   freight:   tkm and modal share ─► non-road inputs
              road trucks from targets ─► utilisation ─► efficiency, tech share
                                │
Phase 2: assemble the historical pickle (transport_ots_pickle.py)
                                   writes ots, fxa, constant
```

Order matters. The fleet needs the pkm (to share the public-transport fleet among
cantons). The efficiency needs the energy, the vkm, the fleet and the emission factors.
Road freight needs the emission factors.

---

## Phase 1a: passengers

### 1. Demand: how far people travel, and by which mode

`processors/transport_demand_pipeline.py` (helpers in `get_data_functions/demand_pkm_vkm.py`)

**What it answers:** for each region and year from 1990 to 2023, how many pkm per person
are travelled by each mode, and how many vkm that means.

**National pkm.** FSO, *Transport de personnes: prestations de transport*
(je-f-11.04.01.02) gives Swiss pkm by mode. Missing values are filled with linear trends
(fitted from 2020 on for the recent years, before 2008 for the early years). FSO's "Cars"
row (coaches) is counted as `bus`, like the survey ("Autocar") and EP2050+ ("coach").

**Regional split.** The Swiss mobility survey (MRMT, *Microrecensement mobilité et
transports*) gives km per person per day by mode for each *grande région* (7 groups of
cantons), for 2000, 2005, 2010, 2015 and 2021. Each canton first gets the values of its
grande région. The total is then rescaled so that it matches the canton's own survey
figure (MRMT cantonal key figures, 2015 and 2021). For 2000, 2005 and 2010 the average of
the 2015 and 2021 corrections is used. The mode split stays that of the grande région.

**Filling the years between surveys.** The survey and FSO don't count the same thing, so
both are brought to FSO units:
1. For Switzerland, the survey values are filled between survey years by following the
   FSO pkm per person curve. The ratio *FSO / survey* gives an adjustment factor per year
   and mode. Switzerland's final pkm per person is FSO's.
2. Each canton's survey values are filled by following the Swiss survey curve, then
   multiplied by the Swiss adjustment factor.

Total pkm = pkm per person × population.

**vkm.** FSO, *prestations kilométriques et mouvements des véhicules* (je-f-11.04.01.01)
gives Swiss vkm by mode (2023 is extrapolated from 2020 on; a 2004 spike in metro/tram is
removed). The "Cars privés" row (coaches) is counted as `bus`, as for pkm. Swiss occupancy
is *FSO pkm / FSO vkm*. Every canton gets the Swiss occupancy, and its vkm is its pkm
divided by that occupancy. Bike and walk have no vkm.

**Output:** `dm_pkm_cap` [pkm/cap], `dm_pkm` [pkm] and `dm_vkm` [vkm], by mode.

---

### 2. Fleet: how many vehicles of each technology

`processors/passenger_fleet_pipeline.py` (helpers in `get_data_functions/passenger_fleet.py`)

**What it answers:** how many cars, motorbikes, buses, trains and trams exist, by
technology, and how many new cars and motorbikes are registered each year.

**Cars and motorbikes, stock.**
- 2005 onwards: Swiss Stats `CH1.MFZ_IVS / DF_MFZ_0_GENERAL`, by canton and fuel.
- 1990–2004: the STAT-TAB archive `px-x-1103020100_101` (a local CSV). It has an "Other"
  fuel category, which is shared among the newer technologies (hybrids, plug-ins, gas,
  fuel cell) in proportion to their cumulated new registrations since 2005.

**Cars and motorbikes, new registrations.**
- 2005 onwards: Swiss Stats `CH1.MFZ_IVS / DF_IVS_0_GENERAL`, by canton and fuel.
- 1990–2004: the Swiss total comes from FSO table gr-e-11.03.02.02.01a. Each canton gets
  its 2005 share of the Swiss total, and the technology split is that of 2005.

**Buses, trains, trams.** FSO, *Transports publics – séries chronologiques détaillées*
(`tra_public_transport.xlsx`) gives the Swiss fleet. Each canton gets the Swiss fleet ×
*canton pkm / Swiss pkm* for that mode. The bus fleet is public-operator buses only,
while bus pkm and vkm include coaches. So bus vkm per vehicle is high, and the
electric-bus efficiency (energy / vkm, step 5) is lower than a city-bus-only value.

**Minibuses.** The Swiss Stats group "Passenger vehicles" (code 200: mostly minibuses up
to 3.5 t, plus coaches and buses) is counted with cars (`LDV`), in the stock and in new
registrations, because most of them are minibuses used like cars.

**Output:** `dm_private_fleet` (stock and new vehicles, `LDV` and `2W`) and
`dm_public_fleet` (stock, `bus`, `rail`, `metrotram`), in number of vehicles.

---

### 3. Renewal and waste: how many vehicles are scrapped

`processors/passenger_renewal_rate_and_waste_pipeline.py`

**What it answers:** how many vehicles leave the fleet each year, and what renewal rate
that implies.

**Cars and motorbikes.** Scrapped vehicles follow from the stock balance:

```
waste(t)       = fleet(t−1) − fleet(t) + new(t)
renewal(t−1)   = waste(t) / fleet(t−1)
```

Renewal rates below 0 or above 10 % per year are treated as errors, removed and
interpolated. New vehicles are then recomputed from the cleaned rate (negative values are
interpolated too), and the fleet is rebuilt year by year from the flows. So the final
fleet can differ slightly from the official count.

**Buses, trains, trams.** The renewal rate is fixed: 1/30 per year for rail, 1/20 for
metro/tram and 1/10 for buses (lifetimes of 30, 20 and 10 years). New vehicles follow
from the same balance.

**Output:** a dict with `passenger_private-fleet`, `passenger_public-fleet`,
`passenger_renewal-rate` [%], `passenger_new-vehicles` and `passenger_waste-fleet`
[number].

---

### 4. Emission factors and energy

**Fuel emission factors** (`processors/passenger_emission_factors.py`). IPCC 2006
Guidelines, Volume 2, Chapter 3 (mobile combustion), converted from kg/TJ to g/MJ. For
CO₂: diesel 74.1, gasoline 69.3, gas 56.1, kerosene 73.3, marine fuel oil 77.4 g/MJ. CH₄
and N₂O are included too. These are used for passengers, freight and aviation.

Emissions from electricity (electric cars, trains, trams, trolleybuses) are not counted in
the transport module. They are computed in the energy module.

**Energy** (`processors/passenger_energy_pipeline.py`, helpers in
`get_data_functions/transport_energy.py`). EP2050+ detailed results, *Verkehrssektor*,
sheet "04 Energieverbrauch Strasse", give Swiss road energy by mode and fuel. Rail energy
comes from the EP2050+ technical report, Kap. 8, Abb. 112, all assigned to `CEV`.

---

### 5. Efficiency: how much energy each vehicle needs per km

`processors/passenger_efficiency_pipeline.py` (helpers in `get_data_functions/efficiency.py`)

**What it answers:** the energy use per km [MJ/km] of the fleet and of new vehicles, for
each mode and technology.

**Cars with an engine (combustion and plug-in hybrid).** Swiss Stats gives the number of
cars in each CO₂ class (0–50 g/km, 51–100 g/km, …), by canton:
- fleet: `DF_MFZ_1_EMISSION`, used from 2016;
- new cars: `DF_IVS_1_EMISSION`, used from 2017.

Each class is replaced by its midpoint, which gives an average g CO₂/km. Cars of unknown
class are spread over the known classes (with the Swiss distribution when more than 15 %
of a canton's cars are unknown). Earlier years take the first available value, and the
fleet series is smoothed (3-year moving average, applied twice). Dividing by the CO₂
emission factor [g/MJ] gives MJ/km.

**All other modes, and electric or fuel-cell cars.** Efficiency = EP2050+ energy / vkm.
The vkm by technology is the FSO vkm × the fleet technology share. Electric car,
fuel-cell car and electric motorbike values are noisy, so each is replaced by its average
over all years (electric cars: 2011 and 2012 are left out of the average as outliers).
Metro/tram comes from JRC-IDEES 2021: 45 kgoe/100 km in 2000 and 32.2 kgoe/100 km in
2021, joined by a straight line. For these modes, new vehicles get the fleet value, and
every canton gets the Swiss value.

**Correction to EP2050+.** For 2W gasoline, LDV diesel, gas and gasoline, and bus diesel,
the Swiss fleet efficiency is set to *EP2050+ energy / EP2050+ vkm*, each year.
- 2W and bus: every canton and new vehicles get this Swiss value, like the other modes.
- LDV: the CO₂-based values are multiplied by *EP2050+ efficiency / Swiss CO₂-based fleet
  efficiency*. So Switzerland matches EP2050+, while cantons and new cars keep their
  CO₂-based ratio to the Swiss fleet. For LDV gas, years before 2007 use the 2007 factor
  (the EP2050+ value is not usable before then).

**Output:** one DataMatrix with `tra_passenger_veh-efficiency_fleet` and
`tra_passenger_veh-efficiency_new` [MJ/km], mode × technology.

---

### 6. EP2050+ vkm

`processors/transport_calib_vkm.py`

Reads EP2050+ road vkm by mode and technology (sheet "03 Fahrleistung", WWB). It is
passed in memory to the vkm adjustment factor and the efficiency step (step 5) for the
EP2050+ correction, and written to `transport/Switzerland/data/datamatrix/calibration_vkm.pickle`,
which the model does not read.

The official totals used to check the model (SFOE energy by fuel, EP2050+ energy by
mode and technology, FOEN emissions by mode) are built and cached by
`scripts/transport_validation_CH.py` in `scripts/data/`.

---

## Phase 1b: aviation

Aviation is computed **for Switzerland only**. Its "fleet" is counted in seats, not
aircraft. Cantons get copies in Phase 2.

### 7. Aviation part 1: downloads

`processors/aviation_part1_pipeline_CH.py` (helpers in `get_data_functions/aviation_part1.py`)

It downloads and caches the FSO civil aviation table (`tra_aviation_CH.xlsx`), which gives
km flown per Swiss resident per year, every 5 years (from the MRMT survey). Missing years
are filled following world aviation pkm per person (Our World in Data, divided by World
Bank population). 2020 is set from 2021 with the world 2020/2021 ratio. The output is the
Swiss aviation pkm per person, used as the starting point of step 8.

### 8. Aviation history and fixed inputs

`processors/aviation_ots_pipeline_CH.py`

**Demand.** The survey years (2000, 2005, 2010, 2015, 2021) are kept as they are. Every
other year is scaled from a nearby survey year by the change in international aviation
CO₂ in the FOEN inventory (for example 2016–2019 from 2015, 2020 from 2021, 2022–2023 from
2015). Residents' pkm is kept at least 1.05 × the territorial pkm
(`aviation_pkm_suisse.csv`).

**Other historical inputs**, mostly from local Excel files in `data/`:

| Input | How it is built |
|---|---|
| Efficiency of new aircraft | `aviation_energy_intensity_fleet.xlsx` |
| Technology share of new aircraft | `aviation_passenger_technology_share_new_ots.xlsx` |
| Occupancy | Swiss (`aviation_occupancy_ots.xlsx`) and world (`aviation_occupancy_pondere_ots.xlsx`) values, weighted by the territorial and non-territorial pkm |
| Utilisation rate | `aviation_utilisation-rate-OTS.xlsx`. 2022–2023 follow the 2005–2019 trend of *utilisation / territorial pkm* |
| Seats | pkm × population / (occupancy × utilisation rate) |
| New and retired seats | Retirement rate from `aviation_waste_suisse.xlsx`. New seats = change in seats + retired seats, at least 0. Seats for 2020–2023 are rebuilt from the flows, and the utilisation rate is recomputed to match |

**Share of local emissions** (`fxa/share-local-emissions`): territorial skm / residents'
skm. It decides which part of the residents' aviation emissions is counted on Swiss
territory.

---

## Phase 1c: freight

Freight is **territorial**: all goods moved on Swiss soil, whatever the vehicle's
registration. This matches the targets it is calibrated to (EP2050+ road vkm and energy,
FOEN CO₂). Only the **truck fleet** is the Swiss-registered fleet, because that is what
the industry module needs. All values are national; only tkm is scaled per canton.

### 9. Freight tkm and modal share

`processors/freight_tkm_modal_share.py` (helpers in `get_data_functions/freight_data.py`)

**What it answers:** how many tkm are moved each year, and the share of each mode.

| Mode | Source |
|---|---|
| HDVH, HDVM | FSO heavy-vehicle goods survey (GTS, `ts-x-11.05-GTS-E26.csv`), all registrations and traffic types. Foreign trucks are about 44 % of HDVH tkm |
| HDVL | EP2050+ van vkm (ZERO Basis) × 0.2 t/vkm |
| rail | FSO public transport table T7.2.1, gaps 1991–1994 and 1996–1999 interpolated |
| IWW | FSO Rhine imports and exports (`gr-e-11.05.04b.csv`) × 34 km (the Swiss navigable Rhine, Basel to the German border) |
| marine | 0 |
| aviation | Set so that it is always 4.4 % of the total (calibrated to Swiss airport cargo, about 1.3 bn tkm) |

**Cantons.** Each canton gets its share of the tkm of Swiss-registered trucks that start
in the canton (GTS cantonal survey, 2016–2020 average, `su-f-11.05-GTS16-20-P19.xlsx`).
For example Vaud: 767,635 / 11,013,466 thousand tkm = 6.97 %. This share is applied to
every mode. IWW is 0 except in Basel-Stadt and Basel-Landschaft. Aviation is recomputed as
4.4 % of the canton total.

**Output:** `freight_tkm` (`tra_freight_tkm-total-demand` [bn tkm]) and
`freight_modal-share` [%] by mode.

---

### 10. Non-road freight vehicles

`processors/freight_mode_other_fxa.py`

The model needs a fleet size for rail, IWW, aviation and marine. Fleet sizes are fixed:
7,000 rail wagons (SBB report), 4 Rhine barges (Port of Basel), 12 cargo aircraft (FOCA),
1 ship (marine freight is zero). tkm per vehicle = mode tkm / fleet size. Renewal rates
are fixed EUCalc defaults: 15.59 %/year for IWW and aviation, 40.95 %/year for rail.

**Output:** `freight_mode_other` with `tra_freight_tkm-by-veh` [tkm] and
`tra_freight_renewal-rate` [%].

---

### 11. Road freight: inputs worked back from official targets

`processors/freight_road_ots.py` (readers in `get_data_functions/freight_data.py`)

**What it answers:** for HDVH, HDVM and HDVL, the load factor, utilisation rate, renewal
rate, technology share and efficiency that make the model reproduce official numbers for
the historical years.

The model computes, for each truck type:

```
vkm    = tkm / load factor
fleet  = vkm / utilisation rate
energy = vkm × technology share × efficiency
CO₂    = energy × (1 − biofuel share) × emission factor
```

Each input is inverted from a target. When sources overlap, FSO and FOEN come first,
EP2050+ second, literature last.

| Target | Source |
|---|---|
| tkm | FSO GTS (step 9) |
| vkm | EP2050+ sheet "03 Fahrleistung". Heavy-truck vkm is split between HDVH and HDVM by fleet, with HDVH trucks assumed to drive 2 × more km per year |
| Fleet by technology | FSO stock (Swiss Stats `DF_MFZ_0_GENERAL`) from 2005. 1990–2004 from EP2050+ sheet "02 Flottenbestand", scaled to FSO in 2005 |
| New trucks | FSO first registrations (`DF_IVS_0_GENERAL`) from 2005. Before, worked back from the fleet with the same balance as step 3, with renewal rates capped at 20 %/year |
| Energy by fuel | EP2050+ sheet "04 Energieverbrauch Strasse". Heavy-truck energy is split between HDVH and HDVM by vkm, with HDVH using 1.38 × more fuel per km (HBEFA 4.2). Fuel energy is then rescaled each year so that CO₂ matches FOEN for heavy trucks (HDVH + HDVM) and light trucks (HDVL) |

From these:
- load factor = tkm / vkm (HDVL fixed at 0.2 t/vkm);
- utilisation rate = vkm / fleet [vkm/vehicle/year]. Because vkm is territorial and the
  fleet is Swiss-registered, this also carries the km of foreign trucks;
- efficiency = energy of a technology / (vkm × its fleet share). Technologies under 1 % of
  the fleet use a fixed ratio to diesel (BEV 0.30, FCEV 0.33, PHEV-diesel 0.55,
  ICE-gasoline 1.12, …; HBEFA, IEA, ICCT). Diesel efficiency is then adjusted so that each
  truck type's CO₂ matches its target exactly.

**Output:** a dict per truck type with vkm, load factor, utilisation rate, renewal rate,
fleet, new trucks, technology shares, efficiency and energy. Steps 12–15 turn it into
DataMatrices.

---

### 12–15. Freight DataMatrices

| Step | File | Output |
|---|---|---|
| 12 | `processors/freight_utilization_rate.py` | `freight_utilization-rate`: `tra_freight_load-factor` [tkm/vkm] and `tra_freight_utilisation-rate` [vkm/year], road |
| 13 | `processors/freight_mode_road_fxa.py` | `freight_mode_road`: `tra_freight_lifetime` [vkm per vehicle over its life] = utilisation rate / renewal rate |
| 14 | `processors/freight_efficiency_tech_share.py` | `freight_vehicle-efficiency_new` [MJ/km road, MJ/tkm non-road] and `freight_technology-share_new` [%] |
| 15 | `processors/freight_tech_fxa.py` | `freight_tech`: `tra_freight_technology-share_fleet` and `tra_freight_vehicle-efficiency_fleet` |

In step 14, road efficiency of new trucks equals the fleet efficiency of step 11, and the
technology share of new trucks comes from FSO first registrations (fleet split before
2005). Rail efficiency is set so that rail tkm × efficiency = EP2050+ freight rail energy
(Abb. 112), with 90 % electric and 10 % diesel and diesel using 2.7 × more energy per tkm.
IWW (0.429 MJ/tkm), marine (0.160) and aviation (19.077) are literature constants.

In step 15, the model keeps these fleet values for all historical years, so they set
historical energy directly.

---

## Phase 2: assembling the historical pickle

`processors/transport_ots_pickle.py`

This step collects the outputs of Phase 1 and arranges them into the structure the
transport module expects. Most outputs are copied as they are, but a few are derived here:

- **Total pkm per person** (`ots/pkm`): sum over modes.
- **Modal share**: pkm per person of each mode / total.
- **Occupancy**: pkm / vkm, by mode.
- **Utilisation rate**: vkm / fleet (all technologies together), by mode.
- **Technology share**: fleet and new vehicles, normalised over technologies. Private and
  public fleets are joined.
- **Biofuel blend** (`ots/fuel-mix`): share of biofuel in road and rail fuel, 0 before
  2000, 0.41 % in 2000, 5.34 % in 2023. The values come from EU JRC-IDEES, used as a Swiss
  proxy (the federal blend rule is the same in every canton).

**Aviation is merged in.** The Swiss aviation data of step 8 is added as an extra mode to
the passenger DataMatrices. Each canton gets a copy. Values per person or per vehicle
(pkm per person, occupancy, efficiency) are copied as they are. Absolute Swiss totals (new
and retired seats) are multiplied by the canton's share of the Swiss population in 2023.

It then writes `ots`, `fxa` and `constant` to `transport.pickle`.

---

## How cantons get their values

| Input | Cantonal source |
|---|---|
| pkm per person, by mode | MRMT survey (grande région split, cantonal total), see step 1 |
| vkm | canton pkm / Swiss occupancy |
| Car and motorbike fleet, new vehicles | Swiss Stats by canton (2005+). Before 2005: archive by canton (stock), Swiss total × 2005 share (new) |
| Bus, rail, tram fleet | Swiss fleet × canton share of Swiss pkm |
| Car efficiency (combustion, plug-in) | Swiss Stats CO₂ classes by canton |
| Freight tkm | GTS share of tkm starting in the canton (2016–2020) |
| Aviation absolute totals | Swiss value × 2023 population share |
| Everything else | copy of Switzerland |

---

## Output: `transport.pickle` (historical part)

```
DM_transport
├── ots
│   ├── pkm                             tra_pkm-cap [pkm/cap], all modes together
│   ├── passenger_modal-share           [mode]
│   ├── passenger_occupancy             [pkm/vkm, mode, aviation included]
│   ├── passenger_utilization-rate      [vkm/veh, mode, aviation included]
│   ├── passenger_technology-share_new  [mode × technology]
│   ├── passenger_veh-efficiency_new    [MJ/km, mode × technology]
│   ├── passenger_aviation-pkm          tra_pkm-cap [pkm/cap], residents
│   ├── fuel-mix                        tra_fuel-mix [biofuel/efuel × road/rail/aviation/IWW/marine]
│   ├── freight_tkm                     tra_freight_tkm-total-demand [bn tkm]
│   ├── freight_modal-share             [freight mode]
│   ├── freight_utilization-rate        load factor [tkm/vkm], utilisation rate [vkm/year], road
│   ├── freight_vehicle-efficiency_new  [MJ/km road, MJ/tkm non-road, mode × technology]
│   └── freight_technology-share_new    [mode × technology]
├── fxa
│   ├── passenger_tech                  fleet tech share, new vehicles, waste, fleet efficiency
│   ├── freight_mode_other              tkm per vehicle, renewal rate  [IWW, aviation, marine, rail]
│   ├── freight_mode_road               lifetime [vkm]  [HDVH, HDVL, HDVM]
│   ├── freight_tech                    fleet tech share, fleet efficiency
│   └── share-local-emissions           aviation: territorial / residents
└── constant                            cp_tra_emission-factor [g/MJ, gas × fuel], ConstantDataMatrix
```

All DataMatrices have `Country` = Switzerland and the 26 cantons (plus any other country
already in the pickle). `ots` matrices cover 1990–2023. The pickle also holds future
values and further fixed inputs used by the scenarios, which are not listed here.

---

## Data sources

| Source | Used for | Access | Cached as (in `data/`) |
|---|---|---|---|
| FSO je-f-11.04.01.02, *prestations de transport* | Swiss pkm by mode | Download | `tra_pkm_CH.xlsx` |
| FSO je-f-11.04.01.01, *prestations kilométriques* | Swiss vkm by mode | Download | `tra_vkm_CH.xlsx` |
| MRMT, daily distance by grande région (2000, 2005, 2010, 2015, 2021) | Mode split per canton | Download (2010+), local file (2000, 2005) | `tra_pkm_CH_reg_<year>.xls(x)` |
| MRMT cantonal key figures 2015 and 2021 | Canton total km per day | Download | `MRMT/2015/<canton>.xlsx`, `MRMT/su-f-11.04.03-MZ-2021-A2_Kant.xlsx` |
| Swiss Stats `DF_MFZ_0_GENERAL` | Car, motorbike and truck fleet, 2005+ | SDMX API | `tra_tot_fleet_swiss_stat.pickle`, `tra_freight_fleet_bfs.pickle` |
| Swiss Stats `DF_IVS_0_GENERAL` | New cars, motorbikes and trucks, 2005+ | SDMX API | `tra_new_fleet_swiss_stats.pickle`, `tra_freight_new_bfs.pickle` |
| STAT-TAB `px-x-1103020100_101` | Car and motorbike fleet before 2005 | Local file | `px-x-1103020100_101.csv` |
| FSO gr-e-11.03.02.02.01a | Swiss new vehicles 1990–2004 | Local file (download by hand) | `tra_new-vehicles_CH_1990-2023.csv` |
| FSO *Transports publics – séries chronologiques détaillées* | Bus, rail, tram fleet; rail freight tkm (T7.2.1) | Download | `tra_public_transport.xlsx` |
| Swiss Stats `DF_MFZ_1_EMISSION` / `DF_IVS_1_EMISSION` | CO₂ per km of the car fleet / new cars, by canton | SDMX API | `tra_veh_efficiency_swiss_stat_co2.pickle`, `tra_new-veh_efficiency_co2.pickle` |
| EP2050+ detailed results, *Verkehrssektor* | Road energy, vkm, truck fleet and new registrations | Download (zip) | `EP2050_sectors/`, `tra_EP2050_energy_demand_private.pickle`, `tra_EP2050_vkm_demand_private.pickle` |
| EP2050+ technical report, Kap. 8, Abb. 112 | Rail energy (passenger and freight); aviation energy (checks) | Download (zip) | `EP2050_tables/`, `tra_EP2050_energy_demand_rail.pickle` |
| FSO GTS heavy-vehicle survey | Road tkm (HDVH, HDVM); canton freight shares | Local files | `Freight/ts-x-11.05-GTS-E26.csv`, `Freight/su-f-11.05-GTS16-20-P19.xlsx` |
| FSO Rhine imports and exports | IWW tkm | Local file | `Freight/gr-e-11.05.04b.csv` |
| FOEN greenhouse-gas inventory (2025-04) | Truck CO₂ target; aviation CO₂ growth; emission checks | Local file | `Evolution_GHG_since_1990_2025-04.xlsx` |
| SFOE *Gesamtenergiestatistik*, T17e | Energy checks only | Download | `statistique_globale_suisse_energie.xlsx` |
| FSO civil aviation | Aviation km per resident, every 5 years | Download | `tra_aviation_CH.xlsx` |
| Our World in Data; World Bank population | World aviation pkm per person | Local file; download | `tra_global_aviation-demand.csv`, `lfs_world_population_WB.xls` |
| Aviation input tables (occupancy, utilisation, retirement, seats, efficiency, tech share, territorial pkm) | Aviation history | Local files, origin not documented in the code | `aviation_*.xlsx`, `aviation_pkm_suisse.csv` |
| IPCC 2006 Guidelines, Vol. 2, Ch. 3 | Fuel emission factors | Hard-coded | – |
| JRC-IDEES 2021 | Metro/tram efficiency; biofuel blend | Hard-coded | – |
| HBEFA 4.2, IEA *Future of Trucks*, ICCT | Truck efficiency ratios | Hard-coded | – |
| SBB, Port of Basel, FOCA; EUCalc | Non-road freight fleet sizes; renewal rates | Hard-coded | – |
