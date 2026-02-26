# TransitionCompassModel — Multi-sector climate pathway calculation engine

Python library that computes emission, energy, and environmental indicator pathways sector by sector, given policy lever settings. Powers the [TransitionCompassViz](https://github.com/EPFL-ENAC/leure-speed-to-zero) visualization platform.

## Live Platforms

- **Production**: [https://transition-compass.epfl.ch/](https://transition-compass.epfl.ch/)
- **Development**: [https://transition-compass-dev.epfl.ch/](https://transition-compass-dev.epfl.ch/)

## Architecture

The model is structured around independent sector modules coordinated by an interaction runner:

```
transition_compass_model/
├── model/
│   ├── interactions.py          # Runs all sectors and resolves cross-sector dependencies
│   ├── agriculture_module.py
│   ├── ammonia_module.py
│   ├── buildings_module.py
│   ├── buildings/               # Sub-modules for buildings sector
│   ├── climate_module.py
│   ├── district_heating_module.py
│   ├── emissions_module.py
│   ├── energy_module.py         # EnergyScope LP (Pyomo/AMPL)
│   ├── energy/                  # Sub-modules for energy sector
│   ├── forestry_module.py
│   ├── industry_module.py
│   ├── landuse_module.py
│   ├── lca_module.py
│   ├── lifestyles_module.py
│   ├── minerals_module.py
│   ├── oilrefinery_module.py
│   ├── power_module.py
│   ├── transport_module.py
│   └── transport/               # Sub-modules for transport sector
└── _database/
    └── data/
        └── datamatrix/          # Regional data matrices (Vaud, Switzerland, EU27)
```

**Sectors**: Agriculture, Ammonia, Buildings, Climate, District Heating, Emissions, Energy (LP), Forestry, Industry, Land Use, LCA, Lifestyles, Minerals, Oil Refinery, Power, Transport

Given a set of policy lever values (integers 1–4 representing ambition levels), `interactions.py` runs the relevant sector modules and returns time-series results for emissions, energy demand, and environmental indicators.

## Install as a package

```bash
pip install git+https://github.com/EPFL-ENAC/transition-compass-model.git
```

Or with a specific tag:

```bash
pip install git+https://github.com/EPFL-ENAC/transition-compass-model.git@v1.0.0
```

## Local development

See **[DEVELOPMENT.md](DEVELOPMENT.md)** for:

- Standalone install with `uv`
- Working alongside the app in editable mode
- Switching between local and remote model versions
- Tagging a release and triggering automated deployment
