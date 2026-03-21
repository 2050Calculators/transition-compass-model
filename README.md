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
pip install git+https://github.com/2050Calculators/transition-compass-model.git
```

Or pin to a specific release tag:

```bash
pip install git+https://github.com/2050Calculators/transition-compass-model.git@v1.0.0
```

## Usage

Once installed, the typical workflow is:

1. Load lever settings (policy ambition levels 1–4 per sector)
2. Load input data matrices from the database pickles, filtered to your countries of interest
3. Call `runner()` with the lever settings, time horizon, input data, and sector list
4. Use the returned `output` (time-series `DataMatrix`) and `KPI` dict for analysis

```python
import logging
from transition_compass_model.model.common.config_loader import load_lever_config
from transition_compass_model.model.common.auxiliary_functions import (
    filter_country_and_load_data_from_pickles,
)
from transition_compass_model.model.interactions import runner

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Policy lever settings (loaded from config/lever_position.json)
lever_setting = load_lever_config()

# Time horizon: [start_ots, end_ots, start_fts, end_fts, fts_step]
years_setting = [1990, 2023, 2025, 2050, 5]

country_list = ["Switzerland", "EU27", "Vaud"]

sectors = [
    "climate",
    "lifestyles",
    "buildings",
    "transport",
    "industry",
    "forestry",
    "agriculture",
    "ammonia",
    "lca",
]

# Load and filter input data matrices from the bundled database pickles
DM_input = filter_country_and_load_data_from_pickles(
    country_list=country_list, modules_list=sectors
)

# Run the model
output, KPI = runner(lever_setting, years_setting, DM_input, sectors, logger)
```

To customise lever settings, edit `transition_compass_model/config/lever_position.json` (each value is an integer 1–4).

## Local development

See **[DEVELOPMENT.md](DEVELOPMENT.md)** for:

- Standalone install with `uv`
- Running the model and individual sector modules from the terminal or an IDE (VS Code, Spyder, PyCharm)
- Working alongside the app in editable mode
- Switching between local and remote model versions
- Tagging a release and triggering automated deployment
