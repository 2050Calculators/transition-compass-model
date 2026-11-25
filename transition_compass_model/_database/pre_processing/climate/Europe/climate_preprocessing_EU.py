import os

import numpy as np
from _database.pre_processing.api_routine_Eurostat import get_data_api_eurostat

from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    linear_forecast_BAU_w_noise,
    my_pickle_dump,
)

years_ots = create_years_list(start_year=1990, end_year=2023, step=1)
years_fts = create_years_list(start_year=2025, end_year=2050, step=5)

# Fetch EU27 aggregate HDD/CDD from Eurostat (nrg_chdd_a).
# EU27_2020 is the Eurostat code for the EU-27 aggregate; it maps to "EU27" via
# the internal dict_iso2 in get_data_api_eurostat.
filter_dict = {"geo\\TIME_PERIOD": ["EU27_2020"]}
mapping_dim = {"Country": "geo\\TIME_PERIOD", "Variables": "indic_nrg"}
dm_hdd_cdd = get_data_api_eurostat(
    "nrg_chdd_a", filter_dict, mapping_dim, unit="daysK", years=years_ots
)
dm_hdd_cdd.rename_col(["CDD", "HDD"], ["clm_CDD", "clm_HDD"], dim="Variables")

# Forecast FTS with a linear BAU projection (all 4 levels identical for climate)
dm_hdd_cdd_fts = linear_forecast_BAU_w_noise(
    dm_hdd_cdd, start_t=1990, years_ots=years_ots, years_fts=years_fts
)
dm_hdd_cdd_fts.array = np.maximum(0, dm_hdd_cdd_fts.array)
dm_hdd_cdd.append(dm_hdd_cdd_fts, dim="Years")

# Add days-below-15 and days-above-24 as zeros — Eurostat does not provide these.
# Buildings formula: HDD_adj = HDD + N_below × (Tint − 18), so N=0 → HDD_adj = HDD_ref.
zeros = np.zeros_like(dm_hdd_cdd[:, :, "clm_CDD"])
dm_hdd_cdd.add(
    zeros, dim="Variables", col_label="clm_days-below-15", dummy=True, unit="days"
)
dm_hdd_cdd.add(
    zeros, dim="Variables", col_label="clm_days-above-24", dummy=True, unit="days"
)

# AC uptake: uptake = 0.815 × (1 − e^(−0.00225 × CDD))
# https://www.sciencedirect.com/science/article/pii/S0306261921014756
arr_uptake = 0.815 * (1 - np.exp(-0.00225 * dm_hdd_cdd[:, :, "clm_CDD"]))
dm_hdd_cdd.add(
    arr_uptake, dim="Variables", col_label="clm_AC-uptake", dummy=True, unit="%"
)

# Split into OTS and FTS
dm_ots = dm_hdd_cdd.filter({"Years": years_ots})
dm_fts = dm_hdd_cdd.filter({"Years": years_fts})

# Build pickle structure matching climate.pickle layout
DM_climate_new = {"ots": {"temp": dm_ots}, "fts": {"temp": {}}}
for lev in range(1, 5):
    DM_climate_new["fts"]["temp"][lev] = dm_fts.copy()

# Merge EU27 into climate.pickle (preserves existing CH/canton data)
current_file_directory = os.path.dirname(os.path.abspath(__file__))
pickle_file = os.path.join(
    current_file_directory, "../../../data/datamatrix/climate.pickle"
)
my_pickle_dump(DM_climate_new, pickle_file)

print("Hello")
