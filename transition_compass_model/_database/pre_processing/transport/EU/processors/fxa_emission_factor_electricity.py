import os
import pickle
import warnings

import numpy as np

from transition_compass_model.model.common.data_matrix_class import DataMatrix

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(
    _THIS_DIR, "../data/datamatrix/fxa_emission_factor_electricity.pickle"
)

# Country list from old script (transport_fxa_emission-factor-electricity.py)
_COUNTRIES = [
    "Belgium",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Denmark",
    "EU27",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Ireland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "Poland",
    "Portugal",
    "Romania",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
    "United Kingdom",
]


def run(years_ots: list, years_fts: list):
    """
    Build electricity emission-factor FXA for all EU countries.

    The old EU script (transport_fxa_emission-factor-electricity.py) loaded these values
    from the CH transport pickle and copied them to all EU countries. Since the CH pickle
    is not available in the new preprocessing flow, the same constants are reproduced
    inline (identical to CH electricity_emissions_pipeline.py):
    - CO2: 31.1 g/MJ for OTS (1990–2023), linearly interpolated to 0 at 2050
    - CH4, N2O: 0 g/MJ throughout

    No residency/territorial distinction applies — emission factors are technology
    properties, not demand-based.

    Parameters
    ----------
    years_ots, years_fts : list

    Returns
    -------
    dm_emifact : DataMatrix
        Variable='tra_emission-factor', Cat1=[CH4, CO2, N2O], Cat2=[electricity].
        Countries: Austria + _COUNTRIES (29 total).
    """
    if os.path.exists(_CACHE_PATH):
        print("  (cached)")
        with open(_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    # Rebuild dm_emifact_ch — values the old script would have read from the CH pickle
    # (same constants as CH electricity_emissions_pipeline.py)
    dm_emifact_ch = DataMatrix(
        col_labels={
            "Country": ["Switzerland"],
            "Years": years_ots + years_fts,
            "Variables": ["tra_emission-factor"],
            "Categories1": ["CH4", "CO2", "N2O"],
            "Categories2": ["electricity"],
        },
        units={"tra_emission-factor": "g/MJ"},
    )
    arr = dm_emifact_ch.array
    arr[:] = 0  # CH script used np.zeros as base; CH4 and N2O stay 0
    idx = dm_emifact_ch.idx
    arr[:, idx[1990] : idx[2023] + 1, 0, idx["CO2"], 0] = 31.1
    arr[:, idx[2025] : idx[2050], 0, idx["CO2"], 0] = np.nan
    arr[:, idx[2050], 0, idx["CO2"], 0] = 0
    dm_emifact_ch.array = arr
    dm_emifact_ch.fill_nans(dim_to_interp="Years")

    # Copy to all EU countries — verbatim from old script
    dm_temp = dm_emifact_ch.copy()
    dm_temp.rename_col("Switzerland", "Austria", "Country")
    dm_emifact = dm_temp.copy()
    for c in _COUNTRIES:
        dm_temp = dm_emifact_ch.copy()
        dm_temp.rename_col("Switzerland", c, "Country")
        dm_emifact.append(dm_temp, "Country")

    ################
    ##### SAVE #####
    ################

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(dm_emifact, f)
    return dm_emifact


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    from params import years_fts, years_ots

    run(years_ots, years_fts)
