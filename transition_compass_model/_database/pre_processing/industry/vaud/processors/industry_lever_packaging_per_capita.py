import os
import pickle

from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    load_pop,
)


def run(dm_pop_ots, years_ots):
    """Return per-capita packaging demand for Vaud — same as CH (national standards)."""
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    pickle_file = os.path.join(
        current_file_directory, "../../../../data/datamatrix/industry.pickle"
    )
    with open(pickle_file, "rb") as handle:
        DM_ch = pickle.load(handle)

    dm_pack = DM_ch["ots"]["paperpack"].filter({"Country": ["Switzerland"]}).copy()
    dm_pack.rename_col("Switzerland", "Vaud", dim="Country")

    return dm_pack


if __name__ == "__main__":
    years_ots = create_years_list(1990, 2023, 1)
    dm_pop_ots = load_pop(["Vaud"], years_list=years_ots)
    run(dm_pop_ots, years_ots)
