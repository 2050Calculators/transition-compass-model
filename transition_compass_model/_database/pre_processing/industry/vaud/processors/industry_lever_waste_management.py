import os
import pickle

from transition_compass_model.model.common.auxiliary_functions import create_years_list


def run(years_ots):
    """Return EOL waste-management shares for Vaud — same as CH (national regulations apply)."""
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    pickle_file = os.path.join(
        current_file_directory, "../../../../data/datamatrix/industry.pickle"
    )
    with open(pickle_file, "rb") as handle:
        DM_ch = pickle.load(handle)

    dm_waste = (
        DM_ch["ots"]["eol-waste-management"].filter({"Country": ["Switzerland"]}).copy()
    )
    dm_waste.rename_col("Switzerland", "Vaud", dim="Country")

    return dm_waste


if __name__ == "__main__":
    years_ots = create_years_list(1990, 2023, 1)
    run(years_ots)
