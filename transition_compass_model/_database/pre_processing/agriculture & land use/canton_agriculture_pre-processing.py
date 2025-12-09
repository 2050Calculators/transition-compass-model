import os
import pickle
import pandas as pd


###############################################################################
# Read Agriculture data
###############################################################################
def read_pickle():

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory, "../../data/datamatrix/agriculture.pickle")
    with open(f, "rb") as handle:
        DM_pickle = pickle.load(handle)

        return DM_pickle


###############################################################################
# Switzerland to Vaud : Duplicates
###############################################################################


def write_duplicate(dm_duplicate):
    DM_agriculture = read_pickle()

    # Result as tuple
    result = {}

    for path in dm_duplicate:
        # Find in DM
        source = DM_agriculture
        try:
            for key in path:
                source = source[key]
        except KeyError:
            continue

        # Build selection
        cur_dst = result
        for key in path[:-1]:  # all keys except the last
            cur_dst = cur_dst.setdefault(key, {})
        cur_dst[path[-1]] = source  # final key gets the object

    return result


def dash_to_dm(row):
    # Extract DM specs
    keys = []
    dm_0 = row["DM"]
    if pd.notna(dm_0):
        keys.append(dm_0)
    # Add Sub-DM if present
    dm_1 = row.get("Sub-DM")
    if dm_1 is not None and pd.notna(dm_1):
        keys.append(dm_1)

    # Add Sub-sub-DM if present
    dm_2 = row.get("Sub-sub-DM")
    if dm_2 is not None and pd.notna(dm_2):
        keys.append(dm_2)
    return tuple(keys)


def downscale_dashboard(canton):
    # Read the downscale dashboard
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(
        current_file_directory, "downscale/agriculture_downscale_to_canton.xlsx"
    )
    dash = pd.read_excel(f, sheet_name=canton)
    dash = dash.drop(columns=["Comments"])

    # Extract the DM specs (duplicates)
    filtered_dash = dash[dash["Downscale"] == "duplicate"]
    dm_duplicate = [dash_to_dm(row) for _, row in filtered_dash.iterrows()]
    dm_duplicate = list(dict.fromkeys(dm_duplicate))

    # Extract the DM specs (discard)
    # Extract the DM specs (data)

    return dm_duplicate


###############################################################################
# Switzerland to Canton : Constsnt
###############################################################################


def import_constant():
    DM_agriculture = read_pickle()
    DM_constant = {}
    DM_constant["constant"] = DM_agriculture["constant"]

    return DM_constant


###############################################################################
# Main
###############################################################################

canton = "Vaud"
dm_duplicate = downscale_dashboard(canton)

# Duplicates
DM_duplicates = write_duplicate(dm_duplicate)
DM_constant = import_constant()


DM_pickle = read_pickle()
print("Hello")
