import os
import pickle


def run(DM_input, years_ots):

    # directories
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    DM = {"ots": dict()}
    DM["ots"]["footprint"] = {}

    for key in DM_input.keys():
        DM["ots"]["footprint"][key] = DM_input[key].copy()

    # save intermediate
    f = os.path.join(current_file_directory, "../data/datamatrix/lca_ots.pickle")
    with open(f, "wb") as handle:
        pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return DM
