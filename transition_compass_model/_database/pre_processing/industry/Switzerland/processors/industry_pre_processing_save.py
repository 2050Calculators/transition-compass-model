import pickle
import os


def run(DM_input):

    # directories
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # save
    f = os.path.join(
        current_file_directory, "../data/datamatrix/industry_pre_processing.pickle"
    )
    with open(f, "wb") as handle:
        pickle.dump(DM_input, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return
