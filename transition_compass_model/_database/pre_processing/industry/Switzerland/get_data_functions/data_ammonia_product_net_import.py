import pandas as pd
import os
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def get_ammonia_trade_data(current_file_directory):

    # get data
    filepath = os.path.join(
        current_file_directory, "../data/ammonia/FAOSTAT_data_en_8-22-2025.csv"
    )
    df = pd.read_csv(filepath)
    df.columns
    df["Item"].unique()
    df = df.loc[:, ["Element", "Item", "Year", "Unit", "Value"]]

    # aggregate
    fao_mapping = {
        # Ammonia (raw material)
        "Ammonia, anhydrous": "ammonia",
        # fertilizers
        "Ammonium nitrate (AN)": "fertilizer",
        "Ammonium sulphate": "fertilizer",
        "Calcium ammonium nitrate (CAN) and other mixtures with calcium carbonate": "fertilizer",
        "Diammonium phosphate (DAP)": "fertilizer",
        "Monoammonium phosphate (MAP)": "fertilizer",
        "NPK fertilizers": "fertilizer",
        "Other nitrogenous fertilizers, n.e.c.": "fertilizer",
        "Other NP compounds": "fertilizer",
        "Urea": "fertilizer",
        "Urea and ammonium nitrate solutions (UAN)": "fertilizer",
        # Not ammonia-based (others)
        "Fertilizers n.e.c.": "other",
        "Other phosphatic fertilizers, n.e.c.": "other",
        "Other potassic fertilizers, n.e.c.": "other",
        "Phosphate rock": "other",
        "PK compounds": "other",
        "Potassium chloride (muriate of potash) (MOP)": "other",
        "Potassium nitrate": "other",  # can involve ammonia indirectly, but generally treated as potassic
        "Potassium sulphate (sulphate of potash) (SOP)": "other",
        "Sodium nitrate": "other",
        "Superphosphates above 35%": "other",
        "Superphosphates, other": "other",
    }
    for key in fao_mapping.keys():
        df.loc[df["Item"] == key, "Item"] = fao_mapping[key]
    df = df.groupby(["Element", "Item", "Year", "Unit"], as_index=False)["Value"].agg(
        "sum"
    )
    df["Item"] = df["Element"] + "_" + df["Item"] + "[" + df["Unit"] + "]"
    df = df.loc[:, ["Item", "Year", "Value"]]
    df = df.pivot(index=["Year"], columns="Item", values="Value").reset_index()
    df["Country"] = "Switzerland"
    df.rename(columns={"Year": "Years"}, inplace=True)
    dm = DataMatrix.create_from_df(df, 1)
    dm.rename_col(
        ["Export quantity", "Import quantity"], ["export", "import"], "Variables"
    )
    dm.drop("Categories1", "other")

    return dm
