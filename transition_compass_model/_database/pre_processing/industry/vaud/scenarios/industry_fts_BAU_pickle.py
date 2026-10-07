import os
import pickle

from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
    linear_fitting,
    my_pickle_dump,
    sort_pickle,
)


def _make_fts_flat(DM, name, years_fts, based_on):
    """Extrapolate OTS lever flat into FTS; store 4 identical levels (BAU)."""
    dm = DM["ots"][name].copy()
    dm = linear_fitting(dm, years_fts, based_on=based_on)
    DM["fts"][name] = {}
    for level in range(1, 5):
        DM["fts"][name][level] = dm.filter({"Years": years_fts})


def run(DM_industry, DM_ammonia, country_list, years_ots, years_fts):
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # Load CH pickles for policy-scenario levers
    industry_pickle_file = os.path.join(
        current_file_directory, "../../../../data/datamatrix/industry.pickle"
    )
    with open(industry_pickle_file, "rb") as handle:
        DM_ch = pickle.load(handle)

    ammonia_pickle_file = os.path.join(
        current_file_directory, "../../../../data/datamatrix/ammonia.pickle"
    )
    with open(ammonia_pickle_file, "rb") as handle:
        DM_amm_ch = pickle.load(handle)

    # ---- Industry FTS ----
    DM_industry["fts"] = {}

    # Levers held flat from OTS end year (same as CH approach)
    for name in [
        "product-net-import",
        "material-net-import",
        "paperpack",
        "eol-waste-management",
    ]:
        _make_fts_flat(DM_industry, name, years_fts, based_on=[2023])

    # Policy-scenario levers: copy FTS levels from CH (all 4 ambition levels)
    other_levers = [
        "material-switch",
        "material-efficiency",
        "technology-development",
        "cc",
        "technology-share",
        "energy-carrier-mix",
        "eol-material-recovery",
    ]
    for lev in other_levers:
        DM_industry["fts"][lev] = {}
        for level in range(1, 5):
            dm_temp = (
                DM_ch["fts"][lev][level].filter({"Country": ["Switzerland"]}).copy()
            )
            dm_temp.rename_col("Switzerland", "Vaud", dim="Country")
            DM_industry["fts"][lev][level] = dm_temp

    # Save Vaud industry data into the main industry.pickle
    my_pickle_dump(DM_new=DM_industry, local_pickle_file=industry_pickle_file)
    sort_pickle(industry_pickle_file)

    # ---- Ammonia FTS ----
    DM_ammonia["fts"] = {}

    amm_flat_levers = ["product-net-import", "material-net-import"]
    for name in amm_flat_levers:
        _make_fts_flat(DM_ammonia, name, years_fts, based_on=[2023])

    amm_other_levers = [
        "material-efficiency",
        "eol-material-recovery",
        "technology-development",
        "cc",
        "energy-carrier-mix",
    ]
    for lev in amm_other_levers:
        DM_ammonia["fts"][lev] = {}
        for level in range(1, 5):
            dm_temp = (
                DM_amm_ch["fts"][lev][level].filter({"Country": ["Switzerland"]}).copy()
            )
            dm_temp.rename_col("Switzerland", "Vaud", dim="Country")
            DM_ammonia["fts"][lev][level] = dm_temp

    # Save Vaud ammonia data into the main ammonia.pickle
    my_pickle_dump(DM_new=DM_ammonia, local_pickle_file=ammonia_pickle_file)
    sort_pickle(ammonia_pickle_file)

    return DM_industry, DM_ammonia


if __name__ == "__main__":
    years_ots = create_years_list(1990, 2023, 1)
    years_fts = create_years_list(2025, 2050, 5)

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    for fname in ["industry_ots.pickle", "ammonia_ots.pickle"]:
        path = os.path.join(current_file_directory, f"../data/datamatrix/{fname}")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{fname} not found — run industry_preprocessing_main_VD.py first"
            )

    with open(
        os.path.join(current_file_directory, "../data/datamatrix/industry_ots.pickle"),
        "rb",
    ) as f:
        DM_industry = pickle.load(f)
    with open(
        os.path.join(current_file_directory, "../data/datamatrix/ammonia_ots.pickle"),
        "rb",
    ) as f:
        DM_ammonia = pickle.load(f)

    run(DM_industry, DM_ammonia, ["Vaud"], years_ots, years_fts)
