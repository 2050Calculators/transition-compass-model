import os
import pickle

from transition_compass_model.model.common.auxiliary_functions import create_years_list


def run(years_ots, years_fts):
    """Return energy-demand FXA for Vaud — same per-unit intensities as CH.

    Vaud industry uses the same production technologies as Switzerland overall,
    so energy intensity (TWh/Mt of material) is identical. Vaud-specific production
    volumes are handled separately in fxa/prod.
    """
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    industry_pickle = os.path.join(
        current_file_directory, "../../../../data/datamatrix/industry.pickle"
    )
    with open(industry_pickle, "rb") as f:
        DM_ch = pickle.load(f)

    dm_excl_ch = (
        DM_ch["fxa"]["energy-demand-excl-feedstock"]
        .filter({"Country": ["Switzerland"]})
        .copy()
    )
    dm_excl_ch.rename_col("Switzerland", "Vaud", dim="Country")

    dm_feed_ch = (
        DM_ch["fxa"]["energy-demand-feedstock"]
        .filter({"Country": ["Switzerland"]})
        .copy()
    )
    dm_feed_ch.rename_col("Switzerland", "Vaud", dim="Country")

    ammonia_pickle = os.path.join(
        current_file_directory, "../../../../data/datamatrix/ammonia.pickle"
    )
    with open(ammonia_pickle, "rb") as f:
        DM_amm = pickle.load(f)

    dm_excl_amm = (
        DM_amm["fxa"]["energy-demand-excl-feedstock"]
        .filter({"Country": ["Switzerland"]})
        .copy()
    )
    dm_excl_amm.rename_col("Switzerland", "Vaud", dim="Country")

    dm_feed_amm = (
        DM_amm["fxa"]["energy-demand-feedstock"]
        .filter({"Country": ["Switzerland"]})
        .copy()
    )
    dm_feed_amm.rename_col("Switzerland", "Vaud", dim="Country")

    return {
        "industry": {
            "energy-demand-excl-feedstock": dm_excl_ch,
            "energy-demand-feedstock": dm_feed_ch,
        },
        "ammonia": {
            "energy-demand-excl-feedstock": dm_excl_amm,
            "energy-demand-feedstock": dm_feed_amm,
        },
    }


if __name__ == "__main__":
    years_ots = create_years_list(1990, 2023, 1)
    years_fts = create_years_list(2025, 2050, 5)
    run(years_ots, years_fts)
