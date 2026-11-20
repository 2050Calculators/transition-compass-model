"""vkm-by-technology adjustment factor (Switzerland, applied to all cantons).

At run time, passenger vkm by technology is computed as
    vkm by tech = vkm by mode x fleet share by tech
which assumes that every vehicle of a mode drives the same distance per year,
whatever its technology. EP2050 shows this is not true (e.g. diesel cars drive
more km per year than petrol cars).

This processor computes, for Switzerland, one constant factor per mode and
technology:
    f[mode, tech] = mean over 1990-2015 of ( EP2050 vkm share / model fleet share )
Shares are computed only over the technologies that EP2050 covers (e.g. EP2050
has no trolleybus, so bus CEV is left out of the comparison).

At run time (model/transport/workflows.py) the fleet share is multiplied by f and
renormalised per mode, so total vkm by mode is unchanged and only the split
between technologies moves. Modes / technologies without usable EP2050 data get
f = 1. The same Swiss factor is copied to every canton.
"""

import os
import pickle

import numpy as np

from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

VAR_NAME = "tra_passenger_vkm-adj-factor"
MODES = ["2W", "LDV", "bus"]  # road passenger modes covered by EP2050
CALIB_YEARS = create_years_list(1990, 2015, 1)
MIN_SHARE = 0.01  # skip years where a tech has < 1 % share (unstable ratio)
MIN_VALID_YEARS = 5  # need at least this many usable years, else f = 1


def compute_yearly_factor(dm_fleet_share_CH, dm_vkm_ep2050):
    """Return (techs, arr_ratio[year, mode, tech], arr_valid[year, mode, tech])."""
    techs = sorted(
        set(dm_vkm_ep2050.col_labels["Categories2"]).intersection(
            dm_fleet_share_CH.col_labels["Categories2"]
        )
    )
    sel = {"Years": CALIB_YEARS, "Categories1": MODES, "Categories2": techs}
    dm_ep = dm_vkm_ep2050.filter({"Country": ["Switzerland"], **sel})
    dm_m = dm_fleet_share_CH.filter({"Country": ["Switzerland"], **sel})
    # shares over the common technologies only
    dm_ep.normalise("Categories2")
    dm_m.normalise("Categories2")
    s_ep = np.nan_to_num(dm_ep.array[0, :, 0, :, :])
    s_m = np.nan_to_num(dm_m.array[0, :, 0, :, :])
    valid = (s_ep >= MIN_SHARE) & (s_m >= MIN_SHARE)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(valid, s_ep / s_m, np.nan)
    return techs, ratio, valid


def print_variability(techs, ratio, valid, f_const):
    print("\nvkm adjustment factor by mode x tech, from yearly values 1990-2015")
    print(
        f"{'mode':5s} {'tech':14s} {'n_yrs':>5s} {'mean':>6s} {'min':>6s} "
        f"{'max':>6s} {'CV%':>5s} {'1990':>6s} {'2015':>6s}  used"
    )
    for i, m in enumerate(MODES):
        for j, t in enumerate(techs):
            n = int(valid[:, i, j].sum())
            if n == 0:
                continue
            r = ratio[:, i, j]
            cv = 100 * np.nanstd(r) / np.nanmean(r)
            print(
                f"{m:5s} {t:14s} {n:5d} {np.nanmean(r):6.3f} {np.nanmin(r):6.3f} "
                f"{np.nanmax(r):6.3f} {cv:5.1f} {r[0]:6.3f} {r[-1]:6.3f}  "
                f"{f_const[i, j]:.3f}"
            )


def run(dm_fleet_share_CH, dm_vkm_ep2050, years_ots, years_fts, country_list):
    """
    dm_fleet_share_CH: model fleet share by tech (Categories1 mode, Categories2 tech),
                       Switzerland, must cover 1990-2015.
    dm_vkm_ep2050: EP2050 vkm by mode x tech, Switzerland.
    Returns a DataMatrix with VAR_NAME [-], constant over years_ots + years_fts,
    copied to every country in country_list.
    """
    techs, ratio, valid = compute_yearly_factor(dm_fleet_share_CH, dm_vkm_ep2050)
    n_valid = valid.sum(axis=0)
    f_const = np.where(n_valid >= MIN_VALID_YEARS, np.nanmean(ratio, axis=0), 1.0)
    f_const = np.nan_to_num(f_const, nan=1.0)
    print_variability(techs, ratio, valid, f_const)

    years = list(years_ots) + list(years_fts)
    dm = DataMatrix(
        col_labels={
            "Country": list(country_list),
            "Years": years,
            "Variables": [VAR_NAME],
            "Categories1": MODES,
            "Categories2": techs,
        },
        units={VAR_NAME: "-"},
    )
    dm.array[...] = f_const[np.newaxis, np.newaxis, np.newaxis, :, :]
    return dm


if __name__ == "__main__":
    # Standalone: compute the factor from the fleet share already stored in
    # transport.pickle and add it to transport.pickle as fxa.
    from transition_compass_model._database.pre_processing.params import (
        country_list,
        years_fts,
        years_ots,
    )
    from transition_compass_model.model.common.auxiliary_functions import (
        my_pickle_dump,
    )

    this_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(this_dir, "../../../../data/datamatrix/transport.pickle")
    with open(pickle_file, "rb") as handle:
        DM_transport = pickle.load(handle)
    dm_share = DM_transport["fxa"]["passenger_tech"].filter(
        {
            "Country": ["Switzerland"],
            "Variables": ["tra_passenger_technology-share_fleet"],
        }
    )
    with open(
        os.path.join(this_dir, "../data/tra_EP2050_vkm_demand_private.pickle"), "rb"
    ) as handle:
        dm_ep = pickle.load(handle)

    dm_factor = run(dm_share, dm_ep, years_ots, years_fts, country_list)
    my_pickle_dump(
        DM_new={"fxa": {"passenger_vkm-adj-factor": dm_factor}},
        local_pickle_file=pickle_file,
    )
