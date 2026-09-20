"""Blue food (Swiss domestic freshwater fish) inputs for the TCAF LCA workflow.

Adds three fixed-assumption datamatrices to `_database/data/datamatrix/TCAF.pickle`
under `fxa["fish"]`. TCAF_preprocessing.py rebuilds the whole pickle from raw LCIA
files that are not in this repo, so this script extends the existing pickle
instead. Re-running it overwrites the same three keys and nothing else.

Scope: Swiss DOMESTIC production of `ffish` (FAOSTAT FBS "Freshwater Fish", which
holds the FishStatJ freshwater and diadromous finfish, i.e. ~99% of Swiss fish
production). Imports are not covered.

  fxa["fish"]["lca-ch"]       impacts [impact-unit / kg live weight] per method
                              (aquaculture, capture) x 14 ReCiPe categories
  fxa["fish"]["ssr"]          self-sufficiency ratio, FBS production / (food+feed+processed)
  fxa["fish"]["capture-ref"]  wild-catch tonnage [t live weight] (FishStatJ)

Inputs (all in data/blue-food/, derived from the TCAF_fisheries_aquaculture project
and FAOSTAT / FishStatJ; see docs/blue-food-lca.md in the app repo):
  fish_lca_processes_per_kg_whole.csv   physical ReCiPe impacts of the LCA processes
  fish_ch_production_fishstatj.csv      Swiss production by ISSCAAP group and source
  fish_ch_fbs_freshwater-fish.csv       Swiss FBS balance of Freshwater Fish

Run from this folder:  python TCAF_fish_preprocessing.py
"""

import os
import pickle

import numpy as np
import pandas as pd

from tcaf_model.model.common.data_matrix_class import DataMatrix

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "blue-food")
PICKLE = os.path.normpath(os.path.join(HERE, "../../data/datamatrix/TCAF.pickle"))

FOOD = "ffish"
METHODS = ["aquaculture", "capture"]

# Years averaged for the Swiss production mix that weights the LCA, and for the
# values held constant over the forecast years (FBS is rounded to 1000 t from 2014,
# so a single year is noisy).
MIX_YEARS = range(2018, 2023)
HOLD_YEARS = range(2019, 2024)

# Capture proxy: Swiss lake fisheries are small-boat passive-gear fisheries with no
# feed. The only passive-gear finfish process in the fisheries LCA is the trammel
# net one, so it stands for all wild capture (single-process proxy).
CAPTURE_PROXY_GEAR = "Trammel net"


def _impacts(dm_lca_ch):
    return list(dm_lca_ch.col_labels["Categories3"])


def aquaculture_lca(processes, production, impacts):
    """Production-weighted aquaculture impact vector [impact-unit/kg live weight].

    Per Swiss ISSCAAP group (same rule as tcaf_total_assemble.extrapolate_division
    in the fisheries project): the mean of the group's own aquaculture processes if
    there are any (observed), else the unweighted mean of the observed groups of the
    same division (division_pooled). Groups are then weighted by their mean Swiss
    aquaculture tonnage over MIX_YEARS.
    Returns (vector, basis) with `basis` one row per Swiss group for the audit CSV.
    """
    aq = processes[processes["production_env"] == "Aquaculture"]
    group_mean = aq.groupby(["division", "isscaap_group"])[impacts].mean()

    swiss = production[
        (production["source"] == "aquaculture") & production["year"].isin(MIX_YEARS)
    ]
    weight = (
        swiss.groupby(["division", "isscaap_group"])["tonnes_live_weight"]
        .mean()
        .loc[lambda s: s > 0]
    )

    rows, vecs, w = [], [], []
    for (division, group), tonnes in weight.items():
        if (division, group) in group_mean.index:
            vec = group_mean.loc[(division, group)]
            source = "observed"
            n = int((aq["isscaap_group"] == group).sum())
        elif division in group_mean.index.get_level_values(0):
            vec = group_mean.loc[division].mean()
            source = "division_pooled"
            n = int(len(group_mean.loc[division]))
        else:
            raise ValueError(f"no aquaculture LCA for division {division!r}")
        vecs.append(vec.values.astype(float))
        w.append(tonnes)
        rows.append(
            {
                "method": "aquaculture",
                "isscaap_group": group,
                "division": division,
                "swiss_tonnes_mean_2018_2022": round(float(tonnes), 1),
                "lca_source": source,
                "n_groups_or_processes": n,
                "global-warming_kgCO2e_per_kg": round(float(vec["global-warming"]), 3),
            }
        )
    vector = np.average(np.array(vecs), axis=0, weights=np.array(w))
    return vector, pd.DataFrame(rows)


def capture_lca(processes, impacts):
    proxy = processes[processes["lca_gear"] == CAPTURE_PROXY_GEAR]
    if proxy.empty:
        raise ValueError(f"no capture process with gear {CAPTURE_PROXY_GEAR!r}")
    vector = proxy[impacts].mean().values.astype(float)
    basis = pd.DataFrame(
        [
            {
                "method": "capture",
                "isscaap_group": "all Swiss wild capture",
                "division": "",
                "swiss_tonnes_mean_2018_2022": np.nan,
                "lca_source": f"proxy: {CAPTURE_PROXY_GEAR}",
                "n_groups_or_processes": len(proxy),
                "global-warming_kgCO2e_per_kg": round(
                    float(proxy["global-warming"].mean()), 3
                ),
            }
        ]
    )
    return vector, basis


def ssr_series(fbs):
    """FBS self-sufficiency: production / (food + feed + processed), the same
    formula as the model's livestock SSR (livestock_preprocessing.py)."""
    f = fbs.fillna({"feed_kt": 0.0, "processed_kt": 0.0})
    ssr = f["production_kt"] / (f["food_kt"] + f["feed_kt"] + f["processed_kt"])
    return pd.Series(ssr.values, index=f["year"].astype(int))


def capture_series(production):
    cap = production[production["source"] == "capture"]
    return cap.groupby("year")["tonnes_live_weight"].sum()


def on_model_years(ots, years):
    """Value per model year: observed for years in `ots`, mean of HOLD_YEARS after."""
    hold = float(ots.loc[list(HOLD_YEARS)].mean())
    return np.array(
        [float(ots.loc[int(y)]) if int(y) in ots.index else hold for y in years]
    )


def build(dm_lca_ch):
    years = list(dm_lca_ch.col_labels["Years"])
    impacts = _impacts(dm_lca_ch)

    processes = pd.read_csv(os.path.join(DATA, "fish_lca_processes_per_kg_whole.csv"))
    production = pd.read_csv(os.path.join(DATA, "fish_ch_production_fishstatj.csv"))
    fbs = pd.read_csv(os.path.join(DATA, "fish_ch_fbs_freshwater-fish.csv"))

    v_aq, basis_aq = aquaculture_lca(processes, production, impacts)
    v_cap, basis_cap = capture_lca(processes, impacts)
    pd.concat([basis_aq, basis_cap]).to_csv(
        os.path.join(DATA, "fish_lca_ch_ffish_basis.csv"), index=False
    )

    # (Country, Years, Variables, Categories1 food, Categories2 method, Categories3 impact)
    arr = np.tile(
        np.stack([v_aq, v_cap])[np.newaxis, np.newaxis, np.newaxis, np.newaxis],
        (1, len(years), 1, 1, 1, 1),
    )
    dm_lca = DataMatrix.based_on(
        arr, format=dm_lca_ch, change={"Categories1": [FOOD], "Categories2": METHODS}
    )

    flat = {
        "Categories2": None,
        "Categories3": None,
    }
    ssr = on_model_years(ssr_series(fbs), years)
    dm_ssr = DataMatrix.based_on(
        ssr[np.newaxis, :, np.newaxis, np.newaxis],
        format=dm_lca_ch,
        change={"Variables": ["fish_ssr"], "Categories1": [FOOD], **flat},
        units={"fish_ssr": "-"},
    )
    cap = on_model_years(capture_series(production), years)
    dm_cap = DataMatrix.based_on(
        cap[np.newaxis, :, np.newaxis, np.newaxis],
        format=dm_lca_ch,
        change={"Variables": ["fish_capture-ref"], "Categories1": [FOOD], **flat},
        units={"fish_capture-ref": "t"},
    )
    return {"lca-ch": dm_lca, "ssr": dm_ssr, "capture-ref": dm_cap}, (
        v_aq,
        v_cap,
        impacts,
    )


def main():
    with open(PICKLE, "rb") as handle:
        DM_TCAF = pickle.load(handle)

    dm_fish, (v_aq, v_cap, impacts) = build(DM_TCAF["fxa"]["lca"]["lca-switzerland"])
    DM_TCAF["fxa"]["fish"] = dm_fish

    with open(PICKLE, "wb") as handle:
        pickle.dump(DM_TCAF, handle, protocol=pickle.HIGHEST_PROTOCOL)

    i = impacts.index("global-warming")
    print(f"written {PICKLE}")
    print(
        f"global-warming [kg CO2e / kg live weight]: aquaculture {v_aq[i]:.2f}, capture {v_cap[i]:.2f}"
    )
    ssr = dm_fish["ssr"]
    yrs = list(ssr.col_labels["Years"])
    print(
        "SSR 1990 / 2010 / 2023 / 2050:",
        [
            round(float(ssr.array[0, yrs.index(y), 0, 0]), 4)
            for y in (1990, 2010, 2023, 2050)
        ],
    )


if __name__ == "__main__":
    main()
