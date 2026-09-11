import numpy as np

from transition_compass_model.model.common.data_matrix_class import DataMatrix

# lever name -> dm_capacity Category1
lever_categories = {
    "nuclear-capacity": "Nuclear",
    "onshore-wind-capacity": "WindOn",
    "pv-capacity": "PV-roof",
}

lever_var = "pow_capacity-lever-max"

# Plateau targets are rounded to the nearest whole GW *before* flooring the curve with
# them, so the whole floored portion of the curve lands on a value the pyomo model's
# number_of_units constraint ([Eq. 1.7] in ses_pyomo.py) can actually represent: F_Mult must
# be an exact integer multiple of ref_size["NUCLEAR"], which is 1 GW in ses_main.json - a
# coarse unit for two non-uniform real plants (1.010 + 1.233 GW). Rounding only the last
# (endyr) point instead would leave a false dip/jump right at 2050, since years before it
# never touch the pyomo model and would otherwise keep the exact, un-rounded real-world MW
# figure. !FIXME: keep this in sync with ref_size["NUCLEAR"] in ses_main.json if that ever
# changes.
NUCLEAR_REF_SIZE_MW = 1000


def round_to_ref_size(value_mw, ref_size_mw):
    if ref_size_mw <= 0:
        return value_mw
    return round(value_mw / ref_size_mw) * ref_size_mw


def _lever_dm(country, year, category, value_mw):
    dm_lever = DataMatrix(
        col_labels={
            "Country": [country],
            "Years": [year],
            "Variables": [lever_var],
            "Categories1": [category],
        },
        units={lever_var: "MW"},
    )
    dm_lever.array[:] = value_mw
    return dm_lever


def _lever_dm_series(country, years, category, values_mw):
    dm_lever = DataMatrix(
        col_labels={
            "Country": [country],
            "Years": list(years),
            "Variables": [lever_var],
            "Categories1": [category],
        },
        units={lever_var: "MW"},
    )
    dm_lever.array = np.array(values_mw, dtype=float).reshape(1, len(years), 1, 1)
    return dm_lever


def run(dm_capacity, reactor_list, years_ots, years_fts):
    # ots/fts structure matching the rest of the codebase (e.g. transport_preprocessing_CH.py):
    # DM["ots"][lever] = real historical value, DM["fts"][lever][level] = one DataMatrix per
    # lever level (1-4). Unlike other sectors, the pyomo energy model only ever produces a
    # single endyr snapshot, so each fts-level DataMatrix holds Years=[endyr] only, not a full
    # 2025-2050 series.
    endyr = years_fts[-1]

    DM_ots = dict()
    DM_fts = dict()
    for lever_name, category in lever_categories.items():
        dm_ots_lever = dm_capacity.filter(
            {
                "Country": ["Switzerland"],
                "Years": years_ots,
                "Variables": ["pow_existing-capacity"],
                "Categories1": [category],
            }
        )
        dm_ots_lever.rename_col("pow_existing-capacity", lever_var, dim="Variables")
        DM_ots[lever_name] = dm_ots_lever

    # Nuclear: never a straight-line fit between 2025 and 2050, always a step function, for
    # every lever level. Beznau1/2 retire on their real, known end-of-life dates (already in
    # Nexus-e's own Pmax forecast). Gosgen and Liebstadt have no announced closure date (Swiss
    # law does not impose a fixed operating-license limit on existing reactors), so their fate
    # is a genuine scenario choice, encoded as 4 distinct real-world states rather than a
    # smooth ramp:
    # 1 = Nexus-e's own forecast as-is (full phase-out, reaching 0 by 2045 - BAU)
    # 2 = Gosgen also retires, Liebstadt keeps running (floored at Liebstadt's Pmax)
    # 3 = both Gosgen and Liebstadt keep running, no new build (floored at their combined Pmax)
    # 4 = same as 3, plus ~2 GW of new nuclear active right at endyr (a step at the very last
    #     year, not a ramp - loosely based on an ETH study)
    # !FIXME: confirm the assumed Gosgen/Liebstadt retirement narrative for levels 1/2 (there is
    # no real announced date to anchor them to), and whether the level-4 new-build figure is
    # additive on top of the existing fleet (assumed here) or a replacement total.
    nuclear_liebstadt_mw = round_to_ref_size(
        reactor_list["Liebstadt"]["Pmax"], NUCLEAR_REF_SIZE_MW
    )
    nuclear_both_raw_mw = (
        reactor_list["Gosgen"]["Pmax"] + reactor_list["Liebstadt"]["Pmax"]
    )
    nuclear_both_mw = round_to_ref_size(nuclear_both_raw_mw, NUCLEAR_REF_SIZE_MW)
    nuclear_new_build_mw = 2000
    nuclear_both_new_mw = round_to_ref_size(
        nuclear_both_raw_mw + nuclear_new_build_mw, NUCLEAR_REF_SIZE_MW
    )

    dm_nuclear_raw = dm_capacity.filter(
        {
            "Country": ["Switzerland"],
            "Years": years_fts,
            "Variables": ["pow_capacity-Pmax"],
            "Categories1": ["Nuclear"],
        }
    )
    nuclear_raw_curve = dm_nuclear_raw.array.reshape(len(years_fts))

    nuclear_level_curves = {
        1: nuclear_raw_curve,
        2: np.maximum(nuclear_raw_curve, nuclear_liebstadt_mw),
        3: np.maximum(nuclear_raw_curve, nuclear_both_mw),
    }
    # Level 4: same floor as level 3 through 2045, then a step up to the (rounded) new-build
    # total right at endyr - a step, not a ramp.
    nuclear_level_curves[4] = nuclear_level_curves[3].copy()
    nuclear_level_curves[4][-1] = nuclear_both_new_mw

    DM_fts["nuclear-capacity"] = {
        level: _lever_dm_series("Switzerland", years_fts, "Nuclear", curve)
        for level, curve in nuclear_level_curves.items()
    }

    # Wind/PV: level 1 = today's installed capacity (no further deployment), level 4 =
    # technical potential (pow_capacity-Pmax), levels 2/3 linearly interpolated. Only endyr
    # matters here (no decommissioning story to preserve), so a single-point fts DataMatrix
    # is enough.
    for lever_name in ["onshore-wind-capacity", "pv-capacity"]:
        category = lever_categories[lever_name]
        level1_value = dm_capacity[
            "Switzerland", endyr, "pow_existing-capacity", category
        ]
        level4_value = dm_capacity["Switzerland", endyr, "pow_capacity-Pmax", category]
        DM_fts[lever_name] = {
            level: _lever_dm(
                "Switzerland",
                endyr,
                category,
                level1_value + (level - 1) / 3 * (level4_value - level1_value),
            )
            for level in [1, 2, 3, 4]
        }

    return DM_ots, DM_fts
