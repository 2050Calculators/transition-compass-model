import numpy as np
import pandas as pd

from transition_compass_model.model.common.data_matrix_class import DataMatrix


def extract_oil_gas_capacity_data(local_filename, years_ots):
    df = pd.read_excel(local_filename)
    dict_name = {"Inst. Leistung (MW)": "Pmax", "Brennstoff": "Type"}
    df.rename(dict_name, axis=1, inplace=True)
    df = df[["Canton", "Pmax", "Type", "StartYr", "EndYr"]]
    # df = df.pivot(columns='Type', index=['Canton', 'StartYr', 'EndYr'], values='Pmax')
    # df.reset_index(inplace=True)
    df["EndYr"] = df["EndYr"].fillna(2050)
    col_labels_dict = {
        "Country": list(df["Canton"].unique()),
        "Years": years_ots,
        "Variables": ["pow_capacity-Pmax"],
        "Categories1": ["Oil", "Gas"],
    }
    dm = DataMatrix(col_labels_dict, units={"pow_capacity-Pmax": "MW"})
    dm.array = np.zeros(tuple([len(list) for list in col_labels_dict.values()]))
    idx = dm.idx
    for yr in years_ots:
        for row in range(len(df)):
            StartYr = df.iloc[row]["StartYr"]
            EndYr = df.iloc[row]["EndYr"]
            if StartYr <= yr and yr <= EndYr:
                cntr = df.iloc[row]["Canton"]
                cat = df.iloc[row]["Type"]
                dm.array[idx[cntr], idx[yr], idx["pow_capacity-Pmax"], idx[cat]] += (
                    df.iloc[row]["Pmax"]
                )
    dm_CH = dm.groupby({"Switzerland": ".*"}, dim="Country", regex=True, inplace=False)
    dm.append(dm_CH, dim="Country")
    return dm


def run(years_ots):
    local_filename = "data/oil_gas_power_plants.xlsx"
    dm_capacity_oilgas_ots = extract_oil_gas_capacity_data(local_filename, years_ots)

    return dm_capacity_oilgas_ots


def split_oilgas_by_capacity_factor(dm_production, dm_capacity, years_ots):
    # SECTION - Determine split Oil-Gas into GasCC, GasSC, Oil and Cogen using capacity factors
    dm_prod_oilgas = dm_production.filter(
        {
            "Country": ["Switzerland"],
            "Categories1": ["Oil-Gas"],
            "Variables": ["pow_production"],
        }
    )
    dm_cap_oilgas = dm_capacity.filter_w_regex(
        {
            "Variables": "pow_existing-capacity",
            "Categories1": ".*Gas.*|Oil",
            "Country": "Switzerland",
        }
    )
    dm_cap_oilgas.filter({"Years": years_ots}, inplace=True)
    dm_cap_oilgas.normalise("Categories1", inplace=True, keep_original=False)
    arr_prod_split = dm_cap_oilgas.array * dm_prod_oilgas.array
    dm_production.add(
        arr_prod_split,
        dim="Categories1",
        col_label=dm_cap_oilgas.col_labels["Categories1"],
    )
    dm_production.drop("Categories1", "Oil-Gas")
    cat_matching = list(
        set(dm_capacity.col_labels["Categories1"]).intersection(
            set(dm_production.col_labels["Categories1"])
        )
    )
    dm_cap_match = dm_capacity.filter(
        {
            "Categories1": cat_matching,
            "Country": ["Switzerland"],
            "Variables": ["pow_existing-capacity"],
            "Years": years_ots,
        }
    )
    dm_prod_match = dm_production.filter({"Categories1": cat_matching})
    dm_cap_match.append(dm_prod_match, dim="Variables")
    dm_cap_match.change_unit(
        "pow_existing-capacity", old_unit="MW", new_unit="GW", factor=1e-3
    )
    dm_cap_match.operation(
        "pow_production",
        "/",
        "pow_existing-capacity",
        out_col="pow_cap-fact",
        unit="TWh/GW",
    )
    dm_cap_match.change_unit(
        "pow_cap-fact", old_unit="TWh/GW", new_unit="%", factor=8.760, operator="/"
    )

    mask = dm_cap_match[:, :, "pow_cap-fact", :] > 1
    dm_cap_match[:, :, "pow_cap-fact", :][mask] = np.nan
    dm_cap_match.fill_nans("Years")

    dm_cap_match.change_unit(
        "pow_cap-fact", old_unit="%", new_unit="TWh/GW", factor=8.760, operator="*"
    )
    dm_cap_match.operation(
        "pow_production", "/", "pow_cap-fact", out_col="pow_capacity", unit="GW"
    )
    dm_cap_match.change_unit("pow_capacity", old_unit="GW", new_unit="MW", factor=1000)

    dm_cap_match.filter({"Variables": ["pow_capacity"]}, inplace=True)

    dm_cap_ots = dm_capacity.filter({"Years": years_ots})
    dm_capacity.drop(dim="Years", col_label=years_ots)
    for cat in dm_cap_match.col_labels["Categories1"]:
        dm_cap_ots["Switzerland", :, "pow_existing-capacity", cat] = dm_cap_match[
            "Switzerland", :, "pow_capacity", cat
        ]
    dm_capacity.append(dm_cap_ots, dim="Years")
    dm_capacity.sort("Years")

    return dm_production, dm_capacity
