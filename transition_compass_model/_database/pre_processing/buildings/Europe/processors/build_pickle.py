"""
EU27 buildings pickle builder.

Assembles all preprocessed DataMatrices into the DM_buildings structure
(ots / fts / fxa / constant), filters to EU27, and saves to buildings.pickle.

Two-step build
--------------
1. run_ots(...) : Build OTS + FXA + constant → save to _OTS_CACHE_PATH (cached).
                  Returns DM_buildings_ots dict with keys "ots", "fxa", "constant".
2. run(...)     : Add FTS on top → my_pickle_dump to buildings.pickle.
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd

from transition_compass_model._database.pre_processing.api_routine_Eurostat import (
    get_data_api_eurostat,
)
from transition_compass_model.model.common.auxiliary_functions import (
    eurostat_iso2_dict,
    linear_fitting,
    my_pickle_dump,
)
from transition_compass_model.model.common.constant_data_matrix_class import (
    ConstantDataMatrix,
)
from transition_compass_model.model.common.data_matrix_class import DataMatrix

warnings.simplefilter("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(_THIS_DIR, "../data/")
_LIFESTYLES_PATH = os.path.join(
    _THIS_DIR, "../../../lifestyles/Europe/data/lifestyles_allcountries.pickle"
)
_BUILDINGS_PICKLE = os.path.join(
    _THIS_DIR, "../../../../data/datamatrix/buildings.pickle"
)
_OTS_CACHE_PATH = os.path.join(_THIS_DIR, "../data/datamatrix/build_ots_EU.pickle")
_OUTPUT_PATH = _BUILDINGS_PICKLE

_DICT_ISO2 = eurostat_iso2_dict()
_DICT_ISO2.pop("CH")

NON_RES_CATS = ["education", "health", "hotels", "offices", "other", "trade"]
ENERGY_CLASSES = ["B", "C", "D", "E", "F"]


#################################################################
########################### FUNCTIONS ###########################
#################################################################


def df_excel_to_dm(
    df, names_dict, var_name, unit, num_cat, keep_first=False, country="Switzerland"
):
    if np.nan in df.columns:
        df.drop(columns=np.nan, inplace=True)
    df.rename(columns={df.columns[0]: "Variables"}, inplace=True)
    df.set_index("Variables", inplace=True)
    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df = df[df.apply(lambda row: row.map(pd.api.types.is_number), axis=1).any(axis=1)]
    df_clean = df.loc[
        :, df.apply(lambda col: col.map(pd.api.types.is_number)).any(axis=0)
    ].copy()
    df_filter = df_clean.loc[names_dict.keys()].copy()
    df_filter = df_filter.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    df_filter.reset_index(inplace=True)
    df_filter["Variables"] = df_filter["Variables"].replace(names_dict)
    if keep_first:
        df_filter = df_filter.drop_duplicates(subset=["Variables"], keep="first")
    df_filter = df_filter.groupby(["Variables"]).sum()
    df_filter.reset_index(inplace=True)

    df_filter["Country"] = country
    df_T = pd.melt(
        df_filter,
        id_vars=["Variables", "Country"],
        var_name="Years",
        value_name="values",
    )
    df_pivot = df_T.pivot_table(
        index=["Country", "Years"],
        columns=["Variables"],
        values="values",
        aggfunc="sum",
    )
    df_pivot = df_pivot.add_suffix("[" + unit + "]")
    df_pivot = df_pivot.add_prefix(var_name + "_")
    df_pivot.reset_index(inplace=True)

    df_pivot["Years"] = pd.to_numeric(df_pivot["Years"], errors="coerce")
    df_pivot = df_pivot.dropna(subset=["Years"])

    dm = DataMatrix.create_from_df(df_pivot, num_cat=num_cat)
    return dm


def recompute_floor_area_per_capita(dm_all, dm_pop):
    dm_floor_stock = dm_all.filter(
        {
            "Variables": ["bld_floor-area_stock"],
            "Categories1": ["single-family-households", "multi-family-households"],
        },
        inplace=False,
    )
    dm_floor_stock.group_all(dim="Categories2")
    dm_floor_stock.group_all(dim="Categories1")
    dm_floor_stock.append(dm_pop, dim="Variables")
    dm_floor_stock.operation(
        "bld_floor-area_stock",
        "/",
        "lfs_population_total",
        out_col="lfs_floor-intensity_space-cap",
        unit="m2/cap",
    )
    dm_floor_stock.filter(
        {"Variables": ["lfs_floor-intensity_space-cap"]}, inplace=True
    )
    return dm_floor_stock


def get_household_size_eustat(years_ots):
    filter_dict = {"geo\\TIME_PERIOD": list(_DICT_ISO2.keys()), "freq": "A"}
    mapping_dim = {"Country": "geo\\TIME_PERIOD", "Variables": "freq"}
    dm_hh_size = get_data_api_eurostat(
        "ilc_lvph01", filter_dict, mapping_dim, unit="people", years=years_ots
    )
    dm_hh_size.rename_col("A", "lfs_household-size", dim="Variables")
    linear_fitting(dm_hh_size, years_ots)
    return dm_hh_size


def compute_building_mix(dm_all, dm_srv_floor=None, envelope_cat_new_nonres=None):
    res_types = ["multi-family-households", "single-family-households"]
    dm_res = dm_all.filter(
        {
            "Variables": ["bld_floor-area_stock", "bld_floor-area_new"],
            "Categories1": res_types,
        },
        inplace=False,
    )

    if dm_srv_floor is not None:
        energy_classes = dm_srv_floor.col_labels["Categories2"]

        dm_nonres_stock = dm_srv_floor.copy()
        dm_nonres_stock.rename_col(
            "bld_floor-area_services", "bld_floor-area_stock", "Variables"
        )

        dm_nonres_new = dm_nonres_stock.copy()
        dm_nonres_new.rename_col(
            "bld_floor-area_stock", "bld_floor-area_new", "Variables"
        )
        dm_nonres_new.array[:] = 0.0
        if envelope_cat_new_nonres:
            idx = dm_nonres_new.idx
            years_list = dm_nonres_new.col_labels["Years"]
            for cat, (yr_start, yr_end) in envelope_cat_new_nonres.items():
                if cat in energy_classes:
                    i_cat = energy_classes.index(cat)
                    for i_yr, yr in enumerate(years_list):
                        if yr_start <= yr <= yr_end and yr in idx:
                            if i_yr > 0:
                                prev_yr = years_list[i_yr - 1]
                                delta = np.maximum(
                                    0,
                                    dm_nonres_stock.array[:, idx[yr], 0, :, :].sum(
                                        axis=-1
                                    )
                                    - dm_nonres_stock.array[
                                        :, idx[prev_yr], 0, :, :
                                    ].sum(axis=-1),
                                )
                            else:
                                next_yr = years_list[i_yr + 1]
                                delta = np.maximum(
                                    0,
                                    dm_nonres_stock.array[:, idx[next_yr], 0, :, :].sum(
                                        axis=-1
                                    )
                                    - dm_nonres_stock.array[:, idx[yr], 0, :, :].sum(
                                        axis=-1
                                    ),
                                )
                            dm_nonres_new.array[:, idx[yr], 0, :, i_cat] = delta

        dm_nonres_stock.append(dm_nonres_new, dim="Variables")
        dm_res.append(dm_nonres_stock, dim="Categories1")

    dm_building_mix = dm_res.flatten()
    dm_building_mix.normalise("Categories1", keep_original=True)
    dm_building_mix.deepen()
    dm_building_mix.rename_col(
        ["bld_floor-area_stock_share", "bld_floor-area_new_share"],
        ["bld_building-mix_stock", "bld_building-mix_new"],
        dim="Variables",
    )
    dm_building_mix.filter(
        {"Variables": ["bld_building-mix_stock", "bld_building-mix_new"]}, inplace=True
    )
    return dm_building_mix


def compute_building_age(dm_stock_cat, years_fts, first_bld):
    dm_age = dm_stock_cat.filter({"Variables": ["bld_floor-area_stock"]})
    dm_age.rename_col("bld_floor-area_stock", "bld_age", dim="Variables")
    dm_age.change_unit("bld_age", 1, "m2", "years")
    dm_age.add(np.nan, dim="Years", col_label=years_fts, dummy=True)
    years_all = np.array(dm_age.col_labels["Years"])
    nb_cntr = len(dm_age.col_labels["Country"])
    idx = dm_age.idx
    for bld_type in dm_stock_cat.col_labels["Categories1"]:
        for cat, start_yr in first_bld.items():
            arr_age = years_all - start_yr
            arr_age = np.maximum(arr_age, 0)
            for idx_c in range(nb_cntr):
                dm_age.array[idx_c, :, idx["bld_age"], idx[bld_type], idx[cat]] = (
                    arr_age
                )
    return dm_age


def calculate_heating_eff_fts(dm_heating_eff, years_fts, maximum_eff):
    dm_heat_pump = dm_heating_eff.filter({"Categories1": ["heat-pump"]})
    dm_heating_eff.drop(dim="Categories1", col_label="heat-pump")
    linear_fitting(dm_heating_eff, years_fts, based_on=list(range(2015, 2023)))
    dm_heating_eff.array = np.minimum(dm_heating_eff.array, maximum_eff)
    linear_fitting(dm_heat_pump, years_fts, based_on=list(range(2015, 2023)))
    dm_heating_eff.append(dm_heat_pump, dim="Categories1")
    return dm_heating_eff.filter({"Years": years_fts})


def get_energy_demand_useful_dhw(years_ots, years_fts):
    def _from_jrc(country_code, country_name):
        file = os.path.join(
            _THIS_DIR,
            f"../../../industry/eu/data/JRC-IDEES-2021/{country_code}/JRC-IDEES-2021_Residential_{country_code}.xlsx",
        )
        df = pd.read_excel(file, sheet_name="RES_hh_tes")
        df_temp = df[15:16].copy()
        names_map = {"Water heating": "dhw"}
        return df_excel_to_dm(
            df_temp,
            names_map,
            var_name="jrc",
            unit="ktoe useful",
            num_cat=0,
            country=country_name,
        )

    dict_temp = _DICT_ISO2.copy()
    dict_temp.pop("EU27_2020")
    dict_temp.pop("UK")

    dm_dhw = _from_jrc("EU27", "EU27")
    for key, name in dict_temp.items():
        dm_dhw.append(_from_jrc(key, name), "Country")
    dm_dhw.change_unit("jrc_dhw", 11630, "ktoe useful", "MWh")

    with open(_LIFESTYLES_PATH, "rb") as handle:
        DM_lfs = pickle.load(handle)
    dm_pop = DM_lfs["ots"]["pop"]["lfs_population_"].copy()
    dm_pop.append(DM_lfs["fts"]["pop"]["lfs_population_"][1], "Years")
    dm_pop = dm_pop.filter(
        {"Country": dm_dhw.col_labels["Country"], "Years": dm_dhw.col_labels["Years"]}
    )
    dm_pop.sort("Years")
    del DM_lfs

    dm_dhw.append(dm_pop, "Variables")
    dm_dhw.operation(
        "jrc_dhw",
        "/",
        "lfs_population_total",
        "Variables",
        out_col="bld_hw_demand",
        unit="MWh/cap",
    )
    dm_dhw_cap = dm_dhw.filter({"Variables": ["bld_hw_demand"]})
    linear_fitting(dm_dhw_cap, years_ots)
    linear_fitting(dm_dhw_cap, years_fts)
    return dm_dhw_cap


def get_domapp(DM_bld, years_ots, years_fts):
    def _from_jrc(country_code, country_name):
        base = os.path.join(
            _THIS_DIR,
            f"../../../industry/eu/data/JRC-IDEES-2021/{country_code}/JRC-IDEES-2021_Residential_{country_code}.xlsx",
        )
        df = pd.read_excel(base, sheet_name="RES_se-appl")
        names_map = {
            "Refrigerators and freezers": "refrigerator",
            "Washing machine": "washing-machine",
            "Clothes dryer": "tumble-dryer",
            "Dishwasher": "dishwasher",
            "TV and multimedia": "TV",
            "ICT equipment": "PC",
        }
        df_temp = pd.concat([df[107:111].copy(), df[112:114].copy()])
        dm_out = df_excel_to_dm(
            df_temp,
            names_map,
            var_name="bld_appliances_stock",
            unit="unit/household",
            num_cat=1,
            country=country_name,
        )
        df_temp = pd.concat([df[29:33].copy(), df[34:36].copy()])
        dm_temp = df_excel_to_dm(
            df_temp,
            names_map,
            var_name="bld_appliances_stock_absolute",
            unit="unit",
            num_cat=1,
            country=country_name,
        )
        dm_out.append(dm_temp, "Variables")
        df_temp = pd.concat([df[55:59].copy(), df[60:62].copy()])
        dm_temp = df_excel_to_dm(
            df_temp,
            names_map,
            var_name="bld_appliances_replaced",
            unit="unit",
            num_cat=1,
            country=country_name,
        )
        dm_temp.add(np.nan, "Years", [2000], dummy=True)
        dm_temp.sort("Years")
        dm_out.append(dm_temp, "Variables")
        df_temp = pd.concat([df[3:7].copy(), df[8:10].copy()])
        dm_temp = df_excel_to_dm(
            df_temp,
            names_map,
            var_name="bld_appliances_electricity-demand_absolute",
            unit="ktoe",
            num_cat=1,
            country=country_name,
        )
        dm_out.append(dm_temp, "Variables")
        return dm_out

    dict_temp = _DICT_ISO2.copy()
    dict_temp.pop("EU27_2020")
    dict_temp.pop("UK")

    dm_domapp = _from_jrc("EU27", "EU27")
    for key, name in dict_temp.items():
        dm_domapp.append(_from_jrc(key, name), "Country")

    dm_domapp.operation(
        "bld_appliances_replaced",
        "/",
        "bld_appliances_stock_absolute",
        "Variables",
        "bld_appliances_retirement-rate",
        "%",
    )
    dm_domapp.change_unit(
        "bld_appliances_electricity-demand_absolute", 1.163 * 1e7, "ktoe", "kWh"
    )
    dm_domapp.operation(
        "bld_appliances_electricity-demand_absolute",
        "/",
        "bld_appliances_stock_absolute",
        "Variables",
        "bld_appliances_electricity-demand",
        "kWh/unit",
    )
    dm_domapp = dm_domapp.filter(
        {
            "Variables": [
                "bld_appliances_electricity-demand",
                "bld_appliances_retirement-rate",
                "bld_appliances_stock",
            ]
        }
    )
    dm_temp = dm_domapp.copy()
    dm_domapp = dm_temp.filter({"Variables": ["bld_appliances_electricity-demand"]})
    dm_domapp.append(
        dm_temp.filter({"Variables": ["bld_appliances_retirement-rate"]}), "Variables"
    )
    dm_domapp.append(
        dm_temp.filter({"Variables": ["bld_appliances_stock"]}), "Variables"
    )

    dm_domapp.add(np.nan, "Categories1", ["freezer"], dummy=True)
    dm_domapp[..., "freezer"] = dm_domapp[..., "refrigerator"]
    dm_domapp.sort("Categories1")

    linear_fitting(dm_domapp, list(range(1990, 2000)), based_on=[2000, 2001])
    linear_fitting(dm_domapp, list(range(2022, 2023 + 1)), based_on=[2020, 2021])
    linear_fitting(dm_domapp, years_fts, based_on=[2023])

    dm_ch = DM_bld["fxa"]["appliances"].filter({"Years": dm_domapp.col_labels["Years"]})
    dm_domapp.add(
        np.nan,
        "Categories1",
        ["monitor", "laptop", "set-top-box", "oven-and-stove"],
        dummy=True,
    )
    for country in dm_domapp.col_labels["Country"]:
        for product in [
            "PC",
            "TV",
            "monitor",
            "laptop",
            "set-top-box",
            "oven-and-stove",
        ]:
            dm_domapp[country, :, :, product] = dm_ch["Switzerland", :, :, product]
    dm_domapp.sort("Categories1")
    return dm_domapp


def get_services(years_ots, years_fts):
    def _demand_from_jrc(country_code, country_name):
        base = os.path.join(
            _THIS_DIR,
            f"../../../industry/eu/data/JRC-IDEES-2021/{country_code}/JRC-IDEES-2021_Tertiary_{country_code}.xlsx",
        )
        df = pd.read_excel(base, sheet_name="SER_hh_tes")
        df_temp = df[3:14].copy()
        names_map = {
            "Solids": "coal",
            "Liquified petroleum gas (LPG)": "gas",
            "Diesel oil": "heating-oil",
            "Gas heat pumps": "heat-pump",
            "Conventional gas heaters": "gas",
            "Biomass": "wood",
            "Geothermal": "heat-pump",
            "Distributed heat": "district-heating",
            "Advanced electric heating": "heat-pump",
            "Conventional electric heating": "electricity",
            "Electricity in circulation and other use": "electricity",
        }
        dm_jrc_sh = df_excel_to_dm(
            df_temp,
            names_map,
            var_name="bld_services_tech-mix_space-heating",
            unit="ktoe",
            num_cat=1,
            country=country_name,
        )
        dm_jrc_sh.add(0, "Categories1", "biogas", dummy=True)
        dm_jrc_sh.add(0, "Categories1", "biomass", dummy=True)
        dm_jrc_sh.add(0, "Categories1", "solar", dummy=True)
        dm_jrc_sh.sort("Categories1")
        dm_jrc_sh.deepen("_", based_on="Variables")
        dm_jrc_sh.switch_categories_order("Categories2", "Categories1")

        df_temp = df[18:26]
        names_map = {
            "Solids": "coal",
            "Liquified petroleum gas (LPG)": "gas",
            "Diesel oil": "heating-oil",
            "Natural gas": "gas",
            "Biomass": "wood",
            "Distributed heat": "district-heating",
            "Electricity": "electricity",
            "Solar": "solar",
        }
        dm_jrc_wh = df_excel_to_dm(
            df_temp,
            names_map,
            var_name="bld_services_tech-mix_hot-water",
            unit="ktoe",
            num_cat=1,
            country=country_name,
        )
        dm_jrc_wh.add(0, "Categories1", "biogas", dummy=True)
        dm_jrc_wh.add(0, "Categories1", "biomass", dummy=True)
        dm_jrc_wh.add(0, "Categories1", "heat-pump", dummy=True)
        dm_jrc_wh.sort("Categories1")
        dm_jrc_wh.deepen("_", based_on="Variables")
        dm_jrc_wh.switch_categories_order("Categories2", "Categories1")

        dm_jrc = dm_jrc_sh.copy()
        dm_jrc.append(dm_jrc_wh, "Categories1")
        dm_jrc.sort("Categories1")

        df2 = pd.read_excel(base, sheet_name="SER_se-appl")
        df_temp = df2[3:5]
        names_map = {"Street lighting": "street", "Building lighting": "building"}
        dm_jrc_light = df_excel_to_dm(
            df_temp,
            names_map,
            var_name="bld_services_tech-mix_lighting_electricity",
            unit="ktoe",
            num_cat=1,
            country=country_name,
        )
        dm_jrc_light.group_all("Categories1")
        dm_jrc_light.deepen_twice()
        categories_missing = dm_jrc.col_labels["Categories2"].copy()
        categories_missing.remove("electricity")
        for cat in categories_missing:
            dm_jrc_light.add(0, "Categories2", cat, dummy=True)
        dm_jrc.append(dm_jrc_light, "Categories1")

        dm_elec = dm_jrc.filter({"Categories2": ["electricity"]})
        dm_elec.groupby(
            {"elec": ["hot-water", "space-heating"]}, "Categories1", inplace=True
        )
        for cat in categories_missing:
            dm_elec.add(0, "Categories2", cat, dummy=True)
        dm_jrc[..., "electricity"] = 0
        dm_jrc.drop("Categories1", "lighting")
        dm_jrc.append(dm_elec, "Categories1")
        dm_jrc.sort("Categories1")
        dm_jrc.sort("Categories2")
        return dm_jrc

    def _eff_from_jrc(country_code, country_name):
        base = os.path.join(
            _THIS_DIR,
            f"../../../industry/eu/data/JRC-IDEES-2021/{country_code}/JRC-IDEES-2021_Tertiary_{country_code}.xlsx",
        )
        df = pd.read_excel(base, sheet_name="SER_hh_eff")
        df_temp = df[3:13].copy()
        names_map = {
            "Solids": "coal",
            "Liquified petroleum gas (LPG)": "gas",
            "Diesel oil": "heating-oil",
            "Biomass": "wood",
            "Distributed heat": "district-heating",
            "Advanced electric heating": "heat-pump",
            "Conventional electric heating": "electricity",
        }
        dm_jrc_sh = df_excel_to_dm(
            df_temp,
            names_map,
            var_name="bld_efficiency_space-heating",
            unit="%",
            num_cat=1,
            country=country_name,
        )
        dm_jrc_sh.add(dm_jrc_sh[..., "gas"], "Categories1", "biogas", dummy=True)
        dm_jrc_sh.add(dm_jrc_sh[..., "wood"], "Categories1", "biomass", dummy=True)
        dm_jrc_sh.add(1, "Categories1", "solar", dummy=True)
        dm_jrc_sh.sort("Categories1")
        dm_jrc_sh.deepen("_", based_on="Variables")
        dm_jrc_sh.switch_categories_order("Categories2", "Categories1")

        df_temp = df[18:26]
        names_map = {
            "Solids": "coal",
            "Liquified petroleum gas (LPG)": "gas",
            "Diesel oil": "heating-oil",
            "Biomass": "wood",
            "Distributed heat": "district-heating",
            "Electricity": "electricity",
            "Solar": "solar",
        }
        dm_jrc_wh = df_excel_to_dm(
            df_temp,
            names_map,
            var_name="bld_efficiency_water-heating",
            unit="%",
            num_cat=1,
            country=country_name,
        )
        dm_jrc_wh.add(dm_jrc_wh[..., "gas"], "Categories1", "biogas", dummy=True)
        dm_jrc_wh.add(dm_jrc_wh[..., "wood"], "Categories1", "biomass", dummy=True)
        dm_jrc_wh.add(np.nan, "Categories1", "heat-pump", dummy=True)
        dm_jrc_wh.sort("Categories1")
        dm_jrc_wh.deepen("_", based_on="Variables")
        dm_jrc_wh.switch_categories_order("Categories2", "Categories1")

        dm_jrc = dm_jrc_sh.copy()
        dm_jrc.append(dm_jrc_wh, "Categories1")
        dm_jrc.sort("Categories1")
        dm_jrc.rename_col("bld_efficiency", "bld_services_efficiency", "Variables")
        dm_jrc.group_all("Categories1", aggregation="mean")
        return dm_jrc

    dict_temp = _DICT_ISO2.copy()
    dict_temp.pop("EU27_2020")
    dict_temp.pop("UK")

    dm_ued = _demand_from_jrc("EU27", "EU27")
    for key, name in dict_temp.items():
        dm_ued.append(_demand_from_jrc(key, name), "Country")
    dm_ued.change_unit("bld_services_tech-mix", 0.01163, "ktoe", "TWh")

    dm_demand = dm_ued.copy()
    dm_demand.group_all("Categories2")
    dm_demand.rename_col(
        "bld_services_tech-mix", "bld_services_useful-energy", "Variables"
    )
    linear_fitting(dm_demand, list(range(1990, 2000)), based_on=[2000])
    linear_fitting(dm_demand, list(range(2022, 2023 + 1)))
    linear_fitting(dm_demand, years_fts)
    for y in [2022, 2023]:
        dm_demand[:, y, :, "lighting"] = dm_demand[:, 2021, :, "lighting"]
    for y in years_fts:
        dm_demand[:, y, :, "lighting"] = dm_demand[:, 2021, :, "lighting"]

    dm_tech_mix = dm_ued.copy()
    dm_tech_mix.normalise("Categories2")
    dm_tech_mix.add(0, "Categories2", "other-tech", dummy=True)
    dm_tech_mix.sort("Categories2")
    linear_fitting(dm_tech_mix, list(range(1990, 2000)), based_on=[2000])
    linear_fitting(dm_tech_mix, list(range(2022, 2023 + 1)), based_on=[2021])
    linear_fitting(dm_tech_mix, years_fts, based_on=[2023])

    dm_eff = _eff_from_jrc("EU27", "EU27")
    for key, name in dict_temp.items():
        dm_eff.append(_eff_from_jrc(key, name), "Country")
    linear_fitting(dm_eff, years_ots)
    dm_eff_hp = dm_eff.filter({"Categories1": ["heat-pump"]})
    dm_eff.drop("Categories1", "heat-pump")
    linear_fitting(dm_eff, years_fts, max_t0=1, max_tb=1)
    linear_fitting(dm_eff_hp, years_fts, max_t0=4, max_tb=4)
    dm_eff.append(dm_eff_hp, "Categories1")
    dm_eff.add(0.6, "Categories1", "other-tech", dummy=True)
    dm_eff.sort("Categories1")

    return dm_demand, dm_tech_mix, dm_eff


def get_domapp_other(years_ots, years_fts):
    def _from_jrc(country_code, country_name):
        base = os.path.join(
            _THIS_DIR,
            f"../../../industry/eu/data/JRC-IDEES-2021/{country_code}/JRC-IDEES-2021_Residential_{country_code}.xlsx",
        )
        df = pd.read_excel(base, sheet_name="RES_se-appl")
        df_temp = df[11:13].copy()
        names_map = {
            "Lighting ": "lighting",
            "Other appliances (vacuum cleaners, irons etc.)": "other",
        }
        return df_excel_to_dm(
            df_temp,
            names_map,
            var_name="bld",
            unit="ktoe",
            num_cat=1,
            country=country_name,
        )

    dict_temp = _DICT_ISO2.copy()
    dict_temp.pop("EU27_2020")
    dict_temp.pop("UK")

    dm_domapp = _from_jrc("EU27", "EU27")
    for key, name in dict_temp.items():
        dm_domapp.append(_from_jrc(key, name), "Country")

    dm_domapp_lighting = dm_domapp.filter({"Categories1": ["lighting"]}).flatten()
    dm_domapp_lighting.rename_col(
        "bld_lighting", "bld_residential-lighting", "Variables"
    )
    dm_domapp_lighting.change_unit("bld_residential-lighting", 0.01163, "ktoe", "TWh")
    linear_fitting(
        dm_domapp_lighting, list(range(1990, 2000)), based_on=list(range(2000, 2009))
    )
    linear_fitting(
        dm_domapp_lighting, list(range(2022, 2024)), based_on=list(range(2020, 2022))
    )
    linear_fitting(dm_domapp_lighting, years_fts)

    dm_domapp_other = dm_domapp.filter({"Categories1": ["other"]}).flatten()
    dm_domapp_other.rename_col(
        "bld_other", "bld_energy-demand_other_electricity", "Variables"
    )
    dm_domapp_other.change_unit(
        "bld_energy-demand_other_electricity", 11630000, "ktoe", "kWh"
    )

    with open(_LIFESTYLES_PATH, "rb") as handle:
        DM_lfs = pickle.load(handle)
    dm_pop = DM_lfs["ots"]["pop"]["lfs_population_"].copy()
    dm_pop.append(DM_lfs["fts"]["pop"]["lfs_population_"][1], "Years")
    dm_pop = dm_pop.filter(
        {
            "Country": dm_domapp_other.col_labels["Country"],
            "Years": dm_domapp_other.col_labels["Years"],
        }
    )
    dm_pop.sort("Years")
    del DM_lfs

    dm_domapp_other.append(dm_pop, "Variables")
    dm_domapp_other.operation(
        "bld_energy-demand_other_electricity",
        "/",
        "lfs_population_total",
        "Variables",
        "per_capita",
        "kWh/cap",
    )
    dm_domapp_other = dm_domapp_other.filter({"Variables": ["per_capita"]})
    dm_domapp_other.rename_col(
        "per_capita", "bld_energy-demand_other_electricity", "Variables"
    )
    linear_fitting(dm_domapp_other, years_ots + years_fts)
    dm_domapp_other.deepen()

    return dm_domapp_lighting, dm_domapp_other


def filter_nested_structure(data, countries=["EU27"]):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, DataMatrix):
                data[key] = value.filter({"Country": countries})
            else:
                filter_nested_structure(value, countries)


#################################################################
########################## MAIN RUN ############################
#################################################################


def run_ots(
    dm_all_cat,
    dm_pop,
    dm_services_floor,
    dm_renovation,
    dm_renov_distr,
    dm_stock_tot,
    dm_waste_tot,
    dm_heating_cat,
    dm_heating_eff_cat,
    dm_heating_eff_all,
    dm_u_value,
    dm_s2f,
    years_ots,
    years_fts,
):
    if os.path.exists(_OTS_CACHE_PATH):
        print("  (cached)")
        with open(_OTS_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    # Load existing CH pickle (used for calibration/appliances CH values)
    with open(_BUILDINGS_PICKLE, "rb") as handle:
        DM_bld = pickle.load(handle)

    DM_buildings = {"ots": dict(), "fxa": dict(), "constant": dict()}

    ###############################################################
    # OTS: services floor area
    ###############################################################
    DM_buildings["ots"]["services-floor-area"] = dm_services_floor.copy()

    ###############################################################
    # FXA: heating energy calibration (proxy from Switzerland)
    ###############################################################
    dm_temp = DM_bld["fxa"]["heating-energy-calibration"].filter(
        {"Country": ["Switzerland"]}
    )
    dm_temp.rename_col("Switzerland", "EU27", "Country")
    country_missing = dm_heating_eff_cat.col_labels["Country"].copy()
    country_missing.remove("EU27")
    for c in country_missing:
        dm_temp.add(dm_temp["EU27", ...], dim="Country", col_label=c)
    dm_temp.sort("Country")
    DM_buildings["fxa"]["heating-energy-calibration"] = dm_temp.copy()

    ###############################################################
    # OTS: floor intensity (space/cap + household size + nonres/cap)
    ###############################################################
    dm_space_cap = recompute_floor_area_per_capita(dm_all_cat, dm_pop)
    dm_lfs_household_size = get_household_size_eustat(years_ots)
    dm_lfs_household_size.drop("Country", "United Kingdom")
    dm_space_cap.append(dm_lfs_household_size, dim="Variables")

    dm_nonres_total = dm_services_floor.group_all("Categories2", inplace=False)
    dm_nonres_total.group_all("Categories1")
    dm_nonres_total.rename_col(
        "bld_floor-area_services", "bld_floor-area_nonres_total", dim="Variables"
    )
    dm_nonres_total.append(dm_pop, dim="Variables")
    dm_nonres_total.operation(
        "bld_floor-area_nonres_total",
        "/",
        "lfs_population_total",
        out_col="bld_floor-intensity_nonres-cap",
        unit="m2/cap",
    )
    dm_nonres_total.filter(
        {"Variables": ["bld_floor-intensity_nonres-cap"]}, inplace=True
    )
    dm_space_cap.append(dm_nonres_total, dim="Variables")

    DM_buildings["ots"]["floor-intensity"] = dm_space_cap.copy()

    ###############################################################
    # OTS: heatcool behaviour
    ###############################################################
    col_label = {
        "Country": dm_stock_tot.col_labels["Country"],
        "Years": years_ots,
        "Variables": ["bld_Tint-heating", "bld_Tint-cooling"],
        "Categories1": dm_stock_tot.col_labels["Categories1"],
        "Categories2": ENERGY_CLASSES,
    }
    dm_Tint_heat = DataMatrix(
        col_labels=col_label, units={"bld_Tint-heating": "C", "bld_Tint-cooling": "C"}
    )
    arr_shape = (
        len(col_label["Country"]),
        len(years_ots),
        len(col_label["Variables"]),
        len(col_label["Categories1"]),
        len(ENERGY_CLASSES),
    )
    # Tint=18°C for all building types and classes: consistent with the Eurostat HDD convention
    # (https://ec.europa.eu/eurostat/cache/metadata/en/nrg_chdd_esms.htm), which defines
    # HDD = Σ(18 − T_mean) for days where T_mean < 15°C. The model HDD adjustment formula adds
    # (Tint − 18) × N_days-below-15 on top of Eurostat HDD, so Tint=18 means no extra correction —
    # the Eurostat HDD is used as-is, consistent with EN ISO 52016-1:2017 and EN 13790.
    dm_Tint_heat.array = 18 * np.ones(arr_shape)
    idx = dm_Tint_heat.idx
    DM_buildings["ots"]["heatcool-behaviour"] = dm_Tint_heat.copy()

    ###############################################################
    # OTS + FXA: building mix
    ###############################################################
    nonres_envelope_cat_new_eu = {"C": (1990, 2010), "B": (2011, 2023)}
    dm_building_mix = compute_building_mix(
        dm_all_cat, dm_services_floor, nonres_envelope_cat_new_eu
    )
    dm_building_mix.add(np.nan, dummy=True, dim="Years", col_label=years_fts)
    DM_buildings["fxa"]["bld_type"] = dm_building_mix.filter(
        {"Variables": ["bld_building-mix_stock"]}
    )

    DM_buildings["ots"]["building-renovation-rate"] = dict()
    dm_tmp = dm_building_mix.filter({"Variables": ["bld_building-mix_new"]})
    DM_buildings["ots"]["building-renovation-rate"]["bld_building-mix"] = dm_tmp.filter(
        {"Years": years_ots}
    )

    ###############################################################
    # OTS: renovation rate + redistribution
    ###############################################################
    dm_rr = dm_renovation.filter({"Variables": ["bld_renovation-rate"]})
    DM_buildings["ots"]["building-renovation-rate"]["bld_renovation-rate"] = (
        dm_rr.copy()
    )

    DM_buildings["ots"]["building-renovation-rate"]["bld_renovation-redistribution"] = (
        dm_renov_distr.copy()
    )

    ###############################################################
    # OTS: demolition rate
    ###############################################################
    dm_tot = dm_stock_tot.copy()
    dm_tot.append(dm_waste_tot, dim="Variables")
    dm_tot.lag_variable("bld_floor-area_stock", shift=1, subfix="_tm1")
    dm_tot.operation(
        "bld_floor-area_waste",
        "/",
        "bld_floor-area_stock_tm1",
        out_col="bld_demolition-rate",
        unit="%",
    )
    dm_demolition_rate = dm_tot.filter({"Variables": ["bld_demolition-rate"]})
    dm_demolition_rate[:, 1990, :, :] = np.nan
    dm_demolition_rate.fill_nans("Years")
    dm_demolition_rate = dm_demolition_rate.filter(
        {
            "Categories1": [
                "education",
                "health",
                "hotels",
                "multi-family-households",
                "offices",
                "other",
                "single-family-households",
                "trade",
            ]
        }
    )
    DM_buildings["ots"]["building-renovation-rate"]["bld_demolition-rate"] = (
        dm_demolition_rate.copy()
    )

    ###############################################################
    # FXA: building age
    ###############################################################
    first_bld = {"F": 1900, "E": 1970, "D": 1980, "C": 1990, "B": 2000}
    dm_stock_cat_res = dm_all_cat.filter(
        {
            "Variables": ["bld_floor-area_stock"],
            "Categories1": ["single-family-households", "multi-family-households"],
        }
    )
    dm_age = compute_building_age(dm_stock_cat_res, years_fts, first_bld)

    first_bld_nonres = {"F": 1900, "E": 1946, "D": 1970, "C": 1990, "B": 2011}
    dm_age_nonres = dm_services_floor.filter(
        {"Variables": ["bld_floor-area_services"]}, inplace=False
    )
    dm_age_nonres.rename_col("bld_floor-area_services", "bld_age", dim="Variables")
    dm_age_nonres.change_unit("bld_age", 1, "m2", "years")
    dm_age_nonres.add(np.nan, dim="Years", col_label=years_fts, dummy=True)
    _years_all_nonres = np.array(dm_age_nonres.col_labels["Years"])
    _idx_nonres = dm_age_nonres.idx
    for _bld_type in NON_RES_CATS:
        for _cat, _start_yr in first_bld_nonres.items():
            _arr_age = np.maximum(_years_all_nonres - _start_yr, 0)
            for _ic in range(len(dm_age_nonres.col_labels["Country"])):
                dm_age_nonres.array[
                    _ic,
                    :,
                    _idx_nonres["bld_age"],
                    _idx_nonres[_bld_type],
                    _idx_nonres[_cat],
                ] = _arr_age
    dm_age.append(dm_age_nonres, dim="Categories1")
    DM_buildings["fxa"]["bld_age"] = dm_age

    ###############################################################
    # FXA: u-value + surface-to-floorarea
    ###############################################################
    DM_buildings["fxa"]["u-value"] = dm_u_value
    DM_buildings["fxa"]["surface-to-floorarea"] = dm_s2f.copy()

    ###############################################################
    # OTS: heating technology mix
    ###############################################################
    DM_buildings["ots"]["heating-technology-fuel"] = dict()

    dm_temp = dm_heating_cat.filter({"Categories1": ["sh"]})
    dm_temp.group_all("Categories1")
    for v in dm_temp.col_labels["Variables"]:
        dm_temp.rename_col(v, "bld_heating-mix_" + v, "Variables")
    dm_temp.deepen(based_on="Variables")
    dm_temp.switch_categories_order("Categories1", "Categories3")
    DM_buildings["ots"]["heating-technology-fuel"]["bld_heating-technology"] = (
        dm_temp.copy()
    )

    dm_temp = dm_heating_cat.filter({"Categories1": ["dhw"]})
    dm_temp.group_all("Categories1")
    for v in dm_temp.col_labels["Variables"]:
        dm_temp.rename_col(v, "bld_hw_tech-mix_" + v, "Variables")
    dm_temp.deepen(based_on="Variables")
    dm_temp = dm_temp.filter(
        {"Categories3": ["multi-family-households", "single-family-households"]}
    )
    dm_temp.group_all("Categories3", aggregation="mean")
    dm_temp.group_all("Categories2", aggregation="mean")
    dm_temp.normalise("Categories1")
    DM_buildings["ots"]["heating-technology-fuel"]["bld_hot-water-technology"] = (
        dm_temp.copy()
    )

    ###############################################################
    # OTS: heating efficiency
    ###############################################################
    dm_heating_eff = dm_heating_eff_cat.copy()
    dm_heating_eff.sort("Categories2")
    dm_heating_eff.filter(
        {"Categories1": dm_heating_cat.col_labels["Categories2"]}, inplace=True
    )
    dm_heating_eff.sort("Categories1")
    dm_heating_eff.switch_categories_order()
    DM_buildings["ots"]["heating-efficiency"] = dm_heating_eff.copy()

    dm_temp = dm_heating_eff_all.filter({"Categories1": ["dhw"]})
    dm_temp.group_all("Categories1")
    dm_temp.filter(
        {
            "Categories1": DM_bld["fxa"]["hot-water"]["hw-efficiency"].col_labels[
                "Categories1"
            ]
        },
        inplace=True,
    )
    dm_temp_fts = calculate_heating_eff_fts(dm_temp.copy(), years_fts, maximum_eff=0.98)
    dm_temp.append(dm_temp_fts, "Years")
    dm_temp[:, :, "bld_heating-efficiency", "heat-pump"] = np.nan
    dm_temp.rename_col(
        "bld_heating-efficiency", "bld_hot-water_efficiency", "Variables"
    )
    dm_heating_eff_dhw = dm_temp.copy()

    ###############################################################
    # Constant: emission factors
    ###############################################################
    JRC_emissions_fact = {
        "coal": 350,
        "heating-oil": 267,
        "gas": 200,
        "wood": 0,
        "solar": 0,
    }
    cdm_emission_fact = ConstantDataMatrix(
        col_labels={
            "Variables": ["bld_CO2-factors"],
            "Categories1": ["coal", "heating-oil", "gas", "wood", "solar"],
        },
        units={"bld_CO2-factors": "kt/TWh"},
    )
    cdm_emission_fact.array = np.zeros(
        (
            len(cdm_emission_fact.col_labels["Variables"]),
            len(cdm_emission_fact.col_labels["Categories1"]),
        )
    )
    idx = cdm_emission_fact.idx
    for key, value in JRC_emissions_fact.items():
        cdm_emission_fact.array[0, idx[key]] = value
    cdm_emission_fact.sort("Categories1")
    DM_buildings["constant"]["emissions"] = cdm_emission_fact

    ###############################################################
    # FXA: electricity emission factor
    ###############################################################
    col_dict = {
        "Country": dm_heating_eff.col_labels["Country"],
        "Years": years_ots + years_fts,
        "Variables": ["bld_CO2-factor"],
        "Categories1": ["electricity"],
    }
    dm_elec = DataMatrix(col_labels=col_dict, units={"bld_CO2-factor": "kt/TWh"})
    arr_elec = np.zeros((len(col_dict["Country"]), 40, 1, 1))
    idx = dm_elec.idx
    arr_elec[:, idx[1990] : idx[2023] + 1, 0, 0] = 112
    arr_elec[:, idx[2025] : idx[2050], 0, 0] = np.nan
    arr_elec[:, idx[2050], 0, 0] = 0
    dm_elec.array = arr_elec
    dm_elec.fill_nans(dim_to_interp="Years")
    DM_buildings["fxa"]["emission-factor-electricity"] = dm_elec

    ###############################################################
    # FXA: district heating emission factor (proxy from CH × factor)
    ###############################################################
    EU27_DH_FACTOR = 3.0
    dm_dh_ch = DM_bld["fxa"]["emission-factor-heating_district"].filter(
        {"Country": ["Switzerland"]}, inplace=False
    )
    col_dict_dh = {
        "Country": ["EU27"],
        "Years": dm_dh_ch.col_labels["Years"],
        "Variables": dm_dh_ch.col_labels["Variables"],
        "Categories1": dm_dh_ch.col_labels["Categories1"],
    }
    dm_dh_eu27 = DataMatrix(col_labels=col_dict_dh, units=dm_dh_ch.units)
    dm_dh_eu27.array = dm_dh_ch.array * EU27_DH_FACTOR
    DM_buildings["fxa"]["emission-factor-heating_district"] = dm_dh_eu27

    ###############################################################
    # FXA: appliances
    ###############################################################
    dm_domapp = get_domapp(DM_bld, years_ots, years_fts)
    DM_buildings["fxa"]["appliances"] = dm_domapp.copy()

    ###############################################################
    # FXA: hot water
    ###############################################################
    dm_dhw_cap = get_energy_demand_useful_dhw(years_ots, years_fts)
    DM_buildings["fxa"]["hot-water"] = dict()
    DM_buildings["fxa"]["hot-water"]["hw-energy-demand"] = dm_dhw_cap.copy()
    DM_buildings["fxa"]["hot-water"]["hw-efficiency"] = dm_heating_eff_dhw.copy()

    ###############################################################
    # FXA: services
    ###############################################################
    dm_demand, dm_tech_mix, dm_eff = get_services(years_ots, years_fts)
    dm_demand.drop("Categories1", "space-heating")

    dm_floor_total = dm_services_floor.group_all("Categories2", inplace=False)
    dm_floor_total.group_all("Categories1", inplace=True)
    idx_dem = dm_demand.idx
    idx_flo = dm_floor_total.idx
    # Last OTS year with floor data — used as fallback for FTS years, which have no
    # floor area in dm_floor_total (OTS-only source). Without this, FTS years would
    # skip normalisation and stay in absolute TWh, causing a ~10× OTS→FTS jump in
    # the module (which multiplies kWh/m² back by floor area for every year).
    last_flo_yr = max(y for y in dm_demand.col_labels["Years"] if y in idx_flo)
    for country in dm_demand.col_labels["Country"]:
        if country not in idx_flo:
            continue
        ic_dem = idx_dem[country]
        ic_flo = idx_flo[country]
        for yr in dm_demand.col_labels["Years"]:
            ref_yr = yr if yr in idx_flo else last_flo_yr
            floor_ref = dm_floor_total.array[ic_flo, idx_flo[ref_yr], 0]
            if floor_ref > 0:
                dm_demand.array[ic_dem, idx_dem[yr], :, :] *= 1e9 / floor_ref
    dm_demand.units["bld_services_useful-energy"] = "kWh/m2"

    DM_buildings["fxa"]["services"] = dict()
    DM_buildings["fxa"]["services"]["services_demand"] = dm_demand.copy()
    DM_buildings["fxa"]["services"]["services_tech-mix"] = dm_tech_mix.copy()
    DM_buildings["fxa"]["services"]["services_efficiencies"] = dm_eff.copy()

    ###############################################################
    # FXA: lighting + other electricity demand
    ###############################################################
    dm_domapp_lighting, dm_domapp_other = get_domapp_other(years_ots, years_fts)
    DM_buildings["fxa"]["lighting"] = dm_domapp_lighting.copy()
    DM_buildings["fxa"]["other-electricity-demand"] = dm_domapp_other.copy()

    result = DM_buildings
    os.makedirs(os.path.dirname(_OTS_CACHE_PATH), exist_ok=True)
    with open(_OTS_CACHE_PATH, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved OTS cache → {_OTS_CACHE_PATH}")
    return result


def run(DM_buildings_ots, DM_fts, country_list=["EU27"]):
    filter_nested_structure(DM_buildings_ots["ots"], country_list)
    filter_nested_structure(DM_buildings_ots["fxa"], country_list)
    filter_nested_structure(DM_fts, country_list)
    DM_buildings = {**DM_buildings_ots, "fts": DM_fts}
    os.makedirs(os.path.dirname(_OUTPUT_PATH), exist_ok=True)
    my_pickle_dump(DM_buildings, _OUTPUT_PATH)
    print(f"  Saved buildings.pickle → {_OUTPUT_PATH}")
