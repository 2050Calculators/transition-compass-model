import os
import re

import numpy as np
import pandas as pd

from transition_compass_model._database.pre_processing.industry.eu.get_data_functions.data_material_net_import import (
    get_faostat_sawnwood_data,
    get_prodcom_data,
)


def get_fxa_energy_demand_df(current_file_directory):
    # Returns (df_final, df_final_feedstock) with columns:
    # ["year", "variable", "value", "tech", "energy_demand_type"]
    # where variable = "tech_carrier[TWh/Mt]" and year is the JRC data year.
    # Year range is detected automatically from the JRC Excel file.
    # JRC-IDEES-2021 covers 2000-2021; older editions may cover back to 1990.

    ###########################################################
    ############## CATEGORIES OF ENERGY CARRIERS ##############
    ###########################################################

    enercarr_jrc = [
        "Diesel oil and liquid biofuels",
        "Natural gas and biogas",
        "Solar and geothermal",
        "Ambient heat",
        "Electricity",
        "Solids",
        "Fuel oil",
        "Derived gases",
        "Coke",
        "LPG",
        "Refinery gas",
        "Other liquids",
        "Biomass and waste",
        "Distributed steam",
    ]
    electric_usual = ["Lighting", "Air compressors", "Motor drives", "Fans and pumps"]

    enercarr_jrc_agg_map = [
        "Diesel oil and liquid biofuels",
        "Natural gas and biogas",
        "Solar and geothermal",
        "Ambient heat",
        "Electricity",
        "Hard coal and others",
        "Fuel oil",
        "Derived gases",
        "Coke",
        "LPG",
        "Refinery gas",
        "Other liquids",
        "Biomass and waste",
        "Distributed steam",
    ]

    enercarr_jrc_agg_map_adj = {
        "Diesel oil and liquid biofuels": [
            "Diesel oil (without biofuels)",
            "Liquid biofuels",
        ],
        "Natural gas and biogas": ["Natural gas", "Biogas"],
        "Solar and geothermal": ["Solar", "Geothermal"],
    }
    enercarr_jrc_agg = enercarr_jrc_agg_map.copy()
    for key in enercarr_jrc_agg_map_adj.keys():
        enercarr_jrc_agg = enercarr_jrc_agg + enercarr_jrc_agg_map_adj[key]
        enercarr_jrc_agg.remove(key)

    enercarr_calc_map = [
        "electricity",
        "electricity",
        "solid-ff-coal",
        "liquid-ff-oil",
        "gas-ff-natural",
        "solid-ff-coal",
        "gas-ff-natural",
        "gas-ff-natural",
        "liquid-ff-oil",
        "Biomass and waste",
        "electricity",
        "liquid-ff-diesel",
        "liquid-bio",
        "gas-ff-natural",
        "gas-bio",
        "electricity",
        "electricity",
    ]

    waste_municipal = 76.7
    waste_industrial = 36.6
    waste = waste_municipal + waste_industrial
    solid_biomass = 471
    tot = waste + solid_biomass
    dict_adj_biomass_waste = {
        "solid-waste": np.round(waste / tot, 2),
        "solid-bio": np.round(solid_biomass / tot, 2),
    }

    ktoe_to_twh = 1000 * 11.63 / 1000000
    gj_to_twh = 2.7777777777778 * 1e-7

    filepath = os.path.join(
        current_file_directory,
        "../data/JRC-IDEES-2021/EU27/JRC-IDEES-2021_Industry_EU27.xlsx",
    )

    # detect available year columns dynamically from Excel
    _df_hdr = pd.read_excel(filepath, "ISI", nrows=1)
    years = sorted(
        [c for c in _df_hdr.columns if isinstance(c, int) and 1900 < c < 2100]
    )
    del _df_hdr

    del waste_municipal, waste_industrial, waste, solid_biomass, tot

    # Load PRODCOM production for non-modelled sectors.
    # These sectors use a value-added index (M€) as "Physical output" in JRC IDEES,
    # not tonnes. We replace it with actual PRODCOM production (kg→Mt) so the stored
    # FXA intensity is correctly in TWh/Mt.
    _dm_prodcom = get_prodcom_data("ds-056120", current_file_directory)
    _faostat_sawnwood = get_faostat_sawnwood_data(current_file_directory)
    _TECHS_CODE_TO_MAT = {
        "FBT": "fbt",
        "MAE": "mae",
        "OIS": [
            "ois",
            "asphalt",
        ],  # asphalt (C23.99.13) was remapped out of OIS PRODCOM; keep in denominator
        "TEL": "textiles",
        "TRE": "tra-equip",
        "WWP": "wwp",
    }

    def _prod_override_df(mat, techs_calc, jrc_years):
        """Return DataFrame[tech, year, production (Mt), unit_production] from PRODCOM.

        mat may be a string or list of strings (summed across materials).
        Fills NaN early years via backward fill from the first available year.
        """
        idx = _dm_prodcom.idx
        prodcom_years = _dm_prodcom.col_labels["Years"]
        mats = [mat] if isinstance(mat, str) else mat
        prod_by_year = {}
        for m in mats:
            for yr in prodcom_years:
                val = _dm_prodcom.array[
                    idx["EU27"], idx[yr], idx["material-demand"], idx[m]
                ]
                prod_by_year[yr] = prod_by_year.get(yr, 0) + (
                    0 if np.isnan(val) else val
                )
        available = {yr: v for yr, v in prod_by_year.items() if v > 0}
        if not available:
            raise ValueError(
                f"No PRODCOM production data available for material '{mat}'"
            )
        avail_years = sorted(available.keys())
        avail_vals = [available[y] for y in avail_years]
        rows = []
        for yr in jrc_years:
            if yr in available:
                prod_kg = available[yr]
            elif yr < avail_years[0]:
                prod_kg = avail_vals[0]
            elif yr > avail_years[-1]:
                prod_kg = avail_vals[-1]
            else:
                prod_kg = float(np.interp(yr, avail_years, avail_vals))
            rows.append(
                {
                    "tech": techs_calc[0],
                    "year": yr,
                    "production": prod_kg / 1e9,  # kg → Mt
                    "unit_production": "Mt",
                }
            )
        return pd.DataFrame(rows)

    ###########################################################
    ##### INNER FUNCTION: get_energy_intensity (year-aware) ###
    ###########################################################

    def get_energy_intensity(
        filepath,
        techs_code,
        techs,
        techs_calc,
        dict_subtechs_elec,
        years,
        df_po_override=None,
    ):
        # get energy consumption aggregated across techs (toe)
        df = pd.read_excel(filepath, techs_code)
        id_var = df.columns[0]
        df = df.loc[:, [id_var] + years]
        ls_temp = list(df.iloc[:, 0].isin(["Energy consumption (ktoe)"]))
        if techs_code in ["OIS"]:
            ls_temp = list(df.iloc[:, 0].isin(["Energy consumption (ktoe)*"]))
        index_row_start = [i + 1 for i, x in enumerate(ls_temp) if x]
        df_temp = df.iloc[range(index_row_start[0], len(df)), :]
        ls_temp = list(df_temp.iloc[:, 0].isin(["by subsector (calibration output)"]))
        index_row_end = [i + 1 for i, x in enumerate(ls_temp) if x]
        if len(index_row_end) > 0:
            index_row_end = index_row_end[0] - 1
        else:
            ls_temp = list(df_temp.iloc[:, 0].isin(["CO2 emissions (kt CO2)"]))
            index_row_end = [i - 1 for i, x in enumerate(ls_temp) if x][0]
        df_ec = df_temp.iloc[range(1, index_row_end), :]
        # keep year dimension
        df_ec = pd.melt(df_ec, id_vars=id_var, var_name="year")
        df_ec["year"] = df_ec["year"].astype(int)
        df_ec = df_ec.rename(columns={id_var: "energy_carrier"})
        df_ec = df_ec.loc[
            ~df_ec["energy_carrier"].isin(
                ["Solids", "Liquids", "Gas", "RES and wastes"]
            )
        ]

        # get physical output — keep year dimension
        ls_temp = list(df.iloc[:, 0])
        ls_temp = [str(i) for i in ls_temp]
        ls_temp = [
            bool(re.search("Physical output", i, re.IGNORECASE)) for i in ls_temp
        ]
        index_row_start = [i + 1 for i, x in enumerate(ls_temp) if x][0]
        if techs_code in ["FBT", "TRE", "MAE", "TEL", "WWP", "OIS"]:
            index_row_start = [i for i, x in enumerate(ls_temp) if x][0]
        index_row_end = index_row_start + len(techs)
        if techs_code == "NFM":
            index_row_end = index_row_start + len(techs) + 1
        df_po = df.iloc[range(index_row_start, index_row_end), :]
        df_po = pd.melt(df_po, id_vars=id_var, var_name="year")
        df_po["year"] = df_po["year"].astype(int)
        df_po = df_po.rename(columns={id_var: "tech"})
        for i in range(0, len(techs)):
            idx = [bool(re.search(techs[i], string)) for string in df_po["tech"]]
            df_po.loc[idx, "tech"] = techs_calc[i]
        if techs_code in ["FBT", "TRE", "MAE", "TEL", "WWP", "OIS"]:
            df_po["tech"] = techs_calc[0]

        # get specific data on final energy consumption (fec)
        df = pd.read_excel(filepath, techs_code + "_fec")
        id_var = df.columns[0]
        df = df.loc[:, [id_var] + years]
        ls_temp = list(
            df.iloc[:, 0].isin(
                ["Detailed split of energy consumption by subsector (ktoe)"]
            )
        )
        index_row_start = [i + 1 for i, x in enumerate(ls_temp) if x]
        if techs_code in ["FBT", "TRE", "MAE", "TEL", "WWP", "OIS"]:
            ls_temp = list(
                df.iloc[:, 0].isin(["Detailed split of energy consumption (ktoe)"])
            )
            index_row_start = [i + 2 for i, x in enumerate(ls_temp) if x]
        ls_temp = list(
            df.iloc[:, 0].isin(["Market shares of energy uses by subsector (%)"])
        )
        index_row_end = [i - 2 for i, x in enumerate(ls_temp) if x]
        if techs_code in ["FBT", "TRE", "MAE", "TEL", "WWP", "OIS"]:
            ls_temp = list(df.iloc[:, 0].isin(["Market shares of energy uses (%)"]))
            index_row_end = [i - 1 for i, x in enumerate(ls_temp) if x]
        df = df.loc[range(index_row_start[0], index_row_end[0]), :]
        DF = {"fec": df.copy()}

        # get specific data on useful energy demand (ued)
        df = pd.read_excel(filepath, techs_code + "_ued")
        id_var = df.columns[0]
        df = df.loc[:, [id_var] + years]
        ls_temp = list(
            df.iloc[:, 0].isin(
                ["Detailed split of useful energy demand by subsector (ktoe)"]
            )
        )
        index_row_start = [i + 1 for i, x in enumerate(ls_temp) if x]
        if techs_code in ["FBT", "TRE", "MAE", "TEL", "WWP", "OIS"]:
            ls_temp = list(
                df.iloc[:, 0].isin(["Detailed split of useful energy demand (ktoe)"])
            )
            index_row_start = [i + 2 for i, x in enumerate(ls_temp) if x]
        ls_temp = list(
            df.iloc[:, 0].isin(
                ["Market shares of useful energy demand by subsector (%)"]
            )
        )
        index_row_end = [i - 2 for i, x in enumerate(ls_temp) if x]
        if techs_code in ["FBT", "TRE", "MAE", "TEL", "WWP", "OIS"]:
            ls_temp = list(
                df.iloc[:, 0].isin(["Market shares of useful energy demand (%)"])
            )
            index_row_end = [i - 1 for i, x in enumerate(ls_temp) if x]
        df = df.loc[range(index_row_start[0], index_row_end[0]), :]
        DF["ued"] = df.copy()

        # process each fec/ued sheet, keeping year dimension
        DF_techs = {}
        for energy_key in DF.keys():
            df = DF[energy_key].copy()

            for i in range(0, len(techs)):
                # subset rows for tech i
                ls_temp = list(df.iloc[:, 0].isin([techs[i]]))
                index_row_start = [i + 1 for i, x in enumerate(ls_temp) if x]
                if i != len(techs) - 1:
                    ls_temp = list(df.iloc[:, 0].isin([techs[i + 1]]))
                    if techs_code == "PPA":
                        index_row_end = [i - 1 for i, x in enumerate(ls_temp) if x]
                    else:
                        index_row_end = [i - 2 for i, x in enumerate(ls_temp) if x]
                else:
                    index_row_end = [len(df)]
                df_temp = df.iloc[range(index_row_start[0], index_row_end[0]), :]

                # aggregate subtechs to electricity (year columns summed)
                id_var = df_temp.columns[0]
                df_temp = df_temp.loc[
                    df_temp[id_var].isin(enercarr_jrc + dict_subtechs_elec[techs[i]]), :
                ]
                mysubset = dict_subtechs_elec[techs[i]].copy()
                for lbl in [
                    "Lighting",
                    "Air compressors",
                    "Motor drives",
                    "Fans and pumps",
                ]:
                    mysubset.remove(lbl)
                df_temp.loc[df_temp[id_var].isin(mysubset), id_var] = "Electricity"
                df_temp = df_temp.groupby([id_var], as_index=False).agg(sum)

                # melt to long format, preserving year
                df_temp = pd.melt(df_temp, id_vars=id_var, var_name="year")
                df_temp["year"] = df_temp["year"].astype(int)
                df_temp = df_temp.rename(columns={id_var: "energy_carrier"})
                # columns: ["energy_carrier", "year", "value"]

                # apply enercarr_jrc_agg_map (simple rename, no year needed)
                for idx in range(0, len(enercarr_jrc)):
                    df_temp.loc[
                        df_temp["energy_carrier"] == enercarr_jrc[idx], "energy_carrier"
                    ] = enercarr_jrc_agg_map[idx]

                # adjust split carriers using df_ec per year
                for key in enercarr_jrc_agg_map_adj.keys():
                    ec_sub = df_ec.loc[
                        df_ec["energy_carrier"].isin(enercarr_jrc_agg_map_adj[key]),
                        ["energy_carrier", "year", "value"],
                    ]
                    ec_total = (
                        ec_sub.groupby("year", as_index=False)["value"]
                        .sum()
                        .rename(columns={"value": "ec_total"})
                    )
                    for idx in range(0, len(enercarr_jrc_agg_map_adj[key])):
                        sub_carr = enercarr_jrc_agg_map_adj[key][idx]
                        ec_part = ec_sub.loc[
                            ec_sub["energy_carrier"] == sub_carr, ["year", "value"]
                        ].rename(columns={"value": "ec_part"})
                        ec_ratio = pd.merge(ec_part, ec_total, on="year")
                        ec_ratio["ratio"] = ec_ratio["ec_part"] / ec_ratio["ec_total"]
                        ec_ratio["ratio"] = ec_ratio["ratio"].fillna(0)
                        ec_ratio = ec_ratio[["year", "ratio"]]
                        key_rows = df_temp.loc[
                            df_temp["energy_carrier"] == key,
                            ["energy_carrier", "year", "value"],
                        ].copy()
                        key_rows = pd.merge(key_rows, ec_ratio, on="year")
                        key_rows["value"] = key_rows["value"] * key_rows["ratio"]
                        key_rows["energy_carrier"] = sub_carr
                        key_rows = key_rows[["energy_carrier", "year", "value"]]
                        df_temp = pd.concat([df_temp, key_rows], ignore_index=True)
                    df_temp = df_temp.loc[df_temp["energy_carrier"] != key, :]

                # map to calculator carrier names
                for idx in range(0, len(enercarr_jrc_agg)):
                    df_temp.loc[
                        df_temp["energy_carrier"] == enercarr_jrc_agg[idx],
                        "energy_carrier",
                    ] = enercarr_calc_map[idx]
                df_temp = df_temp.groupby(
                    ["energy_carrier", "year"], as_index=False
                ).agg(sum)
                df_temp.loc[
                    df_temp["energy_carrier"] == "Lighting", "energy_carrier"
                ] = "lighting"
                df_temp.loc[
                    df_temp["energy_carrier"].isin(
                        ["Air compressors", "Motor drives", "Fans and pumps"]
                    ),
                    "energy_carrier",
                ] = "electricity-else"
                df_temp = df_temp.groupby(
                    ["energy_carrier", "year"], as_index=False
                ).agg(sum)

                # split biomass and waste
                for key in dict_adj_biomass_waste.keys():
                    bw_rows = df_temp.loc[
                        df_temp["energy_carrier"] == "Biomass and waste",
                        ["year", "value"],
                    ].copy()
                    bw_rows["energy_carrier"] = key
                    bw_rows["value"] = bw_rows["value"] * dict_adj_biomass_waste[key]
                    df_temp = pd.concat(
                        [df_temp, bw_rows[["energy_carrier", "year", "value"]]],
                        ignore_index=True,
                    )
                df_temp = df_temp.loc[
                    df_temp["energy_carrier"] != "Biomass and waste", :
                ]

                # add hydrogen (zero, all years)
                all_years_temp = df_temp["year"].unique().tolist()
                h2_rows = pd.DataFrame(
                    {"energy_carrier": "hydrogen", "year": all_years_temp, "value": 0.0}
                )
                df_temp = pd.concat([df_temp, h2_rows], ignore_index=True)

                DF_techs[energy_key + "_" + techs[i]] = df_temp

        # combine techs
        def get_df(t, t_calc, e):
            df_t = DF_techs[e + "_" + t].copy()
            df_t["tech"] = t_calc
            df_t["energy_demand_type"] = e
            return df_t

        df_fec = pd.concat(
            [get_df(t, t_calc, "fec") for t, t_calc in zip(techs, techs_calc)]
        )
        df_ued = pd.concat(
            [get_df(t, t_calc, "ued") for t, t_calc in zip(techs, techs_calc)]
        )
        df = pd.concat([df_fec, df_ued])

        # normalize by production per year (objective: TWh/Mt)
        if df_po_override is not None:
            # PRODCOM production in Mt, per year — replaces JRC value-added index
            df_po_norm = df_po_override
        else:
            df_po_norm = df_po.copy()
            df_po_norm["value"] = df_po_norm["value"] / 1000  # Kt → Mt
            df_po_norm = df_po_norm.rename(columns={"value": "production"})
            df_po_norm["unit_production"] = "Mt"

        df["value"] = df["value"] * ktoe_to_twh  # ktoe → TWh
        df["unit"] = "TWh"
        df = pd.merge(df, df_po_norm, how="left", on=["tech", "year"])
        df["value"] = df["value"] / df["production"]

        # build variable names
        df["variable"] = [
            tech + "_" + enercarr + "[" + unit + "/" + unit_production + "]"
            for tech, enercarr, unit, unit_production in zip(
                df["tech"], df["energy_carrier"], df["unit"], df["unit_production"]
            )
        ]
        df = df.loc[:, ["year", "variable", "value", "tech", "energy_demand_type"]]

        # check (diagnostic only)
        df_po_mean = df_po.groupby("tech", as_index=False)["value"].mean()
        df_po_mean.columns = ["tech", "total"]
        df_po_mean["unit_production"] = "Kt"
        df_fec_agg = df_fec.groupby("tech", as_index=False)["value"].sum()
        df_fec_agg = pd.merge(df_fec_agg, df_po_mean, how="left", on="tech")
        df_fec_agg["ratio"] = df_fec_agg["value"] / df_fec_agg["total"]
        df_energyint_check = df_fec_agg.copy()

        DF_out = {
            "tot": df_fec,
            "tot-ued": df_ued,
            "energy-intensity-check": df_energyint_check,
            "energy-intensity": df,
            "production": df_po,
        }
        return DF_out

    ###########################################################
    ########################## STEEL ##########################
    ###########################################################

    techs_code = "ISI"
    techs = ["Integrated steelworks", "Electric arc"]
    techs_calc = ["steel-BF-BOF", "steel-scrap-EAF"]
    dict_subtechs_elec = {
        "Integrated steelworks": electric_usual
        + [
            "Steel: Furnaces, refining and rolling - Electric",
            "Steel: Product finishing - Electric",
        ],
        "Electric arc": electric_usual
        + [
            "Steel: Electric arc",
            "Steel: Furnaces, refining and rolling - Electric",
            "Steel: Product finishing - Electric",
        ],
    }

    DF = get_energy_intensity(
        filepath=filepath,
        techs_code=techs_code,
        techs=techs,
        techs_calc=techs_calc,
        dict_subtechs_elec=dict_subtechs_elec,
        years=years,
    )
    df = DF["energy-intensity"].copy()

    # steel-hisarna: 20% less energy than steel-BF-BOF
    df_hi = df.loc[df["tech"] == "steel-BF-BOF", :].copy()
    df_hi["value"] = df_hi["value"] * (1 - 0.2)
    df_hi["tech"] = "steel-hisarna"
    df_hi["variable"] = [
        i.replace("steel-BF-BOF", "steel-hisarna") for i in df_hi["variable"]
    ]
    df = pd.concat([df, df_hi])

    # steel-hydrog-DRI: 3.48 TWh/Mt, 50% H2 + 50% electricity
    df_dri = df_hi.copy()
    df_dri["value"] = 0
    df_dri["variable"] = [
        i.replace("steel-hisarna", "steel-hydrog-DRI") for i in df_dri["variable"]
    ]
    df_dri["tech"] = "steel-hydrog-DRI"
    for yr in df_dri["year"].unique():
        mask = df_dri["variable"].str.contains(
            "steel-hydrog-DRI_electricity[", regex=False
        ) & (df_dri["year"] == yr)
        df_dri.loc[mask, "value"] = 0.5 * 3.48
        mask = df_dri["variable"].str.contains(
            "steel-hydrog-DRI_hydrogen[", regex=False
        ) & (df_dri["year"] == yr)
        df_dri.loc[mask, "value"] = 0.5 * 3.48
    df = pd.concat(
        [df, df_dri.loc[:, ["year", "variable", "value", "tech", "energy_demand_type"]]]
    )

    df_final = df.copy()
    del DF, df, df_dri, df_hi, key, techs, techs_calc, techs_code

    ######################################################################
    ########################## CEMENT AND GLASS ##########################
    ######################################################################

    techs_code = "NMM"
    techs = ["Cement", "Ceramics & other NMM", "Glass production"]
    techs_calc = ["cement", "ceramics", "glass-glass"]
    dict_subtechs_elec = {
        "Cement": electric_usual
        + [
            "Cement: Grinding, milling of raw material",
            "Cement: Grinding, packaging and precasting (electricity)",
        ],
        "Ceramics & other NMM": electric_usual
        + [
            "Ceramics: Mixing of raw material",
            "Ceramics: Microwave drying and sintering",
            "Ceramics: Electric kiln",
            "Ceramics: Electric furnace",
        ],
        "Glass production": electric_usual
        + [
            "Glass: Electric melting tank",
            "Glass: Forming",
            "Glass: Annealing - electric",
            "Glass: Finishing processes",
        ],
    }

    DF = get_energy_intensity(
        filepath=filepath,
        techs_code=techs_code,
        techs=techs,
        techs_calc=techs_calc,
        dict_subtechs_elec=dict_subtechs_elec,
        years=years,
    )
    df = DF["energy-intensity"].copy()

    # save ceramics absolute energy (ktoe) to fold into OIS intensity below:
    # ceramics PRODCOM codes (bricks, tiles, refractories) are mapped to OIS,
    # but their JRC energy comes from the NMM sheet. Use absolute ktoe (not intensity)
    # because JRC ceramics production is in "kt bricks-eq." — a virtual unit incompatible
    # with PRODCOM mass-based production used for OIS normalisation.
    df_ceramics_fec_abs = DF["tot"].loc[DF["tot"]["tech"] == "ceramics"].copy()
    df_ceramics_ued_abs = DF["tot-ued"].loc[DF["tot-ued"]["tech"] == "ceramics"].copy()

    # drop ceramics from NMM output (cement + glass only)
    df = df.loc[df["tech"] != "ceramics", :]

    # add zero solid-waste and solid-bio for glass (not present in JRC for glass)
    all_glass_years = df.loc[df["tech"] == "glass-glass", "year"].unique().tolist()
    for ftype in ["fec", "ued"]:
        for carrier in ["solid-waste", "solid-bio"]:
            zero_rows = pd.DataFrame(
                {
                    "year": all_glass_years,
                    "variable": f"glass-glass_{carrier}[TWh/Mt]",
                    "value": 0.0,
                    "tech": "glass-glass",
                    "energy_demand_type": ftype,
                }
            )
            df = pd.concat([df, zero_rows], ignore_index=True)
    df.sort_values(["energy_demand_type", "tech"], inplace=True)

    # cement subtechs: dry-kiln, wet-kiln, geopolym
    # use JRC carrier shares from "cement", rescaled by literature total energy
    ls_cement_tech_ec = {
        "cement-dry-kiln": 3.38 * gj_to_twh * 1000000,
        "cement-wet-kiln": 6.34 * gj_to_twh * 1000000,
        "cement-geopolym": 0.65,
    }
    for ctech, lit_total in ls_cement_tech_ec.items():
        df_cem_fec = df.loc[
            (df["tech"] == "cement") & (df["energy_demand_type"] == "fec"), :
        ].copy()
        df_year_total = (
            df_cem_fec.groupby("year", as_index=False)["value"]
            .sum()
            .rename(columns={"value": "total"})
        )
        df_cem_fec = pd.merge(df_cem_fec, df_year_total, on="year")
        df_cem_fec["share"] = df_cem_fec["value"] / df_cem_fec["total"]
        df_cem_fec["value"] = df_cem_fec["share"] * lit_total
        df_cem_fec.drop(columns=["total", "share"], inplace=True)
        df_cem_fec["tech"] = ctech
        df_cem_fec["variable"] = [
            v.replace("cement", ctech) for v in df_cem_fec["variable"]
        ]
        df = pd.concat([df, df_cem_fec], ignore_index=True)

    # cement subtechs UED: apply efficiency from cement (averaged across years)
    df_ced = df.loc[df["tech"] == "cement", :]
    df_eff_cem = df_ced.pivot_table(
        index="variable", columns="energy_demand_type", values="value", aggfunc="mean"
    ).reset_index()
    df_eff_cem["efficiency"] = df_eff_cem["ued"] / df_eff_cem["fec"]
    df_eff_cem.loc[
        df_eff_cem["variable"] == "cement_hydrogen[TWh/Mt]", "efficiency"
    ] = 0.5
    df_eff_cem = df_eff_cem[["variable", "efficiency"]]

    for ctech in ls_cement_tech_ec.keys():
        df_sub_fec = df.loc[
            (df["tech"] == ctech) & (df["energy_demand_type"] == "fec"), :
        ].copy()
        df_sub_fec["var_cem"] = [
            v.replace(ctech, "cement") for v in df_sub_fec["variable"]
        ]
        df_sub_fec = pd.merge(
            df_sub_fec,
            df_eff_cem.rename(columns={"variable": "var_cem"}),
            on="var_cem",
            how="left",
        )
        df_sub_fec["efficiency"] = df_sub_fec["efficiency"].fillna(0.5)
        df_sub_fec["value"] = df_sub_fec["value"] * df_sub_fec["efficiency"]
        df_sub_fec["energy_demand_type"] = "ued"
        df_sub_fec = df_sub_fec.drop(columns=["var_cem", "efficiency"])
        df = pd.concat([df, df_sub_fec], ignore_index=True)

    df = df.loc[df["tech"] != "cement", :]

    # cement-sec: dry-kiln * (1 - avg_energy_reduction)
    ec_perc_less = np.mean(np.array([0, 0.30, 0.40]))
    df_temp = df.loc[df["tech"] == "cement-dry-kiln", :].copy()
    df_temp["value"] = df_temp["value"] * (1 - ec_perc_less)
    df_temp["tech"] = "cement-sec"
    df_temp["variable"] = [
        i.replace("cement-dry-kiln", "cement-sec") for i in df_temp["variable"]
    ]
    df = pd.concat([df, df_temp])

    # glass-sec: same as glass-glass
    df_temp = df.loc[df["tech"] == "glass-glass", :].copy()
    df_temp["tech"] = "glass-sec"
    df_temp["variable"] = [
        i.replace("glass-glass", "glass-sec") for i in df_temp["variable"]
    ]
    df = pd.concat([df, df_temp])

    df_final = pd.concat([df_final, df])
    del DF, df, df_temp, ls_cement_tech_ec, techs, techs_calc, techs_code

    ######################################################################
    ############################# CHEMICALS ##############################
    ######################################################################

    techs_code = "CHI"
    techs = ["Basic chemicals"]
    techs_calc = ["chem-chem-tech"]
    dict_subtechs_elec = {
        "Basic chemicals": electric_usual
        + [
            "Chemicals: Furnaces - Electric",
            "Chemicals: Process cooling - Electric",
            "Chemicals: Generic electric process",
        ]
    }

    # get energy consumption aggregated (for enercarr_jrc_agg_map_adj splits)
    df = pd.read_excel(filepath, techs_code)
    id_var = df.columns[0]
    df = df.loc[:, [id_var] + years]
    ls_temp = list(df.iloc[:, 0].isin(["Energy consumption (ktoe)"]))
    index_row_start = [i + 1 for i, x in enumerate(ls_temp) if x]
    df_temp = df.iloc[range(index_row_start[0], len(df)), :].reset_index(drop=True)
    df_ec = df_temp.iloc[range(1, 20), :]
    df_ec = pd.melt(df_ec, id_vars=[id_var], var_name="year")
    df_ec["year"] = df_ec["year"].astype(int)
    df_ec = df_ec.rename(columns={id_var: "energy_carrier"})
    df_ec_fs = df_temp.iloc[range(27, 39), :]
    df_ec_fs = pd.melt(df_ec_fs, id_vars=[id_var], var_name="year")
    df_ec_fs["year"] = df_ec_fs["year"].astype(int)
    df_ec_fs = df_ec_fs.rename(columns={id_var: "energy_carrier"})

    # get physical output — keep year dimension
    ls_temp = list(df.iloc[:, 0])
    ls_temp = [str(i) for i in ls_temp]
    ls_temp = [bool(re.search("Physical output", i, re.IGNORECASE)) for i in ls_temp]
    index_row_start = [i + 1 for i, x in enumerate(ls_temp) if x][0]
    index_row_end = index_row_start + len(techs)
    df_po = df.iloc[range(index_row_start, index_row_end), :]
    df_po = pd.melt(df_po, id_vars=id_var, var_name="year")
    df_po["year"] = df_po["year"].astype(int)
    df_po = df_po.rename(columns={id_var: "tech"})
    for i in range(0, len(techs)):
        idx = [bool(re.search(techs[i], string)) for string in df_po["tech"]]
        df_po.loc[idx, "tech"] = techs_calc[i]

    # get fec and ued sheets
    df_fec_sheet = pd.read_excel(filepath, techs_code + "_fec")
    id_var = df_fec_sheet.columns[0]
    df_fec_sheet = df_fec_sheet.loc[:, [id_var] + years]
    ls_temp = list(
        df_fec_sheet.iloc[:, 0].isin(
            ["Detailed split of energy consumption by subsector (ktoe)"]
        )
    )
    index_row_start = [i + 1 for i, x in enumerate(ls_temp) if x]
    ls_temp = list(
        df_fec_sheet.iloc[:, 0].isin(["Market shares of energy uses by subsector (%)"])
    )
    index_row_end = [i - 2 for i, x in enumerate(ls_temp) if x]
    df_fec_sheet = df_fec_sheet.loc[range(index_row_start[0], index_row_end[0]), :]
    DF_temp = {"fec": df_fec_sheet.copy()}

    df_ued_sheet = pd.read_excel(filepath, techs_code + "_ued")
    id_var = df_ued_sheet.columns[0]
    df_ued_sheet = df_ued_sheet.loc[:, [id_var] + years]
    ls_temp = list(
        df_ued_sheet.iloc[:, 0].isin(
            ["Detailed split of useful energy demand by subsector (ktoe)"]
        )
    )
    index_row_start = [i + 1 for i, x in enumerate(ls_temp) if x]
    ls_temp = list(
        df_ued_sheet.iloc[:, 0].isin(
            ["Market shares of useful energy demand by subsector (%)"]
        )
    )
    index_row_end = [i - 2 for i, x in enumerate(ls_temp) if x]
    df_ued_sheet = df_ued_sheet.loc[range(index_row_start[0], index_row_end[0]), :]
    DF_temp["ued"] = df_ued_sheet.copy()

    df_final_feedstock = pd.DataFrame()
    for energy_demand_type in ["fec", "ued"]:
        df = DF_temp[energy_demand_type].copy()
        id_var = df.columns[0]

        i = 0
        ls_temp = list(df.iloc[:, 0].isin([techs[i]]))
        index_row_start = [i + 1 for i, x in enumerate(ls_temp) if x]
        ls_temp = list(df.iloc[:, 0].isin(["Other chemicals"]))
        index_row_end = [i - 2 for i, x in enumerate(ls_temp) if x]
        df_temp = df.iloc[range(index_row_start[0], index_row_end[0]), :]

        # separate feedstock rows
        ls_temp = list(
            df.iloc[:, 0].isin(["Chemicals: Feedstock (energy used as raw material)"])
        )
        index_row_start_fs = [i + 1 for i, x in enumerate(ls_temp) if x][0]
        index_row_end_fs = index_row_start_fs + 8
        df_temp_fs = df.iloc[range(index_row_start_fs, index_row_end_fs), :]
        ls_temp2 = list(
            df_temp.iloc[:, 0].isin(
                ["Chemicals: Feedstock (energy used as raw material)"]
            )
        )
        index_row_start_fs2 = [i + 1 for i, x in enumerate(ls_temp2) if x][0]
        index_row_end_fs2 = index_row_start_fs2 + 8
        df_temp = df_temp.loc[
            [
                i not in range(index_row_start_fs2 - 1, index_row_end_fs2)
                for i in range(len(df_temp))
            ],
            :,
        ]

        # separate process cooling natural gas
        df_temp_coolinggas = df_temp.loc[
            df_temp.iloc[:, 0] == "Chemicals: Process cooling - Natural gas and biogas",
            :,
        ]
        df_temp = df_temp.loc[
            df_temp.iloc[:, 0] != "Chemicals: Process cooling - Natural gas and biogas",
            :,
        ]

        # aggregate subtechs to electricity (year columns summed)
        id_var = df_temp.columns[0]
        df_temp = df_temp.loc[
            df_temp[id_var].isin(enercarr_jrc + dict_subtechs_elec[techs[i]]), :
        ]
        mytechs = dict_subtechs_elec[techs[i]].copy()
        mytechs.remove("Lighting")
        df_temp.loc[df_temp[id_var].isin(mytechs), id_var] = "Electricity"
        df_temp = df_temp.groupby([id_var], as_index=False).agg(sum)

        # add process cooling gas to natural gas (still wide format)
        for y in years:
            df_temp.loc[df_temp[id_var] == "Natural gas and biogas", y] = sum(
                np.array(df_temp.loc[df_temp[id_var] == "Natural gas and biogas", y]),
                np.array(df_temp_coolinggas.loc[:, y]),
            )

        # melt to long format, preserving year
        df_temp = pd.melt(df_temp, id_vars=id_var, var_name="year")
        df_temp["year"] = df_temp["year"].astype(int)
        df_temp = df_temp.rename(columns={id_var: "energy_carrier"})

        df_temp_fs = pd.melt(df_temp_fs, id_vars=id_var, var_name="year")
        df_temp_fs["year"] = df_temp_fs["year"].astype(int)
        df_temp_fs = df_temp_fs.rename(columns={id_var: "energy_carrier"})

        # apply JRC carrier mapping
        for idx in range(0, len(enercarr_jrc)):
            df_temp.loc[
                df_temp["energy_carrier"] == enercarr_jrc[idx], "energy_carrier"
            ] = enercarr_jrc_agg_map[idx]

        # adjust split carriers using df_ec per year
        for key in enercarr_jrc_agg_map_adj.keys():
            ec_sub = df_ec.loc[
                df_ec["energy_carrier"].isin(enercarr_jrc_agg_map_adj[key]),
                ["energy_carrier", "year", "value"],
            ]
            ec_total = (
                ec_sub.groupby("year", as_index=False)["value"]
                .sum()
                .rename(columns={"value": "ec_total"})
            )
            for idx in range(0, len(enercarr_jrc_agg_map_adj[key])):
                sub_carr = enercarr_jrc_agg_map_adj[key][idx]
                ec_part = ec_sub.loc[
                    ec_sub["energy_carrier"] == sub_carr, ["year", "value"]
                ].rename(columns={"value": "ec_part"})
                ec_ratio = pd.merge(ec_part, ec_total, on="year")
                ec_ratio["ratio"] = ec_ratio["ec_part"] / ec_ratio["ec_total"]
                ec_ratio["ratio"] = ec_ratio["ratio"].fillna(0)
                ec_ratio = ec_ratio[["year", "ratio"]]
                key_rows = df_temp.loc[
                    df_temp["energy_carrier"] == key,
                    ["energy_carrier", "year", "value"],
                ].copy()
                key_rows = pd.merge(key_rows, ec_ratio, on="year")
                key_rows["value"] = key_rows["value"] * key_rows["ratio"]
                key_rows["energy_carrier"] = sub_carr
                key_rows = key_rows[["energy_carrier", "year", "value"]]
                df_temp = pd.concat([df_temp, key_rows], ignore_index=True)
            df_temp = df_temp.loc[df_temp["energy_carrier"] != key, :]

        # map to calculator carrier names
        for idx in range(0, len(enercarr_jrc_agg)):
            df_temp.loc[
                df_temp["energy_carrier"] == enercarr_jrc_agg[idx], "energy_carrier"
            ] = enercarr_calc_map[idx]
        df_temp = df_temp.groupby(["energy_carrier", "year"], as_index=False).agg(sum)
        df_temp.loc[df_temp["energy_carrier"] == "Lighting", "energy_carrier"] = (
            "lighting"
        )

        # split biomass and waste
        for key in dict_adj_biomass_waste.keys():
            bw_rows = df_temp.loc[
                df_temp["energy_carrier"] == "Biomass and waste", ["year", "value"]
            ].copy()
            bw_rows["energy_carrier"] = key
            bw_rows["value"] = bw_rows["value"] * dict_adj_biomass_waste[key]
            df_temp = pd.concat(
                [df_temp, bw_rows[["energy_carrier", "year", "value"]]],
                ignore_index=True,
            )
        df_temp = df_temp.loc[df_temp["energy_carrier"] != "Biomass and waste", :]

        # add hydrogen
        all_years_temp = df_temp["year"].unique().tolist()
        h2_rows = pd.DataFrame(
            {"energy_carrier": "hydrogen", "year": all_years_temp, "value": 0.0}
        )
        df_temp = pd.concat([df_temp, h2_rows], ignore_index=True)
        df_temp["tech"] = techs_calc[0]

        # normalize by production per year (ktoe → TWh/Mt)
        df_po_temp = df_po.copy()
        df_po_temp = df_po_temp.rename(columns={"value": "production"})
        df_po_temp["production"] = df_po_temp["production"] / 1000  # Kt → Mt
        df_po_temp["unit_production"] = "Mt"

        df_excl = df_temp.copy()
        df_excl["value"] = df_excl["value"] * ktoe_to_twh
        df_excl["unit"] = "TWh"
        df_excl = pd.merge(df_excl, df_po_temp, how="left", on=["tech", "year"])
        df_excl["value"] = df_excl["value"] / df_excl["production"]
        df_excl["variable"] = [
            tech + "_" + enercarr + "[" + unit + "/" + unit_production + "]"
            for tech, enercarr, unit, unit_production in zip(
                df_excl["tech"],
                df_excl["energy_carrier"],
                df_excl["unit"],
                df_excl["unit_production"],
            )
        ]
        df_excl = df_excl.loc[:, ["year", "variable", "value", "tech"]]

        # chem-sec: same as chem-chem-tech
        df_temp_sec = df_excl.loc[df_excl["tech"] == "chem-chem-tech", :].copy()
        df_temp_sec["tech"] = "chem-sec"
        df_temp_sec["variable"] = [
            i.replace("chem-chem-tech", "chem-sec") for i in df_temp_sec["variable"]
        ]
        df_excl = pd.concat([df_excl, df_temp_sec])
        df_excl["energy_demand_type"] = energy_demand_type
        df_final = pd.concat([df_final, df_excl])

        # feedstock processing
        df_temp_fs_proc = df_temp_fs.copy()
        for idx in range(0, len(enercarr_jrc)):
            df_temp_fs_proc.loc[
                df_temp_fs_proc["energy_carrier"] == enercarr_jrc[idx], "energy_carrier"
            ] = enercarr_jrc_agg_map[idx]

        # adjust split carriers using df_ec_fs per year
        for key in enercarr_jrc_agg_map_adj.keys():
            ec_sub_fs = df_ec_fs.loc[
                df_ec_fs["energy_carrier"].isin(enercarr_jrc_agg_map_adj[key]),
                ["energy_carrier", "year", "value"],
            ]
            ec_total_fs = (
                ec_sub_fs.groupby("year", as_index=False)["value"]
                .sum()
                .rename(columns={"value": "ec_total"})
            )
            for idx in range(0, len(enercarr_jrc_agg_map_adj[key])):
                sub_carr = enercarr_jrc_agg_map_adj[key][idx]
                ec_part_fs = ec_sub_fs.loc[
                    ec_sub_fs["energy_carrier"] == sub_carr, ["year", "value"]
                ].rename(columns={"value": "ec_part"})
                ec_ratio_fs = pd.merge(ec_part_fs, ec_total_fs, on="year")
                ec_ratio_fs["ratio"] = ec_ratio_fs["ec_part"] / ec_ratio_fs["ec_total"]
                ec_ratio_fs["ratio"] = ec_ratio_fs["ratio"].fillna(0)
                ec_ratio_fs = ec_ratio_fs[["year", "ratio"]]
                key_rows_fs = df_temp_fs_proc.loc[
                    df_temp_fs_proc["energy_carrier"] == key,
                    ["energy_carrier", "year", "value"],
                ].copy()
                key_rows_fs = pd.merge(key_rows_fs, ec_ratio_fs, on="year")
                key_rows_fs["value"] = key_rows_fs["value"] * key_rows_fs["ratio"]
                key_rows_fs["energy_carrier"] = sub_carr
                key_rows_fs = key_rows_fs[["energy_carrier", "year", "value"]]
                df_temp_fs_proc = pd.concat(
                    [df_temp_fs_proc, key_rows_fs], ignore_index=True
                )
            df_temp_fs_proc = df_temp_fs_proc.loc[
                df_temp_fs_proc["energy_carrier"] != key, :
            ]

        for idx in range(0, len(enercarr_jrc_agg)):
            df_temp_fs_proc.loc[
                df_temp_fs_proc["energy_carrier"] == enercarr_jrc_agg[idx],
                "energy_carrier",
            ] = enercarr_calc_map[idx]
        df_temp_fs_proc = df_temp_fs_proc.groupby(
            ["energy_carrier", "year"], as_index=False
        ).agg(sum)

        for key in dict_adj_biomass_waste.keys():
            bw_rows = df_temp_fs_proc.loc[
                df_temp_fs_proc["energy_carrier"] == "Biomass and waste",
                ["year", "value"],
            ].copy()
            bw_rows["energy_carrier"] = key
            bw_rows["value"] = bw_rows["value"] * dict_adj_biomass_waste[key]
            df_temp_fs_proc = pd.concat(
                [df_temp_fs_proc, bw_rows[["energy_carrier", "year", "value"]]],
                ignore_index=True,
            )
        df_temp_fs_proc = df_temp_fs_proc.loc[
            df_temp_fs_proc["energy_carrier"] != "Biomass and waste", :
        ]

        all_years_fs = df_temp_fs_proc["year"].unique().tolist()
        extras_fs = pd.DataFrame(
            {
                "energy_carrier": [
                    "electricity",
                    "gas-bio",
                    "liquid-bio",
                    "solid-waste",
                    "solid-bio",
                ]
                * len(all_years_fs),
                "year": sorted(all_years_fs * 5),
                "value": 0.0,
            }
        )
        df_temp_fs_proc = pd.concat([df_temp_fs_proc, extras_fs], ignore_index=True)
        df_temp_fs_proc = df_temp_fs_proc.groupby(
            ["energy_carrier", "year"], as_index=False
        ).agg(sum)

        # Diesel oil → liquid-ff-diesel, Naphtha → liquid-ff-oil
        df_temp_fs_proc.loc[
            df_temp_fs_proc["energy_carrier"] == "Diesel oil", "energy_carrier"
        ] = "liquid-ff-diesel"
        df_temp_fs_proc.loc[
            df_temp_fs_proc["energy_carrier"] == "Naphtha", "energy_carrier"
        ] = "liquid-ff-oil"
        df_temp_fs_proc = df_temp_fs_proc.groupby(
            ["energy_carrier", "year"], as_index=False
        ).agg(sum)

        # add hydrogen
        h2_rows_fs = pd.DataFrame(
            {"energy_carrier": "hydrogen", "year": all_years_fs, "value": 0.0}
        )
        df_temp_fs_proc = pd.concat([df_temp_fs_proc, h2_rows_fs], ignore_index=True)
        df_temp_fs_proc["tech"] = techs_calc[0]

        # normalize feedstock by production per year
        df_fs_norm = df_temp_fs_proc.copy()
        df_fs_norm["value"] = df_fs_norm["value"] * ktoe_to_twh
        df_fs_norm["unit"] = "TWh"
        df_fs_norm = pd.merge(df_fs_norm, df_po_temp, how="left", on=["tech", "year"])
        df_fs_norm["value"] = df_fs_norm["value"] / df_fs_norm["production"]
        df_fs_norm["variable"] = [
            tech + "_" + enercarr + "[" + unit + "/" + unit_production + "]"
            for tech, enercarr, unit, unit_production in zip(
                df_fs_norm["tech"],
                df_fs_norm["energy_carrier"],
                df_fs_norm["unit"],
                df_fs_norm["unit_production"],
            )
        ]
        df_fs_norm = df_fs_norm.loc[:, ["year", "variable", "value", "tech"]]

        df_temp_fs_sec = df_fs_norm.loc[
            df_fs_norm["tech"] == "chem-chem-tech", :
        ].copy()
        df_temp_fs_sec["tech"] = "chem-sec"
        df_temp_fs_sec["variable"] = [
            i.replace("chem-chem-tech", "chem-sec") for i in df_temp_fs_sec["variable"]
        ]
        df_fs_norm = pd.concat([df_fs_norm, df_temp_fs_sec])
        df_fs_norm["energy_demand_type"] = energy_demand_type
        df_final_feedstock = pd.concat([df_final_feedstock, df_fs_norm])

    del (
        df,
        df_ec,
        df_ec_fs,
        df_excl,
        df_fec_sheet,
        df_fs_norm,
        df_po,
        df_po_temp,
        df_temp,
        df_temp_coolinggas,
        df_temp_fs,
        df_temp_fs_proc,
        df_temp_fs_sec,
        df_temp_sec,
        df_ued_sheet,
        DF_temp,
        i,
        id_var,
        idx,
        index_row_end,
        index_row_start,
        key,
        ls_temp,
        techs,
        techs_calc,
        techs_code,
    )

    ####################################################################
    ############################# AMMONIA ##############################
    ####################################################################

    years_df = pd.DataFrame({"year": years})

    # excluding feedstock (broadcast same value across all JRC years)
    df_amm_base = pd.DataFrame(
        {
            "energy_carrier": [
                "lighting",
                "electricity",
                "gas-bio",
                "gas-ff-natural",
                "hydrogen",
                "liquid-bio",
                "liquid-ff-oil",
                "liquid-ff-diesel",
                "solid-bio",
                "solid-ff-coal",
                "solid-waste",
            ],
            "value": [0, 0.74 + 5 + 1.7 - 4.3, 0, 0, 0, 0, 10.9, 0, 0, 0, 0],
        }
    )
    df_amm_base["value"] = df_amm_base["value"] * gj_to_twh * 1000000
    df_amm_base["tech"] = "ammonia-tech"
    df_amm = df_amm_base.merge(years_df, how="cross")
    df_amm["variable"] = [
        tech + "_" + ec + "[TWh/Mt]"
        for tech, ec in zip(df_amm["tech"], df_amm["energy_carrier"])
    ]
    df_amm = df_amm.loc[:, ["year", "variable", "value", "tech"]]
    df_amm["energy_demand_type"] = "fec"
    df_final = pd.concat([df_final, df_amm])

    # feedstock (broadcast across all JRC years)
    energy_feedstock_gj = 21
    df_amm_fs_base = pd.DataFrame(
        {
            "energy_carrier": [
                "lighting",
                "electricity",
                "gas-bio",
                "gas-ff-natural",
                "hydrogen",
                "liquid-bio",
                "liquid-ff-oil",
                "liquid-ff-diesel",
                "solid-bio",
                "solid-ff-coal",
                "solid-waste",
            ],
            "value": [
                0,
                0,
                energy_feedstock_gj / 3,
                energy_feedstock_gj / 3,
                energy_feedstock_gj / 3,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        }
    )
    df_amm_fs_base["value"] = df_amm_fs_base["value"] * gj_to_twh * 1000000
    df_amm_fs_base["tech"] = "ammonia-tech"
    df_amm_fs = df_amm_fs_base.merge(years_df, how="cross")
    df_amm_fs["variable"] = [
        tech + "_" + ec + "[TWh/Mt]"
        for tech, ec in zip(df_amm_fs["tech"], df_amm_fs["energy_carrier"])
    ]
    df_amm_fs = df_amm_fs.loc[:, ["year", "variable", "value", "tech"]]
    df_amm_fs["energy_demand_type"] = "fec"
    df_final_feedstock = pd.concat([df_final_feedstock, df_amm_fs])

    del df_amm, df_amm_base, df_amm_fs, df_amm_fs_base, energy_feedstock_gj

    # ammonia UED: apply chem-chem-tech efficiency (averaged across years)
    df_final.sort_values(
        ["tech", "energy_demand_type", "variable", "year"], inplace=True
    )
    df_final_feedstock.sort_values(
        ["tech", "energy_demand_type", "variable", "year"], inplace=True
    )

    df_eff_chem = df_final.loc[df_final["tech"] == "chem-chem-tech", :]
    df_eff_chem_piv = df_eff_chem.pivot_table(
        index="variable", columns="energy_demand_type", values="value", aggfunc="mean"
    ).reset_index()
    df_eff_chem_piv["efficiency"] = df_eff_chem_piv["ued"] / df_eff_chem_piv["fec"]
    df_eff_chem_piv.loc[
        df_eff_chem_piv["variable"] == "chem-chem-tech_hydrogen[TWh/Mt]", "efficiency"
    ] = 0.5
    df_eff_chem_piv = df_eff_chem_piv[["variable", "efficiency"]]

    for amm_df, target_df in [
        (df_final, "df_final"),
        (df_final_feedstock, "df_final_feedstock"),
    ]:
        df_amm_fec = amm_df.loc[amm_df["tech"] == "ammonia-tech", :].copy()
        df_amm_fec["var_chem"] = [
            v.replace("ammonia-tech", "chem-chem-tech") for v in df_amm_fec["variable"]
        ]
        df_amm_fec = pd.merge(
            df_amm_fec,
            df_eff_chem_piv.rename(columns={"variable": "var_chem"}),
            on="var_chem",
            how="left",
        )
        df_amm_fec["efficiency"] = df_amm_fec["efficiency"].fillna(0.5)
        df_amm_fec["value"] = df_amm_fec["value"] * df_amm_fec["efficiency"]
        df_amm_fec["energy_demand_type"] = "ued"
        df_amm_fec = df_amm_fec.drop(columns=["var_chem", "efficiency"])
        if target_df == "df_final":
            df_final = pd.concat([df_final, df_amm_fec])
        else:
            df_final_feedstock = pd.concat([df_final_feedstock, df_amm_fec])

    ###########################################################
    ################ PULP AND PAPER PRODUCTION ################
    ###########################################################

    techs_code = "PPA"
    techs = ["Pulp production", "Paper production", "Printing and media reproduction"]
    techs_calc = ["paper-sec", "paper-tech", "printing-media-tech"]
    dict_subtechs_elec = {
        "Pulp production": electric_usual
        + [
            "Pulp: Wood preparation, grinding",
            "Pulp: Pulping electric",
            "Pulp: Cleaning",
        ],
        "Paper production": electric_usual
        + [
            "Paper: Stock preparation - Mechanical",
            "Paper: Paper machine - Electricity",
            "Paper: Product finishing - Electricity",
        ],
        "Printing and media reproduction": electric_usual + ["Printing and publishing"],
    }

    DF = get_energy_intensity(
        filepath=filepath,
        techs_code=techs_code,
        techs=techs,
        techs_calc=techs_calc,
        dict_subtechs_elec=dict_subtechs_elec,
        years=years,
    )
    df = DF["energy-intensity"].copy()
    df_po = DF["production"].copy()
    df = df.loc[df["tech"] != "printing-media-tech", :].copy()

    # Combined paper+pulp intensity: model tracks only final paper tonnes, so total energy
    # (JRC pulp step + JRC paper step) must be divided by paper production, not pulp production.
    # combined_intensity = pulp_int × (pulp_prod/paper_prod) + paper_int (per carrier, per year)
    df_pulp_prod = df_po.loc[df_po["tech"] == "paper-sec", ["year", "value"]].rename(
        columns={"value": "pulp_prod"}
    )
    df_paper_prod = df_po.loc[df_po["tech"] == "paper-tech", ["year", "value"]].rename(
        columns={"value": "paper_prod"}
    )
    df_prod_ratio = pd.merge(df_pulp_prod, df_paper_prod, on="year")
    df_prod_ratio["ratio"] = df_prod_ratio["pulp_prod"] / df_prod_ratio[
        "paper_prod"
    ].replace(0, np.nan)

    df["carrier_unit"] = df["variable"].str.replace(r"^[^_]+_", "", regex=True)
    df_pulp = df.loc[
        df["tech"] == "paper-sec",
        ["year", "carrier_unit", "value", "energy_demand_type"],
    ].rename(columns={"value": "pulp_int"})
    df_paper = df.loc[
        df["tech"] == "paper-tech",
        ["year", "carrier_unit", "value", "energy_demand_type"],
    ].rename(columns={"value": "paper_int"})
    df_merged = pd.merge(
        df_pulp, df_paper, on=["year", "carrier_unit", "energy_demand_type"]
    )
    df_merged = pd.merge(df_merged, df_prod_ratio[["year", "ratio"]], on="year")
    df_merged["combined"] = (
        df_merged["pulp_int"] * df_merged["ratio"] + df_merged["paper_int"]
    )

    rows = []
    for tech in ["paper-sec", "paper-tech"]:
        df_out = df_merged[
            ["year", "carrier_unit", "combined", "energy_demand_type"]
        ].copy()
        df_out["tech"] = tech
        df_out["variable"] = tech + "_" + df_out["carrier_unit"]
        df_out["value"] = df_out["combined"]
        rows.append(df_out[["year", "variable", "value", "tech", "energy_demand_type"]])
    df = pd.concat(rows)

    df_final = pd.concat([df_final, df])
    del (
        DF,
        df,
        df_po,
        df_pulp_prod,
        df_paper_prod,
        df_prod_ratio,
        df_pulp,
        df_paper,
        df_merged,
        rows,
        techs,
        techs_calc,
        techs_code,
    )

    ###########################################################
    ################### ALUMINIUM AND COPPER ##################
    ###########################################################

    techs_code = "NFM"
    techs = [
        "Alumina production",
        "Aluminium - primary production",
        "Aluminium - secondary production",
        "Other non-ferrous metals",
    ]
    techs_calc = ["alumina-tech", "aluminium-prim", "aluminium-sec", "other-nfm"]
    dict_subtechs_elec = {
        "Alumina production": electric_usual,
        "Aluminium - primary production": electric_usual
        + [
            "Aluminium electrolysis (smelting)",
            "Aluminium processing - Electric",
            "Aluminium finishing - Electric",
        ],
        "Aluminium - secondary production": electric_usual
        + [
            "Secondary aluminium - Electric",
            "Aluminium processing - Electric",
            "Aluminium finishing - Electric",
        ],
        "Other non-ferrous metals": electric_usual
        + [
            "Metal production - Electric",
            "Metal processing - Electric",
            "Metal finishing - Electric",
        ],
    }

    DF = get_energy_intensity(
        filepath=filepath,
        techs_code=techs_code,
        techs=techs,
        techs_calc=techs_calc,
        dict_subtechs_elec=dict_subtechs_elec,
        years=years,
    )
    df = DF["energy-intensity"].copy()
    df = df.loc[~df["tech"].isin(["alumina-tech"]), :]
    df.loc[df["tech"] == "other-nfm", "tech"] = "copper-tech"
    df["variable"] = [i.replace("other-nfm", "copper-tech") for i in df["variable"]]

    ec_perc_less = 0.85
    df_temp = df.loc[df["tech"] == "copper-tech", :].copy()
    df_temp["value"] = df_temp["value"] * (1 - ec_perc_less)
    df_temp["tech"] = "copper-sec"
    df_temp["variable"] = [
        i.replace("copper-tech", "copper-sec") for i in df_temp["variable"]
    ]
    df = pd.concat([df, df_temp])

    df_final = pd.concat([df_final, df])
    del DF, df, df_temp, techs, techs_calc, techs_code

    ###########################################################
    ########################## LIME ###########################
    ###########################################################

    energy_consumption_gj_per_tonne = 4.25
    electricity_lime = 4.25 * 0.05
    ec_minus_electricity = energy_consumption_gj_per_tonne - electricity_lime

    df_lime_base = pd.DataFrame(
        {
            "energy_carrier": [
                "lighting",
                "gas-bio",
                "gas-ff-natural",
                "hydrogen",
                "liquid-bio",
                "liquid-ff-diesel",
                "liquid-ff-oil",
                "solid-bio",
                "solid-ff-coal",
                "solid-waste",
            ],
            "value": [0, 0, 0.34, 0, 0, 0.025, 0.025, 0.02, 0.51, 0.08],
        }
    )
    df_lime_base["value"] = df_lime_base["value"] * ec_minus_electricity
    df_elec_lime = pd.DataFrame(
        {"energy_carrier": ["electricity"], "value": [electricity_lime]}
    )
    df_lime_base = pd.concat([df_lime_base, df_elec_lime], ignore_index=True)
    df_lime_base["value"] = df_lime_base["value"] * gj_to_twh * 1000000
    df_lime_base["tech"] = "lime-lime"

    df_lime = df_lime_base.merge(years_df, how="cross")
    df_lime["variable"] = [
        "lime-lime_" + v + "[TWh/Mt]" for v in df_lime["energy_carrier"]
    ]
    df_lime["energy_demand_type"] = "fec"
    df_lime.sort_values(["tech", "energy_demand_type", "variable"], inplace=True)
    df_final = pd.concat([df_final, df_lime])

    # lime UED: apply cement-dry-kiln efficiency (averaged across years)
    df_cdkiln = df_final.loc[df_final["tech"] == "cement-dry-kiln", :]
    df_eff_lime = df_cdkiln.pivot_table(
        index="variable", columns="energy_demand_type", values="value", aggfunc="mean"
    ).reset_index()
    df_eff_lime["efficiency"] = df_eff_lime["ued"] / df_eff_lime["fec"]
    df_eff_lime.loc[
        df_eff_lime["variable"] == "cement-dry-kiln_hydrogen[TWh/Mt]", "efficiency"
    ] = 0.5
    df_eff_lime = df_eff_lime[["variable", "efficiency"]]

    df_lime_ued = df_lime.copy()
    df_lime_ued["var_cdkiln"] = [
        v.replace("lime-lime", "cement-dry-kiln") for v in df_lime_ued["variable"]
    ]
    df_lime_ued = pd.merge(
        df_lime_ued,
        df_eff_lime.rename(columns={"variable": "var_cdkiln"}),
        on="var_cdkiln",
        how="left",
    )
    df_lime_ued["efficiency"] = df_lime_ued["efficiency"].fillna(0.5)
    df_lime_ued["value"] = df_lime_ued["value"] * df_lime_ued["efficiency"]
    df_lime_ued["energy_demand_type"] = "ued"
    df_lime_ued = df_lime_ued.drop(columns=["var_cdkiln", "efficiency"])
    df_final = pd.concat([df_final, df_lime_ued])

    del df_lime, df_lime_base, df_lime_ued, df_elec_lime
    del energy_consumption_gj_per_tonne, electricity_lime, ec_minus_electricity

    ###########################################################
    #### FOOD, MACHINERY, OIS, TEXTILES, TRANSPORT-EQUIP, WWP #
    ###########################################################

    sector_specs = [
        (
            "FBT",
            ["Food, beverages and tobacco"],
            ["fbt-tech"],
            {
                "Food, beverages and tobacco": electric_usual
                + [
                    "Food: Direct Heat - Electric",
                    "Food: Direct Heat - Microwave",
                    "Food: Process Heat - Electric",
                    "Food: Process Heat - Microwave",
                    "Food: Electric drying",
                    "Food: Freeze drying",
                    "Food: Microwave drying",
                    "Food: Thermal cooling",
                    "Food: Electric cooling",
                    "Food: Electric machinery",
                ]
            },
            None,
            None,
        ),
        (
            "MAE",
            ["Machinery equipment"],
            ["mae-tech"],
            {
                "Machinery equipment": electric_usual
                + [
                    "Mach. Eq.: Electric Foundries",
                    "Mach. Eq.: Thermal connection",
                    "Mach. Eq.: Electric connection",
                    "Mach. Eq.: Heat treatment - Electric",
                    "Mach. Eq.: General machinery",
                    "Mach. Eq.: Product finishing",
                ]
            },
            None,
            None,
        ),
        (
            "OIS",
            ["Other industrial sectors"],
            ["ois-tech"],
            {
                "Other industrial sectors": electric_usual
                + [
                    "Other Industrial sectors: Electric processing",
                    "Other Industries: Electric drying",
                    "Other Industries: Thermal cooling",
                    "Other Industrial sectors: Diesel motors (incl. biofuels)",
                    "Other Industrial sectors: Electric machinery",
                ]
            },
            "ois-tech",
            "ois-sec",
        ),
        (
            "TEL",
            ["Textiles and leather"],
            ["textiles-tech"],
            {
                "Textiles and leather": electric_usual
                + [
                    "Textiles: Electric general machinery",
                    "Textiles: Electric drying",
                    "Textiles: Microwave drying",
                    "Textiles: Finishing Electric",
                ]
            },
            None,
            None,
        ),
        (
            "TRE",
            ["Transport equipment"],
            ["tra-equip-tech"],
            {
                "Transport equipment": electric_usual
                + [
                    "Trans. Eq.: Electric Foundries",
                    "Trans. Eq.: Thermal connection",
                    "Trans. Eq.: Electric connection",
                    "Trans. Eq.: Heat treatment - Electric",
                    "Trans. Eq.: General machinery",
                    "Trans. Eq.: Product finishing",
                ]
            },
            None,
            None,
        ),
        (
            "WWP",
            ["Wood and wood products"],
            ["wwp-tech"],
            {
                "Wood and wood products": electric_usual
                + [
                    "Wood: Electric mechanical processes",
                    "Wood: Electric drying",
                    "Wood: Microwave drying",
                    "Wood: Finishing Electric",
                ]
            },
            None,
            None,
        ),
    ]

    for (
        techs_code,
        techs,
        techs_calc,
        dict_subtechs_elec,
        sec_src,
        sec_name,
    ) in sector_specs:
        df_po_override = None
        if techs_code in _TECHS_CODE_TO_MAT:
            df_po_override = _prod_override_df(
                _TECHS_CODE_TO_MAT[techs_code], techs_calc, years
            )
        if techs_code == "WWP":
            # Add full FAOSTAT total sawnwood to PRODCOM chips+boards to get C16 total
            # production as intensity denominator. FAOSTAT covers both construction (~50%)
            # and non-construction (~50%) sawnwood. timber-tech gets the same intensity, so:
            # model_energy = (timber_prod + wwp_prod) × intensity ≈ C16_total_energy.
            # Source: https://www.fao.org/faostat/en/#data/FO
            _fao_avg = np.mean(list(_faostat_sawnwood.values()))
            df_po_override["production"] += df_po_override["year"].map(
                lambda yr: _faostat_sawnwood.get(yr, _fao_avg)
            )
        DF = get_energy_intensity(
            filepath=filepath,
            techs_code=techs_code,
            techs=techs,
            techs_calc=techs_calc,
            dict_subtechs_elec=dict_subtechs_elec,
            years=years,
            df_po_override=df_po_override,
        )
        df = DF["energy-intensity"].copy()
        if techs_code == "OIS":
            # fold ceramics (bricks, tiles, refractories) JRC energy into OIS.
            # ceramics PRODCOM codes are in OIS; their JRC energy comes from the NMM sheet.
            # addendum[TWh/Mt] = ceramics_abs_energy[ktoe] × ktoe_to_twh / PRODCOM_OIS_production[Mt]
            # (cannot use intensity × production-ratio: JRC ceramics production is in "kt bricks-eq.",
            # a virtual unit incompatible with PRODCOM mass-based Mt)
            df_ois_prod = df_po_override[["year", "production"]].copy()  # Mt
            for df_cer_abs, etype in [
                (df_ceramics_fec_abs, "fec"),
                (df_ceramics_ued_abs, "ued"),
            ]:
                df_add = df_cer_abs.copy()
                df_add = pd.merge(df_add, df_ois_prod, on="year")
                df_add["value"] = df_add["value"] * ktoe_to_twh / df_add["production"]
                df_add["production"] = None
                df_add = df_add.drop(columns=["production"])
                df_add["tech"] = "ois-tech"
                df_add["energy_demand_type"] = etype
                df_add["variable"] = "ois-tech_" + df_add["energy_carrier"] + "[TWh/Mt]"
                df_add = df_add[
                    ["year", "variable", "value", "tech", "energy_demand_type"]
                ]
                df = pd.concat([df, df_add])
            df = df.groupby(
                ["year", "variable", "tech", "energy_demand_type"], as_index=False
            )["value"].sum()
        if sec_src is not None and sec_name is not None:
            df_temp = df.loc[df["tech"] == sec_src, :].copy()
            df_temp["tech"] = sec_name
            df_temp["variable"] = [
                i.replace(sec_src, sec_name) for i in df_temp["variable"]
            ]
            df = pd.concat([df, df_temp])
        df_final = pd.concat([df_final, df])

    # timber-tech: same C16 intensity as wwp-tech (JRC WWP sheet covers all C16 output;
    # both use the same C16-total denominator so timber + wwp energy sums to C16 energy).
    df_temp = df_final.loc[df_final["tech"] == "wwp-tech", :].copy()
    df_temp["tech"] = "timber-tech"
    df_temp["variable"] = [
        i.replace("wwp-tech", "timber-tech") for i in df_temp["variable"]
    ]
    df_final = pd.concat([df_final, df_temp])

    # timber-sec: same intensity as timber-tech (secondary timber from building demolition)
    df_temp2 = df_temp.copy()
    df_temp2["tech"] = "timber-sec"
    df_temp2["variable"] = [
        i.replace("timber-tech", "timber-sec") for i in df_temp2["variable"]
    ]
    df_final = pd.concat([df_final, df_temp2])

    df_final = df_final.sort_values(["tech", "energy_demand_type", "variable", "year"])

    # Asphalt — hot-mix asphalt (HMA) plant energy intensity.
    # Total intensity: 0.075 TWh/Mt (aggregate dryer + bitumen heating + electricity + other).
    # Source: EAPA annual statistics https://eapa.org/asphalt-industry/statistics/
    # Carrier split derived from EAPA HMA plant audit data:
    #   gas-ff-natural  0.048 TWh/Mt  (aggregate dryer, ~64% of total)
    #   https://eapa.org/asphalt-industry/statistics/
    #   liquid-ff-diesel  0.010 TWh/Mt  (mobile plant machinery, ~13%)
    #   https://eapa.org/asphalt-industry/statistics/
    #   electricity  0.009 TWh/Mt  (conveyors, controls, ~12%)
    #   https://eapa.org/asphalt-industry/statistics/
    #   liquid-ff-oil  0.006 TWh/Mt  (fuel oil burners in older EU plants, ~8%)
    #   https://eapa.org/asphalt-industry/statistics/
    #   solid-ff-coal  0.002 TWh/Mt  (coal-fired plants in some EU member states, ~3%)
    #   https://eapa.org/asphalt-industry/statistics/
    # asphalt-sec (RAP) uses the same intensity: RAP still requires aggregate heating;
    # virgin bitumen saving does not significantly reduce total thermal energy demand.
    # Source: EAPA annual statistics https://eapa.org/asphalt-industry/statistics/
    _asphalt_carriers = {
        "gas-ff-natural": 0.048,
        "liquid-ff-diesel": 0.010,
        "electricity": 0.009,
        "liquid-ff-oil": 0.006,
        "solid-ff-coal": 0.002,
    }
    _asphalt_rows = []
    for yr in years:
        for carrier, intensity in _asphalt_carriers.items():
            _asphalt_rows.append(
                {
                    "year": yr,
                    "variable": f"asphalt-tech_{carrier}[TWh/Mt]",
                    "tech": "asphalt-tech",
                    "energy_demand_type": "fec",
                    "value": intensity,
                }
            )
    df_asphalt = pd.DataFrame(_asphalt_rows)
    df_asphalt_sec = df_asphalt.copy()
    df_asphalt_sec["tech"] = "asphalt-sec"
    df_asphalt_sec["variable"] = df_asphalt_sec["variable"].str.replace(
        "asphalt-tech", "asphalt-sec", regex=False
    )
    df_final = pd.concat([df_final, df_asphalt, df_asphalt_sec], ignore_index=True)
    df_final = df_final.sort_values(["tech", "energy_demand_type", "variable", "year"])

    return df_final, df_final_feedstock
