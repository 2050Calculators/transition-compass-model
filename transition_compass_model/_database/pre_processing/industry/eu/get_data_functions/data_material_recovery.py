import os

import numpy as np
import pandas as pd


def get_material_recovery_data():
    # get data
    _here = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(
        _here, "../data/Literature/literature_review_material_recovery.xlsx"
    )
    df = pd.read_excel(filepath)

    # name first 2 columns
    df.rename(
        columns={"Unnamed: 0": "material", "Unnamed: 1": "material-sub"}, inplace=True
    )

    # melt
    indexes = ["material", "material-sub"]
    df = pd.melt(df, id_vars=indexes, var_name="variable")

    # save materials lists
    material_sub_alu = ["cast-aluminium", "wrought-aluminium"]
    material_sub_steel = [
        "cast-iron",
        "iron",
        "steel",
        "galvanized-steel",
        "stainless-steel",
    ]
    material_sub_pla = [
        "plastics-ABS",
        "plastics-PP",
        "plastics-PA",
        "plastics-PBT",
        "plastics-PE",
        "plastics-PMMA",
        "plastics-POM",
        "plastics-EPMD",
        "plastics-EPS",
        "plastics-PS",
        "plastics-PU",
        "plastics-PUR",
        "plastics-PET",
        "plastics-PVC",
        "plastics-carbon-fiber-reinforced",
        "plastics-glass-fiber-reinforced",
        "plastics-mixture",
        "plastics-other",
    ]
    material_current = [
        "aluminium",
        "ammonia",
        "concrete-and-inert",
        "plastics-total",
        "copper",
        "glass",
        "lime",
        "paper",
        "iron_&_steel",
        "wood",
        "HDPE",
        "latex",
        "paint",
        "resin",
        "rubber",
        "fibreglass-composites",
        "fluids-and-lubricants",
        "refrigerant-R-134a",
        "high-impact polystyrene",
        "polychlorinated biphenyl",
    ]
    material_current_correct_name = [
        "aluminium",
        "ammonia",
        "cement",
        "chem",
        "copper",
        "glass",
        "lime",
        "paper",
        "steel",
        "timber",
        "chem",
        "chem",
        "chem",
        "chem",
        "chem",
        "chem",
        "chem",
        "chem",
        "chem",
        "chem",
    ]

    def aggregate_materials(
        df, variable, material_current, material_current_correct_name
    ):
        # get df for one variable
        df_temp = df.loc[df["variable"] == variable, :]

        # drop na in value
        df_temp = df_temp.dropna(subset=["value"])

        # rename missing material with sub material and drop sub material
        df_temp.loc[df_temp["material"].isnull(), "material"] = df_temp.loc[
            df_temp["material"].isnull(), "material-sub"
        ]
        df_temp = df_temp.loc[:, ["material", "variable", "value"]]

        # aggregate sub materials if any
        df_temp.loc[df_temp["material"].isin(material_sub_pla), "material"] = (
            "plastics-total"
        )
        df_temp.loc[df_temp["material"].isin(material_sub_alu), "material"] = (
            "aluminium"
        )
        df_temp.loc[df_temp["material"].isin(material_sub_steel), "material"] = (
            "iron_&_steel"
        )
        df_temp.loc[df_temp["value"] == 0, "value"] = np.nan
        df_temp = df_temp.groupby(["material", "variable"], as_index=False)[
            "value"
        ].agg(np.mean)

        # get df with materials of current model and change their names
        df_temp1 = df_temp.loc[df_temp["material"].isin(material_current), :]
        for i in range(0, len(material_current)):
            df_temp1.loc[df_temp1["material"] == material_current[i], "material"] = (
                material_current_correct_name[i]
            )
        df_temp1 = df_temp1.groupby(["material", "variable"], as_index=False)[
            "value"
        ].agg(np.mean)

        # get other materials, sum them and concat with others
        df_temp2 = df_temp.loc[~df_temp["material"].isin(material_current), :]
        df_temp2 = df_temp2.groupby(["variable"], as_index=False)["value"].agg(np.mean)
        df_temp2["material"] = "other"
        df_temp = pd.concat([df_temp1, df_temp2])

        # return
        return df_temp

    variabs = df["variable"].unique()
    DF = {}
    for v in variabs:
        DF[v] = aggregate_materials(
            df,
            v,
            material_current=material_current,
            material_current_correct_name=material_current_correct_name,
        )
    df_agg = pd.concat(DF.values(), ignore_index=True)

    # if na put zero
    df_agg.loc[df_agg["value"].isnull(), "value"] = 0

    # check
    df_check = df_agg.groupby(["variable"], as_index=False)["value"].agg(np.mean)

    # substitue nan with zero
    df_agg.loc[df_agg["value"].isnull(), "value"] = 0

    # Assumptions

    # trucks and buses, I will assume that they are the same of vehicles
    # # Source: https://horizoneuropencpportal.eu/sites/default/files/2023-09/acea-position-paper-end-of-life-vehicles-directive-trucks-buses-2020.pdf
    # Page 3:
    # Industry believes that the re-use and recycling of second raw materials is important as well. In fact,
    # this is already part of the business models of many vehicle manufacturers today. Throughout the 19
    # years that HDVs have been outside the scope of the ELV Directive, the vehicle recycling industry
    # has handled, treated and de-polluted trucks and buses in a way similar to passenger cars and thus
    # basically already applies existing environmental legislation to HDVs.

    # trains and mt: Table 2 https://www.sciencedirect.com/science/article/pii/S0956053X16305396?casa_token=URoJ4M0WLRAAAAAA:0TjVLbKhEiDy3Il7b9CmbTjEDelPNlZpF5SBRZWb_mvNeayULwxjW3BW_wAHfKQR-_8tioFj6_HQ#b0255

    # planes: https://www.easa.europa.eu/en/document-library/research-reports/study-assessment-environmental-sustainability-status-aviation
    # High recovery rates are reported by manufacturers for the case of aircraft that have been retired in the last decade.
    # For aircraft reaching their EoL during the last decade (with a high share of metals in their structure), recovery rates of
    # around 95% can be achieved by recyclers. These high recovery rates involve the use of downcycling and dealing with aircraft
    # that have a high percentage of metallic parts. Thus, for the case of recent aircraft models containing a significant
    # percentage of composite structural parts, which will enter the EoL phase in the future, new recycling technologies will
    # be required in order to keep the high reusability and recyclability rates.
    # put same recovery rates than trains (the recovery rates will be similar by material, i.e. aluminium high, composites low, etc).

    df["variable"].unique()

    # map to products we have in the calc (by taking the mean across products)
    dict_map = {
        "vehicles": [
            "ELV_shredding-and-dismantling_recycling-best",
            "ELV_shredding-and-dismantling_recovery-network-lowest",
            "ELV_shredding-and-dismantling_recovery-network-highest",
            "ELV_dismantling-mechanical-separation-and-recycling_recycling-best",
            "ELV_dismantling-mechanical-separation-and-recycling_recovery-network-lowest",
            "ELV_dismantling-mechanical-separation-and-recycling_recovery-network-highest",
            "ELV_dismantling-separation-and-dedicated-recycling_processes-recycling-best",
            "ELV_dismantling-separation-and-dedicated-recycling_processes-recovery-network-lowest",
            "ELV_dismantling-separation-and-dedicated-recycling_processes-recovery-network-highest",
        ],
        "battery-lion": [
            "LIB_pyrometallurgy-smelting_lowest",
            "LIB_pyrometallurgy-smelting_highest",
            "LIB_pyrometallurgy_carbothermal-reduction-roasting_lowest",
            "LIB_pyrometallurgy_carbothermal-reduction-roasting_highest",
            "LIB_hydrometallurgy_leaching-organic_recovery-network-lowest",
            "LIB_hydrometallurgy_leaching-organic_recovery-network-highest",
            "LIB_hydrometallurgy_leaching-inorganic_recycling-best",
            "LIB_hydrometallurgy_bio-leaching",
            "LIB_hydrometallurgy_deep-eutectic-solvents",
        ],
        "computer": ["PC_recycling"],
        "fridge": ["fridge_total-recovery"],
        "dishwasher": [
            "dishwasher_combined-treatment"
        ],  # I take the combined treatment for now, as otherwise we would need to see how to combine the 4 we have (sometimes take average, sometimes take max, etc)
        "electronics": ["WEEE"],
        # "mt" : ["metrotram_light-dismantling", "	metrotram_deep-dismantling"], for the moment we do not have metro as product in industry, so in transport we have aggregated mt to trains, so we will simply use the mat decomp of trains for now
        "train": [
            "train_ICE-Diesel_light-dismantling",
            "train_ICE-Diesel_deep-dismantling",
            "train_CEV_light-dismantling",
            "train_CEV_deep-dismantling",
        ],
        "plastic-pack": ["plastic-packaging"],
        "glass-pack": ["glass-packaging"],
        "paper-pack": ["paper-packaging"],
        "aluminium-pack": ["aluminium-pack_low", "aluminium-pack_high"],
        "floor-area": ["floor-area-new-residential"],
    }

    for key in dict_map.keys():
        df_agg.loc[df_agg["variable"].isin(dict_map[key]), "variable"] = key
    df_agg.loc[df_agg["value"] == 0, "value"] = np.nan
    df_agg = df_agg.groupby(["variable", "material"], as_index=False)["value"].agg(
        np.mean
    )
    df_agg = df_agg.loc[df_agg["variable"].isin(list(dict_map.keys())), :]

    # fix units
    df_agg["value"] = df_agg["value"] / 100

    # # check
    # df_check = df_agg.groupby(["variable"], as_index=False)["value"].agg(np.mean)

    return df_agg
