import numpy as np
import pandas as pd


def get_material_decomposition_data():
    # get data
    filepath = "../data/Literature/literature_review_material_composition.xlsx"
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
        df_temp = df_temp.groupby(["material", "variable"], as_index=False)[
            "value"
        ].agg(sum)

        # get df with materials of current model and change their names
        df_temp1 = df_temp.loc[df_temp["material"].isin(material_current), :]
        for i in range(0, len(material_current)):
            df_temp1.loc[df_temp1["material"] == material_current[i], "material"] = (
                material_current_correct_name[i]
            )
        df_temp1 = df_temp1.groupby(["material", "variable"], as_index=False)[
            "value"
        ].agg(sum)

        # get other materials, sum them and concat with others
        df_temp2 = df_temp.loc[~df_temp["material"].isin(material_current), :]
        df_temp2 = df_temp2.groupby(["variable"], as_index=False)["value"].agg(sum)
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

    # check
    df_check = df_agg.groupby(["variable"], as_index=False)["value"].agg(sum)

    # fix cement
    # cement is just a fraction of concrete-and-inert, I will apply an adjustment factor of 0.12 and put rest to other
    df_temp = df_agg.copy()
    df_agg.loc[df_agg["material"] == "cement", "value"] = (
        df_temp.loc[df_temp["material"] == "cement", "value"] * 0.12
    )
    df_agg.loc[df_agg["material"] == "other", "value"] = (
        df_temp.loc[df_temp["material"] == "other", "value"].values
        + (df_temp.loc[df_temp["material"] == "cement", "value"] * (1 - 0.12)).values
    )

    # fix cement for domapp: domestic appliances do not contain cement, move to other
    domapp_vars = [
        "larger-appliances_fridge[kg/unit]",
        "larger-appliances_dishwasher[kg/unit]",
        "larger-appliances_washing-machine[kg/num]",
        "larger-appliances_freezer[kg/num]",
        "larger-appliances_dryer[kg/unit]",
    ]
    for v in domapp_vars:
        cement_val = df_agg.loc[
            (df_agg["variable"] == v) & (df_agg["material"] == "cement"), "value"
        ]
        if not cement_val.empty:
            df_agg.loc[
                (df_agg["variable"] == v) & (df_agg["material"] == "other"), "value"
            ] += cement_val.values[0]
            df_agg.loc[
                (df_agg["variable"] == v) & (df_agg["material"] == "cement"), "value"
            ] = 0

    # fix lime
    # move some mass from other to lime
    df_agg.loc[df_agg["variable"] == "floor-area-new-residential[kg/m2]", :]
    df_agg.loc[
        (df_agg["variable"] == "floor-area-new-residential[kg/m2]")
        & (df_agg["material"] == "lime"),
        "value",
    ] = 10
    df_agg.loc[
        (df_agg["variable"] == "floor-area-new-residential[kg/m2]")
        & (df_agg["material"] == "other"),
        "value",
    ] = (
        df_agg.loc[
            (df_agg["variable"] == "floor-area-new-residential[kg/m2]")
            & (df_agg["material"] == "other"),
            "value",
        ]
        - 10
    )
    df_agg.loc[df_agg["variable"] == "floor-area-new-non-residential[kg/m2]", :]
    df_agg.loc[
        (df_agg["variable"] == "floor-area-new-non-residential[kg/m2]")
        & (df_agg["material"] == "lime"),
        "value",
    ] = 10
    df_agg.loc[
        (df_agg["variable"] == "floor-area-new-non-residential[kg/m2]")
        & (df_agg["material"] == "other"),
        "value",
    ] = (
        df_agg.loc[
            (df_agg["variable"] == "floor-area-new-non-residential[kg/m2]")
            & (df_agg["material"] == "other"),
            "value",
        ]
        - 10
    )

    # map to products we have in the calc (by taking the mean across products)
    dict_map = {
        "LDV_ICE-gasoline[kg/num]": ["LDV_ICE-gasoline[kg/unit]"],
        "LDV_ICE-diesel[kg/num]": ["LDV_ICE-diesel[kg/unit]"],
        "LDV_PHEV-gasoline[kg/num]": ["LDV_HEV[kg/unit]"],
        "LDV_BEV[kg/num]": ["LDV_BEV[kg/unit]"],
        "LDV_FCEV[kg/num]": ["LDV_FCEV[kg/unit]"],
        "HDV_ICE-diesel[kg/num]": [
            "HDVH_ICE-Class-8-day-cab-truck[kg/unit]",
            "HDVH_ICE-Class-8-sleeper-cab-truck[kg/unit]",
            "HDVM_ICE-Class-6-PnD-truck[kg/unit]",
        ],
        "HDV_PHEV-diesel[kg/num]": [
            "HDVH_HEV-Class-8-day-cab-truck[kg/unit]",
            "HDVH_HEV-Class-8-sleeper-cab-truck[kg/unit]",
            "HDVM_HEV-Class-6-PnD-truck[kg/unit]",
        ],
        "HDV_BEV[kg/num]": [
            "HDVH_BEV-Class-8-day-cab-truck[kg/unit]",
            "HDVH_BEV-Class-8-sleeper-cab-truck[kg/unit]",
            "HDVM_EV-Class-6-PnD-truck[kg/unit]",
        ],
        "HDV_FCEV[kg/num]": [
            "HDVH_FCEV-Class-8-day-cab-truck[kg/unit]",
            "HDVH_FCV-Class-8-sleeper-cab-truck[kg/unit]",
            "HDVM_FCV-Class-6-PnD-truck[kg/unit]",
        ],
        "computer[kg/num]": ["electronics_PC[kg/unit]"],
        "dryer[kg/num]": ["larger-appliances_dryer[kg/unit]"],
        "tv[kg/num]": ["electronics_TV[kg/unit]"],
        "phone[kg/num]": ["electronics_phones[kg/unit]"],
        "fridge[kg/num]": ["larger-appliances_fridge[kg/unit]"],
        "dishwasher[kg/num]": ["larger-appliances_dishwasher[kg/unit]"],
        "wmachine[kg/num]": ["larger-appliances_washing-machine[kg/num]"],
        "freezer[kg/num]": ["larger-appliances_freezer[kg/num]"],
        "floor-area-new-residential[kg/m2]": ["floor-area-new-residential[kg/m2]"],
        "floor-area-new-non-residential[kg/m2]": [
            "floor-area-new-non-residential[kg/m2]"
        ],
        "floor-area-reno-residential[t/m2]": ["floor-area-reno-residential[t/m2]"],
        "floor-area-reno-non-residential[t/m2]": [],
        "new-dhg-pipe[t/km]": ["District heating pipes [t/km]"],
        "ships_ICE[t/num]": ["Ships [t/num]"],
        # "Trains [t/num]" in the literature review is a full freight consist (63.2 wagons).
        # industry_const_material_decomposition.py overrides this with a weighted average
        # of passenger and freight EU-configs based on CH OTS new-vehicle counts.
        "trains_CEV[t/num]": ["Trains [t/num]"],
        "planes_ICE[t/num]": ["Planes [t/num]"],
        "road[t/km]": ["Road [t/km]"],
        "rail[t/km]": ["Rail [t/km]"],
        "trolley-cables[t/km]": ["Trolley-cables [t/km]"],
        "fertilizer[t/t]": ["Fertilizer [t/t]"],
        "plastic-pack[t/t]": ["Plastic packaging [t/t]"],
        "paper-pack[t/t]": ["Paper packaging [t/t]"],
        "aluminium-pack[t/t]": ["Aluminium packaging [t/t]"],
        "glass-pack[t/t]": ["Glass packaging [t/t]"],
        "paper-print[t/t]": ["Paper printing and graphic [t/t]"],
        "paper-san[t/t]": ["Paper sanitary and household [t/t]"],
        "battery-lion-HDV_BEV[kg/num]": ["Battery Li-Ion-HDVL_EV[kg/unit]"],
        "battery-lion-HDV_PHEV[kg/num]": ["Battery Li-Ion-HDVL_PHEV[kg/unit]"],
        "battery-lion-LDV_BEV[kg/num]": ["Battery Li-Ion-LDV_EV[kg/unit]"],
        "battery-lion-LDV_PHEV[kg/num]": ["Battery Li-Ion-LDV_PHEV[kg/unit]"],
    }

    for key in dict_map.keys():
        df_agg.loc[df_agg["variable"].isin(dict_map[key]), "variable"] = key
    df_agg = df_agg.groupby(["variable", "material"], as_index=False)["value"].agg(
        np.mean
    )

    # # check
    # df_check = df_agg.groupby(["variable"], as_index=False)['value'].agg(sum)

    # fix units
    df_agg.loc[df_agg["variable"] == "floor-area-new-residential[kg/m2]", "value"] = (
        df_agg.loc[df_agg["variable"] == "floor-area-new-residential[kg/m2]", "value"]
        / 1000
    )
    df_agg.loc[
        df_agg["variable"] == "floor-area-new-non-residential[kg/m2]", "value"
    ] = (
        df_agg.loc[
            df_agg["variable"] == "floor-area-new-non-residential[kg/m2]", "value"
        ]
        / 1000
    )
    df_agg.loc[
        df_agg["variable"] == "floor-area-new-residential[kg/m2]", "variable"
    ] = "floor-area-new-residential[t/m2]"
    df_agg.loc[
        df_agg["variable"] == "floor-area-new-non-residential[kg/m2]", "variable"
    ] = "floor-area-new-non-residential[t/m2]"
    import re

    variabs = df_agg["variable"].unique()
    ls_temp = list(np.array(variabs)[[bool(re.search("kg", i)) for i in variabs]])
    ls_temp1 = [i.replace("kg", "t") for i in ls_temp]
    for i in range(0, len(ls_temp)):
        df_agg.loc[df_agg["variable"] == ls_temp[i], "value"] = (
            df_agg.loc[df_agg["variable"] == ls_temp[i], "value"] / 1000
        )
        df_agg.loc[df_agg["variable"] == ls_temp[i], "variable"] = ls_temp1[i]
    # df_agg["variable"].unique()

    # # check
    # df_check = df_agg.groupby(["variable"], as_index=False)['value'].agg(sum)

    # # fix units
    # import re
    # variables = np.array(df["variable"].unique())
    # variables = variables[[bool(re.search("kg/",v)) for v in variables]]
    # for v in variables:
    #     df_agg.loc[df["variable"] == v,"value"] = df_agg.loc[df["variable"] == v,"value"]/1000
    #     df_agg.loc[df["variable"] == v,"variable"] = v.replace("kg","t")

    return df_agg
