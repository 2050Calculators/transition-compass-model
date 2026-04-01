import os
import pandas as pd
import re
from transition_compass_model.model.common.data_matrix_class import DataMatrix


def get_data_waste_vehicles(current_file_directory):

    # source on website: https://www.bafu.admin.ch/bafu/en/home/topics/waste/guide-to-waste-a-z/end-of-life-vehicles.html
    # "Approximately 200,000 vehicles are withdrawn from circulation every year in Switzerland.
    # A large proportion of them are completely roadworthy or only have minor damage or faults.
    # These vehicles are classified as second-hand articles. They are often exported and sold on used car markets abroad."
    # so exports should be quite large (including the second hand), and within collected
    # re-use could be set of being low (as most of the re-use are exported). Though in our case re-use also include
    # re-use of parts, so possibly a value that is similar to EU27 might be good.

    # Other source: https://files.designer.hoststar.ch/68/86/68867f3a-1338-425f-a1e3-ec7dd88eb254.pdf

    # The number of vehicles processed in Swiss shredder plants continues to decline. Only a little over
    # 37,000 end-of-life vehicles were recycled in the year
    # under review – ten years ago the figure was as high
    # as 100,000. The fall can be attributed both to the
    # increasing age of the vehicles, as they are recycled
    # later, and to the higher share of exports compared
    # with before.

    # While the number of recycled vehicles is falling, the
    # declared quantity of automobile shredder residue
    # (ASR) is actually rising and now stands at 41,000
    # tonnes, 22.4 per cent of which comes from end-oflife vehicles. ASR is processed by a thermal technique
    # in waste incineration plants only, with 91 per cent
    # of incinerations taking place in Swiss plants. Overall,
    # ASR only accounts for one per cent of total incinerated waste. The recycling process does not end with
    # incineration, however: further valuable metals are
    # recovered from the filter dust and slag. At the same
    # time, the waste heat from the flue gases is used to
    # generate electricity and district heating.

    # TAKEN OFFROAD: all vehicles that exit stock
    # VEHICLES CANCELLED IN SWITZERLAND: all vehicles that become waste
    # VEHICLES SHREDDED IN SWITZERLAND: all vehicles that are shredded
    # difference between 2 above: An unknown number of deregistered vehicles are parked in garages, second-hand dealers and scrap yards

    # layer 1: TAKEN OFFROAD should be total, then VEHICLES CANCELLED should be collected, then uncollected like EU27,
    # then rest is export (littered zero)
    # layer 2: SHREDDED IN SWITZERLAND is all incineration (if we assume that a car weights 1.2 tonnes, then shredded 2024 is
    # 44.4 tonnes, and data on incineration in 2024 is 41 tonnes).
    # For recycling there is something but it must be little, so remaining of difference between cancelled and shredded can be assigned
    # to either second hand or landfil (with numbers similar to EU, or can use Déchets spéciaux suisses)

    # make dm
    filepath = os.path.join(
        current_file_directory, "../data/waste/vehicles/vehicle_statistics.xlsx"
    )
    df = pd.read_excel(filepath)
    df = df.loc[
        :,
        [
            "Year",
            "Taken Offroad",
            "Vehicles Cancelled in Switzerland",
            "Vehicles Shredded in Switzerland",
            "Difference Cancelled vs Shredded",
        ],
    ]
    df.columns = [
        "Years",
        "waste-tot[num]",
        "waste-collected[num]",
        "energy-recovery[num]",
        "layer2-else[num]",
    ]
    df["Country"] = "Switzerland"
    dm = DataMatrix.create_from_df(df, 0)

    return dm


def subset_with_key_word(df, word):
    myindex = [bool(re.search(word, s, re.IGNORECASE)) for s in df["Sorte de déchet"]]
    return df.loc[myindex, :]


def get_data_special_waste(current_file_directory):

    # get data dechet speciaux
    filepath = os.path.join(
        current_file_directory,
        "../data/waste/2_Statistique des déchets spéciaux/ALL.xlsx",
    )
    df = pd.read_excel(filepath)
    df = subset_with_key_word(df, "batteries")
    df = df.loc[df["type"] == "traités sur le territoire national", :]
    df = df.loc[:, ["year", "Total"]]
    df = pd.concat([df, pd.DataFrame({"year": [2013, 2024], "Total": [5, 0]})])
    df.sort_values(["year"], inplace=True)

    return df


def extract_waste_data_from_dechets_speciaux(current_file_directory, word, variable):

    filepath = os.path.join(
        current_file_directory,
        "../data/waste/2_Statistique des déchets spéciaux/ALL.xlsx",
    )
    df = pd.read_excel(filepath)
    df = subset_with_key_word(df, word)
    df = df.loc[
        df["type"].isin(["traités sur le territoire national", "exportation"]), :
    ]
    df = df.dropna()
    df.columns = [
        "Years",
        "variable",
        "Sorte de déchet",
        "total",
        "landfill",
        "energy-recovery",
        "treated-chem-bio",
        "recycling",
    ]
    df = df.loc[:, df.columns != "Sorte de déchet"]
    df.loc[df["variable"] == "traités sur le territoire national", "variable"] = (
        "domestic"
    )
    df.loc[df["variable"] == "exportation", "variable"] = "export"
    df = pd.melt(df, id_vars=["Years", "variable"], var_name="type")
    df["variable"] = df["variable"] + "_" + df["type"] + "[t]"
    df = df.loc[:, ["Years", "variable", "value"]]
    df = df.pivot(index=["Years"], columns="variable", values="value").reset_index()
    df["Country"] = "Switzerland"
    dm = DataMatrix.create_from_df(df, 1)
    dm_temp = dm.filter({"Variables": ["export"], "Categories1": ["total"]})
    dm_temp.rename_col("export", variable, "Variables")
    dm_temp.rename_col("total", "export", "Categories1")
    dm.drop("Variables", "export")
    dm.rename_col("domestic", variable, "Variables")
    dm.append(dm_temp, "Categories1")
    dm.groupby(
        {"landfill": ["landfill", "treated-chem-bio"]}, "Categories1", inplace=True
    )

    return dm
