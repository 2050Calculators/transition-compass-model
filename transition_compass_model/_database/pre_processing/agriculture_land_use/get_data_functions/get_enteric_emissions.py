import faostat
import pandas as pd


def get_enteric_1990_2021(file_dict, list_countries):
    try:
        df_enteric_1990_2021 = pd.read_csv(file_dict["enteric"])
    except OSError:
        list_elements = [
            "Enteric fermentation (Emissions CH4)",
            "Manure management (Emissions CH4)",
            "Stocks",
        ]

        list_items = ["All Animals > (List)"]
        list_sources = ["FAO TIER 1"]

        # 1990 - 2021
        code = "GLE"
        my_countries = [faostat.get_par(code, "area")[c] for c in list_countries]
        my_elements = [faostat.get_par(code, "elements")[e] for e in list_elements]
        my_items = [faostat.get_par(code, "item")[i] for i in list_items]
        my_sources = [faostat.get_par(code, "sources")[i] for i in list_sources]
        list_years = [
            "1990",
            "1991",
            "1992",
            "1993",
            "1994",
            "1995",
            "1996",
            "1997",
            "1998",
            "1999",
            "2000",
            "2001",
            "2002",
            "2003",
            "2004",
            "2005",
            "2006",
            "2007",
            "2008",
            "2009",
            "2010",
            "2011",
            "2012",
            "2013",
            "2014",
            "2015",
            "2016",
            "2017",
            "2018",
            "2019",
            "2020",
            "2021",
            "2022",
        ]
        my_years = [faostat.get_par(code, "year")[y] for y in list_years]

        my_pars = {
            "area": my_countries,
            "element": my_elements,
            "item": my_items,
            "year": my_years,
            "source": my_sources,
        }
        df_enteric_1990_2021 = faostat.get_data_df(code, pars=my_pars, strval=False)

        # Renaming item as the same animal (for meat and live/producing/slaugthered animals)
        df_enteric_1990_2021.loc[
            df_enteric_1990_2021["Item"].str.contains(
                "Cattle, dairy", case=False, na=False
            ),
            "Item",
        ] = "Dairy cows"
        df_enteric_1990_2021.loc[
            df_enteric_1990_2021["Item"].str.contains(
                "Cattle, non-dairy", case=False, na=False
            ),
            "Item",
        ] = "Cattle"
        df_enteric_1990_2021.loc[
            df_enteric_1990_2021["Item"].str.contains("Goat", case=False, na=False),
            "Item",
        ] = "Goat"
        df_enteric_1990_2021.loc[
            df_enteric_1990_2021["Item"].str.contains(
                "Chickens, broilers", case=False, na=False
            ),
            "Item",
        ] = "Chicken"
        df_enteric_1990_2021.loc[
            df_enteric_1990_2021["Item"].str.contains(
                "Chickens, layers", case=False, na=False
            ),
            "Item",
        ] = "Chicken laying hens"
        df_enteric_1990_2021.loc[
            df_enteric_1990_2021["Item"].str.contains("Duck", case=False, na=False),
            "Item",
        ] = "Duck"
        df_enteric_1990_2021.loc[
            df_enteric_1990_2021["Item"].str.contains("Horse", case=False, na=False),
            "Item",
        ] = "Horse"
        df_enteric_1990_2021.loc[
            df_enteric_1990_2021["Item"].str.contains("Sheep", case=False, na=False),
            "Item",
        ] = "Sheep"
        df_enteric_1990_2021.loc[
            df_enteric_1990_2021["Item"].str.contains("Swine", case=False, na=False),
            "Item",
        ] = "Pig"
        df_enteric_1990_2021.loc[
            df_enteric_1990_2021["Item"].str.contains("Turkey", case=False, na=False),
            "Item",
        ] = "Turkey"
        df_enteric_1990_2021.loc[
            df_enteric_1990_2021["Item"].str.contains("Asse", case=False, na=False),
            "Item",
        ] = "Asse"
        df_enteric_1990_2021.loc[
            df_enteric_1990_2021["Item"].str.contains("Buffalo", case=False, na=False),
            "Item",
        ] = "Buffalo"
        df_enteric_1990_2021.loc[
            df_enteric_1990_2021["Item"].str.contains("Mule", case=False, na=False),
            "Item",
        ] = "Mule"
        df_enteric_1990_2021.loc[
            df_enteric_1990_2021["Item"].str.contains("Camel", case=False, na=False),
            "Item",
        ] = "Other non-specified"
        df_enteric_1990_2021.to_csv("data/faostat/enteric_1990_2021.csv", index=False)
    return df_enteric_1990_2021
