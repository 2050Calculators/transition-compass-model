import faostat
import pandas as pd


def get_losses_csl(file_dict, list_countries):
    try:
        return pd.read_csv(file_dict["losses_csl"])
    except FileNotFoundError:
        print("File not found for losses data.")

        list_elements = ["Losses", "Production Quantity"]
        list_items = ["Animal Products > (List)"]

        code = "FBSH"
        my_countries = [faostat.get_par(code, "area")[c] for c in list_countries]
        my_elements = [faostat.get_par(code, "elements")[e] for e in list_elements]
        my_items = [faostat.get_par(code, "item")[i] for i in list_items]
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
        ]
        my_years = [faostat.get_par(code, "year")[y] for y in list_years]
        my_pars = {
            "area": my_countries,
            "element": my_elements,
            "item": my_items,
            "year": my_years,
        }
        df_losses_csl_1990_2013 = faostat.get_data_df(code, pars=my_pars, strval=False)

        df_losses_csl_1990_2013.loc[
            df_losses_csl_1990_2013["Element"].str.contains(
                "Production Quantity", case=False, na=False
            ),
            "Element",
        ] = "Production"

        list_elements = ["Losses", "Production Quantity"]
        code = "FBS"
        my_countries = [faostat.get_par(code, "area")[c] for c in list_countries]
        my_elements = [faostat.get_par(code, "elements")[e] for e in list_elements]
        my_items = [faostat.get_par(code, "item")[i] for i in list_items]
        list_years = [
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
        ]
        my_years = [faostat.get_par(code, "year")[y] for y in list_years]
        my_pars = {
            "area": my_countries,
            "element": my_elements,
            "item": my_items,
            "year": my_years,
        }
        df_losses_csl_2010_2021 = faostat.get_data_df(code, pars=my_pars, strval=False)
        df_losses_csl_2010_2021.loc[
            df_losses_csl_2010_2021["Element"].str.contains(
                "Production Quantity", case=False, na=False
            ),
            "Element",
        ] = "Production"

        df_losses_csl = pd.concat([df_losses_csl_1990_2013, df_losses_csl_2010_2021])
        df_losses_csl.to_csv(file_dict["losses_csl"], index=False)
        return df_losses_csl


def get_producing_animals_1990_2022(file_dict, list_countries):
    try:
        return pd.read_csv(file_dict["producing_animals"])
    except OSError:
        print("File not found for producing animals data.")
        list_elements = ["Producing Animals/Slaughtered", "Production Quantity"]
        list_items = ["Milk, Total > (List)", "Eggs Primary > (List)"]

        code = "QCL"
        my_countries = [faostat.get_par(code, "area")[c] for c in list_countries]
        my_elements = [faostat.get_par(code, "elements")[e] for e in list_elements]
        my_items = [faostat.get_par(code, "item")[i] for i in list_items]
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
            "2023",
        ]
        my_years = [faostat.get_par(code, "year")[y] for y in list_years]
        my_pars = {
            "area": my_countries,
            "element": my_elements,
            "item": my_items,
            "year": my_years,
        }
        df_producing_animals_1990_2022 = faostat.get_data_df(
            code, pars=my_pars, strval=False
        )

        df_producing_animals_1990_2022 = df_producing_animals_1990_2022[
            df_producing_animals_1990_2022["Unit"] != "1000 No"
        ]

        df_producing_animals_1990_2022.loc[
            df_producing_animals_1990_2022["Item"].str.contains(
                "Cattle", case=False, na=False
            ),
            "Item",
        ] = "Dairy cows"
        df_producing_animals_1990_2022.loc[
            df_producing_animals_1990_2022["Item"].str.contains(
                "Sheep", case=False, na=False
            ),
            "Item",
        ] = "Dairy sheep"
        df_producing_animals_1990_2022.loc[
            df_producing_animals_1990_2022["Item"].str.contains(
                "Goat", case=False, na=False
            ),
            "Item",
        ] = "Dairy goat"
        df_producing_animals_1990_2022.loc[
            df_producing_animals_1990_2022["Item"].str.contains(
                "Buffalo", case=False, na=False
            ),
            "Item",
        ] = "Dairy buffalo"
        df_producing_animals_1990_2022.loc[
            df_producing_animals_1990_2022["Item"].str.contains(
                "Hen eggs", case=False, na=False
            ),
            "Item",
        ] = "Chicken laying hens"
        df_producing_animals_1990_2022.loc[
            df_producing_animals_1990_2022["Item"].str.contains(
                "Eggs from other birds", case=False, na=False
            ),
            "Item",
        ] = "Other laying hens"

        df_producing_animals_1990_2022["Value"] = pd.to_numeric(
            df_producing_animals_1990_2022["Value"], errors="coerce"
        )
        mask = df_producing_animals_1990_2022["Unit"].str.strip() == "1000 An"
        df_producing_animals_1990_2022.loc[mask, "Value"] *= 1000
        df_producing_animals_1990_2022.loc[mask, "Unit"] = "An"

        df_producing_animals_1990_2022.to_csv(
            file_dict["producing_animals"], index=False
        )
        return df_producing_animals_1990_2022


def get_slaughtered_1990_2022(file_dict, list_countries):
    try:
        return pd.read_csv(file_dict["slaughtered"])
    except OSError:
        print("File not found for slaughtered animals data.")
        list_elements = ["Producing Animals/Slaughtered", "Production Quantity"]
        list_items = ["Meat, Total > (List)"]

        code = "QCL"
        my_countries = [faostat.get_par(code, "area")[c] for c in list_countries]
        my_elements = [faostat.get_par(code, "elements")[e] for e in list_elements]
        my_items = [faostat.get_par(code, "item")[i] for i in list_items]
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
            "2023",
        ]
        my_years = [faostat.get_par(code, "year")[y] for y in list_years]
        my_pars = {
            "area": my_countries,
            "element": my_elements,
            "item": my_items,
            "year": my_years,
        }
        df_slaughtered_1990_2022 = faostat.get_data_df(code, pars=my_pars, strval=False)

        df_slaughtered_1990_2022 = df_slaughtered_1990_2022[
            df_slaughtered_1990_2022["Item"] != "Bees"
        ]

        df_slaughtered_1990_2022.loc[
            df_slaughtered_1990_2022["Item"].str.contains("Pig", case=False, na=False),
            "Item",
        ] = "Pig"
        df_slaughtered_1990_2022.loc[
            df_slaughtered_1990_2022["Item"].str.contains(
                "Cattle", case=False, na=False
            ),
            "Item",
        ] = "Cattle"
        df_slaughtered_1990_2022.loc[
            df_slaughtered_1990_2022["Item"].str.contains(
                "Buffalo", case=False, na=False
            ),
            "Item",
        ] = "Cattle"
        df_slaughtered_1990_2022.loc[
            df_slaughtered_1990_2022["Item"].str.contains(
                "Chicken", case=False, na=False
            ),
            "Item",
        ] = "Chicken"
        df_slaughtered_1990_2022.loc[
            df_slaughtered_1990_2022["Item"].str.contains("Duck", case=False, na=False),
            "Item",
        ] = "Duck"
        df_slaughtered_1990_2022.loc[
            df_slaughtered_1990_2022["Item"].str.contains(
                "Turkeys", case=False, na=False
            ),
            "Item",
        ] = "Turkey"
        df_slaughtered_1990_2022.loc[
            df_slaughtered_1990_2022["Item"].str.contains(
                "Geese", case=False, na=False
            ),
            "Item",
        ] = "Goose"
        df_slaughtered_1990_2022.loc[
            df_slaughtered_1990_2022["Item"].str.contains(
                "Pigeon", case=False, na=False
            ),
            "Item",
        ] = "Pigeon"
        df_slaughtered_1990_2022.loc[
            df_slaughtered_1990_2022["Item"].str.contains(
                "Horse", case=False, na=False
            ),
            "Item",
        ] = "Horse"
        df_slaughtered_1990_2022.loc[
            df_slaughtered_1990_2022["Item"].str.contains(
                "Rabbits and hares", case=False, na=False
            ),
            "Item",
        ] = "Rabbit"
        df_slaughtered_1990_2022.loc[
            df_slaughtered_1990_2022["Item"].str.contains(
                "Sheep", case=False, na=False
            ),
            "Item",
        ] = "Sheep"
        df_slaughtered_1990_2022.loc[
            df_slaughtered_1990_2022["Item"].str.contains("Goat", case=False, na=False),
            "Item",
        ] = "Goat"
        df_slaughtered_1990_2022.loc[
            df_slaughtered_1990_2022["Item"].str.contains("Asse", case=False, na=False),
            "Item",
        ] = "Asse"
        df_slaughtered_1990_2022.loc[
            df_slaughtered_1990_2022["Item"].str.contains(
                "Camel", case=False, na=False
            ),
            "Item",
        ] = "Other non-specified"
        df_slaughtered_1990_2022.loc[
            df_slaughtered_1990_2022["Item"].str.contains(
                "Rodent", case=False, na=False
            ),
            "Item",
        ] = "Other non-specified"
        df_slaughtered_1990_2022.loc[
            df_slaughtered_1990_2022["Item"].str.contains(
                "Other", case=False, na=False
            ),
            "Item",
        ] = "Other non-specified"
        df_slaughtered_1990_2022.loc[
            df_slaughtered_1990_2022["Item"].str.contains("Game", case=False, na=False),
            "Item",
        ] = "Game"
        df_slaughtered_1990_2022.loc[
            df_slaughtered_1990_2022["Item"].str.contains("Mule", case=False, na=False),
            "Item",
        ] = "Mule"

        df_slaughtered_1990_2022["Value"] = pd.to_numeric(
            df_slaughtered_1990_2022["Value"], errors="coerce"
        )
        mask = df_slaughtered_1990_2022["Unit"].str.strip() == "1000 An"
        df_slaughtered_1990_2022.loc[mask, "Value"] *= 1000
        df_slaughtered_1990_2022.loc[mask, "Unit"] = "An"
        df_slaughtered_1990_2022.to_csv(file_dict["slaughtered"], index=False)
        return df_slaughtered_1990_2022
