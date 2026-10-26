import faostat
import pandas as pd


def get_diet_faostat(file, list_countries):
    try:
        df_diet = pd.read_csv(file)
    except OSError:
        # FOOD BALANCE SHEETS (FBS) - -------------------------------------------------
        # List of elements
        list_elements = ["Food"]

        list_items = [
            "Cereals - Excluding Beer + (Total)",
            "Fruits - Excluding Wine + (Total)",
            "Oilcrops + (Total)",
            "Pulses + (Total)",
            "Rice (Milled Equivalent)",
            "Starchy Roots + (Total)",
            "Stimulants > (List)",
            "Sugar Crops + (Total)",
            "Vegetables + (Total)",
            "Demersal Fish",
            "Freshwater Fish",
            "Aquatic Animals, Others",
            "Pelagic Fish",
            "Beer",
            "Beverages, Alcoholic",
            "Beverages, Fermented",
            "Wine",
            "Sugar (Raw Equivalent)",
            "Sweeteners, Other",
            "Vegetable Oils + (Total)",
            "Milk - Excluding Butter + (Total)",
            "Eggs + (Total)",
            "Animal fats + (Total)",
            "Offals + (Total)",
            "Bovine Meat",
            "Meat, Other",
            "Pigmeat",
            "Poultry Meat",
            "Mutton & Goat Meat",
            "Fish, Seafood + (Total)",
            "Coffee and products",
            "Grand Total + (Total)",
        ]

        # 1990 - 2013
        # ld = faostat.list_datasets()
        code = "FBSH"
        #
        my_countries = [
            faostat.get_par(code, "area")[c] for c in list_countries
        ]  # faostat.get_par(code, 'elements')
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
        df_diet_1990_2013 = faostat.get_data_df(code, pars=my_pars, strval=False)

        # 2010-2022
        list_items = [
            "Cereals - Excluding Beer + (Total)",
            "Fruits - Excluding Wine + (Total)",
            "Oilcrops + (Total)",
            "Pulses + (Total)",
            "Rice and products",
            "Starchy Roots + (Total)",
            "Stimulants > (List)",
            "Sugar Crops + (Total)",
            "Vegetables + (Total)",
            "Demersal Fish",
            "Freshwater Fish",
            "Aquatic Animals, Others",
            "Pelagic Fish",
            "Beer",
            "Beverages, Alcoholic",
            "Beverages, Fermented",
            "Wine",
            "Sugar (Raw Equivalent)",
            "Sweeteners, Other",
            "Vegetable Oils + (Total)",
            "Milk - Excluding Butter + (Total)",
            "Eggs + (Total)",
            "Animal fats + (Total)",
            "Offals + (Total)",
            "Bovine Meat",
            "Meat, Other",
            "Pigmeat",
            "Poultry Meat",
            "Mutton & Goat Meat",
            "Fish, Seafood + (Total)",
            "Coffee and products",
            "Grand Total + (Total)",
        ]
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
            "2022",
        ]
        my_years = [faostat.get_par(code, "year")[y] for y in list_years]

        my_pars = {
            "area": my_countries,
            "element": my_elements,
            "item": my_items,
            "year": my_years,
        }
        df_diet_2010_2022 = faostat.get_data_df(code, pars=my_pars, strval=False)

        # Renaming the items for name matching
        df_diet_1990_2013.loc[
            df_diet_1990_2013["Item"].str.contains(
                "Rice \(Milled Equivalent\)", case=False, na=False
            ),
            "Item",
        ] = "Rice and products"

        # Concatenating all the years together
        df_diet = pd.concat([df_diet_1990_2013, df_diet_2010_2022])

        # Filtering to keep wanted columns
        columns_to_filter = ["Area", "Element", "Item", "Year", "Value"]
        df_diet = df_diet[columns_to_filter]

        df_diet.to_csv(file, index=False)

    return df_diet
