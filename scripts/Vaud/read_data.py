import pandas as pd


def read_vaud_data():
    bilan = pd.read_excel(
        "data/Vaud/emission_comparaison.xlsx", sheet_name="emission_bilan_carbone"
    )
    bilan["Catégorie"] = bilan["Catégorie"].str.lower()
    bilan = bilan.loc[bilan["Years"] != 1990, :]
    sum_per_year = bilan.groupby(["Years"])[
        [
            "Activité [MWh]",
            "Emissions Territoriales [tCo2eq] GES",  # codespell:ignore GES
            "Emissions Extra-Territoriales  [tCo2eq] GES",  # codespell:ignore GES
            "Emissions Totales  [tCo2eq] GES",  # codespell:ignore GES
        ]
    ].sum()
    sum_per_year["Catégorie"] = "total"
    sum_per_year["Country"] = "Vaud"
    sum_per_year.reset_index(inplace=True)

    sum_per_year

    bilan = pd.concat([bilan, sum_per_year], ignore_index=True)

    groups = bilan.loc[bilan["Years"] != 1990, :].groupby(["Catégorie"])
    DIC_OFF_TO_MODEL = {
        "bois": "wood",
        "chauffage à distance": "district-heating",
        "electricité ": "electricity",
        "gaz": "gas",
        "mazout": "heating-oil",
        "solaire thermique": "solar",
        "total": "total",
    }

    bilan_carbone_tot = pd.read_excel(
        "data/Vaud/bilan_carbone_territoire_2023_public2(8).xlsx",
        sheet_name="Transports - Emissions",
        header=8,
    ).iloc[:, 1:-3]
    bilan_carbone_tot = bilan_carbone_tot[
        bilan_carbone_tot["Année"].isin([2015, 2019, 2023])
    ]
    bilan_carbone_tot = bilan_carbone_tot[
        bilan_carbone_tot["Catégorie"].isin(
            [
                "Mobilité active",
                "Moto/scooter",
                "Voiture",
                "Train",
                "Bus ",
                "Tram/métro",
            ]
        )
    ]
    bilan_carbone_tot = bilan_carbone_tot.drop("Sous-secteur", axis=1)
    bilan_carbone_tot.reset_index(inplace=True, drop=True)
    bilan_carbone_tot.rename(columns={"Année": "Years"}, inplace=True)
    bilan_carbone_tot["Catégorie"] = bilan_carbone_tot["Catégorie"].str.lower()
    bilan_carbone_tot["CO2"].astype(float)
    # display(bilan_carbone_tot)

    sum_per_year = bilan_carbone_tot.groupby(["Years"])[["Activité", "CO2"]].sum()
    sum_per_year["Catégorie"] = "total"
    sum_per_year.reset_index(inplace=True)

    bilan_carbone_tot = pd.concat([bilan_carbone_tot, sum_per_year], ignore_index=True)
    return bilan_carbone_tot, bilan, groups, DIC_OFF_TO_MODEL
