from transition_compass_model.model.common.auxiliary_functions import (
    create_years_list,
)

years_ots = create_years_list(1990, 2023, 1)
years_fts = create_years_list(2025, 2050, 5)

country_list = ["Switzerland", "Vaud", "Fribourg", "Schwyz"]


cantons_name = {
    "accronym_to_name": {
        "AG": "Aargau",
        "AR": "Appenzell Ausserrhoden",
        "AI": "Appenzell Innerrhoden",
        "BL": "Basel-Landschaft",
        "BS": "Basel-Stadt",
        "BE": "Bern",
        "FR": "Fribourg",
        "FB": "Fribourg",
        "GE": "Geneva",
        "GL": "Glarus",
        "GR": "Graubunden",
        "JU": "Jura",
        "LU": "Lucerne",
        "NE": "Neuchatel",
        "NW": "Nidwalden",
        "OW": "Obwalden",
        "SH": "Schaffhausen",
        "SZ": "Schwyz",
        "SO": "Solothurn",
        "SG": "St. Gallen",
        "TI": "Ticino",
        "TG": "Thurgau",
        "UR": "Uri",
        "VD": "Vaud",
        "VS": "Valais",
        "ZG": "Zug",
        "ZH": "Zurich",
    },
    "name_to_accronym": {
        "Aargau": "AG",
        "Appenzell Ausserrhoden": "AR",
        "Appenzell Innerrhoden": "AI",
        "Basel-Landschaft": "BL",
        "Basel-Stadt": "BS",
        "Bern": "BE",
        "Fribourg": "FR",
        "Geneva": "GE",
        "Glarus": "GL",
        "Graubunden": "GR",
        "Jura": "JU",
        "Lucerne": "LU",
        "Neuchatel": "NE",
        "Nidwalden": "NW",
        "Obwalden": "OW",
        "Schaffhausen": "SH",
        "Schwyz": "SZ",
        "Solothurn": "SO",
        "St. Gallen": "SG",
        "Ticino": "TI",
        "Thurgau": "TG",
        "Uri": "UR",
        "Vaud": "VD",
        "Valais": "VS",
        "Zug": "ZG",
        "Zurich": "ZH",
    },
}
