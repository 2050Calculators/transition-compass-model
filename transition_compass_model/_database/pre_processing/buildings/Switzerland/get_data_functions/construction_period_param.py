def load_construction_period_param():
    construction_period_envelope_cat_sfh_old = {
        "F": ["Avant 1919", "1919-1945", "1946-1960", "1961-1970"],
        "E": ["1971-1980"],
        "D": ["1981-1990", "1991-2000"],
        "C": ["2001-2005", "2006-2010"],
        "B": ["2011-2015", "2016-2020", "2021-2023"],
    }
    construction_period_envelope_cat_mfh_old = {
        "F": ["Avant 1919", "1919-1945", "1946-1960", "1961-1970", "1971-1980"],
        "E": ["1981-1990"],
        "D": ["1991-2000"],
        "C": ["2001-2005", "2006-2010"],
        "B": ["2011-2015", "2016-2020", "2021-2023"],
    }

    construction_period_envelope_cat_sfh = {
        "F": ["Period before 1919", "1919-1945", "1946-1960", "1961-1970"],
        "E": ["1971-1980"],
        "D": ["1981-1985", "1986-1990", "1991-1995", "1996-2000"],
        "C": ["2001-2005", "2006-2010"],
        "B": ["2011-2015", "2016-2020", "2021-2025"],
    }
    construction_period_envelope_cat_mfh = {
        "F": ["Period before 1919", "1919-1945", "1946-1960", "1961-1970", "1971-1980"],
        "E": ["1981-1985", "1986-1990"],
        "D": ["1991-1995", "1996-2000"],
        "C": ["2001-2005", "2006-2010"],
        "B": ["2011-2015", "2016-2020", "2021-2025"],
    }
    envelope_cat_new = {"D": (1990, 2000), "C": (2001, 2010), "B": (2011, 2023)}

    global_var = {
        "envelope construction sfh": construction_period_envelope_cat_sfh,
        "envelope construction mfh": construction_period_envelope_cat_mfh,
        "envelope construction sfh old": construction_period_envelope_cat_sfh_old,
        "envelope construction mfh old": construction_period_envelope_cat_mfh_old,
        "envelope cat new": envelope_cat_new,
    }
    return global_var
