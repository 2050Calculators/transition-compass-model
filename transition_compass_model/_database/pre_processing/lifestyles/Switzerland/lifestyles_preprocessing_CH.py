import os
import pickle
import time

import numpy as np

# Still on the old PX-Web API: neither table is on the Swiss Stats SDMX API yet
# (STATPOP population starts in 2010; no population scenario dataflow).
from transition_compass_model._database.pre_processing.api_routines_CH import (
    get_data_api_CH,
)
from transition_compass_model._database.pre_processing.params import (
    country_list,
    years_fts,
    years_ots,
)
from transition_compass_model.model.common.auxiliary_functions import (
    my_pickle_dump,
)


########################
###    POPULATION    ###
### tot & by age new ###
########################
def extract_lfs_pop(years_ots, table_id, file):
    file = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
    try:
        with open(file, "rb") as handle:
            dm_lfs_pop_age = pickle.load(handle)
    except OSError:
        # Demographic balance by age and canton
        structure, title = get_data_api_CH(table_id, mode="example")

        # Extract all age classes
        years_ots_str = [str(y) for y in years_ots]
        dm_lfs_pop_age = None
        for canton in structure["Canton"]:
            filter = {
                "Year": years_ots_str,
                "Canton": canton,
                "Citizenship (category)": "Citizenship (category) - total",  # Swiss and non-Swiss resident
                "Sex": ["Male", "Female"],
                "Age": structure["Age"],
                "Demographic component": "Population on 1 January",
            }

            mapping_dim = {
                "Country": "Canton",
                "Years": "Year",
                "Variables": "Demographic component",
                "Categories1": "Sex",
                "Categories2": "Age",
            }
            # Extract population by age group data. The FSO API rate-limits
            # requests (HTTP 429): get_data_api_CH then returns None or raises
            # KeyError, so wait and retry.
            for attempt in range(5):
                try:
                    dm_lfs_pop_age_canton = get_data_api_CH(
                        table_id,
                        mode="extract",
                        filter=filter,
                        mapping_dims=mapping_dim,
                        units=["inhabitants"],
                    )
                except KeyError:
                    dm_lfs_pop_age_canton = None
                if dm_lfs_pop_age_canton is not None:
                    break
                time.sleep(30)
            else:
                raise RuntimeError(f"FSO API request failed for canton {canton}")
            if dm_lfs_pop_age is None:
                dm_lfs_pop_age = dm_lfs_pop_age_canton
            else:
                dm_lfs_pop_age.append(dm_lfs_pop_age_canton, dim="Country")

        dm_lfs_pop_age.sort("Country")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "wb") as handle:
            pickle.dump(dm_lfs_pop_age, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dm_lfs_pop_age.rename_col_regex("Confederation", "Switzerland", "Country")
    dm_lfs_pop_age.rename_col_regex(" /.*", "", dim="Country")
    dm_lfs_pop_age.rename_col_regex("-", " ", dim="Country")
    dm_lfs_pop_age.rename_col(
        list(cantons_fso_to_en.keys()), list(cantons_fso_to_en.values()), dim="Country"
    )
    dm_lfs_pop_age.sort("Country")

    dm_lfs_pop_age.drop(col_label="No indication", dim="Country")
    dm_lfs_pop_tot = dm_lfs_pop_age.filter({"Categories2": ["Age - total"]})
    dm_lfs_pop_age.drop(dim="Categories2", col_label="Age - total")
    dm_lfs_pop_age.drop(col_label="No indication", dim="Categories2")
    dm_lfs_pop_age.rename_col_regex(" years", "", dim="Categories2")
    dm_lfs_pop_age.rename_col_regex(" year", "", dim="Categories2")
    dm_lfs_pop_age.rename_col("99 or older", "99", dim="Categories2")
    dm_lfs_pop_age.drop(dim="Categories2", col_label="No indication")

    # Group ages by category
    group_dict = {
        "below19": [],
        "age20-29": [],
        "age30-54": [],
        "age55-64": [],
        "above65": [],
    }
    for age in dm_lfs_pop_age.col_labels["Categories2"]:
        if int(age) <= 19:
            group_dict["below19"].append(age)
        elif (int(age) >= 20) and (int(age) <= 29):
            group_dict["age20-29"].append(age)
        elif (int(age) >= 30) and (int(age) <= 54):
            group_dict["age30-54"].append(age)
        elif (int(age) >= 55) and (int(age) <= 64):
            group_dict["age55-64"].append(age)
        elif int(age) >= 65:
            group_dict["above65"].append(age)

    dm_lfs_pop_age.groupby(group_dict, "Categories2", inplace=True, regex=False)
    # Rename sex
    dm_lfs_pop_age.rename_col("Male", "male", dim="Categories1")
    dm_lfs_pop_age.rename_col("Female", "female", dim="Categories1")
    dm_lfs_pop_age.sort("Categories1")
    dm_lfs_pop_age.rename_col(
        "Population on 1 January", "lfs_demography", dim="Variables"
    )
    dm_lfs_pop_age = dm_lfs_pop_age.flatten(sep="-")

    dm_lfs_pop_tot.group_all("Categories2")
    dm_lfs_pop_tot.group_all("Categories1")
    dm_lfs_pop_tot.rename_col(
        "Population on 1 January", "lfs_population_total", dim="Variables"
    )

    # Sort Years
    dm_lfs_pop_age.sort("Years")
    dm_lfs_pop_tot.sort("Years")

    return dm_lfs_pop_age, dm_lfs_pop_tot


def extract_lfs_pop_fts(years_fts, table_id, file):
    file = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
    try:
        with open(file, "rb") as handle:
            dm_pop_fts = pickle.load(handle)
    except OSError:
        structure, title = get_data_api_CH(table_id, mode="example", language="fr")
        # Extract buildings floor area
        scenarios = [
            "Scénario de référence A-00-2025",
            "Scénario B-00-2025 'haut'",
            "Variante A-03-2025 'plus haute espérance de vie à la naissance'",
            "Scénario C-00-2025 'bas'",
        ]
        filter = {
            "Scénario-variante": scenarios,
            "Nationalité (catégorie)": ["Nationalité - total"],
            "Sexe": ["Homme", "Femme"],
            "Classe d'âge": structure["Classe d'âge"],
            "Unité d'observation": ["Population au 1er janvier"],
            "Année": structure["Année"],
        }
        mapping_dim = {
            "Country": "Nationalité (catégorie)",
            "Years": "Année",
            "Variables": "Scénario-variante",
            "Categories1": "Sexe",
            "Categories2": "Classe d'âge",
        }
        unit_all = ["inhabitants"] * len(filter["Scénario-variante"])
        # Get api data
        dm_pop_fts = get_data_api_CH(
            table_id,
            mode="extract",
            filter=filter,
            mapping_dims=mapping_dim,
            units=unit_all,
            language="fr",
        )
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "wb") as handle:
            pickle.dump(dm_pop_fts, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Rename
    dm_pop_fts.rename_col("Nationalité - total", "Switzerland", dim="Country")

    # Filter only fts years
    dm_pop_fts.filter({"Years": years_fts}, inplace=True)
    dm_pop_fts.sort(dim="Years")

    dm_pop_fts_tot = dm_pop_fts.filter({"Categories2": ["Classe d'âge - total"]})
    dm_pop_fts_tot.group_all(dim="Categories2")
    dm_pop_fts_tot.group_all(dim="Categories1")

    dm_pop_fts.drop(dim="Categories2", col_label="Classe d'âge - total")
    dm_pop_fts.rename_col_regex(" ans", "", dim="Categories2")

    # Group ages by category
    group_dict = {
        "below19": [],
        "age20-29": [],
        "age30-54": [],
        "age55-64": [],
        "above65": [],
    }
    min_age = [col.split("-")[0] for col in dm_pop_fts.col_labels["Categories2"]]
    dm_pop_fts.rename_col(
        col_in=dm_pop_fts.col_labels["Categories2"].copy(),
        col_out=min_age,
        dim="Categories2",
    )

    for age in dm_pop_fts.col_labels["Categories2"]:
        if int(age) <= 19:
            group_dict["below19"].append(age)
        elif (int(age) >= 20) and (int(age) <= 29):
            group_dict["age20-29"].append(age)
        elif (int(age) >= 30) and (int(age) <= 54):
            group_dict["age30-54"].append(age)
        elif (int(age) >= 55) and (int(age) <= 64):
            group_dict["age55-64"].append(age)
        elif int(age) >= 65:
            group_dict["above65"].append(age)

    dm_pop_fts.groupby(group_dict, "Categories2", inplace=True, regex=False)
    # Rename sex
    dm_pop_fts.rename_col("Homme", "male", dim="Categories1")
    dm_pop_fts.rename_col("Femme", "female", dim="Categories1")
    dm_pop_fts.sort("Categories1")
    dm_pop_fts = dm_pop_fts.flatten(sep="-")

    # Assign levers
    level_dict = {
        1: "Scénario de référence A-00-2025",
        2: "Scénario B-00-2025 'haut'",
        3: "Variante A-03-2025 'plus haute espérance de vie à la naissance'",
        4: "Scénario C-00-2025 'bas'",
    }
    dict_dm_pop_fts = dict()
    dict_dm_pop_fts_tot = dict()
    for k, v in level_dict.items():
        dict_dm_pop_fts[k] = dm_pop_fts.filter({"Variables": [v]})
        dict_dm_pop_fts[k].rename_col(v, "lfs_demography", dim="Variables")

        dict_dm_pop_fts_tot[k] = dm_pop_fts_tot.filter({"Variables": [v]})
        dict_dm_pop_fts_tot[k].rename_col(v, "lfs_population_total", dim="Variables")

    return dict_dm_pop_fts, dict_dm_pop_fts_tot


def add_cantons_fts_pop(
    dm_lfs_age, dm_lfs_tot_pop, dict_lfs_age_fts, dict_lfs_tot_pop_fts, cantons
):
    # FSO scenarios exist only for Switzerland: each canton keeps its share of
    # the Swiss population in the last ots year
    idx = dm_lfs_tot_pop.idx
    for canton in cantons:
        vaud_share = (
            dm_lfs_tot_pop.array[idx[canton], -1, ...]
            / dm_lfs_tot_pop.array[idx["Switzerland"], -1, ...]
        )
        # Check that ots and fts are harmonised
        for l, dm_fts in dict_lfs_tot_pop_fts.items():
            vaud_arr = vaud_share * dm_fts["Switzerland", :, :]
            dm_fts.add(vaud_arr, dim="Country", col_label=canton)
            # Remove comment to check smoothness
            # dm_fts.append(dm_lfs_tot_pop, dim='Years')
            # dm_fts.sort(dim='Years')
            # dm_fts.datamatrix_plot(title=l)

        idx = dm_lfs_tot_pop.idx
        vaud_share = (
            dm_lfs_age.array[idx[canton], -1, ...]
            / dm_lfs_age.array[idx["Switzerland"], -1, ...]
        )
        for l, dm_fts in dict_lfs_age_fts.items():
            vaud_arr = (
                vaud_share[np.newaxis, np.newaxis, ...] * dm_fts["Switzerland", :, :, :]
            )
            dm_fts.add(vaud_arr, dim="Country", col_label=canton)

            # Remove comment to check smoothness
            # dm_fts.append(dm_lfs_age, dim='Years')
            # dm_fts.sort(dim='Years')
            # dm_fts.datamatrix_plot(title=l)
    for l, dm_fts in dict_lfs_tot_pop_fts.items():
        dm_fts.sort("Country")
    for l, dm_fts in dict_lfs_age_fts.items():
        dm_fts.sort("Country")

    return


# FSO canton names that differ from the names used in the model
cantons_fso_to_en = {"Genève": "Geneva", "Luzern": "Lucerne", "Zürich": "Zurich"}

# Get population total and by age group (ots)
filename = "data/lfs_pop_ots_all_cantons.pickle"
dm_lfs_age, dm_lfs_tot_pop = extract_lfs_pop(
    years_ots, table_id="px-x-0102020000_104", file=filename
)
# Get raw fts pop data (fts)
filename = "data/lfs_pop_fts.pickle"
dict_lfs_age_fts, dict_lfs_tot_pop_fts = extract_lfs_pop_fts(
    years_fts, table_id="px-x-0104000000_101", file=filename
)

# Keep only the regions in params.country_list
missing = set(country_list) - set(dm_lfs_tot_pop.col_labels["Country"])
if missing:
    raise ValueError(f"Regions in params.country_list not found in FSO data: {missing}")
cantons = [c for c in country_list if c != "Switzerland"]
add_cantons_fts_pop(
    dm_lfs_age, dm_lfs_tot_pop, dict_lfs_age_fts, dict_lfs_tot_pop_fts, cantons
)
dm_lfs_age.filter({"Country": country_list}, inplace=True)
dm_lfs_tot_pop.filter({"Country": country_list}, inplace=True)
for dict_fts in [dict_lfs_age_fts, dict_lfs_tot_pop_fts]:
    for dm_fts in dict_fts.values():
        dm_fts.filter({"Country": country_list}, inplace=True)

# Store
DM_lfs = {
    "ots": {"pop": {"lfs_demography_": [], "lfs_population_": []}},
    "fts": {"pop": {"lfs_demography_": dict(), "lfs_population_": dict()}},
}
DM_lfs["ots"]["pop"]["lfs_demography_"] = dm_lfs_age.filter({"Years": years_ots})
for lev in range(4):
    lev = lev + 1
    DM_lfs["fts"]["pop"]["lfs_demography_"][lev] = dict_lfs_age_fts[lev]
DM_lfs["ots"]["pop"]["lfs_population_"] = dm_lfs_tot_pop.filter({"Years": years_ots})
for lev in range(4):
    lev = lev + 1
    DM_lfs["fts"]["pop"]["lfs_population_"][lev] = dict_lfs_tot_pop_fts[lev]

# Save
current_file_directory = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(
    current_file_directory, "../../../data/datamatrix/lifestyles.pickle"
)

my_pickle_dump(DM_new=DM_lfs, local_pickle_file=file)
