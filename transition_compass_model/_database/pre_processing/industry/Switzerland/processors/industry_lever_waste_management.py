import re
import numpy as np
import os
import pandas as pd
import pickle
import warnings

warnings.simplefilter("ignore")
import plotly.io as pio

pio.renderers.default = "browser"

from transition_compass_model.model.common.data_matrix_class import DataMatrix
from transition_compass_model.model.common.auxiliary_functions import linear_fitting
from transition_compass_model.model.common.auxiliary_functions import create_years_list

from _database.pre_processing.industry.Switzerland.get_data_functions.data_waste import (
    get_data_waste_vehicles,
    get_data_special_waste,
    extract_waste_data_from_dechets_speciaux,
)


def get_pickle_eu(current_file_directory):

    filepath = os.path.join(
        current_file_directory, "../../../../data/datamatrix/industry.pickle"
    )
    with open(filepath, "rb") as handle:
        DM = pickle.load(handle)
    dm_eu = DM["ots"]["eol-waste-management"].filter({"Country": ["EU27"]})

    return dm_eu


def waste_vehicles(current_file_directory, dm_eu_vehicles, years_ots):

    dm = get_data_waste_vehicles(current_file_directory)

    # make layer 1
    dm_layer1 = dm.filter({"Variables": ["waste-tot", "waste-collected"]})
    dm_layer1.operation(
        "waste-collected", "/", "waste-tot", "Variables", "waste-collected-share", "%"
    )
    df_temp = dm_layer1.write_df()

    # make uncollected
    years_selection = dm_layer1.col_labels["Years"].copy()
    years_selection.remove(2024)
    dm_temp = dm_eu_vehicles.filter(
        {
            "Country": ["EU27"],
            "Years": years_selection,
            "Categories1": ["waste-uncollected"],
        }
    )
    dm_temp = dm_temp.flatten()
    dm_temp.add(dm_temp[:, 2023, ...], "Years", [2024], dummy=True)
    dm_temp.sort("Years")
    dm_layer1.add(dm_temp.array, "Variables", "waste-uncollected-share", "%")
    df_temp = dm_layer1.write_df()

    # make export
    arr_temp = (
        1
        - dm_layer1[..., "waste-collected-share"]
        - dm_layer1[..., "waste-uncollected-share"]
    )
    dm_layer1.add(arr_temp, "Variables", "export-share", "%")
    df_temp = dm_layer1.write_df()

    # clean layer 1
    dm_layer1.drop("Variables", ["waste-collected", "waste-tot"])
    dm_layer1.rename_col_regex("-share", "", "Variables")
    dm_layer1.add(0, "Variables", "littered", "%", True)

    # make layer 2
    dm_layer2 = dm.filter(
        {"Variables": ["waste-collected", "energy-recovery", "layer2-else"]}
    )
    dm_layer2.operation(
        "energy-recovery",
        "/",
        "waste-collected",
        "Variables",
        "energy-recovery-share",
        "%",
    )
    dm_temp = dm_layer2.filter({"Variables": ["energy-recovery-share"]})
    dm_temp.array[dm_temp.array > 1] = np.nan
    dm_layer2.drop("Variables", "energy-recovery-share")
    dm_layer2.append(dm_temp, "Variables")
    dm_layer2.add(0, "Variables", "incineration-share", "%", True)
    dm_layer2 = linear_fitting(dm_layer2, dm_layer2.col_labels["Years"])
    df_temp = dm_layer2.write_df()

    # get data dechet speciaux
    df = get_data_special_waste(current_file_directory)
    dm_temp = dm_layer2.filter({"Variables": ["waste-collected"]})
    dm_temp.array = dm_temp.array * 1.2
    dm_temp.units["waste-collected"] = "t"
    np.array(df["Total"])[np.newaxis, :, np.newaxis] / dm_temp.array
    # ok so almost zero recycled batteries so far, so we put zero recycled
    dm_layer2.add(0, "Variables", "recycling-share", "%", True)

    # for reuse and landfill, use same percentages than eu27
    years_selection = dm_layer1.col_labels["Years"].copy()
    years_selection.remove(2024)
    dm_temp = dm_eu_vehicles.filter(
        {
            "Country": ["EU27"],
            "Years": years_selection,
            "Categories1": ["landfill", "reuse"],
        }
    )
    dm_temp = dm_temp.flatten()
    dm_temp.rename_col(
        ["vehicles_landfill", "vehicles_reuse"],
        ["landfill-share", "reuse-share"],
        "Variables",
    )
    dm_temp.add(dm_temp[:, 2023, ...], "Years", [2024], dummy=True)
    dm_temp.sort("Years")
    dm_temp.rename_col(["EU27"], ["Switzerland"], "Country")
    dm_temp.append(
        dm_layer2.filter(
            {
                "Variables": [
                    "energy-recovery-share",
                    "incineration-share",
                    "recycling-share",
                ]
            }
        ),
        "Variables",
    )
    arr_temp = (
        1
        - dm_temp[..., "energy-recovery-share"]
        - dm_temp[..., "incineration-share"]
        - dm_temp[..., "recycling-share"]
    )
    dm_temp.add(arr_temp, "Variables", "else-share", "%")
    dm_temp = dm_temp.filter(
        {"Variables": ["landfill-share", "reuse-share", "else-share"]}
    )
    df_temp = dm_temp.write_df()
    mysum = dm_temp[..., "landfill-share"] + dm_temp[..., "reuse-share"]
    arr_temp = dm_temp[..., "landfill-share"] * dm_temp[..., "else-share"] / mysum
    dm_temp.drop("Variables", "landfill-share")
    dm_temp.add(arr_temp, "Variables", "landfill-share", "%")
    arr_temp = dm_temp[..., "reuse-share"] * dm_temp[..., "else-share"] / mysum
    dm_temp.drop("Variables", "reuse-share")
    dm_temp.add(arr_temp, "Variables", "reuse-share", "%")
    df_temp = dm_temp.write_df()
    dm_layer2.append(
        dm_temp.filter({"Variables": ["landfill-share", "reuse-share"]}), "Variables"
    )
    dm_layer2.drop("Variables", ["energy-recovery", "layer2-else", "waste-collected"])
    dm_layer2.rename_col_regex("-share", "", "Variables")
    np.sum(dm_layer2.array, 2)

    # put together
    dm_waste = dm_layer1.copy()
    variables = dm_waste.col_labels["Variables"]
    for v in variables:
        dm_waste.rename_col(v, "vehicles_" + v, "Variables")
    dm_waste.deepen()
    variables = dm_layer2.col_labels["Variables"]
    for v in variables:
        dm_layer2.rename_col(v, "vehicles_" + v, "Variables")
    dm_layer2.deepen()
    dm_waste.append(dm_layer2, "Categories1")
    dm_waste.sort("Categories1")

    # make ots and fts
    dm_waste = linear_fitting(dm_waste, years_ots, based_on=[2012])
    # dm_waste = linear_fitting(dm_waste,years_fts, based_on=[2024])
    # dm_waste.datamatrix_plot()
    dm_waste.drop("Years", [2024])

    return dm_waste


def waste_ships(dm):

    # source: https://shipbreakingplatform.org/platform-publishes-list-2024/
    # in 2024, 100% of swiss ships being dismantled were dismantled in India or other countries
    # so we will put export 100% and rest to zero

    dm[..., "export"] = 1
    variabs = [
        "littered",
        "waste-collected",
        "waste-uncollected",
        "energy-recovery",
        "incineration",
        "landfill",
        "recycling",
        "reuse",
    ]
    for v in variabs:
        dm[..., v] = 0

    return dm


def waste_buildings(current_file_directory, dm_eu_bld, years_ots):

    # yearly pdfs on déchets spéciaux (probably Déchets minéraux or Déchets de chantier non triés problématiques)
    # layer 1 should be collected or exported
    # layer 2 from pdf

    # note: the class "Déchets de chantier non triés problématiques" is only about polluted waste from construction sites
    # not sure how representative can be of the eol of a demolished building.
    dm_bld = extract_waste_data_from_dechets_speciaux(
        current_file_directory,
        "Déchets de chantier non triés problématiques",
        "floor-area-new-residential",
    )

    # make layer1
    dm_layer1 = dm_bld.filter({"Categories1": ["total", "export"]})
    dm_layer1.operation("total", "+", "export", "Categories1", "waste-collected", "t")
    dm_layer1.add(0, "Categories1", "waste-uncollected", "t", True)
    dm_layer1.add(0, "Categories1", "littered", "t", True)
    dm_layer1.drop("Categories1", "total")
    dm_layer1.normalise("Categories1")
    df_temp = dm_layer1.write_df()
    df_temp2 = dm_eu_bld.filter(
        {
            "Country": ["EU27"],
            "Years": dm_layer1.col_labels["Years"],
            "Variables": ["floor-area-new-residential"],
            "Categories1": dm_layer1.col_labels["Categories1"],
        }
    ).write_df()
    # seems ok

    # make layer2
    dm_layer2 = dm_bld.filter(
        {"Categories1": ["energy-recovery", "landfill", "recycling"]}
    )
    dm_layer2.add(0, "Categories1", "reuse", "t", True)
    dm_layer2.add(0, "Categories1", "incineration", "t", True)
    dm_layer2.normalise("Categories1")
    df_temp = dm_layer2.write_df()
    df_temp2 = dm_eu_bld.filter(
        {
            "Country": ["EU27"],
            "Years": dm_layer2.col_labels["Years"],
            "Variables": ["floor-area-new-residential"],
            "Categories1": dm_layer2.col_labels["Categories1"],
        }
    ).write_df()
    # seems ok

    # put together
    dm_waste_bld = dm_layer1.copy()
    dm_waste_bld.append(dm_layer2, "Categories1")
    dm_waste_bld.sort("Categories1")

    # make ots and fts
    dm_waste_bld = linear_fitting(dm_waste_bld, years_ots, based_on=[2014])
    # dm_waste_bld = linear_fitting(dm_waste_bld,years_fts, based_on=[2023])
    # dm_waste_bld.datamatrix_plot()

    # put together with overall waste
    arr_temp = dm_waste_bld.array
    dm_waste_bld.add(arr_temp, "Variables", "floor-area-new-non-residential", "%")

    return dm_waste_bld


def waste_roads(current_file_directory, dm_eu_roads):

    # either like buildings or Matériaux bitumineux de démolition des routes > 20'000 mg/kg HAP
    # note: this is polluted waste from road demolition, so possibly the numbers will be a bit different for overall
    # waste from road demolition. We consider it as an approximation (probably higher bound)

    dm_roads = extract_waste_data_from_dechets_speciaux(
        current_file_directory,
        "Matériaux bitumineux de démolition des routes >",
        "road",
    )
    df_temp = dm_roads.write_df()

    # make layer1
    dm_layer1 = dm_roads.filter({"Categories1": ["total", "export"]})
    dm_layer1.operation("total", "+", "export", "Categories1", "waste-collected", "t")
    dm_layer1.add(0, "Categories1", "waste-uncollected", "t", True)
    dm_layer1.add(0, "Categories1", "littered", "t", True)
    dm_layer1.drop("Categories1", "total")
    dm_layer1.normalise("Categories1")
    df_temp = dm_layer1.write_df()
    df_temp2 = dm_eu_roads.filter(
        {
            "Country": ["EU27"],
            "Years": dm_layer1.col_labels["Years"],
            "Variables": ["road"],
            "Categories1": dm_layer1.col_labels["Categories1"],
        }
    ).write_df()
    # seems ok

    # make layer2
    dm_layer2 = dm_roads.filter(
        {"Categories1": ["energy-recovery", "landfill", "recycling"]}
    )
    dm_layer2.add(0, "Categories1", "reuse", "t", True)
    dm_layer2.add(0, "Categories1", "incineration", "t", True)
    dm_layer2.normalise("Categories1")
    df_temp = dm_layer2.write_df()
    df_temp2 = dm_eu_roads.filter(
        {
            "Country": ["EU27"],
            "Years": dm_layer2.col_labels["Years"],
            "Variables": ["road"],
            "Categories1": dm_layer2.col_labels["Categories1"],
        }
    ).write_df()
    # this one seems a lot towards landfilling and little recycling

    # so I will take the EU values for roads
    dm_eu_roads.rename_col("EU27", "Switzerland", "Country")

    return


def waste_appliances(dm_eu):

    # electronics
    # from here: https://www.bafu.admin.ch/bafu/en/home/topics/waste/guide-to-waste-a-z/electrical-and-electronic-equipment.html
    # "The dismantling and separation of equipment into fractions is mainly carried out in Switzerland.
    # The other processing stages are often carried out abroad because non-ferrous metals processing systems, in particular, are not available in Switzerland."

    # so probably when it comes to the recycling of aluminium and copper (non-ferrous metal), I will have to put a zero
    # everywhere for Switzerland. To be done in the run.
    # however, documents on aluminium packaging are saying that that's recycled (recyclage_des_emballagespourboissonsen2014)
    # maybe recycling of standard aluminium is not currently done, but packaging yes ... to be understood

    # there is some steel in electronics and appliances, so for those two you can probably put the same of Déchets métalliques

    # "Consumers, in turn, are obliged to return equipment. The disposal of used equipment through municipal solid waste or
    # bulk waste collections is prohibited. These regulations are contained in the Ordinance on the Return,
    # Taking Back and Disposal of Electrical and Electronic Equipment (ORDEE).
    # The following categories of electrically operated equipment are regulated by the ORDEE:
    # Electronic entertainment equipment
    # Office, information, communications technology equipment
    # Refrigeration equipment
    # Household equipment
    # Tools (excluding large-scale, stationary industrial tools)
    # Sport and leisure equipment and toys
    # Luminaries and lighting control equipment

    # Note that from 0_Déchets 2023  Quantités produites et recyclées
    # you know how much is recycled, i.e. 132’100 t in 2023

    # Ok so summary:
    # main report is this: https://www.swico.ch/media/filer_public/4b/cf/4bcf42d4-60f6-4fd0-8c3c-4dbcef4e8475/220613-se-fachbericht-en-rz.pdf
    # numbers of overall waste are generally aligned with Déchets Quantités produites et recyclées, under Appareils électriques et électroniques
    # Figure 1 of the report says that around 95% of electronic waste is collected, and about 75% is recycled.
    # Let's see how these numbers compare to eu

    # electronics
    dm_ele = dm_eu.filter({"Country": ["EU27"], "Variables": ["electronics"]})
    dm_ele.rename_col("EU27", "Switzerland", "Country")
    df_temp = dm_ele.filter(
        {"Categories1": ["waste-collected", "waste-uncollected", "export", "littered"]}
    ).write_df()
    # all years: collected 0.8, uncollected 0.2, can be changed to 0.95 and 0.05
    df_temp = dm_ele.filter(
        {
            "Categories1": [
                "recycling",
                "incineration",
                "energy-recovery",
                "reuse",
                "landfill",
            ]
        }
    ).write_df()
    # in 2023: 87% recucling, 5% energy recovery, 5% landfilling, 3% reuse (not too far from the 75% - rest from the graph)

    # fixes
    dm_ele[..., "waste-collected"] = 0.95
    dm_ele[..., "waste-uncollected"] = 0.05
    dm_ele[..., "export"] = (
        0  # assuming export zero: The export and import of such waste requires the authorisation of the FOEN. Export to states that are not members of the OECD or EU is prohibited.
    )
    dm_ele[..., "littered"] = 0

    # appliances
    dm_app = dm_eu.filter({"Country": ["EU27"], "Variables": ["domapp"]})
    dm_app.rename_col("EU27", "Switzerland", "Country")
    df_temp = dm_app.filter(
        {"Categories1": ["waste-collected", "waste-uncollected", "export", "littered"]}
    ).write_df()
    # same than electronics
    df_temp = dm_app.filter(
        {
            "Categories1": [
                "recycling",
                "incineration",
                "energy-recovery",
                "reuse",
                "landfill",
            ]
        }
    ).write_df()
    # 2023: similar to electronics, so not too far from the 75% - rest from the graph

    # fixes
    dm_app[..., "waste-collected"] = 0.95
    dm_app[..., "waste-uncollected"] = 0.05
    dm_app[..., "export"] = (
        0  # assuming export zero: The export and import of such waste requires the authorisation of the FOEN. Export to states that are not members of the OECD or EU is prohibited.
    )
    dm_app[..., "littered"] = 0

    return dm_ele, dm_app


def waste_pack_glass(dm_eu):

    # for glass:
    # https://aureverre.ch/faits-et-chiffres
    # https://www.vetroswiss.ch/fr/vetroswiss/rapport-annuel/

    # Parmi ces bouteilles en verre, 44% sont produites en Suisse, tandis que les 56% restantes sont importés

    # Sur l’ensemble du verre collecté en 2020,
    # environ 64% ont été exportés,
    # 24% recyclés (voir encadré pour la définition du taux de recyclage),
    # 9% décyclés en sable de verre,
    # 2% a été incinéré
    # et seulement 0.6% a été réutilisé

    # so the import-export can be relevant, as for example from Recyclage des emballages pour boissons
    # we see that in 2023 Quantité consommée of verre is 294’737 tonnes, and recycled is 295’753 tonnes
    # but so there must be quite some import (and potentially export, judging from the figures of reports
    # above). For the time dimension, you can assume it's been like this since the 90's.

    # for other packaging: reuse will be set to zero, for export of waste of alu, paper and plastic
    # find data with chat. Then, for layer 1, you can probably take eu data (to see if we have littered
    # there, but i think so), and adapt the export. For layer 2, you can take recycling from je-f-02.03.02.11
    # and set rest to energy-recovery

    dm = dm_eu.filter({"Country": ["EU27"], "Variables": ["glass-pack"]})
    dm.rename_col("EU27", "Switzerland", "Country")

    dm[..., "export"] = 0.64
    dm[..., "waste-collected"] = 1 - 0.64
    dm[..., "waste-uncollected"] = 0
    dm[..., "littered"] = 0

    tot = 0.24 + 0.09 + 0.02 + 0.006
    dm[..., "recycling"] = (0.24 + 0.09) / tot
    dm[..., "energy-recovery"] = (0.02) / tot
    dm[..., "reuse"] = (0.006) / tot
    dm[..., "incineration"] = 0
    dm[..., "landfill"] = 0

    return dm


def waste_pack_plastic(dm_eu):

    # plastic pack

    # for plastic packaging waste management, we will need to see if to consider the PET data (which is only one share of plastic)
    # or rather info from https://www.bafu.admin.ch/bafu/en/home/topics/waste/guide-to-waste-a-z/plastics.html:
    # "Around 790,000 tonnes of plastic waste are generated every year, almost half of which is used for less
    # than a year, e.g. as packaging. Around 83% per cent (660,000 tonnes) of plastic waste is recovered for
    # energy in waste incineration plants and around 2% (10,000 tonnes) in cement works.
    # Around nine per cent (70,000 tonnes) is processed into recycled material.
    # A further six per cent (50,000 tonnes) of plastic waste is reused, for example textiles."

    # So basically only 9% is recycled (while around 80% of PET is recycled, so most of those 9% will be PET)
    # And we could say that around 790,000/2 is the packaging, and 790,000/2/population can be the
    # packaging per capita number (considering only PET would underestimate it)

    # https://plasticrecycler.ch/wp-content/uploads/2025/07/250701_Monitoringbericht_2024_FR_final.pdf
    # Figure 5: we export all of it for tri, then re-import for recycling and energy recovery

    # littering: https://www.news.admin.ch/en/nsb?id=75798
    # https://www.empa.ch/web/s604/mikroplastik-bafu
    # 5000 tons of plastic released into the environment every year
    # Overall, around 5,120 tons of the seven types of plastic are discharged into the environment each year. This is around 0.7% of the total amount of the seven plastics consumed in Switzerland each year (amounting to a total of around 710,000 tons). According to Empa’s modelling, around 4,400 tons of macroplastic are deposited on soils every year.

    dm = dm_eu.filter({"Country": ["EU27"], "Variables": ["plastic-pack"]})
    dm.rename_col("EU27", "Switzerland", "Country")
    # dm.add(0, "Years", [2024], "%", dummy=True)
    dm.array[...] = np.nan
    # dm.units['plastic-pack'] = "t"

    export = 0.3  # (11678-2507-3460)/11678 = 0.49 from the plastic recycler report above seems high, so we put 30%
    littered = 0.007
    collected = 1 - export - littered
    uncollected = 0  # we assume that in CH the uncollected is in littered, and that there are no communes without collection service
    dm[..., "export"] = export
    dm[..., "littered"] = littered
    dm[..., "waste-collected"] = collected
    dm[..., "waste-uncollected"] = uncollected

    dm[..., "energy-recovery"] = 0.85
    dm[..., "recycling"] = 0.09
    dm[..., "landfill"] = 0
    dm[..., "reuse"] = 0.06
    dm[..., "incineration"] = 0

    # # 2024
    # factor_for_littered = 0.03 # as the report covers only around 3% of PET / plastic waste
    # layer1_dict = {"littered" : 5000*factor_for_littered, "export" : 11678-2507-3460,
    #                "waste-collected" : 11695, "waste-uncollected" : 0}
    # layer2_dict = {"energy-recovery" : 199+3460, "recycling" : 2315,
    #                "incineration" : 0, "reuse" : 0, "landfill" : 0}
    # total_layer1 = layer1_dict["export"] + layer1_dict["waste-collected"] + layer1_dict["waste-uncollected"] + layer1_dict["littered"]
    # total_layer2 = layer2_dict["energy-recovery"] + layer2_dict["recycling"] + layer2_dict["incineration"] + layer2_dict["reuse"] + layer2_dict["landfill"]
    # for key in layer1_dict.keys(): dm[:,2024,:,key] = layer1_dict[key]/total_layer1
    # for key in layer2_dict.keys(): dm[:,2024,:,key] = layer2_dict[key]/total_layer2

    # # get time series
    # dm = linear_fitting(dm, years_ots)
    # # df_temp = dm.write_df()
    # # dm.flatten().datamatrix_plot()

    # alternative with total time trends
    # =============================================================================
    # # 2022
    # # https://plasticrecycler.ch/wp-content/uploads/2024/07/230629_Monitoringbericht_2022_final_FR.pdf
    # layer1_dict = {"littered" : 3800, # I set 3500 to have a constant trend with linear fitting
    #                "export" : 9525-2431-2760,
    #                "waste-collected" : 9553, "waste-uncollected" : 0}
    # layer2_dict = {"energy-recovery" : 193+2760, "recycling" : 2069,
    #                "incineration" : 0, "reuse" : 0, "landfill" : 0}
    # total_layer1 = layer1_dict["export"] + layer1_dict["waste-collected"] + layer1_dict["waste-uncollected"] + layer1_dict["littered"]
    # total_layer2 = layer2_dict["energy-recovery"] + layer2_dict["recycling"] + layer2_dict["incineration"] + layer2_dict["reuse"] + layer2_dict["landfill"]
    # for key in layer1_dict.keys(): dm[:,2022,:,key] = layer1_dict[key]/total_layer1
    # for key in layer2_dict.keys(): dm[:,2022,:,key] = layer2_dict[key]/total_layer2
    #
    # # get pet through time for making trends
    # filepath = os.path.join(current_file_directory, '../data/waste/0_Déchets Quantités produites et recyclées/je-f-02.03.02.11.xlsx')
    # df = pd.read_excel(filepath)
    # df.columns = df.iloc[3,:]
    # df = df.loc[df["Matériaux "].isin(["PET "]),:]
    # df = pd.melt(df, id_vars = ["Matériaux ",'Unité'], var_name='year')
    # df.loc[df["value"] == '…',"value"] = np.nan
    # df["value"] = df["value"].astype(float)
    # val_2022 = df.loc[df["year"] == 2022,"value"].values
    # df["change"] = (df["value"] - val_2022)/val_2022
    # for y in list(range(1993,2023)):
    #     dm[:,y,...] = dm[:,2022,...] * (1+df.loc[df["year"]==y,"change"].values)
    #
    # # fill nas
    # dm.drop("Years",years_fts)
    # dm = linear_fitting(dm, years_ots)
    # # dm.flatten().datamatrix_plot()
    # dm.drop("Years",[1990,1991,1992])
    # dm = linear_fitting(dm, years_ots,based_on=[1993])
    # # dm.flatten().datamatrix_plot()
    # dm = linear_fitting(dm, years_fts)
    # # dm.flatten().datamatrix_plot()
    # =============================================================================

    # # put together
    # dm.drop("Years",2024)
    # dm_waste.append(dm, "Variables")

    return dm


def waste_pack_aluminium(dm_eu):

    # note: Switzerland does not have facilities for the recycling of aluminium (https://alu-recycling.ch/fr/le-circuit-de-lalu/ and https://www.bafu.admin.ch/bafu/en/home/topics/waste/guide-to-waste-a-z/electrical-and-electronic-equipment.html)

    dm = dm_eu.filter({"Country": ["EU27"], "Variables": ["aluminium-pack"]})
    dm.rename_col("EU27", "Switzerland", "Country")

    dm[..., "export"] = 1
    dm[..., "waste-collected"] = 0
    dm[..., "waste-uncollected"] = 0
    dm[..., "littered"] = 0

    dm[..., "recycling"] = 0
    dm[..., "energy-recovery"] = 0
    dm[..., "reuse"] = 0
    dm[..., "incineration"] = 0
    dm[..., "landfill"] = 0

    return dm


def waste_pack_paper(dm_eu):

    # note: in theory the recyclable ones are paper pack and paper print, paper san goes all to water / not recyclable

    # https://spkf.ch/wp-content/uploads/2024/06/bk-240521-statistischer-Jahresbericht-RPK-2023.pdf

    dm = dm_eu.filter(
        {"Country": ["EU27"], "Variables": ["paper-pack", "paper-san", "paper-print"]}
    )
    dm.rename_col("EU27", "Switzerland", "Country")

    # figures for 2023 in tonnes
    # pack_collected_separated = 277930
    # pack_collected_mixedbags = 17005
    # print_collected_separated = 343685
    # print_collected_mixedbags = 21029
    # san_consumed = 149706
    # recycled_ch = 680790
    # export_waste_paper = 460357
    # energy_recovery = 40136
    # export_share = export_waste_paper/(pack_collected_separated + pack_collected_mixedbags + print_collected_separated + print_collected_mixedbags)

    # consumption = 1330989
    # export = 460357
    # dm[...,"export"] = export/consumption
    # dm[...,"littered"] = 0.07 # we assume same littered than plastics
    # dm[...,"waste-uncollected"] = 0.03 # RP+K reports recyclable paper still found in residual waste (hence incinerated, not littered) at 40,136 t, which is 3.02 % of total paper consumption (40,136 ÷ 1,330,989). That’s not littering, but it does quantify the mis-sorted share that misses separate collection
    # dm[...,"waste-collected"] = 1 - dm[...,"export"] - dm[...,"littered"] - dm[...,"waste-uncollected"]

    # collected = 1141147
    # recycling = 835333
    # dm[...,"recycling"] = recycling/collected
    # dm[...,"energy-recovery"] = 1 - dm[...,"recycling"]
    # dm[...,"landfill"] = 0
    # dm[...,"incineration"] = 0
    # dm[...,"reuse"] = 0

    # # assume same for paper pack and paper print, and fix paper san (we assume layer 1 same of others, and layer 2 all used for energy recovery)
    # dm[...,"paper-san","recycling"] = 0
    # dm[...,"paper-san","energy-recovery"] = 1

    consumption = 1_330_989  # total paper & cardboard generated
    collected_sep = 1_141_147  # separately collected for recycling
    export_rec = 460_357  # exported recovered paper (subset of collected_sep)
    imports_rec = 154_543  # recovered paper imports (not needed for this split)
    mills_input = (
        835_333  # domestic mills' recovered paper input (includes imports_rec)
    )
    hygiene = (
        149_706  # non-recyclable tissue/hygiene (typically ends in residual waste)
    )
    mis_sorted = 40_136  # recyclable paper left in residual waste (incinerated)

    export_share = export_rec / consumption
    littered_share = 0.07  # we assume same littered than plastics
    waste_uncollected_share = 0.0
    waste_collected_share = (
        1.0 - export_share - littered_share - waste_uncollected_share
    )
    waste_collected_share = max(
        0.0, min(1.0, waste_collected_share)
    )  # clamp for safety
    waste_collected_mass = waste_collected_share * consumption
    # We want "destined to recycling" to be less than 1.0.
    # Use Swiss stats: domestic recycling use ≈ separately collected minus exported recovered paper.
    # (Anything not sent to recycling ends up in residual MSW → energy recovery in Switzerland.)
    domestic_to_recycling_mass = max(0.0, collected_sep - export_rec)
    # Make sure we don't allocate more to recycling than what your Layer-1 says is "collected"
    domestic_to_recycling_mass = min(domestic_to_recycling_mass, waste_collected_mass)
    recycling_share = (
        0.0
        if waste_collected_mass == 0
        else domestic_to_recycling_mass / waste_collected_mass
    )
    recycling_share = max(0.0, min(1.0, recycling_share))  # clamp

    dm[..., "export"] = export_share
    dm[..., "littered"] = littered_share
    dm[..., "waste-uncollected"] = (
        waste_uncollected_share  # assuming in CH all waste is collected
    )
    dm[..., "waste-collected"] = waste_collected_share
    dm[..., "recycling"] = recycling_share
    dm[..., "incineration"] = 0.0
    dm[..., "landfill"] = 0.0
    dm[..., "energy-recovery"] = 1.0 - recycling_share
    dm[..., "reuse"] = 0.0

    # assume same for paper pack and paper print, and fix paper san (we assume layer 1 same of others, and layer 2 all used for energy recovery)
    dm[..., "paper-san", "recycling"] = 0
    dm[..., "paper-san", "energy-recovery"] = 1

    return dm


def run(years_ots):

    # directories
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # get waste management data EU
    dm_eu = get_pickle_eu(current_file_directory)

    # waste vehicles
    dm_eu_vehicles = dm_eu.filter({"Variables": ["vehicles"]})
    dm_waste = waste_vehicles(current_file_directory, dm_eu_vehicles, years_ots)

    # trains (same of EU)
    dm_waste_trains = dm_eu.filter({"Variables": ["trains"]})
    dm_waste_trains.rename_col("EU27", "Switzerland", "Country")
    dm_waste.append(dm_waste_trains, "Variables")

    # planes (same of EU)
    dm_waste_planes = dm_eu.filter({"Variables": ["planes"]})
    dm_waste_planes.rename_col("EU27", "Switzerland", "Country")
    dm_waste.append(dm_waste_planes, "Variables")

    # ships (all exported)
    dm_waste_ships = dm_eu.filter({"Variables": ["ships"]})
    dm_waste_ships.rename_col("EU27", "Switzerland", "Country")
    dm_waste_ships = waste_ships(dm_waste_ships)
    dm_waste.append(dm_waste_ships, "Variables")

    # buildings
    dm_eu_bld = dm_eu.filter({"Variables": ["floor-area-new-residential"]})
    dm_waste_buildings = waste_buildings(current_file_directory, dm_eu_bld, years_ots)
    dm_waste.append(dm_waste_buildings, "Variables")

    # roads (same of EU)
    dm_waste_road = dm_eu.filter({"Variables": ["road"]})
    dm_waste_road.rename_col("EU27", "Switzerland", "Country")
    dm_waste.append(dm_waste_road, "Variables")

    # rail (same of EU)
    dm_waste_rail = dm_eu.filter({"Variables": ["rail"]})
    dm_waste_rail.rename_col("EU27", "Switzerland", "Country")
    dm_waste.append(dm_waste_rail, "Variables")

    # trolley cables (same of EU)
    dm_waste_cables = dm_eu.filter({"Variables": ["trolley-cables"]})
    dm_waste_cables.rename_col("EU27", "Switzerland", "Country")
    dm_waste.append(dm_waste_cables, "Variables")

    # appliances and electronics
    dm_waste_ele, dm_waste_domapp = waste_appliances(dm_eu)
    dm_waste.append(dm_waste_ele, "Variables")
    dm_waste.append(dm_waste_domapp, "Variables")

    # packaging
    dm_waste_pack_glass = waste_pack_glass(dm_eu)
    dm_waste.append(dm_waste_pack_glass, "Variables")
    dm_waste_pack_plastic = waste_pack_plastic(dm_eu)
    dm_waste.append(dm_waste_pack_plastic, "Variables")
    dm_waste_pack_alu = waste_pack_aluminium(dm_eu)
    dm_waste.append(dm_waste_pack_alu, "Variables")
    dm_waste_pack_paper = waste_pack_paper(dm_eu)
    dm_waste.append(dm_waste_pack_paper, "Variables")

    # sort
    dm_waste.sort("Variables")

    return dm_waste


if __name__ == "__main__":

    years_ots = create_years_list(1990, 2023, 1)

    run(years_ots)


###########################################################################################
########################################## WASTE ##########################################
###########################################################################################

# import requests
# import re

# def get_organization_names(search_string):

#     url = 'https://ckan.opendata.swiss/api/3/action/organization_list'
#     response_structure = requests.get(url)
#     data_structure = response_structure.json()
#     org_list = data_structure["result"]
#     bool_idx = [bool(re.search(search_string,s)) for s in org_list]

#     return np.array(org_list)[bool_idx].tolist()

# def get_databases_names_by_organization(organization_id):

#     base_url = "https://ckan.opendata.swiss/api/3/action/organization_show"
#     url = f"{base_url}?id={organization_id}&include_datasets=True"
#     response_structure = requests.get(url)
#     data_structure = response_structure.json()

#     packages = data_structure["result"]["packages"]
#     return [packages[idx]["title"]["fr"] for idx in range(len(packages))]

# # get name of bazg
# get_organization_names("bafu")

# # get all databases of bazg
# mylist = get_databases_names_by_organization("bundesamt-fur-umwelt-bafu")
# np.array(mylist)[[bool(re.search("chet",m)) for m in mylist]]

# note: it seems nor here nor online there is a database (excel) on waste

# note: I will assume that in switzerland all incineration is going for energy recovery
# https://opendata.swiss/en/dataset/b4fa710a-136e-476c-aa71-58bb7f89bea9?


############################
##### TRUCKS AND BUSES #####
############################

# same of vehicles

################################
##### TRAINS AND METROTRAM #####
################################

##################
##### PLANES #####
##################

#################
##### SHIPS #####
#################

#####################
##### BUILDINGS #####
#####################


#################
##### ROADS #####
#################


################
##### RAIL #####
################

##########################
##### TROLLEY CABLES #####
##########################

###################################################
##### LARGER APPLIANCES, AND PC & ELECTRONICS #####
###################################################


##########################
##### GLASS PACKAGES #####
##########################


############################
##### PLASTIC PACKAGES #####
############################


##############################
##### ALUMINIUM PACKAGES #####
##############################


##########################
##### PAPER PACKAGES #####
##########################


# ################
# ##### SAVE #####
# ################

# dm_ots = dm_waste.filter({"Years" : years_ots})
# dm_fts = dm_waste.filter({"Years" : years_fts})
# DM_fts = {1: dm_fts.copy(), 2: dm_fts.copy(), 3: dm_fts.copy(), 4: dm_fts.copy()} # for now we set all levels to be the same
# DM = {"ots" : dm_ots,
#       "fts" : DM_fts}
# f = os.path.join(current_file_directory, '../data/datamatrix/lever_waste-management.pickle')
# with open(f, 'wb') as handle:
#     pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)
