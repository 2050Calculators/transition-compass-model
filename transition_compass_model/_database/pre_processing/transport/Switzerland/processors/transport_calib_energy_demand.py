
import os
import pickle
import warnings
warnings.simplefilter("ignore")
from _database.pre_processing.buildings.Switzerland.get_data_functions import energy_demand_for_calibration as get_data
# from _database.pre_processing.transport.Switzerland.get_data_functions import transport_energy as get_data_ep2050
# from _database.pre_processing.transport.Switzerland.processors.passenger_energy_pipeline import run as get_data_ep2050
import _database.pre_processing.transport.Switzerland.get_data_functions.transport_energy as get_data_ep2050
from transition_compass_model.model.common.auxiliary_functions import create_years_list
    
#####################
##### FUNCTIONS #####
#####################
    
def official_energy_stats(this_dir, years_ots):
    
    # Extract energy demand of households (incl. hot-water, heating, lighting, appliances)
    file_url = 'https://www.bfe.admin.ch/bfe/fr/home/versorgung/statistik-und-geodaten/energiestatistiken/gesamtenergiestatistik.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZnIvcHVibGljYX/Rpb24vZG93bmxvYWQvNzUxOQ==.html'
    local_filename =  os.path.join(this_dir, '../data/statistique_globale_suisse_energie.xlsx')
    parameters = dict()
    mapping = {
        # liquids (subcolumns, not the overall "Total Treibstoffe")
        "gasoline": r".*Essence.*",
        "diesel": r".*Carburant diesel.*",
        "kerosene": r".*Carburants d'aviation.*",

        # electricity (total + optional splits)
        "electricity": r".*Total Electricité.*",
        "electricity-rail": r".*chemins de fer.*",
        "electricity-road": r".*routier.*",
        "electricity-non-road": r".*non routier.*",

        # gas
        "gas-pipeline": r".*Gaz transport par conduites.*",
        "gas-other": r".*Gaz autres transport.*",

        # other
        "coal": r".*Charbon.*",
        "renewables-other": r".*Autres énergies renouvelables.*",
        # "total": r".*Total\s*=\s*100%.*",
    }
    # renewables include: soleil, énergie éolienne, biogaz, chaleur ambiante.
    # It is weird because we also have electricity, that includes wind,
    # and ok maybe solar here is not PV but solar heat, but wind ??
    # It would be useful to know biogaz as a separate
    parameters["mapping"] = mapping
    parameters["var name"] = "tra_energy-demand_by-fuel"   # change to whatever your model expects
    parameters["headers indexes"] = (8, 9, 10)             # likely same pattern as your T17a extraction
    parameters["first row"] = 11                           # likely same as T17a
    parameters["unit"] = None
    parameters["cols to drop"] = r".*%.*|.*extra.*"         # keep TJ columns, drop % columns

    dm = get_data.extract_energy_statistics_data(
        file_url,
        local_filename,
        sheet_name="T17e",
        parameters=parameters,
        years_ots=years_ots,
    )

    # note: for some reason it seems that DataMatrix is imported twice and this is a problem
    from model.common.data_matrix_class import DataMatrix
    df = dm.write_df()  # or to_dataframe/as_df
    dm = DataMatrix.create_from_df(df, num_cat=1)

    # drop coal and gas in pipes
    dm.drop("Categories1", "coal")
    dm.drop("Categories1", "gas-pipeline")

    # convert to TWh
    dm.change_unit("tra_energy-demand_by-fuel", 1/3600, "TJ", "TWh")

    # get data with aggregate electricity
    dm_temp = dm.filter({"Categories1" : ['diesel', 'electricity', 'gas-other', 'gasoline', 'kerosene', 'renewables-other']})
    dm_temp.rename_col("gas-other", "gas", "Categories1")
    dm_temp.sort("Categories1")
    DM = {}
    DM["freight-and-passenger_energy-balance"] = dm_temp.copy()

    # get data on electricity split
    dm_temp = dm.filter({"Categories1" : ['electricity-non-road', 'electricity-rail', 'electricity-road']})
    dm_temp.sort("Categories1")
    dm_temp.change_unit('tra_energy-demand_by-fuel', old_unit='TWh', new_unit='MJ', factor=3.6e9, operator='*')
    DM["freight-and-passenger_electricity-split_energy-balance"] = dm_temp.copy()

    
    return DM

def ep2050_energy_stats(DM, this_dir, years_ots):
    
    # Energy demand for LDV, 2W, bus by technology from EP2050
    # EP2050+_Detailergebnisse 2020-2060_Verkehrssektor_alle Szenarien_2022-04-12
    # file_url = 'https://www.bfe.admin.ch/bfe/de/home/politik/energieperspektiven-2050-plus.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvMTA0NDE=.html'
    # zip_name = os.path.join(this_dir, '../data/EP2050_sectors.zip')
    file_pickle = os.path.join(this_dir, '../data/tra_EP2050_energy_demand_private.pickle')
    # dm_energy_private = get_data_ep2050.extract_EP2050_transport_energy_demand(file_url, zip_name, file_pickle)
    with open(file_pickle, 'rb') as handle: dm_energy_private = pickle.load(handle)
    dm_energy_private.change_unit('tra_energy_demand', old_unit='PJ', new_unit='MJ', factor=1e9, operator='*')
    dm_energy_private.filter({"Years" : years_ots}, inplace=True)
    dm_energy_private.groupby({"ICE-gasoline" : ['biogasoline', 'gasoline']}, "Categories2", inplace=True)

    # Energy demand for rail from EP2050
    # EP2050+_TechnsicherBericht_DatenAbbildungen_Kap 8_2022-04-12, table Abb. 112
    file_url = 'https://www.bfe.admin.ch/bfe/de/home/politik/energieperspektiven-2050-plus.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvMTA5MDQ=.html'
    zip_name = os.path.join(this_dir, '../data/EP2050_tables.zip')
    file_pickle = os.path.join(this_dir, '../data/tra_EP2050_energy_demand_rail.pickle')
    dm_energy_rail = get_data_ep2050.extract_EP2050_transport_energy_demand_rail(file_url, zip_name, file_pickle)
    dm_energy_rail.filter({"Years" : years_ots}, inplace=True)
    dm_energy_rail.change_unit('tra_energy_demand', old_unit='TWh', new_unit='MJ', factor=3.6e9,operator='*')

    with open(file_pickle, 'rb') as handle: dm_energy_rail_aviation = pickle.load(handle)
    dm_aviation = dm_energy_rail_aviation.filter({"Variables" : ['tra_energy_demand_aviation (int.)', 'tra_energy_demand_aviation (nat.)']})
    dm_aviation.groupby({"tra-energy-demand_aviation" : ['tra_energy_demand_aviation (int.)', 'tra_energy_demand_aviation (nat.)']},
                        "Variables", inplace=True)
    dm_aviation.change_unit('tra-energy-demand_aviation', old_unit='PJ', new_unit='MJ', factor=1e9, operator='*')
    dm_aviation.filter({"Years" : years_ots}, inplace=True)

    # get freight
    dm_freight = dm_energy_private.filter({"Categories1" : ['HDVH', 'HDVL']})
    dm_freight = dm_freight.flatten()
    dm_freight_rail = dm_energy_rail.filter({"Categories2" : ["freight"]})
    dm_freight_rail.rename_col("freight", "CEV", "Categories2")
    dm_freight_rail = dm_freight_rail.flatten()
    dm_freight.append(dm_freight_rail, "Categories1")
    dm_freight.deepen(based_on="Categories1")
    DM["freight_EP-2050"] = dm_freight.copy()

    # get passenger
    dm_passenger = dm_energy_private.filter({"Categories1" : ['2W','LDV','bus']})
    dm_passenger = dm_passenger.flatten()
    dm_passenger_rail = dm_energy_rail.filter({"Categories2" : ["passenger"]})
    dm_passenger_rail.rename_col("passenger", "CEV", "Categories2")
    dm_passenger_rail = dm_passenger_rail.flatten()
    dm_passenger.append(dm_passenger_rail, "Categories1")
    dm_passenger.deepen(based_on="Categories1")
    DM["passenger_EP-2050"] = dm_passenger.copy()

    # get aviation
    dm_aviation.rename_col("tra-energy-demand_aviation","tra-energy-demand_aviation_kerosene","Variables")
    dm_aviation.deepen_twice()
    DM["freight-and-passenger_aviation_EP-2050"] = dm_aviation.copy()
    
    return DM


def run(years_ots):
    
    # get dir
    this_dir = os.path.dirname(os.path.abspath(__file__))
    
    # get official statistics data on energy demand
    DM = official_energy_stats(this_dir, years_ots)
    
    # get ep 2050 data on energy demand
    DM = ep2050_energy_stats(DM, this_dir, years_ots)
    
    # save
    f = os.path.join(this_dir, '../data/datamatrix/calibration_energy-demand.pickle')
    with open(f, 'wb') as handle:
        pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return DM

if __name__ == "__main__":
    
    # get years ots
    years_ots = create_years_list(1990, 2023, 1)
    
    # run
    run(years_ots)