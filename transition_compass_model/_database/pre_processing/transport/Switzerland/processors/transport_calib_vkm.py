import numpy as np

from transport_demand_pipeline import run as demand_pkm_vkm_run
from _database.pre_processing.transport.Switzerland.get_data_functions.demand_pkm_vkm import (
    extract_EP2050_transport_vkm_demand,
)
import os
import pickle
from transition_compass_model.model.common.auxiliary_functions import create_years_list, load_pop
from passenger_fleet_pipeline import run as passenger_fleet_run
from passenger_renewal_rate_and_waste_pipeline import (
    run as passenger_ren_rate_waste_adj_run,
)


def get_ep2050_data(this_dir, years_ots):

    # VKM demand for LDV, 2W, bus by technology from EP2050
    # EP2050+_Detailergebnisse 2020-2060_Verkehrssektor_alle Szenarien_2022-04-12
    file_url = "https://www.bfe.admin.ch/bfe/de/home/politik/energieperspektiven-2050-plus.exturl.html/aHR0cHM6Ly9wdWJkYi5iZmUuYWRtaW4uY2gvZGUvcHVibGljYX/Rpb24vZG93bmxvYWQvMTA0NDE=.html"
    zip_name = os.path.join(this_dir, "../data/EP2050_sectors.zip")
    file_pickle = os.path.join(this_dir, "../data/tra_EP2050_vkm_demand_private.pickle")
    dm_vkm_private = extract_EP2050_transport_vkm_demand(
        file_url, zip_name, file_pickle
    )
    dm_vkm_private.filter({"Years": years_ots}, inplace=True)
    DM = {}
    DM["passenger-and-freight_road_EP-2050"] = dm_vkm_private.copy()

    return DM


def run(dm_vkm, dm_LDV_fleet_stock, dm_new_LDV_fleet, years_ots):

    # get dir
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # get ep 2050
    DM = get_ep2050_data(this_dir, years_ots)

    # !FIXME: add vkm check

    # Estimate LDV weighing factors
    dm_vkm_LDV_EP2050 = DM["passenger-and-freight_road_EP-2050"].filter(
        {"Categories1": ["LDV"]}
    )
    ## keep EP2050 calibration data only up to 2015
    dm_vkm_LDV_EP2050.filter({"Years": create_years_list(1990, 2015, 1)}, inplace=True)
    ## extract vkm LDV share by tech
    assert (
        dm_LDV_fleet_stock.col_labels["Categories2"]
        == dm_vkm_LDV_EP2050.col_labels["Categories2"]
    )
    dm_shares = dm_LDV_fleet_stock.normalise("Categories2", inplace=False)
    dm_new_LDV_fleet.filter(
        {"Categories2": dm_shares.col_labels["Categories2"]}, inplace=True
    )
    dm_shares.append(
        dm_new_LDV_fleet.normalise("Categories2", inplace=False), dim="Variables"
    )
    dm_shares.filter({"Country": ["Switzerland"]}, inplace=True)
    dm_shares.filter({"Years": create_years_list(1990, 2015, 1)}, inplace=True)
    dm_shares.append(
        dm_vkm_LDV_EP2050.normalise("Categories2", inplace=False), dim="Variables"
    )
    # Determine weighing factors as
    # vkm by tech (%) = fleet by tech (%) * w + new fleet by tech (%) * (1-w)
    # w = ( vkm by tech (%) - new fleet by tech (%) ) / (fleet by tech (%) - new fleet by tech (%) )
    arr = (
        dm_shares[:, :, "tra_vkm_demand_share", :, :]
        - dm_shares[:, :, "tra_passenger_new-vehicles_share", :, :]
    ) / (
        dm_shares[:, :, "tra_passenger_vehicle-fleet_share", :, :]
        - dm_shares[:, :, "tra_passenger_new-vehicles_share", :, :]
    )
    dm_shares.add(arr, dim="Variables", col_label="tra_vkm_weight", unit="%")
    # FIXME! when plotting the weights, one can see that the ICE diesel and gasoline weights match,
    #  but vary greatly between 0 and 90% from 1995 and 2015, and have a downward trend,
    #  which would forecast to w=0 for the fts. Meaning that vkm share in the future will match fleet share.
    #  Therefore, there is no great advantage to use the weights instead of a direct calibration of vkm at runtime

    # save
    f = os.path.join(this_dir, "../data/datamatrix/calibration_vkm.pickle")
    with open(f, "wb") as handle:
        pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return DM


if __name__ == "__main__":

    # get years ots
    years_ots = create_years_list(1990, 2023, 1)
    country_list = ["Switzerland", "Vaud"]

    dm_pop_ots = load_pop(country_list, years_list=years_ots)

    ## get vkm
    dm_pkm_cap, dm_pkm, dm_vkm = demand_pkm_vkm_run(dm_pop_ots, years_ots)

    ## Total fleet and new-fleet (for private only)
    print("Fleet - private and public")
    dm_private_fleet, dm_public_fleet = passenger_fleet_run(dm_pkm, years_ots)

    ## Renewal rate & Waste + adj fleet
    print("Renewal-rate & Waste + adj fleet")
    # ['passenger_private-fleet', 'passenger_public-fleet', 'passenger_renewal-rate', 'passenger_new-vehicles', 'passenger_waste-fleet']
    DM = passenger_ren_rate_waste_adj_run(dm_private_fleet, dm_public_fleet)
    dm_LDV_fleet_stock = DM["passenger_private-fleet"].filter({"Categories1": ["LDV"]})
    dm_new_LDV_fleet = DM["passenger_new-vehicles"].filter({"Categories1": ["LDV"]})

    ## run
    run(dm_vkm, dm_LDV_fleet_stock, dm_new_LDV_fleet, years_ots)
