
from model.common.data_matrix_class import DataMatrix
import os
import numpy as np
import pickle
from model.common.auxiliary_functions import my_pickle_dump

def tra_industry_interface(
  dm_freight_veh, dm_passenger_veh, dm_infrastructure, write_pickle=False
):
  if "aviation" not in dm_passenger_veh.col_labels["Categories1"]:
    dm_passenger_veh.add(
      np.nan, dim="Categories1", col_label="aviation", dummy=True
    )
    dm_passenger_veh.add(np.nan, dim="Categories2", col_label="ICE", dummy=True)
  dm_veh = dm_passenger_veh.copy()
  dm_veh.groupby({"CEV": ["mt", "CEV"]}, "Categories2", inplace=True)
  dm_veh.groupby({"trains": ["metrotram", "rail"]}, "Categories1", inplace=True)
  dm_veh = dm_veh.filter_w_regex({"Categories1": "LDV|aviation|bus|trains"})
  dm_veh.rename_col(["aviation"], ["planes"], "Categories1")
  dm_veh.rename_col(
    [
      "tra_passenger_new-vehicles",
      "tra_passenger_vehicle-waste",
      "tra_passenger_vehicle-fleet",
    ],
    ["tra_product-demand", "tra_product-waste", "tra_product-stock"],
    dim="Variables",
  )

  # freight
  dm_fre = dm_freight_veh.copy()
  dm_fre.groupby({"HDV": "HDV"}, "Categories1", regex=True, inplace=True)
  dm_fre = dm_fre.filter_w_regex({"Categories1": "HDV|aviation|marine|rail"})
  dm_fre.rename_col("marine", "ships", "Categories1")
  dm_fre.rename_col(
    [
      "tra_freight_new-vehicles",
      "tra_freight_vehicle-waste",
      "tra_freight_vehicle-fleet",
    ],
    ["tra_product-demand", "tra_product-waste", "tra_product-stock"],
    dim="Variables",
  )

  # put together
  cat_missing = list(
    set(dm_veh.col_labels["Categories2"]) - set(
      dm_fre.col_labels["Categories2"])
  )
  dm_veh.add(0, dummy=True, dim="Categories2", col_label=["ICE"])
  dm_veh.append(dm_fre, "Categories1")
  # Rename kerosene and H2 as ICE
  dm_veh.rename_col('ICE', 'ICE_old', 'Categories2')
  dm_veh.groupby({'ICE': ['ICE_old', 'H2', 'kerosene']}, dim='Categories2',
                 inplace=True)
  dm_veh.groupby(
    {"trains": ["trains", "rail"], "planes": ["planes", "aviation"]},
    "Categories1",
    inplace=True,
  )
  dm_veh.sort("Categories1")

  # get infrastructure
  dm_infra_ind = dm_infrastructure.copy()
  dm_infra_ind.rename_col_regex("infra-", "", dim="Categories1")
  dm_infra_ind.rename_col(
    ['tra_infrastructure_waste', "tra_new_infrastructure",
     'tra_tot-infrastructure'],
    ["tra_product-waste", "tra_product-demand", 'tra_product-stock'],
    dim="Variables"
  )

  # fix years in dm_veh
  # TODO: to remove this when we fix it in pre processing
  years = dm_veh.col_labels["Years"].copy()
  for y in years:
    arr_temp = dm_veh[:, y, ...]
    dm_veh.drop("Years", int(y))
    dm_veh.add(arr_temp, "Years", [int(y)])
  dm_veh.sort("Years")

  # ! FIXME add infrastructure in km
  DM_industry = {
    "tra-veh": dm_veh.filter({"Variables": ["tra_product-demand"]}),
    "tra-infra": dm_infra_ind,
    "tra-waste": dm_veh.filter({"Variables": ["tra_product-waste"]}),
    "tra-stock": dm_veh.filter({"Variables": ["tra_product-stock"]}),
  }

  # if write_pickle is True, write pickle
  if write_pickle is True:
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(
      current_file_directory,
      "../../_database/data/interface/transport_to_industry.pickle",
    )
    with open(f, "wb") as handle:
      pickle.dump(DM_industry, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return DM_industry


def tra_minerals_interface(
  dm_freight_new_veh,
  dm_passenger_new_veh,
  DM_industry,
  dm_infrastructure,
  write_pickle=False,
):
  # Group technologies as PHEV, ICE, EV and FCEV
  dm_freight_new_veh.groupby(
    {"PHEV": "PHEV.*", "ICE": "ICE.*", "EV": "BEV|CEV"},
    regex=True,
    inplace=True,
    dim="Categories2",
  )
  # note that mt is later dropped
  dm_passenger_new_veh.groupby(
    {"PHEV": "PHEV.*", "ICE": "ICE.*", "EV": "BEV|CEV|mt"},
    regex=True,
    inplace=True,
    dim="Categories2",
  )
  # keep only certain vehicles
  keep_veh = "HDV.*|2W|LDV|bus"
  dm_keep_new_veh = dm_passenger_new_veh.filter_w_regex(
    {"Categories1": keep_veh})
  dm_keep_new_veh.rename_col(
    "tra_new-vehicles", "tra_product-demand", dim="Variables"
  )
  dm_keep_freight_new_veh = dm_freight_new_veh.filter_w_regex(
    {"Categories1": keep_veh}
  )
  dm_keep_freight_new_veh.rename_col(
    "tra_new-vehicles", "tra_product-demand", dim="Variables"
  )
  # join passenger and freight

  dm_keep_new_veh.append(dm_keep_freight_new_veh, dim="Categories1")
  # flatten to obtain e.g. LDV-EV or HDVL-FCEV
  dm_keep_new_veh = dm_keep_new_veh.flatten()
  dm_keep_new_veh.rename_col_regex("_", "-", "Categories1")

  dm_other = DM_industry["tra-veh"].filter(
    {"Categories1": ["planes", "ships", "trains"]}
  )
  dm_other.groupby(
    {
      "other-planes": ["planes"],
      "other-ships": ["ships"],
      "other-trains": ["trains"],
    },
    dim="Categories1",
    inplace=True,
  )

  dm_keep_new_veh.append(dm_other, dim="Categories1")
  dm_keep_new_veh.rename_col("tra_product-demand", "product-demand",
                             dim="Variables")

  DM_minerals = {"tra_veh": dm_keep_new_veh, "tra_infra": dm_infrastructure}

  # if write_pickle is True, write pickle
  if write_pickle is True:
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(
      current_file_directory,
      "../../_database/data/interface/transport_to_minerals.pickle",
    )
    with open(f, "wb") as handle:
      pickle.dump(DM_minerals, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return DM_minerals


def tra_oilrefinery_interface(dm_pass_energy, dm_freight_energy,
                              write_pickle=False):
  cat_missing = ["kerosene", "marinefueloil"]
  for cat in cat_missing:
    if cat not in dm_pass_energy.col_labels["Categories1"]:
      dm_pass_energy.add(0, dummy=True, col_label=cat, dim="Categories1")
  dm_pass_energy.append(dm_freight_energy, dim="Variables")
  dm_tot_energy = dm_pass_energy.groupby(
    {"tra_energy-demand": ".*"}, dim="Variables", inplace=False, regex=True
  )
  dict_rename = {
    "diesel": "liquid-ff-diesel",
    "marinefueloil": "liquid-ff-fuel-oil",
    "gasoline": "liquid-ff-gasoline",
    "gas": "gas-ff-natural",
    "kerosene": "liquid-ff-kerosene",
  }
  for str_old, str_new in dict_rename.items():
    dm_tot_energy.rename_col(str_old, str_new, dim="Categories1")

  # if write_pickle is True, write pickle
  if write_pickle is True:
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(
      current_file_directory,
      "../../_database/data/interface/transport_to_oil-refinery.pickle",
    )
    with open(f, "wb") as handle:
      pickle.dump(dm_tot_energy, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return dm_tot_energy


def prepare_TPE_output(DM_passenger_out, DM_freight_out):
  # Aviation Energy-demand
  dm_keep_aviation_energy = DM_passenger_out["aviation"]["energy"]
  dm_keep_aviation_energy.groupby(
    {"SAF": "kerosenebio|keroseneefuel"},
    dim="Categories2",
    regex=True,
    inplace=True,
  )
  dm_keep_aviation_energy.filter(
    {"Categories2": ["kerosene", "SAF", "hydrogen", "electricity"]},
    inplace=True
  )

  dm_keep_aviation_emissions = DM_passenger_out["aviation"]["emissions"].copy()
  dm_keep_aviation_local = dm_keep_aviation_emissions.group_all(
    "Categories2", inplace=False
  )
  dm_keep_aviation_local.group_all("Categories2", inplace=True)
  dm_keep_aviation_local.append(
    DM_passenger_out["aviation-share-local"], dim="Variables"
  )
  dm_keep_aviation_local.operation(
    "tra_passenger_emissions",
    "*",
    "tra_share-emissions-local",
    out_col="tra_passenger-emissions-local",
    unit="Mt",
  )
  dm_keep_aviation_local.rename_col(
    "tra_passenger_emissions", "tra_passenger-emissions-total", dim="Variables"
  )
  dm_keep_aviation_local.drop(dim="Variables",
                              col_label="tra_share-emissions-local")

  dm_keep_mode = DM_passenger_out["mode"].filter(
    {
      "Variables": [
        "tra_passenger_transport-demand-by-mode",
        "tra_passenger_energy-demand-by-mode",
        "tra_passenger_vehicle-fleet",
        "tra_passenger_new-vehicles",
        "tra_passenger_transport-demand-vkm",
      ]
    }
  )
  dm_keep_mode.change_unit('tra_passenger_transport-demand-by-mode', old_unit='pkm', new_unit='Bpkm', factor=1e-9)
  dm_keep_mode.change_unit('tra_passenger_transport-demand-vkm', old_unit='vkm', new_unit='Bvkm', factor=1e-9)
  dm_keep_mode.change_unit('tra_passenger_vehicle-fleet', old_unit='number', new_unit='millions', factor=1e-6)



  dm_keep_tech = DM_passenger_out["tech"].filter(
    {"Variables": ["tra_passenger_vehicle-fleet"], "Categories1": ["LDV"]}
  )

  dm_keep_fuel = DM_passenger_out["fuel"]

  dm_keep_energy = DM_passenger_out["energy"].copy()
  dm_keep_energy.drop(dim="Categories1", col_label=["efuel"])

  dm_freight_energy_by_mode = DM_freight_out["mode"].filter(
    {"Variables": ["tra_freight_energy-demand"]}
  )
  dm_freight_energy_by_mode.rename_col(
    "tra_freight_energy-demand",
    "tra_freight_energy-demand-by-mode",
    dim="Variables",
  )
  dm_freight_energy_by_mode.groupby(
    {"HDV": "HDV.*"}, dim="Categories1", inplace=True, regex=True
  )

  dm_freight_energy_by_fuel = DM_freight_out["energy"].copy()
  dm_freight_energy_by_fuel.drop(dim="Categories1",
                                 col_label=["efuel", "ejetfuel"])
  dm_freight_energy_by_fuel.rename_col(
    "tra_freight_total-energy", "tra_freight_energy-demand-by-fuel",
    dim="Variables"
  )

  # Total energy demand
  dm_energy_tot_pass = DM_passenger_out["energy"].copy()
  dm_energy_tot_pass.group_all(dim="Categories1")
  dm_energy_tot = dm_energy_tot_pass.copy()
  dm_energy_freight = DM_freight_out["energy"].copy()
  dm_energy_freight.group_all(dim="Categories1")
  dm_energy_tot.append(dm_energy_freight, dim="Variables")
  dm_energy_tot.groupby(
    {"tra_energy-demand_total": ".*"}, inplace=True, regex=True, dim="Variables"
  )

  dm_tech_HDVH = DM_freight_out['tech'].filter({'Variables': ['tra_freight_technology-share-fleet'], 'Categories1': ['HDVH']})
  dm_tech_HDVH.array = dm_tech_HDVH.array*100

  dm_freight_emissions = DM_freight_out['emissions'].filter({'Categories2': ['CO2']})
  dm_freight_emissions.groupby({'HDV': ['HDVH', 'HDVM', 'HDVL']}, dim='Categories1', inplace=True)

  dm_soft_mobility = DM_passenger_out["soft-mobility"]
  dm_soft_mobility.change_unit("tra_passenger_transport-demand-by-mode", old_unit='pkm', new_unit='Bpkm', factor=1e-9)

  # Merge datamatrices for new-app
  dm_tpe = dm_keep_mode.flattest()
  dm_tpe.append(dm_keep_tech.flattest(), dim="Variables")
  dm_tpe.append(dm_keep_fuel.flattest(), dim="Variables")
  dm_tpe.append(dm_keep_energy.flattest(), dim="Variables")
  dm_tpe.append(dm_freight_energy_by_mode.flattest(), dim="Variables")
  dm_tpe.append(dm_energy_tot.flattest(), dim="Variables")
  dm_tpe.append(dm_freight_energy_by_fuel.flattest(), dim="Variables")
  dm_tpe.append(DM_passenger_out["soft-mobility"].flattest(), dim="Variables")
  dm_tpe.append(DM_passenger_out["emissions"].flattest(), dim="Variables")
  dm_tpe.append(dm_keep_aviation_emissions.flattest(), dim='Variables')
  dm_tpe.append(dm_keep_aviation_local.flattest(), dim='Variables')
  dm_tpe.append(dm_keep_aviation_energy.flattest(), dim='Variables')
  dm_tpe.append(dm_tech_HDVH.flattest(), dim='Variables')
  dm_tpe.append(dm_freight_emissions.flattest(), dim='Variables')

  KPI = []
  yr = 2050
  # Emissions Passenger
  dm_pass_emi = DM_passenger_out["emissions"].copy()
  dm_pass_emi.drop(col_label='aviation', dim='Categories1')
  dm_pass_emi.group_all('Categories1', inplace=True)
  value = dm_pass_emi[0, yr, 'tra_emissions-CO2e_passenger']
  KPI.append({'title': 'Passenger land transport CO2', 'value': value, 'unit': 'Mt'})

  # Aviation CO2
  dm_avia_emi = dm_keep_aviation_emissions.group_all('Categories2', inplace=False)
  value = dm_avia_emi[0, yr, 'tra_passenger_emissions', 'aviation', 'CO2']
  KPI.append({'title': 'Passenger aviation CO2', 'value': value, 'unit': 'Mt'})

  # Energy demand Passenger in TWh
  value = dm_energy_tot_pass[0, yr, 'tra_passenger_energy-demand-by-fuel']
  KPI.append(
    {'title': 'Passenger energy demand', 'value': value, 'unit': 'TWh'})

  # Energy demand Freight in TWh
  value = dm_energy_freight[0, yr, 'tra_freight_total-energy']
  KPI.append(
    {'title': 'Freight energy demand', 'value': value, 'unit': 'TWh'})

  # Share of public transport

  # % EV cars
  dm_LDV_EV = DM_passenger_out['tech'].filter({'Variables': ['tra_passenger_technology-share-fleet'], 'Categories1': ['LDV']}, inplace=False)
  value = dm_LDV_EV[0, yr, 'tra_passenger_technology-share-fleet', 'LDV', 'BEV']*100
  KPI.append({'title': 'Electric car share', 'value': value, 'unit': '%'})

  # % Non fossil fuel trucks
  dm_tech_HDVH.groupby({'Non-ICE': ['BEV', 'FCEV', 'CEV', 'PHEV-diesel', 'PHEV-gasoline']}, dim='Categories2', inplace=True)
  value = dm_tech_HDVH[0, yr, 'tra_freight_technology-share-fleet', 'HDVH', 'Non-ICE']
  KPI.append(
    {'title': 'Low emission truck share', 'value': value, 'unit': '%'})


  # Freight emissions
  return dm_tpe, KPI


def tra_emissions_interface(
    dm_pass_emissions, dm_freight_emissions, write_pickle=False
):

    dm_pass_emissions.rename_col(
        "tra_passenger_emissions", "tra_emissions_passenger", dim="Variables"
    )
    dm_pass_emissions = dm_pass_emissions.flatten().flatten()
    dm_freight_emissions.rename_col(
        "tra_freight_emissions", "tra_emissions_freight", dim="Variables"
    )
    dm_freight_emissions = dm_freight_emissions.flatten().flatten()

    dm_pass_emissions.append(dm_freight_emissions, dim="Variables")

    # if write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../../_database/data/interface/transport_to_emissions.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(dm_pass_emissions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm_pass_emissions


def tra_agriculture_interface(
    dm_freight_agriculture, dm_passenger_agriculture, write_pickle=False
):

    # !FIXME: of all of the bio-energy demand, only the biogas one is accounted for in Agriculture
    dm_agriculture = dm_freight_agriculture
    dm_agriculture.array = dm_agriculture.array + dm_passenger_agriculture.array
    dm_agriculture.rename_col(
        "tra_freight_total-energy", "tra_bioenergy", dim="Variables"
    )
    dm_agriculture.rename_col("biogas", "gas", dim="Categories1")

    # if write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../../_database/data/interface/transport_to_agriculture.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(dm_agriculture, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dm_agriculture


def tra_power_interface(DM_passenger_power, DM_freight_power, write_pickle=False):

    DM_power = DM_passenger_power
    DM_power["hydrogen"].array = (
        DM_power["hydrogen"].array + DM_freight_power["hydrogen"].array
    )
    DM_power["electricity"].add(
        0, dim="Variables", dummy=True, col_label="tra_power-demand_other", unit="GWh"
    )
    DM_power["electricity"].sort("Variables")
    DM_freight_power["electricity"].add(
        0,
        dim="Variables",
        dummy=True,
        col_label="tra_power-demand_aviation",
        unit="GWh",
    )
    DM_freight_power["electricity"].sort("Variables")
    DM_power["electricity"].array = (
        DM_power["electricity"].array + DM_freight_power["electricity"].array
    )

    # if write_pickle is True, write pickle
    if write_pickle is True:
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        f = os.path.join(
            current_file_directory,
            "../../_database/data/interface/transport_to_power.pickle",
        )
        with open(f, "wb") as handle:
            pickle.dump(DM_power, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return DM_power


def tra_energy_interface(DM_passenger_energy, DM_freight_energy,
                         write_pickle=False):
  DM_energy = {'freight': DM_freight_energy, 'passenger': DM_passenger_energy}
  # if write_pickle is True, write pickle
  if write_pickle is True:
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(current_file_directory,
                     '../../_database/data/interface/transport_to_energy.pickle')
    my_pickle_dump(DM_energy, f)

  return DM_energy
