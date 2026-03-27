import pandas as pd
import os


def get_data_lca(current_file_directory):

    # get data
    filepath = os.path.join(
        current_file_directory,
        "../data/psi/SPEED2ZERO. D.4.1.2. LCA data for calculator v.16.12.2024.xlsx",
    )
    df = pd.read_excel(filepath, sheet_name="Data")
    df.columns

    # rename
    df.rename(columns={"EPFL name": "product"}, inplace=True)

    # checks
    len(df["product"].unique())

    # fix product names
    product_rename_dict = {
        "smartphone": "phone",
        "desktop computer": "computer_desktop",
        "TV": "tv",
        "fridge": "fridge",
        "dishwasher": "dishwasher",
        "oven": "oven",
        "washing machine": "wmachine",
        "computer production, laptop": "computer_laptop",
        "battery electric vehicles": "LDV_BEV",
        "plug-in hybrid electric vehicles - diesel": "LDV_PHEV-diesel",
        "plug-in hybrid electric vehicles - gasoline": "LDV_PHEV-gasoline",
        "fuel-cell electric vehicles": "LDV_FCEV",
        "internal-combustion engine vehicles - diesel": "LDV_ICE-diesel",
        "internal-combustion engine vehicles - gasoline": "LDV_ICE-gasoline",
        "internal-combustion engine vehicles - gas": "LDV_ICE-gas",
        "heavy-duty vehicles (freight trucks) light (payload = 7.5 tons), battery electric vehicles": "HDVL_BEV",
        "heavy-duty vehicles (freight trucks) light (payload = 7.5 tons), plug-in hybrid electric vehicles - diesel": "HDVL_PHEV-diesel",
        "heavy-duty vehicles (freight trucks) light (payload = 7.5 tons), fuel-cell electric vehicles": "HDVL_FCEV",
        "heavy-duty vehicles (freight trucks) light (payload = 7.5 tons), internal-combustion engine vehcles - diesel": "HDVL_ICE-diesel",
        "heavy-duty vehicles (freight trucks) light (payload = 7.5 tons), internal-combustion engine vehcles - gas": "HDVL_ICE-gas",
        "heavy-duty vehicles (freight trucks) medium (payload = 18 tons), battery electric vehicles": "HDVM_BEV",
        "heavy-duty vehicles (freight trucks) medium (payload = 18 tons), plug-in hybrid electric vehicles - diesel": "HDVM_PHEV-diesel",
        "heavy-duty vehicles (freight trucks) medium (payload = 18 tons), fuel-cell electric vehicles": "HDVM_FCEV",
        "heavy-duty vehicles (freight trucks) medium (payload = 18 tons), internal-combustion engine vehcles - diesel": "HDVM_ICE-diesel",
        "heavy-duty vehicles (freight trucks) medium (payload = 18 tons), internal-combustion engine vehcles - gas": "HDVM_ICE-gas",
        "heavy-duty vehicles (freight trucks) high (payload = 32 tons), battery electric vehicles": "HDVH_BEV",
        "heavy-duty vehicles (freight trucks) high (payload = 32 tons), plug-in hybrid electric vehicles - diesel": "HDVH_PHEV-diesel",
        "heavy-duty vehicles (freight trucks) high (payload = 32 tons), fuel-cell electric vehicles": "HDVH_FCEV",
        "heavy-duty vehicles (freight trucks) high (payload = 32 tons), internal-combustion engine vehcles - diesel": "HDVH_ICE-diesel",
        "heavy-duty vehicles (freight trucks) high (payload = 32 tons), internal-combustion engine vehcles - gas": "HDVH_ICE-gas",
        "scooter, electric, 4-11kW, NMC battery": "2W_BEV",
        "scooter, gasoline, 4-11kW, EURO-5": "2W_ICE-gasoline",
        "passenger bus, battery electric - opportunity charging, LTO battery, 13m single deck urban bus": "bus_BEV",
        "passenger bus, diesel hybrid, 13m single deck urban bus, EURO-VI": "bus_PHEV-diesel",
        "passenger bus, fuel cell electric, 13m single deck urban bus": "bus_FCEV",
        "Passenger bus, diesel, 13m double deck urban bus, EURO-VI": "bus_ICE-diesel",
        "Passenger bus, compressed gas, 13m single deck urban bus, EURO-VI": "bus_ICE-gas",
        "goods wagon production": "trains_CEV",
        "tram production": "metrotram_mt",
        "aircraft production, belly-freight aircraft, long haul": "planes_ICE",
        "barge tanker production": "ships_ICE",
        "battery, lithium-ion battery for vehicles": "battery-lion_vehicles",
        "battery, lithium-ion battery for electronics": "battery-lion_electronics",
        "mastic asphalt production": "road",
        "railway track construction": "rail",
        "market for aluminium around steel bi-metal wire, 3.67mm external diameter": "trolley-cables",
        "photovoltaic slanted-roof installation, 3 kWp, CIS, laminated, integrated, on roof": "RES-solar-Pvroof_csi",
        "photovoltaic slanted-roof installation, 3kWp, a-Si, laminated, integrated, on roof": "RES-solar-Pvroof_asi",
        "collector field area construction, solar thermal parabolic trough, 50 MW": "RES-solar-csp",
        "wind turbine construction, 2MW, onshore": "RES-wind-onshore",
        "wind power plant construction, 2MW, offshore, fixed parts": "RES-wind-offshore_fixed",
        "wind power plant construction, 2MW, offshore, moving parts": "RES-wind-offshore_moving",
        "hydropower plant construction, reservoir": "RES-other-hydroelectric_reservoir",
        "hydropower plant construction, run-of-river": "RES-other-hydroelectric_runofriver",
        "wave energy converter platform production": "RES-other-marine",
        "electricity production, at hard coal-fired IGCC power plant": "fossil-coal",
        "gas turbine construction, 10MW electrical": "fossil-gas",
        "electricity production, oil": "fossil-oil",
        "electricity production, nuclear, boiling water reactor": "nuclear",
        "electricity production, deep geothermal": "RES-other-geothermal",
        "battery production, Li-ion, NMC811, rechargeable, prismatic": "battery-lion_general",
        "heat pump production, 30kW": "heat-pump",
        "HVAC ventilator": "HVAC-ventilator",
        "heat production, natural gas, at boiler fan burner low-NOx non-modulating <100kW": "boiler-gas_undefined",
        "gas boiler production": "boiler-gas",
        "solar collector system installation, Cu flat plate collector, one-family house, hot water": "solar-collector",
        "heat production, wood pellet, at furnace 9kW, state-of-the-art 2014": "stove_wood",
        "pipeline, supercritical carbon dioxide": "pipes_CO2",
        "carbon dioxide, captured at synthetic natural gas plant, post, 200km pipeline, storage 1000m": "deep-saline-formation_natural-gas-plant",
        "carbon dioxide, captured from hard coal-fired power plant, post, pipeline 200km, storage 1000m": "deep-saline-formation_coal-plant",
        "Desktop computer end of life": "computer-EOL",
        "Smartphone end of life, mechanical treatment": "phone-EOL",
        "Passenger car, glider end of life, shredding": "LDV-EOL_glider",
        "Passenger car, internal combustio engine end of life, shredding": "LDV-EOL_engine",
        "Passenger car, electric car power train end of life, manual dismantling": "LDV-EOL_electric-power-train",
        "Li-ion battery end of life, pyrometallurgical treatment": "battery-lion-EOL_pyromettalurgical",
        "Li-ion battery end of life, hydrometallurgical treatment": "battery-lion-EOL_hydromettalurgical",
    }

    for key in product_rename_dict.keys():
        df.loc[df["product"] == key, "product"] = product_rename_dict[key]

    df_full = df.copy()

    # TODO: as it will be for buildings, also for roads and infrastructure in general
    # run ecoinvent at material level and obtain
    # estimates in combination with material composition (rather than doing at product level)
    # probably you can apply this methodology for all products (so all materials in
    # rows and columns). This will also solve the problems for the products that ecoinvent
    # might not have, like dryer and freezer in domapp

    return df_full
