
import pandas as pd
import numpy as np
import os
import pickle
from transition_compass_model.model.common.data_matrix_class import DataMatrix    
from transition_compass_model.model.common.auxiliary_functions import create_years_list
    
def get_official_emissions_data(this_dir, years_ots):

    # get data
    filepath = os.path.join(this_dir,"../data/Evolution_GHG_since_1990_2025-04.xlsx")
    def get_emissions_data(ghg):
        df = pd.read_excel(filepath, sheet_name=ghg)
        df.columns = df.iloc[3,:]
        df = df.iloc[list(range(12,22)) + [57,58],:]
        df = pd.melt(df, id_vars = ['Cat. (1)', np.nan], var_name='Years')
        df.columns = ["code","variable","Years","value"]
        df = df.loc[:,["variable", "Years","value"]]
        df["Years"] = df["Years"].astype(int)
        df["value"] = df["value"].astype(float)
        df["ghg"] = ghg
        return df
    df = pd.concat([get_emissions_data(ghg) for ghg in ["CO2","CH4","N2O"]])
    # note: for now calibration is done on Mt and not MtCO2eq

    # make dm
    df["ghg"] = df["variable"] + "_" + df["ghg"] + "[Mt]"
    df = df.pivot(index=["Years"], columns="ghg", values='value').reset_index()
    df["Country"] = "Switzerland"
    dm = DataMatrix.create_from_df(df, 1)
    for variable in dm.col_labels["Variables"]: dm.rename_col(variable, variable.lstrip(), "Variables")
    # dm.datamatrix_plot()

    # rename and group
    rename_dict = {'Buses' : "bus", 
                   'Cars' : "LDV", 'Heavy duty trucks' : "HDVH", 
                   'Light duty trucks' : "HDVL", 'Motorcycles' : "2W", 
                   'Domestic aviation (without military)' : "aviation-domestic", 
                   'Domestic navigation' : "marine-domestic", 'Railways' : "rail", 
                   'Road transportation' : "road-transport-total", 'Transport' : "transport-total", 
                   'International aviation' : "aviation-international", 'International navigation': "marine-international"}
    for key in rename_dict.keys(): dm.rename_col(key, rename_dict[key], "Variables")
    dm.groupby({"aviation" : ["aviation-domestic", "aviation-international"], 
                "marine" : ["marine-domestic","marine-international"]}, "Variables", inplace=True)
    dm.drop("Variables", ['road-transport-total', 'transport-total'])

    # add missing years as nan
    years = list(range(1990,2023+1))
    missing = np.array(years)[[y not in dm.col_labels["Years"] for y in years]].tolist()
    dm.add(np.nan, "Years", missing, dummy=True)
    dm.sort("Years")

    # format and save
    for v in dm.col_labels["Variables"]: dm.rename_col(v, 'calib-emissions_' + v, "Variables")
    dm.deepen(based_on="Variables")
    dm.groupby({"trucks" : ['HDVH', 'HDVL']}, "Categories2", inplace=True)
    dm.sort("Categories2")
    dm.switch_categories_order("Categories1","Categories2")
    DM = {}
    DM["freight-and-passenger_emission-balance"] = dm.copy()
    
    # save
    f = os.path.join(this_dir, '../data/datamatrix/calibration_emissions.pickle')
    with open(f, 'wb') as handle:
        pickle.dump(DM, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return DM


def run(years_ots):
    
    # get dir
    this_dir = os.path.dirname(os.path.abspath(__file__))
    
    # get ep 2050
    DM = get_official_emissions_data(this_dir, years_ots)
    
    return DM

if __name__ == "__main__":
    
    # get years ots
    years_ots = create_years_list(1990, 2023, 1)
    
    # run
    run(years_ots)

