import pickle
import pandas as pd
from transition_compass_model.model.common.data_matrix_class import DataMatrix
from transition_compass_model.model.common.auxiliary_functions import (
    add_dummy_country_to_DM,
    filter_DM,
    my_pickle_dump,
    create_years_list,
    linear_fitting,
)

# 1) Import DM_agriculture*
## 1.1) Load DM

data_file = "../../data/datamatrix/agriculture.pickle"
with open(data_file, "rb") as handle:
    DM_agriculture = pickle.load(handle)
with open(data_file, "rb") as handle:
    DM_CH = pickle.load(handle)
filter_DM(DM_CH, {"Country": ["Switzerland", "Vaud"]})

DM_temp = DM_CH.copy()

# 2) Import & format new data
## 2.1) Emission factors for poultry/manure
### 2.1.1.) Import data

file = "manure_fxa.csv"
df = pd.read_csv(file)

#### DM['fxa']['ef_liv_N2O-emission']['fxa_ef_liv_N2O-emission_ef']
#### Categories 1: ['abp-hens-egg', 'meat-poultry']

### 2.1.2.) Formatting to DM

df = df.pivot_table(
    index=["timescale", "geoscale"], columns="variables", values="value"
).reset_index()
df.rename(columns={"geoscale": "Country", "timescale": "Years"}, inplace=True)
df = df.filter(regex="fxa_ef_liv_N2O-emission_ef.*|Country|Years")
dm = DataMatrix.create_from_df(df, num_cat=2)
dm.change_unit(
    "fxa_ef_liv_N2O-emission_ef", old_unit="tN2O/tN", new_unit="N2O/N", factor=1
)
dm.rename_col("meat-oth-animal", "meat-oth-animals", dim="Categories1")

### 2.1.3.) Add Vaud

for i in ["Vaud", "EU27", "Germany"]:
    dm.add(dm["Switzerland", ...], "Country", i)

### 2.1.4.) Add FTS

years_fts = create_years_list(2025, 2050, 5)
linear_fitting(dm, years_fts)

### 2.1.3.) Overwrite DM

DM_agriculture["fxa"]["ef_liv_N2O-emission"] = dm

### 2.1.4.) Overwrite Pickle
print("hello")
f = "../../data/datamatrix/agriculture.pickle"
my_pickle_dump(DM_agriculture, f)
