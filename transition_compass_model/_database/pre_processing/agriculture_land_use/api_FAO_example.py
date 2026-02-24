import faostat
# README on how to use faostat available at
# Step 1. Install faostat package
# https://pypi.org/project/faostat/

list_countries = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark',
                     'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia',
                     'Lithuania', 'Luxembourg', 'Malta', 'Netherlands (Kingdom of the)', 'Poland', 'Portugal', 'Romania', 'Slovakia',
                     'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom of Great Britain and Northern Ireland']

ld = faostat.list_datasets()
code = 'CBH'
pars = faostat.list_pars(code)
my_countries = [faostat.get_par(code, 'area')[c] for c in list_countries]
my_pars = {
    'area': my_countries,
    'element': faostat.get_par(code, 'elements')['Production Quantity'],
    'item': faostat.get_par(code, 'item')['Wheat and products']
}

df = faostat.get_data_df(code, pars=my_pars, strval=False)

print(df.head())