#########################################################################
#
#  Preprocessing of data to feed to EnergyScope solver
#
#  We take data from energyscope-MILP/ses_main.dat (git repository
#  https://github.com/stefanomoret/energyscope-MILP) and we put them in
#  a datamatrix format. Some of these data will be overwritten by input
#  from other modules. ses_main.dat should be saved in ./data/.
#
##########################################################################
import numpy as np
import re
from model.common.data_matrix_class import DataMatrix
from model.common.auxiliary_functions import create_years_list
import pickle

def read_ampl_dat_file(filename):
    with open(filename, "r") as file:
        dat_content = file.read()
    return dat_content


def create_dm_from_scalar_dat(scalar_pattern, country, years_fts):
    var_names, values = zip(*scalar_pattern)
    col_labels = {'Country': [country], 'Years': years_fts, 'Variables': list(var_names)}
    units = {var: '-' for var in var_names}
    dm = DataMatrix(col_labels=col_labels, units=units)
    dm.array = np.zeros(tuple(len(lst) for lst in col_labels.values()))
    dm.array[0, :, :] = np.array(values, dtype=float)
    return dm


def read_table_blocks(filename):
    with open(filename, 'r') as file:
        block = []
        for line in file:
            line = line.strip()
            if line == '':
                if block:
                    # Check if the block seems like a table (contains a colon and list)
                    if any(re.search(r':(?!=)', l) and '\t' in l and 'param' in l for l in block):  # Likely a table block
                        yield [l for l in block if not l.startswith('#')]  # Exclude lines starting with '#'
                block = []
            else:
                block.append(line)
        if block:
            if any(re.search(r':(?!=)', l) and '\t' in l and 'param' in l for l in block): # Likely a table block
                yield [l for l in block if not l.startswith('#')]  # Exclude lines starting with '#'


def get_parameters(filename, country, years_fts):
    dat_content = read_ampl_dat_file(filename)
    # Match single-line scalar parameters
    scalar_pattern = re.findall(r"param (\w+)\s*:=\s*([\d.]+)\s*;", dat_content)
    dm_param = create_dm_from_scalar_dat(scalar_pattern, country, years_fts)

    # Match tables (column-based)
    return dm_param


def extract_tables(filename):
    table_dict = {}
    i = 0
    for block in read_table_blocks(filename):
        # Assuming the block has the param name and columns in the first line
        header = block[0]
        try:
            param_name = header.split(":")[0].split()[1]

        except IndexError:
            param_name = "index"+str(i)
            i = i + 1
            pass

        columns = header.split(":")[1].strip().split("\t")
        # Then we parse the remaining lines (excluding the last semicolon line)
        rows = block[1:-1]  # Skip the first header line and the last line with ';'

        data_dict = {}
        indexes = []
        for row in rows:
            row_values = row.split("\t")
            index = row_values[0]
            indexes.append(index)
            data_dict[index] = {col: float(val) for col, val in zip(columns, row_values[1:])}

        table_dict[param_name] = {'data': data_dict, 'columns': columns, 'rows': indexes}

    return table_dict


def convert_table_dict_to_DM(table_dict, country, years_fts):
    DM = {}

    for var in table_dict.keys():
        if "index" in var:
            variables = table_dict[var]['columns']
            categories1 = table_dict[var]['rows']
            col_labels = {'Country': [country], 'Years': years_fts, 'Variables': variables, 'Categories1': categories1}
            units = {var: '-' for var in variables}
            dm = DataMatrix(col_labels=col_labels, units=units)
            dm.array = np.zeros(tuple(len(lst) for lst in col_labels.values()))
            for var_i in variables:
                for cat1 in categories1:
                    dm[country, :, var_i, cat1] = table_dict[var]['data'][cat1][var_i]
        else:
            variables = [var]
            categories1 = table_dict[var]['rows']
            categories2 = table_dict[var]['columns']
            col_labels = {'Country': [country], 'Years': years_fts, 'Variables': variables, 'Categories1': categories1, 'Categories2': categories2}
            units = {var: '-' for var in variables}
            dm = DataMatrix(col_labels=col_labels, units=units)
            dm.array = np.zeros(tuple(len(lst) for lst in col_labels.values()))
            for cat1 in categories1:
                for cat2 in categories2:
                    dm[country, :, var, cat1, cat2] = table_dict[var]['data'][cat1][cat2]

        DM[var] = dm
    return DM


def get_tables(filename, country, years_fts):

    table_dict = extract_tables(filename)

    DM = convert_table_dict_to_DM(table_dict, country, years_fts)

    return DM


def parse_ampl_dat(dat_content):
    lines = [line.strip() for line in dat_content.split("\n") if line.strip() and not line.startswith("#")]
    headers = lines[0].split()[1:]  # Extract column names (skip "param:")

    index_list = []
    data_list = []

    for line in lines[1:]:
        parts = line.split()
        index_list.append(parts[0])  # First element is the index
        data_list.append([float(x) for x in parts[1:]])  # Convert the rest to float

    data_array = np.array(data_list)
    return index_list, headers, data_array


filename = "data/ses_main.dat"
country = "Switzerland"
years_fts = create_years_list(2025, 2050, 5)
dm_param = get_parameters(filename, country, years_fts)
DM_tables = get_tables(filename, country, years_fts)
DM_tables['param'] = dm_param

f = '../../../data/datamatrix/energy.pickle'
with open(f, 'wb') as handle:
    pickle.dump(DM_tables, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Hello')