import re
import numpy as np
import copy
import pandas as pd
from model.common.io_database import read_database

# ConstantDataMatrix is a class used to deal with constants in a way that is similar to DataMatrix class.
# The main difference if that ConstantDataMatrix has no Country or Years dimensions.
# ConstantDataMatrix contains:
#       - array: numpy array (can be 1D or more)
#       - dim_labels: list ['Variables', 'Categories1', ..]
#       - col_labels: dict that associates each dimension with the list of column labels
#              e.g.{
#                   'Variables': ['biomass-emission-factor', 'agr_carbon-stock', etc],
#                   'Categories1': ['cropland', 'grassland', 'forest']
#                   }
#       - units: dict that contains the unit corresponding to each Variable e.g. units['agr_carbon-stock'] = 'kt'
#       - idx: dictionary that links every label with the array index position
#                   e.g. idx['biomass-emission-factor'] = 0
#                        idx['grassland'] = 1
#              this is used to access the numpy array e.g. cdm.array[idx['biomass-emission-factor'], idx['grassland']]
#              gives as output the biomass emission factor for grassland.


class ConstantDataMatrix:

    def __init__(self, col_labels={}, units={}, idx={}):
        self.array = None
        self.dim_labels = ["Variables"]  # list
        self.col_labels = {}
        self.units = {}

        for i in range(len(col_labels) - 1):
            cat_num = str(i + 1)
            self.dim_labels.append('Categories' + cat_num)

        for k, v in col_labels.items():
            self.col_labels[k] = v.copy()  # dictionary with dim_labels[i] as key

        for k, v in units.items():
            self.units[k] = v  # dictionary

        if len(col_labels) > 0 and len(idx) == 0:
            self.idx = self.index_all()

        elif len(idx) > 0:
            self.idx = {}
            for k, v in idx.items():
                self.idx[k] = v
        return

    def __repr__(self):
        
        if len(self.col_labels) == 1:
            return f'ConstantDataMatrix with shape {self.array.shape} and variables {self.col_labels["Variables"]}'
        if len(self.col_labels) == 2:
            return f'ConstantDataMatrix with shape {self.array.shape}, variables {self.col_labels["Variables"]} and categories1 {self.col_labels["Categories1"]}'
        if len(self.col_labels) == 3:
            return f'ConstantDataMatrix with shape {self.array.shape}, variables {self.col_labels["Variables"]}, categories1 {self.col_labels["Categories1"]} and categories2 {self.col_labels["Categories2"]}'
        if len(self.col_labels) == 4:
            return f'ConstantDataMatrix with shape {self.array.shape}, variables {self.col_labels["Variables"]}, categories1 {self.col_labels["Categories1"]}, categories2 {self.col_labels["Categories2"]} and categories3 {self.col_labels["Categories3"]}'

    def read_data(self, constant, num_cat):
        dims = []

        if num_cat > 3:
            raise Exception("You can only set maximum 3 categories")

        # Add categories dimension if not there before
        for i in range(num_cat):
            i = i + 1
            cat_str = "Categories" + str(i)
            if cat_str not in self.dim_labels:
                self.dim_labels.append(cat_str)

        for i in self.dim_labels:
            dims.append(len(self.col_labels[i]))

        array = np.empty(dims)
        array[:] = np.nan

        # Iterate over the dataframe columns & extract the string _xxx[ as category and the rest as variable
        for (col_i, col) in enumerate(constant['name']):
            last_bracket_index = col.rfind('[')
            v = col[:last_bracket_index]
            data = constant['value'][col_i]
            c = {}
            for i in range(num_cat):
                last_underscore_index = v.rfind('_')
                c[i] = v[last_underscore_index + 1:]
                v = col[:last_underscore_index]
            if num_cat == 0:
                array[self.idx[v]] = data
            if num_cat == 1:
                array[self.idx[v], self.idx[c[0]]] = data
            if num_cat == 2:
                array[self.idx[v], self.idx[c[1]], self.idx[c[0]]] = data
            if num_cat == 3:
                array[self.idx[v], self.idx[c[2]], self.idx[c[1]], self.idx[c[0]]] = data

        self.array = array
        return

    def extract_structure(self, constant, num_cat=1):
            # It reads a dataframe and it extracts its columns as variables and categories
            # it also extracts the Countries and the Years (sorted)
            # These become elements of the class
            if num_cat > 3:
                raise Exception("You can only set maximum 3 categories")

            # Add categories dimension if not there before
            for i in range(num_cat):
                i = i + 1
                cat_str = "Categories" + str(i)
                if cat_str not in self.dim_labels:
                    self.dim_labels.append(cat_str)

            categories = {}
            variables = []
            units = dict()
            i = 1
            for col in constant['name']:
                unit = re.search(r'\[(.*?)\]', col).group(1)
                col_tmp = col.replace(f'[{unit}]', '')
                for i in range(num_cat):
                    i = i + 1
                    last_underscore_index = col_tmp.rfind('_')
                    cat = col_tmp[last_underscore_index + 1:]
                    if i not in categories.keys():
                        categories[i] = [cat]
                    else:
                        if cat not in categories[i]:
                            categories[i].append(cat)
                    col_tmp = col_tmp.replace(f'_{cat}', '')
                var = col_tmp
                if var not in variables:
                    variables.append(var)
                if var in units.keys():
                    if unit != units[var]:
                        print("Variables " + var + " has two different units, change its name")
                else:
                    units[var] = unit

            self.col_labels["Variables"] = sorted(variables)
            for i in range(num_cat):
                i = i + 1
                dim_str = "Categories" + str(num_cat - i + 1)
                self.col_labels[dim_str] = sorted(categories[i])

            self.idx = self.index_all()
            self.units = units

            return

    @classmethod
    def create_from_constant(cls, constant, num_cat):
        dm = cls()
        dm.extract_structure(constant, num_cat)
        dm.read_data(constant, num_cat)
        return dm

    @classmethod
    def extract_constant(cls, const_file, pattern, num_cat):
        # it extract constant from the file const_file (database format) using pattern a filter for 'eucalc-name'.
        # it returns a ConstantDataMatrix with the number of categories set by num_cat
        # it uses the class method 'crate from constant
        def constant_filter(constant, pattern):
            re_pattern = re.compile(pattern)
            labels = constant['name']
            keep_l_i = [(l, i) for (i, l) in enumerate(labels) if re.match(re_pattern, l)]
            keep = {
                'name': [t[0] for t in keep_l_i],
                'value': [constant['value'][t[1]] for t in keep_l_i],
                'idx': [t[0] for t in keep_l_i],
                'units': [constant['units'][t[0]] for t in keep_l_i]
            }
            return keep

        db_const = read_database(const_file, lever='none', db_format=True)
        const = {
            'name': list(db_const['eucalc-name']),
            'value': list(db_const['value']),
            'idx': dict(zip(list(db_const['eucalc-name']), range(len(db_const['eucalc-name'])))),
            'units': dict(zip(list(db_const['eucalc-name']), list(db_const['unit'])))
        }
        tmp = constant_filter(const, pattern)
        cdm_const = ConstantDataMatrix.create_from_constant(tmp, num_cat=num_cat)
        return cdm_const

    def index_all(self):
        idx = {}
        for (di, d) in enumerate(self.dim_labels):
            for (ci, c) in enumerate(self.col_labels[d]):
                idx[c] = ci
        return idx

    def single_index(self, var_names, dim):
        idx_dict = {}
        # If var_names is a list of variable names do a for loop
        if isinstance(var_names, list):
            for v in var_names:
                idx_dict[v] = self.col_labels[dim].index(v)
        # else var_names should be just a string containing a single variable name
        else:
            idx_dict[var_names] = self.col_labels[dim].index(var_names)

        return idx_dict

    def sort(self, dim):
        sort_index = np.argsort(np.array(self.col_labels[dim]))
        self.col_labels[dim] = sorted(self.col_labels[dim])  # sort labels
        for (ci, c) in enumerate(self.col_labels[dim]):  # sort indexes
            self.idx[c] = ci
        a = self.dim_labels.index(dim)
        self.array = np.take(self.array, sort_index, axis=a)  # re-orders the array according to sort_index
        return

    def copy(self):
        array = self.array.copy()
        col_labels = self.col_labels.copy()  # dictionary with dim_labels[i] as key
        units = self.units.copy()
        dim_labels = self.dim_labels.copy()
        idx = self.idx.copy()
        cdm = ConstantDataMatrix(col_labels=col_labels, units=units)
        cdm.dim_labels = dim_labels
        cdm.idx = idx
        cdm.array = array
        return cdm
    
    def filter(self, selected_cols):
        # Sort the subset list based on the order of elements in list1
        sorted_cols = {}
        for d in self.dim_labels:
            if d in selected_cols.keys():
                if selected_cols[d] == "all":
                    sorted_cols[d] = self.col_labels[d].copy()
                else:
                    sorted_cols[d] = sorted(selected_cols[d], key=lambda x: self.col_labels[d].index(x))
            else:
                sorted_cols[d] = self.col_labels[d].copy()
        out = ConstantDataMatrix(col_labels=sorted_cols)
        out.dim_labels = self.dim_labels.copy()
        # Extract list of indices
        cols_idx = []
        for d in self.dim_labels:
            cols_idx.append([self.idx[xi] for xi in sorted_cols[d]])
        mesh = np.ix_(*cols_idx)
        out.array = self.array[mesh].copy()
        out.units = {key: self.units[key] for key in sorted_cols["Variables"]}
        if len(sorted_cols) > 3:
            out.idx = out.index_all()
        return out
    
    def filter_w_regex(self, dict_dim_pattern):
        # Return only a portion of the DataMatrix based on a dict_dim_patter
        # E.g. if we wanted to only keep Austria and France, the dict_dim_pattern would be {'Country':'France|Austria'}
        keep = {}
        for d in self.dim_labels:
            if d in dict_dim_pattern.keys():
                pattern = re.compile(dict_dim_pattern[d])
                keep[d] = [col for col in self.col_labels[d] if re.match(pattern, col)]
            else:
                keep[d] = 'all'
        dm_keep = self.filter(keep)
        return dm_keep
    
    def deepen(self, sep="_", based_on=None):
        # Adds a category to the datamatrix based on the "Variables" names
        idx_old = self.index_all()

        # Add one category to the dim_labels list depending on the current structure
        if self.dim_labels[-1] == "Variables":
            new_dim = 'Categories1'
            root_dim = 'Variables'
            self.dim_labels.append(new_dim)
        elif self.dim_labels[-1] == 'Categories1':
            new_dim = 'Categories2'
            root_dim = 'Categories1'
            self.dim_labels.append(new_dim)
        elif self.dim_labels[-1] == 'Categories2':
            new_dim = 'Categories3'
            root_dim = 'Categories2'
            self.dim_labels.append(new_dim)
        else:
            raise Exception('You cannot deepen (aka add a dimension) to a datamatrix with already 3 categories')

        if based_on is not None:
            root_dim = based_on

        # Add col labels the added dimension and rename col labels of existing dimension
        # It also takes care of rename the units dictionary if relevant
        self.col_labels[new_dim] = []
        root_cols = []
        rename_mapping = {}

        for col in self.col_labels[root_dim]:
            last_underscore_index = col.rfind(sep)
            if last_underscore_index == -1:
                raise Exception('No separator _ could be found in the last category')
            new_cat = col[last_underscore_index + 1:]
            root_cat = col[:last_underscore_index]
            rename_mapping[col] = [root_cat, new_cat]
            # crates col_labels list for the new dimension
            if new_cat not in self.col_labels[new_dim]:
                self.col_labels[new_dim].append(new_cat)
            # renames the existing root_cat dimension
            if root_cat not in root_cols:
                root_cols.append(root_cat)
            # renames units dict
            if root_dim == 'Variables':
                if root_cat not in self.units.keys():
                    self.units[root_cat] = self.units[col]
                self.units.pop(col)
        self.col_labels[root_dim] = sorted(root_cols)
        self.col_labels[new_dim] = sorted(self.col_labels[new_dim])

        # Restructure data array
        idx_new = self.index_all()
        self.idx = idx_new
        array_old = self.array
        dims = []
        for i in self.dim_labels:
            dims.append(len(self.col_labels[i]))
        array_new = np.empty(dims)
        array_new[...] = np.nan
        if based_on is not None:
            a_root = self.dim_labels.index(root_dim)
            array_new = np.moveaxis(array_new, a_root, -2)
            array_old = np.moveaxis(array_old, a_root, -1)
        for col in rename_mapping.keys():
            [root_cat, new_cat] = rename_mapping[col]
            array_new[..., idx_new[root_cat], idx_new[new_cat]] = array_old[..., idx_old[col]]
        if based_on is not None:
            array_new = np.moveaxis(array_new, -2, a_root)
        self.array = array_new
        return
    
    def deepen_twice(self):
        # Adds two dimensions to the datamatrix based on the last dimension column names
        root_dim = self.dim_labels[-1]
        tmp_cols = []

        for col in self.col_labels[root_dim]:
            last_index = col.rfind("_")
            new_col = col[:last_index] + '?' + col[last_index+1:]
            tmp_cols.append(new_col)
            if root_dim == 'Variables':
                self.units[new_col] = self.units[col]
                self.units.pop(col)
        self.col_labels[root_dim] = tmp_cols

        self.deepen(sep='_')

        tmp_cols = []
        root_dim = self.dim_labels[-1]
        for col in self.col_labels[root_dim]:
            last_index = col.rfind("?")
            new_col = col[:last_index] + '_' + col[last_index+1:]
            tmp_cols.append(new_col)
            if root_dim == 'Variables':
                self.units[new_col] = self.units[col]
                self.units.pop(col)
        self.col_labels[root_dim] = tmp_cols

        self.deepen(sep='_')

        return

    def rename_col(self, col_in, col_out, dim):
        # Rename col_labels
        if isinstance(col_in, str):
            col_in = [col_in]
            col_out = [col_out]
        for i in range(len(col_in)):
            # Rename column labels
            ci = self.idx[col_in[i]]
            self.col_labels[dim][ci] = col_out[i]
            # Rename key for units
            if dim == "Variables":
                self.units[col_out[i]] = self.units[col_in[i]]
                self.units.pop(col_in[i])
            # Rename idx
            self.idx[col_out[i]] = self.idx[col_in[i]]
            self.idx.pop(col_in[i])
    
        return

    def rename_col_regex(self, str1, str2, dim):
        # Rename all columns containing str1 with str2
        col_in = [col for col in self.col_labels[dim] if str1 in col]
        col_out = [word.replace(str1, str2) for word in col_in]
        self.rename_col(col_in, col_out, dim=dim)
        return
    
    def switch_categories_order(self, cat1='Categories1', cat2='Categories2'):
        if 'Categories' not in cat1 or 'Categories' not in cat2:
            raise ValueError(' You can only switch the order of two Categories')
        # Extract axis of cat1, cat2
        a1 = self.dim_labels.index(cat1)
        a2 = self.dim_labels.index(cat2)
        # Switch axis in array
        self.array = np.moveaxis(self.array, a1, a2)
        # Switch col_labels
        col1 = self.col_labels[cat1]
        col2 = self.col_labels[cat2]
        self.col_labels[cat1] = col2
        self.col_labels[cat2] = col1
        return
    
    def add(self, new_array, dim, col_label, unit=None, dummy=False):
        # Adds the numpy array new_array to the datamatrix over dimension dim.
        # The label associated with the array is in defined by the string col_label
        # The unit is needed as a string (e.g. 'km') only if dim = 'Variables'
        # It does not return a new datamatrix
        # You can also use to add 'dummy' dimension to a datamatrix,
        # usually before appending it to another that has more categories
        self_shape = self.array.shape
        a = self.dim_labels.index(dim)
        new_shape = list(self_shape)
        if isinstance(col_label, str):
            # if I'm adding only one column
            col_label = [col_label]
            unit = [unit]
        new_shape[a] = len(col_label)
        new_shape = tuple(new_shape)
        # If it is adding a new array of constant value (e.g. nan) to have a dummy dimension:
        if isinstance(new_array, (float, int)) and dummy is True:
            new_array = new_array * np.ones(new_shape)
        elif len(col_label) == 1 and new_array.shape != new_shape:
            new_array = new_array[..., np.newaxis]
            new_array = np.moveaxis(new_array, -1, a)
        # Else check that the new array dimension is correct
        if new_array.shape != new_shape and dummy is False:
            raise AttributeError(f'The new_array should have dimension {new_shape} instead of {new_array.shape}, '
                                 f'unless you want to add dummy dimensions, then you should add dummy = True and new_array should be a float')
        for col in col_label:
            self.col_labels[dim].append(col)
            i_v = self.single_index(col, dim)
            if col not in list(self.idx.keys()):
                self.idx[col] = i_v[col]
            else:
                raise ValueError(f"You are trying to append data under the label {col_label} which already exists")
        if dim == 'Variables':
            for i, col in enumerate(col_label):
                self.units[col] = unit[i]
        self.array = np.concatenate((self.array, new_array), axis=a)

        return


    def drop(self, dim, col_label):
        # It removes the column col_label along dimension dim
        # as well as the data in array associated to it
        # It does not return a new datamatrix
        # Get the axis of the dimension
        a = self.dim_labels.index(dim)
        # if col_label it's a string, check for the columns that match the regex pattern
        if isinstance(col_label, str):
            tmp = [c for c in self.col_labels[dim] if re.match(col_label, c)]
            col_label = tmp
        # remove the data from the matrix
        idx = self.single_index(col_label, dim)  # get index of col_label
        i_val = list(idx.values())
        self.array = np.delete(self.array, i_val, axis=a)  # remove array
        # remove the label
        for c in col_label:
            self.col_labels[dim].remove(c)
            self.idx.pop(c)
        # Re-assign idx for the dimension
        for col_i, col in enumerate(self.col_labels[dim]):
            self.idx[col] = col_i
        # remove the unit
        if dim == "Variables":
            if isinstance(col_label, list):
                for c in col_label:
                    self.units.pop(c)
            else:
                self.units.pop(col_label)
        return

    def flatten(self, sep='_'):
        # you can flatten only if you have at least one category
        assert len(self.dim_labels) > 1
        d_2 = self.dim_labels[-1]
        cols_2 = self.col_labels[d_2]
        d_1 = self.dim_labels[-2]
        cols_1 = self.col_labels[d_1]
        new_shape = []
        new_col_labels = {}
        for d in self.dim_labels:
            if d is not d_1 and d is not d_2:
                new_shape.append(len(self.col_labels[d]))
                new_col_labels[d] = self.col_labels[d].copy()
        new_shape.append(1)
        new_shape = tuple(new_shape)
        new_array = np.empty(shape=new_shape)
        new_array[...] = np.nan
        new_cols = []
        new_units = {}
        i = 0
        for c1 in cols_1:
            for c2 in cols_2:
                col_value = self.array[..., self.idx[c1], self.idx[c2], np.newaxis]
                if not np.isnan(col_value).all():
                    new_cols.append(f'{c1}{sep}{c2}')
                    if i == 0:
                        i = i+1
                        new_array = col_value
                    else:
                        new_array = np.concatenate([new_array, col_value], axis=-1)
                    if d_1 == 'Variables':
                        new_units[f'{c1}{sep}{c2}'] = self.units[c1]
    
        new_col_labels[d_1] = new_cols
        if d_1 == 'Variables':
            dm_new = ConstantDataMatrix(col_labels=new_col_labels, units=new_units)
        else:
            dm_new = ConstantDataMatrix(col_labels=new_col_labels, units=self.units)
        dm_new.array = new_array
        dm_new.idx = dm_new.index_all()
        return dm_new
    
    def append(self, data2, dim):
        # appends DataMatrix data2 to self in dimension dim.
        # The pre-requisite is that all other dimensions match
        dim_lab = self.dim_labels.copy()
        dim_lab.remove(dim)
        # if 'Variables' in dim_lab:
        #    dim_lab.remove("Variables")
        for d in dim_lab:
            if self.col_labels[d] != data2.col_labels[d]:
                self.sort(dim=d)
                data2.sort(dim=d)
                if self.col_labels[d] != data2.col_labels[d]:
                    raise ValueError(f'columns {self.col_labels[d]} do not match columns {data2.col_labels[d]}')
        # Check that units are the same
        if dim != 'Variables':
            if self.units != data2.units:
                raise ValueError(f'The units should be the same')
        # Check that across the dimension where you want to append the labels are different
        cols1 = set(self.col_labels[dim])
        cols2 = set(data2.col_labels[dim])
        same_col = cols2.intersection(cols1)
        if len(same_col) != 0:
            raise Exception("The DataMatrix that you are trying to append contains the same labels across dimension ", dim)

        # Concatenate the two arrays
        a = self.dim_labels.index(dim)
        self.array = np.concatenate((self.array, data2.array), axis=a)
        # Concatenate the two lists of labels across dimension dim
        self.col_labels[dim] = self.col_labels[dim] + data2.col_labels[dim]
        # Re initialise the indexes
        for (ci, c) in enumerate(self.col_labels[dim]):  # sort indexes
            self.idx[c] = ci
        # Add the units if you are appending over "Variables"
        if dim == "Variables":
            self.units = self.units | data2.units
            
    def write_df(self):
        dm = self.copy()
        # years = dm.col_labels["Years"]
        # countries = dm.col_labels["Country"]
        # n_y = len(years)
        # n_c = len(countries)
        # # Repeat countries n_year number of times
        # country_list = [item for item in countries for _ in range(n_y)]
        # years_list = years * n_c
        df = pd.DataFrame()

        num_cat = len(dm.dim_labels) - 1

        if num_cat == 3:
            dm_new = dm.flatten()
            dm.__dict__.update(dm_new.__dict__)  # it replaces self with dm_new
            num_cat = len(dm.dim_labels) - 1

        if num_cat == 2:
            dm_new = dm.flatten()
            dm.__dict__.update(dm_new.__dict__)  # it replaces self with dm_new
            num_cat = len(dm.dim_labels) - 1

        if num_cat == 0:
            for v in dm.col_labels["Variables"]:
                col_name = v + "[" + dm.units[v] + "]"
                col_value = dm.array[dm.idx[v]].flatten()
                df[col_name] = col_value
        if num_cat == 1:
            for v in dm.col_labels["Variables"]:
                for c in dm.col_labels["Categories1"]:
                    col_name = v + "_" + c + "[" + dm.units[v] + "]"
                    col_value = dm.array[dm.idx[v], dm.idx[c]].flatten()
                    if not np.isnan(col_value).all():
                        df[col_name] = col_value
        return df
    
    def group_all(self, dim=str, inplace=True, aggregation = "sum"):
        # Function to drop a dimension by summing all categories
        # Call example: dm_to_group.group_all(dim='Categories2', inplace=True)
        # or dm_grouped = dm_to_group.group_all(dim='Categories1', inplace=False)
        # when inplace = False dm_to_group remains unchanged and the grouped dm is return as output
        if 'Categories' not in dim:
            raise ValueError(f'You can only use group_all() on Categories')
        if inplace:
            dm = self
        else:
            dm = self.copy()
        a = dm.dim_labels.index(dim)
        if aggregation == "sum":
            dm.array = np.nansum(dm.array, axis=a)
        if aggregation == "mean":
            dm.array = np.nanmean(dm.array, axis=a)
        # Remove indexes
        for col in dm.col_labels[dim]:
            dm.idx.pop(col)
        # Rename categories
        categories_to_rename = [cat for cat in dm.dim_labels if 'Categories' in cat]
        categories_to_rename.remove(dim)
        i = 1
        for old_cat in categories_to_rename:
            new_cat = 'Categories' + str(i)
            dm.col_labels[new_cat] = dm.col_labels[old_cat]
            i = i + 1
        # Remove last category and dimension
        last_dim = dm.dim_labels[-1]
        dm.col_labels.pop(last_dim)
        dm.dim_labels = dm.dim_labels[:-1]
        if not inplace:
            return dm
        return

    def __getitem__(self, key):
        if not isinstance(key, tuple):  # Ensure key is a tuple
            key = (key,)
        key = tuple(self.idx.get(k, k) if not isinstance(k, slice) else k for k in key)
        arr = self.array[key]
        return arr

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):  # Ensure key is a tuple
            key = (key,)
        key = tuple(self.idx.get(k, k) if not isinstance(k, slice) else k for k in key)
        self.array[key] = value
        return