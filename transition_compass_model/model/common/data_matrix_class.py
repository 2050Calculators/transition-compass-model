import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import itertools
import copy

# DataMatrix is the by-default class used by the calculator.
# DataMatrix contains:
#       - array: numpy array (can be 3D or more)
#       - dim_labels: list ['Country', 'Years', 'Variables', 'Categories1', ..]
#       - col_labels: dict that associates each dimension with the list of column labels
#              e.g.{
#                   'Country': ['Austria', 'Belgium', ..],
#                   'Years': [1990, 1991, ..., 2015, 2020, ..., 2050]
#                   'Variables': ['tra_passenger_modal-share', 'tra_passenger_occupancy', ...],
#                   'Categories1': ['LDV', '2W', 'rail', 'aviation', ...]
#                   }
#       - units: dict that contains the unit corresponding to each Variable
#               e.g. units['tra_passenger_modal-share'] = '%'
#       - idx: dictionary that links every label with the array index position
#                   e.g. idx['Austria'] = 0
#                        idx['Belgium'] = 1
#                        idx[1990] = 0
#              this is used to access the numpy array e.g.
#              dm.array[idx['Austria'], :, idx['tra_passenger_modal-share'], idx['LDV']]
#              gives the share of light duty vehicles (cars) in Austria for all years.


class DataMatrix:

    def __init__(self, col_labels=dict(), units=dict(), idx=dict(), empty=False):
        # Empty = True does not crate the arrray
        self.dim_labels = ["Country", "Years", "Variables"]  # list
        self.col_labels = {}
        self.units = {}

        for i in range(len(col_labels) - 3):
            cat_num = str(i + 1)
            self.dim_labels.append("Categories" + cat_num)

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
        if not empty:
            arr_shape = []
            for dim in self.dim_labels:
                arr_shape.append(len(self.col_labels[dim]))
            self.array = np.nan*np.ones(tuple(arr_shape))
        return


    def __repr__(self):

        if len(self.col_labels) == 3:
            return f'DataMatrix with shape {self.array.shape} and variables {self.col_labels["Variables"]}'
        if len(self.col_labels) == 4:
            return f'DataMatrix with shape {self.array.shape}, variables {self.col_labels["Variables"]} and categories1 {self.col_labels["Categories1"]}'
        if len(self.col_labels) == 5:
            return f'DataMatrix with shape {self.array.shape}, variables {self.col_labels["Variables"]}, categories1 {self.col_labels["Categories1"]} and categories2 {self.col_labels["Categories2"]}'
        if len(self.col_labels) == 6:
            return f'DataMatrix with shape {self.array.shape}, variables {self.col_labels["Variables"]}, categories1 {self.col_labels["Categories1"]}, categories2 {self.col_labels["Categories2"]} and categories3 {self.col_labels["Categories3"]}'

    def read_data(self, df, num_cat):
        # Function called by the classmethod 'create_from_df' (see below)
        # It is used to transform a dataframe df (table) into a datamatrix by specifying the number of categories
        # ATT: to be run after extract_structure which initialises dim_labels, col_labels and units
        if df.empty:
            ValueError(f"You cannot create a datamatrix from an empty dataframe.")

        dims = []
        df.sort_values(by=["Country", "Years"], inplace=True)
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
        df.set_index(["Country", "Years"], inplace=True)

        # Iterate over the dataframe columns & extract the string _xxx[ as category and the rest as variable
        for col in df.columns:
            last_bracket_index = col.rfind("[")
            v = col[:last_bracket_index]
            series_data = df[col]
            c = {}
            for i in range(num_cat):
                last_underscore_index = v.rfind("_")
                c[i] = v[last_underscore_index + 1 :]
                v = col[:last_underscore_index]
            if num_cat == 0:
                array[:, :, self.idx[v]] = np.reshape(
                    series_data.values, (dims[0], dims[1])
                )
            if num_cat == 1:
                array[:, :, self.idx[v], self.idx[c[0]]] = np.reshape(
                    series_data.values, (dims[0], dims[1])
                )
            if num_cat == 2:
                array[:, :, self.idx[v], self.idx[c[1]], self.idx[c[0]]] = np.reshape(
                    series_data.values, (dims[0], dims[1])
                )
            if num_cat == 3:
                array[
                    :, :, self.idx[v], self.idx[c[2]], self.idx[c[1]], self.idx[c[0]]
                ] = np.reshape(series_data.values, (dims[0], dims[1]))

        df.reset_index(inplace=True)
        self.array = array
        return

    def extract_structure(self, df, num_cat=1):
        # It reads a dataframe, and it extracts its columns as variables and categories
        # it also extracts the Countries and the Years (sorted)
        # These become elements of the class

        # checks if cols 'Country' and 'Years' are in the datafram
        def check_columns(dataframe):
            required_columns = ["Years", "Country"]
            missing_columns = [
                col for col in required_columns if col not in dataframe.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Missing required columns: {', '.join(missing_columns)}"
                )
            return

        if df.empty:
            ValueError(f"You cannot create a datamatrix from an empty dataframe.")

        check_columns(df)

        if num_cat > 3:
            raise Exception("You can only set maximum 3 categories")

        # Add categories dimension if not there before
        for i in range(num_cat):
            i = i + 1
            cat_str = "Categories" + str(i)
            if cat_str not in self.dim_labels:
                self.dim_labels.append(cat_str)

        cols = [col for col in df.columns if col not in ["Country", "Years"]]
        categories = {}
        variables = []
        units = dict()
        for col in cols:
            try:
                unit = re.search(r"\[(.*?)\]", col).group(1)
            except AttributeError:
                raise AttributeError(
                    "Error: try to remove the lever column from the dataframe and make sure all variables have units in eucalc-name"
                )
                exit()
            col_tmp = col.replace(f"[{unit}]", "")
            for i in range(num_cat):
                i = i + 1
                last_underscore_index = col_tmp.rfind("_")
                cat = col_tmp[last_underscore_index + 1 :]
                if i not in categories.keys():
                    categories[i] = [cat]
                else:
                    if cat not in categories[i]:
                        categories[i].append(cat)
                col_tmp = re.sub(f"_{cat}$", "", col_tmp)
            var = col_tmp
            if var not in variables:
                variables.append(var)
            if var in units.keys():
                if unit != units[var]:
                    print(
                        "Variables " + var + " has two different units, change its name"
                    )
            else:
                units[var] = unit

        self.col_labels["Country"] = sorted(list(set(df["Country"])))
        self.col_labels["Years"] = sorted(list(set(df["Years"].astype(int))))
        self.col_labels["Variables"] = sorted(variables)
        for i in range(num_cat):
            i = i + 1
            dim_str = "Categories" + str(num_cat - i + 1)
            self.col_labels[dim_str] = sorted(categories[i])

        self.idx = self.index_all()
        self.units = units

        return

    @classmethod
    def create_from_df(cls, df, num_cat):
        # Creates a datamatrix given a dataframe and the number of categories that we want
        # Note that df needs to have columns 'Country' and 'Years'
        # it returns a datamatrix
        if df.empty:
            ValueError(f'You cannot create a datamatrix from an empty dataframe.')
        dm = cls(empty=True)
        dm.extract_structure(df, num_cat)
        dm.read_data(df, num_cat)
        return dm

    @classmethod
    def based_on(cls, array, format, change=dict(), units=dict()):
        # Creates a datamatrix given an array and a datamatrix (based_on) from which to take the structure
        # If the structure differ, you can define in a dictionary (change) with the dimension that differ,
        # for example:
        # dm_new = DataMatrix.based_on(arr, dm_ref, {'Variables': ['new_var1', new_var2'], 'Categories2': None},
        #                              units = {'new_var1': 'TWh', 'new_var2': 'GWh'})
        dm = format
        # col_labels = copy.deepcopy(dm.col_labels)
        col_labels = {}
        for key, value in dm.col_labels.items():
            col_labels[key] = value.copy()
        dim_labels = dm.dim_labels.copy()
        new_units = dm.units.copy()
        idx = dm.idx.copy()
        for dim in change:
            if dim in col_labels:
                new_labels = change[dim]
                # Either drop the dimension
                if new_labels is None:
                    dim_labels.remove(dim)
                    for col in col_labels[dim]:
                        idx.pop(col)
                    col_labels.pop(dim)
                # Or modify the dimension
                else:
                    if isinstance(new_labels, list):
                        for col in col_labels[dim]:
                            idx.pop(col)
                        col_labels[dim] = new_labels
                        for i, new_col in enumerate(new_labels):
                            idx[new_col] = i
                        if dim == "Variables":
                            new_units = units
                    else:
                        raise ValueError(
                            f"The argument change can only be a list or None"
                        )
            else:
                num_cat = int(
                    dim[-1]
                )  # extract the categorie number that they want to add
                current_cat = len(dim_labels) - 3
                if num_cat != current_cat + 1:
                    raise ValueError(
                        f"You can add Categories{int(current_cat) + 1} not {dim}"
                    )
                dim_labels.append(dim)
                col_labels[dim] = change[dim]
                for i, new_col in enumerate(change[dim]):
                    idx[new_col] = i
        dm_new = DataMatrix(col_labels, new_units, idx)
        dm_new.array = array
        for i, dim in enumerate(dim_labels):
            if len(col_labels[dim]) != array.shape[i]:
                raise ValueError(
                    f"Mismatch between array shape and col_labels for dim={dim}"
                )

        return dm_new

    def read_data_0cat(self, df):
        # use read_data instead
        dims = []
        for i in self.dim_labels:
            dims.append(len(self.col_labels[i]))

        array = np.empty(dims)
        array[:] = np.nan
        df.set_index(["Country", "Years"], inplace=True)

        # Iterate over the dataframe columns
        for col in df.columns:
            last_bracket_index = col.rfind("[")
            v = col[:last_bracket_index]
            series_data = df[col]
            array[:, :, self.idx[v]] = np.reshape(
                series_data.values, (dims[0], dims[1])
            )

        self.array = array

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
            raise AttributeError(
                f"The new_array should have dimension {new_shape} instead of {new_array.shape}, "
                f"unless you want to add dummy dimensions, then you should add dummy = True and new_array should be a float"
            )
        for col in col_label:
            self.col_labels[dim].append(col)
            i_v = self.single_index(col, dim)
            if col not in list(self.idx.keys()):
                self.idx[col] = i_v[col]
            else:
                raise ValueError(f"You are trying to append data under the label {col_label} which already exists")
        if dim == 'Variables':
            if unit is not None:
                for i, col in enumerate(col_label):
                    self.units[col] = unit[i]
            else:
                raise ValueError(f"You need to input the units when adding a variables")
        self.array = np.concatenate((self.array, new_array), axis=a)

    def drop(self, dim, col_label):
        # It removes the column col_label along dimension dim
        # as well as the data in array associated to it
        # It does not return a new datamatrix
        # Get the axis of the dimension
        a = self.dim_labels.index(dim)
        # if col_label it's a string, check for the columns that match the regex pattern
        if isinstance(col_label, str):
            tmp = [c for c in self.col_labels[dim] if re.match(col_label, str(c))]
            col_label = tmp
        # If you are removing years col_label is an integer
        if isinstance(col_label, int):
            col_label = [col_label]
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

    def lag_variable(self, pattern, shift, subfix):
        # It lags the columns over the years based on the regex pattern "pattern"
        # (e.g. 'tra_passenger_.*|tra_freight_.*')
        # the dimension is always 'Variables' and it lags by a integer 'shift'
        # new column labels across dimension 'Variables' are added with subfix 'subfix'
        vars = [
            (vi, v)
            for (vi, v) in enumerate(self.col_labels["Variables"])
            if re.match(pattern, v)
        ]
        dim_label = "Variables"
        for vi, v in vars:
            v_sub = v + subfix  # new variable name
            unit = self.units[v]
            new_array = np.roll(
                self.array[:, :, vi, ...], shift, axis=1
            )  # shift along Years axis
            if shift == 1:
                new_array[:, 0, ...] = new_array[:, 1, ...]  # copy 1991 value to 1990
            elif shift == -1:
                new_array[:, -1, ...] = new_array[:, -2, ...]  # copy 2045 value to 2050
            else:
                raise Exception(
                    "You can only shift by +1 or -1 in lag_variable func of DataMatrix class"
                )
            self.add(
                new_array, dim_label, v_sub, unit
            )  # append new_array to self_array

    def single_index(self, var_names, dim):
        # it extract the positional index of the labels in var_names across dimension dim
        idx_dict = {}
        # If var_names is a list of variable names do a for loop
        if isinstance(var_names, list):
            for v in var_names:
                idx_dict[v] = self.col_labels[dim].index(v)
        # else var_names should be just a string containing a single variable name
        else:
            idx_dict[var_names] = self.col_labels[dim].index(var_names)

        return idx_dict

    def index_all(self):
        # extracts all the indexes and returns the dictionary idx (as well as re-assinging self.idx)
        idx = {}
        for di, d in enumerate(self.dim_labels):
            for ci, c in enumerate(self.col_labels[d]):
                idx[c] = ci
        self.idx = idx

        return idx

    def overwrite_1cat(self, matrix2):
        country = matrix2.col_labels["Country"]
        years = matrix2.col_labels["Years"]
        variables = matrix2.col_labels["Variables"]
        categories = matrix2.col_labels["Categories1"]
        all_cols = set(country + years + variables + categories)
        idx = self.index_all()
        if all_cols.issubset(set(idx.keys())):
            i_c = [idx[x] for x in country]
            i_y = [idx[x] for x in years]
            i_v = [idx[x] for x in variables]
            i_cat = [idx[x] for x in categories]
            mesh = np.ix_(i_c, i_y, i_v, i_cat)  # Create meshgrid
            self.array[mesh] = matrix2.array
        else:
            raise Exception(
                "You are try to overwrite a DataMatrix with another DataMatrix that isn't a subset"
            )

    def fill_nans(self, dim_to_interp):

        axis_to_interp = self.dim_labels.index(dim_to_interp)

        def interpolate_nans(arr, x_values):
            nan_indices = np.isnan(arr)
            if nan_indices.any():
                if len(x_values[~nan_indices]) > 0:
                    arr[nan_indices] = np.interp(
                        x_values[nan_indices], x_values[~nan_indices], arr[~nan_indices]
                    )
            return arr

        # Apply interpolation along the specified axis
        if np.isnan(self.array).any():
            if dim_to_interp == "Years":
                x_values = np.array(self.col_labels["Years"])
            else:
                x_values = np.arange(len(self.array))
            self.array = np.apply_along_axis(
                interpolate_nans, axis_to_interp, self.array, x_values
            )

        return

    def operation(
        self,
        col1,
        operator,
        col2,
        dim="Variables",
        out_col=None,
        unit=None,
        div0="error",
        nansum=False,
        type=float,
    ):
        # operation allows to perform operation between two columns belonging to the same
        # dimensions in DataMatrix and to append/overwrite the result to the dataframe
        i = self.idx
        a = self.dim_labels.index(dim)  #

        self.array = np.moveaxis(self.array, a, -1)  # Move the axis of array to the end

        def interpolate_nans(arr):
            nan_indices = np.isnan(arr)
            if nan_indices.any():
                x_values = np.arange(len(arr))
                if len(x_values[~nan_indices]) > 0:
                    arr[nan_indices] = np.interp(
                        x_values[nan_indices], x_values[~nan_indices], arr[~nan_indices]
                    )
            return arr

        if operator == "/":
            if div0 == "error":
                tmp = self.array[..., i[col1]] / self.array[..., i[col2]]
            if div0 == "interpolate":
                axis_to_interp = 1
                tmp = np.divide(
                    self.array[..., i[col1]],
                    self.array[..., i[col2]],
                    out=np.nan * np.ones_like(self.array[..., i[col1]]),
                    where=self.array[..., i[col2]] != 0,
                )
                # Apply interpolation along the specified axis
                if np.isnan(tmp).any():
                    tmp = np.apply_along_axis(interpolate_nans, axis_to_interp, tmp)

        if operator == "-":
            if not nansum:
                tmp = self.array[..., i[col1]] - self.array[..., i[col2]]
            else:
                tmp = np.nan_to_num(self.array[..., i[col1]]) - np.nan_to_num(
                    self.array[..., i[col2]]
                )

        if operator == "+":
            if not nansum:
                tmp = self.array[..., i[col1]] + self.array[..., i[col2]]
            else:
                tmp = np.nan_to_num(self.array[..., i[col1]]) + np.nan_to_num(
                    self.array[..., i[col2]]
                )

        if operator == "*":
            tmp = self.array[..., i[col1]] * self.array[..., i[col2]]

        self.array = np.moveaxis(self.array, -1, a)

        if out_col is not None:
            self.add(tmp.astype(type), dim, out_col, unit)
        else:
            return tmp

        return

    def write_df(self):
        dm = self.copy()
        years = dm.col_labels["Years"]
        countries = dm.col_labels["Country"]
        n_y = len(years)
        n_c = len(countries)
        # Repeat countries n_year number of times
        country_list = [item for item in countries for _ in range(n_y)]
        years_list = years * n_c
        df = pd.DataFrame(
            data=zip(country_list, years_list), columns=["Country", "Years"]
        )

        num_cat = len(dm.dim_labels) - 3

        if num_cat == 3:
            dm_new = dm.flatten()
            dm.__dict__.update(dm_new.__dict__)  # it replaces self with dm_new
            num_cat = len(dm.dim_labels) - 3

        if num_cat == 2:
            dm_new = dm.flatten()
            dm.__dict__.update(dm_new.__dict__)  # it replaces self with dm_new
            num_cat = len(dm.dim_labels) - 3

        if num_cat == 0:
            for v in dm.col_labels["Variables"]:
                col_name = v + "[" + dm.units[v] + "]"
                col_value = dm.array[:, :, dm.idx[v]].flatten()
                df[col_name] = col_value
        if num_cat == 1:
            for v in dm.col_labels["Variables"]:
                for c in dm.col_labels["Categories1"]:
                    col_name = v + "_" + c + "[" + dm.units[v] + "]"
                    col_value = dm.array[:, :, dm.idx[v], dm.idx[c]].flatten()
                    if not np.isnan(col_value).all():
                        df[col_name] = col_value
        return df

    def fast_write_df(self):
        """
        Convert DataMatrix to a pandas DataFrame with improved performance.
        """
        # Avoid full copy when possible
        if len(self.dim_labels) <= 5:  # Only copy if we need to flatten
            dm = self
        else:
            dm = self.copy()

        years = dm.col_labels["Years"]
        countries = dm.col_labels["Country"]
        n_y = len(years)
        n_c = len(countries)

        # Pre-allocate arrays instead of list comprehensions
        country_indices = np.repeat(np.arange(n_c), n_y)
        year_indices = np.tile(np.arange(n_y), n_c)

        # Create base dataframe with optimized memory usage
        df = pd.DataFrame(
            {
                "Country": np.array(countries)[country_indices],
                "Years": np.array(years)[year_indices],
            }
        )

        num_cat = len(dm.dim_labels) - 3

        # Combine repeated flattening operations
        if num_cat > 1:
            while num_cat > 1:
                dm_new = dm.flatten()
                dm.__dict__.update(dm_new.__dict__)
                num_cat = len(dm.dim_labels) - 3

        # Process variables in a vectorized way
        if num_cat == 0:
            # Prepare all data columns at once
            var_names = dm.col_labels["Variables"]
            var_indices = [dm.idx[v] for v in var_names]
            col_names = [f"{v}[{dm.units[v]}]" for v in var_names]

            # Extract and reshape data efficiently
            all_values = dm.array[:, :, var_indices]
            all_values_flat = all_values.reshape(n_c * n_y, len(var_names))

            # Add columns to dataframe at once
            df_cols = pd.DataFrame(all_values_flat, columns=col_names)
            df = pd.concat([df, df_cols], axis=1)

        elif num_cat == 1:
            # Prepare columns with non-null data
            data_columns = {}
            var_names = dm.col_labels["Variables"]
            cat_names = dm.col_labels["Categories1"]

            # Instead of nested loops, use vectorized operations
            for v_idx, v in enumerate(var_names):
                v_array = dm.array[
                    :, :, dm.idx[v], :
                ]  # Get all categories for this variable

                for c_idx, c in enumerate(cat_names):
                    col_name = f"{v}_{c}[{dm.units[v]}]"
                    col_values = v_array[:, :, c_idx].flatten()

                    # Only add column if it contains non-NaN values
                    if not np.isnan(col_values).all():
                        data_columns[col_name] = col_values

            # Add all columns to dataframe at once
            if data_columns:
                df_cols = pd.DataFrame(data_columns)
                df = pd.concat([df, df_cols], axis=1)

        return df

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

    def filter(self, selected_cols, inplace=False):
        # If you need to create a new output:
        if not inplace:
            # Sort the subset list based on the order of elements in list1
            sorted_cols = {}
            for d in self.dim_labels:
                # if I'm not filtering over this dimension keep all
                if d not in selected_cols.keys():
                    sorted_cols[d] = self.col_labels[d].copy()
                # otherwise keep selected cols but ordered as in the original dm
                else:
                    sorted_cols[d] = sorted(
                        selected_cols[d], key=lambda x: self.col_labels[d].index(x)
                    )
            keep_units = {key: self.units[key] for key in sorted_cols["Variables"]}
            out = DataMatrix(col_labels=sorted_cols, units=keep_units)
            # Extract list of indices
            cols_idx = []
            for d in self.dim_labels:
                cols_idx.append([self.idx[xi] for xi in sorted_cols[d]])
            # Copy filtered array
            mesh = np.ix_(*cols_idx)
            out.array = self.array[mesh].copy()
            # check if out datamatrix is empty
            if (np.array(out.array.shape) == 0).any():
                raise ValueError(
                    ".filter() return an empty datamatrix across at least one dimension"
                )
            return out
        else:
            for dim, col_to_keep in selected_cols.items():
                cols_to_drop = list(set(self.col_labels[dim]) - set(col_to_keep))
                self.drop(dim=dim, col_label=cols_to_drop)
            if (np.array(self.array.shape) == 0).any():
                raise ValueError(
                    ".filter() return an empty datamatrix across at least one dimension"
                )
            return

    def filter_w_regex(self, dict_dim_pattern, inplace=False):
        # Return only a portion of the DataMatrix based on a dict_dim_patter
        # E.g. if we wanted to only keep Austria and France, the dict_dim_pattern would be {'Country':'France|Austria'}
        keep = {}
        for d in dict_dim_pattern.keys():
            pattern = re.compile(dict_dim_pattern[d])
            keep[d] = [col for col in self.col_labels[d] if re.match(pattern, str(col))]
        dm_keep = self.filter(keep, inplace)
        return dm_keep

    def rename_col_regex(self, str1, str2, dim):
        # Rename all columns containing str1 with str2
        col_in = [col for col in self.col_labels[dim] if str1 in col]
        col_out = [word.replace(str1, str2) for word in col_in]
        self.rename_col(col_in, col_out, dim=dim)
        return

    def sort(self, dim):
        sort_index = np.argsort(np.array(self.col_labels[dim]))
        self.col_labels[dim] = sorted(self.col_labels[dim])  # sort labels
        for ci, c in enumerate(self.col_labels[dim]):  # sort indexes
            self.idx[c] = ci
        a = self.dim_labels.index(dim)
        self.array = np.take(
            self.array, sort_index, axis=a
        )  # re-orders the array according to sort_index

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
                    raise ValueError(
                        f"columns {self.col_labels[d]} do not match columns {data2.col_labels[d]}"
                    )
        # Check that units are the same
        if dim != "Variables":
            if self.units != data2.units:
                raise ValueError(f"The units should be the same")
        # Check that across the dimension where you want to append the labels are different
        cols1 = set(self.col_labels[dim])
        cols2 = set(data2.col_labels[dim])
        same_col = cols2.intersection(cols1)
        if len(same_col) != 0:
            raise Exception(
                "The DataMatrix that you are trying to append contains the same labels across dimension ",
                dim,
            )

        # Concatenate the two arrays
        a = self.dim_labels.index(dim)
        self.array = np.concatenate((self.array, data2.array), axis=a)
        # Concatenate the two lists of labels across dimension dim
        self.col_labels[dim] = self.col_labels[dim] + data2.col_labels[dim]
        # Re initialise the indexes
        for ci, c in enumerate(self.col_labels[dim]):  # sort indexes
            self.idx[c] = ci
        # Add the units if you are appending over "Variables"
        if dim == "Variables":
            self.units = self.units | data2.units

        return

    def deepen(self, sep="_", based_on=None):
        # Adds a category to the datamatrix based on the "Variables" names
        idx_old = self.index_all()

        # Add one category to the dim_labels list depending on the current structure
        if self.dim_labels[-1] == "Variables":
            new_dim = "Categories1"
            root_dim = "Variables"
            self.dim_labels.append(new_dim)
        elif self.dim_labels[-1] == "Categories1":
            new_dim = "Categories2"
            root_dim = "Categories1"
            self.dim_labels.append(new_dim)
        elif self.dim_labels[-1] == "Categories2":
            new_dim = "Categories3"
            root_dim = "Categories2"
            self.dim_labels.append(new_dim)
        else:
            raise Exception(
                "You cannot deepen (aka add a dimension) to a datamatrix with already 3 categories"
            )

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
                raise Exception("No separator _ could be found in the last category")
            new_cat = col[last_underscore_index + 1 :]
            root_cat = col[:last_underscore_index]
            rename_mapping[col] = [root_cat, new_cat]
            # crates col_labels list for the new dimension
            if new_cat not in self.col_labels[new_dim]:
                self.col_labels[new_dim].append(new_cat)
            # renames the existing root_cat dimension
            if root_cat not in root_cols:
                root_cols.append(root_cat)
            # renames units dict
            if root_dim == "Variables":
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
            array_new[..., idx_new[root_cat], idx_new[new_cat]] = array_old[
                ..., idx_old[col]
            ]
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
            new_col = col[:last_index] + "?" + col[last_index + 1 :]
            tmp_cols.append(new_col)
            if root_dim == "Variables":
                self.units[new_col] = self.units[col]
                self.units.pop(col)
        self.col_labels[root_dim] = tmp_cols

        self.deepen(sep="_")

        tmp_cols = []
        root_dim = self.dim_labels[-1]
        for col in self.col_labels[root_dim]:
            last_index = col.rfind("?")
            new_col = col[:last_index] + "_" + col[last_index + 1 :]
            tmp_cols.append(new_col)
            if root_dim == "Variables":
                self.units[new_col] = self.units[col]
                self.units.pop(col)
        self.col_labels[root_dim] = tmp_cols

        self.deepen(sep="_")

        return

    def flatten(self, sep="_"):
        # you can flatten only if you have at least one category
        assert len(self.dim_labels) > 3
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
                    new_cols.append(f"{c1}{sep}{c2}")
                    if i == 0:
                        i = i + 1
                        new_array = col_value
                    else:
                        new_array = np.concatenate([new_array, col_value], axis=-1)
                    if d_1 == "Variables":
                        new_units[f"{c1}{sep}{c2}"] = self.units[c1]

        new_col_labels[d_1] = new_cols
        if d_1 == "Variables":
            dm_new = DataMatrix(col_labels=new_col_labels, units=new_units)
        else:
            dm_new = DataMatrix(col_labels=new_col_labels, units=self.units)
        dm_new.array = new_array
        dm_new.idx = dm_new.index_all()
        return dm_new

    def flattest(self):
        while "Categories" in "\t".join(self.dim_labels):
            self = self.flatten()
        return self

    def copy(self):
        dm = DataMatrix(col_labels=self.col_labels, units=self.units, idx=self.idx)
        dm.array = self.array.copy()
        return dm

    def switch_categories_order(self, cat1="Categories1", cat2="Categories2"):
        if "Categories" not in cat1 or "Categories" not in cat2:
            raise ValueError(" You can only switch the order of two Categories")
        # Extract axis of cat1, cat2
        a1 = self.dim_labels.index(cat1)
        a2 = self.dim_labels.index(cat2)
        # Switch axis in array
        self.array = np.swapaxes(self.array, a1, a2)
        # Switch col_labels
        col1 = self.col_labels[cat1]
        col2 = self.col_labels[cat2]
        self.col_labels[cat1] = col2
        self.col_labels[cat2] = col1
        return

    def groupby(
        self, group_cols={}, dim=str, aggregation="sum", regex=False, inplace=False
    ):
        # Sum values in group, e.g.
        # dm.groupby({'road': ['LDV', '2W']}, dim='Categories1') sums LDV and 2W and calls it road
        # dm.groupby({'freight': 'HDV.*|marine.*', 'passenger': 'LDV|bus|aviation'}, dim='Categories2', regex = True)
        # It works also on Variables as long as they have the same unit
        # It works as well for Country and Years if need be
        i = 0
        for out_col, col_to_group in group_cols.items():
            # Extract only the dm with the categories or variables to group
            if regex:
                dm_to_group = self.filter_w_regex({dim: col_to_group})
            else:
                dm_to_group = self.filter({dim: col_to_group})
            # if inplace, drop col_to_group from self
            if inplace:
                self.drop(dim=dim, col_label=col_to_group)
            a = self.dim_labels.index(dim)  # extract the index of the dimension
            new_array = np.moveaxis(dm_to_group.array, a, -1)  # move dimension to end
            if aggregation == "sum":  # nansum
                new_array = np.nansum(new_array, axis=-1, keepdims=True)
            if aggregation == "mean":  # mean
                new_array = np.nanmean(new_array, axis=-1, keepdims=True)
            dm_to_group.array = np.moveaxis(
                new_array, -1, a
            )  # put dimension back to right place
            # remove the idx of the grouped columns
            for col in dm_to_group.col_labels[dim]:
                dm_to_group.idx.pop(col)
            # col_label[dim] should only contain the new name
            dm_to_group.col_labels[dim] = [out_col]
            # Add idx of new column name using iterator
            dm_to_group.idx[out_col] = i
            if dim == "Variables":
                # Check that all the variables have the same unit
                new_unit_set = set(dm_to_group.units.values())
                if len(new_unit_set) != 1:
                    raise ValueError(
                        f"the Variables {col_to_group} in groupby do not have the same unit"
                    )
                dm_to_group.units = {out_col: new_unit_set.pop()}
            if i == 0:
                dm_out = dm_to_group
            else:
                dm_out.append(dm_to_group, dim=dim)
            i = i + 1
        if inplace:
            self.append(dm_out, dim=dim)
            self.sort(dim=dim)
            return
        else:
            dm_out.sort(dim=dim)
            return dm_out

    def group_all(self, dim=str, inplace=True, aggregation="sum"):
        # Function to drop a dimension by summing all categories
        # Call example: dm_to_group.group_all(dim='Categories2', inplace=True)
        # or dm_grouped = dm_to_group.group_all(dim='Categories1', inplace=False)
        # when inplace = False dm_to_group remains unchanged and the grouped dm is return as output
        if "Categories" not in dim:
            raise ValueError(f"You can only use group_all() on Categories")
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
        categories_to_rename = [cat for cat in dm.dim_labels if "Categories" in cat]
        categories_to_rename.remove(dim)
        i = 1
        for old_cat in categories_to_rename:
            new_cat = "Categories" + str(i)
            dm.col_labels[new_cat] = dm.col_labels[old_cat]
            i = i + 1
        # Remove last category and dimension
        last_dim = dm.dim_labels[-1]
        dm.col_labels.pop(last_dim)
        dm.dim_labels = dm.dim_labels[:-1]
        if not inplace:
            return dm
        return

    def change_unit(self, var, factor, old_unit, new_unit, operator="*"):
        idx = self.idx
        if self.units[var] != old_unit:
            raise ValueError(f"The original unit is not {old_unit}")
        self.units[var] = new_unit
        if operator == "*":
            self.array[:, :, idx[var], ...] = self.array[:, :, idx[var], ...] * factor
        elif operator == "/":
            self.array[:, :, idx[var], ...] = self.array[:, :, idx[var], ...] / factor
        else:
            raise ValueError(f"Only * and / operators are possible in change_unit")
        return

    def datamatrix_plot(self, selected_cols={}, title="title", stacked=None):

        if stacked is not None:
            stacked = "one"

        dims = len(self.dim_labels)
        if (dims != 3) & (dims != 4):
            raise Exception(
                "plot function has been implemented only for DataMatrix with max one category"
            )

        i = self.idx

        plot_cols = self.col_labels.copy()

        for key, value in selected_cols.items():
            if value != "all":
                if isinstance(value, str):
                    plot_cols[key] = [value]
                else:
                    plot_cols[key] = value

        years_idx = [i[x] for x in plot_cols["Years"]]
        # Create an empty figure
        fig = px.line(
            x=plot_cols["Years"], labels={"x": "Years", "y": "Values"}, title=title
        )
        fig.data[0]["y"] = np.nan * np.ones(shape=np.shape(fig.data[0]["y"]))
        if dims == 3:
            for c in plot_cols["Country"]:
                for v in plot_cols["Variables"]:
                    y_values = self.array[i[c], years_idx, i[v]]
                    label = c + "_" + v
                    fig.add_scatter(
                        x=plot_cols["Years"],
                        y=y_values,
                        name=label,
                        mode="lines",
                        stackgroup=stacked,
                    )
        if dims == 4:
            for c in plot_cols["Country"]:
                for v in plot_cols["Variables"]:
                    for cat in plot_cols["Categories1"]:
                        y_values = self.array[i[c], years_idx, i[v], i[cat]]
                        label = c + "_" + v + "_" + cat
                        fig.add_scatter(
                            x=plot_cols["Years"],
                            y=y_values,
                            name=label,
                            mode="lines",
                            stackgroup=stacked,
                        )

        fig.show()

        return

    def normalise(self, dim, inplace=True, keep_original=False):

        # Axis over which to normalise
        a = self.dim_labels.index(dim)
        if inplace and not keep_original:
            # Overwrites datamatrix with normalised data and changes the units to '%'
            arr_sum = np.nansum(self.array, axis=a, keepdims=True)
            arr_sum = np.nan_to_num(arr_sum)
            with np.errstate(divide="ignore", invalid="ignore"):
                self.array = np.where(arr_sum != 0, self.array / arr_sum, np.nan)
            for v in self.col_labels["Variables"]:
                self.units[v] = "%"
        else:
            # Create a new normalised array
            arr_data = self.array.copy()
            arr_sum = np.nansum(arr_data, axis=a, keepdims=True)
            arr_sum = np.nan_to_num(arr_sum)
            with np.errstate(divide="ignore", invalid="ignore"):
                arr_data = np.where(arr_sum != 0, arr_data / arr_sum, np.nan)
            new_var_cols = [var + "_share" for var in self.col_labels["Variables"]]
            # Adds the normalised array to the existing data in the same database
            if inplace and keep_original:
                self.add(
                    arr_data,
                    dim="Variables",
                    col_label=new_var_cols,
                    unit=["%"] * len(new_var_cols),
                )
            if not inplace:
                units_new = {}
                vars_new = []
                for var in self.col_labels["Variables"]:
                    var_new = var + "_share"
                    vars_new.append(var_new)
                    units_new[var_new] = "%"
                dm_out = DataMatrix.based_on(
                    arr_data,
                    format=self,
                    change={"Variables": vars_new},
                    units=units_new,
                )
                return dm_out
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