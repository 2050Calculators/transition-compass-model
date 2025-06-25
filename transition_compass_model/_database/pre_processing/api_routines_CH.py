import requests
import numpy as np
from model.common.data_matrix_class import DataMatrix


def json_to_dm(data_json, mapping_dims, mapping_vars, units):

    def get_col_labels_json(json_data):
        # Number of dimensions in the key
        num_dimensions = len(json_data['data'][0]['key'])
        # Initialize a list of sets to store unique elements for each dimension
        dimension_sets = [set() for _ in range(num_dimensions)]

        # Iterate through each dictionary in the JSON data
        for entry in json_data['data']:
            key = entry['key']
            # Update each dimension's set with the current key's elements
            for i in range(num_dimensions):
                dimension_sets[i].add(key[i])

        # Convert sets to lists for the final output
        dimension_lists = [sorted(list(dimension_set)) for dimension_set in dimension_sets]
        return dimension_lists

    def from_json_to_numpy(json_data, json_col_labels):
        # Turn json structure into numpy array
        arr_shape = tuple(len(l) for l in json_col_labels)
        arr = np.empty(arr_shape, dtype=float)  # Ensure dtype supports NaN
        arr.fill(np.nan)  # Default all values to NaN

        for elem in json_data['data']:
            key = elem['key']
            value = elem['values'][0]
            # Match key with position in numpy array
            idx = [json_col_labels[i].index(k) for i, k in enumerate(key)]

            try:
                arr[tuple(idx)] = float(value)
            except ValueError:
                arr[tuple(idx)] = np.nan  # Assign NaN if conversion fails

        return arr

    def get_col_labels_axis_dim(json_col_labels, var_mapping, dim_mapping):
        # Maps json col_labels to dm col_labels and extracts the dim axis
        map_tmp = dict()
        for i, cols_json in enumerate(json_col_labels):
            dim_json = data_json['columns'][i]['text']
            # dict for dimension 'dim_json' to map columns from json indexes to dm col names
            cols_map = var_mapping[dim_json]
            cols_dm = [cols_map[col] for col in cols_json]
            map_tmp[dim_json] = {'cols': cols_dm, 'axis': i}

        col_labels = {}
        dim_axis = {}
        for dim_dm, dim_json in dim_mapping.items():
            col_labels[dim_dm] = map_tmp[dim_json]['cols']
            dim_axis[dim_dm] = map_tmp[dim_json]['axis']

        # Turn years to int
        col_labels['Years'] = [int(y) for y in col_labels['Years']]

        return col_labels, dim_axis

    # Get col labels from json data
    col_labels_json = get_col_labels_json(data_json)

    # From json structure to numpy array (with dimensions ordered like json)
    arr_raw = from_json_to_numpy(data_json, col_labels_json)

    # Get dm col_labels structure and a dictionary with the dims axis
    col_labels, dim_axis = get_col_labels_axis_dim(col_labels_json, mapping_vars, mapping_dims)

    unit_vars = {}
    for i, var in enumerate(col_labels['Variables']):
        unit_vars[var] = units[i]

    dm = DataMatrix(col_labels, unit_vars)
    del i, var, unit_vars

    # Re-order the axis of the numpy array
    dim_labels = dm.dim_labels
    new_order = list(range(len(arr_raw.shape)))
    arr_shape = []
    for axis, dim in enumerate(dim_labels):
        axis_json = dim_axis[dim]
        new_order[axis] = axis_json
        arr_shape.append(len(col_labels[dim]))

    # Create a final_order list to capture the correct permutation
    # For unmapped dimensions, keep their positions in the remaining slots
    mapped_axes = set(dim_axis.values())
    unmapped_axes = [axis for axis in range(len(arr_raw.shape)) if axis not in mapped_axes]

    # Fill final_order respecting the original position for unmapped axes
    final_order = [None] * len(arr_raw.shape)
    used_axes = set()

    # Assign mapped axes first
    for axis, dim in enumerate(dim_labels):
        final_order[axis] = dim_axis[dim]
        used_axes.add(dim_axis[dim])

    # Assign remaining unmapped axes to the available slots
    for i in range(len(final_order)):
        if final_order[i] is None:
            final_order[i] = unmapped_axes.pop(0)

    arr = np.transpose(arr_raw, axes=final_order)
    # Some over unmapped_axes
    nb_axis_unmapped = len(final_order) - len(mapped_axes)
    for i in range(nb_axis_unmapped):
        i = i + 1
        arr = np.nansum(arr, axis=-i, keepdims=True)

    arr = arr.reshape(tuple(arr_shape))

    dm.array = arr

    return dm


def get_data_api_CH(table_id, mode='example', filter=dict(), mapping_dims=dict(), units=[], language='en'):
    # Define the base URL and the specific table_id for the API endpoint
    base_url = "https://www.pxweb.bfs.admin.ch/api/v1"
    base_url_lan = f"{base_url}/{language}"
    url = f"{base_url_lan}/{table_id}/{table_id}.px"
    response_structure = requests.get(url)
    data_structure = response_structure.json()

    # Give as output the structure
    if mode == 'example':
        structure = {}
        for elem in data_structure['variables']:
            structure[elem['text']] = elem['valueTexts']
        title = data_structure['title']
        return structure, title
    # Extract data
    if mode == 'extract':
        if len(filter) == 0:
            raise ValueError(
                'You need to provide the parameters you want to extract as a dictionary based on the structure')
        query = []  # List of  dictionaries
        mapping = {}
        for elem in data_structure['variables']:
            extract = {}  # Dictionary with key 'code' and 'selection'
            extract['code'] = elem['code']  # Extract code name
            key = elem['text']  # Match element with input filter dictionary
            valuetext = filter[key]
            if isinstance(valuetext, str):
                index = elem['valueTexts'].index(valuetext)
                value = [elem['values'][index]]
            else:
                index = [elem['valueTexts'].index(val) for val in valuetext]
                value = [elem['values'][i] for i in index]
            mapping[elem['text']] = {v: vt for v, vt in
                                     zip(value, valuetext if isinstance(valuetext, list) else [valuetext])}
            extract['selection'] = {'filter': 'item', 'values': value}
            query.append(extract)
        payload = {
            "query": query,
            "response": {"format": "json"}
        }

        response = requests.post(url, json=payload)
        if response.status_code == 200:
            # Parse and print the JSON response
            data = response.json()
            dm = json_to_dm(data, mapping_dims, mapping_vars=mapping, units=units)
            return dm
        else:
            print(f"Failed to retrieve data: {response.status_code}")
            print(response.text)
            return
