import numpy as np
import requests

from transition_compass_model.model.common.data_matrix_class import DataMatrix


def sdmx_json_to_dm(data_json, mapping_dims, units, filter=None):
    structure = data_json["data"]["structures"][0]
    dataset = data_json["data"]["dataSets"][0]

    # Series dims are ordered by keyPosition, observation dims come after (usually just TIME_PERIOD).
    # A series key like "0:1:1:1:0:1" and an observation key like "3" are both indices into these lists.
    all_dims = (
        structure["dimensions"]["series"] + structure["dimensions"]["observation"]
    )

    raw_shape = tuple(len(dim["values"]) for dim in all_dims)
    arr_raw = np.full(raw_shape, np.nan, dtype=float)

    for series_key, series_val in dataset["series"].items():
        idx_series = tuple(int(i) for i in series_key.split(":"))
        for obs_key, obs_val in series_val["observations"].items():
            idx_obs = tuple(int(i) for i in obs_key.split(":"))
            value = obs_val[0]
            if value is not None:
                arr_raw[idx_series + idx_obs] = float(value)

    id_to_axis = {dim["id"]: axis for axis, dim in enumerate(all_dims)}
    id_to_labels = {dim["id"]: [v["name"] for v in dim["values"]] for dim in all_dims}

    # The server can return rows beyond what was filtered for - e.g. requesting a
    # hierarchical dimension's child code (like "Passenger cars") also pulls back its
    # parent/aggregate rows ("Motor vehicles", "Total"). Trim each filtered dimension's
    # raw axis down to exactly what was asked for before anything downstream (the units
    # lookup, or nansum-collapsing an unmapped axis) can be corrupted by them - an extra
    # "Total" row surviving into a summed axis would silently double-count real values.
    for dim_id, wanted in (filter or {}).items():
        if dim_id not in id_to_axis:
            continue
        wanted = [wanted] if isinstance(wanted, str) else wanted
        axis = id_to_axis[dim_id]
        labels = id_to_labels[dim_id]
        keep = [i for i, label in enumerate(labels) if label in wanted]
        arr_raw = np.take(arr_raw, keep, axis=axis)
        id_to_labels[dim_id] = [labels[i] for i in keep]

    col_labels = {}
    dim_axis = {}
    for dim_dm, dim_id in mapping_dims.items():
        col_labels[dim_dm] = id_to_labels[dim_id]
        dim_axis[dim_dm] = id_to_axis[dim_id]

    if "Years" in col_labels:
        col_labels["Years"] = [int(y) for y in col_labels["Years"]]

    unit_vars = {var: units[i] for i, var in enumerate(col_labels["Variables"])}

    dm = DataMatrix(col_labels, unit_vars)

    # Re-order the raw axes to match dm.dim_labels, and collapse (sum) any axis that wasn't
    # mapped to a dm dimension, mirroring the approach in api_routines_CH.json_to_dm.
    dim_labels = dm.dim_labels
    mapped_axes = set(dim_axis.values())
    unmapped_axes = [a for a in range(len(all_dims)) if a not in mapped_axes]

    final_order = [dim_axis[dim] for dim in dim_labels] + unmapped_axes
    arr = np.transpose(arr_raw, axes=final_order)

    for i in range(len(unmapped_axes)):
        arr = np.nansum(arr, axis=-(i + 1), keepdims=True)

    arr_shape = tuple(len(col_labels[dim]) for dim in dim_labels)
    arr = arr.reshape(arr_shape)

    dm.array = arr
    dm.sort("Years")

    return dm


# Dimension codes/labels come from the dataflow's metadata (DSD + codelists), not from
# /rest/data: a data pull can be huge (some dataflows are 50-100MB+ of observations),
# while the metadata needed to build a filter is typically far smaller since it scales
# with the number of dimensions/codes, not with the number of observations.
_STRUCTURE_CACHE = {}


def _fetch_structure(agency, dataflow, headers):
    cache_key = (agency, dataflow, headers.get("Accept-Language"))
    if cache_key not in _STRUCTURE_CACHE:
        url = (
            f"https://disseminate.stats.swiss/rest/dataflow/{agency}/{dataflow}/latest"
        )
        struct_headers = dict(
            headers, Accept="application/vnd.sdmx.structure+json;version=1.0"
        )
        response = requests.get(
            url, params={"references": "all"}, headers=struct_headers
        )
        data = response.json()["data"]
        dsd = data["dataStructures"][0]
        dim_list = dsd["dataStructureComponents"]["dimensionList"]
        codelists = {cl["id"]: cl for cl in data.get("codelists", [])}

        # Codelists referenced by the DSD are shared master lists (e.g. a generic economic-
        # activity classification with hundreds of codes), not the handful actually used by
        # this dataflow. The "Actual" content constraint records exactly which codes (and
        # which TIME_PERIOD range) have real observations - it's the metadata equivalent of
        # what /rest/data's own structure would report, without downloading the data.
        actual_codes = {}
        time_range = None
        for cc in data.get("contentConstraints", []):
            if cc.get("type") != "Actual":
                continue
            for kv in cc["cubeRegions"][0]["keyValues"]:
                if "values" in kv:
                    actual_codes[kv["id"]] = set(kv["values"])
                elif kv["id"] == "TIME_PERIOD":
                    time_range = kv.get("timeRange")

        def codelist_values(dim):
            enum_urn = dim.get("localRepresentation", {}).get("enumeration")
            if not enum_urn:
                # e.g. TIME_PERIOD: not a coded dimension, has no codelist at all
                return []
            codelist_id = enum_urn.rsplit(":", 1)[-1].split("(")[0]
            codes = codelists.get(codelist_id, {}).get("codes", [])
            allowed = actual_codes.get(dim["id"])
            if allowed is not None:
                codes = [c for c in codes if c["id"] in allowed]
            return [{"id": c["id"], "name": c["name"]} for c in codes]

        # Makes sure that the dimension are extracted in the correct order based on the
        # position argument
        series_dims = [
            {"id": dim["id"], "values": codelist_values(dim)}
            for dim in sorted(dim_list["dimensions"], key=lambda d: d["position"])
        ]

        observation_dims = []
        for dim in dim_list.get("timeDimensions", []):
            values = []
            if dim["id"] == "TIME_PERIOD" and time_range:
                # Best-effort preview assuming annual data; the actual extract always reads
                # the real periods straight off the /rest/data response, not from this list.
                start_year = int(time_range["startPeriod"]["period"][:4])
                end_year = int(time_range["endPeriod"]["period"][:4])
                values = [
                    {"id": str(y), "name": str(y)}
                    for y in range(start_year, end_year + 1)
                ]
            observation_dims.append({"id": dim["id"], "values": values})

        _STRUCTURE_CACHE[cache_key] = {
            "name": data["dataflows"][0]["name"],
            "dimensions": {"series": series_dims, "observation": observation_dims},
        }
    return _STRUCTURE_CACHE[cache_key]


def get_data_api_swiss_stats(
    agency,
    dataflow,
    mode="example",
    filter=dict(),
    mapping_dims=dict(),
    units=[],
    language="en",
):
    # Swiss Stats SDMX REST API: https://www.bfs.admin.ch (dissemination endpoint replacing STAT-TAB/PX-Web)
    base_url = "https://disseminate.stats.swiss/rest/data"
    headers = {"Accept-Language": language}

    # Give as output the structure
    if mode == "example":
        # Note: TIME_PERIOD (and any other non-coded dimension) comes back with an empty
        # value list here, since it has no fixed codelist - only /rest/data knows which
        # periods actually exist. Use the "startPeriod"/"endPeriod" filter keys for that.
        structure = _fetch_structure(agency, dataflow, headers)
        all_dims = (
            structure["dimensions"]["series"] + structure["dimensions"]["observation"]
        )
        structure_out = {
            dim["id"]: [v["name"] for v in dim["values"]] for dim in all_dims
        }
        title = structure["name"]
        return structure_out, title

    # Extract data
    if mode == "extract":
        if len(filter) == 0:
            raise ValueError(
                "You need to provide the parameters you want to extract as a dictionary based on the structure"
            )

        # Translate filter value names into codes, and get the series dimensions' order
        # (needed to build the dot-separated SDMX key below).
        structure = _fetch_structure(agency, dataflow, headers)
        series_dims = structure["dimensions"]["series"]

        # Build the SDMX key: one dot-separated segment per series dimension, in order.
        # A segment lists the wanted value codes joined by '+' (OR), or is left empty for "all".
        key_segments = []
        for dim in series_dims:
            if dim["id"] not in filter:
                key_segments.append("")
                continue
            wanted = filter[dim["id"]]
            wanted = [wanted] if isinstance(wanted, str) else wanted
            name_to_id = {v["name"]: v["id"] for v in dim["values"]}
            key_segments.append("+".join(name_to_id[w] for w in wanted))
        key = ".".join(key_segments)

        query_params = {"format": "jsondata"}
        if "startPeriod" in filter:
            query_params["startPeriod"] = filter["startPeriod"]
        if "endPeriod" in filter:
            query_params["endPeriod"] = filter["endPeriod"]

        url = f"{base_url}/{agency},{dataflow}/{key}"
        response = requests.get(url, params=query_params, headers=headers)
        if response.status_code == 200:
            data_json = response.json()
            dm = sdmx_json_to_dm(data_json, mapping_dims, units, filter=filter)
            return dm
        else:
            print(f"Failed to retrieve data: {response.status_code}")
            print(response.text)
            return
