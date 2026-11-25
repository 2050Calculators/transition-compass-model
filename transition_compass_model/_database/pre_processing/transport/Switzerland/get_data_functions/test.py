import json

import requests

from transition_compass_model._database.pre_processing.constants import CANTONS_NAME


def get_asset_ids_2015(cantons_name):
    """
    Retrieves the DAM asset IDs for the 2015 MZ Excel files.
    """

    url = "https://dam-api.bfs.admin.ch/hub/api/dam/assets"

    asset_ids = {}

    for canton, acronym in cantons_name.items():
        params = {
            "language": "de",
            "extendedSearch": f"su-f-11.04.03-MZ-2015-A2_{acronym}",
            "limit": 1000,
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        assets = response.json().get("data", [])

        # Search for the corresponding Excel file
        for asset in assets:
            asset_text = json.dumps(asset, ensure_ascii=False)

            if f"A2_{acronym}" in asset_text:
                asset_id = asset.get("ids", {}).get("damId")

                if asset_id:
                    asset_ids[canton] = str(asset_id)
                    break

        if canton not in asset_ids:
            print(f"Asset not found for {canton} ({acronym})")

    return asset_ids


asset_id_2015 = get_asset_ids_2015(CANTONS_NAME["name_to_accronym"])
