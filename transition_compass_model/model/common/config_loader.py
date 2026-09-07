import json
from pathlib import Path


def get_lever_config_path():
    """Get lever_position.json path - works everywhere."""
    return Path(__file__).parent.parent.parent / "config" / "lever_position.json"


def load_lever_config():
    """Load lever configuration from lever_position.json."""
    with open(get_lever_config_path()) as f:
        return json.load(f)[0]
