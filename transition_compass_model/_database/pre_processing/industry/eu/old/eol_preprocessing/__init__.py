"""Preprocessing for Eol Preprocessing."""

# Note: OTS-FTS cannot be imported due to hyphen in filename
# from . import OTS-FTS
from . import TV_PV
from . import elv

# Note: larger-appliances cannot be imported due to hyphen in filename
# from . import larger-appliances
# Note: packaging-waste cannot be imported due to hyphen in filename
# from . import packaging-waste
# Note: pc-and-electronics cannot be imported due to hyphen in filename
# from . import pc-and-electronics

__all__ = [
    # "OTS-FTS",  # Cannot import due to hyphen
    "TV_PV",
    "elv",
    # "larger-appliances",  # Cannot import due to hyphen
    # "packaging-waste",  # Cannot import due to hyphen
    # "pc-and-electronics",  # Cannot import due to hyphen
]
