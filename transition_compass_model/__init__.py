"""Transition Compass Model module."""

import sys as _sys

from . import _database
from . import model


def _register_compat_aliases():
    """Register old-style 'model.*' module aliases for pre-refactor pickle files.

    Pickle files created before the package rename reference classes under the
    path 'model.common.*'. Registering aliases here lets callers use plain
    pickle.load() without a custom Unpickler.
    """
    import transition_compass_model.model.common.data_matrix_class  # noqa: F401

    _mappings = {
        "model": "transition_compass_model.model",
        "model.common": "transition_compass_model.model.common",
        "model.common.data_matrix_class": "transition_compass_model.model.common.data_matrix_class",
    }
    for alias, real in _mappings.items():
        _sys.modules.setdefault(alias, _sys.modules[real])


_register_compat_aliases()

__all__ = [
    "_database",
    "model",
]
