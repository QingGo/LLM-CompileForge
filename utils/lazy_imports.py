"""Lazy import utility shared by all __init__.py modules.

Usage in any __init__.py:
    from utils.lazy_imports import make_lazy_getattr, LAZY_IMPORTS

    LAZY_IMPORTS["MyClass"] = ("engine.my_module", "MyClass")

    def __getattr__(name: str):
        return make_lazy_getattr(__name__, name)
"""

from __future__ import annotations

import importlib
from typing import Any


def make_lazy_getattr(module_name: str, attr_name: str) -> Any:
    """Resolve a lazy import from LAZY_IMPORTS and cache on the module.

    Args:
        module_name: The __name__ of the calling module.
        attr_name: The attribute being accessed.
    """
    import sys
    mod = sys.modules[module_name]
    if attr_name in LAZY_IMPORTS:
        pkg, cls = LAZY_IMPORTS[attr_name]
        imported = importlib.import_module(pkg)
        obj = getattr(imported, cls)
        setattr(mod, attr_name, obj)
        return obj
    raise AttributeError(
        f"module '{module_name}' has no attribute '{attr_name}'"
    )


# Per-module lazy import registries.  Each __init__.py manages its own
# set of deferred imports.

LAZY_IMPORTS: dict[str, tuple[str, str]] = {}
