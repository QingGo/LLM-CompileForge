"""Shared helper for module-level lazy __getattr__ imports.

Usage in __init__.py:
    from compiler._lazy_imports import lazy_imports

    lazy_imports(__name__, globals(), {
        "attr_name": ("module.path", "symbol_name"),
    })
"""

from __future__ import annotations

from typing import Any


def lazy_imports(
    module_name: str,
    module_dict: dict[str, Any],
    attr_map: dict[str, tuple[str, str]],
) -> None:
    """Install lazy __getattr__ on a module.

    Args:
        module_name: The __name__ of the calling module (for error messages).
        module_dict: The calling module's __dict__ (pass globals() at call site).
        attr_map:  {public_attr: (full_module_path, symbol_name_in_module)}
    """
    __all__ = list(attr_map.keys())
    module_dict["__all__"] = __all__
    module_dict["_LAZY_ATTRS"] = frozenset(__all__)

    # Group by module path so each dependent module is imported at most once.
    _mod_to_attrs: dict[str, list[tuple[str, str]]] = {}
    for attr, (mod_path, sym_name) in attr_map.items():
        _mod_to_attrs.setdefault(mod_path, []).append((attr, sym_name))

    def _getattr(name: str) -> Any:
        if name not in module_dict["_LAZY_ATTRS"]:
            raise AttributeError(f"module '{module_name}' has no attribute '{name}'")

        _globals: dict[str, Any] = {}
        for mod_path, attrs in _mod_to_attrs.items():
            mod = __import__(mod_path, fromlist=[s for _, s in attrs])
            for attr_name, sym_name in attrs:
                _globals[attr_name] = getattr(mod, sym_name)

        value = _globals[name]
        module_dict[name] = value
        return value

    module_dict["__getattr__"] = _getattr
