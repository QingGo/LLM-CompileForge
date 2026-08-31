"""Config validation for golden test configs.json.

Validates the schema of tests/data/golden/npy/<model>/configs.json files.
Checks required fields, data types, and file existence of referenced paths.
Uses Python stdlib only — no external dependencies.
"""
import json
import sys
from pathlib import Path
from typing import Any


def _find_workspace_root(start: Path) -> Path:
    """Find workspace root by locating sf-dialect/ directory."""
    current = start.resolve()
    for _ in range(10):
        if (current / "sf-dialect").is_dir():
            return current
        if current.parent == current:
            break
        current = current.parent
    # Fallback: assume cwd is workspace root
    return Path.cwd()


def _fail(msg: str) -> None:
    """Print error message and exit non-zero."""
    print(f"Config validation FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def validate(config_path: str) -> None:
    """Validate a golden test config JSON file.

    Checks:
      1. Valid JSON syntax
      2. Required top-level keys: model, dylib_path, min_cos_threshold, cases
      3. cases is non-empty list
      4. Each case has name (str), seq_len (positive int), desc (str)
      5. If dylib_path is not null, file exists (relative to workspace root)
      6. If safetensors_path is not null, file exists

    Prints "Config validation PASSED" on success, exits non-zero on failure.
    """
    path = Path(config_path)
    if not path.is_file():
        _fail(f"config file not found: {config_path}")

    # Load JSON
    try:
        with open(path) as f:
            config: dict[str, Any] = json.load(f)
    except json.JSONDecodeError as e:
        _fail(f"invalid JSON: {e}")
    except Exception as e:
        _fail(f"failed to read config: {e}")

    # Check required top-level keys
    required_keys = {"model", "dylib_path", "min_cos_threshold", "cases"}
    missing = required_keys - set(config.keys())
    if missing:
        _fail(f"missing required keys: {sorted(missing)}")

    # Check cases is non-empty list
    cases = config.get("cases")
    if not isinstance(cases, list):
        _fail(f"'cases' must be a list, got {type(cases).__name__}")
    if len(cases) == 0:
        _fail("'cases' must be non-empty")

    # Validate each case
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            _fail(f"case[{i}]: must be a dict, got {type(case).__name__}")

        # name: required string
        name = case.get("name")
        if name is None:
            _fail(f"case[{i}]: missing required field 'name'")
        if not isinstance(name, str):
            _fail(f"case[{i}]: 'name' must be a string, got {type(name).__name__}")

        # seq_len: required positive int
        seq_len = case.get("seq_len")
        if seq_len is None:
            _fail(f"case[{i}]: missing required field 'seq_len'")
        if not isinstance(seq_len, int) or isinstance(seq_len, bool):
            _fail(f"case[{i}]: 'seq_len' must be an int, got {type(seq_len).__name__}")
        if seq_len <= 0:
            _fail(f"case[{i}]: 'seq_len' must be positive, got {seq_len}")

        # desc: required string
        desc = case.get("desc")
        if desc is None:
            _fail(f"case[{i}]: missing required field 'desc'")
        if not isinstance(desc, str):
            _fail(f"case[{i}]: 'desc' must be a string, got {type(desc).__name__}")

    # Resolve workspace root for file existence checks
    ws_root = _find_workspace_root(path.parent)

    # Check dylib_path file existence (if not null)
    dylib_path = config.get("dylib_path")
    if dylib_path is not None:
        if not isinstance(dylib_path, str):
            _fail(f"'dylib_path' must be a string or null, got {type(dylib_path).__name__}")
        resolved = ws_root / dylib_path
        if not resolved.is_file():
            _fail(f"dylib not found: {resolved} (from dylib_path={dylib_path})")

    # Check safetensors_path file existence (if not null)
    safetensors_path = config.get("safetensors_path")
    if safetensors_path is not None:
        if not isinstance(safetensors_path, str):
            _fail(f"'safetensors_path' must be a string or null, got {type(safetensors_path).__name__}")
        resolved = ws_root / safetensors_path
        if not resolved.is_file():
            _fail(f"safetensors not found: {resolved} (from safetensors_path={safetensors_path})")

    print("Config validation PASSED")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <config_path>", file=sys.stderr)
        sys.exit(1)
    validate(sys.argv[1])
