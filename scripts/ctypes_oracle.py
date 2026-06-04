"""Ctypes-based oracle for bisect_pipeline_stages.

Provides CtypesOracle class that loads a compiled .dylib, runs forward pass via ctypes,
and compares against Python executor reference using cosine similarity.

Usage:
    from scripts.ctypes_oracle import CtypesOracle
    oracle = CtypesOracle(artifact_dir="compiled/opt_125m_fresh")
    cos = oracle.compare("path/to/model.dylib")
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys

import numpy as np

_log = logging.getLogger("ctypes_oracle")

# Ensure project root is on path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from compiler.sfcf_parser import (  # noqa: E402
    make_memref_descriptor,
    parse_compute_graph,
    parse_sfcf_blob,
    parse_sret_outputs,
    verify_output_shapes,
)
from scripts._cos import cosine_similarity  # noqa: E402


class CtypesOracle:
    """Load artifact weights + Python executor reference, compare any dylib against it.

    On init, parses the SFCF blob from the original compiled dylib and caches the
    compute graph.  Each call to ``compare(dylib_path)`` opens the test dylib,
    resolves kernel symbols against the cached graph, runs the forward pass via
    ctypes, and returns the cosine similarity against the Python executor reference.
    """

    # Matches ctypes_forward.py's default input
    INPUT_IDS = np.array([[2, 32826, 85, 4129], [0, 0, 0, 0]], dtype=np.int64)
    # 128KB for sret output buffer (matches ctypes_forward.py)
    SRET_SIZE = 131072

    def __init__(self, artifact_dir: str = "./compiled/opt_125m_fresh"):
        self.artifact_dir = os.path.abspath(artifact_dir)
        self._py_logits: np.ndarray | None = None
        self._load_sfcf_blob()
        self._load_artifact()
        self._load_reference()

        # Verify parameter binding consistency
        issues = self.verify_bindings()
        for msg in issues:
            _log.info("  %s", msg)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _load_sfcf_blob(self) -> None:
        """Parse and cache the SFCF blob from the original compiled dylib.

        The SFCF blob (compute graph / weight bindings) is embedded during
        the full compilation pipeline but is NOT present in dylibs produced
        by ``compile_with_stages`` (which only runs stages + LLVM backend).
        We therefore parse it once from the original dylib and reuse it.
        """
        orig_dylib = os.path.join(self.artifact_dir, "libopt_125m.dylib")
        if not os.path.exists(orig_dylib):
            raise FileNotFoundError(
                f"Original dylib not found at {orig_dylib} — "
                "compile the model first with scripts/compile.py"
            )

        lib = ctypes.CDLL(orig_dylib)
        try:
            data_ptr = ctypes.cast(
                ctypes.addressof(
                    ctypes.c_int64.in_dll(lib, "serveforge_constants_data")
                ),
                ctypes.c_void_p,
            )
            size_ptr = ctypes.cast(
                ctypes.addressof(
                    ctypes.c_int64.in_dll(lib, "serveforge_constants_size")
                ),
                ctypes.POINTER(ctypes.c_uint64),
            )
            blob_size = size_ptr[0]
            blob_bytes = bytes(
                (ctypes.c_uint8 * blob_size).from_address(data_ptr.value)
            )
        except (ValueError, AttributeError) as exc:
            raise RuntimeError(
                f"Failed to read SFCF blob from {orig_dylib}: {exc}"
            ) from exc

        self._name_mapping, self._sfcf_constants, graph_pos, sfcf_version = parse_sfcf_blob(
            blob_bytes
        )
        self._graph, _ = parse_compute_graph(blob_bytes, graph_pos, version=sfcf_version)
        _log.info(
            "Parsed SFCF blob: %d functions, %d constants, %d name mappings",
            len(self._graph["functions"]),
            len(self._sfcf_constants),
            len(self._name_mapping),
        )

    def _load_artifact(self) -> None:
        """Load all weight tensors from the compiled artifact."""
        from compiler.serialize import load_artifact

        artifact = load_artifact(self.artifact_dir)

        self.all_weights: dict[str, np.ndarray] = {}
        for func in artifact.functions:
            for wname, wtensor in func.weights.items():
                if wname not in self.all_weights:
                    self.all_weights[wname] = np.ascontiguousarray(wtensor.numpy())

        self.hf_key_map = artifact.metadata.get("hf_key_map", {})
        ws = artifact.metadata.get("weight_source", {})
        self.tied_weights = (
            ws.get("tied_weights", {}) or artifact.metadata.get("tied_weights", {})
        )
        self._artifact = artifact

        _log.info(
            "Loaded %d unique weight tensors from %s",
            len(self.all_weights),
            self.artifact_dir,
        )

    def _load_reference(self) -> None:
        """Load (or generate and cache) Python executor reference logits."""
        import torch

        cache_path = "/tmp/py_logits_batch2.npy"
        if os.path.exists(cache_path):
            self._py_logits = np.load(cache_path)
            _log.info("Loaded reference logits from %s", cache_path)
            return

        _log.info("Generating reference logits via MlirExecutor (may take a moment)...")
        from python_runtime.engine.mlir_executor import MlirExecutor
        from python_runtime.hal.pytorch_backend import PyTorchBackend

        backend = PyTorchBackend("cpu")
        executor = MlirExecutor(self._artifact, backend)
        with torch.no_grad():
            logits = executor.forward(torch.tensor(self.INPUT_IDS))
        self._py_logits = np.ascontiguousarray(logits.numpy().astype(np.float32))
        np.save(cache_path, self._py_logits)
        _log.info("Cached reference logits to %s", cache_path)

    # ------------------------------------------------------------------
    # Weight resolution  (mirrors ctypes_forward.main's multi-strategy)
    # ------------------------------------------------------------------

    def _get_weight(self, name: str) -> np.ndarray:
        if name in self.all_weights:
            return self.all_weights[name]
        hf_key = self.hf_key_map.get(name)
        if hf_key and hf_key in self.all_weights:
            return self.all_weights[hf_key]
        for alias_name, primary_name in self.tied_weights.items():
            if primary_name == name:
                alias_hf = self.hf_key_map.get(alias_name)
                if alias_hf and alias_hf in self.all_weights:
                    return self.all_weights[alias_hf]
                if alias_name in self.all_weights:
                    return self.all_weights[alias_name]
        bare_name = name.split(".", 1)[-1] if "." in name else name
        if bare_name != name:
            if bare_name in self.all_weights:
                return self.all_weights[bare_name]
            if bare_name in self._sfcf_constants:
                return np.ascontiguousarray(self._sfcf_constants[bare_name])
        prefixed = f"main_0.{name}"
        if prefixed in self.all_weights:
            return self.all_weights[prefixed]
        if name in self._sfcf_constants:
            return np.ascontiguousarray(self._sfcf_constants[name])
        raise KeyError(f"Weight '{name}' not found")

    # ------------------------------------------------------------------
    # Parameter binding verification
    # ------------------------------------------------------------------

    def dump_weight_binding(self, func_idx: int) -> None:
        """Print weight name at each input position for a given function.

        For func 0, this shows inputs[0] = global_input,
        inputs[1..] = weight names like model_decoder_embed_tokens_weight, etc.
        """
        func_def = self._graph["functions"][func_idx]
        print(f"\n=== Weight Bindings for func_{func_idx} ({func_def['symbol']}) ===")
        print(f"Total inputs: {func_def['num_inputs']}")
        for i, inp in enumerate(func_def["inputs"]):
            binding = inp["binding"]
            shape = inp["shape"]
            if binding[0] == "global_input":
                print(f"  input[{i:3d}]: GLOBAL_INPUT shape={shape}")
            elif binding[0] == "weight":
                key = binding[1]
                print(f"  input[{i:3d}]: WEIGHT '{key}' shape={shape}")
            elif binding[0] == "ssa":
                print(f"  input[{i:3d}]: SSA func_{binding[1]}_output[{binding[2]}] shape={shape}")

    def verify_bindings(self) -> list[str]:
        """Verify compute graph bindings are consistent.

        Returns list of diagnostic messages (empty = perfect).

        Checks:
        1. For each function, count inputs by type (weight/ssa/global_input)
        2. Try to resolve each weight: warn if not found
        3. Report total weight count for diagnostics
        """
        issues: list[str] = []
        for fi, func_def in enumerate(self._graph["functions"]):
            weight_count = sum(
                1 for inp in func_def["inputs"] if inp["binding"][0] == "weight"
            )
            ssa_count = sum(
                1 for inp in func_def["inputs"] if inp["binding"][0] == "ssa"
            )
            global_count = sum(
                1 for inp in func_def["inputs"] if inp["binding"][0] == "global_input"
            )

            total_inputs = len(func_def["inputs"])
            assert (
                weight_count + ssa_count + global_count == total_inputs
            ), f"Input type counts don't add up for func_{fi}"

            issues.append(
                f"func_{fi} ({func_def['symbol']}): "
                f"inputs={total_inputs}, "
                f"weights={weight_count}, "
                f"ssa={ssa_count}, "
                f"global={global_count}"
            )

            # Try to resolve each weight
            missing: list[str] = []
            for inp in func_def["inputs"]:
                if inp["binding"][0] == "weight":
                    try:
                        self._get_weight(inp["binding"][1])
                    except KeyError:
                        missing.append(inp["binding"][1])

            if missing:
                issues.append(
                    f"  !! func_{fi}: {len(missing)} unresolved weights: {missing[:5]}..."
                )

        return issues

    def diagnose_per_function(self, dylib_path: str | None = None) -> None:
        """Run forward pass and print per-function output diagnostics.

        Reports shape, min, max, mean, std for each function output.
        Flags suspicious values: all zeros, all same value, NaN.

        Args:
            dylib_path: Path to dylib (uses original if None).
        """
        if dylib_path is None:
            dylib_path = os.path.join(self.artifact_dir, "libmodel.dylib")
            if not os.path.exists(dylib_path):
                dylib_path = os.path.join(self.artifact_dir, "libopt_125m.dylib")

        # Run comparison to populate func_outputs
        self.compare(dylib_path)

        print(f"\n=== Per-function diagnostic ({os.path.basename(dylib_path)}) ===")
        for fi, outputs in enumerate(self._func_outputs):
            for oi, arr in enumerate(outputs):
                f = arr.ravel().astype(np.float64)
                flags = []
                if np.all(f == 0.0):
                    flags.append("ALL ZEROS")
                if len(f) >= 10 and np.all(f[:10] == f[0]):
                    flags.append("ALL SAME VALUE")
                if np.any(np.isnan(f)):
                    flags.append("HAS NaN")
                if np.any(np.isinf(f)):
                    flags.append("HAS INF")

                flag_str = " ⚠️ " + ", ".join(flags) if flags else ""
                print(f"  func_{fi}_output[{oi:2d}]: "
                      f"shape={list(arr.shape)}, "
                      f"min={f.min():.4f}, max={f.max():.4f}, "
                      f"mean={f.mean():.4f}, std={f.std():.4f}"
                      f"{flag_str}")

    # ------------------------------------------------------------------
    # Main comparison  (the public API)
    # ------------------------------------------------------------------

    def compare(self, dylib_path: str) -> float:
        """Load dylib, run forward pass via ctypes, return cos vs. Python reference.

        Args:
            dylib_path: Path to the compiled .dylib to test.

        Returns:
            Cosine similarity between dylib output and Python executor reference
            (1.0 = identical, 0.0 = completely different).

        Raises:
            FileNotFoundError: dylib does not exist.
            RuntimeError: Failed to run forward pass.
        """
        if not os.path.exists(dylib_path):
            raise FileNotFoundError(f"dylib not found: {dylib_path}")
        if self._py_logits is None:
            raise RuntimeError("Reference logits not loaded")

        lib = ctypes.CDLL(dylib_path)

        # Use the cached compute graph — only kernel symbols come from this dylib.
        func_outputs: list[list[np.ndarray]] = [
            [] for _ in range(len(self._graph["functions"]))
        ]

        for fi, func_def in enumerate(self._graph["functions"]):
            symbol = func_def["symbol"]
            try:
                kernel = getattr(lib, symbol)
            except AttributeError:
                _log.warning("[%2d] %s not found, skipping", fi, symbol)
                continue

            input_descs: list[ctypes.Structure] = []
            input_args: list[ctypes.pointer] = []
            _keep_arrs: list[np.ndarray] = []

            for inp in func_def["inputs"]:
                binding = inp["binding"]
                io_shape = inp["shape"]

                if binding[0] == "global_input":
                    arr = self.INPUT_IDS
                elif binding[0] == "weight":
                    key = binding[1]
                    try:
                        arr = self._get_weight(key)
                    except KeyError:
                        shape = (
                            [int(s) if s > 0 else 1 for s in io_shape] or [1]
                        )
                        arr = np.zeros(shape, dtype=np.float32)
                        _log.warning(
                            "[%2d] weight '%s' not found, zeros shape=%s",
                            fi, key, shape,
                        )
                elif binding[0] == "ssa":
                    pf, oi = binding[1], binding[2]
                    if (
                        pf < len(func_outputs)
                        and oi < len(func_outputs[pf])
                    ):
                        arr = func_outputs[pf][oi]
                    else:
                        shape = (
                            [int(s) if s > 0 else 1 for s in io_shape] or [1]
                        )
                        arr = np.zeros(shape, dtype=np.float32)
                else:
                    raise ValueError(f"Unknown binding: {binding}")

                if arr.dtype != np.float32 and arr.dtype != np.int64:
                    arr = arr.astype(np.float32)
                if not arr.flags["C_CONTIGUOUS"]:
                    arr = np.ascontiguousarray(arr)

                _keep_arrs.append(arr)
                desc = make_memref_descriptor(arr)
                input_descs.append(desc)
                input_args.append(ctypes.byref(desc))

            sret = (ctypes.c_uint8 * self.SRET_SIZE)()
            all_args = [ctypes.byref(sret)] + input_args
            kernel.argtypes = [ctypes.c_void_p] * len(all_args)
            kernel.restype = None
            kernel(*all_args)

            outputs = parse_sret_outputs(bytes(sret), func_def["outputs"])
            func_outputs[fi] = outputs

        # ── Structural sret shape verification ────────────────────────
        shape_errors = verify_output_shapes(
            func_outputs, self._graph["functions"]
        )
        for err in shape_errors:
            _log.warning("sret shape mismatch: %s", err)

        go_func, go_idx = self._graph["global_output"]
        ctypes_logits = func_outputs[go_func][go_idx]
        self._func_outputs = func_outputs
        return cosine_similarity(self._py_logits, ctypes_logits)
