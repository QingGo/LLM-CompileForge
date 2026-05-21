"""Accuracy checker: HF reference vs Python executor vs Rust .dylib.

Dumps baselines to .npy files and computes cosine similarity between all pairs.
Run after any pipeline change to verify compilation correctness.

Usage:
    python scripts/check_accuracy.py                          # full comparison
    python scripts/check_accuracy.py --save-baselines          # save HF + Python refs
    python scripts/check_accuracy.py --compare-rust            # compare Rust CSV vs baselines
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    return float(np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12))


def _patch_transformers_torch():
    import transformers.utils.generic as _generic
    import transformers.utils.import_utils as _iu
    _iu._torch_available = True
    _iu._torch_version = torch.__version__
    _generic._torch_pytree = torch.utils._pytree
    def _flatten(output):
        return list(output.values()), list(output.keys())
    def _unflatten(values, context, output_type=None):
        return (output_type or type(context[0]))(**dict(zip(context, values, strict=False)))
    _generic._model_output_flatten = _flatten
    _generic._model_output_unflatten = _unflatten


def load_hf_opt_125m():
    _patch_transformers_torch()
    from transformers.models.opt.configuration_opt import OPTConfig
    from transformers.models.opt.modeling_opt import OPTForCausalLM
    hub_dir = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--opt-125m")
    snapshots = os.path.join(hub_dir, "snapshots")
    snap = os.listdir(snapshots)[0]
    model_path = os.path.join(snapshots, snap, "pytorch_model.bin")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    config_path = os.path.join(snapshots, snap, "config.json")
    config = OPTConfig.from_pretrained(config_path)
    model = OPTForCausalLM(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def get_input_ids():
    return torch.tensor([[2, 32826, 85, 4129], [0, 0, 0, 0]], dtype=torch.long)


def get_rust_input_ids():
    return [2, 32826, 85, 4129, 0, 0, 0, 0]


def test_python_executor(artifact_dir: str) -> np.ndarray:
    from compiler.serialize import load_artifact
    from engine.mlir_executor import MlirExecutor
    from hal.pytorch_backend import PyTorchBackend
    module = load_artifact(artifact_dir)
    backend = PyTorchBackend("cpu")
    executor = MlirExecutor(module, backend)
    input_ids = get_input_ids()
    logits = executor.forward(input_ids)
    return logits.detach().numpy()


def test_hf_reference() -> np.ndarray:
    model = load_hf_opt_125m()
    input_ids = get_input_ids()
    with torch.no_grad():
        output = model(input_ids)
    return output.logits.detach().numpy()


def run_rust_executor(dylib_dir: str) -> np.ndarray:
    result = subprocess.run(
        [sys.executable, "-c", f"""
import sys; sys.path.insert(0, '.')
from compiler.mlir_artifact import MlirModule, _parse_mlir_text
import os

dylib_path = os.path.join('{dylib_dir}', 'libopt_125m.dylib')
st_path = os.path.expanduser('~/.cache/huggingface/hub/models--facebook--opt-125m/snapshots')
snap = os.listdir(st_path)[0]
st_file = os.path.join(st_path, snap, 'pytorch_model.bin')

import ctypes, numpy as np

lib = ctypes.CDLL(dylib_path)

# main_0 ciface: sret + inputs
main_0 = lib._mlir_ciface_main_0
input_ids = (ctypes.c_int64 * 8)(2, 32826, 85, 4129, 0, 0, 0, 0)

sret = ctypes.create_string_buffer(65536)
main_0(sret, ctypes.byref(input_ids))
print('main_0 called')
"""],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-baselines", action="store_true", help="Save HF + Python baseline .npy files")
    parser.add_argument("--compare-rust", type=str, default=None, help="Rust logits CSV file to compare")
    parser.add_argument("--artifact-dir", type=str, default="./compiled/opt_125m_v8", help="Compiled model directory")
    args = parser.parse_args()

    artifact_dir = args.artifact_dir
    baseline_dir = os.path.join(artifact_dir, "baselines")
    os.makedirs(baseline_dir, exist_ok=True)

    if args.save_baselines or args.compare_rust is None:
        print("\n=== Reference: HuggingFace ===")
        hf_logits = test_hf_reference()
        hf_path = os.path.join(baseline_dir, "hf_logits.npy")
        np.save(hf_path, hf_logits)
        print(f"  Shape: {hf_logits.shape}, First: {hf_logits[0,0,0]:.6f}, Saved: {hf_path}")

        print("\n=== Python Executor (sf dialect) ===")
        py_logits = test_python_executor(artifact_dir)
        py_path = os.path.join(baseline_dir, "python_executor_logits.npy")
        np.save(py_path, py_logits)
        print(f"  Shape: {py_logits.shape}, First: {py_logits[0,0,0]:.6f}, Saved: {py_path}")

        sim = cosine_similarity(hf_logits, py_logits)
        print(f"\n  Cosine(Python Executor vs HF): {sim:.10f}")
        if sim < 0.999:
            print("  ⚠ WARNING: Python executor below 0.999 threshold")
        else:
            print("  ✅ PASS: Python executor > 0.999")
    else:
        hf_logits = np.load(os.path.join(baseline_dir, "hf_logits.npy"))
        py_logits = np.load(os.path.join(baseline_dir, "python_executor_logits.npy"))

    if args.compare_rust:
        print(f"\n=== Rust Executor (from {args.compare_rust}) ===")
        rust_csv = np.loadtxt(args.compare_rust, delimiter=",")
        # Rust model has static batch=2; Python/HF have batch=1.
        # Compare Rust row 0 (same input tokens) against Python/HF.
        if rust_csv.ndim == 1 and rust_csv.size == hf_logits.size:
            rust_logits = rust_csv.reshape(hf_logits.shape)
        elif rust_csv.ndim == 1:
            # Try reshaping as [2, 4, 50272] and take first batch
            b2_logits = rust_csv.reshape(2, 4, 50272)
            rust_logits = b2_logits[0:1]  # (1, 4, 50272)
        else:
            rust_logits = rust_csv[:1]
        rust_path = os.path.join(baseline_dir, "rust_logits.npy")
        np.save(rust_path, rust_logits)
        print(f"  Shape: {rust_logits.shape}, First: {rust_logits[0,0,0]:.6f}")

        sim_vs_hf = cosine_similarity(hf_logits, rust_logits)
        sim_vs_py = cosine_similarity(py_logits, rust_logits)
        print(f"  Cosine(Rust vs HF):              {sim_vs_hf:.10f}")
        print(f"  Cosine(Rust vs Python Executor):  {sim_vs_py:.10f}")
        if sim_vs_hf > 0.999:
            print("  ✅ Rust executor matches HF reference")
        elif sim_vs_py > 0.999:
            print("  ⚠ Rust executor matches Python but differs from HF")
            print("    → Compiler lowering introduces systematic error")
        else:
            print("  ❌ Rust executor differs from both")
            print("    → Rust executor implementation error or input construction bug")


if __name__ == "__main__":
    main()
