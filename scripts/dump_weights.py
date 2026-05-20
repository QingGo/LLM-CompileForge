#!/usr/bin/env python3
"""Dump all model weights from Python (safetensors) for comparison with Rust."""

from __future__ import annotations
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compiler.serialize import load_artifact

ARTIFACT_DIR = "./compiled/opt_125m_fresh"
OUTPUT_DIR = "/tmp/issue45_weights"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load the artifact to get weight names and values from Python executor
print("Loading artifact from", ARTIFACT_DIR)
artifact = load_artifact(ARTIFACT_DIR)
# The artifact has functions, each with weights dict {weight_name: tensor}

# Collect all weights
all_weights = {}
for func in artifact.functions:
    for wname, wtensor in func.weights.items():
        if wname not in all_weights:
            all_weights[wname] = wtensor.numpy()
            
print(f"Found {len(all_weights)} unique weight tensors")

# Save all weights as .npz
npz_path = os.path.join(OUTPUT_DIR, "weights_py.npz")
np.savez(npz_path, **all_weights)
print(f"Saved Python weights to {npz_path}")

# Also save individually as .npy for per-tensor comparison
for wname, w in all_weights.items():
    npy_path = os.path.join(OUTPUT_DIR, f"{wname}.npy")
    np.save(npy_path, w)

print(f"Total tensors: {len(all_weights)}")
for wname, w in all_weights.items():
    print(f"  {wname}: shape={w.shape}, dtype={w.dtype}, first_val={w.ravel()[0]:.6f}, " 
          f"min={w.min():.6f}, max={w.max():.6f}")

# Save a manifest
manifest = {wname: {"shape": list(w.shape), "dtype": str(w.dtype)} 
            for wname, w in all_weights.items()}
json.dump(manifest, open(os.path.join(OUTPUT_DIR, "manifest_py.json"), "w"), indent=2)
print(f"Manifest saved to {OUTPUT_DIR}/manifest_py.json")
