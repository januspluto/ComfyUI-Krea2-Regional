"""Tests for krea2_latent.py (pure CPU, no ComfyUI model needed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# stub comfy.model_management so the node imports headless
import types
mm = types.ModuleType("comfy.model_management")
mm.intermediate_device = lambda: "cpu"
comfy_pkg = sys.modules.setdefault("comfy", types.ModuleType("comfy"))
comfy_pkg.model_management = mm
sys.modules["comfy.model_management"] = mm

import torch
from krea2_latent import (Krea2EmptyLatentImage, compute_dimensions, snap,
                          PRESET_MAP, SNAP, MAX_SIDE_2K)

node = Krea2EmptyLatentImage()

# ---- 1. every preset snaps to multiples of 16 at 1.0 MP
for label, (bw, bh) in PRESET_MAP.items():
    assert bw % SNAP == 0 and bh % SNAP == 0, label
print("1) presets are /16: ok")

# ---- 2. 1:1 at 1.0 MP == 1024x1024, latent shape is [b,16,H/8,W/8]
lat, w, h = node.generate("1:1   (1024 x 1024) square", 1.0, True, 2)
assert (w, h) == (1024, 1024)
assert lat["samples"].shape == (2, 16, 128, 128), lat["samples"].shape
print("2) 1:1 @1MP + latent shape: ok")

# ---- 3. megapixels scales area, preserves ratio, snaps to 16
w2, h2 = compute_dimensions(1024, 1024, 4.0, False)
assert w2 == h2 and w2 % SNAP == 0
assert abs((w2 * h2) / (1024 * 1024) - 4.0) < 0.02, (w2, h2)
print("3) megapixels area scaling: ok")

# ---- 4. clamp_to_2k caps the longest side
w3, h3 = compute_dimensions(1568, 672, 4.2, True)   # wide preset, high MP
assert max(w3, h3) <= MAX_SIDE_2K, (w3, h3)
w4, h4 = compute_dimensions(1568, 672, 4.2, False)
assert max(w4, h4) > MAX_SIDE_2K, "unclamped should exceed 2k"
print("4) clamp_to_2k: ok")

# ---- 5. aspect ratio preserved through scaling (within snap tolerance)
bw, bh = PRESET_MAP["16:9  (1392 x 784)  widescreen"]
w5, h5 = compute_dimensions(bw, bh, 2.0, False)
assert abs((w5 / h5) - (bw / bh)) < 0.02
print("5) ratio preserved: ok")

# ---- 6. snap() floors to at least SNAP and rounds to nearest 16
assert snap(3) == SNAP and snap(1020) == 1024 and snap(1030) == 1024
print("6) snap(): ok")

print("\nall latent tests passed")
