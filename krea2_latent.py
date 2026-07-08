"""
Krea 2 Empty Latent Image
-------------------------
An rgthree-style empty latent node tuned for Krea 2 (K2 Raw / Turbo).

Krea 2 uses the Qwen Image VAE: 16-channel latents, 8x spatial compression,
and pixel dimensions padded to a multiple of 16. Native output range is
roughly 1K (~1 MP) up to 2K (~4.2 MP).

Buckets are aspect-ratio-first: each preset is the multiple-of-16 dimension
pair that best preserves its ratio at a ~1.0 MP baseline. The `megapixels`
dial scales the area from there (1.0 = 1K class, ~4.0 = 2K class), always
snapping to multiples of 16, with an optional clamp keeping the longest
side within Krea 2's 2048px native ceiling.
"""

import math
import torch
import comfy.model_management

# Aspect-ratio buckets at the ~1.0 MP baseline. Each is the multiple-of-16
# pair that most exactly preserves the ratio near 1 MP (all ratios are exact
# except 16:9 / 9:16, which are within 0.13%). Ordered widest -> tallest.
ASPECT_PRESETS = [
    # (label, base_width, base_height)
    ("21:9  (1568 x 672)  cinematic",   1568, 672),
    ("2:1   (1440 x 720)  panorama",    1440, 720),
    ("16:9  (1392 x 784)  widescreen",  1392, 784),
    ("3:2   (1248 x 832)  landscape",   1248, 832),
    ("7:5   (1232 x 880)  landscape",   1232, 880),
    ("4:3   (1152 x 864)  landscape",   1152, 864),
    ("5:4   (1120 x 896)  landscape",   1120, 896),
    ("1:1   (1024 x 1024) square",      1024, 1024),
    ("4:5   (896 x 1120)  portrait",    896, 1120),
    ("3:4   (864 x 1152)  portrait",    864, 1152),
    ("5:7   (880 x 1232)  portrait",    880, 1232),
    ("2:3   (832 x 1248)  portrait",    832, 1248),
    ("9:16  (784 x 1392)  vertical",    784, 1392),
    ("1:2   (720 x 1440)  vertical",    720, 1440),
    ("9:21  (672 x 1568)  cinematic",   672, 1568),
]

PRESET_LABELS = [p[0] for p in ASPECT_PRESETS]
PRESET_MAP = {p[0]: (p[1], p[2]) for p in ASPECT_PRESETS}

LATENT_CHANNELS = 16   # Qwen Image VAE
SPATIAL_FACTOR = 8     # VAE downscale
SNAP = 16              # Krea 2 pads pixel dims to a multiple of 16
MAX_SIDE_2K = 2048     # Krea 2's native 2K ceiling


def snap(value):
    return max(SNAP, int(round(value / SNAP) * SNAP))


def compute_dimensions(base_w, base_h, megapixels, clamp_to_2k):
    """Scale the base bucket to the target area, snapped to SNAP."""
    scale = math.sqrt(megapixels)  # base buckets are ~1.0 MP
    width = base_w * scale
    height = base_h * scale

    if clamp_to_2k:
        longest = max(width, height)
        if longest > MAX_SIDE_2K:
            factor = MAX_SIDE_2K / longest
            width *= factor
            height *= factor

    return snap(width), snap(height)


class Krea2EmptyLatentImage:
    """Empty latent generator sized for Krea 2's native resolutions."""

    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dimensions": (PRESET_LABELS, {"default": "1:1   (1024 x 1024) square"}),
                "megapixels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.25,
                    "max": 4.2,   # ~2048x2048, Krea 2's 2K ceiling
                    "step": 0.05,
                    "tooltip": "Total pixel area relative to the preset. 1.0 = the "
                               "listed 1K size, ~4.0 = 2K class. Krea 2 is trained "
                               "for roughly 1K to 2K output.",
                }),
                "clamp_to_2k": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep the longest side at or below 2048px (Krea 2's "
                               "native ceiling). Wide ratios at high megapixels "
                               "get scaled down proportionally.",
                }),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("LATENT", "WIDTH", "HEIGHT")
    FUNCTION = "generate"
    CATEGORY = "latent"
    DESCRIPTION = ("Empty 16-channel latent for Krea 2 (Qwen Image VAE). Pick an "
                   "aspect ratio, scale it with megapixels (1.0 = 1K, ~4.0 = 2K). "
                   "Dimensions snap to multiples of 16.")

    def generate(self, dimensions, megapixels, clamp_to_2k, batch_size):
        base_w, base_h = PRESET_MAP[dimensions]
        width, height = compute_dimensions(base_w, base_h, megapixels, clamp_to_2k)

        latent = torch.zeros(
            [batch_size, LATENT_CHANNELS, height // SPATIAL_FACTOR, width // SPATIAL_FACTOR],
            device=self.device,
        )
        return ({"samples": latent}, width, height)


LATENT_NODE_CLASS_MAPPINGS = {
    "Krea2EmptyLatentImage": Krea2EmptyLatentImage,
}

LATENT_NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2EmptyLatentImage": "Krea 2 Empty Latent Image",
}

__all__ = ["LATENT_NODE_CLASS_MAPPINGS", "LATENT_NODE_DISPLAY_NAME_MAPPINGS"]
