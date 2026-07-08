"""Optional bridge node: Ideogram 4 caption JSON -> Krea2 regions.

Turns an Ideogram-style caption JSON (e.g. from KJNodes' Ideogram4PromptBuilderKJ)
into a KREA2_REGIONS chain without the canvas builder. The canvas builder
(Krea2RegionalBuilder) supersedes this for most uses — it can import the same
JSON via its `import_json` input AND let you edit the result. This node is kept
for headless / no-canvas workflows.

All shared logic lives in caption_utils.py.
"""

from __future__ import annotations

import logging

import torch

try:
    from .caption_utils import (
        _base_prompt, _bbox_to_frac, _element_prompt, _encode,
        _extract_lora_tags, _load_lora_entry, _loads_caption,
        _loras_for_region, _parse_lora_map, _rect_mask)
except ImportError:  # standalone (tests)
    from caption_utils import (
        _base_prompt, _bbox_to_frac, _element_prompt, _encode,
        _extract_lora_tags, _load_lora_entry, _loads_caption,
        _loras_for_region, _parse_lora_map, _rect_mask)


class Krea2RegionsFromIdeogram:
    """Ideogram-4 caption JSON -> regions with masks, prompts AND LoRAs.

    caption_json can be wired from the KJ builder's `prompt` output or pasted
    straight into the widget. Each placed element becomes one region:
    conditioning (its desc), mask (its bbox), and LoRAs from inline
    `<lora:name:strength>` tags and/or `lora_map` rules. Base loras (from tags
    in high_level_description / background, or `* =>` rules with matcher
    `base`) come out of the `base_loras` output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "caption_json": ("STRING", {"default": "", "multiline": True,
                                 "tooltip": "Wire the KJ builder's `prompt` output here, "
                                            "or paste the caption JSON directly."}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 16384,
                                  "step": 16}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 16384,
                                   "step": 16}),
                "grow_px": ("INT", {"default": 0, "min": 0, "max": 512,
                                    "tooltip": "Dilate every bbox mask by this many pixels."}),
                "feather_px": ("INT", {"default": 0, "min": 0, "max": 512,
                                       "tooltip": "Gaussian-soften mask edges. Soft values "
                                                  "feather regional-LoRA strength; the attention "
                                                  "mask binarizes at 0.5."}),
                "coord_mode": (["normalized", "absolute"], {"default": "normalized",
                               "tooltip": "Match the KJ builder's toolbar setting "
                                          "(normalized 0-1000 grid vs absolute pixels)."}),
                "bbox_order": (["yx", "xy"], {"default": "yx",
                               "tooltip": "Match the KJ builder's toolbar setting "
                                          "(yx = Ideogram [ymin,xmin,ymax,xmax]; "
                                          "xy = Qwen [xmin,ymin,xmax,ymax])."}),
                "lora_map": ("STRING", {"default": "", "multiline": True,
                             "tooltip": "One rule per line: matcher => lora_name @ strength\n"
                                        "matcher: * (all regions), base (base prompt), an "
                                        "integer (1-based region order), or a substring of "
                                        "the region's desc.\n"
                                        "e.g.  wizard => mychar @ 0.9\n"
                                        "Inline <lora:name:0.8> tags inside descs also work."}),
                "include_style_in_base": ("BOOLEAN", {"default": True}),
                "include_unplaced_in_base": ("BOOLEAN", {"default": True,
                                             "tooltip": "Elements without a bbox get folded "
                                                        "into the base prompt."}),
                "include_palettes": ("BOOLEAN", {"default": False}),
                "base_prepend": ("STRING", {"default": "", "multiline": True,
                                            "tooltip": "Prepended to the base prompt "
                                                       "(global LoRA trigger words etc.)."}),
                "base_append": ("STRING", {"default": "", "multiline": True}),
                "region_append": ("STRING", {"default": "",
                                             "tooltip": "Appended to every region prompt "
                                                        "(shared style tag etc.)."}),
            },
            "optional": {
                "prev_regions": ("KREA2_REGIONS",),
                "extra_base_loras": ("KREA2_LORAS",),
            },
        }

    RETURN_TYPES = ("KREA2_REGIONS", "CONDITIONING", "KREA2_LORAS", "MASK",
                    "STRING", "STRING")
    RETURN_NAMES = ("regions", "base_conditioning", "base_loras", "masks",
                    "base_prompt", "region_prompts")
    FUNCTION = "build"
    CATEGORY = "conditioning/krea2_regional"

    def build(self, clip, caption_json, width, height, grow_px, feather_px,
              coord_mode, bbox_order, lora_map, include_style_in_base,
              include_unplaced_in_base, include_palettes, base_prepend,
              base_append, region_append, prev_regions=None,
              extra_base_loras=None):
        rules = _parse_lora_map(lora_map)
        base_rules = [(n, s) for m, n, s in rules if m.lower() == "base"]
        region_rules = [(m, n, s) for m, n, s in rules if m.lower() != "base"]

        cap = _loads_caption(caption_json)
        if cap is None:
            # not JSON -> treat the whole string as a plain base prompt
            clean, base_tags = _extract_lora_tags(caption_json)
            base_text = " ".join(
                t for t in (base_prepend.strip(), clean.strip(),
                            base_append.strip()) if t
            ) or "an image"
            if caption_json.strip():
                logging.warning("[Krea2Regional] caption_json is not valid "
                                "JSON; using it as a plain base prompt.")
            base_loras = list(extra_base_loras or [])
            for name, s in base_tags + base_rules:
                entry = _load_lora_entry(name, s)
                if entry:
                    base_loras.append(entry)
            return (list(prev_regions or []), _encode(clip, base_text),
                    base_loras, torch.zeros(1, height, width), base_text, "")

        cd = cap.get("compositional_deconstruction") or {}
        elements = cd.get("elements") or []

        # base-level inline tags (high_level_description / background)
        hld_clean, hld_tags = _extract_lora_tags(
            str(cap.get("high_level_description", "") or ""))
        bg_clean, bg_tags = _extract_lora_tags(
            str(cd.get("background", "") or ""))
        cap = dict(cap)
        cap["high_level_description"] = hld_clean
        cap["compositional_deconstruction"] = dict(cd, background=bg_clean)

        regions, masks, region_texts, unplaced = [], [], [], []
        ridx = 0
        for el in elements:
            if not isinstance(el, dict):
                continue
            raw_desc = str(el.get("desc", "") or "")
            clean_desc, tags = _extract_lora_tags(raw_desc)
            el = dict(el, desc=clean_desc)
            prompt = _element_prompt(el, include_palettes)
            if not prompt:
                continue
            frac = _bbox_to_frac(el.get("bbox"), coord_mode, bbox_order,
                                 width, height)
            if frac is None:
                if include_unplaced_in_base:
                    unplaced.append(prompt)
                continue
            loras = _loras_for_region(ridx, clean_desc, tags, region_rules)
            if region_append.strip():
                prompt = prompt + ", " + region_append.strip()
            mask = _rect_mask(frac, width, height, grow_px, feather_px)
            cond = _encode(clip, prompt)
            regions.append({"cond": cond[0][0], "mask": mask, "loras": loras})
            masks.append(mask)
            tag_note = " ".join(f"[{e['label']}@{e['strength']}]"
                                for e in loras)
            region_texts.append((prompt + " " + tag_note).strip())
            ridx += 1

        base_text = _base_prompt(cap, unplaced, base_prepend, base_append,
                                 include_style_in_base) or "an image"
        base_cond = _encode(clip, base_text)

        base_loras = list(extra_base_loras or [])
        for name, s in hld_tags + bg_tags + base_rules:
            entry = _load_lora_entry(name, s)
            if entry:
                base_loras.append(entry)

        all_regions = list(prev_regions or []) + regions
        mask_batch = (torch.stack(masks) if masks
                      else torch.zeros(1, height, width))
        return (all_regions, base_cond, base_loras, mask_batch, base_text,
                "\n".join(region_texts))


BRIDGE_NODE_CLASS_MAPPINGS = {
    "Krea2RegionsFromIdeogram": Krea2RegionsFromIdeogram,
}

BRIDGE_NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2RegionsFromIdeogram": "Krea2 Regions from Ideogram JSON",
}
