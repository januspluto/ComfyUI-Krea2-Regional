"""Bridge: Ideogram 4 caption JSON (KJNodes' Ideogram4PromptBuilderKJ / the
AutoBuilder pipeline) -> Krea2 regional prompting.

Wire the KJ builder's `prompt` STRING output (the assembled caption JSON — it
carries every element's desc AND bbox after your canvas edits) into
`caption_json` here. This node then, per element:

  * builds a rectangle MASK on the fly from the element's bbox
    (0-1000 normalized `[ymin, xmin, ymax, xmax]` by default, matching the
    builder's toolbar setting; absolute / xy modes supported),
  * encodes the element's `desc` with the supplied CLIP (Krea2 text encoder),
  * emits a KREA2_REGIONS chain ready for `Krea2 Apply Regional`.

The base/global prompt is assembled from `high_level_description`, the
`background`, the style descriptors, and any *unplaced* elements (no bbox),
then encoded and returned as CONDITIONING for Apply Regional's base input.

Manual regions (e.g. ones carrying per-region LoRAs via `Krea2 Regional
Prompt`) can be merged through `prev_regions`.
"""

from __future__ import annotations

import json
import logging
import math
import re

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# caption parsing
# ---------------------------------------------------------------------------


def _loads_caption(text: str):
    """Parse the caption JSON; tolerate markdown fences and stray prose."""
    if not text or not text.strip():
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    try:
        data = json.loads(s)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        a, b = s.find("{"), s.rfind("}")
        if 0 <= a < b:
            try:
                data = json.loads(s[a:b + 1])
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def _bbox_to_frac(bbox, coord_mode, bbox_order, width, height):
    """bbox -> (x0, y0, x1, y1) fractions of the canvas, or None."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        vals = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if bbox_order == "xy":
        xmin, ymin, xmax, ymax = vals
    else:  # "yx" — Ideogram's [ymin, xmin, ymax, xmax]
        ymin, xmin, ymax, xmax = vals
    sx, sy = (width, height) if coord_mode == "absolute" else (1000.0, 1000.0)
    x0, x1 = sorted((xmin / sx, xmax / sx))
    y0, y1 = sorted((ymin / sy, ymax / sy))
    x0, x1 = max(0.0, x0), min(1.0, x1)
    y0, y1 = max(0.0, y0), min(1.0, y1)
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return x0, y0, x1, y1


def _element_prompt(el, include_palettes):
    desc = str(el.get("desc", "") or "").strip()
    parts = []
    if el.get("type") == "text":
        text = str(el.get("text", "") or "").strip()
        if text:
            parts.append(f'the text "{text}"')
    if desc:
        parts.append(desc)
    if include_palettes:
        pal = [c for c in (el.get("color_palette") or []) if isinstance(c, str)]
        if pal:
            parts.append("color palette: " + ", ".join(pal))
    return ", ".join(parts)


def _base_prompt(cap, unplaced_prompts, prepend, append, include_style):
    sentences = []
    if prepend.strip():
        sentences.append(prepend.strip())
    hld = str(cap.get("high_level_description", "") or "").strip()
    if hld:
        sentences.append(hld)
    cd = cap.get("compositional_deconstruction") or {}
    bg = str(cd.get("background", "") or "").strip()
    if bg:
        sentences.append(bg)
    sentences.extend(p for p in unplaced_prompts if p)
    if include_style:
        sd = cap.get("style_description") or {}
        bits = [str(sd.get(k, "") or "").strip()
                for k in ("aesthetics", "lighting", "photo", "art_style", "medium")]
        style = ", ".join(b for b in bits if b)
        if style:
            sentences.append(style)
    if append.strip():
        sentences.append(append.strip())
    return ". ".join(s.rstrip(". ") for s in sentences if s).strip()


# ---------------------------------------------------------------------------
# mask building
# ---------------------------------------------------------------------------


def _gaussian_kernel1d(sigma: float, device):
    radius = max(1, int(math.ceil(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(xs ** 2) / (2.0 * sigma ** 2))
    return (k / k.sum()), radius


def _grow_feather(m, grow_px, feather_px):
    """(H, W) float mask -> dilated + gaussian-feathered."""
    m = m[None, None]
    if grow_px > 0:
        k = 2 * int(grow_px) + 1
        m = F.max_pool2d(m, kernel_size=k, stride=1, padding=k // 2)
    if feather_px > 0:
        k, r = _gaussian_kernel1d(float(feather_px) / 2.0, m.device)
        m = F.conv2d(m, k.view(1, 1, 1, -1), padding=(0, r))
        m = F.conv2d(m, k.view(1, 1, -1, 1), padding=(r, 0))
    return m[0, 0].clamp(0, 1)


def _rect_mask(frac, width, height, grow_px, feather_px, device="cpu"):
    x0, y0, x1, y1 = frac
    m = torch.zeros(height, width, device=device)
    px0, px1 = int(round(x0 * width)), max(int(round(x1 * width)), int(round(x0 * width)) + 1)
    py0, py1 = int(round(y0 * height)), max(int(round(y1 * height)), int(round(y0 * height)) + 1)
    m[py0:py1, px0:px1] = 1.0
    return _grow_feather(m, grow_px, feather_px)


def _poly_mask(points, width, height, grow_px, feather_px):
    """points: [[x, y], ...] fractions -> (H, W) rasterized polygon mask."""
    import numpy as np
    from PIL import Image, ImageDraw

    img = Image.new("L", (width, height), 0)
    px = [(float(x) * width, float(y) * height) for x, y in points]
    if len(px) >= 3:
        ImageDraw.Draw(img).polygon(px, fill=255)
    m = torch.from_numpy(np.asarray(img, dtype="float32") / 255.0)
    return _grow_feather(m, grow_px, feather_px)


# ---------------------------------------------------------------------------
# per-region LoRA assignment
# ---------------------------------------------------------------------------
#
# Two concise ways to put a LoRA on a region, no extra nodes:
#
# 1. Inline tags in the element's desc (typed on the KJ canvas, or written by
#    the captioning LLM):        a wizard in blue robes <lora:mychar:0.9>
#    Tags in high_level_description / background become BASE (global) LoRAs.
#
# 2. `lora_map` widget — one rule per line:
#        matcher => lora_name @ strength
#    matcher: `*` (every region), an integer (1-based region order),
#    or a case-insensitive substring of the region's desc.
#        wizard => characters/mychar.safetensors @ 0.9
#        1 => watercolor @ 1.0
#        * => detail_boost @ 0.5

_LORA_TAG_RE = re.compile(r"<lora:([^:>]+?)(?::([-\d.]+))?>", re.IGNORECASE)
_LORA_SD_CACHE: dict[str, dict] = {}


def _extract_lora_tags(text: str):
    """Return (clean_text, [(name, strength), ...])."""
    found = [(m.group(1).strip(), float(m.group(2)) if m.group(2) else 1.0)
             for m in _LORA_TAG_RE.finditer(text or "")]
    clean = _LORA_TAG_RE.sub("", text or "")
    clean = re.sub(r"\s{2,}", " ", clean).strip(" ,")
    return clean, found


def _parse_lora_map(text: str):
    """-> list of (matcher, lora_name, strength)."""
    rules = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=>" not in line:
            continue
        matcher, rhs = line.split("=>", 1)
        name, strength = rhs, 1.0
        if "@" in rhs:
            name, s = rhs.rsplit("@", 1)
            try:
                strength = float(s.strip())
            except ValueError:
                name = rhs
        rules.append((matcher.strip(), name.strip(), strength))
    return rules


def _resolve_lora_name(name: str):
    """Fuzzy-match a tag/rule name against files in models/loras."""
    import folder_paths

    candidates = folder_paths.get_filename_list("loras")
    norm = name.replace("\\", "/").lower()
    stems = {}
    for c in candidates:
        cl = c.replace("\\", "/").lower()
        if cl == norm or cl == norm + ".safetensors":
            return c
        stem = cl.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        stems.setdefault(stem, []).append(c)
    base = norm.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if base in stems and len(stems[base]) == 1:
        return stems[base][0]
    subs = [c for c in candidates if base in c.replace("\\", "/").lower()]
    if len(subs) == 1:
        return subs[0]
    return None


def _load_lora_entry(name: str, strength: float):
    """-> {"sd", "strength", "label"} (Krea2ApplyRegional's format) or None."""
    import folder_paths
    import comfy.utils

    resolved = _resolve_lora_name(name)
    if resolved is None:
        logging.warning("[Krea2Regional] lora '%s' not found in models/loras "
                        "(or ambiguous) — skipped.", name)
        return None
    if resolved not in _LORA_SD_CACHE:
        path = folder_paths.get_full_path_or_raise("loras", resolved)
        _LORA_SD_CACHE[resolved] = comfy.utils.load_torch_file(path,
                                                               safe_load=True)
    return {"sd": _LORA_SD_CACHE[resolved], "strength": strength,
            "label": resolved}


def _loras_for_region(idx0, desc, inline_tags, rules):
    """Merge inline tags + matching lora_map rules for one region."""
    picked = list(inline_tags)
    low = (desc or "").lower()
    for matcher, name, strength in rules:
        if matcher == "*":
            picked.append((name, strength))
        elif matcher.isdigit():
            if int(matcher) == idx0 + 1:
                picked.append((name, strength))
        elif matcher.lower() in low:
            picked.append((name, strength))
    out, seen = [], set()
    for name, strength in picked:
        key = (name.lower(), strength)
        if key in seen:
            continue
        seen.add(key)
        entry = _load_lora_entry(name, strength)
        if entry:
            out.append(entry)
    return out


# ---------------------------------------------------------------------------
# node
# ---------------------------------------------------------------------------


def _encode(clip, text: str):
    tokens = clip.tokenize(text)
    cond = clip.encode_from_tokens_scheduled(tokens)
    return cond  # CONDITIONING: [[tensor, dict], ...]


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
