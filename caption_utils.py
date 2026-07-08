"""Shared helpers for Krea2 regional nodes: caption parsing, mask
rasterization (rect + polygon), LoRA tag extraction and loading, prompt
encoding. Imported by the canvas builder and the optional Ideogram bridge."""

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


