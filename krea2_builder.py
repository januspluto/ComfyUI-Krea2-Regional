"""Krea2 Regional Builder — one node for the whole regional setup.

A canvas editor (web/js/krea2_builder.js) where you draw regions (rectangles
or freehand lasso shapes), type each region's prompt, mark it as an object or
rendered TEXT (with a dedicated text field, like the KJ builder), and pick its
LoRA(s) from a searchable dropdown of your models/loras folder.

Import an Ideogram-4 caption (wire the AutoBuilder's `caption_json` into
`import_json`): elements become editable regions, the high-level description /
background / style fields populate the matching widgets (visible + editable),
and inline <lora:name:strength> tags become pre-filled dropdown rows.
`import_mode` works like the KJ builder: "when empty" seeds once and then your
edits win; "always" makes the wired JSON authoritative.

Outputs plug straight into `Krea2 Apply Regional`.
"""

from __future__ import annotations

import json
import logging

import torch

try:
    from .ideogram_bridge import (_encode, _extract_lora_tags,
                                  _load_lora_entry, _loads_caption,
                                  _poly_mask, _rect_mask)
except ImportError:  # standalone (tests)
    from ideogram_bridge import (_encode, _extract_lora_tags,
                                 _load_lora_entry, _loads_caption,
                                 _poly_mask, _rect_mask)


def _caption_to_state(cap):
    """Ideogram caption dict -> (builder state, field values, unplaced descs)."""
    regions, base_extra = [], []
    cd = cap.get("compositional_deconstruction") or {}
    for el in (cd.get("elements") or []):
        if not isinstance(el, dict):
            continue
        desc, tags = _extract_lora_tags(str(el.get("desc", "") or ""))
        rtype = "text" if el.get("type") == "text" else "obj"
        text = str(el.get("text", "") or "").strip()
        loras = [{"name": n, "strength": s} for n, s in tags]
        bb = el.get("bbox")
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            try:
                vals = [float(v) / 1000.0 for v in bb]
                if str(cap.get("bbox_order", "yx")).lower() == "xy":
                    xmin, ymin, xmax, ymax = vals
                else:
                    ymin, xmin, ymax, xmax = vals
            except (TypeError, ValueError):
                ymin = xmin = ymax = xmax = 0.0
            x0, x1 = sorted((max(0.0, xmin), min(1.0, xmax)))
            y0, y1 = sorted((max(0.0, ymin), min(1.0, ymax)))
            if (x1 - x0) > 0 and (y1 - y0) > 0 and (desc or text):
                regions.append({"shape": "rect", "x": x0, "y": y0,
                                "w": x1 - x0, "h": y1 - y0, "desc": desc,
                                "rtype": rtype, "text": text, "loras": loras})
                continue
        if desc or text:
            base_extra.append(_region_prompt_text(rtype, text, desc))

    hld_clean, hld_tags = _extract_lora_tags(
        str(cap.get("high_level_description", "") or ""))
    bg_clean, bg_tags = _extract_lora_tags(str(cd.get("background", "") or ""))
    sd = cap.get("style_description") or {}
    fields = {
        "base_prompt": hld_clean,
        "background": bg_clean,
        "aesthetics": str(sd.get("aesthetics", "") or "").strip(),
        "lighting": str(sd.get("lighting", "") or "").strip(),
        "medium": ", ".join(x for x in (
            str(sd.get("photo", "") or "").strip(),
            str(sd.get("art_style", "") or "").strip(),
            str(sd.get("medium", "") or "").strip()) if x),
    }
    base_loras = [{"name": n, "strength": s} for n, s in hld_tags + bg_tags]
    return ({"regions": regions, "base_loras": base_loras},
            fields, base_extra)


def _region_prompt_text(rtype, text, desc):
    if rtype == "text" and text:
        quoted = f'the text "{text}"'
        return f"{quoted}, {desc}" if desc else quoted
    return desc


def _region_mask(r, width, height, grow_px, feather_px):
    if r.get("shape") == "poly":
        pts = r.get("points") or []
        if len(pts) < 3:
            return None
        return _poly_mask(pts, width, height, grow_px, feather_px)
    try:
        x0 = max(0.0, float(r.get("x", 0)))
        y0 = max(0.0, float(r.get("y", 0)))
        x1 = min(1.0, x0 + float(r.get("w", 0)))
        y1 = min(1.0, y0 + float(r.get("h", 0)))
    except (TypeError, ValueError):
        return None
    if x1 - x0 <= 0.005 or y1 - y0 <= 0.005:
        return None
    return _rect_mask((x0, y0, x1, y1), width, height, grow_px, feather_px)


def _entries(rows):
    out = []
    for r in rows or []:
        name = str(r.get("name", "") or "").strip()
        if not name or name.lower() == "none":
            continue
        try:
            strength = float(r.get("strength", 1.0))
        except (TypeError, ValueError):
            strength = 1.0
        entry = _load_lora_entry(name, strength)
        if entry:
            out.append(entry)
    return out


class Krea2RegionalBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "width": ("INT", {"default": 1024, "min": 64, "max": 16384,
                                  "step": 16}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 16384,
                                   "step": 16}),
                "grow_px": ("INT", {"default": 0, "min": 0, "max": 512}),
                "feather_px": ("INT", {"default": 0, "min": 0, "max": 512}),
                "base_prompt": ("STRING", {"default": "", "multiline": True,
                                "placeholder": "high level description",
                                "tooltip": "Global scene description. Filled "
                                           "from an imported caption."}),
                "background": ("STRING", {"default": "", "multiline": True,
                               "placeholder": "background",
                               "tooltip": "Background description. Filled "
                                          "from an imported caption."}),
                "aesthetics": ("STRING", {"default": "",
                               "placeholder": "aesthetics"}),
                "lighting": ("STRING", {"default": "",
                             "placeholder": "lighting"}),
                "medium": ("STRING", {"default": "",
                           "placeholder": "medium / photo / art style"}),
                "region_append": ("STRING", {"default": "",
                                  "tooltip": "Appended to every region prompt."}),
                "import_mode": (["when empty", "always"],
                                {"default": "when empty"}),
                # managed by the canvas UI; hidden by the frontend
                "regions_data": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "import_json": ("STRING", {"forceInput": True}),
                "image": ("IMAGE",),
                "prev_regions": ("KREA2_REGIONS",),
                "extra_base_loras": ("KREA2_LORAS",),
            },
        }

    RETURN_TYPES = ("KREA2_REGIONS", "CONDITIONING", "KREA2_LORAS", "MASK",
                    "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("regions", "base_conditioning", "base_loras", "masks",
                    "base_prompt", "region_prompts", "width", "height")
    FUNCTION = "build"
    CATEGORY = "conditioning/krea2_regional"

    def build(self, clip, width, height, grow_px, feather_px, base_prompt,
              background, aesthetics, lighting, medium, region_append,
              import_mode, regions_data, import_json=None, image=None,
              prev_regions=None, extra_base_loras=None):
        try:
            state = json.loads(regions_data) if regions_data.strip() else {}
        except json.JSONDecodeError:
            logging.warning("[Krea2Regional] regions_data is corrupt; "
                            "starting empty.")
            state = {}
        state.setdefault("regions", [])
        state.setdefault("base_loras", [])

        fields, base_extra, imported = {}, [], False
        if import_json and import_json.strip():
            cap = _loads_caption(import_json)
            if cap is not None and (import_mode == "always"
                                    or not state["regions"]):
                keep_brightness = state.get("bg_brightness")
                state, fields, base_extra = _caption_to_state(cap)
                if keep_brightness is not None:
                    state["bg_brightness"] = keep_brightness
                imported = True
                # imported field values win only where the widget is empty
                # in "when empty" mode; "always" overwrites
                if import_mode == "always":
                    base_prompt = fields["base_prompt"] or base_prompt
                    background = fields["background"] or background
                    aesthetics = fields["aesthetics"] or aesthetics
                    lighting = fields["lighting"] or lighting
                    medium = fields["medium"] or medium
                else:
                    base_prompt = base_prompt.strip() or fields["base_prompt"]
                    background = background.strip() or fields["background"]
                    aesthetics = aesthetics.strip() or fields["aesthetics"]
                    lighting = lighting.strip() or fields["lighting"]
                    medium = medium.strip() or fields["medium"]

        ui = {"k2b_state": [json.dumps(state)] if imported else []}
        if imported:
            # push the (possibly merged) field values into the visible widgets
            ui["k2b_fields"] = [json.dumps({
                "base_prompt": base_prompt, "background": background,
                "aesthetics": aesthetics, "lighting": lighting,
                "medium": medium,
            })]

        # ---- background preview for the canvas (custom key: the frontend
        # draws it INSIDE the canvas; nothing renders under the node) ----
        if image is not None:
            try:
                ui["k2b_bg"] = [_save_preview(image)]
            except Exception as e:  # preview is cosmetic; never fail the run
                logging.warning("[Krea2Regional] preview failed: %s", e)

        # ---- regions ----
        regions, masks, region_texts = [], [], []
        for r in state["regions"]:
            desc, tags = _extract_lora_tags(str(r.get("desc", "") or ""))
            rtype = r.get("rtype", "obj")
            text = str(r.get("text", "") or "").strip()
            prompt = _region_prompt_text(rtype, text, desc.strip())
            if not prompt:
                continue
            mask = _region_mask(r, width, height, grow_px, feather_px)
            if mask is None:
                continue
            loras = _entries(r.get("loras")) + _entries(
                [{"name": n, "strength": s} for n, s in tags])
            if region_append.strip():
                prompt += ", " + region_append.strip()
            cond = _encode(clip, prompt)
            regions.append({"cond": cond[0][0], "mask": mask, "loras": loras})
            masks.append(mask)
            note = " ".join(f"[{e['label']}@{e['strength']}]" for e in loras)
            region_texts.append((prompt + " " + note).strip())

        # ---- base prompt: description + background + unplaced + style ----
        sentences = [s.strip().rstrip(".") for s in
                     ([base_prompt, background] + base_extra) if s.strip()]
        style = ", ".join(s.strip() for s in (aesthetics, lighting, medium)
                          if s.strip())
        if style:
            sentences.append(style)
        base_text = ". ".join(sentences) or "an image"
        base_cond = _encode(clip, base_text)

        base_loras = list(extra_base_loras or []) + _entries(
            state["base_loras"])

        all_regions = list(prev_regions or []) + regions
        mask_batch = (torch.stack(masks) if masks
                      else torch.zeros(1, height, width))
        return {"ui": ui,
                "result": (all_regions, base_cond, base_loras, mask_batch,
                           base_text, "\n".join(region_texts),
                           width, height)}


def _save_preview(image, max_side=768):
    """IMAGE tensor -> temp png; returns a /view-compatible entry."""
    import os
    import random

    import folder_paths
    from PIL import Image

    arr = (image[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(arr)
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    tmp = folder_paths.get_temp_directory()
    os.makedirs(tmp, exist_ok=True)
    name = f"k2b_bg_{random.randint(0, 10**9)}.png"
    img.save(os.path.join(tmp, name))
    return {"filename": name, "subfolder": "", "type": "temp"}


BUILDER_NODE_CLASS_MAPPINGS = {"Krea2RegionalBuilder": Krea2RegionalBuilder}
BUILDER_NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2RegionalBuilder": "Krea2 Regional Builder (canvas)",
}
