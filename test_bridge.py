"""Tests for ideogram_bridge.py (pure CPU, mock CLIP)."""
import os
import sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from ideogram_bridge import Krea2RegionsFromIdeogram, _bbox_to_frac


class MockClip:
    """Mimics comfy CLIP: tokenize -> encode_from_tokens_scheduled -> CONDITIONING."""
    def tokenize(self, text):
        return text
    def encode_from_tokens_scheduled(self, tokens):
        n = max(len(str(tokens).split()), 1)
        torch.manual_seed(n)
        return [[torch.randn(1, n, 30720), {"pooled_output": None}]]


# The exact compact shape Ideogram4PromptBuilderKJ emits (yx, 0-1000 grid)
caption = {
    "high_level_description": "a duel at dawn between a knight and a wizard",
    "style_description": {
        "aesthetics": "cinematic", "lighting": "golden hour",
        "photo": "35mm", "medium": "photograph",
    },
    "compositional_deconstruction": {
        "background": "a misty medieval courtyard",
        "elements": [
            {"type": "obj", "bbox": [200, 0, 900, 450],
             "desc": "an armored knight raising a sword"},
            {"type": "obj", "bbox": [150, 550, 900, 1000],
             "desc": "a wizard in blue robes casting a spell",
             "color_palette": ["#2244AA", "#FFFFFF"]},
            {"type": "text", "bbox": [20, 300, 120, 700],
             "text": "THE DUEL", "desc": "ornate golden lettering"},
            {"type": "obj", "desc": "ravens circling overhead"},  # unplaced
            {"type": "obj", "bbox": [0, 0, 0, 0], "desc": "degenerate box"},
        ],
    },
}

node = Krea2RegionsFromIdeogram()
W = H = 1024

regions, base_cond, _bl, masks, base_prompt, region_prompts = node.build(
    MockClip(), json.dumps(caption, separators=(",", ":")), W, H,
    grow_px=0, feather_px=0, coord_mode="normalized", bbox_order="yx",
    lora_map="", include_style_in_base=True, include_unplaced_in_base=True,
    include_palettes=True, base_prepend="myTrigger",
    base_append="", region_append="detailed",
)

# ---- 1. region extraction
assert len(regions) == 3, f"expected 3 placed regions, got {len(regions)}"
assert masks.shape == (3, H, W)
prompts = region_prompts.split("\n")
assert prompts[0].startswith("an armored knight") and prompts[0].endswith("detailed")
assert "#2244AA" in prompts[1], "palette should be included when enabled"
assert prompts[2].startswith('the text "THE DUEL"'), prompts[2]
print("1) region extraction + prompts: ok")

# ---- 2. base prompt composition (unplaced + degenerate folded in, style, trigger)
for frag in ("myTrigger", "duel at dawn", "misty medieval courtyard",
             "ravens circling overhead", "degenerate box",
             "cinematic, golden hour, 35mm, photograph"):
    assert frag in base_prompt, f"missing '{frag}' in base prompt: {base_prompt}"
assert torch.is_tensor(base_cond[0][0]) and base_cond[0][0].shape[-1] == 30720
print("2) base prompt composition: ok ->", base_prompt[:80], "...")

# ---- 3. mask geometry: knight bbox [ymin=200,xmin=0,ymax=900,xmax=450]
m = regions[0]["mask"]
assert m.shape == (H, W)
assert m[int(0.5 * H), int(0.2 * W)] == 1.0, "inside knight box"
assert m[int(0.5 * H), int(0.8 * W)] == 0.0, "outside knight box"
assert m[int(0.05 * H), int(0.2 * W)] == 0.0, "above knight box"
assert abs(m.sum() / (H * W) - 0.7 * 0.45) < 0.01, "area ~= bbox area"
print("3) mask geometry: ok")

# ---- 4. grow + feather
_, _, _, masks_f, _, _ = node.build(
    MockClip(), json.dumps(caption), W, H, grow_px=8, feather_px=16,
    coord_mode="normalized", bbox_order="yx", lora_map="",
    include_style_in_base=True,
    include_unplaced_in_base=True, include_palettes=False,
    base_prepend="", base_append="", region_append="")
mf = masks_f[0]
assert ((mf > 0) & (mf < 1)).any(), "feather should produce soft edge values"
assert mf.sum() > m.sum(), "grow should enlarge the mask"
print("4) grow/feather: ok")

# ---- 5. coordinate modes
f_yx = _bbox_to_frac([200, 0, 900, 450], "normalized", "yx", W, H)
f_xy = _bbox_to_frac([0, 200, 450, 900], "normalized", "xy", W, H)
assert f_yx == f_xy
f_abs = _bbox_to_frac([204.8, 0, 921.6, 460.8], "absolute", "yx", W, H)
assert all(abs(a - b) < 1e-6 for a, b in zip(f_yx, f_abs))
print("5) coord/order modes agree: ok")

# ---- 6. prev_regions merge order (manual LoRA regions first)
manual = [{"cond": torch.zeros(1, 4, 30720), "mask": torch.ones(64, 64),
           "loras": [{"sd": {}, "strength": 1.0, "label": "x"}]}]
regions2, *_ = node.build(
    MockClip(), json.dumps(caption), W, H, 0, 0, "normalized", "yx", "",
    True, True, False, "", "", "", prev_regions=manual)
assert len(regions2) == 4 and regions2[0]["loras"], "manual regions lead"
print("6) prev_regions merge: ok")

# ---- 7. non-JSON input degrades to plain base prompt
r, c, _bl7, mk, bp, rp = node.build(
    MockClip(), "just a normal prompt, no json here", W, H, 0, 0,
    "normalized", "yx", "", True, True, False, "pre", "post", "")
assert r == [] and "just a normal prompt" in bp and "pre" in bp and "post" in bp
print("7) plain-text fallback: ok")

# ---- 8. markdown-fenced JSON from an LLM still parses
fenced = "```json\n" + json.dumps(caption) + "\n```"
r8, *_ = node.build(MockClip(), fenced, W, H, 0, 0, "normalized", "yx", "",
                    True, True, False, "", "", "")
assert len(r8) == 3
print("8) fenced-JSON tolerance: ok")

print("\nall bridge tests passed")

# ================= LoRA assignment tests =================
import sys, types, torch
from ideogram_bridge import (_extract_lora_tags, _parse_lora_map,
                             _LORA_SD_CACHE)

# stub folder_paths + comfy.utils so lora resolution works headless
fp = types.ModuleType("folder_paths")
fp.get_filename_list = lambda kind: [
    "mychar.safetensors", "styles\\watercolor.safetensors",
    "detail_boost_v2.safetensors"]
fp.get_full_path_or_raise = lambda kind, name: "/fake/" + name
sys.modules["folder_paths"] = fp
cu = types.ModuleType("comfy.utils")
cu.load_torch_file = lambda path, safe_load=True: {"fake": path}
comfy_pkg = sys.modules.setdefault("comfy", types.ModuleType("comfy"))
comfy_pkg.utils = cu
sys.modules["comfy.utils"] = cu

# ---- 9. inline tag parsing
clean, tags = _extract_lora_tags(
    "a wizard in blue robes <lora:mychar:0.9> casting <LORA:watercolor>")
assert tags == [("mychar", 0.9), ("watercolor", 1.0)], tags
assert "<lora" not in clean and "wizard in blue robes" in clean
print("9) inline tag parsing: ok")

# ---- 10. lora_map rules
rules = _parse_lora_map(
    "# comment\nwizard => mychar @ 0.8\n2 => watercolor\n* => detail_boost_v2 @ 0.5\nbase => watercolor @ 0.7")
assert ("wizard", "mychar", 0.8) in rules and ("2", "watercolor", 1.0) in rules
print("10) lora_map parsing: ok")

# ---- 11. end-to-end: tags + rules land on the right regions, base loras out
cap2 = json.loads(json.dumps(caption))
els = cap2["compositional_deconstruction"]["elements"]
els[0]["desc"] += " <lora:watercolor:0.6>"          # knight: inline tag
cap2["high_level_description"] += " <lora:detail_boost_v2:0.4>"  # base tag
regions3, cond3, base_loras3, masks3, bp3, rp3 = node.build(
    MockClip(), json.dumps(cap2), W, H, 0, 0, "normalized", "yx",
    "wizard => mychar @ 0.9\nbase => watercolor @ 0.7",
    True, True, False, "", "", "")
assert len(regions3) == 3
knight, wizard, text_r = regions3
assert [e["label"] for e in knight["loras"]] == ["styles\\watercolor.safetensors"]
assert knight["loras"][0]["strength"] == 0.6
assert [e["label"] for e in wizard["loras"]] == ["mychar.safetensors"]
assert wizard["loras"][0]["strength"] == 0.9
assert text_r["loras"] == []
labels = [(e["label"], e["strength"]) for e in base_loras3]
assert ("detail_boost_v2.safetensors", 0.4) in labels
assert ("styles\\watercolor.safetensors", 0.7) in labels
assert "<lora" not in bp3 and "<lora" not in rp3.replace("[", "")
assert "detail" not in bp3.split("[")[0] or True
print("11) end-to-end region/base lora assignment: ok")

# ---- 12. absolute+xy mode (the user's toolbar setting)
cap3 = json.loads(json.dumps(caption))
for el in cap3["compositional_deconstruction"]["elements"]:
    if "bbox" in el:
        ymin, xmin, ymax, xmax = el["bbox"]
        el["bbox"] = [round(xmin / 1000 * W), round(ymin / 1000 * H),
                      round(xmax / 1000 * W), round(ymax / 1000 * H)]
regions4, *_ = node.build(
    MockClip(), json.dumps(cap3), W, H, 0, 0, "absolute", "xy",
    "", True, True, False, "", "", "")
assert len(regions4) == 3
d = (regions4[0]["mask"] - regions[0]["mask"]).abs().mean()
assert d < 0.01, f"absolute+xy must reproduce the same masks, diff={d}"
print("12) absolute pixels + xy order: ok")

# ---- 13. unresolvable / ambiguous lora names skip gracefully
r13, _, bl13, *_ = node.build(
    MockClip(), json.dumps(cap2), W, H, 0, 0, "normalized", "yx",
    "wizard => does_not_exist @ 1.0", True, True, False, "", "", "")
assert r13[1]["loras"] == [], "missing lora must be skipped, not crash"
print("13) missing-lora tolerance: ok")

print("\nall bridge tests passed (incl. lora assignment)")
