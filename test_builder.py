"""Tests for krea2_builder.py (mock CLIP + lora folder)."""
import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

# stubs before importing the module
fp = types.ModuleType("folder_paths")
fp.get_filename_list = lambda kind: ["mychar.safetensors",
                                     "styles\\watercolor.safetensors"]
fp.get_full_path_or_raise = lambda kind, name: "/fake/" + name
fp.get_temp_directory = lambda: "/tmp/k2b_test"
sys.modules["folder_paths"] = fp
cu = types.ModuleType("comfy.utils")
cu.load_torch_file = lambda path, safe_load=True: {"fake": path}
pkg = sys.modules.setdefault("comfy", types.ModuleType("comfy"))
pkg.utils = cu
sys.modules["comfy.utils"] = cu

from krea2_builder import Krea2RegionalBuilder


class MockClip:
    def tokenize(self, text):
        return text
    def encode_from_tokens_scheduled(self, tokens):
        n = max(len(str(tokens).split()), 1)
        torch.manual_seed(n)
        return [[torch.randn(1, n, 30720), {}]]


node = Krea2RegionalBuilder()
W = H = 1024

state = {
    "regions": [
        {"x": 0.0, "y": 0.1, "w": 0.45, "h": 0.8,
         "desc": "an armored knight",
         "loras": [{"name": "watercolor", "strength": 0.7}]},
        {"x": 0.55, "y": 0.1, "w": 0.45, "h": 0.8,
         "desc": "a wizard <lora:mychar:0.9>", "loras": []},
    ],
    "base_loras": [{"name": "mychar", "strength": 0.3}],
}

def B(**kw):
    a = dict(clip=MockClip(), width=W, height=H, grow_px=0, feather_px=0,
             base_prompt="", background="", aesthetics="", lighting="",
             medium="", region_append="", import_mode="when empty",
             regions_data="", layout_in_base="off")
    a.update(kw)
    return node.build(**a)

out = B(base_prompt="a duel at dawn", regions_data=json.dumps(state))
regions, base_cond, base_loras, masks, bp, rp, ow, oh = out["result"]

# ---- 1. regions from canvas state, loras from dropdown rows AND inline tags
assert len(regions) == 2 and (ow, oh) == (W, H)
assert [e["label"] for e in regions[0]["loras"]] == ["styles\\watercolor.safetensors"]
assert regions[0]["loras"][0]["strength"] == 0.7
assert [e["label"] for e in regions[1]["loras"]] == ["mychar.safetensors"]
assert regions[1]["loras"][0]["strength"] == 0.9
assert "<lora" not in rp
assert [e["label"] for e in base_loras] == ["mychar.safetensors"]
print("1) canvas state -> regions + loras: ok")

# ---- 2. mask geometry matches the fraction boxes
m = regions[0]["mask"]
assert m.shape == (H, W)
assert m[int(0.5 * H), int(0.2 * W)] == 1.0
assert m[int(0.5 * H), int(0.7 * W)] == 0.0
assert bp == "a duel at dawn"
print("2) mask geometry + base prompt: ok")

# ---- 3. import path: caption JSON seeds the canvas, ui payload carries it
caption = {
    "high_level_description": "city street at night <lora:mychar:0.4>",
    "compositional_deconstruction": {
        "background": "rainy asphalt",
        "elements": [
            {"type": "obj", "bbox": [100, 100, 800, 450],
             "desc": "a neon sign <lora:watercolor:0.6>"},
            {"type": "text", "bbox": [500, 500, 700, 900],
             "text": "OPEN", "desc": "flickering letters"},
            {"type": "obj", "desc": "distant traffic"},
        ],
    },
}
out2 = B(import_json=json.dumps(caption))
r2, _, bl2, _, bp2, rp2, *_ = out2["result"]
assert len(r2) == 2, "two placed elements -> two regions"
assert r2[0]["loras"][0]["label"] == "styles\\watercolor.safetensors"
assert 'the text "OPEN"' in rp2
assert "distant traffic" in bp2 and "rainy asphalt" in bp2
assert [e["label"] for e in bl2] == ["mychar.safetensors"]
seeded = json.loads(out2["ui"]["k2b_state"][0])
assert len(seeded["regions"]) == 2 and seeded["base_loras"], \
    "ui payload must seed the canvas"
print("3) caption import + ui seed: ok")

# ---- 4. editor wins over import in 'when empty' mode
out3 = B(base_prompt="x", regions_data=json.dumps(state),
         import_json=json.dumps(caption))
r3 = out3["result"][0]
assert len(r3) == 2 and "knight" in out3["result"][5]
assert out3["ui"]["k2b_state"] == [], "no reseed when editor has regions"
out4 = B(import_mode="always", regions_data=json.dumps(state),
         import_json=json.dumps(caption))
assert "neon sign" in out4["result"][5], "'always' makes import authoritative"
print("4) import_mode semantics: ok")

# ---- 5. prev_regions + extra_base_loras merge
manual = [{"cond": torch.zeros(1, 4, 30720), "mask": torch.ones(8, 8),
           "loras": []}]
extra = [{"sd": {}, "strength": 1.0, "label": "turbo"}]
out5 = B(base_prompt="p", regions_data=json.dumps(state),
         prev_regions=manual, extra_base_loras=extra)
assert len(out5["result"][0]) == 3
assert out5["result"][2][0]["label"] == "turbo"
print("5) merge inputs: ok")

# ---- 6. text-type regions render as `the text "..."` prompts
state_t = {"regions": [
    {"shape": "rect", "x": 0.1, "y": 0.1, "w": 0.5, "h": 0.3,
     "rtype": "text", "text": "OPEN LATE", "desc": "red neon", "loras": []}],
    "base_loras": []}
out6 = B(regions_data=json.dumps(state_t))
assert out6["result"][5].startswith('the text "OPEN LATE", red neon')
print("6) text regions: ok")

# ---- 7. polygon regions rasterize correctly
tri = {"regions": [
    {"shape": "poly", "desc": "a mountain", "rtype": "obj", "text": "",
     "points": [[0.5, 0.1], [0.1, 0.9], [0.9, 0.9]], "loras": []}],
    "base_loras": []}
out7 = B(regions_data=json.dumps(tri))
m7 = out7["result"][0][0]["mask"]
assert m7.shape == (H, W)
assert m7[int(0.6 * H), int(0.5 * W)] == 1.0, "inside the triangle"
assert m7[int(0.2 * H), int(0.1 * W)] == 0.0, "outside the triangle"
assert m7[int(0.95 * H), int(0.5 * W)] == 0.0, "below the triangle"
print("7) polygon regions: ok")

# ---- 8. style fields compose into the base + import populates k2b_fields
out8 = B(base_prompt="hero shot", background="city street",
         aesthetics="cinematic", lighting="neon glow", medium="35mm photo",
         regions_data=json.dumps(state_t))
bp8 = out8["result"][4]
assert "hero shot" in bp8 and "city street" in bp8
assert "cinematic, neon glow, 35mm photo" in bp8
cap8 = dict(caption)
cap8["style_description"] = {"aesthetics": "grim", "lighting": "dawn",
                             "medium": "oil painting"}
out9 = B(import_json=json.dumps(cap8))
f9 = json.loads(out9["ui"]["k2b_fields"][0])
assert f9["base_prompt"].startswith("city street at night")
assert f9["background"] == "rainy asphalt"
assert f9["aesthetics"] == "grim" and "oil painting" in f9["medium"]
assert "grim, dawn" in out9["result"][4], "imported style must reach the base"
print("8) style fields + import populates widgets: ok")

# ---- 9. bg preview goes out under k2b_bg (nothing renders under the node)
class FakeImg:
    pass
out10 = B(regions_data=json.dumps(state_t))
assert "images" not in out10["ui"], "no node-attached preview anymore"
print("9) clean bg channel: ok")

print("\nall builder tests passed")

# ---- 10. bbox_order: "xy" captions (Qwen-native) land in the same place
cap_xy = json.loads(json.dumps(caption))
cap_xy["bbox_order"] = "xy"
for e in cap_xy["compositional_deconstruction"]["elements"]:
    if "bbox" in e:
        ymin, xmin, ymax, xmax = e["bbox"]
        e["bbox"] = [xmin, ymin, xmax, ymax]
out_xy = B(import_json=json.dumps(cap_xy))
out_yx = B(import_json=json.dumps(caption))
m_xy = out_xy["result"][0][0]["mask"]
m_yx = out_yx["result"][0][0]["mask"]
assert (m_xy - m_yx).abs().max() == 0, "xy captions must map identically"
print("10) bbox_order xy: ok")

print("done")

# ---- 11. layout_in_base: position hints land in the base prompt
out11 = B(base_prompt="a duel at dawn", regions_data=json.dumps(state),
          layout_in_base="position hints")
bp11 = out11["result"][4]
assert "left side of the image" in bp11 and "right side of the image" in bp11, bp11
assert "an armored knight" in bp11 and "a wizard" in bp11
assert "a duel at dawn" in bp11
# off mode keeps subjects out of the base (old behavior)
out11b = B(base_prompt="a duel at dawn", regions_data=json.dumps(state),
           layout_in_base="off")
assert "knight" not in out11b["result"][4]
print("11) position hints in base: ok")

# ---- 12. layout_in_base: full JSON is valid Ideogram-style structure
out12 = B(base_prompt="a duel at dawn", background="a courtyard",
          aesthetics="cinematic", regions_data=json.dumps(state),
          layout_in_base="full JSON")
cap12 = json.loads(out12["result"][4])
els = cap12["compositional_deconstruction"]["elements"]
assert len(els) == 2
assert els[0]["desc"].startswith("an armored knight")
assert len(els[0]["bbox"]) == 4 and all(0 <= v <= 1000 for v in els[0]["bbox"])
# yx order: knight box x:[0,0.45] y:[0.1,0.9] -> [100, 0, 900, 450]
assert els[0]["bbox"] == [100, 0, 900, 450], els[0]["bbox"]
assert cap12["high_level_description"] == "a duel at dawn"
assert cap12["compositional_deconstruction"]["background"] == "a courtyard"
assert "<lora" not in out12["result"][4], "lora tags must not leak into JSON"
print("12) full-JSON structured base: ok")

# ---- 13. text regions carry exact text in both modes; zones make sense
state_z = {"regions": [
    {"shape": "rect", "x": 0.4, "y": 0.02, "w": 0.2, "h": 0.12,
     "rtype": "text", "text": "OPEN", "desc": "neon", "loras": []}],
    "base_loras": []}
out13 = B(regions_data=json.dumps(state_z), layout_in_base="position hints")
assert 'the text "OPEN"' in out13["result"][4]
assert "top of the image" in out13["result"][4], out13["result"][4]
out13j = B(regions_data=json.dumps(state_z), layout_in_base="full JSON")
c13 = json.loads(out13j["result"][4])
assert c13["compositional_deconstruction"]["elements"][0]["text"] == "OPEN"
print("13) text regions + zone naming: ok")

print("layout tests done")

# ---- 14. trigger words must not leak into layout hints
from krea2_builder import _hint_clause, _strip_trigger_lead
assert _hint_clause("xk3wchar, a woman in a red coat") == "a woman in a red coat"
assert _hint_clause("ohwx, a man on a horse") == "a man on a horse"
assert _hint_clause("sunflowers") == "sunflowers", "plain single words survive"
assert _hint_clause("xk3wchar") == "a subject", "tag-like blob alone -> generic"
assert _hint_clause('the text "OPEN", neon') == 'the text "OPEN"'
assert "<" not in _hint_clause("<lora:leftover:1> a castle on a hill")
assert _strip_trigger_lead("m_char, a woman, red coat") == "a woman, red coat"
assert _strip_trigger_lead("sunflowers, golden") == "sunflowers, golden"

state_trig = {"regions": [
    {"shape": "rect", "x": 0.55, "y": 0.55, "w": 0.4, "h": 0.4,
     "rtype": "obj", "text": "",
     "desc": "xk3wchar, a woman in a red coat", "loras": []}],
    "base_loras": []}
out14 = B(regions_data=json.dumps(state_trig), layout_in_base="position hints")
bp14 = out14["result"][4]
assert "xk3wchar" not in bp14, f"trigger leaked into base: {bp14}"
assert "a woman in a red coat" in bp14 and "lower right" in bp14
out14j = B(regions_data=json.dumps(state_trig), layout_in_base="full JSON")
cap14 = json.loads(out14j["result"][4])
assert "xk3wchar" not in cap14["compositional_deconstruction"]["elements"][0]["desc"]
# region conditioning still carries the trigger (that's where the LoRA mask is)
assert "xk3wchar" in out14["result"][5]
print("14) trigger words kept out of base representations: ok")
