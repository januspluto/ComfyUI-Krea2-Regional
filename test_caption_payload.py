"""Regression test for the builder's Caption-button payload contract.

The Caption button sends the FULL graph plus a PreviewAny probe wired to the
captioner, targeted via partial_execution_targets. This test proves that
payload validates against real ComfyUI (and that the older pruned-graph
approach, which produced the reported 400, does not)."""
import asyncio
import os
import sys

sys.path.insert(0, os.environ.get("COMFYUI_PATH", os.path.expanduser("~/ComfyUI")))
from comfy.cli_args import args
args.cpu = True

import nodes
import execution

r = nodes.init_extra_nodes()
if asyncio.iscoroutine(r):
    asyncio.run(r)

assert "PreviewAny" in nodes.NODE_CLASS_MAPPINGS, \
    "PreviewAny output node must exist for the Caption button to work"


class FakeCaptioner:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "image": ("IMAGE",)}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "go"
    CATEGORY = "test"
    def go(self, clip, image):
        return ("{}",)


class FakeClip:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "go"
    CATEGORY = "test"
    def go(self):
        return (None,)


class FakeImg:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "go"
    CATEGORY = "test"
    def go(self):
        return (None,)


nodes.NODE_CLASS_MAPPINGS.update(
    FakeCaptioner=FakeCaptioner, FakeClip=FakeClip, FakeImg=FakeImg)


def base_graph():
    # captioner fed by two sources, PLUS an unrelated node with a MISSING
    # required input (image) — mirrors a real workflow with a sampler branch
    # that isn't ready. The Caption run must ignore it.
    return {
        "1": {"class_type": "FakeClip", "inputs": {}},
        "2": {"class_type": "FakeImg", "inputs": {}},
        "3": {"class_type": "FakeCaptioner",
              "inputs": {"clip": ["1", 0], "image": ["2", 0]}},
        "4": {"class_type": "FakeCaptioner", "inputs": {"clip": ["1", 0]}},
    }


# ---- 1. the NEW payload validates and targets only the probe
prompt = base_graph()
prompt["10000000"] = {"class_type": "PreviewAny",
                      "inputs": {"source": ["3", 0]},
                      "_meta": {"title": "k2b_caption_probe"}}
valid = asyncio.run(execution.validate_prompt("t1", prompt, ["10000000"]))
assert valid[0] is True, f"payload should validate, got {valid[1]}"
assert valid[2] == ["10000000"], valid[2]
print("1) full-graph + PreviewAny probe validates: ok")

# ---- 2. the broken unrelated branch does NOT block it
assert "4" in prompt and prompt["4"]["inputs"].get("image") is None
print("2) unrelated broken branch ignored: ok")

# ---- 3. the OLD pruned approach fails exactly as the user reported (400)
pruned = base_graph()
del pruned["4"]
valid_old = asyncio.run(execution.validate_prompt("t2", pruned, None))
assert valid_old[0] is False
assert valid_old[1]["type"] == "prompt_no_outputs", valid_old[1]
print("3) old pruned approach reproduces prompt_no_outputs (the 400): ok")

print("\nall caption-payload tests passed")
