"""Tests for server_routes.py metadata reading (no running server)."""
import json
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server_routes as sr


def _write_safetensors(path, metadata):
    """Minimal valid .safetensors: header len + JSON header + 1 tiny tensor."""
    header = {"__metadata__": metadata,
              "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\x00\x00\x00\x00")


with tempfile.TemporaryDirectory() as d:
    # ---- 1. ss_tag_frequency (kohya) -> sorted by count
    p1 = os.path.join(d, "a.safetensors")
    _write_safetensors(p1, {
        "ss_tag_frequency": json.dumps({
            "img": {"mychar": 40, "red coat": 12, "smile": 3}}),
        "ss_output_name": "MyChar",
        "ss_base_model_version": "krea2",
    })
    meta = sr._read_safetensors_metadata(p1)
    assert meta["ss_output_name"] == "MyChar"
    tags = sr._tags_from_metadata(meta)
    assert tags[:3] == ["mychar", "red coat", "smile"], tags
    print("1) tag frequency parse + ordering: ok")

    # ---- 2. trigger_words fallback
    p2 = os.path.join(d, "b.safetensors")
    _write_safetensors(p2, {"ss_trigger_words": "foo, bar, baz"})
    tags2 = sr._tags_from_metadata(sr._read_safetensors_metadata(p2))
    assert tags2 == ["foo", "bar", "baz"], tags2
    print("2) trigger_words fallback: ok")

    # ---- 3. activation_text fallback
    p3 = os.path.join(d, "c.safetensors")
    _write_safetensors(p3, {"activation_text": "trigger phrase here"})
    tags3 = sr._tags_from_metadata(sr._read_safetensors_metadata(p3))
    assert tags3 == ["trigger phrase here"], tags3
    print("3) activation_text fallback: ok")

    # ---- 4. no metadata -> empty, no crash
    p4 = os.path.join(d, "d.safetensors")
    _write_safetensors(p4, {})
    assert sr._tags_from_metadata(sr._read_safetensors_metadata(p4)) == []
    print("4) empty metadata: ok")

    # ---- 5. get_lora_info via a stubbed folder_paths
    import types
    fp = types.ModuleType("folder_paths")
    fp.get_full_path = lambda kind, name: p1 if name == "a.safetensors" else None
    sys.modules["folder_paths"] = fp
    sr.folder_paths = fp
    info = sr.get_lora_info("a.safetensors")
    assert info["found"] and "mychar" in info["tags"]
    assert info["metadata"].get("ss_output_name") == "MyChar"
    missing = sr.get_lora_info("nope.safetensors")
    assert missing["found"] is False and missing["tags"] == []
    print("5) get_lora_info: ok")

    # ---- 6. junk file doesn't throw
    p6 = os.path.join(d, "junk.safetensors")
    with open(p6, "wb") as f:
        f.write(b"not a safetensors file at all")
    assert sr._read_safetensors_metadata(p6) == {}
    print("6) corrupt file tolerance: ok")

print("\nall server_routes tests passed")
