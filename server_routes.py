"""Server routes for the Krea2 Regional Builder frontend.

/krea2_regional/lora_info?name=<lora>  -> trained tags + metadata read from
the safetensors header (no network). Used by the LoRA info panel and the
"paste tags" button in the canvas builder.
"""

from __future__ import annotations

import json
import logging
import os
import struct

try:
    from server import PromptServer
    from aiohttp import web
    import folder_paths
    _HAVE_SERVER = True
except Exception:  # imported in tests without a running server
    _HAVE_SERVER = False


def _read_safetensors_metadata(path: str) -> dict:
    """Read only the JSON header of a .safetensors file (first 8 bytes give
    the header length), returning its __metadata__ dict. Cheap: no tensors."""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n <= 0 or n > 100_000_000:
                return {}
            header = json.loads(f.read(n).decode("utf-8", "replace"))
        return header.get("__metadata__", {}) or {}
    except Exception as e:
        logging.debug("[Krea2Regional] metadata read failed for %s: %s",
                      path, e)
        return {}


def _tags_from_metadata(meta: dict) -> list[str]:
    """Extract trained trigger tags from kohya/ai-toolkit-style metadata."""
    freq = meta.get("ss_tag_frequency")
    tags: list[tuple[str, int]] = []
    if freq:
        try:
            data = json.loads(freq) if isinstance(freq, str) else freq
            merged: dict[str, int] = {}
            for bucket in data.values():
                if isinstance(bucket, dict):
                    for tag, count in bucket.items():
                        merged[tag] = merged.get(tag, 0) + int(count)
            tags = sorted(merged.items(), key=lambda kv: -kv[1])
        except Exception:
            tags = []
    if not tags:
        for key in ("ss_trigger_words", "trigger_words", "activation_text",
                    "ss_activation_text"):
            v = meta.get(key)
            if v:
                parts = v if isinstance(v, list) else str(v).split(",")
                return [p.strip() for p in parts if str(p).strip()]
    return [t for t, _ in tags]


def get_lora_info(name: str) -> dict:
    path = None
    try:
        path = folder_paths.get_full_path("loras", name)
    except Exception:
        path = None
    if not path or not os.path.exists(path):
        return {"name": name, "found": False, "tags": [], "metadata": {}}
    meta = _read_safetensors_metadata(path) if path.endswith(
        ".safetensors") else {}
    keep = {k: v for k, v in meta.items()
            if k in ("ss_output_name", "ss_base_model_version",
                     "ss_network_dim", "ss_network_alpha", "modelspec.title",
                     "modelspec.architecture", "ss_sd_model_name")}
    return {"name": name, "found": True,
            "tags": _tags_from_metadata(meta)[:60], "metadata": keep}


if _HAVE_SERVER:
    @PromptServer.instance.routes.get("/krea2_regional/lora_info")
    async def _lora_info_route(request):
        name = request.rel_url.query.get("name", "")
        if not name:
            return web.json_response({"error": "missing name"}, status=400)
        return web.json_response(get_lora_info(name))
