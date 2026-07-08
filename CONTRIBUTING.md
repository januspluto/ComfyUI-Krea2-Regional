# Contributing

Thanks for your interest in improving ComfyUI-Krea2-Regional.

## Running the tests

All tests run on CPU and need no GPU. Three of them import ComfyUI's real
Krea 2 module, so point `COMFYUI_PATH` at a ComfyUI checkout:

```bash
COMFYUI_PATH=/path/to/ComfyUI python test_nodes.py
COMFYUI_PATH=/path/to/ComfyUI python test_bridge.py
COMFYUI_PATH=/path/to/ComfyUI python test_builder.py
python test_server_routes.py
```

The first run in a fresh ComfyUI env may need its Python deps
(`torch`, `einops`, `safetensors`, `transformers`). `test_server_routes.py`
is standalone.

Please run all four before opening a PR, and add a test for new behavior —
the existing tests exercise the real model, so regressions are caught early.

## Frontend

The canvas UI lives in `web/js/krea2_builder.js` (vanilla JS, no build step).
`node --check web/js/krea2_builder.js` catches syntax errors. Actual DOM
behavior has to be verified in a running ComfyUI + browser; please describe
what you tested and include a screenshot for UI changes.

## Scope

Layout, structure, and rationale for the core masking/LoRA machinery are in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Read it before changing the
attention masks or LoRA injection — those pieces are subtle and interact with
ComfyUI's ModelPatcher.

## Reporting LoRA-compatibility issues

If a LoRA logs "keys didn't match", paste a handful of its state-dict key
names into an issue; extending the key normalizer is usually a one-liner.
