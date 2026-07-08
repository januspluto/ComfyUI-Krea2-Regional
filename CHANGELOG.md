# Changelog

## Unreleased

- Fix: **Caption button** returned "caption run rejected (400)". It now sends
  the full graph with a hidden PreviewAny probe targeted via
  `partial_execution_targets`, instead of a pruned graph with no output node
  (which ComfyUI rejects as `prompt_no_outputs`). Covered by
  `test_caption_payload.py`.

## 1.0.0

First public release.

- Single-pass regional prompting for Krea 2 via joint attention masking.
- Per-region LoRA via per-token gated deltas (stacks with normal LoRA loaders;
  covers `blocks.*` and `txtfusion.*` layers).
- Canvas builder node: rect + lasso regions, obj/text region types, searchable
  per-region LoRA dropdowns with a trained-tag info panel, base/background/
  style fields, grid/snap/guides, live reference background + Grab BG, pop-out
  editor, and a Caption button that runs only a connected captioner.
- Apply Regional options: `restrict_img_attn`, `exclusive_masks`.
- Ideogram-JSON bridge node and Qwen3.5 VL captioner system prompt.
- Run-level mask/LoRA caching to avoid per-step allocation churn.
