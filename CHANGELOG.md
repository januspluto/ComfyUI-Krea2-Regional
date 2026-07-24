# Changelog

## 1.1.0 (unreleased)

- Fix: compatibility with current ComfyUI master, which added a
  `ref_latents` positional parameter to Krea2's forward — the wrapper is now
  signature-agnostic, so future parameter insertions can't break it.
- Regression test for issue #1: two LoKr characters + a standard LoRA
  stacking on the same layers all apply together.

- Perf: adaptive-mask capture now runs its affinity matmul in the model's
  native dtype (was fp32 — a full query-tensor copy plus slow matmuls during
  the first steps). Note: `layout_in_base` adds prompt tokens, and attention
  cost scales quadratically with sequence length — "position hints" costs a
  few percent, "full JSON" more; set it to "off" to reclaim the time.

- **Region Lock** (`region_lock_strength` on Apply Regional): latent-only
  identity/structure anchoring — snapshot each region's predicted-clean
  latent once identity has formed (`region_lock_start`) and pull the region
  back toward it each step until `region_lock_end`. No reference image, no
  VAE round-trip; masks follow adaptive refinement automatically.
- Fix: LoRA trigger words prepended to region prompts leaked into the
  `layout_in_base` position hints and could be RENDERED as literal text in
  the hinted corner. Hints now pick a natural-language clause; full-JSON
  descs strip tag-like leading triggers; region conditioning keeps them.
- Fix: lock/feather masks no longer weaken at image borders
  (`count_include_pad`).

- **LoKr support**: Kronecker-factored LoRAs (recent ai-toolkit output) now
  load and apply regionally — previously they matched 0 layers silently.
  Verified numerically against the materialized Kronecker product.
- Fix: grid snap checkbox went dead after loading a workflow (stale object
  reference in the toolbar closures).

- **`layout_in_base`** (Builder, default "position hints"): inject the region
  layout into the base prompt — natural-language placements or a full
  Ideogram-style JSON — so Krea 2 composes subjects where the boxes are,
  instead of the masks fighting an unguided composition.

- **Adaptive masks** (`adaptive_masks` on Apply Regional): FreeFuse-style
  token routing — discover each subject's real silhouette from early-step
  prompt/image attention and snap the region masks to it. "refine boxes"
  and "free" modes, with `adaptive_steps` / `adaptive_threshold` controls.
- **`base_loras_exclude_regions`**: base/style LoRAs apply everywhere except
  inside subject regions.

## Unreleased

- Fix: the builder's per-region LoRA dropdown now refreshes on **R**
  (refresh node definitions) — newly added loras appear without a full page
  reload, via a `refreshComboInNodes` hook.

- Add **Krea 2 Empty Latent Image** node (aspect buckets + megapixels dial,
  VAE-correct 16-channel latents, WIDTH/HEIGHT outputs).
- Refactor shared caption/mask/LoRA helpers into `caption_utils.py`; the
  Ideogram bridge node is now optional (the canvas builder supersedes it).

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
