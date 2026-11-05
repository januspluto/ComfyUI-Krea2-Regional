# Changelog

## 1.2.1 — 2026-11-05

### Regional Builder

- Restore and stabilize canvas attachment, image preview, box editing, and prompt-panel sizing.
- Make High-level description, Background, Aesthetics, Lighting, and Medium independently resizable; preserve their sizes and values when workflows are saved or restored.
- Fix Clear to reset all five prompt fields, region append text, region boxes, and base LoRAs.
- Fix Copy/Paste for caption/layout JSON, raw regions, and plain text in focused prompt fields.
- Keep caption imports, execution feedback, and visible textareas synchronized with serialized widget values; isolate text-entry shortcuts from graph shortcuts.
- Improve builder discoverability: report regions skipped for empty prompts and clarify canvas background loading.

### Regional generation and references

- Support native Krea2 reference latents with the `[text | target | references]` attention layout, including reference-aware masks, LoRA gates, and caches (issue #3).
- Make reference tokens available as global context while keeping regional LoRA deltas off those reference tokens.
- Improve stability for repeated sampling with DynamicVRAM/AIMDO via the `regional safe` model-management option.
- Update Apply Regional defaults: `exclusive_masks=false`, `adaptive_masks=refine boxes`, `adaptive_steps=2`, `adaptive_threshold=0.85`, `region_lock_strength=0.35`, `region_lock_start=0.25`, `region_lock_end=0.70`, `restrict_end_percent=1.0`, `dynamic_vram_mode=regional safe`, and `unmaskable_layers=apply globally`.

### Regional Detailer

- Make `detail_target=auto` safer for large/full-body subjects by limiting their refinement to an estimated head area, with a skip threshold for very large regions.
- Lower the default denoise to `0.18` for a more conservative second pass.

### Tests and documentation

- Add regression coverage for native Krea2 references, builder prompt synchronization, and Detailer auto targeting.
- Refresh user-facing documentation and installation guidance.

## 1.2.0 — 2026-08-06

Peer-reviewed feature release: subject-aware Regional Detailer, full lasso
editing, explicit depth ordering, and a hardening pass across state
handling, LoRA caching, and overlap semantics.

### New node: Krea 2 - Regional Detailer

- Subject-aware refinement pass run after the first sampler + VAE decode.
  Feed it subject silhouettes from any detector (`detection_masks` MASK
  batch from SAM3 etc., or Impact Pack `segs` — duck-typed, no
  dependency): detections are matched to Builder regions by containment x
  coverage with a bounded nearest-center rescue for drifted subjects, each
  subject is cropped around its detected bbox, resampled with that
  region's prompt conditioning and LoRAs applied plainly, and composited
  back through the detected silhouette (grown + feathered). Only each
  subject's own pixels are repainted. Regions without a matching subject
  are left untouched; a connected detector yielding no masks aborts loudly
  instead of silently falling back to box mode. Box mode remains available
  when no detector is connected (bleed-prone with overlapping subjects —
  see tooltip).
- Latent noise-mask anchoring: the silhouette constrains every sampling
  step, not just the final composite. Crop context always comes from the
  immutable first-pass image.
- Upscale-only, aspect-preserving crop sizing (subjects larger than
  `detail_size` sample at native resolution — no softness seam); 5-D
  video-VAE decodes (WanVAE) handled like core VAEDecode; per-subject
  timing logged.
- `shift` option: default "model default" inherits the initial image's
  sigma schedule (recommended, and required for Krea 2 Turbo which is
  distilled at fixed mu). "auto (RAW only, per-crop)" recomputes mu from
  Krea 2 RAW's 256-1280px token endpoints, clamped, reusing the model's
  own sampling object. Default denoise 0.25.
- Recommended wiring: Builder `base_conditioning` (not Apply Regional's
  combined output) into the detailer's `base_conditioning`.

### Builder

- Lasso regions are fully editable: drag vertices (filled circles),
  press-drag an edge midpoint (hollow circles) to split the edge with a
  new vertex pen-tool style, double-click an edge to insert, alt-click a
  vertex to delete (min 3). Bbox scale handles sit just outside the box
  and affine-remap all points; grab-offset compensated.
- Rect <-> lasso conversion button on the selected region.
- Region list is explicit front-to-back depth with up/down reorder
  buttons; top of the list is front and wins overlaps.
- Layout overhaul: canvas and node size finally agree. Sizing is driven by
  node width in graph units (zoom-dependent DOM width no longer
  participates), the width LiteGraph passes during resize drags is
  honored, toolbar chrome height is measured (wrapped button rows),
  the root container is pinned so panels/toolbars span the node, the
  canvas refits live during resize drags, and panel rebuilds schedule a
  settled-DOM height sync — no more shrink loops, overflow at zoom, or
  the region editor hanging past the node bottom.

### Apply Regional / backend

- `unmaskable_layers` is now actually exposed (optional widget, default
  "skip") and forwarded to region-LoRA injection; it was implemented but
  dead plumbing. Base LoRAs keep prior behavior.
- Krea 2 native reference latents are rejected with a clear error while
  regional routing is active — reference tokens extend the image sequence
  past the region masks and previously bypassed restrictions silently.
- Overlap resolution: exact ties deterministically go to the front
  (earlier) region via equality-based selection; genuine strength
  differences of any size win outright. (Interim epsilon-bias approach
  removed after review.)

### State & caching hardening

- Centralized `normalize_builder_state` gate on every state entry point
  (widget JSON, caption import): malformed rects/polys, non-finite
  numbers, wrong-typed fields and junk rows are skipped with warnings
  instead of crashing; coordinates clamped; unknown fields preserved.
- LoRA state-dict cache is a bounded LRU (6 entries) keyed by resolved
  file identity (path, mtime, size): replacing a file invalidates its
  entry; `clear_lora_cache()` exported. Duplicate LoRAs are deduplicated
  after path resolution, keep-first precedence: Builder selection >
  `extra_base_loras` > inline `<lora:...>` tags. Note: base-lora assembly
  order flipped to make this precedence hold; if the same file appeared in
  both, the Builder's strength now wins (logged).


## 1.1.0

- **Scheduled restrict** (`restrict_end_percent` on Apply Regional): keep
  `restrict_img_attn` isolation during the early identity-forming steps,
  then release it so late steps integrate seams/lighting. Sigma-space
  window; token masks and adaptive state survive the mid-run switch.

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
