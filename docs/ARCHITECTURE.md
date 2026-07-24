# Architecture

## Why single-pass works

Krea 2 (K2) is a **single-stream MMDiT**: every transformer block attends over
one sequence laid out as

```
[ base prompt | region-1 prompt | … | region-N prompt | image tokens | padding ]
```

where image tokens map row-major onto the `H/16 × W/16` latent patch grid.
Because text and image tokens share every attention op, both regional
prompting and regional LoRA can be done in a single forward pass, rather than
running the model once per region and blending noise predictions.

## Regional prompting (attention masking)

All prompts (a shared base plus one per region) are concatenated into one text
sequence. A joint attention mask enforces:

- text↔text: block-diagonal per prompt segment (prompts can't see each other)
- base prompt ↔ every image token (keeps the scene coherent)
- region prompt ↔ only that region's image tokens
- image↔image: full by default, or restricted to same-region + background
  when `restrict_img_attn` is on

The mask is injected through ComfyUI's `optimized_attention_override` hook, so
the stock K2 `_forward` runs untouched. The wrapper identifies its own
combined conditioning by sequence length; any other conditioning (negative
prompt, other nodes) passes through unmasked.

## Regional LoRA (per-token gating)

A LoRA is a weight delta and can't be spatially masked at the weight level —
but it can be masked per **token**. Each targeted `Linear` is class-swapped
(a shallow copy with a mixin prepended to its own class) so it computes

```
y = W x + Σ_r  mask_r ⊙ scale_r · (B_r A_r x)
```

The class swap shares the original module's parameters and state-dict keys, so
ComfyUI's weight patching, lowvram casting, and normal `LoraLoader` stacking
all keep working. `mask_r` is 1 on the region's image tokens (soft values
allowed, for feathering) and on its prompt tokens (so trigger words route
through the LoRA), 0 elsewhere. Adapters are gated by a per-call context, so
the CFG-unconditional pass automatically runs the clean base model.

LoRAs are matched to K2 layers by normalizing PEFT/diffusers (`lora_A/B`) and
kohya (`lora_down/up` + `alpha`) key styles against `blocks.*`, `txtfusion.*`,
and `txtmlp.*` layers.

## Masking scenarios

`_adapt_mask` handles the several sequence shapes a token mask may meet:
full `[text|image]` sequence, image-only slices, and the txtfusion refiner's
text-only blocks (which run at a different sequence length). It returns the
appropriately-sliced mask or `None` (skip) so foreign batch sizes from other
custom nodes never crash.

## Run-level caching

Attention masks and per-adapter token masks depend only on geometry and batch
layout, not on the diffusion step. They're built once per sampling run and
cached on the bundle; LoRA weights are cast to the compute device once per
`(uid, device, dtype)` rather than every layer call. This removed the
per-step GPU allocation churn that fragmented `cudaMallocAsync` and caused
sporadic OOM on tight-VRAM setups.

## Exclusive masks

Where grown/feathered region masks overlap on the token grid, `exclusive_masks`
does a winner-take-all: each token keeps only its strongest region. This stops
two nearby boxes from double-applying LoRAs (and sharing attention) in the gap
between them.

## Builder ↔ backend

The canvas builder (`krea2_builder.py` + `web/js/krea2_builder.js`) serializes
its region state into a hidden widget and emits masks/conditioning/loras on
execute. The **Caption** button prunes the graph to just the captioner feeding
`import_json` and POSTs it to `/api/prompt`, so only that node runs. LoRA
trained-tags come from `/krea2_regional/lora_info` (`server_routes.py`), which
reads just the safetensors JSON header — no weight load, no network.

## Adaptive masks (FreeFuse-style routing)

With `adaptive_masks` on, the wrapper counts model calls per sampling run
(runs are detected by the timestep jumping back up) and, during the first
`adaptive_steps` calls, computes its own q·k affinity between image-token
queries and each region's prompt-token keys at four evenly spaced mid-network
blocks (max over the region's tokens — keying on trigger words — mean over
heads, cond rows only). The attention *mask* can't hide this signal because
the affinity matmul is ours, so tokens outside a region's box still register.

After the capture window the accumulated maps are despeckled, normalized, and
thresholded into refined soft masks — constrained to the grown user boxes in
"refine" mode, free competition in "free" mode, with an empty-discovery
fallback to the drawn box — and all cached attention/LoRA masks rebuild once.
This adapts FreeFuse (arXiv:2510.23515), which showed that early-step
attention in flow models reveals subject placement well enough to route
LoRAs without user masks.

## Layout in the base prompt

The attention masks constrain what each token may READ, but nothing in them
tells the model WHERE to compose a subject — the base prompt drives global
composition, and historically it carried no positional signal. Krea 2's
Qwen3-VL encoder reads positional language and Ideogram-style structured
prompts as soft layout guidance (community "JSON area prompting" workflows
are built on exactly this), so the builder's `layout_in_base` injects the
region layout into the base prompt: natural-language placement sentences
derived from each box's zone and size ("position hints"), or the full
structured caption with 0-1000 bboxes ("full JSON"). Steering (base prompt)
and isolation (masks + token-gated LoRAs) then pull in the same direction,
which is what makes generated subjects actually land in their boxes.

## Region Lock (latent-only anchoring)

A post-CFG sampler hook (`set_model_sampler_post_cfg_function`): once the
schedule passes `region_lock_start`, the model's predicted-clean latent (x0)
is snapshotted; until `region_lock_end`, every step pulls the region back:
`denoised += strength * mask * (snapshot - denoised)`. The mold is the
model's own early prediction, so there is no reference image, no VAE
round-trip, and no latent-format conversion. Windows are compared in sigma
space; a rising sigma resets state (new run). The mask upsamples the live
token-grid region masks (including adaptive refinements) to the latent grid
with a border-safe feather. Limitation stated plainly: this stabilizes
identity WITHIN a generation (late-step drift/mutation); pinning a specific
likeness ACROSS seeds inherently requires an external anchor (a reference
image or a LoRA).
