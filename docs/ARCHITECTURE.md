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
