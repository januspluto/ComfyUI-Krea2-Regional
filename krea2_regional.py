"""Krea 2 (K2) regional prompting + per-region LoRA for ComfyUI — single-pass.

Instead of ComfyUI's native multi-pass masked conditioning (one model pass per
region, blended in latent space), this pack runs ONE pass per step:

* All prompts (base + regions) are concatenated into a single Krea2 text
  sequence. K2 is a single-stream MMDiT, so text and image tokens share every
  attention op; a joint attention mask injected via the
  ``optimized_attention_override`` hook keeps each region's image tokens
  talking only to their own prompt (plus the shared base prompt).

* LoRAs are applied per *token*: eligible Linears inside the DiT blocks get a
  class-swap patch that adds ``mask ⊙ scale · (B A x)`` on top of the base
  projection, where the mask covers the region's image tokens (soft values
  allowed) and its prompt tokens. Layers, weights, and state-dict keys are
  untouched, so normal LoraLoader weight patches still apply cleanly on top.

Nodes:
  Krea2LoRA            -> chainable list of LoRA files (+ strength)
  Krea2RegionalPrompt  -> conditioning + mask (+ optional LoRA list) -> region chain
  Krea2ApplyRegional   -> model + base conditioning + regions -> (MODEL, CONDITIONING)
"""

from __future__ import annotations

import copy
import itertools
import logging
import re

import torch
import torch.nn.functional as F

import comfy.patcher_extension
import comfy.utils
from comfy.patcher_extension import WrappersMP

BUNDLE_KEY = "krea2_regional"
CONSUMED_KEY = "krea2_regional_consumed"

# ---------------------------------------------------------------------------
# Runtime context: token masks for the *current* diffusion-model call.
# Adapters only contribute when their uid is present, so any call where the
# wrapper didn't populate this (negative prompt, other models) runs clean.
# ---------------------------------------------------------------------------


class _Ctx:
    def __init__(self):
        self.token_masks: dict[str, torch.Tensor] = {}  # uid -> (B, L, 1)
        self.txtlen = 0
        self.imglen = 0
        self.txtlayers = -1

    def clear(self):
        self.token_masks = {}
        self.txtlen = 0
        self.imglen = 0
        self.txtlayers = -1


CTX = _Ctx()
_UID = itertools.count()


# ---------------------------------------------------------------------------
# LoRA checkpoint parsing (kohya / peft / diffusers key styles)
# ---------------------------------------------------------------------------

_PREFIXES = (
    "diffusion_model.",
    "transformer.",
    "model.",
    "lora_unet_",
    "lora_transformer_",
)


def _normalize_lora_sd(sd: dict) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for k, v in sd.items():
        m = re.match(r"^(.*?)\.(lora_A|lora_down)\.weight$", k)
        if m:
            groups.setdefault(m.group(1), {})["down"] = v
            continue
        m = re.match(r"^(.*?)\.(lora_B|lora_up)\.weight$", k)
        if m:
            groups.setdefault(m.group(1), {})["up"] = v
            continue
        m = re.match(r"^(.*?)\.alpha$", k)
        if m:
            groups.setdefault(m.group(1), {})["alpha"] = float(v)
            continue
    out = {}
    for key, g in groups.items():
        if "down" not in g or "up" not in g:
            continue
        for p in _PREFIXES:
            if key.startswith(p):
                key = key[len(p):]
                break
        out[key] = g
    return out


# ---------------------------------------------------------------------------
# Token-masked LoRA via class swap (paths / params / state_dict unchanged)
# ---------------------------------------------------------------------------


def _adapt_mask(m, x):
    """Fit a full-sequence (B, L, 1) token mask to whatever the current layer
    sees: the joint sequence (DiT blocks), text-only (txtfusion refiners /
    txtmlp), image-only (patch embed), per-token rows (txtfusion layerwise
    blocks, where the batch dim is batch*txtlen), or the 4D projector."""
    if x.ndim == 3:
        s = x.shape[1]
        if s == m.shape[1]:
            mm = m
        elif s == CTX.txtlen:
            mm = m[:, :s]
        elif s == CTX.imglen:
            mm = m[:, CTX.txtlen:CTX.txtlen + s]
        elif (s == CTX.txtlayers and CTX.txtlen > 0
              and x.shape[0] % CTX.txtlen == 0):
            vec = m[:, :CTX.txtlen, 0]            # (Bm, txtlen)
            reps = x.shape[0] // CTX.txtlen
            if vec.shape[0] == 1:
                flat = vec[0].repeat(reps)
            elif vec.shape[0] == reps:
                flat = vec.reshape(-1)
            else:
                return None
            return flat[:, None, None]            # per-row gate
        else:
            return None
        return mm if mm.shape[0] in (1, x.shape[0]) else None
    if x.ndim == 4 and x.shape[1] == CTX.txtlen:  # txtfusion projector
        mm = m[:, :CTX.txtlen]
        return mm[..., None] if mm.shape[0] in (1, x.shape[0]) else None
    return None


class _RegionalLoRAMixin:
    """Mixin prepended onto the layer's own class; adds masked LoRA deltas."""

    def forward(self, x, *args, **kwargs):
        y = super().forward(x, *args, **kwargs)
        adapters = getattr(self, "regional_adapters", None)
        tm = CTX.token_masks
        if not adapters or not tm or not torch.is_tensor(x) \
                or x.ndim not in (3, 4):
            return y
        cast = self.__dict__.setdefault("_k2b_cast", {})
        for uid, (down, up, scale) in adapters.items():
            m = tm.get(uid)
            if m is None:
                continue
            mm = _adapt_mask(m, x)
            if mm is None:
                continue
            ck = (uid, x.device, x.dtype)
            if ck not in cast:
                cast[ck] = (down.to(device=x.device, dtype=x.dtype),
                            up.to(device=x.device, dtype=x.dtype))
            d, u = cast[ck]
            delta = F.linear(F.linear(x, d), u)
            y = y + delta * (scale * mm.to(device=y.device, dtype=y.dtype))
        return y


_CLS_CACHE: dict[type, type] = {}


def _regional_class(base_cls: type) -> type:
    if base_cls not in _CLS_CACHE:
        _CLS_CACHE[base_cls] = type(
            "Regional" + base_cls.__name__, (_RegionalLoRAMixin, base_cls), {}
        )
    return _CLS_CACHE[base_cls]


def _patched_linear(base: torch.nn.Module) -> torch.nn.Module:
    """Shallow-copy `base` and swap in the regional subclass.

    Parameters/buffers dicts are shared with the original, so weight keys,
    ModelPatcher weight patching, and lowvram casting all behave exactly as
    before. Each ApplyRegional execution gets its own copy with its own
    `regional_adapters`, so model clones never interfere.
    """
    patched = copy.copy(base)
    if isinstance(base, _RegionalLoRAMixin):
        patched.regional_adapters = dict(getattr(base, "regional_adapters", {}))
    else:
        patched.__class__ = _regional_class(base.__class__)
        patched.regional_adapters = {}
    return patched


def _inject_lora(model_patcher, lora_sd: dict, uid: str, strength: float) -> int:
    """Attach one LoRA (as adapter `uid`) onto the DiT block Linears."""
    groups = _normalize_lora_sd(lora_sd)

    diffusion_model = model_patcher.get_model_object("diffusion_model")
    lookup: dict[str, str] = {}
    for name, mod in diffusion_model.named_modules():
        if (name.startswith(("blocks.", "txtfusion.", "txtmlp", "first",
                             "last")) and hasattr(mod, "weight")
                and callable(mod)):
            if mod.weight is not None and mod.weight.ndim == 2:
                lookup[name] = name
                lookup[name.replace(".", "_")] = name

    patched, skipped = 0, []
    for key, g in groups.items():
        path = lookup.get(key) or lookup.get(key.replace(".", "_"))
        if path is None:
            skipped.append(key)
            continue
        full = "diffusion_model." + path
        target = _patched_linear(model_patcher.get_model_object(full))
        rank = g["down"].shape[0]
        alpha = g.get("alpha", rank)
        target.regional_adapters[uid] = (
            g["down"],
            g["up"],
            float(strength) * float(alpha) / float(rank),
        )
        model_patcher.add_object_patch(full, target)
        patched += 1

    if skipped:
        logging.warning(
            "[Krea2Regional] lora '%s': %d keys didn't match any DiT block layer "
            "(e.g. %s) — is this a Krea 2 LoRA?", uid, len(skipped), skipped[:3]
        )
    return patched


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------


def _latent_mask(mask: torch.Tensor, h: int, w: int, device) -> torch.Tensor:
    """(B?, H, W) MASK in [0,1] -> flattened soft mask on the patch grid."""
    if mask.ndim == 2:
        mask = mask[None]
    m = mask[:1].float()[None]  # (1, 1, H, W)
    m = F.interpolate(m, size=(h, w), mode="area")[0, 0]
    return m.clamp(0, 1).reshape(-1).to(device)


def _exclusive_soft(soft):
    """Winner-take-all where region masks overlap on the token grid: each
    image token keeps only its strongest region. Stops two grown/feathered
    boxes from double-applying LoRAs (and sharing attention) in the gap
    between them."""
    if soft.shape[0] < 2:
        return soft
    winner = torch.zeros_like(soft, dtype=torch.bool)
    winner.scatter_(0, soft.argmax(dim=0, keepdim=True), True)
    return soft * winner


def _build_allow(segments, region_img_hard, imglen, restrict_img_attn, device):
    """(L, L) bool, True = may attend. segments: list of (start, end) text spans;
    segments[0] is the base prompt. region_img_hard: (R, imglen) bool."""
    txtlen = segments[-1][1]
    L = txtlen + imglen
    i0, i1 = txtlen, txtlen + imglen
    allow = torch.zeros(L, L, dtype=torch.bool, device=device)

    for s, e in segments:  # prompts are mutually isolated
        allow[s:e, s:e] = True

    b0, b1 = segments[0]  # base prompt <-> every image token
    allow[b0:b1, i0:i1] = True
    allow[i0:i1, b0:b1] = True

    for r, (s, e) in enumerate(segments[1:]):
        allow[s:e, i0:i1] |= region_img_hard[r][None, :]
        allow[i0:i1, s:e] |= region_img_hard[r][:, None]

    if restrict_img_attn and region_img_hard.shape[0] > 0:
        shared = (region_img_hard.T.float() @ region_img_hard.float()) > 0
        bg = ~region_img_hard.any(dim=0)
        img_allow = shared | bg[None, :] | bg[:, None]
        img_allow |= torch.eye(imglen, dtype=torch.bool, device=device)
        allow[i0:i1, i0:i1] = img_allow
    else:
        allow[i0:i1, i0:i1] = True

    txt_allow = torch.zeros(txtlen, txtlen, dtype=torch.bool, device=device)
    for s, e in segments:
        txt_allow[s:e, s:e] = True
    return allow, txt_allow


def _build_token_masks(bundle, segments, soft_masks, imglen, batch_rows, device):
    """Per-adapter (B, L, 1) masks. batch_rows: list of bool, True = regional
    (cond) rows, False = rows that must run the clean base model."""
    txtlen = segments[-1][1]
    L = txtlen + imglen
    row = {}

    def bump(uid, vec):
        row[uid] = torch.maximum(row[uid], vec) if uid in row else vec

    for spec in bundle["loras"]:
        uid, weight, seg_idx, mask_idx = (
            spec["uid"], spec["weight"], spec["seg"], spec["mask_idx"],
        )
        vec = torch.zeros(L, device=device)
        if mask_idx is None:  # global (base) lora: whole prompt + whole image
            vec[:] = weight
        else:
            vec[txtlen:] = soft_masks[mask_idx] * weight
            if seg_idx is not None:
                s, e = segments[seg_idx]
                vec[s:e] = weight  # trigger words go through the LoRA too
        bump(uid, vec)

    out = {}
    if all(batch_rows):
        for uid, vec in row.items():
            out[uid] = vec[None, :, None]  # (1, L, 1) -> broadcasts to any batch
    else:
        rows = torch.tensor(batch_rows, device=device, dtype=torch.float32)
        for uid, vec in row.items():
            out[uid] = (rows[:, None] * vec[None, :])[:, :, None]  # (B, L, 1)
    return out


# ---------------------------------------------------------------------------
# The diffusion-model wrapper (WrappersMP.DIFFUSION_MODEL)
# ---------------------------------------------------------------------------


def _diffusion_wrapper(executor, x, timesteps, context, attention_mask=None,
                       transformer_options={}, **kwargs):
    bundle = transformer_options.get(BUNDLE_KEY)
    if (
        bundle is None
        or context is None
        or transformer_options.get(CONSUMED_KEY)
        or context.shape[1] != bundle["txt_total"]
    ):
        # negative prompt / unrelated conditioning / chained duplicate -> clean pass
        return executor(x, timesteps, context, attention_mask,
                        transformer_options=transformer_options, **kwargs)

    model = executor.class_obj  # the SingleStreamDiT instance
    patch = model.patch
    # Mirror _forward exactly: a 5D latent (B, C, T, H, W) is flattened to
    # (B*T, C, H, W) before patchify, so the DiT's effective batch is B*T.
    if x.ndim == 5:
        bs = x.shape[0] * x.shape[2]
    else:
        bs = x.shape[0]
    h_ = -(-x.shape[-2] // patch)  # matches pad_to_patch_size + patchify
    w_ = -(-x.shape[-1] // patch)
    imglen = h_ * w_
    device = x.device

    # which batch rows are regional (cond) vs must stay clean (uncond)?
    cond_or_uncond = transformer_options.get("cond_or_uncond", [0])
    chunks = max(len(cond_or_uncond), 1)
    per = max(bs // chunks, 1)
    batch_rows = []
    for c in cond_or_uncond:
        batch_rows.extend([c == 0] * per)
    batch_rows = (batch_rows + [True] * bs)[:bs]

    segments = bundle["segments"]
    txtlen = bundle["txt_total"]
    L = txtlen + imglen

    # ---- run-level cache: the masks depend only on geometry + batch layout,
    # so build them ONCE per sampling run instead of every step. This kills
    # the per-step GPU alloc churn that fragments cudaMallocAsync and caused
    # sporadic OOM on tight-VRAM setups.
    ck = (h_, w_, bs, tuple(batch_rows), str(device), txtlen)
    rt = bundle.get("_rt")
    if rt is None or rt.get("key") != ck:
        soft = (
            torch.stack([_latent_mask(m, h_, w_, device)
                         for m in bundle["masks"]])
            if bundle["masks"] else torch.zeros(0, imglen, device=device)
        )
        if bundle.get("exclusive_masks", True):
            soft = _exclusive_soft(soft)
        hard = soft > 0.5

        allow, txt_allow = _build_allow(
            segments, hard, imglen, bundle["restrict_img_attn"], device
        )
        # When every row is regional (the usual turbo cfg=1 case), keep batch
        # dim 1 so the mask broadcasts against ANY caller batch — including
        # other custom nodes (e.g. Krea2T-Enhancer) that re-run txtfusion
        # blocks with their own batch size. Only expand per-row when
        # cond/uncond rows are actually mixed.
        uniform = all(batch_rows)
        if uniform:
            joint = allow[None, None]      # (1, 1, L, L)
            txtm = txt_allow[None, None]
        else:
            joint = allow[None].repeat(bs, 1, 1)
            txtm = txt_allow[None].repeat(bs, 1, 1)
            for i, regional in enumerate(batch_rows):
                if not regional:
                    joint[i] = True
                    txtm[i] = True
            joint = joint[:, None]  # (B, 1, L, L)
            txtm = txtm[:, None]
        rt = {"key": ck, "soft": soft, "joint": joint, "txtm": txtm,
              "additive": {}, "token_masks": None}
        bundle["_rt"] = rt
    soft = rt["soft"]
    joint = rt["joint"]
    txtm = rt["txtm"]
    additive_cache = rt["additive"]

    def _additive(sel: torch.Tensor, dtype):
        key = (sel.data_ptr(), dtype)
        if key not in additive_cache:
            neg = torch.finfo(dtype).min if dtype.is_floating_point else -1e9
            additive_cache[key] = torch.zeros(
                sel.shape, dtype=dtype, device=sel.device
            ).masked_fill_(~sel, neg)
        return additive_cache[key]

    prev_override = transformer_options.get("optimized_attention_override")

    def override(func, q, k, v, heads, mask=None, **kw):
        seq = k.shape[2] if k.ndim == 4 else k.shape[1]
        qb = q.shape[0]
        sel = None
        if seq == L:
            sel = joint
        elif seq == txtlen and txtlen != model.txtlayers:
            sel = txtm  # txtfusion refiner blocks
        # A sub-call with a batch our per-row mask can't broadcast to (some
        # wrapper running its own reference pass) runs unmasked rather than
        # crashing — that matches stock behavior for that call.
        if sel is not None and sel.shape[0] not in (1, qb):
            sel = None
        if sel is not None and mask is None:
            mask = _additive(sel, q.dtype)
        if prev_override is not None:
            return prev_override(func, q, k, v, heads, mask=mask, **kw)
        return func(q, k, v, heads, mask=mask, **kw)

    if rt["token_masks"] is None:
        rt["token_masks"] = _build_token_masks(
            bundle, segments, soft, imglen, batch_rows, device
        )
    token_masks = rt["token_masks"]

    transformer_options[CONSUMED_KEY] = True
    transformer_options["optimized_attention_override"] = override
    CTX.token_masks = token_masks
    CTX.txtlen = txtlen
    CTX.imglen = imglen
    CTX.txtlayers = getattr(model, "txtlayers", -1)
    try:
        return executor(x, timesteps, context, attention_mask,
                        transformer_options=transformer_options, **kwargs)
    finally:
        CTX.clear()
        transformer_options.pop(CONSUMED_KEY, None)
        if prev_override is None:
            transformer_options.pop("optimized_attention_override", None)
        else:
            transformer_options["optimized_attention_override"] = prev_override


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


class Krea2LoRA:
    """Chainable list of LoRA files to be applied regionally."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0,
                                       "step": 0.01}),
            },
            "optional": {"prev_loras": ("KREA2_LORAS",)},
        }

    RETURN_TYPES = ("KREA2_LORAS",)
    FUNCTION = "build"
    CATEGORY = "conditioning/krea2_regional"

    def build(self, lora_name, strength, prev_loras=None):
        import folder_paths

        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        entry = {"sd": sd, "strength": strength, "label": lora_name}
        return (list(prev_loras or []) + [entry],)


class Krea2RegionalPrompt:
    """One region: a prompt (CONDITIONING), a MASK, and optional LoRAs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "mask": ("MASK",),
            },
            "optional": {
                "loras": ("KREA2_LORAS",),
                "prev_regions": ("KREA2_REGIONS",),
            },
        }

    RETURN_TYPES = ("KREA2_REGIONS",)
    FUNCTION = "build"
    CATEGORY = "conditioning/krea2_regional"

    def build(self, conditioning, mask, loras=None, prev_regions=None):
        region = {
            "cond": conditioning[0][0],
            "mask": mask,
            "loras": list(loras or []),
        }
        return (list(prev_regions or []) + [region],)


class Krea2ApplyRegional:
    """Patch a Krea 2 model for single-pass regional generation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "regions": ("KREA2_REGIONS",),
                "restrict_img_attn": ("BOOLEAN", {"default": False,
                    "tooltip": "Block image<->image attention across regions. "
                               "Strongest anti-bleed lever; can look collaged."}),
                "exclusive_masks": ("BOOLEAN", {"default": True,
                    "tooltip": "Where grown/feathered region masks overlap, "
                               "each token keeps only its strongest region "
                               "(no double LoRA in the gap)."}),
            },
            "optional": {"base_loras": ("KREA2_LORAS",)},
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    FUNCTION = "apply"
    CATEGORY = "conditioning/krea2_regional"

    def apply(self, model, conditioning, regions, restrict_img_attn,
              exclusive_masks=True, base_loras=None):
        if len(conditioning) != 1:
            logging.warning("[Krea2Regional] base conditioning has %d entries; "
                            "using the first.", len(conditioning))
        base_cond, base_extra = conditioning[0][0], dict(conditioning[0][1])

        m = model.clone()

        # ---- concatenate prompts, record segments ----
        chunks = [base_cond] + [r["cond"].to(base_cond) for r in regions]
        segments, off = [], 0
        for c in chunks:
            segments.append((off, off + c.shape[1]))
            off += c.shape[1]
        combined = torch.cat(chunks, dim=1)

        # ---- attach LoRAs ----
        lora_specs = []
        for entry in (base_loras or []):
            uid = f"k2r{next(_UID)}"
            n = _inject_lora(m, entry["sd"], uid, entry["strength"])
            logging.info("[Krea2Regional] base lora '%s': %d layers",
                         entry["label"], n)
            lora_specs.append({"uid": uid, "weight": 1.0, "seg": None,
                               "mask_idx": None})
        for ridx, region in enumerate(regions):
            for entry in region["loras"]:
                uid = f"k2r{next(_UID)}"
                n = _inject_lora(m, entry["sd"], uid, entry["strength"])
                logging.info("[Krea2Regional] region %d lora '%s': %d layers",
                             ridx, entry["label"], n)
                lora_specs.append({"uid": uid, "weight": 1.0, "seg": ridx + 1,
                                   "mask_idx": ridx})

        bundle = {
            "segments": segments,
            "txt_total": off,
            "masks": [r["mask"] for r in regions],
            "restrict_img_attn": restrict_img_attn,
            "exclusive_masks": exclusive_masks,
            "loras": lora_specs,
        }
        to = m.model_options.setdefault("transformer_options", {})
        to[BUNDLE_KEY] = bundle
        m.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, BUNDLE_KEY,
                               _diffusion_wrapper)

        return (m, [[combined, base_extra]])


NODE_CLASS_MAPPINGS = {
    "Krea2LoRA": Krea2LoRA,
    "Krea2RegionalPrompt": Krea2RegionalPrompt,
    "Krea2ApplyRegional": Krea2ApplyRegional,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2LoRA": "Krea2 Regional LoRA",
    "Krea2RegionalPrompt": "Krea2 Regional Prompt",
    "Krea2ApplyRegional": "Krea2 Apply Regional",
}
