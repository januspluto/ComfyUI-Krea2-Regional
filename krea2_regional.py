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
        # Total image-sequence tokens seen by token-axis Linears:
        # live target tokens + any native Krea2 reference tokens.
        self.imglen = 0
        self.reflen = 0
        self.txtlayers = -1

    def clear(self):
        self.token_masks = {}
        self.txtlen = 0
        self.imglen = 0
        self.reflen = 0
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
        m = re.match(r"^(.*?)\.(lokr_w1|lokr_w1_a|lokr_w1_b|lokr_w2"
                     r"|lokr_w2_a|lokr_w2_b|lokr_t2)$", k)
        if m:
            groups.setdefault(m.group(1), {})[m.group(2)] = v
            continue
        m = re.match(r"^(.*?)\.alpha$", k)
        if m:
            groups.setdefault(m.group(1), {})["alpha"] = float(v)
            continue
    out = {}
    for key, g in groups.items():
        orig_key = key
        entry = None
        if "down" in g and "up" in g:
            entry = {"type": "lora", "down": g["down"], "up": g["up"]}
            if "alpha" in g:
                entry["alpha"] = g["alpha"]
        elif ("lokr_w1" in g or "lokr_w1_a" in g) and \
                ("lokr_w2" in g or "lokr_w2_a" in g):
            if "lokr_t2" in g:
                logging.warning("[Krea2Regional] '%s': tucker LoKr is not "
                                "supported for these linear layers — skipped.",
                                key)
                continue
            dim = None
            if "lokr_w1" in g:
                w1 = g["lokr_w1"].float()
            else:
                dim = g["lokr_w1_b"].shape[0]
                w1 = (g["lokr_w1_a"].float() @ g["lokr_w1_b"].float())
            if "lokr_w2" in g:
                w2 = g["lokr_w2"].float()
            else:
                dim = g["lokr_w2_b"].shape[0]
                w2 = (g["lokr_w2_a"].float() @ g["lokr_w2_b"].float())
            if w1.ndim != 2 or w2.ndim != 2:
                logging.warning("[Krea2Regional] '%s': non-2D LoKr factors "
                                "(conv?) — skipped.", key)
                continue
            alpha_scale = (g["alpha"] / dim
                           if ("alpha" in g and dim) else 1.0)
            entry = {"type": "lokr", "w1": w1, "w2": w2,
                     "alpha_scale": alpha_scale}
        if entry is None:
            continue
        entry["orig"] = orig_key
        for p in _PREFIXES:
            if key.startswith(p):
                key = key[len(p):]
                break
        out[key] = entry
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


def _kron_apply(x, w1, w2):
    """x @ kron(w1, w2).T without materializing the Kronecker product.

    w1: (a1, b1), w2: (a2, b2); x last dim must be b1*b2; result a1*a2.
    Identity: for X = x reshaped (…, b1, b2), Y = w1 @ X @ w2.T reshaped back —
    the same math ComfyUI's own LoKr adapter computes on weights, applied to
    activations.
    """
    a1, b1 = w1.shape
    a2, b2 = w2.shape
    lead = x.shape[:-1]
    xr = x.reshape(*lead, b1, b2)
    t = torch.einsum("...ij,pj->...ip", xr, w2)   # (…, b1, a2)
    yr = torch.einsum("...ip,qi->...qp", t, w1)   # (…, a1, a2)
    return yr.reshape(*lead, a1 * a2)


class _RegionalLoRAMixin:
    """Mixin prepended onto the layer's own class; adds masked LoRA deltas
    (standard low-rank and LoKr / Kronecker adapters)."""

    def forward(self, x, *args, **kwargs):
        y = super().forward(x, *args, **kwargs)
        adapters = getattr(self, "regional_adapters", None)
        tm = CTX.token_masks
        if not adapters or not tm or not torch.is_tensor(x) \
                or x.ndim not in (3, 4):
            return y
        cast = self.__dict__.setdefault("_k2b_cast", {})
        for uid, adapter in adapters.items():
            m = tm.get(uid)
            if m is None:
                continue
            mm = _adapt_mask(m, x)
            if mm is None:
                continue
            kind, wa, wb, scale = adapter
            ck = (uid, x.device, x.dtype)
            if ck not in cast:
                cast[ck] = (wa.to(device=x.device, dtype=x.dtype),
                            wb.to(device=x.device, dtype=x.dtype))
            a, b = cast[ck]
            if kind == "lokr":
                if a.shape[1] * b.shape[1] != x.shape[-1]:
                    continue  # factorization doesn't fit this layer
                delta = _kron_apply(x, a, b)
            else:
                delta = F.linear(F.linear(x, a), b)
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


def _inject_lora(model_patcher, lora_sd: dict, uid: str, strength: float,
                 unmaskable: str = "skip") -> int:
    """Attach one LoRA (as adapter `uid`) onto the DiT block Linears.

    Keys that hit layers OUTSIDE the token sequence (tmlp/tproj timestep
    embedders, etc. — trained by newer ai-toolkit configs) can't be
    region-masked: the timestep vector modulates the whole image through
    AdaLN, so there is no per-token axis. `unmaskable`:
      "skip"           -> leave them out (default; logged as INFO)
      "apply globally" -> merge them as normal weight patches via ComfyUI's
                          own LoRA loader (affects the WHOLE image)
    """
    groups = _normalize_lora_sd(lora_sd)

    diffusion_model = model_patcher.get_model_object("diffusion_model")
    lookup: dict[str, str] = {}
    full_lookup: dict[str, str] = {}
    for name, mod in diffusion_model.named_modules():
        if (hasattr(mod, "weight") and callable(mod)
                and getattr(mod, "weight", None) is not None
                and mod.weight.ndim == 2):
            full_lookup[name] = name
            full_lookup[name.replace(".", "_")] = name
            if name.startswith(("blocks.", "txtfusion.", "txtmlp", "first",
                                "last")):
                lookup[name] = name
                lookup[name.replace(".", "_")] = name

    patched, skipped = 0, []
    global_keys: dict[str, str] = {}
    for key, g in groups.items():
        path = lookup.get(key) or lookup.get(key.replace(".", "_"))
        if path is None:
            gpath = (full_lookup.get(key)
                     or full_lookup.get(key.replace(".", "_")))
            if gpath is not None:
                # real layer, just not a token-sequence one
                global_keys[g["orig"]] = "diffusion_model." + gpath + ".weight"
            else:
                skipped.append(key)
            continue
        full = "diffusion_model." + path
        target = _patched_linear(model_patcher.get_model_object(full))
        if g["type"] == "lokr":
            target.regional_adapters[uid] = (
                "lokr", g["w1"], g["w2"],
                float(strength) * float(g["alpha_scale"]),
            )
        else:
            rank = g["down"].shape[0]
            alpha = g.get("alpha", rank)
            target.regional_adapters[uid] = (
                "lora", g["down"], g["up"],
                float(strength) * float(alpha) / float(rank),
            )
        model_patcher.add_object_patch(full, target)
        patched += 1

    if global_keys:
        if unmaskable == "apply globally":
            import comfy.lora
            patches = comfy.lora.load_lora(lora_sd, global_keys,
                                           log_missing=False)
            if patches:
                model_patcher.add_patches(patches, float(strength))
            logging.info(
                "[Krea2Regional] lora '%s': %d timestep/embedding layers "
                "applied GLOBALLY (whole image) — they can't be "
                "region-masked.", uid, len(global_keys))
        else:
            logging.info(
                "[Krea2Regional] lora '%s': %d timestep/embedding layers "
                "skipped (e.g. %s) — they modulate the whole image and "
                "can't be region-masked. Set unmaskable_layers to 'apply "
                "globally' on Apply Regional to merge them.",
                uid, len(global_keys), list(global_keys)[:3])
    if skipped:
        logging.warning(
            "[Krea2Regional] lora '%s': %d keys didn't match any layer "
            "(e.g. %s) — is this a Krea 2 LoRA?", uid, len(skipped),
            skipped[:3]
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
    between them.

    Depth priority: where strengths are (near-)equal — the common case of
    two hard masks overlapping at 1.0 — the EARLIER region in the list wins
    (top of the Builder's region list = front). Ties are resolved by EQUALITY (first region at the exact
    maximum wins); any genuine strength difference, however small, wins
    outright — no epsilon bias distorting real comparisons."""
    if soft.shape[0] < 2:
        return soft
    mx = soft.max(dim=0, keepdim=True).values
    winner = torch.zeros_like(soft, dtype=torch.bool)
    winner.scatter_(0, (soft == mx).float().argmax(dim=0, keepdim=True),
                    True)
    return soft * winner


def _grow_tokens(soft, h, w, grow=2):
    """Dilate token-grid soft masks by `grow` cells (max-pool)."""
    if grow <= 0 or soft.shape[0] == 0:
        return soft
    k = 2 * int(grow) + 1
    m = soft.view(-1, 1, h, w)
    m = torch.nn.functional.max_pool2d(m, k, stride=1, padding=k // 2)
    return m.view(soft.shape[0], -1)


def _finalize_adaptive(aff, base_soft, mode, threshold, h, w, grow=2):
    """Turn accumulated text->image affinity into refined region masks.

    aff:       (R, imglen) raw accumulated affinity per region
    base_soft: (R, imglen) the user's box masks on the token grid
    mode:      "refine" (discovery constrained to grown boxes) or
               "free"   (pure competition, boxes only as empty-fallback)
    Returns (R, imglen) soft masks in [0, 1].

    This is the FreeFuse idea (arXiv:2510.23515) adapted to Krea2's
    single-stream layout: early-step attention between a subject's prompt
    tokens and the image tokens reveals where the model actually placed the
    subject; snapping the LoRA/attention masks to that silhouette beats a
    static rectangle.
    """
    R = aff.shape[0]
    if R == 0:
        return base_soft
    m = aff.float().view(R, 1, h, w)
    for _ in range(2):  # despeckle
        m = torch.nn.functional.avg_pool2d(m, 3, stride=1, padding=1)
    m = m.view(R, -1)
    mn = m.min(dim=1, keepdim=True).values
    mx = m.max(dim=1, keepdim=True).values
    m = (m - mn) / (mx - mn + 1e-8)

    if mode == "refine":
        prior = _grow_tokens(base_soft, h, w, grow=grow)
        m = m * (prior > 0.05).float()
        # renormalize within the prior so threshold means the same thing
        mx2 = m.max(dim=1, keepdim=True).values
        m = m / (mx2 + 1e-8)

    soft = torch.where(m >= threshold, m, torch.zeros_like(m))

    # a region that discovered nothing falls back to its drawn box
    empty = (soft > 0).sum(dim=1) == 0
    if empty.any():
        soft = soft.clone()
        soft[empty] = base_soft[empty]
    return soft


def _build_allow(segments, region_img_hard, imglen, restrict_img_attn, device,
                 reflen=0):
    """Build the joint attention allow-mask.

    Native ComfyUI Krea2 lays the joint sequence out as::

        [text | live target image | native reference image token(s)]

    ``imglen`` is the LIVE target span only. ``reflen`` is the optional
    appended native-reference span. References are deliberately treated as
    global context in this first compatibility implementation: every prompt
    and every target token may communicate with them. Regional isolation still
    applies to target<->prompt and (when enabled) target<->target routing.
    This preserves stock native-reference semantics while avoiding any false
    claim that a reference has been assigned to one Builder region.
    """
    txtlen = segments[-1][1]
    L = txtlen + imglen + reflen
    i0, i1 = txtlen, txtlen + imglen          # live target span
    r0, r1 = i1, i1 + reflen                  # appended native refs
    allow = torch.zeros(L, L, dtype=torch.bool, device=device)

    for s, e in segments:  # prompts are mutually isolated
        allow[s:e, s:e] = True

    # Base prompt is global scene context: target + native refs.
    b0, b1 = segments[0]
    allow[b0:b1, i0:r1] = True
    allow[i0:r1, b0:b1] = True

    for r, (s, e) in enumerate(segments[1:]):
        # Existing regional prompt <-> target routing remains unchanged.
        allow[s:e, i0:i1] |= region_img_hard[r][None, :]
        allow[i0:i1, s:e] |= region_img_hard[r][:, None]
        # Native references are global carriers, matching stock Krea2's
        # unrestricted reference behavior. Region-specific ref assignment is
        # intentionally a separate future feature.
        if reflen:
            allow[s:e, r0:r1] = True
            allow[r0:r1, s:e] = True

    if restrict_img_attn and region_img_hard.shape[0] > 0:
        shared = (region_img_hard.T.float() @ region_img_hard.float()) > 0
        bg = ~region_img_hard.any(dim=0)
        img_allow = shared | bg[None, :] | bg[:, None]
        img_allow |= torch.eye(imglen, dtype=torch.bool, device=device)
        allow[i0:i1, i0:i1] = img_allow
        if reflen:
            # References remain globally readable even while target regions
            # are isolated from one another. This is the conservative
            # compatibility policy; refs can therefore be a semantic bridge.
            allow[i0:i1, r0:r1] = True
            allow[r0:r1, i0:i1] = True
            allow[r0:r1, r0:r1] = True
    else:
        allow[i0:r1, i0:r1] = True

    txt_allow = torch.zeros(txtlen, txtlen, dtype=torch.bool, device=device)
    for s, e in segments:
        txt_allow[s:e, s:e] = True
    return allow, txt_allow


def _build_token_masks(bundle, segments, soft_masks, imglen, batch_rows, device,
                       reflen=0):
    """Per-adapter token masks for ``[text | target | refs]``.

    Regional LoRAs NEVER act on native-reference tokens: the clean reference
    representation should not be rewritten by a character/style adapter meant
    for a target region. Base/global LoRAs retain their existing global
    semantics and therefore also cover references.
    """
    txtlen = segments[-1][1]
    L = txtlen + imglen + reflen
    target0, target1 = txtlen, txtlen + imglen
    ref0, ref1 = target1, target1 + reflen
    row = {}

    def bump(uid, vec):
        row[uid] = torch.maximum(row[uid], vec) if uid in row else vec

    for spec in bundle["loras"]:
        uid, weight, seg_idx, mask_idx = (
            spec["uid"], spec["weight"], spec["seg"], spec["mask_idx"],
        )
        vec = torch.zeros(L, device=device)
        if mask_idx is None:  # global/base LoRA
            vec[:txtlen] = weight
            if (bundle.get("base_loras_exclude_regions")
                    and soft_masks.shape[0] > 0):
                union = soft_masks.max(dim=0).values
                vec[target0:target1] = weight * (1.0 - union)
            else:
                vec[target0:target1] = weight
            if reflen:
                vec[ref0:ref1] = weight
        else:
            # Target-only spatial gate. Reference tokens stay exactly zero.
            vec[target0:target1] = soft_masks[mask_idx] * weight
            if seg_idx is not None:
                s, e = segments[seg_idx]
                vec[s:e] = weight
        bump(uid, vec)

    out = {}
    if all(batch_rows):
        for uid, vec in row.items():
            out[uid] = vec[None, :, None]
    else:
        rows = torch.tensor(batch_rows, device=device, dtype=torch.float32)
        for uid, vec in row.items():
            out[uid] = (rows[:, None] * vec[None, :])[:, :, None]
    return out


# ---------------------------------------------------------------------------
# The diffusion-model wrapper (WrappersMP.DIFFUSION_MODEL)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------#
# Masked-attention strategy ladder (ported from sd-forge-krea2-regional).
#
# The one thing we NEVER do is let sdpa's silent math-backend fallback run:
# it materializes B*H*L*L fp32 attention (~15 GB/block at 1744x1248) and
# turns steps into half-minutes.  We probe each strategy once per sequence
# length and cache the winner:
#   1. fused sdpa + fp16 additive bias (what the efficient kernel wants)
#   2. fused sdpa + bool mask (half the memory of an additive float mask
#      and keeps sdpa's fused kernels eligible at large sequence lengths)
#   3. hand-tiled attention: exact, bounded memory, eligibility-proof
# ---------------------------------------------------------------------------#

_MASK_MODE: dict = {}       # (L, device, dtype) -> "add" | "bool" | "tiled"
_ADD_CACHE: dict = {}       # (id(mask), dtype) -> additive fp bias


def _additive_mask(mask: torch.Tensor, dtype) -> torch.Tensor:
    """Convert a bool mask to an additive fp bias (finfo.min at masked spots)."""
    key = (id(mask), dtype)
    got = _ADD_CACHE.get(key)
    if got is None or got.device != mask.device:
        _ADD_CACHE.clear()
        neg = torch.finfo(dtype).min if dtype.is_floating_point else -1e9
        got = torch.zeros(
            mask.shape, dtype=dtype, device=mask.device
        ).masked_fill_(~mask, neg)
        _ADD_CACHE[key] = got
    return got


def _tiled_attention(q, k, v, mask, tile=1024):
    """Exact masked attention in query tiles: bounded transient memory
    (~B*H*tile*L scores), low-precision matmuls, fp32 softmax via the
    fused dtype arg (no separate fp32 materialization). Immune to sdpa
    kernel-eligibility heuristics."""
    B, H, L, D = q.shape
    scale = D ** -0.5
    kt = k.transpose(-1, -2).contiguous()
    out = torch.empty_like(q)
    neg = torch.finfo(q.dtype).min
    for s in range(0, L, tile):
        e = min(s + tile, L)
        scores = torch.matmul(q[:, :, s:e], kt)
        scores.mul_(scale)
        mt = mask[:, :, s:e] if mask.shape[2] == L else mask
        scores.masked_fill_(~mt, neg)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        out[:, :, s:e] = torch.matmul(probs.to(v.dtype), v)
    return out


def _fused_sdpa(q, k, v, heads, mask):
    """Run fused sdpa with explicit kernel selection (exclude cudnn: several
    torch builds accept attn_mask on the cudnn backend and silently IGNORE it
    — regional isolation vanishes while everything looks fast and healthy)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    try:
        from comfy.sd import attention_function as _apt
    except Exception:
        from comfy.sd import attention_pytorch as _apt
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                      SDPBackend.FLASH_ATTENTION]):
        return _apt(q, k, v, heads, mask=mask, skip_reshape=True)


def _mask_efficacy_ok(fn_name, dtype, device):
    """Prove the candidate path actually APPLIES masks: a mask that blocks
    everything except self-attention must change the output vs unmasked,
    and must match the exact tiled reference. Run once per mode probe."""
    try:
        torch.manual_seed(0)
        q = torch.randn(1, 2, 16, 8, dtype=dtype, device=device)
        k = torch.randn(1, 2, 16, 8, dtype=dtype, device=device)
        v = torch.randn(1, 2, 16, 8, dtype=dtype, device=device)
        eye = torch.eye(16, dtype=torch.bool, device=device)[None, None]
        m = eye if fn_name == "bool" else _additive_mask(eye, dtype)
        out_m = _fused_sdpa(q, k, v, 2, m)
        out_u = _fused_sdpa(q, k, v, 2, None)
        if (out_m - out_u).abs().max().item() < 1e-3:
            logging.warning(
                "[Krea2Regional] %s-mask path IGNORES masks on this "
                "backend — rejected", fn_name)
            return False
        ref = _tiled_attention(q, k, v, eye)
        b, h, l, d = ref.shape
        ref = ref.transpose(1, 2).reshape(b, l, h * d)
        if (out_m - ref).abs().max().item() > (
                0.05 if dtype != torch.float32 else 1e-3):
            logging.warning(
                "[Krea2Regional] %s-mask path DIVERGES from reference — "
                "rejected", fn_name)
            return False
        return True
    except Exception:
        return False


def _select_masked_sdpa(q, k, v, heads, bool_mask):
    """Probe mask strategies once per (L, device, dtype), then use the cached
    winner for every subsequent call at that sequence length."""
    L = q.shape[2]
    key = (L, str(q.device), str(q.dtype))
    mode = _MASK_MODE.get(key)
    if mode == "add":
        return _fused_sdpa(q, k, v, heads, _additive_mask(bool_mask, q.dtype))
    if mode == "bool":
        return _fused_sdpa(q, k, v, heads, bool_mask)
    if mode == "tiled":
        out = _tiled_attention(q, k, v, bool_mask)
        b, h, l, d = out.shape
        return out.transpose(1, 2).reshape(b, l, h * d)
    # probe once for this sequence length
    for name, fn in (
        ("add", lambda: _fused_sdpa(q, k, v, heads,
                                    _additive_mask(bool_mask, q.dtype))),
        ("bool", lambda: _fused_sdpa(q, k, v, heads, bool_mask)),
    ):
        try:
            if not _mask_efficacy_ok(name, q.dtype, q.device):
                continue
            out = fn()
            _MASK_MODE[key] = name
            logging.info(
                "[Krea2Regional] masked attention @L=%d: fused sdpa "
                "(%s mask, efficacy verified)", L, name)
            return out
        except (RuntimeError, ImportError) as e:
            logging.info(
                "[Krea2Regional] fused sdpa (%s mask) @L=%d rejected: "
                "%s", name, L, str(e)[:300])
            continue
    _MASK_MODE[key] = "tiled"
    logging.info(
        "[Krea2Regional] masked attention @L=%d: fused kernels "
        "unavailable -> tiled attention (exact, bounded memory)", L)
    out = _tiled_attention(q, k, v, bool_mask)
    b, h, l, d = out.shape
    return out.transpose(1, 2).reshape(b, l, h * d)


def _native_ref_latents(rest, kwargs):
    """Find Krea2's native ``ref_latents`` across ComfyUI signature drift."""
    ref = kwargs.get("ref_latents")
    if ref is not None:
        return ref
    # Current core calls wrappers positionally as
    # (..., attention_mask, ref_latents, transformer_options). The attention
    # mask is a Tensor, while ref_latents is a list/tuple of Tensors.
    for a in rest:
        if (isinstance(a, (list, tuple)) and len(a) > 0
                and all(torch.is_tensor(v) for v in a)):
            return a
    return None


def _native_ref_token_count(ref_latents, patch):
    """Match SingleStreamDiT.process_img token counts without encoding refs."""
    if not ref_latents:
        return 0
    total = 0
    patch = max(int(patch), 1)
    for ref in ref_latents:
        if not torch.is_tensor(ref) or ref.ndim < 4:
            continue
        # For 5D Wan-style still latents, core flattens B*T but the spatial
        # token count remains ceil(H/patch)*ceil(W/patch).
        rh, rw = int(ref.shape[-2]), int(ref.shape[-1])
        total += (-(-rh // patch)) * (-(-rw // patch))
    return total


def _diffusion_wrapper(executor, x, timesteps, context, *args, **kwargs):
    # ComfyUI inserts new positional parameters into Krea2's forward over
    # time (ref_latents landed between attention_mask and
    # transformer_options and broke fixed signatures). Stay agnostic: the
    # options dict is either the `transformer_options` kwarg or the last
    # positional dict; everything else passes through untouched, in order.
    transformer_options = kwargs.pop("transformer_options", None)
    rest = list(args)
    if transformer_options is None and rest and isinstance(rest[-1], dict):
        transformer_options = rest.pop()
    if transformer_options is None:
        transformer_options = {}
    bundle = transformer_options.get(BUNDLE_KEY)
    model = executor.class_obj  # the SingleStreamDiT instance

    # Native Krea2 reference latents are appended AFTER the live target image
    # tokens by current ComfyUI core: [text | target | refs]. Only count them
    # when a reference method is actually active; a plain base Krea2 model may
    # receive the conditioning value but intentionally ignore it.
    ref_latents = _native_ref_latents(rest, kwargs)
    ref_method = kwargs.get("ref_latents_method",
                            getattr(model, "default_ref_method", None))
    reflen = (
        _native_ref_token_count(ref_latents, getattr(model, "patch", 1))
        if ref_method is not None else 0
    )

    if (
        bundle is None
        or context is None
        or transformer_options.get(CONSUMED_KEY)
        or context.shape[1] != bundle["txt_total"]
    ):
        # negative prompt / unrelated conditioning / chained duplicate -> clean pass
        return executor(x, timesteps, context, *rest,
                        transformer_options=transformer_options, **kwargs)

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
    L = txtlen + imglen + reflen

    # ---- run-level cache: the masks depend only on geometry + batch layout,
    # so build them ONCE per sampling run instead of every step. This kills
    # the per-step GPU alloc churn that fragments cudaMallocAsync and caused
    # sporadic OOM on tight-VRAM setups.
    ck = (h_, w_, bs, tuple(batch_rows), str(device), txtlen, reflen,
          str(ref_method))
    rt = bundle.get("_rt")
    if rt is None or rt.get("key") != ck:
        rt = {"key": ck, "token_masks": None,
              "masks_built": False, "call": 0, "blk": 0,
              "aff": None, "refined": None, "last_t": None}
        bundle["_rt"] = rt
        if reflen:
            logging.info(
                "[Krea2Regional] native refs active: %d ref token(s); "
                "joint sequence=%d text + %d target + %d ref",
                reflen, txtlen, imglen, reflen)

    ad = bundle.get("adaptive") or {}
    ad_mode = ad.get("mode", "off")
    ad_steps = max(int(ad.get("steps", 2)), 1)

    # ---- per-sampling-run bookkeeping: a timestep that JUMPS UP means a new
    # run started (schedules descend within a run) -> re-capture adaptively
    t0 = float(timesteps.reshape(-1)[0])
    if rt["last_t"] is not None and t0 > rt["last_t"] + 1e-6:
        rt.update(call=0, aff=None, refined=None)
        if ad_mode != "off":
            rt.update(masks_built=False, token_masks=None)
        _ADD_CACHE.clear()
    rt["last_t"] = t0
    rt["call"] += 1
    rt["blk"] = 0  # per-call joint-attention block counter (capture below)

    # ---- scheduled restrict: hard image<->image isolation while identity
    # forms, then release so the late steps can integrate seams/lighting.
    # restrict_end_sigma < 0 means "never release" (classic behavior).
    restrict_now = bool(bundle["restrict_img_attn"]) and (
        t0 >= bundle.get("restrict_end_sigma", -1.0) - 1e-9)
    if rt.get("restrict_now") != restrict_now:
        # only the attention masks change: keep token masks + adaptive state
        rt["restrict_now"] = restrict_now
        rt["masks_built"] = False
        _ADD_CACHE.clear()

    # ---- adaptive finalize: capture window over -> snap masks to the
    # discovered subject silhouettes and rebuild everything once
    capturing = (ad_mode != "off" and rt["refined"] is None
                 and rt["call"] <= ad_steps)
    if (ad_mode != "off" and rt["refined"] is None
            and rt["aff"] is not None and rt["call"] > ad_steps):
        base_soft = (
            torch.stack([_latent_mask(m, h_, w_, device)
                         for m in bundle["masks"]])
            if bundle["masks"] else torch.zeros(0, imglen, device=device)
        )
        rt["refined"] = _finalize_adaptive(
            rt["aff"], base_soft, ad_mode,
            float(ad.get("threshold", 0.45)), h_, w_)
        rt.update(masks_built=False, token_masks=None)
        _ADD_CACHE.clear()

    if not rt["masks_built"]:
        if rt["refined"] is not None:
            soft = rt["refined"]
        else:
            soft = (
                torch.stack([_latent_mask(m, h_, w_, device)
                             for m in bundle["masks"]])
                if bundle["masks"] else torch.zeros(0, imglen, device=device)
            )
        if bundle.get("exclusive_masks", True):
            soft = _exclusive_soft(soft)
        hard = soft > 0.5

        # Pre-allocate the allow mask buffer and fill it in-place.
        # The mask shape is deterministic from geometry (L × L), so we
        # allocate once per run and update the content on restrict toggle
        # or adaptive finalize — no new tensor allocation.
        allow, txt_allow = _build_allow(
            segments, hard, imglen, rt.get("restrict_now", False), device,
            reflen=reflen
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
        rt.update(soft=soft, joint=joint, txtm=txtm, masks_built=True,
                   hard=hard)
    soft = rt["soft"]
    joint = rt["joint"]
    txtm = rt["txtm"]

    # which mid-network blocks to sample affinity from (evenly spaced 4 in
    # the middle half — semantics live there, extremes are noisier)
    if capturing:
        nblocks = len(model.blocks)
        # Sample 2 mid-network blocks instead of 4 — semantics live there
        # and the affinity signal is already strong with half the blocks.
        # This halves the capture matmul cost (2 blocks × R regions × einsum).
        step_ = max(nblocks // 4, 1)
        capture_set = {nblocks // 3, nblocks // 3 + step_}
        rows_t = torch.tensor(batch_rows, device=device)

    prev_override = transformer_options.get("optimized_attention_override")

    def override(func, q, k, v, heads, mask=None, **kw):
        seq = k.shape[2] if k.ndim == 4 else k.shape[1]
        qb = q.shape[0]
        sel = None
        if seq == L:
            sel = joint
            # ---- adaptive capture: accumulate image-query x region-prompt-key
            # affinity at a few mid-network blocks during the early steps.
            # The mask can't hide this — we do our own q.k, so tokens OUTSIDE
            # a region's box still register affinity to its prompt.
            if capturing and q.ndim == 4 and qb == bs:
                blk = rt["blk"]
                rt["blk"] = blk + 1
                if blk in capture_set and len(segments) > 1:
                    with torch.no_grad():
                        if rt["aff"] is None:
                            rt["aff"] = torch.zeros(
                                len(segments) - 1, imglen,
                                dtype=torch.float32, device=q.device)
                        q_img = q[:, :, txtlen:txtlen + imglen]
                        scale_ = q.shape[-1] ** -0.5
                        # Chunked affinity capture: process regions in small
                        # groups to bound peak memory when there are many
                        # regions with long prompts. Each chunk is an
                        # independent accumulate, so order doesn't matter.
                        region_spans = list(segments[1:])
                        chunk_size = 4  # regions per chunk
                        for chunk_start in range(0, len(region_spans), chunk_size):
                            chunk = region_spans[chunk_start:chunk_start + chunk_size]
                            # stack keys for this chunk: (B, H, sum_key, D)
                            k_chunks = [k[:, :, s:e] for s, e in chunk]
                            k_cat = torch.cat(k_chunks, dim=2)
                            att = torch.einsum(
                                "bhid,bhjd->bhij", q_img, k_cat) * scale_
                            # split result back per region for the reduction
                            offset = 0
                            for ci, (s, e) in enumerate(chunk):
                                ridx = chunk_start + ci
                                kl = e - s
                                att_r = att[:, :, :, offset:offset + kl]
                                offset += kl
                                a = att_r.amax(dim=3).float().mean(dim=1)
                                a = a[rows_t[:a.shape[0]]].sum(dim=0)
                                rt["aff"][ridx] += a
        elif seq == txtlen and txtlen != model.txtlayers:
            sel = txtm  # txtfusion refiner blocks
        # A sub-call with a batch our per-row mask can't broadcast to (some
        # wrapper running its own reference pass) runs unmasked rather than
        # crashing — that matches stock behavior for that call.
        if sel is not None and sel.shape[0] not in (1, qb):
            sel = None
        if sel is not None and mask is None:
            # Convert the bool allow-mask to an additive bias for the host
            # attention function. The host's own attention backend (flash,
            # efficient, sage, etc.) handles the mask — we just need to pass
            # it in the right format. Bool masks are kept for the module-level
            # additive cache (half the memory of pre-converting to float).
            #
            # The mask-efficacy ladder (_mask_efficacy_ok) is available as a
            # diagnostic: if the host backend silently ignores masks, the
            # _select_masked_sdpa fallback can be used — but only as a last
            # resort, because it bypasses the host's head-reshape and gate
            # logic. The normal path lets func() handle everything.
            mask = _additive_mask(sel, q.dtype)
        if prev_override is not None:
            return prev_override(func, q, k, v, heads, mask=mask, **kw)
        return func(q, k, v, heads, mask=mask, **kw)

    if rt["token_masks"] is None:
        rt["token_masks"] = _build_token_masks(
            bundle, segments, soft, imglen, batch_rows, device, reflen=reflen
        )
    token_masks = rt["token_masks"]

    transformer_options[CONSUMED_KEY] = True
    transformer_options["optimized_attention_override"] = override
    CTX.token_masks = token_masks
    CTX.txtlen = txtlen
    CTX.imglen = imglen + reflen
    CTX.reflen = reflen
    CTX.txtlayers = getattr(model, "txtlayers", -1)
    try:
        return executor(x, timesteps, context, *rest,
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


# ---------------------------------------------------------------------------
# Region Lock: latent-only structure/identity anchoring (no reference image)
# ---------------------------------------------------------------------------


def _make_region_lock(bundle, strength, start_percent, end_percent):
    """Post-CFG sampler hook that freezes each region's content once formed.

    At `start_percent` through the noise schedule the model has committed to
    each subject's identity/structure; we snapshot its predicted-clean latent
    (x0) and, until `end_percent`, pull the region back toward that snapshot:

        denoised += strength * mask * (snapshot - denoised)

    The mold is the model's OWN early prediction — no reference image, no VAE
    round-trip, no latent-space conversion (the snapshot already lives in
    sampler space). Late-step drift/mutation inside a region is suppressed;
    the background keeps refining freely. Masks follow adaptive refinement
    automatically when it's active. Sampler-level and weight-free, so it
    composes with the LoRA masking and stays fp8-safe.
    """
    state = {"snap": None, "last_sigma": None, "sig_start": None,
             "sig_end": None, "mask": None, "mask_key": None}

    def _lock_mask(denoised):
        h8, w8 = int(denoised.shape[-2]), int(denoised.shape[-1])
        rt = bundle.get("_rt") or {}
        soft = rt.get("soft", None)
        key = (h8, w8, id(soft) if soft is not None else None)
        if state["mask_key"] == key and state["mask"] is not None:
            return state["mask"]
        union = None
        if soft is not None and soft.numel() and len(rt.get("key", ())) >= 2:
            th, tw = rt["key"][0], rt["key"][1]
            u = soft.max(dim=0).values.view(1, 1, th, tw).float()
            union = torch.nn.functional.interpolate(
                u, size=(h8, w8), mode="bilinear", align_corners=False)
        elif bundle.get("masks"):
            ms = []
            for m in bundle["masks"]:
                mm = m
                if mm.ndim == 2:
                    mm = mm[None]
                ms.append(torch.nn.functional.interpolate(
                    mm[:1].float()[None], size=(h8, w8), mode="area")[0, 0])
            union = torch.stack(ms).max(dim=0).values[None, None]
        if union is None:
            union = torch.zeros(1, 1, h8, w8)
        # soft 3x3 edge so the freeze doesn't leave a hard seam
        union = torch.nn.functional.avg_pool2d(
            union, 3, stride=1, padding=1, count_include_pad=False)
        state["mask"] = union.clamp(0, 1)
        state["mask_key"] = key
        return state["mask"]

    def hook(args):
        denoised = args["denoised"]
        sigma = args["sigma"]
        sig = float(sigma.reshape(-1).max())
        if state["sig_start"] is None:
            try:
                ms = args["model"].model_sampling
                state["sig_start"] = float(ms.percent_to_sigma(start_percent))
                state["sig_end"] = float(ms.percent_to_sigma(end_percent))
            except Exception:
                return denoised  # exotic wrapper: fail open
        if state["last_sigma"] is not None and sig > state["last_sigma"] + 1e-6:
            state["snap"] = None  # new sampling run
        state["last_sigma"] = sig

        if sig > state["sig_start"]:
            return denoised  # too early: identity still forming
        if state["snap"] is None:
            state["snap"] = denoised.detach().clone()
            return denoised
        if sig < state["sig_end"]:
            return denoised  # released: let the model integrate seams
        mask = _lock_mask(denoised).to(device=denoised.device,
                                       dtype=denoised.dtype)
        while mask.ndim < denoised.ndim:
            mask = mask.unsqueeze(0)
        snap = state["snap"]
        if snap.shape != denoised.shape:  # batch changed mid-run: recapture
            state["snap"] = denoised.detach().clone()
            return denoised
        return denoised + float(strength) * mask * (snap - denoised)

    return hook


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


def _clone_for_regional(model, dynamic_vram_mode="regional safe"):
    """Clone the input patcher, avoiding AIMDO DynamicVRAM for Regional.

    As of ComfyUI 2026, ModelPatcherDynamic's pinned host buffers can grow
    monotonically across repeated restaging of a patched Krea2 model until
    ``hostbuf_grow`` reaches the per-model reservation cap.  Regional models
    are especially likely to be restaged because object patches / LoRAs make
    the patch signature differ from the base checkpoint.

    Current ComfyUI provides ``get_non_dynamic_delegate()`` specifically to
    obtain a normal ModelPatcher for a dynamic model.  Prefer that path for
    Regional so repeated generations use estimate-based model management
    instead of AIMDO's host-buffer allocator.  The original input MODEL is
    untouched.

    ``keep DynamicVRAM`` preserves the old behavior for users who explicitly
    prefer AIMDO or are on a future ComfyUI where the upstream bug is fixed.
    Compatibility fallbacks keep older ComfyUI versions working.
    """
    mode = str(dynamic_vram_mode or "regional safe").strip().lower()
    keep_dynamic = mode.startswith("keep")

    is_dynamic = False
    try:
        probe = getattr(model, "is_dynamic", None)
        is_dynamic = bool(probe()) if callable(probe) else False
    except Exception:
        is_dynamic = False

    if not is_dynamic or keep_dynamic:
        return model.clone()

    # Preferred current-ComfyUI API.  This also lets the Dynamic patcher
    # maintain/reuse its non-dynamic clone-model override correctly.
    delegate = getattr(model, "get_non_dynamic_delegate", None)
    if callable(delegate):
        try:
            out = delegate()
            logging.info(
                "[Krea2Regional] DynamicVRAM detected: using the non-dynamic "
                "Regional delegate to avoid AIMDO hostbuf_grow accumulation."
            )
            return out
        except Exception as exc:
            logging.warning(
                "[Krea2Regional] get_non_dynamic_delegate() failed (%s); "
                "trying clone(disable_dynamic=True).", exc
            )

    # Supported by current ModelPatcherDynamic.clone(); TypeError is expected
    # on older ComfyUI builds whose clone() predates disable_dynamic.
    try:
        out = model.clone(disable_dynamic=True)
        logging.info(
            "[Krea2Regional] DynamicVRAM detected: cloned Regional model with "
            "disable_dynamic=True to avoid AIMDO host-buffer accumulation."
        )
        return out
    except TypeError:
        logging.warning(
            "[Krea2Regional] This ComfyUI build cannot disable DynamicVRAM "
            "per model. Falling back to the old DynamicVRAM path. If you see "
            "aimdo hostbuf_grow errors, update ComfyUI or launch it with "
            "--disable-dynamic-vram."
        )
        return model.clone()


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
                "exclusive_masks": ("BOOLEAN", {"default": False,
                    "tooltip": "Where grown/feathered region masks overlap, "
                               "each token keeps only its strongest region "
                               "(no double LoRA in the gap)."}),
                "adaptive_masks": (["off", "refine boxes",
                                    "free (ignore boxes)"],
                    {"default": "refine boxes",
                     "tooltip": "FreeFuse-style adaptive routing: watch where "
                                "the model actually places each subject during "
                                "the first steps (prompt-token/image-token "
                                "attention), then snap the region masks to "
                                "those silhouettes. 'refine boxes' keeps "
                                "discovery inside your (slightly grown) "
                                "boxes; 'free' lets regions land anywhere."}),
                "adaptive_steps": ("INT", {"default": 2, "min": 1, "max": 8,
                    "tooltip": "How many model calls to observe before "
                               "snapping the masks."}),
                "adaptive_threshold": ("FLOAT", {"default": 0.85, "min": 0.05,
                    "max": 0.95, "step": 0.05,
                    "tooltip": "Affinity cutoff (fraction of each region's "
                               "peak). Lower = larger discovered regions."}),
                "base_loras_exclude_regions": ("BOOLEAN", {"default": False,
                    "tooltip": "Style/base LoRAs apply everywhere EXCEPT "
                               "inside the subject regions — keeps a style "
                               "LoRA from contaminating character LoRAs."}),
                "region_lock_strength": ("FLOAT", {"default": 0.0, "min": 0.0,
                    "max": 1.0, "step": 0.05,
                    "tooltip": "Latent-only Region Lock: snapshot each "
                               "region's predicted-clean latent once identity "
                               "has formed and pull it back toward that "
                               "snapshot each step. Suppresses late-step "
                               "drift/mutation inside regions. 0 = off; "
                               "0.2-0.4 anchors, 0.7+ freezes hard."}),
                "region_lock_start": ("FLOAT", {"default": 0.25, "min": 0.0,
                    "max": 0.95, "step": 0.05,
                    "tooltip": "When (as a fraction of the schedule) to take "
                               "the snapshot."}),
                "region_lock_end": ("FLOAT", {"default": 0.70, "min": 0.05,
                    "max": 1.0, "step": 0.05,
                    "tooltip": "When to release the lock so the model can "
                               "integrate seams and lighting."}),
                "restrict_end_percent": ("FLOAT", {"default": 1.0,
                    "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": "When restrict_img_attn is on: release the "
                               "image<->image isolation at this fraction of "
                               "the schedule. Identity locks in during the "
                               "restricted window; the open steps afterwards "
                               "integrate seams and lighting. 1.0 = "
                               "restricted for the whole run (classic). "
                               "0.4-0.6 is a good cohesion sweet spot."}),
            },
            "optional": {
                "base_loras": ("KREA2_LORAS",),
                "dynamic_vram_mode": (["regional safe", "keep DynamicVRAM"],
                    {"default": "regional safe",
                     "tooltip": "Recommended: Regional safe clones a DynamicVRAM/AIMDO "
                                "Krea2 model into ComfyUI's normal model-management path. "
                                "This avoids the current repeated-run AIMDO hostbuf_grow "
                                "bug. Keep DynamicVRAM restores the old behavior."}),
                "unmaskable_layers": (["skip", "apply globally"],
                    {"default": "apply globally",
                     "tooltip": "Some LoRA layers (timestep/modulation "
                                "embedders from newer trainers) have no "
                                "spatial token axis and can't be region-"
                                "masked. 'skip' drops them from REGION "
                                "LoRAs (safest for isolation); 'apply "
                                "globally' keeps their full effect but it "
                                "touches the whole image."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    FUNCTION = "apply"
    CATEGORY = "conditioning/krea2_regional"

    def apply(self, model, conditioning, regions, restrict_img_attn,
              exclusive_masks=False, adaptive_masks="refine boxes", adaptive_steps=2,
              adaptive_threshold=0.85, base_loras_exclude_regions=False,
              region_lock_strength=0.35, region_lock_start=0.25,
              region_lock_end=0.70, restrict_end_percent=1.0,
              unmaskable_layers="apply globally", base_loras=None,
              dynamic_vram_mode="regional safe"):
        if len(conditioning) != 1:
            logging.warning("[Krea2Regional] base conditioning has %d entries; "
                            "using the first.", len(conditioning))
        base_cond, base_extra = conditioning[0][0], dict(conditioning[0][1])

        m = _clone_for_regional(model, dynamic_vram_mode)

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
                n = _inject_lora(m, entry["sd"], uid, entry["strength"],
                                 unmaskable=unmaskable_layers)
                logging.info("[Krea2Regional] region %d lora '%s': %d layers",
                             ridx, entry["label"], n)
                lora_specs.append({"uid": uid, "weight": 1.0, "seg": ridx + 1,
                                   "mask_idx": ridx})

        # scheduled restrict: convert the percent to sigma space so it
        # tracks the real noise schedule at any step count. >=0.999 means
        # restricted for the whole run (sentinel -1: never release).
        restrict_end_sigma = -1.0
        if restrict_img_attn and restrict_end_percent < 0.999:
            try:
                ms = model.model.model_sampling
                restrict_end_sigma = float(
                    ms.percent_to_sigma(restrict_end_percent))
            except Exception:
                restrict_end_sigma = -1.0  # unknown sampling: never release

        bundle = {
            "segments": segments,
            "txt_total": off,
            "masks": [r["mask"] for r in regions],
            "restrict_img_attn": restrict_img_attn,
            "restrict_end_sigma": restrict_end_sigma,
            "exclusive_masks": exclusive_masks,
            "base_loras_exclude_regions": base_loras_exclude_regions,
            "adaptive": {
                "mode": {"off": "off", "refine boxes": "refine",
                         "free (ignore boxes)": "free"}.get(
                             adaptive_masks, "off"),
                "steps": adaptive_steps,
                "threshold": adaptive_threshold,
            },
            "loras": lora_specs,
        }
        if region_lock_strength > 0:
            m.set_model_sampler_post_cfg_function(
                _make_region_lock(bundle, region_lock_strength,
                                  region_lock_start, region_lock_end))
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
