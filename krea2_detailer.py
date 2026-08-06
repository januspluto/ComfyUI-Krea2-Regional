"""Krea2 Regional Detailer — subject-aware per-region refinement pass.

Run AFTER the first sampler + VAE decode. Two operating modes:

DETECTION MODE (recommended — connect `detection_masks` or `segs`):
  Feed it subject silhouettes from any detector (SAM3, Impact Pack
  UltralyticsDetector person/face models, etc). The node then:
    1. matches each detected subject to a Builder region by mask overlap
       (with a nearest-center fallback so subjects that DRIFTED out of
       their box still find their region),
    2. crops around the DETECTED subject — not the drawn box,
    3. resamples the crop with that region's prompt conditioning and its
       LoRAs applied plainly (single-subject crop, no token gating),
    4. composites back through the detected silhouette (grown/feathered),
       so only that subject's own pixels are repainted — never the
       background, never a neighbour that strayed into the box.
  Regions with no matching detection are left untouched.

BOX MODE (fallback — nothing connected to `detection_masks`/`segs`):
  Crops and composites by the Builder region masks themselves. Simple, but
  if subjects overlap each other's boxes this repaints whatever sits inside
  the box with that region's identity — which can CREATE bleed. Prefer
  detection mode for multi-character images.

Regions are processed back-to-front (Builder list order = depth), so the
front region wins where feathered composites meet.
"""

import copy
import logging

import torch
import torch.nn.functional as F

import comfy.samplers
import comfy.sd
import nodes

try:
    from .caption_utils import _grow_feather
except ImportError:  # direct script execution / tests
    from caption_utils import _grow_feather


def _mask_bbox(mask, thresh=0.5):
    """(H, W) mask -> (x0, y0, x1, y1) pixel bounds of mask > thresh."""
    ys, xs = torch.where(mask > thresh)
    if xs.numel() == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _round16(v, lo=64):
    """Krea 2 works on 16 px pixel patches; keep sampled crops aligned."""
    return max(lo, int(round(float(v) / 16.0)) * 16)


def _nhwc_resize(img, h, w):
    """(B, H, W, C) bicubic resize with antialias, clamped to [0, 1]."""
    x = img.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(h, w), mode="bicubic", antialias=True)
    return x.permute(0, 2, 3, 1).clamp(0, 1)


def _to_hw(m, H, W):
    """Any mask-ish tensor -> (H, W) float on cpu."""
    m = torch.as_tensor(m, dtype=torch.float32).cpu()
    while m.dim() > 2:
        m = m[0]
    if tuple(m.shape) != (H, W):
        m = F.interpolate(m[None, None], size=(H, W), mode="bilinear",
                          align_corners=False)[0, 0]
    return m.clamp(0, 1)


def _segs_to_masks(segs, H, W):
    """Duck-typed Impact-Pack SEGS -> list of (H, W) float masks.

    SEGS = (shape, [SEG, ...]); each SEG carries cropped_mask + crop_region.
    Parsed via getattr so we don't import (or depend on) Impact Pack."""
    out = []
    try:
        seg_list = segs[1]
    except (TypeError, IndexError):
        return out
    for s in seg_list or []:
        try:
            cm = getattr(s, "cropped_mask", None)
            cr = getattr(s, "crop_region", None)
            if cm is None or cr is None:
                continue
            m = torch.as_tensor(cm, dtype=torch.float32).cpu()
            while m.dim() > 2:
                m = m[0]
            x1, y1, x2, y2 = [int(v) for v in cr]
            x1 = max(0, min(x1, W)); x2 = max(x1 + 1, min(x2, W))
            y1 = max(0, min(y1, H)); y2 = max(y1 + 1, min(y2, H))
            if m.shape != (y2 - y1, x2 - x1):
                m = F.interpolate(m[None, None], size=(y2 - y1, x2 - x1),
                                  mode="bilinear", align_corners=False)[0, 0]
            full = torch.zeros(H, W)
            full[y1:y2, x1:x2] = m.clamp(0, 1)
            out.append(full)
        except Exception as e:
            logging.warning("[Krea2Detailer] skipped a SEG: %s", e)
    return out


def _apply_crop_shift(model_patcher, w, h,
                      base_shift=0.5, max_shift=1.15,
                      min_resolution=256, max_resolution=1280):
    """Apply Krea 2 RAW's resolution-derived timestep-shift ``mu``.

    This is an advanced RAW-only mode. Krea 2 Turbo was distilled with a
    fixed mu=1.15 and should normally keep the model default. ComfyUI's
    ModelSamplingFlux ``shift`` field already stores mu and exponentiates it
    internally, so pass mu directly (never ``exp(mu)``).
    """
    try:
        import comfy.model_sampling as ms

        current = None
        try:
            current = model_patcher.get_model_object("model_sampling")
        except Exception:
            current = getattr(getattr(model_patcher, "model", None),
                              "model_sampling", None)
        if current is None or not isinstance(current, ms.ModelSamplingFlux):
            return model_patcher

        # Krea 2: VAE compression 8 x model patch 2 = one image token per
        # 16x16 pixels. Official RAW endpoints are 256px and 1280px.
        x1 = (min_resolution // 16) ** 2
        x2 = (max_resolution // 16) ** 2
        seq = (max(16, int(w)) // 16) * (max(16, int(h)) // 16)
        seq = max(x1, min(seq, x2))  # avoid extrapolating on tiny/wide crops
        mu = base_shift + (seq - x1) * (max_shift - base_shift) / (x2 - x1)

        m = model_patcher.clone()
        sampling = copy.deepcopy(current)
        sampling.set_parameters(shift=float(mu))
        m.add_object_patch("model_sampling", sampling)
        logging.info("[Krea2Detailer] RAW crop shift: %dx%d, tokens=%d, "
                     "mu=%.4f", w, h, seq, mu)
        return m
    except Exception as e:
        logging.info("[Krea2Detailer] per-crop shift unavailable "
                     "(%s) — using the model's shift as-is.", e)
        return model_patcher


def _center(mask):
    ys, xs = torch.where(mask > 0.5)
    if xs.numel() == 0:
        return None
    return float(xs.float().mean()), float(ys.float().mean())


def _match_detections(dets, region_masks):
    """Greedy 1:1 assignment: region index -> detection mask (or None).

    Pass 1 scores every (region, detection) pair by containment x coverage
    (how much of the subject sits in the box x how much of the box it
    fills) and assigns best-first. Pass 2 rescues DRIFTED subjects: any
    still-unmatched detection goes to the nearest still-unmatched region
    by center distance."""
    n_r, n_d = len(region_masks), len(dets)
    assign = {i: None for i in range(n_r)}
    if n_d == 0:
        return assign
    rbins = [rm > 0.5 for rm in region_masks]
    dbins = [d > 0.5 for d in dets]
    dareas = [max(int(d.sum().item()), 1) for d in dbins]
    rareas = [max(int(r.sum().item()), 1) for r in rbins]
    scored = []
    for i in range(n_r):
        for j in range(n_d):
            if dareas[j] < 64:
                continue
            inter = int((rbins[i] & dbins[j]).sum().item())
            if inter == 0:
                continue
            scored.append((inter / dareas[j] * (inter / rareas[i]), i, j))
    used_r, used_d = set(), set()
    for score, i, j in sorted(scored, reverse=True):
        if i in used_r or j in used_d:
            continue
        assign[i] = dets[j]
        used_r.add(i); used_d.add(j)
    # drift rescue: only pair a leftover subject with a *nearby* region.
    # An unbounded nearest-centre pass can assign the wrong prompt/LoRA to a
    # completely different person when one detection is missing.
    left_d = [j for j in range(n_d) if j not in used_d and dareas[j] >= 64]
    left_r = [i for i in range(n_r) if i not in used_r]
    if left_d and left_r:
        pairs = []
        for i in left_r:
            rc = _center(region_masks[i])
            rb = _mask_bbox(region_masks[i])
            if rc is None or rb is None:
                continue
            rw, rh = rb[2] - rb[0], rb[3] - rb[1]
            max_dist = max(32.0, 0.75 * (rw * rw + rh * rh) ** 0.5)
            for j in left_d:
                dc = _center(dets[j])
                if dc is None:
                    continue
                dist = ((rc[0] - dc[0]) ** 2 + (rc[1] - dc[1]) ** 2) ** 0.5
                if dist <= max_dist:
                    pairs.append((dist, i, j))
        for dist, i, j in sorted(pairs):
            if i in used_r or j in used_d:
                continue
            assign[i] = dets[j]
            used_r.add(i); used_d.add(j)
            logging.info("[Krea2Detailer] region %d matched a drifted "
                         "subject %.0f px from its box", i + 1, dist)
    return assign


class Krea2RegionalDetailer:
    """Detect -> match -> per-subject resample -> composite."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL", {
                    "tooltip": "The BASE model (NOT the output of Apply "
                               "Regional). Each region's LoRAs are applied "
                               "plainly to a clone per crop."}),
                "vae": ("VAE",),
                "regions": ("KREA2_REGIONS", {
                    "tooltip": "From the Builder — carries each region's "
                               "conditioning, mask, and LoRAs."}),
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0,
                                  "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0,
                    "step": 0.01,
                    "tooltip": "How much to redraw per subject. With shift "
                               "on auto the schedule is correctly "
                               "calibrated, so values bite harder than "
                               "before: 0.2-0.25 tightens likeness while "
                               "keeping pose/composition stable; 0.35+ "
                               "starts recomposing inside the mask."}),
            },
            "optional": {
                "detection_masks": ("MASK", {
                    "tooltip": "One mask per detected subject (SAM3, "
                               "SEGS-to-Mask-Batch...). Connecting this "
                               "switches the node to detection mode: crops "
                               "follow the detected subjects and only their "
                               "silhouettes are repainted."}),
                "segs": ("SEGS", {
                    "tooltip": "Impact Pack detections (person/face). Used "
                               "when detection_masks is not connected."}),
                "base_conditioning": ("CONDITIONING", {
                    "tooltip": "Optional: the Builder's base_conditioning, "
                               "used as the conditioning-extras template "
                               "and zeroed-negative fallback."}),
                "negative": ("CONDITIONING",),
                "crop_padding": ("FLOAT", {"default": 0.25, "min": 0.0,
                    "max": 1.0, "step": 0.05,
                    "tooltip": "Context around the subject, as a fraction "
                               "of its bbox's longer side."}),
                "detail_size": ("INT", {"default": 1024, "min": 256,
                    "max": 4096, "step": 16,
                    "tooltip": "The crop is resampled with its longer side "
                               "at this resolution."}),
                "feather_px": ("INT", {"default": 24, "min": 0, "max": 256,
                    "tooltip": "Composite feather at image resolution."}),
                "mask_grow_px": ("INT", {"default": 8, "min": 0, "max": 128,
                    "tooltip": "Detection mode: grow the detected "
                               "silhouette before feathering so hair/edges "
                               "get repainted too."}),
                "shift": (["model default", "auto (RAW only, per-crop)"],
                    {"default": "model default",
                     "tooltip": "'model default' (recommended) reuses the "
                                "shift the initial image was generated "
                                "with — keeping the detail pass in the "
                                "same schedule regime the composition was "
                                "created in, which best preserves existing "
                                "content at low denoise. 'auto' recomputes "
                                "Krea 2 RAW's resolution-derived mu. "
                                "Do not use auto with Turbo: Turbo was "
                                "distilled at fixed mu=1.15."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "applied_masks")
    FUNCTION = "detail"
    CATEGORY = "conditioning/krea2_regional"

    def detail(self, image, model, vae, regions, seed, steps, cfg,
               sampler_name, scheduler, denoise, detection_masks=None,
               segs=None, base_conditioning=None, negative=None,
               crop_padding=0.25, detail_size=1024, feather_px=24,
               mask_grow_px=8, shift="model default"):
        B, H, W, _ = image.shape
        union = torch.zeros(B, H, W)
        if not regions or denoise <= 0.0:
            return (image, union)
        out = image.clone()

        base_dict = {}
        try:
            if base_conditioning and len(base_conditioning[0]) > 1:
                base_dict = dict(base_conditioning[0][1])
        except (TypeError, IndexError):
            pass

        # ---- detections (subject silhouettes), if provided ----
        dets = []
        if detection_masks is not None:
            dm = detection_masks
            if dm.dim() == 2:
                dm = dm[None]
            dets = [_to_hw(dm[j], H, W) for j in range(dm.shape[0])]
        elif segs is not None:
            dets = _segs_to_masks(segs, H, W)
        detection_mode = len(dets) > 0
        if (detection_masks is not None or segs is not None) \
                and not detection_mode:
            logging.warning("[Krea2Detailer] detector input connected but "
                            "yielded no usable masks — nothing detailed. "
                            "(Box mode would risk cross-region bleed, so it "
                            "is not used as a silent fallback.)")
            return (out, union)

        valid_regions = [(idx, r, _to_hw(r["mask"], H, W))
                         for idx, r in enumerate(regions)
                         if r and r.get("mask") is not None]
        region_masks = [item[2] for item in valid_regions]
        if detection_mode:
            assign = _match_detections(dets, region_masks)
            matched = sum(1 for v in assign.values() if v is not None)
            logging.info("[Krea2Detailer] matched %d/%d detections to %d "
                         "regions", matched, len(dets), len(region_masks))

        # back-to-front: the front region (index 0, top of the Builder
        # list) composites LAST and wins where feathered edges meet.
        order = list(range(len(region_masks)))[::-1]

        for b in range(B):
            for local_ridx in order:
                ridx, reg, region_mask = valid_regions[local_ridx]
                reg = reg or {}
                cond = reg.get("cond")
                if cond is None:
                    continue
                if detection_mode:
                    m = assign.get(local_ridx)
                    if m is None:
                        continue  # subject not found: leave pixels alone
                else:
                    m = region_mask
                bbox = _mask_bbox(m)
                if bbox is None:
                    continue
                x0, y0, x1, y1 = bbox
                pad = int(round(crop_padding * max(x1 - x0, y1 - y0)))
                x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
                x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
                cw, ch = x1 - x0, y1 - y0
                if cw < 16 or ch < 16:
                    continue

                # upscale-only: sampling a large subject BELOW its
                # native resolution returns it softer than the pixels
                # around it (visible quality seam). Aspect is preserved by
                # deriving th from the rounded tw, not rounding both.
                scale = max(1.0, detail_size / float(max(cw, ch)))
                tw = _round16(cw * scale)
                th = _round16(tw * ch / cw)
                # Always take context from the immutable first-pass image.
                # Otherwise overlapping subjects recursively consume prior
                # detail passes and amplify texture/identity drift.
                crop = image[b:b + 1, y0:y1, x0:x1, :3]
                crop_up = _nhwc_resize(crop, th, tw)

                # Anchor the context during sampling, not just at the final
                # pixel composite. Without a latent noise mask the entire
                # padded crop is re-noised, so Krea can reinterpret pose and
                # surroundings even though those pixels are discarded later.
                sample_mm = m[y0:y1, x0:x1]
                grow = mask_grow_px if detection_mode else 0
                if grow > 0 or feather_px > 0:
                    sample_mm = _grow_feather(sample_mm, grow, feather_px)
                noise_mask = F.interpolate(
                    sample_mm[None, None], size=(th, tw), mode="bilinear",
                    align_corners=False).clamp(0, 1)
                latent = {"samples": vae.encode(crop_up),
                          "noise_mask": noise_mask}

                # Optional Krea 2 RAW per-crop shift. Turbo should inherit its
                # fixed model-default mu=1.15.
                m_model = model
                if str(shift).startswith("auto"):
                    m_model = _apply_crop_shift(m_model, tw, th)
                # region LoRAs, applied plainly (single-subject crop)
                for e in reg.get("loras") or []:
                    try:
                        m_model = comfy.sd.load_lora_for_models(
                            m_model, None, e["sd"],
                            float(e.get("strength", 1.0)), 0.0)[0]
                    except Exception as err:
                        logging.warning(
                            "[Krea2Detailer] lora '%s' failed to apply: %s",
                            e.get("label", "?"), err)

                pos = [[cond, dict(base_dict)]]
                neg = negative
                if neg is None:
                    nd = dict(base_dict)
                    if isinstance(nd.get("pooled_output"), torch.Tensor):
                        nd["pooled_output"] = torch.zeros_like(
                            nd["pooled_output"])
                    neg = [[torch.zeros_like(cond), nd]]

                import time as _time
                _t0 = _time.time()
                samples = nodes.common_ksampler(
                    m_model, seed + ridx, steps, cfg, sampler_name,
                    scheduler, pos, neg, latent, denoise=denoise)[0]
                det = vae.decode(samples["samples"])
                # video VAEs (e.g. Wan) decode to (B, T, H, W, C); flatten
                # the frame dim like core VAEDecode does and keep frame 0
                if det.dim() == 5:
                    det = det.reshape(-1, det.shape[-3], det.shape[-2],
                                      det.shape[-1])[:1]
                if det.shape[1] != th or det.shape[2] != tw:
                    det = det[:, :th, :tw, :]
                det = _nhwc_resize(det[:, :, :, :3], ch, cw).to(out)

                mm = sample_mm
                mm4 = mm.clamp(0, 1).to(out)[None, :, :, None]
                out[b:b + 1, y0:y1, x0:x1, :3] = \
                    det * mm4 + out[b:b + 1, y0:y1, x0:x1, :3] * (1 - mm4)
                union[b, y0:y1, x0:x1] = torch.maximum(
                    union[b, y0:y1, x0:x1], mm.clamp(0, 1).cpu())
                logging.info("[Krea2Detailer] region %d: %dx%d crop "
                             "sampled at %dx%d in %.1fs", ridx + 1,
                             cw, ch, tw, th, _time.time() - _t0)

        return (out, union)


DETAILER_NODE_CLASS_MAPPINGS = {
    "Krea2RegionalDetailer": Krea2RegionalDetailer,
}
DETAILER_NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2RegionalDetailer": "Krea 2 - Regional Detailer",
}
