"""CPU tests for ComfyUI-Krea2-Regional against ComfyUI's real krea2 module.

Run from the ComfyUI repo root:  python ../ComfyUI-Krea2-Regional/test_nodes.py
"""
import os
import sys

sys.path.insert(0, os.environ.get("COMFYUI_PATH", os.path.expanduser("~/ComfyUI")))
from comfy.cli_args import args
args.cpu = True

import torch
import comfy.ops
import comfy.patcher_extension
from comfy.patcher_extension import WrappersMP
from comfy.ldm.krea2.model import SingleStreamDiT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import krea2_regional as kr

torch.manual_seed(0)
ops = comfy.ops.disable_weight_init

CFG = dict(features=64, tdim=32, txtdim=48, heads=4, kvheads=2, multiplier=4,
           layers=2, patch=2, channels=16, txtlayers=3, txtheads=4, txtkvheads=4)

model = SingleStreamDiT(**CFG, dtype=torch.float32, device="cpu", operations=ops)
for p in model.parameters():
    torch.nn.init.normal_(p, std=0.02)
model.eval()

B, H, W = 2, 16, 16                      # latent 16x16 -> 8x8 patch grid
h_, w_ = H // 2, W // 2
imglen = h_ * w_
seg_lens = [3, 2, 2]                     # base, region1, region2 prompt lengths
txtlen = sum(seg_lens)
segments = [(0, 3), (3, 5), (5, 7)]

x = torch.randn(B, 16, H, W)
timesteps = torch.full((B,), 0.5)
context = torch.randn(B, txtlen, CFG["txtlayers"] * CFG["txtdim"])

def run(model, x, ts, ctx, transformer_options):
    with torch.no_grad():
        return model.forward(x, ts, ctx, transformer_options=transformer_options)

baseline = run(model, x, timesteps, context, {})

def wrapped_opts(bundle, cond_or_uncond=(0,)):
    to = {kr.BUNDLE_KEY: bundle, "cond_or_uncond": list(cond_or_uncond)}
    comfy.patcher_extension.add_wrapper_with_key(
        WrappersMP.DIFFUSION_MODEL, kr.BUNDLE_KEY, kr._diffusion_wrapper, to)
    return to

# masks: region1 = left half, region2 = right half (pixel space 64x64)
m1 = torch.zeros(64, 64); m1[:, :32] = 1.0
m2 = torch.zeros(64, 64); m2[:, 32:] = 1.0

def bundle(loras=(), restrict=False, masks=(m1, m2)):
    return {"segments": segments, "txt_total": txtlen, "masks": list(masks),
            "restrict_img_attn": restrict, "loras": list(loras)}

# ---- 1. wrong txt length (e.g. the negative prompt) -> identical to baseline
neg_ctx = torch.randn(B, 5, CFG["txtlayers"] * CFG["txtdim"])
ref_neg = run(model, x, timesteps, neg_ctx, {})
out_neg = run(model, x, timesteps, neg_ctx, wrapped_opts(bundle()))
assert (ref_neg - out_neg).abs().max() == 0, "non-matching cond must pass through"
print("1) passthrough for non-matching conditioning: ok")

# ---- 2. single segment + no regions == baseline (mask is all-allow)
b_triv = {"segments": [(0, txtlen)], "txt_total": txtlen, "masks": [],
          "restrict_img_attn": False, "loras": []}
out_triv = run(model, x, timesteps, context, wrapped_opts(b_triv))
d = (baseline - out_triv).abs().max().item()
print(f"2) trivial-region equivalence vs baseline: max|diff| = {d:.3e}")
assert d < 1e-5

# ---- 3. regional masking changes output, and region isolation holds:
# permuting REGION-2's prompt content must not change REGION-1's output pixels.
out_reg = run(model, x, timesteps, context, wrapped_opts(bundle()))
assert (out_reg - baseline).abs().max() > 1e-4, "regional mask should alter output"

ctx_perm = context.clone()
ctx_perm[:, 5:7] = torch.randn_like(ctx_perm[:, 5:7])  # replace region-2 prompt
out_perm = run(model, x, timesteps, ctx_perm, wrapped_opts(bundle(restrict=True)))
out_ref = run(model, x, timesteps, context, wrapped_opts(bundle(restrict=True)))
diff_map = (out_perm - out_ref).abs().sum(dim=1)     # (B, H, W) per-pixel diff
left, right = diff_map[:, :, : W // 2], diff_map[:, :, W // 2:]
print(f"3) isolation: left-region diff = {left.max():.2e}, right-region diff = {right.max():.2e}")
assert left.max() < 1e-5 and right.max() > 1e-4, \
    "changing region-2's prompt must only affect region-2 pixels (restricted attn)"

# ---- 4. LoRA class-swap: state_dict keys and normal weight access unchanged
import comfy.model_patcher

# Wrap model in a container with .diffusion_model, as ComfyUI does
class Container(torch.nn.Module):
    def __init__(self, dm):
        super().__init__()
        self.diffusion_model = dm

container = Container(model)
patcher = comfy.model_patcher.ModelPatcher(container, load_device="cpu", offload_device="cpu")

keys_before = set(container.state_dict().keys())
rank, feat = 4, CFG["features"]
mlp_in = model.blocks[0].mlp.down.weight.shape[1]
sd = {
    "transformer.blocks.0.attn.wq.lora_A.weight": torch.randn(rank, feat) * 0.1,
    "transformer.blocks.0.attn.wq.lora_B.weight": torch.randn(feat, rank) * 0.1,
    "lora_unet_blocks_1_mlp_down.lora_down.weight": torch.randn(rank, mlp_in) * 0.1,
    "lora_unet_blocks_1_mlp_down.lora_up.weight": torch.randn(feat, rank) * 0.1,
    "lora_unet_blocks_1_mlp_down.alpha": torch.tensor(2.0),
}
n = kr._inject_lora(patcher, sd, uid="loraA", strength=1.0)
assert n == 2, f"expected 2 patched layers, got {n}"

patcher.patch_model(device_to="cpu")   # applies object patches
assert isinstance(container.diffusion_model.blocks[0].attn.wq, kr._RegionalLoRAMixin)
keys_after = set(container.state_dict().keys())
assert keys_before == keys_after, "state_dict keys must be unchanged by patching"
assert container.diffusion_model.blocks[0].attn.wq.weight is model.blocks[0].attn.wq.weight or \
       torch.equal(container.state_dict()["diffusion_model.blocks.0.attn.wq.weight"],
                   model.blocks[0].attn.wq.weight), "weights must stay shared/resolvable"
print("4) lora class-swap keeps module paths + state_dict keys: ok")

# ---- 5. LoRA gating: no CTX == baseline; region-masked LoRA only changes its region
out_after_patch = run(container.diffusion_model, x, timesteps, context, {})
d = (out_after_patch - baseline).abs().max().item()
assert d < 1e-6, f"LoRA with empty CTX must be inert, diff={d}"

lora_specs = [{"uid": "loraA", "weight": 1.0, "seg": 1, "mask_idx": 0}]  # region 1 = left
out_lora = run(container.diffusion_model, x, timesteps, context,
               wrapped_opts(bundle(loras=lora_specs, restrict=True)))
out_nolora = run(container.diffusion_model, x, timesteps, context,
                 wrapped_opts(bundle(restrict=True)))
dm = (out_lora - out_nolora).abs().sum(dim=1)
print(f"5) lora locality: left diff = {dm[:, :, : W//2].max():.2e}, "
      f"right diff = {dm[:, :, W//2:].max():.2e}")
assert dm[:, :, : W // 2].max() > 1e-4, "LoRA must affect its region"
assert dm[:, :, W // 2:].max() < 1e-5, "LoRA must not leak (restricted attn)"

# ---- 6. batched cond+uncond rows: uncond rows stay clean
out_mixed = run(container.diffusion_model, x, timesteps, context,
                wrapped_opts(bundle(loras=lora_specs), cond_or_uncond=(0, 1)))
row_ref = run(container.diffusion_model, x, timesteps, context, {})
d_uncond = (out_mixed[1] - row_ref[1]).abs().max().item()
d_cond = (out_mixed[0] - row_ref[0]).abs().max().item()
print(f"6) mixed batch: cond-row diff = {d_cond:.2e} (>0), uncond-row diff = {d_uncond:.2e} (=0)")
assert d_uncond < 1e-6 and d_cond > 1e-4

patcher.unpatch_model()
assert not isinstance(container.diffusion_model.blocks[0].attn.wq, kr._RegionalLoRAMixin), \
    "unpatch must restore original layers"
print("7) unpatch restores original modules: ok")

print("\nall tests passed")

# ---- 8. 5D temporal latent (the reported crash: B,C,T,H,W with C=16)
x5 = x[:1].unsqueeze(2)  # (1, 16, 1, H, W) — exactly what model_base passes
ref5 = run(container.diffusion_model, x5, timesteps[:1], context[:1], {})
out5 = run(container.diffusion_model, x5, timesteps[:1], context[:1],
           wrapped_opts(bundle(loras=lora_specs, restrict=True)))
assert out5.shape == ref5.shape
d5 = (out5 - ref5).abs().max().item()
assert d5 > 1e-5, "regional path must actually engage on 5D input"
b5t = {"segments": [(0, txtlen)], "txt_total": txtlen, "masks": [],
       "restrict_img_attn": False, "loras": []}
triv5 = run(container.diffusion_model, x5, timesteps[:1], context[:1],
            wrapped_opts(b5t))
assert (triv5 - ref5).abs().max().item() < 1e-5, "5D trivial == baseline"
print("8) 5D temporal latent (crash regression): ok")

# ---- 9. foreign-batch sub-call (Krea2T-Enhancer style): a wrapper that
# re-runs the txtfusion refiners with its own batch must not crash even when
# our masks are per-row (mixed cond/uncond).
to9 = wrapped_opts(bundle(), cond_or_uncond=(0, 1))  # forces per-row masks
_ = run(container.diffusion_model, x, timesteps, context, to9)  # no crash

# direct probe: per-row mask vs batch-1 query must fall back to unmasked
import krea2_regional as kr2
probe = {}
def fake_func(q, k, v, heads, mask=None, **kw):
    probe["mask"] = mask
    return q
to10 = wrapped_opts(bundle(), cond_or_uncond=(0, 1))
def probing_wrapper(executor, x, ts, ctx, *a, **kw):
    transformer_options = kw.get("transformer_options")
    if transformer_options is None and a and isinstance(a[-1], dict):
        transformer_options = a[-1]
    ov = transformer_options["optimized_attention_override"]
    q1 = torch.zeros(1, 4, txtlen, 8)  # batch-1 refiner call, like the enhancer
    ov(fake_func, q1, q1, q1, 4)
    return executor(x, ts, ctx, *a, **kw)
comfy.patcher_extension.add_wrapper_with_key(
    WrappersMP.DIFFUSION_MODEL, "zz_probe", probing_wrapper, to10)
_ = run(container.diffusion_model, x, timesteps, context, to10)
assert probe["mask"] is None, \
    "per-row mask must be skipped (not crash) for foreign batch sizes"
print("9) foreign-batch sub-call fallback: ok")

print("\nall tests passed (incl. regressions)")

# ---- 10. txtfusion LoRA coverage: keys patch, gate correctly, stay local
txtdim = CFG["txtdim"]
mlp_up = model.txtfusion.refiner_blocks[0].mlp.up.weight.shape[0]
sd_tf = {
    "txtfusion.refiner_blocks.0.attn.wq.lora_A.weight": torch.randn(4, txtdim) * 0.1,
    "txtfusion.refiner_blocks.0.attn.wq.lora_B.weight": torch.randn(txtdim, 4) * 0.1,
    "lora_unet_txtfusion_layerwise_blocks_0_mlp_up.lora_down.weight":
        torch.randn(4, txtdim) * 0.1,
    "lora_unet_txtfusion_layerwise_blocks_0_mlp_up.lora_up.weight":
        torch.randn(mlp_up, 4) * 0.1,
}
n_tf = kr._inject_lora(patcher, sd_tf, uid="loraTF", strength=1.0)
assert n_tf == 2, f"txtfusion keys must patch, got {n_tf}"
patcher.patch_model(device_to="cpu")

ref10 = run(container.diffusion_model, x, timesteps, context, {})
assert (ref10 - baseline).abs().max() < 1e-6, "inert without CTX"

specs_tf = [{"uid": "loraTF", "weight": 1.0, "seg": 1, "mask_idx": 0}]
out_tf = run(container.diffusion_model, x, timesteps, context,
             wrapped_opts(bundle(loras=specs_tf, restrict=True)))
out_ntf = run(container.diffusion_model, x, timesteps, context,
              wrapped_opts(bundle(restrict=True)))
dm10 = (out_tf - out_ntf).abs().sum(dim=1)
print(f"10) txtfusion lora: left diff = {dm10[:, :, : W//2].max():.2e}, "
      f"right diff = {dm10[:, :, W//2:].max():.2e}")
assert dm10[:, :, : W // 2].max() > 1e-5, "txtfusion lora must engage"
assert dm10[:, :, W // 2:].max() < 1e-6, "txtfusion lora must stay in-region"
patcher.unpatch_model()

print("\nall tests passed (incl. txtfusion)")

# ---- 11. exclusive masks: overlapping regions resolve winner-take-all
from krea2_regional import _exclusive_soft
soft_ov = torch.zeros(2, imglen)
soft_ov[0, :40] = 0.9   # region A covers tokens 0..39
soft_ov[1, 30:64] = 0.6 # region B overlaps A on 30..39
ex = _exclusive_soft(soft_ov)
assert (ex[0, 30:40] == 0.9).all() and (ex[1, 30:40] == 0).all(), \
    "stronger region wins the overlap"
assert (ex[1, 40:64] == 0.6).all(), "non-overlap untouched"
assert ((ex > 0).sum(0) <= 1).all(), "masks are disjoint after exclusivity"

# integration: bundle flag flows through the wrapper without breaking anything
m_ov1 = torch.zeros(64, 64); m_ov1[:, :40] = 1.0
m_ov2 = torch.zeros(64, 64); m_ov2[:, 24:] = 1.0   # solid 2-token overlap band
b_ex = {"segments": segments, "txt_total": txtlen, "masks": [m_ov1, m_ov2],
        "restrict_img_attn": True, "exclusive_masks": True, "loras": lora_specs}
b_nx = dict(b_ex, exclusive_masks=False)
out_ex = run(container.diffusion_model, x, timesteps, context, wrapped_opts(b_ex))
out_nx = run(container.diffusion_model, x, timesteps, context, wrapped_opts(b_nx))
assert (out_ex - out_nx).abs().max() > 1e-6, \
    "exclusivity must change behavior when masks overlap"
print("11) exclusive masks (anti-bleed): ok")

print("\nall tests passed (incl. exclusivity)")

# ---- 12. adaptive mask finalize (pure function)
from krea2_regional import _finalize_adaptive, _grow_tokens
Hh = Ww = 8
box0 = torch.zeros(Hh, Ww); box0[:, :4] = 1.0      # left half
box1 = torch.zeros(Hh, Ww); box1[:, 4:] = 1.0      # right half
base = torch.stack([box0.reshape(-1), box1.reshape(-1)])
aff = torch.zeros(2, Hh * Ww)
blob0 = torch.zeros(Hh, Ww); blob0[2:6, 1:4] = 5.0   # subject inside box0
blob1 = torch.zeros(Hh, Ww); blob1[3:7, 5:8] = 5.0   # subject inside box1
aff[0] = blob0.reshape(-1); aff[1] = blob1.reshape(-1)

ref = _finalize_adaptive(aff, base, "refine", 0.45, Hh, Ww)
assert ref.shape == (2, Hh * Ww)
grown0 = _grow_tokens(base[:1], Hh, Ww, 2)[0]
assert ((ref[0] > 0) <= (grown0 > 0.05)).all(), \
    "refine mode must stay inside the grown box"
r0 = ref[0].reshape(Hh, Ww)
assert r0[3, 2] > 0, "discovered blob center kept"
assert r0[3, 3].item() >= 0 and r0[0, 7] == 0, "far corner excluded"
assert ref[0].sum() < base[0].sum(), "refined tighter than the full box"

# free mode with a blob OUTSIDE its box: allowed to move
aff_out = aff.clone()
blob_out = torch.zeros(Hh, Ww); blob_out[2:5, 5:8] = 5.0  # region0 -> right side
aff_out[0] = blob_out.reshape(-1)
fr = _finalize_adaptive(aff_out, base, "free", 0.45, Hh, Ww)
assert fr[0].reshape(Hh, Ww)[3, 6] > 0, "free mode follows the subject"

# empty discovery falls back to the drawn box
aff_empty = torch.zeros(2, Hh * Ww); aff_empty[1] = blob1.reshape(-1)
fb = _finalize_adaptive(aff_empty, base, "refine", 0.99, Hh, Ww)
assert torch.equal(fb[0], base[0]) or (fb[0] > 0).sum() > 0, "fallback works"
print("12) adaptive finalize (refine/free/fallback): ok")

# ---- 13. adaptive capture -> refine cycle on the real model
b_ad = dict(bundle(loras=lora_specs, restrict=True))
b_ad["adaptive"] = {"mode": "refine", "steps": 1, "threshold": 0.3}
to_ad = wrapped_opts(b_ad)
ts_desc = [0.9, 0.6, 0.3]
outs = []
with torch.no_grad():
    for tv in ts_desc:
        outs.append(run(container.diffusion_model, x,
                        torch.full((B,), tv), context, to_ad))
rt = b_ad["_rt"]
assert rt["aff"] is not None and rt["aff"].abs().sum() > 0, \
    "capture must accumulate affinity"
assert rt["refined"] is not None, "masks must be refined after the window"
assert rt["refined"].shape == (2, imglen)
assert rt["call"] == 3
print("13) adaptive capture -> refine on real model: ok")

# ---- 14. new-run detection resets adaptive state (timestep jumps up)
with torch.no_grad():
    run(container.diffusion_model, x, torch.full((B,), 0.95), context, to_ad)
assert rt["call"] == 1 and rt["refined"] is None, \
    "rising timestep must reset per-run adaptive state"
print("14) per-run reset: ok")

# ---- 15. base_loras_exclude_regions carves regions out of global loras
from krea2_regional import _build_token_masks
bx = {"loras": [{"uid": "g", "weight": 1.0, "seg": None, "mask_idx": None}],
      "base_loras_exclude_regions": True}
soft2 = torch.zeros(2, imglen); soft2[0, :10] = 1.0; soft2[1, 20:30] = 0.5
tm = _build_token_masks(bx, segments, soft2, imglen, [True], "cpu")
v = tm["g"][0, :, 0]
assert v[:txtlen].min() == 1.0, "text tokens keep full weight"
assert v[txtlen + 0] == 0.0, "fully-masked region token excluded"
assert abs(v[txtlen + 25].item() - 0.5) < 1e-6, "soft region partially excluded"
assert v[txtlen + 40] == 1.0, "background keeps the base lora"
print("15) base_loras_exclude_regions: ok")

print("\nall tests passed (incl. adaptive)")

# ---- 16. LoKr: _kron_apply matches the materialized Kronecker product
from krea2_regional import _kron_apply, _normalize_lora_sd
torch.manual_seed(7)
for (a1, b1, a2, b2) in [(4, 4, 16, 16), (3, 5, 7, 2), (8, 2, 6, 16)]:
    w1k = torch.randn(a1, b1); w2k = torch.randn(a2, b2)
    xk = torch.randn(2, 11, b1 * b2)
    ref_k = xk @ torch.kron(w1k, w2k).T
    got_k = _kron_apply(xk, w1k, w2k)
    err = (ref_k - got_k).abs().max().item()
    assert err < 1e-4, f"kron mismatch {err} for {(a1,b1,a2,b2)}"
print("16) _kron_apply == materialized kron (incl. rectangular): ok")

# ---- 17. LoKr normalization: direct, factorized+alpha, tucker skip
sd_lokr = {
    "lora_unet_blocks_0_attn_wq.lokr_w1": torch.randn(4, 4) * 0.1,
    "lora_unet_blocks_0_attn_wq.lokr_w2": torch.randn(16, 16) * 0.1,
    "lora_unet_blocks_1_attn_wo.lokr_w1": torch.randn(4, 4) * 0.1,
    "lora_unet_blocks_1_attn_wo.lokr_w2_a": torch.randn(16, 8) * 0.1,
    "lora_unet_blocks_1_attn_wo.lokr_w2_b": torch.randn(8, 16) * 0.1,
    "lora_unet_blocks_1_attn_wo.alpha": torch.tensor(4.0),
    "lora_unet_blocks_1_mlp_gate.lokr_w1": torch.randn(4, 4),
    "lora_unet_blocks_1_mlp_gate.lokr_w2": torch.randn(4, 4, 3, 3),  # conv
    "lora_unet_blocks_0_attn_wk.lokr_w1": torch.randn(4, 4),
    "lora_unet_blocks_0_attn_wk.lokr_w2_a": torch.randn(16, 8),
    "lora_unet_blocks_0_attn_wk.lokr_w2_b": torch.randn(8, 16),
    "lora_unet_blocks_0_attn_wk.lokr_t2": torch.randn(8, 8, 3, 3),  # tucker
}
gk = _normalize_lora_sd(sd_lokr)
assert gk["blocks_0_attn_wq"]["type"] == "lokr"
assert "blocks_1_mlp_gate" not in gk, "conv lokr must be skipped"
assert "blocks_0_attn_wk" not in gk, "tucker lokr must be skipped"
w2_expect = (sd_lokr["lora_unet_blocks_1_attn_wo.lokr_w2_a"].float()
             @ sd_lokr["lora_unet_blocks_1_attn_wo.lokr_w2_b"].float())
assert torch.allclose(gk["blocks_1_attn_wo"]["w2"], w2_expect)
assert abs(gk["blocks_1_attn_wo"]["alpha_scale"] - 4.0 / 8.0) < 1e-6, \
    "alpha/dim from the factorized side"
print("17) lokr normalization (direct/factorized/skips): ok")

# ---- 18. LoKr end-to-end through the injected layer
from krea2_regional import _inject_lora as _inj
n18 = _inj(patcher, sd_lokr, "lokrA", 1.0)
assert n18 == 2, f"expected 2 lokr layers, got {n18}"
patcher.patch_model(device_to="cpu")
wq = container.diffusion_model.blocks[0].attn.wq
xq = torch.randn(1, 9, 64)
kr.CTX.token_masks = {"lokrA": torch.ones(1, 9, 1)}
with torch.no_grad():
    y_l = wq(xq)
    kr.CTX.clear()
    y_base = wq(xq)
W_l = torch.kron(sd_lokr["lora_unet_blocks_0_attn_wq.lokr_w1"].float(),
                 sd_lokr["lora_unet_blocks_0_attn_wq.lokr_w2"].float())
expected = y_base + xq @ W_l.T
err18 = (y_l - expected).abs().max().item()
assert err18 < 1e-4, f"lokr end-to-end mismatch: {err18}"
patcher.unpatch_model()
print(f"18) lokr end-to-end through injected layer: ok (err {err18:.1e})")

print("\nall tests passed (incl. lokr)")

# ---- 19. Region Lock semantics (unit test with a fake sampler)
from krea2_regional import _make_region_lock

class FakeMS:
    def percent_to_sigma(self, p):
        return 1.0 - p  # linear: percent 0.35 -> sigma 0.65

class FakeModel:
    model_sampling = FakeMS()

m_px = torch.zeros(64, 64); m_px[:, :32] = 1.0  # left half locked
b_lock = {"masks": [m_px], "_rt": {}}
hook = _make_region_lock(b_lock, strength=0.5, start_percent=0.35,
                         end_percent=0.85)

lat0 = torch.randn(1, 16, 8, 8)
def call(sig, lat):
    return hook({"denoised": lat, "sigma": torch.tensor([sig]),
                 "model": FakeModel()})

# early (sigma 0.9 > 0.65): untouched, no snapshot
out = call(0.9, lat0)
assert torch.equal(out, lat0)
# capture step (sigma 0.6 <= 0.65): snapshot taken, output unchanged
snap_src = torch.randn(1, 16, 8, 8)
out = call(0.6, snap_src)
assert torch.equal(out, snap_src)
# inside window (0.15 <= sigma < 0.65): pulled toward snapshot in the mask
drift = snap_src + 1.0
out = call(0.4, drift.clone())
left = out[..., :, :2]
right = out[..., :, 6:]          # outside the mask
assert torch.allclose(left, snap_src[..., :, :2] + 0.5, atol=1e-5), \
    "locked area must move halfway back to the snapshot"
assert torch.allclose(right, drift[..., :, 6:], atol=1e-5), \
    "outside the mask must be untouched"
# released (sigma 0.1 < percent_to_sigma(0.85)=0.15): untouched again
out = call(0.1, drift.clone())
assert torch.equal(out, drift)
# new run (sigma rises): snapshot resets, next capture is fresh
out = call(0.9, lat0)
new_src = torch.randn(1, 16, 8, 8)
call(0.6, new_src)
out = call(0.4, (new_src + 1.0).clone())
assert torch.allclose(out[..., :, :2], new_src[..., :, :2] + 0.5, atol=1e-5), \
    "after reset the NEW snapshot anchors"
print("19) region lock (capture/window/release/reset/mask): ok")

# adaptive-refined masks feed the lock automatically
b_lock2 = {"masks": [m_px],
           "_rt": {"key": (8, 8, 1, (True,), "cpu", 7),
                   "soft": torch.zeros(1, 64)}}
b_lock2["_rt"]["soft"][0, :8] = 1.0  # refined: only the top row of tokens
hook2 = _make_region_lock(b_lock2, 1.0, 0.35, 0.85)
h2 = lambda s, l: hook2({"denoised": l, "sigma": torch.tensor([s]),
                         "model": FakeModel()})
h2(0.6, snap_src)
out2 = h2(0.4, (snap_src + 1.0).clone())
assert torch.allclose(out2[..., 4:, :], snap_src[..., 4:, :] + 1.0, atol=1e-5), \
    "rows outside the refined mask stay free"
top_delta = (out2[..., 0, :] - snap_src[..., 0, :]).abs().mean()
bot_delta = (out2[..., 7, :] - snap_src[..., 7, :]).abs().mean()
assert top_delta < bot_delta - 0.3, \
    "refined top row must be anchored far harder than the free bottom"
print("20) region lock follows adaptive-refined masks: ok")

print("\nall tests passed (incl. region lock)")

# ---- 21. two LoKr characters + a standard lora stack on the same layer
# (regression for issue #1: two regional LoKr characters, injected
# separately through the ModelPatcher, must BOTH apply)
import comfy.model_patcher as _mp21
patcher21 = _mp21.ModelPatcher(container, load_device="cpu",
                               offload_device="cpu")

def _lokr21(seed):
    g21 = torch.Generator().manual_seed(seed)
    return {"lora_unet_blocks_0_attn_wq.lokr_w1":
                torch.randn(4, 4, generator=g21) * 0.2,
            "lora_unet_blocks_0_attn_wq.lokr_w2":
                torch.randn(16, 16, generator=g21) * 0.2}

sd21a, sd21b = _lokr21(11), _lokr21(22)
sd21c = {"transformer.blocks.0.attn.wq.lora_A.weight":
             torch.randn(4, CFG["features"]) * 0.1,
         "transformer.blocks.0.attn.wq.lora_B.weight":
             torch.randn(CFG["features"], 4) * 0.1}
_inj(patcher21, sd21a, "issue1_charA", 1.0)
_inj(patcher21, sd21b, "issue1_charB", 1.0)
_inj(patcher21, sd21c, "issue1_style", 1.0)
pend21 = patcher21.object_patches["diffusion_model.blocks.0.attn.wq"]
assert sorted(pend21.regional_adapters) == \
    ["issue1_charA", "issue1_charB", "issue1_style"], \
    "all three adapters must coexist on the pending patch"
patcher21.patch_model(device_to="cpu")
wq21 = container.diffusion_model.blocks[0].attn.wq
x21 = torch.randn(1, 9, CFG["features"])
kr.CTX.token_masks = {u: torch.ones(1, 9, 1)
                      for u in ("issue1_charA", "issue1_charB",
                                "issue1_style")}
with torch.no_grad():
    y21 = wq21(x21)
    kr.CTX.clear()
    y21b = wq21(x21)
WA21 = torch.kron(sd21a["lora_unet_blocks_0_attn_wq.lokr_w1"],
                  sd21a["lora_unet_blocks_0_attn_wq.lokr_w2"])
WB21 = torch.kron(sd21b["lora_unet_blocks_0_attn_wq.lokr_w1"],
                  sd21b["lora_unet_blocks_0_attn_wq.lokr_w2"])
exp21 = (y21b + x21 @ WA21.T + x21 @ WB21.T
         + (x21 @ sd21c["transformer.blocks.0.attn.wq.lora_A.weight"].T)
         @ sd21c["transformer.blocks.0.attn.wq.lora_B.weight"].T)
err21 = (y21 - exp21).abs().max().item()
assert err21 < 1e-4, f"multi-lokr stacking broken: {err21}"
patcher21.unpatch_model()
print(f"21) two LoKr + standard stacking (issue #1): ok (err {err21:.1e})")
