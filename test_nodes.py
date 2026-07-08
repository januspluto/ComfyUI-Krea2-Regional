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
def probing_wrapper(executor, x, ts, ctx, am=None, transformer_options={}, **kw):
    ov = transformer_options["optimized_attention_override"]
    q1 = torch.zeros(1, 4, txtlen, 8)  # batch-1 refiner call, like the enhancer
    ov(fake_func, q1, q1, q1, 4)
    return executor(x, ts, ctx, am, transformer_options=transformer_options, **kw)
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
