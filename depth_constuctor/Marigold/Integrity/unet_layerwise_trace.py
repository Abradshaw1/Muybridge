# import csv, types, torch, torch.fx as fx
# from diffusers import UNet2DConditionModel

# CKPT = "prs-eth/marigold-depth-v1-1"   # --- your model -----

# # ------------------------------------------------------------------
# # 1.  Load the UNet (CPU only)
# # ------------------------------------------------------------------
# unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
#         CKPT, subfolder="unet").cpu().eval()

# # ------------------------------------------------------------------
# # 2.  Dump every *leaf* layer to a CSV  (+ keep python objects)
# # ------------------------------------------------------------------
# leaf_layers = []                # (name, module) tuples
# with open("unet_leaf_inventory.csv", "w", newline="") as fh:
#     wr = csv.writer(fh)
#     wr.writerow(["path", "class", "param_count"])
#     for n, m in unet.named_modules(remove_duplicate=False):
#         if len(list(m.children())) == 0:                  # leaf
#             leaf_layers.append((n, m))
#             wr.writerow([n, m.__class__.__name__,
#                          sum(p.numel() for p in m.parameters())])

# print(f"Wrote layer inventory → unet_leaf_inventory.csv "
#       f"({len(leaf_layers)} leaves)")

# # ------------------------------------------------------------------
# # 3.  Define a *replacement table*  (edit to taste)
# #     key : class to look for
# #     val : lambda taking the old module → returns new module
# # ------------------------------------------------------------------
# import torch.nn as nn

# REPLACEMENTS = {
#     nn.SiLU:   lambda old: nn.ReLU(inplace=True),
#     nn.GroupNorm: lambda old: nn.BatchNorm2d(
#         num_features=old.num_channels, affine=True),
#     # add more rules here ...
# }

# # ------------------------------------------------------------------
# # 4.  Walk the tree and swap in-place
# # ------------------------------------------------------------------
# def patch_unet(model: torch.nn.Module, table):
#     """In-place recursive replacement according to `table`."""
#     replaced = []
#     for name, child in model.named_children():
#         for tgt_cls, builder in table.items():
#             if isinstance(child, tgt_cls):
#                 new_mod = builder(child)
#                 setattr(model, name, new_mod)
#                 replaced.append((name, child.__class__.__name__,
#                                  new_mod.__class__.__name__))
#                 break
#         else:  # no replacement → recurse
#             patch_unet(child, table)
#     return replaced

# changes = patch_unet(unet, REPLACEMENTS)

# print("\nReplacement summary")
# for path, old, new in changes[:20]:      # show first 20
#     print(f"  {path:45s}  {old:10s} → {new}")
# print(f"... total replaced: {len(changes)}")

# # ------------------------------------------------------------------
# # 5.  Sanity check ─ run a forward pass
# # ------------------------------------------------------------------
# x      = torch.randn(1, 8, 64, 64)
# t      = torch.tensor([0])
# enc    = torch.randn(1, 77, 1024)

# with torch.no_grad():
#     y = unet(x, t, enc).sample
# print("\nForward OK  →  output shape:", tuple(y.shape))



# -------------------------------------------------------------
# #  Quant-aware leaf-dump + smart-norm patch  (UNet + VAE)
# # -------------------------------------------------------------
# import csv, torch, torch.nn as nn
# from diffusers import UNet2DConditionModel, AutoencoderKL

# CKPT = "prs-eth/marigold-depth-v1-1"      # ------------ hf repo --------------

# # ---------- helper: dump every leaf to CSV ----------------------------------
# def dump_leaves(model: nn.Module, csv_path: str):
#     with open(csv_path, "w", newline="") as fh:
#         wr = csv.writer(fh)
#         wr.writerow(["path", "class", "param_count"])
#         for n, m in model.named_modules(remove_duplicate=False):
#             if len(list(m.children())) == 0:
#                 wr.writerow([n, m.__class__.__name__,
#                              sum(p.numel() for p in m.parameters())])
#     print(f"Wrote leaf inventory → {csv_path}")

# # ---------- helper: runtime-dim-aware BatchNorm -----------------------------
# class _BNauto(nn.Module):
#     """Acts like BatchNorm2d on 4-D input, BatchNorm1d on 3-D input."""
#     def __init__(self, gn: nn.GroupNorm):
#         super().__init__()
#         C, eps = gn.num_channels, gn.eps
#         self.bn2d = nn.BatchNorm2d(C, eps=eps, affine=True)
#         self.bn1d = nn.BatchNorm1d(C, eps=eps, affine=True)
#     def forward(self, x):
#         return self.bn2d(x) if x.dim() == 4 else self.bn1d(x)

# # ---------- replacement table ------------------------------------------------
# REPLACEMENTS = {
#     nn.SiLU     : lambda _: nn.ReLU(inplace=True),
#     nn.GroupNorm: lambda gn: _BNauto(gn),
# }

# # ---------- helper: recursive patch -----------------------------------------
# def patch(model: nn.Module):
#     replaced = []
#     for name, child in model.named_children():
#         for tgt_cls, builder in REPLACEMENTS.items():
#             if isinstance(child, tgt_cls):
#                 new_mod = builder(child)
#                 setattr(model, name, new_mod)
#                 replaced.append((name, child.__class__.__name__,
#                                  new_mod.__class__.__name__))
#                 break
#         else:                          # nothing matched → descend
#             replaced.extend(patch(child))
#     return replaced

# # ============================================================================ #
# #                               ---  UNet  ---                                 #
# # ============================================================================ #
# unet = UNet2DConditionModel.from_pretrained(CKPT, subfolder="unet").cpu().eval()
# dump_leaves(unet, "unet_leaf_inventory.csv")
# rep = patch(unet)
# print(f"UNet: replaced {len(rep)} modules\n")

# # quick smoke-test
# x   = torch.randn(1, 8, 64, 64)
# t   = torch.tensor([0])
# enc = torch.randn(1, 77, 1024)
# with torch.no_grad():
#     y = unet(x, t, enc).sample
# print("UNet forward OK – out shape:", tuple(y.shape))

# # ============================================================================ #
# #                                ---  VAE  ---                                 #
# # ============================================================================ #
# vae = AutoencoderKL.from_pretrained(CKPT, subfolder="vae").cpu().eval()
# dump_leaves(vae, "vae_leaf_inventory.csv")
# rep = patch(vae)
# print(f"VAE:  replaced {len(rep)} modules\n")

# # VAE smoke-test
# rgb = torch.randn(1, 3, 64, 64)
# with torch.no_grad():
#     z = vae.encode(rgb).latent_dist.sample()
#     recon = vae.decode(z).sample
# print("VAE encode/decode OK – recon shape:", tuple(recon.shape))


# -------------------------------------------------------------
#  Quant-aware inspection / patching script for Marigold U-Net
# -------------------------------------------------------------
# ✓ Lists every leaf layer → CSV
# ✓ Swaps SiLU + GroupNorm variants for QAT-friendly layers
# ✓ Runs a quick forward sanity pass
# -------------------------------------------------------------


import csv, torch, torch.nn as nn
from diffusers import UNet2DConditionModel
import diffusers.models.normalization as dm_norm      # Diffusers GroupNorm

CKPT = "prs-eth/marigold-depth-v1-1"                  # ← repo

# ------------------------------------------------------------------
# 1.  Load U-Net (CPU, eval)
# ------------------------------------------------------------------
unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
        CKPT, subfolder="unet").cpu().eval()

# ------------------------------------------------------------------
# 2.  “Smart” BN = handles 3-D *and* 4-D tensors (no weight cloning)
# ------------------------------------------------------------------
class _SmartBN(nn.Module):
    def __init__(self, gn: nn.GroupNorm):
        super().__init__()
        C = gn.num_channels
        self.bn2d = nn.BatchNorm2d(C, eps=gn.eps, affine=True)
        self.bn1d = nn.BatchNorm1d(C, eps=gn.eps, affine=True)

    def forward(self, x):
        return self.bn2d(x) if x.dim() == 4 else self.bn1d(x)

# ------------------------------------------------------------------
# 3.  Dump every leaf → CSV
# ------------------------------------------------------------------
with open("unet_leaf_inventory.csv", "w", newline="") as fh:
    wr = csv.writer(fh);  wr.writerow(["path", "class", "params"])
    for n, m in unet.named_modules(remove_duplicate=False):
        if len(list(m.children())) == 0:
            wr.writerow([n, m.__class__.__name__,
                         sum(p.numel() for p in m.parameters())])
print("Wrote leaf inventory → unet_leaf_inventory.csv")

# ------------------------------------------------------------------
# 4.  Replacement table
# ------------------------------------------------------------------
REPLACEMENTS = {
    nn.SiLU           : lambda _: nn.ReLU(inplace=True),
    nn.GroupNorm      : lambda old: _SmartBN(old),
    # dm_norm.GroupNorm : lambda old: _SmartBN(old),   # Diffusers wrapper
}

# ------------------------------------------------------------------
# 5.  Walk the tree & swap
# ------------------------------------------------------------------
def patch(mod: nn.Module):
    replaced = []
    for name, child in mod.named_children():
        for cls, builder in REPLACEMENTS.items():
            if isinstance(child, cls):
                new_child = builder(child)
                setattr(mod, name, new_child)
                replaced.append((name, child.__class__.__name__,
                                 new_child.__class__.__name__))
                break
        else:                        # descend if no hit
            replaced.extend(patch(child))
    return replaced

changes = patch(unet)
print(f"Replaced {len(changes)} layers (SiLU / GroupNorm variants)")

# ------------------------------------------------------------------
# 6.  Forward sanity pass
# ------------------------------------------------------------------
x   = torch.randn(1, 8, 64, 64)      # latent + rgb concat → 8 channels
t   = torch.tensor([0])              # dummy timestep
enc = torch.randn(1, 77, 1024)       # text embed

with torch.no_grad():
    y = unet(x, t, enc).sample
print("Forward OK → output shape:", tuple(y.shape))




