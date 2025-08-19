# # -------------------------------------------------------------
# #  Quant-aware inspection / patching script for Marigold VAE
# # -------------------------------------------------------------
# # ✓ Lists every leaf layer → CSV
# # ✓ Optionally swaps selected layers for QAT-friendly variants
# # ✓ Runs a quick encode-decode sanity pass
# # -------------------------------------------------------------
# import csv, torch, torch.nn as nn, torch.fx as fx      # fx not used, but handy
# from diffusers import AutoencoderKL

# CKPT = "prs-eth/marigold-depth-v1-1"        # <<< Same model repo

# # ------------------------------------------------------------------
# # 1.  Load the VAE (CPU only)
# # ------------------------------------------------------------------
# vae: AutoencoderKL = AutoencoderKL.from_pretrained(
#         CKPT, subfolder="vae").cpu().eval()

# # ------------------------------------------------------------------
# # 2.  Dump every *leaf* layer to a CSV  (+ keep python objects)
# # ------------------------------------------------------------------
# leaf_layers = []                 # (name, module) tuples
# with open("vae_leaf_inventory.csv", "w", newline="") as fh:
#     wr = csv.writer(fh)
#     wr.writerow(["path", "class", "param_count"])
#     for n, m in vae.named_modules(remove_duplicate=False):
#         if len(list(m.children())) == 0:                  # leaf
#             leaf_layers.append((n, m))
#             wr.writerow([n,
#                          m.__class__.__name__,
#                          sum(p.numel() for p in m.parameters())])
# print(f"Wrote layer inventory → vae_leaf_inventory.csv "
#       f"({len(leaf_layers)} leaves)")

# # ------------------------------------------------------------------
# # 3.  Define a *replacement table*  (edit to taste)
# #     key : class to look for
# #     val : lambda taking the old module → returns new module
# # ------------------------------------------------------------------
# REPLACEMENTS = {
#     nn.SiLU:      lambda old: nn.ReLU(inplace=True),
#     # nn.GroupNorm: lambda old: nn.BatchNorm2d(
#     #     num_features=old.num_channels, affine=True),
#     # add more rules here if you like …
# }

# # ------------------------------------------------------------------
# # 4.  Walk the tree and swap in-place
# # ------------------------------------------------------------------
# def patch_module(model: torch.nn.Module, table):
#     """Recursively replace sub-modules according to `table`."""
#     replaced = []
#     for name, child in model.named_children():
#         for tgt_cls, builder in table.items():
#             if isinstance(child, tgt_cls):
#                 new_mod = builder(child)
#                 setattr(model, name, new_mod)
#                 replaced.append((name,
#                                  child.__class__.__name__,
#                                  new_mod.__class__.__name__))
#                 break
#         else:                                  # no replacement → recurse
#             patch_module(child, table)
#     return replaced

# changes = patch_module(vae, REPLACEMENTS)

# print("\nReplacement summary")
# for path, old, new in changes[:20]:            # show first 20
#     print(f"  {path:45s}  {old:12s} → {new}")
# print(f"... total replaced: {len(changes)}")

# # ------------------------------------------------------------------
# # 5.  Sanity check ─ run a tiny encode → decode cycle
# # ------------------------------------------------------------------
# x = torch.randn(1, 3, 64, 64)                  # fake RGB
# with torch.no_grad():
#     # forward() returns a DiagonalGaussian; sample() gives latent
#     latent_dist = vae.encode(x).latent_dist
#     z = latent_dist.sample()                   # [1, 4, 8, 8]  (0.18215-scaled)
#     recon = vae.decode(z).sample               # back to RGB
# print("\nEncode/Decode OK  →  recon shape:", tuple(recon.shape))




# -------------------------------------------------------------
#  Quant-aware inspection / patching script for Marigold VAE
# -------------------------------------------------------------
# ✓ lists every leaf layer → CSV
# ✓ swaps SiLU → ReLU and GroupNorm → smart BatchNorm
# ✓ performs an encode-decode smoke test
# -------------------------------------------------------------
import csv, torch, torch.nn as nn
from diffusers import AutoencoderKL

CKPT = "prs-eth/marigold-depth-v1-1"    # HuggingFace repo

# ------------------------------------------------------------------
# 1.  Load VAE (CPU)
# ------------------------------------------------------------------
vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        CKPT, subfolder="vae").cpu().eval()

# ------------------------------------------------------------------
# 2.  Dump every *leaf* layer to CSV
# ------------------------------------------------------------------
with open("vae_leaf_inventory.csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["path", "class", "param_count"])
    for n, m in vae.named_modules(remove_duplicate=False):
        if len(list(m.children())) == 0:            # leaf
            wr.writerow([n, m.__class__.__name__,
                         sum(p.numel() for p in m.parameters())])
print("Wrote leaf inventory → vae_leaf_inventory.csv")

# ------------------------------------------------------------------
# 3.  Smart norm replacement helpers
# ------------------------------------------------------------------
class _BNauto(nn.Module):
    """BatchNorm2d for 4-D input, BatchNorm1d for 3-D input."""
    def __init__(self, gn: nn.GroupNorm):
        super().__init__()
        C, eps = gn.num_channels, gn.eps
        self.bn2d = nn.BatchNorm2d(C, eps=eps, affine=True)
        self.bn1d = nn.BatchNorm1d(C, eps=eps, affine=True)
    def forward(self, x):
        return self.bn2d(x) if x.dim() == 4 else self.bn1d(x)

REPLACEMENTS = {
    nn.SiLU     : lambda _: nn.ReLU(inplace=True),
    nn.GroupNorm: lambda gn: _BNauto(gn),
}

# ------------------------------------------------------------------
# 4.  Recursive patch-in-place
# ------------------------------------------------------------------
def patch(model: nn.Module):
    swapped = []
    for name, child in model.named_children():
        for tgt_cls, builder in REPLACEMENTS.items():
            if isinstance(child, tgt_cls):
                setattr(model, name, builder(child))
                swapped.append((name, child.__class__.__name__,
                                builder(child).__class__.__name__))
                break
        else:
            swapped.extend(patch(child))
    return swapped

changes = patch(vae)
print(f"Replaced {len(changes)} layers (SiLU / GroupNorm)")

# ------------------------------------------------------------------
# 5.  Sanity check – tiny encode→decode
# ------------------------------------------------------------------
x_rgb = torch.randn(1, 3, 64, 64)              # dummy RGB
with torch.no_grad():
    z = vae.encode(x_rgb).latent_dist.sample()  # latent
    recon = vae.decode(z).sample                # back to RGB
print("Encode/Decode OK  →  recon shape:", tuple(recon.shape))




