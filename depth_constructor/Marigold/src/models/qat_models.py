import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

###########################################################
# Small, self-contained UNet and VAE designed for QAT
# ---------------------------------------------------------
# • No external diffusers / transformers dependencies.
# • Time-step conditioning uses a simple sinusoidal + MLP embed.
# • Cross-attention replaced with identity – we rely on dummy
#   text embeddings, so semantic coupling is not required.
# • All sub-modules are pure torch.nn.* layers -> compatible with
#   torch.ao.quantization QAT / PTQ flows.
###########################################################

__all__ = [
    "SinusoidalTimestepEmbed",
    "ResBlock",
    "DownSample",
    "UpSample",
    "QAT_UNet",
    "QAT_VAE",
    "prepare_qat",
]

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _conv(in_ch, out_ch, k=3, s=1, p=1):
    """3x3 Conv + weight init"""
    layer = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
    nn.init.kaiming_normal_(layer.weight, a=0, mode="fan_out", nonlinearity="relu")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class SinusoidalTimestepEmbed(nn.Module):
    """Embed scalar timesteps to channel-dim vector using sinusoid + 2-layer MLP."""

    def __init__(self, dim: int = 1280):
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim * 4)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(dim * 4, dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half - 1)
        emb = torch.exp(torch.arange(half, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1))
        # project
        emb = self.lin2(self.act(self.lin1(emb)))
        return emb


class ResBlock(nn.Module):
    """Standard ResNet block with GN + SiLU"""

    def __init__(self, in_ch: int, out_ch: int, emb_ch: int):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = _conv(in_ch, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = _conv(out_ch, out_ch)
        self.emb_proj = nn.Linear(emb_ch, out_ch)
        if in_ch != out_ch:
            self.shortcut = _conv(in_ch, out_ch, k=1, p=0)
        else:
            self.shortcut = nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        # Add timestep embedding (broadcast spatially)
        h = h + self.emb_proj(self.act(emb))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.shortcut(x)


class DownSample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = _conv(ch, ch, k=3, s=2, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = _conv(ch, ch, k=3, s=1, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# -------------------------------------------------------------------------
# Quantization-friendly UNet (U-Shaped) – roughly aligned with Marigold sizes
# -------------------------------------------------------------------------

class QAT_UNet(nn.Module):
    """Simplified UNet that keeps channel sizes of Marigold UNet but replaces
    cross-attention with identity.
    Inputs:
        x               – image/latent tensor      [B, in_ch, H, W]
        t               – diffusion timestep       [B]
        cond (ignored)  – dummy text embed         [B, 77, 1024]
    Output:  [B, out_ch, H, W]
    """

    def __init__(
        self,
        in_ch: int = 8,
        base_ch: int = 320,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        out_ch: int = 4,
        emb_ch: int = 1280,
    ):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.t_embed = SinusoidalTimestepEmbed(emb_ch)

        # Input conv
        self.conv_in = _conv(in_ch, base_ch)

        # Down path
        self.downs = nn.ModuleList()
        ch = base_ch
        for level, mult in enumerate(ch_mult):
            out_ch_level = base_ch * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock(ch, out_ch_level, emb_ch))
                ch = out_ch_level
            if level != len(ch_mult) - 1:
                self.downs.append(DownSample(ch))

        # Middle
        self.mid1 = ResBlock(ch, ch, emb_ch)
        self.mid2 = ResBlock(ch, ch, emb_ch)

        # Up path
        self.ups = nn.ModuleList()
        for level, mult in reversed(list(enumerate(ch_mult))):
            out_ch_level = base_ch * mult
            for _ in range(num_res_blocks + 1):  # +1 to merge skip
                self.ups.append(ResBlock(ch + out_ch_level, out_ch_level, emb_ch))
                ch = out_ch_level
            if level != 0:
                self.ups.append(UpSample(ch))

        # Output
        self.norm_out = nn.GroupNorm(32, ch)
        self.act = nn.SiLU()
        self.conv_out = _conv(ch, out_ch, k=3, p=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.quant(x)
        emb = self.t_embed(timesteps)

        hs: List[torch.Tensor] = []
        h = self.conv_in(x)
        for module in self.downs:
            if isinstance(module, ResBlock):
                h = module(h, emb)
                hs.append(h)
            else:  # DownSample
                h = module(h)
                hs.append(h)

        # Middle
        h = self.mid1(h, emb)
        h = self.mid2(h, emb)

        # Up
        for module in self.ups:
            if isinstance(module, ResBlock):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = module(h, emb)
            else:  # UpSample
                h = module(h)

        h = self.conv_out(self.act(self.norm_out(h)))
        return self.dequant(h)


# -------------------------------------------------------------------------
# Very small VAE (encoder/decoder) – compatible with Marigold latent scale
# -------------------------------------------------------------------------

class QAT_VAE(nn.Module):
    """Lightweight VAE that encodes 3-ch RGB -> latent (n_ch=4) and decodes back.
    Designed for 256-512px inputs, mirrors Stable-Diffusion VAE but simpler.
    """

    def __init__(self, z_channels: int = 4, ch: int = 128):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        # Encoder
        self.enc_conv1 = _conv(3, ch, k=3, p=1)
        self.enc_conv2 = _conv(ch, ch * 2, k=3, p=1)
        self.down1 = DownSample(ch * 2)
        self.enc_conv3 = _conv(ch * 2, ch * 4, k=3, p=1)
        self.down2 = DownSample(ch * 4)
        self.enc_conv4 = _conv(ch * 4, ch * 8, k=3, p=1)
        self.down3 = DownSample(ch * 8)
        self.avg_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc_mu = nn.Linear(ch * 8 * 8 * 8, z_channels * 8 * 8)
        self.fc_logvar = nn.Linear(ch * 8 * 8 * 8, z_channels * 8 * 8)

        # Decoder
        self.dec_fc = nn.Linear(z_channels * 8 * 8, ch * 8 * 8 * 8)
        self.up1 = UpSample(ch * 8)
        self.dec_conv1 = _conv(ch * 8, ch * 4)
        self.up2 = UpSample(ch * 4)
        self.dec_conv2 = _conv(ch * 4, ch * 2)
        self.up3 = UpSample(ch * 2)
        self.dec_conv3 = _conv(ch * 2, ch)
        self.final_conv = nn.Conv2d(ch, 3, kernel_size=1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.silu(self.enc_conv1(x))
        h = F.silu(self.enc_conv2(h))
        h = self.down1(h)
        h = F.silu(self.enc_conv3(h))
        h = self.down2(h)
        h = F.silu(self.enc_conv4(h))
        h = self.down3(h)
        h = self.avg_pool(h)
        h = torch.flatten(h, 1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z)
        h = h.view(z.size(0), -1, 8, 8)  # [B, ch*8, 8, 8]
        h = self.up1(h)
        h = F.silu(self.dec_conv1(h))
        h = self.up2(h)
        h = F.silu(self.dec_conv2(h))
        h = self.up3(h)
        h = F.silu(self.dec_conv3(h))
        return torch.tanh(self.final_conv(h))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.quant(x)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        recon = self.dequant(recon)
        return recon, mu, logvar


# -------------------------------------------------------------------------
# QAT utilities
# -------------------------------------------------------------------------

def prepare_qat(module: nn.Module, backend: str = "fbgemm") -> nn.Module:
    """Apply default QAT config and insert observers."""
    module.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    torch.ao.quantization.prepare_qat(module, inplace=True)
    return module
