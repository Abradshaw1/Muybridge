import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, propagate_qconfig_
try:
    # diffusers >= 0.24
    from diffusers.models.attention import CrossAttention
except Exception:
    CrossAttention = None  # fallback to name matching

from torch.ao.quantization.fake_quantize import FakeQuantizeBase


# ---- swaps (adapt from your notebook) ---------------------------------
_BAD2GOOD = {
    nn.GELU: lambda _: nn.ReLU(inplace=False),
    nn.SiLU: lambda _: nn.Hardswish(),
}
_BAD_FUNCS = {F.gelu, F.silu}

def _relu(x): return F.relu(x, inplace=False)


def patch_model_for_qat(module: nn.Module):
    """Recursively replace unsupported modules and functional activations."""
    # 1) swap child modules
    for name, child in list(module.named_children()):
        for bad_cls, make_good in _BAD2GOOD.items():
            if isinstance(child, bad_cls):
                setattr(module, name, make_good(child))
                child = getattr(module, name)  # updated ref
                break
        patch_model_for_qat(child)

    # 2) swap stored functional refs
    for attr_name, attr_val in vars(module).items():
        if callable(attr_val) and attr_val in _BAD_FUNCS:
            setattr(module, attr_name, _relu)


def _is_attention(name: str, mod: nn.Module) -> bool:
    if CrossAttention is not None and isinstance(mod, CrossAttention):
        return True
    # name-based fallback
    attn_keywords = ("attn", "to_q", "to_k", "to_v")
    return any(k in name for k in attn_keywords)

def mask_attention_from_qat(root: nn.Module):
    """Set qconfig=None for every attention (and its linears) so prepare_qat skips them."""
    for n, m in root.named_modules():
        if _is_attention(n, m):
            m.qconfig = None


def make_unet_qat_ready(unet: nn.Module, engine: str = "fbgemm") -> nn.Module:
    """
    Sanitize + prepare the given UNet for QAT and return a deep-copied model
    with observers/fake-quant inserted.
    """
    torch.backends.quantized.engine = engine
    patch_model_for_qat(unet)    
    unet.qconfig = get_default_qat_qconfig(engine)
    propagate_qconfig_(unet)
    mask_attention_from_qat(unet)
    qat_unet = copy.deepcopy(unet)
    qat_unet.train()
    prepare_qat(qat_unet, inplace=True)
    return qat_unet





# Probe
import logging
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

def quick_qat_probe(unet):
    def has_act_obs(m):
        return hasattr(m, "activation_post_process") and isinstance(m.activation_post_process, FakeQuantizeBase)

    def has_wfq(m):
        return hasattr(m, "weight_fake_quant") and isinstance(m.weight_fake_quant, FakeQuantizeBase)

    n_obs = 0
    n_wfq = 0
    attn_qat = []

    for name, m in unet.named_modules():
        if has_act_obs(m): n_obs += 1
        if has_wfq(m):     n_wfq += 1
        if "attn" in name.lower():
            if has_act_obs(m) or has_wfq(m) or getattr(m, "qconfig", None) is not None:
                attn_qat.append(name)

    logging.info(
        "[QAT] engine=%s | modules_with_act_observer=%d | modules_with_weight_fake_quant=%d | attn_qat=%d",
        torch.backends.quantized.engine, n_obs, n_wfq, len(attn_qat)
    )
    if attn_qat:
        logging.warning("[QAT] These attention blocks got quantized (they shouldn't): %s",
                        attn_qat[:10])

    # Hard stop while debugging (optional)
    assert len(attn_qat) == 0, "Attention was quantized – fix your masking."



def _qat_signature(self):
    """Return (n_act_observers, n_weight_fq) for the current UNet."""
    n_act, n_wfq = 0, 0
    for m in self.model.unet.modules():
        ap = getattr(m, "activation_post_process", None)
        if isinstance(ap, FakeQuantizeBase):
            n_act += 1
        wfq = getattr(m, "weight_fake_quant", None)
        if isinstance(wfq, FakeQuantizeBase):
            n_wfq += 1
    return n_act, n_wfq

def _log_qat_status(self, where: str):
    if not self.qat_enabled:
        logging.info(f"[{where}] qat_enabled=False (skipping QAT sanity log). unet_id={id(self.model.unet)}")
        return
    n_act, n_wfq = _qat_signature(self)
    # first time we see it, remember it
    if not hasattr(self, "_qat_sig_ref"):
        self._qat_sig_ref = (n_act, n_wfq)
        logging.info(f"[{where}] QAT sig captured: act_obs={n_act}, weight_fq={n_wfq}, "
                     f"unet_id={id(self.model.unet)} training={self.model.unet.training}")
        return

    same = (n_act, n_wfq) == self._qat_sig_ref
    level = logging.INFO if same else logging.WARNING
    logging.log(level,
        f"[{where}] QAT sig now: act_obs={n_act}, weight_fq={n_wfq}, "
        f"unet_id={id(self.model.unet)} training={self.model.unet.training} "
        f"(ref={self._qat_sig_ref})")




# import copy
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.ao.quantization import prepare_qat, QConfig
# from torch.ao.quantization.observer import (
#     MovingAverageMinMaxObserver,
#     PerChannelMinMaxObserver,
# )
# from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize

# # ---- swaps (only activations, keep norms!) ---------------------------------
# _BAD2GOOD = {
#     nn.GELU: lambda _: nn.ReLU(inplace=False),
#     nn.SiLU: lambda _: nn.Hardswish(),   # safer than ReLU for SiLU
# }
# _BAD_FUNCS = {F.gelu, F.silu}

# def _relu(x):  # kept for legacy, not really used now
#     return F.relu(x, inplace=False)

# def patch_model_for_qat(module: nn.Module):
#     """Recursively replace only unsupported *activations*."""
#     for name, child in list(module.named_children()):
#         for bad_cls, make_good in _BAD2GOOD.items():
#             if isinstance(child, bad_cls):
#                 setattr(module, name, make_good(child))
#                 child = getattr(module, name)  # updated ref
#                 break
#         patch_model_for_qat(child)

#     # (optional) swap stored functional refs
#     for attr_name, attr_val in vars(module).items():
#         if callable(attr_val) and attr_val in _BAD_FUNCS:
#             setattr(module, attr_name, _relu)


# def make_unet_qat_ready(unet: nn.Module, engine: str = "fbgemm") -> nn.Module:
#     """
#     Sanitize + prepare the given UNet for QAT and return a deep-copied model
#     with observers/fake-quant inserted. Norms & attention stay FP32.
#     """
#     torch.backends.quantized.engine = engine

#     # ---- 1) swap only activations we want quantized
#     patch_model_for_qat(unet)

#     # ---- 2) define a safer qconfig (prevents zero_point crash)
#     SAFE_QCONFIG = QConfig(
#         activation=FusedMovingAvgObsFakeQuantize.with_args(
#             observer=MovingAverageMinMaxObserver,
#             dtype=torch.quint8,
#             qscheme=torch.per_tensor_affine,
#             quant_min=0, quant_max=255, reduce_range=False,
#         ),
#         weight=FusedMovingAvgObsFakeQuantize.with_args(
#             observer=PerChannelMinMaxObserver,
#             dtype=torch.qint8,
#             qscheme=torch.per_channel_symmetric,
#             quant_min=-127, quant_max=127, reduce_range=False,
#             ch_axis=0,
#         ),
#     )

#     FP32_KEEP = (nn.GroupNorm, nn.LayerNorm, nn.Softmax)

#     def _set_qconfig_recursively(mod: nn.Module, default_qconfig: QConfig):
#         mod.qconfig = default_qconfig
#         for n, ch in mod.named_children():
#             # keep norms & attention blocks in fp32
#             if isinstance(ch, FP32_KEEP) or "attn" in n or "attention" in n:
#                 ch.qconfig = None
#             else:
#                 _set_qconfig_recursively(ch, default_qconfig)

#     _set_qconfig_recursively(unet, SAFE_QCONFIG)

#     # ---- 3) deepcopy + prepare
#     qat_unet = copy.deepcopy(unet)
#     qat_unet.train()
#     prepare_qat(qat_unet, inplace=True)
#     return qat_unet

