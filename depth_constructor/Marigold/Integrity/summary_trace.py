import os, sys, types, torch
import torch
from torchinfo import summary
from torch.fx import symbolic_trace


project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)
print("PYTHONPATH patched to:", project_root)

from marigold import MarigoldDepthPipeline

# Always load on CPU for summary to avoid OOM
device = torch.device("cpu")
dtype = torch.float32

checkpoint = "prs-eth/marigold-depth-v1-1"
pipe = MarigoldDepthPipeline.from_pretrained(checkpoint, torch_dtype=dtype).to(device)

print("\n=== Marigold UNet ===")
pipe.unet.eval()

latent = torch.randn(1, 8, 64, 64)
timestep = torch.tensor([0])
encoder_hidden_states = torch.randn(1, 77, 1024)

info_unet = summary(
    pipe.unet,
    input_data=(latent, timestep, encoder_hidden_states),
    col_names=["input_size", "output_size", "num_params"],
    depth=4,
    verbose=1
)

# from torch.fx import symbolic_trace

# traced_unet = symbolic_trace(pipe.unet)          # model already on CPU & eval
# with torch.no_grad():
#     o_ref = pipe.unet(latent, timestep, encoder_hidden_states).sample
#     o_fx  = traced_unet(latent, timestep, encoder_hidden_states).sample
# print("FX trace – max abs error:", (o_ref - o_fx).abs().max())   # expect < 1e-6

# traced_unet.graph.print_tabular()                # optional: view ops
# traced_unet.save("marigold_unet_fx.pt")          # saved for later QAT / editing

info_vae = info_text = None

if hasattr(pipe, "vae") and pipe.vae:
    print("\n===Marigold VAE ===")
    pipe.vae.eval()
    info_vae = summary(pipe.vae, input_size=(1, 3, 256, 256), depth=3, verbose=1)

# if hasattr(pipe, "text_encoder") and pipe.text_encoder:
#     print("\n===Marigold Text Encoder ===")
#     pipe.text_encoder.eval()
#     dummy_input_ids = torch.randint(low=0, high=49408, size=(1, 77), dtype=torch.long)

#     info_text = summary(
#         pipe.text_encoder,
#         input_data=(dummy_input_ids,),
#         col_names=["input_size", "output_size", "num_params"],
#         depth=2,
#         verbose=1
#     )

# === Combined Totals ===
def safe(x): return x if x is not None else 0

total_params = safe(info_unet.total_params) + safe(info_vae.total_params) #+ safe(info_text.total_params)
trainable_params = safe(info_unet.trainable_params) + safe(info_vae.trainable_params) #+ safe(info_text.trainable_params)
non_trainable_params = total_params - trainable_params
mult_adds = safe(info_unet.total_mult_adds) + safe(info_vae.total_mult_adds) #+ safe(info_text.total_mult_adds)
params_size_mb = total_params * 4 / (1024 ** 2)
estimated_total_mb = params_size_mb + 161.48  # keep this constant (same as earlier)

print("\n\n=== Total Marigold Pipeline Summary ===")
print("-" * 72)
print(f"{'Total params:':<30}{total_params:,}")
print(f"{'Trainable params:':<30}{trainable_params:,}")
print(f"{'Non-trainable params:':<30}{non_trainable_params:,}")
print(f"{'Total mult-adds (M):':<30}{mult_adds:.2f}")
print("=" * 72)
print(f"{'Input size (MB):':<30}{'0.00'}")
print(f"{'Forward/backward pass size (MB):':<30}{'00000 - cannot estimate'}")
print(f"{'Params size (MB):':<30}{params_size_mb:.2f}")
print(f"{'Estimated Total Size (MB):':<30}{estimated_total_mb:.2f}")
print("=" * 72)
#!/usr/bin/env python3
# summary_trace_fx.py
# ---------------------------------------------------------------
# Trace Marigold UNet with torch.fx (safe for later QAT work)
# ---------------------------------------------------------------

#!/usr/bin/env python
# summary_trace_fx.py  (patched – no hidden helper needed)
#!/usr/bin/env python
# layer_inventory_and_patch.py

