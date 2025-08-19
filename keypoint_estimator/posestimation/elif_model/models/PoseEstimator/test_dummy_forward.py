import sys, os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, repo_root)
print("PYTHONPATH patched with:", repo_root)
from elif_model.models.PoseEstimator.elif_pose import ElifPose

import torch
import numpy as np
import random
from torchinfo import summary

def print_tensor_info(name, tensor):
    if isinstance(tensor, (list, tuple)):
        print(f"{name}: {[t.shape for t in tensor]}")
    elif isinstance(tensor, dict):
        print(f"{name}: {{ {', '.join([f'{k}: {v.shape}' for k, v in tensor.items()])} }}")
    else:
        print(f"{name}: {tensor.shape}")


def show_model_summary(model, input_shape):
    print("\n=== Model Summary ===")
    summary(model, input_size=input_shape, col_names=["input_size", "output_size", "num_params"], depth=4)

def main():
    # Set seed for reproducibility
    seed = 21
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build Model
    model = ElifPose()
    model.to(device)
    model.eval()

    # Dummy input (batch_size=2, channels=3, height=256, width=192)
    dummy_imgs = torch.randn(2, 3, 256, 192).to(device)

    print("=== Backbone Forward ===")
    backbone_feats = model.backbone(dummy_imgs)
    print_tensor_info("Backbone output", backbone_feats)

    print("\n=== Head Forward ===")
    # If backbone_feats is a tuple/list, pass the first element to the head
    if isinstance(backbone_feats, (list, tuple)):
        head_input = backbone_feats[0]
    else:
        head_input = backbone_feats
    head_out = model.head(head_input)
    print_tensor_info("Head output", head_out)

    print("\n=== Full Model Forward ===")
    try:
        full_out = model(dummy_imgs)
        print_tensor_info("Full model output", full_out)
    except Exception as e:
        print("Full model forward not implemented or failed:", e)

    print("\n=== Simulate Post-processing with Dummy Bounding Boxes ===")
    dummy_bboxes = torch.tensor([[50, 60, 150, 200], [30, 40, 120, 180]], dtype=torch.float).to(device)
    if hasattr(model.head, "decode"):
        try:
            decoded = model.head.decode(head_out, dummy_bboxes)
            print_tensor_info("Decoded keypoints (post-processed)", decoded)
        except Exception as e:
            print("Decode method failed:", e)
    else:
        print("No decode method found in head; skipping post-processing.")

    show_model_summary(model, input_shape=(1, 3, 256, 192))

if __name__ == "__main__":
    main()