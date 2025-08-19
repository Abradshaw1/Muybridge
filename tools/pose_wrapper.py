# scripts/pose_wrapper.py
import os, cv2, numpy as np, torch
from matplotlib.patches import Circle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

# Safe torch.load patch for legacy checkpoints
import numpy.core.multiarray, functools
if not hasattr(torch.load, "_is_patched"):
    _orig = torch.load
    @functools.wraps(_orig)
    def patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _orig(*args, **kwargs)
    patched_torch_load._is_patched = True
    torch.load = patched_torch_load
torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])

# Repo-local imports (run from repo root)
from keypoint_estimator.posestimation.elif_model.models.PoseEstimator.elif_pose import ElifPose
from keypoint_estimator.posestimation.elif_model.codecs.simcc_label import SimCCLabel
from keypoint_estimator.posestimation.elif_model.datasets.data_augmentation.formatting import PackPoseInputs

def run_pose(image_path: str, ckpt_path: str, device=None, save_pose_vis_to: str=None):
    print(f"[KEYPOINTS] START file='{os.path.basename(image_path)}', ckpt='{os.path.basename(ckpt_path)}'")

HALPE26_NAME = {
  0:"neck", 1:"head_top_mid", 2:"head_left", 3:"head_right",
  4:"l_shoulder_tip", 5:"r_shoulder_tip",
  6:"l_upperarm_root", 7:"r_upperarm_root",
  8:"l_wrist", 9:"r_wrist", 10:"l_elbow",
  11:"r_hip", 12:"l_hip",
  13:"r_knee", 14:"l_knee",
  15:"r_ankle", 16:"l_ankle",
  17:"head_top", 18:"r_clavicle", 19:"pelvis",
  20:"r_heel", 21:"l_small_toe", 22:"r_big_toe",
  23:"l_big_toe", 24:"r_small_toe", 25:"l_heel",
}

def _affine_to_input(img, input_size=(192,256)):
    H_img, W_img = img.shape[:2]
    src_center = np.array([W_img/2, H_img/2], dtype=np.float32)
    src_size   = np.array([W_img, H_img], dtype=np.float32)
    dst_size   = np.array(input_size, dtype=np.float32)
    src_pts = np.array([src_center,
                        src_center + [0, -0.5*src_size[1]],
                        src_center + [0.5*src_size[0], 0]], np.float32)
    dst_center = dst_size/2
    dst_pts = np.array([dst_center,
                        dst_center + [0, -0.5*dst_size[1]],
                        dst_center + [0.5*dst_size[0], 0]], np.float32)
    M = cv2.getAffineTransform(src_pts, dst_pts)
    M_inv = cv2.invertAffineTransform(M)
    img_resized = cv2.warpAffine(img, M, tuple(input_size), flags=cv2.INTER_LINEAR)
    return img_resized, M, M_inv

def run_pose(image_path: str, ckpt_path: str, device=None, save_pose_vis_to: str=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    input_size = (192,256) # (W,H)
    img_resized, M, M_inv = _affine_to_input(img, input_size=input_size)

    packed = {'img': img_resized, 'img_shape': img_resized.shape, 'input_size': input_size}
    packed = PackPoseInputs(packed)
    input_tensor = packed['inputs'].unsqueeze(0).float().to(device)

    model = ElifPose()
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    codec_cfg = dict(input_size=input_size, sigma=(4.9,5.66),
                     simcc_split_ratio=2.0, normalize=False, use_dark=False)
    decoder = SimCCLabel(**codec_cfg)

    with torch.no_grad():
        simcc_x, simcc_y = model(input_tensor)
        kps, scores = decoder.decode(simcc_x.cpu().numpy(), simcc_y.cpu().numpy())

    kps_2d = kps[0].astype(np.float32).reshape(-1,1,2)
    kps_orig = cv2.transform(kps_2d, M_inv).reshape(-1,2)
    scores_vec = scores[0] if scores is not None else np.full((kps_orig.shape[0],), np.nan, np.float32)

    if save_pose_vis_to is not None:
        plt.figure(figsize=(6,6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for (x,y) in kps_orig:
            plt.gca().add_patch(Circle((x,y), 3, color='red'))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_pose_vis_to, dpi=180, bbox_inches='tight', pad_inches=0.0)
        plt.close()
        print(f"[KEYPOINTS] VIZ   '{save_pose_vis_to}'")
    print(f"[KEYPOINTS] DONE")
    H_img, W_img = img.shape[:2]
    meta = dict(W=W_img, H=H_img, M=M, M_inv=M_inv)
    return kps_orig.astype(np.float32), scores_vec.astype(np.float32), meta, save_pose_vis_to
