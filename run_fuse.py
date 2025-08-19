# run_fuse.py
import os, sys, glob, argparse, numpy as np

repo_root = os.path.dirname(os.path.abspath(__file__))

# make local packages importable (single, centralized shim)
marigold_src = os.path.join(repo_root, "depth_constructor", "Marigold")
elif_src     = os.path.join(repo_root, "keypoint_estimator", "posestimation")

for p in (repo_root, marigold_src, elif_src):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")

from tools.pose_wrapper import run_pose
from tools.depth_wrapper import run_marigold_depth_api
from tools.fuse_utils import (
    clip_xy, bilinear_sample, geometric_median,
    backproject_points, project_point,
    plot_overlay_depth_centers, plot_hist_depth, plot_points3d
)

def process_one(rgb_path, args):
    base = os.path.splitext(os.path.basename(rgb_path))[0]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n=== [PROCESS] {base} ===")
    print("[FUSION] START")
    # ---------- Pose
    pose_png = os.path.join(args.out_dir, f"{base}_pose.png")
    kps, scores, meta, _ = run_pose(
        image_path=rgb_path,
        ckpt_path=args.pose_ckpt,
        device=None,
        save_pose_vis_to=pose_png
    )

    # ---------- Depth (Marigold API)
    depth_npy, depth_color_png = run_marigold_depth_api(
        input_rgb_path=rgb_path,
        output_dir=os.path.abspath(args.out_dir),      # write directly to fusion_outputs
        checkpoint=args.marigold_checkpoint,
        fp16=args.fp16,
        denoise_steps=args.denoise_steps,
        processing_res=args.processing_res,
        ensemble_size=args.ensemble_size,
        match_input_res=(not args.output_processing_res),
        resample_method=args.resample_method,
        color_map="Spectral",
        seed=args.seed,
        batch_size=args.batch_size
    )
    Z = np.load(depth_npy).astype(np.float32)  # (H,W)
    H, W = Z.shape

    # ---------- Fuse & centers
    pts2d = clip_xy(np.round(kps).astype(np.int32), W, H)
    kz = np.array([bilinear_sample(Z, float(x), float(y)) for x,y in kps], dtype=np.float32)
    weights = scores / max(np.sum(scores), 1e-6)

    centroid_2d = (weights[:,None] * kps).sum(axis=0)
    geommed_2d  = geometric_median(kps)

    z_centroid_2d = float(bilinear_sample(Z, centroid_2d[0], centroid_2d[1]))
    z_geommed_2d  = float(bilinear_sample(Z, geommed_2d[0],  geommed_2d[1]))

    fx = fy = float(max(H, W))
    cx, cy = (W - 1)/2.0, (H - 1)/2.0
    P3 = backproject_points(kps[:,0], kps[:,1], kz, fx, fy, cx, cy)
    centroid_3d = (weights[:,None] * P3).sum(axis=0)
    geommed_3d  = geometric_median(P3)

    centroid_3d_uv = project_point(centroid_3d, fx, fy, cx, cy)
    geommed_3d_uv  = project_point(geommed_3d,  fx, fy, cx, cy)

    # ---------- Save overlay on depth
    overlay_png = os.path.join(args.out_dir, f"{base}_overlay_centers.png")
    plot_overlay_depth_centers(
        Z, pts2d,
        centers2d=(centroid_2d, geommed_2d),
        centers3d_uv=(centroid_3d_uv, geommed_3d_uv),
        save_path=overlay_png
    )

    # ---------- Save 3D scatter
    scatter3d_png = os.path.join(args.out_dir, f"{base}_points3d.png")
    plot_points3d(P3, centers3d=(centroid_3d, geommed_3d), save_path=scatter3d_png)

    # ---------- Save histogram
    hist_png = os.path.join(args.out_dir, f"{base}_depth_hist.png")

    # stats used by the plot
    kp_mean   = float(np.mean(kz))
    kp_median = float(np.median(kz))
    z_centroid_3d = float(centroid_3d[2])
    z_geommed_3d  = float(geommed_3d[2])

    plot_hist_depth(
        kz,
        save_path=hist_png,
        z_centroid_2d=z_centroid_2d,
        z_geommed_2d=z_geommed_2d,
        z_centroid_3d=z_centroid_3d,
        z_geommed_3d=z_geommed_3d,
    )

    # ---------- Save NPZ summary
    npz_path = os.path.join(args.out_dir, f"{base}_fusion.npz")
    np.savez_compressed(
        npz_path,
        keypoints=kps, scores=scores, depth_at_k=kz, depth_shape=(H,W),
        centroid_2d=centroid_2d, geommed_2d=geommed_2d,
        z_centroid_2d=z_centroid_2d, z_geommed_2d=z_geommed_2d,
        centroid_3d=centroid_3d, geommed_3d=geommed_3d,
        centroid_3d_uv=centroid_3d_uv, geommed_3d_uv=geommed_3d_uv,
        fx=fx, fy=fy, cx=cx, cy=cy,
        kp_mean=kp_mean, kp_median=kp_median,
        z_centroid_3d=z_centroid_3d, z_geommed_3d=z_geommed_3d,
    )

    print(f"[FUSION] DONE  overlay='{overlay_png}', points3d='{scatter3d_png}', hist='{hist_png}', npz='{npz_path}'")
    print(f"=== [COMPLETE] {base} ===\n")

    return {
        "pose_png": pose_png,                              # (1) pose viz
        "depth_color_png": depth_color_png,                # (2) colored depth (saved in fusion_outputs/)
        "overlay_centers_png": overlay_png,                # (3)
        "points3d_png": scatter3d_png,                     # (4)
        "depth_hist_png": hist_png,                        # (5)
        "fusion_npz": npz_path,
        "depth_npy": depth_npy
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="fusion_inputs", help="Folder with input images")
    ap.add_argument("--out_dir", default="fusion_outputs", help="Where to write outputs")
    ap.add_argument("--pose-ckpt", required=True, help="ElifPose checkpoint (.pth)")
    ap.add_argument("--marigold-checkpoint", default="prs-eth/marigold-depth-v1-1",
                    help="HF repo id or local folder with model_index.json")
    ap.add_argument("--fp16", action="store_true", help="Use fp16 for depth")

    # Optional knobs to mirror Marigold CLI
    ap.add_argument("--denoise_steps", type=int, default=None)
    ap.add_argument("--processing_res", type=int, default=None)
    ap.add_argument("--ensemble_size", type=int, default=1)
    ap.add_argument("--output_processing_res", action="store_true")
    ap.add_argument("--resample_method", choices=["bilinear","bicubic","nearest"], default="bilinear")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=0)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    img_list = sorted(
        p for p in glob.glob(os.path.join(args.input_dir, "*"))
        if p.lower().endswith((".png",".jpg",".jpeg"))
    )
    if not img_list:
        raise RuntimeError(f"No images found in {args.input_dir}")

    print(f"[Muybridge] Found {len(img_list)} images in {args.input_dir}")
    for p in img_list:
        print(f"[Muybridge] Processing: {os.path.basename(p)}")
        files = process_one(p, args)
        for k,v in files.items():
            print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()






