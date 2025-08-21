import os, time
from typing import Optional
import numpy as np
from PIL import Image
import torch
from marigold import MarigoldDepthPipeline, MarigoldDepthOutput

_PIPE: MarigoldDepthPipeline | None = None  # cache

def _load_pipe(checkpoint: str, fp16: bool) -> MarigoldDepthPipeline:
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    print(f"[DEPTH] Initializing pipeline (ckpt='{checkpoint}', fp16={fp16})")
    dtype   = torch.float16 if fp16 else torch.float32
    variant = "fp16" if fp16 else None
    local   = os.path.isdir(checkpoint)

    pipe = MarigoldDepthPipeline.from_pretrained(
        checkpoint, variant=variant, torch_dtype=dtype, local_files_only=local
    )
    try: pipe.enable_xformers_memory_efficient_attention()
    except Exception: pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _PIPE = pipe.to(device)
    print(f"[DEPTH] Pipeline ready on {device}")
    return _PIPE


def run_marigold_depth_api(
    input_rgb_path: str,
    output_dir: str,
    checkpoint: str = "prs-eth/marigold-depth-v1-1",
    fp16: bool = True,
    *,
    denoise_steps: Optional[int] = None,
    processing_res: Optional[int] = None,
    ensemble_size: int = 1,
    match_input_res: bool = True,
    resample_method: str = "bilinear",
    color_map: Optional[str] = "Spectral",
    seed: Optional[int] = None,
    batch_size: int = 0
) -> tuple[str, Optional[str]]:
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_rgb_path))[0]
    npy_dir = os.path.join(output_dir, "depth_npy")
    os.makedirs(npy_dir, exist_ok=True)
    depth_npy_path = os.path.join(npy_dir, f"{base}_depth.npy")
    depth_color_png_path = os.path.join(output_dir, f"{base}_depth_color.png") if color_map else None

    img = Image.open(input_rgb_path).convert("RGB")

    generator = None
    if seed is not None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=dev).manual_seed(seed)

    pipe = _load_pipe(checkpoint, fp16=fp16)

    print(f"[DEPTH] START  file='{os.path.basename(input_rgb_path)}'")
    with torch.no_grad():
        out: MarigoldDepthOutput = pipe(
            img,
            denoising_steps=denoise_steps,
            ensemble_size=ensemble_size,
            processing_res=processing_res,
            match_input_res=match_input_res,
            batch_size=batch_size,
            color_map=color_map,
            show_progress_bar=False,
            resample_method=resample_method,
            generator=generator,
        )
    depth = out.depth_np.astype(np.float32)
    np.save(depth_npy_path, depth)
    if color_map and out.depth_colored is not None:
        out.depth_colored.save(depth_color_png_path)
    print(f"[DEPTH] DONE   file='{os.path.basename(input_rgb_path)}' -> npy='{depth_npy_path}'"
          + (f", color='{depth_color_png_path}'" if depth_color_png_path else ""), flush=True)
    return depth_npy_path, depth_color_png_path




