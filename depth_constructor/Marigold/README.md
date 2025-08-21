# Marigold Computer Vision

This is a pruned and quantized implementation of Marigold, a diffusion-based training and evaluation routine developed by collaborators at ETH Zurich

### 📦 Repository

Clone the repository (requires git):

```bash
git clone https://github.com/prs-eth/Marigold.git
cd Marigold
```

### 💻 Dependencies

Install the dependencies:

```bash
python -m venv venv/marigold
source venv/marigold/bin/activate
pip install -r requirements.txt
```

Keep the environment activated before running the inference script. 
Activate the environment again after restarting the terminal session.

## 🏃 Testing on your images

### 📷 Prepare images

Use selected images from our paper:

```bash
bash script/download_sample_data.sh
```

Or place your images in a directory, for example, under `input/in-the-wild_example`, and run the following inference command.

### 🚀 Run inference (for practical usage)

```bash
# Depth
python script/depth/run.py \
    --checkpoint prs-eth/marigold-depth-v1-1 \
    --input_rgb_dir input/in-the-wild_example \
    --output_dir output/in-the-wild_example \
    --fp16
```

### ⚙️ Inference settings

The default settings are optimized for the best results. However, the behavior of the code can be customized:

- `--half_precision` or `--fp16`: Run with half-precision (16-bit float) to have faster speed and reduced VRAM usage, but might lead to suboptimal results.

- `--ensemble_size`: Number of inference passes in the ensemble. Larger values tend to give better results in evaluations at the cost of slower inference; for most cases 1 is enough. Default: 1.

- `--denoise_steps`: Number of denoising diffusion steps. Default settings are defined in the model checkpoints and are sufficient for most cases.

- By default, the inference script resizes input images to the *processing resolution*, and then resizes the prediction back to the original resolution. This gives the best quality, as Stable Diffusion, from which Marigold is derived, performs best at 768x768 resolution.  
  
  - `--processing_res`: the processing resolution; set as 0 to process the input resolution directly. When unassigned (`None`), will read default setting from model config. Default: `None`.
  - `--output_processing_res`: produce output at the processing resolution instead of upsampling it to the input resolution. Default: False.
  - `--resample_method`: the resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic`, or `nearest`. Default: `bilinear`.

- `--seed`: Random seed can be set to ensure additional reproducibility. Default: None (unseeded). Note: forcing `--batch_size 1` helps to increase reproducibility. To ensure full reproducibility, [deterministic mode](https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms) needs to be used.
- `--batch_size`: Batch size of repeated inference. Default: 0 (best value determined automatically).
- `--color_map`: [Colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html) used to colorize the depth prediction. Default: Spectral. Set to `None` to skip colored depth map generation.
- `--apple_silicon`: Use Apple Silicon MPS acceleration.


### 🎮 Run inference (for academic comparisons)

These settings correspond to our paper. For academic comparison, please run with the settings below (if you only want to do fast inference on your own images, you can set `--ensemble_size 1`).

```bash
# Depth
python script/depth/run.py \
    --checkpoint prs-eth/marigold-depth-v1-1 \
    --denoise_steps 1 \
    --ensemble_size 10 \
    --input_rgb_dir input/in-the-wild_example \
    --output_dir output/in-the-wild_example
```

```bash
# Depth (the original CVPR version)
python script/depth/run.py \
    --checkpoint prs-eth/marigold-depth-v1-0 \
    --denoise_steps 50 \
    --ensemble_size 10 \
    --input_rgb_dir input/in-the-wild_example \
    --output_dir output/in-the-wild_example
```

You can find all results in the `output` directory. Enjoy!


### ⬇ Checkpoint cache

By default, the checkpoint ([depth](https://huggingface.co/prs-eth/marigold-depth-v1-1), [normals](https://huggingface.co/prs-eth/marigold-normals-v1-1), [iid](https://huggingface.co/prs-eth/marigold-iid-appearance-v1-1))  is stored in the Hugging Face cache.
The `HF_HOME` environment variable defines its location and can be overridden, e.g.:

```bash
export HF_HOME=$(pwd)/cache
```

Alternatively, use the following script to download the checkpoint weights locally:

```bash
bash script/download_weights.sh marigold-depth-v1-1           # depth checkpoint
# bash script/download_weights.sh marigold-depth-v1-0         # CVPR depth checkpoint
```

At inference, specify the checkpoint path:

```bash
# Depth
python script/depth/run.py \
    --checkpoint checkpoint/marigold-depth-v1-1 \
    --denoise_steps 4 \
    --ensemble_size 1 \
    --input_rgb_dir input/in-the-wild_example \
    --output_dir output/in-the-wild_example
```

## 🦿 Evaluation on test datasets <a name="evaluation"></a>
Install additional dependencies:

```bash
pip install -r requirements+.txt -r requirements.txt
``` 

Set data directory variable (also needed in evaluation scripts) and download the evaluation datasets ([depth](https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset), [normals](https://share.phys.ethz.ch/~pf/bingkedata/marigold/marigold_normals/evaluation_dataset)) into the corresponding subfolders:

```bash
export BASE_DATA_DIR=<YOUR_DATA_DIR>  # Set target data directory

# Depth
wget -r -np -nH --cut-dirs=4 -R "index.html*" -P ${BASE_DATA_DIR} https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/
```

Run inference and evaluation scripts, for example:

```bash
# Depth
bash script/depth/eval/11_infer_nyu.sh  # Run inference
bash script/depth/eval/12_eval_nyu.sh   # Evaluate predictions
```

```bash
# Depth (the original CVPR version)
bash script/depth/eval_old/11_infer_nyu.sh  # Run inference
bash script/depth/eval_old/12_eval_nyu.sh   # Evaluate predictions
```

Note: although the seed has been set, the results might still be slightly different on different hardware.

## 🏋️ Training

Based on the previously created environment, install extended requirements:

```bash
pip install -r requirements++.txt -r requirements+.txt -r requirements.txt
```

Set environment parameters for the data directory:

```bash
export BASE_DATA_DIR=YOUR_DATA_DIR        # directory of training data
export BASE_CKPT_DIR=YOUR_CHECKPOINT_DIR  # directory of pretrained checkpoint
```

Download Stable Diffusion v2 [checkpoint](https://huggingface.co/stabilityai/stable-diffusion-2) into `${BASE_CKPT_DIR}`

### Prepare for training data
**Depth**

Prepare for [Hypersim](https://github.com/apple/ml-hypersim) and [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) datasets and save into `${BASE_DATA_DIR}`. Please refer to [this README](script/depth/dataset_preprocess/hypersim/README.md) for Hypersim preprocessing.


### Run training script

```bash
# Depth
python script/depth/train.py --config config/train_marigold_depth.yaml
```

Resume from a checkpoint, e.g.:

```bash
# Depth
python script/depth/train.py --resume_run output/marigold_base/checkpoint/latest
```

### Compose checkpoint:
Only the U-Net and scheduler config are updated during training. They are saved in the training directory. To use the inference pipeline with your training result:
- replace `unet` folder in Marigold checkpoints with that in the `checkpoint` output folder.
- replace the `scheduler/scheduler_config.json` file in Marigold checkpoints with `checkpoint/scheduler_config.json` generated during training.
Then refer to [this section](#evaluation) for evaluation.

**Note**: Although random seeds have been set, the training result might be slightly different on different hardwares. It's recommended to train without interruption.

## 🎓 Citation

Please cite our papers:

```bibtex
@InProceedings{ke2023repurposing,
  title={Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation},
  author={Bingxin Ke and Anton Obukhov and Shengyu Huang and Nando Metzger and Rodrigo Caye Daudt and Konrad Schindler},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}

@misc{ke2025marigold,
  title={Marigold: Affordable Adaptation of Diffusion-Based Image Generators for Image Analysis},
  author={Bingxin Ke and Kevin Qu and Tianfu Wang and Nando Metzger and Shengyu Huang and Bo Li and Anton Obukhov and Konrad Schindler},
  year={2025},
  eprint={2505.09358},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## 🎫 License

This code of this work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

The models are licensed under RAIL++-M License (as defined in the [LICENSE-MODEL](LICENSE-MODEL.txt))

By downloading and using the code and model you agree to the terms in [LICENSE](LICENSE.txt) and [LICENSE-MODEL](LICENSE-MODEL.txt) respectively.
