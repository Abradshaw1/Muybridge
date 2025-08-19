# ElifPose Pruning Experiments

This directory contains experiments for pruning the ElifPose pose estimation model to reduce model size and computational requirements while maintaining accuracy. The pruning leverages the Group Fisher method from the [MMRazor](https://github.com/open-mmlab/mmrazor) library.

## Directory Structure

```
elif_pose_pruning/
├── work_dir/                  # Results and logs from pruning runs
│   ├── flops_0.51.pth         # Weights of the pruned model
│   ├── best_AUC_epoch_60.pth  # Best checkpoint by AUC
│   ├── epoch_60.pth           # Checkpoint after 60 epochs
│   ├── fix_subnet.json        # Subnetwork structure after pruning
│   ├── last_checkpoint        # Last checkpoint marker
│   ├── group_fisher_finetune_rtmpose-s_8xb256-420e_coco-256x192.py # Finetune config (copy)
│   ├── group_fisher_prune_rtmpose-s_8xb256-420e_coco-256x192.py    # Prune config (copy)
│   └── ...                    # Additional logs, folders, and configs
├── rtmpose-t_8xb1024-700e_body8-halpe26-256x192.py # Model config for pruning
├── rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.pth # Original weights
├── group_fisher_prune_rtmpose-s_8xb256-420e_coco-256x192.py    # Main pruning config
├── group_fisher_finetune_rtmpose-s_8xb256-420e_coco-256x192.py # Main finetuning config
```

## Folder and File Descriptions

### 1. `work_dir/`
- Stores all **outputs and logs** from pruning and finetuning runs.
- `flops_0.51.pth`: Weights of the pruned model achieving ~51% FLOPs of the original.
- `best_AUC_epoch_60.pth`: Model checkpoint with best AUC during finetuning.
- `epoch_60.pth`: Final checkpoint after 60 epochs of finetuning.
- `fix_subnet.json`: JSON file describing the pruned subnetwork structure.
- `last_checkpoint`: Marker for the last checkpoint saved.
- `group_fisher_finetune_rtmpose-s_8xb256-420e_coco-256x192.py`: Copy of the finetuning config used for reproducibility.
- `group_fisher_prune_rtmpose-s_8xb256-420e_coco-256x192.py`: Copy of the pruning config used for reproducibility.
- Subfolders (e.g., `20240627_011734/`): Additional logs, tensorboard files, or run-specific outputs.

### 2. Top-level Configuration and Weight Files
- `rtmpose-t_8xb1024-700e_body8-halpe26-256x192.py`: Model configuration file for the pose estimator used during pruning.
- `rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.pth`: Original model weights (pre-pruning).
- `group_fisher_prune_rtmpose-s_8xb256-420e_coco-256x192.py`: Main configuration for running Group Fisher pruning.
- `group_fisher_finetune_rtmpose-s_8xb256-420e_coco-256x192.py`: Main configuration for finetuning the pruned model.

## Pruning Methodology

Pruning is performed using the **Group Fisher** method, as implemented in the [MMRazor](https://github.com/open-mmlab/mmrazor) library. This approach enables structured pruning of neural network layers to optimize for FLOPs and parameter count, with minimal loss in accuracy. For details, see the MMRazor documentation.

## How to Run Pruning

1. **Install MMRazor** and its dependencies (see their [installation guide](https://github.com/open-mmlab/mmrazor)).
2. **Configure your experiment** using the provided `.py` config files for both pruning and finetuning.
3. **Run the pruning script** as described in the MMRazor docs, e.g.:
   ```bash
   python tools/prune.py <group_fisher_prune_config.py>
   ```
4. **Finetune the pruned model** (optional) using the corresponding config.

All results, logs, and pruned model weights will be saved in `work_dir/`.

## Acknowledgments

This work builds upon the open-source tools provided by the [MMRazor](https://github.com/open-mmlab/mmrazor) library for model pruning.