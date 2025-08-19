# ElifPose Model Training and Validation

This directory contains the complete pipeline for building, training, and validating the ElifPose pose estimation model. ElifPose is designed for robust human pose estimation and muscle-tendon length (MTU) analysis, and is used as the backbone for the ElifPose iOS app. The codebase is inspired by and partially ported from [MMPose](https://github.com/open-mmlab/mmpose), but is reimplemented in pure PyTorch for transparency, flexibility, and research extensibility.

## Directory Overview

```
elif_model/
├── codecs/      # Keypoint codecs (SimCC and related logic)
├── datasets/    # Data loaders and augmentation for COCO, Halpe26, etc.
├── evaluation/  # Loss functions and model evaluation metrics
├── models/      # All model components: backbone, neck, head, and the main ElifPose model
├── structures/  # Custom data structures for input/output
├── training/    # Training scripts and configs
├── utils/       # Utility functions (general purpose)
├── __init__.py  # Package marker
├── ReadME.md    # This documentation
```

- **codecs/**: Implements SimCC and related logic for converting model outputs into keypoint predictions.
- **datasets/**: Contains loaders and augmentation pipelines for COCO, Halpe26, and other datasets. Supports multi-dataset training.
- **evaluation/**: Functions for calculating loss and evaluating model predictions using standard metrics.
- **models/**: Contains all core model architectures and layers. The ElifPose model is defined in `models/PoseEstimator/elif_pose.py` and is constructed from the following key modules:
  - **PoseEstimator/**: Main folder for ElifPose model definitions.
    - `elif_pose.py`: Defines the `ElifPose` model class, which integrates the backbone, head, and (optionally) neck modules for pose estimation.
    - `test_dummy_forward.py`: Utility/testing script for model forward passes.
  - **backbones/CSPNeXt.py**: Implements the `CSPNeXt` backbone, a high-performance convolutional neural network for feature extraction, as used in RTMDet and ElifPose.
  - **necks/cspnext_pafpn.py**: (Optional) Implements the `CSPNeXtPAFPN` neck module, a path aggregation feature pyramid network for multi-scale feature fusion. Not always used in the default ElifPose config, but available for experiments.
  - **heads/rtmcc_head.py**: Implements the `RTMCCHead`, a top-down head module for keypoint prediction, featuring a large-kernel convolution, MLP, and Gated Attention Unit (GAU), as used in RTMPose and ElifPose.

  These modules are modular and can be extended or replaced for research. The default ElifPose model uses the CSPNeXt backbone and RTMCCHead, optionally with the CSPNeXtPAFPN neck for advanced feature aggregation.
- **structures/**: Custom classes for handling model inputs, outputs, and intermediate representations.
- **training/**: Scripts and configurations for model training. See `training/` for usage details.
- **utils/**: Utility functions used across the codebase (e.g., logging, checkpointing, etc).

## Getting Started

1. **Install dependencies** (see main readme, posest env should be activated and cover eveerything).
3. **Train the model**:
   - Navigate to `training/` and run either hte qunaitzed or non qunaitzed training script, e.g.:
     ```bash
     python training/train.py
     python training/QAT_train.py 
     ```
   - configs are set manually in the training scripts
   - Validation happens every user defined epoch during training.
   
## Acknowledgments

This project builds on [MMPose](https://github.com/open-mmlab/mmpose) and incorporates custom modifications for flexibility and research use. If you use this codebase in your research, please cite the original MMPose repository and any relevant publications.
