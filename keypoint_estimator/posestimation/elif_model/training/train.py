"""
ElifPose Training Script
=======================

This script trains the ElifPose model for human pose estimation using the COCO WholeBody and Halpe datasets.
It sets up data loaders, builds the model, runs the training loop, performs validation, and saves checkpoints.

Requirements:
- Dataset folders and annotation files as specified below.
- Correct Python environment with all dependencies installed.
- Environment variable DATASETS_ROOT can be set to override the default datasets path.

Usage:
    python train.py

Author: ETH Zurich Digital Circuits and Systems Group
Date: 2025-06-30
"""

import sys
import os
import numpy as np
import random

# --- Path setup ---
# Path to the directory containing 'elif_model'
PROJECT_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Path to the 'elif_model' directory itself
PROJECT_ROOT = os.path.join(PROJECT_PARENT, 'elif_model')

if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("Project parent:", PROJECT_PARENT)
print("Project root:", PROJECT_ROOT)

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.dataloader import DataLoader

from elif_model.datasets.CocoWholeBodyDataset import CocoWholeBodyDataSet
from elif_model.datasets.data_augmentation.formatting import CustomCollatePoseEstimation
from elif_model.models.PoseEstimator.elif_pose import ElifPose
from elif_model.training.helper import evaluate, print_size_of_model, train_one_epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_keypoints = 26
input_size = (192, 256)

# --- Training configuration ---
max_epochs = 50
base_lr = 4e-3
train_batch_size = 512
val_batch_size = 64

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)

# --- Set seed for reproducibility ---
torch.manual_seed(randomness['seed'])
torch.cuda.manual_seed(randomness['seed'])
np.random.seed(randomness['seed'])
random.seed(randomness['seed'])

# --- Optimizer configuration (AdamW with optional grad clipping) ---
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.0),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# --- Learning rate scheduler configuration ---
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
]

# --- Keypoint mapping for dataset compatibility ---
coco_halpe26 = [(i, i) for i in range(17)] + [(17, 20), (18, 22), (19, 24),
                                              (20, 21), (21, 23), (22, 25)]

halpe_halpe26 = [(i, i) for i in range(26)]

# --- Dataset root setup ---
# Use DATASETS_ROOT env variable if set, else default to ../datasets/abradshaw
DATASETS_ROOT = os.environ.get(
    'DATASETS_ROOT',
    os.path.abspath(os.path.join(PROJECT_PARENT, '..', '..', '..', 'datasets', 'abradshaw')) # Change to your datasets root
)

print("Datasets root:", DATASETS_ROOT)

# --- Annotation file paths ---
coco_ann_file_train = os.path.join(DATASETS_ROOT, 'COCO', 'annotations', 'coco_wholebody_train_v1.0.json')
halpe_ann_file_train = os.path.join(DATASETS_ROOT, 'Halpe', 'annotations', 'halpe_train_v1.json')
coco_ann_file_val = os.path.join(DATASETS_ROOT, 'COCO', 'annotations', 'coco_wholebody_val_v1.0.json')
halpe_ann_file_val = os.path.join(DATASETS_ROOT, 'Halpe', 'annotations', 'halpe_val_v1.json')

# --- Dataset parameters ---
input_size = (192, 256)
codec_cfg = dict(
    input_size=input_size,
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# --- Training dataset setup ---
COCO_train = CocoWholeBodyDataSet(ann_file=coco_ann_file_train, 
                                    codec_cfg=codec_cfg, 
                                    input_size=input_size, 
                                    data_root=os.path.join(DATASETS_ROOT, 'COCO', 'train2017'),
                                    keypoint_mapping=coco_halpe26,
                                    num_keypoints=num_keypoints,
                                    dataset_type='coco_train',
                                    meta_cfg_file=os.path.join(PROJECT_ROOT, 'datasets', 'coco_wholebody.py'))

Halpe_train = CocoWholeBodyDataSet(ann_file=halpe_ann_file_train, 
                                     codec_cfg=codec_cfg, 
                                     input_size=input_size,
                                     data_root=os.path.join(DATASETS_ROOT, 'Halpe', 'hico_20160224_det', 'images', 'train2015'),
                                     keypoint_mapping=halpe_halpe26,
                                     num_keypoints=num_keypoints,
                                     dataset_type='halpe_train',
                                     meta_cfg_file=os.path.join(PROJECT_ROOT, 'datasets', 'halpe26.py'))

train_dataset = torch.utils.data.ConcatDataset([COCO_train, Halpe_train])
#dataset = Halpe_Dataset
train_dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=CustomCollatePoseEstimation, shuffle=True)

# --- Validation dataset setup ---
COCO_val = CocoWholeBodyDataSet(ann_file=coco_ann_file_val, 
                                    codec_cfg=codec_cfg, 
                                    input_size=input_size, 
                                    data_root=os.path.join(DATASETS_ROOT, 'COCO', 'val2017'),
                                    keypoint_mapping=coco_halpe26,
                                    num_keypoints=num_keypoints,
                                    dataset_type='coco_val',
                                    meta_cfg_file=os.path.join(PROJECT_ROOT, 'datasets', 'coco_wholebody.py'))

Halpe_val = CocoWholeBodyDataSet(ann_file=halpe_ann_file_val, 
                                     codec_cfg=codec_cfg, 
                                     input_size=input_size,
                                     data_root=os.path.join(DATASETS_ROOT, 'Halpe', 'hico_20160224_det', 'images', 'test2015'),
                                     keypoint_mapping=halpe_halpe26,
                                     num_keypoints=num_keypoints,
                                     dataset_type='halpe_val',
                                     meta_cfg_file=os.path.join(PROJECT_ROOT, 'datasets', 'halpe26.py'))

val_dataset = torch.utils.data.ConcatDataset([COCO_val, Halpe_val])
val_dataloader = DataLoader(train_dataset, batch_size=val_batch_size, collate_fn=CustomCollatePoseEstimation, shuffle=True)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Build Model
ElifPoseModel = ElifPose()
ElifPoseModel.to(device)

# --- Optimizer and scheduler ---
optimizer = optim.AdamW(ElifPoseModel.parameters(), lr=base_lr, weight_decay=0.0)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=10)

# --- Training loop ---
best_acc = 0
for epoch in range(max_epochs):
    print(f"\nEpoch {epoch+1}/{max_epochs}")
    train_one_epoch(ElifPoseModel, train_dataloader, optimizer, scheduler, epoch, device, optim_wrapper, len(train_dataset))

    # Save checkpoint after each epoch
    torch.save(ElifPoseModel.state_dict(), f"elif_model/training/elif_pose_model.pth")
    print(f"Model saved for epoch {epoch}")
    
    # --- Validation ---
    acc = evaluate(ElifPoseModel, val_dataloader, device, len(val_dataset))

    # Save best model
    if np.mean(acc.avg) > best_acc:
        best_acc = np.mean(acc.avg)
        torch.save(ElifPoseModel.state_dict(), f"elif_model/training/bests.pth")
        print(f"New best model saved at epoch {epoch}")
