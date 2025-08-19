"""
Quantization-Aware Training (QAT) Script for ElifPose
====================================================

This script performs quantization-aware training for the ElifPose model using the COCO WholeBody and Halpe datasets.
It loads or initializes a float model, prepares it for QAT, trains with quantization observers, and benchmarks both float and quantized models.

Requirements:
- Dataset folders and annotation files as specified below.
- Correct Python environment with all dependencies installed.
- Environment variable DATASETS_ROOT can be set to override the default datasets path.
- Model fusion logic (if needed) should be implemented in the model definition.

Usage:
    python QAT_train.py

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
from elif_model.training.helper import evaluate, print_size_of_model, train_one_epoch, load_model, run_benchmark

num_keypoints = 26
input_size = (192, 256)

# --- Runtime/training parameters ---
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
DATASETS_ROOT = os.environ.get('DATASETS_ROOT', os.path.abspath(os.path.join(PROJECT_PARENT, '..', '..', '..', 'datasets', 'abradshaw'))) # Change to your datasets root
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

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model creation ---
ElifPoseModel = ElifPose()

# --- Optimizer and scheduler ---
optimizer = optim.AdamW(ElifPoseModel.parameters(), lr=base_lr, weight_decay=0.0)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=10)

# --- Model directory and float model file ---
saved_model_dir = '/datasets/basokure/elif_model/saved_models/'
float_model_file = 'float_model.pth'

# --- Load the float model if available, otherwise instantiate a fresh model ---
float_model_path = os.path.join(saved_model_dir, float_model_file)
if os.path.exists(float_model_path):
    print(f"Loading existing float model from {float_model_path}")
    qat_model = load_model(ElifPose(), float_model_path)
    qat_model = qat_model.to(device)
else:
    print(f"Float model not found at {float_model_path}, instantiating new model.")
    qat_model = ElifPose()
    qat_model = qat_model.to(device)

# --- Model fusion for QAT (optional, see model definition for implementation) ---
# TODO: Add fusion logic for QAT (e.g., qat_model.fuse_model(is_qat=True))
# For now, skipping fusion to enable script execution
# qat_model.fuse_model(is_qat=True)

# --- Evaluate the float model before QAT ---
float_model_acc = evaluate(qat_model, val_dataloader, device)
print("Float model accuracy:", float_model_acc)

# --- Prepare model for Quantization-Aware Training (QAT) ---
torch.ao.quantization.prepare_qat(qat_model, inplace=True)

# --- QAT Training loop ---
best_acc = 0
for epoch in range(max_epochs):
    qat_model = train_one_epoch(qat_model, train_dataloader, optimizer, scheduler, epoch, device, optim_wrapper, len(train_dataset))
    if epoch > 40:
        # Freeze quantizer parameters
        qat_model.apply(torch.ao.quantization.disable_observer)
    if epoch > 45:
        # Freeze batch norm mean and variance estimates
        qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    
    quantized_model = torch.ao.quantization.convert(qat_model.eval(), inplace=False)
    qat_model_acc = evaluate(quantized_model, device, val_dataloader)
    print(f"Epoch: {epoch} Accuracy: {qat_model_acc}")
    if qat_model_acc > best_acc:
        best_acc = qat_model_acc
        torch.save(qat_model.state_dict(), saved_model_dir + 'qat_model.pth')
        print("Model saved")

# --- Benchmark both float and QAT models ---
float_model_time = run_benchmark(saved_model_dir + float_model_file, val_dataloader, 200)
qat_model_time = run_benchmark(saved_model_dir + 'qat_model.pth', val_dataloader, 200)

print("Float model inference time:", float_model_time)
print("QAT model inference time:", qat_model_time)

# --- Print the size of the models ---
print_size_of_model(qat_model, 'Float model')
print_size_of_model(quantized_model, 'QAT model')