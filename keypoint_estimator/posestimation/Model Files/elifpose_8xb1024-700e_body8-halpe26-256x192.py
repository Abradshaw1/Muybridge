"""
ElifPose Training Configuration
==============================

This configuration file defines all parameters for training ElifPose (RTMPose-based) on COCO and Halpe datasets with SimCC representation.

- Assumes datasets are located at DATASETS_ROOT (set via environment variable or defaults to ../../datasets/abradshaw). # Change to your datasets root
- Designed for top-down, multi-dataset pose estimation with 26 keypoints (Body8 + Halpe26).
- Includes all data pipelines, augmentation, model, optimizer, and evaluation settings for reproducible training.

Author: ETH Zurich Digital Circuits and Systems Group
Date: 2025-06-30
"""

import os

# ---- Dataset Root Path ----
# Set DATASETS_ROOT from environment or use default (edit as needed for your setup)
DATASETS_ROOT = os.environ.get(
    'DATASETS_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'datasets', 'abradshaw'))  # Change to your datasets root
)
print("Using DATASETS_ROOT:", DATASETS_ROOT)
# ---- KEYPOINT MAPPINGS ----
# These define how keypoints from various datasets are mapped to the unified Halpe26 format.
aic_halpe26 = [
    (0, 6), (1, 8), (2, 10), (3, 5), (4, 7), (5, 9), (6, 12), (7, 14),
    (8, 16), (9, 11), (10, 13), (11, 15), (12, 17), (13, 18)
]
auto_scale_lr = dict(base_batch_size=1024)
backend_args = dict(backend='local')
base_lr = 0.004
coco_halpe26 = [
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
    (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16),
    (17, 20), (18, 22), (19, 24), (20, 21), (21, 23), (22, 25)
]
codec = dict(
    input_size=(
        192,
        256,
    ),
    normalize=False,
    sigma=(
        4.9,
        5.66,
    ),
    simcc_split_ratio=2.0,
    type='SimCCLabel',
    use_dark=False)
crowdpose_halpe26 = [
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11), (7, 12),
    (8, 13), (9, 14), (10, 15), (11, 16), (12, 17), (13, 18)
]
custom_hooks = [
    dict(
        switch_epoch=670,  # Epoch at which to switch training pipeline
        switch_pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),  # Load images from disk
            dict(type='GetBBoxCenterScale'),  # Compute bbox center and scale
            dict(direction='horizontal', type='RandomFlip'),  # Random horizontal flip
            dict(type='RandomHalfBody'),  # Half-body augmentation
            dict(
                rotate_factor=80,
                scale_factor=[0.6, 1.4],
                shift_factor=0.0,
                type='RandomBBoxTransform'),  # Random bbox rotation/scale/shift
            dict(input_size=(192, 256), type='TopdownAffine'),  # Affine transform to input size
            dict(
                transforms=[
                    dict(p=0.1, type='Blur'),  # Random blur
                    dict(p=0.1, type='MedianBlur'),  # Random median blur
                    dict(
                        max_height=0.4, max_holes=1, max_width=0.4,
                        min_height=0.2, min_holes=1, min_width=0.2,
                        p=0.5, type='CoarseDropout'),  # Random dropout
                ],
                type='Albumentation'),  # Use Albumentations library for augmentation
            dict(
                encoder=dict(
                    input_size=(192, 256),
                    normalize=False,
                    sigma=(4.9, 5.66),
                    simcc_split_ratio=2.0,
                    type='SimCCLabel',
                    use_dark=False),
                type='GenerateTarget',
                use_dataset_keypoint_weights=True),  # Generate SimCC targets
            dict(type='PackPoseInputs'),  # Pack inputs for training
        ],
        type='mmdet.PipelineSwitchHook'),  # Custom pipeline switch hook
]

data_mode = 'topdown'
data_root = ''
# ---- DATASETS ----
# Define dataset dictionaries for COCO and Halpe, including annotation paths, image roots, and keypoint mapping.
dataset_coco = dict(
    ann_file=os.path.join(DATASETS_ROOT, 'COCO', 'annotations', 'coco_wholebody_train_v1.0.json'),  # COCO WholeBody annotation file
    data_mode='topdown',
    data_prefix=dict(img=os.path.join(DATASETS_ROOT, 'COCO', 'images', 'train2017')),  # Image root
    data_root='',
    pipeline=[
        dict(
            mapping=[
                (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9),
                (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16),
                (17, 20), (18, 22), (19, 24), (20, 21), (21, 23), (22, 25)
            ],
            num_keypoints=26,  # Unified Halpe26 keypoints
            type='KeypointConverter'),  # Maps COCO to Halpe26 format
    ],
    type='CocoWholeBodyDataset')
# ---- HALPE DATASET DEFINITION ----
dataset_halpe = dict(
    ann_file=os.path.join(DATASETS_ROOT, 'Halpe', 'annotations', 'halpe_train_v1.json'),  # Halpe annotation file
    data_mode='topdown',
    data_prefix=dict(
        img=os.path.join(DATASETS_ROOT, 'Halpe', 'hico_20160224_det', 'images', 'train2015')),  # Image root for Halpe
    data_root='',
    pipeline=[
        dict(
            mapping=[
                (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9),
                (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17),
                (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)
            ],
            num_keypoints=26,  # Unified Halpe26 keypoints
            type='KeypointConverter'),  # Maps Halpe to Halpe26 format (identity)
    ],
    type='HalpeDataset')

dataset_type = 'CocoWholeBodyDataset'  # Default dataset type for training

# ---- DEFAULT HOOKS AND TRAINING SETTINGS ----
default_hooks = dict(
    badcase=dict(
        _scope_='mmpose',
        badcase_thr=5,       
        enable=False,        
        metric_type='loss',
        out_dir='badcase',   
        type='BadCaseAnalysisHook'),
    checkpoint=dict(
        _scope_='mmpose',
        interval=10,         
        max_keep_ckpts=2,    
        rule='greater',
        save_best='AUC',     
        type='CheckpointHook'),
    logger=dict(_scope_='mmpose', interval=50, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmpose', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmpose', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmpose', type='IterTimerHook'),
    visualization=dict(
        _scope_='mmpose', enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
halpe_halpe26 = [
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
    (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15),
    (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)
]

input_size = (
    192,
    256,
)
jhmdb_halpe26 = [
    (0, 18), (2, 17), (3, 6), (4, 5), (5, 12), (6, 11), (7, 8), (8, 7),
    (9, 14), (10, 13), (11, 10), (12, 9), (13, 16), (14, 15)
]

launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(
    _scope_='mmpose',
    by_epoch=True,
    num_digits=6,
    type='LogProcessor',
    window_size=50)
max_epochs = 700
model = dict(
    backbone=dict(
        _scope_='mmdet',
        act_cfg=dict(type='ReLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.167,
        expand_ratio=0.5,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-tiny_udp-body7_210e-256x192-a3775292_20230504.pth',
            prefix='backbone.',
            type='Pretrained'),
        norm_cfg=dict(type='SyncBN'),
        out_indices=(4, ),
        spp_kernel_sizes=[
            3,
            5,
            7,
        ],
        type='CSPNeXt',
        widen_factor=0.375),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            input_size=(
                192,
                256,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        final_layer_kernel_size=7,
        gau_cfg=dict(
            act_fn='ReLU',
            drop_path=0.0,
            dropout_rate=0.0,
            expansion_factor=2,
            hidden_dims=256,
            pos_enc=False,
            s=128,
            use_rel_bias=False),
        in_channels=384,
        in_featuremap_size=(
            6,
            8,
        ),
        input_size=(
            192,
            256,
        ),
        loss=dict(
            beta=10.0,
            label_softmax=True,
            type='KLDiscretLoss',
            use_target_weight=True),
        out_channels=26,
        simcc_split_ratio=2.0,
        type='RTMCCHead'),
    test_cfg=dict(flip_test=True),
    type='TopdownPoseEstimator')

mpii_halpe26 = [
    (0, 16), (1, 14), (2, 12), (3, 11), (4, 13), (5, 15), (8, 18), (9, 17),
    (10, 10), (11, 8), (12, 6), (13, 5), (14, 7), (15, 9)
]

num_keypoints = 26
ochuman_halpe26 = [
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
    (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16)
]

optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.004, type='AdamW', weight_decay=0.0),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=1e-05,
        type='LinearLR'),
    dict(
        T_max=350,
        begin=350,
        by_epoch=True,
        convert_to_iter_based=True,
        end=700,
        eta_min=0.0002,
        type='CosineAnnealingLR'),
]
posetrack_halpe26 = [
    (0, 0), (2, 17), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),
    (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16)
]

randomness = dict(seed=21)
resume = False
stage2_num_epochs = 30
test_cfg = dict()
test_dataloader = dict(
    batch_size=64,
    dataset=dict(
        datasets=[
            dict(
                ann_file=os.path.join(DATASETS_ROOT, 'COCO', 'annotations', 'coco_wholebody_val_v1.0.json'),
                data_mode='topdown',
                data_prefix=dict(
                    img=os.path.join(DATASETS_ROOT, 'COCO', 'images', 'val2017')),
                data_root='',
                pipeline=[
                    dict(
                        mapping=[
                            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
                            (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15),
                            (16, 16), (17, 20), (18, 22), (19, 24), (20, 21), (21, 23), (22, 25)
                        ],
                        num_keypoints=26,
                        type='KeypointConverter'),
                ],
                type='CocoWholeBodyDataset'),
            dict(
                ann_file=os.path.join(DATASETS_ROOT, 'Halpe', 'annotations', 'halpe_val_v1.json'),
                data_mode='topdown',
                data_prefix=dict(
                    img=os.path.join(DATASETS_ROOT, 'COCO', 'images', 'val2017')),
                data_root='',
                pipeline=[
                    dict(
                        mapping=[
                            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
                            (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15),
                            (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23),
                            (24, 24), (25, 25)
                        ],
                        
                        num_keypoints=26,
                        type='KeypointConverter'),
                ],
                type='HalpeDataset'),
        ],
        metainfo=dict(from_file='configs/_base_/datasets/halpe26.py'),
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CombinedDataset'),
    drop_last=False,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(thr=0.1, type='PCKAccuracy'),
    dict(type='AUC'),
]
train_batch_size = 256
train_cfg = dict(by_epoch=True, max_epochs=700, val_interval=10)
train_dataloader = dict(
    batch_size=256,
    dataset=dict(
        datasets=[
            dict(
                ann_file=os.path.join(DATASETS_ROOT, 'COCO', 'annotations', 'coco_wholebody_train_v1.0.json'),
                data_mode='topdown',
                data_prefix=dict(
                    img=os.path.join(DATASETS_ROOT, 'COCO', 'images', 'train2017')),
                data_root='',
                pipeline=[
                    dict(
                        mapping=[
                            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
                            (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15),
                            (16, 16), (17, 20), (18, 22), (19, 24), (20, 21), (21, 23), (22, 25)
                        ],
                        
                        num_keypoints=26,
                        type='KeypointConverter'),
                ],
                type='CocoWholeBodyDataset'),
            dict(
                ann_file=os.path.join(DATASETS_ROOT, 'Halpe', 'annotations', 'halpe_train_v1.json'),
                data_mode='topdown',
                data_prefix=dict(
                    img=os.path.join(DATASETS_ROOT, 'Halpe', 'hico_20160224_det', 'images', 'train2015')
                ),
                data_root='',
                pipeline=[
                    dict(
                        mapping=[
                            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
                            (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15),
                            (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23),
                            (24, 24), (25, 25)
                        ],
                        num_keypoints=26,
                        type='KeypointConverter'),
                ],
                type='HalpeDataset'),
        ],
        metainfo=dict(from_file='configs/_base_/datasets/halpe26.py'),
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(type='RandomHalfBody'),
            dict(
                rotate_factor=80,
                scale_factor=[
                    0.6,
                    1.4,
                ],
                type='RandomBBoxTransform'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='PhotometricDistortion'),
            dict(
                transforms=[
                    dict(p=0.1, type='Blur'),
                    dict(p=0.1, type='MedianBlur'),
                    dict(
                        max_height=0.4,
                        max_holes=1,
                        max_width=0.4,
                        min_height=0.2,
                        min_holes=1,
                        min_width=0.2,
                        p=1.0,
                        type='CoarseDropout'),
                ],
                type='Albumentation'),
            dict(
                encoder=dict(
                    input_size=(
                        192,
                        256,
                    ),
                    normalize=False,
                    sigma=(
                        4.9,
                        5.66,
                    ),
                    simcc_split_ratio=2.0,
                    type='SimCCLabel',
                    use_dark=False),
                type='GenerateTarget',
                use_dataset_keypoint_weights=True),
            dict(type='PackPoseInputs'),
        ],
        test_mode=False,
        type='CombinedDataset'),
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(type='RandomHalfBody'),
    dict(
        rotate_factor=80,
        scale_factor=[
            0.6,
            1.4,
        ],
        type='RandomBBoxTransform'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(type='PhotometricDistortion'),
    dict(
        transforms=[
            dict(p=0.1, type='Blur'),
            dict(p=0.1, type='MedianBlur'),
            dict(
                max_height=0.4,
                max_holes=1,
                max_width=0.4,
                min_height=0.2,
                min_holes=1,
                min_width=0.2,
                p=1.0,
                type='CoarseDropout'),
        ],
        type='Albumentation'),
    dict(
        encoder=dict(
            input_size=(
                192,
                256,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        type='GenerateTarget',
        use_dataset_keypoint_weights=True),
    dict(type='PackPoseInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(type='RandomHalfBody'),
    dict(
        rotate_factor=80,
        scale_factor=[
            0.6,
            1.4,
        ],
        shift_factor=0.0,
        type='RandomBBoxTransform'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(
        transforms=[
            dict(p=0.1, type='Blur'),
            dict(p=0.1, type='MedianBlur'),
            dict(
                max_height=0.4,
                max_holes=1,
                max_width=0.4,
                min_height=0.2,
                min_holes=1,
                min_width=0.2,
                p=0.5,
                type='CoarseDropout'),
        ],
        type='Albumentation'),
    dict(
        encoder=dict(
            input_size=(
                192,
                256,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        type='GenerateTarget',
        use_dataset_keypoint_weights=True),
    dict(type='PackPoseInputs'),
]
val_batch_size = 64
val_cfg = dict()
val_coco = dict(
    ann_file=os.path.join(DATASETS_ROOT, 'COCO', 'annotations', 'coco_wholebody_val_v1.0.json'),
    data_mode='topdown',
    data_prefix=dict(img=os.path.join(DATASETS_ROOT, 'COCO', 'images', 'val2017')),
    data_root='',
    pipeline=[
        dict(
            mapping=[
                (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
                (8, 8), (9, 9), (10, 10), (11, 11), (12, 12),
                (13, 13), (14, 14), (15, 15), (16, 16), (17, 20), (18, 22), (19, 24), (20, 21),
                (21, 23), (22, 25)
            ],
            num_keypoints=26,
            type='KeypointConverter'),
    ],
    type='CocoWholeBodyDataset')
val_dataloader = dict(
    batch_size=64,
    dataset=dict(
        datasets=[
            dict(
                ann_file=os.path.join(DATASETS_ROOT, 'COCO', 'annotations', 'coco_wholebody_val_v1.0.json'),
                data_mode='topdown',
                data_prefix=dict(
                    img=os.path.join(DATASETS_ROOT, 'COCO', 'images', 'val2017')),
                data_root='',
                pipeline=[
                    dict(
                        mapping=[
                            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
                            (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15),
                            (16, 16), (17, 20), (18, 22), (19, 24), (20, 21), (21, 23), (22, 25)
                        ],
                        
                        num_keypoints=26,
                        type='KeypointConverter'),
                ],
                type='CocoWholeBodyDataset'),
            dict(
                ann_file=
                '/datasets/basokure/Halpe/annotations/halpe_val_v1.json',
                data_mode='topdown',
                data_prefix=dict(
                    img=os.path.join(DATASETS_ROOT, 'COCO', 'images', 'val2017')),
                data_root='',
                pipeline=[
                    dict(
                        mapping=[
                            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
                            (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15),
                            (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23),
                            (24, 24), (25, 25)
                        ],
                        
                        num_keypoints=26,
                        type='KeypointConverter'),
                ],
                type='HalpeDataset'),
        ],
        metainfo=dict(from_file='configs/_base_/datasets/halpe26.py'),
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CombinedDataset'),
    drop_last=False,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(thr=0.1, type='PCKAccuracy'),
    dict(type='AUC'),
]
val_halpe = dict(
    ann_file=os.path.join(DATASETS_ROOT, 'Halpe', 'annotations', 'halpe_val_v1.json'),
    data_mode='topdown',
    data_prefix=dict(img=os.path.join(DATASETS_ROOT, 'COCO', 'images', 'val2017')),
    data_root='',
    pipeline=[
        dict(
            mapping=[
                (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
                (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15),
                (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23),
                (24, 24), (25, 25)
            ],
            
            num_keypoints=26,
            type='KeypointConverter'),
    ],
    type='HalpeDataset')
val_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(type='PackPoseInputs'),
]
vis_backends = [
    dict(_scope_='mmpose', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmpose',
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/elifpose_8xb1024-700e_body8-halpe26-256x192'
