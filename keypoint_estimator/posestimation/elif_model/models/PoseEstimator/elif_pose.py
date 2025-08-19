"""
ElifPose: High-level pose estimation model definition for ElifPose project.

This module defines the ElifPose neural network for human pose estimation using a CSPNeXt backbone and RTMCC head.
It provides methods for feature extraction, forward inference, loss computation, and prediction post-processing.

Author: ETH Zurich Digital Circuts and Systems Group
Date: 2025-06-30
"""

import torch
import torch.nn as nn
from itertools import zip_longest
from typing import Optional, Union, Tuple
import os
from torch import Tensor

from elif_model.utils.typing import (InstanceList, PixelDataList, SampleList)

from elif_model.models.backbones.CSPNeXt import CSPNeXt
from elif_model.models.heads.rtmcc_head import RTMCCHead

class ElifPose(nn.Module):
    """
    ElifPose neural network for human pose estimation.

    This model uses a CSPNeXt backbone and RTMCC head for keypoint detection.
    It provides methods for feature extraction, forward pass, loss calculation,
    and prediction post-processing.

    Attributes:
        input_size (tuple): Input image size (height, width).
        num_keypoints (int): Number of keypoints to predict.
        codec (dict): Codec configuration for keypoint encoding/decoding.
        backbone (nn.Module): Feature extraction backbone.
        head (nn.Module): Keypoint prediction head.
        test_cfg (dict): Test configuration.
        data_preprocessor (dict): Preprocessing parameters (mean, std, etc).
        with_neck (bool): Whether to use a neck module (not used).
        with_head (bool): Whether to use the head module.
    """
    input_size = (192, 256)
    num_keypoints = 26
    codec = dict(
        input_size=input_size,
        sigma=(4.9, 5.66),
        simcc_split_ratio=2.0,
        normalize=False,
        use_dark=False)

    def __init__(
        self,
        data_preprocessor_params: dict = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True
        ),
        backbone_params: dict = dict(
            arch='P5',
            expand_ratio=0.5,
            deepen_factor=0.33,
            widen_factor=0.5,
            out_indices=(4, ),
            channel_attention=True,
            act_cfg=dict(type='SiLU')
        ),
        head_params: dict = dict(
            in_channels=512,
            out_channels=num_keypoints,
            input_size=input_size,
            in_featuremap_size=tuple([s // 32 for s in input_size]),
            simcc_split_ratio=codec['simcc_split_ratio'],
            final_layer_kernel_size=7,
            gau_cfg=dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='SiLU',
                use_rel_bias=False,
                pos_enc=False),
            loss=dict(
                use_target_weight=True,
                beta=10.,
                label_softmax=True),
            decoder=codec),
        test_cfg=dict(flip_test=True)
    ):
        super().__init__()
        self.backbone = CSPNeXt(**backbone_params)
        self.head = RTMCCHead(**head_params)
        self.test_cfg = test_cfg
        self.data_preprocessor = data_preprocessor_params
        self.with_neck = False
        self.with_head = True
        # Initialize model weights
        self.init_model(pretrained=True)

    def init_model(self, pretrained: bool = True):
        """
        Initialize model weights from checkpoint if pretrained is True.

        Args:
            pretrained (bool): Whether to load pretrained weights.
        """
        if pretrained:
            print('Loading pretrained weights for CSPNeXt')
            MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            checkpoint_path = os.path.join(MODEL_ROOT, 'models', 'backbones', 'cspnext.pth')
            checkpoint = torch.load(checkpoint_path)
            # Split checkpoint into backbone and head weights
            state_dict = checkpoint['state_dict']
            state_dict_head = {}
            state_dict_backbone = {}
            for key in list(state_dict.keys()):
                if key.startswith('head'):
                    state_dict_head[key.replace('head.', '')] = state_dict.pop(key)
                if key.startswith('backbone'):
                    state_dict_backbone[key.replace('backbone.', '')] = state_dict.pop(key)
            print("Initializing weights for backbone")
            self.backbone.load_state_dict(state_dict_backbone, strict=True)
            print("Initializing weights for head")
            init_weights_head(self.head)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """
        Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of computed losses.
        """
        feats = self.extract_feat(inputs)
        losses = dict()
        if self.with_head:
            losses.update(self.head.loss(feats, data_samples))
        return losses

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W)
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples

        Returns:
            list[:obj:`PoseDataSample`]: The pose estimation results of the
            input images. The return value is `PoseDataSample` instances with
            ``pred_instances`` and ``pred_fields``(optional) field , and
            ``pred_instances`` usually contains the following keys:

                - keypoints (Tensor): predicted keypoint coordinates in shape
                    (num_instances, K, D) where K is the keypoint number and D
                    is the keypoint dimension
                - keypoint_scores (Tensor): predicted keypoint scores in shape
                    (num_instances, K)
        """
        assert self.with_head, (
            'The model must have head to perform prediction.')
        if self.test_cfg.get('flip_test', False):
            _feats = self.extract_feat(inputs)
            _feats_flip = self.extract_feat(inputs.flip(-1))
            feats = [_feats, _feats_flip]
        else:
            feats = self.extract_feat(inputs)

        preds = self.head.predict(feats, data_samples, test_cfg=self.test_cfg)

        if isinstance(preds, tuple):
            batch_pred_instances, batch_pred_fields = preds
        else:
            batch_pred_instances = preds
            batch_pred_fields = None

        results = self.add_pred_to_datasample(batch_pred_instances,
                                              batch_pred_fields, data_samples)

        return results

    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:
        """Add predictions into data samples.

        Args:
            batch_pred_instances (List[InstanceData]): The predicted instances
                of the input data batch
            batch_pred_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            batch_data_samples (List[PoseDataSample]): The input data batch

        Returns:
            List[PoseDataSample]: A list of data samples where the predictions
            are stored in the ``pred_instances`` field of each data sample.
        """
        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []
        output_keypoint_indices = self.test_cfg.get('output_keypoint_indices',
                                                    None)

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            gt_instances = data_sample.gt_instances

            # convert keypoint coordinates from input space to image space
            input_center = data_sample.metainfo['input_center']
            input_scale = data_sample.metainfo['input_scale']
            input_size = data_sample.metainfo['input_size']

            pred_instances.keypoints[..., :2] = \
                pred_instances.keypoints[..., :2] / input_size * input_scale \
                + input_center - 0.5 * input_scale
            if 'keypoints_visible' not in pred_instances:
                pred_instances.keypoints_visible = \
                    pred_instances.keypoint_scores

            if output_keypoint_indices is not None:
                # select output keypoints with given indices
                num_keypoints = pred_instances.keypoints.shape[1]
                for key, value in pred_instances.all_items():
                    if key.startswith('keypoint'):
                        pred_instances.set_field(
                            value[:, output_keypoint_indices], key)

            # add bbox information into pred_instances
            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                if output_keypoint_indices is not None:
                    # select output heatmap channels with keypoint indices
                    # when the number of heatmap channel matches num_keypoints
                    for key, value in pred_fields.all_items():
                        if value.shape[0] != num_keypoints:
                            continue
                        pred_fields.set_field(value[output_keypoint_indices],
                                              key)
                data_sample.pred_fields = pred_fields

        return batch_data_samples


    def forward(self, inputs: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """
        Forward pass through the network (backbone and head).

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            Union[Tensor, Tuple[Tensor]]: Output of the network.
        """
        x = self.extract_feat(inputs)
        if self.with_head:
            x = self.head.forward(x)
        return x

    def extract_feat(self, inputs: Tensor) -> Tensor:
        """
        Extract features from input images using the backbone.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H, W).

        Returns:
            Tensor: Extracted features (first level).
        """
        inputs = self.preprocess(inputs)
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)  # Neck not used by default
        return x[0]


    def preprocess(self, inputs: Tensor) -> Tensor:
        """
        Preprocess the input data (normalize using mean and std).

        Args:
            inputs (Tensor): Input data with shape (N, C, H, W).

        Returns:
            Tensor: Preprocessed data.
        """
        # Convert to float32
        inputs = inputs.to(torch.float32)
        # Move mean and std to correct device
        if inputs.get_device() >= 0:
            mean = torch.tensor(self.data_preprocessor['mean']).reshape(1, 3, 1, 1).cuda()
            std = torch.tensor(self.data_preprocessor['std']).reshape(1, 3, 1, 1).cuda()
        else:
            mean = torch.tensor(self.data_preprocessor['mean']).reshape(1, 3, 1, 1)
            std = torch.tensor(self.data_preprocessor['std']).reshape(1, 3, 1, 1)
        # Normalize
        inputs = (inputs - mean) / std
        return inputs

def init_weights_head(module: nn.Module):
    """
    Initialize weights of the head module (Conv2d, Linear, BatchNorm2d).

    Args:
        module (nn.Module): Module to initialize.
    """
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, 0, 0.001)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0, 0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.running_mean, 0)
        nn.init.constant_(module.running_var, 1)
