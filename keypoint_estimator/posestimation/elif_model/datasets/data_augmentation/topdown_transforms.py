# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from mmengine import is_seq_of

from elif_model.datasets.transforms import get_udp_warp_matrix, get_warp_matrix
from copy import deepcopy
from elif_model.codecs.simcc_label import SimCCLabel

class TopdownAffine():
    """Get the bbox image as the model input by affine transform.

    Required Keys:

        - img
        - bbox_center
        - bbox_scale
        - bbox_rotation (optional)
        - keypoints (optional)

    Modified Keys:

        - img
        - bbox_scale

    Added Keys:

        - input_size
        - transformed_keypoints

    Args:
        input_size (Tuple[int, int]): The input image size of the model in
            [w, h]. The bbox region will be cropped and resize to `input_size`
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 use_udp: bool = False) -> None:
        super().__init__()

        assert is_seq_of(input_size, int) and len(input_size) == 2, (
            f'Invalid input_size {input_size}')

        self.input_size = input_size
        self.use_udp = use_udp

    @staticmethod
    def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float):
        """Reshape the bbox to a fixed aspect ratio.

        Args:
            bbox_scale (np.ndarray): The bbox scales (w, h) in shape (n, 2)
            aspect_ratio (float): The ratio of ``w/h``

        Returns:
            np.darray: The reshaped bbox scales in (n, 2)
        """

        w, h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(w > h * aspect_ratio,
                              np.hstack([w, w / aspect_ratio]),
                              np.hstack([h * aspect_ratio, h]))
        return bbox_scale

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`TopdownAffine`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        w, h = self.input_size
        warp_size = (int(w), int(h))

        # reshape bbox to fixed aspect ratio
        results['bbox_scale'] = self._fix_aspect_ratio(
            results['bbox_scale'], aspect_ratio=w / h)

        # TODO: support multi-instance
        assert results['bbox_center'].shape[0] == 1, (
            'Top-down heatmap only supports single instance. Got invalid '
            f'shape of bbox_center {results["bbox_center"].shape}.')

        center = results['bbox_center'][0]
        scale = results['bbox_scale'][0]
        if 'bbox_rotation' in results:
            rot = results['bbox_rotation'][0]
        else:
            rot = 0.

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(
                center, scale, rot, output_size=(w, h))
        else:
            warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

        if isinstance(results['img'], list):
            results['img'] = [
                cv2.warpAffine(
                    img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
                for img in results['img']
            ]
        else:
            results['img'] = cv2.warpAffine(
                results['img'], warp_mat, warp_size, flags=cv2.INTER_LINEAR)

        if results.get('keypoints', None) is not None:
            if results.get('transformed_keypoints', None) is not None:
                transformed_keypoints = results['transformed_keypoints'].copy()
            else:
                transformed_keypoints = results['keypoints'].copy()
            # Only transform (x, y) coordinates
            transformed_keypoints[..., :2] = cv2.transform(
                results['keypoints'][..., :2], warp_mat)
            results['transformed_keypoints'] = transformed_keypoints

        results['input_size'] = (w, h)
        results['input_center'] = center
        results['input_scale'] = scale

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(input_size={self.input_size}, '
        repr_str += f'use_udp={self.use_udp})'
        return repr_str
    
class GenerateTarget():
    """Encode keypoints into Target.

    The generated target is usually the supervision signal of the model
    learning, e.g. heatmaps or regression labels.

    Required Keys:

        - keypoints
        - keypoints_visible
        - dataset_keypoint_weights

    Added Keys:

        - The keys of the encoded items from the codec will be updated into
            the results, e.g. ``'heatmaps'`` or ``'keypoint_weights'``. See
            the specific codec for more details.

    Args:
        encoder (dict | list[dict]): The codec config for keypoint encoding.
            Both single encoder and multiple encoders (given as a list) are
            supported
        multilevel (bool): Determine the method to handle multiple encoders.
            If ``multilevel==True``, generate multilevel targets from a group
            of encoders of the same type (e.g. multiple :class:`MSRAHeatmap`
            encoders with different sigma values); If ``multilevel==False``,
            generate combined targets from a group of different encoders. This
            argument will have no effect in case of single encoder. Defaults
            to ``False``
        use_dataset_keypoint_weights (bool): Whether use the keypoint weights
            from the dataset meta information. Defaults to ``False``
        target_type (str, deprecated): This argument is deprecated and has no
            effect. Defaults to ``None``
    """

    def __init__(self,
                 encoder: dict,
                 target_type: Optional[str] = None,
                 multilevel: bool = False,
                 use_dataset_keypoint_weights: bool = False) -> None:
        super().__init__()

        self.encoder_cfg = deepcopy(encoder)
        self.multilevel = multilevel
        self.use_dataset_keypoint_weights = use_dataset_keypoint_weights

        self.encoder = SimCCLabel(**self.encoder_cfg)

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`GenerateTarget`.

        See ``transform()`` method of :class:`BaseTransform` for details.
        """

        if results.get('transformed_keypoints', None) is not None:
            # use keypoints transformed by TopdownAffine
            keypoints = results['transformed_keypoints']
        elif results.get('keypoints', None) is not None:
            # use original keypoints
            keypoints = results['keypoints']
        else:
            raise ValueError(
                'GenerateTarget requires \'transformed_keypoints\' or'
                ' \'keypoints\' in the results.')

        keypoints_visible = results['keypoints_visible']
        if keypoints_visible.ndim == 3 and keypoints_visible.shape[2] == 2:
            keypoints_visible, keypoints_visible_weights = \
                keypoints_visible[..., 0], keypoints_visible[..., 1]
            results['keypoints_visible'] = keypoints_visible
            results['keypoints_visible_weights'] = keypoints_visible_weights

        # Encoded items from the encoder(s) will be updated into the results.
        # Please refer to the document of the specific codec for details about
        # encoded items.
        if not isinstance(self.encoder, list):
            # For single encoding, the encoded items will be directly added
            # into results.
            auxiliary_encode_kwargs = {
                key: results[key]
                for key in self.encoder.auxiliary_encode_keys
            }
            encoded = self.encoder.encode(
                keypoints=keypoints,
                keypoints_visible=keypoints_visible,
                **auxiliary_encode_kwargs)

            if self.encoder.field_mapping_table:
                encoded[
                    'field_mapping_table'] = self.encoder.field_mapping_table
            if self.encoder.instance_mapping_table:
                encoded['instance_mapping_table'] = \
                    self.encoder.instance_mapping_table
            if self.encoder.label_mapping_table:
                encoded[
                    'label_mapping_table'] = self.encoder.label_mapping_table

        else:
            encoded_list = []
            _field_mapping_table = dict()
            _instance_mapping_table = dict()
            _label_mapping_table = dict()
            for _encoder in self.encoder:
                auxiliary_encode_kwargs = {
                    key: results[key]
                    for key in _encoder.auxiliary_encode_keys
                }
                encoded_list.append(
                    _encoder.encode(
                        keypoints=keypoints,
                        keypoints_visible=keypoints_visible,
                        **auxiliary_encode_kwargs))

                _field_mapping_table.update(_encoder.field_mapping_table)
                _instance_mapping_table.update(_encoder.instance_mapping_table)
                _label_mapping_table.update(_encoder.label_mapping_table)

            if self.multilevel:
                # For multilevel encoding, the encoded items from each encoder
                # should have the same keys.

                keys = encoded_list[0].keys()
                if not all(_encoded.keys() == keys
                           for _encoded in encoded_list):
                    raise ValueError(
                        'Encoded items from all encoders must have the same '
                        'keys if ``multilevel==True``.')

                encoded = {
                    k: [_encoded[k] for _encoded in encoded_list]
                    for k in keys
                }

            else:
                # For combined encoding, the encoded items from different
                # encoders should have no overlapping items, except for
                # `keypoint_weights`. If multiple `keypoint_weights` are given,
                # they will be multiplied as the final `keypoint_weights`.

                encoded = dict()
                keypoint_weights = []

                for _encoded in encoded_list:
                    for key, value in _encoded.items():
                        if key == 'keypoint_weights':
                            keypoint_weights.append(value)
                        elif key not in encoded:
                            encoded[key] = value
                        else:
                            raise ValueError(
                                f'Overlapping item "{key}" from multiple '
                                'encoders, which is not supported when '
                                '``multilevel==False``')

                if keypoint_weights:
                    encoded['keypoint_weights'] = keypoint_weights

            if _field_mapping_table:
                encoded['field_mapping_table'] = _field_mapping_table
            if _instance_mapping_table:
                encoded['instance_mapping_table'] = _instance_mapping_table
            if _label_mapping_table:
                encoded['label_mapping_table'] = _label_mapping_table

        if self.use_dataset_keypoint_weights and 'keypoint_weights' in encoded:
            if isinstance(encoded['keypoint_weights'], list):
                for w in encoded['keypoint_weights']:
                    w = w * results['dataset_keypoint_weights']
            else:
                encoded['keypoint_weights'] = encoded[
                    'keypoint_weights'] * results['dataset_keypoint_weights']

        results.update(encoded)

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += (f'(encoder={str(self.encoder_cfg)}, ')
        repr_str += ('use_dataset_keypoint_weights='
                     f'{self.use_dataset_keypoint_weights})')
        return repr_str











