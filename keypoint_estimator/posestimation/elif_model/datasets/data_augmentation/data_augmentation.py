from elif_model.datasets.transforms import bbox_xyxy2cs, flip_bbox, imflip
from elif_model.datasets.keypoint_transforms import flip_keypoints  
import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple, Union
import albumentations

import cv2
import numpy as np
from scipy.stats import truncnorm
import os

from elif_model.datasets.data_augmentation.colorspace import bgr2hsv, hsv2bgr
from elif_model.datasets.transforms import get_warp_matrix
from elif_model.codecs.simcc_label import SimCCLabel

def LoadImage(base_path, results: dict) -> Optional[dict]:
    """ Loads an image from specified path
    """
    filename = results['img_path']
    file_path = os.path.join(base_path, filename)
    
    
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    #float_image = img.astype(np.float32) / 255.
    float_image = img
    #img = Image.open(file_path)
    #convert_tensor = transforms.ToTensor()
    #tensor_image = convert_tensor(img)

    results["img"] = float_image
    results['img_shape'] = float_image.shape

    return results

def GetBBoxCenterScale(results: dict, padding=1.25) -> Optional[dict]:

    bbox = results['bbox']
    center, scale = bbox_xyxy2cs(bbox, padding=padding)
    results['bbox_center'] = center
    results['bbox_scale'] = scale

    return results

def fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float):
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

def TopDownAffine(results : dict, input_size : tuple):
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
    Returns:
        results (dict): Results containing warpped image and keypoints
    """

    w, h = input_size
    warp_size = (int(w), int(h))

    # reshape bbox to fixed aspect ratio
    results['bbox_scale'] = fix_aspect_ratio(
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

def GenerateTarget(results: dict, encoder_cfg: dict, use_dataset_keypoint_weights: bool) -> Optional[dict]:
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

    encoder = SimCCLabel(**encoder_cfg)

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
    if not isinstance(encoder, list):
        # For single encoding, the encoded items will be directly added
        # into results.

        encoded = encoder.encode(
            keypoints=keypoints,
            keypoints_visible=keypoints_visible)

        if encoder.field_mapping_table:
            encoded[
                'field_mapping_table'] = encoder.field_mapping_table
        if encoder.instance_mapping_table:
            encoded['instance_mapping_table'] = \
                encoder.instance_mapping_table
        if encoder.label_mapping_table:
            encoded[
                'label_mapping_table'] = encoder.label_mapping_table


    if use_dataset_keypoint_weights and 'keypoint_weights' in encoded:
        if isinstance(encoded['keypoint_weights'], list):
            for w in encoded['keypoint_weights']:
                w = w * results['keypoint_weights']
        else:
            encoded['keypoint_weights'] = encoded[
                'keypoint_weights'] * results['keypoint_weights']

    results.update(encoded)

    return results

def KeyPointConverter(results: dict, num_keypoints: int,
                 mapping: Union[List[Tuple[int, int]], List[Tuple[Tuple,
                                                                  int]]]):
    

    if len(mapping):
        source_index, target_index = zip(*mapping)
    else:
        source_index, target_index = [], []

    src1, src2 = [], []
    interpolation = False
    for x in source_index:
        if isinstance(x, (list, tuple)):
            assert len(x) == 2, 'source_index should be a list/tuple of ' \
                                'length 2'
            src1.append(x[0])
            src2.append(x[1])
            interpolation = True
        else:
            src1.append(x)
            src2.append(x)

    # When paired source_indexes are input,
    # keep a self.source_index2 for interpolation
    if interpolation:
        source_index2 = src2

    source_index = src1
    target_index = list(target_index)
    interpolation = interpolation

    """Transforms the keypoint results to match the target keypoints."""
    num_instances = results['keypoints'].shape[0]

    if 'keypoints_visible' not in results:
        results['keypoints_visible'] = np.ones(
            (num_instances, results['keypoints'].shape[1]))

    if len(results['keypoints_visible'].shape) > 2:
        results['keypoints_visible'] = results['keypoints_visible'][:, :, 0]

    # Initialize output arrays
    keypoints = np.zeros((num_instances, num_keypoints, 3))
    keypoints_visible = np.zeros((num_instances, num_keypoints))
    key = 'keypoints_3d' if 'keypoints_3d' in results else 'keypoints'
    c = results[key].shape[-1]

    flip_indices = results.get('flip_indices', None)

    # Create a mask to weight visibility loss
    keypoints_visible_weights = keypoints_visible.copy()
    keypoints_visible_weights[:, target_index] = 1.0

    # Interpolate keypoints if pairs of source indexes provided
    if interpolation:
        keypoints[:, target_index, :c] = 0.5 * (
            results[key][:, source_index] +
            results[key][:, source_index2])
        keypoints_visible[:, target_index] = results[
            'keypoints_visible'][:, source_index] * results[
                'keypoints_visible'][:, source_index2]
        # Flip keypoints if flip_indices provided
        if flip_indices is not None:
            for i, (x1, x2) in enumerate(
                    zip(source_index, source_index2)):
                idx = flip_indices[x1] if x1 == x2 else i
                flip_indices[i] = idx if idx < num_keypoints else i
            flip_indices = flip_indices[:len(source_index)]
    # Otherwise just copy from the source index
    else:
        keypoints[:,
                    target_index, :c] = results[key][:,
                                                    source_index]
        keypoints_visible[:, target_index] = results[
            'keypoints_visible'][:, source_index]

    # Update the results dict
    results['keypoints'] = keypoints[..., :2]
    results['keypoints_visible'] = np.stack(
        [keypoints_visible, keypoints_visible_weights], axis=2)
    if 'keypoints_3d' in results:
        results['keypoints_3d'] = keypoints
        results['lifting_target'] = keypoints[results['target_idx']]
        results['lifting_target_visible'] = keypoints_visible[
            results['target_idx']]
    results['flip_indices'] = flip_indices

    return results

def RandomFlip(prob, results: dict) -> dict:
    """The transform function of :class:`RandomFlip`.

    See ``transform()`` method of :class:`BaseTransform` for details.

    Args:
        results (dict): The result dict

    Returns:
        dict: The result dict.
    """

    if np.random.rand() > 1:
        flip_dir = None
    else:
        flip_dir = 'horizontal'

    if flip_dir is None:
        results['flip'] = False
        results['flip_direction'] = None
    else:
        results['flip'] = True
        results['flip_direction'] = flip_dir

        h, w = results.get('input_size', results['img_shape'])
        # flip image and mask
        if isinstance(results['img'], list):
            results['img'] = [
                imflip(img, direction=flip_dir) for img in results['img']
            ]
        else:
            results['img'] = imflip(results['img'], direction=flip_dir)

        if 'img_mask' in results:
            results['img_mask'] = imflip(
                results['img_mask'], direction=flip_dir)

        # flip bboxes
        if results.get('bbox', None) is not None:
            results['bbox'] = flip_bbox(
                results['bbox'],
                image_size=(w, h),
                bbox_format='xyxy',
                direction=flip_dir)

        if results.get('bbox_center', None) is not None:
            results['bbox_center'] = flip_bbox(
                results['bbox_center'],
                image_size=(w, h),
                bbox_format='center',
                direction=flip_dir)

        # flip keypoints
        if results.get('keypoints', None) is not None:
            print("Flipping the keypoints")
            keypoints, keypoints_visible = flip_keypoints(
                results['keypoints'],
                results.get('keypoints_visible', None),
                image_size=(w, h),
                flip_indices=results['flip_indices'],
                direction=flip_dir)

            results['keypoints'] = keypoints
            results['keypoints_visible'] = keypoints_visible

    return results

class RandomHalfBody():
    """Data augmentation with half-body transform that keeps only the upper or
    lower body at random.

    Required Keys:

        - keypoints
        - keypoints_visible
        - upper_body_ids
        - lower_body_ids

    Modified Keys:

        - bbox
        - bbox_center
        - bbox_scale

    Args:
        min_total_keypoints (int): The minimum required number of total valid
            keypoints of a person to apply half-body transform. Defaults to 8
        min_half_keypoints (int): The minimum required number of valid
            half-body keypoints of a person to apply half-body transform.
            Defaults to 2
        padding (float): The bbox padding scale that will be multilied to
            `bbox_scale`. Defaults to 1.5
        prob (float): The probability to apply half-body transform when the
            keypoint number meets the requirement. Defaults to 0.3
    """

    def __init__(self,
                 min_total_keypoints: int = 9,
                 min_upper_keypoints: int = 2,
                 min_lower_keypoints: int = 3,
                 padding: float = 1.5,
                 prob: float = 0.3,
                 upper_prioritized_prob: float = 0.7) -> None:
        super().__init__()
        self.min_total_keypoints = min_total_keypoints
        self.min_upper_keypoints = min_upper_keypoints
        self.min_lower_keypoints = min_lower_keypoints
        self.padding = padding
        self.prob = prob
        self.upper_prioritized_prob = upper_prioritized_prob

    def _get_half_body_bbox(self, keypoints: np.ndarray,
                            half_body_ids: List[int]
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """Get half-body bbox center and scale of a single instance.

        Args:
            keypoints (np.ndarray): Keypoints in shape (K, D)
            upper_body_ids (list): The list of half-body keypont indices

        Returns:
            tuple: A tuple containing half-body bbox center and scale
            - center: Center (x, y) of the bbox
            - scale: Scale (w, h) of the bbox
        """

        selected_keypoints = keypoints[half_body_ids]
        center = selected_keypoints.mean(axis=0)[:2]
        x1, y1 = selected_keypoints.min(axis=0)
        x2, y2 = selected_keypoints.max(axis=0)
        w = x2 - x1
        h = y2 - y1
        scale = np.array([w, h], dtype=center.dtype) * self.padding

        return center, scale

    def _random_select_half_body(self, keypoints_visible: np.ndarray,
                                 upper_body_ids: List[int],
                                 lower_body_ids: List[int]
                                 ) -> List[Optional[List[int]]]:
        """Randomly determine whether applying half-body transform and get the
        half-body keyponit indices of each instances.

        Args:
            keypoints_visible (np.ndarray, optional): The visibility of
                keypoints in shape (N, K, 1) or (N, K, 2).
            upper_body_ids (list): The list of upper body keypoint indices
            lower_body_ids (list): The list of lower body keypoint indices

        Returns:
            list[list[int] | None]: The selected half-body keypoint indices
            of each instance. ``None`` means not applying half-body transform.
        """

        if keypoints_visible.ndim == 3:
            keypoints_visible = keypoints_visible[..., 0]

        half_body_ids = []

        for visible in keypoints_visible:
            if visible.sum() < self.min_total_keypoints:
                indices = None
            elif np.random.rand() > self.prob:
                indices = None
            else:
                upper_valid_ids = [i for i in upper_body_ids if visible[i] > 0]
                lower_valid_ids = [i for i in lower_body_ids if visible[i] > 0]

                num_upper = len(upper_valid_ids)
                num_lower = len(lower_valid_ids)

                prefer_upper = np.random.rand() < self.upper_prioritized_prob
                if (num_upper < self.min_upper_keypoints
                        and num_lower < self.min_lower_keypoints):
                    indices = None
                elif num_lower < self.min_lower_keypoints:
                    indices = upper_valid_ids
                elif num_upper < self.min_upper_keypoints:
                    indices = lower_valid_ids
                else:
                    indices = (
                        upper_valid_ids if prefer_upper else lower_valid_ids)

            half_body_ids.append(indices)

        return half_body_ids

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`HalfBodyTransform`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        half_body_ids = self._random_select_half_body(
            keypoints_visible=results['keypoints_visible'],
            upper_body_ids=results['upper_body_ids'],
            lower_body_ids=results['lower_body_ids'])

        bbox_center = []
        bbox_scale = []
        for i, indices in enumerate(half_body_ids):
            if indices is None:
                bbox_center.append(results['bbox_center'][i])
                bbox_scale.append(results['bbox_scale'][i])
            else:
                _center, _scale = self._get_half_body_bbox(
                    results['keypoints'][i], indices)
                bbox_center.append(_center)
                bbox_scale.append(_scale)

        results['bbox_center'] = np.stack(bbox_center)
        results['bbox_scale'] = np.stack(bbox_scale)
        return results
    
class RandomBBoxTransform():
    r"""Rnadomly shift, resize and rotate the bounding boxes.

    Required Keys:

        - bbox_center
        - bbox_scale

    Modified Keys:

        - bbox_center
        - bbox_scale

    Added Keys:
        - bbox_rotation

    Args:
        shift_factor (float): Randomly shift the bbox in range
            :math:`[-dx, dx]` and :math:`[-dy, dy]` in X and Y directions,
            where :math:`dx(y) = x(y)_scale \cdot shift_factor` in pixels.
            Defaults to 0.16
        shift_prob (float): Probability of applying random shift. Defaults to
            0.3
        scale_factor (Tuple[float, float]): Randomly resize the bbox in range
            :math:`[scale_factor[0], scale_factor[1]]`. Defaults to (0.5, 1.5)
        scale_prob (float): Probability of applying random resizing. Defaults
            to 1.0
        rotate_factor (float): Randomly rotate the bbox in
            :math:`[-rotate_factor, rotate_factor]` in degrees. Defaults
            to 80.0
        rotate_prob (float): Probability of applying random rotation. Defaults
            to 0.6
    """

    def __init__(self,
                 shift_factor: float = 0.16,
                 shift_prob: float = 0.3,
                 scale_factor: Tuple[float, float] = (0.5, 1.5),
                 scale_prob: float = 1.0,
                 rotate_factor: float = 80.0,
                 rotate_prob: float = 0.6) -> None:
        super().__init__()

        self.shift_factor = shift_factor
        self.shift_prob = shift_prob
        self.scale_factor = scale_factor
        self.scale_prob = scale_prob
        self.rotate_factor = rotate_factor
        self.rotate_prob = rotate_prob

    @staticmethod
    def _truncnorm(low: float = -1.,
                   high: float = 1.,
                   size: tuple = ()) -> np.ndarray:
        """Sample from a truncated normal distribution."""
        return truncnorm.rvs(low, high, size=size).astype(np.float32)

    def _get_transform_params(self, num_bboxes: int) -> Tuple:
        """Get random transform parameters.

        Args:
            num_bboxes (int): The number of bboxes

        Returns:
            tuple:
            - offset (np.ndarray): Offset factor of each bbox in shape (n, 2)
            - scale (np.ndarray): Scaling factor of each bbox in shape (n, 1)
            - rotate (np.ndarray): Rotation degree of each bbox in shape (n,)
        """
        random_v = self._truncnorm(size=(num_bboxes, 4))
        offset_v = random_v[:, :2]
        scale_v = random_v[:, 2:3]
        rotate_v = random_v[:, 3]

        # Get shift parameters
        offset = offset_v * self.shift_factor
        offset = np.where(
            np.random.rand(num_bboxes, 1) < self.shift_prob, offset, 0.)

        # Get scaling parameters
        scale_min, scale_max = self.scale_factor
        mu = (scale_max + scale_min) * 0.5
        sigma = (scale_max - scale_min) * 0.5
        scale = scale_v * sigma + mu
        scale = np.where(
            np.random.rand(num_bboxes, 1) < self.scale_prob, scale, 1.)

        # Get rotation parameters
        rotate = rotate_v * self.rotate_factor
        rotate = np.where(
            np.random.rand(num_bboxes) < self.rotate_prob, rotate, 0.)

        return offset, scale, rotate

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`RandomBboxTransform`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        bbox_scale = results['bbox_scale']
        num_bboxes = bbox_scale.shape[0]

        offset, scale, rotate = self._get_transform_params(num_bboxes)

        results['bbox_center'] = results['bbox_center'] + offset * bbox_scale
        results['bbox_scale'] = results['bbox_scale'] * scale
        results['bbox_rotation'] = rotate

        return results
    
class PhotometricDistortion():
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Sequence[float] = (0.5, 1.5),
                 saturation_range: Sequence[float] = (0.5, 1.5),
                 hue_delta: int = 18) -> None:
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def _random_flags(self) -> Sequence[float]:
        """Generate the random flags for subsequent transforms.

        Returns:
            Sequence[Number]: a sequence of numbers that indicate whether to
                do the corresponding transforms.
        """
        # contrast_mode == 0 --> do random contrast first
        # contrast_mode == 1 --> do random contrast last
        contrast_mode = np.random.randint(2)
        # whether to apply brightness distortion
        brightness_flag = np.random.randint(2)
        # whether to apply contrast distortion
        contrast_flag = np.random.randint(2)
        # the mode to convert color from BGR to HSV
        hsv_mode = np.random.randint(4)
        # whether to apply channel swap
        swap_flag = np.random.randint(2)

        # the beta in `self._convert` to be added to image array
        # in brightness distortion
        brightness_beta = np.random.uniform(-self.brightness_delta,
                                            self.brightness_delta)
        # the alpha in `self._convert` to be multiplied to image array
        # in contrast distortion
        contrast_alpha = np.random.uniform(self.contrast_lower,
                                           self.contrast_upper)
        # the alpha in `self._convert` to be multiplied to image array
        # in saturation distortion to hsv-formatted img
        saturation_alpha = np.random.uniform(self.saturation_lower,
                                             self.saturation_upper)
        # delta of hue to add to image array in hue distortion
        hue_delta = np.random.randint(-self.hue_delta, self.hue_delta)
        # the random permutation of channel order
        swap_channel_order = np.random.permutation(3)

        return (contrast_mode, brightness_flag, contrast_flag, hsv_mode,
                swap_flag, brightness_beta, contrast_alpha, saturation_alpha,
                hue_delta, swap_channel_order)

    def _convert(self,
                 img: np.ndarray,
                 alpha: float = 1,
                 beta: float = 0) -> np.ndarray:
        """Multiple with alpha and add beta with clip.

        Args:
            img (np.ndarray): The image array.
            alpha (float): The random multiplier.
            beta (float): The random offset.

        Returns:
            np.ndarray: The updated image array.
        """
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def transform(self, results: dict) -> dict:
        """The transform function of :class:`PhotometricDistortion` to perform
        photometric distortion on images.

        See ``transform()`` method of :class:`BaseTransform` for details.


        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        assert 'img' in results, '`img` is not found in results'
        img = results['img']

        (contrast_mode, brightness_flag, contrast_flag, hsv_mode, swap_flag,
         brightness_beta, contrast_alpha, saturation_alpha, hue_delta,
         swap_channel_order) = self._random_flags()

        # random brightness distortion
        if brightness_flag:
            img = self._convert(img, beta=brightness_beta)

        # contrast_mode == 0 --> do random contrast first
        # contrast_mode == 1 --> do random contrast last
        if contrast_mode == 1:
            if contrast_flag:
                img = self._convert(img, alpha=contrast_alpha)

        if hsv_mode:
            # random saturation/hue distortion
            img = bgr2hsv(img)
            if hsv_mode == 1 or hsv_mode == 3:
                # apply saturation distortion to hsv-formatted img
                img[:, :, 1] = self._convert(
                    img[:, :, 1], alpha=saturation_alpha)
            if hsv_mode == 2 or hsv_mode == 3:
                # apply hue distortion to hsv-formatted img
                img[:, :, 0] = img[:, :, 0].astype(int) + hue_delta
            img = hsv2bgr(img)

        if contrast_mode == 1:
            if contrast_flag:
                img = self._convert(img, alpha=contrast_alpha)

        # randomly swap channels
        if swap_flag:
            img = img[..., swap_channel_order]

        results['img'] = img
        return results

class Albumentation():
    """Albumentation augmentation (pixel-level transforms only).

    Adds custom pixel-level transformations from Albumentations library.
    Please visit `https://albumentations.ai/docs/`
    to get more information.

    Note: we only support pixel-level transforms.
    Please visit `https://github.com/albumentations-team/`
    `albumentations#pixel-level-transforms`
    to get more information about pixel-level transforms.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        transforms (List[dict]): A list of Albumentation transforms.
            An example of ``transforms`` is as followed:
            .. code-block:: python

                [
                    dict(
                        type='RandomBrightnessContrast',
                        brightness_limit=[0.1, 0.3],
                        contrast_limit=[0.1, 0.3],
                        p=0.2),
                    dict(type='ChannelShuffle', p=0.1),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Blur', blur_limit=3, p=1.0),
                            dict(type='MedianBlur', blur_limit=3, p=1.0)
                        ],
                        p=0.1),
                ]
        keymap (dict | None): key mapping from ``input key`` to
            ``albumentation-style key``.
            Defaults to None, which will use {'img': 'image'}.
    """

    def __init__(self,
                 transforms: List[dict],
                 keymap: Optional[dict] = None) -> None:
        if albumentations is None:
            raise RuntimeError('albumentations is not installed')

        self.transforms = transforms

        self.aug = albumentations.Compose(
            [self.albu_builder(t) for t in self.transforms])

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
            }
        else:
            self.keymap_to_albu = keymap

    def albu_builder(self, cfg: dict) -> albumentations:
        """Import a module from albumentations.

        It resembles some of :func:`build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            albumentations.BasicTransform: The constructed transform object
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            obj_cls = getattr(albumentations, obj_type)
        elif isinstance(obj_type, type):
            obj_cls = obj_type
        else:
            raise TypeError(f'type must be a str, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    def transform(self, results: dict) -> dict:
        """The transform function of :class:`Albumentation` to apply
        albumentations transforms.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): Result dict from the data pipeline.

        Return:
            dict: updated result dict.
        """
        # map result dict to albumentations format
        results_albu = {}
        for k, v in self.keymap_to_albu.items():
            assert k in results, \
                f'The `{k}` is required to perform albumentations transforms'
            results_albu[v] = results[k]

        # Apply albumentations transforms
        results_albu = self.aug(**results_albu)

        # map the albu results back to the original format
        for k, v in self.keymap_to_albu.items():
            results[k] = results_albu[v]

        return results
    
class KeypointConverter():
    """Change the order of keypoints according to the given mapping.

    Required Keys:

        - keypoints
        - keypoints_visible

    Modified Keys:

        - keypoints
        - keypoints_visible

    Args:
        num_keypoints (int): The number of keypoints in target dataset.
        mapping (list): A list containing mapping indexes. Each element has
            format (source_index, target_index)

    Example:
        >>> import numpy as np
        >>> # case 1: 1-to-1 mapping
        >>> # (0, 0) means target[0] = source[0]
        >>> self = KeypointConverter(
        >>>     num_keypoints=3,
        >>>     mapping=[
        >>>         (0, 0), (1, 1), (2, 2), (3, 3)
        >>>     ])
        >>> results = dict(
        >>>     keypoints=np.arange(34).reshape(2, 3, 2),
        >>>     keypoints_visible=np.arange(34).reshape(2, 3, 2) % 2)
        >>> results = self(results)
        >>> assert np.equal(results['keypoints'],
        >>>                 np.arange(34).reshape(2, 3, 2)).all()
        >>> assert np.equal(results['keypoints_visible'],
        >>>                 np.arange(34).reshape(2, 3, 2) % 2).all()
        >>>
        >>> # case 2: 2-to-1 mapping
        >>> # ((1, 2), 0) means target[0] = (source[1] + source[2]) / 2
        >>> self = KeypointConverter(
        >>>     num_keypoints=3,
        >>>     mapping=[
        >>>         ((1, 2), 0), (1, 1), (2, 2)
        >>>     ])
        >>> results = dict(
        >>>     keypoints=np.arange(34).reshape(2, 3, 2),
        >>>     keypoints_visible=np.arange(34).reshape(2, 3, 2) % 2)
        >>> results = self(results)
    """

    def __init__(self, num_keypoints: int,
                 mapping: Union[List[Tuple[int, int]], List[Tuple[Tuple,
                                                                  int]]]):
        self.num_keypoints = num_keypoints
        self.mapping = mapping
        if len(mapping):
            source_index, target_index = zip(*mapping)
        else:
            source_index, target_index = [], []

        src1, src2 = [], []
        interpolation = False
        for x in source_index:
            if isinstance(x, (list, tuple)):
                assert len(x) == 2, 'source_index should be a list/tuple of ' \
                                    'length 2'
                src1.append(x[0])
                src2.append(x[1])
                interpolation = True
            else:
                src1.append(x)
                src2.append(x)

        # When paired source_indexes are input,
        # keep a self.source_index2 for interpolation
        if interpolation:
            self.source_index2 = src2

        self.source_index = src1
        self.target_index = list(target_index)
        self.interpolation = interpolation

    def transform(self, results: dict) -> dict:
        """Transforms the keypoint results to match the target keypoints."""
        num_instances = results['keypoints'].shape[0]

        if 'keypoints_visible' not in results:
            results['keypoints_visible'] = np.ones(
                (num_instances, results['keypoints'].shape[1]))

        if len(results['keypoints_visible'].shape) > 2:
            results['keypoints_visible'] = results['keypoints_visible'][:, :,
                                                                        0]

        # Initialize output arrays
        keypoints = np.zeros((num_instances, self.num_keypoints, 3))
        keypoints_visible = np.zeros((num_instances, self.num_keypoints))
        key = 'keypoints_3d' if 'keypoints_3d' in results else 'keypoints'
        c = results[key].shape[-1]

        flip_indices = results.get('flip_indices', None)

        # Create a mask to weight visibility loss
        keypoints_visible_weights = keypoints_visible.copy()
        keypoints_visible_weights[:, self.target_index] = 1.0

        # Interpolate keypoints if pairs of source indexes provided
        if self.interpolation:
            keypoints[:, self.target_index, :c] = 0.5 * (
                results[key][:, self.source_index] +
                results[key][:, self.source_index2])
            keypoints_visible[:, self.target_index] = results[
                'keypoints_visible'][:, self.source_index] * results[
                    'keypoints_visible'][:, self.source_index2]
            # Flip keypoints if flip_indices provided
            if flip_indices is not None:
                for i, (x1, x2) in enumerate(
                        zip(self.source_index, self.source_index2)):
                    idx = flip_indices[x1] if x1 == x2 else i
                    flip_indices[i] = idx if idx < self.num_keypoints else i
                flip_indices = flip_indices[:len(self.source_index)]
        # Otherwise just copy from the source index
        else:

            keypoints[:,
                      self.target_index, :c] = results[key][:,
                                                            self.source_index]
            keypoints_visible[:, self.target_index] = results[
                'keypoints_visible'][:, self.source_index]

        # Update the results dict
        results['keypoints'] = keypoints[..., :2]
        results['keypoints_visible'] = np.stack(
            [keypoints_visible, keypoints_visible_weights], axis=2)
        if 'keypoints_3d' in results:
            results['keypoints_3d'] = keypoints
            results['lifting_target'] = keypoints[results['target_idx']]
            results['lifting_target_visible'] = keypoints_visible[
                results['target_idx']]
        results['flip_indices'] = flip_indices

        return results