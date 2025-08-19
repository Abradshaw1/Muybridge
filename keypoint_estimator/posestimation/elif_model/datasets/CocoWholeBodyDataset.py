# Copyright (c) OpenMMLab. All rights reserved.
import copy
import numpy as np
import os.path as osp

from itertools import filterfalse
from xtcocotools.coco import COCO
from copy import deepcopy

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .base import BaseDataset

from elif_model.datasets.data_augmentation.data_augmentation import LoadImage, GetBBoxCenterScale, TopDownAffine, GenerateTarget, RandomFlip
from elif_model.datasets.data_augmentation.data_augmentation import RandomBBoxTransform, RandomHalfBody, PhotometricDistortion, Albumentation
from elif_model.datasets.data_augmentation.data_augmentation import KeypointConverter
from elif_model.datasets.parse_pose_metainfo import parse_pose_metainfo
from mmengine.fileio import exists, get_local_path, load

from .transforms import bbox_xywh2xyxy

class CocoWholeBodyDataSet(BaseDataset):
    """Base class for COCO-style datasets.

    Args:
        ann_file (str): Annotation file path. Default: ''.
        bbox_file (str, optional): Detection result file path. If
            ``bbox_file`` is set, detected bboxes loaded from this file will
            be used instead of ground-truth bboxes. This setting is only for
            evaluation, i.e., ignored when ``test_mode`` is ``False``.
            Default: ``None``.
        data_mode (str): Specifies the mode of data samples: ``'topdown'`` or
            ``'bottomup'``. In ``'topdown'`` mode, each data sample contains
            one instance; while in ``'bottomup'`` mode, each data sample
            contains all instances in a image. Default: ``'topdown'``
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Default: ``None``.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Default: ``None``.
        data_prefix (dict, optional): Prefix for training data.
            Default: ``dict(img='')``.
        filter_cfg (dict, optional): Config for filter data. Default: `None`.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Default: ``None`` which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy.
            Default: ``True``.
        pipeline (list, optional): Processing pipeline. Default: [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Default: ``False``.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Default: ``False``.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Default: 1000.
        sample_interval (int, optional): The sample interval of the dataset.
            Default: 1.
    """

    METAINFO: dict = dict()

    def __init__(self,
                 ann_file: str = '',
                 bbox_file: Optional[str] = None,
                 data_mode: str = 'topdown',
                 data_root: Optional[str] = None,
                 codec_cfg: Optional[dict] = None,
                 input_size: Tuple[int, int] = (256, 192),
                 data_prefix: dict = dict(img=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 sample_interval: int = 1,
                 num_keypoints: int = 26,
                 keypoint_mapping: Optional[dict] = None,
                 dataset_type: Optional[str] = None,
                 meta_cfg_file: Optional[str] = None,
                 mode: str = 'train'):

        if data_mode not in {'topdown', 'bottomup'}:
            raise ValueError(
                f'{self.__class__.__name__} got invalid data_mode: '
                f'{data_mode}. Should be "topdown" or "bottomup".')
        self.data_mode = data_mode

        if bbox_file:
            if self.data_mode != 'topdown':
                raise ValueError(
                    f'{self.__class__.__name__} is set to {self.data_mode}: '
                    'mode, while "bbox_file" is only '
                    'supported in topdown mode.')

            if not test_mode:
                raise ValueError(
                    f'{self.__class__.__name__} has `test_mode==False` '
                    'while "bbox_file" is only '
                    'supported when `test_mode==True`.')
        self.bbox_file = bbox_file
        self.sample_interval = sample_interval
        self.input_size = input_size    
        self.codec_cfg = codec_cfg
        self.keypoint_mapping = keypoint_mapping
        self.num_keypoints = num_keypoints
        self.mode = mode
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            dataset_name=dataset_type)

        self._metainfo = parse_pose_metainfo(cfg_file=meta_cfg_file)

        if self.test_mode:
            # save the ann_file into MessageHub for CocoMetric
            print("IMPLEMENT TESTING LATER")
            #message = MessageHub.get_current_instance()
            #dataset_name = self.metainfo['dataset_name']
            #message.update_info_dict(
            #    {f'{dataset_name}_ann_file': self.ann_file})

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        :class:`BaseCocoStyleDataset` overrides this method from
        :class:`mmengine.dataset.BaseDataset` to add the metainfo into
        the ``data_info`` before it is passed to the pipeline.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)

        # Mixed image transformations require multiple source images for
        # effective blending. Therefore, we assign the 'dataset' field in
        # `data_info` to provide these auxiliary images.
        # Note: The 'dataset' assignment should not occur within the
        # `get_data_info` function, as doing so may cause the mixed image
        # transformations to stall or hang.
        data_info['dataset'] = self
        data_info['input_size'] = self.input_size

        if self.mode == 'train':
            # Train Pipeline
            data_info = LoadImage(self.data_root, data_info)

            #data_info = RandomFlip(prob = 0.5, results=data_info)
            data_info = GetBBoxCenterScale(data_info)
            data_info = KeypointConverter(num_keypoints=self.num_keypoints, mapping=self.keypoint_mapping).transform(data_info)

            # Define which keypoints are upper body keypoints and which are lower body keypoints
            data_info['upper_body_ids'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18]  # Head, Neck, Upper Body, Arms
            data_info['lower_body_ids'] = [11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25]	# Lower Body, Legs, Torso

            data_info = RandomHalfBody().transform(data_info)
            data_info = RandomBBoxTransform(scale_factor=[0.6, 1.4], rotate_factor=80).transform(data_info)
            data_info = TopDownAffine(data_info, self.input_size)
            data_info = PhotometricDistortion().transform(data_info) 
            transforms=[
                dict(type='Blur', p=0.1),
                dict(type='MedianBlur', p=0.1),
                dict(
                    type='CoarseDropout',
                    max_holes=1,
                    max_height=0.4,
                    max_width=0.4,
                    min_holes=1,
                    min_height=0.2,
                    min_width=0.2,
                    p=1.0),
            ]
            data_info = Albumentation(transforms).transform(data_info)   
            data_info = GenerateTarget(data_info, self.codec_cfg, use_dataset_keypoint_weights=False)
        else:     
            # Validation Pipeline
            data_info = LoadImage(self.data_root, data_info)
            data_info = GetBBoxCenterScale(data_info)

            data_info = KeypointConverter(num_keypoints=self.num_keypoints, mapping=self.keypoint_mapping).transform(data_info)
            data_info = TopDownAffine(data_info, self.input_size)
            data_info = GenerateTarget(data_info, self.codec_cfg, use_dataset_keypoint_weights=False)

        return data_info

    def get_data_info(self, idx: int) -> dict:
        """Get data info by index.

        Args:
            idx (int): Index of data info.

        Returns:
            dict: Data info.
        """
        data_info = super().get_data_info(idx)


        # Add metainfo items that are required in the pipeline and the model
        metainfo_keys = [
            'dataset_name', 'upper_body_ids', 'lower_body_ids', 'flip_pairs',
            'dataset_keypoint_weights', 'flip_indices', 'skeleton_links'
        ]

        for key in metainfo_keys:
            assert key not in data_info, (
                f'"{key}" is a reserved key for `metainfo`, but already '
                'exists in the `data_info`.')
        
            data_info[key] = deepcopy(self._metainfo[key])
        return data_info

    def load_data_list(self) -> List[dict]:
        """Load data list from COCO annotation file or person detection result
        file."""

        if self.bbox_file:
            data_list = self._load_detection_results()
        else:
            instance_list, image_list = self._load_annotations()

        data_list = self._get_topdown_data_infos(instance_list)

        return data_list

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        """Load data from annotations in COCO format."""

        assert osp.isfile(self.ann_file), (
            f'Annotation file `{self.ann_file}`does not exist')

        with get_local_path(self.ann_file) as local_path:
            self.coco = COCO(local_path)
        # set the metainfo about categories, which is a list of dict
        # and each dict contains the 'id', 'name', etc. about this category

        instance_list = []
        image_list = []

        for img_id in self.coco.getImgIds():
            if img_id % self.sample_interval != 0:
                continue
            img = self.coco.loadImgs(img_id)[0]
            img.update({
                'img_id':
                img_id,
                'img_path':
                osp.join(self.data_prefix['img'], img['file_name']),
            })
            image_list.append(img)

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            for ann in self.coco.loadAnns(ann_ids):

                instance_info = self.parse_data_info(
                    dict(raw_ann_info=ann, raw_img_info=img))

                # skip invalid instance annotation.
                if not instance_info:
                    continue

                instance_list.append(instance_info)
        return instance_list, image_list

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        """Parse raw COCO annotation of an instance.

        Args:
            raw_data_info (dict): Raw data information loaded from
                ``ann_file``. It should have following contents:

                - ``'raw_ann_info'``: Raw annotation of an instance
                - ``'raw_img_info'``: Raw information of the image that
                    contains the instance

        Returns:
            dict | None: Parsed instance annotation
        """

        ann = raw_data_info['raw_ann_info']
        img = raw_data_info['raw_img_info']

        # filter invalid instance
        if 'bbox' not in ann or 'keypoints' not in ann:
            return None

        img_w, img_h = img['width'], img['height']

        # get bbox in shape [1, 4], formatted as xywh
        x, y, w, h = ann['bbox']
        x1 = np.clip(x, 0, img_w - 1)
        y1 = np.clip(y, 0, img_h - 1)
        x2 = np.clip(x + w, 0, img_w - 1)
        y2 = np.clip(y + h, 0, img_h - 1)

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        if ann.get('foot_kpts', None) is not None:
            ann['keypoints'] = ann['keypoints'] + ann['foot_kpts']
            ann['num_keypoints'] = len(ann['keypoints']) // 3
        # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
        _keypoints = np.array(
            ann['keypoints'], dtype=np.float32).reshape(1, -1, 3)
        keypoints = _keypoints[..., :2]
        keypoints_visible = np.minimum(1, _keypoints[..., 2])

        if 'num_keypoints' in ann:
            num_keypoints = ann['num_keypoints']
        else:
            num_keypoints = np.count_nonzero(keypoints.max(axis=2))

        if 'area' in ann:
            area = np.array(ann['area'], dtype=np.float32)
        else:
            area = np.clip((x2 - x1) * (y2 - y1) * 0.53, a_min=1.0, a_max=None)
            area = np.array(area, dtype=np.float32)

        data_info = {
            'img_id': ann['image_id'],
            'img_path': img['img_path'],
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
            'num_keypoints': num_keypoints,
            'keypoints': keypoints,
            'keypoints_visible': keypoints_visible,
            'area': area,
            'iscrowd': ann.get('iscrowd', 0),
            'segmentation': ann.get('segmentation', None),
            'id': ann['id'],
            'category_id': np.array(ann['category_id']),
            # store the raw annotation of the instance
            # it is useful for evaluation without providing ann_file
            'raw_ann_info': copy.deepcopy(ann),
        }

        if 'crowdIndex' in img:
            data_info['crowd_index'] = img['crowdIndex']

        return data_info

    @staticmethod
    def _is_valid_instance(data_info: Dict) -> bool:
        """Check a data info is an instance with valid bbox and keypoint
        annotations."""
        # crowd annotation
        if 'iscrowd' in data_info and data_info['iscrowd']:
            return False
        # invalid keypoints
        if 'num_keypoints' in data_info and data_info['num_keypoints'] == 0:
            return False
        # invalid bbox
        if 'bbox' in data_info:
            bbox = data_info['bbox'][0]
            w, h = bbox[2:4] - bbox[:2]
            if w <= 0 or h <= 0:
                return False
        # invalid keypoints
        if 'keypoints' in data_info:
            if np.max(data_info['keypoints']) <= 0:
                return False
        return True

    def _get_topdown_data_infos(self, instance_list: List[Dict]) -> List[Dict]:
        """Organize the data list in top-down mode."""
        # sanitize data samples
        data_list_tp = list(filter(self._is_valid_instance, instance_list))

        return data_list_tp


    def _load_detection_results(self) -> List[dict]:
        """Load data from detection results with dummy keypoint annotations."""

        assert exists(self.ann_file), (
            f'Annotation file `{self.ann_file}` does not exist')
        assert exists(
            self.bbox_file), (f'Bbox file `{self.bbox_file}` does not exist')
        # load detection results
        det_results = load(self.bbox_file)
        
        #assert is_list_of(
        #    det_results,
        #    dict), (f'BBox file `{self.bbox_file}` should be a list of dict, '
        #            f'but got {type(det_results)}')

        # load coco annotations to build image id-to-name index
        with get_local_path(self.ann_file) as local_path:
            self.coco = COCO(local_path)
        # set the metainfo about categories, which is a list of dict
        # and each dict contains the 'id', 'name', etc. about this category
        self._metainfo['CLASSES'] = self.coco.loadCats(self.coco.getCatIds())

        num_keypoints = self.metainfo['num_keypoints']
        data_list = []
        id_ = 0
        for det in det_results:
            # remove non-human instances
            if det['category_id'] != 1:
                continue

            img = self.coco.loadImgs(det['image_id'])[0]

            img_path = osp.join(self.data_prefix['img'], img['file_name'])
            bbox_xywh = np.array(
                det['bbox'][:4], dtype=np.float32).reshape(1, 4)
            bbox = bbox_xywh2xyxy(bbox_xywh)
            bbox_score = np.array(det['score'], dtype=np.float32).reshape(1)

            # use dummy keypoint location and visibility
            keypoints = np.zeros((1, num_keypoints, 2), dtype=np.float32)
            keypoints_visible = np.ones((1, num_keypoints), dtype=np.float32)

            data_list.append({
                'img_id': det['image_id'],
                'img_path': img_path,
                'img_shape': (img['height'], img['width']),
                'bbox': bbox,
                'bbox_score': bbox_score,
                'keypoints': keypoints,
                'keypoints_visible': keypoints_visible,
                'id': id_,
            })

            id_ += 1

        return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg. Defaults return full
        ``data_list``.

        If 'bbox_score_thr` in filter_cfg, the annotation with bbox_score below
        the threshold `bbox_score_thr` will be filtered out.
        """

        data_list = self.data_list

        if self.filter_cfg is None:
            return data_list

        # filter out annotations with a bbox_score below the threshold
        if 'bbox_score_thr' in self.filter_cfg:

            if self.data_mode != 'topdown':
                raise ValueError(
                    f'{self.__class__.__name__} is set to {self.data_mode} '
                    'mode, while "bbox_score_thr" is only supported in '
                    'topdown mode.')

            thr = self.filter_cfg['bbox_score_thr']
            data_list = list(
                filterfalse(lambda ann: ann['bbox_score'] < thr, data_list))

        return data_list