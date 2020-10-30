# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from DensePose.densepose.data.datasets import builtin  # just to register data
from DensePose.densepose.config import add_densepose_config, add_hrnet_config, add_dataset_category_config
from DensePose.densepose.densepose_head import ROI_DENSEPOSE_HEAD_REGISTRY
from DensePose.densepose.evaluator import DensePoseCOCOEvaluator
from DensePose.densepose.roi_head import DensePoseROIHeads
from DensePose.densepose.data.structures import DensePoseDataRelative, DensePoseList, DensePoseTransformData
from DensePose.densepose.modeling.test_time_augmentation import (
    DensePoseGeneralizedRCNNWithTTA,
    DensePoseDatasetMapperTTA,
)
from DensePose.densepose.utils.transform import load_from_cfg
from DensePose.densepose.modeling.hrfpn import build_hrfpn_backbone
