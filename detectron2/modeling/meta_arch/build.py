# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

from detectron2.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    # torch.distributed.init_process_group("nccl", rank=1, world_size=1)
    # model.backbone = torch.nn.parallel.DistributedDataParallel(model.backbone)
    return model
