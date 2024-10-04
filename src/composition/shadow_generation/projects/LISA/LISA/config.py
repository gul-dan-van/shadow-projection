# -*- coding: utf-8 -*-


from src.composition.shadow_generation.network.detectron2.config import CfgNode as CN


def add_lisa_config(cfg):
    """
    Add config for RelationNet.
    """
    _C = cfg

    _C.MODEL.LISA = CN()

