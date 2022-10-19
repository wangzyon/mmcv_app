from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry
import warnings

MODELS = Registry('models', parent=MMCV_MODELS)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
SEPARATORS = MODELS
CLUSTERS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_cluster(cfg):
    """Build cluster."""
    return CLUSTERS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg, **kwargs):
    """Build loss."""
    return LOSSES.build(cfg, default_args=kwargs)


def build_separator(cfg, train_cfg=None, test_cfg=None):
    """Build separator."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn('train_cfg and test_cfg is deprecated, ' 'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return SEPARATORS.build(cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
