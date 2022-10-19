from mmcv.utils import Registry

GENERATOR = Registry('generator')
SIGNAL = Registry('signal')

__all__ = ["SIGNAL", "GENERATOR"]
