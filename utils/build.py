from utils.config import Config, ConfigDict
from geoseg.models.backbones import *
from geoseg.models.segheads import *
from geoseg.models.classifiers import *
from geoseg.models.basemodules import *
from geoseg.losses import *

def build_from_cfg(cfg):
    if not isinstance(cfg, (dict, ConfigDict, Config)):
        raise TypeError(
            f'cfg should be a dict, ConfigDict or Config, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(
                '`cfg` must contain the key "type", '
                f'but got {cfg}')
    obj_type = cfg.pop('type')
    obj_cls = eval(obj_type)
    obj = obj_cls(**cfg)
    return obj