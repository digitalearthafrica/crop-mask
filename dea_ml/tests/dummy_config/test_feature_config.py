import os.path as osp

from dea_ml.config.config_parser import parse_config
from dea_ml.config.product_feature_config import FeaturePathConfig


def test_parse_config():
    cwd = osp.dirname(__file__)
    dummy_config = parse_config(osp.join(cwd, "default.config"))
    should_be = FeaturePathConfig()
    assert dummy_config == should_be
