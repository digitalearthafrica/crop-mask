from pyhocon import ConfigFactory
import os.path as osp
from dea_ml.config.product_feature_config import FeaturePathConfig
from odc.stats.model import OutputProduct, DateTimeRange


def parse_config(default: str = None) -> FeaturePathConfig:
    """
    parse the hocon type of config
    :param default: config file path
    :return: parsed data class
    """
    cwd = osp.dirname(__file__)
    config_file_path = default

    if not default:
        config_file_path = osp.join(cwd, "product_feature_paths.config")
    config = ConfigFactory.parse_file(config_file_path)

    prd_properties = dict(config.get("prd_properties").as_plain_ordered_dict())
    product_fields = config.get("product_fields").as_plain_ordered_dict()
    product_fields["properties"] = prd_properties

    output_product = OutputProduct(**product_fields)

    FPConfig = FeaturePathConfig()
    FPConfig.PRODUCT_VERSION = config.get("PRODUCT_VERSION")
    FPConfig.PRODUCT_NAME = config.get("PRODUCT_NAME")
    FPConfig.DATA_PATH = config.get("PATH.DATA_PATH")
    FPConfig.REMOTE_PATH = config.get("PATH.REMOTE_PATH")
    FPConfig.TIF_path = config.get("PATH.TIF_path")
    FPConfig.model_path = config.get("PATH.model_path")
    FPConfig.tiles_geojson = config.get("PATH.tiles_geojson")
    FPConfig.rename_dict = config.get("bands.rename_dict")
    FPConfig.url_slope = config.get("PATH.url_slope")
    FPConfig.rainfall_path = config.get("PATH.rainfall_path").as_plain_ordered_dict()
    FPConfig.training_features = config.get("training_features")
    FPConfig.resolution = config.get("resolution")
    FPConfig.time = config.get("time")
    FPConfig.datetime_range = DateTimeRange(config.get("datetime_range"))
    FPConfig.output_crs = config.get("output_crs")
    FPConfig.query = config.get("query").as_plain_ordered_dict()
    FPConfig.product = output_product

    return FPConfig
