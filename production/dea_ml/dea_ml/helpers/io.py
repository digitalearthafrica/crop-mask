import os.path as osp
from typing import Tuple, Dict, Optional

import requests
from dea_ml.config.product_feature_config import FeaturePathConfig


def prepare_the_io_path(
    config: FeaturePathConfig, tile_indx: str
) -> Tuple[str, Dict[str, str], str]:
    """
    use sandbox local path to mimic the target s3 prefixes. The path follow our nameing rule:
    <product_name>/version/<x>/<y>/<year>/<product_name>_<x>_<y>_<timeperiod>_<band>.<extension>
    the name in config a crop_mask_eastern_product.yaml and the github repo for those proudct config
    :param config: configureation dataclass as the FeaturePathConfig
    :param tile_indx: <x>/<y>
    :return:
    """

    start_year = config.datetime_range.start.year
    tile_year_prefix = f"{tile_indx}/{start_year}"
    file_prefix = f"{config.product.name}/{tile_year_prefix}"

    output_fld = osp.join(
        config.DATA_PATH,
        config.product.name,
        config.product.version,
        tile_year_prefix,
    )

    mask_path = osp.join(
        output_fld,
        file_prefix.replace("/", "_") + "_mask.tif",
    )

    prob_path = osp.join(
        output_fld,
        file_prefix.replace("/", "_") + "_prob.tif",
    )
    
    filtered_path = osp.join(
        output_fld,
        file_prefix.replace("/", "_") + "_filtered.tif",
    )

    paths = {"mask": mask_path, "prob": prob_path, "filtered": filtered_path}

    metadata_path = mask_path.replace("_mask.tif", ".json")

    assert set(paths.keys()) == set(
        config.product.measurements
    ), "file number can not cover the measurement number, \
    each measurement a tif file. pls check the stac json schema requirements."

    return output_fld, paths, metadata_path


def download_file(url: str, local_filename: Optional[str] = None) -> str:
    """
    Download file on github with ```?raw=true```
    https://github....gm_mads_two_seasons_ml_model_20210301.joblib?raw=true
    :param url:
    :param local_filename:
    :return:
    """
    local_filename = (
        local_filename if local_filename else osp.join("/tmp", url.split("/")[-1])
    )
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename
