import os
import os.path as osp
import re
from typing import Dict, List, Tuple

from dea_ml.config.product_feature_config import FeaturePathConfig
from dea_ml.core.africa_geobox import AfricaGeobox


def create_features(
    x: int,
    y: int,
    config: FeaturePathConfig,
    geobox_dict: AfricaGeobox,
    feature_func=None,
    dask_chunks={},
):
    """
    Given a dataset (xarray.DataArray or xr.Dataset) and feature layer
    function, return the feature layers as a xarray Dataset, ready for imput
    into the downstream prediction functions.

    Parameters:
    -----------

    :param x: tile index x
    :param y: time inde y
    :param config: FeaturePathConfig containing the model path and product info`et al.
    :param geobox_dict: geobox will calculate the tile geometry from the tile index

    Returns:
    --------
        subfolder path and the xarray dataset of the features

    """
    # this folder naming x, y will change
    subfld = "x{x:03d}/y{y:03d}".format(x=x, y=y)
    geobox = geobox_dict[(x, y)]

    # call the function on the two 6-month gm+tmads
    model_input = feature_func(geobox).chunk(dask_chunks)

    return subfld, geobox, model_input


def get_xy_from_task(taskstr: str) -> Tuple[int, int]:
    """
    extract the x y from task string
    :param taskstr:
    :return:
    """
    x_str, y_str = taskstr.split("/")[:2]
    return int(x_str.replace("x", "")), int(y_str.replace("y", ""))


def extract_dt_from_model_path(path: str) -> str:
    """
    extract date string from the file name
    :param path:
    :return:
    """
    return re.search(r"_(\d{8})", path).groups()[0]


def extract_xy_from_title(title: str) -> Tuple[int, int]:
    """
    split the x, y out from title
    :param title:
    :return:
    """
    x_str, y_str = title.split(",")
    return int(x_str), int(y_str)


def get_tifs_paths(dirname: str, subfld: str) -> Dict[str, List[str]]:
    """
    generated src tifs dictionnary, season on and two, or more seasons
    :param dirname: dir path name
    :param subfld: subfolder in string type
    """
    all_tifs = os.walk(osp.join(dirname, subfld))
    # l0_dir, l0_subfld, _ = all_tifs[0]
    return dict(
        (l1_dir, l1_files)
        for level, (l1_dir, _, l1_files) in enumerate(all_tifs)
        if level > 0 and (".ipynb" not in l1_dir)
    )
