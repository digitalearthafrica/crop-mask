import os.path as osp
from typing import Tuple, Dict

from dataclasses import dataclass
from odc.stats.model import DateTimeRange, OutputProduct

__PROJ_VERSION__ = "v0.1.5"


@dataclass
class FeaturePathConfig:
    """
    This is a configureation data class for the prediction and result stac json.
    The product version will align to the project version in the pyproject.toml file.
    product version and name is critical for stac json

    """

    # change here if you have different version rules for the product name
    PRODUCT_VERSION = __PROJ_VERSION__
    PRODUCT_NAME = "crop_mask_eastern"
    # data path
    DATA_PATH = "/g/data/u23/data/"
    REMOTE_PATH = "s3://deafrica-data-dev-af/"
    TIF_path = osp.join(DATA_PATH, "tifs20")
    model_path = "/g/data/u23/crop-mask/eastern_cropmask/results/gm_mads_two_seasons_ml_model_20210301.joblib"
    model_type = "gm_mads_two_seasons"
    # if you want to use alias of band keep this, otherwise use None
    rename_dict = {  # "nir_1": "nir",
        "B02": "blue",
        "B03": "green",
        "B04": "red",
        "B05": "red_edge_1",
        "B06": "red_edge_2",
        "B07": "red_edge_3",
        "B08": "nir",
        "B8A": "nir_narrow",
        "B11": "swir_1",
        "B12": "swir_2",
        "BCMAD": "bcdev",
        "EMAD": "edev",
        "SMAD": "sdev",
    }

    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    rainfall_path = {
        "_S1": "/g/data/CHIRPS/cumulative_alltime/CHPclim_jan_jun_cumulative_rainfall.nc",
        "_S2": "/g/data/CHIRPS/cumulative_alltime/CHPclim_jul_dec_cumulative_rainfall.nc",
    }
    s1_key, s2_key = "2019-01--P6M", "2019-07--P6M"
    resolution = (-20, 20)
    time = ("2019-01", "2019-12")
    datetime_range = DateTimeRange(time[0], "P12M")
    output_crs = "epsg:6933"
    query = {
        "time": time,
        "resolution": resolution,
        "output_crs": output_crs,
        "group_by": "solar_day",
    }
    training_features = [
        "red_S1",
        "blue_S1",
        "green_S1",
        "nir_S1",
        "swir_1_S1",
        "swir_2_S1",
        "red_edge_1_S1",
        "red_edge_2_S1",
        "red_edge_3_S1",
        "edev_S1",
        "sdev_S1",
        "bcdev_S1",
        "NDVI_S1",
        "LAI_S1",
        "MNDWI_S1",
        "rain_S1",
        "red_S2",
        "blue_S2",
        "green_S2",
        "nir_S2",
        "swir_1_S2",
        "swir_2_S2",
        "red_edge_1_S2",
        "red_edge_2_S2",
        "red_edge_3_S2",
        "edev_S2",
        "sdev_S2",
        "bcdev_S2",
        "NDVI_S2",
        "LAI_S2",
        "MNDWI_S2",
        "rain_S2",
        "slope",
    ]

    prd_properties = {
        "odc:file_format": "GeoTIFF",
        "odc:producer": "digitalearthafrica.org",
        "odc:product": f"{PRODUCT_NAME}",
        "proj:epsg": 6933,
        "crop-mask-model": osp.basename(model_path),
    }
    product = OutputProduct(
        name=PRODUCT_NAME,
        version=PRODUCT_VERSION,
        short_name=PRODUCT_NAME,
        location=REMOTE_PATH,  # place holder
        properties=prd_properties,
        measurements=("mask", "prob"),
        href=f"https://explorer.digitalearth.africa/products/{PRODUCT_NAME}",
    )


def prepare_the_io_path(
    config: FeaturePathConfig, tile_indx: str
) -> Tuple[str, Dict[str, str], str]:
    """
    use sandbox local path to mimic the target s3 prefixes. The path follow our nameing rule:
    <product_name>/version/<x>/<y>/<year>/<product_name>_<x>_<y>_<timeperiod>_<band>.<extension>
    the name in config a crop_mask_eastern_product.yaml and the github repo for those proudct config
    @param config: configureation dataclass as the FeaturePathConfig
    @param tile_indx: <x>/<y>
    @return:
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

    paths = {"mask": mask_path, "prob": prob_path}

    metadata_path = mask_path.replace("_mask.tif", ".json")

    assert set(paths.keys()) == set(
        config.product.measurements
    ), "file number can not cover the measurement number, \
    each measurement a tif file. pls check the stac json schema requirements."

    return output_fld, paths, metadata_path
