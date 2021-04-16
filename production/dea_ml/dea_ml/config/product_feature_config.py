import os.path as osp

from dataclasses import dataclass
from odc.stats.model import DateTimeRange, OutputProduct

__PROJ_VERSION__ = "v0.1.8"


@dataclass
class FeaturePathConfig:
    """
    This is a configuration dataclass for the prediction and result stac json.
    The product version will align to the project version in the pyproject.toml file.
    product version and name is critical for stac json
    """

    # change here if you have different version rules for the product name
    PRODUCT_VERSION = __PROJ_VERSION__
    PRODUCT_NAME = "crop_mask_eastern"
    # data path
    DATA_PATH = "/g/data/crop_mask_eastern_data/"
    REMOTE_PATH = "s3://deafrica-data-dev-af/"
    TIF_path = osp.join(DATA_PATH, "tifs10")
    model_path = "https://github.com/digitalearthafrica/crop-mask/blob/main/eastern_cropmask/results/gm_mads_two_seasons_ml_model_20210401.joblib?raw=true"  # noqa
    model_type = "gm_mads_two_seasons"
    tiles_geojson = "https://github.com/digitalearthafrica/crop-mask/blob/main/eastern_cropmask/data/s2_tiles_eastern_aez.geojson?raw=true"  # noqa
    
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
        "_S1": "/g/data/u23/raw_data/CHIRPS/CHPclim_jan_jun_cumulative_rainfall.nc",
        "_S2": "/g/data/u23/raw_data/CHIRPS/CHPclim_jul_dec_cumulative_rainfall.nc",
    }
    # list the requird feature here
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
    # s1_key, s2_key = "2019-01--P6M", "2019-07--P6M"
    resolution = (-10, 10)
    # the time actually is the time range, required by datacube query
    # the datetime_range is required by OutputProduct of odc-stats model
    time = ("2019-01", "2019-12")
    datetime_range = DateTimeRange(time[0], "P12M")
    output_crs = "epsg:6933"
    # query is required by open datacube
    query = {
        "time": time,
        "resolution": resolution,
        "output_crs": output_crs,
        "group_by": "solar_day",
    }
    # the prd_properties is required by the stac json
    prd_properties = {
        "odc:file_format": "GeoTIFF",
        "odc:producer": "digitalearthafrica.org",
        "odc:product": f"{PRODUCT_NAME}",
        "proj:epsg": 6933,
        "crop-mask-model": osp.basename(model_path),
    }
    # the OutputProduct is required by stac json
    product = OutputProduct(
        name=PRODUCT_NAME,
        version=PRODUCT_VERSION,
        short_name=PRODUCT_NAME,
        location=REMOTE_PATH,  # place holder
        properties=prd_properties,
        measurements=("mask", "prob", "filtered"),
        href=f"https://explorer.digitalearth.africa/products/{PRODUCT_NAME}",
    )

    def __repr__(self):
        return f"<{self.PRODUCT_NAME}>.<{self.PRODUCT_VERSION}>"
