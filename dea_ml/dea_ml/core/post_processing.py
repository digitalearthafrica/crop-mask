from typing import Tuple

import xarray as xr
from datacube import Datacube
from datacube.utils.geometry import GeoBox
from datacube.utils.geometry import Geometry
from dea_ml.config.product_feature_config import FeaturePathConfig


def post_processing(
    data: xr.Dataset,
    predicted: xr.Dataset,
    config: FeaturePathConfig,
    geobox_used: GeoBox,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    filter prediction results with post processing filters.
    :param data: raw data with all features to run prediction
    :param predicted: The prediction results
    :param config:  FeaturePathConfig configureation
    :param geobox_used: Geobox used to generate the prediciton feature
    :return:
    """
    # post prediction filtering
    predict = predicted.Predictions
    query = config.query.copy()
    # Update dc query with geometry
    # geobox_used = self.geobox_dict[(x, y)]
    query["geopolygon"] = Geometry(geobox_used.extent.geom, crs=geobox_used.crs)

    dc = Datacube(app=__name__)
    # mask with WOFS
    # wofs_query = query.pop("measurements")
    wofs = dc.load(product="ga_ls8c_wofs_2_summary", **query)
    wofs = wofs.frequency > 0.2  # threshold
    predict = predict.where(~wofs, 0)

    # mask steep slopes
    slope = data.slope > 35
    predict = predict.where(~slope, 0)

    # mask where the elevation is above 3600m
    query.pop("time")
    elevation = dc.load(product="srtm", **query)
    elevation = elevation.elevation > 3600
    predict = predict.where(~elevation.squeeze(), 0)
    return predict, predict.Probabilities
