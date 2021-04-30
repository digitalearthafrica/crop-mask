import os
import shutil

import gdal
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from datacube import Datacube
from datacube.testutils.io import rio_slurp_xarray
from datacube.utils.cog import write_cog
from datacube.utils.geometry import GeoBox
from datacube.utils.geometry import assign_crs
from deafrica_tools.classification import HiddenPrints
from deafrica_tools.spatial import xr_rasterize
from odc.algo import xr_reproject
from rsgislib.segmentation import segutils
from scipy.ndimage.measurements import _stats


def post_processing(
    predicted: xr.Dataset,
    geobox_used: GeoBox,
) -> xr.DataArray:
    """
    filter prediction results with post processing filters.
    :param predicted: The prediction results
    :param geobox_used: Geobox used to generate the prediciton feature

    """

    dc = Datacube(app=__name__)

    # create gdf from geom to help with masking
    df = pd.DataFrame({"col1": [0]})
    df["geometry"] = geobox_used.extent.geom
    gdf = gpd.GeoDataFrame(df, geometry=df["geometry"], crs=geobox_used.crs)

    # Mask dataset to set pixels outside the polygon to `NaN`
    with HiddenPrints():
        mask = xr_rasterize(gdf, predicted)
    predicted = predicted.where(mask).astype("float32")

    # mask with WDPA
    wdpa = xr.open_rasterio("/g/data/crop_mask_eastern_data/WDPA_eastern.tif").squeeze()
    wdpa = xr_reproject(wdpa, predicted.geobox, "nearest")
    wdpa = wdpa.astype(bool)
    predicted = predicted.compute().where(~wdpa).astype("float32")

    # write out ndvi for image seg
    ndvi = assign_crs(predicted[["NDVI_S1", "NDVI_S2"]], crs=predicted.geobox.crs)
    write_cog(ndvi.to_array(), "Eastern_tile_NDVI.tif", overwrite=True)

    # grab predictions and proba for post process filtering
    predict = predicted.Predictions
    proba = predicted.Probabilities
    proba = proba.where(predict == 1, 100 - proba)  # crop proba only

    # -----------------image seg---------------------------------------------
    print("  image segmentation...")
    # store temp files somewhere
    directory = "tmp"
    if not os.path.exists(directory):
        os.mkdir(directory)

    tmp = "tmp/"

    # inputs to image seg
    tiff_to_segment = "Eastern_tile_NDVI.tif"
    kea_file = "Eastern_tile_NDVI.kea"
    segmented_kea_file = "Eastern_tile_segmented.kea"

    # convert tiff to kea
    gdal.Translate(
        destName=kea_file, srcDS=tiff_to_segment, format="KEA", outputSRS="EPSG:6933"
    )

    # run image seg
    with HiddenPrints():
        segutils.runShepherdSegmentation(
            inputImg=kea_file,
            outputClumps=segmented_kea_file,
            tmpath=tmp,
            numClusters=60,
            minPxls=100,
        )

    # open segments
    segments = xr.open_rasterio(segmented_kea_file).squeeze().values

    # calculate mode
    print("  calculating mode...")
    count, _sum = _stats(predict, labels=segments, index=segments)
    mode = _sum > (count / 2)
    mode = xr.DataArray(
        mode, coords=predict.coords, dims=predict.dims, attrs=predict.attrs
    )

    # remove the tmp folder
    shutil.rmtree(tmp)
    os.remove(kea_file)
    os.remove(segmented_kea_file)
    os.remove(tiff_to_segment)

    # --Post processing---------------------------------------------------------------
    print("  masking with WOfS,slope,elevation")
    # mask with WOFS
    wofs = dc.load(product="ga_ls8c_wofs_2_summary", like=geobox_used)
    wofs = wofs.frequency > 0.2  # threshold
    predict = predict.where(~wofs, 0)
    proba = proba.where(~wofs, 0)
    mode = mode.where(~wofs, 0)

    # mask steep slopes
    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    slope = rio_slurp_xarray(url_slope, gbox=predicted.geobox)
    slope = slope > 35
    predict = predict.where(~slope, 0)
    proba = proba.where(~slope, 0)
    mode = mode.where(~slope, 0)

    # mask where the elevation is above 3600m
    elevation = dc.load(product="dem_srtm", like=predicted.geobox)
    elevation = elevation.elevation > 3600  # threshold
    predict = predict.where(~elevation.squeeze(), 0)
    proba = proba.where(~elevation.squeeze(), 0)
    mode = mode.where(~elevation.squeeze(), 0)

    # set dtype
    predict = predict.astype(np.int8)
    proba = proba.astype(np.float32)
    mode = mode.astype(np.int8)

    return predict, proba, mode
