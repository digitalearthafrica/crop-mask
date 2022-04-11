import os
import shutil
from typing import Any, Dict, Tuple

import dask
import geopandas as gpd
import numpy as np
import xarray as xr
from datacube import Datacube
from datacube.testutils.io import rio_slurp_xarray
from datacube.utils.cog import write_cog
from datacube.utils.geometry import assign_crs
from deafrica_tools.classification import HiddenPrints
from deafrica_tools.spatial import xr_rasterize
from osgeo import gdal
from rsgislib.segmentation import segutils
from scipy.ndimage._measurements import _stats


@dask.delayed
def image_segmentation(ndvi, predict):
    write_cog(ndvi.to_array().compute(), "NDVI.tif", overwrite=True)

    # store temp files somewhere
    directory = "tmp"
    if not os.path.exists(directory):
        os.mkdir(directory)

    tmp = "tmp/"

    # inputs to image seg
    tiff_to_segment = "NDVI.tif"
    kea_file = "NDVI.kea"
    segmented_kea_file = "segmented.kea"

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

    # convert kea to tif
    kwargs = {
        "outputType": gdal.GDT_Float32,
    }

    gdal.Translate(
        destName=segmented_kea_file[:-3] + "tif",
        srcDS=segmented_kea_file,
        outputSRS="EPSG:6933",
        format="GTiff",
        **kwargs
    )

    # open segments
    segments = xr.open_rasterio(segmented_kea_file[:-3] + "tif").squeeze().values

    # calculate mode
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
    os.remove(segmented_kea_file[:-3] + "tif")

    return mode.chunk({})


def post_processing(
    predicted: xr.Dataset, urls: Dict[str, Any]
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Run the delayed post_processing functions, then create a lazy
    xr.Dataset to satisfy odc-stats
    """
    dc = Datacube(app="Crop mask")

    # Set an explicit NODATA value
    NODATA = 255

    # Grab predictions and probability for post process filtering
    predict = predicted.Predictions
    proba = predicted.Probabilities
    proba = proba.where(predict == 1, 100 - proba)  # crop proba only

    # write out ndvi for image seg
    ndvi = assign_crs(predicted[["NDVI_S1", "NDVI_S2"]], crs=predicted.geobox.crs)

    # call function with dask delayed
    filtered = image_segmentation(ndvi, predict)

    # convert delayed object to dask array
    filtered = dask.array.from_delayed(
        filtered.squeeze(), shape=predict.shape, dtype=np.uint8
    )

    # convert dask array to xr.Datarray
    filtered = xr.DataArray(filtered, coords=predict.coords, attrs=predict.attrs)

    # --Post process masking----------------------------------------

    # merge back together for masking
    output_bands = {"mask": predict, "prob": proba, "filtered": filtered}
    ds = xr.Dataset(output_bands)
    # Set nodata attr per band
    for name in output_bands.keys():
        ds[name].attrs["nodata"] = NODATA

    # Mask out classification beyond region boundary
    gdf = gpd.read_file(urls["aez"])
    with HiddenPrints():
        gdf_mask = xr_rasterize(gdf, predicted)
    gdf_mask = gdf_mask.chunk({})
    ds = ds.where(gdf_mask, NODATA)

    # Mask with WDPA
    wdpa = rio_slurp_xarray(urls["wdpa"], gbox=predicted.geobox)
    wdpa = wdpa.chunk({})
    wdpa = wdpa.astype(bool)
    ds = ds.where(~wdpa, NODATA)

    # Mask with WOFS
    # TODO: Make the year configurable and threshold configurable
    wofs = dc.load(product="wofs_ls_summary_annual", like=predicted.geobox, dask_chunks={}, time=("2019"))
    wofs = wofs.frequency > 0.20  # threshold
    ds = ds.where(~wofs, NODATA)

    # Mask steep slopes
    # TODO: Make the threshold configurable
    slope = dc.load(product="dem_srtm_deriv", like=predicted.geobox, measurements=["slope"], dask_chunks={})
    slope = slope > 50
    ds = ds.where(~slope.squeeze(), NODATA)

    # Mask where the elevation is above 3600m
    # TODO: Make the threshold configurable
    elevation = dc.load(product="dem_srtm", like=predicted.geobox, dask_chunks={})
    elevation = elevation.elevation > 3600  # threshold
    ds = ds.where(~elevation.squeeze(), NODATA)

    # Clean up datasets that were used for masking
    del gdf, gdf_mask, wdpa, wofs, slope, elevation

    return ds.squeeze()
