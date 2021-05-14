import os
import shutil
from typing import Tuple, Dict, Any

import gdal
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
from rsgislib.segmentation import segutils
from scipy.ndimage.measurements import _stats


@dask.delayed
def post_processing_delayed(
    predicted: xr.Dataset, urls: Dict[str, Any]
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Decorate the post processing operations with
    dask delayed so they work with the input-->reduce
    model of odc-stats
    """
    dc = Datacube(app=__name__)

    # write out ndvi for image seg
    ndvi = assign_crs(predicted[["NDVI_S1", "NDVI_S2"]], crs=predicted.geobox.crs)
    write_cog(ndvi.to_array(), "Eastern_tile_NDVI.tif", overwrite=True).compute()

    # grab predictions and proba for post process filtering
    predict = predicted.Predictions
    proba = predicted.Probabilities
    proba = proba.where(predict == 1, 100 - proba)  # crop proba only

    # -----------------image seg---------------------------------------------
    print("  image segmentation...")

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

    # --Post process masking----------------------------------------
    print("  masking with AEZ,WDPA,WOfS,slope & elevation")

    # merge back together for masking
    ds = xr.Dataset({"mask": predict, "prob": proba, "filtered": mode})
    ds = ds.chunk({})
    # mask out classification beyond AEZ boundary
    gdf = gpd.read_file(urls["aez"])
    with HiddenPrints():
        mask = xr_rasterize(gdf, predicted)
    mask = mask.chunk({})
    ds = ds.where(mask, 0)

    # mask with WDPA
    wdpa = rio_slurp_xarray(urls["wdpa"], gbox=predicted.geobox)
    wdpa = wdpa.astype(bool).chunk({})
    ds = ds.where(~wdpa, 0)

    # mask with WOFS
    wofs = dc.load(
        product="ga_ls8c_wofs_2_summary", like=predicted.geobox, dask_chunks={}
    )
    wofs = wofs.frequency > 0.2  # threshold
    ds = ds.where(~wofs, 0)

    # mask steep slopes
    slope = rio_slurp_xarray(urls["slope"], gbox=predicted.geobox)
    slope = slope > 35
    ds = ds.where(~slope.chunk({}), 0)

    # mask where the elevation is above 3600m
    elevation = dc.load(product="dem_srtm", like=predicted.geobox, dask_chunks={})
    elevation = elevation.elevation > 3600  # threshold
    ds = ds.where(~elevation.squeeze(), 0)

    # set dtype
    ds["mask"] = ds["mask"].astype(np.int8)
    ds["prob"] = ds["prob"].astype(np.float32)
    ds["filtered"] = ds["filtered"].astype(np.int8)


def post_processing(
    predicted: xr.Dataset, urls: Dict[str, Any]
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:

    """
    Run the delayed post_processing functions
    """
    return dask.delayed(post_processing_delayed(predicted, urls))
