import xarray as xr
import geopandas as gpd
import pandas as pd
from datacube import Datacube
from datacube.utils.geometry import GeoBox
from datacube.utils.geometry import Geometry

import sys
import os
import gdal
import shutil
import numpy as np
import subprocess as sp
from joblib import load
from odc.algo import xr_reproject
from datacube.utils import geometry
from datacube.utils.cog import write_cog
from rsgislib.segmentation import segutils
from scipy.ndimage.measurements import _stats
from datacube.utils.geometry import assign_crs
from datacube.testutils.io import rio_slurp_xarray

from dea_ml.config.product_feature_config import FeaturePathConfig
from deafrica_tools.spatial import xr_rasterize
from deafrica_tools.classification import HiddenPrints

def post_processing(
    predicted: xr.Dataset,
    
) -> xr.DataArray:
    """
    filter prediction results with post processing filters.
    :param predicted: The prediction results

    """
    
    dc = Datacube(app=__name__)
    
    #write out ndvi for image seg
    ndvi = assign_crs(predicted[['NDVI_S1', 'NDVI_S2']], crs=predicted.geobox.crs)
    write_cog(ndvi.to_array(), 'Eastern_tile_NDVI.tif',overwrite=True)
    
    #grab predictions and proba for post process filtering
    predict=predicted.Predictions
    proba=predicted.Probabilities
    proba=proba.where(predict==1, 100-proba) #crop proba only
    
    #-----------------image seg---------------------------------------------
    print('  image segmentation...')
    #store temp files somewhere
    directory='tmp'
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    tmp='tmp/'

    #inputs to image seg
    tiff_to_segment = 'Eastern_tile_NDVI.tif'
    kea_file = 'Eastern_tile_NDVI.kea'
    segmented_kea_file = 'Eastern_tile_segmented.kea'

    #convert tiff to kea
    gdal.Translate(destName=kea_file,
                   srcDS=tiff_to_segment,
                   format='KEA',
                   outputSRS='EPSG:6933')
    
    #run image seg
    with HiddenPrints():
        segutils.runShepherdSegmentation(inputImg=kea_file,
                                             outputClumps=segmented_kea_file,
                                             tmpath=tmp,
                                             numClusters=60,
                                             minPxls=100)
    
    #open segments
    segments=xr.open_rasterio(segmented_kea_file).squeeze().values
    
    #calculate mode
    print('  calculating mode...')
    count, _sum =_stats(predict, labels=segments, index=segments)
    mode = _sum > (count/2)
    mode = xr.DataArray(mode, coords=predict.coords, dims=predict.dims, attrs=predict.attrs)
    
    #remove the tmp folder
    shutil.rmtree(tmp)
    os.remove(kea_file)
    os.remove(segmented_kea_file)
    os.remove(tiff_to_segment)
    
    #--Post process masking---------------------------------------------------------------
    print("  masking with AEZ,WDPA,WOfS,slope & elevation")    
    
    # mask out classification beyond AEZ boundary
    gdf = gpd.read_file('data/Eastern.shp').to_crs('EPSG:6933')
    with HiddenPrints():
        mask = xr_rasterize(gdf, predicted)
    predict = predict.where(mask,0)
    proba = proba.where(mask, 0)
    mode = mode.where(mask,0)
    
    # mask with WDPA
    wdpa = xr.open_rasterio('/g/data/crop_mask_eastern_data/WDPA_eastern.tif').squeeze()
    wdpa = xr_reproject(wdpa, predicted.geobox, "nearest")
    wdpa = wdpa.astype(bool)
    predict = predict.where(~wdpa, 0)
    proba = proba.where(~wdpa, 0)
    mode = mode.where(~wdpa, 0)
    
    #mask with WOFS
    wofs=dc.load(product='ga_ls8c_wofs_2_summary',like=predicted.geobox)
    wofs=wofs.frequency > 0.2 # threshold
    predict=predict.where(~wofs, 0)
    proba=proba.where(~wofs, 0)
    mode=mode.where(~wofs, 0)

    #mask steep slopes
    url_slope="https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    slope=rio_slurp_xarray(url_slope, gbox=predicted.geobox)
    slope=slope > 35
    predict=predict.where(~slope, 0)
    proba=proba.where(~slope, 0)
    mode=mode.where(~slope, 0)

    #mask where the elevation is above 3600m
    elevation=dc.load(product='dem_srtm', like=predicted.geobox)
    elevation=elevation.elevation > 3600 # threshold
    predict=predict.where(~elevation.squeeze(), 0)
    proba=proba.where(~elevation.squeeze(), 0)
    mode=mode.where(~elevation.squeeze(), 0)
    
    #set dtype
    predict=predict.astype(np.int8)
    proba=proba.astype(np.float32)
    mode=mode.astype(np.int8)

    return predict, proba, mode
