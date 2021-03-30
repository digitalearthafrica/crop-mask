"""
This script exists simply to keep the notebooks
'Extract_training_data.ipynb' and 'Predict.ipynb' tidy by
not clutering them up with custom training data functions.
"""
import os
import os.path as osp
import re
import pyproj
import dask
import hdstats
import datacube
import numpy as np
import sys
import xarray as xr
import warnings
import dask.array as da
from datacube.utils.geometry import assign_crs
from datacube.testutils.io import rio_slurp_xarray
from odc.algo import randomize, reshape_for_geomedian, xr_reproject, xr_geomedian
from odc.algo._dask import reshape_yxbt
from typing import List, Optional, Dict, Tuple
from pyproj import Proj, transform

sys.path.append('../../Scripts')
from deafrica_bandindices import calculate_indices
from deafrica_temporal_statistics import xr_phenology, temporal_statistics
from deafrica_classificationtools import HiddenPrints
from deafrica_datahandling import load_ard

warnings.filterwarnings("ignore")


def gm_mads_two_seasons_training(ds):
    dc = datacube.Datacube(app='training')
    ds = ds / 10000
    ds1 = ds.sel(time=slice('2019-01', '2019-06'))
    ds2 = ds.sel(time=slice('2019-07', '2019-12')) 

    def fun(ds, era):
        #geomedian and tmads
        gm_mads = xr_geomedian_tmad(ds)
        gm_mads = calculate_indices(gm_mads,
                               index=['NDVI','LAI','MNDWI'],
                               drop=False,
                               normalise=False,
                               collection='s2')
        
        gm_mads['sdev'] = -np.log(gm_mads['sdev'])
        gm_mads['bcdev'] = -np.log(gm_mads['bcdev'])
        gm_mads['edev'] = -np.log(gm_mads['edev'])
        
        #rainfall climatology
        if era == '_S1':
            chirps = assign_crs(xr.open_rasterio('/g/data/CHIRPS/cumulative_alltime/CHPclim_jan_jun_cumulative_rainfall.nc'),  crs='epsg:4326')
        if era == '_S2':
            chirps = assign_crs(xr.open_rasterio('/g/data/CHIRPS/cumulative_alltime/CHPclim_jul_dec_cumulative_rainfall.nc'),  crs='epsg:4326')
        
        chirps = xr_reproject(chirps,ds.geobox,"bilinear")
        gm_mads['rain'] = chirps
        
        for band in gm_mads.data_vars:
            gm_mads = gm_mads.rename({band:band+era})
        
        return gm_mads
    
    epoch1 = fun(ds1, era='_S1')
    epoch2 = fun(ds2, era='_S2')
    
    #slope
    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    slope = rio_slurp_xarray(url_slope, gbox=ds.geobox)
    slope = slope.to_dataset(name='slope')
    
    result = xr.merge([epoch1,
                       epoch2,
                       slope],compat='override')

    return result.squeeze()

def gm_mads_two_seasons_predict(ds):
    dc = datacube.Datacube(app='training')
    ds = ds / 10000
    ds1 = ds.sel(time=slice('2019-01', '2019-06'))
    ds2 = ds.sel(time=slice('2019-07', '2019-12'))

    def fun(ds, era):
        #geomedian and tmads
        #gm_mads = xr_geomedian_tmad(ds)
        gm_mads = xr_geomedian_tmad_new(ds).compute()
        gm_mads = calculate_indices(gm_mads,
                               index=['NDVI','LAI','MNDWI'],
                               drop=False,
                               normalise=False,
                               collection='s2')
        
        gm_mads['sdev'] = -np.log(gm_mads['sdev'])
        gm_mads['bcdev'] = -np.log(gm_mads['bcdev'])
        gm_mads['edev'] = -np.log(gm_mads['edev'])
        gm_mads = gm_mads.chunk({'x':1000,'y':1000})
        
        #rainfall climatology
        if era == '_S1':
            chirps = assign_crs(xr.open_rasterio('/g/data/CHIRPS/cumulative_alltime/CHPclim_jan_jun_cumulative_rainfall.nc'),
                                crs='epsg:4326')
        if era == '_S2':
            chirps = assign_crs(xr.open_rasterio('/g/data/CHIRPS/cumulative_alltime/CHPclim_jul_dec_cumulative_rainfall.nc'),
                                crs='epsg:4326')
        
        chirps = xr_reproject(chirps,ds.geobox,"bilinear")
        chirps = chirps.chunk({'x':1000,'y':1000})
        gm_mads['rain'] = chirps
        
        for band in gm_mads.data_vars:
            gm_mads = gm_mads.rename({band:band+era})
        
        return gm_mads
    
    epoch1 = fun(ds1, era='_S1')
    epoch2 = fun(ds2, era='_S2')
    
    #slope
    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    slope = rio_slurp_xarray(url_slope, gbox=ds.geobox)
    slope = slope.to_dataset(name='slope').chunk({'x':1000,'y':1000})
    
    result = xr.merge([epoch1,
                       epoch2,
                       slope],compat='override')

    return result.squeeze()


def xr_geomedian_tmad(ds, axis='time', where=None, **kw):
    """
    :param ds: xr.Dataset|xr.DataArray|numpy array
    Other parameters:
    **kwargs -- passed on to pcm.gnmpcm
       maxiters   : int         1000
       eps        : float       0.0001
       num_threads: int| None   None
    """

    import hdstats
    def gm_tmad(arr, **kw):
        """
        arr: a high dimensional numpy array where the last dimension will be reduced. 
    
        returns: a numpy array with one less dimension than input.
        """
        gm = hdstats.nangeomedian_pcm(arr, **kw)
        nt = kw.pop('num_threads', None)
        emad = hdstats.emad_pcm(arr, gm, num_threads=nt)[:,:, np.newaxis]
        smad = hdstats.smad_pcm(arr, gm, num_threads=nt)[:,:, np.newaxis]
        bcmad = hdstats.bcmad_pcm(arr, gm, num_threads=nt)[:,:, np.newaxis]
        return np.concatenate([gm, emad, smad, bcmad], axis=-1)


    def norm_input(ds, axis):
        if isinstance(ds, xr.DataArray):
            xx = ds
            if len(xx.dims) != 4:
                raise ValueError("Expect 4 dimensions on input: y,x,band,time")
            if axis is not None and xx.dims[3] != axis:
                raise ValueError(f"Can only reduce last dimension, expect: y,x,band,{axis}")
            return None, xx, xx.data
        elif isinstance(ds, xr.Dataset):
            xx = reshape_for_geomedian(ds, axis)
            return ds, xx, xx.data
        else:  # assume numpy or similar
            xx_data = ds
            if xx_data.ndim != 4:
                raise ValueError("Expect 4 dimensions on input: y,x,band,time")
            return None, None, xx_data

    kw.setdefault('nocheck', False)
    kw.setdefault('num_threads', 1)
    kw.setdefault('eps', 1e-6)

    ds, xx, xx_data = norm_input(ds, axis)
    is_dask = dask.is_dask_collection(xx_data)

    if where is not None:
        if is_dask:
            raise NotImplementedError("Dask version doesn't support output masking currently")

        if where.shape != xx_data.shape[:2]:
            raise ValueError("Shape for `where` parameter doesn't match")
        set_nan = ~where
    else:
        set_nan = None

    if is_dask:
        if xx_data.shape[-2:] != xx_data.chunksize[-2:]:
            xx_data = xx_data.rechunk(xx_data.chunksize[:2] + (-1, -1))

        data = da.map_blocks(lambda x: gm_tmad(x, **kw),
                             xx_data,
                             name=randomize('geomedian'),
                             dtype=xx_data.dtype, 
                             chunks=xx_data.chunks[:-2] + (xx_data.chunks[-2][0]+3,),
                             drop_axis=3)
    else:
        data = gm_tmad(xx_data, **kw)

    if set_nan is not None:
        data[set_nan, :] = np.nan

    if xx is None:
        return data

    dims = xx.dims[:-1]
    cc = {k: xx.coords[k] for k in dims}
    cc[dims[-1]] = np.hstack([xx.coords[dims[-1]].values,['edev', 'sdev', 'bcdev']])
    xx_out = xr.DataArray(data, dims=dims, coords=cc)

    if ds is None:
        xx_out.attrs.update(xx.attrs)
        return xx_out

    ds_out = xx_out.to_dataset(dim='band')
    for b in ds.data_vars.keys():
        src, dst = ds[b], ds_out[b]
        dst.attrs.update(src.attrs)

    return assign_crs(ds_out, crs=ds.geobox.crs)

# def merge_tifs_into_ds(
#     root_fld: str,
#     tifs: List[str],
#     rename_dict: Optional[Dict] = None,
#     tifs_min_num=8,
# ) -> xr.Dataset:
#     """
#     Will be replaced with dc.load(gm_6month) once they've been produced.
    
#     use os.walk to get the all files under a folder, it just merge the half year tifs.
#     We need combine two half-year tifs ds and add (calculated indices, rainfall, and slope)
#     @param tifs: tifs with the bands
#     @param root_fld: the parent folder for the sub_fld
#     @param tifs_min_num: geo-median tifs is 16 a tile idx
#     @param rename_dict: we can put the rename dictionary here
#     @return:
#     """
#     assert len(tifs) > tifs_min_num
#     cache = []
#     for tif in tifs:
#         if tif.endswith(".tif"):
#             band_name = re.search(r"_([A-Za-z0-9]+).tif", tif).groups()[0]
#             if band_name in ["rgba", "COUNT"]:
#                 continue

#             band_array = assign_crs(xr.open_rasterio(osp.join(root_fld, tif)).squeeze().to_dataset(name=band_name), crs='epsg:6933')
#             cache.append(band_array)
#     # clean up output
#     output = xr.merge(cache).squeeze()
#     output = output.drop(["band"])
    
#     return output.rename(rename_dict) if rename_dict else output

# def get_tifs_paths(dirname, subfld):
#     """
#     generated src tifs dictionnary, season on and two, or more seasons
#     @param dirname:
#     @param subfld:
#     @return:
#     """
#     all_tifs = os.walk(osp.join(dirname, subfld))
    
#     return dict(
#         (l1_dir, l1_files)
#         for level, (l1_dir, _, l1_files) in enumerate(all_tifs)
#         if level > 0
#     )

# def features(ds, era):
#     #normalise SR and edev bands
#     for band in ds.data_vars:
#         if band not in ['sdev', 'bcdev']:
#             ds[band] = ds[band] / 10000

#     gm_mads = calculate_indices(ds,
#                            index=['NDVI','LAI','MNDWI'],
#                            drop=False,
#                            normalise=False,
#                            collection='s2')

#     gm_mads['sdev'] = -np.log(gm_mads['sdev'])
#     gm_mads['bcdev'] = -np.log(gm_mads['bcdev'])
#     gm_mads['edev'] = -np.log(gm_mads['edev'])

#     #rainfall climatology
#     if era == '_S1':
#         chirps = assign_crs(xr.open_rasterio('/g/data/CHIRPS/cumulative_alltime/CHPclim_jan_jun_cumulative_rainfall.nc'),
#                             crs='epsg:4326')
            
#     if era == '_S2':
#         chirps = assign_crs(xr.open_rasterio('/g/data/CHIRPS/cumulative_alltime/CHPclim_jul_dec_cumulative_rainfall.nc'),
#                             crs='epsg:4326')

#     #Clip CHIRPS to ~ S2 tile boundaries so we can handle NaNs local to S2 tile
#     xmin, xmax = ds.x.values[0], ds.x.values[-1]
#     ymin, ymax = ds.y.values[0], ds.y.values[-1]
#     inProj = Proj('epsg:6933')
#     outProj = Proj('epsg:4326')
#     xmin,ymin = transform(inProj,outProj,xmin,ymin)
#     xmax,ymax = transform(inProj,outProj,xmax,ymax)

#     #create lat/lon indexing slices - buffer S2 bbox by 0.05deg
#     if (xmin < 0) & (xmax < 0):
#         x_slice=list(np.arange(xmin+0.05, xmax-0.05, -0.05))
#     else:
#         x_slice=list(np.arange(xmax-0.05, xmin+0.05, 0.05))
    
#     if (ymin < 0) & (ymax < 0):
#         y_slice=list(np.arange(ymin+0.05, ymax-0.05, -0.05))
#     else:
#         y_slice=list(np.arange(ymin-0.05, ymax+0.05, 0.05))
    
#     #index global chirps using buffered s2 tile bbox
#     chirps=assign_crs(chirps.sel(x=y_slice,y=x_slice, method='nearest'))
    
#     #fill any NaNs in CHIRPS with local (s2-tile bbox) mean
#     chirps=chirps.fillna(chirps.mean())
    
#     #reproject to match satellite data
#     chirps = xr_reproject(chirps,ds.geobox,"bilinear")
#     gm_mads['rain'] = chirps

#     for band in gm_mads.data_vars:
#         gm_mads = gm_mads.rename({band:band+era})

#     return gm_mads

# def gm_mads_two_seasons_production(x,y):
#     """
#     Feature layer function for production run of
#     eastern crop-mask
#     """            
#     rename_dict = {
#         "B02": "blue",
#         "B03": "green",
#         "B04": "red",
#         "B05": "red_edge_1",
#         "B06": "red_edge_2",
#         "B07": "red_edge_3",
#         "B08": "nir",
#         "B8A": "nir_narrow",
#         "B11": "swir_1",
#         "B12": "swir_2",
#         "BCMAD": "bcdev",
#         "EMAD": "edev",
#         "SMAD": "sdev",
#     }
    
#     training_features = [
#         "red_S1",
#         "blue_S1",
#         "green_S1",
#         "nir_S1",
#         "swir_1_S1",
#         "swir_2_S1",
#         "red_edge_1_S1",
#         "red_edge_2_S1",
#         "red_edge_3_S1",
#         "edev_S1",
#         "sdev_S1",
#         "bcdev_S1",
#         "NDVI_S1",
#         "LAI_S1",
#         "MNDWI_S1",
#         "rain_S1",
#         "red_S2",
#         "blue_S2",
#         "green_S2",
#         "nir_S2",
#         "swir_1_S2",
#         "swir_2_S2",
#         "red_edge_1_S2",
#         "red_edge_2_S2",
#         "red_edge_3_S2",
#         "edev_S2",
#         "sdev_S2",
#         "bcdev_S2",
#         "NDVI_S2",
#         "LAI_S2",
#         "MNDWI_S2",
#         "rain_S2",
#         "slope",
#     ]
    
#     DATA_PATH = "/g/data/u23/data/"
#     TIF_path = osp.join(DATA_PATH, "tifs20")
#     subfld = "x{x:+04d}/y{y:+04d}/".format(x=x, y=y)
#     P6M_tifs = get_tifs_paths(TIF_path, subfld)
    
#     seasoned_ds = {}
#     for k, tifs in P6M_tifs.items():
#         era = "_S1" if "2019-01--P6M" in k else "_S2"
#         base_ds = merge_tifs_into_ds(
#                 k, tifs, rename_dict=rename_dict
#             )
        
#         seasoned_ds[era]=base_ds   
    
#     #convert from bands to features
#     epoch1 = features(seasoned_ds['_S1'], era='_S1')
#     epoch2 = features(seasoned_ds['_S2'], era='_S2')

#     #append slope
#     url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
#     slope = rio_slurp_xarray(url_slope, epoch2.geobox)
#     slope = slope.to_dataset(name='slope')

#     #merge everything
#     result = xr.merge([epoch1,
#                        epoch2,
#                        slope],
#                       compat='override')

#     #order the features correctly
#     result = result[training_features]
#     result = result.astype(np.float32)

#     return result.squeeze()


def xr_geomedian_tmad_new(ds, **kw):
    """
    Same as other one but uses reshape_yxbt instead of
    reshape_for_geomedian
    """

    import hdstats
    def gm_tmad(arr, **kw):
        """
        arr: a high dimensional numpy array where the last dimension will be reduced. 
    
        returns: a numpy array with one less dimension than input.
        """
        gm = hdstats.nangeomedian_pcm(arr, **kw)
        nt = kw.pop('num_threads', None)
        emad = hdstats.emad_pcm(arr, gm, num_threads=nt)[:,:, np.newaxis]
        smad = hdstats.smad_pcm(arr, gm, num_threads=nt)[:,:, np.newaxis]
        bcmad = hdstats.bcmad_pcm(arr, gm, num_threads=nt)[:,:, np.newaxis]
        return np.concatenate([gm, emad, smad, bcmad], axis=-1)


    def norm_input(ds):
        if isinstance(ds, xr.Dataset):
            xx = reshape_yxbt(ds, yx_chunks=500)
            return ds, xx, xx.data

    kw.setdefault('nocheck', False)
    kw.setdefault('num_threads', 1)
    kw.setdefault('eps', 1e-6)

    ds, xx, xx_data = norm_input(ds)
    is_dask = dask.is_dask_collection(xx_data)

    if is_dask:
        data = da.map_blocks(lambda x: gm_tmad(x, **kw),
                             xx_data,
                             name=randomize('geomedian'),
                             dtype=xx_data.dtype, 
                             chunks=xx_data.chunks[:-2] + (xx_data.chunks[-2][0]+3,),
                             drop_axis=3)
    
    dims = xx.dims[:-1]
    cc = {k: xx.coords[k] for k in dims}
    cc[dims[-1]] = np.hstack([xx.coords[dims[-1]].values,['edev', 'sdev', 'bcdev']])
    xx_out = xr.DataArray(data, dims=dims, coords=cc)

    if ds is None:
        xx_out.attrs.update(xx.attrs)
        return xx_out

    ds_out = xx_out.to_dataset(dim='band')
    for b in ds.data_vars.keys():
        src, dst = ds[b], ds_out[b]
        dst.attrs.update(src.attrs)

    return assign_crs(ds_out, crs=ds.geobox.crs)


