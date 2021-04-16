import sys

import datacube
import numpy as np
import xarray as xr
from datacube.testutils.io import rio_slurp_xarray
from datacube.utils.geometry import assign_crs
from odc.algo import xr_reproject
from pyproj import Proj, transform
import os.path as osp
cwd = osp.dirname(__file__)

from deafrica_tools.bandindices import calculate_indices

def gm_mads_two_seasons_prediction(geobox, dask_chunks):
    """
    Feature layer function for production run of
    eastern crop-mask


    """
    dc = datacube.Datacube(app="prediction")
    
    # load the data
    measurements = [
        "blue",
        "green",
        "red",
        "nir",
        "swir_1",
        "swir_2",
        "red_edge_1",
        "red_edge_2",
        "red_edge_3",
        "bcdev",
        "edev",
        "sdev",
    ]
    ds1 = dc.load(
        product="gm_s2_semiannual", time="2019-01", measurements=measurements, like=geobox,
        dask_chunks=dask_chunks
    )
    ds2 = dc.load(
        product="gm_s2_semiannual", time="2019-07", measurements=measurements, like=geobox,
        dask_chunks=dask_chunks
    )

    dss = {"S1": ds1, "S2": ds2}

    def fun(ds, era):
        # normalise SR and edev bands
        for band in ds.data_vars:
            if band not in ["sdev", "bcdev"]:
                ds[band] = ds[band] / 10000

        gm_mads = calculate_indices(
            ds,
            index=["NDVI", "LAI", "MNDWI"],
            drop=False,
            normalise=False,
            collection="s2",
        )

        gm_mads["sdev"] = -np.log(gm_mads["sdev"])
        gm_mads["bcdev"] = -np.log(gm_mads["bcdev"])
        gm_mads["edev"] = -np.log(gm_mads["edev"])

        # rainfall climatology
        if era == "_S1":
            chirps = assign_crs(
                xr.open_rasterio(
                    "/g/data/u23/raw_data/CHIRPS/CHPclim_jan_jun_cumulative_rainfall.nc"
                ),
                crs="epsg:4326",
            )
        if era == "_S2":
            chirps = assign_crs(
                xr.open_rasterio(
                    "/g/data/u23/raw_data/CHIRPS/CHPclim_jul_dec_cumulative_rainfall.nc"
                ),
                crs="epsg:4326",
            )

        # Clip CHIRPS to ~ S2 tile boundaries so we can handle NaNs local to S2 tile
        xmin, xmax = ds.x.values[0], ds.x.values[-1]
        ymin, ymax = ds.y.values[0], ds.y.values[-1]
        inProj = Proj("epsg:6933")
        outProj = Proj("epsg:4326")
        xmin, ymin = transform(inProj, outProj, xmin, ymin)
        xmax, ymax = transform(inProj, outProj, xmax, ymax)

        # create lat/lon indexing slices - buffer S2 bbox by 0.05deg
        if (xmin < 0) & (xmax < 0):
            x_slice = list(np.arange(xmin + 0.05, xmax - 0.05, -0.05))
        else:
            x_slice = list(np.arange(xmax - 0.05, xmin + 0.05, 0.05))

        if (ymin < 0) & (ymax < 0):
            y_slice = list(np.arange(ymin + 0.05, ymax - 0.05, -0.05))
        else:
            y_slice = list(np.arange(ymin - 0.05, ymax + 0.05, 0.05))

        # index global chirps using buffered s2 tile bbox
        chirps = assign_crs(chirps.sel(x=y_slice, y=x_slice, method="nearest"))

        # fill any NaNs in CHIRPS with local (s2-tile bbox) mean
        chirps = chirps.fillna(chirps.mean())
        chirps = xr_reproject(chirps, ds.geobox, "bilinear")
        chirps = chirps.chunk(dask_chunks)
        gm_mads["rain"] = chirps

        for band in gm_mads.data_vars:
            gm_mads = gm_mads.rename({band: band + era})

        return gm_mads

    epoch1 = fun(dss["S1"], era="_S1")
    epoch2 = fun(dss["S1"], era="_S2")

    # slope
    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    slope = rio_slurp_xarray(url_slope, gbox=ds1.geobox)
    slope = slope.to_dataset(name="slope").chunk(dask_chunks)

    result = xr.merge([epoch1, epoch2, slope], compat="override")

    result = result.astype(np.float32)
    return result.squeeze()
