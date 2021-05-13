from typing import Dict, List, Tuple, Any

import numpy as np
import xarray as xr
from datacube.testutils.io import rio_slurp_xarray
from datacube.utils.geometry import assign_crs
from deafrica_tools.bandindices import calculate_indices
from odc.algo import xr_reproject
from odc.algo.io import load_with_native_transform
from odc.stats.model import Task
from pyproj import Proj, transform


def get_xy_from_task(taskstr: str) -> Tuple[int, int]:
    """
    extract the x y from task string
    :param taskstr:
    :return:
    """
    x_str, y_str = taskstr.split("/")[:2]
    return int(x_str.replace("x", "")), int(y_str.replace("y", ""))


def common_ops(ds, era):
    # normalise SR and edev bands
    for band in ds.data_vars:
        if band not in ["sdev", "bcdev"]:
            ds[band] = ds[band] / 10000

    # add indices
    gm_mads = calculate_indices(
        ds,
        index=["NDVI", "LAI", "MNDWI"],
        drop=False,
        normalise=False,
        collection="s2",
    )

    # normalise gms using -log
    gm_mads["sdev"] = -np.log(gm_mads["sdev"])
    gm_mads["bcdev"] = -np.log(gm_mads["bcdev"])
    gm_mads["edev"] = -np.log(gm_mads["edev"])

    return gm_mads


def add_chirps(
    urls: Dict[Any, Any],
    ds: xr.Dataset,
    era: str,
    training: bool = True,
    dask_chunks: Dict[Any, Any] = {"x": "auto", "y": "auto"},
) -> xr.Dataset:
    # load rainfall climatology
    if era == "_S1":
        chirps = rio_slurp_xarray(urls["chirps"][0])
    if era == "_S2":
        chirps = rio_slurp_xarray(urls["chirps"][1])

    if training:
        chirps = xr_reproject(chirps, ds.geobox, "bilinear")
        ds["rain"] = chirps

    else:
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
        chirps = assign_crs(
            chirps.sel(longitude=y_slice, latitude=x_slice, method="nearest")
        )

        # fill any NaNs in CHIRPS with local (s2-tile bbox) mean
        chirps = chirps.fillna(chirps.mean())
        chirps = xr_reproject(chirps, ds.geobox, "bilinear")
        chirps = chirps.chunk(dask_chunks)
        ds["rain"] = chirps

    # rename bands to include era
    for band in ds.data_vars:
        ds = ds.rename({band: band + era})

    return ds


def gm_mads_two_seasons_training(ds):
    # load the data
    dss = {"S1": ds.isel(time=0), "S2": ds.isel(time=1)}

    # create features
    epoch1 = common_ops(dss["S1"], era="_S1")
    epoch1 = add_chirps(epoch1, era="_S1")
    epoch2 = common_ops(dss["S2"], era="_S2")
    epoch2 = add_chirps(epoch2, era="_S2")

    # add slope
    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    slope = rio_slurp_xarray(url_slope, gbox=ds.geobox)
    slope = slope.to_dataset(name="slope")

    result = xr.merge([epoch1, epoch2, slope], compat="override")

    return result.astype(np.float32).squeeze()


def drop_nan_nodata(xx):
    """
    We pass this function to the
    native_transform parameter of
    load_with_native_transform in order
    to strip off the no-data values
    """
    for dv in xx.data_vars.values():
        if dv.attrs.get("nodata", "") == "NaN":
            dv.attrs.pop("nodata")
    return xx


def gm_mads_two_seasons_prediction(
    task: Task,
    measurements: List[str],
    urls: Dict[Any, Any],
    dask_chunks: Dict[str, Any] = {},
) -> xr.Dataset:
    """
    Feature layer function for production run of
    eastern crop-mask. Similar to the training function
    but data is loaded internally, CHIRPS is reprojected differently,
    and dask chunks are used.
    """

    # load semi-annual geomedians
    ds = load_with_native_transform(
        task.datasets,
        geobox=task.geobox,
        native_transform=lambda x: drop_nan_nodata(x),
        bands=measurements,
        chunks=dask_chunks,
        resampling="bilinear",
    )

    dss = {
        "S1": ds.isel(spec=0).drop(["spatial_ref", "spec"]),
        "S2": ds.isel(spec=1).drop(["spatial_ref", "spec"]),
    }

    # create features
    epoch1 = common_ops(dss["S1"], era="_S1")
    epoch1 = add_chirps(
        urls, epoch1, era="_S1", training=False, dask_chunks=dask_chunks
    )

    epoch2 = common_ops(dss["S2"], era="_S2")
    epoch2 = add_chirps(
        urls, epoch2, era="_S2", training=False, dask_chunks=dask_chunks
    )

    # add slope
    url_slope = urls["slope"]
    slope = rio_slurp_xarray(url_slope, gbox=ds.geobox)
    slope = slope.to_dataset(name="slope").chunk(dask_chunks)

    result = xr.merge([epoch1, epoch2, slope], compat="override")

    result = result.astype(np.float32)
    return result.squeeze()
