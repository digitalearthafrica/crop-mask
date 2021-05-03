import os
import os.path as osp
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import xarray as xr
from datacube.testutils.io import rio_slurp_xarray
from datacube.utils.geometry import assign_crs, GeoBox
from dea_ml.config.product_feature_config import FeaturePathConfig
from dea_ml.core.africa_geobox import AfricaGeobox
from odc.algo import xr_reproject
from pyproj import Proj, transform


def create_features(
    x: int,
    y: int,
    config: FeaturePathConfig,
    geobox_dict: AfricaGeobox,
    feature_func=None,
    dask_chunks={},
):
    """
    Given a dataset (xarray.DataArray or xr.Dataset) and feature layer
    function, return the feature layers as a xarray Dataset, ready for imput
    into the downstream prediction functions.

    Parameters:
    -----------

    :param x: tile index x
    :param y: time inde y
    :param config: FeaturePathConfig containing the model path and product info`et al.
    :param geobox_dict: geobox will calculate the tile geometry from the tile index

    Returns:
    --------
        subfolder path and the xarray dataset of the features

    """
    # this folder naming x, y will change
    subfld = "x{x:03d}/y{y:03d}".format(x=x, y=y)
    geobox = geobox_dict[(x, y)]

    # call the function on the two 6-month gm+tmads
    model_input = feature_func(geobox).chunk(dask_chunks)

    return subfld, geobox, model_input


def get_xy_from_task(taskstr: str) -> Tuple[int, int]:
    """
    extract the x y from task string
    :param taskstr:
    :return:
    """
    x_str, y_str = taskstr.split("/")[:2]
    return int(x_str.replace("x", "")), int(y_str.replace("y", ""))


def extract_dt_from_model_path(path: str) -> str:
    """
    extract date string from the file name
    :param path:
    :return:
    """
    return re.search(r"_(\d{8})", path).groups()[0]


def extract_xy_from_title(title: str) -> Tuple[int, int]:
    """
    split the x, y out from title
    :param title:
    :return:
    """
    x_str, y_str = title.split(",")
    return int(x_str), int(y_str)


def get_tifs_paths(dirname: str, subfld: str) -> Dict[str, List[str]]:
    """
    generated src tifs dictionnary, season on and two, or more seasons
    :param dirname: dir path name
    :param subfld: subfolder in string type
    """
    all_tifs = os.walk(osp.join(dirname, subfld))
    # l0_dir, l0_subfld, _ = all_tifs[0]
    return dict(
        (l1_dir, l1_files)
        for level, (l1_dir, _, l1_files) in enumerate(all_tifs)
        if level > 0 and (".ipynb" not in l1_dir)
    )


def merge_tile_ds(
    x: int,
    y: int,
    config: FeaturePathConfig,
    geobox_dict: Dict[Tuple, GeoBox],
    gm_ds: Optional[xr.Dataset] = None,
) -> Tuple[str, GeoBox, xr.Dataset]:
    """
    overall all tile tifs and additional features merged  here,
    the xarray dataset, 3 extra indi:withces, integration of rainfall, slope with gm_ds
    :param gm_ds:
    :param x: tile index x
    :param y: time inde y
    :param config: FeaturePathConfig containing the model path and product info`et al.
    :param geobox_dict: geobox will calculate the tile geometry from the tile index
    :return: subfolder path and the xarray dataset of the features
    """
    # this folder naming x, y will change
    subfld = "x{x:+04d}/y{y:+04d}".format(x=x, y=y)
    P6M_tifs: Dict = get_tifs_paths(config.TIF_path, subfld)
    geobox = geobox_dict[(x, y)]
    seasoned_ds = {}
    for k, tifs in P6M_tifs.items():
        era = "_S1" if "2019-01--P6M" in k else "_S2"
        if not gm_ds:
            # no prepare base ds
            base_ds = merge_tifs_into_ds(k, tifs, rename_dict=config.rename_dict)
        else:
            base_ds = gm_ds
        # TODO: to validate the 6month geomedia is down scaled already.
        base_ds = down_scale_gm_band(base_ds)

        seasoned_ds[era] = complete_gm_mads(base_ds, geobox, era)

    slope = (
        rio_slurp_xarray(config.url_slope, gbox=geobox)
        .drop("spatial_ref")
        .to_dataset(name="slope")
    )

    return (
        subfld,
        geobox,
        xr.merge(
            [seasoned_ds["_S1"], seasoned_ds["_S2"], slope], compat="override"
        ).chunk({"x": -1, "y": -1}),
    )


def calculate_indices(ds: xr.Dataset) -> xr.Dataset:
    """
    add calculate_indices into the datasets
    :param ds: input ds with nir, red, green bands
    :return: ds with new bands
    """
    inices_dict = {
        "NDVI": lambda ds: (ds.nir - ds.red) / (ds.nir + ds.red),
        "LAI": lambda ds: (
            3.618
            * ((2.5 * (ds.nir - ds.red)) / (ds.nir + 6 * ds.red - 7.5 * ds.blue + 1))
            - 0.118
        ),
        "MNDWI": lambda ds: (ds.green - ds.swir_1) / (ds.green + ds.swir_1),
    }

    for k, func in inices_dict.items():
        ds[k] = func(ds)

    ds["sdev"] = -np.log(ds["sdev"])
    ds["bcdev"] = -np.log(ds["bcdev"])
    ds["edev"] = -np.log(ds["edev"])

    return ds


def merge_tifs_into_ds(
    root_fld: str,
    tifs: List[str],
    rename_dict: Optional[Dict] = None,
    tifs_min_num=8,
) -> xr.Dataset:
    """
    use os.walk to get the all files under a folder, it just merge the half year tifs.
    We need combine two half-year tifs ds and add (calculated indices, rainfall, and slope)
    :param tifs: tifs with the bands
    :param root_fld: the parent folder for the sub_fld
    :param tifs_min_num: geo-median tifs is 16 a tile idx
    :param rename_dict: we can put the rename dictionary here
    :return:
    """
    # TODO: create dummy datasets to test merge tis
    assert len(tifs) > tifs_min_num
    cache = []
    for tif in tifs:
        if tif.endswith(".tif"):
            band_name = re.search(r"_([A-Za-z0-9]+).tif", tif).groups()[0]
            if band_name in ["rgba", "COUNT"]:
                continue

            band_array = assign_crs(
                xr.open_rasterio(osp.join(root_fld, tif))
                .squeeze()
                .to_dataset(name=band_name),
                crs="epsg:6933",
            )
            cache.append(band_array)
    # clean up output
    output = xr.merge(cache).squeeze()
    output.attrs["crs"] = "epsg:{}".format(output["spatial_ref"].values)
    output.attrs["tile-task-str"] = "/".join(root_fld.split("/")[-3:])
    output = output.drop(["spatial_ref", "band"])
    return output.rename(rename_dict) if rename_dict else output


def chirp_clip(ds: xr.Dataset, chirps: xr.DataArray) -> xr.DataArray:
    """
     fill na with mean on chirps data
    :param ds: geomedian collected with certain geobox
    :param chirps: rainfall data
    :return: chirps data without na
    """
    # TODO: test with dummy ds and chirps
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
    return xr_reproject(chirps, ds.geobox, "bilinear").drop(["spatial_ref"]).squeeze()


def complete_gm_mads(era_base_ds: xr.Dataset, geobox: GeoBox, era: str) -> xr.Dataset:
    """
    merge the geomedian and rainfall chirps data together
    :param era_base_ds:
    :param geobox:
    :param era:
    :return:
    """
    # TODO: this is half year data, require integration tests
    # TODO: load this data once use dask publish (?)
    gm_mads = assign_crs(calculate_indices(era_base_ds))

    rainfall = assign_crs(
        xr.open_rasterio(FeaturePathConfig.rainfall_path[era]), crs="epsg:4326"
    )

    rainfall = chirp_clip(gm_mads, rainfall)

    rainfall = (
        xr_reproject(rainfall, geobox, "bilinear")
        .drop(["band", "spatial_ref"])
        .squeeze()
    )
    gm_mads["rain"] = rainfall

    return gm_mads.rename(
        dict((var_name, str(var_name) + era.upper()) for var_name in gm_mads.data_vars)
    )


def down_scale_gm_band(
    ds: xr.Dataset, exclude: Tuple[str, str] = ("sdev", "bcdev"), scale=10_000
) -> xr.Dataset:
    """
    down scale band not in exclude list.
    :param ds:
    :param exclude:
    :param scale:
    :return:
    """
    for band in ds.data_vars:
        if band not in exclude:
            ds[band] = ds[band] / scale
    return ds


def gm_rainfall_single_season(
    geomedian_with_mads: xr.Dataset,
    rainfall: xr.DataArray,
) -> xr.Dataset:
    """
    generate gm-semiannual with rainfall, query sample see bellow
    :param dc: Datacube context
    :param query: require fields above
    :param season_time_dict: define the time range for each crop season
    :param rainfall_dict: cache the rainfall data with dict
    :param season_key: one of {'_S1', '_S2'}
    :return: gm with rainfall
    """

    # remove time dim
    geomedian_with_mads = geomedian_with_mads.drop("time")
    # scale
    geomedian_with_mads = down_scale_gm_band(geomedian_with_mads)
    geomedian_with_mads = assign_crs(calculate_indices(geomedian_with_mads))

    # rainfall
    rainfall = chirp_clip(geomedian_with_mads, rainfall)

    geomedian_with_mads["rain"] = rainfall

    return geomedian_with_mads.drop("spatial_ref").squeeze()


def merge_two_season_feature(
    seasoned_ds: Dict[str, xr.Dataset], url_slope: str
) -> xr.Dataset:
    """
    combine the two season datasets and add slope to build the machine learning feature
    :param seasoned_ds:  gm+indices+rainfall
    :param config: FeaturePathConfig has slop url
    :return: merged xr Dataset
    """
    slope = (
        rio_slurp_xarray(url_slope, gbox=seasoned_ds["_S1"].geobox)
        .drop("spatial_ref")
        .to_dataset(name="slope")
    )
    renamed_seasoned_ds = {}
    for k, v in seasoned_ds.items():
        renamed_seasoned_ds[k] = v.rename(
            dict((str(band), str(band) + k) for band in v.data_vars)
        )

    return xr.merge(
        [renamed_seasoned_ds["_S1"], renamed_seasoned_ds["_S2"], slope],
        compat="override",
    ).chunk({"x": -1, "y": -1})
