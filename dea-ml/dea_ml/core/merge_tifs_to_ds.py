import datetime
import json
import math
import os
import os.path as osp
import re
import uuid
from typing import List, Optional, Dict, Tuple

import click
import joblib
import numpy as np
import psutil
import xarray as xr
from datacube import Datacube
from datacube.testutils.io import rio_slurp_xarray
from datacube.utils.cog import write_cog
from datacube.utils.dask import start_local_dask
from datacube.utils.geometry import assign_crs, GeoBox, Geometry
from datacube.utils.rio import configure_s3_access
from odc.algo import xr_reproject
from odc.io.cgroups import get_cpu_quota, get_mem_quota
from odc.stats._cli_common import setup_logging
from pyproj import Proj, transform

from dea_ml.core.africa_geobox import AfricaGeobox
from dea_ml.core.cm_prediction import predict_xr
from dea_ml.core.product_feature_config import FeaturePathConfig
from dea_ml.core.stac_to_dc import StacIntoDc


def get_max_mem() -> int:
    """
    Max available memory, takes into account pod resource allocation
    """
    total = psutil.virtual_memory().total
    mem_quota = get_mem_quota()
    if mem_quota is None:
        return total
    return min(mem_quota, total)


def get_max_cpu() -> int:
    """
    Max available CPU (rounded up if fractional), takes into account pod
    resource allocation
    """
    ncpu = get_cpu_quota()
    if ncpu is not None:
        return int(math.ceil(ncpu))
    return psutil.cpu_count()


class PredictFromFeature:
    """
    This only covers 2019 case in sandbox now. Check configureation in FeaturePathConfig before use
    run this.
    # todo: add context  to this classpredicted
    """

    def __init__(self, client=None):
        self.geobox_dict = None
        if not client:
            nthreads = get_max_cpu()
            memory_limit = get_max_mem()
            client = start_local_dask(
                threads_per_worker=nthreads,
                processes=False,
                memory_limit=int(0.9 * memory_limit),
            )
            configure_s3_access(aws_unsigned=True, cloud_defaults=True, client=client)
        self.client = client

    def merge_ds_exec(self, x: int, y: int) -> Tuple[str, xr.Dataset]:
        """
        merge the xarray dataset
        # TODO: move to feature building step
        @param x: tile index x
        #param y: time inde y
        @return: subfolder path and the xarray dataset of the features
        """
        subfld = "x{x:+04d}/y{y:+04d}".format(x=x, y=y)
        P6M_tifs: Dict = get_tifs_paths(FeaturePathConfig.TIF_path, subfld)
        geobox = self.geobox_dict[(x, y)]
        seasoned_ds = {}
        for k, tifs in P6M_tifs.items():
            era = "_S1" if "2019-01--P6M" in k else "_S2"
            base_ds = merge_tifs_into_ds(
                k, tifs, rename_dict=FeaturePathConfig.rename_dict
            )
            # TODO: rewrite from here for dc load half year geomedian
            base_ds = down_scale_gm_band(base_ds)

            seasoned_ds[era] = complete_gm_mads(base_ds, geobox, era)

        slope = (
            rio_slurp_xarray(FeaturePathConfig.url_slope, gbox=geobox)
            .drop("spatial_ref")
            .to_dataset(name="slope")
        )
        return (
            subfld,
            xr.merge(
                [seasoned_ds["_S1"], seasoned_ds["_S2"], slope], compat="override"
            ).chunk({"x": -1, "y": -1}),
        )

    def run(self, taskstr: str):
        """
        run the prediction here, default crs='epsg:4326'
        The sample of a feature:
        @return: None
        """
        setup_logging()
        import logging  # noqa pylint: disable=import-outside-toplevel

        _log = logging.getLogger(__name__)

        self.geobox_dict = AfricaGeobox(
            resolution=FeaturePathConfig.resolution, crs=FeaturePathConfig.output_crs
        ).geobox_dict

        x, y = get_xy_from_task(taskstr)

        # step 1: collect the geomedian, indices, rainfall, slope as feature
        # TODO: add more interface in merge_ds_exec, dc.load and x, y
        subfld, data = self.merge_ds_exec(x, y)
        input_data = data[FeaturePathConfig.training_features]

        # step 2: load trained model
        model = joblib.load(FeaturePathConfig.model_path)

        # step 3: prediction
        predicted = predict_xr(
            model,
            input_data,
            clean=True,
            proba=True,
        )

        _log.info("... Dask computing ...")
        # predicted.compute()
        predicted.persist()

        # post prediction filtering
        predict = predicted.Predictions

        query = FeaturePathConfig.query.copy()

        # Update dc query with geometry
        geobox_used = self.geobox_dict[(x, y)]
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
        predict = predict.astype(np.uint8)

        output_fld, paths, metadata_path = prepare_the_io_path(subfld)

        if not osp.exists(output_fld):
            os.makedirs(output_fld)

        _log.info("collecting mask and write cog.")
        write_cog(
            predict.compute(),
            paths["mask"],
            overwrite=True,
        )

        _log.info("collecting prob and write cog.")
        write_cog(
            predicted.Probabilities.astype(np.uint8).compute(),
            paths["prob"],
            overwrite=True,
        )

        _log.info("collecting the stac json and write out.")

        processing_dt = datetime.datetime.now()

        uuid_hex = uuid.uuid4()
        remoe_path = dict((k, osp.basename(p)) for k, p in paths.items())
        remote_metadata_path = metadata_path.replace(
            FeaturePathConfig.DATA_PATH, FeaturePathConfig.REMOTE_PATH
        )
        stac_doc = StacIntoDc.render_metadata(
            FeaturePathConfig.product,
            geobox_used,
            (x, y),
            FeaturePathConfig.datetime_range,
            uuid_hex,
            remoe_path,
            remote_metadata_path,
            processing_dt,
        )

        with open(metadata_path, "w") as fh:
            json.dump(stac_doc, fh, indent=2)


def calculate_indices(ds: xr.Dataset) -> xr.Dataset:
    """
    add calculate_indices into the datasets
    @param ds: input ds with nir, red, green bands
    @return: ds with new bands
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
    @param tifs: tifs with the bands
    @param root_fld: the parent folder for the sub_fld
    @param tifs_min_num: geo-median tifs is 16 a tile idx
    @param rename_dict: we can put the rename dictionary here
    @return:
    """
    # TODO: create dummy datasets to test mergue tis
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
    # Todo: xmin < 0 and xmax < 0,  x_slice = [], unit tests
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
    return chirps.fillna(chirps.mean())


def complete_gm_mads(era_base_ds: xr.Dataset, geobox: GeoBox, era: str) -> xr.Dataset:
    """
    merge the geomedian and rainfall chirps data together
    :param era_base_ds:
    :param geobox:
    :param era:
    :return:
    """
    # TODO: this is half year data, require integration tests
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
    for band in ds.data_vars:
        if band not in exclude:
            ds[band] = ds[band] / scale
    return ds


def get_xy_from_task(taskstr: str) -> Tuple[int, int]:
    x_str, y_str = taskstr.split("/")[:2]
    return int(x_str.replace("x", "")), int(y_str.replace("y", ""))


def extract_dt_from_model_path(path: str) -> str:
    return re.search(r"_(\d{8})", path).groups()[0]


def prepare_the_io_path(tile_indx: str) -> Tuple[str, Dict[str, str], str]:
    """
    use sandbox local path to mimic the target s3 prefixes. The path follow our nameing rule:
    <product_name>/version/<x>/<y>/<year>/<product_name>_<x>_<y>_<timeperiod>_<band>.<extension>
    the name in config a crop_mask_eastern_product.yaml and the github repo for those proudct config
    @param tile_indx: <x>/<y>
    @return:
    """

    start_year = FeaturePathConfig.datetime_range.start.year
    tile_year_prefix = f"{tile_indx}/{start_year}"
    file_prefix = f"{FeaturePathConfig.product.name}/{tile_year_prefix}"

    output_fld = osp.join(
        FeaturePathConfig.DATA_PATH,
        FeaturePathConfig.product.name,
        FeaturePathConfig.product.version,
        tile_year_prefix,
    )

    mask_path = osp.join(
        output_fld,
        file_prefix.replace("/", "_") + "_mask.tif",
    )

    prob_path = osp.join(
        output_fld,
        file_prefix.replace("/", "_") + "_prob.tif",
    )

    paths = {"mask": mask_path, "prob": prob_path}

    metadata_path = mask_path.replace("_mask.tif", ".json")

    assert set(paths.keys()) == set(FeaturePathConfig.product.measurements)

    return output_fld, paths, metadata_path


def extract_xy_from_title(title: str) -> Tuple[int, int]:
    """
    split the x, y out from title
    @param title:
    @return:
    """
    x_str, y_str = title.split(",")
    return int(x_str), int(y_str)


def get_tifs_paths(dirname: str, subfld: str) -> Dict[str, List[str]]:
    """
    generated src tifs dictionnary, season on and two, or more seasons

    """
    all_tifs = os.walk(osp.join(dirname, subfld))
    # l0_dir, l0_subfld, _ = all_tifs[0]
    return dict(
        (l1_dir, l1_files)
        for level, (l1_dir, _, l1_files) in enumerate(all_tifs)
        if level > 0 and (".ipynb" not in l1_dir)
    )


@click.command("tile-predict")
@click.argument("task-str", type=str, nargs=1)
def main(task_str):
    worker = PredictFromFeature()
    worker.run(task_str)
