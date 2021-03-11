import datetime
import json
import math
import os
import os.path as osp
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import click
import joblib
import numpy as np
import xarray as xr
from datacube import Datacube
from datacube.model import GridSpec
from datacube.testutils.io import rio_slurp_xarray
from datacube.utils.cog import write_cog
from datacube.utils.geometry import assign_crs, GeoBox, CRS, box, Geometry
from odc.algo import xr_reproject
from odc.stats._cli_common import setup_logging
from odc.stats.model import DateTimeRange, OutputProduct
from pyproj import Proj, transform

from dea_ml.core.cm_prediction import predict_xr
from dea_ml.core.stac_to_dc import StacIntoDc


@dataclass
class FeaturePathConfig:
    DATA_PATH = "/g/data/u23/data/"
    REMOTE_PATH = "s3://deafrica-data-dev-af/"
    PRODUCT_NAME = "crop_mask_eastern"
    PRODUCT_VERSION = "v0.1.4"

    TIF_path = osp.join(DATA_PATH, "tifs20")
    model_path = "/g/data/u23/crop-mask/eastern_cropmask/results/gm_mads_two_seasons_ml_model_20210301.joblib"
    model_type = "gm_mads_two_seasons"
    rename_dict = {  # "nir_1": "nir",
        "B02": "blue",
        "B03": "green",
        "B04": "red",
        "B05": "red_edge_1",
        "B06": "red_edge_2",
        "B07": "red_edge_3",
        "B08": "nir",
        "B8A": "nir_narrow",
        "B11": "swir_1",
        "B12": "swir_2",
        "BCMAD": "bcdev",
        "EMAD": "edev",
        "SMAD": "sdev",
    }

    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    rainfall_path = {
        "_S1": "/g/data/CHIRPS/cumulative_alltime/CHPclim_jan_jun_cumulative_rainfall.nc",
        "_S2": "/g/data/CHIRPS/cumulative_alltime/CHPclim_jul_dec_cumulative_rainfall.nc",
    }
    s1_key, s2_key = "2019-01--P6M", "2019-07--P6M"
    resolution = (-20, 20)
    time = ("2019-01", "2019-12")
    datetime_range = DateTimeRange(time[0], "P12M")
    output_crs = "epsg:6933"
    query = {
        "time": time,
        "resolution": resolution,
        "output_crs": output_crs,
        "group_by": "solar_day",
    }
    training_features = [
        "red_S1",
        "blue_S1",
        "green_S1",
        "nir_S1",
        "swir_1_S1",
        "swir_2_S1",
        "red_edge_1_S1",
        "red_edge_2_S1",
        "red_edge_3_S1",
        "edev_S1",
        "sdev_S1",
        "bcdev_S1",
        "NDVI_S1",
        "LAI_S1",
        "MNDWI_S1",
        "rain_S1",
        "red_S2",
        "blue_S2",
        "green_S2",
        "nir_S2",
        "swir_1_S2",
        "swir_2_S2",
        "red_edge_1_S2",
        "red_edge_2_S2",
        "red_edge_3_S2",
        "edev_S2",
        "sdev_S2",
        "bcdev_S2",
        "NDVI_S2",
        "LAI_S2",
        "MNDWI_S2",
        "rain_S2",
        "slope",
    ]

    prd_properties = {
        "odc:file_format": "GeoTIFF",
        "odc:producer": "digitalearthafrica.org",
        "odc:product": f"{PRODUCT_NAME}",
        "proj:epsg": 6933,
        "crop-mask-model": osp.basename(model_path),
    }
    product = OutputProduct(
        name=PRODUCT_NAME,
        version=PRODUCT_VERSION,
        short_name=PRODUCT_NAME,
        location=REMOTE_PATH,  # place holder
        properties=prd_properties,
        measurements=("mask", "prob"),
        href=f"https://explorer.digitalearth.africa/products/{PRODUCT_NAME}",
    )


class AfricaGeobox:
    """
    generate the geobox for each tile according to the longitude ande latitude bounds.
    """

    def __init__(self, resolution: Tuple[int, int] = (-20, 20), crs: str = "epsg:6933"):
        target_crs = CRS(crs)
        self.albers_africa_N = GridSpec(
            crs=target_crs,
            tile_size=(96_000.0, 96_000.0),  # default
            resolution=resolution,
        )
        africa = box(-18, -38, 60, 30, "epsg:4326")
        self.africa_projected = africa.to_crs(crs, resolution=math.inf)

    def tile_geobox(self, tile_index: Tuple[int, int]) -> GeoBox:
        return self.albers_africa_N.tile_geobox(tile_index)

    @property
    def geobox_dict(self) -> Dict:
        return dict(self.albers_africa_N.tiles(self.africa_projected.boundingbox))


class TifsToFeature:
    """
    This only covers 2019 case in sandbox now. Check configureation in FeaturePathConfig before use
    run this.
    """

    def __init__(self):
        self.geobox_dict = None

    @staticmethod
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
                * (
                    (2.5 * (ds.nir - ds.red))
                    / (ds.nir + 6 * ds.red - 7.5 * ds.blue + 1)
                )
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

    @staticmethod
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

    @staticmethod
    def source_path_inter(
        source_path: Optional[str] = None, file_num_threshold: int = 8
    ):
        p = Path(source_path)
        return [item for item in os.walk(p) if len(item[2]) > file_num_threshold]

    @staticmethod
    def chirp_clip(ds: xr.Dataset, chirps: xr.DataArray) -> xr.DataArray:

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

    @staticmethod
    def complete_gm_mads(
        era_base_ds: xr.Dataset, geobox: GeoBox, era: str
    ) -> xr.Dataset:
        gm_mads = assign_crs(TifsToFeature.calculate_indices(era_base_ds))

        rainfall = assign_crs(
            xr.open_rasterio(FeaturePathConfig.rainfall_path[era]), crs="epsg:4326"
        )

        rainfall = TifsToFeature.chirp_clip(gm_mads, rainfall)

        rainfall = (
            xr_reproject(rainfall, geobox, "bilinear")
            .drop(["band", "spatial_ref"])
            .squeeze()
        )
        gm_mads["rain"] = rainfall

        return gm_mads.rename(
            dict(
                (var_name, str(var_name) + era.upper())
                for var_name in gm_mads.data_vars
            )
        )

    @staticmethod
    def get_tifs_paths(dirname: str, subfld: str) -> Dict[str, List[str]]:
        """
        generated src tifs dictionnary, season on and two, or more seasons
        @param dirname:
        @param subfld:
        @return:
        """
        all_tifs = os.walk(osp.join(dirname, subfld))
        # l0_dir, l0_subfld, _ = all_tifs[0]
        return dict(
            (l1_dir, l1_files)
            for level, (l1_dir, _, l1_files) in enumerate(all_tifs)
            if level > 0 and (".ipynb" not in l1_dir)
        )

    @staticmethod
    def extract_xy_from_title(title: str) -> Tuple[int, int]:
        """
        split the x, y out from title
        @param title:
        @return:
        """
        x_str, y_str = title.split(",")
        return int(x_str), int(y_str)

    @staticmethod
    def down_scale_gm_band(
        ds: xr.Dataset, exclude: Tuple[str, str] = ("sdev", "bcdev")
    ) -> xr.Dataset:
        for band in ds.data_vars:
            if band not in exclude:
                ds[band] = ds[band] / 10_000
        return ds

    def merge_ds_exec(self, x: int, y: int) -> Tuple[str, xr.Dataset]:
        """
        merge the xarray dataset
        # TODO: move to feature building step
        @param x: tile index x
        #param y: time inde y
        @return: subfolder path and the xarray dataset of the features
        """
        subfld = "x{x:+04d}/y{y:+04d}".format(x=x, y=y)
        P6M_tifs: Dict = self.get_tifs_paths(FeaturePathConfig.TIF_path, subfld)
        geobox = self.geobox_dict[(x, y)]
        seasoned_ds = {}
        for k, tifs in P6M_tifs.items():
            era = "_S1" if "2019-01--P6M" in k else "_S2"
            base_ds = self.merge_tifs_into_ds(
                k, tifs, rename_dict=FeaturePathConfig.rename_dict
            )

            base_ds = self.down_scale_gm_band(base_ds)

            seasoned_ds[era] = self.complete_gm_mads(base_ds, geobox, era)

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

    @staticmethod
    def get_xy_from_task(taskstr: str) -> Tuple[int, int]:
        x_str, y_str = taskstr.split("/")[:2]
        return int(x_str.replace("x", "")), int(y_str.replace("y", ""))

    @staticmethod
    def extract_dt_from_model_path(path: str) -> str:
        return re.search(r"_(\d{8})", path).groups()[0]

    @staticmethod
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

    def run(self, taskstr: str):
        """
        run the prediction here, default crs='epsg:4326'
        The sample of a feature:
        {'type': 'Feature',
         'properties': {'title': '+0029,-0009',
          'utc_offset': 2.0,
          'total': 662,
          'days': 144,
          'ID': None,
          'CODE': None,
          'COUNTRY': 'Eastern'},
         'geometry': {'type': 'Polygon',
          'coordinates': [[[29.578102733293587, -6.030902789057211],
            [29.60006326882379, -6.030902789057211],
            [29.848803296292893, -6.030902789057211],
            ...
            [29.578102733293587, -6.030902789057211]]]}}
        @return: None
        """
        setup_logging()
        import logging  # noqa pylint: disable=import-outside-toplevel

        _log = logging.getLogger(__name__)

        self.geobox_dict = AfricaGeobox(
            resolution=FeaturePathConfig.resolution, crs=FeaturePathConfig.output_crs
        ).geobox_dict

        x, y = self.get_xy_from_task(taskstr)

        # step 1: collect the geomedian, indices, rainfall, slope as feature
        # TODO: for the new features, we prepare it as similar band tifs and merge it later?
        # TODO: feature catalog refer to intake
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

        output_fld, paths, metadata_path = self.prepare_the_io_path(subfld)

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


@click.command("tile-predict")
@click.argument("task-str", type=str, nargs=1)
def main(task_str):
    worker = TifsToFeature()
    worker.run(task_str)
