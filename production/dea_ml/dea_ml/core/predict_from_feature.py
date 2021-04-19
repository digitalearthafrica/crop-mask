import datetime
import json
import logging
import math
import os
import os.path as osp
import uuid
from typing import Optional, Dict, List

import psutil
import xarray as xr
from datacube.utils.cog import write_cog
from datacube.utils.dask import start_local_dask
from datacube.utils.geometry import GeoBox
from datacube.utils.geometry import assign_crs
from datacube.utils.rio import configure_s3_access
from dea_ml.config.product_feature_config import FeaturePathConfig
from dea_ml.core.feature_layer import get_xy_from_task
from dea_ml.core.stac_to_dc import StacIntoDc
from dea_ml.helpers.io import prepare_the_io_path
from deafrica_tools.classification import predict_xr
from distributed import Client
from odc.io.cgroups import get_cpu_quota, get_mem_quota
from odc.stats._cli_common import setup_logging


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


class PredictContext:
    """
    This only covers 2019 case in sandbox now.
    Check configuration in FeaturePathConfig before use run this.
    refer to feature build: https://gist.github.com/cbur24/436a18145c2ac291247360c99ae053be
    """

    def __init__(
        self,
        config: Optional[FeaturePathConfig] = None,
        geobox_dict: Optional[Dict] = None,
        client: Optional[Client] = None,
    ):
        self.config = config if config else FeaturePathConfig()
        self.geobox_dict = geobox_dict
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

        setup_logging()
        self._log = logging.getLogger(__name__)

    def save_data(
        self,
        subfld: str,
        predict: xr.DataArray,
        probabilites: xr.DataArray,
        filtered: xr.DataArray,
        geobox_used: GeoBox,
    ):
        """
        save the prediction results to local folder, prepare stac json
        :param subfld: local subfolder to save the prediction tifs, x{x:03d}/y{y:03d}
        :param predict: predicted binary class label array
        :param probabilites: prediction probabilities array
        :param geobox_used: geobox used for the features for prediction
        :return: None
        """
        output_fld, paths, metadata_path = prepare_the_io_path(self.config, subfld)
        x, y = get_xy_from_task(subfld)
        if not osp.exists(output_fld):
            os.makedirs(output_fld)

        self._log.info("collecting mask and write cog.")
        write_cog(
            assign_crs(predict.compute(), crs=geobox_used.crs),
            paths["mask"],
            overwrite=True,
        )

        self._log.info("collecting prob and write cog.")
        write_cog(
            assign_crs(probabilites.compute(), crs=geobox_used.crs),
            paths["prob"],
            overwrite=True,
        )

        self._log.info("collecting filtered and write cog.")
        write_cog(
            assign_crs(filtered.compute(), crs=geobox_used.crs),
            paths["filtered"],
            overwrite=True,
        )

        self._log.info("collecting the stac json and write out.")

        processing_dt = datetime.datetime.now()

        uuid_hex = uuid.uuid4()
        remoe_path = dict((k, osp.basename(p)) for k, p in paths.items())
        remote_metadata_path = metadata_path.replace(
            self.config.DATA_PATH, self.config.REMOTE_PATH
        )
        stac_doc = StacIntoDc.render_metadata(
            self.config.product,
            geobox_used,
            (x, y),
            self.config.datetime_range,
            uuid_hex,
            remoe_path,
            remote_metadata_path,
            processing_dt,
        )

        with open(metadata_path, "w") as fh:
            json.dump(stac_doc, fh, indent=2)


def predict_with_model(
    training_features: List[str], model, data: xr.Dataset, chunk_size: Dict = None
) -> xr.Dataset:
    """
    run the prediction here
    """
    # step 1: select features
    input_data = data[training_features]

    # step 2: prediction
    predicted = predict_xr(
        model,
        input_data,
        chunk_size=chunk_size,
        clean=True,
        proba=True,
        return_input=True,
    )

    predicted["Predictions"] = predicted["Predictions"].astype("int8")
    predicted["Probabilities"] = predicted["Probabilities"].astype("float32")

    return predicted


# @click.command("tile-predict")
# @click.argument("task-str", type=str, nargs=1)
# def main(task_str):
#     worker = PredictFromFeature()
#     worker.run(task_str)
