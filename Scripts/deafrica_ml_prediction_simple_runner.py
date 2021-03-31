import glob
import json
import logging
import math
import os.path as osp
import time

import psutil

# from dask.distributed import Client, LocalCluster
from distributed import LocalCluster, Client
from odc.io.cgroups import get_cpu_quota, get_mem_quota
from odc.stats._cli_common import setup_logging

from dea_ml.core.predict_from_feature import PredictContext
from dea_ml.core.feature_layer import extract_xy_from_title
from dea_ml.config.product_feature_config import FeaturePathConfig


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


def main():
    setup_logging()

    _log = logging.getLogger(__name__)

    # nthreads = get_max_cpu()
    # memory_limit = get_max_mem()

    # prepare the tile index into the json
    with open("../eastern_cropmask/data/s2_tiles_eastern_aez.geojson") as fhin:
        raw = json.load(fhin)
        tile_indicies = [
            extract_xy_from_title(feature["properties"]["title"])
            for feature in raw["features"]
        ]
        tasks = [f"x{x:+04d}/y{y:+04d}" for x, y in tile_indicies]

    config = FeaturePathConfig()
    output_fld = osp.join(
        config.DATA_PATH,
        config.product.name,
        config.product.version,
    )

    # tasks = ["x+029/y+000/2019-P6M", "x+048/y+010"]
    tasks = tasks[-2:]

    with LocalCluster(processes=False) as cluster:
        with Client(cluster) as client:
            worker = PredictContext(client=client)
            for task in tasks:
                tile_indx = "/".join(task.split("/")[:2])

                file_prefix = f"{tile_indx}"
                output_path = osp.join(output_fld, file_prefix, "*")
                if glob.glob(output_path):
                    _log.warning(f"tile {output_path} is done already. Skipping...")
                    continue
                _log.info(f"proessing tiles for task {output_path}.")

                t0 = time.time()
                worker.run(task)
                t1 = time.time()
                wall_time = (t1 - t0) / 60
                _log.info(f"time used {wall_time:.4f}")


if __name__ == "__main__":
    main()
