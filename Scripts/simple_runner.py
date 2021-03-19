import glob
import json
import logging
import math
import os
import os.path as osp
import time

import psutil

from odc.io.cgroups import get_cpu_quota, get_mem_quota
from odc.stats._cli_common import setup_logging

from dea_ml.core.merge_tifs_to_ds import PredictFromFeature
from dea_ml.core.product_feature_config import FeaturePathConfig
from distributed import wait, secede
from distributed import Client, LocalCluster


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


setup_logging()

_log = logging.getLogger(__name__)


def main():
    nthreads = get_max_cpu()
    memory_limit = get_max_mem()

    with open("/home/jovyan/wa/u23/notebooks/s2_tiles_eastern_aez_tasks.json") as fhin:
        tasks = json.load(fhin)

    output_fld = osp.join(
        FeaturePathConfig.DATA_PATH,
        FeaturePathConfig.product.name,
        FeaturePathConfig.product.version,
    )

    CWD = osp.dirname(__file__)
    my_env = os.environ.copy()
    my_env["PYTHONPATH"] = CWD

    # manually add tasks
    tasks = ["x+029/y+000/2019-P6M", "x+048/y+010"]

    # n_workers=1, threads_per_worker=nthreads, processes=True, memory_limit=memory_limit
    # with LocalCluster(
    #     n_workers=1,
    #     threads_per_worker=int(0.9 * nthreads),
    #     processes=False,
    #     memory_limit=memory_limit,
    # ) as cluster, Client(cluster) as client:
    # client = Client('scheduler:8786')
    # client = Client('tcp://10.95.105.231:8786') # working
    with LocalCluster() as cluster:
        with Client(cluster) as client:
            worker = PredictFromFeature(client=client)
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
            # worker.client.shutdown() # use this with manually cluster

        
if __name__ == '__main__':
    main()

