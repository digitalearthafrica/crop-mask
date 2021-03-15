import glob
import json
import logging
import math
import os
import os.path as osp
import time

import psutil

# from datacube.utils.dask import start_local_dask
# from distributed import Client
from datacube.utils.dask import start_local_dask
from odc.io.cgroups import get_cpu_quota, get_mem_quota
from odc.stats._cli_common import setup_logging

from dea_ml.core.merge_tifs_to_ds import PredictFromFeature
from dea_ml.core.product_feature_config import FeaturePathConfig

# sys.path.append("/home/jovyan/wa/u23/dea_ai_core/src")


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

nthreads = get_max_cpu()
memory_limit = get_max_mem()

client = start_local_dask(
    threads_per_worker=nthreads, processes=False, memory_limit=memory_limit
)
# client = Client(address="scheduler:8786")

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


for task in tasks:
    tile_indx = "/".join(task.split("/")[:2])

    file_prefix = f"{tile_indx}"
    output_path = osp.join(output_fld, file_prefix, "*")
    if glob.glob(output_path):
        _log.warning(f"tile {output_path} is done already. Skipping...")
        continue
    _log.info(f"proessing tiles for task {output_path}. (2019-01 and 2019-07)")

    t0 = time.time()
    # cmd = [
    #     "python3",
    #     "dea_ai_core/tasks/merge_tifs_to_ds.py",
    #     f"task",
    # ]
    # subprocess.run(cmd, env=my_env)
    worker = PredictFromFeature()
    worker.run(task)
    del worker
    t1 = time.time()
    wall_time = (t1 - t0) / 60
    _log.info(f"time used {wall_time:.4f}")
