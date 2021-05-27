from datetime import datetime
from typing import (
    Iterable,
    Iterator,
    Optional,
    Any,
)

import xarray as xr
from odc.algo import wait_for_future
from odc.stats.model import Task, TaskResult
from odc.stats.proc import TaskRunner

Future = Any


class CMTaskRunner(TaskRunner):
    """
    use modified TaskRunner for crop-mask prediction
    """

    def _run(self, tasks: Iterable[Task]) -> Iterator[TaskResult]:
        cfg = self._cfg
        client = self.client()
        sink = self.sink
        proc = self.proc
        check_exists = cfg.overwrite is False
        _log = self._log

        for task in tasks:
            _log.info(f"Starting processing of {task.location}")
            tk = task.source
            if tk is not None:
                t0 = tk.start_time
            else:
                t0 = datetime.utcnow()
            if check_exists:
                path = sink.uri(task)
                _log.debug(f"Checking if can skip {path}")
                if sink.exists(task):
                    _log.info(f"Skipped task @ {path}")
                    if tk:
                        _log.info("Notifying completion via SQS")
                        tk.done()

                    yield TaskResult(task, path, skipped=True)
                    continue
            # TODO: verify the updated code here.
            _log.debug("Building Dask Graph")
            input_data = proc.input_data(task)
            if not input_data:
                _log.warn(f"{task.tile_index} has no rainfall, or missing data.")
                if tk:
                    tk.cancel()
                continue
            ds = proc.reduce(input_data)

            _log.debug(f"Submitting to Dask ({task.location})")
            ds = client.persist(ds, fifo_timeout="1ms")

            aux: Optional[xr.Dataset] = None
            rgba = proc.rgba(ds)
            if rgba is not None:
                aux = xr.Dataset(dict(rgba=rgba))

            cog = sink.dump(task, ds, aux)
            cog = client.compute(cog, fifo_timeout="1ms")

            _log.debug("Waiting for completion")
            cancelled = False

            for (dt, t_now) in wait_for_future(cog, cfg.future_poll_interval, t0=t0):
                if cfg.heartbeat_filepath is not None:
                    self._register_heartbeat(cfg.heartbeat_filepath)
                if tk:
                    tk.extend_if_needed(
                        cfg.job_queue_max_lease, cfg.renew_safety_margin
                    )
                if cfg.max_processing_time > 0 and dt > cfg.max_processing_time:
                    _log.error(
                        f"Task {task.location} failed to finish on time: {dt}>{cfg.max_processing_time}"
                    )
                    cancelled = True
                    cog.cancel()
                    break

            if cancelled:
                result = TaskResult(task, error="Cancelled due to timeout")
            else:
                result = self._safe_result(cog, task)

            if result:
                _log.info(f"Finished processing of {result.task.location}")
                if tk:
                    _log.info("Notifying completion via SQS")
                    tk.done()
            else:
                if tk:
                    tk.cancel()

            yield result
