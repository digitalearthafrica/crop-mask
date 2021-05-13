import importlib
import logging
import sys

import click
from odc.stats._cli_common import main, setup_logging, click_resolution
from odc.stats.model import TaskRunnerConfig
from odc.stats.proc import TaskRunner


# Todo: upgrade this into fsspec link can access git raw file.
def click_yaml_cfg(*args, **kw):
    """
    @click_yaml_cfg("--custom-flag", help="Whatever help")
    """

    def _parse(ctx, param, value):
        if value is not None:
            from dea_ml.helpers.io import parse_yaml_file_or_inline

            try:
                return parse_yaml_file_or_inline(value)
            except Exception as e:
                raise click.ClickException(str(e)) from None

    return click.option(*args, callback=_parse, **kw)


@main.command("run")
@click.option("--threads", type=int, help="Number of worker threads")
@click.option("--memory-limit", type=str, help="Limit memory used by Dask cluster")
@click.option(
    "--dryrun",
    is_flag=True,
    help="Do not run computation just print what work will be done",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=None,
    help="Do not check if output already exists",
)
@click.option(
    "--heartbeat-filepath",
    type=str,
    help="Path to store pod's heartbeats when running stats as K8 jobs",
)
@click.option(
    "--public/--no-public",
    is_flag=True,
    default=None,
    help="Mark outputs for public access (default: no)",
)
@click.option(
    "--location", type=str, help="Output location prefix as a uri: s3://bucket/path/"
)
@click.option("--max-processing-time", type=int, help="Max seconds per task")
@click.option("--from-sqs", type=str, help="Read tasks from SQS", default="")
@click_yaml_cfg("--config", help="Runner Config")
@click.option(
    "--plugin",
    type=str,
    help="Which stats plugin to run",
)
@click_yaml_cfg(
    "--plugin-config", help="Config for plugin in yaml format, file or text"
)
@click_yaml_cfg("--cog-config", help="Configure COG options")
@click.option("--resampling", type=str, help="Input resampling strategy, e.g. average")
@click_resolution("--resolution", help="Override output resolution")
@click.argument("filedb", type=str, nargs=1)
@click.argument("tasks", type=str, nargs=-1)
def run(
    filedb,
    tasks,
    from_sqs,
    config,
    plugin_config,
    cog_config,
    resampling,
    resolution,
    plugin,
    dryrun,
    threads,
    memory_limit,
    overwrite,
    public,
    location,
    max_processing_time,
    heartbeat_filepath,
):
    """
    Run Stats.

    Task could be one of the 3 things

    \b
    1. Comma-separated triplet: period,x,y or 'x[+-]<int>/y[+-]<int>/period
       2019--P1Y,+003,-004
       2019--P1Y/3/-4          `/` is also accepted
       x+003/y-004/2019--P1Y   is accepted as well
    2. A zero based index
    3. A slice following python convention <start>:<stop>[:<step]
        ::10 -- every tenth task: 0,10,20,..
       1::10 -- every tenth but skip first one 1, 11, 21 ..
        :100 -- first 100 tasks

    If no tasks are supplied and --from-sqs is not used, the whole file will be processed.
    """
    setup_logging()

    # from odc.stats._plugins import import_all
    # from dea_ml.plugin.gm_ml_pred import PredGMS2

    _log = logging.getLogger(__name__)

    if from_sqs:
        if dryrun:
            print("Can not dry run from SQS")
            sys.exit(1)
        if len(tasks):
            print("Supply either <tasks> or --from-sqs")
            sys.exit(2)

    # import_all()
    importlib.import_module("dea_ml.plugins.gm_ml_pred")

    if config is None:
        config = {}

    _cfg = dict(**config)
    s3_acl = "public-read" if public else None

    cfg_from_cli = {
        k: v
        for k, v in dict(
            filedb=filedb,
            plugin=plugin,
            threads=threads,
            memory_limit=memory_limit,
            output_location=location,
            s3_acl=s3_acl,
            overwrite=overwrite,
            max_processing_time=max_processing_time,
            heartbeat_filepath=heartbeat_filepath,
        ).items()
        if v is not None and v != ""
    }

    _log.info(f"Config overrides: {cfg_from_cli}")

    _cfg.update(cfg_from_cli)
    #
    if plugin_config is not None:
        _cfg["plugin_config"] = plugin_config
    else:
        raise Exception("Missing plugin config, --plugin-config=???")

    if resampling is not None and len(resampling) > 0:
        _cfg.setdefault("plugin_config", {})["resampling"] = resampling

    if cog_config is not None:
        _cfg["cog_opts"] = cog_config

    # prepare _cfg and used here
    cfg = TaskRunnerConfig(**_cfg)
    _log.info(f"Using this config: {cfg}")
    runner = TaskRunner(cfg, resolution=resolution)
    if dryrun:
        check_exists = runner.verify_setup()
        for task in runner.dry_run(tasks, check_exists=check_exists):
            print(task.meta)
        sys.exit(0)

    if not runner.verify_setup():
        print("Failed to verify setup, exiting")
        sys.exit(1)

    result_stream = runner.run(sqs=from_sqs) if from_sqs else runner.run(tasks=tasks)

    total = 0
    finished = 0
    skipped = 0
    errored = 0
    for result in result_stream:
        total += 1
        task = result.task
        if result:
            if result.skipped:
                skipped += 1
                _log.info(f"Skipped task #{total:,d}: {task.location} {task.uuid}")
            else:
                finished += 1
                _log.info(f"Finished task #{total:,d}: {task.location} {task.uuid}")
        else:
            errored += 1
            _log.error(f"Failed task #{total:,d}: {task.location} {task.uuid}")

        _log.info(f"T:{total:,d}, OK:{finished:,d}, S:{skipped:,d}, E:{errored:,d}")

    _log.info(
        f"Completed processing {total:,d} tasks, OK:{finished:,d}, S:{skipped:,d}, E:{errored:,d}"
    )

    _log.info("Shutting down Dask cluster")
    del runner
    _log.info("Calling sys.exit(0)")
    sys.exit(0)
