import argparse
import json
import subprocess
from typing import Sequence, Tuple

import fsspec
import pandas as pd
from datacube.utils.geometry import Geometry
from odc.dscache.tools.tiling import GRIDS


def gen_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--task-csv", help="task csv file.")
    parse.add_argument("--geojson", help="the absolute path of the geojson file")
    parse.add_argument("--outfile", help="output task file")
    parse.add_argument(
        "--publish",
        help="publish the indices directly to the gaven sqs and db url",
        default=True,
    )
    parse.add_argument(
        "--sqs",
        default="deafrica-dev-eks-stats-geomedian-semiannual",
        help="default sqs is the semiannual one",
    )
    parse.add_argument(
        "--db",
        default="s3://deafrica-data-dev-af/crop_mask_eastern/0-1-0/gm_s2_semiannual_all.db",
        help="task db file url",
    )
    return parse.parse_args()


def publish_task(task_slices: Sequence[Tuple], db_url: str, sqs: str):
    """
    publish the task_df index onto SQS defined
    odc-stats publish-tasks s3://deafrica-data-dev-af/crop_mask_eastern/0-1-0/gm_s2_semiannual_all.db \
    deafrica-dev-eks-stats-geomedian-semiannual 4005:4010
    """
    assert all([db_url, sqs]), "must have all required arguments, db_url, sqs"
    template = ["odc-stats", "publish-tasks", db_url, sqs]
    for start, end in task_slices:
        cmd = template.copy()
        cmd.append(f"{start}:{end}")
        print("Excuting {}".format(" ".join(cmd)))
        subprocess.call(cmd)
    print("Done!")


def gen_slices(task_df: pd.DataFrame) -> Sequence[Tuple]:
    tasks_slices = []
    # sorted the indices first, then extract related indices
    indices = sorted(task_df["Index"])
    start: int = indices[0]
    for cur, next in zip(indices[:-1], indices[1:]):
        if next - cur > 1:
            tasks_slices.append((start, cur + 1))
            start = next
    if next - cur > 1:
        tasks_slices.append((next, next + 1))
    else:
        tasks_slices.append((start, next + 1))
    return tasks_slices


def main():
    args = gen_args()
    if not args.geojson:
        raise ValueError("No geojson file specified")
    if not args.outfile:
        raise ValueError("No output file specified")
    with fsspec.open(args.geojson) as fhin:
        data = json.load(fhin)

    geom = Geometry(data["features"][0]["geometry"], crs=data["crs"]["properties"]["name"])
    africa10 = GRIDS["africa_10"]

    task_df = pd.read_csv(args.task_csv)
    aez_tasks = []
    for row in task_df.itertuples():
        tmp_geom = africa10.tile_geobox((row.X, row.Y)).extent
        if geom.contains(tmp_geom) or geom.overlaps(tmp_geom):
            aez_tasks.append(row)
    output_df = pd.DataFrame(aez_tasks)
    output_df.to_csv(args.outfile, index=False)
    print("Generated tasks from geojson.")
    if args.publish:
        tasks_slices = gen_slices(output_df)
        publish_task(tasks_slices, args.db, args.sqs)


if __name__ == "__main__":
    main()
