import json
import sys
from logging import getLogger
from typing import List, Set, Dict

import click

logger = getLogger(__name__)


# def geofeature_to_tileidx(feature: Dict) -> Tuple[int, int]:
#     title = feature["properties"]["title"]
#     x_str, y_str = title.split(",")
#     return int(x_str), int(y_str)


def extract_taskstr_from_geojson(time_range: str, geojson: Dict) -> List[str]:
    """
    transfer geojosn into task list
    TODO: update the xy_str with new requirements
    :param time_range: name pattern follow the odc-stats, 2019-01--P6M
    :param: geojson: is the parsed dict from json containing the tile index
    """
    taskstr_set: Set[str] = set()
    for feat in geojson["features"]:
        x, y = feat["properties"]["title"].split(",")
        # xy_str = "{:s}/{:+}/{:+}".format(time_range, int(x), int(y))
        xy_str = "x{x:+04d}/y{y:+04d}/{time_range:s}".format(time_range=time_range, x=int(x), y=int(y))
        taskstr_set.add(xy_str)
    return sorted(taskstr_set)


@click.group(help="Transform geojson into the task list")
def main():
    pass


@main.command("json2tasks")
@click.argument("in-path", type=str)
@click.argument("temporal-range", type=str)
@click.argument("out-path", type=str)
def json_to_tasks(in_path: str, temporal_range: str, out_path: str):
    """
    Transform geojson into task string list.
    geojson features must have title in properties, like {'properties': {'title': '+0041,+0013',...
    @param in_path: absolute path input path, or relative path
    @param temporal_range:Time range use for the geomedian, '2019-01--P6M', '2019-07--P6M'
    @param out_path: absolute path of output result file.
    @return: None
    # TODO: add s3 access support, pick up geojson in s3, push result to s3. Otherwise, manually with awscli
    """
    logger.info(f"reading source file {in_path}")
    with open(in_path) as fh_in:
        geojson = json.load(fh_in)
    taskstrs: List[str] = extract_taskstr_from_geojson(temporal_range, geojson)
    with open(out_path, "w") as fh_out:
        json.dump(taskstrs, fh_out, indent=2)
    logger.info(f"output task string list into {out_path}")


if __name__ == "__main__":
    sys.exit(main())
