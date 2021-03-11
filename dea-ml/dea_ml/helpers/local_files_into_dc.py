import json
import os
import re
from pathlib import Path
from typing import Tuple, List

from datacube import Datacube
from odc.index._index import from_metadata_stream
from odc.index.stac import stac_transform

__doc__ = """before insert the data remember to insert products first, like below
datacube product add crop_mask_eastern.yaml
"""
dc = Datacube()
ROOT = os.path.dirname(__file__)

# ad hoc dir for the sample data folder
DATA_DIR = os.path.join(ROOT, "../../seed/v0.1.4")


def get_xy(part: str) -> Tuple[int, int]:
    x, y = list(map(int, re.findall(r"(\d{3})", part)))
    return x, y


def fetch_stac_json_files(data_folder: str) -> List[Tuple[str, List]]:
    return [
        (fld, data_files)
        for fld, subfld, data_files in os.walk(data_folder)
        if not subfld and (".ipynb" not in fld)
    ]


def collect_datasets(data_folder: str):
    files = fetch_stac_json_files(data_folder)

    for fld, data_files in files:
        for fn in data_files:
            if not fn.endswith(".json"):
                continue
            print(f"processing {fn}")
            full_fn = Path(fld).joinpath(fn)
            with open(full_fn) as fhin:
                rendered = json.load(fhin)

            stac_doc = stac_transform(rendered)
            metapath = list(
                filter(lambda item: item["rel"] in "self", rendered["links"])
            )[0]["href"]
            # TODO: if s3 access available use s3-to-dc directly
            yield list(from_metadata_stream([(metapath, stac_doc)], dc.index))[0][0]


def index_insert():
    data_folder = DATA_DIR
    for dataset in collect_datasets(data_folder):
        dc.index.datasets.add(dataset)
    print("Done!")


if __name__ == "__main__":
    index_insert()
