import os.path as osp
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, Union

import boto3
import fsspec
import joblib
import requests
from botocore import UNSIGNED
from botocore.config import Config

try:
    from ruamel.yaml import YAML

    _YAML_C = YAML(typ="safe", pure=False)
except ImportError:
    _YAML_C = None


def download_file(url: str, local_filename: Optional[str] = None) -> str:
    """
    Download file on github with ```?raw=true```
    https://github....gm_mads_two_seasons_ml_model_20210301.joblib?raw=true
    TODO: use fsspec to avoid download file to hard driver.
    :param url:
    :param local_filename:
    :return:
    """
    local_filename = (
        local_filename if local_filename else osp.join("/tmp", url.split("/")[-1])
    )
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


def read_joblib(path):
    """
    Function to load a joblib file from an s3 bucket or local directory.
    :param: path: an s3 bucket or local directory path where the file is stored
    :return:  joblib model: Joblib file loaded
    """

    # Path is an s3 bucket
    if path.startswith("s3://"):
        s3_bucket, s3_key = path.split("/")[2], path.split("/")[3:]
        s3_key = "/".join(s3_key)
        with BytesIO() as f:
            boto3.client(
                "s3", config=Config(signature_version=UNSIGNED)
            ).download_fileobj(Bucket=s3_bucket, Key=s3_key, Fileobj=f)
            f.seek(0)
            return joblib.load(f)
    elif path.startswith("https://"):
        with fsspec.open(path) as fh:
            return joblib.load(fh)
    # Path is a local directory
    else:
        with open(path, "rb") as f:
            return joblib.load(f)


PathLike = Union[str, Path]
RawDoc = Union[str, bytes]


def slurp(fname: PathLike, binary: bool = False) -> RawDoc:
    """fname -> str|bytes.

    binary=True -- read bytes not text
    """
    mode = "rb" if binary else "rt"

    with open(fname, mode) as f:
        return f.read()


def _parse_yaml_yaml(s: str) -> Dict[str, Any]:
    import yaml

    return yaml.load(s, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


def _parse_yaml_ruamel(s: str) -> Dict[str, Any]:
    return _YAML_C.load(s)


parse_yaml = _parse_yaml_yaml if _YAML_C is None else _parse_yaml_ruamel


def _guess_is_file(s: str):
    try:
        return Path(s).exists()
    except IOError:
        return False


def parse_yaml_file_or_inline(s: str) -> Dict[str, Any]:
    """
    Accept on input either a path to yaml file or yaml text, return parsed yaml document.
    """
    if "git" in s:
        with fsspec.open(s) as fh:
            txt = fh.read()
            result = parse_yaml(txt)
            return result

    if _guess_is_file(s):
        txt = slurp(s, binary=False)
        assert isinstance(txt, str)
    else:
        txt = s

    result = parse_yaml(txt)
    if isinstance(result, str):
        raise IOError(f"No such file: {s}")

    return result
