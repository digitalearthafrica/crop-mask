# -*- coding: utf-8 -*-
from setuptools import setup

# remember to add the folder paths here
packages = [
    "dea_ml",
    "dea_ml.core",
    "dea_ml.helpers",
    "dea_ml.pipeline",
    "dea_ml.plugins",
]

package_data = {"": ["*"], "dea_ml": ["samples/*"]}

setup_kwargs = {
    "name": "dea-ml",
    "version": "0.2.0",
    "description": "crop mask prediction",
    "long_description": None,
    "author": "deafrica team",
    "author_email": None,
    "maintainer": "Chad Burton",
    "maintainer_email": "chad.burton@ga.gov.au",
    "url": None,
    "packages": packages,
    "package_data": package_data,
    "python_requires": ">=3.6,<4.0",
    "entry_points": {
        "console_scripts": [
            "cm-pred = dea_ml.pipeline.pred_run:main",
            "cm-tsk = dea_ml.helpers.geojson_defined_tasks:main",
        ]
    },
#     "install_requires": [
#         "datacube",
#         "odc_dscache",
#         "odc_algo",
#         "odc_stats",
#         "pystac",
#         "joblib",
#         "dask_ml",
#         "fsspec",
#         "rsgislib",
#         "gdal",
#         "requests",
#         "rasterstats",
#         "aiohttp",
#         "s3fs",
#         "partd",
#     ],
}

setup(**setup_kwargs)
