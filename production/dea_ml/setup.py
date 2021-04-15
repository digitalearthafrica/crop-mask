# -*- coding: utf-8 -*-
from setuptools import setup

# remember to add the folder paths here
packages = \
['dea_ml', 'dea_ml.config', 'dea_ml.core', 'dea_ml.helpers']

package_data = \
{'': ['*'], 'dea_ml': ['samples/*']}

setup_kwargs = {
    'name': 'dea-ml',
    'version': '0.1.8',
    'description': '',
    'long_description': None,
    'author': 'deafrica team',
    'author_email': None,
    'maintainer': 'Jinjun Sun',
    'maintainer_email': 'jijun.sun@ga.gov.au',
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'python_requires': '>=3.6,<4.0',
}


setup(**setup_kwargs)
