import pytest
from dea_ml.helpers.io import parse_yaml_file_or_inline


@pytest.fixture
def github_config_path():
    return "https://raw.githubusercontent.com/digitalearthafrica/crop-mask/main/production/dea_ml/dea_ml/config/ml_config.yaml"  # noqa


def test_parse_remote_config_file(github_config_path):
    result = parse_yaml_file_or_inline(github_config_path)
    assert isinstance(result, dict)
    assert {"rename_dict", "urls"}.issubset(set(result.keys()))
    assert {"slope", "model", "chirps"}.issubset(result["urls"])
