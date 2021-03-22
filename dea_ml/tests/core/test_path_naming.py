import pytest

from dea_ml.config.product_feature_config import FeaturePathConfig
from dea_ml.helpers.io import prepare_the_io_path


@pytest.fixture
def dummy_tile():
    return "x+029/y+000"


def test_prepare_path(dummy_tile):
    config = FeaturePathConfig()
    output_fld, _, metadata_path = prepare_the_io_path(config, dummy_tile)
    assert (
        "/".join(
            [
                config.PRODUCT_NAME,
                config.PRODUCT_VERSION,
                dummy_tile,
            ]
        )
        in output_fld
    )

    assert metadata_path.startswith(output_fld)
