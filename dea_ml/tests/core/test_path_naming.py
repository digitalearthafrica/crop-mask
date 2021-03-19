import pytest

from dea_ml.core.product_feature_config import FeaturePathConfig


@pytest.fixture
def dummy_tile():
    return "x+029/y+000"


def test_prepare_path(dummy_tile):
    config = FeaturePathConfig()
    output_fld, _, metadata_path = config.prepare_the_io_path(dummy_tile)
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
