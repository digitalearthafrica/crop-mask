import pytest

from dea_ml.core.merge_tifs_to_ds import prepare_the_io_path
from dea_ml.core.product_feature_config import FeaturePathConfig


@pytest.fixture
def dummy_tile():
    return "x+029/y+000"


def test_prepare_path(dummy_tile):
    output_fld, _, metadata_path = prepare_the_io_path(dummy_tile)
    assert (
        "/".join(
            [
                FeaturePathConfig.PRODUCT_NAME,
                FeaturePathConfig.PRODUCT_VERSION,
                dummy_tile,
            ]
        )
        in output_fld
    )

    assert metadata_path.startswith(output_fld)
