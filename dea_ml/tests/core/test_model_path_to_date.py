import pytest

from dea_ml.core.merge_tifs_to_ds import extract_dt_from_model_path


@pytest.mark.parametrize(
    "model_path,should_be",
    [
        ("crop-mask/model/gm_mads_two_seasons_ml_model_20201215.joblib", "20201215"),
    ],
)
def test_extract_dt_str_from_path(model_path, should_be):
    assert should_be == extract_dt_from_model_path(model_path)
