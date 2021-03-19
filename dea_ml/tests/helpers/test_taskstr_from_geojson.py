import pytest

from dea_ml.helpers.json_to_taskstr import extract_taskstr_from_geojson


@pytest.fixture
def time_range():
    return "2019-01--P6M"


@pytest.fixture
def geojson_src():
    return {
        "features": [
            {"properties": {"title": "+0029,-0009"}},
            {"properties": {"title": "+0129,-0109"}},
        ]
    }


def test_extract_taskstr_from_geojson(time_range, geojson_src):
    result = extract_taskstr_from_geojson(time_range, geojson_src)
    assert {"x+029/y-009/2019-01--P6M", "x+129/y-109/2019-01--P6M"} == set(result)
