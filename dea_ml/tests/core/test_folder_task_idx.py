import pytest

from dea_ml.core.feature_layer import get_xy_from_task


# TODO: refer to https://github.com/spulec/moto/blob/master/tests/test_sqs/test_sqs.py
@pytest.mark.parametrize(
    "taskstr,should_be",
    [
        ("x+0001/y+0000/2019-01--P6M", (1, 0)),
        ("x+0001/y+0001/2019-01--P6M", (1, 1)),
        ("x+0001/y-0001/2019-01--P6M", (1, -1)),
    ],
)
def test_xy_from_taskstr(taskstr, should_be):
    assert get_xy_from_task(taskstr) == should_be
