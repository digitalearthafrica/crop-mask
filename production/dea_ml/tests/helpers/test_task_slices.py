from dea_ml.helpers.geojson_defined_tasks import gen_slices
import pytest
import pandas as pd


@pytest.fixture
def task_df():
    tasks = {"Index": [1, 2, 3, 5, 7, 8, 11]}
    return pd.DataFrame(tasks)


def test_gen_slices(task_df):
    slices = gen_slices(task_df)
    should_be = set([(1, 4), (5, 6), (7, 9), (11, 12)])
    assert set(slices) == should_be


def test_gen_slices_with_consective_last_two(task_df):
    task_df_without_last = task_df.iloc[:-1]
    slices = gen_slices(task_df_without_last)
    should_be = set([(1, 4), (5, 6), (7, 9)])
    assert set(slices) == should_be
