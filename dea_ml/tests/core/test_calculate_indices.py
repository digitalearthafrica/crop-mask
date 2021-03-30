# from itertools import chain

import numpy as np
import pytest
import xarray as xr


# from dea_ml.core.feature_layer import calculate_indices


@pytest.fixture
def dummy_ds():
    tmp = np.ones((2, 2))

    ds = xr.Dataset(
        {
            "nir": (["x", "y"], tmp.copy()),
            "swir_1": (["x", "y"], tmp.copy()),
            "red": (["x", "y"], tmp.copy()),
            "green": (["x", "y"], tmp.copy()),
            "blue": (["x", "y"], tmp.copy()),
            "sdev": (["x", "y"], tmp.copy()),
            "bcdev": (["x", "y"], tmp.copy()),
            "edev": (["x", "y"], tmp.copy()),
        },
    )
    return ds


# @pytest.mark.skip(reason="use manual feature_layer function")
# def test_calculate_indices_fun(dummy_ds):
#     result = calculate_indices(dummy_ds)
#     assert set(i for i in chain.from_iterable(result.LAI.values)) == {-0.118}
#     assert set(i for i in chain.from_iterable(result.NDVI.values)) == {0.0}
#     assert set(i for i in chain.from_iterable(result.MNDWI.values)) == {0.0}
