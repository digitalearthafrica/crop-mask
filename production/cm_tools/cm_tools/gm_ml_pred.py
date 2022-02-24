import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import dask.array as da
import fsspec
import joblib
import xarray as xr
from dask_ml.wrappers import ParallelPostFit
from datacube.model import Dataset
from datacube.utils.geometry import GeoBox, assign_crs
from odc.stats.plugins import StatsPluginInterface
from odc.stats.plugins._registry import register

from cm_tools.feature_layer import gm_mads_two_seasons_prediction
from cm_tools.post_processing import post_processing

_log = logging.getLogger(__name__)


# Copied from
# https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/blob/main/Tools/deafrica_tools/classification.py
def predict_xr(
    model,
    input_xr,
    chunk_size=None,
    persist=False,
    proba=False,
    clean=True,
    return_input=False,
):
    """
    Using dask-ml ParallelPostfit(), runs  the parallel
    predict and predict_proba methods of sklearn
    estimators. Useful for running predictions
    on a larger-than-RAM datasets.
    Last modified: September 2020
    Parameters
    ----------
    model : scikit-learn model or compatible object
        Must have a .predict() method that takes numpy arrays.
    input_xr : xarray.DataArray or xarray.Dataset.
        Must have dimensions 'x' and 'y'
    chunk_size : int
        The dask chunk size to use on the flattened array. If this
        is left as None, then the chunks size is inferred from the
        .chunks method on the `input_xr`
    persist : bool
        If True, and proba=True, then 'input_xr' data will be
        loaded into distributed memory. This will ensure data
        is not loaded twice for the prediction of probabilities,
        but this will only work if the data is not larger than
        distributed RAM.
    proba : bool
        If True, predict probabilities
    clean : bool
        If True, remove Infs and NaNs from input and output arrays
    return_input : bool
        If True, then the data variables in the 'input_xr' dataset will
        be appended to the output xarray dataset.
    Returns
    ----------
    output_xr : xarray.Dataset
        An xarray.Dataset containing the prediction output from model.
        if proba=True then dataset will also contain probabilites, and
        if return_input=True then dataset will have the input feature layers.
        Has the same spatiotemporal structure as input_xr.
    """
    # if input_xr isn't dask, coerce it
    dask = True
    if not bool(input_xr.chunks):
        dask = False
        input_xr = input_xr.chunk({"x": len(input_xr.x), "y": len(input_xr.y)})

    # set chunk size if not supplied
    if chunk_size is None:
        chunk_size = int(input_xr.chunks["x"][0]) * int(input_xr.chunks["y"][0])

    def _predict_func(model, input_xr, persist, proba, clean, return_input):
        x, y, crs = input_xr.x, input_xr.y, input_xr.geobox.crs

        input_data = []

        for var_name in input_xr.data_vars:
            input_data.append(input_xr[var_name])

        input_data_flattened = []

        for arr in input_data:
            data = arr.data.flatten().rechunk(chunk_size)
            input_data_flattened.append(data)

        # reshape for prediction
        input_data_flattened = da.array(input_data_flattened).transpose()

        if clean == True:
            input_data_flattened = da.where(
                da.isfinite(input_data_flattened), input_data_flattened, 0
            )

        if (proba == True) & (persist == True):
            # persisting data so we don't require loading all the data twice
            input_data_flattened = input_data_flattened.persist()

        # apply the classification
        print("predicting...")
        out_class = model.predict(input_data_flattened)

        # Mask out NaN or Inf values in results
        if clean == True:
            out_class = da.where(da.isfinite(out_class), out_class, 0)

        # Reshape when writing out
        out_class = out_class.reshape(len(y), len(x))

        # stack back into xarray
        output_xr = xr.DataArray(out_class, coords={"x": x, "y": y}, dims=["y", "x"])

        output_xr = output_xr.to_dataset(name="Predictions")

        if proba == True:
            print("   probabilities...")
            out_proba = model.predict_proba(input_data_flattened)

            # convert to %
            out_proba = da.max(out_proba, axis=1) * 100.0

            if clean == True:
                out_proba = da.where(da.isfinite(out_proba), out_proba, 0)

            out_proba = out_proba.reshape(len(y), len(x))

            out_proba = xr.DataArray(
                out_proba, coords={"x": x, "y": y}, dims=["y", "x"]
            )
            output_xr["Probabilities"] = out_proba

        if return_input == True:
            print("   input features...")
            # unflatten the input_data_flattened array and append
            # to the output_xr containin the predictions
            arr = input_xr.to_array()
            stacked = arr.stack(z=["y", "x"])

            # handle multivariable output
            output_px_shape = ()
            if len(input_data_flattened.shape[1:]):
                output_px_shape = input_data_flattened.shape[1:]

            output_features = input_data_flattened.reshape(
                (len(stacked.z), *output_px_shape)
            )

            # set the stacked coordinate to match the input
            output_features = xr.DataArray(
                output_features,
                coords={"z": stacked["z"]},
                dims=[
                    "z",
                    *["output_dim_" + str(idx) for idx in range(len(output_px_shape))],
                ],
            ).unstack()

            # convert to dataset and rename arrays
            output_features = output_features.to_dataset(dim="output_dim_0")
            data_vars = list(input_xr.data_vars)
            output_features = output_features.rename(
                {i: j for i, j in zip(output_features.data_vars, data_vars)}
            )

            # merge with predictions
            output_xr = xr.merge([output_xr, output_features], compat="override")

        return assign_crs(output_xr, str(crs))

    if dask == True:
        # convert model to dask predict
        model = ParallelPostFit(model)
        with joblib.parallel_backend("dask", wait_for_workers_timeout=20):
            output_xr = _predict_func(
                model, input_xr, persist, proba, clean, return_input
            )

    else:
        output_xr = _predict_func(
            model, input_xr, persist, proba, clean, return_input
        ).compute()

    return output_xr


class PredGMS2(StatsPluginInterface):
    """
    Prediction from GeoMAD
    """

    source_product = "gm_s2_semiannual"
    target_product = "crop_mask_<region>"

    def __init__(
        self,
        urls: Dict[str, Any],
        bands: Optional[Tuple] = None,
    ):
        # target band to be saved
        self.urls = urls
        self.bands = ("mask", "prob", "filtered")

    @property
    def measurements(self) -> Tuple[str, ...]:
        return self.bands

    def input_data(self, datasets: Sequence[Dataset], geobox: GeoBox) -> xr.Dataset:
        """
        Assemble the input data and do prediction here.

        """
        # create the features
        measurements = [
            "blue",
            "green",
            "red",
            "nir",
            "swir_1",
            "swir_2",
            "red_edge_1",
            "red_edge_2",
            "red_edge_3",
            "bcdev",
            "edev",
            "sdev",
        ]

        input_data = gm_mads_two_seasons_prediction(
            datasets, geobox, measurements, self.urls
        )

        if not input_data:
            return None
        # read in model
        model = joblib.load(self.urls["model"]).set_params(n_jobs=1)

        # ------Run predictions--------
        # step 1: select features
        # load the column names from the training data file to ensure
        # the bands are in the right order
        with fsspec.open(self.urls["td"], "r") as file:
            header = file.readline()
        column_names = header.split()[1:][1:]

        # reorder input data according to column names
        input_data = input_data[column_names]

        # step 2: prediction
        predicted = predict_xr(
            model=model,
            input_xr=input_data,
            clean=True,
            proba=True,
            return_input=True,
        )

        predicted["Predictions"] = predicted["Predictions"].astype("uint8")
        predicted["Probabilities"] = predicted["Probabilities"].astype("uint8")

        # rechunk on the way out
        return predicted.chunk({"x": -1, "y": -1})

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        return post_processing(xx, self.urls)


register("pred-gm-s2", PredGMS2)
