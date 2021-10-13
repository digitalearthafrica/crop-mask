import logging
from typing import Tuple, Dict, Optional, Any

import xarray as xr
import fsspec
import joblib
from odc.stats import _plugins
from odc.stats.model import Task, StatsPluginInterface

from cm_tools.feature_layer import gm_mads_two_seasons_prediction
from cm_tools.post_processing import post_processing
from deafrica_tools.classification import predict_xr

_log = logging.getLogger(__name__)


class PredGMS2(StatsPluginInterface):
    """
    Prediction from GeoMAD
    """

    source_product = "gm_s2_semiannual"
    target_product = "crop_mask_eastern"

    def __init__(
        self,
        urls: Dict[str, Any],
        rename_dict: Dict[str, str],
        bands: Optional[Tuple] = None,
    ):
        # target band to be saved
        self.urls = urls
        self.rename_dict = rename_dict
        self.bands = bands if bands else ("mask", "prob", "filtered")

    @property
    def measurements(self) -> Tuple[str, ...]:
        return self.bands


    def input_data(self, task: Task) -> xr.Dataset:
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

        input_data = gm_mads_two_seasons_prediction(task, measurements, self.urls)

        if not input_data:
            return None
        # read in model
        model = joblib.load(self.urls["model"]).set_params(n_jobs=1)

        #------Run predictions--------
        # step 1: select features
        # load the column names from the
        # training data file to ensure
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


_plugins.register("pred-gm-s2", PredGMS2)
